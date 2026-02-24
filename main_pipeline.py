from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, List

from analysis import AnalysisNote, generate_comparison_plots, generate_optimization_plots
from benchmarks import BenchmarkSuite
from evaluators import CFDSimulation, OpenFOAMRANSEvaluator, EvaluationResult
from geometry import MOCSolver, NozzleGeometry, MinimumLengthNozzle
from optimization import Optimizer, OptimizationRunner


# ---------------------------------------------------------------------------
# Import run_multiobjective from scripts/ (no __init__.py there)
# ---------------------------------------------------------------------------
def _import_multiobjective():
    spec = importlib.util.spec_from_file_location(
        "multiobjective_rank",
        Path(__file__).parent / "scripts" / "multiobjective_rank.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.run_multiobjective


try:
    _run_multiobjective = _import_multiobjective()
except Exception:
    _run_multiobjective = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Default configuration (small test numbers for quick validation)
# ---------------------------------------------------------------------------
def default_config() -> Dict[str, Any]:
    return {
        "out_dir": "out/pipeline",
        # ---- MOC baseline ----
        "moc": {
            "mach_exit": 2.5,
            "pressure_ratio": 0.08,
            "gamma": 1.4,
            "geometry": {
                "throat_y": 0.02,
                "exit_y": 0.065,
                "length": 0.24,
                "n_points": 180,
            },
        },
        # ---- Evaluator (Quasi-1D defaults; OpenFOAM keys used when backend=openfoam) ----
        "evaluator": {
            "backend": "quasi1d",
            "campaign": "design_supersonic",
            "gamma": 1.4,
            "gas_constant": 287.0,
            "stagnation_temperature": 1000.0,
            "stagnation_pressure": 6.0e5,
            "ambient_pressure": 8.0e4,
            "n_samples": 90,
            # Quasi-1D loss model knobs
            "friction_scale": 0.2,
            "cf_multiplier": 1.0,
            "curvature_scale": 0.05,
            "bl_displacement_scale": 1.0,
            "discharge_coefficient": 0.985,
            "enable_shock_model": True,
            "shock_trigger_ratio": 0.55,
            "divergence_scale": 1.0,
            "dimension": "2d_planar",
            "depth": 0.02,
            "use_gpu": False,
            # OpenFOAM RANS options (only used when backend=openfoam)
            "fallback_on_failure": False,
            "mesh_nx": 120,
            "mesh_ny": 54,
            "mesh_nz": 12,
            "end_time": 0.004,
            "write_interval": 0.0004,
            "max_co": 0.3,
            "delta_t_init": 1e-7,
            "enable_delta_t_abort": True,
            "delta_t_abort_threshold": 1e-80,
            "delta_t_abort_streak": 20,
            "inlet_velocity": 20.0,
            "k_inlet": 1.0,
            "epsilon_inlet": 50.0,
            "p_initial": 5.4e5,
            "u_initial": 1.0,
            "t_initial": 1000.0,
            "outlet_bc_mode": "wave_transmissive",
            "sampling_nx": 81,
            "convergence_window": 5,
            "convergence_tol": {
                "p_out_rel_std": 0.02,
                "mdot_rel_std": 0.02,
                "ux_out_rel_std": 0.03,
            },
            "min_writes": 8,
            "require_converged_series": True,
            "allow_unsteady_overexpanded": True,
            "overexpanded_unsteady_tol": {
                "p_out_rel_std_max": 0.20,
                "mdot_rel_std_max": 0.80,
                "ux_out_rel_std_max": 1.20,
            },
        },
        # ---- Optimization (GA with small test numbers) ----
        "optimization": {
            "algorithm": "evolutionary",
            "seed": 42,
            "bounds": {
                "exit_radius": [0.055, 0.075],
                "length": [0.20, 0.32],
                "shape": [1.4, 2.2],
            },
            # GA hyper-parameters (small for quick testing)
            "population": 8,
            "generations": 3,
            "sigma": 0.12,
            "n_points": 180,
            "profile": "bezier_like",
            # Multi-objective weights (thrust maximised, loss minimised)
            "w_thrust": 0.7,
            "w_pressure_loss": 0.3,
            # Parallel workers for batch evaluation (1 = sequential)
            "n_workers": 1,
            # RANS validation: how many top Pareto candidates to validate
            "rans_top_k": 3,
            # Penalty terms forwarded to Optimizer._objective_score
            "objective_terms": {
                "design_shock_penalty": 0.10,
                "overexpanded_instability_penalty": 0.05,
                "overexpanded_outlet_osc_penalty": 0.05,
                "overexpanded_wall_rms_penalty": 0.03,
            },
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_geometry_from_params(
    params: Dict[str, float],
    throat_radius: float,
    n_points: int,
    profile: str,
    gid: str,
) -> NozzleGeometry:
    """Build a parametrised nozzle geometry for optimisation candidates."""
    return NozzleGeometry.fromParams(
        {
            "throat_radius": throat_radius,
            "exit_radius": params["exit_radius"],
            "length": params["length"],
            "n_points": n_points,
            "profile": profile,
            "shape": params.get("shape", 1.8),
            "metadata": {"id": gid, "source": "OPT"},
        }
    )


def _log(msg: str, t0: float) -> None:
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"[{mins:02d}:{secs:02d}]  {msg}", flush=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _write_run_meta(out_dir: Path, config: Dict[str, Any], backend: str) -> None:
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "backend": backend,
        "config_effective": config,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 4-Phase Pipeline
#   Phase 1 -- MOC Baseline (Prandtl-Meyer isentropic)
#   Phase 2 -- GA Optimisation with Quasi-1D evaluator
#   Phase 3 -- RANS Validation (optional, only when backend=openfoam)
#   Phase 4 -- Benchmark Comparison & Report
# ---------------------------------------------------------------------------
def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    out_dir  = Path(config["out_dir"])
    moc_dir  = out_dir / "moc"
    opt_dir  = out_dir / "opt"
    rans_dir = out_dir / "rans"
    comp_dir = out_dir / "comp"
    for d in [out_dir, moc_dir, opt_dir, rans_dir, comp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    moc_cfg  = config["moc"]
    eval_cfg = dict(config["evaluator"])
    opt_cfg  = config["optimization"]
    backend  = str(eval_cfg.get("backend", "quasi1d")).lower()

    _log(
        f"Pipeline START  |  backend={backend}  "
        f"algorithm={opt_cfg.get('algorithm', '?')}",
        t0,
    )
    _log(f"Output -> {out_dir.resolve()}", t0)
    _write_run_meta(out_dir, config, backend=backend)

    # Shared evaluator knobs
    eval_cfg["design_pressure_ratio"] = float(
        moc_cfg.get("pressure_ratio", 0.10)
    )
    throat  = float(moc_cfg["geometry"]["throat_y"])
    n_pts   = int(opt_cfg.get("n_points", 180))
    profile = str(opt_cfg.get("profile", "bezier_like"))

    # Quasi-1D config (always available, even when RANS is the main backend)
    q1d_cfg = dict(eval_cfg)
    q1d_cfg["backend"] = "quasi1d"

    # ==================================================================
    # PHASE 1 -- MOC Baseline
    # ==================================================================
    _log("=" * 56, t0)
    _log("PHASE 1  |  MOC Baseline (Prandtl-Meyer isentropic)", t0)
    _log("=" * 56, t0)

    moc_solver = MOCSolver(
        machExit=float(moc_cfg["mach_exit"]),
        pressureRatio=float(moc_cfg["pressure_ratio"]),
        gamma=float(moc_cfg.get("gamma", 1.4)),
    )
    moc_geometry = moc_solver.generateGeometry({**moc_cfg["geometry"]})
    moc_geometry.metadata["id"] = "moc_baseline"
    _log(
        f"  MOC geometry  |  Me={moc_cfg['mach_exit']}  "
        f"throat={throat * 1000:.1f} mm  points={len(moc_geometry.control_points)}",
        t0,
    )

    # Export geometry artefacts
    moc_geometry.exportGeo(str(moc_dir / "moc_geometry.csv"))
    moc_geometry.plotProfile(str(moc_dir / "moc_geometry.png"))

    # MLN auxiliary 2-panel plot (geometry + wall-angle distribution)
    _mln = MinimumLengthNozzle.from_params(
        moc_cfg["geometry"],
        mach_exit=float(moc_cfg["mach_exit"]),
        gamma=float(moc_cfg.get("gamma", 1.4)),
    )
    _mln.plot(str(moc_dir / "mln_geometry.png"))

    # Characteristic network plot (textbook style)
    moc_solver.plotCharacteristics(
        moc_geometry, str(moc_dir / "moc_characteristics.png")
    )

    # Evaluate MOC baseline with Quasi-1D for reference metrics
    moc_eval = CFDSimulation(
        geometry=moc_geometry, solverConfig=q1d_cfg, resultPath=str(out_dir)
    )
    moc_result = moc_eval.extractResults()
    moc_result.geometryId = "moc_baseline"
    moc_result.saveToJSON(str(moc_dir / "moc_q1d_result.json"))
    _log(
        f"  MOC Q1D  |  thrust={moc_result.thrust:.3f} N  "
        f"loss={moc_result.pressureLoss:.5f}",
        t0,
    )

    # ==================================================================
    # PHASE 2 -- GA Optimisation (Quasi-1D)
    # ==================================================================
    _log("=" * 56, t0)
    _log("PHASE 2  |  GA Optimisation with Quasi-1D evaluator", t0)
    _log("=" * 56, t0)

    pop  = int(opt_cfg.get("population", 8))
    gens = int(opt_cfg.get("generations", 3))
    _log(f"  population={pop}  generations={gens}  ~{pop * gens} evals", t0)

    # Thread-safe candidate counter
    _gid_counter = 0
    _gid_lock = threading.Lock()

    def _next_gid() -> str:
        nonlocal _gid_counter
        with _gid_lock:
            idx = _gid_counter
            _gid_counter += 1
        return f"opt_{idx:04d}"

    def objective(param_dict: Dict[str, float]) -> EvaluationResult:
        gid = _next_gid()
        geom = build_geometry_from_params(param_dict, throat, n_pts, profile, gid)
        local_eval = CFDSimulation(
            geometry=geom, solverConfig=q1d_cfg, resultPath=str(out_dir)
        )
        t_start = time.time()
        try:
            result = local_eval.extractResults()
            result.geometryId = gid
            dt_s = time.time() - t_start
            _log(
                f"  {gid}  thrust={result.thrust:8.2f} N  "
                f"loss={result.pressureLoss:.4f}  [{dt_s:.1f}s]",
                t0,
            )
            return result
        except Exception as exc:
            dt_s = time.time() - t_start
            _log(f"  {gid}  FAILED [{dt_s:.1f}s]  {exc}", t0)
            return EvaluationResult(
                machProfile=[],
                pressureLoss=1.0,
                thrust=-1.0e30,
                geometryId=gid,
                metadata={
                    "status": "failed",
                    "error": str(exc),
                    "params": dict(param_dict),
                },
            )

    search_space = {
        k: v for k, v in opt_cfg.items() if k not in ("algorithm", "seed")
    }
    search_space["campaign"] = str(eval_cfg.get("campaign", "")).lower()

    optimizer = Optimizer(
        searchSpace=search_space,
        objectiveFunc=objective,
        algorithm=str(opt_cfg.get("algorithm", "evolutionary")),
        seed=int(opt_cfg.get("seed", 42)),
    )
    runner = OptimizationRunner(
        optimizer=optimizer,
        evaluator=moc_eval,  # placeholder; objective() creates per-candidate evals
        historyPath=str(opt_dir / "optimization_history.json"),
    )

    runner.start()
    runner.saveHistory()
    _log(f"  GA DONE  |  {len(optimizer._history)} evaluations", t0)

    # ---- Multi-objective Pareto analysis ----
    _log("  Running Pareto / multi-objective ranking ...", t0)
    mo_summary: Dict[str, Any] = {}
    if _run_multiobjective is not None and len(optimizer._history) > 1:
        history_records: List[Dict[str, Any]] = []
        for r, p in zip(optimizer._history, optimizer._history_params):
            rec = r.to_dict()
            rec["params"] = dict(p)
            history_records.append(rec)
        try:
            mo_summary = _run_multiobjective(
                records=history_records,
                out_dir=str(opt_dir / "multiobjective"),
                w_thrust=float(opt_cfg.get("w_thrust", 0.7)),
                w_pressure_loss=float(opt_cfg.get("w_pressure_loss", 0.3)),
                moc_thrust=float(moc_result.thrust),
                moc_loss=float(moc_result.pressureLoss),
            )
        except Exception as exc:
            mo_summary = {"error": str(exc)}

    # ---- Select best candidate (Pareto knee-point preferred) ----
    if mo_summary and "best_pareto_knee" in mo_summary:
        knee_idx = int(
            mo_summary["best_pareto_knee"].get("candidate_index", -1)
        )
        if 0 <= knee_idx < len(optimizer._history):
            selected_idx = knee_idx
            _log(f"  Selected Pareto knee-point candidate #{knee_idx}", t0)
        else:
            selected_idx = optimizer._best_idx()
            _log(f"  Knee invalid -> best-score #{selected_idx}", t0)
    else:
        selected_idx = optimizer._best_idx()
        _log(f"  Best-score candidate #{selected_idx}", t0)

    best_params = optimizer._history_params[selected_idx]
    best_q1d    = optimizer._history[selected_idx]
    _log(
        f"  Best  |  thrust={best_q1d.thrust:.3f} N  "
        f"loss={best_q1d.pressureLoss:.5f}  params={best_params}",
        t0,
    )

    optimized_geometry = build_geometry_from_params(
        best_params, throat, n_pts, profile, "optimized_best"
    )
    optimized_geometry.exportGeo(str(opt_dir / "optimized_geometry.csv"))
    optimized_geometry.plotProfile(str(opt_dir / "optimized_geometry.png"))

    # ==================================================================
    # PHASE 3 -- RANS Validation (optional)
    # ==================================================================
    rans_results: Dict[str, Any] = {}

    if backend == "openfoam":
        _log("=" * 56, t0)
        _log("PHASE 3  |  RANS Validation (OpenFOAM shockFluid)", t0)
        _log("=" * 56, t0)

        rans_top_k = int(opt_cfg.get("rans_top_k", 3))

        # Pick the top-K candidates by optimizer score for RANS validation
        scored_idxs = sorted(
            range(len(optimizer._history)),
            key=lambda i: optimizer._objective_score(optimizer._history[i]),
            reverse=True,
        )[:rans_top_k]

        rans_cfg = {
            **eval_cfg,
            "mach_exit": float(moc_cfg.get("mach_exit", 2.3)),
            "pressure_ratio": float(moc_cfg.get("pressure_ratio", 0.10)),
        }

        for idx in scored_idxs:
            p = optimizer._history_params[idx]
            gid = f"rans_{idx:04d}"
            geom = build_geometry_from_params(p, throat, n_pts, profile, gid)
            rans_eval = OpenFOAMRANSEvaluator(
                geometry=geom, solverConfig=rans_cfg, resultPath=str(rans_dir)
            )
            _log(f"  RANS evaluating #{idx} ({gid}) ...", t0)
            try:
                r = rans_eval.extractResults()
                r.geometryId = gid
                r.saveToJSON(str(rans_dir / f"{gid}_result.json"))
                rans_results[gid] = {
                    "candidate_index": idx,
                    "thrust": float(r.thrust),
                    "pressureLoss": float(r.pressureLoss),
                    "params": dict(p),
                    "status": "ok",
                }
                _log(
                    f"  {gid}  thrust={r.thrust:.3f} N  "
                    f"loss={r.pressureLoss:.5f}  [RANS OK]",
                    t0,
                )
            except Exception as exc:
                rans_results[gid] = {
                    "candidate_index": idx,
                    "status": "failed",
                    "error": str(exc),
                    "params": dict(p),
                }
                _log(f"  {gid}  RANS FAILED  {exc}", t0)

        (rans_dir / "rans_summary.json").write_text(
            json.dumps(rans_results, indent=2), encoding="utf-8"
        )

        # If any RANS succeeded, adopt the best RANS candidate
        ok_rans = {
            k: v for k, v in rans_results.items() if v.get("status") == "ok"
        }
        if ok_rans:
            best_rans_key = max(ok_rans, key=lambda k: ok_rans[k]["thrust"])
            br = ok_rans[best_rans_key]
            _log(
                f"  Best RANS: {best_rans_key}  "
                f"thrust={br['thrust']:.3f} N",
                t0,
            )
            best_params = br["params"]
            optimized_geometry = build_geometry_from_params(
                best_params, throat, n_pts, profile, "rans_best"
            )
            optimized_geometry.exportGeo(
                str(opt_dir / "optimized_geometry_rans.csv")
            )
    else:
        _log("=" * 56, t0)
        _log("PHASE 3  |  RANS Validation  [skipped: backend=quasi1d]", t0)
        _log("=" * 56, t0)

    # ==================================================================
    # PHASE 4 -- Benchmark Comparison & Report
    # ==================================================================
    _log("=" * 56, t0)
    _log("PHASE 4  |  Benchmark Comparison & Report", t0)
    _log("=" * 56, t0)

    bench_eval = CFDSimulation(
        geometry=moc_geometry, solverConfig=q1d_cfg, resultPath=str(out_dir)
    )
    suite = BenchmarkSuite(
        mocGeometry=moc_geometry,
        optimizedGeometry=optimized_geometry,
        evaluator=bench_eval,
    )

    comparison: Dict[str, Any]
    benchmark_error = ""
    try:
        suite.runAll()
        comparison = suite.save(str(comp_dir))

        # Optimisation convergence plots
        generate_optimization_plots(
            history=optimizer._history,
            out_dir=str(opt_dir),
            moc_thrust=(
                suite.mocResult.thrust if suite.mocResult else None
            ),
            moc_pressure_loss=(
                suite.mocResult.pressureLoss if suite.mocResult else None
            ),
            population=(
                pop
                if str(opt_cfg.get("algorithm", "")).lower()
                in {"ga", "cma-es", "evolutionary"}
                else None
            ),
            moc_geometry=moc_geometry,
            throat_radius=throat,
            n_points=n_pts,
            profile=profile,
            gamma=float(eval_cfg.get("gamma", 1.4)),
            gas_constant=float(eval_cfg.get("gas_constant", 287.0)),
            mach_exit=float(moc_cfg.get("mach_exit", 2.0)),
        )

        # Side-by-side comparison plots (MOC vs optimised)
        if suite.mocResult is not None and suite.optResult is not None:
            generate_comparison_plots(
                moc_geometry=moc_geometry,
                optimized_geometry=optimized_geometry,
                moc_result=suite.mocResult,
                optimized_result=suite.optResult,
                out_dir=str(comp_dir),
                solver_config=eval_cfg,
                mach_exit=float(moc_cfg.get("mach_exit", 2.0)),
            )

        # Field plots for each geometry
        bench_eval.geometry = moc_geometry
        bench_eval.extractResults()
        bench_eval.plotFields(str(moc_dir / "moc_fields.png"))

        bench_eval.geometry = optimized_geometry
        bench_eval.extractResults()
        bench_eval.plotFields(str(opt_dir / "optimized_fields.png"))

    except Exception as exc:
        benchmark_error = str(exc)
        comparison = {"status": "failed", "error": benchmark_error}
        (comp_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )

    # ---- Analysis report ----
    note = AnalysisNote(
        title="Nozzle Design Benchmark Report",
        notes=(
            "4-phase pipeline: MOC baseline (Prandtl-Meyer) -> "
            "GA optimisation (Quasi-1D) -> optional RANS validation -> "
            "benchmark comparison.  Pareto knee-point selection."
        ),
    )
    for k, v in comparison.items():
        note.addMetric(k, v)
    note.addMetric("optimizer_algorithm", optimizer.algorithm)
    note.addMetric("optimizer_evaluations", len(optimizer._history))
    note.addMetric("ga_population", pop)
    note.addMetric("ga_generations", gens)
    note.addMetric("best_parameters", best_params)
    note.addMetric("selected_candidate_index", selected_idx)

    if optimizer._history:
        thrust_vals = [float(r.thrust) for r in optimizer._history]
        loss_vals   = [float(r.pressureLoss) for r in optimizer._history]
        note.addMetric(
            "thrust_mean", sum(thrust_vals) / len(thrust_vals)
        )
        note.addMetric(
            "pressure_loss_mean", sum(loss_vals) / len(loss_vals)
        )
    if mo_summary:
        note.addMetric("multiobjective_summary", mo_summary)
    if rans_results:
        note.addMetric("rans_validation", rans_results)

    note.save(str(out_dir / "report.md"))
    note.saveJSON(str(out_dir / "report.json"))

    elapsed_total = time.time() - t0
    summary = {
        "status": "ok" if not benchmark_error else "failed",
        "out_dir": str(out_dir),
        "moc_geometry": str(moc_dir / "moc_geometry.csv"),
        "optimized_geometry": str(opt_dir / "optimized_geometry.csv"),
        "comparison": comparison,
        "multiobjective": mo_summary,
        "rans_validation": rans_results,
        "elapsed_seconds": round(elapsed_total, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _log(
        f"Pipeline DONE  |  status={summary['status']}  "
        f"elapsed={elapsed_total:.1f}s",
        t0,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nozzle design benchmark: MOC -> GA/Q1D -> RANS -> Report"
    )
    parser.add_argument(
        "--config", type=str, default="", help="JSON config file"
    )
    parser.add_argument(
        "--algorithm", type=str, default="",
        help="Override optimiser algorithm (evolutionary|random|grid|bayesian_proxy)",
    )
    parser.add_argument(
        "--out", type=str, default="", help="Override output directory"
    )
    parser.add_argument(
        "--backend", type=str, default="",
        help="Override evaluator backend (quasi1d|openfoam)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    if args.config:
        user_cfg = json.loads(
            Path(args.config).read_text(encoding="utf-8-sig")
        )
        for key, value in user_cfg.items():
            if (
                isinstance(value, dict)
                and key in cfg
                and isinstance(cfg[key], dict)
            ):
                cfg[key].update(value)
            else:
                cfg[key] = value

    if args.algorithm:
        cfg["optimization"]["algorithm"] = args.algorithm
    if args.out:
        cfg["out_dir"] = args.out
    if args.backend:
        cfg["evaluator"]["backend"] = args.backend

    summary = run_pipeline(cfg)

    # ---------- Formatted results ----------
    cmp = summary.get("comparison", {})
    mo  = summary.get("multiobjective", {})
    sep = "-" * 56

    print()
    print(sep)
    print("  NOZZLE DESIGN BENCHMARK - RESULTS")
    print(sep)
    print(f"  Status       : {summary.get('status', '?').upper()}")
    print(f"  Elapsed      : {summary.get('elapsed_seconds', 0):.1f} s")
    print(f"  Output dir   : {summary.get('out_dir', '')}")
    print(sep)

    if cmp and cmp.get("status") != "failed":
        print("  PERFORMANCE COMPARISON  (MOC baseline vs GA-optimised)")
        fmt = "  {:<25s}  {:>10s}  {:>10s}  {:>12s}"
        print(fmt.format("", "MOC", "OPT", "delta"))
        moc_t  = cmp.get("moc_thrust", 0)
        opt_t  = cmp.get("optimized_thrust", 0)
        moc_pl = cmp.get("moc_pressure_loss", 0)
        opt_pl = cmp.get("optimized_pressure_loss", 0)
        d_t    = cmp.get("delta_thrust", 0)
        d_t_p  = cmp.get("delta_thrust_percent", 0)
        d_pl   = cmp.get("delta_pressure_loss", 0)
        print(
            f"  {'Thrust [N]':<25s}  {moc_t:>10.3f}  {opt_t:>10.3f}"
            f"  {d_t:>+10.3f} ({d_t_p:+.2f}%)"
        )
        print(
            f"  {'Pressure loss [-]':<25s}  {moc_pl:>10.4f}  {opt_pl:>10.4f}"
            f"  {d_pl:>+10.4f}"
        )
        print(sep)

    if mo and "best_pareto_knee" in mo:
        knee = mo["best_pareto_knee"]
        p = knee.get("params", {})
        print("  PARETO KNEE-POINT CANDIDATE")
        print(f"  Thrust       : {knee.get('thrust', 0):.3f} N")
        print(f"  Pressure loss: {knee.get('pressureLoss', 0):.4f}")
        print(f"  exit_radius  : {p.get('exit_radius', 0) * 1000:.2f} mm")
        print(f"  length       : {p.get('length', 0) * 1000:.1f} mm")
        print(f"  shape        : {p.get('shape', 0):.3f}")
        pf = mo.get("pareto_front_size", "?")
        nc = mo.get("n_candidates", "?")
        print(f"  Pareto front : {pf} / {nc} candidates")
        print(sep)

    rans_v = summary.get("rans_validation", {})
    if rans_v:
        ok_r = {k: v for k, v in rans_v.items() if v.get("status") == "ok"}
        print(f"  RANS Validated: {len(ok_r)} / {len(rans_v)} candidates")
        for k, v in ok_r.items():
            print(
                f"    {k}  thrust={v['thrust']:.3f} N  "
                f"loss={v['pressureLoss']:.5f}"
            )
        print(sep)

    print(f"  report -> {summary.get('out_dir', '')}/report.md")
    print(sep)
    print()


if __name__ == "__main__":
    main()
