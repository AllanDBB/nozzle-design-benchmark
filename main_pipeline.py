from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any

from analysis import AnalysisNote, generate_comparison_plots, generate_optimization_plots
from benchmarks import BenchmarkSuite
from evaluators import CFDSimulation, OpenFOAMRANSEvaluator, EvaluationResult
from geometry import MOCSolver, NozzleGeometry
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


def default_config() -> Dict[str, Any]:
    return {
        "out_dir": "out/pipeline",
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
        "evaluator": {
            "backend": "openfoam",
            "gamma": 1.4,
            "stagnation_temperature": 1000.0,
            "stagnation_pressure": 6.0e5,
            "ambient_pressure": 8.0e4,
            "n_samples": 90,
            "fallback_on_failure": False,
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
            "mesh_nx": 180,
            "mesh_ny": 80,
            "mesh_nz": 12,
            "end_time": 1500,
            "write_interval": 300,
            "inlet_velocity": 20.0,
            "k_inlet": 1.0,
            "epsilon_inlet": 50.0,
            "p_initial": 5.4e5,
            "u_initial": 1.0,
            "t_initial": 1000.0,
            # GPU acceleration for quasi-1D solver (requires CuPy).
            "use_gpu": False,
        },
        "optimization": {
            "algorithm": "random",
            "seed": 42,
            "bounds": {
                "exit_radius": [0.055, 0.075],
                "length": [0.20, 0.32],
                "shape": [1.4, 2.2],
            },
            "n_samples": 3,
            "n_points": 180,
            "profile": "bezier_like",
            # Multi-objective weights (thrust maximised, loss minimised).
            "w_thrust": 1.0,
            "w_pressure_loss": 0.0,
            # Set to true to use Pareto-knee candidate instead of best-score.
            "use_pareto_knee": False,
            # Parallel workers for population evaluation (1 = sequential).
            "n_workers": 1,
        },
    }


def build_geometry_from_params(params: Dict[str, float], throat_radius: float, n_points: int, profile: str, gid: str) -> NozzleGeometry:
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
    tag = f"[{mins:02d}:{secs:02d}]"
    print(f"{tag}  {msg}", flush=True)


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    moc_dir  = out_dir / "moc"
    opt_dir  = out_dir / "opt"
    comp_dir = out_dir / "comp"
    for _d in [moc_dir, opt_dir, comp_dir]:
        _d.mkdir(parents=True, exist_ok=True)

    backend = str(config.get("evaluator", {}).get("backend", "quasi1d")).lower()
    n_cands = config.get("optimization", {}).get("n_samples", "?")
    algorithm = config.get("optimization", {}).get("algorithm", "?")
    _log(f"Pipeline START  |  backend={backend}  algorithm={algorithm}  n_samples={n_cands}", t0)
    _log(f"Output → {out_dir.resolve()}", t0)
    if backend == "openfoam":
        _log("⚑  OpenFOAM RANS mode — each candidate will launch rhoSimpleFoam in Docker", t0)

    # 1) Generate MOC baseline geometry.
    moc_cfg = config["moc"]
    moc_solver = MOCSolver(
        machExit=float(moc_cfg["mach_exit"]),
        pressureRatio=float(moc_cfg["pressure_ratio"]),
        gamma=float(moc_cfg.get("gamma", 1.4)),
    )
    moc_geometry = moc_solver.generateGeometry({**moc_cfg["geometry"]})
    moc_geometry.metadata["id"] = "moc_baseline"

    _log(f"MOC geometry generated  |  Me={moc_cfg['mach_exit']}  throat={moc_cfg['geometry']['throat_y']*1000:.1f} mm", t0)
    moc_geometry.exportGeo(str(moc_dir / "moc_geometry.csv"))
    moc_geometry.plotProfile(str(moc_dir / "moc_geometry.png"))
    moc_solver.plotCharacteristics(moc_geometry, str(moc_dir / "moc_characteristics.png"))

    # 2) Optimize parametrized geometry using CFD-like evaluator.
    eval_cfg = dict(config["evaluator"])
    opt_cfg = config["optimization"]
    require_rans = bool(eval_cfg.get("require_rans_converged", False))

    if require_rans and str(eval_cfg.get("backend", "openfoam")).lower() != "openfoam":
        raise ValueError("require_rans_converged=true requires evaluator.backend='openfoam'")

    def is_rans_converged(result: EvaluationResult) -> bool:
        return str(result.metadata.get("backend", "")).lower() == "openfoam_rans"

    def make_evaluator(geometry: NozzleGeometry):
        backend = str(eval_cfg.get("backend", "quasi1d")).lower()
        if backend == "openfoam":
            return OpenFOAMRANSEvaluator(geometry=geometry, solverConfig=eval_cfg, resultPath=str(out_dir))
        return CFDSimulation(geometry=geometry, solverConfig=eval_cfg, resultPath=str(out_dir))

    evaluator = make_evaluator(moc_geometry)
    throat = float(moc_cfg["geometry"]["throat_y"])
    n_points = int(opt_cfg.get("n_points", 180))
    profile = str(opt_cfg.get("profile", "bezier_like"))

    # Thread-safe counter so parallel workers get unique IDs.
    _gid_counter = 0
    _gid_lock = threading.Lock()

    def _next_gid() -> str:
        nonlocal _gid_counter
        with _gid_lock:
            idx = _gid_counter
            _gid_counter += 1
        return f"opt_candidate_{idx:04d}"

    def objective(param_dict: Dict[str, float]):
        gid = _next_gid()
        geom = build_geometry_from_params(param_dict, throat, n_points, profile, gid)
        # Create a fresh evaluator per call so parallel workers don't share state.
        local_eval = make_evaluator(geom)
        t_start = time.time()
        try:
            result = local_eval.extractResults()
            result.geometryId = gid
            elapsed_c = time.time() - t_start
            status_tag = result.metadata.get("backend", backend)
            _log(
                f"  {gid}  thrust={result.thrust:8.2f} N  loss={result.pressureLoss:.4f}"
                f"  [{status_tag}  {elapsed_c:.1f}s]",
                t0,
            )
            if require_rans and not is_rans_converged(result):
                raise RuntimeError(
                    f"Candidate {gid} not RANS-converged (backend={result.metadata.get('backend', 'unknown')})"
                )
            return result
        except Exception as exc:
            elapsed_c = time.time() - t_start
            _log(f"  {gid}  FAILED  [{elapsed_c:.1f}s]  {exc}", t0)
            # Penalize failed CFD candidates so optimization can continue.
            return EvaluationResult(
                machProfile=[],
                pressureLoss=1.0,
                thrust=-1.0e30,
                geometryId=gid,
                metadata={"status": "failed", "error": str(exc), "params": dict(param_dict)},
            )

    optimizer = Optimizer(
        searchSpace={k: v for k, v in opt_cfg.items() if k not in ("algorithm", "seed")},
        objectiveFunc=objective,
        algorithm=str(opt_cfg.get("algorithm", "ga")),
        seed=int(opt_cfg.get("seed", 42)),
    )
    _log(f"Optimization START  |  {n_cands} candidates  backend={backend}", t0)
    runner = OptimizationRunner(optimizer=optimizer, evaluator=evaluator, historyPath=str(opt_dir / "optimization_history.json"))
    runner.start()
    runner.saveHistory()
    _log(f"Optimization DONE   |  evaluated {len(optimizer._history)} candidates", t0)

    # ── Multi-objective ranking (Pareto analysis on full history) ──────────
    _log("Running Pareto / multi-objective ranking…", t0)
    mo_summary: Dict[str, Any] = {}
    if _run_multiobjective is not None and len(optimizer._history) > 1:
        history_records = []
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
            )
        except Exception as _exc:
            mo_summary = {"error": str(_exc)}

    # ── Select best candidate ──────────────────────────────────────────────
    selected_idx: int
    use_pareto_knee = bool(opt_cfg.get("use_pareto_knee", False))
    if require_rans:
        valid_idxs = [i for i, r in enumerate(optimizer._history) if is_rans_converged(r)]
        if not valid_idxs:
            raise RuntimeError(
                "No RANS-converged candidates found. "
                "All evaluations failed to converge in OpenFOAM (or fell back)."
            )
        selected_idx = max(valid_idxs, key=lambda i: optimizer._objective_score(optimizer._history[i]))
    elif use_pareto_knee and mo_summary and "best_pareto_knee" in mo_summary:
        knee_idx = int(mo_summary["best_pareto_knee"].get("candidate_index", -1))
        selected_idx = knee_idx if 0 <= knee_idx < len(optimizer._history) else optimizer._best_idx()
    else:
        selected_idx = optimizer._best_idx()

    best_params = optimizer._history_params[selected_idx]


    optimized_geometry = build_geometry_from_params(best_params, throat, n_points, profile, "optimized_best")
    optimized_geometry.exportGeo(str(opt_dir / "optimized_geometry.csv"))
    optimized_geometry.plotProfile(str(opt_dir / "optimized_geometry.png"))

    # 3) Benchmark MOC vs optimized with same evaluator.
    _log("Benchmark START  |  evaluating MOC and optimized geometry…", t0)
    bench_eval = make_evaluator(moc_geometry)
    suite = BenchmarkSuite(mocGeometry=moc_geometry, optimizedGeometry=optimized_geometry, evaluator=bench_eval)
    comparison: Dict[str, Any]
    benchmark_error = ""
    try:
        suite.runAll()
        if require_rans:
            if suite.mocResult is None or suite.optResult is None:
                raise RuntimeError("Benchmark results missing.")
            if not is_rans_converged(suite.mocResult):
                raise RuntimeError("MOC benchmark is not RANS-converged.")
            if not is_rans_converged(suite.optResult):
                raise RuntimeError("Optimized benchmark is not RANS-converged.")
        comparison = suite.save(str(comp_dir))
        generate_optimization_plots(
            history=optimizer._history,
            out_dir=str(opt_dir),
            moc_thrust=suite.mocResult.thrust if suite.mocResult is not None else None,
            moc_pressure_loss=suite.mocResult.pressureLoss if suite.mocResult is not None else None,
            population=int(opt_cfg.get("population", 0)) if str(opt_cfg.get("algorithm", "")).lower() in {"ga", "cma-es", "evolutionary"} else None,
            moc_geometry=moc_geometry,
            throat_radius=throat,
            n_points=n_points,
            profile=profile,
            gamma=float(eval_cfg.get("gamma", 1.4)),
            gas_constant=float(eval_cfg.get("gas_constant", 287.0)),
            mach_exit=float(moc_cfg.get("mach_exit", 2.0)),
        )
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

        bench_eval.geometry = moc_geometry
        bench_eval.extractResults()
        bench_eval.plotFields(str(moc_dir / "moc_fields.png"))

        bench_eval.geometry = optimized_geometry
        bench_eval.extractResults()
        bench_eval.plotFields(str(opt_dir / "optimized_fields.png"))
    except Exception as exc:
        benchmark_error = str(exc)
        comparison = {
            "status": "failed",
            "error": benchmark_error,
        }
        (comp_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # 4) Analysis report.
    note = AnalysisNote(
        title="Nozzle Design Benchmark Report",
        notes=(
            "Comparative study between MOC baseline and CFD-optimized geometry. "
            "Optimization is used as diagnostic reference, not replacement of MOC."
        ),
    )
    for k, v in comparison.items():
        note.addMetric(k, v)
    note.addMetric("optimizer_algorithm", optimizer.algorithm)
    note.addMetric("optimizer_evaluations", len(optimizer._history))
    note.addMetric("best_parameters", best_params)
    note.addMetric("selected_candidate_index", selected_idx)
    if mo_summary:
        note.addMetric("multiobjective_summary", mo_summary)

    note.save(str(out_dir / "report.md"))
    note.saveJSON(str(out_dir / "report.json"))

    elapsed_total = time.time() - t0
    summary = {
        "status": "ok" if not benchmark_error else "failed",
        "out_dir": str(out_dir),
        "moc_geometry": str(moc_dir / "moc_geometry.csv"),
        "optimized_geometry": str(opt_dir / "optimized_geometry.csv"),
        "opt_dir": str(opt_dir),
        "moc_dir": str(moc_dir),
        "comp_dir": str(comp_dir),
        "comparison": comparison,
        "multiobjective": mo_summary,
        "elapsed_seconds": round(elapsed_total, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"Pipeline DONE   |  status={summary['status']}  elapsed={elapsed_total:.1f}s", t0)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full nozzle design benchmark pipeline")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config file")
    parser.add_argument("--algorithm", type=str, default="", help="Override optimizer algorithm")
    parser.add_argument("--out", type=str, default="", help="Override output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    if args.config:
        user_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        # shallow merge by sections
        for key, value in user_cfg.items():
            if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    if args.algorithm:
        cfg["optimization"]["algorithm"] = args.algorithm
    if args.out:
        cfg["out_dir"] = args.out

    summary = run_pipeline(cfg)

    # ── Formatted results output ──────────────────────────────────────────
    cmp = summary.get("comparison", {})
    mo  = summary.get("multiobjective", {})
    sep = "─" * 52

    print()
    print(sep)
    print("  NOZZLE DESIGN BENCHMARK — RESULTS")
    print(sep)
    print(f"  Status          : {summary.get('status', '?').upper()}")
    print(f"  Elapsed         : {summary.get('elapsed_seconds', 0):.1f} s")
    print(f"  Output dir      : {summary.get('out_dir', '')}")
    print(sep)

    if cmp and cmp.get("status") != "failed":
        print("  PERFORMANCE COMPARISON")
        print(f"  {'':25s}  {'MOC':>10s}  {'OPT':>10s}  {'Δ':>10s}")
        moc_t  = cmp.get("moc_thrust", 0)
        opt_t  = cmp.get("optimized_thrust", 0)
        moc_pl = cmp.get("moc_pressure_loss", 0)
        opt_pl = cmp.get("optimized_pressure_loss", 0)
        d_t_pct = cmp.get("delta_thrust_percent", 0)
        print(f"  {'Thrust [N]':25s}  {moc_t:>10.3f}  {opt_t:>10.3f}  {cmp.get('delta_thrust', 0):>+10.3f} ({d_t_pct:+.2f}%)")
        print(f"  {'Pressure loss [-]':25s}  {moc_pl:>10.4f}  {opt_pl:>10.4f}  {cmp.get('delta_pressure_loss', 0):>+10.4f}")
        print(sep)

    if mo and "best_by_thrust" in mo:
        best = mo.get("best_by_thrust", {})
        p    = best.get("params", {})
        print("  BEST CANDIDATE (by thrust)")
        print(f"  Thrust          : {best.get('thrust', 0):.3f} N")
        print(f"  Pressure loss   : {best.get('pressureLoss', 0):.4f}")
        print(f"  exit_radius     : {p.get('exit_radius', 0)*1000:.2f} mm")
        print(f"  length          : {p.get('length', 0)*1000:.1f} mm")
        print(f"  shape           : {p.get('shape', 0):.3f}")
        pf = mo.get("pareto_front_size", "?")
        nc = mo.get("n_candidates", "?")
        print(f"  Pareto front    : {pf} / {nc} candidates")
        print(sep)

    print(f"  report.md  → {summary.get('out_dir', '')}/report.md")
    print(sep)
    print()


if __name__ == "__main__":
    main()
