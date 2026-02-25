from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import random
import subprocess
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
            "campaign": "design_supersonic",
            "gamma": 1.4,
            "stagnation_temperature": 1000.0,
            "stagnation_pressure": 6.0e5,
            "ambient_pressure": 8.0e4,
            "n_samples": 90,
            "fallback_on_failure": False,
            "require_rans_converged": True,
            "require_converged_series": True,
            "allow_unsteady_overexpanded": True,
            "use_window_averages_overexpanded": True,
            "min_writes": 8,
            "convergence_window": 5,
            "convergence_tol": {
                "p_out_rel_std": 0.02,
                "mdot_rel_std": 0.02,
                "ux_out_rel_std": 0.03,
            },
            "overexpanded_unsteady_tol": {
                "p_out_rel_std_max": 0.20,
                "mdot_rel_std_max": 0.80,
                "ux_out_rel_std_max": 1.20,
            },
            "enable_delta_t_abort": True,
            "delta_t_abort_threshold": 1e-80,
            "delta_t_abort_streak": 20,
            "outlet_bc_mode": "wave_transmissive",
            "sampling_nx": 81,
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
            "mesh_nx": 120,
            "mesh_ny": 54,
            "mesh_nz": 12,
            "end_time": 0.004,
            "write_interval": 0.0004,
            "max_co": 0.3,
            "delta_t_init": 1e-7,
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
            "strategy": "single_fidelity",
            "algorithm": "evolutionary",
            "seed": 42,
            "bounds": {
                "exit_radius": [0.050, 0.080],
                "length": [0.18, 0.32],
                "theta_max": [10.0, 40.0],
                "inflection_frac": [0.15, 0.55],
                "throat_angle": [1.0, 10.0],
            },
            "population": 24,
            "generations": 12,
            "sigma": 0.15,
            "sigma_decay": 0.88,
            "tournament_k": 3,
            "n_samples": 30,
            "n_points": 180,
            "profile": "rao",
            # Seed MOC contour into initial population (filled at runtime).
            "seed_moc": True,
            # Multi-objective weights (thrust maximised, loss minimised).
            "w_thrust": 1.0,
            "w_pressure_loss": 0.3,
            # Set to true to use Pareto-knee candidate instead of best-score.
            "use_pareto_knee": True,
            # Parallel workers for population evaluation (1 = sequential).
            "n_workers": 1,
            "low_fidelity_samples": 300,
            "high_fidelity_top_k": 24,
            "objective_terms": {
                "design_shock_penalty": 0.10,
                "overexpanded_instability_penalty": 0.05,
                "overexpanded_outlet_osc_penalty": 0.05,
                "overexpanded_wall_rms_penalty": 0.03,
            },
        },
    }


def build_geometry_from_params(params: Dict[str, float], throat_radius: float, n_points: int, profile: str, gid: str) -> NozzleGeometry:
    p: Dict[str, Any] = {
        "throat_radius": throat_radius,
        "exit_radius": params["exit_radius"],
        "length": params["length"],
        "n_points": n_points,
        "profile": profile,
        "metadata": {"id": gid, "source": "OPT"},
    }
    # Forward profile-specific params.
    if profile == "rao":
        p["theta_max"] = params.get("theta_max", 22.0)
        p["inflection_frac"] = params.get("inflection_frac", 0.35)
        p["throat_angle"] = params.get("throat_angle", 3.0)
    else:
        p["shape"] = params.get("shape", 1.8)
    return NozzleGeometry.fromParams(p)


def _log(msg: str, t0: float) -> None:
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    tag = f"[{mins:02d}:{secs:02d}]"
    print(f"{tag}  {msg}", flush=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_run_meta(out_dir: Path, config: Dict[str, Any], backend: str, solver: str) -> None:
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "backend": backend,
        "solver": solver,
        "config_effective": config,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _detect_solver_from_case(case_dir: str) -> str:
    try:
        p = Path(case_dir)
        candidates = [p / "log.shockFluid", p / "log.rhoSimpleFoam", p / "log.fluid"]
        for log in candidates:
            if not log.exists():
                continue
            text = log.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if "Exec" in line and "-solver" in line:
                    parts = line.strip().split()
                    if parts:
                        return parts[-1]
            if log.name == "log.shockFluid":
                return "shockFluid"
            if log.name == "log.rhoSimpleFoam":
                return "rhoSimpleFoam"
            if log.name == "log.fluid":
                return "fluid"
    except Exception:
        pass
    return "unknown"


def _uniform_candidates(bounds: Dict[str, Any], n: int, seed: int) -> list[Dict[str, float]]:
    rng = random.Random(seed)
    cands: list[Dict[str, float]] = []
    for _ in range(max(1, n)):
        row: Dict[str, float] = {}
        for k, (lo, hi) in bounds.items():
            row[k] = rng.uniform(float(lo), float(hi))
        cands.append(row)
    return cands


def _normalized_screen_score(result: EvaluationResult, terms: Dict[str, float], campaign: str) -> float:
    thrust = max(float(result.thrust), -1.0e30)
    loss = max(0.0, min(1.0, float(result.pressureLoss)))
    score = thrust * (1.0 - 0.35 * loss)
    shock_present = bool((result.shock or {}).get("present", False))
    if campaign == "design_supersonic" and shock_present:
        score *= max(0.0, 1.0 - float(terms.get("design_shock_penalty", 0.10)))
    if campaign == "overexpanded_sea_level":
        x_std = (result.shock or {}).get("x_std")
        if x_std is not None:
            score *= max(0.0, 1.0 - float(terms.get("overexpanded_instability_penalty", 0.05)) * min(1.0, float(x_std) / 0.05))
        conv_metrics = (result.convergence or {}).get("metrics", {})
        if isinstance(conv_metrics, dict):
            osc_index = max(
                float(conv_metrics.get("p_out_rel_std", 0.0)) / 0.02,
                float(conv_metrics.get("mdot_rel_std", 0.0)) / 0.02,
                float(conv_metrics.get("ux_out_rel_std", 0.0)) / 0.03,
            )
            score *= max(0.0, 1.0 - float(terms.get("overexpanded_outlet_osc_penalty", 0.05)) * min(1.0, osc_index))
            wall_rel = conv_metrics.get("wall_p_rel_rms", result.metadata.get("wall_pressure_rel_rms"))
            if wall_rel is not None:
                score *= max(0.0, 1.0 - float(terms.get("overexpanded_wall_rms_penalty", 0.03)) * min(1.0, float(wall_rel) / 0.05))
    return score


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
    _log(f"Output -> {out_dir.resolve()}", t0)
    solver_name = "shockFluid" if backend == "openfoam" else "quasi1d"
    _write_run_meta(out_dir, config, backend=backend, solver=solver_name)
    if backend == "openfoam":
        _log("OpenFOAM RANS mode - each candidate runs shockFluid (density-based Kurganov) in Docker", t0)

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
    require_rans_cfg = bool(eval_cfg.get("require_rans_converged", False))
    require_rans = require_rans_cfg and str(eval_cfg.get("backend", "quasi1d")).lower() == "openfoam"
    eval_cfg["design_pressure_ratio"] = float(moc_cfg.get("pressure_ratio", 0.10))

    if require_rans_cfg and str(eval_cfg.get("backend", "quasi1d")).lower() != "openfoam":
        require_rans = False

    def is_rans_converged(result: EvaluationResult) -> bool:
        if str(result.metadata.get("backend", "")).lower() != "openfoam_rans":
            return False
        if bool(eval_cfg.get("require_converged_series", True)):
            conv = result.convergence or {}
            campaign = str(eval_cfg.get("campaign", "")).lower()
            if campaign == "overexpanded_sea_level" and bool(eval_cfg.get("allow_unsteady_overexpanded", True)):
                return bool(conv.get("series_usable", conv.get("converged_series", False)))
            return bool(conv.get("converged_series", False))
        return True

    def make_evaluator(geometry: NozzleGeometry):
        backend = str(eval_cfg.get("backend", "quasi1d")).lower()
        if backend == "openfoam":
            # Inject mach_exit from MOC config so shockFluid can hot-start at
            # the correct supersonic IC rather than having to guess from geometry.
            cfg_with_mach = {
                **eval_cfg,
                "mach_exit": float(moc_cfg.get("mach_exit", 2.3)),
                "pressure_ratio": float(moc_cfg.get("pressure_ratio", 0.10)),
            }
            return OpenFOAMRANSEvaluator(geometry=geometry, solverConfig=cfg_with_mach, resultPath=str(out_dir))
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

    strategy = str(opt_cfg.get("strategy", "single_fidelity")).lower()
    search_space = {k: v for k, v in opt_cfg.items() if k not in ("algorithm", "seed")}
    search_space["campaign"] = str(eval_cfg.get("campaign", "")).lower()
    search_space["objective_terms"] = dict(opt_cfg.get("objective_terms", {}))

    # Seed MOC contour into the initial GA population so the optimizer
    # starts from the known-good baseline instead of purely random guesses.
    if bool(opt_cfg.get("seed_moc", True)) and profile == "rao":
        moc_rao_params = moc_geometry.extract_rao_params()
        _log(f"  MOC seed  |  theta_max={moc_rao_params['theta_max']:.1f} deg  "
             f"inflect={moc_rao_params['inflection_frac']:.3f}  "
             f"throat_ang={moc_rao_params['throat_angle']:.1f} deg", t0)
        search_space.setdefault("seed_candidates", []).append(moc_rao_params)

    optimizer = Optimizer(
        searchSpace=search_space,
        objectiveFunc=objective,
        algorithm=str(opt_cfg.get("algorithm", "ga")),
        seed=int(opt_cfg.get("seed", 42)),
    )
    runner = OptimizationRunner(optimizer=optimizer, evaluator=evaluator, historyPath=str(opt_dir / "optimization_history.json"))

    if strategy == "multifidelity_screen":
        bounds = dict(opt_cfg.get("bounds", {}))
        low_n = int(opt_cfg.get("low_fidelity_samples", 300))
        high_k = int(opt_cfg.get("high_fidelity_top_k", 24))
        seed = int(opt_cfg.get("seed", 42))
        campaign = str(eval_cfg.get("campaign", "")).lower()
        terms = dict(opt_cfg.get("objective_terms", {}))
        low_cfg = dict(eval_cfg)
        low_cfg["backend"] = "quasi1d"
        low_candidates = _uniform_candidates(bounds=bounds, n=low_n, seed=seed)
        _log(f"Optimization START  |  strategy=multifidelity_screen  low={len(low_candidates)}  high_top_k={high_k}", t0)

        scored: list[tuple[float, Dict[str, float]]] = []
        for i, p in enumerate(low_candidates):
            gid = f"screen_candidate_{i:05d}"
            geom = build_geometry_from_params(p, throat, n_points, profile, gid)
            r_low = CFDSimulation(geometry=geom, solverConfig=low_cfg, resultPath=str(out_dir)).extractResults()
            scored.append((_normalized_screen_score(r_low, terms=terms, campaign=campaign), p))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected_params = [dict(p) for _, p in scored[: max(1, min(high_k, len(scored)))]]

        high_results: list[EvaluationResult] = []
        for p in selected_params:
            high_results.append(objective(p))

        optimizer._history = high_results
        optimizer._history_params = selected_params
        runner.bestResult = optimizer._history[optimizer._best_idx()] if optimizer._history else None
        runner.saveHistory()
        _log(f"Optimization DONE   |  evaluated high-fidelity {len(optimizer._history)} candidates", t0)
    else:
        _log(f"Optimization START  |  {n_cands} candidates  backend={backend}", t0)
        runner.start()
        runner.saveHistory()
        _log(f"Optimization DONE   |  evaluated {len(optimizer._history)} candidates", t0)

    #  Multi-objective ranking (Pareto analysis on full history) 
    _log("Running Pareto / multi-objective ranking", t0)
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

    #  Select best candidate 
    selected_idx: int
    use_pareto_knee = bool(opt_cfg.get("use_pareto_knee", strategy == "multifidelity_screen"))
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
    _log("Benchmark START  |  evaluating MOC and optimized geometry", t0)
    bench_eval = make_evaluator(moc_geometry)
    suite = BenchmarkSuite(mocGeometry=moc_geometry, optimizedGeometry=optimized_geometry, evaluator=bench_eval)
    comparison: Dict[str, Any]
    benchmark_error = ""
    try:
        suite.runAll()
        if backend == "openfoam" and suite.mocResult is not None:
            case_dir = str((suite.mocResult.metadata or {}).get("case_dir", ""))
            detected = _detect_solver_from_case(case_dir)
            if detected != "unknown":
                solver_name = detected
                _write_run_meta(out_dir, config, backend=backend, solver=solver_name)
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
    if optimizer._history:
        thrust_vals = [float(r.thrust) for r in optimizer._history]
        loss_vals = [float(r.pressureLoss) for r in optimizer._history]
        note.addMetric("thrust_mean", sum(thrust_vals) / len(thrust_vals))
        note.addMetric("pressure_loss_mean", sum(loss_vals) / len(loss_vals))
        shock_x_vals = [float(r.shock.get("x")) for r in optimizer._history if isinstance(r.shock, dict) and r.shock.get("x") is not None]
        if shock_x_vals:
            sx_mean = sum(shock_x_vals) / len(shock_x_vals)
            sx_var = sum((x - sx_mean) * (x - sx_mean) for x in shock_x_vals) / len(shock_x_vals)
            note.addMetric("shock_x_mean", sx_mean)
            note.addMetric("shock_x_std", sx_var ** 0.5)
        wall_rms_vals = [float((r.metadata or {}).get("wall_pressure_rms")) for r in optimizer._history if (r.metadata or {}).get("wall_pressure_rms") is not None]
        if wall_rms_vals:
            note.addMetric("wall_pressure_rms", sum(wall_rms_vals) / len(wall_rms_vals))
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
        user_cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
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

    # Formatted results output
    cmp = summary.get("comparison", {})
    mo  = summary.get("multiobjective", {})
    sep = "-" * 52

    print()
    print(sep)
    print("  NOZZLE DESIGN BENCHMARK - RESULTS")
    print(sep)
    print(f"  Status          : {summary.get('status', '?').upper()}")
    print(f"  Elapsed         : {summary.get('elapsed_seconds', 0):.1f} s")
    print(f"  Output dir      : {summary.get('out_dir', '')}")
    print(sep)

    if cmp and cmp.get("status") != "failed":
        print("  PERFORMANCE COMPARISON")
        print(f"  {'':25s}  {'MOC':>10s}  {'OPT':>10s}  {'d':>10s}")
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
        if "theta_max" in p:
            print(f"  theta_max       : {p.get('theta_max', 0):.2f} deg")
            print(f"  inflection_frac : {p.get('inflection_frac', 0):.4f}")
            print(f"  throat_angle    : {p.get('throat_angle', 0):.2f} deg")
        elif "shape" in p:
            print(f"  shape           : {p.get('shape', 0):.3f}")
        pf = mo.get("pareto_front_size", "?")
        nc = mo.get("n_candidates", "?")
        print(f"  Pareto front    : {pf} / {nc} candidates")
        print(sep)

    print(f"  report.md  -> {summary.get('out_dir', '')}/report.md")
    print(sep)
    print()


if __name__ == "__main__":
    main()

