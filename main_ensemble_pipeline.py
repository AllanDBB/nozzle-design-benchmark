"""Multi-Algorithm Computational Intelligence Pipeline for Nozzle Design.

A "make it fly" ensemble pipeline that races four swarm-intelligence
algorithms against each other:

  - **PSO**       — Standard Particle Swarm Optimisation
  - **MOPSO-LF**  — Multi-Objective PSO with Lévy Flights (3 objectives)
  - **Firefly**   — Firefly Algorithm with Lévy perturbation
  - **ABC**       — Artificial Bee Colony with Pareto archive (3 objectives)

Multi-objective optimisation uses three objectives:
  1. Thrust      (maximise)
  2. Pressure loss (minimise)
  3. Nozzle length (minimise — shorter = lighter)

The pipeline expands the design space to 5D (exit_radius, length,
straighten_frac, ramp_power, throat_angle_deg) and produces rich
comparative visualisations.

Pipeline phases
---------------
Phase 1 — MOC Baseline       (2-D planar Prandtl-Meyer isentropic)
Phase 2 — Ensemble CI Race   (PSO + MOPSO-LF + Firefly + ABC, Q1D evaluator)
Phase 3 — RANS Validation    (OpenFOAM shockFluid, if available)
Phase 4 — Multi-fidelity Comparison Table
Phase 5 — Comparative Plots + MLN characteristic lines
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analysis import AnalysisNote, generate_comparison_plots
from analysis.SwarmPlots import generate_swarm_plots, plot_mln_with_characteristics
from analysis.EnsemblePlots import generate_ensemble_plots
from benchmarks import BenchmarkSuite
from evaluators import CFDSimulation, OpenFOAMRANSEvaluator, EvaluationResult
from geometry import MOCSolver, NozzleGeometry, MinimumLengthNozzle
from optimization import OptimizationRunner
from optimization.EnsembleRunner import EnsembleRunner


# ---------------------------------------------------------------------------
# Import run_multiobjective from scripts/ (no __init__.py there)
# ---------------------------------------------------------------------------
def _import_multiobjective():
    spec = importlib.util.spec_from_file_location(
        "multiobjective_rank",
        Path(__file__).parent / "scripts" / "multiobjective_rank.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_multiobjective


try:
    _run_multiobjective = _import_multiobjective()
except Exception:
    _run_multiobjective = None


# ---------------------------------------------------------------------------
# Default configuration — Ensemble CI
# ---------------------------------------------------------------------------
def default_config() -> Dict[str, Any]:
    return {
        "out_dir": "out/ensemble_pipeline",
        # ---- MOC baseline ----
        "moc": {
            "mach_exit": 2.3,
            "pressure_ratio": 0.08,
            "gamma": 1.4,
            "geometry": {
                "throat_y": 0.02,
                "exit_y": 0.06,
                "length": 0.22,
                "n_points": 180,
            },
        },
        # ---- Evaluator ----
        "evaluator": {
            "backend": "quasi1d",
            "campaign": "design_supersonic",
            "gamma": 1.4,
            "gas_constant": 287.0,
            "stagnation_temperature": 900.0,
            "stagnation_pressure": 3.0e5,
            "ambient_pressure": 2.4e4,
            "n_samples": 90,
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
            # OpenFOAM RANS options
            "fallback_on_failure": True,
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
            "p_initial": 2.7e5,
            "u_initial": 1.0,
            "t_initial": 900.0,
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
        # ---- Ensemble CI Optimisation (5D design space) ----
        "optimization": {
            "algorithms": ["pso", "mopso_lf", "firefly", "abc"],
            "seed": 42,
            # 5D design space
            "bounds": {
                "exit_radius": [0.050, 0.070],
                "length": [0.18, 0.28],
                "straighten_frac": [0.30, 0.70],
                "ramp_power": [0.40, 1.20],
                "throat_angle_deg": [1.0, 8.0],
            },
            # -- Shared hyper-parameters --
            "swarm_size": 30,
            "iterations": 15,
            "n_points": 180,
            "profile": "mln",
            # PSO-specific
            "w_max": 0.9,
            "w_min": 0.4,
            "c1": 2.0,
            "c2": 2.0,
            "v_max_frac": 0.25,
            # MOPSO-specific
            "archive_size": 100,
            "levy_prob": 0.15,
            "levy_scale": 0.01,
            "turbulence_k": 5,
            # Firefly-specific
            "beta0": 1.0,
            "gamma_fa": 1.0,
            "alpha_fa": 0.25,
            "chaos_k": 5,
            # ABC-specific
            "colony_size": 30,
            "limit": 0,
            "onlooker_ratio": 1.0,
            # Multi-objective weights
            "w_thrust": 0.7,
            "w_pressure_loss": 0.3,
            # Parallel workers
            "n_workers": 1,
            # RANS validation top-K
            "rans_top_k": 5,
            # Penalty terms
            "objective_terms": {
                "design_shock_penalty": 0.10,
                "overexpanded_instability_penalty": 0.05,
                "overexpanded_outlet_osc_penalty": 0.05,
                "overexpanded_wall_rms_penalty": 0.03,
            },
            # MOC characteristic lines on nozzle plots
            "n_char_lines": 80,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_geometry_from_params(
    params: Dict[str, float],
    throat_radius: float,
    n_points: int,
    mach_exit: float,
    gamma: float,
    gid: str,
) -> NozzleGeometry:
    """Build a NozzleGeometry from the 5D parameter dict."""
    mln = MinimumLengthNozzle(
        mach_exit=mach_exit,
        throat_radius=throat_radius,
        exit_radius=float(params["exit_radius"]),
        length=float(params["length"]),
        gamma=gamma,
        n_points=n_points,
        straighten_frac=float(params.get("straighten_frac", 0.45)),
        ramp_power=float(params.get("ramp_power", 0.7)),
        throat_angle_deg=float(params.get("throat_angle_deg", 4.0)),
    )
    geom = mln.build()
    geom.metadata["id"] = gid
    geom.metadata["source"] = "Ensemble-CI"
    return geom


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
        "optimizer": "Ensemble CI (PSO + MOPSO-LF + Firefly)",
        "config_effective": config,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _openfoam_available() -> bool:
    """Check if OpenFOAM is reachable — natively or via Docker."""
    try:
        r = subprocess.run(
            ["bash", "-lc",
             "source /opt/openfoam13/etc/bashrc 2>/dev/null; command -v blockMesh"],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return True
    except Exception:
        pass
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _compute_performance_metrics(
    result: EvaluationResult,
    geometry: NozzleGeometry,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    gamma = float(cfg.get("gamma", 1.4))
    gas_r = float(cfg.get("gas_constant", 287.0))
    p0 = float(cfg.get("stagnation_pressure", 3.0e5))
    t0 = float(cfg.get("stagnation_temperature", 900.0))
    depth = float(cfg.get("depth", 0.02))
    dim = str(cfg.get("dimension", "2d_planar"))
    cd = float(cfg.get("discharge_coefficient", 0.985))
    g0 = 9.80665

    throat_y = geometry.throat_radius
    if dim in ("2d_planar", "3d_channel"):
        a_throat = 2.0 * throat_y * depth
    else:
        a_throat = math.pi * throat_y ** 2

    mdot = (
        cd * a_throat * p0
        / math.sqrt(max(t0, 1e-6))
        * math.sqrt(gamma / gas_r)
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )

    thrust = float(result.thrust)
    c_f = thrust / max(p0 * a_throat, 1e-12)
    isp = thrust / max(mdot * g0, 1e-12)
    exit_mach = result.machProfile[-1] if result.machProfile else 0.0
    wall_angles = geometry.wall_angles()
    exit_angle_deg = math.degrees(abs(wall_angles[-1])) if wall_angles else 0.0
    eta_div = float(result.metadata.get("eta_div", 1.0))
    pressure_recovery = 1.0 - float(result.pressureLoss)

    return {
        "thrust_N": round(thrust, 4),
        "C_F": round(c_f, 5),
        "Isp_s": round(isp, 2),
        "eta_div": round(eta_div, 5),
        "exit_mach": round(exit_mach, 4),
        "exit_angle_deg": round(exit_angle_deg, 3),
        "pressure_recovery": round(pressure_recovery, 5),
        "pressure_loss": round(float(result.pressureLoss), 5),
        "mdot_kg_s": round(mdot, 6),
    }


def _fidelity_gap(q1d: Dict[str, Any], rans: Dict[str, Any]) -> Dict[str, Any]:
    gap: Dict[str, Any] = {}
    for key in ("thrust_N", "C_F", "Isp_s", "exit_mach", "pressure_loss"):
        vq = float(q1d.get(key, 0))
        vr = float(rans.get(key, 0))
        ref = max(abs(vr), abs(vq), 1e-12)
        gap[f"delta_{key}"] = round(vr - vq, 6)
        gap[f"delta_{key}_pct"] = round(100.0 * (vr - vq) / ref, 3)
    return gap


# ---------------------------------------------------------------------------
# 5-Phase Ensemble CI Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    out_dir = Path(config["out_dir"])
    moc_dir = out_dir / "moc"
    opt_dir = out_dir / "ensemble"
    rans_dir = out_dir / "rans"
    comp_dir = out_dir / "comp"
    for d in [out_dir, moc_dir, opt_dir, rans_dir, comp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    moc_cfg = config["moc"]
    eval_cfg = dict(config["evaluator"])
    opt_cfg = config["optimization"]
    backend = str(eval_cfg.get("backend", "quasi1d")).lower()

    algorithms = opt_cfg.get("algorithms", ["pso", "mopso_lf", "firefly"])
    swarm_size = int(opt_cfg.get("swarm_size", 30))
    iterations = int(opt_cfg.get("iterations", 15))
    n_char = int(opt_cfg.get("n_char_lines", 80))
    total_evals_per_algo = swarm_size * (iterations + 1)
    total_evals = total_evals_per_algo * len(algorithms)
    n_dims = len(opt_cfg.get("bounds", {}))

    _log(
        f"Ensemble CI Pipeline START  |  backend={backend}  "
        f"algorithms={algorithms}  swarm={swarm_size}  iter={iterations}  "
        f"~{total_evals} evals ({n_dims}D)  char_lines={n_char}",
        t0,
    )
    _log(f"Output -> {out_dir.resolve()}", t0)
    _write_run_meta(out_dir, config, backend=backend)

    eval_cfg["design_pressure_ratio"] = float(moc_cfg.get("pressure_ratio", 0.10))
    throat = float(moc_cfg["geometry"]["throat_y"])
    n_pts = int(opt_cfg.get("n_points", 180))
    mach_e = float(moc_cfg.get("mach_exit", 2.3))
    gamma = float(moc_cfg.get("gamma", 1.4))
    p_ratio = float(moc_cfg.get("pressure_ratio", 0.08))

    q1d_cfg = dict(eval_cfg)
    q1d_cfg["backend"] = "quasi1d"

    rans_cfg = {
        **eval_cfg,
        "backend": "openfoam",
        "mach_exit": mach_e,
        "pressure_ratio": p_ratio,
    }

    run_rans = backend == "openfoam"
    if run_rans and not _openfoam_available():
        _log("WARNING: backend=openfoam but OpenFOAM not available — RANS disabled", t0)
        run_rans = False

    # ==================================================================
    # PHASE 1 — MOC Baseline
    # ==================================================================
    _log("=" * 60, t0)
    _log("PHASE 1  |  MOC Baseline (2-D planar Prandtl-Meyer)", t0)
    _log("=" * 60, t0)

    moc_solver = MOCSolver(
        machExit=mach_e,
        pressureRatio=p_ratio,
        gamma=gamma,
    )
    moc_geometry = moc_solver.generateGeometry({**moc_cfg["geometry"]})
    moc_geometry.metadata["id"] = "moc_baseline"
    _log(
        f"  MOC geometry  |  Me={mach_e}  "
        f"throat={throat*1000:.1f} mm  "
        f"exit={float(moc_cfg['geometry']['exit_y'])*1000:.1f} mm  "
        f"L={float(moc_cfg['geometry']['length'])*1000:.1f} mm",
        t0,
    )

    moc_geometry.exportGeo(str(moc_dir / "moc_geometry.csv"))
    moc_geometry.plotProfile(str(moc_dir / "moc_geometry.png"))

    # MLN auxiliary plot
    _mln = MinimumLengthNozzle.from_params(
        moc_cfg["geometry"], mach_exit=mach_e, gamma=gamma
    )
    _mln.plot(str(moc_dir / "mln_geometry.png"))

    # Characteristic network
    plot_mln_with_characteristics(
        moc_geometry,
        mach_exit=mach_e,
        pressure_ratio=p_ratio,
        gamma=gamma,
        savepath=str(moc_dir / "moc_characteristics.png"),
        n_char_lines=n_char,
    )

    # Q1D evaluation of baseline
    moc_q1d_eval = CFDSimulation(
        geometry=moc_geometry, solverConfig=q1d_cfg, resultPath=str(out_dir)
    )
    moc_q1d_result = moc_q1d_eval.extractResults()
    moc_q1d_result.geometryId = "moc_baseline_q1d"
    moc_q1d_result.saveToJSON(str(moc_dir / "moc_q1d_result.json"))
    moc_q1d_metrics = _compute_performance_metrics(
        moc_q1d_result, moc_geometry, q1d_cfg
    )
    _log(
        f"  MOC Q1D  |  F={moc_q1d_metrics['thrust_N']:.3f} N  "
        f"C_F={moc_q1d_metrics['C_F']:.4f}  "
        f"Isp={moc_q1d_metrics['Isp_s']:.1f} s  "
        f"Me={moc_q1d_metrics['exit_mach']:.3f}",
        t0,
    )

    # ==================================================================
    # PHASE 2 — Ensemble CI Race
    # ==================================================================
    _log("=" * 60, t0)
    _log("PHASE 2  |  Ensemble CI Race (" + " + ".join(a.upper() for a in algorithms) + ")", t0)
    _log("=" * 60, t0)
    _log(
        f"  algorithms={algorithms}  swarm_size={swarm_size}  iterations={iterations}  "
        f"design_space={n_dims}D  ~{total_evals} total evals",
        t0,
    )

    _gid_counter = 0
    _gid_lock = threading.Lock()

    def _next_gid() -> str:
        nonlocal _gid_counter
        with _gid_lock:
            idx = _gid_counter
            _gid_counter += 1
        return f"ci_{idx:04d}"

    def objective(param_dict: Dict[str, float]) -> EvaluationResult:
        gid = _next_gid()
        geom = build_geometry_from_params(param_dict, throat, n_pts, mach_e, gamma, gid)
        local_eval = CFDSimulation(
            geometry=geom, solverConfig=q1d_cfg, resultPath=str(out_dir)
        )
        t_start = time.time()
        try:
            result = local_eval.extractResults()
            result.geometryId = gid
            result.metadata["params"] = dict(param_dict)
            dt_s = time.time() - t_start
            _log(
                f"  {gid}  F={result.thrust:8.2f} N  "
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
        k: v for k, v in opt_cfg.items()
        if k not in ("algorithms", "seed")
    }
    search_space["campaign"] = str(eval_cfg.get("campaign", "")).lower()

    ensemble = EnsembleRunner(
        searchSpace=search_space,
        objectiveFunc=objective,
        algorithms=algorithms,
        seed=int(opt_cfg.get("seed", 42)),
    )

    best_result = ensemble.run()
    ensemble.print_summary()
    ensemble_summary = ensemble.summary()

    (opt_dir / "ensemble_summary.json").write_text(
        json.dumps(ensemble_summary, indent=2), encoding="utf-8"
    )

    _log(f"  Ensemble DONE  |  {ensemble.total_evals} total evaluations", t0)
    _log(f"  Winner: {ensemble.best_algorithm}  F={best_result.thrust:.3f} N", t0)

    # ---- Pareto analysis on merged history ----
    _log("  Running Pareto / multi-objective ranking ...", t0)
    mo_summary: Dict[str, Any] = {}
    if _run_multiobjective is not None and len(ensemble._history) > 1:
        history_records: List[Dict[str, Any]] = []
        for r, p in zip(ensemble._history, ensemble._history_params):
            rec = r.to_dict()
            rec["params"] = dict(p)
            history_records.append(rec)
        try:
            mo_summary = _run_multiobjective(
                records=history_records,
                out_dir=str(opt_dir / "multiobjective"),
                w_thrust=float(opt_cfg.get("w_thrust", 0.7)),
                w_pressure_loss=float(opt_cfg.get("w_pressure_loss", 0.3)),
                moc_thrust=float(moc_q1d_result.thrust),
                moc_loss=float(moc_q1d_result.pressureLoss),
            )
        except Exception as exc:
            mo_summary = {"error": str(exc)}

    # ---- Select best candidate ----
    best_params = dict(ensemble.best_params)
    _log(
        f"  Best CI  |  F={best_result.thrust:.3f} N  "
        f"loss={best_result.pressureLoss:.5f}  algo={ensemble.best_algorithm}  "
        f"params={best_params}",
        t0,
    )

    optimized_geometry = build_geometry_from_params(
        best_params, throat, n_pts, mach_e, gamma, "ci_best"
    )
    optimized_geometry.exportGeo(str(opt_dir / "ci_best_geometry.csv"))
    optimized_geometry.plotProfile(str(opt_dir / "ci_best_geometry.png"))

    # MLN plot for the best geometry
    _opt_mln = MinimumLengthNozzle(
        mach_exit=mach_e,
        throat_radius=throat,
        exit_radius=float(best_params["exit_radius"]),
        length=float(best_params["length"]),
        gamma=gamma,
        n_points=n_pts,
        straighten_frac=float(best_params.get("straighten_frac", 0.45)),
        ramp_power=float(best_params.get("ramp_power", 0.7)),
        throat_angle_deg=float(best_params.get("throat_angle_deg", 4.0)),
    )
    _opt_mln.plot(str(opt_dir / "ci_best_mln_geometry.png"))

    # MOC characteristic network for the CI-best nozzle
    opt_moc_solver = MOCSolver(
        machExit=mach_e, pressureRatio=p_ratio, gamma=gamma,
    )
    opt_moc_geom = opt_moc_solver.generateGeometry({
        "throat_y": throat,
        "exit_y": float(best_params["exit_radius"]),
        "length": float(best_params["length"]),
        "n_points": n_pts,
    })
    plot_mln_with_characteristics(
        opt_moc_geom,
        mach_exit=mach_e,
        pressure_ratio=p_ratio,
        gamma=gamma,
        savepath=str(opt_dir / "ci_best_characteristics.png"),
        n_char_lines=n_char,
    )

    opt_q1d_metrics = _compute_performance_metrics(
        best_result, optimized_geometry, q1d_cfg
    )

    # ==================================================================
    # PHASE 3 — RANS Validation
    # ==================================================================
    _log("=" * 60, t0)
    moc_rans_result: Optional[EvaluationResult] = None
    moc_rans_metrics: Optional[Dict[str, Any]] = None
    opt_rans_result: Optional[EvaluationResult] = None
    opt_rans_metrics: Optional[Dict[str, Any]] = None
    rans_candidates: Dict[str, Any] = {}

    if run_rans:
        _log("PHASE 3  |  RANS Validation (OpenFOAM shockFluid)", t0)
        _log("=" * 60, t0)

        # 3a) MOC baseline RANS
        _log("  3a) RANS on MOC baseline geometry ...", t0)
        try:
            moc_rans_eval = OpenFOAMRANSEvaluator(
                geometry=moc_geometry, solverConfig=rans_cfg, resultPath=str(rans_dir),
            )
            moc_rans_result = moc_rans_eval.extractResults()
            moc_rans_result.geometryId = "moc_baseline_rans"
            moc_rans_result.saveToJSON(str(rans_dir / "moc_rans_result.json"))
            moc_rans_metrics = _compute_performance_metrics(
                moc_rans_result, moc_geometry, eval_cfg
            )
            _log(
                f"  MOC RANS  |  F={moc_rans_metrics['thrust_N']:.3f} N  "
                f"C_F={moc_rans_metrics['C_F']:.4f}  "
                f"Me={moc_rans_metrics['exit_mach']:.3f}",
                t0,
            )
        except Exception as exc:
            _log(f"  MOC RANS FAILED: {exc}", t0)

        # 3b) CI-best RANS
        _log("  3b) RANS on CI-best candidate ...", t0)
        try:
            opt_rans_eval = OpenFOAMRANSEvaluator(
                geometry=optimized_geometry, solverConfig=rans_cfg,
                resultPath=str(rans_dir),
            )
            opt_rans_result = opt_rans_eval.extractResults()
            opt_rans_result.geometryId = "ci_best_rans"
            opt_rans_result.saveToJSON(str(rans_dir / "ci_rans_result.json"))
            opt_rans_metrics = _compute_performance_metrics(
                opt_rans_result, optimized_geometry, eval_cfg
            )
            _log(
                f"  CI RANS  |  F={opt_rans_metrics['thrust_N']:.3f} N  "
                f"C_F={opt_rans_metrics['C_F']:.4f}  "
                f"Me={opt_rans_metrics['exit_mach']:.3f}",
                t0,
            )
        except Exception as exc:
            _log(f"  CI RANS FAILED: {exc}", t0)

        # 3c) Pareto-informed RANS candidates
        #     Priority: Pareto knee > best-by-loss > top-by-thrust
        rans_top_k = int(opt_cfg.get("rans_top_k", 5))

        # Build full scored list
        all_scored = []
        for ar in ensemble.algo_results:
            for i, (r, p) in enumerate(zip(ar.history, ar.history_params)):
                all_scored.append((float(r.thrust), r, p, ar.algorithm, i))

        extra_candidates: List[Tuple[Any, ...]] = []
        seen_params = {str(best_params)}

        def _try_add(result, params, algo_name, idx_in_algo, tag=""):
            key = str(params)
            if key not in seen_params and len(extra_candidates) < max(0, rans_top_k - 1):
                extra_candidates.append((result, params, algo_name, idx_in_algo, tag))
                seen_params.add(key)

        # (i) Pareto knee from multi-objective analysis
        knee_info = mo_summary.get("best_pareto_knee", {}) if mo_summary else {}
        if knee_info and knee_info.get("params"):
            knee_idx = int(knee_info.get("candidate_index", -1))
            if 0 <= knee_idx < len(ensemble._history):
                _try_add(
                    ensemble._history[knee_idx],
                    ensemble._history_params[knee_idx],
                    "pareto_knee", knee_idx, "knee",
                )

        # (ii) Best by pressure loss
        loss_sorted = sorted(all_scored, key=lambda x: float(x[1].pressureLoss))
        if loss_sorted:
            _, r_bl, p_bl, a_bl, i_bl = loss_sorted[0]
            _try_add(r_bl, p_bl, a_bl, i_bl, "best_loss")

        # (iii) Remaining slots filled by top thrust
        all_scored.sort(key=lambda x: x[0], reverse=True)
        for _, r, p, algo, idx in all_scored:
            _try_add(r, p, algo, idx, "top_thrust")

        if extra_candidates:
            _log(f"  3c) RANS on {len(extra_candidates)} Pareto-informed candidates ...", t0)
            for entry in extra_candidates:
                r, p, algo, idx = entry[0], entry[1], entry[2], entry[3]
                tag = entry[4] if len(entry) > 4 else ""
                gid = f"rans_{algo}_{idx:04d}"
                if tag:
                    gid = f"rans_{tag}_{algo}_{idx:04d}"
                geom = build_geometry_from_params(p, throat, n_pts, mach_e, gamma, gid)
                _log(f"    RANS {gid} ...", t0)
                try:
                    e = OpenFOAMRANSEvaluator(
                        geometry=geom, solverConfig=rans_cfg, resultPath=str(rans_dir),
                    )
                    rr = e.extractResults()
                    rr.geometryId = gid
                    rr.saveToJSON(str(rans_dir / f"{gid}_result.json"))
                    rm = _compute_performance_metrics(rr, geom, eval_cfg)
                    rans_candidates[gid] = {
                        "algorithm": algo,
                        "status": "ok",
                        "params": dict(p),
                        **rm,
                    }
                    _log(f"    {gid}  F={rm['thrust_N']:.3f} N  [OK]", t0)
                except Exception as exc:
                    rans_candidates[gid] = {
                        "algorithm": algo,
                        "status": "failed",
                        "error": str(exc),
                        "params": dict(p),
                    }
                    _log(f"    {gid}  FAILED: {exc}", t0)

        (rans_dir / "rans_candidates.json").write_text(
            json.dumps(rans_candidates, indent=2), encoding="utf-8"
        )
    else:
        _log(
            "PHASE 3  |  RANS Validation  [skipped: no Docker / backend=quasi1d]",
            t0,
        )
        _log("=" * 60, t0)

    # ==================================================================
    # PHASE 4 — Multi-fidelity Comparison Table
    # ==================================================================
    _log("=" * 60, t0)
    _log("PHASE 4  |  Multi-fidelity Comparison", t0)
    _log("=" * 60, t0)

    fidelity_table: Dict[str, Any] = {
        "moc_q1d": moc_q1d_metrics,
        "ci_best_q1d": opt_q1d_metrics,
        "ci_winner_algorithm": ensemble.best_algorithm,
    }
    if moc_rans_metrics:
        fidelity_table["moc_rans"] = moc_rans_metrics
        fidelity_table["moc_fidelity_gap"] = _fidelity_gap(moc_q1d_metrics, moc_rans_metrics)
    if opt_rans_metrics:
        fidelity_table["ci_rans"] = opt_rans_metrics
        fidelity_table["ci_fidelity_gap"] = _fidelity_gap(opt_q1d_metrics, opt_rans_metrics)

    (comp_dir / "fidelity_table.json").write_text(
        json.dumps(fidelity_table, indent=2), encoding="utf-8"
    )

    # Print table
    _log("", t0)
    hdr = f"  {'Metric':<22s}  {'MOC-Q1D':>10s}  {'CI-Q1D':>10s}"
    sep_cols = 4
    if moc_rans_metrics:
        hdr += f"  {'MOC-RANS':>10s}"
        sep_cols += 1
    if opt_rans_metrics:
        hdr += f"  {'CI-RANS':>10s}"
        sep_cols += 1
    table_sep = "  " + "-" * (sep_cols * 12 + 22)
    _log(table_sep, t0)
    _log(hdr, t0)
    _log(table_sep, t0)

    def _fc(val: float, fmt: str) -> str:
        return format(val, fmt).rjust(10)

    for key, label, fmt in [
        ("thrust_N", "Thrust [N]", ".3f"),
        ("C_F", "C_F [-]", ".4f"),
        ("Isp_s", "Isp [s]", ".1f"),
        ("exit_mach", "Mach exit [-]", ".4f"),
        ("pressure_loss", "Pressure loss [-]", ".5f"),
        ("pressure_recovery", "p_t/p_0 [-]", ".5f"),
        ("eta_div", "eta_div [-]", ".4f"),
        ("exit_angle_deg", "Exit angle [deg]", ".2f"),
    ]:
        row = f"  {label:<22s}"
        row += f"  {_fc(moc_q1d_metrics.get(key, 0), fmt)}"
        row += f"  {_fc(opt_q1d_metrics.get(key, 0), fmt)}"
        if moc_rans_metrics:
            row += f"  {_fc(moc_rans_metrics.get(key, 0), fmt)}"
        if opt_rans_metrics:
            row += f"  {_fc(opt_rans_metrics.get(key, 0), fmt)}"
        _log(row, t0)
    _log(table_sep, t0)
    _log(f"  Winner algorithm: {ensemble.best_algorithm.upper()}", t0)

    if fidelity_table.get("moc_fidelity_gap"):
        fg = fidelity_table["moc_fidelity_gap"]
        _log(
            f"  MOC fidelity gap:  dF={fg['delta_thrust_N_pct']:+.2f}%  "
            f"dMe={fg['delta_exit_mach_pct']:+.2f}%",
            t0,
        )
    if fidelity_table.get("ci_fidelity_gap"):
        fg = fidelity_table["ci_fidelity_gap"]
        _log(
            f"  CI fidelity gap:  dF={fg['delta_thrust_N_pct']:+.2f}%  "
            f"dMe={fg['delta_exit_mach_pct']:+.2f}%",
            t0,
        )
    _log("", t0)

    # ==================================================================
    # PHASE 5 — Comparative Plots + Benchmark
    # ==================================================================
    _log("=" * 60, t0)
    _log("PHASE 5  |  Comparative Plots & Benchmark", t0)
    _log("=" * 60, t0)

    final_moc_result = moc_rans_result if moc_rans_result else moc_q1d_result
    final_opt_result = opt_rans_result if opt_rans_result else best_result

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

        # --- Per-algorithm SI diagnostic plots (winner) ---
        winner_ar = None
        for ar in ensemble.algo_results:
            if ar.algorithm == ensemble.best_algorithm:
                winner_ar = ar
                break

        if winner_ar is not None:
            # Create a lightweight proxy object for generate_swarm_plots
            class _OptimizerProxy:
                def __init__(self, ar, ss):
                    self._gbest_trace = ar.gbest_trace
                    self._inertia_trace = ar.inertia_trace
                    self._diversity_trace = ar.diversity_trace
                    self._particle_traces = ar.particle_traces
                    self.searchSpace = ss

            proxy = _OptimizerProxy(winner_ar, search_space)
            generate_swarm_plots(
                optimizer=proxy,
                history=winner_ar.history,
                out_dir=str(opt_dir / winner_ar.algorithm),
                swarm_size=swarm_size,
                moc_thrust=float(moc_q1d_result.thrust),
                moc_pressure_loss=float(moc_q1d_result.pressureLoss),
                moc_geometry=moc_geometry,
                throat_radius=throat,
                mach_exit=mach_e,
                gamma=gamma,
                pressure_ratio=p_ratio,
                n_points=n_pts,
                n_char_lines=n_char,
            )

        # --- Ensemble comparative plots ---
        bounds = opt_cfg.get("bounds", {})
        bounds_tuples = {k: tuple(v) for k, v in bounds.items()}

        n_ensemble_plots = generate_ensemble_plots(
            algo_results=ensemble.algo_results,
            out_dir=str(opt_dir),
            bounds=bounds_tuples,
            moc_thrust=float(moc_q1d_result.thrust),
            moc_loss=float(moc_q1d_result.pressureLoss),
        )
        _log(f"  Generated {n_ensemble_plots} ensemble comparative plots", t0)

        # Comparison plots (MOC vs CI-best)
        if suite.mocResult is not None and suite.optResult is not None:
            generate_comparison_plots(
                moc_geometry=moc_geometry,
                optimized_geometry=optimized_geometry,
                moc_result=suite.mocResult,
                optimized_result=suite.optResult,
                out_dir=str(comp_dir),
                solver_config=eval_cfg,
                mach_exit=mach_e,
            )

        # Field plots
        bench_eval.geometry = moc_geometry
        bench_eval.extractResults()
        bench_eval.plotFields(str(moc_dir / "moc_fields.png"))

        bench_eval.geometry = optimized_geometry
        bench_eval.extractResults()
        bench_eval.plotFields(str(opt_dir / "ci_best_fields.png"))

    except Exception as exc:
        benchmark_error = str(exc)
        comparison = {"status": "failed", "error": benchmark_error}
        (comp_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )

    # ---- Analysis report ----
    note = AnalysisNote(
        title="Nozzle Design Benchmark — Ensemble CI (PSO + MOPSO-LF + Firefly + ABC)",
        notes=(
            "5-phase ensemble pipeline: (1) MOC baseline via Prandtl-Meyer isentropic "
            "characteristic mesh, (2) Multi-algorithm CI race — PSO, MOPSO with Lévy "
            "Flights, Firefly Algorithm, and Artificial Bee Colony — all searching a 5D "
            "design space (exit_radius, length, straighten_frac, ramp_power, "
            "throat_angle_deg) with 3-objective Pareto (thrust, pressure loss, length), "
            "(3) RANS validation of CI-best, Pareto knee, best-by-loss, and top-K via "
            "OpenFOAM shockFluid, (4) multi-fidelity comparison table, (5) ensemble "
            "comparative plots including algorithm race, Pareto overlay, diversity "
            "comparison, search-space heatmaps, box plots, and radar chart.  "
            "The pipeline picks the overall winner via 3-objective Pareto dominance."
        ),
    )
    for k, v in comparison.items():
        note.addMetric(k, v)

    note.addMetric("optimizer_algorithm", f"Ensemble CI ({', '.join(algorithms)})")
    note.addMetric("optimizer_total_evaluations", ensemble.total_evals)
    note.addMetric("ensemble_summary", ensemble_summary)
    note.addMetric("winner_algorithm", ensemble.best_algorithm)
    note.addMetric("design_space_dimensions", n_dims)
    note.addMetric("best_parameters", best_params)
    note.addMetric("n_characteristic_lines", n_char)

    note.addMetric("moc_q1d_performance", moc_q1d_metrics)
    note.addMetric("ci_q1d_performance", opt_q1d_metrics)
    if moc_rans_metrics:
        note.addMetric("moc_rans_performance", moc_rans_metrics)
    if opt_rans_metrics:
        note.addMetric("ci_rans_performance", opt_rans_metrics)
    note.addMetric("fidelity_table", fidelity_table)

    if ensemble._history:
        thrust_vals = [float(r.thrust) for r in ensemble._history]
        loss_vals = [float(r.pressureLoss) for r in ensemble._history]
        note.addMetric("thrust_mean", sum(thrust_vals) / len(thrust_vals))
        note.addMetric(
            "thrust_std",
            (sum((t - sum(thrust_vals) / len(thrust_vals)) ** 2 for t in thrust_vals)
             / len(thrust_vals)) ** 0.5,
        )
        note.addMetric("pressure_loss_mean", sum(loss_vals) / len(loss_vals))
    if mo_summary:
        note.addMetric("multiobjective_summary", mo_summary)
    if rans_candidates:
        note.addMetric("rans_candidates", rans_candidates)

    note.save(str(out_dir / "report.md"))
    note.saveJSON(str(out_dir / "report.json"))

    elapsed_total = time.time() - t0
    summary = {
        "status": "ok" if not benchmark_error else "failed",
        "out_dir": str(out_dir),
        "moc_geometry": str(moc_dir / "moc_geometry.csv"),
        "ci_geometry": str(opt_dir / "ci_best_geometry.csv"),
        "comparison": comparison,
        "ensemble": ensemble_summary,
        "multiobjective": mo_summary,
        "fidelity_table": fidelity_table,
        "rans_candidates": rans_candidates,
        "rans_available": run_rans,
        "elapsed_seconds": round(elapsed_total, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _log(
        f"Ensemble CI Pipeline DONE  |  status={summary['status']}  "
        f"elapsed={elapsed_total:.1f}s  rans={'yes' if run_rans else 'no'}",
        t0,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble CI nozzle design: MOC -> PSO+MOPSO+Firefly+ABC -> RANS -> Compare"
    )
    parser.add_argument(
        "--config", type=str, default="",
        help="JSON config file (overrides defaults)",
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Override output directory",
    )
    parser.add_argument(
        "--backend", type=str, default="",
        help="Override evaluator backend (quasi1d|openfoam)",
    )
    parser.add_argument(
        "--swarm-size", type=int, default=0,
        help="Override swarm size (per algorithm)",
    )
    parser.add_argument(
        "--iterations", type=int, default=0,
        help="Override iterations (per algorithm)",
    )
    parser.add_argument(
        "--algorithms", type=str, default="",
        help="Comma-separated algorithms to run (pso,mopso_lf,firefly)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: small swarm (size=6 iter=2) for fast smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    if args.config:
        user_cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        for key, value in user_cfg.items():
            if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    if args.out:
        cfg["out_dir"] = args.out
    if args.backend:
        cfg["evaluator"]["backend"] = args.backend
    if args.swarm_size > 0:
        cfg["optimization"]["swarm_size"] = args.swarm_size
    if args.iterations > 0:
        cfg["optimization"]["iterations"] = args.iterations
    if args.algorithms:
        cfg["optimization"]["algorithms"] = [
            a.strip() for a in args.algorithms.split(",") if a.strip()
        ]
    if args.quick:
        cfg["optimization"]["swarm_size"] = 6
        cfg["optimization"]["iterations"] = 2
        cfg["optimization"]["rans_top_k"] = 2

    summary = run_pipeline(cfg)

    # ---------- Formatted results ----------
    cmp = summary.get("comparison", {})
    ens = summary.get("ensemble", {})
    ftbl = summary.get("fidelity_table", {})
    sep = "-" * 66

    print()
    print(sep)
    print("  NOZZLE DESIGN - ENSEMBLE COMPUTATIONAL INTELLIGENCE")
    print(sep)
    print(f"  Status       : {summary.get('status', '?').upper()}")
    print(f"  Elapsed      : {summary.get('elapsed_seconds', 0):.1f} s")
    print(f"  RANS enabled : {'yes' if summary.get('rans_available') else 'no'}")
    print(f"  Algorithms   : {', '.join(ens.get('algorithms_run', []))}")
    print(f"  Winner       : {ens.get('winner', '?').upper()} *")
    print(f"  Total evals  : {ens.get('total_evaluations', 0)}")
    print(f"  Output dir   : {summary.get('out_dir', '')}")
    print(sep)

    # Per-algorithm results
    if ens.get("per_algorithm"):
        print("  ALGORITHM RESULTS")
        fmt_h = "  {:<14s}  {:>12s}  {:>10s}  {:>6s}  {:>7s}"
        print(fmt_h.format("Algorithm", "Thrust [N]", "Loss", "Evals", "Time"))
        for row in ens["per_algorithm"]:
            winner = " *" if row["algorithm"] == ens.get("winner") else ""
            print(
                f"  {row['algorithm']:<14s}  {row['thrust_N']:>12.4f}  "
                f"{row['pressure_loss']:>10.6f}  {row['n_evals']:>6d}  "
                f"{row['elapsed_s']:>6.1f}s{winner}"
            )
        print(sep)

    if cmp and cmp.get("status") != "failed":
        print("  Q1D COMPARISON  (MOC baseline vs CI-best)")
        moc_t = cmp.get("moc_thrust", 0)
        opt_t = cmp.get("optimized_thrust", 0)
        moc_pl = cmp.get("moc_pressure_loss", 0)
        opt_pl = cmp.get("optimized_pressure_loss", 0)
        d_t = cmp.get("delta_thrust", 0)
        d_t_p = cmp.get("delta_thrust_percent", 0)
        d_pl = cmp.get("delta_pressure_loss", 0)
        print(
            f"  {'Thrust [N]':<25s}  {moc_t:>10.3f}  {opt_t:>10.3f}"
            f"  {d_t:>+10.3f} ({d_t_p:+.2f}%)"
        )
        print(
            f"  {'Pressure loss [-]':<25s}  {moc_pl:>10.4f}  {opt_pl:>10.4f}"
            f"  {d_pl:>+10.4f}"
        )
        print(sep)

    if ftbl.get("moc_fidelity_gap") or ftbl.get("ci_fidelity_gap"):
        print("  MULTI-FIDELITY GAP  (Q1D vs RANS)")
        if ftbl.get("moc_fidelity_gap"):
            fg = ftbl["moc_fidelity_gap"]
            print(
                f"  MOC:  dThrust={fg.get('delta_thrust_N_pct', 0):+.2f}%  "
                f"dMach={fg.get('delta_exit_mach_pct', 0):+.2f}%"
            )
        if ftbl.get("ci_fidelity_gap"):
            fg = ftbl["ci_fidelity_gap"]
            print(
                f"  CI:   dThrust={fg.get('delta_thrust_N_pct', 0):+.2f}%  "
                f"dMach={fg.get('delta_exit_mach_pct', 0):+.2f}%"
            )
        print(sep)

    # Best params
    if ens.get("per_algorithm"):
        winner_row = next(
            (r for r in ens["per_algorithm"] if r["algorithm"] == ens.get("winner")),
            None
        )
        if winner_row and winner_row.get("params"):
            p = winner_row["params"]
            print("  BEST DESIGN PARAMETERS")
            for k, v in p.items():
                if isinstance(v, float):
                    print(f"    {k:<22s}: {v:.6f}")
                else:
                    print(f"    {k:<22s}: {v}")
            print(sep)

    print(f"  report -> {summary.get('out_dir', '')}/report.md")
    print(sep)
    print()


if __name__ == "__main__":
    main()
