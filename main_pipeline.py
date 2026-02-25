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
from typing import Dict, Any, List, Optional, Tuple

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
# Default configuration -- production-sized GA, real RANS integration
# ---------------------------------------------------------------------------
def default_config() -> Dict[str, Any]:
    return {
        "out_dir": "out/pipeline",
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
            # Quasi-1D loss model
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
        # ---- Optimization ----
        "optimization": {
            "algorithm": "evolutionary",
            "seed": 42,
            "bounds": {
                "exit_radius": [0.050, 0.070],
                "length": [0.18, 0.28],
                "shape": [1.3, 2.2],
            },
            # -- GA hyper-parameters (production) --
            "population": 20,
            "generations": 8,
            "sigma": 0.10,
            "n_points": 180,
            "profile": "bezier_like",
            # Multi-objective weights
            "w_thrust": 0.7,
            "w_pressure_loss": 0.3,
            # Parallel workers for batch evaluation
            "n_workers": 1,
            # RANS validation: top Pareto candidates to validate
            "rans_top_k": 5,
            # Penalty terms
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


def _docker_available() -> bool:
    """Check if Docker daemon is reachable (needed for OpenFOAM RANS)."""
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Performance metric helpers (paper-quality derived quantities)
# ---------------------------------------------------------------------------
def _compute_performance_metrics(
    result: EvaluationResult,
    geometry: NozzleGeometry,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive paper-grade metrics from an EvaluationResult.

    Returns dict with:
      - C_F          : thrust coefficient  F / (p0 * A_throat)
      - Isp          : specific impulse  F / (mdot * g0)   [s]
      - eta_div      : divergence efficiency
      - exit_mach    : Mach at last profile station
      - exit_angle_deg : wall half-angle at exit [deg]
      - pressure_recovery : p_t_exit / p0
    """
    gamma = float(cfg.get("gamma", 1.4))
    gas_r = float(cfg.get("gas_constant", 287.0))
    p0    = float(cfg.get("stagnation_pressure", 3.0e5))
    t0    = float(cfg.get("stagnation_temperature", 900.0))
    pa    = float(cfg.get("ambient_pressure", 2.4e4))
    depth = float(cfg.get("depth", 0.02))
    dim   = str(cfg.get("dimension", "2d_planar"))
    cd    = float(cfg.get("discharge_coefficient", 0.985))
    g0    = 9.80665

    throat_y = geometry.throat_radius
    if dim in ("2d_planar", "3d_channel"):
        a_throat = 2.0 * throat_y * depth
    else:
        a_throat = math.pi * throat_y ** 2

    # Choked mass flow
    mdot = (
        cd * a_throat * p0
        / math.sqrt(max(t0, 1e-6))
        * math.sqrt(gamma / gas_r)
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )

    thrust = float(result.thrust)
    c_f  = thrust / max(p0 * a_throat, 1e-12)
    isp  = thrust / max(mdot * g0, 1e-12)

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


def _fidelity_gap(q1d_metrics: Dict[str, Any], rans_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Compute relative differences between Q1D and RANS metrics (for paper Table)."""
    gap: Dict[str, Any] = {}
    for key in ("thrust_N", "C_F", "Isp_s", "exit_mach", "pressure_loss"):
        v_q = float(q1d_metrics.get(key, 0))
        v_r = float(rans_metrics.get(key, 0))
        ref = max(abs(v_r), abs(v_q), 1e-12)
        gap[f"delta_{key}"] = round(v_r - v_q, 6)
        gap[f"delta_{key}_pct"] = round(100.0 * (v_r - v_q) / ref, 3)
    return gap


# ---------------------------------------------------------------------------
# 5-Phase Pipeline
#   Phase 1 -- MOC Baseline (Prandtl-Meyer isentropic)
#   Phase 2 -- GA Optimisation (Quasi-1D surrogate)
#   Phase 3 -- RANS Validation (MOC + top-K GA candidates)
#   Phase 4 -- Multi-fidelity Comparison (Q1D vs RANS table)
#   Phase 5 -- Report & Plots
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
        f"algorithm={opt_cfg.get('algorithm', '?')}  "
        f"pop={opt_cfg.get('population','?')}  gen={opt_cfg.get('generations','?')}",
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

    # Quasi-1D config (always available as the cheap surrogate)
    q1d_cfg = dict(eval_cfg)
    q1d_cfg["backend"] = "quasi1d"

    # RANS config (only used when RANS is enabled)
    rans_cfg = {
        **eval_cfg,
        "backend": "openfoam",
        "mach_exit": float(moc_cfg.get("mach_exit", 2.3)),
        "pressure_ratio": float(moc_cfg.get("pressure_ratio", 0.10)),
    }

    # Determine if RANS is actually available
    run_rans = backend == "openfoam"
    if run_rans and not _docker_available():
        _log("WARNING: backend=openfoam but Docker not available -- RANS disabled", t0)
        run_rans = False

    # ==================================================================
    # PHASE 1 -- MOC Baseline
    # ==================================================================
    _log("=" * 60, t0)
    _log("PHASE 1  |  MOC Baseline (2-D planar Prandtl-Meyer)", t0)
    _log("=" * 60, t0)

    moc_solver = MOCSolver(
        machExit=float(moc_cfg["mach_exit"]),
        pressureRatio=float(moc_cfg["pressure_ratio"]),
        gamma=float(moc_cfg.get("gamma", 1.4)),
    )
    moc_geometry = moc_solver.generateGeometry({**moc_cfg["geometry"]})
    moc_geometry.metadata["id"] = "moc_baseline"
    _log(
        f"  MOC geometry  |  Me={moc_cfg['mach_exit']}  "
        f"throat={throat*1000:.1f} mm  "
        f"exit={float(moc_cfg['geometry']['exit_y'])*1000:.1f} mm  "
        f"L={float(moc_cfg['geometry']['length'])*1000:.1f} mm",
        t0,
    )

    # Export artefacts
    moc_geometry.exportGeo(str(moc_dir / "moc_geometry.csv"))
    moc_geometry.plotProfile(str(moc_dir / "moc_geometry.png"))

    # MLN auxiliary plot
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

    # Evaluate MOC baseline with Quasi-1D
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
    # PHASE 2 -- GA Optimisation (Quasi-1D surrogate)
    # ==================================================================
    _log("=" * 60, t0)
    _log("PHASE 2  |  GA Optimisation (Quasi-1D surrogate)", t0)
    _log("=" * 60, t0)

    pop  = int(opt_cfg.get("population", 20))
    gens = int(opt_cfg.get("generations", 8))
    total_evals = pop * gens
    _log(f"  population={pop}  generations={gens}  ~{total_evals} evals", t0)

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
        evaluator=moc_q1d_eval,
        historyPath=str(opt_dir / "optimization_history.json"),
    )

    runner.start()
    runner.saveHistory()
    _log(f"  GA DONE  |  {len(optimizer._history)} evaluations", t0)

    # ---- Pareto analysis ----
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
                moc_thrust=float(moc_q1d_result.thrust),
                moc_loss=float(moc_q1d_result.pressureLoss),
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
        f"  Best Q1D  |  F={best_q1d.thrust:.3f} N  "
        f"loss={best_q1d.pressureLoss:.5f}  params={best_params}",
        t0,
    )

    optimized_geometry = build_geometry_from_params(
        best_params, throat, n_pts, profile, "optimized_best"
    )
    optimized_geometry.exportGeo(str(opt_dir / "optimized_geometry.csv"))
    optimized_geometry.plotProfile(str(opt_dir / "optimized_geometry.png"))

    opt_q1d_metrics = _compute_performance_metrics(
        best_q1d, optimized_geometry, q1d_cfg
    )

    # ==================================================================
    # PHASE 3 -- RANS Validation
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

        # 3a) RANS on MOC baseline (critical for paper: Q1D vs RANS on same geometry)
        _log("  3a) RANS on MOC baseline geometry ...", t0)
        try:
            moc_rans_eval = OpenFOAMRANSEvaluator(
                geometry=moc_geometry,
                solverConfig=rans_cfg,
                resultPath=str(rans_dir),
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
                f"Isp={moc_rans_metrics['Isp_s']:.1f} s  "
                f"Me={moc_rans_metrics['exit_mach']:.3f}",
                t0,
            )
        except Exception as exc:
            _log(f"  MOC RANS FAILED: {exc}", t0)

        # 3b) RANS on the GA-selected best candidate
        _log("  3b) RANS on GA-optimised best candidate ...", t0)
        try:
            opt_rans_eval = OpenFOAMRANSEvaluator(
                geometry=optimized_geometry,
                solverConfig=rans_cfg,
                resultPath=str(rans_dir),
            )
            opt_rans_result = opt_rans_eval.extractResults()
            opt_rans_result.geometryId = "optimized_best_rans"
            opt_rans_result.saveToJSON(str(rans_dir / "opt_rans_result.json"))
            opt_rans_metrics = _compute_performance_metrics(
                opt_rans_result, optimized_geometry, eval_cfg
            )
            _log(
                f"  OPT RANS  |  F={opt_rans_metrics['thrust_N']:.3f} N  "
                f"C_F={opt_rans_metrics['C_F']:.4f}  "
                f"Isp={opt_rans_metrics['Isp_s']:.1f} s  "
                f"Me={opt_rans_metrics['exit_mach']:.3f}",
                t0,
            )
        except Exception as exc:
            _log(f"  OPT RANS FAILED: {exc}", t0)

        # 3c) RANS on additional top Pareto candidates
        rans_top_k = int(opt_cfg.get("rans_top_k", 5))
        scored_idxs = sorted(
            range(len(optimizer._history)),
            key=lambda i: optimizer._objective_score(optimizer._history[i]),
            reverse=True,
        )
        # Exclude the already-evaluated selected_idx
        extra_idxs = [i for i in scored_idxs if i != selected_idx][:max(0, rans_top_k - 1)]

        if extra_idxs:
            _log(f"  3c) RANS on {len(extra_idxs)} additional top candidates ...", t0)
            for idx in extra_idxs:
                p = optimizer._history_params[idx]
                gid = f"rans_{idx:04d}"
                geom = build_geometry_from_params(p, throat, n_pts, profile, gid)
                _log(f"    RANS #{idx} ({gid}) ...", t0)
                try:
                    e = OpenFOAMRANSEvaluator(
                        geometry=geom,
                        solverConfig=rans_cfg,
                        resultPath=str(rans_dir),
                    )
                    r = e.extractResults()
                    r.geometryId = gid
                    r.saveToJSON(str(rans_dir / f"{gid}_result.json"))
                    rm = _compute_performance_metrics(r, geom, eval_cfg)
                    rans_candidates[gid] = {
                        "candidate_index": idx,
                        "status": "ok",
                        "params": dict(p),
                        **rm,
                    }
                    _log(
                        f"    {gid}  F={rm['thrust_N']:.3f} N  "
                        f"loss={rm['pressure_loss']:.5f}  [OK]",
                        t0,
                    )
                except Exception as exc:
                    rans_candidates[gid] = {
                        "candidate_index": idx,
                        "status": "failed",
                        "error": str(exc),
                        "params": dict(p),
                    }
                    _log(f"    {gid}  FAILED: {exc}", t0)

        (rans_dir / "rans_candidates.json").write_text(
            json.dumps(rans_candidates, indent=2), encoding="utf-8"
        )
    else:
        _log("PHASE 3  |  RANS Validation  [skipped: no Docker / backend=quasi1d]", t0)
        _log("=" * 60, t0)

    # ==================================================================
    # PHASE 4 -- Multi-fidelity Comparison Table
    # ==================================================================
    _log("=" * 60, t0)
    _log("PHASE 4  |  Multi-fidelity Comparison", t0)
    _log("=" * 60, t0)

    fidelity_table: Dict[str, Any] = {
        "moc_q1d": moc_q1d_metrics,
        "opt_q1d": opt_q1d_metrics,
    }
    if moc_rans_metrics:
        fidelity_table["moc_rans"] = moc_rans_metrics
        fidelity_table["moc_fidelity_gap"] = _fidelity_gap(
            moc_q1d_metrics, moc_rans_metrics
        )
    if opt_rans_metrics:
        fidelity_table["opt_rans"] = opt_rans_metrics
        fidelity_table["opt_fidelity_gap"] = _fidelity_gap(
            opt_q1d_metrics, opt_rans_metrics
        )

    (comp_dir / "fidelity_table.json").write_text(
        json.dumps(fidelity_table, indent=2), encoding="utf-8"
    )

    # Print paper-ready comparison table
    _log("", t0)
    hdr = f"  {'Metric':<22s}  {'MOC-Q1D':>10s}  {'OPT-Q1D':>10s}"
    sep_cols = 4
    if moc_rans_metrics:
        hdr += f"  {'MOC-RANS':>10s}"
        sep_cols += 1
    if opt_rans_metrics:
        hdr += f"  {'OPT-RANS':>10s}"
        sep_cols += 1
    table_sep = "  " + "-" * (sep_cols * 12 + 22)
    _log(table_sep, t0)
    _log(hdr, t0)
    _log(table_sep, t0)

    def _fc(val: float, fmt: str) -> str:
        return format(val, fmt).rjust(10)

    for key, label, fmt in [
        ("thrust_N",          "Thrust [N]",        ".3f"),
        ("C_F",               "C_F [-]",           ".4f"),
        ("Isp_s",             "Isp [s]",           ".1f"),
        ("exit_mach",         "Mach exit [-]",     ".4f"),
        ("pressure_loss",     "Pressure loss [-]", ".5f"),
        ("pressure_recovery", "p_t/p_0 [-]",       ".5f"),
        ("eta_div",           "eta_div [-]",       ".4f"),
        ("exit_angle_deg",    "Exit angle [deg]",  ".2f"),
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

    if fidelity_table.get("moc_fidelity_gap"):
        fg = fidelity_table["moc_fidelity_gap"]
        _log(
            f"  MOC fidelity gap:  dF={fg['delta_thrust_N_pct']:+.2f}%  "
            f"dMe={fg['delta_exit_mach_pct']:+.2f}%  "
            f"dLoss={fg['delta_pressure_loss_pct']:+.2f}%",
            t0,
        )
    if fidelity_table.get("opt_fidelity_gap"):
        fg = fidelity_table["opt_fidelity_gap"]
        _log(
            f"  OPT fidelity gap:  dF={fg['delta_thrust_N_pct']:+.2f}%  "
            f"dMe={fg['delta_exit_mach_pct']:+.2f}%  "
            f"dLoss={fg['delta_pressure_loss_pct']:+.2f}%",
            t0,
        )
    _log("", t0)

    # ==================================================================
    # PHASE 5 -- Benchmark Plots & Report
    # ==================================================================
    _log("=" * 60, t0)
    _log("PHASE 5  |  Benchmark Plots & Report", t0)
    _log("=" * 60, t0)

    # Use RANS results for comparison if available, otherwise Q1D
    final_moc_result = moc_rans_result if moc_rans_result else moc_q1d_result
    final_opt_result = opt_rans_result if opt_rans_result else best_q1d

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
            moc_thrust=float(moc_q1d_result.thrust),
            moc_pressure_loss=float(moc_q1d_result.pressureLoss),
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

        # Side-by-side comparison plots
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

        # Field plots
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
            "5-phase pipeline: (1) MOC baseline via Prandtl-Meyer isentropic "
            "characteristic mesh, (2) evolutionary/GA optimisation with a "
            "Quasi-1D surrogate evaluator, (3) RANS validation using OpenFOAM "
            "shockFluid (density-based Kurganov), (4) multi-fidelity Q1D-vs-RANS "
            "comparison table, (5) benchmark plots and report.  "
            "Pareto knee-point selection for multi-objective trade-off."
        ),
    )
    # Comparison metrics
    for k, v in comparison.items():
        note.addMetric(k, v)

    # GA summary
    note.addMetric("optimizer_algorithm", optimizer.algorithm)
    note.addMetric("optimizer_evaluations", len(optimizer._history))
    note.addMetric("ga_population", pop)
    note.addMetric("ga_generations", gens)
    note.addMetric("best_parameters", best_params)
    note.addMetric("selected_candidate_index", selected_idx)

    # Performance metrics
    note.addMetric("moc_q1d_performance", moc_q1d_metrics)
    note.addMetric("opt_q1d_performance", opt_q1d_metrics)
    if moc_rans_metrics:
        note.addMetric("moc_rans_performance", moc_rans_metrics)
    if opt_rans_metrics:
        note.addMetric("opt_rans_performance", opt_rans_metrics)
    note.addMetric("fidelity_table", fidelity_table)

    # Population statistics
    if optimizer._history:
        thrust_vals = [float(r.thrust) for r in optimizer._history]
        loss_vals   = [float(r.pressureLoss) for r in optimizer._history]
        note.addMetric("thrust_mean", sum(thrust_vals) / len(thrust_vals))
        note.addMetric("thrust_std", (sum((t - sum(thrust_vals)/len(thrust_vals))**2 for t in thrust_vals) / len(thrust_vals))**0.5)
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
        "optimized_geometry": str(opt_dir / "optimized_geometry.csv"),
        "comparison": comparison,
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
        f"Pipeline DONE  |  status={summary['status']}  "
        f"elapsed={elapsed_total:.1f}s  rans={'yes' if run_rans else 'no'}",
        t0,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nozzle design benchmark: MOC -> GA/Q1D -> RANS -> Compare -> Report"
    )
    parser.add_argument(
        "--config", type=str, default="",
        help="JSON config file (overrides defaults)",
    )
    parser.add_argument(
        "--algorithm", type=str, default="",
        help="Override optimiser algorithm (evolutionary|random|grid|bayesian_proxy)",
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
        "--quick", action="store_true",
        help="Quick mode: small GA (pop=6 gen=2) for fast smoke tests",
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
    if args.quick:
        cfg["optimization"]["population"] = 6
        cfg["optimization"]["generations"] = 2
        cfg["optimization"]["rans_top_k"] = 2

    summary = run_pipeline(cfg)

    # ---------- Formatted results ----------
    cmp   = summary.get("comparison", {})
    mo    = summary.get("multiobjective", {})
    ftbl  = summary.get("fidelity_table", {})
    sep   = "-" * 60

    print()
    print(sep)
    print("  NOZZLE DESIGN BENCHMARK - RESULTS")
    print(sep)
    print(f"  Status       : {summary.get('status', '?').upper()}")
    print(f"  Elapsed      : {summary.get('elapsed_seconds', 0):.1f} s")
    print(f"  RANS enabled : {'yes' if summary.get('rans_available') else 'no'}")
    print(f"  Output dir   : {summary.get('out_dir', '')}")
    print(sep)

    # Q1D comparison
    if cmp and cmp.get("status") != "failed":
        print("  Q1D COMPARISON  (MOC baseline vs GA-optimised)")
        fmt_h = "  {:<25s}  {:>10s}  {:>10s}  {:>12s}"
        print(fmt_h.format("", "MOC", "OPT", "delta"))
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

    # Fidelity gap (paper's key result)
    if ftbl.get("moc_fidelity_gap") or ftbl.get("opt_fidelity_gap"):
        print("  MULTI-FIDELITY GAP  (Q1D vs RANS)")
        if ftbl.get("moc_fidelity_gap"):
            fg = ftbl["moc_fidelity_gap"]
            print(
                f"  MOC:  dThrust={fg.get('delta_thrust_N_pct', 0):+.2f}%  "
                f"dMach={fg.get('delta_exit_mach_pct', 0):+.2f}%  "
                f"dLoss={fg.get('delta_pressure_loss_pct', 0):+.2f}%"
            )
        if ftbl.get("opt_fidelity_gap"):
            fg = ftbl["opt_fidelity_gap"]
            print(
                f"  OPT:  dThrust={fg.get('delta_thrust_N_pct', 0):+.2f}%  "
                f"dMach={fg.get('delta_exit_mach_pct', 0):+.2f}%  "
                f"dLoss={fg.get('delta_pressure_loss_pct', 0):+.2f}%"
            )
        print(sep)

    # Pareto knee
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

    # RANS candidates
    rc = summary.get("rans_candidates", {})
    if rc:
        ok_r = {k: v for k, v in rc.items() if v.get("status") == "ok"}
        print(f"  RANS CANDIDATES: {len(ok_r)} / {len(rc)} converged")
        for k, v in ok_r.items():
            print(
                f"    {k}  F={v.get('thrust_N', 0):.3f} N  "
                f"Isp={v.get('Isp_s', 0):.1f} s  "
                f"loss={v.get('pressure_loss', 0):.5f}"
            )
        print(sep)

    print(f"  report -> {summary.get('out_dir', '')}/report.md")
    print(sep)
    print()


if __name__ == "__main__":
    main()
