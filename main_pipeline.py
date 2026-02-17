from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from analysis import AnalysisNote
from benchmarks import BenchmarkSuite
from evaluators import CFDSimulation
from geometry import MOCSolver, NozzleGeometry
from optimization import Optimizer, OptimizationRunner


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
            "gamma": 1.4,
            "stagnation_temperature": 1000.0,
            "stagnation_pressure": 1.5e6,
            "ambient_pressure": 1.0e4,
            "n_samples": 90,
            "friction_scale": 0.2,
            "cf_multiplier": 1.0,
            "curvature_scale": 0.05,
            "bl_displacement_scale": 1.0,
            "discharge_coefficient": 0.985,
            "enable_shock_model": True,
            "shock_trigger_ratio": 0.55,
            "divergence_scale": 1.0,
        },
        "optimization": {
            "algorithm": "ga",
            "seed": 42,
            "bounds": {
                "exit_radius": [0.050, 0.085],
                "length": [0.16, 0.35],
                "shape": [1.1, 2.4],
            },
            "population": 18,
            "generations": 10,
            "n_points": 180,
            "profile": "bezier_like",
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


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate MOC baseline geometry.
    moc_cfg = config["moc"]
    moc_solver = MOCSolver(
        machExit=float(moc_cfg["mach_exit"]),
        pressureRatio=float(moc_cfg["pressure_ratio"]),
        gamma=float(moc_cfg.get("gamma", 1.4)),
    )
    moc_geometry = moc_solver.generateGeometry({**moc_cfg["geometry"]})
    moc_geometry.metadata["id"] = "moc_baseline"

    moc_geometry.exportGeo(str(out_dir / "moc_geometry.csv"))
    moc_geometry.plotProfile(str(out_dir / "moc_geometry.png"))

    # 2) Optimize parametrized geometry using CFD-like evaluator.
    eval_cfg = dict(config["evaluator"])
    opt_cfg = config["optimization"]

    evaluator = CFDSimulation(geometry=moc_geometry, solverConfig=eval_cfg, resultPath=str(out_dir))
    throat = float(moc_cfg["geometry"]["throat_y"])
    n_points = int(opt_cfg.get("n_points", 180))
    profile = str(opt_cfg.get("profile", "bezier_like"))

    def objective(param_dict: Dict[str, float]):
        gid = f"opt_candidate_{len(optimizer._history):04d}"
        geom = build_geometry_from_params(param_dict, throat, n_points, profile, gid)
        evaluator.geometry = geom
        result = evaluator.extractResults()
        result.geometryId = gid
        return result

    optimizer = Optimizer(
        searchSpace={k: v for k, v in opt_cfg.items() if k != "algorithm" and k != "seed"},
        objectiveFunc=objective,
        algorithm=str(opt_cfg.get("algorithm", "ga")),
        seed=int(opt_cfg.get("seed", 42)),
    )
    runner = OptimizationRunner(optimizer=optimizer, evaluator=evaluator, historyPath=str(out_dir / "optimization_history.json"))
    runner.start()
    runner.saveHistory()

    best_params = dict(runner.bestResult.metadata.get("best_params", {})) if runner.bestResult else {}
    if not best_params:
        idx = max(range(len(optimizer._history)), key=lambda i: optimizer._history[i].thrust)
        best_params = optimizer._history_params[idx]

    optimized_geometry = build_geometry_from_params(best_params, throat, n_points, profile, "optimized_best")
    optimized_geometry.exportGeo(str(out_dir / "optimized_geometry.csv"))
    optimized_geometry.plotProfile(str(out_dir / "optimized_geometry.png"))

    # 3) Benchmark MOC vs optimized with same evaluator.
    bench_eval = CFDSimulation(geometry=moc_geometry, solverConfig=eval_cfg, resultPath=str(out_dir))
    suite = BenchmarkSuite(mocGeometry=moc_geometry, optimizedGeometry=optimized_geometry, evaluator=bench_eval)
    suite.runAll()
    comparison = suite.save(str(out_dir))

    bench_eval.geometry = moc_geometry
    bench_eval.extractResults()
    bench_eval.plotFields(str(out_dir / "moc_fields.png"))

    bench_eval.geometry = optimized_geometry
    bench_eval.extractResults()
    bench_eval.plotFields(str(out_dir / "optimized_fields.png"))

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

    note.save(str(out_dir / "report.md"))
    note.saveJSON(str(out_dir / "report.json"))

    summary = {
        "out_dir": str(out_dir),
        "moc_geometry": str(out_dir / "moc_geometry.csv"),
        "optimized_geometry": str(out_dir / "optimized_geometry.csv"),
        "comparison": comparison,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
