from pathlib import Path
import json

from benchmarks import BenchmarkSuite
from evaluators import CFDSimulation, OpenFOAMRANSEvaluator
from geometry import MOCSolver, NozzleGeometry


def main() -> None:
    out = Path("out/benchmark_only")
    out.mkdir(parents=True, exist_ok=True)

    moc_solver = MOCSolver(machExit=2.4, pressureRatio=0.09)
    moc_geom = moc_solver.generateGeometry(
        {"throat_y": 0.02, "exit_y": 0.062, "length": 0.22, "n_points": 180}
    )
    moc_geom.metadata["id"] = "moc_benchmark"

    opt_geom = NozzleGeometry.fromParams(
        {
            "throat_radius": 0.02,
            "exit_radius": 0.06,
            "length": 0.20,
            "profile": "bezier_like",
            "shape": 1.4,
            "metadata": {"id": "optimized_reference"},
        }
    )

    solver_cfg = {
        "backend": "openfoam",
        "gamma": 1.4,
        "stagnation_temperature": 1000.0,
        "stagnation_pressure": 6.0e5,
        "ambient_pressure": 8.0e4,
        "fallback_on_failure": False,
        "friction_scale": 0.2,
        "curvature_scale": 0.05,
        "bl_displacement_scale": 1.0,
        "shock_trigger_ratio": 0.55,
        "divergence_scale": 1.0,
        "dimension": "2d_planar",
        "depth": 0.02,
        "mesh_nx": 120,
        "mesh_ny": 50,
        "end_time": 1200,
        "write_interval": 300,
        "inlet_velocity": 20.0,
        "k_inlet": 1.0,
        "epsilon_inlet": 50.0,
        "p_initial": 5.4e5,
        "u_initial": 1.0,
        "t_initial": 1000.0,
    }
    if str(solver_cfg.get("backend", "quasi1d")).lower() == "openfoam":
        evaluator = OpenFOAMRANSEvaluator(geometry=moc_geom, solverConfig=solver_cfg, resultPath=str(out))
    else:
        evaluator = CFDSimulation(geometry=moc_geom, solverConfig=solver_cfg, resultPath=str(out))

    suite = BenchmarkSuite(mocGeometry=moc_geom, optimizedGeometry=opt_geom, evaluator=evaluator)
    suite.runAll()
    comparison = suite.save(str(out))
    (out / "comparison_print.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
