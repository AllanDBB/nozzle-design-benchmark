from pathlib import Path
import json

from benchmarks import BenchmarkSuite
from evaluators import CFDSimulation
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

    evaluator = CFDSimulation(
        geometry=moc_geom,
        solverConfig={
            "gamma": 1.4,
            "stagnation_temperature": 1000.0,
            "stagnation_pressure": 1.5e6,
            "ambient_pressure": 1.0e4,
            "friction_scale": 0.2,
            "curvature_scale": 0.05,
            "bl_displacement_scale": 1.0,
            "shock_trigger_ratio": 0.55,
            "divergence_scale": 1.0,
        },
        resultPath=str(out),
    )

    suite = BenchmarkSuite(mocGeometry=moc_geom, optimizedGeometry=opt_geom, evaluator=evaluator)
    suite.runAll()
    comparison = suite.save(str(out))
    (out / "comparison_print.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
