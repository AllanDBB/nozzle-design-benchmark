from pathlib import Path

from evaluators import CFDSimulation
from geometry import NozzleGeometry
from optimization import Optimizer, OptimizationRunner


def main() -> None:
    out = Path("out/optimization_only")
    out.mkdir(parents=True, exist_ok=True)

    throat = 0.02
    eval_cfg = {
        "gamma": 1.4,
        "stagnation_temperature": 1000.0,
        "stagnation_pressure": 1.5e6,
        "ambient_pressure": 1.0e4,
        "friction_scale": 0.2,
        "curvature_scale": 0.05,
        "bl_displacement_scale": 1.0,
        "shock_trigger_ratio": 0.55,
        "divergence_scale": 1.0,
    }

    seed_geometry = NozzleGeometry.fromParams(
        {
            "throat_radius": throat,
            "exit_radius": 0.06,
            "length": 0.22,
            "profile": "bezier_like",
            "shape": 1.8,
            "metadata": {"id": "seed"},
        }
    )
    evaluator = CFDSimulation(geometry=seed_geometry, solverConfig=eval_cfg, resultPath=str(out))

    def objective(params):
        geom = NozzleGeometry.fromParams(
            {
                "throat_radius": throat,
                "exit_radius": params["exit_radius"],
                "length": params["length"],
                "profile": "bezier_like",
                "shape": params["shape"],
                "metadata": {"id": "opt_candidate"},
            }
        )
        evaluator.geometry = geom
        return evaluator.extractResults()

    optimizer = Optimizer(
        searchSpace={
            "bounds": {
                "exit_radius": (0.05, 0.085),
                "length": (0.16, 0.35),
                "shape": (1.1, 2.4),
            },
            "population": 18,
            "generations": 10,
        },
        objectiveFunc=objective,
        algorithm="ga",
        seed=42,
    )

    runner = OptimizationRunner(
        optimizer=optimizer,
        evaluator=evaluator,
        historyPath=str(out / "optimization_history.json"),
    )
    runner.start()
    runner.saveHistory()
    optimizer.logBest()


if __name__ == "__main__":
    main()
