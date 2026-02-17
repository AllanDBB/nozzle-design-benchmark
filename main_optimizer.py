from pathlib import Path

from evaluators import CFDSimulation, OpenFOAMRANSEvaluator
from geometry import NozzleGeometry
from optimization import Optimizer, OptimizationRunner


def main() -> None:
    out = Path("out/optimization_only")
    out.mkdir(parents=True, exist_ok=True)

    throat = 0.02
    eval_cfg = {
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
    if str(eval_cfg.get("backend", "quasi1d")).lower() == "openfoam":
        evaluator = OpenFOAMRANSEvaluator(geometry=seed_geometry, solverConfig=eval_cfg, resultPath=str(out))
    else:
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
            "n_samples": 3,
        },
        objectiveFunc=objective,
        algorithm="random",
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
