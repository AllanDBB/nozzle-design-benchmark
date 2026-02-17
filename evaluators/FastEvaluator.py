from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .EvaluationResult import EvaluationResult
from geometry import NozzleGeometry
from .CFDSimulation import CFDSimulation


@dataclass
class FastEvaluator:
    """Low-cost evaluator for optimization warm-up or screening."""

    model: Any = None
    gamma: float = 1.4

    def predict(self, geometry: NozzleGeometry) -> EvaluationResult:
        if hasattr(self.model, "predict"):
            return self.model.predict(geometry)

        area_ratio = geometry.expansion_ratio
        mach_exit = CFDSimulation._mach_from_area(area_ratio, self.gamma)
        pressure_ratio = (1.0 + 0.5 * (self.gamma - 1.0) * mach_exit * mach_exit) ** (
            -self.gamma / (self.gamma - 1.0)
        )
        pressure_loss = 1.0 - pressure_ratio

        # Mild heuristic penalty vs ideal CFD-like output.
        pressure_loss = min(0.999, pressure_loss + 0.01 * max(area_ratio - 4.0, 0.0))

        thrust = area_ratio * mach_exit * (1.0 - pressure_loss)
        return EvaluationResult(
            machProfile=[1.0, mach_exit],
            pressureLoss=pressure_loss,
            thrust=thrust,
            geometryId=str(geometry.metadata.get("id", "geometry_fast")),
            metadata={"mode": "fast"},
        )

    def train(self, dataset: List[Any]) -> None:
        self.model = {"dataset_size": len(dataset), "dataset": dataset}
