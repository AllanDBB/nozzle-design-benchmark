from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
import json


@dataclass
class EvaluationResult:
    """Container for evaluator outputs."""

    machProfile: List[float]
    pressureLoss: float
    thrust: float
    geometryId: str
    temperatureProfile: List[float] = field(default_factory=list)
    pressureProfile: List[float] = field(default_factory=list)
    xProfile: List[float] = field(default_factory=list)
    velocityProfile: List[float] = field(default_factory=list)
    convergence: Dict[str, Any] = field(default_factory=dict)
    shock: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compareWith(self, other: "EvaluationResult") -> Dict[str, Any]:
        delta_thrust = self.thrust - other.thrust
        rel_thrust_pct = 100.0 * delta_thrust / other.thrust if other.thrust else 0.0
        return {
            "self_geometry": self.geometryId,
            "other_geometry": other.geometryId,
            "delta_thrust": delta_thrust,
            "delta_thrust_percent": rel_thrust_pct,
            "delta_pressure_loss": self.pressureLoss - other.pressureLoss,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def saveToJSON(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
