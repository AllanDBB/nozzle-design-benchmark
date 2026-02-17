from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import json

from geometry import NozzleGeometry
from evaluators import EvaluationResult


@dataclass
class BenchmarkSuite:
    """Benchmark MOC geometry against optimized geometry using same evaluator."""

    mocGeometry: NozzleGeometry
    optimizedGeometry: NozzleGeometry
    evaluator: Any
    mocResult: Optional[EvaluationResult] = None
    optResult: Optional[EvaluationResult] = None

    def runAll(self) -> None:
        self.evaluator.geometry = self.mocGeometry
        self.mocResult = self.evaluator.extractResults()

        self.evaluator.geometry = self.optimizedGeometry
        self.optResult = self.evaluator.extractResults()

    def compare(self) -> Dict[str, Any]:
        if self.mocResult is None or self.optResult is None:
            raise RuntimeError("Run runAll() before compare().")

        comparison = self.optResult.compareWith(self.mocResult)
        comparison["moc_thrust"] = self.mocResult.thrust
        comparison["optimized_thrust"] = self.optResult.thrust
        comparison["moc_pressure_loss"] = self.mocResult.pressureLoss
        comparison["optimized_pressure_loss"] = self.optResult.pressureLoss
        return comparison

    def save(self, outdir: str) -> Dict[str, Any]:
        if self.mocResult is None or self.optResult is None:
            raise RuntimeError("Run runAll() before save().")

        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)

        self.mocResult.saveToJSON(str(out / "moc_result.json"))
        self.optResult.saveToJSON(str(out / "optimized_result.json"))

        comparison = self.compare()
        with open(out / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        return comparison
