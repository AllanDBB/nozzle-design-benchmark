from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from .Optimizer import Optimizer
from evaluators import CFDSimulation, EvaluationResult


@dataclass
class OptimizationRunner:
    """Orchestrates optimizer execution and artifact persistence."""

    optimizer: Optimizer
    evaluator: CFDSimulation
    bestResult: Optional[EvaluationResult] = None
    historyPath: str = "out/optimization/optimization_history.json"

    def start(self) -> None:
        self.bestResult = self.optimizer.run()

    def saveHistory(self) -> None:
        if not self.optimizer._history:
            raise RuntimeError("No history to save. Run optimizer first.")

        path = Path(self.historyPath)
        path.parent.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []
        for result, params in zip(self.optimizer._history, self.optimizer._history_params):
            row = asdict(result)
            row["params"] = params
            records.append(row)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        if self.bestResult is not None:
            best_path = path.with_name("best_result.json")
            self.bestResult.saveToJSON(str(best_path))
