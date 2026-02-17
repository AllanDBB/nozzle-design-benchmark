from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import json


@dataclass
class AnalysisNote:
    """Container for analysis notes and derived metrics."""

    title: str
    notes: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def addMetric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def toMarkdown(self) -> str:
        lines = [f"# {self.title}", "", self.notes.strip(), "", "## Metrics"]
        for key, value in self.metrics.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.toMarkdown(), encoding="utf-8")

    def saveJSON(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"title": self.title, "notes": self.notes, "metrics": self.metrics}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
