from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Tuple
import random
import math

from evaluators import EvaluationResult


@dataclass
class Optimizer:
    """Flexible optimization driver over a parametrized search space."""

    searchSpace: Dict[str, Any]
    objectiveFunc: Callable[[Dict[str, float]], EvaluationResult]
    algorithm: str = "random"
    seed: int = 42
    _history: List[EvaluationResult] = field(default_factory=list)
    _history_params: List[Dict[str, float]] = field(default_factory=list)

    def _bounds(self) -> Dict[str, Tuple[float, float]]:
        bounds = self.searchSpace.get("bounds")
        if not bounds:
            raise ValueError("searchSpace['bounds'] is required")
        return bounds

    def _sample_uniform(self, rng: random.Random) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for key, (lo, hi) in self._bounds().items():
            params[key] = rng.uniform(float(lo), float(hi))
        return params

    def _grid_candidates(self) -> List[Dict[str, float]]:
        bounds = self._bounds()
        points = int(self.searchSpace.get("grid_points", 4))
        keys = list(bounds.keys())

        axes: List[List[float]] = []
        for key in keys:
            lo, hi = bounds[key]
            if points <= 1:
                axes.append([(lo + hi) * 0.5])
            else:
                axes.append([lo + (hi - lo) * i / (points - 1) for i in range(points)])

        candidates: List[Dict[str, float]] = [{}]
        for key, axis in zip(keys, axes):
            next_candidates: List[Dict[str, float]] = []
            for base in candidates:
                for val in axis:
                    c = dict(base)
                    c[key] = float(val)
                    next_candidates.append(c)
            candidates = next_candidates
        return candidates

    def _clamp(self, params: Dict[str, float]) -> Dict[str, float]:
        bounds = self._bounds()
        out = dict(params)
        for k, (lo, hi) in bounds.items():
            out[k] = min(max(out[k], lo), hi)
        return out

    def _evaluate(self, params: Dict[str, float]) -> EvaluationResult:
        result = self.objectiveFunc(params)
        self._history.append(result)
        self._history_params.append(dict(params))
        return result

    def _run_random(self, rng: random.Random) -> None:
        n = int(self.searchSpace.get("n_samples", 30))
        for _ in range(n):
            self._evaluate(self._sample_uniform(rng))

    def _run_grid(self) -> None:
        for params in self._grid_candidates():
            self._evaluate(params)

    def _run_evolutionary(self, rng: random.Random) -> None:
        pop_size = int(self.searchSpace.get("population", 18))
        generations = int(self.searchSpace.get("generations", 10))
        elite = max(2, int(pop_size * 0.25))
        sigma = float(self.searchSpace.get("sigma", 0.12))

        population = [self._sample_uniform(rng) for _ in range(pop_size)]
        for _ in range(generations):
            scored: List[Tuple[EvaluationResult, Dict[str, float]]] = []
            for p in population:
                scored.append((self._evaluate(p), p))
            scored.sort(key=lambda x: x[0].thrust, reverse=True)
            parents = [dict(p) for _, p in scored[:elite]]

            new_pop = parents[:]
            while len(new_pop) < pop_size:
                a, b = rng.sample(parents, 2)
                child: Dict[str, float] = {}
                for k in a.keys():
                    alpha = rng.random()
                    v = alpha * a[k] + (1.0 - alpha) * b[k]
                    lo, hi = self._bounds()[k]
                    v += rng.gauss(0.0, sigma * (hi - lo))
                    child[k] = v
                new_pop.append(self._clamp(child))
            population = new_pop

    def _run_bayesian_proxy(self, rng: random.Random) -> None:
        # Lightweight trust-region heuristic (not full GP).
        warmup = int(self.searchSpace.get("warmup", 12))
        n_iter = int(self.searchSpace.get("n_iter", 24))
        jitter = float(self.searchSpace.get("jitter", 0.08))

        for _ in range(warmup):
            self._evaluate(self._sample_uniform(rng))

        for _ in range(max(0, n_iter - warmup)):
            idx_best = max(range(len(self._history)), key=lambda i: self._history[i].thrust)
            best_p = self._history_params[idx_best]
            cand: Dict[str, float] = {}
            for k, (lo, hi) in self._bounds().items():
                span = hi - lo
                cand[k] = best_p[k] + rng.gauss(0.0, jitter * span)
            self._evaluate(self._clamp(cand))

    def run(self) -> EvaluationResult:
        rng = random.Random(self.seed)
        self._history = []
        self._history_params = []

        algo = self.algorithm.lower()
        if algo == "random":
            self._run_random(rng)
        elif algo == "grid":
            self._run_grid()
        elif algo in {"ga", "cma-es", "evolutionary"}:
            self._run_evolutionary(rng)
        elif algo in {"bayesian", "bayesian_proxy"}:
            self._run_bayesian_proxy(rng)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        if not self._history:
            raise RuntimeError("Optimizer did not evaluate any candidates")
        idx = max(range(len(self._history)), key=lambda i: self._history[i].thrust)
        best = self._history[idx]
        best.metadata["best_params"] = self._history_params[idx]
        return best

    def logBest(self) -> None:
        if not self._history:
            raise RuntimeError("No optimization history. Run optimizer first.")
        idx = max(range(len(self._history)), key=lambda i: self._history[i].thrust)
        best = self._history[idx]
        print(f"Best thrust: {best.thrust:.3f} N | geometry={best.geometryId} | params={self._history_params[idx]}")
