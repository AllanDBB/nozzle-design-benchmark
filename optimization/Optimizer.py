from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Tuple, Optional
import random
import math
import threading
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluators import EvaluationResult


@dataclass
class Optimizer:
    """Flexible optimization driver over a parametrized search space.

    Supports parallel batch evaluation (set ``n_workers`` > 1 in searchSpace)
    and weighted multi-objective scoring (``w_thrust`` + ``w_pressure_loss``).
    """

    searchSpace: Dict[str, Any]
    objectiveFunc: Callable[[Dict[str, float]], EvaluationResult]
    algorithm: str = "random"
    seed: int = 42
    _history: List[EvaluationResult] = field(default_factory=list)
    _history_params: List[Dict[str, float]] = field(default_factory=list)
    # Thread-safety primitives (not included in repr/comparison).
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _counter: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._counter = itertools.count(0)

    # ------------------------------------------------------------------ #
    #  Multi-objective scoring helpers                                     #
    # ------------------------------------------------------------------ #

    def _w_thrust(self) -> float:
        return float(self.searchSpace.get("w_thrust", 1.0))

    def _w_loss(self) -> float:
        return float(self.searchSpace.get("w_pressure_loss", 0.0))

    def _objective_score(self, result: EvaluationResult) -> float:
        """Weighted score: maximize thrust, minimize pressure loss.

        Higher is always better (for consistent parent/best selection).
        """
        wt = self._w_thrust()
        wl = self._w_loss()
        total = max(wt + wl, 1e-9)
        # Normalise pressure loss to [0..1] heuristic: penalise loss contribution.
        return (wt / total) * result.thrust - (wl / total) * result.pressureLoss * result.thrust

    def _best_idx(self) -> int:
        return max(range(len(self._history)), key=lambda i: self._objective_score(self._history[i]))

    # ------------------------------------------------------------------ #
    #  Search space helpers                                                #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Evaluation (single + batch)                                        #
    # ------------------------------------------------------------------ #

    def _next_id(self) -> int:
        """Thread-safe evaluation counter."""
        with self._lock:
            return next(self._counter)

    def _evaluate(self, params: Dict[str, float]) -> EvaluationResult:
        result = self.objectiveFunc(params)
        with self._lock:
            self._history.append(result)
            self._history_params.append(dict(params))
        return result

    def _evaluate_batch(self, candidates: List[Dict[str, float]]) -> List[EvaluationResult]:
        """Evaluate a batch of candidates, in parallel when n_workers > 1.

        Parallel workers require ``objectiveFunc`` to be thread-safe
        (i.e. each call must create its own evaluator/state independently).
        """
        n_workers = int(self.searchSpace.get("n_workers", 1))
        if n_workers <= 1 or len(candidates) <= 1:
            return [self._evaluate(p) for p in candidates]

        n_workers = min(n_workers, len(candidates))
        index_map: Dict[int, Tuple[EvaluationResult, Dict[str, float]]] = {}

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self.objectiveFunc, p): (idx, p)
                for idx, p in enumerate(candidates)
            }
            for future in as_completed(futures):
                idx, p = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = EvaluationResult(
                        machProfile=[],
                        pressureLoss=1.0,
                        thrust=-1.0e30,
                        geometryId=f"parallel_failed_{idx}",
                        metadata={"status": "failed", "error": str(exc)},
                    )
                index_map[idx] = (result, p)

        # Append to history in original candidate order for reproducibility.
        results: List[EvaluationResult] = []
        with self._lock:
            for i in range(len(candidates)):
                result, p = index_map[i]
                self._history.append(result)
                self._history_params.append(dict(p))
                results.append(result)
        return results

    # ------------------------------------------------------------------ #
    #  Algorithm implementations                                          #
    # ------------------------------------------------------------------ #

    def _run_random(self, rng: random.Random) -> None:
        n = int(self.searchSpace.get("n_samples", 30))
        candidates = [self._sample_uniform(rng) for _ in range(n)]
        self._evaluate_batch(candidates)

    def _run_grid(self) -> None:
        self._evaluate_batch(self._grid_candidates())

    def _run_evolutionary(self, rng: random.Random) -> None:
        pop_size = int(self.searchSpace.get("population", 18))
        generations = int(self.searchSpace.get("generations", 10))
        elite = max(2, int(pop_size * 0.25))
        sigma = float(self.searchSpace.get("sigma", 0.12))

        population = [self._sample_uniform(rng) for _ in range(pop_size)]
        for _ in range(generations):
            # Evaluate entire generation in one (potentially parallel) batch.
            gen_results = self._evaluate_batch(population)

            scored: List[Tuple[float, Dict[str, float]]] = [
                (self._objective_score(r), p) for r, p in zip(gen_results, population)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
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

        # Parallel warmup batch.
        warmup_cands = [self._sample_uniform(rng) for _ in range(warmup)]
        self._evaluate_batch(warmup_cands)

        for _ in range(max(0, n_iter - warmup)):
            idx_best = self._best_idx()
            best_p = self._history_params[idx_best]
            cand: Dict[str, float] = {}
            for k, (lo, hi) in self._bounds().items():
                span = hi - lo
                cand[k] = best_p[k] + rng.gauss(0.0, jitter * span)
            self._evaluate(self._clamp(cand))

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> EvaluationResult:
        rng = random.Random(self.seed)
        with self._lock:
            self._history = []
            self._history_params = []
            self._counter = itertools.count(0)

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
        idx = self._best_idx()
        best = self._history[idx]
        best.metadata["best_params"] = self._history_params[idx]
        return best

    def logBest(self) -> None:
        if not self._history:
            raise RuntimeError("No optimization history. Run optimizer first.")
        idx = self._best_idx()
        best = self._history[idx]
        score = self._objective_score(best)
        print(
            f"Best score: {score:.4g} | thrust: {best.thrust:.3f} N | "
            f"pressureLoss: {best.pressureLoss:.5f} | "
            f"geometry={best.geometryId} | params={self._history_params[idx]}"
        )

