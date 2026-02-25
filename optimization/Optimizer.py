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

    @staticmethod
    def _norm01(value: float, lo: float, hi: float) -> float:
        if abs(hi - lo) < 1e-12:
            return 0.5
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    def _objective_score(self, result: EvaluationResult, reference: Optional[List[EvaluationResult]] = None) -> float:
        """Physics-based fitness: raw thrust with pressure-loss penalty.

        score = thrust * (1 - w_loss * pressureLoss)

        This avoids the normalisation instability of the previous approach
        (where the score depended on the current population range) and
        keeps the units of thrust [N] so the GA always maximises physical
        performance.
        """
        if result.metadata.get("status") == "failed":
            return -1e30

        thrust = float(result.thrust)
        loss = max(0.0, min(1.0, float(result.pressureLoss)))

        wl = self._w_loss()
        # Scale so w_pressure_loss=0.3 penalises ~30% of thrust per unit loss.
        loss_penalty = 1.0 - wl * loss
        score = thrust * max(loss_penalty, 0.01)

        # Eta_div bonus: reward contours with low exit-angle divergence loss.
        eta_div = float(result.metadata.get("eta_div", 1.0))
        score *= max(eta_div, 0.5)

        # Campaign-specific penalties (shocks, oscillation, etc.)
        campaign = str(self.searchSpace.get("campaign", "")).lower()
        terms = self.searchSpace.get("objective_terms", {}) if isinstance(self.searchSpace.get("objective_terms", {}), dict) else {}
        shock = result.shock if isinstance(result.shock, dict) else {}
        shock_present = bool(shock.get("present", False) or result.metadata.get("shock_present", False))
        if campaign == "design_supersonic" and shock_present:
            score *= max(0.0, 1.0 - float(terms.get("design_shock_penalty", 0.10)))
        if campaign == "overexpanded_sea_level":
            x_std = shock.get("x_std", result.metadata.get("shock_x_std"))
            if x_std is not None:
                penalty = float(terms.get("overexpanded_instability_penalty", 0.05))
                score *= max(0.0, 1.0 - penalty * min(1.0, float(x_std) / 0.05))
            conv_metrics = (result.convergence or {}).get("metrics", {})
            if isinstance(conv_metrics, dict):
                osc_index = max(
                    float(conv_metrics.get("p_out_rel_std", 0.0)) / 0.02,
                    float(conv_metrics.get("mdot_rel_std", 0.0)) / 0.02,
                    float(conv_metrics.get("ux_out_rel_std", 0.0)) / 0.03,
                )
                osc_penalty = float(terms.get("overexpanded_outlet_osc_penalty", 0.05))
                score *= max(0.0, 1.0 - osc_penalty * min(1.0, osc_index))
                wall_rel = conv_metrics.get("wall_p_rel_rms", result.metadata.get("wall_pressure_rel_rms"))
                if wall_rel is not None:
                    wall_penalty = float(terms.get("overexpanded_wall_rms_penalty", 0.03))
                    score *= max(0.0, 1.0 - wall_penalty * min(1.0, float(wall_rel) / 0.05))
        return score

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
        sigma_init = float(self.searchSpace.get("sigma", 0.12))
        sigma_decay = float(self.searchSpace.get("sigma_decay", 0.85))
        tournament_k = max(2, min(5, int(self.searchSpace.get("tournament_k", 3))))

        # Build initial population: inject seed candidates first.
        seed_candidates: List[Dict[str, float]] = list(
            self.searchSpace.get("seed_candidates", [])
        )
        population: List[Dict[str, float]] = []
        for sc in seed_candidates[:pop_size]:
            population.append(self._clamp(dict(sc)))
        # Fill remaining with random samples.
        while len(population) < pop_size:
            population.append(self._sample_uniform(rng))

        sigma = sigma_init
        for gen_idx in range(generations):
            # Evaluate entire generation in one (potentially parallel) batch.
            gen_results = self._evaluate_batch(population)

            scored: List[Tuple[float, Dict[str, float]]] = [
                (self._objective_score(r, reference=gen_results), p)
                for r, p in zip(gen_results, population)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            parents = [dict(p) for _, p in scored[:elite]]

            # Build next generation: elites + offspring via tournament.
            new_pop = [dict(p) for p in parents]
            while len(new_pop) < pop_size:
                # Tournament selection for two parents.
                pool_a = rng.sample(scored, min(tournament_k, len(scored)))
                pool_b = rng.sample(scored, min(tournament_k, len(scored)))
                a = max(pool_a, key=lambda x: x[0])[1]
                b = max(pool_b, key=lambda x: x[0])[1]
                child: Dict[str, float] = {}
                for k in a.keys():
                    alpha = rng.random()
                    v = alpha * a[k] + (1.0 - alpha) * b[k]
                    lo, hi = self._bounds()[k]
                    v += rng.gauss(0.0, sigma * (hi - lo))
                    child[k] = v
                new_pop.append(self._clamp(child))
            population = new_pop
            # Adaptive sigma decay: reduce mutation as the GA converges.
            sigma = max(0.01, sigma * sigma_decay)

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
