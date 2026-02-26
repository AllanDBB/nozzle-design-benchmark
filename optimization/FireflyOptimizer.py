"""Firefly Algorithm (FA) optimiser for continuous nozzle design.

The Firefly Algorithm, introduced by Xin-She Yang (2008), is a
nature-inspired swarm-intelligence metaheuristic based on the flashing
behaviour of fireflies.  Brighter fireflies attract dimmer ones,
providing implicit multi-modal search without explicit velocity vectors.

Key features of this implementation
------------------------------------
  - **Adaptive attractiveness** β(r) = β₀ · e^{-γ·r²}
  - **Lévy-flight randomisation** (same Mantegna approach as MOPSO)
  - **Chaos perturbation** every K iterations (logistic map)
  - **Ranking-based brightness** (normalised objective)

References
----------
Yang, X.-S. (2008)  - Original Firefly Algorithm
Yang, X.-S. (2010)  - Nature-Inspired Metaheuristic Algorithms (2nd ed.)
"""

from __future__ import annotations

import itertools
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from evaluators import EvaluationResult


# ====================================================================== #
#  Helpers (shared with MOPSO)                                             #
# ====================================================================== #

def _levy_step(rng: random.Random, beta: float = 1.5) -> float:
    sigma_u = (
        math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = rng.gauss(0, sigma_u)
    v = rng.gauss(0, 1.0)
    return u / (abs(v) ** (1.0 / beta))


# ====================================================================== #
#  Firefly data structure                                                  #
# ====================================================================== #

@dataclass
class _Firefly:
    position: Dict[str, float]
    brightness: float = 0.0   # proportional to objective


# ====================================================================== #
#  FireflyOptimizer                                                        #
# ====================================================================== #

@dataclass
class FireflyOptimizer:
    """Firefly Algorithm optimiser with Lévy-flight perturbation.

    searchSpace keys
    ----------------
    bounds       : dict[str, (lo, hi)]
    swarm_size   : int   (30)
    iterations   : int   (15)
    beta0        : float (1.0)  — base attractiveness
    gamma_fa     : float (1.0)  — light absorption coefficient
    alpha_fa     : float (0.25) — randomisation scaling
    levy_prob    : float (0.1)  — probability of Lévy perturbation
    chaos_k      : int   (5)    — chaos perturbation every K iters
    """

    searchSpace: Dict[str, Any]
    objectiveFunc: Callable[[Dict[str, float]], EvaluationResult]
    algorithm: str = "firefly"
    seed: int = 42

    _history: List[EvaluationResult] = field(default_factory=list)
    _history_params: List[Dict[str, float]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False,
                                  repr=False, compare=False)
    _counter: Any = field(default=None, init=False, repr=False, compare=False)

    # Diagnostic traces (compatible with SwarmPlots)
    _gbest_trace: List[float] = field(default_factory=list)
    _inertia_trace: List[float] = field(default_factory=list)
    _diversity_trace: List[float] = field(default_factory=list)
    _particle_traces: List[List[Dict[str, float]]] = field(default_factory=list)

    def __post_init__(self):
        self._counter = itertools.count(0)

    # ---- scoring ----------------------------------------------------------

    def _w_thrust(self) -> float:
        return float(self.searchSpace.get("w_thrust", 0.7))

    def _w_loss(self) -> float:
        return float(self.searchSpace.get("w_pressure_loss", 0.3))

    def _scalar_score(self, thrust: float, loss: float) -> float:
        return self._w_thrust() * thrust - self._w_loss() * loss * 1e3

    def _objective_score(self, result: EvaluationResult,
                         reference: Optional[List[EvaluationResult]] = None) -> float:
        if result.metadata.get("status") == "failed":
            return -1.0
        ref = reference or self._history or [result]
        thrusts = [float(r.thrust) for r in ref]
        lo, hi = min(thrusts), max(thrusts)
        tn = (float(result.thrust) - lo) / max(hi - lo, 1e-12)
        ln = max(0.0, min(1.0, float(result.pressureLoss)))
        wt, wl = self._w_thrust(), self._w_loss()
        t = max(wt + wl, 1e-9)
        return (wt / t) * tn + (wl / t) * (1.0 - ln)

    # ---- bounds -----------------------------------------------------------

    def _bounds(self) -> Dict[str, Tuple[float, float]]:
        b = self.searchSpace.get("bounds")
        if not b:
            raise ValueError("searchSpace['bounds'] is required")
        return {k: tuple(v) for k, v in b.items()}

    def _clamp(self, p: Dict[str, float]) -> Dict[str, float]:
        out = dict(p)
        for k, (lo, hi) in self._bounds().items():
            out[k] = min(max(out[k], lo), hi)
        return out

    # ---- evaluation -------------------------------------------------------

    def _evaluate(self, params: Dict[str, float]) -> EvaluationResult:
        result = self.objectiveFunc(params)
        with self._lock:
            self._history.append(result)
            self._history_params.append(dict(params))
        return result

    def _evaluate_batch(self, candidates: List[Dict[str, float]]) -> List[EvaluationResult]:
        n_w = int(self.searchSpace.get("n_workers", 1))
        if n_w <= 1:
            return [self._evaluate(p) for p in candidates]
        results_map: Dict[int, Tuple[EvaluationResult, Dict[str, float]]] = {}
        with ThreadPoolExecutor(max_workers=min(n_w, len(candidates))) as pool:
            futs = {pool.submit(self.objectiveFunc, p): (i, p) for i, p in enumerate(candidates)}
            for f in as_completed(futs):
                idx, p = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    res = EvaluationResult(machProfile=[], pressureLoss=1.0,
                                           thrust=-1e30, geometryId=f"fa_fail_{idx}",
                                           metadata={"status": "failed", "error": str(e)})
                results_map[idx] = (res, p)
        out: List[EvaluationResult] = []
        with self._lock:
            for i in range(len(candidates)):
                r, p = results_map[i]
                self._history.append(r)
                self._history_params.append(dict(p))
                out.append(r)
        return out

    # ---- Euclidean distance (normalised) ----------------------------------

    def _distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        bounds = self._bounds()
        d2 = 0.0
        for k in bounds:
            lo, hi = bounds[k]
            d = (a[k] - b[k]) / max(hi - lo, 1e-12)
            d2 += d * d
        return math.sqrt(d2)

    # ---- Firefly core -----------------------------------------------------

    def _run_firefly(self, rng: random.Random) -> None:
        bounds = self._bounds()
        keys = list(bounds.keys())
        n_ff = int(self.searchSpace.get("swarm_size", 30))
        n_iter = int(self.searchSpace.get("iterations", 15))
        beta0 = float(self.searchSpace.get("beta0", 1.0))
        gamma_fa = float(self.searchSpace.get("gamma_fa", 1.0))
        alpha0 = float(self.searchSpace.get("alpha_fa", 0.25))
        levy_prob = float(self.searchSpace.get("levy_prob", 0.10))
        chaos_k = int(self.searchSpace.get("chaos_k", 5))

        # Initialise population
        init_pos = [
            {k: rng.uniform(*bounds[k]) for k in keys} for _ in range(n_ff)
        ]
        init_results = self._evaluate_batch(init_pos)

        swarm: List[_Firefly] = []
        for pos, res in zip(init_pos, init_results):
            score = self._objective_score(res, reference=init_results)
            swarm.append(_Firefly(position=dict(pos), brightness=score))

        g_best = max(swarm, key=lambda f: f.brightness)
        g_best_score = g_best.brightness

        self._gbest_trace.append(g_best_score)
        self._inertia_trace.append(alpha0)
        self._particle_traces.append([dict(f.position) for f in swarm])
        self._diversity_trace.append(self._diversity(swarm, keys, bounds))

        for it in range(n_iter):
            # Adaptive alpha (decay)
            alpha = alpha0 * (0.97 ** it)

            new_positions: List[Dict[str, float]] = []

            # Sort by brightness descending
            order = sorted(range(n_ff), key=lambda i: swarm[i].brightness, reverse=True)

            for ii in range(n_ff):
                i = order[ii]
                fi = swarm[i]
                moved = False
                for jj in range(n_ff):
                    j = order[jj]
                    fj = swarm[j]
                    if fj.brightness > fi.brightness:
                        r = self._distance(fi.position, fj.position)
                        beta = beta0 * math.exp(-gamma_fa * r * r)
                        new_pos: Dict[str, float] = {}
                        for k in keys:
                            lo, hi = bounds[k]
                            span = hi - lo
                            rand_part = alpha * (rng.random() - 0.5) * span

                            # Lévy perturbation
                            if rng.random() < levy_prob:
                                rand_part += _levy_step(rng) * 0.01 * span

                            new_pos[k] = (
                                fi.position[k]
                                + beta * (fj.position[k] - fi.position[k])
                                + rand_part
                            )
                        fi.position = self._clamp(new_pos)
                        moved = True
                        break  # Move toward brightest attractor

                if not moved:
                    # Brightest firefly: random walk
                    new_pos = {}
                    for k in keys:
                        lo, hi = bounds[k]
                        span = hi - lo
                        step = alpha * (rng.random() - 0.5) * span
                        if rng.random() < levy_prob:
                            step += _levy_step(rng) * 0.01 * span
                        new_pos[k] = fi.position[k] + step
                    fi.position = self._clamp(new_pos)

                new_positions.append(dict(fi.position))

            # Chaos perturbation every chaos_k iterations
            if chaos_k > 0 and (it + 1) % chaos_k == 0:
                n_perturb = max(1, n_ff // 5)
                brightness_sorted = sorted(range(n_ff), key=lambda i: swarm[i].brightness)
                for idx in brightness_sorted[:n_perturb]:
                    new_positions[idx] = {k: rng.uniform(*bounds[k]) for k in keys}
                    swarm[idx].position = dict(new_positions[idx])

            iter_results = self._evaluate_batch(new_positions)

            for i, (fi, res) in enumerate(zip(swarm, iter_results)):
                fi.brightness = self._objective_score(res)

            current_best = max(swarm, key=lambda f: f.brightness)
            if current_best.brightness > g_best_score:
                g_best_score = current_best.brightness
                g_best = current_best

            self._gbest_trace.append(g_best_score)
            self._inertia_trace.append(alpha)
            self._particle_traces.append([dict(f.position) for f in swarm])
            self._diversity_trace.append(self._diversity(swarm, keys, bounds))

    @staticmethod
    def _diversity(swarm, keys, bounds):
        if len(swarm) <= 1:
            return 0.0
        n = len(swarm)
        t = 0.0
        for k in keys:
            vals = [f.position[k] for f in swarm]
            m = sum(vals) / n
            var = sum((v - m) ** 2 for v in vals) / n
            sp = bounds[k][1] - bounds[k][0]
            t += math.sqrt(var) / max(sp, 1e-12)
        return t / len(keys)

    # ---- public API -------------------------------------------------------

    def _best_idx(self) -> int:
        return max(range(len(self._history)),
                   key=lambda i: self._objective_score(self._history[i]))

    def run(self) -> EvaluationResult:
        rng = random.Random(self.seed)
        with self._lock:
            self._history = []
            self._history_params = []
            self._counter = itertools.count(0)
            self._gbest_trace = []
            self._inertia_trace = []
            self._diversity_trace = []
            self._particle_traces = []

        self._run_firefly(rng)

        if not self._history:
            raise RuntimeError("FireflyOptimizer: no evaluations")
        idx = self._best_idx()
        best = self._history[idx]
        best.metadata["best_params"] = self._history_params[idx]
        best.metadata["algorithm"] = "firefly"
        return best

    def logBest(self) -> None:
        if not self._history:
            return
        idx = self._best_idx()
        b = self._history[idx]
        print(
            f"[Firefly] best thrust={b.thrust:.3f} N  loss={b.pressureLoss:.5f}  "
            f"params={self._history_params[idx]}"
        )
