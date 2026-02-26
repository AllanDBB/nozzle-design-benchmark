"""Particle Swarm Optimisation (PSO) with adaptive inertia for nozzle design.

Replaces evolutionary / GA search with a swarm-intelligence approach
better suited to the continuous, low-dimensional parameter space of
convergent-divergent nozzle geometry (exit_radius, length, straighten_frac).

Algorithm
---------
Standard PSO with:
  - Adaptive inertia weight  w : w_max → w_min  over iterations
  - Cognitive coefficient    c1 = 2.0  (personal best attraction)
  - Social coefficient       c2 = 2.0  (global best attraction)
  - Velocity clamping to ±v_max  (fraction of span)
  - Optional constriction factor (χ) when c1+c2 > 4

References
----------
Kennedy, J. & Eberhart, R. (1995) - Particle Swarm Optimization
Shi, Y. & Eberhart, R. (1998)   - Modified PSO with inertia weight
Clerc, M. & Kennedy, J. (2002)  - Constriction factor approach
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import itertools
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluators import EvaluationResult


# ---------------------------------------------------------------------- #
#  Particle data structure                                                 #
# ---------------------------------------------------------------------- #

@dataclass
class _Particle:
    """Single PSO particle."""
    position: Dict[str, float]
    velocity: Dict[str, float]
    best_position: Dict[str, float]
    best_score: float = -math.inf


# ---------------------------------------------------------------------- #
#  SwarmOptimizer                                                          #
# ---------------------------------------------------------------------- #

@dataclass
class SwarmOptimizer:
    """Particle Swarm Optimiser over a bounded continuous search space.

    searchSpace keys
    ----------------
    bounds : dict[str, (lo, hi)]       - required
    swarm_size : int                   - number of particles (default 30)
    iterations : int                   - PSO iterations     (default 15)
    w_max : float                      - initial inertia    (default 0.9)
    w_min : float                      - final inertia      (default 0.4)
    c1 : float                         - cognitive coeff    (default 2.0)
    c2 : float                         - social coeff       (default 2.0)
    v_max_frac : float                 - velocity clamp as fraction of span (0.25)
    n_workers : int                    - parallel evaluation threads

    Multi-objective scoring
    -----------------------
    w_thrust, w_pressure_loss          - same semantics as Optimizer
    campaign, objective_terms          - penalty modifiers

    The public interface mirrors ``Optimizer`` so that ``OptimizationRunner``
    and ``BenchmarkSuite`` work unchanged.
    """

    searchSpace: Dict[str, Any]
    objectiveFunc: Callable[[Dict[str, float]], EvaluationResult]
    algorithm: str = "pso"
    seed: int = 42

    _history: List[EvaluationResult] = field(default_factory=list)
    _history_params: List[Dict[str, float]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False,
                                  repr=False, compare=False)
    _counter: Any = field(default=None, init=False, repr=False, compare=False)

    # PSO diagnostic traces (filled during run)
    _gbest_trace: List[float] = field(default_factory=list)
    _inertia_trace: List[float] = field(default_factory=list)
    _diversity_trace: List[float] = field(default_factory=list)
    _particle_traces: List[List[Dict[str, float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._counter = itertools.count(0)

    # ------------------------------------------------------------------ #
    #  Scoring (identical logic to Optimizer for drop-in compatibility)    #
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

    def _objective_score(
        self,
        result: EvaluationResult,
        reference: Optional[List[EvaluationResult]] = None,
    ) -> float:
        if result.metadata.get("status") == "failed":
            return -1.0

        ref = reference if reference is not None else (
            self._history if self._history else [result]
        )
        thrust_vals = [float(r.thrust) for r in ref]
        t_lo, t_hi = min(thrust_vals), max(thrust_vals)
        thrust_norm = self._norm01(float(result.thrust), t_lo, t_hi)
        loss_norm = max(0.0, min(1.0, float(result.pressureLoss)))

        wt = self._w_thrust()
        wl = self._w_loss()
        total = max(wt + wl, 1e-9)
        score = (wt / total) * thrust_norm + (wl / total) * (1.0 - loss_norm)

        # Campaign-specific penalties
        campaign = str(self.searchSpace.get("campaign", "")).lower()
        terms = self.searchSpace.get("objective_terms", {})
        if not isinstance(terms, dict):
            terms = {}
        shock = result.shock if isinstance(result.shock, dict) else {}
        shock_present = bool(
            shock.get("present", False)
            or result.metadata.get("shock_present", False)
        )
        if campaign == "design_supersonic" and shock_present:
            score *= max(0.0, 1.0 - float(terms.get("design_shock_penalty", 0.10)))
        if campaign == "overexpanded_sea_level":
            x_std = shock.get("x_std", result.metadata.get("shock_x_std"))
            if x_std is not None:
                pen = float(terms.get("overexpanded_instability_penalty", 0.05))
                score *= max(0.0, 1.0 - pen * min(1.0, float(x_std) / 0.05))
            conv_m = (result.convergence or {}).get("metrics", {})
            if isinstance(conv_m, dict):
                osc = max(
                    float(conv_m.get("p_out_rel_std", 0.0)) / 0.02,
                    float(conv_m.get("mdot_rel_std", 0.0)) / 0.02,
                    float(conv_m.get("ux_out_rel_std", 0.0)) / 0.03,
                )
                osc_pen = float(terms.get("overexpanded_outlet_osc_penalty", 0.05))
                score *= max(0.0, 1.0 - osc_pen * min(1.0, osc))
                wall_rel = conv_m.get(
                    "wall_p_rel_rms",
                    result.metadata.get("wall_pressure_rel_rms"),
                )
                if wall_rel is not None:
                    wp = float(terms.get("overexpanded_wall_rms_penalty", 0.03))
                    score *= max(0.0, 1.0 - wp * min(1.0, float(wall_rel) / 0.05))
        return score

    def _best_idx(self) -> int:
        return max(
            range(len(self._history)),
            key=lambda i: self._objective_score(self._history[i]),
        )

    # ------------------------------------------------------------------ #
    #  Search-space helpers                                                #
    # ------------------------------------------------------------------ #

    def _bounds(self) -> Dict[str, Tuple[float, float]]:
        b = self.searchSpace.get("bounds")
        if not b:
            raise ValueError("searchSpace['bounds'] is required")
        return b

    def _sample_uniform(self, rng: random.Random) -> Dict[str, float]:
        return {
            k: rng.uniform(float(lo), float(hi))
            for k, (lo, hi) in self._bounds().items()
        }

    def _clamp(self, params: Dict[str, float]) -> Dict[str, float]:
        out = dict(params)
        for k, (lo, hi) in self._bounds().items():
            out[k] = min(max(out[k], lo), hi)
        return out

    # ------------------------------------------------------------------ #
    #  Evaluation helpers                                                  #
    # ------------------------------------------------------------------ #

    def _evaluate(self, params: Dict[str, float]) -> EvaluationResult:
        result = self.objectiveFunc(params)
        with self._lock:
            self._history.append(result)
            self._history_params.append(dict(params))
        return result

    def _evaluate_batch(
        self, candidates: List[Dict[str, float]]
    ) -> List[EvaluationResult]:
        n_workers = int(self.searchSpace.get("n_workers", 1))
        if n_workers <= 1 or len(candidates) <= 1:
            return [self._evaluate(p) for p in candidates]

        n_workers = min(n_workers, len(candidates))
        index_map: Dict[int, Tuple[EvaluationResult, Dict[str, float]]] = {}

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(self.objectiveFunc, p): (i, p)
                for i, p in enumerate(candidates)
            }
            for fut in as_completed(futures):
                idx, p = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = EvaluationResult(
                        machProfile=[],
                        pressureLoss=1.0,
                        thrust=-1.0e30,
                        geometryId=f"pso_fail_{idx}",
                        metadata={"status": "failed", "error": str(exc)},
                    )
                index_map[idx] = (res, p)

        results: List[EvaluationResult] = []
        with self._lock:
            for i in range(len(candidates)):
                res, p = index_map[i]
                self._history.append(res)
                self._history_params.append(dict(p))
                results.append(res)
        return results

    # ------------------------------------------------------------------ #
    #  PSO core                                                            #
    # ------------------------------------------------------------------ #

    def _run_pso(self, rng: random.Random) -> None:
        bounds = self._bounds()
        keys = list(bounds.keys())
        n_particles = int(self.searchSpace.get("swarm_size", 30))
        n_iter = int(self.searchSpace.get("iterations", 15))
        w_max = float(self.searchSpace.get("w_max", 0.9))
        w_min = float(self.searchSpace.get("w_min", 0.4))
        c1 = float(self.searchSpace.get("c1", 2.0))
        c2 = float(self.searchSpace.get("c2", 2.0))
        v_frac = float(self.searchSpace.get("v_max_frac", 0.25))

        # Velocity limits
        v_max: Dict[str, float] = {
            k: v_frac * (hi - lo) for k, (lo, hi) in bounds.items()
        }

        # Constriction factor χ (Clerc & Kennedy 2002)
        phi = c1 + c2
        if phi > 4.0:
            chi = 2.0 / abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi))
        else:
            chi = 1.0

        # Initialise swarm
        swarm: List[_Particle] = []
        init_positions = [self._sample_uniform(rng) for _ in range(n_particles)]

        # Evaluate initial positions as a batch
        init_results = self._evaluate_batch(init_positions)

        for pos, res in zip(init_positions, init_results):
            vel = {k: rng.uniform(-v_max[k], v_max[k]) for k in keys}
            score = self._objective_score(res, reference=init_results)
            swarm.append(_Particle(
                position=dict(pos),
                velocity=vel,
                best_position=dict(pos),
                best_score=score,
            ))

        # Global best
        g_best_idx = max(range(n_particles), key=lambda i: swarm[i].best_score)
        g_best_pos = dict(swarm[g_best_idx].best_position)
        g_best_score = swarm[g_best_idx].best_score

        # Record initial state
        self._gbest_trace.append(g_best_score)
        self._inertia_trace.append(w_max)
        self._particle_traces.append([dict(p.position) for p in swarm])
        self._diversity_trace.append(self._compute_diversity(swarm, keys, bounds))

        # Iterative loop
        for it in range(n_iter):
            # Adaptive inertia (linear decay)
            w = w_max - (w_max - w_min) * it / max(n_iter - 1, 1)

            # Update velocities and positions
            new_positions: List[Dict[str, float]] = []
            for p in swarm:
                new_vel: Dict[str, float] = {}
                new_pos: Dict[str, float] = {}
                for k in keys:
                    r1 = rng.random()
                    r2 = rng.random()
                    cognitive = c1 * r1 * (p.best_position[k] - p.position[k])
                    social = c2 * r2 * (g_best_pos[k] - p.position[k])
                    vk = chi * (w * p.velocity[k] + cognitive + social)
                    # Clamp velocity
                    vk = max(-v_max[k], min(v_max[k], vk))
                    new_vel[k] = vk
                    new_pos[k] = p.position[k] + vk
                p.velocity = new_vel
                p.position = self._clamp(new_pos)
                new_positions.append(dict(p.position))

            # Batch evaluation
            iter_results = self._evaluate_batch(new_positions)

            # Update personal & global bests
            for i, (p, res) in enumerate(zip(swarm, iter_results)):
                score = self._objective_score(res, reference=iter_results)
                if score > p.best_score:
                    p.best_score = score
                    p.best_position = dict(p.position)
                if score > g_best_score:
                    g_best_score = score
                    g_best_pos = dict(p.position)

            # Diagnostics
            self._gbest_trace.append(g_best_score)
            self._inertia_trace.append(w)
            self._particle_traces.append([dict(p.position) for p in swarm])
            self._diversity_trace.append(
                self._compute_diversity(swarm, keys, bounds)
            )

    @staticmethod
    def _compute_diversity(
        swarm: List[_Particle],
        keys: List[str],
        bounds: Dict[str, Tuple[float, float]],
    ) -> float:
        """Normalised swarm diversity: average std-dev across dimensions."""
        if len(swarm) <= 1:
            return 0.0
        n = len(swarm)
        total = 0.0
        for k in keys:
            vals = [p.position[k] for p in swarm]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            span = bounds[k][1] - bounds[k][0]
            total += math.sqrt(var) / max(span, 1e-12)
        return total / len(keys)

    # ------------------------------------------------------------------ #
    #  Public API  (same interface as Optimizer)                            #
    # ------------------------------------------------------------------ #

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

        self._run_pso(rng)

        if not self._history:
            raise RuntimeError("SwarmOptimizer did not evaluate any candidates")
        idx = self._best_idx()
        best = self._history[idx]
        best.metadata["best_params"] = self._history_params[idx]
        return best

    def logBest(self) -> None:
        if not self._history:
            raise RuntimeError("No history. Run optimizer first.")
        idx = self._best_idx()
        best = self._history[idx]
        score = self._objective_score(best)
        print(
            f"Best score: {score:.4g} | thrust: {best.thrust:.3f} N | "
            f"pressureLoss: {best.pressureLoss:.5f} | "
            f"geometry={best.geometryId} | params={self._history_params[idx]}"
        )
