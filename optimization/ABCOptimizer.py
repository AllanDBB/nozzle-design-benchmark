"""Artificial Bee Colony (ABC) Optimiser for nozzle design.

The ABC algorithm (Karaboga 2005) models the foraging behaviour of a
honey-bee colony with three groups:

  - **Employed bees** — exploit known food sources (solutions)
  - **Onlooker bees** — probabilistically select sources by fitness
  - **Scout bees** — abandon exhausted sources and explore randomly

This multi-objective variant maintains a Pareto archive with crowding
distance and uses three objectives:

  1. **Thrust** (maximise)
  2. **Pressure loss** (minimise)
  3. **Nozzle length** (minimise — shorter = lighter)

Key features
------------
  - Roulette-wheel selection for onlookers (fitness-proportional)
  - Abandonment limit: if a source is not improved for `limit` cycles,
    its employed bee becomes a scout
  - Lévy-flight perturbation for scouts (wide exploration)
  - Opposition-based initialisation for better initial coverage
  - Pareto archive with crowding-distance pruning (shared with MOPSO)

References
----------
Karaboga, D. (2005) - An Idea Based on Honey Bee Swarm for Numerical
    Optimization. Technical Report TR06.
Karaboga, D. & Basturk, B. (2007) - A powerful and efficient algorithm
    for numerical function optimization: ABC algorithm.
Akay, B. & Karaboga, D. (2012) - A modified ABC algorithm for
    real-parameter optimization.
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
#  Pareto helpers (same as MOPSO — kept local to avoid circular imports)   #
# ====================================================================== #

def _dominates(a: Dict[str, float], b: Dict[str, float],
               objectives: List[str], directions: List[int]) -> bool:
    """True if *a* Pareto-dominates *b*.  direction=+1 maximise, -1 minimise."""
    dom = False
    for obj, d in zip(objectives, directions):
        va = d * a[obj]
        vb = d * b[obj]
        if va < vb:
            return False
        if va > vb:
            dom = True
    return dom


def _crowding_distance(archive: List[Dict[str, float]],
                       objectives: List[str]) -> List[float]:
    n = len(archive)
    if n <= 2:
        return [float("inf")] * n
    dists = [0.0] * n
    for obj in objectives:
        idx_sorted = sorted(range(n), key=lambda i: archive[i][obj])
        dists[idx_sorted[0]] = float("inf")
        dists[idx_sorted[-1]] = float("inf")
        span = archive[idx_sorted[-1]][obj] - archive[idx_sorted[0]][obj]
        if span < 1e-15:
            continue
        for k in range(1, n - 1):
            dists[idx_sorted[k]] += (
                (archive[idx_sorted[k + 1]][obj] - archive[idx_sorted[k - 1]][obj])
                / span
            )
    return dists


# ====================================================================== #
#  Lévy flight (Mantegna)                                                  #
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
#  Food source (solution)                                                  #
# ====================================================================== #

@dataclass
class _FoodSource:
    position: Dict[str, float]
    objectives: Dict[str, float]  # thrust, pressure_loss, nozzle_length
    fitness: float = 0.0
    trial: int = 0   # consecutive non-improvement cycles


# ====================================================================== #
#  ABCOptimizer                                                            #
# ====================================================================== #

@dataclass
class ABCOptimizer:
    """Artificial Bee Colony optimiser with multi-objective Pareto archive.

    searchSpace keys
    ----------------
    bounds         : dict[str, (lo, hi)]
    colony_size    : int   (30)  — number of food sources (employed bees)
    iterations     : int   (30)  — foraging cycles
    limit          : int   (0)   — abandonment limit; 0 = auto (dim * colony_size / 2)
    levy_prob      : float (0.15) — Lévy flight probability for scouts
    levy_scale     : float (0.01) — Lévy step scale
    archive_size   : int   (100) — maximum Pareto archive size
    onlooker_ratio : float (1.0) — onlookers per employed bee
    w_thrust, w_pressure_loss  — scalarisation weights (for single-best tracking)
    """

    searchSpace: Dict[str, Any]
    objectiveFunc: Callable[[Dict[str, float]], EvaluationResult]
    algorithm: str = "abc"
    seed: int = 42

    _history: List[EvaluationResult] = field(default_factory=list)
    _history_params: List[Dict[str, float]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False,
                                  repr=False, compare=False)
    _counter: Any = field(default=None, init=False, repr=False, compare=False)

    # Diagnostic traces (compatible with SwarmPlots)
    _gbest_trace: List[float] = field(default_factory=list)
    _inertia_trace: List[float] = field(default_factory=list)   # not used by ABC, stores abandonment rate
    _diversity_trace: List[float] = field(default_factory=list)
    _particle_traces: List[List[Dict[str, float]]] = field(default_factory=list)
    _archive_size_trace: List[int] = field(default_factory=list)

    # Pareto archive
    archive: List[Dict[str, float]] = field(default_factory=list)

    # Three-objective definition
    _OBJ_NAMES: List[str] = field(default_factory=lambda: ["thrust", "pressure_loss", "nozzle_length"])
    _OBJ_DIRS: List[int] = field(default_factory=lambda: [+1, -1, -1])

    def __post_init__(self):
        self._counter = itertools.count(0)

    # ---- objective helpers ------------------------------------------------

    def _wt(self) -> float:
        return float(self.searchSpace.get("w_thrust", 0.7))

    def _wl(self) -> float:
        return float(self.searchSpace.get("w_pressure_loss", 0.3))

    def _scalar_score(self, thrust: float, loss: float,
                      length: float = 0.0) -> float:
        """Weighted scalarisation for single-best tracking."""
        # Normalise length contribution: shorter is better
        length_ref = float(self.searchSpace.get("bounds", {}).get("length", [0.18, 0.28])[1])
        length_penalty = 0.1 * (length / max(length_ref, 1e-9))
        return self._wt() * thrust - self._wl() * loss * 1e3 - length_penalty

    def _abc_fitness(self, score: float) -> float:
        """Convert objective score to ABC fitness (always >= 0)."""
        if score >= 0:
            return 1.0 + score
        return 1.0 / (1.0 + abs(score))

    # ---- search-space helpers ---------------------------------------------

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
            futs = {pool.submit(self.objectiveFunc, p): (i, p)
                    for i, p in enumerate(candidates)}
            for f in as_completed(futs):
                idx, p = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    res = EvaluationResult(
                        machProfile=[], pressureLoss=1.0, thrust=-1e30,
                        geometryId=f"abc_fail_{idx}",
                        metadata={"status": "failed", "error": str(e)},
                    )
                results_map[idx] = (res, p)
        out: List[EvaluationResult] = []
        with self._lock:
            for i in range(len(candidates)):
                r, p = results_map[i]
                self._history.append(r)
                self._history_params.append(dict(p))
                out.append(r)
        return out

    # ---- archive management -----------------------------------------------

    def _update_archive(self, entry: Dict[str, float]) -> None:
        max_sz = int(self.searchSpace.get("archive_size", 100))
        self.archive = [
            a for a in self.archive
            if not _dominates(entry, a, self._OBJ_NAMES, self._OBJ_DIRS)
        ]
        if not any(_dominates(a, entry, self._OBJ_NAMES, self._OBJ_DIRS)
                   for a in self.archive):
            self.archive.append(entry)
        if len(self.archive) > max_sz:
            cd = _crowding_distance(self.archive, self._OBJ_NAMES)
            paired = sorted(zip(cd, range(len(self.archive))), reverse=True)
            keep = sorted([p[1] for p in paired[:max_sz]])
            self.archive = [self.archive[i] for i in keep]

    def _select_archive_guide(self, rng: random.Random) -> Optional[Dict[str, float]]:
        """Select a guide from archive using roulette on crowding distance."""
        if not self.archive:
            return None
        cd = _crowding_distance(self.archive, self._OBJ_NAMES)
        total = sum(d for d in cd if not math.isinf(d))
        if total <= 0 or any(math.isinf(d) for d in cd):
            return dict(rng.choice(self.archive))
        probs = [d / total for d in cd]
        r = rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return dict(self.archive[i])
        return dict(self.archive[-1])

    # ---- opposition-based initialisation ----------------------------------

    @staticmethod
    def _opposition(pos: Dict[str, float],
                    bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Opposition-based learning: mirror position through search-space centre."""
        return {k: (lo + hi) - pos[k] for k, (lo, hi) in bounds.items()}

    # ---- core ABC ---------------------------------------------------------

    def _run_abc(self, rng: random.Random) -> None:
        bounds = self._bounds()
        keys = list(bounds.keys())
        n_dims = len(keys)
        colony = int(self.searchSpace.get("colony_size",
                      self.searchSpace.get("swarm_size", 30)))
        n_iter = int(self.searchSpace.get("iterations", 30))
        limit = int(self.searchSpace.get("limit", 0))
        if limit <= 0:
            limit = max(n_dims * colony // 2, colony)
        levy_prob = float(self.searchSpace.get("levy_prob", 0.15))
        levy_scale = float(self.searchSpace.get("levy_scale", 0.01))
        onlooker_ratio = float(self.searchSpace.get("onlooker_ratio", 1.0))
        n_onlookers = max(1, int(colony * onlooker_ratio))

        # ---- Phase 0: Opposition-based initialisation ----
        init_pos: List[Dict[str, float]] = []
        for _ in range(colony):
            pos = {k: rng.uniform(*bounds[k]) for k in keys}
            init_pos.append(pos)

        # Generate opposition set and keep the best colony from 2*colony
        opp_pos = [self._opposition(p, bounds) for p in init_pos]
        all_init = init_pos + opp_pos
        all_results = self._evaluate_batch(all_init)

        # Score and pick top colony
        scored = []
        for i, (pos, res) in enumerate(zip(all_init, all_results)):
            t = float(res.thrust)
            l = float(res.pressureLoss)
            nozzle_len = float(pos.get("length", 0.0))
            sc = self._scalar_score(t, l, nozzle_len)
            obj = {"thrust": t, "pressure_loss": l, "nozzle_length": nozzle_len}
            scored.append((sc, i, pos, obj))
        scored.sort(key=lambda x: x[0], reverse=True)

        sources: List[_FoodSource] = []
        for rank, (sc, idx, pos, obj) in enumerate(scored[:colony]):
            fit = self._abc_fitness(sc)
            sources.append(_FoodSource(
                position=dict(pos), objectives=dict(obj),
                fitness=fit, trial=0,
            ))
            entry = {**pos, **obj}
            self._update_archive(entry)

        best_scalar = max(s.fitness for s in sources)
        self._gbest_trace.append(best_scalar)
        self._inertia_trace.append(0.0)  # abandonment rate
        self._particle_traces.append([dict(s.position) for s in sources])
        self._diversity_trace.append(self._diversity(sources, keys, bounds))
        self._archive_size_trace.append(len(self.archive))

        # ---- Iterative foraging cycles ----
        for it in range(n_iter):
            abandoned_count = 0

            # ============ EMPLOYED BEE PHASE ============
            emp_positions: List[Dict[str, float]] = []
            emp_indices: List[int] = []

            for i, src in enumerate(sources):
                # Pick a random partner (different from i)
                partner = i
                while partner == i and colony > 1:
                    partner = rng.randint(0, colony - 1)

                # Generate candidate solution
                dim = rng.choice(keys)
                phi = rng.uniform(-1, 1)
                new_pos = dict(src.position)
                new_pos[dim] = src.position[dim] + phi * (
                    src.position[dim] - sources[partner].position[dim]
                )

                # Archive-guided perturbation on a second dimension
                guide = self._select_archive_guide(rng)
                if guide and rng.random() < 0.3:
                    dim2 = rng.choice(keys)
                    if dim2 in guide:
                        new_pos[dim2] = new_pos[dim2] + 0.5 * rng.uniform(-1, 1) * (
                            guide[dim2] - new_pos[dim2]
                        )

                new_pos = self._clamp(new_pos)
                emp_positions.append(new_pos)
                emp_indices.append(i)

            emp_results = self._evaluate_batch(emp_positions)

            for j, (new_pos, res) in enumerate(zip(emp_positions, emp_results)):
                i = emp_indices[j]
                t = float(res.thrust)
                l = float(res.pressureLoss)
                nozzle_len = float(new_pos.get("length", 0.0))
                obj = {"thrust": t, "pressure_loss": l, "nozzle_length": nozzle_len}
                sc = self._scalar_score(t, l, nozzle_len)
                new_fit = self._abc_fitness(sc)

                # Greedy selection (multi-objective aware)
                if _dominates(obj, sources[i].objectives,
                              self._OBJ_NAMES, self._OBJ_DIRS) or new_fit > sources[i].fitness:
                    sources[i].position = dict(new_pos)
                    sources[i].objectives = dict(obj)
                    sources[i].fitness = new_fit
                    sources[i].trial = 0
                else:
                    sources[i].trial += 1

                entry = {**new_pos, **obj}
                self._update_archive(entry)

            # ============ ONLOOKER BEE PHASE ============
            # Fitness-proportional (roulette wheel) selection
            total_fit = sum(s.fitness for s in sources)
            if total_fit <= 0:
                probs = [1.0 / colony] * colony
            else:
                probs = [s.fitness / total_fit for s in sources]

            onl_positions: List[Dict[str, float]] = []
            onl_indices: List[int] = []

            for _ in range(n_onlookers):
                # Roulette selection
                r = rng.random()
                cum = 0.0
                selected = 0
                for si, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        selected = si
                        break

                src = sources[selected]
                partner = selected
                while partner == selected and colony > 1:
                    partner = rng.randint(0, colony - 1)

                dim = rng.choice(keys)
                phi = rng.uniform(-1, 1)
                new_pos = dict(src.position)
                new_pos[dim] = src.position[dim] + phi * (
                    src.position[dim] - sources[partner].position[dim]
                )
                new_pos = self._clamp(new_pos)
                onl_positions.append(new_pos)
                onl_indices.append(selected)

            onl_results = self._evaluate_batch(onl_positions)

            for j, (new_pos, res) in enumerate(zip(onl_positions, onl_results)):
                i = onl_indices[j]
                t = float(res.thrust)
                l = float(res.pressureLoss)
                nozzle_len = float(new_pos.get("length", 0.0))
                obj = {"thrust": t, "pressure_loss": l, "nozzle_length": nozzle_len}
                sc = self._scalar_score(t, l, nozzle_len)
                new_fit = self._abc_fitness(sc)

                if _dominates(obj, sources[i].objectives,
                              self._OBJ_NAMES, self._OBJ_DIRS) or new_fit > sources[i].fitness:
                    sources[i].position = dict(new_pos)
                    sources[i].objectives = dict(obj)
                    sources[i].fitness = new_fit
                    sources[i].trial = 0
                else:
                    sources[i].trial += 1

                entry = {**new_pos, **obj}
                self._update_archive(entry)

            # ============ SCOUT BEE PHASE ============
            for i, src in enumerate(sources):
                if src.trial >= limit:
                    abandoned_count += 1
                    # Lévy-augmented random exploration
                    new_pos: Dict[str, float] = {}
                    for k in keys:
                        lo, hi = bounds[k]
                        if rng.random() < levy_prob:
                            step = _levy_step(rng) * levy_scale * (hi - lo)
                            new_pos[k] = src.position[k] + step
                        else:
                            new_pos[k] = rng.uniform(lo, hi)
                    new_pos = self._clamp(new_pos)

                    scout_res = self._evaluate([new_pos])[0] if False else None
                    # Evaluate single scout
                    result = self.objectiveFunc(new_pos)
                    with self._lock:
                        self._history.append(result)
                        self._history_params.append(dict(new_pos))

                    t = float(result.thrust)
                    l = float(result.pressureLoss)
                    nozzle_len = float(new_pos.get("length", 0.0))
                    obj = {"thrust": t, "pressure_loss": l, "nozzle_length": nozzle_len}
                    sc = self._scalar_score(t, l, nozzle_len)

                    sources[i] = _FoodSource(
                        position=dict(new_pos), objectives=dict(obj),
                        fitness=self._abc_fitness(sc), trial=0,
                    )
                    entry = {**new_pos, **obj}
                    self._update_archive(entry)

            # ---- Diagnostics ----
            best_fit = max(s.fitness for s in sources)
            best_scalar = max(best_scalar, best_fit)
            aband_rate = abandoned_count / max(colony, 1)

            self._gbest_trace.append(best_scalar)
            self._inertia_trace.append(aband_rate)
            self._particle_traces.append([dict(s.position) for s in sources])
            self._diversity_trace.append(self._diversity(sources, keys, bounds))
            self._archive_size_trace.append(len(self.archive))

    @staticmethod
    def _diversity(sources: List[_FoodSource], keys: List[str],
                   bounds: Dict[str, Tuple[float, float]]) -> float:
        if len(sources) <= 1:
            return 0.0
        n = len(sources)
        t = 0.0
        for k in keys:
            vals = [s.position[k] for s in sources]
            m = sum(vals) / n
            var = sum((v - m) ** 2 for v in vals) / n
            sp = bounds[k][1] - bounds[k][0]
            t += math.sqrt(var) / max(sp, 1e-12)
        return t / len(keys)

    # ---- public API (compatible with SwarmOptimizer) ----------------------

    def _best_idx(self) -> int:
        return max(
            range(len(self._history)),
            key=lambda i: self._scalar_score(
                float(self._history[i].thrust),
                float(self._history[i].pressureLoss),
                float(self._history_params[i].get("length", 0.0)),
            ),
        )

    def _objective_score(self, result: EvaluationResult,
                         reference: Optional[List[EvaluationResult]] = None) -> float:
        """Compatibility method for generate_swarm_plots."""
        if result.metadata.get("status") == "failed":
            return -1.0
        ref = reference or self._history or [result]
        thrusts = [float(r.thrust) for r in ref]
        lo, hi = min(thrusts), max(thrusts)
        tn = (float(result.thrust) - lo) / max(hi - lo, 1e-12)
        ln = max(0.0, min(1.0, float(result.pressureLoss)))
        wt, wl = self._wt(), self._wl()
        t = max(wt + wl, 1e-9)
        return (wt / t) * tn + (wl / t) * (1.0 - ln)

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
            self._archive_size_trace = []
            self.archive = []

        self._run_abc(rng)

        if not self._history:
            raise RuntimeError("ABCOptimizer: no evaluations")
        idx = self._best_idx()
        best = self._history[idx]
        best.metadata["best_params"] = self._history_params[idx]
        best.metadata["algorithm"] = "abc"
        best.metadata["archive_size"] = len(self.archive)
        best.metadata["pareto_archive"] = self.archive
        return best

    def logBest(self) -> None:
        if not self._history:
            return
        idx = self._best_idx()
        b = self._history[idx]
        print(
            f"[ABC] best thrust={b.thrust:.3f} N  loss={b.pressureLoss:.5f}  "
            f"archive={len(self.archive)}  params={self._history_params[idx]}"
        )
