"""Multi-Objective Particle Swarm Optimisation with Lévy Flights (MOPSO-LF).

A true multi-objective PSO that maintains an external Pareto archive with
crowding-distance selection.  Incorporates:

  - **Lévy flights** for occasional long-range jumps (escape local optima)
  - **Chaos-based initialisation** (logistic map → better coverage)
  - **Adaptive inertia** (w_max → w_min linear decay)
  - **Constriction factor χ** (Clerc & Kennedy 2002)
  - **Turbulence operator** — re-inject worst 10 % particles every K iters

References
----------
Coello Coello, C.A. & Lechuga, M.S. (2002) - MOPSO
Deb, K. et al. (2002)             - Crowding distance (NSGA-II)
Yang, X.-S. (2010)                - Lévy-flight random walks
Mantegna, R.A. (1994)             - Mantegna algorithm for stable distributions
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
#  Pareto archive helpers                                                  #
# ====================================================================== #

def _dominates(a: Dict[str, float], b: Dict[str, float],
               objectives: List[str], directions: List[int]) -> bool:
    """True if *a* Pareto-dominates *b*.  direction=+1 maximise, -1 minimise."""
    dominated = False
    for obj, d in zip(objectives, directions):
        va = d * a[obj]
        vb = d * b[obj]
        if va < vb:
            return False
        if va > vb:
            dominated = True
    return dominated


def _crowding_distance(archive: List[Dict[str, float]],
                       objectives: List[str]) -> List[float]:
    """Compute NSGA-II crowding distance for the archive."""
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
#  Lévy flight (Mantegna algorithm)                                        #
# ====================================================================== #

def _levy_step(rng: random.Random, beta: float = 1.5) -> float:
    """Single Lévy-stable random step via Mantegna's algorithm."""
    sigma_u = (
        math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = rng.gauss(0, sigma_u)
    v = rng.gauss(0, 1.0)
    return u / (abs(v) ** (1.0 / beta))


# ====================================================================== #
#  Chaos-based initialisation                                              #
# ====================================================================== #

def _logistic_sequence(n: int, x0: float = 0.37, r: float = 3.999) -> List[float]:
    """Logistic-map chaotic sequence in (0, 1)."""
    seq = [x0]
    for _ in range(n - 1):
        x0 = r * x0 * (1.0 - x0)
        seq.append(x0)
    return seq


# ====================================================================== #
#  Particle                                                                #
# ====================================================================== #

@dataclass
class _MOParticle:
    position: Dict[str, float]
    velocity: Dict[str, float]
    best_position: Dict[str, float]
    # Multi-objective: personal-best stores objective values
    best_objectives: Dict[str, float] = field(default_factory=dict)


# ====================================================================== #
#  MOPSO optimiser                                                         #
# ====================================================================== #

@dataclass
class MOPSOOptimizer:
    """Multi-Objective PSO with Lévy flights and Pareto archive.

    searchSpace keys
    ----------------
    bounds        : dict[str, (lo, hi)]
    swarm_size    : int   (30)
    iterations    : int   (15)
    archive_size  : int   (100)
    levy_prob     : float (0.15)  — probability of Lévy jump per dim
    levy_scale    : float (0.01)  — Lévy step scaling
    turbulence_k  : int   (5)    — re-inject worst 10 % every K iters
    w_max, w_min, c1, c2, v_max_frac — same as standard PSO
    """

    searchSpace: Dict[str, Any]
    objectiveFunc: Callable[[Dict[str, float]], EvaluationResult]
    algorithm: str = "mopso_levy"
    seed: int = 42

    _history: List[EvaluationResult] = field(default_factory=list)
    _history_params: List[Dict[str, float]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False,
                                  repr=False, compare=False)
    _counter: Any = field(default=None, init=False, repr=False, compare=False)

    # Diagnostic traces
    _gbest_trace: List[float] = field(default_factory=list)
    _inertia_trace: List[float] = field(default_factory=list)
    _diversity_trace: List[float] = field(default_factory=list)
    _particle_traces: List[List[Dict[str, float]]] = field(default_factory=list)
    _archive_size_trace: List[int] = field(default_factory=list)

    # Pareto archive: [{param_k: v, "thrust": v, "pressure_loss": v}, ...]
    archive: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        self._counter = itertools.count(0)

    # ---- objective helpers ------------------------------------------------

    _OBJ_NAMES = ["thrust", "pressure_loss"]
    _OBJ_DIRS = [+1, -1]   # maximise thrust, minimise loss

    def _wt(self) -> float:
        return float(self.searchSpace.get("w_thrust", 0.7))

    def _wl(self) -> float:
        return float(self.searchSpace.get("w_pressure_loss", 0.3))

    def _scalar_score(self, thrust: float, loss: float) -> float:
        """Weighted scalarisation (for single-best tracking only)."""
        return self._wt() * thrust - self._wl() * loss * 1e3

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
            futs = {pool.submit(self.objectiveFunc, p): (i, p) for i, p in enumerate(candidates)}
            for f in as_completed(futs):
                idx, p = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    res = EvaluationResult(machProfile=[], pressureLoss=1.0,
                                           thrust=-1e30, geometryId=f"mopso_fail_{idx}",
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

    # ---- archive management -----------------------------------------------

    def _update_archive(self, entry: Dict[str, float]) -> None:
        max_sz = int(self.searchSpace.get("archive_size", 100))
        # Remove entries dominated by new one
        self.archive = [
            a for a in self.archive
            if not _dominates(entry, a, self._OBJ_NAMES, self._OBJ_DIRS)
        ]
        # Only add if not dominated by any current member
        if not any(_dominates(a, entry, self._OBJ_NAMES, self._OBJ_DIRS)
                   for a in self.archive):
            self.archive.append(entry)
        # Trim by crowding distance
        if len(self.archive) > max_sz:
            cd = _crowding_distance(self.archive, self._OBJ_NAMES)
            paired = sorted(zip(cd, range(len(self.archive))), reverse=True)
            keep = sorted([p[1] for p in paired[:max_sz]])
            self.archive = [self.archive[i] for i in keep]

    def _select_guide(self, rng: random.Random) -> Dict[str, float]:
        """Select a guide from archive using roulette on crowding distance."""
        if not self.archive:
            return {}
        cd = _crowding_distance(self.archive, self._OBJ_NAMES)
        total = sum(cd) if not any(math.isinf(d) for d in cd) else 0.0
        if total <= 0:
            return dict(rng.choice(self.archive))
        probs = [d / total for d in cd]
        r = rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return dict(self.archive[i])
        return dict(self.archive[-1])

    # ---- core MOPSO -------------------------------------------------------

    def _run_mopso(self, rng: random.Random) -> None:
        bounds = self._bounds()
        keys = list(bounds.keys())
        n_particles = int(self.searchSpace.get("swarm_size", 30))
        n_iter = int(self.searchSpace.get("iterations", 15))
        w_max = float(self.searchSpace.get("w_max", 0.9))
        w_min = float(self.searchSpace.get("w_min", 0.4))
        c1 = float(self.searchSpace.get("c1", 2.0))
        c2 = float(self.searchSpace.get("c2", 2.0))
        v_frac = float(self.searchSpace.get("v_max_frac", 0.25))
        levy_prob = float(self.searchSpace.get("levy_prob", 0.15))
        levy_scale = float(self.searchSpace.get("levy_scale", 0.01))
        turb_k = int(self.searchSpace.get("turbulence_k", 5))

        v_max = {k: v_frac * (hi - lo) for k, (lo, hi) in bounds.items()}

        phi = c1 + c2
        chi = 2.0 / abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi)) if phi > 4 else 1.0

        # ---- Chaos initialisation ----
        n_dims = len(keys)
        chaos_seqs = [_logistic_sequence(n_particles, x0=0.1 + 0.05 * d)
                      for d in range(n_dims)]
        init_positions: List[Dict[str, float]] = []
        for j in range(n_particles):
            pos = {}
            for d, k in enumerate(keys):
                lo, hi = bounds[k]
                pos[k] = lo + chaos_seqs[d][j] * (hi - lo)
            init_positions.append(pos)

        init_results = self._evaluate_batch(init_positions)

        swarm: List[_MOParticle] = []
        for pos, res in zip(init_positions, init_results):
            vel = {k: rng.uniform(-v_max[k], v_max[k]) for k in keys}
            obj = {"thrust": float(res.thrust), "pressure_loss": float(res.pressureLoss)}
            swarm.append(_MOParticle(
                position=dict(pos), velocity=vel,
                best_position=dict(pos), best_objectives=dict(obj),
            ))
            entry = {**pos, **obj}
            self._update_archive(entry)

        # Track best scalar score for diagnostics
        best_scalar = max(
            self._scalar_score(float(r.thrust), float(r.pressureLoss))
            for r in init_results
        )
        self._gbest_trace.append(best_scalar)
        self._inertia_trace.append(w_max)
        self._particle_traces.append([dict(p.position) for p in swarm])
        self._diversity_trace.append(self._diversity(swarm, keys, bounds))
        self._archive_size_trace.append(len(self.archive))

        # ---- Iterative loop ----
        for it in range(n_iter):
            w = w_max - (w_max - w_min) * it / max(n_iter - 1, 1)

            new_positions: List[Dict[str, float]] = []
            for p in swarm:
                guide = self._select_guide(rng)
                new_vel: Dict[str, float] = {}
                new_pos: Dict[str, float] = {}
                for k in keys:
                    lo, hi = bounds[k]
                    r1, r2 = rng.random(), rng.random()
                    cog = c1 * r1 * (p.best_position[k] - p.position[k])
                    soc = c2 * r2 * (guide.get(k, p.position[k]) - p.position[k])
                    vk = chi * (w * p.velocity[k] + cog + soc)
                    vk = max(-v_max[k], min(v_max[k], vk))

                    # Lévy flight mutation
                    if rng.random() < levy_prob:
                        step = _levy_step(rng) * levy_scale * (hi - lo)
                        vk += step

                    new_vel[k] = vk
                    new_pos[k] = p.position[k] + vk
                p.velocity = new_vel
                p.position = self._clamp(new_pos)
                new_positions.append(dict(p.position))

            # Turbulence: re-inject worst 10 % particles every turb_k iters
            if turb_k > 0 and (it + 1) % turb_k == 0:
                scores = [
                    self._scalar_score(p.best_objectives.get("thrust", 0),
                                       p.best_objectives.get("pressure_loss", 1))
                    for p in swarm
                ]
                n_reinject = max(1, n_particles // 10)
                worst_idx = sorted(range(n_particles), key=lambda i: scores[i])[:n_reinject]
                for wi in worst_idx:
                    new_positions[wi] = {
                        k: rng.uniform(*bounds[k]) for k in keys
                    }
                    swarm[wi].position = dict(new_positions[wi])

            iter_results = self._evaluate_batch(new_positions)

            for p, res in zip(swarm, iter_results):
                obj = {"thrust": float(res.thrust), "pressure_loss": float(res.pressureLoss)}
                entry = {**p.position, **obj}
                self._update_archive(entry)

                # Update personal best — dominated?
                if _dominates(obj, p.best_objectives, self._OBJ_NAMES, self._OBJ_DIRS):
                    p.best_objectives = dict(obj)
                    p.best_position = dict(p.position)
                elif not _dominates(p.best_objectives, obj, self._OBJ_NAMES, self._OBJ_DIRS):
                    # Non-dominated: keep with 50 % probability
                    if rng.random() < 0.5:
                        p.best_objectives = dict(obj)
                        p.best_position = dict(p.position)

            sc = max(
                self._scalar_score(float(r.thrust), float(r.pressureLoss))
                for r in iter_results
            )
            best_scalar = max(best_scalar, sc)
            self._gbest_trace.append(best_scalar)
            self._inertia_trace.append(w)
            self._particle_traces.append([dict(p.position) for p in swarm])
            self._diversity_trace.append(self._diversity(swarm, keys, bounds))
            self._archive_size_trace.append(len(self.archive))

    @staticmethod
    def _diversity(swarm, keys, bounds):
        if len(swarm) <= 1:
            return 0.0
        n = len(swarm)
        t = 0.0
        for k in keys:
            vals = [p.position[k] for p in swarm]
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
            ),
        )

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

        self._run_mopso(rng)

        if not self._history:
            raise RuntimeError("MOPSO did not evaluate any candidates")
        idx = self._best_idx()
        best = self._history[idx]
        best.metadata["best_params"] = self._history_params[idx]
        best.metadata["algorithm"] = "mopso_levy"
        best.metadata["archive_size"] = len(self.archive)
        best.metadata["pareto_archive"] = self.archive
        return best

    def logBest(self) -> None:
        if not self._history:
            return
        idx = self._best_idx()
        b = self._history[idx]
        print(
            f"[MOPSO-LF] best thrust={b.thrust:.3f} N  loss={b.pressureLoss:.5f}  "
            f"archive={len(self.archive)}  params={self._history_params[idx]}"
        )
