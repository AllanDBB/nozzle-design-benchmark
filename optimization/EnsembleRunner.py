"""Ensemble Meta-Optimiser — runs multiple CI algorithms and merges results.

Runs up to four swarm-intelligence algorithms in sequence (or a
user-selected subset), merges all candidate histories, and selects the
overall best via 3-objective Pareto dominance.

This is the "make it fly" module: each algorithm explores the design space
with different dynamics, and the ensemble picks the winner.

Supported algorithms
--------------------
  pso       — Standard PSO with adaptive inertia (SwarmOptimizer)
  mopso_lf  — Multi-Objective PSO with Lévy flights, 3 objectives (MOPSO)
  firefly   — Firefly Algorithm with Lévy perturbation
  abc       — Artificial Bee Colony with Pareto archive, 3 objectives

Multi-objective selection
------------------------
The winner is chosen by 3-objective Pareto dominance (thrust ↑,
pressure_loss ↓, nozzle_length ↓) with a scalarised tie-break.

Usage
-----
    runner = EnsembleRunner(searchSpace=..., objectiveFunc=...,
                            algorithms=["pso", "mopso_lf", "firefly", "abc"])
    result = runner.run()
    runner.summary()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from evaluators import EvaluationResult


# ====================================================================== #
#  Pareto dominance helper                                                 #
# ====================================================================== #

def _dominates_objs(a: Dict[str, float], b: Dict[str, float],
                    objectives: List[str], directions: List[int]) -> bool:
    """True if *a* Pareto-dominates *b*. direction=+1 maximise, -1 minimise."""
    dom = False
    for obj, d in zip(objectives, directions):
        va = d * a.get(obj, 0.0)
        vb = d * b.get(obj, 0.0)
        if va < vb:
            return False
        if va > vb:
            dom = True
    return dom


# ====================================================================== #
#  Algorithm registry                                                      #
# ====================================================================== #

_REGISTRY: Dict[str, type] = {}


def _ensure_registry():
    global _REGISTRY
    if _REGISTRY:
        return
    from optimization.SwarmOptimizer import SwarmOptimizer
    from optimization.MOPSO import MOPSOOptimizer
    from optimization.FireflyOptimizer import FireflyOptimizer
    from optimization.ABCOptimizer import ABCOptimizer
    _REGISTRY = {
        "pso": SwarmOptimizer,
        "mopso_lf": MOPSOOptimizer,
        "firefly": FireflyOptimizer,
        "abc": ABCOptimizer,
    }


# ====================================================================== #
#  Per-algorithm result bundle                                             #
# ====================================================================== #

@dataclass
class AlgorithmResult:
    """Stores results from one algorithm run."""
    algorithm: str
    best_result: EvaluationResult
    best_params: Dict[str, float]
    history: List[EvaluationResult]
    history_params: List[Dict[str, float]]
    elapsed_s: float
    n_evals: int
    gbest_trace: List[float]
    diversity_trace: List[float]
    inertia_trace: List[float]
    particle_traces: List[List[Dict[str, float]]]


# ====================================================================== #
#  EnsembleRunner                                                          #
# ====================================================================== #

@dataclass
class EnsembleRunner:
    """Meta-optimiser that runs multiple CI algorithms and merges results.

    Parameters
    ----------
    searchSpace : dict
        Same format as SwarmOptimizer (bounds, swarm_size, iterations, ...).
        Algorithm-specific keys are passed through.
    objectiveFunc : callable
        params dict -> EvaluationResult
    algorithms : list[str]
        Which algorithms to run.  Default: all three.
    seed : int
        Base seed.  Each algorithm gets seed + offset.
    """

    searchSpace: Dict[str, Any]
    objectiveFunc: Callable[[Dict[str, float]], EvaluationResult]
    algorithms: List[str] = field(default_factory=lambda: ["pso", "mopso_lf", "firefly", "abc"])
    seed: int = 42

    # Outputs
    _algo_results: List[AlgorithmResult] = field(default_factory=list)
    _best_algorithm: str = ""
    _best_result: Optional[EvaluationResult] = None
    _best_params: Dict[str, float] = field(default_factory=dict)
    _all_history: List[EvaluationResult] = field(default_factory=list)
    _all_history_params: List[Dict[str, float]] = field(default_factory=list)

    def run(self) -> EvaluationResult:
        """Run all algorithms sequentially and return the overall best."""
        _ensure_registry()

        self._algo_results = []
        self._all_history = []
        self._all_history_params = []
        seed_offset = 0

        for algo_name in self.algorithms:
            if algo_name not in _REGISTRY:
                print(f"[Ensemble] WARNING: unknown algorithm '{algo_name}', skipping")
                continue

            algo_cls = _REGISTRY[algo_name]
            algo_seed = self.seed + seed_offset
            seed_offset += 1000

            # Create a copy of search space for this algorithm
            ss = dict(self.searchSpace)

            print(f"\n[Ensemble] === {algo_name.upper()} (seed={algo_seed}) ===")
            t_start = time.time()

            optimizer = algo_cls(
                searchSpace=ss,
                objectiveFunc=self.objectiveFunc,
                algorithm=algo_name,
                seed=algo_seed,
            )

            try:
                best = optimizer.run()
            except Exception as exc:
                print(f"[Ensemble] {algo_name} FAILED: {exc}")
                continue

            elapsed = time.time() - t_start
            n_evals = len(optimizer._history)

            best_idx = optimizer._best_idx()
            best_params = optimizer._history_params[best_idx]

            ar = AlgorithmResult(
                algorithm=algo_name,
                best_result=best,
                best_params=dict(best_params),
                history=list(optimizer._history),
                history_params=list(optimizer._history_params),
                elapsed_s=round(elapsed, 2),
                n_evals=n_evals,
                gbest_trace=list(getattr(optimizer, "_gbest_trace", [])),
                diversity_trace=list(getattr(optimizer, "_diversity_trace", [])),
                inertia_trace=list(getattr(optimizer, "_inertia_trace", [])),
                particle_traces=list(getattr(optimizer, "_particle_traces", [])),
            )
            self._algo_results.append(ar)
            self._all_history.extend(ar.history)
            self._all_history_params.extend(ar.history_params)

            print(
                f"[Ensemble] {algo_name}: F={best.thrust:.3f} N  "
                f"loss={best.pressureLoss:.5f}  evals={n_evals}  "
                f"time={elapsed:.1f}s"
            )

        if not self._algo_results:
            raise RuntimeError("EnsembleRunner: all algorithms failed")

        # Select overall best via Pareto dominance across all algorithm bests.
        # Among non-dominated candidates, pick by weighted scalarisation.
        bests = [
            {
                "thrust": float(a.best_result.thrust),
                "pressure_loss": float(a.best_result.pressureLoss),
                "nozzle_length": float(a.best_params.get("length", 0.0)),
                "idx": i,
            }
            for i, a in enumerate(self._algo_results)
        ]
        obj_names = ["thrust", "pressure_loss", "nozzle_length"]
        obj_dirs = [+1, -1, -1]
        nondom = []
        for i, bi in enumerate(bests):
            dominated = any(
                _dominates_objs(bj, bi, obj_names, obj_dirs)
                for j, bj in enumerate(bests) if j != i
            )
            if not dominated:
                nondom.append(bi)
        if not nondom:
            nondom = bests
        # Scalarised tie-break among non-dominated
        wt = float(self.searchSpace.get("w_thrust", 0.7))
        wl = float(self.searchSpace.get("w_pressure_loss", 0.3))
        winner_entry = max(
            nondom,
            key=lambda b: wt * b["thrust"] - wl * b["pressure_loss"] * 1e3
                          - 0.1 * b["nozzle_length"],
        )
        overall_best_ar = self._algo_results[winner_entry["idx"]]
        self._best_algorithm = overall_best_ar.algorithm
        self._best_result = overall_best_ar.best_result
        self._best_params = dict(overall_best_ar.best_params)

        self._best_result.metadata["ensemble_winner"] = self._best_algorithm
        self._best_result.metadata["ensemble_algorithms"] = self.algorithms
        self._best_result.metadata["best_params"] = dict(self._best_params)
        self._best_result.metadata["nondominated_algorithms"] = [
            self._algo_results[b["idx"]].algorithm for b in nondom
        ]

        return self._best_result

    # ---- Accessors --------------------------------------------------------

    @property
    def algo_results(self) -> List[AlgorithmResult]:
        return self._algo_results

    @property
    def best_algorithm(self) -> str:
        return self._best_algorithm

    @property
    def best_params(self) -> Dict[str, float]:
        return dict(self._best_params)

    @property
    def total_evals(self) -> int:
        return sum(a.n_evals for a in self._algo_results)

    # ---- Compatibility with SwarmPlots ------------------------------------

    @property
    def _history(self) -> List[EvaluationResult]:
        """Merged history for backward compatibility with generate_swarm_plots."""
        return self._all_history

    @property
    def _history_params(self) -> List[Dict[str, float]]:
        return self._all_history_params

    @property
    def _gbest_trace(self) -> List[float]:
        """Return the best winner's trace."""
        for ar in self._algo_results:
            if ar.algorithm == self._best_algorithm:
                return ar.gbest_trace
        return []

    @property
    def _diversity_trace(self) -> List[float]:
        for ar in self._algo_results:
            if ar.algorithm == self._best_algorithm:
                return ar.diversity_trace
        return []

    @property
    def _inertia_trace(self) -> List[float]:
        for ar in self._algo_results:
            if ar.algorithm == self._best_algorithm:
                return ar.inertia_trace
        return []

    @property
    def _particle_traces(self) -> List[List[Dict[str, float]]]:
        for ar in self._algo_results:
            if ar.algorithm == self._best_algorithm:
                return ar.particle_traces
        return []

    def _best_idx(self) -> int:
        """Index into _all_history of the overall best candidate."""
        if not self._all_history:
            return 0
        return max(
            range(len(self._all_history)),
            key=lambda i: (
                float(self._all_history[i].thrust),
                -float(self._all_history[i].pressureLoss),
            ),
        )

    def _objective_score(self, result: EvaluationResult) -> float:
        """Simple scalar score for compatibility."""
        wt = float(self.searchSpace.get("w_thrust", 0.7))
        wl = float(self.searchSpace.get("w_pressure_loss", 0.3))
        return wt * float(result.thrust) - wl * float(result.pressureLoss) * 1e3

    # ---- Summary ----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of all algorithm results."""
        rows = []
        for ar in self._algo_results:
            rows.append({
                "algorithm": ar.algorithm,
                "thrust_N": round(float(ar.best_result.thrust), 4),
                "pressure_loss": round(float(ar.best_result.pressureLoss), 6),
                "nozzle_length_mm": round(float(ar.best_params.get("length", 0.0)) * 1000, 1),
                "n_evals": ar.n_evals,
                "elapsed_s": ar.elapsed_s,
                "params": ar.best_params,
            })
        nondom_algos = []
        if self._best_result and self._best_result.metadata:
            nondom_algos = self._best_result.metadata.get("nondominated_algorithms", [])
        return {
            "algorithms_run": [ar.algorithm for ar in self._algo_results],
            "winner": self._best_algorithm,
            "winner_thrust_N": round(float(self._best_result.thrust), 4) if self._best_result else 0.0,
            "total_evaluations": self.total_evals,
            "nondominated_algorithms": nondom_algos,
            "per_algorithm": rows,
        }

    def print_summary(self) -> None:
        """Print a formatted summary table."""
        s = self.summary()
        sep = "-" * 80
        print(f"\n{sep}")
        print("  ENSEMBLE CI RESULTS - ALGORITHM COMPARISON (3-objective Pareto)")
        print(sep)
        hdr = (f"  {'Algorithm':<14s}  {'Thrust [N]':>12s}  {'Loss':>10s}  "
               f"{'Length':>8s}  {'Evals':>6s}  {'Time':>7s}")
        print(hdr)
        print(f"  {'-' * 14}  {'-' * 12}  {'-' * 10}  {'-' * 8}  {'-' * 6}  {'-' * 7}")
        nondom = s.get("nondominated_algorithms", [])
        for row in s["per_algorithm"]:
            tag = ""
            if row["algorithm"] == s["winner"]:
                tag = " *"
            elif row["algorithm"] in nondom:
                tag = " ~"  # non-dominated but not winner
            print(
                f"  {row['algorithm']:<14s}  {row['thrust_N']:>12.4f}  "
                f"{row['pressure_loss']:>10.6f}  "
                f"{row['nozzle_length_mm']:>7.1f}  "
                f"{row['n_evals']:>6d}  "
                f"{row['elapsed_s']:>6.1f}s{tag}"
            )
        print(sep)
        print(f"  Winner: {s['winner']}  |  Non-dominated: {', '.join(nondom)}")
        print(f"  Total evals: {s['total_evaluations']}  |  (* = winner, ~ = Pareto non-dominated)")
        print(sep)
