"""Comparative visualisations for the multi-algorithm CI ensemble.

Produces publication-quality figures that compare PSO, MOPSO-LF,
Firefly, and ABC results side-by-side:

  1. Algorithm Race — best thrust convergence overlay (per iteration)
  2. Pareto Front Overlay — thrust vs pressure loss, colour by algorithm
  3. Diversity Comparison — swarm spread over iterations
  4. Search-space Exploration Heatmap — algorithm footprint
  5. Box plots — thrust/loss distributions per algorithm
  6. Radar chart — multi-metric algorithm fingerprint
  7. Summary table — comparison image
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# Colour palette for algorithms
ALGO_COLOURS = {
    "pso": "#1f77b4",
    "mopso_lf": "#d62728",
    "firefly": "#2ca02c",
    "abc": "#ff7f0e",
}
ALGO_LABELS = {
    "pso": "PSO",
    "mopso_lf": "MOPSO-LF",
    "firefly": "Firefly",
    "abc": "ABC",
}


def _algo_colour(name: str) -> str:
    return ALGO_COLOURS.get(name, "#7f7f7f")


def _algo_label(name: str) -> str:
    return ALGO_LABELS.get(name, name.upper())


# ====================================================================== #
#  1) Algorithm Race — convergence overlay                                 #
# ====================================================================== #

def plot_algorithm_race(
    algo_results: List[Any],
    out_dir: str,
    *,
    moc_thrust: Optional[float] = None,
    dpi: int = 200,
) -> None:
    """Best-thrust convergence curves, one line per algorithm.

    Instead of using the internal ``gbest_trace`` (which stores
    algorithm-specific composite scores that are *not* comparable
    across PSO / MOPSO / Firefly / ABC), this reconstructs the
    best-so-far thrust per iteration from the evaluation history.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for ar in algo_results:
        if not ar.history:
            continue
        # Determine swarm_size from the traces or fall back to heuristics
        n_iters = len(ar.gbest_trace) if ar.gbest_trace else 1
        n_evals = len(ar.history)
        swarm_size = max(1, round(n_evals / max(n_iters, 1)))

        # Build best-so-far thrust per iteration
        best_thrust: list[float] = []
        running_best = -1e30
        for it in range(n_iters):
            start = it * swarm_size
            end = min(start + swarm_size, n_evals)
            for j in range(start, end):
                t_j = float(ar.history[j].thrust)
                if t_j > running_best:
                    running_best = t_j
            best_thrust.append(running_best)

        iters = list(range(len(best_thrust)))
        colour = _algo_colour(ar.algorithm)
        label = _algo_label(ar.algorithm)
        ax.plot(iters, best_thrust, "o-", color=colour, linewidth=2.0,
                markersize=4, label=label, alpha=0.85)

    if moc_thrust is not None:
        ax.axhline(moc_thrust, color="gray", linestyle="--", linewidth=1.2,
                    label="MOC baseline", alpha=0.6)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Thrust [N]", fontsize=11)
    ax.set_title("Algorithm Race — Convergence Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "ensemble_convergence_race.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  2) Pareto Front Overlay                                                 #
# ====================================================================== #

def plot_pareto_overlay(
    algo_results: List[Any],
    out_dir: str,
    *,
    moc_thrust: Optional[float] = None,
    moc_loss: Optional[float] = None,
    dpi: int = 200,
) -> None:
    """Thrust vs Pressure Loss scatter, colour by algorithm."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    for ar in algo_results:
        thrusts = [float(r.thrust) for r in ar.history]
        losses = [float(r.pressureLoss) for r in ar.history]
        colour = _algo_colour(ar.algorithm)
        label = _algo_label(ar.algorithm)
        ax.scatter(losses, thrusts, s=15, alpha=0.35, color=colour,
                   edgecolors="none", label=f"{label} (all)")

        # Highlight best
        ax.scatter(
            [float(ar.best_result.pressureLoss)],
            [float(ar.best_result.thrust)],
            s=120, marker="*", color=colour, edgecolors="black",
            linewidths=0.8, zorder=10, label=f"{label} best"
        )

    # MOC baseline
    if moc_thrust is not None and moc_loss is not None:
        ax.scatter([moc_loss], [moc_thrust], s=150, marker="D",
                   color="gold", edgecolors="black", linewidths=1.0,
                   zorder=11, label="MOC baseline")

    ax.set_xlabel("Pressure Loss [-]", fontsize=11)
    ax.set_ylabel("Thrust [N]", fontsize=11)
    ax.set_title("Pareto Front Overlay — All Algorithms", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(out / "ensemble_pareto_overlay.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  3) Diversity Comparison                                                 #
# ====================================================================== #

def plot_diversity_comparison(
    algo_results: List[Any],
    out_dir: str,
    *,
    dpi: int = 200,
) -> None:
    """Diversity trace for each algorithm on same axes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for ar in algo_results:
        trace = ar.diversity_trace
        if not trace:
            continue
        iters = list(range(len(trace)))
        colour = _algo_colour(ar.algorithm)
        label = _algo_label(ar.algorithm)
        ax.plot(iters, trace, "^-", color=colour, linewidth=1.8,
                markersize=4, label=label, alpha=0.8)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Normalised Diversity", fontsize=11)
    ax.set_title("Swarm Diversity — Algorithm Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "ensemble_diversity_comparison.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  4) Search-space Heatmap — 2D density per algorithm                      #
# ====================================================================== #

def plot_search_heatmap(
    algo_results: List[Any],
    bounds: Dict[str, Tuple[float, float]],
    out_dir: str,
    *,
    dpi: int = 200,
) -> None:
    """2D histograms showing where each algorithm explored (exit_radius vs length)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = list(bounds.keys())
    if len(keys) < 2:
        return

    kx, ky = keys[0], keys[1]  # exit_radius, length typically

    n_algo = len(algo_results)
    if n_algo == 0:
        return

    fig, axes = plt.subplots(1, n_algo, figsize=(5 * n_algo, 5), squeeze=False)

    for col, ar in enumerate(algo_results):
        ax = axes[0, col]
        xs = [p.get(kx, 0) for p in ar.history_params]
        ys = [p.get(ky, 0) for p in ar.history_params]

        if xs and ys:
            ax.hist2d(xs, ys, bins=15, cmap="YlOrRd",
                      range=[list(bounds[kx]), list(bounds[ky])])

        ax.set_xlabel(kx, fontsize=10)
        ax.set_ylabel(ky if col == 0 else "", fontsize=10)
        ax.set_title(_algo_label(ar.algorithm), fontsize=11, fontweight="bold",
                      color=_algo_colour(ar.algorithm))
        ax.set_xlim(bounds[kx])
        ax.set_ylim(bounds[ky])

    plt.suptitle("Search-Space Exploration Density", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "ensemble_search_heatmap.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  5) Box plots — thrust / loss distributions per algorithm                #
# ====================================================================== #

def plot_algorithm_boxplots(
    algo_results: List[Any],
    out_dir: str,
    *,
    moc_thrust: Optional[float] = None,
    moc_loss: Optional[float] = None,
    dpi: int = 200,
) -> None:
    """Side-by-side box plots for thrust and pressure loss."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not algo_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Thrust
    thrust_data = []
    labels = []
    colours = []
    for ar in algo_results:
        thrusts = [float(r.thrust) for r in ar.history if float(r.thrust) > -1e20]
        thrust_data.append(thrusts)
        labels.append(_algo_label(ar.algorithm))
        colours.append(_algo_colour(ar.algorithm))

    bp1 = ax1.boxplot(thrust_data, labels=labels, patch_artist=True,
                       widths=0.5, showmeans=True,
                       meanprops=dict(marker="D", markeredgecolor="black",
                                      markerfacecolor="gold", markersize=6))
    for patch, c in zip(bp1["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    if moc_thrust is not None:
        ax1.axhline(moc_thrust, color="gray", linestyle="--", alpha=0.6,
                     label="MOC baseline")
        ax1.legend(fontsize=8)

    ax1.set_ylabel("Thrust [N]", fontsize=11)
    ax1.set_title("Thrust Distribution", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    # Pressure loss
    loss_data = []
    for ar in algo_results:
        losses = [float(r.pressureLoss) for r in ar.history if float(r.pressureLoss) < 0.99]
        loss_data.append(losses)

    bp2 = ax2.boxplot(loss_data, labels=labels, patch_artist=True,
                       widths=0.5, showmeans=True,
                       meanprops=dict(marker="D", markeredgecolor="black",
                                      markerfacecolor="gold", markersize=6))
    for patch, c in zip(bp2["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    if moc_loss is not None:
        ax2.axhline(moc_loss, color="gray", linestyle="--", alpha=0.6,
                     label="MOC baseline")
        ax2.legend(fontsize=8)

    ax2.set_ylabel("Pressure Loss [-]", fontsize=11)
    ax2.set_title("Pressure Loss Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Algorithm Performance — Box Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "ensemble_boxplots.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  6) Radar Chart — multi-metric fingerprint                               #
# ====================================================================== #

def plot_algorithm_radar(
    algo_results: List[Any],
    out_dir: str,
    *,
    dpi: int = 200,
) -> None:
    """Radar (spider) chart comparing algorithms across multiple metrics."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not algo_results:
        return

    # Compute metrics per algorithm
    metrics_names = [
        "Best Thrust",
        "Min Loss",
        "Mean Thrust",
        "Exploration",
        "Convergence Speed",
    ]
    n_metrics = len(metrics_names)
    algo_data = {}

    all_thrusts_max = max(
        (float(ar.best_result.thrust) for ar in algo_results),
        default=1.0
    )
    all_loss_min = min(
        (float(ar.best_result.pressureLoss) for ar in algo_results),
        default=0.0
    )

    for ar in algo_results:
        thrusts = [float(r.thrust) for r in ar.history if float(r.thrust) > -1e20]
        mean_t = sum(thrusts) / max(len(thrusts), 1)

        # Normalise metrics to [0, 1]
        best_t = float(ar.best_result.thrust) / max(all_thrusts_max, 1e-12)
        min_l = 1.0 - float(ar.best_result.pressureLoss)  # higher is better
        mean_t_norm = mean_t / max(all_thrusts_max, 1e-12)

        # Exploration = final diversity (higher = explored more)
        div = ar.diversity_trace[-1] if ar.diversity_trace else 0.0
        exploration = min(div * 5, 1.0)  # scale to ~[0,1]

        # Convergence speed = how quickly best thrust approached its final value
        #   Use thrust from history (comparable) not internal score
        if ar.history and len(ar.history) > 2:
            n_it = len(ar.gbest_trace) if ar.gbest_trace else 1
            n_ev = len(ar.history)
            sw = max(1, round(n_ev / max(n_it, 1)))
            thrust_curve = []
            rbest = -1e30
            for it in range(n_it):
                s = it * sw
                e = min(s + sw, n_ev)
                for j in range(s, e):
                    tj = float(ar.history[j].thrust)
                    if tj > rbest:
                        rbest = tj
                thrust_curve.append(rbest)
            final = thrust_curve[-1]
            half_target = (thrust_curve[0] + final) / 2.0
            speed_idx = 0
            for i, v in enumerate(thrust_curve):
                if v >= half_target:
                    speed_idx = i
                    break
            conv_speed = 1.0 - speed_idx / max(len(thrust_curve) - 1, 1)
        else:
            conv_speed = 0.5

        algo_data[ar.algorithm] = [best_t, min_l, mean_t_norm, exploration, conv_speed]

    # Plot
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for algo_name, values in algo_data.items():
        vals = values + values[:1]  # close polygon
        colour = _algo_colour(algo_name)
        label = _algo_label(algo_name)
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=colour)
        ax.fill(angles, vals, alpha=0.15, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Algorithm Fingerprint — Radar Chart", fontsize=13,
                  fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "ensemble_radar.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  7) Summary table as image                                               #
# ====================================================================== #

def plot_summary_table(
    algo_results: List[Any],
    out_dir: str,
    *,
    moc_thrust: Optional[float] = None,
    moc_loss: Optional[float] = None,
    dpi: int = 200,
) -> None:
    """Render the ensemble summary as a table image."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not algo_results:
        return

    columns = ["Algorithm", "Thrust [N]", "Loss", "Evals", "Time [s]", "Delta MOC"]
    rows = []
    for ar in algo_results:
        t = float(ar.best_result.thrust)
        l = float(ar.best_result.pressureLoss)
        delta = ""
        if moc_thrust is not None:
            pct = 100.0 * (t - moc_thrust) / max(abs(moc_thrust), 1e-12)
            delta = f"{pct:+.3f}%"
        rows.append([
            _algo_label(ar.algorithm),
            f"{t:.3f}",
            f"{l:.5f}",
            str(ar.n_evals),
            f"{ar.elapsed_s:.1f}",
            delta,
        ])

    fig, ax = plt.subplots(figsize=(10, 1 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Colour header
    for j, col in enumerate(columns):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight winner row
    winner_idx = max(range(len(algo_results)),
                     key=lambda i: float(algo_results[i].best_result.thrust))
    for j in range(len(columns)):
        table[winner_idx + 1, j].set_facecolor("#E2EFDA")

    plt.title("Ensemble CI — Algorithm Comparison", fontsize=13,
              fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out / "ensemble_summary_table.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# ====================================================================== #
#  Master function — generate all comparative plots                        #
# ====================================================================== #

def generate_ensemble_plots(
    algo_results: List[Any],
    out_dir: str,
    bounds: Dict[str, Tuple[float, float]],
    *,
    moc_thrust: Optional[float] = None,
    moc_loss: Optional[float] = None,
) -> int:
    """Generate the full suite of ensemble comparative plots.

    Returns the number of plots generated.
    """
    out = Path(out_dir) / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    d = str(out)
    count = 0

    try:
        plot_algorithm_race(algo_results, d, moc_thrust=moc_thrust)
        count += 1
    except Exception as e:
        print(f"[EnsemblePlots] algorithm_race failed: {e}")

    try:
        plot_pareto_overlay(algo_results, d, moc_thrust=moc_thrust, moc_loss=moc_loss)
        count += 1
    except Exception as e:
        print(f"[EnsemblePlots] pareto_overlay failed: {e}")

    try:
        plot_diversity_comparison(algo_results, d)
        count += 1
    except Exception as e:
        print(f"[EnsemblePlots] diversity_comparison failed: {e}")

    try:
        plot_search_heatmap(algo_results, bounds, d)
        count += 1
    except Exception as e:
        print(f"[EnsemblePlots] search_heatmap failed: {e}")

    try:
        plot_algorithm_boxplots(algo_results, d, moc_thrust=moc_thrust, moc_loss=moc_loss)
        count += 1
    except Exception as e:
        print(f"[EnsemblePlots] boxplots failed: {e}")

    try:
        plot_algorithm_radar(algo_results, d)
        count += 1
    except Exception as e:
        print(f"[EnsemblePlots] radar failed: {e}")

    try:
        plot_summary_table(algo_results, d, moc_thrust=moc_thrust, moc_loss=moc_loss)
        count += 1
    except Exception as e:
        print(f"[EnsemblePlots] summary_table failed: {e}")

    return count
