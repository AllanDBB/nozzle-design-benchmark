"""Swarm Intelligence specific visualisations.

Plots tailored to PSO diagnostics:
  - Global-best convergence per iteration
  - Inertia weight schedule
  - Swarm diversity (normalised spread)
  - Particle trajectory scatter in 2-D parameter slices
  - MLN nozzle geometry evolution with 50 MOC characteristic lines
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from geometry import NozzleGeometry, MinimumLengthNozzle, MOCSolver


# ====================================================================== #
#  Helpers                                                                 #
# ====================================================================== #

def _get_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _legend_unique(ax: Any = None) -> None:
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    seen: set = set()
    h2, l2 = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            h2.append(h)
            l2.append(lab)
    if l2:
        ax.legend(h2, l2, fontsize=8)


# ====================================================================== #
#  1) PSO convergence: global-best score per iteration                     #
# ====================================================================== #

def plot_gbest_convergence(
    gbest_trace: List[float],
    out_dir: str,
    *,
    dpi: int = 180,
) -> None:
    """Global-best objective score vs PSO iteration."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    iters = list(range(len(gbest_trace)))
    ax.plot(iters, gbest_trace, "o-", color="tab:blue", linewidth=2.0,
            markersize=5, label="Global best score")
    ax.set_xlabel("PSO Iteration")
    ax.set_ylabel("Objective score (higher = better)")
    ax.set_title("PSO Convergence — Global Best")
    ax.grid(True, alpha=0.25)
    _legend_unique(ax)
    plt.tight_layout()
    plt.savefig(out / "pso_gbest_convergence.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  2) Inertia weight schedule                                              #
# ====================================================================== #

def plot_inertia_schedule(
    inertia_trace: List[float],
    out_dir: str,
    *,
    dpi: int = 180,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    iters = list(range(len(inertia_trace)))
    ax.plot(iters, inertia_trace, "s-", color="tab:orange", linewidth=1.8,
            markersize=4, label="Inertia w")
    ax.set_xlabel("PSO Iteration")
    ax.set_ylabel("Inertia weight w")
    ax.set_title("Adaptive Inertia Schedule")
    ax.grid(True, alpha=0.25)
    _legend_unique(ax)
    plt.tight_layout()
    plt.savefig(out / "pso_inertia_schedule.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  3) Swarm diversity                                                      #
# ====================================================================== #

def plot_swarm_diversity(
    diversity_trace: List[float],
    out_dir: str,
    *,
    dpi: int = 180,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    iters = list(range(len(diversity_trace)))
    ax.plot(iters, diversity_trace, "^-", color="tab:green", linewidth=1.8,
            markersize=4, label="Normalised diversity")
    ax.set_xlabel("PSO Iteration")
    ax.set_ylabel("Diversity (avg normalised σ)")
    ax.set_title("Swarm Diversity Over Iterations")
    ax.grid(True, alpha=0.25)
    _legend_unique(ax)
    plt.tight_layout()
    plt.savefig(out / "pso_swarm_diversity.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  4) Particle trajectories in 2-D parameter slices                        #
# ====================================================================== #

def plot_particle_trajectories(
    particle_traces: List[List[Dict[str, float]]],
    bounds: Dict[str, Tuple[float, float]],
    out_dir: str,
    *,
    dpi: int = 180,
) -> None:
    """Scatter + trajectory lines for every 2-D slice of the parameter space."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = list(bounds.keys())
    n_iters = len(particle_traces)
    if n_iters < 2 or len(keys) < 2:
        return

    # Generate all 2-D combinations
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            kx, ky = keys[i], keys[j]
            fig, ax = plt.subplots(figsize=(7, 6))

            cmap = cm.get_cmap("viridis", n_iters)

            # Plot each particle's path across iterations
            n_particles = len(particle_traces[0])
            for p_idx in range(n_particles):
                xs_p = [particle_traces[t][p_idx][kx] for t in range(n_iters)]
                ys_p = [particle_traces[t][p_idx][ky] for t in range(n_iters)]
                ax.plot(xs_p, ys_p, color="gray", alpha=0.15, linewidth=0.5,
                        zorder=1)

            # Scatter coloured by iteration
            for t in range(n_iters):
                xs_t = [particle_traces[t][p][kx]
                        for p in range(len(particle_traces[t]))]
                ys_t = [particle_traces[t][p][ky]
                        for p in range(len(particle_traces[t]))]
                alpha = 0.3 + 0.7 * t / max(n_iters - 1, 1)
                size = 8 + 20 * t / max(n_iters - 1, 1)
                ax.scatter(xs_t, ys_t, s=size, alpha=alpha,
                           color=cmap(t / max(n_iters - 1, 1)),
                           zorder=2, edgecolors="none")

            # Bounds box
            ax.set_xlim(bounds[kx])
            ax.set_ylim(bounds[ky])
            ax.set_xlabel(kx)
            ax.set_ylabel(ky)
            ax.set_title(f"Particle Trajectories — {kx} vs {ky}")
            ax.grid(True, alpha=0.2)

            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=0, vmax=n_iters - 1),
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Iteration")

            plt.tight_layout()
            fname = f"pso_trajectories_{kx}_vs_{ky}.png"
            plt.savefig(out / fname, dpi=dpi)
            plt.close()


# ====================================================================== #
#  5) Thrust & pressure loss history (like GA plots, but per-particle)     #
# ====================================================================== #

def plot_swarm_history(
    history: Iterable[Any],
    out_dir: str,
    swarm_size: int,
    moc_thrust: Optional[float] = None,
    moc_pressure_loss: Optional[float] = None,
    *,
    dpi: int = 180,
) -> None:
    """Thrust / loss per candidate with swarm-iteration grid lines."""
    records = list(history)
    if not records:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thrusts = [float(_get_field(r, "thrust", 0.0)) for r in records]
    losses = [float(_get_field(r, "pressureLoss", 1.0)) for r in records]

    best_thrust: List[float] = []
    cur = thrusts[0]
    for t in thrusts:
        cur = max(cur, t)
        best_thrust.append(cur)

    best_loss: List[float] = []
    cur_l = losses[0]
    for l in losses:
        cur_l = min(cur_l, l)
        best_loss.append(cur_l)

    xs = list(range(len(records)))

    # -- Thrust --
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, thrusts, color="tab:blue", alpha=0.4, linewidth=0.8,
            label="Particle thrust")
    ax.plot(xs, best_thrust, color="tab:orange", linewidth=2.0,
            label="Global best thrust")
    if moc_thrust is not None:
        ax.axhline(float(moc_thrust), color="tab:green", linestyle="--",
                    linewidth=1.3, label="MOC baseline thrust")
    # Iteration separators
    if swarm_size > 0:
        for g in range(swarm_size, len(xs), swarm_size):
            ax.axvline(g - 0.5, color="gray", alpha=0.15, linewidth=0.8)
    ax.set_xlabel("Candidate index")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("PSO Convergence — Thrust History")
    ax.grid(True, alpha=0.25)
    _legend_unique(ax)
    plt.tight_layout()
    plt.savefig(out / "pso_thrust_history.png", dpi=dpi)
    plt.close()

    # -- Pressure loss --
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, losses, color="tab:red", alpha=0.4, linewidth=0.8,
            label="Particle pressure loss")
    ax.plot(xs, best_loss, color="tab:purple", linewidth=2.0,
            label="Global best pressure loss")
    if moc_pressure_loss is not None:
        ax.axhline(float(moc_pressure_loss), color="tab:green", linestyle="--",
                    linewidth=1.3, label="MOC baseline loss")
    if swarm_size > 0:
        for g in range(swarm_size, len(xs), swarm_size):
            ax.axvline(g - 0.5, color="gray", alpha=0.15, linewidth=0.8)
    ax.set_xlabel("Candidate index")
    ax.set_ylabel("Pressure loss [-]")
    ax.set_title("PSO Convergence — Pressure Loss History")
    ax.grid(True, alpha=0.25)
    _legend_unique(ax)
    plt.tight_layout()
    plt.savefig(out / "pso_pressure_loss_history.png", dpi=dpi)
    plt.close()


# ====================================================================== #
#  6) MLN nozzle with 50 MOC characteristic lines  (textbook style)        #
# ====================================================================== #

def plot_mln_with_characteristics(
    geometry: NozzleGeometry,
    mach_exit: float,
    pressure_ratio: float,
    gamma: float,
    savepath: str,
    *,
    n_char_lines: int = 50,
    title_suffix: str = "",
    dpi: int = 220,
) -> None:
    """Draw an MLN nozzle contour overlaid with *n_char_lines* MOC
    characteristic lines (C- fan + C+ reflections) — textbook style.

    Uses the full ``MOCSolver.plotCharacteristics`` machinery internally
    but forces the resolution to ``n_char_lines``.
    """
    solver = MOCSolver(
        machExit=mach_exit,
        pressureRatio=pressure_ratio,
        gamma=gamma,
        nCharacteristics=n_char_lines,
    )
    solver.plotCharacteristics(geometry, savepath, n_lines=n_char_lines, dpi=dpi)


# ====================================================================== #
#  7) Best-so-far nozzle geometry evolution with 50 char lines             #
# ====================================================================== #

def plot_geometry_evolution(
    history: Iterable[Any],
    out_dir: str,
    throat_radius: float,
    mach_exit: float,
    gamma: float,
    pressure_ratio: float,
    n_points: int = 180,
    moc_geometry: Optional[NozzleGeometry] = None,
    *,
    n_char_lines: int = 50,
    dpi: int = 220,
) -> None:
    """Best-so-far nozzle snapshots at milestones, each with 50 characteristic lines."""
    records = list(history)
    if not records:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find best-so-far index at each evaluation
    best_idx_so_far: List[int] = []
    t_best = float(_get_field(records[0], "thrust", -1e30))
    idx_cur = 0
    for i, r in enumerate(records):
        t = float(_get_field(r, "thrust", -1e30))
        if t >= t_best:
            t_best = t
            idx_cur = i
        best_idx_so_far.append(idx_cur)

    milestones = sorted(set([
        0,
        len(records) // 4,
        len(records) // 2,
        3 * len(records) // 4,
        len(records) - 1,
    ]))

    # Combined figure showing all milestone geometries
    fig, ax = plt.subplots(figsize=(13, 5.5))

    # MOC baseline
    if moc_geometry is not None:
        xm = [p[0] for p in moc_geometry.control_points]
        ym = [p[1] for p in moc_geometry.control_points]
        ax.plot(xm, ym, color="black", linewidth=2.2, label="MOC baseline",
                zorder=10)

    solver = MOCSolver(
        machExit=mach_exit,
        pressureRatio=pressure_ratio,
        gamma=gamma,
        nCharacteristics=n_char_lines,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for mi, m in enumerate(milestones):
        idx = best_idx_so_far[m]
        rec = records[idx]
        params = _get_field(rec, "params", {})
        if not isinstance(params, dict) or not params:
            md = _get_field(rec, "metadata", {})
            if isinstance(md, dict):
                params = md.get("params", md.get("best_params", {}))
        if not isinstance(params, dict) or not params:
            continue
        if not all(k in params for k in ("exit_radius", "length")):
            continue

        try:
            mln = MinimumLengthNozzle(
                mach_exit=mach_exit,
                throat_radius=throat_radius,
                exit_radius=float(params["exit_radius"]),
                length=float(params["length"]),
                gamma=gamma,
                n_points=n_points,
                straighten_frac=float(params.get("straighten_frac", 0.45)),
            )
            geom = mln.build()
        except Exception:
            continue

        xg = [p[0] for p in geom.control_points]
        yg = [p[1] for p in geom.control_points]
        clr = colors[mi % len(colors)]
        ax.plot(xg, yg, linewidth=1.6, color=clr,
                label=f"best@eval {m + 1}", zorder=8)

        # Overlay characteristic lines (lightweight — just C+ from axis)
        _overlay_char_lines(ax, geom, mach_exit, gamma,
                            n_lines=n_char_lines, alpha=0.12, color=clr)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("PSO Geometry Evolution — MLN with Characteristic Lines")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    _legend_unique(ax)
    plt.tight_layout()
    plt.savefig(out / "pso_geometry_evolution.png", dpi=dpi)
    plt.close()

    # Also save individual best nozzle with full 50-line characteristic plot
    best_final_idx = best_idx_so_far[-1]
    best_rec = records[best_final_idx]
    params = _get_field(best_rec, "params", {})
    if not isinstance(params, dict) or not params:
        md = _get_field(best_rec, "metadata", {})
        if isinstance(md, dict):
            params = md.get("params", md.get("best_params", {}))
    if isinstance(params, dict) and all(
        k in params for k in ("exit_radius", "length")
    ):
        try:
            mln = MinimumLengthNozzle(
                mach_exit=mach_exit,
                throat_radius=throat_radius,
                exit_radius=float(params["exit_radius"]),
                length=float(params["length"]),
                gamma=gamma,
                n_points=n_points,
                straighten_frac=float(params.get("straighten_frac", 0.45)),
            )
            best_geom = mln.build()
            # Build a matching MOC geometry so the char network is consistent
            opt_moc_geom = solver.generateGeometry({
                "throat_y": throat_radius,
                "exit_y": float(params["exit_radius"]),
                "length": float(params["length"]),
                "n_points": n_points,
            })
            plot_mln_with_characteristics(
                opt_moc_geom,
                mach_exit=mach_exit,
                pressure_ratio=pressure_ratio,
                gamma=gamma,
                savepath=str(out / "pso_best_nozzle_50_characteristics.png"),
                n_char_lines=n_char_lines,
                dpi=dpi,
            )
        except Exception:
            pass


def _overlay_char_lines(
    ax: Any,
    geometry: NozzleGeometry,
    mach_exit: float,
    gamma: float = 1.4,
    n_lines: int = 50,
    alpha: float = 0.12,
    color: str = "tab:blue",
) -> None:
    """Lightweight C+ characteristic overlay on an existing axes."""
    x_end = geometry.length
    throat = geometry.throat_radius
    n_steps = max(n_lines * 10, 200)
    xs_fine = [i * x_end / n_steps for i in range(n_steps + 1)]

    def _mach_sup(ar: float) -> float:
        if ar <= 1.0:
            return 1.0
        lo, hi = 1.0, 20.0
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            fac = 1.0 + gm1 / 2.0 * mid * mid
            a = (2.0 / gp1 * fac) ** (gp1 / (2.0 * gm1)) / mid
            if a > ar:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    for j in range(n_lines):
        x0 = j * x_end / max(n_lines - 1, 1)
        yw0 = geometry.y_at(x0)
        ar0 = (yw0 / max(throat, 1e-9)) ** 2
        m0 = _mach_sup(max(ar0, 1.0))
        mu0 = math.asin(min(1.0, 1.0 / max(m0, 1.001)))
        dyw = (geometry.y_at(min(x0 + 1e-4, x_end))
               - geometry.y_at(max(x0 - 1e-4, 0.0))) / 2e-4
        theta0 = math.atan(max(dyw, 0.0))
        slope = math.tan(theta0 + mu0)
        rx = [x0]
        ry = [0.0]
        for x in xs_fine:
            if x <= x0:
                continue
            y = slope * (x - x0)
            yw = geometry.y_at(x)
            if y >= yw:
                rx.append(x)
                ry.append(yw)
                break
            rx.append(x)
            ry.append(y)
        if len(rx) > 1:
            ax.plot(rx, ry, color=color, alpha=alpha, linewidth=0.5, zorder=2)


# ====================================================================== #
#  Master function — generate all SI plots                                 #
# ====================================================================== #

def generate_swarm_plots(
    optimizer: Any,
    history: Iterable[Any],
    out_dir: str,
    swarm_size: int,
    moc_thrust: Optional[float] = None,
    moc_pressure_loss: Optional[float] = None,
    moc_geometry: Optional[NozzleGeometry] = None,
    throat_radius: Optional[float] = None,
    mach_exit: float = 2.0,
    gamma: float = 1.4,
    pressure_ratio: float = 0.08,
    n_points: int = 180,
    n_char_lines: int = 50,
) -> None:
    """Generate the full suite of PSO diagnostic plots."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1-3) PSO diagnostics from optimizer traces
    if hasattr(optimizer, "_gbest_trace") and optimizer._gbest_trace:
        plot_gbest_convergence(optimizer._gbest_trace, str(out))
    if hasattr(optimizer, "_inertia_trace") and optimizer._inertia_trace:
        plot_inertia_schedule(optimizer._inertia_trace, str(out))
    if hasattr(optimizer, "_diversity_trace") and optimizer._diversity_trace:
        plot_swarm_diversity(optimizer._diversity_trace, str(out))

    # 4) Particle trajectories
    if hasattr(optimizer, "_particle_traces") and optimizer._particle_traces:
        bounds = optimizer.searchSpace.get("bounds", {})
        plot_particle_trajectories(
            optimizer._particle_traces, bounds, str(out)
        )

    # 5) Thrust / loss history
    plot_swarm_history(
        history, str(out), swarm_size,
        moc_thrust=moc_thrust,
        moc_pressure_loss=moc_pressure_loss,
    )

    # 6-7) Geometry evolution with 50 characteristic lines
    if throat_radius is not None:
        plot_geometry_evolution(
            history, str(out),
            throat_radius=throat_radius,
            mach_exit=mach_exit,
            gamma=gamma,
            pressure_ratio=pressure_ratio,
            n_points=n_points,
            moc_geometry=moc_geometry,
            n_char_lines=n_char_lines,
        )
