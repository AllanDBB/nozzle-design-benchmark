from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, List

import matplotlib.pyplot as plt

from geometry import NozzleGeometry


def _get_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _legend_unique() -> None:
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2 = []
    l2 = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    if l2:
        ax.legend(h2, l2)


def generate_optimization_plots(
    history: Iterable[Any],
    out_dir: str,
    moc_thrust: Optional[float] = None,
    moc_pressure_loss: Optional[float] = None,
    population: Optional[int] = None,
    moc_geometry: Optional[NozzleGeometry] = None,
    throat_radius: Optional[float] = None,
    n_points: int = 180,
    profile: str = "bezier_like",
    gamma: float = 1.4,
    gas_constant: float = 287.0,
) -> None:
    records = list(history)
    if not records:
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thrusts: List[float] = [float(_get_field(r, "thrust", 0.0)) for r in records]
    losses: List[float] = [float(_get_field(r, "pressureLoss", 1.0)) for r in records]

    best_thrust: List[float] = []
    current = thrusts[0]
    for t in thrusts:
        current = max(current, t)
        best_thrust.append(current)

    best_loss: List[float] = []
    current_l = losses[0]
    for l in losses:
        current_l = min(current_l, l)
        best_loss.append(current_l)

    xs = list(range(len(records)))
    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, thrusts, color="tab:blue", alpha=0.45, linewidth=1.0, label="Candidate thrust")
    plt.plot(xs, best_thrust, color="tab:orange", linewidth=2.0, label="Best-so-far thrust")
    if moc_thrust is not None:
        plt.axhline(float(moc_thrust), color="tab:green", linestyle="--", linewidth=1.3, label="MOC thrust")
    if population and population > 0:
        for g in range(population, len(xs), population):
            plt.axvline(g - 0.5, color="gray", alpha=0.15, linewidth=0.8)
    plt.xlabel("Candidate index")
    plt.ylabel("Thrust [N]")
    plt.title("Optimization Convergence (Thrust)")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "optimization_thrust_history.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, losses, color="tab:red", alpha=0.45, linewidth=1.0, label="Candidate pressure loss")
    plt.plot(xs, best_loss, color="tab:purple", linewidth=2.0, label="Best-so-far pressure loss")
    if moc_pressure_loss is not None:
        plt.axhline(float(moc_pressure_loss), color="tab:green", linestyle="--", linewidth=1.3, label="MOC pressure loss")
    if population and population > 0:
        for g in range(population, len(xs), population):
            plt.axvline(g - 0.5, color="gray", alpha=0.15, linewidth=0.8)
    plt.xlabel("Candidate index")
    plt.ylabel("Pressure loss [-]")
    plt.title("Optimization Convergence (Pressure Loss)")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "optimization_pressure_loss_history.png", dpi=180)
    plt.close()

    # Best-so-far by candidate for exit Mach and exit velocity.
    best_exit_mach: List[float] = []
    best_exit_vel: List[float] = []
    m_cur = float(_get_field(records[0], "machProfile", [0.0])[-1] if _get_field(records[0], "machProfile", []) else 0.0)
    t_cur = float(_get_field(records[0], "temperatureProfile", [300.0])[-1] if _get_field(records[0], "temperatureProfile", []) else 300.0)
    v_cur = m_cur * (gamma * gas_constant * max(t_cur, 1e-9)) ** 0.5
    best_idx_so_far: List[int] = []
    idx_cur = 0
    t_best = thrusts[0]
    for i, r in enumerate(records):
        t = float(_get_field(r, "thrust", -1e30))
        if t >= t_best:
            t_best = t
            idx_cur = i
            mp = _get_field(r, "machProfile", []) or []
            tp = _get_field(r, "temperatureProfile", []) or []
            m_cur = float(mp[-1]) if mp else 0.0
            t_cur = float(tp[-1]) if tp else 300.0
            v_cur = m_cur * (gamma * gas_constant * max(t_cur, 1e-9)) ** 0.5
        best_idx_so_far.append(idx_cur)
        best_exit_mach.append(m_cur)
        best_exit_vel.append(v_cur)

    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, best_exit_mach, color="tab:cyan", linewidth=2.0, label="Best-so-far exit Mach")
    if population and population > 0:
        for g in range(population, len(xs), population):
            plt.axvline(g - 0.5, color="gray", alpha=0.15, linewidth=0.8)
    plt.xlabel("Candidate index")
    plt.ylabel("Exit Mach [-]")
    plt.title("Best-so-Far Exit Mach")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "optimization_best_so_far_exit_mach.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, best_exit_vel, color="tab:green", linewidth=2.0, label="Best-so-far exit velocity")
    if population and population > 0:
        for g in range(population, len(xs), population):
            plt.axvline(g - 0.5, color="gray", alpha=0.15, linewidth=0.8)
    plt.xlabel("Candidate index")
    plt.ylabel("Exit velocity [m/s]")
    plt.title("Best-so-Far Exit Velocity")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "optimization_best_so_far_exit_velocity.png", dpi=180)
    plt.close()

    # Best-so-far profile snapshots (Mach/Velocity/Geometry) at milestones.
    milestones = sorted(set([0, len(records) // 4, len(records) // 2, (3 * len(records)) // 4, len(records) - 1]))

    plt.figure(figsize=(9, 4.5))
    for m in milestones:
        idx = best_idx_so_far[m]
        rec = records[idx]
        mach = _get_field(rec, "machProfile", []) or []
        if not mach:
            continue
        length = float(_get_field(rec, "params", {}).get("length", 1.0)) if isinstance(_get_field(rec, "params", {}), dict) else 1.0
        xprof = [i * length / max(len(mach) - 1, 1) for i in range(len(mach))]
        plt.plot(xprof, mach, linewidth=1.7, label=f"best@{m+1} (cand {idx+1})")
    plt.xlabel("x [m]")
    plt.ylabel("Mach [-]")
    plt.title("Best-so-Far Mach Profiles")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "optimization_best_so_far_mach_profiles.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    for m in milestones:
        idx = best_idx_so_far[m]
        rec = records[idx]
        mach = _get_field(rec, "machProfile", []) or []
        temp = _get_field(rec, "temperatureProfile", []) or []
        if not mach or not temp:
            continue
        n = min(len(mach), len(temp))
        vel = [float(mach[i]) * (gamma * gas_constant * max(float(temp[i]), 1e-9)) ** 0.5 for i in range(n)]
        length = float(_get_field(rec, "params", {}).get("length", 1.0)) if isinstance(_get_field(rec, "params", {}), dict) else 1.0
        xprof = [i * length / max(n - 1, 1) for i in range(n)]
        plt.plot(xprof, vel, linewidth=1.7, label=f"best@{m+1} (cand {idx+1})")
    plt.xlabel("x [m]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Best-so-Far Velocity Profiles")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "optimization_best_so_far_velocity_profiles.png", dpi=180)
    plt.close()

    can_plot_geometry = throat_radius is not None
    if can_plot_geometry:
        plt.figure(figsize=(9, 4.5))
        if moc_geometry is not None:
            xm = [p[0] for p in moc_geometry.control_points]
            ym = [p[1] for p in moc_geometry.control_points]
            plt.plot(xm, ym, color="black", linewidth=2.0, label="MOC")
        for m in milestones:
            idx = best_idx_so_far[m]
            params = _get_field(records[idx], "params", {})
            if not isinstance(params, dict):
                continue
            if not all(k in params for k in ("exit_radius", "length")):
                continue
            geom = NozzleGeometry.fromParams(
                {
                    "throat_radius": float(throat_radius),
                    "exit_radius": float(params["exit_radius"]),
                    "length": float(params["length"]),
                    "n_points": int(n_points),
                    "profile": profile,
                    "shape": float(params.get("shape", 1.8)),
                }
            )
            xg = [p[0] for p in geom.control_points]
            yg = [p[1] for p in geom.control_points]
            plt.plot(xg, yg, linewidth=1.6, label=f"best@{m+1} (cand {idx+1})")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Best-so-Far Geometry Evolution")
        plt.grid(True, alpha=0.25)
        _legend_unique()
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(out / "optimization_best_so_far_geometry.png", dpi=180)
        plt.close()
