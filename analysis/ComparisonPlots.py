from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import math

import matplotlib.pyplot as plt

from geometry import NozzleGeometry
from evaluators import EvaluationResult


def _x_from_profile(length: float, n: int) -> List[float]:
    if n <= 1:
        return [0.0]
    return [i * length / (n - 1) for i in range(n)]


def _x_from_result(length: float, profile_len: int, x_profile: List[float]) -> List[float]:
    if x_profile and len(x_profile) == profile_len:
        return x_profile
    return _x_from_profile(length, profile_len)


def _velocity_profile(mach: List[float], temp: List[float], gamma: float, gas_constant: float) -> List[float]:
    if not mach or not temp:
        return []
    n = min(len(mach), len(temp))
    vel: List[float] = []
    for i in range(n):
        a = math.sqrt(max(gamma * gas_constant * max(temp[i], 1e-9), 1e-9))
        vel.append(mach[i] * a)
    return vel


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


def _draw_char_lines(
    ax: Any,
    geometry: NozzleGeometry,
    mach_exit: float,
    gamma: float = 1.4,
    n_lines: int = 18,
    alpha: float = 0.28,
    color: str = "tab:blue",
) -> None:
    """Overlay MOC-style C+ right-running characteristics on *ax*."""
    x_end = geometry.length
    throat = geometry.throat_radius
    n_steps = max(n_lines * 14, 200)
    xs_fine = [i * x_end / n_steps for i in range(n_steps + 1)]

    def _mach_sup(area_ratio: float) -> float:
        if area_ratio <= 1.0:
            return 1.0
        lo, hi = 1.0, 20.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            fac = 1.0 + (gamma - 1.0) / 2.0 * mid * mid
            ar = ((2.0 / (gamma + 1.0)) * fac) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))) / mid
            if ar > area_ratio:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    for j in range(n_lines):
        x0 = j * x_end / max(n_lines - 1, 1)
        y0 = 0.0
        yw0 = geometry.y_at(x0)
        ar0 = (yw0 / max(throat, 1e-9)) ** 2
        m0 = _mach_sup(max(ar0, 1.0))
        mu0 = math.asin(min(1.0, 1.0 / max(m0, 1.001)))
        dyw = (geometry.y_at(min(x0 + 1e-4, x_end)) - geometry.y_at(max(x0 - 1e-4, 0.0))) / 2e-4
        theta0 = math.atan(max(dyw, 0.0))
        slope = math.tan(theta0 + mu0)
        rx: List[float] = [x0]
        ry: List[float] = [y0]
        for x in xs_fine:
            if x <= x0:
                continue
            y = y0 + slope * (x - x0)
            yw = geometry.y_at(x)
            if y >= yw:
                rx.append(x)
                ry.append(yw)
                break
            rx.append(x)
            ry.append(y)
        if len(rx) > 1:
            ax.plot(rx, ry, color=color, alpha=alpha, linewidth=0.75, zorder=2)


def _winner_badge(ax: Any, text: str, bg: str = "limegreen", fg: str = "darkgreen") -> None:
    """Stamp a coloured winner/loser tag in the upper-right of *ax*."""
    ax.text(
        0.98, 0.97, text,
        transform=ax.transAxes,
        fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, alpha=0.35, edgecolor=fg),
        color=fg,
        fontweight="bold",
    )


def generate_comparison_plots(
    moc_geometry: NozzleGeometry,
    optimized_geometry: NozzleGeometry,
    moc_result: EvaluationResult,
    optimized_result: EvaluationResult,
    out_dir: str,
    solver_config: Dict[str, Any],
    mach_exit: float = 2.0,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gamma = float(solver_config.get("gamma", 1.4))
    gas_constant = float(solver_config.get("gas_constant", 287.0))

    # 1) Geometry overlay + MOC characteristic lines
    fig_g, ax_g = plt.subplots(figsize=(9, 4.5))
    xm = [p[0] for p in moc_geometry.control_points]
    ym = [p[1] for p in moc_geometry.control_points]
    xo = [p[0] for p in optimized_geometry.control_points]
    yo = [p[1] for p in optimized_geometry.control_points]
    # Nozzle interior shading (MLN style — upper wall only).
    ax_g.fill_between(xm, 0, ym, color="tab:blue", alpha=0.08)
    ax_g.fill_between(xo, 0, yo, color="tab:orange", alpha=0.08)
    # Characteristic lines behind the walls
    _draw_char_lines(ax_g, moc_geometry, mach_exit, gamma, n_lines=16, alpha=0.22, color="tab:blue")
    _draw_char_lines(ax_g, optimized_geometry, mach_exit, gamma, n_lines=16, alpha=0.22, color="tab:orange")
    ax_g.plot(xm, ym, label="MOC", color="tab:blue", linewidth=2.0, zorder=5)
    ax_g.plot(xo, yo, label="OPT", color="tab:orange", linewidth=2.0, zorder=5)
    ax_g.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_g.set_xlabel("x [m]")
    ax_g.set_ylabel("r [m]")
    ax_g.set_title("Geometry Comparison + Characteristic Lines (MLN style)")
    ax_g.set_ylim(bottom=-0.002)
    ax_g.set_aspect("equal", adjustable="datalim")
    ax_g.grid(True, alpha=0.3)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "compare_geometry.png", dpi=180)
    plt.close()

    # 2) Mach profile comparison
    if moc_result.machProfile and optimized_result.machProfile:
        xmprof = _x_from_result(moc_geometry.length, len(moc_result.machProfile), moc_result.xProfile)
        xoprof = _x_from_result(optimized_geometry.length, len(optimized_result.machProfile), optimized_result.xProfile)
        fig_ma, ax_ma = plt.subplots(figsize=(8, 4))
        ax_ma.plot(xmprof, moc_result.machProfile, label="MOC", color="tab:blue")
        ax_ma.plot(xoprof, optimized_result.machProfile, label="OPT", color="tab:orange")
        ax_ma.set_xlabel("x [m]")
        ax_ma.set_ylabel("Mach  (higher = better expansion)")
        ax_ma.set_title("Mach Along Nozzle")
        ax_ma.grid(True, alpha=0.3)
        _m_moc = moc_result.machProfile[-1]
        _m_opt = optimized_result.machProfile[-1]
        if _m_opt >= _m_moc:
            _winner_badge(ax_ma,
                f"\u25b2 OPT  Me={_m_opt:.2f} (+{(_m_opt - _m_moc) / max(abs(_m_moc), 1e-9) * 100:.1f}%)",
                "bisque", "darkorange")
        else:
            _winner_badge(ax_ma,
                f"\u25b2 MOC  Me={_m_moc:.2f} (+{(_m_moc - _m_opt) / max(abs(_m_opt), 1e-9) * 100:.1f}%)",
                "lightblue", "navy")
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "compare_mach.png", dpi=180)
        plt.close()

    # 3) Velocity profile comparison (requested)
    vm = moc_result.velocityProfile or _velocity_profile(moc_result.machProfile, moc_result.temperatureProfile, gamma, gas_constant)
    vo = optimized_result.velocityProfile or _velocity_profile(optimized_result.machProfile, optimized_result.temperatureProfile, gamma, gas_constant)
    if vm and vo:
        xv_m = _x_from_result(moc_geometry.length, len(vm), moc_result.xProfile)
        xv_o = _x_from_result(optimized_geometry.length, len(vo), optimized_result.xProfile)
        fig_v, ax_v = plt.subplots(figsize=(8, 4))
        ax_v.plot(xv_m, vm, label="MOC", color="tab:blue")
        ax_v.plot(xv_o, vo, label="OPT", color="tab:orange")
        ax_v.set_xlabel("x [m]")
        ax_v.set_ylabel("Velocity [m/s]  (higher = more thrust)")
        ax_v.set_title("Velocity Along Nozzle")
        ax_v.grid(True, alpha=0.3)
        _v_moc = vm[-1] if vm else 0.0
        _v_opt = vo[-1] if vo else 0.0
        if _v_opt >= _v_moc:
            _winner_badge(ax_v,
                f"\u25b2 OPT  Ve={_v_opt:.0f} m/s (+{(_v_opt - _v_moc) / max(abs(_v_moc), 1e-9) * 100:.1f}%)",
                "bisque", "darkorange")
        else:
            _winner_badge(ax_v,
                f"\u25b2 MOC  Ve={_v_moc:.0f} m/s (+{(_v_moc - _v_opt) / max(abs(_v_opt), 1e-9) * 100:.1f}%)",
                "lightblue", "navy")
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "compare_velocity.png", dpi=180)
        plt.close()

    # 4) Pressure profile comparison
    if moc_result.pressureProfile and optimized_result.pressureProfile:
        xp_m = _x_from_result(moc_geometry.length, len(moc_result.pressureProfile), moc_result.xProfile)
        xp_o = _x_from_result(optimized_geometry.length, len(optimized_result.pressureProfile), optimized_result.xProfile)
        fig_p, ax_p = plt.subplots(figsize=(8, 4))
        ax_p.plot(xp_m, moc_result.pressureProfile, label="MOC", color="tab:blue")
        ax_p.plot(xp_o, optimized_result.pressureProfile, label="OPT", color="tab:orange")
        ax_p.set_xlabel("x [m]")
        ax_p.set_ylabel("Pressure [Pa]  (lower exit = better expansion)")
        ax_p.set_title("Pressure Along Nozzle")
        ax_p.grid(True, alpha=0.3)
        _p_moc = moc_result.pressureProfile[-1]
        _p_opt = optimized_result.pressureProfile[-1]
        # lower exit pressure = flow expanded more = better
        if _p_opt <= _p_moc:
            _winner_badge(ax_p,
                f"\u25bc OPT  Pe={_p_opt/1000:.1f} kPa (better expanded)",
                "bisque", "darkorange")
        else:
            _winner_badge(ax_p,
                f"\u25bc MOC  Pe={_p_moc/1000:.1f} kPa (better expanded)",
                "lightblue", "navy")
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "compare_pressure.png", dpi=180)
        plt.close()

    # 5) Temperature profile comparison
    if moc_result.temperatureProfile and optimized_result.temperatureProfile:
        xt_m = _x_from_result(moc_geometry.length, len(moc_result.temperatureProfile), moc_result.xProfile)
        xt_o = _x_from_result(optimized_geometry.length, len(optimized_result.temperatureProfile), optimized_result.xProfile)
        fig_t, ax_t = plt.subplots(figsize=(8, 4))
        ax_t.plot(xt_m, moc_result.temperatureProfile, label="MOC", color="tab:blue")
        ax_t.plot(xt_o, optimized_result.temperatureProfile, label="OPT", color="tab:orange")
        ax_t.set_xlabel("x [m]")
        ax_t.set_ylabel("Temperature [K]  (lower exit = more KE extracted)")
        ax_t.set_title("Temperature Along Nozzle")
        ax_t.grid(True, alpha=0.3)
        _t_moc = moc_result.temperatureProfile[-1]
        _t_opt = optimized_result.temperatureProfile[-1]
        # lower exit temperature = more thermal energy converted to kinetic = better
        if _t_opt <= _t_moc:
            _winner_badge(ax_t,
                f"\u25bc OPT  Te={_t_opt:.0f} K (more energy extracted)",
                "bisque", "darkorange")
        else:
            _winner_badge(ax_t,
                f"\u25bc MOC  Te={_t_moc:.0f} K (more energy extracted)",
                "lightblue", "navy")
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "compare_temperature.png", dpi=180)
        plt.close()

    # 6) Performance comparison — two bar charts + overall winner verdict.
    _thrust_winner = "OPT" if optimized_result.thrust > moc_result.thrust else "MOC"
    _loss_winner = "OPT" if optimized_result.pressureLoss < moc_result.pressureLoss else "MOC"
    _dt_pct = abs(optimized_result.thrust - moc_result.thrust) / max(abs(moc_result.thrust), 1e-9) * 100
    _dl_pct = abs(optimized_result.pressureLoss - moc_result.pressureLoss) / max(abs(moc_result.pressureLoss), 1e-9) * 100

    if _thrust_winner == _loss_winner:
        _trophy = f"[WINNER] {_thrust_winner} wins both metrics"
        _verdict_color = "darkgreen"
    elif _thrust_winner == "OPT":
        _trophy = "[TRADEOFF] OPT thrust higher, MOC pressure loss lower"
        _verdict_color = "darkorange"
    else:
        _trophy = "[TRADEOFF] MOC thrust higher, OPT pressure loss lower"
        _verdict_color = "darkorange"

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    fig.subplots_adjust(top=0.80)

    ax_thrust, ax_loss = axes
    bar_colors = ["tab:blue", "tab:orange"]

    ax_thrust.bar(["MOC", "OPT"], [moc_result.thrust, optimized_result.thrust], color=bar_colors)
    ax_thrust.set_ylabel("Thrust [N]")
    ax_thrust.set_title(f"Thrust  (\u25b2 {_thrust_winner} +{_dt_pct:.1f}%)")
    ax_thrust.grid(True, axis="y", alpha=0.3)
    for bar_x, val in zip(["MOC", "OPT"], [moc_result.thrust, optimized_result.thrust]):
        ax_thrust.text(bar_x, val * 1.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax_loss.bar(["MOC", "OPT"], [moc_result.pressureLoss, optimized_result.pressureLoss], color=bar_colors)
    ax_loss.set_ylabel("Pressure loss [-]  (lower is better)")
    ax_loss.set_title(f"Pr. Loss  (\u25bc {_loss_winner} \u2212{_dl_pct:.1f}%)")
    ax_loss.grid(True, axis="y", alpha=0.3)
    for bar_x, val in zip(["MOC", "OPT"], [moc_result.pressureLoss, optimized_result.pressureLoss]):
        ax_loss.text(bar_x, val * 1.01, f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Performance Comparison\n{_trophy}\n"
        f"Thrust: {_thrust_winner} wins (+{_dt_pct:.1f}%)   |   Pressure loss: {_loss_winner} wins (\u2212{_dl_pct:.1f}%)\n"
        f"Mach: higher exit Mach = better expansion.  Pressure: lower exit p = better.  Temperature: lower exit T = more KE.",
        fontsize=8, color=_verdict_color, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    plt.savefig(out / "compare_performance.png", dpi=180)
    plt.close()
