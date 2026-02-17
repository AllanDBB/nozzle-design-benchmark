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


def generate_comparison_plots(
    moc_geometry: NozzleGeometry,
    optimized_geometry: NozzleGeometry,
    moc_result: EvaluationResult,
    optimized_result: EvaluationResult,
    out_dir: str,
    solver_config: Dict[str, Any],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gamma = float(solver_config.get("gamma", 1.4))
    gas_constant = float(solver_config.get("gas_constant", 287.0))

    # 1) Geometry overlay
    plt.figure(figsize=(8, 4))
    xm = [p[0] for p in moc_geometry.control_points]
    ym = [p[1] for p in moc_geometry.control_points]
    xo = [p[0] for p in optimized_geometry.control_points]
    yo = [p[1] for p in optimized_geometry.control_points]
    plt.plot(xm, ym, label="MOC upper", color="tab:blue")
    plt.plot(xo, yo, label="OPT upper", color="tab:orange")
    plt.plot(xm, [-y for y in ym], color="tab:blue", linestyle="--", alpha=0.5)
    plt.plot(xo, [-y for y in yo], color="tab:orange", linestyle="--", alpha=0.5)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Geometry Comparison")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "compare_geometry.png", dpi=180)
    plt.close()

    # 2) Mach profile comparison
    if moc_result.machProfile and optimized_result.machProfile:
        xmprof = _x_from_profile(moc_geometry.length, len(moc_result.machProfile))
        xoprof = _x_from_profile(optimized_geometry.length, len(optimized_result.machProfile))
        plt.figure(figsize=(8, 4))
        plt.plot(xmprof, moc_result.machProfile, label="MOC", color="tab:blue")
        plt.plot(xoprof, optimized_result.machProfile, label="OPT", color="tab:orange")
        plt.xlabel("x [m]")
        plt.ylabel("Mach")
        plt.title("Mach Along Nozzle")
        plt.grid(True, alpha=0.3)
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "compare_mach.png", dpi=180)
        plt.close()

    # 3) Velocity profile comparison (requested)
    vm = _velocity_profile(moc_result.machProfile, moc_result.temperatureProfile, gamma, gas_constant)
    vo = _velocity_profile(optimized_result.machProfile, optimized_result.temperatureProfile, gamma, gas_constant)
    if vm and vo:
        xv_m = _x_from_profile(moc_geometry.length, len(vm))
        xv_o = _x_from_profile(optimized_geometry.length, len(vo))
        plt.figure(figsize=(8, 4))
        plt.plot(xv_m, vm, label="MOC", color="tab:blue")
        plt.plot(xv_o, vo, label="OPT", color="tab:orange")
        plt.xlabel("x [m]")
        plt.ylabel("Velocity [m/s]")
        plt.title("Velocity Along Nozzle")
        plt.grid(True, alpha=0.3)
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "compare_velocity.png", dpi=180)
        plt.close()

    # 4) Pressure profile comparison
    if moc_result.pressureProfile and optimized_result.pressureProfile:
        xp_m = _x_from_profile(moc_geometry.length, len(moc_result.pressureProfile))
        xp_o = _x_from_profile(optimized_geometry.length, len(optimized_result.pressureProfile))
        plt.figure(figsize=(8, 4))
        plt.plot(xp_m, moc_result.pressureProfile, label="MOC", color="tab:blue")
        plt.plot(xp_o, optimized_result.pressureProfile, label="OPT", color="tab:orange")
        plt.xlabel("x [m]")
        plt.ylabel("Pressure [Pa]")
        plt.title("Pressure Along Nozzle")
        plt.grid(True, alpha=0.3)
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "compare_pressure.png", dpi=180)
        plt.close()

    # 5) Temperature profile comparison
    if moc_result.temperatureProfile and optimized_result.temperatureProfile:
        xt_m = _x_from_profile(moc_geometry.length, len(moc_result.temperatureProfile))
        xt_o = _x_from_profile(optimized_geometry.length, len(optimized_result.temperatureProfile))
        plt.figure(figsize=(8, 4))
        plt.plot(xt_m, moc_result.temperatureProfile, label="MOC", color="tab:blue")
        plt.plot(xt_o, optimized_result.temperatureProfile, label="OPT", color="tab:orange")
        plt.xlabel("x [m]")
        plt.ylabel("Temperature [K]")
        plt.title("Temperature Along Nozzle")
        plt.grid(True, alpha=0.3)
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "compare_temperature.png", dpi=180)
        plt.close()

    # 6) Performance bars
    labels = ["Thrust [N]", "Pressure loss [-]"]
    moc_vals = [moc_result.thrust, moc_result.pressureLoss]
    opt_vals = [optimized_result.thrust, optimized_result.pressureLoss]

    x = [0, 1]
    w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar([i - w / 2 for i in x], moc_vals, width=w, label="MOC", color="tab:blue")
    plt.bar([i + w / 2 for i in x], opt_vals, width=w, label="OPT", color="tab:orange")
    plt.xticks(x, labels)
    plt.title("Performance Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "compare_performance.png", dpi=180)
    plt.close()
