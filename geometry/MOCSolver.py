from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import math

import matplotlib.pyplot as plt

from .NozzleGeometry import NozzleGeometry


@dataclass
class MOCSolver:
    """Simplified Method of Characteristics-inspired nozzle generator."""

    machExit: float
    pressureRatio: float
    nCharacteristics: int = 40
    gamma: float = 1.4

    @staticmethod
    def _prandtl_meyer(mach: float, gamma: float) -> float:
        if mach < 1:
            raise ValueError("mach must be >= 1")
        a = math.sqrt((gamma + 1.0) / (gamma - 1.0))
        b = math.sqrt((gamma - 1.0) / (gamma + 1.0) * (mach * mach - 1.0))
        return a * math.atan(b) - math.atan(math.sqrt(mach * mach - 1.0))

    @staticmethod
    def _pressure_ratio_from_mach(mach: float, gamma: float) -> float:
        return (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** (-gamma / (gamma - 1.0))

    @staticmethod
    def _area_mach_ratio(mach: float, gamma: float) -> float:
        term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach * mach)
        exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return (1.0 / mach) * (term ** exponent)

    @classmethod
    def _mach_from_area(cls, area_ratio: float, gamma: float) -> float:
        if area_ratio <= 1.0 + 1e-6:
            return 1.0

        lo, hi = 1.0 + 1e-7, 12.0
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            f_mid = cls._area_mach_ratio(mid, gamma) - area_ratio
            if abs(f_mid) < 1e-10:
                return mid
            f_lo = cls._area_mach_ratio(lo, gamma) - area_ratio
            if f_lo * f_mid < 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def computeCharacteristics(self) -> List[Dict[str, float]]:
        if self.machExit <= 1:
            raise ValueError("machExit must be > 1")

        machs = [
            1.0 + (self.machExit - 1.0) * i / (self.nCharacteristics - 1)
            for i in range(self.nCharacteristics)
        ]
        nu_exit = self._prandtl_meyer(self.machExit, self.gamma)
        rows: List[Dict[str, float]] = []
        for mach in machs:
            nu = self._prandtl_meyer(mach, self.gamma)
            theta = 0.5 * nu
            p_p0 = self._pressure_ratio_from_mach(mach, self.gamma)
            rows.append({"mach": mach, "nu": nu, "theta": theta, "p_p0": p_p0, "nu_exit": nu_exit})
        return rows

    def generateGeometry(self, params: Dict[str, Any]) -> NozzleGeometry:
        throat_y = float(params["throat_y"])
        exit_y = float(params["exit_y"])
        length = float(params["length"])
        n_points = int(params.get("n_points", 180))
        throat_angle_deg = float(params.get("throat_angle_deg", 4.0))
        ramp_power = float(params.get("ramp_power", 0.7))
        # Fraction of length used for the expansion section [0.4 – 0.7].
        # Remaining fraction is the straightening section (angle → 0 at exit).
        straighten_frac = float(params.get("straighten_frac", 0.45))
        straighten_frac = max(0.3, min(0.75, straighten_frac))

        if n_points < 3:
            raise ValueError("n_points must be >= 3")
        if ramp_power <= 0.0:
            raise ValueError("ramp_power must be > 0")

        nu_exit = self._prandtl_meyer(self.machExit, self.gamma)
        # For a minimum-length MOC nozzle the maximum wall angle equals ν_exit/2.
        theta_max = 0.5 * nu_exit
        theta_start = math.radians(max(0.0, throat_angle_deg))
        theta_start = min(theta_start, 0.9 * theta_max)

        # Split into expansion zone [0, x_sw] and straightening zone [x_sw, L].
        x_switch = (1.0 - straighten_frac) * length

        xs = [i * length / (n_points - 1) for i in range(n_points)]
        thetas: List[float] = []
        for x in xs:
            if x <= x_switch:
                # Expansion: angle rises from theta_start → theta_max.
                s = x / max(x_switch, 1e-12)
                smooth = (s ** ramp_power) * (3.0 - 2.0 * s)
                smooth = max(0.0, min(1.0, smooth))
                thetas.append(theta_start + (theta_max - theta_start) * smooth)
            else:
                # Straightening: angle falls from theta_max → 0 (parallel exit).
                s = (x - x_switch) / max(length - x_switch, 1e-12)
                smooth = s * s * (3.0 - 2.0 * s)          # cubic Hermite, 0→1
                thetas.append(theta_max * (1.0 - smooth))

        ys: List[float] = [throat_y]
        for i in range(1, n_points):
            dx = xs[i] - xs[i - 1]
            ys.append(ys[-1] + math.tan(thetas[i]) * dx)

        if abs(ys[-1] - throat_y) < 1e-12:
            raise ValueError("generated profile did not expand")

        scale = (exit_y - throat_y) / (ys[-1] - throat_y)
        ys = [throat_y + (y - throat_y) * scale for y in ys]

        points: List[Tuple[float, float]] = list(zip(xs, ys))
        return NozzleGeometry.fromMOC(
            {
                "wall_points": points,
                "metadata": {
                    "mach_exit_target": self.machExit,
                    "pressure_ratio_target": self.pressureRatio,
                    "pressure_ratio_ideal": self._pressure_ratio_from_mach(self.machExit, self.gamma),
                    "gamma": self.gamma,
                    "theta_max_deg": math.degrees(theta_max),
                    "x_switch_m": x_switch,
                },
            }
        )

    def plotCharacteristics(
        self,
        geometry: NozzleGeometry,
        savepath: str,
        n_lines: int = 40,
        line_alpha: float = 0.50,
    ) -> None:
        if n_lines < 4:
            n_lines = 4

        x_end = geometry.length
        throat = geometry.throat_radius

        # Fine x grid for ray marching.
        n_steps = n_lines * 12
        xs_grid = [i * x_end / n_steps for i in range(n_steps + 1)]

        plt.figure(figsize=(11, 5))
        # Nozzle walls (upper + lower + centreline).
        wall_x = [p[0] for p in geometry.control_points]
        wall_y = [p[1] for p in geometry.control_points]
        plt.plot(wall_x, wall_y, color="black", linewidth=1.5)
        plt.plot(wall_x, [-y for y in wall_y], color="black", linewidth=1.5)
        plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

        # ------------------------------------------------------------------
        # C+ (right-running) family — originate on the centreline,
        # slope = tan(θ_local + μ_local)  (wall angle + Mach angle).
        # ------------------------------------------------------------------
        for j in range(n_lines):
            x0 = j * x_end / max(n_lines - 1, 1)
            y0 = 0.0

            y_w0 = geometry.y_at(x0)
            ar0 = (y_w0 / max(throat, 1e-9)) ** 2
            m0 = self._mach_from_area(max(ar0, 1.0), self.gamma)
            mu0 = math.asin(1.0 / max(m0, 1.001))

            # Local flow angle at x0 (wall angle gives expansion fan angle ≈ θ).
            slope_w = (geometry.y_at(min(x0 + 1e-4, x_end)) - geometry.y_at(max(x0 - 1e-4, 0.0))) / 2e-4
            theta_local = math.atan(max(slope_w, 0.0))
            alpha = theta_local + mu0          # C+ characteristic slope

            line_x = [x0]
            line_y = [y0]
            for x in xs_grid:
                if x <= x0:
                    continue
                y = y0 + math.tan(alpha) * (x - x0)
                yw = geometry.y_at(x)
                if y >= yw:
                    # Clip to wall.
                    line_x.append(x)
                    line_y.append(yw)
                    break
                line_x.append(x)
                line_y.append(y)

            if len(line_x) > 1:
                plt.plot(line_x, line_y,              color="tab:blue", alpha=line_alpha, linewidth=0.85)
                plt.plot(line_x, [-y for y in line_y], color="tab:blue", alpha=line_alpha, linewidth=0.85)

        # ------------------------------------------------------------------
        # C- (left-running) family — originate on the wall,
        # slope = tan(θ_local − μ_local)  (points downward to centreline).
        # ------------------------------------------------------------------
        for j in range(1, n_lines + 1):
            x0 = j * x_end / max(n_lines, 1)
            if x0 > x_end:
                break
            y0 = geometry.y_at(x0)

            slope_w = (geometry.y_at(min(x0 + 1e-4, x_end)) - geometry.y_at(max(x0 - 1e-4, 0.0))) / 2e-4
            theta_local = math.atan(slope_w)

            ar0 = (y0 / max(throat, 1e-9)) ** 2
            m0 = self._mach_from_area(max(ar0, 1.0), self.gamma)
            mu0 = math.asin(1.0 / max(m0, 1.001))

            beta = theta_local - mu0
            tan_beta = math.tan(beta)
            if abs(tan_beta) < 1e-10:
                continue

            # Ray toward centreline.
            x_hit = x0 - y0 / tan_beta
            x_hit = max(0.0, min(x_end, x_hit))
            y_hit = max(0.0, y0 + tan_beta * (x_hit - x0))

            plt.plot([x0, x_hit], [y0, y_hit],              color="tab:red", alpha=line_alpha, linewidth=0.85)
            plt.plot([x0, x_hit], [-y0, -y_hit], color="tab:red", alpha=line_alpha, linewidth=0.85)

        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"MOC Characteristic Lines  (C+ blue, C− red)  |  Mₑ = {self.machExit:.2f}")
        plt.grid(True, alpha=0.20)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(savepath, dpi=180)
        plt.close()
