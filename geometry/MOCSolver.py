from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import math

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

        if n_points < 3:
            raise ValueError("n_points must be >= 3")

        nu_exit = self._prandtl_meyer(self.machExit, self.gamma)
        theta_max = 0.5 * nu_exit

        xs = [i * length / (n_points - 1) for i in range(n_points)]
        thetas: List[float] = []
        for x in xs:
            s = x / length
            # Smooth ramp to imitate gradual turning in MOC contour.
            smooth = s * s * (3.0 - 2.0 * s)
            thetas.append(theta_max * smooth)

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
                },
            }
        )
