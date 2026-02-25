from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import math
import csv

import matplotlib.pyplot as plt


@dataclass
class NozzleGeometry:
    """2D planar nozzle geometry defined by upper-wall points (x, y)."""

    control_points: List[Tuple[float, float]] = field(default_factory=list)
    throat_radius: float = 0.0
    exit_radius: float = 0.0
    length: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def controlPoints(self) -> List[Tuple[float, float]]:
        return self.control_points

    @property
    def throatRadius(self) -> float:
        return self.throat_radius

    @property
    def exitRadius(self) -> float:
        return self.exit_radius

    @property
    def expansion_ratio(self) -> float:
        return (self.exit_radius / self.throat_radius) ** 2

    def __post_init__(self) -> None:
        if self.throat_radius <= 0:
            raise ValueError("throat_radius must be > 0")
        if self.exit_radius <= self.throat_radius:
            raise ValueError("exit_radius must be > throat_radius")
        if self.length <= 0:
            raise ValueError("length must be > 0")
        if not self.control_points:
            raise ValueError("control_points cannot be empty")
        self._validate_wall_points(
            self.control_points,
            self.length,
            self.throat_radius,
            self.exit_radius,
        )

    @staticmethod
    def _validate_wall_points(
        wall_points: List[Tuple[float, float]],
        length: float,
        throat_y: float,
        exit_y: float,
        tol: float = 1e-9,
    ) -> None:
        prev_x = wall_points[0][0]
        if wall_points[0][1] <= 0:
            raise ValueError("y must be > 0")

        for x, y in wall_points[1:]:
            if y <= 0:
                raise ValueError("y must be > 0")
            if x <= prev_x:
                raise ValueError("x must be strictly increasing")
            prev_x = x

        if abs(wall_points[0][0]) > tol:
            raise ValueError("x must start at 0")
        if abs(wall_points[-1][0] - length) > tol:
            raise ValueError("x must end at length")

        ys = [y for _, y in wall_points]
        if min(ys) < throat_y - tol:
            raise ValueError("y must be >= throat")
        if max(ys) > exit_y + tol:
            raise ValueError("y must be <= exit")
        if abs(ys[0] - throat_y) > tol:
            raise ValueError("y at x=0 must equal throat")
        if abs(ys[-1] - exit_y) > tol:
            raise ValueError("y at x=L must equal exit")

    @classmethod
    def fromMOC(cls, params: Dict[str, Any]) -> "NozzleGeometry":
        wall_points: List[Tuple[float, float]] = list(params["wall_points"])
        if not wall_points:
            raise ValueError("wall_points cannot be empty")

        wall_points.sort(key=lambda p: p[0])
        x0 = wall_points[0][0]
        if x0 != 0:
            wall_points = [(x - x0, y) for x, y in wall_points]

        xs = [x for x, _ in wall_points]
        ys = [y for _, y in wall_points]
        throat = min(ys)
        exit_y = ys[-1]
        length = xs[-1]

        cls._validate_wall_points(wall_points, length, throat, exit_y)
        return cls(
            control_points=wall_points,
            throat_radius=throat,
            exit_radius=exit_y,
            length=length,
            metadata={"source": "MOC", **params.get("metadata", {})},
        )

    @classmethod
    def fromParams(cls, params: Dict[str, Any]) -> "NozzleGeometry":
        throat = float(params["throat_radius"])
        exit_y = float(params["exit_radius"])
        length = float(params["length"])
        n_points = int(params.get("n_points", 120))
        profile = params.get("profile", "cosine")

        if n_points < 3:
            raise ValueError("n_points must be >= 3")

        xs = [i * length / (n_points - 1) for i in range(n_points)]
        if profile == "cosine":
            ys = [
                throat
                + (exit_y - throat) * 0.5 * (1 - math.cos(math.pi * x / length))
                for x in xs
            ]
        elif profile == "bezier_like":
            shape = float(params.get("shape", 1.8))
            ys = [
                throat + (exit_y - throat) * ((x / length) ** shape)
                for x in xs
            ]
        elif profile == "rao":
            # Rao thrust-optimised contour: expansion zone + straightening
            # zone.  Guarantees zero exit angle (parallel exit flow) by
            # construction, like a real Rao / MLN design.
            #
            # Parameters
            # ----------
            # theta_max       max wall half-angle [deg] (default 22)
            # inflection_frac fraction of length where theta_max occurs (0.35)
            # throat_angle    initial wall angle at throat [deg]  (3.0)
            theta_max_deg = float(params.get("theta_max", 22.0))
            inflect_frac  = max(0.10, min(0.65, float(params.get("inflection_frac", 0.35))))
            throat_angle_deg = float(params.get("throat_angle", 3.0))

            theta_max = math.radians(theta_max_deg)
            throat_ang = math.radians(max(0.0, min(throat_angle_deg, 0.9 * theta_max_deg)))
            x_inflect = inflect_frac * length

            # Build wall angles then integrate y
            ys = [throat]
            for i in range(1, n_points):
                x = xs[i]
                dx = xs[i] - xs[i - 1]
                if x <= x_inflect:
                    # Expansion: angle rises throat_ang -> theta_max (cubic Hermite)
                    s = x / max(x_inflect, 1e-12)
                    smooth = s * s * (3.0 - 2.0 * s)
                    theta = throat_ang + (theta_max - throat_ang) * smooth
                else:
                    # Straightening: angle falls theta_max -> 0 (cubic Hermite)
                    s = (x - x_inflect) / max(length - x_inflect, 1e-12)
                    smooth = s * s * (3.0 - 2.0 * s)
                    theta = theta_max * (1.0 - smooth)
                ys.append(ys[-1] + math.tan(theta) * dx)

            # Scale y to match prescribed exit_radius exactly
            raw_rise = ys[-1] - throat
            target_rise = exit_y - throat
            if abs(raw_rise) > 1e-12:
                scale = target_rise / raw_rise
                ys = [throat + (y - throat) * scale for y in ys]
            # Clamp last point precisely
            ys[-1] = exit_y
        else:
            raise ValueError(f"Unsupported profile: {profile}")

        points = list(zip(xs, ys))
        cls._validate_wall_points(points, length, throat, exit_y)
        return cls(
            control_points=points,
            throat_radius=throat,
            exit_radius=exit_y,
            length=length,
            metadata={"source": "PARAMS", **params.get("metadata", {})},
        )

    def y_at(self, x: float) -> float:
        if x <= 0:
            return self.control_points[0][1]
        if x >= self.length:
            return self.control_points[-1][1]

        for i in range(1, len(self.control_points)):
            x0, y0 = self.control_points[i - 1]
            x1, y1 = self.control_points[i]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return self.control_points[-1][1]

    def wall_angles(self) -> List[float]:
        angles: List[float] = []
        for i in range(1, len(self.control_points)):
            x0, y0 = self.control_points[i - 1]
            x1, y1 = self.control_points[i]
            angles.append(math.atan2(y1 - y0, x1 - x0))
        return angles

    def extract_rao_params(self) -> Dict[str, float]:
        """Estimate Rao-profile parameters from this contour.

        Useful for seeding an optimiser with an existing MOC contour so that
        the GA starts in the right neighbourhood.

        Returns dict with keys: exit_radius, length, theta_max, inflection_frac,
        throat_angle.
        """
        angles = self.wall_angles()
        if not angles:
            return {
                "exit_radius": self.exit_radius,
                "length": self.length,
                "theta_max": 20.0,
                "inflection_frac": 0.35,
                "throat_angle": 3.0,
            }
        # Find the maximum wall angle and its position
        max_angle = max(angles)
        max_idx = angles.index(max_angle)
        # x position of max angle (midpoint of the segment)
        x_max = 0.5 * (self.control_points[max_idx][0] + self.control_points[max_idx + 1][0])
        inflect_frac = x_max / max(self.length, 1e-12)
        # Throat angle: average of first few segments
        n_avg = max(1, min(5, len(angles) // 10))
        throat_angle = math.degrees(sum(angles[:n_avg]) / n_avg)

        return {
            "exit_radius": self.exit_radius,
            "length": self.length,
            "theta_max": round(math.degrees(max_angle), 3),
            "inflection_frac": round(max(0.10, min(0.65, inflect_frac)), 4),
            "throat_angle": round(max(0.5, throat_angle), 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_points": self.control_points,
            "throat_radius": self.throat_radius,
            "exit_radius": self.exit_radius,
            "length": self.length,
            "metadata": self.metadata,
        }

    def exportGeo(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(self.control_points)

    def plotProfile(self, savepath: str | None = None) -> None:
        xs = [p[0] for p in self.control_points]
        ys = [p[1] for p in self.control_points]

        plt.figure(figsize=(8, 3.6))
        plt.plot(xs, ys, label="upper wall")
        plt.plot(xs, [-y for y in ys], linestyle="--", label="lower wall")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Nozzle Profile (2D planar)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if savepath:
            plt.tight_layout()
            plt.savefig(savepath, dpi=180)
            plt.close()
        else:
            plt.show()
