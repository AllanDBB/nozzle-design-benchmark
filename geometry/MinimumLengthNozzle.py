"""Minimum-Length Nozzle (MLN) — dedicated geometry builder.

A Minimum-Length Nozzle is the shortest diverging contour that produces a
perfectly uniform, parallel, supersonic exit flow for a given exit Mach
number.  The maximum wall angle equals ν_exit / 2  (half the Prandtl-Meyer
function evaluated at Mₑ), which is the theoretical limit from the MOC
simple-wave analysis.

This class encapsulates all MLN-specific geometry generation.  It is called
internally by :class:`~geometry.MOCSolver.MOCSolver` but can also be used
directly when the MOC characteristic tracing is not needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .NozzleGeometry import NozzleGeometry


@dataclass
class MinimumLengthNozzle:
    """Geometry builder for a Minimum-Length Nozzle (MLN) contour.

    Parameters
    ----------
    mach_exit:
        Design exit Mach number (> 1).
    throat_radius:
        Half-height at the throat [m].
    exit_radius:
        Half-height at the exit plane [m].  Must be > ``throat_radius``.
    length:
        Axial length of the diverging section [m].
    gamma:
        Ratio of specific heats (default 1.4 for air).
    n_points:
        Number of discrete wall points (≥ 3, default 180).
    throat_angle_deg:
        Initial wall inclination just downstream of the throat [°].
        Defaults to 4°.  Clamped to [0, 0.9 · θ_max].
    ramp_power:
        Exponent controlling how fast the wall angle rises in the expansion
        zone.  Values < 1 give a concave-forward ramp (recommended 0.7).
    straighten_frac:
        Fraction of the total length allocated to the *straightening* zone
        (angle falls from θ_max back to 0).  Clamped to [0.30, 0.75],
        default 0.45.
    """

    mach_exit: float
    throat_radius: float
    exit_radius: float
    length: float
    gamma: float = 1.4
    n_points: int = 180
    throat_angle_deg: float = 4.0
    ramp_power: float = 0.7
    straighten_frac: float = 0.45

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if self.mach_exit <= 1.0:
            raise ValueError(f"mach_exit must be > 1  (got {self.mach_exit})")
        if self.throat_radius <= 0.0:
            raise ValueError(f"throat_radius must be > 0  (got {self.throat_radius})")
        if self.exit_radius <= self.throat_radius:
            raise ValueError(
                f"exit_radius ({self.exit_radius}) must be > throat_radius ({self.throat_radius})"
            )
        if self.length <= 0.0:
            raise ValueError(f"length must be > 0  (got {self.length})")
        if self.n_points < 3:
            raise ValueError(f"n_points must be >= 3  (got {self.n_points})")
        if self.ramp_power <= 0.0:
            raise ValueError(f"ramp_power must be > 0  (got {self.ramp_power})")
        self.straighten_frac = max(0.30, min(0.75, self.straighten_frac))

    # ------------------------------------------------------------------ #
    # Isentropic / Prandtl-Meyer helpers                                   #
    # ------------------------------------------------------------------ #

    def _prandtl_meyer(self, mach: float) -> float:
        """Prandtl-Meyer expansion function ν(M) [rad]."""
        gp1 = self.gamma + 1.0
        gm1 = self.gamma - 1.0
        c = math.sqrt(gp1 / gm1)
        return c * math.atan(math.sqrt(gm1 / gp1 * (mach * mach - 1.0))) - math.atan(
            math.sqrt(mach * mach - 1.0)
        )

    @property
    def theta_max(self) -> float:
        """Maximum wall angle for an MLN [rad] = ν(Mₑ) / 2."""
        return 0.5 * self._prandtl_meyer(self.mach_exit)

    @property
    def theta_max_deg(self) -> float:
        """Maximum wall angle [°]."""
        return math.degrees(self.theta_max)

    @property
    def expansion_ratio(self) -> float:
        """Geometric area expansion ratio  Aₑ / A*."""
        return (self.exit_radius / self.throat_radius) ** 2

    # ------------------------------------------------------------------ #
    # Wall-contour generation                                              #
    # ------------------------------------------------------------------ #

    def _build_wall_points(self) -> List[Tuple[float, float]]:
        """Return the raw (x, y) wall contour before exit-radius correction."""
        theta_max = self.theta_max
        theta_start = math.radians(max(0.0, self.throat_angle_deg))
        theta_start = min(theta_start, 0.9 * theta_max)

        x_switch = (1.0 - self.straighten_frac) * self.length

        xs: List[float] = [
            i * self.length / (self.n_points - 1) for i in range(self.n_points)
        ]
        thetas: List[float] = []
        for x in xs:
            if x <= x_switch:
                # Expansion zone: angle rises θ_start → θ_max.
                s = x / max(x_switch, 1e-12)
                smooth = (s ** self.ramp_power) * (3.0 - 2.0 * s)
                smooth = max(0.0, min(1.0, smooth))
                thetas.append(theta_start + (theta_max - theta_start) * smooth)
            else:
                # Straightening zone: angle falls θ_max → 0 (parallel exit).
                s = (x - x_switch) / max(self.length - x_switch, 1e-12)
                smooth = s * s * (3.0 - 2.0 * s)   # cubic Hermite 0 → 1
                thetas.append(theta_max * (1.0 - smooth))

        ys: List[float] = [self.throat_radius]
        for i in range(1, self.n_points):
            dx = xs[i] - xs[i - 1]
            ys.append(ys[-1] + math.tan(thetas[i]) * dx)

        return list(zip(xs, ys))

    def build(self) -> NozzleGeometry:
        """Generate the MLN wall contour and return a :class:`NozzleGeometry`.

        The raw wall integration always starts at ``throat_radius``.  The
        profile is then uniformly scaled in y so that the exit plane lands
        exactly at ``exit_radius``.

        Returns
        -------
        NozzleGeometry
            Validated nozzle geometry object ready for meshing or evaluation.

        Raises
        ------
        ValueError
            If the raw profile did not expand (degenerate parameters).
        """
        raw_points = self._build_wall_points()
        ys_raw = [y for _, y in raw_points]

        raw_rise = ys_raw[-1] - self.throat_radius
        if abs(raw_rise) < 1e-12:
            raise ValueError(
                "MinimumLengthNozzle: generated profile did not expand — "
                "check mach_exit, throat_angle_deg, and ramp_power."
            )

        scale = (self.exit_radius - self.throat_radius) / raw_rise
        scaled_points: List[Tuple[float, float]] = [
            (x, self.throat_radius + (y - self.throat_radius) * scale)
            for x, y in raw_points
        ]

        return NozzleGeometry.fromMOC(
            {
                "wall_points": scaled_points,
                "metadata": {
                    "generator": "MinimumLengthNozzle",
                    "mach_exit": self.mach_exit,
                    "gamma": self.gamma,
                    "theta_max_deg": self.theta_max_deg,
                    "expansion_ratio": self.expansion_ratio,
                    "straighten_frac": self.straighten_frac,
                    "ramp_power": self.ramp_power,
                    "throat_angle_deg": self.throat_angle_deg,
                },
            }
        )

    # ------------------------------------------------------------------ #
    # Convenience constructors                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_params(
        cls,
        params: Dict[str, Any],
        mach_exit: float,
        gamma: float = 1.4,
    ) -> "MinimumLengthNozzle":
        """Construct from the legacy ``params`` dict used by :class:`MOCSolver`.

        Parameters
        ----------
        params:
            Dict with keys ``throat_y``, ``exit_y``, ``length``, and
            optionally ``n_points``, ``throat_angle_deg``, ``ramp_power``,
            ``straighten_frac``.
        mach_exit:
            Exit Mach number.
        gamma:
            Ratio of specific heats.
        """
        return cls(
            mach_exit=mach_exit,
            throat_radius=float(params["throat_y"]),
            exit_radius=float(params["exit_y"]),
            length=float(params["length"]),
            gamma=gamma,
            n_points=int(params.get("n_points", 180)),
            throat_angle_deg=float(params.get("throat_angle_deg", 4.0)),
            ramp_power=float(params.get("ramp_power", 0.7)),
            straighten_frac=float(params.get("straighten_frac", 0.45)),
        )

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def plot(self, savepath: str, *, dpi: int = 180) -> None:
        """Plot the MLN wall contour and wall-angle distribution.

        Produces a two-panel figure:

        * **Top** — 2-D nozzle cross-section (upper wall, lower wall,
          centreline) with a shaded band marking the expansion zone and a
          vertical dashed line at the expansion/straightening boundary *x_sw*.
        * **Bottom** — Wall angle theta(x) [deg] showing the ramp-up in the
          expansion zone, the peak theta_max, and the ramp-down to zero in
          the straightening zone.

        Parameters
        ----------
        savepath:
            File path where the figure is saved (PNG/PDF/SVG).
        dpi:
            Image resolution for raster formats (default 180).
        """
        raw_pts = self._build_wall_points()
        ys_raw = [y for _, y in raw_pts]
        raw_rise = ys_raw[-1] - self.throat_radius
        if abs(raw_rise) < 1e-12:
            raise ValueError("Profile did not expand — cannot plot.")
        scale = (self.exit_radius - self.throat_radius) / raw_rise
        xs = [x for x, _ in raw_pts]
        ys = [self.throat_radius + (y - self.throat_radius) * scale for y in ys_raw]

        # Recompute theta at each point for the angle panel.
        theta_max = self.theta_max
        theta_start = math.radians(max(0.0, self.throat_angle_deg))
        theta_start = min(theta_start, 0.9 * theta_max)
        x_switch = (1.0 - self.straighten_frac) * self.length

        thetas_deg: List[float] = []
        for x in xs:
            if x <= x_switch:
                s = x / max(x_switch, 1e-12)
                smooth = max(0.0, min(1.0, (s ** self.ramp_power) * (3.0 - 2.0 * s)))
                th = theta_start + (theta_max - theta_start) * smooth
            else:
                s = (x - x_switch) / max(self.length - x_switch, 1e-12)
                smooth = s * s * (3.0 - 2.0 * s)
                th = theta_max * (1.0 - smooth)
            thetas_deg.append(math.degrees(th))

        fig, (ax_geo, ax_ang) = plt.subplots(
            2, 1, figsize=(11, 7),
            gridspec_kw={"height_ratios": [2, 1]},
        )

        # ---- Top panel: geometry ----------------------------------------
        # Shaded expansion zone.
        ax_geo.axvspan(0.0, x_switch, alpha=0.08, color="tab:blue",
                       label="Expansion zone")
        ax_geo.axvspan(x_switch, self.length, alpha=0.08, color="tab:orange",
                       label="Straightening zone")
        ax_geo.axvline(x_switch, color="gray", linewidth=1.0,
                       linestyle="--", alpha=0.7)

        # Upper wall.
        ax_geo.plot(xs, ys, color="black", linewidth=1.8, label="Wall")
        # Lower wall (mirror).
        ax_geo.plot(xs, [-y for y in ys], color="black", linewidth=1.8)
        # Centreline.
        ax_geo.axhline(0.0, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.5, label="Centreline")

        # Throat and exit annotations.
        ax_geo.annotate(
            f"throat\n{self.throat_radius*1e3:.1f} mm",
            xy=(0.0, self.throat_radius),
            xytext=(self.length * 0.05, self.exit_radius * 0.75),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=8, color="dimgray",
        )
        ax_geo.annotate(
            f"exit\n{self.exit_radius*1e3:.1f} mm",
            xy=(self.length, self.exit_radius),
            xytext=(self.length * 0.80, self.exit_radius * 1.05),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=8, color="dimgray",
        )
        ax_geo.annotate(
            f"x_sw = {x_switch*1e3:.1f} mm",
            xy=(x_switch, 0.0),
            xytext=(x_switch + self.length * 0.03, -self.exit_radius * 0.6),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=8, color="dimgray",
        )

        ax_geo.set_xlabel("x [m]")
        ax_geo.set_ylabel("y [m]")
        ax_geo.set_title(
            f"Minimum-Length Nozzle  |  "
            f"Me = {self.mach_exit:.2f}  |  "
            f"epsilon = {self.expansion_ratio:.3f}  |  "
            f"L = {self.length*1e3:.1f} mm"
        )
        ax_geo.set_aspect("equal", adjustable="box")
        ax_geo.legend(loc="upper left", fontsize=8)
        ax_geo.grid(True, alpha=0.20)

        # ---- Bottom panel: angle distribution ---------------------------
        ax_ang.axvspan(0.0, x_switch, alpha=0.08, color="tab:blue")
        ax_ang.axvspan(x_switch, self.length, alpha=0.08, color="tab:orange")
        ax_ang.axvline(x_switch, color="gray", linewidth=1.0,
                       linestyle="--", alpha=0.7)
        ax_ang.axhline(self.theta_max_deg, color="tab:red", linewidth=1.0,
                       linestyle=":", alpha=0.8,
                       label=f"theta_max = {self.theta_max_deg:.2f} deg")

        ax_ang.plot(xs, thetas_deg, color="tab:blue", linewidth=1.6,
                    label="Wall angle theta(x)")

        ax_ang.set_xlabel("x [m]")
        ax_ang.set_ylabel("Wall angle [deg]")
        ax_ang.legend(fontsize=8)
        ax_ang.grid(True, alpha=0.20)

        plt.tight_layout()
        plt.savefig(savepath, dpi=dpi)
        plt.close(fig)
