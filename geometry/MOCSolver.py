"""2-D Planar Isentropic Method of Characteristics (MOC) solver.

Solves the steady, inviscid, irrotational, isentropic supersonic flow field
inside a Minimum-Length Nozzle (MLN) using the Method of Characteristics.

Physical assumptions
--------------------
* No viscosity (Euler equations).
* No boundary layer.
* No losses, no internal shocks.
* Ideal calorically-perfect gas (constant gamma).
* Supersonic flow downstream of the sonic throat.

Characteristic relations (2-D planar)
--------------------------------------
Left-running  C+ :   dy/dx = tan(theta + mu)    K- = theta - nu  = const
Right-running C- :   dy/dx = tan(theta - mu)    K+ = theta + nu  = const

where theta is the flow angle, nu is the Prandtl-Meyer function,
and mu = arcsin(1/M).

References
----------
Anderson, J. D., Modern Compressible Flow, 3rd ed., McGraw-Hill.
Zucrow, M. J. & Hoffman, J. D., Gas Dynamics, Vol. II, Wiley.
Courant, R. & Friedrichs, K. O., Supersonic Flow and Shock Waves, Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from .NozzleGeometry import NozzleGeometry


# ======================================================================== #
#  Mesh-point data structure                                                #
# ======================================================================== #

@dataclass
class _MeshPoint:
    """Single point in the characteristic mesh."""
    x:     float = 0.0
    y:     float = 0.0
    theta: float = 0.0   # flow angle [rad]
    nu:    float = 0.0   # Prandtl-Meyer angle [rad]
    M:     float = 1.0   # Mach number
    mu:    float = math.pi / 2  # Mach angle [rad]

    @property
    def Kplus(self) -> float:
        """K+ = theta + nu  (invariant along C-)."""
        return self.theta + self.nu

    @property
    def Kminus(self) -> float:
        """K- = theta - nu  (invariant along C+)."""
        return self.theta - self.nu


# ======================================================================== #
#  MOCSolver                                                                #
# ======================================================================== #

@dataclass
class MOCSolver:
    """2-D planar isentropic MOC solver for a Minimum-Length Nozzle.

    Parameters
    ----------
    machExit : float
        Design exit Mach number (must be > 1).
    pressureRatio : float
        Ambient-to-total pressure ratio pa/p0.
    nCharacteristics : int
        Number of discrete fan waves in the Prandtl-Meyer expansion
        (default 40).
    gamma : float
        Ratio of specific heats (default 1.4 for air).
    """

    machExit: float
    pressureRatio: float
    nCharacteristics: int = 40
    gamma: float = 1.4

    # ------------------------------------------------------------------ #
    #  Isentropic helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prandtl_meyer(M: float, gamma: float) -> float:
        """Prandtl-Meyer expansion angle nu(M) [rad]."""
        if M < 1.0:
            raise ValueError(f"Mach must be >= 1 (got {M})")
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        return (math.sqrt(gp1 / gm1)
                * math.atan(math.sqrt(gm1 / gp1 * (M * M - 1.0)))
                - math.atan(math.sqrt(M * M - 1.0)))

    @staticmethod
    def _mach_from_nu(nu: float, gamma: float) -> float:
        """Inverse Prandtl-Meyer: M such that nu(M) = nu.  Bisection."""
        if nu <= 0.0:
            return 1.0
        lo, hi = 1.0 + 1e-9, 50.0
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        a = math.sqrt(gp1 / gm1)
        for _ in range(150):
            mid = 0.5 * (lo + hi)
            nu_mid = (a * math.atan(math.sqrt(gm1 / gp1 * (mid * mid - 1.0)))
                       - math.atan(math.sqrt(mid * mid - 1.0)))
            if nu_mid < nu:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _mu(M: float) -> float:
        """Mach angle mu = arcsin(1/M) [rad]."""
        return math.asin(min(1.0, 1.0 / max(M, 1.0 + 1e-12)))

    @staticmethod
    def _pressure_ratio_from_mach(M: float, gamma: float) -> float:
        """Isentropic total-to-static pressure ratio  p / p0."""
        return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-gamma / (gamma - 1.0))

    @staticmethod
    def _area_mach_ratio(M: float, gamma: float) -> float:
        """Isentropic area ratio  A / A*."""
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        t = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M * M)
        return (1.0 / M) * t ** (gp1 / (2.0 * gm1))

    # ------------------------------------------------------------------ #
    #  Unit processes                                                      #
    # ------------------------------------------------------------------ #

    def _interior_point(self, pt_cm: _MeshPoint,
                        pt_cp: _MeshPoint) -> _MeshPoint:
        """Interior unit process.

        pt_cm : upstream point on the C- characteristic (carries K+).
        pt_cp : upstream point on the C+ characteristic (carries K-).
        Returns the intersection point.
        """
        Kp = pt_cm.Kplus    # theta + nu along C-
        Km = pt_cp.Kminus   # theta - nu along C+

        theta3 = 0.5 * (Kp + Km)
        nu3    = 0.5 * (Kp - Km)
        M3     = self._mach_from_nu(nu3, self.gamma)
        mu3    = self._mu(M3)

        # Average slopes (predictor-corrector)
        s_cm = math.tan(0.5 * (pt_cm.theta + theta3)
                        - 0.5 * (pt_cm.mu + mu3))
        s_cp = math.tan(0.5 * (pt_cp.theta + theta3)
                        + 0.5 * (pt_cp.mu + mu3))

        denom = s_cm - s_cp
        if abs(denom) < 1e-14:
            x3 = 0.5 * (pt_cm.x + pt_cp.x)
        else:
            x3 = ((pt_cp.y - pt_cm.y
                    + s_cm * pt_cm.x - s_cp * pt_cp.x) / denom)
        y3 = pt_cm.y + s_cm * (x3 - pt_cm.x)

        return _MeshPoint(x=x3, y=y3, theta=theta3, nu=nu3, M=M3, mu=mu3)

    def _axis_point(self, pt_cm: _MeshPoint) -> _MeshPoint:
        """Symmetry-axis unit process (y = 0, theta = 0).

        pt_cm : point on the C- characteristic approaching the axis.
        """
        Kp     = pt_cm.Kplus   # preserved along C-
        theta3 = 0.0
        nu3    = Kp            # theta + nu = Kp, theta = 0
        M3     = self._mach_from_nu(nu3, self.gamma)
        mu3    = self._mu(M3)

        s_cm = math.tan(0.5 * pt_cm.theta - 0.5 * (pt_cm.mu + mu3))
        if abs(s_cm) < 1e-14:
            x3 = pt_cm.x + 1e-6
        else:
            x3 = pt_cm.x - pt_cm.y / s_cm
        return _MeshPoint(x=x3, y=0.0, theta=0.0, nu=nu3, M=M3, mu=mu3)

    def _wall_point(self, pt_cp: _MeshPoint,
                    prev_wall: _MeshPoint) -> _MeshPoint:
        """Wall unit process (streamline boundary, no reflection).

        pt_cp     : last interior point on the C+ heading toward the wall.
        prev_wall : previous wall point (or corner) for the wall streamline.

        For MLN design the wall absorbs the incoming C+ without
        generating a reflected C-.
        """
        theta_w = pt_cp.theta
        nu_w    = pt_cp.nu
        M_w     = pt_cp.M
        mu_w    = pt_cp.mu

        # C+ slope from pt_cp to wall: tan(theta + mu)
        s_cp = math.tan(0.5 * (pt_cp.theta + theta_w)
                        + 0.5 * (pt_cp.mu + mu_w))

        # Wall streamline slope from prev_wall to this point
        avg_theta_w = 0.5 * (prev_wall.theta + theta_w)
        s_wall = math.tan(avg_theta_w)

        denom = s_cp - s_wall
        if abs(denom) < 1e-14:
            x_w = pt_cp.x + 1e-6
        else:
            x_w = ((prev_wall.y - pt_cp.y
                     + s_cp * pt_cp.x - s_wall * prev_wall.x) / denom)
        y_w = prev_wall.y + s_wall * (x_w - prev_wall.x)

        return _MeshPoint(x=x_w, y=y_w, theta=theta_w, nu=nu_w,
                          M=M_w, mu=mu_w)

    # ------------------------------------------------------------------ #
    #  Full mesh computation                                               #
    # ------------------------------------------------------------------ #

    def computeMesh(self, throat_y: float) -> Dict[str, Any]:
        """Build the full 2-D planar isentropic MOC mesh for an MLN.

        Parameters
        ----------
        throat_y : float
            Half-height at the sonic throat.

        Returns
        -------
        dict with keys:
            fan, axis, interior, wall, corner, theta_max, nu_exit, N
        """
        N = self.nCharacteristics
        gamma = self.gamma
        nu_exit = self._prandtl_meyer(self.machExit, gamma)
        theta_max = 0.5 * nu_exit                    # MLN condition

        corner = _MeshPoint(x=0.0, y=throat_y, theta=0.0, nu=0.0,
                            M=1.0, mu=math.pi / 2)

        # -- 1) Kernel (Prandtl-Meyer fan) --------------------------------
        fan: List[_MeshPoint] = []
        for j in range(1, N + 1):
            th = theta_max * j / N
            nu = th                                   # kernel: K- = 0
            M  = self._mach_from_nu(nu, gamma)
            mu = self._mu(M)
            fan.append(_MeshPoint(x=0.0, y=throat_y,
                                  theta=th, nu=nu, M=M, mu=mu))

        # -- 2) March through mesh column-by-column -----------------------
        #  Column j: process C-_{j+1} from the fan.
        #  It crosses C+_1 .. C+_j before reaching the axis.
        axis: List[Optional[_MeshPoint]] = [None] * N
        interior: Dict[Tuple[int, int], _MeshPoint] = {}
        last_cp: List[Optional[_MeshPoint]] = [None] * N

        for j in range(N):
            prev_cm: _MeshPoint = fan[j]   # start from corner

            for i in range(j):
                pt_cp = last_cp[i]
                p = self._interior_point(prev_cm, pt_cp)
                interior[(i, j)] = p
                last_cp[i] = p       # advance C+_i
                prev_cm = p          # advance C-_j

            # C- reaches the axis
            a_j = self._axis_point(prev_cm)
            axis[j] = a_j
            last_cp[j] = a_j        # reflected C+ starts here

        # -- 3) Wall points ------------------------------------------------
        wall: List[_MeshPoint] = []
        # Wall starts at the corner with theta = theta_max
        M_sw = self._mach_from_nu(theta_max, gamma)
        wall_start = _MeshPoint(
            x=0.0, y=throat_y,
            theta=theta_max, nu=theta_max,
            M=M_sw, mu=self._mu(M_sw))

        prev_wall = wall_start
        for i in range(N):
            pt_cp = last_cp[i]
            w = self._wall_point(pt_cp, prev_wall)
            wall.append(w)
            prev_wall = w

        return {
            "fan":       fan,
            "axis":      axis,
            "interior":  interior,
            "wall":      wall,
            "corner":    corner,
            "theta_max": theta_max,
            "nu_exit":   nu_exit,
            "N":         N,
        }

    # ------------------------------------------------------------------ #
    #  Legacy compatibility interface                                      #
    # ------------------------------------------------------------------ #

    def computeCharacteristics(self) -> List[Dict[str, float]]:
        """Return per-fan-wave data (legacy interface)."""
        N = self.nCharacteristics
        nu_exit = self._prandtl_meyer(self.machExit, self.gamma)
        theta_max = 0.5 * nu_exit
        mach_sw = self._mach_from_nu(theta_max, self.gamma)

        rows: List[Dict[str, float]] = []
        for i in range(1, N + 1):
            theta_i = theta_max * i / N
            nu_i    = theta_i
            mach_i  = self._mach_from_nu(nu_i, self.gamma)
            mu_i    = self._mu(mach_i)
            slope_cm = math.tan(theta_i - mu_i)
            nu_c    = 2.0 * theta_i
            mach_c  = self._mach_from_nu(nu_c, self.gamma)
            mu_c    = self._mu(mach_c)
            slope_cp = math.tan(mu_c)

            rows.append({
                "theta_i":         theta_i,
                "nu_i":            nu_i,
                "mach_fan":        mach_i,
                "mu_fan":          mu_i,
                "slope_c_minus":   slope_cm,
                "nu_centreline":   nu_c,
                "mach_centreline": mach_c,
                "mu_centreline":   mu_c,
                "slope_c_plus":    slope_cp,
                "K_minus":         0.0,
                "K_plus_axis":     nu_c,
                "p_p0_fan":        self._pressure_ratio_from_mach(
                    mach_i, self.gamma),
                "nu_exit":         nu_exit,
                "theta_max":       theta_max,
                "mach_sw":         mach_sw,
            })
        return rows

    # ------------------------------------------------------------------ #
    #  Geometry generation                                                 #
    # ------------------------------------------------------------------ #

    def generateGeometry(self, params: Dict[str, Any]) -> NozzleGeometry:
        """Generate the MLN wall contour from the full MOC mesh."""
        throat_y = float(params["throat_y"])
        exit_y   = float(params["exit_y"])
        length   = float(params["length"])
        n_out    = int(params.get("n_points", 180))

        mesh = self.computeMesh(throat_y)
        wall_pts = mesh["wall"]

        # Build raw wall polyline: corner -> wall points
        raw: List[Tuple[float, float]] = [(0.0, throat_y)]
        for wp in wall_pts:
            if wp.x > raw[-1][0]:
                raw.append((wp.x, wp.y))

        raw_length = raw[-1][0]
        raw_exit_y = raw[-1][1]

        if raw_length < 1e-12 or raw_exit_y <= throat_y:
            # Fallback to parametric builder
            from .MinimumLengthNozzle import MinimumLengthNozzle
            mln = MinimumLengthNozzle.from_params(
                params, self.machExit, self.gamma)
            return mln.build()

        # Scale x to match target length, y to match target exit_y
        sx = length / raw_length
        sy = (exit_y - throat_y) / (raw_exit_y - throat_y)
        scaled: List[Tuple[float, float]] = [
            (x * sx, throat_y + (y - throat_y) * sy)
            for x, y in raw
        ]

        # Resample to n_out evenly-spaced x stations
        xs_out = [i * length / (n_out - 1) for i in range(n_out)]
        ys_out: List[float] = []
        idx = 0
        for xo in xs_out:
            while idx < len(scaled) - 2 and scaled[idx + 1][0] < xo:
                idx += 1
            x0, y0 = scaled[idx]
            x1, y1 = scaled[min(idx + 1, len(scaled) - 1)]
            dx = x1 - x0
            t = (xo - x0) / dx if dx > 1e-14 else 0.0
            t = max(0.0, min(1.0, t))
            ys_out.append(y0 + t * (y1 - y0))

        ys_out[0]  = throat_y
        ys_out[-1] = exit_y
        points = list(zip(xs_out, ys_out))

        geom = NozzleGeometry.fromMOC({
            "wall_points": points,
            "metadata": {
                "generator":        "MOCSolver_mesh",
                "mach_exit":        self.machExit,
                "gamma":            self.gamma,
                "theta_max_deg":    math.degrees(mesh["theta_max"]),
                "nu_exit_deg":      math.degrees(mesh["nu_exit"]),
                "n_characteristics": self.nCharacteristics,
            },
        })
        geom.metadata["x_switch_m"] = (
            mesh["wall"][0].x * sx if mesh["wall"] else length * 0.55
        )
        return geom

    # ------------------------------------------------------------------ #
    #  Characteristic-network plot  (textbook style)                       #
    # ------------------------------------------------------------------ #

    def plotCharacteristics(
        self,
        geometry: NozzleGeometry,
        savepath: str,
        n_lines: int | None = None,
        dpi: int = 220,
    ) -> None:
        """Draw the 2-D planar isentropic MOC characteristic network.

        Produces a clean, textbook-style figure (cf. Anderson, Modern
        Compressible Flow; Zucrow & Hoffman) showing:

        * Expansion-fan C- characteristics (throat corner -> axis).
        * Reflected C+ characteristics (axis -> wall).
        * Internal intersection lattice.
        * Sonic line (arc at the throat).
        * Straightening-section annotation.
        * Exit Mach number arrow.
        """
        y_t = geometry.throat_radius
        L   = geometry.length
        y_e = geometry.exit_radius

        # (Re-)compute the mesh at the requested resolution
        N_plot = n_lines if n_lines and n_lines >= 3 else self.nCharacteristics
        saved_N = self.nCharacteristics
        self.nCharacteristics = N_plot
        mesh = self.computeMesh(y_t)
        self.nCharacteristics = saved_N

        N         = mesh["N"]
        fan       = mesh["fan"]
        axis_pts  = mesh["axis"]
        interior  = mesh["interior"]
        wall_pts  = mesh["wall"]
        theta_max = mesh["theta_max"]
        nu_exit   = mesh["nu_exit"]
        corner    = mesh["corner"]

        # -- Rescale mesh to match geometry length / exit_y ----------------
        raw_L  = wall_pts[-1].x if wall_pts else 1.0
        raw_ye = wall_pts[-1].y if wall_pts else y_t + 1e-3
        sx = L / max(raw_L, 1e-12)
        sy = (y_e - y_t) / max(raw_ye - y_t, 1e-12)

        def scale(p: _MeshPoint) -> Tuple[float, float]:
            return (p.x * sx, y_t + (p.y - y_t) * sy)

        # -- Figure --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 5.0))
        ax.set_aspect("equal", adjustable="box")

        lw_char = 0.55   # characteristic line width
        lw_wall = 2.0
        char_color = "0.35"   # dark gray
        wall_color = "black"

        # ---- 1) Wall contour (upper half) --------------------------------
        wx = [p[0] for p in geometry.control_points]
        wy = [p[1] for p in geometry.control_points]
        ax.plot(wx, wy, color=wall_color, linewidth=lw_wall,
                solid_capstyle="round", zorder=5)
        # Axis (centreline)
        ax.plot([0, L], [0, 0], color="black", linewidth=0.7,
                linestyle="-", zorder=3)

        # ---- 2) C- fan: corner -> internal -> axis -----------------------
        for j in range(N):
            path_xy: List[Tuple[float, float]] = [scale(corner)]
            for i in range(j):
                if (i, j) in interior:
                    path_xy.append(scale(interior[(i, j)]))
            path_xy.append(scale(axis_pts[j]))
            ax.plot([p[0] for p in path_xy], [p[1] for p in path_xy],
                    color=char_color, linewidth=lw_char, zorder=2)

        # ---- 3) C+ reflected: axis -> internal -> wall -------------------
        for i in range(N):
            path_xy = [scale(axis_pts[i])]
            for j in range(i + 1, N):
                if (i, j) in interior:
                    path_xy.append(scale(interior[(i, j)]))
            path_xy.append(scale(wall_pts[i]))
            ax.plot([p[0] for p in path_xy], [p[1] for p in path_xy],
                    color=char_color, linewidth=lw_char, zorder=2)

        # ---- 4) Sonic line (arc at throat) -------------------------------
        n_arc = 50
        arc_xs: List[float] = []
        arc_ys: List[float] = []
        for k in range(n_arc + 1):
            ang = (math.pi / 2) * k / n_arc
            arc_xs.append(L * 0.035 * (1.0 - math.cos(ang)))
            arc_ys.append(y_t * (1.0 - math.sin(ang)))
        ax.plot(arc_xs, arc_ys, color="black", linewidth=1.3,
                linestyle="--", zorder=4)

        # ---- 5) Annotations ---------------------------------------------
        # "Sonic Line" label
        ax.annotate(
            "Sonic Line",
            xy=(L * 0.015, y_t * 0.45),
            fontsize=9, fontstyle="italic", color="black",
            ha="left", va="center", rotation=68,
        )

        # "Straightening Section" bracket at top
        x_ss_start = wall_pts[0].x * sx if wall_pts else L * 0.3
        x_ss_end   = L
        y_bracket  = y_e * 1.15
        ax.annotate(
            "", xy=(x_ss_start, y_bracket),
            xytext=(x_ss_end, y_bracket),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
        )
        ax.text(
            0.5 * (x_ss_start + x_ss_end), y_bracket + y_e * 0.05,
            "Straightening Section",
            ha="center", va="bottom", fontsize=10, fontstyle="italic",
        )

        # Exit Mach arrow
        arr_x = L * 1.02
        arr_y = y_e * 0.55
        ax.annotate(
            "", xy=(arr_x + L * 0.07, arr_y),
            xytext=(arr_x, arr_y),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )
        ax.text(arr_x + L * 0.08, arr_y,
                f"$M_e = {self.machExit:.1f}$",
                fontsize=10, va="center")

        # ---- 6) Axes cosmetics ------------------------------------------
        ax.set_xlabel("$x$  [m]", fontsize=11)
        ax.set_ylabel("$y$  [m]", fontsize=11)
        ax.set_title(
            "Method of Characteristics \u2014 2-D Planar Isentropic "
            f"(Minimum-Length Nozzle,  $\\gamma = {self.gamma}$)",
            fontsize=11, pad=18,
        )
        ax.set_xlim(-L * 0.06, L * 1.22)
        ax.set_ylim(-y_e * 0.08, y_bracket + y_e * 0.14)
        ax.tick_params(labelsize=9)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
