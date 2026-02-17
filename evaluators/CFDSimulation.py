from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math
from pathlib import Path

import matplotlib.pyplot as plt

from .EvaluationResult import EvaluationResult
from geometry import NozzleGeometry


@dataclass
class CFDSimulation:
    """CFD-like quasi-1D evaluator with configurable non-ideal effects.

    Physics ingredients (simplified but more realistic than pure isentropic):
    - Isentropic area-Mach backbone (supersonic branch after throat).
    - Effective area reduced by boundary-layer displacement thickness.
    - Total-pressure marching with friction and curvature losses.
    - Optional overexpanded-exit normal-shock correction.
    """

    geometry: NozzleGeometry
    solverConfig: Dict[str, Any]
    resultPath: str
    _lastResult: Optional[EvaluationResult] = None
    _lastX: Optional[List[float]] = None

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
        f_lo = cls._area_mach_ratio(lo, gamma) - area_ratio
        f_hi = cls._area_mach_ratio(hi, gamma) - area_ratio
        if f_lo * f_hi > 0.0:
            # Fallback for extreme ratios outside bracket.
            return hi if f_hi < 0.0 else lo

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = cls._area_mach_ratio(mid, gamma) - area_ratio
            if abs(f_mid) < 1e-10:
                return mid
            if f_lo * f_mid < 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _mu_sutherland(temperature: float) -> float:
        # Air dynamic viscosity [Pa s]
        t_ref = 273.15
        mu_ref = 1.716e-5
        s = 110.4
        t = max(temperature, 120.0)
        return mu_ref * ((t / t_ref) ** 1.5) * (t_ref + s) / (t + s)

    @staticmethod
    def _normal_shock_m2(m1: float, gamma: float) -> float:
        num = 1.0 + 0.5 * (gamma - 1.0) * m1 * m1
        den = gamma * m1 * m1 - 0.5 * (gamma - 1.0)
        return math.sqrt(max(num / max(den, 1e-9), 1e-9))

    @staticmethod
    def _normal_shock_p2_p1(m1: float, gamma: float) -> float:
        return 1.0 + (2.0 * gamma / (gamma + 1.0)) * (m1 * m1 - 1.0)

    @staticmethod
    def _normal_shock_pt2_pt1(m1: float, gamma: float) -> float:
        # Exact normal-shock total-pressure ratio.
        a = ((gamma + 1.0) * m1 * m1) / ((gamma - 1.0) * m1 * m1 + 2.0)
        b = (gamma + 1.0) / (2.0 * gamma * m1 * m1 - (gamma - 1.0))
        return a ** (gamma / (gamma - 1.0)) * b ** (1.0 / (gamma - 1.0))

    def run(self) -> None:
        self._lastResult = self.extractResults()

        outdir = Path(self.resultPath)
        outdir.mkdir(parents=True, exist_ok=True)
        gid = self._lastResult.geometryId
        self._lastResult.saveToJSON(str(outdir / f"{gid}_evaluation.json"))

    def extractResults(self) -> EvaluationResult:
        gamma = float(self.solverConfig.get("gamma", 1.4))
        r = float(self.solverConfig.get("gas_constant", 287.0))
        t0 = float(self.solverConfig.get("stagnation_temperature", 1000.0))
        p0 = float(self.solverConfig.get("stagnation_pressure", 3.0e5))
        pa = float(self.solverConfig.get("ambient_pressure", 101325.0))

        n_samples = int(self.solverConfig.get("n_samples", 120))
        n_samples = max(n_samples, 12)

        friction_scale = float(self.solverConfig.get("friction_scale", 0.2))
        cf_multiplier = float(self.solverConfig.get("cf_multiplier", 1.0))
        curvature_scale = float(self.solverConfig.get("curvature_scale", 0.05))
        bl_scale = float(self.solverConfig.get("bl_displacement_scale", 1.0))
        cd = float(self.solverConfig.get("discharge_coefficient", 0.985))
        shock_model = bool(self.solverConfig.get("enable_shock_model", True))
        shock_trigger_ratio = float(self.solverConfig.get("shock_trigger_ratio", 0.55))
        divergence_scale = float(self.solverConfig.get("divergence_scale", 1.0))
        dimension_mode = str(self.solverConfig.get("dimension", "axisymmetric"))
        depth = float(self.solverConfig.get("depth", 0.02))

        xs = [i * self.geometry.length / (n_samples - 1) for i in range(n_samples)]
        ys = [self.geometry.y_at(x) for x in xs]

        if dimension_mode in ("2d_planar", "3d_channel"):
            area_raw = [2.0 * y * depth for y in ys]
            wetted_perimeter = [
                2.0 * depth if dimension_mode == "2d_planar" else 2.0 * (2.0 * y + depth) for y in ys
            ]
            a_throat = 2.0 * self.geometry.throat_radius * depth
        else:
            area_raw = [math.pi * y * y for y in ys]
            wetted_perimeter = [2.0 * math.pi * y for y in ys]
            a_throat = math.pi * self.geometry.throat_radius * self.geometry.throat_radius

        # Effective area: boundary layer displacement thickness reduces available flow area.
        re_ref = float(self.solverConfig.get("reynolds_reference", 2.0e6))
        delta0 = float(self.solverConfig.get("bl_base_fraction", 0.0015)) * self.geometry.throat_radius
        area_eff: List[float] = []
        for x, y, a in zip(xs, ys, area_raw):
            s = x / max(self.geometry.length, 1e-9)
            # Smooth growth along nozzle, weakly reduced at higher Reynolds.
            re_factor = max((re_ref / 2.0e6) ** -0.2, 0.4)
            delta_star = bl_scale * delta0 * re_factor * math.sqrt(max(s, 0.0))
            y_eff = max(y - delta_star, 0.35 * y)
            if dimension_mode in ("2d_planar", "3d_channel"):
                area_eff.append(2.0 * y_eff * depth)
            else:
                area_eff.append(math.pi * y_eff * y_eff)

        mach_profile: List[float] = []
        temp_profile: List[float] = []
        pressure_profile: List[float] = []
        pt_profile: List[float] = []

        pt = p0
        prev_theta = 0.0

        for i, x in enumerate(xs):
            ar = max(area_eff[i] / max(a_throat, 1e-12), 1.0)
            m = self._mach_from_area(ar, gamma)

            tt = t0 / (1.0 + 0.5 * (gamma - 1.0) * m * m)
            p_static = pt / (1.0 + 0.5 * (gamma - 1.0) * m * m) ** (gamma / (gamma - 1.0))

            a_sound = math.sqrt(gamma * r * max(tt, 1e-6))
            u = m * a_sound
            rho = p_static / (r * max(tt, 1e-6))
            mu = self._mu_sutherland(tt)
            dh = max(4.0 * area_eff[i] / max(wetted_perimeter[i], 1e-9), 1e-6)
            re = max(rho * u * dh / max(mu, 1e-12), 1e3)

            # Turbulent skin-friction (smooth-wall proxy).
            cf = cf_multiplier * 0.074 / (re ** 0.2)

            if i > 0:
                dx = xs[i] - xs[i - 1]
                dy = ys[i] - ys[i - 1]
                theta = math.atan2(dy, max(dx, 1e-12))
                dtheta = theta - prev_theta
                prev_theta = theta

                dpt_fric = friction_scale * (4.0 * cf * dx / dh)
                dpt_turn = curvature_scale * (dtheta * dtheta)
                dpt_frac = min(0.20, max(0.0, dpt_fric + dpt_turn))
                pt *= (1.0 - dpt_frac)
                pt = max(pt, 1e3)

                # Recompute with updated total pressure at this station.
                p_static = pt / (1.0 + 0.5 * (gamma - 1.0) * m * m) ** (gamma / (gamma - 1.0))

            mach_profile.append(m)
            temp_profile.append(tt)
            pressure_profile.append(p_static)
            pt_profile.append(pt)

        mach_exit = mach_profile[-1]
        p_exit = pressure_profile[-1]
        pt_exit = pt_profile[-1]

        shock_applied = False
        if shock_model and mach_exit > 1.2 and p_exit < shock_trigger_ratio * pa:
            # Overexpanded jet -> apply exit normal-shock correction.
            shock_applied = True
            p2_p1 = self._normal_shock_p2_p1(mach_exit, gamma)
            pt2_pt1 = self._normal_shock_pt2_pt1(mach_exit, gamma)
            mach_exit = self._normal_shock_m2(mach_exit, gamma)
            p_exit = p_exit * p2_p1
            pt_exit = pt_exit * pt2_pt1

        t_exit = t0 / (1.0 + 0.5 * (gamma - 1.0) * mach_exit * mach_exit)
        a_exit_sound = math.sqrt(gamma * r * max(t_exit, 1e-6))
        v_exit = mach_exit * a_exit_sound

        exit_angle = 0.0
        wall_angles = self.geometry.wall_angles()
        if wall_angles:
            exit_angle = abs(wall_angles[-1])
        # Divergence efficiency (axial momentum reduction due exit flow angle).
        eta_div = max(0.65, 1.0 - divergence_scale * 0.5 * exit_angle * exit_angle)

        # Choked mass-flow estimate from throat conditions.
        mdot_choked = (
            cd
            * a_throat
            * p0
            / math.sqrt(max(t0, 1e-6))
            * math.sqrt(gamma / r)
            * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )
        mdot = float(self.solverConfig.get("mass_flow", mdot_choked))

        if dimension_mode in ("2d_planar", "3d_channel"):
            a_exit = 2.0 * self.geometry.exit_radius * depth
        else:
            a_exit = math.pi * (self.geometry.exit_radius ** 2)
        thrust = mdot * v_exit * eta_div + (p_exit - pa) * a_exit

        # Non-ideal pressure loss relative to inlet total pressure.
        pressure_loss = max(0.0, min(0.999, 1.0 - (pt_exit / p0)))

        gid = str(self.geometry.metadata.get("id", self.geometry.metadata.get("source", "geometry")))
        self._lastX = xs
        self._lastResult = EvaluationResult(
            machProfile=mach_profile,
            pressureLoss=pressure_loss,
            thrust=thrust,
            geometryId=gid,
            temperatureProfile=temp_profile,
            pressureProfile=pressure_profile,
            metadata={
                "mach_exit": mach_exit,
                "area_ratio_geom": self.geometry.expansion_ratio,
                "area_ratio_effective_exit": area_eff[-1] / max(a_throat, 1e-12),
                "pt_exit": pt_exit,
                "p_exit": p_exit,
                "mdot_used": mdot,
                "mdot_choked": mdot_choked,
                "shock_applied": shock_applied,
                "exit_angle_rad": exit_angle,
                "eta_div": eta_div,
                "dimension": dimension_mode,
            },
        )
        return self._lastResult

    def computeThrust(self) -> float:
        if self._lastResult is None:
            self._lastResult = self.extractResults()
        return self._lastResult.thrust

    def plotFields(self, savepath: str | None = None) -> None:
        if self._lastResult is None:
            self._lastResult = self.extractResults()
        if self._lastX is None:
            return

        fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        ax[0].plot(self._lastX, self._lastResult.machProfile, color="tab:blue")
        ax[0].set_ylabel("Mach")
        ax[0].grid(True, alpha=0.3)
        ax[0].set_title(f"CFD-like Fields - {self._lastResult.geometryId}")

        ax[1].plot(self._lastX, self._lastResult.temperatureProfile, color="tab:red")
        ax[1].set_ylabel("T [K]")
        ax[1].grid(True, alpha=0.3)

        if self._lastResult.pressureProfile:
            ax[2].plot(self._lastX, self._lastResult.pressureProfile, color="tab:green")
        ax[2].set_xlabel("x [m]")
        ax[2].set_ylabel("p [Pa]")
        ax[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=180)
            plt.close()
        else:
            plt.show()
