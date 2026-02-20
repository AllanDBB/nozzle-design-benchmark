from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as _np_cpu

# Optional GPU backend via CuPy (transparent NumPy replacement).
try:
    import cupy as _cupy  # type: ignore
    _CUPY_AVAILABLE = True
except ImportError:
    _cupy = None  # type: ignore
    _CUPY_AVAILABLE = False


def _np_backend(use_gpu: bool = False) -> Any:
    """Return cupy if GPU is requested and available, otherwise numpy."""
    if use_gpu and _CUPY_AVAILABLE:
        return _cupy
    return _np_cpu


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
    def _mu_sutherland_vec(temp_arr: Any, np: Any = _np_cpu) -> Any:
        """Vectorized Sutherland viscosity for arrays."""
        t_ref = 273.15
        mu_ref = 1.716e-5
        s = 110.4
        t = np.maximum(temp_arr, 120.0)
        return mu_ref * (t / t_ref) ** 1.5 * (t_ref + s) / (t + s)

    @classmethod
    def _mach_from_area_batch(cls, ar_arr: Any, gamma: float, np: Any = _np_cpu) -> Any:
        """Batch (vectorized) area-ratio to supersonic Mach via bisection on arrays.

        Each element of ar_arr is solved independently in the supersonic branch.
        """
        exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        factor = 2.0 / (gamma + 1.0)
        ar = np.maximum(ar_arr, 1.0 + 1e-9)

        lo = np.full(ar.shape, 1.0 + 1e-7, dtype=float)
        hi = np.full(ar.shape, 12.0, dtype=float)

        def _f(m: Any) -> Any:
            return (1.0 / m) * (factor * (1.0 + 0.5 * (gamma - 1.0) * m * m)) ** exp - ar

        f_lo = _f(lo)
        for _ in range(64):  # 64 bisections → ~1e-19 tolerance
            mid = 0.5 * (lo + hi)
            f_mid = _f(mid)
            go_lo = (f_lo * f_mid) < 0.0
            lo = np.where(go_lo, lo, mid)
            hi = np.where(go_lo, mid, hi)
            f_lo = np.where(go_lo, f_lo, f_mid)
        return 0.5 * (lo + hi)

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

        n_samples = max(int(self.solverConfig.get("n_samples", 120)), 12)

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
        use_gpu = bool(self.solverConfig.get("use_gpu", False))
        np = _np_backend(use_gpu)

        # ------------------------------------------------------------------ #
        #  Geometry sampling (vectorized via np.interp)                       #
        # ------------------------------------------------------------------ #
        xs_np = np.linspace(0.0, self.geometry.length, n_samples)
        # Vectorized piecewise-linear wall lookup using np.interp
        ctrl_x = _np_cpu.array([p[0] for p in self.geometry.control_points])
        ctrl_y = _np_cpu.array([p[1] for p in self.geometry.control_points])
        if use_gpu and _CUPY_AVAILABLE:
            ys_np = np.array(_np_cpu.interp(
                _np_cpu.array(xs_np.get()),
                ctrl_x, ctrl_y,
            ))
        else:
            ys_np = np.interp(xs_np, ctrl_x, ctrl_y)

        # ------------------------------------------------------------------ #
        #  Area / wetted perimeter                                            #
        # ------------------------------------------------------------------ #
        if dimension_mode in ("2d_planar", "3d_channel"):
            area_raw = 2.0 * ys_np * depth
            if dimension_mode == "2d_planar":
                wetted_perimeter = np.full(n_samples, 2.0 * depth)
            else:
                wetted_perimeter = 2.0 * (2.0 * ys_np + depth)
            a_throat = 2.0 * self.geometry.throat_radius * depth
        else:
            area_raw = math.pi * ys_np * ys_np
            wetted_perimeter = 2.0 * math.pi * ys_np
            a_throat = math.pi * self.geometry.throat_radius * self.geometry.throat_radius

        # ------------------------------------------------------------------ #
        #  Boundary-layer displacement thickness (vectorized)                 #
        # ------------------------------------------------------------------ #
        re_ref = float(self.solverConfig.get("reynolds_reference", 2.0e6))
        delta0 = float(self.solverConfig.get("bl_base_fraction", 0.0015)) * self.geometry.throat_radius
        s_norm = xs_np / max(self.geometry.length, 1e-9)
        re_factor = max((re_ref / 2.0e6) ** -0.2, 0.4)
        delta_star = bl_scale * delta0 * re_factor * np.sqrt(np.maximum(s_norm, 0.0))
        y_eff = np.maximum(ys_np - delta_star, 0.35 * ys_np)

        if dimension_mode in ("2d_planar", "3d_channel"):
            area_eff = 2.0 * y_eff * depth
        else:
            area_eff = math.pi * y_eff * y_eff

        # ------------------------------------------------------------------ #
        #  Vectorized area → Mach (batch bisection)                          #
        # ------------------------------------------------------------------ #
        ar = np.maximum(area_eff / max(a_throat, 1e-12), 1.0)
        mach_arr = self._mach_from_area_batch(ar, gamma, np)

        # ------------------------------------------------------------------ #
        #  Isentropic thermodynamics at each station                         #
        # ------------------------------------------------------------------ #
        mach2 = mach_arr * mach_arr
        i_denom = 1.0 + 0.5 * (gamma - 1.0) * mach2            # isentropic factor
        tt_arr = t0 / i_denom                                   # static temperature
        a_sound_arr = np.sqrt(gamma * r * np.maximum(tt_arr, 1e-6))
        u_arr = mach_arr * a_sound_arr

        # Approximate static pressure using p0 (for friction coefficient only)
        p_approx = p0 / i_denom ** (gamma / (gamma - 1.0))
        rho_approx = p_approx / (r * np.maximum(tt_arr, 1e-6))
        mu_arr = self._mu_sutherland_vec(tt_arr, np)
        dh_arr = np.maximum(4.0 * area_eff / np.maximum(wetted_perimeter, 1e-9), 1e-6)
        re_arr = np.maximum(rho_approx * u_arr * dh_arr / np.maximum(mu_arr, 1e-12), 1e3)
        cf_arr = cf_multiplier * 0.074 / re_arr ** 0.2

        # ------------------------------------------------------------------ #
        #  Curvature / friction losses → cumulative total pressure            #
        # ------------------------------------------------------------------ #
        dx_arr = np.diff(xs_np)                          # shape (n-1,)
        dy_arr = np.diff(ys_np)
        theta_arr = np.arctan2(dy_arr, np.maximum(dx_arr, 1e-12))
        dtheta_arr = np.diff(np.concatenate([np.array([0.0]), theta_arr]))  # (n-1,)

        dpt_fric = friction_scale * (4.0 * cf_arr[1:] * dx_arr / dh_arr[1:])
        dpt_turn = curvature_scale * (dtheta_arr * dtheta_arr)
        dpt_frac = np.minimum(0.20, np.maximum(0.0, dpt_fric + dpt_turn))

        # pt[0]=p0; pt[i] = p0 * prod_{k=1}^{i}(1 - dpt_frac[k])
        pt_arr = np.concatenate([np.array([p0]), p0 * np.cumprod(1.0 - dpt_frac)])

        # Final static pressure using actual pt
        p_static_arr = pt_arr / i_denom ** (gamma / (gamma - 1.0))

        # ------------------------------------------------------------------ #
        #  Convert to Python lists for output                                 #
        # ------------------------------------------------------------------ #
        if use_gpu and _CUPY_AVAILABLE:
            xs = xs_np.get().tolist()
            mach_profile: List[float] = mach_arr.get().tolist()
            temp_profile: List[float] = tt_arr.get().tolist()
            pressure_profile: List[float] = p_static_arr.get().tolist()
            pt_profile_list: List[float] = pt_arr.get().tolist()
            p_static_list = pressure_profile
            ys_list = ys_np.get().tolist()
        else:
            xs = xs_np.tolist()
            mach_profile = mach_arr.tolist()
            temp_profile = tt_arr.tolist()
            pressure_profile = p_static_arr.tolist()
            pt_profile_list = pt_arr.tolist()
            p_static_list = pressure_profile
            ys_list = ys_np.tolist()

        mach_exit: float = mach_profile[-1]
        p_exit: float = pressure_profile[-1]
        pt_exit: float = pt_profile_list[-1]

        # ------------------------------------------------------------------ #
        #  Optional overexpanded-jet shock correction                         #
        # ------------------------------------------------------------------ #
        shock_applied = False
        if shock_model and mach_exit > 1.2 and p_exit < shock_trigger_ratio * pa:
            shock_applied = True
            p2_p1 = self._normal_shock_p2_p1(mach_exit, gamma)
            pt2_pt1 = self._normal_shock_pt2_pt1(mach_exit, gamma)
            mach_exit = self._normal_shock_m2(mach_exit, gamma)
            p_exit = p_exit * p2_p1
            pt_exit = pt_exit * pt2_pt1

        t_exit = t0 / (1.0 + 0.5 * (gamma - 1.0) * mach_exit * mach_exit)
        a_exit_sound = math.sqrt(gamma * r * max(t_exit, 1e-6))
        v_exit = mach_exit * a_exit_sound

        # Keep profiles consistent with post-shock exit state.
        if mach_profile:
            mach_profile[-1] = mach_exit
        if temp_profile:
            temp_profile[-1] = t_exit
        if pressure_profile:
            pressure_profile[-1] = p_exit
        if pt_profile_list:
            pt_profile_list[-1] = pt_exit

        # ------------------------------------------------------------------ #
        #  Divergence efficiency & thrust                                     #
        # ------------------------------------------------------------------ #
        exit_angle = 0.0
        wall_angles = self.geometry.wall_angles()
        if wall_angles:
            exit_angle = abs(wall_angles[-1])
        eta_div = max(0.65, 1.0 - divergence_scale * 0.5 * exit_angle * exit_angle)

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
                "area_ratio_effective_exit": float(area_eff[-1]) / max(a_throat, 1e-12),
                "pt_exit": pt_exit,
                "p_exit": p_exit,
                "mdot_used": mdot,
                "mdot_choked": mdot_choked,
                "shock_applied": shock_applied,
                "exit_angle_rad": exit_angle,
                "eta_div": eta_div,
                "dimension": dimension_mode,
                "use_gpu": use_gpu and _CUPY_AVAILABLE,
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
