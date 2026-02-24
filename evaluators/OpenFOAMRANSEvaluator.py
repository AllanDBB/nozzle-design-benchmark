from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import re
import shutil
import subprocess

from geometry import NozzleGeometry
from .EvaluationResult import EvaluationResult
from .CFDSimulation import CFDSimulation


@dataclass
class OpenFOAMRANSEvaluator:
    """Steady compressible RANS evaluator using OpenFOAM (shockFluid — density-based Kurganov).

    Supports:
    - 2D planar (single-cell thickness, front/back = empty)
    - 3D channel (finite depth, front/back = wall)
    """

    geometry: NozzleGeometry
    solverConfig: Dict[str, Any]
    resultPath: str
    _lastResult: Optional[EvaluationResult] = None

    def _campaign(self) -> str:
        return str(self.solverConfig.get("campaign", "")).strip().lower()

    def _sampling_nx(self) -> int:
        return max(8, int(self.solverConfig.get("sampling_nx", 81)))

    def _sampling_x(self) -> List[float]:
        n = self._sampling_nx()
        if n <= 1:
            return [0.0]
        length = max(float(self.geometry.length), 1e-9)
        return [i * length / (n - 1) for i in range(n)]

    def _outlet_bc_mode(self) -> str:
        mode = str(self.solverConfig.get("outlet_bc_mode", "")).strip().lower()
        if mode:
            return mode

        campaign = self._campaign()
        if campaign == "design_supersonic":
            return "wave_transmissive"
        if campaign == "overexpanded_sea_level":
            return "fixed_pressure"
        return "wave_transmissive"

    def _validate_campaign(self) -> None:
        campaign = self._campaign()
        if campaign not in ("", "design_supersonic", "overexpanded_sea_level"):
            raise ValueError(f"Unsupported evaluator.campaign: {campaign}")

        if campaign == "":
            return

        p0 = float(self.solverConfig.get("stagnation_pressure", 3.0e5))
        pa = float(self.solverConfig.get("ambient_pressure", 9.0e4))
        if p0 <= 0.0:
            raise ValueError("stagnation_pressure must be > 0")
        ratio = pa / p0

        # Use MOC config pressure ratio target if provided by pipeline merge.
        ratio_design = float(self.solverConfig.get("design_pressure_ratio", self.solverConfig.get("pressure_ratio", 0.10)))
        ratio_design = max(1e-6, ratio_design)

        if campaign == "design_supersonic" and ratio > max(0.2, 1.6 * ratio_design):
            raise ValueError(
                f"campaign=design_supersonic incompatible with pa/p0={ratio:.3f}; "
                f"expected near design ratio ~{ratio_design:.3f}"
            )
        if campaign == "overexpanded_sea_level" and ratio <= min(0.2, 1.25 * ratio_design):
            raise ValueError(
                f"campaign=overexpanded_sea_level incompatible with pa/p0={ratio:.3f}; "
                f"expected clearly overexpanded condition above design ratio ~{ratio_design:.3f}"
            )

    def _run_cmd(self, cmd: str, cwd: Path) -> str:
        proc = subprocess.run(
            ["bash", "-lc", f"source /opt/openfoam13/etc/bashrc >/dev/null 2>&1 || true; {cmd}"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenFOAM command failed: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return proc.stdout + proc.stderr

    @staticmethod
    def _parse_delta_t_from_line(line: str) -> Optional[float]:
        m = re.search(r"deltaT\s*=\s*([0-9eE+\-\.]+)", line)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _update_delta_t_abort_state(self, delta_t: float, streak: int) -> Tuple[int, bool]:
        if not bool(self.solverConfig.get("enable_delta_t_abort", True)):
            return streak, False
        threshold = float(self.solverConfig.get("delta_t_abort_threshold", 1e-80))
        needed = max(1, int(self.solverConfig.get("delta_t_abort_streak", 20)))
        if delta_t <= 0.0 or not math.isfinite(delta_t):
            streak += 1
        elif delta_t < threshold:
            streak += 1
        else:
            streak = 0
        return streak, streak >= needed

    def _run_shockfluid(self, case_dir: Path) -> None:
        cmd = "foamRun -solver shockFluid"
        log_path = case_dir / "log.shockFluid"
        proc = subprocess.Popen(
            ["bash", "-lc", f"source /opt/openfoam13/etc/bashrc >/dev/null 2>&1 || true; {cmd}"],
            cwd=str(case_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        streak = 0
        last_dt = None
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as logf:
                if proc.stdout is None:
                    raise RuntimeError("Unable to capture shockFluid stdout")
                for line in proc.stdout:
                    logf.write(line)
                    dt = self._parse_delta_t_from_line(line)
                    if dt is not None:
                        last_dt = dt
                        streak, abort_now = self._update_delta_t_abort_state(dt, streak)
                        if abort_now:
                            threshold = float(self.solverConfig.get("delta_t_abort_threshold", 1e-80))
                            needed = max(1, int(self.solverConfig.get("delta_t_abort_streak", 20)))
                            msg = (
                                f"Early-abort shockFluid: deltaT < {threshold:.3e} "
                                f"for {needed} consecutive steps (last deltaT={dt:.3e})"
                            )
                            logf.write(f"\n{msg}\n")
                            logf.flush()
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait(timeout=5)
                            raise RuntimeError(msg)
                rc = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        if rc != 0:
            extra = f" (last deltaT={last_dt:.3e})" if last_dt is not None else ""
            raise RuntimeError(
                f"OpenFOAM command failed: {cmd}{extra}\n"
                f"Check {log_path} for details."
            )

    def _case_dir(self, gid: str) -> Path:
        case_root = Path(self.resultPath) / "openfoam_cases"
        case_root.mkdir(parents=True, exist_ok=True)
        case_dir = case_root / gid
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def _write_control_dict(self, case_dir: Path) -> None:
        # end_time is in physical seconds (e.g. 0.005 = 5 ms ≈ 25 flow-through times).
        # deltaT is just the initial guess; adjustTimeStep + maxCo will grow it
        # automatically up to the CFL-stable limit every step.
        end_time = float(self.solverConfig.get("end_time", 0.005))
        write_interval = float(self.solverConfig.get("write_interval", end_time / 5.0))
        max_co = float(self.solverConfig.get("max_co", 0.5))
        dt_init = float(self.solverConfig.get("delta_t_init", 1e-7))
        xs = self._sampling_x()
        probe_locations = "\n".join([f"            ({x:.9g} 0 0)" for x in xs])
        txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}}

solver          shockFluid;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          {dt_init};
adjustTimeStep  yes;
maxCo           {max_co};
writeControl    runTime;
writeInterval   {write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// Function objects: outlet performance, centerline profiles, and wall pressure dynamics.
functions
{{
    outletScalars
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        surfaceFormat   none;
        patch           outlet;
        operation       areaAverage;
        fields          (p T);
        writeFields     false;
    }}

    outletVelocity
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        surfaceFormat   none;
        patch           outlet;
        operation       areaAverage;
        fields          (U);
        writeFields     false;
    }}

    outletMassFlow
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        surfaceFormat   none;
        patch           outlet;
        operation       sum;
        fields          (phi);
        writeFields     false;
    }}

    upperWallPressure
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        surfaceFormat   none;
        patch           upperWall;
        operation       areaAverage;
        fields          (p);
        writeFields     false;
    }}

    lowerWallPressure
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        surfaceFormat   none;
        patch           lowerWall;
        operation       areaAverage;
        fields          (p);
        writeFields     false;
    }}

    centerlineProbes
    {{
        type            probes;
        libs            ("libsampling.so");
        writeControl    writeTime;
        fields          (p T U rho);
        probeLocations
        (
{probe_locations}
        );
    }}
}}
"""
        (case_dir / "system" / "controlDict").write_text(txt, encoding="utf-8")

    def _write_fv_schemes(self, case_dir: Path) -> None:
        txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}

// shockFluid: density-based Kurganov scheme.
// No pressure equation — rho/rhoU/rhoE solved directly.
// vanAlbada reconstructors provide TVD-like limiting without negative densities.
fluxScheme      Kurganov;

ddtSchemes
{
    default Euler;
}

gradSchemes
{
    default         Gauss linear;
    limited         cellLimited Gauss linear 1;
    grad(U)         $limited;
    grad(k)         $limited;
    grad(omega)     $limited;
}

divSchemes
{
    default         none;

    // More dissipative bounded upwind helps keep k/omega positive near shocks.
    turbulence      bounded Gauss upwind;
    div(phi,k)      $turbulence;
    div(phi,omega)  $turbulence;

    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;

    reconstruct(rho)    vanAlbada;
    reconstruct(U)      vanAlbadaV;
    reconstruct(T)      vanAlbada;
}

snGradSchemes
{
    default corrected;
}

wallDist
{
    method meshWave;
}
"""
        (case_dir / "system" / "fvSchemes").write_text(txt, encoding="utf-8")

    def _write_fv_solution(self, case_dir: Path) -> None:
        txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}

// shockFluid: density-based solver — no pressure equation.
// The conserved variables (rho, rhoU, rhoE) are updated from the Kurganov
// flux divergence, then U and e are extracted from them.
solvers
{
    "rho.*"
    {
        solver          diagonal;
    }

    "(U|e|k|omega).*"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         2;
        tolerance       1e-9;
        relTol          0.01;
    }
}

PIMPLE
{
    nOuterCorrectors 1;
}
"""
        (case_dir / "system" / "fvSolution").write_text(txt, encoding="utf-8")

        # fvConstraints: limitTemperature prevents SIGFPE from negative internal
        # energy during the transient supersonic startup — shocks can temporarily
        # push T negative in a cell before the TVD limiters stabilise the field.
        fc_txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvConstraints;
}

limitT
{
    type        limitTemperature;
    cellZone    all;
    min         50;
    max         5000;
}

boundK
{
    type        bound;
    field       k;
    min         1e-10;
}

boundOmega
{
    type        bound;
    field       omega;
    min         1e-8;
}

boundRho
{
    type        bound;
    field       rho;
    min         1e-4;
}
"""
        (case_dir / "system" / "fvConstraints").write_text(fc_txt, encoding="utf-8")

        fm_txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvModels;
}
"""
        (case_dir / "system" / "fvModels").write_text(fm_txt, encoding="utf-8")

    def _write_thermo(self, case_dir: Path) -> None:
        txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object physicalProperties;
}

// shockFluid requires sensibleInternalEnergy + eConst + hePsiThermo.
// Sutherland transport is physically correct for high-T compressible flows.
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          eConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}

mixture
{
    specie
    {
        molWeight   28.96;   // air
    }
    thermodynamics
    {
        Cv          717.5;   // Cv = Cp/gamma = 1004.5/1.4
        Hf          0;
    }
    transport
    {
        As          1.458e-06;
        Ts          110.4;
    }
}
"""
        (case_dir / "constant" / "physicalProperties").write_text(txt, encoding="utf-8")

    def _write_turbulence(self, case_dir: Path) -> None:
        txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object momentumTransport;
}

simulationType RAS;

RAS
{
    // kOmegaSST: better near-wall and free-shear behaviour than kEpsilon,
    // and used in the canonical diffuserIntake shockFluid tutorial.
    model           kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""
        (case_dir / "constant" / "momentumTransport").write_text(txt, encoding="utf-8")

    def _write_transport(self, case_dir: Path) -> None:
        txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object transportProperties;
}

transportModel Newtonian;
nu              [0 2 -1 0 0 0 0] 1.5e-05;
"""
        (case_dir / "constant" / "transportProperties").write_text(txt, encoding="utf-8")

    def _write_block_mesh(self, case_dir: Path) -> None:
        nx = int(self.solverConfig.get("mesh_nx", 180))
        ny = int(self.solverConfig.get("mesh_ny", 80))
        mode = str(self.solverConfig.get("dimension", "2d_planar"))

        depth = float(self.solverConfig.get("depth", 0.02))
        if mode == "2d_planar":
            nz = 1
            z0 = -0.5 * depth
            z1 = 0.5 * depth
            front_type = "empty"
            back_type = "empty"
        else:
            nz = int(self.solverConfig.get("mesh_nz", 12))
            z0 = -0.5 * depth
            z1 = 0.5 * depth
            # True 3D channel: z faces are walls (not symmetry planes).
            front_type = "wall"
            back_type = "wall"

        points = self.geometry.control_points
        top0 = points[0][1]
        top1 = points[-1][1]
        x0 = points[0][0]
        x1 = points[-1][0]

        top_spline = "\n".join([f"            ({x:.9g} {y:.9g} {z1:.9g})" for x, y in points])
        bot_spline = "\n".join([f"            ({x:.9g} {-y:.9g} {z1:.9g})" for x, y in points])
        top_spline_b = "\n".join([f"            ({x:.9g} {y:.9g} {z0:.9g})" for x, y in points])
        bot_spline_b = "\n".join([f"            ({x:.9g} {-y:.9g} {z0:.9g})" for x, y in points])

        txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}}

convertToMeters 1;

vertices
(
    ({x0:.9g} {-top0:.9g} {z0:.9g})
    ({x1:.9g} {-top1:.9g} {z0:.9g})
    ({x1:.9g} {top1:.9g} {z0:.9g})
    ({x0:.9g} {top0:.9g} {z0:.9g})
    ({x0:.9g} {-top0:.9g} {z1:.9g})
    ({x1:.9g} {-top1:.9g} {z1:.9g})
    ({x1:.9g} {top1:.9g} {z1:.9g})
    ({x0:.9g} {top0:.9g} {z1:.9g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
    spline 7 6
    (
{top_spline}
    )
    spline 4 5
    (
{bot_spline}
    )
    spline 3 2
    (
{top_spline_b}
    )
    spline 0 1
    (
{bot_spline_b}
    )
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    upperWall
    {{
        type wall;
        faces ((3 7 6 2));
    }}
    lowerWall
    {{
        type wall;
        faces ((0 1 5 4));
    }}
    front
    {{
        type {front_type};
        faces ((0 3 2 1));
    }}
    back
    {{
        type {back_type};
        faces ((4 5 6 7));
    }}
);

mergePatchPairs();
"""
        (case_dir / "system" / "blockMeshDict").write_text(txt, encoding="utf-8")

    def _isentropic_mach_from_area_ratio(self, AR: float, gamma: float, supersonic: bool) -> float:
        """Solve A/A* → Mach via isentropic area relation (Newton iteration)."""
        import math as _m
        # A/A* = (1/M) * ((2/(γ+1)) * (1 + (γ-1)/2 * M²))^((γ+1)/(2*(γ-1)))
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        exp = gp1 / (2.0 * gm1)
        M = 2.0 if supersonic else 0.3  # initial guess
        for _ in range(80):
            fac = 1.0 + 0.5 * gm1 * M * M
            AR_calc = (1.0 / M) * (2.0 / gp1 * fac) ** exp
            # dAR/dM
            dAR = (AR_calc / M) * (-1.0 + M * M * gm1 / fac)
            err = AR_calc - AR
            if abs(err) < 1e-10 * AR:
                break
            M -= err / dAR
            M = max(1e-6, M) if not supersonic else max(1.0001, M)
        return M

    def _write_1d_isentropic_ic(self, case_dir: Path) -> None:
        """Write non-uniform initial fields based on 1-D isentropic profile.

        Cell centres are estimated from blockMesh geometry.  In the converging
        section cells get the subsonic isentropic state; in the diverging section
        they get the supersonic state.  This "hot start" lets shockFluid reach
        the quasi-steady supersonic solution within 1-2 flow-through times
        without the transient shocks that crash the solver from a uniform IC.
        """
        import math as _m

        p0 = float(self.solverConfig.get("stagnation_pressure", 1.5e6))
        pa = float(self.solverConfig.get("ambient_pressure", 1.0e5))
        t0_gas = float(self.solverConfig.get("stagnation_temperature", 1000.0))
        k_in = float(self.solverConfig.get("k_inlet", 5.0))
        gamma = float(self.solverConfig.get("gamma", 1.4))
        _R = 287.0
        nx = int(self.solverConfig.get("mesh_nx", 80))
        ny = int(self.solverConfig.get("mesh_ny", 36))
        mode = str(self.solverConfig.get("dimension", "2d_planar"))
        nz = 1 if mode == "2d_planar" else int(self.solverConfig.get("mesh_nz", 12))

        pts = self.geometry.control_points  # list of (x, y) — nozzle wall

        # Find throat: minimum y value in control points → maximum curvature
        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        x0_geo = x_coords[0]
        x1_geo = x_coords[-1]
        y0 = y_coords[0]  # inlet half-height
        yn = y_coords[-1]  # exit half-height
        ithroat = min(range(len(y_coords)), key=lambda i: y_coords[i])
        x_throat = x_coords[ithroat]
        y_throat = y_coords[ithroat]
        A_throat = y_throat  # 2D: area ∝ y (half-height per unit depth)

        # Build interpolating function: x → half-height y(x)
        import bisect as _bs

        def y_at_x(x: float) -> float:
            if x <= x0_geo:
                return y0
            if x >= x1_geo:
                return yn
            i = _bs.bisect_left(x_coords, x)
            if i == 0:
                return y_coords[0]
            if i >= len(x_coords):
                return y_coords[-1]
            x_lo, x_hi = x_coords[i - 1], x_coords[i]
            y_lo, y_hi = y_coords[i - 1], y_coords[i]
            t = (x - x_lo) / (x_hi - x_lo + 1e-30)
            return y_lo + t * (y_hi - y_lo)

        # Cell centres along x — blockMesh uniform spacing
        dx = (x1_geo - x0_geo) / nx
        x_centers = [x0_geo + (i + 0.5) * dx for i in range(nx)]

        # Per-cell 1D isentropic state
        p_cells: list[float] = []
        t_cells: list[float] = []
        u_cells: list[float] = []

        for xc in x_centers:
            y_local = max(y_at_x(xc), 1e-6)
            AR = y_local / A_throat  # area ratio (per unit depth)
            supersonic = (xc > x_throat)
            if AR <= 1.0 + 1e-4:
                M = 1.0
            else:
                M = self._isentropic_mach_from_area_ratio(AR, gamma, supersonic)
            fac = 1.0 + 0.5 * (gamma - 1.0) * M * M
            p_c = p0 / fac ** (gamma / (gamma - 1.0))
            t_c = t0_gas / fac
            a_c = _m.sqrt(gamma * _R * max(t_c, 1.0))
            u_c = M * a_c
            p_cells.append(max(p_c, 100.0))
            t_cells.append(max(t_c, 1.0))
            u_cells.append(u_c)

        # Inlet state (M≈0.3 isentropic)
        AR_inlet = y0 / A_throat
        M_inlet = self._isentropic_mach_from_area_ratio(AR_inlet, gamma, False)
        fac_in = 1.0 + 0.5 * (gamma - 1.0) * M_inlet ** 2
        p_inlet = p0 / fac_in ** (gamma / (gamma - 1.0))
        t_inlet = t0_gas / fac_in
        u_inlet = M_inlet * _m.sqrt(gamma * _R * max(t_inlet, 1.0))

        # omega from k
        l_mix = 0.07 * 2.0 * y_throat
        omega_in = _m.sqrt(k_in) / (0.09 ** 0.25 * max(l_mix, 1e-6))

        is_2d = mode == "2d_planar"
        front_p = "empty" if is_2d else "zeroGradient"
        back_p = "empty" if is_2d else "zeroGradient"
        front_u = "empty" if is_2d else "noSlip"
        back_u = "empty" if is_2d else "noSlip"
        front_t = "empty" if is_2d else "zeroGradient"
        back_t = "empty" if is_2d else "zeroGradient"
        front_k = "empty" if is_2d else "kqRWallFunction"
        back_k = "empty" if is_2d else "kqRWallFunction"
        front_om = "empty" if is_2d else "omegaWallFunction"
        back_om = "empty" if is_2d else "omegaWallFunction"
        front_nut = "empty" if is_2d else "nutkWallFunction"
        back_nut = "empty" if is_2d else "nutkWallFunction"
        front_alphat = "empty" if is_2d else "compressible::alphatWallFunction"
        back_alphat = "empty" if is_2d else "compressible::alphatWallFunction"

        # Non-uniform internal field: each "strip" of ny*nz cells at same x gets the same value.
        n_cells = nx * ny * nz
        # OpenFOAM blockMesh orders cells: k (z) varies fastest, then j (y), then i (x).
        # So cell index = i*ny*nz + j*nz + k.  For a fixed i (x-slice), all ny*nz cells
        # share the same x-center → same 1D value.
        def _nonuniform_scalar(vals_per_x: list, ny: int, nz: int) -> str:
            n = len(vals_per_x) * ny * nz
            lines = [f"nonuniform List<scalar>", f"{n}", "("]
            for v in vals_per_x:
                for _ in range(ny * nz):
                    lines.append(f"{v:.6g}")
            lines.append(")")
            return "\n".join(lines)

        def _nonuniform_vector(ux_per_x: list, ny: int, nz: int) -> str:
            n = len(ux_per_x) * ny * nz
            lines = [f"nonuniform List<vector>", f"{n}", "("]
            for ux in ux_per_x:
                for _ in range(ny * nz):
                    lines.append(f"({ux:.6g} 0 0)")
            lines.append(")")
            return "\n".join(lines)

        p_nonuniform = _nonuniform_scalar(p_cells, ny, nz)
        t_nonuniform = _nonuniform_scalar(t_cells, ny, nz)
        u_nonuniform = _nonuniform_vector(u_cells, ny, nz)
        k_uniform = k_in
        om_uniform = omega_in

        # lInf: reference length for waveTransmissive acoustic-wave correction.
        l_inf = float(x1_geo - x0_geo)
        outlet_bc_mode = self._outlet_bc_mode()
        if outlet_bc_mode == "fixed_pressure":
            p_outlet_bc = f"""type            fixedValue;
        value           uniform {pa:.4f};"""
        elif outlet_bc_mode == "wave_transmissive":
            p_outlet_bc = f"""type            waveTransmissive;
        field           p;
        psi             psi;
        gamma           {gamma};
        fieldInf        {pa:.4f};
        lInf            {l_inf:.6g};
        value           uniform {pa:.4f};"""
        else:
            raise ValueError(f"Unsupported outlet_bc_mode: {outlet_bc_mode}")

        p_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}
dimensions [1 -1 -2 0 0 0 0];
internalField {p_nonuniform};
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {p_inlet:.4f};
    }}
    outlet
    {{
        {p_outlet_bc}
    }}
    upperWall {{ type zeroGradient; }}
    lowerWall {{ type zeroGradient; }}
    front {{ type {front_p}; }}
    back {{ type {back_p}; }}
}}
"""

        u_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField {u_nonuniform};
boundaryField
{{
    inlet     {{ type fixedValue; value uniform ({u_inlet:.4f} 0 0); }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform (0 0 0);
    }}
    upperWall {{ type noSlip; }}
    lowerWall {{ type noSlip; }}
    front {{ type {front_u}; }}
    back {{ type {back_u}; }}
}}
"""

        t_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object T;
}}
dimensions [0 0 0 1 0 0 0];
internalField {t_nonuniform};
boundaryField
{{
    inlet     {{ type fixedValue; value uniform {t_inlet:.2f}; }}
    outlet    {{ type zeroGradient; }}
    upperWall {{ type zeroGradient; }}
    lowerWall {{ type zeroGradient; }}
    front {{ type {front_t}; }}
    back {{ type {back_t}; }}
}}
"""

        k_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object k;
}}
dimensions [0 2 -2 0 0 0 0];
internalField uniform {k_uniform};
boundaryField
{{
    inlet     {{ type turbulentIntensityKineticEnergyInlet; intensity 0.005; value uniform {k_uniform}; }}
    outlet    {{ type inletOutlet; inletValue uniform {k_uniform}; value uniform {k_uniform}; }}
    upperWall {{ type kqRWallFunction; value uniform 1e-10; }}
    lowerWall {{ type kqRWallFunction; value uniform 1e-10; }}
    front {{ type {front_k}; {'value uniform 1e-10;' if not is_2d else ''} }}
    back {{ type {back_k}; {'value uniform 1e-10;' if not is_2d else ''} }}
}}
"""

        om_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object omega;
}}
dimensions [0 0 -1 0 0 0 0];
internalField uniform {om_uniform:.2f};
boundaryField
{{
    inlet     {{ type fixedValue; value uniform {om_uniform:.2f}; }}
    outlet    {{ type inletOutlet; inletValue uniform {om_uniform:.2f}; value uniform {om_uniform:.2f}; }}
    upperWall {{ type omegaWallFunction; value uniform {om_uniform:.2f}; }}
    lowerWall {{ type omegaWallFunction; value uniform {om_uniform:.2f}; }}
    front {{ type {front_om}; {'value uniform ' + f'{om_uniform:.2f};' if not is_2d else ''} }}
    back {{ type {back_om}; {'value uniform ' + f'{om_uniform:.2f};' if not is_2d else ''} }}
}}
"""

        nut_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object nut;
}}
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet     {{ type calculated; value uniform 0; }}
    outlet    {{ type calculated; value uniform 0; }}
    upperWall {{ type nutkWallFunction; value uniform 0; }}
    lowerWall {{ type nutkWallFunction; value uniform 0; }}
    front {{ type {front_nut}; {'value uniform 0;' if not is_2d else ''} }}
    back {{ type {back_nut}; {'value uniform 0;' if not is_2d else ''} }}
}}
"""

        alphat_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object alphat;
}}
dimensions [1 -1 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet     {{ type calculated; value uniform 0; }}
    outlet    {{ type calculated; value uniform 0; }}
    upperWall {{ type compressible::alphatWallFunction; value uniform 0; }}
    lowerWall {{ type compressible::alphatWallFunction; value uniform 0; }}
    front {{ type {front_alphat}; {'value uniform 0;' if not is_2d else ''} }}
    back {{ type {back_alphat}; {'value uniform 0;' if not is_2d else ''} }}
}}
"""

        (case_dir / "0" / "p").write_text(p_txt, encoding="utf-8")
        (case_dir / "0" / "U").write_text(u_txt, encoding="utf-8")
        (case_dir / "0" / "T").write_text(t_txt, encoding="utf-8")
        (case_dir / "0" / "k").write_text(k_txt, encoding="utf-8")
        (case_dir / "0" / "omega").write_text(om_txt, encoding="utf-8")
        (case_dir / "0" / "nut").write_text(nut_txt, encoding="utf-8")
        (case_dir / "0" / "alphat").write_text(alphat_txt, encoding="utf-8")

    def _parse_postprocessing_dat(self, dat_path: Path) -> List[float]:
        """Parse the last data row of a surfaceFieldValue.dat into a flat float list.

        Handles scalar fields ``(time  val)`` and vector fields
        ``(time  (Ux Uy Uz))`` by stripping parentheses.
        Returns an empty list if the file is missing or empty.
        """
        if not dat_path.exists():
            return []
        text = dat_path.read_text(encoding="utf-8", errors="replace")
        data_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not data_lines:
            return []
        last = data_lines[-1].replace("(", " ").replace(")", " ")
        try:
            return [float(v) for v in last.split()]
        except ValueError:
            return []

    def _extract_outlet_values(self, case_dir: Path) -> "tuple[float, float, float, float]":
        """Read averaged outlet p, mag(U), T and mass-flow from postProcessing.

        Returns ``(p_out, u_out, t_out, mdot)``.
        Raises RuntimeError if the required function-object output is absent.
        """
        post = case_dir / "postProcessing"

        def _latest_dat(name: str) -> Path:
            d = post / name
            if not d.exists():
                raise RuntimeError(
                    f"postProcessing/{name} not found – function objects may not "
                    f"have run. Check {case_dir / 'log.shockFluid'} for errors."
                )
            time_dirs = sorted(
                [p for p in d.iterdir() if p.is_dir()],
                key=lambda p: float(p.name) if p.name.replace(".", "", 1).isdigit() else 0.0,
            )
            if not time_dirs:
                raise RuntimeError(f"No time directories found in postProcessing/{name}")
            return time_dirs[-1] / "surfaceFieldValue.dat"

        # ── outlet scalars: [time, p, T] ──────────────────────────────────
        sc = self._parse_postprocessing_dat(_latest_dat("outletScalars"))
        if len(sc) < 3:
            raise RuntimeError(f"outletScalars parse failed, got {sc}")
        p_out = sc[1]
        t_out = sc[2]

        # ── outlet velocity: [time, Ux, Uy, Uz] ──────────────────────────
        vel = self._parse_postprocessing_dat(_latest_dat("outletVelocity"))
        if len(vel) < 4:
            raise RuntimeError(f"outletVelocity parse failed, got {vel}")
        u_out = math.sqrt(vel[1] ** 2 + vel[2] ** 2 + vel[3] ** 2)

        # ── outlet mass-flow: [time, phi_sum] ─────────────────────────────
        phi = self._parse_postprocessing_dat(_latest_dat("outletMassFlow"))
        if len(phi) < 2:
            raise RuntimeError(f"outletMassFlow parse failed, got {phi}")
        mdot = abs(phi[1])

        return p_out, u_out, t_out, mdot

    def _parse_postprocessing_series(self, dat_path: Path) -> List[List[float]]:
        if not dat_path.exists():
            return []
        rows: List[List[float]] = []
        for line in dat_path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            clean = raw.replace("(", " ").replace(")", " ")
            try:
                rows.append([float(v) for v in clean.split()])
            except ValueError:
                continue
        return rows

    @staticmethod
    def _latest_postprocessing_file(post_root: Path, name: str) -> Path:
        d = post_root / name
        if not d.exists():
            raise RuntimeError(f"postProcessing/{name} not found")
        time_dirs = sorted(
            [p for p in d.iterdir() if p.is_dir()],
            key=lambda p: float(p.name) if p.name.replace(".", "", 1).isdigit() else 0.0,
        )
        if not time_dirs:
            raise RuntimeError(f"No time directories found in postProcessing/{name}")
        return time_dirs[-1] / "surfaceFieldValue.dat"

    def _extract_outlet_series(self, case_dir: Path) -> Dict[str, List[float]]:
        post = case_dir / "postProcessing"
        scalars = self._parse_postprocessing_series(self._latest_postprocessing_file(post, "outletScalars"))
        velocity = self._parse_postprocessing_series(self._latest_postprocessing_file(post, "outletVelocity"))
        massflow = self._parse_postprocessing_series(self._latest_postprocessing_file(post, "outletMassFlow"))
        if not scalars or not velocity or not massflow:
            raise RuntimeError("Missing outlet postProcessing series")

        n = min(len(scalars), len(velocity), len(massflow))
        times: List[float] = []
        p: List[float] = []
        t: List[float] = []
        ux: List[float] = []
        uy: List[float] = []
        uz: List[float] = []
        umag: List[float] = []
        mdot: List[float] = []
        for i in range(n):
            sc = scalars[i]
            vel = velocity[i]
            phi = massflow[i]
            if len(sc) < 3 or len(vel) < 4 or len(phi) < 2:
                continue
            times.append(sc[0])
            p.append(sc[1])
            t.append(sc[2])
            ux.append(vel[1])
            uy.append(vel[2])
            uz.append(vel[3])
            umag.append(math.sqrt(vel[1] * vel[1] + vel[2] * vel[2] + vel[3] * vel[3]))
            mdot.append(abs(phi[1]))

        if not times:
            raise RuntimeError("Unable to parse outlet postProcessing series")
        return {"time": times, "p": p, "T": t, "Ux": ux, "Uy": uy, "Uz": uz, "U": umag, "mdot": mdot}

    def _extract_wall_pressure_series(self, case_dir: Path) -> Dict[str, List[float]]:
        post = case_dir / "postProcessing"
        upper_rows = self._parse_postprocessing_series(self._latest_postprocessing_file(post, "upperWallPressure"))
        lower_rows = self._parse_postprocessing_series(self._latest_postprocessing_file(post, "lowerWallPressure"))
        if not upper_rows or not lower_rows:
            raise RuntimeError("Missing wall pressure postProcessing series")

        n = min(len(upper_rows), len(lower_rows))
        times: List[float] = []
        upper: List[float] = []
        lower: List[float] = []
        mean_vals: List[float] = []
        delta_vals: List[float] = []
        for i in range(n):
            up = upper_rows[i]
            lo = lower_rows[i]
            if len(up) < 2 or len(lo) < 2:
                continue
            t = up[0]
            p_up = up[1]
            p_lo = lo[1]
            times.append(t)
            upper.append(p_up)
            lower.append(p_lo)
            mean_vals.append(0.5 * (p_up + p_lo))
            delta_vals.append(p_up - p_lo)
        if not times:
            raise RuntimeError("Unable to parse wall pressure postProcessing series")
        return {
            "time": times,
            "upper": upper,
            "lower": lower,
            "mean": mean_vals,
            "delta": delta_vals,
        }

    def _parse_probe_scalar_file(self, path: Path) -> Tuple[List[float], List[List[float]]]:
        if not path.exists():
            return [], []
        times: List[float] = []
        values: List[List[float]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            times.append(t)
            values.append(vals)
        return times, values

    def _parse_probe_vector_file(self, path: Path) -> Tuple[List[float], List[List[Tuple[float, float, float]]]]:
        if not path.exists():
            return [], []
        times: List[float] = []
        rows: List[List[Tuple[float, float, float]]] = []
        vec_re = re.compile(r"\(([^()]+)\)")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            head = raw.split(maxsplit=1)
            if len(head) < 2:
                continue
            try:
                t = float(head[0])
            except ValueError:
                continue
            vecs: List[Tuple[float, float, float]] = []
            for m in vec_re.finditer(head[1]):
                nums = m.group(1).split()
                if len(nums) != 3:
                    continue
                try:
                    vecs.append((float(nums[0]), float(nums[1]), float(nums[2])))
                except ValueError:
                    continue
            if vecs:
                times.append(t)
                rows.append(vecs)
        return times, rows

    def _extract_centerline_series(self, case_dir: Path) -> Dict[str, Any]:
        root = case_dir / "postProcessing" / "centerlineProbes" / "0"
        if not root.exists():
            raise RuntimeError("centerlineProbes output not found")
        times_p, p_rows = self._parse_probe_scalar_file(root / "p")
        times_t, t_rows = self._parse_probe_scalar_file(root / "T")
        times_rho, rho_rows = self._parse_probe_scalar_file(root / "rho")
        times_u, u_rows = self._parse_probe_vector_file(root / "U")
        if not times_p or not times_t or not times_u:
            raise RuntimeError("centerline probe files are incomplete")

        n = min(len(times_p), len(times_t), len(times_rho) if times_rho else len(times_p), len(times_u))
        gamma = float(self.solverConfig.get("gamma", 1.4))
        r = float(self.solverConfig.get("gas_constant", 287.0))
        xs = self._sampling_x()
        n_probe = len(xs)

        series: List[Dict[str, Any]] = []
        for i in range(n):
            p_vals = p_rows[i][:n_probe]
            t_vals = t_rows[i][:n_probe]
            rho_vals = rho_rows[i][:n_probe] if rho_rows else [0.0] * len(p_vals)
            u_vecs = u_rows[i][:n_probe]
            if not p_vals or not t_vals or not u_vecs:
                continue
            ux = [v[0] for v in u_vecs]
            uy = [v[1] for v in u_vecs]
            uz = [v[2] for v in u_vecs]
            umag = [math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) for v in u_vecs]
            mach = []
            for u_i, t_i in zip(umag, t_vals):
                a_i = math.sqrt(max(gamma * r * max(t_i, 1e-9), 1e-9))
                mach.append(u_i / max(a_i, 1e-9))
            series.append(
                {
                    "time": times_p[i],
                    "p": p_vals,
                    "T": t_vals,
                    "rho": rho_vals,
                    "Ux": ux,
                    "Uy": uy,
                    "Uz": uz,
                    "U": umag,
                    "Mach": mach,
                }
            )
        if not series:
            raise RuntimeError("No usable centerline probe samples")
        return {"x": xs, "series": series}

    @staticmethod
    def _rel_std(values: List[float]) -> float:
        if len(values) < 2:
            return 1.0
        mean = sum(values) / len(values)
        if abs(mean) < 1e-12:
            return 1.0
        var = sum((v - mean) * (v - mean) for v in values) / len(values)
        return math.sqrt(max(var, 0.0)) / abs(mean)

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) * (v - mean) for v in values) / len(values)
        return math.sqrt(max(var, 0.0))

    @staticmethod
    def _window_mean(values: List[float], window: int) -> float:
        if not values:
            return 0.0
        w = max(1, min(int(window), len(values)))
        tail = values[-w:]
        return sum(tail) / len(tail)

    def _compute_wall_pressure_metrics(self, wall: Dict[str, List[float]]) -> Dict[str, float]:
        n = len(wall.get("time", []))
        if n < 2:
            return {}
        window = max(2, int(self.solverConfig.get("convergence_window", 5)))
        w = min(window, n)
        p_mean = wall.get("mean", [])[-w:]
        p_delta = wall.get("delta", [])[-w:]
        rms = self._std(p_mean)
        rel_rms = self._rel_std(p_mean)
        delta_rms = self._std(p_delta)
        return {
            "wall_p_rms": rms,
            "wall_p_rel_rms": rel_rms,
            "wall_p_delta_rms": delta_rms,
            "wall_p_window": w,
        }

    def _compute_series_convergence(self, outlet: Dict[str, List[float]]) -> Dict[str, Any]:
        n = len(outlet.get("time", []))
        min_writes = max(2, int(self.solverConfig.get("min_writes", 8)))
        window = max(2, int(self.solverConfig.get("convergence_window", 5)))
        conv_tol = self.solverConfig.get("convergence_tol", {})
        tol_p = float(conv_tol.get("p_out_rel_std", 0.02))
        tol_mdot = float(conv_tol.get("mdot_rel_std", 0.02))
        tol_ux = float(conv_tol.get("ux_out_rel_std", 0.03))
        if n < min_writes:
            return {
                "converged_series": False,
                "n_writes": n,
                "min_writes": min_writes,
                "window": window,
                "reason": "not_enough_writes",
                "metrics": {},
            }

        w = min(window, n)
        p_rel = self._rel_std(outlet["p"][-w:])
        mdot_rel = self._rel_std(outlet["mdot"][-w:])
        ux_rel = self._rel_std(outlet["Ux"][-w:])
        converged_series = (p_rel <= tol_p) and (mdot_rel <= tol_mdot) and (ux_rel <= tol_ux)
        return {
            "converged_series": converged_series,
            "n_writes": n,
            "min_writes": min_writes,
            "window": w,
            "reason": "ok" if converged_series else "window_std_exceeded",
            "metrics": {
                "p_out_rel_std": p_rel,
                "mdot_rel_std": mdot_rel,
                "ux_out_rel_std": ux_rel,
            },
            "tolerances": {
                "p_out_rel_std": tol_p,
                "mdot_rel_std": tol_mdot,
                "ux_out_rel_std": tol_ux,
            },
        }

    def _series_gate(self, convergence: Dict[str, Any]) -> Tuple[bool, str]:
        require = bool(self.solverConfig.get("require_converged_series", True))
        if not require:
            return True, "series_requirement_disabled"

        if bool(convergence.get("converged_series", False)):
            return True, "steady_window"

        campaign = self._campaign()
        if campaign != "overexpanded_sea_level":
            return False, str(convergence.get("reason", "window_std_exceeded"))

        if not bool(self.solverConfig.get("allow_unsteady_overexpanded", True)):
            return False, str(convergence.get("reason", "window_std_exceeded"))

        n_writes = int(convergence.get("n_writes", 0))
        min_writes = int(convergence.get("min_writes", max(2, int(self.solverConfig.get("min_writes", 8)))))
        metrics = convergence.get("metrics", {}) if isinstance(convergence.get("metrics", {}), dict) else {}
        unsteady_tol = self.solverConfig.get("overexpanded_unsteady_tol", {})
        if not isinstance(unsteady_tol, dict):
            unsteady_tol = {}

        p_max = float(unsteady_tol.get("p_out_rel_std_max", 0.10))
        mdot_max = float(unsteady_tol.get("mdot_rel_std_max", 0.35))
        ux_max = float(unsteady_tol.get("ux_out_rel_std_max", 0.45))

        p_rel = float(metrics.get("p_out_rel_std", 1.0e9))
        mdot_rel = float(metrics.get("mdot_rel_std", 1.0e9))
        ux_rel = float(metrics.get("ux_out_rel_std", 1.0e9))

        if n_writes >= min_writes and p_rel <= p_max and mdot_rel <= mdot_max and ux_rel <= ux_max:
            return True, "accepted_unsteady_overexpanded"
        return False, "unsteady_overexpanded_exceeds_limits"

    @staticmethod
    def _detect_shock_from_profile(x_profile: List[float], mach: List[float], pressure: List[float]) -> Dict[str, Any]:
        n = min(len(x_profile), len(mach), len(pressure))
        if n < 3:
            return {"present": False, "x": None, "strength": None}
        best_idx = -1
        best_strength = 0.0
        for i in range(1, n - 1):
            m_up = 0.5 * (mach[i - 1] + mach[i])
            m_down = 0.5 * (mach[i] + mach[i + 1])
            p_up = max(pressure[i - 1], 1e-9)
            p_down = pressure[i + 1]
            p_ratio = p_down / p_up
            if m_up > 1.15 and m_down < 0.95 and p_ratio > 1.15 and p_ratio > best_strength:
                best_strength = p_ratio
                best_idx = i
        if best_idx < 0:
            return {"present": False, "x": None, "strength": None}
        return {"present": True, "x": x_profile[best_idx], "strength": best_strength}

    def _compute_shock_series(self, centerline: Dict[str, Any]) -> Dict[str, Any]:
        x_profile = centerline.get("x", [])
        series = centerline.get("series", [])
        shock_x_series: List[float] = []
        shock_strength_series: List[float] = []
        for row in series:
            d = self._detect_shock_from_profile(x_profile, row.get("Mach", []), row.get("p", []))
            if d["present"] and d["x"] is not None and d["strength"] is not None:
                shock_x_series.append(float(d["x"]))
                shock_strength_series.append(float(d["strength"]))
        final = self._detect_shock_from_profile(
            x_profile,
            series[-1].get("Mach", []) if series else [],
            series[-1].get("p", []) if series else [],
        )
        if len(shock_x_series) >= 2:
            mean_x = sum(shock_x_series) / len(shock_x_series)
            var_x = sum((x - mean_x) * (x - mean_x) for x in shock_x_series) / len(shock_x_series)
            x_std = math.sqrt(max(var_x, 0.0))
        else:
            x_std = None
        return {
            "present": bool(final.get("present", False)),
            "x": final.get("x"),
            "strength": final.get("strength"),
            "x_series": shock_x_series,
            "strength_series": shock_strength_series,
            "x_std": x_std,
        }

    def _check_convergence(self, case_dir: Path) -> bool:
        """Return True if shockFluid reached the end of its run without abort."""
        log_path = case_dir / "log.shockFluid"
        if not log_path.exists():
            return False
        text = log_path.read_text(encoding="utf-8", errors="replace")
        # shockFluid ends cleanly with "End"; abort shows "FATAL"
        return "End" in text and "FATAL" not in text

    def _build_case(self, case_dir: Path) -> None:
        for p in [
            case_dir / "system",
            case_dir / "constant",
            case_dir / "0",
        ]:
            p.mkdir(parents=True, exist_ok=True)

        self._write_control_dict(case_dir)
        self._write_fv_schemes(case_dir)
        self._write_fv_solution(case_dir)
        self._write_thermo(case_dir)
        self._write_turbulence(case_dir)
        self._write_transport(case_dir)
        self._write_block_mesh(case_dir)
        self._write_1d_isentropic_ic(case_dir)

    def run(self) -> None:
        self._lastResult = self.extractResults()

    def extractResults(self) -> EvaluationResult:
        self._validate_campaign()
        gid = str(self.geometry.metadata.get("id", self.geometry.metadata.get("source", "geometry")))
        case_dir = self._case_dir(gid)
        self._build_case(case_dir)

        try:
            self._run_cmd("blockMesh > log.blockMesh 2>&1", case_dir)
            self._run_cmd("checkMesh > log.checkMesh 2>&1", case_dir)
            self._run_shockfluid(case_dir)
        except Exception as exc:
            if not bool(self.solverConfig.get("fallback_on_failure", False)):
                raise
            fallback = CFDSimulation(
                geometry=self.geometry,
                solverConfig=self.solverConfig,
                resultPath=self.resultPath,
            ).extractResults()
            fallback.metadata["backend"] = "openfoam_fallback"
            fallback.metadata["openfoam_error"] = str(exc)
            fallback.metadata["case_dir"] = str(case_dir)
            self._lastResult = fallback
            return fallback

        converged = self._check_convergence(case_dir)
        if not converged:
            msg = (
                f"shockFluid did not converge for case {gid}. "
                f"Inspect {case_dir / 'log.shockFluid'} for details."
            )
            if not bool(self.solverConfig.get("fallback_on_failure", False)):
                raise RuntimeError(msg)
            # Fallback with convergence warning attached.
            fallback = CFDSimulation(
                geometry=self.geometry,
                solverConfig=self.solverConfig,
                resultPath=self.resultPath,
            ).extractResults()
            fallback.metadata["backend"] = "openfoam_fallback"
            fallback.metadata["convergence_warning"] = msg
            fallback.metadata["case_dir"] = str(case_dir)
            self._lastResult = fallback
            return fallback

        try:
            outlet = self._extract_outlet_series(case_dir)
            centerline = self._extract_centerline_series(case_dir)
        except Exception as exc:
            if not bool(self.solverConfig.get("fallback_on_failure", False)):
                raise
            fallback = CFDSimulation(
                geometry=self.geometry,
                solverConfig=self.solverConfig,
                resultPath=self.resultPath,
            ).extractResults()
            fallback.metadata["backend"] = "openfoam_fallback"
            fallback.metadata["postprocess_error"] = str(exc)
            fallback.metadata["case_dir"] = str(case_dir)
            self._lastResult = fallback
            return fallback
        wall: Dict[str, List[float]] = {}
        wall_warning = ""
        try:
            wall = self._extract_wall_pressure_series(case_dir)
        except Exception as exc:
            wall_warning = str(exc)

        gamma = float(self.solverConfig.get("gamma", 1.4))
        r = float(self.solverConfig.get("gas_constant", 287.0))
        p0 = float(self.solverConfig.get("stagnation_pressure", 1.5e6))
        pa = float(self.solverConfig.get("ambient_pressure", 1.0e4))
        require_converged_series = bool(self.solverConfig.get("require_converged_series", True))
        campaign = self._campaign()
        use_window_averages = (
            campaign == "overexpanded_sea_level"
            and bool(self.solverConfig.get("use_window_averages_overexpanded", True))
        )

        convergence = self._compute_series_convergence(outlet)
        wall_metrics = self._compute_wall_pressure_metrics(wall) if wall else {}
        if wall_metrics:
            convergence.setdefault("metrics", {}).update(wall_metrics)
        series_usable, series_gate_reason = self._series_gate(convergence)
        convergence["series_usable"] = series_usable
        convergence["series_gate_reason"] = series_gate_reason
        if not series_usable:
            msg = f"Series-based acceptance failed for case {gid}: {convergence}"
            if not bool(self.solverConfig.get("fallback_on_failure", False)):
                raise RuntimeError(msg)
            fallback = CFDSimulation(
                geometry=self.geometry,
                solverConfig=self.solverConfig,
                resultPath=self.resultPath,
            ).extractResults()
            fallback.metadata["backend"] = "openfoam_fallback"
            fallback.metadata["convergence_warning"] = msg
            fallback.metadata["case_dir"] = str(case_dir)
            self._lastResult = fallback
            return fallback

        w_avg = int(convergence.get("window", max(2, int(self.solverConfig.get("convergence_window", 5)))))
        if use_window_averages:
            p_out = self._window_mean(outlet["p"], w_avg)
            t_out = self._window_mean(outlet["T"], w_avg)
            ux_out = self._window_mean(outlet["Ux"], w_avg)
            uy_out = self._window_mean(outlet["Uy"], w_avg)
            uz_out = self._window_mean(outlet["Uz"], w_avg)
            u_out = self._window_mean(outlet["U"], w_avg)
            mdot = self._window_mean(outlet["mdot"], w_avg)
        else:
            p_out = outlet["p"][-1]
            t_out = outlet["T"][-1]
            ux_out = outlet["Ux"][-1]
            uy_out = outlet["Uy"][-1]
            uz_out = outlet["Uz"][-1]
            u_out = outlet["U"][-1]
            mdot = outlet["mdot"][-1]

        a = math.sqrt(max(gamma * r * max(t_out, 1e-9), 1e-9))
        m_out = u_out / max(a, 1e-9)
        p0_out = p_out * (1.0 + 0.5 * (gamma - 1.0) * m_out * m_out) ** (gamma / (gamma - 1.0))
        pressure_loss = max(0.0, min(0.999, 1.0 - p0_out / p0))

        dimension = str(self.solverConfig.get("dimension", "2d_planar"))
        depth = float(self.solverConfig.get("depth", 0.02))
        if dimension in ("2d_planar", "3d_channel"):
            a_exit = 2.0 * self.geometry.exit_radius * depth
        else:
            a_exit = math.pi * self.geometry.exit_radius ** 2
        thrust = mdot * u_out + (p_out - pa) * a_exit

        final_center = centerline["series"][-1]
        x_profile = centerline["x"][:]
        mach_profile = final_center["Mach"][:]
        temp_profile = final_center["T"][:]
        pressure_profile = final_center["p"][:]
        velocity_profile = final_center["U"][:]
        shock = self._compute_shock_series(centerline)
        convergence["log_converged"] = converged
        convergence["require_converged_series"] = require_converged_series
        convergence["outlet_time"] = outlet["time"][-1]
        convergence["window_averaged_outlet"] = bool(use_window_averages)
        convergence["window_averaged_size"] = w_avg if use_window_averages else 1

        self._lastResult = EvaluationResult(
            machProfile=mach_profile,
            pressureLoss=pressure_loss,
            thrust=thrust,
            geometryId=gid,
            temperatureProfile=temp_profile,
            pressureProfile=pressure_profile,
            xProfile=x_profile,
            velocityProfile=velocity_profile,
            convergence=convergence,
            shock=shock,
            metadata={
                "backend": "openfoam_rans",
                "converged": converged,
                "dimension": dimension,
                "case_dir": str(case_dir),
                "outlet_pressure": p_out,
                "outlet_ux": ux_out,
                "outlet_uy": uy_out,
                "outlet_uz": uz_out,
                "outlet_velocity": u_out,
                "outlet_temperature": t_out,
                "outlet_mach": m_out,
                "mass_flow": mdot,
                "series_writes": len(outlet["time"]),
                "campaign": campaign or "unspecified",
                "outlet_bc_mode": self._outlet_bc_mode(),
                "shock_present": bool(shock.get("present", False)),
                "shock_x": shock.get("x"),
                "shock_strength": shock.get("strength"),
                "shock_x_std": shock.get("x_std"),
                "series_usable": bool(convergence.get("series_usable", False)),
                "series_gate_reason": convergence.get("series_gate_reason"),
                "window_averaged_outlet": bool(use_window_averages),
                "wall_pressure_rms": wall_metrics.get("wall_p_rms"),
                "wall_pressure_rel_rms": wall_metrics.get("wall_p_rel_rms"),
                "wall_pressure_delta_rms": wall_metrics.get("wall_p_delta_rms"),
                "wall_pressure_warning": wall_warning,
            },
        )
        return self._lastResult

    def computeThrust(self) -> float:
        if self._lastResult is None:
            self._lastResult = self.extractResults()
        return self._lastResult.thrust

    def plotFields(self, savepath: str | None = None) -> None:
        # For RANS backend, field plotting is done with ParaView/OpenFOAM post-processing.
        return None
