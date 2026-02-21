from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
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

// Function objects: average outlet scalars and compute mass-flow rate.
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

    turbulence      Gauss limitedLinear 1;
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

        # lInf: reference length for waveTransmissive acoustic-wave correction
        # Use the domain x-length (nozzle length) as the reference
        l_inf = float(x1_geo - x0_geo)

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
        type            waveTransmissive;
        field           p;
        psi             psi;
        gamma           {gamma};
        fieldInf        {pa:.4f};
        lInf            {l_inf:.6g};
        value           uniform {pa:.4f};
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
        gid = str(self.geometry.metadata.get("id", self.geometry.metadata.get("source", "geometry")))
        case_dir = self._case_dir(gid)
        self._build_case(case_dir)

        try:
            self._run_cmd("blockMesh > log.blockMesh 2>&1", case_dir)
            self._run_cmd("checkMesh > log.checkMesh 2>&1", case_dir)
            self._run_cmd("foamRun -solver shockFluid > log.shockFluid 2>&1", case_dir)
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
            p_out, u_out, t_out, mdot = self._extract_outlet_values(case_dir)
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

        gamma = float(self.solverConfig.get("gamma", 1.4))
        r = float(self.solverConfig.get("gas_constant", 287.0))
        p0 = float(self.solverConfig.get("stagnation_pressure", 1.5e6))
        pa = float(self.solverConfig.get("ambient_pressure", 1.0e4))

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

        n = int(self.solverConfig.get("n_samples", 60))
        t0_cfg = float(self.solverConfig.get("stagnation_temperature", 1000.0))
        mach_profile = [1.0 + (m_out - 1.0) * i / max(n - 1, 1) for i in range(n)]
        temp_profile = [t0_cfg / (1.0 + 0.5 * (gamma - 1.0) * m * m) for m in mach_profile]
        pressure_profile = [p0 / (1.0 + 0.5 * (gamma - 1.0) * m * m) ** (gamma / (gamma - 1.0)) for m in mach_profile]

        self._lastResult = EvaluationResult(
            machProfile=mach_profile,
            pressureLoss=pressure_loss,
            thrust=thrust,
            geometryId=gid,
            temperatureProfile=temp_profile,
            pressureProfile=pressure_profile,
            metadata={
                "backend": "openfoam_rans",
                "converged": converged,
                "dimension": dimension,
                "case_dir": str(case_dir),
                "outlet_pressure": p_out,
                "outlet_velocity": u_out,
                "outlet_temperature": t_out,
                "outlet_mach": m_out,
                "mass_flow": mdot,
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
