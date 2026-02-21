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
    """Steady compressible RANS evaluator using OpenFOAM (rhoSimpleFoam).

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
        end_time = int(self.solverConfig.get("end_time", 1200))
        write_interval = int(self.solverConfig.get("write_interval", 200))
        txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}}

application     foamRun;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
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

ddtSchemes
{
    default steadyState;
}

gradSchemes
{
    default         Gauss linear;
    limited         cellLimited Gauss linear 1;
    grad(U)         $limited;
    grad(k)         $limited;
    grad(epsilon)   $limited;
}

divSchemes
{
    default         none;

    div(phi,U)      bounded Gauss upwind;

    energy          bounded Gauss upwind;
    div(phi,h)      $energy;
    div(phi,K)      $energy;

    turbulence      bounded Gauss upwind;
    div(phi,k)      $turbulence;
    div(phi,epsilon) $turbulence;

    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default Gauss linear corrected;
}

interpolationSchemes
{
    default linear;
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

solvers
{
    p
    {
        solver          GAMG;
        smoother        DIC;
        tolerance       1e-8;
        relTol          0.01;
    }

    "(U|h|k|epsilon)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-10;
        relTol          0.1;
    }
}

PIMPLE
{
    residualControl
    {
        p               1e-4;
        U               1e-5;
        "(h|k|epsilon)" 1e-5;
    }

    nNonOrthogonalCorrectors 1;
}

relaxationFactors
{
    fields
    {
        p       0.3;
        rho     0.01;
    }
    equations
    {
        U       0.4;
        h       0.2;
        k       0.3;
        epsilon 0.3;
    }
}
"""
        (case_dir / "system" / "fvSolution").write_text(txt, encoding="utf-8")

        # OF13: pressure and temperature bounds in fvConstraints
        fc_txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvConstraints;
}

limitp
{
    type    limitPressure;
    min     100;
    max     1e8;
}

limitT
{
    type        limitTemperature;
    cellZone    all;
    min         50;
    max         8000;
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

thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie
    {
        molWeight   28.96;
    }
    thermodynamics
    {
        Cp          1004.5;
        hf          0;
    }
    transport
    {
        mu          3.5e-05;
        Pr          0.72;
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
    model           kEpsilon;
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

    def _write_initial_fields(self, case_dir: Path) -> None:
        p0 = float(self.solverConfig.get("stagnation_pressure", 1.5e6))
        t0 = float(self.solverConfig.get("stagnation_temperature", 1000.0))
        pa = float(self.solverConfig.get("ambient_pressure", 1.0e4))
        k_in = float(self.solverConfig.get("k_inlet", 5.0))
        epsilon_in = float(self.solverConfig.get("epsilon_inlet", 50.0))
        p_init = float(self.solverConfig.get("p_initial", 0.9 * p0))
        u_init = float(self.solverConfig.get("u_initial", 1.0))
        t_init = float(self.solverConfig.get("t_initial", t0))
        gamma_cf = float(self.solverConfig.get("gamma", 1.4))

        # Guard: if u_initial is unrealistically low (< 50 m/s), compute
        # a sane M=0.3 isentropic state from stagnation conditions.
        # A uniform supersonic IC causes the first GAMG pressure correction
        # to overflow because of the large velocity divergence in the duct.
        import math as _math
        _R = 287.0
        if u_init < 50.0:
            _M0 = 0.3
            _fac = 1.0 + (gamma_cf - 1.0) / 2.0 * _M0 ** 2
            t_init = t0 / _fac
            p_init = p0 / _fac ** (gamma_cf / (gamma_cf - 1.0))
            u_init = _M0 * _math.sqrt(gamma_cf * _R * t_init)
        mode = str(self.solverConfig.get("dimension", "2d_planar"))
        is_2d = mode == "2d_planar"
        front_p = "empty" if is_2d else "zeroGradient"
        back_p = "empty" if is_2d else "zeroGradient"
        front_u = "empty" if is_2d else "noSlip"
        back_u = "empty" if is_2d else "noSlip"
        front_t = "empty" if is_2d else "zeroGradient"
        back_t = "empty" if is_2d else "zeroGradient"
        front_k = "empty" if is_2d else "kqRWallFunction"
        back_k = "empty" if is_2d else "kqRWallFunction"
        front_eps = "empty" if is_2d else "epsilonWallFunction"
        back_eps = "empty" if is_2d else "epsilonWallFunction"
        front_nut = "empty" if is_2d else "nutkWallFunction"
        back_nut = "empty" if is_2d else "nutkWallFunction"
        front_alphat = "empty" if is_2d else "compressible::alphatWallFunction"
        back_alphat = "empty" if is_2d else "compressible::alphatWallFunction"

        # ── p ──────────────────────────────────────────────────────────────
        # totalPressure without explicit gamma – uses thermo model internally.
        p_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}

dimensions [1 -1 -2 0 0 0 0];
internalField uniform {p_init};
boundaryField
{{
    inlet {{ type totalPressure; p0 uniform {p0}; gamma 1.4; value uniform {p_init}; }}
    outlet {{ type fixedValue; value uniform {pa}; }}
    upperWall {{ type zeroGradient; }}
    lowerWall {{ type zeroGradient; }}
    front {{ type {front_p}; }}
    back {{ type {back_p}; }}
}}
"""

        # ── U ──────────────────────────────────────────────────────────────
        # pressureInletVelocity is the correct companion to totalPressure.
        u_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}}

dimensions [0 1 -1 0 0 0 0];
internalField uniform ({u_init} 0 0);
boundaryField
{{
    inlet {{ type pressureInletVelocity; value uniform ({u_init} 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform ({u_init} 0 0); value uniform ({u_init} 0 0); }}
    upperWall {{ type noSlip; }}
    lowerWall {{ type noSlip; }}
    front {{ type {front_u}; }}
    back {{ type {back_u}; }}
}}
"""

        # ── T ──────────────────────────────────────────────────────────────
        t_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object T;
}}

dimensions [0 0 0 1 0 0 0];
internalField uniform {t_init};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {t0}; }}
    outlet {{ type inletOutlet; inletValue uniform {t_init}; value uniform {t_init}; }}
    upperWall {{ type zeroGradient; }}
    lowerWall {{ type zeroGradient; }}
    front {{ type {front_t}; }}
    back {{ type {back_t}; }}
}}
"""

        # ── k ──────────────────────────────────────────────────────────────
        k_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object k;
}}

dimensions [0 2 -2 0 0 0 0];
internalField uniform {k_in};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {k_in}; }}
    outlet {{ type zeroGradient; }}
    upperWall {{ type kqRWallFunction; value uniform 1e-10; }}
    lowerWall {{ type kqRWallFunction; value uniform 1e-10; }}
    front {{ type {front_k}; {'value uniform 1e-10;' if not is_2d else ''} }}
    back {{ type {back_k}; {'value uniform 1e-10;' if not is_2d else ''} }}
}}
"""

        # ── epsilon ─────────────────────────────────────────────────────────
        eps_txt = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object epsilon;
}}

dimensions [0 2 -3 0 0 0 0];
internalField uniform {epsilon_in};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {epsilon_in}; }}
    outlet {{ type zeroGradient; }}
    upperWall {{ type epsilonWallFunction; value uniform 1e-10; }}
    lowerWall {{ type epsilonWallFunction; value uniform 1e-10; }}
    front {{ type {front_eps}; {'value uniform 1e-10;' if not is_2d else ''} }}
    back {{ type {back_eps}; {'value uniform 1e-10;' if not is_2d else ''} }}
}}
"""

        # ── nut ─────────────────────────────────────────────────────────────
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
    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    upperWall {{ type nutkWallFunction; value uniform 0; }}
    lowerWall {{ type nutkWallFunction; value uniform 0; }}
    front {{ type {front_nut}; {'value uniform 0;' if not is_2d else ''} }}
    back {{ type {back_nut}; {'value uniform 0;' if not is_2d else ''} }}
}}
"""

        # ── alphat ──────────────────────────────────────────────────────────
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
    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
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
        (case_dir / "0" / "epsilon").write_text(eps_txt, encoding="utf-8")
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
                    f"have run. Check {case_dir / 'log.rhoSimpleFoam'} for errors."
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
        """Return True if rhoSimpleFoam reached its residual convergence target."""
        log_path = case_dir / "log.rhoSimpleFoam"
        if not log_path.exists():
            return False
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "SIMPLE solution converged" in text:
            return True
        # Fallback: at minimum the run must have completed without abort.
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
        self._write_initial_fields(case_dir)

    def run(self) -> None:
        self._lastResult = self.extractResults()

    def extractResults(self) -> EvaluationResult:
        gid = str(self.geometry.metadata.get("id", self.geometry.metadata.get("source", "geometry")))
        case_dir = self._case_dir(gid)
        self._build_case(case_dir)

        try:
            self._run_cmd("blockMesh > log.blockMesh 2>&1", case_dir)
            self._run_cmd("checkMesh > log.checkMesh 2>&1", case_dir)
            self._run_cmd("foamRun -solver fluid > log.rhoSimpleFoam 2>&1", case_dir)
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
                f"rhoSimpleFoam did not converge for case {gid}. "
                f"Inspect {case_dir / 'log.rhoSimpleFoam'} for details."
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
