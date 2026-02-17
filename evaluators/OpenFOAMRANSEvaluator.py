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
    - 3D channel (finite depth, front/back = symmetryPlane)
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

application     rhoSimpleFoam;
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
    default cellLimited Gauss linear 1;
}

divSchemes
{
    default none;
    div(phi,U)      Gauss upwind;
    div(phi,(p|rho)) Gauss upwind;
    div(phi,p)      Gauss upwind;
    div(phi,rho)    Gauss upwind;
    div(phid,p)     Gauss upwind;
    div(phi,e)      Gauss upwind;
    div(phi,h)      Gauss upwind;
    div(phi,K)      Gauss upwind;
    div(phi,epsilon)  Gauss upwind;
    div(phi,k)      Gauss upwind;
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
        solver GAMG;
        tolerance 1e-7;
        relTol 0.01;
        smoother DICGaussSeidel;
    }

    "(U|e|k|epsilon)"
    {
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-8;
        relTol 0.05;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        p 1e-4;
        U 1e-5;
        "(k|epsilon|e)" 1e-5;
    }
}

relaxationFactors
{
    fields
    {
        p 0.4;
    }
    equations
    {
        U 0.5;
        e 0.7;
        k 0.7;
        epsilon 0.7;
    }
}
"""
        (case_dir / "system" / "fvSolution").write_text(txt, encoding="utf-8")

    def _write_thermo(self, case_dir: Path) -> None:
        txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object thermophysicalProperties;
}

thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
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
        Hf          0;
    }
    transport
    {
        As          1.4792e-06;
        Ts          116;
    }
}
"""
        (case_dir / "constant" / "thermophysicalProperties").write_text(txt, encoding="utf-8")

    def _write_turbulence(self, case_dir: Path) -> None:
        txt = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object turbulenceProperties;
}

simulationType RAS;

RAS
{
    RASModel kEpsilon;
    turbulence on;
    printCoeffs on;
}
"""
        (case_dir / "constant" / "turbulenceProperties").write_text(txt, encoding="utf-8")

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
            front_type = "symmetryPlane"
            back_type = "symmetryPlane"

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
        u_in = float(self.solverConfig.get("inlet_velocity", 30.0))
        k_in = float(self.solverConfig.get("k_inlet", 5.0))
        epsilon_in = float(self.solverConfig.get("epsilon_inlet", 50.0))
        p_init = float(self.solverConfig.get("p_initial", 0.9 * p0))
        u_init = float(self.solverConfig.get("u_initial", max(1.0, 0.05 * u_in)))
        t_init = float(self.solverConfig.get("t_initial", t0))

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
    inlet {{ type totalPressure; p0 uniform {p0}; gamma 1.4; value uniform {pa}; }}
    outlet {{ type totalPressure; p0 uniform {pa}; gamma 1.4; value uniform {pa}; }}
    upperWall {{ type zeroGradient; }}
    lowerWall {{ type zeroGradient; }}
    front {{ type empty; }}
    back {{ type empty; }}
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
internalField uniform ({u_init} 0 0);
boundaryField
{{
    inlet {{ type pressureInletOutletVelocity; value uniform ({u_in} 0 0); }}
    outlet {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
    upperWall {{ type noSlip; }}
    lowerWall {{ type noSlip; }}
    front {{ type empty; }}
    back {{ type empty; }}
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
internalField uniform {t_init};
boundaryField
{{
    inlet {{ type totalTemperature; T0 uniform {t0}; gamma 1.4; value uniform {t0}; }}
    outlet {{ type zeroGradient; }}
    upperWall {{ type zeroGradient; }}
    lowerWall {{ type zeroGradient; }}
    front {{ type empty; }}
    back {{ type empty; }}
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
internalField uniform {k_in};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {k_in}; }}
    outlet {{ type inletOutlet; inletValue uniform {k_in}; value uniform {k_in}; }}
    upperWall {{ type kqRWallFunction; value uniform 1e-10; }}
    lowerWall {{ type kqRWallFunction; value uniform 1e-10; }}
    front {{ type empty; }}
    back {{ type empty; }}
}}
"""

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
    outlet {{ type inletOutlet; inletValue uniform {epsilon_in}; value uniform {epsilon_in}; }}
    upperWall {{ type epsilonWallFunction; value uniform 1e-10; }}
    lowerWall {{ type epsilonWallFunction; value uniform 1e-10; }}
    front {{ type empty; }}
    back {{ type empty; }}
}}
"""

        nut_txt = """FoamFile
{
    version 2.0;
    format ascii;
    class volScalarField;
    object nut;
}

dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    upperWall { type nutkWallFunction; value uniform 0; }
    lowerWall { type nutkWallFunction; value uniform 0; }
    front { type empty; }
    back { type empty; }
}
"""
        alphat_txt = """FoamFile
{
    version 2.0;
    format ascii;
    class volScalarField;
    object alphat;
}

dimensions [1 -1 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    upperWall { type compressible::alphatWallFunction; value uniform 0; }
    lowerWall { type compressible::alphatWallFunction; value uniform 0; }
    front { type empty; }
    back { type empty; }
}
"""

        mode = str(self.solverConfig.get("dimension", "2d_planar"))
        if mode != "2d_planar":
            for field_txt in (p_txt, u_txt, t_txt, k_txt, eps_txt, nut_txt):
                field_txt = field_txt.replace("type empty;", "type symmetryPlane;")
            p_txt = p_txt.replace("type empty;", "type symmetryPlane;")
            u_txt = u_txt.replace("type empty;", "type symmetryPlane;")
            t_txt = t_txt.replace("type empty;", "type symmetryPlane;")
            k_txt = k_txt.replace("type empty;", "type symmetryPlane;")
            eps_txt = eps_txt.replace("type empty;", "type symmetryPlane;")
            nut_txt = nut_txt.replace("type empty;", "type symmetryPlane;")
            alphat_txt = alphat_txt.replace("type empty;", "type symmetryPlane;")

        (case_dir / "0" / "p").write_text(p_txt, encoding="utf-8")
        (case_dir / "0" / "U").write_text(u_txt, encoding="utf-8")
        (case_dir / "0" / "T").write_text(t_txt, encoding="utf-8")
        (case_dir / "0" / "k").write_text(k_txt, encoding="utf-8")
        (case_dir / "0" / "epsilon").write_text(eps_txt, encoding="utf-8")
        (case_dir / "0" / "nut").write_text(nut_txt, encoding="utf-8")
        (case_dir / "0" / "alphat").write_text(alphat_txt, encoding="utf-8")

    def _parse_last_float(self, text: str) -> float:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not nums:
            raise RuntimeError(f"Unable to parse numeric value from:\n{text}")
        return float(nums[-1])

    def _run_post(self, case_dir: Path, func: str) -> float:
        out = self._run_cmd(f"postProcess -latestTime -func \"{func}\"", case_dir)
        return self._parse_last_float(out)

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
            self._run_cmd("blockMesh > log.blockMesh", case_dir)
            self._run_cmd("checkMesh > log.checkMesh", case_dir)
            self._run_cmd("rhoSimpleFoam > log.rhoSimpleFoam", case_dir)
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

        p_out = self._run_post(case_dir, "patchAverage(name=outlet,p)")
        u_out = self._run_post(case_dir, "patchAverage(name=outlet,mag(U))")
        t_out = self._run_post(case_dir, "patchAverage(name=outlet,T)")
        mdot = abs(self._run_post(case_dir, "patchIntegrate(name=outlet,phi)"))

        gamma = float(self.solverConfig.get("gamma", 1.4))
        r = float(self.solverConfig.get("gas_constant", 287.0))
        p0 = float(self.solverConfig.get("stagnation_pressure", 1.5e6))
        pa = float(self.solverConfig.get("ambient_pressure", 1.0e4))

        a = math.sqrt(max(gamma * r * max(t_out, 1e-9), 1e-9))
        m_out = u_out / max(a, 1e-9)
        p0_out = p_out * (1.0 + 0.5 * (gamma - 1.0) * m_out * m_out) ** (gamma / (gamma - 1.0))
        pressure_loss = max(0.0, min(0.999, 1.0 - p0_out / p0))

        depth = float(self.solverConfig.get("depth", 0.02))
        a_exit = 2.0 * self.geometry.exit_radius * depth
        thrust = mdot * u_out + (p_out - pa) * a_exit

        n = int(self.solverConfig.get("n_samples", 60))
        mach_profile = [1.0 + (m_out - 1.0) * i / max(n - 1, 1) for i in range(n)]
        temp_profile = [float(self.solverConfig.get("stagnation_temperature", 1000.0)) / (1.0 + 0.5 * (gamma - 1.0) * m * m) for m in mach_profile]
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
                "dimension": str(self.solverConfig.get("dimension", "2d_planar")),
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
        # For RANS backend, field plotting should be done with ParaView/OpenFOAM tools.
        return None
