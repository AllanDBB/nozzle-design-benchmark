from __future__ import annotations

from pathlib import Path

import pytest

from evaluators import EvaluationResult
from evaluators.OpenFOAMRANSEvaluator import OpenFOAMRANSEvaluator
from geometry import NozzleGeometry, MinimumLengthNozzle
from optimization import Optimizer


def _geom() -> NozzleGeometry:
    mln = MinimumLengthNozzle(
        mach_exit=2.3,
        throat_radius=0.02,
        exit_radius=0.06,
        length=0.22,
        gamma=1.4,
        n_points=41,
    )
    geom = mln.build()
    geom.metadata["id"] = "unit_geom"
    return geom


def _eval(tmp_path: Path, **cfg) -> OpenFOAMRANSEvaluator:
    base = {
        "backend": "openfoam",
        "campaign": "design_supersonic",
        "stagnation_pressure": 300000.0,
        "ambient_pressure": 24000.0,
        "stagnation_temperature": 900.0,
        "gamma": 1.4,
        "gas_constant": 287.0,
        "min_writes": 8,
        "convergence_window": 5,
        "convergence_tol": {
            "p_out_rel_std": 0.02,
            "mdot_rel_std": 0.02,
            "ux_out_rel_std": 0.03,
        },
    }
    base.update(cfg)
    return OpenFOAMRANSEvaluator(geometry=_geom(), solverConfig=base, resultPath=str(tmp_path))


def test_parse_surface_field_value_series(tmp_path: Path) -> None:
    ev = _eval(tmp_path)
    p = tmp_path / "surfaceFieldValue.dat"
    p.write_text(
        "# header\n"
        "0  9.0e4  7.5e2\n"
        "0.0001   (1.0e5) (7.0e2)\n",
        encoding="utf-8",
    )
    rows = ev._parse_postprocessing_series(p)
    assert len(rows) == 2
    assert rows[0][0] == pytest.approx(0.0)
    assert rows[1][1] == pytest.approx(100000.0)


def test_convergence_gate_rejects_unsteady(tmp_path: Path) -> None:
    ev = _eval(tmp_path, min_writes=8, convergence_window=5)
    outlet = {
        "time": [0.0, 1, 2, 3, 4, 5, 6, 7],
        "p": [100000, 100500, 99500, 101000, 99000, 102000, 98000, 103000],
        "T": [800.0] * 8,
        "Ux": [300, 360, 280, 370, 260, 380, 250, 390],
        "Uy": [0.0] * 8,
        "Uz": [0.0] * 8,
        "U": [300.0] * 8,
        "mdot": [0.35, 0.40, 0.32, 0.41, 0.31, 0.43, 0.30, 0.45],
    }
    conv = ev._compute_series_convergence(outlet)
    assert conv["converged_series"] is False
    assert conv["reason"] in {"window_std_exceeded", "not_enough_writes"}


def test_shock_detector_known_jump(tmp_path: Path) -> None:
    ev = _eval(tmp_path)
    x = [0.0, 0.05, 0.10, 0.15, 0.20]
    m = [1.8, 1.7, 1.4, 0.2, 0.2]
    p = [20000, 22000, 24000, 40000, 42000]
    shock = ev._detect_shock_from_profile(x, m, p)
    assert shock["present"] is True
    assert shock["x"] is not None
    assert shock["strength"] is not None
    assert shock["strength"] > 1.15


def test_openfoam_result_uses_sampled_profiles(tmp_path: Path) -> None:
    class DummyEvaluator(OpenFOAMRANSEvaluator):
        def _build_case(self, case_dir: Path) -> None:
            for d in [case_dir / "system", case_dir / "constant", case_dir / "0"]:
                d.mkdir(parents=True, exist_ok=True)

        def _run_cmd(self, cmd: str, cwd: Path) -> str:
            return "ok"

        def _run_shockfluid(self, case_dir: Path) -> None:
            (case_dir / "log.shockFluid").write_text("End\n", encoding="utf-8")

        def _check_convergence(self, case_dir: Path) -> bool:
            return True

        def _extract_outlet_series(self, case_dir: Path):
            return {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "p": [50000.0] * 8,
                "T": [500.0] * 8,
                "Ux": [500.0] * 8,
                "Uy": [0.0] * 8,
                "Uz": [0.0] * 8,
                "U": [500.0] * 8,
                "mdot": [0.30] * 8,
            }

        def _extract_centerline_series(self, case_dir: Path):
            x = [0.0, 0.05, 0.10, 0.15]
            final = {
                "time": 7.0,
                "p": [120000.0, 90000.0, 60000.0, 50000.0],
                "T": [820.0, 760.0, 640.0, 500.0],
                "rho": [1.0, 1.0, 1.0, 1.0],
                "Ux": [50.0, 250.0, 450.0, 500.0],
                "Uy": [0.0, 0.0, 0.0, 0.0],
                "Uz": [0.0, 0.0, 0.0, 0.0],
                "U": [50.0, 250.0, 450.0, 500.0],
                "Mach": [0.10, 0.55, 1.20, 1.35],
            }
            series = [dict(final) for _ in range(8)]
            return {"x": x, "series": series}

    ev = DummyEvaluator(
        geometry=_geom(),
        solverConfig={
            "backend": "openfoam",
            "campaign": "design_supersonic",
            "stagnation_pressure": 300000.0,
            "ambient_pressure": 24000.0,
            "stagnation_temperature": 900.0,
            "gamma": 1.4,
            "gas_constant": 287.0,
            "require_converged_series": True,
            "min_writes": 8,
            "convergence_window": 5,
            "convergence_tol": {"p_out_rel_std": 0.02, "mdot_rel_std": 0.02, "ux_out_rel_std": 0.03},
        },
        resultPath=str(tmp_path),
    )

    r = ev.extractResults()
    assert r.xProfile == [0.0, 0.05, 0.10, 0.15]
    assert r.machProfile == [0.10, 0.55, 1.20, 1.35]
    assert r.pressureProfile == [120000.0, 90000.0, 60000.0, 50000.0]


def test_campaign_config_guardrails(tmp_path: Path) -> None:
    ev = _eval(
        tmp_path,
        campaign="design_supersonic",
        stagnation_pressure=300000.0,
        ambient_pressure=90000.0,
        pressure_ratio=0.08,
    )
    with pytest.raises(ValueError):
        ev._validate_campaign()


def test_windows_ascii_logging() -> None:
    data = Path("main_pipeline.py").read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    data.decode("ascii")


def test_wall_pressure_rms_metrics(tmp_path: Path) -> None:
    ev = _eval(tmp_path)
    wall = {
        "time": [0, 1, 2, 3, 4],
        "upper": [95000.0, 96000.0, 94000.0, 97000.0, 93000.0],
        "lower": [94500.0, 95500.0, 93500.0, 96500.0, 92500.0],
        "mean": [94750.0, 95750.0, 93750.0, 96750.0, 92750.0],
        "delta": [500.0, 500.0, 500.0, 500.0, 500.0],
    }
    m = ev._compute_wall_pressure_metrics(wall)
    assert m["wall_p_rms"] > 0.0
    assert m["wall_p_rel_rms"] > 0.0
    assert m["wall_p_delta_rms"] == pytest.approx(0.0)


def test_overexpanded_score_penalizes_outlet_oscillation() -> None:
    opt = Optimizer(
        searchSpace={
            "bounds": {"x": (0.0, 1.0)},
            "campaign": "overexpanded_sea_level",
            "objective_terms": {
                "overexpanded_instability_penalty": 0.05,
                "overexpanded_outlet_osc_penalty": 0.05,
                "overexpanded_wall_rms_penalty": 0.03,
            },
        },
        objectiveFunc=lambda _: EvaluationResult(machProfile=[], pressureLoss=0.0, thrust=0.0, geometryId="dummy"),
        algorithm="random",
    )
    stable = EvaluationResult(
        machProfile=[],
        pressureLoss=0.20,
        thrust=120.0,
        geometryId="stable",
        convergence={"metrics": {"p_out_rel_std": 0.005, "mdot_rel_std": 0.005, "ux_out_rel_std": 0.010, "wall_p_rel_rms": 0.01}},
        shock={"x_std": 0.005, "present": True},
    )
    unsteady = EvaluationResult(
        machProfile=[],
        pressureLoss=0.20,
        thrust=120.0,
        geometryId="unsteady",
        convergence={"metrics": {"p_out_rel_std": 0.080, "mdot_rel_std": 0.070, "ux_out_rel_std": 0.100, "wall_p_rel_rms": 0.20}},
        shock={"x_std": 0.08, "present": True},
    )

    ref = [stable, unsteady]
    s_stable = opt._objective_score(stable, reference=ref)
    s_unsteady = opt._objective_score(unsteady, reference=ref)
    assert s_stable > s_unsteady


def test_overexpanded_series_gate_accepts_controlled_unsteady(tmp_path: Path) -> None:
    ev = _eval(
        tmp_path,
        campaign="overexpanded_sea_level",
        require_converged_series=True,
        allow_unsteady_overexpanded=True,
        overexpanded_unsteady_tol={
            "p_out_rel_std_max": 0.10,
            "mdot_rel_std_max": 0.35,
            "ux_out_rel_std_max": 0.45,
        },
    )
    conv = {
        "converged_series": False,
        "n_writes": 9,
        "min_writes": 8,
        "reason": "window_std_exceeded",
        "metrics": {
            "p_out_rel_std": 0.0,
            "mdot_rel_std": 0.22,
            "ux_out_rel_std": 0.30,
        },
    }
    usable, reason = ev._series_gate(conv)
    assert usable is True
    assert reason == "accepted_unsteady_overexpanded"


def test_delta_t_abort_state(tmp_path: Path) -> None:
    ev = _eval(
        tmp_path,
        enable_delta_t_abort=True,
        delta_t_abort_threshold=1e-10,
        delta_t_abort_streak=3,
    )
    streak, abort = ev._update_delta_t_abort_state(1e-11, 0)
    assert streak == 1 and abort is False
    streak, abort = ev._update_delta_t_abort_state(1e-12, streak)
    assert streak == 2 and abort is False
    streak, abort = ev._update_delta_t_abort_state(1e-13, streak)
    assert streak == 3 and abort is True
