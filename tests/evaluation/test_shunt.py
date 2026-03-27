"""Tests for shunt analysis built on IV trace containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.shunt as shunt
from superconductivity.evaluation.ivdata import IVTrace, IVTraces


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    gshunt_uS: float,
    nonlinear_coeff: float = 0.05,
) -> IVTrace:
    v_mV = np.linspace(-1.0, 1.0, 401, dtype=np.float64)
    t_s = np.linspace(0.0, 10.0, v_mV.size, dtype=np.float64)
    i_nA = gshunt_uS * v_mV + nonlinear_coeff * v_mV**3
    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "V_mV": v_mV,
        "I_nA": i_nA,
        "t_s": t_s,
    }


def _make_spec() -> shunt.ShuntSpec:
    return shunt.ShuntSpec(
        delta_mV=1.0,
        subgap_range=(0.0, 0.1),
        min_points=3,
    )


def test_get_shunt_returns_shunttrace_for_single_trace() -> None:
    """Single-trace shunt fit should recover the known conductance."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        gshunt_uS=0.2,
    )

    out = shunt.get_shunt(trace=trace, spec=_make_spec())

    assert out["specific_key"] == "a"
    assert out["index"] == 0
    assert out["yvalue"] == pytest.approx(1.0)
    assert out["Gshunt_uS"] == pytest.approx(0.2, abs=5e-4)
    assert out["Rshunt_MOhm"] == pytest.approx(5.0, abs=0.02)
    assert out["points"] > 0
    assert out["V_fit_mV"].ndim == 1
    assert out["I_fit_nA"].ndim == 1


def test_get_shunt_returns_infinite_resistance_for_zero_conductance() -> None:
    """Zero fitted conductance should map to an infinite shunt resistance."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        gshunt_uS=0.0,
        nonlinear_coeff=0.0,
    )

    out = shunt.get_shunt(trace=trace, spec=_make_spec())

    assert out["Gshunt_uS"] == pytest.approx(0.0, abs=5e-4)
    assert np.isinf(out["Rshunt_MOhm"])


def test_get_shunt_traces_returns_collection_with_lookup_methods() -> None:
    """Collection fits should mirror the IV/offset container API."""
    traces = IVTraces(
        traces=[
            _make_iv_trace("a", 0, 1.0, gshunt_uS=0.2),
            _make_iv_trace("b", 1, 5.0, gshunt_uS=0.5),
        ],
    )

    out = shunt.get_shunt_traces(
        traces=traces,
        spec=_make_spec(),
        show_progress=False,
    )

    assert out.spec.delta_mV == pytest.approx(1.0)
    assert out.keys == ["a", "b"]
    assert np.allclose(out.yvalues, np.asarray([1.0, 5.0]))
    assert np.allclose(out.Gshunt_uS, np.asarray([0.2, 0.5]), atol=5e-4)
    assert np.allclose(out.Rshunt_MOhm, np.asarray([5.0, 2.0]), atol=0.02)
    assert out.by_key("a")["Gshunt_uS"] == pytest.approx(0.2, abs=5e-4)
    assert out.by_value(5.0)["specific_key"] == "b"


def test_get_shunt_traces_parallel_matches_single_worker() -> None:
    """Trace-level parallelism should preserve numerical results."""
    traces = IVTraces(
        traces=[
            _make_iv_trace("a", 0, 1.0, gshunt_uS=0.2),
            _make_iv_trace("b", 1, 5.0, gshunt_uS=0.5),
        ],
    )
    spec_obj = _make_spec()

    serial = shunt.get_shunt_traces(
        traces=traces,
        spec=spec_obj,
        show_progress=False,
        workers=1,
    )
    parallel = shunt.get_shunt_traces(
        traces=traces,
        spec=spec_obj,
        show_progress=False,
        workers=2,
    )

    assert np.allclose(parallel.Gshunt_uS, serial.Gshunt_uS)
    assert np.allclose(parallel.Rshunt_MOhm, serial.Rshunt_MOhm)
    assert np.array_equal(parallel.points, serial.points)


def test_shunttraces_by_value_rejects_ambiguous_matches() -> None:
    """Duplicate yvalues should require the plural lookup method."""
    traces = shunt.ShuntTraces(
        spec=_make_spec(),
        traces=[
            {
                "specific_key": "a",
                "index": 0,
                "yvalue": 1.0,
                "Gshunt_uS": 0.1,
                "Rshunt_MOhm": 10.0,
                "Iintercept_nA": 0.0,
                "rmse_nA": 0.0,
                "points": 5,
                "V_fit_mV": np.asarray([-0.1, 0.0, 0.1]),
                "I_fit_nA": np.asarray([-0.01, 0.0, 0.01]),
            },
            {
                "specific_key": "b",
                "index": 1,
                "yvalue": 1.0,
                "Gshunt_uS": 0.2,
                "Rshunt_MOhm": 5.0,
                "Iintercept_nA": 0.0,
                "rmse_nA": 0.0,
                "points": 5,
                "V_fit_mV": np.asarray([-0.1, 0.0, 0.1]),
                "I_fit_nA": np.asarray([-0.02, 0.0, 0.02]),
            },
        ],
    )

    with pytest.raises(ValueError, match="matches multiple traces"):
        traces.by_value(1.0)

    matches = traces.all_by_value(1.0)
    assert [trace["specific_key"] for trace in matches] == ["a", "b"]


def test_get_shunt_rejects_empty_fit_window() -> None:
    """A trace with no selected tuples should fail clearly."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        gshunt_uS=0.2,
    )
    spec_obj = shunt.ShuntSpec(
        delta_mV=3.0,
        subgap_range=(0.5, 0.6),
        min_points=3,
    )

    with pytest.raises(ValueError, match="has no points in the \\|V\\| window"):
        shunt.get_shunt(trace=trace, spec=spec_obj)
