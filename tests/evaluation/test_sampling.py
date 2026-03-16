"""Tests for offset-corrected IV sampling containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.sampling as sampling
from superconductivity.evaluation.ivdata import IVTrace, IVTraces
from superconductivity.evaluation.offset import OffsetTrace, OffsetTraces
from superconductivity.utilities.constants import G_0_muS
from superconductivity.utilities.functions import bin_y_over_x
from superconductivity.utilities.functions import upsample as upsample_xy


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    v_shift_mV: float,
    i_shift_nA: float,
) -> IVTrace:
    t_s = np.linspace(0.0, 10.0, 401, dtype=np.float64)
    v_true_mV = np.linspace(-2.0, 2.0, t_s.size, dtype=np.float64)
    i_true_nA = v_true_mV + 0.2 * v_true_mV**3
    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "V_mV": v_true_mV + v_shift_mV,
        "I_nA": i_true_nA + i_shift_nA,
        "t_s": t_s,
    }


def _make_offset_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    v_shift_mV: float,
    i_shift_nA: float,
) -> OffsetTrace:
    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "dGerr_G0": np.asarray([0.0]),
        "dRerr_R0": np.asarray([0.0]),
        "Voff_mV": v_shift_mV,
        "Ioff_nA": i_shift_nA,
    }


def _make_spec() -> sampling.SamplingSpec:
    return sampling.SamplingSpec(
        nu_Hz=20.0,
        upsample=4,
        Vbin_mV=np.linspace(-2.0, 2.0, 81),
        Ibin_nA=np.linspace(-4.0, 4.0, 81),
    )


def _manual_sampling(
    trace: IVTrace,
    offset: OffsetTrace,
    spec: sampling.SamplingSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_raw = np.asarray(trace["t_s"], dtype=np.float64)
    v_raw_mV = np.asarray(trace["V_mV"], dtype=np.float64)
    i_raw_nA = np.asarray(trace["I_nA"], dtype=np.float64)

    t_bins_s = np.arange(
        np.min(t_raw),
        np.max(t_raw) + 0.5 * spec.dt_s,
        spec.dt_s,
        dtype=np.float64,
    )
    if t_bins_s.size < 2:
        t_bins_s = np.linspace(np.min(t_raw), np.max(t_raw), 2, dtype=np.float64)

    v_down_mV = bin_y_over_x(t_raw, v_raw_mV, t_bins_s)
    i_down_nA = bin_y_over_x(t_raw, i_raw_nA, t_bins_s)
    finite = np.isfinite(v_down_mV) & np.isfinite(i_down_nA)
    v_down_mV = v_down_mV[finite]
    i_down_nA = i_down_nA[finite]

    i_corr_nA = i_down_nA - float(offset["Ioff_nA"])
    v_corr_mV = v_down_mV - float(offset["Voff_mV"])
    i_over_nA, v_over_mV = upsample_xy(
        i_corr_nA,
        v_corr_mV,
        factor=spec.upsample,
        method="linear",
    )

    v_sampled_mV = bin_y_over_x(i_over_nA, v_over_mV, spec.Ibin_nA)
    i_sampled_nA = bin_y_over_x(v_over_mV, i_over_nA, spec.Vbin_mV)
    dG_G0 = np.gradient(i_sampled_nA, spec.Vbin_mV) / G_0_muS
    dR_R0 = np.gradient(v_sampled_mV, spec.Ibin_nA) * G_0_muS
    return i_sampled_nA, v_sampled_mV, dG_G0, dR_R0


def test_get_sampling_matches_manual_notebook_flow() -> None:
    """Single-trace sampling should match the manual notebook loop."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )
    offset = _make_offset_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )
    spec = _make_spec()

    out = sampling.get_sampling(trace=trace, offset=offset, spec=spec)
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _manual_sampling(
        trace=trace,
        offset=offset,
        spec=spec,
    )

    assert out["specific_key"] == "a"
    assert out["index"] == 0
    assert out["yvalue"] == pytest.approx(1.0)
    assert out["Voff_mV"] == pytest.approx(0.4)
    assert out["Ioff_nA"] == pytest.approx(0.3)
    assert np.allclose(out["Vbin_mV"], spec.Vbin_mV)
    assert np.allclose(out["Ibin_nA"], spec.Ibin_nA)
    assert np.allclose(out["I_nA"], i_exp_nA, equal_nan=True)
    assert np.allclose(out["V_mV"], v_exp_mV, equal_nan=True)
    assert np.allclose(out["dG_G0"], dG_exp_G0, equal_nan=True)
    assert np.allclose(out["dR_R0"], dR_exp_R0, equal_nan=True)


def test_get_samplings_returns_collection_with_lookup_methods() -> None:
    """Sampling collection should mirror the IV/offset container API."""
    traces = IVTraces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )
    offsets = OffsetTraces(
        spec=object(),  # type: ignore[arg-type]
        traces=[
            _make_offset_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_offset_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )
    spec = _make_spec()

    out = sampling.get_samplings(
        traces=traces,
        offsets=offsets,
        spec=spec,
        show_progress=False,
    )

    assert out.keys == ["a", "b"]
    assert np.allclose(out.yvalues, np.asarray([1.0, 5.0]))
    assert np.allclose(out.Voff_mV, np.asarray([0.4, 0.2]))
    assert np.allclose(out.Ioff_nA, np.asarray([0.3, 0.1]))
    assert np.allclose(out.Vbin_mV, spec.Vbin_mV)
    assert np.allclose(out.Ibin_nA, spec.Ibin_nA)
    assert out.I_nA.shape == (2, spec.Vbin_mV.size)
    assert out.V_mV.shape == (2, spec.Ibin_nA.size)
    assert out.dG_G0.shape == (2, spec.Vbin_mV.size)
    assert out.dR_R0.shape == (2, spec.Ibin_nA.size)
    assert out.by_key("a")["Voff_mV"] == pytest.approx(0.4)
    assert out.by_value(5.0)["specific_key"] == "b"


def test_get_samplings_rejects_length_mismatch() -> None:
    """Trace and offset collections must stay aligned."""
    traces = IVTraces(
        traces=[_make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)],
    )
    offsets = OffsetTraces(
        spec=object(),  # type: ignore[arg-type]
        traces=[
            _make_offset_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_offset_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )

    with pytest.raises(ValueError, match="same length"):
        sampling.get_samplings(
            traces=traces,
            offsets=offsets,
            spec=_make_spec(),
            show_progress=False,
        )


def test_get_sampling_rejects_mismatched_trace_and_offset() -> None:
    """Single-trace sampling should fail on mismatched metadata."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )
    offset = _make_offset_trace(
        specific_key="b",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )

    with pytest.raises(ValueError, match="specific_key"):
        sampling.get_sampling(
            trace=trace,
            offset=offset,
            spec=_make_spec(),
        )
