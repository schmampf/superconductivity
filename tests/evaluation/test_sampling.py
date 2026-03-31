"""Tests for explicit IV sampling stages and containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.sampling as sampling
from superconductivity.evaluation.analysis.offset import OffsetTraces
from superconductivity.evaluation.traces import TraceMeta
from superconductivity.evaluation.traces import Trace, Traces
from superconductivity.utilities.constants import G_0_muS
from superconductivity.utilities.functions import bin_y_over_x
from superconductivity.utilities.functions import upsample as upsample_xy


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    v_shift_mV: float,
    i_shift_nA: float,
) -> Trace:
    t_s = np.linspace(0.0, 10.0, 401, dtype=np.float64)
    v_true_mV = np.linspace(-2.0, 2.0, t_s.size, dtype=np.float64)
    i_true_nA = v_true_mV + 0.2 * v_true_mV**3
    return {
        "meta": TraceMeta(
            specific_key=specific_key,
            index=index,
            yvalue=yvalue,
        ),
        "V_mV": v_true_mV + v_shift_mV,
        "I_nA": i_true_nA + i_shift_nA,
        "t_s": t_s,
    }


def _make_spec() -> sampling.SamplingSpec:
    return sampling.SamplingSpec(
        Vbins_mV=np.linspace(-2.0, 2.0, 81),
        Ibins_nA=np.linspace(-4.0, 4.0, 81),
        nu_Hz=40.0,
        upsample=4,
    )


def _make_offset(v_shift_mV: float, i_shift_nA: float) -> dict[str, object]:
    return {
        "dGerr_G0": np.asarray([0.0]),
        "dRerr_R0": np.asarray([0.0]),
        "Voff_mV": float(v_shift_mV),
        "Ioff_nA": float(i_shift_nA),
    }


def _manual_binning(
    trace: Trace,
    spec: sampling.SamplingSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i_over_nA, v_over_mV = upsample_xy(
        np.asarray(trace["I_nA"], dtype=np.float64),
        np.asarray(trace["V_mV"], dtype=np.float64),
        factor=spec.upsample,
        method="linear",
    )

    v_sampled_mV = bin_y_over_x(i_over_nA, v_over_mV, spec.Ibins_nA)
    i_sampled_nA = bin_y_over_x(v_over_mV, i_over_nA, spec.Vbins_mV)
    dG_G0 = np.gradient(i_sampled_nA, spec.Vbins_mV) / G_0_muS
    dR_R0 = np.gradient(v_sampled_mV, spec.Ibins_nA) * G_0_muS
    return i_sampled_nA, v_sampled_mV, dG_G0, dR_R0


def test_downsampling_uses_sampling_spec_nu_hz() -> None:
    """The high-level downsampling wrapper should use ``nu_Hz`` from the spec."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )
    spec = _make_spec()

    downsampled = sampling.downsampling(trace, samplingspec=spec)
    expected = sampling.downsample_trace(trace, nu_Hz=spec.nu_Hz)

    assert np.allclose(downsampled["t_s"], expected["t_s"])
    assert np.allclose(downsampled["V_mV"], expected["V_mV"])
    assert np.allclose(downsampled["I_nA"], expected["I_nA"])


def test_offset_correction_subtracts_offsets() -> None:
    """The explicit offset-correction stage should shift the trace arrays."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )
    corrected = sampling.offset_correction(
        trace,
        offsetanalysis=_make_offset(0.4, 0.3),
    )

    assert np.allclose(
        corrected["V_mV"],
        np.asarray(trace["V_mV"], dtype=np.float64) - 0.4,
    )
    assert np.allclose(
        corrected["I_nA"],
        np.asarray(trace["I_nA"], dtype=np.float64) - 0.3,
    )


def test_binning_matches_manual_flow_for_corrected_trace() -> None:
    """Binning should match the manual notebook-style sampling flow."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.0,
        i_shift_nA=0.0,
    )
    spec = _make_spec()

    out = sampling.binning(trace, samplingspec=spec)
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _manual_binning(trace, spec)

    assert np.allclose(out["I_nA"], i_exp_nA, equal_nan=True)
    assert np.allclose(out["V_mV"], v_exp_mV, equal_nan=True)
    assert np.allclose(out["dG_G0"], dG_exp_G0, equal_nan=True)
    assert np.allclose(out["dR_R0"], dR_exp_R0, equal_nan=True)


def test_sampling_matches_manual_pipeline_with_offsets() -> None:
    """Full sampling should apply offset correction before binning."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )
    spec = _make_spec()
    offset = _make_offset(0.4, 0.3)

    out = sampling.sample(
        trace,
        samplingspec=spec,
        offsetanalysis=offset,
    )
    corrected = sampling.offset_correction(trace, offsetanalysis=offset)
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _manual_binning(corrected, spec)

    assert np.allclose(out["Vbins_mV"], spec.Vbins_mV)
    assert np.allclose(out["Ibins_nA"], spec.Ibins_nA)
    assert out["meta"] == trace["meta"]
    assert np.allclose(out["I_nA"], i_exp_nA, equal_nan=True)
    assert np.allclose(out["V_mV"], v_exp_mV, equal_nan=True)
    assert np.allclose(out["dG_G0"], dG_exp_G0, equal_nan=True)
    assert np.allclose(out["dR_R0"], dR_exp_R0, equal_nan=True)


def test_sampling_returns_collection() -> None:
    """Collection sampling should return stacked sampled results."""
    traces = Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )
    offsets = OffsetTraces(
        traces=[
            _make_offset(0.4, 0.3),
            _make_offset(0.2, 0.1),
        ],
    )
    spec = _make_spec()

    out = sampling.sample(
        traces,
        samplingspec=spec,
        offsetanalysis=offsets,
        show_progress=False,
    )

    assert isinstance(out, sampling.Samples)
    assert np.allclose(out.Vbins_mV, spec.Vbins_mV)
    assert np.allclose(out.Ibins_nA, spec.Ibins_nA)
    assert out.specific_keys == ["a", "b"]
    assert np.array_equal(out.indices, np.asarray([0, 1], dtype=np.int64))
    assert np.allclose(out.yvalues, np.asarray([1.0, 5.0]))
    assert out[0]["meta"] == traces[0]["meta"]
    assert out.I_nA.shape == (2, spec.Vbins_mV.size)
    assert out.V_mV.shape == (2, spec.Ibins_nA.size)
    assert out.dG_G0.shape == (2, spec.Vbins_mV.size)
    assert out.dR_R0.shape == (2, spec.Ibins_nA.size)


def test_sampling_with_smoothing_returns_collection() -> None:
    """The full pipeline should optionally return smoothed sampled data."""
    traces = Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )
    offsets = OffsetTraces(
        traces=[
            _make_offset(0.4, 0.3),
            _make_offset(0.2, 0.1),
        ],
    )
    spec = _make_spec()
    smooth_spec = sampling.SmoothingSpec(
        median_bins=3,
        sigma_bins=1.0,
    )

    out = sampling.sample(
        traces,
        samplingspec=spec,
        smoothingspec=smooth_spec,
        offsetanalysis=offsets,
        show_progress=False,
    )

    assert isinstance(out, sampling.Samples)


def test_offset_correction_rejects_length_mismatch() -> None:
    """Collection offset correction should require aligned lengths."""
    traces = Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )
    offsets = OffsetTraces(traces=[_make_offset(0.4, 0.3)])

    with pytest.raises(ValueError, match="same length"):
        sampling.offset_correction(traces, offsetanalysis=offsets)
