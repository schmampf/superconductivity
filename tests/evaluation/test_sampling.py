"""Tests for explicit IV sampling stages and containers."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import superconductivity.api as api_module
import superconductivity.evaluation as evaluation_module
import superconductivity.evaluation.sampling as sampling
from superconductivity.evaluation.analysis.offset import OffsetTraces
from superconductivity.evaluation.traces import Trace, TraceMeta, Traces
from superconductivity.utilities.constants import G_0_muS
from superconductivity.utilities.functions import bin_y_over_x
from superconductivity.utilities.functions import upsample as upsample_xy

pipeline_mod = importlib.import_module(
    "superconductivity.evaluation.sampling.pipeline",
)


def test_sampling_exports_expose_upsampling_and_drop_smoothing_spec() -> None:
    """Public exports should expose upsampling and remove SmoothingSpec."""
    assert hasattr(sampling, "SamplingSpec")
    assert hasattr(sampling, "upsampling")
    assert not hasattr(sampling, "SmoothingSpec")

    assert hasattr(evaluation_module, "SamplingSpec")
    assert hasattr(evaluation_module, "upsampling")
    assert not hasattr(evaluation_module, "SmoothingSpec")

    assert hasattr(api_module, "SamplingSpec")
    assert hasattr(api_module, "upsampling")
    assert not hasattr(api_module, "SmoothingSpec")


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


def _make_spec(**updates: object) -> sampling.SamplingSpec:
    defaults: dict[str, object] = {
        "Vbins_mV": np.linspace(-2.0, 2.0, 81),
        "Ibins_nA": np.linspace(-4.0, 4.0, 81),
        "apply_smoothing": False,
        "nu_Hz": 40.0,
        "N_up": 4,
    }
    defaults.update(updates)
    return sampling.SamplingSpec(**defaults)


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
    v_trace_mV = np.asarray(trace["V_mV"], dtype=np.float64)
    i_trace_nA = np.asarray(trace["I_nA"], dtype=np.float64)
    v_sampled_mV = bin_y_over_x(i_trace_nA, v_trace_mV, spec.Ibins_nA)
    i_sampled_nA = bin_y_over_x(v_trace_mV, i_trace_nA, spec.Vbins_mV)
    dG_G0 = np.gradient(i_sampled_nA, spec.Vbins_mV) / G_0_muS
    dR_R0 = np.gradient(v_sampled_mV, spec.Ibins_nA) * G_0_muS
    return i_sampled_nA, v_sampled_mV, dG_G0, dR_R0


def _legacy_hidden_upsample_binning(
    trace: Trace,
    spec: sampling.SamplingSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i_over_nA, v_over_mV = upsample_xy(
        np.asarray(trace["I_nA"], dtype=np.float64),
        np.asarray(trace["V_mV"], dtype=np.float64),
        factor=spec.N_up,
        method="linear",
    )
    v_sampled_mV = bin_y_over_x(i_over_nA, v_over_mV, spec.Ibins_nA)
    i_sampled_nA = bin_y_over_x(v_over_mV, i_over_nA, spec.Vbins_mV)
    dG_G0 = np.gradient(i_sampled_nA, spec.Vbins_mV) / G_0_muS
    dR_R0 = np.gradient(v_sampled_mV, spec.Ibins_nA) * G_0_muS
    return i_sampled_nA, v_sampled_mV, dG_G0, dR_R0


def test_downsampling_uses_sampling_spec_nu_hz() -> None:
    """The high-level downsampling wrapper should use ``nu_Hz`` from the spec."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec()

    downsampled = sampling.downsampling(trace, samplingspec=spec)
    expected = sampling.downsample_trace(trace, nu_Hz=spec.nu_Hz)

    assert np.allclose(downsampled["t_s"], expected["t_s"])
    assert np.allclose(downsampled["V_mV"], expected["V_mV"])
    assert np.allclose(downsampled["I_nA"], expected["I_nA"])


def test_upsampling_densifies_trace_and_preserves_metadata() -> None:
    """Explicit upsampling should interpolate all trace arrays."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec(N_up=5)

    upsampled = sampling.upsampling(trace, samplingspec=spec)

    assert upsampled["meta"] == trace["meta"]
    assert upsampled["t_s"].size == trace["t_s"].size * spec.N_up
    assert upsampled["V_mV"].size == trace["V_mV"].size * spec.N_up
    assert upsampled["I_nA"].size == trace["I_nA"].size * spec.N_up
    assert upsampled["t_s"][0] == pytest.approx(trace["t_s"][0])
    assert upsampled["t_s"][-1] == pytest.approx(trace["t_s"][-1])


def test_offset_correction_subtracts_offsets() -> None:
    """The explicit offset-correction stage should shift the trace arrays."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
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


def test_binning_matches_manual_flow_for_upsampled_trace() -> None:
    """Pure binning should match manual binning of the prepared trace."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.0, i_shift_nA=0.0)
    spec = _make_spec()
    prepared = sampling.upsampling(trace, samplingspec=spec)

    out = sampling.binning(prepared, samplingspec=spec)
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _manual_binning(prepared, spec)

    assert np.allclose(out["I_nA"], i_exp_nA, equal_nan=True)
    assert np.allclose(out["V_mV"], v_exp_mV, equal_nan=True)
    assert np.allclose(out["dG_G0"], dG_exp_G0, equal_nan=True)
    assert np.allclose(out["dR_R0"], dR_exp_R0, equal_nan=True)


def test_binning_of_upsampled_trace_matches_previous_hidden_upsample() -> None:
    """Explicit upsampling should preserve the old hidden binning behavior."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.0, i_shift_nA=0.0)
    spec = _make_spec()

    out = sampling.binning(
        sampling.upsampling(trace, samplingspec=spec),
        samplingspec=spec,
    )
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _legacy_hidden_upsample_binning(
        trace,
        spec,
    )

    assert np.allclose(out["I_nA"], i_exp_nA, equal_nan=True)
    assert np.allclose(out["V_mV"], v_exp_mV, equal_nan=True)
    assert np.allclose(out["dG_G0"], dG_exp_G0, equal_nan=True)
    assert np.allclose(out["dR_R0"], dR_exp_R0, equal_nan=True)


def test_sampling_matches_manual_pipeline_with_offsets() -> None:
    """Full sampling should use the explicit stage order."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec()
    offset = _make_offset(0.4, 0.3)

    out = sampling.sample(trace, samplingspec=spec, offsetanalysis=offset)
    corrected = sampling.offset_correction(trace, offsetanalysis=offset)
    downsampled = sampling.downsampling(corrected, samplingspec=spec)
    upsampled = sampling.upsampling(downsampled, samplingspec=spec)
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _manual_binning(
        upsampled,
        spec,
    )

    assert np.allclose(out["Vbins_mV"], spec.Vbins_mV)
    assert np.allclose(out["Ibins_nA"], spec.Ibins_nA)
    assert out["meta"] == trace["meta"]
    assert np.allclose(out["I_nA"], i_exp_nA, equal_nan=True)
    assert np.allclose(out["V_mV"], v_exp_mV, equal_nan=True)
    assert np.allclose(out["dG_G0"], dG_exp_G0, equal_nan=True)
    assert np.allclose(out["dR_R0"], dR_exp_R0, equal_nan=True)


def test_sampling_without_offsetanalysis_uses_zero_offsets() -> None:
    """Missing offset input should behave like explicit zero offsets."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec()
    zero_offset = _make_offset(0.0, 0.0)

    implicit = sampling.sample(trace, samplingspec=spec, offsetanalysis=None)
    explicit = sampling.sample(trace, samplingspec=spec, offsetanalysis=zero_offset)

    assert np.allclose(implicit["I_nA"], explicit["I_nA"], equal_nan=True)
    assert np.allclose(implicit["V_mV"], explicit["V_mV"], equal_nan=True)
    assert np.allclose(implicit["dG_G0"], explicit["dG_G0"], equal_nan=True)
    assert np.allclose(implicit["dR_R0"], explicit["dR_R0"], equal_nan=True)


def test_sample_calls_stages_in_explicit_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline orchestration should follow the explicit stage order."""
    calls: list[str] = []
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec(apply_smoothing=True, median_bins=3, sigma_bins=1.0)
    sample_out = {
        "meta": trace["meta"],
        "Vbins_mV": spec.Vbins_mV,
        "Ibins_nA": spec.Ibins_nA,
        "I_nA": np.zeros_like(spec.Vbins_mV),
        "V_mV": np.zeros_like(spec.Ibins_nA),
        "dG_G0": np.zeros_like(spec.Vbins_mV),
        "dR_R0": np.zeros_like(spec.Ibins_nA),
    }

    def _offset(traces, *, offsetanalysis):
        calls.append("offset")
        assert offsetanalysis is not None
        return traces

    def _downsampling(traces, *, samplingspec, show_progress):
        calls.append("downsample")
        return traces

    def _upsampling(traces, *, samplingspec, show_progress):
        calls.append("upsample")
        return traces

    def _binning(traces, *, samplingspec, show_progress):
        calls.append("binning")
        return sample_out

    def _smooth(samples, *, samplingspec, show_progress):
        calls.append("smooth")
        return samples

    monkeypatch.setattr(pipeline_mod, "offset_correction", _offset)
    monkeypatch.setattr(pipeline_mod, "downsampling", _downsampling)
    monkeypatch.setattr(pipeline_mod, "upsampling", _upsampling)
    monkeypatch.setattr(pipeline_mod, "binning", _binning)
    monkeypatch.setattr(pipeline_mod, "smooth", _smooth)

    out = sampling.sample(trace, samplingspec=spec, offsetanalysis=None)

    assert out is sample_out
    assert calls == ["offset", "downsample", "upsample", "binning", "smooth"]


@pytest.mark.parametrize(
    ("spec", "expected_calls"),
    [
        (
            _make_spec(apply_offset_correction=False),
            ["downsample", "upsample", "binning"],
        ),
        (
            _make_spec(apply_downsampling=False),
            ["offset", "upsample", "binning"],
        ),
        (
            _make_spec(apply_upsampling=False),
            ["offset", "downsample", "binning"],
        ),
        (
            _make_spec(
                apply_offset_correction=False,
                apply_downsampling=False,
                apply_upsampling=False,
                apply_smoothing=True,
                median_bins=3,
                sigma_bins=1.0,
            ),
            ["binning", "smooth"],
        ),
    ],
)
def test_sampling_skips_disabled_stages(
    monkeypatch: pytest.MonkeyPatch,
    spec: sampling.SamplingSpec,
    expected_calls: list[str],
) -> None:
    """Disabled stages should behave as true no-ops in the pipeline."""
    calls: list[str] = []
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    sample_out = {
        "meta": trace["meta"],
        "Vbins_mV": spec.Vbins_mV,
        "Ibins_nA": spec.Ibins_nA,
        "I_nA": np.zeros_like(spec.Vbins_mV),
        "V_mV": np.zeros_like(spec.Ibins_nA),
        "dG_G0": np.zeros_like(spec.Vbins_mV),
        "dR_R0": np.zeros_like(spec.Ibins_nA),
    }

    monkeypatch.setattr(
        pipeline_mod,
        "offset_correction",
        lambda traces, *, offsetanalysis: calls.append("offset") or traces,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "downsampling",
        lambda traces, *, samplingspec, show_progress: (
            calls.append("downsample") or traces
        ),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "upsampling",
        lambda traces, *, samplingspec, show_progress: (
            calls.append("upsample") or traces
        ),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "binning",
        lambda traces, *, samplingspec, show_progress: (
            calls.append("binning") or sample_out
        ),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "smooth",
        lambda samples, *, samplingspec, show_progress: (
            calls.append("smooth") or samples
        ),
    )

    sampling.sample(trace, samplingspec=spec, offsetanalysis=_make_offset(0.4, 0.3))

    assert calls == expected_calls


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
    spec = _make_spec(apply_smoothing=True, median_bins=3, sigma_bins=1.0)

    out = sampling.sample(
        traces,
        samplingspec=spec,
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
