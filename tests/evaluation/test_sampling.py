"""Tests for explicit IV sampling stages and containers."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import superconductivity.api as api_module
import superconductivity.evaluation as evaluation_module
import superconductivity.evaluation.sampling as sampling
from superconductivity.utilities.meta import Dataset, data
from superconductivity.evaluation.traces import Trace, Traces
from superconductivity.utilities.constants import G0_muS
from superconductivity.utilities.functions.binning import bin
from superconductivity.utilities.functions.upsampling import upsample as upsample_xy
from superconductivity.utilities.meta.label import label
from superconductivity.utilities.transport import TransportDatasetSpec

pipeline_mod = importlib.import_module(
    "superconductivity.evaluation.sampling.pipeline",
)


def test_sampling_exports_expose_upsampling() -> None:
    """Public exports should expose the explicit upsampling stage."""
    assert hasattr(sampling, "SamplingSpec")
    assert hasattr(sampling, "upsampling")

    assert hasattr(evaluation_module, "SamplingSpec")
    assert hasattr(evaluation_module, "upsampling")

    assert hasattr(api_module, "SamplingSpec")
    assert hasattr(api_module, "upsampling")


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
    return Trace(
        V_mV=v_true_mV + v_shift_mV,
        I_nA=i_true_nA + i_shift_nA,
        t_s=t_s,
    )


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


def _manual_binning(
    trace: Trace,
    spec: sampling.SamplingSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v_trace_mV = np.asarray(trace["V_mV"], dtype=np.float64)
    i_trace_nA = np.asarray(trace["I_nA"], dtype=np.float64)
    v_sampled_mV = bin(z=v_trace_mV, x=i_trace_nA, xbins=spec.Ibins_nA)
    i_sampled_nA = bin(z=i_trace_nA, x=v_trace_mV, xbins=spec.Vbins_mV)
    dG_G0 = np.gradient(i_sampled_nA, spec.Vbins_mV) / G0_muS
    dR_R0 = np.gradient(v_sampled_mV, spec.Ibins_nA) * G0_muS
    return i_sampled_nA, v_sampled_mV, dG_G0, dR_R0


def _explicit_upsample_then_bin(
    trace: Trace,
    spec: sampling.SamplingSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i_over_nA = upsample_xy(
        np.asarray(trace["I_nA"], dtype=np.float64),
        N_up=spec.N_up,
        axis=0,
        method="linear",
    )
    v_over_mV = upsample_xy(
        np.asarray(trace["V_mV"], dtype=np.float64),
        N_up=spec.N_up,
        axis=0,
        method="linear",
    )
    v_sampled_mV = bin(z=v_over_mV, x=i_over_nA, xbins=spec.Ibins_nA)
    i_sampled_nA = bin(z=i_over_nA, x=v_over_mV, xbins=spec.Vbins_mV)
    dG_G0 = np.gradient(i_sampled_nA, spec.Vbins_mV) / G0_muS
    dR_R0 = np.gradient(v_sampled_mV, spec.Ibins_nA) * G0_muS
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


def test_upsampling_densifies_trace_and_preserves_axes() -> None:
    """Explicit upsampling should interpolate all trace arrays."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec(N_up=5)

    upsampled = sampling.upsampling(trace, samplingspec=spec)

    assert (
        np.asarray(upsampled["t_s"]).size == np.asarray(trace["t_s"]).size * spec.N_up
    )
    assert (
        np.asarray(upsampled["V_mV"]).size == np.asarray(trace["V_mV"]).size * spec.N_up
    )
    assert (
        np.asarray(upsampled["I_nA"]).size == np.asarray(trace["I_nA"]).size * spec.N_up
    )
    assert upsampled["t_s"][0] == pytest.approx(trace["t_s"][0])
    assert upsampled["t_s"][-1] == pytest.approx(trace["t_s"][-1])


def test_offset_correction_subtracts_offsets() -> None:
    """The explicit offset-correction stage should shift the trace arrays."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec(
        Voff_mV=data("Voff_mV", np.asarray([0.4], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.3], dtype=np.float64)),
    )
    corrected = sampling.offset_correction(
        trace,
        samplingspec=spec,
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

    exp_v, exp_i = sampling.binning(prepared, samplingspec=spec)
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _manual_binning(prepared, spec)

    assert np.allclose(exp_v.I_nA.values, i_exp_nA, equal_nan=True)
    assert np.allclose(exp_i.V_mV.values, v_exp_mV, equal_nan=True)
    assert np.allclose(exp_v.dG_G0.values, dG_exp_G0, equal_nan=True)
    assert np.allclose(exp_i.dR_R0.values, dR_exp_R0, equal_nan=True)


def test_binning_of_upsampled_trace_matches_direct_reference() -> None:
    """Explicit upsampling should match the direct upsample-then-bin reference."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.0, i_shift_nA=0.0)
    spec = _make_spec()

    exp_v, exp_i = sampling.binning(
        sampling.upsampling(trace, samplingspec=spec),
        samplingspec=spec,
    )
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _explicit_upsample_then_bin(
        trace,
        spec,
    )

    assert np.allclose(exp_v.I_nA.values, i_exp_nA, equal_nan=True)
    assert np.allclose(exp_i.V_mV.values, v_exp_mV, equal_nan=True)
    assert np.allclose(exp_v.dG_G0.values, dG_exp_G0, equal_nan=True)
    assert np.allclose(exp_i.dR_R0.values, dR_exp_R0, equal_nan=True)


def test_sampling_matches_manual_pipeline_with_offsets() -> None:
    """Full sampling should use the explicit stage order."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec(
        Voff_mV=data("Voff_mV", np.asarray([0.4], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.3], dtype=np.float64)),
    )

    exp_v, exp_i = sampling.sample(trace, samplingspec=spec)
    corrected = sampling.offset_correction(
        trace,
        samplingspec=spec,
    )
    downsampled = sampling.downsampling(corrected, samplingspec=spec)
    upsampled = sampling.upsampling(downsampled, samplingspec=spec)
    i_exp_nA, v_exp_mV, dG_exp_G0, dR_exp_R0 = _manual_binning(
        upsampled,
        spec,
    )

    assert np.allclose(exp_v.V_mV.values, spec.Vbins_mV)
    assert np.allclose(exp_i.I_nA.values, spec.Ibins_nA)
    assert np.allclose(exp_v.I_nA.values, i_exp_nA, equal_nan=True)
    assert np.allclose(exp_i.V_mV.values, v_exp_mV, equal_nan=True)
    assert np.allclose(exp_v.dG_G0.values, dG_exp_G0, equal_nan=True)
    assert np.allclose(exp_i.dR_R0.values, dR_exp_R0, equal_nan=True)
    assert isinstance(exp_v, TransportDatasetSpec)
    assert isinstance(exp_i, TransportDatasetSpec)
    assert exp_v.V_mV.order == 0
    assert exp_i.I_nA.order == 0


def test_sampling_without_offset_spec_uses_zero_offsets() -> None:
    """Missing offset input should behave like explicit zero offsets."""
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    implicit_spec = _make_spec()
    explicit_spec = _make_spec(
        Voff_mV=data("Voff_mV", np.asarray([0.0], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.0], dtype=np.float64)),
    )

    implicit_v, implicit_i = sampling.sample(trace, samplingspec=implicit_spec)
    explicit_v, explicit_i = sampling.sample(trace, samplingspec=explicit_spec)

    assert np.allclose(implicit_v.I_nA.values, explicit_v.I_nA.values, equal_nan=True)
    assert np.allclose(implicit_i.V_mV.values, explicit_i.V_mV.values, equal_nan=True)
    assert np.allclose(implicit_v.dG_G0.values, explicit_v.dG_G0.values, equal_nan=True)
    assert np.allclose(implicit_i.dR_R0.values, explicit_i.dR_R0.values, equal_nan=True)


def test_sample_calls_stages_in_explicit_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline orchestration should follow the explicit stage order."""
    calls: list[str] = []
    trace = _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3)
    spec = _make_spec(apply_smoothing=True, median_bins=3, sigma_bins=1.0)
    sample_out = sampling.binning(
        sampling.upsampling(trace, samplingspec=spec),
        samplingspec=spec,
    )

    def _offset(traces, *, samplingspec):
        calls.append("offset")
        assert samplingspec is not None
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

    out = sampling.sample(trace, samplingspec=spec)

    assert out == sample_out
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
    sample_out = sampling.binning(
        sampling.upsampling(trace, samplingspec=spec),
        samplingspec=spec,
    )

    monkeypatch.setattr(
        pipeline_mod,
        "offset_correction",
        lambda traces, *, samplingspec: calls.append("offset") or traces,
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

    sampling.sample(trace, samplingspec=spec)

    assert calls == expected_calls


def test_sampling_returns_collection() -> None:
    """Collection sampling should return stacked sampled results."""
    traces = Traces.from_fields(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
        specific_keys=["a", "b"],
        indices=[0, 1],
        yvalues=[1.0, 5.0],
        y_label=None,
    )
    spec = _make_spec(
        Voff_mV=data("Voff_mV", np.asarray([0.4, 0.2], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.3, 0.1], dtype=np.float64)),
    )

    exp_v, exp_i = sampling.sample(traces, samplingspec=spec, show_progress=False)

    assert isinstance(exp_v, TransportDatasetSpec)
    assert isinstance(exp_i, TransportDatasetSpec)
    assert np.allclose(exp_v.V_mV.values, spec.Vbins_mV)
    assert np.allclose(exp_i.I_nA.values, spec.Ibins_nA)
    assert np.allclose(exp_v.y.values, np.asarray([1.0, 5.0]))
    assert np.allclose(exp_i.y.values, np.asarray([1.0, 5.0]))
    assert exp_v.y.code_label == "y"
    assert exp_i.y.code_label == "y"
    assert exp_v.I_nA.values.shape == (2, spec.Vbins_mV.size)
    assert exp_i.V_mV.values.shape == (2, spec.Ibins_nA.size)
    assert exp_v.dG_G0.values.shape == (2, spec.Vbins_mV.size)
    assert exp_i.dR_R0.values.shape == (2, spec.Ibins_nA.size)


def test_sampling_uses_trace_label_for_collection_axis() -> None:
    """The sampled collection axis should inherit the trace label."""
    traces = Traces.from_fields(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
        specific_keys=["nu=1dBm", "nu=5dBm"],
        indices=[0, 1],
        yvalues=[1.0, 5.0],
        y_label=label("Aout_mV"),
    )
    spec = _make_spec(
        Voff_mV=data("Voff_mV", np.asarray([0.4, 0.2], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.3, 0.1], dtype=np.float64)),
    )

    exp_v, exp_i = sampling.sample(traces, samplingspec=spec, show_progress=False)

    assert exp_v.Aout_mV.values.shape == (2,)
    assert exp_i.Aout_mV.values.shape == (2,)
    assert np.allclose(exp_v.Aout_mV.values, np.asarray([1.0, 5.0]))
    assert np.allclose(exp_i.Aout_mV.values, np.asarray([1.0, 5.0]))


def test_sampling_with_smoothing_returns_collection() -> None:
    """The full pipeline should optionally return smoothed sampled data."""
    traces = Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )
    spec = _make_spec(
        Voff_mV=data("Voff_mV", np.asarray([0.4, 0.2], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.3, 0.1], dtype=np.float64)),
        apply_smoothing=True,
        median_bins=3,
        sigma_bins=1.0,
    )

    exp_v, exp_i = sampling.sample(traces, samplingspec=spec, show_progress=False)

    assert isinstance(exp_v, TransportDatasetSpec)
    assert isinstance(exp_i, TransportDatasetSpec)


def test_offset_correction_rejects_length_mismatch() -> None:
    """Collection offset correction should require aligned lengths."""
    traces = Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )
    spec = _make_spec(
        Voff_mV=data("Voff_mV", np.asarray([0.4], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.3], dtype=np.float64)),
    )

    with pytest.raises(ValueError, match="must match the number of traces"):
        sampling.offset_correction(traces, samplingspec=spec)
