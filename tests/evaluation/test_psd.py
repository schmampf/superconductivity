"""Tests for PSD helpers built on IV trace containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.analysis.psd as psd
from superconductivity.evaluation.traces import Trace, Traces
from superconductivity.evaluation.sampling import downsample_trace
from superconductivity.utilities.meta import AxisSpec, DataSpec, ParamSpec


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    scale: float = 1.0,
) -> Trace:
    i_nA = scale * np.asarray([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
    return Trace(
        I_nA=i_nA,
        V_mV=2.0 * i_nA,
        t_s=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    )


def _make_psd_trace(value: float) -> psd.PSDTrace:
    return psd.PSDTrace(
        f_Hz=np.asarray([0.0, 0.5], dtype=np.float64),
        I_psd_nA2_per_Hz=np.asarray([value, value], dtype=np.float64),
        V_psd_mV2_per_Hz=np.asarray([2.0 * value, 2.0 * value], dtype=np.float64),
        nu_Hz=1.0,
        nyquist_Hz=0.5,
    )


def test_psd_analysis_returns_psd_data() -> None:
    """Single-trace PSD should return PSD data for the given trace."""
    trace = _make_iv_trace("nu=1dBm", 0, 1.0)

    out = psd.psd_analysis(
        trace,
        spec=psd.PSDSpec(detrend=False),
    )

    assert np.allclose(out["f_Hz"], np.asarray([0.0, 0.25, 0.5]))
    assert np.allclose(
        out["V_psd_mV2_per_Hz"],
        4.0 * out["I_psd_nA2_per_Hz"],
    )
    assert out["nu_Hz"] == pytest.approx(1.0)
    assert out["nyquist_Hz"] == pytest.approx(0.5)


def test_psd_analysis_returns_collection() -> None:
    """Multi-trace PSD should return a stacked PSD container."""
    traces = Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0),
            _make_iv_trace("b", 1, 5.0, scale=2.0),
        ],
    )

    out = psd.psd_analysis(
        traces,
        spec=psd.PSDSpec(detrend=False),
    )

    assert len(out) == 2
    assert out.keys() == (
        "y",
        "i",
        "indices",
        "skeys",
        "specific_keys",
        "f_Hz",
        "I_psd_nA2_per_Hz",
        "V_psd_mV2_per_Hz",
        "nu_Hz",
        "nyquist_Hz",
    )
    assert out.specific_keys == traces.specific_keys
    assert np.allclose(out.indices, traces.indices)
    assert np.allclose(out.yvalues, traces.yvalues)
    assert len(out.I_psd_nA2_per_Hz) == 2
    assert len(out.V_psd_mV2_per_Hz) == 2
    assert len(out.f_Hz) == 2
    assert isinstance(out.f_Hz[0], AxisSpec)
    assert isinstance(out.I_psd_nA2_per_Hz[0], DataSpec)
    assert isinstance(out.V_psd_mV2_per_Hz[0], DataSpec)
    assert isinstance(out.nu_Hz[0], ParamSpec)
    assert isinstance(out.nyquist_Hz[0], ParamSpec)
    assert out.i == out.index


def test_downsample_trace_uses_nu_hz_as_sample_rate() -> None:
    """Downsampling should control the PSD Nyquist band."""
    i_nA = np.asarray([0.0, 1.0, 0.0, -1.0] * 2, dtype=np.float64)
    trace = Trace(
        I_nA=i_nA,
        V_mV=2.0 * i_nA,
        t_s=np.arange(i_nA.size, dtype=np.float64),
    )

    downsampled = downsample_trace(trace, nu_Hz=0.5)
    out = psd.psd_analysis(
        downsampled,
        spec=psd.PSDSpec(detrend=False),
    )

    assert np.asarray(downsampled["t_s"]).size == 4
    assert out["nyquist_Hz"] == pytest.approx(0.25)
    assert out["nu_Hz"] == pytest.approx(0.5)


def test_psdtraces_reject_empty_input() -> None:
    """PSDTraces should require at least one trace."""
    with pytest.raises(ValueError, match="traces must not be empty"):
        psd.PSDTraces(traces=[])


def test_psdtraces_support_positional_access_only() -> None:
    """PSDTraces should preserve positional access to payload results."""
    traces = psd.PSDTraces(
        traces=[
            _make_psd_trace(0.0),
            _make_psd_trace(1.0),
        ],
    )

    assert traces[0]["nyquist_Hz"] == pytest.approx(0.5)
    assert traces[1]["nu_Hz"] == pytest.approx(1.0)
    assert traces.keys() == (
        "y",
        "i",
        "indices",
        "skeys",
        "specific_keys",
        "f_Hz",
        "I_psd_nA2_per_Hz",
        "V_psd_mV2_per_Hz",
        "nu_Hz",
        "nyquist_Hz",
    )
