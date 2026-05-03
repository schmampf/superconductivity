"""Tests for PSD helpers built on IV trace containers."""

from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation.traces import Trace, Traces


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


def test_trace_exposes_lazy_psd_properties() -> None:
    """Trace should expose lazy PSD and sampling-rate properties."""
    trace = _make_iv_trace("nu=1dBm", 0, 1.0)
    assert trace.dt_s == pytest.approx(1.0)
    assert trace.nu_Hz == pytest.approx(1.0)
    assert np.allclose(trace.f_Hz, np.asarray([0.0, 0.25, 0.5]))
    assert np.allclose(trace.Vpsd_mV2s, 4.0 * trace.Ipsd_nA2s)


def test_psd_analysis_returns_collection() -> None:
    """Collection-level PSD views should mirror the contained traces."""
    traces = Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0),
            _make_iv_trace("b", 1, 5.0, scale=2.0),
        ],
    )
    assert len(traces.f_Hz) == 2
    assert len(traces.Ipsd_nA2s) == 2
    assert len(traces.Vpsd_mV2s) == 2
    assert np.allclose(traces.nu_Hz, [1.0, 1.0])
    assert np.allclose(traces.dt_s, [1.0, 1.0])


def test_downsample_trace_uses_nu_hz_as_sample_rate() -> None:
    """Downsampling should control the PSD Nyquist band."""
    i_nA = np.asarray([0.0, 1.0, 0.0, -1.0] * 2, dtype=np.float64)
    trace = Trace(
        I_nA=i_nA,
        V_mV=2.0 * i_nA,
        t_s=np.arange(i_nA.size, dtype=np.float64),
    )

    downsampled = trace.resample(nu_Hz=0.5)

    assert np.asarray(downsampled["t_s"]).size == 4
    assert downsampled.nu_Hz == pytest.approx(0.5)
    assert np.allclose(downsampled.f_Hz, np.asarray([0.0, 0.125, 0.25]))


def test_traces_expose_resample_and_psd_views() -> None:
    """Trace collections should expose the same derived views as traces."""
    traces = Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0),
            _make_iv_trace("b", 1, 5.0, scale=2.0),
        ],
    )

    resampled = traces.resample(nu_Hz=1.0)

    assert len(resampled) == 2
    assert np.allclose(resampled.dt_s, [1.0, 1.0])
    assert np.allclose(resampled.nu_Hz, [1.0, 1.0])
    assert len(resampled.f_Hz) == 2
    assert len(resampled.Ipsd_nA2s) == 2
    assert len(resampled.Vpsd_mV2s) == 2
