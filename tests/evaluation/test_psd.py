"""Tests for PSD helpers built on IV trace containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.psd as psd
from superconductivity.evaluation.ivdata import IVTrace, IVTraces


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    scale: float = 1.0,
) -> IVTrace:
    i_nA = scale * np.asarray([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "I_nA": i_nA,
        "V_mV": 2.0 * i_nA,
        "t_s": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    }


def _make_psd_trace(
    specific_key: str,
    index: int,
    yvalue: float,
) -> psd.PSDTrace:
    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "raw_I_psd_nA2_per_Hz": np.asarray([float(index)], dtype=np.float64),
        "raw_V_psd_mV2_per_Hz": np.asarray([float(index)], dtype=np.float64),
        "raw_f_Hz": np.asarray([0.0], dtype=np.float64),
        "raw_sigma_I_nA": float(index),
        "raw_sigma_V_mV": 2.0 * float(index),
        "raw_nu_Hz": 1.0,
        "raw_nyquist_Hz": 0.5,
        "raw_sigma_cutoff_Hz": 0.5,
        "downsampled_I_psd_nA2_per_Hz": np.asarray(
            [10.0 + float(index)],
            dtype=np.float64,
        ),
        "downsampled_V_psd_mV2_per_Hz": np.asarray(
            [20.0 + float(index)],
            dtype=np.float64,
        ),
        "downsampled_f_Hz": np.asarray([0.0], dtype=np.float64),
        "downsampled_sigma_I_nA": 10.0 + float(index),
        "downsampled_sigma_V_mV": 20.0 + float(index),
        "downsampled_nu_Hz": 0.5,
        "downsampled_nyquist_Hz": 0.25,
        "downsampled_sigma_cutoff_Hz": 0.25,
    }


def test_get_psd_returns_downsampled_trace_and_metadata() -> None:
    """Single-trace PSD should return the downsampled trace and metadata."""
    trace = _make_iv_trace("nu=1dBm", 0, 1.0)

    downsampled_trace, out = psd.get_psd(
        trace=trace,
        spec=psd.PSDSpec(nu_Hz=1.0, detrend=False),
    )

    assert downsampled_trace["specific_key"] == trace["specific_key"]
    assert downsampled_trace["index"] == trace["index"]
    assert downsampled_trace["yvalue"] == trace["yvalue"]
    assert np.allclose(downsampled_trace["t_s"], trace["t_s"])
    assert np.allclose(downsampled_trace["I_nA"], trace["I_nA"])
    assert np.allclose(downsampled_trace["V_mV"], trace["V_mV"])
    assert out["specific_key"] == "nu=1dBm"
    assert out["index"] == 0
    assert out["yvalue"] == pytest.approx(1.0)
    assert np.allclose(out["raw_f_Hz"], np.asarray([0.0, 0.25, 0.5]))
    assert np.allclose(
        out["downsampled_f_Hz"],
        np.asarray([0.0, 0.25, 0.5]),
    )
    assert np.allclose(
        out["raw_V_psd_mV2_per_Hz"],
        4.0 * out["raw_I_psd_nA2_per_Hz"],
    )
    assert np.allclose(
        out["downsampled_V_psd_mV2_per_Hz"],
        4.0 * out["downsampled_I_psd_nA2_per_Hz"],
    )
    assert out["raw_sigma_I_nA"] > 0.0
    assert out["raw_sigma_V_mV"] == pytest.approx(2.0 * out["raw_sigma_I_nA"])
    assert out["downsampled_sigma_I_nA"] > 0.0
    assert out["downsampled_sigma_V_mV"] == pytest.approx(
        2.0 * out["downsampled_sigma_I_nA"]
    )
    assert out["raw_nu_Hz"] == pytest.approx(1.0)
    assert out["downsampled_nu_Hz"] == pytest.approx(1.0)
    assert out["raw_sigma_cutoff_Hz"] == pytest.approx(0.5)
    assert out["downsampled_sigma_cutoff_Hz"] == pytest.approx(0.5)


def test_get_psds_returns_downsampled_collection_and_lookup_methods() -> None:
    """Multi-trace PSD should mirror the IV collection API."""
    traces = IVTraces(
        traces=[
            _make_iv_trace("a", 0, 1.0),
            _make_iv_trace("b", 1, 5.0, scale=2.0),
        ],
    )

    downsampled_traces, out = psd.get_psds(
        traces=traces,
        spec=psd.PSDSpec(nu_Hz=1.0, detrend=False),
    )

    assert isinstance(downsampled_traces, IVTraces)
    assert len(downsampled_traces) == 2
    assert downsampled_traces[0]["specific_key"] == "a"
    assert downsampled_traces[1]["specific_key"] == "b"
    assert np.allclose(downsampled_traces[0]["t_s"], traces[0]["t_s"])
    assert len(out) == 2
    assert out.keys == ["a", "b"]
    assert np.allclose(out.yvalues, np.asarray([1.0, 5.0]))
    assert out.by_key("a")["index"] == 0
    assert out.by_value(5.0)["specific_key"] == "b"
    assert len(out.raw_I_psd_nA2_per_Hz) == 2
    assert len(out.raw_V_psd_mV2_per_Hz) == 2
    assert len(out.raw_f_Hz) == 2
    assert len(out.downsampled_I_psd_nA2_per_Hz) == 2
    assert len(out.downsampled_V_psd_mV2_per_Hz) == 2
    assert len(out.downsampled_f_Hz) == 2
    assert np.allclose(
        out.raw_sigma_V_mV,
        2.0 * out.raw_sigma_I_nA,
    )
    assert np.allclose(
        out.downsampled_sigma_V_mV,
        2.0 * out.downsampled_sigma_I_nA,
    )


def test_get_psd_uses_nu_hz_as_sample_rate() -> None:
    """Band-limited sigma should use the target sample-rate Nyquist band."""
    i_nA = np.asarray([0.0, 1.0, 0.0, -1.0] * 2, dtype=np.float64)
    trace: IVTrace = {
        "specific_key": "nu=1dBm",
        "index": 0,
        "yvalue": 1.0,
        "I_nA": i_nA,
        "V_mV": 2.0 * i_nA,
        "t_s": np.arange(i_nA.size, dtype=np.float64),
    }

    downsampled_trace, out = psd.get_psd(
        trace=trace,
        spec=psd.PSDSpec(nu_Hz=0.5, detrend=False),
    )

    assert downsampled_trace["t_s"].size == 4
    assert out["raw_sigma_cutoff_Hz"] == pytest.approx(0.25)
    assert out["downsampled_sigma_cutoff_Hz"] == pytest.approx(0.25)
    assert out["downsampled_nyquist_Hz"] == pytest.approx(0.25)
    assert out["downsampled_nu_Hz"] == pytest.approx(0.5)


def test_psdtraces_by_value_rejects_ambiguous_matches() -> None:
    """Duplicate yvalues should require the plural lookup method."""
    traces = psd.PSDTraces(
        traces=[
            _make_psd_trace("a", 0, 1.0),
            _make_psd_trace("b", 1, 1.0),
        ],
    )

    with pytest.raises(ValueError, match="matches multiple traces"):
        traces.by_value(1.0)

    matches = traces.all_by_value(1.0)
    assert [trace["specific_key"] for trace in matches] == ["a", "b"]
