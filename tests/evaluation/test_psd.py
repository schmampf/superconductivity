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
        "I_psd_nA2_per_Hz": np.asarray([float(index)], dtype=np.float64),
        "V_psd_mV2_per_Hz": np.asarray([float(index)], dtype=np.float64),
        "f_Hz": np.asarray([0.0], dtype=np.float64),
    }


def test_get_psd_returns_psdtrace_with_metadata() -> None:
    """Single-trace PSD should keep IV metadata."""
    trace = _make_iv_trace("nu=1dBm", 0, 1.0)

    out = psd.get_psd(
        trace=trace,
        detrend=False,
        window="none",
    )

    assert out["specific_key"] == "nu=1dBm"
    assert out["index"] == 0
    assert out["yvalue"] == pytest.approx(1.0)
    assert np.allclose(out["f_Hz"], np.asarray([0.0, 0.25, 0.5]))
    assert np.allclose(
        out["V_psd_mV2_per_Hz"],
        4.0 * out["I_psd_nA2_per_Hz"],
    )


def test_get_psds_returns_collection_with_lookup_methods() -> None:
    """Multi-trace PSD should mirror the IV collection API."""
    traces = IVTraces(
        traces=[
            _make_iv_trace("a", 0, 1.0),
            _make_iv_trace("b", 1, 5.0, scale=2.0),
        ],
    )

    out = psd.get_psds(
        traces=traces,
        detrend=False,
        window="none",
    )

    assert len(out) == 2
    assert out.keys == ["a", "b"]
    assert np.allclose(out.yvalues, np.asarray([1.0, 5.0]))
    assert out.by_key("a")["index"] == 0
    assert out.by_value(5.0)["specific_key"] == "b"
    assert len(out.I_psd_nA2_per_Hz) == 2
    assert len(out.V_psd_mV2_per_Hz) == 2
    assert len(out.f_Hz) == 2


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
