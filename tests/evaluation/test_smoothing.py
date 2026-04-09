"""Tests for sampled-IV smoothing containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.sampling as smoothing
from superconductivity.evaluation.sampling import Sample, Samples, SamplingSpec
from superconductivity.evaluation.traces import TraceMeta


def _make_sampling_spec(**updates: object) -> SamplingSpec:
    defaults: dict[str, object] = {
        "N_up": 4,
        "Vbins_mV": np.linspace(-1.0, 1.0, 9),
        "Ibins_nA": np.linspace(-2.0, 2.0, 11),
        "apply_smoothing": True,
    }
    defaults.update(updates)
    return SamplingSpec(**defaults)


def _make_sampling_trace(
    specific_key: str,
    index: int,
    yvalue: float,
) -> Sample:
    vbin_mV = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    ibin_nA = np.linspace(-2.0, 2.0, 11, dtype=np.float64)
    i_nA = np.asarray(
        [np.nan, np.nan, -0.95, -0.05, 0.35, 0.10, 1.05, np.nan, np.nan],
        dtype=np.float64,
    )
    v_mV = np.asarray(
        [
            np.nan,
            np.nan,
            -0.98,
            -0.72,
            -0.18,
            0.05,
            0.32,
            0.86,
            1.02,
            np.nan,
            np.nan,
        ],
        dtype=np.float64,
    )
    return {
        "meta": TraceMeta(
            specific_key=specific_key,
            index=index,
            yvalue=yvalue,
        ),
        "Vbins_mV": vbin_mV,
        "Ibins_nA": ibin_nA,
        "I_nA": i_nA,
        "V_mV": v_mV,
        "dG_G0": np.full(vbin_mV.shape, np.nan, dtype=np.float64),
        "dR_R0": np.full(ibin_nA.shape, np.nan, dtype=np.float64),
    }


def _roughness(y: np.ndarray) -> float:
    """Estimate curve roughness from first differences on finite support."""
    finite = np.isfinite(y)
    if np.sum(finite) < 3:
        return 0.0
    return float(np.nanstd(np.diff(y[finite])))


def test_smoothing_preserves_nan_edges_and_metadata() -> None:
    """Smoothing should keep metadata and unsupported edges unchanged."""
    trace = _make_sampling_trace("a", 0, 1.0)
    spec = _make_sampling_spec(median_bins=3, sigma_bins=1.0)

    out = smoothing.smooth(trace, samplingspec=spec)

    assert out["meta"] == trace["meta"]
    assert np.allclose(out["Vbins_mV"], trace["Vbins_mV"])
    assert np.allclose(out["Ibins_nA"], trace["Ibins_nA"])
    assert np.isnan(out["I_nA"][:2]).all()
    assert np.isnan(out["I_nA"][-2:]).all()
    assert np.isnan(out["V_mV"][:2]).all()
    assert np.isnan(out["V_mV"][-2:]).all()
    assert np.isfinite(out["I_nA"][2:-2]).all()
    assert np.isfinite(out["V_mV"][2:-2]).all()


def test_smoothing_reduces_curve_roughness() -> None:
    """The default median->gaussian pipeline should damp roughness."""
    trace = _make_sampling_trace("a", 0, 1.0)
    spec = _make_sampling_spec(median_bins=3, sigma_bins=1.2)

    out = smoothing.smooth(trace, samplingspec=spec)

    assert _roughness(out["I_nA"]) < _roughness(trace["I_nA"])
    assert _roughness(out["V_mV"]) < _roughness(trace["V_mV"])


def test_smoothing_returns_collection() -> None:
    """Smoothed collection should return stacked sampled results."""
    samplings = Samples(
        traces=[
            _make_sampling_trace("a", 0, 1.0),
            _make_sampling_trace("b", 1, 5.0),
        ],
    )
    spec = _make_sampling_spec(median_bins=3, sigma_bins=1.0)

    out = smoothing.smooth(
        samplings,
        samplingspec=spec,
        show_progress=False,
    )

    assert out.specific_keys == ["a", "b"]
    assert np.array_equal(out.indices, np.asarray([0, 1], dtype=np.int64))
    assert np.allclose(out.yvalues, np.asarray([1.0, 5.0]))
    assert out.I_nA.shape == (2, 9)
    assert out.V_mV.shape == (2, 11)
    assert out.dG_G0.shape == (2, 9)
    assert out.dR_R0.shape == (2, 11)


def test_sampling_spec_rejects_even_median_window() -> None:
    """Median smoothing should use a centered odd-sized window."""
    with pytest.raises(ValueError, match="must be odd"):
        _make_sampling_spec(median_bins=4, sigma_bins=1.0)


def test_smoothing_returns_identity_when_filters_are_disabled() -> None:
    """Zero-width smoothing should preserve the sampled trace exactly."""
    trace = _make_sampling_trace("a", 0, 1.0)
    spec = _make_sampling_spec(median_bins=0, sigma_bins=0.0)

    out = smoothing.smooth(trace, samplingspec=spec)

    assert np.allclose(out["I_nA"], trace["I_nA"], equal_nan=True)
    assert np.allclose(out["V_mV"], trace["V_mV"], equal_nan=True)
    assert np.allclose(out["dG_G0"], trace["dG_G0"], equal_nan=True)
    assert np.allclose(out["dR_R0"], trace["dR_R0"], equal_nan=True)
