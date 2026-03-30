"""Tests for sampled-IV smoothing containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.smoothing as smoothing
from superconductivity.evaluation.sampling import (
    SamplingSpec,
    SamplingTrace,
    SamplingTraces,
)


def _make_sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        upsample=4,
        Vbin_mV=np.linspace(-1.0, 1.0, 9),
        Ibin_nA=np.linspace(-2.0, 2.0, 11),
    )


def _make_sampling_trace(
    specific_key: str,
    index: int,
    yvalue: float,
) -> SamplingTrace:
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
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "Voff_mV": 0.1,
        "Ioff_nA": -0.2,
        "Vbin_mV": vbin_mV,
        "Ibin_nA": ibin_nA,
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


def test_get_smoothed_sampling_preserves_nan_edges_and_metadata() -> None:
    """Smoothing should keep metadata and unsupported edges unchanged."""
    trace = _make_sampling_trace("a", 0, 1.0)
    spec = smoothing.SmoothingSpec(
        smooth=True,
        median_bins=3,
        sigma_bins=1.0,
    )

    out = smoothing.get_smoothed_sampling(trace, spec)

    assert out["specific_key"] == "a"
    assert out["index"] == 0
    assert out["yvalue"] == pytest.approx(1.0)
    assert out["Voff_mV"] == pytest.approx(0.1)
    assert out["Ioff_nA"] == pytest.approx(-0.2)
    assert np.allclose(out["Vbin_mV"], trace["Vbin_mV"])
    assert np.allclose(out["Ibin_nA"], trace["Ibin_nA"])
    assert np.isnan(out["I_nA"][:2]).all()
    assert np.isnan(out["I_nA"][-2:]).all()
    assert np.isnan(out["V_mV"][:2]).all()
    assert np.isnan(out["V_mV"][-2:]).all()
    assert np.isfinite(out["I_nA"][2:-2]).all()
    assert np.isfinite(out["V_mV"][2:-2]).all()


def test_get_smoothed_sampling_reduces_curve_roughness() -> None:
    """The default median->gaussian pipeline should damp roughness."""
    trace = _make_sampling_trace("a", 0, 1.0)
    spec = smoothing.SmoothingSpec(
        smooth=True,
        median_bins=3,
        sigma_bins=1.2,
    )

    out = smoothing.get_smoothed_sampling(trace, spec)

    assert _roughness(out["I_nA"]) < _roughness(trace["I_nA"])
    assert _roughness(out["V_mV"]) < _roughness(trace["V_mV"])


def test_get_smoothed_samplings_returns_collection_with_lookup_methods() -> None:
    """Smoothed collection should mirror the other container APIs."""
    sampling_spec = _make_sampling_spec()
    samplings = SamplingTraces(
        spec=sampling_spec,
        traces=[
            _make_sampling_trace("a", 0, 1.0),
            _make_sampling_trace("b", 1, 5.0),
        ],
    )
    smooth_spec = smoothing.SmoothingSpec(
        smooth=True,
        median_bins=3,
        sigma_bins=1.0,
    )

    out = smoothing.get_smoothed_samplings(
        samplings=samplings,
        spec=smooth_spec,
        show_progress=False,
    )

    assert out.sampling_spec is sampling_spec
    assert out.smoothing_spec is smooth_spec
    assert out.keys == ["a", "b"]
    assert np.allclose(out.yvalues, np.asarray([1.0, 5.0]))
    assert np.allclose(out.Voff_mV, np.asarray([0.1, 0.1]))
    assert np.allclose(out.Ioff_nA, np.asarray([-0.2, -0.2]))
    assert out.I_nA.shape == (2, 9)
    assert out.V_mV.shape == (2, 11)
    assert out.dG_G0.shape == (2, 9)
    assert out.dR_R0.shape == (2, 11)
    assert out.by_key("a")["specific_key"] == "a"
    assert out.by_value(5.0)["specific_key"] == "b"


def test_smoothing_spec_rejects_even_median_window() -> None:
    """Median smoothing should use a centered odd-sized window."""
    with pytest.raises(ValueError, match="must be odd"):
        smoothing.SmoothingSpec(
            smooth=True,
            median_bins=4,
            sigma_bins=1.0,
        )


def test_get_smoothed_sampling_returns_identity_when_disabled() -> None:
    """Disabled smoothing should preserve the sampled trace exactly."""
    trace = _make_sampling_trace("a", 0, 1.0)
    spec = smoothing.SmoothingSpec(
        smooth=False,
        median_bins=3,
        sigma_bins=1.2,
    )

    out = smoothing.get_smoothed_sampling(trace, spec)

    assert np.allclose(out["I_nA"], trace["I_nA"], equal_nan=True)
    assert np.allclose(out["V_mV"], trace["V_mV"], equal_nan=True)
    assert np.allclose(out["dG_G0"], trace["dG_G0"], equal_nan=True)
    assert np.allclose(out["dR_R0"], trace["dR_R0"], equal_nan=True)
