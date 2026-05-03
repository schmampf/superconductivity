"""Tests for transport dataset smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from superconductivity.utilities.meta import axis, data
from superconductivity.utilities.transport import TransportDatasetSpec
from superconductivity.evaluation.sampling import SamplingSpec


def _make_sampling_spec(**updates: object) -> SamplingSpec:
    defaults: dict[str, object] = {
        "cutoff_Hz": 137.0,
        "sampling_Hz": 137.0,
        "Vbins_mV": np.linspace(-1.0, 1.0, 9),
        "Ibins_nA": np.linspace(-2.0, 2.0, 11),
        "apply_smoothing": True,
    }
    defaults.update(updates)
    return SamplingSpec(**defaults)


def _make_dataset(
    i_nA: np.ndarray,
    *,
    yvalues: np.ndarray | None = None,
) -> TransportDatasetSpec:
    data_entries = (data("I_nA", np.asarray(i_nA, dtype=np.float64)),)
    if yvalues is None:
        return TransportDatasetSpec(
            data=data_entries,
            axes=(axis("V_mV", values=np.linspace(-1.0, 1.0, 9), order=0),),
        )
    return TransportDatasetSpec(
        data=data_entries,
        axes=(
            axis("y", values=np.asarray(yvalues, dtype=np.float64), order=0),
            axis("V_mV", values=np.linspace(-1.0, 1.0, 9), order=1),
        ),
    )


def _roughness(y: np.ndarray) -> float:
    """Estimate curve roughness from first differences on finite support."""
    finite = np.isfinite(y)
    if np.sum(finite) < 3:
        return 0.0
    return float(np.nanstd(np.diff(y[finite])))


def test_smoothing_preserves_nan_edges_and_metadata() -> None:
    """Smoothing should keep metadata and unsupported edges unchanged."""
    trace = _make_dataset(
        np.asarray(
            [np.nan, np.nan, -0.95, -0.05, 0.35, 0.10, 1.05, np.nan, np.nan],
            dtype=np.float64,
        ),
    )
    spec = _make_sampling_spec(median_bins=3, sigma_bins=1.0)

    out = trace.smooth(
        median_bins=spec.median_bins,
        sigma_bins=spec.sigma_bins,
        mode=spec.mode,
    )

    assert np.isnan(out["I_nA"].values[:2]).all()
    assert np.isnan(out["I_nA"].values[-2:]).all()
    assert np.isfinite(out["I_nA"].values[2:-2]).all()


def test_smoothing_reduces_curve_roughness() -> None:
    """The default median->gaussian pipeline should damp roughness."""
    trace = _make_dataset(
        np.asarray(
            [np.nan, np.nan, -0.95, -0.05, 0.35, 0.10, 1.05, np.nan, np.nan],
            dtype=np.float64,
        ),
    )
    spec = _make_sampling_spec(median_bins=3, sigma_bins=1.2)

    out = trace.smooth(
        median_bins=spec.median_bins,
        sigma_bins=spec.sigma_bins,
        mode=spec.mode,
    )

    assert _roughness(np.asarray(out["I_nA"].values, dtype=np.float64)) < _roughness(
        np.asarray(trace["I_nA"].values, dtype=np.float64)
    )


def test_smoothing_returns_collection() -> None:
    """Smoothed collection should return stacked sampled results."""
    sample_a = _make_dataset(
        np.asarray([np.nan, np.nan, -0.95, -0.05, 0.35, 0.10, 1.05, np.nan, np.nan]),
    )
    sample_b = _make_dataset(
        np.asarray([np.nan, np.nan, -0.65, -0.15, 0.25, 0.05, 0.85, np.nan, np.nan]),
    )
    collection = TransportDatasetSpec(
        data=(
            data(
                "I_nA",
                np.vstack(
                    [
                        np.asarray(sample_a["I_nA"].values, dtype=np.float64),
                        np.asarray(sample_b["I_nA"].values, dtype=np.float64),
                    ]
                ),
            ),
        ),
        axes=(
            axis("y", values=np.asarray([1.0, 5.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.linspace(-1.0, 1.0, 9), order=1),
        ),
    )
    spec = _make_sampling_spec(median_bins=3, sigma_bins=1.0)

    out = collection.smooth(
        median_bins=spec.median_bins,
        sigma_bins=spec.sigma_bins,
        mode=spec.mode,
    )

    assert np.allclose(out.axes[0].values, np.asarray([1.0, 5.0]))
    assert out.I_nA.values.shape == (2, 9)


def test_sampling_spec_rejects_even_median_window() -> None:
    """Median smoothing should use a centered odd-sized window."""
    with pytest.raises(ValueError, match="must be odd"):
        _make_sampling_spec(median_bins=4, sigma_bins=1.0)


def test_smoothing_returns_identity_when_filters_are_disabled() -> None:
    """Zero-width smoothing should preserve the sampled trace exactly."""
    trace = _make_dataset(
        np.asarray([np.nan, np.nan, -0.95, -0.05, 0.35, 0.10, 1.05, np.nan, np.nan]),
    )
    spec = _make_sampling_spec(median_bins=0, sigma_bins=0.0)

    out = trace.smooth(
        median_bins=spec.median_bins,
        sigma_bins=spec.sigma_bins,
        mode=spec.mode,
    )

    assert np.allclose(out["I_nA"].values, trace["I_nA"].values, equal_nan=True)
