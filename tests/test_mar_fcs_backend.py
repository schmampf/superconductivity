"""Focused smoke tests for the MAR FCS backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from superconductivity.models.mar.backend.carlosfcs import fcs_curve


def _fcs_available() -> bool:
    """Return whether the compiled FCS backend is importable."""
    return callable(fcs_curve)


@pytest.mark.skipif(not _fcs_available(), reason="compiled FCS backend missing")
def test_fcs_curve_returns_finite_currents() -> None:
    """The FCS backend returns finite total and charge-resolved currents."""
    currents = np.array(
        fcs_curve(
            0.5,
            0.0,
            0.18,
            0.16,
            1e-4,
            1e-4,
            np.array([0.1, 0.2], dtype=float),
            2,
            64,
            1,
        ),
        dtype=float,
    )

    assert currents.shape == (2, 3)
    assert np.isfinite(currents).all()


@pytest.mark.skipif(not _fcs_available(), reason="compiled FCS backend missing")
def test_fcs_curve_is_safe_under_threaded_calls() -> None:
    """Concurrent backend calls complete without non-finite results."""

    def worker(seed: int) -> np.ndarray:
        voltages = np.array(
            [
                0.05 + 0.01 * ((seed + 1) % 5),
                0.10 + 0.02 * ((seed + 2) % 3),
            ],
            dtype=float,
        )
        return np.array(
            fcs_curve(
                0.5,
                0.0,
                0.18,
                0.16,
                1e-4,
                1e-4,
                voltages,
                2,
                64,
                1,
            ),
            dtype=float,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(8)))

    stacked = np.stack(results)
    assert np.isfinite(stacked).all()
