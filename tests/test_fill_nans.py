from __future__ import annotations

import numpy as np
import pytest

from superconductivity.utilities.functions.fill_nans import fill


def test_fill_1d_interpolate_matches_linear_gap() -> None:
    y = np.array([0.0, np.nan, 2.0, np.nan, 4.0], dtype=np.float64)

    out = fill(y, method="interpolate")

    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_fill_1d_nearest_uses_closest_neighbors() -> None:
    y = np.array([0.0, np.nan, np.nan, 3.0], dtype=np.float64)

    out = fill(y, method="nearest")

    expected = np.array([0.0, 0.0, 3.0, 3.0], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_fill_2d_axis0() -> None:
    y = np.array(
        [[0.0, np.nan, 2.0], [1.0, 3.0, np.nan], [2.0, 5.0, 6.0]],
        dtype=np.float64,
    )

    out = fill(y, axis=0, method="interpolate")

    expected = np.array(
        [[0.0, 3.0, 2.0], [1.0, 3.0, 4.0], [2.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_fill_value_replaces_nans_with_constant() -> None:
    y = np.array([1.0, np.nan, 3.0], dtype=np.float64)

    out = fill(y, method="value", value=-1.5)

    expected = np.array([1.0, -1.5, 3.0], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_fill_ragged_returns_list() -> None:
    y = [
        np.array([0.0, np.nan, 2.0], dtype=np.float64),
        np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float64),
    ]

    out = fill(y, axis=-1, method="nearest")

    assert isinstance(out, list)
    np.testing.assert_allclose(out[0], np.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(
        out[1],
        np.array([[1.0, 1.0], [4.0, 4.0]], dtype=np.float64),
    )


def test_fill_invalid_method_raises() -> None:
    y = np.array([0.0, np.nan], dtype=np.float64)

    with pytest.raises(ValueError, match="Unsupported method"):
        fill(y, method="bad")

