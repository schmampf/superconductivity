from __future__ import annotations

import numpy as np
import pytest

from superconductivity.utilities.functions.binning import bin
from superconductivity.utilities.meta.axis import axis
from superconductivity.utilities.meta.dataset import Dataset, dataset
from superconductivity.utilities.legacy.functions import bin_y_over_x


def test_bin_1d_matches_legacy_helper() -> None:
    x = np.array([-0.9, -0.7, -0.1, 0.2, 0.8], dtype=np.float64)
    z = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    xbins = np.linspace(-1.0, 1.0, 5, dtype=np.float64)

    out = bin(z, x, xbins)
    ref = bin_y_over_x(x, z, xbins)

    np.testing.assert_allclose(out, ref, equal_nan=True)


def test_bin_2d_axis_minus1_with_shared_1d_x() -> None:
    z = np.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
        dtype=np.float64,
    )
    x = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float64)
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins, axis=-1)

    expected = np.array([[1.5, 3.5], [15.0, 35.0]], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_bin_2d_axis0_with_shared_1d_x() -> None:
    z = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=np.float64,
    )
    x = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float64)
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins, axis=0)

    expected = np.array([[1.5, 15.0], [3.5, 35.0]], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_bin_same_shape_x_works_elementwise() -> None:
    z = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64)
    x = np.array([[-0.8, -0.1, 0.9], [-0.2, 0.2, 0.8]], dtype=np.float64)
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins, axis=1)

    expected = np.array([[1.5, 3.0], [10.0, 25.0]], dtype=np.float64)
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_bin_3d_negative_axis_replaces_selected_axis() -> None:
    z = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    x = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float64)
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins, axis=-1)

    assert out.shape == (2, 3, 2)
    np.testing.assert_allclose(out[..., 0], 0.5 * (z[..., 0] + z[..., 1]))
    np.testing.assert_allclose(out[..., 1], 0.5 * (z[..., 2] + z[..., 3]))


def test_bin_ragged_returns_list_of_arrays() -> None:
    z = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
    ]
    x = [
        np.array([-0.8, 0.0, 0.8], dtype=np.float64),
        np.array([-0.9, -0.1, 0.1, 0.9], dtype=np.float64),
    ]
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins)

    assert isinstance(out, list)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], np.array([1.0, 2.5]), equal_nan=True)
    np.testing.assert_allclose(out[1], np.array([15.0, 35.0]), equal_nan=True)


def test_bin_ragged_accepts_shared_1d_x_when_lengths_match() -> None:
    z = [
        np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64),
        np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
    ]
    x = np.array([-0.8, 0.0, 0.8], dtype=np.float64)
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins, axis=1)

    assert isinstance(out, list)
    np.testing.assert_allclose(out[0], np.array([[1.0, 2.5], [10.0, 25.0]]))
    np.testing.assert_allclose(out[1], np.array([[4.0, 5.5]]))


def test_bin_1d_z_with_2d_x_rebins_per_column() -> None:
    z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = np.array(
        [
            [-0.8, -0.2],
            [0.0, 0.2],
            [0.8, 0.9],
        ],
        dtype=np.float64,
    )
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins, axis=0)

    expected = np.array([[1.0, 1.0], [2.5, 2.5]], dtype=np.float64)
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_bin_1d_z_with_nd_x_uses_selected_axis() -> None:
    z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = np.array(
        [
            [[-0.8, 0.0, 0.8]],
            [[-0.6, 0.2, 0.9]],
        ],
        dtype=np.float64,
    )
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins, axis=-1)

    expected = np.array(
        [
            [[1.0, 2.5]],
            [[1.0, 2.5]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_bin_ragged_1d_z_with_nd_x_returns_list() -> None:
    z = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([4.0, 5.0], dtype=np.float64),
    ]
    x = [
        np.array([[-0.8, 0.0, 0.8]], dtype=np.float64),
        np.array([[-0.2, 0.7]], dtype=np.float64),
    ]
    xbins = np.array([-0.5, 0.5], dtype=np.float64)

    out = bin(z, x, xbins, axis=-1)

    assert isinstance(out, list)
    np.testing.assert_allclose(out[0], np.array([[1.0, 2.5]]), equal_nan=True)
    np.testing.assert_allclose(out[1], np.array([[4.0, 5.0]]), equal_nan=True)


def test_bin_empty_bins_are_nan() -> None:
    z = np.array([1.0, 2.0], dtype=np.float64)
    x = np.array([-0.5, 0.5], dtype=np.float64)
    xbins = np.array([-0.5, 0.0, 0.5], dtype=np.float64)

    out = bin(z, x, xbins)

    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], np.array([1.0, 2.0]))


def test_bin_rejects_invalid_axis() -> None:
    z = np.ones((2, 3), dtype=np.float64)
    x = np.arange(3, dtype=np.float64)
    xbins = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="axis"):
        bin(z, x, xbins, axis=2)


def test_bin_rejects_dense_shape_mismatch() -> None:
    z = np.ones((2, 3), dtype=np.float64)
    x = np.ones((2, 2), dtype=np.float64)
    xbins = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="same shape"):
        bin(z, x, xbins, axis=1)


def test_bin_rejects_non_1d_xbins() -> None:
    z = np.ones(3, dtype=np.float64)
    x = np.arange(3, dtype=np.float64)
    xbins = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        bin(z, x, xbins)


def test_bin_accepts_axis_metadata_without_explicit_axis() -> None:
    z = dataset("I_nA", [1.0, 2.0, 3.0, 4.0])
    x = axis("V_mV", values=[-0.75, -0.25, 0.25, 0.75], order=0)
    xbins = axis("V_mV", values=[-0.5, 0.5], order=0)

    out = bin(z, x, xbins)

    expected = np.array([1.5, 3.5], dtype=np.float64)
    np.testing.assert_allclose(out.values, expected)
    assert isinstance(out, type(z))


def test_bin_accepts_dataset_z_and_axis_inputs() -> None:
    z = dataset(
        "trace",
        [1.0, 2.0, 3.0, 4.0],
        axes=axis("V_mV", values=[0.0, 1.0, 2.0, 3.0]),
    )
    x = axis("V_mV", values=[-0.75, -0.25, 0.25, 0.75], order=0)
    xbins = axis("V_mV", values=[-0.5, 0.5], order=0)

    out = bin(z, x, xbins)

    expected = np.array([1.5, 3.5], dtype=np.float64)
    np.testing.assert_allclose(out.values, expected)
    assert isinstance(out, type(z))
    assert out.axes == (xbins,)


def test_bin_rejects_incompatible_ragged_x() -> None:
    z = [np.ones(3, dtype=np.float64), np.ones(4, dtype=np.float64)]
    x = np.arange(3, dtype=np.float64)
    xbins = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Shared x"):
        bin(z, x, xbins)


def test_bin_rejects_nd_x_that_does_not_match_1d_z_length() -> None:
    z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = np.ones((2, 4), dtype=np.float64)
    xbins = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="1D z"):
        bin(z, x, xbins, axis=1)
