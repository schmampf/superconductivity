from __future__ import annotations

import numpy as np
import pytest

from superconductivity.utilities.functions.upsampling import upsample


def test_upsample_1d_linear_matches_expected_grid() -> None:
    z = np.array([0.0, 10.0, 20.0], dtype=np.float64)

    out = upsample(z, N_up=2)

    expected = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_upsample_2d_last_axis() -> None:
    z = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float64)

    out = upsample(z, N_up=3, axis=-1)

    assert out.shape == (2, 6)
    np.testing.assert_allclose(out[:, 0], z[:, 0])
    np.testing.assert_allclose(out[:, -1], z[:, -1])


def test_upsample_2d_first_axis() -> None:
    z = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float64)

    out = upsample(z, N_up=2, axis=0)

    expected = np.array(
        [
            [0.0, 10.0],
            [6.66666667, 16.66666667],
            [13.33333333, 23.33333333],
            [20.0, 30.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected)


def test_upsample_negative_axis_on_3d_array() -> None:
    z = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)

    out = upsample(z, N_up=2, axis=-2)

    assert out.shape == (2, 6, 2)
    np.testing.assert_allclose(out[:, 0, :], z[:, 0, :])
    np.testing.assert_allclose(out[:, -1, :], z[:, -1, :])


def test_upsample_nearest_repeats_nearest_samples() -> None:
    z = np.array([1.0, 5.0, 9.0], dtype=np.float64)

    out = upsample(z, N_up=2, method="nearest")

    expected = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_upsample_ragged_returns_list() -> None:
    z = [
        np.array([0.0, 10.0], dtype=np.float64),
        np.array([[1.0, 3.0, 5.0]], dtype=np.float64),
    ]

    out = upsample(z, N_up=2, axis=-1)

    assert isinstance(out, list)
    np.testing.assert_allclose(out[0], np.array([0.0, 3.33333333, 6.66666667, 10.0]))
    np.testing.assert_allclose(
        out[1],
        np.array([[1.0, 1.8, 2.6, 3.4, 4.2, 5.0]], dtype=np.float64),
    )


def test_upsample_tuple_input_becomes_ragged_sequence() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    z = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    out = upsample((x, z), N_up=2)

    assert isinstance(out, list)
    np.testing.assert_allclose(
        out[0],
        np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        out[1],
        np.array([10.0, 14.0, 18.0, 22.0, 26.0, 30.0], dtype=np.float64),
    )


def test_upsample_tuple_input_with_2d_items_is_ragged() -> None:
    x = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    z = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

    out = upsample((x, z), N_up=2, axis=1)

    assert isinstance(out, list)
    assert out[0].shape == (2, 4)
    assert out[1].shape == (2, 4)
    np.testing.assert_allclose(out[0][:, 0], x[:, 0])
    np.testing.assert_allclose(out[1][:, -1], z[:, -1])


def test_upsample_tuple_input_with_same_shape_items_is_ragged() -> None:
    x = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    z = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

    out = upsample((x, z), N_up=2, axis=1)

    assert isinstance(out, list)
    assert out[0].shape == (2, 4)
    assert out[1].shape == (2, 4)
    np.testing.assert_allclose(out[0][:, 0], x[:, 0])
    np.testing.assert_allclose(out[1][:, -1], z[:, -1])


def test_upsample_factor_one_returns_input_shape() -> None:
    z = np.arange(6, dtype=np.float64).reshape(2, 3)

    out = upsample(z, N_up=1, axis=1)

    assert out.shape == z.shape
    np.testing.assert_allclose(out, z)


def test_upsample_invalid_axis_raises() -> None:
    z = np.ones((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="axis"):
        upsample(z, N_up=2, axis=2)


def test_upsample_invalid_method_raises() -> None:
    z = np.ones(3, dtype=np.float64)

    with pytest.raises(ValueError, match="Unsupported method"):
        upsample(z, N_up=2, method="cubic")

