"""General-purpose mean binning helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)


def bin(
    z: np.ndarray | Sequence[np.ndarray],
    x: np.ndarray | Sequence[np.ndarray],
    xbins: np.ndarray,
    axis: int = -1,
) -> np.ndarray | list[np.ndarray]:
    """Bin ``z`` over ``x`` onto the 1D grid ``xbins``.

    Parameters
    ----------
    z
        Values to bin. Can be a dense NumPy array or a list/tuple of arrays.
    x
        Coordinate values associated with ``z``. Must either match ``z`` in
        shape, be 1D along the selected axis, or be a list/tuple matching a
        ragged ``z`` input.
    xbins
        1D bin centers defining the output grid.
    axis
        Axis used for binning, similar to NumPy axis arguments.

    Returns
    -------
    np.ndarray | list[np.ndarray]
        Mean values in each bin. Empty bins are returned as ``NaN``.
    """
    xbins_arr = _validate_xbins(xbins)

    if _is_ragged_sequence(z):
        return _bin_ragged(z, x, xbins_arr, axis)
    if _is_ragged_sequence(x):
        raise ValueError("x may only be a sequence when z is also a sequence.")
    return _bin_dense(np.asarray(z), np.asarray(x), xbins_arr, axis)


def _bin_ragged(
    z_seq: Sequence[np.ndarray],
    x: np.ndarray | Sequence[np.ndarray],
    xbins: np.ndarray,
    axis: int,
) -> list[np.ndarray]:
    if len(z_seq) == 0:
        raise ValueError("z must not be empty.")

    if _is_ragged_sequence(x):
        x_seq = list(x)
        if len(x_seq) != len(z_seq):
            raise ValueError("x and z must contain the same number of items.")
        return [
            _bin_dense(np.asarray(z_i), np.asarray(x_i), xbins, axis)
            for z_i, x_i in zip(z_seq, x_seq)
        ]

    x_arr = np.asarray(x)
    out: list[np.ndarray] = []
    for z_i in z_seq:
        z_arr = np.asarray(z_i)
        try:
            out.append(_bin_dense(z_arr, x_arr, xbins, axis))
        except ValueError as exc:
            raise ValueError(
                "Shared x must match the selected axis length of every ragged z item."
            ) from exc
    return out


def _bin_dense(
    z: np.ndarray,
    x: np.ndarray,
    xbins: np.ndarray,
    axis: int,
) -> np.ndarray:
    if z.ndim == 0:
        raise ValueError("z must have at least one dimension.")

    z_arr = np.asarray(z, dtype=np.float64)
    require_min_size(z_arr, 1, "z")
    require_all_finite(z_arr, "z")

    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        raise ValueError("x must have at least one dimension.")

    axis_norm = _normalize_axis(axis, max(z_arr.ndim, x_arr.ndim))
    x_work, z_work = _prepare_dense_pair(z_arr, x_arr, axis_norm)
    z_work = np.moveaxis(z_work, axis_norm, -1)
    x_work = np.moveaxis(x_work, axis_norm, -1)

    leading_shape = z_work.shape[:-1]
    n_bins = xbins.size
    z_flat = z_work.reshape(-1, z_work.shape[-1])
    x_flat = x_work.reshape(-1, x_work.shape[-1])

    out = np.empty((z_flat.shape[0], n_bins), dtype=np.float64)
    for i, (x_row, z_row) in enumerate(zip(x_flat, z_flat)):
        out[i] = _bin_1d(z_row, x_row, xbins)

    out = out.reshape(*leading_shape, n_bins)
    return np.moveaxis(out, -1, axis_norm)


def _prepare_dense_pair(
    z: np.ndarray,
    x: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim == z.ndim and x.shape == z.shape:
        x_arr = np.asarray(x, dtype=np.float64)
        require_all_finite(x_arr, "x")
        return x_arr, z

    if x.ndim == 1:
        x_arr = to_1d_float64(x, "x")
        axis_z = _normalize_axis(axis, z.ndim)
        if x_arr.size != z.shape[axis_z]:
            raise ValueError("1D x must match z along the selected axis.")
        reshape = [1] * z.ndim
        reshape[axis_z] = x_arr.size
        x_broadcast = np.broadcast_to(x_arr.reshape(reshape), z.shape)
        return x_broadcast, z

    if z.ndim == 1:
        x_arr = np.asarray(x, dtype=np.float64)
        require_all_finite(x_arr, "x")
        axis_x = _normalize_axis(axis, x_arr.ndim)
        if x_arr.shape[axis_x] != z.size:
            raise ValueError("ND x must match the length of 1D z along the selected axis.")
        reshape = [1] * x_arr.ndim
        reshape[axis_x] = z.size
        z_broadcast = np.broadcast_to(z.reshape(reshape), x_arr.shape)
        return x_arr, z_broadcast

    x_arr = np.asarray(x, dtype=np.float64)
    require_same_shape(x_arr, z, "x", "z")
    require_all_finite(x_arr, "x")
    return x_arr, z


def _bin_1d(z: np.ndarray, x: np.ndarray, xbins: np.ndarray) -> np.ndarray:
    x_arr = to_1d_float64(x, "x")
    z_arr = to_1d_float64(z, "z")
    require_same_shape(x_arr, z_arr, "x", "z")
    require_all_finite(x_arr, "x")
    require_all_finite(z_arr, "z")

    edges = _bin_edges_from_centers(xbins)
    counts, _ = np.histogram(x_arr, bins=edges)
    sums, _ = np.histogram(x_arr, bins=edges, weights=z_arr)

    out = np.full(xbins.shape, np.nan, dtype=np.float64)
    nonzero = counts > 0
    out[nonzero] = sums[nonzero] / counts[nonzero]
    return out


def _bin_edges_from_centers(xbins: np.ndarray) -> np.ndarray:
    x_nu = np.append(xbins, 2.0 * xbins[-1] - xbins[-2])
    return x_nu - (x_nu[1] - x_nu[0]) / 2.0


def _validate_xbins(xbins: np.ndarray) -> np.ndarray:
    xbins_arr = to_1d_float64(xbins, "xbins")
    require_min_size(xbins_arr, 2, "xbins")
    require_all_finite(xbins_arr, "xbins")
    if np.any(np.diff(xbins_arr) <= 0.0):
        raise ValueError("xbins must be strictly increasing.")
    return xbins_arr


def _normalize_axis(axis: int, ndim: int) -> int:
    axis_int = int(axis)
    if axis_int < -ndim or axis_int >= ndim:
        raise ValueError(f"axis {axis_int} is out of bounds for array of ndim {ndim}.")
    return axis_int % ndim


def _is_ragged_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


__all__ = ["bin"]
