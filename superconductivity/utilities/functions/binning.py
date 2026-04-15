"""General-purpose mean binning helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..meta.axis import AxisSpec
from ..meta.dataset import Dataset
from ..meta.utils import infer_axis, unwrap_dataset_value
from ..safety import (
    is_ragged_sequence,
    normalize_axis,
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)


def bin(
    z: np.ndarray | Sequence[np.ndarray] | Dataset,
    x: np.ndarray | Sequence[np.ndarray] | AxisSpec,
    xbins: np.ndarray | AxisSpec,
    axis: int | None = None,
) -> np.ndarray | list[np.ndarray] | Dataset:
    """Bin ``z`` over ``x`` onto the 1D grid ``xbins``."""
    z_arr = unwrap_dataset_value(z)
    x_arr = unwrap_dataset_value(x)
    xbins_arr = _validate_xbins(unwrap_dataset_value(xbins))
    axis_inferred = infer_axis(axis, x)

    if is_ragged_sequence(z_arr):
        result = _bin_ragged(z_arr, x_arr, xbins_arr, axis_inferred)
    else:
        if is_ragged_sequence(x_arr):
            raise ValueError(
                "x may only be a sequence when z is also a sequence.",
            )
        result = _bin_dense(
            np.asarray(z_arr), np.asarray(x_arr), xbins_arr, axis_inferred
        )
    if isinstance(z, Dataset):
        axes = (xbins,) if isinstance(xbins, AxisSpec) else ()
        return Dataset(
            code_label=z.code_label,
            print_label=z.print_label,
            html_label=z.html_label,
            latex_label=z.latex_label,
            values=result,
            axes=axes,
            params=z.params,
        )
    return result


def _bin_ragged(
    z_seq: Sequence[np.ndarray],
    x: np.ndarray | Sequence[np.ndarray],
    xbins: np.ndarray,
    axis: int,
) -> list[np.ndarray]:
    if len(z_seq) == 0:
        raise ValueError("z must not be empty.")

    if is_ragged_sequence(x):
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
                (
                    "Shared x must match the selected axis"
                    "length of every ragged z item."
                )
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

    axis_norm = normalize_axis(axis, max(z_arr.ndim, x_arr.ndim))
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
        axis_z = normalize_axis(axis, z.ndim)
        if x_arr.size != z.shape[axis_z]:
            raise ValueError("1D x must match z along the selected axis.")
        reshape = [1] * z.ndim
        reshape[axis_z] = x_arr.size
        x_broadcast = np.broadcast_to(x_arr.reshape(reshape), z.shape)
        return x_broadcast, z

    if z.ndim == 1:
        x_arr = np.asarray(x, dtype=np.float64)
        require_all_finite(x_arr, "x")
        axis_x = normalize_axis(axis, x_arr.ndim)
        if x_arr.shape[axis_x] != z.size:
            raise ValueError(
                "ND x must match the length of 1D z along the selected axis."
            )
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
    xbins_arr = np.asarray(xbins, dtype=np.float64)
    if xbins_arr.ndim != 1:
        raise ValueError("xbins must be 1D.")
    require_min_size(xbins_arr, 2, "xbins")
    require_all_finite(xbins_arr, "xbins")
    if np.any(np.diff(xbins_arr) <= 0.0):
        raise ValueError("xbins must be strictly increasing.")
    return xbins_arr


__all__ = ["bin"]
