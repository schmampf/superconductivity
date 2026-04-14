"""Axis-aware NaN filling helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..meta.dataset import Dataset
from ..meta.utils import unwrap_dataset_value, wrap_dataset_value
from ..safety import is_ragged_sequence, normalize_axis, require_min_size
from ..types import NDArray64


def fill(
    y: np.ndarray | Sequence[np.ndarray] | Dataset,
    axis: int = -1,
    method: str = "interpolate",
    value: float = 0.0,
) -> np.ndarray | list[np.ndarray] | Dataset:
    """Fill NaN values along one axis."""
    if is_ragged_sequence(y):
        return [
            fill(
                item,
                axis=axis,
                method=method,
                value=value,
            )
            for item in y
        ]
    values = unwrap_dataset_value(y)
    return wrap_dataset_value(
        y,
        _fill_dense(np.asarray(values), axis=axis, method=method, value=value),
    )


def _fill_dense(
    y: np.ndarray,
    *,
    axis: int,
    method: str,
    value: float,
) -> np.ndarray:
    if y.ndim == 0:
        raise ValueError("y must have at least one dimension.")

    y_arr = np.asarray(y, dtype=np.float64)
    require_min_size(y_arr, 1, "y")

    axis_norm = normalize_axis(axis, y_arr.ndim)
    y_work = np.moveaxis(y_arr, axis_norm, -1)
    leading_shape = y_work.shape[:-1]
    n = y_work.shape[-1]
    y_flat = y_work.reshape(-1, n)

    method_key = method.strip().lower()
    out_flat = np.empty_like(y_flat)
    for i, row in enumerate(y_flat):
        out_flat[i] = _fill_1d(row, method=method_key, value=value)

    out = out_flat.reshape(*leading_shape, n)
    return np.moveaxis(out, -1, axis_norm)


def _fill_1d(
    y: NDArray64,
    *,
    method: str,
    value: float,
) -> NDArray64:
    if method not in {"interpolate", "nearest", "value"}:
        raise ValueError(
            "Unsupported method. Use 'interpolate', 'nearest', or 'value'."
        )

    finite = np.isfinite(y)
    n_finite = int(np.sum(finite))
    if n_finite == y.size:
        return np.asarray(y, dtype=np.float64)

    if method == "value":
        out = np.asarray(y, dtype=np.float64).copy()
        out[~finite] = float(value)
        return out

    if n_finite == 0:
        return np.asarray(y, dtype=np.float64)

    x = np.arange(y.size, dtype=np.float64)
    x_f = x[finite]
    y_f = y[finite]
    order = np.argsort(x_f)
    x_f = x_f[order]
    y_f = y_f[order]

    if method == "interpolate":
        if n_finite < 2:
            return np.asarray(y, dtype=np.float64)
        out = np.asarray(y, dtype=np.float64).copy()
        out[~finite] = np.interp(x[~finite], x_f, y_f)
        return out

    out = np.asarray(y, dtype=np.float64).copy()
    x_missing = x[~finite]
    pos = np.searchsorted(x_f, x_missing)
    left = np.clip(pos - 1, 0, x_f.size - 1)
    right = np.clip(pos, 0, x_f.size - 1)
    d_left = np.abs(x_missing - x_f[left])
    d_right = np.abs(x_missing - x_f[right])
    choose_left = d_left <= d_right
    idx = np.where(choose_left, left, right)
    out[~finite] = y_f[idx]
    return out


__all__ = ["fill"]
