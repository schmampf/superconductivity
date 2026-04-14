"""General-purpose axis-aware upsampling helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..meta.axis import axis as make_axis
from ..meta.dataset import Dataset
from ..meta.utils import unwrap_dataset_value, wrap_dataset_value
from ..safety import (
    is_pair_input,
    is_ragged_sequence,
    normalize_axis,
    require_all_finite,
    require_min_size,
    require_same_shape,
)


def upsample(
    z: np.ndarray | Sequence[np.ndarray] | Dataset,
    N_up: int = 100,
    axis: int = -1,
    method: str = "linear",
) -> np.ndarray | list[np.ndarray] | Dataset:
    """Upsample arrays along one axis by sample-index interpolation."""
    if isinstance(z, Dataset):
        return _upsample_dataset(
            z,
            N_up=N_up,
            axis=axis,
            method=method,
        )
    if is_pair_input(z):
        return _upsample_pair(
            z,
            N_up=N_up,
            axis=axis,
            method=method,
        )
    if is_ragged_sequence(z):
        return [
            wrap_dataset_value(
                item,
                _upsample_dense(
                    unwrap_dataset_value(item),
                    N_up=N_up,
                    axis=axis,
                    method=method,
                ),
            )
            for item in z
        ]
    return wrap_dataset_value(
        z,
        _upsample_dense(
            unwrap_dataset_value(z),
            N_up=N_up,
            axis=axis,
            method=method,
        ),
    )


def _upsample_dataset(
    dataset: Dataset, *, N_up: int, axis: int, method: str
) -> Dataset:
    data_up = _upsample_dense(
        dataset.values,
        N_up=N_up,
        axis=axis,
        method=method,
    )
    axis_norm = normalize_axis(axis, np.asarray(dataset.values).ndim)
    axes = list(dataset.axes)
    if len(axes) == 1:
        idx = 0
    else:
        matches = [i for i, ax in enumerate(axes) if ax.order == axis_norm]
        if len(matches) != 1:
            raise ValueError(
                "Dataset upsampling requires exactly one matching axis order."
            )
        idx = matches[0]
    if axes:
        axis_spec = axes[idx]
        axes[idx] = make_axis(
            axis_spec.label,
            values=_upsample_dense(
                axis_spec.values,
                N_up=N_up,
                axis=0,
                method=method,
            ),
            order=axis_spec.order,
        )
    return Dataset(
        label=dataset.label,
        html_label=dataset.html_label,
        latex_label=dataset.latex_label,
        values=data_up,
        axes=tuple(axes),
        params=dataset.params,
    )


def _upsample_pair(
    pair: tuple[np.ndarray, np.ndarray],
    *,
    N_up: int,
    axis: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pair) != 2:
        raise ValueError("paired upsample input must be a tuple of (x, z).")

    x_raw, z_raw = pair
    x_arr = np.asarray(unwrap_dataset_value(x_raw), dtype=np.float64)
    z_arr = np.asarray(unwrap_dataset_value(z_raw), dtype=np.float64)
    if x_arr.ndim == 0 or z_arr.ndim == 0:
        raise ValueError("x and z must have at least one dimension.")

    factor = int(N_up)
    if factor <= 1:
        return np.asarray(x_arr), wrap_dataset_value(z_raw, np.asarray(z_arr))

    require_min_size(z_arr, 1, "z")
    require_all_finite(z_arr, "z")
    require_min_size(x_arr, 1, "x")
    require_all_finite(x_arr, "x")

    if x_arr.ndim == z_arr.ndim:
        axis_norm = normalize_axis(axis, z_arr.ndim)
        require_same_shape(x_arr, z_arr, "x", "z")
        return (
            _upsample_dense(
                x_arr,
                N_up=factor,
                axis=axis_norm,
                method=method,
            ),
            wrap_dataset_value(
                z_raw,
                _upsample_dense(
                    z_arr,
                    N_up=factor,
                    axis=axis_norm,
                    method=method,
                ),
            ),
        )

    if x_arr.ndim != 1:
        raise ValueError(
            "For paired upsampling, x must be 1D or have the same shape as z."
        )

    axis_norm = normalize_axis(axis, z_arr.ndim)
    if x_arr.size != z_arr.shape[axis_norm]:
        raise ValueError("1D x must match z along the selected axis.")

    x_up = _upsample_dense(
        x_arr,
        N_up=factor,
        axis=0,
        method=method,
    )
    z_up = wrap_dataset_value(
        z_raw,
        _upsample_dense(
            z_arr,
            N_up=factor,
            axis=axis_norm,
            method=method,
        ),
    )
    return x_up, z_up


def _upsample_pair(
    pair: tuple[np.ndarray, np.ndarray],
    *,
    N_up: int,
    axis: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pair) != 2:
        raise ValueError("paired upsample input must be a tuple of (x, z).")

    x_raw, z_raw = pair
    x_arr = np.asarray(unwrap_dataset_value(x_raw), dtype=np.float64)
    z_arr = np.asarray(unwrap_dataset_value(z_raw), dtype=np.float64)
    if x_arr.ndim == 0 or z_arr.ndim == 0:
        raise ValueError("x and z must have at least one dimension.")

    factor = int(N_up)
    if factor <= 1:
        return np.asarray(x_arr), wrap_dataset_value(z_raw, np.asarray(z_arr))

    require_min_size(z_arr, 1, "z")
    require_all_finite(z_arr, "z")
    require_min_size(x_arr, 1, "x")
    require_all_finite(x_arr, "x")

    if x_arr.ndim == z_arr.ndim:
        axis_norm = normalize_axis(axis, z_arr.ndim)
        require_same_shape(x_arr, z_arr, "x", "z")
        return (
            _upsample_dense(
                x_arr,
                N_up=factor,
                axis=axis_norm,
                method=method,
            ),
            wrap_dataset_value(
                z_raw,
                _upsample_dense(
                    z_arr,
                    N_up=factor,
                    axis=axis_norm,
                    method=method,
                ),
            ),
        )

    if x_arr.ndim != 1:
        raise ValueError(
            "For paired upsampling, x must be 1D or have the same shape as z."
        )

    axis_norm = normalize_axis(axis, z_arr.ndim)
    if x_arr.size != z_arr.shape[axis_norm]:
        raise ValueError("1D x must match z along the selected axis.")

    x_up = _upsample_dense(
        x_arr,
        N_up=factor,
        axis=0,
        method=method,
    )
    z_up = wrap_dataset_value(
        z_raw,
        _upsample_dense(
            z_arr,
            N_up=factor,
            axis=axis_norm,
            method=method,
        ),
    )
    return x_up, z_up


def _upsample_dense(
    z: np.ndarray,
    *,
    N_up: int,
    axis: int,
    method: str,
) -> np.ndarray:
    if z.ndim == 0:
        raise ValueError("z must have at least one dimension.")

    factor = int(N_up)
    if factor <= 1:
        return np.asarray(z)

    axis_norm = normalize_axis(axis, z.ndim)
    z_arr = np.asarray(z, dtype=np.float64)
    require_min_size(z_arr, 1, "z")
    require_all_finite(z_arr, "z")

    z_work = np.moveaxis(z_arr, axis_norm, -1)
    n = z_work.shape[-1]
    if n < 2:
        return np.asarray(z_arr)

    t = np.arange(n, dtype=np.float64)
    t_new = np.linspace(0.0, float(n - 1), n * factor, dtype=np.float64)
    z_flat = z_work.reshape(-1, n)

    method_key = method.strip().lower()
    if method_key == "linear":
        out_flat = np.empty((z_flat.shape[0], t_new.size), dtype=np.float64)
        for i, row in enumerate(z_flat):
            out_flat[i] = np.interp(t_new, t, row)
    elif method_key == "nearest":
        idx = np.rint(t_new).astype(np.int64)
        idx = np.clip(idx, 0, n - 1)
        out_flat = z_flat[:, idx]
    else:
        raise ValueError("Unsupported method. Use 'linear' or 'nearest'.")

    out = out_flat.reshape(*z_work.shape[:-1], t_new.size)
    return np.moveaxis(out, -1, axis_norm)


__all__ = ["upsample"]
