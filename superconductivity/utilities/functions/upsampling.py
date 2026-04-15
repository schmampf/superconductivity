"""General-purpose axis-aware upsampling helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..meta.axis import axis as make_axis
from ..meta.dataset import Dataset
from ..meta.utils import unwrap_dataset_value, wrap_dataset_value
from ..safety import (
    is_ragged_sequence,
    normalize_axis,
    require_all_finite,
    require_min_size,
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
            axis_spec.code_label,
            values=_upsample_dense(
                axis_spec.values,
                N_up=N_up,
                axis=0,
                method=method,
            ),
            order=axis_spec.order,
        )
    return Dataset(
        code_label=dataset.code_label,
        print_label=dataset.print_label,
        html_label=dataset.html_label,
        latex_label=dataset.latex_label,
        values=data_up,
        axes=tuple(axes),
        params=dataset.params,
    )


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
