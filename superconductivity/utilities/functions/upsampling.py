"""General-purpose axis-aware upsampling helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..safety import (
    is_ragged_sequence,
    normalize_axis,
    require_all_finite,
    require_min_size,
)


def upsample(
    z: np.ndarray | Sequence[np.ndarray],
    N_up: int = 100,
    axis: int = -1,
    method: str = "linear",
) -> np.ndarray | list[np.ndarray]:
    """Upsample arrays along one axis by sample-index interpolation."""
    if is_ragged_sequence(z):
        return [
            _upsample_dense(
                np.asarray(item),
                N_up=N_up,
                axis=axis,
                method=method,
            )
            for item in z
        ]
    return _upsample_dense(
        np.asarray(z),
        N_up=N_up,
        axis=axis,
        method=method,
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
