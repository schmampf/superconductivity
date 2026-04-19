"""Shared helpers for metadata-aware containers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .axis import AxisSpec
from .data import DataSpec


def unwrap_dataset_value(value: np.ndarray | DataSpec | AxisSpec) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        return value  # preserve ragged sequences for callers that handle them
    if isinstance(value, DataSpec):
        return value.values
    if isinstance(value, AxisSpec):
        return value.values
    return np.asarray(value)


def infer_axis(
    axis: int | None, x: np.ndarray | AxisSpec | Sequence[np.ndarray]
) -> int:
    """Infer an axis index from explicit input or axis metadata."""
    if axis is not None:
        return int(axis)
    if isinstance(x, AxisSpec):
        return int(x.order)
    return -1
