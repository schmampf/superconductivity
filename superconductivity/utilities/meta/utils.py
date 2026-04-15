"""Shared helpers for metadata-aware containers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .axis import AxisSpec
from .dataset import Dataset
from .param import ParamSpec


def unwrap_dataset_value(value: np.ndarray | Dataset | AxisSpec) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        return value  # preserve ragged sequences for callers that handle them
    if isinstance(value, Dataset):
        return value.values
    if isinstance(value, AxisSpec):
        return value.values
    return np.asarray(value)


def wrap_dataset_value(template: np.ndarray | Dataset, values: np.ndarray) -> np.ndarray | Dataset:
    if isinstance(template, Dataset):
        return Dataset(
            code_label=template.code_label,
            print_label=template.print_label,
            html_label=template.html_label,
            latex_label=template.latex_label,
            values=values,
            axes=template.axes,
            params=template.params,
        )
    return values


def coerce_axes(axes: Sequence[AxisSpec] | AxisSpec) -> tuple[AxisSpec, ...]:
    if isinstance(axes, AxisSpec):
        return (axes,)
    return tuple(axes)


def coerce_params(
    params: Sequence[ParamSpec] | ParamSpec,
) -> tuple[ParamSpec, ...]:
    if isinstance(params, ParamSpec):
        return (params,)
    return tuple(params)


def infer_axis(
    axis: int | None, x: np.ndarray | AxisSpec | Sequence[np.ndarray]
) -> int:
    """Infer an axis index from explicit input or axis metadata."""
    if axis is not None:
        return int(axis)
    if isinstance(x, AxisSpec):
        return int(x.order)
    return -1
