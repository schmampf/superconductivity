"""Axis metadata and constructors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..safety import require_all_finite, require_min_size
from ..types import NDArray64
from .data import DataSpec
from .label import label


@dataclass(frozen=True, slots=True)
class AxisSpec(DataSpec):
    """One labeled, strictly increasing axis grid."""

    order: int

    def __post_init__(self) -> None:
        DataSpec.__post_init__(self)
        values = _validate_axis(self.values, "values")
        try:
            kind = int(self.order)
        except (TypeError, ValueError) as exc:
            raise ValueError("order must be a non-negative integer.") from exc
        if kind < 0:
            raise ValueError("order must be a non-negative integer.")
        object.__setattr__(self, "order", kind)
        object.__setattr__(self, "values", values)

    @property
    def axis(self) -> NDArray64:
        return np.asarray(self.values, dtype=np.float64)


def axis(
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
    order: int = 0,
) -> AxisSpec:
    meta = label(name)
    if values is not None:
        axis_values = _validate_axis(values, "values")
    else:
        if min_value is None or max_value is None or bins is None:
            raise ValueError(
                "min_value, max_value, and bins"
                " are required when values is not given."
            )
        axis_values = _linspace_axis(min_value, max_value, bins)
    return AxisSpec(
        code_label=meta.code_label,
        print_label=meta.print_label,
        html_label=meta.html_label,
        latex_label=meta.latex_label,
        values=axis_values,
        order=order,
    )


def _linspace_axis(min_value: float, max_value: float, bins: int) -> NDArray64:
    start = float(min_value)
    stop = float(max_value)
    count = int(bins)
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError("min_value and max_value must be finite.")
    if count < 2:
        raise ValueError("bins must be >= 2.")
    return np.linspace(start, stop, count, dtype=np.float64)


def _validate_axis(
    values: Sequence[float] | NDArray64,
    name: str,
) -> NDArray64:
    axis_values = np.asarray(values, dtype=np.float64).reshape(-1)
    require_min_size(axis_values, 2, name)
    require_all_finite(axis_values, name)
    if np.any(np.diff(axis_values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return axis_values


__all__ = ["AxisSpec", "axis"]
