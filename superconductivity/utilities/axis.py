"""Shared axis metadata and generic axis construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

import numpy as np

from .label import LabelMeta
from .safety import require_all_finite, require_min_size
from .types import NDArray64

AxisKind: TypeAlias = Literal["x", "y"]


@dataclass(frozen=True, slots=True)
class AxisSpec(LabelMeta):
    """One labeled, strictly increasing axis grid."""

    axis: Sequence[float] | NDArray64
    kind: AxisKind

    def __post_init__(self) -> None:
        LabelMeta.__post_init__(self)
        axis = _validate_axis(self.axis, "axis")
        if self.kind not in ("x", "y"):
            raise ValueError("kind must be 'x' or 'y'.")
        object.__setattr__(self, "axis", axis)


def construct_axis(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
    meta: LabelMeta,
    kind: AxisKind = "y",
) -> AxisSpec:
    """Construct one labeled axis from either values or linspace semantics."""
    if values is not None:
        axis = _validate_axis(values, "values")
    else:
        if min_value is None or max_value is None or bins is None:
            raise ValueError(
                "min_value, max_value, and bins are required when values is not given.",
            )
        axis = _linspace_axis(min_value, max_value, bins)
    return AxisSpec(
        axis=axis,
        kind=kind,
        label=meta.label,
        html_label=meta.html_label,
        latex_label=meta.latex_label,
    )


def _linspace_axis(
    min_value: float,
    max_value: float,
    bins: int,
) -> NDArray64:
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
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    require_min_size(axis, 2, name)
    require_all_finite(axis, name)
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return axis


__all__ = [
    "AxisKind",
    "AxisSpec",
    "construct_axis",
]
