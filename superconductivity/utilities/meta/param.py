"""Parameter metadata and constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import NDArray64
from .label import LabelSpec, label


@dataclass(frozen=True, slots=True)
class ParamSpec(LabelSpec):
    """Parameter metadata shared by fitting and GUI tables."""

    __array_priority__ = 1000.0

    value: float | Sequence[float] | NDArray64
    error: float | Sequence[float] | NDArray64 | None = None
    lower: float | None = None
    upper: float | None = None
    fixed: bool = False

    def __post_init__(self) -> None:
        LabelSpec.__post_init__(self)
        value = _coerce_value(self.value)
        object.__setattr__(self, "value", value)
        if self.lower is not None:
            object.__setattr__(self, "lower", float(self.lower))
        if self.upper is not None:
            object.__setattr__(self, "upper", float(self.upper))
        if self.error is not None:
            error = _coerce_value(self.error)
            _validate_error_shape(value, error)
            object.__setattr__(self, "error", error)
        object.__setattr__(self, "fixed", bool(self.fixed))

    def __float__(self) -> float:
        if isinstance(self.value, np.ndarray):
            if self.value.ndim != 0:
                raise TypeError(
                    "Cannot convert non-scalar ParamSpec to float.",
                )
            return float(self.value)
        return float(self.value)

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        """Return one NumPy scalar view for array arithmetic."""
        array = np.asarray(float(self), dtype=np.float64)
        if dtype is not None:
            return np.asarray(array, dtype=dtype)
        return array

    def _as_float(self, other: object) -> float:
        if isinstance(other, ParamSpec):
            return float(other)
        return float(np.asarray(other, dtype=np.float64))

    def __mul__(self, other: object):
        return float(self) * np.asarray(other, dtype=np.float64)

    def __rmul__(self, other: object):
        return np.asarray(other, dtype=np.float64) * float(self)

    def __truediv__(self, other: object):
        return float(self) / np.asarray(other, dtype=np.float64)

    def __rtruediv__(self, other: object):
        return np.asarray(other, dtype=np.float64) / float(self)

    def __add__(self, other: object):
        return float(self) + np.asarray(other, dtype=np.float64)

    def __radd__(self, other: object):
        return np.asarray(other, dtype=np.float64) + float(self)

    def __sub__(self, other: object):
        return float(self) - np.asarray(other, dtype=np.float64)

    def __rsub__(self, other: object):
        return np.asarray(other, dtype=np.float64) - float(self)


def param(
    name: str,
    value: float | Sequence[float] | NDArray64,
    *,
    error: float | Sequence[float] | NDArray64 | None = None,
    lower: float | None = None,
    upper: float | None = None,
    fixed: bool = False,
) -> ParamSpec:
    try:
        meta = label(name)
    except KeyError:
        meta = LabelSpec(code_label=name, html_label=name, latex_label=name)
    return ParamSpec(
        code_label=meta.code_label,
        print_label=meta.print_label,
        html_label=meta.html_label,
        latex_label=meta.latex_label,
        value=value,
        error=error,
        lower=lower,
        upper=upper,
        fixed=fixed,
    )


def _coerce_value(
    value: float | Sequence[float] | NDArray64,
) -> float | NDArray64:
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    if hasattr(value, "__array__"):
        array = np.asarray(value, dtype=np.float64)
        if getattr(array, "ndim", 1) == 0:
            return float(array)
        return array
    return float(value)


def _validate_error_shape(
    value: float | NDArray64,
    error: float | NDArray64,
) -> None:
    value_is_array = isinstance(value, np.ndarray)
    error_is_array = isinstance(error, np.ndarray)
    if value_is_array != error_is_array:
        raise ValueError("error must match value shape.")
    if value_is_array and error_is_array and value.shape != error.shape:
        raise ValueError("error must match value shape.")


__all__ = ["ParamSpec", "param"]
