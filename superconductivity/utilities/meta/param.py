"""Parameter metadata and constructors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..types import NDArray64
from .data import DataSpec, _coerce_values
from .label import label


@dataclass(frozen=True, slots=True)
class ParamSpec(DataSpec):
    """Parameter metadata shared by fitting and GUI tables."""

    error: float | Sequence[float] | NDArray64 | None = None
    lower: float | None = None
    upper: float | None = None
    fixed: bool = False

    def __post_init__(self) -> None:
        DataSpec.__post_init__(self)
        if self.lower is not None:
            object.__setattr__(self, "lower", float(self.lower))
        if self.upper is not None:
            object.__setattr__(self, "upper", float(self.upper))
        if self.error is not None:
            error = _coerce_values(self.error)
            _validate_error_shape(self.values, error)
            object.__setattr__(self, "error", error)
        object.__setattr__(self, "fixed", bool(self.fixed))

    def __float__(self) -> float:
        return DataSpec.__float__(self)

    def _scalar_value(self) -> float:
        return float(self)

    def __mul__(self, other: object):
        return self._binary_op(other, np.multiply)

    def __rmul__(self, other: object):
        return self._binary_rop(other, np.multiply)

    def __truediv__(self, other: object):
        return self._binary_op(other, np.divide)

    def __rtruediv__(self, other: object):
        return self._binary_rop(other, np.divide)

    def __add__(self, other: object):
        return self._binary_op(other, np.add)

    def __radd__(self, other: object):
        return self._binary_rop(other, np.add)

    def __sub__(self, other: object):
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other: object):
        return self._binary_rop(other, np.subtract)

    def __pow__(self, other: object):
        return self._binary_op(other, np.power)

    def __rpow__(self, other: object):
        return self._binary_rop(other, np.power)


def param(
    name: str,
    value: float | Sequence[float] | NDArray64,
    *,
    error: float | Sequence[float] | NDArray64 | None = None,
    lower: float | None = None,
    upper: float | None = None,
    fixed: bool = False,
) -> ParamSpec:
    meta = label(name)
    return ParamSpec(
        code_label=meta.code_label,
        print_label=meta.print_label,
        html_label=meta.html_label,
        latex_label=meta.latex_label,
        values=value,
        error=error,
        lower=lower,
        upper=upper,
        fixed=fixed,
    )


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
