"""Shared value-bearing metadata helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..types import NDArray64
from .label import LabelSpec, label


@dataclass(frozen=True, slots=True)
class DataSpec(LabelSpec):
    """Label metadata coupled to a scalar or array-like value payload."""

    __array_priority__ = 1000.0

    values: float | Sequence[float] | NDArray64

    def __post_init__(self) -> None:
        LabelSpec.__post_init__(self)
        object.__setattr__(self, "values", _coerce_values(self.values))

    @property
    def data(self) -> DataSpec:
        """Return self for compatibility with older container-style code."""
        return self

    @property
    def value(self):
        """Compatibility alias for the stored payload."""
        return self.values

    @property
    def shape(self) -> tuple[int, ...]:
        """Return NumPy-style shape of the stored payload."""
        return tuple(np.asarray(self.values).shape)

    def __float__(self) -> float:
        values = self.values
        if isinstance(values, np.ndarray):
            if values.ndim != 0:
                raise TypeError(
                    f"Cannot convert non-scalar {type(self).__name__} to float.",
                )
            return float(values)
        return float(values)

    def __int__(self) -> int:
        return int(float(self))

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        array = np.asarray(self.values, dtype=np.float64)
        if dtype is not None:
            return np.asarray(array, dtype=dtype)
        return array

    def __len__(self) -> int:
        array = np.asarray(self.values)
        if array.ndim == 0:
            raise TypeError(f"len() of scalar {type(self).__name__}.")
        return int(array.size)

    def __iter__(self):
        array = np.asarray(self.values)
        if array.ndim == 0:
            raise TypeError(f"'{type(self).__name__}' object is not iterable.")
        return iter(np.asarray(array).tolist())

    def __getitem__(self, index: int | slice):
        return np.asarray(self.values)[index]

    def _coerce_other(self, other: object) -> np.ndarray | float:
        if isinstance(other, DataSpec):
            return np.asarray(other.values, dtype=np.float64)
        return np.asarray(other, dtype=np.float64)

    def _binary_op(self, other: object, op):
        return op(np.asarray(self.values, dtype=np.float64), self._coerce_other(other))

    def _binary_rop(self, other: object, op):
        return op(self._coerce_other(other), np.asarray(self.values, dtype=np.float64))

    def _unary_op(self, op):
        return op(np.asarray(self.values, dtype=np.float64))

    def __pos__(self):
        return self._unary_op(lambda values: +values)

    def __neg__(self):
        return self._unary_op(lambda values: -values)

    def __abs__(self):
        return self._unary_op(np.abs)

    def __add__(self, other: object):
        return self._binary_op(other, np.add)

    def __radd__(self, other: object):
        return self._binary_rop(other, np.add)

    def __sub__(self, other: object):
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other: object):
        return self._binary_rop(other, np.subtract)

    def __mul__(self, other: object):
        return self._binary_op(other, np.multiply)

    def __rmul__(self, other: object):
        return self._binary_rop(other, np.multiply)

    def __truediv__(self, other: object):
        return self._binary_op(other, np.divide)

    def __rtruediv__(self, other: object):
        return self._binary_rop(other, np.divide)

    def __pow__(self, other: object):
        return self._binary_op(other, np.power)

    def __rpow__(self, other: object):
        return self._binary_rop(other, np.power)

    def __lt__(self, other: object):
        return self._binary_op(other, np.less)

    def __le__(self, other: object):
        return self._binary_op(other, np.less_equal)

    def __gt__(self, other: object):
        return self._binary_op(other, np.greater)

    def __ge__(self, other: object):
        return self._binary_op(other, np.greater_equal)

    def __eq__(self, other: object):  # type: ignore[override]
        return self._binary_op(other, np.equal)

    def __ne__(self, other: object):  # type: ignore[override]
        return self._binary_op(other, np.not_equal)


def data(
    name: str,
    values: float | Sequence[float] | NDArray64,
) -> DataSpec:
    """Construct one labeled numeric payload."""
    meta = label(name)
    return DataSpec(
        code_label=meta.code_label,
        print_label=meta.print_label,
        html_label=meta.html_label,
        latex_label=meta.latex_label,
        values=values,
    )


def _coerce_values(
    values: float | Sequence[float] | NDArray64,
) -> float | NDArray64:
    if isinstance(values, (list, tuple)):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            return float(array)
        return array
    if hasattr(values, "__array__"):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            return float(array)
        return array
    return float(values)


__all__ = ["DataSpec", "data"]
