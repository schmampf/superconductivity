"""Dataset metadata and constructors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .axis import AxisSpec
from .label import LabelSpec, label
from .param import ParamSpec
from ..types import NDArray64


@dataclass(frozen=True, slots=True)
class Dataset(LabelSpec):
    """Labeled numeric data with attached axes and parameters."""

    values: Sequence[float] | NDArray64
    axes: tuple[AxisSpec, ...] = ()
    params: tuple[ParamSpec, ...] = ()

    def __post_init__(self) -> None:
        LabelSpec.__post_init__(self)
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "axes", _coerce_axes(self.axes))
        object.__setattr__(self, "params", _coerce_params(self.params))

    @property
    def data(self) -> Dataset:
        return self


def dataset(
    name: str,
    values: Sequence[float] | NDArray64,
    *,
    axes: Sequence[AxisSpec] | AxisSpec = (),
    params: Sequence[ParamSpec] | ParamSpec = (),
) -> Dataset:
    """Construct a dataset-level metadata object."""
    try:
        meta = label(name)
    except KeyError:
        meta = LabelSpec(
            code_label=name,
            print_label=name,
            html_label=name,
            latex_label=name,
        )
    return Dataset(
        code_label=meta.code_label,
        print_label=meta.print_label,
        html_label=meta.html_label,
        latex_label=meta.latex_label,
        values=values,
        axes=axes,
        params=params,
    )


def _coerce_axes(
    axes: Sequence[AxisSpec] | AxisSpec,
) -> tuple[AxisSpec, ...]:
    if isinstance(axes, AxisSpec):
        return (axes,)
    return tuple(axes)


def _coerce_params(
    params: Sequence[ParamSpec] | ParamSpec,
) -> tuple[ParamSpec, ...]:
    if isinstance(params, ParamSpec):
        return (params,)
    return tuple(params)


__all__ = ["Dataset", "dataset"]
