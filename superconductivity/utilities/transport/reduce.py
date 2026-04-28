"""Helpers for enriching sampled transport datasets with reduction metadata."""

from __future__ import annotations

import numpy as np

from ..meta.axis import AxisSpec, axis
from ..meta.param import ParamSpec, param
from .dataset import TransportDatasetSpec


def reduce(
    exp_v: TransportDatasetSpec,
    exp_i: TransportDatasetSpec,
    *,
    Delta_meV: AxisSpec | ParamSpec | float | np.ndarray | list[float] | None = None,
    GN_G0: AxisSpec | ParamSpec | float | np.ndarray | list[float] | None = None,
    nu_GHz: AxisSpec | ParamSpec | float | np.ndarray | list[float] | None = None,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Attach reduction metadata to sampled transport datasets."""
    collection_axis_v = _collection_axis(exp_v)
    collection_axis_i = _collection_axis(exp_i)
    _require_matching_collection_axes(collection_axis_v, collection_axis_i)

    delta_entry = _coerce_reduction_entry(
        "Delta_meV",
        Delta_meV,
        collection_axis=collection_axis_v,
    )
    gn_entry = _coerce_reduction_entry(
        "GN_G0",
        GN_G0,
        collection_axis=collection_axis_v,
    )
    nu_entry = _coerce_reduction_entry(
        "nu_GHz",
        nu_GHz,
        collection_axis=collection_axis_v,
    )

    reduced_v = exp_v
    reduced_i = exp_i
    for code_label, entry in (
        ("Delta_meV", delta_entry),
        ("GN_G0", gn_entry),
        ("nu_GHz", nu_entry),
    ):
        if entry is None:
            continue
        reduced_v = _replace_entry(reduced_v, code_label, entry)
        reduced_i = _replace_entry(reduced_i, code_label, entry)
    return reduced_v, reduced_i


def _collection_axis(sample: TransportDatasetSpec) -> AxisSpec:
    collection_axes = [
        axis_entry
        for axis_entry in sample.axes
        if axis_entry.code_label not in {"V_mV", "I_nA"}
    ]
    if len(collection_axes) != 1:
        raise ValueError("reduce requires exactly one collection axis.")
    return collection_axes[0]


def _require_matching_collection_axes(left: AxisSpec, right: AxisSpec) -> None:
    if left.order != right.order:
        raise ValueError("collection axes must have the same order.")
    left_values = np.asarray(left.values, dtype=np.float64)
    right_values = np.asarray(right.values, dtype=np.float64)
    if left_values.shape != right_values.shape or not np.allclose(left_values, right_values):
        raise ValueError("collection axes must have identical values.")


def _coerce_reduction_entry(
    code_label: str,
    value: AxisSpec | ParamSpec | float | np.ndarray | list[float] | None,
    *,
    collection_axis: AxisSpec,
) -> AxisSpec | ParamSpec | None:
    if value is None:
        return None
    if isinstance(value, AxisSpec):
        if value.order != collection_axis.order:
            raise ValueError(
                f"{code_label} axis order must match the collection axis order.",
            )
        if np.asarray(value.values, dtype=np.float64).shape != np.asarray(
            collection_axis.values,
            dtype=np.float64,
        ).shape:
            raise ValueError(
                f"{code_label} axis length must match the collection axis length.",
            )
        return value
    if isinstance(value, ParamSpec):
        return value
    if np.isscalar(value):
        return param(code_label, float(value))

    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        return param(code_label, float(values))
    if values.ndim == 1 and values.shape == np.asarray(
        collection_axis.values,
        dtype=np.float64,
    ).shape:
        return axis(
            code_label,
            values=values,
            order=collection_axis.order,
        )
    raise ValueError(
        f"{code_label} must be scalar or a 1D array matching the collection axis.",
    )


def _replace_entry(
    sample: TransportDatasetSpec,
    code_label: str,
    entry: AxisSpec | ParamSpec,
) -> TransportDatasetSpec:
    reduced = sample.remove(code_label)
    if isinstance(entry, AxisSpec):
        return reduced.add(**{code_label: entry})
    return reduced.add(**{code_label: entry})


__all__ = ["reduce"]
