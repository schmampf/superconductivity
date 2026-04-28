"""Helpers for enriching sampled transport datasets with reduction metadata."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..meta.axis import AxisSpec, axis
from ..meta.param import ParamSpec, param
from ._inputs import normalize_samples, restore_samples
from .dataset import TransportDatasetSpec


def reduced_units(
    sample_or_samples: TransportDatasetSpec | Sequence[TransportDatasetSpec],
    *,
    Delta_meV: AxisSpec | ParamSpec | float | np.ndarray | list[float] | None = None,
    GN_G0: AxisSpec | ParamSpec | float | np.ndarray | list[float] | None = None,
    nu_GHz: AxisSpec | ParamSpec | float | np.ndarray | list[float] | None = None,
) -> TransportDatasetSpec | Sequence[TransportDatasetSpec]:
    """Attach reduction metadata to sampled transport datasets."""
    samples, was_sequence, sequence_type = normalize_samples(sample_or_samples)
    collection_axes = tuple(_collection_axis(sample) for sample in samples)
    first_axis = collection_axes[0]
    for collection_axis in collection_axes[1:]:
        _require_matching_collection_axes(first_axis, collection_axis)

    delta_entry = _coerce_reduction_entry(
        "Delta_meV",
        Delta_meV,
        collection_axis=first_axis,
    )
    gn_entry = _coerce_reduction_entry(
        "GN_G0",
        GN_G0,
        collection_axis=first_axis,
    )
    nu_entry = _coerce_reduction_entry(
        "nu_GHz",
        nu_GHz,
        collection_axis=first_axis,
    )

    reduced_samples: list[TransportDatasetSpec] = []
    for sample in samples:
        reduced = sample
        for code_label, entry in (
            ("Delta_meV", delta_entry),
            ("GN_G0", gn_entry),
            ("nu_GHz", nu_entry),
        ):
            if entry is None:
                continue
            reduced = _replace_entry(reduced, code_label, entry)
        reduced_samples.append(reduced)
    return restore_samples(
        tuple(reduced_samples),
        was_sequence=was_sequence,
        sequence_type=sequence_type,
    )


def _collection_axis(sample: TransportDatasetSpec) -> AxisSpec:
    collection_axes = [
        axis_entry
        for axis_entry in sample.axes
        if axis_entry.code_label not in {"V_mV", "I_nA"}
    ]
    if len(collection_axes) != 1:
        raise ValueError("reduced_units requires exactly one collection axis.")
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


__all__ = ["reduced_units"]
