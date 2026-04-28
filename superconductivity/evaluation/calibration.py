"""Collection-axis calibration helpers for sampled transport datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..utilities.meta import AxisSpec, LabelSpec, TransportDatasetSpec, axis
from ..utilities.meta.label import label as make_label
from ..utilities.types import NDArray64


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    """Normalized collection-axis calibration specification."""

    label: str | LabelSpec
    transform: Callable[..., NDArray64] | None = None
    params: float | Sequence[float] = 1.0
    lookup: Sequence[float] | NDArray64 | None = None

    def __post_init__(self) -> None:
        if isinstance(self.label, str):
            object.__setattr__(self, "label", make_label(self.label))
        elif not isinstance(self.label, LabelSpec):
            raise ValueError("label must be a string or LabelSpec.")

        has_transform = self.transform is not None
        has_lookup = self.lookup is not None
        if has_transform == has_lookup:
            raise ValueError(
                "exactly one of transform or lookup must be provided.",
            )

        object.__setattr__(self, "params", _normalize_params(self.params))
        if has_lookup:
            object.__setattr__(self, "lookup", _normalize_lookup(self.lookup))


def calibrate(
    exp_v: TransportDatasetSpec,
    exp_i: TransportDatasetSpec,
    *,
    calibrationspec: CalibrationSpec,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Calibrate the shared collection axis on sampled transport datasets."""
    axis_v = _collection_axis(exp_v)
    axis_i = _collection_axis(exp_i)
    _require_matching_axes(axis_v, axis_i)

    calibrated_values = _apply_calibration(
        np.asarray(axis_v.values, dtype=np.float64),
        calibrationspec,
    )
    calibrated_axis = axis(
        calibrationspec.label.code_label,
        values=calibrated_values,
        order=axis_v.order,
    )

    return (
        _replace_collection_axis(exp_v, axis_v, calibrated_axis),
        _replace_collection_axis(exp_i, axis_i, calibrated_axis),
    )


def _collection_axis(sample: TransportDatasetSpec) -> AxisSpec:
    collection_axes = [
        axis_entry
        for axis_entry in sample.axes
        if axis_entry.code_label not in {"V_mV", "I_nA"}
    ]
    if len(collection_axes) != 1:
        raise ValueError(
            "calibration requires exactly one collection axis on each sample.",
        )
    return collection_axes[0]


def _require_matching_axes(left: AxisSpec, right: AxisSpec) -> None:
    if left.order != right.order:
        raise ValueError("collection axes must have the same order.")
    left_values = np.asarray(left.values, dtype=np.float64)
    right_values = np.asarray(right.values, dtype=np.float64)
    if left_values.shape != right_values.shape or not np.allclose(
        left_values,
        right_values,
        equal_nan=False,
    ):
        raise ValueError("collection axes must have identical values.")


def _apply_calibration(
    source_axis: NDArray64,
    spec: CalibrationSpec,
) -> NDArray64:
    if spec.transform is not None:
        mapped = np.asarray(
            spec.transform(source_axis, *spec.params),
            dtype=np.float64,
        )
    else:
        assert spec.lookup is not None
        mapped = np.asarray(spec.lookup, dtype=np.float64)

    if mapped.shape != source_axis.shape:
        raise ValueError(
            "calibrated axis must have the same shape as the source axis.",
        )
    if np.any(~np.isfinite(mapped)):
        raise ValueError("calibrated axis must be finite.")
    return mapped


def _replace_collection_axis(
    sample: TransportDatasetSpec,
    old_axis: AxisSpec,
    new_axis: AxisSpec,
) -> TransportDatasetSpec:
    axes = tuple(
        new_axis if axis_entry.code_label == old_axis.code_label else axis_entry
        for axis_entry in sample.axes
    )
    return TransportDatasetSpec(
        data=sample.data,
        axes=axes,
        params=sample.params,
    )


def _normalize_params(params: float | Sequence[float]) -> tuple[float, ...]:
    if np.isscalar(params):
        values = (float(params),)
    else:
        values = tuple(float(value) for value in params)
        if len(values) == 0:
            raise ValueError("params must not be empty.")
    if not all(np.isfinite(value) for value in values):
        raise ValueError("params must be finite.")
    return values


def _normalize_lookup(values: Sequence[float] | NDArray64 | None) -> NDArray64:
    assert values is not None
    lookup = np.asarray(values, dtype=np.float64).reshape(-1)
    if lookup.size == 0:
        raise ValueError("lookup must not be empty.")
    return lookup


__all__ = [
    "CalibrationSpec",
    "calibrate",
]
