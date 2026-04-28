"""Axis-aware transport-dataset mapping helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ..functions.fill_nans import fill as fill_nans
from ..functions.upsampling import upsample
from ..meta.axis import AxisSpec, axis
from ..meta.data import DataSpec
from ..meta.param import ParamSpec
from ._inputs import normalize_samples, restore_samples
from .dataset import TransportDatasetSpec


def mapping(
    sample_or_samples: TransportDatasetSpec | Sequence[TransportDatasetSpec],
    *,
    axis: str,
    N_up: int | None = None,
    xbins: np.ndarray | None = None,
    fill: str | None = None,
    fill_value: float = 0.0,
) -> TransportDatasetSpec | Sequence[TransportDatasetSpec]:
    """Apply axis-aware mapping steps to one or many transport datasets."""
    if N_up is None and xbins is None and fill is None:
        raise ValueError("mapping requires at least one operation.")

    samples, was_sequence, sequence_type = normalize_samples(sample_or_samples)
    mapped_samples: list[TransportDatasetSpec] = []

    for sample in samples:
        target_axis = _require_axis(sample, axis)
        target_values = np.asarray(target_axis.values, dtype=np.float64)
        if np.any(~np.isfinite(target_values)):
            raise ValueError("selected axis must contain only finite values.")

        result = sample
        current_axis = target_axis
        if N_up is not None:
            result, current_axis = _linear_upsample(
                result,
                current_axis,
                int(N_up),
            )
        if xbins is not None:
            result, current_axis = _map_on_axis(result, current_axis, xbins)
        if fill is not None:
            result = _fill_nan(
                result,
                current_axis,
                method=str(fill),
                value=float(fill_value),
            )
        mapped_samples.append(result)

    return restore_samples(
        tuple(mapped_samples),
        was_sequence=was_sequence,
        sequence_type=sequence_type,
    )


def _linear_upsample(
    sample: TransportDatasetSpec,
    target_axis: AxisSpec,
    N_up: int,
) -> tuple[TransportDatasetSpec, AxisSpec]:
    if N_up <= 1:
        return sample, target_axis
    new_axis_values = np.asarray(
        upsample(np.asarray(target_axis.values, dtype=np.float64), N_up=N_up, axis=0),
        dtype=np.float64,
    )
    new_axis = _replace_axis_values(target_axis, new_axis_values)
    return _transform_sample(
        sample,
        target_axis,
        transform=lambda entry, values: _upsample_values(
            entry=entry,
            values=values,
            target_axis=target_axis,
            N_up=N_up,
        ),
        selected_replacement=new_axis,
    ), new_axis


def _map_on_axis(
    sample: TransportDatasetSpec,
    target_axis: AxisSpec,
    xbins: np.ndarray,
) -> tuple[TransportDatasetSpec, AxisSpec]:
    new_axis_values = _validate_xbins(xbins)
    new_axis = _replace_axis_values(target_axis, new_axis_values)
    source_axis_values = np.asarray(target_axis.values, dtype=np.float64)
    return _transform_sample(
        sample,
        target_axis,
        transform=lambda entry, values: _remap_values(
            entry=entry,
            values=values,
            source_axis=source_axis_values,
            target_axis=target_axis,
            xbins=new_axis_values,
        ),
        selected_replacement=new_axis,
    ), new_axis


def _fill_nan(
    sample: TransportDatasetSpec,
    target_axis: AxisSpec,
    *,
    method: str,
    value: float,
) -> TransportDatasetSpec:
    fill_key = method.strip().lower()
    if fill_key not in {"interpolate", "nearest", "value"}:
        raise ValueError(
            "Unsupported fill method. Use 'interpolate', 'nearest', or 'value'."
        )
    return _transform_sample(
        sample,
        target_axis,
        transform=lambda entry, values: _fill_values(
            entry=entry,
            values=values,
            target_axis=target_axis,
            method=fill_key,
            value=value,
        ),
        selected_replacement=target_axis,
        fill_selected_axis=False,
    )


def _transform_sample(
    sample: TransportDatasetSpec,
    target_axis: AxisSpec,
    *,
    transform,
    selected_replacement: AxisSpec,
    fill_selected_axis: bool = True,
) -> TransportDatasetSpec:
    new_data: list[DataSpec] = []
    new_axes: list[AxisSpec] = []
    new_params: list[ParamSpec] = []

    for entry in sample.data:
        if _is_aligned_entry(sample, entry, target_axis):
            new_data.append(replace(entry, values=transform(entry, entry.values)))
        else:
            _require_unambiguous_entry(sample, entry, target_axis)
            new_data.append(entry)

    for entry in sample.axes:
        if entry.code_label == target_axis.code_label:
            new_axes.append(selected_replacement)
            continue
        if _is_aligned_entry(sample, entry, target_axis):
            if not fill_selected_axis or fill_selected_axis:
                new_axes.append(replace(entry, values=transform(entry, entry.values)))
        else:
            _require_unambiguous_entry(sample, entry, target_axis)
            new_axes.append(entry)

    for entry in sample.params:
        if _is_aligned_entry(sample, entry, target_axis):
            new_params.append(replace(entry, values=transform(entry, entry.values)))
        else:
            _require_unambiguous_entry(sample, entry, target_axis)
            new_params.append(entry)

    return TransportDatasetSpec(
        data=tuple(new_data),
        axes=tuple(new_axes),
        params=tuple(new_params),
    )


def _upsample_values(
    *,
    entry: DataSpec | AxisSpec | ParamSpec,
    values: object,
    target_axis: AxisSpec,
    N_up: int,
) -> object:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    axis_index = _entry_axis_index(entry, target_axis)
    return np.asarray(
        upsample(array, N_up=N_up, axis=axis_index, method="linear"),
        dtype=np.float64,
    )


def _remap_values(
    *,
    entry: DataSpec | AxisSpec | ParamSpec,
    values: object,
    source_axis: np.ndarray,
    target_axis: AxisSpec,
    xbins: np.ndarray,
) -> object:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    axis_index = _entry_axis_index(entry, target_axis)
    return _interp_along_axis(
        array,
        source_axis=source_axis,
        xbins=xbins,
        axis=axis_index,
    )


def _fill_values(
    *,
    entry: DataSpec | AxisSpec | ParamSpec,
    values: object,
    target_axis: AxisSpec,
    method: str,
    value: float,
) -> object:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    axis_index = _entry_axis_index(entry, target_axis)
    return np.asarray(
        fill_nans(array, axis=axis_index, method=method, value=value),
        dtype=np.float64,
    )


def _interp_along_axis(
    values: np.ndarray,
    *,
    source_axis: np.ndarray,
    xbins: np.ndarray,
    axis: int,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    work = np.moveaxis(values_arr, axis, -1)
    flat = work.reshape(-1, work.shape[-1])
    out = np.full((flat.shape[0], xbins.size), np.nan, dtype=np.float64)

    for index, row in enumerate(flat):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        x_valid = source_axis[finite]
        y_valid = row[finite]
        if x_valid.size == 1:
            nearest = int(np.argmin(np.abs(xbins - x_valid[0])))
            out[index, nearest] = y_valid[0]
            continue
        out[index] = np.interp(
            xbins,
            x_valid,
            y_valid,
            left=np.nan,
            right=np.nan,
        )

    reshaped = out.reshape(*work.shape[:-1], xbins.size)
    return np.moveaxis(reshaped, -1, axis)


def _require_axis(sample: TransportDatasetSpec, code_label: str) -> AxisSpec:
    entry = sample._find_axis(str(code_label))
    if entry is None:
        raise ValueError(f"sample is missing axis '{code_label}'.")
    return entry


def _replace_axis_values(axis_entry: AxisSpec, values: np.ndarray) -> AxisSpec:
    return axis(
        axis_entry.code_label,
        values=np.asarray(values, dtype=np.float64),
        order=axis_entry.order,
    )


def _is_aligned_entry(
    sample: TransportDatasetSpec,
    entry: DataSpec | AxisSpec | ParamSpec,
    target_axis: AxisSpec,
) -> bool:
    array = np.asarray(entry.values, dtype=np.float64)
    if array.ndim == 0:
        return False
    target_len = np.asarray(target_axis.values, dtype=np.float64).size
    if isinstance(entry, AxisSpec):
        return entry.order == target_axis.order and array.size == target_len

    if array.ndim == 1:
        return array.size == target_len

    data_shape = sample._data_shape
    if tuple(array.shape) == data_shape:
        return array.shape[target_axis.order] == target_len
    return False


def _require_unambiguous_entry(
    sample: TransportDatasetSpec,
    entry: DataSpec | AxisSpec | ParamSpec,
    target_axis: AxisSpec,
) -> None:
    array = np.asarray(entry.values, dtype=np.float64)
    if array.ndim <= 1:
        return
    data_shape = sample._data_shape
    if tuple(array.shape) == data_shape:
        return
    try:
        broadcast = np.broadcast_to(array, data_shape)
    except ValueError:
        return
    if broadcast.shape[target_axis.order] == np.asarray(target_axis.values).size:
        raise ValueError(
            f"entry '{entry.code_label}' is broadcastable to the selected dimension "
            "but cannot be transformed unambiguously from its stored shape."
        )


def _entry_axis_index(
    entry: DataSpec | AxisSpec | ParamSpec,
    target_axis: AxisSpec,
) -> int:
    if isinstance(entry, AxisSpec):
        return 0
    array = np.asarray(entry.values, dtype=np.float64)
    if array.ndim == 1:
        return 0
    return target_axis.order


def _validate_xbins(xbins: np.ndarray) -> np.ndarray:
    values = np.asarray(xbins, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("xbins must contain at least two values.")
    if np.any(~np.isfinite(values)):
        raise ValueError("xbins must contain only finite values.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("xbins must be strictly increasing.")
    return values


__all__ = ["mapping"]
