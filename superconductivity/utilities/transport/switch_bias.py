"""Transport-dataset bias switching helpers."""

from __future__ import annotations

import numpy as np

from ..types import NDArray64
from ..meta.axis import AxisSpec, axis
from ..meta.data import DataSpec
from ..meta.param import ParamSpec
from .dataset import TransportDatasetSpec
from .mapping import mapping


def switch_bias(
    sample: TransportDatasetSpec,
    *,
    I_nA: AxisSpec | NDArray64 | None = None,
    V_mV: AxisSpec | NDArray64 | None = None,
    N_up: int | None = None,
    fill: str | None = None,
    fill_value: float = 0.0,
) -> TransportDatasetSpec:
    """Switch one transport dataset between voltage- and current-bias views."""
    if (I_nA is None) == (V_mV is None):
        raise ValueError("Exactly one of I_nA or V_mV must be provided.")

    if I_nA is not None:
        source_axis_label = "V_mV"
        source_data_label = "I_nA"
        target_axis = _coerce_target_axis(
            "I_nA",
            I_nA,
            order=_require_axis(sample, "V_mV").order,
        )
    else:
        source_axis_label = "I_nA"
        source_data_label = "V_mV"
        target_axis = _coerce_target_axis(
            "V_mV",
            V_mV,
            order=_require_axis(sample, "I_nA").order,
        )

    _require_canonical_source(
        sample,
        source_axis_label=source_axis_label,
        source_data_label=source_data_label,
    )

    prepared = sample
    if fill is not None:
        prepared = mapping(
            prepared,
            axis=source_axis_label,
            fill=fill,
            fill_value=fill_value,
        )
    if N_up is not None:
        prepared = mapping(
            prepared,
            axis=source_axis_label,
            N_up=N_up,
        )

    source_axis = _require_axis(prepared, source_axis_label)
    source_data = _require_data(prepared, source_data_label)
    source_transport = np.asarray(source_data.values, dtype=np.float64)
    _require_monotonic_transport(
        source_transport,
        axis=source_axis.order,
        code_label=source_data_label,
    )

    return _switch_transport_dataset(
        prepared,
        source_axis=source_axis,
        source_transport=source_transport,
        target_axis=target_axis,
    )


def _switch_transport_dataset(
    sample: TransportDatasetSpec,
    *,
    source_axis: AxisSpec,
    source_transport: NDArray64,
    target_axis: AxisSpec,
) -> TransportDatasetSpec:
    new_data: list[DataSpec] = []
    new_axes: list[AxisSpec] = []
    new_params: list[ParamSpec] = []
    switched_transport_data: DataSpec | None = None

    for entry in sample.data:
        if entry.code_label == source_transport_label(source_axis.code_label):
            continue
        if entry.code_label == source_axis.code_label:
            remapped = _remap_entry(
                entry=entry,
                source_transport=source_transport,
                target_axis=target_axis,
                source_axis=source_axis,
            )
            new_data.append(_as_data(entry, remapped))
            continue
        if _is_aligned_entry(sample, entry, source_axis):
            remapped = _remap_entry(
                entry=entry,
                source_transport=source_transport,
                target_axis=target_axis,
                source_axis=source_axis,
            )
            new_data.append(_as_data(entry, remapped))
            continue
        _require_unambiguous_entry(sample, entry, source_axis)
        new_data.append(entry)

    for entry in sample.axes:
        if entry.code_label == source_axis.code_label:
            switched_transport_data = _as_data(
                entry,
                _remap_entry(
                    entry=entry,
                    source_transport=source_transport,
                    target_axis=target_axis,
                    source_axis=source_axis,
                ),
            )
            new_axes.append(target_axis)
            continue
        if _is_aligned_entry(sample, entry, source_axis):
            remapped = _remap_entry(
                entry=entry,
                source_transport=source_transport,
                target_axis=target_axis,
                source_axis=source_axis,
            )
            new_data.append(_as_data(entry, remapped))
            continue
        _require_unambiguous_entry(sample, entry, source_axis)
        new_axes.append(entry)

    for entry in sample.params:
        if _is_aligned_entry(sample, entry, source_axis):
            remapped = _remap_entry(
                entry=entry,
                source_transport=source_transport,
                target_axis=target_axis,
                source_axis=source_axis,
            )
            new_data.append(_as_data(entry, remapped))
            continue
        _require_unambiguous_entry(sample, entry, source_axis)
        new_params.append(entry)

    if switched_transport_data is None:
        raise ValueError("source transport axis could not be converted to data.")
    new_data.append(switched_transport_data)

    return TransportDatasetSpec(
        data=tuple(new_data),
        axes=tuple(new_axes),
        params=tuple(new_params),
    )
def _coerce_target_axis(
    code_label: str,
    value: AxisSpec | NDArray64 | None,
    *,
    order: int,
) -> AxisSpec:
    if isinstance(value, AxisSpec):
        if value.code_label != code_label:
            raise ValueError(f"target axis must be labeled '{code_label}'.")
        if value.order != order:
            raise ValueError("target axis order must match the source transport axis.")
        _validate_target_values(value.values)
        return value
    if value is None:
        raise ValueError(f"target axis '{code_label}' is required.")
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    _validate_target_values(values)
    return axis(code_label, values=values, order=order)


def _validate_target_values(values: object) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("target axis must contain at least two values.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("target axis must contain only finite values.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("target axis must be strictly increasing.")


def _require_canonical_source(
    sample: TransportDatasetSpec,
    *,
    source_axis_label: str,
    source_data_label: str,
) -> None:
    if sample._find_axis(source_axis_label) is None:
        raise ValueError(f"sample must contain '{source_axis_label}' as an axis.")
    if sample._find_data(source_data_label) is None:
        raise ValueError(f"sample must contain '{source_data_label}' as data.")
    other_axis = source_transport_label(source_axis_label)
    if sample._find_axis(other_axis) is not None:
        raise ValueError(
            f"sample is already in incompatible bias form because '{other_axis}' is an axis."
        )


def _require_axis(sample: TransportDatasetSpec, code_label: str) -> AxisSpec:
    axis_entry = sample._find_axis(code_label)
    if axis_entry is None:
        raise ValueError(f"sample is missing axis '{code_label}'.")
    return axis_entry


def _require_data(sample: TransportDatasetSpec, code_label: str) -> DataSpec:
    data_entry = sample._find_data(code_label)
    if data_entry is None:
        raise ValueError(f"sample is missing data '{code_label}'.")
    return data_entry


def _require_monotonic_transport(
    values: NDArray64,
    *,
    axis: int,
    code_label: str,
) -> None:
    work = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    flat = work.reshape(-1, work.shape[-1])
    for row in flat:
        if np.any(~np.isfinite(row)):
            raise ValueError(
                f"{code_label} must be finite before switching bias."
            )
        diff = np.diff(row)
        if np.all(diff > 0.0) or np.all(diff < 0.0):
            continue
        raise ValueError(
            f"{code_label} must be strictly monotonic along the source transport axis."
        )


def _remap_entry(
    *,
    entry: DataSpec | AxisSpec | ParamSpec,
    source_transport: NDArray64,
    target_axis: AxisSpec,
    source_axis: AxisSpec,
) -> NDArray64:
    y_values = np.asarray(entry.values, dtype=np.float64)
    if y_values.ndim == 0:
        raise ValueError("scalar entries must not be remapped.")
    x_work = np.moveaxis(np.asarray(source_transport, dtype=np.float64), source_axis.order, -1)
    if y_values.ndim == 1:
        if y_values.size != x_work.shape[-1]:
            raise ValueError(
                f"entry '{entry.code_label}' is not aligned with the source transport axis."
            )
        y_flat = np.broadcast_to(
            y_values.reshape(1, y_values.size),
            (x_work.reshape(-1, x_work.shape[-1]).shape[0], y_values.size),
        )
        out_shape = (*x_work.shape[:-1],)
        restore_axis = source_axis.order
    else:
        y_work = np.moveaxis(y_values, source_axis.order, -1)
        y_flat = y_work.reshape(-1, y_work.shape[-1])
        if y_flat.shape[0] == 1 and x_work.reshape(-1, x_work.shape[-1]).shape[0] > 1:
            y_flat = np.broadcast_to(
                y_flat,
                (
                    x_work.reshape(-1, x_work.shape[-1]).shape[0],
                    y_flat.shape[-1],
                ),
            )
            out_shape = (*x_work.shape[:-1],)
        elif y_flat.shape[0] == x_work.reshape(-1, x_work.shape[-1]).shape[0]:
            out_shape = (*y_work.shape[:-1],)
        else:
            raise ValueError(
                f"entry '{entry.code_label}' cannot be remapped onto the target bias shape."
            )
        restore_axis = source_axis.order

    x_flat = x_work.reshape(-1, x_work.shape[-1])
    target_values = np.asarray(target_axis.values, dtype=np.float64)
    out = np.full((y_flat.shape[0], target_values.size), np.nan, dtype=np.float64)

    for index, (x_row, y_row) in enumerate(zip(x_flat, y_flat)):
        if np.all(np.diff(x_row) < 0.0):
            x_row = x_row[::-1]
            y_row = y_row[::-1]
        out[index] = np.interp(
            target_values,
            x_row,
            y_row,
            left=np.nan,
            right=np.nan,
        )

    reshaped = out.reshape(*out_shape, target_values.size)
    return np.moveaxis(reshaped, -1, restore_axis)


def _is_aligned_entry(
    sample: TransportDatasetSpec,
    entry: DataSpec | AxisSpec | ParamSpec,
    source_axis: AxisSpec,
) -> bool:
    array = np.asarray(entry.values, dtype=np.float64)
    if array.ndim == 0:
        return False
    source_len = np.asarray(source_axis.values, dtype=np.float64).size
    if isinstance(entry, AxisSpec):
        return entry.order == source_axis.order and array.size == source_len
    if array.ndim == 1:
        return array.size == source_len
    data_shape = sample._data_shape
    if tuple(array.shape) == data_shape:
        return array.shape[source_axis.order] == source_len
    return False


def _require_unambiguous_entry(
    sample: TransportDatasetSpec,
    entry: DataSpec | AxisSpec | ParamSpec,
    source_axis: AxisSpec,
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
    if broadcast.shape[source_axis.order] == np.asarray(source_axis.values).size:
        raise ValueError(
            f"entry '{entry.code_label}' is broadcastable to the switched dimension "
            "but cannot be transformed unambiguously from its stored shape."
        )


def _as_data(entry: DataSpec | AxisSpec | ParamSpec, values: NDArray64) -> DataSpec:
    return DataSpec(
        code_label=entry.code_label,
        print_label=entry.print_label,
        html_label=entry.html_label,
        latex_label=entry.latex_label,
        values=values,
    )


def source_transport_label(axis_label: str) -> str:
    if axis_label == "V_mV":
        return "I_nA"
    if axis_label == "I_nA":
        return "V_mV"
    raise ValueError("switch_bias only supports V_mV and I_nA transport axes.")


__all__ = ["switch_bias"]
