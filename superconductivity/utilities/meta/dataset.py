"""Label-addressable dataset container."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .axis import AxisSpec, axis
from .data import DataSpec, data
from .label import LabelSpec
from .param import ParamSpec, param


@dataclass(frozen=True, slots=True, init=False)
class Dataset:
    """Aggregate data payloads, axes, and parameters by code label."""

    data: tuple[DataSpec, ...] = field(default_factory=tuple)
    axes: tuple[AxisSpec, ...] = field(default_factory=tuple)
    params: tuple[ParamSpec, ...] = field(default_factory=tuple)
    _lookup: dict[str, LabelSpec] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        *,
        data: object = (),
        axes: object = (),
        params: object = (),
    ) -> None:
        normalized_data = _normalize_data_entries(data)
        normalized_axes = _normalize_axis_entries(axes)
        normalized_params = _normalize_param_entries(params)
        lookup = _build_lookup(
            normalized_data,
            normalized_axes,
            normalized_params,
        )
        object.__setattr__(self, "data", normalized_data)
        object.__setattr__(self, "axes", normalized_axes)
        object.__setattr__(self, "params", normalized_params)
        object.__setattr__(self, "_lookup", lookup)

    def __getattr__(self, name: str) -> LabelSpec:
        try:
            lookup = object.__getattribute__(self, "_lookup")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        try:
            return lookup[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> LabelSpec:
        return self._lookup[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._lookup

    def __iter__(self):
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._lookup)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._lookup)

    def items(self) -> tuple[tuple[str, LabelSpec], ...]:
        return tuple(self._lookup.items())

    def values(self) -> tuple[LabelSpec, ...]:
        return tuple(self._lookup.values())

    def add(self, **entries: object) -> Dataset:
        """Return a new dataset with additional entries merged in."""
        added = dataset(**entries)
        return Dataset(
            data=(*self.data, *added.data),
            axes=(*self.axes, *added.axes),
            params=(*self.params, *added.params),
        )

    def remove(self, *code_labels: str) -> Dataset:
        """Return a new dataset without the specified code labels."""
        labels = {str(label) for label in code_labels}
        return Dataset(
            data=tuple(
                entry for entry in self.data if entry.code_label not in labels
            ),
            axes=tuple(
                entry for entry in self.axes if entry.code_label not in labels
            ),
            params=tuple(
                entry for entry in self.params if entry.code_label not in labels
            ),
        )


def dataset(**entries: object) -> Dataset:
    """Construct a dataset from label-addressable keyword entries.

    Parameters
    ----------
    **entries
        Keyworded dataset entries. The special keyword ``data`` may contain
        one or more payload entries. Other keywords are interpreted by the
        type of their value:

        - :class:`DataSpec` -> payload entry
        - :class:`AxisSpec` -> axis entry
        - :class:`ParamSpec` -> parameter entry
        - scalar -> parameter entry
        - array-like -> data payload entry

    Returns
    -------
    Dataset
        The normalized dataset container.
    """
    data_entries: list[DataSpec] = []
    axis_entries: list[AxisSpec] = []
    param_entries: list[ParamSpec] = []

    explicit_data = entries.pop("data", None)
    data_entries.extend(_normalize_data_entries(explicit_data))

    for name, entry in entries.items():
        if isinstance(entry, AxisSpec):
            axis_entries.append(entry)
            continue
        if isinstance(entry, ParamSpec):
            param_entries.append(entry)
            continue
        if isinstance(entry, DataSpec):
            data_entries.append(entry)
            continue
        if np.isscalar(entry):
            param_entries.append(param(name, entry))
            continue
        data_entries.append(data(name, entry))

    return Dataset(
        data=tuple(data_entries),
        axes=tuple(axis_entries),
        params=tuple(param_entries),
    )


def validate_gridded_dataset(
    ds: Dataset,
    *,
    required_data: Iterable[str] | None = None,
    required_axes: Iterable[str] | None = None,
    required_params: Iterable[str] | None = None,
    finite_data: bool = False,
) -> None:
    """Validate that one dataset follows aligned-grid invariants."""
    if not ds.data:
        raise ValueError("gridded dataset requires at least one data entry.")
    if not ds.axes:
        raise ValueError("gridded dataset requires at least one axis entry.")

    data_shape = _single_shape(
        entries=ds.data,
        entry_kind="data",
        finite=finite_data,
        non_empty=True,
    )
    _validate_entries(
        entries=ds.axes,
        entry_kind="axis",
        finite=True,
        non_empty=True,
    )
    _validate_axis_orders(ds, data_shape=data_shape)

    _require_labels(ds, required_data or (), kind="data")
    _require_labels(ds, required_axes or (), kind="axis")
    _require_labels(ds, required_params or (), kind="param")
    _require_scalar_params(ds, required_params or ())


def gridded_dataset(
    *,
    required_data: Iterable[str] | None = None,
    required_axes: Iterable[str] | None = None,
    required_params: Iterable[str] | None = None,
    finite_data: bool = False,
    **entries: object,
) -> Dataset:
    """Construct and validate one aligned-grid dataset."""
    ds = dataset(**entries)
    validate_gridded_dataset(
        ds,
        required_data=required_data,
        required_axes=required_axes,
        required_params=required_params,
        finite_data=finite_data,
    )
    return ds


def _normalize_data_entries(
    value: object,
) -> tuple[DataSpec, ...]:
    if value is None:
        return ()
    if isinstance(value, DataSpec):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(_coerce_data_entry(name, entry) for name, entry in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        if all(isinstance(entry, DataSpec) for entry in value):
            return tuple(value)
        if all(_is_named_entry(entry) for entry in value):
            return tuple(
                _coerce_data_entry(str(name), entry_value)
                for name, entry_value in value
            )
        return tuple(
            _make_payload_entry(f"data_{index}", entry)
            for index, entry in enumerate(value)
        )
    return (_make_payload_entry("data", value),)


def _normalize_axis_entries(
    value: object,
) -> tuple[AxisSpec, ...]:
    if value is None:
        return ()
    if isinstance(value, AxisSpec):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(_coerce_axis_entry(name, entry) for name, entry in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        if all(isinstance(entry, AxisSpec) for entry in value):
            return tuple(value)
        if all(_is_named_entry(entry) for entry in value):
            return tuple(
                _coerce_axis_entry(str(name), entry_value)
                for name, entry_value in value
            )
    raise TypeError(
        "axes must be AxisSpec objects, named pairs, mappings, or empty.",
    )


def _normalize_param_entries(
    value: object,
) -> tuple[ParamSpec, ...]:
    if value is None:
        return ()
    if isinstance(value, ParamSpec):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(_coerce_param_entry(name, entry) for name, entry in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        if all(isinstance(entry, ParamSpec) for entry in value):
            return tuple(value)
        if all(_is_named_entry(entry) for entry in value):
            return tuple(
                _coerce_param_entry(str(name), entry_value)
                for name, entry_value in value
            )
    raise TypeError(
        "params must be ParamSpec objects, named pairs, mappings, or empty.",
    )


def _coerce_data_entry(name: str, entry: object) -> DataSpec:
    if isinstance(entry, DataSpec):
        if entry.code_label != name:
            raise ValueError(
                f"data entry label '{entry.code_label}' does not match '{name}'."
            )
        return entry
    return data(name, entry)  # type: ignore[arg-type]


def _coerce_axis_entry(name: str, entry: object) -> AxisSpec:
    if isinstance(entry, AxisSpec):
        if entry.code_label != name:
            raise ValueError(
                f"axis label '{entry.code_label}' does not match '{name}'."
            )
        return entry
    return axis(name, values=entry, order=0)  # type: ignore[arg-type]


def _coerce_param_entry(name: str, entry: object) -> ParamSpec:
    if isinstance(entry, ParamSpec):
        if entry.code_label != name:
            raise ValueError(
                f"param label '{entry.code_label}' does not match '{name}'."
            )
        return entry
    return param(name, entry)  # type: ignore[arg-type]


def _make_payload_entry(name: str, entry: object) -> DataSpec:
    if isinstance(entry, DataSpec):
        return entry
    return data(name, entry)  # type: ignore[arg-type]


def _build_lookup(
    data_entries: Sequence[DataSpec],
    axis_entries: Sequence[AxisSpec],
    param_entries: Sequence[ParamSpec],
) -> dict[str, LabelSpec]:
    lookup: dict[str, LabelSpec] = {}
    for entry in (*data_entries, *axis_entries, *param_entries):
        code_label = entry.code_label
        if code_label in lookup:
            raise ValueError(f"Duplicate code_label '{code_label}' in Dataset.")
        lookup[code_label] = entry
    return lookup


def _is_named_entry(entry: object) -> bool:
    return isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str)

def _single_shape(
    *,
    entries: Sequence[DataSpec | AxisSpec],
    entry_kind: str,
    finite: bool,
    non_empty: bool,
) -> tuple[int, ...]:
    first_shape: tuple[int, ...] | None = None
    for entry in entries:
        values = np.asarray(entry.values, dtype=np.float64)
        if non_empty and values.size == 0:
            raise ValueError(f"{entry_kind} '{entry.code_label}' must be non-empty.")
        if finite and np.any(~np.isfinite(values)):
            raise ValueError(f"{entry_kind} '{entry.code_label}' must be finite.")
        shape = tuple(values.shape)
        if first_shape is None:
            first_shape = shape
            continue
        if shape != first_shape:
            raise ValueError(
                f"all {entry_kind} entries must have identical shape."
            )
    if first_shape is None:
        raise ValueError(f"missing {entry_kind} entries.")
    return first_shape


def _validate_entries(
    *,
    entries: Sequence[DataSpec | AxisSpec],
    entry_kind: str,
    finite: bool,
    non_empty: bool,
) -> None:
    for entry in entries:
        values = np.asarray(entry.values, dtype=np.float64)
        if non_empty and values.size == 0:
            raise ValueError(f"{entry_kind} '{entry.code_label}' must be non-empty.")
        if finite and np.any(~np.isfinite(values)):
            raise ValueError(f"{entry_kind} '{entry.code_label}' must be finite.")


def _validate_axis_orders(ds: Dataset, *, data_shape: tuple[int, ...]) -> None:
    """Validate that each axis order targets a matching data dimension."""
    ndim = len(data_shape)
    for entry in ds.axes:
        axis_values = np.asarray(entry.values, dtype=np.float64)
        axis_order = int(entry.order)
        if axis_order < 0 or axis_order >= ndim:
            raise ValueError(
                f"axis '{entry.code_label}' order {axis_order} is out of bounds "
                f"for data ndim {ndim}."
            )
        if axis_values.ndim == 1:
            if axis_values.size != data_shape[axis_order]:
                raise ValueError(
                    f"axis '{entry.code_label}' length {axis_values.size} does not "
                    f"match data dimension {axis_order} size {data_shape[axis_order]}."
                )
            continue
        if tuple(axis_values.shape) != data_shape:
            raise ValueError(
                f"axis '{entry.code_label}' must be 1D with length matching "
                "its order dimension or have full data shape."
            )


def _require_labels(ds: Dataset, labels: Iterable[str], *, kind: str) -> None:
    if kind == "data":
        available = {entry.code_label for entry in ds.data}
    elif kind == "axis":
        available = {entry.code_label for entry in ds.axes}
    elif kind == "param":
        available = {entry.code_label for entry in ds.params}
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(f"unknown kind '{kind}'.")
    missing = sorted(set(labels) - available)
    if missing:
        raise ValueError(f"missing required {kind} labels: {', '.join(missing)}")


def _require_scalar_params(ds: Dataset, labels: Iterable[str]) -> None:
    by_label = {entry.code_label: entry for entry in ds.params}
    for label_name in labels:
        entry = by_label.get(label_name)
        if entry is None:
            continue
        values = np.asarray(entry.values, dtype=np.float64)
        if values.ndim != 0:
            raise ValueError(
                f"required param '{label_name}' must be scalar-valued."
            )


__all__ = [
    "Dataset",
    "dataset",
    "validate_gridded_dataset",
    "gridded_dataset",
]
