"""Typed reduced transport dataset helpers."""

from __future__ import annotations

import numpy as np

from ..constants import G0_muS
from ..types import NDArray64
from .axis import AxisSpec
from .data import DataSpec
from .dataset import Dataset, dataset, validate_gridded_dataset
from .label import label
from .param import ParamSpec

_G0_uS = float(G0_muS)
_DERIVED_LABELS = (
    "dG_uS",
    "dR_MOhm",
    "G_uS",
    "R_MOhm",
    "eV_Delta",
    "eI_DeltaG0",
    "eI_DeltaGN",
    "dG_G0",
    "dG_GN",
    "dR_R0",
    "dR_RN",
    "G_G0",
    "G_GN",
    "R_R0",
    "R_RN",
    "eA_hnu",
    "eA_Delta",
    "hnu_Delta",
    "Tc_K",
    "T_Tc",
    "DeltaT_meV",
    "DeltaT_Delta",
    "gamma_Delta",
    "sigmaV_Delta",
)


class TransportDatasetSpec(Dataset):
    """Validated transport dataset with lazy derived properties."""

    def __init__(
        self,
        *,
        data: object = (),
        axes: object = (),
        params: object = (),
    ) -> None:
        super().__init__(data=data, axes=axes, params=params)
        validate_reduced_dataset(self)

    def add(self, **entries: object) -> TransportDatasetSpec:
        """Return a new transport dataset with additional entries merged in."""
        added = dataset(**entries)
        return TransportDatasetSpec(
            data=(*self.data, *added.data),
            axes=(*self.axes, *added.axes),
            params=(*self.params, *added.params),
        )

    def remove(self, *code_labels: str) -> TransportDatasetSpec:
        """Return a new transport dataset without the specified code labels."""
        labels = {str(label) for label in code_labels}
        return TransportDatasetSpec(
            data=tuple(entry for entry in self.data if entry.code_label not in labels),
            axes=tuple(entry for entry in self.axes if entry.code_label not in labels),
            params=tuple(
                entry for entry in self.params if entry.code_label not in labels
            ),
        )

    def __getitem__(self, key: str):
        if key in self._lookup:
            return self._lookup[key]
        try:
            return self._derived_entries()[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (
            key in self._lookup or key in self._derived_entries()
        )

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> tuple[str, ...]:
        return tuple((*self._lookup.keys(), *self._derived_entries().keys()))

    def items(self) -> tuple[tuple[str, object], ...]:
        merged = dict(self._lookup)
        merged.update(self._derived_entries())
        return tuple(merged.items())

    def values(self) -> tuple[object, ...]:
        return tuple(value for _, value in self.items())

    @property
    def dG_uS(self) -> DataSpec:
        if self._transport_axis.code_label == "V_mV":
            values = np.gradient(
                self._transport_data_values,
                self._transport_axis_values,
                axis=self._transport_axis.order,
            )
            return self._derived_data("dG_uS", values)
        return self._derived_data("dG_uS", self._safe_reciprocal(self.dR_MOhm))

    @property
    def dR_MOhm(self) -> DataSpec:
        if self._transport_axis.code_label == "I_nA":
            values = np.gradient(
                self._transport_data_values,
                self._transport_axis_values,
                axis=self._transport_axis.order,
            )
            return self._derived_data("dR_MOhm", values)
        return self._derived_data("dR_MOhm", self._safe_reciprocal(self.dG_uS))

    @property
    def G_uS(self) -> DataSpec:
        return self._derived_data(
            "G_uS",
            self._safe_divide(self._current_values, self._voltage_values),
        )

    @property
    def R_MOhm(self) -> DataSpec:
        return self._derived_data(
            "R_MOhm",
            self._safe_divide(self._voltage_values, self._current_values),
        )

    @property
    def eV_Delta(self) -> DataSpec | ParamSpec:
        return self._derived_binary_value(
            "eV_Delta",
            "V_mV",
            "Delta_meV",
            self._safe_divide,
        )

    @property
    def eI_DeltaG0(self) -> DataSpec | ParamSpec:
        return self._derived_value(
            "eI_DeltaG0",
            self._safe_divide(self._current_values, self._Delta_values * _G0_uS),
        )

    @property
    def eI_DeltaGN(self) -> DataSpec | ParamSpec:
        return self._derived_value(
            "eI_DeltaGN",
            self._safe_divide(
                self._current_values,
                self._Delta_values * self._conductance_values,
            ),
        )

    @property
    def dG_G0(self) -> DataSpec:
        return self._derived_data("dG_G0", self._safe_divide(self.dG_uS, _G0_uS))

    @property
    def dG_GN(self) -> DataSpec:
        return self._derived_data(
            "dG_GN",
            self._safe_divide(self.dG_uS, self._conductance_values),
        )

    @property
    def dR_R0(self) -> DataSpec:
        return self._derived_data(
            "dR_R0",
            np.asarray(self.dR_MOhm, dtype=np.float64) * _G0_uS,
        )

    @property
    def dR_RN(self) -> DataSpec:
        return self._derived_data(
            "dR_RN",
            np.asarray(self.dR_MOhm, dtype=np.float64) * self._conductance_values,
        )

    @property
    def G_G0(self) -> DataSpec:
        return self._derived_data("G_G0", self._safe_divide(self.G_uS, _G0_uS))

    @property
    def G_GN(self) -> DataSpec:
        return self._derived_data(
            "G_GN",
            self._safe_divide(self.G_uS, self._conductance_values),
        )

    @property
    def R_R0(self) -> DataSpec:
        return self._derived_data(
            "R_R0",
            np.asarray(self.R_MOhm, dtype=np.float64) * _G0_uS,
        )

    @property
    def R_RN(self) -> DataSpec:
        return self._derived_data(
            "R_RN",
            np.asarray(self.R_MOhm, dtype=np.float64) * self._conductance_values,
        )

    @property
    def eA_hnu(self) -> DataSpec | ParamSpec:
        return self._derived_binary_value(
            "eA_hnu",
            "A_mV",
            "nu_GHz",
            self._safe_divide,
        )

    @property
    def eA_Delta(self) -> DataSpec | ParamSpec:
        return self._derived_binary_value(
            "eA_Delta",
            "A_mV",
            "Delta_meV",
            self._safe_divide,
        )

    @property
    def hnu_Delta(self) -> DataSpec | ParamSpec:
        return self._derived_binary_value(
            "hnu_Delta",
            "nu_GHz",
            "Delta_meV",
            self._safe_divide,
        )

    @property
    def Tc_K(self) -> DataSpec:
        from ...models.basics import get_Tc_K

        return self._derived_entry(
            "Tc_K",
            np.vectorize(get_Tc_K, otypes=[np.float64])(
                np.asarray(self._find_entry("Delta_meV").values, dtype=np.float64)
            ),
            self._find_entry("Delta_meV"),
        )

    @property
    def T_Tc(self) -> DataSpec | ParamSpec:
        return self._derived_binary_value(
            "T_Tc",
            "T_K",
            "Tc_K",
            self._safe_divide,
        )

    @property
    def DeltaT_meV(self) -> DataSpec:
        from ...models.basics import get_DeltaT_meV

        return self._derived_binary_value(
            "DeltaT_meV",
            "Delta_meV",
            "T_K",
            lambda delta, temperature: np.vectorize(
                get_DeltaT_meV,
                otypes=[np.float64],
            )(delta, temperature),
        )

    @property
    def DeltaT_Delta(self) -> DataSpec | ParamSpec:
        return self._derived_binary_value(
            "DeltaT_Delta",
            "DeltaT_meV",
            "Delta_meV",
            self._safe_divide,
        )

    @property
    def sigmaV_Delta(self) -> DataSpec | ParamSpec:
        sigma_entry = self._find_entry("sigmaV_mV")
        if sigma_entry is None:
            raise ValueError("sigmaV_Delta requires a 'sigmaV_mV' axis or param entry.")
        return self._derived_binary_value(
            "sigmaV_Delta",
            "sigmaV_mV",
            "Delta_meV",
            self._safe_divide,
        )

    @property
    def gamma_Delta(self) -> DataSpec:
        gamma_entry = self._find_entry("gamma_meV")
        if gamma_entry is None:
            raise ValueError("gamma_Delta requires a 'gamma_meV' axis or param entry.")
        return self._derived_binary_value(
            "gamma_Delta",
            "gamma_meV",
            "Delta_meV",
            self._safe_divide,
        )

    @property
    def _transport_axis(self) -> AxisSpec:
        axis_entry = self._find_axis("I_nA")
        if axis_entry is not None:
            return axis_entry
        axis_entry = self._find_axis("V_mV")
        if axis_entry is not None:
            return axis_entry
        raise AttributeError("TransportDatasetSpec is missing its transport axis.")

    @property
    def _transport_data(self) -> DataSpec:
        data_entry = self._find_data("I_nA")
        if data_entry is not None:
            return data_entry
        data_entry = self._find_data("V_mV")
        if data_entry is not None:
            return data_entry
        raise AttributeError("TransportDatasetSpec is missing its transport data.")

    @property
    def _transport_axis_values(self) -> NDArray64:
        return np.asarray(self._transport_axis.values, dtype=np.float64)

    @property
    def _transport_data_values(self) -> NDArray64:
        return np.asarray(self._transport_data.values, dtype=np.float64)

    @property
    def _data_shape(self) -> tuple[int, ...]:
        return tuple(self._transport_data_values.shape)

    @property
    def _current_values(self) -> NDArray64:
        entry = self._find_entry("I_nA")
        if entry is None:
            raise AttributeError("TransportDatasetSpec is missing 'I_nA'.")
        return self._entry_values_full(entry)

    @property
    def _voltage_values(self) -> NDArray64:
        entry = self._find_entry("V_mV")
        if entry is None:
            raise AttributeError("TransportDatasetSpec is missing 'V_mV'.")
        return self._entry_values_full(entry)

    @property
    def _Delta_values(self) -> NDArray64:
        entry = self._find_entry("Delta_meV")
        if entry is None:
            raise AttributeError("TransportDatasetSpec is missing 'Delta_meV'.")
        return self._entry_values_full(entry)

    @property
    def _amplitude_values(self) -> NDArray64:
        entry = self._find_entry("A_mV")
        if entry is None:
            raise AttributeError("TransportDatasetSpec is missing 'A_mV'.")
        return self._entry_values_full(entry)

    @property
    def _frequency_values(self) -> NDArray64:
        entry = self._find_entry("nu_GHz")
        if entry is None:
            raise AttributeError("TransportDatasetSpec is missing 'nu_GHz'.")
        return self._entry_values_full(entry)

    @property
    def _temperature_values(self) -> NDArray64:
        entry = self._find_entry("T_K")
        if entry is None:
            raise AttributeError("TransportDatasetSpec is missing 'T_K'.")
        return self._entry_values_full(entry)

    @property
    def _conductance_values(self) -> NDArray64:
        entry = self._find_entry("GN_G0")
        if entry is None:
            raise AttributeError("TransportDatasetSpec is missing 'GN_G0'.")
        return self._entry_values_full(entry) * _G0_uS

    def _derived_data(self, code_label: str, values: object) -> DataSpec:
        meta = label(code_label)
        return DataSpec(
            code_label=meta.code_label,
            print_label=meta.print_label,
            html_label=meta.html_label,
            latex_label=meta.latex_label,
            values=values,
        )

    def _derived_axis(
        self,
        code_label: str,
        values: object,
        axis_entry: AxisSpec,
    ) -> AxisSpec:
        meta = label(code_label)
        return AxisSpec(
            code_label=meta.code_label,
            print_label=meta.print_label,
            html_label=meta.html_label,
            latex_label=meta.latex_label,
            values=values,
            order=axis_entry.order,
        )

    def _derived_param(self, code_label: str, values: object) -> ParamSpec:
        meta = label(code_label)
        return ParamSpec(
            code_label=meta.code_label,
            print_label=meta.print_label,
            html_label=meta.html_label,
            latex_label=meta.latex_label,
            values=values,
        )

    def _derived_entry(
        self,
        code_label: str,
        values: object,
        source: AxisSpec | DataSpec | ParamSpec | None = None,
    ) -> AxisSpec | DataSpec | ParamSpec:
        array = np.asarray(values, dtype=np.float64)
        if isinstance(source, AxisSpec):
            return self._derived_axis(code_label, array.reshape(-1), source)
        if isinstance(source, ParamSpec) and array.ndim == 0:
            return self._derived_param(code_label, float(array))
        if array.ndim == 0:
            return self._derived_param(code_label, float(array))
        return self._derived_data(code_label, array)

    def _derived_value(
        self,
        code_label: str,
        values: object,
    ) -> AxisSpec | DataSpec | ParamSpec:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            return self._derived_param(code_label, float(array))
        return self._derived_data(code_label, array)

    def _derived_binary_value(
        self,
        code_label: str,
        left_label: str,
        right_label: str,
        op,
    ) -> AxisSpec | DataSpec | ParamSpec:
        left = self._get_entry_or_derived(left_label)
        right = self._get_entry_or_derived(right_label)
        if left is None or right is None:
            raise AttributeError(
                f"{code_label} requires '{left_label}' and '{right_label}'."
            )
        source_axis = self._single_axis_source(left, right)
        if source_axis is not None:
            values = op(
                np.asarray(source_axis.values, dtype=np.float64),
                self._other_scalar_value(left, right, source_axis),
            )
            return self._derived_axis(code_label, values, source_axis)
        if self._all_scalar_sources(left, right):
            values = op(
                np.asarray(left.values, dtype=np.float64),
                np.asarray(right.values, dtype=np.float64),
            )
            return self._derived_param(code_label, float(np.asarray(values)))
        values = op(
            self._entry_values_full(left),
            self._entry_values_full(right),
        )
        return self._derived_data(code_label, values)

    def _single_axis_source(
        self,
        *entries: AxisSpec | DataSpec | ParamSpec,
    ) -> AxisSpec | None:
        if any(
            isinstance(entry, DataSpec) and not isinstance(entry, (AxisSpec, ParamSpec))
            for entry in entries
        ):
            return None
        axis_entries = [entry for entry in entries if isinstance(entry, AxisSpec)]
        if len(axis_entries) != 1:
            return None
        if any(
            isinstance(entry, ParamSpec)
            and np.asarray(entry.values, dtype=np.float64).ndim != 0
            for entry in entries
            if not isinstance(entry, AxisSpec)
        ):
            return None
        return axis_entries[0]

    def _all_scalar_sources(
        self,
        *entries: AxisSpec | DataSpec | ParamSpec,
    ) -> bool:
        return all(
            np.asarray(entry.values, dtype=np.float64).ndim == 0 for entry in entries
        )

    def _other_scalar_value(
        self,
        left: AxisSpec | DataSpec | ParamSpec,
        right: AxisSpec | DataSpec | ParamSpec,
        axis_entry: AxisSpec,
    ) -> NDArray64:
        other = right if left is axis_entry else left
        return np.asarray(other.values, dtype=np.float64)

    def _derived_entries(self) -> dict[str, object]:
        entries: dict[str, object] = {}
        for name in _DERIVED_LABELS:
            if name in self._lookup:
                continue
            try:
                entries[name] = getattr(self, name)
            except (AttributeError, ValueError):
                continue
        return entries

    def _find_axis(self, code_label: str) -> AxisSpec | None:
        return next(
            (entry for entry in self.axes if entry.code_label == code_label),
            None,
        )

    def _find_data(self, code_label: str) -> DataSpec | None:
        return next(
            (entry for entry in self.data if entry.code_label == code_label),
            None,
        )

    def _find_param(self, code_label: str) -> ParamSpec | None:
        return next(
            (entry for entry in self.params if entry.code_label == code_label),
            None,
        )

    def _find_entry(self, code_label: str) -> AxisSpec | DataSpec | ParamSpec | None:
        return (
            self._find_axis(code_label)
            or self._find_data(code_label)
            or self._find_param(code_label)
        )

    def _get_entry_or_derived(
        self,
        code_label: str,
    ) -> AxisSpec | DataSpec | ParamSpec | None:
        entry = self._find_entry(code_label)
        if entry is not None:
            return entry
        if code_label in _DERIVED_LABELS:
            derived = getattr(self, code_label, None)
            if isinstance(derived, (AxisSpec, DataSpec, ParamSpec)):
                return derived
        return None

    def _entry_values_full(
        self,
        entry: AxisSpec | DataSpec | ParamSpec,
    ) -> NDArray64:
        values = np.asarray(entry.values, dtype=np.float64)
        if values.ndim == 0:
            return np.broadcast_to(values, self._data_shape).astype(np.float64)
        if tuple(values.shape) == self._data_shape:
            return values.astype(np.float64)
        if isinstance(entry, AxisSpec):
            reshape = [1] * len(self._data_shape)
            reshape[entry.order] = values.size
            reshaped = values.reshape(tuple(reshape))
            return np.broadcast_to(reshaped, self._data_shape).astype(np.float64)
        return np.broadcast_to(values, self._data_shape).astype(np.float64)

    def _safe_divide(
        self,
        numerator: float | NDArray64,
        denominator: float | NDArray64,
    ) -> float | NDArray64:
        numerator_arr = np.asarray(numerator, dtype=np.float64)
        denominator_arr = np.asarray(denominator, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                denominator_arr != 0.0,
                numerator_arr / denominator_arr,
                np.nan,
            )
        return _as_output(result)

    def _safe_reciprocal(self, value: float | NDArray64) -> float | NDArray64:
        value_arr = np.asarray(value, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(value_arr != 0.0, 1.0 / value_arr, np.nan)
        return _as_output(result)


def reduced_dataset(**entries: object) -> TransportDatasetSpec:
    """Construct and validate a transport dataset."""
    ds = dataset(**entries)
    return TransportDatasetSpec(data=ds.data, axes=ds.axes, params=ds.params)


def validate_reduced_dataset(ds: Dataset) -> None:
    """Validate the reduced transport contract on top of ``Dataset``."""
    _require_entry(ds, "I_nA")
    _require_entry(ds, "V_mV")

    i_axis = _has_axis(ds, "I_nA")
    v_axis = _has_axis(ds, "V_mV")
    i_data = _has_data(ds, "I_nA")
    v_data = _has_data(ds, "V_mV")

    if i_axis and v_axis:
        raise ValueError("Exactly one of 'I_nA' and 'V_mV' must be a data entry.")
    if i_data and v_data:
        raise ValueError("Exactly one of 'I_nA' and 'V_mV' must be an axis entry.")
    if i_axis == v_axis:
        raise ValueError("Exactly one of 'I_nA' and 'V_mV' must be an axis entry.")
    if i_data == v_data:
        raise ValueError("Exactly one of 'I_nA' and 'V_mV' must be a data entry.")

    transport_axis = _find_axis(ds, "I_nA") or _find_axis(ds, "V_mV")
    transport_data = _find_data(ds, "I_nA") or _find_data(ds, "V_mV")
    if transport_axis is None or transport_data is None:
        raise ValueError(
            "TransportDatasetSpec requires one transport axis and one transport data entry."
        )

    validate_gridded_dataset(ds)

    data_shape = tuple(np.asarray(transport_data.values, dtype=np.float64).shape)
    axis_values = np.asarray(transport_axis.values, dtype=np.float64)
    if axis_values.size != data_shape[transport_axis.order]:
        raise ValueError(
            f"axis '{transport_axis.code_label}' length {axis_values.size} does not "
            f"match data dimension {transport_axis.order} size "
            f"{data_shape[transport_axis.order]}."
        )

    for code_label in ("Delta_meV", "GN_G0", "A_mV", "nu_GHz"):
        entry = _find_entry(ds, code_label)
        if entry is None:
            continue
        _ensure_broadcastable(entry, data_shape=data_shape, code_label=code_label)

    for code_label in ("T_K", "T_Tc"):
        entry = _find_entry(ds, code_label)
        if entry is None:
            continue
        _ensure_broadcastable(entry, data_shape=data_shape, code_label=code_label)


def _require_entry(ds: Dataset, code_label: str) -> None:
    if _find_entry(ds, code_label) is None:
        raise ValueError(f"TransportDatasetSpec requires '{code_label}'.")


def _has_axis(ds: Dataset, code_label: str) -> bool:
    return _find_axis(ds, code_label) is not None


def _has_data(ds: Dataset, code_label: str) -> bool:
    return _find_data(ds, code_label) is not None


def _find_axis(ds: Dataset, code_label: str) -> AxisSpec | None:
    return next(
        (entry for entry in ds.axes if entry.code_label == code_label),
        None,
    )


def _find_data(ds: Dataset, code_label: str) -> DataSpec | None:
    return next(
        (entry for entry in ds.data if entry.code_label == code_label),
        None,
    )


def _find_param(ds: Dataset, code_label: str) -> ParamSpec | None:
    return next(
        (entry for entry in ds.params if entry.code_label == code_label),
        None,
    )


def _find_entry(ds: Dataset, code_label: str) -> AxisSpec | DataSpec | ParamSpec | None:
    return (
        _find_axis(ds, code_label)
        or _find_data(ds, code_label)
        or _find_param(ds, code_label)
    )


def _ensure_broadcastable(
    entry: AxisSpec | DataSpec | ParamSpec,
    *,
    data_shape: tuple[int, ...],
    code_label: str,
) -> None:
    values = np.asarray(entry.values, dtype=np.float64)
    if values.ndim == 0:
        return
    if isinstance(entry, AxisSpec):
        if values.size != data_shape[entry.order]:
            raise ValueError(
                f"'{code_label}' axis length {values.size} does not match "
                f"data dimension {entry.order} size {data_shape[entry.order]}."
            )
        return
    try:
        np.broadcast_to(values, data_shape)
    except ValueError as exc:
        raise ValueError(
            f"'{code_label}' values are not broadcast-compatible with data shape."
        ) from exc


def _as_output(values: NDArray64) -> float | NDArray64:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return array


__all__ = [
    "TransportDatasetSpec",
    "reduced_dataset",
    "validate_reduced_dataset",
]
