"""Helpers for extracting traces from HDF5 data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from ...utilities.meta import AxisSpec, Dataset, DataSpec, axis, data
from ...utilities.meta.label import LabelSpec
from ...utilities.safety import require_all_finite
from ...utilities.types import NDArray64
from .file import FileSpec, _import_h5py, _require_measurement, _to_measurement_path
from .keys import Keys, KeysSpec, _coerce_numeric_yvalues, get_keys


@dataclass(slots=True)
class TraceSpec:
    """Configuration for loading traces from one measurement."""

    amp_voltage: float = 1.0
    amp_current: float = 1.0
    r_ref_ohm: float = 51.689e3
    trigger_values: int | Sequence[int] | None = 1
    skip: int | tuple[int, int] = 0
    subtract_offset: bool = True
    time_relative: bool = True

    def keys(self) -> tuple[str, ...]:
        """Return public mapping-style keys."""
        return (
            "amp_voltage",
            "amp_current",
            "r_ref_ohm",
            "trigger_values",
            "skip",
            "subtract_offset",
            "time_relative",
        )


@dataclass(frozen=True, slots=True, init=False)
class Trace(Dataset):
    """One trace stored as a dataset."""

    def __init__(
        self,
        *,
        I_nA: NDArray64,
        V_mV: NDArray64,
        t_s: NDArray64,
    ) -> None:
        trace_ds = Dataset(
            data=(
                data("I_nA", I_nA),
                data("V_mV", V_mV),
            ),
            axes=(axis("t_s", values=t_s, order=0),),
        )
        object.__setattr__(self, "data", trace_ds.data)
        object.__setattr__(self, "axes", trace_ds.axes)
        object.__setattr__(self, "params", trace_ds.params)
        object.__setattr__(self, "_lookup", trace_ds._lookup)

    def __getitem__(self, key: str):
        return Dataset.__getitem__(self, key)

    @property
    def t_s(self) -> AxisSpec:
        return Dataset.__getitem__(self, "t_s")

    @property
    def I_nA(self) -> DataSpec:
        return Dataset.__getitem__(self, "I_nA")

    @property
    def V_mV(self) -> DataSpec:
        return Dataset.__getitem__(self, "V_mV")


@dataclass(slots=True)
class Traces:
    """Container for multiple traces."""

    traces: list[Trace]
    y: AxisSpec | None = field(init=False, default=None)
    index: AxisSpec | None = field(init=False, default=None)
    skeys: tuple[str, ...] = field(init=False, default_factory=tuple)
    _indices: NDArray64 = field(init=False)

    def __post_init__(self) -> None:
        """Build list views from ``traces``."""
        for trace in self.traces:
            if not isinstance(trace, Trace):
                raise TypeError("Traces accepts Trace objects only.")

        object.__setattr__(self, "skeys", tuple())
        object.__setattr__(
            self,
            "_indices",
            np.arange(len(self.traces), dtype=np.float64),
        )
        object.__setattr__(self, "index", _build_index_axis(self._indices))
        object.__setattr__(self, "y", None)

    @classmethod
    def from_fields(
        cls,
        *,
        traces: list[Trace],
        specific_keys: Sequence[str],
        indices: Sequence[int],
        yvalues: Sequence[object],
        y_label: LabelSpec | None,
    ) -> Traces:
        collection = cls(traces=traces)
        object.__setattr__(collection, "skeys", tuple(specific_keys))
        object.__setattr__(
            collection,
            "_indices",
            np.asarray(indices, dtype=np.float64).reshape(-1),
        )
        object.__setattr__(
            collection,
            "index",
            _build_index_axis(np.asarray(indices, dtype=np.float64).reshape(-1)),
        )
        object.__setattr__(
            collection,
            "y",
            _build_y_axis(
                values=_coerce_numeric_yvalues(indices=indices, yvalues=yvalues),
                specific_keys=specific_keys,
                label_spec=y_label,
            ),
        )
        return collection

    def __len__(self) -> int:
        """Return number of traces."""
        return len(self.traces)

    def __iter__(self) -> Iterator[Trace]:
        """Iterate over traces."""
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice,
    ) -> Trace | list[Trace]:
        """Return trace(s) by positional index."""
        return self.traces[index]

    def keys(self) -> tuple[str, ...]:
        """Return public mapping-style keys."""
        keys = ["y", "i", "indices", "skeys", "specific_keys", "t_s", "I_nA", "V_mV"]
        if self.y is not None and self.y.code_label != "y":
            keys.insert(1, self.y.code_label)
        return tuple(keys)

    @property
    def specific_keys(self) -> list[str]:
        """Return ordered specific keys."""
        return list(self.skeys)

    @property
    def indices(self) -> NDArray64:
        """Return ordered positional indices."""
        return np.asarray(self._indices, dtype=np.float64)

    @property
    def yvalues(self) -> NDArray64:
        """Return ordered y-values."""
        if self.y is None:
            return np.asarray(self._indices, dtype=np.float64)
        return np.asarray(self.y.values, dtype=np.float64)

    @property
    def I_nA(self) -> list[DataSpec]:
        """Return per-trace current arrays as data specs."""
        return [trace.I_nA for trace in self.traces]

    @property
    def V_mV(self) -> list[DataSpec]:
        """Return per-trace voltage arrays as data specs."""
        return [trace.V_mV for trace in self.traces]

    @property
    def t_s(self) -> list[AxisSpec]:
        """Return per-trace time axes."""
        return [trace.t_s for trace in self.traces]

    @property
    def i(self) -> AxisSpec | None:
        """Return the positional index axis."""
        return self.index

    def __getattr__(self, name: str):
        if self.y is not None and name == self.y.code_label:
            return self.y
        raise AttributeError(name)


def _normalize_skip(
    skip: int | tuple[int, int],
) -> tuple[int, int]:
    """Normalize symmetric or explicit edge trimming."""
    if isinstance(skip, int):
        if skip < 0:
            raise ValueError("skip must be >= 0.")
        return skip, skip

    if (
        isinstance(skip, tuple)
        and len(skip) == 2
        and isinstance(skip[0], int)
        and isinstance(skip[1], int)
    ):
        skip_start, skip_end = int(skip[0]), int(skip[1])
        if skip_start < 0 or skip_end < 0:
            raise ValueError("skip values must be >= 0.")
        return skip_start, skip_end

    raise ValueError("skip must be int or tuple[int, int].")


def _normalize_keys_and_yvalues(
    keys: Sequence[str],
    yvalues: Sequence[object] | np.ndarray,
) -> tuple[list[str], list[object]]:
    """Normalize and validate collection metadata."""
    keys_list = list(keys)
    if len(keys_list) == 0:
        raise ValueError("keys must not be empty.")
    for key in keys_list:
        if not isinstance(key, str) or key == "":
            raise ValueError("keys must contain only non-empty strings.")

    yvalues_array = np.asarray(yvalues, dtype=object).reshape(-1)
    if yvalues_array.size == 0:
        raise ValueError("yvalues must not be empty.")
    if yvalues_array.size != len(keys_list):
        raise ValueError("keys and yvalues must have the same length.")
    return keys_list, yvalues_array.tolist()


def _normalize_index(
    index: int,
    size: int,
) -> int:
    """Normalize one possibly-negative index."""
    idx = int(index)
    if idx < 0:
        idx += size
    if idx < 0 or idx >= size:
        raise IndexError(
            f"index {index} is out of range for {size} traces.",
        )
    return idx


def _find_index_for_value(
    yvalues: NDArray64,
    yvalue: float,
) -> int:
    """Return the unique trace index for one y-value."""
    atol = np.finfo(np.float64).eps * max(1.0, abs(yvalue)) * 8.0
    matches = np.flatnonzero(
        np.isclose(yvalues, yvalue, rtol=0.0, atol=atol),
    )
    if matches.size == 0:
        raise KeyError(f"yvalue {yvalue!r} was not found.")
    if matches.size > 1:
        raise ValueError(
            f"yvalue {yvalue!r} matches multiple traces. Use index or specific_key.",
        )
    return int(matches[0])


def _resolve_trace_reference(
    keys: Keys,
    specific_key: str | None = None,
    yvalue: float | None = None,
    index: int | None = None,
) -> tuple[str, int | None, float]:
    """Resolve selectors into ``(specific_key, index, yvalue)``."""
    if specific_key is None and yvalue is None and index is None:
        raise ValueError(
            "Provide at least one of specific_key, yvalue, or index.",
        )

    keys_sorted = list(keys.specific_keys)
    yvalues_sorted = np.asarray(keys.yvalues, dtype=np.float64)

    resolved_key = specific_key
    resolved_index: int | None = None
    resolved_value: float | None = None

    if index is not None:
        assert keys_sorted is not None
        assert yvalues_sorted is not None
        resolved_index = _normalize_index(index, len(keys_sorted))
        key_from_index = keys_sorted[resolved_index]
        if resolved_key is not None and resolved_key != key_from_index:
            raise ValueError(
                "specific_key and index refer to different traces.",
            )
        resolved_key = key_from_index
        resolved_value = float(yvalues_sorted[resolved_index])

    if yvalue is not None:
        value = float(yvalue)
        if not np.isfinite(value):
            raise ValueError("yvalue must be finite.")

        assert keys_sorted is not None
        assert yvalues_sorted is not None
        index_from_value = _find_index_for_value(yvalues_sorted, value)
        key_from_value = keys_sorted[index_from_value]

        if resolved_key is not None and resolved_key != key_from_value:
            raise ValueError(
                "specific_key and yvalue refer to different traces.",
            )
        if resolved_index is not None and resolved_index != index_from_value:
            raise ValueError(
                "index and yvalue refer to different traces.",
            )

        resolved_key = key_from_value
        resolved_index = index_from_value
        resolved_value = float(yvalues_sorted[index_from_value])

    if resolved_key is None:
        raise ValueError(
            "Could not resolve a trace. Provide specific_key, yvalue, or index.",
        )

    if resolved_value is None and specific_key is not None:
        try:
            key_index = keys_sorted.index(specific_key)
        except ValueError:
            resolved_value = float(_normalize_index(0, 1))
        else:
            resolved_value = float(yvalues_sorted[key_index])

    return resolved_key, resolved_index, resolved_value


def _load_trace_from_file(
    file: object,
    full_path: str,
    amp_voltage: float,
    amp_current: float,
    r_ref_ohm: float,
    trigger_values: int | Sequence[int] | None,
    skip: tuple[int, int],
    subtract_offset: bool,
    time_relative: bool,
) -> Trace:
    """Load one trace from one open HDF5 file."""
    if full_path not in file:
        raise KeyError(f"Dataset path not found: '{full_path}'.")

    sweep = file[f"{full_path}/sweep/adwin"]
    t_s = np.asarray(sweep["time"], dtype=np.float64)
    v1_sweep_V = np.asarray(sweep["V1"], dtype=np.float64)
    v2_sweep_V = np.asarray(sweep["V2"], dtype=np.float64)
    trigger = np.asarray(sweep["trigger"])

    if trigger_values is not None:
        trigger_keep = np.atleast_1d(np.asarray(trigger_values))
        require_all_finite(trigger_keep, name="trigger_values")
        mask = np.isin(trigger, trigger_keep)
        t_s = t_s[mask]
        v1_sweep_V = v1_sweep_V[mask]
        v2_sweep_V = v2_sweep_V[mask]

    skip_start, skip_end = skip
    n_total = t_s.size
    n_trim = skip_start + skip_end
    if n_trim > 0:
        if n_trim >= n_total:
            raise ValueError(
                "skip removes all points. Reduce skip or use another trace.",
            )
        end_idx = None if skip_end == 0 else -skip_end
        t_s = t_s[skip_start:end_idx]
        v1_sweep_V = v1_sweep_V[skip_start:end_idx]
        v2_sweep_V = v2_sweep_V[skip_start:end_idx]

    V_mV = v1_sweep_V / float(amp_voltage) * 1e3
    I_nA = v2_sweep_V / (float(amp_current) * float(r_ref_ohm)) * 1e9

    if subtract_offset:
        offset = file[f"{full_path}/offset/adwin"]
        v1_offset_V = np.asarray(offset["V1"], dtype=np.float64)
        v2_offset_V = np.asarray(offset["V2"], dtype=np.float64)
        v_offset_mV = np.nanmean(v1_offset_V / float(amp_voltage) * 1e3)
        i_offset_nA = np.nanmean(
            v2_offset_V / (float(amp_current) * float(r_ref_ohm)) * 1e9,
        )
        V_mV = V_mV - float(v_offset_mV)
        I_nA = I_nA - float(i_offset_nA)

    finite = np.isfinite(t_s) & np.isfinite(V_mV) & np.isfinite(I_nA)
    if np.any(~finite):
        t_s = t_s[finite]
        V_mV = V_mV[finite]
        I_nA = I_nA[finite]

    if time_relative and t_s.size > 0:
        t_s = t_s - t_s[0]

    return Trace(
        I_nA=np.asarray(I_nA, dtype=np.float64),
        V_mV=np.asarray(V_mV, dtype=np.float64),
        t_s=np.asarray(t_s, dtype=np.float64),
    )


def _build_y_axis(
    values: NDArray64,
    specific_keys: Sequence[str],
    label_spec: LabelSpec | None,
) -> AxisSpec | None:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        numeric.size < 2
        or np.any(~np.isfinite(numeric))
        or np.any(np.diff(numeric) <= 0.0)
    ):
        return None
    if label_spec is None:
        return axis(_infer_code_label(specific_keys), values=numeric, order=0)
    return AxisSpec(
        code_label=label_spec.code_label,
        print_label=label_spec.print_label,
        html_label=label_spec.html_label,
        latex_label=label_spec.latex_label,
        values=numeric,
        order=0,
    )


def _build_index_axis(values: NDArray64) -> AxisSpec | None:
    if np.asarray(values, dtype=np.float64).size < 2:
        return None
    return axis("index", values=np.asarray(values, dtype=np.float64), order=0)


def _infer_code_label(specific_keys: Sequence[str]) -> str:
    """Infer one code label from the specific-key prefix."""
    for specific_key in specific_keys:
        if "=" in specific_key:
            candidate = specific_key.split("=", 1)[0].strip()
            if candidate != "":
                return candidate
    return "y"


def _load_traces_from_keys(
    *,
    filespec: FileSpec,
    keys: Keys,
    tracespec: TraceSpec,
) -> Traces:
    """Load traces for one measurement in the provided order."""
    if tracespec.amp_voltage <= 0.0 or not np.isfinite(tracespec.amp_voltage):
        raise ValueError("amp_voltage must be finite and > 0.")
    if tracespec.amp_current <= 0.0 or not np.isfinite(tracespec.amp_current):
        raise ValueError("amp_current must be finite and > 0.")
    if tracespec.r_ref_ohm <= 0.0 or not np.isfinite(tracespec.r_ref_ohm):
        raise ValueError("r_ref_ohm must be finite and > 0.")

    path, resolved_measurement = _require_measurement(
        h5path=filespec,
    )
    skip_pair = _normalize_skip(tracespec.skip)
    keys_list, yvalues_list = _normalize_keys_and_yvalues(
        keys=keys.specific_keys,
        yvalues=keys.yvalues,
    )

    h5py = _import_h5py()
    traces: list[Trace] = []
    with h5py.File(path, "r") as file:
        for index, (specific_key, value) in enumerate(
            zip(keys_list, yvalues_list),
        ):
            full_path = _to_measurement_path(
                measurement=resolved_measurement,
                specific_key=specific_key,
            )
            trace = _load_trace_from_file(
                file=file,
                full_path=full_path,
                amp_voltage=tracespec.amp_voltage,
                amp_current=tracespec.amp_current,
                r_ref_ohm=tracespec.r_ref_ohm,
                trigger_values=tracespec.trigger_values,
                skip=skip_pair,
                subtract_offset=tracespec.subtract_offset,
                time_relative=tracespec.time_relative,
            )
            traces.append(trace)

    return Traces.from_fields(
        traces=traces,
        specific_keys=keys.specific_keys,
        indices=np.asarray(keys.indices, dtype=np.int64).tolist(),
        yvalues=keys.yvalues,
        y_label=keys._spec.label,
    )


def get_traces(
    *,
    filespec: FileSpec,
    keys: Keys | None = None,
    keysspec: KeysSpec | None = None,
    tracespec: TraceSpec | None = None,
    specific_key: str | None = None,
    yvalue: float | None = None,
    index: int | None = None,
) -> Trace | Traces:
    """Load traces from file, key, and trace specifications."""
    resolved_tracespec = TraceSpec() if tracespec is None else tracespec
    has_selector = specific_key is not None or yvalue is not None or index is not None
    if keys is not None and keysspec is not None:
        raise ValueError("Provide either keys or keysspec, not both.")
    if keys is not None and has_selector:
        raise ValueError(
            "Provide either keys or trace selectors, not both.",
        )

    resolved_keys = get_keys(h5path=filespec, spec=keysspec) if keys is None else keys
    if has_selector:
        resolved_key, resolved_index, resolved_value = _resolve_trace_reference(
            resolved_keys,
            specific_key=specific_key,
            yvalue=yvalue,
            index=index,
        )
        resolved_keys = Keys.from_fields(
            specific_keys=[resolved_key],
            indices=np.asarray(
                [0 if resolved_index is None else resolved_index],
                dtype=np.int64,
            ),
            yvalues=[resolved_value],
            spec=resolved_keys._spec,
        )

    traces = _load_traces_from_keys(
        filespec=filespec,
        keys=resolved_keys,
        tracespec=resolved_tracespec,
    )
    if len(traces) == 1:
        return traces[0]
    return traces


__all__ = ["TraceSpec", "Trace", "Traces", "get_traces"]
