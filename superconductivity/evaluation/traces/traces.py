"""Helpers for extracting traces from HDF5 data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence, TypedDict

import numpy as np

from ...utilities.safety import require_all_finite
from ...utilities.types import NDArray64
from .file import FileSpec, _import_h5py, _require_measurement, _to_measurement_path
from .keys import (
    Keys,
    KeysSpec,
    _extract_value_from_specific_key,
    get_keys,
)


class Trace(TypedDict):
    """One trace with metadata and time axis."""

    specific_key: str
    index: int | None
    yvalue: float | None
    I_nA: NDArray64
    V_mV: NDArray64
    t_s: NDArray64


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


@dataclass(slots=True)
class Traces:
    """Container for multiple traces with lookup helpers."""

    traces: list[Trace]
    keys: list[str] = field(init=False)
    yvalues: NDArray64 = field(init=False)
    I_nA: list[NDArray64] = field(init=False)
    V_mV: list[NDArray64] = field(init=False)
    t_s: list[NDArray64] = field(init=False)
    _indices_by_key: dict[str, list[int]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Build list views and lookup tables from ``traces``."""
        self.keys = []
        self.I_nA = []
        self.V_mV = []
        self.t_s = []
        yvalues: list[float] = []
        indices_by_key: dict[str, list[int]] = {}

        for index, trace in enumerate(self.traces):
            specific_key = trace["specific_key"]
            self.keys.append(specific_key)
            self.I_nA.append(trace["I_nA"])
            self.V_mV.append(trace["V_mV"])
            self.t_s.append(trace["t_s"])
            yvalue = trace["yvalue"]
            yvalues.append(np.nan if yvalue is None else float(yvalue))
            indices_by_key.setdefault(specific_key, []).append(index)

        self.yvalues = np.asarray(yvalues, dtype=np.float64)
        self._indices_by_key = indices_by_key

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

    def all_by_key(
        self,
        specific_key: str,
    ) -> list[Trace]:
        """Return all traces with one exact specific key."""
        indices = self._indices_by_key.get(specific_key, [])
        return [self.traces[index] for index in indices]

    def by_key(
        self,
        specific_key: str,
    ) -> Trace:
        """Return one trace for one exact specific key."""
        return self._resolve_unique_match(
            matches=self.all_by_key(specific_key),
            selector_name="specific_key",
            selector_value=specific_key,
            plural_hint="all_by_key",
        )

    def all_by_value(
        self,
        yvalue: float,
    ) -> list[Trace]:
        """Return all traces with one y-value."""
        indices = self._find_indices_by_value(yvalue)
        return [self.traces[index] for index in indices]

    def by_value(
        self,
        yvalue: float,
    ) -> Trace:
        """Return one trace for one y-value."""
        return self._resolve_unique_match(
            matches=self.all_by_value(yvalue),
            selector_name="yvalue",
            selector_value=yvalue,
            plural_hint="all_by_value",
        )

    def _find_indices_by_value(
        self,
        yvalue: float,
    ) -> list[int]:
        """Return positional indices that match one y-value."""
        value = float(yvalue)
        if not np.isfinite(value):
            raise ValueError("yvalue must be finite.")

        atol = np.finfo(np.float64).eps * max(1.0, abs(value)) * 8.0
        matches = np.flatnonzero(
            np.isclose(self.yvalues, value, rtol=0.0, atol=atol),
        )
        return matches.tolist()

    @staticmethod
    def _resolve_unique_match(
        matches: list[Trace],
        selector_name: str,
        selector_value: str | float,
        plural_hint: str,
    ) -> Trace:
        """Return one match or raise a clear selector error."""
        if len(matches) == 0:
            raise KeyError(
                f"{selector_name} {selector_value!r} was not found.",
            )
        if len(matches) > 1:
            raise ValueError(
                f"{selector_name} {selector_value!r} matches multiple "
                f"traces. Use index or {plural_hint}(...).",
            )
        return matches[0]


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
    yvalues: Sequence[float] | np.ndarray,
) -> tuple[list[str], NDArray64]:
    """Normalize and validate collection metadata."""
    keys_list = list(keys)
    if len(keys_list) == 0:
        raise ValueError("keys must not be empty.")
    for key in keys_list:
        if not isinstance(key, str) or key == "":
            raise ValueError("keys must contain only non-empty strings.")

    yvalues_array = np.asarray(yvalues, dtype=np.float64).reshape(-1)
    if yvalues_array.size == 0:
        raise ValueError("yvalues must not be empty.")
    if yvalues_array.size != len(keys_list):
        raise ValueError("keys and yvalues must have the same length.")
    require_all_finite(yvalues_array, name="yvalues")
    return keys_list, np.asarray(yvalues_array, dtype=np.float64)


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
            f"yvalue {yvalue!r} matches multiple traces. "
            "Use index or Traces.all_by_value(...).",
        )
    return int(matches[0])


def _resolve_trace_reference(
    keys: Keys,
    specific_key: str | None = None,
    yvalue: float | None = None,
    index: int | None = None,
) -> tuple[str, int | None, float | None]:
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
            resolved_value = float(
                _extract_value_from_specific_key(
                    specific_key=specific_key,
                    strip0=keys._spec.strip0,
                    strip1=keys._spec.strip1,
                ),
            )
        except ValueError:
            resolved_value = None

    return resolved_key, resolved_index, resolved_value


def _load_trace_from_file(
    file: object,
    full_path: str,
    specific_key: str,
    index: int | None,
    yvalue: float | None,
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

    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "I_nA": np.asarray(I_nA, dtype=np.float64),
        "V_mV": np.asarray(V_mV, dtype=np.float64),
        "t_s": np.asarray(t_s, dtype=np.float64),
    }


def _load_traces_from_keys(
    *,
    filespec: FileSpec,
    keys: Sequence[str],
    yvalues: Sequence[float] | np.ndarray,
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
    keys_list, yvalues_array = _normalize_keys_and_yvalues(
        keys=keys,
        yvalues=yvalues,
    )

    h5py = _import_h5py()
    traces: list[Trace] = []
    with h5py.File(path, "r") as file:
        for index, (specific_key, value) in enumerate(
            zip(keys_list, yvalues_array),
        ):
            full_path = _to_measurement_path(
                measurement=resolved_measurement,
                specific_key=specific_key,
            )
            trace = _load_trace_from_file(
                file=file,
                full_path=full_path,
                specific_key=specific_key,
                index=index,
                yvalue=float(value),
                amp_voltage=tracespec.amp_voltage,
                amp_current=tracespec.amp_current,
                r_ref_ohm=tracespec.r_ref_ohm,
                trigger_values=tracespec.trigger_values,
                skip=skip_pair,
                subtract_offset=tracespec.subtract_offset,
                time_relative=tracespec.time_relative,
            )
            traces.append(trace)

    return Traces(traces=traces)


def get_traces(
    *,
    filespec: FileSpec,
    keys: Keys | None = None,
    keysspec: KeysSpec | None = None,
    tracespec: TraceSpec | None = None,
    specific_key: str | None = None,
    yvalue: float | None = None,
    index: int | None = None,
) -> Traces:
    """Load traces from file, key, and trace specifications."""
    resolved_tracespec = TraceSpec() if tracespec is None else tracespec
    has_selector = specific_key is not None or yvalue is not None or index is not None
    if keys is not None and keysspec is not None:
        raise ValueError("Provide either keys or keysspec, not both.")
    if keys is not None and has_selector:
        raise ValueError(
            "Provide either keys or trace selectors, not both.",
        )

    resolved_keys = (
        get_keys(filespec=filespec, keysspec=keysspec) if keys is None else keys
    )
    if has_selector:
        resolved_key, resolved_index, resolved_value = _resolve_trace_reference(
            resolved_keys,
            specific_key=specific_key,
            yvalue=yvalue,
            index=index,
        )
        resolved_keys = Keys(
            specific_keys=[resolved_key],
            indices=np.asarray(
                [0 if resolved_index is None else resolved_index],
                dtype=np.int64,
            ),
            yvalues=np.asarray(
                [np.nan if resolved_value is None else resolved_value],
                dtype=np.float64,
            ),
            _spec=resolved_keys._spec,
        )

    return _load_traces_from_keys(
        filespec=filespec,
        keys=resolved_keys.specific_keys,
        yvalues=resolved_keys.yvalues,
        tracespec=resolved_tracespec,
    )


__all__ = ["TraceSpec", "Trace", "Traces", "get_traces"]
