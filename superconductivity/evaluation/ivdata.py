"""Helpers for extracting IV traces from HDF5 data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence, TypedDict

import numpy as np

from ..utilities.safety import require_all_finite
from ..utilities.types import NDArray64
from .keys import (
    _extract_value_from_specific_key,
    _import_h5py,
    _to_measurement_path,
    list_specific_keys_and_values,
)


class IVTrace(TypedDict):
    """One IV trace with metadata and time axis."""

    specific_key: str
    index: int | None
    yvalue: float | None
    I_nA: NDArray64
    V_mV: NDArray64
    t_s: NDArray64


@dataclass(slots=True)
class IVTraces:
    """Container for multiple IV traces with lookup helpers."""

    traces: list[IVTrace]
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

    def __iter__(self) -> Iterator[IVTrace]:
        """Iterate over traces."""
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice,
    ) -> IVTrace | list[IVTrace]:
        """Return trace(s) by positional index."""
        return self.traces[index]

    def all_by_key(
        self,
        specific_key: str,
    ) -> list[IVTrace]:
        """Return all traces with one exact specific key."""
        indices = self._indices_by_key.get(specific_key, [])
        return [self.traces[index] for index in indices]

    def by_key(
        self,
        specific_key: str,
    ) -> IVTrace:
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
    ) -> list[IVTrace]:
        """Return all traces with one y-value."""
        indices = self._find_indices_by_value(yvalue)
        return [self.traces[index] for index in indices]

    def by_value(
        self,
        yvalue: float,
    ) -> IVTrace:
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
        matches: list[IVTrace],
        selector_name: str,
        selector_value: str | float,
        plural_hint: str,
    ) -> IVTrace:
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
            "Use index or get_ivs(...).all_by_value(...).",
        )
    return int(matches[0])


def _resolve_iv_reference(
    h5path: str | Path,
    measurement: str,
    specific_key: str | None = None,
    yvalue: float | None = None,
    index: int | None = None,
    strip0: str = "=",
    strip1: str | None = None,
    remove_key: str | Sequence[str] | None = None,
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None = None,
) -> tuple[str, int | None, float | None]:
    """Resolve selectors into ``(specific_key, index, yvalue)``."""
    if specific_key is None and yvalue is None and index is None:
        raise ValueError(
            "Provide at least one of specific_key, yvalue, or index.",
        )

    keys_sorted: list[str] | None = None
    yvalues_sorted: NDArray64 | None = None
    if index is not None or yvalue is not None:
        keys_raw, yvalues_raw = list_specific_keys_and_values(
            h5path=h5path,
            measurement=measurement,
            strip0=strip0,
            strip1=strip1,
            remove_key=remove_key,
            add_key=add_key,
        )
        keys_sorted = list(keys_raw)
        yvalues_sorted = np.asarray(yvalues_raw, dtype=np.float64)

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
        if (
            resolved_index is not None
            and resolved_index != index_from_value
        ):
            raise ValueError(
                "index and yvalue refer to different traces.",
            )

        resolved_key = key_from_value
        resolved_index = index_from_value
        resolved_value = float(yvalues_sorted[index_from_value])

    if resolved_key is None:
        raise ValueError(
            "Could not resolve a trace. Provide specific_key, yvalue, "
            "or index.",
        )

    if resolved_value is None and specific_key is not None:
        try:
            resolved_value = float(
                _extract_value_from_specific_key(
                    specific_key=specific_key,
                    strip0=strip0,
                    strip1=strip1,
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
) -> IVTrace:
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


def get_iv(
    h5path: str | Path,
    measurement: str,
    specific_key: str | None = None,
    yvalue: float | None = None,
    index: int | None = None,
    amp_voltage: float = 1.0,
    amp_current: float = 1.0,
    r_ref_ohm: float = 51.689e3,
    trigger_values: int | Sequence[int] | None = 1,
    skip: int | tuple[int, int] = 0,
    subtract_offset: bool = True,
    time_relative: bool = True,
    strip0: str = "=",
    strip1: str | None = None,
    remove_key: str | Sequence[str] | None = None,
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None = None,
) -> IVTrace:
    """Load one IV trace, optionally resolved by yvalue or index.

    Parameters
    ----------
    h5path : str | pathlib.Path
        HDF5 file path.
    measurement : str
        Measurement name, e.g. ``"frequency_at_15GHz"``.
    specific_key : str | None, default=None
        Exact dataset key below the measurement group.
    yvalue : float | None, default=None
        Numeric y-value used to resolve one trace.
    index : int | None, default=None
        Trace index in the y-sorted order returned by
        :func:`list_specific_keys_and_values`.
    amp_voltage : float, default=1.0
        Voltage-channel amplification factor.
    amp_current : float, default=1.0
        Current-channel amplification factor.
    r_ref_ohm : float, default=51.689e3
        Reference resistor in ohms used for current conversion.
    trigger_values : int | Sequence[int] | None, default=1
        Trigger value(s) to keep from ``sweep/adwin/trigger``.
    skip : int | tuple[int, int], default=0
        Number of points to trim from the filtered trace edges.
    subtract_offset : bool, default=True
        If ``True``, subtract mean offset from ``offset/adwin``.
    time_relative : bool, default=True
        If ``True``, shift the time axis so ``t_s[0] == 0``.
    strip0 : str, default="="
        Start delimiter used to parse y-values from specific keys.
    strip1 : str | None, default=None
        End delimiter used to parse y-values from specific keys.
    remove_key : str or sequence of str, optional
        Exact specific-key names to ignore during selector resolution.
    add_key : tuple or sequence of tuple, optional
        Exact specific-key additions as ``(key, value)`` pairs.

    Returns
    -------
    IVTrace
        One trace with metadata, current, voltage, and time arrays.
    """
    if amp_voltage <= 0.0 or not np.isfinite(amp_voltage):
        raise ValueError("amp_voltage must be finite and > 0.")
    if amp_current <= 0.0 or not np.isfinite(amp_current):
        raise ValueError("amp_current must be finite and > 0.")
    if r_ref_ohm <= 0.0 or not np.isfinite(r_ref_ohm):
        raise ValueError("r_ref_ohm must be finite and > 0.")

    skip_pair = _normalize_skip(skip)
    resolved_key, resolved_index, resolved_value = _resolve_iv_reference(
        h5path=h5path,
        measurement=measurement,
        specific_key=specific_key,
        yvalue=yvalue,
        index=index,
        strip0=strip0,
        strip1=strip1,
        remove_key=remove_key,
        add_key=add_key,
    )

    h5py = _import_h5py()
    path = Path(h5path).expanduser()
    full_path = _to_measurement_path(
        measurement=measurement,
        specific_key=resolved_key,
    )

    with h5py.File(path, "r") as file:
        return _load_trace_from_file(
            file=file,
            full_path=full_path,
            specific_key=resolved_key,
            index=resolved_index,
            yvalue=resolved_value,
            amp_voltage=amp_voltage,
            amp_current=amp_current,
            r_ref_ohm=r_ref_ohm,
            trigger_values=trigger_values,
            skip=skip_pair,
            subtract_offset=subtract_offset,
            time_relative=time_relative,
        )


def get_ivs(
    h5path: str | Path,
    measurement: str,
    keys: Sequence[str],
    yvalues: Sequence[float] | np.ndarray,
    amp_voltage: float = 1.0,
    amp_current: float = 1.0,
    r_ref_ohm: float = 51.689e3,
    trigger_values: int | Sequence[int] | None = 1,
    skip: int | tuple[int, int] = 0,
    subtract_offset: bool = True,
    time_relative: bool = True,
) -> IVTraces:
    """Load IV traces for one measurement in the provided order.

    Parameters
    ----------
    h5path : str | pathlib.Path
        HDF5 file path.
    measurement : str
        Measurement name, e.g. ``"frequency_at_15GHz"``.
    keys : Sequence[str]
        Exact dataset keys below the measurement group. Order is preserved.
    yvalues : sequence of float or np.ndarray
        Y-values paired with ``keys``. Must have the same length and contain
        only finite values.
    amp_voltage : float, default=1.0
        Voltage-channel amplification factor.
    amp_current : float, default=1.0
        Current-channel amplification factor.
    r_ref_ohm : float, default=51.689e3
        Reference resistor in ohms used for current conversion.
    trigger_values : int | Sequence[int] | None, default=1
        Trigger value(s) to keep from ``sweep/adwin/trigger``.
    skip : int | tuple[int, int], default=0
        Number of points to trim from the filtered trace edges.
    subtract_offset : bool, default=True
        If ``True``, subtract mean offset from ``offset/adwin``.
    time_relative : bool, default=True
        If ``True``, shift each time axis so ``t_s[0] == 0``.

    Returns
    -------
    IVTraces
        Collection with list views and lookup helpers.
    """
    if amp_voltage <= 0.0 or not np.isfinite(amp_voltage):
        raise ValueError("amp_voltage must be finite and > 0.")
    if amp_current <= 0.0 or not np.isfinite(amp_current):
        raise ValueError("amp_current must be finite and > 0.")
    if r_ref_ohm <= 0.0 or not np.isfinite(r_ref_ohm):
        raise ValueError("r_ref_ohm must be finite and > 0.")

    skip_pair = _normalize_skip(skip)
    keys_list, yvalues_array = _normalize_keys_and_yvalues(
        keys=keys,
        yvalues=yvalues,
    )

    h5py = _import_h5py()
    path = Path(h5path).expanduser()
    traces: list[IVTrace] = []
    with h5py.File(path, "r") as file:
        for index, (specific_key, value) in enumerate(
            zip(keys_list, yvalues_array),
        ):
            full_path = _to_measurement_path(
                measurement=measurement,
                specific_key=specific_key,
            )
            trace = _load_trace_from_file(
                file=file,
                full_path=full_path,
                specific_key=specific_key,
                index=index,
                yvalue=float(value),
                amp_voltage=amp_voltage,
                amp_current=amp_current,
                r_ref_ohm=r_ref_ohm,
                trigger_values=trigger_values,
                skip=skip_pair,
                subtract_offset=subtract_offset,
                time_relative=time_relative,
            )
            traces.append(trace)

    return IVTraces(traces=traces)


__all__ = ["IVTrace", "IVTraces", "get_iv", "get_ivs"]
