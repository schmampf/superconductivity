"""Helpers for extracting traces from HDF5 data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from ...utilities.meta import AxisSpec, Dataset, DataSpec, axis, data
from ...utilities.safety import require_all_finite
from ...utilities.types import NDArray64
from .file import FileSpec, _import_h5py, _require_measurement, _to_measurement_path
from .keys import Keys, KeysSpec, get_keys


def _import_scipy_signal():
    """Import SciPy signal filters lazily."""
    try:
        from scipy.signal import butter, sosfilt, sosfiltfilt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SciPy is required for low-pass filtering. Install it with "
            "'pip install scipy'.",
        ) from exc
    return butter, sosfilt, sosfiltfilt


@dataclass(slots=True)
class TraceSpec:
    """Configuration for loading traces from one measurement."""

    AmpV: float = 1.0
    AmpI: float = 1.0
    Rref_Ohm: float = 51.689e3
    trigger_values: int | Sequence[int] | None = 1
    skip_edges: int | tuple[int, int] = 0
    subtract_offset: bool = True
    time_relative: bool = True

    def keys(self) -> tuple[str, ...]:
        """Return public mapping-style keys."""
        return (
            "AmpV",
            "AmpI",
            "Rref_Ohm",
            "trigger_values",
            "skip_edges",
            "subtract_offset",
            "time_relative",
        )


@dataclass(frozen=True, slots=True, init=False)
class Trace(Dataset):
    """One trace stored as a dataset."""

    _dt_s: float | None = field(init=False, default=None, repr=False, compare=False)
    _psd_cache: tuple[NDArray64, NDArray64, NDArray64] | None = field(
        init=False, default=None, repr=False, compare=False
    )

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
        object.__setattr__(self, "_dt_s", None)
        object.__setattr__(self, "_psd_cache", None)

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

    @property
    def dt_s(self) -> float:
        """Return the median time spacing."""
        if self._dt_s is None:
            t_values = np.asarray(self.t_s.values, dtype=np.float64).reshape(-1)
            if t_values.size < 2:
                raise ValueError("t_s must contain at least two points.")
            diffs = np.diff(t_values)
            if np.any(diffs <= 0.0):
                raise ValueError("t_s must be strictly increasing.")
            object.__setattr__(self, "_dt_s", float(np.median(diffs)))
        return float(self._dt_s)

    @property
    def nu_Hz(self) -> float:
        """Return the sampling rate."""
        return 1.0 / self.dt_s

    def _compute_psd(self) -> tuple[NDArray64, NDArray64, NDArray64]:
        """Compute current and voltage PSD arrays."""
        t_arr = np.asarray(self.t_s.values, dtype=np.float64).reshape(-1)
        i_arr = np.asarray(self.I_nA.values, dtype=np.float64).reshape(-1)
        v_arr = np.asarray(self.V_mV.values, dtype=np.float64).reshape(-1)

        n = int(t_arr.size)
        if n < 2:
            raise ValueError("t_s must contain at least two points.")

        Iwork = i_arr - np.mean(i_arr)
        Vwork = v_arr - np.mean(v_arr)
        w = np.hanning(n).astype(np.float64)
        fs_Hz = self.nu_Hz
        w2_sum = float(np.sum(w * w))
        if not np.isfinite(w2_sum) or w2_sum <= 0.0:
            raise ValueError("Invalid window normalization.")

        Ifft_nA = np.fft.rfft(Iwork * w)
        Vfft_mV = np.fft.rfft(Vwork * w)

        Ipsd_nA2s = (np.abs(Ifft_nA) ** 2) / (fs_Hz * w2_sum)
        Vpsd_mV2s = (np.abs(Vfft_mV) ** 2) / (fs_Hz * w2_sum)
        if n % 2 == 0:
            if Ipsd_nA2s.size > 2:
                Ipsd_nA2s[1:-1] *= 2.0
                Vpsd_mV2s[1:-1] *= 2.0
        else:
            if Ipsd_nA2s.size > 1:
                Ipsd_nA2s[1:] *= 2.0
                Vpsd_mV2s[1:] *= 2.0
        f_Hz = np.fft.rfftfreq(n, d=self.dt_s)
        return (
            np.asarray(Ipsd_nA2s, dtype=np.float64),
            np.asarray(Vpsd_mV2s, dtype=np.float64),
            np.asarray(f_Hz, dtype=np.float64),
        )

    def _psd(self) -> tuple[NDArray64, NDArray64, NDArray64]:
        """Return cached PSD arrays."""
        if self._psd_cache is None:
            object.__setattr__(self, "_psd_cache", self._compute_psd())
        return self._psd_cache

    @property
    def f_Hz(self) -> NDArray64:
        """Return the PSD frequency axis."""
        return np.asarray(self._psd()[2], dtype=np.float64)

    @property
    def Ipsd_nA2s(self) -> NDArray64:
        """Return the current PSD."""
        return np.asarray(self._psd()[0], dtype=np.float64)

    @property
    def Vpsd_mV2s(self) -> NDArray64:
        """Return the voltage PSD."""
        return np.asarray(self._psd()[1], dtype=np.float64)

    def resample(self, *, nu_Hz: float) -> Trace:
        """Return one trace resampled onto one target rate."""
        nu_Hz = float(nu_Hz)
        if not np.isfinite(nu_Hz) or nu_Hz <= 0.0:
            raise ValueError("nu_Hz must be finite and > 0.")

        t_raw = np.asarray(self.t_s.values, dtype=np.float64)
        v_raw_mV = np.asarray(self.V_mV.values, dtype=np.float64)
        i_raw_nA = np.asarray(self.I_nA.values, dtype=np.float64)
        if t_raw.size < 2:
            raise ValueError("t_s must contain at least two points.")
        if not (
            t_raw.shape == v_raw_mV.shape == i_raw_nA.shape
        ):
            raise ValueError("t_s, V_mV, and I_nA must have the same shape.")
        if np.any(~np.isfinite(t_raw)) or np.any(~np.isfinite(v_raw_mV)) or np.any(
            ~np.isfinite(i_raw_nA)
        ):
            raise ValueError("Trace arrays must be finite.")
        if np.any(np.diff(t_raw) <= 0.0):
            raise ValueError("t_s must be strictly increasing.")

        t_min = float(np.min(t_raw))
        t_max = float(np.max(t_raw))
        dt_s = 1.0 / nu_Hz
        t_new = np.arange(t_min, t_max + 0.5 * dt_s, dt_s, dtype=np.float64)
        if t_new.size < 2:
            t_new = np.linspace(t_min, t_max, 2, dtype=np.float64)

        v_new = np.interp(t_new, t_raw, v_raw_mV)
        i_new = np.interp(t_new, t_raw, i_raw_nA)
        return Trace(
            I_nA=np.asarray(i_new, dtype=np.float64),
            V_mV=np.asarray(v_new, dtype=np.float64),
            t_s=np.asarray(t_new, dtype=np.float64),
        )

    def low_pass(self, *, cutoff_Hz: float, order: int = 4) -> Trace:
        """Return one trace low-pass filtered in the time domain."""
        cutoff_Hz = float(cutoff_Hz)
        if not np.isfinite(cutoff_Hz) or cutoff_Hz <= 0.0:
            raise ValueError("cutoff_Hz must be finite and > 0.")
        order = int(order)
        if order < 1:
            raise ValueError("order must be at least 1.")

        t_raw = np.asarray(self.t_s.values, dtype=np.float64)
        v_raw_mV = np.asarray(self.V_mV.values, dtype=np.float64)
        i_raw_nA = np.asarray(self.I_nA.values, dtype=np.float64)
        if t_raw.size < 2:
            raise ValueError("t_s must contain at least two points.")
        if not (
            t_raw.shape == v_raw_mV.shape == i_raw_nA.shape
        ):
            raise ValueError("t_s, V_mV, and I_nA must have the same shape.")
        if np.any(~np.isfinite(t_raw)) or np.any(~np.isfinite(v_raw_mV)) or np.any(
            ~np.isfinite(i_raw_nA)
        ):
            raise ValueError("Trace arrays must be finite.")
        if np.any(np.diff(t_raw) <= 0.0):
            raise ValueError("t_s must be strictly increasing.")

        nyquist_Hz = 0.5 * self.nu_Hz
        if cutoff_Hz >= nyquist_Hz:
            raise ValueError("cutoff_Hz must be below the Nyquist frequency.")

        butter, sosfilt, sosfiltfilt = _import_scipy_signal()
        sos = butter(
            order,
            cutoff_Hz / nyquist_Hz,
            btype="lowpass",
            output="sos",
        )
        if t_raw.size < max(3, 3 * order):
            v_filt_mV = sosfilt(sos, v_raw_mV)
            i_filt_nA = sosfilt(sos, i_raw_nA)
        else:
            v_filt_mV = sosfiltfilt(sos, v_raw_mV)
            i_filt_nA = sosfiltfilt(sos, i_raw_nA)
        return Trace(
            I_nA=np.asarray(i_filt_nA, dtype=np.float64),
            V_mV=np.asarray(v_filt_mV, dtype=np.float64),
            t_s=np.asarray(t_raw, dtype=np.float64),
        )

    def offset(self, *, Voff_mV: float = 0.0, Ioff_nA: float = 0.0) -> Trace:
        """Return one trace with constant voltage and current offsets removed."""
        Voff_mV = float(Voff_mV)
        Ioff_nA = float(Ioff_nA)
        return Trace(
            I_nA=np.asarray(self.I_nA.values, dtype=np.float64) - Ioff_nA,
            V_mV=np.asarray(self.V_mV.values, dtype=np.float64) - Voff_mV,
            t_s=np.asarray(self.t_s.values, dtype=np.float64),
        )


@dataclass(slots=True, kw_only=True)
class Traces(Keys):
    """Container for multiple traces."""

    yaxis: AxisSpec | None = None
    indices: NDArray64 | None = None
    skeys: tuple[str, ...] = field(default_factory=tuple)
    traces: list[Trace]

    def __post_init__(self) -> None:
        """Build list views from ``traces``."""
        for trace in self.traces:
            if not isinstance(trace, Trace):
                raise TypeError("Traces accepts Trace objects only.")
        if self.indices is None:
            object.__setattr__(
                self,
                "indices",
                np.arange(len(self.traces), dtype=np.float64),
            )
        else:
            object.__setattr__(
                self,
                "indices",
                np.asarray(self.indices, dtype=np.float64).reshape(-1),
            )
        object.__setattr__(self, "skeys", tuple(self.skeys))
        if len(self.skeys) not in {0, len(self.traces)}:
            raise ValueError("specific_keys must match the number of traces.")

    def __len__(self) -> int:
        """Return number of traces."""
        return len(self.traces)

    def __iter__(self) -> Iterator[Trace]:
        """Iterate over traces."""
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice | str,
    ) -> Trace | list[Trace]:
        """Return trace(s) by positional index."""
        if isinstance(index, (int, slice)):
            return self.traces[index]
        return Keys.__getitem__(self, index)

    def keys(self) -> tuple[str, ...]:
        """Return public mapping-style keys."""
        return Keys.keys(self) + ("t_s", "I_nA", "V_mV")

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

    def __getattr__(self, name: str):
        if self.yaxis is not None and name == self.yaxis.code_label:
            return self.yaxis
        raise AttributeError(name)

    @property
    def dt_s(self) -> list[float]:
        """Return the per-trace median time spacing."""
        return [trace.dt_s for trace in self.traces]

    @property
    def nu_Hz(self) -> list[float]:
        """Return the per-trace sampling rates."""
        return [trace.nu_Hz for trace in self.traces]

    @property
    def f_Hz(self) -> list[NDArray64]:
        """Return the per-trace PSD frequency axes."""
        return [trace.f_Hz for trace in self.traces]

    @property
    def Ipsd_nA2s(self) -> list[NDArray64]:
        """Return the per-trace current PSDs."""
        return [trace.Ipsd_nA2s for trace in self.traces]

    @property
    def Vpsd_mV2s(self) -> list[NDArray64]:
        """Return the per-trace voltage PSDs."""
        return [trace.Vpsd_mV2s for trace in self.traces]

    def resample(self, *, nu_Hz: float) -> Traces:
        """Return one resampled trace collection."""
        return Traces(
            traces=[trace.resample(nu_Hz=nu_Hz) for trace in self.traces],
            skeys=self.skeys,
            indices=self.indices,
            yaxis=self.yaxis,
        )

    def low_pass(self, *, cutoff_Hz: float, order: int = 4) -> Traces:
        """Return one low-pass filtered trace collection."""
        return Traces(
            traces=[trace.low_pass(cutoff_Hz=cutoff_Hz, order=order) for trace in self.traces],
            skeys=self.skeys,
            indices=self.indices,
            yaxis=self.yaxis,
        )

    def offset(
        self,
        *,
        Voff_mV: Sequence[float] | NDArray64 | float = 0.0,
        Ioff_nA: Sequence[float] | NDArray64 | float = 0.0,
    ) -> Traces:
        """Return one offset-corrected trace collection."""
        if np.isscalar(Voff_mV):
            voff = np.full(len(self.traces), float(Voff_mV), dtype=np.float64)
        else:
            voff = np.asarray(Voff_mV, dtype=np.float64).reshape(-1)
        if np.isscalar(Ioff_nA):
            ioff = np.full(len(self.traces), float(Ioff_nA), dtype=np.float64)
        else:
            ioff = np.asarray(Ioff_nA, dtype=np.float64).reshape(-1)
        if voff.size != len(self.traces):
            raise ValueError("Voff_mV must match the number of traces.")
        if ioff.size != len(self.traces):
            raise ValueError("Ioff_nA must match the number of traces.")
        return Traces(
            traces=[
                trace.offset(
                    Voff_mV=float(voff[index]),
                    Ioff_nA=float(ioff[index]),
                )
                for index, trace in enumerate(self.traces)
            ],
            skeys=self.skeys,
            indices=self.indices,
            yaxis=self.yaxis,
        )


def _normalize_skip_edges(
    skip_edges: int | tuple[int, int],
) -> tuple[int, int]:
    """Normalize symmetric or explicit edge trimming."""
    if isinstance(skip_edges, int):
        if skip_edges < 0:
            raise ValueError("skip_edges must be >= 0.")
        return skip_edges, skip_edges

    if (
        isinstance(skip_edges, tuple)
        and len(skip_edges) == 2
        and isinstance(skip_edges[0], int)
        and isinstance(skip_edges[1], int)
    ):
        skip_start, skip_end = int(skip_edges[0]), int(skip_edges[1])
        if skip_start < 0 or skip_end < 0:
            raise ValueError("skip_edges values must be >= 0.")
        return skip_start, skip_end

    raise ValueError("skip_edges must be int or tuple[int, int].")


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


def _load_trace_from_file(
    file: object,
    full_path: str,
    AmpV: float,
    AmpI: float,
    Rref_Ohm: float,
    trigger_values: int | Sequence[int] | None,
    skip_edges: tuple[int, int],
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

    skip_start, skip_end = skip_edges
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

    V_mV = v1_sweep_V / float(AmpV) * 1e3
    I_nA = v2_sweep_V / (float(AmpI) * float(Rref_Ohm)) * 1e9

    if subtract_offset:
        offset = file[f"{full_path}/offset/adwin"]
        v1_offset_V = np.asarray(offset["V1"], dtype=np.float64)
        v2_offset_V = np.asarray(offset["V2"], dtype=np.float64)
        v_offset_mV = np.nanmean(v1_offset_V / float(AmpV) * 1e3)
        i_offset_nA = np.nanmean(
            v2_offset_V / (float(AmpI) * float(Rref_Ohm)) * 1e9,
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


def _load_traces_from_keys(
    *,
    filespec: FileSpec,
    keys: Keys,
    tracespec: TraceSpec,
) -> list[Trace]:
    """Load traces for one measurement in the provided order."""
    if tracespec.AmpV <= 0.0 or not np.isfinite(tracespec.AmpV):
        raise ValueError("AmpV must be finite and > 0.")
    if tracespec.AmpI <= 0.0 or not np.isfinite(tracespec.AmpI):
        raise ValueError("AmpI must be finite and > 0.")
    if tracespec.Rref_Ohm <= 0.0 or not np.isfinite(tracespec.Rref_Ohm):
        raise ValueError("Rref_Ohm must be finite and > 0.")

    path, resolved_measurement = _require_measurement(
        h5path=filespec,
    )
    skip_pair = _normalize_skip_edges(tracespec.skip_edges)
    keys_list, yvalues_list = _normalize_keys_and_yvalues(
        keys=keys.specific_keys,
        yvalues=keys.yaxis.values,
    )

    h5py = _import_h5py()
    traces: list[Trace] = []
    with h5py.File(path, "r") as file:
        for specific_key, value in zip(keys_list, yvalues_list):
            full_path = _to_measurement_path(
                measurement=resolved_measurement,
                specific_key=specific_key,
            )
            trace = _load_trace_from_file(
                file=file,
                full_path=full_path,
                AmpV=tracespec.AmpV,
                AmpI=tracespec.AmpI,
                Rref_Ohm=tracespec.Rref_Ohm,
                trigger_values=tracespec.trigger_values,
                skip_edges=skip_pair,
                subtract_offset=tracespec.subtract_offset,
                time_relative=tracespec.time_relative,
            )
            traces.append(trace)

    return traces


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

    resolved_keys = (
        get_keys(filespec=filespec, keysspec=keysspec) if keys is None else keys
    )
    if has_selector:
        if specific_key is None and yvalue is None and index is None:
            raise ValueError(
                "Provide at least one of specific_key, yvalue, or index.",
            )

        keys_sorted = list(resolved_keys.specific_keys)
        yvalues_sorted = np.asarray(resolved_keys.yaxis.values, dtype=np.float64)
        resolved_key = specific_key
        resolved_index: int | None = None
        resolved_value: float | None = None

        if index is not None:
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
        if resolved_value is None:
            resolved_value = float(yvalues_sorted[keys_sorted.index(resolved_key)])

        code_label = "y"
        for specific_key in keys_sorted:
            if "=" in specific_key:
                candidate = specific_key.split("=", 1)[0].strip()
                if candidate != "":
                    code_label = candidate
                    break

        resolved_keys = Keys(
            skeys=(resolved_key,),
            indices=np.asarray(
                [0 if resolved_index is None else resolved_index],
                dtype=np.int64,
            ),
            yaxis=AxisSpec(
                code_label=code_label,
                print_label=resolved_keys.yaxis.print_label,
                html_label=resolved_keys.yaxis.html_label,
                latex_label=resolved_keys.yaxis.latex_label,
                values=np.asarray([resolved_value], dtype=np.float64),
                order=resolved_keys.yaxis.order,
            ),
        )

    trace_list = _load_traces_from_keys(
        filespec=filespec,
        keys=resolved_keys,
        tracespec=resolved_tracespec,
    )
    traces = Traces(
        traces=trace_list,
        skeys=resolved_keys.specific_keys,
        indices=np.asarray(resolved_keys.indices, dtype=np.float64),
        yaxis=resolved_keys.yaxis,
    )
    if len(traces) == 1:
        return traces[0]
    return traces


__all__ = ["TraceSpec", "Trace", "Traces", "get_traces"]
