"""Helpers for PSD analysis and PSD-driven IV downsampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, TypedDict

import numpy as np

from ..utilities.functions import bin_y_over_x
from ..utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ..utilities.types import NDArray64
from .ivdata import IVTrace, IVTraces


@dataclass(slots=True)
class PSDSpec:
    """Configuration for PSD analysis and downsampling.

    Parameters
    ----------
    nu_Hz : float
        Target sampling rate in Hz for the downsampled trace.
    detrend : bool, default=True
        If ``True``, subtract the mean before computing the PSD.
    """

    nu_Hz: float
    detrend: bool = True

    def __post_init__(self) -> None:
        """Validate scalar PSD settings."""
        nu_Hz = float(self.nu_Hz)
        if not np.isfinite(nu_Hz) or nu_Hz <= 0.0:
            raise ValueError("nu_Hz must be finite and > 0.")
        self.nu_Hz = nu_Hz
        self.detrend = bool(self.detrend)


class PSDTrace(TypedDict):
    """One PSD result with raw and downsampled analysis."""

    specific_key: str
    index: int | None
    yvalue: float | None
    raw_f_Hz: NDArray64
    raw_I_psd_nA2_per_Hz: NDArray64
    raw_V_psd_mV2_per_Hz: NDArray64
    raw_sigma_I_nA: float
    raw_sigma_V_mV: float
    raw_nu_Hz: float
    raw_nyquist_Hz: float
    raw_sigma_cutoff_Hz: float
    downsampled_f_Hz: NDArray64
    downsampled_I_psd_nA2_per_Hz: NDArray64
    downsampled_V_psd_mV2_per_Hz: NDArray64
    downsampled_sigma_I_nA: float
    downsampled_sigma_V_mV: float
    downsampled_nu_Hz: float
    downsampled_nyquist_Hz: float
    downsampled_sigma_cutoff_Hz: float


@dataclass(slots=True)
class PSDTraces:
    """Container for multiple PSD results with lookup helpers."""

    traces: list[PSDTrace]
    keys: list[str] = field(init=False)
    yvalues: NDArray64 = field(init=False)
    raw_I_psd_nA2_per_Hz: list[NDArray64] = field(init=False)
    raw_V_psd_mV2_per_Hz: list[NDArray64] = field(init=False)
    raw_f_Hz: list[NDArray64] = field(init=False)
    raw_sigma_I_nA: NDArray64 = field(init=False)
    raw_sigma_V_mV: NDArray64 = field(init=False)
    raw_nu_Hz: NDArray64 = field(init=False)
    raw_nyquist_Hz: NDArray64 = field(init=False)
    raw_sigma_cutoff_Hz: NDArray64 = field(init=False)
    downsampled_I_psd_nA2_per_Hz: list[NDArray64] = field(init=False)
    downsampled_V_psd_mV2_per_Hz: list[NDArray64] = field(init=False)
    downsampled_f_Hz: list[NDArray64] = field(init=False)
    downsampled_sigma_I_nA: NDArray64 = field(init=False)
    downsampled_sigma_V_mV: NDArray64 = field(init=False)
    downsampled_nu_Hz: NDArray64 = field(init=False)
    downsampled_nyquist_Hz: NDArray64 = field(init=False)
    downsampled_sigma_cutoff_Hz: NDArray64 = field(init=False)
    _indices_by_key: dict[str, list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build list views and lookup tables from ``traces``."""
        self.keys = []
        self.raw_I_psd_nA2_per_Hz = []
        self.raw_V_psd_mV2_per_Hz = []
        self.raw_f_Hz = []
        raw_sigma_I_nA: list[float] = []
        raw_sigma_V_mV: list[float] = []
        raw_nu_Hz: list[float] = []
        raw_nyquist_Hz: list[float] = []
        raw_sigma_cutoff_Hz: list[float] = []
        self.downsampled_I_psd_nA2_per_Hz = []
        self.downsampled_V_psd_mV2_per_Hz = []
        self.downsampled_f_Hz = []
        downsampled_sigma_I_nA: list[float] = []
        downsampled_sigma_V_mV: list[float] = []
        downsampled_nu_Hz: list[float] = []
        downsampled_nyquist_Hz: list[float] = []
        downsampled_sigma_cutoff_Hz: list[float] = []
        yvalues: list[float] = []
        indices_by_key: dict[str, list[int]] = {}

        for index, trace in enumerate(self.traces):
            specific_key = trace["specific_key"]
            self.keys.append(specific_key)
            self.raw_I_psd_nA2_per_Hz.append(trace["raw_I_psd_nA2_per_Hz"])
            self.raw_V_psd_mV2_per_Hz.append(trace["raw_V_psd_mV2_per_Hz"])
            self.raw_f_Hz.append(trace["raw_f_Hz"])
            raw_sigma_I_nA.append(float(trace["raw_sigma_I_nA"]))
            raw_sigma_V_mV.append(float(trace["raw_sigma_V_mV"]))
            raw_nu_Hz.append(float(trace["raw_nu_Hz"]))
            raw_nyquist_Hz.append(float(trace["raw_nyquist_Hz"]))
            raw_sigma_cutoff_Hz.append(float(trace["raw_sigma_cutoff_Hz"]))
            self.downsampled_I_psd_nA2_per_Hz.append(
                trace["downsampled_I_psd_nA2_per_Hz"]
            )
            self.downsampled_V_psd_mV2_per_Hz.append(
                trace["downsampled_V_psd_mV2_per_Hz"]
            )
            self.downsampled_f_Hz.append(trace["downsampled_f_Hz"])
            downsampled_sigma_I_nA.append(
                float(trace["downsampled_sigma_I_nA"])
            )
            downsampled_sigma_V_mV.append(
                float(trace["downsampled_sigma_V_mV"])
            )
            downsampled_nu_Hz.append(float(trace["downsampled_nu_Hz"]))
            downsampled_nyquist_Hz.append(
                float(trace["downsampled_nyquist_Hz"])
            )
            downsampled_sigma_cutoff_Hz.append(
                float(trace["downsampled_sigma_cutoff_Hz"])
            )
            yvalue = trace["yvalue"]
            yvalues.append(np.nan if yvalue is None else float(yvalue))
            indices_by_key.setdefault(specific_key, []).append(index)

        self.yvalues = np.asarray(yvalues, dtype=np.float64)
        self.raw_sigma_I_nA = np.asarray(raw_sigma_I_nA, dtype=np.float64)
        self.raw_sigma_V_mV = np.asarray(raw_sigma_V_mV, dtype=np.float64)
        self.raw_nu_Hz = np.asarray(raw_nu_Hz, dtype=np.float64)
        self.raw_nyquist_Hz = np.asarray(raw_nyquist_Hz, dtype=np.float64)
        self.raw_sigma_cutoff_Hz = np.asarray(
            raw_sigma_cutoff_Hz,
            dtype=np.float64,
        )
        self.downsampled_sigma_I_nA = np.asarray(
            downsampled_sigma_I_nA,
            dtype=np.float64,
        )
        self.downsampled_sigma_V_mV = np.asarray(
            downsampled_sigma_V_mV,
            dtype=np.float64,
        )
        self.downsampled_nu_Hz = np.asarray(
            downsampled_nu_Hz,
            dtype=np.float64,
        )
        self.downsampled_nyquist_Hz = np.asarray(
            downsampled_nyquist_Hz,
            dtype=np.float64,
        )
        self.downsampled_sigma_cutoff_Hz = np.asarray(
            downsampled_sigma_cutoff_Hz,
            dtype=np.float64,
        )
        self._indices_by_key = indices_by_key

    def __len__(self) -> int:
        """Return number of traces."""
        return len(self.traces)

    def __iter__(self) -> Iterator[PSDTrace]:
        """Iterate over traces."""
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice,
    ) -> PSDTrace | list[PSDTrace]:
        """Return trace(s) by positional index."""
        return self.traces[index]

    def all_by_key(
        self,
        specific_key: str,
    ) -> list[PSDTrace]:
        """Return all traces with one exact specific key."""
        indices = self._indices_by_key.get(specific_key, [])
        return [self.traces[index] for index in indices]

    def by_key(
        self,
        specific_key: str,
    ) -> PSDTrace:
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
    ) -> list[PSDTrace]:
        """Return all traces with one y-value."""
        indices = self._find_indices_by_value(yvalue)
        return [self.traces[index] for index in indices]

    def by_value(
        self,
        yvalue: float,
    ) -> PSDTrace:
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
        matches: list[PSDTrace],
        selector_name: str,
        selector_value: str | float,
        plural_hint: str,
    ) -> PSDTrace:
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


def _get_sample_rate_Hz(t_s: NDArray64, *, name: str) -> float:
    """Return the sample rate from one strictly increasing time axis."""
    t_arr = to_1d_float64(t_s, name)
    require_min_size(t_arr, 2, name=name)
    require_all_finite(t_arr, name=name)

    dt = np.diff(t_arr)
    if np.any(dt <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")

    dt_s = float(np.median(dt))
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"Invalid time spacing in {name}.")

    return 1.0 / dt_s


def _downsample_iv_trace(
    trace: IVTrace,
    nu_Hz: float,
) -> tuple[NDArray64, NDArray64, NDArray64]:
    """Downsample one raw IV trace onto a uniform time grid."""
    label = trace["specific_key"]
    t_raw = to_1d_float64(trace["t_s"], f"{label}.t_s")
    v_raw_mV = to_1d_float64(trace["V_mV"], f"{label}.V_mV")
    i_raw_nA = to_1d_float64(trace["I_nA"], f"{label}.I_nA")

    require_same_shape(
        t_raw,
        v_raw_mV,
        name_a=f"{label}.t_s",
        name_b=f"{label}.V_mV",
    )
    require_same_shape(
        t_raw,
        i_raw_nA,
        name_a=f"{label}.t_s",
        name_b=f"{label}.I_nA",
    )
    require_min_size(t_raw, 2, name=f"{label}.t_s")
    require_all_finite(t_raw, name=f"{label}.t_s")
    require_all_finite(v_raw_mV, name=f"{label}.V_mV")
    require_all_finite(i_raw_nA, name=f"{label}.I_nA")

    nu_Hz = float(nu_Hz)
    if not np.isfinite(nu_Hz) or nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be finite and > 0.")

    t_min = float(np.min(t_raw))
    t_max = float(np.max(t_raw))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError(f"{label}.t_s must span at least two time points.")

    dt_s = 1.0 / nu_Hz
    t_bins_s = np.arange(
        t_min,
        t_max + 0.5 * dt_s,
        dt_s,
        dtype=np.float64,
    )
    if t_bins_s.size < 2:
        t_bins_s = np.linspace(t_min, t_max, 2, dtype=np.float64)

    v_down_mV = bin_y_over_x(x=t_raw, y=v_raw_mV, x_bins=t_bins_s)
    i_down_nA = bin_y_over_x(x=t_raw, y=i_raw_nA, x_bins=t_bins_s)
    finite = np.isfinite(t_bins_s) & np.isfinite(v_down_mV) & np.isfinite(i_down_nA)
    t_down_s = t_bins_s[finite]
    v_down_mV = v_down_mV[finite]
    i_down_nA = i_down_nA[finite]
    require_min_size(v_down_mV, 2, name=f"{label} downsampled trace")

    return (
        np.asarray(t_down_s, dtype=np.float64),
        np.asarray(v_down_mV, dtype=np.float64),
        np.asarray(i_down_nA, dtype=np.float64),
    )


def _get_downsampled_ivtrace(
    trace: IVTrace,
    *,
    spec: PSDSpec,
) -> IVTrace:
    """Return one IV trace resampled according to ``spec``."""
    t_s, V_mV, I_nA = _downsample_iv_trace(trace=trace, nu_Hz=spec.nu_Hz)
    return {
        "specific_key": trace["specific_key"],
        "index": trace["index"],
        "yvalue": trace["yvalue"],
        "I_nA": I_nA,
        "V_mV": V_mV,
        "t_s": t_s,
    }


def _get_psd_arrays(
    I_nA: NDArray64,
    V_mV: NDArray64,
    t_s: NDArray64,
    detrend: bool = True,
    window: str = "hann",
    enforce_uniform: bool = True,
    uniform_rtol: float = 1e-2,
) -> tuple[NDArray64, NDArray64, NDArray64]:
    """Compute one-sided PSD arrays from one ``I(t)``, ``V(t)`` trace."""
    I_arr = to_1d_float64(I_nA, "I_nA")
    V_arr = to_1d_float64(V_mV, "V_mV")
    t_arr = to_1d_float64(t_s, "t_s")

    require_same_shape(I_arr, V_arr, name_a="I_nA", name_b="V_mV")
    require_same_shape(I_arr, t_arr, name_a="I_nA", name_b="t_s")
    require_min_size(I_arr, 2, name="I_nA")

    require_all_finite(I_arr, name="I_nA")
    require_all_finite(V_arr, name="V_mV")
    require_all_finite(t_arr, name="t_s")

    dt = np.diff(t_arr)
    if np.any(dt <= 0.0):
        raise ValueError("t_s must be strictly increasing.")

    dt_s = float(np.median(dt))
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("Invalid time spacing in t_s.")

    if enforce_uniform:
        if uniform_rtol < 0.0 or not np.isfinite(uniform_rtol):
            raise ValueError("uniform_rtol must be finite and >= 0.")
        if not np.allclose(dt, dt_s, rtol=uniform_rtol, atol=0.0):
            raise ValueError(
                "t_s is not uniformly sampled within tolerance. "
                "Set enforce_uniform=False or resample first.",
            )

    n = I_arr.size
    window_key = window.strip().lower()
    if window_key in {"hann", "hanning"}:
        w = np.hanning(n).astype(np.float64)
    elif window_key in {"none", "rect", "boxcar"}:
        w = np.ones(n, dtype=np.float64)
    else:
        raise ValueError("Unsupported window. Use 'hann' or 'none'.")

    I_work = I_arr - np.mean(I_arr) if detrend else I_arr
    V_work = V_arr - np.mean(V_arr) if detrend else V_arr

    fs_Hz = 1.0 / dt_s
    w2_sum = float(np.sum(w * w))
    if not np.isfinite(w2_sum) or w2_sum <= 0.0:
        raise ValueError("Invalid window normalization.")

    I_fft_nA = np.fft.rfft(I_work * w)
    V_fft_mV = np.fft.rfft(V_work * w)

    I_psd_nA2_per_Hz = (np.abs(I_fft_nA) ** 2) / (fs_Hz * w2_sum)
    V_psd_mV2_per_Hz = (np.abs(V_fft_mV) ** 2) / (fs_Hz * w2_sum)

    if n % 2 == 0:
        if I_psd_nA2_per_Hz.size > 2:
            I_psd_nA2_per_Hz[1:-1] *= 2.0
            V_psd_mV2_per_Hz[1:-1] *= 2.0
    else:
        if I_psd_nA2_per_Hz.size > 1:
            I_psd_nA2_per_Hz[1:] *= 2.0
            V_psd_mV2_per_Hz[1:] *= 2.0

    f_Hz = np.fft.rfftfreq(n, d=dt_s)

    return (
        np.asarray(I_psd_nA2_per_Hz, dtype=np.float64),
        np.asarray(V_psd_mV2_per_Hz, dtype=np.float64),
        np.asarray(f_Hz, dtype=np.float64),
    )


def _get_sigma_from_psd(
    psd_per_Hz: NDArray64,
    f_Hz: NDArray64,
    *,
    psd_name: str,
    cutoff_Hz: float,
) -> float:
    """Return RMS noise from a one-sided PSD up to ``cutoff_Hz``."""
    psd_arr = to_1d_float64(psd_per_Hz, psd_name)
    f_arr = to_1d_float64(f_Hz, "f_Hz")

    require_same_shape(psd_arr, f_arr, name_a=psd_name, name_b="f_Hz")
    require_min_size(psd_arr, 1, name=psd_name)
    require_all_finite(psd_arr, name=psd_name)
    require_all_finite(f_arr, name="f_Hz")

    if np.any(np.diff(f_arr) < 0.0):
        raise ValueError("f_Hz must be sorted in ascending order.")

    cutoff_Hz = float(cutoff_Hz)
    if not np.isfinite(cutoff_Hz) or cutoff_Hz < 0.0:
        raise ValueError("cutoff_Hz must be finite and >= 0.")
    if cutoff_Hz <= f_arr[0]:
        return 0.0
    if cutoff_Hz < f_arr[-1]:
        upper = int(np.searchsorted(f_arr, cutoff_Hz, side="right"))
        f_fit = f_arr[:upper]
        psd_fit = psd_arr[:upper]
        if f_fit[-1] < cutoff_Hz:
            psd_at_cutoff = float(np.interp(cutoff_Hz, f_arr, psd_arr))
            f_fit = np.concatenate((f_fit, np.asarray([cutoff_Hz])))
            psd_fit = np.concatenate(
                (psd_fit, np.asarray([psd_at_cutoff])),
            )
        variance = float(np.trapezoid(psd_fit, f_fit))
        return float(np.sqrt(max(variance, 0.0)))

    variance = float(np.trapezoid(psd_arr, f_arr))
    return float(np.sqrt(max(variance, 0.0)))


def _get_sigma_cutoff_Hz(*, spec: PSDSpec, nyquist_Hz: float) -> float:
    """Return the RMS integration cutoff for one PSD trace."""
    return float(min(float(spec.nu_Hz) / 2.0, float(nyquist_Hz)))


def _build_single_psd_result(
    trace: IVTrace,
    *,
    spec: PSDSpec,
) -> tuple[IVTrace, PSDTrace]:
    """Return one downsampled trace and one combined PSD result."""
    downsampled_trace = _get_downsampled_ivtrace(trace, spec=spec)

    raw_I_psd, raw_V_psd, raw_f_Hz = _get_psd_arrays(
        I_nA=trace["I_nA"],
        V_mV=trace["V_mV"],
        t_s=trace["t_s"],
        detrend=spec.detrend,
    )
    raw_nu_Hz = _get_sample_rate_Hz(trace["t_s"], name="raw t_s")
    raw_nyquist_Hz = raw_nu_Hz / 2.0
    raw_sigma_cutoff_Hz = _get_sigma_cutoff_Hz(
        spec=spec,
        nyquist_Hz=raw_nyquist_Hz,
    )

    down_I_psd, down_V_psd, down_f_Hz = _get_psd_arrays(
        I_nA=downsampled_trace["I_nA"],
        V_mV=downsampled_trace["V_mV"],
        t_s=downsampled_trace["t_s"],
        detrend=spec.detrend,
    )
    downsampled_nu_Hz = _get_sample_rate_Hz(
        downsampled_trace["t_s"],
        name="downsampled t_s",
    )
    downsampled_nyquist_Hz = downsampled_nu_Hz / 2.0
    downsampled_sigma_cutoff_Hz = _get_sigma_cutoff_Hz(
        spec=spec,
        nyquist_Hz=downsampled_nyquist_Hz,
    )

    psd: PSDTrace = {
        "specific_key": trace["specific_key"],
        "index": trace["index"],
        "yvalue": trace["yvalue"],
        "raw_f_Hz": raw_f_Hz,
        "raw_I_psd_nA2_per_Hz": raw_I_psd,
        "raw_V_psd_mV2_per_Hz": raw_V_psd,
        "raw_sigma_I_nA": _get_sigma_from_psd(
            raw_I_psd,
            raw_f_Hz,
            psd_name="raw_I_psd_nA2_per_Hz",
            cutoff_Hz=raw_sigma_cutoff_Hz,
        ),
        "raw_sigma_V_mV": _get_sigma_from_psd(
            raw_V_psd,
            raw_f_Hz,
            psd_name="raw_V_psd_mV2_per_Hz",
            cutoff_Hz=raw_sigma_cutoff_Hz,
        ),
        "raw_nu_Hz": raw_nu_Hz,
        "raw_nyquist_Hz": raw_nyquist_Hz,
        "raw_sigma_cutoff_Hz": raw_sigma_cutoff_Hz,
        "downsampled_f_Hz": down_f_Hz,
        "downsampled_I_psd_nA2_per_Hz": down_I_psd,
        "downsampled_V_psd_mV2_per_Hz": down_V_psd,
        "downsampled_sigma_I_nA": _get_sigma_from_psd(
            down_I_psd,
            down_f_Hz,
            psd_name="downsampled_I_psd_nA2_per_Hz",
            cutoff_Hz=downsampled_sigma_cutoff_Hz,
        ),
        "downsampled_sigma_V_mV": _get_sigma_from_psd(
            down_V_psd,
            down_f_Hz,
            psd_name="downsampled_V_psd_mV2_per_Hz",
            cutoff_Hz=downsampled_sigma_cutoff_Hz,
        ),
        "downsampled_nu_Hz": downsampled_nu_Hz,
        "downsampled_nyquist_Hz": downsampled_nyquist_Hz,
        "downsampled_sigma_cutoff_Hz": downsampled_sigma_cutoff_Hz,
    }
    return downsampled_trace, psd


def get_psd(
    trace: IVTrace,
    *,
    spec: PSDSpec,
) -> tuple[IVTrace, PSDTrace]:
    """Return one downsampled trace and one PSD result.

    Parameters
    ----------
    trace : IVTrace
        Raw IV trace with time axis.
    spec : PSDSpec
        PSD/downsampling settings.

    Returns
    -------
    tuple[IVTrace, PSDTrace]
        The downsampled trace and the combined PSD result containing both
        raw and downsampled spectra.
    """
    return _build_single_psd_result(trace, spec=spec)


def get_psds(
    traces: IVTraces,
    *,
    spec: PSDSpec,
) -> tuple[IVTraces, PSDTraces]:
    """Return downsampled traces and PSD results for one collection.

    Parameters
    ----------
    traces : IVTraces
        Raw IV traces with time axes.
    spec : PSDSpec
        PSD/downsampling settings.

    Returns
    -------
    tuple[IVTraces, PSDTraces]
        The downsampled traces and combined PSD results for each input trace.
    """
    downsampled_traces: list[IVTrace] = []
    psd_traces: list[PSDTrace] = []
    for trace in traces:
        downsampled_trace, psd_trace = _build_single_psd_result(trace, spec=spec)
        downsampled_traces.append(downsampled_trace)
        psd_traces.append(psd_trace)
    return IVTraces(traces=downsampled_traces), PSDTraces(traces=psd_traces)


__all__ = ["PSDSpec", "PSDTrace", "PSDTraces", "get_psd", "get_psds"]
