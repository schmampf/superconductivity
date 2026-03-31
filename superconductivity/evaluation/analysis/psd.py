"""Helpers for PSD analysis on IV traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, TypedDict, overload

import numpy as np

from ...utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ...utilities.types import NDArray64
from ..traces import Trace, Traces


@dataclass(slots=True)
class PSDSpec:
    """Configuration for PSD analysis.

    Parameters
    ----------
    detrend : bool, default=True
        If ``True``, subtract the mean before computing the PSD.
    """

    detrend: bool = True

    def __post_init__(self) -> None:
        """Validate scalar PSD settings."""
        self.detrend = bool(self.detrend)


class PSDTrace(TypedDict):
    """One PSD result."""

    f_Hz: NDArray64
    I_psd_nA2_per_Hz: NDArray64
    V_psd_mV2_per_Hz: NDArray64
    nu_Hz: float
    nyquist_Hz: float


@dataclass(slots=True)
class PSDTraces:
    """Container for multiple PSD results."""

    traces: list[PSDTrace]
    I_psd_nA2_per_Hz: list[NDArray64] = field(init=False)
    V_psd_mV2_per_Hz: list[NDArray64] = field(init=False)
    f_Hz: list[NDArray64] = field(init=False)
    nu_Hz: NDArray64 = field(init=False)
    nyquist_Hz: NDArray64 = field(init=False)

    def __post_init__(self) -> None:
        """Build list and stacked-array views from ``traces``."""
        if len(self.traces) == 0:
            raise ValueError("traces must not be empty.")

        self.I_psd_nA2_per_Hz = []
        self.V_psd_mV2_per_Hz = []
        self.f_Hz = []
        nu_Hz: list[float] = []
        nyquist_Hz: list[float] = []

        for trace in self.traces:
            self.I_psd_nA2_per_Hz.append(trace["I_psd_nA2_per_Hz"])
            self.V_psd_mV2_per_Hz.append(trace["V_psd_mV2_per_Hz"])
            self.f_Hz.append(trace["f_Hz"])
            nu_Hz.append(float(trace["nu_Hz"]))
            nyquist_Hz.append(float(trace["nyquist_Hz"]))
        self.nu_Hz = np.asarray(nu_Hz, dtype=np.float64)
        self.nyquist_Hz = np.asarray(nyquist_Hz, dtype=np.float64)

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


def _build_single_psd_result(
    trace: Trace,
    *,
    spec: PSDSpec,
) -> PSDTrace:
    """Return one PSD result."""
    I_psd, V_psd, f_Hz = _get_psd_arrays(
        I_nA=trace["I_nA"],
        V_mV=trace["V_mV"],
        t_s=trace["t_s"],
        detrend=spec.detrend,
    )
    nu_Hz = _get_sample_rate_Hz(trace["t_s"], name="t_s")
    nyquist_Hz = nu_Hz / 2.0

    psd: PSDTrace = {
        "f_Hz": f_Hz,
        "I_psd_nA2_per_Hz": I_psd,
        "V_psd_mV2_per_Hz": V_psd,
        "nu_Hz": nu_Hz,
        "nyquist_Hz": nyquist_Hz,
    }
    return psd


@overload
def psd_analysis(
    traces: Trace,
    *,
    spec: PSDSpec,
) -> PSDTrace: ...


@overload
def psd_analysis(
    traces: Traces,
    *,
    spec: PSDSpec,
) -> PSDTraces: ...


def psd_analysis(
    traces: Trace | Traces,
    *,
    spec: PSDSpec,
) -> PSDTrace | PSDTraces:
    """Return PSD analysis for one trace or one trace collection."""
    if isinstance(traces, Traces):
        return PSDTraces(
            traces=[_build_single_psd_result(trace, spec=spec) for trace in traces]
        )
    return _build_single_psd_result(traces, spec=spec)


__all__ = ["PSDSpec", "PSDTrace", "PSDTraces", "psd_analysis"]
