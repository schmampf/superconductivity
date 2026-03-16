"""Helpers for power spectral densities derived from IV traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, TypedDict

import numpy as np

from ..utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ..utilities.types import NDArray64
from .ivdata import IVTrace, IVTraces


class PSDTrace(TypedDict):
    """One PSD trace with metadata."""

    specific_key: str
    index: int | None
    yvalue: float | None
    I_psd_nA2_per_Hz: NDArray64
    V_psd_mV2_per_Hz: NDArray64
    f_Hz: NDArray64


@dataclass(slots=True)
class PSDTraces:
    """Container for multiple PSD traces with lookup helpers."""

    traces: list[PSDTrace]
    keys: list[str] = field(init=False)
    yvalues: NDArray64 = field(init=False)
    I_psd_nA2_per_Hz: list[NDArray64] = field(init=False)
    V_psd_mV2_per_Hz: list[NDArray64] = field(init=False)
    f_Hz: list[NDArray64] = field(init=False)
    _indices_by_key: dict[str, list[int]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Build list views and lookup tables from ``traces``."""
        self.keys = []
        self.I_psd_nA2_per_Hz = []
        self.V_psd_mV2_per_Hz = []
        self.f_Hz = []
        yvalues: list[float] = []
        indices_by_key: dict[str, list[int]] = {}

        for index, trace in enumerate(self.traces):
            specific_key = trace["specific_key"]
            self.keys.append(specific_key)
            self.I_psd_nA2_per_Hz.append(trace["I_psd_nA2_per_Hz"])
            self.V_psd_mV2_per_Hz.append(trace["V_psd_mV2_per_Hz"])
            self.f_Hz.append(trace["f_Hz"])
            yvalue = trace["yvalue"]
            yvalues.append(np.nan if yvalue is None else float(yvalue))
            indices_by_key.setdefault(specific_key, []).append(index)

        self.yvalues = np.asarray(yvalues, dtype=np.float64)
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


def get_psd(
    trace: IVTrace,
    detrend: bool = True,
    window: str = "hann",
    enforce_uniform: bool = True,
    uniform_rtol: float = 1e-2,
) -> PSDTrace:
    """Compute one PSD trace from one IV trace."""
    I_psd_nA2_per_Hz, V_psd_mV2_per_Hz, f_Hz = _get_psd_arrays(
        I_nA=trace["I_nA"],
        V_mV=trace["V_mV"],
        t_s=trace["t_s"],
        detrend=detrend,
        window=window,
        enforce_uniform=enforce_uniform,
        uniform_rtol=uniform_rtol,
    )

    return {
        "specific_key": trace["specific_key"],
        "index": trace["index"],
        "yvalue": trace["yvalue"],
        "I_psd_nA2_per_Hz": I_psd_nA2_per_Hz,
        "V_psd_mV2_per_Hz": V_psd_mV2_per_Hz,
        "f_Hz": f_Hz,
    }


def get_psds(
    traces: IVTraces,
    detrend: bool = True,
    window: str = "hann",
    enforce_uniform: bool = True,
    uniform_rtol: float = 1e-2,
) -> PSDTraces:
    """Compute PSD traces for one collection of IV traces."""
    return PSDTraces(
        traces=[
            get_psd(
                trace=trace,
                detrend=detrend,
                window=window,
                enforce_uniform=enforce_uniform,
                uniform_rtol=uniform_rtol,
            )
            for trace in traces
        ],
    )


__all__ = ["PSDTrace", "PSDTraces", "get_psd", "get_psds"]
