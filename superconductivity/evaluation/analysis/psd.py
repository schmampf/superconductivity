"""Helpers for PSD analysis on IV traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence, overload

import numpy as np

from ...utilities.meta import AxisSpec, Dataset, DataSpec, ParamSpec, axis, data, param
from ...utilities.meta.label import LabelSpec
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

    def keys(self) -> tuple[str, ...]:
        """Return public mapping-style keys."""
        return ("detrend",)


def _build_y_axis(
    values: NDArray64,
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
        return axis("y", values=numeric, order=0)
    return AxisSpec(
        code_label=label_spec.code_label,
        print_label=label_spec.print_label,
        html_label=label_spec.html_label,
        latex_label=label_spec.latex_label,
        values=numeric,
        order=0,
    )


def _build_index_axis(values: NDArray64) -> AxisSpec | None:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    if numeric.size < 2:
        return None
    return axis("index", values=numeric, order=0)


def _coerce_numeric_yvalues(
    indices: Sequence[int] | NDArray[np.int64],
    yvalues: Sequence[object],
) -> NDArray64:
    numeric: list[float] = []
    valid_numeric = True
    for value in yvalues:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool,
        ):
            cast = float(value)
            if not np.isfinite(cast):
                valid_numeric = False
                break
            numeric.append(cast)
        else:
            valid_numeric = False
            break
    if valid_numeric:
        return np.asarray(numeric, dtype=np.float64)
    return np.asarray(indices, dtype=np.float64).reshape(-1)


@dataclass(frozen=True, slots=True, init=False)
class PSDTrace(Dataset):
    """One PSD result stored as a dataset."""

    def __init__(
        self,
        *,
        f_Hz: NDArray64,
        I_psd_nA2_per_Hz: NDArray64,
        V_psd_mV2_per_Hz: NDArray64,
        nu_Hz: float,
        nyquist_Hz: float,
    ) -> None:
        psd_ds = Dataset(
            data=(
                data("I_psd_nA2_per_Hz", I_psd_nA2_per_Hz),
                data("V_psd_mV2_per_Hz", V_psd_mV2_per_Hz),
            ),
            axes=(axis("f_Hz", values=f_Hz, order=0),),
            params=(
                param("nu_Hz", nu_Hz),
                param("nyquist_Hz", nyquist_Hz),
            ),
        )
        object.__setattr__(self, "data", psd_ds.data)
        object.__setattr__(self, "axes", psd_ds.axes)
        object.__setattr__(self, "params", psd_ds.params)
        object.__setattr__(self, "_lookup", psd_ds._lookup)

    def __getitem__(self, key: str):
        if key in {"f_Hz", "I_psd_nA2_per_Hz", "V_psd_mV2_per_Hz"}:
            return np.asarray(Dataset.__getitem__(self, key).values, dtype=np.float64)
        if key in {"nu_Hz", "nyquist_Hz"}:
            return float(Dataset.__getitem__(self, key).values)
        return Dataset.__getitem__(self, key)

    @property
    def f_Hz(self) -> AxisSpec:
        """Return the PSD frequency axis."""
        return Dataset.__getitem__(self, "f_Hz")

    @property
    def I_psd_nA2_per_Hz(self) -> DataSpec:
        """Return the current PSD payload."""
        return Dataset.__getitem__(self, "I_psd_nA2_per_Hz")

    @property
    def V_psd_mV2_per_Hz(self) -> DataSpec:
        """Return the voltage PSD payload."""
        return Dataset.__getitem__(self, "V_psd_mV2_per_Hz")

    @property
    def nu_Hz(self) -> ParamSpec:
        """Return the sampling-rate parameter."""
        return Dataset.__getitem__(self, "nu_Hz")

    @property
    def nyquist_Hz(self) -> ParamSpec:
        """Return the Nyquist-frequency parameter."""
        return Dataset.__getitem__(self, "nyquist_Hz")


@dataclass(slots=True)
class PSDTraces:
    """Container for multiple PSD results."""

    traces: list[PSDTrace]
    y: AxisSpec | None = field(init=False, default=None)
    index: AxisSpec | None = field(init=False, default=None)
    skeys: tuple[str, ...] = field(init=False, default_factory=tuple)
    _indices: NDArray64 = field(init=False)

    def __post_init__(self) -> None:
        if len(self.traces) == 0:
            raise ValueError("traces must not be empty.")
        if not all(isinstance(trace, PSDTrace) for trace in self.traces):
            raise TypeError("traces must contain PSDTrace objects only.")
        object.__setattr__(self, "_indices", np.arange(len(self.traces), dtype=np.float64))
        object.__setattr__(self, "index", _build_index_axis(self._indices))
        object.__setattr__(self, "y", None)

    @classmethod
    def from_fields(
        cls,
        *,
        traces: list[PSDTrace],
        specific_keys: Sequence[str],
        indices: Sequence[int],
        yvalues: Sequence[object],
        y_label: LabelSpec | None,
    ) -> PSDTraces:
        """Build one PSD trace collection with collection metadata."""
        collection = cls(traces=traces)
        object.__setattr__(collection, "skeys", tuple(specific_keys))
        indices_array = np.asarray(indices, dtype=np.float64).reshape(-1)
        object.__setattr__(collection, "_indices", indices_array)
        object.__setattr__(collection, "index", _build_index_axis(indices_array))
        object.__setattr__(
            collection,
            "y",
            _build_y_axis(
                values=_coerce_numeric_yvalues(indices=indices, yvalues=yvalues),
                label_spec=y_label,
            ),
        )
        return collection

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

    def __getattr__(self, name: str):
        if self.y is not None and name == self.y.code_label:
            return self.y
        raise AttributeError(name)

    def keys(self) -> tuple[str, ...]:
        """Return public mapping-style keys."""
        keys = ["y", "i", "indices", "skeys", "specific_keys"]
        if self.y is not None and self.y.code_label != "y":
            keys.insert(1, self.y.code_label)
        keys.extend(
            [
                "f_Hz",
                "I_psd_nA2_per_Hz",
                "V_psd_mV2_per_Hz",
                "nu_Hz",
                "nyquist_Hz",
            ],
        )
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
    def i(self) -> AxisSpec | None:
        """Return the positional index axis."""
        return self.index

    @property
    def I_psd_nA2_per_Hz(self) -> list[NDArray64]:
        """Return per-trace current PSD arrays."""
        return [trace.I_psd_nA2_per_Hz for trace in self.traces]

    @property
    def V_psd_mV2_per_Hz(self) -> list[NDArray64]:
        """Return per-trace voltage PSD arrays."""
        return [trace.V_psd_mV2_per_Hz for trace in self.traces]

    @property
    def f_Hz(self) -> list[AxisSpec]:
        """Return per-trace frequency axes."""
        return [trace.f_Hz for trace in self.traces]

    @property
    def nu_Hz(self) -> list[ParamSpec]:
        """Return per-trace sampling frequencies."""
        return [trace.nu_Hz for trace in self.traces]

    @property
    def nyquist_Hz(self) -> list[ParamSpec]:
        """Return per-trace Nyquist frequencies."""
        return [trace.nyquist_Hz for trace in self.traces]

    def __getattr__(self, name: str):
        if self.y is not None and name == self.y.code_label:
            return self.y
        raise AttributeError(name)


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

    return PSDTrace(
        f_Hz=f_Hz,
        I_psd_nA2_per_Hz=I_psd,
        V_psd_mV2_per_Hz=V_psd,
        nu_Hz=nu_Hz,
        nyquist_Hz=nyquist_Hz,
    )

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
        return PSDTraces.from_fields(
            traces=[_build_single_psd_result(trace, spec=spec) for trace in traces],
            specific_keys=traces.specific_keys,
            indices=traces.indices,
            yvalues=traces.yaxis.values if traces.yaxis is not None else traces.indices,
            y_label=None if traces.y is None else traces.y,
        )
    return _build_single_psd_result(traces, spec=spec)


__all__ = ["PSDSpec", "PSDTrace", "PSDTraces", "psd_analysis"]
