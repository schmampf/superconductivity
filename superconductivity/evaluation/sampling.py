"""Sampling configuration and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utilities.meta import AxisSpec, DataSpec, axis, data
from ..utilities.safety import (
    require_all_finite,
    require_min_size,
    require_positive_float,
    to_1d_float64,
)
from ..utilities.transport import TransportDatasetSpec
from ..utilities.types import NDArray64
from .traces import Trace, Traces
from ..utilities.functions.binning import bin


@dataclass(slots=True)
class SamplingSpec:
    """Configuration for the explicit sampling pipeline."""

    Vbins_mV: Sequence[float] | NDArray64
    Ibins_nA: Sequence[float] | NDArray64
    Voff_mV: DataSpec | Sequence[float] | NDArray64 | None = None
    Ioff_nA: DataSpec | Sequence[float] | NDArray64 | None = None
    cutoff_Hz: float = 137.0
    sampling_Hz: float = 137.0
    median_bins: int = 3
    sigma_bins: float = 2.0
    mode: str = "nearest"
    apply_offset: bool = True
    apply_lowpass: bool = True
    apply_resampling: bool = True
    apply_smoothing: bool = True

    def __post_init__(self) -> None:
        self.cutoff_Hz = require_positive_float(self.cutoff_Hz, name="cutoff_Hz")
        self.sampling_Hz = require_positive_float(
            self.sampling_Hz,
            name="sampling_Hz",
        )

        if self.Voff_mV is not None:
            values = np.asarray(
                (
                    self.Voff_mV.values
                    if isinstance(self.Voff_mV, DataSpec)
                    else self.Voff_mV
                ),
                dtype=np.float64,
            ).reshape(-1)
            if values.size == 0:
                raise ValueError("Voff_mV must not be empty.")
            require_all_finite(values, name="Voff_mV")
        if self.Ioff_nA is not None:
            values = np.asarray(
                (
                    self.Ioff_nA.values
                    if isinstance(self.Ioff_nA, DataSpec)
                    else self.Ioff_nA
                ),
                dtype=np.float64,
            ).reshape(-1)
            if values.size == 0:
                raise ValueError("Ioff_nA must not be empty.")
            require_all_finite(values, name="Ioff_nA")

        self.apply_offset = bool(self.apply_offset)
        self.apply_lowpass = bool(self.apply_lowpass)
        self.apply_resampling = bool(self.apply_resampling)
        self.apply_smoothing = bool(self.apply_smoothing)

        median_bins = int(self.median_bins)
        if median_bins < 0:
            raise ValueError("median_bins must be >= 0.")
        if median_bins > 1 and median_bins % 2 == 0:
            raise ValueError("median_bins must be odd when > 1.")
        self.median_bins = median_bins

        sigma_bins = float(self.sigma_bins)
        if not np.isfinite(sigma_bins) or sigma_bins < 0.0:
            raise ValueError("sigma_bins must be finite and >= 0.")
        self.sigma_bins = sigma_bins

        mode = str(self.mode).strip().lower()
        if mode == "":
            raise ValueError("mode must not be empty.")
        self.mode = mode

        self.Vbins_mV = to_1d_float64(self.Vbins_mV, "Vbins_mV")
        self.Ibins_nA = to_1d_float64(self.Ibins_nA, "Ibins_nA")

        require_min_size(self.Vbins_mV, 2, name="Vbins_mV")
        require_min_size(self.Ibins_nA, 2, name="Ibins_nA")
        require_all_finite(self.Vbins_mV, name="Vbins_mV")
        require_all_finite(self.Ibins_nA, name="Ibins_nA")

    def keys(self) -> tuple[str, ...]:
        return (
            "Vbins_mV",
            "Ibins_nA",
            "Voff_mV",
            "Ioff_nA",
            "apply_offset",
            "apply_lowpass",
            "apply_resampling",
            "apply_smoothing",
            "cutoff_Hz",
            "sampling_Hz",
            "median_bins",
            "sigma_bins",
            "mode",
        )


def _import_tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tqdm is required for progress display. Install it with "
            "'pip install tqdm'.",
        ) from exc
    return tqdm


def _build_transport_sample_pair(
    *,
    Vbins_mV: NDArray64,
    Ibins_nA: NDArray64,
    I_nA: NDArray64,
    V_mV: NDArray64,
    yaxis: AxisSpec | None = None,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    Vbins_mV = np.asarray(Vbins_mV, dtype=np.float64)
    Ibins_nA = np.asarray(Ibins_nA, dtype=np.float64)
    I_nA = np.asarray(I_nA, dtype=np.float64)
    V_mV = np.asarray(V_mV, dtype=np.float64)

    if yaxis is None:
        exp_v = TransportDatasetSpec(
            data=(data("I_nA", I_nA),),
            axes=(axis("V_mV", values=Vbins_mV, order=0),),
        )
        exp_i = TransportDatasetSpec(
            data=(data("V_mV", V_mV),),
            axes=(axis("I_nA", values=Ibins_nA, order=0),),
        )
        return exp_v, exp_i

    yvalues = np.asarray(yaxis.values, dtype=np.float64).reshape(-1)
    exp_v = TransportDatasetSpec(
        data=(data("I_nA", I_nA),),
        axes=(
            axis(yaxis.code_label, values=yvalues, order=0),
            axis("V_mV", values=Vbins_mV, order=1),
        ),
    )
    exp_i = TransportDatasetSpec(
        data=(data("V_mV", V_mV),),
        axes=(
            axis(yaxis.code_label, values=yvalues, order=0),
            axis("I_nA", values=Ibins_nA, order=1),
        ),
    )
    return exp_v, exp_i


def binning(
    traces: Traces,
    samplingspec: SamplingSpec,
    show_progress: bool = True,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    iterable: Traces | list[Trace] = traces
    if show_progress:
        tqdm = _import_tqdm()
        iterable = tqdm(
            iterable,
            total=len(traces),
            desc="sampling",
            unit="trace",
        )

    sampled_v = []
    sampled_i = []
    v_bins = np.asarray(samplingspec.Vbins_mV, dtype=np.float64)
    i_bins = np.asarray(samplingspec.Ibins_nA, dtype=np.float64)
    for trace in iterable:
        v_trace_mV = np.asarray(trace.V_mV.values, dtype=np.float64)
        i_trace_nA = np.asarray(trace.I_nA.values, dtype=np.float64)
        sampled_v.append(
            np.asarray(
                bin(
                    z=i_trace_nA,
                    x=v_trace_mV,
                    xbins=samplingspec.Vbins_mV,
                ),
                dtype=np.float64,
            )
        )
        sampled_i.append(
            np.asarray(
                bin(
                    z=v_trace_mV,
                    x=i_trace_nA,
                    xbins=samplingspec.Ibins_nA,
                ),
                dtype=np.float64,
            )
        )

    if len(sampled_v) == 0:
        raise ValueError("sample lists must not be empty.")
    yaxis = (
        traces.yaxis
        if traces.yaxis is not None
        else axis(
            "y",
            values=np.asarray(traces.indices, dtype=np.float64),
            order=0,
        )
    )
    return _build_transport_sample_pair(
        Vbins_mV=v_bins,
        Ibins_nA=i_bins,
        I_nA=np.vstack(sampled_v),
        V_mV=np.vstack(sampled_i),
        yaxis=yaxis,
    )


def sample(
    traces: Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = True,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Run the full sampling pipeline and return voltage- and current-bias datasets."""
    if samplingspec.apply_offset:
        voff = 0.0 if samplingspec.Voff_mV is None else samplingspec.Voff_mV
        ioff = 0.0 if samplingspec.Ioff_nA is None else samplingspec.Ioff_nA
        traces = traces.offset(Voff_mV=voff, Ioff_nA=ioff)
    if samplingspec.apply_lowpass:
        traces = traces.low_pass(cutoff_Hz=samplingspec.cutoff_Hz)
    if samplingspec.apply_resampling:
        traces = traces.resample(nu_Hz=samplingspec.sampling_Hz)
    samples = binning(
        traces,
        samplingspec=samplingspec,
        show_progress=show_progress,
    )
    if samplingspec.apply_smoothing:
        exp_v, exp_i = samples
        return (
            exp_v.smooth(
                median_bins=samplingspec.median_bins,
                sigma_bins=samplingspec.sigma_bins,
                mode=samplingspec.mode,
            ),
            exp_i.smooth(
                median_bins=samplingspec.median_bins,
                sigma_bins=samplingspec.sigma_bins,
                mode=samplingspec.mode,
            ),
        )
    return samples


__all__ = [
    "SamplingSpec",
    "binning",
    "sample",
]
