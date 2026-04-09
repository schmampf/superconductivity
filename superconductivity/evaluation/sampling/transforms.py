"""Single-stage sampling and smoothing transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np

from ...utilities.constants import G_0_muS
from ...utilities.functions import bin_y_over_x, fill_nans
from ...utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ...utilities.types import NDArray64
from ..traces import Trace, Traces
from .containers import Sample, Samples
from .specs import SamplingSpec, _validate_downsample_rate_Hz

if TYPE_CHECKING:
    from ..analysis import OffsetTrace, OffsetTraces


def _import_tqdm():
    """Import tqdm lazily."""
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tqdm is required for progress display. Install it with "
            "'pip install tqdm'.",
        ) from exc
    return tqdm


def _import_scipy_ndimage():
    """Import SciPy ndimage filters lazily."""
    try:
        from scipy.ndimage import gaussian_filter1d, median_filter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SciPy is required for smoothing. Install it with 'pip install " "scipy'.",
        ) from exc
    return gaussian_filter1d, median_filter


def _downsample_trace_arrays(
    trace: Trace,
    *,
    nu_Hz: float,
) -> tuple[NDArray64, NDArray64, NDArray64]:
    """Downsample one raw IV trace onto a uniform time grid."""
    label = trace["meta"].specific_key
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

    nu_Hz = _validate_downsample_rate_Hz(nu_Hz)
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


def _copy_trace_with_arrays(
    trace: Trace,
    *,
    V_mV: NDArray64,
    I_nA: NDArray64,
    t_s: NDArray64 | None = None,
) -> Trace:
    """Copy one trace while replacing its numeric arrays."""
    t_values = np.asarray(trace["t_s"] if t_s is None else t_s, dtype=np.float64)
    return {
        "meta": trace["meta"],
        "I_nA": np.asarray(I_nA, dtype=np.float64),
        "V_mV": np.asarray(V_mV, dtype=np.float64),
        "t_s": t_values,
    }


def _copy_sample(
    sample: Sample,
    **updates: object,
) -> Sample:
    """Copy one sampled trace and normalize numeric arrays."""
    copied: Sample = {
        "meta": sample["meta"],
        "Vbins_mV": np.asarray(sample["Vbins_mV"], dtype=np.float64),
        "Ibins_nA": np.asarray(sample["Ibins_nA"], dtype=np.float64),
        "I_nA": np.asarray(sample["I_nA"], dtype=np.float64),
        "V_mV": np.asarray(sample["V_mV"], dtype=np.float64),
        "dG_G0": np.asarray(sample["dG_G0"], dtype=np.float64),
        "dR_R0": np.asarray(sample["dR_R0"], dtype=np.float64),
    }
    copied.update(updates)
    return copied


def downsample_trace(
    trace: Trace,
    *,
    nu_Hz: float,
) -> Trace:
    """Return one IV trace resampled to one target sampling rate."""
    t_s, V_mV, I_nA = _downsample_trace_arrays(trace=trace, nu_Hz=nu_Hz)
    return _copy_trace_with_arrays(trace, V_mV=V_mV, I_nA=I_nA, t_s=t_s)


def downsample_traces(
    traces: Traces,
    *,
    nu_Hz: float,
    show_progress: bool = True,
) -> Traces:
    """Return one IV-trace collection resampled to one target rate."""
    iterable: Iterator[Trace] | Traces = traces
    if show_progress:
        tqdm = _import_tqdm()
        iterable = tqdm(
            traces,
            total=len(traces),
            desc="downsample_traces",
            unit="trace",
        )
    return Traces(
        traces=[downsample_trace(trace, nu_Hz=nu_Hz) for trace in iterable],
    )


def _prepare_trace_for_sampling(trace: Trace) -> tuple[NDArray64, NDArray64]:
    """Return one IV trace as float64 arrays for binning."""
    return (
        np.asarray(trace["V_mV"], dtype=np.float64),
        np.asarray(trace["I_nA"], dtype=np.float64),
    )


def _sample_trace(
    trace: Trace,
    samplingspec: SamplingSpec,
) -> Sample:
    """Sample one prepared IV trace onto fixed V/I grids."""
    v_trace_mV, i_trace_nA = _prepare_trace_for_sampling(trace=trace)

    v_sampled_mV = bin_y_over_x(
        x=i_trace_nA,
        y=v_trace_mV,
        x_bins=samplingspec.Ibins_nA,
    )
    i_sampled_nA = bin_y_over_x(
        x=v_trace_mV,
        y=i_trace_nA,
        x_bins=samplingspec.Vbins_mV,
    )

    dG_G0 = np.gradient(i_sampled_nA, samplingspec.Vbins_mV) / G_0_muS
    dR_R0 = np.gradient(v_sampled_mV, samplingspec.Ibins_nA) * G_0_muS

    return {
        "meta": trace["meta"],
        "Vbins_mV": np.asarray(samplingspec.Vbins_mV, dtype=np.float64),
        "Ibins_nA": np.asarray(samplingspec.Ibins_nA, dtype=np.float64),
        "I_nA": np.asarray(i_sampled_nA, dtype=np.float64),
        "V_mV": np.asarray(v_sampled_mV, dtype=np.float64),
        "dG_G0": np.asarray(dG_G0, dtype=np.float64),
        "dR_R0": np.asarray(dR_R0, dtype=np.float64),
    }


def _sample_traces(
    traces: Traces,
    samplingspec: SamplingSpec,
    show_progress: bool = True,
) -> Samples:
    """Sample one collection of prepared IV traces."""
    iterable: Iterator[Trace] | Traces = traces
    if show_progress:
        tqdm = _import_tqdm()
        iterable = tqdm(
            iterable,
            total=len(traces),
            desc="sampling",
            unit="trace",
        )

    sampled = [
        _sample_trace(trace=trace, samplingspec=samplingspec) for trace in iterable
    ]
    return Samples(traces=sampled)


def _smooth_supported_segment(
    y: NDArray64,
    spec: SamplingSpec,
) -> NDArray64:
    """Smooth one 1D curve on its finite supported segment."""
    y_arr = np.asarray(y, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(y_arr))
    if finite_idx.size == 0:
        return np.asarray(y_arr, dtype=np.float64)

    lo = int(finite_idx[0])
    hi = int(finite_idx[-1]) + 1
    y_segment = np.asarray(y_arr[lo:hi], dtype=np.float64)
    y_segment = fill_nans(y_segment, method="linear")

    gaussian_filter1d, median_filter = _import_scipy_ndimage()
    if spec.median_bins > 1:
        y_segment = np.asarray(
            median_filter(y_segment, size=spec.median_bins, mode=spec.mode),
            dtype=np.float64,
        )
    if spec.sigma_bins > 0.0:
        y_segment = np.asarray(
            gaussian_filter1d(y_segment, sigma=spec.sigma_bins, mode=spec.mode),
            dtype=np.float64,
        )

    y_out = np.full_like(y_arr, np.nan, dtype=np.float64)
    y_out[lo:hi] = y_segment
    return y_out


def _smooth_sample(
    sample: Sample,
    spec: SamplingSpec,
) -> Sample:
    """Smooth one sampled IV trace and recompute derivatives."""
    if spec.median_bins <= 1 and spec.sigma_bins <= 0.0:
        return _copy_sample(sample)

    vbin_mV = np.asarray(sample["Vbins_mV"], dtype=np.float64)
    ibin_nA = np.asarray(sample["Ibins_nA"], dtype=np.float64)
    i_smooth_nA = _smooth_supported_segment(sample["I_nA"], spec=spec)
    v_smooth_mV = _smooth_supported_segment(sample["V_mV"], spec=spec)
    dG_G0 = np.gradient(i_smooth_nA, vbin_mV) / G_0_muS
    dR_R0 = np.gradient(v_smooth_mV, ibin_nA) * G_0_muS

    return _copy_sample(
        sample,
        Vbins_mV=np.asarray(vbin_mV, dtype=np.float64),
        Ibins_nA=np.asarray(ibin_nA, dtype=np.float64),
        I_nA=np.asarray(i_smooth_nA, dtype=np.float64),
        V_mV=np.asarray(v_smooth_mV, dtype=np.float64),
        dG_G0=np.asarray(dG_G0, dtype=np.float64),
        dR_R0=np.asarray(dR_R0, dtype=np.float64),
    )


def _smooth_samples(
    samples: Samples,
    spec: SamplingSpec,
    show_progress: bool = False,
) -> Samples:
    """Smooth one collection of sampled IV traces."""
    iterable: Iterator[Sample] | Samples = samples
    if show_progress:
        tqdm = _import_tqdm()
        iterable = tqdm(
            samples,
            total=len(samples),
            desc="smoothing",
            unit="trace",
        )

    traces = [_smooth_sample(sample=sample, spec=spec) for sample in iterable]
    return Samples(traces=traces)


def _offset_arrays_from_analysis(
    offsetanalysis: OffsetTrace | OffsetTraces,
    *,
    count: int,
) -> tuple[NDArray64, NDArray64]:
    """Return aligned per-trace offsets from one offset analysis result."""
    from ..analysis import OffsetTraces

    if isinstance(offsetanalysis, dict):
        if count != 1:
            raise ValueError("OffsetTrace can only be used with one IV trace.")
        return (
            np.asarray([float(offsetanalysis["Voff_mV"])], dtype=np.float64),
            np.asarray([float(offsetanalysis["Ioff_nA"])], dtype=np.float64),
        )

    if not isinstance(offsetanalysis, OffsetTraces):
        raise TypeError("offsetanalysis must be an OffsetTrace or OffsetTraces.")
    if len(offsetanalysis) != count:
        raise ValueError("traces and offsetanalysis must have the same length.")
    return (
        np.asarray(offsetanalysis.Voff_mV, dtype=np.float64),
        np.asarray(offsetanalysis.Ioff_nA, dtype=np.float64),
    )


def _offset_correct_trace(
    trace: Trace,
    *,
    Voff_mV: float,
    Ioff_nA: float,
) -> Trace:
    """Apply scalar voltage and current offsets to one IV trace."""
    return _copy_trace_with_arrays(
        trace,
        V_mV=np.asarray(trace["V_mV"], dtype=np.float64) - float(Voff_mV),
        I_nA=np.asarray(trace["I_nA"], dtype=np.float64) - float(Ioff_nA),
    )


def _upsample_trace(
    trace: Trace,
    *,
    factor: int,
) -> Trace:
    """Oversample one IV trace by interpolating each trace array."""
    if int(factor) == 1:
        return _copy_trace_with_arrays(
            trace,
            V_mV=np.asarray(trace["V_mV"], dtype=np.float64),
            I_nA=np.asarray(trace["I_nA"], dtype=np.float64),
        )

    factor = int(factor)
    t_raw = np.asarray(trace["t_s"], dtype=np.float64)
    v_raw_mV = np.asarray(trace["V_mV"], dtype=np.float64)
    i_raw_nA = np.asarray(trace["I_nA"], dtype=np.float64)
    sample_index = np.arange(t_raw.size, dtype=np.float64)
    sample_index_over = np.linspace(
        0.0,
        float(t_raw.size - 1),
        t_raw.size * factor,
        dtype=np.float64,
    )
    t_over_s = np.interp(sample_index_over, sample_index, t_raw)
    v_over_mV = np.interp(sample_index_over, sample_index, v_raw_mV)
    i_over_nA = np.interp(sample_index_over, sample_index, i_raw_nA)
    return _copy_trace_with_arrays(
        trace,
        V_mV=np.asarray(v_over_mV, dtype=np.float64),
        I_nA=np.asarray(i_over_nA, dtype=np.float64),
        t_s=t_over_s,
    )


def downsampling(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = False,
) -> Trace | Traces:
    """Downsample one trace or trace collection using ``samplingspec.nu_Hz``."""
    if isinstance(traces, Traces):
        return downsample_traces(
            traces,
            nu_Hz=samplingspec.nu_Hz,
            show_progress=show_progress,
        )
    return downsample_trace(traces, nu_Hz=samplingspec.nu_Hz)


def offset_correction(
    traces: Trace | Traces,
    *,
    offsetanalysis: OffsetTrace | OffsetTraces,
) -> Trace | Traces:
    """Apply one offset analysis result to one trace or trace collection."""
    if isinstance(traces, Traces):
        voff_mV, ioff_nA = _offset_arrays_from_analysis(
            offsetanalysis,
            count=len(traces),
        )
        return Traces(
            traces=[
                _offset_correct_trace(
                    trace,
                    Voff_mV=float(voff_mV[index]),
                    Ioff_nA=float(ioff_nA[index]),
                )
                for index, trace in enumerate(traces)
            ],
        )

    voff_mV, ioff_nA = _offset_arrays_from_analysis(offsetanalysis, count=1)
    return _offset_correct_trace(
        traces,
        Voff_mV=float(voff_mV[0]),
        Ioff_nA=float(ioff_nA[0]),
    )


def upsampling(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = False,
) -> Trace | Traces:
    """Upsample one trace or trace collection in the ``I-V`` plane."""
    if isinstance(traces, Traces):
        iterable: Iterator[Trace] | Traces = traces
        if show_progress:
            tqdm = _import_tqdm()
            iterable = tqdm(
                traces,
                total=len(traces),
                desc="upsampling",
                unit="trace",
            )
        return Traces(
            traces=[
                _upsample_trace(trace, factor=samplingspec.N_up)
                for trace in iterable
            ],
        )
    return _upsample_trace(traces, factor=samplingspec.N_up)


def binning(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = False,
) -> Sample | Samples:
    """Bin already corrected traces onto the sampling grids."""
    if isinstance(traces, Traces):
        return _sample_traces(
            traces,
            samplingspec=samplingspec,
            show_progress=show_progress,
        )
    return _sample_trace(traces, samplingspec=samplingspec)


def smooth(
    samples: Sample | Samples,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = False,
) -> Sample | Samples:
    """Smooth one sampled trace or one sampled-trace collection."""
    if isinstance(samples, Samples):
        return _smooth_samples(
            samples=samples,
            spec=samplingspec,
            show_progress=show_progress,
        )
    return _smooth_sample(samples, spec=samplingspec)


__all__ = [
    "downsample_trace",
    "downsample_traces",
    "downsampling",
    "upsampling",
    "offset_correction",
    "binning",
    "smooth",
]
