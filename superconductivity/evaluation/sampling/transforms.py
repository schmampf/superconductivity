"""Single-stage sampling and smoothing transforms."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ...utilities.constants import G0_muS
from ...utilities.meta import Dataset, axis, data
from ...utilities.transport import TransportDatasetSpec
from ...utilities.functions.binning import bin
from ...utilities.functions.fill_nans import fill as fill_nans
from ...utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ...utilities.types import NDArray64
from ..traces import Trace, Traces
from .specs import SamplingSpec, _validate_downsample_rate_Hz


def _trace_axis_entry(sample: TransportDatasetSpec):
    """Return the collection axis entry for one sampled transport dataset."""
    for axis_entry in sample.axes:
        if axis_entry.code_label not in {"V_mV", "I_nA"}:
            return axis_entry
    raise AttributeError("sample is missing its trace axis.")

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
    label = "trace"
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

    v_down_mV = bin(z=v_raw_mV, x=t_raw, xbins=t_bins_s)
    i_down_nA = bin(z=i_raw_nA, x=t_raw, xbins=t_bins_s)
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
    t_values = np.asarray(trace.t_s.values if t_s is None else t_s, dtype=np.float64)
    return Trace(
        I_nA=np.asarray(I_nA, dtype=np.float64),
        V_mV=np.asarray(V_mV, dtype=np.float64),
        t_s=t_values,
    )


def _build_transport_sample_pair(
    *,
    Vbins_mV: NDArray64,
    Ibins_nA: NDArray64,
    I_nA: NDArray64,
    V_mV: NDArray64,
    yvalues: NDArray64 | None = None,
    y_label: str | None = None,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Build one or many transport datasets from sampled arrays."""
    Vbins_mV = np.asarray(Vbins_mV, dtype=np.float64)
    Ibins_nA = np.asarray(Ibins_nA, dtype=np.float64)
    I_nA = np.asarray(I_nA, dtype=np.float64)
    V_mV = np.asarray(V_mV, dtype=np.float64)

    if yvalues is None:
        exp_v = TransportDatasetSpec(
            data=(data("I_nA", I_nA),),
            axes=(axis("V_mV", values=Vbins_mV, order=0),),
        )
        exp_i = TransportDatasetSpec(
            data=(data("V_mV", V_mV),),
            axes=(axis("I_nA", values=Ibins_nA, order=0),),
        )
        return exp_v, exp_i

    yvalues = np.asarray(yvalues, dtype=np.float64).reshape(-1)
    trace_axis_label = "y" if y_label is None else str(y_label)
    exp_v = TransportDatasetSpec(
        data=(data("I_nA", I_nA),),
        axes=(
            axis(trace_axis_label, values=yvalues, order=0),
            axis("V_mV", values=Vbins_mV, order=1),
        ),
    )
    exp_i = TransportDatasetSpec(
        data=(data("V_mV", V_mV),),
        axes=(
            axis(trace_axis_label, values=yvalues, order=0),
            axis("I_nA", values=Ibins_nA, order=1),
        ),
    )
    return exp_v, exp_i


def _stack_transport_samples(
    exp_v_list: list[TransportDatasetSpec],
    exp_i_list: list[TransportDatasetSpec],
    *,
    yvalues: NDArray64,
    y_label: str | None,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Stack one list of sampled transport datasets along one trace axis."""
    if len(exp_v_list) == 0 or len(exp_i_list) == 0:
        raise ValueError("sample lists must not be empty.")
    v_bins = np.asarray(exp_v_list[0]["V_mV"].values, dtype=np.float64)
    i_bins = np.asarray(exp_i_list[0]["I_nA"].values, dtype=np.float64)
    return _build_transport_sample_pair(
        Vbins_mV=v_bins,
        Ibins_nA=i_bins,
        I_nA=np.vstack([np.asarray(item["I_nA"].values, dtype=np.float64) for item in exp_v_list]),
        V_mV=np.vstack([np.asarray(item["V_mV"].values, dtype=np.float64) for item in exp_i_list]),
        yvalues=yvalues,
        y_label=y_label,
    )


def downsample_trace(
    trace: Trace,
    *,
    nu_Hz: float,
) -> Trace:
    """Return one IV trace resampled to one target sampling rate."""
    return trace.resample(nu_Hz=nu_Hz)


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
        traces=[trace.resample(nu_Hz=nu_Hz) for trace in iterable],
        skeys=traces.specific_keys,
        indices=np.asarray(traces.indices, dtype=np.float64),
        yaxis=traces.yaxis,
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
    *,
    y_label: str | None = None,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Sample one prepared IV trace onto fixed V/I grids."""
    v_trace_mV, i_trace_nA = _prepare_trace_for_sampling(trace=trace)

    v_sampled_mV = bin(
        z=v_trace_mV,
        x=i_trace_nA,
        xbins=samplingspec.Ibins_nA,
    )
    i_sampled_nA = bin(
        z=i_trace_nA,
        x=v_trace_mV,
        xbins=samplingspec.Vbins_mV,
    )

    return _build_transport_sample_pair(
        Vbins_mV=np.asarray(samplingspec.Vbins_mV, dtype=np.float64),
        Ibins_nA=np.asarray(samplingspec.Ibins_nA, dtype=np.float64),
        I_nA=np.asarray(i_sampled_nA, dtype=np.float64),
        V_mV=np.asarray(v_sampled_mV, dtype=np.float64),
        y_label=y_label,
    )


def _sample_traces(
    traces: Traces,
    samplingspec: SamplingSpec,
    show_progress: bool = True,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
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
        _sample_trace(
            trace=trace,
            samplingspec=samplingspec,
            y_label=None if traces.y is None else traces.y.code_label,
        )
        for trace in iterable
    ]
    exp_v_list = [sample[0] for sample in sampled]
    exp_i_list = [sample[1] for sample in sampled]
    return _stack_transport_samples(
        exp_v_list,
        exp_i_list,
        yvalues=np.asarray(
            traces.yaxis.values if traces.yaxis is not None else traces.indices,
            dtype=np.float64,
        ),
        y_label=None if traces.y is None else traces.y.code_label,
    )


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
    y_segment = fill_nans(y_segment, method="interpolate")

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
    sample: TransportDatasetSpec,
    spec: SamplingSpec,
    *,
    Vbins_mV: NDArray64,
    Ibins_nA: NDArray64,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Smooth one sampled IV trace and recompute derivatives."""
    if spec.median_bins <= 1 and spec.sigma_bins <= 0.0:
        return _build_transport_sample_pair(
            Vbins_mV=Vbins_mV,
            Ibins_nA=Ibins_nA,
            I_nA=np.asarray(sample["I_nA"].values, dtype=np.float64),
            V_mV=np.asarray(sample["V_mV"].values, dtype=np.float64),
        )

    i_smooth_nA = _smooth_supported_segment(sample["I_nA"].values, spec=spec)
    v_smooth_mV = _smooth_supported_segment(sample["V_mV"].values, spec=spec)

    return _build_transport_sample_pair(
        Vbins_mV=Vbins_mV,
        Ibins_nA=Ibins_nA,
        I_nA=np.asarray(i_smooth_nA, dtype=np.float64),
        V_mV=np.asarray(v_smooth_mV, dtype=np.float64),
    )


def _smooth_samples(
    samples: tuple[TransportDatasetSpec, TransportDatasetSpec],
    spec: SamplingSpec,
    show_progress: bool = False,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Smooth one collection of sampled IV traces."""
    exp_v, exp_i = samples
    iterable = range(np.asarray(exp_v["I_nA"].values, dtype=np.float64).shape[0])
    if show_progress:
        tqdm = _import_tqdm()
        iterable = tqdm(
            iterable,
            total=np.asarray(exp_v["I_nA"].values, dtype=np.float64).shape[0],
            desc="smoothing",
            unit="trace",
        )

    traces = [
        _smooth_sample(
            sample=Dataset(
                data=(
                    data("I_nA", np.asarray(exp_v["I_nA"].values, dtype=np.float64)[index]),
                    data("V_mV", np.asarray(exp_i["V_mV"].values, dtype=np.float64)[index]),
                ),
            ),
            spec=spec,
            Vbins_mV=np.asarray(exp_v["V_mV"].values, dtype=np.float64),
            Ibins_nA=np.asarray(exp_i["I_nA"].values, dtype=np.float64),
        )
        for index in iterable
    ]
    exp_v_list = [trace[0] for trace in traces]
    exp_i_list = [trace[1] for trace in traces]
    return _stack_transport_samples(
        exp_v_list,
        exp_i_list,
        yvalues=np.asarray(_trace_axis_entry(exp_v).values, dtype=np.float64),
        y_label=_trace_axis_entry(exp_v).code_label,
    )


def _offset_arrays_from_spec(
    samplingspec: SamplingSpec,
    *,
    count: int,
) -> tuple[NDArray64, NDArray64]:
    """Return aligned offsets from one sampling spec."""
    if samplingspec.Voff_mV is None:
        voff = np.zeros(count, dtype=np.float64)
    else:
        voff = np.asarray(samplingspec.Voff_mV.values, dtype=np.float64).reshape(-1)
        if voff.size != count:
            raise ValueError("Voff_mV must match the number of traces.")

    if samplingspec.Ioff_nA is None:
        ioff = np.zeros(count, dtype=np.float64)
    else:
        ioff = np.asarray(samplingspec.Ioff_nA.values, dtype=np.float64).reshape(-1)
        if ioff.size != count:
            raise ValueError("Ioff_nA must match the number of traces.")

    return voff, ioff


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
    samplingspec: SamplingSpec,
) -> Trace | Traces:
    """Apply offsets from one sampling spec to one trace or collection."""
    if isinstance(traces, Traces):
        voff_mV, ioff_nA = _offset_arrays_from_spec(
            samplingspec,
            count=len(traces),
        )
    else:
        voff_mV, ioff_nA = _offset_arrays_from_spec(
            samplingspec,
            count=1,
        )

    if isinstance(traces, Traces):
        return Traces(
            traces=[
                _offset_correct_trace(
                    trace,
                    Voff_mV=float(voff_mV[index]),
                    Ioff_nA=float(ioff_nA[index]),
                )
                for index, trace in enumerate(traces)
            ],
            skeys=traces.specific_keys,
            indices=np.asarray(traces.indices, dtype=np.float64),
            yaxis=traces.yaxis,
        )

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
                _upsample_trace(trace, factor=samplingspec.N_up) for trace in iterable
            ],
            skeys=traces.specific_keys,
            indices=np.asarray(traces.indices, dtype=np.float64),
            yaxis=traces.yaxis,
        )
    return _upsample_trace(traces, factor=samplingspec.N_up)


def binning(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = False,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Bin already corrected traces onto the sampling grids."""
    if isinstance(traces, Traces):
        return _sample_traces(
            traces,
            samplingspec=samplingspec,
            show_progress=show_progress,
        )
    return _sample_trace(traces, samplingspec=samplingspec)


def smooth(
    samples: tuple[TransportDatasetSpec, TransportDatasetSpec],
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = False,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Smooth one sampled trace or one sampled-trace collection."""
    exp_v, exp_i = samples
    if np.asarray(exp_v["I_nA"].values, dtype=np.float64).ndim == 1:
        return _smooth_sample(
            exp_v,
            spec=samplingspec,
            Vbins_mV=np.asarray(exp_v["V_mV"].values, dtype=np.float64),
            Ibins_nA=np.asarray(exp_i["I_nA"].values, dtype=np.float64),
        )
    return _smooth_samples(samples, spec=samplingspec, show_progress=show_progress)


__all__ = [
    "downsample_trace",
    "downsample_traces",
    "downsampling",
    "upsampling",
    "offset_correction",
    "binning",
    "smooth",
]
