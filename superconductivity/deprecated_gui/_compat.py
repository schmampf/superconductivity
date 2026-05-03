from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

NDArray64 = Any
G0_muS = 77.48


@dataclass(slots=True)
class DataSpec:
    code_label: str
    values: object

    @property
    def value(self):
        return self.values


@dataclass(slots=True)
class ParamSpec(DataSpec):
    pass


@dataclass(slots=True)
class LabelSpec:
    code_label: str
    print_label: str | None = None
    html_label: str | None = None
    latex_label: str | None = None


def data(name: str, values: object) -> DataSpec:
    return DataSpec(code_label=name, values=values)


def param(name: str, value: object) -> ParamSpec:
    return ParamSpec(code_label=name, values=value)

Trace = Any
Traces = Any


@dataclass(slots=True)
class TraceMeta:
    specific_key: str
    index: int
    yvalue: object = None


@dataclass(slots=True)
class FileSpec:
    h5path: str
    location: str
    measurement: str


@dataclass(slots=True)
class LabelSpec:
    code_label: str
    print_label: str | None = None
    html_label: str | None = None
    latex_label: str | None = None


@dataclass(slots=True)
class KeysSpec:
    strip0: str = "="
    strip1: str = ""
    remove_key: str = ""
    add_key: str = ""
    norm: str = ""
    label: LabelSpec | None = None
    limits: object = None


@dataclass(slots=True)
class Keys:
    specific_keys: Sequence[str]
    indices: Sequence[float]
    yvalues: Sequence[object]
    spec: KeysSpec | None = None

    @property
    def yaxis(self):
        return self.spec.label if self.spec is not None else None


@dataclass(slots=True)
class TraceSpec:
    amp_voltage: float = 1.0
    amp_current: float = 1.0
    r_ref_ohm: float = 51.689e3
    trigger_values: int | Sequence[int] | None = 1
    skip: int | tuple[int, int] = 0
    subtract_offset: bool = True
    time_relative: bool = True
OffsetDataset = Any
BCSModelConfig = Any
ParameterSpec = Any
SolutionDict = dict[str, object]
PSDTrace = dict[str, object]
PSDTraces = list[PSDTrace]
Sample = dict[str, object]
Samples = list[Sample]


def numeric_yvalue(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def get_keys(*args, **kwargs):
    from ..evaluation.traces.keys import get_keys as _get_keys

    return _get_keys(*args, **kwargs)


def get_traces(*args, **kwargs):
    from ..evaluation.traces.traces import get_traces as _get_traces

    return _get_traces(*args, **kwargs)


def _import_bcs():
    from ..optimizers import bcs as _bcs

    return _bcs


def get_model_config(*args, **kwargs):
    return _import_bcs().get_model_config(*args, **kwargs)


def get_model_key(*args, **kwargs):
    return _import_bcs().get_model_key(*args, **kwargs)


def get_model_spec(*args, **kwargs):
    return _import_bcs().get_model_spec(*args, **kwargs)


def make_bcs_parameters(*args, **kwargs):
    return _import_bcs().make_bcs_parameters(*args, **kwargs)


def make_noise_parameters(*args, **kwargs):
    return _import_bcs().make_noise_parameters(*args, **kwargs)


def make_pat_addon_parameters(*args, **kwargs):
    return _import_bcs().make_pat_addon_parameters(*args, **kwargs)


@dataclass(slots=True)
class PSDSpec:
    detrend: bool = True


@dataclass(slots=True)
class SamplingSpec:
    Vbins_mV: DataSpec | Sequence[float] | NDArray64
    Ibins_nA: DataSpec | Sequence[float] | NDArray64
    apply_offset_correction: bool = True
    apply_downsampling: bool = True
    apply_upsampling: bool = True
    apply_smoothing: bool = True
    nu_Hz: float = 43.7
    N_up: int = 1000
    median_bins: int = 3
    sigma_bins: float = 2.0
    mode: str = "nearest"
    Voff_mV: DataSpec | Sequence[float] | NDArray64 | float | None = None
    Ioff_nA: DataSpec | Sequence[float] | NDArray64 | float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.Vbins_mV, DataSpec):
            self.Vbins_mV = data("Vbins_mV", self.Vbins_mV)
        if not isinstance(self.Ibins_nA, DataSpec):
            self.Ibins_nA = data("Ibins_nA", self.Ibins_nA)
        if self.Voff_mV is not None and not isinstance(self.Voff_mV, DataSpec):
            self.Voff_mV = data("Voff_mV", self.Voff_mV)
        if self.Ioff_nA is not None and not isinstance(self.Ioff_nA, DataSpec):
            self.Ioff_nA = data("Ioff_nA", self.Ioff_nA)

    def as_evaluation_spec(self) -> object:
        from ..evaluation.sampling import SamplingSpec as _SamplingSpec

        return _SamplingSpec(
            Vbins_mV=self.Vbins_mV.values,
            Ibins_nA=self.Ibins_nA.values,
            Voff_mV=None if self.Voff_mV is None else self.Voff_mV.values,
            Ioff_nA=None if self.Ioff_nA is None else self.Ioff_nA.values,
            cutoff_Hz=self.nu_Hz,
            sampling_Hz=self.nu_Hz,
            median_bins=self.median_bins,
            sigma_bins=self.sigma_bins,
            mode=self.mode,
            apply_offset=self.apply_offset_correction,
            apply_lowpass=self.apply_downsampling,
            apply_resampling=self.apply_upsampling,
            apply_smoothing=self.apply_smoothing,
        )


@dataclass(slots=True)
class OffsetSpec:
    Vbins_mV: DataSpec | Sequence[float] | NDArray64
    Ibins_nA: DataSpec | Sequence[float] | NDArray64
    Voff_mV: DataSpec | Sequence[float] | NDArray64
    Ioff_nA: DataSpec | Sequence[float] | NDArray64
    nu_Hz: ParamSpec | float = 43.7
    N_up: ParamSpec | int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.Vbins_mV, DataSpec):
            self.Vbins_mV = data("Vbins_mV", self.Vbins_mV)
        if not isinstance(self.Ibins_nA, DataSpec):
            self.Ibins_nA = data("Ibins_nA", self.Ibins_nA)
        if not isinstance(self.Voff_mV, DataSpec):
            self.Voff_mV = data("Voff_mV", self.Voff_mV)
        if not isinstance(self.Ioff_nA, DataSpec):
            self.Ioff_nA = data("Ioff_nA", self.Ioff_nA)
        if not isinstance(self.nu_Hz, ParamSpec):
            self.nu_Hz = param("nu_Hz", self.nu_Hz)
        if not isinstance(self.N_up, ParamSpec):
            self.N_up = param("N_up", self.N_up)

    def as_evaluation_spec(self) -> object:
        from ..evaluation.offset import OffsetSpec as _OffsetSpec

        return _OffsetSpec(
            Vbins_mV=self.Vbins_mV.values,
            Ibins_nA=self.Ibins_nA.values,
            Voff_mV=self.Voff_mV.values,
            Ioff_nA=self.Ioff_nA.values,
            cutoff_Hz=float(self.nu_Hz),
            sampling_Hz=float(self.nu_Hz),
            Voffscan_mV=self.Voff_mV.values,
            Ioffscan_nA=self.Ioff_nA.values,
        )


def _to_psd_trace(trace: Trace, *, detrend: bool) -> PSDTrace:
    t_s = np.asarray(trace.t_s.values, dtype=np.float64).reshape(-1)
    i = np.asarray(trace.I_nA.values, dtype=np.float64).reshape(-1)
    v = np.asarray(trace.V_mV.values, dtype=np.float64).reshape(-1)
    if detrend:
        i = i - np.mean(i)
        v = v - np.mean(v)
    n = int(t_s.size)
    dt_s = trace.dt_s
    fs_Hz = trace.nu_Hz
    w = np.hanning(n).astype(np.float64)
    w2_sum = float(np.sum(w * w))
    i_fft = np.fft.rfft(i * w)
    v_fft = np.fft.rfft(v * w)
    i_psd = (np.abs(i_fft) ** 2) / (fs_Hz * w2_sum)
    v_psd = (np.abs(v_fft) ** 2) / (fs_Hz * w2_sum)
    if n % 2 == 0:
        if i_psd.size > 2:
            i_psd[1:-1] *= 2.0
            v_psd[1:-1] *= 2.0
    else:
        if i_psd.size > 1:
            i_psd[1:] *= 2.0
            v_psd[1:] *= 2.0
    return {
        "f_Hz": np.asarray(np.fft.rfftfreq(n, d=dt_s), dtype=np.float64),
        "I_psd_nA2_per_Hz": np.asarray(i_psd, dtype=np.float64),
        "V_psd_mV2_per_Hz": np.asarray(v_psd, dtype=np.float64),
        "nu_Hz": float(fs_Hz),
        "nyquist_Hz": float(fs_Hz / 2.0),
    }


def psd_analysis(
    traces: Trace | Traces,
    *,
    spec: PSDSpec | None = None,
) -> PSDTrace | PSDTraces:
    detrend = True if spec is None else bool(spec.detrend)
    if isinstance(traces, list):
        return [_to_psd_trace(trace, detrend=detrend) for trace in traces]
    return _to_psd_trace(traces, detrend=detrend)


def _wrap_sample(trace: Trace, pair: tuple[object, object]) -> Sample:
    exp_v, exp_i = pair
    return {
        "meta": trace["meta"],
        "_exp_v": exp_v,
        "_exp_i": exp_i,
        "Vbins_mV": np.asarray(exp_v["V_mV"].values, dtype=np.float64).copy(),
        "Ibins_nA": np.asarray(exp_i["I_nA"].values, dtype=np.float64).copy(),
        "I_nA": np.asarray(exp_v["I_nA"].values, dtype=np.float64).copy(),
        "V_mV": np.asarray(exp_i["V_mV"].values, dtype=np.float64).copy(),
        "dG_G0": np.asarray(exp_v.dG_G0.values, dtype=np.float64).copy(),
        "dR_R0": np.asarray(exp_i.dR_R0.values, dtype=np.float64).copy(),
    }


def sample(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = True,
    offsetanalysis: OffsetDataset | None = None,
) -> Sample | Samples:
    if offsetanalysis is not None:
        samplingspec = replace(
            samplingspec,
            Voff_mV=float(offsetanalysis["Voff_mV"]),
            Ioff_nA=float(offsetanalysis["Ioff_nA"]),
        )
    spec = samplingspec.as_evaluation_spec()
    from ..evaluation.sampling import sample as _sample

    if hasattr(traces, "traces"):
        return _sample(traces, samplingspec=spec, show_progress=show_progress)
    if isinstance(traces, list):
        return [
            _wrap_sample(trace, _sample(Traces(traces=[trace]), samplingspec=spec, show_progress=show_progress))
            for trace in traces
        ]
    pair = _sample(Traces(traces=[traces]), samplingspec=spec, show_progress=show_progress)
    return _wrap_sample(traces, pair)


def binning(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = True,
) -> Sample | Samples:
    return sample(traces, samplingspec=samplingspec, show_progress=show_progress)


def offset_analysis(
    traces: Trace | Traces,
    *,
    spec: OffsetSpec,
    show_progress: bool = True,
    workers: int = 8,
) -> OffsetDataset:
    from ..evaluation.offset import offset_analysis as _offset_analysis

    eval_spec = spec.as_evaluation_spec()
    if not hasattr(traces, "traces") and not isinstance(traces, list):
        traces = Traces(traces=[traces])
    return _offset_analysis(traces, spec=eval_spec, show_progress=show_progress, workers=workers)


def downsample_trace(trace: Trace, *, nu_Hz: float) -> Trace:
    return trace.resample(nu_Hz=nu_Hz)


def downsampling(
    trace: Trace,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = False,
) -> Trace:
    del show_progress
    out = trace
    if samplingspec.apply_downsampling and samplingspec.apply_lowpass:
        out = out.low_pass(cutoff_Hz=samplingspec.nu_Hz)
    if samplingspec.apply_downsampling or samplingspec.apply_upsampling:
        out = out.resample(nu_Hz=samplingspec.nu_Hz)
    return out


def offset_correction(
    trace: Trace,
    *,
    offsetanalysis: OffsetDataset | None = None,
) -> Trace:
    if offsetanalysis is None:
        return trace
    return trace.offset(
        Voff_mV=float(offsetanalysis["Voff_mV"]),
        Ioff_nA=float(offsetanalysis["Ioff_nA"]),
    )


def upsampling(
    trace: Trace,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = False,
) -> Trace:
    del show_progress
    if not samplingspec.apply_upsampling:
        return trace
    return trace.resample(nu_Hz=samplingspec.nu_Hz)


def smooth(
    sampling: Sample | Samples,
    *,
    samplingspec: SamplingSpec,
) -> Sample | Samples:
    if isinstance(sampling, list):
        return [smooth(item, samplingspec=samplingspec) for item in sampling]
    exp_v = sampling["_exp_v"].smooth(
        median_bins=samplingspec.median_bins,
        sigma_bins=samplingspec.sigma_bins,
        mode=samplingspec.mode,
    )
    exp_i = sampling["_exp_i"].smooth(
        median_bins=samplingspec.median_bins,
        sigma_bins=samplingspec.sigma_bins,
        mode=samplingspec.mode,
    )
    out = dict(sampling)
    out["_exp_v"] = exp_v
    out["_exp_i"] = exp_i
    out["I_nA"] = np.asarray(exp_v["I_nA"].values, dtype=np.float64).copy()
    out["V_mV"] = np.asarray(exp_i["V_mV"].values, dtype=np.float64).copy()
    out["dG_G0"] = np.asarray(exp_v.dG_G0.values, dtype=np.float64).copy()
    out["dR_R0"] = np.asarray(exp_i.dR_R0.values, dtype=np.float64).copy()
    return out
