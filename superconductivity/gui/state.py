from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

from ..evaluation.traces import Trace, numeric_yvalue
from ..evaluation.analysis import OffsetSpec, OffsetTrace
from ..evaluation.analysis import PSDTrace
from ..evaluation.sampling import Sample, SamplingSpec
from ..optimizers.bcs import (
    BCSModelConfig,
    ParameterSpec,
    SolutionDict,
    fit_model,
    get_model_spec,
)
from ..utilities.types import NDArray64

_DEFAULT_MODEL = "bcs_conv_jax"
_DEFAULT_SHARED_NU_HZ = 13.7
_EXPERIMENTAL_TITLES = {
    "nu_Hz": "<i>&nu;</i> (Hz)",
    "detrend": "Detrend",
}


class GUIStateDict(TypedDict):
    active_index: int
    trace: Trace
    psd: PSDTrace
    offset: OffsetTrace
    sampling: Sample
    fit: Optional[SolutionDict]


def _default_offset_spec(nu_Hz: float) -> OffsetSpec:
    return OffsetSpec(
        Vbins_mV=np.linspace(-0.5, 0.5, 51, dtype=np.float64),
        Ibins_nA=np.linspace(-5.0, 5.0, 181, dtype=np.float64),
        Voff_mV=np.linspace(-0.045, 0.045, 451, dtype=np.float64),
        Ioff_nA=np.linspace(-0.35, 0.35, 701, dtype=np.float64),
        nu_Hz=float(nu_Hz),
        N_up=10,
    )


def _default_sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        Vbins_mV=np.linspace(-1.6, 1.6, 1601, dtype=np.float64),
        Ibins_nA=np.linspace(-30.0, 30.0, 2001, dtype=np.float64),
        apply_offset_correction=True,
        apply_downsampling=True,
        apply_upsampling=True,
        apply_smoothing=True,
        nu_Hz=43.7,
        N_up=1000,
        median_bins=3,
        sigma_bins=2.0,
        mode="nearest",
    )


def _trace_label(index: int, trace: Trace) -> str:
    meta = trace["meta"]
    yvalue = meta.yvalue
    if yvalue is None:
        ytext = "n/a"
    elif numeric_yvalue(yvalue) is not None:
        ytext = f"{float(numeric_yvalue(yvalue)):.6g}"
    else:
        ytext = str(yvalue)
    return f"{index}: {meta.specific_key} | y={ytext}"


def _linspace_from_values(
    start: float,
    stop: float,
    count: int,
    *,
    name: str,
    min_count: int,
) -> NDArray64:
    start = float(start)
    stop = float(stop)
    count = int(count)
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError(f"{name} start/stop must be finite.")
    if count < min_count:
        raise ValueError(f"{name} count must be >= {min_count}.")
    return np.linspace(start, stop, count, dtype=np.float64)


def _fit_sampling_trace(
    sampling: Sample,
    *,
    model: str | BCSModelConfig,
    parameters: list[ParameterSpec],
    maxfev: Optional[int],
) -> SolutionDict:
    V_full = np.asarray(sampling["Vbins_mV"], dtype=np.float64)
    I_full = np.asarray(sampling["I_nA"], dtype=np.float64)
    finite = np.isfinite(V_full) & np.isfinite(I_full)
    if int(np.sum(finite)) < 3:
        raise ValueError("Not enough finite sampled I(V) points for fitting.")

    solution = fit_model(
        V_full[finite],
        I_full[finite],
        model=model,
        parameters=parameters,
        maxfev=maxfev,
    )
    model_spec = get_model_spec(model)
    guess = np.asarray(
        [parameter.guess for parameter in solution["params"]],
        dtype=np.float64,
    )
    values = np.asarray(
        [
            parameter.value if parameter.value is not None else parameter.guess
            for parameter in solution["params"]
        ],
        dtype=np.float64,
    )
    return {
        "V_mV": V_full,
        "I_exp_nA": I_full,
        "I_ini_nA": np.asarray(
            model_spec.function(V_full, *guess),
            dtype=np.float64,
        ),
        "I_fit_nA": np.asarray(
            model_spec.function(V_full, *values),
            dtype=np.float64,
        ),
        "params": solution["params"],
        "weights": None,
        "maxfev": maxfev,
    }


__all__ = [
    "GUIStateDict",
    "_DEFAULT_MODEL",
    "_DEFAULT_SHARED_NU_HZ",
    "_EXPERIMENTAL_TITLES",
    "_default_offset_spec",
    "_default_sampling_spec",
    "_fit_sampling_trace",
    "_linspace_from_values",
    "_trace_label",
]
