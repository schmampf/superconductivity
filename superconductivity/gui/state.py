from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

from ..evaluation.ivdata import IVTrace
from ..evaluation.offset import OffsetSpec, OffsetTrace
from ..evaluation.psd import PSDTrace
from ..evaluation.sampling import SamplingSpec, SamplingTrace
from ..optimizers.bcs import (
    BCSModelConfig,
    ParameterSpec,
    SolutionDict,
    fit_model,
    get_model_spec,
)
from ..utilities.types import NDArray64

_DEFAULT_MODEL = "bcs_conv_jax"
_EXPERIMENTAL_TITLES = {
    "nu_Hz": "<i>&nu;</i> (Hz)",
    "detrend": "Detrend",
    "sigma_I_nA": "<i>&sigma;<sub>I</sub></i> (nA)",
    "sigma_V_mV": "<i>&sigma;<sub>V</sub></i> (mV)",
}


class GUIStateDict(TypedDict):
    active_index: int
    trace: IVTrace
    psd: PSDTrace
    offset: OffsetTrace
    sampling: SamplingTrace
    fit: Optional[SolutionDict]


def _default_offset_spec(nu_Hz: float) -> OffsetSpec:
    return OffsetSpec(
        Vbins_mV=np.linspace(-0.5, 0.5, 51, dtype=np.float64),
        Ibins_nA=np.linspace(-5.0, 5.0, 181, dtype=np.float64),
        Voff_mV=np.linspace(-0.045, 0.045, 451, dtype=np.float64),
        Ioff_nA=np.linspace(-0.35, 0.35, 701, dtype=np.float64),
        nu_Hz=float(nu_Hz),
        upsample=10,
    )


def _default_sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        upsample=10,
        Vbin_mV=np.linspace(-0.5, 0.5, 51, dtype=np.float64),
        Ibin_nA=np.linspace(-5.0, 5.0, 181, dtype=np.float64),
    )


def _trace_label(index: int, trace: IVTrace) -> str:
    yvalue = trace["yvalue"]
    if yvalue is None:
        ytext = "n/a"
    else:
        ytext = f"{float(yvalue):.6g}"
    return f"{index}: {trace['specific_key']} | y={ytext}"


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
    sampling: SamplingTrace,
    *,
    model: str | BCSModelConfig,
    parameters: list[ParameterSpec],
    maxfev: Optional[int],
) -> SolutionDict:
    V_full = np.asarray(sampling["Vbin_mV"], dtype=np.float64)
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
    "_EXPERIMENTAL_TITLES",
    "_default_offset_spec",
    "_default_sampling_spec",
    "_fit_sampling_trace",
    "_linspace_from_values",
    "_trace_label",
]
