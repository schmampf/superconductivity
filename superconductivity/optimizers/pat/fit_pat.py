"""PAT optimizer entry point that mirrors the legacy fit_I_nA API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, TypedDict

import numpy as np
from scipy.optimize import curve_fit

from ...utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ...utilities.types import NDArray64
from .models import get_model


@dataclass
class ParameterSpec:
    name: str
    label: str
    lower: float
    upper: float
    guess: float
    fixed: bool = False
    value: Optional[float] = None
    error: Optional[float] = None


_DEFAULT_PARAMETERS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        name="GN_G0",
        label="<i>G</i><sub>N</sub> (<i>G</i><sub>0</sub>)",
        lower=0.0,
        upper=10.0,
        guess=0.189,
    ),
    ParameterSpec(
        name="T_K",
        label="<i>T</i> (K)",
        lower=0.0,
        upper=1.5,
        guess=0.236,
    ),
    ParameterSpec(
        name="Delta_meV",
        label="<i>Δ</i> (meV)",
        lower=0.18,
        upper=0.21,
        guess=0.195,
    ),
    ParameterSpec(
        name="gamma_meV",
        label="<i>γ</i> (meV)",
        lower=1e-3,
        upper=25e-3,
        guess=4e-3,
    ),
    ParameterSpec(
        name="A_mV",
        label="<i>A</i> (mV)",
        lower=0.0,
        upper=1.0,
        guess=0.0,
        fixed=True,
    ),
    ParameterSpec(
        name="nu_GHz",
        label="<i>ν</i> (GHz)",
        lower=1.0,
        upper=20.0,
        guess=7.8,
    ),
)

PARAMETER_NAMES: tuple[str, ...] = tuple(param.name for param in _DEFAULT_PARAMETERS)
DEFAULT_PARAMETERS: tuple[ParameterSpec, ...] = _DEFAULT_PARAMETERS


class SolutionDict(TypedDict):
    V_mV: NDArray64
    I_exp_nA: NDArray64
    I_ini_nA: NDArray64
    I_fit_nA: NDArray64
    params: Sequence[ParameterSpec]
    weights: Optional[NDArray64]
    maxfev: Optional[int]


def _clone_parameters(
    parameters: Optional[Sequence[ParameterSpec]] = None,
) -> list[ParameterSpec]:
    if parameters is None:
        return [replace(param) for param in _DEFAULT_PARAMETERS]

    if len(parameters) != len(_DEFAULT_PARAMETERS):
        raise ValueError(
            "Parameter list must contain six entries in the expected order."
        )

    cloned: list[ParameterSpec] = []
    for expected, provided in zip(_DEFAULT_PARAMETERS, parameters):
        if expected.name != provided.name:
            raise ValueError(
                f"Parameter '{provided.name}' does not match expected '{expected.name}'"
            )
        cloned.append(replace(provided))
    return cloned


def _parameters_to_arrays(
    parameters: Sequence[ParameterSpec],
) -> tuple[NDArray64, NDArray64, NDArray64, NDArray64]:
    guess = np.array([param.guess for param in parameters], dtype=np.float64)
    lower = np.array([param.lower for param in parameters], dtype=np.float64)
    upper = np.array([param.upper for param in parameters], dtype=np.float64)
    fixed = np.array([param.fixed for param in parameters], dtype=bool)
    return guess, lower, upper, fixed


def _weights_to_sigma(
    weights: Optional[NDArray64], length: int, *, min_weight: float = 0.0
) -> tuple[NDArray64, NDArray[np.bool_]]:
    if weights is None:
        arr = np.ones(length, dtype=np.float64)
    else:
        arr = np.asarray(weights, dtype=np.float64)
        if arr.shape != (length,):
            raise ValueError("weights must be the same length as I_nA.")
    mask = arr > min_weight
    if not np.any(mask):
        raise ValueError("At least one weight must be positive to perform the fit.")
    sigma = np.full((length,), np.nan, dtype=np.float64)
    sigma[mask] = 1.0 / np.sqrt(arr[mask])
    return sigma, mask


def fit_pat(
    V_mV: NDArray64,
    I_nA: NDArray64,
    *,
    parameters: Optional[Sequence[ParameterSpec]] = None,
    weights: Optional[NDArray64] = None,
    maxfev: Optional[int] = None,
    E_mV: Optional[NDArray64] = None,
) -> SolutionDict:
    V = to_1d_float64(V_mV, "V_mV")
    require_all_finite(V, "V_mV")
    require_min_size(V, 3, "V_mV")

    I = to_1d_float64(I_nA, "I_nA")
    require_all_finite(I, "I_nA")
    require_same_shape(I, V, "I_nA", "V_mV")

    parameter_list = _clone_parameters(parameters)
    guess_full, lower_full, upper_full, fixed_full = _parameters_to_arrays(
        parameter_list
    )
    mask_full = ~fixed_full
    guess_free = guess_full[mask_full]
    lower_free = lower_full[mask_full]
    upper_free = upper_full[mask_full]

    sigma, mask = _weights_to_sigma(weights, length=I.size)

    function, parameter_mask = get_model(model="pat", E_mV=E_mV)
    free_mask = parameter_mask & mask_full

    def fixed_function(V_mV: NDArray64, *free_params: tuple[float, ...]) -> NDArray64:
        full = guess_full.copy()
        full[free_mask] = free_params
        return function(V_mV, *full[parameter_mask])

    if free_mask.sum() == 0:
        popt_free = np.empty((0,), dtype=np.float64)
        cov_free = np.zeros((0, 0), dtype=np.float64)
    else:
        popt, pcov = curve_fit(
            f=fixed_function,
            xdata=V[mask],
            ydata=I[mask],
            sigma=sigma[mask],
            absolute_sigma=False,
            p0=guess_free,
            bounds=(lower_free, upper_free),
            maxfev=maxfev,
        )
        popt_free = np.array(popt, dtype=np.float64)
        cov_free = np.array(pcov, dtype=np.float64)
    perr_free = (
        np.sqrt(np.diag(cov_free))
        if cov_free.size
        else np.zeros((0,), dtype=np.float64)
    )

    popt_full = guess_full.copy()
    popt_full[free_mask] = popt_free

    perr_full = np.zeros_like(guess_full, dtype=np.float64)
    perr_full[free_mask] = perr_free

    I_ini = function(V, *guess_full[parameter_mask])
    I_fit = function(V, *popt_full[parameter_mask])

    for idx, param in enumerate(parameter_list):
        param.value = float(popt_full[idx])
        param.error = float(perr_full[idx])

    solution: SolutionDict = {
        "V_mV": V,
        "I_exp_nA": I,
        "I_ini_nA": np.array(I_ini, dtype=np.float64),
        "I_fit_nA": np.array(I_fit, dtype=np.float64),
        "params": tuple(parameter_list),
        "weights": None if weights is None else np.asarray(weights, dtype=np.float64),
        "maxfev": maxfev,
    }
    return solution
