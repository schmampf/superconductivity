from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

from ...utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ...utilities.types import NDArray64
from .parameters import ParameterSpec
from .registry import BCSModelConfig, get_model_spec


class SolutionDict(TypedDict):
    V_mV: NDArray64
    I_exp_nA: NDArray64
    I_ini_nA: NDArray64
    I_fit_nA: NDArray64
    params: Sequence[ParameterSpec]
    weights: Optional[NDArray64]
    maxfev: Optional[int]


def _clone_parameters(
    *,
    model: str | BCSModelConfig,
    parameters: Optional[Sequence[ParameterSpec]] = None,
) -> list[ParameterSpec]:
    model_spec = get_model_spec(model)
    defaults = model_spec.parameters
    if parameters is None:
        return [replace(parameter) for parameter in defaults]

    if len(parameters) != len(defaults):
        raise ValueError(
            f"Model '{model}' expects {len(defaults)} parameters, "
            f"got {len(parameters)}."
        )

    cloned: list[ParameterSpec] = []
    for default, provided in zip(defaults, parameters):
        if default.name != provided.name:
            raise ValueError(
                f"Parameter '{provided.name}' does not match expected "
                f"'{default.name}'."
            )
        cloned.append(replace(provided))
    return cloned


def _parameters_to_arrays(
    parameters: Sequence[ParameterSpec],
) -> tuple[NDArray64, NDArray64, NDArray64, NDArray[np.bool_]]:
    guess = np.array([parameter.guess for parameter in parameters], dtype=np.float64)
    lower = np.array([parameter.lower for parameter in parameters], dtype=np.float64)
    upper = np.array([parameter.upper for parameter in parameters], dtype=np.float64)
    fixed = np.array([parameter.fixed for parameter in parameters], dtype=bool)
    return guess, lower, upper, fixed


def _weights_to_sigma(
    weights: Optional[NDArray64],
    *,
    length: int,
) -> tuple[Optional[NDArray64], NDArray[np.bool_]]:
    if weights is None:
        return None, np.ones(length, dtype=bool)

    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError("weights must have the same shape as I_nA.")
    require_all_finite(array, "weights")
    if np.any(array < 0.0):
        raise ValueError("weights must be non-negative.")

    mask = array > 0.0
    if not np.any(mask):
        raise ValueError("At least one weight must be positive.")

    sigma = np.full(length, np.nan, dtype=np.float64)
    sigma[mask] = 1.0 / np.sqrt(array[mask])
    return sigma, mask


def fit_model(
    V_mV: NDArray64,
    I_nA: NDArray64,
    *,
    model: str | BCSModelConfig,
    parameters: Optional[Sequence[ParameterSpec]] = None,
    weights: Optional[NDArray64] = None,
    maxfev: Optional[int] = None,
) -> SolutionDict:
    model_spec = get_model_spec(model)
    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_model requires scipy. Install scipy in the active environment."
        ) from exc

    V = to_1d_float64(V_mV, "V_mV")
    require_all_finite(V, "V_mV")
    require_min_size(V, 3, "V_mV")

    I = to_1d_float64(I_nA, "I_nA")
    require_all_finite(I, "I_nA")
    require_same_shape(I, V, "I_nA", "V_mV")

    parameter_list = _clone_parameters(model=model, parameters=parameters)
    guess, lower, upper, fixed = _parameters_to_arrays(parameter_list)
    free = ~fixed

    sigma, mask = _weights_to_sigma(weights, length=I.size)

    def fixed_function(V_axis: NDArray64, *free_values: float) -> NDArray64:
        full = guess.copy()
        full[free] = np.asarray(free_values, dtype=np.float64)
        return np.asarray(model_spec.function(V_axis, *full), dtype=np.float64)

    if np.any(free):
        popt, pcov = curve_fit(
            f=fixed_function,
            xdata=V[mask],
            ydata=I[mask],
            p0=guess[free],
            sigma=None if sigma is None else sigma[mask],
            absolute_sigma=False,
            bounds=(lower[free], upper[free]),
            maxfev=maxfev,
        )
        popt_free = np.asarray(popt, dtype=np.float64)
        cov_free = np.asarray(pcov, dtype=np.float64)
        perr_free = np.sqrt(np.diag(cov_free))
    else:
        popt_free = np.empty((0,), dtype=np.float64)
        perr_free = np.empty((0,), dtype=np.float64)

    popt_full = guess.copy()
    popt_full[free] = popt_free

    perr_full = np.zeros_like(guess)
    perr_full[free] = perr_free

    I_ini = np.asarray(model_spec.function(V, *guess), dtype=np.float64)
    I_fit = np.asarray(model_spec.function(V, *popt_full), dtype=np.float64)

    for index, parameter in enumerate(parameter_list):
        parameter.value = float(popt_full[index])
        parameter.error = float(perr_full[index])

    return {
        "V_mV": V,
        "I_exp_nA": I,
        "I_ini_nA": I_ini,
        "I_fit_nA": I_fit,
        "params": tuple(parameter_list),
        "weights": None if weights is None else np.asarray(weights, dtype=np.float64),
        "maxfev": maxfev,
    }
