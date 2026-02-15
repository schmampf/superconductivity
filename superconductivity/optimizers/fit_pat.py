"""
Module Doc String
"""

from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..utilities.types import ModelFunction, ModelType, NDArray64, ParameterType
from .models import get_model
from .optimizers import optimizers


class SolutionDict(TypedDict):
    optimizer: str
    model: str
    V_mV: NDArray64
    I_exp_nA: NDArray64
    I_ini_nA: NDArray64
    I_fit_nA: NDArray64
    guess: NDArray64
    lower: NDArray64
    upper: NDArray64
    fixed: NDArray[np.bool]
    popt: NDArray64
    pcov: NDArray64
    perr: NDArray64
    E_mV: Optional[NDArray64]
    weights: Optional[NDArray64]
    maxfev: Optional[int]
    G_N: float
    T_K: float
    Delta_meV: float
    gamma_meV: float
    A_mV: float
    nu_GHz: float


def fit_I_nA(
    V_mV: NDArray64,
    I_nA: NDArray64,
    G_N: ParameterType = (1.0, (0, 10.0), False),
    T_K: ParameterType = (0.2, (0, 1.5), False),
    Delta_meV: ParameterType = (0.195, (0.18, 0.21), False),
    gamma_meV: ParameterType = (1e-3, (1e-3, 25e-3), False),
    A_mV: ParameterType = (1.0, (0, 10.0), False),
    nu_GHz: ParameterType = (7.8, (1.0, 20.0), False),
    E_mV: Optional[NDArray64] = None,
    weights: Optional[NDArray64] = None,
    model: str = "pat",
    optimizer: str = "curve_fit",
    maxfev: Optional[int] = None,
) -> SolutionDict:
    """
    Doc String
    """

    # Define Parameter
    G_N_0, (G_N_lower, G_N_upper), G_N_fixed = G_N
    T_K_0, (T_K_lower, T_K_upper), T_K_fixed = T_K
    (
        Delta_meV_0,
        (Delta_meV_lower, Delta_meV_upper),
        Delta_meV_fixed,
    ) = Delta_meV
    (
        gamma_meV_0,
        (gamma_meV_lower, gamma_meV_upper),
        gamma_meV_fixed,
    ) = gamma_meV
    A_mV_0, (A_mV_lower, A_mV_upper), A_mV_fixed = A_mV
    nu_GHz_0, (nu_GHz_lower, nu_GHz_upper), nu_GHz_fixed = nu_GHz

    guess_full: NDArray64 = np.array(
        [
            G_N_0,
            T_K_0,
            Delta_meV_0,
            gamma_meV_0,
            A_mV_0,
            nu_GHz_0,
        ],
        dtype="float64",
    )

    lower_full: NDArray64 = np.array(
        [
            G_N_lower,
            T_K_lower,
            Delta_meV_lower,
            gamma_meV_lower,
            A_mV_lower,
            nu_GHz_lower,
        ],
        dtype="float64",
    )

    upper_full: NDArray64 = np.array(
        [
            G_N_upper,
            T_K_upper,
            Delta_meV_upper,
            gamma_meV_upper,
            A_mV_upper,
            nu_GHz_upper,
        ],
        dtype="float64",
    )

    fixed: NDArray[np.bool] = np.array(
        [
            G_N_fixed,
            T_K_fixed,
            Delta_meV_fixed,
            gamma_meV_fixed,
            A_mV_fixed,
            nu_GHz_fixed,
        ],
        dtype="bool",
    )

    # get model
    chosen_model: ModelType = get_model(model=model, E_mV=E_mV)
    function: ModelFunction = chosen_model[0]
    parameter_mask: NDArray[np.bool] = chosen_model[1]

    free_mask = parameter_mask & ~fixed

    guess = guess_full[free_mask]
    lower = lower_full[free_mask]
    upper = upper_full[free_mask]

    def fixed_function(
        V_mV: NDArray64,
        *free_params: tuple[float, ...],
    ) -> NDArray64:
        full_params = guess_full.copy()
        full_params[free_mask] = free_params
        return function(V_mV, *full_params[parameter_mask])

    # optimize with optimizer
    popt, pcov, perr = optimizers(
        optimizer=optimizer,
        function=fixed_function,
        x_data=V_mV,
        y_data=I_nA,
        weights=weights,
        guess=guess,
        lower=lower,
        upper=upper,
        maxfev=maxfev,
    )

    popt_full: NDArray64 = guess_full.copy()
    popt_full[free_mask] = popt

    pcov_full: NDArray64 = np.zeros((len(fixed), len(fixed)), dtype=np.float64)
    pcov_full[np.ix_(free_mask, free_mask)] = pcov

    perr_full: NDArray64 = np.full_like(fixed, 0.0, dtype=np.float64)
    perr_full[free_mask] = perr

    I_exp_nA: NDArray64 = I_nA
    I_ini_nA: NDArray64 = np.array(
        function(V_mV, *guess_full[parameter_mask]), dtype=np.float64
    )
    I_fit_nA: NDArray64 = np.array(
        function(V_mV, *popt_full[parameter_mask]), dtype=np.float64
    )

    solution: SolutionDict = {
        "optimizer": optimizer,
        "model": model,
        "V_mV": V_mV,
        "I_exp_nA": I_exp_nA,
        "I_ini_nA": I_ini_nA,
        "I_fit_nA": I_fit_nA,
        "guess": guess_full,
        "lower": lower_full,
        "upper": upper_full,
        "fixed": fixed,
        "popt": popt_full,
        "pcov": pcov_full,
        "perr": perr_full,
        "E_mV": E_mV,
        "weights": weights,
        "maxfev": maxfev,
        "G_N": popt_full[0],
        "T_K": popt_full[1],
        "Delta_mV": popt_full[2],
        "gamma_mV": popt_full[3],
        "A_mV": popt_full[4],
        "nu_GHz": popt_full[5],
    }

    return solution
