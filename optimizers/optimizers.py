"""
document sting
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit  # type: ignore

from models import NDArray64, ModelFunction


def weights_to_sigma(
    weights: NDArray64,
    min_weight: float = 0.0,
) -> tuple[NDArray64, NDArray[np.bool]]:
    """Map reliability weights in [0,1] to sigma for curve_fit."""
    # Avoid exact zeros to prevent infinite weight
    mask = weights > min_weight
    weights = np.where(mask, weights, np.nan)
    sigma = 1.0 / np.sqrt(weights)
    return sigma, mask


def optimizers(
    optimizer: str,
    function: ModelFunction,
    x_data: NDArray64,
    y_data: NDArray64,
    weights: Optional[NDArray64],
    guess: NDArray64,
    lower: NDArray64,
    upper: NDArray64,
    maxfev: Optional[int],
) -> tuple[NDArray64, NDArray64, NDArray64]:
    """
    document sting

    maxfev : int, optional
        The maximum number of calls to the function. If zero, then
        ``100*(N+1)`` is the maximum where N is the number of elements
        in `x0`.
    """
    if weights is None:
        weights = np.ones_like(y_data, dtype="float64")

    sigma, mask = weights_to_sigma(weights=weights)

    match optimizer:

        case "curve_fit":
            results: tuple[NDArray64, NDArray64] = curve_fit(
                f=function,
                xdata=x_data[mask],
                ydata=y_data[mask],
                sigma=sigma[mask],
                absolute_sigma=False,
                p0=guess,
                bounds=(lower, upper),
                maxfev=maxfev,
            )
            popt: NDArray64 = np.array(results[0], dtype=np.float64)
            pcov: NDArray64 = np.array(results[1], dtype=np.float64)
            perr = np.sqrt(np.diag(pcov))

        # case "curve_fit_jax":
        #     from jaxfit import CurveFit  # type: ignore
        #     jcf = CurveFit()
        #     popt, pcov, *_ = jcf.curve_fit(  # type: ignore
        #         f=function,
        #         xdata=V_mV,
        #         ydata=I_nA,
        #         sigma=sigma,
        #         absolute_sigma=True,
        #         p0=guess,
        #         bounds=(lower, upper),
        #         maxfev=maxfev,
        #     )
        #     perr = np.sqrt(np.diag(pcov))

        case _:
            raise KeyError("Optimizer not found.")
    return popt, pcov, perr
