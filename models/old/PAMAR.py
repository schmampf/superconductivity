import numpy as np
from numpy.typing import NDArray

from scipy.special import jv
from scipy.interpolate import RegularGridInterpolator

import sys

# sys.path.append("/Users/oliver/Documents/p5control-bluefors-evaluation")

from .constants import h_e_pVs


def get_I_nA(
    A_mV: NDArray[np.float64],
    V_mV: NDArray[np.float64],
    I_nA: NDArray[np.float64],
    nu_GHz: float,
    N: int = 200,
    M: int = 10,
) -> NDArray[np.float64]:

    nu_mV = nu_GHz * h_e_pVs
    A = A_mV / nu_mV
    n = np.arange(-N, N + 1)
    m = np.arange(1, M + 1)
    I_nA = I_nA[:, 1:].T

    nn, AA, mm = np.meshgrid(n, A, m)
    JJ_n = jv(nn, mm * AA)
    JJ_n_2 = JJ_n * JJ_n

    II_nA = I_nA
    II_nA = II_nA[np.newaxis, :, :]
    II_nA = II_nA * np.ones((2 * N + 1, M, V_mV.shape[0]))

    interp = RegularGridInterpolator(
        (n, m, V_mV),
        II_nA,
        bounds_error=False,
        fill_value=None,
    )

    mm, nn, VV_mV = np.meshgrid(m, n, V_mV)
    VV_nm_mV = VV_mV - nn / mm * nu_mV
    II_nm_nA = interp(np.stack([nn, mm, VV_nm_mV], axis=-1))

    JJ_n_2 = JJ_n_2[:, :, :, np.newaxis]
    II_nA = II_nm_nA[np.newaxis, :, :, :]
    I_nA = JJ_n_2 * II_nA

    I_nA = np.where(np.isnan(I_nA), 0.0, I_nA)
    I_nA = np.sum(I_nA, axis=1)
    I_nA = np.sum(I_nA, axis=1)

    return I_nA
