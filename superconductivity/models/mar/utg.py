import numpy as np
from numpy.typing import NDArray

from scipy.special import jv
from scipy.interpolate import RegularGridInterpolator

from utilities.constants import h_e_pVs


def get_I_nA(
    A_mV: NDArray[np.float64],
    V_mV: NDArray[np.float64],
    I_nA: NDArray[np.float64],
    nu_GHz: float = 10.0,
    N: int = 200,
    M: int = 10,
) -> NDArray[np.float64]:

    nu_mV = nu_GHz * h_e_pVs
    A___ = A_mV / nu_mV
    _n__ = np.arange(-N, N + 1)
    __m_ = np.arange(1, M + 1)
    ___V_mV = V_mV
    __II_nA = I_nA[:, 1:].T

    nnn_, AAA_, mmm_ = np.meshgrid(_n__, A___, __m_)
    JJJ__n = jv(nnn_, mmm_ * AAA_)
    JJJ__n_2 = JJJ__n * JJJ__n

    _III_nA = __II_nA[np.newaxis, :, :]
    _III_nA = _III_nA * np.ones(
        (
            _n__.shape[0],
            __m_.shape[0],
            ___V_mV.shape[0],
        )
    )

    interp = RegularGridInterpolator(
        (_n__, __m_, ___V_mV),
        _III_nA,
        bounds_error=False,
        fill_value=None,
    )

    _mmm, _nnn, _VVV_mV = np.meshgrid(__m_, _n__, ___V_mV)
    _VVV_nm_mV = _VVV_mV - _nnn / _mmm * nu_mV
    _III_nm_nA = interp(np.stack([_nnn, _mmm, _VVV_nm_mV], axis=-1))

    JJJJ_n_2 = JJJ__n_2[:, :, :, np.newaxis]
    IIII_nA = _III_nm_nA[np.newaxis, :, :, :]
    IIII_nA = JJJJ_n_2 * IIII_nA

    IIII_nA = np.where(np.isnan(IIII_nA), 0.0, IIII_nA)
    I_II_nA = np.sum(IIII_nA, axis=1)  # over n

    I_II_nA = I_II_nA.transpose(0, 2, 1)  # A, I, m
    I__I_nA_0 = np.sum(I_II_nA, axis=2)  # over m
    I_II_nA_0 = I__I_nA_0[:, :, np.newaxis]

    I_nA = np.concatenate((I_II_nA_0, I_II_nA), axis=2)

    return I_nA
