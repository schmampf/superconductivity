"""
This module contains functions to compute current I_nA for
given parameters using Bessel functions and interpolation.

Functions:
- get_I_nA: Computes I_nA based on given inputs.
- get_I_nA_of_T: Computes I_nA with temperature dependence.

Naming Convention
    Dimensions are given in form of (A_mV, n, V_mV)
    Name convention is given, by repeating the symbol for each dimension.
    However if its not in a certain dimension, there is an underscore. e.g.:
    (I_I_nA) = (A_mV, V_mV)
    (JJ__n_2_) = (A_mV, n)
    (JJJ_n_2_) = (A_mV, n, V_mV)

    the last underscore is a spacer between symbol and unit. e.g.
    [nu_mV] = mV
    [I_I_nA] = nA
    [JJ__n_2_] = arb. units

    underscore in between correspond to indices of the symbol, so in the end:
    JJ__n_2_ = $J_n^2$, (A_mV, n), [arb. units]

Author: Your Name
Date: 2025-08-11
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv

from utilities.constants import h_e_pVs

ERROR = "A_mV must be float or array. I_nA must match shape of (A_mV, V_mV)."


def get_I_pat_nA(
    A_mV: float | NDArray[np.float64],
    V_mV: NDArray[np.float64],
    I_nA: NDArray[np.float64],
    nu_GHz: float,
    N: int = 100,
) -> NDArray[np.float64]:
    A_is_float = isinstance(A_mV, float)
    A_is_array = isinstance(A_mV, np.ndarray)
    shape_A = np.array(A_mV).shape
    dim_I = I_nA.ndim
    I_is_shape_of_V = I_nA.shape == V_mV.shape
    if A_is_array:
        I_is_shape_of_AV = I_nA.shape == (A_mV.shape[0], V_mV.shape[0])
    else:
        I_is_shape_of_AV = False
    if A_is_float and I_is_shape_of_V:
        return get_I_pat_nA_from_I0_A0(
            A_mV=A_mV,
            V_mV=V_mV,
            I_nA=I_nA,
            nu_GHz=nu_GHz,
            N=N,
        )
    elif A_is_array and shape_A == (1,) and I_is_shape_of_V:
        return get_I_pat_nA_from_I0_A0(
            A_mV=float(A_mV),
            V_mV=V_mV,
            I_nA=I_nA,
            nu_GHz=nu_GHz,
            N=N,
        )
    elif A_is_array and dim_I == 1 and I_is_shape_of_V:
        return get_I_pat_nA_from_I0(
            A_mV=A_mV,
            V_mV=V_mV,
            I_nA=I_nA,
            nu_GHz=nu_GHz,
            N=N,
        )
    elif A_is_array and dim_I == 2 and I_is_shape_of_AV:
        return get_I_pat_nA_from_IT(
            A_mV=A_mV,
            V_mV=V_mV,
            I_nA=I_nA,
            nu_GHz=nu_GHz,
            N=N,
        )
    else:
        raise KeyError(ERROR)


def get_I_pat_nA_from_I0_A0(
    A_mV: float,
    V_mV: NDArray[np.float64],
    I_nA: NDArray[np.float64],
    nu_GHz: float,
    N: int = 100,
) -> NDArray[np.float64]:
    """
    A_mV: float
    V_mV: 1D Array
    I_0_nA: 1D Array (matching V_mV in shape)
    nu_GHz: float
    N: int

    Dimensions are given in form of (n, V_mV).
    """
    nu_mV = nu_GHz * h_e_pVs
    A_arbu = A_mV / nu_mV
    n__arbu = np.arange(-N, N + 1, 1)
    n__mV = n__arbu * nu_mV
    _V_mV = V_mV
    _I_nA = I_nA

    J__n_arbu = jv(n__arbu, A_arbu)
    J__n_2_arbu = J__n_arbu * J__n_arbu
    JJ_n_2_arbu = np.meshgrid(_V_mV, J__n_2_arbu)[1]

    II_0_nA = np.meshgrid(_I_nA, n__mV)[0]
    interp = RegularGridInterpolator(
        (n__mV, _V_mV),
        II_0_nA,
        bounds_error=False,
        fill_value=None,
    )
    VV_mV, nn_mV = np.meshgrid(_V_mV, n__mV)
    VV_n_mV = VV_mV - nn_mV
    nV2_n_mV = np.stack([nn_mV, VV_n_mV], axis=-1)
    II_n_nA = interp(nV2_n_mV)

    II_nA = JJ_n_2_arbu * II_n_nA

    _I_nA = np.sum(II_nA, axis=0)

    return _I_nA


def get_I_pat_nA_from_I0(
    A_mV: NDArray[np.float64],
    V_mV: NDArray[np.float64],
    I_nA: NDArray[np.float64],
    nu_GHz: float,
    N: int = 100,
) -> NDArray[np.float64]:
    """
    A_mV: 1D Array
    V_mV: 1D Array
    I_0_nA: 1D Array (matching V_mV in shape)
    nu_GHz: float
    N: int

    Dimensions are given in form of (A_mV, n, V_mV).
    """

    A___mV = A_mV
    __V_mV = V_mV
    __I_nA = I_nA

    nu_mV = nu_GHz * h_e_pVs
    _n__arbu = np.arange(-N, N + 1, 1)
    A___arbu = A___mV / nu_mV
    _n__mV = _n__arbu * nu_mV

    nn__arbu, AA__arbu = np.meshgrid(_n__arbu, A___arbu)
    JJ__n_arbu = jv(nn__arbu, AA__arbu)
    JJ__n_2_arbu = JJ__n_arbu * JJ__n_arbu

    _II_nA = np.meshgrid(__I_nA, _n__mV)[0]
    interp = RegularGridInterpolator(
        (_n__mV, V_mV),
        _II_nA,
        bounds_error=False,
        fill_value=None,
    )

    _VV_mV, _nn_mV = np.meshgrid(__V_mV, _n__mV)
    _VV_n_mV = _VV_mV - _nn_mV
    _nV2_n_mV = np.stack([_nn_mV, _VV_n_mV], axis=-1)
    _II_n_nA = interp(_nV2_n_mV)

    JJJ_n_2_arbu = JJ__n_2_arbu[:, :, np.newaxis]
    III_n_nA = _II_n_nA[np.newaxis, :, :]
    III_nA = JJJ_n_2_arbu * III_n_nA

    I_I_nA = np.sum(III_nA, axis=1)
    return I_I_nA


def get_I_pat_nA_from_IT(
    A_mV: NDArray[np.float64],
    V_mV: NDArray[np.float64],
    I_nA: NDArray[np.float64],
    nu_GHz: float,
    N: int = 1000,
) -> NDArray[np.float64]:
    """
    A_mV: 1D Array
    V_mV: 1D Array
    I_0_nA: 2D Array (matching (A_mV, V_mV) in shape)
    nu_GHz: float
    N: int

    Dimensions are given in form of (n_arbu, A_mV, V_mV)
    """

    # constant
    nu_mV: float = nu_GHz * h_e_pVs

    # renaming
    _A__mV: NDArray[np.float64] = A_mV
    __V_mV: NDArray[np.float64] = V_mV
    _II_nA: NDArray[np.float64] = I_nA

    # n - axis
    n___arbu: NDArray[np.float64] = np.arange(-N, N + 1, 1, dtype="float64")

    # rescaling
    _A__arbu: NDArray[np.float64] = _A__mV / nu_mV
    n___mV: NDArray[np.float64] = n___arbu * nu_mV

    # calc. meshgrid, with shape (n, A)
    AA__arbu, nn__arbu = np.meshgrid(_A__arbu, n___arbu)

    # calc. meshgrid & ones, with shape (n, A, V)
    AAA_mV, nnn_mV, VVV_mV = np.meshgrid(_A__mV, n___mV, __V_mV)
    ooones: NDArray[np.float64] = np.ones(
        (n___mV.shape[0], _A__mV.shape[0], __V_mV.shape[0]), dtype="float64"
    )

    # calculate $J_n$ (n, A)
    JJ__n_arbu = jv(nn__arbu, AA__arbu)
    JJ__n_2_arbu = JJ__n_arbu * JJ__n_arbu

    # $I$ (n, A, V) [nA]
    III_nA = _II_nA[np.newaxis, :, :]
    III_nA = III_nA * ooones

    # Define RegularGridInterpolator in order to shift I(V) to I(V_n)
    I_n_nA = RegularGridInterpolator(
        (n___mV, _A__mV, __V_mV),
        III_nA,
        bounds_error=False,
        fill_value=None,
    )

    # Calculate shift in V_n and then I_n(V_n)
    VVV_mV -= nnn_mV
    nAV3_n_mV = np.stack([nnn_mV, AAA_mV, VVV_mV], axis=-1)
    III_n_nA = I_n_nA(nAV3_n_mV)

    # add third dimension to $J_n$
    JJJ_n_2_arbu = JJ__n_2_arbu[:, :, np.newaxis]

    # Multiply J_n * I_n
    III_nA = JJJ_n_2_arbu * III_n_nA
    III_nA = np.where(np.isnan(III_nA), 0.0, III_nA)

    # sum over n
    _II_nA = np.sum(III_nA, axis=0)

    return _II_nA
