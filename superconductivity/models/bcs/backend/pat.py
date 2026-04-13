"""Photon-assisted tunneling helpers."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

from ....utilities.constants import h_pVs
from ....utilities.functions_jax import get_dydx, jinterp_y_of_x
from ....utilities.types import JInterpolator, JNDArray, NDArray64


def get_I_pat_nA(
    V_mV: NDArray64,
    I_nA: NDArray64,
    A_mV: NDArray64 | float,
    nu_GHz: float = 10.0,
    n_max: int = 100,
    m: int = 1,
    exp: int = 2,
) -> NDArray64:
    """Apply a Tien-Gordon PAT transform to a current trace.

    Parameters
    ----------
    V_mV
        Bias-voltage grid in mV.
    I_nA
        Base current on ``V_mV`` in nA.
    A_mV
        Drive amplitude in mV. A scalar returns a 1D trace; a 1D array returns
        one trace per amplitude.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum PAT sideband order.
    m
        Harmonic index.
    exp
        Power applied to the Bessel weights.

    Returns
    -------
    NDArray64
        PAT-transformed current trace(s).
    """
    V_mV = np.asarray(V_mV, dtype=np.float64)
    I_nA = np.asarray(I_nA, dtype=np.float64)

    A_mV_arr = np.asarray(A_mV, dtype=np.float64)
    scalar_A = A_mV_arr.ndim == 0
    A_mV_1d: NDArray64 = np.atleast_1d(A_mV_arr)

    nu_mV = float(nu_GHz) * float(h_pVs)
    a_m: NDArray64 = m * A_mV_1d / nu_mV
    n = np.arange(-n_max, n_max + 1, dtype=np.int32)

    GN_G0_muS = np.mean(np.stack(get_dydx(x=V_mV, y=I_nA)), axis=0)
    I_i_nA: JInterpolator = jinterp_y_of_x(x=V_mV, y=I_nA, dydx=GN_G0_muS)

    V_nm_mV: NDArray64 = (n / m) * nu_mV

    J_n = jv(n[:, None], a_m[None, :])
    if exp == 1:
        J_n_pow = J_n
    elif exp == 2:
        J_n_pow = J_n * J_n
    else:
        J_n_pow = J_n**exp

    I_pat_j: JNDArray = _pat_kernel(
        V_mV=jnp.asarray(V_mV, dtype=jnp.float64),
        V_nm_mV=jnp.asarray(V_nm_mV, dtype=jnp.float64),
        J_n_pow=jnp.asarray(J_n_pow, dtype=jnp.float64),
        I_i_nA=I_i_nA,
    )
    I_pat = np.asarray(I_pat_j, dtype=np.float64)
    if scalar_A:
        return I_pat[0]
    return I_pat


@partial(jax.jit, static_argnames=("I_i_nA",))
def _pat_kernel(
    V_mV: JNDArray,
    V_nm_mV: JNDArray,
    J_n_pow: JNDArray,
    I_i_nA: JInterpolator,
) -> JNDArray:
    """Evaluate the PAT sideband sum."""
    V_shift_mV: JNDArray = V_mV[None, :] - V_nm_mV[:, None]
    I_shift_nA: JNDArray = I_i_nA(V_shift_mV)
    return jnp.einsum("na,nv->av", J_n_pow, I_shift_nA)


__all__ = ["get_I_pat_nA"]
