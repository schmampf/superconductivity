from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

from ...utilities.constants import h_e_pVs
from ...utilities.functions_jax import get_dydx, jinterp_y_of_x
from ...utilities.types import JInterpolator, JNDArray, NDArray64


def get_I_pat_nA(
    V_mV: NDArray64,
    I_nA: NDArray64,
    A_mV: NDArray64 | float,
    *,
    nu_GHz: float,
    n_max: int,
    m: int = 1,
    exp: int = 2,
) -> NDArray64:
    V_mV = np.asarray(V_mV, dtype=np.float64)
    I_nA = np.asarray(I_nA, dtype=np.float64)

    A_array = np.asarray(A_mV, dtype=np.float64)
    scalar_A = A_array.ndim == 0
    A_1d = np.atleast_1d(A_array)

    nu_mV = float(nu_GHz) * float(h_e_pVs)
    a_m = m * A_1d / nu_mV
    n = np.arange(-n_max, n_max + 1, dtype=np.int32)

    slope = float(np.mean(np.stack(get_dydx(x=V_mV, y=I_nA)), axis=0))
    interpolator: JInterpolator = jinterp_y_of_x(x=V_mV, y=I_nA, dydx=slope)

    V_nm_mV = (n / m) * nu_mV
    J_n = jv(n[:, None], a_m[None, :])
    if exp == 1:
        J_n_2 = J_n
    elif exp == 2:
        J_n_2 = J_n * J_n
    else:
        J_n_2 = J_n**exp

    I_pat = np.asarray(
        _pat_kernel(
            V_mV=jnp.asarray(V_mV, dtype=jnp.float64),
            V_nm_mV=jnp.asarray(V_nm_mV, dtype=jnp.float64),
            J_n_2=jnp.asarray(J_n_2, dtype=jnp.float64),
            interpolator=interpolator,
        ),
        dtype=np.float64,
    )
    return I_pat[0] if scalar_A else I_pat


@partial(jax.jit, static_argnames=("interpolator",))
def _pat_kernel(
    V_mV: JNDArray,
    V_nm_mV: JNDArray,
    J_n_2: JNDArray,
    interpolator: JInterpolator,
) -> JNDArray:
    V_shift = V_mV[None, :] - V_nm_mV[:, None]
    I_shift = interpolator(V_shift)
    return jnp.einsum("na,nv->av", J_n_2, I_shift)
