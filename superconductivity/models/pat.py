from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

from ..utilities.constants import h_e_pVs
from ..utilities.functions_jax import get_dydx, jinterp_y_of_x
from ..utilities.types import JInterpolator, JNDArray, NDArray64


def get_I_pat_nA(
    V_mV: NDArray64,  # (Nv,)
    I_nA: NDArray64,  # (Nv,)
    A_mV: NDArray64,  # (Na,)
    nu_GHz: float = 10.0,
    n_max: int = 100,
    m: int = 1,
) -> NDArray64:  # (Na, Nv)
    """
    Returns I_PAT[A, V] with shape (Na, Nv).
    Assumes V_mV is sorted ascending for the interpolation.
    """
    nu_mV: float = nu_GHz * h_e_pVs
    a_m: NDArray64 = m * A_mV / nu_mV  # (Na,)

    # Build up to order n
    n = np.arange(-n_max, n_max + 1, dtype=np.int32)  # (Nn,)

    # Host-side slope estimate is OK, but if you want pure-JAX, replace it.
    # Here: simple endpoint slope in JAX (stable, cheap):
    G_N_muS: float = np.mean(np.stack(get_dydx(x=V_mV, y=I_nA)), axis=0)

    # Build interpolator once per trace; ensure V is sorted for jnp.interp
    I_i_nA: JInterpolator = jinterp_y_of_x(x=V_mV, y=I_nA, dydx=G_N_muS)

    # Voltage shifts  # (Nn,)
    V_nm_mV: NDArray64 = (n / m) * nu_mV

    # Bessel weights  # (Nn, Na)
    J_n = jv(n[:, None], a_m[None, :])  # (Nn, Na)
    J_n_2 = J_n * J_n

    # switch to jnp
    V_mV_j: JNDArray = jnp.asarray(V_mV, dtype=jnp.float64)
    V_nm_mV_j: JNDArray = jnp.array(V_nm_mV, dtype=jnp.float64)
    J_n_2_j: JNDArray = jnp.asarray(J_n_2, dtype=jnp.float64)

    # pat_kernel returns I_PAT with shape (Na, Nv)
    I_pat_j: JNDArray = pat_kernel(
        V_mV=V_mV_j,
        V_nm_mV=V_nm_mV_j,
        J_n_2=J_n_2_j,
        I_i_nA=I_i_nA,
    )  # (Na, Nv)

    I_pat: NDArray64 = np.asarray(I_pat_j, dtype=np.float64)

    return I_pat


@partial(jax.jit, static_argnames=("I_i_nA"))
def pat_kernel(
    V_mV: JNDArray,  # (Nv,)
    V_nm_mV: JNDArray,  # (Nn,)
    J_n_2: JNDArray,  # (Nn, Na)
    I_i_nA: JInterpolator,
) -> JNDArray:

    # Shifted voltages: shape (Nn, Nv)
    V_shift_mV: JNDArray = V_mV[None, :] - V_nm_mV[:, None]

    # Evaluate I(V - V_nm): shape (Nn, Nv)
    I_shift_nA: JNDArray = I_i_nA(V_shift_mV)

    # Contract over n: (Nn, Na) · (Nn, Nv) -> (Na, Nv)
    I_pat: JNDArray = jnp.einsum("na,nv->av", J_n_2, I_shift_nA)

    return I_pat


def get_I_pamar_nA(
    V_mV: NDArray64,  # (Nv,)
    I_nA: NDArray64,  # (Nv,Nm)
    A_mV: NDArray64,  # (Na,)
    nu_GHz: float = 10.0,
    n_max: int = 100,
    m_max: int = 10,
) -> NDArray64:  # (Na, Nv)

    m = np.arange(1, m_max + 1, 1)
    I_pat_m_nA = np.zeros((m.shape[0], A_mV.shape[0], V_mV.shape[0]))
    for i_m, m_i in enumerate(m):
        I_pat_m_nA[i_m, :, :] = get_I_pat_nA(
            V_mV=V_mV,
            I_nA=I_nA[:, i_m],
            A_mV=A_mV,
            nu_GHz=nu_GHz,
            m=m_i,
            n_max=n_max,
        )
    I_pamar_nA = np.sum(I_pat_m_nA, axis=0)
    return I_pamar_nA
