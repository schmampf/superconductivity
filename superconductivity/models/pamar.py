from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

from ..utilities.constants import h_e_pVs
from ..utilities.functions_jax import get_dydx, jnp_interp_y_of_x
from ..utilities.types import JInterpolator, JNDArray, NDArray64
from .pat import get_I_pat_nA


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
