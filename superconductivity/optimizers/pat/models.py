from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from numpy.typing import NDArray

from ...models.bcs_jnp import G_0_muS_jax, get_Delta_jnp_meV, get_i_jnp_meV
from ...models.pat import get_I_pat_nA
from ...utilities.types import ModelType, NDArray64

jax.config.update("jax_enable_x64", True)

_E_GRID: NDArray64 = np.linspace(-2.0, 2.0, 2001, dtype=np.float64)
_E_GRID_JAX: Array = jnp.array(_E_GRID, dtype=jnp.float64)
_T_G_N_MAX = 50


@jit
def _dynes_jnp(
    V_mV: Array,
    G_N: Array,
    T_K: Array,
    Delta_mV: Array,
    gamma_meV: Array,
) -> Array:
    current_vectorized = jax.vmap(
        lambda V_mV: get_i_jnp_meV(
            V_meV=V_mV,
            E_meV=_E_GRID_JAX,
            Delta_1_meV=Delta_mV,
            Delta_2_meV=Delta_mV,
            T_K=T_K,
            gamma_1_meV=gamma_meV,
            gamma_2_meV=gamma_meV,
        ),
        in_axes=0,
    )
    I_meV = current_vectorized(V_mV)
    I_nA = I_meV * G_N * G_0_muS_jax
    return I_nA


def get_model(
    model: str = "pat",
    E_mV: Optional[NDArray64] = None,
    N: Optional[int] = None,
) -> ModelType:
    del E_mV, N

    match model:
        case "pat":

            def get_bcs_pat(
                V_mV: NDArray64,
                G_N: float,
                T_K: float,
                Delta_mV: float,
                gamma_meV: float,
                A_mV: float,
                nu_GHz: float,
            ) -> NDArray64:
                V_mV_jax = jnp.array(V_mV, dtype=jnp.float64)

                Delta_T_meV: Array = get_Delta_jnp_meV(
                    Delta_meV=jnp.array(Delta_mV, dtype=jnp.float64),
                    T_K=jnp.array(T_K, dtype=jnp.float64),
                )

                I_dynes_jax: Array = _dynes_jnp(
                    V_mV=V_mV_jax,
                    G_N=jnp.array(G_N, dtype=jnp.float64),
                    T_K=jnp.array(T_K, dtype=jnp.float64),
                    Delta_mV=jnp.array(Delta_T_meV, dtype=jnp.float64),
                    gamma_meV=jnp.array(gamma_meV, dtype=jnp.float64),
                )
                I_dynes: NDArray64 = np.array(I_dynes_jax, dtype=np.float64)
                return get_I_pat_nA(
                    V_mV=V_mV,
                    I_nA=I_dynes,
                    A_mV=A_mV,
                    nu_GHz=nu_GHz,
                    n_max=_T_G_N_MAX,
                )

            parameter_mask: NDArray[np.bool_] = np.full((6,), True, dtype=bool)
            return get_bcs_pat, parameter_mask
        case _:
            raise KeyError("model not found.")
