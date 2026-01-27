import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from jax import Array, jit, jacfwd
from jaxfit import CurveFit

from models.bcs_jnp import G_0_muS_jax, currents, thermal_energy_gap

# importlib.reload(sys.modules["theory.models.dynes_jnp"])


E_mV_jax: Array = jnp.array(E_mV, dtype="float64")


@jit
def get_dynes_with_derivative(
    V_mV: Array,
    tau: Array,
    T_K: Array,
    Delta_mV: Array,
    Gamma_mV: Array,
) -> tuple[Array, Array]:

    def I_nA_func(V: Array) -> Array:
        Delta_T_mV: Array = thermal_energy_gap(Delta_meV=Delta_mV, T_K=T_K)
        I_mV: Array = currents(
            V_meV=V,
            E_meV=E_mV_jax,
            T_K=T_K,
            Delta_meV=Delta_T_mV,
            Gamma_meV=Gamma_mV,
        )
        I_nA: Array = I_mV * tau * G_0_muS_jax
        return I_nA

    I_nA = I_nA_func(V_mV)
    dIdV = jacfwd(I_nA_func)(V_mV)
    return I_nA, dIdV


def get_dynes_combined(
    V_mV: NDArray,
    E_mV: NDArray,
    tau: float,
    T_K: float,
    Delta_mV: float,
    Gamma_mV: float,
) -> Array:
    V_mV_jax: Array = jnp.array(V_mV, dtype="float64")

    return get_dynes_with_derivative(V_mV=jn)
