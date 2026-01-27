import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax, Array

from utilities.types import NDArray64

from utilities.functions_jax import bin_y_over_x

from utilities.constants import k_B_meV
from utilities.constants import G_0_muS


k_B_meV_jax: Array = jnp.array(k_B_meV)
G_0_muS_jax: Array = jnp.array(G_0_muS)

const176: Array = jnp.array(1.76)
const174: Array = jnp.array(1.74)


jax.config.update("jax_enable_x64", True)


@jit
def thermal_energy_gap(Delta_meV: Array, T_K: Array) -> Array:
    T_C = Delta_meV / (const176 * k_B_meV_jax)
    return jnp.select(
        [T_K < 0.0, T_K == 0.0, (T_K > 0.0) & (T_K < T_C), T_C <= T_K],
        [
            jnp.full_like(Delta_meV, jnp.nan),
            Delta_meV,
            Delta_meV * jnp.tanh(const174 * jnp.sqrt(T_C / T_K - 1.0)),
            jnp.full_like(Delta_meV, 0.0),
        ],
        default=jnp.nan,
    )


@jit
def fermi_distribution(E_meV: Array, T_K: Array) -> Array:
    exponent = E_meV / (k_B_meV_jax * T_K)
    exponent = jnp.clip(exponent, -100.0, 100.0)
    return jnp.where(
        T_K == 0,
        jnp.where(E_meV < 0, 1.0, 0.0),
        1 / (jnp.exp(exponent) + 1),
    )


@jit
def density_of_states(E_meV: Array, Delta_meV: Array, gamma_meV: Array) -> Array:
    # Ensure complex energy
    E_meV_complex = E_meV + 1j * gamma_meV

    # Calculate denominator safely
    N = E_meV_complex / jnp.sqrt(E_meV_complex**2 - Delta_meV**2)
    N = jnp.abs(jnp.real(N))

    # Clip unphysical values and set NaNs to 0
    N = jnp.nan_to_num(N, nan=0.0, posinf=100.0, neginf=0.0)
    N = jnp.clip(N, 0.0, 100.0)
    return N


@jit
def current(
    V_meV: Array,
    E_meV: Array,
    Delta_1_meV: Array,
    Delta_2_meV: Array,
    T_K: Array,
    gamma_1_meV: Array,
    gamma_2_meV: Array,
) -> Array:
    V_mV_over_2 = V_meV / 2
    E1_meV = E_meV - V_mV_over_2
    E2_meV = E_meV + V_mV_over_2
    N1 = density_of_states(
        E_meV=E1_meV,
        Delta_meV=Delta_1_meV,
        gamma_meV=gamma_1_meV,
    )
    N2 = density_of_states(
        E_meV=E2_meV,
        Delta_meV=Delta_2_meV,
        gamma_meV=gamma_2_meV,
    )
    f1 = fermi_distribution(
        E_meV=E1_meV,
        T_K=T_K,
    )
    f2 = fermi_distribution(
        E_meV=E2_meV,
        T_K=T_K,
    )
    integrand = (N1 * N2) * (f1 - f2)
    integrand = jnp.nan_to_num(integrand, nan=0.0, posinf=100.0, neginf=0.0)
    I_meV = jnp.trapezoid(integrand, E_meV, axis=0)
    return I_meV


# for fitting symmetrical junctions
@jit
def currents(
    V_meV: Array,
    E_meV: Array,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> Array:
    current_vectorized = vmap(
        lambda V_mV: current(
            V_meV=V_mV,
            E_meV=E_meV,
            Delta_1_meV=Delta_meV,
            Delta_2_meV=Delta_meV,
            T_K=T_K,
            gamma_1_meV=gamma_meV,
            gamma_2_meV=gamma_meV,
        ),
        in_axes=0,
    )
    return current_vectorized(V_meV)


@jit
def current_over_voltage(
    V_meV: Array,
    V_0_meV: Array,
    E_meV: Array,
    Delta_meV: Array,
    T_K: Array,
    gamma_meV: Array,
) -> Array:

    # thermal energy gap
    Delta_meV = thermal_energy_gap(Delta_meV=Delta_meV, T_K=T_K)

    # Delta_eV = jnp.atleast_1d(Delta_eV)
    # gamma_eV = jnp.atleast_1d(gamma_eV)

    # vectorized current function (over V)
    current_vectorized = vmap(
        lambda V_meV: current(
            V_meV=V_meV,
            E_meV=E_meV,
            Delta_1_meV=Delta_meV[0],
            Delta_2_meV=Delta_meV[1],
            T_K=T_K,
            gamma_1_meV=gamma_meV[0],
            gamma_2_meV=gamma_meV[1],
        ),
        in_axes=0,
    )
    I_meV = lax.cond(
        jnp.all(Delta_meV == 0.0),
        lambda _: V_meV,
        lambda _: current_vectorized(V_meV),
        operand=None,
    )

    I_nA = I_meV * G_0_muS_jax

    # extend to full symmetric I-V curve
    I_nA = jnp.concatenate((I_nA, -jnp.flip(I_nA[1:])))
    V_meV = jnp.concatenate((V_meV, -jnp.flip(V_meV[1:])))

    # bin to original V grid
    I_nA = bin_y_over_x(V_meV, I_nA, V_0_meV)

    return I_nA


def get_I_nA(
    V_mV: NDArray64,
    G_N: float = 1.0,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_min_meV: float = 1e-4,
) -> NDArray64:

    if isinstance(Delta_meV, float):
        Delta_meV: tuple[float, float] = Delta_meV, Delta_meV
    elif isinstance(Delta_meV, tuple):
        Delta_meV: tuple[float, float] = Delta_meV
    else:
        raise KeyError("Delta_meV must be float | tuple[float, float]")

    if isinstance(gamma_meV, float):
        gamma_meV_tuple: tuple[float, float] = gamma_meV, gamma_meV
    elif isinstance(gamma_meV, tuple):
        gamma_meV_tuple: tuple[float, float] = gamma_meV
    else:
        raise KeyError("gamma_meV must be float | tuple[float, float]")
    gamma_meV: NDArray64 = np.array(gamma_meV_tuple, dtype="float64")

    V_0_mV = V_mV
    # voltage axis
    V_max_meV = np.max(np.abs(V_0_mV))
    dV_meV = np.abs(np.nanmax(V_0_mV) - np.nanmin(V_0_mV)) / (V_0_mV.shape[0] - 1)

    # parameter
    Delta_max_meV = np.max(Delta_meV)
    gamma_meV = np.where(
        gamma_meV < gamma_min_meV,
        gamma_min_meV,
        gamma_meV,
    )

    # energy axis
    E_max_meV = np.max([Delta_max_meV * 10, V_max_meV])
    dE_meV = np.min([dV_meV, gamma_min_meV])

    # initialize jax.numpy
    V_meV_jax = jnp.arange(0.0, V_max_meV + dV_meV, dV_meV)
    V_0_meV_jax = jnp.array(V_0_mV)
    E_meV_jax = jnp.arange(-E_max_meV, E_max_meV + dE_meV, dE_meV)
    Delta_meV_jax = jnp.array(Delta_meV)
    T_K_jax = jnp.array(T_K)
    gamma_meV_jax = jnp.array(gamma_meV)

    # do majix
    I_nA = current_over_voltage(
        V_meV=V_meV_jax,
        V_0_meV=V_0_meV_jax,
        E_meV=E_meV_jax,
        Delta_meV=Delta_meV_jax,
        T_K=T_K_jax,
        gamma_meV=gamma_meV_jax,
    )
    I_nA = np.array(I_nA) * G_N

    return I_nA
