from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ...utilities.constants import G_0_muS, k_B_meV
from ...utilities.types import NDArray64
from .bcs_np import get_delta_np

jax.config.update("jax_enable_x64", True)

_G0 = float(G_0_muS)
_K_B = float(k_B_meV)
_G0_JAX = jnp.array(_G0, dtype=jnp.float64)
_K_B_JAX = jnp.array(_K_B, dtype=jnp.float64)
_CONST_176 = jnp.array(1.764, dtype=jnp.float64)
_CONST_174 = jnp.array(1.74, dtype=jnp.float64)


@jax.jit
def get_delta_jnp(delta_meV: jnp.ndarray, T_K: jnp.ndarray) -> jnp.ndarray:
    T_c = delta_meV / (_CONST_176 * _K_B_JAX)
    safe_T_K = jnp.where(T_K == 0.0, 1.0, T_K)
    thermal_delta = delta_meV * jnp.tanh(
        _CONST_174 * jnp.sqrt(jnp.maximum(T_c / safe_T_K - 1.0, 0.0))
    )
    return jnp.where(
        T_K < 0.0,
        jnp.full_like(delta_meV, jnp.nan),
        jnp.where(
            T_K == 0.0,
            delta_meV,
            jnp.where(T_K < T_c, thermal_delta, jnp.zeros_like(delta_meV)),
        ),
    )


@jax.jit
def get_fermi_jnp(E_meV: jnp.ndarray, T_K: jnp.ndarray) -> jnp.ndarray:
    exponent = jnp.clip(E_meV / (_K_B_JAX * T_K), -100.0, 100.0)
    return jnp.where(
        T_K == 0.0,
        jnp.where(E_meV < 0.0, 1.0, 0.0),
        1.0 / (jnp.exp(exponent) + 1.0),
    )


@jax.jit
def get_dos_jnp(
    E_meV: jnp.ndarray,
    Delta_meV: jnp.ndarray,
    gamma_meV: jnp.ndarray,
) -> jnp.ndarray:
    E_complex = E_meV + 1j * gamma_meV
    dos = jnp.real(E_complex / jnp.sqrt(E_complex**2 - Delta_meV**2))
    dos = jnp.abs(dos)
    dos = jnp.nan_to_num(dos, nan=0.0, posinf=100.0, neginf=0.0)
    clipped = jnp.clip(dos, 0.0, 100.0)
    return jnp.where(Delta_meV == 0.0, jnp.ones_like(E_meV), clipped)


@jax.jit
def integral_current_jnp(
    V_mV: jnp.ndarray,
    E_mV: jnp.ndarray,
    *,
    G_N: jnp.ndarray,
    T_K: jnp.ndarray,
    Delta_1_meV: jnp.ndarray,
    Delta_2_meV: jnp.ndarray,
    gamma_1_meV: jnp.ndarray,
    gamma_2_meV: jnp.ndarray,
) -> jnp.ndarray:
    delta_1 = get_delta_jnp(Delta_1_meV, T_K)
    delta_2 = get_delta_jnp(Delta_2_meV, T_K)
    ohmic = V_mV * (G_N * _G0_JAX)

    def _one_voltage(voltage: jnp.ndarray) -> jnp.ndarray:
        E_1 = E_mV - voltage / 2.0
        E_2 = E_mV + voltage / 2.0
        dos_1 = get_dos_jnp(E_1, delta_1, gamma_1_meV)
        dos_2 = get_dos_jnp(E_2, delta_2, gamma_2_meV)
        f_1 = get_fermi_jnp(E_1, T_K)
        f_2 = get_fermi_jnp(E_2, T_K)
        integrand = dos_1 * dos_2 * (f_1 - f_2)
        return jnp.trapezoid(integrand, E_mV)

    current_meV = jax.vmap(_one_voltage)(V_mV)
    current_nA = current_meV * (G_N * _G0_JAX)
    return jnp.where((delta_1 == 0.0) & (delta_2 == 0.0), ohmic, current_nA)


@jax.jit
def convolution_spectrum_jnp(
    E_mV: jnp.ndarray,
    *,
    G_N: jnp.ndarray,
    T_K: jnp.ndarray,
    Delta_1_meV: jnp.ndarray,
    Delta_2_meV: jnp.ndarray,
    gamma_1_meV: jnp.ndarray,
    gamma_2_meV: jnp.ndarray,
) -> jnp.ndarray:
    delta_1 = get_delta_jnp(Delta_1_meV, T_K)
    delta_2 = get_delta_jnp(Delta_2_meV, T_K)
    dos_1 = get_dos_jnp(E_mV, delta_1, gamma_1_meV)
    dos_2 = get_dos_jnp(E_mV, delta_2, gamma_2_meV)
    f = get_fermi_jnp(E_mV, T_K)
    occupied_1 = dos_1 * f
    occupied_2 = dos_2 * f
    empty_1 = dos_1 * (1.0 - f)
    empty_2 = dos_2 * (1.0 - f)
    dE = E_mV[1] - E_mV[0]
    forward = jnp.correlate(empty_2, occupied_1, mode="full") * dE
    backward = jnp.correlate(occupied_2, empty_1, mode="full") * dE
    return (forward - backward) * (G_N * _G0_JAX)


def _interpolate_convolution_trace(
    V_mV: NDArray64,
    E_mV: NDArray64,
    current_nA: NDArray64,
    *,
    G_N: float,
) -> NDArray64:
    step = float(E_mV[1] - E_mV[0])
    current_axis = np.arange(-(E_mV.size - 1), E_mV.size, dtype=np.float64) * step
    ohmic = np.asarray(V_mV, dtype=np.float64) * (float(G_N) * _G0)
    result = np.interp(
        np.asarray(V_mV, dtype=np.float64),
        current_axis,
        np.asarray(current_nA, dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    invalid = ~np.isfinite(result)
    if np.any(invalid):
        result[invalid] = ohmic[invalid]
    return result


def integral_jax(
    V_mV: NDArray64,
    E_mV: NDArray64,
    G_N: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> NDArray64:
    return np.asarray(
        integral_current_jnp(
            jnp.asarray(V_mV, dtype=jnp.float64),
            jnp.asarray(E_mV, dtype=jnp.float64),
            G_N=jnp.asarray(G_N, dtype=jnp.float64),
            T_K=jnp.asarray(T_K, dtype=jnp.float64),
            Delta_1_meV=jnp.asarray(Delta_meV, dtype=jnp.float64),
            Delta_2_meV=jnp.asarray(Delta_meV, dtype=jnp.float64),
            gamma_1_meV=jnp.asarray(gamma_meV, dtype=jnp.float64),
            gamma_2_meV=jnp.asarray(gamma_meV, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )


def convolution_jax(
    V_mV: NDArray64,
    E_mV: NDArray64,
    G_N: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> NDArray64:
    delta_meV = get_delta_np(Delta_meV, T_K)
    if delta_meV == 0.0:
        return np.asarray(V_mV, dtype=np.float64) * (G_N * _G0)

    current_nA = np.asarray(
        convolution_spectrum_jnp(
            jnp.asarray(E_mV, dtype=jnp.float64),
            G_N=jnp.asarray(G_N, dtype=jnp.float64),
            T_K=jnp.asarray(T_K, dtype=jnp.float64),
            Delta_1_meV=jnp.asarray(Delta_meV, dtype=jnp.float64),
            Delta_2_meV=jnp.asarray(Delta_meV, dtype=jnp.float64),
            gamma_1_meV=jnp.asarray(gamma_meV, dtype=jnp.float64),
            gamma_2_meV=jnp.asarray(gamma_meV, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    return _interpolate_convolution_trace(
        np.asarray(V_mV, dtype=np.float64),
        np.asarray(E_mV, dtype=np.float64),
        current_nA,
        G_N=G_N,
    )
