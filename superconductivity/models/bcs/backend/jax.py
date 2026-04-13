"""JAX BCS current kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ....utilities.constants import G_0_muS
from ....utilities.types import NDArray64
from ...basics.jnp import get_Delta_jnp_meV, get_dos_jnp, get_f_jnp
from .np import interpolate_convolution_trace_np

jax.config.update("jax_enable_x64", True)

_G0 = float(G_0_muS)
_G0_JAX = jnp.array(_G0, dtype=jnp.float64)


@jax.jit
def integral_current_jnp(
    V_mV: jnp.ndarray,
    E_mV: jnp.ndarray,
    *,
    GN_G0: jnp.ndarray,
    T_K: jnp.ndarray,
    Delta_1_meV: jnp.ndarray,
    Delta_2_meV: jnp.ndarray,
    gamma_1_meV: jnp.ndarray,
    gamma_2_meV: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the direct SIS tunneling integral on ``V_mV``."""
    delta_1 = get_Delta_jnp_meV(Delta_1_meV, T_K)
    delta_2 = get_Delta_jnp_meV(Delta_2_meV, T_K)
    ohmic = V_mV * (GN_G0 * _G0_JAX)

    def _one_voltage(voltage: jnp.ndarray) -> jnp.ndarray:
        E_1 = E_mV - voltage / 2.0
        E_2 = E_mV + voltage / 2.0
        dos_1 = get_dos_jnp(E_1, delta_1, gamma_1_meV)
        dos_2 = get_dos_jnp(E_2, delta_2, gamma_2_meV)
        f_1 = get_f_jnp(E_1, T_K)
        f_2 = get_f_jnp(E_2, T_K)
        integrand = dos_1 * dos_2 * (f_1 - f_2)
        return jnp.trapezoid(integrand, E_mV)

    current_meV = jax.vmap(_one_voltage)(V_mV)
    current_nA = current_meV * (GN_G0 * _G0_JAX)
    return jnp.where((delta_1 == 0.0) & (delta_2 == 0.0), ohmic, current_nA)


@jax.jit
def convolution_spectrum_jnp(
    E_mV: jnp.ndarray,
    *,
    GN_G0: jnp.ndarray,
    T_K: jnp.ndarray,
    Delta_1_meV: jnp.ndarray,
    Delta_2_meV: jnp.ndarray,
    gamma_1_meV: jnp.ndarray,
    gamma_2_meV: jnp.ndarray,
) -> jnp.ndarray:
    """Build the convolution spectrum on the energy grid ``E_mV``."""
    delta_1 = get_Delta_jnp_meV(Delta_1_meV, T_K)
    delta_2 = get_Delta_jnp_meV(Delta_2_meV, T_K)
    dos_1 = get_dos_jnp(E_mV, delta_1, gamma_1_meV)
    dos_2 = get_dos_jnp(E_mV, delta_2, gamma_2_meV)
    f = get_f_jnp(E_mV, T_K)
    occupied_1 = dos_1 * f
    occupied_2 = dos_2 * f
    empty_1 = dos_1 * (1.0 - f)
    empty_2 = dos_2 * (1.0 - f)
    dE = E_mV[1] - E_mV[0]
    forward = jnp.correlate(empty_2, occupied_1, mode="full") * dE
    backward = jnp.correlate(occupied_2, empty_1, mode="full") * dE
    return (forward - backward) * (GN_G0 * _G0_JAX)


def integral_jax(
    V_mV: NDArray64,
    E_mV: NDArray64,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> NDArray64:
    """Evaluate the symmetric SIS integral model."""
    return np.asarray(
        integral_current_jnp(
            jnp.asarray(V_mV, dtype=jnp.float64),
            jnp.asarray(E_mV, dtype=jnp.float64),
            GN_G0=jnp.asarray(GN_G0, dtype=jnp.float64),
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
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> NDArray64:
    """Evaluate the symmetric SIS convolution model."""
    delta_meV = float(np.asarray(get_Delta_jnp_meV(jnp.asarray(Delta_meV), jnp.asarray(T_K))))
    if delta_meV == 0.0:
        return np.asarray(V_mV, dtype=np.float64) * (float(GN_G0) * _G0)

    current_nA = np.asarray(
        convolution_spectrum_jnp(
            jnp.asarray(E_mV, dtype=jnp.float64),
            GN_G0=jnp.asarray(GN_G0, dtype=jnp.float64),
            T_K=jnp.asarray(T_K, dtype=jnp.float64),
            Delta_1_meV=jnp.asarray(Delta_meV, dtype=jnp.float64),
            Delta_2_meV=jnp.asarray(Delta_meV, dtype=jnp.float64),
            gamma_1_meV=jnp.asarray(gamma_meV, dtype=jnp.float64),
            gamma_2_meV=jnp.asarray(gamma_meV, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    return interpolate_convolution_trace_np(
        np.asarray(V_mV, dtype=np.float64),
        np.asarray(E_mV, dtype=np.float64),
        current_nA,
        GN_G0=GN_G0,
    )


__all__ = [
    "convolution_jax",
    "convolution_spectrum_jnp",
    "integral_current_jnp",
    "integral_jax",
]
