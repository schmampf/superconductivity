"""JAX BCS current kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ....utilities.types import NDArray64
from ...basics.jnp import get_DeltaT_meV_jnp, get_dos_jnp, get_f_jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def integral_jnp(
    V_mV: jnp.ndarray,
    E_meV: jnp.ndarray,
    T1_K: jnp.ndarray,
    T2_K: jnp.ndarray,
    Delta1_meV: jnp.ndarray,
    Delta2_meV: jnp.ndarray,
    gamma1_meV: jnp.ndarray,
    gamma2_meV: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the two-lead SIS integral model with unit conductance."""
    DeltaT1_meV = get_DeltaT_meV_jnp(Delta1_meV, T1_K)
    DeltaT2_meV = get_DeltaT_meV_jnp(Delta2_meV, T2_K)

    def _one_voltage(voltage: jnp.ndarray) -> jnp.ndarray:
        E1_meV = E_meV - voltage / 2.0
        E2_meV = E_meV + voltage / 2.0
        dos1 = get_dos_jnp(E1_meV, DeltaT1_meV, gamma1_meV)
        dos2 = get_dos_jnp(E2_meV, DeltaT2_meV, gamma2_meV)
        f1 = get_f_jnp(E1_meV, T1_K)
        f2 = get_f_jnp(E2_meV, T2_K)
        integrand = dos1 * dos2 * (f1 - f2)
        return jnp.trapezoid(integrand, E_meV)

    I_mV = jax.vmap(_one_voltage)(V_mV)
    return jnp.where((DeltaT1_meV == 0.0) & (DeltaT2_meV == 0.0), V_mV, I_mV)


@jax.jit
def convolution_spectrum_jnp(
    E_meV: jnp.ndarray,
    T1_K: jnp.ndarray,
    T2_K: jnp.ndarray,
    Delta1_meV: jnp.ndarray,
    Delta2_meV: jnp.ndarray,
    gamma1_meV: jnp.ndarray,
    gamma2_meV: jnp.ndarray,
) -> jnp.ndarray:
    """Build the convolution spectrum on the energy grid ``E_meV``."""
    DeltaT1_meV = get_DeltaT_meV_jnp(Delta1_meV, T1_K)
    DeltaT2_meV = get_DeltaT_meV_jnp(Delta2_meV, T2_K)
    dos1 = get_dos_jnp(E_meV, DeltaT1_meV, gamma1_meV)
    dos2 = get_dos_jnp(E_meV, DeltaT2_meV, gamma2_meV)
    occupied1 = dos1 * get_f_jnp(E_meV, T1_K)
    occupied2 = dos2 * get_f_jnp(E_meV, T2_K)
    empty1 = dos1 * (1.0 - get_f_jnp(E_meV, T1_K))
    empty2 = dos2 * (1.0 - get_f_jnp(E_meV, T2_K))
    dE_meV = E_meV[1] - E_meV[0]
    forward = jnp.correlate(empty2, occupied1, mode="full") * dE_meV
    backward = jnp.correlate(occupied2, empty1, mode="full") * dE_meV
    return forward - backward


@jax.jit
def interpolate_convolution_jnp(
    V_mV: jnp.ndarray,
    E_meV: jnp.ndarray,
    I_mV: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate the convolution spectrum back onto the requested bias grid."""
    dE_meV = E_meV[1] - E_meV[0]
    Egrid_meV = (
        jnp.arange(
            -(E_meV.size - 1),
            E_meV.size,
            dtype=jnp.float64,
        )
        * dE_meV
    )
    result = jnp.interp(V_mV, Egrid_meV, I_mV)
    return jnp.where(jnp.isfinite(result), result, V_mV)


def convolution_jnp(
    V_mV: NDArray64,
    E_meV: NDArray64,
    T1_K: float,
    T2_K: float,
    Delta1_meV: float,
    Delta2_meV: float,
    gamma1_meV: float,
    gamma2_meV: float,
) -> NDArray64:
    """Evaluate the two-lead SIS convolution model with unit conductance."""
    DeltaT1_meV = float(
        np.asarray(get_DeltaT_meV_jnp(jnp.asarray(Delta1_meV), jnp.asarray(T1_K)))
    )
    DeltaT2_meV = float(
        np.asarray(get_DeltaT_meV_jnp(jnp.asarray(Delta2_meV), jnp.asarray(T2_K)))
    )
    if DeltaT1_meV == 0.0 and DeltaT2_meV == 0.0:
        return np.asarray(V_mV, dtype=np.float64)

    I_mV = np.asarray(
        convolution_spectrum_jnp(
            jnp.asarray(E_meV, dtype=jnp.float64),
            jnp.asarray(T1_K, dtype=jnp.float64),
            jnp.asarray(T2_K, dtype=jnp.float64),
            jnp.asarray(DeltaT1_meV, dtype=jnp.float64),
            jnp.asarray(DeltaT2_meV, dtype=jnp.float64),
            jnp.asarray(gamma1_meV, dtype=jnp.float64),
            jnp.asarray(gamma2_meV, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    return np.asarray(
        interpolate_convolution_jnp(
            jnp.asarray(V_mV, dtype=jnp.float64),
            jnp.asarray(E_meV, dtype=jnp.float64),
            jnp.asarray(I_mV, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )


__all__ = [
    "convolution_jnp",
    "convolution_spectrum_jnp",
    "integral_jnp",
    "interpolate_convolution_jnp",
]
