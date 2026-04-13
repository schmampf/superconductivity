"""JAX BCS thermal and spectral helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ...utilities.constants import k_B_meV
from ...utilities.types import JNDArray

jax.config.update("jax_enable_x64", True)

_K_B_JAX = jnp.array(k_B_meV, dtype=jnp.float64)
_CONST_176 = jnp.array(1.764, dtype=jnp.float64)
_CONST_174 = jnp.array(1.74, dtype=jnp.float64)


@jax.jit
def get_T_c_jnp_K(Delta_meV: JNDArray) -> JNDArray:
    """Estimate the BCS critical temperature from a gap.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap ``Delta(0)`` in meV.

    Returns
    -------
    JNDArray
        Critical temperature in kelvin.
    """
    return Delta_meV / (_CONST_176 * _K_B_JAX)


@jax.jit
def get_Delta_jnp_meV(Delta_meV: JNDArray, T_K: JNDArray) -> JNDArray:
    """Return the weak-coupling BCS gap at temperature ``T_K``.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap ``Delta(0)`` in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    JNDArray
        Thermal gap ``Delta(T)`` in meV.
    """
    T_c_K = get_T_c_jnp_K(Delta_meV)
    safe_T_K = jnp.where(T_K == 0.0, 1.0, T_K)
    thermal_delta = Delta_meV * jnp.tanh(
        _CONST_174 * jnp.sqrt(jnp.maximum(T_c_K / safe_T_K - 1.0, 0.0))
    )
    return jnp.where(
        T_K < 0.0,
        jnp.full_like(Delta_meV, jnp.nan),
        jnp.where(
            T_K == 0.0,
            Delta_meV,
            jnp.where(T_K < T_c_K, thermal_delta, jnp.zeros_like(Delta_meV)),
        ),
    )


@jax.jit
def get_f_jnp(E_meV: JNDArray, T_K: JNDArray) -> JNDArray:
    """Evaluate the Fermi--Dirac occupation in meV units.

    Parameters
    ----------
    E_meV
        Energies in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    JNDArray
        Occupation values ``f(E)``.
    """
    safe_T_K = jnp.where(T_K == 0.0, 1.0, T_K)
    exponent = jnp.clip(E_meV / (_K_B_JAX * safe_T_K), -100.0, 100.0)
    thermal_f = 1.0 / (jnp.exp(exponent) + 1.0)
    return jnp.where(
        T_K < 0.0,
        jnp.full_like(E_meV, jnp.nan),
        jnp.where(T_K == 0.0, jnp.where(E_meV < 0.0, 1.0, 0.0), thermal_f),
    )


@jax.jit
def get_dos_jnp(
    E_meV: JNDArray,
    Delta_meV: JNDArray,
    gamma_meV: JNDArray,
) -> JNDArray:
    """Return the Dynes-broadened BCS density of states.

    Parameters
    ----------
    E_meV
        Energies in meV.
    Delta_meV
        Superconducting gap in meV.
    gamma_meV
        Dynes broadening in meV.

    Returns
    -------
    JNDArray
        Dimensionless density of states normalized to the normal state.
    """
    E_complex = E_meV + 1j * gamma_meV
    dos = E_complex / jnp.sqrt(E_complex**2 - Delta_meV**2)
    dos = jnp.abs(jnp.real(dos))
    dos = jnp.nan_to_num(dos, nan=0.0, posinf=100.0, neginf=0.0)
    clipped = jnp.clip(dos, 0.0, 100.0)
    return jnp.where(Delta_meV == 0.0, jnp.ones_like(E_meV), clipped)


__all__ = [
    "get_T_c_jnp_K",
    "get_Delta_jnp_meV",
    "get_f_jnp",
    "get_dos_jnp",
]
