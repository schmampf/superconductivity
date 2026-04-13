"""Shared BCS-style thermal and spectral helper functions.

These helpers live outside the transport model modules so they can be reused
across tunnel, MAR, and bound-state code without pulling in a full current
model implementation.
"""

from __future__ import annotations

import numpy as np

from ..utilities.constants import k_B_meV
from ..utilities.types import NDArray64


def get_T_c_K(Delta_meV: float = 0.18) -> float:
    """Estimate the BCS critical temperature from a zero-temperature gap.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap ``Delta(0)`` in meV.

    Returns
    -------
    float
        Critical temperature in kelvin.
    """
    return Delta_meV / (1.764 * k_B_meV)


def get_Delta_meV(Delta_meV: float, T_K: float) -> float:
    """Return the weak-coupling BCS gap at temperature ``T_K``.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap ``Delta(0)`` in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    float
        Thermal gap ``Delta(T)`` in meV.
    """
    T_c_K = get_T_c_K(Delta_meV)
    if T_K < 0:
        raise ValueError("Temperature (K) must be non-negative.")
    if T_K >= T_c_K:
        return 0.0
    if T_K == 0:
        return Delta_meV
    return Delta_meV * np.tanh(1.74 * np.sqrt(T_c_K / T_K - 1.0))


def get_f(E_meV: NDArray64, T_K: float) -> NDArray64:
    """Evaluate the Fermi--Dirac occupation in meV units.

    Parameters
    ----------
    E_meV
        Energies in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Occupation values ``f(E)``.
    """
    if T_K < 0:
        raise ValueError("Temperature (K) must be non-negative.")
    E_meV = np.asarray(E_meV, dtype=np.float64)
    if T_K == 0:
        return np.where(E_meV < 0.0, 1.0, 0.0)
    exponent = np.clip(E_meV / (k_B_meV * T_K), -100.0, 100.0)
    return 1.0 / (np.exp(exponent) + 1.0)


def get_dos(E_meV: NDArray64, Delta_meV: float, gamma_meV: float) -> NDArray64:
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
    NDArray64
        Dimensionless density of states normalized to the normal state.
    """
    if Delta_meV < 0:
        raise ValueError("Energy gap (meV) must be non-negative.")
    if gamma_meV < 0:
        raise ValueError("Dynes parameter (meV) must be non-negative.")

    E_complex_meV = np.asarray(E_meV, dtype=np.complex128) + 1j * gamma_meV
    denom = np.sqrt(E_complex_meV * E_complex_meV - Delta_meV * Delta_meV)
    dos = np.real(E_complex_meV / denom)
    dos = np.abs(dos, dtype=np.float64)
    dos[np.isnan(dos)] = 0.0
    return np.clip(dos, 0.0, 100.0)


__all__ = [
    "get_T_c_K",
    "get_Delta_meV",
    "get_f",
    "get_dos",
]
