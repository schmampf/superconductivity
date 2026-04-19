"""NumPy BCS thermal and spectral helpers."""

from __future__ import annotations

import numpy as np

from ...utilities.constants import kB_meV_K
from ...utilities.types import NDArray64


def get_Tc_K(Delta_meV: float = 0.18) -> float:
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
    return float(Delta_meV) / (1.764 * float(kB_meV_K))


def get_DeltaT_meV(Delta_meV: float, T_K: float) -> float:
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
    delta_0 = float(Delta_meV)
    temperature = float(T_K)
    if delta_0 < 0.0:
        raise ValueError("Delta_meV must be non-negative.")
    if temperature < 0.0:
        raise ValueError("T_K must be non-negative.")
    if temperature == 0.0:
        return delta_0

    T_c_K = get_Tc_K(delta_0)
    if temperature >= T_c_K:
        return 0.0
    return delta_0 * np.tanh(1.74 * np.sqrt(T_c_K / temperature - 1.0))


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
    temperature = float(T_K)
    if temperature < 0.0:
        raise ValueError("T_K must be non-negative.")

    energy = np.asarray(E_meV, dtype=np.float64)
    if temperature == 0.0:
        return np.where(energy < 0.0, 1.0, 0.0)

    exponent = np.clip(energy / (float(kB_meV_K) * temperature), -100.0, 100.0)
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
    delta = float(Delta_meV)
    gamma = float(gamma_meV)
    if delta < 0.0:
        raise ValueError("Delta_meV must be non-negative.")
    if gamma < 0.0:
        raise ValueError("gamma_meV must be non-negative.")
    if delta == 0.0:
        return np.ones_like(E_meV, dtype=np.float64)

    E_complex_meV = np.asarray(E_meV, dtype=np.complex128) + 1j * gamma
    denominator = np.sqrt(E_complex_meV * E_complex_meV - delta * delta)
    dos = np.real(E_complex_meV / denominator)
    dos = np.abs(dos, dtype=np.float64)
    dos[np.isnan(dos)] = 0.0
    return np.clip(dos, 0.0, 100.0)


__all__ = [
    "get_Tc_K",
    "get_DeltaT_meV",
    "get_f",
    "get_dos",
]
