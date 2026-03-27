from __future__ import annotations

import numpy as np

from ...utilities.constants import G_0_muS, k_B_meV
from ...utilities.types import NDArray64

_G0 = float(G_0_muS)
_K_B = float(k_B_meV)


def get_delta_np(delta_meV: float, T_K: float) -> float:
    if T_K < 0.0:
        raise ValueError("T_K must be non-negative.")
    if delta_meV < 0.0:
        raise ValueError("Delta_meV must be non-negative.")
    if T_K == 0.0:
        return float(delta_meV)

    T_c = float(delta_meV) / (1.764 * _K_B)
    if T_K >= T_c:
        return 0.0
    return float(delta_meV) * np.tanh(1.74 * np.sqrt(T_c / T_K - 1.0))


def get_fermi_np(E_meV: np.ndarray, T_K: float) -> np.ndarray:
    if T_K == 0.0:
        return np.where(E_meV < 0.0, 1.0, 0.0)
    exponent = np.clip(E_meV / (_K_B * T_K), -100.0, 100.0)
    return 1.0 / (np.exp(exponent) + 1.0)


def get_dos_np(E_meV: np.ndarray, Delta_meV: float, gamma_meV: float) -> np.ndarray:
    if Delta_meV == 0.0:
        return np.ones_like(E_meV, dtype=np.float64)

    E_complex = np.asarray(E_meV, dtype=np.complex128) + 1j * gamma_meV
    dos = np.real(E_complex / np.sqrt(E_complex**2 - Delta_meV**2))
    dos = np.maximum(dos, 0.0)
    dos = np.nan_to_num(dos, nan=0.0, posinf=100.0, neginf=0.0)
    return np.clip(dos, 0.0, 100.0)


def integral_current_np(
    V_mV: NDArray64,
    E_mV: NDArray64,
    *,
    G_N: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
) -> NDArray64:
    delta_1 = get_delta_np(Delta_1_meV, T_K)
    delta_2 = get_delta_np(Delta_2_meV, T_K)
    if delta_1 == 0.0 and delta_2 == 0.0:
        return np.asarray(V_mV, dtype=np.float64) * (G_N * _G0)

    V = np.asarray(V_mV, dtype=np.float64)[:, None]
    E = np.asarray(E_mV, dtype=np.float64)[None, :]
    E_1 = E - V / 2.0
    E_2 = E + V / 2.0

    dos_1 = get_dos_np(E_1, delta_1, gamma_1_meV)
    dos_2 = get_dos_np(E_2, delta_2, gamma_2_meV)
    f_1 = get_fermi_np(E_1, T_K)
    f_2 = get_fermi_np(E_2, T_K)
    integrand = dos_1 * dos_2 * (f_1 - f_2)
    current_meV = np.trapezoid(
        integrand,
        np.asarray(E_mV, dtype=np.float64),
        axis=1,
    )
    return current_meV * (G_N * _G0)


def sis_integral_np(
    V_mV: NDArray64,
    E_mV: NDArray64,
    G_N: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> NDArray64:
    return integral_current_np(
        V_mV=V_mV,
        E_mV=E_mV,
        G_N=G_N,
        T_K=T_K,
        Delta_1_meV=Delta_meV,
        Delta_2_meV=Delta_meV,
        gamma_1_meV=gamma_meV,
        gamma_2_meV=gamma_meV,
    )


def sin_integral_np(
    V_mV: NDArray64,
    E_mV: NDArray64,
    G_N: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> NDArray64:
    return integral_current_np(
        V_mV=V_mV,
        E_mV=E_mV,
        G_N=G_N,
        T_K=T_K,
        Delta_1_meV=Delta_meV,
        Delta_2_meV=0.0,
        gamma_1_meV=gamma_meV,
        gamma_2_meV=0.0,
    )
