"""NumPy BCS current kernels."""

from __future__ import annotations

import numpy as np

from ....utilities.constants import G_0_muS
from ....utilities.types import NDArray64
from ...basics import get_Delta_meV, get_dos, get_f

_G0 = float(G_0_muS)


def integral_current_np(
    V_mV: NDArray64,
    E_mV: NDArray64,
    *,
    GN_G0: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
) -> NDArray64:
    """Evaluate the direct SIS tunneling integral on ``V_mV``."""
    delta_1 = get_Delta_meV(Delta_1_meV, T_K)
    delta_2 = get_Delta_meV(Delta_2_meV, T_K)
    ohmic = np.asarray(V_mV, dtype=np.float64) * (float(GN_G0) * _G0)
    if delta_1 == 0.0 and delta_2 == 0.0:
        return ohmic

    V = np.asarray(V_mV, dtype=np.float64)[:, None]
    E = np.asarray(E_mV, dtype=np.float64)[None, :]
    E_1 = E - V / 2.0
    E_2 = E + V / 2.0

    dos_1 = get_dos(E_1, delta_1, gamma_1_meV)
    dos_2 = get_dos(E_2, delta_2, gamma_2_meV)
    f_1 = get_f(E_1, T_K)
    f_2 = get_f(E_2, T_K)
    integrand = dos_1 * dos_2 * (f_1 - f_2)
    current_meV = np.trapezoid(integrand, np.asarray(E_mV, dtype=np.float64), axis=1)
    return current_meV * (float(GN_G0) * _G0)


def convolution_spectrum_np(
    E_mV: NDArray64,
    *,
    GN_G0: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
) -> NDArray64:
    """Build the convolution spectrum on the energy grid ``E_mV``."""
    delta_1 = get_Delta_meV(Delta_1_meV, T_K)
    delta_2 = get_Delta_meV(Delta_2_meV, T_K)
    dos_1 = get_dos(np.asarray(E_mV, dtype=np.float64), delta_1, gamma_1_meV)
    dos_2 = get_dos(np.asarray(E_mV, dtype=np.float64), delta_2, gamma_2_meV)
    f = get_f(np.asarray(E_mV, dtype=np.float64), T_K)
    occupied_1 = dos_1 * f
    occupied_2 = dos_2 * f
    empty_1 = dos_1 * (1.0 - f)
    empty_2 = dos_2 * (1.0 - f)
    dE = float(np.asarray(E_mV, dtype=np.float64)[1] - np.asarray(E_mV, dtype=np.float64)[0])
    forward = np.correlate(empty_2, occupied_1, mode="full") * dE
    backward = np.correlate(occupied_2, empty_1, mode="full") * dE
    return (forward - backward) * (float(GN_G0) * _G0)


def interpolate_convolution_trace_np(
    V_mV: NDArray64,
    E_mV: NDArray64,
    current_nA: NDArray64,
    *,
    GN_G0: float,
) -> NDArray64:
    """Interpolate the convolution spectrum back onto the requested bias grid."""
    step = float(np.asarray(E_mV, dtype=np.float64)[1] - np.asarray(E_mV, dtype=np.float64)[0])
    current_axis = np.arange(
        -(np.asarray(E_mV, dtype=np.float64).size - 1),
        np.asarray(E_mV, dtype=np.float64).size,
        dtype=np.float64,
    ) * step
    ohmic = np.asarray(V_mV, dtype=np.float64) * (float(GN_G0) * _G0)
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


def integral_np(
    V_mV: NDArray64,
    E_mV: NDArray64,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> NDArray64:
    """Evaluate the symmetric SIS integral model."""
    return integral_current_np(
        V_mV=V_mV,
        E_mV=E_mV,
        GN_G0=GN_G0,
        T_K=T_K,
        Delta_1_meV=Delta_meV,
        Delta_2_meV=Delta_meV,
        gamma_1_meV=gamma_meV,
        gamma_2_meV=gamma_meV,
    )


def convolution_np(
    V_mV: NDArray64,
    E_mV: NDArray64,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> NDArray64:
    """Evaluate the symmetric SIS convolution model."""
    delta_meV = get_Delta_meV(Delta_meV, T_K)
    if delta_meV == 0.0:
        return np.asarray(V_mV, dtype=np.float64) * (float(GN_G0) * _G0)

    current_nA = convolution_spectrum_np(
        np.asarray(E_mV, dtype=np.float64),
        GN_G0=GN_G0,
        T_K=T_K,
        Delta_1_meV=Delta_meV,
        Delta_2_meV=Delta_meV,
        gamma_1_meV=gamma_meV,
        gamma_2_meV=gamma_meV,
    )
    return interpolate_convolution_trace_np(
        V_mV,
        E_mV,
        current_nA,
        GN_G0=GN_G0,
    )


__all__ = [
    "convolution_np",
    "convolution_spectrum_np",
    "integral_current_np",
    "integral_np",
    "interpolate_convolution_trace_np",
]
