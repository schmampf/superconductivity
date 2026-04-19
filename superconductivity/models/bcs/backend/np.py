"""NumPy BCS current kernels."""

from __future__ import annotations

import numpy as np

from ....utilities.types import NDArray64
from ...basics import get_DeltaT_meV, get_dos, get_f


def integral_np(
    V_mV: NDArray64,
    E_meV: NDArray64,
    T1_K: float,
    T2_K: float,
    Delta1_meV: float,
    Delta2_meV: float,
    gamma1_meV: float,
    gamma2_meV: float,
) -> NDArray64:
    """Evaluate the two-lead SIS integral model with unit conductance."""
    DeltaT1_meV = get_DeltaT_meV(Delta1_meV, T1_K)
    DeltaT2_meV = get_DeltaT_meV(Delta2_meV, T2_K)
    V0_mV = np.asarray(V_mV, dtype=np.float64)
    if DeltaT1_meV == 0.0 and DeltaT2_meV == 0.0:
        return V0_mV

    Vgrid_mV = np.asarray(V_mV, dtype=np.float64)[:, None]
    Egrid_meV = np.asarray(E_meV, dtype=np.float64)[None, :]
    E1_meV = Egrid_meV - Vgrid_mV / 2.0
    E2_meV = Egrid_meV + Vgrid_mV / 2.0

    dos1 = get_dos(E1_meV, DeltaT1_meV, gamma1_meV)
    dos2 = get_dos(E2_meV, DeltaT2_meV, gamma2_meV)
    f1 = get_f(E1_meV, T1_K)
    f2 = get_f(E2_meV, T2_K)
    integrand = dos1 * dos2 * (f1 - f2)
    return np.trapezoid(integrand, np.asarray(E_meV, dtype=np.float64), axis=1)


def convolution_spectrum_np(
    E_meV: NDArray64,
    T1_K: float,
    T2_K: float,
    Delta1_meV: float,
    Delta2_meV: float,
    gamma1_meV: float,
    gamma2_meV: float,
) -> NDArray64:
    """Build the convolution spectrum on the energy grid ``E_meV``."""
    Egrid_meV = np.asarray(E_meV, dtype=np.float64)
    DeltaT1_meV = get_DeltaT_meV(Delta1_meV, T1_K)
    DeltaT2_meV = get_DeltaT_meV(Delta2_meV, T2_K)
    dos1 = get_dos(Egrid_meV, DeltaT1_meV, gamma1_meV)
    dos2 = get_dos(Egrid_meV, DeltaT2_meV, gamma2_meV)
    occupied1 = dos1 * get_f(Egrid_meV, T1_K)
    occupied2 = dos2 * get_f(Egrid_meV, T2_K)
    empty1 = dos1 * (1.0 - get_f(Egrid_meV, T1_K))
    empty2 = dos2 * (1.0 - get_f(Egrid_meV, T2_K))
    dE_meV = float(Egrid_meV[1] - Egrid_meV[0])
    forward = np.correlate(empty2, occupied1, mode="full") * dE_meV
    backward = np.correlate(occupied2, empty1, mode="full") * dE_meV
    return forward - backward


def interpolate_convolution_np(
    V_mV: NDArray64,
    E_meV: NDArray64,
    I_mV: NDArray64,
) -> NDArray64:
    """Interpolate the convolution spectrum back onto the requested bias grid."""
    V_mV = np.asarray(V_mV, dtype=np.float64)
    Egrid_meV = np.asarray(E_meV, dtype=np.float64)
    dE_meV = float(Egrid_meV[1] - Egrid_meV[0])
    Egrid_meV = (
        np.arange(
            -(Egrid_meV.size - 1),
            Egrid_meV.size,
            dtype=np.float64,
        )
        * dE_meV
    )
    result = np.interp(
        V_mV,
        Egrid_meV,
        np.asarray(I_mV, dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    invalid = ~np.isfinite(result)
    if np.any(invalid):
        result[invalid] = V_mV[invalid]
    return result


def convolution_np(
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
    DeltaT1_meV = get_DeltaT_meV(Delta1_meV, T1_K)
    DeltaT2_meV = get_DeltaT_meV(Delta2_meV, T2_K)
    if DeltaT1_meV == 0.0 and DeltaT2_meV == 0.0:
        return np.asarray(V_mV, dtype=np.float64)

    I_mV = convolution_spectrum_np(
        np.asarray(E_meV, dtype=np.float64),
        T1_K=T1_K,
        T2_K=T2_K,
        Delta1_meV=Delta1_meV,
        Delta2_meV=Delta2_meV,
        gamma1_meV=gamma1_meV,
        gamma2_meV=gamma2_meV,
    )
    return interpolate_convolution_np(
        V_mV,
        E_meV,
        I_mV,
    )


__all__ = [
    "convolution_np",
    "convolution_spectrum_np",
    "integral_np",
    "interpolate_convolution_np",
]
