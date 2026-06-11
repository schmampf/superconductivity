"""NumPy BCS current kernels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ....utilities.constants import kB_meV_K
from ....utilities.types import NDArray64
from ...basics import get_DeltaT_meV, get_dos, get_f

_ADAPTIVE_GAMMA_FLOOR_MEV = 1e-12
_ADAPTIVE_EPSABS = 1e-10
_ADAPTIVE_EPSREL = 1e-7
_ADAPTIVE_LIMIT = 100
_ADAPTIVE_EDGE_WINDOW_MEV = 2e-2


def adaptive_np(
    V_mV: NDArray64,
    E_meV: NDArray64,
    T1_K: float,
    T2_K: float,
    Delta1_meV: float,
    Delta2_meV: float,
    gamma1_meV: float,
    gamma2_meV: float,
) -> NDArray64:
    """Evaluate the SIS integral with gap-edge-aware adaptive quadrature.

    Exactly zero Dynes broadening is evaluated using a small positive
    numerical floor. The energy integration bounds are taken from
    ``E_meV``; its interior spacing is otherwise ignored.

    Parameters
    ----------
    V_mV
        Bias voltages in mV.
    E_meV
        Energy support whose first and last entries define the integration
        bounds in meV.
    T1_K, T2_K
        Lead temperatures in kelvin.
    Delta1_meV, Delta2_meV
        Zero-temperature superconducting gaps in meV.
    gamma1_meV, gamma2_meV
        Dynes broadenings in meV.

    Returns
    -------
    NDArray64
        Current divided by the normal-state conductance, in mV.
    """
    quad = _import_quad()
    energy = np.asarray(E_meV, dtype=np.float64)
    if energy.ndim != 1 or energy.size < 2:
        raise ValueError("E_meV must be a 1D array with at least two points.")
    if not np.all(np.isfinite(energy)):
        raise ValueError("E_meV must be finite.")
    lower_meV = float(energy[0])
    upper_meV = float(energy[-1])
    if lower_meV >= upper_meV:
        raise ValueError("E_meV bounds must be strictly increasing.")

    gamma1 = _effective_adaptive_gamma(gamma1_meV)
    gamma2 = _effective_adaptive_gamma(gamma2_meV)
    DeltaT1_meV = get_DeltaT_meV(Delta1_meV, T1_K)
    DeltaT2_meV = get_DeltaT_meV(Delta2_meV, T2_K)
    voltages = np.asarray(V_mV, dtype=np.float64)
    if DeltaT1_meV == 0.0 and DeltaT2_meV == 0.0:
        return voltages.copy()

    output = np.empty_like(voltages, dtype=np.float64)
    for index, voltage in np.ndenumerate(voltages):
        output[index] = _adaptive_voltage_integral(
            float(voltage),
            lower_meV=lower_meV,
            upper_meV=upper_meV,
            T1_K=float(T1_K),
            T2_K=float(T2_K),
            DeltaT1_meV=DeltaT1_meV,
            DeltaT2_meV=DeltaT2_meV,
            gamma1_meV=gamma1,
            gamma2_meV=gamma2,
            quad=quad,
        )
    return output


def _import_quad() -> Callable[..., tuple[float, float]]:
    """Import SciPy quadrature lazily for the optional adaptive kernel."""
    try:
        from scipy.integrate import quad
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The adaptive BCS kernel requires scipy. "
            "Install scipy in the active environment."
        ) from exc
    return quad


def _effective_adaptive_gamma(gamma_meV: float) -> float:
    """Validate Dynes broadening and apply the adaptive numerical floor."""
    gamma = float(gamma_meV)
    if not np.isfinite(gamma):
        raise ValueError("gamma_meV must be finite.")
    if gamma < 0.0:
        raise ValueError("gamma_meV must be non-negative.")
    return max(gamma, _ADAPTIVE_GAMMA_FLOOR_MEV)


def _adaptive_voltage_integral(
    voltage_mV: float,
    *,
    lower_meV: float,
    upper_meV: float,
    T1_K: float,
    T2_K: float,
    DeltaT1_meV: float,
    DeltaT2_meV: float,
    gamma1_meV: float,
    gamma2_meV: float,
    quad: Callable[..., tuple[float, float]],
) -> float:
    """Integrate one voltage after splitting around shifted gap edges."""
    edges = (
        (voltage_mV / 2.0 - DeltaT1_meV, gamma1_meV),
        (voltage_mV / 2.0 + DeltaT1_meV, gamma1_meV),
        (-voltage_mV / 2.0 - DeltaT2_meV, gamma2_meV),
        (-voltage_mV / 2.0 + DeltaT2_meV, gamma2_meV),
    )
    points = _adaptive_breakpoints(
        edges,
        lower_meV=lower_meV,
        upper_meV=upper_meV,
    )

    def integrand(energy_meV: float) -> float:
        E1_meV = energy_meV - voltage_mV / 2.0
        E2_meV = energy_meV + voltage_mV / 2.0
        dos1 = _dynes_dos_scalar(E1_meV, DeltaT1_meV, gamma1_meV)
        dos2 = _dynes_dos_scalar(E2_meV, DeltaT2_meV, gamma2_meV)
        occupation1 = _fermi_scalar(E1_meV, T1_K)
        occupation2 = _fermi_scalar(E2_meV, T2_K)
        return dos1 * dos2 * (occupation1 - occupation2)

    result = 0.0
    for interval_lower, interval_upper in zip(points[:-1], points[1:]):
        value, _ = quad(
            integrand,
            float(interval_lower),
            float(interval_upper),
            epsabs=_ADAPTIVE_EPSABS,
            epsrel=_ADAPTIVE_EPSREL,
            limit=_ADAPTIVE_LIMIT,
        )
        result += value
    return float(result)


def _adaptive_breakpoints(
    edges: tuple[tuple[float, float], ...],
    *,
    lower_meV: float,
    upper_meV: float,
) -> NDArray64:
    """Return sorted integration splits concentrated around gap edges."""
    points = [lower_meV, upper_meV]
    for edge_meV, gamma_meV in edges:
        if lower_meV < edge_meV < upper_meV:
            points.append(edge_meV)
        scale_meV = gamma_meV
        while scale_meV <= _ADAPTIVE_EDGE_WINDOW_MEV:
            left = edge_meV - scale_meV
            right = edge_meV + scale_meV
            if lower_meV < left < upper_meV:
                points.append(left)
            if lower_meV < right < upper_meV:
                points.append(right)
            scale_meV *= 10.0
    return np.unique(np.asarray(points, dtype=np.float64))


def _dynes_dos_scalar(
    energy_meV: float,
    Delta_meV: float,
    gamma_meV: float,
) -> float:
    """Return the unclipped Dynes DOS for scalar adaptive integration."""
    if Delta_meV == 0.0:
        return 1.0
    energy = complex(energy_meV, gamma_meV)
    denominator = np.sqrt(energy * energy - Delta_meV * Delta_meV)
    return float(abs(np.real(energy / denominator)))


def _fermi_scalar(energy_meV: float, T_K: float) -> float:
    """Return the scalar Fermi occupation without allocating arrays."""
    if T_K < 0.0:
        raise ValueError("T_K must be non-negative.")
    if T_K == 0.0:
        return float(energy_meV < 0.0)
    exponent = np.clip(energy_meV / (float(kB_meV_K) * T_K), -100.0, 100.0)
    return float(1.0 / (np.exp(exponent) + 1.0))


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
    "adaptive_np",
    "convolution_np",
    "convolution_spectrum_np",
    "integral_np",
    "interpolate_convolution_np",
]
