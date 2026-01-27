"""Andreev bound state (ABS) and reference current-phase relations.

This module collects small numerical helpers used throughout the project to
construct current-phase relations (CPRs), ABS energies, and reference critical
currents in a *consistent unit convention*.

The routines are primarily intended for figure generation and sanity checks.
They are not optimized for high-performance simulations.

Units and conventions
---------------------
- Energies are expressed in meV and voltages in mV.
- Temperatures are expressed in K.
- `G_N` denotes the *dimensionless* normal conductance

      g = G / G_0,

  i.e. the physical conductance is `G = G_N * G_0`.
- “Dimensionless current” refers to the normalization

      I_dimless = I / (G_0 * Δ(0)),

  where `Δ(0)` is the zero-temperature gap in meV and `G_0` is the conductance
  quantum in µS.
- Conversion to nA uses

      I[nA] = I_dimless * (G_0[µS] * Δ(0)[meV]),

  because µS·mV = nA and 1 meV/e corresponds to 1 mV.

Notes
-----
Several functions here return *dimensionless* quantities even when the input
contains physical units. This is deliberate: it keeps the scaling with `G_N`
explicit and makes it easy to compare against analytic limits.
"""

import numpy as np

from tqdm import tqdm

from utilities.types import NDArray64

from utilities.constants import k_B_meV
from utilities.constants import G_0_muS

from models.bcs_np import get_Delta_meV


def get_Ic_ab(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> float:
    """Ambegaokar--Baratoff critical current in dimensionless units.

    This returns the dimensionless critical current

        I_c / (G_0 * Δ(0)),

    i.e. current normalized to `G_0 * Δ(0)`.

    Parameters
    ----------
    Delta_meV
        Zero-temperature superconducting gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
        For a tunnel junction, the AB result scales linearly with `G_N`.
    T_K
        Temperature in kelvin.

    Returns
    -------
    float
        Dimensionless AB critical current `I_c/(G_0*Δ(0))`.

    Notes
    -----
    Physical Ambegaokar--Baratoff relation:

        I_c(T) R_N = (π Δ(T) / 2e) tanh(Δ(T) / (2 k_B T)).

    Using `R_N = 1/(G_N G_0)` and normalizing by `G_0 Δ(0)` yields the
    dimensionless form implemented here.
    """
    Delta_T_meV = get_Delta_meV(Delta_meV=Delta_meV, T_K=T_K)
    if T_K > 0.0:
        k_T = np.tanh(Delta_T_meV / (2 * k_B_meV * T_K))
    else:
        k_T = 1.0
    IC_AB = np.pi / 2 * G_N * Delta_T_meV / Delta_meV * k_T
    return IC_AB


def get_Ic_ab_nA(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> float:
    """Ambegaokar--Baratoff critical current in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature superconducting gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.

    Notes
    -----
    Conversion uses `I[nA] = I[dimensionless] * (G_0[µS] * Δ(0)[meV])` since
    µS·mV = nA and 1 meV/e corresponds to 1 mV.

    Returns
    -------
    float
        AB critical current in nA.
    """
    IC_AB = get_Ic_ab(Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    IC_AB_nA = IC_AB * G_0_muS * Delta_meV
    return IC_AB_nA


def get_cpr_ab(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """Josephson CPR in the tunnel limit (sinusoidal, AB scale).

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Dimensionless CPR values `I(φ)/(G_0*Δ(0))`.
    """
    I_C = get_Ic_ab(Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    return I_C * np.sin(phi)


def get_cpr_ab_nA(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """Tunnel-junction CPR in nA (sinusoidal, AB scale).

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        CPR values in nA.
    """
    I_C_nA = get_Ic_ab_nA(Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    return I_C_nA * np.sin(phi)


def get_E_abs(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    tau: float = 1.0,
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
) -> NDArray64:
    """ABS energy (dimensionless) for a single channel.

    The returned energy is normalized to the zero-temperature gap Δ(0):

        E_dimless(φ) = E(φ) / Δ(0).

    Temperature enters only through the gap suppression Δ(T).

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    tau
        Channel transmission probability τ ∈ [0, 1].
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Dimensionless ABS energy `E(φ)/Δ(0)`.
    """
    Delta_T_meV = get_Delta_meV(Delta_meV=Delta_meV, T_K=T_K)
    E_ABS = np.sqrt(1 - tau * np.sin(phi / 2) ** 2) * Delta_T_meV / Delta_meV
    return E_ABS


def get_E_abs_meV(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    tau: float = 1.0,
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
) -> NDArray64:
    """ABS energy in meV for a single channel.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    tau
        Channel transmission probability τ ∈ [0, 1].
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        ABS energy in meV.
    """
    E_ABS = get_E_abs(phi=phi, tau=tau, Delta_meV=Delta_meV, T_K=T_K)
    E_ABS_meV = E_ABS * Delta_meV
    return E_ABS_meV


def get_cpr_abs(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """Single-channel ABS supercurrent CPR (dimensionless).

    The CPR is computed from the ABS dispersion via a numerical derivative.
    The return value is dimensionless in the project convention

        I(φ) / (G_0 * Δ(0)).

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Channel transmission probability τ ∈ [0, 1].
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Dimensionless supercurrent `I(φ)/(G_0*Δ(0))`.

    Notes
    -----
    - Finite temperature is included via the usual occupation factor

          tanh(E(φ) / (2 k_B T)).

      Here `E(φ)` is in units of Δ(0), hence the factor `E * Δ(0)` inside the
      tanh argument.
    - The overall scale corresponds to the single-channel contribution;
    converting to nA uses `I[nA] = I[dimensionless] * (G_0[µS] * Δ(0)[meV])`.
    """
    E_abs = get_E_abs(phi=phi, tau=tau, Delta_meV=Delta_meV, T_K=T_K)

    I_abs = -2 * np.pi * np.gradient(E_abs, phi)

    if T_K > 0.0:
        I_abs *= np.tanh(E_abs * Delta_meV / (2 * k_B_meV * T_K))
    return I_abs


def get_cpr_abs_nA(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """Single-channel ABS CPR in nA.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Channel transmission probability τ ∈ [0, 1].
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Supercurrent in nA.
    """
    CPR = get_cpr_abs(phi=phi, Delta_meV=Delta_meV, tau=tau, T_K=T_K)
    CPR_nA = CPR * G_0_muS * Delta_meV
    return CPR_nA


def get_rho(
    tau: NDArray64 = np.arange(1e-5, 1, 1e-5, dtype=np.float64),
    G_N: float = 1.0,
    eps: float = 1e-8,
) -> NDArray64:
    """Dorokhov/DMPK transmission density for a short diffusive contact.

    The implemented density is

        ρ(τ) = G_N / (2 τ √(1-τ)),

    normalized such that

        ∫_0^1 dτ ρ(τ) τ = G_N = G/G_0.

    Parameters
    ----------
    tau
        Transmission grid τ ∈ (0, 1).
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    eps
        Small cutoff to avoid divergences at τ=0 and τ=1.

    Returns
    -------
    NDArray64
        Density ρ(τ) such that ∫_0^1 dτ ρ(τ) τ = G_N = G/G_0.
    """
    tau = np.clip(tau, eps, 1 - eps)
    rho = G_N / (2 * tau * np.sqrt(1 - tau))
    return rho


def get_cpr_ko1(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    dtau: float = 1e-5,
) -> NDArray64:
    """Kulik--Omelyanchuk (KO-1) CPR for a short diffusive weak link.

    KO-1 corresponds to a *diffusive* contact described by the Dorokhov/DMPK
    transmission distribution ρ(τ), integrating the single-channel ABS CPR
    over τ.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.
    dtau
        Transmission step size for the numerical τ integral.

    Returns
    -------
    NDArray64
        Dimensionless KO-1 CPR `I(φ)/(G_0*Δ(0))`.

    Notes
    -----
    Numerical accuracy is controlled by `dtau` and the phase grid density.
    """

    tau = np.arange(dtau, 1, dtau, dtype=np.float64)
    rho = get_rho(tau=tau, G_N=G_N)

    # evaluate I(phi, tau) and integrate over tau
    I_tau_phi = np.empty((tau.size, phi.size), dtype=np.float64)
    for i, tau_i in enumerate(tau):
        I_tau_phi[i, :] = get_cpr_abs(
            phi=phi,
            Delta_meV=Delta_meV,
            tau=tau_i,
            T_K=T_K,
        )

    # integral over tau
    I_phi = np.trapezoid(rho[:, None] * I_tau_phi, tau, axis=0)
    return np.asarray(I_phi)


def get_cpr_ko1_nA(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    dtau: float = 1e-5,
) -> NDArray64:
    """KO-1 CPR in nA.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.
    dtau
        Transmission step size for the numerical τ integral.

    Returns
    -------
    NDArray64
        KO-1 supercurrent in nA.
    """
    CPR_KO1 = get_cpr_ko1(
        phi=phi,
        Delta_meV=Delta_meV,
        G_N=G_N,
        T_K=T_K,
        dtau=dtau,
    )
    CPR_KO1_nA = CPR_KO1 * G_0_muS * Delta_meV
    return CPR_KO1_nA


def get_cpr_ko2(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """Kulik--Omelyanchuk (KO-2) CPR for a short ballistic contact.

    KO-2 corresponds to a *ballistic* short contact with effectively perfect
    transmissions and scales linearly with `G_N`.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Dimensionless KO-2 CPR `I(φ)/(G_0*Δ(0))`.
    """
    return (
        get_cpr_abs(
            phi=phi,
            Delta_meV=Delta_meV,
            tau=1.0,
            T_K=T_K,
        )
        * G_N
    )


def get_cpr_ko2_nA(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """KO-2 CPR in nA.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        KO-2 supercurrent in nA.
    """
    CPR_KO2 = get_cpr_ko2(phi=phi, Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    CPR_KO2_nA = CPR_KO2 * G_0_muS * Delta_meV
    return CPR_KO2_nA


def get_Ic_abs(
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
    n_phi: int = 501,
) -> float:
    """Critical current of the single-channel ABS CPR (dimensionless).

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Channel transmission probability τ ∈ [0, 1].
    T_K
        Temperature in kelvin.
    n_phi
        Number of phase points used to locate the maximum.

    Returns
    -------
    float
        Dimensionless critical current `I_c/(G_0*Δ(0))`.

    Notes
    -----
    The result depends weakly on `n_phi` for sharply peaked CPRs (τ→1).
    """
    phi = np.linspace(0, 2 * np.pi, n_phi)
    CPR = get_cpr_abs(phi=phi, Delta_meV=Delta_meV, tau=tau, T_K=T_K)
    I_C = np.max(CPR)
    return I_C


def get_Ic_abs_nA(
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
    n_phi: int = 501,
) -> float:
    """Critical current of the single-channel ABS CPR in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Channel transmission probability τ ∈ [0, 1].
    T_K
        Temperature in kelvin.
    n_phi
        Number of phase points used to locate the maximum.

    Returns
    -------
    float
        Critical current in nA.
    """
    IC_ABS = get_Ic_abs(
        Delta_meV=Delta_meV,
        tau=tau,
        T_K=T_K,
        n_phi=n_phi,
    )
    IC_ABS_nA = IC_ABS * G_0_muS * Delta_meV
    return IC_ABS_nA


def get_Ic_ko1(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    dtau: float = 1e-4,
    n_phi: int = 501,
) -> float:
    """Critical current of the KO-1 CPR (dimensionless).

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.
    dtau
        Transmission step size for the numerical τ integral.
    n_phi
        Number of phase points used to locate the maximum.

    Returns
    -------
    float
        Dimensionless KO-1 critical current `I_c/(G_0*Δ(0))`.
    """
    phi = np.linspace(0, 2 * np.pi, n_phi)
    CPR = get_cpr_ko1(
        phi=phi,
        Delta_meV=Delta_meV,
        G_N=G_N,
        T_K=T_K,
        dtau=dtau,
    )
    I_C = np.max(CPR)
    return I_C


def get_Ic_ko1_nA(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    dtau: float = 1e-4,
    n_phi: int = 501,
) -> float:
    """Critical current of the KO-1 CPR in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.
    dtau
        Transmission step size for the numerical τ integral.
    n_phi
        Number of phase points used to locate the maximum.

    Returns
    -------
    float
        Critical current in nA.
    """
    IC_KO1 = get_Ic_ko1(
        Delta_meV=Delta_meV,
        G_N=G_N,
        T_K=T_K,
        dtau=dtau,
        n_phi=n_phi,
    )
    IC_KO1_nA = IC_KO1 * G_0_muS * Delta_meV
    return IC_KO1_nA


def get_Ic_ko2(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    n_phi: int = 501,
) -> float:
    """Critical current of the KO-2 CPR (dimensionless).

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.
    n_phi
        Number of phase points used to locate the maximum.

    Returns
    -------
    float
        Dimensionless KO-2 critical current `I_c/(G_0*Δ(0))`.
    """
    phi = np.linspace(0, 2 * np.pi, n_phi)
    CPR = get_cpr_ko2(phi=phi, Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    I_C = np.max(CPR)
    return I_C


def get_Ic_ko2_nA(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    n_phi: int = 501,
) -> float:
    """Critical current of the KO-2 CPR in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.
    n_phi
        Number of phase points used to locate the maximum.

    Returns
    -------
    float
        Critical current in nA.
    """
    IC_KO2 = get_Ic_ko2(
        Delta_meV=Delta_meV,
        G_N=G_N,
        T_K=T_K,
        n_phi=n_phi,
    )
    IC_KO2_nA = IC_KO2 * G_0_muS * Delta_meV
    return IC_KO2_nA


def get_IcT_ab(
    T_K: NDArray64,
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
) -> NDArray64:
    """Temperature dependence of AB critical current (dimensionless).

    Parameters
    ----------
    T_K
        Temperatures in kelvin.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).

    Returns
    -------
    NDArray64
        Dimensionless AB critical current array `I_c(T)/(G_0*Δ(0))`.
    """
    ICT_AB = np.full_like(T_K, np.nan)
    for i, t_K in enumerate(T_K):
        ICT_AB[i] = get_Ic_ab(
            Delta_meV=Delta_meV,
            G_N=G_N,
            T_K=t_K,
        )
    return ICT_AB


def get_IcT_ab_nA(
    T_K: NDArray64,
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
) -> NDArray64:
    """Temperature dependence of AB critical current (nA).

    Parameters
    ----------
    T_K
        Temperatures in kelvin.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).

    Returns
    -------
    NDArray64
        AB critical current in nA.
    """
    ICT_AB = get_IcT_ab(
        T_K=T_K,
        Delta_meV=Delta_meV,
        G_N=G_N,
    )
    ICT_AB_nA = ICT_AB * G_0_muS * Delta_meV
    return ICT_AB_nA


def get_IcT_abs(
    T_K: NDArray64,
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    n_phi: int = 501,
) -> NDArray64:
    """Temperature dependence of the single-channel ABS critical current.

    Parameters
    ----------
    T_K
        Temperatures in kelvin.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Channel transmission probability τ ∈ [0, 1].
    n_phi
        Number of phase points used to locate the maximum.

    Returns
    -------
    NDArray64
        Dimensionless critical current `I_c(T)/(G_0*Δ(0))`.
    """
    ICT_ABS = np.full_like(T_K, np.nan)
    for i, t_K in enumerate(T_K):
        ICT_ABS[i] = get_Ic_abs(
            Delta_meV=Delta_meV,
            tau=tau,
            T_K=t_K,
            n_phi=n_phi,
        )
    return ICT_ABS


def get_IcT_abs_nA(
    T_K: NDArray64,
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    n_phi: int = 501,
) -> NDArray64:
    """
    Temperature dependence of the single-channel ABS critical current (nA).
    """
    ICT_ABS = get_IcT_abs(
        T_K=T_K,
        Delta_meV=Delta_meV,
        tau=tau,
        n_phi=n_phi,
    )
    ICT_ABS_nA = ICT_ABS * G_0_muS * Delta_meV
    return ICT_ABS_nA


def get_IcT_ko1(
    T_K: NDArray64,
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    dtau: float = 1e-4,
    n_phi: int = 501,
) -> NDArray64:
    """Temperature dependence of the KO-1 critical current (dimensionless)."""
    ICT_KO1 = np.full_like(T_K, np.nan)
    for i, t_K in enumerate(tqdm(T_K)):
        ICT_KO1[i] = get_Ic_ko1(
            Delta_meV=Delta_meV,
            G_N=G_N,
            T_K=t_K,
            dtau=dtau,
            n_phi=n_phi,
        )
    return ICT_KO1


def get_IcT_ko1_nA(
    T_K: NDArray64,
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    dtau: float = 1e-4,
    n_phi: int = 501,
) -> NDArray64:
    """Temperature dependence of the KO-1 critical current (nA)."""
    ICT_KO1 = get_IcT_ko1(
        T_K=T_K,
        Delta_meV=Delta_meV,
        G_N=G_N,
        dtau=dtau,
        n_phi=n_phi,
    )
    ICT_KO1_nA = ICT_KO1 * G_0_muS * Delta_meV
    return ICT_KO1_nA


def get_IcT_ko2(
    T_K: NDArray64,
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    n_phi: int = 501,
) -> NDArray64:
    """Temperature dependence of the KO-2 critical current (dimensionless)."""
    ICT_KO2 = np.full_like(T_K, np.nan)
    for i, t_K in enumerate(T_K):
        ICT_KO2[i] = get_Ic_ko2(
            Delta_meV=Delta_meV,
            G_N=G_N,
            T_K=t_K,
            n_phi=n_phi,
        )
    return ICT_KO2


def get_IcT_ko2_nA(
    T_K: NDArray64,
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    n_phi: int = 501,
) -> NDArray64:
    """Temperature dependence of the KO-2 critical current (nA)."""
    ICT_KO2 = get_IcT_ko2(
        T_K=T_K,
        Delta_meV=Delta_meV,
        G_N=G_N,
        n_phi=n_phi,
    )
    ICT_KO2_nA = ICT_KO2 * G_0_muS * Delta_meV
    return ICT_KO2_nA
