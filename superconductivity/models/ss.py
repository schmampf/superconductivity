"""Shapiro-step visualization helpers.

This module provides small numerical ..utilities to construct idealized
(delta-comb) spectra of integer and fractional Shapiro steps from a
Fourier-expanded current--phase relation (CPR). The routines here are meant
for quick figure generation and sanity checks (they are not dynamical RSJ/RCSJ
simulations and do not include noise or realistic broadening).

Units and conventions
---------------------
- Voltages are expressed in mV and energies in meV.
- Frequencies are expressed in GHz unless stated otherwise.
- `G_N` denotes the dimensionless normal conductance `g = G/G_0`.
  The corresponding physical conductance is `G = G_N * G_0`.
- `nu_mV = (h\nu/e)` is the voltage quantum in mV; for a junction with
  Cooper-pair charge `2e` the Shapiro voltages occur at `V_n = n (h\nu)/(2e)`.

"""

import numpy as np
from scipy.special import jv

from ..utilities.types import NDArray64

from ..utilities.constants import G_0_muS
from ..utilities.constants import h_e_pVs

from .abs import get_Ic_ab
from .abs import get_Ic_ab_nA
from .abs import get_cpr_abs


def get_I_p(
    phi: NDArray64,
    I_phi: NDArray64,
    p_max: int = 10,
):
    """Compute sine-series Fourier coefficients for a 2π-periodic CPR.

    Parameters
    ----------
    phi
        Uniform phase grid on [0, 2π) (use `endpoint=False`).
    I_phi
        CPR samples I(φ) on the same grid.
    p_max
        Maximum harmonic order p to compute.

    Returns
    -------
    NDArray64
        Array of coefficients `I_p` such that
        I(φ) ≈ ∑_{p=1}^{p_{max}} I_p sin(pφ).

    Notes
    -----
    This routine uses direct numerical projection. For strongly skewed CPRs
    (e.g. τ→1 with a cusp near φ=π) increase the grid density.
    """
    # spacing (uniform)
    dphi = phi[1] - phi[0]

    coeffs = np.empty(p_max, dtype=np.float64)
    for p in range(1, p_max + 1):
        coeffs[p - 1] = (1 / np.pi) * np.sum(I_phi * np.sin(p * phi)) * dphi
    return coeffs


def get_I_p_abs(
    Delta_meV: float = 0.18,
    tau: float | NDArray64 = 1.0,
    T_K: float = 0.0,
    p_max: int = 10,
) -> NDArray64:
    """Return the harmonic amplitudes I_p for an ABS CPR (normalized).

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Transmission probability τ ∈ [0,1].
    T_K
        Temperature in kelvin.
    p_max
        Maximum harmonic order p.

    Returns
    -------
    NDArray64
        Harmonic amplitudes I_p (dimensionless).
    """
    phi = np.linspace(0, 2 * np.pi, 1001, endpoint=False)

    I_p = np.zeros((p_max))
    tau = np.asarray(tau)

    for tau_i in tau:
        I_phi = get_cpr_abs(phi=phi, Delta_meV=Delta_meV, tau=tau_i, T_K=T_K)
        I_p += get_I_p(phi=phi, I_phi=I_phi, p_max=p_max)
    return I_p


def get_I_p_abs_nA(
    Delta_meV: float = 0.18,
    tau: float | NDArray64 = 1.0,
    T_K: float = 0.0,
    p_max: int = 10,
) -> NDArray64:
    """Return the harmonic amplitudes I_p for an ABS CPR in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Transmission probability τ ∈ [0,1].
    T_K
        Temperature in kelvin.
    p_max
        Maximum harmonic order p.

    Returns
    -------
    NDArray64
        Harmonic amplitudes I_p in nA.
    """
    I_p_ABS = get_I_p_abs(
        Delta_meV=Delta_meV,
        tau=tau,
        T_K=T_K,
        p_max=p_max,
    )
    I_p_ABS_nA = I_p_ABS * G_0_muS * Delta_meV
    return I_p_ABS_nA


def do_I_fss(
    V_mV: NDArray64,
    A_mV: NDArray64,
    I: NDArray64 = np.array([1.0]),
    nu_GHz: float = 10.0,
    n_max: int = 1000,
) -> NDArray64:
    """Construct a discrete (delta-comb) Shapiro / fractional-Shapiro spectrum.

    This helper places contributions at commensurate voltages

        V_{n/p} = (n/p) (hν) / (2e),

    with weights derived from the CPR harmonics and Bessel functions.

    Parameters
    ----------
    V_mV
        Voltage grid in mV (used only to place peaks onto nearest bins).
    A_mV
        Drive amplitudes V_ac in mV.
    I
        Array of harmonic amplitudes I_p (dimensionless), where index p runs
        from 1..len(I).
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n included in the sum.

    Returns
    -------
    NDArray64
        Array with shape (len(A_mV), len(V_mV)) containing peak amplitudes.
        The amplitude unit follows the unit of the supplied harmonic array `I`
        (dimensionless vs nA).

    Notes
    -----
    This is a visualization tool (not a dynamical RCSJ simulation). Peaks are
    accumulated onto the nearest voltage bin.
    """
    I_p = np.copy(I)

    m = 2
    # Voltage quantum (h\nu/e) expressed in mV.
    # Here `h_e_pVs = (h/e)` in pV·s, so multiplying by ν in GHz yields mV.
    nu_mV = nu_GHz * h_e_pVs
    I_fSS = np.zeros((A_mV.shape[0], V_mV.shape[0]))

    a = np.arange(0, A_mV.shape[0], 1, dtype=np.float64)
    p = np.arange(1, len(I_p) + 1, 1)
    n = np.arange(0, n_max + 1, 1)
    for i_a, _ in enumerate(a):
        for _, n_i in enumerate(n):
            for i_p, p_i in enumerate(p):
                V_np_mV = n_i / p_i * nu_mV / m
                alpha_p = 2 * p_i * A_mV[i_a] / nu_mV
                if np.abs(V_np_mV) <= np.nanmax(np.abs(V_mV)):
                    i_V = [
                        np.argmin(np.abs(V_mV - V_np_mV)),
                        np.argmin(np.abs(V_mV + V_np_mV)),
                    ]
                    J_np = jv(n_i, float(alpha_p))
                    I_np = np.abs(J_np) * np.abs(I_p[i_p])
                    I_fSS[i_a, i_V] += I_np
    return I_fSS


def get_I_ss(
    V_mV: NDArray64,
    A_mV: NDArray64,
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_max: int = 1000,
) -> NDArray64:
    """Integer Shapiro spectrum for a sinusoidal tunnel-junction CPR.

    Parameters
    ----------
    V_mV
        Voltage grid in mV.
    A_mV
        Drive amplitudes V_ac in mV.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n.

    Returns
    -------
    NDArray64
        Dimensionless peak amplitudes placed on the voltage grid.
        The amplitude is in the same normalization as `get_IC_AB`, i.e.
        `I/(G_0*Δ(0))`.
    """
    I_C = get_Ic_ab(Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    I_SS = do_I_fss(
        V_mV=V_mV,
        A_mV=A_mV,
        I=np.array([I_C]),
        nu_GHz=nu_GHz,
        n_max=n_max,
    )
    return I_SS


def get_I_ss_nA(
    V_mV: NDArray64,
    A_mV: NDArray64,
    G_N: float = 1.0,
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_max: int = 1000,
) -> NDArray64:
    """Integer Shapiro spectrum in nA.

    Parameters
    ----------
    V_mV
        Voltage grid in mV.
    A_mV
        Drive amplitudes V_ac in mV.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n.

    Returns
    -------
    NDArray64
        Peak amplitudes in nA placed on the voltage grid.
    """
    I_C_nA = get_Ic_ab_nA(Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    I_SS_nA = do_I_fss(
        V_mV=V_mV, A_mV=A_mV, I=np.array([I_C_nA]), nu_GHz=nu_GHz, n_max=n_max
    )
    return I_SS_nA


def get_I_fss(
    V_mV: NDArray64,
    A_mV: NDArray64,
    tau: NDArray64 = np.array([1.0]),
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_max: int = 1000,
    p_max: int = 10,
) -> NDArray64:
    """Fractional Shapiro spectrum from a Fourier-expanded ABS CPR.

    Parameters
    ----------
    V_mV
        Voltage grid in mV.
    A_mV
        Drive amplitudes V_ac in mV.
    tau
        Iterable of channel transmissions used to build the CPR.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n.
    p_max
        Maximum CPR harmonic order p included.

    Returns
    -------
    NDArray64
        Dimensionless peak amplitudes on the voltage grid.

    Notes
    -----
    The CPR harmonics are obtained by projecting the ABS CPR onto sine
    harmonics. The resulting peak weights scale with |I_p J_n(p a)|.
    """
    I_p = get_I_p_abs(
        tau=tau,
        Delta_meV=Delta_meV,
        T_K=T_K,
        p_max=p_max,
    )
    I_fSS = do_I_fss(
        V_mV=V_mV,
        A_mV=A_mV,
        I=I_p,
        nu_GHz=nu_GHz,
        n_max=n_max,
    )
    return I_fSS


def get_I_fss_nA(
    V_mV: NDArray64,
    A_mV: NDArray64,
    tau: NDArray64 = np.array([1.0]),
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_max: int = 1000,
    p_max: int = 10,
) -> NDArray64:
    """Fractional Shapiro spectrum in nA.

    Parameters
    ----------
    V_mV
        Voltage grid in mV.
    A_mV
        Drive amplitudes V_ac in mV.
    tau
        Iterable of channel transmissions used to build the CPR.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n.
    p_max
        Maximum CPR harmonic order p included.

    Returns
    -------
    NDArray64
        Peak amplitudes in nA on the voltage grid.
    """
    I_fSS = get_I_fss(
        V_mV=V_mV,
        A_mV=A_mV,
        tau=tau,
        Delta_meV=Delta_meV,
        T_K=T_K,
        nu_GHz=nu_GHz,
        n_max=n_max,
        p_max=p_max,
    )
    I_fSS_nA = I_fSS * G_0_muS * Delta_meV
    return I_fSS_nA
