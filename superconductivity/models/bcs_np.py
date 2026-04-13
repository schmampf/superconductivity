"""BCS tunnel-junction quasiparticle current utilities.

This module implements a minimal Dynes-BCS quasiparticle (QP) tunneling model
for SIS and NIS junctions, used to generate `I_QP(V)` characteristics.

Units and conventions
---------------------
- Voltages are expressed in mV and energies in meV.
- Temperatures are expressed in K.
- `G_N` denotes the dimensionless normal conductance `g = G/G_0`.
  The physical conductance is `G = G_N * G_0` with `G_0` in µS.
- Currents are returned in nA.

Notes
-----
The implementation is intended for robust figure generation and fitting, not
for maximally efficient or fully general microscopic modeling.
"""

import numpy as np

from ..utilities.constants import G_0_muS
from ..utilities.functions import bin_y_over_x
from ..utilities.types import NDArray64
from .basics import get_Delta_meV, get_T_c_K, get_dos, get_f


def get_I_bcs_nA(
    V_mV: NDArray64,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    G_N: float = 0.5,
    T_K: float = 0.0,
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
) -> NDArray64:
    """Quasiparticle tunneling current I(V) for NIS/SIS with Dynes DOS.

    Computes the QP current via energy integration of the tunneling expression
    using Dynes-broadened BCS densities of states.

    Parameters
    ----------
    V_mV
        Bias voltage array in mV.
    Delta_meV
        Superconducting gap(s) in meV. Provide a single float for symmetric
        junctions, or a tuple (Δ1, Δ2) for asymmetric cases.
    G_N
        Dimensionless normal conductance `g = G/G_0` (so `G = G_N * G_0`).
    T_K
        Temperature in kelvin.
    gamma_meV
        Dynes broadening(s) in meV. Provide a single float for symmetric
        junctions, or a tuple (γ1, γ2) for asymmetric cases.
    gamma_meV_min
        Minimal Dynes broadening used as a lower bound for numerical stability.

    Returns
    -------
    NDArray64
        Quasiparticle current in nA, evaluated on `V_mV`.

    Notes
    -----
    - For T >= T_c (i.e. Δ(T)=0), returns the ohmic line I = G V.
    - Internally, the integration grid is constructed from the voltage spacing
      and `gamma_meV_min` to resolve sharp features near the gap edge.
    - The implementation assumes particle-hole symmetry and mirrors the
      computed I(V>=0) branch to negative voltages.
    """
    G_N_muS = G_N * G_0_muS

    # Calculate Current, assuming Ohmic behavior
    I_NN_nA = V_mV * G_N_muS

    # take care of type and asymetric case

    if isinstance(Delta_meV, float):
        Delta1_meV, Delta2_meV = Delta_meV, Delta_meV
    elif isinstance(Delta_meV, tuple):
        Delta1_meV, Delta2_meV = Delta_meV
    else:
        raise KeyError("Delta_meV must be float | tuple[float, float]")

    if isinstance(gamma_meV, float):
        gamma1_meV, gamma2_meV = gamma_meV, gamma_meV
    elif isinstance(gamma_meV, tuple):
        gamma1_meV, gamma2_meV = gamma_meV
    else:
        raise KeyError("gamma_meV must be float | tuple[float, float]")

    Delta1_meV_T = get_Delta_meV(Delta_meV=Delta1_meV, T_K=T_K)
    Delta2_meV_T = get_Delta_meV(Delta_meV=Delta2_meV, T_K=T_K)

    gamma1_meV = gamma_meV_min if gamma1_meV < gamma_meV_min else gamma1_meV
    gamma2_meV = gamma_meV_min if gamma2_meV < gamma_meV_min else gamma2_meV

    Delta_meV_T_max = max(Delta1_meV_T, Delta2_meV_T)
    if Delta_meV_T_max == 0.0:
        return I_NN_nA

    # Determine stepsize in V and E
    dV_mV = np.abs(np.nanmax(V_mV) - np.nanmin(V_mV)) / (len(V_mV) - 1)
    V_max_mV = np.max(np.abs(V_mV))

    E_max_meV = np.max([Delta_meV_T_max * 10, V_max_mV])
    dE_meV = np.min([dV_mV, gamma_meV_min])

    # create V and E axis
    V_mV_temp = np.arange(0.0, V_max_mV + dV_mV, dV_mV, dtype="float64")
    E_meV = np.arange(-E_max_meV, E_max_meV + dE_meV, dE_meV, dtype="float64")

    # create meshes
    energy_eV_mesh, voltage_eV_mesh = np.meshgrid(E_meV, V_mV_temp / 2)
    energy1_eV_mesh = energy_eV_mesh - voltage_eV_mesh
    energy2_eV_mesh = energy_eV_mesh + voltage_eV_mesh

    # Calculate the Fermi-Dirac Distribution
    f_E_meV = get_f(E_meV=E_meV, T_K=T_K)
    f1 = np.interp(energy1_eV_mesh, E_meV, f_E_meV, left=1.0, right=0.0)
    f2 = np.interp(energy2_eV_mesh, E_meV, f_E_meV, left=1.0, right=0.0)
    integrand = f1 - f2

    if Delta1_meV_T > 0.0:
        n1 = get_dos(
            E_meV=E_meV,
            Delta_meV=Delta1_meV_T,
            gamma_meV=gamma1_meV,
        )
        # Interpolate the shifted DOS
        N1 = np.interp(energy1_eV_mesh, E_meV, n1, left=1.0, right=1.0)
        integrand *= N1

    if Delta2_meV_T > 0.0:
        n2 = get_dos(
            E_meV=E_meV,
            Delta_meV=Delta2_meV_T,
            gamma_meV=gamma2_meV,
        )
        N2 = np.interp(energy2_eV_mesh, E_meV, n2, left=1.0, right=1.0)
        integrand *= N2

    # Clean up the integrand
    integrand[np.isnan(integrand)] = 0.0

    # Do integration and normalization
    I_meV = np.trapezoid(integrand, E_meV, axis=1)
    I_nA = np.array(I_meV, dtype="float64") * G_N_muS

    # add negative voltage values
    I_nA = np.concatenate((I_nA, -np.flip(I_nA[1:])))
    V_mV_temp = np.concatenate((V_mV_temp, -np.flip(V_mV_temp[1:])))

    # bin over originally obtained V-axis
    I_nA = bin_y_over_x(V_mV_temp, I_nA, V_mV)

    # Fill up values, that are not calculated with dynes
    I_nA = np.where(
        np.abs(V_mV) >= 10 * Delta_meV_T_max,
        I_NN_nA,
        I_nA,
    )

    return I_nA
