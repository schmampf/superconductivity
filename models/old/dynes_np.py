import numpy as np
import sys

sys.path.append("/Users/oliver/Documents/p5control-bluefors-evaluation")

from theory.models.constants import k_B_eV
from theory.models.constants import G_0_S as G_0

from theory.models.functions import bin_y_over_x


def handle_dynes_parameter_eV(
    dynes_parameter_eV: float | tuple[float, float] = 0.0,
    min_dynes_paramter_eV: float = 1e-6,
) -> tuple[float, float]:
    """Ensure Dynes parameters are non-negative and above a threshold."""
    if isinstance(dynes_parameter_eV, (int, float)):
        if dynes_parameter_eV < 0:
            raise ValueError("Dynes parameter (eV) must be non-negative.")
        if dynes_parameter_eV < min_dynes_paramter_eV:
            # warnings.warn(
            #     f"Dynes parameter Γ = {dynes_parameter_eV:.1e} eV is below threshold; using {min_dynes_paramter_eV:.1e} eV instead.",
            #     category=UserWarning,
            # )
            dynes_parameter_eV = min_dynes_paramter_eV
        Gamma1_eV = dynes_parameter_eV
        Gamma2_eV = dynes_parameter_eV
    elif isinstance(dynes_parameter_eV, tuple) and len(dynes_parameter_eV) == 2:
        Gamma1_eV, Gamma2_eV = dynes_parameter_eV
        if Gamma1_eV < 0 or Gamma2_eV < 0:
            raise ValueError("Dynes parameters (eV) must be non-negative.")
        # if Gamma1_eV < min_dynes_paramter_eV or Gamma2_eV < min_dynes_paramter_eV:
        # warnings.(
        #     f"Dynes parameters Γ = {Gamma1_eV:.1e} eV and Γ warn= {Gamma2_eV:.1e} eV are below threshold; using {min_dynes_paramter_eV:.1e} eV instead.",
        #     category=UserWarning,
        # )
        Gamma1_eV = max(Gamma1_eV, min_dynes_paramter_eV)
        Gamma2_eV = max(Gamma2_eV, min_dynes_paramter_eV)
    else:
        raise ValueError("dynes_parameter_eV must be a float or a tuple of two floats.")
    return (Gamma1_eV, Gamma2_eV)


def handle_energy_gap_eV(
    energy_gap_eV: float | tuple[float, float] = 0.0,
    temperature_K: float = 0.04,
    thermal_energy_gap: bool = True,
) -> tuple[float, float, float]:
    """Ensure energy gaps are non-negative and above a threshold."""
    if isinstance(energy_gap_eV, (int, float)):
        if energy_gap_eV < 0:
            raise ValueError("Energy gap (eV) must be non-negative.")
        max_Delta_eV = energy_gap_eV
        if thermal_energy_gap:
            energy_gap_eV = relative_energy_gap(
                Delta_eV=energy_gap_eV, T_K=temperature_K
            )
        Delta1_eV = energy_gap_eV
        Delta2_eV = energy_gap_eV
    elif isinstance(energy_gap_eV, tuple) and len(energy_gap_eV) == 2:
        Delta1_eV, Delta2_eV = energy_gap_eV
        if Delta1_eV < 0 or Delta2_eV < 0:
            raise ValueError("Energy gaps (eV) must be non-negative.")

        max_Delta_eV = max(Delta1_eV, Delta2_eV)
        if thermal_energy_gap:
            Delta1_eV = relative_energy_gap(Delta_eV=Delta1_eV, T_K=temperature_K)
            Delta2_eV = relative_energy_gap(Delta_eV=Delta2_eV, T_K=temperature_K)
    else:
        raise ValueError("energy_gap_ev must be a float or a tuple of two floats.")

    return (Delta1_eV, Delta2_eV, max_Delta_eV)


def relative_energy_gap(Delta_eV: float, T_K: float) -> float:
    """Calculates the energy gap in eV at a given temperature."""

    T_C_K = Delta_eV / (1.76 * k_B_eV)  # Critical temperature in Kelvin
    if T_K < 0:
        raise ValueError("Temperature (K) must be non-negative.")
    if T_K >= T_C_K:
        # warnings.warn(f"Estimated T_C: {T_C_K:.2f} K", category=UserWarning)
        return 0.0
    elif T_K == 0:
        return Delta_eV
    else:
        # BCS theory: Delta(T) = Delta(0) * tanh(1.74 * sqrt(Tc/T - 1))
        return Delta_eV * np.tanh(1.74 * np.sqrt(T_C_K / T_K - 1))


def fermi_distribution(E_eV: np.ndarray, T_K: float) -> np.ndarray:
    """Fermi-Dirac distribution at zero and finite temperature."""
    if T_K < 0:
        raise ValueError("Temperature (K) must be non-negative.")
    elif T_K == 0:
        f = np.where(E_eV < 0, 1.0, 0.0)
    else:
        exponent = E_eV / (k_B_eV * T_K)
        exponent = np.clip(exponent, -100, 100)
        f = 1 / (np.exp(exponent) + 1)
    return f


def density_of_states(E_eV: np.ndarray, Delta_eV: float, Gamma_eV: float) -> np.ndarray:
    """Computes the density of states for a superconductor using the Dynes model."""
    if Delta_eV < 0:
        raise ValueError("Energy gap (eV) must be non-negative.")
    if Gamma_eV < 0:
        raise ValueError("Dynes parameter (eV) must be non-negative.")

    E_complex = np.asarray(E_eV, dtype="complex128") + 1j * Gamma_eV

    Delta2 = Delta_eV * Delta_eV
    E_complex_2 = np.multiply(E_complex, E_complex)
    denom = np.sqrt(E_complex_2 - Delta2)
    N_E = np.divide(E_complex, denom)
    N_E = np.real(N_E)

    N_E = np.abs(N_E, dtype="float64")
    N_E[np.isnan(N_E)] = 0.0
    N_E = np.clip(N_E, 0, 100.0)

    return N_E


def get_current_dynes(
    voltage_mV: np.ndarray,
    energy_gap_eV: int | float | tuple[float, float] = 0,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_eV: float | tuple[float, float] = 0.0,
    thermal_energy_gap: bool = True,
    min_dynes_paramter_eV: float = 1e-7,
) -> np.ndarray:
    """Computes the tunneling current for a symmetric superconductor–superconductor (S–S) junction
    using the Dynes density of states, over an array of voltages voltage_mV.

    Parameters
    ----------
    voltage_mV : np.ndarray
        Array of bias voltages (in mV).
    energy_gap_V : float
        Superconducting gap (in V).
    conductance : float
        Conductance in units of G_0.
    temperature_K : float
        Temperature (in Kelvin).
    dynes_parameter_V : float
        Dynes parameter (in V).

    Returns
    -------
    np.ndarray
        Tunneling current (in nA) for each bias voltage.
    """

    # convert voltage in mV
    voltage_V = voltage_mV * 1e-3

    # Calculate Current, assuming Ohmic behavior
    ohms_current_nA = voltage_V * G_0 * transmission * 1e9

    # take care of type and asymetric case
    Delta1_eV, Delta2_eV, max_Delta_eV = handle_energy_gap_eV(
        energy_gap_eV=energy_gap_eV,
        temperature_K=temperature_K,
        thermal_energy_gap=thermal_energy_gap,
    )

    # if Gap is 0, assuming Ohmic behavior
    if max_Delta_eV == 0.0:
        return ohms_current_nA

    # take care of type and asymetric case
    Gamma1_eV, Gamma2_eV = handle_dynes_parameter_eV(
        dynes_parameter_eV=dynes_parameter_eV,
        min_dynes_paramter_eV=min_dynes_paramter_eV,
    )

    # Determine stepsize in V and E
    dV_eV = np.abs(np.nanmax(voltage_V) - np.nanmin(voltage_V)) / (len(voltage_V) - 1)
    dE_eV = min(dV_eV, min_dynes_paramter_eV)  # Ensure dV_eV is not too small

    # Determine max values of V and E
    V_max_eV = np.max(np.abs(voltage_V))
    E_max_eV = min(max_Delta_eV * 10, V_max_eV * 3)

    # create V and E axis
    voltage_eV = np.arange(0, V_max_eV + dV_eV, dV_eV)
    energy_eV = np.arange(-E_max_eV, E_max_eV + dE_eV, dE_eV)
    print("voltage", voltage_eV.shape, voltage_eV)
    print("energy", energy_eV.shape, energy_eV)

    # use float32 to safe on memory
    energy_eV = energy_eV.astype(np.float32)
    voltage_eV = voltage_eV.astype(np.float32)

    # create meshes
    energy_eV_mesh, voltage_eV_mesh = np.meshgrid(energy_eV, voltage_eV / 2)
    energy1_eV_mesh = energy_eV_mesh - voltage_eV_mesh
    energy2_eV_mesh = energy_eV_mesh + voltage_eV_mesh

    # calculate density of states (DOS)
    n1 = density_of_states(
        E_eV=energy_eV,
        Delta_eV=Delta1_eV,
        Gamma_eV=Gamma1_eV,
    )
    n2 = density_of_states(
        E_eV=energy_eV,
        Delta_eV=Delta2_eV,
        Gamma_eV=Gamma2_eV,
    )

    # Interpolate the shifted DOS
    N1 = np.interp(energy1_eV_mesh, energy_eV, n1, left=1.0, right=1.0)
    N2 = np.interp(energy2_eV_mesh, energy_eV, n2, left=1.0, right=1.0)

    # Calculate the Fermi-Dirac Distribution
    f = fermi_distribution(E_eV=energy_eV, T_K=temperature_K)
    f1 = np.interp(energy1_eV_mesh, energy_eV, f, left=1.0, right=0.0)
    f2 = np.interp(energy2_eV_mesh, energy_eV, f, left=1.0, right=0.0)

    # Calculate and clean up the integrand
    integrand = N1 * N2 * (f1 - f2)
    integrand[np.isnan(integrand)] = 0.0

    # Do integration and normalization
    current_dynes = np.trapezoid(integrand, energy_eV, axis=1)
    current_dynes_nA = np.array(current_dynes, dtype=float) * G_0 * transmission * 1e9

    # add negative voltage values
    current_dynes_nA = np.concatenate(
        (current_dynes_nA, -np.flip(current_dynes_nA[1:]))
    )
    voltage_eV = np.concatenate((voltage_eV, -np.flip(voltage_eV[1:])))

    # bin over originally obtained V-axis
    current_dynes_nA = bin_y_over_x(voltage_eV, current_dynes_nA, voltage_V)

    # Fill up values, that are not calculated with dynes
    current_dynes_nA = np.where(
        np.abs(voltage_V) >= 10 * max_Delta_eV,
        ohms_current_nA,
        current_dynes_nA,
    )

    return current_dynes_nA
