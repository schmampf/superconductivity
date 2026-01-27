import numpy as np

from theory.utilities.types import NDArray64

from theory.utilities.functions import bin_y_over_x

from theory.utilities.constants import G_0_muS

from theory.models.bcs import Delta_meV_of_T, f_of_E


def Z_of_tau(tau: float) -> float:
    return np.sqrt(1.0 / tau - 1.0)


def AB_of_E(
    E_meV: NDArray64,
    Delta_meV: float,
    Z: float,
    gamma_meV: float,
) -> tuple[
    NDArray64,
    NDArray64,
]:
    E_meV = np.array(np.abs(E_meV) + 1j * gamma_meV, dtype="complex128")
    E_meV += 1e-300  # avoid runtime warning
    u2 = 0.5 * (1 + np.sqrt(E_meV**2 - Delta_meV**2) / E_meV)
    v2 = 1 - u2
    Z2 = np.square(Z)

    alpha = np.real(u2)
    beta = np.real(v2)
    etta = np.abs(np.imag(u2))
    alpha2 = np.square(alpha)
    beta2 = np.square(beta)
    etta2 = np.square(etta)
    diff = alpha - beta

    gamma2 = np.square(alpha + Z2 * diff) + np.square(etta * (2.0 * Z2 + 1.0))
    gamma2 += 1e-300  # avoid runtime warning
    A = np.sqrt(np.abs((alpha2 + etta2) * (beta2 + etta2))) / gamma2
    term1 = np.square(diff * Z - 2.0 * etta)
    term2 = np.square(2.0 * etta * Z + diff)
    B = (Z2 * (term1 + term2)) / gamma2
    return (
        np.clip(A, 0.0, 1.0).astype(np.float64),
        np.clip(B, 0.0, 1.0).astype(np.float64),
    )


def get_I_nA(
    V_mV: NDArray64,
    Delta_meV: float = 0.18,
    tau: float = 0.5,
    T_K: float = 0.0,
    gamma_meV: float = 0.0,
    gamma_meV_min: float = 1e-4,
) -> NDArray64:

    G_N_muS = tau * G_0_muS
    I_NN_nA = V_mV * G_N_muS

    Delta_meV_T = Delta_meV_of_T(Delta_meV=Delta_meV, T_K=T_K)

    if Delta_meV_T == 0.0:
        return np.vstack((I_NN_nA, I_NN_nA, np.zeros_like(I_NN_nA)))

    gamma_meV = gamma_meV_min if gamma_meV < gamma_meV_min else gamma_meV

    # Determine stepsize in V and E
    dV_mV = np.abs(np.nanmax(V_mV) - np.nanmin(V_mV)) / (len(V_mV) - 1)
    V_max_mV = np.max(np.abs(V_mV))

    E_max_meV = np.max([Delta_meV_T * 10, V_max_mV])
    dE_meV = np.min([dV_mV, gamma_meV_min])

    # create V and E axis
    V_mV_temp = np.arange(0.0, V_max_mV + dV_mV, dV_mV, dtype="float64")
    E_meV = np.arange(-E_max_meV, E_max_meV + dE_meV, dE_meV, dtype="float64")

    f1 = f_of_E(E_meV=E_meV, T_K=T_K)
    Z = Z_of_tau(tau)
    A, B = AB_of_E(
        E_meV=E_meV,
        Delta_meV=Delta_meV_T,
        Z=Z,
        gamma_meV=gamma_meV,
    )

    I_2e_mV = np.empty_like(V_mV_temp)
    I_1e_mV = np.empty_like(V_mV_temp)
    for i, meV in enumerate(V_mV_temp):
        f2 = f_of_E(E_meV=E_meV - meV, T_K=T_K)
        df = f2 - f1
        I_2e_mV[i] = np.trapezoid((2 * A) * df, E_meV)
        I_1e_mV[i] = np.trapezoid((1 - B - A) * df, E_meV)
    I_2e_nA = np.array(I_2e_mV, dtype="float64") * G_0_muS
    I_1e_nA = np.array(I_1e_mV, dtype="float64") * G_0_muS

    # add negative voltage values
    I_2e_nA = np.concatenate((I_2e_nA, -np.flip(I_2e_nA[1:])))
    I_1e_nA = np.concatenate((I_1e_nA, -np.flip(I_1e_nA[1:])))
    V_mV_temp = np.concatenate((V_mV_temp, -np.flip(V_mV_temp[1:])))

    # bin over originally obtained V-axis
    I_2e_nA = bin_y_over_x(V_mV_temp, I_2e_nA, V_mV)
    I_1e_nA = bin_y_over_x(V_mV_temp, I_1e_nA, V_mV)

    I_tot_nA = I_2e_nA + I_1e_nA
    I_nA = np.vstack((I_tot_nA, I_1e_nA, I_2e_nA)).T

    return I_nA
