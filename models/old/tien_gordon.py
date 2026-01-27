import numpy as np
from numpy.typing import NDArray
from scipy.special import jv
from scipy.interpolate import interp1d
from scipy.constants import e, h
from tqdm import tqdm


def get_tien_gordon_pat(
    voltage_nu_V: NDArray[np.float64],
    voltage_bias_V: NDArray[np.float64],
    current_0: NDArray[np.float64],
    energy_gap_eV: float,
    nu: float,
    N: int = 1000,
) -> NDArray[np.float64]:

    energy_nu_J = e * voltage_nu_V
    energy_bias_J = e * voltage_bias_V
    energy_gap_J = e * energy_gap_eV
    photon_energy_J = h * nu

    energy_bias = energy_bias_J / energy_gap_J
    photon_energy = photon_energy_J / energy_gap_J

    # Tien Gordon
    current_tg_pat = np.full(
        (energy_nu_J.shape[0], energy_bias_J.shape[0]), 0, dtype="float64"
    )

    interpolated_I = interp1d(
        energy_bias,
        current_0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # sum over photons
    for n in range(-N, N + 1):
        # Bessel function squared (n)
        J_n_2 = jv(n, energy_nu_J / photon_energy_J) ** 2

        # Shift I_0 by n
        I_n = interpolated_I(energy_bias - n * photon_energy)

        # Calculate the n'th current
        I_n, J_n_2 = np.meshgrid(I_n, J_n_2)

        # Sum up current
        current_tg_pat += J_n_2 * I_n

    return current_tg_pat


def get_tien_gordon_pamar(
    voltage_nu_V: NDArray[np.float64],
    voltage_bias_V: NDArray[np.float64],
    current_0: NDArray[np.float64],
    energy_gap_eV: float,
    nu: float,
    N: int = 1000,
    M: int = 10,
) -> NDArray[np.float64]:

    energy_nu_J = e * voltage_nu_V
    energy_bias_J = e * voltage_bias_V
    energy_gap_J = e * energy_gap_eV
    photon_energy_J = h * nu

    energy_bias = energy_bias_J / energy_gap_J
    photon_energy = photon_energy_J / energy_gap_J

    # Tien Gordon
    current_tg_pamar = np.full(
        (energy_nu_J.shape[0], energy_bias_J.shape[0]), 0, dtype="float64"
    )

    total_steps = (M - 1) * (2 * N + 1)
    with tqdm(total=total_steps, desc="PAMAR total") as pbar:
        # sum over charges
        for m in range(1, M):
            interpolated_I = interp1d(
                energy_bias,
                current_0[:, m],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            # sum over photons
            for n in range(-N, N + 1):
                # Bessel function squared (n)
                J_n_2 = jv(n, m * energy_nu_J / photon_energy_J) ** 2
                # Shift I_0 by n
                I_n = interpolated_I(energy_bias - n / m * photon_energy)

                # Calculate the n'th current
                I_n, J_n_2 = np.meshgrid(I_n, J_n_2)

                # Sum up current
                current_tg_pamar += J_n_2 * I_n
                pbar.update(1)

    return current_tg_pamar
