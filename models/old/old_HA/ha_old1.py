import os, io
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def bin_y_over_x(
    x: np.ndarray,
    y: np.ndarray,
    x_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    # Extend bin edges for histogram: shift by half a bin width for center alignment
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])  # Add one final edge
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2

    # Count how many x-values fall into each bin
    _count, _ = np.histogram(x, bins=x_nu, weights=None)
    _count = np.array(_count, dtype="float64")
    _count[_count == 0] = np.nan

    # Sum of y-values in each bin
    _sum, _ = np.histogram(x, bins=x_nu, weights=y)

    # Return mean y per bin and count
    return _sum / _count, _count


def run_ha(
    V_param_V=[0, -1e-6, 10],
    temperature_K: float = 0.0,
    energy_gap_V: float = 2e-4,
    dynes_parameter_V: float = 0.0,
    transmission: float = 0.5,
) -> np.ndarray:
    """
    Run the Fortran I-V solver for a given set of physical parameters.

    Parameters
    ----------
    V_param_V : np.ndarray
        Array of voltages (in V) to sweep.
    temperature_K : float, optional
        Temperature in Kelvin.
    energy_gap_V : float, optional
        Superconducting gap in Volts.
    dynes_parameter_V : float, optional
        Dynes parameter in Volts.
    transmission : float, optional
        Transmission coefficient [0, 1].

    Returns
    -------
    data : np.ndarray
        data (as returned by solver).
    """

    workdir = "/Users/oliver/Documents/p5control-bluefors-evaluation/theory/carlosha/"
    ha_exe = os.path.join(workdir, "ha")
    ha_in = os.path.join(workdir, "ha.in")
    tmp_input_dir = os.path.join(workdir, ".tmp")
    os.makedirs(tmp_input_dir, exist_ok=True)

    tmp_in = os.path.join(
        tmp_input_dir,
        f"[{V_param_V[0]:.3e}, {V_param_V[1]:.3e}, {V_param_V[2]:.3e}].in",
    )

    if dynes_parameter_V <= 0:
        dynes_parameter_V = 1e-8

    with open(ha_in, "r") as f:
        lines = f.readlines()

    lines[0] = f" {temperature_K:.5f} (temp in K)\n"
    lines[1] = f" {energy_gap_V*1e3:.6f} {energy_gap_V*1e3:.6f} (gap1,gap2 in meV)\n"
    lines[2] = f" {transmission:.5f} (transmission)\n"
    lines[3] = (
        f" {dynes_parameter_V*1e3:.6f} {dynes_parameter_V*1e3:.6f} (eta1,eta2 = broadening in meV)\n"
    )
    lines[4] = (
        f"{V_param_V[0]*1e3:.8f} {V_param_V[1]*1e3:.8f} {V_param_V[2]*1e3:.8f} (vi,vf,vstep in mV)\n"
    )

    with open(tmp_in, "w") as f:
        f.writelines(lines)

    try:
        proc = subprocess.run(
            [ha_exe],
            stdin=open(tmp_in, "r"),
            capture_output=True,
            text=True,
            cwd=workdir,
        )
    finally:
        if os.path.isfile(tmp_in):
            os.remove(tmp_in)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    data = np.genfromtxt(io.StringIO(proc.stdout), dtype="float64")
    if data.size == 0:
        raise RuntimeError(
            "Fortran code produced no output. Check input sweep range and step."
        )

    # voltage = data[:, 0] * 1e-3  # Convert voltage to V
    # currents = data[:, 1] * 1e-9  # Convert currents to A

    return data


def get_current_ha(
    voltage_V: np.ndarray,
    temperature_K: float = 0.0,
    energy_gap_V: float = 2e-4,
    dynes_parameter_V: float = 0.0,
    transmission: float = 0.5,
    n_worker: int = 16,
) -> np.ndarray:
    """
    Get the current for a given voltage sweep using the HA solver.

    Parameters
    ----------
    voltage_V : np.ndarray
        Array of voltages (in V) to sweep.
    temperature_K : float, optional
        Temperature in Kelvin.
    energy_gap_V : float, optional
        Superconducting gap in Volts.
    dynes_parameter_V : float, optional
        Dynes parameter in Volts.
    transmission : float, optional
        Transmission coefficient [0, 1].

    Returns
    -------
    current : np.ndarray
        Current values corresponding to the input voltages.
    """

    stepsize_V = np.abs(np.nanmax(voltage_V) - np.nanmin(voltage_V)) / (
        len(voltage_V) - 1
    )
    final_value_V = np.nanmax(np.abs(voltage_V))

    chunksize = int(np.ceil((final_value_V / stepsize_V / n_worker)))

    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        futures = []
        for i in range(n_worker):
            V_param_V = [
                i * stepsize_V * chunksize - stepsize_V,
                (i + 1) * chunksize * stepsize_V,
                stepsize_V,
            ]
            futures.append(
                executor.submit(
                    run_ha,
                    V_param_V=V_param_V,
                    temperature_K=temperature_K,
                    energy_gap_V=energy_gap_V,
                    dynes_parameter_V=dynes_parameter_V,
                    transmission=transmission,
                )
            )

        temp_voltage = np.array([], dtype="float64")
        temp_currents = np.array([], dtype="float64")

        for j, future in enumerate(as_completed(futures)):
            result = future.result()
            v = result[:, 0] * 1e-3  # Convert voltage to V
            i = result[:, 1] * 1e-9  # Convert currents to A

            temp_voltage = np.concatenate((temp_voltage, v, -v), axis=0)
            temp_currents = np.concatenate((temp_currents, i, -i), axis=0)

        current = bin_y_over_x(
            x=temp_voltage,
            y=temp_currents,
            x_bins=voltage_V,
        )[0]

    return current
