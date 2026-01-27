import os, io
import subprocess
import numpy as np
from tqdm import tqdm
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


def run_fcs(
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

    nmax: int = 10
    iw: int = 2003
    nchi: int = 66

    workdir = "/Users/oliver/Documents/p5control-bluefors-evaluation/theory/carlosfcs/"
    fcs_exe = os.path.join(workdir, "fcs")
    fcs_in = os.path.join(workdir, f"fcs.in")
    tmp_input_dir = os.path.join(workdir, ".tmp")
    os.makedirs(tmp_input_dir, exist_ok=True)

    tmp_in = os.path.join(
        tmp_input_dir,
        f"[{V_param_V[0]:.3e}, {V_param_V[1]:.3e}, {V_param_V[2]:.3e}].in",
    )

    if dynes_parameter_V <= 0:
        dynes_parameter_V = 1e-7

    with open(fcs_in, "r") as f:
        lines = f.readlines()

    lines[0] = f"{transmission:.5f} (transmission)\n"
    lines[1] = f"{temperature_K:.5f} (temp in K)\n"
    lines[2] = f"{energy_gap_V*1e3:.6f} {energy_gap_V*1e3:.6f} (gap1,gap2 in meV)\n"
    lines[3] = (
        f"{dynes_parameter_V*1e3:.6f} {dynes_parameter_V*1e3:.6f} (eta1,eta2 = broadening in meV)\n"
    )
    lines[4] = (
        f"{V_param_V[0]*1e3:.8f} {V_param_V[1]*1e3:.8f} {V_param_V[2]*1e3:.8f} (vi,vf,vstep in mV)\n"
    )
    lines[5] = f"{nmax} {iw} {nchi} (nmax,iw,nchi)\n"

    with open(tmp_in, "w") as f:
        f.writelines(lines)

    try:
        proc = subprocess.run(
            [fcs_exe],
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
    # currents = data[:, 1:] * 1e-9  # Convert currents to A

    return data


def get_current_fcs(
    voltage_V: np.ndarray,
    temperature_K: float = 0.0,
    energy_gap_V: float = 2e-4,
    dynes_parameter_V: float = 0.0,
    transmission: float = 0.5,
    n_worker: int = 8,
    max_chunksize=5,
) -> np.ndarray:
    """
    Get the current for a given set of physical parameters using the FCS solver.

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
    currents : np.ndarray
        Currents in A (as returned by solver).
    """

    nmax: int = 10

    chunksize = min(
        [max_chunksize, np.ceil(voltage_V[voltage_V >= 0].shape[0] / n_worker)]
    )

    stepsize_V = np.abs(np.nanmax(voltage_V) - np.nanmin(voltage_V)) / (
        len(voltage_V) - 1
    )
    final_value_V = np.nanmax(np.abs(voltage_V))

    num = int(np.ceil((final_value_V / stepsize_V + 1) / chunksize))

    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        futures = []
        for i in range(num):
            V_param_V = [
                i * stepsize_V * chunksize,
                min(
                    [
                        (i + 1) * chunksize * stepsize_V - stepsize_V,
                        final_value_V + stepsize_V,
                    ]
                ),
                stepsize_V,
            ]
            futures.append(
                executor.submit(
                    run_fcs,
                    V_param_V=V_param_V,
                    temperature_K=temperature_K,
                    energy_gap_V=energy_gap_V,
                    dynes_parameter_V=dynes_parameter_V,
                    transmission=transmission,
                )
            )

        temp_voltage = np.full((1), np.nan, dtype="float64")
        temp_currents = np.full((1, nmax + 1), np.nan, dtype="float64")

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="IV simulations", unit="sim"
        ):
            result = future.result()
            v = result[:, 0] * 1e-3  # Convert voltage to V
            i = result[:, 1:] * 1e-9  # Convert currents to A

            temp_voltage = np.concatenate((temp_voltage, v, -v), axis=0)
            temp_currents = np.concatenate((temp_currents, i, -i), axis=0)

        # remove NaN values from temp_voltages and temp_currents
        valid_indices = ~np.isnan(temp_voltage)
        temp_voltages = temp_voltage[valid_indices]
        temp_currents = temp_currents[valid_indices, :]

        # Initialize the output array for currents
        currents = np.full((voltage_V.shape[0], nmax + 1), np.nan)

        for i in range(temp_currents.shape[1]):
            currents[:, i] = bin_y_over_x(
                x=temp_voltages,
                y=temp_currents[:, i],
                x_bins=voltage_V,
            )[0]

    return currents


# voltage = np.linspace(-0.1e-3, 0.11e-3, 430)
# currents = get_current_FCS2(
#     voltage_V=voltage,
#     temperature_K=0.5,
#     energy_gap_V=100e-6,
#     dynes_parameter_V=50e-6,
#     transmission=0.8,
# )
# print(currents)
# import matplotlib.pyplot as plt

# for i in range(currents.shape[1]):
#     plt.plot(voltage * 1e6, currents[:, i] * 1e9, label=f"$m = {i}$")
# plt.xlabel("Voltage (µV)")
# plt.ylabel("Current (nA)")
# plt.title("FCS IV Curve")
# plt.legend()
# plt.grid()
# plt.show()
