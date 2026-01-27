import os, io
import subprocess
import numpy as np


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


def get_current_HA(
    voltage_V: np.ndarray,
    temperature_K: float = 0.0,
    energy_gap_V: float = 2e-4,
    dynes_parameter_V: float = 0.0,
    transmission: float = 0.5,
) -> np.ndarray:
    """
    Run the Fortran I-V solver for a given set of physical parameters.

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
    voltage : np.ndarray
        Voltages in V (as returned by solver).
    current : np.ndarray
        Currents in A (as returned by solver).
    """

    workdir = "/Users/oliver/Documents/p5control-bluefors-evaluation/theory/CarlosHA/"
    iv_exe = os.path.join(workdir, "iv")
    iv_in = os.path.join(workdir, "iv.in")
    tmp_in = os.path.join(workdir, ".tmp_iv.in")

    # vstep_mV = (
    #     np.abs(np.nanmax(voltage_V) - np.nanmin(voltage_V)) / (len(voltage_V) - 1) * 1e3
    # )
    # vi_mV = np.nanmin(voltage_V) * 1e3 - vstep_mV
    # vf_mV = np.nanmax(voltage_V) * 1e3 - vstep_mV

    vstep_mV = (
        np.abs(np.nanmax(voltage_V) - np.nanmin(voltage_V)) / (len(voltage_V) - 1) * 1e3
    )
    vi_mV = 0
    vf_mV = np.nanmax(np.abs(voltage_V)) * 1e3

    with open(iv_in, "r") as f:
        lines = f.readlines()

    if dynes_parameter_V <= 0:
        dynes_parameter_V = 1e-8

    lines[0] = f" {temperature_K:.5f} (temp in K)\n"
    lines[1] = f" {energy_gap_V*1e3:.6f} {energy_gap_V*1e3:.6f} (gap1,gap2 in meV)\n"
    lines[2] = f" {transmission:.5f} (transmission)\n"
    lines[3] = (
        f" {dynes_parameter_V*1e3:.6f} {dynes_parameter_V*1e3:.6f} (eta1,eta2 = broadening in meV)\n"
    )
    lines[4] = f" {vi_mV:.8f}  {vf_mV:.8f}  {vstep_mV:.8f} (vi,vf,vstep in mV)\n"

    with open(tmp_in, "w") as f:
        f.writelines(lines)

    try:
        proc = subprocess.run(
            [iv_exe],
            stdin=open(tmp_in, "r"),
            capture_output=True,
            text=True,
            cwd=workdir,
        )
    finally:
        if os.path.isfile(tmp_in):
            os.remove(tmp_in)

    if proc.returncode != 0:
        print(f"Error running Fortran code:")
        raise RuntimeError(proc.stderr)

    data = np.genfromtxt(io.StringIO(proc.stdout), dtype="float64")
    if data.size == 0:
        raise RuntimeError(
            "Fortran code produced no output. Check input sweep range and step."
        )

    temp_voltage = np.concatenate((data[:, 0], -data[:, 0])) * 1e-3
    temp_current = np.concatenate((data[:, 1], -data[:, 1])) * 1e-9

    current = bin_y_over_x(
        x=temp_voltage,
        y=temp_current,
        x_bins=voltage_V,
    )[0]

    return current


# voltage = np.linspace(-1.2e-3, 1.2e-3, 241)
# current = get_current_HA(
#     voltage_V=voltage,
#     temperature_K=0.1,
#     energy_gap_V=189e-6,
#     dynes_parameter_V=0,
#     transmission=0.203,
# )
# import matplotlib.pyplot as plt

# plt.plot(voltage, current)
# plt.show()
