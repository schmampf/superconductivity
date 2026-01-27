import os, io
import subprocess
import numpy as np
import time


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


def get_current_FCS(
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

    nmax: int = 10
    iw: int = 2003
    nchi: int = 66

    workdir = "/Users/oliver/Documents/p5control-bluefors-evaluation/theory/CarlosFCS/"
    iv_exe = os.path.join(workdir, "fcs")
    iv_in = os.path.join(workdir, "fcs.in")
    tmp_in = os.path.join(workdir, ".tmp_fcs.in")

    vstep_mV = (
        np.abs(np.nanmax(voltage_V) - np.nanmin(voltage_V)) / (len(voltage_V) - 1) * 1e3
    )
    vi_mV = 0
    vf_mV = np.nanmax(np.abs(voltage_V)) * 1e3

    with open(iv_in, "r") as f:
        lines = f.readlines()

    if dynes_parameter_V <= 0:
        dynes_parameter_V = 1e-7

    lines[0] = f" {transmission:.5f} (transmission)\n"
    lines[1] = f" {temperature_K:.5f} (temp in K)\n"
    lines[2] = f" {energy_gap_V*1e3:.6f} {energy_gap_V*1e3:.6f} (gap1,gap2 in meV)\n"
    lines[3] = (
        f" {dynes_parameter_V*1e3:.6f} {dynes_parameter_V*1e3:.6f} (eta1,eta2 = broadening in meV)\n"
    )
    lines[4] = f" {vi_mV:.8f}  {vf_mV:.8f}  {vstep_mV:.8f} (vi,vf,vstep in mV)\n"
    lines[4] = f" {vi_mV:.8f}  {vf_mV:.8f}  {vstep_mV:.8f} (vi,vf,vstep in mV)\n"
    lines[5] = f"{nmax}  {iw}  {nchi}  (nmax,iw,nchi)\n"

    with open(tmp_in, "w") as f:
        f.writelines(lines)

    try:
        toc = time.time()
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
        tic = time.time()
        print(f"Execution time: {tic - toc:.2f} seconds")

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    data = np.genfromtxt(io.StringIO(proc.stdout), dtype="float64")
    if data.size == 0:
        raise RuntimeError(
            "Fortran code produced no output. Check input sweep range and step."
        )

    temp_voltage = np.concatenate((data[:, 0], -data[:, 0])) * 1e-3
    temp_currents = np.concatenate((data[:, 1:], -data[:, 1:])) * 1e-9

    currents = np.full((voltage_V.shape[0], nmax + 1), np.nan)

    for i in range(nmax + 1):
        currents[:, i] = bin_y_over_x(
            x=temp_voltage,
            y=temp_currents[:, i],
            x_bins=voltage_V,
        )[0]

    return currents


# voltage = np.linspace(-0.1e-3, 0.3e-3, 101)
# print(voltage)
# currents = get_current_FCS(
#     voltage_V=voltage,
#     temperature_K=0.5,
#     energy_gap_V=100e-6,
#     dynes_parameter_V=50e-6,
#     transmission=0.8,
# )

# import matplotlib.pyplot as plt

# for i in range(currents.shape[1]):
#     plt.plot(voltage * 1e6, currents[:, i] * 1e9, label=f"$m = {i}$")
# plt.xlabel("Voltage (µV)")
# plt.ylabel("Current (nA)")
# plt.title("FCS IV Curve")
# plt.legend()
# plt.grid()
# plt.show()
