import os
import io
import hashlib
import pickle
import subprocess
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

WORK_DIR = "/Users/oliver/Documents/p5control-bluefors-evaluation/theory/carlosfcs/"
CACHE_DIR = os.path.join(WORK_DIR, ".cache")
TMP_DIR = os.path.join(WORK_DIR, ".tmp")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

FCS_EXE = os.path.join(WORK_DIR, "fcs")
FCS_IN = os.path.join(WORK_DIR, f"fcs.in")

MAX_CHARGES: int = 10


def hash_params(*params):
    m = hashlib.sha256()
    for p in params:
        if isinstance(p, np.ndarray):
            m.update(p.tobytes())
        else:
            m.update(str(p).encode())
    return m.hexdigest()


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
    voltage_V: float,
    temperature_K: float = 0.0,
    energy_gap_V: float = 2e-4,
    dynes_parameter_V: float = 0.0,
    transmission: float = 0.5,
) -> np.ndarray:
    """
    Run the Fortran I-V solver for a given set of physical parameters.

    Parameters
    ----------
    voltage_V : float
        voltage (in V)
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

    nmax: int = MAX_CHARGES  # 10
    iw: int = 2003
    nchi: int = 66

    tmp_in = os.path.join(
        TMP_DIR,
        f"{voltage_V:.3e}.in",
    )

    if dynes_parameter_V <= 0:
        dynes_parameter_V = 1e-7

    with open(FCS_IN, "r") as f:
        lines = f.readlines()

    lines[0] = f"{transmission:.5f} (transmission)\n"
    lines[1] = f"{temperature_K:.5f} (temp in K)\n"
    lines[2] = f"{energy_gap_V*1e3:.6f} {energy_gap_V*1e3:.6f} (gap1,gap2 in meV)\n"
    lines[3] = (
        f"{dynes_parameter_V*1e3:.6f} {dynes_parameter_V*1e3:.6f} (eta1,eta2 = broadening in meV)\n"
    )
    lines[4] = f"{voltage_V*1e3:.8f} {voltage_V*1e3:.8f} 1.0 (vi,vf,vstep in mV)\n"
    lines[5] = f"{nmax} {iw} {nchi} (nmax,iw,nchi)\n"

    with open(tmp_in, "w") as f:
        f.writelines(lines)

    try:
        proc = subprocess.run(
            [FCS_EXE],
            stdin=open(tmp_in, "r"),
            capture_output=True,
            text=True,
            cwd=WORK_DIR,
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

    return data


def run_multiple_fcs(
    voltage_V: list,
    energy_gap_V: float = 2e-4,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_V: float = 0.0,
    n_worker: int = 16,
) -> (np.ndarray, np.ndarray):
    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        futures = []
        for v in voltage_V:
            futures.append(
                executor.submit(
                    run_fcs,
                    voltage_V=v,
                    temperature_K=temperature_K,
                    energy_gap_V=energy_gap_V,
                    dynes_parameter_V=dynes_parameter_V,
                    transmission=transmission,
                )
            )

        new_voltages = []
        new_currents = []

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="IV simulations",
            unit="sim",
        ):
            result = future.result()
            v = np.array(result[0]) * 1e-3  # Convert voltage to V
            i = np.array(result[1:]) * 1e-9  # Convert currents to A

            new_voltages.append(v)
            new_currents.append(i)

    return np.array(new_voltages), np.array(new_currents)


def get_current_fcs(
    voltage_V: np.ndarray,
    energy_gap_V: float = 2e-4,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_V: float = 0.0,
    n_worker: int = 16,
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

    max_charges: int = MAX_CHARGES

    # Create a cache key based on physical parameters except voltage
    key = hash_params(transmission, energy_gap_V, temperature_K, dynes_parameter_V)
    cached_file = os.path.join(CACHE_DIR, f"{key}.npz")

    # Calculate voltage values, that are needed in principle
    stepsize_V = np.abs(np.nanmax(voltage_V) - np.nanmin(voltage_V)) / (
        len(voltage_V) - 1
    )
    final_value_V = np.nanmax(np.abs(voltage_V))
    input_voltage_V = np.arange(0, final_value_V + stepsize_V, stepsize_V)

    # Load existing cache if present
    if os.path.exists(cached_file):
        cache_data = np.load(cached_file)

        cached_voltages = cache_data["V"]
        cached_currents = cache_data["I"]
    else:
        cached_voltages = np.array([], dtype=float)
        cached_currents = np.empty((0, max_charges + 1), dtype=float)

    # Identify which voltages are missing in cache
    def is_voltage_cached(v, tolerance: float = 1e-15):
        return np.any(np.isclose(cached_voltages, v, atol=tolerance))

    uncached_voltages = np.array(
        [v for v in input_voltage_V if not is_voltage_cached(v)]
    )

    # stashed voltages and currents (in voltage_V and are cached)
    stashed_voltages = np.array([v for v in cached_voltages if is_voltage_cached(v)])
    stashed_currents = cached_currents[cached_voltages == stashed_voltages]

    # If there are missing voltages, compute them
    if uncached_voltages.size > 0:
        uncached_voltages, uncached_currents = run_multiple_fcs(
            voltage_V=list(uncached_voltages),
            energy_gap_V=energy_gap_V,
            transmission=transmission,
            temperature_K=temperature_K,
            dynes_parameter_V=dynes_parameter_V,
            n_worker=n_worker,
        )
    else:
        uncached_voltages = np.array([], dtype=float)
        uncached_currents = np.empty((0, max_charges + 1), dtype=float)

    # caching voltages
    cached_voltages = np.concatenate((cached_voltages, uncached_voltages))
    cached_currents = np.concatenate((cached_currents, uncached_currents))

    # Sort cached data by voltage (nicht unbedingt nötig, just for OCD)
    sort_idx = np.argsort(cached_voltages)
    cached_voltages = cached_voltages[sort_idx]
    cached_currents = cached_currents[sort_idx, :]

    # Save updated cache
    np.savez(
        cached_file,
        I=cached_currents,
        V=cached_voltages,
        Delta=energy_gap_V,
        tau=transmission,
        T=temperature_K,
        Gamma=dynes_parameter_V,
    )

    # Prepare return data
    tmp_voltages = np.concatenate(
        (stashed_voltages, -stashed_voltages, uncached_voltages, -uncached_voltages)
    )
    tmp_currents = np.concatenate(
        (stashed_currents, -stashed_currents, uncached_currents, -uncached_currents)
    )

    binned_currents = np.full((voltage_V.shape[0], max_charges + 1), np.nan)

    # Bin new data onto voltage_V bins
    # We want to get currents at each missing voltage value (including negative)
    for i_col in range(max_charges + 1):
        binned_currents[:, i_col] = bin_y_over_x(
            x=tmp_voltages,
            y=tmp_currents[:, i_col],
            x_bins=voltage_V,
        )[0]

    return binned_currents
