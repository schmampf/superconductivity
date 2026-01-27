import os
import io
import hashlib
import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

WORK_DIR = (
    "/Users/oliver/Documents/p5control-bluefors-evaluation/theory/models/carlosha/"
)
CACHE_DIR = os.path.join(WORK_DIR, ".cache")
TMP_DIR = os.path.join(WORK_DIR, ".tmp")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

HA_EXE = os.path.join(WORK_DIR, "ha")
HA_IN = os.path.join(WORK_DIR, "ha.in")

VOLTAGE_PRECISION: int = 6
GAP_PRECISION: int = 6
TAU_PRECISION: int = 4
TEMP_PRECISION: int = 4
DYNES_PRECISION: int = 6


def my_hash(
    energy_gap_eV: float = 2e-4,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_eV: float = 0.0,
) -> str:
    """Generates a hash for the given parameters to uniquely identify a simulation."""
    string = "cache_ha_"
    string += f"Delta={energy_gap_eV:.{GAP_PRECISION}f}eV_"
    string += f"tau={transmission:.{TAU_PRECISION}f}_"
    string += f"T={temperature_K:.{TEMP_PRECISION}f}K_"
    string += f"Gamma={dynes_parameter_eV:.{DYNES_PRECISION}f}eV"
    print(len(string))
    return string


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


def run_ha(
    voltage_mV: float,
    energy_gap_V: float = 2e-4,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_V: float = 0.0,
) -> np.ndarray:

    key = hash_params(voltage_mV)
    tmp_in = os.path.join(
        TMP_DIR,
        f"{key}.in",
    )

    if dynes_parameter_V <= 1e-9:
        dynes_parameter_V = 1e-9

    if temperature_K <= 1e-7:
        temperature_K = 1e-7

    dv = 1e-5  # mV

    with open(HA_IN, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines[0] = f"{temperature_K:.12f}\n"  # K
    lines[1] = f"{energy_gap_V*1e3:.12f} {energy_gap_V*1e3:.12f}\n"  # mV
    lines[2] = f"{transmission:.12f}\n"  # [0, 1]
    lines[3] = f"{dynes_parameter_V*1e3:.12f} {dynes_parameter_V*1e3:.12f}\n"  # mV
    lines[4] = f"{voltage_mV-dv:.6f} {voltage_mV-dv:.6f} {dv:.6f} \n"  # mV

    with open(tmp_in, "w", encoding="utf-8") as f:
        f.writelines(lines)

    try:
        proc = subprocess.run(
            [HA_EXE],
            stdin=open(tmp_in, "r", encoding="utf-8"),
            capture_output=True,
            text=True,
            cwd=WORK_DIR,
            check=True,
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


def run_multiple_ha(
    voltage_mV: list,
    energy_gap_V: float = 2e-4,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_V: float = 0.0,
    n_worker: int = 16,
) -> tuple[np.ndarray, np.ndarray]:

    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        futures = []
        for v in voltage_mV:
            futures.append(
                executor.submit(
                    run_ha,
                    voltage_mV=v,
                    energy_gap_V=energy_gap_V,
                    transmission=transmission,
                    temperature_K=temperature_K,
                    dynes_parameter_V=dynes_parameter_V,
                )
            )

        new_voltages_mV = []
        new_currents_nA = []

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="IV simulations",
            unit="sim",
        ):
            result = future.result()
            v = result[0] - 1e-5  # dv in input file
            i = result[1]

            new_voltages_mV.append(v)
            new_currents_nA.append(i)

        new_voltages_mV = np.array(new_voltages_mV)
        new_currents_nA = np.array(new_currents_nA)

    return new_voltages_mV, new_currents_nA


def get_current_ha(
    voltage_mV: np.ndarray,
    energy_gap_eV: float = 2e-4,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_eV: float = 0.0,
    n_worker: int = 16,
) -> np.ndarray:

    energy_gap_V = np.round(energy_gap_V, decimals=GAP_PRECISION)
    transmission = np.round(transmission, decimals=TAU_PRECISION)
    temperature_K = np.round(temperature_K, decimals=TEMP_PRECISION)
    dynes_parameter_eV = np.round(dynes_parameter_eV, decimals=DYNES_PRECISION)

    # --- Create a cache key based on physical parameters except voltage ---
    # This key uniquely identifies the simulation parameters to enable caching
    key = my_hash(
        energy_gap_eV=energy_gap_eV,
        transmission=transmission,
        temperature_K=temperature_K,
        dynes_parameter_eV=dynes_parameter_eV,
    )
    cached_file = os.path.join(CACHE_DIR, f"{key}.npz")

    # --- Prepare the voltage sweep range ---
    # Calculate step size based on input voltage array to create a consistent voltage grid
    stepsize_mV = np.abs(np.nanmax(voltage_mV) - np.nanmin(voltage_mV)) / (
        len(voltage_mV) - 1
    )
    final_value_mV = np.nanmax(np.abs(voltage_mV))
    # Generate voltage points from 0 to max absolute voltage with calculated step size
    input_voltage_mV = np.arange(
        0, final_value_mV + stepsize_mV, stepsize_mV, dtype=float
    )

    # Round voltages to avoid floating point precision issues when comparing
    input_voltage_mV = np.round(input_voltage_mV, decimals=VOLTAGE_PRECISION)

    # --- Load existing cached simulation results if available ---
    if os.path.exists(cached_file):
        cache_data = np.load(cached_file)

        cached_voltages_mV = cache_data["V"]
        cached_currents_nA = cache_data["I"]

        # Round cached data for consistent comparison and sorting
        cached_voltages_mV = np.round(
            cached_voltages_mV,
            decimals=VOLTAGE_PRECISION,
        )
        cached_currents_nA = np.round(
            cached_currents_nA,
            decimals=VOLTAGE_PRECISION,
        )

    else:
        # Initialize empty arrays if no cache exists
        cached_voltages_mV = np.empty((0), dtype=float)
        cached_currents_nA = np.empty((0), dtype=float)

    # --- Determine which voltages still need simulation (uncached) ---
    cache_in_input = np.isin(input_voltage_mV, cached_voltages_mV)
    uncached_voltages_mV = input_voltage_mV[np.logical_not(cache_in_input)]

    # Extract cached voltages and currents that are relevant for the input voltage range
    input_in_cache = np.isin(cached_voltages_mV, input_voltage_mV)
    stashed_voltages_mV = cached_voltages_mV[input_in_cache]
    stashed_currents_nA = cached_currents_nA[input_in_cache]

    print(f"cached values: {stashed_voltages_mV.shape[0]}/{input_voltage_mV.shape[0]}")

    # --- Run simulations for uncached voltages ---
    if uncached_voltages_mV.size > 0:
        uncached_voltages_mV, uncached_currents_nA = run_multiple_ha(
            voltage_mV=list(uncached_voltages_mV),
            energy_gap_V=energy_gap_V,
            transmission=transmission,
            temperature_K=temperature_K,
            dynes_parameter_V=dynes_parameter_V,
            n_worker=n_worker,
        )
    else:
        # If no uncached voltages, initialize empty arrays accordingly
        uncached_voltages_mV = np.array([], dtype=float)
        uncached_currents_nA = np.array([], dtype=float)

    # --- Update cache with newly computed results ---
    cached_voltages_mV = np.concatenate((cached_voltages_mV, uncached_voltages_mV))
    cached_currents_nA = np.concatenate((cached_currents_nA, uncached_currents_nA))

    # Sort cached voltages and corresponding currents to maintain order
    sort_idx = np.argsort(cached_voltages_mV)
    cached_voltages_mV = cached_voltages_mV[sort_idx]
    cached_currents_nA = cached_currents_nA[sort_idx]

    # Round cached voltages again to ensure consistency
    cached_voltages_mV = np.round(cached_voltages_mV, decimals=VOLTAGE_PRECISION)
    cached_currents_nA = np.round(cached_currents_nA, decimals=VOLTAGE_PRECISION)

    # Save updated cache to disk for future reuse
    np.savez(
        cached_file,
        V=cached_voltages_mV,
        I=cached_currents_nA,
        Delta=energy_gap_V,
        tau=transmission,
        T=temperature_K,
        Gamma=dynes_parameter_V,
    )

    # --- Prepare the final voltage and current arrays ---
    # The Fortran code only simulates positive voltages; here we generate symmetric data
    # by concatenating positive and negative voltages and their corresponding currents.
    tmp_voltages_mV = np.concatenate(
        (
            stashed_voltages_mV,
            -stashed_voltages_mV,
            uncached_voltages_mV,
            -uncached_voltages_mV,
        )
    )
    tmp_currents_nA = np.concatenate(
        (
            stashed_currents_nA,
            -stashed_currents_nA,
            uncached_currents_nA,
            -uncached_currents_nA,
        )
    )

    # --- Bin the currents over the original input voltages ---
    # Since tmp_voltages_mV may not exactly match input voltages, binning averages currents
    # in bins centered at the input voltages to produce aligned output.
    currents_nA = bin_y_over_x(
        x=tmp_voltages_mV,
        y=tmp_currents_nA,
        x_bins=voltage_mV,
    )[0]

    currents_nA = np.round(currents_nA, decimals=VOLTAGE_PRECISION)

    return currents_nA
