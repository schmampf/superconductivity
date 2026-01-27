import os
import io
import hashlib
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
    V_param_V=[0, -1e-6, 10],
    energy_gap_V: float = 2e-4,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_V: float = 0.0,
) -> np.ndarray:

    nmax: int = MAX_CHARGES  # 10
    iw: int = 2003
    nchi: int = 66

    tmp_in = os.path.join(
        TMP_DIR,
        f"[{V_param_V[0]:.3e}, {V_param_V[1]:.3e}, {V_param_V[2]:.3e}].in",
    )

    if dynes_parameter_V <= 0:
        dynes_parameter_V = 1e-7

    with open(FCS_IN, "r", encoding="utf-8") as f:
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

    with open(tmp_in, "w", encoding="utf-8") as f:
        f.writelines(lines)

    try:
        proc = subprocess.run(
            [FCS_EXE],
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


def get_current_fcs(
    voltages_V: np.ndarray,
    temperature_K: float = 0.0,
    energy_gap_V: float = 2e-4,
    dynes_parameter_V: float = 0.0,
    transmission: float = 0.5,
    n_worker: int = 16,
) -> np.ndarray:

    max_charges: int = MAX_CHARGES

    stepsize_V = np.abs(np.nanmax(voltages_V) - np.nanmin(voltages_V)) / (
        len(voltages_V) - 1
    )
    final_value_V = np.nanmax(np.abs(voltages_V))

    tmp_voltages = np.arange(0, final_value_V, stepsize_V)

    # Create a cache key based on physical parameters except voltage
    key = hash_params(
        tmp_voltages, transmission, energy_gap_V, temperature_K, dynes_parameter_V
    )
    cached_file = os.path.join(CACHE_DIR, f"{key}.npz")

    # Load existing cache if present
    if os.path.exists(cached_file):
        cache_data = np.load(cached_file)
        tmp_currents = cache_data["I"]
    else:
        v_num = tmp_voltages.shape[0]
        v_num_per_worker = int(np.ceil(v_num / n_worker))
        if v_num_per_worker == 1:
            raise KeyError(
                "Warning: increase resolution of voltages_V or decrease n_worker."
            )

        with ThreadPoolExecutor(max_workers=n_worker) as executor:
            futures = []
            for n in range(n_worker):
                V_param_V = [
                    n * (v_num_per_worker * stepsize_V),
                    min(
                        [
                            (n + 1) * (v_num_per_worker * stepsize_V),
                            final_value_V + 2 * stepsize_V,
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

        simulated_voltages = np.array([], dtype=float)
        simulated_currents = np.empty((0, max_charges + 1), dtype=float)

        for future in as_completed(futures):
            result = future.result()
            v = result[:, 0] * 1e-3  # Convert voltage to V
            print(v)
            i = result[:, 1:] * 1e-9  # Convert currents to A

            simulated_voltages = np.concatenate((simulated_voltages, v), axis=0)
            simulated_currents = np.concatenate((simulated_currents, i), axis=0)

        tmp_currents = np.full((tmp_voltages.shape[0], max_charges + 1), np.nan)

        for i in range(max_charges + 1):
            tmp_currents[:, i] = bin_y_over_x(
                x=simulated_voltages,
                y=simulated_currents[:, i],
                x_bins=tmp_voltages,
            )[0]

        np.savez(
            cached_file,
            I=tmp_currents,
            V=tmp_voltages,
            Delta=energy_gap_V,
            tau=transmission,
            T=temperature_K,
            Gamma=dynes_parameter_V,
        )

    tmp_voltages = np.concatenate((tmp_voltages, -tmp_voltages))
    tmp_currents = np.concatenate((tmp_currents, -tmp_currents))

    currents = np.full((voltages_V.shape[0], max_charges + 1), np.nan)
    for i in range(max_charges + 1):
        currents[:, i] = bin_y_over_x(
            x=tmp_voltages,
            y=tmp_currents[:, i],
            x_bins=voltages_V,
        )[0]

    return currents
