import io
import numpy as np
import os
import subprocess
import sys

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

HOME_DIR = "/Users/oliver/Documents/p5control-bluefors-evaluation"
sys.path.append(HOME_DIR)

from ..utilities.types import NDArray64

from ..utilities.functions import cache_hash_pbar
from ..utilities.functions import bin_y_over_x

from ..utilities.constants import V_tol_mV
from ..utilities.constants import tau_tol
from ..utilities.constants import T_tol_K
from ..utilities.constants import Delta_tol_meV
from ..utilities.constants import gamma_tol_meV

# number of maximum charges
from ..utilities.constants import m_max
from ..utilities.constants import iw
from ..utilities.constants import nchi


WORK_DIR = os.path.join(HOME_DIR, "theory/models/carlosfcs/")
CACHE_DIR = os.path.join(WORK_DIR, ".cache_pbar")
FCS_EXE = os.path.join(WORK_DIR, "fcs")
os.makedirs(CACHE_DIR, exist_ok=True)


def run_fcs(
    V_mV: float,
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
) -> NDArray64:

    string = ""
    string += f"{tau:.{tau_tol}f}\n"  # [0, 1]
    string += f"{T_K:.{T_tol_K}f}\n"  # K
    string += f"{Delta_1_meV:.{Delta_tol_meV}f} {Delta_2_meV:.{Delta_tol_meV}f}\n"  # mV
    string += f"{gamma_1_meV:.{gamma_tol_meV}f} {gamma_2_meV:.{gamma_tol_meV}f}\n"  # mV
    string += f"{V_mV:.{V_tol_mV}f} {V_mV:.{V_tol_mV}f} 1.0 \n"  # mV
    string += f"{m_max} {iw} {nchi}"

    proc = subprocess.run(
        [FCS_EXE],
        input=string,
        capture_output=True,
        text=True,
        cwd=WORK_DIR,
        check=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    data = np.genfromtxt(io.StringIO(proc.stdout), dtype="float64")
    if data.size == 0:
        raise RuntimeError(
            "Fortran code produced no output. Check input sweep range and step."
        )

    return data


def run_multiple_fcs(
    V_mV: list,
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
    n_worker: int = 16,
) -> tuple[NDArray64, NDArray64]:

    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        futures = []
        for V in V_mV:
            futures.append(
                executor.submit(
                    run_fcs,
                    V_mV=V,
                    tau=tau,
                    T_K=T_K,
                    Delta_1_meV=Delta_1_meV,
                    Delta_2_meV=Delta_2_meV,
                    gamma_1_meV=gamma_1_meV,
                    gamma_2_meV=gamma_2_meV,
                )
            )

        all_V_mV = []
        all_I_nA = []

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="IV simulations",
            unit="sim",
        ):
            result = future.result()
            v = np.array(result[0], dtype="float64")
            i = np.array(result[1:], dtype="float64")

            all_V_mV.append(v)
            all_I_nA.append(i)

        all_V_mV = np.array(all_V_mV, dtype="float64")
        all_I_nA = np.array(all_I_nA, dtype="float64")

    return all_V_mV, all_I_nA


def get_I_fcs_pbar_nA(
    V_mV: NDArray64,
    tau: float = 0.5,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
    n_worker: int = 16,
) -> NDArray64:

    if tau == 0.0:
        return np.zeros((V_mV.shape[0], m_max + 1))

    if isinstance(Delta_meV, float):
        Delta_meV_tuple: tuple[float, float] = Delta_meV, Delta_meV
    elif isinstance(Delta_meV, tuple):
        Delta_meV_tuple: tuple[float, float] = Delta_meV
    else:
        raise KeyError("Delta_meV must be float | tuple[float, float]")
    Delta_meV: NDArray64 = np.array(Delta_meV_tuple, dtype="float64")

    if isinstance(gamma_meV, float):
        gamma_meV_tuple: tuple[float, float] = gamma_meV, gamma_meV
    elif isinstance(gamma_meV, tuple):
        gamma_meV_tuple: tuple[float, float] = gamma_meV
    else:
        raise KeyError("gamma_meV must be float | tuple[float, float]")
    gamma_meV: NDArray64 = np.array(gamma_meV_tuple, dtype="float64")
    gamma_meV = np.where(gamma_meV > gamma_meV_min, gamma_meV, gamma_meV_min)

    V_mV = np.round(V_mV, decimals=V_tol_mV)
    tau = np.round(tau, decimals=tau_tol)
    T_K = np.round(T_K, decimals=T_tol_K)
    Delta_meV = np.round(Delta_meV, decimals=Delta_tol_meV)
    gamma_meV = np.round(gamma_meV, decimals=gamma_tol_meV)

    # voltage axis
    V_0_mV = V_mV
    V_max_mV = np.max(np.abs(V_0_mV))
    dV_mV = np.abs(np.nanmax(V_0_mV) - np.nanmin(V_0_mV)) / (V_0_mV.shape[0] - 1)

    cache_key = cache_hash_pbar(
        tau=tau,
        T_K=T_K,
        Delta_1_meV=Delta_meV[0],
        Delta_2_meV=Delta_meV[1],
        gamma_1_meV=gamma_meV[0],
        gamma_2_meV=gamma_meV[1],
        string="FCS",
    )
    cached_file = os.path.join(CACHE_DIR, f"{cache_key}.npz")

    V_eV = np.arange(0, V_max_mV + dV_mV, dV_mV)
    V_eV = np.round(V_eV, decimals=V_tol_mV)

    if os.path.exists(cached_file):
        cache_data = np.load(cached_file)
        V_cached_mV = np.round(cache_data["V_mV"], decimals=V_tol_mV)
        I_cached_nA = np.round(cache_data["I_nA"], decimals=V_tol_mV)
    else:
        V_cached_mV = np.empty((0), dtype="float64")
        I_cached_nA = np.empty((0, m_max + 1), dtype="float64")

    logic_uncached = np.logical_not(np.isin(V_eV, V_cached_mV))
    V_uncached_mV = V_eV[logic_uncached]

    logic_stashed = np.isin(V_cached_mV, V_eV)
    V_stashed_mV = V_cached_mV[logic_stashed]
    I_stashed_nA = I_cached_nA[logic_stashed]

    print(f"cached values: {V_stashed_mV.shape[0]}/{V_eV.shape[0]}")

    if V_uncached_mV.size > 0:
        V_uncached_mV, I_uncached_nA = run_multiple_fcs(
            V_mV=list(V_uncached_mV),
            tau=tau,
            T_K=T_K,
            Delta_1_meV=Delta_meV[0],
            Delta_2_meV=Delta_meV[1],
            gamma_1_meV=gamma_meV[0],
            gamma_2_meV=gamma_meV[1],
            n_worker=n_worker,
        )
    else:
        V_uncached_mV = np.array([], dtype="float64")
        I_uncached_nA = np.empty((0, m_max + 1), dtype="float64")

    # update cache
    V_cached_mV = np.concatenate((V_cached_mV, V_uncached_mV))
    I_cached_nA = np.concatenate((I_cached_nA, I_uncached_nA))
    sort_idx = np.argsort(V_cached_mV)
    V_cached_mV = np.round(V_cached_mV[sort_idx], decimals=V_tol_mV)
    I_cached_nA = np.round(I_cached_nA[sort_idx, :], decimals=V_tol_mV)

    # Save updated cache to disk for future reuse
    np.savez(
        cached_file,
        V_mV=V_cached_mV,
        I_nA=I_cached_nA,
        tau=tau,
        T_K=T_K,
        Delta_1_meV=Delta_meV[0],
        Delta_2_meV=Delta_meV[1],
        gamma_1_meV=gamma_meV[0],
        gamma_2_meV=gamma_meV[1],
    )

    V_out_mV = np.concatenate(
        (
            V_stashed_mV,
            -V_stashed_mV,
            V_uncached_mV,
            -V_uncached_mV,
        )
    )
    I_out_nA = np.concatenate(
        (
            I_stashed_nA,
            -I_stashed_nA,
            I_uncached_nA,
            -I_uncached_nA,
        )
    )

    V_out_mV = np.round(V_out_mV, decimals=V_tol_mV)
    I_out_nA = np.round(I_out_nA, decimals=V_tol_mV)

    I_nA = np.full((V_0_mV.shape[0], m_max + 1), np.nan)
    for i_col in range(m_max + 1):
        I_nA[:, i_col] = bin_y_over_x(
            x=V_out_mV,
            y=I_out_nA[:, i_col],
            x_bins=V_0_mV,
        )
    I_nA = np.round(I_nA, decimals=V_tol_mV)

    return np.array(I_nA, dtype="float64")
