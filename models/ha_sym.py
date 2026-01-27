import numpy as np
import os
import sys
import logging
from importlib import reload
from contextlib import contextmanager

import theory.models.carlosha.ha_sym as ha_sym

from theory.models.bcs import Delta_meV_of_T

from theory.utilities.types import NDArray64

from theory.utilities.functions import cache_hash_sym
from theory.utilities.functions import cache_hash_nuni
from theory.utilities.functions import bin_y_over_x

from theory.utilities.constants import G_0_muS
from theory.utilities.constants import k_B_meV

from theory.utilities.constants import V_tol_mV
from theory.utilities.constants import tau_tol
from theory.utilities.constants import T_tol_K
from theory.utilities.constants import Delta_tol_meV
from theory.utilities.constants import gamma_tol_meV

HOME_DIR = "/Users/oliver/Documents/p5control-bluefors-evaluation"
sys.path.append(HOME_DIR)

WORK_DIR = os.path.join(HOME_DIR, "theory/models/carlosha/")
CACHE_DIR = os.path.join(WORK_DIR, ".cache_sym")
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure logging
reload(logging)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_I_nA(
    V_mV: NDArray64,
    tau: float = 0.5,
    T_K: float = 0.0,
    Delta_meV: float = 0.18,
    gamma_meV: float = 1e-4,
    gamma_meV_min: float = 1e-4,
    # n_worker: int = 16,
    caching: bool = True,
) -> NDArray64:

    if tau == 0.0:
        return np.zeros_like(V_mV)

    Delta_T_meV = Delta_meV_of_T(Delta_meV, T_K)
    if Delta_T_meV == 0.0:
        return V_mV * G_0_muS * tau

    gamma_meV = gamma_meV_min if gamma_meV < gamma_meV_min else gamma_meV

    # voltage axis
    V_0_mV = V_mV
    V_max_mV = np.max(np.abs(V_0_mV))
    dV_mV = np.abs(np.nanmax(V_0_mV) - np.nanmin(V_0_mV)) / (V_0_mV.shape[0] - 1)
    V_mV = np.arange(dV_mV, V_max_mV + dV_mV, dV_mV, dtype="float64")

    cached_file: str = "dump.pyz"

    if caching:
        V_mV = np.round(V_mV, decimals=V_tol_mV)
        tau = np.round(tau, decimals=tau_tol)
        T_K = np.round(T_K, decimals=T_tol_K)
        Delta_meV = np.round(Delta_meV, decimals=Delta_tol_meV)
        gamma_meV = np.round(gamma_meV, decimals=gamma_tol_meV)

        cache_key = cache_hash_sym(
            V_max_mV=V_max_mV,
            dV_mV=dV_mV,
            tau=tau,
            T_K=T_K,
            Delta_meV=Delta_meV,
            gamma_meV=gamma_meV,
            string="ha_sym",
        )
        cached_file = os.path.join(CACHE_DIR, f"{cache_key}.npz")

    if os.path.exists(cached_file) and caching:
        cache_data = np.load(cached_file)
        V_mV: NDArray64 = cache_data["V_mV"].astype("float64")
        I_nA: NDArray64 = cache_data["I_nA"].astype("float64")
    else:
        T_Delta = k_B_meV * T_K / Delta_T_meV
        gamma_Delta = gamma_meV / Delta_T_meV
        V_Delta = V_mV / Delta_T_meV

        E_max: float = np.max(
            [
                10.0 * Delta_meV / Delta_T_meV,
                np.max(V_mV) / Delta_T_meV,
            ]
        )

        I_Delta: NDArray64 = np.array(
            ha_sym.ha_sym_curve(
                tau,
                T_Delta,
                gamma_Delta,
                -E_max,
                E_max,
                V_Delta,
            ),
            dtype=np.float64,
        )
        I_nA = I_Delta * Delta_T_meV * G_0_muS

        if caching:
            # save to cache
            np.savez(
                cached_file,
                V_mV=V_mV,
                I_nA=I_nA,
                tau=tau,
                T_K=T_K,
                Delta_meV=Delta_meV,
                gamma_meV=gamma_meV,
            )

    # make symmetric
    V_mV = np.concatenate((V_mV, np.zeros((1)), -V_mV))
    I_nA = np.concatenate((I_nA, np.zeros((1)), -I_nA))

    I_nA = bin_y_over_x(
        x=V_mV,
        y=I_nA,
        x_bins=V_0_mV,
    )
    return I_nA


def get_I_nA_nonuniform(
    V_mV: NDArray64,
    tau: float = 0.5,
    T_K: float = 0.0,
    Delta_meV: float = 0.18,
    gamma_meV: float = 1e-4,
    gamma_meV_min: float = 1e-4,
    caching: bool = True,
) -> NDArray64:

    if tau == 0.0:
        return np.zeros_like(V_mV)

    Delta_T_meV = Delta_meV_of_T(Delta_meV, T_K)
    if Delta_T_meV == 0.0:
        return V_mV * G_0_muS * tau

    gamma_meV = gamma_meV_min if gamma_meV < gamma_meV_min else gamma_meV

    cached_file: str = "dump.pyz"

    V_mV: NDArray64 = np.round(V_mV, decimals=V_tol_mV)
    tau: float = np.round(tau, decimals=tau_tol)
    T_K: float = np.round(T_K, decimals=T_tol_K)
    Delta_meV: float = np.round(Delta_meV, decimals=Delta_tol_meV)
    gamma_meV: float = np.round(gamma_meV, decimals=gamma_tol_meV)

    V_cached_mV: NDArray64 = np.empty((0), dtype="float64")
    I_cached_nA: NDArray64 = np.empty((0), dtype="float64")

    if caching:
        cache_key = cache_hash_nuni(
            tau=tau,
            T_K=T_K,
            Delta_meV=Delta_meV,
            gamma_meV=gamma_meV,
            string="ha_sym_nuni",
        )
        cached_file = os.path.join(CACHE_DIR, f"{cache_key}.npz")

        if os.path.exists(cached_file):
            cache_data = np.load(cached_file)

            V_cached_mV = np.round(
                np.array(cache_data["V_mV"], dtype=np.float64),
                decimals=V_tol_mV,
            )
            I_cached_nA = np.round(
                np.array(cache_data["I_nA"], dtype=np.float64),
                decimals=V_tol_mV,
            )

    # which V_mV are in V_cached_mV
    logic_cached = np.isin(V_mV, V_cached_mV)

    # which V_mV are not in V_cached_mV
    logic_uncached = np.logical_not(logic_cached)

    V_uncached_mV = V_mV[logic_uncached]
    I_uncached_nA = np.full_like(V_uncached_mV, np.nan)

    logger.info(
        "cached values: %s/%s",
        V_mV.shape[0] - V_uncached_mV.shape[0],
        V_mV.shape[0],
    )

    if V_uncached_mV.size > 0:

        T_Delta = k_B_meV * T_K / Delta_T_meV
        gamma_Delta = gamma_meV / Delta_T_meV
        V_Delta = V_uncached_mV / Delta_T_meV

        E_max: float = np.max(
            [
                10.0 * Delta_meV / Delta_T_meV,
                np.max(V_Delta),
            ]
        )

        I_Delta: NDArray64 = np.array(
            ha_sym.ha_sym_curve(
                tau,
                T_Delta,
                gamma_Delta,
                -E_max,
                E_max,
                V_Delta,
            ),
            dtype=np.float64,
        )
        I_uncached_nA = I_Delta * Delta_T_meV * G_0_muS

        # update cache
        V_cached_mV = np.concatenate((V_cached_mV, V_uncached_mV))
        I_cached_nA = np.concatenate((I_cached_nA, I_uncached_nA))

    sort_idx = np.argsort(V_cached_mV)
    V_cached_mV = np.round(V_cached_mV[sort_idx], decimals=V_tol_mV)
    I_cached_nA = np.round(I_cached_nA[sort_idx], decimals=V_tol_mV)

    if V_uncached_mV.size > 0 and caching:
        np.savez(
            cached_file,
            V_mV=V_cached_mV,
            I_nA=I_cached_nA,
            tau=tau,
            T_K=T_K,
            Delta_meV=Delta_meV,
            gamma_meV=gamma_meV,
        )

    # --- Build I_nA in the exact order (and multiplicity) of V_mV ---

    # V_cached_mV is sorted; use searchsorted for fast lookup
    idx = np.searchsorted(V_cached_mV, V_mV)

    # safety check: all requested voltages must exist in cache
    if not np.allclose(V_cached_mV[idx], V_mV):
        raise RuntimeError(
            "Cache grid does not contain all requested voltages. "
            "Rounding / V_tol_mV mismatch?"
        )

    I_nA = I_cached_nA[idx]
    return I_nA


# ---------------------------------------------------------------------------
# Parallel worker for non-uniform voltage grids
# ---------------------------------------------------------------------------


@contextmanager
def suppress_logging():
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def ha_sym_nonuniform_worker(
    V_mV: NDArray64,
    tau: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    caching: bool = False,
) -> NDArray64:
    """
    Small top-level worker wrapper so it can be used with ProcessPoolExecutor.

    Parameters are simple Python / NumPy types so they are easily picklable.
    """
    with suppress_logging():
        return get_I_nA_nonuniform(
            V_mV=V_mV,
            tau=float(tau),
            T_K=float(T_K),
            Delta_meV=float(Delta_meV),
            gamma_meV=float(gamma_meV),
            caching=caching,
        )
