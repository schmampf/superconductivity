"""FCS MAR model with parameter-keyed HDF5 caching."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ....utilities.constants import G0_muS, kB_meV_K
from ....utilities.types import NDArray64
from ...basics import get_Delta_meV
from ..backend import carlosfcs as fcs
from ..core import (
    V_TOL_MV,
    FCSParams,
    dequantize_voltage_mV,
    ensure_curve_cached,
    lookup_currents,
    reconstruct_odd_current,
    unique_positive_voltage_q,
)
from ..core.voltage import quantize_voltage_mV

_MAR_DIR = Path(__file__).resolve().parents[1]
CACHE_FILE = _MAR_DIR / "cache" / "cache.h5"
CACHE_ROOT_GROUP = "fcs/curves"
NMAX_DEFAULT = 10
_IW_DEFAULT = 2003
_NCHI_DEFAULT = 66

logger = logging.getLogger(__name__)


def _group_path(params: FCSParams) -> str:
    """Return the HDF5 group path for one FCS parameter tuple."""
    return f"{CACHE_ROOT_GROUP}/{params.cache_key()}"


def _evaluate_positive_curve(
    V_positive_mV: NDArray64,
    params: FCSParams,
) -> NDArray64:
    """Evaluate the FCS backend on strictly positive voltages."""
    if V_positive_mV.size == 0:
        return np.empty((0, params.nmax + 1), dtype=np.float64)

    Delta_1_T_meV = get_Delta_meV(params.Delta_1_meV, params.T_K)
    Delta_2_T_meV = get_Delta_meV(params.Delta_2_meV, params.T_K)
    if Delta_1_T_meV == 0.0 and Delta_2_T_meV == 0.0:
        I_ohmic_nA = V_positive_mV * G0_muS * params.tau
        return np.column_stack(
            (
                I_ohmic_nA,
                np.zeros((V_positive_mV.size, params.nmax), dtype=np.float64),
            )
        )

    if not callable(getattr(fcs, "fcs_curve", None)):
        raise ImportError(
            "The FCS backend is not importable. "
            "Rebuild the compiled backend before evaluating fcs."
        )

    return np.array(
        fcs.fcs_curve(
            params.tau,
            kB_meV_K * params.T_K,
            Delta_1_T_meV,
            Delta_2_T_meV,
            params.gamma_1_meV,
            params.gamma_2_meV,
            V_positive_mV,
            params.nmax,
            params.iw,
            params.nchi,
        ),
        dtype=np.float64,
    )


def _ensure_positive_curve_cached(
    V_positive_mV: NDArray64,
    params: FCSParams,
    caching: bool,
) -> tuple[NDArray64, NDArray64]:
    """Ensure all requested positive voltages exist in the cache."""
    V_positive_q = unique_positive_voltage_q(V_positive_mV)

    def evaluate_missing_q(V_missing_q: NDArray64) -> NDArray64:
        if caching and V_missing_q.size > 0:
            logger.debug(
                "fcs cache miss for %s: +%s voltages",
                params.cache_key(),
                int(V_missing_q.size),
            )
        return _evaluate_positive_curve(
            V_positive_mV=dequantize_voltage_mV(V_missing_q),
            params=params,
        )

    return ensure_curve_cached(
        cache_file=CACHE_FILE,
        group_path=_group_path(params),
        attrs=params.attrs(),
        V_requested_q=V_positive_q,
        evaluate_missing_q=evaluate_missing_q,
        caching=caching,
    )


def get_I_fcs_nA(
    V_mV: NDArray64,
    tau: float = 0.5,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
    nmax: int = NMAX_DEFAULT,
    caching: bool = True,
) -> NDArray64:
    """Return total and charge-resolved FCS currents on any voltage grid."""
    V_mV = np.array(V_mV, dtype=np.float64, copy=False)
    if tau == 0.0:
        return np.zeros((V_mV.shape[0], int(nmax) + 1), dtype=np.float64)

    params = FCSParams.from_raw(
        tau=tau,
        T_K=T_K,
        Delta_meV=Delta_meV,
        gamma_meV=gamma_meV,
        gamma_meV_min=gamma_meV_min,
        nmax=nmax,
        iw=_IW_DEFAULT,
        nchi=_NCHI_DEFAULT,
    )

    V_mV = np.round(V_mV, decimals=V_TOL_MV)
    if not np.any(V_mV != 0.0):
        return np.zeros((V_mV.shape[0], params.nmax + 1), dtype=np.float64)

    V_positive_mV = np.abs(V_mV[V_mV != 0.0])
    V_cached_q, I_cached_nA = _ensure_positive_curve_cached(
        V_positive_mV=V_positive_mV,
        params=params,
        caching=caching,
    )
    I_positive_nA = lookup_currents(
        V_lookup_q=quantize_voltage_mV(V_positive_mV),
        V_cached_q=V_cached_q,
        I_cached_nA=I_cached_nA,
    )
    return reconstruct_odd_current(
        V_mV=V_mV,
        positive_lookup_nA=I_positive_nA,
    )


__all__ = [
    "get_I_fcs_nA",
]
