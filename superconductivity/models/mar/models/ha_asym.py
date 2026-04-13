"""Asymmetric HA MAR model with parameter-keyed HDF5 caching."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ....utilities.constants import G0_muS, kB_meV_K
from ....utilities.types import NDArray64
from ...basics import get_Delta_meV
from ..backend import carlosha_asym as ha_asym
from ..core import (
    V_TOL_MV,
    AsymmetricHAParams,
    dequantize_voltage_mV,
    ensure_curve_cached,
    lookup_currents,
    reconstruct_odd_current,
    unique_positive_voltage_q,
)
from ..core.voltage import quantize_voltage_mV

_MAR_DIR = Path(__file__).resolve().parents[1]
CACHE_FILE = _MAR_DIR / "cache" / "cache.h5"
CACHE_ROOT_GROUP = "ha_asym/curves"

logger = logging.getLogger(__name__)


def _group_path(params: AsymmetricHAParams) -> str:
    """Return the HDF5 group path for one asymmetric HA parameter tuple."""
    return f"{CACHE_ROOT_GROUP}/{params.cache_key()}"


def _evaluate_positive_curve(
    V_positive_mV: NDArray64,
    params: AsymmetricHAParams,
) -> NDArray64:
    """Evaluate the asymmetric HA backend on strictly positive voltages."""
    if V_positive_mV.size == 0:
        return np.empty((0,), dtype=np.float64)

    delta_ref_meV = params.Delta_1_meV
    if delta_ref_meV <= 0.0:
        raise ValueError("Left gap reference must be positive for ha_asym.")

    Delta_1_T_meV = get_Delta_meV(params.Delta_1_meV, params.T_K)
    Delta_2_T_meV = get_Delta_meV(params.Delta_2_meV, params.T_K)
    if Delta_1_T_meV == 0.0 and Delta_2_T_meV == 0.0:
        return V_positive_mV * G0_muS * params.tau

    if not callable(getattr(ha_asym, "ha_asym_curve", None)):
        raise ImportError(
            "The asymmetric HA backend is not importable. "
            "Rebuild the compiled backend before evaluating ha_asym."
        )

    I_ref = np.array(
        ha_asym.ha_asym_curve(
            params.tau,
            kB_meV_K * params.T_K / delta_ref_meV,
            Delta_1_T_meV / delta_ref_meV,
            Delta_2_T_meV / delta_ref_meV,
            params.gamma_1_meV / delta_ref_meV,
            params.gamma_2_meV / delta_ref_meV,
            V_positive_mV / delta_ref_meV,
        ),
        dtype=np.float64,
    )
    return I_ref * delta_ref_meV * G0_muS


def _ensure_positive_curve_cached(
    V_positive_mV: NDArray64,
    params: AsymmetricHAParams,
    caching: bool,
) -> tuple[NDArray64, NDArray64]:
    """Ensure all requested positive voltages exist in the cache."""
    V_positive_q = unique_positive_voltage_q(V_positive_mV)

    def evaluate_missing_q(V_missing_q: NDArray64) -> NDArray64:
        if caching and V_missing_q.size > 0:
            logger.debug(
                "ha_asym cache miss for %s: +%s voltages",
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


def get_I_ha_asym_nA(
    V_mV: NDArray64,
    tau: float = 0.5,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
    caching: bool = True,
) -> NDArray64:
    """Return the asymmetric HA current for an arbitrary voltage grid."""
    V_mV = np.array(V_mV, dtype=np.float64, copy=False)
    if tau == 0.0:
        return np.zeros_like(V_mV)

    params = AsymmetricHAParams.from_raw(
        tau=tau,
        T_K=T_K,
        Delta_meV=Delta_meV,
        gamma_meV=gamma_meV,
        gamma_meV_min=gamma_meV_min,
    )

    V_mV = np.round(V_mV, decimals=V_TOL_MV)
    if not np.any(V_mV != 0.0):
        return np.zeros_like(V_mV)

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
    "get_I_ha_asym_nA",
]
