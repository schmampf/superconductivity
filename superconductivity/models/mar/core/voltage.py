"""Voltage-grid helpers for MAR caches."""

from __future__ import annotations

import numpy as np

from ....utilities.types import NDArray64

V_TOL_MV: int = 6

VOLTAGE_SCALE = 10**V_TOL_MV


def quantize_voltage_mV(V_mV: NDArray64) -> NDArray64:
    """Quantize voltages to exact integer cache bins."""
    return np.rint(V_mV * VOLTAGE_SCALE).astype(np.int64)


def dequantize_voltage_mV(V_q: NDArray64) -> NDArray64:
    """Convert integer voltage bins back to millivolt floats."""
    return V_q.astype(np.float64) / VOLTAGE_SCALE


def unique_positive_voltage_q(V_mV: NDArray64) -> NDArray64:
    """Return unique positive quantized voltages for one requested grid."""
    V_mV = np.round(np.abs(V_mV), decimals=V_TOL_MV)
    positive = V_mV[V_mV > 0.0]
    return np.unique(quantize_voltage_mV(positive))


def reconstruct_odd_current(
    V_mV: NDArray64,
    positive_lookup_nA: NDArray64,
) -> NDArray64:
    """Reconstruct an odd current from positive-voltage lookup data."""
    nonzero = V_mV != 0.0
    if positive_lookup_nA.ndim == 1:
        I_out_nA = np.zeros_like(V_mV)
        I_out_nA[nonzero] = np.sign(V_mV[nonzero]) * positive_lookup_nA
        return I_out_nA

    I_out_nA = np.zeros(
        (V_mV.shape[0],) + positive_lookup_nA.shape[1:],
        dtype=positive_lookup_nA.dtype,
    )
    sign = np.sign(V_mV[nonzero]).reshape(
        (-1,) + (1,) * (positive_lookup_nA.ndim - 1),
    )
    I_out_nA[nonzero] = sign * positive_lookup_nA
    return I_out_nA


__all__ = [
    "V_TOL_MV",
    "VOLTAGE_SCALE",
    "dequantize_voltage_mV",
    "quantize_voltage_mV",
    "reconstruct_odd_current",
    "unique_positive_voltage_q",
]
