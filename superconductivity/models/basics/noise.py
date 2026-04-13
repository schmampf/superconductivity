"""Voltage-noise helpers shared by BCS model composition."""

from __future__ import annotations

import numpy as np

from ...utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ...utilities.types import NDArray64

_PADDING_SIGMA = 6.0


def make_bias_support_grid(
    V_mV: NDArray64,
    sigmaV_mV: float,
    *,
    padding_sigma: float = _PADDING_SIGMA,
) -> NDArray64:
    """Build an extended support grid for voltage-noise averaging.

    Parameters
    ----------
    V_mV
        Target voltage grid in mV. Must be 1D, finite, and strictly
        increasing.
    sigmaV_mV
        Standard deviation of the voltage noise in mV.
    padding_sigma
        Number of voltage-noise widths used as padding on both sides.

    Returns
    -------
    NDArray64
        Extended voltage grid with the same step as ``V_mV``.
    """
    V = _validate_curve_inputs(V_mV, np.zeros_like(V_mV, dtype=np.float64))[0]
    sigma_V = _validate_sigma(sigmaV_mV, "sigmaV_mV")
    if sigma_V == 0.0:
        return V.copy()

    step = float(np.median(np.diff(V)))
    pad = max(float(padding_sigma) * sigma_V, 2.0 * step)
    count = int(np.ceil(pad / step))
    start = float(V[0]) - count * step
    stop = float(V[-1]) + count * step
    size = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, size, dtype=np.float64)


def apply_voltage_noise(
    V_mV: NDArray64,
    I_nA: NDArray64,
    sigmaV_mV: float,
    order: int,
) -> NDArray64:
    """Average a theory curve over Gaussian voltage fluctuations.

    The output is always returned on the input support grid ``V_mV``.
    Typical usage is to evaluate a model on a padded support grid built with
    :func:`make_bias_support_grid`, smooth it here, and then explicitly
    interpolate back to the originally requested voltage grid in the caller.

    Parameters
    ----------
    V_mV
        Voltage support points of the theory curve in mV. Must be strictly
        increasing.
    I_nA
        Theory current on ``V_mV`` in nA.
    sigmaV_mV
        Standard deviation of the voltage fluctuations in mV.
    order
        Kernel resolution hint. The current implementation performs a
        deterministic Gaussian-kernel average on the internal support grid and
        requires ``order >= 2``.

    Returns
    -------
    NDArray64
        Mean current on ``V_mV`` after voltage-noise averaging.
    """
    V_support, I_support = _validate_curve_inputs(V_mV, I_nA)
    sigma_V = _validate_sigma(sigmaV_mV, "sigmaV_mV")
    if sigma_V == 0.0:
        return I_support.copy()

    order_int = int(order)
    if order_int < 2:
        raise ValueError("order must be >= 2.")

    step = float(np.median(np.diff(V_support)))
    if not np.allclose(np.diff(V_support), step, rtol=1e-8, atol=1e-12):
        return _apply_voltage_noise_general(
            V_support,
            I_support,
            sigma_V,
        )

    sigma_bins = sigma_V / step
    radius = max(int(np.ceil(_PADDING_SIGMA * sigma_bins)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    I_padded = np.pad(I_support, radius, mode="edge")
    return np.convolve(I_padded, kernel, mode="valid")


def _apply_voltage_noise_general(
    V_mV: NDArray64,
    I_nA: NDArray64,
    sigmaV_mV: float,
) -> NDArray64:
    """Fallback Gaussian average for nonuniform support grids."""
    edges = np.empty(V_mV.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (V_mV[:-1] + V_mV[1:])
    edges[0] = V_mV[0] - 0.5 * (V_mV[1] - V_mV[0])
    edges[-1] = V_mV[-1] + 0.5 * (V_mV[-1] - V_mV[-2])
    widths = np.diff(edges)
    delta = (V_mV[:, None] - V_mV[None, :]) / sigmaV_mV
    weights = np.exp(-0.5 * delta**2) * widths[None, :]
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.asarray(weights @ I_nA, dtype=np.float64)


def _validate_curve_inputs(
    V_mV: NDArray64,
    I_nA: NDArray64,
) -> tuple[NDArray64, NDArray64]:
    V = to_1d_float64(V_mV, "V_mV")
    I = to_1d_float64(I_nA, "I_nA")
    require_same_shape(V, I, "V_mV", "I_nA")
    require_all_finite(V, "V_mV")
    require_all_finite(I, "I_nA")
    require_min_size(V, 2, "V_mV")
    if np.any(np.diff(V) <= 0.0):
        raise ValueError("V_mV must be strictly increasing.")
    return V, I


def _validate_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not np.isfinite(sigma):
        raise ValueError(f"{name} must be finite.")
    if sigma < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return sigma


__all__ = [
    "apply_voltage_noise",
    "make_bias_support_grid",
]
