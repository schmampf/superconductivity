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
    sigma_V_mV: float,
    *,
    padding_sigma: float = _PADDING_SIGMA,
) -> NDArray64:
    """Build an extended support grid for voltage-noise averaging.

    Parameters
    ----------
    V_mV : NDArray64
        Target voltage grid in mV. Must be 1D, finite, and strictly
        increasing.
    sigma_V_mV : float
        Standard deviation of the voltage noise in mV.
    padding_sigma : float, optional
        Number of voltage-noise widths used as padding on both sides.

    Returns
    -------
    NDArray64
        Extended voltage grid with the same step as ``V_mV``.
    """
    V = _validate_curve_inputs(V_mV, np.zeros_like(V_mV, dtype=np.float64))[0]
    sigma_V = _validate_sigma(sigma_V_mV, "sigma_V_mV")
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
    V_support_mV: NDArray64,
    I_support_nA: NDArray64,
    sigma_V_mV: float,
    order: int,
    *,
    V_out_mV: NDArray64 | None = None,
) -> NDArray64:
    """Average a theory curve over Gaussian voltage fluctuations.

    Parameters
    ----------
    V_support_mV : NDArray64
        Voltage support points of the theory curve in mV. Must be strictly
        increasing.
    I_support_nA : NDArray64
        Theory current on ``V_support_mV`` in nA.
    sigma_V_mV : float
        Standard deviation of the voltage fluctuations in mV.
    order : int
        Kernel resolution hint. The current implementation performs a
        deterministic Gaussian-kernel average on the internal support
        grid and requires ``order >= 2``.
    V_out_mV : NDArray64 | None, optional
        Output voltage grid. Defaults to ``V_support_mV``.

    Returns
    -------
    NDArray64
        Mean current on ``V_out_mV`` after voltage-noise averaging.
    """
    V_support, I_support = _validate_curve_inputs(V_support_mV, I_support_nA)
    sigma_V = _validate_sigma(sigma_V_mV, "sigma_V_mV")
    V_out = V_support if V_out_mV is None else to_1d_float64(V_out_mV, "V_out_mV")
    require_all_finite(V_out, "V_out_mV")
    require_min_size(V_out, 2, "V_out_mV")
    if np.any(np.diff(V_out) <= 0.0):
        raise ValueError("V_out_mV must be strictly increasing.")
    if sigma_V == 0.0:
        return np.interp(V_out, V_support, I_support)

    order_int = int(order)
    if order_int < 2:
        raise ValueError("order must be >= 2.")

    step = float(np.median(np.diff(V_support)))
    if not np.allclose(np.diff(V_support), step, rtol=1e-8, atol=1e-12):
        return _apply_voltage_noise_general(
            V_support,
            I_support,
            sigma_V,
            V_out,
        )

    sigma_bins = sigma_V / step
    radius = max(int(np.ceil(_PADDING_SIGMA * sigma_bins)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    I_padded = np.pad(I_support, radius, mode="edge")
    I_smoothed = np.convolve(I_padded, kernel, mode="valid")
    return np.asarray(np.interp(V_out, V_support, I_smoothed), dtype=np.float64)


def _apply_voltage_noise_general(
    V_support_mV: NDArray64,
    I_support_nA: NDArray64,
    sigma_V_mV: float,
    V_out_mV: NDArray64,
) -> NDArray64:
    """Fallback Gaussian average for nonuniform support grids."""
    edges = np.empty(V_support_mV.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (V_support_mV[:-1] + V_support_mV[1:])
    edges[0] = V_support_mV[0] - 0.5 * (V_support_mV[1] - V_support_mV[0])
    edges[-1] = (
        V_support_mV[-1]
        + 0.5 * (V_support_mV[-1] - V_support_mV[-2])
    )
    widths = np.diff(edges)
    delta = (V_out_mV[:, None] - V_support_mV[None, :]) / sigma_V_mV
    weights = np.exp(-0.5 * delta**2) * widths[None, :]
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights /= weights_sum
    return np.asarray(weights @ I_support_nA, dtype=np.float64)


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
