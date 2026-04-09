from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...utilities.types import NDArray64

_PADDING_SIGMA = 5.0


def apply_gap_distribution(
    curve_fn: Callable[[float], NDArray64],
    Delta_meV: float,
    sigma_Delta_meV: float,
    order: int,
) -> NDArray64:
    """Average a theory curve over a Gaussian gap distribution.

    Parameters
    ----------
    curve_fn : Callable[[float], NDArray64]
        Callable that returns the theory current for one gap value in meV.
    Delta_meV : float
        Mean gap value in meV.
    sigma_Delta_meV : float
        Standard deviation of the gap distribution in meV.
    order : int
        Number of support points used to sample the continuous gap
        distribution. Values below 3 are promoted to 3 internally.

    Returns
    -------
    NDArray64
        Mean current after averaging over the Gaussian gap distribution.

    Notes
    -----
    The Gaussian support is truncated to ``Delta >= 0`` and to a finite
    window of ``±5 sigma`` around the mean gap. The resulting weights are
    renormalized by direct numerical integration, which approximates a
    continuous channel average rather than a sparse quadrature sum.
    """
    delta = _validate_delta(Delta_meV)
    sigma = _validate_sigma(sigma_Delta_meV)
    order_int = int(order)
    if order_int < 2:
        raise ValueError("order must be >= 2.")

    if sigma == 0.0:
        return np.asarray(curve_fn(delta), dtype=np.float64)

    lower = max(0.0, delta - _PADDING_SIGMA * sigma)
    upper = delta + _PADDING_SIGMA * sigma
    if not upper > lower:
        return np.asarray(curve_fn(0.0), dtype=np.float64)

    count = max(order_int, 3)
    support = np.linspace(lower, upper, count, dtype=np.float64)
    weights = np.exp(-0.5 * ((support - delta) / sigma) ** 2)
    norm = float(np.trapezoid(weights, support))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.asarray(curve_fn(delta), dtype=np.float64)

    curves = np.stack(
        [
            np.asarray(curve_fn(float(delta_value)), dtype=np.float64)
            for delta_value in support
        ],
        axis=0,
    )
    weighted_curves = weights[:, None] * curves
    averaged = np.trapezoid(weighted_curves, support, axis=0) / norm
    return np.asarray(averaged, dtype=np.float64)


def _validate_delta(value: float) -> float:
    delta = float(value)
    if not np.isfinite(delta):
        raise ValueError("Delta_meV must be finite.")
    if delta < 0.0:
        raise ValueError("Delta_meV must be >= 0.")
    return delta


def _validate_sigma(value: float) -> float:
    sigma = float(value)
    if not np.isfinite(sigma):
        raise ValueError("sigma_Delta_meV must be finite.")
    if sigma < 0.0:
        raise ValueError("sigma_Delta_meV must be >= 0.")
    return sigma


__all__ = ["apply_gap_distribution"]
