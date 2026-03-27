from __future__ import annotations

import numpy as np

from ...utilities.types import NDArray64


def _is_uniform_axis(V_mV: NDArray64) -> bool:
    spacing = np.diff(np.asarray(V_mV, dtype=np.float64))
    if spacing.size == 0:
        return True
    return bool(np.allclose(spacing, spacing[0], rtol=1e-6, atol=1e-12))


def _trapz_weights(V_mV: NDArray64) -> NDArray64:
    axis = np.asarray(V_mV, dtype=np.float64)
    weights = np.empty_like(axis)
    weights[0] = 0.5 * (axis[1] - axis[0])
    weights[-1] = 0.5 * (axis[-1] - axis[-2])
    weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return weights


def apply_voltage_noise(
    V_mV: NDArray64,
    I_nA: NDArray64,
    sigma_V_mV: float,
) -> NDArray64:
    V = np.asarray(V_mV, dtype=np.float64)
    I = np.asarray(I_nA, dtype=np.float64)
    sigma = float(sigma_V_mV)

    if sigma <= 0.0:
        return I.copy()

    if _is_uniform_axis(V):
        from scipy.ndimage import gaussian_filter1d

        dV = float(V[1] - V[0])
        sigma_pts = sigma / dV
        if sigma_pts <= 1e-12:
            return I.copy()

        smoothed = gaussian_filter1d(
            I,
            sigma=sigma_pts,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
        norm = gaussian_filter1d(
            np.ones_like(I),
            sigma=sigma_pts,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
        return np.divide(smoothed, norm, out=I.copy(), where=norm > 0.0)

    dV = _trapz_weights(V)
    delta = V[None, :] - V[:, None]
    kernel = np.exp(-0.5 * (delta / sigma) ** 2)
    kernel *= dV[None, :]
    norm = np.sum(kernel, axis=1, keepdims=True)
    return np.divide(kernel @ I, norm[:, 0], out=I.copy(), where=norm[:, 0] > 0.0)
