from typing import Sequence

import numpy as np
from scipy.constants import Boltzmann, e, hbar

from ...utilities.constants import G0_muS
from ...utilities.functions import bin_y_over_x, fill_nans
from ...utilities.safety import require_same_shape
from ...utilities.types import NDArray64
from .helper import prepare_uniform_inverse_lookup_table, upsample_linear_values_np


def _estimate_I_c_nA(I_sw_nA: Sequence[float] | np.ndarray) -> float:
    """Estimate critical current from harmonic supercurrent amplitudes.

    Parameters
    ----------
    I_sw_nA : Sequence[float] | np.ndarray
        Harmonic amplitudes of the supercurrent relation
        ``I_s(phi) = sum_n I_sw_nA[n-1] * sin(n * phi)``.

    Returns
    -------
    float
        Estimated ``I_c`` in nA.
    """
    harmonics = np.atleast_1d(np.asarray(I_sw_nA, dtype=np.float64))
    if harmonics.size == 0:
        return 0.0
    phi = np.linspace(-np.pi, np.pi, 4097, dtype=np.float64)
    I_phi = np.zeros_like(phi)
    for idx, amp_nA in enumerate(harmonics, start=1):
        I_phi += amp_nA * np.sin(idx * phi)
    return float(np.nanmax(np.abs(I_phi)))


def _get_switching_probability(
    I_up_nA: np.ndarray,
    I_c_nA: float,
    T_eff_K: float,
    ramp_rate_nA_per_s: float,
    attempt_rate_Hz: float,
) -> np.ndarray:
    """Compute cumulative switching probability along an up-ramp.

    Parameters
    ----------
    I_up_nA : np.ndarray
        Monotonic non-negative current grid for an up sweep.
    I_c_nA : float
        Critical current in nA.
    T_eff_K : float
        Effective noise temperature in kelvin.
    ramp_rate_nA_per_s : float
        Current ramp speed in nA/s.
    attempt_rate_Hz : float
        Attempt rate prefactor in Hz.

    Returns
    -------
    np.ndarray
        Switching probability ``P_sw(I)`` on ``I_up_nA``.
    """
    I_up = np.asarray(I_up_nA, dtype=np.float64)
    P_sw = np.zeros_like(I_up)
    if I_up.size == 0:
        return P_sw
    if I_c_nA <= 0.0:
        return np.ones_like(I_up)
    if T_eff_K <= 0.0:
        return (I_up >= I_c_nA).astype(np.float64)

    I_c_A = I_c_nA * 1e-9
    E_J_over_kB_K = hbar * I_c_A / (2.0 * e * Boltzmann)
    i = np.clip(I_up / I_c_nA, 0.0, None)

    gamma_Hz = np.full_like(i, attempt_rate_Hz)
    below = i < 1.0
    if np.any(below):
        i_b = i[below]
        term = np.sqrt(np.maximum(1.0 - i_b * i_b, 0.0)) - i_b * np.arccos(i_b)
        barrier = 2.0 * E_J_over_kB_K * term / max(T_eff_K, 1e-12)
        barrier = np.clip(barrier, 0.0, 80.0)
        gamma_Hz[below] = attempt_rate_Hz * np.exp(-barrier)

    survival = 1.0
    for idx in range(1, I_up.size):
        dI_nA = I_up[idx] - I_up[idx - 1]
        dt_s = max(dI_nA / ramp_rate_nA_per_s, 0.0)
        survival *= np.exp(-gamma_Hz[idx] * dt_s)
        if I_up[idx] >= I_c_nA:
            survival = 0.0
        P_sw[idx] = 1.0 - survival
    return np.clip(P_sw, 0.0, 1.0)


def _gaussian_smooth_uniform(
    y: np.ndarray,
    dx: float,
    sigma_x: float,
) -> np.ndarray:
    """Gaussian-smooth uniformly sampled data.

    Parameters
    ----------
    y : np.ndarray
        Input samples.
    dx : float
        Grid spacing in x-units.
    sigma_x : float
        Gaussian standard deviation in x-units.

    Returns
    -------
    np.ndarray
        Smoothed samples. Returns ``y`` unchanged for non-positive
        ``sigma_x`` or invalid spacing.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.size < 3 or sigma_x <= 0.0 or dx <= 0.0:
        return y_arr

    sigma_pts = sigma_x / dx
    if sigma_pts < 1e-6:
        return y_arr

    radius = max(int(np.ceil(4.0 * sigma_pts)), 1)
    idx = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (idx / sigma_pts) ** 2)
    kernel /= np.sum(kernel)

    y_pad = np.pad(y_arr, (radius, radius), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def get_I_rstj_slow_nA(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
    I_sw_nA: float | Sequence[float],
    T_eff_K: float = 0.15,
    GN_G0: float = 1.0,
    ramp_rate_nA_per_s: float = 1.0,
    attempt_rate_GHz: float = 10.0,
    sigma_I_nA: float = 0.0,
    dI_max_nA: float = 0.25,
    min_i_grid_points: int = 4001,
    include_shunt: bool = True,
    upsample: int = 100,
    fill_nan: bool = True,
) -> NDArray64:
    """Compute slow-ramp RSTJ-like I(V) with Kramers switching statistics.

    This model targets slow DC sweeps without microwave drive. It separates
    fast phase dynamics from the sweep timescale by computing a switching
    probability ``P_sw(I)`` from a thermal escape-rate model, then mixing a
    zero-voltage state with the running branch.

    Parameters
    ----------
    V_mV : NDArray64
        Voltage bins (mV) where the output current is returned.
    I_qp_nA : NDArray64
        Quasiparticle I(V) branch in nA.
    I_sw_nA : float | Sequence[float]
        Harmonic amplitudes of the supercurrent relation.
    T_eff_K : float, default=0.15
        Effective noise temperature in kelvin.
    GN_G0 : float, default=1.0
        Dimensionless normal conductance ``G_N = G/G_0`` used for the shunt.
    ramp_rate_nA_per_s : float, default=1.0
        DC current ramp speed in nA/s.
    attempt_rate_GHz : float, default=10.0
        Escape-rate prefactor in GHz.
    sigma_I_nA : float, default=0.0
        RMS current-noise broadening (nA) applied to ``P_sw(I)`` by Gaussian
        convolution. Set to 0 to disable.
    dI_max_nA : float, default=0.25
        Maximum current step (nA) for internal switching-probability
        integration. Smaller values improve resolution at the cost of runtime.
    min_i_grid_points : int, default=4001
        Minimum number of points in the internal non-negative current grid.
    include_shunt : bool, default=True
        If ``True``, include the ohmic shunt term ``I_R = G V`` in the running
        branch lookup: ``I_qp_eff(V) = I_qp(V) + G V``.
    upsample : int, default=100
        Upsampling factor used before binning ``I(V)``.
    fill_nan : bool, default=True
        If ``True``, linearly fill NaN output bins after binning.

    Returns
    -------
    NDArray64
        Simulated current in nA on ``V_mV``.
    """
    if GN_G0 <= 0.0 or not np.isfinite(GN_G0):
        raise ValueError("GN_G0 must be finite and > 0.")
    if ramp_rate_nA_per_s <= 0.0 or not np.isfinite(ramp_rate_nA_per_s):
        raise ValueError("ramp_rate_nA_per_s must be finite and > 0.")
    if attempt_rate_GHz <= 0.0 or not np.isfinite(attempt_rate_GHz):
        raise ValueError("attempt_rate_GHz must be finite and > 0.")
    if sigma_I_nA < 0.0 or not np.isfinite(sigma_I_nA):
        raise ValueError("sigma_I_nA must be finite and >= 0.")
    if dI_max_nA <= 0.0 or not np.isfinite(dI_max_nA):
        raise ValueError("dI_max_nA must be finite and > 0.")
    if min_i_grid_points < 2:
        raise ValueError("min_i_grid_points must be >= 2.")
    if upsample <= 0:
        raise ValueError("upsample must be > 0.")

    V_arr_mV = np.asarray(V_mV, dtype=np.float64).reshape(-1)
    I_qp_arr_nA = np.asarray(I_qp_nA, dtype=np.float64).reshape(-1)
    require_same_shape(V_arr_mV, I_qp_arr_nA, "V_mV", "I_qp_nA")

    G_uS = float(GN_G0) * float(G0_muS)
    I_eff_nA = I_qp_arr_nA + G_uS * V_arr_mV
    if not include_shunt:
        I_eff_nA = I_qp_arr_nA

    I_lut_nA, V_lut_mV, _, _ = prepare_uniform_inverse_lookup_table(
        V_mV=V_arr_mV,
        I_qp_nA=I_eff_nA,
    )
    I_bias_nA = np.linspace(
        np.min(I_lut_nA),
        np.max(I_lut_nA),
        (V_arr_mV.shape[0] - 1) * 3 + 1,
        dtype=np.float64,
    )
    V_run_mV = np.interp(
        I_bias_nA,
        I_lut_nA,
        V_lut_mV,
        left=float(V_lut_mV[0]),
        right=float(V_lut_mV[-1]),
    )

    I_c_nA = _estimate_I_c_nA(np.atleast_1d(I_sw_nA))
    I_abs_max_nA = float(np.max(np.abs(I_bias_nA)))

    n_by_step = int(np.ceil(max(I_abs_max_nA, 0.0) / dI_max_nA)) + 1
    n_i_grid = max(int(min_i_grid_points), int(I_bias_nA.size), n_by_step)
    n_i_grid = min(n_i_grid, 200_001)
    I_up_nA = np.linspace(0.0, I_abs_max_nA, n_i_grid, dtype=np.float64)

    P_up = _get_switching_probability(
        I_up_nA=I_up_nA,
        I_c_nA=I_c_nA,
        T_eff_K=float(T_eff_K),
        ramp_rate_nA_per_s=float(ramp_rate_nA_per_s),
        attempt_rate_Hz=float(attempt_rate_GHz) * 1e9,
    )

    if I_up_nA.size > 1 and sigma_I_nA > 0.0:
        dI_grid_nA = float(I_up_nA[1] - I_up_nA[0])
        P_up = _gaussian_smooth_uniform(
            y=P_up,
            dx=dI_grid_nA,
            sigma_x=float(sigma_I_nA),
        )
        P_up = np.clip(P_up, 0.0, 1.0)
        P_up = np.maximum.accumulate(P_up)
        P_up[0] = 0.0

    P_bias = np.interp(
        np.abs(I_bias_nA),
        I_up_nA,
        P_up,
        left=0.0,
        right=1.0,
    )
    V_avg_mV = P_bias * V_run_mV

    I_over_nA = upsample_linear_values_np(I_bias_nA, upsample=upsample)
    V_over_mV = upsample_linear_values_np(V_avg_mV, upsample=upsample)
    I_out_nA = bin_y_over_x(
        x=V_over_mV,
        y=I_over_nA,
        x_bins=V_arr_mV,
    )
    if fill_nan:
        I_out_nA = fill_nans(
            y=I_out_nA,
            x=V_arr_mV,
            method="linear",
        )
    return np.asarray(I_out_nA, dtype=np.float64)
