from typing import Sequence

import jax.numpy as jnp
import numpy as np

from ...utilities.constants import h_pVskB_meV_K
from ...utilities.types import NDArray64

JF32 = jnp.float32
JI32 = jnp.int32

JF32EPS: JF32 = np.float32(1e-9)

JPI32 = np.float32(np.pi)
JTWO_PI32 = np.float32(2.0 * np.pi)
JTWO_MPI32 = np.float32(2e-3 * np.pi)
Jh_pVs32 = np.float32(h_h_pVs


def stack_scalar_model_over_A(
    model_fn,
    *,
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
    I_sw_nA: Sequence[float],
    A_mV: float | Sequence[float],
    model_kwargs: dict,
) -> NDArray64:
    """Call a scalar-A model over one or many amplitudes, then stack."""
    A_arr_mV = np.atleast_1d(np.asarray(A_mV, dtype=np.float64))
    if A_arr_mV.size == 0:
        raise ValueError("A_mV must not be empty.")
    A_is_scalar = np.isscalar(A_mV)

    out = [
        model_fn(
            V_mV=V_mV,
            I_qp_nA=I_qp_nA,
            I_sw_nA=I_sw_nA,
            A_mV=float(a_mV),
            **model_kwargs,
        )
        for a_mV in A_arr_mV
    ]
    out_arr = np.asarray(out, dtype=np.float64)
    if A_is_scalar:
        return out_arr[0]
    return out_arr


def suggest_dt_Nt(
    *,
    V_mV: NDArray64,
    A_mV: float,
    nu_GHz: float,
    npp_mw: int = 200,
    npp_j: int = 50,
    npp_th: int = 50,
    npp_rc: int = 30,
    n_periods: int = 10,
    safety: float = 1.0,
    T_K: float | None = None,
    C_pF: float | None = None,
    I_qp_nA: NDArray64 | None = None,
) -> tuple[float, int]:
    """Suggest dt (ps) and Nt with microwave/Josephson/thermal/RC constraints."""
    if nu_GHz <= 0:
        raise ValueError("nu_GHz must be > 0.")
    if n_periods <= 0:
        raise ValueError("n_periods must be > 0.")
    if npp_th <= 0:
        raise ValueError("npp_th must be > 0.")
    if safety <= 0:
        raise ValueError("safety must be > 0.")

    T_mw_ps = 1000.0 / nu_GHz
    dt_mw_ps = T_mw_ps / npp_mw

    V_max_mV = float(np.nanmax(np.abs(np.asarray(V_mV, dtype=np.float32))))
    V_inst_max_mV = V_max_mV + abs(A_mV)

    if V_inst_max_mV <= 0:
        dt_j_ps = float("inf")
    else:
        fJ_GHz = 483.5979 * V_inst_max_mV
        Tj_ps = 1000.0 / fJ_GHz
        dt_j_ps = Tj_ps / npp_j

    dt_ps = min(dt_mw_ps, dt_j_ps)

    if T_K is not None and T_K > 0:
        # Resolve thermal phase-fluctuation scale via thermal voltage V_th = k_B T / e.
        V_th_mV = float(kB_meV_K) * float(T_K)
        f_th_GHz = 483.5979 * V_th_mV
        if np.isfinite(f_th_GHz) and f_th_GHz > 0:
            T_th_ps = 1000.0 / f_th_GHz
            dt_th_ps = T_th_ps / npp_th
            dt_ps = min(dt_ps, dt_th_ps)

    if C_pF is not None and I_qp_nA is not None and C_pF > 0:
        V_arr = np.asarray(V_mV, dtype=np.float32)
        I_arr = np.asarray(I_qp_nA, dtype=np.float32)
        if V_arr.size == I_arr.size and V_arr.size > 1:
            with np.errstate(invalid="ignore"):
                G_muS = np.gradient(I_arr, V_arr)
            G_max_muS = float(np.nanmax(np.abs(G_muS)))
            if np.isfinite(G_max_muS) and G_max_muS > 0:
                tau_rc_ps = (float(C_pF) / G_max_muS) * 1e6
                dt_rc_ps = tau_rc_ps / npp_rc
                dt_ps = min(dt_ps, dt_rc_ps)

    dt_ps = dt_ps / safety

    npp = max(4, int(round(T_mw_ps / dt_ps)))
    dt_ps = T_mw_ps / npp
    Nt = int(n_periods * npp)
    return float(dt_ps), int(Nt)


def prepare_uniform_lookup_table(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
) -> tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    """Return finite, monotonic, uniform lookup table arrays in float32."""
    V = np.asarray(V_mV, dtype=np.float32).reshape(-1)
    I = np.asarray(I_qp_nA, dtype=np.float32).reshape(-1)

    mask = np.isfinite(V) & np.isfinite(I)
    V = V[mask]
    I = I[mask]

    if V.size < 2:
        raise ValueError("RCSTJ lookup table needs at least 2 finite points.")

    order = np.argsort(V)
    V = V[order]
    I = I[order]

    V_unique, inv = np.unique(V, return_inverse=True)
    if V_unique.size != V.size:
        I_acc = np.zeros(V_unique.shape[0], dtype=np.float64)
        count = np.zeros(V_unique.shape[0], dtype=np.int64)
        np.add.at(I_acc, inv, I.astype(np.float64))
        np.add.at(count, inv, 1)
        I = (I_acc / np.maximum(count, 1)).astype(np.float32)
        V = V_unique.astype(np.float32)

    if V.size < 2:
        raise ValueError("RCSTJ lookup table collapsed to fewer than 2 points.")

    V_uniform = np.linspace(V[0], V[-1], V.size, dtype=np.float32)
    I_uniform = np.interp(V_uniform, V, I).astype(np.float32)

    dv_mV = np.float32(V_uniform[1] - V_uniform[0])
    inv_dv_per_mV = np.float32(1.0 / max(float(dv_mV), float(JF32EPS)))
    return V_uniform, I_uniform, np.float32(V_uniform[0]), inv_dv_per_mV


def prepare_uniform_inverse_lookup_table(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
) -> tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    """
    Build a uniform inverse lookup table for V(I) from tabulated I(V).

    Returns
    -------
    I_uniform_nA, V_uniform_mV, I0_nA, inv_dI_per_nA
        Uniform current grid, corresponding voltage values, and indexing
        constants for O(1) lookup with `lookup_linear_uniform_clamped`.
    """
    V_uniform_mV, I_uniform_nA, _, _ = prepare_uniform_lookup_table(
        V_mV=V_mV,
        I_qp_nA=I_qp_nA,
    )

    # Invert by sorting in current and averaging duplicate current samples.
    order = np.argsort(I_uniform_nA)
    I_sorted = I_uniform_nA[order]
    V_sorted = V_uniform_mV[order]

    I_unique, inv = np.unique(I_sorted, return_inverse=True)
    if I_unique.size != I_sorted.size:
        V_acc = np.zeros(I_unique.shape[0], dtype=np.float64)
        count = np.zeros(I_unique.shape[0], dtype=np.int64)
        np.add.at(V_acc, inv, V_sorted.astype(np.float64))
        np.add.at(count, inv, 1)
        V_sorted = (V_acc / np.maximum(count, 1)).astype(np.float32)
        I_sorted = I_unique.astype(np.float32)

    if I_sorted.size < 2:
        raise ValueError("Inverse lookup table needs at least 2 unique points.")

    I_uniform_out = np.linspace(
        I_sorted[0],
        I_sorted[-1],
        I_sorted.size,
        dtype=np.float32,
    )
    V_uniform_out = np.interp(I_uniform_out, I_sorted, V_sorted).astype(np.float32)

    dI_nA = np.float32(I_uniform_out[1] - I_uniform_out[0])
    inv_dI_per_nA = np.float32(1.0 / max(float(dI_nA), float(JF32EPS)))
    return I_uniform_out, V_uniform_out, np.float32(I_uniform_out[0]), inv_dI_per_nA


def lookup_linear_uniform_clamped(
    x0_mV: jnp.ndarray,
    inv_dv_per_mV: jnp.ndarray,
    y_grid: jnp.ndarray,
    x_q_mV: jnp.ndarray,
) -> jnp.ndarray:
    """Linear lookup on a uniform grid with endpoint clamping."""
    n = y_grid.shape[0]
    idx_f = (x_q_mV - x0_mV) * inv_dv_per_mV
    idx_f = jnp.clip(
        idx_f,
        jnp.asarray(0.0, dtype=JF32),
        jnp.asarray(n - 1, dtype=JF32),
    )

    i = jnp.floor(idx_f).astype(JI32)
    i = jnp.clip(i, 0, n - 2)

    w = idx_f - i.astype(JF32)
    y0 = y_grid[i]
    y1 = y_grid[i + 1]
    return y0 + w * (y1 - y0)


def oversample_linear_np(
    x: NDArray64,
    y: NDArray64,
    upsample: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample paired 1D arrays using NumPy linear interpolation.

    Parameters
    ----------
    x : NDArray64
        Input x-values.
    y : NDArray64
        Input y-values sampled on the same parameter grid as ``x``.
    upsample : int, default=100
        Integer upsampling factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Upsampled ``(x, y)`` arrays with shape ``(len(x) * upsample,)``.
    """
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    x_up = upsample_linear_values_np(x_arr, upsample=upsample)
    y_up = upsample_linear_values_np(y_arr, upsample=upsample)
    return x_up, y_up


def upsample_linear_values_np(
    values: NDArray64,
    upsample: int = 100,
) -> np.ndarray:
    """Upsample one 1D array with NumPy linear interpolation.

    Parameters
    ----------
    values : NDArray64
        Input values sampled on an evenly spaced parameter grid.
    upsample : int, default=100
        Integer upsampling factor.

    Returns
    -------
    np.ndarray
        Upsampled values with shape ``(len(values) * upsample,)``.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if upsample <= 1 or arr.size < 2:
        return arr

    n = arr.size
    n_out = n * int(upsample)
    t_src = np.arange(n, dtype=np.float64)
    t_dst = np.linspace(0.0, float(n - 1), n_out, dtype=np.float64)
    return np.interp(t_dst, t_src, arr)
