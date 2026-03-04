import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ...utilities.functions import bin_y_over_x, oversample
from ...utilities.types import NDArray64
from .helper import (
    JF32,
    JH_E_PVSJF32,
    JI32,
    JPI32,
    JTWO_MPI32,
    JTWO_PI32,
    lookup_linear_uniform_clamped,
    prepare_uniform_inverse_lookup_table,
    suggest_dt_Nt,
)


def get_I_rsj_nA(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
    I_sw_nA: float | Sequence[float],
    A_mV: float | Sequence[float] = 0.5,
    nu_GHz: float = 10.0,
    n_periods: int = 10,
    burn_fraction: float = 0.3,
) -> NDArray64:
    """Compute I(V) for RSJ from a float32 inverse lookup-table simulation."""
    if n_periods <= 0:
        raise ValueError("n_periods must be > 0.")
    if not (0.0 <= burn_fraction <= 1.0):
        raise ValueError("burn_fraction must be in [0, 1].")

    A_arr_mV = np.atleast_1d(np.asarray(A_mV, dtype=np.float32))
    if A_arr_mV.size == 0:
        raise ValueError("A_mV must not be empty.")
    A_is_scalar = np.isscalar(A_mV)

    I_sw_arr = np.atleast_1d(np.asarray(I_sw_nA, dtype=np.float32))
    if I_sw_arr.size == 0:
        raise ValueError("I_sw_nA must not be empty.")
    if not np.all(np.isfinite(I_sw_arr)):
        raise ValueError("I_sw_nA must contain only finite values.")

    I_lut_nA, V_iqp_lut_mV, I_lut0_nA, inv_dI_per_nA = (
        prepare_uniform_inverse_lookup_table(
            V_mV=V_mV,
            I_qp_nA=I_qp_nA,
        )
    )

    dt_ps, Nt = suggest_dt_Nt(
        V_mV=np.asarray(V_mV),
        A_mV=float(np.max(np.abs(A_arr_mV))),
        nu_GHz=nu_GHz,
        n_periods=n_periods,
    )

    I_bias_nA = np.linspace(
        np.min(I_lut_nA),
        np.max(I_lut_nA),
        (np.asarray(V_mV).shape[0] - 1) * 3 + 1,
        dtype=np.float32,
    )

    burn_index = int(burn_fraction * Nt)
    burn_index = int(np.clip(burn_index, 0, max(0, Nt - 1)))

    V_rsj_mV = np.asarray(
        simulate_rsj_with_pat_vac_batch(
            I_nA=jnp.asarray(I_bias_nA, dtype=JF32),
            A_mV=jnp.asarray(A_arr_mV, dtype=JF32),
            I_sw_nA=jnp.asarray(I_sw_arr, dtype=JF32),
            I_lut0_nA=jnp.asarray(I_lut0_nA, dtype=JF32),
            inv_dI_per_nA=jnp.asarray(inv_dI_per_nA, dtype=JF32),
            V_iqp_lut_mV=jnp.asarray(V_iqp_lut_mV, dtype=JF32),
            nu_GHz=jnp.asarray(nu_GHz, dtype=JF32),
            dt_ps=jnp.asarray(dt_ps, dtype=JF32),
            Nt=Nt,
            burn_index=jnp.asarray(burn_index, dtype=JI32),
        ),
        dtype=np.float32,
    )

    I_rsj_all_nA = np.full(
        (A_arr_mV.size, np.asarray(V_mV).shape[0]),
        np.nan,
        dtype=np.float64,
    )
    for i in range(A_arr_mV.size):
        I_bias_nA_over, V_rsj_mV_over = oversample(
            x=I_bias_nA,
            y=V_rsj_mV[i],
        )
        I_rsj_all_nA[i] = bin_y_over_x(
            x=V_rsj_mV_over,
            y=I_bias_nA_over,
            x_bins=np.asarray(V_mV),
        )

    if A_is_scalar:
        return I_rsj_all_nA[0]
    return I_rsj_all_nA


@functools.partial(jax.jit, static_argnames=("Nt",))
def simulate_rsj_with_pat_vac_batch(
    I_nA: jnp.ndarray,
    A_mV: jnp.ndarray,
    I_sw_nA: jnp.ndarray,
    I_lut0_nA: jnp.ndarray,
    inv_dI_per_nA: jnp.ndarray,
    V_iqp_lut_mV: jnp.ndarray,
    nu_GHz: jnp.ndarray,
    dt_ps: jnp.ndarray,
    Nt: int,
    burn_index: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched RSJ simulation with O(1) inverse lookup V(I).

    Returns mean slow voltage (mV) with shape (A_mV.size, I_nA.size).
    """
    I_nA = jnp.ravel(jnp.asarray(I_nA, dtype=JF32))
    A_mV = jnp.ravel(jnp.asarray(A_mV, dtype=JF32))
    I_sw_nA = jnp.ravel(jnp.asarray(I_sw_nA, dtype=JF32))
    harm_idx = jnp.arange(1, I_sw_nA.shape[0] + 1, dtype=JF32)[None, None, :]

    A_col = A_mV[:, None]
    I_row = I_nA[None, :]

    two_mpi = jnp.asarray(JTWO_MPI32, dtype=JF32)
    pi = jnp.asarray(JPI32, dtype=JF32)
    two_pi = jnp.asarray(JTWO_PI32, dtype=JF32)
    h_e_pVsJF32 = jnp.asarray(JH_E_PVSJF32, dtype=JF32)

    w_THz = nu_GHz * two_mpi
    a = jnp.asarray(2.0, dtype=JF32) * A_col / (h_e_pVsJF32 * nu_GHz)

    n = jnp.arange(Nt, dtype=JF32)
    t_ps = n * dt_ps
    sinwt_all = jnp.sin(w_THz * t_ps)
    coswt_all = jnp.cos(w_THz * t_ps)

    def step(n_i, carry):
        phi, V_sum_mV = carry
        sinwt = sinwt_all[n_i]
        coswt = coswt_all[n_i]

        phi_tot = phi + a * sinwt
        I_sc_nA = jnp.sum(
            jnp.sin(phi_tot[..., None] * harm_idx) * I_sw_nA,
            axis=-1,
        )
        I_for_qp_nA = I_row - I_sc_nA

        V_qp_mV = lookup_linear_uniform_clamped(
            x0_mV=I_lut0_nA,
            inv_dv_per_mV=inv_dI_per_nA,
            y_grid=V_iqp_lut_mV,
            x_q_mV=I_for_qp_nA,
        )
        V_mV = V_qp_mV - A_col * coswt

        phi_next = phi + V_mV * two_mpi * dt_ps / h_e_pVsJF32
        phi_next = jnp.mod(phi_next + pi, two_pi) - pi

        keep = (n_i >= burn_index).astype(JF32)
        V_sum_next_mV = V_sum_mV + V_mV * keep
        return (phi_next, V_sum_next_mV)

    shape_2d = (A_mV.shape[0], I_nA.shape[0])
    phi0 = jnp.zeros(shape_2d, dtype=JF32)
    Vsum0_mV = jnp.zeros(shape_2d, dtype=JF32)
    carry0 = (phi0, Vsum0_mV)

    _, V_sum_mV = jax.lax.fori_loop(
        0,
        Nt,
        step,
        carry0,
    )

    n_kept = jnp.maximum(Nt - burn_index, 1)
    return V_sum_mV / jnp.asarray(n_kept, dtype=JF32)
