import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ...utilities.functions import bin_y_over_x, upsample
from ...utilities.safety import require_all_finite
from ...utilities.types import NDArray64
from .helper import (
    JF32,
    JF32EPS,
    JI32,
    JPI32,
    JTWO_MPI32,
    JTWO_PI32,
    Jh_pVs32,
    lookup_linear_uniform_clamped,
    prepare_uniform_lookup_table,
    suggest_dt_Nt,
)


def get_I_rcsj_nA(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
    I_sw_nA: float | Sequence[float],
    C_pF: float = 1.0,
    A_mV: float | Sequence[float] = 0.5,
    nu_GHz: float = 10.0,
    n_periods: int = 30,
    burn_fraction: float = 0.3,
) -> NDArray64:
    """Compute I(V) for RCSJ from a float32 lookup-table simulation."""
    if C_pF <= 0:
        raise ValueError("C_pF must be > 0 for RCSJ simulation.")
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
    require_all_finite(I_sw_arr, name="I_sw_nA")

    (
        V_lut_mV,
        I_lut_nA,
        V_lut0_mV,
        inv_dv_per_mV,
    ) = prepare_uniform_lookup_table(
        V_mV=V_mV,
        I_qp_nA=I_qp_nA,
    )

    dt_ps, Nt = suggest_dt_Nt(
        V_mV=V_lut_mV,
        A_mV=float(np.max(np.abs(A_arr_mV))),
        nu_GHz=nu_GHz,
        n_periods=n_periods,
        C_pF=C_pF,
        I_qp_nA=I_lut_nA,
    )

    I_bias_nA = np.linspace(
        np.min(I_lut_nA),
        np.max(I_lut_nA),
        (np.asarray(V_mV).shape[0] - 1) * 3 + 1,
        dtype=np.float32,
    )

    burn_index = int(burn_fraction * Nt)
    burn_index = int(np.clip(burn_index, 0, max(0, Nt - 1)))

    V_rcsj_mV = np.asarray(
        simulate_rcsj_with_pat_vac_batch(
            I_nA=jnp.asarray(I_bias_nA, dtype=JF32),
            A_mV=jnp.asarray(A_arr_mV, dtype=JF32),
            I_sw_nA=jnp.asarray(I_sw_arr, dtype=JF32),
            C_pF=jnp.asarray(C_pF, dtype=JF32),
            V_lut0_mV=jnp.asarray(V_lut0_mV, dtype=JF32),
            inv_dv_per_mV=jnp.asarray(inv_dv_per_mV, dtype=JF32),
            I_qp_lut_nA=jnp.asarray(I_lut_nA, dtype=JF32),
            nu_GHz=jnp.asarray(nu_GHz, dtype=JF32),
            dt_ps=jnp.asarray(dt_ps, dtype=JF32),
            Nt=Nt,
            burn_index=jnp.asarray(burn_index, dtype=JI32),
        ),
        dtype=np.float32,
    )

    I_rcsj_all_nA = np.full(
        (A_arr_mV.size, np.asarray(V_mV).shape[0]),
        np.nan,
        dtype=np.float64,
    )
    for i in range(A_arr_mV.size):
        I_bias_nA_over, V_rcsj_mV_over = upsample(
            x=I_bias_nA,
            y=V_rcsj_mV[i],
        )
        I_rcsj_all_nA[i] = bin_y_over_x(
            x=V_rcsj_mV_over,
            y=I_bias_nA_over,
            x_bins=np.asarray(V_mV),
        )

    if A_is_scalar:
        return I_rcsj_all_nA[0]
    return I_rcsj_all_nA


@functools.partial(jax.jit, static_argnames=("Nt",))
def simulate_rcsj_with_pat_vac_batch(
    I_nA: jnp.ndarray,
    A_mV: jnp.ndarray,
    I_sw_nA: jnp.ndarray,
    C_pF: jnp.ndarray,
    V_lut0_mV: jnp.ndarray,
    inv_dv_per_mV: jnp.ndarray,
    I_qp_lut_nA: jnp.ndarray,
    nu_GHz: jnp.ndarray,
    dt_ps: jnp.ndarray,
    Nt: int,
    burn_index: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched RCSJ simulation with O(1) uniform-grid lookup.

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
    h_pVs32 = jnp.asarray(Jh_h_pVs, dtype=JF32)

    w_THz = nu_GHz * two_mpi
    a = jnp.asarray(2.0, dtype=JF32) * A_col / (h_pVs32 * nu_GHz)

    dt_ps = jnp.asarray(dt_ps, dtype=JF32)
    dV_fac = (
        dt_ps
        * jnp.asarray(1e-6, dtype=JF32)
        / jnp.maximum(C_pF, jnp.asarray(JF32EPS, dtype=JF32))
    )

    n = jnp.arange(Nt, dtype=JF32)
    t_ps = n * dt_ps
    sinwt_all = jnp.sin(w_THz * t_ps)
    coswt_all = jnp.cos(w_THz * t_ps)

    v_span_mV = jnp.asarray(I_qp_lut_nA.shape[0] - 1, dtype=JF32)
    v_span_mV /= inv_dv_per_mV
    v_lut_min = V_lut0_mV
    v_lut_max = V_lut0_mV + v_span_mV

    v_margin = jnp.asarray(2.0, dtype=JF32) + jnp.asarray(2.0, dtype=JF32)
    v_margin *= jnp.abs(A_col)
    v_min = v_lut_min - v_margin
    v_max = v_lut_max + v_margin

    def step(n_i, carry):
        phi, V_mV, V_sum_mV = carry
        sinwt = sinwt_all[n_i]
        coswt = coswt_all[n_i]

        phi_tot = phi + a * sinwt
        V_tot_mV = V_mV + A_col * coswt

        I_qp_nA = lookup_linear_uniform_clamped(
            x0_mV=V_lut0_mV,
            inv_dv_per_mV=inv_dv_per_mV,
            y_grid=I_qp_lut_nA,
            x_q_mV=V_tot_mV,
        )
        I_sc_nA = jnp.sum(
            jnp.sin(phi_tot[..., None] * harm_idx) * I_sw_nA,
            axis=-1,
        )

        I_cap_nA = I_row - I_sc_nA - I_qp_nA
        V_next_mV = V_mV + dV_fac * I_cap_nA
        V_next_mV = jnp.clip(V_next_mV, v_min, v_max)

        phi_next = phi + V_next_mV * two_mpi * dt_ps / h_pVs32
        phi_next = jnp.mod(phi_next + pi, two_pi) - pi

        keep = (n_i >= burn_index).astype(JF32)
        V_sum_next_mV = V_sum_mV + V_next_mV * keep

        return (phi_next, V_next_mV, V_sum_next_mV)

    shape_2d = (A_mV.shape[0], I_nA.shape[0])
    phi0 = jnp.zeros(shape_2d, dtype=JF32)
    V0_mV = jnp.zeros(shape_2d, dtype=JF32)
    Vsum0_mV = jnp.zeros(shape_2d, dtype=JF32)
    carry0 = (phi0, V0_mV, Vsum0_mV)

    _, _, V_sum_mV = jax.lax.fori_loop(
        0,
        Nt,
        step,
        carry0,
    )

    n_kept = jnp.maximum(Nt - burn_index, 1)
    return V_sum_mV / jnp.asarray(n_kept, dtype=JF32)
