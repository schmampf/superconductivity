import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ...utilities.constants import G0_muS, kB_meV_K
from ...utilities.functions import bin_y_over_x
from ...utilities.safety import require_all_finite
from ...utilities.types import NDArray64
from .helper import (JF32, JF32EPS, JI32, JPI32, JTWO_MPI32, JTWO_PI32,
                     Jh_pVsJF32, lookup_linear_uniform_clamped,
                     prepare_uniform_inverse_lookup_table, suggest_dt_Nt,
     h_pVs         upsample_linear_values_np)


def get_I_rstj_nA(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
    I_sw_nA: float | Sequence[float],
    T_K: float = 0.0,
    A_mV: float | Sequence[float] = 0.5,
    nu_GHz: float = 10.0,
    n_periods: int = 30,
    burn_fraction: float = 0.3,
    seed: int = 1,
    GN_G0: float = 1.0,
    n_realizations: int = 1,
    include_shunt: bool = True,
) -> NDArray64:
    """Compute I(V) for RSTJ (RSJ + thermal noise) in float32.

    Parameters
    ----------
    V_mV : NDArray64
        Voltage bins (mV) on which to return I(V).
    I_qp_nA : NDArray64
        Quasiparticle I(V) curve in nA used for lookup.
    I_sw_nA : float | Sequence[float]
        Harmonic critical-current amplitudes (nA).
    T_K : float, default=0.0
        Temperature in kelvin.
    A_mV : float | Sequence[float], default=0.5
        Microwave amplitude(s) in mV.
    nu_GHz : float, default=10.0
        Microwave frequency in GHz.
    n_periods : int, default=30
        Number of microwave periods to simulate.
    burn_fraction : float, default=0.3
        Fraction of initial samples discarded for averaging.
    seed : int, default=1
        PRNG seed for thermal noise.
    GN_G0 : float, default=1.0
        Dimensionless normal conductance ``GN_G0 = G/G_0`` used to set the
        thermal-noise resistor via ``R_noise = 1 / (GN_G0 * G_0)``.
    n_realizations : int, default=1
        Number of independent stochastic trajectories to average per
        ``(A_mV, I_bias)`` point. Larger values reduce Monte-Carlo noise and
        better sample rare thermal slips, at increased runtime.
    include_shunt : bool, default=True
        If ``True``, include the constant shunt branch ``I_R = G V`` in the
        deterministic lookup by replacing ``I_qp_nA`` with
        ``I_qp_nA + (GN_G0 * G_0) * V_mV``. This is the standard RSJ-style
        parallel resistor and usually improves finite-T rounding near zero
        voltage.

    Returns
    -------
    NDArray64
        Simulated current map in nA on ``V_mV``.
    """
    if n_periods <= 0:
        raise ValueError("n_periods must be > 0.")
    if not (0.0 <= burn_fraction <= 1.0):
        raise ValueError("burn_fraction must be in [0, 1].")
    if GN_G0 <= 0.0 or not np.isfinite(GN_G0):
        raise ValueError("GN_G0 must be finite and > 0.")
    if n_realizations <= 0:
        raise ValueError("n_realizations must be > 0.")

    A_arr_mV = np.atleast_1d(np.asarray(A_mV, dtype=np.float32))
    if A_arr_mV.size == 0:
        raise ValueError("A_mV must not be empty.")
    A_is_scalar = np.isscalar(A_mV)

    I_sw_arr = np.atleast_1d(np.asarray(I_sw_nA, dtype=np.float32))
    if I_sw_arr.size == 0:
        raise ValueError("I_sw_nA must not be empty.")
    require_all_finite(I_sw_arr, name="I_sw_nA")

    G_uS = float(GN_G0) * float(G0_muS)
    R_noise_MOhm = np.float32(1.0 / max(G_uS, float(JF32EPS)))
    V_arr_mV = np.asarray(V_mV, dtype=np.float32)
    I_qp_arr_nA = np.asarray(I_qp_nA, dtype=np.float32)
    I_qp_eff_nA = I_qp_arr_nA + np.float32(G_uS) * V_arr_mV
    if not include_shunt:
        I_qp_eff_nA = I_qp_arr_nA

    I_lut_nA, V_iqp_lut_mV, I_lut0_nA, inv_dI_per_nA = (
        prepare_uniform_inverse_lookup_table(
            V_mV=V_arr_mV,
            I_qp_nA=I_qp_eff_nA,
        )
    )

    dt_ps, Nt = suggest_dt_Nt(
        V_mV=V_arr_mV,
        A_mV=float(np.max(np.abs(A_arr_mV))),
        nu_GHz=nu_GHz,
        n_periods=n_periods,
        T_K=T_K,
    )

    I_bias_nA = np.linspace(
        np.min(I_lut_nA),
        np.max(I_lut_nA),
        (np.asarray(V_mV).shape[0] - 1) * 3 + 1,
        dtype=np.float32,
    )

    burn_index = int(burn_fraction * Nt)
    burn_index = int(np.clip(burn_index, 0, max(0, Nt - 1)))

    V_rstj_mV = np.asarray(
        simulate_rstj_with_pat_vac_batch(
            I_nA=jnp.asarray(I_bias_nA, dtype=JF32),
            A_mV=jnp.asarray(A_arr_mV, dtype=JF32),
            I_sw_nA=jnp.asarray(I_sw_arr, dtype=JF32),
            T_K=jnp.asarray(T_K, dtype=JF32),
            I_lut0_nA=jnp.asarray(I_lut0_nA, dtype=JF32),
            inv_dI_per_nA=jnp.asarray(inv_dI_per_nA, dtype=JF32),
            V_iqp_lut_mV=jnp.asarray(V_iqp_lut_mV, dtype=JF32),
            R_noise_MOhm=jnp.asarray(R_noise_MOhm, dtype=JF32),
            nu_GHz=jnp.asarray(nu_GHz, dtype=JF32),
            dt_ps=jnp.asarray(dt_ps, dtype=JF32),
            Nt=Nt,
            burn_index=jnp.asarray(burn_index, dtype=JI32),
            seed=jnp.asarray(seed, dtype=JI32),
            n_realizations=int(n_realizations),
        ),
        dtype=np.float32,
    )

    I_rstj_all_nA = np.full(
        (A_arr_mV.size, np.asarray(V_mV).shape[0]),
        np.nan,
        dtype=np.float64,
    )
    V_bins_mV = np.asarray(V_mV)
    I_bias_nA_over = upsample_linear_values_np(I_bias_nA)
    for i in range(A_arr_mV.size):
        V_rstj_mV_over = upsample_linear_values_np(V_rstj_mV[i])
        I_rstj_all_nA[i] = bin_y_over_x(
            x=V_rstj_mV_over,
            y=I_bias_nA_over,
            x_bins=V_bins_mV,
        )

    if A_is_scalar:
        return I_rstj_all_nA[0]
    return I_rstj_all_nA


@functools.partial(jax.jit, static_argnames=("Nt", "n_realizations"))
def simulate_rstj_with_pat_vac_batch(
    I_nA: jnp.ndarray,
    A_mV: jnp.ndarray,
    I_sw_nA: jnp.ndarray,
    T_K: jnp.ndarray,
    I_lut0_nA: jnp.ndarray,
    inv_dI_per_nA: jnp.ndarray,
    V_iqp_lut_mV: jnp.ndarray,
    R_noise_MOhm: jnp.ndarray,
    nu_GHz: jnp.ndarray,
    dt_ps: jnp.ndarray,
    Nt: int,
    burn_index: jnp.ndarray,
    seed: jnp.ndarray,
    n_realizations: int,
) -> jnp.ndarray:
    """
    Batched RSTJ simulation with thermal current noise.

    Returns mean slow voltage (mV) with shape (A_mV.size, I_nA.size).
    """
    I_nA = jnp.ravel(jnp.asarray(I_nA, dtype=JF32))
    A_mV = jnp.ravel(jnp.asarray(A_mV, dtype=JF32))
    I_sw_nA = jnp.ravel(jnp.asarray(I_sw_nA, dtype=JF32))
    if I_sw_nA.shape[0] == 1:
        I_sw_1_nA = I_sw_nA[0]

        def supercurrent_nA(phi_tot: jnp.ndarray) -> jnp.ndarray:
            return jnp.sin(phi_tot) * I_sw_1_nA

    else:
        harm_idx = jnp.arange(
            1,
            I_sw_nA.shape[0] + 1,
            dtype=JF32,
        )[None, None, :]

        def supercurrent_nA(phi_tot: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(
                jnp.sin(phi_tot[..., None] * harm_idx) * I_sw_nA,
                axis=-1,
            )

    A_col = A_mV[None, :, None]
    I_row = I_nA[None, None, :]

    two_mpi = jnp.asarray(JTWO_MPI32, dtype=JF32)
    pi = jnp.asarray(JPI32, dtype=JF32)
    two_pi = jnp.asarray(JTWO_PI32, dtype=JF32)
    h_pVsJF32 = jnp.asarray(Jh_pVsJF32, dtype=JF32)

    w_THz = nu_GHz * two_mpi
    a = jnp.asarray(2.0, dtype=JF32) * A_col / (h_pVsJF32 * nu_GHz)

    sigma_mV2 = (
        jnp.asarray(2.0, dtype=JF32)
        * jnp.asarray(kB_meV_K, dtype=JF32)
        * jnp.maximum(T_K, jnp.asarray(0.0, dtype=JF32))
    h_pVsnp.maximum(dt_ps, jnh_pVsay(JF32EPS, dtype=JF32))
    )
    dVdI_const_MOhm = jnp.maximum(
        R_noise_MOhm,h_pVs
        jnp.asarray(JF32EPS, dtype=JF32),
    )

    n = jnp.arange(Nt, dtype=JF32)
    t_ps = n * dt_ps
    sinwt_all = jnp.sin(w_THz * t_ps)
    coswt_all = jnp.cos(w_THz * t_ps)

    def step(n_i, carry):
        phi, V_sum_mV, key = carry
        sinwt = sinwt_all[n_i]
        coswt = coswt_all[n_i]

        phi_tot = phi + a * sinwt

        I_sc_nA = supercurrent_nA(phi_tot)
        I_for_qp0_nA = I_row - I_sc_nA
        sigma_nA = jnp.sqrt(sigma_mV2 / dVdI_const_MOhm)
        key, sub = jax.random.split(key)
        I_noise_nA = sigma_nA * jax.random.normal(
            sub,
            shape=I_for_qp0_nA.shape,
            dtype=JF32,
        )

        I_for_qp_nA = I_for_qp0_nA + I_noise_nA
        V_qp_mV = lookup_linear_uniform_clamped(
            x0_mV=I_lut0_nA,
            inv_dv_per_mV=inv_dI_per_nA,
            y_grid=V_iqp_lut_mV,
            x_q_mV=I_for_qp_nA,
        )
        V_mV = V_qp_mV - A_col * coswt

        phi_next = phi + V_mV * two_mpi * dt_ps / h_pVsJF32
        phi_next = jnp.mod(phi_next + pi, two_pi) - pi

        keep = (n_i >= burn_index).astype(JF32)
        V_sum_next_mV = V_sum_mV + V_mV * keep
        return (phi_next, V_sum_next_mV, key)

    shape_3d = (n_realizations, A_mV.shape[0], I_nA.shape[0])
    phi0 = jnp.zeros(shape_3d, dtype=JF32)
    Vsum0_mV = jnp.zeros(shape_3d, dtype=JF32)h_pVs
    key0 = jax.random.PRNGKey(seed)
    carry0 = (phi0, Vsum0_mV, key0)

    _, V_sum_mV, _ = jax.lax.fori_loop(
        0,
        Nt,
        step,
        carry0,
    )

    n_kept = jnp.maximum(Nt - burn_index, 1)
    V_mean_mV = V_sum_mV / jnp.asarray(n_kept, dtype=JF32)
    return jnp.mean(V_mean_mV, axis=0)
