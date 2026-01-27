from typing import Callable
from typing import TypeAlias

import numpy as np


# Optional JAX acceleration (used for RSJ sweeps)
import jax
from jax import config as _jax_config


import jax.numpy as jnp
from jax import lax

from ..utilities.constants import G_0_muS, e, h
from ..utilities.types import NDArray64

# Enable float64 if available (recommended for RSJ phase integration stability).
# If the installed jaxlib backend does not support x64, JAX will fall back.
_jax_config.update("jax_enable_x64", True)

# -------------------------------------------------------------------------
# JAX-accelerated RSJ sweep (vectorized, JIT-compatible)
# -------------------------------------------------------------------------

Interpolator: TypeAlias = Callable[[jnp.ndarray], jnp.ndarray]


def make_interp_V_QP_mV(
    I_bias_nA: NDArray64,
    V_QP_mV: NDArray64,
    G_N: float,
) -> Interpolator:

    mask = np.logical_not(np.isnan(V_QP_mV))
    v_grid = jnp.array(V_QP_mV[mask])
    i_grid = jnp.array(I_bias_nA[mask])

    i0 = i_grid[0]
    i1 = i_grid[-1]
    v0 = v_grid[0]
    v1 = v_grid[-1]

    # Ohmic asymptote: i = G_N * v  ->  dv/di = 1/G_N
    dvdi = 1.0 / (G_N * G_0_muS)

    def _v_of_i(i_test: jnp.ndarray) -> jnp.ndarray:
        i_test = jnp.asarray(i_test)

        v_in = jnp.interp(i_test, i_grid, v_grid)  # clamps outside by default

        v_left = v0 + dvdi * (i_test - i0)
        v_right = v1 + dvdi * (i_test - i1)

        v_out = jnp.where(i_test < i0, v_left, v_in)
        v_out = jnp.where(i_test > i1, v_right, v_out)
        return v_out

    return jax.jit(_v_of_i)


def sim_V_RSJ_mV_jax(
    I_bias_nA: NDArray64,
    A_bias_nA: NDArray64,
    interp_V_QP_mV: Interpolator,
    I_p_nA: NDArray64,
    nu_GHz: float,
    n_periods_total: int = 500,
    n_periods_discard: int = 200,
    n_steps_per_period: int = 200,
) -> NDArray64:
    if jax is None or jnp is None or lax is None:
        raise ImportError(
            "JAX is not available. Install jax/jaxlib to use sim_V_RSJ_mV_jax."
        )

    # Convert inputs to JAX arrays and ensure expected ranks.
    # Users sometimes pass scalars; vmap requires at least 1D inputs.
    I_bias_nA = jnp.atleast_1d(jnp.asarray(I_bias_nA, dtype=jnp.float64))
    A_bias_nA = jnp.atleast_1d(jnp.asarray(A_bias_nA, dtype=jnp.float64))
    I_p_nA = jnp.atleast_1d(jnp.asarray(I_p_nA, dtype=jnp.float64))

    # Precompute time grid
    dn_steps_per_period = 1.0 / float(n_steps_per_period)
    n_steps_total = int(n_periods_total * n_steps_per_period)
    n_steps_discard = int(n_periods_discard * n_steps_per_period)

    omega = 2.0 * np.pi * nu_GHz * 1e9
    T_s = 1.0 / (nu_GHz * 1e9)
    dt_s = T_s * dn_steps_per_period

    n = jnp.arange(n_steps_total, dtype=jnp.int32)
    t = n.astype(jnp.float64) * dt_s
    sin_omega_t = jnp.sin(omega * t)

    # Josephson relation: dphi/dt = 2eV/h -> phi += (4*pi*e/h) * V * dt
    dphi_pmV = (4.0 * np.pi * e / h) * dt_s * 1e-3

    # Harmonic indices p = 1..P
    p = jnp.arange(1, I_p_nA.size + 1, dtype=jnp.float64)

    # ---------------------------------------------------------------------
    # RSJ integrator for one (I_dc, A_ac)
    # ---------------------------------------------------------------------
    def _for_one_A(A_ac_nA: jnp.ndarray) -> jnp.ndarray:
        # Vectorized over all DC bias points simultaneously.
        I_dc_nA_vec = I_bias_nA  # shape (n_I,)

        def step(carry, x):
            phi_vec, sumV_vec, count, k = carry
            sin_val = x

            # Total bias current at this time for all I_dc
            I_t_nA_vec = I_dc_nA_vec + A_ac_nA * sin_val

            # Supercurrent for all I_dc: I_s(phi) = sum_p I_p sin(p phi)
            I_sc_nA_vec = jnp.sum(
                I_p_nA[:, None] * jnp.sin(p[:, None] * phi_vec[None, :]), axis=0
            )

            # QP current and corresponding voltage via inverse QP curve
            I_qp_nA_vec = I_t_nA_vec - I_sc_nA_vec
            V_mV_vec = interp_V_QP_mV(I_qp_nA_vec)

            # Phase update (Josephson relation) + wrap to (-pi, pi]
            phi_vec = phi_vec + V_mV_vec * dphi_pmV
            phi_vec = jnp.mod(phi_vec + jnp.pi, 2.0 * jnp.pi) - jnp.pi

            take = k >= n_steps_discard
            sumV_vec = sumV_vec + jnp.where(take, V_mV_vec, 0.0)
            count = count + jnp.where(take, 1.0, 0.0)
            k = k + 1

            return (phi_vec, sumV_vec, count, k), None

        carry0 = (
            jnp.zeros_like(I_dc_nA_vec, dtype=jnp.float64),
            jnp.zeros_like(I_dc_nA_vec, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0, dtype=jnp.int32),
        )

        carry_f, _ = lax.scan(step, carry0, sin_omega_t)
        _, sumV_vec, count, _ = carry_f
        return sumV_vec / jnp.maximum(count, 1.0)

    # vmap over the AC amplitudes; jit the whole map so compilation happens once per shape.
    V_avg_mV = jax.jit(jax.vmap(_for_one_A))(A_bias_nA)

    # Return as NumPy array for compatibility with the rest of the file.
    return np.asarray(V_avg_mV)
