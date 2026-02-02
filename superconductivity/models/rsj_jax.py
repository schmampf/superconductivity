# Optional JAX acceleration (used for RSJ sweeps)
import jax
import jax.numpy as jnp
import numpy as np
from jax import config as _jax_config
from jax import lax

from ..utilities.constants import G_0_muS, h_e_pVs
from ..utilities.types import JInterpolator, NDArray64

# Enable float64 if available (recommended for RSJ phase integration stability).
# If the installed jaxlib backend does not support x64, JAX will fall back.
_jax_config.update("jax_enable_x64", True)

# -------------------------------------------------------------------------
# JAX-accelerated RSJ sweep (vectorized, JIT-compatible)
# -------------------------------------------------------------------------

# Interpolator: TypeAlias = Callable[[jnp.ndarray], jnp.ndarray]


# def make_interp_V_QP_mV(
#     I_bias_nA: NDArray64,
#     V_QP_mV: NDArray64,
#     G_N: float,
# ) -> Interpolator:

#     mask = np.logical_not(np.isnan(V_QP_mV))
#     v_grid = jnp.array(V_QP_mV[mask])
#     i_grid = jnp.array(I_bias_nA[mask])

#     i0 = i_grid[0]
#     i1 = i_grid[-1]
#     v0 = v_grid[0]
#     v1 = v_grid[-1]

#     # Ohmic asymptote: i = G_N * v  ->  dv/di = 1/G_N
#     dvdi = 1.0 / (G_N * G_0_muS)

#     def _v_of_i(i_test: jnp.ndarray) -> jnp.ndarray:
#         i_test = jnp.asarray(i_test)

#         v_in = jnp.interp(i_test, i_grid, v_grid)  # clamps outside by default

#         v_left = v0 + dvdi * (i_test - i0)
#         v_right = v1 + dvdi * (i_test - i1)

#         v_out = jnp.where(i_test < i0, v_left, v_in)
#         v_out = jnp.where(i_test > i1, v_right, v_out)
#         return v_out

#     return jax.jit(_v_of_i)
