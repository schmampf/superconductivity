from functools import partial
from pickle import load as pickle_load
from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, device_put, jit, vmap
from models.bcs_jnp import bin_y_over_x as bin_y_over_x_jax
from numpy.typing import NDArray
from optimizers.fit_pincode_helper import (
    chi2_for_all,
    generate_constrained_pincodes,
    normalize_I,
    normalize_V,
)

from ..utilities.constants import G_0_muS
from ..utilities.functions import NDArray64, bin_y_over_x

# jax.config.update("jax_enable_x64", True)

# GLOBAL IMPORT

with open("optimizers/database/pincode_database.pickle", "rb") as file:
    THEO: Dict[str, NDArray64] = pickle_load(file)

V_THEO: NDArray64 = np.array(THEO["voltage"], dtype=np.float64)
V_THEO_JAX: Array = device_put(V_THEO)

I_THEO: NDArray64 = np.array(THEO["current"], dtype=np.float64)
I_THEO_JAX: Array = device_put(I_THEO)

TAU_THEO: NDArray64 = np.array(THEO["transmission"], dtype=np.float64)
TAU_THEO_JAX: Array = device_put(TAU_THEO)


def handle_G_N(
    V_mV: NDArray64,
    I_nA: NDArray64,
    G_N: Optional[float] = None,
    Delta_0_meV: float = 0.18,
    V_threshhold: float = 2.5,
) -> float:
    if G_N is None:
        logic: NDArray[np.bool] = V_mV >= V_threshhold * Delta_0_meV
        G_N: float = np.nanmean(
            np.gradient(
                I_nA[logic] / G_0_muS,
                V_mV[logic],
            )
        )
    return G_N


def get_pincode(
    V_mV: NDArray64,
    I_nA: NDArray64,
    Delta_meV: float = 0.18,
    ch_max: int = 4,
    G_N: Optional[float] = None,
):
    G_N: float = handle_G_N(V_mV, I_nA, G_N)

    all_pincode_indices: NDArray[np.int32] = generate_constrained_pincodes(
        ch_max=ch_max,
        tau=TAU_THEO,
        G_N=G_N,
    )

    # convert exp data to jnp
    V_mV_jax: Array = jnp.array(V_mV, dtype=jnp.float64)
    I_nA_jax: Array = jnp.array(I_nA, dtype=jnp.float64)
    Delta_meV_jax: Array = jnp.array(Delta_meV, dtype=jnp.float64)
    all_pincode_indices_jax: Array = jnp.array(all_pincode_indices, dtype=jnp.int32)

    # normalize experimental data
    V_exp: Array = normalize_V(V_mV_jax, Delta_meV_jax)
    I_exp: Array = normalize_I(I_nA_jax, Delta_meV_jax)

    # Bin experimental current onto the theoretical voltage grid
    I_exp_bin: Array = bin_y_over_x_jax(V_exp, I_exp, V_THEO_JAX)

    # Mask out voltages where I_exp is not finite to reduce computation
    valid: Array = jnp.isfinite(I_exp_bin)
    I_exp_valid: Array = device_put(I_exp_bin[valid])
    I_theo_valid: Array = device_put(I_THEO_JAX[:, valid])

    # compile chi2 function
    @jit
    @partial(vmap, in_axes=0)
    def chi2_for_batch(pincode_indices_batch: Array) -> Array:
        """
        Compute chi² for a batch of pincodes.
        """
        # I_theo_valid has shape (N_tau, N_valid).
        # After take: (ch_max, N_valid) for this pincode.
        # Sum over channels (axis=0) to get I_model(V) with shape (N_valid,).
        I_model_batch = jnp.sum(
            jnp.take(I_theo_valid, pincode_indices_batch, axis=0),
            axis=0,  # sum over channels
        )

        # Now I_model and I_exp_valid are both (N_valid,)
        diff = I_model_batch - I_exp_valid

        # Sum over voltages to get scalar chi² for this pincode
        return jnp.sum(diff**2, axis=0)

    # evaluate chi2 (within batches, to prevent kernel from crashing)
    chi2: Array = chi2_for_all(
        chi2_for_batch,
        all_pincode_indices_jax,
        batch_size=50_000,
    )

    # get best fitting pincode
    best_index: int = int(jnp.argmin(chi2))
    best_pincode_indices: NDArray[np.int32] = all_pincode_indices[best_index]
    best_pincode: NDArray64 = TAU_THEO[np.flip(best_pincode_indices)]

    I_fit = np.sum(I_THEO[best_pincode_indices], axis=0)

    I_fit_nA = (
        bin_y_over_x(
            V_THEO,
            I_fit,
            V_mV / Delta_meV,
        )
        * G_0_muS
        * Delta_meV
    )

    return best_pincode, I_fit_nA

    # pincode_weights: Array = jnp.exp(-0.5 * (chi2 - jnp.min(chi2)))
    # pincode_weights: Array = pincode_weights / jnp.sum(pincode_weights)

    # flat_pincode_indices: NDArray[np.int32] = all_pincode_indices.reshape(-1)
    # flat_pincode_weights: Array = jnp.repeat(pincode_weights, ch_max)

    # index_weights = jnp.bincount(
    #     flat_pincode_indices, flat_pincode_weights, length=TAU_THEO.shape[0]
    # )
    # index_weights: Array = index_weights / jnp.max(index_weights)

    # return np.array(TAU_THEO), np.array(index_weights), best_pincode


# def get_weights(
#     V_mV: NDArray64,
#     weights: Optional[NDArray64] = None,
# ) -> Array:
#     if weights is None:
#         W_exp: Array = jnp.ones_like(V_mV, dtype=jnp.float64)
#     else:
#         W_exp: Array = jnp.array(weights, dtype=jnp.float64)
#     return W_exp
