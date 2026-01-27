from itertools import combinations_with_replacement
from functools import partial
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

import jax.numpy as jnp
from jax import device_put
from jax import vmap
from jax import jit
from jax import Array

from theory.utilities.types import NDArray64
from theory.utilities.constants import G_0_muS

FMap = Callable[[Array], Array]


# jax.config.update("jax_enable_x64", True)


@jit
def normalize_V(
    V_mV: Array,
    Delta_meV: Array,
) -> Array:
    return jnp.array(
        V_mV / Delta_meV,
        dtype=jnp.float64,
    )


@jit
def normalize_I(
    I_nA: Array,
    Delta_meV: Array,
) -> Array:
    return jnp.array(
        I_nA / (Delta_meV * G_0_muS),
        dtype=jnp.float64,
    )


@partial(vmap, in_axes=(0, None, None))
def chi2_for_single_pincode_indices(
    pincode_indices: Array,  # shape (k,)
    I_exp: Array,  # shape (N_V,)
    I_theo: Array,  # shape (N_tau, N_V)
) -> Array:
    """
    Evaluate chi² for a given set of pincode pincode_indices.
    """
    pincode_indices = pincode_indices.astype(jnp.int32)

    I_model = jnp.sum(jnp.take(I_theo, pincode_indices, axis=0), axis=0)

    chi2 = jnp.sum((I_model - I_exp) ** 2)
    return chi2


def chi2_for_all(
    chi2_for_batch: FMap,
    all_pincode_indices: Array,
    batch_size: int = 50_000,
) -> Array:
    """
    Apply a vmapped JAX function `func` to `data_np` in batches.

    Parameters
    ----------
    func :
        A JAX function that takes an array of shape (batch, ...) and returns
        an array of shape (batch, ...).
    all_pincode_indices :
        NumPy array of inputs to be split into batches along axis 0.
    batch_size :
        Maximum batch size to send to `func` at once.

    Returns
    -------
    Array
        Concatenated result of all batches.
    """
    n_total = all_pincode_indices.shape[0]
    chunks: list[Array] = []
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        pincode_indices_batch = device_put(
            all_pincode_indices[start:stop],
        )
        chunks.append(
            chi2_for_batch(
                pincode_indices_batch,
            )
        )
    return jnp.concatenate(chunks, axis=0)


def generate_pincodes(
    ch_max: int,
    tau: NDArray64,
) -> NDArray[np.int32]:
    """
    Generate all unique pincodes as combinations *with replacement* of indices 0..max_index.

    Each pincode has length ``ch_max`` and is sorted (i0 <= i1 <= ...),
    so permutations are automatically removed but repeated indices
    (multiple channels with the same transmission) are allowed.
    """
    n_tau = tau.shape[0]
    combs = list(combinations_with_replacement(range(n_tau), ch_max))
    return np.array(combs, dtype=np.int32)


def generate_constrained_pincodes(
    ch_max: int,
    tau: NDArray64,
    G_N: float,
    T_tol: float = 0.3,
) -> NDArray[np.int32]:
    """
    Generate pincodes (combinations with replacement of indices) whose
    total transmission sum lies in [T_min, T_max].

    Parameters
    ----------
    ch_max :
        Number of channels in the pincode.
    G_N, T_tol :
        Inclusive bounds for sum(tau_i) over the pincode.

    Returns
    -------
    pincodes :
        2D array of shape (N_keep, ch_max) with int32 indices into tau_theo.
    """
    n_tau = tau.shape[0]
    tau_max = np.max(tau)

    T_min, T_max = G_N - T_tol, G_N + T_tol

    results: list[list[int]] = []

    def backtrack(
        pos: int,
        start_idx: int,
        current_sum: float,
        current_indices: list[int],
    ):
        # pos: how many entries have been chosen so far (0..ch_max)
        # start_idx: minimum index allowed at this position
        # (for non-decreasing order)

        # Prune if we already exceed T_max
        if current_sum > T_max:
            return

        # If we've filled all positions, check the sum window
        if pos == ch_max:
            if T_min <= current_sum <= T_max:
                results.append(current_indices.copy())
            return

        remaining_slots = ch_max - pos

        # Lower-bound pruning:
        # even if we fill all remaining slots with tau_max, can we reach T_min?
        max_possible_sum = current_sum + remaining_slots * tau_max
        if max_possible_sum < T_min:
            # No way to ever reach T_min from here
            return

        # Try indices from start_idx upwards (combinations_with_replacement)
        for idx in range(start_idx, n_tau):
            tau_val = tau[idx]
            new_sum = current_sum + tau_val

            # Upper-bound pruning: if at this idx we already exceed T_max,
            # all larger idx will also exceed T_max (tau is sorted).
            if new_sum > T_max:
                break

            current_indices.append(idx)
            backtrack(pos + 1, idx, new_sum, current_indices)
            current_indices.pop()

    backtrack(pos=0, start_idx=0, current_sum=0.0, current_indices=[])

    if not results:
        return np.empty((0, ch_max), dtype=np.int32)

    return np.array(results, dtype=np.int32)
