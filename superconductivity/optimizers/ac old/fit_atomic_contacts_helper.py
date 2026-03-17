from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array, device_put
from numpy.typing import NDArray

from ..utilities.constants import G_0_muS
from ..utilities.types import NDArray64

FMap = Callable[[Array], Array]


def handle_G_N_exp(
    V_exp_mV: NDArray64,
    I_exp_nA: NDArray64,
    Delta_exp_meV: float = 0.18,
    V_threshhold_Delta: float = 3.0,
) -> tuple[float, NDArray[np.bool]]:
    V_threshold_mV = V_threshhold_Delta * Delta_exp_meV
    logic: NDArray[np.bool] = V_exp_mV >= V_threshold_mV
    G_N_0: float = np.nanmean(
        np.gradient(
            I_exp_nA[logic] / G_0_muS,
            V_exp_mV[logic],
        ),
        dtype=np.float64,
    )
    return G_N_0, logic


def centers_to_edges_center_clipped(x_centers: np.ndarray) -> np.ndarray:
    """
    Construct bin edges from non-uniform bin centers.

    Convention:
    - inner edges are midpoints between centers
    - first bin starts at first center
    - last bin ends at last center

    So bins are:
        [c0, mid01),
        [mid01, mid12),
        ...
        [mid_(N-2,N-1), c_(N-1)]

    Parameters
    ----------
    x_centers : (M,) array
        Strictly increasing bin centers.

    Returns
    -------
    x_edges : (M+1,) array
        Bin edges for use with np.digitize, etc.
    """
    x_centers = np.asarray(x_centers, dtype=float)
    if x_centers.ndim != 1:
        raise ValueError("x_centers must be 1D.")
    if not np.all(np.diff(x_centers) > 0):
        raise ValueError("x_centers must be strictly increasing.")

    M = x_centers.size
    if M == 1:
        # degenerate case: a single bin that is just that center
        return np.array([x_centers[0], x_centers[0]], dtype=float)

    mid = 0.5 * (x_centers[:-1] + x_centers[1:])  # length M-1

    x_edges = np.empty(M + 1, dtype=float)
    x_edges[0] = x_centers[0]
    x_edges[1:-1] = mid
    x_edges[-1] = x_centers[-1]
    return x_edges


def bin_1d_nonuniform(
    x: np.ndarray,
    y: np.ndarray,
    x_edges: np.ndarray,
    statistic: Literal["mean", "sum", "count"] = "mean",
) -> np.ndarray:
    """
    Bin y(x) onto non-uniform x_edges.

    Parameters
    ----------
    x : (N,) array
        Sample positions.
    y : (N,) array
        Values at those positions.
    x_edges : (M+1,) array
        Monotonic bin edges; bin j is [x_edges[j], x_edges[j+1]).
    statistic : {"mean", "sum", "count"}, optional
        Aggregation per bin.

    Returns
    -------
    out : (M,) array
        Binned values for each bin.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_edges = np.asarray(x_edges)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x_edges.ndim != 1:
        raise ValueError("x_edges must be 1D.")
    if not np.all(np.diff(x_edges) >= 0):
        raise ValueError("x_edges must be non-decreasing.")

    # bin indices for each x (0..M-1), where M = len(x_edges)-1
    idx = np.digitize(x, x_edges) - 1
    M = len(x_edges) - 1

    # keep only points that actually fall into a bin
    mask = (idx >= 0) & (idx < M)
    idx = idx[mask]
    y = y[mask]

    if statistic == "count":
        return np.bincount(idx, minlength=M).astype(float)

    sum_y = np.bincount(idx, weights=y, minlength=M)
    count = np.bincount(idx, minlength=M)

    if statistic == "sum":
        return sum_y

    out = np.full(M, np.nan, dtype=float)
    nz = count > 0
    out[nz] = sum_y[nz] / count[nz]
    return out


def remap_to_nonuniform_centers(
    x: np.ndarray,
    y: np.ndarray,
    x_centers: np.ndarray,
    statistic: Literal["mean", "sum", "count"] = "mean",
) -> np.ndarray:
    """
    Remap y(x) onto a non-uniform axis specified by bin centers.

    Bins are:
        [c0, mid01),
        [mid01, mid12),
        ...
        [mid_(N-2,N-1), c_(N-1)]

    Parameters
    ----------
    x, y : (N,) arrays
        Source data (must be same shape).
    x_centers : (M,) array
        Target axis, interpreted as bin centers (strictly increasing).
    statistic : {"mean", "sum", "count"}, optional

    Returns
    -------
    y_binned : (M,) array
        Aggregated values on the non-uniform axis given by x_centers.
    """
    edges = centers_to_edges_center_clipped(x_centers)
    return bin_1d_nonuniform(x, y, edges, statistic=statistic)


def generate_constrained_pincodes(
    ch_max: int,
    tau: NDArray64,
    Tau_min: float,
    Tau_max: float = 0.3,
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
        if current_sum > Tau_max:
            return

        # If we've filled all positions, check the sum window
        if pos == ch_max:
            if Tau_min <= current_sum <= Tau_max:
                results.append(current_indices.copy())
            return

        remaining_slots = ch_max - pos

        # Lower-bound pruning:
        # even if we fill all remaining slots with tau_max, can we reach T_min?
        max_possible_sum = current_sum + remaining_slots * Tau_max
        if max_possible_sum < Tau_min:
            # No way to ever reach T_min from here
            return

        # Try indices from start_idx upwards (combinations_with_replacement)
        for idx in range(start_idx, n_tau):
            tau_val = tau[idx]
            new_sum = current_sum + tau_val

            # Upper-bound pruning: if at this idx we already exceed T_max,
            # all larger idx will also exceed T_max (tau is sorted).
            if new_sum > Tau_max:
                break

            current_indices.append(idx)
            backtrack(pos + 1, idx, new_sum, current_indices)
            current_indices.pop()

    backtrack(pos=0, start_idx=0, current_sum=0.0, current_indices=[])

    if not results:
        return np.empty((0, ch_max), dtype=np.int32)

    return np.array(results, dtype=np.int32)


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
