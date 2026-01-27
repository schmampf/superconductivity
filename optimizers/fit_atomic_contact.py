"""
Docstring for theory.optimizers.fit_atomic_contact
"""

# region imports
from typing import Optional
from typing import TypedDict

from time import time

from functools import partial

import numpy as np
from numpy.typing import NDArray

import jax.numpy as jnp
from jax import device_put
from jax import vmap
from jax import jit
from jax import Array

from tqdm.auto import tqdm
from scipy.optimize import curve_fit

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from theory.utilities.constants import G_0_muS
from theory.utilities.types import NDArray64

from theory.models.ha_sym import get_I_nA
from theory.models.ha_sym import ha_sym_nonuniform_worker

from theory.optimizers.fit_atomic_contacts_helper import (
    remap_to_nonuniform_centers,
    generate_constrained_pincodes,
    chi2_for_all,
)

# endregion


def gaussian(x: NDArray64, A: float, x_0: float, HWHM: float) -> NDArray64:
    X: NDArray64 = (x - x_0) / HWHM
    N: float = -np.log(2)
    return A * np.exp(N * X**2)


class SolutionDict(TypedDict):
    V_exp_mV: NDArray64
    I_exp_nA: NDArray64
    G_exp: NDArray64
    w_exp: NDArray64

    V_theo_mV: NDArray64
    I_theo_nA: NDArray64
    G_theo: NDArray64
    w_theo: NDArray64

    channels_theo: NDArray[np.int32]
    tau_theo: NDArray64
    T_theo_K: NDArray64
    Delta_theo_meV: NDArray64
    gamma_theo_meV: NDArray64

    weights: NDArray64
    w_channels: NDArray64
    w_T_K: NDArray64
    w_Delta_meV: NDArray64
    w_gamma_meV: NDArray64
    w_para: NDArray64

    i_channels: NDArray[np.int32]
    i_T_K: int
    i_Delta_meV: int
    i_gamma_meV: int

    I_fit_nA: NDArray64
    G_fit: NDArray64

    tau_A: NDArray64
    tau_fit: NDArray64
    tau_hwhm: NDArray64
    Tau_fit: float
    T_fit_K: float
    Delta_fit_meV: float
    gamma_fit_meV: float

    mask: NDArray[np.bool]
    V_masked_mV: NDArray64
    w_masked: NDArray64

    I_exp_masked_nA: NDArray64
    I_fit_masked_nA: NDArray64
    I_theo_masked_nA: NDArray64

    G_exp_masked: NDArray64
    G_fit_masked: NDArray64
    G_theo_masked: NDArray64

    weights_fit_masked: NDArray64
    chi2_fit_masked: NDArray64

    Tau_min: float
    Tau_max: float
    weighted: bool
    all_pincode_indices: NDArray[np.int32]

    fitting_time: float
    generation_time: float


def fit_atomic_contact(
    V_exp_mV: NDArray64,
    I_exp_nA: NDArray64,
    Tau_min: float,
    Tau_max: float,
    n_channel: int = 6,
    n_worker: int = 8,
    w_exp: Optional[NDArray64] = None,
    w_theo: Optional[NDArray64] = None,
    V_theo_mV: Optional[NDArray64] = None,
    tau_theo: Optional[NDArray64] = None,
    T_theo_K: Optional[NDArray64] = None,
    Delta_theo_meV: Optional[NDArray64] = None,
    gamma_theo_meV: Optional[NDArray64] = None,
    pbar_gen_theo: bool = False,
    pbar_fit_data: bool = True,
) -> SolutionDict:
    """
    Docstring for fit_atomic_contact

    :param V_exp_mV: Voltage axis of experimental data (mV).
    :type V_exp_mV: NDArray64
    :param I_exp_nA: Current axis of experimental data (nA).
    :type I_exp_nA: NDArray64
    :param n_channel: Number of channels (default 6).
    :type n_channel: int
    :param n_worker: Number of available Threads (default 8)
    :type n_worker: int
    :param tau_tol: Tolerance in total transmission for sum of channels
    :type tau_tol: float
    :param G_N_0: Initial guess for normal conductance (G_0). May be None.
    :type G_N_0: Optional[float]
    :param V_theo_mV: Parameter space of voltage (def. Al, mV).
    :type V_theo_mV: Optional[NDArray64]
    :param tau_theo: Parameter space of transmission [0, 1] (def. Al).
    :type tau_theo: Optional[NDArray64]
    :param T_theo_K: Parameter space of temperature. (def. BF, K)
    :type T_theo_K: Optional[NDArray64]
    :param Delta_theo_meV: Parameter space of Delta (def. Al, meV).
    :type Delta_theo_meV: Optional[NDArray64]
    :param gamma_theo_meV: Parameter space of gamma (meV).
    :type gamma_theo_meV: Optional[NDArray64]
    :param pbar_gen_theo: Progressbar for generating theo data.
    :type pbar_gen_theo: bool
    :param pbar_fit_data: Progressbar for actual fitting.
    :type pbar_fit_data: bool
    """

    # region define parameter space
    if V_theo_mV is None:
        V_theo_mV: NDArray64 = np.concatenate(
            (
                np.arange(0.01, 0.3, 0.01, dtype=np.float64),
                np.arange(0.3, 1.5, 0.01, dtype=np.float64),
                np.arange(1.5, 2.5, 0.02, dtype=np.float64),
                np.arange(2.5, 5, 0.05, dtype=np.float64),
            ),
        )
        V_theo_mV *= 0.18

    if tau_theo is None:
        tau_theo: NDArray64 = np.concatenate(
            (
                np.arange(0.0, 0.6, 0.05, dtype=np.float64),
                np.arange(0.6, 0.9, 0.01, dtype=np.float64),
                np.arange(0.9, 1.0, 0.02, dtype=np.float64),
            ),
        )

    if T_theo_K is None:
        T_theo_K: NDArray64 = np.array(
            [0, 0.1, 0.2, 0.3],
            dtype=np.float64,
        )

    if Delta_theo_meV is None:
        Delta_theo_meV: NDArray64 = np.array(
            [
                0.170,
                0.175,
                0.180,
                0.182,
                0.184,
                0.186,
                0.188,
                0.190,
                0.192,
                0.194,
                0.196,
                0.198,
            ],
            dtype=np.float64,
        )

    if gamma_theo_meV is None:
        # gamma_theo_meV: NDArray64 = np.logspace(
        #     -4,
        #     0,
        #     9,
        #     dtype=np.float64,
        # )
        gamma_theo_meV: NDArray64 = np.arange(
            0.001,
            0.011,
            0.001,
            dtype=np.float64,
        )

    channels_theo: NDArray[np.int32] = np.arange(0, n_channel, dtype=np.int32)

    n_V: int = V_theo_mV.shape[0]
    n_tau: int = tau_theo.shape[0]
    n_T: int = T_theo_K.shape[0]
    n_Delta: int = Delta_theo_meV.shape[0]
    n_gamma: int = gamma_theo_meV.shape[0]
    # endregion

    # region generate theo data
    time0 = time()
    jobs: list[
        tuple[
            tuple[int, int, int, int],
            tuple[NDArray64, float, float, float, float, bool],
        ]
    ] = []
    for itau, tau_i in enumerate(tau_theo):
        for iT, T_K_i in enumerate(T_theo_K):
            for idel, Delta_i in enumerate(Delta_theo_meV):
                for igam, gamma_i in enumerate(gamma_theo_meV):
                    idx = (itau, iT, idel, igam)
                    args = (
                        V_theo_mV,
                        tau_i,
                        T_K_i,
                        Delta_i,
                        gamma_i,
                        True,
                    )
                    jobs.append((idx, args))

    I_theo_nA = np.empty((n_tau, n_T, n_Delta, n_gamma, n_V), dtype=np.float64)

    if pbar_gen_theo:
        p_total = tqdm(
            total=len(jobs),
            desc="(τ, T, Δ, γ)",
        )
        p_para = tqdm(
            total=(n_T * n_Delta * n_gamma),
            desc="(T, Δ, γ)",
            leave=True,
        )
        p_tau = tqdm(
            total=n_tau,
            desc="(τ)",
            leave=True,
        )

    with ProcessPoolExecutor(max_workers=n_worker) as ex:
        futures = {
            ex.submit(
                ha_sym_nonuniform_worker,
                *args,
            ): idx
            for idx, args in jobs
        }

        for fut in as_completed(futures):
            itau, iT, idel, igam = futures[fut]
            i_theo_nA = fut.result()
            I_theo_nA[itau, iT, idel, igam, :] = i_theo_nA

            if pbar_gen_theo:
                p_total.update(1)

                p_tau.n = itau + 1
                p_tau.refresh()

                n_para = iT * n_Delta * n_gamma + idel * n_gamma + igam
                p_para.n = n_para
                p_para.refresh()

    G_theo = np.where(V_theo_mV != 0, I_theo_nA / V_theo_mV / G_0_muS, np.nan)
    generation_time = time() - time0
    # endregion

    # region map exp data and mask nans
    weighted: bool = False
    if w_exp is not None:
        w_theo: NDArray64 = remap_to_nonuniform_centers(
            x=V_exp_mV,
            y=w_exp,
            x_centers=V_theo_mV,
            statistic="mean",
        )
        weighted: bool = True
    else:
        w_theo: NDArray64 = np.full_like(V_theo_mV, 1.0)
        w_exp: NDArray64 = np.full_like(V_exp_mV, np.nan)

    G_exp = np.where(V_exp_mV != 0, I_exp_nA / V_exp_mV / G_0_muS, np.nan)

    I_exp_nuni_nA: NDArray64 = remap_to_nonuniform_centers(
        x=V_exp_mV,
        y=I_exp_nA,
        x_centers=V_theo_mV,
        statistic="mean",
    )
    G_exp_nuni = np.where(V_theo_mV != 0, I_exp_nuni_nA / V_theo_mV / G_0_muS, np.nan)

    mask: NDArray[np.bool] = np.logical_and(
        np.logical_not(np.isnan(G_exp_nuni)),
        np.logical_not(np.isnan(w_theo)),
    )

    w_masked: NDArray64 = w_theo[mask]
    w_masked_jax: Array = device_put(w_masked)

    V_masked_mV: NDArray64 = V_theo_mV[mask]

    I_exp_masked_nA: NDArray64 = I_exp_nuni_nA[mask]
    G_exp_masked: NDArray64 = G_exp_nuni[mask]

    I_theo_masked_nA: NDArray64 = I_theo_nA[:, :, :, :, mask]
    G_theo_masked: NDArray64 = G_theo[:, :, :, :, mask]

    # endregion

    # region pincode possibilities
    all_pincode_indices: NDArray[np.int32] = generate_constrained_pincodes(
        ch_max=n_channel,
        tau=tau_theo,
        Tau_min=Tau_min,
        Tau_max=Tau_max,
    )

    all_pincode_indices_jax: Array = device_put(all_pincode_indices)
    # endregion

    # region fitting
    weights: NDArray64 = np.empty(
        (n_channel, n_tau, n_T, n_Delta, n_gamma),
        dtype=np.float64,
    )

    if pbar_fit_data:
        p_total = tqdm(total=n_T * n_Delta * n_gamma, desc="(T, Δ, γ)")

    time0 = time()
    for i_T_K, T_K_i in enumerate(T_theo_K):
        for i_Delta_meV, Delta_meV_i in enumerate(Delta_theo_meV):
            for i_gamma_meV, _ in enumerate(gamma_theo_meV):
                G_temp_theo = G_theo_masked[
                    :,
                    i_T_K,
                    i_Delta_meV,
                    i_gamma_meV,
                    :,
                ]  # / Delta_meV_i

                G_temp_exp = G_exp_masked  # / (Delta_meV_i * G_0_muS)

                G_temp_theo_jax: Array = device_put(G_temp_theo)
                G_temp_exp_jax: Array = device_put(G_temp_exp)

                if weighted:
                    # compile chi2 function
                    @jit
                    @partial(vmap, in_axes=0)
                    def chi2_for_batch(pincode_indices_batch: Array) -> Array:
                        """
                        Compute chi² for a batch of pincodes.
                        """
                        # I_theo_valid has shape (N_tau, N_valid).
                        # After take: (ch_max, N_valid) for this pincode.
                        # Sum over channels (axis=0) to get I_model(V)
                        # with shape (N_valid,).
                        G_model_batch = jnp.sum(
                            jnp.take(
                                G_temp_theo_jax,
                                pincode_indices_batch,
                                axis=0,
                            ),
                            axis=0,  # sum over channels
                        )

                        # Now I_model and I_exp_valid are both (N_valid,)
                        diff = G_model_batch - G_temp_exp_jax

                        # Sum over voltages to get scalar chi² for this pincode
                        return jnp.sum(w_masked_jax * diff**2, axis=0)

                else:
                    # compile chi2 function
                    @jit
                    @partial(vmap, in_axes=0)
                    def chi2_for_batch(pincode_indices_batch: Array) -> Array:
                        """
                        Compute chi² for a batch of pincodes.
                        """
                        # I_theo_valid has shape (N_tau, N_valid).
                        # After take: (ch_max, N_valid) for this pincode.
                        # Sum over channels (axis=0) to get I_model(V)
                        # with shape (N_valid,).
                        G_model_batch = jnp.sum(
                            jnp.take(
                                G_temp_theo_jax,
                                pincode_indices_batch,
                                axis=0,
                            ),
                            axis=0,  # sum over channels
                        )

                        # Now I_model and I_exp_valid are both (N_valid,)
                        diff = G_model_batch - G_temp_exp_jax

                        # Sum over voltages to get scalar chi² for this pincode
                        return jnp.sum(diff**2, axis=0)

                # evaluate chi2
                # (within batches, to prevent kernel from crashing)
                chi2_jax: Array = chi2_for_all(
                    chi2_for_batch,
                    all_pincode_indices_jax,
                    batch_size=50_000,
                )
                temp_weights: Array = jnp.exp(-0.5 * chi2_jax)

                index_weights: NDArray64 = np.zeros(
                    (n_channel, n_tau), dtype=np.float64
                )
                for i in range(n_channel):
                    index = jnp.bincount(
                        all_pincode_indices_jax[:, i],
                        temp_weights,
                        minlength=tau_theo.shape[0],
                    )
                    index_weights[i, :] = np.array(index, dtype=np.float64)

                weights[:, :, i_T_K, i_Delta_meV, i_gamma_meV] = index_weights

                if pbar_fit_data:
                    p_total.update(1)

    fitting_time = time() - time0
    # endregion

    # region evaluation
    w_T_K: NDArray64 = np.mean(weights, axis=(0, 1, 3, 4))
    w_Delta_meV: NDArray64 = np.mean(weights, axis=(0, 1, 2, 4))
    w_gamma_meV: NDArray64 = np.mean(weights, axis=(0, 1, 2, 3))
    w_para: NDArray64 = np.mean(weights, axis=(0, 1))

    indices: tuple[np.int64, ...] = np.unravel_index(
        np.argmax(w_para),
        np.shape(w_para),
    )
    i_T_K: int = int(indices[0])
    i_Delta_meV: int = int(indices[1])
    i_gamma_meV: int = int(indices[2])

    T_fit_K: float = T_theo_K[i_T_K]
    Delta_fit_meV: float = Delta_theo_meV[i_Delta_meV]
    gamma_fit_meV: float = gamma_theo_meV[i_gamma_meV]

    w_channels: NDArray64 = weights[:, :, i_T_K, i_Delta_meV, i_gamma_meV]

    i_channels: NDArray[np.int32] = np.flip(np.argmax(w_channels, axis=1))

    tau_A: NDArray64 = np.zeros_like(channels_theo, dtype=np.float64)
    tau_fit: NDArray64 = np.zeros_like(tau_A)
    tau_hwhm: NDArray64 = np.zeros_like(tau_A)
    for i in channels_theo:
        popt, _ = curve_fit(gaussian, tau_theo, w_channels[i, :])
        tau_A[i] = popt[0]
        tau_fit[i] = popt[1]
        tau_hwhm[i] = np.abs(popt[2])

    Tau_fit: float = float(np.sum(tau_fit))

    I_fit_nA: NDArray64 = np.zeros_like(V_exp_mV)
    for tau_fit_i in tau_fit:
        I_fit_nA += get_I_nA(
            V_mV=V_exp_mV,
            tau=tau_fit_i,
            T_K=T_fit_K,
            Delta_meV=Delta_fit_meV,
            gamma_meV=gamma_fit_meV,
            caching=False,
        )
    G_fit: NDArray64 = np.where(
        V_exp_mV != 0,
        I_fit_nA / V_exp_mV / G_0_muS,
        np.nan,
    )

    I_fit_masked_nA: NDArray64 = np.sum(
        I_theo_masked_nA[
            i_channels,
            i_T_K,
            i_Delta_meV,
            i_gamma_meV,
            :,
        ],
        axis=0,
    )
    G_fit_masked: NDArray64 = np.where(
        V_masked_mV != 0,
        I_fit_masked_nA / V_masked_mV / G_0_muS,
        np.nan,
    )

    chi2_fit_masked: NDArray64 = (G_fit_masked - G_exp_masked) ** 2

    if weighted:
        chi2_fit_masked *= w_masked

    weights_fit_masked: NDArray64 = np.exp(-0.5 * chi2_fit_masked)

    # endregion

    results: SolutionDict = {
        "V_exp_mV": V_exp_mV,
        "I_exp_nA": I_exp_nA,
        "G_exp": G_exp,
        "w_exp": w_exp,
        "V_theo_mV": V_theo_mV,
        "I_theo_nA": I_theo_nA,
        "G_theo": G_theo,
        "w_theo": w_theo,
        "channels_theo": channels_theo,
        "tau_theo": tau_theo,
        "T_theo_K": T_theo_K,
        "Delta_theo_meV": Delta_theo_meV,
        "gamma_theo_meV": gamma_theo_meV,
        "weights": weights,
        "w_T_K": w_T_K,
        "w_Delta_meV": w_Delta_meV,
        "w_gamma_meV": w_gamma_meV,
        "w_channels": w_channels,
        "w_para": w_para,
        "i_T_K": i_T_K,
        "i_Delta_meV": i_Delta_meV,
        "i_gamma_meV": i_gamma_meV,
        "i_channels": i_channels,
        "I_fit_nA": I_fit_nA,
        "G_fit": G_fit,
        "tau_A": tau_A,
        "tau_fit": tau_fit,
        "tau_hwhm": tau_hwhm,
        "Tau_fit": Tau_fit,
        "T_fit_K": T_fit_K,
        "Delta_fit_meV": Delta_fit_meV,
        "gamma_fit_meV": gamma_fit_meV,
        "mask": mask,
        "V_masked_mV": V_masked_mV,
        "w_masked": w_masked,
        "I_exp_masked_nA": I_exp_masked_nA,
        "I_fit_masked_nA": I_fit_masked_nA,
        "I_theo_masked_nA": I_theo_masked_nA,
        "G_exp_masked": G_exp_masked,
        "G_fit_masked": G_fit_masked,
        "G_theo_masked": G_theo_masked,
        "weights_fit_masked": weights_fit_masked,
        "chi2_fit_masked": chi2_fit_masked,
        "Tau_min": Tau_min,
        "Tau_max": Tau_max,
        "weighted": weighted,
        "all_pincode_indices": all_pincode_indices,
        "fitting_time": fitting_time,
        "generation_time": generation_time,
    }
    return results
