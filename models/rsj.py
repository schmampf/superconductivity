"""Overdamped RSJ wrappers for combined Shapiro steps and PAT-like QP branches.

This file provides convenience wrappers around the JAX-based overdamped RSJ
solver (`sim_V_RSJ_mV_jax`). The workflow is:

1) Build a quasiparticle branch `I_QP(V)` (BCS/Dynes + optional photon-assisted
   tunneling, or microscopic mesoscopic/FCS-based models).
2) Convert it to a voltage-as-a-function-of-current mapping `V_QP(I)` via
   binning/inversion (`bin_y_over_x`) and build a JIT-compiled interpolator.
3) Run an overdamped RSJ phase-dynamics simulation in current bias to obtain
   the time-averaged voltage <V> for each (I_dc, I_ac).
4) Invert <V>(I) back to I(V) for plotting.

Units and conventions
---------------------
- Inputs `V_bias` and `I_bias` are *dimensionless* axes used for plotting.
  They are converted internally via

    V_bias_mV = V_bias * Delta_meV,
    I_bias_nA = I_bias * (G_0_muS * Delta_meV),

  using µS·mV = nA.
- `A_bias` is a *dimensionless* microwave voltage amplitude (scaled by Δ).
  Internally, `A_bias_mV = A_bias * Delta_meV`.
- `A_bias_nA` is an effective AC *current* amplitude used by the RSJ solver.
  The mapping from voltage amplitude to current amplitude is parametrized by
  `kappa` and the conductance scale (see function docstrings).

Notes
-----
These helpers are designed for figure generation and qualitative modeling.
They are not a substitute for a full circuit model (e.g. frequency-dependent
embedding impedances, noise, heating, etc.).
"""

import numpy as np

from tqdm import tqdm

from utilities.cpd5 import NDArray64

from utilities.constants import G_0_muS

from utilities.functions import bin_y_over_x
from utilities.functions import oversample

from models.bcs_np import get_I_bcs_jnp_nA as get_I_bcs_nA
from models.fcs_pbar import get_I_nA as get_I_fcs_nA
from models.tg import get_I_pat_nA_from_I0 as get_I_pat_nA
from models.utg import get_I_nA as get_I_utg_nA

from models.abs import get_IC_AB_nA
from models.shapiro import get_I_p_ABS_nA


from models.rsj_jax import Interpolator
from models.rsj_jax import make_interp_V_QP_mV
from models.rsj_jax import sim_V_RSJ_mV_jax


def get_I_rsj_nA(
    V_bias: NDArray64,
    I_bias: NDArray64,
    A_bias: float | NDArray64,
    G_N: float = 1.0,
    I_SW: float = 1.0,
    kappa: float = 1.0,
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    gamma_meV: float = 1e-3,
    nu_GHz: float = 10.0,
    n_max: int = 100,
    n_periods_total: int = 10,
    n_periods_discard: int = 1,
    n_steps_per_period: int = 200,
):
    """Compute an effective I(V) including overdamped RSJ dynamics and PAT-QP branch.

    This function models a tunnel junction in the overdamped (capacitance-free)
    RSJ limit under microwave irradiation, where Shapiro-step features and a
    photon-assisted quasiparticle background occur simultaneously.

    The quasiparticle branch is generated as:

    - `I_BCS_nA(V)` from `theory.models.bcs.get_I_nA` (Dynes DOS), and
    - `I_PAT_nA(V)` from `theory.models.tg.get_I_pat_nA`.

    The RSJ simulation itself is performed in current bias using
    `sim_V_RSJ_mV_jax` with a JIT-compiled interpolator for `V_QP(I)`.

    Parameters
    ----------
    V_bias
        Dimensionless voltage axis, later converted to mV by
        `V_bias_mV = V_bias * Delta_meV`.
    I_bias
        Dimensionless current axis, later converted to nA by
        `I_bias_nA = I_bias * (G_0_muS * Delta_meV)`.
    A_bias
        Dimensionless microwave voltage amplitude(s) (scaled by Δ).
        Can be a scalar or a 1D array.
    G_N
        Dimensionless normal conductance `g = G/G_0`.
        Used to scale the QP branch and the effective AC current drive.
    I_SW
        Switching/scale factor applied to the superconducting contribution.
        For the tunnel-junction case here, the CPR is sinusoidal with
        amplitude `I_C`.
    kappa
        Phenomenological conversion factor mapping microwave voltage amplitude
        to an effective AC current amplitude: `A_bias_nA *= kappa * G_N`.
        (This keeps the model modular; you may set this from calibration.)
    Delta_meV
        Zero-temperature gap Δ(0) in meV used as the normalization scale.
    T_K
        Temperature in kelvin.
    gamma_meV
        Dynes broadening parameter γ in meV.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index used in the PAT sum.

    Returns
    -------
    NDArray64
        Simulated current in nA evaluated on `V_bias`.

        If `A_bias` is a scalar, returns a 1D array with shape `(V_bias.size,)`.
        If `A_bias` is an array, returns a 2D array with shape
        `(len(A_bias), V_bias.size)`.

    Notes
    -----
    - The returned I(V) is obtained by a numerical inversion of <V>(I) from the
      RSJ simulation.
    - Resolution of Shapiro steps depends strongly on the density of `I_bias`.
      If steps appear with nearly equal height, increase `I_bias` resolution.
    """

    A_bias = np.atleast_1d(np.asarray(A_bias, dtype=np.float64))
    I_rsj_nA = np.full((A_bias.shape[0], V_bias.shape[0]), np.nan)

    V_bias_mV = V_bias * Delta_meV
    A_bias_mV = A_bias * Delta_meV
    I_bias_nA = I_bias * G_0_muS * Delta_meV
    A_bias_nA = A_bias * G_0_muS * Delta_meV

    A_bias_nA *= kappa * G_N

    # generate QP current
    I_BCS_nA = get_I_bcs_nA(
        V_mV=V_bias_mV,
        G_N=G_N,
        Delta_meV=(Delta_meV, Delta_meV),
        T_K=T_K,
        gamma_meV=gamma_meV,
    )
    I_PAT_nA = get_I_pat_nA(
        A_mV=A_bias_mV,
        V_mV=V_bias_mV,
        I_nA=I_BCS_nA,
        nu_GHz=nu_GHz,
        N=n_max,
    )
    I_QP_nA = I_PAT_nA

    # generate SC
    I_C_nA = get_IC_AB_nA(Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    I_p_nA = I_SW * np.array([I_C_nA])

    for i, a_bias_nA in enumerate(tqdm(A_bias_nA)):
        I_QP_nA_over, V_bias_mV_over = oversample(I_QP_nA[i, :], V_bias_mV)
        v_QP_mV = bin_y_over_x(I_QP_nA_over, V_bias_mV_over, I_bias_nA)

        interp_V_QP_mV: Interpolator = make_interp_V_QP_mV(
            I_bias_nA=I_bias_nA,
            V_QP_mV=v_QP_mV,
            G_N=G_N,
        )

        v_mv_mV = sim_V_RSJ_mV_jax(
            I_bias_nA=I_bias_nA,
            A_bias_nA=a_bias_nA,
            interp_V_QP_mV=interp_V_QP_mV,
            I_p_nA=I_p_nA,
            nu_GHz=nu_GHz,
            n_periods_total=n_periods_total,
            n_periods_discard=n_periods_discard,
            n_steps_per_period=n_steps_per_period,
        )[0, :]
        v_mv_mV_over, I_bias_nA_over = oversample(v_mv_mV, I_bias_nA)
        i_rsj_nA = bin_y_over_x(v_mv_mV_over, I_bias_nA_over, V_bias_mV)
        I_rsj_nA[i, :] = i_rsj_nA

    return I_rsj_nA


def get_I_rsj_meso_nA(
    V_bias: NDArray64,
    I_bias: NDArray64,
    A_bias: float | NDArray64,
    tau: float | NDArray64 = 1.0,
    I_p_nA: NDArray64 = np.array([], dtype=np.float64),
    I_SW: float = 1.0,
    kappa: float = 1.0,
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    gamma_meV: float = 1e-4,
    nu_GHz: float = 10.0,
    n_max: int = 100,
    m_max: int = 10,
    p_max: int = 10,
    n_periods_total: int = 10,
    n_periods_discard: int = 1,
    n_steps_per_period: int = 200,
):
    """Compute an effective I(V) for a mesoscopic weak link with QP+PAMAR background.

    This function is the mesoscopic analogue of `get_I_rsj_nA`. The
    quasiparticle branch is constructed from microscopic/FCS-based current
    contributions (including photon-assisted MAR-like processes via UTG), and
    the superconducting CPR harmonics are obtained from an ABS-based Fourier
    expansion.

    Parameters
    ----------
    V_bias
        Dimensionless voltage axis, later converted to mV by
        `V_bias_mV = V_bias * Delta_meV`.
    I_bias
        Dimensionless current axis, later converted to nA by
        `I_bias_nA = I_bias * (G_0_muS * Delta_meV)`.
    A_bias
        Dimensionless microwave voltage amplitude(s) (scaled by Δ).
        Must be array-like (shape `(n_A,)`).
    tau
        Channel transmissions τ. May be a scalar or 1D array. The dimensionless
        conductance used internally is `G_N = sum(tau)`.
    I_p_nA
        Optional array of CPR harmonic amplitudes in nA.
        If empty, harmonics are computed from ABS CPR via
        `theory.models.shapiro.get_I_p_ABS_nA`.
    I_SW
        Global scaling factor applied to the superconducting harmonics.
    kappa
        Phenomenological conversion factor mapping microwave voltage amplitude
        to an effective AC current amplitude: `A_bias_nA *= kappa * G_N`.
    Delta_meV
        Zero-temperature gap Δ(0) in meV used as the normalization scale.
    T_K
        Temperature in kelvin.
    gamma_meV
        Broadening parameter used by the microscopic/FCS model.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index used in UTG/PAMAR sums.
    m_max
        Maximum MAR order used by the UTG/PAMAR model.
    p_max
        Maximum CPR harmonic order used when generating `I_p_nA` from ABS.

    Returns
    -------
    NDArray64
        Simulated current in nA evaluated on `V_bias`.

        If `A_bias` is a scalar, returns a 1D array with shape `(V_bias.size,)`.
        If `A_bias` is an array, returns a 2D array with shape
        `(len(A_bias), V_bias.size)`.

    Notes
    -----
    - The RSJ simulation uses a tabulated+interpolated `V_QP(I)` for each
      microwave amplitude to stabilize extrapolation at the edges.
    - Step visibility depends on the `I_bias` sampling and the number of
      simulated periods/steps per period.
    """
    V_bias = np.asarray(V_bias)
    I_bias = np.asarray(I_bias)
    A_bias = np.asarray(A_bias)
    tau = np.asarray(tau)

    I_rsj_nA = np.full((A_bias.shape[0], V_bias.shape[0]), np.nan)
    I_FCS_nA = np.zeros((tau.shape[0], V_bias.shape[0], m_max + 1))

    V_bias_mV = V_bias * Delta_meV
    A_bias_mV = A_bias * Delta_meV
    I_bias_nA = I_bias * G_0_muS * Delta_meV
    A_bias_nA = A_bias * G_0_muS * Delta_meV

    G_N = np.sum(tau)
    A_bias_nA *= kappa * G_N

    # generate QP current
    for i_tau, tau_i in enumerate(tau):
        I_FCS_nA[i_tau, :, :] = get_I_fcs_nA(
            V_mV=V_bias_mV,
            Delta_meV=Delta_meV,
            tau=tau_i,
            T_K=T_K,
            gamma_meV=gamma_meV,
        )
    I_FCS_nA = np.sum(I_FCS_nA, axis=0)
    I_PAMAR_nA = get_I_utg_nA(
        V_mV=V_bias_mV,
        A_mV=A_bias_mV,
        I_nA=I_FCS_nA,
        nu_GHz=nu_GHz,
        N=n_max,
        M=m_max,
    )
    I_PAMAR_nA = np.sum(I_PAMAR_nA[:, :, 1:], axis=2)
    I_QP_nA = I_PAMAR_nA

    # generate SC
    if np.shape(I_p_nA) == (0,):
        I_p_nA = get_I_p_ABS_nA(
            tau=tau,
            Delta_meV=Delta_meV,
            T_K=T_K,
            p_max=p_max,
        )
    I_p_nA *= I_SW

    for i, a_bias_nA in enumerate(tqdm(A_bias_nA)):
        I_QP_nA_over, V_bias_mV_over = oversample(I_QP_nA[i, :], V_bias_mV)
        v_QP_mV = bin_y_over_x(I_QP_nA_over, V_bias_mV_over, I_bias_nA)

        interp_V_QP_mV: Interpolator = make_interp_V_QP_mV(
            I_bias_nA=I_bias_nA,
            V_QP_mV=v_QP_mV,
            G_N=G_N,
        )

        v_mv_mV = sim_V_RSJ_mV_jax(
            I_bias_nA=I_bias_nA,
            A_bias_nA=a_bias_nA,
            interp_V_QP_mV=interp_V_QP_mV,
            I_p_nA=I_p_nA,
            nu_GHz=nu_GHz,
            n_periods_total=n_periods_total,
            n_periods_discard=n_periods_discard,
            n_steps_per_period=n_steps_per_period,
        )[0, :]
        v_mv_mV_over, I_bias_nA_over = oversample(v_mv_mV, I_bias_nA)
        i_rsj_nA = bin_y_over_x(v_mv_mV_over, I_bias_nA_over, V_bias_mV)
        I_rsj_nA[i, :] = i_rsj_nA

    return np.squeeze(I_rsj_nA)
