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

# Optional JAX acceleration (used for RSJ sweeps)
import jax
import jax.numpy as jnp
import numpy as np
from jax import config as _jax_config
from jax import lax
from tqdm import tqdm

from ..utilities.constants import G_0_muS, e, h, h_e_pVs
from ..utilities.functions import bin_y_over_x, oversample
from ..utilities.functions_jax import get_dydx, jnp_interp_y_of_x
from ..utilities.types import JInterpolator, NDArray64
from .abs import get_Ic_ab_nA
from .bcs_jnp import get_I_bcs_jnp_nA as get_I_bcs_nA
from .fcs_pbar import get_I_fcs_pbar_nA as get_I_fcs_nA
from .pat import get_I_pamar_nA, get_I_pat_nA
from .ss import get_I_p_abs_nA

# Enable float64 if available (recommended for RSJ phase integration stability).
# If the installed jaxlib backend does not support x64, JAX will fall back.
_jax_config.update("jax_enable_x64", True)


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
        T_K=T_K,
        Delta_meV=(Delta_meV, Delta_meV),
        gamma_meV=gamma_meV,
    )
    I_PAT_nA = get_I_pat_nA(
        V_mV=V_bias_mV,
        I_nA=I_BCS_nA,
        A_mV=A_bias_mV,
        nu_GHz=nu_GHz,
        n_max=n_max,
        m=1,
    )
    I_QP_nA = I_PAT_nA

    # generate SC
    I_C_nA = get_Ic_ab_nA(Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    I_p_nA = I_SW * np.array([I_C_nA])

    for i, a_bias_nA in enumerate(tqdm(A_bias_nA)):
        I_QP_nA_over, V_bias_mV_over = oversample(I_QP_nA[i, :], V_bias_mV)
        v_QP_mV = bin_y_over_x(I_QP_nA_over, V_bias_mV_over, I_bias_nA)

        # dydx = get_dydx(
        #     x=I_bias_nA,
        #     y=v_QP_mV,
        # )

        interp_V_QP_mV: JInterpolator = jnp_interp_y_of_x(
            x=I_bias_nA,
            y=v_QP_mV,
            dydx=G_N,
        )

        v_mv_mV = sim_V_RSJ_mV_jax(
            I_bias_nA=I_bias_nA,
            A_bias_nA=a_bias_nA,
            V_QP_mV=interp_V_QP_mV,
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
    I_PAMAR_nA = get_I_pamar_nA(
        V_mV=V_bias_mV,
        A_mV=A_bias_mV,
        I_nA=I_FCS_nA[:, 1:],
        nu_GHz=nu_GHz,
        n_max=n_max,
        m_max=m_max,
    )
    I_QP_nA = I_PAMAR_nA

    # generate SC
    if np.shape(I_p_nA) == (0,):
        I_p_nA = get_I_p_abs_nA(
            tau=tau,
            Delta_meV=Delta_meV,
            T_K=T_K,
            p_max=p_max,
        )
    I_p_nA *= I_SW

    for i, a_bias_nA in enumerate(tqdm(A_bias_nA)):
        I_QP_nA_over, V_bias_mV_over = oversample(I_QP_nA[i, :], V_bias_mV)
        v_QP_mV = bin_y_over_x(I_QP_nA_over, V_bias_mV_over, I_bias_nA)

        interp_V_QP_mV: JInterpolator = jnp_interp_y_of_x(
            x=I_bias_nA,
            y=v_QP_mV,
            dydx=G_N,
        )

        v_mv_mV = sim_V_RSJ_mV_jax(
            I_bias_nA=I_bias_nA,
            A_bias_nA=a_bias_nA,
            V_QP_mV=interp_V_QP_mV,
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


def sim_V_RSJ_mV_jax(
    I_bias_nA: NDArray64,
    A_bias_nA: NDArray64,
    V_QP_mV: JInterpolator,
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
    dphi_pmV = h_e_pVs

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
            V_mV_vec = V_QP_mV(I_qp_nA_vec)

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
