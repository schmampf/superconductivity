import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..utilities.constants import h_e_pVs
from ..utilities.functions import bin_y_over_x, oversample
from ..utilities.functions_jax import JInterpolator, get_dydx, jinterp_y_of_x
from ..utilities.types import NDArray64


def get_I_rsj_nA(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
    I_sw_nA: Sequence[float],
    A_mV: float = 0.5,
    nu_GHz: float = 10.0,
    n_periods: int = 10,
    n_burns: int = 3,
):

    dt_ps, Nt = suggest_dt_Nt(
        V_mV=V_mV,
        A_mV=A_mV,
        nu_GHz=nu_GHz,
        n_periods=n_periods,
    )

    # Accept scalar or sequence; RSJ simulator expects a scalar switching/critical current.
    I_sw_arr: NDArray64 = np.atleast_1d(np.asarray(I_sw_nA, dtype=np.float64))
    if I_sw_arr.size != 1:
        raise ValueError("get_I_rsj_nA currently supports only a single I_sw_nA value.")
    I_sw_scalar_nA: float = float(I_sw_arr[0])

    I_bias_nA: NDArray64 = np.linspace(
        np.min(I_qp_nA),
        np.max(I_qp_nA),
        (V_mV.shape[0] - 1) * 3 + 1,
    )

    I_qp_nA_over, V_mV_over = oversample(
        x=I_qp_nA,
        y=V_mV,
    )

    V_qp_mV: NDArray64 = bin_y_over_x(
        x=I_qp_nA_over,
        y=V_mV_over,
        x_bins=I_bias_nA,
    )

    dVdI_muS_raw = get_dydx(
        x=I_bias_nA,
        y=V_qp_mV,
    )
    # `get_dydx` may return a vector (e.g. left/right slopes). We need a scalar slope
    # for the extrapolation in `jinterp_y_of_x` to avoid broadcasting to shape (2,).
    dVdI_muS: float = np.mean(np.asarray(dVdI_muS_raw, dtype=np.float64))

    V_iqp_mV: JInterpolator = jinterp_y_of_x(
        x=I_bias_nA,
        y=V_qp_mV,
        dydx=dVdI_muS,
    )

    I_bias_nA_jnp = jnp.array(I_bias_nA)
    V_rsj_mV = simulate_rsj_with_pat_vac_vmap(
        I_nA=I_bias_nA_jnp,
        A_mV=A_mV,
        I_sw_nA=I_sw_scalar_nA,
        V_iqp_mV=V_iqp_mV,
        nu_GHz=nu_GHz,
        dt_ps=dt_ps,
        Nt=Nt,
    )
    burn_index = int(n_burns / n_periods * Nt)
    V_rsj_mV = np.mean(V_rsj_mV[:, burn_index:], axis=1)

    I_bias_nA_over, V_rsj_mV_over = oversample(
        x=I_bias_nA,
        y=V_rsj_mV,
    )

    I_rsj_nA = bin_y_over_x(
        x=V_rsj_mV_over,
        y=I_bias_nA_over,
        x_bins=V_mV,
    )

    return I_rsj_nA


import math


def suggest_dt_Nt(
    *,
    V_mV: NDArray64,
    A_mV: float,
    nu_GHz: float,
    npp_mw: int = 200,
    npp_j: int = 50,
    n_periods: int = 10,
    safety: float = 1.0,
) -> tuple[float, int]:
    """
    Suggest dt (ps) and Nt for RSJ/RCSJ simulations.

    Constraints
    ----------
    - Resolve microwave: dt <= T_mw / npp_mw, with T_mw = 1/nu_max
    - Resolve Josephson rotation at max instantaneous voltage:
        f_J = (2e/h) V ≈ 483.5979 GHz/mV * V_mV
        dt <= T_J / npp_j

      Use V_inst_max = V_max_mV + A_max_mV as a conservative bound.

    Nt choice
    ---------
    Choose Nt so the simulation spans `n_periods` microwave periods and
    uses an integer number of points per period.

    Returns
    -------
    dt_ps, Nt
    """

    # microwave period in ps
    T_mw_ps = 1000.0 / nu_GHz  # since 1 GHz -> 1000 ps period
    dt_mw_ps = T_mw_ps / npp_mw

    V_max_mV = np.max(V_mV)

    # Josephson frequency in GHz/mV
    V_inst_max_mV = V_max_mV + A_mV
    if V_inst_max_mV == 0:
        dt_j_ps = float("inf")
    else:
        fJ_GHz = 483.5979 * V_inst_max_mV
        Tj_ps = 1000.0 / fJ_GHz
        dt_j_ps = Tj_ps / npp_j

    dt_ps = min(dt_mw_ps, dt_j_ps) / safety

    # enforce integer samples per mw period
    npp = max(4, int(round(T_mw_ps / dt_ps)))
    dt_ps = T_mw_ps / npp

    Nt = int(n_periods * npp)
    return float(dt_ps), int(Nt)


@functools.partial(jax.jit, static_argnames=("V_iqp_mV", "Nt"))
def simulate_rsj_with_pat_vac(
    I_nA: float,
    A_mV: float,
    I_sw_nA: float,
    V_iqp_mV: JInterpolator,
    nu_GHz: float,
    dt_ps: float,
    Nt: int,
):
    two_mpi = 2e-3 * np.pi
    w_THz = nu_GHz * two_mpi

    a = 2.0 * A_mV / (h_e_pVs * nu_GHz)
    I_sw_nA = jnp.asarray(I_sw_nA, dtype=jnp.float64).reshape(())

    def step(carry, n):
        phi = carry
        t_ps = n * dt_ps
        sinwt = jnp.sin(w_THz * t_ps)
        coswt = jnp.cos(w_THz * t_ps)

        phi_tot = phi + a * sinwt

        # Total voltage across junction from inverse quasiparticle branch
        I_for_qp = I_nA - I_sw_nA * jnp.sin(phi_tot)
        V_qp_mV = V_iqp_mV(I_for_qp)

        # Slow voltage (subtract the imposed RF voltage component)
        V_mV = V_qp_mV - A_mV * coswt

        # Phase update uses slow voltage
        phi_next = phi + V_mV * two_mpi * dt_ps / h_e_pVs
        # Keep phi bounded to avoid float blow-up at high V
        phi_next = jnp.mod(phi_next + jnp.pi, 2.0 * jnp.pi) - jnp.pi

        return phi_next, (V_mV, V_qp_mV)

    phi0 = 0.0
    _, (V_mV, V_qp_mV) = jax.lax.scan(step, phi0, jnp.arange(Nt))
    # V_mV = jnp.sum(V_mV)
    return V_mV


@functools.partial(jax.jit, static_argnames=("V_iqp_mV", "Nt"))
def simulate_rsj_with_pat_vac_vmap(
    I_nA: jnp.ndarray,
    A_mV: float,
    I_sw_nA: float,
    V_iqp_mV: JInterpolator,
    nu_GHz: float,
    dt_ps: float,
    Nt: int,
):
    """Vectorized wrapper over `simulate_rsj_with_pat_vac`.

    Parameters
    ----------
    I_nA:
        1D array of bias currents (nA). The function is vmapped over this axis.

    Returns
    -------
    V_mV, Vtot_mV:
        Arrays with shape (I_nA.size, Nt).
    """

    def _one(I_scalar_nA: float):
        return simulate_rsj_with_pat_vac(
            I_nA=I_scalar_nA,
            A_mV=A_mV,
            I_sw_nA=I_sw_nA,
            V_iqp_mV=V_iqp_mV,
            nu_GHz=nu_GHz,
            dt_ps=dt_ps,
            Nt=Nt,
        )

    V_mV = jax.vmap(_one, in_axes=0, out_axes=0)(I_nA)
    return V_mV
    return V_mV
    return V_mV
