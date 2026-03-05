from typing import Sequence

from ..utilities.types import NDArray64
from .rcsj_models.rcsj import get_I_rcsj_nA as get_I_rcsj_model_nA
from .rcsj_models.rcstj import get_I_rcstj_nA as get_I_rcstj_model_nA
from .rcsj_models.rsj import get_I_rsj_nA as get_I_rsj_model_nA
from .rcsj_models.rstj import get_I_rstj_nA as get_I_rstj_model_nA
from .rcsj_models.rstj_slow import (
    get_I_rstj_slow_nA as get_I_rstj_slow_model_nA,
)


def get_I_rcsj_nA(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
    I_sw_nA: float | Sequence[float],
    A_mV: float | Sequence[float] = 0.5,
    C_pF: float = 0.0,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_periods: int = 30,
    burn_fraction: float = 0.3,
    seed: int = 1,
    G_min_muS: float = 1e-6,
    GN_G0: float = 1.0,
    n_realizations: int = 1,
    include_shunt: bool = True,
    model: str = "auto",
) -> NDArray64:
    """
    Unified dispatcher for RSJ/RSTJ/RCSJ/RCSTJ.

    Auto-selection rules (`model="auto"`):
    - `|C_pF| <= C_tol_pF`, `|T_K| <= T_tol_K` -> RSJ
    - `|C_pF| <= C_tol_pF`, `|T_K| >  T_tol_K` -> RSTJ
    - `|C_pF| >  C_tol_pF`, `|T_K| <= T_tol_K` -> RCSJ
    - `|C_pF| >  C_tol_pF`, `|T_K| >  T_tol_K` -> RCSTJ

    Set `model` explicitly to one of:
    `"rsj"`, `"rstj"`, `"rcsj"`, `"rcstj"`.
    """

    model_key = model.strip().lower()

    C_tol_pF: float = 1e-12
    T_tol_K: float = 1e-12

    if model_key == "auto":
        has_C = abs(C_pF) > C_tol_pF
        has_T = abs(T_K) > T_tol_K
        if not has_C and not has_T:
            model_key = "rsj"
        elif not has_C and has_T:
            model_key = "rstj"
        elif has_C and not has_T:
            model_key = "rcsj"
        else:
            model_key = "rcstj"

    if model_key == "rsj":
        return get_I_rsj_model_nA(
            V_mV=V_mV,
            I_qp_nA=I_qp_nA,
            I_sw_nA=I_sw_nA,
            A_mV=A_mV,
            nu_GHz=nu_GHz,
            n_periods=n_periods,
            burn_fraction=burn_fraction,
        )

    if model_key == "rstj":
        return get_I_rstj_model_nA(
            V_mV=V_mV,
            I_qp_nA=I_qp_nA,
            I_sw_nA=I_sw_nA,
            A_mV=A_mV,
            T_K=T_K,
            nu_GHz=nu_GHz,
            n_periods=n_periods,
            burn_fraction=burn_fraction,
            seed=seed,
            GN_G0=GN_G0,
            n_realizations=n_realizations,
            include_shunt=include_shunt,
        )

    if model_key == "rcsj":
        return get_I_rcsj_model_nA(
            V_mV=V_mV,
            I_qp_nA=I_qp_nA,
            I_sw_nA=I_sw_nA,
            C_pF=C_pF,
            A_mV=A_mV,
            nu_GHz=nu_GHz,
            n_periods=n_periods,
            burn_fraction=burn_fraction,
        )

    if model_key == "rcstj":
        return get_I_rcstj_model_nA(
            V_mV=V_mV,
            I_qp_nA=I_qp_nA,
            I_sw_nA=I_sw_nA,
            T_K=T_K,
            C_pF=C_pF,
            A_mV=A_mV,
            nu_GHz=nu_GHz,
            n_periods=n_periods,
            burn_fraction=burn_fraction,
            seed=seed,
            G_min_muS=G_min_muS,
        )

    raise ValueError(
        "Unknown model. Use 'auto', 'rsj', 'rstj', 'rcsj', or 'rcstj'.",
    )


def get_I_rstj_slow_nA(
    V_mV: NDArray64,
    I_qp_nA: NDArray64,
    I_sw_nA: float | Sequence[float],
    T_eff_K: float = 0.15,
    GN_G0: float = 1.0,
    ramp_rate_nA_per_s: float = 1.0,
    attempt_rate_GHz: float = 10.0,
    sigma_I_nA: float = 0.0,
    dI_max_nA: float = 0.25,
    min_i_grid_points: int = 4001,
    include_shunt: bool = True,
    upsample: int = 100,
    fill_nan: bool = True,
) -> NDArray64:
    """Slow-ramp thermal-switching I(V) without microwave drive."""
    return get_I_rstj_slow_model_nA(
        V_mV=V_mV,
        I_qp_nA=I_qp_nA,
        I_sw_nA=I_sw_nA,
        T_eff_K=T_eff_K,
        GN_G0=GN_G0,
        ramp_rate_nA_per_s=ramp_rate_nA_per_s,
        attempt_rate_GHz=attempt_rate_GHz,
        sigma_I_nA=sigma_I_nA,
        dI_max_nA=dI_max_nA,
        min_i_grid_points=min_i_grid_points,
        include_shunt=include_shunt,
        upsample=upsample,
        fill_nan=fill_nan,
    )
