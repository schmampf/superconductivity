from typing import Sequence

from ..utilities.types import NDArray64
from .rcsj_models.rcsj import get_I_rcsj_nA as get_I_rcsj_model_nA
from .rcsj_models.rcstj import get_I_rcstj_nA as get_I_rcstj_model_nA
from .rcsj_models.rsj import get_I_rsj_nA as get_I_rsj_model_nA
from .rcsj_models.rstj import get_I_rstj_nA as get_I_rstj_model_nA


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
