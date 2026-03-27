from __future__ import annotations

from ._common import (
    BCS_SIS_INT_HTML,
    DEFAULT_E_MV,
    PAT_N_MAX,
    PAT_SUFFIX_HTML,
    make_model_info,
    make_pat_parameters,
)
from ._bcs_jax import sis_integral_jax
from ._pat import get_I_pat_nA
from .registry import ModelSpec


def _model(
    V_mV,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    A_mV: float,
    nu_GHz: float,
):
    base_current = sis_integral_jax(
        V_mV,
        DEFAULT_E_MV,
        GN_G0,
        T_K,
        Delta_meV,
        gamma_meV,
    )
    if A_mV == 0.0:
        return base_current
    return get_I_pat_nA(
        V_mV,
        base_current,
        A_mV,
        nu_GHz=nu_GHz,
        n_max=PAT_N_MAX,
    )


MODEL = ModelSpec(
    key="pat_sis_int_jax",
    label="PAT SIS integral (JAX)",
    function=_model,
    parameters=make_pat_parameters(),
    info=make_model_info(
        backend="jax",
        junction="SIS",
        kernel="integral+pat",
        E_mV=DEFAULT_E_MV,
        n_max=PAT_N_MAX,
    ),
    html=BCS_SIS_INT_HTML + PAT_SUFFIX_HTML,
)
