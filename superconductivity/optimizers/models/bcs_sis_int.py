from __future__ import annotations

from ._common import BCS_SIS_INT_HTML, DEFAULT_E_MV, make_bcs_parameters, make_model_info
from ._bcs import sis_integral_np
from .registry import ModelSpec


def _model(
    V_mV,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
):
    return sis_integral_np(V_mV, DEFAULT_E_MV, GN_G0, T_K, Delta_meV, gamma_meV)


MODEL = ModelSpec(
    key="bcs_sis_int",
    label="BCS SIS integral",
    function=_model,
    parameters=make_bcs_parameters(),
    info=make_model_info(
        backend="numpy",
        junction="SIS",
        kernel="integral",
        E_mV=DEFAULT_E_MV,
    ),
    html=BCS_SIS_INT_HTML,
)
