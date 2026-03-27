from __future__ import annotations

from ._bcs import sin_integral_np
from ._common import BCS_SIN_INT_HTML, DEFAULT_E_MV, make_bcs_parameters, make_model_info
from .registry import ModelSpec


def _model(
    V_mV,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
):
    return sin_integral_np(V_mV, DEFAULT_E_MV, GN_G0, T_K, Delta_meV, gamma_meV)


MODEL = ModelSpec(
    key="bcs_sin_int",
    label="BCS SIN integral",
    function=_model,
    parameters=make_bcs_parameters(),
    info=make_model_info(
        backend="numpy",
        junction="SIN",
        kernel="integral",
        E_mV=DEFAULT_E_MV,
    ),
    html=BCS_SIN_INT_HTML,
)
