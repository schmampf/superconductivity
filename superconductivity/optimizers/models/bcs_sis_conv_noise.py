from __future__ import annotations

from ._bcs_jax import sis_convolution_jax
from ._common import (
    BCS_SIS_CONV_HTML,
    DEFAULT_E_MV,
    NOISE_SUFFIX_HTML,
    make_bcs_noise_parameters,
    make_model_info,
)
from ._noise import apply_voltage_noise
from .registry import ModelSpec


def _model(
    V_mV,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    sigma_V_mV: float,
):
    base_current = sis_convolution_jax(
        V_mV,
        DEFAULT_E_MV,
        GN_G0,
        T_K,
        Delta_meV,
        gamma_meV,
    )
    return apply_voltage_noise(V_mV, base_current, sigma_V_mV)


MODEL = ModelSpec(
    key="bcs_sis_conv_noise",
    label="BCS SIS convolution + noise",
    function=_model,
    parameters=make_bcs_noise_parameters(),
    info=make_model_info(
        backend="jax+numpy",
        junction="SIS",
        kernel="convolution+noise",
        E_mV=DEFAULT_E_MV,
    ),
    html=BCS_SIS_CONV_HTML + NOISE_SUFFIX_HTML,
)
