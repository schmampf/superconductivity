from __future__ import annotations

import numpy as np

from ...utilities.types import ModelType
from ..models import get_model_spec

_LEGACY_MODEL_MAP = {
    "bcs": "bcs_sis_int_jax",
    "dynes": "bcs_sis_int_jax",
    "integral": "bcs_sis_int_jax",
    "conv": "bcs_sis_conv_jax",
    "conv_dynes": "bcs_sis_conv_jax",
    "pat": "pat_sis_int_jax",
    "bcs+pat": "pat_sis_int_jax",
    "dynes+pat": "pat_sis_int_jax",
    "integral_pat": "pat_sis_int_jax",
    "conv_pat": "pat_sis_conv_jax",
    "conv+pat": "pat_sis_conv_jax",
}
_LEGACY_PARAMETER_ORDER = (
    "GN_G0",
    "T_K",
    "Delta_meV",
    "gamma_meV",
    "A_mV",
    "nu_GHz",
)


def _resolve_model(model: str) -> str:
    return _LEGACY_MODEL_MAP.get(model, model)


def _parameter_mask(model_key: str) -> np.ndarray:
    names = {parameter.name for parameter in get_model_spec(model_key).parameters}
    return np.array(
        [name in names for name in _LEGACY_PARAMETER_ORDER],
        dtype=bool,
    )


def get_model(
    model: str = "pat",
    E_mV=None,
    N=None,
) -> ModelType:
    if E_mV is not None:
        raise ValueError("E_mV is fixed by the new optimizer model registry.")
    if N is not None:
        raise ValueError("N is fixed by the new optimizer model registry.")

    model_key = _resolve_model(model)
    model_spec = get_model_spec(model_key)
    return model_spec.function, _parameter_mask(model_key)
