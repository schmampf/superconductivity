"""Legacy PAT wrapper around the new registry-driven optimizer."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from ..fit_model import SolutionDict, fit_model
from ..models import ParameterSpec
from ..models._common import make_pat_parameters

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

DEFAULT_PARAMETERS: tuple[ParameterSpec, ...] = tuple(
    replace(parameter) for parameter in make_pat_parameters()
)
PARAMETER_NAMES: tuple[str, ...] = tuple(
    parameter.name for parameter in DEFAULT_PARAMETERS
)


def _resolve_model(model: str) -> str:
    return _LEGACY_MODEL_MAP.get(model, model)


def fit_pat(
    V_mV,
    I_nA,
    *,
    parameters: Optional[Sequence[ParameterSpec]] = None,
    weights=None,
    maxfev: Optional[int] = None,
    E_mV=None,
    model: str = "pat",
) -> SolutionDict:
    if E_mV is not None:
        raise ValueError("E_mV is fixed by the new PAT model registry.")

    return fit_model(
        V_mV,
        I_nA,
        model=_resolve_model(model),
        parameters=parameters,
        weights=weights,
        maxfev=maxfev,
    )
