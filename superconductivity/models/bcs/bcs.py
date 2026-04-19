"""Top-level BCS model composition helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from ...utilities.constants import G0_muS
from ...utilities.types import NDArray64
from ..basics.noise import evaluate_with_voltage_noise
from .backend import Backend, E0_meV, Kernel, Nmax_

_NOISE_OVERSAMPLE = 64
_G0_uS = float(G0_muS)
_ModelFunction: TypeAlias = Callable[
    [NDArray64, NDArray64, float, float, float, float, float, float], NDArray64
]
_SWEEP_PARAM_ORDER: tuple[str, ...] = (
    "GN_G0",
    "T_K",
    "Delta_meV",
    "gamma_meV",
    "nu_GHz",
    "A_mV",
    "sigmaV_mV",
)


def get_Ibcs_nA(
    V_mV: NDArray64,
    GN_G0: NDArray64 | float,
    T_K: NDArray64 | float,
    Delta_meV: NDArray64 | float,
    gamma_meV: NDArray64 | float,
    nu_GHz: NDArray64 | float = 0.0,
    A_mV: NDArray64 | float = 0.0,
    sigmaV_mV: NDArray64 | float = 0.0,
    *,
    backend: Backend = "jax",
    kernel: Kernel = "conv",
) -> NDArray64:
    """Evaluate the public BCS current model.

    The model evaluates the selected core BCS kernel first, then applies PAT
    and voltage-noise post-processing if requested.

    Parameters
    ----------
    V_mV
        Requested bias-voltage grid in mV. Must be 1D, finite, and strictly
        increasing.
    GN_G0
        Normal-state conductance in units of ``G0``.
    T_K
        Temperature in kelvin.
    Delta_meV
        Zero-temperature superconducting gap in meV.
    gamma_meV
        Dynes broadening in meV.
    nu_GHz
        Drive frequency in GHz.
    A_mV
        PAT drive amplitude in mV. A scalar returns one current trace. A 1D
        array returns one trace per amplitude.
    sigmaV_mV
        RMS voltage noise in mV.
    backend
        Core BCS backend, ``"np"`` or ``"jax"``.
    kernel
        Core BCS kernel, ``"int"`` or ``"conv"``.

    Returns
    -------
    NDArray64
        Model current in nA. For all-scalar inputs, shape is ``(Nv,)``.
        If any sweep input is 1D, output shape is
        ``(*sweep_shape, Nv)`` with sweep axes ordered by parameter
        signature.
    """
    V_requested = _validate_voltage_axis(V_mV)
    base_function = _resolve_base_function(kernel=kernel, backend=backend)
    GN_values, GN_scalar = _normalize_sweep_values(
        GN_G0,
        "GN_G0",
        validator=_validate_finite_scalar,
    )
    T_values, T_scalar = _normalize_sweep_values(
        T_K,
        "T_K",
        validator=_validate_finite_scalar,
    )
    Delta_values, Delta_scalar = _normalize_sweep_values(
        Delta_meV,
        "Delta_meV",
        validator=_validate_finite_scalar,
    )
    gamma_values, gamma_scalar = _normalize_sweep_values(
        gamma_meV,
        "gamma_meV",
        validator=_validate_finite_scalar,
    )
    nu_values, nu_scalar = _normalize_sweep_values(
        nu_GHz,
        "nu_GHz",
        validator=_validate_finite_scalar,
    )
    A_values, A_scalar = _normalize_sweep_values(
        A_mV,
        "A_mV",
        validator=_validate_finite_scalar,
    )
    sigma_values, sigma_scalar = _normalize_sweep_values(
        sigmaV_mV,
        "sigmaV_mV",
        validator=_validate_nonnegative_scalar,
    )

    sweep_vectors: dict[str, NDArray64] = {
        "GN_G0": GN_values,
        "T_K": T_values,
        "Delta_meV": Delta_values,
        "gamma_meV": gamma_values,
        "nu_GHz": nu_values,
        "A_mV": A_values,
        "sigmaV_mV": sigma_values,
    }
    scalar_flags: dict[str, bool] = {
        "GN_G0": GN_scalar,
        "T_K": T_scalar,
        "Delta_meV": Delta_scalar,
        "gamma_meV": gamma_scalar,
        "nu_GHz": nu_scalar,
        "A_mV": A_scalar,
        "sigmaV_mV": sigma_scalar,
    }
    sweep_names = [name for name in _SWEEP_PARAM_ORDER if not scalar_flags[name]]
    sweep_shape = tuple(sweep_vectors[name].size for name in sweep_names)

    base_cache: dict[tuple[float, float, float, float], NDArray64] = {}
    pat_cache: dict[tuple[float, float, float, float, float, float], NDArray64] = {}
    noise_cache: dict[
        tuple[float, float, float, float, float, float, float], NDArray64
    ] = {}

    if not sweep_names:
        params = {
            "GN_G0": float(GN_values[0]),
            "T_K": float(T_values[0]),
            "Delta_meV": float(Delta_values[0]),
            "gamma_meV": float(gamma_values[0]),
            "nu_GHz": float(nu_values[0]),
            "A_mV": float(A_values[0]),
            "sigmaV_mV": float(sigma_values[0]),
        }
        non_gn = _evaluate_non_gn_current(
            V_requested=V_requested,
            base_function=base_function,
            T_K_value=params["T_K"],
            Delta_value=params["Delta_meV"],
            gamma_value=params["gamma_meV"],
            nu_value=params["nu_GHz"],
            A_value=params["A_mV"],
            sigma_value=params["sigmaV_mV"],
            base_cache=base_cache,
            pat_cache=pat_cache,
            noise_cache=noise_cache,
        )
        return np.asarray(non_gn * (params["GN_G0"] * _G0_uS), dtype=np.float64)

    output = np.empty(sweep_shape + (V_requested.size,), dtype=np.float64)
    for sweep_index in np.ndindex(*sweep_shape):
        values: dict[str, float] = {}
        sweep_pos = 0
        for name in _SWEEP_PARAM_ORDER:
            if scalar_flags[name]:
                values[name] = float(sweep_vectors[name][0])
            else:
                values[name] = float(sweep_vectors[name][sweep_index[sweep_pos]])
                sweep_pos += 1
        non_gn = _evaluate_non_gn_current(
            V_requested=V_requested,
            base_function=base_function,
            T_K_value=values["T_K"],
            Delta_value=values["Delta_meV"],
            gamma_value=values["gamma_meV"],
            nu_value=values["nu_GHz"],
            A_value=values["A_mV"],
            sigma_value=values["sigmaV_mV"],
            base_cache=base_cache,
            pat_cache=pat_cache,
            noise_cache=noise_cache,
        )
        output[sweep_index] = np.asarray(
            non_gn * (values["GN_G0"] * _G0_uS),
            dtype=np.float64,
        )
    return output


def _evaluate_non_gn_current(
    *,
    V_requested: NDArray64,
    base_function: _ModelFunction,
    T_K_value: float,
    Delta_value: float,
    gamma_value: float,
    nu_value: float,
    A_value: float,
    sigma_value: float,
    base_cache: dict[tuple[float, float, float, float], NDArray64],
    pat_cache: dict[tuple[float, float, float, float, float, float], NDArray64],
    noise_cache: dict[
        tuple[float, float, float, float, float, float, float], NDArray64
    ],
) -> NDArray64:
    """Evaluate current with GN fixed to 1 for one parameter combination."""
    base_key = (
        T_K_value,
        Delta_value,
        gamma_value,
        sigma_value,
    )
    pat_key = (
        T_K_value,
        Delta_value,
        gamma_value,
        sigma_value,
        A_value,
        nu_value,
    )

    def evaluate_pat_on_grid(V_evaluate: NDArray64) -> NDArray64:
        if base_key not in base_cache:
            base_cache[base_key] = np.asarray(
                base_function(
                    V_evaluate,
                    E0_meV,
                    T_K_value,
                    T_K_value,
                    Delta_value,
                    Delta_value,
                    gamma_value,
                    gamma_value,
                ),
                dtype=np.float64,
            )
        base_current = base_cache[base_key]

        if pat_key not in pat_cache:
            if A_value == 0.0:
                pat_cache[pat_key] = base_current
            else:
                nu_positive = _validate_positive_scalar(nu_value, "nu_GHz")
                from .backend.pat import pat_kernel

                pat_cache[pat_key] = np.asarray(
                    pat_kernel(
                        V_evaluate,
                        base_current,
                        A_value,
                        nu_GHz=nu_positive,
                        n_max=Nmax_,
                    ),
                    dtype=np.float64,
                )
        return pat_cache[pat_key]

    noise_key = (
        T_K_value,
        Delta_value,
        gamma_value,
        sigma_value,
        A_value,
        nu_value,
        float(V_requested.size),
    )
    if noise_key not in noise_cache:
        noise_cache[noise_key] = evaluate_with_voltage_noise(
            V_requested,
            evaluate_pat_on_grid,
            sigma_value,
            _NOISE_OVERSAMPLE,
        )
    return noise_cache[noise_key]


def _resolve_base_function(*, kernel: Kernel, backend: Backend) -> _ModelFunction:
    if backend == "np":
        from .backend.np import convolution_np, integral_np

        return integral_np if kernel == "int" else convolution_np
    if backend == "jax":
        from .backend.jnp import convolution_jnp, integral_jnp

        return integral_jnp if kernel == "int" else convolution_jnp
    raise ValueError(
        "backend must be 'np' or 'jax' and kernel must be 'int' or 'conv'."
    )


def _normalize_sweep_values(
    value: NDArray64 | float,
    name: str,
    *,
    validator: Callable[[float, str], float],
) -> tuple[NDArray64, bool]:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        return np.asarray([validator(float(values), name)], dtype=np.float64), True
    if values.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array.")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    validated = np.asarray(
        [validator(float(item), name) for item in values],
        dtype=np.float64,
    )
    return validated, False


def _validate_voltage_axis(V_mV: NDArray64) -> NDArray64:
    values = np.asarray(V_mV, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("V_mV must be 1D.")
    if values.size < 2:
        raise ValueError("V_mV must contain at least two points.")
    if not np.all(np.isfinite(values)):
        raise ValueError("V_mV must be finite.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("V_mV must be strictly increasing.")
    return values


def _validate_finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _validate_nonnegative_scalar(value: float, name: str) -> float:
    scalar = _validate_finite_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return scalar


def _validate_positive_scalar(value: float, name: str) -> float:
    scalar = _validate_finite_scalar(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return scalar


from .sim import sim_bcs

__all__ = [
    "Backend",
    "Kernel",
    "get_Ibcs_nA",
    "sim_bcs",
]
