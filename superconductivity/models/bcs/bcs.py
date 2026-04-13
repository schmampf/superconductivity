"""Top-level BCS model composition helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...utilities.types import NDArray64
from ..basics.noise import apply_voltage_noise, make_bias_support_grid
from .backend import (
    DEFAULT_E_MV,
    PAT_N_MAX,
    Backend,
    Kernel,
)

_NOISE_OVERSAMPLE = 64
_ModelFunction = Callable[[NDArray64, NDArray64, float, float, float, float], NDArray64]

def get_Ibcs_nA(
    V_mV: NDArray64,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    nu_GHz: float = 0.0,
    A_mV: NDArray64 | float = 0.0,
    sigmaV_mV: float = 0.0,
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
        Model current in nA. The return shape is ``(Nv,)`` for scalar
        ``A_mV`` and ``(Na, Nv)`` for array ``A_mV``.
    """
    V_requested = _validate_voltage_axis(V_mV)
    base_function = _resolve_base_function(kernel=kernel, backend=backend)
    GN_G0_value = _validate_finite_scalar(GN_G0, "GN_G0")
    T_K_value = _validate_finite_scalar(T_K, "T_K")
    Delta_value = _validate_finite_scalar(Delta_meV, "Delta_meV")
    gamma_value = _validate_finite_scalar(gamma_meV, "gamma_meV")
    sigma_value = _validate_nonnegative_scalar(sigmaV_mV, "sigmaV_mV")
    A_values, scalar_amplitude = _normalize_amplitudes(A_mV)

    V_evaluate = (
        make_bias_support_grid(V_requested, sigma_value)
        if sigma_value > 0.0
        else V_requested
    )
    current = np.asarray(
        base_function(
            V_evaluate,
            DEFAULT_E_MV,
            GN_G0_value,
            T_K_value,
            Delta_value,
            gamma_value,
        ),
        dtype=np.float64,
    )

    if scalar_amplitude:
        amplitude = float(A_values[0])
        if amplitude != 0.0:
            nu_value = _validate_positive_scalar(nu_GHz, "nu_GHz")
            from .backend.pat import get_I_pat_nA

            current_out: NDArray64 = np.asarray(
                get_I_pat_nA(
                    V_evaluate,
                    current,
                    amplitude,
                    nu_GHz=nu_value,
                    n_max=PAT_N_MAX,
                ),
                dtype=np.float64,
            )
        else:
            current_out = current
    else:
        if np.all(A_values == 0.0):
            current_out = np.repeat(current[None, :], A_values.size, axis=0)
        else:
            nu_value = _validate_positive_scalar(nu_GHz, "nu_GHz")
            from .backend.pat import get_I_pat_nA

            current_out = np.asarray(
                get_I_pat_nA(
                    V_evaluate,
                    current,
                    A_values,
                    nu_GHz=nu_value,
                    n_max=PAT_N_MAX,
                ),
                dtype=np.float64,
            )

    if sigma_value == 0.0:
        return current_out
    if current_out.ndim == 1:
        return apply_voltage_noise(
            V_evaluate,
            current_out,
            sigma_value,
            _NOISE_OVERSAMPLE,
            V_out_mV=V_requested,
        )
    return np.asarray(
        [
            apply_voltage_noise(
                V_evaluate,
                row,
                sigma_value,
                _NOISE_OVERSAMPLE,
                V_out_mV=V_requested,
            )
            for row in current_out
        ],
        dtype=np.float64,
    )


def _resolve_base_function(*, kernel: Kernel, backend: Backend) -> _ModelFunction:
    if backend == "np":
        from .backend.np import convolution_np, integral_np

        return integral_np if kernel == "int" else convolution_np
    if backend == "jax":
        from .backend.jax import convolution_jax, integral_jax

        return integral_jax if kernel == "int" else convolution_jax
    raise ValueError("backend must be 'np' or 'jax' and kernel must be 'int' or 'conv'.")


def _normalize_amplitudes(A_mV: NDArray64 | float) -> tuple[NDArray64, bool]:
    values = np.asarray(A_mV, dtype=np.float64)
    if values.ndim == 0:
        if not np.isfinite(values):
            raise ValueError("A_mV must be finite.")
        return np.asarray([float(values)], dtype=np.float64), True
    if values.ndim != 1:
        raise ValueError("A_mV must be a scalar or a 1D array.")
    if values.size == 0:
        raise ValueError("A_mV must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("A_mV must be finite.")
    return np.asarray(values, dtype=np.float64), False


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


__all__ = [
    "Backend",
    "Kernel",
    "get_Ibcs_nA",
]
