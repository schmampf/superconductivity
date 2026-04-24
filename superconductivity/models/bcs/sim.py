"""BCS simulation dataset entrypoint."""

from __future__ import annotations

import numpy as np

from ...utilities.meta import (
    AxisSpec,
    ParamSpec,
    TransportDatasetSpec,
    axis,
    dataset,
    param,
)
from ...utilities.types import NDArray64
from .backend import Backend, Kernel
from .bcs import (
    _SWEEP_PARAM_ORDER,
    _normalize_sweep_values,
    _validate_finite_scalar,
    _validate_nonnegative_scalar,
    _validate_voltage_axis,
    get_Ibcs_nA,
)


def sim_bcs(
    V_mV: AxisSpec,
    GN_G0: ParamSpec | AxisSpec,
    T_K: ParamSpec | AxisSpec,
    Delta_meV: ParamSpec | AxisSpec,
    gamma_meV: ParamSpec | AxisSpec,
    nu_GHz: ParamSpec | AxisSpec | None = None,
    A_mV: ParamSpec | AxisSpec | None = None,
    sigmaV_mV: ParamSpec | AxisSpec | None = None,
    *,
    backend: Backend = "jax",
    kernel: Kernel = "conv",
):
    """Simulate one BCS grid and return a gridded dataset."""
    if not isinstance(V_mV, AxisSpec):
        raise TypeError("V_mV must be an AxisSpec.")
    if nu_GHz is None:
        nu_GHz = param("nu_GHz", 0.0)
    if A_mV is None:
        A_mV = param("A_mV", 0.0)
    if sigmaV_mV is None:
        sigmaV_mV = param("sigmaV_mV", 0.0)

    V_requested = _validate_voltage_axis(np.asarray(V_mV.values, dtype=np.float64))
    GN_values, GN_scalar = _normalize_sweep_values(
        _spec_values(GN_G0, "GN_G0"),
        "GN_G0",
        validator=_validate_finite_scalar,
    )
    T_values, T_scalar = _normalize_sweep_values(
        _spec_values(T_K, "T_K"),
        "T_K",
        validator=_validate_finite_scalar,
    )
    Delta_values, Delta_scalar = _normalize_sweep_values(
        _spec_values(Delta_meV, "Delta_meV"),
        "Delta_meV",
        validator=_validate_finite_scalar,
    )
    gamma_values, gamma_scalar = _normalize_sweep_values(
        _spec_values(gamma_meV, "gamma_meV"),
        "gamma_meV",
        validator=_validate_finite_scalar,
    )
    nu_values, nu_scalar = _normalize_sweep_values(
        _spec_values(nu_GHz, "nu_GHz"),
        "nu_GHz",
        validator=_validate_finite_scalar,
    )
    A_values, A_scalar = _normalize_sweep_values(
        _spec_values(A_mV, "A_mV"),
        "A_mV",
        validator=_validate_finite_scalar,
    )
    sigma_values, sigma_scalar = _normalize_sweep_values(
        _spec_values(sigmaV_mV, "sigmaV_mV"),
        "sigmaV_mV",
        validator=_validate_nonnegative_scalar,
    )

    I_nA = np.asarray(
        get_Ibcs_nA(
            V_mV=V_requested,
            GN_G0=GN_values if not GN_scalar else float(GN_values[0]),
            T_K=T_values if not T_scalar else float(T_values[0]),
            Delta_meV=Delta_values if not Delta_scalar else float(Delta_values[0]),
            gamma_meV=gamma_values if not gamma_scalar else float(gamma_values[0]),
            nu_GHz=nu_values if not nu_scalar else float(nu_values[0]),
            A_mV=A_values if not A_scalar else float(A_values[0]),
            sigmaV_mV=sigma_values if not sigma_scalar else float(sigma_values[0]),
            backend=backend,
            kernel=kernel,
        ),
        dtype=np.float64,
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
    v_order = len(sweep_names)

    axis_entries: dict[str, object] = {
        "V_mV": axis("V_mV", values=V_requested, order=v_order),
    }
    for order, name in enumerate(sweep_names):
        axis_entries[name] = axis(name, values=sweep_vectors[name], order=order)

    data_entries: dict[str, object] = {
        "I_nA": I_nA,
    }

    param_entries: dict[str, object] = {}
    for name in _SWEEP_PARAM_ORDER:
        if not scalar_flags[name]:
            continue
        value = float(sweep_vectors[name][0])
        param_entries[name] = param(name, value)

    ds = dataset(
        **data_entries,
        **axis_entries,
        **param_entries,
    )
    return TransportDatasetSpec(data=ds.data, axes=ds.axes, params=ds.params)


def _spec_values(value: ParamSpec | AxisSpec, name: str) -> NDArray64 | float:
    if not isinstance(value, (ParamSpec, AxisSpec)):
        raise TypeError(f"{name} must be a ParamSpec or AxisSpec.")
    return value.values
