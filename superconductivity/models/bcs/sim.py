"""BCS simulation dataset entrypoint."""

from __future__ import annotations

import numpy as np

from ...utilities.constants import G0_muS, h_pVs
from ...utilities.meta import AxisSpec, ParamSpec, axis, data, gridded_dataset, param
from ...utilities.types import NDArray64
from ..basics import get_DeltaT_meV as get_DeltaT_meV_scalar
from ..basics import get_Tc_K
from .backend import Backend, Kernel
from .bcs import (
    _SWEEP_PARAM_ORDER,
    _normalize_sweep_values,
    _validate_finite_scalar,
    _validate_nonnegative_scalar,
    _validate_voltage_axis,
    get_Ibcs_nA,
)

_G0_uS = float(G0_muS)
_h_pVs = float(h_pVs)


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
    sweep_shape_only = tuple(I_nA.shape[:-1])
    data_shape = tuple(I_nA.shape)
    v_order = len(sweep_names)

    axis_entries: dict[str, object] = {
        "V_mV": axis("V_mV", values=V_requested, order=v_order),
    }
    for order, name in enumerate(sweep_names):
        axis_entries[name] = axis(name, values=sweep_vectors[name], order=order)

    V_full = np.broadcast_to(
        V_requested.reshape((1,) * len(sweep_names) + (V_requested.size,)),
        data_shape,
    )
    GN_full = _broadcast_parameter(
        values=GN_values,
        scalar=GN_scalar,
        order=sweep_names.index("GN_G0") if "GN_G0" in sweep_names else -1,
        shape=data_shape,
    )
    Delta_full = _broadcast_parameter(
        values=Delta_values,
        scalar=Delta_scalar,
        order=sweep_names.index("Delta_meV") if "Delta_meV" in sweep_names else -1,
        shape=data_shape,
    )
    A_full = _broadcast_parameter(
        values=A_values,
        scalar=A_scalar,
        order=sweep_names.index("A_mV") if "A_mV" in sweep_names else -1,
        shape=data_shape,
    )
    nu_full = _broadcast_parameter(
        values=nu_values,
        scalar=nu_scalar,
        order=sweep_names.index("nu_GHz") if "nu_GHz" in sweep_names else -1,
        shape=data_shape,
    )
    T_full = _broadcast_parameter(
        values=T_values,
        scalar=T_scalar,
        order=sweep_names.index("T_K") if "T_K" in sweep_names else -1,
        shape=data_shape,
    )
    gamma_full = _broadcast_parameter(
        values=gamma_values,
        scalar=gamma_scalar,
        order=sweep_names.index("gamma_meV") if "gamma_meV" in sweep_names else -1,
        shape=data_shape,
    )
    sigma_full = _broadcast_parameter(
        values=sigma_values,
        scalar=sigma_scalar,
        order=sweep_names.index("sigmaV_mV") if "sigmaV_mV" in sweep_names else -1,
        shape=data_shape,
    )

    GN_sweep = _broadcast_sweep_parameter(
        values=GN_values,
        scalar=GN_scalar,
        order=sweep_names.index("GN_G0") if "GN_G0" in sweep_names else -1,
        shape=sweep_shape_only,
    )
    T_sweep = _broadcast_sweep_parameter(
        values=T_values,
        scalar=T_scalar,
        order=sweep_names.index("T_K") if "T_K" in sweep_names else -1,
        shape=sweep_shape_only,
    )
    Delta_sweep = _broadcast_sweep_parameter(
        values=Delta_values,
        scalar=Delta_scalar,
        order=sweep_names.index("Delta_meV") if "Delta_meV" in sweep_names else -1,
        shape=sweep_shape_only,
    )
    gamma_sweep = _broadcast_sweep_parameter(
        values=gamma_values,
        scalar=gamma_scalar,
        order=sweep_names.index("gamma_meV") if "gamma_meV" in sweep_names else -1,
        shape=sweep_shape_only,
    )
    nu_sweep = _broadcast_sweep_parameter(
        values=nu_values,
        scalar=nu_scalar,
        order=sweep_names.index("nu_GHz") if "nu_GHz" in sweep_names else -1,
        shape=sweep_shape_only,
    )
    A_sweep = _broadcast_sweep_parameter(
        values=A_values,
        scalar=A_scalar,
        order=sweep_names.index("A_mV") if "A_mV" in sweep_names else -1,
        shape=sweep_shape_only,
    )
    sigma_sweep = _broadcast_sweep_parameter(
        values=sigma_values,
        scalar=sigma_scalar,
        order=sweep_names.index("sigmaV_mV") if "sigmaV_mV" in sweep_names else -1,
        shape=sweep_shape_only,
    )

    denom_delta_g0 = Delta_full * _G0_uS
    denom_delta_gn = Delta_full * (GN_full * _G0_uS)
    G_muS = np.asarray(np.gradient(I_nA, V_requested, axis=-1), dtype=np.float64)
    dIdV_muS = G_muS
    with np.errstate(divide="ignore", invalid="ignore"):
        eI_DeltaG0 = np.where(denom_delta_g0 != 0.0, I_nA / denom_delta_g0, np.nan)
        eI_DeltaGN = np.where(denom_delta_gn != 0.0, I_nA / denom_delta_gn, np.nan)
        G_G0 = G_muS / _G0_uS
        G_GN = np.where(GN_full != 0.0, G_muS / (GN_full * _G0_uS), np.nan)
        dIdV_G0 = dIdV_muS / _G0_uS
        dIdV_GN = np.where(GN_full != 0.0, dIdV_muS / (GN_full * _G0_uS), np.nan)
        R_MOhm = np.where(G_muS != 0.0, 1.0 / G_muS, np.nan)
        dVdI_MOhm = np.where(dIdV_muS != 0.0, 1.0 / dIdV_muS, np.nan)
        R_R0 = R_MOhm * _G0_uS
        R_RN = R_MOhm * (GN_full * _G0_uS)
        dVdI_R0 = dVdI_MOhm * _G0_uS
        dVdI_RN = dVdI_MOhm * (GN_full * _G0_uS)
        eV_Delta = np.where(Delta_full != 0.0, V_full / Delta_full, np.nan)
        eA_hnu = np.where(
            (nu_full * _h_pVs) != 0.0,
            A_full / (nu_full * _h_pVs),
            np.nan,
        )

    data_entries: dict[str, object] = {
        "I_nA": data("I_nA", I_nA),
        "eI_DeltaG0": data("eI_DeltaG0", eI_DeltaG0),
        "eI_DeltaGN": data("eI_DeltaGN", eI_DeltaGN),
        "G_muS": data("G_muS", G_muS),
        "dIdV_muS": data("dIdV_muS", dIdV_muS),
        "G_G0": data("G_G0", G_G0),
        "G_GN": data("G_GN", G_GN),
        "dIdV_G0": data("dIdV_G0", dIdV_G0),
        "dIdV_GN": data("dIdV_GN", dIdV_GN),
        "R_MOhm": data("R_MOhm", R_MOhm),
        "R_R0": data("R_R0", R_R0),
        "R_RN": data("R_RN", R_RN),
        "dVdI_MOhm": data("dVdI_MOhm", dVdI_MOhm),
        "dVdI_R0": data("dVdI_R0", dVdI_R0),
        "dVdI_RN": data("dVdI_RN", dVdI_RN),
        "V_mV_data": data("V_mV_data", V_full),
        "eV_Delta_data": data("eV_Delta_data", eV_Delta),
    }

    param_entries: dict[str, object] = {}
    for name in _SWEEP_PARAM_ORDER:
        if not scalar_flags[name]:
            continue
        value = float(sweep_vectors[name][0])
        param_entries[name] = param(name, value)

    with np.errstate(divide="ignore", invalid="ignore"):
        Tc_sweep = np.asarray(
            np.vectorize(get_Tc_K, otypes=[np.float64])(Delta_sweep),
            dtype=np.float64,
        )
        DeltaT_sweep = np.asarray(
            np.vectorize(get_DeltaT_meV_scalar, otypes=[np.float64])(
                Delta_sweep, T_sweep
            ),
            dtype=np.float64,
        )
        T_Tc_sweep = np.where(Tc_sweep != 0.0, T_sweep / Tc_sweep, np.nan)
        DeltaT_Delta_sweep = np.where(
            Delta_sweep != 0.0, DeltaT_sweep / Delta_sweep, np.nan
        )
        gamma_Delta_sweep = np.where(
            Delta_sweep != 0.0, gamma_sweep / Delta_sweep, np.nan
        )
        hnu_Delta_sweep = np.where(
            Delta_sweep != 0.0, (nu_sweep * _h_pVs) / Delta_sweep, np.nan
        )
        eA_hnu_sweep = np.where(
            (nu_sweep * _h_pVs) != 0.0, A_sweep / (nu_sweep * _h_pVs), np.nan
        )
        sigmaV_Delta_sweep = np.where(
            Delta_sweep != 0.0, sigma_sweep / Delta_sweep, np.nan
        )
        GN_muS_sweep = GN_sweep * _G0_uS
        RN_MOhm_sweep = np.where(GN_muS_sweep != 0.0, 1.0 / GN_muS_sweep, np.nan)
        RN_R0_sweep = np.where(GN_sweep != 0.0, 1.0 / GN_sweep, np.nan)

    _add_sweep_derived(
        name="T_Tc",
        values_sweep=T_Tc_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="DeltaT_meV",
        values_sweep=DeltaT_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="DeltaT_Delta",
        values_sweep=DeltaT_Delta_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="gamma_Delta",
        values_sweep=gamma_Delta_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="hnu_Delta",
        values_sweep=hnu_Delta_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="eA_hnu",
        values_sweep=eA_hnu_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="sigmaV_Delta",
        values_sweep=sigmaV_Delta_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="GN_muS",
        values_sweep=GN_muS_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="RN_MOhm",
        values_sweep=RN_MOhm_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )
    _add_sweep_derived(
        name="RN_R0",
        values_sweep=RN_R0_sweep,
        sweep_names=sweep_names,
        axis_entries=axis_entries,
        param_entries=param_entries,
        data_entries=data_entries,
        data_shape=data_shape,
    )

    return gridded_dataset(
        **data_entries,
        **axis_entries,
        **param_entries,
        required_data=("I_nA", "G_muS", "dIdV_muS"),
        required_axes=("V_mV",),
    )


def _broadcast_parameter(
    *,
    values: NDArray64,
    scalar: bool,
    order: int,
    shape: tuple[int, ...],
) -> NDArray64:
    if scalar:
        return np.full(shape, float(values[0]), dtype=np.float64)
    if order < 0:
        return np.full(shape, float(values[0]), dtype=np.float64)
    reshape = [1] * len(shape)
    reshape[order] = values.size
    return np.broadcast_to(values.reshape(tuple(reshape)), shape).astype(np.float64)


def _spec_values(value: ParamSpec | AxisSpec, name: str) -> NDArray64 | float:
    if not isinstance(value, (ParamSpec, AxisSpec)):
        raise TypeError(f"{name} must be a ParamSpec or AxisSpec.")
    return value.values


def _broadcast_sweep_parameter(
    *,
    values: NDArray64,
    scalar: bool,
    order: int,
    shape: tuple[int, ...],
) -> NDArray64:
    if not shape:
        return np.asarray(float(values[0]), dtype=np.float64)
    if scalar or order < 0:
        return np.full(shape, float(values[0]), dtype=np.float64)
    reshape = [1] * len(shape)
    reshape[order] = values.size
    return np.broadcast_to(values.reshape(tuple(reshape)), shape).astype(np.float64)


def _add_sweep_derived(
    *,
    name: str,
    values_sweep: NDArray64,
    sweep_names: list[str],
    axis_entries: dict[str, object],
    param_entries: dict[str, object],
    data_entries: dict[str, object],
    data_shape: tuple[int, ...],
) -> None:
    arr = np.asarray(values_sweep, dtype=np.float64)
    if arr.ndim == 0:
        param_entries[name] = param(name, float(arr))
        return
    if arr.ndim == 1 and len(sweep_names) == 1:
        try:
            axis_entries[name] = axis(name, values=arr, order=0)
            return
        except ValueError:
            pass
    expanded = np.broadcast_to(arr[..., None], data_shape)
    data_entries[name] = data(name, expanded)
