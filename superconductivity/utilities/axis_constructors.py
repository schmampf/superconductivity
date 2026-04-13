"""Axis constructor helpers."""

from __future__ import annotations

from typing import Sequence

from .axis import AxisSpec, construct_axis
from .label import (
    label_A_hnu,
    label_A_mV,
    label_I_GNDelta,
    label_I_nA,
    label_T_K,
    label_T_Tc,
    label_V_Delta,
    label_V_mV,
    label_hnu_Delta,
    label_nu_GHz,
)
from .types import NDArray64


def axis_V_mV(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_V_mV(), kind="x")


def axis_I_nA(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_I_nA(), kind="y")


def axis_A_mV(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_A_mV(), kind="y")


def axis_nu_GHz(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_nu_GHz(), kind="y")


def axis_T_K(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_T_K(), kind="y")


def axis_V_Delta(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_V_Delta(), kind="x")


def axis_I_GNDelta(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_I_GNDelta(), kind="x")


def axis_A_hnu(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_A_hnu(), kind="y")


def axis_hnu_Delta(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_hnu_Delta(), kind="y")


def axis_T_Tc(
    min_value: float | None = None,
    max_value: float | None = None,
    bins: int | None = None,
    *,
    values: Sequence[float] | NDArray64 | None = None,
) -> AxisSpec:
    return construct_axis(min_value, max_value, bins, values=values, meta=label_T_Tc(), kind="y")


# Backwards-compatible aliases for current callers.
construct_axis = construct_axis
construct_V_mV = axis_V_mV
construct_I_nA = axis_I_nA
construct_A_mV = axis_A_mV
construct_nu_GHz = axis_nu_GHz
construct_T_K = axis_T_K
construct_V_Delta = axis_V_Delta
construct_I_GNDelta = axis_I_GNDelta
construct_A_hnu = axis_A_hnu
construct_hnu_Delta = axis_hnu_Delta
construct_T_Tc = axis_T_Tc


__all__ = [
    "axis_V_mV",
    "axis_I_nA",
    "axis_A_mV",
    "axis_nu_GHz",
    "axis_T_K",
    "axis_V_Delta",
    "axis_I_GNDelta",
    "axis_A_hnu",
    "axis_hnu_Delta",
    "axis_T_Tc",
    "construct_axis",
    "construct_V_mV",
    "construct_I_nA",
    "construct_A_mV",
    "construct_nu_GHz",
    "construct_T_K",
    "construct_V_Delta",
    "construct_I_GNDelta",
    "construct_A_hnu",
    "construct_hnu_Delta",
    "construct_T_Tc",
]
