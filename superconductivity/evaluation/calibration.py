"""Axis calibration helpers for experimental-to-physical mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

from ..utilities.meta.axis import AxisSpec
from ..utilities.safety import require_all_finite, require_min_size
from ..utilities.types import NDArray64
from .sampling.containers import Samples, make_samples
from .traces.keys import KeysSpec

CalibrationMode = Literal["function", "lookup"]
GapFillMode = Literal["nan", "nearest", "interpolate"]


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    """Normalized axis-calibration specification."""

    mode: CalibrationMode
    transform: Callable[..., NDArray64] | None = None
    params: float | Sequence[float] = 1.0
    lookup: Sequence[float] | NDArray64 | None = None
    gap_fill: GapFillMode = "nan"

    def __post_init__(self) -> None:
        if self.mode not in ("function", "lookup"):
            raise ValueError("mode must be 'function' or 'lookup'.")

        gap_fill = str(self.gap_fill).strip().lower()
        if gap_fill not in ("nan", "nearest", "interpolate"):
            raise ValueError(
                "gap_fill must be 'nan', 'nearest', or 'interpolate'.",
            )
        object.__setattr__(self, "gap_fill", gap_fill)
        object.__setattr__(self, "params", _normalize_params(self.params))

        if self.mode == "function":
            if self.transform is None:
                raise ValueError("transform is required when mode='function'.")
        else:
            if self.lookup is None:
                raise ValueError("lookup is required when mode='lookup'.")
            lookup = _validate_axis(self.lookup, "lookup")
            if np.any(np.diff(lookup) <= 0.0):
                raise ValueError("lookup must be strictly increasing.")
            object.__setattr__(self, "lookup", lookup)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Calibrated samples and resolved axis metadata."""

    samples: Samples
    axisspec: AxisSpec
    source_axis: NDArray64
    mapped_axis: NDArray64
    calibrated_axis: NDArray64


def calibrate(
    *,
    samples: Samples,
    axisspec: AxisSpec,
    keysspec: KeysSpec,
    calibrationspec: CalibrationSpec,
) -> CalibrationResult:
    """Calibrate a sampled trace collection onto one physical axis grid."""
    del keysspec  # Reserved for future label and parsing integration.

    source_axis = np.asarray(samples.yvalues, dtype=np.float64)
    target_axis = np.asarray(axisspec.axis, dtype=np.float64)
    require_min_size(source_axis, 2, "samples.yvalues")
    require_all_finite(source_axis, "samples.yvalues")
    require_min_size(target_axis, 2, "axisspec.axis")
    require_all_finite(target_axis, "axisspec.axis")
    if np.any(np.diff(target_axis) <= 0.0):
        raise ValueError("axisspec.axis must be strictly increasing.")

    calibrated_axis = _apply_calibration(source_axis, calibrationspec)
    calibrated_axis = _fill_axis_gaps(calibrated_axis, calibrationspec.gap_fill)
    if calibrated_axis.size != target_axis.size:
        raise ValueError(
            "calibrated axis and axisspec.axis must have the same length.",
        )

    ordered = np.argsort(calibrated_axis)
    mapped_axis = np.asarray(calibrated_axis[ordered], dtype=np.float64)
    samples_out = _copy_samples_with_calibrated_axis(
        samples,
        np.asarray(target_axis[ordered], dtype=np.float64),
        order=ordered,
    )

    return CalibrationResult(
        samples=samples_out,
        axisspec=axisspec,
        source_axis=np.asarray(source_axis[ordered], dtype=np.float64),
        mapped_axis=mapped_axis,
        calibrated_axis=np.asarray(target_axis[ordered], dtype=np.float64),
    )


def _apply_calibration(
    source_axis: NDArray64,
    spec: CalibrationSpec,
) -> NDArray64:
    if spec.mode == "function":
        assert spec.transform is not None
        mapped = spec.transform(source_axis, *spec.params)
        mapped_arr = np.asarray(mapped, dtype=np.float64)
        if mapped_arr.shape != source_axis.shape:
            raise ValueError("mapped axis must match source_axis shape.")
        return mapped_arr

    assert spec.lookup is not None
    lookup = np.asarray(spec.lookup, dtype=np.float64).reshape(-1)
    if lookup.size != source_axis.size:
        raise ValueError("lookup must match samples.yvalues length.")
    return np.asarray(lookup, dtype=np.float64)


def _copy_samples_with_calibrated_axis(
    samples: Samples,
    calibrated_axis: NDArray64,
    *,
    order: NDArray64,
) -> Samples:
    if len(samples) != calibrated_axis.size:
        raise ValueError("samples and calibrated_axis must have the same length.")
    indices = np.asarray(order, dtype=np.int64)
    return make_samples(
        Vbins_mV=np.asarray(samples["Vbins_mV"], dtype=np.float64),
        Ibins_nA=np.asarray(samples["Ibins_nA"], dtype=np.float64),
        I_nA=np.asarray(samples["I_nA"], dtype=np.float64)[indices],
        V_mV=np.asarray(samples["V_mV"], dtype=np.float64)[indices],
        yvalues=np.asarray(calibrated_axis, dtype=np.float64),
    )


def _fill_axis_gaps(axis_values: NDArray64, gap_fill: GapFillMode) -> NDArray64:
    axis_arr = np.asarray(axis_values, dtype=np.float64).reshape(-1)
    if np.all(np.isfinite(axis_arr)):
        return axis_arr
    if gap_fill == "nan":
        return axis_arr
    if gap_fill == "nearest":
        return _fill_nearest(axis_arr)
    if axis_arr.size < 2:
        raise ValueError("Cannot interpolate axis gaps with fewer than two values.")
    finite_idx = np.flatnonzero(np.isfinite(axis_arr))
    if finite_idx.size < 2:
        raise ValueError(
            "Cannot interpolate axis gaps with fewer than two finite values.",
        )
    filled = axis_arr.copy()
    x = np.arange(filled.size, dtype=np.float64)
    filled[~np.isfinite(filled)] = np.interp(
        x[~np.isfinite(filled)],
        x[finite_idx],
        filled[finite_idx],
    )
    return filled


def _fill_nearest(axis_values: NDArray64) -> NDArray64:
    filled = np.asarray(axis_values, dtype=np.float64).copy()
    finite_idx = np.flatnonzero(np.isfinite(filled))
    if finite_idx.size == 0:
        raise ValueError("Cannot fill gaps when all axis values are missing.")
    if finite_idx.size == 1:
        filled[:] = filled[finite_idx[0]]
        return filled
    x = np.arange(filled.size, dtype=np.float64)
    finite_x = x[finite_idx]
    finite_y = filled[finite_idx]
    filled[~np.isfinite(filled)] = finite_y[
        np.argmin(
            np.abs(finite_x[:, None] - x[~np.isfinite(filled)][None, :]),
            axis=0,
        )
    ]
    return filled


def _normalize_params(params: float | Sequence[float]) -> tuple[float, ...]:
    if np.isscalar(params):
        return (float(params),)
    values = tuple(float(value) for value in params)
    if len(values) == 0:
        raise ValueError("params must not be empty.")
    return values


def _validate_axis(values: Sequence[float] | NDArray64, name: str) -> NDArray64:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    require_min_size(arr, 2, name)
    require_all_finite(arr, name)
    return arr


__all__ = [
    "AxisSpec",
    "CalibrationMode",
    "GapFillMode",
    "CalibrationSpec",
    "CalibrationResult",
    "calibrate",
]
