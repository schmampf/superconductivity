"""Sampling and smoothing configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...utilities.safety import require_all_finite, require_min_size, to_1d_float64
from ...utilities.types import NDArray64


def _validate_downsample_rate_Hz(nu_Hz: float) -> float:
    """Validate one downsampling rate in Hz."""
    nu_Hz = float(nu_Hz)
    if not np.isfinite(nu_Hz) or nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be finite and > 0.")
    return nu_Hz


@dataclass(slots=True)
class SamplingSpec:
    """Configuration for IV sampling.

    Parameters
    ----------
    Vbins_mV : sequence of float
        Voltage bias support used for the sampled ``I(V)`` curve.
    Ibins_nA : sequence of float
        Current bias support used for the sampled ``V(I)`` curve.
    nu_Hz : float, default=1.0
        Target downsampling rate in Hz for the time-domain trace.
    upsample : int, default=1
        Oversampling factor used before binning.
    """

    Vbins_mV: Sequence[float] | NDArray64
    Ibins_nA: Sequence[float] | NDArray64
    nu_Hz: float = 1.0
    upsample: int = 1

    def __post_init__(self) -> None:
        """Normalize grids and validate scalar settings."""
        self.nu_Hz = _validate_downsample_rate_Hz(self.nu_Hz)

        upsample = int(self.upsample)
        if upsample <= 0:
            raise ValueError("upsample must be > 0.")
        self.upsample = upsample

        self.Vbins_mV = to_1d_float64(self.Vbins_mV, "Vbins_mV")
        self.Ibins_nA = to_1d_float64(self.Ibins_nA, "Ibins_nA")

        require_min_size(self.Vbins_mV, 2, name="Vbins_mV")
        require_min_size(self.Ibins_nA, 2, name="Ibins_nA")
        require_all_finite(self.Vbins_mV, name="Vbins_mV")
        require_all_finite(self.Ibins_nA, name="Ibins_nA")


@dataclass(slots=True)
class SmoothingSpec:
    """Configuration for smoothing sampled IV curves.

    Parameters
    ----------
    median_bins : int, default=3
        Median filter window in bins. Use ``0`` or ``1`` to disable it.
    sigma_bins : float, default=2.0
        Gaussian smoothing width in bins. Use ``0`` to disable it.
    mode : str, default="nearest"
        Boundary mode forwarded to SciPy's 1D filters.
    """

    median_bins: int = 3
    sigma_bins: float = 2.0
    mode: str = "nearest"

    def __post_init__(self) -> None:
        """Validate smoothing parameters."""
        median_bins = int(self.median_bins)
        if median_bins < 0:
            raise ValueError("median_bins must be >= 0.")
        if median_bins > 1 and median_bins % 2 == 0:
            raise ValueError("median_bins must be odd when > 1.")
        self.median_bins = median_bins

        sigma_bins = float(self.sigma_bins)
        if not np.isfinite(sigma_bins) or sigma_bins < 0.0:
            raise ValueError("sigma_bins must be finite and >= 0.")
        self.sigma_bins = sigma_bins

        mode = str(self.mode).strip().lower()
        if mode == "":
            raise ValueError("mode must not be empty.")
        self.mode = mode


__all__ = ["SamplingSpec", "SmoothingSpec"]
