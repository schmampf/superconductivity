"""Full sampling pipeline orchestration."""

from __future__ import annotations

import numpy as np

from ...utilities.meta import TransportDatasetSpec
from ..traces import Trace, Traces
from .specs import SamplingSpec
from .transforms import (
    binning,
    downsampling,
    offset_correction,
    smooth,
    upsampling,
)


def _sample_result(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = True,
) -> object:
    """Run the full sampling pipeline and return the two transport datasets."""
    prepared = traces
    if samplingspec.apply_offset_correction:
        prepared = offset_correction(
            prepared,
            samplingspec=samplingspec,
        )
    if samplingspec.apply_downsampling:
        prepared = downsampling(
            prepared,
            samplingspec=samplingspec,
            show_progress=show_progress,
        )
    if samplingspec.apply_upsampling:
        prepared = upsampling(
            prepared,
            samplingspec=samplingspec,
            show_progress=show_progress,
        )
    samples = binning(
        prepared,
        samplingspec=samplingspec,
        show_progress=show_progress,
    )
    if samplingspec.apply_smoothing:
        return smooth(
            samples,
            samplingspec=samplingspec,
            show_progress=show_progress,
        )
    return samples


def sample(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    show_progress: bool = True,
) -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    """Run the full sampling pipeline and return voltage- and current-bias datasets."""
    exp_v, exp_i = _sample_result(
        traces,
        samplingspec=samplingspec,
        show_progress=show_progress,
    )
    return exp_v, exp_i


__all__ = ["sample"]
