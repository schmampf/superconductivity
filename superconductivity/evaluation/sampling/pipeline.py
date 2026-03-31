"""Full sampling pipeline orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..traces import Trace, Traces
from .containers import Sample, Samples
from .specs import SamplingSpec, SmoothingSpec
from .transforms import (
    binning,
    downsampling,
    offset_correction,
    smooth,
)

if TYPE_CHECKING:
    from ..analysis import OffsetTrace, OffsetTraces


def sample(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    smoothingspec: SmoothingSpec | None = None,
    offsetanalysis: OffsetTrace | OffsetTraces | None = None,
    show_progress: bool = True,
) -> Sample | Samples:
    """Run the full sampling pipeline on one trace or one collection."""
    prepared = downsampling(
        traces,
        samplingspec=samplingspec,
        show_progress=show_progress,
    )
    if offsetanalysis is not None:
        prepared = offset_correction(
            prepared,
            offsetanalysis=offsetanalysis,
        )
    samples = binning(
        prepared,
        samplingspec=samplingspec,
        show_progress=show_progress,
    )
    if smoothingspec is not None:
        return smooth(
            samples,
            smoothingspec=smoothingspec,
            show_progress=show_progress,
        )
    return samples


__all__ = ["sample"]
