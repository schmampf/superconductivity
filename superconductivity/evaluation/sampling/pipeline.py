"""Full sampling pipeline orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..traces import Trace, Traces
from .containers import Sample, Samples
from .specs import SamplingSpec
from .transforms import (
    binning,
    downsampling,
    offset_correction,
    smooth,
    upsampling,
)

if TYPE_CHECKING:
    from ..analysis import OffsetTrace, OffsetTraces


def _zero_offset_trace() -> dict[str, object]:
    """Return one zero-valued offset analysis result."""
    return {
        "dGerr_G0": np.zeros((0,), dtype=np.float64),
        "dRerr_R0": np.zeros((0,), dtype=np.float64),
        "Voff_mV": 0.0,
        "Ioff_nA": 0.0,
    }


def sample(
    traces: Trace | Traces,
    *,
    samplingspec: SamplingSpec,
    offsetanalysis: OffsetTrace | OffsetTraces | None = None,
    show_progress: bool = True,
) -> Sample | Samples:
    """Run the full sampling pipeline on one trace or one collection."""
    prepared = traces
    if samplingspec.apply_offset_correction:
        if isinstance(traces, Traces):
            from ..analysis import OffsetTraces
            resolved_offsetanalysis = (
                offsetanalysis
                if offsetanalysis is not None
                else OffsetTraces(
                    traces=[_zero_offset_trace() for _ in range(len(traces))]
                )
            )
        else:
            resolved_offsetanalysis = (
                offsetanalysis if offsetanalysis is not None else _zero_offset_trace()
            )
        prepared = offset_correction(
            prepared,
            offsetanalysis=resolved_offsetanalysis,
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


__all__ = ["sample"]
