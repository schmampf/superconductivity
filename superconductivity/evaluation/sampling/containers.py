"""Sampling result container types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, TypedDict

import numpy as np
from numpy.typing import NDArray

from ...utilities.types import NDArray64
from ..traces import TraceMeta
from ..traces.meta import numeric_yvalue


class Sample(TypedDict):
    """One sampled IV result."""

    meta: TraceMeta
    Vbins_mV: NDArray64
    Ibins_nA: NDArray64
    I_nA: NDArray64
    V_mV: NDArray64
    dG_G0: NDArray64
    dR_R0: NDArray64


@dataclass(slots=True)
class Samples:
    """Container for sampled IV results.

    Parameters
    ----------
    traces : list of Sample
        Sampled traces on shared voltage and current grids.
    """

    traces: list[Sample]
    Vbins_mV: NDArray64 = field(init=False)
    Ibins_nA: NDArray64 = field(init=False)
    I_nA: NDArray64 = field(init=False)
    V_mV: NDArray64 = field(init=False)
    dG_G0: NDArray64 = field(init=False)
    dR_R0: NDArray64 = field(init=False)

    def __post_init__(self) -> None:
        """Build stacked arrays from ``traces``."""
        if len(self.traces) == 0:
            raise ValueError("traces must not be empty.")

        i_rows: list[NDArray64] = []
        v_rows: list[NDArray64] = []
        dg_rows: list[NDArray64] = []
        dr_rows: list[NDArray64] = []

        vbin_ref = np.asarray(self.traces[0]["Vbins_mV"], dtype=np.float64)
        ibin_ref = np.asarray(self.traces[0]["Ibins_nA"], dtype=np.float64)

        for trace in self.traces:
            vbin = np.asarray(trace["Vbins_mV"], dtype=np.float64)
            ibin = np.asarray(trace["Ibins_nA"], dtype=np.float64)
            if not np.array_equal(vbin, vbin_ref):
                raise ValueError("All traces must share the same Vbins_mV grid.")
            if not np.array_equal(ibin, ibin_ref):
                raise ValueError("All traces must share the same Ibins_nA grid.")

            i_rows.append(np.asarray(trace["I_nA"], dtype=np.float64))
            v_rows.append(np.asarray(trace["V_mV"], dtype=np.float64))
            dg_rows.append(np.asarray(trace["dG_G0"], dtype=np.float64))
            dr_rows.append(np.asarray(trace["dR_R0"], dtype=np.float64))

        self.Vbins_mV = np.asarray(vbin_ref, dtype=np.float64)
        self.Ibins_nA = np.asarray(ibin_ref, dtype=np.float64)
        self.I_nA = np.vstack(i_rows)
        self.V_mV = np.vstack(v_rows)
        self.dG_G0 = np.vstack(dg_rows)
        self.dR_R0 = np.vstack(dr_rows)

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice,
    ) -> Sample | list[Sample]:
        return self.traces[index]

    @property
    def metas(self) -> list[TraceMeta]:
        """Return ordered per-sample metadata."""
        return [trace["meta"] for trace in self.traces]

    @property
    def specific_keys(self) -> list[str]:
        """Return ordered specific keys."""
        return [meta.specific_key for meta in self.metas]

    @property
    def indices(self) -> NDArray[np.int64]:
        """Return ordered positional indices."""
        indices: list[int] = []
        for meta in self.metas:
            if meta.index is None:
                raise ValueError("Sample metadata must include indices.")
            indices.append(int(meta.index))
        return np.asarray(indices, dtype=np.int64)

    @property
    def yvalues(self) -> NDArray64:
        """Return ordered y-values."""
        values = [
            np.nan
            if numeric_yvalue(meta.yvalue) is None
            else float(numeric_yvalue(meta.yvalue))
            for meta in self.metas
        ]
        return np.asarray(values, dtype=np.float64)


__all__ = ["Sample", "Samples"]
