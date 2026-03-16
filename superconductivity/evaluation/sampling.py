"""Helpers for offset-corrected IV sampling derived from IV traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence, TypedDict

import numpy as np

from ..utilities.constants import G_0_muS
from ..utilities.functions import bin_y_over_x
from ..utilities.functions import upsample as upsample_xy
from ..utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ..utilities.types import NDArray64
from .ivdata import IVTrace, IVTraces
from .offset import OffsetTrace, OffsetTraces


def _import_tqdm():
    """Import tqdm lazily."""
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tqdm is required for progress display. Install it with "
            "'pip install tqdm'.",
        ) from exc
    return tqdm


@dataclass(slots=True)
class SamplingSpec:
    """Configuration for offset-corrected IV sampling."""

    nu_Hz: float
    upsample: int
    Vbin_mV: Sequence[float] | NDArray64
    Ibin_nA: Sequence[float] | NDArray64

    def __post_init__(self) -> None:
        """Normalize grids and validate scalar settings."""
        nu_Hz = float(self.nu_Hz)
        if not np.isfinite(nu_Hz) or nu_Hz <= 0.0:
            raise ValueError("nu_Hz must be finite and > 0.")
        self.nu_Hz = nu_Hz

        upsample = int(self.upsample)
        if upsample <= 0:
            raise ValueError("upsample must be > 0.")
        self.upsample = upsample

        self.Vbin_mV = to_1d_float64(self.Vbin_mV, "Vbin_mV")
        self.Ibin_nA = to_1d_float64(self.Ibin_nA, "Ibin_nA")

        require_min_size(self.Vbin_mV, 2, name="Vbin_mV")
        require_min_size(self.Ibin_nA, 2, name="Ibin_nA")
        require_all_finite(self.Vbin_mV, name="Vbin_mV")
        require_all_finite(self.Ibin_nA, name="Ibin_nA")

    @property
    def dt_s(self) -> float:
        """Temporary time-grid spacing used before IV binning."""
        return 0.5 / self.nu_Hz


class SamplingTrace(TypedDict):
    """One offset-corrected sampled IV result with metadata."""

    specific_key: str
    index: int | None
    yvalue: float | None
    Voff_mV: float
    Ioff_nA: float
    Vbin_mV: NDArray64
    Ibin_nA: NDArray64
    I_nA: NDArray64
    V_mV: NDArray64
    dG_G0: NDArray64
    dR_R0: NDArray64


@dataclass(slots=True)
class SamplingTraces:
    """Container for multiple sampled IV results."""

    spec: SamplingSpec
    traces: list[SamplingTrace]
    keys: list[str] = field(init=False)
    yvalues: NDArray64 = field(init=False)
    Voff_mV: NDArray64 = field(init=False)
    Ioff_nA: NDArray64 = field(init=False)
    Vbin_mV: NDArray64 = field(init=False)
    Ibin_nA: NDArray64 = field(init=False)
    I_nA: NDArray64 = field(init=False)
    V_mV: NDArray64 = field(init=False)
    dG_G0: NDArray64 = field(init=False)
    dR_R0: NDArray64 = field(init=False)
    _indices_by_key: dict[str, list[int]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Build stacked arrays and lookup tables from ``traces``."""
        if len(self.traces) == 0:
            raise ValueError("traces must not be empty.")

        self.keys = []
        yvalues: list[float] = []
        voffs_mV: list[float] = []
        ioffs_nA: list[float] = []
        i_rows: list[NDArray64] = []
        v_rows: list[NDArray64] = []
        dg_rows: list[NDArray64] = []
        dr_rows: list[NDArray64] = []
        indices_by_key: dict[str, list[int]] = {}

        vbin_ref = np.asarray(self.traces[0]["Vbin_mV"], dtype=np.float64)
        ibin_ref = np.asarray(self.traces[0]["Ibin_nA"], dtype=np.float64)

        for index, trace in enumerate(self.traces):
            specific_key = trace["specific_key"]
            self.keys.append(specific_key)
            yvalue = trace["yvalue"]
            yvalues.append(np.nan if yvalue is None else float(yvalue))
            voffs_mV.append(float(trace["Voff_mV"]))
            ioffs_nA.append(float(trace["Ioff_nA"]))

            vbin = np.asarray(trace["Vbin_mV"], dtype=np.float64)
            ibin = np.asarray(trace["Ibin_nA"], dtype=np.float64)
            if not np.array_equal(vbin, vbin_ref):
                raise ValueError("All traces must share the same Vbin_mV grid.")
            if not np.array_equal(ibin, ibin_ref):
                raise ValueError("All traces must share the same Ibin_nA grid.")

            i_rows.append(np.asarray(trace["I_nA"], dtype=np.float64))
            v_rows.append(np.asarray(trace["V_mV"], dtype=np.float64))
            dg_rows.append(np.asarray(trace["dG_G0"], dtype=np.float64))
            dr_rows.append(np.asarray(trace["dR_R0"], dtype=np.float64))
            indices_by_key.setdefault(specific_key, []).append(index)

        self.yvalues = np.asarray(yvalues, dtype=np.float64)
        self.Voff_mV = np.asarray(voffs_mV, dtype=np.float64)
        self.Ioff_nA = np.asarray(ioffs_nA, dtype=np.float64)
        self.Vbin_mV = np.asarray(vbin_ref, dtype=np.float64)
        self.Ibin_nA = np.asarray(ibin_ref, dtype=np.float64)
        self.I_nA = np.vstack(i_rows)
        self.V_mV = np.vstack(v_rows)
        self.dG_G0 = np.vstack(dg_rows)
        self.dR_R0 = np.vstack(dr_rows)
        self._indices_by_key = indices_by_key

    def __len__(self) -> int:
        """Return number of traces."""
        return len(self.traces)

    def __iter__(self) -> Iterator[SamplingTrace]:
        """Iterate over traces."""
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice,
    ) -> SamplingTrace | list[SamplingTrace]:
        """Return trace(s) by positional index."""
        return self.traces[index]

    def all_by_key(
        self,
        specific_key: str,
    ) -> list[SamplingTrace]:
        """Return all traces with one exact specific key."""
        indices = self._indices_by_key.get(specific_key, [])
        return [self.traces[index] for index in indices]

    def by_key(
        self,
        specific_key: str,
    ) -> SamplingTrace:
        """Return one trace for one exact specific key."""
        return self._resolve_unique_match(
            matches=self.all_by_key(specific_key),
            selector_name="specific_key",
            selector_value=specific_key,
            plural_hint="all_by_key",
        )

    def all_by_value(
        self,
        yvalue: float,
    ) -> list[SamplingTrace]:
        """Return all traces with one y-value."""
        indices = self._find_indices_by_value(yvalue)
        return [self.traces[index] for index in indices]

    def by_value(
        self,
        yvalue: float,
    ) -> SamplingTrace:
        """Return one trace for one y-value."""
        return self._resolve_unique_match(
            matches=self.all_by_value(yvalue),
            selector_name="yvalue",
            selector_value=yvalue,
            plural_hint="all_by_value",
        )

    def _find_indices_by_value(
        self,
        yvalue: float,
    ) -> list[int]:
        """Return positional indices that match one y-value."""
        value = float(yvalue)
        if not np.isfinite(value):
            raise ValueError("yvalue must be finite.")

        atol = np.finfo(np.float64).eps * max(1.0, abs(value)) * 8.0
        matches = np.flatnonzero(
            np.isclose(self.yvalues, value, rtol=0.0, atol=atol),
        )
        return matches.tolist()

    @staticmethod
    def _resolve_unique_match(
        matches: list[SamplingTrace],
        selector_name: str,
        selector_value: str | float,
        plural_hint: str,
    ) -> SamplingTrace:
        """Return one match or raise a clear selector error."""
        if len(matches) == 0:
            raise KeyError(
                f"{selector_name} {selector_value!r} was not found.",
            )
        if len(matches) > 1:
            raise ValueError(
                f"{selector_name} {selector_value!r} matches multiple "
                f"traces. Use index or {plural_hint}(...).",
            )
        return matches[0]


def _prepare_trace_for_sampling(
    trace: IVTrace,
    spec: SamplingSpec,
) -> tuple[NDArray64, NDArray64]:
    """Downsample one raw IV trace in time before offset correction."""
    label = trace["specific_key"]
    t_raw = to_1d_float64(trace["t_s"], f"{label}.t_s")
    v_raw_mV = to_1d_float64(trace["V_mV"], f"{label}.V_mV")
    i_raw_nA = to_1d_float64(trace["I_nA"], f"{label}.I_nA")

    require_same_shape(
        t_raw,
        v_raw_mV,
        name_a=f"{label}.t_s",
        name_b=f"{label}.V_mV",
    )
    require_same_shape(
        t_raw,
        i_raw_nA,
        name_a=f"{label}.t_s",
        name_b=f"{label}.I_nA",
    )
    require_min_size(t_raw, 2, name=f"{label}.t_s")
    require_all_finite(t_raw, name=f"{label}.t_s")
    require_all_finite(v_raw_mV, name=f"{label}.V_mV")
    require_all_finite(i_raw_nA, name=f"{label}.I_nA")

    t_min = float(np.min(t_raw))
    t_max = float(np.max(t_raw))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError(f"{label}.t_s must span at least two time points.")

    t_bins_s = np.arange(
        t_min,
        t_max + 0.5 * spec.dt_s,
        spec.dt_s,
        dtype=np.float64,
    )
    if t_bins_s.size < 2:
        t_bins_s = np.linspace(t_min, t_max, 2, dtype=np.float64)

    v_down_mV = bin_y_over_x(x=t_raw, y=v_raw_mV, x_bins=t_bins_s)
    i_down_nA = bin_y_over_x(x=t_raw, y=i_raw_nA, x_bins=t_bins_s)
    finite = np.isfinite(v_down_mV) & np.isfinite(i_down_nA)
    v_down_mV = v_down_mV[finite]
    i_down_nA = i_down_nA[finite]
    require_min_size(v_down_mV, 2, name=f"{label} downsampled trace")

    return (
        np.asarray(v_down_mV, dtype=np.float64),
        np.asarray(i_down_nA, dtype=np.float64),
    )


def _validate_trace_offset_pair(
    trace: IVTrace,
    offset: OffsetTrace,
) -> None:
    """Validate that one offset result belongs to one IV trace."""
    if trace["specific_key"] != offset["specific_key"]:
        raise ValueError(
            "trace and offset must refer to the same specific_key.",
        )

    trace_index = trace["index"]
    offset_index = offset["index"]
    if (
        trace_index is not None
        and offset_index is not None
        and int(trace_index) != int(offset_index)
    ):
        raise ValueError("trace and offset must refer to the same index.")

    trace_y = trace["yvalue"]
    offset_y = offset["yvalue"]
    if trace_y is None or offset_y is None:
        return

    atol = np.finfo(np.float64).eps * max(1.0, abs(float(trace_y))) * 8.0
    if not np.isclose(float(trace_y), float(offset_y), rtol=0.0, atol=atol):
        raise ValueError("trace and offset must refer to the same yvalue.")


def get_sampling(
    trace: IVTrace,
    offset: OffsetTrace,
    spec: SamplingSpec,
) -> SamplingTrace:
    """Sample one offset-corrected IV trace onto fixed V/I grids."""
    _validate_trace_offset_pair(trace, offset)
    v_down_mV, i_down_nA = _prepare_trace_for_sampling(trace=trace, spec=spec)

    i_corr_nA = i_down_nA - float(offset["Ioff_nA"])
    v_corr_mV = v_down_mV - float(offset["Voff_mV"])
    i_over_nA, v_over_mV = upsample_xy(
        x=i_corr_nA,
        y=v_corr_mV,
        factor=spec.upsample,
        method="linear",
    )

    v_sampled_mV = bin_y_over_x(
        x=i_over_nA,
        y=v_over_mV,
        x_bins=spec.Ibin_nA,
    )
    i_sampled_nA = bin_y_over_x(
        x=v_over_mV,
        y=i_over_nA,
        x_bins=spec.Vbin_mV,
    )

    dG_G0 = np.gradient(i_sampled_nA, spec.Vbin_mV) / G_0_muS
    dR_R0 = np.gradient(v_sampled_mV, spec.Ibin_nA) * G_0_muS

    return {
        "specific_key": trace["specific_key"],
        "index": trace["index"],
        "yvalue": trace["yvalue"],
        "Voff_mV": float(offset["Voff_mV"]),
        "Ioff_nA": float(offset["Ioff_nA"]),
        "Vbin_mV": np.asarray(spec.Vbin_mV, dtype=np.float64),
        "Ibin_nA": np.asarray(spec.Ibin_nA, dtype=np.float64),
        "I_nA": np.asarray(i_sampled_nA, dtype=np.float64),
        "V_mV": np.asarray(v_sampled_mV, dtype=np.float64),
        "dG_G0": np.asarray(dG_G0, dtype=np.float64),
        "dR_R0": np.asarray(dR_R0, dtype=np.float64),
    }


def get_samplings(
    traces: IVTraces,
    offsets: OffsetTraces,
    spec: SamplingSpec,
    show_progress: bool = True,
) -> SamplingTraces:
    """Sample one collection of offset-corrected IV traces."""
    if len(traces) != len(offsets):
        raise ValueError("traces and offsets must have the same length.")

    iterable: Iterator[tuple[IVTrace, OffsetTrace]] | zip[IVTrace, OffsetTrace]
    iterable = zip(traces, offsets)
    if show_progress:
        tqdm = _import_tqdm()
        iterable = tqdm(
            iterable,
            total=len(traces),
            desc="get_samplings",
            unit="trace",
        )

    sampled = [
        get_sampling(
            trace=trace,
            offset=offset,
            spec=spec,
        )
        for trace, offset in iterable
    ]
    return SamplingTraces(spec=spec, traces=sampled)


__all__ = [
    "SamplingSpec",
    "SamplingTrace",
    "SamplingTraces",
    "get_sampling",
    "get_samplings",
]
