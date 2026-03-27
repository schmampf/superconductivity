"""Helpers for offset-corrected IV sampling derived from IV traces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator, Sequence, TypedDict

import numpy as np

from ..utilities.constants import G_0_muS
from ..utilities.functions import bin_y_over_x
from ..utilities.functions import upsample as upsample_xy
from ..utilities.safety import (
    require_all_finite,
    require_min_size,
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

    upsample: int
    Vbin_mV: Sequence[float] | NDArray64
    Ibin_nA: Sequence[float] | NDArray64
    Voff_mV: float | Sequence[float] | NDArray64 = 0.0
    Ioff_nA: float | Sequence[float] | NDArray64 = 0.0

    def __post_init__(self) -> None:
        """Normalize grids and validate scalar settings."""
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

        self.Voff_mV = _normalize_offset_values(self.Voff_mV, name="Voff_mV")
        self.Ioff_nA = _normalize_offset_values(self.Ioff_nA, name="Ioff_nA")


def _normalize_offset_values(
    values: float | Sequence[float] | NDArray64,
    *,
    name: str,
) -> float | NDArray64:
    """Normalize one offset input as scalar or 1D float64 array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        value = float(array)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value
    if array.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D.")
    array = np.asarray(array, dtype=np.float64)
    require_min_size(array, 1, name=name)
    require_all_finite(array, name=name)
    return array


def _offsets_as_array(
    values: float | NDArray64,
    *,
    count: int,
    name: str,
) -> NDArray64:
    """Return one scalar offset or one per-trace offset array."""
    if isinstance(values, np.ndarray):
        if values.size == 1:
            return np.full(count, float(values[0]), dtype=np.float64)
        if values.size != count:
            raise ValueError(
                f"{name} must be scalar or have length {count}.",
            )
        return np.asarray(values, dtype=np.float64)
    return np.full(count, float(values), dtype=np.float64)


def _offset_as_scalar(
    values: float | NDArray64,
    *,
    name: str,
) -> float:
    """Return one scalar offset for single-trace sampling."""
    if isinstance(values, np.ndarray):
        if values.size != 1:
            raise ValueError(
                f"{name} must be scalar for get_sampling(...).",
            )
        return float(values[0])
    return float(values)


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
) -> tuple[NDArray64, NDArray64]:
    """Return one IV trace as float64 arrays for binning."""
    v_down_mV = np.asarray(trace["V_mV"], dtype=np.float64)
    i_down_nA = np.asarray(trace["I_nA"], dtype=np.float64)
    return (
        np.asarray(v_down_mV, dtype=np.float64),
        np.asarray(i_down_nA, dtype=np.float64),
    )


def fill_sampling_spec_from_offset(
    spec: SamplingSpec,
    offset: OffsetTrace,
) -> SamplingSpec:
    """Copy fitted offsets into one sampling spec."""
    return replace(
        spec,
        Voff_mV=float(offset["Voff_mV"]),
        Ioff_nA=float(offset["Ioff_nA"]),
    )


def fill_sampling_spec_from_offsets(
    spec: SamplingSpec,
    offsets: OffsetTraces,
) -> SamplingSpec:
    """Copy one offset collection into one sampling spec."""
    return replace(
        spec,
        Voff_mV=np.asarray(offsets.Voff_mV, dtype=np.float64),
        Ioff_nA=np.asarray(offsets.Ioff_nA, dtype=np.float64),
    )


def fill_sampling_specs_from_offsets(
    spec: SamplingSpec,
    offsets: OffsetTraces,
) -> list[SamplingSpec]:
    """Copy one shared sampling setup across one offset collection."""
    return _expand_sampling_spec(
        fill_sampling_spec_from_offsets(spec, offsets),
        count=len(offsets),
    )


def get_sampling(
    trace: IVTrace,
    spec: SamplingSpec,
) -> SamplingTrace:
    """Sample one prepared IV trace onto fixed V/I grids."""
    v_down_mV, i_down_nA = _prepare_trace_for_sampling(trace=trace)

    Ioff_nA = _offset_as_scalar(spec.Ioff_nA, name="Ioff_nA")
    Voff_mV = _offset_as_scalar(spec.Voff_mV, name="Voff_mV")
    i_corr_nA = i_down_nA - Ioff_nA
    v_corr_mV = v_down_mV - Voff_mV
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
        "Voff_mV": Voff_mV,
        "Ioff_nA": Ioff_nA,
        "Vbin_mV": np.asarray(spec.Vbin_mV, dtype=np.float64),
        "Ibin_nA": np.asarray(spec.Ibin_nA, dtype=np.float64),
        "I_nA": np.asarray(i_sampled_nA, dtype=np.float64),
        "V_mV": np.asarray(v_sampled_mV, dtype=np.float64),
        "dG_G0": np.asarray(dG_G0, dtype=np.float64),
        "dR_R0": np.asarray(dR_R0, dtype=np.float64),
    }


def _expand_sampling_spec(
    spec: SamplingSpec,
    *,
    count: int,
) -> list[SamplingSpec]:
    """Expand one sampling spec to one scalar-offset spec per trace."""
    Voff_mV = _offsets_as_array(spec.Voff_mV, count=count, name="Voff_mV")
    Ioff_nA = _offsets_as_array(spec.Ioff_nA, count=count, name="Ioff_nA")
    return [
        replace(
            spec,
            Voff_mV=float(Voff_mV[index]),
            Ioff_nA=float(Ioff_nA[index]),
        )
        for index in range(count)
    ]


def _normalize_sampling_specs(
    traces: IVTraces,
    specs: SamplingSpec | Sequence[SamplingSpec],
) -> list[SamplingSpec]:
    """Expand one shared spec or validate one spec per trace."""
    if isinstance(specs, SamplingSpec):
        return _expand_sampling_spec(specs, count=len(traces))

    specs_list = list(specs)
    if len(traces) != len(specs_list):
        raise ValueError("traces and specs must have the same length.")
    return [replace(spec) for spec in specs_list]


def _validate_shared_sampling_settings(specs: Sequence[SamplingSpec]) -> SamplingSpec:
    """Validate that one spec collection shares the same sampling grids."""
    reference = replace(specs[0])
    for spec in specs[1:]:
        if int(spec.upsample) != int(reference.upsample):
            raise ValueError("All specs must share the same upsample.")
        if not np.array_equal(spec.Vbin_mV, reference.Vbin_mV):
            raise ValueError("All specs must share the same Vbin_mV grid.")
        if not np.array_equal(spec.Ibin_nA, reference.Ibin_nA):
            raise ValueError("All specs must share the same Ibin_nA grid.")
    return replace(
        reference,
        Voff_mV=np.asarray(
            [float(spec.Voff_mV) for spec in specs],
            dtype=np.float64,
        ),
        Ioff_nA=np.asarray(
            [float(spec.Ioff_nA) for spec in specs],
            dtype=np.float64,
        ),
    )


def get_samplings(
    traces: IVTraces,
    specs: SamplingSpec | Sequence[SamplingSpec],
    show_progress: bool = True,
) -> SamplingTraces:
    """Sample one collection of prepared IV traces."""
    specs_list = _normalize_sampling_specs(traces=traces, specs=specs)
    if isinstance(specs, SamplingSpec):
        collection_spec = replace(specs)
    else:
        collection_spec = _validate_shared_sampling_settings(specs_list)

    iterable: Iterator[tuple[IVTrace, SamplingSpec]] | zip[IVTrace, SamplingSpec]
    iterable = zip(traces, specs_list)
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
            spec=spec,
        )
        for trace, spec in iterable
    ]
    return SamplingTraces(
        spec=collection_spec,
        traces=sampled,
    )


__all__ = [
    "SamplingSpec",
    "SamplingTrace",
    "SamplingTraces",
    "fill_sampling_spec_from_offset",
    "fill_sampling_spec_from_offsets",
    "fill_sampling_specs_from_offsets",
    "get_sampling",
    "get_samplings",
]
