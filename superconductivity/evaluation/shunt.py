"""Helpers for shunt-resistance estimates derived from IV traces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Iterator, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)
from ..utilities.types import NDArray64
from .ivdata import IVTrace, IVTraces


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
class ShuntSpec:
    """Configuration for subgap shunt fits on IV traces."""

    delta_mV: float
    subgap_range: Sequence[float] | NDArray64 = (0.0, 0.2)
    min_points: int = 2

    def __post_init__(self) -> None:
        """Normalize the fit window and validate scalar settings."""
        delta_mV = float(self.delta_mV)
        if not np.isfinite(delta_mV) or delta_mV <= 0.0:
            raise ValueError("delta_mV must be finite and > 0.")
        self.delta_mV = delta_mV

        subgap_range = to_1d_float64(self.subgap_range, "subgap_range")
        if subgap_range.size != 2:
            raise ValueError("subgap_range must contain exactly two values.")
        require_all_finite(subgap_range, name="subgap_range")
        lower = float(subgap_range[0])
        upper = float(subgap_range[1])
        if lower < 0.0:
            raise ValueError("subgap_range lower bound must be >= 0.")
        if upper <= lower:
            raise ValueError(
                "subgap_range upper bound must be greater than the lower "
                "bound.",
            )
        self.subgap_range = np.asarray([lower, upper], dtype=np.float64)

        min_points = int(self.min_points)
        if min_points < 2:
            raise ValueError("min_points must be >= 2.")
        self.min_points = min_points

    @property
    def window_mV(self) -> tuple[float, float]:
        """Return the absolute-voltage fit window in mV."""
        return (
            float(self.subgap_range[0] * self.delta_mV),
            float(self.subgap_range[1] * self.delta_mV),
        )


class ShuntTrace(TypedDict):
    """One shunt-fit result with metadata and selected tuples."""

    specific_key: str
    index: int | None
    yvalue: float | None
    Gshunt_uS: float
    Rshunt_MOhm: float
    Iintercept_nA: float
    rmse_nA: float
    points: int
    V_fit_mV: NDArray64
    I_fit_nA: NDArray64


@dataclass(slots=True)
class ShuntTraces:
    """Container for multiple shunt-fit results."""

    spec: ShuntSpec
    traces: list[ShuntTrace]
    keys: list[str] = field(init=False)
    yvalues: NDArray64 = field(init=False)
    Gshunt_uS: NDArray64 = field(init=False)
    Rshunt_MOhm: NDArray64 = field(init=False)
    points: NDArray[np.int64] = field(init=False)
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
        g_rows: list[float] = []
        r_rows: list[float] = []
        points: list[int] = []
        indices_by_key: dict[str, list[int]] = {}

        for index, trace in enumerate(self.traces):
            specific_key = trace["specific_key"]
            self.keys.append(specific_key)
            yvalue = trace["yvalue"]
            yvalues.append(np.nan if yvalue is None else float(yvalue))
            g_rows.append(float(trace["Gshunt_uS"]))
            r_rows.append(float(trace["Rshunt_MOhm"]))
            points.append(int(trace["points"]))
            indices_by_key.setdefault(specific_key, []).append(index)

        self.yvalues = np.asarray(yvalues, dtype=np.float64)
        self.Gshunt_uS = np.asarray(g_rows, dtype=np.float64)
        self.Rshunt_MOhm = np.asarray(r_rows, dtype=np.float64)
        self.points = np.asarray(points, dtype=np.int64)
        self._indices_by_key = indices_by_key

    def __len__(self) -> int:
        """Return number of traces."""
        return len(self.traces)

    def __iter__(self) -> Iterator[ShuntTrace]:
        """Iterate over traces."""
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice,
    ) -> ShuntTrace | list[ShuntTrace]:
        """Return trace(s) by positional index."""
        return self.traces[index]

    def all_by_key(
        self,
        specific_key: str,
    ) -> list[ShuntTrace]:
        """Return all traces with one exact specific key."""
        indices = self._indices_by_key.get(specific_key, [])
        return [self.traces[index] for index in indices]

    def by_key(
        self,
        specific_key: str,
    ) -> ShuntTrace:
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
    ) -> list[ShuntTrace]:
        """Return all traces with one y-value."""
        indices = self._find_indices_by_value(yvalue)
        return [self.traces[index] for index in indices]

    def by_value(
        self,
        yvalue: float,
    ) -> ShuntTrace:
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
        matches: list[ShuntTrace],
        selector_name: str,
        selector_value: str | float,
        plural_hint: str,
    ) -> ShuntTrace:
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


def _prepare_trace_for_shunt(
    trace: IVTrace,
) -> tuple[NDArray64, NDArray64]:
    """Validate one trace and return ``(V_mV, I_nA)`` arrays."""
    label = trace["specific_key"]
    v_mV = to_1d_float64(trace["V_mV"], f"{label}.V_mV")
    i_nA = to_1d_float64(trace["I_nA"], f"{label}.I_nA")

    require_same_shape(
        v_mV,
        i_nA,
        name_a=f"{label}.V_mV",
        name_b=f"{label}.I_nA",
    )
    require_min_size(v_mV, 2, name=f"{label}.V_mV")
    require_all_finite(v_mV, name=f"{label}.V_mV")
    require_all_finite(i_nA, name=f"{label}.I_nA")
    return v_mV, i_nA


def _conductance_to_resistance(g_uS: float) -> float:
    """Convert subgap conductance in uS to shunt resistance in MOhm."""
    atol = np.finfo(np.float64).eps * max(1.0, abs(g_uS)) * 8.0
    if np.isclose(g_uS, 0.0, rtol=0.0, atol=atol):
        return float(np.inf)
    return float(1.0 / g_uS)


def get_shunt(
    trace: IVTrace,
    spec: ShuntSpec,
) -> ShuntTrace:
    """Estimate one per-trace shunt from a low-bias subgap ``I(V)`` fit.

    Notes
    -----
    The fit window is defined by ``spec.subgap_range`` and applied to
    ``abs(V_mV) / delta_mV``. The fit itself is ``I_nA = G_uS * V_mV + I0``,
    so the slope in ``nA / mV`` is numerically equal to ``uS``.
    """
    v_mV, i_nA = _prepare_trace_for_shunt(trace=trace)
    window_lo_mV, window_hi_mV = spec.window_mV
    window_mask = (
        (np.abs(v_mV) >= window_lo_mV)
        & (np.abs(v_mV) <= window_hi_mV)
    )
    if not np.any(window_mask):
        raise ValueError(
            f"{trace['specific_key']} has no points in the |V| window "
            f"[{window_lo_mV:g}, {window_hi_mV:g}] mV.",
        )

    v_fit_mV = v_mV[window_mask]
    i_fit_nA = i_nA[window_mask]
    if v_fit_mV.size < spec.min_points:
        raise ValueError(
            f"{trace['specific_key']} needs at least {spec.min_points} "
            "points in the selected subgap window.",
        )

    v_span_mV = float(np.ptp(v_fit_mV))
    atol = np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(v_fit_mV))),
    ) * 8.0
    if np.isclose(v_span_mV, 0.0, rtol=0.0, atol=atol):
        raise ValueError(
            f"{trace['specific_key']} selected voltage span must be > 0 "
            "for a linear fit.",
        )

    gshunt_uS, intercept_nA = np.polyfit(v_fit_mV, i_fit_nA, deg=1)
    residual_nA = i_fit_nA - (gshunt_uS * v_fit_mV + intercept_nA)
    rmse_nA = float(np.sqrt(np.mean(residual_nA**2)))

    gshunt_uS = float(gshunt_uS)
    return {
        "specific_key": trace["specific_key"],
        "index": trace["index"],
        "yvalue": trace["yvalue"],
        "Gshunt_uS": gshunt_uS,
        "Rshunt_MOhm": _conductance_to_resistance(gshunt_uS),
        "Iintercept_nA": float(intercept_nA),
        "rmse_nA": rmse_nA,
        "points": int(v_fit_mV.size),
        "V_fit_mV": np.asarray(v_fit_mV, dtype=np.float64),
        "I_fit_nA": np.asarray(i_fit_nA, dtype=np.float64),
    }


def get_shunt_traces(
    traces: IVTraces,
    spec: ShuntSpec,
    show_progress: bool = True,
    workers: int = 1,
) -> ShuntTraces:
    """Estimate shunts for one collection of IV traces."""
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be > 0.")

    trace_iterable: Iterator[IVTrace] | IVTraces
    trace_iterable = traces
    if show_progress:
        tqdm = _import_tqdm()
        trace_iterable = tqdm(
            traces,
            total=len(traces),
            desc="get_shunt",
            unit="trace",
        )

    compute_one = partial(get_shunt, spec=spec)
    if worker_count == 1:
        out_traces = [compute_one(trace) for trace in trace_iterable]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            out_traces = list(executor.map(compute_one, trace_iterable))

    return ShuntTraces(spec=spec, traces=out_traces)


__all__ = [
    "ShuntSpec",
    "ShuntTrace",
    "ShuntTraces",
    "get_shunt",
    "get_shunt_traces",
]
