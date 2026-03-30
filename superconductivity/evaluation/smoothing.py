"""Helpers for smoothing sampled IV curves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, TypedDict

import numpy as np

from ..utilities.constants import G_0_muS
from ..utilities.functions import fill_nans
from ..utilities.types import NDArray64
from .sampling import SamplingSpec, SamplingTrace, SamplingTraces


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


def _import_scipy_ndimage():
    """Import SciPy ndimage filters lazily."""
    try:
        from scipy.ndimage import gaussian_filter1d, median_filter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SciPy is required for smoothing. Install it with " "'pip install scipy'.",
        ) from exc
    return gaussian_filter1d, median_filter


@dataclass(slots=True)
class SmoothingSpec:
    """Configuration for smoothing sampled IV curves.

    Parameters
    ----------
    smooth
        Whether smoothing is enabled. If ``False``,
        :func:`get_smoothed_sampling` returns the input trace unchanged.
    median_bins
        Median filter window in bins. Use ``0`` or ``1`` to disable the median
        step. Values larger than ``1`` must be odd.
    sigma_bins
        Gaussian smoothing width in bins. Use ``0`` to disable the Gaussian
        step.
    mode
        Boundary mode forwarded to SciPy's 1D filters. ``"nearest"`` is a
        robust default for finite supported segments.
    """

    smooth: bool = True
    median_bins: int = 3
    sigma_bins: float = 2.0
    mode: str = "nearest"

    def __post_init__(self) -> None:
        """Validate smoothing parameters."""
        self.smooth = bool(self.smooth)

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


class SmoothedSamplingTrace(TypedDict):
    """One smoothed sampled IV result with metadata."""

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
class SmoothedSamplingTraces:
    """Container for multiple smoothed sampled IV results."""

    sampling_spec: SamplingSpec
    smoothing_spec: SmoothingSpec
    traces: list[SmoothedSamplingTrace]
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

    def __iter__(self) -> Iterator[SmoothedSamplingTrace]:
        """Iterate over traces."""
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice,
    ) -> SmoothedSamplingTrace | list[SmoothedSamplingTrace]:
        """Return trace(s) by positional index."""
        return self.traces[index]

    def all_by_key(
        self,
        specific_key: str,
    ) -> list[SmoothedSamplingTrace]:
        """Return all traces with one exact specific key."""
        indices = self._indices_by_key.get(specific_key, [])
        return [self.traces[index] for index in indices]

    def by_key(
        self,
        specific_key: str,
    ) -> SmoothedSamplingTrace:
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
    ) -> list[SmoothedSamplingTrace]:
        """Return all traces with one y-value."""
        indices = self._find_indices_by_value(yvalue)
        return [self.traces[index] for index in indices]

    def by_value(
        self,
        yvalue: float,
    ) -> SmoothedSamplingTrace:
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
        matches: list[SmoothedSamplingTrace],
        selector_name: str,
        selector_value: str | float,
        plural_hint: str,
    ) -> SmoothedSamplingTrace:
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


def _smooth_supported_segment(
    y: NDArray64,
    spec: SmoothingSpec,
) -> NDArray64:
    """Smooth one 1D curve on its finite supported segment."""
    y_arr = np.asarray(y, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(y_arr))
    if finite_idx.size == 0:
        return np.asarray(y_arr, dtype=np.float64)

    lo = int(finite_idx[0])
    hi = int(finite_idx[-1]) + 1
    y_segment = np.asarray(y_arr[lo:hi], dtype=np.float64)
    y_segment = fill_nans(y_segment, method="linear")

    gaussian_filter1d, median_filter = _import_scipy_ndimage()
    if spec.median_bins > 1:
        y_segment = np.asarray(
            median_filter(
                y_segment,
                size=spec.median_bins,
                mode=spec.mode,
            ),
            dtype=np.float64,
        )
    if spec.sigma_bins > 0.0:
        y_segment = np.asarray(
            gaussian_filter1d(
                y_segment,
                sigma=spec.sigma_bins,
                mode=spec.mode,
            ),
            dtype=np.float64,
        )

    y_out = np.full_like(y_arr, np.nan, dtype=np.float64)
    y_out[lo:hi] = y_segment
    return y_out


def get_smoothed_sampling(
    sampling_trace: SamplingTrace,
    spec: SmoothingSpec,
) -> SmoothedSamplingTrace:
    """Smooth one sampled IV trace and recompute derivatives."""
    if not spec.smooth:
        return {
            "specific_key": sampling_trace["specific_key"],
            "index": sampling_trace["index"],
            "yvalue": sampling_trace["yvalue"],
            "Voff_mV": float(sampling_trace["Voff_mV"]),
            "Ioff_nA": float(sampling_trace["Ioff_nA"]),
            "Vbin_mV": np.asarray(
                sampling_trace["Vbin_mV"],
                dtype=np.float64,
            ),
            "Ibin_nA": np.asarray(
                sampling_trace["Ibin_nA"],
                dtype=np.float64,
            ),
            "I_nA": np.asarray(sampling_trace["I_nA"], dtype=np.float64),
            "V_mV": np.asarray(sampling_trace["V_mV"], dtype=np.float64),
            "dG_G0": np.asarray(
                sampling_trace["dG_G0"],
                dtype=np.float64,
            ),
            "dR_R0": np.asarray(
                sampling_trace["dR_R0"],
                dtype=np.float64,
            ),
        }

    vbin_mV = np.asarray(sampling_trace["Vbin_mV"], dtype=np.float64)
    ibin_nA = np.asarray(sampling_trace["Ibin_nA"], dtype=np.float64)
    i_smooth_nA = _smooth_supported_segment(sampling_trace["I_nA"], spec=spec)
    v_smooth_mV = _smooth_supported_segment(sampling_trace["V_mV"], spec=spec)
    dG_G0 = np.gradient(i_smooth_nA, vbin_mV) / G_0_muS
    dR_R0 = np.gradient(v_smooth_mV, ibin_nA) * G_0_muS

    return {
        "specific_key": sampling_trace["specific_key"],
        "index": sampling_trace["index"],
        "yvalue": sampling_trace["yvalue"],
        "Voff_mV": float(sampling_trace["Voff_mV"]),
        "Ioff_nA": float(sampling_trace["Ioff_nA"]),
        "Vbin_mV": np.asarray(vbin_mV, dtype=np.float64),
        "Ibin_nA": np.asarray(ibin_nA, dtype=np.float64),
        "I_nA": np.asarray(i_smooth_nA, dtype=np.float64),
        "V_mV": np.asarray(v_smooth_mV, dtype=np.float64),
        "dG_G0": np.asarray(dG_G0, dtype=np.float64),
        "dR_R0": np.asarray(dR_R0, dtype=np.float64),
    }


def get_smoothed_samplings(
    samplings: SamplingTraces,
    spec: SmoothingSpec,
    show_progress: bool = False,
) -> SmoothedSamplingTraces:
    """Smooth one collection of sampled IV traces."""
    iterable: Iterator[SamplingTrace] | SamplingTraces = samplings
    if show_progress:
        tqdm = _import_tqdm()
        iterable = tqdm(
            samplings,
            total=len(samplings),
            desc="get_smoothed_samplings",
            unit="trace",
        )

    traces = [
        get_smoothed_sampling(
            sampling_trace=sampling_trace,
            spec=spec,
        )
        for sampling_trace in iterable
    ]
    return SmoothedSamplingTraces(
        sampling_spec=samplings.spec,
        smoothing_spec=spec,
        traces=traces,
    )


__all__ = [
    "SmoothingSpec",
    "SmoothedSamplingTrace",
    "SmoothedSamplingTraces",
    "get_smoothed_sampling",
    "get_smoothed_samplings",
]
