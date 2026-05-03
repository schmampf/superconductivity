"""Offset analysis for IV traces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..utilities.constants import G0_muS
from ..utilities.meta import Dataset, axis, data
from ..utilities.meta.axis import AxisSpec
from ..utilities.safety import (
    require_all_finite,
    require_min_size,
    require_positive_int,
    to_1d_float64,
)
from ..utilities.types import NDArray64
from .sampling import SamplingSpec, sample
from .traces import Traces


def _import_tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tqdm is required for progress display. Install it with "
            "'pip install tqdm'.",
        ) from exc
    return tqdm


@dataclass(slots=True, kw_only=True)
class OffsetSpec(SamplingSpec):
    """Configuration for offset analysis on IV traces."""

    Voffscan_mV: Sequence[float] | NDArray64
    Ioffscan_nA: Sequence[float] | NDArray64

    def __post_init__(self) -> None:
        SamplingSpec.__post_init__(self)
        self.Voffscan_mV = to_1d_float64(self.Voffscan_mV, "Voffscan_mV")
        self.Ioffscan_nA = to_1d_float64(self.Ioffscan_nA, "Ioffscan_nA")
        require_min_size(self.Voffscan_mV, 2, name="Voffscan_mV")
        require_min_size(self.Ioffscan_nA, 2, name="Ioffscan_nA")
        require_all_finite(self.Voffscan_mV, name="Voffscan_mV")
        require_all_finite(self.Ioffscan_nA, name="Ioffscan_nA")

    def keys(self) -> tuple[str, ...]:
        return (*super().keys(), "Voffscan_mV", "Ioffscan_nA")


class OffsetDataset(Dataset):
    """Stacked offset-analysis result."""

    def __init__(
        self,
        *,
        yaxis: AxisSpec,
        dGerr_G0: NDArray64,
        dRerr_R0: NDArray64,
        Voff_mV: NDArray64,
        Ioff_nA: NDArray64,
        Voffscan_mV: NDArray64,
        Ioffscan_nA: NDArray64,
    ) -> None:
        dataset = Dataset(
            data=(
                data("dGerr_G0", dGerr_G0),
                data("dRerr_R0", dRerr_R0),
                data("Voff_mV", Voff_mV),
                data("Ioff_nA", Ioff_nA),
            ),
            axes=(
                yaxis,
                axis("Voffscan_mV", values=Voffscan_mV, order=1),
                axis("Ioffscan_nA", values=Ioffscan_nA, order=1),
            ),
        )
        super().__init__(data=dataset.data, axes=dataset.axes, params=dataset.params)
        object.__setattr__(self, "_yaxis", yaxis)

    @property
    def yaxis(self) -> AxisSpec:
        return self._yaxis

    @property
    def y(self) -> AxisSpec:
        return self._yaxis

    def keys(self) -> tuple[str, ...]:
        return ("y", "yaxis", *self._lookup.keys())

    @property
    def Voffscan_mV(self) -> AxisSpec:
        return Dataset.__getitem__(self, "Voffscan_mV")

    @property
    def Ioffscan_nA(self) -> AxisSpec:
        return Dataset.__getitem__(self, "Ioffscan_nA")


def _nanargmin_finite(values: NDArray64) -> int:
    finite = np.isfinite(values)
    if not np.any(finite):
        return int(values.size // 2)
    idx_finite = np.where(finite)[0]
    return int(idx_finite[np.argmin(values[finite])])


def _score_sampled_voltage(
    samples: tuple[Dataset, Dataset],
    *,
    bins: NDArray64,
) -> NDArray64:
    exp_v, _ = samples
    values = np.asarray(exp_v["I_nA"].values, dtype=np.float64)
    if values.ndim == 2:
        values = values[0]
    g_uS = np.gradient(values, bins)
    g_G0 = g_uS / G0_muS
    g_sym = np.abs(g_G0 - np.flip(g_G0, axis=0))
    return np.asarray(np.nanmean(g_sym), dtype=np.float64)


def _score_sampled_current(
    samples: tuple[Dataset, Dataset],
    *,
    bins: NDArray64,
) -> NDArray64:
    _, exp_i = samples
    values = np.asarray(exp_i["V_mV"].values, dtype=np.float64)
    if values.ndim == 2:
        values = values[0]
    r_MOhm = np.gradient(values, bins)
    r_R0 = r_MOhm * G0_muS
    r_sym = np.abs(r_R0 - np.flip(r_R0, axis=0))
    return np.asarray(np.nanmean(r_sym), dtype=np.float64)


def _offset_errors_for_scan(
    traces: Traces,
    spec: OffsetSpec,
    show_progress: bool = False,
) -> tuple[NDArray64, NDArray64, float, float]:
    g_err_G0 = []
    r_err_R0 = []
    v_scan = [float(v_off) for v_off in spec.Voffscan_mV]
    i_scan = [float(i_off) for i_off in spec.Ioffscan_nA]
    if show_progress:
        tqdm = _import_tqdm()
        v_scan = tqdm(
            v_scan,
            total=len(v_scan),
            desc="V offset",
            unit="offset",
            leave=False,
        )
    for off in v_scan:
        samples = sample(
            traces.offset(Voff_mV=off, Ioff_nA=0.0),
            samplingspec=replace(spec, apply_offset=False),
            show_progress=False,
        )
        g_err_G0.append(
            _score_sampled_voltage(
                samples, bins=np.asarray(spec.Vbins_mV, dtype=np.float64)
            )
        )
    if show_progress:
        tqdm = _import_tqdm()
        i_scan = tqdm(
            i_scan,
            total=len(i_scan),
            desc="I offset",
            unit="offset",
            leave=False,
        )
    for off in i_scan:
        samples = sample(
            traces.offset(Voff_mV=0.0, Ioff_nA=off),
            samplingspec=replace(spec, apply_offset=False),
            show_progress=False,
        )
        r_err_R0.append(
            _score_sampled_current(
                samples, bins=np.asarray(spec.Ibins_nA, dtype=np.float64)
            )
        )
    g_err_G0_arr = np.asarray(g_err_G0, dtype=np.float64)
    r_err_R0_arr = np.asarray(r_err_R0, dtype=np.float64)
    return (
        g_err_G0_arr,
        r_err_R0_arr,
        float(spec.Voffscan_mV[_nanargmin_finite(g_err_G0_arr)]),
        float(spec.Ioffscan_nA[_nanargmin_finite(r_err_R0_arr)]),
    )


def offset_analysis(
    traces: Traces,
    spec: OffsetSpec,
    show_progress: bool = True,
    workers: int = 8,
) -> OffsetDataset:
    worker_count = require_positive_int(workers, name="workers")
    traces_for_scan = list(traces)

    if worker_count == 1:
        out_rows = [
            _offset_errors_for_scan(
                traces=Traces(traces=[trace], skeys=["trace"], indices=[0]),
                spec=spec,
                show_progress=show_progress,
            )
            for trace in traces_for_scan
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            mapped = executor.map(
                lambda trace: _offset_errors_for_scan(
                    traces=Traces(traces=[trace], skeys=["trace"], indices=[0]),
                    spec=spec,
                    show_progress=False,
                ),
                traces_for_scan,
            )
            if show_progress:
                tqdm = _import_tqdm()
                mapped = tqdm(
                    mapped,
                    total=len(traces_for_scan),
                    desc="offset_analysis",
                    unit="trace",
                )
            out_rows = list(mapped)

    g_rows, r_rows, voffs, ioffs = zip(*out_rows, strict=True)
    y_axis = (
        traces.yaxis
        if traces.yaxis is not None
        else axis(
            "y",
            values=np.asarray(traces.indices, dtype=np.float64),
            order=0,
        )
    )
    return OffsetDataset(
        yaxis=y_axis,
        dGerr_G0=np.vstack(g_rows),
        dRerr_R0=np.vstack(r_rows),
        Voff_mV=np.asarray(voffs, dtype=np.float64),
        Ioff_nA=np.asarray(ioffs, dtype=np.float64),
        Voffscan_mV=np.asarray(spec.Voffscan_mV, dtype=np.float64),
        Ioffscan_nA=np.asarray(spec.Ioffscan_nA, dtype=np.float64),
    )


__all__ = ["OffsetSpec", "OffsetDataset", "offset_analysis"]
