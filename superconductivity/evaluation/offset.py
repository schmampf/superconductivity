"""Helpers for offset analysis derived from IV traces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Iterator, Sequence, TypedDict

import numpy as np

from ..utilities.constants import G_0_muS
from ..utilities.functions import bin_y_over_x, fill_nans
from ..utilities.functions import upsample as upsample_xy
from ..utilities.safety import (require_all_finite, require_min_size,
                                to_1d_float64)
from ..utilities.types import NDArray64
from .ivdata import IVTrace, IVTraces
from .psd import _downsample_iv_trace


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


@lru_cache(maxsize=1)
def _import_jax_offset_kernels():
    """Import JAX helpers lazily for offset analysis."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "JAX is required for backend='jax'. Install it with "
            "'pip install jax jaxlib'.",
        ) from exc

    from ..utilities.functions_jax import jbin_y_over_x

    @jax.jit
    def _bin_offsets(
        x: NDArray64,
        y: NDArray64,
        x_bins: NDArray64,
        x_off: NDArray64,
    ) -> NDArray64:
        stacked = jax.vmap(
            lambda off: jbin_y_over_x(x - off, y, x_bins),
        )(x_off)
        return jnp.swapaxes(stacked, 0, 1)

    @jax.jit
    def _metric_from_offsets(
        x: NDArray64,
        y: NDArray64,
        x_bins: NDArray64,
        x_off: NDArray64,
        scale: float,
    ) -> NDArray64:
        y_binned = _bin_offsets(x=x, y=y, x_bins=x_bins, x_off=x_off)
        dydx = jnp.gradient(y_binned, x_bins, axis=0)
        y_sym = jnp.abs(dydx / scale - jnp.flip(dydx / scale, axis=0))
        return jnp.nanmean(y_sym, axis=0)

    return jax, _metric_from_offsets


@dataclass(slots=True)
class OffsetSpec:
    """Configuration for offset analysis on IV traces."""

    Vbins_mV: Sequence[float] | NDArray64
    Ibins_nA: Sequence[float] | NDArray64
    Voff_mV: Sequence[float] | NDArray64
    Ioff_nA: Sequence[float] | NDArray64
    nu_Hz: float
    upsample: int = 10

    def __post_init__(self) -> None:
        """Normalize grids and validate scalar settings."""
        self.Vbins_mV = to_1d_float64(self.Vbins_mV, "Vbins_mV")
        self.Ibins_nA = to_1d_float64(self.Ibins_nA, "Ibins_nA")
        self.Voff_mV = to_1d_float64(self.Voff_mV, "Voff_mV")
        self.Ioff_nA = to_1d_float64(self.Ioff_nA, "Ioff_nA")

        require_min_size(self.Vbins_mV, 2, name="Vbins_mV")
        require_min_size(self.Ibins_nA, 2, name="Ibins_nA")
        require_min_size(self.Voff_mV, 1, name="Voff_mV")
        require_min_size(self.Ioff_nA, 1, name="Ioff_nA")

        require_all_finite(self.Vbins_mV, name="Vbins_mV")
        require_all_finite(self.Ibins_nA, name="Ibins_nA")
        require_all_finite(self.Voff_mV, name="Voff_mV")
        require_all_finite(self.Ioff_nA, name="Ioff_nA")

        nu_Hz = float(self.nu_Hz)
        if not np.isfinite(nu_Hz) or nu_Hz <= 0.0:
            raise ValueError("nu_Hz must be finite and > 0.")
        self.nu_Hz = nu_Hz

        upsample = int(self.upsample)
        if upsample <= 0:
            raise ValueError("upsample must be > 0.")
        self.upsample = upsample

    @property
    def dt_s(self) -> float:
        """Temporary time-grid spacing used for offset analysis.

        Notes
        -----
        ``nu_Hz`` is interpreted as the literal sampling rate, so the
        spacing is ``dt_s = 1 / nu_Hz``.
        """
        return 1.0 / self.nu_Hz


class OffsetTrace(TypedDict):
    """One offset-analysis result with metadata."""

    specific_key: str
    index: int | None
    yvalue: float | None
    dGerr_G0: NDArray64
    dRerr_R0: NDArray64
    Voff_mV: float
    Ioff_nA: float


@dataclass(slots=True)
class OffsetTraces:
    """Container for multiple offset-analysis results."""

    spec: OffsetSpec
    traces: list[OffsetTrace]
    keys: list[str] = field(init=False)
    yvalues: NDArray64 = field(init=False)
    dGerr_G0: NDArray64 = field(init=False)
    dRerr_R0: NDArray64 = field(init=False)
    Voff_mV: NDArray64 = field(init=False)
    Ioff_nA: NDArray64 = field(init=False)
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
        dg_rows: list[NDArray64] = []
        dr_rows: list[NDArray64] = []
        indices_by_key: dict[str, list[int]] = {}

        for index, trace in enumerate(self.traces):
            specific_key = trace["specific_key"]
            self.keys.append(specific_key)
            yvalue = trace["yvalue"]
            yvalues.append(np.nan if yvalue is None else float(yvalue))
            voffs_mV.append(float(trace["Voff_mV"]))
            ioffs_nA.append(float(trace["Ioff_nA"]))
            dg_rows.append(np.asarray(trace["dGerr_G0"], dtype=np.float64))
            dr_rows.append(np.asarray(trace["dRerr_R0"], dtype=np.float64))
            indices_by_key.setdefault(specific_key, []).append(index)

        self.yvalues = np.asarray(yvalues, dtype=np.float64)
        self.Voff_mV = np.asarray(voffs_mV, dtype=np.float64)
        self.Ioff_nA = np.asarray(ioffs_nA, dtype=np.float64)
        self.dGerr_G0 = np.vstack(dg_rows)
        self.dRerr_R0 = np.vstack(dr_rows)
        self._indices_by_key = indices_by_key

    def __len__(self) -> int:
        """Return number of traces."""
        return len(self.traces)

    def __iter__(self) -> Iterator[OffsetTrace]:
        """Iterate over traces."""
        return iter(self.traces)

    def __getitem__(
        self,
        index: int | slice,
    ) -> OffsetTrace | list[OffsetTrace]:
        """Return trace(s) by positional index."""
        return self.traces[index]

    def all_by_key(
        self,
        specific_key: str,
    ) -> list[OffsetTrace]:
        """Return all traces with one exact specific key."""
        indices = self._indices_by_key.get(specific_key, [])
        return [self.traces[index] for index in indices]

    def by_key(
        self,
        specific_key: str,
    ) -> OffsetTrace:
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
    ) -> list[OffsetTrace]:
        """Return all traces with one y-value."""
        indices = self._find_indices_by_value(yvalue)
        return [self.traces[index] for index in indices]

    def by_value(
        self,
        yvalue: float,
    ) -> OffsetTrace:
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
        matches: list[OffsetTrace],
        selector_name: str,
        selector_value: str | float,
        plural_hint: str,
    ) -> OffsetTrace:
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


def _bin_y_over_x_offsets(
    x: NDArray64,
    y: NDArray64,
    x_bins: NDArray64,
    x_off: NDArray64,
) -> NDArray64:
    """Bin ``y(x - x_off[k])`` onto ``x_bins`` for all offsets."""
    out = np.full((x_bins.size, x_off.size), np.nan, dtype=np.float64)
    for j_off, off in enumerate(x_off):
        out[:, j_off] = bin_y_over_x(x=x - off, y=y, x_bins=x_bins)
    return out


def _nanargmin_finite(values: NDArray64) -> int:
    """Return index of smallest finite value, fallback to center index."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return int(values.size // 2)
    idx_finite = np.where(finite)[0]
    return int(idx_finite[np.argmin(values[finite])])


def _prepare_trace_for_offset(
    trace: IVTrace,
    spec: OffsetSpec,
) -> tuple[NDArray64, NDArray64]:
    """Downsample in time, then upsample in index for offset analysis."""
    _, v_down_mV, i_down_nA = _downsample_iv_trace(
        trace=trace,
        nu_Hz=spec.nu_Hz,
    )

    v_mV, i_nA = upsample_xy(
        x=v_down_mV,
        y=i_down_nA,
        factor=spec.upsample,
        method="linear",
    )
    return (
        np.asarray(v_mV, dtype=np.float64),
        np.asarray(i_nA, dtype=np.float64),
    )


def _resolve_backend(
    backend: str,
    workers: int = 1,
) -> str:
    """Resolve backend selection with optional JAX fallback."""
    key = backend.strip().lower()
    if key == "auto":
        if int(workers) != 1:
            return "numpy"
        try:
            _import_jax_offset_kernels()
        except ImportError:
            return "numpy"
        return "jax"
    if key not in {"numpy", "jax"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'jax'.")
    if key == "jax":
        if int(workers) != 1:
            raise ValueError(
                "workers > 1 is not supported with backend='jax'. "
                "Use backend='numpy' for trace-level parallelism.",
            )
        _import_jax_offset_kernels()
    return key


def _compute_offset_errors_numpy(
    v_mV: NDArray64,
    i_nA: NDArray64,
    spec: OffsetSpec,
) -> tuple[NDArray64, NDArray64]:
    """Compute offset objective arrays with the NumPy implementation."""
    i_vs_v = _bin_y_over_x_offsets(
        x=v_mV,
        y=i_nA,
        x_bins=spec.Vbins_mV,
        x_off=spec.Voff_mV,
    )
    g_uS = np.gradient(i_vs_v, spec.Vbins_mV, axis=0)
    g_G0 = g_uS / G_0_muS
    g_sym = np.abs(g_G0 - np.flip(g_G0, axis=0))
    g_err_G0 = np.nanmean(g_sym, axis=0)
    g_err_G0 = fill_nans(g_err_G0, x=spec.Voff_mV, method="linear")

    v_vs_i = _bin_y_over_x_offsets(
        x=i_nA,
        y=v_mV,
        x_bins=spec.Ibins_nA,
        x_off=spec.Ioff_nA,
    )
    r_MOhm = np.gradient(v_vs_i, spec.Ibins_nA, axis=0)
    r_R0 = r_MOhm * G_0_muS
    r_sym = np.abs(r_R0 - np.flip(r_R0, axis=0))
    r_err_R0 = np.nanmean(r_sym, axis=0)
    r_err_R0 = fill_nans(r_err_R0, x=spec.Ioff_nA, method="linear")
    return (
        np.asarray(g_err_G0, dtype=np.float64),
        np.asarray(r_err_R0, dtype=np.float64),
    )


def _compute_offset_errors_jax(
    v_mV: NDArray64,
    i_nA: NDArray64,
    spec: OffsetSpec,
) -> tuple[NDArray64, NDArray64]:
    """Compute offset objective arrays with the JAX implementation."""
    jax, metric_from_offsets = _import_jax_offset_kernels()

    g_err_G0 = metric_from_offsets(
        x=v_mV,
        y=i_nA,
        x_bins=spec.Vbins_mV,
        x_off=spec.Voff_mV,
        scale=G_0_muS,
    )
    r_err_R0 = metric_from_offsets(
        x=i_nA,
        y=v_mV,
        x_bins=spec.Ibins_nA,
        x_off=spec.Ioff_nA,
        scale=1.0 / G_0_muS,
    )
    g_err_G0_np = fill_nans(
        np.asarray(jax.device_get(g_err_G0), dtype=np.float64),
        x=spec.Voff_mV,
        method="linear",
    )
    r_err_R0_np = fill_nans(
        np.asarray(jax.device_get(r_err_R0), dtype=np.float64),
        x=spec.Ioff_nA,
        method="linear",
    )
    return (
        np.asarray(g_err_G0_np, dtype=np.float64),
        np.asarray(r_err_R0_np, dtype=np.float64),
    )


def get_offset(
    trace: IVTrace,
    spec: OffsetSpec,
    backend: str = "numpy",
) -> OffsetTrace:
    """Find one per-trace offset via symmetry of ``G(V)`` and ``R(I)``.

    Notes
    -----
    Use ``backend="jax"`` to evaluate the offset grids with the JAX
    histogram kernel. The default remains ``"numpy"`` to avoid JAX compile
    overhead on small calls.
    """
    v_mV, i_nA = _prepare_trace_for_offset(trace=trace, spec=spec)
    backend_key = _resolve_backend(backend, workers=1)
    if backend_key == "jax":
        g_err_G0, r_err_R0 = _compute_offset_errors_jax(
            v_mV=v_mV,
            i_nA=i_nA,
            spec=spec,
        )
    else:
        g_err_G0, r_err_R0 = _compute_offset_errors_numpy(
            v_mV=v_mV,
            i_nA=i_nA,
            spec=spec,
        )

    j_v = _nanargmin_finite(g_err_G0)
    j_i = _nanargmin_finite(r_err_R0)
    return {
        "specific_key": trace["specific_key"],
        "index": trace["index"],
        "yvalue": trace["yvalue"],
        "dGerr_G0": np.asarray(g_err_G0, dtype=np.float64),
        "dRerr_R0": np.asarray(r_err_R0, dtype=np.float64),
        "Voff_mV": float(spec.Voff_mV[j_v]),
        "Ioff_nA": float(spec.Ioff_nA[j_i]),
    }


def get_offsets(
    traces: IVTraces,
    spec: OffsetSpec,
    show_progress: bool = True,
    backend: str = "jax",
    workers: int = 1,
) -> OffsetTraces:
    """Find offsets for one collection of IV traces.

    Notes
    -----
    ``workers`` parallelizes across traces for the NumPy backend. JAX is kept
    single-worker because it already vectorizes over the offset grid and
    competing thread pools usually slow it down.
    """
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be > 0.")
    backend_key = _resolve_backend(backend, workers=worker_count)

    trace_iterable: Iterator[IVTrace] | IVTraces
    trace_iterable = traces
    if show_progress:
        tqdm = _import_tqdm()
        trace_iterable = tqdm(
            traces,
            total=len(traces),
            desc="get_offsets",
            unit="trace",
        )

    compute_one = partial(get_offset, spec=spec, backend=backend_key)
    if worker_count == 1:
        out_traces = [compute_one(trace) for trace in trace_iterable]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            out_traces = list(executor.map(compute_one, trace_iterable))

    return OffsetTraces(spec=spec, traces=out_traces)


__all__ = [
    "OffsetSpec",
    "OffsetTrace",
    "OffsetTraces",
    "get_offset",
    "get_offsets",
]
]
]
