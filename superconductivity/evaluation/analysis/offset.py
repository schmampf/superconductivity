"""Helpers for offset analysis derived from IV traces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Iterator, Sequence, TypedDict, overload

import numpy as np

from ...utilities.constants import G0_muS
from ...utilities.functions.binning import bin
from ...utilities.functions.fill_nans import fill as fill_nans
from ...utilities.functions.upsampling import upsample as upsample_xy
from ...utilities.meta.axis import AxisSpec
from ...utilities.meta.param import ParamSpec
from ...utilities.safety import require_all_finite, require_min_size, to_1d_float64
from ...utilities.types import NDArray64
from ..sampling import downsample_trace
from ..traces import Trace, Traces


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

    from ...utilities.legacy.functions_jax import jbin_y_over_x

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

    Vbins_mV: AxisSpec | Sequence[float] | NDArray64
    Ibins_nA: AxisSpec | Sequence[float] | NDArray64
    Voff_mV: AxisSpec | Sequence[float] | NDArray64
    Ioff_nA: AxisSpec | Sequence[float] | NDArray64
    nu_Hz: ParamSpec | float
    N_up: ParamSpec | int = 10

    def __post_init__(self) -> None:
        """Normalize grids and validate scalar settings."""
        self.Vbins_mV = self._coerce_axis_spec("Vbins_mV", self.Vbins_mV)
        self.Ibins_nA = self._coerce_axis_spec("Ibins_nA", self.Ibins_nA)
        self.Voff_mV = self._coerce_axis_spec("Voff_mV", self.Voff_mV)
        self.Ioff_nA = self._coerce_axis_spec("Ioff_nA", self.Ioff_nA)

        self.nu_Hz = self._coerce_param_spec("nu_Hz", self.nu_Hz)
        self.N_up = self._coerce_param_spec("N_up", self.N_up)

        nu_Hz = float(self.nu_Hz.values)
        if not np.isfinite(nu_Hz) or nu_Hz <= 0.0:
            raise ValueError("nu_Hz must be finite and > 0.")
        n_up = int(self.N_up.values)
        if n_up <= 0:
            raise ValueError("N_up must be > 0.")
        if float(self.nu_Hz.values) != nu_Hz:
            self.nu_Hz = ParamSpec(
                values=nu_Hz,
                code_label=self.nu_Hz.code_label,
                print_label=self.nu_Hz.print_label,
                html_label=self.nu_Hz.html_label,
                latex_label=self.nu_Hz.latex_label,
                error=self.nu_Hz.error,
                lower=self.nu_Hz.lower,
                upper=self.nu_Hz.upper,
                fixed=self.nu_Hz.fixed,
            )
        if int(self.N_up.values) != n_up:
            self.N_up = ParamSpec(
                values=n_up,
                code_label=self.N_up.code_label,
                print_label=self.N_up.print_label,
                html_label=self.N_up.html_label,
                latex_label=self.N_up.latex_label,
                error=self.N_up.error,
                lower=self.N_up.lower,
                upper=self.N_up.upper,
                fixed=self.N_up.fixed,
            )

    @staticmethod
    def _coerce_axis_spec(
        name: str,
        value: AxisSpec | Sequence[float] | NDArray64,
    ) -> AxisSpec:
        """Normalize one axis input to one ``AxisSpec``."""
        if isinstance(value, AxisSpec):
            if value.order != 1:
                raise ValueError(f"{name} must have order 1.")
            return value

        values = to_1d_float64(value, name)
        require_min_size(values, 2, name=name)
        require_all_finite(values, name=name)
        return AxisSpec(
            values=values,
            code_label=name,
            print_label=name,
            html_label=name,
            latex_label=name,
            order=1,
        )

    @staticmethod
    def _coerce_param_spec(
        name: str,
        value: ParamSpec | float | int,
    ) -> ParamSpec:
        """Normalize one scalar setting to one ``ParamSpec``."""
        if isinstance(value, ParamSpec):
            return value

        return ParamSpec(
            values=value,
            code_label=name,
            print_label=name,
            html_label=name,
            latex_label=name,
            error=None,
            lower=None,
            upper=None,
            fixed=True,
        )


class OffsetTrace(TypedDict):
    """One offset-analysis result."""

    dGerr_G0: NDArray64
    dRerr_R0: NDArray64
    Voff_mV: float
    Ioff_nA: float


@dataclass(slots=True)
class OffsetTraces:
    """Container for multiple offset-analysis results."""

    traces: list[OffsetTrace]
    dGerr_G0: NDArray64 = field(init=False)
    dRerr_R0: NDArray64 = field(init=False)
    Voff_mV: NDArray64 = field(init=False)
    Ioff_nA: NDArray64 = field(init=False)

    def __post_init__(self) -> None:
        """Build stacked arrays from ``traces``."""
        if len(self.traces) == 0:
            raise ValueError("traces must not be empty.")

        voffs_mV: list[float] = []
        ioffs_nA: list[float] = []
        dg_rows: list[NDArray64] = []
        dr_rows: list[NDArray64] = []

        for trace in self.traces:
            voffs_mV.append(float(trace["Voff_mV"]))
            ioffs_nA.append(float(trace["Ioff_nA"]))
            dg_rows.append(np.asarray(trace["dGerr_G0"], dtype=np.float64))
            dr_rows.append(np.asarray(trace["dRerr_R0"], dtype=np.float64))

        self.Voff_mV = np.asarray(voffs_mV, dtype=np.float64)
        self.Ioff_nA = np.asarray(ioffs_nA, dtype=np.float64)
        self.dGerr_G0 = np.vstack(dg_rows)
        self.dRerr_R0 = np.vstack(dr_rows)

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


def _bin_y_over_x_offsets(
    x: NDArray64,
    y: NDArray64,
    x_bins: NDArray64,
    x_off: NDArray64,
) -> NDArray64:
    """Bin ``y(x - x_off[k])`` onto ``x_bins`` for all offsets."""
    out = np.full((x_bins.size, x_off.size), np.nan, dtype=np.float64)
    for j_off, off in enumerate(x_off):
        out[:, j_off] = bin(z=y, x=x - off, xbins=x_bins)
    return out


def _nanargmin_finite(values: NDArray64) -> int:
    """Return index of smallest finite value, fallback to center index."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return int(values.size // 2)
    idx_finite = np.where(finite)[0]
    return int(idx_finite[np.argmin(values[finite])])


def _prepare_trace_for_offset(
    trace: Trace,
    spec: OffsetSpec,
) -> tuple[NDArray64, NDArray64]:
    """Downsample in time, then upsample in index for offset analysis."""
    downsampled = downsample_trace(trace, nu_Hz=spec.nu_Hz)
    v_down_mV = np.asarray(downsampled["V_mV"], dtype=np.float64)
    i_down_nA = np.asarray(downsampled["I_nA"], dtype=np.float64)

    v_mV = upsample_xy(
        v_down_mV,
        N_up=int(spec.N_up.values),
        axis=0,
        method="linear",
    )
    i_nA = upsample_xy(
        i_down_nA,
        N_up=int(spec.N_up.values),
        axis=0,
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
        x_bins=spec.Vbins_mV.values,
        x_off=spec.Voff_mV.values,
    )
    g_uS = np.gradient(i_vs_v, spec.Vbins_mV.values, axis=0)
    g_G0 = g_uS / G0_muS
    g_sym = np.abs(g_G0 - np.flip(g_G0, axis=0))
    g_err_G0 = np.nanmean(g_sym, axis=0)
    g_err_G0 = fill_nans(g_err_G0, method="interpolate")

    v_vs_i = _bin_y_over_x_offsets(
        x=i_nA,
        y=v_mV,
        x_bins=spec.Ibins_nA.values,
        x_off=spec.Ioff_nA.values,
    )
    r_MOhm = np.gradient(v_vs_i, spec.Ibins_nA.values, axis=0)
    r_R0 = r_MOhm * G0_muS
    r_sym = np.abs(r_R0 - np.flip(r_R0, axis=0))
    r_err_R0 = np.nanmean(r_sym, axis=0)
    r_err_R0 = fill_nans(r_err_R0, method="interpolate")
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
        x_bins=spec.Vbins_mV.values,
        x_off=spec.Voff_mV.values,
        scale=float(G0_muS),
    )
    r_err_R0 = metric_from_offsets(
        x=i_nA,
        y=v_mV,
        x_bins=spec.Ibins_nA.values,
        x_off=spec.Ioff_nA.values,
        scale=1.0 / float(G0_muS),
    )
    g_err_G0_np = fill_nans(
        np.asarray(jax.device_get(g_err_G0), dtype=np.float64),
        method="interpolate",
    )
    r_err_R0_np = fill_nans(
        np.asarray(jax.device_get(r_err_R0), dtype=np.float64),
        method="interpolate",
    )
    return (
        np.asarray(g_err_G0_np, dtype=np.float64),
        np.asarray(r_err_R0_np, dtype=np.float64),
    )


def _offset_analysis_one(
    trace: Trace,
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
        "dGerr_G0": np.asarray(g_err_G0, dtype=np.float64),
        "dRerr_R0": np.asarray(r_err_R0, dtype=np.float64),
        "Voff_mV": float(spec.Voff_mV.values[j_v]),
        "Ioff_nA": float(spec.Ioff_nA.values[j_i]),
    }


@overload
def offset_analysis(
    traces: Traces,
    spec: OffsetSpec,
    show_progress: bool = True,
    backend: str = "jax",
    workers: int = 1,
) -> OffsetTraces: ...


@overload
def offset_analysis(
    traces: Trace,
    spec: OffsetSpec,
    show_progress: bool = True,
    backend: str = "jax",
    workers: int = 1,
) -> OffsetTrace: ...


def offset_analysis(
    traces: Trace | Traces,
    spec: OffsetSpec,
    show_progress: bool = True,
    backend: str = "jax",
    workers: int = 1,
) -> OffsetTrace | OffsetTraces:
    """Run offset analysis for one trace or one collection.

    Notes
    -----
    ``workers`` parallelizes across traces for the NumPy backend. JAX is kept
    single-worker because it already vectorizes over the offset grid and
    competing thread pools usually slow it down.
    """
    if not isinstance(traces, Traces):
        _ = show_progress, workers
        return _offset_analysis_one(traces, spec=spec, backend=backend)

    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be > 0.")
    backend_key = _resolve_backend(backend, workers=worker_count)

    trace_iterable: Iterator[Trace] | Traces
    trace_iterable = traces
    if show_progress:
        tqdm = _import_tqdm()
        trace_iterable = tqdm(
            traces,
            total=len(traces),
            desc="offset_analysis",
            unit="trace",
        )

    compute_one = partial(_offset_analysis_one, spec=spec, backend=backend_key)
    if worker_count == 1:
        out_traces = [compute_one(trace) for trace in trace_iterable]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            out_traces = list(executor.map(compute_one, trace_iterable))

    return OffsetTraces(
        traces=out_traces,
    )


__all__ = [
    "OffsetSpec",
    "OffsetTrace",
    "OffsetTraces",
    "offset_analysis",
]
