"""Thesis-oriented surface panel helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from superconductivity.style.cpd4 import cmap as default_cmap
from superconductivity.utilities.types import LIM, NDArray64

from ._common import (
    apply_matplotlib_axis_ticks,
    apply_matplotlib_axis_spacing,
    apply_matplotlib_zaxis_side,
    prepare_waterfall_data,
    redraw_matplotlib_axes_on_top,
    resolve_axis_range,
    resolve_axes_rect_inches,
    resolve_color_range,
    set_matplotlib_axis_visible,
    set_matplotlib_background_style,
    set_matplotlib_box_aspect,
)


def _resolve_axis_spacing(values: NDArray64) -> float:
    """Return a representative positive spacing for one sampled axis."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        return 1.0

    diffs = np.abs(np.diff(arr))
    finite = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def _surface_facecolors(
    *,
    z: NDArray64,
    x: NDArray64,
    y: NDArray64,
    cmap_mpl: ListedColormap,
    norm: mcolors.Normalize,
    alpha: float,
    shading: bool,
    light_azdeg: float,
    light_altdeg: float,
    shade_strength: float,
) -> np.ndarray:
    """Build per-vertex surface facecolors with optional light shading."""
    rgba = np.asarray(cmap_mpl(norm(z)), dtype=np.float64)
    rgba[..., 3] *= float(np.clip(alpha, 0.0, 1.0))
    if not shading:
        return rgba

    light_source = mcolors.LightSource(
        azdeg=float(light_azdeg),
        altdeg=float(light_altdeg),
    )
    shaded_rgb = light_source.shade_rgb(
        rgba[..., :3],
        np.asarray(z, dtype=np.float64),
        dx=_resolve_axis_spacing(x),
        dy=_resolve_axis_spacing(y),
        blend_mode="overlay",
    )
    strength = float(np.clip(shade_strength, 0.0, 1.0))
    rgba[..., :3] = (
        (1.0 - strength) * rgba[..., :3] + strength * shaded_rgb
    )
    return np.clip(rgba, 0.0, 1.0)


def _resolve_trace_band_span(
    *,
    local_gap: float,
    trace_width: float,
) -> float:
    """Return the local y-span used for one highlighted trace band."""
    width_scale = 4.0 * max(float(trace_width), 0.0)
    return float(
        min(
            0.24 * float(local_gap) * width_scale,
            0.9 * float(local_gap),
        )
    )


def _build_surface_trace_strips(
    y: NDArray64,
    z: NDArray64,
    *,
    trace_y: NDArray64,
    trace_width: float,
) -> list[tuple[NDArray64, NDArray64]]:
    """Build narrow surface strips for selected trace rows."""
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    traces = np.asarray(trace_y, dtype=np.float64).reshape(-1)
    if y_arr.size < 2 or traces.size == 0:
        return []

    diffs = np.diff(y_arr)
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError("surface y values must be strictly monotonic.")

    descending = bool(diffs[0] < 0.0)
    y_base = y_arr[::-1] if descending else y_arr
    z_base = z_arr[::-1, :] if descending else z_arr
    traces_base = traces[::-1] if descending else traces
    eps = max(1e-12, 1e-9 * max(float(np.ptp(y_base)), 1.0))
    strips: list[tuple[NDArray64, NDArray64]] = []

    for value in traces_base:
        idx = int(np.argmin(np.abs(y_base - float(value))))
        center = float(y_base[idx])
        if idx == 0:
            local_gap = float(abs(y_base[1] - y_base[0]))
        elif idx == y_base.size - 1:
            local_gap = float(abs(y_base[-1] - y_base[-2]))
        else:
            local_gap = float(
                min(
                    abs(y_base[idx] - y_base[idx - 1]),
                    abs(y_base[idx + 1] - y_base[idx]),
                )
            )
        span_width = _resolve_trace_band_span(
            local_gap=local_gap,
            trace_width=trace_width,
        )
        halfwidth = 0.5 * span_width
        if idx == 0:
            lower = center - halfwidth
            upper = min(center + halfwidth, y_base[idx + 1] - eps)
        elif idx == y_base.size - 1:
            lower = max(center - halfwidth, y_base[idx - 1] + eps)
            upper = center + halfwidth
        else:
            lower = max(center - halfwidth, y_base[idx - 1] + eps)
            upper = min(center + halfwidth, y_base[idx + 1] - eps)
        if lower >= upper:
            continue
        y_strip = np.asarray((lower, upper), dtype=np.float64)
        z_strip = np.empty((2, z_base.shape[1]), dtype=np.float64)
        for column_index in range(z_base.shape[1]):
            z_strip[:, column_index] = np.interp(
                y_strip,
                y_base,
                z_base[:, column_index],
            )
        if descending:
            strips.append((y_strip[::-1], z_strip[::-1, :]))
        else:
            strips.append((y_strip, z_strip))

    return strips


def _build_surface_trace_bands(
    y: NDArray64,
    *,
    trace_y: NDArray64,
    trace_width: float,
) -> list[tuple[NDArray64, NDArray64, int]]:
    """Return local y refinement rows and row weights for trace bands."""
    y_arr = np.asarray(y, dtype=np.float64)
    traces = np.asarray(trace_y, dtype=np.float64).reshape(-1)
    if y_arr.size < 2 or traces.size == 0:
        return []
    diffs = np.diff(y_arr)
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError("surface y values must be strictly monotonic.")
    eps = max(1e-12, 1e-9 * max(float(np.ptp(y_arr)), 1.0))
    bands: list[tuple[NDArray64, NDArray64, int]] = []

    for value in traces:
        idx = int(np.argmin(np.abs(y_arr - float(value))))
        center = float(y_arr[idx])
        if idx == 0:
            local_gap = float(abs(y_arr[1] - y_arr[0]))
            span_width = _resolve_trace_band_span(
                local_gap=local_gap,
                trace_width=trace_width,
            )
            if span_width <= eps:
                continue
            rows = center + span_width * np.asarray(
                [0.0, 0.25, 0.5, 0.75, 1.0],
                dtype=np.float64,
            )
            rows[-1] = min(rows[-1], float(y_arr[1] - eps))
            weights = np.asarray(
                [1.0, 0.72, 0.42, 0.16, 0.04],
                dtype=np.float64,
            )
            center_index = 0
        elif idx == y_arr.size - 1:
            local_gap = float(abs(y_arr[-1] - y_arr[-2]))
            span_width = _resolve_trace_band_span(
                local_gap=local_gap,
                trace_width=trace_width,
            )
            if span_width <= eps:
                continue
            rows = center - span_width * np.asarray(
                [1.0, 0.75, 0.5, 0.25, 0.0],
                dtype=np.float64,
            )
            rows[0] = max(rows[0], float(y_arr[-2] + eps))
            weights = np.asarray(
                [0.04, 0.16, 0.42, 0.72, 1.0],
                dtype=np.float64,
            )
            center_index = 4
        else:
            local_gap = float(
                min(
                    abs(center - y_arr[idx - 1]),
                    abs(y_arr[idx + 1] - center),
                )
            )
            span_width = _resolve_trace_band_span(
                local_gap=local_gap,
                trace_width=trace_width,
            )
            if span_width <= eps:
                continue
            inner = 0.22 * span_width
            outer = 0.5 * span_width
            rows = np.asarray(
                [
                    center - outer,
                    center - inner,
                    center,
                    center + inner,
                    center + outer,
                ],
                dtype=np.float64,
            )
            rows[0] = max(rows[0], float(y_arr[idx - 1] + eps))
            rows[-1] = min(rows[-1], float(y_arr[idx + 1] - eps))
            weights = np.asarray(
                [0.08, 0.55, 1.0, 0.55, 0.08],
                dtype=np.float64,
            )
            center_index = 2
        bands.append((rows, weights, center_index))

    return bands


def _embed_surface_trace_bands(
    *,
    y: NDArray64,
    z: NDArray64,
    trace_y: NDArray64,
    trace_z: NDArray64,
    trace_width: float,
) -> tuple[NDArray64, NDArray64, NDArray64]:
    """Embed highlighted traces into a locally refined surface mesh."""
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    trace_y_arr = np.asarray(trace_y, dtype=np.float64).reshape(-1)
    trace_z_arr = np.asarray(trace_z, dtype=np.float64)
    if y_arr.size < 2 or trace_y_arr.size == 0:
        return (
            y_arr,
            z_arr,
            np.zeros(max(y_arr.size - 1, 0), dtype=np.float64),
        )
    diffs = np.diff(y_arr)
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError("surface y values must be strictly monotonic.")

    descending = bool(diffs[0] < 0.0)
    y_base = y_arr[::-1] if descending else y_arr
    z_base = z_arr[::-1, :] if descending else z_arr
    trace_y_base = trace_y_arr[::-1] if descending else trace_y_arr
    trace_z_base = trace_z_arr[::-1, :] if descending else trace_z_arr
    bands = _build_surface_trace_bands(
        y_base,
        trace_y=trace_y_base,
        trace_width=trace_width,
    )
    if not bands:
        return (
            y_arr,
            z_arr,
            np.zeros(max(y_arr.size - 1, 0), dtype=np.float64),
        )

    y_refined = np.unique(
        np.concatenate(
            [y_base]
            + [band_rows for band_rows, _, _ in bands]
        )
    )
    z_refined = np.empty(
        (y_refined.size, z_base.shape[1]),
        dtype=np.float64,
    )
    for col_index in range(z_base.shape[1]):
        z_refined[:, col_index] = np.interp(
            y_refined,
            y_base,
            z_base[:, col_index],
        )
    cell_weights = np.zeros(max(y_refined.size - 1, 0), dtype=np.float64)

    for (band_rows, band_weights, center_index), trace_row in zip(
        bands,
        trace_z_base,
        strict=False,
    ):
        row_indices = np.searchsorted(y_refined, band_rows)
        row_indices = np.clip(row_indices, 0, y_refined.size - 1)
        z_refined[row_indices[center_index], :] = np.asarray(
            trace_row,
            dtype=np.float64,
        )
        del band_weights
        if center_index == 0:
            cell_indices = row_indices[:2]
        elif center_index == row_indices.size - 1:
            cell_indices = row_indices[-3:-1]
        else:
            cell_indices = row_indices[1:3]
        cell_indices = np.clip(cell_indices, 0, cell_weights.size - 1)
        cell_weights[cell_indices] = 1.0

    if descending:
        return (
            y_refined[::-1],
            z_refined[::-1, :],
            cell_weights[::-1],
        )
    return y_refined, z_refined, cell_weights


def _surface_cell_facecolors(
    facecolors: np.ndarray,
) -> np.ndarray:
    """Convert per-vertex facecolors to per-cell facecolors."""
    rgba = np.asarray(facecolors, dtype=np.float64)
    return np.clip(
        0.25
        * (
            rgba[:-1, :-1, :]
            + rgba[1:, :-1, :]
            + rgba[:-1, 1:, :]
            + rgba[1:, 1:, :]
        ),
        0.0,
        1.0,
    )


def _blend_surface_trace_facecolors(
    facecolors: np.ndarray,
    *,
    cell_weights: NDArray64,
    trace_color: str,
    trace_alpha: float,
) -> np.ndarray:
    """Blend highlighted trace cells into surface facecolors."""
    rgba = np.asarray(facecolors, dtype=np.float64).copy()
    weights = np.asarray(cell_weights, dtype=np.float64).reshape(-1, 1, 1)
    blend = np.clip(
        weights * float(np.clip(trace_alpha, 0.0, 1.0)),
        0.0,
        1.0,
    )
    trace_rgb = np.asarray(
        mcolors.to_rgb(trace_color),
        dtype=np.float64,
    ).reshape(1, 1, 3)
    rgba[..., :3] = (
        (1.0 - blend) * rgba[..., :3]
        + blend * trace_rgb
    )
    return np.clip(rgba, 0.0, 1.0)


def _interp_rows_to_x(
    x: NDArray64,
    rows: NDArray64,
    target_x: NDArray64,
) -> NDArray64:
    """Interpolate one or more rows onto a target x grid."""
    x_arr = np.asarray(x, dtype=np.float64)
    rows_arr = np.asarray(rows, dtype=np.float64)
    target_arr = np.asarray(target_x, dtype=np.float64)

    descending = bool(x_arr.size >= 2 and x_arr[0] > x_arr[-1])
    if descending:
        source_x = x_arr[::-1]
        source_rows = rows_arr[:, ::-1]
        target_base = target_arr[::-1]
    else:
        source_x = x_arr
        source_rows = rows_arr
        target_base = target_arr

    interpolated = np.empty(
        (rows_arr.shape[0], target_arr.size),
        dtype=np.float64,
    )
    for row_index, row in enumerate(source_rows):
        row_interp = np.interp(target_base, source_x, row)
        interpolated[row_index] = (
            row_interp[::-1] if descending else row_interp
        )
    return interpolated


def _oversample_surface_x(
    x: NDArray64,
    z: NDArray64,
    *,
    factor: int,
) -> tuple[NDArray64, NDArray64]:
    """Densify a regular surface mesh along x only."""
    oversample = int(factor)
    if oversample < 1:
        raise ValueError("surface_x_oversample must be at least 1.")

    x_arr = np.asarray(x, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    if oversample == 1 or x_arr.size < 2:
        return x_arr, z_arr

    diffs = np.diff(x_arr)
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError("surface x values must be strictly monotonic.")

    descending = bool(diffs[0] < 0.0)
    x_base = x_arr[::-1] if descending else x_arr
    z_base = z_arr[:, ::-1] if descending else z_arr
    pieces = [
        np.linspace(
            x_base[index],
            x_base[index + 1],
            oversample + 1,
            dtype=np.float64,
        )[:-1]
        for index in range(x_base.size - 1)
    ]
    x_dense = np.concatenate(
        (
            *pieces,
            np.array([x_base[-1]], dtype=np.float64),
        )
    )
    z_dense = np.empty((z_base.shape[0], x_dense.size), dtype=np.float64)
    for row_index, row in enumerate(z_base):
        z_dense[row_index] = np.interp(x_dense, x_base, row)

    if descending:
        return x_dense[::-1], z_dense[:, ::-1]
    return x_dense, z_dense


def _resolve_trace_strip_lift(
    z_values: NDArray64,
) -> float:
    """Return a tiny z-lift so trace strips do not z-fight the surface."""
    arr = np.asarray(z_values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1e-9
    span = float(np.nanmax(finite) - np.nanmin(finite))
    if span == 0.0:
        return 1e-9
    return max(1e-9, 6e-3 * span)


def get_thesis_surface_matplotlib(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
    *,
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    clim: LIM = None,
    trace_step: int = 1,
    surface_x_oversample: int = 10,
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    show_xaxis: bool = True,
    show_yaxis: bool = True,
    show_zaxis: bool = True,
    z_axis_side: str = "default",
    invert_xaxis: bool = False,
    invert_yaxis: bool = True,
    labelspacing: float | Sequence[float] | None = None,
    ticklabelspacing: float | Sequence[float] | None = None,
    ticks: Sequence[object | None] | None = None,
    ticklabels: Sequence[object | None] | None = None,
    clabel: Optional[str] = None,
    trace_y: NDArray64 | None = None,
    trace_z: NDArray64 | None = None,
    trace_color: str = "black",
    trace_width: float = 0.6,
    trace_alpha: float = 1.0,
    surface_alpha: float = 1.0,
    surface_shading: bool = True,
    surface_light_azdeg: float = 315.0,
    surface_light_altdeg: float = 40.0,
    surface_shade_strength: float = 0.5,
    surface_rasterized: bool = True,
    cmap_mpl: ListedColormap = default_cmap(),
    show_colorbar: bool = False,
    box_aspect: Sequence[float] | None = (1.0, 1.0, 1.0),
    box_zoom: float = 1.0,
    elev: float = 20.0,
    azim: float = -70.0,
    figsize: tuple[float, float] = (6.2, 3.9),
    axes_rect: Sequence[float] | None = None,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a thesis-ready Matplotlib 3D surface panel.

    Parameters
    ----------
    x
        1D x-axis values of shape ``(Nx,)``.
    y
        1D sweep values of shape ``(Ny,)``.
    z
        2D measurement array of shape ``(Ny, Nx)`` with ``z[y_i, x_j]``.
    xlim
        Optional x-range used to crop the surface before plotting.
    ylim
        Optional y-range used to crop the surface before plotting.
    zlim
        Optional z-axis limits.
    clim
        Optional color limits for the surface.
    trace_step
        Keep every ``trace_step``-th fixed-``y`` trace.
    surface_x_oversample
        Densify the surface mesh along the x direction by subdividing each
        original x interval into ``surface_x_oversample`` pieces. The
        default is ``10``; use ``1`` to keep the native x grid.
    xlabel, ylabel, zlabel
        Axis labels.
    show_xaxis, show_yaxis, show_zaxis
        Whether the axis labels and ticks should remain visible.
    z_axis_side
        Visual side used for the z-axis label and tick labels. Use
        ``"default"``, ``"left"``, or ``"right"``.
    invert_xaxis
        Whether to display the x-axis in descending order.
    invert_yaxis
        Whether to display the y-axis in descending order.
    labelspacing
        Optional scalar or ``(x, y, z)`` spacing where ``0`` places labels on
        the axis line and ``1`` matches Matplotlib's default 3D spacing.
        Values below ``0`` and above ``1`` are allowed.
    ticklabelspacing
        Optional scalar or ``(x, y, z)`` spacing where ``0`` places tick
        labels on the axis line and ``1`` matches Matplotlib's default 3D
        spacing. Values below ``0`` and above ``1`` are allowed.
    ticks
        Optional ``(x, y, z)`` tick tuple. Each entry may be a tick sequence
        or ``None`` to leave that axis unchanged.
    ticklabels
        Optional ``(x, y, z)`` tick-label tuple. Each entry may be a label
        sequence or ``None`` to leave that axis unchanged.
    clabel
        Optional colorbar label. Defaults to ``zlabel``.
    trace_y, trace_z
        Optional fixed-``y`` trace rows to emphasize on top of the surface.
        ``trace_z`` must use the same x-grid as ``x`` and have shape
        ``(Ntrace, Nx)``.
    trace_color
        Color used for optional surface trace highlights.
    trace_width
        Width control for optional surface trace highlights.
    trace_alpha
        Opacity used for optional surface trace highlights.
    surface_alpha
        Opacity of the surface colors.
    surface_shading
        Whether to modulate the height colormap with a light-based shading
        field to improve surface readability in print.
    surface_light_azdeg
        Azimuth of the virtual light source in degrees.
    surface_light_altdeg
        Altitude of the virtual light source in degrees.
    surface_shade_strength
        Blend factor between unshaded and light-shaded colors. ``0`` keeps
        pure height coloring and ``1`` uses the full light-based modulation.
    surface_rasterized
        Whether the surface data layer should be rasterized while keeping
        axes, ticks, and labels as vector output. When enabled, optional
        trace highlights are embedded into a locally refined single surface
        mesh before rasterization.
    cmap_mpl
        Matplotlib colormap used for the surface.
    show_colorbar
        Whether to add a Matplotlib colorbar.
    box_aspect
        Displayed box aspect ratio ``(x_ratio, y_ratio, z_ratio)``. Use
        ``None`` to scale the box from the visible data ranges. The default
        is ``(1, 1, 1)``.
    box_zoom
        Overall scale of the rendered 3D box inside the panel.
    elev
        Elevation angle passed to Matplotlib's 3D view.
    azim
        Azimuth angle passed to Matplotlib's 3D view.
    figsize
        Figure size used when ``ax`` is not provided.
    axes_rect
        Optional axes rectangle ``(left, bottom, width, height)`` in inches
        inside the figure canvas. Only used when ``ax`` is not provided.
    ax
        Optional existing 3D axes.

    Returns
    -------
    fig, ax
        Matplotlib figure and 3D axes containing the surface panel.
    """
    x_sel, y_sel, z_sel = prepare_waterfall_data(
        x=x,
        y=y,
        z=z,
        xlim=xlim,
        ylim=ylim,
        trace_step=trace_step,
    )
    overlay_x: NDArray64 | None = None
    overlay_y: NDArray64 | None = None
    overlay_z: NDArray64 | None = None
    trace_strips: list[tuple[NDArray64, NDArray64]] = []
    if (trace_y is None) != (trace_z is None):
        raise ValueError("trace_y and trace_z must be provided together.")
    if trace_y is not None and trace_z is not None:
        overlay_x, overlay_y, overlay_z = prepare_waterfall_data(
            x=x,
            y=trace_y,
            z=trace_z,
            xlim=xlim,
            ylim=ylim,
            trace_step=1,
        )
    x_plot, z_plot = _oversample_surface_x(
        x_sel,
        z_sel,
        factor=surface_x_oversample,
    )
    y_plot = y_sel
    trace_cell_weights = np.zeros(
        max(y_plot.size - 1, 0),
        dtype=np.float64,
    )
    if (
        overlay_x is not None
        and overlay_y is not None
        and overlay_z is not None
    ):
        overlay_z_plot = _interp_rows_to_x(
            overlay_x,
            overlay_z,
            x_plot,
        )
        if surface_rasterized:
            y_plot, z_plot, trace_cell_weights = _embed_surface_trace_bands(
                y=y_sel,
                z=z_plot,
                trace_y=overlay_y,
                trace_z=overlay_z_plot,
                trace_width=trace_width,
            )
        else:
            trace_strips = _build_surface_trace_strips(
                y_sel,
                z_sel,
                trace_y=overlay_y,
                trace_width=trace_width,
            )
    rect = None

    if ax is None:
        fig = plt.figure(figsize=figsize)
        rect = resolve_axes_rect_inches(figsize, axes_rect)
        if rect is None:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_axes(rect, projection="3d")
    else:
        fig = ax.figure

    cmin, cmax = resolve_color_range(z_plot, clim=clim)
    z_range = resolve_axis_range(z_plot, zlim)
    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    surface_facecolors = _surface_facecolors(
        z=z_plot,
        x=x_plot,
        y=y_plot,
        cmap_mpl=cmap_mpl,
        norm=norm,
        alpha=surface_alpha,
        shading=surface_shading,
        light_azdeg=surface_light_azdeg,
        light_altdeg=surface_light_altdeg,
        shade_strength=surface_shade_strength,
    )
    surface_facecolors = _surface_cell_facecolors(surface_facecolors)
    if surface_rasterized and np.any(trace_cell_weights > 0.0):
        surface_facecolors = _blend_surface_trace_facecolors(
            surface_facecolors,
            cell_weights=trace_cell_weights,
            trace_color=trace_color,
            trace_alpha=trace_alpha,
        )
    xg, yg = np.meshgrid(x_plot, y_plot, indexing="xy")
    surface = ax.plot_surface(
        xg,
        yg,
        z_plot,
        facecolors=surface_facecolors,
        shade=False,
        linewidth=0.0,
        edgecolor="none",
        antialiased=False,
        rstride=1,
        cstride=1,
    )
    surface.set_zorder(0)
    surface.set_alpha(float(np.clip(surface_alpha, 0.0, 1.0)))
    surface.set_rasterized(bool(surface_rasterized))

    if trace_strips and not surface_rasterized:
        trace_rgba = np.asarray(
            mcolors.to_rgba(trace_color, alpha=trace_alpha),
            dtype=np.float64,
        )
        trace_z_lift = _resolve_trace_strip_lift(z_plot)
        for y_strip, z_strip in trace_strips:
            xg_strip, yg_strip = np.meshgrid(
                x_sel,
                y_strip,
                indexing="xy",
            )
            z_strip_lifted = z_strip + trace_z_lift
            strip_facecolors = np.broadcast_to(
                trace_rgba,
                z_strip.shape + (4,),
            ).copy()
            strip = ax.plot_surface(
                xg_strip,
                yg_strip,
                z_strip_lifted,
                facecolors=strip_facecolors,
                shade=False,
                linewidth=0.0,
                edgecolor="none",
                antialiased=False,
                rstride=1,
                cstride=1,
            )
            strip.set_zorder(10)
            strip.set_alpha(float(np.clip(trace_alpha, 0.0, 1.0)))
            strip.set_rasterized(False)
            try:
                strip.set_sort_zpos(float(z_range[1] + trace_z_lift))
            except AttributeError:
                pass

    if show_colorbar:
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap_mpl)
        scalar_mappable.set_array([])
        fig.colorbar(
            scalar_mappable,
            ax=ax,
            shrink=0.72,
            pad=0.04,
            label=zlabel if clabel is None else clabel,
        )

    x_range = resolve_axis_range(x_plot, xlim)
    y_range = resolve_axis_range(y_plot, ylim)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(*z_range)
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    apply_matplotlib_axis_spacing(
        ax,
        labelspacing=labelspacing,
        ticklabelspacing=ticklabelspacing,
    )
    apply_matplotlib_axis_ticks(
        ax,
        ticks=ticks,
        ticklabels=ticklabels,
    )
    set_matplotlib_axis_visible(ax, axis_name="x", visible=show_xaxis)
    set_matplotlib_axis_visible(ax, axis_name="y", visible=show_yaxis)
    set_matplotlib_axis_visible(ax, axis_name="z", visible=show_zaxis)
    ax.view_init(elev=elev, azim=azim)

    set_matplotlib_box_aspect(
        ax=ax,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        box_aspect=box_aspect,
        zoom=box_zoom,
    )
    if rect is not None:
        try:
            ax.set_position(rect, which="both")
        except TypeError:
            ax.set_position(rect)
    set_matplotlib_background_style(ax)
    if show_zaxis:
        apply_matplotlib_zaxis_side(
            ax,
            side=z_axis_side,
        )

    if surface_rasterized:
        redraw_matplotlib_axes_on_top(ax, axis_names=("x", "y", "z"))

    return fig, ax


__all__ = ["get_thesis_surface_matplotlib"]
