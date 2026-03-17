"""Thesis-oriented heatmap panel helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from superconductivity.style.cpd4 import cmap as default_cmap
from superconductivity.utilities.types import LIM, NDArray64

from ._common import (
    apply_matplotlib_axis_ticks,
    apply_matplotlib_axis_spacing,
    apply_matplotlib_zaxis_side,
    hide_matplotlib_panes,
    hide_matplotlib_zaxis,
    prepare_waterfall_data,
    redraw_matplotlib_axes_on_top,
    resolve_axis_range,
    resolve_axes_rect_inches,
    resolve_color_range,
    set_matplotlib_axis_visible,
    set_matplotlib_background_style,
    set_matplotlib_box_aspect,
)


def _centers_to_edges(values: NDArray64) -> NDArray64:
    """Convert 1D sample centers into cell-edge coordinates."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1D array.")
    if arr.size == 1:
        step = 1.0
        return np.array(
            [arr[0] - 0.5 * step, arr[0] + 0.5 * step],
            dtype=np.float64,
        )

    midpoints = 0.5 * (arr[1:] + arr[:-1])
    first = arr[0] - 0.5 * (arr[1] - arr[0])
    last = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    return np.concatenate(
        (
            np.array([first], dtype=np.float64),
            midpoints,
            np.array([last], dtype=np.float64),
        )
    )


def _build_heatmap_quads(
    x_edges: NDArray64,
    y_edges: NDArray64,
    *,
    z_level: float,
    overlap: float = 0.0,
) -> NDArray64:
    """Build flat 3D quads for one heatmap cell per data value."""
    x_arr = np.asarray(x_edges, dtype=np.float64)
    y_arr = np.asarray(y_edges, dtype=np.float64)
    overlap_value = float(overlap)
    if x_arr.ndim != 1 or x_arr.size < 2:
        raise ValueError("x_edges must contain at least two 1D values.")
    if y_arr.ndim != 1 or y_arr.size < 2:
        raise ValueError("y_edges must contain at least two 1D values.")
    if not np.isfinite(overlap_value):
        raise ValueError("heatmap_cell_overlap must be finite.")
    if overlap_value < 0.0:
        raise ValueError("heatmap_cell_overlap must be non-negative.")

    left = x_arr[:-1]
    right = x_arr[1:]
    bottom = y_arr[:-1]
    top = y_arr[1:]
    x_pad = 0.5 * overlap_value * (right - left)
    y_pad = 0.5 * overlap_value * (top - bottom)

    left = left - x_pad
    right = right + x_pad
    bottom = bottom - y_pad
    top = top + y_pad

    x0, y0 = np.meshgrid(left, bottom, indexing="xy")
    x1, y1 = np.meshgrid(right, top, indexing="xy")
    z_plane = np.full(x0.shape, float(z_level), dtype=np.float64)
    polygons = np.empty(x0.shape + (4, 3), dtype=np.float64)
    polygons[..., 0, 0] = x0
    polygons[..., 0, 1] = y0
    polygons[..., 0, 2] = z_plane
    polygons[..., 1, 0] = x1
    polygons[..., 1, 1] = y0
    polygons[..., 1, 2] = z_plane
    polygons[..., 2, 0] = x1
    polygons[..., 2, 1] = y1
    polygons[..., 2, 2] = z_plane
    polygons[..., 3, 0] = x0
    polygons[..., 3, 1] = y1
    polygons[..., 3, 2] = z_plane
    return polygons.reshape(-1, 4, 3)


def _build_heatmap_rgba(
    z: NDArray64,
    *,
    cmap_mpl: ListedColormap,
    norm: mcolors.Normalize,
    alpha: float,
) -> np.ndarray:
    """Map a 2D scalar field onto an RGBA image."""
    values = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(values)
    color_values = np.where(finite, values, norm.vmin)
    rgba = np.asarray(cmap_mpl(norm(color_values)), dtype=np.float64)
    rgba[..., 3] *= float(np.clip(alpha, 0.0, 1.0))
    rgba[..., 3] *= finite.astype(np.float64)
    return np.round(np.flipud(rgba) * 255.0).astype(np.uint8)


def _project_plane_quad(
    ax: plt.Axes,
    *,
    x_edges: NDArray64,
    y_edges: NDArray64,
    z_level: float,
) -> NDArray64:
    """Project the four heatmap-plane corners into display coordinates."""
    corners_3d = np.array(
        [
            [x_edges[0], y_edges[-1], z_level],
            [x_edges[-1], y_edges[-1], z_level],
            [x_edges[-1], y_edges[0], z_level],
            [x_edges[0], y_edges[0], z_level],
        ],
        dtype=np.float64,
    )
    x_proj, y_proj, _ = proj3d.proj_transform(
        corners_3d[:, 0],
        corners_3d[:, 1],
        corners_3d[:, 2],
        ax.get_proj(),
    )
    return np.asarray(
        ax.transData.transform(np.column_stack((x_proj, y_proj))),
        dtype=np.float64,
    )


def _solve_perspective_coefficients(
    destination: NDArray64,
    source: NDArray64,
) -> tuple[float, ...]:
    """Solve a PIL perspective transform from destination to source."""
    dest = np.asarray(destination, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    if dest.shape != (4, 2) or src.shape != (4, 2):
        raise ValueError("destination and source must both be shaped (4, 2).")

    matrix = np.zeros((8, 8), dtype=np.float64)
    rhs = np.zeros(8, dtype=np.float64)
    for index, ((x_val, y_val), (u_val, v_val)) in enumerate(zip(dest, src)):
        row = 2 * index
        matrix[row] = (
            x_val,
            y_val,
            1.0,
            0.0,
            0.0,
            0.0,
            -u_val * x_val,
            -u_val * y_val,
        )
        matrix[row + 1] = (
            0.0,
            0.0,
            0.0,
            x_val,
            y_val,
            1.0,
            -v_val * x_val,
            -v_val * y_val,
        )
        rhs[row] = u_val
        rhs[row + 1] = v_val

    return tuple(float(value) for value in np.linalg.solve(matrix, rhs))


class _WarpedHeatmapImageArtist(Artist):
    """Draw a heatmap as a perspective-warped RGBA image on a 3D plane."""

    def __init__(
        self,
        ax: plt.Axes,
        *,
        rgba: np.ndarray,
        x_edges: NDArray64,
        y_edges: NDArray64,
        z_level: float,
    ) -> None:
        super().__init__()
        self.set_figure(ax.figure)
        self.axes = ax
        self._source_rgba = np.asarray(rgba, dtype=np.uint8)
        self._source_image = Image.fromarray(self._source_rgba, mode="RGBA")
        self._x_edges = np.asarray(x_edges, dtype=np.float64)
        self._y_edges = np.asarray(y_edges, dtype=np.float64)
        self._z_level = float(z_level)

    def draw(self, renderer: object) -> None:
        """Draw the warped heatmap image onto the projected plane."""
        if not self.get_visible():
            return

        quad = _project_plane_quad(
            self.axes,
            x_edges=self._x_edges,
            y_edges=self._y_edges,
            z_level=self._z_level,
        )
        if not np.all(np.isfinite(quad)):
            return

        min_x = float(np.min(quad[:, 0]))
        max_x = float(np.max(quad[:, 0]))
        min_y = float(np.min(quad[:, 1]))
        max_y = float(np.max(quad[:, 1]))
        if max_x <= min_x or max_y <= min_y:
            return

        magnification = float(renderer.get_image_magnification())
        width_px = max(1, int(np.ceil((max_x - min_x) * magnification)))
        height_px = max(1, int(np.ceil((max_y - min_y) * magnification)))
        destination = np.empty((4, 2), dtype=np.float64)
        destination[:, 0] = (quad[:, 0] - min_x) * magnification
        destination[:, 1] = (max_y - quad[:, 1]) * magnification
        source = np.array(
            [
                [0.0, 0.0],
                [self._source_rgba.shape[1], 0.0],
                [self._source_rgba.shape[1], self._source_rgba.shape[0]],
                [0.0, self._source_rgba.shape[0]],
            ],
            dtype=np.float64,
        )
        coefficients = _solve_perspective_coefficients(destination, source)
        warped = self._source_image.transform(
            (width_px, height_px),
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.NEAREST,
            fillcolor=(0, 0, 0, 0),
        )

        gc = renderer.new_gc()
        try:
            gc.set_clip_rectangle(self.axes.bbox)
            renderer.draw_image(
                gc,
                min_x,
                min_y,
                np.flipud(np.array(warped, dtype=np.uint8, copy=True)),
            )
        finally:
            gc.restore()


def get_thesis_heatmap_matplotlib(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
    *,
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    clim: LIM = None,
    trace_step: int = 1,
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    show_xaxis: bool = True,
    show_yaxis: bool = True,
    invert_xaxis: bool = False,
    invert_yaxis: bool = True,
    z_axis_side: str = "default",
    labelspacing: float | Sequence[float] | None = None,
    ticklabelspacing: float | Sequence[float] | None = None,
    ticks: Sequence[object | None] | None = None,
    ticklabels: Sequence[object | None] | None = None,
    clabel: Optional[str] = None,
    z_level: float | None = None,
    heatmap_alpha: float = 1.0,
    heatmap_mode: str = "warped_image",
    heatmap_cell_overlap: float = 0.01,
    show_zaxis: bool = False,
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
    """Create a thesis-ready Matplotlib 3D heatmap panel.

    Parameters
    ----------
    x
        1D x-axis values of shape ``(Nx,)``.
    y
        1D sweep values of shape ``(Ny,)``.
    z
        2D measurement array of shape ``(Ny, Nx)`` with ``z[y_i, x_j]``.
    xlim
        Optional x-range used to crop the heatmap before plotting.
    ylim
        Optional y-range used to crop the heatmap before plotting.
    zlim
        Optional z-axis limits used as the shared 3D reference span.
    clim
        Optional color limits for the heatmap.
    trace_step
        Keep every ``trace_step``-th fixed-``y`` trace.
    xlabel, ylabel, zlabel
        Axis labels.
    show_xaxis, show_yaxis
        Whether the x/y axis labels and ticks should remain visible.
    invert_xaxis
        Whether to display the x-axis in descending order.
    z_axis_side
        Visual side used for the z-axis label and tick labels when the
        z-axis is shown. Use ``"default"``, ``"left"``, or ``"right"``.
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
    z_level
        Z position of the flat heatmap plane. When omitted, the plane is
        placed at the lower bound of the resolved z-range.
    heatmap_alpha
        Opacity of the heatmap colors.
    heatmap_mode
        Rendering mode for the heatmap plane. Use ``"warped_image"`` for a
        perspective-warped RGBA image with one source pixel per data bin, or
        ``"vector_cells"`` for one vector polygon per bin.
    heatmap_cell_overlap
        Relative overlap added between neighboring heatmap cells to hide
        PDF seam artifacts in ``"vector_cells"`` mode. ``0`` keeps exact
        cell edges.
    show_zaxis
        Whether to keep the native Matplotlib z-axis decorations.
    cmap_mpl
        Matplotlib colormap used for the heatmap.
    show_colorbar
        Whether to add a Matplotlib colorbar.
    box_aspect
        Displayed box aspect ratio ``(x_ratio, y_ratio, z_ratio)``. Use
        ``None`` to scale the box from the visible data ranges.
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
        Matplotlib figure and 3D axes containing the heatmap panel.
    """
    x_sel, y_sel, z_sel = prepare_waterfall_data(
        x=x,
        y=y,
        z=z,
        xlim=xlim,
        ylim=ylim,
        trace_step=trace_step,
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

    cmin, cmax = resolve_color_range(z_sel, clim=clim)
    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    x_edges = _centers_to_edges(x_sel)
    y_edges = _centers_to_edges(y_sel)
    z_range = resolve_axis_range(z_sel, zlim)
    plane_level = z_range[0] if z_level is None else float(z_level)
    if plane_level < z_range[0] or plane_level > z_range[1]:
        raise ValueError("z_level must lie within the resolved z-range.")
    if heatmap_mode == "warped_image":
        heatmap = _WarpedHeatmapImageArtist(
            ax,
            rgba=_build_heatmap_rgba(
                z_sel,
                cmap_mpl=cmap_mpl,
                norm=norm,
                alpha=heatmap_alpha,
            ),
            x_edges=x_edges,
            y_edges=y_edges,
            z_level=plane_level,
        )
        ax.add_artist(heatmap)
    elif heatmap_mode == "vector_cells":
        facecolors = (
            _build_heatmap_rgba(
                z_sel,
                cmap_mpl=cmap_mpl,
                norm=norm,
                alpha=heatmap_alpha,
            )
            .reshape(z_sel.shape[0], z_sel.shape[1], 4)
            .astype(np.float64)
            / 255.0
        )
        heatmap = Poly3DCollection(
            _build_heatmap_quads(
                x_edges,
                y_edges,
                z_level=plane_level,
                overlap=heatmap_cell_overlap,
            ),
            facecolors=np.flipud(facecolors).reshape(-1, 4),
            edgecolors="none",
            linewidths=0.0,
            antialiaseds=False,
            zsort="min",
        )
        heatmap.set_snap(False)
        ax.add_collection3d(heatmap)
    else:
        raise ValueError(
            "heatmap_mode must be 'warped_image' or 'vector_cells'."
        )

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

    x_range = resolve_axis_range(x_sel, xlim)
    y_range = resolve_axis_range(y_sel, ylim)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(*z_range)
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    if show_zaxis:
        ax.set_zlabel(zlabel)
    else:
        hide_matplotlib_zaxis(ax)
    hide_matplotlib_panes(ax, axis_names=("x", "y"))
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
        redraw_matplotlib_axes_on_top(ax, axis_names=("x", "y", "z"))
    else:
        redraw_matplotlib_axes_on_top(ax, axis_names=("x", "y"))

    return fig, ax


__all__ = ["get_thesis_heatmap_matplotlib"]
