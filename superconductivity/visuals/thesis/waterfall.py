"""Thesis-oriented waterfall panel helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt

from superconductivity.utilities.types import LIM, NDArray64

from ._common import (
    apply_matplotlib_axis_ticks,
    apply_matplotlib_axis_spacing,
    apply_matplotlib_zaxis_side,
    iter_finite_trace_segments,
    prepare_waterfall_data,
    resolve_axis_range,
    resolve_axes_rect_inches,
    select_trace_rows,
    set_matplotlib_axis_visible,
    set_matplotlib_background_style,
    set_matplotlib_box_aspect,
)


def get_thesis_waterfall_matplotlib(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
    *,
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    trace_step: int = 1,
    trace_count: int | None = None,
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
    line_color: str = "black",
    line_width: float = 0.6,
    alpha: float = 1.0,
    box_aspect: Sequence[float] | None = (1.0, 1.0, 1.0),
    box_zoom: float = 1.0,
    elev: float = 20.0,
    azim: float = -70.0,
    figsize: tuple[float, float] = (6.2, 3.9),
    axes_rect: Sequence[float] | None = None,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a thesis-ready Matplotlib 3D waterfall panel.

    Parameters
    ----------
    x
        1D x-axis values of shape ``(Nx,)``.
    y
        1D sweep values of shape ``(Ny,)``.
    z
        2D measurement array of shape ``(Ny, Nx)`` with ``z[y_i, x_j]``.
    xlim
        Optional x-range used to crop the traces before plotting.
    ylim
        Optional y-range used to crop the waterfall stack before plotting.
    zlim
        Optional z-axis limits.
    trace_step
        Plot every ``trace_step``-th fixed-``y`` trace.
    trace_count
        Number of waterfall traces to draw after applying ``trace_step``.
        The first trace is always kept.
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
    line_color
        Line color used for all traces.
    line_width
        Matplotlib line width.
    alpha
        Trace alpha.
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
        Matplotlib figure and 3D axes containing the waterfall panel.
    """
    x_sel, y_sel, z_sel = prepare_waterfall_data(
        x=x,
        y=y,
        z=z,
        xlim=xlim,
        ylim=ylim,
        trace_step=trace_step,
    )
    y_sel, z_sel = select_trace_rows(
        y_sel,
        z_sel,
        trace_count=trace_count,
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

    n_segments = 0
    for x_run, y_run, z_run in iter_finite_trace_segments(
        x=x_sel,
        y=y_sel,
        z=z_sel,
    ):
        ax.plot(
            x_run,
            y_run,
            z_run,
            color=line_color,
            linewidth=line_width,
            alpha=alpha,
        )
        n_segments += 1

    if n_segments == 0:
        raise ValueError("No finite waterfall traces remain after selection.")

    x_range = resolve_axis_range(x_sel, xlim)
    y_range = resolve_axis_range(y_sel, ylim)
    z_range = resolve_axis_range(z_sel, zlim)
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

    return fig, ax


__all__ = ["get_thesis_waterfall_matplotlib"]
