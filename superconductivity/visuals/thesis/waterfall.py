"""Thesis-oriented waterfall panel helpers."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from superconductivity.utilities.types import LIM, NDArray64

from ._common import (
    iter_finite_trace_segments,
    prepare_waterfall_data,
    resolve_axis_range,
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
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    line_color: str = "black",
    line_width: float = 0.6,
    alpha: float = 1.0,
    elev: float = 20.0,
    azim: float = -70.0,
    figsize: tuple[float, float] = (6.2, 3.9),
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
    xlabel, ylabel, zlabel
        Axis labels.
    line_color
        Line color used for all traces.
    line_width
        Matplotlib line width.
    alpha
        Trace alpha.
    elev
        Elevation angle passed to Matplotlib's 3D view.
    azim
        Azimuth angle passed to Matplotlib's 3D view.
    figsize
        Figure size used when ``ax`` is not provided.
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

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)

    set_matplotlib_box_aspect(
        ax=ax,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    )

    return fig, ax


__all__ = ["get_thesis_waterfall_matplotlib"]
