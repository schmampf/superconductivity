"""Thesis-oriented surface panel helpers."""

from __future__ import annotations

from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from superconductivity.style.cpd4 import cmap as default_cmap
from superconductivity.utilities.types import LIM, NDArray64

from ._common import (
    prepare_waterfall_data,
    resolve_axis_range,
    resolve_color_range,
    set_matplotlib_box_aspect,
)


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
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    clabel: Optional[str] = None,
    surface_alpha: float = 1.0,
    cmap_mpl: ListedColormap = default_cmap(),
    show_colorbar: bool = False,
    elev: float = 20.0,
    azim: float = -70.0,
    figsize: tuple[float, float] = (6.2, 3.9),
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
    xlabel, ylabel, zlabel
        Axis labels.
    clabel
        Optional colorbar label. Defaults to ``zlabel``.
    surface_alpha
        Opacity of the surface colors.
    cmap_mpl
        Matplotlib colormap used for the surface.
    show_colorbar
        Whether to add a Matplotlib colorbar.
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

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    cmin, cmax = resolve_color_range(z_sel, clim=clim)
    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    xg, yg = np.meshgrid(x_sel, y_sel, indexing="xy")
    facecolors = cmap_mpl(norm(z_sel))
    facecolors[..., 3] *= float(np.clip(surface_alpha, 0.0, 1.0))
    ax.plot_surface(
        xg,
        yg,
        z_sel,
        facecolors=facecolors,
        shade=False,
        linewidth=0.0,
        antialiased=True,
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


__all__ = ["get_thesis_surface_matplotlib"]
