"""Plotly helpers for 2D map-style data ``(x, y, z[y, x])``."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap

from superconductivity.style.cpd4 import cmap
from superconductivity.utilities.types import LIM, NDArray64

from ..helper import (
    check_xyz,
    get_clim,
    get_compressed_indices,
    get_xylim_indices,
)
from .helper import mpl_cmap_to_plotly


def get_axis(
    lim: LIM = None,
    label: str = "axis label",
    logscale: bool = False,
    showticklabels: bool = True,
    values: Optional[list[float]] = None,
    labels: Optional[list[str]] = None,
):
    axis = dict(
        range=list(lim) if lim else None,
        title=label,
        type="log" if logscale else "linear",  # 'linear' or 'log'
        showticklabels=showticklabels,
        tickvals=values,
        ticktext=labels,
    )

    return axis


def get_surface(
    x: NDArray64,  # (Nx,)
    y: NDArray64,  # (Ny,)
    z: NDArray64,  # (Ny, Nx)
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    clim: LIM = None,
    xlabel: str = "<i>eV/</i>Δ<sub>0</sub>",
    ylabel: str = "<i>eA/hν</i>",
    zlabel: str = "d<i>I</i>/d<i>V</i> (G<sub>0</sub>)",
    cmap_mpl: ListedColormap = cmap(),
):
    x, y, z = check_xyz(
        x=x,
        y=y,
        z=z,
    )
    ix, iy = get_xylim_indices(
        x=x,
        y=y,
        xlim=xlim,
        ylim=ylim,
    )
    x = x[ix]
    y = y[iy]
    z = z[np.ix_(iy, ix)]

    cmin, cmax = get_clim(z=z, zlim=zlim, clim=clim)
    colorscale = mpl_cmap_to_plotly(cmap_mpl=cmap_mpl)

    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        cmin=cmin,
        cmax=cmax,
        colorscale=colorscale,
        showscale=False,
    )

    xaxis = get_axis(lim=xlim, label=xlabel)
    yaxis = get_axis(lim=ylim, label=ylabel)
    zaxis = get_axis(lim=zlim, label=zlabel)

    scene: go.layout.scene = dict(
        xaxis=xaxis,
        yaxis=yaxis,
        zaxis=zaxis,
    )

    layout = go.Layout(
        scene=scene,
    )
    fig = go.Figure(
        data=[surface],
        layout=layout,
        skip_invalid=False,
    )
    return fig


def get_plain(
    x: NDArray64,  # (Nx,)
    y: NDArray64,  # (Ny,)
    z: NDArray64,  # (Ny, Nx)
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    xlabel: str = "<i>eV/</i>Δ<sub>0</sub>",
    ylabel: str = "<i>eA/hν</i>",
    zlabel: str = "d<i>I</i>/d<i>V</i> (G<sub>0</sub>)",
    k: int = 0,
):
    x, y, z = check_xyz(
        x=x,
        y=y,
        z=z,
    )
    ix, iy = get_xylim_indices(
        x=x,
        y=y,
        xlim=xlim,
        ylim=ylim,
    )
    x = x[ix]
    y = y[iy]
    z = z[np.ix_(iy, ix)]

    ylabel = f"{ylabel}"

    trace = go.Scatter(
        x=x,
        y=z[k, :],
    )

    xaxis = get_axis(lim=xlim, label=xlabel)
    zaxis = get_axis(lim=zlim, label=zlabel)

    layout = go.Layout(
        xaxis=xaxis,
        yaxis=zaxis,
    )
    fig = go.Figure(
        data=[trace],
        layout=layout,
        skip_invalid=False,
    )

    return fig


def get_slider(
    x: NDArray64,  # (Nx,)
    y: NDArray64,  # (Ny,)
    z: NDArray64,  # (Ny, Nx)
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    xlabel: str = "<i>eV/</i>Δ<sub>0</sub>",
    ylabel: str = "<i>eA/hν</i>",
    zlabel: str = "d<i>I</i>/d<i>V</i> (G<sub>0</sub>)",
    k_static: Optional[list[int]] = None,
):
    """2D slice plot with a y-slider plus optional static reference slices.

    The figure contains:
    - One *dynamic* trace (trace index 0) showing z[y_k, :] vs x.
        The slider updates this.
    - Zero or more *static* traces showing fixed y-indices (k_static).
        These never change.

    Notes
    -----
    - The slider always starts at k=0 (active=0).
    - This function assumes z has shape (Ny, Nx) with z[k, j] = z(y_k, x_j).
    """
    x, y, z = check_xyz(
        x=x,
        y=y,
        z=z,
    )
    ix, iy = get_xylim_indices(
        x=x,
        y=y,
        xlim=xlim,
        ylim=ylim,
    )
    x = x[ix]
    y = y[iy]
    z = z[np.ix_(iy, ix)]

    iy = get_compressed_indices(y, z)
    y = y[iy]
    z = z[iy, :]

    trace_dyn = go.Scatter(
        x=x,
        y=z[0, :],
        mode="lines",
        name=f"k=0 @ y={y[0]:g}",
    )
    traces = [trace_dyn]

    if k_static is None:
        k_static = []

    Ny = int(len(y))
    k_static_unique: list[int] = []
    for k in k_static:
        kk = int(k)
        if kk < 0:
            kk += Ny
        if 0 <= kk < Ny and kk != 0 and kk not in k_static_unique:
            k_static_unique.append(kk)

    for kk in k_static_unique:
        traces.append(
            go.Scatter(
                x=x,
                y=z[kk, :],
                mode="lines",
                name=f"k={kk} @ y={y[kk]:g}",
            )
        )

    xaxis = get_axis(lim=xlim, label=xlabel)
    yaxis = get_axis(lim=zlim, label=zlabel)

    steps = []
    for k, _ in enumerate(y):
        steps.append(
            dict(
                label=f"{y[k]:g}",
                method="restyle",
                args=[
                    {"y": [z[k, :]], "name": [f"k={k} @ y={y[k]:g}"]},
                    [0],
                ],
            )
        )

    layout = go.Layout(
        xaxis=xaxis,
        yaxis=yaxis,
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": f"{ylabel}: "},
                pad={"t": 20},
                steps=steps,
            )
        ],
    )

    fig = go.Figure(
        data=traces,
        layout=layout,
        skip_invalid=False,
    )

    return fig


def get_heatmap(
    x: NDArray64,  # (Nx,)
    y: NDArray64,  # (Ny,)
    z: NDArray64,  # (Ny, Nx)
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    clim: LIM = None,
    xlabel: str = "<i>eV/</i>Δ<sub>0</sub>",
    ylabel: str = "<i>eA/hν</i>",
    zlabel: str = "d<i>I</i>/d<i>V</i> (G<sub>0</sub>)",
    cmap_mpl: ListedColormap = cmap(),
    show_scale: bool = True,
):
    x, y, z = check_xyz(
        x=x,
        y=y,
        z=z,
    )
    ix, iy = get_xylim_indices(
        x=x,
        y=y,
        xlim=xlim,
        ylim=ylim,
    )
    x = x[ix]
    y = y[iy]
    z = z[np.ix_(iy, ix)]

    cmin, cmax = get_clim(z=z, zlim=zlim, clim=clim)
    colorscale = mpl_cmap_to_plotly(cmap_mpl=cmap_mpl)

    colorbar = dict(
        title=zlabel,
        len=0.8,
        thickness=18,
        tickformat=".1f",
    )

    heatmap = go.Heatmap(
        x=x,
        y=y,
        z=z,
        zmin=cmin,
        zmax=cmax,
        colorscale=colorscale,
        showscale=show_scale,
        colorbar=colorbar,
    )

    xaxis = get_axis(lim=xlim, label=xlabel)
    yaxis = get_axis(lim=ylim, label=ylabel)

    layout = go.Layout(
        xaxis=xaxis,
        yaxis=yaxis,
    )

    fig = go.Figure(
        data=[heatmap],
        layout=layout,
        skip_invalid=False,
    )

    return fig


def get_all(
    x: NDArray64,  # (Nx,)
    y: NDArray64,  # (Ny,)
    z: NDArray64,  # (Ny, Nx)
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    clim: LIM = None,
    xlabel: str = "<i>eV/</i>Δ<sub>0</sub>",
    ylabel: str = "<i>eA/hν</i>",
    zlabel: str = "d<i>I</i>/d<i>V</i> (G<sub>0</sub>)",
    cmap_mpl: ListedColormap = cmap(),
    k_static: Optional[list[int]] = None,
    show_scale: bool = True,
    name: Optional[str] = None,
    scheme: str = "standard",
    dataset: str = "dataset",
):

    fig_surface = get_surface(
        x=x,
        y=y,
        z=z,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        clim=clim,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        cmap_mpl=cmap_mpl,
    )

    fig_slider = get_slider(
        x=x,
        y=y,
        z=z,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        k_static=k_static,
    )

    fig_heatmap = get_heatmap(
        x=x,
        y=y,
        z=z,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        clim=clim,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        cmap_mpl=cmap_mpl,
        show_scale=show_scale,
    )

    if name is not None:
        figs = [fig_surface, fig_slider, fig_heatmap]
        labels = ["surface", "slider", "heatmap"]
        for i, fig in enumerate(figs):
            fig_title = f"{name} {labels[i]} {i}"
            save_figure(
                fig=fig,
                title=fig_title,
                scheme=scheme,
                out_dir=dataset,
            )
    return fig_surface, fig_slider, fig_heatmap


def save_figure(
    fig: go.Figure,
    title: str,
    scheme: str = "standard",
    out_dir: str | Path = ".",
    auto_open: bool = False,
    full_html: bool = True,
):
    """
    Save a Plotly figure to an HTML file with configurable JavaScript bundling.

    The figure is written to ``<out_dir>/<sanitized_title>.html``.
    The ``scheme`` controls how Plotly.js is provided to the HTML page:

    - ``"online"``: load Plotly.js from the Plotly CDN
        (small HTML, requires internet).
    - ``"standard"``: reference a local ``plotly.min.js`` in ``out_dir``
        (generated by Plotly if missing) to avoid duplicating Plotly.js
        across many HTML files.
    - ``"stand_alone"``: embed Plotly.js directly into the HTML
        (single-file offline).

    Parameters
    ----------
    fig
        Plotly figure to save.
    title
        Used as the filename stem (sanitized for filesystem safety).
    scheme
        One of ``{"online", "standard", "stand_alone"}``. Controls how
        Plotly.js is included.
    out_dir
        Output directory.
    auto_open
        If True, open the written HTML in the default web browser.
    full_html
        If True, write a complete standalone HTML document. If False, write an
        HTML fragment (div + script) suitable for embedding into a larger page.

    Raises
    ------
    ValueError
        If ``scheme`` is not one of ``{"online", "standard", "stand_alone"}``.

    Notes
    -----
    - ``scheme="standard"`` uses Plotly's ``include_plotlyjs="directory"``
    behavior, which writes a shared Plotly.js bundle into ``out_dir`` the
    first time.

    """
    out_path = Path(out_dir, "html")
    out_path.mkdir(parents=True, exist_ok=True)

    scheme_norm = scheme.strip().lower()

    if scheme_norm in {"online", "standard", "stand_alone"}:
        html_path = out_path / f"{title}.html"

        if scheme_norm == "online":
            include_plotlyjs: Union[bool, str] = "cdn"
        elif scheme_norm == "standard":
            # Plotly will write a local plotly.min.js into out_dir if needed.
            include_plotlyjs = "directory"
        else:  # html_stand_alone
            include_plotlyjs = True

        fig.write_html(
            str(html_path),
            full_html=full_html,
            include_plotlyjs=include_plotlyjs,
            config={"responsive": True},
            auto_open=auto_open,
        )
    else:
        raise ValueError(
            "scheme must be one of: 'online', 'standard', 'stand_alone'",
        )
