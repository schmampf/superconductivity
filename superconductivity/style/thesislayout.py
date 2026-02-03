from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure

from ..utilities.types import NDArray64

Textwidth: float = 4.25279  # in
Textheight: float = 6.85173  # in


Local: str = "/Users/oliver/Documents/superconductivity/thesis/"

Remote: str = "/Users/oliver/Documents/dissertation/"

Style: str = "/Users/oliver/Documents/superconductivity/superconductivity/style/"

plt.style.use(f"{Style}thesisstyle.mplstyle")


def save_figure(
    fig: Figure,
    title: Optional[str],
    path_pgf: Optional[str] = None,
    path_pdf: Optional[str] = None,
):
    if path_pgf is None:
        path_pgf = Remote

    if path_pdf is None:
        path_pdf = Local

    # save figure
    if title is not None:
        pgf_dir = Path(path_pgf).expanduser().resolve()
        png_dir = Path(path_pdf).expanduser().resolve()

        # Allow `title` to include subfolders (e.g. "figs/foo")
        # and/or a suffix.
        t = Path(title)
        t = t.with_suffix("") if t.suffix else t

        out_pgf = (pgf_dir / t).with_suffix(".pgf")
        out_png = (png_dir / t).with_suffix(".png")

        out_pgf.parent.mkdir(parents=True, exist_ok=True)
        out_png.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(out_pgf)
        fig.savefig(out_png)


def get_ext(
    x: NDArray64,
    y: NDArray64,
) -> tuple[float, float, float, float]:
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    dx = np.abs(x_max - x_min) / (np.shape(x)[0] + 1) / 2
    dy = np.abs(y_max - y_min) / (np.shape(y)[0] + 1) / 2
    ext = (x_min - dx, x_max + dx, y_min - dy, y_max + dy)
    return ext


def get_figure(
    figsize: tuple[float, float] = (1.7, 0.85),
    facecolor: Optional[str] = None,
    subfigure: bool = True,
    padding: Optional[tuple[float, float]] = None,
):
    if padding is None:
        padding = (0.2, 0.2)

    fig, ax = plt.subplots(
        figsize=figsize,
        facecolor=facecolor,
    )

    fig.subfigure: bool = subfigure
    fig.padding: Optional[tuple[float, float]] = padding

    return fig, ax


def theory_layout(
    fig: Figure,
    ax: Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    padding: Optional[tuple[float, float]] = None,
    base_padding: Optional[tuple[float, float]] = None,
    path_pgf: Optional[str] = None,
    path_pdf: Optional[str] = None,
):
    if xlabel is None:
        xlabel = "$x$"

    if ylabel is None:
        ylabel = "$y$"

    if padding is None:
        padding = fig.padding

    if base_padding is None:
        base_padding = (0.08, 0.08)

    if fig.subfigure:
        ax.labelsize = 7
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.tick_params(labelsize=ax.labelsize)
    else:
        ax.labelsize = 8

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        labelsize=ax.labelsize,
        pad=1.5,
    )

    ms: float = 6.0
    fontsize: float = 8
    if fig.subfigure:
        ms: float = 4.0
        fontsize: float = 7

    # Remove frame
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Simplify ticks
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        labelsize=ax.labelsize,
        color="k",
        labelcolor="k",
    )

    # get limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Draw arrow heads
    ax.plot(x1, y0, ">", color="k", ms=ms, clip_on=False)
    ax.plot(x0, y1, "^", color="k", ms=ms, clip_on=False)

    # get config stuff
    w0, h0 = fig.get_size_inches()
    px, py = fig.padding
    bpx, bpy = base_padding
    w1 = w0 - (px + 2 * bpx)
    h1 = h0 - (py + 2 * bpy)
    dx = np.abs(x1 - x0)
    dy = np.abs(y1 - y0)

    # make layout
    ax.set_position(
        (
            (bpx + px) / w0,
            (bpy + py) / h0,
            w1 / w0,
            h1 / h0,
        ),
    )

    # labels
    ax.text(
        # x = x1 + bpx / w1 * dx, #label on the right
        # ha="right",
        x=x0 + dx / 2,  # label in the middle
        ha="center",
        y=y0 - (py + bpy) / h1 * dy,
        s=xlabel,
        va="bottom",
        fontsize=fontsize,
    )
    ax.text(
        x0 - (bpx + px) / w1 * dx,
        y0 + dy / 2,
        ylabel,
        rotation=90,
        ha="left",
        va="center",
        fontsize=fontsize,
    )

    # set limits
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))

    save_figure(fig, title, path_pgf, path_pdf)


def map_layout(
    fig: Figure,
    ax: Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    padding: Optional[tuple[float, float]] = None,
    base_padding: Optional[tuple[float, float]] = None,
    path_pgf: Optional[str] = None,
    path_pdf: Optional[str] = None,
):
    if padding is None:
        padding = fig.padding

    if base_padding is None:
        base_padding = (0.08, 0.08)

    if fig.subfigure:
        ax.labelsize = 7
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.tick_params(labelsize=ax.labelsize)
    else:
        ax.labelsize = 8

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        labelsize=ax.labelsize,
        pad=1.5,
    )

    ms: float = 6.0
    fontsize: float = 8
    if fig.subfigure:
        ms: float = 4.0
        fontsize: float = 7

    # Remove frame
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)

    # Simplify ticks
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        labelsize=ax.labelsize,
        color="k",
        labelcolor="k",
    )

    # get limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # get config stuff
    w0, h0 = fig.get_size_inches()
    px, py = fig.padding
    bpx, bpy = base_padding
    w1 = w0 - (px + 2 * bpx)
    h1 = h0 - (py + 2 * bpy)
    dx = np.abs(x1 - x0)
    dy = np.abs(y1 - y0)

    # make layout
    ax.set_position(
        (
            (bpx + px) / w0,
            (bpy + py) / h0,
            w1 / w0,
            h1 / h0,
        ),
    )

    # labels
    ax.text(
        # x = x1 + bpx / w1 * dx, #label on the right
        # ha="right",
        x=x0 + dx / 2,  # label in the middle
        ha="center",
        y=y0 - (py + bpy) / h1 * dy,
        s=xlabel,
        va="bottom",
        fontsize=fontsize,
    )
    ax.text(
        x0 - (bpx + px) / w1 * dx,
        y0 + dy / 2,
        ylabel,
        rotation=90,
        ha="left",
        va="center",
        fontsize=fontsize,
    )

    # set limits
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))

    save_figure(fig, title, path_pgf, path_pdf)
