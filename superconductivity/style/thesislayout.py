from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure

from ..utilities.types import NDArray64

Textwidth: float = 4.25279  # in
Textheight: float = 6.85173  # in
PngDpi: int = 1800


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRYOLAB_ROOT = _REPO_ROOT.parent
_STYLE_DIR = Path(__file__).resolve().parent

Local: str = f"{(_REPO_ROOT / 'thesis').resolve()}/"

Remote: str = f"{(_CRYOLAB_ROOT / 'dissertation').resolve()}/"

Style: str = f"{_STYLE_DIR.resolve()}/"

plt.style.use(str(_STYLE_DIR / "thesisstyle.mplstyle"))


def _normalize_padding(
    padding: Optional[tuple[float, ...]],
    default: tuple[float, float],
) -> tuple[float, float, float, float]:
    if padding is None:
        padding = default

    if len(padding) == 2:
        left, bottom = padding
        return left, bottom, 0.0, 0.0

    if len(padding) == 4:
        left, bottom, right, top = padding
        return left, bottom, right, top

    raise ValueError("padding must be a 2-tuple or 4-tuple.")


def save_figure(
    fig: Figure,
    title: Optional[str],
    path_pgf: Optional[str] = None,
    path_png: Optional[str] = None,
    png_dpi: int = PngDpi,
):
    if path_pgf is None:
        path_pgf = Remote

    if path_png is None:
        path_png = Local

    # save figure
    if title is not None:
        pgf_dir = Path(path_pgf).expanduser().resolve()
        png_dir = Path(path_png).expanduser().resolve()

        # Allow `title` to include subfolders (e.g. "figs/foo")
        # and/or a suffix.
        t = Path(title)
        t = t.with_suffix("") if t.suffix else t

        out_pgf = (pgf_dir / t).with_suffix(".pgf")
        out_pdf = (pgf_dir / t).with_suffix(".pdf")
        out_png = (png_dir / t).with_suffix(".png")

        out_pgf.parent.mkdir(parents=True, exist_ok=True)
        out_png.parent.mkdir(parents=True, exist_ok=True)

        # PGF for LaTeX (keep normal)
        fig.savefig(out_pgf)
        fig.savefig(out_pdf)

        # PNG for slides/web: transparent background (no white fill)
        fig.savefig(
            out_png,
            dpi=png_dpi,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )


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


def get_figures(
    nrows: int = 7,
    ncols: int = 4,
    figsize: Optional[tuple[float, float]] = None,
    facecolor: Optional[str] = None,
    subfigure: bool = True,
    padding: Optional[tuple[float, ...]] = None,
):
    if figsize is None:
        figsize = (Textwidth, Textheight)

    padding = _normalize_padding(padding, default=(0.38, 0.28))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        facecolor=facecolor,
        squeeze=False,
    )

    fig.subfigure = subfigure
    fig.padding = padding
    fig.nrows = nrows
    fig.ncols = ncols

    return fig, list(axes.ravel())


def daumenkino_layout(
    fig: Figure,
    axes: list[Axes] | np.ndarray,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticks: Optional[NDArray64] = None,
    yticks: Optional[NDArray64] = None,
    xticklabels: Optional[list[str]] = None,
    yticklabels: Optional[list[str]] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    padding: Optional[tuple[float, ...]] = None,
    base_padding: Optional[tuple[float, float]] = None,
    show_inner_ticklabels: bool = False,
    path_pgf: Optional[str] = None,
    path_pdf: Optional[str] = None,
):
    axes_arr = np.asarray(axes, dtype=object)
    axes_flat = list(axes_arr.ravel())

    if nrows is None:
        nrows = getattr(fig, "nrows", None)
    if ncols is None:
        ncols = getattr(fig, "ncols", None)
    if nrows is None or ncols is None:
        n_axes = len(axes_flat)
        ncols = 4
        nrows = int(np.ceil(n_axes / ncols))

    if len(axes_flat) != nrows * ncols:
        raise ValueError("nrows * ncols must match the number of axes.")

    if padding is None:
        padding = getattr(fig, "padding", None)
    left_pad, bottom_pad, right_pad, top_pad = _normalize_padding(
        padding,
        default=(0.38, 0.28),
    )

    if base_padding is None:
        base_padding = (0.04, 0.04)

    is_subfigure = getattr(fig, "subfigure", True)
    labelsize = 7 if is_subfigure else 8
    fontsize = 7 if is_subfigure else 8

    w0, h0 = fig.get_size_inches()
    bpx, bpy = base_padding
    left = (left_pad + bpx) / w0
    right = 1.0 - (right_pad + bpx) / w0
    bottom = (bottom_pad + bpy) / h0
    top = 1.0 - (top_pad + bpy) / h0

    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=0.0,
        hspace=0.0,
    )

    for index, ax in enumerate(axes_flat):
        row = index // ncols
        col = index % ncols
        is_left = col == 0
        is_bottom = row == nrows - 1

        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(True)

        ax.tick_params(
            axis="both",
            direction="in",
            length=3,
            labelsize=labelsize,
            pad=1.5,
            color="k",
            labelcolor="k",
            top=True,
            right=True,
        )

        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)

        ax.set_xlabel(xlabel if is_bottom else "", fontsize=fontsize)
        ax.set_ylabel(ylabel if is_left else "", fontsize=fontsize)

        if not show_inner_ticklabels:
            ax.tick_params(
                labelbottom=is_bottom,
                labelleft=is_left,
            )

    save_figure(fig, title, path_pgf, path_pdf)
    return axes_flat


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
