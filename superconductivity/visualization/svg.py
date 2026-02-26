"""SVG export helpers.

This module provides small utilities to generate clean, print-ready axis
graphics as SVG/PNG using Matplotlib. The primary use case is creating
consistent axis arrows with tick marks and labels for laser cutting or
plot overlays.

Notes
-----
All physical sizes are specified in millimeters and are converted to Matplotlib
figure sizes internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def save_axis(
    values: np.ndarray,
    label: str,
    title: str,
    dataset: str | Path,
    ticks: Optional[np.ndarray] = None,
    ticklabels: Optional[Sequence[str]] = None,
    tickformat: str = ".2g",
    nticks: int = 5,
    vertical: bool = False,
    height_mm: float = 16.0,
    arrowlength_mm: float = 100.0,
    bleed_left_mm: float = 5.0,
    bleed_right_mm: float = 10.0,
    tick_len_mm: float = 2.0,
    tick_label_pad_mm: float = 1.0,
    label_pad_mm: float = 2.5,
    arrow_width_mm: float = 0.6,
    fontsize_mm: float = 5,
    arrowsize_mm: float = 7,
) -> plt.figure:
    """Save a single axis graphic as SVG/PNG.

    The axis is drawn in physical units (millimeters) using a Matplotlib figure
    and exported to ``<dataset>/svg/<title>.svg`` and
    ``<dataset>/png/<title>.png``.

    The arrow represents the data interval ``[min(values), max(values)]`` mapped
    to ``arrowlength_mm``. The arrow starts at ``bleed_left_mm`` and the
    resulting canvas is extended by ``bleed_left_mm`` and ``bleed_right_mm``.

    Parameters
    ----------
    values
        1D array used to infer the data limits when ``ticks`` is None.
    label
        Axis label text.
    title
        Output filename stem (without extension).
    dataset
        Output directory.
    ticks
        Optional tick positions in data units. If None, tick positions are
        generated using Matplotlib's tick locator with at most ``nticks`` ticks.
    ticklabels
        Optional explicit tick label strings corresponding to ``ticks``.
    tickformat
        Format string used for numeric tick labels when ``ticklabels`` is None.
    nticks
        Target number of ticks when ``ticks`` is None.
    vertical
        If True, draw a vertical axis (right side, ticks to the left). If
        False, draw a horizontal axis (top, ticks downward).
    height_mm
        Physical size in mm perpendicular to the axis direction.
    arrowlength_mm
        Physical axis length in mm corresponding to the data interval.
    bleed_left_mm, bleed_right_mm
        Extra margin in mm added before/after the data interval.
    tick_len_mm
        Tick length in mm.
    tick_label_pad_mm
        Padding in mm between tick marks and tick labels.
    label_pad_mm
        Padding in mm between the canvas edge and the axis label.
    arrow_width_mm
        Arrow line width in mm.
    fontsize_mm
        Font size in mm.
    arrowsize_mm
        Arrow head size in mm.

    Returns
    -------
    fig
        Matplotlib figure instance that was saved.

    Raises
    ------
    ValueError
        If the inferred limits are invalid or if provided ticks/labels are
        inconsistent.
    """
    v = np.asarray(values, dtype=float).ravel()

    lower = float(np.nanmin(v))
    upper = float(np.nanmax(v))

    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        raise ValueError("lim must be finite and have different values.")
    if lower > upper:
        lower, upper = upper, lower

    if ticks is None:
        # Use Matplotlib's tick locator to generate "nice" tick values.
        # nbins is the maximum number of intervals; number of ticks may be
        # nbins+1.
        nbins = max(int(nticks) - 1, 1)
        locator = mticker.MaxNLocator(nbins=nbins)
        ticks_arr = np.asarray(locator.tick_values(lower, upper), dtype=float)

        # Keep only ticks inside [lower, upper].
        ticks_arr = ticks_arr[(ticks_arr >= lower) & (ticks_arr <= upper)]

        # If the locator returned too many ticks, downsample to at most nticks.
        if ticks_arr.size > int(nticks):
            idx = np.linspace(0, ticks_arr.size - 1, int(nticks)).round().astype(int)
            ticks_arr = ticks_arr[idx]

        # As a final fallback, ensure we have at least two ticks.
        if ticks_arr.size < 2:
            ticks_arr = np.linspace(lower, upper, max(int(nticks), 2))
    else:
        ticks_arr = np.asarray(ticks, dtype=float).ravel()
        if ticks_arr.size == 0:
            raise ValueError("ticks must be non-empty.")

    if ticklabels is not None and len(ticklabels) != ticks_arr.size:
        raise ValueError("ticklabels must have same length as ticks.")

    long_mm = float(arrowlength_mm + bleed_left_mm + bleed_right_mm)
    if long_mm <= 0 or height_mm <= 0:
        raise ValueError("arrowlength_mm and height_mm must be positive.")

    fig_width_mm = height_mm if vertical else long_mm
    fig_height_mm = long_mm if vertical else height_mm

    fig_w = fig_width_mm / 25.4
    fig_h = fig_height_mm / 25.4

    mm_to_pt = 72.0 / 25.4
    lw_pt = float(arrow_width_mm * mm_to_pt)
    fontsize_pt = float(fontsize_mm * mm_to_pt)
    arrowsize_pt = float(arrowsize_mm * mm_to_pt)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])

    # Work in mm coordinates
    ax.set_xlim(0.0, fig_width_mm)
    ax.set_ylim(0.0, fig_height_mm)
    ax.axis("off")

    data_start_mm = float(bleed_left_mm)

    def data_to_mm(val: float) -> float:
        t = (val - lower) / (upper - lower)
        return data_start_mm + t * float(arrowlength_mm)

    # Place arrow as close as possible to the outer edge without clipping.
    # The arrow tip position is given by `xy` in `annotate`. We keep the tip
    # inside the canvas by using small edge insets.
    inset_top_mm = max(0.3 * arrowsize_mm, 0.1)
    inset_right_mm = 0.1

    if not vertical:
        y0 = fig_height_mm - inset_top_mm

        ax.annotate(
            "",
            xy=(fig_width_mm - inset_right_mm, y0),
            xytext=(data_start_mm, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=lw_pt,
                color="black",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrowsize_pt,
            ),
        )

        for i, t in enumerate(ticks_arr):
            x = data_to_mm(float(t))
            if x < 0.0 or x > fig_width_mm:
                continue

            ax.plot(
                [x, x],
                [y0, y0 - tick_len_mm],
                color="black",
                linewidth=lw_pt,
                solid_capstyle="butt",
            )

            lab = (
                format(float(t), tickformat)
                if ticklabels is None
                else str(ticklabels[i])
            )
            ax.text(
                x,
                y0 - tick_len_mm - tick_label_pad_mm,
                lab,
                ha="center",
                va="top",
                fontsize=fontsize_pt,
                color="black",
            )

        if label:
            ax.text(
                0.5 * fig_width_mm,
                label_pad_mm,
                label,
                ha="center",
                va="center",
                fontsize=fontsize_pt,
                color="black",
            )
    else:
        x0 = fig_width_mm - inset_top_mm

        ax.annotate(
            "",
            xy=(x0, fig_height_mm - inset_right_mm),
            xytext=(x0, data_start_mm),
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=lw_pt,
                color="black",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrowsize_pt,
            ),
        )

        if label:
            ax.text(
                label_pad_mm,
                0.5 * fig_height_mm,
                label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=fontsize_pt,
                color="black",
            )

        for i, t in enumerate(ticks_arr):
            y = data_to_mm(float(t))
            if y < 0.0 or y > fig_height_mm:
                continue

            ax.plot(
                [x0, x0 - tick_len_mm],
                [y, y],
                color="black",
                linewidth=lw_pt,
                solid_capstyle="butt",
            )

            lab = (
                format(float(t), tickformat)
                if ticklabels is None
                else str(ticklabels[i])
            )
            ax.text(
                x0 - tick_len_mm - tick_label_pad_mm,
                y,
                lab,
                ha="right",
                va="center",
                rotation=90,
                fontsize=fontsize_pt,
                color="black",
            )

    out_path_svg = Path(dataset, "svg") / f"{title}.svg"
    out_path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_svg, format="svg")

    out_path_png = Path(dataset, "png") / f"{title}.png"
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, format="png")

    plt.close(fig)
    return fig


def save_data(
    ax: plt.Axes,
    dataset: str | Path,
    title: str,
    *,
    width_mm: float = 100.0,
    height_mm: float = 100.0,
    dpi: int = 600,
    transparent: bool = True,
    tick_len_mm: float = 2.0,
    tick_width_mm: float = 0.6,
    grid_width_mm: float = 0.4,
) -> None:

    dataset = Path(dataset)
    dataset.mkdir(parents=True, exist_ok=True)

    # Accept either a Figure or an Axes-like object.
    target_ax = ax
    target_fig = target_ax.figure

    # Set exact output size.
    target_fig.set_size_inches(width_mm / 25.4, height_mm / 25.4, forward=True)
    target_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if target_ax is not None:
        target_ax.set_position([0, 0, 1, 1])

        mm_to_pt = 72.0 / 25.4
        tick_len_pt = float(tick_len_mm * mm_to_pt)
        tick_w_pt = float(tick_width_mm * mm_to_pt)
        grid_w_pt = float(grid_width_mm * mm_to_pt)

        # Keep ticks (if present) but move them inside and hide outer labels.
        target_ax.tick_params(
            direction="in",
            length=tick_len_pt,
            width=tick_w_pt,
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )

        target_ax.grid(True, which="both", linewidth=grid_w_pt)

        # Remove spines and axis labels/titles to keep only the inner content.
        for s in target_ax.spines.values():
            s.set_visible(False)
        target_ax.set_xlabel("")
        target_ax.set_ylabel("")
        target_ax.set_title("")

    # Save with exact canvas size (no tight bbox).
    out_path_svg = Path(dataset, "svg") / f"data {title}.svg"
    out_path_svg.parent.mkdir(parents=True, exist_ok=True)
    target_fig.savefig(
        out_path_svg,
        format="svg",
        dpi=dpi,
        transparent=transparent,
        bbox_inches=None,
        pad_inches=0,
    )

    out_path_png = Path(dataset, "png") / f"data {title}.png"
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    target_fig.savefig(
        out_path_png,
        format="png",
        dpi=dpi,
        transparent=transparent,
        bbox_inches=None,
        pad_inches=0,
    )

    return None
