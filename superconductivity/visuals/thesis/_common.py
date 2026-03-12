"""Shared helpers for thesis waterfall panels."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from superconductivity.utilities.types import LIM, NDArray64

from ..helper import check_xyz, get_xylim_indices, normalize_lim


def resolve_axis_range(
    values: NDArray64,
    lim: LIM = None,
) -> tuple[float, float]:
    """Resolve a finite axis range from data and an optional limit.

    Parameters
    ----------
    values
        Array containing the data range to inspect.
    lim
        Optional ``(lo, hi)`` bounds where either side may be ``None``.

    Returns
    -------
    lo, hi
        Finite axis limits.
    """
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).any():
        raise ValueError("Axis values must contain at least one finite value.")

    lower = float(np.nanmin(arr))
    upper = float(np.nanmax(arr))
    lo, hi = normalize_lim(lim)
    lo = lower if lo is None else float(lo)
    hi = upper if hi is None else float(hi)

    if lo == hi:
        pad = 0.5 if lo == 0.0 else 0.05 * abs(lo)
        lo -= pad
        hi += pad
    return lo, hi


def resolve_color_range(
    values: NDArray64,
    clim: LIM = None,
) -> tuple[float, float]:
    """Resolve finite color limits for a scalar field.

    Parameters
    ----------
    values
        Scalar field used for color mapping.
    clim
        Optional explicit color limits.

    Returns
    -------
    cmin, cmax
        Finite color limits.
    """
    return resolve_axis_range(values, clim)


def resolve_flat_panel_range(
    z_level: float = 0.0,
    *,
    pad: float = 0.5,
) -> tuple[float, float]:
    """Return a non-zero z-range anchored at a flat panel level.

    Parameters
    ----------
    z_level
        Z position of the flat panel.
    pad
        Half-depth used to keep a non-zero axis span for 3D rendering.

    Returns
    -------
    tuple of float
        Lower and upper z-limits, with the panel located at the lower bound.
    """
    pad_value = max(float(pad), 1e-9)
    return float(z_level), float(z_level + 2.0 * pad_value)


def prepare_waterfall_data(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
    *,
    xlim: LIM = None,
    ylim: LIM = None,
    trace_step: int = 1,
) -> tuple[NDArray64, NDArray64, NDArray64]:
    """Validate, crop, and optionally subsample waterfall input data.

    Parameters
    ----------
    x
        1D x-axis values of shape ``(Nx,)``.
    y
        1D y-axis values of shape ``(Ny,)``.
    z
        2D field of shape ``(Ny, Nx)``.
    xlim
        Optional x-range used to crop the data.
    ylim
        Optional y-range used to crop the data.
    trace_step
        Keep every ``trace_step``-th fixed-``y`` trace.

    Returns
    -------
    x_sel, y_sel, z_sel
        Cropped and subsampled arrays.
    """
    if int(trace_step) < 1:
        raise ValueError("trace_step must be at least 1.")

    x_arr, y_arr, z_arr = check_xyz(x=x, y=y, z=z)
    ix, iy = get_xylim_indices(
        x=np.asarray(x_arr, dtype=np.float64),
        y=np.asarray(y_arr, dtype=np.float64),
        xlim=xlim,
        ylim=ylim,
    )

    x_sel = np.asarray(x_arr[ix], dtype=np.float64)
    y_sel = np.asarray(y_arr[iy], dtype=np.float64)[:: int(trace_step)]
    z_sel = np.asarray(z_arr[np.ix_(iy, ix)], dtype=np.float64)[
        :: int(trace_step)
    ]
    return x_sel, y_sel, z_sel


def iter_finite_trace_segments(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
) -> Iterator[tuple[NDArray64, NDArray64, NDArray64]]:
    """Yield contiguous finite 3D line segments for a waterfall stack.

    Parameters
    ----------
    x
        1D x-axis values.
    y
        1D y-axis values.
    z
        2D field values ``z[y_i, x_j]``.

    Yields
    ------
    x_run, y_run, z_run
        Consecutive finite line segments.
    """
    for y_value, z_row in zip(y, z):
        finite = np.isfinite(x) & np.isfinite(z_row)
        if not np.any(finite):
            continue

        indices = np.flatnonzero(finite)
        split_at = np.flatnonzero(np.diff(indices) > 1) + 1
        for run in np.split(indices, split_at):
            if run.size < 2:
                continue

            x_run = x[run]
            y_run = np.full(run.size, float(y_value), dtype=np.float64)
            z_run = z_row[run]
            yield x_run, y_run, z_run


def hide_matplotlib_zaxis(ax: plt.Axes) -> None:
    """Hide the native Matplotlib z-axis decorations.

    Parameters
    ----------
    ax
        Matplotlib 3D axis to simplify.
    """
    ax.set_zticks([])
    ax.set_zticklabels([])
    ax.set_zlabel("")

    line = getattr(ax.zaxis, "line", None)
    if line is not None:
        try:
            line.set_visible(False)
        except AttributeError:
            line.set_linewidth(0.0)

    pane = getattr(ax.zaxis, "pane", None)
    if pane is not None:
        try:
            pane.set_visible(False)
        except AttributeError:
            pass

    if hasattr(ax.zaxis, "_axinfo"):
        ax.zaxis._axinfo["grid"]["linewidth"] = 0.0


def set_matplotlib_box_aspect(
    ax: plt.Axes,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    zoom: float = 1.0,
) -> None:
    """Set a data-like 3D box aspect when supported by Matplotlib.

    Parameters
    ----------
    ax
        Matplotlib 3D axis whose box aspect should be configured.
    x_range
        Visible x-axis range.
    y_range
        Visible y-axis range.
    z_range
        Visible z-axis range.
    zoom
        Optional Matplotlib 3D zoom factor used to make the box fill more
        of the subplot area when supported.
    """
    aspect = (
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0],
    )
    try:
        ax.set_box_aspect(aspect, zoom=float(zoom))
    except TypeError:
        ax.set_box_aspect(aspect)
    except AttributeError:
        pass


__all__ = [
    "hide_matplotlib_zaxis",
    "iter_finite_trace_segments",
    "prepare_waterfall_data",
    "resolve_axis_range",
    "resolve_color_range",
    "resolve_flat_panel_range",
    "set_matplotlib_box_aspect",
]
