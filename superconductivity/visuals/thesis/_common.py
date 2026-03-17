"""Shared helpers for thesis waterfall panels."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np

from superconductivity.style.cpd5 import seegrau10, seegrau65
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


def select_trace_rows(
    y: NDArray64,
    z: NDArray64,
    *,
    trace_count: int | None = None,
) -> tuple[NDArray64, NDArray64]:
    """Select a fixed number of waterfall traces.

    Parameters
    ----------
    y
        1D sweep values of shape ``(Ny,)``.
    z
        2D field of shape ``(Ny, Nx)``.
    trace_count
        Number of traces to keep. When omitted, all traces are kept. The
        first trace is always included and additional traces are spaced
        evenly over the available rows.

    Returns
    -------
    y_sel, z_sel
        Trace-limited sweep values and field.
    """
    if trace_count is None:
        return y, z

    keep_count = int(trace_count)
    if keep_count < 1:
        raise ValueError("trace_count must be at least 1.")

    total = int(len(y))
    if total == 0:
        return y, z
    if keep_count >= total:
        return y, z

    keep = np.rint(
        np.linspace(0, total - 1, keep_count, dtype=np.float64)
    ).astype(int)
    return y[keep], z[keep]


def resolve_axes_rect_inches(
    figsize: tuple[float, float],
    axes_rect: Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    """Convert an axes rectangle from inches to figure fractions.

    Parameters
    ----------
    figsize
        Figure size in inches ``(width, height)``.
    axes_rect
        Optional axes rectangle ``(left, bottom, width, height)`` in inches.

    Returns
    -------
    tuple of float or None
        Rectangle normalized to figure fractions, or ``None`` when no custom
        rectangle was requested.
    """
    if axes_rect is None:
        return None

    fig = np.asarray(tuple(figsize), dtype=np.float64)
    rect = np.asarray(tuple(axes_rect), dtype=np.float64)
    if fig.shape != (2,):
        raise ValueError("figsize must contain exactly two values.")
    if rect.shape != (4,):
        raise ValueError("axes_rect must contain four values.")
    if not np.all(np.isfinite(fig)):
        raise ValueError("figsize values must be finite.")
    if not np.all(np.isfinite(rect)):
        raise ValueError("axes_rect values must be finite.")
    if np.any(fig <= 0.0):
        raise ValueError("figsize values must be positive.")
    if np.any(rect[2:] <= 0.0):
        raise ValueError("axes_rect width and height must be positive.")
    if np.any(rect[:2] < 0.0):
        raise ValueError("axes_rect left and bottom must be non-negative.")
    if rect[0] + rect[2] > fig[0] or rect[1] + rect[3] > fig[1]:
        raise ValueError("axes_rect must fit inside the figure size.")

    return (
        float(rect[0] / fig[0]),
        float(rect[1] / fig[1]),
        float(rect[2] / fig[0]),
        float(rect[3] / fig[1]),
    )


def resolve_axis_triplet_float(
    value: float | Sequence[float] | None,
    *,
    name: str,
) -> tuple[float, float, float] | None:
    """Resolve an optional scalar or ``(x, y, z)`` float triplet.

    Parameters
    ----------
    value
        Scalar applied to all axes, explicit triplet, or ``None``.
    name
        Parameter name used in validation errors.

    Returns
    -------
    tuple of float or None
        Normalized ``(x, y, z)`` values, or ``None`` when omitted.
    """
    if value is None:
        return None

    if np.isscalar(value):
        values = np.full(3, float(value), dtype=np.float64)
    else:
        values = np.asarray(tuple(value), dtype=np.float64)

    if values.shape != (3,):
        raise ValueError(f"{name} must be a scalar or contain three values.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} values must be finite.")
    return tuple(float(component) for component in values)


def resolve_axis_triplet_value(
    value: Sequence[object | None] | None,
    *,
    name: str,
) -> tuple[object | None, object | None, object | None] | None:
    """Resolve an optional ``(x, y, z)`` value triplet.

    Parameters
    ----------
    value
        Explicit triplet or ``None``.
    name
        Parameter name used in validation errors.

    Returns
    -------
    tuple or None
        Normalized ``(x, y, z)`` values, or ``None`` when omitted.
    """
    if value is None:
        return None

    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values.")
    return values[0], values[1], values[2]


def _resolve_matplotlib_spacing(
    spacing: float | Sequence[float] | None,
    *,
    zero_offset: float,
    name: str,
) -> tuple[float, float, float] | None:
    """Map normalized thesis spacing onto Matplotlib 3D pad values.

    Parameters
    ----------
    spacing
        Scalar or ``(x, y, z)`` spacing where ``0`` means the axis line and
        ``1`` means Matplotlib's current default 3D offset.
    zero_offset
        Positive built-in Matplotlib 3D offset in points corresponding to a
        normalized spacing of ``0``.
    name
        Parameter name used in validation errors.

    Returns
    -------
    tuple of float or None
        Matplotlib pad values in points, or ``None`` when omitted.
    """
    values = resolve_axis_triplet_float(
        spacing,
        name=name,
    )
    if values is None:
        return None
    return tuple(
        float(zero_offset * (component - 1.0)) for component in values
    )


def apply_matplotlib_axis_spacing(
    ax: plt.Axes,
    *,
    labelspacing: float | Sequence[float] | None = None,
    ticklabelspacing: float | Sequence[float] | None = None,
) -> None:
    """Apply normalized per-axis label and tick-label spacing.

    Parameters
    ----------
    ax
        Matplotlib axis to modify.
    labelspacing
        Scalar or ``(x, y, z)`` spacing where ``0`` means the axis line and
        ``1`` means Matplotlib's default 3D label offset.
    ticklabelspacing
        Scalar or ``(x, y, z)`` spacing where ``0`` means the axis line and
        ``1`` means Matplotlib's default 3D tick-label offset.
    """
    labelpad_values = _resolve_matplotlib_spacing(
        labelspacing,
        zero_offset=21.0,
        name="labelspacing",
    )
    if labelpad_values is not None:
        ax.xaxis.labelpad = labelpad_values[0]
        ax.yaxis.labelpad = labelpad_values[1]
        ax.zaxis.labelpad = labelpad_values[2]

    tickpad_values = _resolve_matplotlib_spacing(
        ticklabelspacing,
        zero_offset=8.0,
        name="ticklabelspacing",
    )
    if tickpad_values is not None:
        for axis_name, pad in zip(("x", "y", "z"), tickpad_values):
            getattr(ax, f"{axis_name}axis").set_tick_params(pad=pad)


def apply_matplotlib_axis_ticks(
    ax: plt.Axes,
    *,
    ticks: Sequence[object | None] | None = None,
    ticklabels: Sequence[object | None] | None = None,
) -> None:
    """Apply optional per-axis ticks and tick labels.

    Parameters
    ----------
    ax
        Matplotlib axis to modify.
    ticks
        Optional ``(x, y, z)`` tuple where each entry is a tick sequence or
        ``None`` to leave that axis unchanged.
    ticklabels
        Optional ``(x, y, z)`` tuple where each entry is a tick-label
        sequence or ``None`` to leave that axis unchanged.
    """
    tick_values = resolve_axis_triplet_value(
        ticks,
        name="ticks",
    )
    ticklabel_values = resolve_axis_triplet_value(
        ticklabels,
        name="ticklabels",
    )
    for index, axis_name in enumerate(("x", "y", "z")):
        axis_ticks = None if tick_values is None else tick_values[index]
        axis_labels = None if ticklabel_values is None else ticklabel_values[index]
        if axis_ticks is not None:
            getattr(ax, f"set_{axis_name}ticks")(axis_ticks)
        if axis_labels is not None:
            if axis_ticks is None:
                current_ticks = getattr(ax, f"get_{axis_name}ticks")()
                getattr(ax, f"set_{axis_name}ticks")(current_ticks)
            getattr(ax, f"set_{axis_name}ticklabels")(axis_labels)


def apply_matplotlib_zaxis_side(
    ax: plt.Axes,
    *,
    side: str = "default",
) -> None:
    """Place z-axis labels and tick labels on the requested visual side.

    Parameters
    ----------
    ax
        Matplotlib 3D axis to modify.
    side
        One of ``"default"``, ``"left"``, or ``"right"``. ``"left"`` and
        ``"right"`` are resolved from Matplotlib's internal ``"lower"`` and
        ``"upper"`` z-axis positions for the current 3D view.
    """
    if side == "default":
        ax.zaxis.set_label_position("default")
        ax.zaxis.set_ticks_position("default")
        return
    if side not in {"left", "right"}:
        raise ValueError("side must be 'default', 'left', or 'right'.")

    x_centers: dict[str, float] = {}
    for position in ("lower", "upper"):
        ax.zaxis.set_label_position(position)
        ax.zaxis.set_ticks_position(position)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        tick_x = []
        for tick in ax.zaxis.get_major_ticks():
            if not tick.label1.get_text():
                continue
            bbox = tick.label1.get_window_extent(renderer)
            tick_x.append(float(0.5 * (bbox.x0 + bbox.x1)))

        if tick_x:
            x_centers[position] = float(np.mean(tick_x))
            continue

        if ax.zaxis.label.get_text():
            bbox = ax.zaxis.label.get_window_extent(renderer)
            x_centers[position] = float(0.5 * (bbox.x0 + bbox.x1))
            continue

        raise ValueError("Could not resolve the projected z-axis side.")

    if side == "left":
        resolved = min(x_centers, key=x_centers.get)
    else:
        resolved = max(x_centers, key=x_centers.get)
    ax.zaxis.set_label_position(resolved)
    ax.zaxis.set_ticks_position(resolved)


def redraw_matplotlib_axes_on_top(
    ax: plt.Axes,
    *,
    axis_names: Sequence[str] = ("x", "y", "z"),
) -> None:
    """Redraw selected 3D axes after collections so they stay visible.

    Parameters
    ----------
    ax
        Matplotlib 3D axis to modify.
    axis_names
        Names of axes that should be redrawn on top of collections.
    """
    resolved_axis_names = tuple(axis_names)
    for axis_name in resolved_axis_names:
        if getattr(ax, f"{axis_name}axis", None) is None:
            raise ValueError(f"Unknown axis name: {axis_name!r}.")

    if hasattr(ax, "_thesis_front_axis_names"):
        ax._thesis_front_axis_names = resolved_axis_names
        return

    original_draw = ax.draw

    def _wrapped_draw(self: plt.Axes, renderer: object) -> None:
        original_draw(renderer)
        if not getattr(self, "_axis3don", True):
            return
        for axis_name in self._thesis_front_axis_names:
            getattr(self, f"{axis_name}axis").draw(renderer)

    ax._thesis_front_axis_names = resolved_axis_names
    ax.draw = MethodType(_wrapped_draw, ax)


def hide_matplotlib_boundary_gridlines(
    ax: plt.Axes,
    *,
    axis_names: Sequence[str] = ("x", "y", "z"),
    atol: float = 1e-12,
) -> None:
    """Suppress 3D grid segments that lie on pane boundaries.

    Matplotlib draws grid lines for ticks at the axis limits, which creates
    visible lines where colored back panes meet. This wrapper keeps the grid
    but skips the boundary-aligned segments.

    Parameters
    ----------
    ax
        Matplotlib 3D axis to modify.
    axis_names
        Names of axes whose boundary grid segments should be suppressed.
    atol
        Absolute tolerance used when matching tick locations to the axis
        limits.
    """
    resolved_axis_names = tuple(axis_names)
    for axis_name in resolved_axis_names:
        axis_obj = getattr(ax, f"{axis_name}axis", None)
        if axis_obj is None:
            raise ValueError(f"Unknown axis name: {axis_name!r}.")

        if hasattr(axis_obj, "_thesis_draw_grid_original"):
            continue

        original_draw_grid = axis_obj.draw_grid
        axis_obj._thesis_draw_grid_original = original_draw_grid

        def _wrapped_draw_grid(
            self: plt.Axes,
            renderer: object,
            *,
            _atol: float = float(atol),
        ) -> None:
            if not self.axes._draw_grid:
                return

            renderer.open_group("grid3d", gid=self.get_gid())
            ticks = self._update_ticks()
            if len(ticks):
                info = self._axinfo
                index = info["i"]
                mins, maxs, tc, highs = self._get_coord_info()

                lower = float(min(mins[index], maxs[index]))
                upper = float(max(mins[index], maxs[index]))
                keep_locs = []
                for tick in ticks:
                    loc = float(tick.get_loc())
                    if np.isclose(loc, lower, rtol=0.0, atol=_atol):
                        continue
                    if np.isclose(loc, upper, rtol=0.0, atol=_atol):
                        continue
                    keep_locs.append(loc)

                if keep_locs:
                    minmax = np.where(highs, maxs, mins)
                    maxmin = np.where(~highs, maxs, mins)
                    xyz0 = np.tile(minmax, (len(keep_locs), 1))
                    xyz0[:, index] = keep_locs
                    lines = np.stack([xyz0, xyz0, xyz0], axis=1)
                    lines[:, 0, index - 2] = maxmin[index - 2]
                    lines[:, 2, index - 1] = maxmin[index - 1]
                    self.gridlines.set_segments(lines)
                    gridinfo = info["grid"]
                    self.gridlines.set_color(gridinfo["color"])
                    self.gridlines.set_linewidth(gridinfo["linewidth"])
                    self.gridlines.set_linestyle(gridinfo["linestyle"])
                    self.gridlines.do_3d_projection()
                    self.gridlines.draw(renderer)

            renderer.close_group("grid3d")

        axis_obj.draw_grid = MethodType(_wrapped_draw_grid, axis_obj)


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


def set_matplotlib_axis_visible(
    ax: plt.Axes,
    *,
    axis_name: str,
    visible: bool,
) -> None:
    """Show or hide one Matplotlib axis label and tick labels.

    Parameters
    ----------
    ax
        Matplotlib axis to modify.
    axis_name
        Axis name, for example ``"x"``, ``"y"``, or ``"z"``.
    visible
        Whether the axis label and tick labels should remain visible.
    """
    if visible:
        return

    getattr(ax, f"set_{axis_name}label")("")
    getattr(ax, f"set_{axis_name}ticklabels")([])


def hide_matplotlib_panes(
    ax: plt.Axes,
    *,
    axis_names: Sequence[str] = ("x", "y", "z"),
) -> None:
    """Hide selected Matplotlib 3D panes and their pane grids.

    Parameters
    ----------
    ax
        Matplotlib 3D axis to simplify.
    axis_names
        Names of axes whose panes should be hidden.
    """
    for axis_name in axis_names:
        axis_obj = getattr(ax, f"{axis_name}axis", None)
        if axis_obj is None:
            continue

        pane = getattr(axis_obj, "pane", None)
        if pane is not None:
            try:
                pane.set_visible(False)
            except AttributeError:
                try:
                    pane.fill = False
                except AttributeError:
                    pass

        if hasattr(axis_obj, "_axinfo"):
            axis_obj._axinfo["grid"]["linewidth"] = 0.0


def set_matplotlib_background_style(
    ax: plt.Axes,
    *,
    axis_names: Sequence[str] = ("x", "y", "z"),
) -> None:
    """Apply the thesis 3D pane and grid styling.

    Parameters
    ----------
    ax
        Matplotlib 3D axis to modify.
    axis_names
        Names of axes whose panes and grids should be styled.
    """
    fig_patch = getattr(ax.figure, "patch", None)
    if fig_patch is not None:
        fig_patch.set_alpha(0.0)
        fig_patch.set_facecolor("none")

    ax_patch = getattr(ax, "patch", None)
    if ax_patch is not None:
        ax_patch.set_alpha(0.0)
        ax_patch.set_facecolor("none")

    pane_color = (
        float(seegrau10[0]),
        float(seegrau10[1]),
        float(seegrau10[2]),
        1.0,
    )
    grid_color = (
        float(seegrau65[0]),
        float(seegrau65[1]),
        float(seegrau65[2]),
        1.0,
    )
    pane_edge_color = (0.0, 0.0, 0.0, 0.0)
    for axis_name in axis_names:
        axis_obj = getattr(ax, f"{axis_name}axis", None)
        if axis_obj is None:
            continue

        pane = getattr(axis_obj, "pane", None)
        if pane is not None:
            try:
                pane.set_facecolor(pane_color)
            except AttributeError:
                pass
            try:
                pane.set_alpha(1.0)
            except AttributeError:
                pass
            try:
                pane.set_edgecolor(pane_edge_color)
            except AttributeError:
                pass
            try:
                pane.set_linewidth(0.0)
            except AttributeError:
                pass

        if hasattr(axis_obj, "_axinfo"):
            axis_obj._axinfo["grid"]["color"] = grid_color

    hide_matplotlib_boundary_gridlines(ax, axis_names=axis_names)


def flatten_box_aspect_z(
    box_aspect: Sequence[float] | None,
    *,
    z_scale: float = 1e-3,
) -> Sequence[float] | None:
    """Return a box aspect with a strongly compressed z component.

    Parameters
    ----------
    box_aspect
        Optional displayed box aspect ratio ``(x_ratio, y_ratio, z_ratio)``.
    z_scale
        Multiplicative factor applied to the z component.

    Returns
    -------
    tuple of float or None
        Aspect ratio with a collapsed z component, or ``None`` when no
        explicit aspect ratio was requested.
    """
    if box_aspect is None:
        return None

    values = np.asarray(tuple(box_aspect), dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("box_aspect must contain exactly three values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("box_aspect values must be finite.")
    if np.any(values <= 0.0):
        raise ValueError("box_aspect values must be positive.")

    flattened = values.copy()
    flattened[2] = max(
        float(flattened[2] * z_scale),
        float(np.finfo(np.float64).eps),
    )
    return tuple(float(value) for value in flattened)


def set_matplotlib_box_aspect(
    ax: plt.Axes,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    box_aspect: Sequence[float] | None = None,
    zoom: float = 1.0,
) -> None:
    """Set a 3D box aspect when supported by Matplotlib.

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
    box_aspect
        Optional displayed box aspect ratio ``(x_ratio, y_ratio, z_ratio)``.
        When omitted, the data spans are used.
    zoom
        Optional Matplotlib 3D zoom factor used to make the box fill more
        of the subplot area when supported.
    """
    zoom_value = float(zoom)
    if not np.isfinite(zoom_value):
        raise ValueError("zoom must be finite.")
    if zoom_value <= 0.0:
        raise ValueError("zoom must be positive.")

    if box_aspect is None:
        aspect = (
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            z_range[1] - z_range[0],
        )
    else:
        values = np.asarray(tuple(box_aspect), dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(
                "box_aspect must contain exactly three values."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("box_aspect values must be finite.")
        if np.any(values <= 0.0):
            raise ValueError("box_aspect values must be positive.")
        aspect = tuple(float(value) for value in values)
    try:
        ax.set_box_aspect(aspect, zoom=zoom_value)
    except TypeError:
        ax.set_box_aspect(aspect)
    except AttributeError:
        pass


__all__ = [
    "apply_matplotlib_axis_ticks",
    "apply_matplotlib_axis_spacing",
    "apply_matplotlib_zaxis_side",
    "flatten_box_aspect_z",
    "redraw_matplotlib_axes_on_top",
    "hide_matplotlib_panes",
    "hide_matplotlib_boundary_gridlines",
    "hide_matplotlib_zaxis",
    "iter_finite_trace_segments",
    "prepare_waterfall_data",
    "resolve_axis_range",
    "resolve_axes_rect_inches",
    "resolve_axis_triplet_value",
    "resolve_axis_triplet_float",
    "resolve_color_range",
    "resolve_flat_panel_range",
    "select_trace_rows",
    "set_matplotlib_axis_visible",
    "set_matplotlib_background_style",
    "set_matplotlib_box_aspect",
]
