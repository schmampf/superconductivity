"""Smoke tests for thesis-oriented waterfall exports."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from superconductivity.style.cpd5 import seegrau10, seegrau65
from superconductivity.visuals.thesis.heatmap import (
    get_thesis_heatmap_matplotlib,
)
from superconductivity.visuals.thesis.latex import (
    build_stack_pgf,
    compile_thesis_preview,
    export_stacked_waterfall_thesis,
)
from superconductivity.visuals.thesis.surface import (
    get_thesis_surface_matplotlib,
)
from superconductivity.visuals.thesis.waterfall import (
    get_thesis_waterfall_matplotlib,
)


def _waterfall_data(
    nx: int = 41,
    ny: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a smooth stacked spectrum dataset."""
    x = np.linspace(850.0, 1350.0, nx, dtype=np.float64)
    y = np.linspace(600.0, 840.0, ny, dtype=np.float64)
    center = 1080.0 + 90.0 * np.sin((y - 600.0) / 55.0)
    width = 24.0 + 4.0 * np.cos((y - 600.0) / 75.0)
    amplitude = 900.0 + 400.0 * np.cos((y - 600.0) / 40.0)
    z = amplitude[:, None] * np.exp(
        -((x[None, :] - center[:, None]) ** 2) / (2.0 * width[:, None] ** 2)
    )
    return x, y, z.astype(np.float64)


def _relative_box_aspect(ax: matplotlib.axes.Axes) -> np.ndarray:
    """Return the applied 3D box aspect normalized to the x direction."""
    aspect = np.asarray(ax.get_box_aspect(), dtype=np.float64)
    return aspect / aspect[0]


def _box_scale(ax: matplotlib.axes.Axes) -> float:
    """Return one absolute component of the stored 3D box aspect."""
    return float(np.asarray(ax.get_box_aspect(), dtype=np.float64)[0])


def _axes_bounds(
    ax: matplotlib.axes.Axes,
    *,
    original: bool = True,
) -> tuple[float, float, float, float]:
    """Return the axes rectangle in figure fractions."""
    bounds = ax.get_position(original=original).bounds
    return tuple(float(value) for value in bounds)


def _tick_base_pads(
    ax: matplotlib.axes.Axes,
) -> tuple[float, float, float]:
    """Return the configured base tick-label padding for x/y/z."""
    def _axis_pad(axis: matplotlib.axis.Axis) -> float:
        ticks = axis.get_major_ticks()
        if not ticks:
            return float("nan")
        return float(ticks[0]._base_pad)

    return (
        _axis_pad(ax.xaxis),
        _axis_pad(ax.yaxis),
        _axis_pad(ax.zaxis),
    )


def _tick_texts(
    ax: matplotlib.axes.Axes,
    axis_name: str,
) -> tuple[str, ...]:
    """Return the current tick-label texts for one axis."""
    return tuple(
        tick.get_text()
        for tick in getattr(ax, f"get_{axis_name}ticklabels")()
    )


def _zaxis_text_center_x(
    ax: matplotlib.axes.Axes,
) -> float:
    """Return the projected x-center of the z-axis text."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    tick_x = []
    for tick in ax.zaxis.get_major_ticks():
        if not tick.label1.get_text():
            continue
        bbox = tick.label1.get_window_extent(renderer)
        tick_x.append(float(0.5 * (bbox.x0 + bbox.x1)))
    if tick_x:
        return float(np.mean(tick_x))

    bbox = ax.zaxis.label.get_window_extent(renderer)
    return float(0.5 * (bbox.x0 + bbox.x1))


def _axes_center_x(
    ax: matplotlib.axes.Axes,
) -> float:
    """Return the projected x-center of the axes box."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer)
    return float(0.5 * (bbox.x0 + bbox.x1))


def _surface_edgecolor_shape(
    ax: matplotlib.axes.Axes,
) -> tuple[int, int]:
    """Return the edgecolor array shape of the first surface collection."""
    return tuple(int(value) for value in ax.collections[0].get_edgecolor().shape)


def _surface_face_count(
    ax: matplotlib.axes.Axes,
) -> int:
    """Return the number of colored faces in the first surface collection."""
    return int(ax.collections[0].get_facecolor().shape[0])


def _heatmap_artist(
    ax: matplotlib.axes.Axes,
) -> matplotlib.artist.Artist:
    """Return the default warped heatmap artist."""
    return ax.artists[0]


def _pane_alpha(
    ax: matplotlib.axes.Axes,
    axis_name: str,
) -> float:
    """Return the facecolor alpha of one 3D pane."""
    pane = getattr(ax, f"{axis_name}axis").pane
    facecolor = pane.get_facecolor()
    return float(facecolor[3])


def _pane_rgb(
    ax: matplotlib.axes.Axes,
    axis_name: str,
) -> tuple[float, float, float]:
    """Return the facecolor rgb tuple of one 3D pane."""
    facecolor = getattr(ax, f"{axis_name}axis").pane.get_facecolor()
    return tuple(float(value) for value in facecolor[:3])


def _grid_rgb(
    ax: matplotlib.axes.Axes,
    axis_name: str,
) -> tuple[float, float, float]:
    """Return the grid rgb tuple for one 3D axis."""
    color = getattr(ax, f"{axis_name}axis")._axinfo["grid"]["color"]
    return tuple(float(value) for value in color[:3])


def _pane_edge_linewidth(
    ax: matplotlib.axes.Axes,
    axis_name: str,
) -> float:
    """Return the outline linewidth of one 3D pane."""
    return float(getattr(ax, f"{axis_name}axis").pane.get_linewidth())


def _grid_axis_locations(
    ax: matplotlib.axes.Axes,
    axis_name: str,
) -> np.ndarray:
    """Return the constant axis coordinate of each 3D grid segment."""
    ax.figure.canvas.draw()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
    segments = np.asarray(
        getattr(ax, f"{axis_name}axis").gridlines._segments3d,
        dtype=np.float64,
    )
    return segments[:, 0, axis_index]


def test_thesis_panel_builders_smoke() -> None:
    """The thesis panel helpers should return native Matplotlib 3D axes."""
    x, y, z = _waterfall_data()

    fig_w, ax_w = get_thesis_waterfall_matplotlib(
        x,
        y,
        z,
        trace_step=3,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )
    fig_s, ax_s = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        trace_step=2,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )
    fig_h, ax_h = get_thesis_heatmap_matplotlib(
        x,
        y,
        z,
        trace_step=2,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )

    assert fig_w is ax_w.figure
    assert fig_s is ax_s.figure
    assert fig_h is ax_h.figure
    assert ax_w.name == "3d"
    assert ax_s.name == "3d"
    assert ax_h.name == "3d"
    assert ax_w.get_zlabel() == "Counts"
    assert ax_s.get_zlabel() == "Counts"
    assert ax_h.get_zlabel() == ""
    assert len(ax_w.lines) == 3
    assert len(ax_h.get_zticks()) == 0
    assert ax_w.get_ylim()[0] > ax_w.get_ylim()[1]
    assert ax_s.get_ylim()[0] > ax_s.get_ylim()[1]
    assert ax_h.get_ylim()[0] > ax_h.get_ylim()[1]
    np.testing.assert_allclose(
        _relative_box_aspect(ax_w),
        np.ones(3),
    )
    np.testing.assert_allclose(
        _relative_box_aspect(ax_s),
        np.ones(3),
    )
    np.testing.assert_allclose(
        _relative_box_aspect(ax_h),
        np.ones(3),
    )
    assert ax_h.zaxis.get_visible()
    assert not ax_h.zaxis.line.get_visible()
    assert not ax_h.xaxis.pane.get_visible()
    assert not ax_h.yaxis.pane.get_visible()
    assert _surface_edgecolor_shape(ax_s) == (0, 4)
    assert ax_s.collections[0].get_rasterized()
    assert len(ax_h.artists) == 1
    assert _heatmap_artist(ax_h).__class__.__name__ == "_WarpedHeatmapImageArtist"
    assert _heatmap_artist(ax_h)._source_rgba.shape[:2] == z[::2].shape
    np.testing.assert_allclose(
        ax_h.get_zlim(),
        ax_s.get_zlim(),
    )


def test_thesis_panels_use_konstanz_pane_and_grid_colors() -> None:
    """Thesis 3D panels should use the configured Konstanz grey styling."""
    x, y, z = _waterfall_data()

    fig_w, ax_w = get_thesis_waterfall_matplotlib(x, y, z)
    fig_s, ax_s = get_thesis_surface_matplotlib(x, y, z)
    fig_h, ax_h = get_thesis_heatmap_matplotlib(x, y, z)

    assert fig_w.patch.get_alpha() == 0.0
    assert fig_s.patch.get_alpha() == 0.0
    assert fig_h.patch.get_alpha() == 0.0
    assert ax_w.patch.get_alpha() == 0.0
    assert ax_s.patch.get_alpha() == 0.0
    assert ax_h.patch.get_alpha() == 0.0
    np.testing.assert_allclose(_pane_rgb(ax_w, "x"), seegrau10)
    np.testing.assert_allclose(_pane_rgb(ax_w, "y"), seegrau10)
    np.testing.assert_allclose(_pane_rgb(ax_w, "z"), seegrau10)
    np.testing.assert_allclose(_pane_rgb(ax_s, "x"), seegrau10)
    np.testing.assert_allclose(_pane_rgb(ax_s, "y"), seegrau10)
    np.testing.assert_allclose(_pane_rgb(ax_s, "z"), seegrau10)
    assert _pane_alpha(ax_w, "x") == 1.0
    assert _pane_alpha(ax_w, "y") == 1.0
    assert _pane_alpha(ax_w, "z") == 1.0
    assert _pane_alpha(ax_s, "x") == 1.0
    assert _pane_alpha(ax_s, "y") == 1.0
    assert _pane_alpha(ax_s, "z") == 1.0
    assert _pane_edge_linewidth(ax_w, "x") == 0.0
    assert _pane_edge_linewidth(ax_w, "y") == 0.0
    assert _pane_edge_linewidth(ax_w, "z") == 0.0
    assert _pane_edge_linewidth(ax_s, "x") == 0.0
    assert _pane_edge_linewidth(ax_s, "y") == 0.0
    assert _pane_edge_linewidth(ax_s, "z") == 0.0
    assert ax_w.xaxis.line.get_visible()
    assert ax_w.yaxis.line.get_visible()
    assert ax_w.zaxis.line.get_visible()
    assert ax_s.xaxis.line.get_visible()
    assert ax_s.yaxis.line.get_visible()
    assert ax_s.zaxis.line.get_visible()
    np.testing.assert_allclose(_grid_rgb(ax_w, "x"), seegrau65)
    np.testing.assert_allclose(_grid_rgb(ax_w, "y"), seegrau65)
    np.testing.assert_allclose(_grid_rgb(ax_w, "z"), seegrau65)
    np.testing.assert_allclose(_grid_rgb(ax_s, "x"), seegrau65)
    np.testing.assert_allclose(_grid_rgb(ax_s, "y"), seegrau65)
    np.testing.assert_allclose(_grid_rgb(ax_s, "z"), seegrau65)
    x_grid_locs = _grid_axis_locations(ax_s, "x")
    assert not np.any(np.isclose(x_grid_locs, ax_s.get_xlim()[0]))
    assert not np.any(np.isclose(x_grid_locs, ax_s.get_xlim()[1]))


def test_thesis_heatmap_redraws_axes_on_top() -> None:
    """Heatmap axes should be redrawn above the surface collection."""
    x, y, z = _waterfall_data()

    fig, ax = get_thesis_heatmap_matplotlib(
        x,
        y,
        z,
        show_zaxis=False,
    )
    draw_counts = {"x": 0, "y": 0, "z": 0}

    for axis_name in ("x", "y", "z"):
        axis = getattr(ax, f"{axis_name}axis")
        original_draw = axis.draw

        def _wrapped_draw(
            renderer: object,
            *,
            _axis_name: str = axis_name,
            _original_draw: object = original_draw,
        ) -> None:
            draw_counts[_axis_name] += 1
            _original_draw(renderer)

        axis.draw = _wrapped_draw

    fig.canvas.draw()

    assert draw_counts["x"] >= 2
    assert draw_counts["y"] >= 2
    assert draw_counts["z"] == 1


def test_thesis_panel_builders_accept_custom_box_aspect() -> None:
    """Custom thesis box aspects should be applied consistently."""
    x, y, z = _waterfall_data()
    box_aspect = (1.0, 2.5, 0.75)

    _, ax_w = get_thesis_waterfall_matplotlib(
        x,
        y,
        z,
        box_aspect=box_aspect,
    )
    _, ax_s = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        box_aspect=box_aspect,
    )
    _, ax_h = get_thesis_heatmap_matplotlib(
        x,
        y,
        z,
        box_aspect=box_aspect,
    )

    expected = np.asarray(box_aspect, dtype=np.float64) / box_aspect[0]
    np.testing.assert_allclose(
        _relative_box_aspect(ax_w),
        expected,
    )
    np.testing.assert_allclose(
        _relative_box_aspect(ax_s),
        expected,
    )
    np.testing.assert_allclose(
        _relative_box_aspect(ax_h),
        expected,
    )


def test_thesis_heatmap_vector_cells_mode_keeps_one_face_per_bin() -> None:
    """Vector-cell mode should still render one polygon per retained bin."""
    x, y, z = _waterfall_data()

    _, ax = get_thesis_heatmap_matplotlib(
        x,
        y,
        z,
        trace_step=2,
        heatmap_mode="vector_cells",
    )

    assert _surface_edgecolor_shape(ax) == (0, 4)
    assert _surface_face_count(ax) == z[::2].size


def test_thesis_surface_keeps_native_face_count() -> None:
    """Surface plotting should keep the native mesh without downsampling."""
    x, y, z = _waterfall_data(nx=9, ny=5)

    _, ax = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_x_oversample=1,
    )

    assert _surface_face_count(ax) == (len(x) - 1) * (len(y) - 1)


def test_thesis_surface_allows_x_oversampling() -> None:
    """Surface plotting can densify the mesh along x only."""
    x, y, z = _waterfall_data(nx=9, ny=5)

    _, ax = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_x_oversample=3,
    )

    assert _surface_face_count(ax) == 3 * (len(x) - 1) * (len(y) - 1)


def test_thesis_surface_colors_follow_height() -> None:
    """Surface colors should come from the surface heights, not fixed faces."""
    x, y, z = _waterfall_data(nx=9, ny=5)

    _, ax = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_shading=False,
        surface_rasterized=False,
    )

    surface = ax.collections[0]
    facecolors = np.round(surface.get_facecolor()[:, :3], decimals=6)
    assert np.unique(facecolors, axis=0).shape[0] > 1


def test_thesis_surface_shading_changes_facecolors() -> None:
    """Optional light shading should alter the rendered surface colors."""
    x, y, z = _waterfall_data(nx=9, ny=5)

    _, ax_flat = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_shading=False,
        surface_rasterized=False,
    )
    _, ax_shaded = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_shading=True,
        surface_shade_strength=1.0,
        surface_rasterized=False,
    )

    flat = ax_flat.collections[0].get_facecolor()
    shaded = ax_shaded.collections[0].get_facecolor()
    assert flat.shape == shaded.shape
    assert not np.allclose(flat, shaded)


def test_thesis_surface_allows_vector_opt_out() -> None:
    """Surface rasterization can be disabled explicitly."""
    x, y, z = _waterfall_data()

    _, ax = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_rasterized=False,
    )

    assert not ax.collections[0].get_rasterized()


def test_thesis_panel_builders_accept_normalized_axis_spacing() -> None:
    """Axis label and tick-label spacing should be normalized."""
    x, y, z = _waterfall_data()

    _, ax = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        labelspacing=(0.0, 0.5, 1.0),
        ticklabelspacing=(0.0, 0.5, 1.0),
    )

    np.testing.assert_allclose(
        (ax.xaxis.labelpad, ax.yaxis.labelpad, ax.zaxis.labelpad),
        (-21.0, -10.5, 0.0),
    )
    np.testing.assert_allclose(
        _tick_base_pads(ax),
        (-8.0, -4.0, 0.0),
    )


def test_thesis_panel_builders_accept_custom_ticks_and_ticklabels() -> None:
    """Custom ticks and tick labels should be applied per axis."""
    x, y, z = _waterfall_data()
    ticks = (
        (float(x[0]), float(x[-1])),
        (float(y[0]), float(y[-1])),
        (float(np.min(z)), float(np.max(z))),
    )
    ticklabels = (
        ("x lo", "x hi"),
        ("y lo", "y hi"),
        ("z lo", "z hi"),
    )

    _, ax = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        ticks=ticks,
        ticklabels=ticklabels,
    )

    np.testing.assert_allclose(ax.get_xticks(), ticks[0])
    np.testing.assert_allclose(ax.get_yticks(), ticks[1])
    np.testing.assert_allclose(ax.get_zticks(), ticks[2])
    assert _tick_texts(ax, "x") == ticklabels[0]
    assert _tick_texts(ax, "y") == ticklabels[1]
    assert _tick_texts(ax, "z") == ticklabels[2]


def test_thesis_panel_builders_accept_visual_zaxis_side() -> None:
    """The z-axis side knob should resolve to visual left and right."""
    x, y, z = _waterfall_data()

    _, ax_left = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        azim=135.0,
        z_axis_side="left",
    )
    _, ax_right = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        azim=135.0,
        z_axis_side="right",
    )

    assert _zaxis_text_center_x(ax_left) < _zaxis_text_center_x(ax_right)


def test_thesis_panel_builders_accept_axes_rect_in_inches() -> None:
    """A custom axes rectangle in inches should place the panel predictably."""
    x, y, z = _waterfall_data()

    _, ax = get_thesis_waterfall_matplotlib(
        x,
        y,
        z,
        figsize=(6.0, 4.0),
        axes_rect=(0.75, 0.5, 4.5, 3.0),
    )

    np.testing.assert_allclose(
        _axes_bounds(ax),
        (0.75 / 6.0, 0.5 / 4.0, 4.5 / 6.0, 3.0 / 4.0),
    )


def test_thesis_waterfall_trace_count_keeps_first_trace() -> None:
    """Trace count should keep the first waterfall trace."""
    x, y, z = _waterfall_data()

    _, ax = get_thesis_waterfall_matplotlib(
        x,
        y,
        z,
        trace_count=2,
    )

    assert len(ax.lines) == 2
    np.testing.assert_allclose(
        ax.lines[0].get_data_3d()[1],
        np.full_like(ax.lines[0].get_data_3d()[1], y[0]),
    )


def test_thesis_panel_builders_allow_non_inverted_yaxis() -> None:
    """The default y-axis inversion can be disabled explicitly."""
    x, y, z = _waterfall_data()

    _, ax = get_thesis_heatmap_matplotlib(
        x,
        y,
        z,
        invert_yaxis=False,
    )

    assert ax.get_ylim()[0] < ax.get_ylim()[1]


def test_thesis_panel_builders_allow_inverted_xaxis() -> None:
    """The thesis panel builders should allow reversed x-axis direction."""
    x, y, z = _waterfall_data()

    _, ax_w = get_thesis_waterfall_matplotlib(
        x,
        y,
        z,
        invert_xaxis=True,
    )
    _, ax_s = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        invert_xaxis=True,
    )
    _, ax_h = get_thesis_heatmap_matplotlib(
        x,
        y,
        z,
        invert_xaxis=True,
    )

    assert ax_w.get_xlim()[0] > ax_w.get_xlim()[1]
    assert ax_s.get_xlim()[0] > ax_s.get_xlim()[1]
    assert ax_h.get_xlim()[0] > ax_h.get_xlim()[1]


def test_thesis_panel_builders_allow_axis_visibility_flags() -> None:
    """Axis labels and ticks can be hidden explicitly."""
    x, y, z = _waterfall_data()

    _, ax = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        show_xaxis=False,
        show_yaxis=False,
        show_zaxis=False,
    )

    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_zlabel() == ""
    assert len(ax.get_xticks()) > 0
    assert len(ax.get_yticks()) > 0
    assert len(ax.get_zticks()) > 0


def test_thesis_panel_builders_accept_custom_box_zoom() -> None:
    """Panel box zoom should scale the rendered 3D object size."""
    x, y, z = _waterfall_data()

    _, ax_w_default = get_thesis_waterfall_matplotlib(x, y, z)
    _, ax_w_zoom = get_thesis_waterfall_matplotlib(
        x,
        y,
        z,
        box_zoom=0.6,
    )
    _, ax_h_default = get_thesis_heatmap_matplotlib(x, y, z)
    _, ax_h_zoom = get_thesis_heatmap_matplotlib(
        x,
        y,
        z,
        box_zoom=0.6,
    )

    np.testing.assert_allclose(
        _box_scale(ax_w_zoom) / _box_scale(ax_w_default),
        0.6,
    )
    np.testing.assert_allclose(
        _box_scale(ax_h_zoom) / _box_scale(ax_h_default),
        0.6,
    )


def test_export_stacked_waterfall_thesis_writes_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thesis export helper should write panel and stack assets."""
    x, y, z = _waterfall_data()
    saved_files: list[Path] = []
    saved_box_scales: dict[str, float] = {}
    saved_axis_state: dict[str, dict[str, object]] = {}
    saved_axis_padding: dict[str, dict[str, object]] = {}
    saved_axes_bounds: dict[str, tuple[float, float, float, float]] = {}
    saved_dpi: dict[str, float] = {}
    saved_backend: dict[str, str | None] = {}
    saved_rcparams: dict[str, dict[str, object]] = {}
    saved_geometry: dict[str, int] = {}
    displayed: dict[str, object] = {}
    custom_ticks = (
        (float(x[0]), float(x[-1])),
        (float(y[0]), float(y[-1])),
        (float(np.min(z)), float(np.max(z))),
    )
    custom_ticklabels = (
        ("x lo", "x hi"),
        ("y lo", "y hi"),
        ("z lo", "z hi"),
    )

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"saved:{path.suffix}:{kwargs.get('format', '')}",
            encoding="utf-8",
        )
        saved_files.append(path)
        saved_backend[path.name] = kwargs.get("backend")
        if self.axes:
            saved_box_scales[path.name] = _box_scale(self.axes[0])
            saved_axes_bounds[path.name] = _axes_bounds(self.axes[0])
            saved_dpi[path.name] = float(kwargs["dpi"])
            if path.name.startswith("waterfall."):
                saved_geometry[path.name] = len(self.axes[0].lines)
            else:
                saved_geometry[path.name] = _surface_face_count(self.axes[0])
            saved_axis_state[path.name] = {
                "xlabel": self.axes[0].get_xlabel(),
                "ylabel": self.axes[0].get_ylabel(),
                "zlabel": self.axes[0].get_zlabel(),
                "zlabel_position": self.axes[0].zaxis.get_label_position(),
                "zticks_position": self.axes[0].zaxis.get_ticks_position(),
                "ztext_x": _zaxis_text_center_x(self.axes[0]),
                "axes_center_x": _axes_center_x(self.axes[0]),
                "xlim": tuple(float(v) for v in self.axes[0].get_xlim()),
                "xticks": tuple(self.axes[0].get_xticks()),
                "yticks": tuple(self.axes[0].get_yticks()),
                "zticks": tuple(self.axes[0].get_zticks()),
                "xticklabels": _tick_texts(self.axes[0], "x"),
                "yticklabels": _tick_texts(self.axes[0], "y"),
                "zticklabels": _tick_texts(self.axes[0], "z"),
            }
            saved_axis_padding[path.name] = {
                "labelpad": (
                    self.axes[0].xaxis.labelpad,
                    self.axes[0].yaxis.labelpad,
                    self.axes[0].zaxis.labelpad,
                ),
                "ticklabelpad": _tick_base_pads(self.axes[0]),
            }
            saved_rcparams[path.name] = {
                "axes.unicode_minus": matplotlib.rcParams["axes.unicode_minus"],
                "font.family": tuple(matplotlib.rcParams["font.family"]),
                "font.sans-serif": tuple(
                    matplotlib.rcParams["font.sans-serif"][:3]
                ),
                "font.size": float(matplotlib.rcParams["font.size"]),
                "axes.labelsize": float(matplotlib.rcParams["axes.labelsize"]),
                "xtick.labelsize": float(matplotlib.rcParams["xtick.labelsize"]),
                "ytick.labelsize": float(matplotlib.rcParams["ytick.labelsize"]),
                "legend.fontsize": float(matplotlib.rcParams["legend.fontsize"]),
                "mathtext.fontset": matplotlib.rcParams["mathtext.fontset"],
                "pgf.rcfonts": matplotlib.rcParams["pgf.rcfonts"],
                "pgf.preamble": matplotlib.rcParams["pgf.preamble"],
            }

    compiled: dict[str, Path | str] = {}

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        compiled["preview_tex"] = tex_path
        compiled["latex_command"] = latex_command
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex._display_pdf_preview",
        lambda pdf_path, *, height=720, png_dpi=200.0: displayed.update(
            {
                "pdf_path": Path(pdf_path),
                "height": int(height),
                "png_dpi": float(png_dpi),
            }
        ),
    )
    monkeypatch.setitem(matplotlib.rcParams, "axes.unicode_minus", True)
    monkeypatch.setitem(matplotlib.rcParams, "mathtext.fontset", "dejavusans")
    monkeypatch.setitem(matplotlib.rcParams, "font.family", ["serif"])
    monkeypatch.setitem(
        matplotlib.rcParams,
        "font.sans-serif",
        ["Not Arial", "Not Helvetica", "Not DejaVu Sans"],
    )
    monkeypatch.setitem(matplotlib.rcParams, "font.size", 11.0)
    monkeypatch.setitem(matplotlib.rcParams, "axes.labelsize", 10.0)
    monkeypatch.setitem(matplotlib.rcParams, "xtick.labelsize", 9.0)
    monkeypatch.setitem(matplotlib.rcParams, "ytick.labelsize", 9.0)

    stale_local_dir = tmp_path / "local" / "figures" / "demo_stack"
    stale_local_dir.mkdir(parents=True, exist_ok=True)
    stale_local_file = stale_local_dir / "stale.txt"
    stale_local_file.write_text("stale", encoding="utf-8")
    stale_remote_dir = tmp_path / "remote" / "figures" / "demo_stack"
    stale_remote_dir.mkdir(parents=True, exist_ok=True)
    stale_remote_file = stale_remote_dir / "stale.txt"
    stale_remote_file.write_text("stale", encoding="utf-8")
    stale_remote_pdf = (tmp_path / "remote" / "figures" / "demo_stack").with_suffix(
        ".pdf"
    )
    stale_remote_pdf.write_text("stale", encoding="utf-8")

    export = export_stacked_waterfall_thesis(
        x,
        y,
        z,
        name="demo_stack",
        sub_dir="figures",
        local_dir=tmp_path / "local",
        remote_dir=tmp_path / "remote",
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
        trace_step=2,
        x_axis=(False, False, True),
        y_axis=(False, False, True),
        z_axis=(True, True, False),
        z_axis_side="right",
        invert_xaxis=True,
        posx=(0.12, 0.12, 0.12),
        labelspacing=(0.0, 0.5, 1.0),
        ticklabelspacing=(0.0, 0.5, 1.0),
        ticks=custom_ticks,
        ticklabels=custom_ticklabels,
        heatmap_mode="vector_cells",
        box_aspect=(1.0, 2.0, 0.8),
        box_zoom=0.9,
        waterfall_box_zoom=0.45,
        heatmap_box_zoom=0.55,
        axes_rect=(0.6, 0.35, 4.8, 2.8),
        figsize=(2.1, 4.2),
    )

    expected_saved = {
        export.waterfall_pdf,
        export.surface_pdf,
        export.heatmap_pdf,
        export.waterfall_pgf,
        export.surface_pgf,
        export.heatmap_pgf,
    }
    assert expected_saved.issubset(set(saved_files))
    assert saved_dpi[export.waterfall_pdf.name] == 1800.0
    assert saved_dpi[export.surface_pdf.name] == 3600.0
    assert saved_dpi[export.heatmap_pdf.name] == 1800.0
    assert saved_dpi[export.waterfall_pgf.name] == 1800.0
    assert saved_dpi[export.surface_pgf.name] == 3600.0
    assert saved_dpi[export.heatmap_pgf.name] == 1800.0
    assert saved_backend[export.waterfall_pdf.name] == "pgf"
    assert saved_backend[export.surface_pdf.name] == "pgf"
    assert saved_backend[export.heatmap_pdf.name] == "pgf"
    assert saved_backend[export.waterfall_pgf.name] == "pgf"
    assert saved_backend[export.surface_pgf.name] == "pgf"
    assert saved_backend[export.heatmap_pgf.name] == "pgf"
    assert saved_geometry[export.waterfall_pdf.name] == 4
    assert saved_geometry[export.surface_pdf.name] == 10 * (
        (z.shape[1] - 1) * (z.shape[0] - 1)
    )
    assert saved_geometry[export.heatmap_pdf.name] == z.size
    assert export.stack_dir == tmp_path / "local" / "figures" / "demo_stack"
    assert export.main_pgf.exists()
    assert export.main_tex.exists()
    assert export.main_pdf.exists()
    assert export.remote_stack_dir == (
        tmp_path / "remote" / "figures" / "demo_stack"
    )
    assert export.remote_stack_dir.exists()
    assert not stale_local_file.exists()
    assert not stale_remote_file.exists()
    assert not stale_remote_pdf.exists()
    assert not (tmp_path / "remote" / "figures" / "demo_stack.pdf").exists()
    stack_pgf = export.main_pgf.read_text(encoding="utf-8")
    assert {
        path.name for path in export.remote_stack_dir.iterdir()
    } == {
        "main.pgf",
        "waterfall.pgf",
        "surface.pgf",
        "heatmap.pgf",
    }
    assert (
        export.remote_stack_dir / "main.pgf"
    ).read_text(encoding="utf-8") == stack_pgf

    assert r"\begin{pgfpicture}" in stack_pgf
    assert r"\begin{minipage}[t][4.2in][t]{2.1in}" in stack_pgf
    assert r"\def\thesispanelwidth{6.2in}" in stack_pgf
    assert r"\def\thesispanelheight{3.9in}" in stack_pgf
    assert r"\def\thesisposxwaterfall{0.12in}" in stack_pgf
    assert r"\def\thesisposxsurface{0.12in}" in stack_pgf
    assert r"\def\thesisposxheatmap{0.12in}" in stack_pgf
    assert r"\def\thesisposywaterfall{2.8in}" in stack_pgf
    assert r"\def\thesisposysurface{1.4in}" in stack_pgf
    assert r"\def\thesisposyheatmap{0in}" in stack_pgf
    assert (
        r"\pgftext[left,bottom,at={\pgfpoint{\thesisposxheatmap}"
        r"{\thesisposyheatmap}}]{\input{heatmap.pgf}}"
    ) in stack_pgf
    heatmap_index = stack_pgf.index(r"\input{heatmap.pgf}")
    surface_index = stack_pgf.index(r"\input{surface.pgf}")
    waterfall_index = stack_pgf.index(r"\input{waterfall.pgf}")
    assert heatmap_index < surface_index < waterfall_index

    preview = export.main_tex.read_text(encoding="utf-8")
    assert r"\documentclass[varwidth,border=0pt]{standalone}" in preview
    assert r"\usepackage{iftex}" in preview
    assert r"\usepackage{xunicode}" in preview
    assert r"\usepackage{fontspec}" in preview
    assert r"\setmainfont{Arial}" in preview
    assert r"\setsansfont{Arial}" in preview
    assert r"\usepackage{cmbright}" in preview
    assert r"\usepackage{pgf}" in preview
    assert r"\usepackage{varwidth}" in preview
    assert "main.pgf" in preview
    assert r"\input{main.pgf}" in preview
    assert not (export.remote_stack_dir / "main.tex").exists()
    assert r"\usepackage{adjustbox}" not in preview
    assert compiled["preview_tex"] == export.main_tex
    assert compiled["latex_command"] == "xelatex"
    assert displayed["pdf_path"] == export.main_pdf
    assert displayed["height"] == 720
    assert displayed["png_dpi"] == 200.0
    np.testing.assert_allclose(
        saved_box_scales[export.waterfall_pdf.name]
        / saved_box_scales[export.surface_pdf.name],
        0.45 / 0.9,
    )
    assert saved_axis_state[export.waterfall_pdf.name]["xlim"][0] > (
        saved_axis_state[export.waterfall_pdf.name]["xlim"][1]
    )
    assert saved_axis_state[export.waterfall_pdf.name]["xlabel"] == ""
    assert saved_axis_state[export.waterfall_pdf.name]["ylabel"] == ""
    assert saved_axis_state[export.waterfall_pdf.name]["zlabel"] == "Counts"
    assert (
        saved_axis_state[export.waterfall_pdf.name]["ztext_x"]
        > saved_axis_state[export.waterfall_pdf.name]["axes_center_x"]
    )
    assert len(saved_axis_state[export.waterfall_pdf.name]["xticks"]) > 0
    assert len(saved_axis_state[export.waterfall_pdf.name]["yticks"]) > 0
    assert len(saved_axis_state[export.waterfall_pdf.name]["zticks"]) > 0
    assert saved_axis_state[export.surface_pdf.name]["xlabel"] == ""
    assert saved_axis_state[export.surface_pdf.name]["ylabel"] == ""
    assert saved_axis_state[export.surface_pdf.name]["zlabel"] == "Counts"
    assert (
        saved_axis_state[export.surface_pdf.name]["ztext_x"]
        > saved_axis_state[export.surface_pdf.name]["axes_center_x"]
    )
    assert len(saved_axis_state[export.surface_pdf.name]["xticks"]) > 0
    assert len(saved_axis_state[export.surface_pdf.name]["yticks"]) > 0
    assert len(saved_axis_state[export.surface_pdf.name]["zticks"]) > 0
    assert saved_axis_state[export.surface_pdf.name]["zticklabels"] == (
        "z lo",
        "z hi",
    )
    assert saved_axis_state[export.heatmap_pdf.name]["xlabel"] != ""
    assert saved_axis_state[export.heatmap_pdf.name]["ylabel"] != ""
    assert saved_axis_state[export.heatmap_pdf.name]["zlabel"] == ""
    np.testing.assert_allclose(
        saved_axis_state[export.heatmap_pdf.name]["xticks"],
        custom_ticks[0],
    )
    np.testing.assert_allclose(
        saved_axis_state[export.heatmap_pdf.name]["yticks"],
        custom_ticks[1],
    )
    assert len(saved_axis_state[export.heatmap_pdf.name]["xticks"]) > 0
    assert len(saved_axis_state[export.heatmap_pdf.name]["yticks"]) > 0
    assert len(saved_axis_state[export.heatmap_pdf.name]["zticks"]) == 0
    assert saved_axis_state[export.heatmap_pdf.name]["xticklabels"] == (
        "x lo",
        "x hi",
    )
    assert saved_axis_state[export.heatmap_pdf.name]["yticklabels"] == (
        "y lo",
        "y hi",
    )
    np.testing.assert_allclose(
        saved_axis_padding[export.waterfall_pdf.name]["labelpad"],
        (-21.0, -10.5, 0.0),
    )
    np.testing.assert_allclose(
        saved_axis_padding[export.surface_pdf.name]["ticklabelpad"],
        (-8.0, -4.0, 0.0),
    )
    np.testing.assert_allclose(
        saved_axes_bounds[export.waterfall_pdf.name],
        (0.6 / 6.2, 0.35 / 3.9, 4.8 / 6.2, 2.8 / 3.9),
    )
    assert (
        saved_rcparams[export.waterfall_pdf.name]["axes.unicode_minus"] is False
    )
    assert saved_rcparams[export.waterfall_pdf.name]["font.family"] == (
        "sans-serif",
    )
    assert saved_rcparams[export.waterfall_pdf.name]["font.sans-serif"] == (
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    )
    assert saved_rcparams[export.waterfall_pdf.name]["font.size"] == 8.0
    assert saved_rcparams[export.waterfall_pdf.name]["axes.labelsize"] == 7.0
    assert saved_rcparams[export.waterfall_pdf.name]["xtick.labelsize"] == 7.0
    assert saved_rcparams[export.waterfall_pdf.name]["ytick.labelsize"] == 7.0
    assert saved_rcparams[export.waterfall_pdf.name]["legend.fontsize"] == 7.0
    assert saved_rcparams[export.waterfall_pdf.name]["pgf.rcfonts"] is False
    assert (
        r"\usepackage{fontspec}"
        in saved_rcparams[export.waterfall_pdf.name]["pgf.preamble"]
    )
    assert (
        r"\usepackage{cmbright}"
        in saved_rcparams[export.waterfall_pdf.name]["pgf.preamble"]
    )
    assert (
        saved_rcparams[export.waterfall_pdf.name]["mathtext.fontset"]
        == "dejavusans"
    )
    assert matplotlib.rcParams["axes.unicode_minus"] is True
    assert matplotlib.rcParams["mathtext.fontset"] == "dejavusans"
    assert tuple(matplotlib.rcParams["font.family"]) == ("serif",)
    assert tuple(matplotlib.rcParams["font.sans-serif"][:3]) == (
        "Not Arial",
        "Not Helvetica",
        "Not DejaVu Sans",
    )
    assert matplotlib.rcParams["font.size"] == 11.0
    assert matplotlib.rcParams["axes.labelsize"] == 10.0
    assert matplotlib.rcParams["xtick.labelsize"] == 9.0
    assert matplotlib.rcParams["ytick.labelsize"] == 9.0


def test_build_stack_pgf_allows_negative_positions_with_fixed_canvas(
    tmp_path: Path,
) -> None:
    """Fixed stack canvases should preserve negative absolute positions."""
    main_pgf = tmp_path / "stack" / "main.pgf"
    stack_pgf = build_stack_pgf(
        waterfall_pgf=main_pgf.parent / "waterfall.pgf",
        surface_pgf=main_pgf.parent / "surface.pgf",
        heatmap_pgf=main_pgf.parent / "heatmap.pgf",
        main_pgf=main_pgf,
        subfigsize=(1.0, 1.0),
        posx=(0.0, 0.1, 0.2),
        posy=(1.0, 0.4, -0.2),
        figsize=(2.0, 2.0),
    )

    assert r"\def\thesisposxheatmap{0.2in}" in stack_pgf
    assert r"\def\thesisposyheatmap{-0.2in}" in stack_pgf
    assert r"\def\thesisstackwidth{2.0in}" in stack_pgf
    assert r"\def\thesisstackheight{2.0in}" in stack_pgf


def test_build_stack_pgf_reanchors_negative_positions_without_canvas(
    tmp_path: Path,
) -> None:
    """Natural stack canvases should re-anchor negative panel positions."""
    main_pgf = tmp_path / "stack" / "main.pgf"
    stack_pgf = build_stack_pgf(
        waterfall_pgf=main_pgf.parent / "waterfall.pgf",
        surface_pgf=main_pgf.parent / "surface.pgf",
        heatmap_pgf=main_pgf.parent / "heatmap.pgf",
        main_pgf=main_pgf,
        subfigsize=(1.0, 1.0),
        posx=(0.0, -0.2, 0.1),
        posy=(1.0, 0.4, -0.2),
    )

    assert r"\def\thesisposxwaterfall{0.2in}" in stack_pgf
    assert r"\def\thesisposxsurface{0in}" in stack_pgf
    assert r"\def\thesisposxheatmap{0.3in}" in stack_pgf
    assert r"\def\thesisposywaterfall{1.2in}" in stack_pgf
    assert r"\def\thesisposysurface{0.6in}" in stack_pgf
    assert r"\def\thesisposyheatmap{0in}" in stack_pgf
    assert r"\def\thesisstackwidth{1.3in}" in stack_pgf
    assert r"\def\thesisstackheight{2.2in}" in stack_pgf


def test_export_stacked_waterfall_thesis_non_subfigure_fonts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-subfigure export should use the thesis mplstyle font sizes."""
    x, y, z = _waterfall_data()
    saved_rcparams: dict[str, dict[str, float]] = {}

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("saved", encoding="utf-8")
        saved_rcparams[path.name] = {
            "font.size": float(matplotlib.rcParams["font.size"]),
            "axes.labelsize": float(matplotlib.rcParams["axes.labelsize"]),
            "xtick.labelsize": float(matplotlib.rcParams["xtick.labelsize"]),
            "ytick.labelsize": float(matplotlib.rcParams["ytick.labelsize"]),
            "legend.fontsize": float(matplotlib.rcParams["legend.fontsize"]),
        }

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )

    export = export_stacked_waterfall_thesis(
        x,
        y,
        z,
        name="demo_stack",
        local_dir=tmp_path,
        remote_dir=None,
        subfigure=False,
    )

    assert saved_rcparams[export.waterfall_pdf.name]["font.size"] == 9.0
    assert saved_rcparams[export.waterfall_pdf.name]["axes.labelsize"] == 8.0
    assert saved_rcparams[export.waterfall_pdf.name]["xtick.labelsize"] == 8.0
    assert saved_rcparams[export.waterfall_pdf.name]["ytick.labelsize"] == 8.0
    assert saved_rcparams[export.waterfall_pdf.name]["legend.fontsize"] == 8.0


def test_export_stacked_waterfall_thesis_can_skip_inline_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stack exporter can suppress inline preview display explicitly."""
    x, y, z = _waterfall_data()
    preview_calls: list[Path] = []

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("saved", encoding="utf-8")

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex._display_pdf_preview",
        lambda pdf_path, *, height=720, png_dpi=200.0: preview_calls.append(
            Path(pdf_path)
        ),
    )

    export_stacked_waterfall_thesis(
        x,
        y,
        z,
        name="demo_stack",
        local_dir=tmp_path,
        remote_dir=None,
        show_preview=False,
    )

    assert preview_calls == []


def test_export_stacked_waterfall_thesis_allows_surface_x_oversampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stack exporter should forward surface x oversampling."""
    x, y, z = _waterfall_data(nx=9, ny=5)
    saved_surface_faces: dict[str, int] = {}

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("saved", encoding="utf-8")
        if path.name.startswith("surface.") and self.axes:
            saved_surface_faces[path.name] = _surface_face_count(self.axes[0])

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )

    export_stacked_waterfall_thesis(
        x,
        y,
        z,
        name="demo_stack",
        local_dir=tmp_path,
        remote_dir=None,
        surface_x_oversample=3,
    )

    expected_faces = 3 * (len(x) - 1) * (len(y) - 1)
    assert saved_surface_faces["surface.pdf"] == expected_faces
    assert saved_surface_faces["surface.pgf"] == expected_faces


def test_export_stacked_waterfall_thesis_allows_surface_pdf_dpi_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The surface panel should accept a dedicated save DPI override."""
    x, y, z = _waterfall_data()
    saved_dpi: dict[str, float] = {}

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("saved", encoding="utf-8")
        saved_dpi[path.name] = float(kwargs["dpi"])

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )

    export_stacked_waterfall_thesis(
        x,
        y,
        z,
        name="demo_stack",
        local_dir=tmp_path,
        remote_dir=None,
        pdf_dpi=1200.0,
        surface_pdf_dpi=2400.0,
    )

    assert saved_dpi["waterfall.pdf"] == 1200.0
    assert saved_dpi["surface.pdf"] == 2400.0
    assert saved_dpi["heatmap.pdf"] == 1200.0
    assert saved_dpi["waterfall.pgf"] == 1200.0
    assert saved_dpi["surface.pgf"] == 2400.0
    assert saved_dpi["heatmap.pgf"] == 1200.0


def test_export_stacked_waterfall_thesis_waterfall_traces_accepts_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integer waterfall_traces should keep the old trace-count behavior."""
    x, y, z = _waterfall_data()
    saved_line_counts: dict[str, int] = {}

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("saved", encoding="utf-8")
        if self.axes and path.name.startswith("waterfall."):
            saved_line_counts[path.name] = len(self.axes[0].lines)

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )

    export_stacked_waterfall_thesis(
        x,
        y,
        z,
        name="demo_stack",
        local_dir=tmp_path,
        remote_dir=None,
        waterfall_traces=3,
    )

    assert saved_line_counts["waterfall.pdf"] == 3
    assert saved_line_counts["waterfall.pgf"] == 3


def test_export_stacked_waterfall_thesis_waterfall_traces_accepts_yvalues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit y-values should select the nearest waterfall traces."""
    x, y, z = _waterfall_data()
    requested_y = (
        float(y[1] + 7.0),
        float(y[4] - 11.0),
        float(y[6] - 3.0),
    )
    selected_y = (float(y[1]), float(y[4]), float(y[6]))
    saved_yvalues: dict[str, tuple[float, ...]] = {}

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("saved", encoding="utf-8")
        if self.axes and path.name.startswith("waterfall."):
            saved_yvalues[path.name] = tuple(
                float(np.unique(line.get_data_3d()[1])[0]) for line in self.axes[0].lines
            )

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )

    export_stacked_waterfall_thesis(
        x,
        y,
        z,
        name="demo_stack",
        local_dir=tmp_path,
        remote_dir=None,
        waterfall_traces=requested_y,
    )

    assert saved_yvalues["waterfall.pdf"] == selected_y
    assert saved_yvalues["waterfall.pgf"] == selected_y


def test_thesis_surface_trace_overlay_adds_vector_strip() -> None:
    """Surface trace overlays should add a separate non-rasterized strip."""
    x, y, z = _waterfall_data()

    fig_plain, ax_plain = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_x_oversample=1,
    )
    fig_overlay, ax_overlay = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        trace_y=np.asarray([y[2]], dtype=np.float64),
        trace_z=np.asarray(z[[2]], dtype=np.float64),
        surface_x_oversample=1,
        surface_rasterized=False,
    )

    try:
        plain_count = len(ax_plain.collections)
        overlay_count = len(ax_overlay.collections)
        overlay_strip = ax_overlay.collections[1]
    finally:
        plt.close(fig_plain)
        plt.close(fig_overlay)

    assert plain_count == 1
    assert overlay_count > plain_count
    assert not ax_overlay.collections[0].get_rasterized()
    assert overlay_strip.get_rasterized() is not True
    assert overlay_strip.get_zorder() > ax_overlay.collections[0].get_zorder()
    assert overlay_strip.get_facecolor().shape[0] == x.shape[0] - 1


def test_thesis_surface_trace_overlay_includes_first_trace() -> None:
    """The first selected trace should also create a surface strip."""
    x, y, z = _waterfall_data()

    fig_plain, ax_plain = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_x_oversample=1,
    )
    fig_overlay, ax_overlay = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        trace_y=np.asarray([y[0]], dtype=np.float64),
        trace_z=np.asarray(z[[0]], dtype=np.float64),
        surface_x_oversample=1,
        surface_rasterized=False,
    )

    try:
        plain_count = len(ax_plain.collections)
        overlay_count = len(ax_overlay.collections)
    finally:
        plt.close(fig_plain)
        plt.close(fig_overlay)

    assert plain_count == 1
    assert overlay_count > plain_count


def test_thesis_surface_trace_overlay_embeds_raster_band() -> None:
    """Rasterized surface traces should stay inside one surface collection."""
    x, y, z = _waterfall_data()

    fig_plain, ax_plain = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        surface_x_oversample=1,
    )
    fig_overlay, ax_overlay = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        trace_y=np.asarray([y[0]], dtype=np.float64),
        trace_z=np.asarray(z[[0]], dtype=np.float64),
        surface_x_oversample=1,
    )

    try:
        plain_count = len(ax_plain.collections)
        overlay_count = len(ax_overlay.collections)
        plain_faces = _surface_face_count(ax_plain)
        overlay_faces = _surface_face_count(ax_overlay)
    finally:
        plt.close(fig_plain)
        plt.close(fig_overlay)

    assert plain_count == 1
    assert overlay_count == 1
    assert ax_overlay.collections[0].get_rasterized()
    assert overlay_faces > plain_faces


def test_export_stacked_waterfall_thesis_waterfall_traces_with_xlim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waterfall selection should still export cleanly when x-limits crop."""
    x, y, z = _waterfall_data()
    saved_line_counts: dict[str, int] = {}

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("saved", encoding="utf-8")
        if self.axes and path.name.startswith("waterfall."):
            saved_line_counts[path.name] = len(self.axes[0].lines)

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )

    export_stacked_waterfall_thesis(
        x,
        y,
        z,
        name="demo_stack",
        local_dir=tmp_path,
        remote_dir=None,
        xlim=(950.0, 1150.0),
        waterfall_traces=3,
    )

    assert saved_line_counts["waterfall.pdf"] == 3


def test_compile_thesis_preview_reports_missing_latex(
    tmp_path: Path,
) -> None:
    """Missing LaTeX commands should raise a clear runtime error."""
    preview_tex = tmp_path / "preview.tex"
    preview_tex.write_text(
        r"\documentclass{standalone}\begin{document}\end{document}",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="was not found"):
        compile_thesis_preview(
            preview_tex,
            latex_command="definitely-not-a-real-latex-command",
        )


def test_compile_thesis_preview_cleans_auxiliary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful standalone compiles should remove auxiliary byproducts."""
    preview_tex = tmp_path / "preview.tex"
    preview_tex.write_text(
        r"\documentclass{standalone}\begin{document}\end{document}",
        encoding="utf-8",
    )

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        assert check is True
        assert capture_output is True
        assert text is True
        preview_tex.with_suffix(".pdf").write_text("compiled", encoding="utf-8")
        for suffix in (".aux", ".log", ".out", ".xdv", ".synctex.gz"):
            preview_tex.with_suffix(suffix).write_text("aux", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.subprocess.run",
        fake_run,
    )

    pdf_path = compile_thesis_preview(
        preview_tex,
        latex_command="xelatex",
    )

    assert pdf_path == preview_tex.with_suffix(".pdf")
    assert pdf_path.exists()
    assert preview_tex.exists()
    for suffix in (".aux", ".log", ".out", ".xdv", ".synctex.gz"):
        assert not preview_tex.with_suffix(suffix).exists()
