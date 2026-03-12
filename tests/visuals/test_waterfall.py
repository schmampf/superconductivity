"""Smoke tests for waterfall visualizations."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from superconductivity.visuals.waterfall import (
    get_waterfall_heatmap_matplotlib,
    get_waterfall_heatmap_plotly,
    get_waterfall_heatmap_stacked_matplotlib,
    get_waterfall_heatmap_stacked_plotly,
    get_waterfall_surface_heatmap_stacked_matplotlib,
    get_waterfall_matplotlib,
    get_waterfall_plotly,
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


def test_plotly_waterfall_smoke() -> None:
    """The Plotly waterfall helper should emit one segmented 3D line trace."""
    x, y, z = _waterfall_data()

    fig = get_waterfall_plotly(
        x,
        y,
        z,
        trace_step=2,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )

    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter3d"
    assert fig.data[0].mode == "lines"
    assert sum(value is None for value in fig.data[0].x) == 4
    assert fig.layout.scene.xaxis.title.text == "Emission Wavelength (nm)"
    assert fig.layout.scene.yaxis.title.text == "Excitation (arb.)"
    assert fig.layout.scene.zaxis.title.text == "Counts"


def test_matplotlib_waterfall_smoke() -> None:
    """The Matplotlib waterfall helper should create one line per trace."""
    x, y, z = _waterfall_data()

    fig, ax = get_waterfall_matplotlib(
        x,
        y,
        z,
        trace_step=3,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 3
    assert ax.get_xlabel() == "Emission Wavelength (nm)"
    assert ax.get_ylabel() == "Excitation (arb.)"
    assert ax.get_zlabel() == "Counts"


def test_plotly_waterfall_heatmap_smoke() -> None:
    """The combined Plotly helper should create separated aligned panels."""
    x, y, z = _waterfall_data()

    fig = get_waterfall_heatmap_plotly(
        x,
        y,
        z,
        trace_step=2,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
        surface=True,
    )

    assert len(fig.data) == 6
    assert fig.data[0].type == "surface"
    assert fig.data[1].type == "surface"
    assert fig.data[2].type == "scatter3d"
    assert fig.data[2].mode == "lines"
    assert fig.data[3].mode == "lines"
    assert fig.data[4].mode == "markers"
    assert fig.data[5].mode == "lines"
    assert fig.data[0].showscale is False
    assert fig.layout.scene.zaxis.title.text == "Counts"


def test_matplotlib_waterfall_heatmap_smoke() -> None:
    """The combined Matplotlib helper should add frames and connectors."""
    x, y, z = _waterfall_data()

    fig, ax = get_waterfall_heatmap_matplotlib(
        x,
        y,
        z,
        trace_step=3,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
        surface=True,
    )

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 9
    assert len(ax.collections) >= 2
    assert len(fig.axes) == 1
    assert ax.get_zlabel() == "Counts"


def test_plotly_waterfall_heatmap_stacked_smoke() -> None:
    """The stacked Plotly helper should create two aligned 3D scenes."""
    x, y, z = _waterfall_data()

    fig = get_waterfall_heatmap_stacked_plotly(
        x,
        y,
        z,
        trace_step=2,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
        surface=True,
    )

    assert len(fig.data) == 3
    assert fig.data[0].type == "surface"
    assert fig.data[1].type == "scatter3d"
    assert fig.data[2].type == "surface"
    assert fig.data[2].showscale is False
    assert fig.layout.scene.camera.eye == fig.layout.scene2.camera.eye
    assert fig.layout.scene2.xaxis.title.text == "Emission Wavelength (nm)"
    assert fig.layout.scene2.zaxis.visible is False


def test_matplotlib_waterfall_heatmap_stacked_smoke() -> None:
    """The stacked Matplotlib helper should create two 3D subplots."""
    x, y, z = _waterfall_data()

    fig, (ax_top, ax_bottom) = get_waterfall_heatmap_stacked_matplotlib(
        x,
        y,
        z,
        trace_step=3,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
        surface=True,
    )

    assert fig is ax_top.figure
    assert fig is ax_bottom.figure
    assert ax_top.name == "3d"
    assert ax_bottom.name == "3d"
    assert len(ax_top.lines) == 3
    assert len(ax_bottom.collections) >= 1
    assert len(fig.axes) == 2
    assert ax_top.get_xlim() == ax_bottom.get_xlim()
    assert ax_top.get_ylim() == ax_bottom.get_ylim()
    assert len(ax_bottom.get_zticks()) == 0
    assert ax_bottom.get_zlim()[0] == pytest.approx(0.0)


def test_matplotlib_waterfall_surface_heatmap_stacked_smoke() -> None:
    """Three stacked Matplotlib panels should share x/y and add connectors."""
    x, y, z = _waterfall_data()

    fig, (ax_top, ax_mid, ax_bottom) = (
        get_waterfall_surface_heatmap_stacked_matplotlib(
            x,
            y,
            z,
            trace_step=3,
            xlabel="Emission Wavelength (nm)",
            ylabel="Excitation (arb.)",
            zlabel="Counts",
        )
    )

    assert fig is ax_top.figure
    assert fig is ax_mid.figure
    assert fig is ax_bottom.figure
    assert ax_top.name == "3d"
    assert ax_mid.name == "3d"
    assert ax_bottom.name == "3d"
    assert len(ax_top.lines) == 3
    assert len(ax_mid.collections) >= 1
    assert len(ax_bottom.collections) >= 1
    assert ax_mid.get_shared_x_axes().joined(ax_top, ax_mid)
    assert ax_bottom.get_shared_x_axes().joined(ax_top, ax_bottom)
    assert ax_mid.get_shared_y_axes().joined(ax_top, ax_mid)
    assert ax_bottom.get_shared_y_axes().joined(ax_top, ax_bottom)
    assert ax_top.get_xlim() == ax_mid.get_xlim() == ax_bottom.get_xlim()
    assert ax_top.get_ylim() == ax_mid.get_ylim() == ax_bottom.get_ylim()
    assert ax_top.get_zorder() > ax_mid.get_zorder() > ax_bottom.get_zorder()
    top_box = ax_top.get_position()
    mid_box = ax_mid.get_position()
    bottom_box = ax_bottom.get_position()
    assert top_box.width > 0.5
    assert mid_box.width > top_box.width
    assert top_box.y0 < mid_box.y1
    assert mid_box.y0 < bottom_box.y1
    assert len(fig.artists) == 4
    assert len(ax_mid.get_zticks()) == 0
    assert len(ax_bottom.get_zticks()) == 0
    assert ax_bottom.get_zlim()[0] == pytest.approx(0.0)
    assert ax_top.patch.get_alpha() == pytest.approx(0.0)
    assert ax_mid.patch.get_alpha() == pytest.approx(0.0)
    assert ax_bottom.patch.get_alpha() == pytest.approx(0.0)
    assert not ax_bottom.xaxis.pane.get_visible()
    assert not ax_bottom.yaxis.pane.get_visible()
    assert not ax_bottom.zaxis.pane.get_visible()
