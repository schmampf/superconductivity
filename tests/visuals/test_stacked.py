"""Smoke tests for one-axis stacked 3D waterfall visualizations."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np

matplotlib.use("Agg")

from mpl_toolkits.mplot3d.art3d import Line3DCollection

from superconductivity.visuals.stacked import (
    get_waterfall_surface_heatmap_banded_matplotlib,
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


def test_banded_waterfall_surface_heatmap_matplotlib_smoke() -> None:
    """The one-axis stacked helper should create one 3D banded axis."""
    x, y, z = _waterfall_data()

    fig, ax = get_waterfall_surface_heatmap_banded_matplotlib(
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
    assert len(ax.get_zticks()) == 0
    assert len(ax.collections) >= 5
    assert sum(isinstance(c, Line3DCollection) for c in ax.collections) >= 3
    labels = {text.get_text() for text in ax.texts}
    assert "Counts" in labels
