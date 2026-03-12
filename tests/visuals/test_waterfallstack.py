"""Smoke tests for projected stacked waterfall visualizations."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib.collections import LineCollection, PolyCollection

from superconductivity.visuals.waterfallstack import (
    get_projected_waterfall_stack_matplotlib,
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


def test_projected_waterfall_stack_matplotlib_smoke() -> None:
    """The projected stack helper should render into one 2D Matplotlib axis."""
    x, y, z = _waterfall_data()

    fig, ax = get_projected_waterfall_stack_matplotlib(
        x,
        y,
        z,
        trace_step=2,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )

    assert fig is ax.figure
    assert ax.name == "rectilinear"
    assert not ax.axison
    assert len(ax.collections) >= 10
    assert sum(isinstance(c, PolyCollection) for c in ax.collections) == 2
    assert sum(isinstance(c, LineCollection) for c in ax.collections) >= 8
    labels = {text.get_text() for text in ax.texts}
    assert "Emission Wavelength (nm)" in labels
    assert "Excitation (arb.)" in labels
    assert "Counts" in labels


def test_projected_waterfall_stack_matplotlib_frame_mode() -> None:
    """Optional frame rendering should replace connectors cleanly."""
    x, y, z = _waterfall_data()

    fig, ax = get_projected_waterfall_stack_matplotlib(
        x,
        y,
        z,
        trace_step=3,
        show_frames=True,
        show_connectors=False,
    )

    assert fig is ax.figure
    assert len(ax.collections) >= 10
    assert sum(isinstance(c, PolyCollection) for c in ax.collections) == 2
    assert sum(isinstance(c, LineCollection) for c in ax.collections) >= 8
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert np.isfinite(xlim).all()
    assert np.isfinite(ylim).all()
