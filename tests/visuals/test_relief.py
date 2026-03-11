"""Regression tests for visible relief extraction."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from superconductivity.visuals.plotly.maps import get_relief, get_surface_relief
from superconductivity.visuals.relief import (
    extract_visible_relief,
    extract_visible_relief_from_mesh,
    prepare_relief_mesh,
)


def _grid(n: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a square grid and its mesh coordinates."""
    x = np.linspace(-4.0, 4.0, n, dtype=np.float64)
    y = np.linspace(-4.0, 4.0, n, dtype=np.float64)
    return x, y, *np.meshgrid(x, y, indexing="xy")


def test_single_peak_relief_regression() -> None:
    """A single isolated peak should yield stable visible outline geometry."""
    x, y, xg, yg = _grid()
    z = np.exp(-((xg + 0.5) ** 2 + yg**2) / 2.0)

    relief = extract_visible_relief(
        x,
        y,
        z,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
    )

    assert len(relief.polylines) == 7
    assert len(relief.world_segments or []) == 63
    assert sum(poly.shape[0] for poly in relief.polylines) == 70
    assert all(poly.ndim == 2 and poly.shape[1] == 2 for poly in relief.polylines)

    (xmin, xmax), (ymin, ymax) = relief.screen_bounds
    assert xmin < xmax
    assert ymin < ymax
    assert np.isfinite([xmin, xmax, ymin, ymax]).all()


def test_two_peak_occlusion_reduces_visible_segments() -> None:
    """Overlapping peaks should hide part of the farther silhouette."""
    x, y, xg, yg = _grid()
    z_near = 0.95 * np.exp(-((xg + 1.0) ** 2 + yg**2) / 1.6)
    z_far = 0.8 * np.exp(-((xg - 1.2) ** 2 + yg**2) / 1.6)

    relief_near = extract_visible_relief(
        x,
        y,
        z_near,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
    )
    relief_far = extract_visible_relief(
        x,
        y,
        z_far,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
    )
    relief_combo = extract_visible_relief(
        x,
        y,
        z_near + z_far,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
    )

    assert len(relief_combo.polylines) == 11
    assert len(relief_combo.world_segments or []) == 69
    assert len(relief_combo.world_segments or []) < (
        len(relief_near.world_segments or [])
        + len(relief_far.world_segments or [])
    )


def test_three_peak_relief_regression() -> None:
    """Three peaks should produce a richer but stable outline set."""
    x, y, xg, yg = _grid()
    z = (
        0.85 * np.exp(-((xg + 2.0) ** 2 + (yg + 0.2) ** 2) / 1.4)
        + 0.65 * np.exp(-((xg + 0.2) ** 2 + (yg - 0.1) ** 2) / 1.1)
        + 1.05 * np.exp(-((xg - 1.8) ** 2 + (yg + 0.1) ** 2) / 1.5)
    )

    relief = extract_visible_relief(
        x,
        y,
        z,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
    )

    assert len(relief.polylines) == 13
    assert len(relief.world_segments or []) == 134
    assert sum(poly.shape[0] for poly in relief.polylines) == 147


def test_projection_modes_stay_finite_and_differ() -> None:
    """Perspective and orthographic projections should both work."""
    x, y, xg, yg = _grid()
    z = np.exp(-((xg + 0.5) ** 2 + yg**2) / 2.0)

    relief_persp = extract_visible_relief(
        x,
        y,
        z,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
        projection="perspective",
    )
    relief_ortho = extract_visible_relief(
        x,
        y,
        z,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
        projection="orthographic",
    )

    assert len(relief_persp.polylines) == 7
    assert len(relief_ortho.polylines) == 8

    persp_span = (
        relief_persp.screen_bounds[0][1] - relief_persp.screen_bounds[0][0]
    )
    ortho_span = (
        relief_ortho.screen_bounds[0][1] - relief_ortho.screen_bounds[0][0]
    )
    assert persp_span < ortho_span


@pytest.mark.parametrize(
    ("observer", "target", "expected_polylines", "expected_segments"),
    [
        ((0.0, -6.0, 4.0), (0.0, 0.0, 0.3), 1, 32),
        ((-8.0, -0.2, 0.45), (0.0, 0.0, 0.25), 5, 58),
        ((-7.5, -3.0, 1.0), (0.5, 0.5, 0.3), 9, 58),
    ],
)
def test_off_center_and_nearly_tangent_views(
    observer: tuple[float, float, float],
    target: tuple[float, float, float],
    expected_polylines: int,
    expected_segments: int,
) -> None:
    """Representative off-center viewpoints should remain stable."""
    x, y, xg, yg = _grid(n=9)
    z = np.exp(-(xg**2 + yg**2) / 2.5)

    relief = extract_visible_relief(
        x,
        y,
        z,
        observer=observer,
        target=target,
    )

    assert len(relief.polylines) == expected_polylines
    assert len(relief.world_segments or []) == expected_segments


def test_plotly_relief_smoke() -> None:
    """The Plotly helper should emit a pure line-art figure."""
    x, y, xg, yg = _grid()
    z = np.exp(-((xg + 0.5) ** 2 + yg**2) / 2.0)

    fig = get_relief(
        x,
        y,
        z,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
    )

    assert len(fig.data) == 7
    assert all(trace.type == "scatter" for trace in fig.data)
    assert all(trace.mode == "lines" for trace in fig.data)
    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False


def test_plotly_surface_relief_smoke() -> None:
    """The surface overlay helper should emit one surface and one line trace."""
    x, y, xg, yg = _grid()
    z = np.exp(-((xg + 0.5) ** 2 + yg**2) / 2.0)

    fig = get_surface_relief(
        x,
        y,
        z,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
    )

    assert len(fig.data) == 2
    assert fig.data[0].type == "surface"
    assert fig.data[1].type == "scatter3d"
    assert fig.data[1].mode == "lines"
    assert fig.layout.scene.camera.eye is not None


def test_prepared_mesh_matches_direct_extraction() -> None:
    """Prepared-mesh extraction should match the direct convenience path."""
    x, y, xg, yg = _grid()
    z = (
        0.85 * np.exp(-((xg + 2.0) ** 2 + (yg + 0.2) ** 2) / 1.4)
        + 0.65 * np.exp(-((xg + 0.2) ** 2 + (yg - 0.1) ** 2) / 1.1)
        + 1.05 * np.exp(-((xg - 1.8) ** 2 + (yg + 0.1) ** 2) / 1.5)
    )
    kwargs = dict(
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
    )

    direct = extract_visible_relief(x, y, z, **kwargs)
    mesh = prepare_relief_mesh(x, y, z)
    prepared = extract_visible_relief_from_mesh(mesh, **kwargs)

    assert len(prepared.polylines) == len(direct.polylines)
    assert len(prepared.world_segments or []) == len(direct.world_segments or [])
    prepared_bounds = np.asarray(prepared.screen_bounds, dtype=np.float64).ravel()
    direct_bounds = np.asarray(direct.screen_bounds, dtype=np.float64).ravel()
    assert prepared_bounds == pytest.approx(direct_bounds)


def test_progress_smoke() -> None:
    """Progress-enabled extraction should run without changing outputs."""
    x, y, xg, yg = _grid(n=9)
    z = np.exp(-((xg + 0.5) ** 2 + yg**2) / 2.0)

    direct = extract_visible_relief(
        x,
        y,
        z,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
        progress=True,
    )
    mesh = prepare_relief_mesh(x, y, z)
    prepared = extract_visible_relief_from_mesh(
        mesh,
        observer=(-10.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.25),
        progress=True,
    )

    assert len(direct.polylines) > 0
    assert len(prepared.polylines) == len(direct.polylines)
