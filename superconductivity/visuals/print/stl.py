"""STL export helpers.

This module provides utilities to convert a 2D height field ``z`` into a
watertight STL solid suitable for 3D printing. The height field is
interpreted as a regular grid and is scaled to a square footprint with a
configurable base thickness.

Conventions
-----------
The input field follows the convention ``z[y_i, x_j]`` with shape
``(Ny, Nx)``. The generated mesh uses a uniform grid in x/y spanning
``[0, cube]``.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def tri_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute a unit normal vector for a triangle.

    Parameters
    ----------
    v0, v1, v2
        Triangle vertices as 1D arrays of shape ``(3,)``.

    Returns
    -------
    n
        Unit normal vector of shape ``(3,)``. If the triangle is degenerate
        (zero area), returns ``(0, 0, 0)``.
    """
    n = np.cross(v1 - v0, v2 - v0)
    nn = np.linalg.norm(n)
    if nn == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return n / nn


def write_3D_print(
    z: np.ndarray,
    zlim: Optional[Tuple[float, float]] = None,
    zbase: float = 3.0,
    cube: float = 100.0,
    dataset: str | Path = "dataset",
    title: str = "test",
) -> None:
    """Write a watertight STL solid from a 2D height field.

    The function interprets ``z`` as a regular 2D height field on a uniform
    grid. It maps the x/y footprint to a square of side length ``cube`` (mm)
    and scales ``z`` into the height interval ``[zbase, cube]`` (mm). A flat
    bottom plane at ``z=0`` and closed side walls are added to produce a
    watertight solid.

    Parameters
    ----------
    z
        2D height field of shape ``(Ny, Nx)``.
    zlim
        Optional value limits ``(zmin, zmax)`` used for clipping and scaling.
        If None, uses ``(nanmin(z), nanmax(z))``.
    zbase
        Base thickness in mm. The bottom of the solid is at ``z=0`` and the
        top surface starts at ``z=zbase``.
    cube
        Cube side length in mm. The x/y footprint spans ``[0, cube]`` and the
        maximum top height is ``z=cube``.
    dataset
        Output directory. The STL file is written to ``<dataset>/stl``.
    title
        Output filename stem and STL solid name.

    Raises
    ------
    ValueError
        If ``z`` is not 2D, too small, ``zlim`` is invalid, or parameters are
        inconsistent (for example ``zbase >= cube``).

    Notes
    -----
    The STL is written in ASCII format. For large grids, ASCII files can
    become large; consider downsampling ``z`` before export if needed.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must be 2D (Ny, Nx).")
    ny, nx = z.shape
    if ny < 2 or nx < 2:
        raise ValueError("z must have at least shape (2, 2).")

    dataset = Path(dataset, "stl")
    dataset.mkdir(parents=True, exist_ok=True)

    if cube <= 0:
        raise ValueError("cube must be positive.")
    if zbase < 0:
        raise ValueError("zbase must be >= 0.")
    if zbase >= cube:
        raise ValueError("zbase must be < cube.")

    if zlim is None:
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
    else:
        zmin, zmax = map(float, zlim)
        if zmin > zmax:
            zmin, zmax = zmax, zmin

    if not np.isfinite(zmin) or not np.isfinite(zmax):
        raise ValueError("zlim (or z data) must be finite.")
    if zmax == zmin:
        raise ValueError("zlim range is zero; cannot scale.")

    # Uniform x/y grid mapped to [0, cube]
    x = np.linspace(0.0, cube, nx)
    y = np.linspace(0.0, cube, ny)
    xg, yg = np.meshgrid(x, y)

    # Clip and scale z into [zbase, cube]
    zc = np.clip(z, zmin, zmax)
    z_top = zbase + (zc - zmin) * (cube - zbase) / (zmax - zmin)
    z_bot = np.zeros_like(z_top)

    facets = []

    def add_tri(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> None:
        n = tri_normal(v0, v1, v2)
        facets.append((n, v0, v1, v2))

    for i in range(ny - 1):
        for j in range(nx - 1):
            v00 = np.array([xg[i, j], yg[i, j], z_top[i, j]])
            v01 = np.array([xg[i, j + 1], yg[i, j + 1], z_top[i, j + 1]])
            v10 = np.array([xg[i + 1, j], yg[i + 1, j], z_top[i + 1, j]])
            v11 = np.array(
                [xg[i + 1, j + 1], yg[i + 1, j + 1], z_top[i + 1, j + 1]]
            )

            add_tri(v00, v10, v11)
            add_tri(v00, v11, v01)

    for i in range(ny - 1):
        for j in range(nx - 1):
            v00 = np.array([xg[i, j], yg[i, j], z_bot[i, j]])
            v01 = np.array([xg[i, j + 1], yg[i, j + 1], z_bot[i, j + 1]])
            v10 = np.array([xg[i + 1, j], yg[i + 1, j], z_bot[i + 1, j]])
            v11 = np.array(
                [xg[i + 1, j + 1], yg[i + 1, j + 1], z_bot[i + 1, j + 1]]
            )

            add_tri(v00, v11, v10)
            add_tri(v00, v01, v11)

    i = 0
    for j in range(nx - 1):
        vt0 = np.array([xg[i, j], yg[i, j], z_top[i, j]])
        vt1 = np.array([xg[i, j + 1], yg[i, j + 1], z_top[i, j + 1]])
        vb0 = np.array([xg[i, j], yg[i, j], z_bot[i, j]])
        vb1 = np.array([xg[i, j + 1], yg[i, j + 1], z_bot[i, j + 1]])
        add_tri(vt0, vb1, vb0)
        add_tri(vt0, vt1, vb1)

    i = ny - 1
    for j in range(nx - 1):
        vt0 = np.array([xg[i, j], yg[i, j], z_top[i, j]])
        vt1 = np.array([xg[i, j + 1], yg[i, j + 1], z_top[i, j + 1]])
        vb0 = np.array([xg[i, j], yg[i, j], z_bot[i, j]])
        vb1 = np.array([xg[i, j + 1], yg[i, j + 1], z_bot[i, j + 1]])
        add_tri(vt0, vb0, vb1)
        add_tri(vt0, vb1, vt1)

    j = 0
    for i in range(ny - 1):
        vt0 = np.array([xg[i, j], yg[i, j], z_top[i, j]])
        vt1 = np.array([xg[i + 1, j], yg[i + 1, j], z_top[i + 1, j]])
        vb0 = np.array([xg[i, j], yg[i, j], z_bot[i, j]])
        vb1 = np.array([xg[i + 1, j], yg[i + 1, j], z_bot[i + 1, j]])
        add_tri(vt0, vb0, vb1)
        add_tri(vt0, vb1, vt1)

    j = nx - 1
    for i in range(ny - 1):
        vt0 = np.array([xg[i, j], yg[i, j], z_top[i, j]])
        vt1 = np.array([xg[i + 1, j], yg[i + 1, j], z_top[i + 1, j]])
        vb0 = np.array([xg[i, j], yg[i, j], z_bot[i, j]])
        vb1 = np.array([xg[i + 1, j], yg[i + 1, j], z_bot[i + 1, j]])
        add_tri(vt0, vb1, vb0)
        add_tri(vt0, vt1, vb1)

    with open(f"{dataset}/{title}.stl", "w", encoding="utf-8") as f:
        f.write(f"solid {title}\n")
        for n, v0, v1, v2 in facets:
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {title}\n")


__all__ = ["tri_normal", "write_3D_print"]
