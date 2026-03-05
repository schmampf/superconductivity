"""STL export helpers.

This module provides utilities to convert a 2D height field ``z`` into a
watertight STL solid suitable for 3D printing. The height field is interpreted
as a regular grid and is scaled to a square footprint with a configurable base
thickness.

Conventions
-----------
The input field follows the convention ``z[y_i, x_j]`` with shape ``(Ny, Nx)``.
The generated mesh uses a uniform grid in x/y spanning ``[0, cube]``.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def tri_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute a unit normal vector for a triangle.

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
    z: np.ndarray,  # (Ny, Nx)
    zlim: Optional[Tuple[float, float]] = None,
    zbase: float = 3.0,  # mm
    cube: float = 100.0,  # mm
    dataset: str | Path = "dataset",
    title: str = "test",
) -> None:
    """Write a watertight STL solid from a 2D height field.

    The function interprets ``z`` as a regular 2D height field on
    a uniform grid. It maps the x/y footprint to a square of side
    length ``cube`` (mm) and scales ``z`` into the height interval
    ``[zbase, cube]`` (mm). A flat bottom plane at ``z=0`` and
    closed side walls are added to produce a watertight solid.

    Parameters
    ----------
    z
        2D height field of shape ``(Ny, Nx)``.
    zlim
        Optional value limits ``(zmin, zmax)`` used for clipping and scaling.
        If None, uses ``(nanmin(z), nanmax(z))``.
    zbase
        Base thickness in mm. The bottom of the solid is at ``z=0`` and the top
        surface starts at ``z=zbase``.
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
        inconsistent (e.g. ``zbase >= cube``).

    Notes
    -----
    The STL is written in ASCII format. For large grids, ASCII files can become
    large; consider downsampling ``z`` before export if needed.
    """

    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must be 2D (Ny, Nx).")
    Ny, Nx = z.shape
    if Ny < 2 or Nx < 2:
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
    x = np.linspace(0.0, cube, Nx)
    y = np.linspace(0.0, cube, Ny)
    X, Y = np.meshgrid(x, y)  # (Ny, Nx)

    # Clip and scale z into [zbase, cube]
    zc = np.clip(z, zmin, zmax)
    Z_top = zbase + (zc - zmin) * (cube - zbase) / (zmax - zmin)
    Z_bot = np.zeros_like(Z_top)  # bottom plane at 0

    facets = []  # list of (n, v0, v1, v2)

    def add_tri(v0, v1, v2):
        n = tri_normal(v0, v1, v2)
        facets.append((n, v0, v1, v2))

    # --- Top surface (two triangles per cell)
    for i in range(Ny - 1):
        for j in range(Nx - 1):
            v00 = np.array([X[i, j], Y[i, j], Z_top[i, j]])
            v01 = np.array([X[i, j + 1], Y[i, j + 1], Z_top[i, j + 1]])
            v10 = np.array([X[i + 1, j], Y[i + 1, j], Z_top[i + 1, j]])
            v11 = np.array(
                [
                    X[i + 1, j + 1],
                    Y[i + 1, j + 1],
                    Z_top[i + 1, j + 1],
                ]
            )

            add_tri(v00, v10, v11)
            add_tri(v00, v11, v01)

    # --- Bottom surface (reverse winding)
    for i in range(Ny - 1):
        for j in range(Nx - 1):
            v00 = np.array([X[i, j], Y[i, j], Z_bot[i, j]])
            v01 = np.array([X[i, j + 1], Y[i, j + 1], Z_bot[i, j + 1]])
            v10 = np.array([X[i + 1, j], Y[i + 1, j], Z_bot[i + 1, j]])
            v11 = np.array(
                [
                    X[i + 1, j + 1],
                    Y[i + 1, j + 1],
                    Z_bot[i + 1, j + 1],
                ]
            )

            add_tri(v00, v11, v10)
            add_tri(v00, v01, v11)

    # --- Side walls (4 borders)
    # y = 0 edge (i=0)
    i = 0
    for j in range(Nx - 1):
        vt0 = np.array([X[i, j], Y[i, j], Z_top[i, j]])
        vt1 = np.array([X[i, j + 1], Y[i, j + 1], Z_top[i, j + 1]])
        vb0 = np.array([X[i, j], Y[i, j], Z_bot[i, j]])
        vb1 = np.array([X[i, j + 1], Y[i, j + 1], Z_bot[i, j + 1]])
        add_tri(vt0, vb1, vb0)
        add_tri(vt0, vt1, vb1)

    # y = cube edge (i=Ny-1)
    i = Ny - 1
    for j in range(Nx - 1):
        vt0 = np.array([X[i, j], Y[i, j], Z_top[i, j]])
        vt1 = np.array([X[i, j + 1], Y[i, j + 1], Z_top[i, j + 1]])
        vb0 = np.array([X[i, j], Y[i, j], Z_bot[i, j]])
        vb1 = np.array([X[i, j + 1], Y[i, j + 1], Z_bot[i, j + 1]])
        add_tri(vt0, vb0, vb1)
        add_tri(vt0, vb1, vt1)

    # x = 0 edge (j=0)
    j = 0
    for i in range(Ny - 1):
        vt0 = np.array([X[i, j], Y[i, j], Z_top[i, j]])
        vt1 = np.array([X[i + 1, j], Y[i + 1, j], Z_top[i + 1, j]])
        vb0 = np.array([X[i, j], Y[i, j], Z_bot[i, j]])
        vb1 = np.array([X[i + 1, j], Y[i + 1, j], Z_bot[i + 1, j]])
        add_tri(vt0, vb0, vb1)
        add_tri(vt0, vb1, vt1)

    # x = cube edge (j=Nx-1)
    j = Nx - 1
    for i in range(Ny - 1):
        vt0 = np.array([X[i, j], Y[i, j], Z_top[i, j]])
        vt1 = np.array([X[i + 1, j], Y[i + 1, j], Z_top[i + 1, j]])
        vb0 = np.array([X[i, j], Y[i, j], Z_bot[i, j]])
        vb1 = np.array([X[i + 1, j], Y[i + 1, j], Z_bot[i + 1, j]])
        add_tri(vt0, vb1, vb0)
        add_tri(vt0, vt1, vb1)

    # --- Write ASCII STL
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
        f.write(f"endsolid {title}\n")
