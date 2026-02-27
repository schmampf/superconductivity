"""Luanti export helpers.

This module provides utilities to export 2D scalar fields (e.g. measurement
maps) into binary formats that are easy to consume from Luanti/Minetest mods.

Currently supported
-------------------
- Unsigned 16-bit little-endian raw heightmaps (row-major) plus a JSON metadata
  sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from PIL import Image

from superconductivity.style.cpd4 import cmap

ROOT = Path(__file__).resolve().parent


def export_dataset(
    z: np.ndarray,
    *,
    title: str = ".",
    name: str = "map",
    zlim: Optional[Tuple[float, float]] = None,
    colormap: Optional[str | ListedColormap] = cmap(),
    palette_name: str = "cmap",
    hmin: int = 0,
    hmax: int = 80,
    xlen: int = 200,
    ylen: int = 200,
) -> Path:
    """Export a 2D array as a Luanti-compatible raw heightmap.

    The heightmap is written as unsigned 16-bit little-endian values in
    row-major order (C order). A JSON sidecar is written containing the array
    shape and scaling metadata.

    Parameters
    ----------
    z
        2D array of shape (Ny, Nx) containing the scalar field.
    title
        Dataset title (folder name under `datasets/`).
    name
        Basename for output files (e.g. `name.u16le`, `name.json`).
    zlim
        Optional scaling range `(z_scale_min, z_scale_max)` used for
        normalization (i.e. mapping into the requested height range).
        Values outside this range are NOT clipped; instead they map to heights
        below/above `hmin`/`hmax`. If None, uses the 2nd and 98th percentiles.
    colormap
        Optional Matplotlib colormap name (e.g. "viridis") or a
        `matplotlib.colors.ListedColormap`. If provided, a 256×1 palette PNG is
        exported into the dataset folder and referenced in the JSON sidecar.
    palette_name
        Basename (without extension) for the exported palette PNG inside the
        dataset folder. Default is "cmap" -> "cmap.png".
    hmin, hmax
        Target height range in blocks.
    xlen
        Target grid width (number of columns) after remapping `z` onto a
        regular grid using nearest-neighbor sampling.
    ylen
        Target grid height (number of rows) after remapping `z` onto a
        regular grid using nearest-neighbor sampling.

    Returns
    -------
    raw_path
        Path to the written `.u16le` file.

    Raises
    ------
    ValueError
        If `zlim` is invalid.
    ImportError
        If `colormap` is given but matplotlib/pillow are unavailable.
    """
    out = ROOT / "datasets" / title
    out.mkdir(parents=True, exist_ok=True)

    if colormap is not None:
        if cm is None or Image is None:
            raise ImportError(
                "Palette export requires matplotlib and pillow. "
                "Install them or call export_dataset without cmap_name."
            )
        if isinstance(colormap, str):
            colormap = cm.get_cmap(colormap, 256)
        # (256,3)
        rgb = (colormap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb[np.newaxis, :, :], mode="RGB")  # 1x256

        palette_path = out / f"{palette_name}.png"
        img.save(palette_path)

    z = np.asarray(z, dtype=float)
    z = remap_to_regular_grid_nearest(z, size_x=xlen, size_y=ylen)

    # Scaling range used for normalization ("knob"). This range defines which
    # values map exactly to hmin/hmax, but we do NOT clip the data to it.
    if zlim is None:
        z_scale_min, z_scale_max = np.nanpercentile(z, [2, 98])
    else:
        z_scale_min, z_scale_max = float(zlim[0]), float(zlim[1])

    if (
        not np.isfinite(z_scale_min)
        or not np.isfinite(z_scale_max)
        or z_scale_min == z_scale_max
    ):
        raise ValueError("zlim must be finite and have different values")

    # Map full (unclipped) z values using the scaling range.
    t = (z - z_scale_min) / (z_scale_max - z_scale_min)
    h_f = t * (hmax - hmin) + hmin

    # Convert to uint16 safely: allow overshoots, but clamp to uint16 range.
    h_i = np.rint(h_f)
    h_i = np.clip(h_i, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    h = h_i

    raw_path = out / f"{name}.u16le"
    h.tofile(raw_path)

    meta = {
        "nx": int(h.shape[1]),
        "ny": int(h.shape[0]),
        "hmin": int(hmin),
        "hmax": int(hmax),
        # Scaling range used for z->height normalization
        "z_scale_min": float(z_scale_min),
        "z_scale_max": float(z_scale_max),
        # Backward-compatible aliases (historically named zmin/zmax)
        "zmin": float(z_scale_min),
        "zmax": float(z_scale_max),
        # Actual data range (unclipped)
        "z_data_min": float(np.nanmin(z)),
        "z_data_max": float(np.nanmax(z)),
        "palette": f"{palette_name}.png",
        "format": "u16le_rowmajor",
    }
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2))

    return raw_path


def remap_to_regular_grid_nearest(
    src: np.ndarray,
    size_x: int,
    size_y: int,
) -> np.ndarray:
    """
    Remap a 2D array to a regular (size_y, size_x) grid using nearest-neighbor
    sampling (closest value). Oversampling naturally fills gaps by repetition.

    Parameters
    ----------
    src
        Source array of shape (NY, NX).
    size_x
        Target grid width (number of columns).
    size_y
        Target grid height (number of rows).

    Returns
    -------
    out
        Remapped array of shape (size_y, size_x).
    """
    if src.ndim != 2:
        raise ValueError(f"src must be 2D, got shape={src.shape!r}")
    if size_x <= 0 or size_y <= 0:
        raise ValueError("size_x and size_y must be positive")

    ny, nx = src.shape

    # Map target pixel centers to source pixel centers
    sx = nx / size_x
    sy = ny / size_y

    xs = (np.arange(size_x) + 0.5) * sx - 0.5
    ys = (np.arange(size_y) + 0.5) * sy - 0.5

    ix = np.rint(xs).astype(np.int64)
    iy = np.rint(ys).astype(np.int64)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # Fancy indexing: broadcast iy (rows) against ix (cols)
    out = src[iy[:, None], ix[None, :]]
    return out
    return out
    return out
    return out
