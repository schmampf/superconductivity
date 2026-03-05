"""Shared visualization array and limit helpers.

Conventions
-----------
The scalar field uses the convention ``z[y_i, x_j]`` with shapes
``x: (Nx,)``, ``y: (Ny,)``, and ``z: (Ny, Nx)``.
"""

from typing import Optional, Tuple

import numpy as np

from superconductivity.utilities.types import LIM, NDArray, NDArray64


def normalize_lim(
    lim: LIM,
) -> tuple[Optional[float], Optional[float]]:
    """Normalize a (lo, hi) limit tuple.

    Parameters
    ----------
    lim
        Limit tuple ``(lo, hi)`` where each bound may be ``None``.

    Returns
    -------
    lo, hi
        Bounds with swapped order corrected (if both are not ``None``).
    """
    if lim is None:
        return None, None
    lo, hi = lim
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi


def check_xyz(
    x: NDArray64,  # (Nx,)
    y: NDArray64,  # (Ny,)
    z: NDArray64,  # (Ny, Nx)
) -> tuple[NDArray64, NDArray64, NDArray64]:
    """Validate and coerce x, y, z arrays.

    Parameters
    ----------
    x
        1D x-axis array of shape ``(Nx,)``.
    y
        1D y-axis array of shape ``(Ny,)``.
    z
        2D field array of shape ``(Ny, Nx)`` using ``z[y_i, x_j]``.

    Returns
    -------
    x_arr, y_arr, z_arr
        NumPy arrays with validated shapes.

    Raises
    ------
    ValueError
        If the input arrays do not have the expected dimensions or shapes.
    """
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 2:
        raise ValueError("Expected x,y 1D and z 2D.")
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"Expected z.shape == (len(y), len(x)) == ({y.size}, {x.size}), got {z.shape}."
        )
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    z_arr = np.asarray(z)

    return x_arr, y_arr, z_arr


def split_paren_suffix(s: str) -> Tuple[str, str]:
    """Split a label into a main part and a trailing parenthesized suffix.

    Only a trailing pattern of the form ``" ... ( ... )"`` is split. If present,
    this function returns the text before the space preceding ``("``, and the
    content inside parentheses prefixed by a single space.

    Parameters
    ----------
    s
        Input label string.

    Returns
    -------
    main
        Label without the trailing parenthesized suffix.
    suffix
        Suffix extracted from the parentheses, prefixed with a leading space, or an
        empty string if no suffix is present.
    """
    s = s.strip()
    if not s.endswith(")"):
        return s, ""

    lpar = s.rfind("(")
    if lpar == -1:
        return s, ""

    # Require a space right before '(' to match your style: " ... ( ... )"
    if lpar == 0 or s[lpar - 1] != " ":
        return s, ""

    main = s[: (lpar - 1)].rstrip()
    inside = s[(lpar + 1) : -1].strip()
    if inside != "":
        inside = " " + inside
    return main, inside


def get_clim(
    z: NDArray64,  # (Ny, Nx)
    zlim: LIM = None,
    clim: LIM = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Determine color limits (clim) for a 2D field.

    If ``clim`` is provided, it takes precedence. Otherwise, missing bounds are
    filled from ``zlim`` if provided, and remaining missing bounds are estimated
    from the data using ``mean(z) ± std(z)`` (NaN-safe).

    Parameters
    ----------
    z
        2D field array of shape ``(Ny, Nx)``.
    zlim
        Optional z-axis limits ``(lo, hi)``.
    clim
        Optional explicit color limits ``(lo, hi)``.

    Returns
    -------
    clim_min, clim_max
        Color limits suitable for Plotly ``cmin``/``cmax`` or ``zmin``/``zmax``.
    """
    z_mean = np.nanmean(z)
    z_std = np.nanstd(z)

    zini_min = z_mean - z_std
    zini_max = z_mean + z_std

    if zlim is not None:
        zlim_min, zlim_max = zlim
    else:
        zlim_min, zlim_max = None, None

    if clim is not None:
        clim_min, clim_max = clim
    else:
        clim_min, clim_max = None, None

    if clim_min is None:
        if zlim_min is None:
            clim_min = zini_min
        else:
            clim_min = zlim_min

    if clim_max is None:
        if zlim_max is None:
            clim_max = zini_max
        else:
            clim_max = zlim_max

    return clim_min, clim_max


def get_xylim_indices(
    x: NDArray64,  # (Nx,)
    y: NDArray64,  # (Ny,)
    xlim: LIM = None,
    ylim: LIM = None,
) -> tuple[NDArray[np.bool], NDArray[np.bool]]:
    """Compute boolean masks selecting x/y values inside given limits.

    Parameters
    ----------
    x
        1D x-axis array of shape ``(Nx,)``.
    y
        1D y-axis array of shape ``(Ny,)``.
    xlim
        Optional x limits ``(lo, hi)``. Bounds may be ``None``.
    ylim
        Optional y limits ``(lo, hi)``. Bounds may be ``None``.

    Returns
    -------
    ix, iy
        Boolean masks selecting the requested region.

    Raises
    ------
    ValueError
        If the masks would select no points.
    """
    xlo, xhi = normalize_lim(xlim)
    ylo, yhi = normalize_lim(ylim)

    ix = np.ones(x.shape[0], dtype=bool)
    iy = np.ones(y.shape[0], dtype=bool)

    if xlo is not None:
        ix &= x >= xlo
    if xhi is not None:
        ix &= x <= xhi

    if ylo is not None:
        iy &= y >= ylo
    if yhi is not None:
        iy &= y <= yhi

    # If the mask removes everything, fail loudly (better than returning empty arrays silently)
    if not np.any(ix):
        raise ValueError(f"xlim={xlim} leaves no x points.")
    if not np.any(iy):
        raise ValueError(f"ylim={ylim} leaves no y points.")

    return ix, iy


def get_compressed_indices(
    y: NDArray,  # (Ny,)
    z: NDArray,  # (Ny, Nx)
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    equal_nan: bool = True,
) -> NDArray[np.bool]:
    """
    Compute indices that compress consecutive duplicate rows in ``z``.

    This is intended for data where many neighboring ``y`` positions share the same
    (or numerically near-identical) row ``z[i, :]``. The function keeps the first
    row of each run and marks subsequent duplicates for removal.

    Parameters
    ----------
    y
        1D y-axis array of shape ``(Ny,)``.
    z
        2D field array of shape ``(Ny, Nx)``.
    rtol
        Relative tolerance passed to ``np.allclose``.
    atol
        Absolute tolerance passed to ``np.allclose``.
    equal_nan
        If True, treat NaNs at the same positions as equal.

    Returns
    -------
    keep
        Boolean mask of shape ``(Ny,)`` indicating rows to keep.
    keep_idx
        Integer indices of kept rows (equivalent to ``np.flatnonzero(keep)``).

    Raises
    ------
    ValueError
        If ``y`` is not 1D, ``z`` is not 2D, or the leading dimensions mismatch.
    """
    y = np.asarray(y)
    z = np.asarray(z)

    if y.ndim != 1 or z.ndim != 2 or z.shape[0] != y.size:
        raise ValueError("Expected y.shape == (Ny,) and z.shape == (Ny, Nx).")

    Ny = y.size
    if Ny == 0:
        return y, z, np.array([], dtype=int)
    if Ny == 1:
        return y, z, np.array([0], dtype=int)

    keep = np.ones(Ny, dtype=bool)

    # Compare each row to the previous row; drop if it is (almost) the same.
    for i in range(1, Ny):
        same = np.allclose(z[i], z[i - 1], rtol=rtol, atol=atol, equal_nan=equal_nan)
        if same:
            keep[i] = False

    keep_idx = np.flatnonzero(keep)

    return keep
    return keep
    return keep
