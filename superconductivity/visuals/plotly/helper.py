"""Helpers specific to Plotly rendering."""

from typing import List, Tuple

from matplotlib.colors import ListedColormap

from superconductivity.style.cpd4 import cmap
from superconductivity.style.cpd5 import seeblau100
from superconductivity.utilities.types import COLOR, Number


def mpl_cmap_to_plotly(
    cmap_mpl: ListedColormap = cmap(),
    n: int = 256,
) -> List[Tuple[float, str]]:
    """Convert a Matplotlib colormap to a Plotly colorscale.

    Parameters
    ----------
    cmap_mpl
        Matplotlib colormap (e.g. ``ListedColormap``) callable with values in
        ``[0, 1]`` returning RGBA floats.
    n
        Number of samples used to discretize the colorscale.

    Returns
    -------
    colorscale
        List of ``(t, color)`` pairs where ``t`` is in ``[0, 1]`` and
        ``color`` is a Plotly color string (``rgba(r,g,b,a)``).
    """
    scale = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b, a = cmap_mpl(t)
        scale.append((t, f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.6g})"))
    return scale


def mpl_color_to_plotly(color: COLOR = seeblau100) -> str:
    """Convert an RGB/RGBA color tuple to a Plotly color string.

    Parameters
    ----------
    color
        RGB or RGBA tuple. Values may be floats in ``[0, 1]`` (Matplotlib
        style) or integers in ``[0, 255]``.

    Returns
    -------
    color_str
        Plotly color string ``rgb(r,g,b)`` or ``rgba(r,g,b,a)``.

    Raises
    ------
    ValueError
        If ``color`` does not have length 3 or 4, or contains invalid values.
    """
    vals = tuple(color)
    if len(vals) not in (3, 4):
        raise ValueError(
            f"Expected RGB or RGBA tuple, got length {len(vals)}.",
        )

    def to_255(v: Number) -> int:
        if isinstance(v, bool):
            raise ValueError("Boolean is not a valid color channel value.")
        v_float = float(v)
        if v_float <= 1.0:
            v_float *= 255.0
        return int(round(min(255.0, max(0.0, v_float))))

    r = to_255(vals[0])
    g = to_255(vals[1])
    b = to_255(vals[2])

    if len(vals) == 3:
        return f"rgb({r},{g},{b})"

    a_raw = float(vals[3])
    a = a_raw if a_raw <= 1.0 else a_raw / 255.0
    a = min(1.0, max(0.0, a))

    return f"rgba({r},{g},{b},{a:.6g})"
