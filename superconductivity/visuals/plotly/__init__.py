"""Compatibility facade for Plotly visualization helpers.

Prefer importing from:
- ``superconductivity.visuals.plotly.maps`` for ``x/y/z`` map APIs.
- ``superconductivity.visuals.plotly.tuples`` for tuple/list trace APIs.
"""

from .template import register_house_template

register_house_template()

from .html import build_html
from .maps import (
    get_all,
    get_axis,
    get_heatmap,
    get_plain,
    get_slider,
    get_surface,
    save_figure,
)
from .tuples import get_tuple_slider

__all__ = [
    "get_axis",
    "get_surface",
    "get_plain",
    "get_slider",
    "get_heatmap",
    "get_all",
    "save_figure",
    "build_html",
    "get_tuple_slider",
]
