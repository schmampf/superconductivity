"""Smoke tests for print-oriented visualization helpers."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np

matplotlib.use("Agg")

from superconductivity.visuals.print import (
    save_axis,
    tri_normal,
    write_3D_print,
)


def test_print_package_exports() -> None:
    """The print package should expose the moved STL and SVG helpers."""
    assert callable(save_axis)
    assert callable(write_3D_print)
    normal = tri_normal(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    )
    assert np.allclose(normal, np.asarray([0.0, 0.0, 1.0]))
