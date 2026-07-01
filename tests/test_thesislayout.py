from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from superconductivity.style.thesislayout import (  # noqa: E402
    daumenkino_layout,
    get_figures,
)


def test_get_figures_returns_row_major_axes() -> None:
    fig, axes = get_figures(nrows=7, ncols=4, figsize=(4.2, 6.0))

    try:
        assert len(axes) == 28
        assert axes[0].get_subplotspec().rowspan.start == 0
        assert axes[0].get_subplotspec().colspan.start == 0
        assert axes[-1].get_subplotspec().rowspan.start == 6
        assert axes[-1].get_subplotspec().colspan.start == 3
    finally:
        plt.close(fig)


def test_daumenkino_layout_uses_outer_labels_and_touching_axes() -> None:
    fig, axes = get_figures(nrows=7, ncols=4, figsize=(4.2, 6.0))
    xticks = np.array([-1.0, 0.0, 1.0])
    yticks = np.array([0.0, 1.0])

    try:
        daumenkino_layout(
            fig,
            axes,
            xlabel="$V$",
            ylabel="$I$",
            xticks=xticks,
            yticks=yticks,
        )

        for index, ax in enumerate(axes):
            row = index // 4
            col = index % 4
            assert ax.get_xlabel() == ("$V$" if row == 6 else "")
            assert ax.get_ylabel() == ("$I$" if col == 0 else "")
            assert all(
                label.get_visible() == (row == 6)
                for label in ax.get_xticklabels()
            )
            assert all(
                label.get_visible() == (col == 0)
                for label in ax.get_yticklabels()
            )
            np.testing.assert_allclose(ax.get_xticks(), xticks)
            np.testing.assert_allclose(ax.get_yticks(), yticks)
            assert all(spine.get_visible() for spine in ax.spines.values())

        first = axes[0].get_position().bounds
        right_neighbor = axes[1].get_position().bounds
        lower_neighbor = axes[4].get_position().bounds

        assert np.isclose(first[0] + first[2], right_neighbor[0])
        assert np.isclose(lower_neighbor[1] + lower_neighbor[3], first[1])
    finally:
        plt.close(fig)
