from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from superconductivity.style.thesislayout import (  # noqa: E402
    daumenkino_layout,
    get_figures,
)


def _right_margin(ax: matplotlib.axes.Axes) -> float:
    bounds = ax.get_position().bounds
    return 1.0 - (bounds[0] + bounds[2])


def _top_margin(ax: matplotlib.axes.Axes) -> float:
    bounds = ax.get_position().bounds
    return 1.0 - (bounds[1] + bounds[3])


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


def test_daumenkino_layout_keeps_legacy_two_value_padding() -> None:
    fig, axes = get_figures(
        nrows=7,
        ncols=4,
        figsize=(4.2, 6.0),
        padding=(0.38, 0.28),
    )

    try:
        daumenkino_layout(fig, axes)

        top_right = axes[3].get_position().bounds
        bottom_left = axes[-4].get_position().bounds

        assert np.isclose(_right_margin(axes[3]), 0.04 / 4.2)
        assert np.isclose(_top_margin(axes[0]), 0.04 / 6.0)
        assert np.isclose(bottom_left[0], (0.38 + 0.04) / 4.2)
        assert np.isclose(bottom_left[1], (0.28 + 0.04) / 6.0)
        assert np.isclose(
            top_right[0] + top_right[2],
            1.0 - 0.04 / 4.2,
        )
    finally:
        plt.close(fig)


def test_daumenkino_layout_accepts_four_sided_padding() -> None:
    fig, axes = get_figures(
        nrows=7,
        ncols=4,
        figsize=(4.2, 6.0),
        padding=(0.38, 0.28, 0.35, 0.05),
    )

    try:
        daumenkino_layout(fig, axes)

        first = axes[0].get_position().bounds
        right_neighbor = axes[1].get_position().bounds
        lower_neighbor = axes[4].get_position().bounds

        assert np.isclose(_right_margin(axes[3]), (0.35 + 0.04) / 4.2)
        assert np.isclose(_top_margin(axes[0]), (0.05 + 0.04) / 6.0)
        assert np.isclose(first[0] + first[2], right_neighbor[0])
        assert np.isclose(lower_neighbor[1] + lower_neighbor[3], first[1])
    finally:
        plt.close(fig)


def test_daumenkino_layout_rejects_invalid_padding_length() -> None:
    with pytest.raises(ValueError, match="2-tuple or 4-tuple"):
        get_figures(padding=(0.1, 0.2, 0.3))

    fig, axes = get_figures()
    try:
        with pytest.raises(ValueError, match="2-tuple or 4-tuple"):
            daumenkino_layout(fig, axes, padding=(0.1,))
    finally:
        plt.close(fig)
