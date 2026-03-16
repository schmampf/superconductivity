"""Tests for IV-data key parsing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation.keys import (
    list_specific_keys_and_values,
    sort_specific_keys_by_value,
)


def test_sort_specific_keys_by_value_supports_single_remove_key() -> None:
    """A single remove_key should filter exact matches before parsing."""
    keys, values = sort_specific_keys_by_value(
        ["nu=5dBm", "no_irradiation", "nu=-2dBm"],
        strip0="=",
        strip1="dBm",
        remove_key="no_irradiation",
    )

    assert keys == ["nu=-2dBm", "nu=5dBm"]
    assert np.allclose(values, np.asarray([-2.0, 5.0]))


def test_sort_specific_keys_by_value_accepts_multiple_remove_keys() -> None:
    """A remove_key list should filter multiple exact matches."""
    keys, values = sort_specific_keys_by_value(
        ["off", "nu=1dBm", "hold"],
        strip0="=",
        strip1="dBm",
        remove_key=["off", "hold"],
    )

    assert keys == ["nu=1dBm"]
    assert np.allclose(values, np.asarray([1.0]))


def test_sort_specific_keys_by_value_supports_single_add_key() -> None:
    """A single add_key tuple should be inserted before sorting."""
    keys, values = sort_specific_keys_by_value(
        ["nu=5dBm", "nu=-2dBm"],
        strip0="=",
        strip1="dBm",
        add_key=("no_irradiation", 0.0),
    )

    assert keys == ["nu=-2dBm", "no_irradiation", "nu=5dBm"]
    assert np.allclose(values, np.asarray([-2.0, 0.0, 5.0]))


def test_sort_specific_keys_by_value_accepts_multiple_added_keys() -> None:
    """A list of add_key tuples should be inserted and sorted."""
    keys, values = sort_specific_keys_by_value(
        ["nu=5dBm"],
        strip0="=",
        strip1="dBm",
        add_key=[("off", -1.0), ("no_irradiation", 0.0)],
    )

    assert keys == ["off", "no_irradiation", "nu=5dBm"]
    assert np.allclose(values, np.asarray([-1.0, 0.0, 5.0]))


def test_sort_specific_keys_by_value_rejects_nonfinite_added_value() -> None:
    """Added keys must have finite numeric values."""
    with pytest.raises(ValueError, match="must be finite"):
        sort_specific_keys_by_value(
            ["nu=5dBm"],
            strip0="=",
            strip1="dBm",
            add_key=("off", np.nan),
        )


def test_sort_specific_keys_by_value_rejects_empty_result() -> None:
    """Removing every key without additions should raise a clear error."""
    with pytest.raises(ValueError, match="remove_key and add_key"):
        sort_specific_keys_by_value(
            ["off"],
            remove_key="off",
        )


def test_list_specific_keys_and_values_forwards_remove_and_add_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HDF5 convenience wrapper should pass through key edits."""

    def fake_list_specific_keys(
        h5path: str,
        measurement: str,
    ) -> list[str]:
        assert h5path == "dummy.h5"
        assert measurement == "frequency_at_15GHz"
        return ["nu=3dBm", "off", "no_irradiation"]

    monkeypatch.setattr(
        "superconductivity.evaluation.keys.list_specific_keys",
        fake_list_specific_keys,
    )

    keys, values = list_specific_keys_and_values(
        "dummy.h5",
        "frequency_at_15GHz",
        strip0="=",
        strip1="dBm",
        remove_key=["off", "no_irradiation"],
        add_key=("no_irradiation", 0.0),
    )

    assert keys == ["no_irradiation", "nu=3dBm"]
    assert np.allclose(values, np.asarray([0.0, 3.0]))
