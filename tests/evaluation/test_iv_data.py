"""Tests for IV-data key parsing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation.iv_data import (
    list_specific_keys_and_values,
    sort_specific_keys_by_value,
)


def test_sort_specific_keys_by_value_supports_exact_overrides() -> None:
    """Exact-key overrides should bypass string parsing."""
    keys, values = sort_specific_keys_by_value(
        ["nu=5dBm", "no_irradiation", "nu=-2dBm"],
        strip0="=",
        strip1="dBm",
        value_overrides={"no_irradiation": 0.0},
    )

    assert keys == ["nu=-2dBm", "no_irradiation", "nu=5dBm"]
    assert np.allclose(values, np.asarray([-2.0, 0.0, 5.0]))


def test_sort_specific_keys_by_value_accepts_tuple_overrides() -> None:
    """Tuple-sequence overrides should work like mappings."""
    keys, values = sort_specific_keys_by_value(
        ["off", "nu=1dBm"],
        strip0="=",
        strip1="dBm",
        value_overrides=[("off", -1.0)],
    )

    assert keys == ["off", "nu=1dBm"]
    assert np.allclose(values, np.asarray([-1.0, 1.0]))


def test_sort_specific_keys_by_value_rejects_nonfinite_override() -> None:
    """Override values must be finite."""
    with pytest.raises(ValueError, match="must be finite"):
        sort_specific_keys_by_value(
            ["off"],
            value_overrides={"off": np.nan},
        )


def test_list_specific_keys_and_values_forwards_value_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HDF5 convenience wrapper should pass through overrides."""

    def fake_list_specific_keys(
        h5path: str,
        measurement: str,
    ) -> list[str]:
        assert h5path == "dummy.h5"
        assert measurement == "frequency_at_15GHz"
        return ["nu=3dBm", "no_irradiation"]

    monkeypatch.setattr(
        "superconductivity.evaluation.iv_data.list_specific_keys",
        fake_list_specific_keys,
    )

    keys, values = list_specific_keys_and_values(
        "dummy.h5",
        "frequency_at_15GHz",
        strip0="=",
        strip1="dBm",
        value_overrides={"no_irradiation": 0.0},
    )

    assert keys == ["no_irradiation", "nu=3dBm"]
    assert np.allclose(values, np.asarray([0.0, 3.0]))
