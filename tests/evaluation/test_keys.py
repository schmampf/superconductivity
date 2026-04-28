"""Tests for IV-data key parsing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation.traces.file import FileSpec
from superconductivity.evaluation.traces.keys import (
    KeysSpec,
    get_keys,
)
from superconductivity.utilities.meta.label import LabelSpec


def test_get_keys_forwards_remove_and_add_keys(
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
        "superconductivity.evaluation.traces.keys.list_specific_keys",
        fake_list_specific_keys,
    )

    out = get_keys(
        "dummy.h5",
        "frequency_at_15GHz",
        strip0="=",
        strip1="dBm",
        remove_key=["off", "no_irradiation"],
        add_key=("no_irradiation", 0.0),
    )

    assert out["specific_keys"] == ["no_irradiation", "nu=3dBm"]
    assert np.allclose(out["yvalues"], np.asarray([0.0, 3.0]))
    assert np.array_equal(out["indices"], np.asarray([0, 1]))
    assert out.keys() == ("y", "nu", "i", "indices", "skeys", "specific_keys")
    assert out.label == "nu (dBm)"
    assert out._spec.label is None


def test_get_keys_accepts_filespec_and_keysspec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The convenience API should accept the spec objects directly."""

    def fake_list_specific_keys(
        h5path: FileSpec,
        measurement: str | None = None,
    ) -> list[str]:
        assert isinstance(h5path, FileSpec)
        assert measurement == "frequency_at_15GHz"
        return ["nu=3dBm", "off", "no_irradiation"]

    monkeypatch.setattr(
        "superconductivity.evaluation.traces.keys.list_specific_keys",
        fake_list_specific_keys,
    )

    out = get_keys(
        filespec=FileSpec(
            h5path="dummy.h5",
            measurement="frequency_at_15GHz",
        ),
        keysspec=KeysSpec(
            strip0="=",
            strip1="dBm",
            remove_key=["off", "no_irradiation"],
            add_key=("no_irradiation", 0.0),
            label=LabelSpec(
                code_label="nu_GHz",
                print_label="nu_GHz",
                html_label="<i>nu</i> (dBm)",
                latex_label=r"$\nu$ (dBm)",
            ),
        ),
    )

    assert out["specific_keys"] == ["no_irradiation", "nu=3dBm"]
    assert np.allclose(out["yvalues"], np.asarray([0.0, 3.0]))
    assert out.keys() == (
        "y",
        "nu_GHz",
        "i",
        "indices",
        "skeys",
        "specific_keys",
    )
    assert out.label == "nu_GHz"
    assert out._spec.label.html_label == "<i>nu</i> (dBm)"


def test_get_keys_accepts_keysspec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HDF5 convenience wrapper should accept a bundled KeysSpec."""

    def fake_list_specific_keys(
        h5path: str,
        measurement: str,
    ) -> list[str]:
        assert h5path == "dummy.h5"
        assert measurement == "frequency_at_15GHz"
        return ["nu=3dBm", "off", "no_irradiation"]

    monkeypatch.setattr(
        "superconductivity.evaluation.traces.keys.list_specific_keys",
        fake_list_specific_keys,
    )

    out = get_keys(
        "dummy.h5",
        "frequency_at_15GHz",
        spec=KeysSpec(
            strip0="=",
            strip1="dBm",
            remove_key=["off", "no_irradiation"],
            add_key=("no_irradiation", 0.0),
            label=LabelSpec(
                code_label="nu",
                print_label="nu (dBm)",
                html_label="<i>nu</i> (dBm)",
                latex_label=r"$\nu$ (dBm)",
            ),
        ),
    )

    assert out["specific_keys"] == ["no_irradiation", "nu=3dBm"]
    assert np.allclose(out["yvalues"], np.asarray([0.0, 3.0]))
    assert out.label == "nu (dBm)"
    assert out._spec.label.html_label == "<i>nu</i> (dBm)"


def test_get_keys_accepts_add_key_as_tuple_of_tuples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tuple-of-tuples add_key input should be treated as multiple additions."""

    def fake_list_specific_keys(
        h5path: str,
        measurement: str,
    ) -> list[str]:
        assert h5path == "dummy.h5"
        assert measurement == "frequency_at_15GHz"
        return ["nu=3dBm"]

    monkeypatch.setattr(
        "superconductivity.evaluation.traces.keys.list_specific_keys",
        fake_list_specific_keys,
    )

    out = get_keys(
        "dummy.h5",
        "frequency_at_15GHz",
        strip0="=",
        strip1="dBm",
        add_key=(("no_irradiation", 0.0), ("no_irradiation", 0.005)),
    )

    assert out.specific_keys == ["no_irradiation", "no_irradiation", "nu=3dBm"]
    assert np.allclose(out.yvalues, np.asarray([0.0, 0.005, 3.0]))


def test_get_keys_rejects_invalid_norm() -> None:
    """KeysSpec.norm must be finite and positive."""
    with pytest.raises(ValueError, match="norm must be finite and > 0"):
        get_keys(
            filespec=FileSpec(
                h5path="dummy.h5",
                measurement="frequency_at_15GHz",
            ),
            spec=KeysSpec(
                strip0="=",
                strip1="dBm",
                norm=0.0,
            ),
        )


def test_get_keys_rejects_spec_plus_explicit_args() -> None:
    """KeysSpec must not be mixed with the legacy explicit arguments."""
    with pytest.raises(ValueError, match="either spec or individual"):
        get_keys(
            filespec=FileSpec(
                h5path="dummy.h5",
                measurement="frequency_at_15GHz",
            ),
            strip1="dBm",
            spec=KeysSpec(strip0="=", strip1="dBm"),
        )


def test_get_keys_falls_back_to_index_when_value_is_non_numeric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-numeric extracted values should fall back to positional indices."""

    def fake_list_specific_keys(
        h5path: str,
        measurement: str,
    ) -> list[str]:
        assert h5path == "dummy.h5"
        assert measurement == "frequency_at_15GHz"
        return ["mode=alpha", "mode=beta"]

    monkeypatch.setattr(
        "superconductivity.evaluation.traces.keys.list_specific_keys",
        fake_list_specific_keys,
    )

    out = get_keys(
        "dummy.h5",
        "frequency_at_15GHz",
        strip0="=",
    )

    assert out.specific_keys == ["mode=alpha", "mode=beta"]
    assert np.allclose(out.yvalues, np.asarray([0.0, 1.0]))
    assert out.y.code_label == "mode"


def test_get_keys_supports_strip0_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parsing should start at the beginning when strip0 is None."""

    def fake_list_specific_keys(
        h5path: str,
        measurement: str,
    ) -> list[str]:
        assert h5path == "dummy.h5"
        assert measurement == "frequency_at_15GHz"
        return ["0.5", "1.5", "2.5"]

    monkeypatch.setattr(
        "superconductivity.evaluation.traces.keys.list_specific_keys",
        fake_list_specific_keys,
    )

    out = get_keys(
        "dummy.h5",
        "frequency_at_15GHz",
        strip0=None,
    )

    assert out.specific_keys == ["0.5", "1.5", "2.5"]
    assert np.allclose(out.yvalues, np.asarray([0.5, 1.5, 2.5]))
    assert out.y.code_label == "y"
