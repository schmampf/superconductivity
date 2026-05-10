from __future__ import annotations

from pathlib import Path

import numpy as np

from superconductivity.utilities.cache import (
    ProjectCache,
    cache_summary,
    entry_kind,
    list_caches,
    load_cache,
    make_cache,
    project_cache_path,
    save_cache,
)
from superconductivity.evaluation import SamplingSpec
from superconductivity.utilities.meta import axis, data
from superconductivity.utilities.transport import TransportDatasetSpec


def _make_transport_dataset() -> TransportDatasetSpec:
    return TransportDatasetSpec(
        data=data("I_nA", np.asarray([0.0, 1.0, 2.0], dtype=np.float64)),
        axes=axis(
            "V_mV",
            values=np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
            order=0,
        ),
    )


def _make_sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        Vbins_mV=np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
        Ibins_nA=np.asarray([-2.0, 0.0, 2.0], dtype=np.float64),
        cutoff_Hz=10.0,
        sampling_Hz=20.0,
    )


def test_project_cache_default_path_uses_repo_projects() -> None:
    cache = make_cache("demo")

    assert cache.path.name == "projects"
    assert cache.path.parent.name == "superconductivity"
    assert cache.file_path == cache.path / "demo.pkl"


def test_project_cache_default_path_is_single_pickle(tmp_path: Path) -> None:
    path = project_cache_path(tmp_path, "demo")

    assert path == tmp_path / "demo.pkl"


def test_project_cache_round_trips_arbitrary_items(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset()
    cache.offsetspec = _make_sampling_spec()
    cache.offsetanalysis = {"Voff_mV": np.asarray([0.1, 0.2])}

    path = cache.save_cache()
    loaded = load_cache("demo", path=tmp_path)

    assert path == tmp_path / "demo.pkl"
    assert loaded.name == "demo"
    assert loaded.path == tmp_path
    assert loaded.file_path == path
    assert loaded.keys() == ("exp_v", "offsetspec", "offsetanalysis")
    assert isinstance(loaded.exp_v, TransportDatasetSpec)
    assert np.allclose(loaded.exp_v.I_nA.values, [0.0, 1.0, 2.0])
    assert isinstance(loaded.offsetspec, SamplingSpec)
    assert np.allclose(loaded.offsetspec.Vbins_mV, [-1.0, 0.0, 1.0])
    assert np.allclose(loaded.offsetanalysis["Voff_mV"], [0.1, 0.2])


def test_project_cache_supports_dict_style_access(tmp_path: Path) -> None:
    cache = ProjectCache(name="demo", path=tmp_path)
    cache["exp_v"] = _make_transport_dataset()
    cache["offsetspec"] = _make_sampling_spec()
    cache["name"] = "stored item with reserved name"

    assert "exp_v" in cache
    assert cache.exp_v is cache["exp_v"]
    assert cache.get("missing", "fallback") == "fallback"
    assert cache["name"] == "stored item with reserved name"
    assert cache.name == "demo"
    assert tuple(iter(cache)) == ("exp_v", "offsetspec", "name")
    assert len(cache.values()) == 3

    del cache["name"]
    assert cache.keys() == ("exp_v", "offsetspec")


def test_project_cache_remove_entries(tmp_path: Path) -> None:
    cache = ProjectCache(name="demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset()
    cache.offsetspec = _make_sampling_spec()
    cache.offsetanalysis = {"Voff_mV": np.asarray([0.1, 0.2])}

    cache.remove("offsetspec")

    assert cache.keys() == ("exp_v", "offsetanalysis")
    assert "offsetspec" not in cache

    cache.remove("exp_v", "offsetanalysis")

    assert cache.keys() == ()


def test_project_cache_save_and_load_by_explicit_path(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)
    cache.answer = 42
    explicit_path = tmp_path / "cache-file.pkl"

    saved_path = save_cache(cache, path=explicit_path)
    loaded = load_cache(explicit_path)

    assert saved_path == explicit_path
    assert loaded.answer == 42


def test_project_cache_imports_from_utilities(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset()

    saved_path = save_cache(cache)
    loaded = load_cache(saved_path)

    assert loaded.name == "demo"
    assert np.allclose(loaded.exp_v.I_nA.values, [0.0, 1.0, 2.0])


def test_list_caches_discovers_project_cache_files(tmp_path: Path) -> None:
    make_cache("beta", path=tmp_path).save_cache()
    make_cache("alpha", path=tmp_path).save_cache()
    (tmp_path / "root.pkl").write_text("ignore", encoding="utf-8")

    assert list_caches(tmp_path) == ("alpha", "beta", "root")


def test_list_caches_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    assert list_caches(tmp_path) == ()


def test_entry_kind_groups_common_transportlab_objects(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset()
    cache.offsetspec = _make_sampling_spec()
    cache.answer = 42

    assert entry_kind(cache.exp_v) == "transport"
    assert entry_kind(cache.offsetspec) == "spec"
    assert entry_kind(cache.answer) == "misc"


def test_cache_summary_returns_table_friendly_rows(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset()
    cache.offsetspec = _make_sampling_spec()
    cache.answer = 42

    rows = cache_summary(cache)

    assert rows[0] == {
        "key": "exp_v",
        "kind": "transport",
        "type": "TransportDatasetSpec",
        "summary": "shape 3; axes V_mV",
    }
    assert rows[1] == {
        "key": "offsetspec",
        "kind": "spec",
        "type": "SamplingSpec",
        "summary": "13 keys",
    }
    assert rows[2] == {
        "key": "answer",
        "kind": "misc",
        "type": "int",
        "summary": "42",
    }


def test_remove_updates_cache_summary(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset()
    cache.offsetspec = _make_sampling_spec()

    cache.remove("offsetspec")

    assert cache_summary(cache) == (
        {
            "key": "exp_v",
            "kind": "transport",
            "type": "TransportDatasetSpec",
            "summary": "shape 3; axes V_mV",
        },
    )
