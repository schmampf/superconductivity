from __future__ import annotations

import numpy as np

from superconductivity.models import ha_new


class _FakeDataset:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.array(data, copy=True)

    def __getitem__(self, key):
        if key is Ellipsis:
            return self._data
        return self._data[key]


class _FakeGroup:
    def __init__(self) -> None:
        self.children: dict[str, _FakeGroup | _FakeDataset] = {}
        self.attrs: dict[str, object] = {}

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __getitem__(self, key: str):
        node = self
        for part in key.split("/"):
            if not part:
                continue
            if not isinstance(node, _FakeGroup) or part not in node.children:
                raise KeyError(key)
            node = node.children[part]
        return node

    def __delitem__(self, key: str) -> None:
        del self.children[key]

    def require_group(self, key: str) -> "_FakeGroup":
        if key in self.children:
            node = self.children[key]
            if not isinstance(node, _FakeGroup):
                raise TypeError(key)
            return node
        group = _FakeGroup()
        self.children[key] = group
        return group

    def create_group(self, key: str) -> "_FakeGroup":
        group = _FakeGroup()
        self.children[key] = group
        return group

    def create_dataset(self, key: str, data, **_: object) -> _FakeDataset:
        dataset = _FakeDataset(np.array(data, copy=True))
        self.children[key] = dataset
        return dataset

    def keys(self):
        return self.children.keys()


class _FakeFile(_FakeGroup):
    def __enter__(self) -> "_FakeFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeH5py:
    def __init__(self) -> None:
        self.files: dict[str, _FakeFile] = {}

    def File(self, path, mode: str = "a") -> _FakeFile:
        key = str(path)
        if key not in self.files:
            self.files[key] = _FakeFile()
        return self.files[key]


def test_parameter_only_cache_reuses_existing_points(
    monkeypatch,
    tmp_path,
) -> None:
    fake_h5py = _FakeH5py()
    monkeypatch.setattr(ha_new, "_import_h5py", lambda: fake_h5py)
    monkeypatch.setattr(ha_new, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ha_new, "CACHE_FILE", tmp_path / "ha_sym.h5")
    ha_new.CACHE_FILE.touch()

    calls: list[np.ndarray] = []

    def fake_curve(
        tau: float,
        T_Delta: float,
        gamma_Delta: float,
        E_min: float,
        E_max: float,
        V_Delta: np.ndarray,
    ) -> np.ndarray:
        calls.append(np.array(V_Delta, copy=True))
        return np.array(V_Delta, dtype=np.float64)

    monkeypatch.setattr(ha_new.ha_sym, "ha_sym_curve", fake_curve)

    first = ha_new.get_I_ha_sym_nA(
        np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        tau=0.5,
        T_K=0.0,
        Delta_meV=0.2,
        gamma_meV=1e-4,
        caching=True,
    )
    second = ha_new.get_I_ha_sym_nA(
        np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float64),
        tau=0.5,
        T_K=0.0,
        Delta_meV=0.2,
        gamma_meV=1e-4,
        caching=True,
    )

    assert len(calls) == 2
    assert calls[0].shape == (1,)
    assert calls[1].shape == (1,)
    assert first[1] == 0.0
    np.testing.assert_allclose(first[[0, 2]], -first[[2, 0]])
    np.testing.assert_allclose(second[[0, 3]], -second[[3, 0]])

    root = fake_h5py.files[str(tmp_path / "ha_sym.h5")]
    curves = root["curves"]
    assert len(curves.children) == 1
    group = next(iter(curves.children.values()))
    assert group.attrs["voltage_count"] == 2


def test_warm_cache_avoids_recomputing_existing_points(
    monkeypatch,
    tmp_path,
) -> None:
    fake_h5py = _FakeH5py()
    monkeypatch.setattr(ha_new, "_import_h5py", lambda: fake_h5py)
    monkeypatch.setattr(ha_new, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ha_new, "CACHE_FILE", tmp_path / "ha_sym.h5")

    calls: list[np.ndarray] = []

    def fake_curve(
        tau: float,
        T_Delta: float,
        gamma_Delta: float,
        E_min: float,
        E_max: float,
        V_Delta: np.ndarray,
    ) -> np.ndarray:
        calls.append(np.array(V_Delta, copy=True))
        return np.array(V_Delta, dtype=np.float64)

    monkeypatch.setattr(ha_new.ha_sym, "ha_sym_curve", fake_curve)

    V_cached_mV, I_cached_nA = ha_new.warm_cache_ha_sym(
        V_max_mV=2.0,
        dV_mV=0.5,
        tau=0.4,
        T_K=0.0,
        Delta_meV=0.2,
        gamma_meV=1e-4,
    )
    assert len(calls) == 1
    np.testing.assert_allclose(V_cached_mV, [0.5, 1.0, 1.5, 2.0])
    assert I_cached_nA.shape == (4,)

    calls.clear()
    result = ha_new.get_I_ha_sym_nA(
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64),
        tau=0.4,
        T_K=0.0,
        Delta_meV=0.2,
        gamma_meV=1e-4,
        caching=True,
    )

    assert calls == []
    assert result.shape == (5,)


def test_explore_cached_ha_sym_returns_metadata_and_arrays(
    monkeypatch,
    tmp_path,
) -> None:
    fake_h5py = _FakeH5py()
    monkeypatch.setattr(ha_new, "_import_h5py", lambda: fake_h5py)
    monkeypatch.setattr(ha_new, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ha_new, "CACHE_FILE", tmp_path / "ha_sym.h5")
    ha_new.CACHE_FILE.touch()

    def fake_curve(
        tau: float,
        T_Delta: float,
        gamma_Delta: float,
        E_min: float,
        E_max: float,
        V_Delta: np.ndarray,
    ) -> np.ndarray:
        return np.array(V_Delta, dtype=np.float64)

    monkeypatch.setattr(ha_new.ha_sym, "ha_sym_curve", fake_curve)

    ha_new.get_I_ha_sym_nA(
        np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        tau=0.5,
        T_K=0.0,
        Delta_meV=0.2,
        gamma_meV=1e-4,
        caching=True,
    )
    ha_new.get_I_ha_sym_nA(
        np.array([-0.5, 0.0, 0.5], dtype=np.float64),
        tau=0.4,
        T_K=0.1,
        Delta_meV=0.25,
        gamma_meV=2e-4,
        caching=True,
    )

    summary = ha_new.explore_cached_ha_sym()
    assert len(summary) == 2
    assert summary[0]["voltage_count"] == 1
    assert summary[0]["V_min_mV"] is not None
    assert "V_mV" not in summary[0]

    detailed = ha_new.explore_cached_ha_sym(include_data=True)
    assert len(detailed) == 2
    assert np.all(np.array(detailed[0]["V_mV"]) > 0.0)
    assert np.array(detailed[0]["I_nA"]).shape == np.array(
        detailed[0]["V_mV"],
    ).shape
