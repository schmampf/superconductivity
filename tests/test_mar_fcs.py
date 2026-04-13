from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from superconductivity.models.basics import get_Delta_meV
from superconductivity.models.mar import fcs as mar_fcs
from superconductivity.models.mar.core import FCSParams
from superconductivity.models.mar.core import load_curve
from superconductivity.utilities.constants import k_B_meV


def test_fcs_direct_path_works_without_cache(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_curve(
        trans_in: float,
        temp_in: float,
        delta1_T_in: float,
        delta2_T_in: float,
        eta1_in: float,
        eta2_in: float,
        voltages: np.ndarray,
        nmax_in: int,
        iw_in: int,
        nchi_in: int,
    ) -> np.ndarray:
        captured["trans_in"] = trans_in
        captured["temp_in"] = temp_in
        captured["delta1_T_in"] = delta1_T_in
        captured["delta2_T_in"] = delta2_T_in
        captured["eta1_in"] = eta1_in
        captured["eta2_in"] = eta2_in
        captured["voltages"] = np.asarray(voltages, dtype=np.float64)
        weights = np.arange(1, nmax_in + 2, dtype=np.float64)
        return np.asarray(voltages, dtype=np.float64)[:, None] * weights[None, :]

    monkeypatch.setattr(
        mar_fcs.fcs,
        "fcs_curve",
        fake_curve,
    )

    V_mV = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float64)
    Delta_1_meV = 0.2
    Delta_2_meV = 0.15
    gamma_1_meV = 1e-4
    gamma_2_meV = 2e-4
    T_K = 0.5
    I_nA = mar_fcs.get_I_fcs_nA(
        V_mV=V_mV,
        tau=0.4,
        T_K=T_K,
        Delta_meV=(Delta_1_meV, Delta_2_meV),
        gamma_meV=(gamma_1_meV, gamma_2_meV),
        nmax=2,
        caching=False,
    )

    assert captured["trans_in"] == 0.4
    assert captured["temp_in"] == pytest.approx(k_B_meV * T_K)
    assert captured["delta1_T_in"] == pytest.approx(get_Delta_meV(Delta_1_meV, T_K))
    assert captured["delta2_T_in"] == pytest.approx(get_Delta_meV(Delta_2_meV, T_K))
    assert captured["eta1_in"] == pytest.approx(gamma_1_meV)
    assert captured["eta2_in"] == pytest.approx(gamma_2_meV)
    np.testing.assert_allclose(
        captured["voltages"],
        np.array([0.1, 0.2], dtype=np.float64),
    )
    assert I_nA.shape == (V_mV.size, 3)
    np.testing.assert_allclose(I_nA, -I_nA[::-1])
    np.testing.assert_allclose(
        I_nA[-2:, 0],
        np.array([0.1, 0.2], dtype=np.float64),
    )


def test_fcs_cache_reuses_missing_positive_bins(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[np.ndarray] = []

    def fake_curve(
        trans_in: float,
        temp_in: float,
        delta1_in: float,
        delta2_in: float,
        eta1_in: float,
        eta2_in: float,
        voltages: np.ndarray,
        nmax_in: int,
        iw_in: int,
        nchi_in: int,
    ) -> np.ndarray:
        calls.append(np.array(voltages, copy=True, dtype=np.float64))
        weights = np.arange(1, nmax_in + 2, dtype=np.float64)
        return np.asarray(voltages, dtype=np.float64)[:, None] * weights[None, :]

    monkeypatch.setattr(mar_fcs, "CACHE_FILE", tmp_path / "cache.h5")
    monkeypatch.setattr(
        mar_fcs.fcs,
        "fcs_curve",
        fake_curve,
    )

    V_first_mV = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    I_first_nA = mar_fcs.get_I_fcs_nA(
        V_mV=V_first_mV,
        tau=0.4,
        nmax=2,
        caching=True,
    )
    assert I_first_nA.shape == (3, 3)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], np.array([1.0], dtype=np.float64))

    V_second_mV = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    I_second_nA = mar_fcs.get_I_fcs_nA(
        V_mV=V_second_mV,
        tau=0.4,
        nmax=2,
        caching=True,
    )
    assert I_second_nA.shape == (5, 3)
    assert len(calls) == 2
    np.testing.assert_allclose(calls[1], np.array([0.5], dtype=np.float64))
    np.testing.assert_allclose(I_second_nA, -I_second_nA[::-1])

    params = FCSParams.from_raw(
        tau=0.4,
        T_K=0.0,
        Delta_meV=(0.18, 0.18),
        gamma_meV=0.0,
        gamma_meV_min=1e-4,
        nmax=2,
        iw=2003,
        nchi=66,
    )
    V_cached_q, I_cached_nA = load_curve(
        cache_file=mar_fcs.CACHE_FILE,
        group_path=f"fcs/curves/{params.cache_key()}",
    )
    np.testing.assert_array_equal(
        V_cached_q,
        np.array([500_000, 1_000_000], dtype=np.int64),
    )
    assert I_cached_nA.shape == (2, 3)
