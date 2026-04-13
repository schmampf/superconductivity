from __future__ import annotations

import numpy as np
import pytest

from superconductivity.models.mar import ha_asym as mar_ha_asym
from superconductivity.models.mar import get_I_ha_sym_nA
from superconductivity.models.basics import get_Delta_meV
from superconductivity.utilities.constants import G_0_muS, k_B_meV


def test_ha_asym_direct_path_works_without_cache(monkeypatch) -> None:
    captured: dict[str, np.ndarray | float] = {}

    def fake_curve(
        tau: float,
        temp_reduced: float,
        Delta_1_reduced: float,
        Delta_2_reduced: float,
        gamma_1_reduced: float,
        gamma_2_reduced: float,
        V_positive_reduced: np.ndarray,
    ) -> np.ndarray:
        captured["tau"] = tau
        captured["temp_reduced"] = temp_reduced
        captured["Delta_1_reduced"] = Delta_1_reduced
        captured["Delta_2_reduced"] = Delta_2_reduced
        captured["gamma_1_reduced"] = gamma_1_reduced
        captured["gamma_2_reduced"] = gamma_2_reduced
        captured["V_positive_reduced"] = np.array(
            V_positive_reduced,
            dtype=np.float64,
        )
        return np.array(V_positive_reduced, dtype=np.float64) * tau

    monkeypatch.setattr(
        mar_ha_asym.ha_asym,
        "ha_asym_curve",
        fake_curve,
    )

    V_mV = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float64)
    Delta_1_meV = 0.2
    Delta_2_meV = 0.15
    gamma_1_meV = 1e-4
    gamma_2_meV = 2e-4
    T_K = 0.5
    I_nA = mar_ha_asym.get_I_ha_asym_nA(
        V_mV=V_mV,
        tau=0.4,
        T_K=T_K,
        Delta_meV=(Delta_1_meV, Delta_2_meV),
        gamma_meV=(gamma_1_meV, gamma_2_meV),
        caching=False,
    )

    assert captured["tau"] == pytest.approx(0.4)
    assert captured["temp_reduced"] == pytest.approx(k_B_meV * T_K / Delta_1_meV)
    assert captured["Delta_1_reduced"] == pytest.approx(
        get_Delta_meV(Delta_1_meV, T_K) / Delta_1_meV
    )
    assert captured["Delta_2_reduced"] == pytest.approx(
        get_Delta_meV(Delta_2_meV, T_K) / Delta_1_meV
    )
    assert captured["gamma_1_reduced"] == pytest.approx(gamma_1_meV / Delta_1_meV)
    assert captured["gamma_2_reduced"] == pytest.approx(gamma_2_meV / Delta_1_meV)
    np.testing.assert_allclose(
        captured["V_positive_reduced"],
        np.array([0.1, 0.2], dtype=np.float64) / Delta_1_meV,
    )
    np.testing.assert_allclose(I_nA, -I_nA[::-1])
    np.testing.assert_allclose(
        I_nA[-2:],
        0.4 * np.array([0.1, 0.2], dtype=np.float64) * G_0_muS,
    )


@pytest.mark.skipif(
    not callable(getattr(mar_ha_asym.ha_asym, "ha_asym_curve", None)),
    reason="compiled asymmetric HA backend is unavailable",
)
def test_ha_asym_matches_ha_sym_in_symmetric_limit() -> None:
    V_mV = np.linspace(-0.4, 0.4, 9, dtype=np.float64)

    I_sym_nA = get_I_ha_sym_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=0.0,
        Delta_meV=0.18,
        gamma_meV=1e-4,
        caching=False,
    )
    I_asym_nA = mar_ha_asym.get_I_ha_asym_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=0.0,
        Delta_meV=(0.18, 0.18),
        gamma_meV=(1e-4, 1e-4),
        caching=False,
    )

    np.testing.assert_allclose(I_asym_nA, I_sym_nA, rtol=3e-2, atol=5e-2)
