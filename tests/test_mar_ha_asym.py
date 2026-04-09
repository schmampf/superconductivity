from __future__ import annotations

import numpy as np

from superconductivity.models import ha_asym as legacy_ha_asym
from superconductivity.models.mar import ha_asym as mar_ha_asym


def test_legacy_ha_asym_module_reexports_mar_function() -> None:
    assert legacy_ha_asym.get_I_ha_asym_nA is mar_ha_asym.get_I_ha_asym_nA


def test_ha_asym_direct_path_works_without_cache(monkeypatch) -> None:
    def fake_curve(
        T_K: float,
        Delta_1_meV: float,
        Delta_2_meV: float,
        tau: float,
        gamma_1_meV: float,
        gamma_2_meV: float,
        V_positive_mV: np.ndarray,
    ) -> np.ndarray:
        return np.array(V_positive_mV, dtype=np.float64) * tau

    monkeypatch.setattr(
        mar_ha_asym.ha_asym,
        "ha_asym_curve",
        fake_curve,
    )

    V_mV = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float64)
    I_nA = legacy_ha_asym.get_I_ha_asym_nA(
        V_mV=V_mV,
        tau=0.4,
        T_K=0.0,
        Delta_meV=(0.2, 0.15),
        gamma_meV=(1e-4, 2e-4),
        caching=False,
    )

    np.testing.assert_allclose(I_nA, -I_nA[::-1])
