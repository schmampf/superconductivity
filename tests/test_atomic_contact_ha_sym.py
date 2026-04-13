from __future__ import annotations

import numpy as np

from superconductivity.models.mar import ha_sym as mar_ha_sym


def test_ha_sym_direct_path_works_without_cache(
    monkeypatch,
) -> None:
    def fake_curve(
        tau: float,
        T_Delta: float,
        gamma_Delta: float,
        E_min: float,
        E_max: float,
        V_Delta: np.ndarray,
    ) -> np.ndarray:
        return np.array(V_Delta, dtype=np.float64)

    monkeypatch.setattr(
        mar_ha_sym.ha_sym,
        "ha_sym_curve",
        fake_curve,
    )

    V_mV = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float64)
    I_direct = mar_ha_sym.get_I_ha_sym_nA(
        V_mV=V_mV,
        tau=0.4,
        T_K=0.0,
        Delta_meV=0.2,
        gamma_meV=1e-4,
        caching=False,
    )

    np.testing.assert_allclose(I_direct, -I_direct[::-1])
