from __future__ import annotations

import numpy as np

from superconductivity.models.mar import btk as mar_btk


def test_get_Z_btk_matches_closed_form() -> None:
    assert mar_btk.get_Z_btk(0.5) == 1.0


def test_get_I_btk_nA_returns_three_channels() -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    I_nA = mar_btk.get_I_btk_nA(
        V_mV=V_mV,
        Delta_meV=0.18,
        tau=0.5,
        T_K=0.0,
        gamma_meV=1e-4,
    )

    assert I_nA.shape == (3, 3)
    assert np.isfinite(I_nA).all()
    np.testing.assert_allclose(I_nA[:, 0], -I_nA[::-1, 0])
