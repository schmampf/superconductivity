from __future__ import annotations

import numpy as np
import pytest

from superconductivity.models.mar import get_Imar_nA
from superconductivity.models.mar import mar as mar_dispatch
from superconductivity.utilities.constants import G_0_muS


def test_get_Imar_nA_dispatches_to_ha_sym(monkeypatch) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    def fake_ha_sym(**kwargs) -> np.ndarray:
        assert kwargs["Delta_meV"] == pytest.approx(0.18)
        assert kwargs["gamma_meV"] == pytest.approx(1e-4)
        return np.array([-1.0, 0.0, 1.0], dtype=np.float64)

    monkeypatch.setattr(mar_dispatch, "get_I_ha_sym_nA", fake_ha_sym)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_asym_nA",
        lambda **kwargs: pytest.fail("ha_asym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_btk_nA",
        lambda **kwargs: pytest.fail("btk should not be used"),
    )

    I_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=0.0,
        Delta_meV=(0.18, 0.18),
        gamma_meV=(1e-4, 1e-4),
    )

    np.testing.assert_allclose(I_nA, np.array([-1.0, 0.0, 1.0]))


def test_get_Imar_nA_dispatches_to_ha_asym(monkeypatch) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    def fake_ha_asym(**kwargs) -> np.ndarray:
        assert kwargs["Delta_meV"] == pytest.approx((0.18, 0.12))
        assert kwargs["gamma_meV"] == pytest.approx((1e-4, 2e-4))
        return np.array([-2.0, 0.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_sym_nA",
        lambda **kwargs: pytest.fail("ha_sym should not be used"),
    )
    monkeypatch.setattr(mar_dispatch, "get_I_ha_asym_nA", fake_ha_asym)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_btk_nA",
        lambda **kwargs: pytest.fail("btk should not be used"),
    )

    I_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=0.0,
        Delta_meV=(0.18, 0.12),
        gamma_meV=(1e-4, 2e-4),
    )

    np.testing.assert_allclose(I_nA, np.array([-2.0, 0.0, 2.0]))


def test_get_Imar_nA_dispatches_to_btk_when_one_gap_closes(monkeypatch) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    captured: dict[str, float] = {}

    def fake_btk(**kwargs) -> np.ndarray:
        captured["Delta_meV"] = kwargs["Delta_meV"]
        captured["gamma_meV"] = kwargs["gamma_meV"]
        return np.column_stack(
            (
                np.array([-3.0, 0.0, 3.0], dtype=np.float64),
                np.zeros((3, 2), dtype=np.float64),
            )
        )

    monkeypatch.setattr(mar_dispatch, "get_I_btk_nA", fake_btk)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_sym_nA",
        lambda **kwargs: pytest.fail("ha_sym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_asym_nA",
        lambda **kwargs: pytest.fail("ha_asym should not be used"),
    )

    I_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=0.5,
        Delta_meV=(0.05, 0.20),
        gamma_meV=(1e-4, 2e-4),
    )

    assert captured["Delta_meV"] == pytest.approx(0.20)
    assert captured["gamma_meV"] == pytest.approx(2e-4)
    np.testing.assert_allclose(I_nA, np.array([-3.0, 0.0, 3.0]))


def test_get_Imar_nA_returns_ohmic_current_when_both_gaps_close() -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    I_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=2.0,
        Delta_meV=(0.05, 0.06),
    )

    np.testing.assert_allclose(I_nA, V_mV * 0.5 * G_0_muS)


def test_get_Imar_nA_sums_multiple_tau_values(monkeypatch) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    def fake_ha_sym(**kwargs) -> np.ndarray:
        return kwargs["tau"] * np.array([-10.0, 0.0, 10.0], dtype=np.float64)

    monkeypatch.setattr(mar_dispatch, "get_I_ha_sym_nA", fake_ha_sym)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_asym_nA",
        lambda **kwargs: pytest.fail("ha_asym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_btk_nA",
        lambda **kwargs: pytest.fail("btk should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_fcs_nA",
        lambda **kwargs: pytest.fail("fcs should not be used"),
    )

    I_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=(0.2, 0.3),
        T_K=0.0,
        Delta_meV=(0.18, 0.18),
        gamma_meV=(1e-4, 1e-4),
    )

    np.testing.assert_allclose(I_nA, np.array([-5.0, 0.0, 5.0]))


def test_get_Imar_nA_tau_resolved_returns_per_transmission_currents(
    monkeypatch,
) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    def fake_ha_sym(**kwargs) -> np.ndarray:
        return kwargs["tau"] * np.array([-10.0, 0.0, 10.0], dtype=np.float64)

    monkeypatch.setattr(mar_dispatch, "get_I_ha_sym_nA", fake_ha_sym)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_asym_nA",
        lambda **kwargs: pytest.fail("ha_asym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_btk_nA",
        lambda **kwargs: pytest.fail("btk should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_fcs_nA",
        lambda **kwargs: pytest.fail("fcs should not be used"),
    )

    I_nA, Ix_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=(0.2, 0.3),
        T_K=0.0,
        Delta_meV=(0.18, 0.18),
        gamma_meV=(1e-4, 1e-4),
        tau_resolved=True,
    )

    np.testing.assert_allclose(I_nA, np.array([-5.0, 0.0, 5.0]))
    np.testing.assert_allclose(
        Ix_nA,
        np.array(
            [
                [-2.0, -3.0],
                [0.0, 0.0],
                [2.0, 3.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(I_nA, Ix_nA.sum(axis=1))


def test_get_Imar_nA_charge_resolved_uses_fcs_for_superconducting_leads(
    monkeypatch,
) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    def fake_fcs(**kwargs) -> np.ndarray:
        assert kwargs["nmax"] == 3
        return np.array(
            [
                [-5.0, -2.0, -3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 2.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(mar_dispatch, "get_I_fcs_nA", fake_fcs)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_sym_nA",
        lambda **kwargs: pytest.fail("ha_sym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_asym_nA",
        lambda **kwargs: pytest.fail("ha_asym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_btk_nA",
        lambda **kwargs: pytest.fail("btk should not be used"),
    )

    I_nA, Iq_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=0.0,
        Delta_meV=(0.18, 0.18),
        gamma_meV=(1e-4, 1e-4),
        charge_resolved=True,
        qmax=3,
    )

    np.testing.assert_allclose(I_nA, np.array([-5.0, 0.0, 5.0]))
    np.testing.assert_allclose(
        Iq_nA,
        np.array(
            [
                [-2.0, -3.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(I_nA, Iq_nA.sum(axis=1))


def test_get_Imar_nA_charge_and_tau_resolved_return_all_aggregates(
    monkeypatch,
) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    def fake_fcs(**kwargs) -> np.ndarray:
        tau = kwargs["tau"]
        return np.array(
            [
                [-5.0 * tau, -2.0 * tau, -3.0 * tau, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [5.0 * tau, 2.0 * tau, 3.0 * tau, 0.0],
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(mar_dispatch, "get_I_fcs_nA", fake_fcs)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_sym_nA",
        lambda **kwargs: pytest.fail("ha_sym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_asym_nA",
        lambda **kwargs: pytest.fail("ha_asym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_btk_nA",
        lambda **kwargs: pytest.fail("btk should not be used"),
    )

    I_nA, Ix_nA, Iq_nA, Ixq_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=(0.2, 0.3),
        T_K=0.0,
        Delta_meV=(0.18, 0.18),
        gamma_meV=(1e-4, 1e-4),
        charge_resolved=True,
        tau_resolved=True,
        qmax=3,
    )

    np.testing.assert_allclose(
        Ix_nA,
        np.array(
            [
                [-1.0, -1.5],
                [0.0, 0.0],
                [1.0, 1.5],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        Iq_nA,
        np.array(
            [
                [-1.0, -1.5, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.5, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        Ixq_nA,
        np.array(
            [
                [[-0.4, -0.6, 0.0], [-0.6, -0.9, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.4, 0.6, 0.0], [0.6, 0.9, 0.0]],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(I_nA, np.array([-2.5, 0.0, 2.5]))
    np.testing.assert_allclose(I_nA, Ix_nA.sum(axis=1))
    np.testing.assert_allclose(I_nA, Iq_nA.sum(axis=1))
    np.testing.assert_allclose(Ix_nA, Ixq_nA.sum(axis=2))
    np.testing.assert_allclose(Iq_nA, Ixq_nA.sum(axis=1))


def test_get_Imar_nA_charge_resolved_maps_btk_channels(monkeypatch) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    def fake_btk(**kwargs) -> np.ndarray:
        return np.array(
            [
                [-6.0, -2.0, -4.0],
                [0.0, 0.0, 0.0],
                [6.0, 2.0, 4.0],
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(mar_dispatch, "get_I_btk_nA", fake_btk)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_sym_nA",
        lambda **kwargs: pytest.fail("ha_sym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_asym_nA",
        lambda **kwargs: pytest.fail("ha_asym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_fcs_nA",
        lambda **kwargs: pytest.fail("fcs should not be used"),
    )

    I_nA, Iq_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=0.5,
        Delta_meV=(0.05, 0.20),
        gamma_meV=(1e-4, 2e-4),
        charge_resolved=True,
        qmax=4,
    )

    np.testing.assert_allclose(I_nA, np.array([-6.0, 0.0, 6.0]))
    np.testing.assert_allclose(
        Iq_nA,
        np.array(
            [
                [-2.0, -4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(I_nA, Iq_nA.sum(axis=1))


def test_get_Imar_nA_scalar_tau_still_has_tau_axis_when_resolved(
    monkeypatch,
) -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    def fake_btk(**kwargs) -> np.ndarray:
        return np.array(
            [
                [-6.0, -2.0, -4.0],
                [0.0, 0.0, 0.0],
                [6.0, 2.0, 4.0],
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(mar_dispatch, "get_I_btk_nA", fake_btk)
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_sym_nA",
        lambda **kwargs: pytest.fail("ha_sym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_ha_asym_nA",
        lambda **kwargs: pytest.fail("ha_asym should not be used"),
    )
    monkeypatch.setattr(
        mar_dispatch,
        "get_I_fcs_nA",
        lambda **kwargs: pytest.fail("fcs should not be used"),
    )

    I_nA, Ix_nA, Iq_nA, Ixq_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=0.5,
        Delta_meV=(0.05, 0.20),
        gamma_meV=(1e-4, 2e-4),
        charge_resolved=True,
        tau_resolved=True,
        qmax=4,
    )

    assert Ix_nA.shape == (3, 1)
    assert Iq_nA.shape == (3, 4)
    assert Ixq_nA.shape == (3, 1, 4)
    np.testing.assert_allclose(I_nA, Ix_nA[:, 0])
    np.testing.assert_allclose(Iq_nA, Ixq_nA[:, 0, :])


def test_get_Imar_nA_charge_resolved_ohmic_has_only_1e_channel() -> None:
    V_mV = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    I_nA, Iq_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=0.5,
        T_K=2.0,
        Delta_meV=(0.05, 0.06),
        charge_resolved=True,
        qmax=3,
    )

    np.testing.assert_allclose(I_nA, V_mV * 0.5 * G_0_muS)
    np.testing.assert_allclose(Iq_nA[:, 0], I_nA)
    np.testing.assert_allclose(Iq_nA[:, 1:], 0.0)
    np.testing.assert_allclose(I_nA, Iq_nA.sum(axis=1))


def test_get_Imar_nA_charge_resolved_requires_qmax_at_least_two() -> None:
    with pytest.raises(
        ValueError,
        match="qmax must be at least 2 when charge_resolved=True",
    ):
        get_Imar_nA(
            V_mV=np.array([-0.1, 0.0, 0.1], dtype=np.float64),
            charge_resolved=True,
            qmax=1,
        )


def test_get_Imar_nA_rejects_empty_tau_sequence() -> None:
    with pytest.raises(ValueError, match="tau must contain at least one value"):
        get_Imar_nA(
            V_mV=np.array([-0.1, 0.0, 0.1], dtype=np.float64),
            tau=[],
        )
