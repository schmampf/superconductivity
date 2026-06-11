from __future__ import annotations

import builtins
import importlib
import importlib.util

import numpy as np
import pytest

from superconductivity.models.basics import get_DeltaT_meV
from superconductivity.models.bcs.backend import E0_meV
from superconductivity.models.bcs.backend.np import adaptive_np, integral_np
from superconductivity.models.bcs.bcs import get_Ibcs_nA, sim_bcs
from superconductivity.utilities.constants import G0_muS
from superconductivity.utilities.meta import axis, param

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="scipy is unavailable",
)

np_backend = importlib.import_module("superconductivity.models.bcs.backend.np")


def test_adaptive_kernel_resolves_low_gamma_gap_edges() -> None:
    Delta_meV = 0.19
    T_K = 0.1
    DeltaT_meV = get_DeltaT_meV(Delta_meV, T_K)
    positive = np.array(
        [0.379, 2.0 * DeltaT_meV, 0.4, 0.46],
        dtype=np.float64,
    )
    V_mV = np.concatenate((-positive[::-1], [0.0], positive))

    current = adaptive_np(
        V_mV,
        E0_meV,
        T_K,
        T_K,
        Delta_meV,
        Delta_meV,
        1e-7,
        1e-7,
    )

    assert np.all(np.isfinite(current))
    np.testing.assert_allclose(current, -current[::-1], rtol=0.0, atol=1e-12)
    assert current[-4] < 3e-5
    assert current[-3] == pytest.approx(0.1492245661, rel=2e-6)
    assert current[-2] == pytest.approx(0.3218661631, rel=2e-6)
    assert current[-1] == pytest.approx(0.3905903001, rel=2e-6)


def test_adaptive_kernel_ignores_interior_energy_grid_alignment() -> None:
    V_mV = np.array([0.379, 0.4, 0.46], dtype=np.float64)
    shifted_interior = np.concatenate(
        (
            [-4.0],
            np.linspace(-3.9993, 3.9987, 173, dtype=np.float64),
            [4.0],
        )
    )

    reference = adaptive_np(
        V_mV,
        E0_meV,
        0.1,
        0.1,
        0.19,
        0.19,
        1e-7,
        1e-7,
    )
    shifted = adaptive_np(
        V_mV,
        shifted_interior,
        0.1,
        0.1,
        0.19,
        0.19,
        1e-7,
        1e-7,
    )

    np.testing.assert_allclose(shifted, reference, rtol=0.0, atol=1e-12)


def test_adaptive_kernel_matches_fine_integral_at_moderate_gamma() -> None:
    V_mV = np.array([-0.46, -0.4, 0.4, 0.46], dtype=np.float64)
    E_meV = np.linspace(-1.0, 1.0, 100_001, dtype=np.float64)

    adaptive = adaptive_np(
        V_mV,
        E_meV,
        0.1,
        0.1,
        0.19,
        0.19,
        0.002,
        0.002,
    )
    reference = integral_np(
        V_mV,
        E_meV,
        0.1,
        0.1,
        0.19,
        0.19,
        0.002,
        0.002,
    )

    np.testing.assert_allclose(adaptive, reference, rtol=2e-7, atol=1e-10)


def test_adaptive_kernel_zero_gamma_uses_numerical_floor() -> None:
    V_mV = np.array([0.4], dtype=np.float64)

    zero = adaptive_np(V_mV, E0_meV, 0.1, 0.1, 0.19, 0.19, 0.0, 0.0)
    floor = adaptive_np(V_mV, E0_meV, 0.1, 0.1, 0.19, 0.19, 1e-12, 1e-12)

    np.testing.assert_array_equal(zero, floor)


def test_adaptive_kernel_rejects_negative_gamma() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        adaptive_np(
            np.array([0.4]),
            E0_meV,
            0.1,
            0.1,
            0.19,
            0.19,
            -1e-7,
            1e-7,
        )


def test_adaptive_kernel_reports_missing_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "scipy.integrate":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="adaptive BCS kernel requires scipy"):
        np_backend._import_quad()


def test_public_adaptive_kernel_scales_normal_conductance() -> None:
    V_mV = np.array([-0.4, 0.0, 0.4], dtype=np.float64)
    unit_current = adaptive_np(
        V_mV,
        E0_meV,
        0.1,
        0.1,
        0.19,
        0.19,
        1e-7,
        1e-7,
    )

    current = get_Ibcs_nA(
        V_mV,
        GN_G0=0.2,
        T_K=0.1,
        Delta_meV=0.19,
        gamma_meV=1e-7,
        backend="np",
        kernel="adaptive",
    )

    np.testing.assert_allclose(
        current,
        unit_current * (0.2 * float(G0_muS)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_public_adaptive_kernel_rejects_jax_backend() -> None:
    with pytest.raises(ValueError, match="requires backend='np'"):
        get_Ibcs_nA(
            np.array([-0.4, 0.0, 0.4]),
            GN_G0=0.2,
            T_K=0.1,
            Delta_meV=0.19,
            gamma_meV=1e-7,
            backend="jax",
            kernel="adaptive",
        )


def test_sim_bcs_supports_adaptive_gamma_sweep() -> None:
    V_mV = np.array([-0.4, 0.0, 0.4], dtype=np.float64)

    dataset = sim_bcs(
        V_mV=axis("V_mV", values=V_mV, order=1),
        GN_G0=param("GN_G0", 0.2),
        T_K=param("T_K", 0.1),
        Delta_meV=param("Delta_meV", 0.19),
        gamma_meV=axis("gamma_meV", values=[1e-7, 2e-7], order=0),
        backend="np",
        kernel="adaptive",
    )

    assert dataset.I_nA.values.shape == (2, V_mV.size)
    assert np.all(np.isfinite(dataset.I_nA.values))
