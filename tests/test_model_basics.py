from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from superconductivity.models.basics import get_DeltaT_meV, get_dos, get_f, get_Tc_K


def test_get_DeltaT_meV_matches_expected_limits() -> None:
    Delta_meV = 0.18
    T_c_K = get_Tc_K(Delta_meV)

    assert get_DeltaT_meV(Delta_meV, 0.0) == pytest.approx(Delta_meV)
    assert get_DeltaT_meV(Delta_meV, T_c_K) == pytest.approx(0.0)
    assert get_DeltaT_meV(Delta_meV, T_c_K + 0.1) == pytest.approx(0.0)


def test_get_f_zero_temperature_is_a_step_function() -> None:
    E_meV = np.array([-1.0, -0.1, 0.0, 0.1, 1.0], dtype=np.float64)
    np.testing.assert_allclose(get_f(E_meV, 0.0), [1.0, 1.0, 0.0, 0.0, 0.0])


def test_get_dos_is_even_in_energy() -> None:
    E_meV = np.array([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], dtype=np.float64)
    dos = get_dos(E_meV, Delta_meV=0.18, gamma_meV=1e-4)
    np.testing.assert_allclose(dos, dos[::-1])


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="jax is unavailable",
)
def test_basics_jnp_matches_numpy_helpers() -> None:
    import jax.numpy as jnp

    from superconductivity.models.basics.jnp import (
        get_DeltaT_meV_jnp,
        get_dos_jnp,
        get_f_jnp,
        get_Tc_K_jnp,
    )

    Delta_meV = 0.18
    T_K = 0.9
    E_meV = np.array([-0.3, -0.1, 0.1, 0.3], dtype=np.float64)
    gamma_meV = 1e-4

    T_c_jnp = np.asarray(get_Tc_K_jnp(jnp.asarray(Delta_meV)))
    Delta_jnp = np.asarray(get_DeltaT_meV_jnp(jnp.asarray(Delta_meV), jnp.asarray(T_K)))
    f_jnp = np.asarray(get_f_jnp(jnp.asarray(E_meV), jnp.asarray(T_K)))
    dos_jnp = np.asarray(
        get_dos_jnp(
            jnp.asarray(E_meV),
            jnp.asarray(Delta_meV),
            jnp.asarray(gamma_meV),
        )
    )

    assert T_c_jnp == pytest.approx(get_Tc_K(Delta_meV))
    assert Delta_jnp == pytest.approx(get_DeltaT_meV(Delta_meV, T_K))
    np.testing.assert_allclose(f_jnp, get_f(E_meV, T_K))
    np.testing.assert_allclose(dos_jnp, get_dos(E_meV, Delta_meV, gamma_meV))
