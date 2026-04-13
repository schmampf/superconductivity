from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from superconductivity.models.bcs import get_I_pat_nA
from superconductivity.models.bcs.bcs import get_Ibcs_nA
from superconductivity.models.bcs.backend import PAT_N_MAX
from superconductivity.models.basics.noise import apply_voltage_noise, make_bias_support_grid

_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
_JAX_AVAILABLE = importlib.util.find_spec("jax") is not None



def test_api_exports_new_bcs_entry_point() -> None:
    pytest.importorskip("matplotlib")

    import superconductivity.api as sc

    assert sc.get_Ibcs_nA is get_Ibcs_nA



@pytest.mark.skipif(not _JAX_AVAILABLE, reason="jax is unavailable")
def test_get_Ibcs_nA_defaults_to_jax_convolution() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)

    default = get_Ibcs_nA(V_mV, 0.2, 0.1, 0.19, 0.002)
    explicit = get_Ibcs_nA(
        V_mV,
        0.2,
        0.1,
        0.19,
        0.002,
        backend="jax",
        kernel="conv",
    )

    assert default.shape == V_mV.shape
    assert np.all(np.isfinite(default))
    assert np.allclose(default, explicit)



@pytest.mark.skipif(not _JAX_AVAILABLE, reason="jax is unavailable")
def test_get_Ibcs_nA_resolves_backend_kernel_matrix() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)

    for backend in ("np", "jax"):
        for kernel in ("int", "conv"):
            current = get_Ibcs_nA(
                V_mV,
                0.2,
                0.1,
                0.19,
                0.002,
                backend=backend,
                kernel=kernel,
            )
            assert current.shape == V_mV.shape
            assert np.all(np.isfinite(current))



@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy is unavailable")
def test_get_Ibcs_nA_scalar_amplitude_returns_one_curve() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)

    current = get_Ibcs_nA(
        V_mV,
        0.2,
        0.1,
        0.19,
        0.002,
        nu_GHz=8.5,
        A_mV=0.15,
        backend="np",
    )

    assert current.shape == V_mV.shape



@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy is unavailable")
def test_get_Ibcs_nA_array_amplitude_returns_curve_stack() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    amplitudes = np.array([0.0, 0.05, 0.1], dtype=np.float64)

    current = get_Ibcs_nA(
        V_mV,
        0.2,
        0.1,
        0.19,
        0.002,
        nu_GHz=8.5,
        A_mV=amplitudes,
        backend="np",
    )

    assert current.shape == (amplitudes.size, V_mV.size)



def test_get_Ibcs_nA_zero_amplitude_is_plain_bcs() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)

    plain = get_Ibcs_nA(V_mV, 0.2, 0.1, 0.19, 0.002, backend="np")
    pat_zero = get_Ibcs_nA(
        V_mV,
        0.2,
        0.1,
        0.19,
        0.002,
        nu_GHz=8.5,
        A_mV=0.0,
        backend="np",
    )

    assert np.allclose(plain, pat_zero)



@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy is unavailable")
def test_get_Ibcs_nA_zero_noise_is_identity() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)

    plain = get_Ibcs_nA(
        V_mV,
        0.2,
        0.1,
        0.19,
        0.002,
        nu_GHz=8.5,
        A_mV=0.15,
        backend="np",
    )
    no_noise = get_Ibcs_nA(
        V_mV,
        0.2,
        0.1,
        0.19,
        0.002,
        nu_GHz=8.5,
        A_mV=0.15,
        sigmaV_mV=0.0,
        backend="np",
    )

    assert np.allclose(plain, no_noise)



@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy is unavailable")
def test_get_Ibcs_nA_matches_explicit_pat_then_noise_staging() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    direct = get_Ibcs_nA(
        V_mV,
        0.2,
        0.1,
        0.19,
        0.002,
        nu_GHz=8.5,
        A_mV=0.15,
        sigmaV_mV=0.04,
        backend="np",
        kernel="conv",
    )

    V_support = make_bias_support_grid(V_mV, 0.04)
    base = get_Ibcs_nA(
        V_support,
        0.2,
        0.1,
        0.19,
        0.002,
        backend="np",
        kernel="conv",
    )
    pat = get_I_pat_nA(
        V_support,
        base,
        0.15,
        nu_GHz=8.5,
        n_max=PAT_N_MAX,
    )
    staged = apply_voltage_noise(
        V_support,
        pat,
        0.04,
        64,
        V_out_mV=V_mV,
    )

    assert np.allclose(direct, staged)
