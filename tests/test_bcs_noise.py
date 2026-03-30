import numpy as np

from superconductivity.optimizers.bcs.noise import (
    apply_voltage_noise,
    make_bias_support_grid,
)


def test_voltage_noise_zero_limit_is_identity() -> None:
    V_mV = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    I_nA = np.sin(V_mV)

    out = apply_voltage_noise(
        V_mV,
        I_nA,
        sigma_V_mV=0.0,
        order=32,
    )

    assert np.allclose(out, I_nA)


def test_voltage_noise_is_deterministic() -> None:
    V_mV = np.linspace(-1.0, 1.0, 31, dtype=np.float64)
    I_nA = V_mV + 0.1 * V_mV**3

    out_1 = apply_voltage_noise(
        V_mV,
        I_nA,
        sigma_V_mV=0.04,
        order=32,
    )
    out_2 = apply_voltage_noise(
        V_mV,
        I_nA,
        sigma_V_mV=0.04,
        order=32,
    )

    assert np.allclose(out_1, out_2)


def test_voltage_noise_stays_finite_for_nonzero_sigma() -> None:
    V_mV = np.linspace(-1.5, 1.5, 41, dtype=np.float64)
    I_nA = np.tanh(3.0 * V_mV)

    out = apply_voltage_noise(
        V_mV,
        I_nA,
        sigma_V_mV=0.08,
        order=64,
    )

    assert out.shape == V_mV.shape
    assert np.all(np.isfinite(out))


def test_voltage_noise_smooths_sharp_features() -> None:
    V_mV = np.linspace(-1.0, 1.0, 401, dtype=np.float64)
    I_nA = np.tanh(18.0 * V_mV)

    out = apply_voltage_noise(
        V_mV,
        I_nA,
        sigma_V_mV=0.08,
        order=64,
    )

    assert np.max(np.abs(np.diff(out))) < np.max(np.abs(np.diff(I_nA)))


def test_bias_support_grid_extends_voltage_range() -> None:
    V_mV = np.linspace(-1.0, 1.0, 21, dtype=np.float64)

    out = make_bias_support_grid(
        V_mV,
        sigma_V_mV=0.1,
    )

    assert out[0] < V_mV[0]
    assert out[-1] > V_mV[-1]
    assert np.isclose(
        np.median(np.diff(out)),
        np.median(np.diff(V_mV)),
    )
    assert out.size > V_mV.size
