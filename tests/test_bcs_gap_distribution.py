import numpy as np

from superconductivity.optimizers.bcs.gap_distribution import (
    apply_gap_distribution,
)


def test_gap_distribution_zero_limit_is_identity() -> None:
    curve = apply_gap_distribution(
        lambda delta: np.array([delta, delta**2], dtype=np.float64),
        Delta_meV=0.2,
        sigma_Delta_meV=0.0,
        order=11,
    )

    assert np.allclose(curve, np.array([0.2, 0.04], dtype=np.float64))


def test_gap_distribution_is_deterministic() -> None:
    out_1 = apply_gap_distribution(
        lambda delta: np.array([np.sin(delta), np.cos(delta)], dtype=np.float64),
        Delta_meV=0.19,
        sigma_Delta_meV=0.01,
        order=11,
    )
    out_2 = apply_gap_distribution(
        lambda delta: np.array([np.sin(delta), np.cos(delta)], dtype=np.float64),
        Delta_meV=0.19,
        sigma_Delta_meV=0.01,
        order=11,
    )

    assert np.allclose(out_1, out_2)


def test_gap_distribution_smooths_convex_gap_dependence() -> None:
    out = apply_gap_distribution(
        lambda delta: np.array([delta**2], dtype=np.float64),
        Delta_meV=0.2,
        sigma_Delta_meV=0.015,
        order=11,
    )

    assert out[0] > 0.2**2


def test_gap_distribution_stays_finite_when_truncated() -> None:
    out = apply_gap_distribution(
        lambda delta: np.array([delta, np.exp(-delta)], dtype=np.float64),
        Delta_meV=0.01,
        sigma_Delta_meV=0.03,
        order=11,
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
