from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from superconductivity.models.basics.noise import (
    evaluate_with_voltage_noise,
    make_bias_support_grid,
)
from superconductivity.models.bcs import bcs as bcs_module
from superconductivity.models.bcs import pat_kernel, sim_bcs
from superconductivity.models.bcs.backend import Nmax_
from superconductivity.models.bcs.bcs import get_Ibcs_nA
from superconductivity.utilities.meta import (
    AxisSpec,
    ParamSpec,
    axis,
    param,
)
from superconductivity.utilities.transport import TransportDatasetSpec

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

    def _evaluate_pat(V_support: np.ndarray) -> np.ndarray:
        base = get_Ibcs_nA(
            V_support,
            0.2,
            0.1,
            0.19,
            0.002,
            backend="np",
            kernel="conv",
        )
        return pat_kernel(
            V_support,
            base,
            0.15,
            nu_GHz=8.5,
            n_max=Nmax_,
        )

    staged = evaluate_with_voltage_noise(V_mV, _evaluate_pat, 0.04, 64)

    assert np.allclose(direct, staged)


def test_evaluate_with_voltage_noise_zero_noise_uses_requested_grid() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    seen: list[np.ndarray] = []

    def _evaluate(V_eval: np.ndarray) -> np.ndarray:
        seen.append(np.asarray(V_eval, dtype=np.float64))
        return 2.0 * np.asarray(V_eval, dtype=np.float64)

    current = evaluate_with_voltage_noise(V_mV, _evaluate, 0.0, 64)

    assert len(seen) == 1
    assert np.array_equal(seen[0], V_mV)
    assert np.array_equal(current, 2.0 * V_mV)


def test_evaluate_with_voltage_noise_nonzero_noise_uses_support_grid() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    seen: list[np.ndarray] = []

    def _evaluate(V_eval: np.ndarray) -> np.ndarray:
        seen.append(np.asarray(V_eval, dtype=np.float64))
        return np.asarray(V_eval, dtype=np.float64)

    current = evaluate_with_voltage_noise(V_mV, _evaluate, 0.04, 64)

    assert len(seen) == 1
    assert current.shape == V_mV.shape
    assert seen[0].shape != V_mV.shape
    assert np.array_equal(seen[0], make_bias_support_grid(V_mV, 0.04))
    assert np.allclose(current, V_mV)


def test_get_Ibcs_nA_scalar_inputs_return_1d_curve() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    current = get_Ibcs_nA(
        V_mV,
        0.2,
        0.1,
        0.19,
        0.002,
        backend="np",
        kernel="conv",
    )
    assert current.shape == (V_mV.size,)


def test_get_Ibcs_nA_multi_parameter_cartesian_sweep_shape() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    current = get_Ibcs_nA(
        V_mV,
        GN_G0=np.array([0.2, 0.3], dtype=np.float64),
        T_K=np.array([0.05, 0.1, 0.2], dtype=np.float64),
        Delta_meV=0.19,
        gamma_meV=0.002,
        A_mV=np.array([0.0, 0.1], dtype=np.float64),
        nu_GHz=8.5,
        sigmaV_mV=0.0,
        backend="np",
        kernel="conv",
    )
    assert current.shape == (2, 3, 2, V_mV.size)


def test_get_Ibcs_nA_gn_sweep_reuses_base_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    call_count = {"base": 0}

    def _counting_base(
        V_eval: np.ndarray,
        E_eval: np.ndarray,
        T1: float,
        T2: float,
        Delta1: float,
        Delta2: float,
        gamma1: float,
        gamma2: float,
    ) -> np.ndarray:
        call_count["base"] += 1
        return np.asarray(V_eval, dtype=np.float64)

    monkeypatch.setattr(
        bcs_module,
        "_resolve_base_function",
        lambda *, kernel, backend: _counting_base,
    )

    current = bcs_module.get_Ibcs_nA(
        V_mV,
        GN_G0=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        T_K=0.1,
        Delta_meV=0.19,
        gamma_meV=0.002,
        A_mV=0.0,
        sigmaV_mV=0.0,
        backend="np",
        kernel="conv",
    )

    assert current.shape == (3, V_mV.size)
    assert call_count["base"] == 1


def test_sim_bcs_returns_transport_dataset_scalar() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    ds = sim_bcs(
        V_mV=axis("V_mV", values=V_mV, order=0),
        GN_G0=param("GN_G0", 0.2),
        T_K=param("T_K", 0.1),
        Delta_meV=param("Delta_meV", 0.19),
        gamma_meV=param("gamma_meV", 0.002),
        backend="np",
        kernel="conv",
    )
    assert isinstance(ds, TransportDatasetSpec)
    assert ds.I_nA.values.shape == (V_mV.size,)
    assert ds.V_mV.values.shape == (V_mV.size,)
    assert ds.dG_uS.values.shape == (V_mV.size,)
    assert ds.dR_MOhm.values.shape == (V_mV.size,)
    assert ds.eV_Delta.values.shape == (V_mV.size,)
    assert ds.eI_DeltaG0.values.shape == (V_mV.size,)
    assert ds.eI_DeltaGN.values.shape == (V_mV.size,)
    assert isinstance(ds.eV_Delta, AxisSpec)
    assert isinstance(ds.Tc_K, ParamSpec)
    assert isinstance(ds.T_Tc, ParamSpec)
    assert isinstance(ds.DeltaT_meV, ParamSpec)
    assert isinstance(ds.DeltaT_Delta, ParamSpec)
    assert isinstance(ds.gamma_Delta, ParamSpec)
    axis_labels = {entry.code_label for entry in ds.axes}
    param_labels = {entry.code_label for entry in ds.params}
    data_labels = {entry.code_label for entry in ds.data}
    assert axis_labels == {"V_mV"}
    assert {
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "nu_GHz",
        "A_mV",
        "sigmaV_mV",
    } <= param_labels
    assert data_labels == {"I_nA"}
    assert "G_muS" not in data_labels
    assert "dG_uS" not in data_labels
    assert "eI_DeltaG0" not in data_labels
    assert "eI_DeltaGN" not in data_labels


def test_sim_bcs_returns_transport_dataset_with_sweeps() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    ds = sim_bcs(
        V_mV=axis("V_mV", values=V_mV, order=3),
        GN_G0=param("GN_G0", np.array([0.2, 0.3], dtype=np.float64)),
        T_K=param("T_K", np.array([0.05, 0.1], dtype=np.float64)),
        Delta_meV=param("Delta_meV", 0.19),
        gamma_meV=param("gamma_meV", 0.002),
        A_mV=param("A_mV", np.array([0.0, 0.1], dtype=np.float64)),
        nu_GHz=param("nu_GHz", 8.5),
        sigmaV_mV=param("sigmaV_mV", 0.0),
        backend="np",
        kernel="conv",
    )
    assert isinstance(ds, TransportDatasetSpec)
    assert ds.I_nA.values.shape == (2, 2, 2, V_mV.size)
    assert ds.dG_GN.values.shape == (2, 2, 2, V_mV.size)
    assert isinstance(ds.eA_hnu, AxisSpec)
    assert ds.eA_hnu.values.shape == (2,)
    assert ds.eA_hnu.order == 2
    assert isinstance(ds.hnu_Delta, ParamSpec)
    axis_by_label = {entry.code_label: entry for entry in ds.axes}
    assert axis_by_label["GN_G0"].order == 0
    assert axis_by_label["T_K"].order == 1
    assert axis_by_label["A_mV"].order == 2
    assert axis_by_label["V_mV"].order == 3
    assert axis_by_label["GN_G0"].values.shape == (2,)
    assert axis_by_label["T_K"].values.shape == (2,)
    assert axis_by_label["A_mV"].values.shape == (2,)
    assert axis_by_label["V_mV"].values.shape == (V_mV.size,)
    param_labels = {entry.code_label for entry in ds.params}
    assert {"Delta_meV", "gamma_meV", "nu_GHz", "sigmaV_mV"} <= param_labels
    assert "GN_G0" not in param_labels
    assert "T_K" not in param_labels
    assert "A_mV" not in param_labels
    data_labels = {entry.code_label for entry in ds.data}
    assert data_labels == {"I_nA"}


def test_sim_bcs_accepts_axis_as_sweep_parameter() -> None:
    V = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    ds = sim_bcs(
        V_mV=axis("V_mV", values=V, order=1),
        GN_G0=axis("GN_G0", values=[0.2, 0.3, 0.4], order=0),
        T_K=param("T_K", 0.1),
        Delta_meV=param("Delta_meV", 0.19),
        gamma_meV=param("gamma_meV", 0.002),
        backend="np",
        kernel="conv",
    )
    assert ds.I_nA.values.shape == (3, V.size)
