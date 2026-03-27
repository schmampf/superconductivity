from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pytest

from superconductivity.optimizers.pat import (
    BatchParameterSpec,
    DEFAULT_PARAMETERS,
    fit_pats,
)

fit_pats_module = importlib.import_module("superconductivity.optimizers.pat.fit_pats")


def _make_trace_solution(
    V: np.ndarray,
    I: np.ndarray,
    parameters,
    *,
    value_offset: float,
    trace_index: int,
) -> dict:
    params = []
    for idx, base in enumerate(parameters):
        params.append(
            replace(
                base,
                value=float(base.guess + value_offset + trace_index + idx),
                error=float(0.01 * (idx + 1)),
            )
        )
    return {
        "V_mV": np.asarray(V, dtype=np.float64),
        "I_exp_nA": np.asarray(I, dtype=np.float64),
        "I_ini_nA": np.asarray(I + 0.1, dtype=np.float64),
        "I_fit_nA": np.asarray(I + 0.2, dtype=np.float64),
        "params": tuple(params),
        "weights": None,
        "maxfev": 200,
    }


def test_fit_pats_broadcasts_scalar_guess(monkeypatch) -> None:
    V = np.linspace(-1.0, 1.0, 9)
    traces = np.vstack((np.sin(V), np.cos(V)))
    calls: list[dict[str, object]] = []

    def fake_fit_pat(V_mV, I_nA, **kwargs):
        calls.append(kwargs)
        return _make_trace_solution(
            V_mV,
            I_nA,
            kwargs["parameters"],
            value_offset=1.0,
            trace_index=len(calls) - 1,
        )

    monkeypatch.setattr(fit_pats_module, "fit_pat", fake_fit_pat)

    parameters = [
        BatchParameterSpec(
            name=param.name,
            label=param.label,
            lower=param.lower,
            upper=param.upper,
            guess=param.guess,
            fixed=param.fixed,
        )
        for param in DEFAULT_PARAMETERS
    ]

    solution = fit_pats(V, traces, parameters=parameters, maxfev=200)

    assert len(calls) == 2
    assert all(kwargs["maxfev"] == 200 for kwargs in calls)
    assert np.allclose(solution["params"][0].guess, [parameters[0].guess] * 2)
    assert solution["params"][0].value is not None
    assert solution["params"][0].value.shape == (2,)
    assert solution["I_fit_nA"].shape == traces.shape


def test_fit_pats_accepts_per_trace_guess_and_weight_broadcast(monkeypatch) -> None:
    V = np.linspace(-1.0, 1.0, 7)
    traces = np.vstack((np.sin(V), np.cos(V), np.tanh(V)))
    weights = np.linspace(1.0, 2.0, V.size)
    recorded_guesses: list[float] = []
    recorded_weights: list[np.ndarray] = []

    def fake_fit_pat(V_mV, I_nA, **kwargs):
        recorded_guesses.append(kwargs["parameters"][0].guess)
        recorded_weights.append(kwargs["weights"].copy())
        return _make_trace_solution(
            V_mV,
            I_nA,
            kwargs["parameters"],
            value_offset=2.0,
            trace_index=len(recorded_guesses) - 1,
        )

    monkeypatch.setattr(fit_pats_module, "fit_pat", fake_fit_pat)

    parameters = [
        BatchParameterSpec(
            name=param.name,
            label=param.label,
            lower=param.lower,
            upper=param.upper,
            guess=[0.1, 0.2, 0.3] if idx == 0 else param.guess,
            fixed=param.fixed,
        )
        for idx, param in enumerate(DEFAULT_PARAMETERS)
    ]

    solution = fit_pats(V, traces, parameters=parameters, weights=weights)

    assert recorded_guesses == [0.1, 0.2, 0.3]
    assert all(np.allclose(weight, weights) for weight in recorded_weights)
    assert solution["weights"] is not None
    assert solution["weights"].shape == traces.shape


def test_fit_pats_rejects_wrong_guess_length() -> None:
    V = np.linspace(-1.0, 1.0, 7)
    traces = np.vstack((np.sin(V), np.cos(V)))
    parameters = [
        BatchParameterSpec(
            name=param.name,
            label=param.label,
            lower=param.lower,
            upper=param.upper,
            guess=[0.1, 0.2, 0.3] if idx == 0 else param.guess,
            fixed=param.fixed,
        )
        for idx, param in enumerate(DEFAULT_PARAMETERS)
    ]

    with pytest.raises(ValueError, match="guess must be scalar or length 2"):
        fit_pats(V, traces, parameters=parameters)


def test_fit_pats_updates_progress_bar(monkeypatch) -> None:
    V = np.linspace(-1.0, 1.0, 7)
    traces = np.vstack((np.sin(V), np.cos(V), np.tanh(V)))
    calls: list[int] = []
    closed = {"value": False}

    class FakeProgress:
        def update(self, n: int = 1) -> None:
            calls.append(n)

        def close(self) -> None:
            closed["value"] = True

    def fake_fit_pat(V_mV, I_nA, **kwargs):
        return _make_trace_solution(
            V_mV,
            I_nA,
            kwargs["parameters"],
            value_offset=3.0,
            trace_index=len(calls),
        )

    monkeypatch.setattr(fit_pats_module, "fit_pat", fake_fit_pat)
    monkeypatch.setattr(
        fit_pats_module,
        "_make_progress",
        lambda total, show_progress: FakeProgress(),
    )

    solution = fit_pats(V, traces, show_progress=True)

    assert solution["I_fit_nA"].shape == traces.shape
    assert calls == [1, 1, 1]
    assert closed["value"]


def test_fit_pats_forwards_model_to_single_trace_fitter(monkeypatch) -> None:
    V = np.linspace(-1.0, 1.0, 7)
    traces = np.vstack((np.sin(V), np.cos(V)))
    models: list[str] = []

    def fake_fit_pat(V_mV, I_nA, **kwargs):
        models.append(kwargs["model"])
        return _make_trace_solution(
            V_mV,
            I_nA,
            kwargs["parameters"],
            value_offset=4.0,
            trace_index=len(models) - 1,
        )

    monkeypatch.setattr(fit_pats_module, "fit_pat", fake_fit_pat)

    solution = fit_pats(V, traces, model="conv_pat")

    assert solution["I_fit_nA"].shape == traces.shape
    assert models == ["conv_pat", "conv_pat"]
