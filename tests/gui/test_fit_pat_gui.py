from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from superconductivity.optimizers.pat import DEFAULT_PARAMETERS
from superconductivity.optimizers.pat.process import load_solution, save_solution

fit_pat_gui_module = importlib.import_module(
    "superconductivity.optimizers.pat.fit_pat_gui"
)
fit_pat_gui = fit_pat_gui_module.fit_pat_gui


def _make_solution(V: np.ndarray, I: np.ndarray) -> dict:
    params = []
    for idx, base in enumerate(DEFAULT_PARAMETERS):
        params.append(
            replace(
                base,
                guess=float(base.guess),
                fit_value=float(base.guess + idx + 0.1),
                fit_error=0.1,
            )
        )
    return {
        "V_mV": np.asarray(V, dtype=np.float64),
        "I_exp_nA": np.asarray(I, dtype=np.float64),
        "I_ini_nA": np.asarray(I, dtype=np.float64),
        "I_fit_nA": np.asarray(I + 0.2, dtype=np.float64),
        "params": tuple(params),
        "weights": None,
        "maxfev": 123,
    }


class _FakeProcess:
    def __init__(self, solution_dir: Path) -> None:
        self.solution_dir = solution_dir
        self.returncode: int | None = None
        self.pid = 4321

    def poll(self) -> int | None:
        return self.returncode


def test_solution_round_trip(tmp_path: Path) -> None:
    V = np.linspace(-1.0, 1.0, 7)
    I = np.sin(V)
    solution = _make_solution(V, I)

    save_solution(tmp_path, solution)
    loaded = load_solution(tmp_path)

    assert loaded is not None
    assert np.allclose(loaded["V_mV"], solution["V_mV"])
    assert np.allclose(loaded["I_fit_nA"], solution["I_fit_nA"])
    assert loaded["maxfev"] == solution["maxfev"]
    assert loaded["params"][0].name == solution["params"][0].name
    assert loaded["params"][0].fit_value == solution["params"][0].fit_value


def test_fit_pat_gui_returns_solution_after_normal_exit(monkeypatch) -> None:
    V = np.linspace(-1.0, 1.0, 11)
    I = np.cos(V)
    created: dict[str, _FakeProcess] = {}

    def fake_start(**kwargs) -> _FakeProcess:
        process = _FakeProcess(kwargs["solution_dir"])
        created["process"] = process
        return process

    def fake_wait(process: _FakeProcess, wait_interval: float) -> int:
        save_solution(process.solution_dir, _make_solution(V, I))
        process.returncode = 0
        return 0

    monkeypatch.setattr(fit_pat_gui_module, "_start_fit_pat_gui_process", fake_start)
    monkeypatch.setattr(fit_pat_gui_module, "_wait_for_worker", fake_wait)

    solution = fit_pat_gui(V_mV=V, I_nA=I)

    assert solution is not None
    assert np.allclose(solution["I_exp_nA"], I)
    assert created["process"].returncode == 0


def test_fit_pat_gui_interrupt_returns_last_solution(monkeypatch) -> None:
    V = np.linspace(-1.0, 1.0, 11)
    I = np.cos(V)

    def fake_start(**kwargs) -> _FakeProcess:
        return _FakeProcess(kwargs["solution_dir"])

    def fake_wait(process: _FakeProcess, wait_interval: float) -> int:
        save_solution(process.solution_dir, _make_solution(V, I))
        raise KeyboardInterrupt

    def fake_shutdown(process: _FakeProcess, **kwargs) -> int:
        process.returncode = 0
        return 0

    monkeypatch.setattr(fit_pat_gui_module, "_start_fit_pat_gui_process", fake_start)
    monkeypatch.setattr(fit_pat_gui_module, "_wait_for_worker", fake_wait)
    monkeypatch.setattr(
        fit_pat_gui_module,
        "_shutdown_fit_pat_gui_process",
        fake_shutdown,
    )

    solution = fit_pat_gui(V_mV=V, I_nA=I)

    assert solution is not None
    assert solution["params"][0].fit_error == 0.1


def test_fit_pat_gui_interrupt_without_fit_returns_none(monkeypatch) -> None:
    V = np.linspace(-1.0, 1.0, 11)
    I = np.cos(V)

    def fake_start(**kwargs) -> _FakeProcess:
        return _FakeProcess(kwargs["solution_dir"])

    def fake_wait(process: _FakeProcess, wait_interval: float) -> int:
        raise KeyboardInterrupt

    def fake_shutdown(process: _FakeProcess, **kwargs) -> int:
        process.returncode = 0
        return 0

    monkeypatch.setattr(fit_pat_gui_module, "_start_fit_pat_gui_process", fake_start)
    monkeypatch.setattr(fit_pat_gui_module, "_wait_for_worker", fake_wait)
    monkeypatch.setattr(
        fit_pat_gui_module,
        "_shutdown_fit_pat_gui_process",
        fake_shutdown,
    )

    assert fit_pat_gui(V_mV=V, I_nA=I) is None


def test_fit_pat_gui_nonzero_exit_without_solution_raises(monkeypatch) -> None:
    V = np.linspace(-1.0, 1.0, 11)
    I = np.cos(V)

    def fake_start(**kwargs) -> _FakeProcess:
        return _FakeProcess(kwargs["solution_dir"])

    def fake_wait(process: _FakeProcess, wait_interval: float) -> int:
        process.returncode = 2
        return 2

    monkeypatch.setattr(fit_pat_gui_module, "_start_fit_pat_gui_process", fake_start)
    monkeypatch.setattr(fit_pat_gui_module, "_wait_for_worker", fake_wait)

    with pytest.raises(RuntimeError, match="exited with status 2"):
        fit_pat_gui(V_mV=V, I_nA=I)
