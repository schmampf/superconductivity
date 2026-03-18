from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .fit_pat import SolutionDict
from .process import (
    load_solution,
    shutdown_fit_pat_gui_process,
    start_fit_pat_gui_process,
)


def _wait_for_worker(process: subprocess.Popen[bytes], wait_interval: float) -> int:
    while True:
        try:
            return int(process.wait(timeout=wait_interval))
        except subprocess.TimeoutExpired:
            continue


def fit_pat_gui(
    V_mV: np.ndarray,
    I_nA: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    maxfev: Optional[int] = None,
    python_executable: Optional[Path] = None,
    wait_interval: float = 1.0,
) -> Optional[SolutionDict]:
    """Run the PAT GUI in a subprocess until the notebook cell is interrupted.

    Parameters
    ----------
    V_mV : np.ndarray
        Voltage axis in millivolts.
    I_nA : np.ndarray
        Current trace in nanoamperes.
    weights : np.ndarray | None, optional
        Optional optimizer weights.
    maxfev : int | None, optional
        Maximum optimizer evaluations.
    python_executable : Path | None, optional
        Interpreter used for the child process. Defaults to the shared project
        virtual environment.
    wait_interval : float, optional
        Poll interval, in seconds, while waiting for the worker to exit.

    Returns
    -------
    SolutionDict | None
        The last successfully persisted PAT solution, or ``None`` if the GUI
        was closed before any fit completed.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="fit_pat_gui_"))
    V_path = temp_dir / "V_mV.npy"
    I_path = temp_dir / "I_nA.npy"
    np.save(V_path, np.asarray(V_mV, dtype=np.float64))
    np.save(I_path, np.asarray(I_nA, dtype=np.float64))

    weights_path = None
    if weights is not None:
        weights_path = temp_dir / "weights.npy"
        np.save(weights_path, np.asarray(weights, dtype=np.float64))

    try:
        process = start_fit_pat_gui_process(
            voltage_path=V_path,
            current_path=I_path,
            solution_dir=temp_dir,
            weights_path=weights_path,
            maxfev=maxfev,
            python_executable=python_executable,
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    interrupted = False
    try:
        _wait_for_worker(process, wait_interval)
    except KeyboardInterrupt:
        interrupted = True
        shutdown_fit_pat_gui_process(process)

    solution = load_solution(temp_dir)
    returncode = process.poll()

    try:
        if returncode not in (None, 0) and solution is None:
            raise RuntimeError(
                "PAT GUI worker exited with status "
                f"{returncode} before producing a solution."
            )
        if interrupted and process.returncode is None:
            raise RuntimeError("PAT GUI worker did not shut down cleanly.")
        return solution
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
