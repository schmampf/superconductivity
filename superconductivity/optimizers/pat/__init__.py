from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np

from .fit_pat import DEFAULT_PARAMETERS, PARAMETER_NAMES, ParameterSpec, SolutionDict
from .fit_pats import BatchParameterSpec, BatchSolutionDict


def fit_pat(*args, **kwargs):
    from .fit_pat import fit_pat as _fit_pat

    return _fit_pat(*args, **kwargs)


def fit_pats(
    V_mV,
    I_nA,
    *,
    parameters: Optional[Sequence[ParameterSpec | BatchParameterSpec]] = None,
    weights=None,
    maxfev: Optional[int] = None,
    E_mV=None,
    model: str = "pat",
    show_progress: bool = False,
) -> BatchSolutionDict:
    from .fit_pats import fit_pats as _fit_pats

    return _fit_pats(
        V_mV,
        I_nA,
        parameters=parameters,
        weights=weights,
        maxfev=maxfev,
        E_mV=E_mV,
        model=model,
        show_progress=show_progress,
    )


def fit_pat_gui(
    V_mV: np.ndarray,
    I_nA: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    maxfev: Optional[int] = None,
    model: str = "pat",
    python_executable: Optional[Path] = None,
    wait_interval: float = 1.0,
) -> Optional[SolutionDict]:
    from .fit_pat_gui import fit_pat_gui as _fit_pat_gui

    return _fit_pat_gui(
        V_mV,
        I_nA,
        weights=weights,
        maxfev=maxfev,
        model=model,
        python_executable=python_executable,
        wait_interval=wait_interval,
    )

__all__ = [
    "fit_pat",
    "fit_pats",
    "DEFAULT_PARAMETERS",
    "PARAMETER_NAMES",
    "ParameterSpec",
    "BatchParameterSpec",
    "SolutionDict",
    "BatchSolutionDict",
    "fit_pat_gui",
]
