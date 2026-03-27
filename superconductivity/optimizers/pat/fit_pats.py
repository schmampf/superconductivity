"""Batch PAT optimizer built on top of :func:`fit_pat`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

from ...utilities.safety import require_all_finite, require_min_size, to_1d_float64
from ...utilities.types import NDArray64
from .fit_pat import DEFAULT_PARAMETERS, ParameterSpec, fit_pat


GuessLike = float | Sequence[float] | NDArray64
FixedLike = bool | Sequence[bool] | NDArray[np.bool_]


@dataclass
class BatchParameterSpec:
    """Batch PAT parameter specification.

    Parameters
    ----------
    name : str
        Stable parameter identifier.
    label : str
        Display label used in user interfaces.
    lower : float
        Lower optimizer bound shared across traces.
    upper : float
        Upper optimizer bound shared across traces.
    guess : float | Sequence[float] | NDArray64
        Initial guess. Scalars are broadcast to every trace.
    fixed : bool | Sequence[bool] | NDArray[np.bool_], optional
        Fixed-parameter mask. Scalars are broadcast to every trace.
    value : NDArray64 | None, optional
        Per-trace fitted values returned by :func:`fit_pats`.
    error : NDArray64 | None, optional
        Per-trace fit errors returned by :func:`fit_pats`.
    """

    name: str
    label: str
    lower: float
    upper: float
    guess: GuessLike
    fixed: FixedLike = False
    value: Optional[NDArray64] = None
    error: Optional[NDArray64] = None


class BatchSolutionDict(TypedDict):
    V_mV: NDArray64
    I_exp_nA: NDArray64
    I_ini_nA: NDArray64
    I_fit_nA: NDArray64
    params: Sequence[BatchParameterSpec]
    weights: Optional[NDArray64]
    maxfev: Optional[int]


class _NullProgress:
    def update(self, n: int = 1) -> None:
        return

    def close(self) -> None:
        return


def _prepare_trace_matrix(I_nA: NDArray, points: int) -> NDArray64:
    data = np.asarray(I_nA, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim != 2:
        raise ValueError("I_nA must be 1D or 2D.")

    if data.shape[1] != points:
        raise ValueError("Each IV trace must match the length of V_mV.")

    require_all_finite(data, "I_nA")
    require_min_size(data, 1, "I_nA")
    return data


def _make_progress(total: int, *, show_progress: bool) -> object:
    if not show_progress:
        return _NullProgress()

    from tqdm.auto import tqdm

    return tqdm(total=total, desc="PAT fits")


def _normalize_guess(value: GuessLike, *, n_traces: int, name: str) -> NDArray64:
    if np.isscalar(value):
        return np.full((n_traces,), float(value), dtype=np.float64)

    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n_traces,):
        raise ValueError(f"{name}.guess must be scalar or length {n_traces}.")
    return arr


def _normalize_fixed(value: FixedLike, *, n_traces: int, name: str) -> NDArray[np.bool_]:
    if np.isscalar(value):
        return np.full((n_traces,), bool(value), dtype=bool)

    arr = np.asarray(value, dtype=bool)
    if arr.shape != (n_traces,):
        raise ValueError(f"{name}.fixed must be scalar or length {n_traces}.")
    return arr


def _prepare_weight_matrix(
    weights: Optional[NDArray64],
    *,
    n_traces: int,
    points: int,
) -> Optional[NDArray64]:
    if weights is None:
        return None

    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape == (points,):
        return np.broadcast_to(arr, (n_traces, points)).copy()
    if arr.shape == (n_traces, points):
        return arr.copy()

    raise ValueError(
        "weights must be None, length len(V_mV), or shape (n_traces, len(V_mV))."
    )


def _default_batch_parameters(n_traces: int) -> list[BatchParameterSpec]:
    return [
        BatchParameterSpec(
            name=param.name,
            label=param.label,
            lower=param.lower,
            upper=param.upper,
            guess=np.full((n_traces,), param.guess, dtype=np.float64),
            fixed=np.full((n_traces,), param.fixed, dtype=bool),
        )
        for param in DEFAULT_PARAMETERS
    ]


def _clone_batch_parameters(
    parameters: Optional[Sequence[ParameterSpec | BatchParameterSpec]],
    *,
    n_traces: int,
) -> list[BatchParameterSpec]:
    if parameters is None:
        return _default_batch_parameters(n_traces)

    if len(parameters) != len(DEFAULT_PARAMETERS):
        raise ValueError(
            "Parameter list must contain six entries in the expected order."
        )

    cloned: list[BatchParameterSpec] = []
    for expected, provided in zip(DEFAULT_PARAMETERS, parameters):
        if expected.name != provided.name:
            raise ValueError(
                f"Parameter '{provided.name}' does not match expected "
                f"'{expected.name}'."
            )

        guess = _normalize_guess(
            provided.guess,
            n_traces=n_traces,
            name=provided.name,
        )
        fixed = _normalize_fixed(
            provided.fixed,
            n_traces=n_traces,
            name=provided.name,
        )
        cloned.append(
            BatchParameterSpec(
                name=provided.name,
                label=provided.label,
                lower=provided.lower,
                upper=provided.upper,
                guess=guess.copy(),
                fixed=fixed.copy(),
            )
        )
    return cloned


def fit_pats(
    V_mV: NDArray64,
    I_nA: NDArray64,
    *,
    parameters: Optional[Sequence[ParameterSpec | BatchParameterSpec]] = None,
    weights: Optional[NDArray64] = None,
    maxfev: Optional[int] = None,
    E_mV: Optional[NDArray64] = None,
    model: str = "pat",
    show_progress: bool = False,
) -> BatchSolutionDict:
    """Fit one or many PAT IV traces.

    Parameters
    ----------
    V_mV : NDArray64
        Shared voltage axis in millivolts.
    I_nA : NDArray64
        Single IV trace with shape ``(n_points,)`` or batch of traces with shape
        ``(n_traces, n_points)``.
    parameters : Sequence[ParameterSpec | BatchParameterSpec] | None, optional
        Parameter definitions. Scalars in ``guess`` and ``fixed`` are broadcast
        across all traces; sequences must have length ``n_traces``.
    weights : NDArray64 | None, optional
        Optional fit weights. Accepted shapes are ``(n_points,)`` for broadcast
        weights or ``(n_traces, n_points)`` for per-trace weights.
    maxfev : int | None, optional
        Maximum optimizer evaluations passed through to each single-trace fit.
    E_mV : NDArray64 | None, optional
        Energy grid forwarded to the model evaluation.
    model : str, optional
        Model identifier forwarded to :func:`fit_pat`. Use ``"pat"`` for the
        existing integral-based PAT model or ``"conv_pat"`` for the
        convolution-based alternative.
    show_progress : bool, optional
        Whether to display a ``tqdm`` progress bar while fitting traces.

    Returns
    -------
    BatchSolutionDict
        Batch fit result with stacked current arrays and per-trace parameter
        values/errors.
    """
    V = to_1d_float64(V_mV, "V_mV")
    require_all_finite(V, "V_mV")
    require_min_size(V, 3, "V_mV")

    traces = _prepare_trace_matrix(I_nA, points=V.size)
    n_traces, points = traces.shape
    weight_matrix = _prepare_weight_matrix(
        weights,
        n_traces=n_traces,
        points=points,
    )
    batch_parameters = _clone_batch_parameters(parameters, n_traces=n_traces)

    initial = np.empty_like(traces)
    fitted = np.empty_like(traces)
    values = np.empty((len(batch_parameters), n_traces), dtype=np.float64)
    errors = np.empty((len(batch_parameters), n_traces), dtype=np.float64)
    progress = _make_progress(n_traces, show_progress=show_progress)

    try:
        for trace_idx in range(n_traces):
            trace_parameters = [
                ParameterSpec(
                    name=param.name,
                    label=param.label,
                    lower=param.lower,
                    upper=param.upper,
                    guess=float(np.asarray(param.guess)[trace_idx]),
                    fixed=bool(np.asarray(param.fixed)[trace_idx]),
                )
                for param in batch_parameters
            ]
            trace_weights = None if weight_matrix is None else weight_matrix[trace_idx]
            solution = fit_pat(
                V,
                traces[trace_idx],
                parameters=trace_parameters,
                weights=trace_weights,
                maxfev=maxfev,
                E_mV=E_mV,
                model=model,
            )
            initial[trace_idx] = solution["I_ini_nA"]
            fitted[trace_idx] = solution["I_fit_nA"]
            for param_idx, param in enumerate(solution["params"]):
                values[param_idx, trace_idx] = float(param.value)
                errors[param_idx, trace_idx] = float(param.error)
            progress.update(1)
    finally:
        progress.close()

    result_parameters = [
        BatchParameterSpec(
            name=param.name,
            label=param.label,
            lower=param.lower,
            upper=param.upper,
            guess=np.asarray(param.guess, dtype=np.float64).copy(),
            fixed=np.asarray(param.fixed, dtype=bool).copy(),
            value=values[param_idx].copy(),
            error=errors[param_idx].copy(),
        )
        for param_idx, param in enumerate(batch_parameters)
    ]

    return {
        "V_mV": V,
        "I_exp_nA": traces.copy(),
        "I_ini_nA": initial,
        "I_fit_nA": fitted,
        "params": tuple(result_parameters),
        "weights": None if weight_matrix is None else weight_matrix,
        "maxfev": maxfev,
    }
