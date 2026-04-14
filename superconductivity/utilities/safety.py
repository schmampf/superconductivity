from typing import Sequence

import numpy as np


def to_1d_float64(x: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Convert array-like input to a 1D float64 NumPy array.

    Parameters
    ----------
    x : Sequence[float] | np.ndarray
        Input array-like object.
    name : str
        Parameter name used in the error message.

    Returns
    -------
    np.ndarray
        Flattened ``float64`` array.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def require_same_shape(
    a: np.ndarray,
    b: np.ndarray,
    name_a: str = "a",
    name_b: str = "b",
) -> None:
    """Require two arrays to have identical shapes.

    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.
    name_a : str, default="a"
        Name of the first array used in the error message.
    name_b : str, default="b"
        Name of the second array used in the error message.

    Raises
    ------
    ValueError
        If ``a.shape != b.shape``.
    """
    if np.shape(a) != np.shape(b):
        raise ValueError(f"{name_a} and {name_b} must have the same shape.")


def require_min_size(
    arr: np.ndarray,
    min_size: int,
    name: str = "arr",
) -> None:
    """Require an array-like object to have at least ``min_size`` entries.

    Parameters
    ----------
    arr : np.ndarray
        Input array-like object.
    min_size : int
        Minimum allowed size.
    name : str, default="arr"
        Name used in the error message.

    Raises
    ------
    ValueError
        If ``arr.size < min_size``.
    """
    if min_size < 0:
        raise ValueError("min_size must be >= 0.")
    if np.size(arr) < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values.")


def require_all_finite(
    arr: np.ndarray,
    name: str = "arr",
) -> None:
    """Require all values in an array-like object to be finite.

    Parameters
    ----------
    arr : np.ndarray
        Input array-like object to validate.
    name : str, default="arr"
        Name used in the error message.

    Raises
    ------
    ValueError
        If any value in ``arr`` is not finite.
    """
    if not np.all(np.isfinite(np.asarray(arr))):
        raise ValueError(f"{name} must contain only finite values.")



def is_ragged_sequence(value: object) -> bool:
    """Return True for Python list/tuple ragged inputs."""
    return isinstance(value, (list, tuple))


def normalize_axis(axis: int, ndim: int) -> int:
    """Validate and normalize a possibly-negative axis index."""
    axis_int = int(axis)
    if axis_int < -ndim or axis_int >= ndim:
        raise ValueError(
            f"axis {axis_int} is out of bounds for array of ndim {ndim}."
        )
    return axis_int % ndim


def is_pair_input(value: object) -> bool:
    """Return True for a 2-tuple of non-ragged pair inputs."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and not any(isinstance(item, (list, tuple)) for item in value)
    )
