from typing import Sequence

import numpy as np

from ..safety import require_min_size, require_same_shape, to_1d_float64
from ..types import NDArray64


# performance binning
def bin_y_over_x(
    x: NDArray64,
    y: NDArray64,
    x_bins: NDArray64,
) -> NDArray64:
    """Bin `y` over `x` onto a fixed `x_bins` grid using histogram
    accumulation.

    The function treats `x_bins` as bin centers (uniform spacing assumed) and
    constructs corresponding bin edges. For each bin it computes the mean of
    `y` values whose `x` values fall into that bin. Empty bins are returned as
    NaN.

    Parameters
    ----------
    x:
        1D array of x-values for each sample.
    y:
        1D array of y-values for each sample. Must have the same shape as `x`.
    x_bins:
        1D array of bin centers defining the output grid. Assumed to be
        uniformly spaced.

    Returns
    -------
    y_binned:
        1D array of shape `x_bins.shape` containing the per-bin mean of `y`.
        Bins with zero samples are NaN.

    Notes
    -----
    Internally, counts and weighted sums are computed via `numpy.histogram`.
    The rightmost bin edge is extrapolated to preserve the number of bins.
    """
    x_arr = to_1d_float64(x, "x")
    y_arr = to_1d_float64(y, "y")
    x_bins_arr = to_1d_float64(x_bins, "x_bins")

    require_same_shape(x_arr, y_arr, name_a="x", name_b="y")
    require_min_size(x_bins_arr, 2, name="x_bins")

    x_nu = np.append(x_bins_arr, 2 * x_bins_arr[-1] - x_bins_arr[-2])
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2
    count, _ = np.histogram(x_arr, bins=x_nu, weights=None)
    count = np.where(count == 0, np.nan, count)
    summe, _ = np.histogram(x_arr, bins=x_nu, weights=y_arr)
    return summe / count


def upsample(
    x: NDArray64,
    y: NDArray64,
    factor: int = 100,
    method: str = "linear",
) -> tuple[NDArray64, NDArray64]:
    """Upsample paired 1D arrays ``(x, y)`` by an integer factor.

    Parameters
    ----------
    x:
        1D array of x-values.
    y:
        1D array of y-values. Must have the same shape as `x`.
    factor:
        Integer oversampling factor. If `factor <= 1`, the input is returned
        unchanged.
    method:
        Interpolation method. Supported values are `"linear"` and
        `"nearest"`.
    Returns
    -------
    x_new:
        Oversampled x-array with shape `(len(x) * factor,)`.
    y_new:
        Oversampled y-array with shape `(len(y) * factor,)`.

    Raises
    ------
    ValueError
        If `x` and `y` do not have the same shape, or if `method` is
        unsupported.

    Notes
    -----
    This routine interpolates both `x` and `y` on a uniform sample-index
    grid, matching the legacy behavior where output length is
    ``len(x) * factor``.
    """
    x_arr = to_1d_float64(x, "x")
    y_arr = to_1d_float64(y, "y")

    require_same_shape(x_arr, y_arr, name_a="x", name_b="y")

    if factor <= 1:
        return x_arr, y_arr

    n = x_arr.size
    if n < 2:
        return x_arr, y_arr

    t = np.arange(n, dtype=np.float64)
    t_new = np.linspace(
        0.0,
        float(n - 1),
        n * factor,
        dtype=np.float64,
    )

    method_key = method.strip().lower()
    if method_key == "linear":
        x_new = np.interp(t_new, t, x_arr)
        y_new = np.interp(t_new, t, y_arr)
        return x_new, y_new

    if method_key == "nearest":
        idx = np.rint(t_new).astype(np.int64)
        idx = np.clip(idx, 0, n - 1)
        return x_arr[idx], y_arr[idx]

    raise ValueError(
        "Unsupported method. Use 'linear' or 'nearest'.",
    )


def fill_nans(
    y: NDArray64,
    x: NDArray64 | None = None,
    method: str = "linear",
) -> NDArray64:
    """Fill NaN entries in a 1D array.

    Parameters
    ----------
    y:
        1D values that may include NaNs.
    x:
        Optional 1D reference axis. If omitted, sample index is used.
    method:
        Fill method: `"linear"` or `"nearest"`.

    Returns
    -------
    y_filled:
        1D array with NaNs replaced where possible. If no finite values exist,
        returns ``y`` unchanged. For `"linear"`, at least two finite values are
        required.
    """
    y_arr = to_1d_float64(y, "y")
    x_arr = (
        np.arange(y_arr.size, dtype=np.float64)
        if x is None
        else to_1d_float64(x, "x")
    )
    require_same_shape(x_arr, y_arr, name_a="x", name_b="y")

    finite = np.isfinite(y_arr)
    n_finite = int(np.sum(finite))
    if n_finite == y_arr.size or n_finite == 0:
        return y_arr

    y_out = y_arr.copy()
    x_f = x_arr[finite]
    y_f = y_arr[finite]
    order = np.argsort(x_f)
    x_f = x_f[order]
    y_f = y_f[order]

    method_key = method.strip().lower()
    if method_key == "linear":
        if n_finite < 2:
            return y_arr
        y_out[~finite] = np.interp(x_arr[~finite], x_f, y_f)
        return y_out

    if method_key == "nearest":
        x_missing = x_arr[~finite]
        pos = np.searchsorted(x_f, x_missing)
        left = np.clip(pos - 1, 0, x_f.size - 1)
        right = np.clip(pos, 0, x_f.size - 1)
        d_left = np.abs(x_missing - x_f[left])
        d_right = np.abs(x_missing - x_f[right])
        choose_left = d_left <= d_right
        idx = np.where(choose_left, left, right)
        y_out[~finite] = y_f[idx]
        return y_out

    raise ValueError("Unsupported method. Use 'linear' or 'nearest'.")


def ragged_to_array(
    arrays: Sequence[Sequence[float] | np.ndarray],
    fill_value: float | None = None,
) -> np.ndarray:
    """Convert a ragged list of 1D arrays into a 2D array.

    Parameters
    ----------
    arrays:
        Sequence of array-like inputs. Each entry is flattened to 1D.
    fill_value:
        Value used to fill missing entries. If ``None``, output dtype is
        ``object`` and missing entries are ``None``.

    Returns
    -------
    np.ndarray
        2D array with shape ``(len(arrays), max_len)``.

    Raises
    ------
    ValueError
        If ``arrays`` is empty.
    """
    arr_list = [np.asarray(arr).reshape(-1) for arr in arrays]
    if len(arr_list) == 0:
        raise ValueError("arrays must not be empty.")

    n_rows = len(arr_list)
    n_cols = max(arr.size for arr in arr_list)

    if fill_value is None:
        out = np.full((n_rows, n_cols), None, dtype=object)
        for i, arr in enumerate(arr_list):
            if arr.size > 0:
                out[i, : arr.size] = arr.astype(object, copy=False)
        return out

    out_dtype = np.result_type(
        np.asarray(fill_value).dtype,
        *[arr.dtype for arr in arr_list],
    )
    out = np.full(
        (n_rows, n_cols),
        fill_value,
        dtype=out_dtype,
    )
    for i, arr in enumerate(arr_list):
        if arr.size > 0:
            out[i, : arr.size] = arr.astype(out_dtype, copy=False)
    return out
