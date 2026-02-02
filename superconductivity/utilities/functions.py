import numpy as np
import torch

from .constants import Delta_tol_meV, T_tol_K, V_tol_mV, gamma_tol_meV, tau_tol
from .types import NDArray64


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
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2
    count, _ = np.histogram(x, bins=x_nu, weights=None)
    count = np.where(count == 0, np.nan, count)
    summe, _ = np.histogram(x, bins=x_nu, weights=y)
    return summe / count


def oversample(
    x: NDArray64,
    y: NDArray64,
    upsample: int = 100,
    upsample_method: str = "linear",
) -> tuple[NDArray64, NDArray64]:
    """Upsample paired 1D arrays `(x, y)` by an integer factor using PyTorch
    interpolation.

    The input arrays are stacked as two channels and interpolated along the
    last axis using `torch.nn.Upsample`. The returned arrays have length
    `len(x) * upsample`.

    Parameters
    ----------
    x:
        1D array of x-values.
    y:
        1D array of y-values. Must have the same shape as `x`.
    upsample:
        Integer upsampling factor. If `upsample <= 1`, the input is returned
        unchanged.
    upsample_method:
        Interpolation mode passed to `torch.nn.Upsample`, e.g. "linear"
        (default),"nearest", "area". For 1D signals, "linear" and "nearest"
        are typical.

    Returns
    -------
    x_new:
        Upsampled x-array with shape `(len(x) * upsample,)`.
    y_new:
        Upsampled y-array with shape `(len(y) * upsample,)`.

    Raises
    ------
    ValueError
        If `x` and `y` do not have the same shape.

    Notes
    -----
    This routine interpolates both `x` and `y` as if they were sampled
    on a uniform parameter grid. If `x` is non-uniform and you intend to
    resample `y(x)` onto a denser x-grid, prefer explicit resampling
    (e.g. `np.interp`, `scipy.interpolate`) using a new x-grid of your choice.
    """
    if upsample <= 1:
        return x, y

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    k = np.stack([x, y])[None, ...]  # (1, 2, N)
    k_torch = torch.tensor(k, dtype=torch.float32)

    align = (
        True
        if upsample_method in {"linear", "bilinear", "bicubic", "trilinear"}
        else None
    )
    m = torch.nn.Upsample(
        scale_factor=upsample, mode=upsample_method, align_corners=align
    )
    big = m(k_torch)

    x_new = big[0, 0, :].numpy().astype(np.float64)
    y_new = big[0, 1, :].numpy().astype(np.float64)
    return x_new, y_new


# cache hashes
def cache_hash(
    V_max_mV: float,
    dV_mV: float,
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
    string: str = "HA",
) -> str:
    string += "_"
    string += f"V_max={V_max_mV:.{V_tol_mV}f}mV_"
    string += f"dV={dV_mV:.{V_tol_mV}f}mV_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta=({Delta_1_meV:.{Delta_tol_meV}f},"
    string += f"{Delta_2_meV:.{Delta_tol_meV}f})meV_"
    string += f"gamma=({gamma_1_meV:.{gamma_tol_meV}f},"
    string += f"{gamma_2_meV:.{gamma_tol_meV}f})meV"
    return string


def cache_hash_pbar(
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
    string: str = "FCS",
) -> str:
    string += "_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta=({Delta_1_meV:.{Delta_tol_meV}f},"
    string += f"{Delta_2_meV:.{Delta_tol_meV}f})meV_"
    string += f"gamma=({gamma_1_meV:.{gamma_tol_meV}f},"
    string += f"{gamma_2_meV:.{gamma_tol_meV}f})meV"
    return string


def cache_hash_nuni(
    tau: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    string: str = "FCS",
) -> str:
    string += "_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta={Delta_meV:.{Delta_tol_meV}f}meV_"
    string += f"gamma={gamma_meV:.{gamma_tol_meV}f}meV"
    return string


def cache_hash_sym(
    V_max_mV: float,
    dV_mV: float,
    tau: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    string: str = "ha_sym",
) -> str:
    string += "_"
    string += f"V_max={V_max_mV:.{V_tol_mV}f}mV_"
    string += f"dV={dV_mV:.{V_tol_mV}f}mV_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta={Delta_meV:.{Delta_tol_meV}f}meV_"
    string += f"gamma=({gamma_meV:.{gamma_tol_meV}f})meV"
    return string
