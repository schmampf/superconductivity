"""JAX/Numpy utility functions.

This module collects small numerical helpers used across the evaluation code.
The focus is on lightweight building blocks that are either `jax.jit`-compiled
(where appropriate) or explicitly host-side (NumPy) when JAX is not required.

Functions
---------
bin_y_over_x:
    Histogram-based binning/averaging of `y` over `x`.
get_dydx:
    Estimate left/right endpoint slopes from linear fits.
jnp_interp_y_of_x:
    Build a JAX-callable 1D interpolator with linear extrapolation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import config as _jax_config

from .types import JInterpolator, JNDArray, NDArray64

_jax_config.update("jax_enable_x64", True)


@jax.jit
def jbin_y_over_x(
    x: JNDArray,
    y: JNDArray,
    x_bins: JNDArray,
) -> JNDArray:
    """Bin `y` over `x` using histogram-based averaging.

    Computes the mean value of `y` in each bin of `x`, where the bin centers
    are given by `x_bins`.

    Parameters
    ----------
    x:
        Sample positions.
    y:
        Sample values associated with `x`.
    x_bins:
        Bin centers (assumed uniformly spaced). The function constructs bin
        edges by shifting by half a bin width.

    Returns
    -------
    JNDArray
        Mean `y` value per bin. Bins with zero counts are returned as NaN.

    Notes
    -----
    - Implemented via `jax.numpy.histogram` with weights.
    - The function is `jax.jit`-compiled.
    """

    # Extend bin edges for histogram:
    # shift by half a bin width for center alignment
    x_nu = jnp.append(x_bins, 2 * x_bins[-1] - x_bins[-2])

    # Add one final edge
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2

    # Count how many x-values fall into each bin
    _count, _ = jnp.histogram(x, bins=x_nu, weights=None)
    _count = jnp.where(_count == 0, jnp.nan, _count)

    # Sum of y-values in each bin
    _sum, _ = jnp.histogram(x, bins=x_nu, weights=y)

    # Return mean y per bin and count
    return _sum / _count


def jinterp_y_of_x(
    x: NDArray64,
    y: NDArray64,
    dydx: float,
) -> JInterpolator:
    """Create a JAX-callable 1D linear interpolator with linear extrapolation.

    The returned function performs linear interpolation on `(x, y)` using
    `jax.numpy.interp`. For values outside the sampled range, it performs
    linear extrapolation with slope `dydx` anchored at the first/last sample.

    Parameters
    ----------
    x, y:
        Sample points and values. NaN pairs are removed prior to building the
        interpolation grid.
    dydx:
        Slope used for linear extrapolation outside the sampled x-range.

    Returns
    -------
    JInterpolator
        A function `f(x_in)` returning interpolated/extrapolated values.

    Notes
    -----
    - Preprocessing is done with NumPy (`np.asarray`, NaN filtering) and thus
      happens on the host.
    - The returned callable is `jax.jit`-compiled. Compilation occurs on the
      first call for a given input shape/dtype.
    """

    x_np: NDArray64 = np.asarray(x, dtype=np.float64)
    y_np: NDArray64 = np.asarray(y, dtype=np.float64)

    mask = np.logical_and(
        np.logical_not(np.isnan(x_np)),
        np.logical_not(np.isnan(y_np)),
    )

    x_np = x_np[mask]
    y_np = y_np[mask]

    if x_np.size == 0:
        raise ValueError(
            "jnp_interp_y_of_x: no valid (non-NaN) samples in x/y.",
        )

    x_grid = jnp.asarray(x_np, dtype=jnp.float64)
    y_grid = jnp.asarray(y_np, dtype=jnp.float64)
    dydx64 = jnp.asarray(dydx, dtype=jnp.float64)

    x0 = x_grid[0]
    x1 = x_grid[-1]
    y0 = y_grid[0]
    y1 = y_grid[-1]

    def _y_of_x(x_in: JNDArray) -> JNDArray:
        x_in = jnp.asarray(x_in, dtype=jnp.float64)
        y_out = jnp.interp(x_in, x_grid, y_grid)
        y_out = jnp.where(x_in < x0, y0 + dydx64 * (x_in - x0), y_out)
        y_out = jnp.where(x_in > x1, y1 + dydx64 * (x_in - x1), y_out)
        return y_out

    return jax.jit(_y_of_x)


def get_dydx(
    x: NDArray64,
    y: NDArray64,
    frac: float = 0.1,
) -> tuple[float, float]:
    """Estimate endpoint slopes dy/dx from the left and right edges.

    Fits a first-order polynomial (line) to the first and last fraction of the
    data and returns the corresponding slopes.

    Parameters
    ----------
    x, y:
        Input data arrays of equal length.
    frac:
        Fraction of samples used for each endpoint fit (default: 0.1).

    Returns
    -------
    tuple[float, float]
        `(dydx_left, dydx_right)` from linear fits to the left and right edge.

    Notes
    -----
    This uses `numpy.polyfit` and therefore runs on the host (NumPy), not JAX.
    """
    i = int(np.ceil(x.shape[0] * frac))
    return (
        np.polyfit(x[:+i], y[:+i], deg=1)[0],
        np.polyfit(x[-i:], y[-i:], deg=1)[0],
    )
