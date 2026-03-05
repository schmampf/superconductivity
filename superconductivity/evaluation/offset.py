from typing import Sequence, TypedDict

import numpy as np

from ..utilities.constants import G_0_muS
from ..utilities.functions import bin_y_over_x, fill_nans
from ..utilities.functions import upsample as upsample_xy
from ..utilities.safety import require_same_shape, to_1d_float64
from ..utilities.types import NDArray64


class OffsetResult(TypedDict):
    """Return type for :func:`get_offset`."""

    dGerr_G0: NDArray64
    dRerr_R0: NDArray64
    Voff_mV: NDArray64
    Ioff_nA: NDArray64


def _bin_y_over_x_offsets(
    x: np.ndarray,
    y: np.ndarray,
    x_bins: np.ndarray,
    x_off: np.ndarray,
) -> np.ndarray:
    """Bin ``y(x - x_off[k])`` onto ``x_bins`` for all offsets.

    Parameters
    ----------
    x : np.ndarray
        1D x-values.
    y : np.ndarray
        1D y-values.
    x_bins : np.ndarray
        Binning grid (bin centers).
    x_off : np.ndarray
        Candidate offsets.

    Returns
    -------
    np.ndarray
        Array with shape ``(len(x_bins), len(x_off))``.
    """
    out = np.full((x_bins.size, x_off.size), np.nan, dtype=np.float64)
    for j_off, off in enumerate(x_off):
        out[:, j_off] = bin_y_over_x(x=x - off, y=y, x_bins=x_bins)
    return out


def _nanargmin_finite(values: np.ndarray) -> int:
    """Return index of smallest finite value, fallback to center index.

    Parameters
    ----------
    values : np.ndarray
        1D array of objective values.

    Returns
    -------
    int
        Index of best finite value, or center index if no finite value exists.
    """
    finite = np.isfinite(values)
    if not np.any(finite):
        return int(values.size // 2)
    idx_finite = np.where(finite)[0]
    return int(idx_finite[np.argmin(values[finite])])


def get_offset(
    v_list_mV: Sequence[Sequence[float] | np.ndarray],
    i_list_nA: Sequence[Sequence[float] | np.ndarray],
    V_mV: Sequence[float] | np.ndarray,
    I_nA: Sequence[float] | np.ndarray,
    V_off_range_mV: Sequence[float] | np.ndarray,
    I_off_range_nA: Sequence[float] | np.ndarray,
    upsample: int = 10,
) -> OffsetResult:
    """Find per-curve offset via symmetry of ``G(V)`` and ``R(I)``.

    The objective functions match your notebook plots:
    ``<|G(V) - G(-V)|>`` in ``G_0`` units and
    ``<|R(I) - R(-I)|>`` in ``R_0`` units.

    Parameters
    ----------
    v_list_mV : Sequence[Sequence[float] | np.ndarray]
        List of raw voltage traces in mV.
    i_list_nA : Sequence[Sequence[float] | np.ndarray]
        List of raw current traces in nA. Must match ``v_list_mV`` length.
    V_mV : Sequence[float] | np.ndarray
        Voltage grid (bin centers) used for ``I(V)`` and ``G(V)``.
    I_nA : Sequence[float] | np.ndarray
        Current grid (bin centers) used for ``V(I)`` and ``R(I)``.
    V_off_range_mV : Sequence[float] | np.ndarray
        Candidate voltage offsets in mV.
    I_off_range_nA : Sequence[float] | np.ndarray
        Candidate current offsets in nA.
    upsample : int, default=10
        Linear index-based oversampling factor applied per input trace.

    Returns
    -------
    OffsetResult
        Dictionary with:
        ``dGerr_G0`` : shape ``(N_curves, N_Voff)``
            ``<|G(V)-G(-V)|>`` in ``G_0``.
        ``dRerr_R0`` : shape ``(N_curves, N_Ioff)``
            ``<|R(I)-R(-I)|>`` in ``R_0``.
        ``Voff_mV`` : shape ``(N_curves,)``
            Best voltage offset in mV.
        ``Ioff_nA`` : shape ``(N_curves,)``
            Best current offset in nA.
    """
    if len(v_list_mV) != len(i_list_nA):
        raise ValueError("v_list_mV and i_list_nA must have same length.")
    if upsample <= 0:
        raise ValueError("upsample must be > 0.")

    v_grid_mV = to_1d_float64(V_mV, "V_mV")
    i_grid_nA = to_1d_float64(I_nA, "I_nA")
    v_off_grid_mV = to_1d_float64(V_off_range_mV, "V_off_range_mV")
    i_off_grid_nA = to_1d_float64(I_off_range_nA, "I_off_range_nA")

    n_curves = len(v_list_mV)
    dGerr_G0 = np.full(
        (n_curves, v_off_grid_mV.size),
        np.nan,
        dtype=np.float64,
    )
    dRerr_R0 = np.full(
        (n_curves, i_off_grid_nA.size),
        np.nan,
        dtype=np.float64,
    )
    Voff_mV = np.full(n_curves, np.nan, dtype=np.float64)
    Ioff_nA = np.full(n_curves, np.nan, dtype=np.float64)

    for j_curve, (v_curve, i_curve) in enumerate(zip(v_list_mV, i_list_nA)):
        v_raw_mV = to_1d_float64(v_curve, f"v_list_mV[{j_curve}]")
        i_raw_nA = to_1d_float64(i_curve, f"i_list_nA[{j_curve}]")
        require_same_shape(
            v_raw_mV,
            i_raw_nA,
            name_a=f"v_list_mV[{j_curve}]",
            name_b=f"i_list_nA[{j_curve}]",
        )

        v_mV, i_nA = upsample_xy(
            x=v_raw_mV,
            y=i_raw_nA,
            factor=upsample,
            method="linear",
        )

        i_vs_v = _bin_y_over_x_offsets(
            x=v_mV,
            y=i_nA,
            x_bins=v_grid_mV,
            x_off=v_off_grid_mV,
        )
        g_uS = np.gradient(i_vs_v, v_grid_mV, axis=0)
        g_G0 = g_uS / G_0_muS
        g_sym = np.abs(g_G0 - np.flip(g_G0, axis=0))
        g_err_G0 = np.nanmean(g_sym, axis=0)
        dGerr_G0[j_curve, :] = g_err_G0

        j_v = _nanargmin_finite(g_err_G0)
        v_off_mV = float(v_off_grid_mV[j_v])
        Voff_mV[j_curve] = v_off_mV

        v_vs_i = _bin_y_over_x_offsets(
            x=i_nA,
            y=v_mV,
            x_bins=i_grid_nA,
            x_off=i_off_grid_nA,
        )
        r_MOhm = np.gradient(v_vs_i, i_grid_nA, axis=0)
        r_R0 = r_MOhm * G_0_muS
        r_sym = np.abs(r_R0 - np.flip(r_R0, axis=0))
        r_err_R0 = np.nanmean(r_sym, axis=0)
        dRerr_R0[j_curve, :] = r_err_R0

        j_i = _nanargmin_finite(r_err_R0)
        i_off_nA = float(i_off_grid_nA[j_i])
        Ioff_nA[j_curve] = i_off_nA

        # Stabilize metric arrays in rare sparse-binning cases.
        dGerr_G0[j_curve, :] = fill_nans(
            dGerr_G0[j_curve, :],
            x=v_off_grid_mV,
            method="linear",
        )
        dRerr_R0[j_curve, :] = fill_nans(
            dRerr_R0[j_curve, :],
            x=i_off_grid_nA,
            method="linear",
        )

    return {
        "dGerr_G0": np.asarray(dGerr_G0, dtype=np.float64),
        "dRerr_R0": np.asarray(dRerr_R0, dtype=np.float64),
        "Voff_mV": np.asarray(Voff_mV, dtype=np.float64),
        "Ioff_nA": np.asarray(Ioff_nA, dtype=np.float64),
    }
