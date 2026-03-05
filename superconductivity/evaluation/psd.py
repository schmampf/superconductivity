import numpy as np

from ..utilities.safety import (
    require_all_finite,
    require_min_size,
    require_same_shape,
    to_1d_float64,
)


def get_psd(
    I_nA: np.ndarray,
    V_mV: np.ndarray,
    t_s: np.ndarray,
    detrend: bool = True,
    window: str = "hann",
    enforce_uniform: bool = True,
    uniform_rtol: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-sided power spectral densities of ``I(t)`` and ``V(t)``.

    Parameters
    ----------
    I_nA : np.ndarray
        Current time trace in nA.
    V_mV : np.ndarray
        Voltage time trace in mV.
    t_s : np.ndarray
        Time axis in seconds.
    detrend : bool, default=True
        If ``True``, subtract the mean from ``I_nA`` and ``V_mV`` before PSD.
    window : str, default="hann"
        Window type. Supported values are ``"hann"`` and ``"none"``.
    enforce_uniform : bool, default=True
        If ``True``, require approximately uniform time spacing.
    uniform_rtol : float, default=1e-2
        Relative tolerance used in uniform-spacing validation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(I_psd_nA2_per_Hz, V_psd_mV2_per_Hz, f_Hz)`` where:
        - ``I_psd_nA2_per_Hz`` is the one-sided current PSD.
        - ``V_psd_mV2_per_Hz`` is the one-sided voltage PSD.
        - ``f_Hz`` is the one-sided frequency axis in Hz.

    Raises
    ------
    ValueError
        If inputs have mismatched shapes, non-finite values, invalid time
        spacing, or unsupported window type.
    """
    I_arr = to_1d_float64(I_nA, "I_nA")
    V_arr = to_1d_float64(V_mV, "V_mV")
    t_arr = to_1d_float64(t_s, "t_s")

    require_same_shape(I_arr, V_arr, name_a="I_nA", name_b="V_mV")
    require_same_shape(I_arr, t_arr, name_a="I_nA", name_b="t_s")
    require_min_size(I_arr, 2, name="I_nA")

    require_all_finite(I_arr, name="I_nA")
    require_all_finite(V_arr, name="V_mV")
    require_all_finite(t_arr, name="t_s")

    dt = np.diff(t_arr)
    if np.any(dt <= 0.0):
        raise ValueError("t_s must be strictly increasing.")

    dt_s = float(np.median(dt))
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("Invalid time spacing in t_s.")

    if enforce_uniform:
        if uniform_rtol < 0.0 or not np.isfinite(uniform_rtol):
            raise ValueError("uniform_rtol must be finite and >= 0.")
        if not np.allclose(dt, dt_s, rtol=uniform_rtol, atol=0.0):
            raise ValueError(
                "t_s is not uniformly sampled within tolerance. "
                "Set enforce_uniform=False or resample first.",
            )

    n = I_arr.size
    window_key = window.strip().lower()
    if window_key in {"hann", "hanning"}:
        w = np.hanning(n).astype(np.float64)
    elif window_key in {"none", "rect", "boxcar"}:
        w = np.ones(n, dtype=np.float64)
    else:
        raise ValueError("Unsupported window. Use 'hann' or 'none'.")

    I_work = I_arr - np.mean(I_arr) if detrend else I_arr
    V_work = V_arr - np.mean(V_arr) if detrend else V_arr

    fs_Hz = 1.0 / dt_s
    w2_sum = float(np.sum(w * w))
    if not np.isfinite(w2_sum) or w2_sum <= 0.0:
        raise ValueError("Invalid window normalization.")

    I_fft_nA = np.fft.rfft(I_work * w)
    V_fft_mV = np.fft.rfft(V_work * w)

    I_psd_nA2_per_Hz = (np.abs(I_fft_nA) ** 2) / (fs_Hz * w2_sum)
    V_psd_mV2_per_Hz = (np.abs(V_fft_mV) ** 2) / (fs_Hz * w2_sum)

    if n % 2 == 0:
        if I_psd_nA2_per_Hz.size > 2:
            I_psd_nA2_per_Hz[1:-1] *= 2.0
            V_psd_mV2_per_Hz[1:-1] *= 2.0
    else:
        if I_psd_nA2_per_Hz.size > 1:
            I_psd_nA2_per_Hz[1:] *= 2.0
            V_psd_mV2_per_Hz[1:] *= 2.0

    f_Hz = np.fft.rfftfreq(n, d=dt_s)

    return (
        np.asarray(I_psd_nA2_per_Hz, dtype=np.float64),
        np.asarray(V_psd_mV2_per_Hz, dtype=np.float64),
        np.asarray(f_Hz, dtype=np.float64),
    )
