import os
import sys
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from hdf5view.mainwindow import MainWindow
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication

from ..style.cpd5 import seeblau100, seegrau100
from .evaluation import DataDict, MeasurementDict
from .functions import bin_y_over_x
from .types import NDArray64


def show_measurement(
    measurement: MeasurementDict,
):
    file_name = os.path.join(measurement["directory"], measurement["file_name"])
    app = QApplication(sys.argv)
    app.setOrganizationName("hdf5view")
    app.setApplicationName("hdf5view")
    app.setWindowIcon(QIcon("icons:hdf5view.svg"))
    window = MainWindow(app)
    window.show()
    window.open_file(file_name)
    app.exec_()


def show_amplifications(
    data: DataDict,
):

    file_name = os.path.join(
        data["file"]["directory"],
        data["file"]["file_name"],
    )
    t0 = min(t.min() for t in data["t_offset_s"])
    t1 = max(t.max() for t in data["t_sweep_s"])
    time = np.linspace(t0, t1, 101)

    with h5py.File(file_name, "r") as data_file:

        if data_file.__contains__("status/femto"):
            femto_key = "femto"
        elif data_file.__contains__("status/femtos"):
            femto_key = "femtos"
        else:
            raise KeyError("Femtos not found!")

        femto_data = np.array(data_file[f"status/{femto_key}"])

        amp1 = bin_y_over_x(femto_data["time"], femto_data["amp_A"], time)
        amp2 = bin_y_over_x(femto_data["time"], femto_data["amp_B"], time)

        plt.close(1000)
        plt.figure(1000, figsize=(6, 1.5))
        plt.semilogy(
            time - np.min(time),
            amp1,
            "-",
            label="Voltage Amplification 1",
            color=seeblau100,
        )
        plt.semilogy(
            time - np.min(time),
            amp2,
            "--",
            label="Voltage Amplification 2",
            color=seegrau100,
        )
        plt.legend()
        plt.title("Femto Amplifications (Status)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplification")
        plt.tight_layout()
        plt.show()


def _periodogram_psd(t: NDArray64, y: NDArray64) -> tuple[NDArray64, NDArray64]:
    """Compute a simple one-sided periodogram PSD for uniformly sampled data.

    Notes
    -----
    - This assumes `t` is (approximately) uniformly sampled.
    - Output units are (y-units)^2 / Hz.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if t.size < 4 or y.size < 4:
        raise ValueError("Need at least 4 samples for PSD.")

    # Ensure matching lengths
    n = min(t.size, y.size)
    t = t[:n]
    y = y[:n]

    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time axis for PSD (non-positive dt).")

    # Detrend (remove mean)
    y = y - np.nanmean(y)

    # Replace NaNs with 0 after detrending (conservative for PSD)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Hann window to reduce leakage
    w = np.hanning(n)
    yw = y * w

    # One-sided FFT
    Y = np.fft.rfft(yw)
    f = np.fft.rfftfreq(n, dt)

    # PSD normalization for windowed periodogram:
    # Pxx = |FFT|^2 / (fs * sum(w^2))
    fs = 1.0 / dt
    denom = fs * np.sum(w**2)
    Pxx = (np.abs(Y) ** 2) / denom

    # Convert to one-sided PSD (except DC and Nyquist)
    if n % 2 == 0:
        # even n includes Nyquist bin
        Pxx[1:-1] *= 2.0
    else:
        Pxx[1:] *= 2.0

    return f.astype(np.float64), Pxx.astype(np.float64)


def _plot_psd_bundle(
    t_list: list[NDArray64],
    y_list: list[NDArray64],
    title: str,
    y_label: str,
) -> None:
    if len(t_list) == 0 or len(y_list) == 0:
        print(f"No data for: {title}")
        return

    plt.figure(figsize=(6, 3.2))

    # Plot individual PSDs lightly, plus an average curve if possible
    f_ref: Optional[NDArray64] = None
    P_stack: list[NDArray64] = []

    for i, (t, y) in enumerate(zip(t_list, y_list)):
        try:
            f, P = _periodogram_psd(t, y)
        except Exception as exc:
            print(f"Skipping trace {i} in '{title}': {exc}")
            continue

        plt.semilogy(
            f[1:],
            P[1:],
            ".",
            markersize=0.5,
            alpha=0.25,
            linewidth=0.8,
            color=seegrau100,
        )

        if f_ref is None:
            f_ref = f
            P_stack.append(P)
        else:
            # Only average traces with identical frequency grid
            if f.shape == f_ref.shape and np.allclose(f, f_ref):
                P_stack.append(P)

    if f_ref is not None and len(P_stack) >= 2:
        P_mean = np.nanmean(np.stack(P_stack, axis=0), axis=0)
        plt.semilogy(
            f_ref[1:],
            P_mean[1:],
            linewidth=2.0,
            label="mean",
            color=seeblau100,
        )
        plt.legend(loc="best")

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


def show_psd(
    data: DataDict,
    sweep: bool = True,
    offset: bool = True,
    V_mV: bool = True,
    I_nA: bool = True,
):

    t_offset_s = data["t_offset_s"]
    V_offset_mV = data["V_offset_mV"]
    I_offset_nA = data["I_offset_nA"]

    t_sweep_s = data["t_sweep_s"]
    V_sweep_mV = data["V_sweep_mV"]
    I_sweep_nA = data["I_sweep_nA"]

    # Voltage PSDs
    if V_mV:
        if offset:
            _plot_psd_bundle(
                t_list=t_offset_s,
                y_list=V_offset_mV,
                title="Voltage PSD (offset)",
                y_label=r"$S_V$ (mV$^2$/Hz)",
            )
        if sweep:
            _plot_psd_bundle(
                t_list=t_sweep_s,
                y_list=V_sweep_mV,
                title="Voltage PSD (sweep)",
                y_label=r"$S_V$ (mV$^2$/Hz)",
            )

    # Current PSDs
    if I_nA:
        if offset:
            _plot_psd_bundle(
                t_list=t_offset_s,
                y_list=I_offset_nA,
                title="Current PSD (offset)",
                y_label=r"$S_I$ (nA$^2$/Hz)",
            )
        if sweep:
            _plot_psd_bundle(
                t_list=t_sweep_s,
                y_list=I_sweep_nA,
                title="Current PSD (sweep)",
                y_label=r"$S_I$ (nA$^2$/Hz)",
            )
