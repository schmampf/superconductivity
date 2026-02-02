import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import NDArray

from ..style.cpd4 import cmap, colors
from ..utilities.constants import G_0_muS
from ..utilities.types import NDArray64
from .fit_atomic_contact import SolutionDict, gaussian


def plot_atomic_contact(
    results: SolutionDict,
):
    # region get_data
    V_exp_mV: NDArray64 = results["V_exp_mV"]
    I_exp_nA: NDArray64 = results["I_exp_nA"]
    G_exp: NDArray64 = results["G_exp"]
    w_exp: NDArray64 = results["w_exp"]

    V_theo_mV: NDArray64 = results["V_theo_mV"]
    I_theo_nA: NDArray64 = results["I_theo_nA"]
    G_theo: NDArray64 = results["G_theo"]
    w_theo: NDArray64 = results["w_theo"]

    channels_theo: NDArray[np.int32] = results["channels_theo"]
    tau_theo: NDArray64 = results["tau_theo"]
    T_theo_K: NDArray64 = results["T_theo_K"]
    Delta_theo_meV: NDArray64 = results["Delta_theo_meV"]
    gamma_theo_meV: NDArray64 = results["gamma_theo_meV"]

    weights: NDArray64 = results["weights"]
    w_channels: NDArray64 = results["w_channels"]
    w_T_K: NDArray64 = results["w_T_K"]
    w_Delta_meV: NDArray64 = results["w_Delta_meV"]
    w_gamma_meV: NDArray64 = results["w_gamma_meV"]
    w_para: NDArray64 = results["w_para"]

    i_channels: NDArray[np.int32] = results["i_channels"]
    i_T_K: int = results["i_T_K"]
    i_Delta_meV: int = results["i_Delta_meV"]
    i_gamma_meV: int = results["i_gamma_meV"]

    I_fit_nA: NDArray64 = results["I_fit_nA"]
    G_fit: NDArray64 = results["G_fit"]

    tau_A: NDArray64 = results["tau_A"]
    tau_fit: NDArray64 = results["tau_fit"]
    tau_hwhm: NDArray64 = results["tau_hwhm"]
    Tau_fit: float = results["Tau_fit"]
    T_fit_K: float = results["T_fit_K"]
    Delta_fit_meV: float = results["Delta_fit_meV"]
    gamma_fit_meV: float = results["gamma_fit_meV"]

    mask: NDArray[np.bool] = results["mask"]
    V_masked_mV: NDArray64 = results["V_masked_mV"]
    w_masked: NDArray64 = results["w_masked"]

    I_exp_masked_nA: NDArray64 = results["I_exp_masked_nA"]
    I_fit_masked_nA: NDArray64 = results["I_fit_masked_nA"]
    I_theo_masked_nA: NDArray64 = results["I_theo_masked_nA"]

    G_exp_masked: NDArray64 = results["G_exp_masked"]
    G_fit_masked: NDArray64 = results["G_fit_masked"]
    G_theo_masked: NDArray64 = results["G_theo_masked"]

    weights_fit_masked: NDArray64 = results["weights_fit_masked"]
    chi2_fit_masked: NDArray64 = results["chi2_fit_masked"]
    all_pincode_indices: NDArray[np.int32] = results["all_pincode_indices"]

    Tau_min: float = results["Tau_min"]
    Tau_max: float = results["Tau_max"]
    weighted: bool = results["weighted"]

    fitting_time: float = results["fitting_time"]
    generation_time: float = results["generation_time"]

    n_channels: int = channels_theo.shape[0]
    n_tau: int = tau_theo.shape[0]
    n_T: int = T_theo_K.shape[0]
    n_Delta: int = Delta_theo_meV.shape[0]
    n_gamma: int = gamma_theo_meV.shape[0]
    n_indices: int = all_pincode_indices.shape[0]
    n_para: int = n_T * n_Delta * n_gamma

    # endregion

    # region printing
    statistics = True
    if statistics:
        print(f"generation_time = {generation_time:.2f} s")
        print(f"fitting_time = {fitting_time:.2f} s")
        print(f"--------------------------")

        print(f"{n_indices} possibilities within")
        print(f"({Tau_min} - {Tau_max}) G_0, for")
        print(f"{n_channels} channels and")
        print(f"{n_tau} transmissions")
        print(f"--------------------------")

        print(f"for {n_para} parameter")
        print(f"- {n_T} temperatures")
        print(f"- {n_Delta} energy gaps")
        print(f"- {n_gamma} dynes parameter")
        print(f"==========================")
        print(" ")

    fit_results = True
    if fit_results:
        for i, tau_i in enumerate(tau_fit):
            print(f"τ_{i} = {tau_i:.2f} ({tau_hwhm[i]:.2f})")
        print(f"Στ = {Tau_fit:.2f}")
        print(f"--------------------------")

        print(f"T_fit = {T_fit_K} K")
        print(f"Δ_fit = {Delta_fit_meV*1e3:.1f} μeV")
        print(f"γ_fit = {gamma_fit_meV*1e3:.2f} μeV")
        print(f"==========================")

    # endregion

    # region calcs
    dIdV_exp: NDArray64 = np.gradient(I_exp_nA, V_exp_mV) / G_0_muS
    dIdV_fit: NDArray64 = np.gradient(I_fit_nA, V_exp_mV) / G_0_muS

    G_N_min: NDArray64 = np.ones_like(V_exp_mV) * Tau_min
    G_N_max: NDArray64 = np.ones_like(V_exp_mV) * Tau_max
    G_N_fit: NDArray64 = np.ones_like(V_exp_mV) * Tau_fit

    # endregion

    subplot: tuple[Figure, list[Axes]] = plt.subplots(
        ncols=2,
        nrows=4,
        figsize=(9, 12),
    )
    fig: Figure = subplot[0]
    axs: list[Axes] = subplot[1]

    # axs stuff
    ax_I_fit: Axes = axs[0, 0]
    ax_G_fit: Axes = axs[1, 0]
    ax_dIdV_fit: Axes = axs[2, 0]
    ax_w_fit: Axes = axs[3, 0]

    ax_Tem: Axes = axs[0, 1]
    ax_Del: Axes = axs[1, 1]
    ax_gam: Axes = axs[2, 1]
    ax_tau: Axes = axs[3, 1]

    for ax in [ax_G_fit, ax_dIdV_fit, ax_w_fit]:
        ax.sharex(ax_I_fit)
    for ax in [ax_Tem, ax_Del, ax_gam]:
        ax.sharex(ax_tau)

    for ax in [
        ax_I_fit,
        ax_G_fit,
        ax_dIdV_fit,
        ax_w_fit,
        ax_Tem,
        ax_Del,
        ax_gam,
        ax_tau,
    ]:
        ax.tick_params(
            direction="in",  # Ticks nach innen
            top=True,  # obere Ticks ein
            bottom=True,  # untere Ticks ein
            left=True,  # linke Ticks ein
            right=True,  # rechte Ticks ein
            which="both",  # sowohl Major- als auch Minor-Ticks
        )

    # add title
    tau_fit_temp = np.flip(np.round(tau_fit, decimals=2))
    title = f"$T={T_fit_K:.2f}\\,$K$, \\Delta={Delta_fit_meV*1e3:.1f}\\,$µeV$, \\gamma={gamma_fit_meV*1e3:.1f}\\,$µeV"
    title += "$, \\tau_i=\\{$"
    for tau in tau_fit_temp:
        title += f"${tau:.2f},\\,$"
    title = title[:-5] + "\\}$"

    fig.suptitle(title)

    # labels
    ax_I_fit.set_ylabel("$I$ (nA)")
    ax_G_fit.set_ylabel("$I/V$ ($G_0$)")
    ax_dIdV_fit.set_ylabel("d$I/$d$V$ ($G_0$)")
    ax_w_fit.set_ylabel("$w=\\mathrm{exp}(-\\chi^2\\,/\\,2)$")

    ax_w_fit.set_xlabel("$V$ (mV)")

    for ax in [ax_tau, ax_Tem, ax_Del, ax_gam]:
        ax.set_ylabel("$\\langle w \\rangle_V$")
    ax_tau.set_xlabel("$\\tau_i$")

    # Current
    ax_I_fit.plot(V_exp_mV, I_fit_nA, "-", color="red", label="fit")
    ax_I_fit.plot(V_exp_mV, I_exp_nA, ".", color=colors(2), ms=1, label="exp data")
    ax_I_fit.plot(
        V_masked_mV, I_exp_masked_nA, "x", color=colors(0), ms=3, label="masked data"
    )

    # Conductance
    ax_G_fit.plot(V_exp_mV, G_fit, "-", color="red")
    ax_G_fit.plot(V_exp_mV, G_exp, ".", color=colors(2), ms=1)
    ax_G_fit.plot(V_masked_mV, G_exp_masked, "x", color=colors(0), ms=3)

    # differential conductance
    ax_dIdV_fit.plot(V_exp_mV, dIdV_fit, "-", color="red")
    ax_dIdV_fit.plot(V_exp_mV, dIdV_exp, ".", color=colors(2), ms=1)

    ax_dIdV_fit.fill_between(V_exp_mV, G_N_min, G_N_max, color="lightblue")
    ax_dIdV_fit.plot(V_exp_mV, G_N_fit, color=colors(0))

    # weigths/chi fit
    ax_w_fit.plot(V_masked_mV, weights_fit_masked, "x", color=colors(0), ms=3)

    # transmissions
    theo = np.linspace(0, 1, 1001)
    for i_ch in range(0, n_channels):
        ax_tau.plot(
            tau_theo,
            w_channels[i_ch, :],
            "x",
            ms=2,
            color=cmap()(i_ch / n_channels),
        )
        ax_tau.plot(
            theo,
            gaussian(theo, tau_A[i_ch], tau_fit[i_ch], tau_hwhm[i_ch]),
            "-",
            color=cmap()(i_ch / n_channels),
        )

    ylim = ax_tau.get_ylim()

    for i_ch in range(0, n_channels):
        ax_tau.vlines(
            x=tau_fit[i_ch],
            ymin=ylim[0],
            ymax=tau_A[i_ch],
            color=cmap()(i_ch / n_channels),
            zorder=0,
        )
        ax_tau.hlines(
            xmin=tau_fit[i_ch] - tau_hwhm[i_ch],
            xmax=tau_fit[i_ch] + tau_hwhm[i_ch],
            y=tau_A[i_ch] / 2,
            color=cmap()(i_ch / n_channels),
            zorder=0,
        )
    ax_tau.set_ylim(ylim)

    # legend stuff
    ylim = ax_I_fit.get_ylim()

    ax_I_fit.fill_between(
        V_exp_mV,
        np.full_like(V_exp_mV, -10 * np.abs(ylim[0])),
        np.full_like(V_exp_mV, -11 * np.abs(ylim[0])),
        label="$\{G_\\mathrm{N}\}_0$",
        color="lightblue",
    )
    ax_I_fit.plot(
        V_exp_mV,
        np.full_like(V_exp_mV, -10 * np.abs(ylim[0])),
        label="$G_{\\mathrm{N}}^\\mathrm{fit}$",
        color=colors(0),
    )
    ax_I_fit.set_ylim(ylim)
    ax_I_fit.legend(
        fontsize=8, markerscale=0.7, handlelength=1.2, labelspacing=0.2, borderpad=0.2
    )

    # region T, Delta, gamma
    parameter: list[tuple[Axes, NDArray64, NDArray64, str]] = [
        (
            ax_Tem,
            np.mean(weights, axis=(3, 4)),
            T_theo_K,
            "$T$ (K)",
        ),
        (
            ax_Del,
            np.mean(weights, axis=(2, 4)),
            Delta_theo_meV,
            "$\\Delta$ (meV)",
        ),
        (
            ax_gam,
            np.mean(weights, axis=(2, 3)),
            gamma_theo_meV,
            "$\\gamma$ (meV)",
        ),
    ]
    axs0: list[Axes] = []

    for ax, y, z, z_str in parameter:
        w_y = np.mean(y, axis=(0, 1))

        ax0: Axes = inset_axes(
            ax,
            width="30%",  # width = 30% of parent_bbox
            height="30%",  # height : 1 inch
            loc="upper right",
        )
        axs0.append(ax0)

        ax0.set_xlabel(z_str)
        # ax0.set_ylabel("$w$")

        for i_z in range(z.shape[0]):
            if i_z != np.argmax(w_y):
                color = cmap()(i_z / z.shape[0])
            else:
                color = "r"

            ax.plot(
                tau_theo,
                y[0, :, i_z],
                color=color,
                label=f"{z[i_z]}",
            )

            for i_ch in range(1, n_channels):
                ax.plot(tau_theo, y[i_ch, :, i_z], color=color)

            ax0.plot(z[i_z], w_y[i_z], ".", color=color)

        y_lim = ax0.get_ylim()
        ax0.vlines(
            z[np.argmax(w_y)],
            ymin=y_lim[0],
            ymax=y_lim[1],
            color="red",
            alpha=0.3,
            zorder=0,
        )
        ax0.set_ylim(y_lim)
