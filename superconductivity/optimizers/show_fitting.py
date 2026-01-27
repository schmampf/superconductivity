import numpy as np
import matplotlib.pyplot as plt

from utilities.constants import G_0_muS, e, h
from utilities.plotting.cpd4 import colors, cmap

from optimizers.fit_pat import NDArray64, SolutionDict


def show_fitting(solution: SolutionDict, num: int = 0):

    color_pos = colors(3)
    color_neg = colors(3, 0.3)
    color_fit = colors(5)
    color_fit_neg = colors(5)
    color_ini = colors(4, 0.5)
    color_ini_neg = colors(4, 0.5)
    color_w = "grey"

    V_mV: NDArray64 = np.array(solution["V_mV"], dtype=np.float64)

    I_exp: NDArray64 = np.array(solution["I_exp_nA"], dtype=np.float64)
    I_ini: NDArray64 = np.array(solution["I_ini_nA"], dtype=np.float64)
    I_fit: NDArray64 = np.array(solution["I_fit_nA"], dtype=np.float64)

    popt: NDArray64 = np.array(solution["popt"], dtype=np.float64)
    perr: NDArray64 = np.array(solution["perr"], dtype=np.float64)

    Delta_mV: float = solution["Delta_mV"]
    G_N: float = solution["G_N"]

    V_nan_0_mV = np.where(V_mV == 0.0, np.nan, V_mV)
    m_pos = V_mV > 0
    m_neg = V_mV < 0

    g_exp = I_exp / V_nan_0_mV / G_0_muS
    g_ini = I_ini / V_nan_0_mV / G_0_muS
    g_fit = I_fit / V_nan_0_mV / G_0_muS

    G_exp = np.gradient(I_exp, V_mV) / G_0_muS
    G_ini = np.gradient(I_ini, V_mV) / G_0_muS
    G_fit = np.gradient(I_fit, V_mV) / G_0_muS

    V_mV_pos = V_mV[m_pos]
    V_mV_neg = -V_mV[m_neg]

    I_exp_pos = I_exp[m_pos]
    I_exp_neg = -I_exp[m_neg]
    I_ini_pos = I_ini[m_pos]
    I_ini_neg = -I_ini[m_neg]
    I_fit_pos = I_fit[m_pos]
    I_fit_neg = -I_fit[m_neg]

    g_exp_pos = g_exp[m_pos]
    g_exp_neg = g_exp[m_neg]
    g_ini_pos = g_ini[m_pos]
    g_ini_neg = g_ini[m_neg]
    g_fit_pos = g_fit[m_pos]
    g_fit_neg = g_fit[m_neg]

    ug_ini_pos = +(g_ini_pos**2) - g_exp_pos**2
    ug_ini_neg = -(g_ini_neg**2) + g_exp_neg**2
    ug_fit_pos = +(g_fit_pos**2) - g_exp_pos**2
    ug_fit_neg = -(g_fit_neg**2) + g_exp_neg**2

    G_exp_pos = G_exp[m_pos]
    G_exp_neg = G_exp[m_neg]
    G_ini_pos = G_ini[m_pos]
    G_ini_neg = G_ini[m_neg]
    G_fit_pos = G_fit[m_pos]
    G_fit_neg = G_fit[m_neg]

    uG_ini_pos = +(G_ini_pos**2) - G_exp_pos**2
    uG_ini_neg = -(G_ini_neg**2) + G_exp_neg**2
    uG_fit_pos = +(G_fit_pos**2) - G_exp_pos**2
    uG_fit_neg = -(G_fit_neg**2) + G_exp_neg**2

    plt.close(num)
    fig, axs = plt.subplots(
        num=num, nrows=5, sharex=True, height_ratios=(2, 1, 1, 1, 1), figsize=(6, 9)
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.05)

    def VtoE(V_mV):
        return V_mV / Delta_mV

    def EtoV(E_meV):
        return E_meV * Delta_mV

    def G_G_0_to_G_arbu(x):
        return x / G_N

    def G_arbu_to_G_G_0(x):
        return x * G_N

    def I_nA_to_I_arbu(x):
        return x / (1e-9 * 2 * e / h * Delta_mV * 1e-3)

    def I_arbu_to_I_nA(x):
        return x * (1e-9 * 2 * e / h * Delta_mV * 1e-3)

    axs_2 = []
    for ax in axs:
        ax.tick_params(
            direction="in",  # Ticks nach innen
            top=False,  # obere Ticks ein
            bottom=True,  # untere Ticks ein
            left=True,  # linke Ticks ein
            right=False,  # rechte Ticks ein
            which="both",  # sowohl Major- als auch Minor-Ticks
        )
        ax.grid()
        ax.tick_params(labelbottom=False)
        ax_2 = ax.secondary_xaxis("top", functions=(VtoE, EtoV))
        ax_2.tick_params(direction="in", top=True)
        ax_2.tick_params(labeltop=False)
        axs_2.append(ax_2)
    axs_2[0].tick_params(labeltop=True)
    axs[-1].tick_params(labelbottom=True)

    (ax_I, ax_g, ax_G, ax_ug, ax_uG) = axs
    (ax_I_2, ax_g_2, ax_G_2, ax_ug_2, ax_uG_2) = axs_2
    axs_s = []
    for ax in [ax_ug, ax_uG]:
        ax_s = ax.twinx()
        ax_s.tick_params(axis="y", labelcolor=color_w, direction="in", color=color_w)
        ax_s.set_ylabel("$\\sigma$ (arb. u.)", color=color_w)
        ax_s.set_ylim((-0.1, 1.1))
        axs_s.append(ax_s)

    axs_ = []
    for ax in [ax_g, ax_G]:
        ax_ = ax.secondary_yaxis("right", functions=(G_G_0_to_G_arbu, G_arbu_to_G_G_0))
        ax_.tick_params(direction="in", right=True)
        ax_.tick_params(labelleft=False)
        axs_.append(ax_)

    ax_I_ = ax_I.secondary_yaxis("right", functions=(I_nA_to_I_arbu, I_arbu_to_I_nA))
    ax_I_.tick_params(direction="in", right=True)
    ax_I_.tick_params(labelleft=False)

    ax_I_2.set_xlabel("$E$ ($\\Delta$)")
    ax_uG.set_xlabel("$V$ (mV)")

    ax_I.set_ylabel("$I$ (nA)")
    ax_I_.set_ylabel("$I$ ($2e/h \\cdot \\Delta$)")
    ax_g.set_ylabel("$I/V$ ($G_0$)")
    axs_[0].set_ylabel("$I/V$ ($G$)")
    ax_G.set_ylabel("d$I/$d$V$ ($G_0$)")
    axs_[1].set_ylabel("d$I/$d$V$ ($G$)")
    ax_ug.set_ylabel("$u_{I/V}$ ($G_0^2$)")
    ax_uG.set_ylabel("$u_{\\mathrm{d}I/\\mathrm{d}V}$ ($G_0^2$)")

    ax_I.plot(
        V_mV_pos,
        I_exp_pos,
        ".",
        label="$I_\\mathrm{exp}^\\rightarrow$",
        color=color_pos,
        ms=2,
        zorder=13,
    )
    ax_I.plot(
        V_mV_neg,
        I_exp_neg,
        ".",
        label="$I_\\mathrm{exp}^\\leftarrow$",
        color=color_neg,
        ms=2,
        zorder=13,
    )
    ax_I.plot(V_mV_pos, I_fit_pos, label="$\\mathrm{fit}$", color=color_fit, zorder=12)
    ax_I.plot(V_mV_neg, I_fit_neg, color=color_fit, zorder=12)
    ax_I.plot(V_mV_pos, I_ini_pos, label="$\\mathrm{ini}$", color=color_ini, zorder=11)
    ax_I.plot(V_mV_neg, I_ini_neg, color=color_ini, zorder=11)
    ax_I.legend()

    ax_g.plot(V_mV_pos, g_exp_pos, ".", color=color_pos, ms=2, zorder=13)
    ax_g.plot(V_mV_neg, g_exp_neg, ".", color=color_neg, ms=2, zorder=13)
    ax_g.plot(V_mV_pos, g_fit_pos, color=color_fit, zorder=12)
    ax_g.plot(V_mV_neg, g_fit_neg, color=color_fit, zorder=12)
    ax_g.plot(V_mV_pos, g_ini_pos, color=color_ini, zorder=11)
    ax_g.plot(V_mV_neg, g_ini_neg, color=color_ini, zorder=11)

    ax_G.plot(V_mV_pos, G_exp_pos, ".", color=color_pos, ms=2, zorder=13)
    ax_G.plot(V_mV_neg, G_exp_neg, ".", color=color_neg, ms=2, zorder=13)
    ax_G.plot(V_mV_pos, G_fit_pos, color=color_fit, zorder=12)
    ax_G.plot(V_mV_neg, G_fit_neg, color=color_fit, zorder=12)
    ax_G.plot(V_mV_pos, G_ini_pos, color=color_ini, zorder=11)
    ax_G.plot(V_mV_neg, G_ini_neg, color=color_ini, zorder=11)

    ax_ug.plot(
        V_mV_pos, np.full_like(V_mV_pos, 0.0), ".", color=color_pos, ms=2, zorder=13
    )
    ax_ug.plot(V_mV_pos, ug_fit_pos, color=color_fit, zorder=12)
    ax_ug.plot(V_mV_neg, -ug_fit_neg, color=color_fit_neg, zorder=12)
    ax_ug.plot(V_mV_pos, ug_ini_pos, color=color_ini, zorder=11)
    ax_ug.plot(V_mV_neg, -ug_ini_neg, color=color_ini_neg, zorder=11)

    ax_uG.plot(
        V_mV_pos, np.full_like(V_mV_pos, 0.0), ".", color=color_pos, ms=2, zorder=13
    )
    ax_uG.plot(V_mV_pos, uG_fit_pos, color=color_fit, zorder=12)
    ax_uG.plot(V_mV_neg, -uG_fit_neg, color=color_fit_neg, zorder=12)
    ax_uG.plot(V_mV_pos, uG_ini_pos, color=color_ini, zorder=11)
    ax_uG.plot(V_mV_neg, -uG_ini_neg, color=color_ini_neg, zorder=11)

    axs_s[0].plot(V_mV_pos, np.full_like(V_mV_pos, 1.0), color=color_w)
    axs_s[1].plot(V_mV_pos, np.full_like(V_mV_pos, 0.0), color=color_w)

    # add text box for the statistics
    stats = ""
    stats += f"$G={popt[0]:.4f}$"
    stats += f"$\\,({int(np.round(perr[0]*1e4))})$" if perr[0] != 0 else ""
    stats += "$\\,G_0$"
    stats += f"\n$T={popt[1]*1e3:.1f}$"
    stats += f"$\\,({int(np.round(perr[1]*1e4))})$" if perr[1] != 0 else ""
    stats += "$\\,$mK"
    stats += f"\n$\\Delta={popt[2]*1e3:.1f}$"
    stats += f"$\\,({int(np.round(perr[2]*1e4))})$" if perr[2] != 0 else ""
    stats += "$\\,$µeV"
    stats += f"\n$\\Gamma={popt[3]*1e3:.2f}$"
    stats += f"$\\,({int(np.round(perr[3]*1e5))})$" if perr[3] != 0 else ""
    stats += "$\\,$µeV"

    if "pat" in solution["model"]:
        stats += f"\n$A={popt[4]*1e3:.2f}$"
        stats += f"$\\,({int(np.round(perr[4]*1e5))})$" if perr[4] != 0 else ""
        stats += "$\\,$µeV"
        stats += f"\n$\\nu={popt[5]:.2f}$"
        stats += f"$\\,({int(np.round(perr[5]*1e2))})$" if perr[5] != 0 else ""
        stats += "$\\,$GHz"
    bbox = dict(boxstyle="round", fc="lightgrey", ec="grey", alpha=0.5)
    ax_I.text(
        0.05,
        0.55,
        stats,
        fontsize=7,
        bbox=bbox,
        transform=ax_I.transAxes,
        horizontalalignment="left",
    )
    fig.suptitle(f"{solution['optimizer']}, {solution['model']}")
    fig.tight_layout()


def show_stats(solution: SolutionDict):
    print(f"# Model: '{solution['model']}'")
    print(f"# Optimizer: '{solution['optimizer']}'")

    print(f"\n# --- paramters ---")
    print(
        f"# τ = {solution['popt'][0]}{f' ({solution['perr'][0]})' if not solution['fixed'][0] else ''}"
    )
    print(
        f"# T = {solution['popt'][1]}{f' ({solution['perr'][1]})' if not solution['fixed'][1] else ''} K"
    )
    print(
        f"# Δ = {solution['popt'][2]}{f' ({solution['perr'][2]})' if not solution['fixed'][2] else ''} mV"
    )
    print(
        f"# Γ = {solution['popt'][3]}{f' ({solution['perr'][3]})' if not solution['fixed'][3] else ''} mV"
    )

    if "pat" in solution["model"]:
        print(
            f"# A = {solution['popt'][4]}{f' ({solution['perr'][4]})' if not solution['fixed'][4] else ''} mV"
        )
        print(
            f"# ν = {solution['popt'][5]}{f' ({solution['perr'][5]})' if not solution['fixed'][5] else ''} GHz"
        )

    print("\n# --- input ---")
    print("# solution = fit_current(\n#     V_mV=V_mV,\n#     I_nA=I_nA,")
    print(f"#     G_N = ", end="")
    print(f"({solution['guess'][0]:.05f}, ", end="")
    print(f"({solution['lower'][0]:.05f}, ", end="")
    print(f"{solution['upper'][0]:.05f}), ", end="")
    print(f"{solution['fixed'][0]}),")

    print("#     T_K = ", end="")
    print(f"({solution['guess'][1]:.05f}, ", end="")
    print(f"({solution['lower'][1]:.05f}, ", end="")
    print(f"{solution['upper'][1]:.05f}), ", end="")
    print(f"{solution['fixed'][1]}),")

    print("#     Delta_mV = ", end="")
    print(f"({solution['guess'][2]:.05f}, ", end="")
    print(f"({solution['lower'][2]:.05f}, ", end="")
    print(f"{solution['upper'][2]:.05f}), ", end="")
    print(f"{solution['fixed'][2]}),")

    print("#     Gamma_mV = ", end="")
    print(f"({solution['guess'][3]:.05f}, ", end="")
    print(f"({solution['lower'][3]:.05f}, ", end="")
    print(f"{solution['upper'][3]:.05f}), ", end="")
    print(f"{solution['fixed'][3]}),")

    print("#     A_mV = ", end="")
    print(f"({solution['guess'][4]:.05f}, ", end="")
    print(f"({solution['lower'][4]:.05f}, ", end="")
    print(f"{solution['upper'][4]:.05f}), ", end="")
    print(f"{solution['fixed'][4]}),")

    print("#     nu_GHz = ", end="")
    print(f"({solution['guess'][5]:05.02f}, ", end="")
    print(f"({solution['lower'][5]:05.02f}, ", end="")
    print(f"{solution['upper'][5]:05.02f}), ", end="")
    print(f"{solution['fixed'][5]}),")

    print(f'#     model="{solution['model']}",')
    print(f'#     optimizer="{solution['optimizer']}",')
    print("# )")
