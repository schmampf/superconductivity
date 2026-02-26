from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from superconductivity.utilities.constants import G_0_muS

Float1D = NDArray[np.floating]
Float2D = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class IVData:
    """
    Container for a single 2D measurement map and its common representations.

    Conventions
    -----------
    - x-axis (bias axis) uses shape (Nx,)
      V_bias_exp_mV, I_bias_exp_nA
    - y-axis (parameter axis) uses shape (Ny,)
      y_exp (or any other y-axis you provide)
    - measured maps use shape (Ny, Nx)
      V_data_exp_mV, I_data_exp_nA, dIdV_exp_G0, dVdI_exp_R0

    The class also provides reduced-dimension axes/fields using G_N and Delta.
    """

    # --- required raw axes / data
    V_bias_exp_mV: Float1D  # (Nx,)
    I_bias_exp_nA: Float1D  # (Nx,)
    V_data_exp_mV: Float2D  # (Ny, Nx)
    I_data_exp_nA: Float2D  # (Ny, Nx)

    # --- optional precomputed differential quantities (exp units)
    dIdV_exp_G0: Optional[Float2D] = None  # (Ny, Nx)
    dVdI_exp_R0: Optional[Float2D] = None  # (Ny, Nx)

    # --- y-axis definition
    y_exp: Optional[Float1D] = None  # (Ny,)

    # --- additional information along y_axis (e.g.: temperature)
    T_exp_K: Optional[Float1D] = None  # (Ny,)

    # --- normalization parameters
    G_N: float = 1.0
    Delta_meV: float = 1.0
    y_coupling: float = 1.0

    # --- preferred labels (HTML + TeX)
    labels_html: dict[str, str] = field(
        default_factory=lambda: {
            "V_exp_mV": "<i>V</i> (mV)",
            "I_exp_nA": "<i>I</i> (nA)",
            "y_exp": "<i>A</i> (mV)",
            "V_theo": "<i>eV/</i>Δ<sub>0</sub>",
            "I_theo": "<i>eI/G<sub>N</sub></i>Δ<sub>0</sub>",
            "y_theo": "<i>eA/hν</i>",
            "dIdV_exp_G0": "d<i>I</i>/d<i>V</i> (G<sub>0</sub>)",
            "dVdI_exp_R0": "d<i>V</i>/d<i>I</i> (R<sub>0</sub>)",
            "dIdV_theo_GN": "d<i>I</i>/d<i>V</i> (G<sub>N</sub>)",
            "dVdI_theo_RN": "d<i>V</i>/d<i>I</i> (R<sub>N</sub>)",
            "T_exp_K": "<i>T<sub>exp</sub></i> (K)",
        }
    )

    labels_tex: dict[str, str] = field(
        default_factory=lambda: {
            "V_exp_mV": r"$V\,(\mathrm{mV})$",
            "I_exp_nA": r"$I\,(\mathrm{nA})$",
            "y_exp": r"$A\,(\mathrm{mV})$",
            "V_theo": r"$eV/\Delta_0$",
            "I_theo": r"$eI/(G_\mathrm{N}\Delta_0)$",
            "y_theo": r"$eA/(h\nu)$",
            "dIdV_exp_G0": r"$\mathrm{d}I/\mathrm{d}V\,(G_0)$",
            "dVdI_exp_R0": r"$\mathrm{d}V/\mathrm{d}I\,(R_0)$",
            "dIdV_theo_GN": r"$\mathrm{d}I/\mathrm{d}V\,(G_\mathrm{N})$",
            "dVdI_theo_RN": r"$\mathrm{d}V/\mathrm{d}I\,(R_\mathrm{N})$",
            "T_exp_K": r"$T_\mathrm{exp}\,(\mathrm{K})$",
        }
    )

    def __post_init__(self) -> None:
        # Coerce to float arrays
        Vb = np.asarray(self.V_bias_exp_mV, dtype=float)
        Ib = np.asarray(self.I_bias_exp_nA, dtype=float)
        Vd = np.asarray(self.V_data_exp_mV, dtype=float)
        Id = np.asarray(self.I_data_exp_nA, dtype=float)

        if Vb.ndim != 1 or Ib.ndim != 1:
            raise ValueError("V_bias_exp_mV and I_bias_exp_nA must be 1D.")
        if Vd.ndim != 2 or Id.ndim != 2:
            raise ValueError("V_data_exp_mV and I_data_exp_nA must be 2D (Ny, Nx).")
        if Vb.shape[0] != Id.shape[1]:
            raise ValueError(
                "V_bias_exp_mV and I_data_exp_nA must have the same shape in x."
            )
        if Ib.shape[0] != Vd.shape[1]:
            raise ValueError(
                "I_bias_exp_nA and V_data_exp_mV must have the same shape in x."
            )
        if Vd.shape[0] != Id.shape[0]:
            raise ValueError(
                "V_data_exp_mV and I_data_exp_nA must have the same shape in y."
            )

        Ny, NVx = Vd.shape
        _, NIx = Id.shape
        if Vb.size != NIx:
            raise ValueError(
                "len(V_bias_exp_mV) must match I_data_exp_nA.shape[1].",
            )
        if Ib.size != NVx:
            raise ValueError(
                "len(I_bias_exp_nA) must match V_data_exp_mV.shape[1].",
            )

        object.__setattr__(self, "V_bias_exp_mV", Vb)
        object.__setattr__(self, "I_bias_exp_nA", Ib)
        object.__setattr__(self, "V_data_exp_mV", Vd)
        object.__setattr__(self, "I_data_exp_nA", Id)

        if self.y_exp is not None:
            y_exp = np.asarray(self.y_exp, dtype=float)
            if y_exp.ndim != 1 or y_exp.size != Ny:
                raise ValueError("y_exp must be 1D with len == Ny.")
            object.__setattr__(self, "y_exp", y_exp)

        if self.T_exp_K is not None:
            T = np.asarray(self.T_exp_K, dtype=float)
            if T.ndim != 1 or T.size != Ny:
                raise ValueError("T_exp_K must be 1D with len == Ny.")
            object.__setattr__(self, "T_exp_K", T)

        # Differential quantities: compute if missing
        dIdV = self.dIdV_exp_G0
        if dIdV is None:
            dIdV = self._compute_dIdV_G0(Vb, Id)
        else:
            dIdV = np.asarray(dIdV, dtype=float)
        if dIdV.shape != (Ny, NIx):
            raise ValueError("dIdV_G0 must have shape (Ny, NIx).")
        object.__setattr__(self, "dIdV_exp_G0", dIdV)

        dVdI = self.dVdI_exp_R0
        if dVdI is None:
            dVdI = self._compute_dVdI_R0(Ib, Vd)
        else:
            dVdI = np.asarray(dVdI, dtype=float)
        if dVdI.shape != (Ny, NVx):
            raise ValueError("dVdI_R0 must have shape (Ny, NVx).")
        object.__setattr__(self, "dVdI_exp_R0", dVdI)

        if self.Delta_meV <= 0:
            raise ValueError("Delta_meV must be > 0.")
        if self.G_N <= 0:
            raise ValueError("G_N must be > 0.")

    # ---------- reduced axes ----------
    @property
    def y_theo(self) -> Float1D:
        return self.y_exp / (self.y_coupling)

    @property
    def V_bias_theo(self) -> Float1D:
        return self.V_bias_exp_mV / (self.Delta_meV)

    @property
    def V_data_theo(self) -> Float2D:
        return self.V_data_exp_mV / (self.Delta_meV)

    @property
    def I_bias_theo(self) -> Float1D:
        return self.I_bias_exp_nA / (self.G_N * G_0_muS * self.Delta_meV)

    @property
    def I_data_theo(self) -> Float2D:
        return self.I_data_exp_nA / (self.G_N * G_0_muS * self.Delta_meV)

    @property
    def dIdV_theo_GN(self) -> Float2D:
        assert self.dIdV_exp_G0 is not None
        return self.dIdV_exp_G0 / self.G_N

    @property
    def dVdI_theo_RN(self) -> Float2D:
        assert self.dVdI_exp_R0 is not None
        return self.dVdI_exp_R0 * self.G_N

    # ---------- internal helpers ----------
    @staticmethod
    def _compute_dIdV_G0(
        V_bias_exp_mV: Float1D,
        I_data_exp_nA: Float2D,
    ) -> Float2D:
        """
        Compute dI/dV along the x-axis for each y-row using np.gradient.
        """
        # gradient accepts a 1D coordinate array for axis spacing
        dIdV = np.gradient(I_data_exp_nA, V_bias_exp_mV, axis=1) / G_0_muS
        return np.asarray(dIdV, dtype=float)

    @staticmethod
    def _compute_dVdI_R0(
        I_bias_exp_nA: Float1D,
        V_data_exp_mV: Float2D,
    ) -> Float2D:
        """
        Compute dI/dV along the x-axis for each y-row using np.gradient.
        """
        # gradient accepts a 1D coordinate array for axis spacing
        dVdI = np.gradient(V_data_exp_mV, I_bias_exp_nA, axis=1) * G_0_muS
        return np.asarray(dVdI, dtype=float)

    def with_ylabels(
        self,
        y_exp_html: Optional[str] = None,
        y_exp_tex: Optional[str] = None,
        y_theo_html: Optional[str] = None,
        y_theo_tex: Optional[str] = None,
        y_exp_key: str = "y_exp",
        y_theo_key: str = "y_theo",
    ) -> "IVData":
        """Return a copy with updated y-axis labels.

        This class is frozen, so label updates are performed by returning a new
        instance with modified `labels_html` and/or `labels_tex` dictionaries.

        Parameters
        ----------
        y_exp_html
            New HTML label for the experimental y-axis.
        y_exp_tex
            New LaTeX label for the experimental y-axis.
        y_theo_html
            New HTML label for the reduced/theoretical y-axis.
        y_theo_tex
            New LaTeX label for the reduced/theoretical y-axis.
        y_exp_key
            Key in `labels_html`/`labels_tex` to update for experimental y.
        y_theo_key
            Key in `labels_html`/`labels_tex` to update for theoretical y.

        Returns
        -------
        iv
            New `IVData` instance with updated label dictionaries.
        """
        labels_html = dict(self.labels_html)
        labels_tex = dict(self.labels_tex)

        if y_exp_html is not None:
            labels_html[y_exp_key] = y_exp_html
        if y_theo_html is not None:
            labels_html[y_theo_key] = y_theo_html

        if y_exp_tex is not None:
            labels_tex[y_exp_key] = y_exp_tex
        if y_theo_tex is not None:
            labels_tex[y_theo_key] = y_theo_tex

        return replace(self, labels_html=labels_html, labels_tex=labels_tex)

    # ---------- plot-spec generation ----------
    @property
    def plot_specs(self) -> dict[str, dict[str, Any]]:
        """
        Return a dict of plot specs suitable for your get_all(...).

        The returned dict maps names like 'IV_exp' to kwargs:
        {x, y, z, xlabel, ylabel, zlabel}.
        """

        specs: dict[str, dict[str, Any]] = {
            "IV_exp": dict(
                x=self.V_bias_exp_mV,
                y=self.y_exp,
                z=self.I_data_exp_nA,
                xlabel=self.labels_html["V_exp_mV"],
                ylabel=self.labels_html["y_exp"],
                zlabel=self.labels_html["I_exp_nA"],
            ),
            "dIdV_exp": dict(
                x=self.V_bias_exp_mV,
                y=self.y_exp,
                z=self.dIdV_exp_G0,
                xlabel=self.labels_html["V_exp_mV"],
                ylabel=self.labels_html["y_exp"],
                zlabel=self.labels_html["dIdV_exp_G0"],
            ),
            "VI_exp": dict(
                x=self.I_bias_exp_nA,
                y=self.y_exp,
                z=self.V_data_exp_mV,
                xlabel=self.labels_html["I_exp_nA"],
                ylabel=self.labels_html["y_exp"],
                zlabel=self.labels_html["V_exp_mV"],
            ),
            "dVdI_exp": dict(
                x=self.I_bias_exp_nA,
                y=self.y_exp,
                z=self.dVdI_exp_R0,
                xlabel=self.labels_html["I_exp_nA"],
                ylabel=self.labels_html["y_exp"],
                zlabel=self.labels_html["dVdI_exp_R0"],
            ),
            "IV_theo": dict(
                x=self.V_bias_theo,
                y=self.y_theo,
                z=self.I_data_theo,
                xlabel=self.labels_html["V_theo"],
                ylabel=self.labels_html["y_theo"],
                zlabel=self.labels_html["I_theo"],
            ),
            "dIdV_theo": dict(
                x=self.V_bias_theo,
                y=self.y_theo,
                z=self.dIdV_theo_GN,
                xlabel=self.labels_html["V_theo"],
                ylabel=self.labels_html["y_theo"],
                zlabel=self.labels_html["dIdV_theo_GN"],
            ),
            "VI_theo": dict(
                x=self.I_bias_theo,
                y=self.y_theo,
                z=self.V_data_theo,
                xlabel=self.labels_html["I_theo"],
                ylabel=self.labels_html["y_theo"],
                zlabel=self.labels_html["V_theo"],
            ),
            "dVdI_theo": dict(
                x=self.I_bias_theo,
                y=self.y_theo,
                z=self.dVdI_theo_RN,
                xlabel=self.labels_html["I_theo"],
                ylabel=self.labels_html["y_theo"],
                zlabel=self.labels_html["dVdI_theo_RN"],
            ),
        }
        return specs


# @dataclass
# class AxisView:
#     lim: LIM = None
#     ticks: Optional[np.ndarray] = None
#     ticklabels: Optional[Sequence[str]] = None
#     tickformat: str = ".3g"


# @dataclass
# class View4:
#     """Style for x/y/z/c (no semantics, just limits/ticks)."""

#     x: AxisView = field(default_factory=AxisView)
#     y: AxisView = field(default_factory=AxisView)
#     z: AxisView = field(default_factory=AxisView)
#     c: AxisView = field(default_factory=AxisView)


# @dataclass(frozen=True)
# class PlotRecipe:
#     """
#     Select which IVData fields to use as x/y/z/c for each representation.
#     """

#     name: str
#     x_key: str
#     y_key: str
#     z_key: Optional[str] = None  # slider uses z but no c
#     c_key: Optional[str] = None  # heatmap/surface use c; default c=z
#     xlabel_key: str = ""
#     ylabel_key: str = ""
#     zlabel_key: str = ""
#     clabel_key: str = ""  # optional if you want separate colorbar label

# recipes = {
#     "IV_exp": PlotRecipe(
#         name="IV_exp",
#         x_key="V_bias_exp_mV",
#         y_key="y_exp",
#         z_key="I_data_exp_nA",
#         xlabel_key="V_exp_mV",
#         ylabel_key="y_exp",
#         zlabel_key="I_exp_nA",
#     ),
#     "dIdV_exp": PlotRecipe(
#         name="dIdV_exp",
#         x_key="V_bias_exp_mV",
#         y_key="y_exp",
#         z_key="dIdV_exp_G0",     # for slider
#         c_key="dIdV_exp_G0",     # for heatmap/surface
#         xlabel_key="V_exp_mV",
#         ylabel_key="y_exp",
#         zlabel_key="dIdV_exp_G0",
#         clabel_key="dIdV_exp_G0",
#     ),
#     # ... etc for *_theo
# }
