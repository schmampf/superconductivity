"""Sampling result container types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...utilities.meta import TransportDatasetSpec, axis, data
from ...utilities.types import NDArray64


@dataclass(frozen=True, slots=True)
class Sample:
    """One sampled result split into voltage- and current-bias datasets."""

    exp_v: TransportDatasetSpec
    exp_i: TransportDatasetSpec

    def __getitem__(self, key: str):
        if key == "exp_v":
            return self.exp_v
        if key == "exp_i":
            return self.exp_i
        if key == "Vbins_mV":
            return np.asarray(self.exp_v.V_mV.values, dtype=np.float64)
        if key == "Ibins_nA":
            return np.asarray(self.exp_i.I_nA.values, dtype=np.float64)
        if key == "I_nA":
            return np.asarray(self.exp_v.I_nA.values, dtype=np.float64)
        if key == "V_mV":
            return np.asarray(self.exp_i.V_mV.values, dtype=np.float64)
        if key == "dG_G0":
            return np.asarray(self.exp_v.dG_G0.values, dtype=np.float64)
        if key == "dR_R0":
            return np.asarray(self.exp_i.dR_R0.values, dtype=np.float64)
        raise KeyError(key)

    def copy(self) -> dict[str, NDArray64 | TransportDatasetSpec]:
        """Return a legacy-style copy used by the GUI/tests."""
        return {
            "exp_v": self.exp_v,
            "exp_i": self.exp_i,
            "Vbins_mV": np.asarray(self["Vbins_mV"], dtype=np.float64).copy(),
            "Ibins_nA": np.asarray(self["Ibins_nA"], dtype=np.float64).copy(),
            "I_nA": np.asarray(self["I_nA"], dtype=np.float64).copy(),
            "V_mV": np.asarray(self["V_mV"], dtype=np.float64).copy(),
            "dG_G0": np.asarray(self["dG_G0"], dtype=np.float64).copy(),
            "dR_R0": np.asarray(self["dR_R0"], dtype=np.float64).copy(),
        }


@dataclass(frozen=True, slots=True, init=False)
class Samples:
    """Stacked sampled results as two aggregated transport datasets."""

    exp_v: TransportDatasetSpec
    exp_i: TransportDatasetSpec
    yvalues: NDArray64

    def __init__(
        self,
        *,
        exp_v: TransportDatasetSpec | None = None,
        exp_i: TransportDatasetSpec | None = None,
        yvalues: NDArray64 | None = None,
        traces: list[dict[str, object]] | None = None,
    ) -> None:
        if traces is not None:
            built = _samples_from_traces(traces)
            object.__setattr__(self, "exp_v", built.exp_v)
            object.__setattr__(self, "exp_i", built.exp_i)
            object.__setattr__(self, "yvalues", built.yvalues)
            return
        if exp_v is None or exp_i is None or yvalues is None:
            raise TypeError("Samples requires exp_v, exp_i, and yvalues.")
        object.__setattr__(self, "exp_v", exp_v)
        object.__setattr__(self, "exp_i", exp_i)
        object.__setattr__(
            self,
            "yvalues",
            np.asarray(yvalues, dtype=np.float64).reshape(-1),
        )

    def __getitem__(self, key: str):
        if isinstance(key, int):
            return make_sample(
                Vbins_mV=np.asarray(self["Vbins_mV"], dtype=np.float64),
                Ibins_nA=np.asarray(self["Ibins_nA"], dtype=np.float64),
                I_nA=np.asarray(self["I_nA"], dtype=np.float64)[key],
                V_mV=np.asarray(self["V_mV"], dtype=np.float64)[key],
            )
        if key == "exp_v":
            return self.exp_v
        if key == "exp_i":
            return self.exp_i
        if key == "yvalues":
            return np.asarray(self.yvalues, dtype=np.float64)
        if key == "Vbins_mV":
            return np.asarray(self.exp_v.V_mV.values, dtype=np.float64)
        if key == "Ibins_nA":
            return np.asarray(self.exp_i.I_nA.values, dtype=np.float64)
        if key == "I_nA":
            return np.asarray(self.exp_v.I_nA.values, dtype=np.float64)
        if key == "V_mV":
            return np.asarray(self.exp_i.V_mV.values, dtype=np.float64)
        if key == "dG_G0":
            return np.asarray(self.exp_v.dG_G0.values, dtype=np.float64)
        if key == "dR_R0":
            return np.asarray(self.exp_i.dR_R0.values, dtype=np.float64)
        raise KeyError(key)

    def __len__(self) -> int:
        return int(self.yvalues.size)

    def __iter__(self):
        return iter(self[index] for index in range(len(self)))

    @property
    def Vbins_mV(self) -> NDArray64:
        return np.asarray(self["Vbins_mV"], dtype=np.float64)

    @property
    def Ibins_nA(self) -> NDArray64:
        return np.asarray(self["Ibins_nA"], dtype=np.float64)

    @property
    def I_nA(self) -> NDArray64:
        return np.asarray(self["I_nA"], dtype=np.float64)

    @property
    def V_mV(self) -> NDArray64:
        return np.asarray(self["V_mV"], dtype=np.float64)

    @property
    def dG_G0(self) -> NDArray64:
        return np.asarray(self["dG_G0"], dtype=np.float64)

    @property
    def dR_R0(self) -> NDArray64:
        return np.asarray(self["dR_R0"], dtype=np.float64)


def make_sample(
    *,
    Vbins_mV: NDArray64,
    Ibins_nA: NDArray64,
    I_nA: NDArray64,
    V_mV: NDArray64,
) -> Sample:
    """Construct one sampled result."""
    exp_v = TransportDatasetSpec(
        data=(data("I_nA", np.asarray(I_nA, dtype=np.float64)),),
        axes=(axis("V_mV", values=np.asarray(Vbins_mV, dtype=np.float64), order=0),),
        params=(),
    )
    exp_i = TransportDatasetSpec(
        data=(data("V_mV", np.asarray(V_mV, dtype=np.float64)),),
        axes=(axis("I_nA", values=np.asarray(Ibins_nA, dtype=np.float64), order=0),),
        params=(),
    )
    return Sample(exp_v=exp_v, exp_i=exp_i)


def make_samples(
    *,
    Vbins_mV: NDArray64,
    Ibins_nA: NDArray64,
    I_nA: NDArray64,
    V_mV: NDArray64,
    yvalues: NDArray64,
) -> Samples:
    """Construct stacked sampled results."""
    exp_v = TransportDatasetSpec(
        data=(data("I_nA", np.asarray(I_nA, dtype=np.float64)),),
        axes=(
            axis("y", values=np.asarray(yvalues, dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray(Vbins_mV, dtype=np.float64), order=1),
        ),
        params=(),
    )
    exp_i = TransportDatasetSpec(
        data=(data("V_mV", np.asarray(V_mV, dtype=np.float64)),),
        axes=(
            axis("y", values=np.asarray(yvalues, dtype=np.float64), order=0),
            axis("I_nA", values=np.asarray(Ibins_nA, dtype=np.float64), order=1),
        ),
        params=(),
    )
    return Samples(
        exp_v=exp_v,
        exp_i=exp_i,
        yvalues=np.asarray(yvalues, dtype=np.float64),
    )


def _samples_from_traces(traces: list[dict[str, object]]) -> Samples:
    if len(traces) == 0:
        raise ValueError("traces must not be empty.")
    Vbins_mV = np.asarray(traces[0]["Vbins_mV"], dtype=np.float64)
    Ibins_nA = np.asarray(traces[0]["Ibins_nA"], dtype=np.float64)
    I_nA = np.vstack([np.asarray(trace["I_nA"], dtype=np.float64) for trace in traces])
    V_mV = np.vstack([np.asarray(trace["V_mV"], dtype=np.float64) for trace in traces])
    yvalues = np.arange(len(traces), dtype=np.float64)
    return make_samples(
        Vbins_mV=Vbins_mV,
        Ibins_nA=Ibins_nA,
        I_nA=I_nA,
        V_mV=V_mV,
        yvalues=yvalues,
    )


__all__ = ["Sample", "Samples", "make_sample", "make_samples"]
