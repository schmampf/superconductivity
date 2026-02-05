import os
from pathlib import Path
from typing import Optional, TypedDict

import h5py
import numpy as np

from .constants import G_0_muS
from .functions import bin_y_over_x
from .types import NDArray64


class MeasurementDict(TypedDict):
    directory: str
    file_name: str
    measurement: str
    keys: list[str]
    key_values: NDArray64


class DataDict(TypedDict):
    file: str
    y_keys: list[str]
    y_values: NDArray64
    amp1: float
    amp2: float
    rref: float
    t_offset_s: list[NDArray64]
    t_sweep_s: list[NDArray64]
    V_offset_mV: list[NDArray64]
    I_offset_nA: list[NDArray64]
    V_sweep_mV: list[NDArray64]
    I_sweep_nA: list[NDArray64]


def load_measurement(
    directory: str,
    file: str,
    measurement: str,
) -> MeasurementDict:

    path = Path(directory).expanduser()
    # set path_str to whatever you want to check

    # find nearest existing directory
    if not path.exists():
        base = path
        while not base.exists():
            parent = base.parent
            if parent == base:  # reached filesystem root
                break
            base = parent

        print("requested dir:", path)
        print("nearest existing dir:", base)

        dirs = sorted(x.name for x in base.iterdir() if x.is_dir())
        files = sorted(x.name for x in base.iterdir() if x.is_file())
        for d in dirs:
            print("  -", d)
        for f in files:
            print("  -", f)
        return

    files = os.listdir(directory)
    if file not in files:
        print("File not Found! choose from:")
        for f in files:
            print("-", f)
        return

    file_name = os.path.join(directory, f"{file}")
    with h5py.File(file_name, "r") as data_file:
        if not data_file.__contains__(f"measurement/{measurement}"):
            measurements = list(data_file["measurement"].keys())
            for m in measurements:
                print("- ", m)

    file_keys: MeasurementDict = {
        "directory": directory,
        "file_name": file,
        "measurement": measurement,
    }
    return file_keys


def load_keys(
    measurement: MeasurementDict,
    indices: tuple[Optional[int], Optional[int]] = (None, None),
    zero_key: Optional[str] = "no_irradiation",
    remove_keys: list[str] = [],
    add_keys: list[tuple[str, float]] = [],
) -> MeasurementDict:

    i0, i1 = indices

    file_name = os.path.join(measurement["directory"], measurement["file_name"])

    if zero_key is not None:
        add_keys.append((zero_key, 0.0))
        remove_keys.append(zero_key)

    with h5py.File(file_name, "r") as data_file:
        measurement_keys = list(
            data_file.get(f"measurement/{measurement['measurement']}")
        )

    m_keys: list[str] = []
    m_key_values: list[np.float64] = []

    for key in measurement_keys:
        if key not in remove_keys:
            key = f"{measurement['measurement']}/{key}"
            key_value = np.float64(key[i0:i1])
            m_keys.append(key)
            m_key_values.append(key_value)

    for key, key_value in add_keys:
        key = f"{measurement['measurement']}/{key}"
        key_value = np.float64(key_value)
        m_keys.append(key)
        m_key_values.append(key_value)

    key_values = np.asarray(m_key_values, dtype=np.float64)
    sorting = np.argsort(key_values)

    measurement["keys"] = np.asarray(m_keys, dtype=object)[sorting].tolist()
    measurement["key_values"] = key_values[sorting]

    return measurement


def load_data(
    measurement: MeasurementDict,
    amp: tuple[float, float] = (1, 1),
    rref: float = 51.689e3,
) -> DataDict:
    data = DataDict()
    data["file"] = os.path.join(
        measurement["directory"],
        measurement["file_name"],
    )
    data["y_keys"] = measurement["keys"]
    data["y_values"] = measurement["key_values"]
    data["amp1"] = amp[0]
    data["amp2"] = amp[1]
    data["rref"] = rref
    return data


def load_voltage_and_current(
    data: DataDict,
    trigger: int = 0,
) -> DataDict:

    data["t_offset_s"] = []
    data["V_offset_mV"] = []
    data["I_offset_nA"] = []

    data["t_sweep_s"] = []
    data["V_sweep_mV"] = []
    data["I_sweep_nA"] = []

    with h5py.File(data["file"], "r") as file:
        for y_key in data["y_keys"]:
            full_key: str = f"measurement/{y_key}"

            t_offset_s = file[full_key]["offset"]["adwin"]["time"]
            V1_offset_V = file[full_key]["offset"]["adwin"]["V1"]
            V2_offset_V = file[full_key]["offset"]["adwin"]["V1"]

            V_offset_mV = V1_offset_V / (data["amp1"]) * 1e3
            I_offset_nA = V2_offset_V / (data["amp2"] * data["rref"]) * 1e9

            data["t_offset_s"].append(t_offset_s)
            data["V_offset_mV"].append(V_offset_mV)
            data["I_offset_nA"].append(I_offset_nA)

            v_offset_mV = np.nanmean(V_offset_mV)
            i_offset_nA = np.nanmean(I_offset_nA)

            trigger_temp = file[full_key]["sweep"]["adwin"]["trigger"]
            mask = trigger_temp == (trigger + 1)
            t_sweep_s = file[full_key]["sweep"]["adwin"]["time"][mask]
            V1_sweep_V = file[full_key]["sweep"]["adwin"]["V1"][mask]
            V2_sweep_V = file[full_key]["sweep"]["adwin"]["V1"][mask]

            V_sweep_mV = V1_sweep_V / (data["amp1"]) * 1e3
            I_sweep_nA = V2_sweep_V / (data["amp2"] * data["rref"]) * 1e9

            V_sweep_mV -= v_offset_mV
            I_sweep_nA -= i_offset_nA

            data["t_sweep_s"].append(t_sweep_s)
            data["V_sweep_mV"].append(V_sweep_mV)
            data["I_sweep_nA"].append(I_sweep_nA)

    return data


def do_something(
    data: DataDict,
    nu_sampling_Hz: float = 37,
) -> DataDict:

    dt_s = 0.5 / nu_sampling_Hz
    y_len = data["y_values"].shape[0]

    temp_t_offset_s = data["t_offset_s"]
    temp_V_offset_mV = data["V_offset_mV"]
    temp_I_offset_nA = data["I_offset_nA"]

    temp_t_sweep_s = data["t_sweep_s"]
    temp_V_sweep_mV = data["V_sweep_mV"]
    temp_I_sweep_nA = data["I_sweep_nA"]

    data["t_offset_s"] = []
    data["V_offset_mV"] = []
    data["I_offset_nA"] = []

    data["t_sweep_s"] = []
    data["V_sweep_mV"] = []
    data["I_sweep_nA"] = []

    for i in range(y_len):
        t_offset_max_s = np.nanmax(temp_t_offset_s[i])
        t_offset_min_s = np.nanmin(temp_t_offset_s[i])
        t_sweep_max_s = np.nanmax(temp_t_sweep_s[i])
        t_sweep_min_s = np.nanmin(temp_t_sweep_s[i])

        t_offset_s = np.arange(
            t_offset_min_s,
            t_offset_max_s,
            dt_s,
        )

        V_offset_mV = bin_y_over_x(
            temp_t_offset_s[i],
            temp_V_offset_mV[i],
            t_offset_s,
        )

        I_offset_nA = bin_y_over_x(
            temp_t_offset_s[i],
            temp_I_offset_nA[i],
            t_offset_s,
        )

        t_sweep_s = np.arange(
            t_sweep_min_s,
            t_sweep_max_s,
            dt_s,
        )

        V_sweep_mV = bin_y_over_x(
            temp_t_sweep_s[i],
            temp_V_sweep_mV[i],
            t_sweep_s,
        )

        I_sweep_nA = bin_y_over_x(
            temp_t_sweep_s[i],
            temp_I_sweep_nA[i],
            t_sweep_s,
        )

        data["t_offset_s"].append(t_offset_s)
        data["V_offset_mV"].append(V_offset_mV)
        data["I_offset_nA"].append(I_offset_nA)

        data["t_sweep_s"].append(t_sweep_s)
        data["V_sweep_mV"].append(V_sweep_mV)
        data["I_sweep_nA"].append(I_sweep_nA)

    return data
