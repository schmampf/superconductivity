from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .fit_pat import ParameterSpec, SolutionDict


def _arrays_path(directory: Path) -> Path:
    return directory / "solution_arrays.npz"


def _metadata_path(directory: Path) -> Path:
    return directory / "solution_metadata.json"


def clear_solution(directory: Path) -> None:
    """Remove any persisted PAT solution artifacts from ``directory``."""
    _arrays_path(directory).unlink(missing_ok=True)
    _metadata_path(directory).unlink(missing_ok=True)


def save_solution(directory: Path, solution: SolutionDict) -> None:
    """Persist a PAT solution to ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)

    arrays_path = _arrays_path(directory)
    metadata_path = _metadata_path(directory)
    arrays_tmp = arrays_path.with_suffix(".tmp.npz")
    metadata_tmp = metadata_path.with_suffix(".tmp.json")

    weights = solution["weights"]
    has_weights = weights is not None

    np.savez(
        arrays_tmp,
        V_mV=np.asarray(solution["V_mV"], dtype=np.float64),
        I_exp_nA=np.asarray(solution["I_exp_nA"], dtype=np.float64),
        I_ini_nA=np.asarray(solution["I_ini_nA"], dtype=np.float64),
        I_fit_nA=np.asarray(solution["I_fit_nA"], dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64) if has_weights else np.array([]),
    )

    metadata: dict[str, Any] = {
        "maxfev": solution["maxfev"],
        "has_weights": has_weights,
        "params": [asdict(param) for param in solution["params"]],
    }
    metadata_tmp.write_text(json.dumps(metadata), encoding="utf-8")

    os.replace(arrays_tmp, arrays_path)
    os.replace(metadata_tmp, metadata_path)


def load_solution(directory: Path) -> Optional[SolutionDict]:
    """Load a persisted PAT solution from ``directory`` if present."""
    arrays_path = _arrays_path(directory)
    metadata_path = _metadata_path(directory)

    if not arrays_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as data:
        weights = None
        if metadata["has_weights"]:
            weights = np.asarray(data["weights"], dtype=np.float64)

        params = tuple(ParameterSpec(**param) for param in metadata["params"])
        return {
            "V_mV": np.asarray(data["V_mV"], dtype=np.float64),
            "I_exp_nA": np.asarray(data["I_exp_nA"], dtype=np.float64),
            "I_ini_nA": np.asarray(data["I_ini_nA"], dtype=np.float64),
            "I_fit_nA": np.asarray(data["I_fit_nA"], dtype=np.float64),
            "params": params,
            "weights": weights,
            "maxfev": metadata["maxfev"],
        }


def _find_run_fit_pat_gui_script() -> Path:
    current = Path(__file__).resolve()
    while current != current.parent:
        candidate = current / "run_fit_pat_gui.py"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError("Cannot find run_fit_pat_gui.py in any parent directory")


def start_fit_pat_gui_process(
    *,
    voltage_path: Path,
    current_path: Path,
    solution_dir: Path,
    weights_path: Optional[Path] = None,
    maxfev: Optional[int] = None,
    model: str = "pat",
    python_executable: Optional[Path] = None,
) -> subprocess.Popen[bytes]:
    script_path = _find_run_fit_pat_gui_script()
    repo_root = script_path.parent
    python_cmd = python_executable or (repo_root / ".venv/bin/python")

    cmd = [
        str(python_cmd),
        str(script_path),
        "--V",
        str(voltage_path),
        "--I",
        str(current_path),
        "--result-dir",
        str(solution_dir),
    ]
    if weights_path is not None:
        cmd += ["--weights", str(weights_path)]
    if maxfev is not None:
        cmd += ["--maxfev", str(maxfev)]
    cmd += ["--model", model]

    return subprocess.Popen(cmd, start_new_session=True)


def _signal_process_group(process: subprocess.Popen[bytes], sig: int) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def shutdown_fit_pat_gui_process(
    process: subprocess.Popen[bytes],
    *,
    interrupt_timeout: float = 5.0,
    terminate_timeout: float = 2.0,
) -> int:
    if process.returncode is not None:
        return int(process.returncode)

    _signal_process_group(process, signal.SIGINT)
    try:
        return int(process.wait(timeout=interrupt_timeout))
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGTERM)

    try:
        return int(process.wait(timeout=terminate_timeout))
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        return int(process.wait(timeout=terminate_timeout))


def _assert_local_socket_bindable() -> None:
    """Fail fast when the current environment cannot host a local server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except OSError as exc:
        raise RuntimeError(
            "Cannot bind a local TCP socket for the PAT GUI server. "
            "Run the notebook in a local interpreter with localhost access."
        ) from exc


def _reserve_local_port() -> int:
    """Reserve a localhost port number for the child server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server_ready(
    url: str,
    *,
    timeout: float = 10.0,
    interval: float = 0.1,
) -> bool:
    """Poll the server URL until it responds or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 400:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(interval)
    return False


def run_fit_pat_gui_server(
    V_mV,
    I_nA,
    *,
    weights=None,
    maxfev=None,
    model: str = "pat",
    solution_dir: Optional[Path] = None,
    port: int = 0,
    autoreload: bool = False,
    startup_timeout: float = 10.0,
) -> Optional[SolutionDict]:
    """Run the PAT GUI server in-process until interrupted."""
    import panel as pn
    from panel.io.server import get_server

    from .panel import PatFitPanel

    pn.extension("plotly")
    _assert_local_socket_bindable()
    if port == 0:
        port = _reserve_local_port()

    callback = None
    if solution_dir is not None:
        solution_dir = Path(solution_dir)
        clear_solution(solution_dir)

        def callback(solution: Optional[SolutionDict]) -> None:
            if solution is None:
                clear_solution(solution_dir)
            else:
                save_solution(solution_dir, solution)

    panel_widget = PatFitPanel(
        V_mV=V_mV,
        I_nA=I_nA,
        weights=weights,
        maxfev=maxfev,
        model=model,
        on_solution_changed=callback,
    )
    layout = pn.panel(panel_widget.layout, sizing_mode="stretch_both")

    server = get_server(
        layout,
        port=port,
        address="127.0.0.1",
        websocket_origin=[f"127.0.0.1:{port}", f"localhost:{port}"],
        show=False,
        start=False,
        autoreload=autoreload,
    )
    server.start()

    host = server.address or "127.0.0.1"
    root_url = getattr(server, "root_url", "") or ""
    url = getattr(server, "url", f"http://{host}:{server.port}{root_url}")
    startup_error: list[str] = []

    def _open_browser_when_ready() -> None:
        if _wait_for_server_ready(url, timeout=startup_timeout):
            print(
                "Panel server ready at "
                f"{url} (Ctrl+C to stop and return the latest solution)"
            )
            webbrowser.open_new_tab(url)
            return

        startup_error.append(
            "PAT GUI server started but never became reachable at "
            f"{url} within {startup_timeout:.1f} seconds."
        )
        server.io_loop.add_callback(server.io_loop.stop)

    threading.Thread(
        target=_open_browser_when_ready,
        name="pat-gui-browser-opener",
        daemon=True,
    ).start()

    server.run_until_shutdown()

    if startup_error:
        raise RuntimeError(startup_error[0])

    if solution_dir is not None:
        return load_solution(solution_dir)
    return panel_widget._solution
