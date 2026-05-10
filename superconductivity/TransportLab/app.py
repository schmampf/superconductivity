"""Barebone TransportLab Panel application."""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path

from ..utilities.cache import ProjectCache
from .cache_tab import cache_tab
from .evaluation_tab import evaluation_tab
from .fitting_tab import fitting_tab
from .simulation_tab import simulation_tab
from .visualization import visualization_app as _visualization_app

_ACTIVE_TRANSPORT_LAB_SERVER = None
_DEFAULT_TRANSPORT_LAB_PORT = 5010
_VISUALIZATION_TITLE = "Visualization"
_WORKSPACE_TITLE = "Workspace"


def _import_panel():
    """Import and initialize Panel for TransportLab."""
    try:
        import panel as pn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Panel must be installed to run TransportLab.") from exc

    try:
        pn.extension("tabulator")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Failed to initialize Panel extensions.") from exc
    return pn


@dataclass(slots=True)
class TransportLabSession:
    """Minimal shared state for the two TransportLab browser pages."""

    project_path: Path
    cache: ProjectCache | None = None


def make_session(
    *,
    project_path: str | Path | None = None,
    cache: ProjectCache | None = None,
) -> TransportLabSession:
    """Create one minimal TransportLab session."""
    if project_path is None:
        root = (
            cache.path
            if cache is not None
            else Path(__file__).resolve().parents[2] / "projects"
        )
    else:
        root = Path(project_path)
    return TransportLabSession(project_path=root, cache=cache)


def visualization_app(session: TransportLabSession | None = None):
    """Build the barebone visualization page."""
    pn = _import_panel()
    session = session or make_session()
    return _visualization_app(pn, session)


def workspace_app(session: TransportLabSession | None = None):
    """Build the barebone workspace page with cache/pipeline tabs."""
    pn = _import_panel()
    session = session or make_session()
    return pn.Column(
        pn.Tabs(
            ("Cache", cache_tab(pn, session)),
            ("Evaluation", evaluation_tab(pn, session)),
            ("Fitting", fitting_tab(pn, session)),
            ("Simulation", simulation_tab(pn, session)),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )


def transport_lab_apps(
    session: TransportLabSession | None = None,
) -> dict[str, object]:
    """Return Panel apps for the two TransportLab browser routes."""
    session = session or make_session()
    return {
        "/visualization": visualization_app(session),
        "/workspace": workspace_app(session),
    }


def serve_transport_lab(
    *,
    project_path: str | Path | None = None,
    cache: ProjectCache | None = None,
    port: int = _DEFAULT_TRANSPORT_LAB_PORT,
    host: str = "localhost",
    open_browser: bool = True,
    threaded: bool = True,
    verbose: bool = True,
    stop_existing: bool = True,
):
    """Serve TransportLab and optionally open visualization/workspace tabs."""
    global _ACTIVE_TRANSPORT_LAB_SERVER

    pn = _import_panel()
    if (
        stop_existing
        and _ACTIVE_TRANSPORT_LAB_SERVER is not None
        and hasattr(_ACTIVE_TRANSPORT_LAB_SERVER, "stop")
    ):
        _ACTIVE_TRANSPORT_LAB_SERVER.stop()
        _ACTIVE_TRANSPORT_LAB_SERVER = None

    session = make_session(
        project_path=project_path,
        cache=cache,
    )
    apps = transport_lab_apps(session)
    server = pn.serve(
        apps,
        port=port,
        show=False,
        threaded=threaded,
        title={
            "/visualization": _VISUALIZATION_TITLE,
            "/workspace": _WORKSPACE_TITLE,
        },
        verbose=verbose,
    )
    _ACTIVE_TRANSPORT_LAB_SERVER = server

    if open_browser:
        resolved_port = getattr(server, "port", None) or port
        if int(resolved_port) == 0:
            raise RuntimeError(
                "Cannot open TransportLab browser tabs for an automatically "
                "selected port. Pass an explicit port."
            )
        resolved_port = int(resolved_port)
        for route in ("/visualization", "/workspace"):
            webbrowser.open(f"http://{host}:{resolved_port}{route}")
    return server


__all__ = [
    "TransportLabSession",
    "make_session",
    "serve_transport_lab",
    "transport_lab_apps",
    "visualization_app",
    "workspace_app",
]
