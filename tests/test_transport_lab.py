from __future__ import annotations

from pathlib import Path

import pytest

pn = pytest.importorskip("panel")

from superconductivity.TransportLab import make_session, transport_lab_apps, workspace_app
from superconductivity.TransportLab import app as transport_lab_app
from superconductivity.utilities.cache import make_cache


def test_transport_lab_apps_define_two_routes(tmp_path: Path) -> None:
    session = make_session(project_path=tmp_path)

    apps = transport_lab_apps(session)

    assert tuple(apps) == ("/visualization", "/workspace")


def test_transport_lab_default_session_uses_repo_projects_without_cache() -> None:
    session = make_session()

    assert session.project_path.name == "projects"
    assert session.project_path.parent.name == "superconductivity"
    assert session.cache is None


def test_workspace_has_cache_and_pipeline_tabs(tmp_path: Path) -> None:
    session = make_session(project_path=tmp_path)

    layout = workspace_app(session)
    tabs = layout.objects[0]

    assert isinstance(tabs, pn.Tabs)
    assert tabs._names == ["Cache", "Evaluation", "Fitting", "Simulation"]


def test_make_session_uses_explicit_active_cache(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)

    session = make_session(cache=cache)

    assert session.project_path == tmp_path
    assert session.cache is cache


def test_serve_transport_lab_opens_visualization_and_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[str] = []
    served: dict[str, object] = {}

    class FakeServer:
        port = 12345

        def stop(self) -> None:
            return None

    class FakePanel:
        @staticmethod
        def serve(apps: dict[str, object], **kwargs: object) -> FakeServer:
            served["apps"] = apps
            served["kwargs"] = kwargs
            return FakeServer()

    monkeypatch.setattr(transport_lab_app, "_import_panel", lambda: FakePanel)
    monkeypatch.setattr(
        transport_lab_app,
        "transport_lab_apps",
        lambda _session: {"/visualization": object(), "/workspace": object()},
    )
    monkeypatch.setattr(transport_lab_app.webbrowser, "open", opened.append)

    server = transport_lab_app.serve_transport_lab(
        project_path=tmp_path,
        open_browser=True,
        threaded=True,
        verbose=False,
    )

    assert isinstance(server, FakeServer)
    assert tuple(served["apps"]) == ("/visualization", "/workspace")
    assert served["kwargs"]["show"] is False
    assert served["kwargs"]["title"] == {
        "/visualization": "Visualization",
        "/workspace": "Workspace",
    }
    assert opened == [
        "http://localhost:12345/visualization",
        "http://localhost:12345/workspace",
    ]
