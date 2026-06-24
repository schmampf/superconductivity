from __future__ import annotations

from pathlib import Path

import pytest

pn = pytest.importorskip("panel")

from superconductivity.TransportLab import make_session, transport_lab_apps, workspace_app
from superconductivity.TransportLab import app as transport_lab_app
from superconductivity.TransportLab import visualization as visualization_module
from superconductivity.TransportLab import visualization_app
from superconductivity.utilities.meta import axis, data, param
from superconductivity.utilities.cache import make_cache
from superconductivity.utilities.transport import TransportDatasetSpec


def _make_transport_dataset_1d() -> TransportDatasetSpec:
    return TransportDatasetSpec(
        axes=(axis("V_mV", values=[0.0, 1.0, 2.0], order=0),),
        data=(data("I_nA", [0.0, 2.0, 4.0]),),
        params=(param("Delta_meV", 2.0),),
    )


def _make_transport_dataset_2d() -> TransportDatasetSpec:
    return TransportDatasetSpec(
        axes=(
            axis("T_K", values=[1.0, 2.0], order=0),
            axis("V_mV", values=[0.0, 1.0, 2.0], order=1),
        ),
        data=(data("I_nA", [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]),),
    )


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


def test_visualization_dataset_frame_lists_transport_entries(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset_1d()
    cache.answer = 42
    session = make_session(cache=cache)

    frame = visualization_module._dataset_frame(session)

    assert frame["key"].tolist() == ["exp_v"]


def test_visualization_quantity_frames_filter_by_shape_and_axis_order() -> None:
    dataset = _make_transport_dataset_2d()

    z_frame = visualization_module._z_frame(dataset)
    x_frame = visualization_module._axis_frame(dataset, order=1)
    y_frame = visualization_module._axis_frame(dataset, order=0)

    assert "I_nA" in z_frame["key"].tolist()
    assert "dG_uS" in z_frame["key"].tolist()
    assert "V_mV" not in z_frame["key"].tolist()
    assert x_frame["key"].tolist() == ["V_mV"]
    assert y_frame["key"].tolist() == ["T_K"]


def test_visualization_builds_trace_figure() -> None:
    figure = visualization_module._figure_for_dataset(
        _make_transport_dataset_1d(),
        "trace",
        "V_mV",
        None,
        "I_nA",
    )

    assert figure.data[0].type == "scatter"


def test_visualization_builds_heatmap_and_surface_figures() -> None:
    dataset = _make_transport_dataset_2d()

    heatmap = visualization_module._figure_for_dataset(
        dataset,
        "heatmap",
        "V_mV",
        "T_K",
        "I_nA",
    )
    surface = visualization_module._figure_for_dataset(
        dataset,
        "surface",
        "V_mV",
        "T_K",
        "I_nA",
    )

    assert heatmap.data[0].type == "heatmap"
    assert surface.data[0].type == "surface"
    assert heatmap.layout.xaxis.title.text == "<i>V</i> (mV)"
    assert heatmap.layout.yaxis.title.text == "<i>T</i> (K)"
    assert heatmap.data[0].colorbar.title.text == "<i>I</i> (nA)"


def test_visualization_uses_indices_when_axes_are_missing_or_incompatible() -> None:
    dataset = _make_transport_dataset_2d()

    x, y, z = visualization_module._xyz_values(
        dataset,
        x_key=None,
        y_key=None,
        z_key="I_nA",
    )
    x_bad, y_bad, _ = visualization_module._xyz_values(
        dataset,
        x_key="T_K",
        y_key="V_mV",
        z_key="I_nA",
    )

    assert z.shape == (2, 3)
    assert x.tolist() == [0.0, 1.0, 2.0]
    assert y.tolist() == [0.0, 1.0]
    assert x_bad.tolist() == [0.0, 1.0, 2.0]
    assert y_bad.tolist() == [0.0, 1.0]

    figure = visualization_module._figure_for_dataset(
        dataset,
        "heatmap",
        "T_K",
        "V_mV",
        "I_nA",
    )

    assert figure.layout.xaxis.title.text == "x index"
    assert figure.layout.yaxis.title.text == "y index"


def test_visualization_app_builds_with_active_cache(tmp_path: Path) -> None:
    cache = make_cache("demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset_1d()
    session = make_session(cache=cache)

    layout = visualization_app(session)

    assert len(layout.objects) == 4


def test_visualization_updates_when_active_cache_changes(tmp_path: Path) -> None:
    session = make_session(project_path=tmp_path)
    layout = visualization_app(session)
    dataset_table = layout.objects[0].objects[0]

    cache = make_cache("demo", path=tmp_path)
    cache.exp_v = _make_transport_dataset_1d()
    session.cache = cache
    session.notify_cache_changed()

    assert dataset_table.value["key"].tolist() == ["exp_v"]
