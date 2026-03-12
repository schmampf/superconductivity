"""Smoke tests for thesis-oriented waterfall exports."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from superconductivity.visuals.thesis.heatmap import (
    get_thesis_heatmap_matplotlib,
)
from superconductivity.visuals.thesis.latex import (
    compile_thesis_preview,
    export_stacked_waterfall_thesis,
)
from superconductivity.visuals.thesis.surface import (
    get_thesis_surface_matplotlib,
)
from superconductivity.visuals.thesis.waterfall import (
    get_thesis_waterfall_matplotlib,
)


def _waterfall_data(
    nx: int = 41,
    ny: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a smooth stacked spectrum dataset."""
    x = np.linspace(850.0, 1350.0, nx, dtype=np.float64)
    y = np.linspace(600.0, 840.0, ny, dtype=np.float64)
    center = 1080.0 + 90.0 * np.sin((y - 600.0) / 55.0)
    width = 24.0 + 4.0 * np.cos((y - 600.0) / 75.0)
    amplitude = 900.0 + 400.0 * np.cos((y - 600.0) / 40.0)
    z = amplitude[:, None] * np.exp(
        -((x[None, :] - center[:, None]) ** 2) / (2.0 * width[:, None] ** 2)
    )
    return x, y, z.astype(np.float64)


def test_thesis_panel_builders_smoke() -> None:
    """The thesis panel helpers should return native Matplotlib 3D axes."""
    x, y, z = _waterfall_data()

    fig_w, ax_w = get_thesis_waterfall_matplotlib(
        x,
        y,
        z,
        trace_step=3,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )
    fig_s, ax_s = get_thesis_surface_matplotlib(
        x,
        y,
        z,
        trace_step=2,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )
    fig_h, ax_h = get_thesis_heatmap_matplotlib(
        x,
        y,
        z,
        trace_step=2,
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
    )

    assert fig_w is ax_w.figure
    assert fig_s is ax_s.figure
    assert fig_h is ax_h.figure
    assert ax_w.name == "3d"
    assert ax_s.name == "3d"
    assert ax_h.name == "3d"
    assert ax_w.get_zlabel() == "Counts"
    assert ax_s.get_zlabel() == "Counts"
    assert ax_h.get_zlabel() == ""
    assert len(ax_w.lines) == 3
    assert len(ax_h.get_zticks()) == 0


def test_export_stacked_waterfall_thesis_writes_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thesis export helper should write panel and wrapper assets."""
    x, y, z = _waterfall_data()
    saved_files: list[Path] = []

    def fake_savefig(
        self: Figure,
        fname: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"saved:{path.suffix}:{kwargs.get('format', '')}",
            encoding="utf-8",
        )
        saved_files.append(path)

    compiled: dict[str, Path | str] = {}

    def fake_compile(
        preview_tex: str | Path,
        *,
        latex_command: str = "pdflatex",
    ) -> Path:
        tex_path = Path(preview_tex)
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_text("compiled", encoding="utf-8")
        compiled["preview_tex"] = tex_path
        compiled["latex_command"] = latex_command
        return pdf_path

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(
        "superconductivity.visuals.thesis.latex.compile_thesis_preview",
        fake_compile,
    )

    export = export_stacked_waterfall_thesis(
        x,
        y,
        z,
        stem="figures/demo_stack",
        path_pgf=tmp_path / "pgf",
        path_pdf=tmp_path / "pdf",
        xlabel="Emission Wavelength (nm)",
        ylabel="Excitation (arb.)",
        zlabel="Counts",
        compile_preview=True,
        stack_gap="-0.5em",
        latex_command="pdflatex",
        pgf_texsystem="pdflatex",
    )

    shipped_waterfall_pdf = export.stacked_pgf.with_name("demo_stack_waterfall.pdf")
    shipped_surface_pdf = export.stacked_pgf.with_name("demo_stack_surface.pdf")
    shipped_heatmap_pdf = export.stacked_pgf.with_name("demo_stack_heatmap.pdf")
    expected_saved = {
        export.waterfall_pdf,
        export.surface_pdf,
        export.heatmap_pdf,
        shipped_waterfall_pdf,
        shipped_surface_pdf,
        shipped_heatmap_pdf,
    }
    assert expected_saved.issubset(set(saved_files))
    assert export.stacked_pgf.exists()
    assert export.preview_tex.exists()
    assert export.preview_pdf.exists()

    wrapper = export.stacked_pgf.read_text(encoding="utf-8")
    assert "demo_stack_waterfall.pdf" in wrapper
    assert "demo_stack_surface.pdf" in wrapper
    assert "demo_stack_heatmap.pdf" in wrapper
    assert shipped_waterfall_pdf.as_posix() not in wrapper
    assert shipped_surface_pdf.as_posix() not in wrapper
    assert shipped_heatmap_pdf.as_posix() not in wrapper
    assert r"\vspace{-0.5em}" in wrapper
    assert r"\includegraphics[width=\linewidth]" in wrapper

    preview = export.preview_tex.read_text(encoding="utf-8")
    assert r"\documentclass[varwidth,border=2pt]{standalone}" in preview
    assert "demo_stack_waterfall.pdf" in preview
    assert "demo_stack_surface.pdf" in preview
    assert "demo_stack_heatmap.pdf" in preview
    assert r"\includegraphics[width=\linewidth]" in preview
    assert compiled["preview_tex"] == export.preview_tex
    assert compiled["latex_command"] == "pdflatex"


def test_compile_thesis_preview_reports_missing_latex(
    tmp_path: Path,
) -> None:
    """Missing LaTeX commands should raise a clear runtime error."""
    preview_tex = tmp_path / "preview.tex"
    preview_tex.write_text(
        r"\documentclass{standalone}\begin{document}\end{document}",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="was not found"):
        compile_thesis_preview(
            preview_tex,
            latex_command="definitely-not-a-real-latex-command",
        )
