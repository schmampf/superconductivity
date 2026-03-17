"""LaTeX/PDF export helpers for thesis waterfall panel stacks."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

try:
    from IPython import get_ipython
    from IPython.display import FileLink, Image, display
except ImportError:  # pragma: no cover - optional notebook integration.
    FileLink = None
    Image = None
    display = None

    def get_ipython() -> None:
        """Return ``None`` when IPython is not available."""
        return None

from superconductivity.style.cpd4 import cmap as default_cmap
from superconductivity.utilities.types import LIM, NDArray64

from ._common import prepare_waterfall_data, select_trace_rows
from .heatmap import get_thesis_heatmap_matplotlib
from .surface import get_thesis_surface_matplotlib
from .waterfall import get_thesis_waterfall_matplotlib

_THESIS_MPLSTYLE = (
    Path(__file__).resolve().parents[2] / "style" / "thesisstyle.mplstyle"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CRYOLAB_ROOT = _REPO_ROOT.parent
_DEFAULT_LOCAL_DIR = _REPO_ROOT / "thesis"
_DEFAULT_REMOTE_DIR = _CRYOLAB_ROOT / "dissertation"
_STACK_PGF_PREAMBLE_LINES = (
    r"\usepackage{xunicode}",
    r"\usepackage{fontspec}",
    r"\defaultfontfeatures{Mapping=tex-text}",
    r"\setmainfont{Arial}",
    r"\setsansfont{Arial}",
    r"\usepackage{cmbright}",
)
_STACK_BASE_FONT_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join(_STACK_PGF_PREAMBLE_LINES),
}


def _build_stack_font_rcparams(*, subfigure: bool) -> dict[str, object]:
    """Build the stack font rcParams for normal or subfigure usage."""
    rc_params = dict(_STACK_BASE_FONT_RCPARAMS)
    if subfigure:
        rc_params.update(
            {
                "font.size": 8.0,
                "axes.labelsize": 7.0,
                "xtick.labelsize": 7.0,
                "ytick.labelsize": 7.0,
                "legend.fontsize": 7.0,
            }
        )
    return rc_params


@dataclass(frozen=True)
class StackedThesisExport:
    """Resolved output paths for a thesis waterfall stack export.

    Attributes
    ----------
    name
        Output bundle name without a file suffix.
    stack_dir
        Local bundle directory containing the panel PGFs and `main.*`.
    waterfall_pgf
        PGF export of the waterfall panel.
    surface_pgf
        PGF export of the surface panel.
    heatmap_pgf
        PGF export of the heatmap panel.
    main_pgf
        Thesis-importable PGF wrapper stacking the three panel PGFs.
    main_png
        Notebook-friendly stacked PNG preview.
    remote_stack_dir
        Optional shipped PGF bundle directory under the remote directory.
    """

    name: Path
    stack_dir: Path
    waterfall_pgf: Path
    surface_pgf: Path
    heatmap_pgf: Path
    main_pgf: Path
    main_png: Path
    remote_stack_dir: Path | None


def _normalize_name(name: str | Path) -> Path:
    """Normalize an output bundle name by stripping a file suffix if present.

    Parameters
    ----------
    name
        Output name which may optionally already contain a suffix.

    Returns
    -------
    Path
        Normalized path fragment without a suffix.
    """
    path = Path(name)
    return path.with_suffix("") if path.suffix else path


def _bundle_file_path(
    stack_dir: Path,
    filename: str,
) -> Path:
    """Build one path inside a local stack bundle directory."""
    return stack_dir / filename


def _resolve_stack_name(
    *,
    name: str | Path,
) -> Path:
    """Validate and normalize one stack bundle name."""
    name_path = _normalize_name(name)
    if name_path.is_absolute() or len(name_path.parts) != 1:
        raise ValueError("name must be a single relative path component.")
    return name_path


def _build_export_paths(
    *,
    name: str | Path,
    sub_dir: str | Path,
    local_dir: str | Path,
    remote_dir: str | Path | None,
) -> StackedThesisExport:
    """Resolve all filesystem paths used by the stack export.

    Parameters
    ----------
    name
        Shared bundle name for the generated local and remote bundle
        directories.
    sub_dir
        Relative subdirectory under the local and remote roots.
    local_dir
        Base directory receiving the local stack bundle.
    remote_dir
        Optional base directory receiving the shipped PGF bundle.

    Returns
    -------
    StackedThesisExport
        Dataclass containing all resolved output paths.
    """
    name_path = _resolve_stack_name(name=name)
    sub_path = Path(sub_dir)
    if sub_path.is_absolute():
        raise ValueError("sub_dir must be relative.")

    local_root = Path(local_dir).expanduser().resolve()
    stack_dir = (local_root / sub_path / name_path).resolve()
    remote_stack_dir = None
    if remote_dir is not None:
        remote_root = Path(remote_dir).expanduser().resolve()
        remote_stack_dir = (remote_root / sub_path / name_path).resolve()

    return StackedThesisExport(
        name=name_path,
        stack_dir=stack_dir,
        waterfall_pgf=_bundle_file_path(stack_dir, "waterfall.pgf"),
        surface_pgf=_bundle_file_path(stack_dir, "surface.pgf"),
        heatmap_pgf=_bundle_file_path(stack_dir, "heatmap.pgf"),
        main_pgf=_bundle_file_path(stack_dir, "main.pgf"),
        main_png=_bundle_file_path(stack_dir, "main.png"),
        remote_stack_dir=remote_stack_dir,
    )


def _ensure_parent(path: Path) -> None:
    """Ensure the parent directory of a path exists.

    Parameters
    ----------
    path
        Path whose parent directory should be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_local_stack_dir(export: StackedThesisExport) -> None:
    """Remove any previous local bundle before creating new outputs."""
    if export.stack_dir.exists():
        shutil.rmtree(export.stack_dir)
    export.stack_dir.mkdir(parents=True, exist_ok=True)


def _prepare_remote_stack_dir(export: StackedThesisExport) -> None:
    """Remove any previous shipped bundle before creating new outputs."""
    remote_stack_dir = export.remote_stack_dir
    if remote_stack_dir is None:
        return

    if remote_stack_dir.exists():
        shutil.rmtree(remote_stack_dir)


def _save_figure_pgf(
    fig: plt.Figure,
    *,
    pgf_path: Path,
    dpi: float,
) -> None:
    """Save one Matplotlib figure as PGF.

    Parameters
    ----------
    fig
        Figure to save.
    pgf_path
        Destination PGF file.
    dpi
        Save DPI used for rasterized artists embedded in the PGF sidecars.
    """
    _ensure_parent(pgf_path)
    fig.savefig(
        pgf_path,
        format="pgf",
        backend="pgf",
        transparent=True,
        facecolor="none",
        edgecolor="none",
        dpi=float(dpi),
    )


def _iter_pgf_sidecar_images(pgf_path: Path) -> tuple[Path, ...]:
    """Return raster sidecars referenced by one exported PGF file."""
    pattern = f"{pgf_path.stem}-img*"
    return tuple(
        sorted(
            path for path in pgf_path.parent.glob(pattern) if path.is_file()
        )
    )


def _ship_remote_pgf_bundle(export: StackedThesisExport) -> None:
    """Copy the remote PGF bundle required for thesis-side imports."""
    remote_stack_dir = export.remote_stack_dir
    if remote_stack_dir is None:
        return

    if remote_stack_dir.exists():
        shutil.rmtree(remote_stack_dir)
    remote_stack_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = (
        export.main_pgf,
        export.waterfall_pgf,
        export.surface_pgf,
        export.heatmap_pgf,
        *_iter_pgf_sidecar_images(export.waterfall_pgf),
        *_iter_pgf_sidecar_images(export.surface_pgf),
        *_iter_pgf_sidecar_images(export.heatmap_pgf),
    )
    for source in files_to_copy:
        shutil.copy2(source, remote_stack_dir / source.name)


def _render_preview_png(
    pdf_path: Path,
    *,
    png_path: Path,
    dpi: float = 200.0,
) -> Path:
    """Render the first PDF page into a PNG preview using Ghostscript."""
    input_pdf = Path(pdf_path).expanduser().resolve()
    output_png = Path(png_path).expanduser().resolve()
    _ensure_parent(output_png)
    if output_png.exists():
        output_png.unlink()

    command = [
        "gs",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pngalpha",
        f"-r{float(dpi):g}",
        "-dFirstPage=1",
        "-dLastPage=1",
        f"-sOutputFile={output_png}",
        str(input_pdf),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Ghostscript command 'gs' was not found.") from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or exc.stderr or ""
        raise RuntimeError(
            "PNG preview rendering failed.\n" + output.strip()
        ) from exc

    return output_png


def _display_png_preview(
    png_path: Path,
    *,
    height: int = 720,
) -> None:
    """Display a PNG preview inline when running in IPython."""
    if (
        get_ipython() is None
        or FileLink is None
        or display is None
        or Image is None
    ):
        return

    preview_path = Path(png_path).expanduser().resolve()
    try:
        display(Image(filename=str(preview_path), height=int(height)))
    except Exception:  # pragma: no cover - notebook frontend dependent.
        pass
    display(FileLink(str(preview_path)))


def _relative_posix_path(path: Path, start: Path) -> str:
    """Return a relative path formatted for LaTeX ``\\input``.

    Parameters
    ----------
    path
        Target file path.
    start
        Directory from which the relative path should be computed.

    Returns
    -------
    str
        Relative path using forward slashes.
    """
    return Path(os.path.relpath(path, start)).as_posix()


def _resolve_panel_zoom(
    box_zoom: float,
    *,
    override: float | None,
) -> float:
    """Resolve one panel zoom from a shared default and an override."""
    return float(box_zoom) if override is None else float(override)


def _resolve_waterfall_traces(
    y: NDArray64,
    z: NDArray64,
    *,
    waterfall_traces: int | Sequence[float] | None,
    trace_step: int,
) -> tuple[NDArray64, NDArray64, int, int | None]:
    """Resolve waterfall trace selection for count or nearest y-values."""
    if waterfall_traces is None:
        return y, z, int(trace_step), None

    if isinstance(waterfall_traces, (int, np.integer)) and not isinstance(
        waterfall_traces, bool
    ):
        return y, z, int(trace_step), int(waterfall_traces)

    if isinstance(waterfall_traces, (str, bytes)):
        raise ValueError(
            "waterfall_traces must be an int or a sequence of y-values.",
        )

    values = np.asarray(tuple(waterfall_traces), dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("waterfall_traces must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("waterfall_traces y-values must be finite.")

    matched_indices: list[int] = []
    for value in values:
        matched_indices.append(int(np.argmin(np.abs(y - float(value)))))

    keep = np.asarray(sorted(set(matched_indices)), dtype=int)
    return y[keep], z[keep], 1, None


def _resolve_surface_trace_overlay(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
    *,
    ylim: LIM,
    trace_step: int,
    trace_count: int | None,
) -> tuple[NDArray64, NDArray64]:
    """Resolve the actual waterfall rows that should be overlaid."""
    _, y_sel, z_sel = prepare_waterfall_data(
        x=x,
        y=y,
        z=z,
        ylim=ylim,
        trace_step=trace_step,
    )
    return select_trace_rows(
        y_sel,
        z_sel,
        trace_count=trace_count,
    )


def _resolve_axis_triplet(
    axis: Sequence[bool],
    *,
    name: str,
) -> tuple[bool, bool, bool]:
    """Validate one ``(waterfall, surface, heatmap)`` axis tuple."""
    values = tuple(bool(value) for value in axis)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three booleans.")
    return values[0], values[1], values[2]


def _format_latex_length(
    value: str | float | int,
    *,
    name: str,
) -> str:
    """Format one LaTeX length from a string or inches value."""
    if isinstance(value, str):
        return value

    length = float(value)
    if not np.isfinite(length):
        raise ValueError(f"{name} must be finite.")
    return f"{length}in"


def _resolve_standalone_figsize(
    figsize: Sequence[str | float | int] | None,
) -> tuple[str, str] | None:
    """Resolve the optional standalone stack size in LaTeX lengths."""
    if figsize is None:
        return None

    values = tuple(figsize)
    if len(values) != 2:
        raise ValueError("figsize must contain exactly two values.")
    return (
        _format_latex_length(values[0], name="figsize[0]"),
        _format_latex_length(values[1], name="figsize[1]"),
    )


def _resolve_panel_figsize(
    subfigsize: Sequence[str | float | int],
) -> tuple[str, str]:
    """Resolve the per-panel figure size in LaTeX lengths."""
    values = tuple(subfigsize)
    if len(values) != 2:
        raise ValueError("subfigsize must contain exactly two values.")
    return (
        _format_latex_length(values[0], name="subfigsize[0]"),
        _format_latex_length(values[1], name="subfigsize[1]"),
    )


def _resolve_numeric_inch_length(
    value: str | float | int,
    *,
    name: str,
) -> float:
    """Resolve one numeric inch length for automatic panel placement."""
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*in\s*",
            value,
        )
        if match is None:
            raise ValueError(
                f"{name} must be numeric inches when used for auto layout."
            )
        return float(match.group(1))

    length = float(value)
    if not np.isfinite(length):
        raise ValueError(f"{name} must be finite.")
    return length


def _resolve_numeric_figsize(
    figsize: Sequence[str | float | int] | None,
) -> tuple[float, float] | None:
    """Resolve the optional stack canvas size in numeric inches."""
    if figsize is None:
        return None

    values = tuple(figsize)
    if len(values) != 2:
        raise ValueError("figsize must contain exactly two values.")
    return (
        _resolve_numeric_inch_length(values[0], name="figsize[0]"),
        _resolve_numeric_inch_length(values[1], name="figsize[1]"),
    )


def _resolve_panel_position_triplet(
    values: Sequence[float | int | None],
    *,
    default: Sequence[float],
    name: str,
) -> tuple[float, float, float]:
    """Resolve one ``(waterfall, surface, heatmap)`` position tuple."""
    entries = tuple(values)
    if len(entries) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    defaults = tuple(float(value) for value in default)
    if len(defaults) != 3:
        raise ValueError(f"{name} default must contain exactly three values.")

    resolved: list[float] = []
    for index, (value, fallback) in enumerate(zip(entries, defaults, strict=True)):
        if value is None:
            resolved.append(fallback)
            continue

        position = float(value)
        if not np.isfinite(position):
            raise ValueError(f"{name}[{index}] must be finite.")
        resolved.append(position)

    return tuple(resolved)  # type: ignore[return-value]


def _default_panel_positions(
    *,
    subfigsize: tuple[float, float],
    canvas_height: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return default absolute panel positions inside the stack canvas."""
    return (
        (0.0, 0.0, 0.0),
        (
            2.0 * canvas_height / 3.0,
            canvas_height / 3.0,
            0.0,
        ),
    )


def _resolve_panel_positions(
    *,
    posx: Sequence[float | int | None] | None,
    posy: Sequence[float | int | None] | None,
    subfigsize: tuple[float, float],
    figsize: Sequence[str | float | int] | None,
) -> tuple[tuple[str, str, str], tuple[str, str, str], float, float]:
    """Resolve panel positions and the natural stack canvas size."""
    panel_width = float(subfigsize[0])
    panel_height = float(subfigsize[1])
    numeric_figsize = _resolve_numeric_figsize(figsize)
    if numeric_figsize is None:
        canvas_height = 3.0 * panel_height
    else:
        canvas_height = numeric_figsize[1]

    default_posx, default_posy = _default_panel_positions(
        subfigsize=subfigsize,
        canvas_height=canvas_height,
    )
    resolved_posx = _resolve_panel_position_triplet(
        (None, None, None) if posx is None else posx,
        default=default_posx,
        name="posx",
    )
    resolved_posy = _resolve_panel_position_triplet(
        (None, None, None) if posy is None else posy,
        default=default_posy,
        name="posy",
    )
    if numeric_figsize is None:
        min_x = min(resolved_posx)
        min_y = min(resolved_posy)
        translated_posx = tuple(value - min_x for value in resolved_posx)
        translated_posy = tuple(value - min_y for value in resolved_posy)
        natural_width = max(resolved_posx) - min_x + panel_width
        natural_height = max(resolved_posy) - min_y + panel_height
    else:
        translated_posx = resolved_posx
        translated_posy = resolved_posy
        natural_width = max(resolved_posx) + panel_width
        natural_height = max(resolved_posy) + panel_height

    return (
        tuple(f"{value:g}in" for value in translated_posx),
        tuple(f"{value:g}in" for value in translated_posy),
        natural_width,
        natural_height,
    )


def _save_stack_preview_png(
    *,
    main_pgf: Path,
    png_path: Path,
    figsize: Sequence[str | float | int] | None,
    dpi: float,
    latex_command: str | None = None,
) -> Path:
    """Compile a temporary stacked preview and rasterize it to PNG."""
    resolved_main_pgf = Path(main_pgf).expanduser().resolve()
    _ensure_parent(png_path)
    with tempfile.TemporaryDirectory(
        prefix=".thesis-stack-preview-",
        dir=resolved_main_pgf.parent,
    ) as tmp_dir:
        temp_root = Path(tmp_dir)
        for source in sorted(resolved_main_pgf.parent.iterdir()):
            if not source.is_file():
                continue
            if source.suffix == ".pgf" or source.name.endswith(".png"):
                shutil.copy2(source, temp_root / source.name)

        temp_main_pgf = temp_root / resolved_main_pgf.name
        temp_tex = temp_root / "main.tex"
        temp_tex.write_text(
            build_preview_tex(
                main_pgf=temp_main_pgf,
                main_tex=temp_tex,
                figsize=figsize,
            ),
            encoding="utf-8",
        )
        temp_pdf = compile_thesis_preview(
            temp_tex,
            latex_command=latex_command,
        )
        return _render_preview_png(
            temp_pdf,
            png_path=png_path,
            dpi=float(dpi),
        )


def build_stack_pgf(
    *,
    waterfall_pgf: Path,
    surface_pgf: Path,
    heatmap_pgf: Path,
    main_pgf: Path,
    subfigsize: tuple[float, float],
    posx: Sequence[float | int | None] | None = (0.0, 0.0, 0.0),
    posy: Sequence[float | int | None] | None = (None, None, None),
    figsize: Sequence[str | float | int] | None = None,
) -> str:
    """Build a thesis-importable PGF wrapper from the panel PGFs.

    Parameters
    ----------
    waterfall_pgf, surface_pgf, heatmap_pgf
        Panel PGFs stacked inside the wrapper.
    main_pgf
        Output wrapper path used for relative path resolution.
    subfigsize
        Outer panel size ``(width, height)`` used for each exported panel.
    posx
        Absolute x positions ``(waterfall, surface, heatmap)`` in inches.
        ``None`` entries fall back to the default left-aligned value ``0``.
        Negative values are allowed.
    posy
        Absolute y positions ``(waterfall, surface, heatmap)`` in inches.
        ``None`` entries divide the stack canvas height into three equal
        slots and place ``(waterfall, surface, heatmap)`` at those bottoms.
        Negative values are allowed.
    figsize
        Optional outer stack canvas size ``(width, height)``. When provided,
        the wrapper establishes its own minipage so that stacking also works
        reliably when imported inside a macro argument such as
        ``\\subfigure{...}``.
    """
    waterfall_rel = _relative_posix_path(waterfall_pgf, main_pgf.parent)
    surface_rel = _relative_posix_path(surface_pgf, main_pgf.parent)
    heatmap_rel = _relative_posix_path(heatmap_pgf, main_pgf.parent)
    panel_width, panel_height = _resolve_panel_figsize(subfigsize)
    resolved_posx, resolved_posy, natural_width, natural_height = (
        _resolve_panel_positions(
            posx=posx,
            posy=posy,
            subfigsize=subfigsize,
            figsize=figsize,
        )
    )
    resolved_figsize = _resolve_standalone_figsize(figsize)
    if resolved_figsize is None:
        stack_width = f"{natural_width:g}in"
        stack_height = f"{natural_height:g}in"
    else:
        stack_width, stack_height = resolved_figsize

    lines = [r"\begingroup"]
    if resolved_figsize is not None:
        lines.append(
            rf"\begin{{minipage}}[t][{stack_height}][t]{{{stack_width}}}"
        )
    lines.extend(
        [
            rf"\def\thesispanelwidth{{{panel_width}}}",
            rf"\def\thesispanelheight{{{panel_height}}}",
            rf"\def\thesisstackwidth{{{stack_width}}}",
            rf"\def\thesisstackheight{{{stack_height}}}",
            rf"\def\thesisposxwaterfall{{{resolved_posx[0]}}}",
            rf"\def\thesisposxsurface{{{resolved_posx[1]}}}",
            rf"\def\thesisposxheatmap{{{resolved_posx[2]}}}",
            rf"\def\thesisposywaterfall{{{resolved_posy[0]}}}",
            rf"\def\thesisposysurface{{{resolved_posy[1]}}}",
            rf"\def\thesisposyheatmap{{{resolved_posy[2]}}}",
            r"\begin{pgfpicture}",
            r"\pgfpathrectangle{\pgfpointorigin}{"
            r"\pgfpoint{\thesisstackwidth}{\thesisstackheight}}",
            r"\pgfusepath{use as bounding box}",
            r"\pgftext[left,bottom,at={"
            r"\pgfpoint{\thesisposxheatmap}{\thesisposyheatmap}}"
            rf"]{{\input{{{heatmap_rel}}}}}",
            r"\pgftext[left,bottom,at={"
            r"\pgfpoint{\thesisposxsurface}{\thesisposysurface}}"
            rf"]{{\input{{{surface_rel}}}}}",
            r"\pgftext[left,bottom,at={"
            r"\pgfpoint{\thesisposxwaterfall}{\thesisposywaterfall}}"
            rf"]{{\input{{{waterfall_rel}}}}}",
            r"\end{pgfpicture}",
        ]
    )
    if resolved_figsize is not None:
        lines.append(r"\end{minipage}")
    lines.extend([r"\endgroup", ""])
    return "\n".join(lines)


def _build_preview_preamble_lines() -> list[str]:
    """Build the standalone LaTeX preamble for compiled stack previews."""
    return [
        r"\usepackage{iftex}",
        r"\ifPDFTeX",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{cmbright}",
        r"\else",
        *_STACK_PGF_PREAMBLE_LINES,
        r"\fi",
        r"\usepackage{pgf}",
        r"\usepackage{graphicx}",
        r"\usepackage{varwidth}",
    ]


def build_preview_tex(
    *,
    main_pgf: Path,
    main_tex: Path,
    figsize: Sequence[str | float | int] | None = None,
) -> str:
    """Build a standalone preview document from the stacked PGF wrapper.

    Parameters
    ----------
    main_pgf
        PGF wrapper stacking the panel PGFs.
    main_tex
        Standalone `.tex` path. The wrapper path is made relative to its
        parent directory.
    figsize
        Optional standalone canvas size ``(width, height)``. The actual outer
        box is now emitted by ``main.pgf``; this argument only controls the
        standalone document border size. It is accepted here for API symmetry.

    Returns
    -------
    str
        Standalone LaTeX document content.
    """
    main_rel = _relative_posix_path(main_pgf, main_tex.parent)
    resolved_figsize = _resolve_standalone_figsize(figsize)
    border = "0pt" if resolved_figsize is not None else "2pt"
    lines = [
        rf"\documentclass[varwidth,border={border}]{{standalone}}",
        *_build_preview_preamble_lines(),
    ]
    lines.extend(
        [
            r"\pagestyle{empty}",
            r"\begin{document}",
        ]
    )
    if resolved_figsize is None:
        lines.extend(
            [
                rf"\input{{{main_rel}}}",
            ]
        )
    else:
        lines.extend(
            [
                rf"\input{{{main_rel}}}",
            ]
        )
    lines.extend(
        [
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_latex_command(
    *,
    latex_command: str | None,
    pgf_texsystem: str | None,
) -> str:
    """Resolve the LaTeX engine used for preview compilation.

    Parameters
    ----------
    latex_command
        Explicit LaTeX command requested by the caller.
    pgf_texsystem
        Explicit Matplotlib PGF tex system requested by the caller.

    Returns
    -------
    str
        LaTeX engine name to use for compilation.
    """
    if latex_command is not None:
        return latex_command
    if pgf_texsystem is not None:
        return pgf_texsystem
    return str(matplotlib.rcParams["pgf.texsystem"])


def compile_thesis_preview(
    main_tex: str | Path,
    *,
    latex_command: str | None = None,
) -> Path:
    """Compile a standalone thesis stack document into PDF.

    Parameters
    ----------
    main_tex
        Standalone LaTeX file produced by ``build_preview_tex``.
    latex_command
        LaTeX command used for compilation. When omitted, the current
        Matplotlib ``pgf.texsystem`` setting is used.

    Returns
    -------
    Path
        Compiled stacked PDF path.

    Raises
    ------
    RuntimeError
        If the LaTeX compiler is missing or returns a non-zero exit code.
    """
    def _cleanup_preview_auxiliary_files(tex_file: Path) -> None:
        """Remove standalone build byproducts after a successful compile."""
        aux_suffixes = (
            ".aux",
            ".log",
            ".out",
            ".xdv",
            ".synctex.gz",
        )
        for suffix in aux_suffixes:
            aux_path = tex_file.with_suffix(suffix)
            if aux_path.exists():
                aux_path.unlink()

    tex_path = Path(main_tex).expanduser().resolve()
    command_name = _resolve_latex_command(
        latex_command=latex_command,
        pgf_texsystem=None,
    )
    command = [
        command_name,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]

    try:
        subprocess.run(
            command,
            cwd=tex_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"LaTeX command {command_name!r} was not found.") from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or exc.stderr or ""
        raise RuntimeError(
            "LaTeX preview compilation failed.\n" + output.strip()
        ) from exc

    _cleanup_preview_auxiliary_files(tex_path)
    return tex_path.with_suffix(".pdf")


def export_stacked_waterfall_thesis(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
    *,
    name: str | Path,
    sub_dir: str | Path = "",
    local_dir: str | Path = _DEFAULT_LOCAL_DIR,
    remote_dir: str | Path | None = _DEFAULT_REMOTE_DIR,
    xlim: LIM = None,
    ylim: LIM = None,
    zlim: LIM = None,
    clim: LIM = None,
    trace_step: int = 1,
    surface_trace_step: int = 1,
    heatmap_trace_step: int = 1,
    surface_x_oversample: int = 10,
    waterfall_traces: int | Sequence[float] | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    x_axis: Sequence[bool] = (True, True, True),
    y_axis: Sequence[bool] = (True, True, True),
    z_axis: Sequence[bool] = (True, True, False),
    z_axis_side: str = "default",
    invert_xaxis: bool = False,
    invert_yaxis: bool = True,
    labelspacing: float | Sequence[float] | None = None,
    ticklabelspacing: float | Sequence[float] | None = None,
    ticks: Sequence[object | None] | None = None,
    ticklabels: Sequence[object | None] | None = None,
    clabel: str | None = None,
    line_color: str = "black",
    line_width: float = 0.6,
    line_alpha: float = 1.0,
    surface_alpha: float = 1.0,
    surface_shading: bool = True,
    surface_light_azdeg: float = 315.0,
    surface_light_altdeg: float = 40.0,
    surface_shade_strength: float = 0.5,
    surface_rasterized: bool = True,
    heatmap_alpha: float = 1.0,
    heatmap_mode: str = "warped_image",
    heatmap_z_level: float | None = None,
    heatmap_cell_overlap: float = 0.01,
    cmap_mpl: ListedColormap = default_cmap(),
    box_aspect: Sequence[float] | None = (1.0, 1.0, 1.0),
    box_zoom: float = 1.0,
    waterfall_box_zoom: float | None = None,
    surface_box_zoom: float | None = None,
    heatmap_box_zoom: float | None = None,
    elev: float = 20.0,
    azim: float = -70.0,
    subfigsize: tuple[float, float] = (6.2, 3.9),
    subfigure: bool = True,
    axes_rect: Sequence[float] | None = None,
    posx: Sequence[float | int | None] | None = (0.0, 0.0, 0.0),
    posy: Sequence[float | int | None] | None = (None, None, None),
    figsize: Sequence[str | float | int] | None = None,
    pdf_dpi: float = 1800.0,
    surface_pdf_dpi: float | None = None,
    show_preview: bool = True,
    preview_height: int = 720,
    preview_png_dpi: float = 200.0,
    latex_command: str | None = None,
    pgf_texsystem: str | None = "xelatex",
) -> StackedThesisExport:
    """Export a three-panel waterfall stack for LaTeX-based composition.

    Parameters
    ----------
    x
        1D x-axis values of shape ``(Nx,)``.
    y
        1D sweep values of shape ``(Ny,)``.
    z
        2D measurement array of shape ``(Ny, Nx)`` with ``z[y_i, x_j]``.
    name
        Bundle name used for the local stack directory and the shipped remote
        PGF bundle directory.
    sub_dir
        Relative subdirectory under the local and remote roots.
    local_dir
        Local root directory receiving the stack bundle. Defaults to the
        `thesis/` directory in the superconductivity repo.
    remote_dir
        Optional remote root directory receiving the shipped `{name}/` PGF
        bundle. Defaults to the dissertation repo root. Pass ``None`` to skip
        the remote copy.
    xlim, ylim, zlim
        Optional axis limits shared by the exported panels.
    clim
        Optional color limits for the surface and heatmap panels.
    trace_step
        Plot every ``trace_step``-th fixed-``y`` trace in the waterfall
        panel.
    surface_trace_step
        Keep every ``surface_trace_step``-th fixed-``y`` row in the surface
        panel. Defaults to ``1`` so the native surface mesh is preserved.
    heatmap_trace_step
        Keep every ``heatmap_trace_step``-th fixed-``y`` row in the heatmap
        panel. Defaults to ``1`` so the native heatmap bins are preserved.
    surface_x_oversample
        Densify the surface mesh along x by subdividing each original x
        interval into ``surface_x_oversample`` pieces. The default is
        ``10``; use ``1`` to keep the native x grid.
    waterfall_traces
        Waterfall trace selection. Pass an integer to keep the old behavior of
        drawing that many traces after applying ``trace_step``. Pass a
        sequence of y-values to draw the closest corresponding waterfall rows
        instead. Explicit y-value selection bypasses ``trace_step`` for the
        waterfall panel. Whenever this parameter is set, the same selected
        traces are also overlaid on the surface panel.
    xlabel, ylabel, zlabel
        Axis labels.
    x_axis, y_axis, z_axis
        Visibility tuples ``(waterfall, surface, heatmap)`` controlling axis
        labels and ticks for the exported panels.
    z_axis_side
        Visual side used for the z-axis label and tick labels. Use
        ``"default"``, ``"left"``, or ``"right"``.
    invert_xaxis
        Whether to display the x-axis in descending order across all panels.
    invert_yaxis
        Whether to display the y-axis in descending order across all panels.
    labelspacing
        Optional scalar or ``(x, y, z)`` spacing where ``0`` places labels on
        the axis line and ``1`` matches Matplotlib's default 3D spacing.
        Values below ``0`` and above ``1`` are allowed.
    ticklabelspacing
        Optional scalar or ``(x, y, z)`` spacing where ``0`` places tick
        labels on the axis line and ``1`` matches Matplotlib's default 3D
        spacing. Values below ``0`` and above ``1`` are allowed.
    ticks
        Optional shared ``(x, y, z)`` tick tuple. Each entry may be a tick
        sequence or ``None`` to leave that axis unchanged.
    ticklabels
        Optional shared ``(x, y, z)`` tick-label tuple. Each entry may be a
        label sequence or ``None`` to leave that axis unchanged.
    clabel
        Optional colorbar label. Defaults to ``zlabel``.
    line_color
        Waterfall line color.
    line_width
        Waterfall line width.
    line_alpha
        Waterfall line opacity.
    surface_alpha
        Surface panel opacity.
    surface_shading
        Whether to modulate the surface colormap with light-based shading.
    surface_light_azdeg
        Azimuth of the virtual surface light source in degrees.
    surface_light_altdeg
        Altitude of the virtual surface light source in degrees.
    surface_shade_strength
        Blend factor between pure height coloring and the shaded variant.
    surface_rasterized
        Whether the surface data layer should be rasterized while keeping
        axes, ticks, and labels as vector output.
    heatmap_alpha
        Heatmap panel opacity.
    heatmap_mode
        Rendering mode for the heatmap panel. Use ``"warped_image"`` for a
        perspective-warped RGBA image or ``"vector_cells"`` for vector quads.
    heatmap_z_level
        Z position used for the flat heatmap plane. When omitted, the heatmap
        plane is anchored at the lower bound of the resolved z-range.
    heatmap_cell_overlap
        Relative overlap added between neighboring heatmap cells to hide
        vector-PDF seam artifacts. ``0`` keeps exact cell edges.
    cmap_mpl
        Matplotlib colormap used for the surface and heatmap panels.
    box_aspect
        Displayed box aspect ratio ``(x_ratio, y_ratio, z_ratio)`` shared by
        the exported panels. Use ``None`` to scale from the visible data
        ranges. The default is ``(1, 1, 1)``.
    box_zoom
        Shared overall scale of the rendered 3D boxes inside the panels.
    waterfall_box_zoom
        Optional zoom override for the waterfall panel.
    surface_box_zoom
        Optional zoom override for the surface panel.
    heatmap_box_zoom
        Optional zoom override for the heatmap panel.
    elev
        Elevation angle passed to Matplotlib's 3D view.
    azim
        Azimuth angle passed to Matplotlib's 3D view.
    subfigsize
        Figure size shared by the three exported panels.
    subfigure
        Whether to use the smaller thesis subfigure font sizes for the panel
        exports. When ``False``, the thesis mplstyle font sizes are used.
    axes_rect
        Optional shared axes rectangle ``(left, bottom, width, height)`` in
        inches inside each panel figure.
    posx
        Absolute x positions ``(waterfall, surface, heatmap)`` in inches
        inside the stack canvas. ``None`` entries fall back to ``0``.
        Negative values are allowed.
    posy
        Absolute y positions ``(waterfall, surface, heatmap)`` in inches
        inside the stack canvas. ``None`` entries divide the stack canvas
        height into three equal slots and place the panels there. Negative
        values are allowed.
    figsize
        Optional compiled stacked-PDF size ``(width, height)``. When
        provided, the standalone `main.tex` and resulting `main.pdf` use
        that exact outer canvas size without rescaling the full stack.
    pdf_dpi
        Save DPI used for rasterized artists embedded in the exported panel
        PGFs.
    surface_pdf_dpi
        Optional dedicated save DPI for the surface panel PGF export.
        When omitted, the surface panel uses ``max(pdf_dpi, 3600)`` so
        rasterized surface bands remain cleaner under zoom.
    show_preview
        Whether to display ``main.png`` inline when running in IPython or
        Jupyter. This has no effect in non-IPython contexts.
    preview_height
        Height in pixels used for the inline PNG preview.
    preview_png_dpi
        DPI used to render ``main.png`` from the three panel figures.
    latex_command
        LaTeX command used for the temporary preview compile that produces
        ``main.png``. No `main.tex` or `main.pdf` are kept in the output
        bundle.
    pgf_texsystem
        LaTeX engine used as the default preview compiler and PGF export
        system. Defaults to ``"xelatex"``.

    Returns
    -------
    StackedThesisExport
        Dataclass containing the resolved asset paths.
    """
    export = _build_export_paths(
        name=name,
        sub_dir=sub_dir,
        local_dir=local_dir,
        remote_dir=remote_dir,
    )
    _prepare_local_stack_dir(export)
    _prepare_remote_stack_dir(export)
    command_name = _resolve_latex_command(
        latex_command=latex_command,
        pgf_texsystem=pgf_texsystem,
    )
    waterfall_zoom = _resolve_panel_zoom(
        box_zoom,
        override=waterfall_box_zoom,
    )
    surface_zoom = _resolve_panel_zoom(
        box_zoom,
        override=surface_box_zoom,
    )
    heatmap_zoom = _resolve_panel_zoom(
        box_zoom,
        override=heatmap_box_zoom,
    )
    resolved_surface_pdf_dpi = (
        max(float(pdf_dpi), 3600.0)
        if surface_pdf_dpi is None
        else float(surface_pdf_dpi)
    )
    waterfall_xaxis, surface_xaxis, heatmap_xaxis = _resolve_axis_triplet(
        x_axis,
        name="x_axis",
    )
    waterfall_yaxis, surface_yaxis, heatmap_yaxis = _resolve_axis_triplet(
        y_axis,
        name="y_axis",
    )
    waterfall_zaxis, surface_zaxis, heatmap_zaxis = _resolve_axis_triplet(
        z_axis,
        name="z_axis",
    )
    waterfall_y, waterfall_z, waterfall_trace_step_resolved, waterfall_trace_count = (
        _resolve_waterfall_traces(
            y=y,
            z=z,
            waterfall_traces=waterfall_traces,
            trace_step=trace_step,
        )
    )
    surface_trace_y: NDArray64 | None = None
    surface_trace_z: NDArray64 | None = None
    if waterfall_traces is not None:
        surface_trace_y, surface_trace_z = _resolve_surface_trace_overlay(
            x=x,
            y=waterfall_y,
            z=waterfall_z,
            ylim=ylim,
            trace_step=waterfall_trace_step_resolved,
            trace_count=waterfall_trace_count,
        )

    style_context = plt.style.context(str(_THESIS_MPLSTYLE))
    rc_params = _build_stack_font_rcparams(subfigure=subfigure)
    if pgf_texsystem is not None:
        rc_params["pgf.texsystem"] = pgf_texsystem
    rc_context = matplotlib.rc_context(rc_params)
    with style_context, rc_context:
        fig_waterfall, _ = get_thesis_waterfall_matplotlib(
            x=x,
            y=waterfall_y,
            z=waterfall_z,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            trace_step=waterfall_trace_step_resolved,
            trace_count=waterfall_trace_count,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            show_xaxis=waterfall_xaxis,
            show_yaxis=waterfall_yaxis,
            show_zaxis=waterfall_zaxis,
            z_axis_side=z_axis_side,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            labelspacing=labelspacing,
            ticklabelspacing=ticklabelspacing,
            ticks=ticks,
            ticklabels=ticklabels,
            line_color=line_color,
            line_width=line_width,
            alpha=line_alpha,
            box_aspect=box_aspect,
            box_zoom=waterfall_zoom,
            elev=elev,
            azim=azim,
            figsize=subfigsize,
            axes_rect=axes_rect,
        )
        fig_surface, _ = get_thesis_surface_matplotlib(
            x=x,
            y=y,
            z=z,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            clim=clim,
            trace_step=surface_trace_step,
            surface_x_oversample=surface_x_oversample,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            show_xaxis=surface_xaxis,
            show_yaxis=surface_yaxis,
            show_zaxis=surface_zaxis,
            z_axis_side=z_axis_side,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            labelspacing=labelspacing,
            ticklabelspacing=ticklabelspacing,
            ticks=ticks,
            ticklabels=ticklabels,
            clabel=clabel,
            trace_y=surface_trace_y,
            trace_z=surface_trace_z,
            trace_color=line_color,
            trace_width=line_width,
            trace_alpha=line_alpha,
            surface_alpha=surface_alpha,
            surface_shading=surface_shading,
            surface_light_azdeg=surface_light_azdeg,
            surface_light_altdeg=surface_light_altdeg,
            surface_shade_strength=surface_shade_strength,
            surface_rasterized=surface_rasterized,
            cmap_mpl=cmap_mpl,
            show_colorbar=False,
            box_aspect=box_aspect,
            box_zoom=surface_zoom,
            elev=elev,
            azim=azim,
            figsize=subfigsize,
            axes_rect=axes_rect,
        )
        fig_heatmap, _ = get_thesis_heatmap_matplotlib(
            x=x,
            y=y,
            z=z,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            clim=clim,
            trace_step=heatmap_trace_step,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            show_xaxis=heatmap_xaxis,
            show_yaxis=heatmap_yaxis,
            invert_xaxis=invert_xaxis,
            z_axis_side=z_axis_side,
            invert_yaxis=invert_yaxis,
            labelspacing=labelspacing,
            ticklabelspacing=ticklabelspacing,
            ticks=ticks,
            ticklabels=ticklabels,
            clabel=clabel,
            z_level=heatmap_z_level,
            heatmap_alpha=heatmap_alpha,
            heatmap_mode=heatmap_mode,
            heatmap_cell_overlap=heatmap_cell_overlap,
            show_zaxis=heatmap_zaxis,
            cmap_mpl=cmap_mpl,
            show_colorbar=False,
            box_aspect=box_aspect,
            box_zoom=heatmap_zoom,
            elev=elev,
            azim=azim,
            figsize=subfigsize,
            axes_rect=axes_rect,
        )

        try:
            _save_figure_pgf(
                fig_waterfall,
                pgf_path=export.waterfall_pgf,
                dpi=pdf_dpi,
            )
            _save_figure_pgf(
                fig_surface,
                pgf_path=export.surface_pgf,
                dpi=resolved_surface_pdf_dpi,
            )
            _save_figure_pgf(
                fig_heatmap,
                pgf_path=export.heatmap_pgf,
                dpi=pdf_dpi,
            )
        finally:
            plt.close(fig_waterfall)
            plt.close(fig_surface)
            plt.close(fig_heatmap)

    _ensure_parent(export.main_pgf)
    export.main_pgf.write_text(
        build_stack_pgf(
            waterfall_pgf=export.waterfall_pgf,
            surface_pgf=export.surface_pgf,
            heatmap_pgf=export.heatmap_pgf,
            main_pgf=export.main_pgf,
            subfigsize=subfigsize,
            posx=posx,
            posy=posy,
            figsize=figsize,
        ),
        encoding="utf-8",
    )
    _save_stack_preview_png(
        main_pgf=export.main_pgf,
        png_path=export.main_png,
        figsize=figsize,
        dpi=float(preview_png_dpi),
        latex_command=command_name,
    )
    if show_preview:
        _display_png_preview(
            export.main_png,
            height=int(preview_height),
        )
    _ship_remote_pgf_bundle(export)

    return export


__all__ = [
    "StackedThesisExport",
    "build_stack_pgf",
    "build_preview_tex",
    "compile_thesis_preview",
    "export_stacked_waterfall_thesis",
]
