"""HTML dashboard helpers for Plotly exports."""

import re
import webbrowser
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional

_DIV_RE = re.compile(
    r'(<div id="[^"]+" class="plotly-graph-div"[^>]*>.*?</div>)',
    re.S,
)
_SCRIPT_RE = re.compile(r"(<script[^>]*>.*?</script>)", re.S)


@dataclass(frozen=True)
class _PlotFile:
    """Parsed plot metadata from an HTML filename."""

    block: str
    label: str
    column: int
    path: Path


def _extract_plotly_fragment(html_text: str) -> str:
    """Extract a Plotly div+script fragment from a full HTML document.

    Parameters
    ----------
    html_text
        Full HTML document as a string.

    Returns
    -------
    str
        HTML snippet containing the Plotly graph div and its render script.

    Raises
    ------
    ValueError
        If the Plotly div or matching inline render script is missing.
    """
    mdiv = _DIV_RE.search(html_text)
    if not mdiv:
        raise ValueError("Could not find plotly-graph-div <div> in HTML.")

    div = mdiv.group(1)
    tail = html_text[mdiv.end() :]

    for mscript in _SCRIPT_RE.finditer(tail):
        script = mscript.group(1)
        if "src=" in script:
            continue
        if "Plotly.newPlot" in script or "Plotly.react" in script:
            return div + "\n" + script

    raise ValueError("Could not find inline Plotly script after graph div.")


def _split_stem_and_column(stem: str) -> Optional[tuple[list[str], int]]:
    """Split a filename stem into tokens and trailing integer column index.

    Parameters
    ----------
    stem
        Filename stem without extension.

    Returns
    -------
    tuple[list[str], int] or None
        ``(tokens_without_column, column_index)`` when valid, else None.
    """
    tokens = stem.split()
    if len(tokens) < 3:
        return None
    if not tokens[-1].isdigit():
        return None
    return tokens[:-1], int(tokens[-1])


def _longest_common_prefix(token_rows: list[list[str]]) -> list[str]:
    """Return the longest common prefix across token rows."""
    if not token_rows:
        return []

    prefix = token_rows[0][:]
    for row in token_rows[1:]:
        max_len = min(len(prefix), len(row))
        match_len = 0
        while match_len < max_len and prefix[match_len] == row[match_len]:
            match_len += 1
        prefix = prefix[:match_len]
        if not prefix:
            break
    return prefix


def _collect_plot_files(html_files: list[Path]) -> list[_PlotFile]:
    """Parse indexed plot filenames into block/column metadata.

    Expected filename scheme
    ------------------------
    ``<block title> <column label> <column index>.html``.

    Notes
    -----
    The block title is inferred per leading-token group by longest common
    prefix, which supports names such as:
    ``0 name1 surface 0``, ``0 name1 slider 1``, ``0 name1 heatmap 2``.
    """
    grouped: dict[str, list[tuple[Path, list[str], int]]] = {}

    for html_path in html_files:
        parsed = _split_stem_and_column(html_path.stem)
        if parsed is None:
            continue
        tokens, column = parsed
        grouped.setdefault(tokens[0], []).append((html_path, tokens, column))

    plot_files: list[_PlotFile] = []
    for entries in grouped.values():
        if len(entries) == 1:
            html_path, tokens, column = entries[0]
            block_tokens = tokens[:-1]
            label_tokens = [tokens[-1]]
            if not block_tokens:
                raise ValueError(
                    f"Filename '{html_path.name}' needs block and label text."
                )
            plot_files.append(
                _PlotFile(
                    block=" ".join(block_tokens),
                    label=" ".join(label_tokens),
                    column=column,
                    path=html_path,
                )
            )
            continue

        token_rows = [tokens for _, tokens, _ in entries]
        block_tokens = _longest_common_prefix(token_rows)
        shortest = min(len(tokens) for tokens in token_rows)
        if not block_tokens or len(block_tokens) >= shortest:
            sample = entries[0][0].name
            raise ValueError(
                "Could not infer block/label split from indexed filenames. "
                "Use names like '<block> <label> <index>.html'. "
                f"Example seen: '{sample}'."
            )

        block_title = " ".join(block_tokens)
        for html_path, tokens, column in entries:
            label_tokens = tokens[len(block_tokens) :]
            if not label_tokens:
                raise ValueError(
                    f"Filename '{html_path.name}' is missing a column label."
                )
            plot_files.append(
                _PlotFile(
                    block=block_title,
                    label=" ".join(label_tokens),
                    column=column,
                    path=html_path,
                )
            )

    return plot_files


def _block_sort_key(block_title: str) -> tuple[int, int, str]:
    """Sort blocks by numeric first token when present."""
    first = block_title.split(maxsplit=1)[0]
    if first.isdigit():
        return (0, int(first), block_title.lower())
    return (1, 0, block_title.lower())


def build_html(
    dataset: str | Path,
    title: str = "main",
    title_html: str = "main_html",
    text: str = "",
    iframe_height_px: int = 400,
    plotlyjs_path: Optional[str | Path] = None,
    auto_open: bool = False,
):
    """Build a standalone dashboard HTML from indexed Plotly HTML files.

    Parameters
    ----------
    dataset
        Dataset directory containing an ``html`` subfolder with Plotly exports.
    title
        Output filename stem. The page is written to
        ``<dataset>/<title>.html``.
    title_html
        HTML title shown in the browser tab and as page header.
    iframe_height_px
        Fixed height of each plot cell in pixels.
    plotlyjs_path
        Optional path to a local Plotly.js bundle.
        If None, the function searches ``<dataset>/html`` for common names.
    auto_open
        If True, open the written HTML file in the default web browser.

    Raises
    ------
    ValueError
        If required files or filename structure are missing.
    """
    folder = Path(dataset, "html")
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    if plotlyjs_path is None:
        candidates = [
            folder / "plotly.min.js",
            folder / "plotly-latest.min.js",
            folder / "plotly.js",
        ]
        plotlyjs = next((path for path in candidates if path.exists()), None)
        if plotlyjs is None:
            raise ValueError(
                "Could not find plotly.min.js in folder. "
                "Save at least one figure with "
                "include_plotlyjs='directory' first, "
                "or pass plotlyjs_path explicitly."
            )
        plotlyjs_path = plotlyjs

    plotlyjs_text = Path(plotlyjs_path).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    html_files = sorted(path for path in folder.glob("*.html"))
    plot_files = _collect_plot_files(html_files)
    if not plot_files:
        raise ValueError(
            "No indexed plot files found. Expected names like "
            "'<block> <label> <index>.html'."
        )

    groups: dict[str, dict[int, _PlotFile]] = {}
    for plot_file in plot_files:
        group = groups.setdefault(plot_file.block, {})
        if plot_file.column in group:
            other = group[plot_file.column]
            raise ValueError(
                "Duplicate column index in block "
                f"'{plot_file.block}' for column {plot_file.column}: "
                f"'{other.path.name}' and '{plot_file.path.name}'."
            )
        group[plot_file.column] = plot_file

    def cell_fragment(path: Optional[Path]) -> str:
        if path is None:
            return '<div class="missing">—</div>'
        html_text = path.read_text(encoding="utf-8", errors="ignore")
        return _extract_plotly_fragment(html_text)

    rows: list[str] = []
    for block in sorted(groups.keys(), key=_block_sort_key):
        columns = groups[block]
        ncols = max(columns) + 1

        labels_html: list[str] = []
        cells_html: list[str] = []
        for column in range(ncols):
            plot_file = columns.get(column)
            if plot_file is None:
                labels_html.append(f'<div class="celllabel">column {column}</div>')
                cells_html.append(
                    '<div class="cell"><div class="missing">—</div></div>'
                )
                continue
            labels_html.append(
                f'<div class="celllabel">{escape(plot_file.label)}</div>'
            )
            cells_html.append(
                f'<div class="cell">{cell_fragment(plot_file.path)}</div>'
            )

        rows.append(
            f"""
    <div class="block" style="--ncols: {ncols};">
      <div class="blocktitle">{escape(block)}</div>
      {"".join(labels_html)}
      {"".join(cells_html)}
    </div>
    """.strip()
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>

  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica,
        Arial, sans-serif;
      margin: 18px;
      background: #fff;
    }}
    h1 {{ margin: 0 0 14px 0; }}

    .block {{
      display: grid;
      grid-template-columns: repeat(var(--ncols), minmax(0, 1fr));
      grid-template-rows: auto auto 1fr;
      gap: 12px;
      margin: 10px 0 18px 0;
    }}

    .blocktitle {{
      grid-column: 1 / -1;
      font-size: 12px;
      font-weight: 700;
      color: #333;
      line-height: 1.2;
      padding: 2px 0 4px 0;
      word-break: break-word;
    }}

    .celllabel {{
      font-size: 12px;
      font-weight: 700;
      color: #333;
      line-height: 1.2;
      word-break: break-word;
    }}

    .cell {{
      height: {iframe_height_px}px;
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 8px;
      background: white;
      overflow: hidden;
      position: relative;
    }}

    .cell > .plotly-graph-div {{
      width: 100% !important;
      height: 100% !important;
    }}

    .missing {{
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: rgba(0,0,0,0.5);
    }}
  </style>

  <script>
  {plotlyjs_text}
  </script>
</head>

<body>
  <h1>{title_html}</h1>
  {"".join(rows)}

</body>
</html>
"""

    out_path = Path(dataset, f"{title}.html")
    out_path.write_text(html, encoding="utf-8")
    if auto_open:
        webbrowser.open(out_path.resolve().as_uri())
