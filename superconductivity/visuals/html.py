"""HTML dashboard helpers.

This module builds a single standalone HTML "dashboard" from multiple Plotly
HTML exports stored in a dataset directory. It is intended for quick
side-by-side comparison of related plots (e.g. surface/slider/heatmap) across
multiple datasets.

Notes
-----
The dashboard is created by extracting the Plotly div+script fragments from the
individual HTML files and embedding them into one page. Plotly.js is inlined
once (typically from a local ``plotly.min.js`` produced by
``include_plotlyjs='directory'``).
"""

import re
import webbrowser
from pathlib import Path
from typing import Dict, Literal, Optional

_DIV_RE = re.compile(
    r'(<div id="[^"]+" class="plotly-graph-div"[^>]*>.*?</div>)',
    re.S,
)
_SCRIPT_RE = re.compile(r"(<script[^>]*>.*?</script>)", re.S)
Slot = Literal["surface", "slider", "heatmap"]


def _extract_plotly_fragment(html_text: str) -> str:
    """Extract a Plotly div+script fragment from an HTML document.

    The function searches for the first element with class
    ``plotly-graph-div`` and then finds the first subsequent inline
    ``<script>``
    block that contains a Plotly render call (``Plotly.newPlot`` or
    ``Plotly.react``). External scripts (``src=...``) are ignored.

    Parameters
    ----------
    html_text
        Full HTML document as a string.

    Returns
    -------
    fragment
        HTML snippet containing the Plotly graph div and the
        corresponding inline
        script required to render it.

    Raises
    ------
    ValueError
        If the Plotly div or an inline Plotly script cannot be found.
    """
    mdiv = _DIV_RE.search(html_text)
    if not mdiv:
        raise ValueError("Could not find plotly-graph-div <div> in HTML.")

    div = mdiv.group(1)
    # take scripts after div; pick the first inline one (not src=...)
    tail = html_text[mdiv.end() :]

    for m in _SCRIPT_RE.finditer(tail):
        script = m.group(1)
        # skip external scripts
        if "src=" in script:
            continue
        # heuristic: must reference Plotly
        if "Plotly.newPlot" in script or "Plotly.react" in script:
            return div + "\n" + script

    raise ValueError("Could not find inline Plotly script after graph div.")


def build_html(
    dataset: str | Path,
    title: str = "main",
    title_html: str = "main_html",
    iframe_height_px: int = 400,  # used as cell height
    plotlyjs_path: Optional[str | Path] = None,
    auto_open: bool = False,
):
    """Build a standalone dashboard HTML from per-plot Plotly HTML files.

    The function looks for files in ``<dataset>/html`` whose stems end with
    ``_surface``, ``_slider``, or ``_heatmap``. For each base name, it builds a
    row consisting of three plot cells (surface/slider/heatmap). The resulting
    page inlines Plotly.js once and embeds each plot via its extracted div
    +script
    fragment.

    Parameters
    ----------
    dataset
        Dataset directory containing an ``html`` subfolder with Plotly exports.
    title
        Output filename stem. The page is written to
        ``<dataset>/<title>.html``.
    title_html
        HTML title shown in the browser tab and as the page header.
    iframe_height_px
        Fixed height of each plot cell in pixels.
    plotlyjs_path
        Optional path to a local Plotly.js bundle.
        If None, the function searches
        ``<dataset>/html`` for common names (e.g. ``plotly.min.js``).
    auto_open
        If True, open the written HTML file in the default web browser.

    Raises
    ------
    ValueError
        If the expected directory structure or required files are missing.

    Notes
    -----
    This builder intentionally produces a single-file HTML artifact. It does
    not
    attempt to make plots interact with each other; each plot is rendered
    independently within the same page.
    """
    folder = Path(dataset, "html")
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    # Locate plotly.min.js to inline
    if plotlyjs_path is None:
        # common filenames produced by Plotly's include_plotlyjs="directory"
        candidates = [
            folder / "plotly.min.js",
            folder / "plotly-latest.min.js",
            folder / "plotly.js",
        ]
        plotlyjs = next((p for p in candidates if p.exists()), None)
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

    # Collect files
    html_files = sorted(p for p in folder.glob("*.html"))

    groups: Dict[str, Dict[str, Path]] = {}
    for p in html_files:
        stem = p.stem
        for suf, slot in [
            ("_surface", "surface"),
            ("_slider", "slider"),
            ("_heatmap", "heatmap"),
        ]:
            if stem.endswith(suf):
                base = stem[: -len(suf)]
                groups.setdefault(base, {})[slot] = p
                break

    if not groups:
        raise ValueError("No *_surface/_slider/_heatmap.html files found.")

    bases = sorted(groups.keys())

    def cell_fragment(p: Optional[Path]) -> str:
        if p is None:
            return '<div class="missing">—</div>'
        txt = p.read_text(encoding="utf-8", errors="ignore")
        frag = _extract_plotly_fragment(txt)
        return frag

    rows = []
    for base in bases:
        g = groups[base]
        rows.append(
            f"""
    <div class="block">
      <div class="blocktitle">{base}</div>
      <div class="cell">{cell_fragment(g.get("surface"))}</div>
      <div class="cell">{cell_fragment(g.get("slider"))}</div>
      <div class="cell">{cell_fragment(g.get("heatmap"))}</div>
    </div>
    """.strip()
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title_html}</title>

  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 18px;
      background: #fff;
    }}
    h1 {{ margin: 0 0 12px 0; }}

    .header {{
      display: grid;
      grid-template-columns: 260px 1fr 1fr 1fr;
      gap: 12px;
      align-items: end;
      margin-bottom: 10px;
    }}
    .header div {{
      font-size: 12px;
      font-weight: 700;
      color: #333;
    }}

    .block{{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;   /* 3 plot columns */
        grid-template-rows: auto 1fr;         /* title row + plots row */
        gap: 12px;
        margin: 10px 0 18px 0;
    }}

    .blocktitle{{
        grid-column: 1 / -1;                 /* span all 3 columns */
        font-size: 12px;
        font-weight: 700;
        color: #333;
        line-height: 1.2;
        padding: 2px 0 4px 0;
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

    /* Plotly uses inline sizing; make the container fill the cell */
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
  <div class="header">
    <div>Dataset</div>
    <div>Surface</div>
    <div>Slider</div>
    <div>Heatmap</div>
  </div>

  {"".join(rows)}

</body>
</html>
"""

    out_path = Path(dataset, f"{title}.html")
    out_path.write_text(html, encoding="utf-8")
    if auto_open:
        webbrowser.open(out_path.resolve().as_uri())
