# TransportLab State of the Art

Last updated: 2026-06-24

## Purpose and Direction

TransportLab is the new graphical analysis environment for the
`superconductivity` package. It replaces the architecture in
`superconductivity/deprecated_gui/`. The deprecated GUI remains reference
material for scientific workflows and plotting ideas, but new development
belongs in `superconductivity/TransportLab/`.

The new GUI is built with Panel and currently exposes two browser routes:

- `/visualization` is the main data-viewing workspace.
- `/workspace` contains the Cache, Evaluation, Fitting, and Simulation tabs.

The central application state is a `TransportLabSession`. One session owns one
project path and one active `ProjectCache`. Analysis stages are expected to
read named entries from this cache, write named results back to it, and notify
the visualization page when the active cache changes.

## Current Implementation

### Application shell

- `serve_transport_lab()` starts both Panel routes on one local server.
- The default port is `5010`.
- The visualization and workspace pages share one `TransportLabSession`.
- The session now supports cache-change callbacks so independently rendered
  browser pages can refresh when another page loads or changes the active
  cache.
- Panel initializes the Tabulator and Plotly extensions.

### Cache workspace

- The Cache tab can select a project directory on macOS.
- It scans the project's `.cache/` directory and loads an existing
  `ProjectCache`.
- Selecting a cache displays its top-level keys.
- Selecting a top-level key displays one further level of nested keys.
- Selecting a nested key displays its Python representation.
- Loading a cache notifies registered visualization callbacks.

The current cache browser is inspection-oriented. Cache creation, explicit
saving, deletion, entry summaries, robust user-facing error messages, and
general recursive inspection are not implemented yet.

### Visualization workspace

- The page lists cache entries that are instances of
  `TransportDatasetSpec`.
- Multiple datasets can be selected and rendered in a two-column Plotly grid.
- Available plot modes are `trace`, `heatmap`, and `surface`.
- The quantity tables distinguish the horizontal axis, vertical axis, and
  plotted data quantity using axis order and array shape.
- One-dimensional data use a plain trace plot.
- Two-dimensional data use a slider trace view, heatmap, or surface.
- Missing or incompatible axes fall back to integer indices.
- Plot labels use the dataset entries' HTML labels where available.
- Rendering failures are collected and shown in a warning pane.
- The dataset list refreshes when the active cache changes.

This work is implemented locally but was not committed when this document was
written.

### Pipeline workspaces

The Evaluation, Fitting, and Simulation tabs exist in the workspace route, but
each is still an explicit empty placeholder. No end-to-end analysis pipeline
is available through TransportLab yet.

## Worktree Snapshot

At the time of this snapshot, branch `main` is one local commit ahead of
`origin/main`. That local commit contains thesis-figure updates and must remain
separate from TransportLab work.

Relevant modified files:

- `pipeline_test.ipynb`
- `superconductivity/TransportLab/README.md`
- `superconductivity/TransportLab/app.py`
- `superconductivity/TransportLab/cache.py`
- `superconductivity/TransportLab/visualization.py`
- `superconductivity/evaluation/offset.py`
- `superconductivity/models/bcs/backend/__init__.py`
- `tests/test_fit_model.py`
- `tests/test_transport_lab.py`

Relevant untracked files:

- `test_gui.ipynb`, a minimal TransportLab launcher notebook.
- `superconductivity/TransportLab/state-of-the-art.md`, this restart document.

The notebooks and non-GUI Python changes are separate concerns. In particular,
`pipeline_test.ipynb` contains local measurement paths and should be reviewed
before committing. `templates.py` was identified as unrelated card-template
data during review and must not be added to a TransportLab commit if it is
present in a future worktree.

## Running the GUI

Run Python from the repository root so the package resolves correctly:

```bash
cd /Users/oliver/Documents/cryolab/superconductivity
../.venv/bin/python
```

Then launch TransportLab:

```python
from superconductivity.TransportLab import serve_transport_lab

server = serve_transport_lab()
```

This opens:

- `http://localhost:5010/visualization`
- `http://localhost:5010/workspace`

For a non-interactive smoke test:

```python
from superconductivity.TransportLab import serve_transport_lab

server = serve_transport_lab(open_browser=False, verbose=False)
server.stop()
```

The untracked `test_gui.ipynb` provides the same minimal launcher for notebook
use.

## Verification

Run the focused TransportLab tests:

```bash
../.venv/bin/python -m pytest tests/test_transport_lab.py
```

Run the fitting tests that cover the adaptive BCS kernel:

```bash
../.venv/bin/python -m pytest tests/test_fit_model.py
```

Before committing GUI work, also perform the non-interactive launch smoke test
shown above. Record any failures in this document rather than assuming the
current uncommitted implementation is stable.

Verification completed on 2026-06-24:

- `tests/test_transport_lab.py`: 12 passed.
- `tests/test_fit_model.py`: 3 passed.
- A reachable `/visualization` route was confirmed on temporary port `5011`,
  then the server was stopped.
- Both pytest runs emitted only a sandbox-specific warning because
  `.pytest_cache` was not writable.

## Ordered Development Plan

1. Stabilize the current cache and visualization implementation. Resolve test
   failures, exercise both browser routes with a real project cache, and define
   clear empty and error states.
2. Complete project-cache management. Add cache creation, explicit saving,
   entry removal, compact summaries, recursive inspection, and user-facing load
   errors.
3. Implement Evaluation. Build the workflow from measurement-file selection
   through trace loading, offset correction, filtering, resampling, and
   transport sampling.
4. Implement Fitting. Start with BCS fitting, then add MAR and RSJ model
   workflows after the cache and visualization interfaces are stable.
5. Implement Simulation. Use the same cache-backed `TransportDatasetSpec`
   representation so simulated, measured, and fitted results can be compared
   in the visualization workspace.
6. Add presentation and secondary features. Persist visualization choices,
   improve layout and navigation, add exports, and expand user documentation
   only after the scientific workflows work end to end.

Every completed stage must:

- read inputs from the active `ProjectCache`;
- write durable outputs under explicit user-visible names;
- notify the visualization page after cache changes;
- preserve enough parameters and metadata to reproduce the result;
- include focused unit tests and at least one application-level smoke test.

## Recommended Commit Sequence

Keep the existing local thesis-figure commit unchanged. Create new commits in
this order:

1. `document current TransportLab state and roadmap`
   - Add this document.
   - Link it from `superconductivity/TransportLab/README.md`.
2. `implement TransportLab cache-backed visualization`
   - Commit `app.py`, `cache.py`, `visualization.py`, and
     `tests/test_transport_lab.py`.
   - Include `test_gui.ipynb` only if it remains a clean minimal launcher with
     no embedded local data or large stored output.
3. `support adaptive BCS fitting`
   - Commit the BCS backend type update and its fitting tests.
4. `fix offset specification key lookup`
   - Commit the offset change separately unless verification proves it is
     required by the GUI commit.
5. `update transport evaluation pipeline notebook`
   - Commit `pipeline_test.ipynb` only after removing stale output and reviewing
     its absolute measurement-data paths.

Do not stage or commit `templates.py` with this work.
