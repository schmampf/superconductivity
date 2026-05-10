# TransportLab

TransportLab is the transport-focused analysis workspace for `superconductivity`.
It is intended as the replacement direction for the old GUI layer, but it is not
a thin rename of that interface. The goal is a transport-first application that
organizes the work around a cache-centered, visualization-first layout:

- the main window is a visualization workspace
- a second browser-level area holds the analysis workspace
- inside that workspace, the project cache and three producer pipelines live as
  tabs:
  - cache
  - evaluation
  - fitting
  - simulation

The old `superconductivity/deprecated_gui/` and `superconductivity/visuals/`
trees remain useful as reference material for workflow ideas, plot composition,
and export patterns, but they are not the target architecture for TransportLab.

## Purpose

TransportLab is meant to help with real experimental transport data:

- inspect trace collections
- correct offsets and prepare traces for analysis
- resample, filter, and sample transport curves
- compare data against model fits
- run transport simulations and inspect the results alongside measured data

The application should feel like a transport analysis bench, not a generic plot
viewer. The central project object is a `ProjectCache`, and the central
scientific representation inside it remains the `TransportDataset`. The main
screen is where the user sees cache entries selected for comparison:

- measured traces
- sampled transport data
- fit results
- simulation outputs
- comparisons between those representations

Export is intentionally not part of the first-pass architecture. It can be
added later as a dedicated follow-up layer once the visualization and analysis
flow are stable.

## Core Pipelines

### Cache

The cache is the authoritative project state for a TransportLab session. A
session has one active `ProjectCache`, stored as a trusted local pickle file
under the project `.cache/` directory.

The Cache tab is responsible for project-level state management:

- scan `.cache/` and list available cache files
- create or load the active cache
- show `cache.name`, `cache.path`, and `cache.file_path`
- list entries with key, type, and compact summary
- remove selected entries with `cache.remove(...)`
- save edits explicitly

Entry keys are user-controlled names such as `exp_v`, `cal_i`, `sim_bcs`,
`fit_bcs`, and `offsetanalysis`. The cache does not impose a fixed structure:
pipelines suggest names, and the user decides which entries become durable
workspace state.

### Visualization

Visualization is the entry point and the gathering point for the app:

- show raw and transformed transport data side by side
- plot trace collections and sampled transport datasets
- compare fit curves against measured data
- inspect simulation outputs in the same workspace

This is the main user-facing screen. It consumes selectable cache entries and
does not need to know which pipeline produced them. Compatible entries can be
filtered or grouped by object type, such as `TransportDatasetSpec`, `Traces`,
`Dataset`, specs, and miscellaneous objects.

### Evaluation

Evaluation covers the data-handling workflow around measured traces:

- load trace collections from a measurement
- inspect keys, labels, and trace metadata
- apply offset correction, low-pass filtering, and resampling
- sample traces into transport datasets
- derive PSD and other direct trace views where needed

This pipeline is about preparing and understanding measured transport data. Its
durable outputs are added to the active cache under explicit names.

### Fitting

Fitting compares sampled or transformed transport data against model families:

- BCS-style curves
- MAR-style models
- Josephson / RSJ-style fits where relevant

The fitting workflow reads prepared transport entries from the active cache, not
raw file metadata. It writes selected models, parameters, and fit results back
to the active cache as first-class state.

### Simulation

Simulation is the forward-model side of the tool:

- generate model transport curves
- compare simulated curves against measured or fitted data
- inspect parameter sweeps and derived response maps

This pipeline reads inputs from the active cache and writes simulated transport
datasets back to it. It stands beside evaluation and fitting as a peer producer
workflow, not as an afterthought.

## Design Direction

TransportLab should use a small number of explicit concepts:

- `ProjectCache` for authoritative project state
- `TransportDataset` for the shared structural output of all pipelines
- `Trace` and `Traces` for measured input data in the evaluation pipeline
- derived trace properties for sampling rate, frequency axis, and PSD views
- transport datasets as the visible representation of sampled, fitted, and
  simulated results
- analysis specs for evaluation, fitting, and simulation stages
- a visualization workspace that can host selected cache entries together

The UI should expose the visualization workspace first, then the cache-centered
analysis tabs beneath it. A user should be able to move from raw traces to
sampled transport data, to fit, to simulation, and back into a shared visual
comparison space by adding, visualizing, and deleting named cache entries.

## Expected Inputs and Outputs

Typical inputs:

- one active project cache, created or loaded from `.cache/`
- evaluation specs, as the pipeline uses them today, but not required
- outputs from modelling, analysis evaluation, and fitting if they already
  exist
- trace collections when the user wants to inspect measured data directly

Typical outputs:

- visualization scenes and comparisons
- updated cache entries that represent working project state
- transient visualization state and derived views

The cache is not a publication or archival export format. It is the trusted
local working state for TransportLab. Export remains separate and can be added
later as a dedicated layer once visualization and analysis workflows are stable.

The README intentionally stays at this level. It is the project brief for the
new GUI direction, not a compatibility guide for the old one.
