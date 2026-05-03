# TransportLab

TransportLab is the transport-focused analysis workspace for `superconductivity`.
It is intended as the replacement direction for the old GUI layer, but it is not
a thin rename of that interface. The goal is a transport-first application that
organizes the work around a visualization-first layout:

- the main window is a visualization workspace
- a second browser-level area holds the analysis workspace
- inside that workspace, three equal pipelines live as tabs:
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
viewer. The central object is a `TransportDataset`, and the main screen is
where the user sees the derived representations:

- measured traces
- sampled transport data
- fit results
- simulation outputs
- comparisons between those representations

Export is intentionally not part of the first-pass architecture. It can be
added later as a dedicated follow-up layer once the visualization and analysis
flow are stable.

## Core Pipelines

### Visualization

Visualization is the entry point and the gathering point for the app:

- show raw and transformed transport data side by side
- plot trace collections and sampled transport datasets
- compare fit curves against measured data
- inspect simulation outputs in the same workspace

This is the main user-facing screen. The other pipelines feed it.

### Evaluation

Evaluation covers the data-handling workflow around measured traces:

- load trace collections from a measurement
- inspect keys, labels, and trace metadata
- apply offset correction, low-pass filtering, and resampling
- sample traces into transport datasets
- derive PSD and other direct trace views where needed

This pipeline is about preparing and understanding measured transport data.

### Fitting

Fitting compares sampled or transformed transport data against model families:

- BCS-style curves
- MAR-style models
- Josephson / RSJ-style fits where relevant

The fitting workflow should operate on prepared transport data, not on raw file
metadata. It should make the selected model, parameters, and fit result visible
as first-class state.

### Simulation

Simulation is the forward-model side of the tool:

- generate model transport curves
- compare simulated curves against measured or fitted data
- inspect parameter sweeps and derived response maps

This pipeline should stand beside evaluation and fitting as a peer workflow, not
as an afterthought.

## Design Direction

TransportLab should use a small number of explicit concepts:

- `TransportDataset` for the shared structural output of all pipelines
- `Trace` and `Traces` for measured input data in the evaluation pipeline
- derived trace properties for sampling rate, frequency axis, and PSD views
- transport datasets as the visible representation of sampled, fitted, and
  simulated results
- analysis specs for evaluation, fitting, and simulation stages
- a visualization workspace that can host all of those representations together

The UI should expose the visualization workspace first, then the three analysis
tabs beneath it. A user should be able to move from raw traces to sampled
transport data, to fit, to simulation, and back into a shared visual comparison
space without switching mental models or encountering legacy mode names.

## Expected Inputs and Outputs

Typical inputs:

- evaluation specs, as the pipeline uses them today, but not required
- outputs from modelling, analysis evaluation, and fitting if they already
  exist
- trace collections when the user wants to inspect measured data directly

Typical outputs:

- visualization scenes and comparisons
- no new persisted output format yet
- transient visualization state and derived views

The README intentionally stays at this level. It is the project brief for the
new GUI direction, not a compatibility guide for the old one.
