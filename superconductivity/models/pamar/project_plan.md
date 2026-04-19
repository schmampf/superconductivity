# PAMAR Solver Project Plan

## Project goal

Build a new solver for superconducting point contacts under dc bias plus microwave drive that:

1. reproduces `ha.for` in the zero-drive limit,
2. supports a second harmonic index for microwave sidebands,
3. computes the dc current including photon-assisted MAR,
4. is structured cleanly enough that diagnostics, figure scripts, and later extensions remain manageable.

The plan assumes that `ha.for` is used as a **dc reference implementation**, but **not** as the direct implementation base for the driven solver.

---

## Overall timeline

Realistic estimate: **3–5 weeks** of focused work.

- **Week 1:** architecture + frozen dc references
- **Week 2:** zero-drive solver skeleton + dc regression
- **Week 3:** microwave sidebands + first PAMAR output
- **Week 4:** convergence studies + validation against expected structure
- **Week 5:** cleanup, acceleration, and figure generation

This can compress if the gauge and indexing work smoothly. It expands if the harmonic coupling conventions are not fully clear at the start.

---

## Core design principle

The new solver should be written such that `Nph = 0` and `Nph > 0` are the **same problem structurally**.

That means:

- no separate “dc solver” and “driven solver” cores,
- the driven architecture must reduce naturally to the dc case,
- zero microwave amplitude must collapse exactly to the dc solver.

This is the most important design decision in the whole project.

---

## Recommended project structure

```text
pamar_solver/
    DESIGN.md
    constants.py
    indexing.py
    bcs.py
    blocks.py
    recursion.py
    current.py
    solver.py
    validation.py
    figure_dc_regression.py
    figure_pamar_smoke_test.py
    tests/
        test_indexing.py
        test_bcs.py
        test_zero_drive.py
        test_truncation.py
```

### Module responsibilities

#### `DESIGN.md`
Short design note that freezes conventions before implementation starts.

#### `constants.py`
Physical constants, units, prefactors, and normalization conventions.

#### `indexing.py`
Mapping between composite indices `(j, n, alpha)` and flat matrix indices.

#### `bcs.py`
Equilibrium BCS lead Green functions and related helpers.

#### `blocks.py`
Construction of local block matrices for fixed energy and parameters.

#### `recursion.py`
The block recursion along the dc MAR ladder index `j`.

#### `current.py`
Assembly of the dc current from the solved blocks.

#### `solver.py`
Top-level API to compute `I(V)`.

#### `validation.py`
Regression and convergence helpers.

#### `figure_*.py`
Standalone scripts for sanity checks and paper-style figures.

---

## Phase 0 — pre-project specification

**Estimated effort:** 0.5–1.5 days

### Goal
Remove ambiguity before implementation begins.

### Tasks
Write `DESIGN.md` and freeze:

- internal units,
- meaning of transmission and hopping parameters,
- gauge choice for the microwave drive,
- harmonic index definitions,
- truncation conventions,
- definition of the dc observable in the driven problem,
- expected solver inputs and outputs.

### Deliverable
- `DESIGN.md`

### Comment
This file prevents repeated rewrites of indexing and gauge logic later.

---

## Phase 1 — extract dc reference behavior from `ha.for`

**Estimated effort:** 1–2 days

### Goal
Turn `ha.for` into a stable numerical oracle for the dc problem.

### Tasks
- Run `ha.for` for a small set of trusted parameters.
- Save reference `I(V)` curves for several transmissions.
- Document input/output meanings and quirks.
- Record cutoffs, normalization, and any special handling.

### Suggested reference set
- one low transmission case,
- one intermediate transmission case,
- one high transmission case.

For example:
- `tau = 0.1`
- `tau = 0.5`
- `tau = 0.9`

### Deliverables
- `reference_dc_tau_0p1.npz`
- `reference_dc_tau_0p5.npz`
- `reference_dc_tau_0p9.npz`
- short note describing generation settings

### Hard rule
Do not move on before frozen regression targets exist.

---

## Phase 2 — implement the indexing layer

**Estimated effort:** 0.5–1 day

### Goal
Make the composite basis unambiguous and testable.

### Tasks
- Define `j_vals`, `n_vals`, and Nambu index handling.
- Implement mapping:
  - `(j_idx, n_idx, alpha_idx) -> flat_idx`
  - `flat_idx -> (j_idx, n_idx, alpha_idx)`
- Add helpers for block slicing and sanity checks.

### Deliverables
- `indexing.py`
- `tests/test_indexing.py`

### Comment
This is a boring file, but it is one of the highest-risk parts of the project.

---

## Phase 3 — implement equilibrium BCS primitives

**Estimated effort:** 1–2 days

### Goal
Separate the trusted dc physics primitives from the old solver body.

### Tasks
- Reimplement equilibrium retarded, advanced, and Keldysh BCS Green functions.
- Verify symmetries and limiting behavior.
- Compare numerically to the relevant objects in `ha.for`.

### Deliverables
- `bcs.py`
- `tests/test_bcs.py`

### Comment
This phase should feel routine. If it does not, the conventions are probably not yet fixed well enough.

---

## Phase 4 — build the new solver architecture with `Nph = 0`

**Estimated effort:** 2–4 days

### Goal
Implement the new driven-style architecture first in the zero-sideband limit.

### Tasks
- Define block objects:
  - `e_r(j)`, `e_a(j)`
  - `v_plus(j)`, `v_minus(j)`
- With `Nph = 0`, these reduce to 2×2 Nambu objects.
- Implement recursion over `j`.
- Implement dc current assembly.

### Deliverable
- working `solver.py` that runs in the zero-drive case

### Milestone A
The new solver reproduces `ha.for` in the zero-drive limit within numerical tolerance.

### Hard rule
Do not add microwave sidebands before Milestone A is passed.

---

## Phase 5 — automated dc regression tests

**Estimated effort:** 1–2 days

### Goal
Lock down correctness before adding driven structure.

### Tasks
- Compare new solver vs `ha.for` for several transmissions.
- Compute error norms.
- Check convergence with respect to:
  - MAR cutoff `Nj`,
  - energy integration density,
  - any broadening or cutoff choices.
- Generate comparison plots.

### Deliverables
- `tests/test_zero_drive.py`
- `validation.py`
- `figure_dc_regression.py`

### Comment
This is the checkpoint that protects the rest of the project from drifting onto a wrong dc base.

---

## Phase 6 — add microwave coupling structure

**Estimated effort:** 2–5 days

### Goal
Generalize the block construction from `Nph = 0` to arbitrary `Nph`.

### Tasks
- Implement the chosen microwave gauge.
- Encode or derive sideband coupling coefficients.
- Build the sideband block matrices.
- Check that zero microwave amplitude collapses exactly to the dc structure.

### Deliverables
- updated `blocks.py`
- tests verifying:
  - amplitude-zero limit,
  - sideband diagonal structure at zero drive,
  - expected coupling behavior with sideband truncation

### Comment
This is the most delicate algebraic phase.

### Milestone B
At finite microwave amplitude, the solver runs and produces numerically sane current curves that connect smoothly to the dc limit.

---

## Phase 7 — first PAMAR smoke tests

**Estimated effort:** 2–4 days

### Goal
See the first photon-assisted MAR structure in a controlled setting.

### Tasks
- Start with low transmission.
- Use moderate sideband truncation.
- Scan microwave amplitude and frequency.
- Inspect additional subgap features relative to the dc case.
- Check whether threshold positions are qualitatively plausible.

### Deliverables
- `figure_pamar_smoke_test.py`
- exploratory notebook or script with first driven curves

### Comment
The question in this phase is:
“Do the additional structures appear in plausible places?”

It is **not** yet:
“Is the solver paper-grade?”

---

## Phase 8 — convergence and truncation study

**Estimated effort:** 3–5 days

### Goal
Determine whether observed features are physical or truncation artifacts.

### Tasks
Sweep:

- sideband cutoff `Nph`,
- MAR cutoff `Nj`,
- energy integration density.

Then test stability of:

- threshold positions,
- amplitudes,
- qualitative shape of the subgap structure.

### Deliverables
- `tests/test_truncation.py`
- convergence plots
- recommended default truncation settings

### Suggested study structure
For example:

- `Nph = 3, 5, 7, 9`
- `Nj = 21, 31, 41`
- coarse, medium, fine energy grids

### Comment
This phase is where many plausible-looking implementations fail.

---

## Phase 9 — validation against expected paper structure

**Estimated effort:** 2–5 days

### Goal
Check whether the solver reproduces the expected qualitative and semi-quantitative PAMAR behavior.

### Tasks
- Reproduce representative parameter scans.
- Compare threshold positions of photon-assisted MAR features.
- Examine transmission dependence.
- Check that the low-transmission limit connects sensibly to simpler PAT intuition.
- Document any mismatch honestly.

### Deliverables
- validation figures
- short note on agreements and mismatches

### Comment
This phase may expose remaining issues in gauge choice, prefactors, or current assembly.

---

## Phase 10 — refactor, accelerate, and document

**Estimated effort:** 2–5 days

### Goal
Improve speed and maintainability only after the physics is stable.

### Tasks
- Profile bottlenecks.
- Cache repeated coefficients where possible.
- Decide whether JAX is beneficial.
- Clean up public API vs internal helpers.
- Add usage examples.

### Deliverables
- cleaned `solver.py`
- short usage documentation
- performance notes

### Recommendation
Start in NumPy. Move to JAX only after the equations and indexing are stable.

---

## Weekly plan

### Week 1
- Write `DESIGN.md`
- Freeze dc reference curves from `ha.for`
- Implement indexing utilities
- Implement equilibrium BCS primitives

**Checkpoint:** trusted references and tested physics primitives exist

### Week 2
- Implement new recursion-based architecture with `Nph = 0`
- Reproduce the dc result
- Build regression plots and tests

**Checkpoint:** Milestone A passed

### Week 3
- Add microwave sideband coupling
- Run first driven calculations
- Perform first smoke tests

**Checkpoint:** Milestone B passed

### Week 4
- Run convergence studies
- Debug truncation issues
- Validate against expected PAMAR structure

**Checkpoint:** solver appears physically stable

### Week 5
- Refactor
- Profile
- Accelerate where useful
- Generate robust figure scripts

**Checkpoint:** usable research tool

---

## Success criteria

The project is successful when all of the following are true:

1. `Nph = 0` reproduces `ha.for` across several transmissions.
2. Zero microwave amplitude reproduces the dc result within numerical tolerance.
3. Increasing `Nph` and `Nj` does not qualitatively move the main PAMAR features.
4. Low-transmission driven results connect sensibly to PAT-like intuition.
5. Finite-transmission driven results show stable additional subgap features at plausible shifted MAR thresholds.

---

## Minimal viable version

If the full project becomes too large, the minimal useful target is:

- new solver architecture,
- exact zero-drive reproduction of `ha.for`,
- first stable PAMAR smoke test for one or two transmissions.

That is already a meaningful outcome.

---

## How Codex can help

Codex is most useful for the implementation surface area, not the core physics decisions.

### Good Codex tasks
- scaffold module structure,
- implement indexing utilities,
- translate trusted BCS primitives,
- write recursion skeletons,
- generate tests,
- build regression scripts,
- refactor block assembly code.

### Bad Codex tasks
- deciding the correct gauge from vague prose,
- choosing the driven formalism for you,
- judging whether a curve is physically correct rather than just plausible.

### Recommended usage pattern
Use Codex as a disciplined implementation assistant with narrow, well-defined tasks. Do not use it as the unsupervised theorist for the project.

---

## Main risks

### 1. Wrong indexing
Produces structured but incorrect curves.

### 2. Wrong gauge or sign convention
Often difficult to detect because some symmetries still survive.

### 3. Insufficient truncation
Creates fake sideband structure or unstable amplitudes.

### 4. Adding microwaves before validating the dc base
Wastes time on debugging the wrong problem.

---

## Very first next action

Before any substantial coding:

1. write the one-page `DESIGN.md`,
2. freeze three dc reference curves from `ha.for`.

That will determine whether the project remains compact and controlled or becomes chaotic.
