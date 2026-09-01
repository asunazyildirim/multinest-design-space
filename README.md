# Adaptation of multi-ellipsoidal nested sampling for design space characterisation

Code and data for the report of the same name.

*See [Declaration of generative AI use](#declaration-of-generative-ai-use) at
the end.*

The proposed sampler bounds the live points with a recursively constructed set
of ellipsoids, in the manner of MultiNest, instead of the single enclosing
ellipsoid used by the MAGNUS/NSFeas baseline. This repository contains the
sampler, the four comparison studies, and the saved runs needed to reproduce
every figure and table in the report without re-running anything expensive.

---

## Quick start

```bash
pip install -r requirements.txt

python benchmark_2d.py report      # Section 3.2.1 figure + tables
python gtcd_case.py --report       # Section 3.2.2 tables
python tp_case_study_tables.py     # Section 3.2.3 tables
python tp_case_study_analysis.py   # Section 3.2.3 process figure
python compare_feasible_spaces.py  # Section 3.2.3 comparison figure
```

Those five commands reproduce every number quoted in Section 3.2 from the
archives shipped here. None of them needs Aspen HYSYS, MAGNUS, or a long run.

Use the pinned numpy and scipy if you intend to re-run a sampler and compare
its counters against the report: a different build gives the same design space
but slightly different counts. Redrawing the figures and tables from the
archives works on any version.

---

## What each file is for

### The sampler

| File | Purpose |
|---|---|
| `multinest_sampler.py` | The whole method: design-space scaling, the three feasibility measures (`P_f`, VaR, CVaR), ellipsoid construction and enlargement, the recursive multi-ellipsoidal decomposition, sampling from the ellipsoidal union with overlap correction, adaptive reconstruction, mode separation and freezing. Also holds `EXAMPLES`, the catalogue of geometric test problems, the `Visualizer`, and the frame recorder/player. Every other script imports it. |

Running `python multinest_sampler.py` opens the example catalogue and runs one
problem interactively. Entries 3 (Banana + Island), 6 (Three torus rings) and 8
(Single ellipse) carry the `seed` and `snapshots` settings that produced
**Figure 3**; they write `snapshots/<title>_sweep<N>.png`. Snapshot sweeps must
be named before the run, so the workflow is: run once, read the sweep count off
the log, pin the interesting sweeps, re-run with the same seed.

### Section 3.1 — geometric behaviour

| File | Produces |
|---|---|
| `reference_grids.py` | **Figure 2** — the VaR maps of the three geometric test problems, evaluated on a fine grid by Monte-Carlo propagation, with the design-space boundary as a single iso-line. Writes `reference_grids_output/`. |

### Section 3.2.1 — two-dimensional benchmarks

| File | Produces |
|---|---|
| `benchmark_2d.py` | Kusumo et al. (2020) and Banana + Island, proposed sampler vs MAGNUS. `run` executes the proposed sampler and writes one `.npz` per (problem, seed); `report` reads every `.npz` in the output directory, whichever sampler wrote it, and produces the figure and the three tables. |
| `magnus_wsl/magnus_benchmark_2d.py` | The NSFeas half. Runs under WSL and writes `.npz` files in the same schema into the same directory. |

`report` needs neither MAGNUS nor WSL — the NSFeas archives are already in
`benchmark_2d_output/`.

### Section 3.2.2 — four-dimensional GTCD

| File | Produces |
|---|---|
| `gtcd_case.py` | The GTCD trellis charts and the computational-performance table. `--report` prints the tables from the saved runs and exits; `--replot --npz <file>` redraws a trellis without sampling. |
| `magnus_wsl/magnus_gtcd_case.py`, `run_magnus_gtcd.sh` | The MAGNUS half. |

### Section 3.2.3 — CO₂-to-methanol case study

The run itself (needs Aspen HYSYS with `methanol.hsc` open):

| File | Purpose |
|---|---|
| `run_sampler_kinetics_serial.py` | The run reported in the paper: one HYSYS instance, one solve at a time, so the comparison against MAGNUS measures the samplers rather than the hardware. Writes `tp_design_space_serial.{npz,csv}`. Section 1 of the file is the only part meant to be edited. |
| `live_monitor.py` | Imported by the driver above. Rewrites a snapshot PNG as the run goes and pickles the frames, so a six-hour unattended run can be watched and replayed. |
| `simulation_runner.py` | `run_case` — drives one flowsheet solve. Warms the recycle up in stages instead of handing the solver the hard problem at once, and returns the carbon- and energy-efficiency outputs the constraints are written against. |
| `convergence.py` | Whether the flowsheet actually converged. `Solver.IsSolving == False` only means HYSYS stopped; three independent checks read from the live COM interface decide whether the answer is usable. Imported by `simulation_runner.py`. |
| `hysys_connection.py` | Binds to the open case through the Running Object Table. |
| `hysys_bridge_server.py`, `start_bridge_server.bat` | The Windows half of the MAGNUS run. MAGNUS lives in WSL and cannot reach a Windows COM server, so this process owns the HYSYS connection and answers solve requests through a spool directory on the shared NTFS volume. Serial by design. |
| `magnus_wsl/magnus_tp_case_study.py`, `run_magnus_tp.sh` | The WSL client for the above. |
| `magnus_wsl/magnus_tp_case_study_physbox.py` | The same client with one change: NSFeas searches the physical (T, P) box rather than the normalised control box. Not a second result — a check that the normalisation is immaterial, which it is: the two reproduce each other point for point. The numbers are in its docstring. |

The reference sweep behind the process figure:

| File | Purpose |
|---|---|
| `sensitivity_parallel_batch_v2.py` | The 130-point (10 temperature × 13 pressure) sweep that Figure X(a–c) interpolates. Already run — its output is `output/sensitivity_results_20260731_174218.csv`. Only needed to repeat the sweep. |
| `flowsheet_copier.py` | Makes the four `methanol_instance*.hsc` copies the sweep above distributes over. Needs the in-house `hysyspy` package. |

Figures and tables (no HYSYS needed):

| File | Produces |
|---|---|
| `tp_case_study_analysis.py` | **Figure X** — (a) carbon efficiency, (b) energy efficiency, (c) the nominal feasible region, from the reference sweep. Also prints the reasoning behind the 92 % / 81.3 % thresholds. |
| `compare_feasible_spaces.py` | **Figure Y** — proposed vs MAGNUS feasible spaces, drawn on one shared pair of axes so the two regions are comparable. Feasibility is read from the merit column, not from the sampler's role bookkeeping. |
| `plot_design_space.py` | One run on its own: live/dead/rejected points, and the same points coloured by merit. Imported by `compare_feasible_spaces.py`. |
| `tp_case_study_tables.py` | **Tables X and Y** — certified region, computational performance, mode separation, and the wall-clock breakdown including the runtime at a common per-solve cost. Every figure comes from the saved archives; nothing is typed in. |

---

## Data

| Path | What it is |
|---|---|
| `reference_grids_output/*.npz` | Monte-Carlo reference fields for the geometric problems. |
| `benchmark_2d_output/points_proposed_*.npz` | Proposed-sampler runs, Kusumo and Banana + Island, seeds 1 and 11. |
| `benchmark_2d_output/points_nsfeas_*.npz` | The MAGNUS/NSFeas runs of the same two problems. |
| `gtcd_output/gtcd_points_s11.npz` | Proposed sampler on GTCD, `N_L = 5000`. |
| `gtcd_output/gtcd_points_magnus_s11.npz` | MAGNUS on GTCD. |
| `output/sensitivity_results_20260731_174218.csv` | The 130-point T–P reference sweep. |
| `tp_design_space_serial.{npz,csv}` | The reported CO₂-to-methanol run of the proposed sampler. |
| `magnus_tp_output/magnus_tp_bridge_NL500.{npz,csv}` | The MAGNUS run of the same problem, 18 August, timing-instrumented, no warm start, 0 cache hits. |
| `magnus_tp_output/magnus_tp_physbox_bridge_NL500.{npz,csv}`, `logs/magnus_tp_physbox_bridge_NL500.log` | The same MAGNUS run with the search box left unnormalised. Identical to the row above: the same 1612 points to 2.3e-11, the same 657 / 455 split, the same efficiency. Assembled over three sessions after two HYSYS crashes, so it carries 1096 warm cache hits and its wall clock measures nothing. |
| `logs/tp_run_serial.log`, `logs/magnus_tp_bridge_NL500.log` | The raw run logs behind the wall-clock figures in Table Y. |
The Aspen HYSYS flowsheet behind the CO₂-to-methanol study, `methanol.hsc`, is
not included here. The archived runs above are what it produced, and the code
that drives it is in the table further up, so everything except the flowsheet
itself is present.

The analytical archives can be regenerated by running the samplers — the seeds
are fixed and recorded. The HYSYS-derived files cannot be regenerated without a
HYSYS licence, and the MAGNUS archives need a MAGNUS build under WSL, so all of
them are kept here.

Generated figures and workbooks are not tracked; the commands in *Quick start*
rewrite them in seconds.

---

## What you need

| To do this | You need |
|---|---|
| Redraw every figure and table in the report | any recent numpy/scipy; the archives are just read |
| Re-run the analytical studies (Sections 3.1, 3.2.1, 3.2.2) | the pinned numpy/scipy, plus time — see *Quick start* |
| Re-run the CO₂-to-methanol characterisation | Windows, Aspen HYSYS V14 with `methanol.hsc` open, `pywin32` |
| Re-run the MAGNUS baseline | WSL/Ubuntu with MAGNUS built — <https://github.com/omega-icl/magnus> |

MAGNUS is a C++ library with no Windows build, which is why its half of every
comparison lives in `magnus_wsl/` and reaches HYSYS through the bridge server.
Wall-clock times are therefore not directly comparable between the two
samplers: the proposed sampler is Python on Windows, MAGNUS is C++ under WSL.
Table Y in the report normalises for this by recomputing the MAGNUS runtime at
the proposed sampler's mean per-solve cost.

---

## Layout note

The Python modules sit flat at the repository root on purpose: they import each
other by bare module name and resolve their data relative to their own location
or to the working directory. Moving them into packages would mean rewriting
those paths, so the data is grouped into subdirectories instead and the code is
left as it ran.

---

## Declaration of generative AI use

Generative AI tools (Claude, Anthropic; ChatGPT, OpenAI) were used in producing
this repository, as follows.

**Code development and debugging.** These tools were used throughout, on all
of the code in this repository. The bridge between MAGNUS and Aspen HYSYS is a
case where they contributed the design and not only the code: MAGNUS runs
under WSL and HYSYS is a Windows program, so the two communicate by exchanging
files through a shared folder. That arrangement, and the protocol it uses,
came from these tools (`hysys_bridge_server.py` and the `BridgeBackend` class
in `magnus_wsl/magnus_tp_case_study.py`).

**Code comments and docstrings.** Most of the explanatory comments and
docstrings here were drafted or refined with these tools. They are deliberately
retained: they are how the author read and reasoned about the code while
writing it, and their presence was agreed with the project supervisor.

**Reading the reference implementation.** The sampler is a port of MultiNest
(Feroz et al., 2009), whose reference implementation is in Fortran, a language
the author does not read. Claude was used to work through that source line by
line. The citations of the form `[kmeans3 L259-279]` in `multinest_sampler.py`
record which routine each part of the port follows. The port itself, and the
decisions about what to keep, change, or leave out, are the author's.

**Test-problem definitions.** The analytical forms of the geometric test
problems in the sampler's example catalogue — including the Banana + Island,
three-torus and single-ellipse cases behind Figure 2 — were drafted with
Claude. Which problems to include, and what each is there to test, was decided
by the author.

**This README** was written with Claude.

Not produced by these tools: the design of the study, the selection of the
comparisons and test problems, the runs behind the reported results, the
results, and their interpretation. All AI output was reviewed and verified by
the author, who takes full responsibility for the content of this repository.
