"""
Two-dimensional benchmark comparison: the proposed multi-ellipsoidal nested
sampler against MAGNUS/NSFeas.

This is the Windows half of a two-machine study. It has two jobs, selected by
subcommand, and they are deliberately separate:

    python benchmark_2d.py run       runs the PROPOSED sampler and writes one
                                     .npz per (problem, seed)
    python benchmark_2d.py report    reads every .npz in the output directory,
                                     whichever sampler wrote it, and produces
                                     the paper's figure and tables

The NSFeas half runs under WSL/Ubuntu (``magnus_benchmark_2d.py``), because
``magnus`` is a C++ library with no Windows build. It writes .npz files in the
SAME schema into the same directory, so ``report`` needs neither that library
nor that machine: copy the files across and re-run it.

Keeping ``report`` separate also means the tables and the figure can be
restyled in seconds without re-running a sampler.

PROBLEMS
--------
Both come from ``multinest_sampler.EXAMPLES`` and are defined once there, so
the two halves of the study and the paper's Section 2.5 cannot drift apart:

    kusumo   s = theta*d1^2 + d2,  theta ~ N(1, sqrt(0.3)),  0.20 <= s <= 0.75
             The published benchmark of Kusumo et al. (2020), reproduced at
             their own settings (alpha* = 0.95, N_L = 500, N_theta = 100).
    banana   s = theta*(d1^2 - 0.5) + d2 + 0.3*d1,  theta ~ N(1, 0.5),
             0.00 <= s <= 0.40.  Two disconnected regions of unequal size.

WHAT IS REPORTED, AND WHY IT IS P AND NOT VaR
---------------------------------------------
Both samplers are DRIVEN by VaR_alpha*[G], which is continuous in d and
therefore orders the live points strictly; the feasibility probability P is
recorded from the same scenario sweep at no extra model evaluations. Since
VaR_alpha*[G](d) <= 0 if and only if P(d) >= alpha*, the two identify the same
certified set, and the results are reported in P: it carries the reliability
interpretation and it is the quantity Kusumo et al. tabulate.

The reliability bands used here are theirs (Table 2): 0.95 <= P, 0.70-0.95,
0.50-0.70, 0.25-0.50, P < 0.25. They are also the contour levels drawn on the
figure, so a band in the table is a line on the plot.

SEEDS
-----
One run per problem, at ``SEED``. The proposed sampler is stochastic, so a
different seed gives a different trajectory, but no seed study is reported,
because there is nothing to compare it against: NSFeas exposes no sampler
seed -- its proposals come from a Sobol sequence seeded with a constant in its
own source -- and repeating it at the same scenario seed was measured to give
bit-identical live points on both problems. Quoting a spread for one side only
would invite the comparison to be read as noise against no-noise. ``--seed``
overrides the value used here.

Note that NSFeas is not seed-free: its answer DOES change with the scenario
seed, because that changes the K scenarios it is given. What it lacks is a
seed for the search itself.

Usage
-----
    python benchmark_2d.py run                     # both problems
    python benchmark_2d.py run --problems banana --seed 23
    python benchmark_2d.py report                  # figure + tables + xlsx
    python benchmark_2d.py report --no-figure      # tables only
    python benchmark_2d.py report --field          # keep the shaded P field
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")           # batch: no window, nothing to block on

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multinest_sampler as mn

# The seed population announces itself once it takes longer than a couple of
# seconds, which is meant for a slow simulator and only breaks up this
# script's own one-line-per-run progress. run() is already silenced through
# log_every=None; this is the other channel.
mn.SEED_PROGRESS_AFTER = float("inf")

OUT_DIR = "benchmark_2d_output"
REF_DIR = "reference_grids_output"          # written by reference_grids.py

# Index into multinest_sampler.EXAMPLES, plus the title prefix that index is
# expected to carry. The check turns a reordered catalogue into a loud error
# instead of a silently mislabelled figure.
PROBLEMS = {
    "kusumo": dict(index=3, expect="4 ·", label="Kusumo et al. (2020)"),
    "banana": dict(index=2, expect="3 ·", label="Banana and island"),
}

SEED = 1                       # one run per problem; see the note below

# Kusumo et al. (2020), Table 2. Also CriterionDisplay.P_ALPHAS, so the table
# rows and the figure contours are the same thresholds.
BANDS = [(0.95, 1.01, "0.95 ≤ P"),
         (0.70, 0.95, "0.70 ≤ P < 0.95"),
         (0.50, 0.70, "0.50 ≤ P < 0.70"),
         (0.25, 0.50, "0.25 ≤ P < 0.50"),
         (0.00, 0.25, "P < 0.25")]

PROPOSED = "proposed"
NSFEAS = "nsfeas"
METHOD_LABEL = {PROPOSED: "Proposed", NSFEAS: "MAGNUS/NSFeas"}


# ======================================================================
# 1. SHARED
# ======================================================================

def resolve(key: str) -> dict:
    """The catalogue entry for one problem, with its identity checked."""
    meta = PROBLEMS[key]
    ex = mn.EXAMPLES[meta["index"]]
    if not ex["title"].startswith(meta["expect"]):
        raise SystemExit(
            f"catalogue mismatch for '{key}': EXAMPLES[{meta['index']}] is "
            f"{ex['title']!r}, expected one starting {meta['expect']!r}. "
            "The catalogue was reordered — fix PROBLEMS in this file.")
    return ex


def npz_name(problem: str, method: str, seed: Optional[int]) -> str:
    """One naming scheme for both halves of the study.

    ``magnus_benchmark_2d.py`` writes the same names, which is what lets
    ``report`` load everything without knowing who produced it.
    """
    tag = "" if seed is None else f"_s{seed}"
    return f"points_{method}_{problem}{tag}.npz"


def settings(ex: dict) -> dict:
    """The numerical settings both samplers must agree on."""
    kw = ex["sampler_kw"]
    return dict(N_L=int(kw["N_L"]), N_theta=int(kw["N_theta"]),
                alpha_star=float(kw["alpha_star"]),
                criterion=kw.get("feas_criterion", "VaR"))


# ======================================================================
# 2. RUN  (proposed sampler)
# ======================================================================

def run_one(key: str, seed: int, out_dir: str) -> dict:
    ex = resolve(key)
    st = settings(ex)
    kw = ex["sampler_kw"]

    model = mn.ProcessModel(equation=ex["equation"],
                            uncertainty=ex["uncertainty"],
                            constraints=ex["constraints"],
                            name=ex["title"])
    estimator = model.make_estimator(uncertainty=ex["uncertainty"],
                                     N_theta=st["N_theta"],
                                     feas_criterion=st["criterion"])

    # Everything in sampler_kw belongs to MultiNestSampler EXCEPT the three
    # keys that belong elsewhere: N_theta and feas_criterion configure the
    # estimator (built above) and seed is an argument of run().
    sampler_kw = {k: v for k, v in kw.items()
                  if k not in ("N_theta", "feas_criterion", "seed")}

    print(f"  {key:8} seed {seed:<4} N_L={st['N_L']} "
          f"N_theta={st['N_theta']} {st['criterion']} …", end="", flush=True)

    t0 = time.perf_counter()
    res = mn.MultiNestSampler(estimator=estimator,
                              design_space=ex["design_space"],
                              **sampler_kw).run(seed=seed, log_every=None)
    wall = time.perf_counter() - t0

    path = os.path.join(out_dir, npz_name(key, PROPOSED, seed))
    np.savez_compressed(
        path,
        live_points=res.live_points, live_merit=res.live_merit,
        live_probs=res.live_probs, live_mode_ids=res.live_mode_ids,
        dead_points=res.dead_points, dead_merit=res.dead_merit,
        dead_probs=res.dead_probs,
        rejected_points=res.rejected_points,
        rejected_merit=res.rejected_merit,
        rejected_probs=res.rejected_probs,
        criterion=st["criterion"], alpha_star=st["alpha_star"],
        N_L=st["N_L"], n_theta=st["N_theta"], seed=seed,
        n_replacements=int(res.n_replacements),
        n_candidates=int(res.n_candidate_estimates),
        model_runs=int(res.total_model_runs),
        wall_s=float(wall),
        converged=bool(res.converged),
        n_modes=int(len(np.unique(res.live_mode_ids))),
        sampler=PROPOSED, problem=key,
    )
    print(f" {wall:6.1f}s  {res.n_replacements:,} repl, "
          f"{res.n_candidate_estimates:,} cand"
          f"{'' if res.converged else '  [NOT CONVERGED]'}")
    return dict(wall=wall, converged=res.converged)


def cmd_run(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    print(f"Proposed sampler — {len(args.problems)} problem(s), "
          f"seed {args.seed}\n")
    stalled = []
    for key in args.problems:
        r = run_one(key, args.seed, args.out)
        if not r["converged"]:
            stalled.append(key)
    if stalled:
        print("\n  WARNING — runs that stopped before certifying every live "
              "point:")
        for key in stalled:
            print(f"    {key}")
    print(f"\nwrote .npz files to {args.out}/")


# ======================================================================
# 3. LOAD
# ======================================================================

def load_runs(out_dir: str) -> Dict[tuple, dict]:
    """Every .npz in ``out_dir``, keyed by (problem, method, seed)."""
    runs = {}
    if not os.path.isdir(out_dir):
        raise SystemExit(f"no such directory: {out_dir} — run the 'run' "
                         "subcommand first")
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith("points_") and name.endswith(".npz")):
            continue
        d = np.load(os.path.join(out_dir, name), allow_pickle=False)
        method = str(d["sampler"])
        problem = str(d["problem"])
        seed = int(d["seed"]) if "seed" in d.files else None
        r = {k: d[k] for k in d.files}

        # NSFeas returns a probability of zero as about -7.5e-16, and a
        # membership test of the form ``p >= 0`` then silently drops every
        # such point: 1,072 of 2,956 on one of these problems, which made the
        # table totals disagree with the candidate counts. A probability
        # estimate cannot lie outside [0, 1], so clip here, once, where both
        # the tables and the figure pick it up. The magnitude involved is
        # 1e-16; nothing but the sign is being corrected.
        for key in ("live_probs", "dead_probs", "rejected_probs"):
            if key in r and np.size(r[key]):
                r[key] = np.clip(np.asarray(r[key], dtype=float), 0.0, 1.0)

        runs[(problem, method, seed)] = r
    if not runs:
        raise SystemExit(f"no points_*.npz found in {out_dir}/")
    return runs


def check_settings(runs: dict, problems: List[str]) -> List[str]:
    """Every run must have used this problem's N_L, N_theta and alpha*.

    The NSFeas half runs on another machine and cannot import the catalogue,
    so it writes those three numbers out by hand. Duplicated constants drift;
    this is where the drift is caught, before it becomes a table comparing two
    different experiments.
    """
    problems_seen = {(p, m) for (p, m, _) in runs}
    complaints = []
    for key in problems:
        st = settings(resolve(key))
        for (p, m) in sorted(problems_seen):
            if p != key:
                continue
            for _, _, seed in [k for k in runs if k[0] == p and k[1] == m]:
                r = runs[(p, m, seed)]
                for field, want in (("N_L", st["N_L"]),
                                    ("n_theta", st["N_theta"]),
                                    ("alpha_star", st["alpha_star"])):
                    if field not in r:
                        continue
                    got = float(np.asarray(r[field]))
                    if abs(got - float(want)) > 1e-9:
                        complaints.append(
                            f"{key} / {METHOD_LABEL.get(m, m)}"
                            f"{'' if seed is None else f' seed {seed}'}: "
                            f"{field} = {got:g}, catalogue says {want:g}")
    return complaints


def pick(runs: dict, problem: str, method: str,
         seed: Optional[int] = None) -> Optional[dict]:
    """One run. ``seed=None`` takes whatever single run exists (NSFeas)."""
    if seed is not None and (problem, method, seed) in runs:
        return runs[(problem, method, seed)]
    matches = [v for (p, m, _), v in runs.items()
               if p == problem and m == method]
    return matches[0] if matches else None


def seeds_of(runs: dict, problem: str, method: str) -> List[int]:
    return sorted(s for (p, m, s) in runs
                  if p == problem and m == method and s is not None)


# ======================================================================
# 4. TABLE A — reliability-range breakdown
# ======================================================================

def band_counts(probs: np.ndarray) -> List[int]:
    v = np.asarray(probs, dtype=float).ravel()
    return [int(np.sum((v >= lo) & (v < hi))) for lo, hi, _ in BANDS]


def methods_present(runs: dict, key: str) -> List[str]:
    return [m for m in (PROPOSED, NSFEAS) if pick(runs, key, m) is not None]


def block_a(runs: dict, key: str) -> pd.DataFrame:
    """Kusumo et al. Table 2 for one problem: rows are reliability ranges,
    columns are samplers.

    Counted over EVERY point the run scored — live, dead and rejected — so the
    lower bands are populated. Restricting to live points would put the whole
    count in the top band for both samplers and say nothing.
    """
    data = {}
    for method in methods_present(runs, key):
        r = pick(runs, key, method)
        probs = np.concatenate([
            np.asarray(r[k], dtype=float).ravel()
            for k in ("live_probs", "dead_probs", "rejected_probs")
            if k in r and np.size(r[k])])
        counts = band_counts(probs)

        # Every scored design must land in exactly one band, so the total has
        # to be N_L + N_cand. It once was not: a sampler that returns P = 0 as
        # -7.5e-16 fell out of the lowest band and 36 % of the points vanished
        # without a trace. Assert the arithmetic rather than trust it.
        want = int(r["N_L"]) + int(r["n_candidates"])
        got = int(sum(counts))
        if got != want:
            raise SystemExit(
                f"\n  band counts do not add up for {key} / "
                f"{METHOD_LABEL[method]}: {got:,} binned against "
                f"{want:,} = N_L + N_cand.\n  {want - got:,} scored designs "
                f"fell into no band — check the probabilities in the .npz "
                f"for values outside [0, 1] or NaN.")

        data[METHOD_LABEL[method]] = counts + [got]

    index = [lab for _, _, lab in BANDS] + ["Total"]
    return pd.DataFrame(data, index=index)


# ======================================================================
# 5. TABLE B — computational performance
# ======================================================================

def block_b(runs: dict, key: str) -> pd.DataFrame:
    """Computational performance for one problem: rows are measures, columns
    are samplers.

    Wall time is reported but is the weakest of these numbers: the two
    implementations are Python and C++, and on problems whose model is an
    analytic expression it measures the language rather than the algorithm.
    N_model is the metric that carries over to an expensive simulator.
    """
    # Row order: accepted and rejected first, their total below them.
    # Both samplers need roughly the same number of ACCEPTED replacements --
    # that is a property of the problem, not of the method -- and differ in
    # how many candidates they had to try to get them. Putting N_acc and
    # N_rej adjacent makes that visible; leading with N_cand buries it.
    data = {}
    for method in methods_present(runs, key):
        r = pick(runs, key, method)
        acc, cand = int(r["n_replacements"]), int(r["n_candidates"])
        eff = 100.0 * acc / cand if cand else float("nan")
        data[METHOD_LABEL[method]] = [
            f"{int(r['N_L']):,}",
            f"{acc:,}", f"{cand - acc:,}", f"{cand:,}", f"{eff:.1f}",
            f"{int(r['model_runs']):,}", f"{float(r['wall_s']):.1f}"]

    index = ["N_L", "N_acc", "N_rej", "N_cand", "eta_samp (%)",
             "N_model", "wall (s)"]
    return pd.DataFrame(data, index=index)


def block_c(runs: dict, key: str) -> pd.DataFrame:
    """Separated modes for one problem.

    NSFeas maintains a single live-point population and has no notion of a
    mode, so its column is a dash rather than a blank: the absence is the
    structural difference under comparison, not missing data.
    """
    data = {}
    for method in methods_present(runs, key):
        r = pick(runs, key, method)
        if "live_mode_ids" in r and np.size(r["live_mode_ids"]):
            ids = np.asarray(r["live_mode_ids"]).ravel()
            lab, cnt = np.unique(ids, return_counts=True)
            data[METHOD_LABEL[method]] = [
                str(len(lab)),
                ", ".join(f"{int(c):,}" for c in sorted(cnt)[::-1])]
        else:
            data[METHOD_LABEL[method]] = ["—", "—"]

    return pd.DataFrame(data, index=["Modes", "Live points per mode"])


# ======================================================================
# 6. FIGURE — 2 x 2, rows = problem, columns = sampler
# ======================================================================

def load_reference(problem: str, ref_dir: str):
    """The P field written by ``reference_grids.py --criterion P``."""
    path = os.path.join(ref_dir, f"reference_{problem}_P.npz")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            f"  build it with:  python reference_grids.py --criterion P "
            f"--cases {problem}")
    d = np.load(path, allow_pickle=False)
    return d["D1"], d["D2"], d["V"], float(d["alpha_star"])


def scored_points(r: dict):
    """Every point the run evaluated — live, dead and rejected — with its P.

    Kusumo et al. plot the whole sample set rather than the final live set,
    and Table 2 counts the same points. Restricting to live points would paint
    every marker in the top band and lose the reliability colouring entirely.
    """
    pts, prb = [], []
    for pk, qk in (("live_points", "live_probs"),
                   ("dead_points", "dead_probs"),
                   ("rejected_points", "rejected_probs")):
        if pk in r and np.size(r[pk]):
            pts.append(np.asarray(r[pk], dtype=float).reshape(-1, 2))
            prb.append(np.asarray(r[qk], dtype=float).ravel())
    return np.vstack(pts), np.concatenate(prb)


def nominal_mask(ex: dict, D1, D2):
    """Where the constraints hold at the NOMINAL parameter value.

    Kusumo et al. shade this set grey behind the probabilistic contours, and
    the contrast is part of the message: the nominal region is the optimistic
    answer, and the certified region sits strictly inside it. It costs nothing
    to compute -- one evaluation per grid node at theta = mu, no sampling.
    """
    mu = float(np.atleast_1d(getattr(ex["uncertainty"], "mu", 1.0))[0])
    flat = np.column_stack([D1.ravel(), D2.ravel()])
    s = np.array([np.atleast_1d(ex["equation"](d, mu)).ravel()
                  for d in flat])
    ok = np.ones(s.shape[0], dtype=bool)
    for i, (lo, hi) in enumerate(ex["constraints"]):
        ok &= (s[:, i] >= lo) & (s[:, i] <= hi)
    return ok.reshape(D1.shape)


def _axis_name(name: str) -> str:
    """``d1`` -> ``$d_1$``, so the index sets as a subscript.

    Applied to whatever the design space calls its variables rather than to
    a hard-coded pair, and names that are not letters-then-digits are left
    exactly as they are -- a physical name like ``recycle_fraction`` must
    not be dragged into mathtext, where the underscore would subscript.
    """
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    return f"${match.group(1)}_{{{match.group(2)}}}$" if match else name


def draw_panel(ax, ref, disp, ds, points, probs, title: str,
               nominal=None, field: bool = False) -> None:
    """One panel: nominal region, iso-reliability contours, sampled points.

    The shaded criterion field is OFF by default. Colouring both the
    background and the markers from the same scale makes them compete, and
    the markers are the result; the field is only context. Kusumo et al.
    leave it white for the same reason. ``--field`` restores it as a neutral
    greyscale, which stays out of the markers' way.
    """
    D1, D2, V, _ = ref
    if field:
        ax.pcolormesh(D1, D2, V, cmap="Greys", vmin=0.0, vmax=1.0,
                      alpha=0.35, shading="auto", zorder=0)
    if nominal is not None:
        ax.contourf(D1, D2, nominal.astype(float), levels=[0.5, 1.5],
                    colors=["0.82"], zorder=1)

    ax.scatter(points[:, 0], points[:, 1], s=6,
               c=disp.point_colors(probs), lw=0, zorder=3)
    # Contours last and on top: a point and the iso-line bounding its band
    # share a colour by design, so the line has to be drawn over the markers
    # or it disappears into them exactly where it matters most.
    for level, colour, _lab in disp.contour_levels():
        ax.contour(D1, D2, V, levels=[level], colors=[colour],
                   linewidths=1.7, zorder=4)
    names = getattr(ds, "names", None) or ["d1", "d2"]
    ax.set_xlabel(_axis_name(names[0]))
    ax.set_ylabel(_axis_name(names[1]))
    ax.set_xlim(ds.bounds[0])
    ax.set_ylim(ds.bounds[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)


def figure(runs: dict, problems: List[str], out_dir: str, ref_dir: str,
           dpi: int, field: bool = False) -> None:
    """Final live points of each sampler over the reference field.

    Two panels per problem rather than one overlaid panel, following Figure 2
    of Kusumo et al.: at 500-900 points per sampler an overlay is unreadable,
    and the comparison the reader makes is between two answers, not between
    two colours.
    """
    methods = [m for m in (PROPOSED, NSFEAS)
               if any(pick(runs, k, m) is not None for k in problems)]
    fig, axes = plt.subplots(len(problems), len(methods),
                             figsize=(5.4 * len(methods), 4.9 * len(problems)),
                             squeeze=False)

    for i, key in enumerate(problems):
        ex = resolve(key)
        ref = load_reference(key, ref_dir)
        disp = mn.CriterionDisplay("P", ref[3])
        nom = nominal_mask(ex, ref[0], ref[1])
        for j, method in enumerate(methods):
            ax = axes[i][j]
            r = pick(runs, key, method)
            if r is None:
                ax.text(0.5, 0.5, f"{METHOD_LABEL[method]}\nnot run",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            pts, prb = scored_points(r)
            draw_panel(ax, ref, disp, ex["design_space"], pts, prb,
                       f"({'abcd'[i * len(methods) + j]}) "
                       f"{PROBLEMS[key]['label']} — {METHOD_LABEL[method]}",
                       nominal=nom, field=field)

    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = os.path.join(out_dir, f"figure_benchmark_2d.{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


# ======================================================================
# 7. REPORT
# ======================================================================

TABLES = [("A", "sampled points by reliability range "
                "(every point scored by the run)", "A_reliability_ranges"),
          ("B", "computational performance", "B_performance"),
          ("C", "separated modes", "C_modes")]


def blocks(runs: dict, key: str) -> dict:
    """The three blocks for one problem, all shaped the same way: measures
    down the rows, samplers across the columns. One block per problem rather
    than one long table, so a reader compares the two columns of a block
    instead of hunting for the paired row."""
    return {"A": block_a(runs, key), "B": block_b(runs, key),
            "C": block_c(runs, key)}


def show(letter: str, caption: str, per_problem: dict) -> None:
    title = f"Table {letter} — {caption}"
    print(f"\n{title}\n{'=' * len(title)}")
    for label, df in per_problem.items():
        if df.empty:
            continue
        print(f"\n  {label}")
        print("  " + "-" * len(label))
        print("\n".join("  " + line for line in
                        df.to_string().splitlines()))


def to_sheet(writer, sheet: str, per_problem: dict) -> None:
    """Stack each problem's block down one sheet, with its name above it, so
    the whole table is a single paste into the manuscript."""
    row = 0
    for label, df in per_problem.items():
        if df.empty:
            continue
        pd.DataFrame({label: []}).to_excel(writer, sheet_name=sheet,
                                           startrow=row, index=False)
        df.to_excel(writer, sheet_name=sheet, startrow=row + 1)
        row += len(df) + 4


def cmd_report(args) -> None:
    runs = load_runs(args.out)
    present = sorted({(p, m) for (p, m, _) in runs})
    print(f"loaded {len(runs)} run file(s) from {args.out}/")
    for p, m in present:
        # A sampler seed is recorded only where one exists. NSFeas has none --
        # its proposals come from a Sobol sequence seeded in its own source --
        # but the SCENARIO draw does have one, and that is what makes its run
        # reproducible, so say which rather than just "deterministic".
        ss = seeds_of(runs, p, m)
        if ss:
            how = "  sampler seed " + ", ".join(map(str, ss))
        else:
            r = pick(runs, p, m) or {}
            sc = r.get("scenario_seed")
            how = ("  deterministic sampler"
                   + (f", scenario seed {int(sc)}" if sc is not None else ""))
        print(f"    {p:8} {METHOD_LABEL.get(m, m):16}{how}")

    complaints = check_settings(runs, args.problems)
    if complaints:
        print("\n  SETTINGS MISMATCH — these runs are not comparable:")
        for c in complaints:
            print(f"    {c}")
        raise SystemExit(
            "\n  Refusing to build a table from runs at different settings. "
            "Fix PROBLEMS in magnus_benchmark_2d.py (or the catalogue) and "
            "re-run the affected side.")

    missing = [(k, m) for k in args.problems for m in (PROPOSED, NSFEAS)
               if pick(runs, k, m) is None]
    if missing:
        print("\n  NOT PRESENT (the corresponding rows/panels are omitted):")
        for k, m in missing:
            where = ("run 'benchmark_2d.py run'" if m == PROPOSED
                     else "run magnus_benchmark_2d.py under WSL and copy the "
                          ".npz files here")
            print(f"    {PROBLEMS[k]['label']} / {METHOD_LABEL[m]}  →  {where}")

    built = {key: blocks(runs, key) for key in args.problems
             if methods_present(runs, key)}
    laid_out = {letter: {PROBLEMS[k]["label"]: b[letter]
                         for k, b in built.items()}
                for letter, _c, _s in TABLES}

    # Print before writing. The .xlsx is routinely open in Excel while these
    # numbers are being read, and a locked file must not cost the tables.
    for letter, caption, _sheet in TABLES:
        show(letter, caption, laid_out[letter])

    xlsx = os.path.join(args.out, "tables_benchmark_2d.xlsx")
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            for letter, _caption, sheet in TABLES:
                to_sheet(writer, sheet, laid_out[letter])
        print(f"\n  saved {xlsx}")
    except PermissionError:
        print(f"\n  could NOT write {xlsx} — the file is open in another "
              f"program. The tables above are complete; close it and re-run "
              f"to refresh the spreadsheet.")

    if not args.no_figure:
        figure(runs, args.problems, args.out, args.ref, args.dpi, args.field)


# ======================================================================
# 8. CLI
# ======================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="2-D benchmark: proposed sampler vs MAGNUS/NSFeas")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = dict(problems=dict(nargs="+", default=list(PROBLEMS),
                                choices=list(PROBLEMS)))

    r = sub.add_parser("run", help="run the proposed sampler")
    r.add_argument("--problems", **common["problems"])
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--out", default=OUT_DIR)
    r.set_defaults(func=cmd_run)

    q = sub.add_parser("report", help="tables and figure from saved runs")
    q.add_argument("--problems", **common["problems"])
    q.add_argument("--out", default=OUT_DIR)
    q.add_argument("--ref", default=REF_DIR,
                   help="directory holding reference_<problem>_P.npz")
    q.add_argument("--no-figure", action="store_true")
    q.add_argument("--field", action="store_true",
                   help="shade the P field behind the points, in greyscale; "
                        "off by default so the markers carry the colour")
    q.add_argument("--dpi", type=int, default=300)
    q.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
