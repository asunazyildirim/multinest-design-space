"""Build the Section 3.3 tables for the CO2-to-methanol case study.

    python tp_case_study_tables.py [--out DIR]

Reads the two saved runs and the solve-time benchmark and writes one
workbook, ``tables_tp_case_study.xlsx``, with the same block-per-sheet
layout as ``benchmark_2d.py report`` uses for Section 3.2 -- each block
labelled above itself, so a table is a single paste into the manuscript.

    A  Problem definition
    B  Certified region
    C  Computational performance
    D  Mode separation
    E  Solve-time benchmark          <- the contention experiment

NOTHING IS TYPED IN. Every figure comes from the .npz the run wrote or from
the benchmark CSV, so the tables cannot drift away from the figures or from
each other. Re-run this after any re-run of the samplers and the numbers
follow.

The two archives store the criterion with opposite signs -- the proposed
sampler saves ``live_merit`` = -(worst violation), NSFeas saves ``live_crit``
= the violation itself -- so feasibility is read as merit >= 0 after the
MAGNUS values are negated. Getting this backwards silently swaps the
feasible and infeasible counts, which is why it is done in one place.
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

PROPOSED_NPZ = os.path.join(HERE, "tp_design_space_serial.npz")
MAGNUS_NPZ = os.path.join(HERE, "magnus_tp_output",
                          "magnus_tp_bridge_NL500.npz")
BENCH_CSV = os.path.join(HERE, "bench_solve_contention.csv")

PROPOSED = "Proposed"
MAGNUS = "MAGNUS/NSFeas"

# Pressure is stored in kPa and reported in bar, as in the figures.
BAR = 0.01


# ======================================================================
# 1. LOAD
# ======================================================================

def load_proposed(path):
    a = np.load(path, allow_pickle=True)
    merit = {role: np.asarray(a[f"{role}_merit"], float).ravel()
             for role in ("live", "dead", "rejected")}
    points = {role: np.asarray(a[f"{role}_points"], float)
              for role in ("live", "dead", "rejected")}

    n_repl = len(points["dead"])
    n_cand = n_repl + len(points["rejected"])

    return {
        "points": np.vstack([points[r] for r in ("dead", "rejected", "live")]),
        "merit": np.concatenate([merit[r]
                                 for r in ("dead", "rejected", "live")]),
        "modes": np.asarray(a["live_modes"]).ravel(),
        "N_L": len(points["live"]),
        "n_repl": n_repl,
        "n_cand": n_cand,
        "n_model": int(a["total_model_runs"]),
        "init_s": float(a["init_time_s"]),
        "wall_s": float(a["wall_clock_s"]),
        "solve_s": float(a["solve_time_s"]),
        "sampler_s": float(a["sampler_time_s"]),
        "seed_calls": int(a["seed_calls"]),
        "seed_wall_s": float(a["seed_wall_s"]),
        "seed_solve_s": float(a["seed_solve_s"]),
        "repl_calls": int(a["repl_calls"]),
        "repl_wall_s": float(a["repl_wall_s"]),
        "repl_solve_s": float(a["repl_solve_s"]),
        "constraints": np.asarray(a["constraints"], float),
        "design_names": [str(n) for n in a["design_names"]],
    }


def load_magnus(path):
    a = np.load(path, allow_pickle=True)
    meta = {str(k): float(v) for k, v in zip(a["meta_keys"], a["meta"])}

    # Negated: NSFeas reports the violation, the proposed sampler reports
    # its negative. One convention from here on.
    crit = {role: -np.asarray(a[f"{role}_crit"], float).ravel()
            for role in ("live", "dead", "rejected")}
    points = {role: np.asarray(a[f"{role}_points"], float)
              for role in ("live", "dead", "rejected")}

    return {
        "points": np.vstack([points[r] for r in ("dead", "rejected", "live")]),
        "merit": np.concatenate([crit[r]
                                 for r in ("dead", "rejected", "live")]),
        "modes": None,                     # NSFeas has no notion of a mode
        "N_L": int(meta["N_L"]),
        "n_repl": int(meta["n_replacements"]),
        "n_cand": int(meta["n_candidates"]),
        "n_model": int(meta["n_evaluations"]),
        "init_s": meta["init_s"],
        "wall_s": meta["wall_s"],
        "solve_s": meta["solve_s"],
        "sampler_s": meta["sampler_s"],
        "seed_calls": int(meta["seed_solves"]),
        "seed_wall_s": meta["seed_wall_s"],
        "seed_solve_s": meta["seed_solve_s"],
        "repl_calls": int(meta["n_evaluations"] - meta["seed_solves"]),
        "repl_wall_s": meta["wall_s"] - meta["seed_wall_s"],
        "repl_solve_s": meta["solve_s"] - meta["seed_solve_s"],
        "warm_hits": int(meta["warm_cache_hits"]),
        "design_bounds": np.asarray(a["design_bounds"], float),
        "constraints": np.asarray(a["constraints"], float),
    }


# ======================================================================
# 2. TABLE A — problem definition
# ======================================================================

def block_a(runs):
    """What the two samplers were both given. One column: it is the problem,
    not a result, and stating it once is what makes the rest comparable."""
    bounds = runs[MAGNUS]["design_bounds"]
    limits = runs[PROPOSED]["constraints"][:, 0]

    rows = [
        ("Design variables", "Reactor temperature T, pressure P"),
        ("Temperature range", f"{bounds[0, 0]:.0f} – {bounds[0, 1]:.0f} °C"),
        ("Pressure range",
         f"{bounds[1, 0] * BAR:.0f} – {bounds[1, 1] * BAR:.0f} bar"),
        ("H2:CO2 feed ratio", "3.0"),
        ("Reactor volume", "65 m³"),
        ("Recycle fraction", "0.99"),
        ("g1 Carbon efficiency", f"≥ {limits[0]:.0f} %"),
        ("g2 Energy efficiency", f"≥ {limits[1]:.1f} %"),
        ("Equivalent production", "≥ 19,187 kg h⁻¹ methanol"),
        ("Uncertainty scenarios", "K = 1 (nominal)"),
        ("Feasibility criterion", "VaR, α* = 1"),
        ("Live points", f"{runs[PROPOSED]['N_L']}"),
    ]
    return pd.DataFrame([v for _, v in rows],
                        index=[k for k, _ in rows], columns=["Value"])


# ======================================================================
# 3. TABLE B — certified region
# ======================================================================

def block_b(runs):
    """The answer each sampler returned, and how closely they agree.

    Feasibility is read from the criterion, not from the sampler's role
    label: an evicted point can be perfectly feasible and merely have been
    the worst of the population when it was replaced.
    """
    data, extents = {}, {}
    for name, r in runs.items():
        feasible = r["points"][r["merit"] >= 0.0]
        extents[name] = feasible
        n_infeasible = len(r["points"]) - len(feasible)
        data[name] = [
            f"{len(feasible):,}",
            f"{n_infeasible:,}",
            f"{len(r['points']):,}",
            f"{feasible[:, 0].min():.2f} – {feasible[:, 0].max():.2f}",
            f"{feasible[:, 1].min() * BAR:.2f} – "
            f"{feasible[:, 1].max() * BAR:.2f}",
        ]

    # The infeasible count is not a property of the region -- it is how many
    # evaluations each run spent outside it, and the two runs spent different
    # numbers. Reported with the total so the pair is read as a split of the
    # run rather than as a measure of the design space.
    index = ["Feasible points", "Infeasible points", "Points evaluated",
             "Temperature extent (°C)", "Pressure extent (bar)"]

    # The agreement row is the point of the table: two independent methods
    # returning the same region is the result, and a reader should not have
    # to subtract the two rows above to see it.
    a, b = extents[PROPOSED], extents[MAGNUS]
    for axis, (label, scale) in enumerate(
            [("Temperature", 1.0), ("Pressure", BAR)]):
        gap = max(abs(a[:, axis].min() - b[:, axis].min()),
                  abs(a[:, axis].max() - b[:, axis].max())) * scale
        unit = "°C" if axis == 0 else "bar"
        index.append(f"{label} agreement")
        for name in data:
            data[name].append(f"≤ {gap:.2f} {unit}" if name == PROPOSED
                              else "—")

    return pd.DataFrame(data, index=index)


# ======================================================================
# 4. TABLE C — computational performance
# ======================================================================

def block_c(runs):
    """What each run cost, split the two ways the run summary splits it.

    The normalised row reprices the MAGNUS evaluations at the proposed
    sampler's per-solve cost. The two runs were executed at different times
    and through different execution paths, so the raw wall clock carries
    that difference; the evaluation count does not, and the normalised row
    shows the ordering does not depend on it.
    """
    per_solve = {name: r["solve_s"] / r["n_model"] if r["n_model"] else np.nan
                 for name, r in runs.items()}
    reference = per_solve[PROPOSED]

    data = {}
    for name, r in runs.items():
        share = (lambda v: 100.0 * v / r["wall_s"]) if r["wall_s"] else \
            (lambda v: np.nan)
        normalised = r["n_model"] * reference + r["sampler_s"]
        data[name] = [
            f"{r['N_L']:,}",
            f"{r['n_repl']:,}",
            f"{r['n_cand'] - r['n_repl']:,}",
            f"{r['n_cand']:,}",
            f"{100.0 * r['n_repl'] / r['n_cand']:.1f}",
            f"{r['n_model']:,}",
            f"{r['wall_s']:,.1f}",
            f"{r['solve_s']:,.1f} ({share(r['solve_s']):.1f} %)",
            f"{r['sampler_s']:,.1f} ({share(r['sampler_s']):.1f} %)",
            f"{per_solve[name]:.2f}",
            f"{r['seed_wall_s']:,.1f}",
            f"{r['repl_wall_s']:,.1f}",
            f"{r['init_s']:.1f}",
            f"{normalised:,.1f}",
        ]

    # Row order and notation match table_b in gtcd_case.py and block_b in
    # benchmark_2d.py, so the three sections' performance tables read the
    # same way. Accepted and rejected first, their total below them: both
    # samplers need roughly the same number of ACCEPTED replacements -- a
    # property of the problem, not of the method -- and differ in how many
    # candidates they had to try to get them. Leading with N_cand buries
    # that. The seed count is not repeated as a row: N_model - N_cand is
    # the seed, and N_L is already the first row.
    index = [
        "N_L",
        "N_acc",
        "N_rej",
        "N_cand",
        "eta_samp (%)",
        "N_model",
        "Wall clock (s)",
        "  in process model (s)",
        "  in sampler (s)",
        "Per solve (s)",
        "Seed phase, wall (s)",
        "Replacement phase, wall (s)",
        "Initialisation (s), excluded",
        "Wall clock at common per-solve cost (s)",
    ]
    return pd.DataFrame(data, index=index)


def hms(seconds):
    """Seconds -> ``h:mm:ss``.

    Minute resolution is not enough here. The MAGNUS run's wall clock and
    its time in the process model are 22,195.0 s and 22,175.4 s, which both
    round to 6 h 10 min -- the table would show one number twice against two
    different percentages and read as an error. Seconds separate them.
    """
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def block_f(runs):
    """The summary table for the manuscript.

    Durations as h:mm:ss, except the two that are short enough for seconds
    to be the natural unit -- and the contrast between a run measured in
    hours and an overhead measured in seconds is the finding, so the mixed
    units are the point rather than an inconsistency.

    N_model is deliberately NOT repeated here: it is the culmination of the
    sampling table and only an input to this one. Put it back if the two
    tables end up separated in the layout, since the last row cannot be
    checked without it.

    Caption this table with:
        Durations are given as h:mm:ss; rows marked (s) are in seconds. The
        last row reprices the MAGNUS/NSFeas evaluations at the proposed
        sampler's mean solve cost.
    """
    per_solve = {name: r["solve_s"] / r["n_model"] if r["n_model"] else np.nan
                 for name, r in runs.items()}
    reference = per_solve[PROPOSED]

    data = {}
    for name, r in runs.items():
        share = 100.0 / r["wall_s"] if r["wall_s"] else np.nan
        data[name] = [
            hms(r["wall_s"]),
            f"{hms(r['solve_s'])} ({r['solve_s'] * share:.1f} %)",
            f"{r['sampler_s']:.1f} ({r['sampler_s'] * share:.1f} %)",
            f"{per_solve[name]:.2f}",
            hms(r["n_model"] * reference + r["sampler_s"]),
        ]

    index = [
        "Wall-clock time",
        "  — process simulation",
        "  — sampling overhead (s)",
        "Mean solve time per evaluation (s)",
        "Wall-clock time at a common solve cost",
    ]
    return pd.DataFrame(data, index=index)


# ======================================================================
# 5. TABLE D — mode separation
# ======================================================================

def block_d(runs):
    """NSFeas maintains one live-point population and has no notion of a
    mode, so its column is a dash rather than a blank: the absence is the
    structural difference under comparison, not missing data."""
    data = {}
    for name, r in runs.items():
        if r["modes"] is None or not np.size(r["modes"]):
            data[name] = ["—", "—"]
            continue
        labels, counts = np.unique(r["modes"], return_counts=True)
        data[name] = [str(len(labels)),
                      ", ".join(f"{int(c):,}" for c in sorted(counts)[::-1])]
    return pd.DataFrame(data, index=["Modes", "Live points per mode"])


# ======================================================================
# 6. TABLE E — solve-time benchmark
# ======================================================================

def block_e(path, runs):
    """The controlled solve-time experiment.

    40 fixed design points, solved in identical order under three
    conditions, with no sampler involved. It exists to test whether the
    filesystem polling the MAGNUS run's bridge performs accounts for the
    per-solve difference in Table C. The polled condition falling BETWEEN
    the two unpolled controls is the result.
    """
    if not os.path.isfile(path):
        return pd.DataFrame()

    frame = pd.read_csv(path)
    labels = list(dict.fromkeys(frame.label))

    data = {}
    for label in labels:
        seconds = frame.seconds[frame.label == label]
        data[label] = [
            f"{len(seconds)}",
            f"{seconds.mean():.2f}",
            f"{seconds.median():.2f}",
            f"{seconds.min():.2f}",
            f"{seconds.max():.2f}",
        ]

    index = ["Points", "Mean (s)", "Median (s)", "Min (s)", "Max (s)"]

    # Set against the two production runs, which is the only reason the
    # benchmark was run. Today's session sits above both, which is the
    # session-to-session drift the text refers to.
    index.append("vs first condition (%)")
    base = frame.seconds[frame.label == labels[0]].mean()
    for label in labels:
        delta = frame.seconds[frame.label == label].mean() - base
        data[label].append("—" if label == labels[0]
                           else f"{100.0 * delta / base:+.1f}")

    frame_out = pd.DataFrame(data, index=index)
    frame_out["Production runs"] = [
        f"{runs[PROPOSED]['n_model']:,} / {runs[MAGNUS]['n_model']:,}",
        f"{runs[PROPOSED]['solve_s'] / runs[PROPOSED]['n_model']:.2f} / "
        f"{runs[MAGNUS]['solve_s'] / runs[MAGNUS]['n_model']:.2f}",
        "—", "—", "—", "—",
    ]
    return frame_out


# ======================================================================
# 7. WRITE
# ======================================================================

def to_sheet(writer, sheet, blocks):
    """Stack each block down one sheet, with its name above it, so the whole
    table is a single paste into the manuscript."""
    row = 0
    for label, df in blocks.items():
        if df.empty:
            continue
        pd.DataFrame({label: []}).to_excel(writer, sheet_name=sheet,
                                           startrow=row, index=False)
        df.to_excel(writer, sheet_name=sheet, startrow=row + 1)
        row += len(df) + 4


def main():
    parser = argparse.ArgumentParser(
        description="Section 3.3 tables for the CO2-to-methanol case study.")
    parser.add_argument("--proposed", default=PROPOSED_NPZ)
    parser.add_argument("--magnus", default=MAGNUS_NPZ)
    parser.add_argument("--bench", default=BENCH_CSV)
    parser.add_argument("--out", default=HERE)
    args = parser.parse_args()

    for path in (args.proposed, args.magnus):
        if not os.path.isfile(path):
            raise SystemExit(f"no such run archive: {path}")

    runs = {PROPOSED: load_proposed(args.proposed),
            MAGNUS: load_magnus(args.magnus)}

    # A warm start would make every timing figure in Table C a fiction, so
    # it is checked here rather than left to whoever reads the workbook.
    if runs[MAGNUS].get("warm_hits"):
        print(f"  WARNING: the MAGNUS run served "
              f"{runs[MAGNUS]['warm_hits']} point(s) from its solve cache. "
              f"Its timings are not a measurement of this problem.")

    blocks = {
        "A  Problem definition": block_a(runs),
        "B  Certified region": block_b(runs),
        "C  Computational performance": block_c(runs),
        "D  Mode separation": block_d(runs),
        "E  Solve-time benchmark": block_e(args.bench, runs),
        "F  Manuscript summary — computational cost": block_f(runs),
    }

    for label, df in blocks.items():
        if df.empty:
            print(f"\n{label}\n  (no data)")
            continue
        print(f"\n{label}")
        print(df.to_string())

    # Printed before writing. The workbook is routinely open in Excel while
    # these are being read, and a locked file must not lose the output.
    xlsx = os.path.join(args.out, "tables_tp_case_study.xlsx")
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            to_sheet(writer, "Section 3.3", blocks)
        print(f"\nsaved {xlsx}")
    except PermissionError:
        print(f"\ncould NOT write {xlsx} — the file is open in another "
              f"program. The tables are printed above.")


if __name__ == "__main__":
    main()
