"""
Design-space characterisation over (T, P), Rxn-8 activation energies as theta.
SERIAL VARIANT -- one HYSYS instance, one solve at a time.

This is run_sampler_kinetics.py with the parallel path removed. It exists so
the CO2-to-methanol case study can be timed against a reference method that
cannot distribute its model evaluations: comparing a 4-instance run against a
single-threaded one would credit the sampler for hardware rather than for
needing fewer solves. Everything else -- design space, thresholds, sampler
settings -- is identical, so the two runs are directly comparable and the
parallel version can be used for the same study once the timing is recorded.

Section 1 is the only part you edit; the rest reads those names.

K = 1, so this is a nominal characterisation -- the baseline a later
multi-scenario run is compared against. To characterise under uncertainty,
add rows to THETA_SAMPLES and weights to THETA_WEIGHTS.

Requires the HYSYS case open.  Run with:

    python run_sampler_kinetics_serial.py

No case path is configured: the script attaches to whichever case is open in
HYSYS. Set HYSYS_CASE only when several are open at once.
"""

import atexit
import os
import sys
import time
import traceback

# The sampler, run_case and the connection helper are imported as siblings of
# this file, so a clone of the repository runs without any path editing. Only
# this directory is added: other checkouts on the machine may carry their own
# convergence.py / sensitivity_study.py, and picking those up silently would
# change what a "converged" solve means.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pythoncom

import multinest_sampler as mn

from hysys_connection import get_open_hysys_case
from simulation_runner import run_case, initialise_hysys_objects


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

# --- 1a. Case -------------------------------------------------------
# Nothing to set. The case must already be OPEN in HYSYS, and section 2
# attaches to whichever one that is -- no path is configured, because the
# script cannot open a case anyway and a hard-coded path only adds a second
# thing that has to agree with what is on screen.
#
# Set HYSYS_CASE only to disambiguate when SEVERAL cases are open at once:
#
#   $env:HYSYS_CASE = "C:\HYSYS_cases\methanol_instance0.hsc"   (PowerShell)
#   set HYSYS_CASE=C:\HYSYS_cases\methanol_instance0.hsc        (cmd)
#
# It must name a case that is open; it does not cause one to be opened.
CASE_PATH = os.environ.get("HYSYS_CASE")     # None -> discover it

# --- 1b. Design variables -------------------------------------------
# Order = how run_design unpacks d = first CSV columns. Names double as
# run_case keywords.
#
# (T, P). Measured on the 130-point sweep, the two metrics behave
# differently in the two directions:
#
#                        in T                    in P
#   C_eff       ridge, peak 250-260 C     RISES  (+5.4 pts at 300 C)
#   E_eff       ridge, peak 240-250 C     FALLS  (-1.7 pts across the box)
#
# The interior ridge closes the region in T; the split in P is what makes
# both constraints bite -- higher pressure buys conversion and pays for it
# in compressor work, which is what energy efficiency charges for. At the
# thresholds in 1d the region comes out CLOSED and interior, touching none
# of the four box edges. See tp_case_study_proposal.md.
#
# This box also suits the uncertainty better: Ea enters through
# exp(-Ea/RT), so theta interacts with a design variable here, where at a
# fixed T it could only shift the region rigidly.
DESIGN_NAMES  = ["temperature", "pressure"]
DESIGN_BOUNDS = [
    (210.0,   300.0),      # temperature [C]    -- Cu/ZnO/Al2O3 window
    (5000.0, 11000.0),     # pressure    [kPa]  -- 50 to 110 bar
]

# --- 1c. Fixed inputs -----------------------------------------------
# Ratio at the stoichiometry of CO2 + 3 H2 -> CH3OH + H2O, recycle at the
# top of the range the ratio/recycle grid covered. These are the values the
# 130-point T-P sweep was measured at, so the thresholds in 1d mean what
# they were measured to mean.
FIXED_KWARGS = {
    "ratio":            3.0,
    "volume":           65.0,
    "recycle_fraction": 0.99,
}

# --- 1d. Constrained outputs ----------------------------------------
# OUTPUT_KEYS and CONSTRAINTS are read row by row and must stay in step.
#
# The study specifies THREE things -- methanol production, carbon
# efficiency, energy efficiency -- but only TWO of them are independent,
# and no choice of thresholds can change that. The CO2 fresh feed is fixed
# at 650 kmol/h over this box, so C_eff = methanol_kmol/650 x 100 and the
# two are proportional. Measured on all 130 points of the T-P sweep:
#
#   MeOH [kg/h] / C_eff [%]  =  208.5096 .. 208.5713    (spread 296 ppm)
#
# So a production target IS a carbon-efficiency target; whichever is
# tighter binds and the other cannot cut a point it admits. g1 below is
# written in the C_eff form to avoid a duplicate row, and reads as
#
#   C_eff >= 92 %   <=>   MeOH >= 19187 kg/h   (~153.5 kt/yr at 8000 h)
#
# Thresholds measured on sensitivity_results_20260731_174218.csv (10 x 13
# full factorial at ratio 3.0, V 65, recycle 0.99, 130/130 converged):
# this pair certifies 24.6 % of the box as a CLOSED INTERIOR region that
# touches none of the four box edges, spanning T 240-270 C, P 55-105 bar.
# C_eff cuts 18 points E_eff lets through, E_eff cuts 14 that C_eff lets
# through, so neither is decoration.
OUTPUT_KEYS = [
    "C_efficiency_pct",
    "energy_efficiency_pct",
]

CONSTRAINTS = [
    (92.0, np.inf),        # carbon efficiency  [%]  == MeOH >= 19187 kg/h
    (81.3, np.inf),        # energy efficiency  [%]  -- falls with pressure
]

# Alternatives on the same grid, if the region wants resizing.
# Format: (C_eff, E_eff) -> feasible %, C cuts / E cuts, box edges touched.
#   (92.0, 81.4)  ->  20.8 %   11 / 19,  0 edges
#   (92.5, 81.3)  ->  15.4 %   30 /  8,  0 edges   -- the sensitivity case
#   (91.5, 81.3)  ->  30.0 %   11 / 24,  1 edge    -- opens at a box edge
#   (92.0, 81.0)  ->  32.3 %   29 /  4,  1 edge    -- E_eff nearly idle
#   (93.0, ...)   ->  DO NOT: 4.6 % or less, too few points to resolve

assert len(OUTPUT_KEYS) == len(CONSTRAINTS)

# --- 1e. Uncertainty scenarios --------------------------------------
# theta = (Rxn-8 forward Ea, reverse Ea), the RWGS step of vandal2013, in
# HYSYS internal energy units.
EA_FORWARD_NOMINAL = 98089.87724
EA_REVERSE_NOMINAL = 58392.0

THETA_SAMPLES = np.array([[EA_FORWARD_NOMINAL, EA_REVERSE_NOMINAL]])
THETA_WEIGHTS = np.array([1.0])

assert THETA_SAMPLES.ndim == 2
assert len(THETA_SAMPLES) == len(THETA_WEIGHTS)

# --- 1f. Sampler ----------------------------------------------------
# Only what the study fixes. F_threshold, min_pt and multimodal are left
# unset so MultiNestSampler supplies its own defaults (1.1, 2*(D+1) and
# True) -- one less set of numbers to justify in the paper, and nothing
# here was tuned away from them except min_pt.
FEAS_CRITERION = "VaR"       # "P" | "VaR" | "CVaR"
N_L            = 500
ALPHA_STAR     = 1

RUN_SEED       = 20
LOG_EVERY      = 1
LOG_HEARTBEAT  = 20

# --- 1g. Live monitor -----------------------------------------------
# Rewrites OUT_SNAPSHOT every MONITOR_EVERY seconds while the run goes:
# leave an image viewer open on it and it acts as a live display. Costs
# nothing -- it draws state the sampler already emits, no extra solves.
# Set MONITOR_EVERY to None to switch it off.
#
# Worth switching off for the run whose wall clock is being reported: the
# redraw is cheap but it is not part of the method being timed.
MONITOR_EVERY = 60.0

# Written next to this script, with a _serial suffix so a serial run and a
# parallel one can sit side by side without either overwriting the other.
OUT_NPZ      = os.path.join(HERE, "tp_design_space_serial.npz")
OUT_CSV      = os.path.join(HERE, "tp_design_space_serial.csv")
OUT_SNAPSHOT = os.path.join(HERE, "tp_live_serial.png")

# --- 1h. Console log ------------------------------------------------
# Everything the run prints -- the progress lines, the convergence
# complaints, the traceback if it dies -- mirrored to a text file as it is
# printed. Truncated at start like OUT_CSV, so copy it elsewhere before
# re-running if the previous run's log still matters.
#
# NOT multinest_sampler's _TeeLogger: that one holds the whole log in
# memory and writes it in save(), so a run killed four hours in leaves
# nothing behind. This flushes every line.
#
# Set to None to switch it off.
OUT_LOG = os.path.join(HERE, "tp_run_serial.log")


class _Tee:
    """Mirror one stream to a file, writing through on every call."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        if not self._handle.closed:
            self._handle.write(text)
            # Flushed per write, not per buffer: the whole point is that
            # the log survives a Ctrl-C or a HYSYS crash mid-run.
            self._handle.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        if not self._handle.closed:
            self._handle.flush()

    def isatty(self):
        # tqdm and friends ask before deciding to draw a bar.
        return self._stream.isatty()


if OUT_LOG:
    # stderr is teed too, or the traceback from a failed run would be the
    # one thing missing from the log.
    _LOG_HANDLE = open(OUT_LOG, "w", encoding="utf-8")
    _REAL_STDOUT, _REAL_STDERR = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, _LOG_HANDLE)
    sys.stderr = _Tee(sys.stderr, _LOG_HANDLE)

    def _close_log():
        # Put the real streams back BEFORE closing the file. Python flushes
        # sys.stdout after the atexit handlers have run; if that flush lands
        # on a tee whose file is already shut, the interpreter exits 120
        # with "Exception ignored in sys.unraisablehook" -- a clean run
        # would look like a failed one.
        sys.stdout, sys.stderr = _REAL_STDOUT, _REAL_STDERR
        _LOG_HANDLE.close()

    atexit.register(_close_log)
    print(f"console log        : {OUT_LOG}")


# ======================================================================
# 2. CONNECT
# ======================================================================
# One connection, used by every solve.
#
# get_open_hysys_case takes a full path and matches it against the
# running-object table, so the path it is given has to be the one the open
# case was opened from -- a copy sitting on disk at that path is not enough.
# Rather than ask for a path that has to agree with what HYSYS has on
# screen, the ROT is read first and the open case supplies its own.
def _open_hysys_cases():
    """Full paths of every .hsc currently registered in the ROT."""
    pythoncom.CoInitialize()
    rot, ctx = pythoncom.GetRunningObjectTable(), pythoncom.CreateBindCtx(0)
    enum, names = rot.EnumRunning(), []
    while True:
        items = enum.Next(1)
        if not items:
            return names
        try:
            name = items[0].GetDisplayName(ctx, None)
        except Exception:
            continue           # entries that refuse to name themselves
        if name.lower().endswith(".hsc"):
            names.append(name)


_open   = _open_hysys_cases()
_listed = "\n".join(f"    {name}" for name in _open) or "    (none)"

if CASE_PATH is None:
    # Exactly one open case is the normal way to run this, so that case is
    # taken without asking. Several open at once is NOT resolved by picking
    # the first: the ROT order is arbitrary, and silently driving the wrong
    # flowsheet produces results that look fine and belong to another case.
    if len(_open) == 1:
        CASE_PATH = _open[0]
    else:
        raise SystemExit(
            ("No HYSYS case is open.\n"
             if not _open else
             f"{len(_open)} HYSYS cases are open; which one is ambiguous.\n")
            + f"open in HYSYS right now:\n{_listed}\n"
            + ("Open the case in HYSYS and re-run."
               if not _open else
               'Choose one:\n    $env:HYSYS_CASE = "<full path from above>"')
            + "\nThis script attaches to a running instance; it does not "
              "launch one."
        )
elif CASE_PATH.lower() not in [name.lower() for name in _open]:
    raise SystemExit(
        f"HYSYS_CASE names a case that is not open: {CASE_PATH}\n"
        f"  file exists on disk : {os.path.isfile(CASE_PATH)}\n"
        f"open in HYSYS right now:\n{_listed}\n"
        "Open that case, name one of the above, or unset HYSYS_CASE to let "
        "the script take the open case by itself."
    )

# Timed separately from the run. Attaching to the running-object table and
# resolving the flowsheet objects is a fixed cost of driving HYSYS at all --
# a reference method solving the same flowsheet pays it too -- so it is
# reported apart from the sampling rather than folded into it.
_t = time.perf_counter()

pythoncom.CoInitialize()
CASE    = get_open_hysys_case(CASE_PATH)
OBJECTS = initialise_hysys_objects(CASE)
OBJECTS["case_name"] = os.path.basename(CASE_PATH)

INIT_TIME = time.perf_counter() - _t

print(f"case               : {CASE_PATH}")
print("mode               : serial (one HYSYS instance)")
print(f"initialisation     : {INIT_TIME:.1f} s")


# ======================================================================
# 3. SIMULATOR
# ======================================================================
# Time spent inside HYSYS, accumulated across every solve. Subtracting it
# from the run's wall clock leaves the sampler's own cost -- the ellipsoid
# decomposition, the rejection loop, the bookkeeping -- which is the term
# the flowsheet study is meant to show being outweighed by needing fewer
# solves. A run's total is otherwise dominated by HYSYS and says nothing
# about the sampler.
MODEL_TIME  = 0.0
MODEL_CALLS = 0

# Marked when the seed population finishes, so the run splits into the two
# phases that behave differently: the seed is N_L independent solves and is
# the only part a multi-instance run can distribute, while the replacement
# loop is sequential by construction -- each candidate is drawn from
# ellipsoids fitted to the population the previous one changed. Reporting
# one total hides that the two respond differently to more hardware.
#
# The seed is one batch_merit_and_P call over exactly N_L points with no
# redraws, so the first N_L * K calls to run_design ARE the seed and
# nothing else. K is the solves one estimate costs -- one per scenario
# under WeightedScenarios -- written from THETA_SAMPLES so adding rows in
# 1e moves this boundary with them instead of silently misplacing it.
SEED_CALLS      = N_L * len(THETA_SAMPLES)
SEED_END_T      = None       # perf_counter when the seed's last solve returned
SEED_MODEL_TIME = None       # MODEL_TIME at that same instant


def run_design(d, theta):
    """One HYSYS solve. Returns the outputs in CONSTRAINTS order.

    Raises on a bad solve rather than returning a number: the model has
    on_failure="infeasible", so the raise becomes NaN upstream.
    """
    global MODEL_TIME, MODEL_CALLS, SEED_END_T, SEED_MODEL_TIME

    ea_forward, ea_reverse = theta

    # Keyed off DESIGN_NAMES rather than hard-coded, so changing the design
    # space in 1b is a one-place edit.
    where = ", ".join(f"{name}={value:g}"
                      for name, value in zip(DESIGN_NAMES, d))

    # try/finally, not a plain pair of calls: an unconverged solve costs
    # just as much HYSYS time as a converged one and the raise below must
    # not lose it, or the sampler would be charged for it instead.
    _t0 = time.perf_counter()
    try:
        out = run_case(
            OBJECTS,
            rxn8_ea_forward = float(ea_forward),
            rxn8_ea_reverse = float(ea_reverse),
            verbose         = False,   # one timing line per point would drown
            **dict(zip(DESIGN_NAMES, (float(v) for v in d))),
            **FIXED_KWARGS,            # the sampler's own progress table
        )
    finally:
        MODEL_TIME  += time.perf_counter() - _t0
        MODEL_CALLS += 1

        if SEED_END_T is None and MODEL_CALLS >= SEED_CALLS:
            SEED_END_T      = time.perf_counter()
            SEED_MODEL_TIME = MODEL_TIME

    if not out["converged"]:
        raise RuntimeError(
            f"unconverged at {where}: "
            f"{out['convergence_detail'] or out['write_error']}")

    values = [out[key] for key in OUTPUT_KEYS]
    if any(v is None for v in values):
        raise RuntimeError(
            f"converged but an output was missing at {where}: {values}")

    return np.array(values, dtype=float)


# ======================================================================
# 4. MODEL, ESTIMATOR, SAMPLER
# ======================================================================
uncertainty = mn.WeightedScenarios(
    theta_samples = THETA_SAMPLES,
    weights       = THETA_WEIGHTS,
    normalise     = True,
)

model = mn.BlackBoxModel(
    simulator   = run_design,
    uncertainty = uncertainty,
    constraints = CONSTRAINTS,
    name        = "HYSYS methanol synthesis (Rxn-8 kinetics as theta)",
    on_failure  = "infeasible",
)

# The model's own estimator: every merit/probability evaluation calls
# run_design in this process. ParallelEstimator is deliberately not
# imported -- an unused import here would invite someone to switch it on
# and quietly invalidate the timing this script exists to produce.
estimator = model.make_estimator(
    uncertainty    = uncertainty,
    N_theta        = 1,          # ignored by WeightedScenarios
    feas_criterion = FEAS_CRITERION,
)

design_space = mn.DesignSpace(bounds=DESIGN_BOUNDS, names=DESIGN_NAMES)

sampler = mn.MultiNestSampler(
    estimator    = estimator,
    design_space = design_space,
    N_L          = N_L,
    alpha_star   = ALPHA_STAR,
)

# ======================================================================
# 5. RUN
# ======================================================================
monitor = None
if MONITOR_EVERY:
    from live_monitor import LiveMonitor
    monitor = LiveMonitor(out_png=OUT_SNAPSHOT, design_names=DESIGN_NAMES,
                          bounds=DESIGN_BOUNDS,
                          every_seconds=MONITOR_EVERY)
    print(f"live snapshot      : {OUT_SNAPSHOT}  "
          f"(every {MONITOR_EVERY:.0f} s)")

# Timed around sampler.run alone: the connection above and the CSV write
# below are setup, not method. This is the number the case study reports.
# MODEL_TIME is read at the same two instants so the split below covers
# exactly this interval and nothing else.
t_start          = time.perf_counter()
model_time_start = MODEL_TIME

try:
    result = sampler.run(
        seed           = RUN_SEED,
        log_every      = LOG_EVERY,
        log_heartbeat  = LOG_HEARTBEAT,
        frame_callback = monitor,
    )
except Exception:
    traceback.print_exc()
    print("\nRun aborted. Check solver.CanSolve is True and Rxn-8 Ea reads "
          f"{EA_FORWARD_NOMINAL} / {EA_REVERSE_NOMINAL}")
    raise
finally:
    # Written even on an abort: hours of frames are worth keeping when a
    # run dies three quarters of the way through.
    if monitor is not None:
        monitor.close()

WALL_CLOCK  = time.perf_counter() - t_start
SOLVE_TIME  = MODEL_TIME - model_time_start   # inside HYSYS
SAMPLER_TIME = WALL_CLOCK - SOLVE_TIME        # everything else in run()

# Phase split. None when the seed never completed -- a run killed partway
# through it -- rather than a zero that would read as "the seed was free".
if SEED_END_T is None:
    SEED_WALL = SEED_SOLVE = None
    REPL_WALL = REPL_SOLVE = None
    REPL_CALLS = 0
else:
    SEED_WALL  = SEED_END_T - t_start
    SEED_SOLVE = SEED_MODEL_TIME - model_time_start
    REPL_WALL  = WALL_CLOCK - SEED_WALL
    REPL_SOLVE = SOLVE_TIME - SEED_SOLVE
    REPL_CALLS = MODEL_CALLS - SEED_CALLS


# ======================================================================
# 6. RESULTS
# ======================================================================
live_points = result.live_points
live_probs  = result.live_probs
live_merit  = result.live_merit
live_modes  = result.live_mode_ids


np.savez(
    OUT_NPZ,
    design_names  = np.array(DESIGN_NAMES),
    theta_samples = THETA_SAMPLES,
    theta_weights = THETA_WEIGHTS,
    constraints   = np.array(CONSTRAINTS, dtype=float),

    live_points   = live_points,        # the certified design space
    live_probs    = live_probs,
    live_merit    = live_merit,
    live_modes    = live_modes,

    dead_points   = result.dead_points,      # evicted
    dead_probs    = result.dead_probs,
    dead_merit    = result.dead_merit,

    # Rejected candidates never entered the population, but each cost a
    # HYSYS solve and they are the only record of where infeasible starts.
    rejected_points = result.rejected_points,
    rejected_probs  = result.rejected_probs,
    rejected_merit  = result.rejected_merit,

    # Recorded alongside the points so the reported timing can be traced
    # back to the run that produced them.
    init_time_s      = np.array(INIT_TIME),
    wall_clock_s     = np.array(WALL_CLOCK),
    solve_time_s     = np.array(SOLVE_TIME),
    sampler_time_s   = np.array(SAMPLER_TIME),
    model_calls      = np.array(MODEL_CALLS),
    total_model_runs = np.array(result.total_model_runs),

    # Seed vs replacement loop. NaN when the seed did not finish; np.savez
    # cannot store None, and NaN is the value that will not be mistaken for
    # a measured zero if it reaches a plot.
    seed_calls       = np.array(SEED_CALLS),
    seed_wall_s      = np.array(np.nan if SEED_WALL  is None else SEED_WALL),
    seed_solve_s     = np.array(np.nan if SEED_SOLVE is None else SEED_SOLVE),
    repl_calls       = np.array(REPL_CALLS),
    repl_wall_s      = np.array(np.nan if REPL_WALL  is None else REPL_WALL),
    repl_solve_s     = np.array(np.nan if REPL_SOLVE is None else REPL_SOLVE),
)
print(f"saved              : {OUT_NPZ}")

# Every point the run kept, tagged by role; filter role == "live" for the
# certified space alone. merit is the informative column: under VaR it is
# -(worst violation), so positive inside and negative outside. P is binary
# at K = 1. mode means nothing off the live set, so those get -1.
#
# all_points_and_merits() and feas_probabilities("all") share an order
# (dead, rejected, live), so the columns line up.
all_points, all_merit, all_roles = result.all_points_and_merits()
all_probs = result.feas_probabilities("all")
all_modes = np.concatenate([
    np.full(all_points.shape[0] - live_points.shape[0], -1, dtype=int),
    live_modes,
])

with open(OUT_CSV, "w", encoding="utf-8", newline="") as handle:
    handle.write(",".join(DESIGN_NAMES + ["P", "merit", "mode", "role"]) + "\n")
    for point, prob, merit, mode, role in zip(
            all_points, all_probs, all_merit, all_modes, all_roles):
        coords = ",".join(f"{value:.10g}" for value in point)
        handle.write(f"{coords},{prob:.10g},{merit:.10g},{mode},{role}\n")

print(f"saved              : {OUT_CSV}  ({all_points.shape[0]} rows)")

# ======================================================================
# 7. TIMING SUMMARY
# ======================================================================
# Every number the comparison against a non-parallel reference method needs,
# printed together so none of them can be quoted from a different run.
#
# The split is the point of it. Wall clock alone credits or blames the
# sampler for HYSYS; "sampler" below is what the method costs once the
# flowsheet is charged separately, and it is the term that stays flat while
# the solve cost grows with the model.
runs = result.total_model_runs
print("\n--- serial run summary -------------------------------------")
print(f"initialisation     : {INIT_TIME:8.1f} s   (excluded from the below)")
print(f"wall clock         : {WALL_CLOCK:8.1f} s   "
      f"({WALL_CLOCK / 60.0:.1f} min)")
print(f"  in HYSYS         : {SOLVE_TIME:8.1f} s   "
      f"({100.0 * SOLVE_TIME / WALL_CLOCK:.1f} %)")
print(f"  in sampler       : {SAMPLER_TIME:8.1f} s   "
      f"({100.0 * SAMPLER_TIME / WALL_CLOCK:.1f} %)")
print(f"model evaluations  : {runs:8d}")

if MODEL_CALLS:
    print(f"seconds per solve  : {SOLVE_TIME / MODEL_CALLS:8.2f} s   "
          f"(mean over {MODEL_CALLS} calls)")

# The seed is the distributable half; the replacement loop is not. Quoting
# a speed-up against the total would attribute to the method what only the
# seed can deliver.
if SEED_WALL is None:
    print("  seed             :      n/a       (seed did not complete)")
else:
    print(f"  seed             : {SEED_WALL:8.1f} s   "
          f"({SEED_CALLS} solves, {SEED_SOLVE:.1f} s in HYSYS)")
    print(f"  replacement      : {REPL_WALL:8.1f} s   "
          f"({REPL_CALLS} solves, {REPL_SOLVE:.1f} s in HYSYS)")

# run_design counts every call it made; total_model_runs is what the sampler
# believes it asked for. They agree unless something outside sampler.run
# called the model, so a mismatch means the split above covers the wrong
# interval and is worth knowing about before the numbers reach a table.
if MODEL_CALLS != runs:
    print(f"NOTE: run_design was called {MODEL_CALLS} times but the sampler "
          f"reports {runs} model runs.")

print(f"terminated         : {result.termination_reason}  "
      f"({result.n_uncertified_live} live points uncertified)")
print("------------------------------------------------------------")
