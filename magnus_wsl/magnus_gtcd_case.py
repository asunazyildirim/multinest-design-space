"""
The GTCD 4-D case study, ported to MAGNUS / NSFeas.

Companion to ``gtcd_case.py`` on the Windows side. Same equations, same
bounds, same constraints, same reliability target, same live-point count --
so the two samplers can be put side by side on an identical problem.

    GTCD -- Gas Transmission Compressor Design, D = 4.
    Bouhlel, Bartoli, Otsmane & Morlier (2018); used for design-space
    characterisation by Marich et al. (2025).

      min  f(x) = p*8.61e5 * x1^0.5 * x2 * x3^(-2/3) * x4^(-0.5)
                  + 3.69e4 * x3 + 7.72e8 * x1^(-1) * x2^0.219
                  - 765.43e6 * x1^(-1)
      s.t. g1(x) = x4*x2^(-2) + x2^(-2) - 1 <= 0
           f(x) <= 1.1e7
           20 <= x1 <= 50,  1 <= x2 <= 10,  20 <= x3 <= 50,  0.1 <= x4 <= 60

The design space is {x : g1 <= 0 AND f <= F_MAX} -- 7.465% of the box,
measured on 200M uniform samples and cross-checked against 67M Sobol.

WHY THIS CASE IS DIRECTLY COMPARABLE AND THE LADDER IS NOT
----------------------------------------------------------
``magnus_multidim_examples.py`` carries a caveat that blocks any head-to-head
number: NSFEAS takes ONE fixed scenario set for a whole run, while the Python
sampler's ``FeasibilityEstimator`` redraws theta on every design-point
evaluation. Same K, different statistics -- the fixed set flatters the
acceptance rate and lets the sampler overfit those K scenarios.

That caveat does not apply here, because the default run is the NOMINAL
design space: theta = 1 with probability 1, ONE scenario. A single fixed
scenario and a single redrawn scenario are the same scenario. Both codes
evaluate exactly the same deterministic criterion at every design point, so
evaluation counts, acceptance rates and the shape of the answer ARE
comparable. This is the cleanest comparison available between the two
implementations, which is most of why this case is worth porting.

Passing ``--sigma`` turns the probabilistic variant back on and the caveat
comes back with it; the npz records which mode was used.

MAPPING ONTO NSFEAS
-------------------
    N_L                  -> options.NUMLIVE
    alpha* = 1           -> options.FEASTHRES = 0.0
    VaR                  -> options.FEASCRIT  = options.VAR
    g1 <= 0              -> the expression g1 itself
    f  <= F_MAX          -> f - F_MAX
    theta = 1            -> set_parameter([p], [1.0])
    box bounds           -> set_control(d, LB, UB)

Every term goes in as a plain DAG expression -- pymc supports the fractional
powers this model needs (``x**0.5``, ``x**(-2/3)``, ``x**0.219``) natively,
so unlike family B of the ladder there is no ``FFCustom`` black box here and
NSFEAS sees the whole structure.

A NOTE ON SCALE
---------------
The two constraints live on wildly different scales: g1 is O(1) while
f - F_MAX runs to about 1.6e7. NSFEAS orders live points by the worst
violation, so the objective cap dominates the criterion until it is met.
That is a property of the problem, not of either sampler, and it is
identical on the Python side -- which is exactly why it does not disturb the
comparison.

OUTPUT
------
Each run writes ``gtcd_points_magnus_s<seed>.npz`` using the SAME key names
``gtcd_case.py`` writes, so the Windows side can draw the identical trellis
chart for this run and put the two figures next to each other:

    python gtcd_case.py --replot --npz <path to the npz>

``live_merit`` is stored as ``-live_crit``: NSFEAS reports VaR[G] directly
(feasible <= 0) while the Python side stores the sampler's higher-is-better
merit and negates it on read. Storing both keeps the file readable by either
convention.

Usage
-----
MAGNUS is a Linux C++ library, so this half runs under Linux (or WSL) while
the sampler it is compared against runs on the Windows side. ``pymc`` and
``magnus`` are compiled extensions: they load only under the Python minor
version they were built for, and a system upgrade that moves ``python3``
forward will break them with "module was compiled for Python 3.x". Activate
the interpreter they were built against before running anything here, and
make sure MAGNUS's own PYTHONPATH and LD_LIBRARY_PATH are exported (the
installer's shell profile normally does this):

    source <path to that environment>/bin/activate
    python --version
    python magnus_gtcd_case.py

Write straight into the other side's output directory and the comparison
tables pick the run up with no copying:

    python magnus_gtcd_case.py --out <path to the gtcd_output directory>

then, on the Windows side:

    python gtcd_case.py --report

Other forms:

    python magnus_gtcd_case.py --selfcheck      # verify the DAG transcription
    python magnus_gtcd_case.py --N_L 1500 --seed 11
    python magnus_gtcd_case.py --sigma 0.05 --n-theta 100

``run_magnus_gtcd.sh`` wraps the same thing and exports the paths itself, for
a shell where the profile has not been sourced.
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import pymc
    from magnus import NSFeas
except ImportError as exc:                                  # pragma: no cover
    # The two ways this fails are a missing PYTHONPATH and the wrong Python.
    # Both produce an ImportError whose own text explains neither, so say it
    # here rather than let the next person rediscover it.
    raise SystemExit(
        f"\n  cannot import the MAGNUS interface: {exc}\n\n"
        f"  running under Python {sys.version.split()[0]}.\n\n"
        f"  pymc and magnus are compiled extensions and load only under the\n"
        f"  Python minor version they were built for. Activate that\n"
        f"  interpreter, and check MAGNUS's PYTHONPATH is exported:\n\n"
        f"      source <that environment>/bin/activate\n"
        f"      python {os.path.basename(__file__)}\n"
    ) from None


# ----------------------------------------------------------------------
# 1. THE PROBLEM  -- must stay identical to gtcd_case.py
# ----------------------------------------------------------------------
F_MAX = 1.1e7

LB = [20.0, 1.0, 20.0, 0.1]
UB = [50.0, 10.0, 50.0, 60.0]
NAMES = ["x1", "x2", "x3", "x4"]

# A design known to be feasible. Used by --selfcheck only.
NOMINAL = np.array([50.0, 8.0, 24.5, 40.0])

# alpha* = 1, matching gtcd_case.py on the Windows side. This is a NOMINAL
# characterisation -- theta = 1 with probability 1 -- so requiring feasibility
# with probability 1 is what the problem asks, and FEASTHRES becomes 1 - 1 = 0.
# At K = 1 that is the same computation as any other threshold (a single
# scenario IS every quantile of itself), so this changes the label and not the
# answer; but the label is printed on the figures and quoted in the paper, and
# 0.95 there would suggest a 5 % tail that a one-scenario problem cannot have.
ALPHA_STAR = 1.0
N_LIVE_DEFAULT = 5000

# NSFEAS knobs, carried over unchanged from magnus_nsfeas_examples.py and
# magnus_multidim_examples.py so this case is comparable with those too.
COMMON_NS = dict(DISPLEVEL=1, NUMPROP=8, MAXITER=0,
                 ELLCONF=0.99, ELLMAG=0.30, ELLRED=0.20)


def build_gtcd(d, p):
    """
    GTCD as DAG expressions.

    Parameters
    ----------
    d : sequence of 4 FFVar -- the design variables.
    p : FFVar -- theta, which scales the 8.61e5 compressor-cost
        coefficient (an uncertain unit cost).

    Returns
    -------
    (g1, f) : the constraint expression and the objective expression.
    """
    x1, x2, x3, x4 = d[0], d[1], d[2], d[3]

    g1 = x4 * x2 ** -2 + x2 ** -2 - 1.0
    f = (p * 8.61e5 * x1 ** 0.5 * x2 * x3 ** (-2.0 / 3.0) * x4 ** -0.5
         + 3.69e4 * x3
         + 7.72e8 * x2 ** 0.219 * x1 ** -1
         - 765.43e6 * x1 ** -1)
    return g1, f


def gtcd_numpy(x, theta=1.0):
    """
    The same model in plain numpy, for --selfcheck.

    Written out a second time on purpose: the DAG above is a transcription,
    and a transcription is exactly the kind of thing that acquires a typo
    without failing. This is what it is checked against.
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    g1 = x4 * x2 ** -2.0 + x2 ** -2.0 - 1.0
    f = (theta * 8.61e5 * x1 ** 0.5 * x2 * x3 ** (-2.0 / 3.0) * x4 ** -0.5
         + 3.69e4 * x3
         + 7.72e8 * x2 ** 0.219 / x1
         - 765.43e6 / x1)
    return g1, f


# ----------------------------------------------------------------------
# 2. SELF-CHECK
# ----------------------------------------------------------------------
def selfcheck(n_mc=200_000, n_eval=500, seed=0):
    """
    Verify the DAG against the closed form, and both against the paper.

    Three failures are possible when porting a model to a DAG and all three
    are silent: a mistyped coefficient, an exponent that binds differently
    under operator overloading than it does in numpy, and a constraint
    written with the wrong sign. The first two show up as a mismatch
    between ``build_gtcd`` and ``gtcd_numpy``; the third as a feasible
    fraction that is no longer ~7%.

    Measured values, theta = 1 (200M uniform samples; 67M scrambled Sobol
    agrees to four significant figures):

        g1 <= 0 alone .................... 52.333%   +-0.007  (95% CI)
        g1 <= 0 and f <= 1.1e7 ...........  7.465%   +-0.004  (95% CI)

    This check runs 200k samples, where the 95% interval on the second
    number is about +-0.12 percentage points. So 7.3%-7.6% is a pass --
    the tolerance below is set to that, not tighter -- and a difference of
    that size between two machines is sampling noise, not a port bug. A
    mistyped coefficient does not land inside the window.
    """
    rng = np.random.default_rng(seed)

    DAG = pymc.FFGraph()
    DAG.options.MAXTHREAD = 1
    d = DAG.add_vars(4, "d")
    p = DAG.add_var("p")
    g1_expr, f_expr = build_gtcd(d, p)
    dep, var = [g1_expr, f_expr], list(d) + [p]

    X = rng.uniform(LB, UB, size=(n_eval, 4))
    T = rng.normal(1.0, 0.05, n_eval)
    worst_g, worst_f = 0.0, 0.0
    for x, t in zip(X, T):
        got = DAG.eval(dep, var, list(x) + [float(t)])
        want = gtcd_numpy(x, t)
        worst_g = max(worst_g, abs(got[0] - want[0]))
        worst_f = max(worst_f, abs(got[1] - want[1]) / max(abs(want[1]), 1.0))

    print(f"\n  self-check")
    print(f"    DAG vs numpy, {n_eval} random points (theta ~ N(1, 0.05)):")
    print(f"      g1  max abs err     : {worst_g:.3e}")
    print(f"      f   max rel err     : {worst_f:.3e}")

    Y = rng.uniform(LB, UB, size=(n_mc, 4))
    g1 = Y[:, 3] * Y[:, 1] ** -2.0 + Y[:, 1] ** -2.0 - 1.0
    f = (8.61e5 * Y[:, 0] ** 0.5 * Y[:, 1] * Y[:, 2] ** (-2.0 / 3.0)
         * Y[:, 3] ** -0.5 + 3.69e4 * Y[:, 2]
         + 7.72e8 * Y[:, 1] ** 0.219 / Y[:, 0] - 765.43e6 / Y[:, 0])
    frac_g1 = float(np.mean(g1 <= 0.0))
    frac_both = float(np.mean((g1 <= 0.0) & (f <= F_MAX)))
    print(f"    feasible fraction, {n_mc:,} uniform samples (theta = 1):")
    print(f"      g1 <= 0 alone       : {frac_g1:7.3%}   (measured 52.333%)")
    print(f"      g1 <= 0, f <= {F_MAX:.1e}: {frac_both:7.3%}   "
          f"(measured 7.465%, +-0.12 pp at this n)")

    g_nom, f_nom = gtcd_numpy(NOMINAL)
    ok_nom = g_nom <= 0.0 and f_nom <= F_MAX
    print(f"    nominal {NOMINAL.tolist()}")
    print(f"      g1 = {g_nom:.4f}, f = {f_nom:.4e}  ->  feasible = {ok_nom}")

    ok = (worst_g < 1e-9 and worst_f < 1e-12 and ok_nom
          and abs(frac_both - 0.07465) < 0.005)
    print(f"    VERDICT             : {'PASS' if ok else 'FAIL'}\n")
    return ok


# ----------------------------------------------------------------------
# 3. REPORTING
# ----------------------------------------------------------------------
RELIABILITY_EDGES = [0.95, 0.70, 0.50, 0.25]


def reliability_table(prob, edges=RELIABILITY_EDGES):
    """Bin the per-point feasibility probability P into reliability ranges."""
    rows, remaining, hi = [], np.asarray(prob).ravel(), None
    for lo in edges:
        mask = remaining >= lo
        label = f"{lo:.2f} <= P" if hi is None else f"{lo:.2f} <= P < {hi:.2f}"
        rows.append((label, int(mask.sum())))
        remaining, hi = remaining[~mask], lo
    rows.append((f"P < {edges[-1]:.2f}", int(remaining.size)))
    rows.append(("Total", int(np.asarray(prob).size)))
    return rows


def report(NS, n_live_opt, n_theta, sigma, alpha, elapsed):
    """Print what NSFEAS knows about the run and return the arrays."""
    live_pts, live_crit, live_prob, _ = NS.live_points
    dead_pts, dead_crit, dead_prob, _ = NS.dead_points
    disc_pts, disc_crit, disc_prob, _ = NS.discard_points

    n_live, n_dead, n_disc = len(live_pts), len(dead_pts), len(disc_pts)
    n_cand = n_dead + n_disc
    runs = int(NS.stats.numfct)
    eff = n_dead / n_cand if n_cand else float("nan")

    allprob = np.concatenate([np.asarray(live_prob).ravel(),
                              np.asarray(dead_prob).ravel(),
                              np.asarray(disc_prob).ravel()])

    print()
    print("-" * 70)
    print("GTCD 4-D on NSFEAS")
    print("-" * 70)
    for label, count in reliability_table(allprob):
        print(f"  {label:<22}{count:>14d}")
    print()
    print(f"    Criterion         : VaR  (alpha = {alpha:g}, feasible <= 0)")
    print(f"    Uncertainty       : "
          + ("theta = 1 (nominal DS), K = 1" if sigma is None
             else f"theta ~ N(1, {sigma:g}), K = {n_theta}"))
    print(f"    Live points       : {n_live}")
    print(f"    Dead points       : {n_dead}")
    print(f"    Discarded points  : {n_disc}")
    print(f"    Candidate evals   : {n_cand}")
    print(f"    Model runs        : {runs:,}  (K = {n_theta} per eval)")
    print(f"    Failed evals      : {int(NS.stats.numerr)}")
    print(f"    Sampling eff.     : {eff:.3f}  (dead / candidates)")
    if n_live:
        print(f"    Worst live value  : {float(np.max(live_crit)):.4e}")
    print(f"    Wall time         : {elapsed:.1f}s")
    print()

    return dict(
        live_points=np.asarray(live_pts, dtype=float),
        live_crit=np.asarray(live_crit, dtype=float).ravel(),
        live_probs=np.asarray(live_prob, dtype=float).ravel(),
        dead_points=np.asarray(dead_pts, dtype=float),
        dead_crit=np.asarray(dead_crit, dtype=float).ravel(),
        dead_probs=np.asarray(dead_prob, dtype=float).ravel(),
        disc_points=np.asarray(disc_pts, dtype=float),
        disc_crit=np.asarray(disc_crit, dtype=float).ravel(),
        disc_probs=np.asarray(disc_prob, dtype=float).ravel(),
        n_live=n_live, n_dead=n_dead, n_disc=n_disc, n_candidates=n_cand,
        model_runs=runs, efficiency=eff, n_failed=int(NS.stats.numerr),
        wall_s=round(elapsed, 2),
    )


# ----------------------------------------------------------------------
# 4. RUNNER
# ----------------------------------------------------------------------
def run(n_live=N_LIVE_DEFAULT, sigma=None, n_theta=100,
        alpha_star=ALPHA_STAR, seed=11, out_dir=None):
    """
    Build the DAG, run NSFEAS once, write the points, return the summary.

    ``sigma=None`` is the nominal design space -- theta = 1 with
    probability 1, a single scenario. That is the default because it is
    the mode in which this run is directly comparable with the Python
    sampler; see the module docstring.
    """
    K = 1 if sigma is None else n_theta
    alpha = 1.0 - alpha_star

    DAG = pymc.FFGraph()
    DAG.options.MAXTHREAD = 1
    d = DAG.add_vars(4, "d")
    p = DAG.add_var("p")
    g1, f = build_gtcd(d, p)

    NS = NSFeas()
    for key, value in COMMON_NS.items():
        setattr(NS.options, key, value)
    NS.options.NUMLIVE = n_live
    NS.options.FEASCRIT = NS.options.VAR
    NS.options.FEASTHRES = alpha

    NS.set_dag(DAG)
    # Both expressions are "<= 0 when feasible": g1 already is, and the
    # objective cap becomes f - F_MAX. No lower bounds -- neither output
    # has one, so nothing is added for them.
    NS.set_constraint([g1, f - F_MAX])
    NS.set_control(d, LB, UB)

    if sigma is None:
        # One scenario at theta = 1. Same call shape as the nominal branch
        # of magnus_nsfeas_examples.run_example.
        NS.set_parameter([p], [1.0])
    else:
        rng = np.random.default_rng(seed)
        NS.set_parameter([p], [[v] for v in rng.normal(1.0, sigma, K)])

    print()
    print("=" * 70)
    print("GTCD 4D - gas transmission compressor (D=4)   on MAGNUS / NSFeas")
    print("  g1 = x4/x2^2 + 1/x2^2 - 1 <= 0,  f <= 1.1e7  |  ~7% feasible")
    print(f"  N_L={n_live}  K={K}  alpha*={alpha_star}  criterion=VaR  "
          f"seed={seed}")
    print("  " + ("nominal: theta = 1 with probability 1"
                  if sigma is None else f"theta ~ N(1, {sigma:g})"))
    print("=" * 70)

    t0 = time.time()
    NS.setup()
    NS.sample()
    elapsed = time.time() - t0

    summary = report(NS, n_live, K, sigma, alpha, elapsed)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = out_dir or os.path.join(here, "magnus_gtcd_output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"gtcd_points_magnus_s{seed}.npz")

    # Key names match gtcd_case.py's writer so the Windows-side reader and
    # trellis code work on this file unchanged. NSFEAS's "discarded" is
    # what that side calls "rejected"; merit is -criterion for VaR.
    np.savez_compressed(
        path,
        live_points=summary["live_points"],
        live_merit=-summary["live_crit"],
        live_probs=summary["live_probs"],
        dead_points=summary["dead_points"],
        dead_merit=-summary["dead_crit"],
        dead_probs=summary["dead_probs"],
        rejected_points=summary["disc_points"],
        rejected_merit=-summary["disc_crit"],
        rejected_probs=summary["disc_probs"],
        criterion="VaR",
        alpha_star=float(alpha_star),
        sigma=float("nan") if sigma is None else float(sigma),
        n_theta=int(K),
        seed=int(seed),
        wall_s=float(elapsed),
        # Provenance, so a file can never be mistaken for a Python-side run.
        sampler="magnus_nsfeas",
        scenario_mode="nominal" if sigma is None else "fixed",
        live_crit=summary["live_crit"],
        n_candidates=int(summary["n_candidates"]),
        model_runs=int(summary["model_runs"]),
        efficiency=float(summary["efficiency"]),
        n_failed=int(summary["n_failed"]),
    )
    print(f"    written           : {path}")
    print()
    print("    To draw the same trellis chart as the Python run:")
    print(f"        python gtcd_case.py --replot --npz {path}")
    summary["path"] = path
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The GTCD 4-D case study on MAGNUS / NSFeas.")
    parser.add_argument("--N_L", dest="n_live", type=int,
                        default=N_LIVE_DEFAULT,
                        help=f"live points (default {N_LIVE_DEFAULT})")
    parser.add_argument("--sigma", type=float, default=None,
                        help="switch to the probabilistic variant, "
                             "theta ~ N(1, sigma); omitted means the nominal "
                             "theta = 1")
    parser.add_argument("--n-theta", type=int, default=100,
                        help="scenarios per design point (only with --sigma)")
    parser.add_argument("--alpha-star", type=float, default=ALPHA_STAR)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=None)
    parser.add_argument("--selfcheck", action="store_true",
                        help="verify the DAG against the closed form and exit")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return 0 if selfcheck() else 1

    run(n_live=args.n_live, sigma=args.sigma, n_theta=args.n_theta,
        alpha_star=args.alpha_star, seed=args.seed, out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
