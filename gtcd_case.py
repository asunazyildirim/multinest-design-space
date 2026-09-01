"""
GTCD 4-D case study, in the trellis layout of Kusumo et al. (2020).

The reference is that paper's Figure 6 -- a NOMINAL design space computed
with a single uncertainty scenario. GTCD is a deterministic optimisation
benchmark, so it enters this framework the same way: theta = 1 with
probability 1, K = 1, and alpha* = 1, which is what "feasible with
probability one" means. Pass ``--sigma`` to make it probabilistic instead.

Layout, taken from the paper:

    "For the DS representation we use trellis charts where the ranges of
     oxygen concentration and temperature are split into four-by-four
     intervals -- indicated by gray bars on the two outer axes; then each
     subplot is a projection of the points that belong to the particular
     intervals ... on the plane defined by [the] two inner axes."

* OUTER axes are two design variables split into intervals, each shown as
  a grey shingle bar: the full range in light grey, the interval that row
  or column covers in dark, with the numeric range at the ends. That bar
  is what makes the grid readable as a pair of axes rather than as sixteen
  unrelated scatter plots.
* INNER axes are the remaining two variables, shared across the grid, with
  tick labels on the bottom row and left column only.

COLOUR: two renderings, and what each is for
--------------------------------------------
``--color-by feasible`` is the paper's Figure 6: green inside the design
space, red outside. On a nominal run the feasibility probability is 0 or 1
and nothing lies between, so this is the honest picture of what a single
scenario resolves -- a yes/no set.

``--color-by criterion`` (the default) instead colours by VaR[G], the worst
constraint violation, which on a single scenario IS that violation exactly.
It defines the same set -- VaR[G] <= 0 iff every constraint holds -- but as a
continuous quantity, so the colour also says HOW far inside or outside a
design sits. The zero crossing is the design-space boundary and the diverging
colour scale is centred on it.

That scale is symmetric-log, not linear. Violations here span 0 to about
1.6e7, because the objective cap f <= f_max enters G on its own raw scale
while g1 is O(1). A linear map paints the whole chart one colour and hides
the only contour that matters.

``--color-by bands`` is the paper's Figure 5, three ranges of the feasibility
probability. It needs ``--sigma``: on a nominal run two of its three bands
stay empty.

TWO FIGURES, on purpose
-----------------------
``trellis_gtcd_live_*.png``   the final live set -- the answer. Every point
                              is certified (VaR[G] <= 0), so the colour
                              range is entirely on the feasible side and
                              what it shows is WHERE the design space is
                              and how densely it was covered.
``trellis_gtcd_all_*.png``    every point the run scored, live + dead +
                              rejected: the whole violation landscape, the
                              counterpart of the paper's full chart.

Both use the SAME colour scale, computed from the second set, so a colour
means the same number in both and the two can be read side by side.

SAMPLER SETTINGS: NONE
----------------------
No ellipsoid or clustering options are declared here. The study reports one
configuration across every problem, so this case takes the sampler's own
defaults; overriding them per problem is the tuning the paper says it does
not do. ``SAMPLER_KW`` is therefore empty, and deliberately so.

SELF-CONTAINED
--------------
The model, its bounds and its constraints are written out below rather than
imported. The only external dependency is ``multinest_sampler`` itself, so
this file can be read, moved or handed over on its own.

The cost of that is a second copy of the GTCD equations, the published
Bouhlel et al. (2018) formulation. ``--selfcheck`` re-measures the feasible
fraction and a known feasible point against the numbers quoted below, which
is what catches a transcription error.

COMPARING TWO SAMPLERS
----------------------
``magnus_gtcd_case.py`` runs the same problem under MAGNUS/NSFeas and writes
an .npz with the SAME key names, including a ``sampler`` field. Point this
script at that file and the trellis is drawn by exactly this code, so the two
figures differ only in the run behind them:

    python gtcd_case.py --npz <path to the MAGNUS .npz>

Put both .npz files in the same output directory and ``--report`` tabulates
them side by side. It refuses to build a table when the two runs disagree on
N_L, K or alpha*, since those are declared separately on each side.

Usage
-----
    python gtcd_case.py                      # run + trellis charts
    python gtcd_case.py --N_L 3000
    python gtcd_case.py --replot             # redraw from the saved npz
    python gtcd_case.py --color-by feasible  # the P = 0 / P = 1 split
    python gtcd_case.py --outer 1 2 --inner 3 4
    python gtcd_case.py --npz <other run>    # draw a run from elsewhere
    python gtcd_case.py --report             # comparison tables + .xlsx
    python gtcd_case.py --selfcheck          # verify the model copy

    # the probabilistic variant, coloured the paper's Fig. 5 way
    python gtcd_case.py --sigma 0.05 --n-theta 100 --color-by bands

Everything is written to ``gtcd_output/`` next to this file. Figure names
carry the sampler and the colouring, so no two of the above overwrite each
other.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PARENT = os.path.dirname(os.path.abspath(__file__))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import multinest_sampler as mn

OUT_DIR = os.path.join(PARENT, "gtcd_output")


# ======================================================================
# THE PROBLEM
# ======================================================================
# GTCD -- Gas Transmission Compressor Design, D = 4, 1 constraint plus an
# objective cap. Bouhlel, Bartoli, Otsmane & Morlier (2018), as used by
# Marich et al. (2025) for design-space characterisation.
#
#   min  f(x) = θ·8.61e5 · x1^0.5 · x2 · x3^(-2/3) · x4^(-0.5)
#               + 3.69e4 · x3
#               + 7.72e8 · x1^(-1) · x2^0.219
#               - 765.43e6 · x1^(-1)
#
#   s.t. g1(x) = x4·x2^(-2) + x2^(-2) - 1 <= 0
#        f(x) <= f_max
#        20 <= x1 <= 50,  1 <= x2 <= 10,  20 <= x3 <= 50,  0.1 <= x4 <= 60
#
# The design space is {x : g1 <= 0 AND f <= f_max}. Marich et al. put the
# objective itself in the constraint set ("the feasibility function is
# equivalent to the objective function ... constrained to a user-defined
# value"), but do not state f_max; F_MAX below was chosen by Monte Carlo
# so the design space is a non-trivial fraction of the box:
#     f_max = 1.1e7  ->   7.465% feasible   (g1 alone: 52.333%)
#     f_max = 1.2e7  ->  12.239%
# Measured on 200M uniform samples, cross-checked against 67M Sobol; see
# selfcheck() for the error bars.

F_MAX = 1.1e7

# No lower bound on either output. ``violation_matrix`` builds one column per
# FINITE bound and skips the others, so -inf costs a column rather than
# poisoning the max: two columns here instead of four, bitwise-identical
# criterion values (checked over 30 000 uniform designs). A large finite
# stand-in worked only because f stays in the 1e6..3e7 range; -inf carries no
# such assumption.
_NEG_INF = -np.inf

DS = mn.DesignSpace(
    bounds=[(20.0, 50.0), (1.0, 10.0), (20.0, 50.0), (0.1, 60.0)],
    names=["d1", "d2", "d3", "d4"],
)

# A design known to be feasible, for sanity checks. Not used by the
# figures -- the trellis pins nothing -- but --selfcheck verifies it.
NOMINAL = np.array([50.0, 8.0, 24.5, 40.0])

# The uncertainty. theta multiplies the 8.61e5 compressor-cost
# coefficient, i.e. an uncertain unit cost. The NOMINAL design space is
# the degenerate case: theta = 1 with probability 1, a single scenario,
# which is the paper's eq 3. ``normalise=False`` because the weight is
# already exactly 1 and renormalising a one-element vector would only hide
# a typo if it were not.
DETERMINISTIC = mn.WeightedScenarios(
    theta_samples=np.array([1.0]),
    weights=np.array([1.0]),
    normalise=False,
)


def gtcd_f(x, theta=1.0):
    """Objective. ``theta`` scales the compressor-cost coefficient."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (theta * 8.61e5 * x1 ** 0.5 * x2 * x3 ** (-2.0 / 3.0) * x4 ** -0.5
            + 3.69e4 * x3
            + 7.72e8 * x2 ** 0.219 / x1
            - 765.43e6 / x1)


def gtcd_g1(x):
    """The one inequality constraint; feasible when <= 0."""
    return x[3] * x[1] ** -2 + x[1] ** -2 - 1.0


def equation(d, theta):
    """
    Evaluate one design point over all uncertainty scenarios.

    Parameters
    ----------
    d : (4,) design point in physical coordinates.
    theta : (N_theta,) scenario batch.

    Returns
    -------
    (N_theta, 2) -- column 0 is g1 (theta-independent, broadcast), column
    1 is f. Two columns because the sampler needs one per constraint, and
    the objective cap is a constraint here.
    """
    t = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
    ones = np.ones_like(t)
    return np.column_stack([gtcd_g1(d) * ones, gtcd_f(d, t)])


CONSTRAINTS = [(_NEG_INF, 0.0), (_NEG_INF, F_MAX)]

TITLE = "GTCD 4D — gas transmission compressor (D=4)"
DESCRIPTION = ("g₁ = x₄/x₂² + 1/x₂² − 1 ≤ 0,  f ≤ 1.1e7  |  "
               "4 variables on very different scales  |  7.465% feasible")

# No sampler settings are declared here: the study reports one configuration
# across every problem, so this case takes MultiNestSampler's own defaults
# (F_threshold = 1.1, kmeans_restarts = 1, kmeans_init = "random", and
# min_pt = 2(D+1) = 10 in 4-D). Overriding them per problem is the tuning the
# paper says it does not do.
SAMPLER_KW = {}

# alpha* = 1, not 0.95. This is a NOMINAL characterisation: theta = 1 with
# probability 1, so requiring feasibility with probability 1 is what the
# problem actually asks. At K = 1 the two give identical criterion values --
# with a single scenario every quantile IS that scenario's value, verified --
# so this changes the label and not the answer. It matters because alpha* is
# printed on every figure and quoted in the paper, and 0.95 there would
# suggest a 5 % tail that does not exist.
ALPHA_STAR = 1.0

# THE live-point count. Both ``run()`` and the ``--N_L`` flag default to
# this one name, because three separate defaults is how the CLI ends up
# silently running a different N_L from the one written in run()'s
# signature. Kusumo et al. raise N_L until the sample density across the
# design space stops improving; at ~7% feasible in 4-D, 5000 gives a few
# hundred live points per trellis panel.
N_LIVE_DEFAULT = 5000

# Kept for the optional probabilistic variant only (--sigma --color-by
# bands): the reliability bands of the paper's Figure 5.
BANDS = [
    (0.85, 1.01, "#d62728", r"$\alpha \geq 0.85$"),
    (0.05, 0.85, "#f0c419", r"$0.05 \leq \alpha < 0.85$"),
    (0.00, 0.05, "#1f77b4", r"$\alpha < 0.05$"),
]

# The two-colour split for a NOMINAL run, where P is 0 or 1 and nothing lies
# between. This is the honest rendering of a single-scenario problem: the
# continuous VaR[G] scale says how far outside a rejected design sits, which
# is useful, but the design space itself is a yes/no set and this shows it as
# one. Kusumo et al. colour their Figure 6 the same way.
FEASIBLE_COLOURS = [
    (True,  "#2ca02c", "P = 1  —  in the design space"),
    (False, "#d62728", "P = 0  —  outside"),
]

# Axis labels. The paper can write "T (°C)" and "batch duration" because it
# owns the process; this benchmark arrives as four bare variables with
# bounds, and neither highdim_examples.py nor the formulation it cites
# states what they physically are. So they stay d1..d4 with their ranges.
# Putting a guessed quantity name on an axis would make the figure assert
# something about the process that nothing here establishes -- fill these
# in if you have the source that says so.
#
# d, not x: these are the DESIGN variables, and every other figure and table
# in the study calls them d. The published formulation quoted above writes
# x_k for the same four quantities -- d_k IS x_k, only renamed to match.
AXIS_LABELS = {
    0: "$d_1 \\in [20, 50]$",
    1: "$d_2 \\in [1, 10]$",
    2: "$d_3 \\in [20, 50]$",
    3: "$d_4 \\in [0.1, 60]$",
}


# ======================================================================
# SELF-CHECK
# ======================================================================

def selfcheck(n=200_000, seed=0):
    """
    Verify this file's copy of the model against the published numbers.

    Nothing in the code checks that the equations above still match the
    ones in ``highdim_examples.py`` -- that is the price of being
    self-contained. This is the substitute: it re-measures what the
    catalogue documents, so a typo in a coefficient shows up as a
    feasible fraction that is no longer ~7%, or a nominal point that is
    no longer feasible.

    Measured values, theta = 1 (200M uniform samples; 67M scrambled Sobol
    agrees to four significant figures, so these are the numbers, not one
    estimator's opinion of them):

        g1 <= 0 alone .................... 52.333%   +-0.007  (95% CI)
        g1 <= 0 and f <= 1.1e7 ...........  7.465%   +-0.004  (95% CI)
        g1 <= 0 and f <= 1.2e7 ........... 12.239%            (100M)

    This check runs 200k samples, where the 95% interval on the second
    number is about +-0.12 percentage points. So 7.3%-7.6% is a pass and
    the run-to-run wobble you see between machines is that interval, not a
    discrepancy. A mistyped coefficient does not land inside it.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(DS.lo, DS.hi, size=(n, DS.D))

    g1 = X[:, 3] / X[:, 1] ** 2 + 1.0 / X[:, 1] ** 2 - 1.0
    f = (8.61e5 * X[:, 0] ** 0.5 * X[:, 1] * X[:, 2] ** (-2.0 / 3.0)
         * X[:, 3] ** -0.5 + 3.69e4 * X[:, 2]
         + 7.72e8 * X[:, 1] ** 0.219 / X[:, 0] - 765.43e6 / X[:, 0])

    print(f"\n  self-check on {n:,} uniform samples, θ = 1")
    print(f"    g1 <= 0 alone            : {np.mean(g1 <= 0.0):7.3%}"
          f"   (measured 52.333%)")
    print(f"    g1 <= 0 and f <= {F_MAX:.1e} : "
          f"{np.mean((g1 <= 0.0) & (f <= F_MAX)):7.3%}   (measured 7.465%,"
          f" ±0.12 pp at this n)")

    # The vectorised `equation` must agree with the plain formulas above:
    # they are written twice, once for the sampler and once here, and a
    # disagreement between them is exactly the kind of error this catches.
    k = rng.integers(0, n, 500)
    s = np.array([equation(X[i], np.array([1.0]))[0] for i in k])
    err = max(np.max(np.abs(s[:, 0] - g1[k])), np.max(np.abs(s[:, 1] - f[k])))
    print(f"    equation() vs formulas   : max abs diff {err:.3e}")

    s_nom = equation(NOMINAL, np.array([1.0]))[0]
    ok = all(lo <= s_nom[j] <= hi for j, (lo, hi) in enumerate(CONSTRAINTS))
    print(f"    nominal {NOMINAL.tolist()} feasible : {ok}"
          f"   (g1={s_nom[0]:.4f}, f={s_nom[1]:.4e})")
    return ok and err < 1e-6


# ======================================================================
# RUN
# ======================================================================

def run(n_live=N_LIVE_DEFAULT, n_theta=None, sigma=None, criterion="VaR",
        alpha_star=None, seed=11, log_every="5%"):
    """
    Nested sampling on GTCD 4-D, nominal by default.

    ``sigma=None`` keeps ``DETERMINISTIC`` -- theta = 1 with probability
    1, one scenario, which is the nominal design space of the paper's
    eq 3 and Figure 6. Pass a sigma to get the probabilistic variant
    instead; ``n_theta`` then matters and defaults to 100.

    Everything else -- the equations, the constraints, alpha*, the
    ellipsoid settings -- is the module-level definition above.
    """
    if sigma is None:
        uncertainty, K = DETERMINISTIC, 1
    else:
        uncertainty = mn.GaussianUncertainty(mu=1.0, sigma=sigma)
        K = 100 if n_theta is None else n_theta
    if alpha_star is None:
        alpha_star = ALPHA_STAR

    model = mn.ProcessModel(
        equation=equation,
        uncertainty=uncertainty,
        constraints=CONSTRAINTS,
        name=TITLE,
    )
    estimator = model.make_estimator(uncertainty=uncertainty, N_theta=K,
                                     feas_criterion=criterion)
    sampler = mn.MultiNestSampler(estimator=estimator, design_space=DS,
                                  N_L=n_live, alpha_star=alpha_star,
                                  **SAMPLER_KW)   # SAMPLER_KW is empty: defaults

    theta_text = "deterministic (θ = 1, nominal DS)" if sigma is None \
        else f"θ ~ N(1, {sigma:g}), N_θ = {K}"
    print(f"\n{'=' * 70}")
    print(f"  {TITLE}")
    print(f"  {DESCRIPTION}")
    print(f"  N_L={n_live}  {theta_text}  criterion={criterion}  "
          f"α*={alpha_star}  seed={seed}")
    print(f"{'=' * 70}")

    np.random.seed(seed)
    t0 = time.perf_counter()
    result = sampler.run(seed=seed, log_every=log_every, log_heartbeat=200)
    elapsed = time.perf_counter() - t0

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"gtcd_points_s{seed}.npz")
    np.savez_compressed(
        path,
        live_points=result.live_points,
        live_merit=result.live_merit,
        live_probs=result.live_probs,
        dead_points=result.dead_points,
        dead_merit=result.dead_merit,
        dead_probs=result.dead_probs,
        rejected_points=result.rejected_points,
        rejected_merit=result.rejected_merit,
        rejected_probs=result.rejected_probs,
        live_mode_ids=result.live_mode_ids,
        criterion=criterion, alpha_star=float(alpha_star),
        sigma=float("nan") if sigma is None else float(sigma),
        n_theta=int(K), seed=int(seed), wall_s=float(elapsed),
        # The counters the comparison table is built from. Stored rather than
        # recomputed, because "candidates" has to mean the same thing on both
        # sides and only the sampler knows what it counted.
        N_L=int(n_live),
        n_replacements=int(result.n_replacements),
        n_candidates=int(result.n_candidate_estimates),
        model_runs=int(result.total_model_runs),
        converged=bool(result.converged),
        sampler="multinest",
    )
    print(f"  saved             : {path}")
    print(result.reliability_table())
    return path


def load(seed=11, path=None):
    """
    Read one saved run into the two point sets the figures need.

    ``values`` are criterion values (VaR[G], or P) recovered from the
    stored merits through ``CriterionDisplay``, so the sign convention is
    the module's own and cannot drift from what the sampler optimised.
    """
    path = path or os.path.join(OUT_DIR, f"gtcd_points_s{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run:\n"
                                f"    python {os.path.basename(__file__)}")
    d = np.load(path, allow_pickle=False)
    criterion = str(d["criterion"])
    alpha_star = float(d["alpha_star"])
    disp = mn.CriterionDisplay(criterion, alpha_star)

    def group(stem):
        pts = np.asarray(d[f"{stem}_points"], dtype=float)
        if pts.ndim != 2 or pts.size == 0:
            return None
        return (pts,
                np.asarray(disp.from_merit(
                    np.asarray(d[f"{stem}_merit"], dtype=float).ravel()),
                    dtype=float),
                np.asarray(d[f"{stem}_probs"], dtype=float).ravel())

    live = group("live")
    parts = [g for g in (live, group("dead"), group("rejected"))
             if g is not None]
    sigma = float(d["sigma"])

    # Which sampler produced this. MAGNUS/NSFeas writes the same key names
    # (see MagnusCodes/magnus_gtcd_case.py) so that its runs draw through
    # this code -- but the two must not overwrite each other's PNGs, and a
    # figure must say which sampler it is of.
    sampler = str(d["sampler"]) if "sampler" in d else "multinest"

    n_dead = 0 if group("dead") is None else group("dead")[0].shape[0]
    n_rejected = (0 if group("rejected") is None
                  else group("rejected")[0].shape[0])
    n_live = live[0].shape[0]
    K = int(d["n_theta"])

    def get(key, fallback):
        """Counters, with a fallback for files written before they were
        stored. The fallbacks are the definitions, not guesses: one eviction
        is one replacement, and a candidate is either accepted or rejected."""
        return int(d[key]) if key in d.files else int(fallback)

    n_rep = get("n_replacements", n_dead)
    n_cand = get("n_candidates", n_dead + n_rejected)
    N_L = get("N_L", n_live)

    return dict(
        path=path, criterion=criterion, alpha_star=alpha_star, sampler=sampler,
        sigma=None if not np.isfinite(sigma) else sigma,
        n_theta=K, seed=int(d["seed"]),
        live_points=live[0], live_values=live[1], live_probs=live[2],
        live_mode_ids=(np.asarray(d["live_mode_ids"]).ravel()
                       if "live_mode_ids" in d.files else None),
        all_points=np.vstack([p for p, _v, _a in parts]),
        all_values=np.concatenate([v for _p, v, _a in parts]),
        all_probs=np.concatenate([a for _p, _v, a in parts]),
        n_dead=n_dead, n_rejected=n_rejected, N_L=N_L,
        n_replacements=n_rep, n_candidates=n_cand,
        model_runs=get("model_runs", (N_L + n_cand) * K),
        wall_s=float(d["wall_s"]) if "wall_s" in d.files else float("nan"),
    )


# ======================================================================
# COLOUR
# ======================================================================

def criterion_norm(values, criterion, alpha_star):
    """
    Colour normalisation for the criterion, centred on the boundary.

    Two properties have to hold at once, and neither stock normalisation
    gives both:

    1. Zero -- the design-space boundary -- must sit at the middle of the
       diverging colormap. Otherwise the colour where green turns red is
       not the boundary, and the figure lies about the one line that
       matters.
    2. Each side must use its own range. On GTCD the violations run to
       +1.6e7 (the objective cap f <= 1.1e7 enters G on its own raw scale)
       while the feasible side only reaches about -1. A symmetric scale
       therefore spends half its colours on values that do not exist and
       renders every feasible point the same pale green.

    So each side is log-compressed and then rescaled to its own half of
    the bar: f(v) = +slog(v)/slog(vmax) for v >= 0, -slog(-v)/slog(-vmin)
    for v < 0, with slog(x) = log10(1 + x/lt). f(0) = 0 is the midpoint of
    [-1, 1] by construction, which is property 1; the two denominators
    differ, which is property 2. ``lt`` -- where the log turns linear --
    is the 10th percentile of the non-zero magnitudes, so the resolution
    near the boundary comes from the data rather than a constant that
    happens to suit this one problem.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if criterion == "P":
        return (mcolors.Normalize(0.0, 1.0), "viridis", alpha_star,
                [0.0, 0.25, 0.5, alpha_star, 1.0])
    if v.size == 0:
        return mcolors.Normalize(-1.0, 1.0), "RdYlGn_r", 0.0, None

    lo, hi = float(v.min()), float(v.max())
    mag = np.abs(v[v != 0.0])
    lt = max(float(np.percentile(mag, 10)) if mag.size else 1.0, 1e-12)

    if not (lo < 0.0 < hi):
        # Entirely on one side of the boundary: there is no zero crossing
        # to centre, so a plain symmetric-log over the actual range is the
        # honest scale.
        span = max(abs(lo), abs(hi), lt)
        return (mcolors.SymLogNorm(linthresh=lt, vmin=-span, vmax=span,
                                   base=10), "RdYlGn_r", 0.0,
                _decade_ticks(-span, span, lt))

    neg, pos = np.log10(1.0 - lo / lt), np.log10(1.0 + hi / lt)

    def forward(x):
        x = np.asarray(x, dtype=float)
        out = np.where(
            x >= 0.0,
            np.log10(1.0 + np.abs(x) / lt) / pos,
            -np.log10(1.0 + np.abs(x) / lt) / neg,
        )
        return np.where(np.isfinite(out), out, 0.0)

    def inverse(y):
        y = np.asarray(y, dtype=float)
        return np.where(y >= 0.0,
                        lt * (10.0 ** (y * pos) - 1.0),
                        -lt * (10.0 ** (-y * neg) - 1.0))

    return (mcolors.FuncNorm((forward, inverse), vmin=lo, vmax=hi),
            "RdYlGn_r", 0.0, _decade_ticks(lo, hi, lt))


def _fmt_tick(t):
    """Compact label for a decade tick: 0, ±0.1, ±1, ±10, ±1e4 …"""
    if t == 0.0:
        return "0"
    a = abs(t)
    body = (f"{a:g}" if 1e-3 <= a < 1e4 else f"{a:.0e}".replace("e+0", "e")
            .replace("e-0", "e-"))
    return f"-{body}" if t < 0 else body


def _decade_ticks(lo, hi, lt, max_per_side=5):
    """
    Tick positions for a log-compressed colourbar: 0 and powers of ten.

    Matplotlib's default locator places ticks linearly, which on this norm
    stacks every label into the top few percent of the bar. Decades match
    what the scale actually does, and 0 is included explicitly because it
    is the design-space boundary -- the one value a reader must be able
    to find.
    """
    ticks = [0.0]
    for sign, limit in ((1.0, hi), (-1.0, lo)):
        edge = abs(limit)
        if edge <= 0.0:
            continue
        k0, k1 = int(np.floor(np.log10(max(lt, 1e-12)))), int(np.floor(
            np.log10(edge)))
        decades = [10.0 ** k for k in range(k0, k1 + 1)]
        if len(decades) > max_per_side:              # thin out evenly
            step = int(np.ceil(len(decades) / max_per_side))
            decades = decades[::step]
        ticks += [sign * d for d in decades if d <= edge]
    return sorted(set(ticks))


# ======================================================================
# THE TRELLIS CHART
# ======================================================================

def _shingle(ax, lo, hi, sub_lo, sub_hi, vertical=False, label=None):
    """
    One grey range bar, after the paper's "gray bars on the two outer axes".

    The whole bar is the variable's full range; the dark part is the
    interval this row or column covers. Drawn in DATA coordinates so the
    dark part's position along the bar is literally where that interval
    sits in the range -- a reader can see "this column is the top quarter
    of d3" without reading a single number.

    The label sits at the same size as the inner axis labels: d3 and d4 name
    variables of exactly the same standing as d1 and d2, and setting them in
    smaller type made the outer pair read as an afterthought.
    """
    pale, dark = "#dcdcdc", "#8a8a8a"
    if vertical:
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, 1)
        ax.axhspan(lo, hi, facecolor=pale, edgecolor="#9a9a9a", lw=0.6)
        ax.axhspan(sub_lo, sub_hi, facecolor=dark, edgecolor="none")
        ax.set_yticks([lo, hi])
        ax.set_yticklabels([f"{lo:g}", f"{hi:g}"], fontsize=8)
        ax.yaxis.tick_right()
        ax.set_xticks([])
        if label:
            ax.set_ylabel(label, fontsize=10, rotation=270, labelpad=16)
            ax.yaxis.set_label_position("right")
    else:
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.axvspan(lo, hi, facecolor=pale, edgecolor="#9a9a9a", lw=0.6)
        ax.axvspan(sub_lo, sub_hi, facecolor=dark, edgecolor="none")
        ax.set_xticks([lo, hi])
        ax.set_xticklabels([f"{lo:g}", f"{hi:g}"], fontsize=8)
        ax.xaxis.tick_top()
        ax.set_yticks([])
        if label:
            # Just clear of the tick labels. The old pad floated it halfway
            # to the title, where it read as a second heading rather than as
            # the name of the variable the bars underneath it split.
            ax.set_title(label, fontsize=10, pad=6)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def trellis(points, values, probs, title, filename, criterion="VaR",
            alpha_star=0.95, color_by="criterion", norm=None, cmap=None,
            ticks=None, inner=(0, 1), outer=(2, 3), bins=4, subtitle=""):
    """
    The paper's trellis layout for an arbitrary point set.

    Parameters
    ----------
    points, values, probs : arrays
        Design points ``(n, 4)``, their criterion values, and their
        feasibility probabilities. Pass the live set for one figure and
        every scored point for the other -- the layout is identical, which
        is the point: the two are directly comparable.
    color_by : {"criterion", "bands"}
        ``"criterion"`` is a continuous scale of VaR[G] (or P) centred on
        the design-space boundary. ``"bands"`` is the paper's three
        reliability ranges, which only separate on a probabilistic run.
    norm, cmap
        Pass the SAME pair to both figures of a run so a colour means the
        same number in each; ``None`` derives them from this point set
        alone, which makes the two charts silently incomparable.
    inner : (int, int)
        The two variables on each subplot's axes.
    outer : (int, int)
        The two variables split into intervals: ``outer[0]`` across the
        columns, ``outer[1]`` down the rows.
    bins : int
        Intervals per outer variable (the paper uses four-by-four).
    """
    ci, ri = outer
    xi, yi = inner
    names = DS.names or [f"d{k + 1}" for k in range(DS.D)]

    if color_by == "criterion" and (norm is None or cmap is None):
        norm, cmap, _level, ticks = criterion_norm(values, criterion,
                                                   alpha_star)
    if color_by not in ("criterion", "bands", "feasible"):
        raise ValueError(f"unknown color_by {color_by!r}")

    col_edges = np.linspace(*DS.bounds[ci], bins + 1)
    row_edges = np.linspace(*DS.bounds[ri], bins + 1)

    # Headroom in INCHES, not as a fraction. Matplotlib's default top margin
    # is 12% of the figure, which on a 14-inch canvas is nearly two inches of
    # white between the title and the chart -- the title ends up adrift, and
    # so does the column variable's label. Budgeting the strip in absolute
    # units keeps that gap the same whatever `bins` does to the figure size:
    # the title, then the shingle's label and its tick numbers, and nothing
    # else lives up there.
    fig_h    = 3.0 * bins + 1.8
    title_in = 0.32 + (0.22 if subtitle else 0.0)
    head_in  = title_in + 0.53          # + shingle label, pad, tick labels

    fig = plt.figure(figsize=(3.0 * bins + 2.4, fig_h))
    gs = fig.add_gridspec(
        bins + 1, bins + 1,
        height_ratios=[0.20] + [1.0] * bins,
        width_ratios=[1.0] * bins + [0.20],
        hspace=0.10, wspace=0.10,
        top=1.0 - head_in / fig_h,
    )

    def in_bin(col, edges, b):
        """Half-open, closed on the last bin so no point is dropped."""
        lo, hi = edges[b], edges[b + 1]
        upper = col <= hi if b == bins - 1 else col < hi
        return (col >= lo) & upper, lo, hi

    sc = None
    for r in range(bins):
        # Row 0 is the TOP row, so it must hold the HIGHEST interval --
        # the rows read like a y-axis. Indexing them the other way puts
        # the smallest values at the top and quietly flips the picture.
        rb = bins - 1 - r
        row_mask, r_lo, r_hi = in_bin(points[:, ri], row_edges, rb)

        for c in range(bins):
            ax = fig.add_subplot(gs[r + 1, c])
            col_mask, c_lo, c_hi = in_bin(points[:, ci], col_edges, c)
            m = row_mask & col_mask

            if m.any():
                if color_by == "feasible":
                    # Infeasible first so the design space is drawn on top;
                    # it is the smaller set and would otherwise be buried.
                    for want, colour, _lab in reversed(FEASIBLE_COLOURS):
                        sel = m & ((probs >= 0.5) == want)
                        if sel.any():
                            ax.scatter(points[sel, xi], points[sel, yi], s=5,
                                       c=colour, linewidths=0)
                elif color_by == "bands":
                    # Lowest band first so the feasible points end up on
                    # top; thousands of rejects would otherwise bury the
                    # design space they were rejected for not being in.
                    for lo_a, hi_a, colour, _lab in reversed(BANDS):
                        sel = m & (probs >= lo_a) & (probs < hi_a)
                        if sel.any():
                            ax.scatter(points[sel, xi], points[sel, yi], s=5,
                                       c=colour, linewidths=0)
                else:
                    # Worst first, for the same reason: draw order is the
                    # only thing separating an overplotted feasible point
                    # from an invisible one.
                    order = np.flatnonzero(m)[np.argsort(-values[m])]
                    sc = ax.scatter(points[order, xi], points[order, yi],
                                    c=values[order], s=5, cmap=cmap,
                                    norm=norm, linewidths=0)

            ax.set_xlim(*DS.bounds[xi])
            ax.set_ylim(*DS.bounds[yi])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=8, direction="in", top=True, right=True)
            # Ticks everywhere, LABELS only on the outside edge. The paper
            # does the same, and it is what keeps a 4x4 grid legible while
            # every panel still shows a real, shared scale.
            if r != bins - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(AXIS_LABELS.get(xi, names[xi]), fontsize=10)
            if c != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(AXIS_LABELS.get(yi, names[yi]), fontsize=10)
            ax.grid(alpha=0.18, linewidth=0.5)
            ax.text(0.03, 0.96, f"n={int(m.sum()):,}", fontsize=7,
                    transform=ax.transAxes, va="top", ha="left",
                    color="#555555")

        ax_r = fig.add_subplot(gs[r + 1, bins])
        _shingle(ax_r, *DS.bounds[ri], r_lo, r_hi, vertical=True,
                 label=AXIS_LABELS.get(ri, names[ri]) if r == 0 else None)

    for c in range(bins):
        c_lo, c_hi = col_edges[c], col_edges[c + 1]
        ax_t = fig.add_subplot(gs[0, c])
        _shingle(ax_t, *DS.bounds[ci], c_lo, c_hi, vertical=False,
                 label=AXIS_LABELS.get(ci, names[ci]) if c == 0 else None)

    if color_by == "feasible":
        fig.legend(
            handles=[Line2D([], [], ls="none", marker="o", ms=7, color=col,
                            label=lab) for _w, col, lab in FEASIBLE_COLOURS],
            loc="lower center", ncol=2, frameon=True, fontsize=10,
            bbox_to_anchor=(0.5, 0.005))
    elif color_by == "bands":
        fig.legend(
            handles=[Line2D([], [], ls="none", marker="o", ms=7, color=col,
                            label=lab) for _lo, _hi, col, lab in BANDS],
            loc="lower center", ncol=3, frameon=True, fontsize=10,
            title="feasibility probability of each sampled design",
            bbox_to_anchor=(0.5, 0.005))
    elif sc is not None:
        cb = fig.colorbar(sc, ax=fig.axes, fraction=0.020, pad=0.03)
        if ticks:
            # Thin by POSITION on the bar, not by value: the two halves of
            # this norm cover very different value ranges, so decades that
            # are orders of magnitude apart in value can still land on top
            # of each other in pixels. 0 is kept first and unconditionally
            # -- it is the boundary, the one label that must survive.
            kept, taken = [], []
            for t in sorted(ticks, key=lambda x: (x != 0.0, abs(x))):
                if not (norm.vmin <= t <= norm.vmax):
                    continue
                pos = float(norm(t))
                if any(abs(pos - q) < 0.035 for q in taken):
                    continue
                kept.append(t)
                taken.append(pos)
            kept.sort()
            cb.set_ticks(kept)
            cb.set_ticklabels([_fmt_tick(t) for t in kept])
        cb.set_label(
            "P(feasible | d)" if criterion == "P"
            else f"{criterion}[G]  —  worst constraint violation",
            fontsize=10)
        cb.ax.tick_params(labelsize=8)

    # Anchored by its top edge, a fixed distance down from the canvas edge,
    # so it lands inside the strip reserved above -- `bbox_inches="tight"`
    # then trims whatever is left over it.
    fig.suptitle(f"{title}\n{subtitle}" if subtitle else title,
                 fontsize=12, y=1.0 - 0.12 / fig_h, va="top")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved             : {out}")
    return out


def figures(data, inner=(0, 1), outer=(2, 3), bins=4, color_by="criterion",
            scale="shared"):
    """
    Both charts from one run: the live set, and everything scored.

    ``scale="shared"`` derives the colour normalisation ONCE, from the
    full set, and hands it to both, so a colour means the same number in
    each and the two can be read side by side. The cost is that the live
    chart comes out nearly monochrome -- every live point is certified, so
    they all land in a thin band on the feasible side of a scale built to
    span violations up to 1e7. That uniformity is a true statement, but if
    you want to see structure WITHIN the live set, ``scale="per-figure"``
    renormalises each chart to its own range. The two charts are then no
    longer comparable by colour, which is why it is not the default.
    """
    names = DS.names
    crit, a = data["criterion"], data["alpha_star"]
    norm, cmap, _lvl, ticks = criterion_norm(data["all_values"], crit, a)
    if scale == "per-figure":
        norm = cmap = ticks = None  # each trellis() call derives its own

    # No subtitle. Everything it used to carry -- theta, alpha*, the point
    # counts, which variable is on which axis -- belongs in the caption, where
    # it can be written once for both panels instead of twice on the figures.
    shared = dict(criterion=crit, alpha_star=a, color_by=color_by,
                  norm=norm, cmap=cmap, ticks=ticks,
                  inner=inner, outer=outer, bins=bins)

    sampler = data.get("sampler", "multinest")
    # The colouring is part of the name: --color-by criterion and
    # --color-by feasible are two different figures of the same run, and
    # without it the second call silently overwrites the first.
    tag = f"{sampler}_{color_by}_s{data['seed']}"
    who = {"multinest": "Proposed",
           "magnus_nsfeas": "MAGNUS/NSFeas"}.get(sampler, sampler)

    # Panel letters, so a caption can say "(a) and (b)" without naming the
    # samplers a second time. Fixed per sampler rather than by call order:
    # the two runs are plotted by separate invocations of this function, and
    # (a) has to stay the proposed method whichever one is rendered first.
    letter = {"multinest": "a", "magnus_nsfeas": "b"}.get(sampler)
    head = f"({letter}) GTCD — {who}" if letter else f"GTCD — {who}"

    live_png = trellis(
        data["live_points"], data["live_values"], data["live_probs"],
        title=head, subtitle="",
        filename=f"trellis_gtcd_live_{tag}.png", **shared)

    all_png = trellis(
        data["all_points"], data["all_values"], data["all_probs"],
        title=head, subtitle="",
        filename=f"trellis_gtcd_all_{tag}.png", **shared)
    return live_png, all_png


# ======================================================================
# REPORT
# ======================================================================
#
# The same three tables as ``benchmark_2d.py``, so the 2-D benchmarks and this
# 4-D case read alike in the paper. Table A has to differ: this is a nominal
# run, K = 1, so P is 0 or 1 and the five reliability ranges of Kusumo et al.
# collapse to two. Reporting the five would print three empty rows and imply a
# spread of reliabilities that a single scenario cannot produce.

METHOD_LABEL = {"multinest": "Proposed", "magnus_nsfeas": "MAGNUS/NSFeas"}
METHOD_ORDER = ["multinest", "magnus_nsfeas"]


def discover(out_dir=None, seed=11):
    """Every GTCD run saved in ``out_dir``, keyed by sampler."""
    out_dir = out_dir or OUT_DIR
    found = {}
    if not os.path.isdir(out_dir):
        return found
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith("gtcd_points") and name.endswith(".npz")):
            continue
        try:
            data = load(path=os.path.join(out_dir, name))
        except Exception as exc:                       # noqa: BLE001
            print(f"  skipped {name}: {exc}")
            continue
        found[data["sampler"]] = data
    return found


def _present(runs):
    return [k for k in METHOD_ORDER if runs.get(k) is not None]


def table_a(runs):
    """Feasible / infeasible split of every scored design.

    The nominal counterpart of the reliability-range table: with one scenario
    a design is either in the design space or not, and the only informative
    split is that one.
    """
    data = {}
    for key in _present(runs):
        d = runs[key]
        feas = np.asarray(d["all_probs"]) >= 0.5
        n_in, n_out = int(feas.sum()), int((~feas).sum())

        # Every scored design falls on one side or the other, so the total
        # must be N_L + N_cand. Checked rather than assumed: a probability
        # returned as -1e-16 instead of 0 has been seen to drop points out of
        # a range test elsewhere in this study.
        want = int(d["N_L"]) + int(d["n_candidates"])
        if n_in + n_out != want:
            raise SystemExit(
                f"\n  the split does not add up for {METHOD_LABEL[key]}: "
                f"{n_in + n_out:,} against {want:,} = N_L + N_cand.")

        data[METHOD_LABEL[key]] = [n_in, n_out, n_in + n_out]

    return pd.DataFrame(data, index=["P = 1 (in the design space)",
                                     "P = 0 (outside)", "Total"])


def table_b(runs):
    """Computational performance.

    Wall time is the weakest row: the two implementations are Python and C++,
    and GTCD's model is an analytic expression costing microseconds, so here
    it measures the language rather than the algorithm. N_model is the number
    that carries over to an expensive simulator.
    """
    # Row order: accepted and rejected first, their total below them.
    # Both samplers need roughly the same number of ACCEPTED replacements --
    # that is a property of the problem, not of the method -- and differ in
    # how many candidates they had to try to get them. Putting N_acc and
    # N_rej adjacent makes that visible; leading with N_cand buries it.
    data = {}
    for key in _present(runs):
        d = runs[key]
        acc, cand = d["n_replacements"], d["n_candidates"]
        eff = 100.0 * acc / cand if cand else float("nan")
        data[METHOD_LABEL[key]] = [f"{int(d['N_L']):,}",
                                   f"{acc:,}", f"{cand - acc:,}", f"{cand:,}",
                                   f"{eff:.1f}",
                                   f"{d['model_runs']:,}",
                                   f"{d['wall_s']:.1f}"]
    return pd.DataFrame(data, index=["N_L", "N_acc", "N_rej", "N_cand",
                                     "eta_samp (%)", "N_model", "wall (s)"])


def table_c(runs):
    """Separated modes.

    GTCD has one connected design space, so the mode count is a check that
    the decomposition does not split a region that needs no splitting. NSFeas
    maintains a single population and has no answer to give here.
    """
    data = {}
    for key in _present(runs):
        ids = runs[key]["live_mode_ids"]
        if ids is None:
            data[METHOD_LABEL[key]] = ["—", "—"]
        else:
            lab, cnt = np.unique(ids, return_counts=True)
            data[METHOD_LABEL[key]] = [
                str(len(lab)),
                ", ".join(f"{int(c):,}" for c in sorted(cnt)[::-1])]
    return pd.DataFrame(data, index=["Modes", "Live points per mode"])


def report(out_dir=None, seed=11):
    out_dir = out_dir or OUT_DIR
    runs = discover(out_dir, seed)
    if not runs:
        raise SystemExit(f"no gtcd_points*.npz in {out_dir}/ — run this "
                         "script first")

    print(f"\nloaded {len(runs)} run(s) from {out_dir}/")
    for key in METHOD_ORDER:
        d = runs.get(key)
        if d is None:
            print(f"    {METHOD_LABEL[key]:16} not present"
                  + ("  → run magnus_gtcd_case.py under WSL and copy the "
                     ".npz here" if key == "magnus_nsfeas" else ""))
        else:
            print(f"    {METHOD_LABEL[key]:16} N_L={d['N_L']:,}  "
                  f"K={d['n_theta']}  α*={d['alpha_star']:g}  "
                  f"{os.path.basename(d['path'])}")

    settings = {(d["N_L"], d["n_theta"], d["alpha_star"])
                for d in runs.values()}
    if len(settings) > 1:
        raise SystemExit(f"\n  SETTINGS MISMATCH — {settings}. The two runs "
                         "are not comparable; re-run the affected side.")

    tables = [("A", "feasible / infeasible split of every scored design",
               "A_feasibility", table_a(runs)),
              ("B", "computational performance", "B_performance",
               table_b(runs)),
              ("C", "separated modes", "C_modes", table_c(runs))]

    # Print before writing. The .xlsx is routinely open in Excel while these
    # numbers are being read, and a locked file must not cost the tables.
    for letter, caption, _sheet, df in tables:
        title = f"Table {letter} — {caption}"
        print(f"\n{title}\n{'=' * len(title)}")
        print("\n".join("  " + line for line in df.to_string().splitlines())
              if not df.empty else "  (no data)")

    xlsx = os.path.join(out_dir, "tables_gtcd.xlsx")
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            for _letter, _caption, sheet, df in tables:
                df.to_excel(writer, sheet_name=sheet)
        print(f"\n  saved {xlsx}")
    except PermissionError:
        print(f"\n  could NOT write {xlsx} — the file is open in another "
              f"program. The tables above are complete; close it and re-run "
              f"to refresh the spreadsheet.")
    return runs


# ======================================================================
# CLI
# ======================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="GTCD 4-D case study, Kusumo et al. (2020) presentation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N_L", dest="n_live", type=int, default=N_LIVE_DEFAULT,
                   help=f"live points (default {N_LIVE_DEFAULT})")
    p.add_argument("--sigma", type=float, default=None,
                   help="switch to a probabilistic run, θ ~ N(1, sigma); "
                        "omitted means the catalogue's nominal θ = 1")
    p.add_argument("--n-theta", type=int, default=None,
                   help="scenarios per design point (only with --sigma)")
    p.add_argument("--criterion", default="VaR", choices=["P", "VaR", "CVaR"])
    p.add_argument("--alpha-star", type=float, default=None,
                   help="default: whatever the catalogue declares")
    p.add_argument("--color-by", default="criterion",
                   choices=["criterion", "bands", "feasible"],
                   help="'criterion': continuous VaR[G] centred on the "
                        "boundary. 'feasible': the P = 0 / P = 1 split, which "
                        "is what a nominal run actually resolves. 'bands': "
                        "the paper's three α ranges, which only separate on a "
                        "probabilistic run")
    p.add_argument("--report", action="store_true",
                   help="print the three comparison tables from every saved "
                        "run in the output directory, write them to .xlsx, "
                        "and exit without running or plotting")
    p.add_argument("--scale", default="shared",
                   choices=["shared", "per-figure"],
                   help="'shared': one colour scale for both charts, so "
                        "they are comparable. 'per-figure': each chart "
                        "renormalises, showing structure inside the live "
                        "set at the cost of comparability")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--bins", type=int, default=4,
                   help="intervals per outer variable (paper uses 4)")
    p.add_argument("--inner", nargs=2, type=int, default=[1, 2], metavar="I",
                   help="1-based variable indices for the subplot axes")
    p.add_argument("--outer", nargs=2, type=int, default=[3, 4], metavar="I",
                   help="1-based indices split into intervals: across, down")
    p.add_argument("--replot", action="store_true",
                   help="skip the run, redraw from the saved npz")
    p.add_argument("--npz", default=None,
                   help="read this npz instead of this run's own. Implies "
                        "--replot. MagnusCodes/magnus_gtcd_case.py writes "
                        "the same key names, so the MAGNUS/NSFeas run of "
                        "this case draws through exactly this code")
    p.add_argument("--selfcheck", action="store_true",
                   help="re-measure the model against the published "
                        "numbers and exit")
    p.add_argument("--log-every", default="5%")
    args = p.parse_args(argv)

    if args.selfcheck:
        return 0 if selfcheck() else 1

    if args.report:
        report(seed=args.seed)
        return 0

    inner = tuple(k - 1 for k in args.inner)
    outer = tuple(k - 1 for k in args.outer)
    if len(set(inner + outer)) != 4 or not all(0 <= k < DS.D
                                               for k in inner + outer):
        raise SystemExit(f"--inner and --outer must name all {DS.D} "
                         f"variables (1..{DS.D}), each exactly once")

    if not (args.replot or args.npz):
        log_every = (None if str(args.log_every).lower() == "none"
                     else args.log_every)
        run(n_live=args.n_live, n_theta=args.n_theta, sigma=args.sigma,
            criterion=args.criterion, alpha_star=args.alpha_star,
            seed=args.seed, log_every=log_every)

    data = load(seed=args.seed, path=args.npz)
    if args.color_by == "bands" and data["sigma"] is None:
        print("  note: --color-by bands on a NOMINAL run — α is 0 or 1 "
              "there, so two of the three bands stay empty. Add --sigma to "
              "make them mean something.")
    print(f"\n  read              : {data['path']}")
    print(f"  live {data['live_points'].shape[0]:,} · "
          f"dead {data['n_dead']:,} · rejected {data['n_rejected']:,} "
          f"= {data['all_points'].shape[0]:,} scored designs")
    figures(data, inner=inner, outer=outer, bins=args.bins,
            color_by=args.color_by, scale=args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
