"""
High-resolution Monte-Carlo reference fields for the geometric-verification
figure (Section 3.1).

WHAT THIS DRAWS
---------------
One panel per problem: the criterion field over the design box, evaluated on a
fine grid by Monte-Carlo propagation of the uncertainty, with the design-space
boundary (VaR[G] = 0, or P = alpha* for the probability criterion) drawn as a
single iso-line. It is the LEFT panel of ``Visualizer.plot_landscape`` and
nothing else -- no sampler output, no ellipsoids, no live points. This is the
ground truth the sampler's answer is later laid over.

Three problems, one geometric job each (see the catalogue in
``multinest_sampler.EXAMPLES``):

    ellipse   Ex 8, single ellipse    -- one convex region; the decomposition
                                         must NOT split when one ellipsoid
                                         already suffices
    banana    Ex 3, banana + island   -- two UNEQUAL disconnected regions;
                                         the hard case for mode separation,
                                         and where freezing has something to
                                         pay for
    torus     Ex 6, three rings       -- three separated, strongly
                                         non-ellipsoidal regions; recursive
                                         decomposition doing real work inside
                                         a single mode

WHY THE SCENARIO SET IS FROZEN BY DEFAULT
-----------------------------------------
``FeasibilityEstimator`` redraws theta on EVERY design-point evaluation, so the
criterion is genuinely stochastic: the same d evaluated twice does not give the
same value. That is correct for the sampler -- it prevents overfitting one
scenario set -- but it is wrong for a reference field. At 400x400 the noise
lands point-by-point and the VaR[G] = 0 contour comes out visibly ragged, which
reads as a property of the problem when it is a property of the estimator.

So ``--fixed`` (the default) draws ONE set of K scenarios and evaluates every
grid point against it, via ``WeightedScenarios``. The field is then smooth,
deterministic and reproducible from ``--seed`` alone. ``--redraw`` restores the
per-point redraw if the noise itself is what you want to look at.

Note the same distinction settles the MAGNUS/NSFeas comparison: NSFeas takes one
fixed scenario set for a whole run, so a fixed set here is also the like-for-like
choice.

COST
----
``mc_criterion_grid`` is a Python loop over grid points, so cost is
n_grid^2 criterion evaluations, each over K scenarios. Measured on this
machine at the defaults (400x400, K = 2000): about a minute per problem.
Raise ``--n-grid`` for a smoother contour, ``--n-theta`` for a less biased one.

Every run also writes the raw field to ``.npz``, so re-styling the figure
(colours, labels, size) is a ``--from-npz`` re-render and not a recomputation.

Usage
-----
    python reference_grids.py                          # all three, defaults
    python reference_grids.py --cases banana torus
    python reference_grids.py --n-grid 600 --n-theta 4000
    python reference_grids.py --from-npz               # re-render, no compute
    python reference_grids.py --combined               # one 1x3 figure too
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")           # batch: no window, nothing to block on

import matplotlib.pyplot as plt
import numpy as np

import multinest_sampler as mn


OUT_DIR = "reference_grids_output"

# Catalogue index -> the job that entry does in Section 3.1. Indices are into
# ``multinest_sampler.EXAMPLES``; the titles are carried along so a reordered
# catalogue fails loudly instead of silently plotting the wrong problem.
CASES = {
    "ellipse": dict(index=7, expect="8 ·", label="Single ellipse",
                    job="one convex region: must not over-split"),
    "banana":  dict(index=2, expect="3 ·", label="Banana and island",
                    job="two unequal disconnected regions"),
    "torus":   dict(index=5, expect="6 ·", label="Three toroidal regions",
                    job="three non-ellipsoidal regions"),
    "kusumo":  dict(index=3, expect="4 ·", label="Kusumo et al. (2020)",
                    job="published benchmark, one connected region"),
}

# Cost is n_grid^2 * K criterion evaluations through a Python loop, but the
# per-call numpy overhead dominates rather than K, so it scales far better
# than the product suggests. Measured here: about 40 s per problem at these
# defaults, the torus (three exp() per scenario) being the slowest.
N_GRID = 400
N_THETA = 2000
SEED = 11

# Upper percentile the colour scale is clipped to, for DISPLAY only. The raw
# field goes to the .npz untouched. Without it the corner values of the
# quadratic problems (VaR up to ~45 on the ellipse against a boundary at 0)
# compress every feasible point into one end of the colourmap and the panel
# reads as blank. None disables the clip and reproduces the sampler's own
# landscape colouring exactly.
CLIP_PCT = 98.0


# ======================================================================
# 1. FIELD
# ======================================================================

def freeze_uncertainty(uncertainty, K: int, seed: int):
    """One fixed scenario set, drawn from ``uncertainty``, as a
    ``WeightedScenarios``.

    Every grid point is then evaluated against the SAME K scenarios, which is
    what makes the field smooth and deterministic. Going through
    ``get_samples_and_weights`` rather than sampling the distribution directly
    keeps this correct for any UncertaintyDistribution, including one that is
    already a fixed set (in which case this is very nearly a no-op).
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        theta, weights = uncertainty.get_samples_and_weights(K)
    finally:
        np.random.set_state(rng_state)
    return mn.WeightedScenarios(theta_samples=theta,
                                weights=np.asarray(weights, dtype=float),
                                normalise=True)


def build_field(ex: dict, n_grid: int, n_theta: int, criterion: str,
                fixed: bool, seed: int):
    """Evaluate the criterion on an ``n_grid`` x ``n_grid`` mesh.

    Returns ``(D1, D2, V)`` with V in RAW criterion units (not merit), the
    same quantity ``Visualizer.V_grid`` holds.
    """
    ds = ex["design_space"]
    alpha_star = ex["sampler_kw"]["alpha_star"]

    uncertainty = ex["uncertainty"]
    if fixed:
        uncertainty = freeze_uncertainty(uncertainty, n_theta, seed)

    model = mn.ProcessModel(
        equation    = ex["equation"],
        uncertainty = uncertainty,
        constraints = ex["constraints"],
        name        = ex["title"],
    )
    estimator = model.make_estimator(
        uncertainty    = uncertainty,
        N_theta        = n_theta,
        feas_criterion = criterion,
    )

    (lo_x, hi_x), (lo_y, hi_y) = ds.bounds[0], ds.bounds[1]
    D1, D2 = np.meshgrid(np.linspace(lo_x, hi_x, n_grid),
                         np.linspace(lo_y, hi_y, n_grid))
    flat = np.column_stack([D1.ravel(), D2.ravel()])

    # Row by row rather than in one list comprehension, purely so a run that
    # takes minutes shows signs of life. The values are identical either way.
    V = np.empty(flat.shape[0], dtype=float)
    t0 = time.perf_counter()
    for r in range(n_grid):
        s = slice(r * n_grid, (r + 1) * n_grid)
        V[s] = [estimator.criterion_value(d, alpha_star) for d in flat[s]]
        if r % 25 == 24 or r == n_grid - 1:
            done = (r + 1) / n_grid
            el = time.perf_counter() - t0
            print(f"    {done:6.1%}  {el:6.1f}s elapsed, "
                  f"{el / done - el:5.1f}s left", end="\r", flush=True)
    print(f"    done in {time.perf_counter() - t0:.1f}s" + " " * 24)

    return D1, D2, V.reshape(D1.shape)


# ======================================================================
# 2. FIGURE
# ======================================================================

def math_name(name: str) -> str:
    """``"d1"`` -> ``"$d_1$"``: a trailing index becomes a real subscript.

    Design-space names are stored as plain identifiers so they can be printed
    and used as dict keys; on an axis they should be set in maths. Anything
    that is not identifier-plus-digits is passed through untouched.
    """
    stem = name.rstrip("0123456789")
    index = name[len(stem):]
    if not stem or not index:
        return name
    return rf"${stem}_{{{index}}}$"


def draw_panel(ax, D1, D2, V, disp, ds, title: str = "",
               colorbar: bool = True, clip_pct: float = None):
    """The left panel of ``plot_landscape``, on a supplied axis.

    Styling is delegated to ``CriterionDisplay`` -- the same object the
    sampler's own figures use -- so this reference field and the result plots
    laid over it carry identical colours, contour and labels.

    ``clip_pct`` clips the COLOUR range at that percentile of the field. The
    contour is still drawn on the unclipped field, so the design-space
    boundary is unaffected -- only how much of the colourmap the far tail is
    allowed to consume.
    """
    field = disp.sanitise_field(V)

    shown = field
    if clip_pct is not None:
        hi = float(np.percentile(field, clip_pct))
        if hi > float(field.min()):
            shown = np.clip(field, None, hi)

    cf = ax.contourf(D1, D2, shown, alpha=0.9, **disp.field_kwargs(shown),
                     extend="max" if shown is not field else "neither")
    ax.set_aspect("equal")

    if colorbar:
        # The clip is recorded in the .npz metadata and in this file's
        # docstring, not on the colourbar: the arrowhead already says the
        # scale is extended, and the boundary contour is unaffected.
        #
        # Attached through an axes divider rather than plt.colorbar(ax=ax),
        # and AFTER set_aspect, so the bar is the height of the SQUARE
        # panel. The plain call sizes the bar from the axes' allotted box,
        # which equal aspect then shrinks the panel inside of -- leaving a
        # colourbar taller than the panel it describes. Width is set as a
        # percentage of the panel, so it stays proportionate whatever
        # figsize the caller picks.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="4%",
                                                 pad=0.10)
        cb  = plt.colorbar(cf, cax=cax)
        cb.set_label(disp.value_label, fontsize=10)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_linewidth(0.6)

    for level, color, _label in disp.contour_levels():
        ax.contour(D1, D2, field, levels=[level], colors=[color],
                   linewidths=2.0)

    names = getattr(ds, "names", None) or ["d1", "d2"]
    ax.set_xlabel(math_name(names[0]))
    ax.set_ylabel(math_name(names[1]))
    ax.set_xlim(ds.bounds[0])
    ax.set_ylim(ds.bounds[1])
    if title:
        ax.set_title(title, fontsize=11)
    return cf


def render(key: str, D1, D2, V, ex: dict, criterion: str, meta: dict,
           out_dir: str, dpi: int, clip_pct: float) -> None:
    disp = mn.CriterionDisplay(criterion, ex["sampler_kw"]["alpha_star"])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    # No title: the single-problem figures carry their caption in the text,
    # and a title here would only duplicate it. The 1xN row still labels its
    # panels, because there the labels distinguish them from one another.
    draw_panel(ax, D1, D2, V, disp, ex["design_space"], clip_pct=clip_pct)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = os.path.join(out_dir, f"reference_{key}_{criterion}.{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"    saved {path}")
    plt.close(fig)


def render_combined(fields: dict, criterion: str, out_dir: str,
                    dpi: int, clip_pct: float) -> None:
    """One 1x3 row, shared layout -- the figure Section 3.1 actually wants.

    Each panel keeps its OWN colour scale, because the three problems have
    unrelated violation magnitudes and a shared scale would flatten two of
    them. The contour is the comparable object across panels, not the colour.
    """
    keys = [k for k in CASES if k in fields]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.8))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        D1, D2, V, ex = fields[key]
        disp = mn.CriterionDisplay(criterion, ex["sampler_kw"]["alpha_star"])
        draw_panel(ax, D1, D2, V, disp, ex["design_space"],
                   title=f"({'abcdef'[keys.index(key)]}) {CASES[key]['label']}",
                   clip_pct=clip_pct)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = os.path.join(out_dir,
                            f"reference_combined_{criterion}.{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


# ======================================================================
# 3. DRIVER
# ======================================================================

def resolve(key: str) -> dict:
    meta = CASES[key]
    ex = mn.EXAMPLES[meta["index"]]
    if not ex["title"].startswith(meta["expect"]):
        raise SystemExit(
            f"catalogue mismatch for '{key}': EXAMPLES[{meta['index']}] is "
            f"{ex['title']!r}, expected one starting {meta['expect']!r}. "
            "The catalogue was reordered -- fix CASES in this file.")
    return ex


def npz_path(out_dir: str, key: str, criterion: str) -> str:
    """The criterion is part of the name on purpose: Section 3.1 wants the
    VaR field (level sets of the driving merit) and Section 3.2 the P field
    (iso-reliability contours of the terminal set), and the banana problem
    appears in both. Without it the second run silently overwrites the
    first."""
    return os.path.join(out_dir, f"reference_{key}_{criterion}.npz")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cases", nargs="+", default=list(CASES),
                   choices=list(CASES))
    p.add_argument("--n-grid", type=int, default=N_GRID,
                   help=f"points per axis (default {N_GRID}); cost is n^2")
    p.add_argument("--n-theta", type=int, default=N_THETA,
                   help=f"uncertainty scenarios (default {N_THETA})")
    p.add_argument("--criterion", default="VaR", choices=["P", "VaR", "CVaR"])
    p.add_argument("--redraw", action="store_true",
                   help="redraw theta per grid point instead of freezing one "
                        "scenario set (noisier field; see the module docstring)")
    p.add_argument("--seed", type=int, default=SEED,
                   help="seed for the frozen scenario set")
    p.add_argument("--from-npz", action="store_true",
                   help="re-render from saved fields, computing nothing")
    p.add_argument("--combined", action="store_true",
                   help="additionally write the 1xN row figure")
    p.add_argument("--clip-pct", type=float, default=CLIP_PCT,
                   help=f"clip the COLOUR scale at this percentile "
                        f"(default {CLIP_PCT:g}); the boundary contour is "
                        f"drawn on the unclipped field either way")
    p.add_argument("--no-clip", action="store_true",
                   help="no colour clipping — reproduces the sampler's own "
                        "landscape colouring exactly")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out", default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    clip_pct = None if args.no_clip else args.clip_pct
    fields = {}

    for key in args.cases:
        ex = resolve(key)
        print(f"\n{key}  —  {ex['title']}")

        if args.from_npz:
            path = npz_path(args.out, key, args.criterion)
            if not os.path.exists(path):
                raise SystemExit(f"no saved field at {path}; run without "
                                 "--from-npz first")
            d = np.load(path, allow_pickle=False)
            D1, D2, V = d["D1"], d["D2"], d["V"]
            criterion = str(d["criterion"])
            print(f"    loaded {path}  ({V.shape[0]}x{V.shape[1]}, "
                  f"K={int(d['n_theta'])}, {criterion})")
        else:
            criterion = args.criterion
            print(f"    {args.n_grid}x{args.n_grid} grid, K={args.n_theta} "
                  f"scenarios, {'REDRAWN' if args.redraw else 'frozen'}")
            D1, D2, V = build_field(ex, args.n_grid, args.n_theta,
                                    criterion, not args.redraw, args.seed)
            np.savez_compressed(
                npz_path(args.out, key, criterion),
                D1=D1, D2=D2, V=V,
                criterion=criterion,
                alpha_star=float(ex["sampler_kw"]["alpha_star"]),
                n_grid=int(args.n_grid), n_theta=int(args.n_theta),
                fixed_scenarios=(not args.redraw), seed=int(args.seed),
                title=ex["title"],
            )
            print(f"    saved {npz_path(args.out, key, criterion)}")

        render(key, D1, D2, V, ex, criterion, CASES[key], args.out,
               args.dpi, clip_pct)
        fields[key] = (D1, D2, V, ex)

    if args.combined and fields:
        print()
        render_combined(fields, criterion, args.out, args.dpi, clip_pct)


if __name__ == "__main__":
    main()
