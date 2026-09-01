"""
Two-dimensional benchmark comparison, MAGNUS/NSFeas half.

Companion to ``benchmark_2d.py`` on the Windows side. It runs NSFeas on the
same two problems, at the same settings, and writes ``.npz`` files in the same
schema, so that

    python benchmark_2d.py report

produces the paper's figure and tables from both samplers without needing this
machine or this library. Copy the ``.npz`` files this script writes into the
Windows ``benchmark_2d_output/`` directory and re-run that command.

    kusumo   s = th*d1^2 + d2,  th ~ N(1, sqrt(0.3)),  0.20 <= s <= 0.75
    banana   s = th*(d1^2 - 0.5) + d2 + 0.3*d1,  th ~ N(1, 0.5),
             0.00 <= s <= 0.40

SETTINGS ARE DUPLICATED, AND CHECKED
------------------------------------
The two halves run on different operating systems and cannot share a module,
so N_L, K and alpha* are written out in ``PROBLEMS`` below as well as in
``multinest_sampler.EXAMPLES`` on the Windows side. Duplicated constants
drift, so every value used here is also stored in the ``.npz``, and
``benchmark_2d.py report`` compares them against its own and refuses to build
a table from settings that do not match. If it complains, this file is what
needs updating.

TWO DIFFERENCES FROM THE WINDOWS SIDE, BOTH DELIBERATE AND BOTH RECORDED
------------------------------------------------------------------------
1.  SCENARIOS.  NSFeas takes one fixed scenario set for a whole run
    (``set_parameter``), so its criterion is deterministic in d. The Python
    estimator redraws theta at every design-point evaluation, so its criterion
    is stochastic. Same distribution, same K, different statistics: the fixed
    set makes the surface smooth and cheap to search but lets the sampler
    overfit those K scenarios; the redraw prevents that at the cost of
    acceptance rate. The difference acts in a known direction -- it depresses
    the Python side's acceptance rate -- and is reported as such rather than
    removed. ``scenario_mode`` in the .npz records which was used.

    The two sides therefore do not see the same theta values, and no choice of
    seed would make them: this file draws from ``np.random.default_rng``, the
    Python side from numpy's legacy global RNG, and it redraws per evaluation
    besides. Only the distribution and K are common. Matching the seed number
    across the two is meaningless and is not attempted.

2.  REPEATS.  NSFeas draws its proposals from a Sobol sequence seeded with a
    constant in its own source (``nsfeas.hpp``: ``_gen.engine().seed(0)``) and
    exposes no RNG seed of its own, so there is no spread to average over.
    ``--seed`` below therefore controls only the SCENARIO draw.

    Measured, not assumed. Running this script twice at the same ``--seed``
    gives bit-identical live points on both problems, and running it at a
    different ``--seed`` does not:

        python3 magnus_benchmark_2d.py --out check_a
        python3 magnus_benchmark_2d.py --out check_b
        python3 magnus_benchmark_2d.py --seed 23 --out check_c
        # a == b : True     a == c : False      (kusumo and banana alike)

    That is the whole of the asymmetry with the Python side, which is
    stochastic and whose seed does change its trajectory.

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
    python magnus_benchmark_2d.py

Write straight into the other side's output directory and the report picks
the runs up with no copying:

    python magnus_benchmark_2d.py --out <path to the benchmark_2d_output dir>

then, on the Windows side:

    python benchmark_2d.py report

Other forms:

    python magnus_benchmark_2d.py --problems banana
    python magnus_benchmark_2d.py --seed 23
"""

import argparse
import os
import time

import sys

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


OUT_DIR = "benchmark_2d_output"
SAMPLER = "nsfeas"

# Kept identical to multinest_sampler.EXAMPLES on the Windows side; see the
# note above on why they are duplicated and how the mismatch is caught.
DESIGN = dict(lb=[-1.0, -1.0], ub=[1.0, 1.0])

# NSFeas options that are not part of the comparison. NUMPROP is its proposals
# per contour; ELL* control the single bounding ellipsoid it grows.
COMMON_NS = dict(DISPLEVEL=1, NUMPROP=8, MAXITER=0,
                 ELLCONF=0.99, ELLMAG=0.30, ELLRED=0.20)


def _eq_kusumo(d, p):
    """s = theta * d1^2 + d2"""
    return p * d[0] ** 2 + d[1]


def _eq_banana(d, p):
    """s = theta * (d1^2 - 0.5) + d2 + 0.3 * d1"""
    return p * (d[0] ** 2 - 0.5) + d[1] + 0.3 * d[0]


PROBLEMS = {
    "kusumo": dict(
        equation=_eq_kusumo,
        mu=1.0, sigma=float(np.sqrt(0.3)), nscen=100,
        constraint=(0.20, 0.75),
        numlive=500, alpha_star=0.95,
    ),
    "banana": dict(
        equation=_eq_banana,
        mu=1.0, sigma=0.5, nscen=100,
        constraint=(0.00, 0.40),
        numlive=900, alpha_star=0.95,
    ),
}


def build_constraints(y, bounds):
    """(lb, ub) on the model output -> DAG expressions that are <= 0."""
    lb, ub = bounds
    cons = []
    if np.isfinite(ub):
        cons.append(y - ub)
    if np.isfinite(lb):
        cons.append(lb - y)
    if not cons:
        raise ValueError("problem declares no active constraint")
    return cons


def run_one(key, seed, out_dir):
    cfg = PROBLEMS[key]
    alpha = 1.0 - cfg["alpha_star"]

    print("=" * 66)
    print(f"{key}   NSFeas   NUMLIVE={cfg['numlive']}  K={cfg['nscen']}  "
          f"alpha*={cfg['alpha_star']}  (FEASTHRES={alpha:g})")
    print("=" * 66)

    DAG = pymc.FFGraph()
    DAG.options.MAXTHREAD = 1          # the model, not the sampler
    d = DAG.add_vars(2, "d")
    p = DAG.add_var("p")
    y = cfg["equation"](d, p)

    NS = NSFeas()
    for name, value in COMMON_NS.items():
        setattr(NS.options, name, value)
    NS.options.FEASCRIT = NS.options.VAR      # P is not orderable; see 2.2
    NS.options.FEASTHRES = alpha
    NS.options.NUMLIVE = cfg["numlive"]
    NS.options.MAXTHREAD = 1                  # single-threaded, as timed
    NS.set_dag(DAG)
    NS.set_constraint(build_constraints(y, cfg["constraint"]))
    NS.set_control(d, DESIGN["lb"], DESIGN["ub"])

    # One fixed scenario set for the whole run -- this is the NSFeas model of
    # uncertainty, and the difference from the Python side noted at the top.
    rng = np.random.default_rng(seed)
    psam = [[v] for v in rng.normal(cfg["mu"], cfg["sigma"], cfg["nscen"])]
    NS.set_parameter([p], psam)

    NS.setup()
    t0 = time.perf_counter()
    NS.sample()
    wall = time.perf_counter() - t0

    live_pts, live_crit, live_prob, _ = NS.live_points
    dead_pts, dead_crit, dead_prob, _ = NS.dead_points
    disc_pts, disc_crit, disc_prob, _ = NS.discard_points

    live_pts = np.asarray(live_pts, dtype=float).reshape(-1, 2)
    dead_pts = np.asarray(dead_pts, dtype=float).reshape(-1, 2)
    disc_pts = np.asarray(disc_pts, dtype=float).reshape(-1, 2)
    live_crit = np.asarray(live_crit, dtype=float).ravel()
    dead_crit = np.asarray(dead_crit, dtype=float).ravel()
    disc_crit = np.asarray(disc_crit, dtype=float).ravel()

    n_rep = len(dead_pts)                     # one eviction, one replacement
    n_cand = n_rep + len(disc_pts)            # accepted + rejected candidates
    model_runs = int(NS.stats.numfct)

    path = os.path.join(out_dir, f"points_{SAMPLER}_{key}.npz")
    np.savez_compressed(
        path,
        live_points=live_pts,
        # merit is higher-is-better; for VaR that is the negated criterion,
        # which is the convention the Windows SamplerResult stores.
        live_merit=-live_crit,
        live_probs=np.asarray(live_prob, dtype=float).ravel(),
        live_crit=live_crit,
        dead_points=dead_pts, dead_merit=-dead_crit,
        dead_probs=np.asarray(dead_prob, dtype=float).ravel(),
        rejected_points=disc_pts, rejected_merit=-disc_crit,
        rejected_probs=np.asarray(disc_prob, dtype=float).ravel(),
        criterion="VaR",
        alpha_star=float(cfg["alpha_star"]),
        N_L=int(cfg["numlive"]),
        n_theta=int(cfg["nscen"]),
        n_replacements=int(n_rep),
        n_candidates=int(n_cand),
        model_runs=model_runs,
        wall_s=float(wall),
        converged=True,
        n_modes=0,                            # no mode separation in NSFeas
        n_failed=int(NS.stats.numerr),
        scenario_mode="fixed",
        scenario_seed=int(seed),
        sampler=SAMPLER, problem=key,
    )

    eff = 100.0 * n_rep / n_cand if n_cand else float("nan")
    print(f"  live {len(live_pts):,}   dead {n_rep:,}   discarded "
          f"{len(disc_pts):,}")
    print(f"  candidates {n_cand:,}   model runs {model_runs:,}   "
          f"efficiency {eff:.1f}%   {wall:.1f}s")
    print(f"  saved {path}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--problems", nargs="+", default=list(PROBLEMS),
                   choices=list(PROBLEMS))
    p.add_argument("--seed", type=int, default=11,
                   help="seed for the SCENARIO draw only; NSFeas exposes no "
                        "sampler seed. This does NOT reproduce the Windows "
                        "side's scenarios even at the same value: that uses a "
                        "different generator and redraws per evaluation. Only "
                        "the distribution and K are shared.")
    p.add_argument("--out", default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for key in args.problems:
        run_one(key, args.seed, args.out)
    print(f"copy {args.out}/points_{SAMPLER}_*.npz into the Windows "
          "benchmark_2d_output/ and run:  python benchmark_2d.py report")


if __name__ == "__main__":
    main()
