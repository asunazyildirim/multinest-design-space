"""Put two feasible design spaces side by side, on identical axes.

    python compare_feasible_spaces.py [proposed.csv] [magnus.csv] [--dark]
                                      [--show] [--separate] [--out FILE]

Writes the two panels as one figure. ``--separate`` also writes them as two
standalone files, on the same limits, for a layout that wants them apart.

The comparison figure for the CO2-to-methanol case study: the same problem
characterised by the proposed sampler and by MAGNUS / NSFeas. Both CSVs are
the schema run_sampler_kinetics_serial.py and magnus_tp_case_study.py both
write -- design columns first, then P, merit, mode, role -- in the same
physical units, so the two panels are directly comparable.

WHAT MAKES THEM COMPARABLE IS THE AXES, and this is the part worth getting
right: both panels are drawn on ONE pair of limits, taken from the union of
the two files. Letting matplotlib fit each panel to its own data would scale
the two regions differently and the figure would misreport the result --
the more localised region would look like the larger one.

Feasibility is read from MERIT, not from role: merit >= 0 is the constraint
verdict (VaR: merit = -(worst violation)), while role records only what the
sampler did with the point. A dead point can be perfectly feasible and
merely have been the worst of the population when it was evicted.

Style -- colours, units, axis furniture -- is imported from
plot_design_space.py rather than repeated, so the single-run figures and
this one cannot drift apart.
"""

import os
import sys

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from plot_design_space import DARK, LIGHT, axis_label, style_axes

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROPOSED = os.path.join(HERE, "tp_design_space_serial.csv")
# The 18 August timed run: --no-warm-start, 0 cache hits, 1612 solves in
# 22195 s. NOT magnus_tp_real.csv, which is the 12 August run made before
# the timing instrumentation existed and whose wall clock cannot be quoted.
DEFAULT_MAGNUS = os.path.join(HERE, "magnus_tp_output",
                              "magnus_tp_bridge_NL500.csv")

PANEL_TITLES = ("(a) CO$_2$-to-methanol — Proposed",
                "(b) CO$_2$-to-methanol — MAGNUS/NSFeas")

# Both CSVs store pressure in kPa, which is what the run scripts pass to
# HYSYS. The case study is written in bar, so the column is converted once
# on load and the axis is labelled to match -- the two must move together,
# which is why the factor and the unit live in one place.
RESCALE = {"pressure": (0.01, "bar")}


def axis_label_rescaled(name):
    """Axis label honouring the conversion above, else the shared one."""
    if name in RESCALE:
        return f"{name.replace('_', ' ').capitalize()} [{RESCALE[name][1]}]"
    return axis_label(name)


def load(path):
    frame = pd.read_csv(path)
    for column in ("merit", "role"):
        if column not in frame.columns:
            raise SystemExit(
                f"{path} has no '{column}' column -- is it one of the design "
                f"space CSVs?")

    for column, (factor, _unit) in RESCALE.items():
        if column in frame.columns:
            frame[column] = frame[column] * factor

    return frame, frame.columns[0], frame.columns[1]


def shared_limits(frames, x_name, y_name):
    """One pair of limits covering both runs, padded by 2 %.

    Taken from the data rather than from a hard-coded design box: the box
    lives in the run scripts, and a copy of it here would be one more thing
    to keep in step. Both runs fill their box densely enough that the union
    is the box to well under a percent.
    """
    limits = []
    for name in (x_name, y_name):
        lo = min(frame[name].min() for frame in frames)
        hi = max(frame[name].max() for frame in frames)
        pad = 0.02 * (hi - lo)
        limits.append((lo - pad, hi + pad))
    return limits


def draw_panel(ax, frame, x_name, y_name, colours, title):
    feasible = frame[frame.merit >= 0.0]
    infeasible = frame[frame.merit < 0.0]

    # Filled circles for both, told apart by colour alone, no legend: the
    # caption names the classes. Infeasible underneath -- it is the more
    # numerous class and it is context, not the answer.
    ax.scatter(infeasible[x_name], infeasible[y_name],
               c=colours["infeasible"], marker="o", s=20, linewidths=0.5,
               edgecolors=colours["surface"], zorder=2)
    ax.scatter(feasible[x_name], feasible[y_name],
               c=colours["feasible"], marker="o", s=28, linewidths=0.6,
               edgecolors=colours["surface"], zorder=3)

    style_axes(ax, colours, x_name, y_name)
    ax.set_xlabel(axis_label_rescaled(x_name), color=colours["ink2"],
                  fontsize=10)
    ax.set_ylabel(axis_label_rescaled(y_name), color=colours["ink2"],
                  fontsize=10)
    ax.set_title(title, color=colours["ink"], fontsize=12, loc="center",
                 pad=10)
    return len(feasible), len(infeasible)


def summarise(name, frame, x_name, y_name):
    """The numbers behind the panel, for the caption."""
    feasible = frame[frame.merit >= 0.0]
    print(f"\n{name}")
    print(f"  points           : {len(frame)}")
    print(f"  feasible         : {len(feasible)}")
    print(f"  infeasible       : {len(frame) - len(feasible)}")
    if feasible.empty:
        print("  NO FEASIBLE POINTS -- check the merit convention")
        return
    for column in (x_name, y_name):
        print(f"  feasible {column:<9}: {feasible[column].min():9.2f} .. "
              f"{feasible[column].max():9.2f}")

    # Not a region volume. Nested sampling concentrates its points inside
    # the region, so this fraction says how the RUN spent its evaluations,
    # not how much of the box is feasible -- and the two samplers spend
    # them differently, which is why it must not be read as an area.
    print(f"  feasible share of sampled points: "
          f"{len(feasible) / len(frame):.1%}  (NOT the region's area)")


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    dark = "--dark" in sys.argv
    show = "--show" in sys.argv

    out_path = None
    if "--out" in sys.argv:
        index = sys.argv.index("--out")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--out needs a filename")
        out_path = sys.argv[index + 1]
        positional = [a for a in positional if a != out_path]

    paths = [positional[0] if len(positional) > 0 else DEFAULT_PROPOSED,
             positional[1] if len(positional) > 1 else DEFAULT_MAGNUS]

    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit(
                f"no such file: {path}\n"
                f"Pass the two CSVs explicitly:\n"
                f"    python compare_feasible_spaces.py <proposed> <magnus>")

    loaded = [load(path) for path in paths]
    frames = [item[0] for item in loaded]
    x_name, y_name = loaded[0][1], loaded[0][2]

    # The panels only mean the same thing if the columns do. Different
    # design variables between the two files would still plot, silently,
    # against axes labelled after the first one.
    for path, (_, x_other, y_other) in zip(paths[1:], loaded[1:]):
        if (x_other, y_other) != (x_name, y_name):
            raise SystemExit(
                f"design variables differ: {paths[0]} has "
                f"({x_name}, {y_name}) but {path} has ({x_other}, {y_other})")

    for path, frame in zip(paths, frames):
        summarise(os.path.basename(path), frame, x_name, y_name)

    colours = DARK if dark else LIGHT
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(colours["surface"])

    for ax, frame, title in zip(axes, frames, PANEL_TITLES):
        draw_panel(ax, frame, x_name, y_name, colours, title)

    (x_lo, x_hi), (y_lo, y_hi) = shared_limits(frames, x_name, y_name)
    for ax in axes:
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

    # Each panel is read as a figure in its own right, so both carry their
    # own axis titles and their own tick labels. sharey is kept for the
    # LIMITS -- that is the part the comparison depends on -- and the tick
    # labels it would suppress on the right are put back explicitly.
    for ax in axes:
        ax.tick_params(labelleft=True, labelbottom=True)

    # tight_layout first for the OUTER margins, then wspace on top of it for
    # the gap BETWEEN the panels. Neither alone does the job: tight_layout
    # discards a wspace set on the gridspec, and its own w_pad pads every
    # edge, widening the outer margins along with the middle. The gap has to
    # be wide enough that the right panel's y-axis title and tick labels read
    # as belonging to it rather than to the panel on its left.
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.22)

    suffix = "_dark" if dark else ""
    if out_path is None:
        out_path = os.path.join(HERE,
                                f"feasible_space_comparison{suffix}.png")
    fig.savefig(out_path, dpi=200, facecolor=colours["surface"])
    print(f"\nsaved: {out_path}")

    # The same two panels as standalone files, for a layout that puts them
    # in separate float environments. Drawn on the SAME limits as the pair
    # above -- that is the whole point, and it is the thing that would go
    # wrong if these were produced by two independent runs of the
    # single-run plotter.
    if "--separate" in sys.argv:
        stem = os.path.splitext(out_path)[0]
        for index, (frame, title) in enumerate(zip(frames, PANEL_TITLES)):
            one, ax = plt.subplots(figsize=(6.4, 5.3))
            one.patch.set_facecolor(colours["surface"])
            draw_panel(ax, frame, x_name, y_name, colours, title)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            one.tight_layout()
            path = f"{stem}_{'ab'[index]}.png"
            one.savefig(path, dpi=200, facecolor=colours["surface"])
            print(f"saved: {path}")
            plt.close(one)

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
