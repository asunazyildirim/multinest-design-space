"""Plot the design space from kinetics_design_space.csv.

    python plot_design_space.py [csv] [--dark] [--show]

Two panels over the same (T, P) axes:

  left    every point the run kept, by role -- live, dead, rejected. This
          is the one that shows where the boundary is: live points alone
          are the inside of the region with no edge, because what defines
          the edge is the rejected points sitting just outside it.
  right   the same points coloured by merit. Under VaR merit is
          -(worst constraint violation), so it reads as margin: positive
          inside, negative outside, zero on the boundary.

and a second file, ``<stem>_feasible.png``:

  one panel, feasible against infeasible, split on merit rather than on
  role. That is the constraint verdict on its own, without the sampler's
  bookkeeping -- the figure to put next to a feasibility statement.

Colours are the reference categorical slots 1-3, which are the three that
clear the all-pairs colour-vision gate a scatter needs. Marker shape
repeats the role, so identity never rests on colour alone.
"""

import os
import sys

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "tp_design_space_serial.csv")

# Role -> (colour, marker, z-order, size). Live sits on top: it is the
# answer. Rejected goes underneath and is usually the most numerous.
LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "live": "#2a78d6", "dead": "#eb6834", "rejected": "#1baf7a",
    "neutral": "#f0efec",
    "feasible": "#1baf7a", "infeasible": "#d6453d",
}
DARK = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "live": "#3987e5", "dead": "#d95926", "rejected": "#199e70",
    "neutral": "#383835",
    "feasible": "#199e70", "infeasible": "#e3564c",
}

# Sequential ramp for magnitude: one hue, light to dark. These are the
# reference blue steps 150 -> 700; the lightest is allowed to recede
# towards the surface because this encodes a continuous quantity, not a
# set of discrete ordered marks.
BLUE_RAMP = LinearSegmentedColormap.from_list("ramp_blue", [
    "#b7d3f6", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])

ROLE_STYLE = {          # role: (marker, size, zorder)
    "rejected": ("x", 26, 2),
    "dead":     ("^", 22, 3),
    "live":     ("o", 34, 4),
}

# Units for the axis labels. The CSV carries the design-variable NAMES only
# -- they are whatever DESIGN_NAMES was set to -- so the units live here,
# keyed by name. Pressure is labelled kPa because that is what the column
# holds; the analysis figures divide by 100 and say bar, this one does not
# convert. A name that is not listed gets no bracket rather than a guessed
# one: a wrong unit on an axis is worse than none.
UNITS = {
    "temperature":      "°C",
    "pressure":         "kPa",
    "volume":           "m$^3$",
    "ratio":            "-",
    "recycle_fraction": "-",
}


def axis_label(name):
    """``temperature`` -> ``Temperature [°C]``."""
    unit = UNITS.get(name)
    pretty = name.replace("_", " ").capitalize()
    return f"{pretty} [{unit}]" if unit else pretty


def load(path):
    frame = pd.read_csv(path)
    if "role" not in frame.columns:
        raise SystemExit(f"{path} has no 'role' column -- is it the CSV "
                         f"run_sampler_kinetics.py writes?")
    x_name, y_name = frame.columns[0], frame.columns[1]
    return frame, x_name, y_name


def summarise(frame):
    """The table view: what the picture says, in numbers."""
    print(f"\n{'role':<10}{'points':>8}{'merit min':>12}{'merit max':>12}")
    print("-" * 42)
    for role in ("live", "dead", "rejected"):
        part = frame[frame.role == role]
        if part.empty:
            continue
        print(f"{role:<10}{len(part):>8}{part.merit.min():>12.3f}"
              f"{part.merit.max():>12.3f}")
    print("-" * 42)
    print(f"{'total':<10}{len(frame):>8}")

    live = frame[frame.role == "live"]
    if not live.empty:
        print(f"\ncertified region: {len(live)} points, "
              f"tightest margin {live.merit.min():.3f}")
        modes = sorted(live["mode"].unique())
        if len(modes) > 1:
            print(f"split into {len(modes)} modes: "
                  + ", ".join(f"{m} ({(live['mode'] == m).sum()})"
                              for m in modes))


def style_axes(ax, colours, x_name, y_name):
    ax.set_facecolor(colours["surface"])
    ax.grid(True, color=colours["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(colours["axis"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=colours["muted"], labelsize=9)
    ax.set_xlabel(axis_label(x_name), color=colours["ink2"], fontsize=10)
    ax.set_ylabel(axis_label(y_name), color=colours["ink2"], fontsize=10)


def plot(frame, x_name, y_name, colours, out_path, show):
    fig, (ax_role, ax_merit) = plt.subplots(
        1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(colours["surface"])

    # ---- left: by role ------------------------------------------------
    for role, (marker, size, zorder) in ROLE_STYLE.items():
        part = frame[frame.role == role]
        if part.empty:
            continue
        # A surface-coloured ring separates marks that overlap. 'x' is
        # unfilled, so it has no ring to give -- matplotlib would warn and
        # ignore the edgecolor.
        ring = {} if marker == "x" else {"edgecolors": colours["surface"]}
        ax_role.scatter(
            part[x_name], part[y_name],
            c=colours[role], marker=marker, s=size, zorder=zorder,
            linewidths=0.9 if marker == "x" else 0.6,
            label=f"{role} ({len(part)})", **ring)

    ax_role.set_title("Points kept, by role", color=colours["ink"],
                      fontsize=11, loc="left", pad=10)
    legend = ax_role.legend(loc="best", frameon=True, fontsize=9,
                            facecolor=colours["surface"],
                            edgecolor=colours["axis"])
    for text in legend.get_texts():
        text.set_color(colours["ink2"])

    # ---- right: margin inside the certified region --------------------
    # Live points only, on their own scale. A diverging scale over ALL the
    # points does not work here: merit runs to about -2000 outside and
    # 0.6-2.2 inside, so the whole feasible range collapses into one shade
    # and the panel says nothing the left one has not already said.
    # Magnitude within one sign is a sequential job -- one hue, light to
    # dark. The rejected and dead points stay as faint context so the
    # region still has an edge to sit against.
    other = frame[frame.role != "live"]
    ax_merit.scatter(other[x_name], other[y_name], c=colours["axis"],
                     marker=".", s=14, zorder=2, label="infeasible")

    live = frame[frame.role == "live"]
    scatter = ax_merit.scatter(
        live[x_name], live[y_name], c=live.merit, cmap=BLUE_RAMP,
        s=38, linewidths=0.6, edgecolors=colours["surface"], zorder=3)

    bar = fig.colorbar(scatter, ax=ax_merit, pad=0.02)
    bar.set_label("merit  =  -(worst constraint violation)",
                  color=colours["ink2"], fontsize=9)
    bar.ax.tick_params(colors=colours["muted"], labelsize=8)
    bar.outline.set_edgecolor(colours["axis"])

    ax_merit.set_title(
        f"Margin inside the certified region  "
        f"({live.merit.min():.2f} to {live.merit.max():.2f})",
        color=colours["ink"], fontsize=11, loc="left", pad=10)

    for ax in (ax_role, ax_merit):
        style_axes(ax, colours, x_name, y_name)

    fig.suptitle("Design space characterisation", color=colours["ink"],
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, facecolor=colours["surface"])
    print(f"\nsaved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_feasibility(frame, x_name, y_name, colours, out_path, show):
    """One panel, two classes: feasible and infeasible.

    Split on MERIT, not on role. Role records what the sampler did with a
    point -- kept it, evicted it, rejected it -- and eviction is a
    bookkeeping event, not a verdict: a dead point can be perfectly
    feasible and merely have been the worst of the population at the time.
    Under VaR merit is -(worst constraint violation), so merit >= 0 is the
    constraint verdict itself and that is what this figure shows.

    (Under FEASCRIT = "P" merit is the probability instead and the
    threshold is alpha*, not 0. The runs this reads use VaR. The counts are
    printed so the split can be checked against the live-point count rather
    than taken on trust.)
    """
    feasible = frame[frame.merit >= 0.0]
    infeasible = frame[frame.merit < 0.0]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    fig.patch.set_facecolor(colours["surface"])

    # Both classes as filled circles, told apart by colour alone -- red for
    # infeasible, green for feasible -- and with no legend and no counts on
    # the figure. Asked for in that form for the paper, where the caption
    # carries what a legend would have said.
    #
    # Red against green is the one categorical pair that deuteranopia and
    # protanopia collapse, and with the marker shapes now identical there is
    # no second channel left to fall back on. It reads correctly for most
    # viewers and in colour print; it is the caption that has to name which
    # region is which for the rest.
    ax.scatter(infeasible[x_name], infeasible[y_name],
               c=colours["infeasible"], marker="o", s=26, linewidths=0.5,
               edgecolors=colours["surface"], zorder=2)

    ax.scatter(feasible[x_name], feasible[y_name],
               c=colours["feasible"], marker="o", s=34, linewidths=0.6,
               edgecolors=colours["surface"], zorder=3)

    style_axes(ax, colours, x_name, y_name)
    ax.set_title("Feasible and infeasible design points",
                 color=colours["ink"], fontsize=11, loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=colours["surface"])
    print(f"saved: {out_path}")

    # Printed so the classification can be checked: every live point should
    # be feasible. A mismatch means the run stopped before certifying them
    # all, which the sampler reports as n_uncertified_live.
    live_infeasible = int((frame[frame.role == "live"].merit < 0.0).sum())
    print(f"  feasible {len(feasible)} / infeasible {len(infeasible)}"
          + (f"   [{live_infeasible} live point(s) still violating -- "
             f"the run did not certify them]" if live_infeasible else ""))

    if show:
        plt.show()
    plt.close(fig)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_path = args[0] if args else DEFAULT_CSV
    dark = "--dark" in sys.argv
    show = "--show" in sys.argv

    if not os.path.isfile(csv_path):
        raise SystemExit(f"no such file: {csv_path}\n"
                         f"run run_sampler_kinetics.py first")

    frame, x_name, y_name = load(csv_path)
    summarise(frame)

    colours = DARK if dark else LIGHT
    stem = os.path.splitext(csv_path)[0]
    suffix = "_dark" if dark else ""
    plot(frame, x_name, y_name, colours, f"{stem}{suffix}.png", show)
    plot_feasibility(frame, x_name, y_name, colours,
                     f"{stem}_feasible{suffix}.png", show)


if __name__ == "__main__":
    main()
