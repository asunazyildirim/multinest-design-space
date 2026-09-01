"""Watch a long sampler run: a snapshot PNG that rewrites itself as it goes.

    from live_monitor import LiveMonitor

    monitor = LiveMonitor(out_png=..., design_names=DESIGN_NAMES,
                          bounds=DESIGN_BOUNDS)
    result = sampler.run(..., frame_callback=monitor)
    monitor.close()

The sampler already emits a read-only snapshot of its state on every
event; this just draws one every EVERY_SECONDS and overwrites the same
file, so an image viewer left open on it acts as a live display. Nothing
extra is computed -- no HYSYS solves, no background grid -- so it costs
essentially nothing on a run whose cost is the flowsheet.

RunPlayer, the module's interactive replay, needs a GUI backend, and the
driver runs under Agg because a long unattended run should not depend on
a window. So the frames are also kept and pickled, and the player or a GIF
can be opened afterwards from another session:

    import pickle
    from multinest_sampler import RunPlayer, save_run_gif
    frames = pickle.load(open("run_frames.pkl", "rb"))
    save_run_gif(frames, "run.gif", fps=8)      # headless
    RunPlayer(frames).show()                    # needs a display

By default only the events that change the population are kept -- one
frame per rejected candidate would multiply the file size for very little
to look at. Pass keep=None to keep everything.
"""

import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "feasible": "#2a78d6", "uncertified": "#eb6834", "context": "#c3c2b7",
}

KEEP_DEFAULT = ("init", "accepted", "refit", "mode_split", "mode_frozen",
                "converged")


class LiveMonitor:
    """Frame callback that keeps frames and redraws a snapshot on a timer.

    Parameters
    ----------
    out_png : str
        Snapshot path, overwritten in place.
    design_names, bounds
        Axis labels and limits; the sampler's own DesignSpace is used when
        these are left out.
    every_seconds : float
        Minimum gap between redraws. The drawing itself takes a fraction
        of a second, but there is no point redrawing faster than anyone
        looks.
    keep : tuple of str, or None
        Which frame events to retain for the replay file. None keeps all.
    frames_path : str, optional
        Where close() pickles the kept frames. Defaults to out_png with a
        .pkl suffix.
    history_csv : str, optional
        Per-sweep progress, for plotting convergence afterwards.
    """

    def __init__(self, out_png, design_names=None, bounds=None,
                 every_seconds=60.0, keep=KEEP_DEFAULT, frames_path=None,
                 history_csv=None):
        self.out_png       = out_png
        self.design_names  = design_names
        self.bounds        = bounds
        self.every_seconds = every_seconds
        self.keep          = keep
        self.frames_path   = frames_path or (os.path.splitext(out_png)[0]
                                             + "_frames.pkl")
        self.history_csv   = history_csv or (os.path.splitext(out_png)[0]
                                             + "_history.csv")

        self.frames    = []
        self.history   = []
        self.started   = time.perf_counter()
        self._last_draw = 0.0
        self._n_seen   = 0

    # ------------------------------------------------------------------
    def __call__(self, frame):
        self._n_seen += 1
        if self.keep is None or frame.event in self.keep:
            self.frames.append(frame)

        threshold = self._threshold(frame)
        self.history.append({
            "seconds":     time.perf_counter() - self.started,
            "sweep":       frame.sweep,
            "replacements": frame.n_replacements,
            "candidates":  frame.n_candidate_evals,
            "feasible":    int(np.sum(frame.live_merit >= threshold)),
            "n_live":      int(frame.live_points.shape[0]),
            "worst_merit": float(np.min(frame.live_merit)),
            "modes":       len(frame.mode_ids),
        })

        now = time.perf_counter()
        if now - self._last_draw >= self.every_seconds or \
                frame.event == "converged":
            self._last_draw = now
            try:
                self.draw(frame)
            except Exception as exc:                    # noqa: BLE001
                # A drawing bug must never take down a run that has been
                # solving for hours.
                print(f"  [monitor] snapshot failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _threshold(frame):
        if frame.merit_threshold is not None:
            return frame.merit_threshold
        return frame.alpha_star if frame.feas_criterion == "P" else 0.0

    def _axes_setup(self, ax, frame):
        names = self.design_names or getattr(frame.design_space, "names", None)
        names = names or ["d1", "d2"]
        bounds = self.bounds or frame.design_space.bounds
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_xlabel(names[0], color=PALETTE["ink2"], fontsize=10)
        ax.set_ylabel(names[1], color=PALETTE["ink2"], fontsize=10)
        ax.set_facecolor(PALETTE["surface"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(PALETTE["axis"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=9)

    # ------------------------------------------------------------------
    def draw(self, frame):
        threshold = self._threshold(frame)
        elapsed = time.perf_counter() - self.started

        fig, (ax_map, ax_hist) = plt.subplots(
            1, 2, figsize=(12.5, 5.2),
            gridspec_kw={"width_ratios": [1.15, 1.0]})
        fig.patch.set_facecolor(PALETTE["surface"])

        # ---- left: where the population is now ------------------------
        dead = np.array(frame.dead_ref[:frame.n_dead]) \
            if frame.n_dead else np.empty((0, 2))
        rejected = np.array(frame.rejected_ref[:frame.n_rejected]) \
            if frame.n_rejected else np.empty((0, 2))
        for points in (dead, rejected):
            if points.size:
                ax_map.scatter(points[:, 0], points[:, 1], s=5,
                               c=PALETTE["context"], alpha=0.45, zorder=2)

        live = frame.live_points
        certified = frame.live_merit >= threshold
        if (~certified).any():
            ax_map.scatter(live[~certified, 0], live[~certified, 1],
                           c=PALETTE["uncertified"], marker="^", s=26,
                           zorder=3,
                           label=f"uncertified ({int((~certified).sum())})")
        if certified.any():
            ax_map.scatter(live[certified, 0], live[certified, 1],
                           c=PALETTE["feasible"], marker="o", s=26,
                           zorder=4,
                           label=f"certified ({int(certified.sum())})")

        legend = ax_map.legend(loc="lower left", fontsize=9, frameon=True,
                               facecolor=PALETTE["surface"],
                               edgecolor=PALETTE["axis"])
        for text in legend.get_texts():
            text.set_color(PALETTE["ink2"])
        self._axes_setup(ax_map, frame)
        ax_map.set_title("Live points", color=PALETTE["ink"], fontsize=11,
                         loc="left", pad=10)

        # ---- right: how it is converging ------------------------------
        seconds = [h["seconds"] / 60.0 for h in self.history]
        feas = [h["feasible"] for h in self.history]
        n_live = self.history[-1]["n_live"]

        ax_hist.plot(seconds, feas, color=PALETTE["feasible"], linewidth=2.0)
        ax_hist.axhline(n_live, color=PALETTE["axis"], linewidth=1.0,
                        linestyle="--")
        ax_hist.text(0.0, n_live, f"  all {n_live} certified = done",
                     color=PALETTE["muted"], fontsize=8, va="bottom")
        ax_hist.set_xlabel("minutes", color=PALETTE["ink2"], fontsize=10)
        ax_hist.set_ylabel("certified live points", color=PALETTE["ink2"],
                           fontsize=10)
        ax_hist.set_ylim(0, n_live * 1.08)
        ax_hist.set_facecolor(PALETTE["surface"])
        ax_hist.grid(True, color=PALETTE["grid"], linewidth=0.6)
        ax_hist.set_axisbelow(True)
        for spine in ax_hist.spines.values():
            spine.set_color(PALETTE["axis"])
        ax_hist.tick_params(colors=PALETTE["muted"], labelsize=9)
        ax_hist.set_title("Progress", color=PALETTE["ink"], fontsize=11,
                          loc="left", pad=10)

        left = int(np.sum(frame.live_merit < threshold))
        fig.suptitle(
            f"sweep {frame.sweep}   ·   {frame.n_replacements} replacements   "
            f"·   {frame.n_candidate_evals} candidates   ·   "
            f"{left} left   ·   {len(frame.mode_ids)} mode(s)   ·   "
            f"{elapsed / 60:.0f} min",
            color=PALETTE["ink"], fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        # Written to a temporary name and moved into place, so a viewer
        # never catches a half-written file. format= is explicit because
        # savefig otherwise infers it from the extension, and ".tmp" is
        # not a format.
        temporary = self.out_png + ".tmp"
        fig.savefig(temporary, format="png", dpi=140,
                    facecolor=PALETTE["surface"])
        plt.close(fig)
        os.replace(temporary, self.out_png)

    # ------------------------------------------------------------------
    def close(self):
        """Write the replay file and the history CSV."""
        if self.frames:
            with open(self.frames_path, "wb") as handle:
                pickle.dump(self.frames, handle)
            print(f"  [monitor] {len(self.frames)} frame(s) "
                  f"(of {self._n_seen} emitted) -> {self.frames_path}")

        if self.history:
            import csv
            with open(self.history_csv, "w", newline="",
                      encoding="utf-8") as handle:
                writer = csv.DictWriter(handle,
                                        fieldnames=list(self.history[0]))
                writer.writeheader()
                writer.writerows(self.history)
            print(f"  [monitor] history -> {self.history_csv}")
        print(f"  [monitor] snapshot -> {self.out_png}")
