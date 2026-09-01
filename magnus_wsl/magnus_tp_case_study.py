"""
The (T, P) methanol case study, ported to MAGNUS / NSFeas.

Companion to ``run_sampler_kinetics.py`` on the Windows side, in the same
sense that ``magnus_multidim_examples.py`` is the companion to
``multidim_examples.py``: same design box, same fixed inputs, same two
constraints, same live-point count, so the two samplers can be put side by
side on one problem.

    design    T in [210, 300] C,  P in [5000, 11000] kPa
    fixed     H2:CO2 = 3.0,  reactor volume 65 m3,  recycle fraction 0.99
    g1        carbon efficiency  >= 92 %    (== MeOH >= 19187 kg/h)
    g2        energy efficiency  >= 81.3 %
    theta     Rxn-8 forward/reverse activation energies, K = 1 (nominal)

At the thresholds above the certified region is a closed interior island
covering 24.6 % of the box on the 130-point reference sweep, touching none of
the four edges. See ``tp_case_study_proposal.md``.

THE MODEL IS NOT HERE
---------------------
It is HYSYS, and HYSYS is a Windows COM server. Nothing in WSL can drive it.
So ``--backend bridge`` (the default) does not solve anything: it writes the
design point into a spool directory on ``/mnt/c`` and waits for
``hysys_bridge_server.py``, running on the Windows side with the case open, to
write the answer back. That server must be started first.

``--backend surrogate`` replaces HYSYS with a bilinear interpolant of the
130-point reference sweep. It exists to check the wiring -- control mapping,
constraint signs, NSFeas options, output files -- in about a second instead of
half a day. It is nominal-theta only, because the sweep is, so it is a dry run
and not a result.

COST
----
Each HYSYS solve takes 20-30 s and this runs serially: one point at a time,
one instance. NSFeas on a 2-D problem of this difficulty spends roughly
NUMLIVE + 1300 evaluations (measured: 1652 at NUMLIVE = 500 on the surrogate,
against family A, D=2 of the ladder, whose certified fraction 23.1 % is the
closest match to this problem's 24.6 %), so at NUMLIVE = 500 expect around
1700 solves and 10-12 hours. The Python run on the same problem took 1254
solves and 2 h 58 m across four HYSYS instances.

Add about 0.4 s per point for the bridge itself -- measured over 624 round
trips, and dominated by how long WSL takes to notice a file Windows has
created rather than by anything tunable here. Against a 21 s solve that is
under 2 %, which is why this is a spool directory and not a socket: a socket
would need an inbound Windows Firewall rule and would save about ten minutes
on a ten-hour run.

(An earlier measurement said 1.4 s. That version had the client reading the
heartbeat on every 50 ms poll, which made the server's own heartbeat writes
fail and retry. Polling a shared file has a cost on both sides of it.)

Two things halve or better that if it is too long:

    --numlive 250     the seed is 500 of the ~1800, so this is a real saving
    --numprop 4       fewer proposals per contour, lower discard count

Every solve is written to disk as it happens (``--cache``), keyed on the exact
(T, P, Ea_f, Ea_r) doubles, and reloaded on the next start. A run that dies at
hour nine keeps everything it paid for.

Usage
-----
    # Windows, with methanol_instance0.hsc open:
    python hysys_bridge_server.py

    # WSL, in this directory:
    bash run_magnus_tp.sh --backend surrogate      # wiring check, ~1 s
    bash run_magnus_tp.sh                          # the real run

REPRODUCIBLE, BUT NOT BY A SEED. NSFeas exposes no RNG seed -- the options
are NUMLIVE, NUMPROP, MAXITER, MAXCPU, MAXERR, ELLCONF, ELLMAG, ELLRED,
FEASCRIT, FEASTHRES, LKH*, DISP* and nothing else -- yet it is deterministic:
the same options produce bit-identical design points, verified by restarting
an interrupted run and watching all 549 completed solves come back as cache
hits.

Two consequences, in opposite directions:

* Repeats are pointless. Running this five times gives five identical
  answers, so there is no sampler variance to average over and no median to
  quote. Where the Python side varies ``seed`` to get a spread, this cannot.
  Vary NUMLIVE instead if a sensitivity is wanted.
* A killed run resumes for free. See ``SolveCache``.
"""

import argparse
import atexit
import json
import os
import sys
import time

import numpy as np
import pymc
from magnus import NSFeas


HERE = os.path.dirname(os.path.abspath(__file__))

# The repository root, one level up from magnus_wsl/, as WSL sees it. Only
# used to find the reference sweep for --backend surrogate; the bridge needs
# nothing from here.
WIN_REPO = os.path.dirname(HERE)
DEFAULT_SWEEP = os.path.join(
    WIN_REPO, "output", "sensitivity_results_20260731_174218.csv")


# ----------------------------------------------------------------------
# 1. PROTOCOL -- must stay identical to run_sampler_kinetics.py section 1
# ----------------------------------------------------------------------
DESIGN_NAMES = ["temperature", "pressure"]
DESIGN_BOUNDS = [
    (210.0, 300.0),        # temperature [C]    -- Cu/ZnO/Al2O3 window
    (5000.0, 11000.0),     # pressure    [kPa]  -- 50 to 110 bar
]

FIXED_KWARGS = {
    "ratio": 3.0,
    "volume": 65.0,
    "recycle_fraction": 0.99,
}

# Read row by row against LOWER_LIMITS and must stay in step. Two rows, not
# three: the CO2 fresh feed is fixed over this box, so methanol production and
# carbon efficiency are the same function up to a constant (measured spread
# 296 ppm over the sweep) and a production target IS a carbon-efficiency
# target.
OUTPUT_KEYS = ["C_efficiency_pct", "energy_efficiency_pct"]
LOWER_LIMITS = [92.0, 81.3]        # g_i = LOWER_i - y_i <= 0 when feasible

# theta = (Rxn-8 forward Ea, reverse Ea), the RWGS step of vandal2013, in
# HYSYS internal energy units. K = 1: a deterministic characterisation run
# through the stochastic machinery, matching the Python baseline. Add rows
# here to make it stochastic -- NSFeas takes the whole list as ONE fixed
# scenario set, which is the caveat recorded in magnus_multidim_examples.py.
EA_FORWARD_NOMINAL = 98089.87724
EA_REVERSE_NOMINAL = 58392.0
THETA_SAMPLES = np.array([[EA_FORWARD_NOMINAL, EA_REVERSE_NOMINAL]])

# NSFeas knobs, carried over unchanged from the dimension ladder so this cell
# is directly comparable with family A at D = 2.
COMMON_NS = dict(DISPLEVEL=1, NUMPROP=8, MAXITER=0,
                 ELLCONF=0.99, ELLMAG=0.30, ELLRED=0.20)

# Matches run_sampler_kinetics.py, which runs at alpha_star = 1.
#
# At K = 1 the choice is immaterial -- every threshold below 1 selects the
# same single scenario -- so this could equally be the ladder's 0.95. It is 1
# because that is what the Python side uses, and because the moment
# THETA_SAMPLES grows the two stop agreeing: alpha* = 1 puts FEASTHRES at 0,
# which is VaR at the worst case rather than at a quantile of it.
ALPHA_STAR = 1.0

# Reference numbers from the 130-point sweep, for the report only.
#
# Two different things, and mixing them up makes a correct run look wrong.
# The GRID figures are cell counts: 32 of the 130 sweep points satisfy both
# constraints, and those cells span T 240-270 C, P 55-105 bar. The CONTINUOUS
# figures are the region the same data implies between the nodes, measured by
# evaluating the bilinear interpolant on a 901 x 601 grid. A correct run's
# live set matches the CONTINUOUS extent -- it should overshoot the cell
# centres, because the boundary lies between them.
V_REF_GRID = 32 / 130                       # 24.6 %, the cell count
V_REF_CONTINUOUS = 0.2607                   # 26.1 %, the interpolated area
REGION_T = (240.0, 270.0)                   # feasible cell centres
REGION_P = (5500.0, 10500.0)
REGION_T_CONT = (239.5, 280.0)              # interpolated region
REGION_P_CONT = (5170.0, 10690.0)


class _Tee:
    """Mirror one stream to a file, writing through on every call.

    Flushed per write rather than per buffer. This run is ten hours long
    and the two things most worth having a record of -- the progress lines
    while it goes, and the traceback if it dies -- are exactly what a
    buffered log loses. Same reason the Windows side does it this way.
    """

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        if not self._handle.closed:
            self._handle.write(text)
            self._handle.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        if not self._handle.closed:
            self._handle.flush()

    def isatty(self):
        return self._stream.isatty()


def start_console_log(path):
    """Mirror stdout and stderr to ``path`` for the rest of the process."""
    handle = open(path, "w", encoding="utf-8")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)

    def close():
        # Put the real streams back BEFORE closing the file. Python flushes
        # sys.stdout after the atexit handlers have run; if that flush lands
        # on a tee whose file is already shut, the interpreter exits 120
        # with "Exception ignored in sys.unraisablehook" -- a clean run
        # would look like a failed one.
        sys.stdout, sys.stderr = real_stdout, real_stderr
        handle.close()

    atexit.register(close)


def to_physical(u):
    """Map a control point in [-1, 1]^2 onto (T [C], P [kPa]).

    NSFeas searches the control box directly and fits ONE ellipsoid over the
    live set. On the physical box the two axes differ by a factor of 70 in
    magnitude, and ELLMAG / ELLRED were tuned on the ladder's [-1, 1]^D cells,
    so the search runs on the unit box and the physical units appear only
    inside the model callback.
    """
    out = []
    for value, (lo, hi) in zip(u, DESIGN_BOUNDS):
        out.append(lo + (hi - lo) * 0.5 * (float(value) + 1.0))
    return out


# ----------------------------------------------------------------------
# 2. BACKENDS -- what actually produces (C_eff, E_eff)
# ----------------------------------------------------------------------
class BridgeBackend:
    """One HYSYS solve per call, over a spool directory on /mnt/c.

    The protocol is two files and a rename, described in
    ``hysys_bridge_server.py``. Renaming into place is what makes it safe: a
    ``res_<id>.json`` that exists is complete, so the client never has to
    guess whether a short read means "still writing" or "empty answer".
    """

    HEARTBEAT = "server.alive"

    # The server refreshes its heartbeat immediately before each solve, so
    # "stale" measures how long the current solve has been running, not how
    # long since the last one finished. 300 s is well past the 20-30 s a real
    # solve takes and well short of the 900 s per-request timeout, so a hung
    # HYSYS is reported as a hang rather than waited out in silence.
    def __init__(self, spool, timeout=900.0, poll=0.05, stale=300.0,
                 check_every=5.0):
        self.spool = spool
        self.timeout = timeout
        self.poll = poll
        self.stale = stale
        self.check_every = check_every
        self._next_id = 1

        # Request ids carry a per-run prefix so two clients on one spool
        # cannot collide. Numbering from 1 in both would have each of them
        # reading the other's answers -- the right JSON for the wrong design
        # point, with nothing anywhere to say so. A ten-hour run is exactly
        # the kind someone restarts while the first is still going.
        self._prefix = f"{os.getpid()}-{int(time.time()) % 100000}"

        if not os.path.isdir(spool):
            raise SystemExit(
                f"spool directory does not exist: {spool}\n"
                f"Start hysys_bridge_server.py on the Windows side first -- "
                f"it creates it.")

        beat = self._heartbeat_age()
        if beat is None:
            raise SystemExit(
                f"no {self.HEARTBEAT} in {spool}\n"
                f"The Windows-side server is not running. Start it with:\n"
                f"    python hysys_bridge_server.py")
        if beat > stale:
            raise SystemExit(
                f"{self.HEARTBEAT} is {beat:.0f} s old -- the server has "
                f"stopped. Restart it before running this.")

        # Tidy away abandoned files, but only ones old enough that no live
        # run could still want them. Deleting every req_/res_ outright would
        # take a concurrent client's in-flight request with it -- which the
        # per-run prefix above exists precisely to make safe.
        now = time.time()
        left = 0
        for name in os.listdir(spool):
            if not name.startswith(("req_", "res_")):
                continue
            path = os.path.join(spool, name)
            try:
                if now - os.path.getmtime(path) > 3600.0:
                    os.remove(path)
                    left += 1
            except OSError:
                pass
        if left:
            print(f"  cleared {left} abandoned spool file(s) over an hour old")

    def _heartbeat_age(self):
        path = os.path.join(self.spool, self.HEARTBEAT)
        try:
            with open(path, encoding="utf-8") as handle:
                return time.time() - float(json.load(handle)["t"])
        except (OSError, ValueError, KeyError):
            return None

    def __call__(self, T, P, ea_forward, ea_reverse, attempts=3):
        """Solve one point. Returns ``(values, "")`` or ``(None, reason)``.

        ``None`` means the PLANT is infeasible there -- an unconverged solve,
        which the sampler treats as a point outside the region. It must never
        mean the TRANSPORT failed. A request that the server could not read is
        re-sent under a new id instead, and if it fails every time this raises
        rather than reporting a design point as infeasible on the strength of
        a filesystem hiccup.
        """
        for attempt in range(attempts):
            values, error, retryable = self._once(T, P, ea_forward, ea_reverse)
            if not retryable:
                return values, error
            print(f"    [bridge] attempt {attempt + 1}/{attempts} at "
                  f"T={T:.2f} P={P:.0f}: {error}")
            sys.stdout.flush()
            time.sleep(1.0)

        raise RuntimeError(
            f"the bridge failed {attempts} times at T={T:g}, P={P:g}: "
            f"{error}\nThis is a transport or configuration failure, not an "
            f"infeasible point, so the run stops rather than certify a region "
            f"around it. Every completed solve is in the cache file.")

    def _once(self, T, P, ea_forward, ea_reverse):
        req_id = f"{self._prefix}_{self._next_id}"
        self._next_id += 1

        request = {
            "id": req_id,
            "kwargs": dict(FIXED_KWARGS,
                           temperature=T, pressure=P,
                           rxn8_ea_forward=ea_forward,
                           rxn8_ea_reverse=ea_reverse),
            "outputs": OUTPUT_KEYS,
        }
        req_path = os.path.join(self.spool, f"req_{req_id}.json")
        tmp = req_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(request, handle)
            handle.flush()
            os.fsync(handle.fileno())

        # Retried for the same reason the server retries: the target lives on
        # NTFS, and a Windows process that opens a file the instant it appears
        # -- Defender, the indexer -- makes the rename fail with a sharing
        # violation. It surfaces here as PermissionError even though this side
        # is Linux, because the failure happens in the Windows filesystem.
        for attempt in range(40):
            try:
                os.replace(tmp, req_path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.25)

        res_name = f"res_{req_id}.json"
        res_path = os.path.join(self.spool, res_name)
        deadline = time.time() + self.timeout
        gone_since = None
        last_check = time.time()

        while True:
            # os.listdir, NOT os.path.exists. Measured on this machine: a
            # path polled only with os.path.exists never turned True in 20 s
            # after Windows created the file, because WSL caches the negative
            # dentry and nothing in that loop invalidates it. A readdir does.
            # os.path.exists is 3.9 ms against listdir's 6.4 ms, so the
            # cheaper call would buy 2.5 ms and lose the answer.
            if res_name in os.listdir(self.spool):
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"no answer to request {req_id} in {self.timeout:.0f} s "
                    f"(T={T:g}, P={P:g})")

            # Not on every poll. The server has to overwrite this file while
            # we read it, and on Windows an open handle here is enough to
            # make its write fail -- so the cheapest way to keep the server
            # alive is to look at its heartbeat rarely. Every few seconds is
            # ample against a threshold measured in minutes.
            now = time.time()
            if now - last_check < self.check_every:
                time.sleep(self.poll)
                continue
            last_check = now

            # Two ways for the server to stop, and they look different from
            # here. Ctrl-C removes the heartbeat on the way out, so it goes
            # MISSING; a crash or a hung solve leaves the last one in place,
            # so it goes STALE. Only checking the second is what made an
            # interrupted server hang this side for the full timeout.
            age = self._heartbeat_age()
            if age is None:
                gone_since = gone_since or now
                if now - gone_since > self.stale:
                    raise RuntimeError(
                        f"the Windows-side server's heartbeat disappeared "
                        f"{now - gone_since:.0f} s ago -- it was stopped "
                        f"while request {req_id} was outstanding "
                        f"(T={T:g}, P={P:g})")
            else:
                gone_since = None
                if age > self.stale:
                    raise RuntimeError(
                        f"Windows-side server went quiet {age:.0f} s ago "
                        f"while request {req_id} was outstanding "
                        f"(T={T:g}, P={P:g})")
            time.sleep(self.poll)

        with open(res_path, encoding="utf-8") as handle:
            response = json.load(handle)
        os.remove(res_path)

        if not response["ok"]:
            return None, response["error"], bool(response.get("retryable"))
        return [response["outputs"][key] for key in OUTPUT_KEYS], "", False


class SurrogateBackend:
    """Bilinear interpolation of the reference sweep. Nominal theta only.

    A dry run, not a result: the sweep was measured at the nominal activation
    energies, so this backend ignores theta entirely. Its value is that it
    reproduces the RIGHT REGION -- the same 24.6 % island -- at zero cost, so
    a wiring mistake shows up as a wrong picture in one second rather than
    after ten hours of solving.
    """

    def __init__(self, path):
        rows = []
        with open(path, encoding="utf-8-sig") as handle:
            import csv
            for row in csv.DictReader(handle):
                if row["converged"] != "True":
                    continue
                rows.append((float(row["temperature_set_C"]),
                             float(row["pressure_set_kPa"]),
                             [float(row[key]) for key in OUTPUT_KEYS]))
        if not rows:
            raise SystemExit(f"no converged rows in {path}")

        self.T = np.array(sorted({r[0] for r in rows}))
        self.P = np.array(sorted({r[1] for r in rows}))
        self.Z = np.full((self.T.size, self.P.size, len(OUTPUT_KEYS)), np.nan)
        for t, p, values in rows:
            self.Z[int(np.searchsorted(self.T, t)),
                   int(np.searchsorted(self.P, p))] = values
        if np.isnan(self.Z).any():
            raise SystemExit(
                f"{path} is not a full factorial -- the surrogate needs one")
        self.n = 0

    def __call__(self, T, P, ea_forward, ea_reverse):
        self.n += 1
        out = []
        for axis, grid, value in ((0, self.T, T), (1, self.P, P)):
            index = int(np.clip(np.searchsorted(grid, value) - 1,
                                0, grid.size - 2))
            span = grid[index + 1] - grid[index]
            out.append((index, (value - grid[index]) / span))
        (i, u), (j, v) = out
        block = self.Z[i:i + 2, j:j + 2, :]
        weights = np.array([[(1 - u) * (1 - v), (1 - u) * v],
                            [u * (1 - v), u * v]])
        return list((weights[:, :, None] * block).sum(axis=(0, 1))), ""


# ----------------------------------------------------------------------
# 3. CACHE
# ----------------------------------------------------------------------
class SolveCache:
    """Memoise solves on the exact input doubles, and mirror them to disk.

    A correctly built two-dependent FFCustom fires the callback once per
    design point (measured: model calls = NS.stats.numfct + 1), so the memo
    is insurance rather than arithmetic -- it costs nothing and it caps the
    damage if a future NSFeas or pyMC evaluates a node twice.

    The disk mirror is what earns its place, and it earns more than expected:
    it makes a killed run RESUMABLE.

    NSFeas offers no seed, so this was written expecting the file to be a
    record and not a warm start. It turns out NSFeas is deterministic --
    given the same options it draws the same points, to the last bit. So a
    restart replays the identical sequence, every point up to where the last
    run died is a cache hit, and the run picks up exactly where it stopped.

    Measured: a run killed by a Windows Update reboot after 549 solves was
    restarted from this file. The 500 seed points came back in 0.00 s
    instead of three hours, and the first fresh solve was the very design
    point that had been in flight when the machine went down.

    The determinism holds only for identical settings. Change NUMLIVE or
    NUMPROP and the sequence changes, so the old rows stop matching -- they
    are then still real data on the (T, P) surface, just not a warm start.
    """

    FIELDS = ("backend", "temperature", "pressure", "ea_forward",
              "ea_reverse", "ok") + tuple(OUTPUT_KEYS)

    def __init__(self, path, backend_name, seed_calls=0, warm_start=True):
        self.path = path
        self.backend_name = backend_name
        self.memo = {}
        self.hits = 0
        self.misses = 0
        self.failures = 0

        # Timing. model_time is the seconds spent inside the backend --
        # HYSYS plus the bridge round trip -- and NOT the seconds spent in
        # NSFeas. Subtracting it from the run's wall clock is the only way
        # to see the sampler's own cost, which on this problem is under one
        # per cent of the total and would otherwise be invisible.
        self.calls = 0
        self.model_time = 0.0
        self.seed_calls = seed_calls
        self.seed_end_t = None       # perf_counter when the seed's last call ended
        self.seed_model_time = None
        self.seed_solves = 0
        self.warm_hits = 0           # hits served from the file, not from this run
        self.loaded_keys = set()     # what came off disk, so warm hits are visible

        if path and os.path.exists(path) and not warm_start:
            # Deliberately not loaded. Reusing a previous run's solves makes
            # the wall clock meaningless -- a cache hit costs microseconds
            # where the solve cost half a minute -- so a run being TIMED has
            # to start cold. Still appended to below, so this run remains
            # resumable even though it does not resume anything.
            print(f"  cache             : {path} exists but was NOT loaded "
                  f"(--no-warm-start); every point will be solved afresh")
        elif path and os.path.exists(path):
            import csv
            loaded, skipped = 0, 0
            with open(path, encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    # A row is only reusable by the model that produced it.
                    # Without this, the recommended sequence -- surrogate
                    # check first, real run second, both defaulting to this
                    # same file -- would let a bilinear interpolation of the
                    # old sweep be served to the real run as though HYSYS had
                    # returned it, with nothing on screen to say so. The
                    # collision needs identical doubles and is therefore rare,
                    # which is exactly what would make it hard to catch.
                    if row.get("backend") != backend_name:
                        skipped += 1
                        continue
                    key = tuple(row[name] for name in self.FIELDS[1:5])
                    if row["ok"] == "1":
                        self.memo[key] = ([float(row[k]) for k in OUTPUT_KEYS],
                                          "")
                    else:
                        self.memo[key] = (None, "cached failure")
                    self.loaded_keys.add(key)
                    loaded += 1
            print(f"  cache             : {loaded} {backend_name} solve(s) "
                  f"reloaded from {path}"
                  + (f", {skipped} from another model ignored"
                     if skipped else ""))
        elif path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(",".join(self.FIELDS) + "\n")

    @staticmethod
    def _key(*values):
        # %.17g round-trips a double exactly, so the in-memory key and the key
        # rebuilt from the file are the same string.
        return tuple(f"{v:.17g}" for v in values)

    def _mark_seed(self):
        """Record where the seed population ended.

        NSFeas evaluates the initial NUMLIVE live points before it proposes
        anything, so the first ``seed_calls`` callbacks ARE the seed. The
        two phases are worth separating because only the seed is a set of
        independent solves -- the replacement loop draws each candidate
        from an ellipsoid fitted to the population the previous one changed
        -- so only the seed could ever be distributed across instances.
        """
        if self.seed_calls and self.seed_end_t is None \
                and self.calls >= self.seed_calls:
            self.seed_end_t = time.perf_counter()
            self.seed_model_time = self.model_time
            self.seed_solves = self.misses

    def get(self, backend, T, P, ea_forward, ea_reverse):
        self.calls += 1
        key = self._key(T, P, ea_forward, ea_reverse)

        if key in self.memo:
            self.hits += 1
            if self.loaded_keys and key in self.loaded_keys:
                self.warm_hits += 1
            self._mark_seed()
            return self.memo[key]

        # Timed with try/finally: a solve that raises cost just as much
        # wall clock as one that returned, and losing it here would charge
        # the time to NSFeas instead of to the model.
        _t0 = time.perf_counter()
        try:
            values, error = backend(T, P, ea_forward, ea_reverse)
        finally:
            self.model_time += time.perf_counter() - _t0

        self.misses += 1
        if values is None:
            self.failures += 1
        self.memo[key] = (values, error)
        self._mark_seed()

        if self.path:
            cells = [self.backend_name] + list(key)
            cells += ["1" if values else "0"]
            cells += [f"{v:.10g}" for v in (values or [float("nan")] *
                                            len(OUTPUT_KEYS))]
            line = ",".join(cells) + "\n"

            # Retried, and never fatal. This file lives on NTFS, so anything
            # on the Windows side that opens it takes a lock -- and the
            # obvious thing to do with a CSV while a ten-hour run is going is
            # to open it in Excel and watch. An unprotected append would then
            # raise, and the run would die at hour six because someone looked
            # at it. Losing one line of the log is not worth that.
            for attempt in range(20):
                try:
                    with open(self.path, "a", encoding="utf-8") as handle:
                        handle.write(line)
                    break
                except OSError as exc:
                    if attempt == 19:
                        print(f"    [cache] could not append after 20 tries "
                              f"({exc}); continuing without logging this "
                              f"solve -- is the file open in Excel?")
                        sys.stdout.flush()
                        break
                    time.sleep(0.5)
        return values, error


# ----------------------------------------------------------------------
# 4. THE RUN
# ----------------------------------------------------------------------
def build_model(backend, cache, limits, progress_every=5):
    """The FFCustom callback: control point + theta -> the two violations."""
    state = {"start": time.time(), "last": 0}

    def model(x):
        T, P = to_physical(x[:2])
        ea_forward, ea_reverse = float(x[2]), float(x[3])

        before = cache.misses
        values, error = cache.get(backend, T, P, ea_forward, ea_reverse)

        if cache.misses != before and progress_every:
            done = cache.misses
            if done % progress_every == 0 or done == 1:
                elapsed = time.time() - state["start"]
                rate = elapsed / done
                print(f"    {done:>5} solves   {elapsed / 60:6.1f} min   "
                      f"{rate:5.1f} s/solve   "
                      f"last T={T:6.1f} P={P:7.0f}"
                      f"{'' if values else '   [FAILED]'}")
                sys.stdout.flush()

        if values is None:
            # Same convention as the Python side's on_failure="infeasible":
            # an unconverged solve is a point outside the region, not a hole
            # in it. Returning the limits themselves is the violation of a
            # zero-efficiency plant -- infeasible, finite, and bounded, so it
            # cannot distort the contour ordering the way a 1e30 would.
            return list(limits)

        return [limits[i] - values[i] for i in range(len(limits))]

    return model


def run(args):
    limits = [args.c_threshold, args.e_threshold]

    # Everything up to and including NS.setup() is initialisation: the bridge
    # handshake, the cache load, the DAG. Reported apart from the sampling
    # rather than folded into it, since a reference method solving the same
    # flowsheet pays its own version of the same fixed cost.
    t_init = time.perf_counter()

    if args.backend == "bridge":
        backend = BridgeBackend(args.spool, timeout=args.timeout)
        backend_label = f"HYSYS via {args.spool}"
    else:
        backend = SurrogateBackend(args.sweep)
        backend_label = f"bilinear surrogate of {os.path.basename(args.sweep)}"

    # The seed is NUMLIVE points, one callback per (point, scenario), so the
    # boundary moves with THETA_SAMPLES rather than being pinned at NUMLIVE.
    cache = SolveCache(args.cache, args.backend,
                       seed_calls=args.numlive * len(THETA_SAMPLES),
                       warm_start=not args.no_warm_start)

    DAG = pymc.FFGraph()
    DAG.options.MAXTHREAD = 1          # the callback drives one HYSYS instance
    d = DAG.add_vars(2, "d")
    p_forward = DAG.add_var("ea_f")
    p_reverse = DAG.add_var("ea_r")

    op = pymc.FFCustom()
    # Every fifth solve on the bridge, which is a line every two or three
    # minutes at 30 s a solve. The first version printed every 25th, chosen
    # when a test solve took 0.2 s; on the real model that is twelve minutes
    # of blank screen, and a run that is working looks indistinguishable from
    # one that has hung.
    op.set_D_eval(build_model(backend, cache, limits,
                              0 if args.backend == "surrogate" else 5))
    # Held in a list because FFVar has no __dict__ and cannot keep the
    # operation alive itself -- the same reason magnus_multidim_examples.py
    # keeps _CUSTOM_OPS.
    keep_alive = [op]
    inputs = list(d) + [p_forward, p_reverse]
    # op(ndep, inputs, uid) -- the THREE-argument overload, which returns one
    # FFVar per dependent. NOT op(inputs, k): that overload's second argument
    # is the operation's uid, not an output index, so it builds k separate
    # single-output operations that each return element 0 of the callback.
    # The DAG prints as Custom[uid][idep], and the difference between a
    # correct and a silently wrong graph is Custom[0][0], Custom[0][1] here
    # against Custom[0][0], Custom[1][0] there. The wrong form does not
    # raise: it enforces the first constraint twice and drops the second.
    constraints = op(len(limits), inputs, 0)

    NS = NSFeas()
    for key, value in dict(COMMON_NS, NUMLIVE=args.numlive,
                           NUMPROP=args.numprop).items():
        setattr(NS.options, key, value)
    NS.options.FEASCRIT = NS.options.VAR
    NS.options.FEASTHRES = 1.0 - args.alpha

    NS.set_dag(DAG)
    NS.set_constraint(constraints)
    NS.set_control(d, [-1.0, -1.0], [1.0, 1.0])
    NS.set_parameter([p_forward, p_reverse],
                     [[float(a), float(b)] for a, b in THETA_SAMPLES])

    print()
    print("=" * 78)
    print("(T, P) methanol case study on MAGNUS / NSFeas")
    print("=" * 78)
    print(f"  design            : "
          + ",  ".join(f"{n} in [{lo:g}, {hi:g}]"
                       for n, (lo, hi) in zip(DESIGN_NAMES, DESIGN_BOUNDS)))
    print(f"  fixed             : "
          + ", ".join(f"{k}={v:g}" for k, v in FIXED_KWARGS.items()))
    print(f"  constraints       : C_eff >= {limits[0]:g} %  "
          f"(MeOH >= {limits[0] * 208.54:.0f} kg/h),  "
          f"E_eff >= {limits[1]:g} %")
    print(f"  uncertainty       : K = {len(THETA_SAMPLES)} scenario(s), "
          f"Rxn-8 Ea forward/reverse")
    print(f"  criterion         : VaR, feasible <= 0, "
          f"FEASTHRES = {NS.options.FEASTHRES:g}  (alpha* = {args.alpha:g})")
    print(f"  N_L               : {args.numlive}   NUMPROP = {args.numprop}")
    print(f"  model             : {backend_label}")
    print(f"  reference         : {V_REF_GRID:.1%} of the box by cell count "
          f"on the 130-point sweep, {V_REF_CONTINUOUS:.1%} interpolated")
    print("=" * 78)
    sys.stdout.flush()

    NS.setup()
    init_time = time.perf_counter() - t_init

    # Timed around NS.sample() alone. cache.model_time is read at the same
    # two instants so the model/sampler split below covers exactly this
    # interval and nothing else.
    started = time.perf_counter()
    model_time_start = cache.model_time
    NS.sample()
    elapsed = time.perf_counter() - started
    solve_time = cache.model_time - model_time_start

    timing = {
        "init": init_time,
        "wall": elapsed,
        "solve": solve_time,
        "sampler": elapsed - solve_time,
        "t_start": started,
        "model_time_start": model_time_start,
    }
    return report_and_save(NS, args, limits, cache, timing, keep_alive)


# ----------------------------------------------------------------------
# 5. RESULTS
# ----------------------------------------------------------------------
def print_run_summary(NS, cache, timing, n_live, live_crit):
    """The timing block, in the same shape the Python side prints.

    The split is the point of it. Wall clock alone credits or blames the
    sampler for HYSYS; "in sampler" is what the method costs once the
    flowsheet is charged separately, and it is the term that stays flat
    while the solve cost grows with the model.
    """
    wall, solve = timing["wall"], timing["solve"]
    share = (lambda v: 100.0 * v / wall) if wall else (lambda v: float("nan"))

    print()
    print("--- MAGNUS run summary -------------------------------------")
    print(f"initialisation     : {timing['init']:8.1f} s   "
          f"(excluded from the below)")
    print(f"wall clock         : {wall:8.1f} s   ({wall / 60.0:.1f} min)")
    print(f"  in HYSYS         : {solve:8.1f} s   ({share(solve):.1f} %)")
    print(f"  in sampler       : {timing['sampler']:8.1f} s   "
          f"({share(timing['sampler']):.1f} %)")
    print(f"model evaluations  : {int(NS.stats.numfct):8d}")

    if cache.misses:
        print(f"seconds per solve  : {solve / cache.misses:8.2f} s   "
              f"(mean over {cache.misses} solves)")

    # The seed is the distributable half; the replacement loop is not.
    # Quoting a speed-up against the total would attribute to the method
    # what only the seed can deliver.
    if cache.seed_end_t is None:
        print("  seed             :      n/a       (seed did not complete)")
    else:
        seed_wall = cache.seed_end_t - timing["t_start"]
        seed_solve = cache.seed_model_time - timing["model_time_start"]
        print(f"  seed             : {seed_wall:8.1f} s   "
              f"({cache.seed_solves} solves, {seed_solve:.1f} s in HYSYS)")
        print(f"  replacement      : {wall - seed_wall:8.1f} s   "
              f"({cache.misses - cache.seed_solves} solves, "
              f"{solve - seed_solve:.1f} s in HYSYS)")

    # NSFeas has no termination_reason, so it is derived the same way the
    # Python side defines it: a run has converged when no live point is
    # still violating. VaR convention here is feasible <= 0.
    uncertified = (int(np.sum(np.asarray(live_crit, dtype=float) > 0.0))
                   if n_live else 0)
    print(f"terminated         : "
          f"{'converged' if not uncertified else 'STOPPED'}  "
          f"({uncertified} live points uncertified)")

    # A warm start makes every number above a fiction: a reloaded solve
    # costs microseconds where the real one cost half a minute. Said loudly
    # rather than left for the reader to infer from a suspiciously low mean.
    if cache.warm_hits:
        print()
        print(f"WARNING: {cache.warm_hits} point(s) came from the cache FILE, "
              f"not from a solve.")
        print("         The times above are not a measurement of this "
              "problem. Re-run with")
        print("         --no-warm-start (or a fresh --cache path) to time it.")
    print("------------------------------------------------------------")


def report_and_save(NS, args, limits, cache, timing, _keep_alive):
    elapsed = timing["wall"]
    live_u, live_crit, live_prob, _ = NS.live_points
    dead_u, dead_crit, dead_prob, _ = NS.dead_points
    disc_u, disc_crit, disc_prob, _ = NS.discard_points

    def physical(points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if points.size == 0:
            return points.reshape(0, 2)
        lo = np.array([b[0] for b in DESIGN_BOUNDS])
        hi = np.array([b[1] for b in DESIGN_BOUNDS])
        return lo + (hi - lo) * 0.5 * (points + 1.0)

    live, dead, disc = physical(live_u), physical(dead_u), physical(disc_u)
    n_live, n_dead, n_disc = len(live), len(dead), len(disc)
    n_cand = n_dead + n_disc
    efficiency = n_dead / n_cand if n_cand else float("nan")

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  Live points       : {n_live}")
    print(f"  Dead points       : {n_dead}")
    print(f"  Discarded points  : {n_disc}")
    print(f"  Candidate evals   : {n_cand}")
    print(f"  Point evaluations : {int(NS.stats.numfct)}  "
          f"(= N_L + candidates)")
    print(f"  Model solves      : {cache.misses}   "
          f"(cache served {cache.hits} repeat call(s))")
    print(f"  Failed solves     : {cache.failures}  "
          f"(counted as infeasible)")
    print(f"  Sampling eff.     : {efficiency:.3f}  (accepted / candidates)")
    if n_live:
        print(f"  Worst live value  : {float(np.max(live_crit)):.4e}  "
              f"(VaR[G], feasible <= 0)")
        print(f"  Live bounding box : T {live[:, 0].min():.1f}-"
              f"{live[:, 0].max():.1f} C,  "
              f"P {live[:, 1].min() / 100:.1f}-{live[:, 1].max() / 100:.1f} "
              f"bar")
        print(f"    vs interpolated : T {REGION_T_CONT[0]:g}-"
              f"{REGION_T_CONT[1]:g} C,  "
              f"P {REGION_P_CONT[0] / 100:g}-{REGION_P_CONT[1] / 100:g} bar"
              f"   <- compare against THIS")
        print(f"    vs cell centres : T {REGION_T[0]:g}-{REGION_T[1]:g} C,  "
              f"P {REGION_P[0] / 100:g}-{REGION_P[1] / 100:g} bar   "
              f"(the boundary lies between nodes, so overshooting these is "
              f"correct)")
        edges = []
        for axis, (lo, hi) in enumerate(DESIGN_BOUNDS):
            width = hi - lo
            if live[:, axis].min() <= lo + 0.01 * width:
                edges.append(f"{DESIGN_NAMES[axis]} low")
            if live[:, axis].max() >= hi - 0.01 * width:
                edges.append(f"{DESIGN_NAMES[axis]} high")
        print(f"  Box edges touched : "
              f"{', '.join(edges) if edges else 'none -- closed interior'}")
    print(f"  Wall time         : {elapsed / 3600:.2f} h  "
          f"({elapsed:.0f} s)")
    if cache.misses:
        print(f"  Per solve         : {elapsed / cache.misses:.1f} s")
    print("=" * 78)

    print_run_summary(NS, cache, timing, n_live, live_crit)

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"{args.backend}_NL{args.numlive}"

    npz_path = os.path.join(args.out, f"magnus_tp_{tag}.npz")
    np.savez_compressed(
        npz_path,
        design_names=np.array(DESIGN_NAMES),
        design_bounds=np.array(DESIGN_BOUNDS, dtype=float),
        theta_samples=THETA_SAMPLES,
        constraints=np.array([[lim, np.inf] for lim in limits], dtype=float),

        live_points=live, live_probs=np.asarray(live_prob).ravel(),
        live_crit=np.asarray(live_crit).ravel(),
        dead_points=dead, dead_probs=np.asarray(dead_prob).ravel(),
        dead_crit=np.asarray(dead_crit).ravel(),
        rejected_points=disc, rejected_probs=np.asarray(disc_prob).ravel(),
        rejected_crit=np.asarray(disc_crit).ravel(),

        # Timing recorded alongside the points so the reported numbers can
        # be traced back to the run that produced them. NaN, not zero, when
        # the seed did not finish -- a zero would read as "the seed was
        # free" if it ever reached a plot.
        meta=np.array([args.numlive, args.numprop, len(THETA_SAMPLES),
                       n_dead, n_cand, int(NS.stats.numfct), cache.misses,
                       cache.failures, round(elapsed, 2),
                       round(timing["init"], 2),
                       round(timing["solve"], 2),
                       round(timing["sampler"], 2),
                       cache.warm_hits,
                       cache.seed_solves,
                       (np.nan if cache.seed_end_t is None else
                        round(cache.seed_end_t - timing["t_start"], 2)),
                       (np.nan if cache.seed_end_t is None else
                        round(cache.seed_model_time
                              - timing["model_time_start"], 2)),
                       ], dtype=float),
        meta_keys=np.array(["N_L", "NUMPROP", "N_theta", "n_replacements",
                            "n_candidates", "n_evaluations", "model_solves",
                            "n_failed", "wall_s",
                            "init_s", "solve_s", "sampler_s",
                            "warm_cache_hits", "seed_solves",
                            "seed_wall_s", "seed_solve_s"]),
        sampler=np.array("magnus-nsfeas"),
        backend=np.array(args.backend),
        scenario_mode=np.array("fixed"),
    )
    print(f"  saved             : {npz_path}")

    # Same schema as tp_design_space.csv, so plot_design_space.py reads this
    # file unchanged. merit = -crit: the Python side reports -(worst
    # violation), NSFeas reports the violation itself.
    csv_path = os.path.join(args.out, f"magnus_tp_{tag}.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(DESIGN_NAMES + ["P", "merit", "mode", "role"])
                     + "\n")
        blocks = [(dead, dead_prob, dead_crit, -1, "dead"),
                  (disc, disc_prob, disc_crit, -1, "rejected"),
                  (live, live_prob, live_crit, 1, "live")]
        rows = 0
        for points, probs, crit, mode, role in blocks:
            probs = np.asarray(probs, dtype=float).ravel()
            crit = np.asarray(crit, dtype=float).ravel()
            for point, prob, value in zip(points, probs, crit):
                coords = ",".join(f"{c:.10g}" for c in point)
                handle.write(f"{coords},{prob:.10g},{-value:.10g},"
                             f"{mode},{role}\n")
                rows += 1
    print(f"  saved             : {csv_path}  ({rows} rows)")
    print()
    print("  Plot it with, on the Windows side:")
    print(f"      python plot_design_space.py "
          f"<this file>")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The (T, P) methanol case study on MAGNUS / NSFeas.")
    parser.add_argument("--backend", choices=["bridge", "surrogate"],
                        default="bridge",
                        help="bridge = real HYSYS through the Windows-side "
                             "server; surrogate = bilinear interpolant of "
                             "the reference sweep, for checking the wiring")
    parser.add_argument("--spool", default="/mnt/c/HYSYS_cases/bridge",
                        help="shared directory the Windows server watches")
    parser.add_argument("--sweep", default=DEFAULT_SWEEP,
                        help="reference sweep CSV, for --backend surrogate")
    parser.add_argument("--cache", default=os.path.join(
        HERE, "magnus_tp_output", "solve_cache.csv"),
        help="solve cache/log; '' disables it (NOT advised -- it is what "
             "stops every design point costing two HYSYS solves)")
    parser.add_argument("--no-warm-start", action="store_true",
                        dest="no_warm_start",
                        help="keep writing the cache but do NOT load it, so "
                             "every point is solved afresh. Use this when the "
                             "run is being TIMED: reloaded solves return in "
                             "microseconds and make the wall clock a fiction")
    parser.add_argument("--numlive", type=int, default=500,
                        help="NUMLIVE; matches the Python run's N_L = 500")
    parser.add_argument("--numprop", type=int, default=COMMON_NS["NUMPROP"])
    parser.add_argument("--alpha", type=float, default=ALPHA_STAR,
                        help="alpha*; FEASTHRES = 1 - alpha. Irrelevant at "
                             "K = 1, where every value below 1 agrees")
    parser.add_argument("--c-threshold", type=float, default=LOWER_LIMITS[0],
                        dest="c_threshold", help="carbon efficiency [%%]")
    parser.add_argument("--e-threshold", type=float, default=LOWER_LIMITS[1],
                        dest="e_threshold", help="energy efficiency [%%]")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for one bridge solve")
    parser.add_argument("--out", default=os.path.join(HERE,
                                                      "magnus_tp_output"))
    parser.add_argument("--tag", default="",
                        help="output filename tag (default backend_NL<n>)")
    parser.add_argument("--no-log", action="store_true", dest="no_log",
                        help="do not mirror the console to "
                             "<out>/magnus_tp_<tag>.log")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)),
                    exist_ok=True)

    # Resolved here rather than in report_and_save so the log file, the npz
    # and the csv all carry the same tag -- a run's three outputs sorting
    # next to each other is the whole point of having one.
    args.tag = args.tag or f"{args.backend}_NL{args.numlive}"

    if not args.no_log:
        # Truncated at start, like the cache is not. Copy the previous log
        # elsewhere first if it still matters.
        log_path = os.path.join(args.out, f"magnus_tp_{args.tag}.log")
        start_console_log(log_path)
        print(f"  console log       : {log_path}")

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
