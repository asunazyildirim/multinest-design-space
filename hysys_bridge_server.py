"""
Windows-side solve server for the MAGNUS port of the (T, P) case study.

MAGNUS/NSFeas lives in WSL; HYSYS is a Windows COM server. Nothing in WSL can
``import pythoncom``, and nothing on Windows can ``import magnus``. So the two
halves talk through the one thing they genuinely share: the NTFS filesystem,
visible as ``C:\\...`` here and ``/mnt/c/...`` there.

This process owns the HYSYS connection and does nothing else. It watches a
spool directory for request files, runs ``run_case`` once per request, and
writes the answer back beside it. ``magnus_tp_case_study.py`` on the WSL side
is the only client.

    request   req_<id>.json   {"id", "kwargs", "outputs"}
    response  res_<id>.json   {"id", "ok", "outputs", "error", "solve_s"}

Both are written to ``<name>.tmp`` and renamed into place, so a reader never
sees half a file -- rename within one directory is atomic on NTFS, and that is
what makes a plain shared folder a usable transport.

SERIAL BY DESIGN. One instance, one solve at a time. NSFeas evaluates the
model one point at a time (MAXTHREAD=1 on the DAG), so a second instance would
sit idle; the parallel machinery in ``parallel_estimator.py`` has no NSFeas
equivalent to feed it.

Usage
-----
Open ``methanol_instance0.hsc`` in HYSYS, then, from an Anaconda prompt in
this directory::

    python hysys_bridge_server.py

Leave it running. Start the WSL side afterwards; requests that arrive before
the server is up simply wait in the spool. Ctrl-C stops it -- if a solve is in
flight it finishes first, so the client is never left waiting on a reply that
will never come.
"""

import argparse
import csv
import inspect
import json
import os
import sys
import time
import traceback

import pythoncom

from hysys_connection import get_open_hysys_case
from simulation_runner import run_case, initialise_hysys_objects


HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SPOOL = r"C:\HYSYS_cases\bridge"
DEFAULT_SWEEP = os.path.join(
    HERE, "output", "sensitivity_results_20260731_174218.csv")


def open_hysys_cases():
    """Full paths of every .hsc currently registered in the ROT.

    get_open_hysys_case matches a full path against the running-object
    table, so the path it is given has to be the one the open case was
    opened from -- a copy sitting on disk at that path is not enough, and
    a default path baked in here is one more thing that has to agree with
    what is on screen. Reading the ROT lets the open case supply its own.
    """
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
            continue                # entries that refuse to name themselves
        if name.lower().endswith(".hsc"):
            names.append(name)


def resolve_case(requested):
    """Pick the case to drive: the one that is open, or the one named."""
    found = open_hysys_cases()
    listed = "\n".join(f"    {name}" for name in found) or "    (none)"

    if requested:
        if requested.lower() in [name.lower() for name in found]:
            return requested
        raise SystemExit(
            f"--case names a case that is not open: {requested}\n"
            f"  file exists on disk : {os.path.isfile(requested)}\n"
            f"open in HYSYS right now:\n{listed}\n"
            "Open that case, name one of the above, or drop --case to let "
            "the server take the open case by itself.")

    # One open case is the normal way to run this, so it is taken without
    # asking. Several open at once is NOT resolved by picking the first:
    # the ROT order is arbitrary, and serving solves from the wrong
    # flowsheet produces answers that look fine and belong to another case.
    if len(found) == 1:
        return found[0]

    raise SystemExit(
        ("No HYSYS case is open.\n" if not found else
         f"{len(found)} HYSYS cases are open; which one is ambiguous.\n")
        + f"open in HYSYS right now:\n{listed}\n"
        + ("Open the case in HYSYS and re-run." if not found else
           "Choose one:  python hysys_bridge_server.py --case \"<full path>\"")
        + "\nThis server attaches to a running instance; it does not launch "
          "one.")

# Written every couple of seconds while idle AND immediately before each
# solve, so the client can tell "the server died an hour ago" from "the server
# is busy" -- otherwise both look the same from the far side and the client
# waits out its full timeout on a dead server.
#
# Refreshing it before the solve rather than only in the idle loop is the part
# that matters: a HYSYS solve takes 20-30 s and blocks this process for all of
# it, so a heartbeat written only between requests is always as old as the
# solve in flight. The client's staleness threshold then has to exceed the
# longest solve, and every second of that margin is a second it spends waiting
# on a server that has actually gone.
HEARTBEAT = "server.alive"
HEARTBEAT_EVERY = 2.0

# run_case raises TypeError on an unexpected keyword. Better to reject the
# request with a readable error than to kill the server on a client typo.
RUN_CASE_KEYS = set(inspect.signature(run_case).parameters) - {"objects"}


class _FakeSolver:
    """A stand-in for HYSYS, for testing the transport and nothing else.

    ``--fake`` answers from a bilinear interpolant of the reference sweep
    instead of solving. It exists so the half of this system that has to
    survive ten unattended hours -- the spool, the atomic renames, the
    heartbeat, the client's timeout -- can be exercised end to end in a
    minute, on a machine with no HYSYS licence in use.

    It ignores the activation energies, because the sweep was measured at
    the nominal pair. NOT a result. Runs with --fake write
    ``"fake": true`` into every response so nothing downstream can mistake
    one for a solve.
    """

    def __init__(self, path, delay=0.0):
        self.delay = delay
        rows = []
        with open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["converged"] == "True":
                    rows.append(row)
        if not rows:
            raise SystemExit(f"no converged rows in {path}")

        self.keys = [k for k in rows[0]
                     if k.endswith(("_pct", "_kg_h", "_kJ_h"))]
        self.T = sorted({float(r["temperature_set_C"]) for r in rows})
        self.P = sorted({float(r["pressure_set_kPa"]) for r in rows})
        self.table = {(float(r["temperature_set_C"]),
                       float(r["pressure_set_kPa"])): r for r in rows}
        if len(self.table) != len(self.T) * len(self.P):
            raise SystemExit(f"{path} is not a full factorial")

    @staticmethod
    def _bracket(grid, value):
        index = min(max(sum(g <= value for g in grid) - 1, 0), len(grid) - 2)
        span = grid[index + 1] - grid[index]
        return index, (value - grid[index]) / span

    def __call__(self, kwargs, wanted):
        if self.delay:
            time.sleep(self.delay)
        T = float(kwargs.get("temperature", self.T[0]))
        P = float(kwargs.get("pressure", self.P[0]))
        i, u = self._bracket(self.T, T)
        j, v = self._bracket(self.P, P)

        out = {}
        for key in wanted:
            corners = [(self.table[(self.T[i + a], self.P[j + b])][key],
                        (u if a else 1 - u) * (v if b else 1 - v))
                       for a in (0, 1) for b in (0, 1)]
            out[key] = sum(float(value) * weight for value, weight in corners)
        return out


def _write_atomic(path, payload, attempts=40, delay=0.25):
    """Serialise to ``path`` via a temporary file and a rename, with retries.

    For the request and response files, where the reader must never see a
    partial write.

    The retries are not defensive padding. ``os.replace`` on Windows fails
    with PermissionError -- WinError 5 -- if ANY other process holds the
    source or target open, and on a freshly created file that other process
    is routinely Defender or the search indexer, which open it to scan the
    moment it appears. Measured here: 411 responses written, then one
    failure, which killed the server and stranded the client. The scan takes
    milliseconds, so a retry loop out to ten seconds turns a fatal race into
    an invisible pause.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())

    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def _write_heartbeat(path, payload, attempts=5):
    """Overwrite the heartbeat in place. Deliberately NOT atomic.

    ``os.replace`` cannot be used here. The heartbeat is the one file that is
    written over and over while another process reads it over and over, and on
    Windows a rename onto a path that someone else holds open fails outright
    with PermissionError -- WinError 5, which killed an earlier version of
    this server mid-run. The WSL client polling across the 9p boundary counts
    as that someone.

    Writing in place trades atomicity for the ability to write at all, and
    nothing needs the atomicity: a client that reads a half-written heartbeat
    gets a JSON error, treats the age as unknown, and looks again. Only an
    unbroken absence of several minutes is read as a dead server.
    """
    for attempt in range(attempts):
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return True
        except OSError:
            if attempt == attempts - 1:
                # Not fatal. A missed heartbeat costs nothing unless they are
                # ALL missed, and dying here would turn a transient file lock
                # into the end of a ten-hour run.
                return False
            time.sleep(0.05)
    return False


def _pending(spool):
    """Request ids waiting in the spool.

    Ids are opaque strings, not integers: the client stamps each one with a
    per-run prefix so that two clients sharing a spool cannot both number
    from 1 and read each other's answers. Nothing here needs to interpret
    them -- a request is served, and its answer goes back under the same
    name.

    scandir rather than glob: it is one readdir, and it does not build a
    second list of names we then have to re-stat.
    """
    ids = []
    with os.scandir(spool) as entries:
        for entry in entries:
            name = entry.name
            if name.startswith("req_") and name.endswith(".json"):
                ids.append(name[4:-5])
    return sorted(ids)


def _serve_one(objects, spool, req_id, verbose, fake=None, first_seen=None,
               parse_grace=10.0):
    """Run one request, or report that it is not readable yet.

    Returns the response dict, or ``None`` meaning "come back later".
    """
    req_path = os.path.join(spool, f"req_{req_id}.json")
    res_path = os.path.join(spool, f"res_{req_id}.json")

    try:
        with open(req_path, encoding="utf-8") as handle:
            request = json.load(handle)
    except (OSError, ValueError) as exc:
        # NOT a failed solve. The client writes the request on the WSL side
        # and renames it into place, but the rename can become visible here
        # before the contents have crossed the 9p boundary, so an empty or
        # truncated read means "not arrived yet", not "malformed".
        #
        # Measured: one request in 454 arrived this way. Answering it as a
        # failure told the client that an ordinary interior design point was
        # infeasible -- a wrong answer, silently, in the middle of the region.
        # So wait for it instead, and only give up after a grace period no
        # filesystem delay could explain.
        if first_seen is not None:
            since = first_seen.setdefault(req_id, time.time())
            if time.time() - since < parse_grace:
                return None

        response = {"id": req_id, "ok": False, "retryable": True,
                    "outputs": {},
                    "error": f"request unreadable after {parse_grace:g} s: "
                             f"{exc}",
                    "solve_s": 0.0}
        _write_atomic(res_path, response)
        try:
            os.remove(req_path)
        except OSError:
            pass
        return response

    kwargs = request.get("kwargs", {})
    wanted = request.get("outputs", [])

    unknown = set(kwargs) - RUN_CASE_KEYS
    if unknown:
        # A client bug, not a transport hiccup and not an infeasible point.
        # retryable, so the client raises rather than folding a typo into the
        # region as a wall of infeasible points.
        response = {"id": req_id, "ok": False, "retryable": True,
                    "outputs": {},
                    "error": f"run_case has no keyword(s) {sorted(unknown)}",
                    "solve_s": 0.0}
        _write_atomic(res_path, response)
        os.remove(req_path)
        return response

    start = time.perf_counter()
    try:
        if fake is not None:
            values = fake(kwargs, wanted)
            response = {"id": req_id, "ok": True, "fake": True,
                        "outputs": values, "error": "",
                        "solve_s": time.perf_counter() - start}
            _write_atomic(res_path, response)
            os.remove(req_path)
            if verbose:
                print(f"  {req_id:>18}  FAKE  {response['solve_s']:5.1f}s  "
                      + ", ".join(f"{k}={v:g}" for k, v in kwargs.items()
                                  if isinstance(v, (int, float))))
                sys.stdout.flush()
            return response

        out = run_case(objects, verbose=False, **kwargs)
        if not out["converged"]:
            detail = out["convergence_detail"] or out["write_error"] \
                or "no detail"
            response = {"id": req_id, "ok": False, "outputs": {},
                        "error": f"unconverged: {detail}",
                        "solve_s": out.get("solve_time_s", 0.0)}
        else:
            values = {key: out.get(key) for key in wanted}
            missing = [k for k, v in values.items() if v is None]
            if missing:
                response = {"id": req_id, "ok": False, "outputs": {},
                            "error": f"converged but missing {missing}",
                            "solve_s": out.get("solve_time_s", 0.0)}
            else:
                response = {"id": req_id, "ok": True,
                            "outputs": {k: float(v) for k, v in values.items()},
                            "error": "",
                            "solve_s": out.get("solve_time_s", 0.0)}
    except Exception as exc:                        # noqa: BLE001
        # A COM failure must not take the server down: the client turns a
        # failed solve into an infeasible point, which is the same thing the
        # Python sampler does with on_failure="infeasible".
        response = {"id": req_id, "ok": False, "outputs": {},
                    "error": f"{type(exc).__name__}: {exc}",
                    "solve_s": time.perf_counter() - start}
        traceback.print_exc()

    _write_atomic(res_path, response)
    # Only now: while req_ exists the request is unanswered, so deleting it
    # before the response lands would lose the request on a crash in between.
    try:
        os.remove(req_path)
    except OSError:
        pass

    if verbose:
        where = ", ".join(f"{k}={v:g}" for k, v in kwargs.items()
                          if isinstance(v, (int, float)))
        mark = "ok " if response["ok"] else "FAIL"
        print(f"  {req_id:>18}  {mark}  {response['solve_s']:5.1f}s  {where}"
              + ("" if response["ok"] else f"  -- {response['error']}"))
        sys.stdout.flush()

    return response


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Serve HYSYS solves to the MAGNUS side over a spool "
                    "directory.")
    parser.add_argument("--spool", default=DEFAULT_SPOOL,
                        help=f"shared directory (default {DEFAULT_SPOOL})")
    parser.add_argument("--case", default="",
                        help="full path of the open HYSYS case to drive. "
                             "Omit it and the server takes whichever case is "
                             "open; pass it only to disambiguate when several "
                             "are open at once")
    parser.add_argument("--poll", type=float, default=0.05,
                        help="seconds between directory checks when idle")
    parser.add_argument("--keep", action="store_true",
                        help="do not clear stale req_/res_ files at start")
    parser.add_argument("--force", action="store_true",
                        help="start even if another server's heartbeat is "
                             "still fresh. Only when you know it is dead")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fake", action="store_true",
                        help="do not connect to HYSYS; answer from the "
                             "reference sweep. Tests the transport, not the "
                             "process -- see _FakeSolver")
    parser.add_argument("--fake-sweep", default=DEFAULT_SWEEP,
                        dest="fake_sweep")
    parser.add_argument("--fake-delay", type=float, default=0.0,
                        dest="fake_delay",
                        help="seconds per fake solve, to imitate the real "
                             "20-30 s and exercise the client's waiting")
    parser.add_argument("--max-solves", type=int, default=0,
                        dest="max_solves",
                        help="stop after this many solves (0 = never)")
    parser.add_argument("--parse-grace", type=float, default=10.0,
                        dest="parse_grace",
                        help="seconds to keep re-reading a request whose "
                             "contents have not crossed from WSL yet")
    args = parser.parse_args(argv)

    os.makedirs(args.spool, exist_ok=True)

    # One server per spool, enforced. Two of them racing on the same
    # directory is not a slow run, it is a WRONG one: both pick up the same
    # request, both write the same res_<id>.json.tmp, and whichever renames
    # second fails with FileNotFoundError because the first already renamed
    # that exact path away. The client sees the leftovers as failed solves
    # and marks perfectly good design points infeasible. Observed here: five
    # spurious failures in 624 solves, from nothing worse than forgetting to
    # close the first server.
    beat_path = os.path.join(args.spool, HEARTBEAT)
    try:
        with open(beat_path, encoding="utf-8") as handle:
            age = time.time() - float(json.load(handle)["t"])
        if age < 60.0 and not args.force:
            raise SystemExit(
                f"another server is already serving {args.spool}\n"
                f"  (its heartbeat is {age:.0f} s old)\n"
                f"Stop it first. Two servers on one spool corrupt each "
                f"other's answers.\n"
                f"If you are certain nothing else is running -- a killed "
                f"server can leave a recent heartbeat behind -- start with "
                f"--force.")
    except (OSError, ValueError, KeyError):
        pass                            # no readable heartbeat: nobody home

    if not args.keep:
        stale = [n for n in os.listdir(args.spool)
                 if n.startswith(("req_", "res_"))]
        for name in stale:
            os.remove(os.path.join(args.spool, name))
        if stale:
            print(f"cleared {len(stale)} stale file(s) from a previous run")

    fake = None
    objects = None
    if args.fake:
        fake = _FakeSolver(args.fake_sweep, args.fake_delay)
    else:
        # Resolved before connecting, so a missing or ambiguous case is
        # reported with the open ones listed rather than as a bare
        # "case not found" naming a path nobody chose.
        args.case = resolve_case(args.case)
        pythoncom.CoInitialize()
        case = get_open_hysys_case(args.case)
        objects = initialise_hysys_objects(case)
        objects["case_name"] = os.path.basename(args.case)

    print()
    print("=" * 70)
    print("HYSYS bridge server" + ("   [FAKE -- transport test only]"
                                   if args.fake else ""))
    print("=" * 70)
    print(f"  case   : "
          + (f"{args.fake_sweep} (interpolated, {args.fake_delay:g} s/solve)"
             if args.fake else args.case))
    print(f"  spool  : {args.spool}")
    print(f"  WSL    : /mnt/{args.spool[0].lower()}"
          f"{args.spool[2:].replace(chr(92), '/')}")
    print("  ready -- start the WSL side now.  Ctrl-C to stop.")
    print("=" * 70)
    sys.stdout.flush()

    served = 0
    failed = 0
    started = time.time()
    last_beat = 0.0
    first_seen = {}                     # req_id -> when it first appeared
    beat_path = os.path.join(args.spool, HEARTBEAT)

    def beat(busy=0):
        _write_heartbeat(beat_path, {"t": time.time(), "served": served,
                                     "failed": failed, "busy": busy})

    try:
        while True:
            now = time.time()
            if now - last_beat >= HEARTBEAT_EVERY:
                beat()
                last_beat = now

            ids = _pending(args.spool)
            if not ids:
                time.sleep(args.poll)
                continue

            for req_id in ids:
                # Before the solve, not after: the solve blocks this process
                # for 20-30 s and the client reads staleness while it waits.
                beat(req_id)
                last_beat = time.time()
                response = _serve_one(objects, args.spool, req_id,
                                      not args.quiet, fake, first_seen,
                                      args.parse_grace)
                if response is None:
                    continue            # still arriving; look again next pass
                first_seen.pop(req_id, None)
                served += 1
                failed += 0 if response["ok"] else 1
                if args.max_solves and served >= args.max_solves:
                    print(f"reached --max-solves {args.max_solves}")
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print()
        print(f"stopped after {served} solve(s), {failed} failed, "
              f"{(time.time() - started) / 60:.1f} min")
    finally:
        # Both, and the .tmp especially: a process killed between the write
        # and the rename leaves it behind, and a stray .tmp in the spool is
        # the one artefact that outlives every other trace of the failure.
        for name in (HEARTBEAT, HEARTBEAT + ".tmp"):
            try:
                os.remove(os.path.join(args.spool, name))
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
