"""Is the flowsheet actually converged?

`Solver.IsSolving == False` only means HYSYS STOPPED. It does not mean the
answer is good -- HYSYS stops just as happily on a recycle that hit its
iteration limit, on an under-specified unit, or on a column that gave up.
On an unattended sweep that difference is the difference between a data
point and a silently wrong number.

Three independent layers, all read from the live COM interface. Every name
below was verified against Aspen HYSYS V14 by reading the HYSYS type
library, not assumed:

  1. Case.GetFlowsheetStatus(mask)   -- the flag HYSYS itself uses to colour
     the status bar. Case-wide: covers the main flowsheet AND every
     sub-flowsheet (columns included). Bitmask of FlowSheetObjStatusFlag_enum.
     Case.GetFlowsheetObjectTypeAndName(mask) returns (type, name) for the
     offending objects, so a failure is reportable rather than just a count.

  2. Recycle.RecycleConvergence     -- Convergence_enum. This is the one
     that matters most on a sweep: a recycle that stops at MaxIterations
     leaves a flowsheet that looks solved and is not.

  3. ColumnFlowsheet.CfsConverged   -- columns solve in their own nested
     solver and can fail without tripping the case-level flags.

Usage:
    from convergence import check_convergence

    report = check_convergence(sim)
    if not report.ok:
        print(report.summary())
"""

from dataclasses import dataclass, field


# --------------------------------------------------
# Enums (from the HYSYS V14 type library)
# --------------------------------------------------

# FlowSheetObjStatusFlag_enum
FLAG_OK               = 1
FLAG_NOT_SOLVED       = 2
FLAG_WARNING          = 4
FLAG_UNDER_SPECIFIED  = 8
FLAG_ERROR            = 16

FLAG_NAMES = {
    FLAG_OK:              "OK",
    FLAG_NOT_SOLVED:      "NotSolved",
    FLAG_WARNING:         "Warning",
    FLAG_UNDER_SPECIFIED: "UnderSpecified",
    FLAG_ERROR:           "Error",
}

# Anything in here means the flowsheet is not trustworthy.
BAD_FLAGS = (FLAG_NOT_SOLVED, FLAG_UNDER_SPECIFIED, FLAG_ERROR)

# Convergence_enum (Recycle.RecycleConvergence)
CONV_UNCONVERGED    = 0
CONV_CONVERGED      = 1
CONV_IGNORED        = 2
CONV_MAX_ITERATIONS = 3

CONV_NAMES = {
    CONV_UNCONVERGED:    "Unconverged",
    CONV_CONVERGED:      "Converged",
    CONV_IGNORED:        "Ignored",
    CONV_MAX_ITERATIONS: "MaxIterations",
}

# HYSYS returns this for a real variable that has no value yet.
EMPTY = -32767.0


def is_empty(value, tol=1e-6):
    """True if HYSYS handed back its 'no value' sentinel instead of a number."""
    try:
        return abs(float(value) - EMPTY) < tol
    except (TypeError, ValueError):
        return value is None


# --------------------------------------------------
# Report
# --------------------------------------------------

@dataclass
class ConvergenceReport:
    ok: bool = True
    counts: dict = field(default_factory=dict)      # flag name -> object count
    problems: list = field(default_factory=list)    # human-readable failures
    warnings: list = field(default_factory=list)    # non-fatal notes
    recycles: list = field(default_factory=list)    # (name, status, iters, max)
    columns: list = field(default_factory=list)     # (name, converged, iters, err)

    def summary(self):
        head = "CONVERGED" if self.ok else "NOT CONVERGED"
        lines = [f"[{head}]  " + "  ".join(
            f"{k}={v}" for k, v in self.counts.items() if v)]

        for p in self.problems:
            lines.append(f"   ! {p}")
        for w in self.warnings:
            lines.append(f"   ~ {w}")

        for name, status, iters, mx in self.recycles:
            mark = " " if status == CONV_CONVERGED else "!"
            lines.append(
                f"   {mark} recycle {name}: {CONV_NAMES.get(status, status)} "
                f"({iters:g}/{mx:g} iterations)")

        for name, conv, iters, err in self.columns:
            mark = " " if conv else "!"
            lines.append(
                f"   {mark} column {name}: "
                f"{'converged' if conv else 'NOT converged'} "
                f"({iters} iterations, equilibrium error {err:.3g})")

        return "\n".join(lines)

    def __bool__(self):
        return self.ok


# --------------------------------------------------
# Layer 1 -- case-wide object status flags
# --------------------------------------------------

def flowsheet_status(case, max_names=8):
    """Count objects per status flag, and name the bad ones.

    Case-wide: sub-flowsheets are included, so a column that fails to solve
    shows up here as well as in layer 3.
    """
    counts, offenders = {}, {}

    for flag in (FLAG_OK, FLAG_NOT_SOLVED, FLAG_WARNING,
                 FLAG_UNDER_SPECIFIED, FLAG_ERROR):
        label = FLAG_NAMES[flag]
        try:
            n = int(case.GetFlowsheetStatus(flag))
        except Exception as exc:
            counts[label] = None
            offenders[label] = [f"<query failed: {type(exc).__name__}>"]
            continue

        counts[label] = n
        if n and flag != FLAG_OK:
            offenders[label] = _object_names(case, flag, max_names)

    return counts, offenders


def _object_names(case, flag, max_names):
    """(type, name) pairs for the objects carrying `flag`."""
    try:
        raw = case.GetFlowsheetObjectTypeAndName(flag)
    except Exception as exc:
        return [f"<name query failed: {type(exc).__name__}>"]

    if not raw:
        return []

    names = []
    for entry in list(raw)[:max_names]:
        try:
            otype, oname = entry
            names.append(f"{oname} ({otype})")
        except Exception:
            names.append(str(entry))
    return names


# --------------------------------------------------
# Flowsheet walking
# --------------------------------------------------

def _all_flowsheets(sim):
    """Main flowsheet plus every sub-flowsheet, one level deep."""
    sheets = [sim.Flowsheet]
    try:
        sheets.extend(sim.Flowsheet.Flowsheets)
    except Exception:
        pass
    return sheets


# --------------------------------------------------
# Layer 2 -- recycles
# --------------------------------------------------

def recycle_status(sim):
    """[(name, Convergence_enum, iterations, max_iterations), ...]"""
    out = []
    for sheet in _all_flowsheets(sim):
        try:
            ops = list(sheet.Operations)
        except Exception:
            continue

        for op in ops:
            try:
                if "recycle" not in str(op.TypeName).lower():
                    continue
            except Exception:
                continue

            try:
                out.append((
                    op.name,
                    int(op.RecycleConvergence),
                    float(op.IterationsValue),
                    float(op.MaximumIterationsValue),
                ))
            except Exception as exc:
                out.append((getattr(op, "name", "<?>"),
                            f"<read failed: {type(exc).__name__}>", 0.0, 0.0))
    return out


# --------------------------------------------------
# Layer 3 -- columns
# --------------------------------------------------

def column_status(sim):
    """[(name, CfsConverged, CurrentIteration, EquilibriumError), ...]

    Columns run a nested solver, so they can fail on their own terms.
    """
    out = []
    for sheet in _all_flowsheets(sim):
        try:
            ops = list(sheet.Operations)
        except Exception:
            continue

        for op in ops:
            # Attribute access rather than `"ColumnFlowsheet" in dir(op)`:
            # dir() does not list COM properties on a raw win32com dynamic
            # Dispatch (the parallel workers), so the dir() test finds no
            # columns there and the check silently passes. Asking for the
            # property works on both HysysPy wrappers and raw Dispatch.
            try:
                cf = op.ColumnFlowsheet
            except Exception:
                continue            # not a column
            if cf is None:
                continue
            try:
                out.append((op.name,
                            bool(cf.CfsConverged),
                            int(cf.CurrentIteration),
                            float(cf.EquilibriumError)))
            except Exception:
                out.append((getattr(op, "name", "<?>"),
                            False, 0, float("nan")))
    return out


# --------------------------------------------------
# The check
# --------------------------------------------------

def check_convergence(sim, warnings_are_fatal=False, ignored_recycles_ok=True):
    """Verify the flowsheet really converged. Read-only; touches no solver state.

    warnings_are_fatal   -- treat flag_Warning objects as a failure. Off by
                            default: a converged HYSYS case routinely carries
                            benign warnings, so this would reject good points.
    ignored_recycles_ok  -- an intentionally ignored recycle is not a failure.
    """
    report = ConvergenceReport()

    # -- layer 1 --------------------------------------------------
    counts, offenders = flowsheet_status(sim.Case)
    report.counts = counts

    for flag in BAD_FLAGS:
        label = FLAG_NAMES[flag]
        n = counts.get(label)
        if n:
            names = ", ".join(offenders.get(label, [])) or "?"
            report.problems.append(f"{n} object(s) {label}: {names}")
            report.ok = False

    n_warn = counts.get(FLAG_NAMES[FLAG_WARNING])
    if n_warn:
        names = ", ".join(offenders.get(FLAG_NAMES[FLAG_WARNING], [])) or "?"
        msg = f"{n_warn} object(s) Warning: {names}"
        if warnings_are_fatal:
            report.problems.append(msg)
            report.ok = False
        else:
            report.warnings.append(msg)

    # -- layer 2 --------------------------------------------------
    report.recycles = recycle_status(sim)
    for name, status, iters, mx in report.recycles:
        if status == CONV_CONVERGED:
            continue
        if status == CONV_IGNORED:
            report.warnings.append(f"recycle {name} is ignored")
            if not ignored_recycles_ok:
                report.ok = False
            continue
        report.problems.append(
            f"recycle {name}: {CONV_NAMES.get(status, status)} "
            f"after {iters:g}/{mx:g} iterations")
        report.ok = False

    # -- layer 3 --------------------------------------------------
    report.columns = column_status(sim)
    for name, conv, iters, err in report.columns:
        if not conv:
            report.problems.append(
                f"column {name} did not converge "
                f"({iters} iterations, equilibrium error {err:.3g})")
            report.ok = False

    return report


def assert_converged(sim, **kwargs):
    """check_convergence(), but raise instead of returning a falsy report."""
    report = check_convergence(sim, **kwargs)
    if not report.ok:
        raise RuntimeError("flowsheet did not converge\n" + report.summary())
    return report


# --------------------------------------------------
# Standalone: report on whatever is loaded right now
# --------------------------------------------------

if __name__ == "__main__":
    from hysyspy import HysysPy

    sim = HysysPy(instance="methanol.hsc")
    solver = sim.Solver

    print(f"case            : {sim.name}")
    print(f"solver busy     : IsSolving={solver.IsSolving} "
          f"IsForgetting={solver.IsForgetting}")
    print(f"solver released : CanSolve={solver.CanSolve}")
    print()
    print(check_convergence(sim).summary())
