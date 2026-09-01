"""Worker-side run_case for the parallel sweep.

Solve strategy (kept from the original runner) -- warm-up in stages instead
of handing the solver the hard problem in one go:

  1. Ignore the column (Twp101) and the flare (CRV-100), relax the purge
     split to 0.1/0.9, and solve. This gives the recycle tears a starting
     point near the new conditions without the two hardest units in the loop.
  2. Write the manipulated variables and solve the simplified flowsheet.
  3. Restore the real split (1 - recycle_fraction, recycle_fraction), solve.
  4. Un-ignore the column and the flare, final solve.

Outputs and the convergence verdict match sensitivity_study.run_case; the
inputs add the two Rxn-8 activation energies on top of it, so this runner's
CSV carries four columns the single-instance one does not. Every read goes
through _read and can come back None on an unconverged case -- the row
records `converged = False` instead of crashing the worker, and one bad
point does not take down the rest of the worker's chunk.
"""

import time
from datetime import datetime

from pywintypes import com_error

from convergence import check_convergence, CONV_NAMES, is_empty


# --------------------------------------------------
# check_convergence expects a HysysPy-shaped object
# --------------------------------------------------

class CaseAdapter:
    """Give a raw COM SimulationCase the .Case / .Flowsheet attributes
    check_convergence expects from a HysysPy instance."""

    def __init__(self, case):
        self.Case = case
        self.Flowsheet = case.Flowsheet


# --------------------------------------------------
# Reading values off a case that may not have solved
# (same helpers as sensitivity_study)
# --------------------------------------------------

def _com_message(exc):
    """Readable one-liner from a com_error, which prints as raw HRESULTs."""
    info = getattr(exc, "excepinfo", None) or ()
    text = info[2] if len(info) > 2 else None
    if text:
        return " ".join(str(text).split())
    scode = info[5] if len(info) > 5 else None
    code = scode if scode is not None else (exc.args[0] if exc.args else None)
    if not isinstance(code, int):
        return str(exc)     # non-COM exception routed here -- show it as-is
    return f"COM error 0x{code & 0xFFFFFFFF:08X}"


def _read(getter, default=None):
    """One COM value, or `default` if HYSYS has none to give.

    Catches com_error and nothing else on purpose: a misspelled property
    still fails loudly as AttributeError instead of becoming a column of
    plausible-looking blanks.
    """
    try:
        value = getter()
    except com_error:
        return default
    return default if is_empty(value) else value


def _scale(value, factor):
    """Unit conversion that tolerates a missing value."""
    return None if value is None else value * factor


def _total(objects, attribute):
    """Sum `attribute` over `objects`, or None if any one of them is missing."""
    values = [_read(lambda o=o: getattr(o, attribute)) for o in objects]
    return None if any(v is None for v in values) else sum(values)


def _lhv_flow_kJ_h(stream):
    """Heating value carried by a stream: LHV x molar flow, in kJ/h."""
    lhv = _read(lambda: stream.LowerHeatValueValue)
    flow = _read(lambda: stream.MolarFlowValue)
    if lhv is None or flow is None:
        return None
    return lhv * flow * 3600


# --------------------------------------------------
# Solving
# --------------------------------------------------

SOLVE_TIMEOUT = 300.0   # s per stage

# The warm-up split while the column and flare are ignored: a loose purge so
# the recycle converges easily before the real (much tighter) split goes in.
WARMUP_SPLIT = (0.1, 0.9)

# Real split when the caller does not sweep recycle_fraction.
DEFAULT_RECYCLE_FRACTION = 0.99


def _solve(solver, case_name, label):
    """Release the solver, wait until it goes idle, return elapsed seconds.

    The clock starts BEFORE `CanSolve = True`: HYSYS usually performs the
    whole solve inside that assignment, so a clock started after it measures
    an already-finished solve.

    The timeout cannot abort a solve Python is blocked inside -- it only
    fires if HYSYS hands control back while still iterating. On timeout the
    loop breaks instead of raising: the convergence check afterwards is what
    decides whether the point is usable, and raising here would kill the
    worker's remaining points.
    """
    start = time.perf_counter()
    solver.CanSolve = True
    while solver.IsForgetting or solver.IsSolving:
        if time.perf_counter() - start > SOLVE_TIMEOUT:
            print(f"{case_name}: [!] still solving after {SOLVE_TIMEOUT:.0f} s "
                  f"{label} -- leaving the verdict to the convergence check",
                  flush=True)
            break
        time.sleep(0.1)
    return time.perf_counter() - start


# --------------------------------------------------
# General sensitivity function
# --------------------------------------------------

def run_case(
        objects,
        pressure=None,
        temperature=None,
        ratio=None,
        volume=None,
        recycle_fraction=None,
        rxn8_ea_forward=None,
        rxn8_ea_reverse=None,
        verbose=True,
):
    """One sensitivity point. `None` for a variable leaves it as HYSYS has it.

    rxn8_ea_forward / rxn8_ea_reverse are the Arrhenius activation energies
    of Rxn-8 (CO2 + H2 -> CO + H2O, the RWGS step of the vandal2013 set), in
    HYSYS internal energy units.

    Careful with those two: unlike the reactor and the streams, a reaction
    belongs to the FLUID PACKAGE. A value written here stays written for
    every later point in this worker's chunk that passes None, and it would
    be baked into the .hsc if that case were ever saved. The
    `rxn8_ea_*_actual` columns record what the reaction really held when the
    row was collected, so a sweep is still readable afterwards -- but sweep
    them on the copies, not on the master case.

    verbose=False drops the one-line timing report that would otherwise
    print for EVERY point. It does not silence the failure reports: a
    rejected write and an unconverged solve still print, so a quiet run
    means a clean run rather than a run with nothing to say.
    """
    case_name = objects["case_name"]
    sim = objects["sim"]
    solver = objects["solver"]

    rin = objects["rin"]
    rout = objects["rout"]
    co2_fresh_in = objects["co2_fresh_in"]
    h2_fresh_in = objects["h2_fresh_in"]
    methanol = objects["methanol"]
    purge = objects["purge"]

    reactor = objects["reactor"]
    rxn8 = objects["rxn8"]
    column = objects["column"]
    flare = objects["flare"]
    split = objects["split"]
    compressors = objects["compressors"]
    expanders = objects["expanders"]

    pressure_cell = objects["pressure_cell"]
    ratio_cell = objects["ratio_cell"]

    h2_index = objects["h2_index"]
    co2_index = objects["co2_index"]

    case_start = time.perf_counter()
    label = (f"(P={pressure}, T={temperature}, ratio={ratio}, V={volume}, "
             f"recycle={recycle_fraction}, "
             f"Ea8f={rxn8_ea_forward}, Ea8r={rxn8_ea_reverse})")

    write_error = None
    solve_time = 0.0

    # Validated up front, before the warm-up touches the flowsheet: a split
    # of 0 or 1 is not rejected by HYSYS, it silently kills the purge or the
    # recycle and the case fails somewhere else looking like a solver problem.
    if recycle_fraction is not None and not 0 < recycle_fraction < 1:
        write_error = (f"recycle_fraction must be in (0, 1), "
                       f"got {recycle_fraction}")
        print(f"{case_name}: [!] {write_error}")
        recycle_fraction = None     # warm-up proceeds on the default split

    final_recycle = (recycle_fraction if recycle_fraction is not None
                     else DEFAULT_RECYCLE_FRACTION)

    # -----------------------------
    # Warm-up stage 1: simplify the flowsheet
    # -----------------------------

    solver.CanSolve = False
    try:
        column.IsIgnored = True
        flare.IsIgnored = True
        split.SplitsValue = WARMUP_SPLIT
    except com_error as exc:
        write_error = write_error or _com_message(exc)
        print(f"{case_name}: [!] warm-up setup failed {label}: "
              f"{_com_message(exc)}")
    finally:
        # The solver gets released whatever happened above, or HYSYS is left
        # frozen for every case after this one.
        solve_time += _solve(solver, case_name, label + " [warm-up]")

    # -----------------------------
    # Warm-up stage 2: write the manipulated variables, solve simplified
    # -----------------------------

    solver.CanSolve = False
    try:
        if pressure is not None:
            pressure_cell.CellValue = pressure

        if temperature is not None:
            rin.TemperatureValue = temperature
            rout.TemperatureValue = temperature

        if ratio is not None:
            ratio_cell.CellValue = ratio

        if volume is not None:
            reactor.TotalVolumeValue = volume

        # Kinetics last: the reaction set is shared state, so a rejected
        # write above should not leave the fluid package half-modified.
        if rxn8_ea_forward is not None:
            rxn8.ForwardActivationEnergy = rxn8_ea_forward

        if rxn8_ea_reverse is not None:
            rxn8.ReverseActivationEnergy = rxn8_ea_reverse

    except com_error as exc:
        # A rejected write is a bad case, not a bad sweep: record it and
        # carry on, same as an unconverged solve.
        write_error = write_error or _com_message(exc)
        print(f"{case_name}: [!] could not set {label}: {_com_message(exc)}")

    finally:
        solve_time += _solve(solver, case_name, label + " [new conditions]")

    # -----------------------------
    # Warm-up stage 3: restore the real split
    # -----------------------------

    solver.CanSolve = False
    try:
        # TEE-100 product order is (purge, VAPREC): the recycle share is the
        # second split, the purge takes the rest.
        split.SplitsValue = (1 - final_recycle, final_recycle)
    except com_error as exc:
        write_error = write_error or _com_message(exc)
        print(f"{case_name}: [!] could not restore split {label}: "
              f"{_com_message(exc)}")
    finally:
        solve_time += _solve(solver, case_name, label + " [real split]")

    # -----------------------------
    # Warm-up stage 4: column and flare back in, final solve
    # -----------------------------

    solver.CanSolve = False
    try:
        column.IsIgnored = False
        flare.IsIgnored = False
    except com_error as exc:
        write_error = write_error or _com_message(exc)
        print(f"{case_name}: [!] could not re-enable column/flare {label}: "
              f"{_com_message(exc)}")
    finally:
        solve_time += _solve(solver, case_name, label + " [final]")

    # -----------------------------
    # Convergence check
    # -----------------------------

    # The solver going idle is not the same as the solve succeeding: HYSYS
    # stops just as happily on a recycle at its iteration limit or a column
    # that gave up. Record the verdict alongside the numbers rather than
    # raising -- a row carrying `converged = False` is filterable afterwards.
    report = check_convergence(sim)
    if not report.ok:
        print(f"{case_name}: {report.summary()}")

    # -----------------------------
    # Collect results
    # -----------------------------

    co2_fresh_in_kmol_h = _scale(_read(lambda: co2_fresh_in.MolarFlowValue), 3600)
    h2_fresh_in_kmol_h = _scale(_read(lambda: h2_fresh_in.MolarFlowValue), 3600)
    methanol_kmol_h = _scale(_read(lambda: methanol.MolarFlowValue), 3600)
    methanol_kg_h = _scale(_read(lambda: methanol.MassFlowValue), 3600)

    # `not co2_fresh_in_kmol_h` covers None and 0.0 together: a zero fresh
    # feed is a real outcome when the solve fails, and dividing by it raises.
    if methanol_kmol_h is None or not co2_fresh_in_kmol_h:
        C_eff = None
    else:
        C_eff = methanol_kmol_h / co2_fresh_in_kmol_h * 100

    # CO2 + 3H2 -> CH3OH + H2O: the most methanol the fresh H2 can make is
    # H2/3. What this misses is exactly the purge loss.
    if methanol_kmol_h is None or not h2_fresh_in_kmol_h:
        H_eff = None
    else:
        H_eff = methanol_kmol_h / (h2_fresh_in_kmol_h / 3) * 100

    # Single-pass CO2 conversion across the reactor, recycle included.
    rin_flows = _read(lambda: rin.ComponentMolarFlowValue)
    rout_flows = _read(lambda: rout.ComponentMolarFlowValue)
    if (rin_flows is None or rout_flows is None
            or len(rin_flows) <= co2_index or len(rout_flows) <= co2_index):
        co2_conv = None
    else:
        co2_in, co2_out = rin_flows[co2_index], rout_flows[co2_index]
        if is_empty(co2_in) or is_empty(co2_out) or not co2_in:
            co2_conv = None
        else:
            co2_conv = (co2_in - co2_out) / co2_in * 100

    # EnergyValue comes back in kJ/s (kW); x3600 puts the duties in kJ/h.
    compressor_duty = _scale(_total(compressors, "EnergyValue"), 3600)
    expander_duty = _scale(_total(expanders, "EnergyValue"), 3600)

    # The whole component array arrives in one COM call, so it is present or
    # absent as a unit -- indexing it needs its own guard.
    purge_flows = _read(lambda: purge.ComponentMassFlowValue)
    if purge_flows is None or len(purge_flows) <= h2_index:
        hydrogen_kg_h = None
    else:
        h2 = purge_flows[h2_index]
        hydrogen_kg_h = None if is_empty(h2) else h2 * 3600

    duty = _scale(_read(lambda: reactor.HeatFlowValue), 3600)

    h2_lhv_kJ_h = _lhv_flow_kJ_h(h2_fresh_in)
    methanol_lhv_kJ_h = _lhv_flow_kJ_h(methanol)

    # Energy efficiency on an LHV basis, everything in kJ/h:
    #   in  = hydrogen feed heating value + compressor work
    #   out = methanol product heating value + expander work
    if None in (h2_lhv_kJ_h, compressor_duty, methanol_lhv_kJ_h, expander_duty):
        energy_efficiency = None
    else:
        energy_in = h2_lhv_kJ_h + compressor_duty
        energy_efficiency = (
            None if energy_in == 0
            else (methanol_lhv_kJ_h + expander_duty) / energy_in * 100
        )

    result = {

        # Inputs, in HYSYS internal units: kPa, C, m3. `set` is what was
        # asked for, `actual` is what HYSYS ended up with.
        "pressure_set_kPa": pressure,
        "temperature_set_C": temperature,
        "ratio_set": ratio,
        "volume_set_m3": volume,
        "recycle_fraction_set": recycle_fraction,
        "rxn8_ea_forward_set": rxn8_ea_forward,
        "rxn8_ea_reverse_set": rxn8_ea_reverse,

        # Actual values
        "pressure_actual_kPa": _read(lambda: pressure_cell.CellValue),
        "Tin_C": _read(lambda: rin.TemperatureValue),
        "Tout_C": _read(lambda: rout.TemperatureValue),
        "ratio_actual": _read(lambda: ratio_cell.CellValue),
        "volume_actual_m3": _read(lambda: reactor.TotalVolumeValue),
        "recycle_fraction_actual": _read(lambda: split.SplitsValue[1]),

        # Read back off the reaction, not echoed from the argument: these
        # persist across points, so `None` in the _set column does not mean
        # the reaction was at its original value.
        "rxn8_ea_forward_actual": _read(lambda: rxn8.ForwardActivationEnergy),
        "rxn8_ea_reverse_actual": _read(lambda: rxn8.ReverseActivationEnergy),

        # Outputs
        "C_efficiency_pct": C_eff,
        "H_efficiency_pct": H_eff,
        "co2_conversion_per_pass_pct": co2_conv,
        "energy_efficiency_pct": energy_efficiency,
        "methanol_kg_h": methanol_kg_h,
        "hydrogen_purge_kg_h": hydrogen_kg_h,
        "reactor_duty_kJ_h": duty,
        "compressor_duty_kJ_h": compressor_duty,
        "expander_duty_kJ_h": expander_duty,
        "h2_lhv_kJ_h": h2_lhv_kJ_h,
        "methanol_lhv_kJ_h": methanol_lhv_kJ_h,

        # Convergence -- filter on `converged` before using a sweep.
        # A rejected write counts as not converged: the solve that followed
        # it was not of the conditions in this row.
        "converged": report.ok and write_error is None,
        "write_error": write_error or "",
        "recycle_status": "; ".join(
            f"{name}:{CONV_NAMES.get(status, status)}({iters:g}/{mx:g})"
            for name, status, iters, mx in report.recycles),
        "convergence_detail": "" if report.ok else " | ".join(report.problems),

        # Timing: solve_time_s is the four solve stages summed, case_time_s
        # is the whole case including COM reads and the convergence walk.
        "solve_time_s": solve_time,
        "case_time_s": time.perf_counter() - case_start,

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if verbose:
        print(
            f"{case_name}: case in {result['case_time_s']:.2f} s "
            f"(solve {solve_time:.2f} s, "
            f"overhead {result['case_time_s'] - solve_time:.2f} s)"
            + ("" if result["converged"] else "  [NOT CONVERGED]")
        )

    return result


def initialise_hysys_objects(case):

    print("Initialising HYSYS objects")

    # No modal dialogs on an unattended run: "Calculated pressure gradient
    # ... is excessively high" arrives as a MODAL dialog mid-solve. Python is
    # blocked inside the COM call at that moment, so nobody is there to click
    # OK and the worker freezes until a human does. The True argument also
    # makes nested solvers STOP at their iteration limit instead of asking.
    # Edits the live preference set of this HYSYS instance -- re-enable under
    # File > Options in the GUI if dialogs are wanted back for interactive
    # work.
    try:
        case.Application.ChangePreferencesToMinimizePopupWindows(True)
        print("modal popups minimized for unattended run")
    except (AttributeError, com_error) as exc:
        print(f"[!] could not minimize popups -- "
              f"dialogs may need closing by hand: {exc}")

    flowsheet = case.Flowsheet
    streams = flowsheet.MaterialStreams
    ops = flowsheet.Operations
    solver = case.Solver

    rin = streams("RinV")
    rout = streams("RoutV")
    co2_fresh_in = streams("co2_1atm")
    h2_fresh_in = streams("h2")
    methanol = streams("Methanol")
    purge = streams("Purge")

    components = list(purge.FluidPackage.Components.Names)
    h2_index = components.index("Hydrogen")
    co2_index = components.index("CO2")

    reactor = ops("Reactor100")

    # Rate parameters live on the REACTION, not on the reactor -- the reactor
    # only holds a pointer to a reaction set (vandal2013, two Langmuir-
    # Hinshelwood reactions). ActiveReactions is positional, so match on the
    # name instead of trusting Item(1) to stay Rxn-8.
    reaction_set = reactor.ReactionSet
    active = reaction_set.ActiveReactions
    rxn8 = next((active.Item(i) for i in range(active.Count)
                 if active.Item(i).name == "Rxn-8"), None)
    if rxn8 is None:
        raise RuntimeError(
            f"Rxn-8 not in reaction set '{reaction_set.name}' -- "
            f"found {list(active.Names)}")

    split = ops("TEE-100")
    column = ops("Twp101")
    flare = ops("CRV-100")

    compressors = [ops(n) for n in
                   ("K-103-2", "K-110", "K-109", "K-100", "K-105", "K-103")]
    expanders = [ops(n) for n in ("Ex-101", "Ex-102")]

    pressure_cell = ops("MeOH Pressure").Cell("A1")
    ratio_cell = ops("co2:h2_ratio_equals_3").Cell("C2")

    return {
        "case_name": case.Name,
        "sim": CaseAdapter(case),
        "solver": solver,
        "rin": rin,
        "rout": rout,
        "co2_fresh_in": co2_fresh_in,
        "h2_fresh_in": h2_fresh_in,
        "methanol": methanol,
        "purge": purge,
        "h2_index": h2_index,
        "co2_index": co2_index,
        "reactor": reactor,
        "rxn8": rxn8,
        "split": split,
        "column": column,
        "flare": flare,
        "compressors": compressors,
        "expanders": expanders,
        "pressure_cell": pressure_cell,
        "ratio_cell": ratio_cell,
    }
