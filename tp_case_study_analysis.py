"""(T, P) feasibility case study on the constraint set the study uses:
methanol production, carbon efficiency, energy efficiency.

Reproduces every number in tp_case_study_proposal.md, and the figure. Reads
the existing 130-point T-P sweep -- no HYSYS, no new solves. Run from the
repository root:

    python tp_case_study_analysis.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

SWEEP = os.path.join("output", "sensitivity_results_20260731_174218.csv")

# Chosen thresholds. The admissible band of each is set by the process; only
# the position inside it is a choice (see proposal, section 4).
C_MIN = 92.0      # carbon efficiency                        [%]
E_MIN = 81.3      # energy efficiency                        [%]
# The production target is the same constraint as C_MIN -- MeOH kg/h is
# proportional to C_eff because the CO2 fresh feed is fixed. Stated for the
# write-up, not imposed a second time.
MEOH_PER_CEFF = 208.55


def load():
    d = pd.read_csv(SWEEP)
    d = d[d.converged == True].copy()          # noqa: E712 -- pandas mask
    d["Wspec"] = (d.compressor_duty_kJ_h / 3600.0) / (d.methanol_kg_h / 1000.0)
    return d


def region_map(d, mask):
    """Feasible points laid out as the T-P grid, high pressure at the top."""
    return d.assign(f=mask.astype(int)).pivot_table(
        index="pressure_set_kPa", columns="temperature_set_C",
        values="f").sort_index(ascending=False)


def edges_touched(g):
    return (int(g.iloc[0].any()) + int(g.iloc[-1].any())
            + int(g.iloc[:, 0].any()) + int(g.iloc[:, -1].any()))


def main():
    d = load()
    print(f"{len(d)} converged points  "
          f"T {d.temperature_set_C.min():.0f}-{d.temperature_set_C.max():.0f} C  "
          f"P {d.pressure_set_kPa.min()/100:.0f}-{d.pressure_set_kPa.max()/100:.0f} bar")

    # --- 1. three constraints, two of them the same one -------------------
    ratio = d.methanol_kg_h / d.C_efficiency_pct
    print("\n-- methanol production vs carbon efficiency --")
    print(f"  MeOH[kg/h] / C_eff[%]  =  {ratio.min():.4f} .. {ratio.max():.4f}"
          f"   (spread {(ratio.max()-ratio.min())/ratio.mean()*1e6:.0f} ppm)")
    print("  The CO2 fresh feed is fixed over this box, so the two are the")
    print("  SAME constraint. Whichever threshold is tighter binds; the other")
    print("  can never cut a point the tighter one admits. No choice of")
    print("  thresholds changes this -- it is arithmetic, not a preference.")
    print(f"  C_eff >= {C_MIN:g} %  <=>  MeOH >= {MEOH_PER_CEFF*C_MIN:.0f} kg/h")

    # --- 2. metric behaviour ---------------------------------------------
    print("\n-- metric behaviour over the box --")
    for m, unit in [("C_efficiency_pct", "%"), ("energy_efficiency_pct", "%"),
                    ("methanol_kg_h", "kg/h")]:
        c = d[["temperature_set_C", "pressure_set_kPa", m]].corr()[m]
        print(f"  {m:22s} {d[m].min():8.2f} - {d[m].max():8.2f} {unit:5s}"
              f"  corr with T {c.temperature_set_C:+.3f}"
              f"  with P {c.pressure_set_kPa:+.3f}")
    print(f"\n  corr(C_eff, E_eff) = "
          f"{d[['C_efficiency_pct', 'energy_efficiency_pct']].corr().iloc[0, 1]:+.3f}")
    print("  Both metrics ridge in T (peak 240-260 C) but split in P:")
    print("  C_eff rises with pressure, E_eff falls. That split is what")
    print("  closes the region -- the correlation alone does not decide it.")

    # --- 3. the problem ---------------------------------------------------
    m_c = d.C_efficiency_pct >= C_MIN
    m_e = d.energy_efficiency_pct >= E_MIN
    feas = m_c & m_e
    g = region_map(d, feas)

    print(f"\n-- C_eff >= {C_MIN:g} % (MeOH >= {MEOH_PER_CEFF*C_MIN:.0f} kg/h), "
          f"E_eff >= {E_MIN:g} % --")
    print(f"  feasible                {feas.sum():3d}/{len(d)} = {feas.mean()*100:.1f} %")
    print(f"  excluded by C_eff only  {(~m_c & m_e).sum():3d}")
    print(f"  excluded by E_eff only  {(m_c & ~m_e).sum():3d}")
    print(f"  box edges touched       {edges_touched(g)}")

    print("\n  P bar / T C   " + " ".join("%3d" % c for c in g.columns))
    for p, row in g.iterrows():
        print("  %9d     " % (p // 100)
              + " ".join("  X" if v else "  ." for v in row))

    # --- 4. robustness ----------------------------------------------------
    CS = (91.0, 91.5, 92.0, 92.5, 93.0)
    ES = (81.0, 81.2, 81.3, 81.4, 81.5)
    print("\n-- robustness: feasible % of box --")
    print(pd.DataFrame(
        [[((d.C_efficiency_pct >= c) & (d.energy_efficiency_pct >= e)).mean()*100
          for e in ES] for c in CS],
        index=[f"C>={c}" for c in CS], columns=[f"E>={e}" for e in ES]).round(1))

    print("\n-- both-active check: (C-only, E-only) excluded, and edges --")
    for c in CS:
        cells = []
        for e in ES:
            a, b = d.C_efficiency_pct >= c, d.energy_efficiency_pct >= e
            f = a & b
            cells.append(f"{(~a & b).sum():2d}/{(a & ~b).sum():2d}/"
                         f"{edges_touched(region_map(d, f))}")
        print(f"  C>={c:5.1f}  " + "  ".join(cells))

    # --- 5. what the region delivers --------------------------------------
    fs = d[feas]
    print("\n-- over the feasible set --")
    print(f"  MeOH  {fs.methanol_kg_h.min():8.0f} - {fs.methanol_kg_h.max():8.0f} kg/h")
    print(f"  C_eff {fs.C_efficiency_pct.min():8.2f} - {fs.C_efficiency_pct.max():8.2f} %")
    print(f"  E_eff {fs.energy_efficiency_pct.min():8.2f} - "
          f"{fs.energy_efficiency_pct.max():8.2f} %")
    print(f"  T     {fs.temperature_set_C.min():8.0f} - "
          f"{fs.temperature_set_C.max():8.0f} C")
    print(f"  P     {fs.pressure_set_kPa.min()/100:8.0f} - "
          f"{fs.pressure_set_kPa.max()/100:8.0f} bar")

    # --- 6. optional third, genuinely independent, constraint -------------
    print("\n-- if a THIRD active constraint is wanted --")
    print("   specific compression work is the only cheap output that is not")
    print("   a restatement of the other two:")
    print(f"   corr(Wspec, C_eff) = "
          f"{d[['Wspec', 'C_efficiency_pct']].corr().iloc[0, 1]:+.3f}   "
          f"corr(Wspec, E_eff) = "
          f"{d[['Wspec', 'energy_efficiency_pct']].corr().iloc[0, 1]:+.3f}")
    for c, e, w in [(90.75, 81.3, 340), (91.0, 81.3, 350)]:
        a = d.C_efficiency_pct >= c
        b = d.energy_efficiency_pct >= e
        x = d.Wspec <= w
        f = a & b & x
        print(f"   C>={c}, E>={e}, W<={w}: {f.sum():3d}/{len(d)} = {f.mean()*100:4.1f} %"
              f"  | each alone removes "
              f"{(b & x).sum()-f.sum()}/{(a & x).sum()-f.sum()}/{(a & b).sum()-f.sum()}"
              f"  edges {edges_touched(region_map(d, f))}")

    # --- figure -----------------------------------------------------------
    T = np.array(sorted(d.temperature_set_C.unique()), float)
    P = np.array(sorted(d.pressure_set_kPa.unique()), float)
    Tf, Pf = np.linspace(T[0], T[-1], 241), np.linspace(P[0], P[-1], 301)
    TT, PP = np.meshgrid(Tf, Pf, indexing="ij")
    pts = np.stack([TT.ravel(), PP.ravel()], -1)

    def interp(m):
        grid = d.pivot_table(index="temperature_set_C",
                             columns="pressure_set_kPa", values=m).values
        return RegularGridInterpolator((T, P), grid)(pts).reshape(TT.shape)

    Ce, Ee = interp("C_efficiency_pct"), interp("energy_efficiency_pct")

    fig, ax = plt.subplots(1, 3, figsize=(17.4, 5.0), constrained_layout=True)

    # Each panel carries a colourbar on its right, so the default spacing
    # puts a colourbar label straight against the next panel's y-axis title.
    # wspace opens the gap between the panel groups; w_pad keeps the outer
    # margins tight rather than growing with it.
    fig.get_layout_engine().set(wspace=0.09, w_pad=0.06)

    c = ax[0].contourf(TT, PP / 100, Ce, levels=14, cmap="viridis")
    fig.colorbar(c, ax=ax[0], label="Carbon efficiency [%]")
    ax[0].contour(TT, PP / 100, Ce, levels=[C_MIN], colors="w", linewidths=2.5)
    ax[0].set_title("(a) Carbon efficiency", fontsize=11)

    c = ax[1].contourf(TT, PP / 100, Ee, levels=14, cmap="cividis")
    fig.colorbar(c, ax=ax[1], label="Energy efficiency [%]")
    ax[1].contour(TT, PP / 100, Ee, levels=[E_MIN], colors="w", linewidths=2.5)
    ax[1].set_title("(b) Energy efficiency", fontsize=11)

    ax[2].contourf(TT, PP / 100, (Ce >= C_MIN) & (Ee >= E_MIN),
                   levels=[.5, 1.5], colors=["#4c9f70"], alpha=.6)
    ax[2].contour(TT, PP / 100, Ce, levels=[C_MIN], colors="#c0392b", linewidths=2.2)
    ax[2].contour(TT, PP / 100, Ee, levels=[E_MIN], colors="#2c6fbb", linewidths=2.2)
    ax[2].scatter(d.temperature_set_C[~feas], d.pressure_set_kPa[~feas] / 100,
                  s=12, c="0.55", marker="x", lw=.9)
    ax[2].scatter(d.temperature_set_C[feas], d.pressure_set_kPa[feas] / 100,
                  s=26, c="#14532d", marker="o", edgecolors="w", lw=.6, zorder=5)
    ax[2].plot([], [], color="#c0392b", lw=2.2, label="$g_1$  carbon / production")
    ax[2].plot([], [], color="#2c6fbb", lw=2.2, label="$g_2$  energy")
    ax[2].legend(loc="upper left", fontsize=8, framealpha=.9)
    ax[2].set_title("(c) Nominal feasible region", fontsize=11)

    for a in ax:
        a.set_xlabel("Temperature [°C]")
        a.set_ylabel("Pressure [bar]")
        a.set_xlim(T[0], T[-1])
        a.set_ylim(P[0] / 100, P[-1] / 100)

    # No suptitle: the panel letters carry the structure and the caption
    # names the problem, which is where a paper figure puts it.
    fig.savefig("tp_proposed_case_study.png", dpi=140)
    print("\nwrote tp_proposed_case_study.png")


if __name__ == "__main__":
    main()
