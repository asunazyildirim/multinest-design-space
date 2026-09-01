from multiprocessing import Pool
import os
import time
import itertools
from datetime import datetime
import pandas as pd

from hysys_connection import get_open_hysys_case

import pythoncom

from simulation_runner import run_case, initialise_hysys_objects

CASE_FOLDER = r"C:\HYSYS_cases"


# # --------------------------------------------------
# # Sensitivity points
# # For the first test, use 20 identical points
# # Replace later with real T/P combinations
# # --------------------------------------------------

# SENSITIVITY_POINTS = [
#     {
#         "pressure": 9000,
#         "temperature": 270,
#         "ratio": 3.0,
#         "volume": 19
#     }
#     for _ in range(20)
# ]

# # --------------------------------------------------
# # Sensitivity points
# # 
# # --------------------------------------------------

# Stage 1 of the design-space characterization: T-P full factorial at the
# paper's nominal ratio/volume/recycle. 10 x 13 = 130 points.
# TEMPERATURES = range(210, 301, 10)      # C
# PRESSURES = range(5000, 11001, 500)     # kPa
# RATIOS = [3.0]
# VOLUMES = [65]                          # m3
# RECYCLES = [None]                       # None -> run_case default (0.99)

# Stage 2a: ratio x recycle at the fixed T-P operating point picked from the
# stage-1 map (production ridge, mid-high pressure). 7 x 9 = 63 points.
# This is the slice where C/energy efficiency should decouple from methanol
# production: the fresh feed and the purge both move here.
TEMPERATURES = [260]                                     # C
PRESSURES = [9000]                                       # kPa
RATIOS = [round(2.4 + 0.2 * i, 1) for i in range(7)]     # 2.4 .. 3.6
VOLUMES = [65]                                           # m3
RECYCLES = [round(0.95 + 0.005 * i, 3) for i in range(9)]  # 0.95 .. 0.99


SENSITIVITY_POINTS = [
    {
        "pressure": p,
        "temperature": t,
        "ratio": r,
        "volume": v,
        "recycle_fraction": rf
    }
    for r, t, p, v, rf in itertools.product(
        RATIOS,
        TEMPERATURES,
        PRESSURES,
        VOLUMES,
        RECYCLES
    )
]

# ---------------------------   -----------------------
# Worker allocation
# One HYSYS instance per worker
# --------------------------------------------------
# Real parallel run: copies methanol_instance{0..3}.hsc open in 4 HYSYS
# windows under CASE_FOLDER.
CASE_PATHS = [
     os.path.join(CASE_FOLDER, f"methanol_instance{i}.hsc")
    for i in range(4)
 ]

# Smoke run: single worker against the case that is open right now.
# CASE_PATHS = [
#     r"c:\Users\any25\Imperial College London"
#     r"\Bernardi, Andrea - Asu_Yildirim\methanol.hsc"
# ]

N_Workers = len(CASE_PATHS)

## round robin allocation
chunks = [
    SENSITIVITY_POINTS[i::N_Workers]
    for i in range(N_Workers)
]

## structured allocation
# chunk_size = len(SENSITIVITY_POINTS) // N_workers

# chunks = [
#     SENSITIVITY_POINTS[i*chunk_size:(i+1)*chunk_size]
#     for i in range(N_workers - 1)
# ]

# chunks.append(
#     SENSITIVITY_POINTS[(N_workers-1)*chunk_size:]
# )

WORKERS = [
    {
        "case": CASE_PATHS[i],
        "points": chunks[i]
    }
    for i in range(N_Workers)
]

# # --------------------------------------------------
# # One complete T/P matrix per ratio
# # --------------------------------------------------
# PRESSURES = range(7000, 11001, 125)
# TEMPERATURES = range(250, 291, 2)

# RATIOS = [3.0, 3.2, 3.4, 3.6]

# VOLUME = 19

# SENSITIVITY_POINTS = []

# for ratio in RATIOS:

#     points = [
#         {
#             "pressure": p,
#             "temperature": t,
#             "ratio": ratio,
#             "volume": VOLUME
#         }
#         for t, p in itertools.product(
#             TEMPERATURES,    
#             PRESSURES
#         )
#     ]

#     SENSITIVITY_POINTS.append(points)



# WORKERS = [
#     {
#         "case": os.path.join(
#             CASE_FOLDER,
#             f"methanol_instance{i}.hsc"
#         ),
#         "points": SENSITIVITY_POINTS[i]
#     }
#     for i in range(4)
# ]

def worker(config):

    case_path = config["case"]
    points = config["points"]

    name = os.path.basename(case_path)

    start = time.time()

    print(f"Starting {name}")


    pythoncom.CoInitialize()

    case = None

    try:

        # -----------------------------------
        # Connect to already open HYSYS case
        # -----------------------------------

        case = get_open_hysys_case(case_path)

        print(f"{name}: connected")
        print(f"{name}: case name = {case.Name}")
        print(
            f"{name}: streams = {case.Flowsheet.MaterialStreams.Count}"
        )


        print(f"{name}: caching HYSYS objects")

        objects = initialise_hysys_objects(case)
        objects["case_name"] = name
        time.sleep(1)

        print(f"{name}: objects cached")


        print(f"{name}: starting sensitivity loop")


        results = []


        for i, point in enumerate(points):

            print(
                f"{name}: sensitivity point {i+1}/{len(points)}"
            )


            # Point keys match run_case's parameters, so a new manipulated
            # variable (e.g. recycle_fraction) only needs adding to the grid.
            result = run_case(objects, **point)


            # Store execution information. The point's variables are NOT
            # copied in again -- run_case already records them as the
            # *_set columns, and duplicating them under a second name
            # ("pressure" next to "pressure_set_kPa") invites analysing
            # the wrong one.
            result["case"] = name
            result["execution_order"] = i + 1

            results.append(result)


            print(
                f"{name}: finished point {i+1}"
            )


        elapsed = time.time() - start

        print(
            f"{name}: completed {len(results)} "
            f"points in {elapsed:.1f} s"
        )


        return results


    except Exception as e:

        print(
            f"FAILED {name}: {e}"
        )

        return [
            {
                "case": name,
                "error": str(e)
            }
        ]


    finally:

        # DO NOT close HYSYS here
        # We did not create it

        pythoncom.CoUninitialize()

        print(
            f"Finished {name}"
        )

if __name__ == "__main__":

    
    print(
        f"Total sensitivity points: "
        f"{len(SENSITIVITY_POINTS)}"
    )

    overall_start = time.time()

    with Pool(processes=N_Workers) as pool:

        results = pool.map(
            worker,
            WORKERS
        )


    print("\nFINAL RESULTS")


    total = 0
    all_results = []

    for worker_results in results:

        print("\nWorker results:")

        for r in worker_results:
            print(r)

            if "error" not in r:
                total += 1
                all_results.append(r)

    elapsed = time.time() - overall_start

    print(
        f"\nCompleted simulations: {total}"
    )

    print(
        f"Total elapsed time: {elapsed:.1f} s ({elapsed/60:.2f} min)"
    )


    # -----------------------------
    # Save results to CSV
    # -----------------------------

    if all_results:

        # Anchored to this script's folder, NOT the process working
        # directory: VS Code's Run can launch with cwd elsewhere (e.g. its
        # own install dir), which is where one sweep's CSV ended up.
        output_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = os.path.join(
            output_folder,
            f"sensitivity_results_{timestamp}.csv"
        )

        df = pd.DataFrame(all_results)

        # Fixed column order: what was asked for, what HYSYS ended up with,
        # the outputs, then the trust/bookkeeping columns. Anything new that
        # run_case starts returning lands at the end instead of vanishing.
        COLUMNS = [
            # inputs as set
            "temperature_set_C", "pressure_set_kPa", "ratio_set",
            "volume_set_m3", "recycle_fraction_set",
            # inputs as HYSYS ended up with them
            "Tin_C", "Tout_C", "pressure_actual_kPa", "ratio_actual",
            "volume_actual_m3", "recycle_fraction_actual",
            # outputs
            "C_efficiency_pct", "H_efficiency_pct",
            "co2_conversion_per_pass_pct", "energy_efficiency_pct",
            "methanol_kg_h", "hydrogen_purge_kg_h",
            "reactor_duty_kJ_h", "compressor_duty_kJ_h",
            "expander_duty_kJ_h", "h2_lhv_kJ_h", "methanol_lhv_kJ_h",
            # convergence -- filter on `converged` before using the sweep
            "converged", "write_error", "recycle_status",
            "convergence_detail",
            # bookkeeping
            "case", "execution_order", "solve_time_s", "case_time_s",
            "timestamp",
        ]
        ordered = [c for c in COLUMNS if c in df.columns]
        extras = [c for c in df.columns if c not in COLUMNS]
        df = df[ordered + extras]

        # Rows in grid order (workers interleave them), not execution order.
        df = df.sort_values(
            by=["temperature_set_C", "pressure_set_kPa",
                "ratio_set", "volume_set_m3", "recycle_fraction_set"]
        )

        df.to_csv(
            filename,
            index=False
        )

        print(
            f"\nCSV saved to: {filename}"
        )

        n_bad = int((~df["converged"].astype(bool)).sum())
        print(f"converged: {len(df) - n_bad}/{len(df)}"
              + (f"  -- {n_bad} row(s) carry converged=False, "
                 f"filter them out before analysis" if n_bad else ""))