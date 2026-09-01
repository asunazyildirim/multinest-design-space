import os

from hysyspy import HysysPy

# SaveCopyAs does NOT create the target folder: without this the copies end
# up stranded as .tmp files in %TEMP% and no error is shown.
FOLDER = r"C:\HYSYS_cases"
os.makedirs(FOLDER, exist_ok=True)

sim = HysysPy(casename="methanol")

sim.make_parallel_copies(
    n=4,
    folder=FOLDER + "\\"
)
