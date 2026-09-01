#!/usr/bin/env bash
#
# Launcher for magnus_gtcd_case.py under WSL / Ubuntu.
#
# Same reason run_magnus_ladder.sh exists: the MAGNUS extension modules are
# plain .so files found through PYTHONPATH, linked against SUNDIALS, Gurobi and
# SNOPT libraries found through LD_LIBRARY_PATH. Both live in ~/.bashrc, which
# an INTERACTIVE shell sources and a script does not -- so the same command
# that works pasted into a terminal fails as `bash file.sh` with a bare
# ModuleNotFoundError. This sets them explicitly and checks the imports before
# starting a run.
#
# Usage, from a WSL terminal (VS Code: Ctrl+Shift+P -> "WSL: Connect to WSL"):
#
#     cd "/mnt/c/Users/asuna/OneDrive/Masaüstü/MagnusCodes"
#     bash run_magnus_gtcd.sh --check          # imports only, ~1 s
#     bash run_magnus_gtcd.sh --selfcheck      # verify the DAG transcription
#     bash run_magnus_gtcd.sh                  # nominal run, N_L = 5000
#     bash run_magnus_gtcd.sh --N_L 1500 --seed 11
#     bash run_magnus_gtcd.sh --sigma 0.05 --n-theta 100
#
# Run --selfcheck once after any edit to the model: it compares the DAG
# against the closed form and the feasible fraction against the published
# 7.4%, which is the only thing standing between a mistyped exponent and a
# run that finishes and is wrong.
#
# Anything after --check is passed straight to magnus_gtcd_case.py.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Environment -- mirrors the exports in ~/.bashrc and /opt/magnus_installer.md.
# ---------------------------------------------------------------------------
export SUNDIALS_HOME="${SUNDIALS_HOME:-/opt/sundials-7.4.0}"
export GUROBI_HOME="${GUROBI_HOME:-/opt/gurobi1203/linux64}"
export SNOPT_HOME="${SNOPT_HOME:-/opt/snopt77}"

export PYTHONPATH="${PYTHONPATH:-}:/opt/mcpp/src/pymc:/opt/cronos/src/interface:/opt/canon/src/interface:/opt/magnus/src/interface"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/x86_64-linux-gnu:${SUNDIALS_HOME}/lib:${GUROBI_HOME}/lib:${SNOPT_HOME}/lib:${HOME}/MultiNest/lib"

# ---------------------------------------------------------------------------
# Interpreter -- the .so files are unversioned, so they load into whichever
# CPython finds them and fail on an ABI mismatch rather than being skipped.
# PYTHON=/path/to/python overrides the search.
# ---------------------------------------------------------------------------
probe() {
    "$1" - <<'PY' >/dev/null 2>&1
import pymc, magnus
PY
}

pick_python() {
    if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
    for cand in "$HOME/py3.13-env/bin/python" \
                /usr/bin/python3.13 /usr/bin/python3 python3; do
        command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ] || continue
        if probe "$cand"; then echo "$cand"; return; fi
    done
    echo ""
}

PY_BIN="$(pick_python)"

if [ -z "$PY_BIN" ]; then
    echo "ERROR: no interpreter could import both pymc and magnus." >&2
    echo >&2
    echo "PYTHONPATH=$PYTHONPATH" >&2
    echo >&2
    echo "Diagnose with, for each candidate interpreter:" >&2
    echo "    <python> -c 'import pymc'" >&2
    echo "    <python> -c 'import magnus'" >&2
    echo >&2
    echo "Two failures are common and look alike:" >&2
    echo "  * ModuleNotFoundError  -> PYTHONPATH is wrong, or the .so was" >&2
    echo "    never built (check /opt/mcpp/src/pymc/pymc.so and" >&2
    echo "    /opt/magnus/src/interface/magnus.so exist)." >&2
    echo "  * ImportError about a missing lib*.so -> LD_LIBRARY_PATH; run" >&2
    echo "    'ldd /opt/magnus/src/interface/magnus.so | grep \"not found\"'" >&2
    echo "    to see which one." >&2
    exit 1
fi

echo "python : $PY_BIN  ($("$PY_BIN" -V 2>&1))"
"$PY_BIN" - <<'PY'
import pymc, magnus, numpy
print(f"pymc   : {pymc.__file__}")
print(f"magnus : {magnus.__file__}")
print(f"numpy  : {numpy.__version__}")
PY
echo

if [ "${1:-}" = "--check" ]; then
    echo "imports OK -- environment is good."
    exit 0
fi

exec "$PY_BIN" "$HERE/magnus_gtcd_case.py" "$@"
