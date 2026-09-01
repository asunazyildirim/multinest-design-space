#!/usr/bin/env bash
#
# Launcher for magnus_tp_case_study.py under WSL / Ubuntu.
#
# Same reason as run_magnus_ladder.sh: the MAGNUS extension modules are found
# through PYTHONPATH and link against SUNDIALS, Gurobi and SNOPT through
# LD_LIBRARY_PATH, both set in ~/.bashrc, which an interactive shell sources
# and a script does not. This sets them explicitly.
#
# THE ORDER MATTERS. Unlike the ladder, this run cannot compute anything on
# its own -- the model is HYSYS, on the Windows side. Start that first:
#
#     Windows, with C:\HYSYS_cases\methanol_instance0.hsc open in HYSYS:
#         cd "...\Asu_Parallel"
#         python hysys_bridge_server.py
#
#     WSL, here:
#         cd "/mnt/c/Users/asuna/OneDrive/Masaüstü/MagnusCodes"
#         bash run_magnus_tp.sh --check                    # imports, ~1 s
#         bash run_magnus_tp.sh --backend surrogate        # wiring, ~1 s
#         bash run_magnus_tp.sh --backend surrogate --numlive 500
#         bash run_magnus_tp.sh                            # the real run
#
# The real run is 10-12 hours of serial HYSYS solves. Do the surrogate pass
# first: it exercises every line except the bridge and finishes in a second.
#
# To rehearse the bridge as well, without a HYSYS licence, run the server with
# --fake: it answers from the reference sweep instead of solving, so the whole
# system -- spool, renames, heartbeat, timeouts, output files -- runs end to
# end in a few minutes.
#
#     python hysys_bridge_server.py --fake        # Windows
#     bash run_magnus_tp.sh --numlive 200         # WSL
#
# Anything after the flags below is passed straight through, so --numlive,
# --numprop, --c-threshold, --tag and the rest all work unchanged.

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
# Interpreter -- unversioned .so files, so the first CPython that imports both
# pymc and magnus wins. PYTHON=/path/to/python overrides the search.
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
    echo "  * ModuleNotFoundError  -> PYTHONPATH, or the .so was never built" >&2
    echo "  * ImportError on a lib*.so -> LD_LIBRARY_PATH; run" >&2
    echo "    'ldd /opt/magnus/src/interface/magnus.so | grep \"not found\"'" >&2
    exit 1
fi

echo "python : $PY_BIN  ($("$PY_BIN" -V 2>&1))"
"$PY_BIN" - <<'PY'
import pymc, magnus, numpy
print(f"pymc   : {pymc.__file__}")
print(f"magnus : {magnus.__file__}")
print(f"numpy  : {numpy.__version__}")
PY

# ---------------------------------------------------------------------------
# The Windows-side server, checked before rather than after the imports: a
# missing heartbeat is the failure this run is most likely to hit, and it is
# cheap to catch here with a message that says what to start.
# ---------------------------------------------------------------------------
SPOOL="/mnt/c/HYSYS_cases/bridge"
for i in "$@"; do
    case "${PREV:-}" in --spool) SPOOL="$i" ;; esac
    PREV="$i"
done

case " $* " in
    *" --backend surrogate "*|*" --check "*|*" --help "*|*" -h "*) ;;
    *)
        if [ ! -f "$SPOOL/server.alive" ]; then
            echo
            echo "WARNING: no $SPOOL/server.alive" >&2
            echo "The Windows side is not running. Start it first:" >&2
            echo "    python hysys_bridge_server.py" >&2
            echo "(--backend surrogate needs no server.)" >&2
            echo >&2
        else
            echo "bridge : $SPOOL  (server heartbeat present)"
        fi
        ;;
esac
echo

case "${1:-}" in
    --check)
        echo "imports OK -- environment is good."
        exit 0
        ;;
esac

exec "$PY_BIN" "$HERE/magnus_tp_case_study.py" "$@"
