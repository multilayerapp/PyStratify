#!/bin/sh
# Create the benchmark environments once (benchmarks/envs/README.md). An environment that already exists is left
# alone, so this is safe to re-run; nothing here runs during benchmarks/verify_all.py.
#
#     sh benchmarks/envs/setup.sh              # every missing environment
#     sh benchmarks/envs/setup.sh meep tmmax   # some of them
#
# PYSTRATIFY_BENCH_HOME (default ~/.multilayer/bench) is where they live; CONDA and PYTHON3 name the conda executable
# and the base Python 3.13 the venvs are made from (defaults: /opt/anaconda3/bin/conda, /opt/anaconda3/bin/python3).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
HOME_DIR=${PYSTRATIFY_BENCH_HOME:-$HOME/.multilayer/bench}
CONDA=${CONDA:-/opt/anaconda3/bin/conda}
PYTHON3=${PYTHON3:-/opt/anaconda3/bin/python3}
mkdir -p "$HOME_DIR"

venv() {  # name: a venv from the base Python, pinned by <name>.txt
    "$PYTHON3" -m venv "$HOME_DIR/$1"
    "$HOME_DIR/$1/bin/python" -m pip install -q -r "$HERE/$1.txt"
}

conda_env() {  # name: a conda env pinned by <name>.conda.txt, then its pip packages
    "$CONDA" create -q -y -p "$HOME_DIR/$1" --file "$HERE/$1.conda.txt"
    grep -v -i '^smuthi==' "$HERE/$1.pip.txt" > "$HOME_DIR/$1.pip.tmp" || true
    "$HOME_DIR/$1/bin/python" -m pip install -q -r "$HOME_DIR/$1.pip.tmp"
    rm -f "$HOME_DIR/$1.pip.tmp"
}

for name in ${*:-bench smuthi meep ngsolve tmmax}; do
    if [ -x "$HOME_DIR/$name/bin/python" ]; then
        echo "$name: present ($HOME_DIR/$name)"
        continue
    fi
    echo "$name: creating $HOME_DIR/$name"
    case $name in
        bench|ngsolve|tmmax) venv "$name" ;;
        meep) conda_env meep ;;
        smuthi)
            # smuthi 2.2.4 builds only on Python <= 3.10 with NumPy < 2, a Fortran compiler and an OpenMP C compiler
            conda_env smuthi
            CC=${SMUTHI_CC:-gcc-16} LDSHARED="${SMUTHI_CC:-gcc-16} -bundle -undefined dynamic_lookup" \
                "$HOME_DIR/smuthi/bin/python" -m pip install -q --no-build-isolation "smuthi==2.2.4" ;;
        *) echo "unknown environment: $name" >&2; exit 2 ;;
    esac
    "$HOME_DIR/$name/bin/python" -m pip install -q --no-deps -e "$ROOT"
    "$HOME_DIR/$name/bin/python" -m pip install -q pytest
done

# MNPBEM (the MATLAB toolbox under GNU Octave) is not a Python environment: mnpbem_octave.py fetches and builds it
# into ~/.local/share/mnpbem once; it needs `brew install octave`.
if command -v octave-cli >/dev/null 2>&1 || command -v octave >/dev/null 2>&1; then
    [ -d "${MNPBEM_HOME:-$HOME/.local/share/mnpbem}/MNPBEM17-octave" ] || "$HOME_DIR/bench/bin/python" "$ROOT/benchmarks/mnpbem_octave.py"
else
    echo "GNU Octave not found: the MNPBEM figures and report will be skipped (brew install octave)"
fi
