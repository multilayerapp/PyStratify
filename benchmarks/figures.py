"""The figures of multilayer.app's Verification pages, written to docs/figures/<name>.json.

    pip install -e ".[docs,benchmarks]"
    python benchmarks/figures.py                  # every figure
    python benchmarks/figures.py NAME ...         # some of them
    python benchmarks/figures.py --env ENV        # the figures of one environment (bench, smuthi, meep, ngsolve, tmmax)
    python benchmarks/figures.py --list           # every figure and its environment
    python benchmarks/figures.py --meta [NAME]    # titles, labels, script and setup refreshed, nothing recomputed

    python benchmarks/verify_all.py               # the whole study, every figure in its own environment

Each Verification page is computed by one script, benchmarks/pages/<page>.py, which a reader runs from a clone. This
file imports those scripts and writes each figure as its script's own `record` returns it (full precision, its RMSE),
adding what the site shows beside the plot: the calculator setup the page freezes and sketches, its one-line summary
or, without a setup, its aside (benchmarks/figure_setups.py). Nothing here is asserted; the tests do that.

A figure needing a package that is not installed (treams, scattnlay, pyGDM2, smuthi, Meep, NGSolve) is skipped with a
note; the MNPBEM figures need GNU Octave (`mnpbem_octave.py`). The environments run one at a time (envs/README.md):
smuthi wants Python <= 3.10 with NumPy < 2, Meep its conda build, pyGDM2 NumPy < 2.4.

`docs/verification.md` names which matrix row each figure belongs to; `python -m pystratify.docs build` carries the
JSON into `docs/manifest.json`, which the site draws.
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "benchmarks", ROOT / "benchmarks" / "pages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _page  # noqa: E402
import films_plane_wave_pymoosh as pymoosh_page  # noqa: E402
from figure_setups import ASIDES, SETUPS, SUMMARIES, pymoosh_setups  # noqa: E402

OUT = ROOT / "docs" / "figures"
PAGE_MODULES = (
    "cylinders_dipole_consistency",
    "cylinders_dipole_meep",
    "cylinders_dipole_ngsolve",
    "cylinders_plane_wave_consistency",
    "cylinders_plane_wave_treams",
    "films_dipole_consistency",
    "films_dipole_meep",
    "films_dipole_pymoosh",
    "films_dipole_smuthi",
    "films_plane_wave_consistency",
    "films_plane_wave_pyelli",
    "films_plane_wave_tmm",
    "films_plane_wave_tmmax",
    "spheres_dipole_consistency",
    "spheres_dipole_meep",
    "spheres_dipole_mnpbem",
    "spheres_dipole_pygdm2",
    "spheres_dipole_smuthi",
    "spheres_dipole_treams",
    "spheres_plane_wave_bhmie",
    "spheres_plane_wave_consistency",
    "spheres_plane_wave_mnpbem",
    "spheres_plane_wave_pygdm2",
    "spheres_plane_wave_scattnlay",
    "spheres_plane_wave_treams",
)

# name -> (environment, the figure's record computed, its fields that need no computing)
FIGURES = {}
for _name, _title, _x, _y, _fn in pymoosh_page.FIGURES:  # it stands alone: its own record, no environment option
    FIGURES[_name] = ("bench", partial(pymoosh_page.record, _name, _title, _x, _y, _fn),
                      dict(id=_name, title=_title, x_label=_x, y_label=_y, log=False, script=pymoosh_page.SCRIPT,
                           function=_fn.__name__))
for _module in PAGE_MODULES:
    __import__(_module)
    for _name, _title, _x, _y, _log, _env, _fn in _page.FIGURES[_module]:
        _script = f"benchmarks/pages/{_module}.py"
        FIGURES[_name] = (_env, partial(_page.record, _name, _title, _x, _y, _fn, _log, _script),
                          dict(id=_name, title=_title, x_label=_x, y_label=_y, log=_log, script=_script,
                               function=_fn.__name__))
SETUPS = {**SETUPS, **pymoosh_setups()}


def site(name):
    """What a figure carries for the site beside its data: its setup, its one-sentence summary, its aside."""
    out = {"calculator": SETUPS.get(name), "summary": SUMMARIES.get(name), "aside": ASIDES.get(name)}
    return {k: v for k, v in out.items() if v}


def write(name, out):
    (OUT / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(names=None, env=None):
    OUT.mkdir(exist_ok=True)
    written = []
    for name, (figure_env, compute, _) in FIGURES.items():
        if (names and name not in names) or (env and figure_env != env):
            continue
        t = time.time()
        try:
            out = {**compute(), **site(name)}
        except ImportError as error:
            print(f"skip {name}: {error}", flush=True)
            continue
        write(name, out)
        written.append(name)
        print(f"{name}: {len(out['x'])} points, {len(out['series'])} series, worst {out['worst']:.2e} "
              f"({time.time() - t:.1f} s)", flush=True)
    return written


def refresh(names):
    """Each figure's title, labels, script and site fields rewritten over its stored data, nothing recomputed."""
    for name in names:
        path = OUT / f"{name}.json"
        out = {k: v for k, v in json.loads(path.read_text()).items() if k not in ("calculator", "summary", "aside")}
        write(name, {**out, **FIGURES[name][2], **site(name)})
        print(f"{name}: meta refreshed", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--list"]:
        for name, (env, _, _) in FIGURES.items():
            print(name, env)
    elif args[:1] == ["--env"]:
        build(env=args[1])
    elif args[:1] == ["--meta"]:
        refresh(args[1:] or list(FIGURES))
    else:
        build(args or None)
