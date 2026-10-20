"""Thin films, point dipole: PyStratify against meep.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh meep                # the meep environment (benchmarks/envs/README.md)              #
#   ~/.multilayer/bench/meep/bin/python benchmarks/pages/films_dipole_meep.py                                       #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-dipole-meep.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
films_dipole_meep.json (the curves the page draws, value for value) and films_dipole_meep.png (needs matplotlib).
"""

from __future__ import annotations

import json  # noqa: F401
import sys
from pathlib import Path

import numpy as np
from numpy import inf  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks", ROOT / "benchmarks" / "pages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pystratify as ps  # noqa: E402,F401
from pystratify import planar  # noqa: E402,F401

from _page import figure, run, series, sig, worst  # noqa: E402,F401


@figure("meep-axial-film", "Film on glass", "λ [µm]", "Γ / Γ₀", env="meep")
def meep_axial_film():
    import meep as mp
    from fdtd_meep import AXIAL_CASES, meep_ldos_axial

    case = "film: dipole 100 nm above 200 nm of n = 2.3 on glass"
    _, dimensions, n, source, names = AXIAL_CASES[case]
    mp.verbosity(0)
    coarse = 1 / np.linspace(0.85, 1.65, 17)
    theirs = {o: meep_ldos_axial(case, o, 100, wavelengths=coarse) / meep_ldos_axial(case, o, 100, False, wavelengths=coarse) for o in names}
    dense = 1 / np.linspace(0.85, 1.65, 161)
    ours = np.array([ps.solve_problem(ps.Problem("films", dimensions, n, w, ps.PointDipole(-source, "electric", 0), 1e-8))["total"][:2] for w in dense])
    out, refs, rel = [], [], []
    for i, o in enumerate(names):
        ref = np.full(dense.size, np.nan)
        ref[::10] = theirs[o]
        out.append(series(o, f"{o}, PyStratify", ours[:, i], "pystratify"))
        refs.append(series(f"{o}_ref", f"{o}, Meep", ref, "reference"))
        rel.append(np.max(np.abs(theirs[o] / ours[::10, i] - 1)))
    return dict(x=sig(dense), series=out + refs, worst=float(max(rel)), worst_label="largest relative difference (the grid's error)",
                note="Meep in cylindrical coordinates around the dipole (m = 0 perpendicular, m = 1 parallel), the film and substrate infinite "
                     "in r, 1 µm of PML; the local density of states over the same run without the film. The film is 20 pixels thick at 100 "
                     "px/µm and converges at about first order: worst 16.9% → 7.4% → 3.7% at 25, 50 and 100 px/µm (FDTD_MEEP.md); the "
                     "test asserts < 5% at 100 and closer than 25.")


if __name__ == "__main__":
    run(__file__)
