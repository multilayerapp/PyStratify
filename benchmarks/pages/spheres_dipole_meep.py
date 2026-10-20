"""Spheres, point dipole: PyStratify against meep.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh meep                # the meep environment (benchmarks/envs/README.md)              #
#   ~/.multilayer/bench/meep/bin/python benchmarks/pages/spheres_dipole_meep.py                                     #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-meep.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_dipole_meep.json (the curves the page draws, value for value) and spheres_dipole_meep.png (needs matplotlib).
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


@figure("meep-axial-sphere", "Dielectric sphere", "λ [µm]", "Γ / Γ₀", env="meep")
def meep_axial_sphere():
    import meep as mp
    from fdtd_meep import AXIAL_CASES, meep_ldos_axial

    case = "sphere n = 2, R = 150 nm, dipole 100 nm outside on its axis"
    _, dimensions, n, source, names = AXIAL_CASES[case]
    mp.verbosity(0)
    coarse = 1 / np.linspace(0.85, 1.65, 17)
    theirs = {o: meep_ldos_axial(case, o, 100, wavelengths=coarse) / meep_ldos_axial(case, o, 100, False, wavelengths=coarse) for o in names}
    dense = 1 / np.linspace(0.85, 1.65, 161)
    ours = np.array([ps.solve_problem(ps.Problem("spheres", dimensions, n, w, ps.PointDipole(source, "electric", 1), 1e-8))["total"][:2] for w in dense])
    out, refs, rel = [], [], []
    for i, o in enumerate(names):
        ref = np.full(dense.size, np.nan)
        ref[::10] = theirs[o]
        out.append(series(o, f"{o}, PyStratify", ours[:, i], "pystratify"))
        refs.append(series(f"{o}_ref", f"{o}, Meep", ref, "reference"))
        rel.append(np.max(np.abs(theirs[o] / ours[::10, i] - 1)))
    return dict(x=sig(dense), series=out + refs, worst=float(max(rel)), worst_label="largest relative difference (the grid's error)",
                note="Meep in cylindrical coordinates (m = 0 for the radial dipole, m = 1 for the tangential), a Gaussian pulse and its "
                     "local density of states at the dipole, over the same run without the sphere so the grid's own dispersion cancels. "
                     "A curved surface on a square grid converges in an oscillating way: 5.5% at 25 px/µm, 1.6–1.7% at 50–100 (the test).")


if __name__ == "__main__":
    run(__file__)
