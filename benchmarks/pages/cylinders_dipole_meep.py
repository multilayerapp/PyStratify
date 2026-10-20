"""Cylinders, point dipole: PyStratify against meep.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh meep                # the meep environment (benchmarks/envs/README.md)              #
#   ~/.multilayer/bench/meep/bin/python benchmarks/pages/cylinders_dipole_meep.py                                   #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-dipole-meep.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
cylinders_dipole_meep.json (the curves the page draws, value for value) and cylinders_dipole_meep.png (needs matplotlib).
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


@figure("meep-cylinder-fibre", "Step-index fibre", "λ [µm]", "Γ / Γ₀", env="meep")
def meep_cylinder_fibre():
    import meep as mp
    from fdtd_meep import CASES, ORIENTATIONS, meep_ldos

    case = "fibre n = 2, R = 150 nm, dipole 100 nm outside"
    radii, n, source = CASES[case]
    mp.verbosity(0)
    coarse = 1 / np.linspace(0.85, 1.65, 17)
    theirs = {o: meep_ldos(case, o, 60, wavelengths=coarse) / meep_ldos(case, o, 60, False, wavelengths=coarse) for o in ORIENTATIONS}
    dense = 1 / np.linspace(0.85, 1.65, 161)
    ours = np.array([ps.solve_problem(ps.Problem("cylinders", radii, n, w, ps.PointDipole(source, "electric"), 1e-6))["total"][:3] for w in dense])
    out, refs, rel = [], [], []
    for i, o in enumerate(ORIENTATIONS):
        ref = np.full(dense.size, np.nan)
        ref[::10] = theirs[o]
        out.append(series(o, f"{o}, PyStratify", ours[:, i], "pystratify"))
        refs.append(series(f"{o}_ref", f"{o}, Meep", ref, "reference"))
        rel.append(np.max(np.abs(theirs[o] / ours[::10, i] - 1)))
    return dict(x=sig(dense), series=out + refs, worst=float(max(rel)), worst_label="largest relative difference (the grid's error)",
                note="Meep in 3D: the fibre runs through the perfectly matched layers, a 1.8 µm cell plus 1 µm of PML per side, two mirror "
                     "planes; a Gaussian pulse and the local density of states at the dipole, over the same run without the fibre so the "
                     "grid's own dispersion cancels. TE₀₁ and TM₀₁ are cut off at 0.678 µm. The difference shrinks with the grid, "
                     "6.9% → 3.2% → 1.7% → 0.74% at 20–60 px/µm (FDTD_MEEP.md); the test runs 20 and 30.")


if __name__ == "__main__":
    run(__file__)
