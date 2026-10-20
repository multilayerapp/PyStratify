"""Cylinders, point dipole: PyStratify against ngsolve.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh ngsolve                # the ngsolve environment (benchmarks/envs/README.md)        #
#   ~/.multilayer/bench/ngsolve/bin/python benchmarks/pages/cylinders_dipole_ngsolve.py                             #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-dipole-ngsolve.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
cylinders_dipole_ngsolve.json (the curves the page draws, value for value) and cylinders_dipole_ngsolve.png (needs matplotlib).
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
from _shared import at  # noqa: E402


@figure("ngsolve-cylinder-fibre", "Step-index fibre", "λ [µm]", "Γ / Γ₀", env="ngsolve")
def ngsolve_cylinder_fibre():
    import ngsolve  # noqa: F401
    from fem_ngsolve import CELL, MESHES, ORIENTATIONS, PML_THICKNESS, RADIUS, INDEX, SOURCE, ngsolve_rates

    points = (0.62, 0.645, 0.66, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2)  # both sides of the TE01/TM01 cutoff, 0.678 um
    theirs = {w: ngsolve_rates(w, *MESHES[1])[0] for w in points}
    dense = np.unique(np.round(np.r_[np.linspace(0.62, 1.2, 59), points], 6))  # NGSolve's wavelengths exactly on the grid
    ours = np.array([ps.solve_problem(ps.Problem("cylinders", [RADIUS], [INDEX, 1.0], w, ps.PointDipole(SOURCE, "electric"), 1e-6))["total"][:3] for w in dense])
    index = {w: int(np.flatnonzero(np.isclose(dense, w))[0]) for w in points}
    out, refs, rel = [], [], []
    for i, o in enumerate(ORIENTATIONS):
        out.append(series(o, f"{o}, PyStratify", ours[:, i], "pystratify"))
        refs.append(series(f"{o}_ref", f"{o}, NGSolve", at(dense, {w: theirs[w][i] for w in points}), "reference"))
        rel.append(max(abs(theirs[w][i] / ours[index[w], i] - 1) for w in points))
    return dict(x=sig(dense), series=out + refs, worst=float(max(rel)), worst_label="largest relative difference (the middle mesh)",
                note=f"Second-order Nédélec elements in a cylindrical cell (radius and half-height {CELL:g} µm) inside a {PML_THICKNESS:g} µm PML, "
                     "radial in ρ and Cartesian in z, which the fibre runs through; mirror symmetry halves it twice, so each orientation "
                     "is a quarter problem (~100k unknowns, the middle mesh: 0.25 µm in vacuum, 0.06 µm in the fibre). The rate is Im(p·E) at "
                     "the dipole over the same on the same mesh with the fibre's permittivity set to 1. TE₀₁ and TM₀₁ are cut off at 0.678 µm; "
                     "just above it their fields reach far into the vacuum. The difference shrinks with the mesh: worst 1.4% → 0.8% → 0.5% at 0.8 µm "
                     "and 4.4% → 1.3% → 0.8% at 0.645 µm (FEM_NGSOLVE.md). "
                     "An earlier spherical cell with a radial PML met the guided modes badly and was off by up to 8–11% here; Meep agrees "
                     "to 0.74% over the same band.")


if __name__ == "__main__":
    run(__file__)
