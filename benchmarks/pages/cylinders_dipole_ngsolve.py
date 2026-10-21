"""Cylinders, point dipole: PyStratify against NGSolve.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh ngsolve                # the ngsolve environment (benchmarks/envs/README.md)        #
#   ~/.multilayer/bench/ngsolve/bin/python benchmarks/pages/cylinders_dipole_ngsolve.py                             #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-dipole-ngsolve.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, cylinders_dipole_ngsolve.json (the curves the page draws, value for value) and
cylinders_dipole_ngsolve.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run

import pystratify as ps


@figure("ngsolve-cylinder-fibre", "Step-index fibre", "λ [µm]", "Γ / Γ₀", env="ngsolve")
def plot_step_index_fibre():
    from fem_ngsolve import CELL, INDEX, MESHES, ORIENTATIONS, PML_THICKNESS, RADIUS, SOURCE, ngsolve_rates

    points = (0.62, 0.645, 0.66, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2)  # both sides of the TE01/TM01 cutoff, 0.678 um
    theirs = np.array([ngsolve_rates(w, *MESHES[1])[0] for w in points])
    lam = np.unique(np.round(np.r_[np.linspace(0.62, 1.2, 59), points], 6))  # NGSolve's wavelengths on the grid
    at = np.searchsorted(lam, points)
    dipole = ps.PointDipole(SOURCE, "electric")
    ours = np.array([ps.solve_problem(ps.Problem("cylinders", [RADIUS], [INDEX, 1.0], w, dipole, 1e-6))["total"][:3]
                     for w in lam])
    ref = np.full(ours.shape, np.nan)
    ref[at] = theirs
    return result(
        lam,
        [line(o, f"{o}, PyStratify", ours[:, i]) for i, o in enumerate(ORIENTATIONS)]
        + [dots(f"{o}_ref", f"{o}, NGSolve", ref[:, i]) for i, o in enumerate(ORIENTATIONS)],
        worst=np.max(np.abs(theirs / ours[at] - 1)),
        worst_label="largest relative difference (the middle mesh)",
        note=f"Second-order Nédélec elements in a cylindrical cell (radius and half-height {CELL:g} µm) inside a "
             f"{PML_THICKNESS:g} µm PML, radial in ρ and Cartesian in z, which the fibre runs through; mirror "
             "symmetry halves it twice, so each orientation is a quarter problem (~100k unknowns, the middle mesh: "
             "0.25 µm in vacuum, 0.06 µm in the fibre). The rate is Im(p·E) at the dipole over the same on the same "
             "mesh with the fibre's permittivity set to 1. TE₀₁ and TM₀₁ are cut off at 0.678 µm; just above it "
             "their fields reach far into the vacuum. The difference shrinks with the mesh: worst 1.4% → 0.8% → "
             "0.5% at 0.8 µm and 4.4% → 1.3% → 0.8% at 0.645 µm (FEM_NGSOLVE.md). An earlier spherical cell with a "
             "radial PML met the guided modes badly and was off by up to 8–11% here; Meep agrees to 0.74% over the "
             "same band.")


if __name__ == "__main__":
    run(__file__)
