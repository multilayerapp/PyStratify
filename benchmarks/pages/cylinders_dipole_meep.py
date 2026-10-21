"""Cylinders, point dipole: PyStratify against Meep.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh meep                # the meep environment (benchmarks/envs/README.md)              #
#   ~/.multilayer/bench/meep/bin/python benchmarks/pages/cylinders_dipole_meep.py                                   #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-dipole-meep.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, cylinders_dipole_meep.json (the curves the page draws, value for value) and
cylinders_dipole_meep.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run

import pystratify as ps


@figure("meep-cylinder-fibre", "Step-index fibre", "λ [µm]", "Γ / Γ₀", env="meep")
def plot_step_index_fibre():
    import meep as mp
    from fdtd_meep import CASES, ORIENTATIONS, meep_ldos

    case = "fibre n = 2, R = 150 nm, dipole 100 nm outside"
    radii, n, source = CASES[case]
    mp.verbosity(0)
    lam = 1 / np.linspace(0.85, 1.65, 161)
    every = 10  # Meep at 17 frequencies, PyStratify at 161
    theirs = {o: meep_ldos(case, o, 60, wavelengths=lam[::every]) / meep_ldos(case, o, 60, False,
                                                                              wavelengths=lam[::every])
              for o in ORIENTATIONS}
    dipole = ps.PointDipole(source, "electric")
    ours = np.array([ps.solve_problem(ps.Problem("cylinders", radii, n, w, dipole, 1e-6))["total"][:3] for w in lam])
    return result(
        lam,
        [line(o, f"{o}, PyStratify", ours[:, i]) for i, o in enumerate(ORIENTATIONS)]
        + [dots(f"{o}_ref", f"{o}, Meep", theirs[o], every, size=lam.size) for o in ORIENTATIONS],
        worst=max(np.max(np.abs(theirs[o] / ours[::every, i] - 1)) for i, o in enumerate(ORIENTATIONS)),
        worst_label="largest relative difference (the grid's error)",
        note="Meep in 3D: the fibre runs through the perfectly matched layers, a 1.8 µm cell plus 1 µm of PML per "
             "side, two mirror planes; a Gaussian pulse and the local density of states at the dipole, over the "
             "same run without the fibre so the grid's own dispersion cancels. TE₀₁ and TM₀₁ are cut off at 0.678 "
             "µm. The difference shrinks with the grid, 6.9% → 3.2% → 1.7% → 0.74% at 20–60 px/µm (FDTD_MEEP.md); "
             "the test runs 20 and 30.")


if __name__ == "__main__":
    run(__file__)
