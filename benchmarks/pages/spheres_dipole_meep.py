"""Spheres, point dipole: PyStratify against Meep.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh meep                # the meep environment (benchmarks/envs/README.md)              #
#   ~/.multilayer/bench/meep/bin/python benchmarks/pages/spheres_dipole_meep.py                                     #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-meep.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_dipole_meep.json (the curves the page draws, value for value) and
spheres_dipole_meep.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run

import pystratify as ps


@figure("meep-axial-sphere", "Dielectric sphere", "λ [µm]", "Γ / Γ₀", env="meep")
def plot_dielectric_sphere():
    import meep as mp
    from fdtd_meep import AXIAL_CASES, meep_ldos_axial

    case = "sphere n = 2, R = 150 nm, dipole 100 nm outside on its axis"
    _, dimensions, n, source, names = AXIAL_CASES[case]
    mp.verbosity(0)
    lam = 1 / np.linspace(0.85, 1.65, 161)
    every = 10  # Meep at 17 frequencies, PyStratify at 161
    theirs = {o: meep_ldos_axial(case, o, 100, wavelengths=lam[::every])
              / meep_ldos_axial(case, o, 100, False, wavelengths=lam[::every]) for o in names}
    dipole = ps.PointDipole(source, "electric", 1)
    ours = np.array([ps.solve_problem(ps.Problem("spheres", dimensions, n, w, dipole, 1e-8))["total"][:2] for w in lam])
    return result(
        lam,
        [line(o, f"{o}, PyStratify", ours[:, i]) for i, o in enumerate(names)]
        + [dots(f"{o}_ref", f"{o}, Meep", theirs[o], every, size=lam.size) for o in names],
        worst=max(np.max(np.abs(theirs[o] / ours[::every, i] - 1)) for i, o in enumerate(names)),
        worst_label="largest relative difference (the grid's error)",
        note="Meep in cylindrical coordinates (m = 0 for the radial dipole, m = 1 for the tangential), a Gaussian "
             "pulse and its local density of states at the dipole, over the same run without the sphere so the "
             "grid's own dispersion cancels. A curved surface on a square grid converges in an oscillating way: "
             "5.5% at 25 px/µm, 1.6–1.7% at 50–100 (the test).")


if __name__ == "__main__":
    run(__file__)
