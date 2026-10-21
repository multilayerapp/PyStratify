"""Thin films, point dipole: PyStratify against Meep.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh meep                # the meep environment (benchmarks/envs/README.md)              #
#   ~/.multilayer/bench/meep/bin/python benchmarks/pages/films_dipole_meep.py                                       #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-dipole-meep.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, films_dipole_meep.json (the curves the page draws, value for value) and films_dipole_meep.png
(needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run

import pystratify as ps


@figure("meep-axial-film", "Film on glass", "λ [µm]", "Γ / Γ₀", env="meep")
def plot_film_on_glass():
    import meep as mp
    from fdtd_meep import AXIAL_CASES, meep_ldos_axial

    case = "film: dipole 100 nm above 200 nm of n = 2.3 on glass"
    _, dimensions, n, source, names = AXIAL_CASES[case]
    mp.verbosity(0)
    lam = 1 / np.linspace(0.85, 1.65, 161)
    every = 10  # Meep at 17 frequencies, PyStratify at 161
    theirs = {o: meep_ldos_axial(case, o, 100, wavelengths=lam[::every])
              / meep_ldos_axial(case, o, 100, False, wavelengths=lam[::every]) for o in names}
    dipole = ps.PointDipole(-source, "electric", 0)
    ours = np.array([ps.solve_problem(ps.Problem("films", dimensions, n, w, dipole, 1e-8))["total"][:2] for w in lam])
    return result(
        lam,
        [line(o, f"{o}, PyStratify", ours[:, i]) for i, o in enumerate(names)]
        + [dots(f"{o}_ref", f"{o}, Meep", theirs[o], every, size=lam.size) for o in names],
        worst=max(np.max(np.abs(theirs[o] / ours[::every, i] - 1)) for i, o in enumerate(names)),
        worst_label="largest relative difference (the grid's error)",
        note="Meep in cylindrical coordinates around the dipole (m = 0 perpendicular, m = 1 parallel), the film and "
             "substrate infinite in r, 1 µm of PML; the local density of states over the same run without the film. "
             "The film is 20 pixels thick at 100 px/µm and converges at about first order: worst 16.9% → 7.4% → "
             "3.7% at 25, 50 and 100 px/µm (FDTD_MEEP.md); the test asserts < 5% at 100 and closer than 25.")


if __name__ == "__main__":
    run(__file__)
