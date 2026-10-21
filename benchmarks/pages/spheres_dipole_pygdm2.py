"""Spheres, point dipole: PyStratify against pyGDM2.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_dipole_pygdm2.py                                                                #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-pygdm2.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_dipole_pygdm2.json (the curves the page draws, value for value) and
spheres_dipole_pygdm2.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run

import pystratify as ps


@figure("gdm-decay", "Dielectric sphere", "distance from the surface [nm]", "Γ / Γ₀")
def plot_dielectric_sphere():
    from gdm_spheres import gdm_rates, gdm_simulation

    gaps = np.linspace(10, 100, 91)
    every = 10  # pyGDM2 every 10 nm
    simulation, dipoles, r_eff = gdm_simulation(40.0, 2.0, 6.0)
    radial, tangential = gdm_rates(simulation, dipoles, 40.0, gaps[::every])
    rates = ps.decay_rates([r_eff], [2.0, 1.0], 600.0, 40.0 + gaps)
    return result(
        gaps,
        [line("rad", "radial, PyStratify", rates.total[:, 0]), line("tan", "tangential, PyStratify", rates.total[:, 1]),
         dots("rad_ref", f"radial, pyGDM2 ({len(dipoles)} dipoles)", radial, every, size=gaps.size),
         dots("tan_ref", f"tangential, pyGDM2 ({len(dipoles)} dipoles)", tangential, every, size=gaps.size)],
        worst=max(np.max(np.abs(radial / rates.total[::every, 0] - 1)),
                  np.max(np.abs(tangential / rates.total[::every, 1] - 1))),
        worst_label="largest relative difference",
        note="pyGDM2's decay_rate couples the emitter to the dipole lattice through the Green dyadic; distances "
             "from the centroid of the lattice, PyStratify for the sphere of the same volume. The lattice is coarse "
             "close to the surface (the test: 10% at 20 nm, 0.5% at 40 nm).")


if __name__ == "__main__":
    run(__file__)
