"""Spheres, plane wave: PyStratify against pyGDM2.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_pygdm2.py                                                            #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-pygdm2.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_plane_wave_pygdm2.json (the curves the page draws, value for value) and
spheres_plane_wave_pygdm2.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run

import pystratify as ps


@figure("gdm-spectrum", "Absorbing sphere", "λ [nm]", "C [nm²]")
def plot_absorbing_sphere():
    from gdm_spheres import gdm_simulation
    from pyGDM2 import linear

    lam = np.linspace(400, 800, 161)
    every = 10  # pyGDM2 every 25 nm
    simulation, dipoles, r_eff = gdm_simulation(40.0, 2.0 + 0.1j, 6.0, lam[::every])
    theirs = np.array([[float(v) for v in linear.extinct(simulation, i)][:2] for i in range(lam[::every].size)])
    cs = ps.cross_sections(ps.solve([r_eff], np.array([[2.0 + 0.1j, 1.0]] * lam.size), lam, l_max=20))
    area = np.pi * r_eff ** 2
    ours_ext, ours_sca = cs.q_ext * area, cs.q_sca * area
    return result(
        lam,
        [line("ext", "C_ext, PyStratify", ours_ext), line("sca", "C_sca, PyStratify", ours_sca),
         dots("ext_ref", f"C_ext, pyGDM2 ({len(dipoles)} dipoles)", theirs[:, 0], every, size=lam.size),
         dots("sca_ref", f"C_sca, pyGDM2 ({len(dipoles)} dipoles)", theirs[:, 1], every, size=lam.size)],
        worst=max(np.max(np.abs(theirs[:, 0] / ours_ext[::every] - 1)),
                  np.max(np.abs(theirs[:, 1] / ours_sca[::every] - 1))),
        worst_label="largest relative difference (the lattice's error at this mesh)",
        note="pyGDM2 (Wiecha) fills the sphere with point dipoles on a hexagonal lattice and solves their coupling; "
             f"PyStratify is evaluated for the sphere of the same volume (r_eff = {r_eff:.2f} nm). A volume lattice "
             "stays ~3% high at the meshes this machine affords; the test asserts a band, and the report follows "
             "four meshes.")


if __name__ == "__main__":
    run(__file__)
