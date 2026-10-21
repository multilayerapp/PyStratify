"""Spheres, point dipole: PyStratify against MNPBEM.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_dipole_mnpbem.py                                                                #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-mnpbem.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_dipole_mnpbem.json (the curves the page draws, value for value) and
spheres_dipole_mnpbem.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run

import pystratify as ps


@figure("mnpbem-decay", "Metal-like sphere", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def plot_metal_like_sphere():
    from mnpbem_spheres import METAL, mnpbem_rates

    gaps = np.geomspace(5, 40, 71)  # from the test's 5 nm: closer, the triangles (~2 nm) no longer resolve the field
    every = 10  # MNPBEM at 8 gaps
    faces, total, radiative = mnpbem_rates(1024, gaps[::every])
    rates = ps.decay_rates([20.0], [np.sqrt(METAL + 0j), 1.0], 600.0, 20.0 + gaps)
    return result(
        gaps,
        [line("tot_rad", "total, radial, PyStratify", rates.total[:, 0]),
         line("tot_tan", "total, tangential, PyStratify", rates.total[:, 1]),
         line("rad_rad", "radiative, radial, PyStratify", rates.radiative[:, 0]),
         dots("tot_rad_ref", f"total, radial, MNPBEM ({faces} faces)", total[:, 0], every, size=gaps.size),
         dots("tot_tan_ref", "total, tangential, MNPBEM", total[:, 1], every, size=gaps.size),
         dots("rad_rad_ref", "radiative, radial, MNPBEM", radiative[:, 0], every, size=gaps.size)],
        worst=np.max(np.abs(total / rates.total[::every] - 1)),
        worst_label="largest relative difference in the total rates (the mesh's error)",
        note="MNPBEM's dipoleret solves the boundary integral equations for the dipole's field; its triangles are "
             "~2 nm, so the difference grows toward the surface (33% for the radial rate at 2 nm, not shown). The "
             "test follows four meshes at 5 nm (1.4% radial, 0.04% tangential at 2044 faces).")


if __name__ == "__main__":
    run(__file__)
