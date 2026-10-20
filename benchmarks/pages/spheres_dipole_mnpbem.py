"""Spheres, point dipole: PyStratify against mnpbem.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_dipole_mnpbem.py                                                                #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-mnpbem.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_dipole_mnpbem.json (the curves the page draws, value for value) and spheres_dipole_mnpbem.png (needs matplotlib).
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


@figure("mnpbem-decay", "Metal-like sphere", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def mnpbem_decay():
    from mnpbem_spheres import METAL, mnpbem_rates

    gaps = np.geomspace(5, 40, 8)  # from the test's 5 nm: closer, the triangles (~2 nm) no longer resolve the near field
    faces, total, radiative = mnpbem_rates(1024, gaps)
    dense = np.geomspace(5, 40, 71)
    rates = ps.decay_rates([20.0], [np.sqrt(METAL + 0j), 1.0], 600.0, 20.0 + dense)
    def sparse(values):
        out = np.full(dense.size, np.nan)
        out[::10] = values
        return out
    rel = np.max(np.abs(total / rates.total[::10] - 1))
    return dict(x=sig(dense), series=[
        series("tot_rad", "total, radial, PyStratify", rates.total[:, 0], "pystratify"), series("tot_tan", "total, tangential, PyStratify", rates.total[:, 1], "pystratify"),
        series("rad_rad", "radiative, radial, PyStratify", rates.radiative[:, 0], "pystratify"),
        series("tot_rad_ref", f"total, radial, MNPBEM ({faces} faces)", sparse(total[:, 0]), "reference"),
        series("tot_tan_ref", "total, tangential, MNPBEM", sparse(total[:, 1]), "reference"),
        series("rad_rad_ref", "radiative, radial, MNPBEM", sparse(radiative[:, 0]), "reference")],
        worst=float(rel), worst_label="largest relative difference in the total rates (the mesh's error)",
        note="MNPBEM's dipoleret solves the boundary integral equations for the dipole's field; its triangles are ~2 nm, so the "
             "difference grows toward the surface (33% for the radial rate at 2 nm, not shown). The test follows four meshes at 5 nm (1.4% radial, 0.04% tangential at 2044 faces).")


if __name__ == "__main__":
    run(__file__)
