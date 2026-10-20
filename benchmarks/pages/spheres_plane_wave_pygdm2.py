"""Spheres, plane wave: PyStratify against pygdm2.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_pygdm2.py                                                            #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-pygdm2.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_plane_wave_pygdm2.json (the curves the page draws, value for value) and spheres_plane_wave_pygdm2.png (needs matplotlib).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks", ROOT / "benchmarks" / "pages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pystratify as ps  # noqa: E402

from _page import figure, run, series, sig  # noqa: E402


@figure("gdm-spectrum", "Absorbing sphere", "λ [nm]", "C [nm²]")
def gdm_spectrum():
    from gdm_spheres import gdm_simulation
    from pyGDM2 import linear

    lam = np.linspace(400, 800, 17)
    simulation, dipoles, r_eff = gdm_simulation(40.0, 2.0 + 0.1j, 6.0, lam)
    theirs = np.array([[float(v) for v in linear.extinct(simulation, i)][:2] for i in range(len(lam))])
    dense = np.linspace(400, 800, 161)
    cs = ps.cross_sections(ps.solve([r_eff], np.array([[2.0 + 0.1j, 1.0]] * dense.size), dense, l_max=20))
    area = np.pi * r_eff ** 2
    ext_ref, sca_ref = np.full(dense.size, np.nan), np.full(dense.size, np.nan)
    ext_ref[::10], sca_ref[::10] = theirs[:, 0], theirs[:, 1]
    ours_ext, ours_sca = cs.q_ext * area, cs.q_sca * area
    return dict(x=sig(dense), series=[series("ext", "C_ext, PyStratify", ours_ext, "pystratify"), series("sca", "C_sca, PyStratify", ours_sca, "pystratify"),
                                      series("ext_ref", f"C_ext, pyGDM2 ({len(dipoles)} dipoles)", ext_ref, "reference"),
                                      series("sca_ref", f"C_sca, pyGDM2 ({len(dipoles)} dipoles)", sca_ref, "reference")],
                worst=float(max(np.max(np.abs(theirs[:, 0] / ours_ext[::10] - 1)), np.max(np.abs(theirs[:, 1] / ours_sca[::10] - 1)))),
                worst_label="largest relative difference (the lattice's error at this mesh)",
                note="pyGDM2 (Wiecha) fills the sphere with point dipoles on a hexagonal lattice and solves their coupling; PyStratify is "
                     "evaluated for the sphere of the same volume (r_eff = %.2f nm). A volume lattice stays ~3%% high at the meshes this "
                     "machine affords; the test asserts a band, and the report follows four meshes." % r_eff)


if __name__ == "__main__":
    run(__file__)
