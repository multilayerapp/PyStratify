"""Cylinders, plane wave: PyStratify against treams.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/cylinders_plane_wave_treams.py                                                          #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-plane-wave-treams.html, computed by this one file from
a PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, cylinders_plane_wave_treams.json (the curves the page draws, value for value) and
cylinders_plane_wave_treams.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps


@figure("treams-cylinder-oblique", "Oblique cylinder", "λ [nm]", "Q")
def plot_oblique_cylinder():
    import treams

    lam = np.linspace(400, 900, 51)
    every = 2  # treams every 20 nm
    radii, n = [0.1], [1.5 + 0.01j, 1.0]

    def beta(w):  # 60° from the axis
        return 2 * np.pi / (w * 1e-3) * np.cos(np.radians(60))

    def ours(w):
        own = ps.cross_widths(ps.solve_cylinder(radii, n, w * 1e-3, beta=beta(w), m_max=12))
        return own["efficiency_extinction"], own["efficiency_scattering"]

    def theirs(w):
        materials = [treams.Material(complex(v) ** 2) for v in n]
        tm = treams.TMatrixC.cylinder([beta(w)], 12, 2 * np.pi / (w * 1e-3), radii, materials)
        return tm.xw_ext_avg / (2 * radii[-1]), tm.xw_sca_avg / (2 * radii[-1])

    got = np.array([ours(w) for w in lam])
    ref = np.array([theirs(w) for w in lam[::every]], float)
    return result(
        lam,
        [line("ext", "Q_ext, PyStratify", got[:, 0]), line("sca", "Q_sca, PyStratify", got[:, 1]),
         dots("ext_ref", "Q_ext, treams", ref[:, 0], every, size=lam.size),
         dots("sca_ref", "Q_sca, treams", ref[:, 1], every, size=lam.size)],
        worst=max(largest(got[::every, i], ref[:, i], relative=True) for i in range(2)),
        worst_label="largest relative difference",
        note="Oblique incidence couples the two polarizations at the surface; both codes average the two incident "
             "polarizations. treams' cylindrical T-matrix is per axial wavenumber, which a plane wave fixes.")


if __name__ == "__main__":
    run(__file__)
