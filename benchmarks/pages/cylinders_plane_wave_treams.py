"""Cylinders, plane wave: PyStratify against treams.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/cylinders_plane_wave_treams.py                                                          #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-plane-wave-treams.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
cylinders_plane_wave_treams.json (the curves the page draws, value for value) and cylinders_plane_wave_treams.png (needs matplotlib).
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

from _page import figure, run, series, sig, worst  # noqa: E402


@figure("treams-cylinder-oblique", "Oblique cylinder", "λ [nm]", "Q")
def treams_cylinder_oblique():
    import treams

    lam = np.linspace(400, 900, 51)
    radii, n = [0.1], [1.5 + 0.01j, 1.0]
    ext, sca, ext_ref, sca_ref = [], [], [], []
    for w in lam:
        beta = 2 * np.pi / (w * 1e-3) * np.cos(np.radians(60))
        sol = ps.solve_cylinder(radii, n, w * 1e-3, beta=beta, m_max=12)
        own = ps.cross_widths(sol)
        tm = treams.TMatrixC.cylinder([beta], 12, 2 * np.pi / (w * 1e-3), radii, [treams.Material(complex(v) ** 2) for v in n])
        ext.append(own["efficiency_extinction"]), sca.append(own["efficiency_scattering"])
        ext_ref.append(tm.xw_ext_avg / (2 * radii[-1])), sca_ref.append(tm.xw_sca_avg / (2 * radii[-1]))
    return dict(x=sig(lam), series=[series("ext", "Q_ext, PyStratify", ext, "pystratify"), series("sca", "Q_sca, PyStratify", sca, "pystratify"),
                                    series("ext_ref", "Q_ext, treams", ext_ref, "reference"), series("sca_ref", "Q_sca, treams", sca_ref, "reference")],
                worst=max(worst(ext, ext_ref, relative=True), worst(sca, sca_ref, relative=True)), worst_label="largest relative difference",
                note="Oblique incidence couples the two polarizations at the surface; both codes average the two incident polarizations. treams' cylindrical T-matrix is per axial wavenumber, which a plane wave fixes.")


if __name__ == "__main__":
    run(__file__)
