"""Spheres, plane wave: PyStratify against mnpbem.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_mnpbem.py                                                            #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-mnpbem.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_plane_wave_mnpbem.json (the curves the page draws, value for value) and spheres_plane_wave_mnpbem.png (needs matplotlib).
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


@figure("mnpbem-spectrum", "Metal-like sphere", "λ [nm]", "C [nm²]")
def mnpbem_spectrum_figure():
    from mnpbem_spheres import METAL, mnpbem_spectrum

    lam = np.linspace(400, 800, 161)
    faces, ext, sca = mnpbem_spectrum("sphere in vacuum", 1024, lam[::10])
    cs = ps.cross_sections(ps.solve([20.0], np.array([[np.sqrt(METAL + 0j), 1.0]] * lam.size), lam))
    area = np.pi * 20.0 ** 2
    ext_ref, sca_ref = np.full(lam.size, np.nan), np.full(lam.size, np.nan)
    ext_ref[::10], sca_ref[::10] = ext, sca
    return dict(x=sig(lam), series=[series("ext", "C_ext, PyStratify", cs.q_ext * area, "pystratify"), series("sca", "C_sca, PyStratify", cs.q_sca * area, "pystratify"),
                                    series("ext_ref", f"C_ext, MNPBEM ({faces} faces)", ext_ref, "reference"), series("sca_ref", f"C_sca, MNPBEM ({faces} faces)", sca_ref, "reference")],
                worst=float(max(np.max(np.abs(ext / (cs.q_ext[::10] * area) - 1)), np.max(np.abs(sca / (cs.q_sca[::10] * area) - 1)))),
                worst_label="largest relative difference (the mesh's error)",
                note="MNPBEM (Hohenester & Trügler) discretizes the surface into triangles and solves the retarded boundary integral "
                     "equations; a constant permittivity, so both codes see the same ε at every wavelength. Percent-level by design: the "
                     "test checks the difference shrinks as the mesh is refined (0.45–4.4% at 2044 faces).")


@figure("mnpbem-thin-nanoshell", "Thin nanoshell", "λ [nm]", "C_ext [nm²]")
def mnpbem_thin_nanoshell():
    from mie_codes import treams_values, scattnlay_values
    import mie_codes
    from mnpbem_spheres import METAL, SILICA, mnpbem_spectrum

    radii, n = [15.0, 20.0], [np.sqrt(SILICA + 0j), np.sqrt(METAL + 0j), 1.0]
    lam = np.linspace(400, 800, 161)
    cs = ps.cross_sections(ps.solve(radii, np.array([n] * lam.size), lam))
    area = np.pi * radii[-1] ** 2
    ours = cs.q_ext * area
    t_ref, s_ref = np.full(lam.size, np.nan), np.full(lam.size, np.nan)
    for i in range(0, lam.size, 10):
        mie_codes.WAVELENGTH = lam[i]  # the Mie wrappers read the module's wavelength
        t_ref[i] = float(treams_values(radii, n)["Q_ext"]) * area
        s_ref[i] = float(scattnlay_values(radii, n)["Q_ext"]) * area
    mie_codes.WAVELENGTH = 600.0
    faces, ext, _ = mnpbem_spectrum("thin nanoshell", 256, lam[5::10])
    m_ref = np.full(lam.size, np.nan)
    m_ref[5::10] = ext
    mie = max(np.nanmax(np.abs(t_ref / ours - 1)), np.nanmax(np.abs(s_ref / ours - 1)))
    return dict(x=sig(lam), series=[series("ext", "C_ext, PyStratify", ours, "pystratify"),
                                    series("treams", "C_ext, treams", t_ref, "reference"), series("scattnlay", "C_ext, scattnlay", s_ref, "reference"),
                                    series("mnpbem", f"C_ext, MNPBEM ({faces} faces)", m_ref, "reference")],
                worst=float(mie), worst_label="largest relative difference among the Mie codes",
                note="Shown, not asserted. The three Mie codes agree to all printed digits; MNPBEM, with elements of 2–4 nm on surfaces 5 nm "
                     "apart, stays about 12% low and barely moves with the mesh: a resolution limit of the boundary elements at the "
                     "meshes this machine affords, not a disagreement about the physics.")


if __name__ == "__main__":
    run(__file__)
