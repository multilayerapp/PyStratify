"""Spheres, plane wave: PyStratify against MNPBEM.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_mnpbem.py                                                            #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-mnpbem.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_plane_wave_mnpbem.json (the curves the page draws, value for value) and
spheres_plane_wave_mnpbem.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps


@figure("mnpbem-spectrum", "Metal-like sphere", "λ [nm]", "C [nm²]")
def plot_metal_like_sphere():
    from mnpbem_spheres import METAL, mnpbem_spectrum

    lam = np.linspace(400, 800, 161)
    every = 10  # MNPBEM every 25 nm
    faces, ext, sca = mnpbem_spectrum("sphere in vacuum", 1024, lam[::every])
    cs = ps.cross_sections(ps.solve([20.0], np.array([[np.sqrt(METAL + 0j), 1.0]] * lam.size), lam))
    area = np.pi * 20.0 ** 2
    ours_ext, ours_sca = cs.q_ext * area, cs.q_sca * area
    return result(
        lam,
        [line("ext", "C_ext, PyStratify", ours_ext), line("sca", "C_sca, PyStratify", ours_sca),
         dots("ext_ref", f"C_ext, MNPBEM ({faces} faces)", ext, every, size=lam.size),
         dots("sca_ref", f"C_sca, MNPBEM ({faces} faces)", sca, every, size=lam.size)],
        worst=max(largest(ext / ours_ext[::every], 1), largest(sca / ours_sca[::every], 1)),
        worst_label="largest relative difference (the mesh's error)",
        note="MNPBEM (Hohenester & Trügler) discretizes the surface into triangles and solves the retarded boundary "
             "integral equations; a constant permittivity, so both codes see the same ε at every wavelength. "
             "Percent-level by design: the test checks the difference shrinks as the mesh is refined (0.45–4.4% at "
             "2044 faces).")


@figure("mnpbem-thin-nanoshell", "Thin nanoshell", "λ [nm]", "C_ext [nm²]")
def plot_thin_nanoshell():
    from mnpbem_spheres import METAL, SILICA, mnpbem_spectrum

    radii, n = [15.0, 20.0], [np.sqrt(SILICA + 0j), np.sqrt(METAL + 0j), 1.0]
    lam = np.linspace(400, 800, 161)
    every, start = 10, 5  # MNPBEM every 25 nm, from 412.5 nm
    ours = ps.cross_sections(ps.solve(radii, np.array([n] * lam.size), lam)).q_ext * np.pi * radii[-1] ** 2
    faces, ext, _ = mnpbem_spectrum("thin nanoshell", 256, lam[start::every])
    return result(
        lam,
        [line("ext", "C_ext, PyStratify", ours),
         dots("ext_ref", f"C_ext, MNPBEM ({faces} faces)", ext, every, start, size=lam.size)],
        worst=largest(ext / ours[start::every], 1),
        worst_label="largest relative difference (the mesh's error, shown, not asserted)",
        note="Shown, not asserted. MNPBEM, with elements of 2–4 nm on surfaces 5 nm apart, stays 11–16% low and "
             "barely moves with the mesh: a resolution limit of the boundary elements at the meshes this machine "
             "affords, not a disagreement about the physics. The Mie codes, treams and scattnlay, agree with "
             "PyStratify on this sphere to 2e-12.")


if __name__ == "__main__":
    run(__file__)
