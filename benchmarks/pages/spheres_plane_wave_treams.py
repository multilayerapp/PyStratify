"""Spheres, plane wave: PyStratify against treams.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_treams.py                                                            #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-treams.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_plane_wave_treams.json (the curves the page draws, value for value) and
spheres_plane_wave_treams.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps


@figure("treams-nanoshell", "Nanoshell", "λ [nm]", "Q")
def plot_nanoshell():
    import treams

    lam = np.linspace(400, 900, 101)
    every = 4  # treams every 20 nm
    radii, n = [10.0, 20.0], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    area = np.pi * radii[-1] ** 2
    cs = ps.cross_sections(ps.solve(radii, np.array([n] * lam.size), lam, l_max=12))

    def theirs(w):
        tm = treams.TMatrix.sphere(12, 2 * np.pi / w, radii, [treams.Material(complex(v) ** 2) for v in n])
        return np.real(tm.xs_ext_avg) / area, np.real(tm.xs_sca_avg) / area

    ext, sca = np.array([theirs(w) for w in lam[::every]]).T
    return result(
        lam,
        [line("ext", "Q_ext, PyStratify", cs.q_ext), line("sca", "Q_sca, PyStratify", cs.q_sca),
         dots("ext_ref", "Q_ext, treams", ext, every, size=lam.size),
         dots("sca_ref", "Q_sca, treams", sca, every, size=lam.size)],
        worst=max(largest(cs.q_ext[::every], ext, relative=True), largest(cs.q_sca[::every], sca, relative=True)),
        worst_label="largest relative difference",
        note="treams (Beutel et al.) builds the T-matrix of a layered sphere from its own vector spherical waves; "
             "the fixed metal index keeps the shell's plasmon at one wavelength, so the comparison is of the two "
             "codes, not of a material.")


@figure("treams-chiral", "Chiral shell", "λ [nm]", "|T|")
def plot_chiral_shell():
    from mie_codes import CHIRAL, chiral_blocks

    radii, n, kappa, mu, _, l_max = CHIRAL["chiral shell"]
    lam = np.linspace(400, 900, 101)
    every = 4  # treams every 20 nm
    ours = np.array([ps.solve_chiral(radii, n, kappa, w, mu, l_max=l_max).t_helicity[0] for w in lam])  # (W, L, 2, 2)
    theirs = np.array([chiral_blocks(radii, n, kappa, mu, w, l_max)[1] for w in lam[::every]])
    pick = (("l1pp", "ℓ = 1, T₊₊", 0, 0, 0), ("l1mm", "ℓ = 1, T₋₋", 0, 1, 1),
            ("l2pp", "ℓ = 2, T₊₊", 1, 0, 0), ("l2mm", "ℓ = 2, T₋₋", 1, 1, 1))
    return result(
        lam,
        [line(k, f"|{lab}|, PyStratify", np.abs(ours[:, ell, i, j])) for k, lab, ell, i, j in pick]
        + [dots(f"{k}_ref", f"|{lab}|, treams", np.abs(theirs[:, ell, i, j]), every, size=lam.size)
           for k, lab, ell, i, j in pick],
        worst=np.max(np.abs(ours[::every] - theirs)) / np.max(np.abs(theirs)),
        worst_label="largest |T − T_treams| / max |T|",
        note="treams (Beutel et al.) builds a chiral multilayer's T-matrix in the helicity basis from its own "
             "vector spherical waves; the two helicities differ because the shell is chiral. The test covers three "
             "chiral multilayers (gold, magnetic, gain in κ).")


@figure("treams-near-field", "Nanoshell near field", "distance from the centre [nm]", "|E_sca|²", log=True)
def plot_nanoshell_near_field():
    from mie_codes import CASES, WAVELENGTH, stratify_fields, treams_scattered_fields

    radii, n = CASES["nanoshell, 10 nm shell"]
    r = np.linspace(20.5, 60, 80)
    every = 2  # treams every 1 nm
    polar, azimuth = np.radians(30.0), np.radians(40.0)
    points = r[:, None] * np.array([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)])
    l_max = ps.truncation_order(radii[-1], n[-1], WAVELENGTH, regime="near") + 20
    sol = ps.solve(radii, np.array([n], complex), np.array([WAVELENGTH]), l_max=l_max)
    e, _ = stratify_fields(sol, points, np.real(n[-1]), incident=False)
    e_ref, _ = treams_scattered_fields(radii, n, points[::every], l_max)
    return result(
        r,
        [line("e", "|E_sca|², PyStratify", np.sum(np.abs(e) ** 2, 1)),
         dots("e_ref", "|E_sca|², treams", np.sum(np.abs(e_ref) ** 2, 1), every, size=r.size)],
        worst=np.max(np.linalg.norm(e[::every] - e_ref, axis=1) / np.linalg.norm(e[::every], axis=1)),
        worst_label="largest relative difference (vector field)",
        note="Along a line at 30° from the incidence axis: treams applies its T-matrix to the expanded plane wave "
             "and evaluates the scattered waves; PyStratify its own near field without the incident wave. The test "
             "covers six spheres, 1.05–3 radii, five angles.")


if __name__ == "__main__":
    run(__file__)
