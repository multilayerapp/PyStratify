"""Spheres, plane wave: PyStratify against scattnlay.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_scattnlay.py                                                         #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-scattnlay.html, computed by this one file
from a PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE
and writes, beside itself, spheres_plane_wave_scattnlay.json (the curves the page draws, value for value) and
spheres_plane_wave_scattnlay.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps


@figure("scattnlay-matryoshka", "Matryoshka", "outer radius [nm]", "Q")
def plot_matryoshka():
    from mie_codes import CASES, scattnlay_values, stratify

    radii, n = CASES["matryoshka metal/silica/metal in water"]
    scale = np.linspace(0.3, 4.0, 120)
    every = 3  # scattnlay at 40 of the 120 sizes
    ours = [stratify(list(np.array(radii) * s), n) for s in scale]
    theirs = [scattnlay_values(list(np.array(radii) * s), n) for s in scale[::every]]
    keys = ("Q_ext", "Q_sca")
    got = {k: np.array([o[k] for o in ours]) for k in keys}
    ref = {k: np.array([float(t[k]) for t in theirs]) for k in keys}
    return result(
        scale * radii[-1],
        [line(k, f"{k}, PyStratify", got[k]) for k in keys]
        + [dots(f"{k}_ref", f"{k}, scattnlay", ref[k], every, size=scale.size) for k in keys],
        worst=max(largest(got[k][::every], ref[k], relative=True) for k in keys),
        worst_label="largest relative difference",
        note="scattnlay (Peña & Pal; Ladutenko et al.) evaluates the multilayer Mie coefficients by its own "
             "recursion; both at the default truncation. The test covers ten spheres to x = 1000, in double and "
             "multiple precision.")


@figure("scattnlay-pattern", "Scattering pattern", "scattering angle [°]", "|S|", log=True)
def plot_scattering_pattern():
    from mie_codes import CASES, EXTRA_ORDERS, WAVELENGTH, scattnlay_values

    radii, n = CASES["large lossless, x = 200"]
    angles = np.linspace(0, 180, 721)
    every = 20  # scattnlay every 5°, PyStratify every 0.25°
    l_max = ps.truncation_order(radii[-1], n[-1], WAVELENGTH) + EXTRA_ORDERS
    sol = ps.solve(radii, np.array([n], complex), np.array([WAVELENGTH]), l_max=l_max)
    s1, s2, _, _ = ps.amplitude_matrix(sol, np.radians(angles))
    theirs = scattnlay_values(radii, n, angles_deg=angles[::every])
    ours = {"S1": np.abs(s1[0]), "S2": np.abs(s2[0])}
    ref = {k: np.abs(np.asarray(theirs[k])) for k in ours}
    return result(
        angles,
        [line(k, f"|{k}|, PyStratify", ours[k]) for k in ours]
        + [dots(f"{k}_ref", f"|{k}|, scattnlay", ref[k], every, size=angles.size) for k in ours],
        worst=max(float(np.max(np.abs(ours[k][::every] - ref[k]) / np.max(ref[k]))) for k in ours),
        worst_label="largest difference relative to the forward peak",
        note="Six decades from the forward peak to the backward glory, with the rainbow near 138°; PyStratify 20 "
             "orders beyond Wiscombe's truncation, scattnlay at its own.")


@figure("scattnlay-near-field", "Matryoshka near field", "x [nm]", "|E|²", log=True)
def plot_matryoshka_near_field():
    from mie_codes import CASES, EXTRA_ORDERS, WAVELENGTH, scattnlay_fields, stratify_fields

    radii, n = CASES["matryoshka metal/silica/metal in water"]
    x = np.linspace(-60, 60, 241)
    every = 6  # fieldnlay every 3 nm, PyStratify every 0.5 nm
    points = np.column_stack([x, np.full_like(x, 1.0), np.zeros_like(x)])  # along the polarisation, 1 nm off the axis
    l_max = ps.truncation_order(radii[-1], n[-1], WAVELENGTH, regime="near") + EXTRA_ORDERS
    sol = ps.solve(radii, np.array([n], complex), np.array([WAVELENGTH]), l_max=l_max)
    e, _ = stratify_fields(sol, points, np.real(n[-1]))
    e_ref, _ = scattnlay_fields(radii, n, points[::every], l_max)
    e_ref = np.asarray(e_ref)
    return result(
        x,
        [line("e", "|E|², PyStratify", np.sum(np.abs(e) ** 2, 1)),
         dots("e_ref", "|E|², scattnlay (fieldnlay)", np.sum(np.abs(e_ref) ** 2, 1), every, size=x.size)],
        worst=float(np.max(np.linalg.norm(e[::every] - e_ref, axis=1) / np.linalg.norm(e[::every], axis=1))),
        worst_label="largest relative difference (vector field)",
        note="Through the gold core, the silica spacer, the gold shell and out into the water, along the "
             "polarisation. scattnlay's fields are double precision; in the 10-layer graded sphere of the report it "
             "returns nan in the three inner layers, where PyStratify's logarithmic coefficients stay finite (shown "
             "there, not asserted).")


if __name__ == "__main__":
    run(__file__)
