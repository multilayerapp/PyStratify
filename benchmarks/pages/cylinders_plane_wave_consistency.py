"""Cylinders, plane wave: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/cylinders_plane_wave_consistency.py                                                     #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-plane-wave-consistency.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
cylinders_plane_wave_consistency.json (the curves the page draws, value for value) and cylinders_plane_wave_consistency.png (needs matplotlib).
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


@figure("cylinder-bessel-direct", "Dielectric cylinder", "radius [nm]", "Q_sca")
def cylinder_bessel_direct():
    from scipy.special import h1vp, hankel1, jv, jvp

    radii_nm = np.linspace(20, 600, 117)
    n, lam = 1.5, 0.6
    ours = {"e": [], "m": []}
    ref = {"e": [], "m": []}
    for R in radii_nm:
        x = 2 * np.pi * R * 1e-3 / lam
        s = ps.solve_cylinder([R * 1e-3], [n, 1.0], lam, m_max=16)
        ours["e"].append(ps.cross_widths(s, "axial-electric")["efficiency_scattering"])
        ours["m"].append(ps.cross_widths(s, "axial-magnetic")["efficiency_scattering"])
        for key, eta in (("e", n), ("m", 1 / n)):
            t = []
            for m in range(-16, 17):
                t.append(-(jvp(m, x) * jv(m, n * x) - eta * jv(m, x) * jvp(m, n * x)) / (h1vp(m, x) * jv(m, n * x) - eta * hankel1(m, x) * jvp(m, n * x)))
            ref[key].append(2 / x * np.sum(np.abs(t) ** 2))
    return dict(x=sig(radii_nm), series=[series("e", "E along the axis, PyStratify", ours["e"], "pystratify"), series("m", "H along the axis, PyStratify", ours["m"], "pystratify"),
                                         series("e_ref", "E along the axis, Bessel equations", ref["e"], "reference"), series("m_ref", "H along the axis, Bessel equations", ref["m"], "reference")],
                worst=max(worst(ours["e"], ref["e"], relative=True), worst(ours["m"], ref["m"], relative=True)), worst_label="largest relative difference",
                note="Each order's coefficient solved directly from the two boundary conditions in SciPy's Bessel functions (Bohren & Huffman §8.4), summed into Q_sca = (2/x) Σ|t_m|², against the cylinder transfer matrices.")


@figure("cylinder-index-matched", "Index-matched cylinder", "x [nm]", "Re E")
def cylinder_index_matched():
    n, wavelength, beta = 1.5, .6, 2.0
    k = 2 * np.pi * n / wavelength
    q = np.sqrt(k * k - beta * beta)
    s = ps.solve_cylinder([.1], [n, n], wavelength, beta=beta, m_max=24)
    x = np.linspace(-.25, .25, 151)
    points = np.column_stack([x, np.full_like(x, .03), np.full_like(x, .01)])
    phase = np.exp(1j * (q * x + beta * .01))
    out, worst_ = [], 0.0
    for pol, label, component, amplitude in (("axial-electric", "E along the axis, E_z", 2, q / k), ("axial-magnetic", "H along the axis, E_y", 1, 1.0)):
        E, _ = s.field(points, pol)
        ours, ref = E[:, component].real, (amplitude * phase).real
        out += [(pol, label, ours, ref)]
        worst_ = max(worst_, float(np.max(np.abs(E[:, component] - amplitude * phase))))
    return dict(x=sig(x * 1e3), series=[series(p, f"{lab}, PyStratify", o, "pystratify") for p, lab, o, _ in out]
                + [series(f"{p}_ref", f"{lab}, the plane wave", r, "reference") for p, lab, _, r in out],
                worst=worst_, worst_label="largest |E − plane wave|",
                note="A cylinder of the same index as its surroundings is no cylinder: the field PyStratify builds from its Bessel expansion "
                     "inside and outside (24 orders) must be the incident plane wave, here at oblique incidence (β = 2 µm⁻¹) along y = 30 nm, "
                     "z = 10 nm across the 100 nm radius; the test also checks |E|² = 1 and |H|² = n² at three points.")


if __name__ == "__main__":
    run(__file__)
