"""Cylinders, plane wave: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/cylinders_plane_wave_consistency.py                                                     #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-plane-wave-consistency.html, computed by this one file
from a PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE
and writes, beside itself, cylinders_plane_wave_consistency.json (the curves the page draws, value for value) and
cylinders_plane_wave_consistency.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps


@figure("cylinder-bessel-direct", "Dielectric cylinder", "radius [nm]", "Q_sca")
def plot_dielectric_cylinder():
    from scipy.special import h1vp, hankel1, jv, jvp

    radii_nm = np.linspace(20, 600, 117)
    every = 3  # the Bessel equations every 15 nm
    n, lam = 1.5, 0.6
    orders = np.arange(-16, 17)

    def bessel_q_sca(R, eta):  # Q_sca = (2/x) Σ|t_m|², each t_m from the two boundary conditions
        x = 2 * np.pi * R * 1e-3 / lam
        t = -((jvp(orders, x) * jv(orders, n * x) - eta * jv(orders, x) * jvp(orders, n * x))
              / (h1vp(orders, x) * jv(orders, n * x) - eta * hankel1(orders, x) * jvp(orders, n * x)))
        return 2 / x * np.sum(np.abs(t) ** 2)

    sols = [ps.solve_cylinder([R * 1e-3], [n, 1.0], lam, m_max=16) for R in radii_nm]
    ours = {"e": np.array([ps.cross_widths(s, "axial-electric")["efficiency_scattering"] for s in sols]),
            "m": np.array([ps.cross_widths(s, "axial-magnetic")["efficiency_scattering"] for s in sols])}
    ref = {key: np.array([bessel_q_sca(R, eta) for R in radii_nm[::every]]) for key, eta in (("e", n), ("m", 1 / n))}
    labels = {"e": "E along the axis", "m": "H along the axis"}
    return result(
        radii_nm,
        [line(k, f"{labels[k]}, PyStratify", ours[k]) for k in labels]
        + [dots(f"{k}_ref", f"{labels[k]}, Bessel equations", ref[k], every, size=radii_nm.size) for k in labels],
        worst=max(largest(ours[k][::every], ref[k], relative=True) for k in labels),
        worst_label="largest relative difference",
        note="Each order's coefficient solved directly from the two boundary conditions in SciPy's Bessel functions "
             "(Bohren & Huffman §8.4), summed into Q_sca = (2/x) Σ|t_m|², against the cylinder transfer matrices.")


@figure("cylinder-index-matched", "Index-matched cylinder", "x [nm]", "Re E")
def plot_index_matched_cylinder():
    n, wavelength, beta = 1.5, .6, 2.0
    k = 2 * np.pi * n / wavelength
    q = np.sqrt(k * k - beta * beta)
    s = ps.solve_cylinder([.1], [n, n], wavelength, beta=beta, m_max=24)
    x = np.linspace(-.25, .25, 201)  # every 2.5 nm
    every = 8  # the plane wave every 20 nm
    points = np.column_stack([x, np.full_like(x, .03), np.full_like(x, .01)])
    phase = np.exp(1j * (q * x + beta * .01))
    cases = (("axial-electric", "E along the axis, E_z", 2, q / k), ("axial-magnetic", "H along the axis, E_y", 1, 1.0))
    fields = {pol: s.field(points, pol)[0][:, component] for pol, _, component, _ in cases}
    return result(
        x * 1e3,
        [line(pol, f"{label}, PyStratify", fields[pol].real) for pol, label, _, _ in cases]
        + [dots(f"{pol}_ref", f"{label}, the plane wave", (amplitude * phase[::every]).real, every, size=x.size)
           for pol, label, _, amplitude in cases],
        worst=max(np.max(np.abs(fields[pol] - amplitude * phase)) for pol, _, _, amplitude in cases),
        worst_label="largest |E − plane wave|",
        note="A cylinder of the same index as its surroundings is no cylinder: the field PyStratify builds from its "
             "Bessel expansion inside and outside (24 orders) must be the incident plane wave, here at oblique "
             "incidence (β = 2 µm⁻¹) along y = 30 nm, z = 10 nm across the 100 nm radius; the test also checks |E|² "
             "= 1 and |H|² = n² at three points.")


if __name__ == "__main__":
    run(__file__)
