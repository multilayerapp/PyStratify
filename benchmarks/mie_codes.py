"""Spheres against two independent Mie codes: scattnlay and treams (benchmarks/MATRIX.md, spheres).

    pip install -e ".[benchmarks]"
    python benchmarks/mie_codes.py        # prints a Markdown report (MIE_CODES.md)

* **scattnlay** (Pena & Pal, Comput. Phys. Commun. 180, 2348 (2009); Ladutenko et al., CPC 214, 225
  (2017)): multilayer Mie by recursive ratios, the reference for large spheres; its multiple-precision
  mode (``mp=True``) is used where double precision is not enough.
* **treams** (Beutel, Fernandez-Corbaton & Rockstuhl, CPC 297, 109076 (2024)): T-matrices of layered,
  also chiral, spheres. It stores the whole T-matrix, 2 l_max (l_max + 2) square, so it joins only
  where l_max stays small (x up to ~15).

Every quantity is a relative difference to PyStratify, except absorption, compared relative to
extinction (it vanishes for lossless spheres), and the amplitudes, compared as
max |S_ours - S_theirs| / max |S| over the angles (both codes use Bohren & Huffman's S1, S2).

PyStratify is compared twice: at its default truncation (Wiscombe's criterion, what a user gets;
designed for cross sections to ~1e-8, so a small metal sphere keeps only l_max = 3 and extinction
and backscattering differ by ~1e-9) and with ``EXTRA_ORDERS`` more orders, where the comparison tests
the formulation at machine precision.
Backscattering efficiency is Q_b = 4 |S1(180 deg)|^2 / x^2 on PyStratify's side. scattnlay works in
the host: size parameters k_host r and indices relative to the host.

The chiral comparison is the one ``tests/test_chiral.py`` asserts: helicity T-matrix elements of
chiral (Pasteur) multilayers against treams, whose negative-helicity waves differ by a sign.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402

WAVELENGTH = 600.0
METAL = np.sqrt(-10 + 1j + 0j)
ANGLES_DEG = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
TREAMS_MAX_X = 15.0
EXTRA_ORDERS = 20

# name -> (outer radii of every layer in nm, indices from the core out, host last)
CASES = {
    "dielectric, x = 1": ([95.5], [1.5, 1.0]),
    "absorbing, x = 10": ([955.0], [1.5 + 0.1j, 1.0]),
    "metal-like, eps = -10 + i, 40 nm": ([20.0], [METAL, 1.0]),
    "high-index (Kerker regime), n = 3.5": ([75.0], [3.5 + 0.01j, 1.0]),
    "nanoshell, 10 nm shell": ([10.0, 20.0], [np.sqrt(2.13), METAL, 1.0]),
    "thin nanoshell, 5 nm shell": ([15.0, 20.0], [np.sqrt(2.13), METAL, 1.0]),
    "matryoshka metal/silica/metal in water": ([10.0, 20.0, 30.0], [METAL, 1.45, METAL, 1.33]),
    "graded index, 10 layers, x = 12": (list(np.linspace(114.6, 1146.0, 10)), list(np.linspace(1.6, 1.35, 10)) + [1.0]),
    "large lossless, x = 200": ([19099.0], [1.33, 1.0]),
    "large absorbing, x = 1000": ([95493.0], [1.5 + 0.001j, 1.0]),
}

# name -> (radii, indices, Pasteur kappa per region, permeability per region, wavelength, l_max)
CHIRAL = {
    "chiral shell": ([40.0, 55.0], [1.45 + 0.01j, 1.6 + 0.02j, 1.33], [0, 0.05 + 0.01j, 0], [1, 1, 1], 614.0, 12),
    "chiral core, gold, magnetic chiral shell": ([30.0, 33.0, 60.0], [1.5, 0.27 + 2.93j, 1.55 + 0.001j, 1.33],
                                                 [0.02, 0, 0.1 + 0.003j, 0], [1, 1, 1.2, 1], 614.0, 20),
    "strongly chiral, gain in kappa": ([200.0, 230.0], [2.0 + 0.05j, 1.5, 1.0], [0.3, -0.2 + 0.01j, 0], [1, 1, 1],
                                       500.0, 30),
}


def size_parameter(radii, n):
    return 2 * np.pi * np.real(n[-1]) * radii[-1] / WAVELENGTH


def stratify(radii, n, extra_orders=0):
    l_max = ps.truncation_order(radii[-1], n[-1], WAVELENGTH) + extra_orders
    sol = ps.solve(radii, np.array([n], complex), np.array([WAVELENGTH]), l_max=l_max)
    sections = ps.cross_sections(sol)
    s1, s2, _, _ = ps.amplitude_matrix(sol, np.radians(ANGLES_DEG))
    x = size_parameter(radii, n)
    return {"Q_ext": sections.q_ext[0], "Q_sca": sections.q_sca[0], "Q_abs": sections.q_abs[0],
            "Q_back": 4 * abs(s1[0, -1]) ** 2 / x ** 2, "S1": s1[0], "S2": s2[0]}


def scattnlay_values(radii, n, multiple_precision=False):
    from scattnlay import scattnlay

    host = np.real(n[-1])
    x = 2 * np.pi * host * np.asarray(radii) / WAVELENGTH
    m = np.asarray(n[:-1], complex) / host
    _, q_ext, q_sca, q_abs, q_back, _, _, _, s1, s2 = scattnlay(x, m, np.radians(ANGLES_DEG), mp=multiple_precision)
    return {"Q_ext": q_ext, "Q_sca": q_sca, "Q_abs": q_abs, "Q_back": q_back, "S1": s1, "S2": s2}


def treams_values(radii, n):
    import treams

    x = size_parameter(radii, n)
    l_max = int(np.ceil(x + 4 * x ** (1 / 3) + 4))
    materials = [treams.Material(complex(v) ** 2) for v in n]
    tmatrix = treams.TMatrix.sphere(l_max, 2 * np.pi / WAVELENGTH, radii, materials)
    area = np.pi * radii[-1] ** 2
    ext, sca = np.real(tmatrix.xs_ext_avg) / area, np.real(tmatrix.xs_sca_avg) / area
    return {"Q_ext": ext, "Q_sca": sca, "Q_abs": ext - sca}


def chiral_against_treams(name):
    """Worst |T_ours - T_treams| / max |T| over orders of the helicity T-matrix."""
    import treams

    radii, n, kappa, mu, wavelength, l_max = CHIRAL[name]
    materials = [treams.Material(v ** 2 / m, m, k) for v, k, m in zip(n, kappa, mu)]
    tmatrix = treams.TMatrix.sphere(l_max, 2 * np.pi / wavelength, radii, materials, poltype="helicity")
    basis, block = tmatrix.basis, np.asarray(tmatrix)
    ours = ps.solve_chiral(radii, n, kappa, wavelength, mu, l_max=l_max).t_helicity[0]
    worst = 0.0
    for l in range(1, l_max + 1):
        plus, minus = (int(np.flatnonzero((basis.l == l) & (basis.m == 0) & (basis.pol == pol))[0]) for pol in (1, 0))
        theirs = np.array([[block[plus, plus], -block[plus, minus]], [-block[minus, plus], block[minus, minus]]])
        worst = max(worst, np.abs(ours[l - 1] - theirs).max() / np.abs(theirs).max())
    return worst


def _difference(ours, theirs, quantity):
    if quantity in ("S1", "S2"):
        return float(np.max(np.abs(ours[quantity] - theirs[quantity])) / np.max(np.abs(theirs[quantity])))
    if quantity == "Q_abs":
        return float(abs(ours[quantity] - theirs[quantity]) / abs(theirs["Q_ext"]))
    return float(abs(ours[quantity] / theirs[quantity] - 1))


def compare(cases=CASES, chiral=CHIRAL):
    """{(case, reference): {quantity: difference}}."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (radii, n) in cases.items():
            default, converged = stratify(radii, n), stratify(radii, n, EXTRA_ORDERS)
            x = size_parameter(radii, n)
            theirs = scattnlay_values(radii, n)
            out[(name, "scattnlay, PyStratify at its default truncation")] = {
                q: _difference(default, theirs, q) for q in theirs}
            out[(name, f"scattnlay, PyStratify + {EXTRA_ORDERS} orders")] = {
                q: _difference(converged, theirs, q) for q in theirs}
            if x > 100:
                precise = scattnlay_values(radii, n, True)
                out[(name, f"scattnlay multiple precision, + {EXTRA_ORDERS} orders")] = {
                    q: _difference(converged, precise, q) for q in precise}
            if x <= TREAMS_MAX_X:
                theirs = treams_values(radii, n)
                out[(name, f"treams, + {EXTRA_ORDERS} orders")] = {q: _difference(converged, theirs, q) for q in theirs}
        for name in chiral:
            out[(name, "treams, chiral T-matrix")] = {"T": chiral_against_treams(name)}
    return out


def report(results):
    quantities = ("Q_ext", "Q_sca", "Q_abs", "Q_back", "S1", "S2", "T")
    print("# PyStratify spheres against scattnlay and treams\n")
    print("Generated by `python benchmarks/mie_codes.py`. Relative difference to PyStratify (absorption: over")
    print("extinction; amplitudes: max |S_ours - S_theirs| / max |S| over "
          f"{', '.join(f'{a:g}' for a in ANGLES_DEG)} deg); {WAVELENGTH:g} nm in vacuum;")
    print("- = not computed by that code. PyStratify's default truncation (Wiscombe) is what a user gets; with")
    print(f"{EXTRA_ORDERS} more orders the comparison tests the formulation. treams joins up to x = 15 (it stores")
    print("the whole T-matrix).\n")
    used = [q for q in quantities if any(q in values for values in results.values())]
    print("| case | reference | " + " | ".join(used) + " |")
    print("|---|---|" + "---|" * len(used))
    for (name, reference), values in results.items():
        cells = [f"{values[q]:.1e}" if q in values else "-" for q in used]
        print(f"| {name} | {reference} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    report(compare())
