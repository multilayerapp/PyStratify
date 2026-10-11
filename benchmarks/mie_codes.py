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

Emitters beside layered spheres (nanoshells, a matryoshka, electric and magnetic dipoles) are compared with
treams too: the first outside code for multilayer emitters (smuthi takes homogeneous spheres only).

The chiral comparison is the one ``tests/test_chiral.py`` asserts: helicity T-matrix elements of
chiral (Pasteur) multilayers against treams, whose negative-helicity waves differ by a sign.

Near fields are compared against scattnlay's ``fieldnlay`` inside every layer and in the host (and the
scattered host field against treams' T-matrix where it fits, l_max <= ``TREAMS_NEAR_MAX_ORDER``), at
points away from the interfaces, as max |F_ours - F_theirs| / |F_ours| (vector norms) over the
points of a region. scattnlay takes positions in units of 1/k_host, has E_x = exp(i k z) and H in
SI (H_y = 1/Z0, Z0 = 4 pi 1e-7 c), so its H Z0 is compared with PyStratify's Gaussian H / n_host;
both use the same truncation. At the centre only l = 1 survives and PyStratify is exact there: the
report also compares it with Bohren & Huffman's d_1 and m c_1 (E(0) = d_1 x_hat, H(0) = m c_1
y_hat for a homogeneous sphere) and lists scattnlay's centre separately, since its fields lose
digits as k r -> 0 (agreeing to ~1e-14 at 0.1 R_0, ~1e-7 at 1e-3 R_0 and ~1e-3 from 1e-6 R_0 down
to the centre itself).
"""

from __future__ import annotations

import contextlib
import os
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


# near fields: cases from CASES (scattnlay's fields are double precision, so small and moderate spheres)
NEAR_FIELD_CASES = ("dielectric, x = 1", "absorbing, x = 10", "metal-like, eps = -10 + i, 40 nm",
                    "high-index (Kerker regime), n = 3.5", "nanoshell, 10 nm shell", "thin nanoshell, 5 nm shell",
                    "matryoshka metal/silica/metal in water", "graded index, 10 layers, x = 12")
NEAR_FIELD_POLAR_DEG = np.array([0.0, 30.0, 90.0, 150.0, 180.0])
NEAR_FIELD_AZIMUTH_DEG = 40.0
HOST_RADII = (1.05, 1.5, 3.0)  # in units of the outer radius
Z0 = 4e-7 * np.pi * 299792458.0  # scattnlay's impedance of free space
TREAMS_NEAR_MAX_ORDER = 40  # treams' dense T-matrix is 2 L (L + 2) square: 150 MB at L = 40, 775 MB at 58


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


def bohren_huffman_centre(m, x):
    """E(0) = d_1 and H(0) = m c_1 (Gaussian, host index 1) of a homogeneous sphere, Bohren & Huffman
    Eq. (4.53) with mu = 1: d_1 = i m / (m psi(mx) xi'(x) - xi(x) psi'(mx)),
    c_1 = i / (psi(mx) xi'(x) / m - xi(x) psi'(mx)), psi_1(z) = z j_1(z), xi_1(z) = z h_1(z)."""
    from scipy.special import spherical_jn, spherical_yn

    def riccati(z):
        j, y = spherical_jn(1, z), spherical_yn(1, z)
        jd, yd = spherical_jn(1, z, derivative=True), spherical_yn(1, z, derivative=True)
        return z * j, z * (j + 1j * y), j + z * jd, (j + 1j * y) + z * (jd + 1j * yd)

    psi_mx, _, dpsi_mx, _ = riccati(complex(m * x))
    _, xi_x, _, dxi_x = riccati(complex(x))
    d1 = 1j * m / (m * psi_mx * dxi_x - xi_x * dpsi_mx)
    c1 = 1j / (psi_mx * dxi_x / m - xi_x * dpsi_mx)
    return d1, m * c1


def near_field_points(radii):
    """{region: [(radius, (N, 3) points in nm)]}: the middle of the core and of every shell
    ("inside", one entry per layer) and three host radii ("host")."""
    polar, azimuth = np.radians(NEAR_FIELD_POLAR_DEG), np.radians(NEAR_FIELD_AZIMUTH_DEG)
    directions = np.stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], 1)
    inner = np.concatenate([[0.0], radii])
    regions = {"inside": [(a + b) / 2 for a, b in zip(inner[:-1], inner[1:])],
               "host": [f * radii[-1] for f in HOST_RADII]}
    return {region: [(r, r * directions) for r in rs] for region, rs in regions.items()}


@contextlib.contextmanager
def _quiet_stdout():
    """scattnlay's C++ prints 'Near-field early convergence ...' on file descriptor 1."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 1)
            yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def scattnlay_fields(radii, n, points, l_max):
    """(E, H Z0) at points (N, 3) in nm, from scattnlay's fieldnlay."""
    from scattnlay import fieldnlay

    host = np.real(n[-1])
    k = 2 * np.pi * host / WAVELENGTH
    x, m = k * np.asarray(radii), np.asarray(n[:-1], complex) / host
    with _quiet_stdout():
        _, e, h = fieldnlay(x, m, *(k * points[:, i] for i in range(3)), nmax=l_max)
    return e, h * Z0


def treams_scattered_fields(radii, n, points, l_max):
    """Scattered (E, H) in the host at points (N, 3) in nm: treams' T-matrix applied to the expanded
    plane wave, both in the parity basis (treams' default T-matrix is in helicities)."""
    import treams

    k0 = 2 * np.pi / WAVELENGTH
    materials = [treams.Material(complex(v) ** 2) for v in n]
    tmatrix = treams.TMatrix.sphere(l_max, k0, radii, materials, poltype="parity")
    incident = treams.plane_wave([0, 0, k0 * np.real(n[-1])], [1, 0, 0], k0=k0, material=materials[-1],
                                 poltype="parity")
    scattered = tmatrix @ incident.expand(tmatrix.basis)
    return np.asarray(scattered.efield(points)), np.asarray(scattered.hfield(points))


def stratify_fields(sol, points, host, incident=True):
    f = ps.near_field(sol, points[:, 0], points[:, 1], points[:, 2], incident=incident)
    return np.stack([f.e[c] for c in "xyz"], 1), np.stack([f.h[c] for c in "xyz"], 1) / host


def _vector_difference(ours, theirs):
    """Worst relative difference over the points; nan (scattnlay returns nan) counts as infinite."""
    d = np.linalg.norm(ours - theirs, axis=1) / np.linalg.norm(ours, axis=1)
    return float(np.max(np.where(np.isnan(d), np.inf, d)))


def compare_near_fields(cases=NEAR_FIELD_CASES):
    """{case: {quantity: difference}}: E and H inside and in the host against scattnlay, the centre
    against Bohren & Huffman (homogeneous spheres) and against scattnlay."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in cases:
            radii, n = CASES[name]
            l_max = ps.truncation_order(radii[-1], n[-1], WAVELENGTH, regime="near") + EXTRA_ORDERS
            sol = ps.solve(radii, np.array([n], complex), np.array([WAVELENGTH]), l_max=l_max)
            host = np.real(n[-1])
            values = {}
            for region, layers in near_field_points(radii).items():
                worst = []
                for radius, points in layers:
                    ours, theirs = stratify_fields(sol, points, host), scattnlay_fields(radii, n, points, l_max)
                    worst.append((radius, _vector_difference(ours[0], theirs[0]), _vector_difference(ours[1], theirs[1])))
                values[f"E {region}"] = max(e for _, e, _ in worst)
                values[f"H {region}"] = max(h for _, _, h in worst)
                if region == "inside":
                    values["inside by layer"] = worst  # (mid radius, E, H) from the core out
            if l_max <= TREAMS_NEAR_MAX_ORDER:
                points = np.concatenate([p for _, p in near_field_points(radii)["host"]])
                ours, theirs = stratify_fields(sol, points, 1.0, incident=False), treams_scattered_fields(
                    radii, n, points, l_max)
                values["E host, treams"] = _vector_difference(ours[0], theirs[0])
                values["H host, treams"] = _vector_difference(ours[1], theirs[1])
            centre = np.zeros((1, 3))
            (e0, h0), (e1, h1) = stratify_fields(sol, centre, host), scattnlay_fields(radii, n, centre, l_max)
            if len(radii) == 1:
                d1, mc1 = bohren_huffman_centre(n[0] / n[1], 2 * np.pi * host * radii[0] / WAVELENGTH)
                values["centre E, BH"] = _vector_difference(e0, np.array([[d1, 0, 0]]))
                values["centre H, BH"] = _vector_difference(h0, np.array([[0, mc1, 0]]))
            values["centre E, scattnlay"] = _vector_difference(e0, e1)
            values["centre H, scattnlay"] = _vector_difference(h0, h1)
            out[name] = values
    return out


# Emitters beside layered spheres against treams: a dipole is the l = 1 singular wave about itself (TM for an
# electric dipole, TE for a magnetic one); treams translates it to regular waves about the sphere, applies the
# T-matrix and evaluates the scattered field at the dipole. The rate is 1 + Re(E_s / E_j), E_j the field at the
# dipole of the regular wave with the same coefficients: the part of its own field that radiates, (2/3) i k^3 p
# in free space, so p* E_j is imaginary and no normalisation (or phase) of treams' waves enters; for a magnetic
# dipole read H for E. treams' T-matrix of a layered sphere is its own recursion.
# name -> (wavelength um, outer radii um, indices core first and host last, gap from the surface um,
#          "electric" | "magnetic", treams l_max ladder)
EMITTER_CASES = {
    "nanoshell, 10 nm silica core, 10 nm shell eps = -10 + i, 5 nm away":
        (0.6, [0.010, 0.020], [np.sqrt(2.13), METAL, 1.0], 0.005, "electric", (20, 40, 50)),
    "matryoshka metal/silica/metal in water, 10 nm away":
        (0.6, [0.010, 0.020, 0.030], [METAL, 1.45, METAL, 1.33], 0.010, "electric", (20, 40, 50)),
    "dielectric core-shell n = 1.5 / 2.5, R = 100 / 150 nm, electric":
        (0.7, [0.100, 0.150], [1.5, 2.5, 1.0], 0.050, "electric", (10, 20)),
    "dielectric core-shell n = 1.5 / 2.5, R = 100 / 150 nm, magnetic":
        (0.7, [0.100, 0.150], [1.5, 2.5, 1.0], 0.050, "magnetic", (10, 20)),
}


def stratify_emitter_rates(name):
    wavelength, radii, n, gap, dipole, _ = EMITTER_CASES[name]
    problem = ps.Problem("spheres", radii, n, wavelength, ps.PointDipole(radii[-1] + gap, dipole, len(radii)), 1e-10)
    return np.asarray(ps.solve_problem(problem)["total"][:2])


def treams_emitter_rates(name, l_max):
    """treams' (radial, tangential) total rates relative to the dipole in the host."""
    import treams

    wavelength, radii, n, gap, dipole, _ = EMITTER_CASES[name]
    k0 = 2 * np.pi / wavelength
    materials = [treams.Material(complex(v) ** 2) for v in n]
    tmatrix = treams.TMatrix.sphere(l_max, k0, radii, materials, poltype="parity")
    z0 = radii[-1] + gap
    source = treams.SphericalWaveBasis.default(1, positions=[[0, 0, z0]])
    pol = 1 if dipole == "electric" else 0
    rates = []
    for m, axis in ((0, 2), (1, 0)):  # along z: m = 0; along x: m = -1 minus m = +1
        c = np.zeros(len(source), complex)
        if m == 0:
            c[(source.m == 0) & (source.pol == pol)] = 1
        else:
            c[(source.m == -1) & (source.pol == pol)], c[(source.m == 1) & (source.pol == pol)] = 1, -1
        common = dict(basis=source, k0=k0, material=materials[-1], poltype="parity")
        own, regular = (treams.PhysicsArray(c, modetype=t, **common) for t in ("singular", "regular"))
        scattered = tmatrix @ own.expand(tmatrix.basis, "regular")
        field = (lambda a: a.efield) if dipole == "electric" else (lambda a: a.hfield)
        at = np.array([[0, 0, z0]])
        e_s, e_j = np.asarray(field(scattered)(at))[0, axis], np.asarray(field(regular)(at))[0, axis]
        rates.append(1 + np.real(e_s / e_j))
    return np.array(rates)


def compare_emitters(cases=tuple(EMITTER_CASES), ladders=None):
    """{case: (PyStratify rates, [(l_max, treams rates)])}."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in cases:
            ladder = (ladders or {}).get(name, EMITTER_CASES[name][-1])
            out[name] = stratify_emitter_rates(name), [(l, treams_emitter_rates(name, l)) for l in ladder]
    return out


def report_emitters(results):
    print("\n## Emitters beside layered spheres\n")
    print("Generated by the same script. Total decay rate relative to the dipole in the host, along (radial) and")
    print("across (tangential) the line through the centre, against treams (the dipole as an l = 1 singular wave,")
    print("translated to the sphere, through its T-matrix and back; rate = 1 + Re(E_s / E_j), H for a magnetic dipole); max")
    print("|treams / PyStratify - 1| per treams l_max, PyStratify at tolerance 1e-10. smuthi has no layered spheres.\n")
    print("| case | PyStratify radial | PyStratify tangential | treams l_max: difference |")
    print("|---|---|---|---|")
    for name, (exact, runs) in results.items():
        cells = ", ".join(f"{l}: {np.max(np.abs(rates / exact - 1)):.1e}" for l, rates in runs)
        print(f"| {name} | {exact[0]:.9f} | {exact[1]:.9f} | {cells} |")


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


def report_near_fields(results):
    quantities = ("E inside", "H inside", "E host", "H host", "E host, treams", "H host, treams", "centre E, BH",
                  "centre H, BH", "centre E, scattnlay", "centre H, scattnlay")
    print("\n## Near fields\n")
    print("Generated by the same script. max |F_ours - F_theirs| / |F_ours| over the points of a region: the")
    print(f"middle of the core and of every shell, and {', '.join(f'{f:g}' for f in HOST_RADII)} outer radii in the host,")
    print(f"each at polar angles {', '.join(f'{a:g}' for a in NEAR_FIELD_POLAR_DEG)} deg and azimuth "
          f"{NEAR_FIELD_AZIMUTH_DEG:g} deg; H compared as")
    print("scattnlay's H Z0 against PyStratify's H / n_host; nan = scattnlay returned nan at a point. Both")
    print(f"codes keep the near-field truncation (Allardice & Le Ru) + {EXTRA_ORDERS} orders. The treams columns are the")
    print(f"scattered field alone in the host, where treams' T-matrix fits (l_max <= {TREAMS_NEAR_MAX_ORDER}); its H is in")
    print("PyStratify's units (H_inc = n_host), so it is compared directly. At the centre PyStratify is")
    print("exact (l = 1 only) and matches Bohren & Huffman's d_1 and m c_1 (BH, homogeneous spheres); scattnlay's")
    print("own fields lose digits as k r -> 0, hence its separate centre columns. - = not applicable.\n")
    print("| case | " + " | ".join(quantities) + " |")
    print("|---|" + "---|" * len(quantities))
    for name, values in results.items():
        cells = [f"{values[q]:.1e}" if q in values else "-" for q in quantities]
        print(f"| {name} | " + " | ".join(cells).replace("inf", "nan") + " |")
    for name, values in results.items():
        if max(values["E inside"], values["H inside"]) <= 1e-10:
            continue
        print(f"\n**{name}, inside, layer by layer** (mid radius: E, H):")
        print(", ".join(f"{r:.4g} nm: {e:.1e}, {h:.1e}".replace("inf", "nan") for r, e, h in values["inside by layer"]) + ".")
        print("scattnlay's internal field degrades toward the core; PyStratify's internal coefficients of every layer")
        print("of this sphere agree with 60-digit transfer matrices to ~2e-12")
        print("(`tests/test_solver.py::test_coefficients_against_60_digit_transfer_matrices`).")


if __name__ == "__main__":
    report(compare())
    report_near_fields(compare_near_fields())
    report_emitters(compare_emitters())
