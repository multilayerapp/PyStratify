"""Comparison figures for multilayer.app's Verification row pages (benchmarks/MATRIX.md, docs/verification.md).

    pip install -e ".[docs,benchmarks]"
    python benchmarks/figures.py            # writes docs/figures/<name>.json for every figure
    python benchmarks/figures.py NAME ...   # some of them

Each figure draws PyStratify against the reference of one matrix row, as curves on a common axis, so the row page
shows the comparison rather than a number: the closed form of a single interface against the planar engine, Mie's
series against the multilayered-sphere recursion, the 60-digit transfer matrices against double precision, treams
against PyStratify for spheres, cylinders and emitters, the Sommerfeld integrals against the film emitter, the fibre's
dispersion relations against the guided poles. The references are the ones the benchmark scripts beside this file use
(`planar_checks.py`, `sphere_checks.py`, `unified_checks.py`, `mie_codes.py`, `emitter_references.py`,
`mp_reference.py`, `pystratify.references`); nothing here is asserted, the tests do that. A figure needing a package
that is not installed (treams) is skipped with a note.

`docs/verification.md` names which row each figure belongs to; `python -m pystratify.docs build` carries the JSON into
`docs/manifest.json`, and the site draws it with the same chart core as its Learn examples (reference curves dashed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy import inf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import pystratify as ps  # noqa: E402
from pystratify import planar  # noqa: E402

OUT = ROOT / "docs" / "figures"
FIGURES = {}


def figure(name, title, x_label, y_label, log=False):
    def register(fn):
        FIGURES[name] = (fn, dict(id=name, title=title, x_label=x_label, y_label=y_label, log=log))
        return fn
    return register


def series(key, label, y, role):
    return {"key": key, "label": label, "y": [float(v) for v in np.asarray(y, float)], "role": role}


def worst(ours, theirs, relative=False):
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    diff = np.max(np.abs(ours - theirs))
    return float(diff / np.max(np.abs(theirs))) if relative else float(diff)


def sig(values, digits=7):
    return [float(f"{v:.{digits}g}") for v in values]


# --- films, plane wave -------------------------------------------------------------------------------------------

@figure("fresnel-oblique", "A single interface, air to glass (n = 1.5) at 600 nm: the Fresnel equations", "angle of incidence [°]", "R")
def fresnel_oblique():
    from planar_checks import _forward_cos, _fresnel_amplitudes

    angles = np.linspace(0, 89.5, 180)
    n1, n2 = 1.0, 1.5
    ours, exact = {"s": [], "p": []}, {"s": [], "p": []}
    for a in angles:
        th = np.radians(a)
        kx = n1 * np.sin(th)
        cos_i, cos_f = _forward_cos(n1, kx), _forward_cos(n2, kx)
        for pol in ("s", "p"):
            ours[pol].append(planar.coh_tmm(pol, [n1, n2], [inf, inf], th, 0.6)["R"])
            exact[pol].append(abs(_fresnel_amplitudes(pol, n1, n2, cos_i, cos_f)[0]) ** 2)
    return dict(x=sig(angles), series=[
        series("R_s", "R_s, PyStratify", ours["s"], "pystratify"), series("R_p", "R_p, PyStratify", ours["p"], "pystratify"),
        series("R_s_ref", "R_s, Fresnel closed form", exact["s"], "reference"), series("R_p_ref", "R_p, Fresnel closed form", exact["p"], "reference")],
        worst=max(worst(ours["s"], exact["s"]), worst(ours["p"], exact["p"])), worst_label="largest |R − R_Fresnel| over both polarizations",
        note="Brewster's angle, where R_p vanishes, is arctan(1.5) = 56.3°; the two curves coincide to machine precision at every angle.")


@figure("airy-slab", "A 300 nm slab of n = 2 in air at normal incidence: the Airy formula", "λ [nm]", "R")
def airy_slab():
    lam = np.linspace(400, 800, 201)
    n, d = 2.0, 0.3
    ours = [planar.coh_tmm("s", [1.0, n, 1.0], [inf, d, inf], 0.0, w * 1e-3)["R"] for w in lam]
    r1 = ((n - 1) / (n + 1)) ** 2
    F = 4 * r1 / (1 - r1) ** 2
    delta = 2 * np.pi * n * d / (lam * 1e-3)
    airy = F * np.sin(delta) ** 2 / (1 + F * np.sin(delta) ** 2)
    return dict(x=sig(lam), series=[series("R", "R, PyStratify", ours, "pystratify"), series("R_ref", "R, Airy formula", airy, "reference")],
                worst=worst(ours, airy), worst_label="largest |R − R_Airy|",
                note="R = F sin²δ / (1 + F sin²δ) with F = 4R₁/(1 − R₁)², R₁ the single-surface reflectance and δ = 2πnd/λ: the fringes of a Fabry–Pérot slab.")


@figure("energy-conservation", "A lossless five-layer stack lit from glass, through the critical angle: 1 − R − T", "angle of incidence in glass [°]", "|1 − R − T|", log=True)
def energy_conservation():
    angles = np.linspace(0, 89, 179)
    n = [1.52, 2.3, 1.38, 2.3, 1.38, 2.3, 1.0]
    d = [inf, 0.08, 0.12, 0.08, 0.12, 0.08, inf]
    out = {}
    for pol in ("s", "p"):
        rows = [planar.coh_tmm(pol, n, d, np.radians(a), 0.55) for a in angles]
        out[pol] = np.maximum(np.abs(1 - np.array([r["R"] + r["T"] for r in rows])), 1e-17)
    return dict(x=sig(angles), series=[series("s", "|1 − R − T|, s polarization", out["s"], "pystratify"), series("p", "|1 − R − T|, p polarization", out["p"], "pystratify")],
                worst=float(max(out["s"].max(), out["p"].max())), worst_label="largest |1 − R − T|",
                note="Beyond the critical angle (41.1° in this glass) T is exactly zero and R is exactly one; the residual is floating-point rounding, floored at 10⁻¹⁷ for the logarithmic axis.")


# --- films, point dipole ----------------------------------------------------------------------------------------

@figure("sommerfeld-film-emitter", "A dipole in air above 60 nm of silver on glass, 633 nm: PyStratify against the Sommerfeld integrals", "height above the silver [nm]", "Γ / Γ₀", log=True)
def sommerfeld_film_emitter():
    import emitter_references as er

    n, d = er.STACKS["air above 60 nm silver on glass"]
    heights = np.geomspace(5, 400, 40)
    ours = np.array([er.stratify_rates(n, d, h) for h in heights])
    ref = np.array([er.reference_rates(n, d, h) for h in heights])
    return dict(x=sig(heights), series=[
        series("perp", "perpendicular, PyStratify", ours[:, 0], "pystratify"), series("par", "parallel, PyStratify", ours[:, 1], "pystratify"),
        series("perp_ref", "perpendicular, Sommerfeld integral", ref[:, 0], "reference"), series("par_ref", "parallel, Sommerfeld integral", ref[:, 1], "reference")],
        worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
        note="The reference integrates the Chance–Prock–Silbey integrands with Parratt's recursion for r_s and r_p, written without PyStratify; quenching rises as d⁻³ at contact and the surface plasmon of the silver sets the breakpoint of the evanescent integral.")


# --- spheres, plane wave ----------------------------------------------------------------------------------------

@figure("mie-homogeneous", "A homogeneous sphere, m = 1.5: Mie's series, one shell and fifty identical shells", "size parameter x", "Q_ext")
def mie_homogeneous():
    from sphere_checks import mie_ab

    xs = np.linspace(0.2, 30, 150)
    lam, m = 1.0, 1.5
    radii = xs / (2 * np.pi)
    ours, fifty, exact = [], [], []
    for x, R in zip(xs, radii):
        lmax = int(x + 4 * x ** (1 / 3) + 4)
        sol = ps.solve([R], np.array([[m, 1.0]]), np.array([lam]), l_max=lmax)
        ours.append(ps.cross_sections(sol).q_ext[0])
        shells = np.linspace(R / 50, R, 50)
        sol50 = ps.solve(shells, np.array([[m] * 50 + [1.0]]), np.array([lam]), l_max=lmax)
        fifty.append(ps.cross_sections(sol50).q_ext[0])
        a, b = mie_ab(m, x, lmax)
        ell = np.arange(1, lmax + 1)
        exact.append(2 / x ** 2 * np.sum((2 * ell + 1) * np.real(a + b)))
    return dict(x=sig(xs), series=[series("q", "Q_ext, one shell (PyStratify)", ours, "pystratify"), series("q50", "Q_ext, fifty identical shells (PyStratify)", fifty, "pystratify"),
                                   series("q_ref", "Q_ext, Bohren & Huffman Eq. 4.53", exact, "reference")],
                worst=max(worst(ours, exact), worst(fifty, exact)), worst_label="largest |Q_ext − Q_Mie| over both curves",
                note="The textbook coefficients a_ℓ, b_ℓ (Riccati–Bessel functions from SciPy) summed directly, against the transfer-matrix recursion with one interface and with fifty interfaces between identical media.")


@figure("bhmie-large-spheres", "Large absorbing spheres, m = 1.5 + 0.1i: PyStratify against BHMIE to x = 5000", "size parameter x", "Q_ext")
def bhmie_large_spheres():
    from pystratify.references import bhmie

    xs = np.geomspace(1, 5000, 120)
    m = 1.5 + 0.1j
    ours, ref = [], []
    for x in xs:
        R = x / (2 * np.pi)
        lmax = int(x + 4 * x ** (1 / 3) + 2) + 20  # both codes 20 orders beyond Wiscombe's truncation
        sol = ps.solve([R], np.array([[m, 1.0]]), np.array([1.0]), l_max=lmax)
        ours.append(ps.cross_sections(sol).q_ext[0])
        ref.append(bhmie(x, m, lmax)[0])
    return dict(x=sig(xs), series=[series("q", "Q_ext, PyStratify", ours, "pystratify"), series("q_ref", "Q_ext, BHMIE (Bohren & Huffman, Appendix A)", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="BHMIE's downward recurrence for the logarithmic derivative, ported from the book's Appendix A, against the logarithmic transfer-matrix recursion, both summed 20 orders beyond Wiscombe's truncation; the x axis is logarithmic and the extinction tends to 2 with the ripple of the edge waves. For a weakly absorbing or lossless sphere BHMIE's double-precision recurrence drifts at large x (3e-4 in Q_ext at x = 1481 for m = 1.33 + 1e-4 i, 6e-4 for m = 1.33), where PyStratify agrees with the same series summed in 40-digit arithmetic (pystratify.references.sphere_extinction) to 2e-13: the absorbing case here is the one the test suite asserts.")


@figure("sixty-digit-coefficients", "A 5 nm metal nanoshell on a 15 nm silica core, 600 nm: Mie coefficients against 60-digit transfer matrices", "order ℓ", "|T_ℓ|", log=True)
def sixty_digit_coefficients():
    from mp_reference import coefficients

    radii, n = [15.0, 20.0], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    lmax = 40
    sol = ps.solve(radii, np.array([n]), np.array([600.0]), l_max=lmax)
    orders = np.arange(1, lmax + 1)
    ours = {ps.TM: np.abs(np.exp(sol.log_t[ps.TM, 0])), ps.TE: np.abs(np.exp(sol.log_t[ps.TE, 0]))}
    ref = {}
    for te in (False, True):
        vals = []
        for l in orders:
            regular, _ = coefficients(radii, n, [1, 1, 1], 600.0, int(l), te)
            vals.append(abs(complex(regular[-1][1])))
        ref[ps.TE if te else ps.TM] = np.array(vals)
    return dict(x=[int(l) for l in orders], series=[
        series("tm", "|T_ℓ|, electric (TM), PyStratify", ours[ps.TM], "pystratify"), series("te", "|T_ℓ|, magnetic (TE), PyStratify", ours[ps.TE], "pystratify"),
        series("tm_ref", "electric, 60 digits", ref[ps.TM], "reference"), series("te_ref", "magnetic, 60 digits", ref[ps.TE], "reference")],
        worst=max(worst(ours[ps.TM], ref[ps.TM], relative=True), worst(ours[ps.TE], ref[ps.TE], relative=True)), worst_label="largest relative difference over 40 orders",
        note="T_ℓ = B_host / A_host of the regular solution: the scattered amplitude of each order. The coefficients fall over fifty decades across the orders shown; double precision follows the 60-digit evaluation of the same recursion all the way down.")


@figure("treams-nanoshell", "A nanoshell (10 nm silica core, 10 nm shell of ε = −10 + i): PyStratify against treams", "λ [nm]", "Q")
def treams_nanoshell():
    import treams

    lam = np.linspace(400, 900, 101)
    radii, n = [10.0, 20.0], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    sol = ps.solve(radii, np.array([n] * lam.size), lam, l_max=12)
    cs = ps.cross_sections(sol)
    area = np.pi * radii[-1] ** 2
    ext, sca = [], []
    for w in lam:
        tm = treams.TMatrix.sphere(12, 2 * np.pi / w, radii, [treams.Material(complex(v) ** 2) for v in n])
        ext.append(np.real(tm.xs_ext_avg) / area), sca.append(np.real(tm.xs_sca_avg) / area)
    return dict(x=sig(lam), series=[series("ext", "Q_ext, PyStratify", cs.q_ext, "pystratify"), series("sca", "Q_sca, PyStratify", cs.q_sca, "pystratify"),
                                    series("ext_ref", "Q_ext, treams", ext, "reference"), series("sca_ref", "Q_sca, treams", sca, "reference")],
                worst=max(worst(cs.q_ext, ext, relative=True), worst(cs.q_sca, sca, relative=True)), worst_label="largest relative difference",
                note="treams (Beutel et al.) builds the T-matrix of a layered sphere from its own vector spherical waves; the fixed metal index keeps the shell's plasmon at one wavelength, so the comparison is of the two codes, not of a material.")


# --- spheres, point dipole --------------------------------------------------------------------------------------

@figure("sphere-decay-mpmath", "A dipole beside a 25 nm sphere of m = 0.05 + 3.5i at 600 nm: decay rates against the 40-digit series", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def sphere_decay_mpmath():
    from pystratify.references import sphere_decay_rates

    gaps = np.geomspace(1, 300, 36)
    radius, m, lam = 25.0, 0.05 + 3.5j, 600.0
    r = radius + gaps
    ours = ps.decay_rates([radius], np.array([m, 1.0]), lam, r, tol=1e-10)
    ref = np.array([sphere_decay_rates(radius, m, lam, rr, orders=400, dps=40) for rr in r])
    return dict(x=sig(gaps), series=[
        series("tot_perp", "total, radial, PyStratify", ours.total[:, 0], "pystratify"), series("tot_par", "total, tangential, PyStratify", ours.total[:, 1], "pystratify"),
        series("rad_perp", "radiative, radial, PyStratify", ours.radiative[:, 0], "pystratify"), series("rad_par", "radiative, tangential, PyStratify", ours.radiative[:, 1], "pystratify"),
        series("tot_perp_ref", "total, radial, 40 digits", ref[:, 0], "reference"), series("tot_par_ref", "total, tangential, 40 digits", ref[:, 1], "reference"),
        series("rad_perp_ref", "radiative, radial, 40 digits", ref[:, 2], "reference"), series("rad_par_ref", "radiative, tangential, 40 digits", ref[:, 3], "reference")],
        worst=worst(np.column_stack([ours.total, ours.radiative]), ref, relative=True), worst_label="largest relative difference over the four rates",
        note="Chew's sums over 400 orders in mpmath at 40 digits against PyStratify's adaptive order and logarithmic amplitudes; the total rate climbs as d⁻³ at contact (quenching) while the radiative rate stays finite.")


@figure("treams-dipole-nanoshell", "A dipole 5–60 nm from a nanoshell (10 nm silica core, 10 nm shell of ε = −10 + i), 600 nm: PyStratify against treams", "gap [nm]", "Γ / Γ₀", log=True)
def treams_dipole_nanoshell():
    import treams

    gaps = np.geomspace(5, 60, 24)
    radii, n = [0.010, 0.020], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    lam, l_max = 0.6, 50
    k0 = 2 * np.pi / lam
    materials = [treams.Material(complex(v) ** 2) for v in n]
    tmatrix = treams.TMatrix.sphere(l_max, k0, radii, materials, poltype="parity")
    ours, ref = [], []
    for gap in gaps:
        problem = ps.Problem("spheres", radii, n, lam, ps.PointDipole(radii[-1] + gap * 1e-3, "electric", len(radii)), 1e-10)
        ours.append(np.asarray(ps.solve_problem(problem)["total"][:2]))
        z0 = radii[-1] + gap * 1e-3
        source = treams.SphericalWaveBasis.default(1, positions=[[0, 0, z0]])
        rates = []
        for m, axis in ((0, 2), (1, 0)):
            c = np.zeros(len(source), complex)
            if m == 0:
                c[(source.m == 0) & (source.pol == 1)] = 1
            else:
                c[(source.m == -1) & (source.pol == 1)], c[(source.m == 1) & (source.pol == 1)] = 1, -1
            common = dict(basis=source, k0=k0, material=materials[-1], poltype="parity")
            own, regular = (treams.PhysicsArray(c, modetype=t, **common) for t in ("singular", "regular"))
            scattered = tmatrix @ own.expand(tmatrix.basis, "regular")
            at = np.array([[0, 0, z0]])
            e_s, e_j = np.asarray(scattered.efield(at))[0, axis], np.asarray(regular.efield(at))[0, axis]
            rates.append(1 + np.real(e_s / e_j))
        ref.append(rates)
    ours, ref = np.array(ours), np.array(ref)
    return dict(x=sig(gaps), series=[
        series("perp", "radial, PyStratify", ours[:, 0], "pystratify"), series("par", "tangential, PyStratify", ours[:, 1], "pystratify"),
        series("perp_ref", "radial, treams", ref[:, 0], "reference"), series("par_ref", "tangential, treams", ref[:, 1], "reference")],
        worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
        note="treams translates the dipole's ℓ = 1 singular wave to the sphere's centre, scatters it with the T-matrix (50 orders; closer than 5 nm its truncation no longer converges, PyStratify's adaptive order does) and reads the scattered field back at the dipole; PyStratify places the jump in its transfer-matrix recursion.")


# --- cylinders --------------------------------------------------------------------------------------------------

@figure("treams-cylinder-oblique", "A cylinder of radius 100 nm, n = 1.5 + 0.01i, at 60° to its axis: cross widths against treams", "λ [nm]", "width / (2a)")
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


@figure("cylinder-bessel-direct", "A cylinder of n = 1.5 in air at 600 nm, normal incidence: the coefficients from the Bessel boundary equations", "radius [nm]", "Q_sca")
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


@figure("fibre-dispersion", "A step-index fibre, n = 2, radius 150 nm, in air: guided modes against the dispersion relations", "λ [µm]", "n_eff")
def fibre_dispersion():
    from scipy.optimize import brentq
    from scipy.special import j0, j1, k0, k1
    from pystratify.cylinder_emission import CylinderSource

    radius, n1, n2 = .15, 2.0, 1.0
    lams = np.linspace(0.25, 0.38, 27)  # above 0.392 µm (V = 2.405) TE01 and TM01 are cut off and only HE11 guides
    te, tm, he, te_ref, tm_ref = [], [], [], [], []
    for lam in lams:
        V = 2 * np.pi / lam * radius

        def relation(b, c):
            u, w = V * np.sqrt(n1 ** 2 - b ** 2), V * np.sqrt(b ** 2 - n2 ** 2)
            return j1(u) / (u * j0(u)) + (c * (n2 / n1) ** 2 if c else 1) * k1(w) / (w * k0(w))

        upper = np.sqrt(n1 ** 2 - (2.404825557695773 / V) ** 2)
        exact = sorted(brentq(relation, n2 + 1e-12, upper - 1e-12, args=(c,), xtol=1e-15) for c in (0, 1))
        poles = np.sort(CylinderSource([radius], [n1, n2], lam, .25).guided_poles())
        # the poles nearest each relation's root are TE01 and TM01; HE11, with no cutoff, is the largest
        te.append(poles[np.argmin(np.abs(poles - exact[0]))]), tm.append(poles[np.argmin(np.abs(poles - exact[1]))]), he.append(poles[-1])
        te_ref.append(exact[0]), tm_ref.append(exact[1])
    return dict(x=sig(lams), series=[series("te", "TE₀₁, PyStratify", te, "pystratify"), series("tm", "TM₀₁, PyStratify", tm, "pystratify"), series("he", "HE₁₁, PyStratify", he, "pystratify"),
                                     series("te_ref", "TE₀₁, dispersion relation", te_ref, "reference"), series("tm_ref", "TM₀₁, dispersion relation", tm_ref, "reference")],
                worst=max(worst(te, te_ref), worst(tm, tm_ref)), worst_label="largest |n_eff − n_eff(relation)| for TE₀₁ and TM₀₁",
                note="Snyder & Love's TE₀₁ and TM₀₁ relations, J₁(u)/(u J₀(u)) + c K₁(w)/(w K₀(w)) = 0, solved by bisection, against the poles PyStratify finds behind a dipole's guided channel; HE₁₁ (no closed form of this kind, and no cutoff) is shown as found; the shorter wavelengths guide higher modes too, not drawn. TE₀₁ and TM₀₁ reach cutoff at 0.392 µm, where V = 2.405.")


# --- CLI ---------------------------------------------------------------------------------------------------------

def build(names=None):
    OUT.mkdir(exist_ok=True)
    written = []
    for name, (fn, meta) in FIGURES.items():
        if names and name not in names:
            continue
        t = time.time()
        try:
            data = fn()
        except ImportError as error:
            print(f"skip {name}: {error}", flush=True)
            continue
        out = {**meta, **data, "script": "benchmarks/figures.py", "function": fn.__name__}
        out["x"] = [float(v) for v in out["x"]]
        for s in out["series"]:
            s["y"] = sig(s["y"], 8)
        (OUT / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
        written.append(name)
        print(f"{name}: {len(out['x'])} points, {len(out['series'])} series, worst {out['worst']:.2e} ({time.time() - t:.1f} s)", flush=True)
    return written


if __name__ == "__main__":
    build(sys.argv[1:] or None)
