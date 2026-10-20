"""Spheres, plane wave: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_consistency.py                                                       #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-consistency.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_plane_wave_consistency.json (the curves the page draws, value for value) and spheres_plane_wave_consistency.png (needs matplotlib).
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


@figure("mie-homogeneous", "Homogeneous sphere", "size parameter x", "Q_ext")
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


@figure("sixty-digit-coefficients", "Metal nanoshell", "order ℓ", "|T_ℓ|", log=True)
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


@figure("centre-field", "Field at the centre", "size parameter x", "|field at the centre|")
def centre_field():
    from mie_codes import bohren_huffman_centre

    xs = np.linspace(0.05, 4, 200)
    m, lam = 3 + 0.01j, 600.0
    e, h, e_ref, h_ref = [], [], [], []
    for x in xs:
        sol = ps.solve([x * lam / (2 * np.pi)], np.array([[m, 1.0]]), np.array([lam]), l_max=30)
        f = ps.near_field(sol, [0.0], [0.0], [0.0])
        d1, mc1 = bohren_huffman_centre(m, x)
        e.append(complex(np.ravel(f.e["x"])[0])), h.append(complex(np.ravel(f.h["y"])[0])), e_ref.append(d1), h_ref.append(mc1)
    e, h, e_ref, h_ref = map(np.array, (e, h, e_ref, h_ref))
    rel = max(np.max(np.abs(e / e_ref - 1)), np.max(np.abs(h / h_ref - 1)))
    return dict(x=sig(xs), series=[
        series("e", "|E(0)|, PyStratify", np.abs(e), "pystratify"), series("h", "|H(0)|, PyStratify", np.abs(h), "pystratify"),
        series("e_ref", "|d₁|, Bohren & Huffman", np.abs(e_ref), "reference"), series("h_ref", "|m c₁|, Bohren & Huffman", np.abs(h_ref), "reference")],
        worst=float(rel), worst_label="largest relative difference (complex fields)",
        note="Only the order ℓ = 1 survives at the centre, so E(0) = d₁ x̂ and H(0) = m c₁ ŷ exactly (Bohren & Huffman Eq. 4.53); "
             "PyStratify sums its full near field at r = 0. The peaks are the sphere's internal resonances.")


@figure("optical-theorem", "Optical theorem", "λ [nm]", "Q")
def optical_theorem():
    from scipy.integrate import trapezoid

    lam = np.linspace(450, 1000, 111)
    radii, n = [50.0, 55.0], [1.45, 0.27 + 2.93j, 1.0]
    th = np.linspace(0, np.pi, 4001)
    ext, sca, ext_ref, sca_ref = [], [], [], []
    area = np.pi * radii[-1] ** 2
    for w in lam:
        sol = ps.solve(radii, n, w, l_max=20)
        cs = ps.cross_sections(sol)
        k = 2 * np.pi / w
        forward, _ = ps.scattering_amplitudes(sol, np.array([0.0]))
        p, q = ps.scattering_amplitudes(sol, th)
        ext.append(cs.q_ext[0]), sca.append(cs.q_sca[0])
        ext_ref.append(4 * np.pi / k ** 2 * np.real(-forward[0, 0]) / area)
        sca_ref.append(np.pi / k ** 2 * trapezoid((np.abs(p[0]) ** 2 + np.abs(q[0]) ** 2) * np.sin(th), th) / area)
    return dict(x=sig(lam), series=[
        series("ext", "Q_ext, sum over orders (PyStratify)", ext, "pystratify"), series("sca", "Q_sca, sum over orders (PyStratify)", sca, "pystratify"),
        series("ext_ref", "Q_ext, optical theorem (forward amplitude)", ext_ref, "reference"), series("sca_ref", "Q_sca, integrated pattern", sca_ref, "reference")],
        worst=max(worst(ext, ext_ref, relative=True), worst(sca, sca_ref, relative=True)), worst_label="largest relative difference",
        note="The extinction from the coefficients against 4π/k² Re S(0), the forward amplitude; the scattering from the coefficients "
             "against the far-field pattern integrated over 4001 angles (the trapezoid rule limits that pair to about 1e-7). "
             "The fixed gold index keeps the comparison about the method, not a material.")


@figure("rayleigh-limit", "Rayleigh limit", "size parameter x", "|a₁| / x³")
def rayleigh_limit():
    xs = np.linspace(0.005, 1.0, 120)
    m, lam = 1.5 + 0.2j, 614.0
    ours = np.array([ps.solve([x * lam / (2 * np.pi)], [m, 1.0], lam, l_max=3).a[0, 0] for x in xs])
    ref = -2j * xs ** 3 / 3 * (m ** 2 - 1) / (m ** 2 + 2)
    small = xs <= 0.01
    return dict(x=sig(xs), series=[series("a1", "|a₁| / x³, PyStratify", np.abs(ours) / xs ** 3, "pystratify"),
                                   series("a1_ref", "Rayleigh: (2/3) |(m² − 1)/(m² + 2)|", np.abs(ref) / xs ** 3, "reference")],
                worst=float(np.max(np.abs(ours[small] / ref[small] - 1))), worst_label="relative difference for x ≤ 0.01",
                note="Rayleigh's a₁ = −(2i/3) x³ (m² − 1)/(m² + 2) is the first term in x, so divided by x³ it is a constant: PyStratify "
                     "meets it as x → 0 and leaves it as the sphere grows (11% at x = 1); the difference falls as x² (1e-7 at x = 1e-4 in the test).")


@figure("lossless-absorbs-nothing", "Lossless multilayer", "size parameter x", "|Q_abs| / Q_ext", log=True)
def lossless_absorbs_nothing():
    rng = np.random.default_rng(3)
    fractions, n = np.sort(rng.uniform(0.05, 1.0, 8)), np.r_[rng.uniform(1.2, 3.5, 8), 1.33]
    xs = np.linspace(0.5, 300, 150)
    lam = 614.0
    out = []
    for x in xs:
        cs = ps.cross_sections(ps.solve(fractions * x * lam / (2 * np.pi), n, lam))
        out.append(max(abs(cs.q_abs[0]) / cs.q_ext[0], 1e-17))
    return dict(x=sig(xs), series=[series("abs", "|Q_abs| / Q_ext, PyStratify", out, "pystratify")],
                worst=float(np.max(out)), worst_label="largest |Q_abs| / Q_ext",
                note="Eight shells of random real indices 1.2–3.5: absorption is the difference of extinction and scattering, so "
                     "anything above rounding would be energy the recursion invents; it stays below 1e-12 (the test's bound) to x = 300, "
                     "growing slowly with the number of orders summed; floored at 1e-17 for the logarithmic axis.")


@figure("chiral-80-digits", "Chiral shell", "order ℓ", "|T|", log=True)
def chiral_80_digits():
    from mp_reference import chiral_t_matrix

    radii, n, kappa, mu = [40.0, 55.0], [1.45 + 0.01j, 1.6 + 0.02j, 1.33], [0, 0.05 + 0.01j, 0], [1, 1, 1]
    L = 40
    sol = ps.solve_chiral(radii, n, kappa, 690.0, mu, l_max=L)
    ours = np.exp(sol.log_t_helicity[0])
    ref = np.array([np.exp(np.array(chiral_t_matrix(radii, n, kappa, mu, 690.0, l), dtype=complex)) for l in range(1, L + 1)])
    orders = list(range(1, L + 1))
    names = {(0, 0): "T₊₊", (1, 1): "T₋₋", (0, 1): "T₊₋"}
    out, refs = [], []
    for (i, j), label in names.items():
        out.append(series(f"t{i}{j}", f"|{label}|, PyStratify", np.abs(ours[:, i, j]), "pystratify"))
        refs.append(series(f"t{i}{j}_ref", f"|{label}|, 80 digits", np.abs(ref[:, i, j]), "reference"))
    return dict(x=orders, series=out + refs,
                worst=float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref))), worst_label="largest |T − T₈₀| / max |T₈₀|",
                note="The helicity blocks of the host T-matrix: direct 4×4 transfer matrices in 80-digit arithmetic (mp_reference.py) "
                     "against PyStratify's scaled recursion in double precision, as the elements fall over many decades.")


@figure("graphene-sphere-plasmon", "Graphene-coated sphere", "radius [nm]", "λ [µm]")
def graphene_sphere_plasmon():
    from scipy.optimize import brentq, minimize_scalar

    e1, e2, order = 2.1, 1.77, 1
    radii = np.linspace(5, 60, 23)
    peaks, qs = [], []
    for radius in radii:
        def quasi_static(lam):
            s_ = ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)
            return (e1 * order + e2 * (order + 1) + 1j * s_ * order * (order + 1) * lam / (2 * np.pi * radius)).real

        lam_qs = brentq(quasi_static, 800, 200000)

        def minus_t(lam):
            sol = ps.solve([radius], [np.sqrt(e1), np.sqrt(e2)], lam, l_max=order, sheets={0: ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)})
            return -abs(np.exp(sol.log_t[0, 0, order - 1]))

        peaks.append(minimize_scalar(minus_t, bounds=(0.9 * lam_qs, 1.2 * lam_qs), method="bounded", options={"xatol": 1e-6 * lam_qs}).x)
        qs.append(lam_qs)
    peaks, qs = np.array(peaks) / 1000, np.array(qs) / 1000
    return dict(x=sig(radii), series=[series("peak", "resonance, PyStratify (peak of |T₁|)", peaks, "pystratify"),
                                      series("qs", "quasi-static: ε₁ℓ + ε₂(ℓ + 1) + iσℓ(ℓ + 1)/(k₀R) = 0", qs, "reference")],
                worst=float(np.abs(peaks[0] / qs[0] - 1)), worst_label="relative shift at R = 5 nm",
                note="The full solution with a conducting sheet on the sphere against the quasi-static resonance condition: they meet as R → 0 "
                     "and part by retardation, too little to see here (5e-5 at 5 nm, 9e-4 at 60 nm); the test asserts the shift stays within (kR)².")


@figure("thin-film-sheet", "Thin film as a sheet", "λ [nm]", "Q_ext")
def thin_film_sheet():
    eps_film, d, radius, core = 15.0 + 5.0j, 1.0, 200.0, 3.5
    lam = np.linspace(500, 900, 161)
    ours, ref = [], []
    for w in lam:
        film = ps.solve([radius, radius + d], [core, np.sqrt(eps_film), 1.33], w, l_max=40)
        sheet = ps.Sheet.from_film(eps_film, d, w, 1.33 ** 2)
        full = ps.solve([radius], [core, 1.33], w, l_max=40, sheets={0: sheet})
        # Q relative to the same geometric section: the sheet's sphere is d smaller
        ours.append(ps.cross_sections(full).ext[0] / (np.pi * (radius + d) ** 2)), ref.append(ps.cross_sections(film).q_ext[0])
    return dict(x=sig(lam), series=[series("sheet", "Q_ext, 2D sheet (PyStratify)", ours, "pystratify"), series("film", "Q_ext, explicit 1 nm film", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Sheet.from_film turns a thin film into surface conductivities with the normal (out-of-plane) term; its error falls as d² "
                     "(halving d quarters it, the test's check), against the film solved as a shell of its own.")


@figure("near-field-truncation", "Near-field truncation", "distance from the centre [nm]", "|E|²", log=True)
def near_field_truncation():
    r = np.linspace(50.5, 400, 150)
    sol = ps.solve([50.0], [0.25 + 3j, 1.33], 600.0, regime="near")
    ref = ps.solve([50.0], [0.25 + 3j, 1.33], 600.0, l_max=120)
    points = (r, 0 * r, 0 * r)  # along the polarisation, where the near field is strongest
    e2 = lambda f: sum(np.abs(np.ravel(f.e[c])) ** 2 for c in "xyz")  # noqa: E731
    ours, theirs = e2(ps.near_field(sol, *points)), e2(ps.near_field(ref, *points))
    return dict(x=sig(r), series=[series("e2", f"|E|², PyStratify near-field truncation ({sol.orders[-1]} orders)", ours, "pystratify"),
                                  series("e2_ref", "|E|², 120 orders", theirs, "reference")],
                worst=worst(ours, theirs, relative=True), worst_label="largest relative difference",
                note="The far-field truncation (Wiscombe's, 5 orders here) is enough for cross sections but not at the surface, where it "
                     "misses |E|² by 3e-5; regime='near', which the calculator uses for fields, adds the orders the distance needs.")


if __name__ == "__main__":
    run(__file__)
