"""Spheres, plane wave: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_consistency.py                                                       #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-consistency.html, computed by this one file
from a PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE
and writes, beside itself, spheres_plane_wave_consistency.json (the curves the page draws, value for value) and
spheres_plane_wave_consistency.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps


@figure("mie-homogeneous", "Homogeneous sphere", "size parameter x", "Q_ext")
def plot_homogeneous_sphere():
    from sphere_checks import mie_ab

    xs = np.linspace(0.2, 30, 150)
    every = 5  # Bohren & Huffman every 1.0 in x
    lam, m = 1.0, 1.5

    def l_max(x):
        return int(x + 4 * x ** (1 / 3) + 4)

    def q_ext(shells, x):
        sol = ps.solve(shells, np.array([[m] * len(shells) + [1.0]]), np.array([lam]), l_max=l_max(x))
        return ps.cross_sections(sol).q_ext[0]

    def mie(x):
        a, b = mie_ab(m, x, l_max(x))
        ell = np.arange(1, l_max(x) + 1)
        return 2 / x ** 2 * np.sum((2 * ell + 1) * np.real(a + b))

    radii = xs / (2 * np.pi)
    ours = np.array([q_ext([R], x) for x, R in zip(xs, radii)])
    fifty = np.array([q_ext(np.linspace(R / 50, R, 50), x) for x, R in zip(xs, radii)])
    exact = np.array([mie(x) for x in xs[::every]])
    return result(
        xs,
        [line("q", "Q_ext, one shell (PyStratify)", ours),
         line("q50", "Q_ext, fifty identical shells (PyStratify)", fifty),
         dots("q_ref", "Q_ext, Bohren & Huffman Eq. 4.53", exact, every, size=xs.size)],
        worst=max(largest(ours[::every], exact), largest(fifty[::every], exact)),
        worst_label="largest |Q_ext − Q_Mie| over both curves",
        note="The textbook coefficients a_ℓ, b_ℓ (Riccati–Bessel functions from SciPy) summed directly, against the "
             "transfer-matrix recursion with one interface and with fifty interfaces between identical media.")


@figure("sixty-digit-coefficients", "Metal nanoshell", "order ℓ", "|T_ℓ|", log=True)
def plot_metal_nanoshell():
    from mp_reference import coefficients

    radii, n = [15.0, 20.0], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    orders = np.arange(1, 41)
    sol = ps.solve(radii, np.array([n]), np.array([600.0]), l_max=orders[-1])
    ours = {pol: np.abs(np.exp(sol.log_t[pol, 0])) for pol in (ps.TM, ps.TE)}
    ref = {pol: np.array([abs(complex(coefficients(radii, n, [1, 1, 1], 600.0, int(ell), pol == ps.TE)[0][-1][1]))
                          for ell in orders]) for pol in (ps.TM, ps.TE)}
    return result(
        orders,
        [line("tm", "|T_ℓ|, electric (TM), PyStratify", ours[ps.TM]),
         line("te", "|T_ℓ|, magnetic (TE), PyStratify", ours[ps.TE]),
         dots("tm_ref", "electric, 60 digits", ref[ps.TM]), dots("te_ref", "magnetic, 60 digits", ref[ps.TE])],
        worst=max(largest(ours[pol], ref[pol], relative=True) for pol in (ps.TM, ps.TE)),
        worst_label="largest relative difference over 40 orders",
        note="T_ℓ = B_host / A_host of the regular solution: the scattered amplitude of each order. The "
             "coefficients fall over fifty decades across the orders shown; double precision follows the 60-digit "
             "evaluation of the same recursion all the way down.")


@figure("centre-field", "Field at the centre", "size parameter x", "|field at the centre|")
def plot_field_at_the_centre():
    from mie_codes import bohren_huffman_centre

    xs = np.linspace(0.05, 4, 200)
    every = 5  # Bohren & Huffman every 0.1 in x
    m, lam = 3 + 0.01j, 600.0
    e, h = [], []
    for x in xs:
        sol = ps.solve([x * lam / (2 * np.pi)], np.array([[m, 1.0]]), np.array([lam]), l_max=30)
        f = ps.near_field(sol, [0.0], [0.0], [0.0])
        e.append(complex(np.ravel(f.e["x"])[0])), h.append(complex(np.ravel(f.h["y"])[0]))
    e, h = np.array(e), np.array(h)
    e_ref, h_ref = np.array([bohren_huffman_centre(m, x) for x in xs[::every]]).T
    return result(
        xs,
        [line("e", "|E(0)|, PyStratify", np.abs(e)), line("h", "|H(0)|, PyStratify", np.abs(h)),
         dots("e_ref", "|d₁|, Bohren & Huffman", np.abs(e_ref), every, size=xs.size),
         dots("h_ref", "|m c₁|, Bohren & Huffman", np.abs(h_ref), every, size=xs.size)],
        worst=max(np.max(np.abs(e[::every] / e_ref - 1)), np.max(np.abs(h[::every] / h_ref - 1))),
        worst_label="largest relative difference (complex fields)",
        note="Only the order ℓ = 1 survives at the centre, so E(0) = d₁ x̂ and H(0) = m c₁ ŷ exactly (Bohren & "
             "Huffman Eq. 4.53); PyStratify sums its full near field at r = 0. The peaks are the sphere's internal "
             "resonances.")


@figure("optical-theorem", "Optical theorem", "λ [nm]", "Q")
def plot_optical_theorem():
    from scipy.integrate import trapezoid

    lam = np.linspace(450, 1000, 111)
    every = 3  # the theorem and the integrated pattern every 15 nm
    radii, n = [50.0, 55.0], [1.45, 0.27 + 2.93j, 1.0]
    th = np.linspace(0, np.pi, 4001)
    area = np.pi * radii[-1] ** 2
    sols = [ps.solve(radii, n, w, l_max=20) for w in lam]
    ext = np.array([ps.cross_sections(sol).q_ext[0] for sol in sols])
    sca = np.array([ps.cross_sections(sol).q_sca[0] for sol in sols])
    ext_ref, sca_ref = [], []
    for w, sol in zip(lam[::every], sols[::every]):
        k = 2 * np.pi / w
        forward, _ = ps.scattering_amplitudes(sol, np.array([0.0]))
        p, q = ps.scattering_amplitudes(sol, th)
        ext_ref.append(4 * np.pi / k ** 2 * np.real(-forward[0, 0]) / area)
        sca_ref.append(np.pi / k ** 2 * trapezoid((np.abs(p[0]) ** 2 + np.abs(q[0]) ** 2) * np.sin(th), th) / area)
    return result(
        lam,
        [line("ext", "Q_ext, sum over orders (PyStratify)", ext),
         line("sca", "Q_sca, sum over orders (PyStratify)", sca),
         dots("ext_ref", "Q_ext, optical theorem (forward amplitude)", ext_ref, every, size=lam.size),
         dots("sca_ref", "Q_sca, integrated pattern", sca_ref, every, size=lam.size)],
        worst=max(largest(ext[::every], ext_ref, relative=True), largest(sca[::every], sca_ref, relative=True)),
        worst_label="largest relative difference",
        note="The extinction from the coefficients against 4π/k² Re S(0), the forward amplitude; the scattering "
             "from the coefficients against the far-field pattern integrated over 4001 angles (the trapezoid rule "
             "limits that pair to about 1e-7). The fixed gold index keeps the comparison about the method, not a "
             "material.")


@figure("rayleigh-limit", "Rayleigh limit", "size parameter x", "|a₁| / x³")
def plot_rayleigh_limit():
    xs = np.linspace(0.005, 1.0, 120)
    every = 3  # Rayleigh every 0.025 in x
    m, lam = 1.5 + 0.2j, 614.0
    ours = np.array([ps.solve([x * lam / (2 * np.pi)], [m, 1.0], lam, l_max=3).a[0, 0] for x in xs])
    ref = -2j * xs[::every] ** 3 / 3 * (m ** 2 - 1) / (m ** 2 + 2)
    small = xs[::every] <= 0.01
    return result(
        xs,
        [line("a1", "|a₁| / x³, PyStratify", np.abs(ours) / xs ** 3),
         dots("a1_ref", "Rayleigh: (2/3) |(m² − 1)/(m² + 2)|", np.abs(ref) / xs[::every] ** 3, every, size=xs.size)],
        worst=np.max(np.abs(ours[::every][small] / ref[small] - 1)),
        worst_label="relative difference for x ≤ 0.01",
        note="Rayleigh's a₁ = −(2i/3) x³ (m² − 1)/(m² + 2) is the first term in x, so divided by x³ it is a "
             "constant: PyStratify meets it as x → 0 and leaves it as the sphere grows (11% at x = 1); the "
             "difference falls as x² (1e-7 at x = 1e-4 in the test).")


@figure("lossless-absorbs-nothing", "Lossless multilayer", "size parameter x", "|Q_abs| / Q_ext", log=True)
def plot_lossless_multilayer():
    rng = np.random.default_rng(3)
    fractions, n = np.sort(rng.uniform(0.05, 1.0, 8)), np.r_[rng.uniform(1.2, 3.5, 8), 1.33]
    xs = np.linspace(0.5, 300, 150)
    lam = 614.0
    out = []
    for x in xs:
        cs = ps.cross_sections(ps.solve(fractions * x * lam / (2 * np.pi), n, lam))
        out.append(max(abs(cs.q_abs[0]) / cs.q_ext[0], 1e-17))
    return result(
        xs,
        [line("abs", "|Q_abs| / Q_ext, PyStratify", out)],
        worst=np.max(out),
        worst_label="largest |Q_abs| / Q_ext",
        note="Eight shells of random real indices 1.2–3.5: absorption is the difference of extinction and "
             "scattering, so anything above rounding would be energy the recursion invents; it stays below 1e-12 "
             "(the test's bound) to x = 300, growing slowly with the number of orders summed; floored at 1e-17 for "
             "the logarithmic axis.")


@figure("chiral-80-digits", "Chiral shell", "order ℓ", "|T|", log=True)
def plot_chiral_shell():
    from mp_reference import chiral_t_matrix

    radii, n, kappa, mu = [40.0, 55.0], [1.45 + 0.01j, 1.6 + 0.02j, 1.33], [0, 0.05 + 0.01j, 0], [1, 1, 1]
    orders = np.arange(1, 41)
    ours = np.exp(ps.solve_chiral(radii, n, kappa, 690.0, mu, l_max=orders[-1]).log_t_helicity[0])
    ref = np.array([np.exp(np.array(chiral_t_matrix(radii, n, kappa, mu, 690.0, int(ell)), dtype=complex))
                    for ell in orders])
    names = {(0, 0): "T₊₊", (1, 1): "T₋₋", (0, 1): "T₊₋"}
    return result(
        orders,
        [line(f"t{i}{j}", f"|{label}|, PyStratify", np.abs(ours[:, i, j])) for (i, j), label in names.items()]
        + [dots(f"t{i}{j}_ref", f"|{label}|, 80 digits", np.abs(ref[:, i, j])) for (i, j), label in names.items()],
        worst=np.max(np.abs(ours - ref)) / np.max(np.abs(ref)),
        worst_label="largest |T − T₈₀| / max |T₈₀|",
        note="The helicity blocks of the host T-matrix: direct 4×4 transfer matrices in 80-digit arithmetic "
             "(mp_reference.py) against PyStratify's scaled recursion in double precision, as the elements fall "
             "over many decades.")


@figure("graphene-sphere-plasmon", "Graphene-coated sphere", "radius [nm]", "λ [µm]")
def plot_graphene_coated_sphere():
    from scipy.optimize import brentq, minimize_scalar

    e1, e2, order = 2.1, 1.77, 1
    radii = np.linspace(5, 60, 23)

    def sheet(lam):
        return ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)

    def quasi_static(lam, radius):
        return (e1 * order + e2 * (order + 1) + 1j * sheet(lam) * order * (order + 1) * lam / (2 * np.pi * radius)).real

    def minus_t(lam, radius):
        sol = ps.solve([radius], [np.sqrt(e1), np.sqrt(e2)], lam, l_max=order, sheets={0: sheet(lam)})
        return -abs(np.exp(sol.log_t[0, 0, order - 1]))

    qs = np.array([brentq(quasi_static, 800, 200000, args=(radius,)) for radius in radii])
    peaks = np.array([minimize_scalar(minus_t, bounds=(0.9 * q, 1.2 * q), args=(radius,), method="bounded",
                                      options={"xatol": 1e-6 * q}).x for radius, q in zip(radii, qs)])
    return result(
        radii,
        [line("peak", "resonance, PyStratify (peak of |T₁|)", peaks / 1000),
         dots("qs", "quasi-static: ε₁ℓ + ε₂(ℓ + 1) + iσℓ(ℓ + 1)/(k₀R) = 0", qs / 1000)],
        worst=np.abs(peaks[0] / qs[0] - 1),
        worst_label="relative shift at R = 5 nm",
        note="The full solution with a conducting sheet on the sphere against the quasi-static resonance condition: "
             "they meet as R → 0 and part by retardation, too little to see here (5e-5 at 5 nm, 9e-4 at 60 nm); the "
             "test asserts the shift stays within (kR)².")


@figure("thin-film-sheet", "Thin film as a sheet", "λ [nm]", "Q_ext")
def plot_thin_film_as_a_sheet():
    eps_film, d, radius, core = 15.0 + 5.0j, 1.0, 200.0, 3.5
    lam = np.linspace(500, 900, 161)
    every = 4  # the explicit film every 10 nm

    def as_sheet(w):  # Q relative to the same geometric section: the sheet's sphere is d smaller
        sheet = ps.Sheet.from_film(eps_film, d, w, 1.33 ** 2)
        full = ps.solve([radius], [core, 1.33], w, l_max=40, sheets={0: sheet})
        return ps.cross_sections(full).ext[0] / (np.pi * (radius + d) ** 2)

    def as_film(w):
        return ps.cross_sections(ps.solve([radius, radius + d], [core, np.sqrt(eps_film), 1.33], w, l_max=40)).q_ext[0]

    ours = np.array([as_sheet(w) for w in lam])
    ref = np.array([as_film(w) for w in lam[::every]])
    return result(
        lam,
        [line("sheet", "Q_ext, 2D sheet (PyStratify)", ours),
         dots("film", "Q_ext, explicit 1 nm film", ref, every, size=lam.size)],
        worst=largest(ours[::every], ref, relative=True),
        worst_label="largest relative difference",
        note="Sheet.from_film turns a thin film into surface conductivities with the normal (out-of-plane) term; "
             "its error falls as d² (halving d quarters it, the test's check), against the film solved as a shell "
             "of its own.")


@figure("near-field-truncation", "Near-field truncation", "distance from the centre [nm]", "|E|²", log=True)
def plot_near_field_truncation():
    r = np.linspace(50.5, 400, 150)
    every = 5  # 120 orders every 11.7 nm
    sol = ps.solve([50.0], [0.25 + 3j, 1.33], 600.0, regime="near")
    ref = ps.solve([50.0], [0.25 + 3j, 1.33], 600.0, l_max=120)

    def e2(sol, r):  # along the polarisation, where the near field is strongest
        f = ps.near_field(sol, r, 0 * r, 0 * r)
        return sum(np.abs(np.ravel(f.e[c])) ** 2 for c in "xyz")

    ours, theirs = e2(sol, r), e2(ref, r[::every])
    return result(
        r,
        [line("e2", f"|E|², PyStratify near-field truncation ({sol.orders[-1]} orders)", ours),
         dots("e2_ref", "|E|², 120 orders", theirs, every, size=r.size)],
        worst=largest(ours[::every], theirs, relative=True),
        worst_label="largest relative difference",
        note="The far-field truncation (Wiscombe's, 5 orders here) is enough for cross sections but not at the "
             "surface, where it misses |E|² by 3e-5; regime='near', which the calculator uses for fields, adds the "
             "orders the distance needs.")


if __name__ == "__main__":
    run(__file__)
