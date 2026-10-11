"""Spheres under plane waves: textbook Mie, extended precision and limits (benchmarks/MATRIX.md, spheres).

    python benchmarks/sphere_checks.py        # prints SPHERE_CHECKS.md (~2 min)

Each check is independent of the code it tests: Bohren & Huffman's coefficients written out with scipy,
their printed reference values, the optical theorem, the Rayleigh limit, energy conservation, 60- and
80-digit transfer matrices (``mp_reference.py``), published polarisabilities of a small chiral sphere,
the quasi-static plasmons of a graphene-coated sphere and the thin-film limit of a 2D sheet. The tests
(``tests/test_physics.py``, ``test_solver.py``, ``test_farfield.py``, ``test_chiral.py``,
``test_sheets.py``) assert every row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq, minimize_scalar
from scipy.special import spherical_jn, spherical_yn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402
from checks import close, flag, report, row  # noqa: E402
from mp_reference import chiral_t_matrix, coefficients  # noqa: E402
from pystratify import TE, TM  # noqa: E402

LAM = 614.0
AU = 0.27 + 2.93j
MATRYOSHKA = ([10.0, 13.0, 36.0, 48.0], [1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33])  # 3-nm Au shell


def riccati(l, z, derivative=False):
    """psi, xi (or derivatives) from scipy's spherical Bessel functions."""
    j, y = spherical_jn(l, z), spherical_yn(l, z)
    if not derivative:
        return z * j, z * (j + 1j * y)
    jd, yd = spherical_jn(l, z, derivative=True), spherical_yn(l, z, derivative=True)
    return j + z * jd, (j + 1j * y) + z * (jd + 1j * yd)


def mie_ab(m, x, lmax):
    """Bohren & Huffman Eq. 4.53, written independently."""
    l = np.arange(1, lmax + 1)
    (psi, xi), (dpsi, dxi) = riccati(l, x), riccati(l, x, True)
    psim, dpsim = riccati(l, m * x)[0], riccati(l, m * x, True)[0]
    a = (m * psim * dpsi - psi * dpsim) / (m * psim * dxi - xi * dpsim)
    b = (psim * dpsi - m * psi * dpsim) / (psim * dxi - m * xi * dpsim)
    return a, b


def _sol(radii, n, mu=None, wavelength=LAM, L=20):
    return ps.solve(radii, n, wavelength, mu, l_max=L)


# ---------------------------------------------------------------- textbook Mie, limits

MIE_CASES = ((1.55, 5.213), (1.5 + 0.1j, 2.0), (AU / 1.33, 1.1))


def homogeneous_sphere_equals_mie(m, x):
    nh = 1.33
    r = x * LAM / (2 * np.pi * nh)
    L = ps.truncation_order(r, nh, LAM) + 5
    S = _sol([r], [m * nh, nh], L=L)
    a, b = mie_ab(m, x, L)
    name = f"homogeneous sphere m = {m:.4g}, x = {x:g}: Bohren & Huffman a_n, b_n"
    return [close(name, "a_n", S.a[0], a, 1e-10, 1e-15), close(name, "b_n", S.b[0], b, 1e-10, 1e-15)]


def fifty_shells_are_one_sphere():
    radii = np.linspace(2.0, 100.0, 50)
    layered = ps.solve(radii, [2.0 + 0.1j] * 50 + [1.33], 600.0, l_max=30)
    single = ps.solve([100.0], [2.0 + 0.1j, 1.33], 600.0, l_max=30)
    name = "fifty shells of one material against one sphere"
    return [close(name, "a_n", layered.a, single.a, 1e-11, 1e-8), close(name, "b_n", layered.b, single.b, 1e-11, 1e-8)]


def bohren_huffman_reference_values():
    x = 2 * np.pi * 0.525 / 0.6328  # BHMIE's case: m = 1.55, a = 0.525 um, lambda = 0.6328 um
    r = x * LAM / (2 * np.pi)
    S = _sol([r], [1.55, 1.0], L=20)
    cs = ps.cross_sections(S)
    l = S.orders
    q_back = np.abs(np.sum((2 * l + 1) * (-1) ** l * (S.a[0] - S.b[0]))) ** 2 / x ** 2
    sol = ps.solve([r], [1.55, 1.0], LAM, l_max=20)
    back = ps.scattering_pattern(sol, np.pi)
    name = "Bohren & Huffman's BHMIE case (m = 1.55, a = 0.525 um, 0.6328 um)"
    return [row(name, "|Q_ext - 3.10543|", abs(cs.q_ext[0] - 3.10543), 1e-5), row(name, "|Q_abs|", abs(cs.q_abs[0]), 1e-12),
            row(name, "|Q_back - 2.92534|", abs(q_back - 2.92534), 1e-5),
            row(name, "|4 pi dC/dOmega(180 deg) / (pi a^2) - 2.92534| (from the pattern)",
                abs(4 * np.pi * back.differential_cross_section[0] / (np.pi * r ** 2) - 2.92534), 1e-5)]


def optical_theorem_and_backscattering():
    S = _sol([50.0, 55.0], [1.45, AU, 1.0])
    cs = ps.cross_sections(S)
    par, per = ps.scattering_amplitudes(S, np.array([0.0, np.pi / 2, np.pi]))
    k = 2 * np.pi / LAM
    p2, _ = ps.scattering_amplitudes(S, np.array([np.pi - 1e-6]))
    th = np.linspace(0, np.pi, 4001)
    p, q = ps.scattering_amplitudes(S, th)
    csca = np.pi / k ** 2 * trapezoid((np.abs(p[0]) ** 2 + np.abs(q[0]) ** 2) * np.sin(th), th)
    name = "silica core, 5 nm gold shell in air"
    return [row(name, "|C_ext / (4 pi/k^2 Re(-S(0))) - 1| (optical theorem)", abs(cs.ext[0] / (4 * np.pi / k ** 2 * np.real(-par[0, 0])) - 1), 1e-12),
            flag(name, "amplitudes finite", np.all(np.isfinite(par))),
            row(name, "|S_par(0) / S_perp(0) - 1|", abs(par[0, 0] / per[0, 0] - 1), 1e-6),
            row(name, "|S(pi - 1e-6) / S(pi) - 1| (backscattering limit)", abs(p2[0, 0] / par[0, 2] - 1), 1e-8),
            row(name, "|integral of the pattern / C_sca - 1|", abs(csca / cs.sca[0] - 1), 1e-5)]


def rayleigh_limit():
    m, x = 1.5 + 0.2j, 1e-4
    a1 = ps.solve([x * LAM / (2 * np.pi)], [m, 1.0], LAM, l_max=3).a[0, 0]
    return [row("sphere m = 1.5 + 0.2i, x = 1e-4", "|a_1 / (-2i x^3/3 (m^2 - 1)/(m^2 + 2)) - 1|",
                abs(a1 / (-2j * x ** 3 / 3 * (m ** 2 - 1) / (m ** 2 + 2)) - 1), 1e-7)]


LOSSLESS_SIZES = (0.5, 20.0, 300.0)


def lossless_multilayers_absorb_nothing(size):
    rng = np.random.default_rng(int(size * 10))
    radii = np.sort(rng.uniform(0.05, 1.0, 8)) * size * LAM / (2 * np.pi)
    n = np.r_[rng.uniform(1.2, 3.5, 8), 1.33]
    cs = ps.cross_sections(ps.solve(radii, n, LAM))
    return [row(f"random lossless 8-shell sphere, x = {size:g}", "|Q_abs| / Q_ext", abs(cs.q_abs[0]) / cs.q_ext[0], 1e-12, "<")]


def amplitude_matrix_against_textbook_mie():
    """S1, S2 from BH a_n, b_n with pi_n = P_n' and tau_n from Legendre-series derivatives."""
    m, x, nh = 1.5 + 0.1j, 4.0, 1.33
    r = x * LAM / (2 * np.pi * nh)
    L = ps.truncation_order(r, nh, LAM) + 5
    sol = ps.solve([r], [m * nh, nh], LAM, l_max=L)
    a, b = mie_ab(m, x, L)
    theta = np.linspace(0, np.pi, 13)
    mu = np.cos(theta)
    s1, s2 = np.zeros(theta.size, complex), np.zeros(theta.size, complex)
    for order in range(1, L + 1):
        p = np.polynomial.legendre.Legendre.basis(order)
        pi = p.deriv()(mu)
        tau = mu * pi - (1 - mu ** 2) * p.deriv(2)(mu)
        w = (2 * order + 1) / (order * (order + 1))
        s1 += w * (a[order - 1] * pi + b[order - 1] * tau)
        s2 += w * (a[order - 1] * tau + b[order - 1] * pi)
    got = ps.amplitude_matrix(sol, theta)
    par, per = ps.scattering_amplitudes(sol, theta)
    name = "sphere m = 1.5 + 0.1i, x = 4: amplitude matrix from Legendre series"
    return [close(name, "S1", got[0][0], s1, 1e-10, 1e-8), close(name, "S2", got[1][0], s2, 1e-10, 1e-8),
            flag(name, "S3 = S4 = 0", np.all(got[2] == 0) and np.all(got[3] == 0)),
            close(name, "S_par against -S2", par, -got[1], 1e-14, 1e-8), close(name, "S_perp against -S1", per, -got[0], 1e-14, 1e-8)]


# ---------------------------------------------------------------- extended precision

def log_error(log_value, exact):
    if exact == 0:
        return 0.0
    d = complex(mp.log(exact)) - log_value
    return abs(d.real) + abs(np.exp(1j * d.imag) - 1)


PRECISION_CASES = (
    ([50, 70, 90], [1.45 + 0.2j, 2.0, 1.6 + 0.05j, 1.0], 1e-10),
    (*MATRYOSHKA, 1e-10),
    ([30, 31, 60], [2.5 + 0.01j, 0.05 + 4.0j, 1.5, 1.33], 1e-10),  # 1-nm Ag-like film
    # nearly index-matched: B is set by a contrast of 1e-6, so it is known to ~ l eps / contrast
    ([40, 60], [1.5, 1.5 + 1e-6, 1.5 + 2e-6], 1e-7),
    ([40, 60], [1.5, 1.6 - 0.02j, 1.33], 1e-10),  # gain shell
    # mie_codes.py's graded sphere, where scattnlay's internal fields fail in the inner layers
    (list(np.linspace(114.6, 1146.0, 10)), list(np.linspace(1.6, 1.35, 10)) + [1.0], 1e-10),
)


def coefficients_against_60_digits(radii, n, tol, wavelength=690.0):
    mu = [1.0] * len(n)
    sol = ps.solve(radii, n, wavelength, mu, l_max=160)
    worst = 0.0
    for pol in (TM, TE):
        for order in (1, 7, 40, 100, 160):
            regular, outgoing = coefficients(radii, n, mu, wavelength, order, te=pol == TE)
            for s in range(len(n)):
                worst = max(worst, log_error(sol.log_a[pol, s, 0, order - 1], regular[s][0]),
                            log_error(sol.log_b[pol, s, 0, order - 1], regular[s][1]),
                            log_error(sol.log_b_out[pol, s, 0, order - 1], outgoing[s][1]))
    label = f"{len(radii)} layers, outer {radii[-1]:.4g} nm, n = {np.round(np.asarray(n, complex)[:3], 4).tolist()}{'...' if len(n) > 3 else ''}"
    return [row(f"Mie coefficients of every shell, {label}: 60-digit transfer matrices (orders 1-160)",
                "max log error of A, B (regular) and B (outgoing)", worst, tol, "<")]


def _norm_error(log_got, log_ref):
    """max |got - ref| / max |ref| for matrices given as complex logarithms."""
    shift = np.max(np.real(log_ref))
    return float(np.max(np.abs(np.exp(log_got - shift) - np.exp(log_ref - shift))))


CHIRAL_T_CASES = (
    ([40.0, 55.0], [1.45 + 0.01j, 1.6 + 0.02j, 1.33], [0, 0.05 + 0.01j, 0], [1, 1, 1], 1e-11),
    ([30.0, 33.0, 60.0], [1.5, AU, 1.55 + 0.001j, 1.33], [0.02, 0, 0.1 + 0.003j, 0], [1, 1, 1.2, 1], 1e-11),
    (MATRYOSHKA[0], MATRYOSHKA[1], [0.01, 0, 0.2 + 0.01j, 0, 0], [1] * 5, 1e-11),
    ([40.0, 60.0], [1.5, 1.6 - 0.02j, 1.33], [0.03, 0.05, 0], [1, 1, 1], 1e-11),  # gain shell
    # a chirality contrast of 1e-6 alone: T is known to ~ eps / contrast
    ([40.0, 60.0], [1.5, 1.5, 1.5], [0, 1e-6, 0], [1, 1, 1], 1e-9),
)


def chiral_t_matrix_against_80_digits(radii, n, kappa, mu, tol):
    sol = ps.solve_chiral(radii, n, kappa, 690.0, mu, l_max=160)
    worst = max(_norm_error(sol.log_t_helicity[0, order - 1], np.array(chiral_t_matrix(radii, n, kappa, mu, 690.0, order)))
                for order in (1, 7, 40, 100, 160))
    return [row(f"chiral multilayer {radii}, kappa {kappa}: 80-digit transfer matrices (orders 1-160)",
                "max |T - T_80| / max |T_80|", worst, tol, "<")]


CHIRAL_LAYER_CASES = (
    ([30.0, 33.0, 60.0], [1.5, AU, 1.55 + 0.001j, 1.33], [0.02, 0, 0.1 + 0.003j, 0], [1, 1, 1.2, 1]),
    (MATRYOSHKA[0], MATRYOSHKA[1], [0.01, 0, 0.2 + 0.01j, 0, 0], [1] * 5),
    ([40.0, 60.0], [1.5, 1.6 - 0.02j, 1.33], [0.03, 0.05, 0], [1, 1, 1]),
)


def chiral_layer_solutions_against_80_digits(radii, n, kappa, mu):
    """Regular (unit incident wave) and outgoing (unit outgoing wave, nothing incoming) solutions in every layer."""
    from pystratify.chiral import log_matmul

    sol = ps.solve_chiral(radii, n, kappa, 690.0, mu, l_max=160)
    worst = 0.0
    for order in (1, 2, 7, 40, 100, 160):
        regular = chiral_t_matrix(radii, n, kappa, mu, 690.0, order, amplitudes_only=True)
        outgoing = chiral_t_matrix(radii, n, kappa, mu, 690.0, order, outgoing=True)
        for j in range(len(radii) + 1):
            alpha = sol.log_alpha[0, j, order - 1]
            beta_out = sol.log_beta_out[0, j, order - 1]
            pairs = [(alpha, regular[j][0]), (beta_out, outgoing[j][1])]
            if j:
                pairs.append((log_matmul(sol.log_r[0, j, order - 1], alpha), regular[j][1]))
            if j < len(radii):
                pairs.append((log_matmul(sol.log_s[0, j, order - 1], beta_out), outgoing[j][0]))
                pairs.append((log_matmul(sol.log_in[0, j, order - 1], regular[j + 1][0]), regular[j][0]))
                pairs.append((log_matmul(sol.log_out[0, j, order - 1], outgoing[j][1]), outgoing[j + 1][1]))
            worst = max(worst, max(_norm_error(got, reference) for got, reference in pairs))
    return [row(f"chiral multilayer {radii}, kappa {kappa}: layer solutions and interface maps, 80 digits",
                "max |got - ref| / max |ref|", worst, 1e-11, "<")]


CHIRAL_SMALL_CASES = ((4.0, 1.0, 0.1), (-4.0 + 0.3j, 1.0, 0.05), (6.0, 1.2, 0.15), (2.25 + 0.1j, 0.9, -0.08), (-2.2 + 0.1j, 1.0, 0.1))


def small_chiral_sphere_polarisabilities(eps, mu, chi):
    """Quasi-static polarisabilities of Guzatov & Klimov, New J. Phys. 14, 123009 (2012), Eq. 48, for the
    Drude-Born-Fedorov medium; in Pasteur form eps_P = eps/s, mu_P = mu/s, kappa = n^2 chi/s, s = 1 - n^2 chi^2,
    and the dipole block is T = (2i/3) k^3 [[a_EE, -i a_EH], [-i a_EH, a_HH]]; (k a)^2 = 1e-6 corrections."""
    k = 2 * np.pi / 600.0
    a = 1e-3 / k
    den = (eps + 2) * (mu + 2) - 4 * chi ** 2 * eps * mu
    a_ee = a ** 3 * ((eps - 1) * (mu + 2) + 2 * chi ** 2 * eps * mu) / den
    a_hh = a ** 3 * ((mu - 1) * (eps + 2) + 2 * chi ** 2 * eps * mu) / den
    a_eh = a ** 3 * 3j * chi * eps * mu / den
    n2 = eps * mu
    s = 1 - n2 * chi ** 2
    n_p = np.sqrt(n2) / s
    n_p = -n_p if n_p.imag < 0 else n_p
    t = ps.solve_chiral([a], [n_p, 1.0], [n2 * chi / s, 0], 600.0, [mu / s, 1.0], l_max=3).t_matrix[0, 0]
    c = 2j * k ** 3 / 3
    name = f"small chiral sphere eps = {eps:g}, mu = {mu:g}, chi = {chi:g}: Guzatov & Klimov Eq. 48"
    return [row(name, "|T_EE / (c a_EE) - 1|", abs(t[0, 0] / (c * a_ee) - 1), 1e-4),
            row(name, "|T_HH / (c a_HH) - 1|", abs(t[1, 1] / (c * a_hh) - 1), 1e-4),
            row(name, "|T_EH / (-i c a_EH) - 1|", abs(t[0, 1] / (-1j * c * a_eh) - 1), 1e-4),
            row(name, "|T_HE / T_EH - 1| (reciprocity)", abs(t[1, 0] / t[0, 1] - 1), 1e-12)]


# ---------------------------------------------------------------- sheets

SHEET_LAM, EPS_FILM = 620.0, 15.0 + 5.0j  # a TMD-like monolayer near its exciton
THIN_FILM_CASES = ((50.0, AU), (200.0, 3.5))


def _t_error(sheet_solution, film_solution, orders=8):
    """Largest error of each T-matrix block element over the first orders, relative to its size."""
    t, t0 = sheet_solution.t_matrix[0, :orders], film_solution.t_matrix[0, :orders]
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.abs(t - t0).max(axis=0) / np.abs(t0).max(axis=0))


def thin_film_limit(radius, core):
    """Sheet.from_film against the explicit film: O(d^2) with the normal term, O(d) (TM) without it."""
    errors, in_plane = [], []
    for d in (1.0, 0.5, 0.25):
        film = ps.solve([radius, radius + d], [core, np.sqrt(EPS_FILM), 1.33], SHEET_LAM, l_max=30)
        sheet = ps.Sheet.from_film(EPS_FILM, d, SHEET_LAM, 1.33 ** 2)
        full = ps.solve([radius], [core, 1.33], SHEET_LAM, l_max=30, sheets={0: sheet})
        planar = ps.solve([radius], [core, 1.33], SHEET_LAM, l_max=30, sheets={0: sheet.conductivity})
        errors.append(np.diag(_t_error(full, film)))
        in_plane.append(np.diag(_t_error(planar, film)))
    errors, in_plane = np.array(errors), np.array(in_plane)  # (d, [TM, TE])
    name = f"a 15 + 5i film on a {radius:g} nm sphere (core {core}) as a 2D sheet, d = 1, 0.5, 0.25 nm"
    return [row(name, "max error ratio per halving, with the normal term (~1/4)", np.max(errors[1:] / errors[:-1]), .3, "<"),
            row(name, "min TM error ratio per halving, in-plane term only (~1/2)", np.min(in_plane[1:, 0] / in_plane[:-1, 0]), .45, ">"),
            close(name, "TE error without against with the normal term", in_plane[:, 1], errors[:, 1], 1e-5, 1e-8),
            row(name, "error at d = 0.25 nm", errors[-1].max(), 2e-3, "<")]


GRAPHENE_ORDERS = (1, 2, 3, 5)


def graphene_sphere_plasmons(order):
    """Resonances of a graphene-coated sphere vs eps_1 l + eps_2 (l + 1) + i sigma l (l + 1)/(k0 R) = 0."""
    radius, e1, e2 = 10.0, 2.1, 1.77

    def quasi_static(lam):
        s = ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)
        return (e1 * order + e2 * (order + 1) + 1j * s * order * (order + 1) * lam / (2 * np.pi * radius)).real

    lam_qs = brentq(quasi_static, 800, 200000)

    def minus_t(lam):
        sol = ps.solve([radius], [np.sqrt(e1), np.sqrt(e2)], lam, l_max=order, sheets={0: ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)})
        return -abs(np.exp(sol.log_t[0, 0, order - 1]))

    peak = minimize_scalar(minus_t, bracket=(0.98 * lam_qs, lam_qs, 1.02 * lam_qs), tol=1e-12).x
    retardation = (2 * np.pi * radius * np.sqrt(e2) / lam_qs) ** 2
    return [row(f"graphene-coated 10 nm sphere, plasmon l = {order}", "|peak / quasi-static resonance - 1| / (k R)^2",
                abs(peak / lam_qs - 1) / retardation, 1, "<")]


def all_rows():
    rows = []
    for case in MIE_CASES:
        rows += homogeneous_sphere_equals_mie(*case)
    rows += fifty_shells_are_one_sphere() + bohren_huffman_reference_values() + optical_theorem_and_backscattering() + rayleigh_limit()
    for size in LOSSLESS_SIZES:
        rows += lossless_multilayers_absorb_nothing(size)
    rows += amplitude_matrix_against_textbook_mie()
    for case in PRECISION_CASES:
        rows += coefficients_against_60_digits(*case)
    for case in CHIRAL_T_CASES:
        rows += chiral_t_matrix_against_80_digits(*case)
    for case in CHIRAL_LAYER_CASES:
        rows += chiral_layer_solutions_against_80_digits(*case)
    for case in CHIRAL_SMALL_CASES:
        rows += small_chiral_sphere_polarisabilities(*case)
    for case in THIN_FILM_CASES:
        rows += thin_film_limit(*case)
    for order in GRAPHENE_ORDERS:
        rows += graphene_sphere_plasmons(order)
    return rows


if __name__ == "__main__":
    report("Spheres under plane waves: textbook Mie, extended precision and limits",
           "Generated by `python benchmarks/sphere_checks.py`; the tests assert every row (lengths in nm). close rows are "
           "numpy's allclose: max |a - b| / (atol + rtol |b|) <= 1.", all_rows())
