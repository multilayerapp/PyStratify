"""Emitters in and around spheres: extended precision, closed forms and papers (benchmarks/MATRIX.md, spheres).

    python benchmarks/sphere_emitter_checks.py        # prints SPHERE_EMITTER_CHECKS.md (~5 min)

References: Majic & Le Ru's Mie decay rates (Appl. Opt. 59, 1293 (2020), Eqs. 34-37) and layered-sphere transfer
matrices summed in mpmath (``pystratify.references``, written separately); the free dipole's far field;
reciprocity with the plane-wave near field; the classical unnormalized series (which overflows); Guzatov &
Klimov's chiral molecule beside a small chiral sphere (New J. Phys. 14, 123009 (2012), Eq. 46); Goncalves et al.'s
Feibelman d-parameter rates (2020, Eqs. 4a, 13a, in mpmath); an mpmath
two-point Green dyadic; finite differences of Bohren & Huffman's modes at 40 digits for quadrupoles; the
non-retarded Casimir-Polder limit. The tests (``tests/test_physics.py``, ``test_farfield.py``,
``test_normalized.py``, ``test_shift.py``, ``test_feibelman.py``, ``test_rates.py``, ``test_pairs.py``, ``test_quadrupoles.py``,
``test_casimir.py``) assert every row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402
from checks import close, flag, report, row  # noqa: E402
from pystratify import TE, TM  # noqa: E402
from pystratify.references import _psi_all, _xi_all, classical_decay_rates, layered_decay_rates, layered_green_sums  # noqa: E402
from pystratify.references import sphere_decay_rates  # noqa: E402

LAM = 614.0
AU = 0.27 + 2.93j
AU_ELM = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm
AU690 = 0.0992 + 3.9183j  # the same at 690 nm


# ---------------------------------------------------------------- decay rates against mpmath

MIE_RATE_CASES = ((50.0, 1.0), (100.0, 5.0), (10.0, 5.0))


def decay_rates_against_mie(radius, gap):
    """Silver at 633 nm, emitter in air, against Majic & Le Ru's Eqs. 34-37 summed in mpmath."""
    ag = 0.093 + 4j
    total_perp, total_par, rad_perp, rad_par = sphere_decay_rates(radius, ag, 633.0, radius + gap, orders=1400)
    r = ps.decay_rates([radius], [ag, 1.0], 633.0, [radius + gap], tol=1e-10)
    name = f"silver sphere R = {radius:g} nm, emitter {gap:g} nm away, 633 nm: Mie decay rates in mpmath"
    return [flag(name, "converged", r.converged.all()),
            close(name, "total (perpendicular, parallel)", r.total[0], [total_perp, total_par], 1e-9),
            close(name, "radiative", r.radiative[0], [rad_perp, rad_par], 1e-12),
            close(name, "nonradiative", r.nonradiative[0], [total_perp - rad_perp, total_par - rad_par], 1e-9)]


LAYERED_RATE_CASES = (
    ([50.0, 55.0], [1.45, AU, 1.33], LAM, 49.0, 1250, 110),  # SiO2@Au nanoshell, 1 nm inside the core
    ([50.0, 55.0], [1.45, AU, 1.33], LAM, 56.0, 1400, 115),  # ... and 1 nm outside
    ([50.0, 52.0], [1.45, AU, 1.33], LAM, 49.5, 2400, 100),  # 2-nm shell, 0.5 nm inside
    ([10.0, 13.0, 36.0, 48.0], [1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33], 690.0, 14.0, 450, 240),  # matryoshka
)


def layered_rates_against_transfer_matrices(radii, n, wavelength, rd, orders, dps):
    total, radiative = layered_decay_rates(radii, n, wavelength, rd, orders, dps)
    r = ps.decay_rates(radii, n, wavelength, [rd], normalization="shell", tol=1e-10)
    name = f"electric dipole at r = {rd:g} nm in {radii} ({wavelength:g} nm): mpmath transfer matrices"
    return [flag(name, "converged", r.converged.all()), close(name, "total", r.total[0], total, 2e-10),
            close(name, "radiative", r.radiative[0], radiative, 1e-12),
            close(name, "nonradiative", r.nonradiative[0], np.subtract(total, radiative), 2e-10)]


MAGNETIC_RATE_CASES = (
    ([50.0], [0.093 + 4j, 1.0], [1, 1], 633.0, 51.0, 1250, 60),  # 1 nm from silver
    ([50.0, 55.0], [1.45, AU, 1.33], [1, 1, 1], LAM, 49.5, 2400, 150),  # inside a nanoshell's core
    ([50.0, 60.0], [AU, 1.45, 1.33], [1, 1, 1], LAM, 55.0, 400, 100),  # in the shell over a metal core
    ([40.0, 45.0, 60.0], [1.45, AU, 2.0, 1.33], [1, 1, 2.0, 1], LAM, 50.0, 400, 120),  # mu = 2 spacer
    ([100.0], [3.7 + 0.005j, 1.0], [1, 1], 800.0, 110.0, 350, 60),  # silicon, magnetic Mie mode
)


def magnetic_rates_against_transfer_matrices(radii, n, mu, wavelength, rd, orders, dps):
    total, radiative = layered_decay_rates(radii, n, wavelength, rd, orders, dps, mu=mu, dipole="magnetic")
    r = ps.decay_rates(radii, n, wavelength, [rd], mu, dipole="magnetic", normalization="shell", tol=1e-10)
    name = f"magnetic dipole at r = {rd:g} nm in {radii}, mu {mu} ({wavelength:g} nm): mpmath transfer matrices"
    return [flag(name, "converged", r.converged.all()), close(name, "total", r.total[0], total, 1e-11),
            close(name, "radiative", r.radiative[0], radiative, 1e-12),
            close(name, "nonradiative", r.nonradiative[0], np.subtract(total, radiative), 1e-10)]


NORMALIZED_CASE = ([50.0, 55.0], [1.45, AU, 1.33], [1, 1, 1], 614.0, 49.5)


def normalized_against_extended_precision(dipole):
    radii, n, mu, lam, r = NORMALIZED_CASE
    total, radiative = layered_decay_rates(radii, n, lam, r, 2400, 150, mu=mu, dipole=dipole)
    norm = ps.normalized_decay_rates(radii, n, lam, r, 2400, mu, dipole)
    tc, rc, last = classical_decay_rates(radii, n, lam, r, 2400, mu, dipole)
    name = f"{dipole} dipole 0.5 nm inside a nanoshell's core, 2400 orders: the normalized route"
    return [close(name, "total against mpmath", norm.total, total, 1e-10), close(name, "radiative against mpmath", norm.radiative, radiative, 1e-12),
            row(name, "last finite order of the classical (unnormalized) form", last, 100, "<"),
            row(name, "max |classical total / mpmath - 1| (it overflows)", np.max(np.abs(tc / total - 1)), .01, ">")]


SHIFT_CASES = {  # radii, n, mu, wavelength, r, digits of the reference
    "nanoshell, host, d = 1 nm": ([50.0, 55.0], [1.45, AU_ELM, 1.33], None, 614.0, 56.0, 140),
    "nanoshell, core, d = 0.5 nm": ([50.0, 55.0], [1.45, AU_ELM, 1.33], None, 614.0, 49.5, 260),
    "lossless sphere, d = 2 nm": ([250.0], [3.87, 1.0], None, 633.0, 252.0, 60),
    "matryoshka": ([10.0, 13.0, 36.0, 48.0], [1.45, AU690, 1.45, AU690, 1.33], None, 690.0, 35.0, 280),
    "between gold and a magnetic shell": ([40.0, 45.0, 60.0], [1.45, AU_ELM, 2.0, 1.33], [1, 1, 2.0, 1], 614.0, 50.0, 200),
}


def shift_against_extended_precision(case, dipole):
    """Rate and frequency shift with the automatic truncation against the complex sums of mpmath transfer matrices."""
    radii, n, mu, lam, r, dps = SHIFT_CASES[case]
    res = ps.normalized_decay_rates(radii, n, lam, r, None, mu, dipole)
    ref = np.array(layered_green_sums(radii, n, lam, r, res.orders, dps, mu=mu, dipole=dipole))
    name = f"{case}, {dipole} dipole: rate and shift against {dps}-digit transfer matrices"
    return [flag(name, "converged by the normalized route", res.converged and res.route == "normalized"),
            row(name, "max |total - (1 + Re g)| / total", np.max(np.abs(res.total - (1 + ref.real)) / res.total), 1e-13),
            row(name, "max |shift - Im g / 2| / |1 + g|", np.max(np.abs(res.shift - ref.imag / 2) / np.abs(1 + ref)), 1e-13)]


# ---------------------------------------------------------------- far fields of a dipole

def _basis(theta, phi):
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    n = np.stack([st * cp, st * sp, ct], -1)
    return n, np.stack([ct * cp, ct * sp, -st], -1), np.stack([-sp, cp, 0 * sp], -1)


FREE_POSITIONS = ([10.0, -5.0, 20.0], [0.0, 40.0, -10.0], [70.0, 20.0, 30.0], [900.0, -400.0, 200.0])
DIPOLES = ("electric", "magnetic")


def dipole_in_homogeneous_medium(position, dipole):
    rng = np.random.default_rng(1)
    theta, phi = rng.uniform(0, np.pi, 9), rng.uniform(-np.pi, np.pi, 9)
    moment = rng.normal(size=3) + 1j * rng.normal(size=3)
    f = ps.dipole_far_field([30.0, 60.0], [1.33] * 3, LAM, position, moment, theta, phi, dipole=dipole)
    n, e_theta, e_phi = _basis(theta, phi)
    phase = np.exp(-1j * 2 * np.pi * 1.33 / LAM * (n @ np.array(position)))[:, None]
    exact = (moment - (n @ moment)[:, None] * n) * phase if dipole == "electric" else -np.cross(n, moment) * phase
    scale = np.abs(moment).max()
    name = f"{dipole} dipole at {position} in index-matched shells: the free far field"
    return [close(name, "E_theta", f.e_theta, np.sum(exact * e_theta, -1), 1e-5, 1e-13 * scale),
            close(name, "E_phi", f.e_phi, np.sum(exact * e_phi, -1), 1e-5, 1e-13 * scale),
            row(name, "|power - 1|", abs(f.power - 1), 1e-13), flag(name, "converged", f.converged),
            row(name, "orders used (the direct field is exact at any distance)", f.orders_used, 30, "<")]


RECIPROCITY_POSITIONS = {"core": [5.0, 10.0, -20.0], "lossy shell": [20.0, -35.0, 30.0], "magnetic shell": [30.0, 40.0, 30.0],
                         "host": [60.0, 50.0, -40.0]}


def reciprocity_with_near_field(where, dipole):
    """F(n) . e = p . E(r0) for the plane wave of polarisation e incident from n (E -> -(mu_h/n_h) H for a
    magnetic dipole), with the incidence rotated onto +z."""
    position = RECIPROCITY_POSITIONS[where]
    radii, n, mu = [40.0, 55.0, 70.0], [1.45, AU, 2.0, 1.33], [1.0, 1.0, 1.3, 1.0]
    sol = ps.solve(radii, n, LAM, mu, l_max=45)
    rng = np.random.default_rng(2)
    theta, phi = rng.uniform(0, np.pi, 5), rng.uniform(-np.pi, np.pi, 5)
    moment = rng.normal(size=3) + 1j * rng.normal(size=3)
    f = ps.dipole_far_field(radii, n, LAM, position, moment, theta, phi, mu, dipole=dipole, l_max=45)
    directions, e_theta, e_phi = _basis(theta, phi)
    worst = 0.0
    for i in range(theta.size):
        for pol, got in ((e_theta[i], f.e_theta[i]), (e_phi[i], f.e_phi[i])):
            rotation = np.array([pol, np.cross(-directions[i], pol), -directions[i]])
            near = ps.near_field(sol, *(rotation @ np.array(position)))
            if dipole == "electric":
                expected = (rotation @ moment) @ np.array([near.e[c] for c in "xyz"])
            else:
                expected = -(mu[-1] / n[-1]) * (rotation @ moment) @ np.array([near.h[c] for c in "xyz"])
            worst = max(worst, abs(got / expected - 1))
    return [row(f"{dipole} dipole in the {where} of a gold/magnetic multilayer: reciprocity with the plane-wave near field",
                "max |F . e / (p . E(r0)) - 1|", worst, 1e-12)]


def radiated_power_equals_decay_rates(dipole):
    radii, n, mu = [50.0, 70.0, 90.0], [1.45, AU, 1.6, 1.33], [1.3, 1.0, 1.7, 1.1]
    worst = 0.0
    for r0 in (40.0, 80.0, 95.0, 300.0):
        rates = ps.decay_rates(radii, n, LAM, [r0], mu, dipole=dipole, tol=1e-10)
        for column, moment in enumerate(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])):
            f = ps.dipole_far_field(radii, n, LAM, [0.0, 0.0, r0], moment, 0.0, 0.0, mu, dipole=dipole)
            worst = max(worst, abs(f.power / rates.radiative[0, column] - 1))
    return [row(f"{dipole} dipole at 40, 80, 95, 300 nm in a magnetic multilayer", "max |far-field power / radiative rate - 1|", worst, 1e-12)]


# ---------------------------------------------------------------- a chiral molecule, papers

GUZATOV_CASES = ((6.0, 1.0, 0.1), (4.0 + 0.1j, 1.0, 0.2), (-4.0 + 0.3j, 1.0, 0.1), (-4.0 + 0.1j, -1.11 + 0.05j, 0.2), (-2.5 + 0.1j, -1.6, 0.2))


def chiral_molecule_guzatov_klimov(eps, mu, chi):
    """Orientation-averaged radiative rate of a molecule (d0, m = -i xi d0), xi = +-0.1, at r0 = 2a from a small
    Drude-Born-Fedorov sphere (their Eq. 48 polarisabilities); agreement to O((k0 a)^2)."""
    wavelength = 600.0
    k = 2 * np.pi / wavelength
    n2 = eps * mu
    s = 1 - n2 * chi ** 2
    n_p = np.sqrt(n2 + 0j) / s
    n_p = -n_p if n_p.imag < 0 else n_p
    den = (eps + 2) * (mu + 2) - 4 * chi ** 2 * eps * mu
    rows = []
    for ka, tol in ((0.01, 3e-4), (0.005, 8e-5)):
        a = ka / k
        r0 = 2 * a
        a_ee = a ** 3 * ((eps - 1) * (mu + 2) + 2 * chi ** 2 * eps * mu) / den
        a_hh = a ** 3 * ((mu - 1) * (eps + 2) + 2 * chi ** 2 * eps * mu) / den
        a_eh = a ** 3 * 3j * chi * eps * mu / den
        for xi in (0.1, -0.1):
            eq46 = 1 + xi ** 2 + 2 / r0 ** 6 * (abs(a_ee - 1j * xi * a_eh) ** 2 + abs(-a_eh - 1j * xi * a_hh) ** 2)
            out = ps.emission_rates([a], [n_p, 1.0], wavelength, [0, 0, r0], [0, 0, 1.0], magnetic_moment=[0, 0, -1j * xi],
                                    mu=[mu / s, 1.0], kappa=[n2 * chi / s, 0], orientation="isotropic", tol=1e-10)
            rows.append(row(f"chiral molecule (xi = {xi:+g}) at 2a from a chiral sphere eps = {eps:g}, mu = {mu:g}, chi = {chi:g}, k a = {ka:g}",
                            "|radiative / Eq. 46 - 1|", abs(out.radiative / (eq46 / (1 + xi ** 2)) - 1), tol))
    return rows


def _mp_two_point(radii, n, lam, a, b, orders, dps=80):
    """u_in(x<) u_out(x>) / W per order, [TM, TE], from mpmath transfer matrices, minus psi(x<) xi(x>)
    for points in one shell; a < b for points in different shells."""
    out = np.zeros((2, orders), complex)
    with mp.workdps(dps):
        R = [mp.mpf(v) for v in radii]
        nn = [mp.mpc(complex(v)) for v in n]
        k = [2 * mp.pi * v / mp.mpf(lam) for v in nn]
        top = orders + 1

        def funcs(z):
            p, x = _psi_all(z, top), _xi_all(z, top)
            dp = [None] + [p[m - 1] - m * p[m] / z for m in range(1, top + 1)]
            dx = [None] + [x[m - 1] - m * x[m] / z for m in range(1, top + 1)]
            return p, dp, x, dx

        N = len(radii)
        iface = [(funcs(k[j] * R[j]), funcs(k[j + 1] * R[j])) for j in range(N)]
        d, d2 = sum(1 for v in radii if a >= v), sum(1 for v in radii if b >= v)
        lo, hi = funcs(k[d] * mp.mpf(min(a, b))), funcs(k[d2] * mp.mpf(max(a, b)))
        for l in range(1, orders + 1):
            for pol, te in ((0, False), (1, True)):
                mats = []
                for j in range(N):
                    (p, dp, x, dx), (pt, dpt, xt, dxt) = iface[j]
                    eta = nn[j] / nn[j + 1]
                    c1, c2 = (eta, 1) if te else (1, eta)
                    m = [[dx[l] * pt[l] * c1 - x[l] * dpt[l] * c2, dx[l] * xt[l] * c1 - x[l] * dxt[l] * c2],
                         [-dp[l] * pt[l] * c1 + p[l] * dpt[l] * c2, -dp[l] * xt[l] * c1 + p[l] * dxt[l] * c2]]
                    mats.append([[v / mp.mpc(0, 1) for v in r_] for r_ in m])  # / W(psi, xi) = i
                A, B = mp.mpf(1), mp.mpf(0)
                for j in range(d):
                    m = mats[j]
                    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
                    A, B = (m[1][1] * A - m[0][1] * B) / det, (-m[1][0] * A + m[0][0] * B) / det
                co = {N: (mp.mpf(0), mp.mpf(1))}
                for j in range(N - 1, d - 1, -1):
                    m, (Ao, Bo) = mats[j], co[j + 1]
                    co[j] = (m[0][0] * Ao + m[0][1] * Bo, m[1][0] * Ao + m[1][1] * Bo)
                W = A * co[d][1] - B * co[d][0]
                u = A * lo[0][l] + B * lo[2][l]
                v = co[d2][0] * hi[0][l] + co[d2][1] * hi[2][l]
                out[pol, l - 1] = complex(u * v / W - (lo[0][l] * hi[2][l] if d == d2 else 0))
    return out


TWO_POINT_CASES = ((56.0, 60.0), (60.0, 56.0), (40.0, 49.0))


def two_point_radial_form(a, b):
    """psi_1 xi_2 [sigma_2 + rho_1 sigma_2 + rho_1 + R S]/(1 - R S) (value-value) per order against mpmath."""
    from pystratify.normalized import _Sweep

    radii, n, lam = [50.0, 55.0], [1.45, AU_ELM, 1.33], 614.0
    L = 120
    k = 2 * np.pi * np.asarray(n, complex) / lam
    t = _Sweep(np.asarray(radii), np.asarray(n, complex), np.ones(3, complex), k, L).at(np.array([a, b]))
    l = np.arange(1, L + 1)
    i1, i2 = (0, 1) if a <= b else (1, 0)  # inner, outer point
    x1, x2 = t["x"][i1], t["x"][i2]
    ref = _mp_two_point(radii, n, lam, a, b, L)
    rows = []
    for p, label in ((TM, "TM"), (TE, "TE")):
        rho1, sig2, rs = t["rho"][p, i1], t["sigma"][p, i2], t["rho"][p, i1] * t["sigma"][p, i1]
        pref = t["P"][p, i2] * np.exp((l + 1) * np.log(x1 / x2) + t["lj"][p, i1] - t["lj"][p, i2])
        form = pref * (sig2 + rho1 * sig2 + rho1 + rs) / (1 - rs)
        w = 2 * l + 1
        rows.append(row(f"two-point Green dyadic, points at {a:g} and {b:g} nm of a gold nanoshell, {label}",
                        "sum (2l+1) |form - mpmath| / sum (2l+1) |mpmath|", np.sum(w * np.abs(form - ref[p])) / np.sum(w * np.abs(ref[p])), 1e-13))
    return rows


def _random_quadrupole(seed=2, scale=20.0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    q = q + q.T
    return scale * (q - np.trace(q) / 3 * np.eye(3))


def _bh_mode(l, m, parity, kind, k, point):
    """Cartesian M and N of Bohren & Huffman at a point, 40 digits."""
    x, y, z = point
    r = mp.sqrt(x * x + y * y + z * z)
    th, ph = mp.acos(z / r), mp.atan2(y, x)
    rho = k * r

    def zf(order, t):
        j = mp.besselj(order + 0.5, t)
        return mp.sqrt(mp.pi / (2 * t)) * (j if kind == "psi" else j + 1j * mp.bessely(order + 0.5, t))

    zl = zf(l, rho)
    dz = zf(l - 1, rho) - l * zl / rho  # (1/rho) d(rho z_l)/drho

    def legendre(order, t):
        return mp.legenp(order, m, t, type=2) * (-1) ** m if order >= m else 0  # no Condon-Shortley phase

    mu_ = mp.cos(th)
    p_lm = legendre(l, mu_)
    dp = (l * mu_ * p_lm - (l + m) * legendre(l - 1, mu_)) / mp.sin(th)  # d P_l^m / d theta
    cm, sm, st = mp.cos(m * ph), mp.sin(m * ph), mp.sin(th)
    a, b = (-sm, cm) if parity == "e" else (cm, sm)  # angular factors of the P/sin and dP/dtheta terms
    m_th, m_ph = m * a * p_lm / st * zl, -b * dp * zl
    n_r = zl / rho * b * l * (l + 1) * p_lm
    n_th, n_ph = b * dp * dz, m * a * p_lm / st * dz
    r_hat = [st * mp.cos(ph), st * mp.sin(ph), mp.cos(th)]
    th_hat = [mp.cos(th) * mp.cos(ph), mp.cos(th) * mp.sin(ph), -st]
    ph_hat = [-mp.sin(ph), mp.cos(ph), 0]
    return ([m_th * th_hat[i] + m_ph * ph_hat[i] for i in range(3)],
            [n_r * r_hat[i] + n_th * th_hat[i] + n_ph * ph_hat[i] for i in range(3)])


QUADRUPOLE_ORDERS = (2, 5)


def quadrupole_coupling(l):
    """(1/6) sum Q_ij d_j W_i of W_s = M + s N at (0, 0, r0), all axial families, both radial kinds."""
    from pystratify.rates import _functionals
    from pystratify.riccati import log_riccati

    mp.mp.dps = 40
    q = _random_quadrupole(scale=1.0)
    k, r0, h = 0.013, 47.0, mp.mpf("1e-9")
    x = k * r0
    log_psi, log_xi = log_riccati(np.array([x + 0j]), l + 1)
    families = [(0, "e"), (1, "e"), (1, "o"), (2, "e"), (2, "o")]
    worst = 0.0
    for kind, logs in (("psi", log_psi[0]), ("xi", log_xi[0])):
        dlog = np.exp(logs[l - 1] - logs[l]) - l / x
        values = _functionals(np.full((1, 1, 2), dlog), np.full((1, 2), x), np.full(2, k), np.zeros((1, 1, 2, 3)), q[None, None],
                              np.array([l * (l + 1.0)]))[0, 0, 0] * np.exp(logs[l]) / x  # undo the division by f/x
        for f, (m, parity) in enumerate(families):
            for c, sign in enumerate((1, -1)):
                base = [mp.mpf("1e-14"), mp.mpf("2e-14"), mp.mpf(r0)]
                total = 0
                for j in range(3):
                    plus, minus = list(base), list(base)
                    plus[j] += h
                    minus[j] -= h
                    (mp_, np_), (mm_, nm_) = (_bh_mode(l, m, parity, kind, mp.mpf(k), pt) for pt in (plus, minus))
                    for i in range(3):
                        total += complex(q[i, j]) * ((mp_[i] + sign * np_[i]) - (mm_[i] + sign * nm_[i])) / (2 * h)
                reference = complex(total / 6)
                worst = max(worst, abs(values[f, c] - reference) / max(abs(reference), 1e-300))
    return [row(f"quadrupole coupling, order l = {l}, all axial families, psi and xi", "max |functional - finite difference| / |reference| (40 digits)",
                worst, 1e-9, "<")]


WP, GAMMA = 1.37e16, 1e14  # a Drude metal (gold-like), rad/s
A0, W0 = 0.05, 2.4e15  # alpha/(4 pi eps_0) = 0.05 nm^3 (an alkali atom), resonance 785 nm


def _eps(xi):
    return np.array([1 + WP ** 2 / (xi * (xi + GAMMA)), 1.0])


def _alpha(xi):
    return A0 / (1 + (xi / W0) ** 2)


def casimir_non_retarded_limit():
    """Within a nanometre of a Drude sphere U -> -C3/d^3, C3 = (hbar/4 pi) int alpha (eps - 1)/(eps + 1) dxi."""
    from pystratify.casimir import HBAR

    f = lambda u: (lambda x: x * _alpha(x) * (_eps(x)[0] - 1) / (_eps(x)[0] + 1))(np.exp(u))  # noqa: E731
    c3 = HBAR / (4 * np.pi) * quad(f, np.log(1e6), np.log(1e21), limit=500, epsabs=0, epsrel=1e-12)[0]
    R = 200.0
    out = ps.casimir_polder([R], _eps, R + np.array([0.5, 1.0]), _alpha, nodes=64)
    ratio = -out.energy * (out.r - R) ** 3 / c3
    name = "an atom 0.5 and 1 nm from a 200 nm Drude sphere: Casimir-Polder"
    return [flag(name, "converged", out.converged.all()), row(name, "max |U d^3 / (-C3) - 1| (non-retarded limit)", np.max(np.abs(ratio - 1)), .015, "<"),
            flag(name, "attractive (U < 0)", np.all(out.energy < 0))]


# ---------------------------------------------------------------- Feibelman d-parameters (Goncalves et al.)

AU_GONCALVES = AU_ELM
GONCALVES_CASES = ((0.3 + 0.1j, 0.0), (0.3 + 0.1j, -0.1 + 0.05j), (-0.2 + 0.05j, 0.15 + 0.02j))


def _goncalves_perp(R, n_m, n_d, lam, r, d_perp, d_par, orders=120):
    """Total rate of a radial dipole in host units from Eqs. (4a) and (13a) of Goncalves et al. (2020)."""
    with mp.workdps(40):
        k0 = 2 * mp.pi / lam
        xm, xd, y = k0 * mp.mpc(n_m) * R, k0 * n_d * R, k0 * n_d * r
        em, ed = mp.mpc(n_m) ** 2, mp.mpf(n_d) ** 2

        def j(l, z):
            return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(l + 0.5, z)

        def h(l, z):
            return j(l, z) + 1j * mp.sqrt(mp.pi / (2 * z)) * mp.bessely(l + 0.5, z)

        def d_psi(l, z):  # (z j_l)'
            return z * j(l - 1, z) - l * j(l, z)

        def d_xi(l, z):
            return z * h(l - 1, z) - l * h(l, z)

        total = mp.mpf(0)
        for l in range(1, orders + 1):
            ll = l * (l + 1)
            num = em * j(l, xm) * d_psi(l, xd) - ed * j(l, xd) * d_psi(l, xm)
            num += (em - ed) * (j(l, xd) * j(l, xm) * ll * d_perp + d_psi(l, xd) * d_psi(l, xm) * d_par) / R
            den = em * j(l, xm) * d_xi(l, xd) - ed * h(l, xd) * d_psi(l, xm)
            den += (em - ed) * (h(l, xd) * j(l, xm) * ll * d_perp + d_xi(l, xd) * d_psi(l, xm) * d_par) / R
            total += (2 * l + 1) * ll * mp.re(-(num / den) * h(l, y) ** 2)
        return float(1 + 1.5 * total / y ** 2)


def sphere_against_goncalves(d_perp, d_par):
    rows = []
    for r in (6.0, 7.0):
        ref = _goncalves_perp(5.0, AU_GONCALVES, 1.0, 614.0, r, d_perp, d_par)
        rates = ps.decay_rates([5.0], [AU_GONCALVES, 1.0], 614.0, r, sheets={0: ps.Feibelman(d_perp, d_par)}, tol=1e-14)
        name = f"5 nm gold sphere with d_perp = {d_perp}, d_par = {d_par}, radial dipole at r = {r:g} nm: Goncalves et al. 2020"
        rows += [flag(name, "normalized route, converged", rates.route == "normalized" and rates.converged.all()),
                 row(name, "|total / Eqs. (4a), (13a) - 1|", abs(rates.total[0, 0] / ref - 1), 1e-13)]
    return rows


def all_rows():
    rows = []
    for case in MIE_RATE_CASES:
        rows += decay_rates_against_mie(*case)
    for case in LAYERED_RATE_CASES:
        rows += layered_rates_against_transfer_matrices(*case)
    for case in MAGNETIC_RATE_CASES:
        rows += magnetic_rates_against_transfer_matrices(*case)
    for dipole in DIPOLES:
        rows += normalized_against_extended_precision(dipole)
    for case in SHIFT_CASES:
        for dipole in DIPOLES:
            rows += shift_against_extended_precision(case, dipole)
    for position in FREE_POSITIONS:
        for dipole in DIPOLES:
            rows += dipole_in_homogeneous_medium(position, dipole)
    for where in RECIPROCITY_POSITIONS:
        for dipole in DIPOLES:
            rows += reciprocity_with_near_field(where, dipole)
    for dipole in DIPOLES:
        rows += radiated_power_equals_decay_rates(dipole)
    for case in GUZATOV_CASES:
        rows += chiral_molecule_guzatov_klimov(*case)
    for case in TWO_POINT_CASES:
        rows += two_point_radial_form(*case)
    for case in GONCALVES_CASES:
        rows += sphere_against_goncalves(*case)
    for l in QUADRUPOLE_ORDERS:
        rows += quadrupole_coupling(l)
    return rows + casimir_non_retarded_limit()


if __name__ == "__main__":
    report("Emitters in and around spheres: extended precision, closed forms and papers",
           "Generated by `python benchmarks/sphere_emitter_checks.py`; the tests assert every row (lengths in nm). close rows "
           "are numpy's allclose: max |a - b| / (atol + rtol |b|) <= 1.", all_rows())
