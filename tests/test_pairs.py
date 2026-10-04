"""Two-point Green's dyadic: free-field series, reciprocity, the coinciding limit and extended precision."""

import mpmath as mp
import numpy as np
import pytest

import pystratify as ps
from pystratify import TE, TM
from pystratify.emission import _local_frame
from pystratify.normalized import _Sweep
from pystratify.pairs import _free_dyadic, _pair_terms
from pystratify.references import _psi_all, _xi_all

AU = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm
SHELL = ([50.0, 55.0], [1.45, AU, 1.33], 614.0)


def _free_series(radii, n, lam, r1, r2, L):
    radii, n = np.asarray(radii), np.asarray(n, complex)
    k = 2 * np.pi * n / lam
    a, b = np.linalg.norm(r1), np.linalg.norm(r2)
    frame = _local_frame(r1)
    v = frame @ r2
    phi = np.arctan2(v[1], v[0])
    frame = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1.0]]) @ frame
    gamma = float(np.arccos(np.clip((frame @ r2)[2] / b, -1, 1)))
    terms, _ = _pair_terms(radii, n, np.ones_like(n), k, L, a, b, gamma, series_free=True)
    ct, st = np.cos(gamma), np.sin(gamma)
    basis = np.array([[st, ct, 0.0], [0.0, 0.0, 1.0], [ct, -st, 0.0]])
    return frame.T @ ((1j * k[-1].real / (6 * np.pi)) * basis @ terms.sum(axis=0)) @ frame


@pytest.mark.parametrize(
    "r1, r2", [([0, 10, 85.0], [30, -20, 64.0]), ([50, 40, 30.0], [10, 5, 63.0]), ([20, -40, 50.0], [-35, 50, 55.0])]
)
def test_free_field_series_is_the_closed_form(r1, r2):
    """The multipole series of the free part, with the angular functions and constants of the scattered
    part, reproduces the closed-form free dyadic: every family, polarization and frame convention."""
    r1, r2 = np.array(r1), np.array(r2)
    g = _free_series([30.0, 60.0], [1.4, 1.4, 1.4], 600.0, r1, r2, 1500)
    g0 = _free_dyadic(2 * np.pi * 1.4 / 600.0, r1, r2)
    assert np.max(np.abs(g - g0)) <= 1e-13 * np.max(np.abs(g0))


def test_index_matched_and_inputs():
    out = ps.green_dyadic([30.0, 60.0], [1.4, 1.4, 1.4], 600.0, [0, 0, 70.0], [10, 20, 80.0])
    assert np.all(out.scattered == 0) and np.allclose(out.total, out.free)
    # points in different shells: the whole field, carried through the (fictitious) interfaces
    for r1, r2 in (([0, 0, 20.0], [10, 20, 80.0]), ([10, 20, 80.0], [3, -4, 40.0])):
        out = ps.green_dyadic([30.0, 60.0], [1.4, 1.4, 1.4], 600.0, r1, r2)
        g0 = _free_dyadic(2 * np.pi * 1.4 / 600.0, np.array(r1), np.array(r2))
        assert np.all(out.free == 0) and np.max(np.abs(out.total - g0)) <= 1e-13 * np.max(np.abs(g0))
    with pytest.raises(ValueError):
        ps.green_dyadic(*SHELL, [0, 0, 52.0], [0, 0, 60.0])  # in the gold shell
    with pytest.raises(ValueError):
        ps.green_dyadic(*SHELL, [0, 0, 60.0], [0, 0, 60.0])


@pytest.mark.parametrize("r1, r2", [([3.0, 1.0, 57.0], [-10.0, 6.0, 58.0]), ([5.0, 0, 40.0], [0, -8.0, 45.0])])
def test_reciprocity(r1, r2):
    g12, g21 = ps.green_dyadic(*SHELL, r1, r2), ps.green_dyadic(*SHELL, r2, r1)
    assert g12.converged and np.max(np.abs(g12.scattered - g21.scattered.T)) <= 1e-13 * np.max(np.abs(g12.scattered))


def test_coinciding_limit_gives_the_rates_and_the_shift():
    r = 57.0
    rates = ps.normalized_decay_rates(*SHELL, r, None, None, "electric")
    g = ps.green_dyadic(*SHELL, [0, 0, r], [0, 1e-5, r])
    s = g.scattered * 6 * np.pi / (1j * g.k)  # = g of the decay rates on the diagonal (local frame = global here)
    assert np.isclose(1 + s[2, 2].real, rates.total[0], rtol=1e-9) and np.isclose(
        1 + s[0, 0].real, rates.total[1], rtol=1e-9
    )
    assert np.isclose(s[2, 2].imag / 2, rates.shift[0], rtol=1e-9)


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
                         [-dp[l] * pt[l] * c1 + p[l] * dpt[l] * c2, -dp[l] * xt[l] * c1 + p[l] * dxt[l] * c2]]  # fmt: skip
                    mats.append([[v / mp.mpc(0, 1) for v in row] for row in m])  # / W(psi, xi) = i
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


@pytest.mark.parametrize("a, b", [(56.0, 60.0), (60.0, 56.0), (40.0, 49.0)])
def test_two_point_radial_form_against_extended_precision(a, b):
    """psi_1 xi_2 [sigma_2 + rho_1 sigma_2 + rho_1 + R S]/(1 - R S) (value-value) per order against mpmath."""
    radii, n, lam = SHELL
    L = 120
    k = 2 * np.pi * np.asarray(n, complex) / lam
    t = _Sweep(np.asarray(radii), np.asarray(n, complex), np.ones(3, complex), k, L).at(np.array([a, b]))
    l = np.arange(1, L + 1)
    i1, i2 = (0, 1) if a <= b else (1, 0)  # inner, outer point
    x1, x2 = t["x"][i1], t["x"][i2]
    ref = _mp_two_point(radii, n, lam, a, b, L)
    for p in (TM, TE):
        rho1, sig2, rs = t["rho"][p, i1], t["sigma"][p, i2], t["rho"][p, i1] * t["sigma"][p, i1]
        pref = t["P"][p, i2] * np.exp((l + 1) * np.log(x1 / x2) + t["lj"][p, i1] - t["lj"][p, i2])
        form = pref * (sig2 + rho1 * sig2 + rho1 + rs) / (1 - rs)
        w = 2 * l + 1
        assert np.sum(w * np.abs(form - ref[p])) <= 1e-13 * np.sum(w * np.abs(ref[p])), p


@pytest.mark.parametrize(
    "radii, n, lam, a, b",
    [
        (*SHELL, 40.0, 60.0),  # core -> host through a gold shell
        ([30.0, 40.0, 50.0], [1.45, AU, 2.0, 1.33], 614.0, 20.0, 45.0),
        ([30.0, 40.0, 50.0], [3.5, 1.2, 2.5, 1.0], 400.0, 25.0, 70.0),  # three dielectric interfaces
    ],
)
def test_cross_shell_radial_form_against_extended_precision(radii, n, lam, a, b):
    """psi_1 C xi_2 (1 + rho_1)(1 + sigma_2)/(1 - R S) with the transmission chain, per order."""
    L = 80
    n = np.asarray(n, complex)
    sweep = _Sweep(np.asarray(radii), n, np.ones(n.size, complex), 2 * np.pi * n / lam, L)
    t = sweep.at(np.array([a, b]))
    d1, d2 = t["shell"]
    l = np.arange(1, L + 1)
    ref = _mp_two_point(radii, n, lam, a, b, L)
    for p in (TM, TE):
        log_c, chain = sweep.transmission(p, d1, d2)
        rho1, sig1, sig2 = t["rho"][p, 0], t["sigma"][p, 0], t["sigma"][p, 1]
        pref = t["P"][p, 1] * np.exp((l + 1) * np.log(a / b) + t["lj"][p, 0] - t["lj"][p, 1] + log_c) * chain
        form = pref * (1 + rho1) * (1 + sig2) / (1 - rho1 * sig1)
        assert np.max(np.abs(form - ref[p]) / np.abs(ref[p])) <= 5e-14, p


@pytest.mark.parametrize("mu", [None, [1, 1, 1.5, 1.0]])
@pytest.mark.parametrize("source", [[5.0, -3.0, 20.0], [10.0, 20.0, 64.0], [-20.0, 30.0, 26.0]])
def test_interface_conditions(source, mu):
    """Tangential E and normal D continuous across a lossless interface, approached from both sides:
    source in the core (outward chain on both sides), in the host (same shell outside, reciprocity
    inside) and in the shell itself (same shell inside, outward chain outside)."""
    radii, n = [30.0, 40.0, 50.0], np.array([1.45, AU, 2.0, 1.33])
    eps = n**2 / (np.ones(4) if mu is None else np.asarray(mu))
    u = np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])

    def limit(side):  # cubic extrapolation to the interface from three points 1e-4 apart
        g = [ps.green_dyadic(radii, n, 614.0, source, (50 + side * j * 1e-4) * u, mu=mu) for j in (1, 2, 3)]
        assert all(v.converged for v in g)
        return 3 * g[0].total - 3 * g[1].total + g[2].total

    inside, outside = limit(-1), limit(+1)
    tangential = (np.eye(3) - np.outer(u, u)) @ (inside - outside)
    assert np.max(np.abs(tangential)) <= 1e-13 * np.max(np.abs(inside))
    assert np.max(np.abs(u @ (eps[2] * inside - eps[3] * outside))) <= 1e-13 * np.max(np.abs(eps[2] * u @ inside))
