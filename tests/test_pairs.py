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
    terms = _pair_terms(radii, n, np.ones_like(n), k, L, a, b, gamma, series_free=True)
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
    with pytest.raises(NotImplementedError):
        ps.green_dyadic(*SHELL, [0, 0, 40.0], [0, 0, 60.0])  # core and host
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
    """u_in(x<) u_out(x>) / W - psi(x<) xi(x>) per order, [TM, TE], from mpmath transfer matrices."""
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
        d = sum(1 for v in radii if a >= v)
        lo, hi = funcs(k[d] * mp.mpf(min(a, b))), funcs(k[d] * mp.mpf(max(a, b)))
        for l in range(1, orders + 1):
            for pol, te in ((0, False), (1, True)):
                mats = []
                for j in range(N):
                    (p, dp, x, dx), (pt, dpt, xt, dxt) = iface[j]
                    eta = nn[j] / nn[j + 1]
                    c1, c2 = (eta, 1) if te else (1, eta)
                    mats.append([[dx[l] * pt[l] * c1 - x[l] * dpt[l] * c2, dx[l] * xt[l] * c1 - x[l] * dxt[l] * c2],
                                 [-dp[l] * pt[l] * c1 + p[l] * dpt[l] * c2, -dp[l] * xt[l] * c1 + p[l] * dxt[l] * c2]])  # fmt: skip
                A, B = mp.mpf(1), mp.mpf(0)
                for j in range(d):
                    m = mats[j]
                    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
                    A, B = (m[1][1] * A - m[0][1] * B) / det, (-m[1][0] * A + m[0][0] * B) / det
                Ao, Bo = mp.mpf(0), mp.mpf(1)
                for j in range(N - 1, d - 1, -1):
                    m = mats[j]
                    Ao, Bo = m[0][0] * Ao + m[0][1] * Bo, m[1][0] * Ao + m[1][1] * Bo
                W = A * Bo - B * Ao
                u = A * lo[0][l] + B * lo[2][l]
                v = Ao * hi[0][l] + Bo * hi[2][l]
                out[pol, l - 1] = complex(u * v / W - lo[0][l] * hi[2][l])
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
