"""Independent references: extended-precision and classical solutions for validation.

Used by the tests, the benchmarks and the figures of the accompanying paper.
Nothing here shares code with the solver (``solver.py``, ``decay.py``) or the
normalized formulation (``normalized.py``):

* :func:`sphere_decay_rates` - Mie decay rates of a dipole outside a
  homogeneous sphere (Majic & Le Ru, Appl. Opt. 59, 1293 (2020), Eqs. 34-37)
  in mpmath, with Riccati-Bessel functions from exact top values and the
  stable recurrences (downward for psi, upward for xi), so any order is exact
  to the working precision;
* :func:`layered_decay_rates` - the same for an emitter inside or outside a
  layered sphere, from high-precision 2x2 transfer matrices;
* :func:`layered_green_forms` and :func:`layered_green_sums` - the complex
  scattered Green's forms per order and their sums for a dipole, whose
  imaginary part gives the frequency shift;
* :func:`bhmie` - Bohren & Huffman's BHMIE in NumPy (logarithmic derivative
  D_n(mx) by downward recurrence), the classic algorithm for large spheres.
  It is accurate to ~1e-13 for absorbing spheres; for weakly absorbing ones
  with x >~ 100 it loses digits (1e-4 at x = 1000, m = 1.33 + 1e-8 i, against
  mpmath), so use :func:`sphere_extinction` there;
* :func:`classical_decay_rates` - the unnormalized transfer-matrix expressions
  (Moroz, Ann. Phys. 315, 352 (2005)) in double precision with SciPy's
  spherical Bessel functions, summed up to the first overflow: the baseline
  that the stable formulations improve on.

The extended-precision functions need mpmath (``pip install pystratify[references]``).
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

try:
    import mpmath as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None

__all__ = [
    "sphere_decay_rates",
    "layered_decay_rates",
    "layered_green_forms",
    "layered_green_sums",
    "sphere_extinction",
    "bhmie",
    "classical_decay_rates",
]


def _require_mpmath():
    if mp is None:
        raise ImportError("extended-precision references need mpmath: pip install mpmath")


def _psi_all(z, top):
    """psi_0..psi_top at complex z (mp), downward from exact top values."""
    f = lambda n: z * mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mp.mpf(1) / 2, z)  # noqa: E731
    out = [mp.mpc(0)] * (top + 2)
    out[top + 1], out[top] = f(top + 1), f(top)
    for n in range(top, 0, -1):
        out[n - 1] = (2 * n + 1) / z * out[n] - out[n + 1]
    return out


def _xi_all(z, top):
    """xi_0..xi_top at z (mp), upward (the dominant solution)."""
    out = [mp.mpc(0)] * (top + 1)
    out[0] = -1j * mp.exp(1j * z)
    out[1] = out[0] / z - mp.exp(1j * z)
    for n in range(1, top):
        out[n + 1] = (2 * n + 1) / z * out[n] - out[n - 1]
    return out


def sphere_decay_rates(radius, m, wavelength, r, n_host=1.0, orders=3000, dps=40):
    """(total_perp, total_par, rad_perp, rad_par) for a dipole at r > radius; mu = 1."""
    _require_mpmath()
    with mp.workdps(dps):
        k = 2 * mp.pi * n_host / mp.mpf(wavelength)
        x, X = k * mp.mpf(radius), k * mp.mpf(r)
        s = mp.mpc(complex(m)) / n_host
        top = orders + 1
        ps_x, ps_sx, ps_X = _psi_all(x, top), _psi_all(s * x, top), _psi_all(X, top)
        xi_x, xi_X = _xi_all(x, top), _xi_all(X, top)
        tot_perp = tot_par = rad_perp = rad_par = mp.mpf(0)
        for n in range(1, orders + 1):
            dps_x = ps_x[n - 1] - n * ps_x[n] / x
            dps_sx = ps_sx[n - 1] - n * ps_sx[n] / (s * x)
            dxi_x = xi_x[n - 1] - n * xi_x[n] / x
            a = (s * ps_sx[n] * dps_x - ps_x[n] * dps_sx) / (s * ps_sx[n] * dxi_x - xi_x[n] * dps_sx)
            b = (ps_sx[n] * dps_x - s * ps_x[n] * dps_sx) / (ps_sx[n] * dxi_x - s * xi_x[n] * dps_sx)
            u, du = ps_X[n], ps_X[n - 1] - n * ps_X[n] / X
            w, dw = xi_X[n], xi_X[n - 1] - n * xi_X[n] / X
            c_perp, c_par = n * (n + 1) * (2 * n + 1), 2 * n + 1
            # T = -a (TM), -b (TE); u_in = psi + T xi, u_out = xi
            tot_perp += c_perp * mp.re((u - a * w) * w)
            tot_par += c_par * (mp.re((u - b * w) * w) + mp.re((du - a * dw) * dw))
            rad_perp += c_perp * abs(u - a * w) ** 2
            rad_par += c_par * (abs(u - b * w) ** 2 + abs(du - a * dw) ** 2)
        return tuple(
            float(v)
            for v in (1.5 * tot_perp / X**4, 0.75 * tot_par / X**2, 1.5 * rad_perp / X**4, 0.75 * rad_par / X**2)
        )


def sphere_extinction(x, m, orders, dps=40):
    """Q_ext of a homogeneous sphere in mpmath (Bohren & Huffman Eq. 4.53)."""
    _require_mpmath()
    with mp.workdps(dps):
        x, s = mp.mpf(x), mp.mpc(complex(m))
        px, psx, xx = _psi_all(x, orders + 1), _psi_all(s * x, orders + 1), _xi_all(x, orders + 1)
        q = mp.mpf(0)
        for n in range(1, orders + 1):
            dpx, dxx = px[n - 1] - n * px[n] / x, xx[n - 1] - n * xx[n] / x
            dpsx = psx[n - 1] - n * psx[n] / (s * x)
            a = (s * psx[n] * dpx - px[n] * dpsx) / (s * psx[n] * dxx - xx[n] * dpsx)
            b = (psx[n] * dpx - s * px[n] * dpsx) / (psx[n] * dxx - s * xx[n] * dpsx)
            q += (2 * n + 1) * mp.re(a + b)
        return float(2 * q / x**2)


def bhmie(x, m, nmax=None):
    """(Q_ext, Q_sca) of a homogeneous sphere, Bohren & Huffman (1983) Appendix A."""
    nstop = int(x + 4 * x ** (1 / 3) + 2) if nmax is None else nmax
    mx = m * x
    nmx = int(max(nstop, abs(mx))) + 16
    d = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        d[n - 1] = n / mx - 1 / (d[n] + n / mx)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = complex(psi1, -chi1)
    q_ext = q_sca = 0.0
    for n in range(1, nstop + 1):
        psi = (2 * n - 1) * psi1 / x - psi0
        chi = (2 * n - 1) * chi1 / x - chi0
        xi = complex(psi, -chi)
        a = ((d[n] / m + n / x) * psi - psi1) / ((d[n] / m + n / x) * xi - xi1)
        b = ((m * d[n] + n / x) * psi - psi1) / ((m * d[n] + n / x) * xi - xi1)
        q_sca += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
        q_ext += (2 * n + 1) * (a.real + b.real)
        psi0, psi1, chi0, chi1, xi1 = psi1, psi, chi1, chi, xi
    return 2 * q_ext / x**2, 2 * q_sca / x**2


def layered_decay_rates(radii, n, wavelength, r, orders, dps=150, mu=None, dipole="electric"):
    """Shell-normalised (total, radiative) rates, each [perp, par], of an electric or magnetic
    dipole at radius r in a lossless shell of a layered sphere, from the 2x2 transfer matrices of
    Moroz (2005) in mpmath.  Direct products cancel like (R_out/R_in)^(2l) across a shell, so
    ``dps`` must exceed ~ 2 l log10 of the largest radius ratio plus the digits wanted.

    With (A_r, B_r) the regular and (A_o, B_o) the outgoing (B = 1 in the host) coefficients of
    the electric field in the emitter's shell and W = A_r B_o - B_r A_o, the per-order Green's
    function is u_reg u_out / W and the radiated amplitude u_reg / W.  The radial electric
    dipole couples to TM, the radial magnetic dipole to TE; the radiative factor is
    n_d mu_d / (n_h mu_h) for both (energy conservation fixes it: total = radiative for a
    lossless sphere).
    """
    _require_mpmath()
    mu = [1.0] * len(n) if mu is None else mu
    with mp.workdps(dps):
        N = len(radii)
        R = [mp.mpf(v) for v in radii]
        nn = [mp.mpc(complex(v)) for v in n]
        mm = [mp.mpc(complex(v)) for v in mu]
        k = [2 * mp.pi * v / mp.mpf(wavelength) for v in nn]
        top = orders + 1

        def funcs(z):
            p, x = _psi_all(z, top), _xi_all(z, top)
            dp = [None] + [p[l - 1] - l * p[l] / z for l in range(1, top + 1)]
            dx = [None] + [x[l - 1] - l * x[l] / z for l in range(1, top + 1)]
            return p, dp, x, dx

        iface = [(funcs(k[j] * R[j]), funcs(k[j + 1] * R[j])) for j in range(N)]
        d = sum(1 for Rj in radii if r >= Rj)
        X = k[d] * mp.mpf(r)
        pe, dpe, xe, dxe = funcs(X)
        radial, other = (False, True) if dipole == "electric" else (True, False)  # TE flag
        tot, rad = [mp.mpf(0)] * 2, [mp.mpf(0)] * 2
        for l in range(1, orders + 1):
            G, Gd, F, Fd = {}, {}, {}, {}
            for te in (False, True):
                mats = []
                for j in range(N):
                    (p, dp, x, dx), (pt, dpt, xt, dxt) = iface[j]
                    eta, mr = nn[j] / nn[j + 1], mm[j] / mm[j + 1]
                    a, b = (eta, mr) if te else (mr, eta)
                    mats.append(
                        [
                            [dx[l] * pt[l] * a - x[l] * dpt[l] * b, dx[l] * xt[l] * a - x[l] * dxt[l] * b],
                            [-dp[l] * pt[l] * a + p[l] * dpt[l] * b, -dp[l] * xt[l] * a + p[l] * dxt[l] * b],
                        ]
                    )
                A, B = mp.mpf(1), mp.mpf(0)
                for j in range(d):  # inner -> outer through the inverse matrices
                    m = mats[j]
                    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
                    A, B = (m[1][1] * A - m[0][1] * B) / det, (-m[1][0] * A + m[0][0] * B) / det
                Ao, Bo = mp.mpf(0), mp.mpf(1)
                for j in range(N - 1, d - 1, -1):
                    m = mats[j]
                    Ao, Bo = m[0][0] * Ao + m[0][1] * Bo, m[1][0] * Ao + m[1][1] * Bo
                W = A * Bo - B * Ao
                u, du = A * pe[l] + B * xe[l], A * dpe[l] + B * dxe[l]
                v, dv = Ao * pe[l] + Bo * xe[l], Ao * dpe[l] + Bo * dxe[l]
                G[te], Gd[te], F[te], Fd[te] = u * v / W, du * dv / W, u / W, du / W
            cp, ct = l * (l + 1) * (2 * l + 1), 2 * l + 1
            tot[0] += cp * mp.re(G[radial])
            tot[1] += ct * (mp.re(G[other]) + mp.re(Gd[radial]))
            rad[0] += cp * abs(F[radial]) ** 2
            rad[1] += ct * (abs(F[other]) ** 2 + abs(Fd[radial]) ** 2)
        f_rad = mp.re(nn[d] * mm[d] / (nn[-1] * mm[-1]))
        X = mp.re(X)
        scale = [1.5 / X**4, 0.75 / X**2]
        return [float(scale[i] * tot[i]) for i in range(2)], [float(f_rad * scale[i] * rad[i]) for i in range(2)]


def layered_green_forms(radii, n, wavelength, r, orders, dps=150, mu=None):
    """Per-order scattered Green's forms at an emitter at radius r in a lossless shell of a layered
    sphere, from the 2x2 transfer matrices of Moroz (2005) in mpmath.

    With u = u_reg and v = u_out (see :func:`layered_decay_rates`) and the free parts of the
    emitter's shell removed, returns complex arrays (2, orders), indexed [TM, TE], of

        G   = u v / W - psi xi,
        G^m = (u v' + u' v) / (2 W) - (psi xi' + psi' xi) / 2,
        G^d = u' v' / W - psi' xi',

    the value-value, value-derivative and derivative-derivative forms (orders 1..``orders``).
    ``dps`` as in :func:`layered_decay_rates`.
    """
    _require_mpmath()
    mu = [1.0] * len(n) if mu is None else mu
    out = np.zeros((3, 2, orders), dtype=complex)
    with mp.workdps(dps):
        N = len(radii)
        R = [mp.mpf(v) for v in radii]
        nn = [mp.mpc(complex(v)) for v in n]
        mm = [mp.mpc(complex(v)) for v in mu]
        k = [2 * mp.pi * v / mp.mpf(wavelength) for v in nn]
        top = orders + 1

        def funcs(z):
            p, x = _psi_all(z, top), _xi_all(z, top)
            dp = [None] + [p[l - 1] - l * p[l] / z for l in range(1, top + 1)]
            dx = [None] + [x[l - 1] - l * x[l] / z for l in range(1, top + 1)]
            return p, dp, x, dx

        iface = [(funcs(k[j] * R[j]), funcs(k[j + 1] * R[j])) for j in range(N)]
        d = sum(1 for Rj in radii if r >= Rj)
        pe, dpe, xe, dxe = funcs(k[d] * mp.mpf(r))
        for l in range(1, orders + 1):
            for pol, te in ((0, False), (1, True)):
                mats = []
                for j in range(N):
                    (p, dp, x, dx), (pt, dpt, xt, dxt) = iface[j]
                    eta, mr = nn[j] / nn[j + 1], mm[j] / mm[j + 1]
                    a, b = (eta, mr) if te else (mr, eta)
                    mats.append(
                        [
                            [dx[l] * pt[l] * a - x[l] * dpt[l] * b, dx[l] * xt[l] * a - x[l] * dxt[l] * b],
                            [-dp[l] * pt[l] * a + p[l] * dpt[l] * b, -dp[l] * xt[l] * a + p[l] * dxt[l] * b],
                        ]
                    )
                A, B = mp.mpf(1), mp.mpf(0)
                for j in range(d):  # inner -> outer through the inverse matrices
                    m = mats[j]
                    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
                    A, B = (m[1][1] * A - m[0][1] * B) / det, (-m[1][0] * A + m[0][0] * B) / det
                Ao, Bo = mp.mpf(0), mp.mpf(1)
                for j in range(N - 1, d - 1, -1):
                    m = mats[j]
                    Ao, Bo = m[0][0] * Ao + m[0][1] * Bo, m[1][0] * Ao + m[1][1] * Bo
                W = A * Bo - B * Ao
                u, du = A * pe[l] + B * xe[l], A * dpe[l] + B * dxe[l]
                v, dv = Ao * pe[l] + Bo * xe[l], Ao * dpe[l] + Bo * dxe[l]
                out[0, pol, l - 1] = complex(u * v / W - pe[l] * xe[l])
                out[1, pol, l - 1] = complex((u * dv + du * v) / (2 * W) - (pe[l] * dxe[l] + dpe[l] * xe[l]) / 2)
                out[2, pol, l - 1] = complex(du * dv / W - dpe[l] * dxe[l])
    return out[0], out[1], out[2]


def layered_green_sums(radii, n, wavelength, r, orders, dps=150, mu=None, dipole="electric"):
    """Complex scattered Green's sums [perp, par] of an electric or magnetic dipole at radius r in a
    lossless shell of a layered sphere (mpmath; see :func:`layered_green_forms`).

    The forms G and G^d of every order, weighted as the decay rates and summed over ``orders``
    multipoles in extended precision, shell normalization: the total rate is 1 + Re(sum) and the
    frequency shift (omega - omega_0)/Gamma_0 is Im(sum)/2.  ``dps`` as in :func:`layered_decay_rates`.
    """
    _require_mpmath()
    mu = [1.0] * len(n) if mu is None else mu
    with mp.workdps(dps):
        N = len(radii)
        R = [mp.mpf(v) for v in radii]
        nn = [mp.mpc(complex(v)) for v in n]
        mm = [mp.mpc(complex(v)) for v in mu]
        k = [2 * mp.pi * v / mp.mpf(wavelength) for v in nn]
        top = orders + 1

        def funcs(z):
            p, x = _psi_all(z, top), _xi_all(z, top)
            dp = [None] + [p[l - 1] - l * p[l] / z for l in range(1, top + 1)]
            dx = [None] + [x[l - 1] - l * x[l] / z for l in range(1, top + 1)]
            return p, dp, x, dx

        iface = [(funcs(k[j] * R[j]), funcs(k[j + 1] * R[j])) for j in range(N)]
        d = sum(1 for Rj in radii if r >= Rj)
        X = k[d] * mp.mpf(r)
        pe, dpe, xe, dxe = funcs(X)
        radial, other = (False, True) if dipole == "electric" else (True, False)  # TE flag
        tot = [mp.mpc(0)] * 2
        for l in range(1, orders + 1):
            G, Gd = {}, {}
            for te in (False, True):
                mats = []
                for j in range(N):
                    (p, dp, x, dx), (pt, dpt, xt, dxt) = iface[j]
                    eta, mr = nn[j] / nn[j + 1], mm[j] / mm[j + 1]
                    a, b = (eta, mr) if te else (mr, eta)
                    mats.append(
                        [
                            [dx[l] * pt[l] * a - x[l] * dpt[l] * b, dx[l] * xt[l] * a - x[l] * dxt[l] * b],
                            [-dp[l] * pt[l] * a + p[l] * dpt[l] * b, -dp[l] * xt[l] * a + p[l] * dxt[l] * b],
                        ]
                    )
                A, B = mp.mpf(1), mp.mpf(0)
                for j in range(d):  # inner -> outer through the inverse matrices
                    m = mats[j]
                    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
                    A, B = (m[1][1] * A - m[0][1] * B) / det, (-m[1][0] * A + m[0][0] * B) / det
                Ao, Bo = mp.mpf(0), mp.mpf(1)
                for j in range(N - 1, d - 1, -1):
                    m = mats[j]
                    Ao, Bo = m[0][0] * Ao + m[0][1] * Bo, m[1][0] * Ao + m[1][1] * Bo
                W = A * Bo - B * Ao
                u, du = A * pe[l] + B * xe[l], A * dpe[l] + B * dxe[l]
                v, dv = Ao * pe[l] + Bo * xe[l], Ao * dpe[l] + Bo * dxe[l]
                G[te] = u * v / W - pe[l] * xe[l]
                Gd[te] = du * dv / W - dpe[l] * dxe[l]
            cp, ct = l * (l + 1) * (2 * l + 1), 2 * l + 1
            tot[0] += cp * G[radial]
            tot[1] += ct * (G[other] + Gd[radial])
        X = mp.re(X)
        scale = [1.5 / X**4, 0.75 / X**2]
        return [complex(scale[i] * tot[i]) for i in range(2)]


def _riccati_double(l, z):
    with np.errstate(all="ignore"):  # overflow to inf/nan is the point of this baseline
        j, y = spherical_jn(l, z), spherical_yn(l, z)
        jd, yd = spherical_jn(l, z, True), spherical_yn(l, z, True)
        return z * j, j + z * jd, z * (j + 1j * y), (j + 1j * y) + z * (jd + 1j * yd)


def classical_decay_rates(radii, n, wavelength, r, orders, mu=None, dipole="electric"):
    """Unnormalized transfer-matrix decay rates in double precision (shell normalization).

    Returns ``(total, radiative, last)``, each rate ``[perp, par]``, summed over
    the orders before the first non-finite term; ``last`` is that number of
    orders.  ``h = j + i y`` is formed from SciPy's ``spherical_jn`` and
    ``spherical_yn``, and the composite matrices are plain products, as in the
    published formulation.
    """
    radii = np.atleast_1d(np.asarray(radii, float))
    n = np.asarray(n, complex)
    mu = np.ones(n.size, complex) if mu is None else np.asarray(mu, complex)
    N = radii.size
    k = 2 * np.pi * n / wavelength
    l = np.arange(1, orders + 1)
    d = int(np.searchsorted(radii, r, side="right"))
    xe = k[d].real * r
    pe, dpe, ze, dze = _riccati_double(l, xe)
    out = {}
    with np.errstate(all="ignore"):
        for te in (False, True):
            mats = []
            for j in range(N):
                p1, dp1, z1, dz1 = _riccati_double(l, k[j] * radii[j])
                p2, dp2, z2, dz2 = _riccati_double(l, k[j + 1] * radii[j])
                eta, mr = n[j] / n[j + 1], mu[j] / mu[j + 1]
                a, b = (eta, mr) if te else (mr, eta)
                mats.append(
                    -1j
                    * np.array(
                        [
                            [dz1 * p2 * a - z1 * dp2 * b, dz1 * z2 * a - z1 * dz2 * b],
                            [-dp1 * p2 * a + p1 * dp2 * b, -dp1 * z2 * a + p1 * dz2 * b],
                        ]
                    )
                )
            A, B = np.ones(orders, complex), np.zeros(orders, complex)
            for j in range(d):
                m = mats[j]
                det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
                A, B = (m[1, 1] * A - m[0, 1] * B) / det, (-m[1, 0] * A + m[0, 0] * B) / det
            Ao, Bo = np.zeros(orders, complex), np.ones(orders, complex)
            for j in range(N - 1, d - 1, -1):
                m = mats[j]
                Ao, Bo = m[0, 0] * Ao + m[0, 1] * Bo, m[1, 0] * Ao + m[1, 1] * Bo
            W = A * Bo - B * Ao
            u, du, v, dv = A * pe + B * ze, A * dpe + B * dze, Ao * pe + Bo * ze, Ao * dpe + Bo * dze
            out[te] = (u * v / W, du * dv / W, u / W, du / W)
        radial, other = (False, True) if dipole == "electric" else (True, False)
        G, Gd, F, Fd = out[radial]
        G2, _, F2, _ = out[other]
        ok = np.cumprod(np.all(np.isfinite([G, Gd, F, Fd, G2, F2]), axis=0)).astype(bool)
        G, Gd, F, Fd, G2, F2 = (np.where(ok, v, 0) for v in (G, Gd, F, Fd, G2, F2))
        c1, c2 = l * (l + 1) * (2 * l + 1), 2 * l + 1
        f_rad = (n[d] * mu[d] / (n[-1] * mu[-1])).real
        total = [1.5 / xe**4 * np.sum(c1 * G.real), 0.75 / xe**2 * np.sum(c2 * (G2.real + Gd.real))]
        radiative = [
            f_rad * 1.5 / xe**4 * np.sum(c1 * np.abs(F) ** 2),
            f_rad * 0.75 / xe**2 * np.sum(c2 * (np.abs(F2) ** 2 + np.abs(Fd) ** 2)),
        ]
    return np.array(total), np.array(radiative), int(ok.sum())
