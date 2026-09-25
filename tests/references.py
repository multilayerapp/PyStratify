"""Independent references for the tests and the convergence benchmarks.

Nothing here shares code with PyStratify:

* :func:`sphere_decay_rates` - Mie decay rates of a dipole outside a
  homogeneous sphere (Majic & Le Ru, Appl. Opt. 59, 1293 (2020), Eqs. 34-37)
  in mpmath, with Riccati-Bessel functions from exact top values and the
  stable recurrences (downward for psi, upward for xi), so any order is exact
  to the working precision;
* :func:`layered_decay_rates` - the same for an emitter inside or outside a
  layered sphere, from high-precision 2x2 transfer matrices;
* :func:`bhmie` - Bohren & Huffman's BHMIE in NumPy (logarithmic derivative
  D_n(mx) by downward recurrence), the classic algorithm for large spheres.
  It is accurate to ~1e-13 for absorbing spheres; for weakly absorbing ones
  with x >~ 100 it loses digits (1e-4 at x = 1000, m = 1.33 + 1e-8 i, against
  mpmath), so use :func:`sphere_extinction` there.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np


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


def layered_decay_rates(radii, n, wavelength, r, orders, dps=150):
    """Shell-normalised (total, radiative) rates, each [perp, par], of an electric dipole at
    radius r in a lossless shell of a layered sphere (mu = 1), from the 2x2 transfer matrices of
    Moroz (2005) in mpmath.  Direct products cancel like (R_out/R_in)^(2l) across a shell, so
    ``dps`` must exceed ~ 2 l log10 of the largest radius ratio plus the digits wanted.

    With (A_r, B_r) the regular and (A_o, B_o) the outgoing (B = 1 in the host) coefficients in
    the emitter's shell and W = A_r B_o - B_r A_o, the per-order Green's function is
    u_reg u_out / W and the radiated amplitude u_reg / W.
    """
    with mp.workdps(dps):
        N = len(radii)
        R = [mp.mpf(v) for v in radii]
        nn = [mp.mpc(complex(v)) for v in n]
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
        out = [mp.mpf(0)] * 4
        for l in range(1, orders + 1):
            G, Gd, F, Fd = {}, {}, {}, {}
            for te in (False, True):
                mats = []
                for j in range(N):
                    (p, dp, x, dx), (pt, dpt, xt, dxt) = iface[j]
                    eta = nn[j] / nn[j + 1]
                    a, b = (eta, 1) if te else (1, eta)
                    mats.append(
                        mp.matrix(
                            [
                                [dx[l] * pt[l] * a - x[l] * dpt[l] * b, dx[l] * xt[l] * a - x[l] * dxt[l] * b],
                                [-dp[l] * pt[l] * a + p[l] * dpt[l] * b, -dp[l] * xt[l] * a + p[l] * dxt[l] * b],
                            ]
                        )
                    )
                reg = mp.matrix([[1], [0]])
                for j in range(d):  # inner -> outer: v_{j+1} = M_j^{-1} v_j
                    m = mats[j]
                    reg = (
                        mp.matrix([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]])
                        * reg
                        / (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
                    )
                outg = mp.matrix([[0], [1]])
                for j in range(N - 1, d - 1, -1):
                    outg = mats[j] * outg
                Ar, Br, Ao, Bo = reg[0], reg[1], outg[0], outg[1]
                W = Ar * Bo - Br * Ao
                u, du = Ar * pe[l] + Br * xe[l], Ar * dpe[l] + Br * dxe[l]
                v, dv = Ao * pe[l] + Bo * xe[l], Ao * dpe[l] + Bo * dxe[l]
                G[te], Gd[te], F[te], Fd[te] = u * v / W, du * dv / W, u / W, du / W
            cp, ct = l * (l + 1) * (2 * l + 1), 2 * l + 1
            out[0] += cp * mp.re(G[False])
            out[1] += ct * (mp.re(G[True]) + mp.re(Gd[False]))
            out[2] += cp * abs(F[False]) ** 2
            out[3] += ct * (abs(F[True]) ** 2 + abs(Fd[False]) ** 2)
        f_rad = mp.re(nn[d] / nn[-1])
        X = mp.re(X)
        tp, tpar = 1.5 * out[0] / X**4, 0.75 * out[1] / X**2
        rp, rpar = f_rad * 1.5 * out[2] / X**4, f_rad * 0.75 * out[3] / X**2
        return [float(tp), float(tpar)], [float(rp), float(rpar)]
