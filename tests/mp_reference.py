"""60-digit reference transfer matrices (STRATIFY's direct products, OSAC Eqs. 10-14) in mpmath.

Used only in tests: extended precision absorbs the overflow and cancellation
that limit the double-precision products."""

import mpmath as mp

mp.mp.dps = 60


def psi(n, z):
    return z * mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mp.mpf(1) / 2, z)


def xi(n, z):
    return z * mp.sqrt(mp.pi / (2 * z)) * mp.hankel1(n + mp.mpf(1) / 2, z)


def dpsi(n, z):
    return psi(n - 1, z) - n * psi(n, z) / z


def dxi(n, z):
    return xi(n - 1, z) - n * xi(n, z) / z


def coeffs(rad, ref, mu, lam, n, pol):
    """plane-wave (A_{N+1}=1) and outgoing (B_{N+1}=1) coefficients per shell, exact"""
    rad = [mp.mpf(r) for r in rad]
    ref = [mp.mpc(complex(v)) for v in ref]
    mu = [mp.mpc(complex(v)) for v in mu]
    N = len(rad)
    k = [2 * mp.pi * v / mp.mpf(lam) for v in ref]
    Tb = []
    for j in range(N):
        x = k[j] * rad[j]
        xt = k[j + 1] * rad[j]
        eta = ref[j] / ref[j + 1]
        m = mu[j] / mu[j + 1]
        a, b = (eta, m) if pol == "m" else (m, eta)
        M = mp.matrix(
            [
                [
                    dxi(n, x) * psi(n, xt) * a - xi(n, x) * dpsi(n, xt) * b,
                    dxi(n, x) * xi(n, xt) * a - xi(n, x) * dxi(n, xt) * b,
                ],
                [
                    -dpsi(n, x) * psi(n, xt) * a + psi(n, x) * dpsi(n, xt) * b,
                    -dpsi(n, x) * xi(n, xt) * a + psi(n, x) * dxi(n, xt) * b,
                ],
            ]
        ) * (-1j)
        Tb.append(M)
    # regular: start core (1,0), apply inverse of T- going out
    v = mp.matrix([[1], [0]])
    reg = [v]
    for j in range(N):
        M = Tb[j]
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        Mi = mp.matrix([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det
        v = Mi * v
        reg.append(v)
    scale = reg[-1][0]
    reg = [r / scale for r in reg]  # A_{N+1}=1
    w = mp.matrix([[0], [1]])
    out = [w]
    for j in range(N - 1, -1, -1):
        w = Tb[j] * w
        out.insert(0, w)
    return [(r[0], r[1]) for r in reg], [(o[0], o[1]) for o in out]
