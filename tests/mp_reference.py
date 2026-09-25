"""Extended-precision references (mpmath), used only in tests.

Precision grows with |Im z| and the order, so the cancellation in
H^(1) = J + iY and the size of high-order values are absorbed.
"""

import mpmath as mp


def _dps(z, n):
    return int(40 + abs(complex(z).imag) + n / 3)


def log_psi_xi(n, z):
    """(log psi_n(z), log xi_n(z)) as Python complex numbers."""
    with mp.workdps(_dps(z, n)):
        zz = mp.mpc(complex(z).real, complex(z).imag)
        f = zz * mp.sqrt(mp.pi / (2 * zz))
        return complex(mp.log(f * mp.besselj(n + 0.5, zz))), complex(mp.log(f * mp.hankel1(n + 0.5, zz)))


def _psi(n, z):
    return z * mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mp.mpf(1) / 2, z)


def _xi(n, z):
    return z * mp.sqrt(mp.pi / (2 * z)) * mp.hankel1(n + mp.mpf(1) / 2, z)


def _dpsi(n, z):
    return _psi(n - 1, z) - n * _psi(n, z) / z


def _dxi(n, z):
    return _xi(n - 1, z) - n * _xi(n, z) / z


def coefficients(radii, n_list, mu_list, wavelength, order, te):
    """Exact per-shell coefficients (A, B) of order ``order``, 60 digits.

    Returns (regular with A_host = 1, outgoing with B_host = 1), each a list
    over shells of (A, B), from the 2x2 transfer matrices of Moroz (2005).
    """
    with mp.workdps(60):
        radii = [mp.mpf(r) for r in radii]
        n_list = [mp.mpc(complex(v)) for v in n_list]
        mu_list = [mp.mpc(complex(v)) for v in mu_list]
        N = len(radii)
        k = [2 * mp.pi * v / mp.mpf(wavelength) for v in n_list]
        matrices = []
        for j in range(N):
            x, xt = k[j] * radii[j], k[j + 1] * radii[j]
            eta, mr = n_list[j] / n_list[j + 1], mu_list[j] / mu_list[j + 1]
            a, b = (eta, mr) if te else (mr, eta)
            n_ = order
            matrices.append(
                mp.matrix(
                    [
                        [
                            _dxi(n_, x) * _psi(n_, xt) * a - _xi(n_, x) * _dpsi(n_, xt) * b,
                            _dxi(n_, x) * _xi(n_, xt) * a - _xi(n_, x) * _dxi(n_, xt) * b,
                        ],
                        [
                            -_dpsi(n_, x) * _psi(n_, xt) * a + _psi(n_, x) * _dpsi(n_, xt) * b,
                            -_dpsi(n_, x) * _xi(n_, xt) * a + _psi(n_, x) * _dxi(n_, xt) * b,
                        ],
                    ]
                )
                * (-1j)
            )
        v = mp.matrix([[1], [0]])
        regular = [v]
        for m in matrices:
            det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
            v = mp.matrix([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]]) / det * v
            regular.append(v)
        scale = regular[-1][0]
        regular = [(c[0] / scale, c[1] / scale) for c in regular]
        w = mp.matrix([[0], [1]])
        outgoing = [w]
        for m in reversed(matrices):
            w = m * w
            outgoing.insert(0, w)
        return regular, [(c[0], c[1]) for c in outgoing]
