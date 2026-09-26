"""Extended-precision references (mpmath), used only in tests.

Precision grows with |Im z| and the order, so the cancellation in
H^(1) = J + iY and the size of high-order values are absorbed.
"""

import mpmath as mp
import numpy as np


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


def chiral_t_matrix(radii, n_list, kappa_list, mu_list, wavelength, order, dps=80, amplitudes_only=False):
    """Logarithms of the exact host T-matrix block (helicity basis [out, in]) of a sphere with chiral layers.

    Direct 4x4 transfer matrices in extended precision, independent of the
    solver's scaled recursion.  In layer s the field of order ``order`` is
    sum_c alpha_c W_c^(1) + beta_c W_c^(3), W_c = M + c N at k_c = k0 (n + c kappa);
    the continuous quantities at radius r are (E_X, E_Z, H_X, H_Z) with
    E_X = v_+ + v_-, E_Z = d_+ - d_-, i Z H_X = v_+ - v_-, i Z H_Z = d_+ + d_-,
    v_c = u_c(x_c)/x_c, d_c = u_c'(x_c)/x_c.

    With ``amplitudes_only`` returns, per layer, (log alpha, log beta) as 2x2
    arrays [channel, incident helicity] for unit incident waves in the host.
    """
    with mp.workdps(dps):
        k0 = 2 * mp.pi / mp.mpf(wavelength)
        layers = [
            (
                [k0 * (mp.mpc(complex(n)) + c * mp.mpc(complex(k))) for c in (1, -1)],
                mp.mpc(complex(m)) / mp.mpc(complex(n)),
            )
            for n, k, m in zip(n_list, kappa_list, mu_list)
        ]

        def boundary(s, r):
            """Columns divided by their E_X value (psi, xi span hundreds of decades), and those values."""
            ks, z = layers[s]
            rows = [[mp.mpc(0)] * 4 for _ in range(4)]
            scale = [mp.mpc(0)] * 4
            for c, sign in enumerate((1, -1)):
                x = ks[c] * mp.mpf(r)
                for col, (f, df) in ((c, (_psi, _dpsi)), (c + 2, (_xi, _dxi))):
                    value = f(order, x)
                    scale[col] = value / x
                    d = df(order, x) / value
                    rows[0][col] = 1
                    rows[1][col] = sign * d
                    rows[2][col] = sign / (1j * z)
                    rows[3][col] = d / (1j * z)
            return mp.matrix(rows), scale

        g = mp.eye(4)
        cumulative = [g]
        for j, r in enumerate(radii):
            (inner, s_in), (outer, s_out) = boundary(j, r), boundary(j + 1, r)
            m = mp.inverse(outer) * inner
            g = mp.matrix([[m[a, b] * s_in[b] / s_out[a] for b in range(4)] for a in range(4)]) * g
            cumulative.append(g)
        top = mp.matrix([[g[0, 0], g[0, 1]], [g[1, 0], g[1, 1]]])
        core = mp.inverse(top)  # core amplitudes giving a unit incident wave of each helicity
        layers = []
        for c in cumulative:
            amplitudes = c * mp.matrix([[core[0, 0], core[0, 1]], [core[1, 0], core[1, 1]], [0, 0], [0, 0]])
            layers.append(
                [
                    [[complex(mp.log(v)) if v != 0 else complex(-mp.inf) for v in (amplitudes[i, 0], amplitudes[i, 1])]]
                    for i in range(4)
                ]
            )
        if amplitudes_only:
            return [
                (np.array([row[0] for row in layer[:2]]), np.array([row[0] for row in layer[2:]])) for layer in layers
            ]
        bottom = mp.matrix([[g[2, 0], g[2, 1]], [g[3, 0], g[3, 1]]])
        t = bottom * mp.inverse(top)
        return [[complex(mp.log(t[i, j])) if t[i, j] != 0 else complex(-mp.inf) for j in range(2)] for i in range(2)]
