"""The normalized formulation (paper equations) against the solver and extended precision."""

import mpmath as mp
import numpy as np
import pytest

import pystratify as ps
from pystratify.normalized import auxiliary
from pystratify.references import classical_decay_rates, layered_decay_rates

AU = 0.27 + 2.93j
CASES = [  # radii, n, mu, wavelength, r
    ([50.0], [0.093 + 4j, 1.0], [1, 1], 633.0, 51.0),
    ([50.0, 55.0], [1.45, AU, 1.33], [1, 1, 1], 614.0, 49.5),
    ([50.0, 55.0], [1.45, AU, 1.33], [1, 1, 1], 614.0, 56.0),
    ([50.0, 60.0], [AU, 1.45, 1.33], [1, 1, 1], 614.0, 55.0),
    ([10.0, 13.0, 36.0, 48.0], [1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33], [1] * 5, 690.0, 35.0),
    ([40.0, 45.0, 60.0], [1.45, AU, 2.0, 1.33], [1, 1, 2.0, 1], 614.0, 50.0),
    ([40.0, 50.0], [1.45, np.sqrt(4.0 * (2 + 0.3j)), 1.33], [1, 2 + 0.3j, 1], 614.0, 55.0),  # lossy mu
]


def test_auxiliary_functions_against_log_riccati():
    for z in (0.7, 12.0 + 0.3j, 1.3 + 40j, 3000.0):
        L = 600
        tol = 1e-10 + 2e-12 * abs(z)  # near real zeros of psi both sides lose ~|z| eps
        A, B, P, lj = auxiliary(z, L)
        lp, lx = ps.log_riccati(np.array([z]), L)
        l = np.arange(1, L)
        assert np.allclose(np.exp(lp[0, l] + lx[0, l]), P[l], rtol=tol, atol=0)
        d_psi = np.exp(lp[0, l - 1] - lp[0, l]) - l / z
        assert np.allclose(A[l], d_psi, rtol=tol, atol=1e-12)
        # jbar = (2l+1)!! psi / z^(l+1)
        log_jbar = lp[0, l] - (l + 1) * np.log(z) + ps.normalized.log_double_factorial(l)
        d = log_jbar - lj[l]
        assert np.max(np.abs(d.real) + np.abs(np.exp(1j * d.imag) - 1)) < tol


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("radii, n, mu, wavelength, r", CASES)
def test_normalized_equals_solver(radii, n, mu, wavelength, r, dipole):
    main = ps.decay_rates(radii, n, wavelength, [r], mu, dipole=dipole, normalization="shell", tol=1e-11)
    norm = ps.normalized_decay_rates(radii, n, wavelength, r, main.orders_used[0], mu, dipole)
    assert np.allclose(norm.total, main.total[0], rtol=1e-9, atol=0)
    assert np.allclose(norm.radiative, main.radiative[0], rtol=1e-11, atol=0)
    assert np.allclose(norm.nonradiative, main.nonradiative[0], rtol=1e-9, atol=0)


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
def test_normalized_against_extended_precision_and_classical(dipole):
    radii, n, mu, lam, r = CASES[1]
    total, radiative = layered_decay_rates(radii, n, lam, r, 2400, 150, mu=mu, dipole=dipole)
    norm = ps.normalized_decay_rates(radii, n, lam, r, 2400, mu, dipole)
    assert np.allclose(norm.total, total, rtol=1e-10, atol=0)
    assert np.allclose(norm.radiative, radiative, rtol=1e-12, atol=0)
    tc, rc, last = classical_decay_rates(radii, n, lam, r, 2400, mu, dipole)
    assert last < 100 and np.max(np.abs(tc / total - 1)) > 0.01  # the unnormalized form overflows


def test_terms_and_free_space_limit():
    norm = ps.normalized_decay_rates([30.0, 60.0], [1.4, 1.4, 1.4], 600.0, 45.0, 40, terms=True)
    assert np.allclose(norm.total, 1, atol=1e-13) and np.allclose(norm.radiative, 1, atol=1e-12)
    assert norm.terms_total.shape == (40, 2)


def _mp_auxiliary(l, z):
    """(P_l, B_l, ln jbar_l, psi_l) in extended precision."""
    with mp.workdps(40 + int(abs(complex(z).imag)) + l // 3):
        zz = mp.mpc(complex(z).real, complex(z).imag)
        f = zz * mp.sqrt(mp.pi / (2 * zz))
        psi, xi = f * mp.besselj(l + 0.5, zz), f * mp.hankel1(l + 0.5, zz)
        dxi = f * mp.hankel1(l - 0.5, zz) - l * xi / zz
        log_jbar = mp.log(mp.fac2(2 * l + 1) * psi / zz ** (l + 1))
        return complex(psi * xi), complex(dxi / xi), complex(log_jbar), complex(psi)


REAL_ZEROS = {  # psi_l(z) = 0: P_l and jbar_l vanish there and A_l has a pole
    "pi": np.pi,
    "2 pi": 2 * np.pi,
    "pi (1 + 1e-12)": np.pi * (1 + 1e-12),
    "pi (1 + 1e-9)": np.pi * (1 + 1e-9),
    "zero of psi_1": float(mp.besseljzero(1.5, 1)),
    "zero of psi_2": float(mp.besseljzero(2.5, 1)),
    "3rd zero of psi_10": float(mp.besseljzero(10.5, 3)),
}


@pytest.mark.parametrize("z", REAL_ZEROS.values(), ids=REAL_ZEROS.keys())
def test_auxiliary_through_real_zeros(z):
    """Products of ratios are 0*inf forms at a real zero of psi_{l-1}; P, B and jbar must not be built that way.

    P, B and jbar keep full precision at every other order, and P/jbar (what the rates need)
    at every order, including the one at the zero."""
    L = 40
    A, B, P, lj = auxiliary(z, L)
    for l in range(1, L + 1):
        p, b, j, psi = _mp_auxiliary(l, z)
        assert abs(B[l] / b - 1) < 1e-13, ("B", l)
        assert abs(P[l] / np.exp(lj[l]) / (p / np.exp(j)) - 1) < 1e-12, ("P/jbar", l)
        at_zero = abs(psi) < 1e-8 * (abs(_mp_auxiliary(l - 1, z)[3]) + abs(_mp_auxiliary(l + 1, z)[3]))
        if not at_zero:
            assert abs(P[l] / p - 1) < 1e-13, ("P", l)
            assert abs(np.exp(lj[l] - j) - 1) < 1e-12, ("jbar", l)


@pytest.mark.parametrize("x", [0.01, 0.7, 3.1, 12.3])
def test_small_real_part_of_P(x):
    """Re P_l = psi_l^2 (real argument), the small real part of the reactive terms, to full precision."""
    L = 400
    P = auxiliary(x, L)[2]
    for l in (1, 2, 5, 10, 20, 50, 100, 200, 400):
        with mp.workdps(60):
            ref = (x * mp.sqrt(mp.pi / (2 * x)) * mp.besselj(l + 0.5, x)) ** 2
        if ref > mp.mpf(10) ** -300:
            assert abs(P[l].real / float(ref) - 1) < 1e-13, l


@pytest.mark.parametrize("z", [2 - 1.5j, 30 - 0.5j, 120 - 3j, 5 - 20j, 5 - 60j])
def test_auxiliary_in_amplifying_media(z):
    """The forward recurrence of B is unstable for strong gain (Im z <= -1); the product form takes over."""
    L = 200
    A, B, P, lj = auxiliary(z, L)
    for l in (1, 2, 3, 5, 10, 30, 60, 100, 200):
        p, b, j, _ = _mp_auxiliary(l, z)
        assert abs(B[l] / b - 1) < 1e-12 and abs(P[l] / p - 1) < 1e-12, l
        assert abs(np.exp(lj[l] - j) - 1) < 1e-12, l


ZERO_CASES = {  # radii, n, wavelength, r: an argument at (the double nearest) a real zero of psi_l
    "emitter at k r = pi": ([50.0], [1.45, 1.33], 532.0, 200.0),
    "interface at k r = pi": ([40.0, 100.0], [1.45, 3.5, 1.33], 700.0, 30.0),
    "interface at a zero of psi_1": (
        [40.0, 100.0], [1.45, 3.5, 1.33], 2 * np.pi * 350.0 / float(mp.besseljzero(1.5, 1)), 101.0,
    ),  # fmt: skip
}


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("case", ZERO_CASES.values(), ids=ZERO_CASES.keys())
def test_rates_at_real_zeros(case, dipole):
    """Decay rates with an emitter or an interface argument at a real zero, against extended precision."""
    radii, n, lam, r = case
    L = ps.decay_rates(radii, n, lam, [r], normalization="shell", tol=1e-13, dipole=dipole, warn=False).orders_used[0]
    total, radiative = layered_decay_rates(radii, n, lam, r, L, 60, dipole=dipole)
    norm = ps.normalized_decay_rates(radii, n, lam, r, L, None, dipole)
    assert np.allclose(norm.total, total, rtol=1e-12, atol=0)
    assert np.allclose(norm.radiative, radiative, rtol=1e-12, atol=0)
