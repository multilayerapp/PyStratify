"""The normalized formulation (paper equations) against the solver and extended precision."""

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
