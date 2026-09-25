"""The overflow-free solver against exact references and the interface conditions."""

import mpmath as mp
import numpy as np
import pytest

import pystratify as ps
from pystratify import TE, TM
from pystratify.riccati import log_riccati

from .mp_reference import coefficients

WAVELENGTH = 690.0
MATRYOSHKA = ([10.0, 13.0, 36.0, 48.0], [1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33])  # 3-nm Au shell


def log_error(log_value, exact):
    if exact == 0:
        return 0.0
    d = complex(mp.log(exact)) - log_value
    return abs(d.real) + abs(np.exp(1j * d.imag) - 1)


@pytest.mark.parametrize(
    "radii, n, tol",
    [
        ([50, 70, 90], [1.45 + 0.2j, 2.0, 1.6 + 0.05j, 1.0], 1e-10),
        (*MATRYOSHKA, 1e-10),
        ([30, 31, 60], [2.5 + 0.01j, 0.05 + 4.0j, 1.5, 1.33], 1e-10),  # 1-nm Ag-like film
        # nearly index-matched: B is set by a contrast of 1e-6, so it is known to ~ l eps / contrast
        ([40, 60], [1.5, 1.5 + 1e-6, 1.5 + 2e-6], 1e-7),
        ([40, 60], [1.5, 1.6 - 0.02j, 1.33], 1e-10),  # gain shell
    ],
)
def test_coefficients_against_60_digit_transfer_matrices(radii, n, tol):
    mu = [1.0] * len(n)
    sol = ps.solve(radii, n, WAVELENGTH, mu, l_max=160)
    for pol in (TM, TE):
        for order in (1, 7, 40, 100, 160):
            regular, outgoing = coefficients(radii, n, mu, WAVELENGTH, order, te=pol == TE)
            for s in range(len(n)):
                assert log_error(sol.log_a[pol, s, 0, order - 1], regular[s][0]) < tol, (pol, order, s)
                assert log_error(sol.log_b[pol, s, 0, order - 1], regular[s][1]) < tol, (pol, order, s)
                assert log_error(sol.log_b_out[pol, s, 0, order - 1], outgoing[s][1]) < tol, (pol, order, s)


def _interface_mismatch(sol, radii, n, mu, j, pol):
    """Relative violation of the two matching conditions at interface j, all orders."""
    l = sol.orders
    k = sol.k[0]
    x = np.array([k[j] * radii[j], k[j + 1] * radii[j]])
    lp, lx = log_riccati(x, l.size)
    eta, mu_ratio = n[j] / n[j + 1], mu[j] / mu[j + 1]
    c_value, c_deriv = (eta, mu_ratio) if pol == TE else (mu_ratio, eta)
    worst = 0.0

    def field(s, side, derivative):
        la, lb = sol.log_a[pol, s, 0], sol.log_b[pol, s, 0]
        shift = np.maximum(la.real + lp[side, l].real, lb.real + lx[side, l].real)
        v = np.exp(la + lp[side, l] - shift) + np.exp(lb + lx[side, l] - shift)
        if derivative:
            v = np.exp(la + lp[side, l - 1] - shift) + np.exp(lb + lx[side, l - 1] - shift) - l / x[side] * v
        return np.log(v) + shift

    for derivative, c in ((False, c_value), (True, c_deriv)):
        d = field(j, 0, derivative) - (np.log(c) + field(j + 1, 1, derivative))
        worst = max(worst, float(np.max(np.abs(d.real) + np.abs(np.exp(1j * d.imag) - 1))))
    return worst


@pytest.mark.parametrize(
    "radii, n, mu",
    [
        (*MATRYOSHKA, [1, 1.3, 1, 1, 1]),
        ([400.0, 402.0], [1.5, 0.1 + 40j, 1.0], [1, 1, 1]),  # 2-nm shell with |n| = 40
        ([30.0, 31.0, 90.0], [1.5, 1.5 + 1e-9, 1.2 - 0.05j, 1.0], [1, 1, 1, 1]),
    ],
)
def test_interface_conditions_hold_at_every_order(radii, n, mu):
    sol = ps.solve(radii, n, WAVELENGTH, mu, l_max=400)
    for pol in (TM, TE):
        for j in range(len(radii)):
            assert _interface_mismatch(sol, np.array(radii), np.array(n), np.array(mu), j, pol) < 1e-9, (pol, j)


def test_batch_equals_single():
    wavelength = np.array([500.0, 600.0, 700.0])
    n = np.array([[1.45, 0.3 + 2.0j + 0.1j * i, 1.33] for i in range(3)])
    batch = ps.solve([40, 50], n, wavelength, l_max=12)
    for i in range(3):
        single = ps.solve([40, 50], n[i], wavelength[i], l_max=12)
        assert np.allclose(batch.a[i], single.a[0], rtol=1e-14) and np.allclose(batch.b[i], single.b[0], rtol=1e-14)
        assert np.allclose(batch.log_a[:, :, i], single.log_a[:, :, 0], rtol=1e-14)


def test_fifty_shells_of_one_material_are_one_sphere():
    radii = np.linspace(2.0, 100.0, 50)
    layered = ps.solve(radii, [2.0 + 0.1j] * 50 + [1.33], 600.0, l_max=30)
    single = ps.solve([100.0], [2.0 + 0.1j, 1.33], 600.0, l_max=30)
    assert np.allclose(layered.a, single.a, rtol=1e-11) and np.allclose(layered.b, single.b, rtol=1e-11)


def test_large_size_parameter_and_extreme_absorption():
    wavelength = 500.0
    radius = 1000 * wavelength / (2 * np.pi)  # x = 1000
    sol = ps.solve([radius], [1.33 + 1e-4j, 1.0], wavelength)
    assert 1.9 < ps.cross_sections(sol).q_ext[0] < 2.3
    sol = ps.solve([5000.0, 5010.0], [1.5, 0.3 + 11j, 1.0], wavelength, l_max=200)  # Im x ~ 700
    for field in (sol.log_a, sol.log_b, sol.log_b_out):
        finite = np.isfinite(field) | (field.real == -np.inf)
        assert finite.all()


def test_default_truncation_uses_the_shortest_wavelength():
    sol = ps.solve([100.0], [1.5, 1.0], np.array([400.0, 800.0]))
    assert sol.orders.size == ps.truncation_order(100.0, 1.0, 400.0)


@pytest.mark.parametrize(
    "radii, n",
    [([50, 40], [1.5, 1.4, 1.0]), ([50], [1.5, 1.4, 1.0]), ([50], [0, 1.0]), ([50], [np.nan, 1.0]), ([-1], [1.5, 1.0])],
)
def test_input_validation(radii, n):
    with pytest.raises(ValueError):
        ps.solve(radii, n, 500.0, l_max=5)
