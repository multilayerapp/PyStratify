"""Log-form Riccati-Bessel functions against extended-precision mpmath."""

import numpy as np
import pytest

from pystratify.riccati import ANCHOR_MARGIN, log_derivatives, log_riccati

from .mp_reference import log_psi_xi


def relative_error(reference, got):
    d = reference - got
    return abs(d.real) + abs(np.exp(1j * d.imag) - 1)


def tolerance(reference):
    """1e-11, widened by the float spacing of the logarithm itself (|log| ~ 2e4 at l = 1500)."""
    return 1e-11 + 1e-14 * abs(reference)


ARGUMENTS = [
    1e-3,
    0.5,
    3.0,
    30.0,
    150 + 2j,
    0.4 + 3j,
    5 + 60j,
    1.47 + 0.046j,
    0.05 + 0.02j,
    12 + 125j,  # thick metal shell
    50 + 700j,  # extreme absorber: J and Y cancel across 1400 e-folds in H^(1)
    3000.0,  # large dielectric sphere
    2.0 - 1.5j,  # gain media
    800 - 2j,
    30 - 0.5j,
]


@pytest.mark.parametrize("z", ARGUMENTS)
def test_every_regime_to_1e_11(z):
    nmax = 1500
    lp, lx = log_riccati(np.array([z]), nmax)
    assert np.all(np.isfinite(lp)) and np.all(np.isfinite(lx))
    turning = int(abs(z))
    orders = {0, 1, 2, 7, 60, 300, nmax}
    orders |= {
        o for o in (turning - 1, turning, turning + ANCHOR_MARGIN, turning + ANCHOR_MARGIN + 1) if 0 <= o <= nmax
    }
    for n in sorted(orders):
        try:
            ref_psi, ref_xi = log_psi_xi(n, z)
        except ValueError:  # mpmath cannot resolve psi ~ 10^-15000; covered by test_asymptotic_orders
            continue
        assert relative_error(ref_psi, lp[0, n]) < tolerance(ref_psi), ("psi", z, n)
        assert relative_error(ref_xi, lx[0, n]) < tolerance(ref_xi), ("xi", z, n)


def test_asymptotic_orders():
    lp, lx = log_riccati(np.array([0.01, 1.0 + 0.5j]), 2000)
    assert np.all(np.isfinite(lp)) and np.all(np.isfinite(lx))
    n = 2000  # psi_n(z) ~ z^(n+1) / (2n+1)!!
    log_double_factorial = np.sum(np.log(np.arange(1, 2 * n + 2, 2)))
    assert lp[0, n].real == pytest.approx((n + 1) * np.log(0.01) - log_double_factorial, rel=1e-10)


def test_batched_shapes_and_mixed_magnitudes():
    z = np.array([[0.2, 40 + 1j], [3 - 0.1j, 700 + 5j]])
    lp, lx = log_riccati(z, 900)
    assert lp.shape == lx.shape == (2, 2, 901)
    for idx in np.ndindex(z.shape):
        single_p, single_x = log_riccati(np.array([z[idx]]), 900)
        for batched, single in ((lp[idx], single_p[0]), (lx[idx], single_x[0])):
            d = batched - single  # logs may differ by 2 pi i
            assert np.max(np.abs(d.real) + np.abs(np.exp(1j * d.imag) - 1)) < 1e-10


def test_log_derivatives_and_wronskian():
    z = np.array([2.0 + 0.1j, 9.0, 40 - 0.3j])
    lp, lx = log_riccati(z, 60)
    orders = np.arange(1, 61)
    d_psi, d_xi = log_derivatives(lp, lx, z, orders)
    wronskian = np.exp(lp[:, orders] + lx[:, orders]) * (d_xi - d_psi)  # psi xi' - psi' xi = i
    assert np.allclose(wronskian, 1j, atol=1e-10)


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_rejects_unusable_arguments(bad):
    with pytest.raises(ValueError):
        log_riccati(np.array([bad]), 5)


@pytest.mark.parametrize("z", [1500 + 1500j, 30000 + 10000j, 2000 + 80000j])
def test_large_complex_arguments_below_the_turning_point(z):
    """Off the real axis psi_n has no zeros, so an AMOS zero is underflow: scaled
    jve underflows long before n = |z| once |Im z| >~ 1000 (at 1500 + 1500i from
    n = 1993), which made large absorbing spheres NaN.  psi comes from the downward
    recurrence and xi from AMOS/upward - independent - so the Wronskian checks both
    (mpmath agrees to 1e-12 at 1500 + 1500i, n = 1000..2200; too slow for the suite)."""
    nmax = int(abs(z)) + 100
    lp, lx = log_riccati(np.array([z]), nmax)
    assert np.all(np.isfinite(lp)) and np.all(np.isfinite(lx))
    n = np.arange(1, nmax + 1)
    # psi_n xi_{n-1} - psi_{n-1} xi_n = i, relative to its first term
    big = lp[0, n] + lx[0, n - 1]
    wronskian = 1 - np.exp(lp[0, n - 1] + lx[0, n] - big)
    target = 1j * np.exp(-big)
    # the logs are ~|z| in size, so their float spacing sets the attainable accuracy
    assert np.max(np.abs(wronskian - target) / (1 + np.abs(target))) < 1e-12 + 3e-14 * abs(z)
