"""Log-form Riccati-Bessel functions against 50-digit mpmath."""

import mpmath as mp
import numpy as np
import pytest

from pystratify.riccati import log_derivatives, log_riccati

mp.mp.dps = 50


def _ref(n, z):
    z = mp.mpc(z.real, z.imag)
    f = z * mp.sqrt(mp.pi / (2 * z))
    return mp.log(f * mp.besselj(n + 0.5, z)), mp.log(f * mp.hankel1(n + 0.5, z))


def _err(ref, got):
    d = complex(ref) - got
    return abs(d.real) + abs(np.exp(1j * d.imag) - 1)  # relative error of the value


ARGS = [1e-3, 0.5, 3.0, 30.0, 150 + 2j, 0.4 + 3j, 5 + 60j, 1.47 + 0.046j, 0.05 + 0.02j, 2.0 - 0.5j, 40 - 3j]


@pytest.mark.parametrize("z", ARGS)
def test_log_riccati_all_orders(z):
    nmax = 300
    lp, lx = log_riccati(np.array([z]), nmax)
    for n in list(range(0, 12)) + list(range(12, nmax + 1, 7)):
        rp, rx = _ref(n, z)
        assert _err(rp, lp[0, n]) < 1e-11, (z, n)
        assert _err(rx, lx[0, n]) < 1e-11, (z, n)


def test_values_where_representable_match_direct():
    z = np.array([0.7, 4.0 + 0.3j])
    lp, lx = log_riccati(z, 20)
    from pystratify.legacy import ric_h, ric_j

    n = np.arange(21)
    assert np.allclose(np.exp(lp), ric_j(n, z[:, None]), rtol=1e-12)
    assert np.allclose(np.exp(lx), ric_h(n, z[:, None]), rtol=1e-12)


def test_extreme_orders_stay_finite():
    lp, lx = log_riccati(np.array([0.01, 1.0 + 0.5j]), 2000)
    assert np.all(np.isfinite(lp)) and np.all(np.isfinite(lx))
    # psi_n ~ z^(n+1)/(2n+1)!!, xi_n ~ -i (2n-1)!!/z^n at large n
    n = 2000
    z = 0.01
    lpsi_asym = (n + 1) * np.log(z) - mp.log(mp.fac2(2 * n + 1))
    assert lp[0, n].real == pytest.approx(float(lpsi_asym), rel=1e-6)


def test_log_derivatives_and_wronskian():
    z = np.array([2.0 + 0.1j, 9.0])
    lp, lx = log_riccati(z, 40)
    l = np.arange(1, 41)
    d1, d3 = log_derivatives(lp, lx, z[:, None][:, 0], l)
    # W[psi, xi] = psi xi (D3 - D1) = i
    w = np.exp(lp[:, l] + lx[:, l]) * (d3 - d1)
    assert np.allclose(w, 1j, atol=1e-10)


def test_zero_argument_rejected():
    with pytest.raises(ValueError):
        log_riccati(np.array([0.0]), 5)
