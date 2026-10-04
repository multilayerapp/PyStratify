"""The normalized core (``normalized_terms``) and the frequency shift against extended precision."""

import mpmath as mp
import numpy as np
import pytest

import pystratify as ps
from pystratify import TE, TM
from pystratify.normalized import _auxiliary_real, auxiliary
from pystratify.references import layered_green_forms, layered_green_sums

AU = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm
AU690 = 0.0992 + 3.9183j  # the same at 690 nm

CASES = {  # radii, n, mu, wavelength, r, digits of the reference
    "nanoshell, host, d = 1 nm": ([50.0, 55.0], [1.45, AU, 1.33], None, 614.0, 56.0, 140),
    "nanoshell, core, d = 0.5 nm": ([50.0, 55.0], [1.45, AU, 1.33], None, 614.0, 49.5, 260),
    "lossless sphere, d = 2 nm": ([250.0], [3.87, 1.0], None, 633.0, 252.0, 60),
    "matryoshka": ([10.0, 13.0, 36.0, 48.0], [1.45, AU690, 1.45, AU690, 1.33], None, 690.0, 35.0, 280),
    "between gold and a magnetic shell": ([40.0, 45.0, 60.0], [1.45, AU, 2.0, 1.33], [1, 1, 2.0, 1], 614.0, 50.0, 200),
}


@pytest.mark.parametrize("x", [0.01, 0.7, 3.1, np.pi, 2 * np.pi, float(mp.besseljzero(1.5, 1)), 12.3, 250.0])
def test_auxiliary_at_many_arguments_equals_scalar(x):
    """The vectorized emitter side runs the scalar recurrences operation for operation."""
    xs = np.array([x, 0.5 * x, 3 * x])
    many = _auxiliary_real(xs, 300)
    for i, xi in enumerate(xs):
        for a, b in zip(auxiliary(xi + 0j, 300), (m[i] for m in many)):
            assert np.array_equal(a, b, equal_nan=True)


def test_terms_at_many_radii_equal_single_radius():
    radii, n, lam = [40.0, 100.0], [1.45, 3.5, 1.33], 700.0
    r = np.array([10.0, 39.0, 41.0, 60.0, 99.5, 100.5, 150.0])  # core, shell and host
    many = ps.normalized_terms(radii, n, lam, r, 120)
    assert list(many.shell) == [0, 0, 1, 1, 1, 2, 2]
    for i, ri in enumerate(r):
        one = ps.normalized_terms(radii, n, lam, ri, 120)
        for name in ("P", "A", "B", "rho", "sigma", "S", "Sm", "Sd", "F", "Fd"):
            assert np.array_equal(getattr(many, name)[:, i], getattr(one, name)[:, 0], equal_nan=True), name


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_shift_against_extended_precision(case, dipole):
    """Rate and shift with the automatic truncation against the complex sums of mpmath transfer matrices."""
    radii, n, mu, lam, r, dps = case
    res = ps.normalized_decay_rates(radii, n, lam, r, None, mu, dipole)
    assert res.converged and res.route == "normalized"
    ref = np.array(layered_green_sums(radii, n, lam, r, res.orders, dps, mu=mu, dipole=dipole))
    assert np.all(np.abs(res.total - (1 + ref.real)) <= 1e-13 * res.total)
    assert np.all(np.abs(res.shift - ref.imag / 2) <= 1e-13 * np.abs(1 + ref))


@pytest.mark.parametrize("case", [CASES[k] for k in list(CASES)[:3]], ids=list(CASES)[:3])
def test_green_forms_per_order(case):
    """P S, P S^m and P S^d of every order against mpmath, TM and TE (errors summed over orders)."""
    radii, n, mu, lam, r, dps = case
    L = 300
    t = ps.normalized_terms(radii, n, lam, r, L, mu)
    refs = layered_green_forms(radii, n, lam, r, L, dps, mu)
    w = 2 * np.arange(1, L + 1) + 1
    for name, ref in zip(("S", "Sm", "Sd"), refs):
        mine = t.P[:, 0] * getattr(t, name)[:, 0]
        for p in (TM, TE):
            assert np.sum(w * np.abs(mine[p] - ref[p])) <= 1e-13 * np.sum(w * np.abs(ref[p])), (name, p)


def test_shift_vanishes_without_scattering():
    res = ps.normalized_decay_rates([30.0, 60.0], [1.4, 1.4, 1.4], 600.0, 45.0, 40)
    assert np.allclose(res.total, 1, atol=1e-13) and np.all(res.shift == 0)


def test_shift_quasistatic_limit_and_signs():
    """0.2 nm from gold the ED shift over the nonradiative rate tends to -Re r/(2 Im r),
    r = (eps - eps_h)/(eps + eps_h), for both orientations (red shift); the radial MD is blue-shifted."""
    eps, eps_h = AU**2, 1.33**2
    ratio = (eps - eps_h) / (eps + eps_h)
    limit = -ratio.real / (2 * ratio.imag)
    ed = ps.normalized_decay_rates([50.0, 55.0], [1.45, AU, 1.33], 614.0, 55.2, None, None, "electric")
    assert np.all(np.abs(ed.shift / ed.nonradiative / limit - 1) < 0.01)
    assert np.all(ed.shift < 0)
    md = ps.normalized_decay_rates([50.0, 55.0], [1.45, AU, 1.33], 614.0, 55.2, None, None, "magnetic")
    assert md.shift[0] > 0


@pytest.mark.parametrize("case", [CASES[k] for k in list(CASES)[1:3]], ids=list(CASES)[1:3])
def test_automatic_truncation_has_converged(case):
    """Doubling the automatic L changes neither the rate nor the reactive tail that makes the shift."""
    radii, n, mu, lam, r, _ = case
    for dipole in ("electric", "magnetic"):
        auto = ps.normalized_decay_rates(radii, n, lam, r, None, mu, dipole)
        twice = ps.normalized_decay_rates(radii, n, lam, r, 2 * auto.orders, mu, dipole)
        scale = np.abs(auto.total + 2j * auto.shift)
        assert np.all(np.abs(twice.total - auto.total) <= 1e-13 * auto.total)
        assert np.all(np.abs(twice.shift - auto.shift) <= 1e-13 * scale)


def test_dipole_series_of_terms():
    radii, n, mu, lam, r, _ = CASES["between gold and a magnetic shell"]
    t = ps.normalized_terms(radii, n, lam, r, 300, mu)
    for dipole in ("electric", "magnetic"):
        g, rad = t.dipole_series(dipole)
        res = ps.normalized_decay_rates(radii, n, lam, r, 300, mu, dipole)
        assert np.allclose(1 + g[0].real.sum(axis=0), res.total, rtol=1e-15, atol=0)
        assert np.allclose(g[0].imag.sum(axis=0) / 2, res.shift, rtol=1e-15, atol=0)
        assert np.allclose(rad[0].sum(axis=0), res.radiative, rtol=1e-15, atol=0)
