"""The normalized route of ``decay_rates`` and ``emission_rates`` (the default) against the logarithmic route,
energy conservation next to lossless interfaces, and the frequency shift of general sources."""

import warnings

import numpy as np
import pytest

import pystratify as ps
from pystratify.rates import _SIGN, _functionals, _tetm_functionals

AU = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm
SHELL = ([40.0, 100.0], [1.45, 3.5, 1.33], 650.0)  # lossless high-index shell in water (Sec. 7 of the paper)


def _traceless(q):
    q = (q + q.T) / 2
    return q - np.trace(q) / 3 * np.eye(3)


RNG = np.random.default_rng(11)
Q_RANDOM = _traceless(RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3)))
SOURCES = {
    "ED radial": dict(moment=[0, 0, 1.0]),
    "ED tangential": dict(moment=[1.0, 0, 0]),
    "MD radial": dict(moment=[0, 0, 1.0], dipole="magnetic"),
    "MD tangential": dict(moment=[0, 1.0, 0], dipole="magnetic"),
    "chiral p || m": dict(moment=[0, 0.6, 0.8], magnetic_moment=[0, 0.18j, 0.24j]),
    "general p + m": dict(moment=[0.3 + 0.1j, -0.2, 0.5j], magnetic_moment=[0.1, 0.4 - 0.2j, -0.3]),
    "EQ zz": dict(moment=[0, 0, 0], quadrupole=np.diag([-0.5, -0.5, 1.0])),
    "EQ xz": dict(moment=[0, 0, 0], quadrupole=np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])),
    "EQ xy": dict(moment=[0, 0, 0], quadrupole=np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])),
    "EQ xx - yy": dict(moment=[0, 0, 0], quadrupole=np.diag([1.0, -1.0, 0])),
    "EQ + p + m": dict(moment=[0.3, 0.1j, 0.2], magnetic_moment=[0.1, 0, 0.2j], quadrupole=Q_RANDOM),
    "MQ xz": dict(moment=[0, 0, 0], magnetic_quadrupole=np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])),
    "EQ + MQ + p + m": dict(
        moment=[0.2, 0, 0.1j], magnetic_moment=[0, 0.3, 0], quadrupole=Q_RANDOM, magnetic_quadrupole=1j * Q_RANDOM.T
    ),
    "chiral, isotropic": dict(moment=[0, 0, 1.0], magnetic_moment=[0, 0, 0.3j], orientation="isotropic"),
    "EQ + p, isotropic": dict(moment=[0, 0, 0.5], quadrupole=Q_RANDOM, orientation="isotropic"),
}


def test_tetm_functionals_reproduce_the_helicity_functionals():
    """The TM (odd in s) and TE (even in s) parts recombine to the functionals of W_s = M + s N."""
    rng = np.random.default_rng(3)
    P, K, L = 3, 2, 40
    l = np.arange(1, L + 1)
    x, kd, zd = rng.uniform(0.5, 9, P), rng.uniform(0.01, 0.05, P), 0.7
    src = rng.normal(size=(P, K, 24)) + 1j * rng.normal(size=(P, K, 24))
    for i in range(P):
        for k in range(K):
            for a in (6, 15):
                src[i, k, a : a + 9] = _traceless(src[i, k, a : a + 9].reshape(3, 3)).ravel()
    dd = rng.normal(size=(P, L)) + 1j * rng.normal(size=(P, L))
    for size in (6, 15, 24):  # dipoles; with an electric quadrupole; with a magnetic one as well
        s = src[:, :, :size]
        quadrupole = size > 6
        cv, cd = _tetm_functionals(x, kd, zd, s, quadrupole, l)
        for i in range(P):
            q = s[i : i + 1, :, None, :3] + (1j * _SIGN / zd)[None, None, :, None] * s[i : i + 1, :, None, 3:6]
            quad = None
            if quadrupole:  # Q_s = Q_e + i s Q_m / Z
                quad = np.repeat(s[i : i + 1, :, None, 6:15], 2, axis=2)
                if size == 24:
                    quad = quad + (1j * _SIGN / zd)[None, None, :, None] * s[i : i + 1, :, None, 15:24]
                quad = quad.reshape(1, K, 2, 3, 3)
            ll = (l * (l + 1)).astype(float)
            ref = _functionals(
                np.repeat(dd[i : i + 1, :, None], 2, axis=2), np.array([[x[i]] * 2]), np.array([kd[i]] * 2), q, quad, ll
            )
            r = (l + 1) / x[i] - dd[i]  # f'/f = (l+1)/x - r
            te = cv[i, ..., 1] - cd[i, ..., 1] * r[None, :, None]
            tm = cv[i, ..., 0] - cd[i, ..., 0] * r[None, :, None]
            for c, s_ in enumerate(_SIGN):
                assert np.allclose(te + s_ * tm, ref[0, ..., c], rtol=1e-13, atol=1e-13 * np.abs(ref).max())


@pytest.mark.parametrize("source", SOURCES.values(), ids=SOURCES.keys())
@pytest.mark.parametrize(
    "structure",
    [
        ([50.0, 55.0], [1.45, AU, 1.33], 614.0, [0, 0, 60.0]),  # 5 nm outside the gold nanoshell
        ([50.0, 55.0], [1.45, AU, 1.33], 614.0, [0, 0, 45.0]),  # 5 nm inside its core
        ([100.0], [3.7 + 0.005j, 1.0], 800.0, [30.0, 0, 108.0]),  # 10 nm from a weakly absorbing Si sphere
    ],
    ids=["nanoshell host", "nanoshell core", "silicon sphere"],
)
def test_normalized_route_agrees_with_logarithmic_route(structure, source):
    """Where the logarithmic route is accurate (its energy balance is), the two agree in every rate."""
    radii, n, lam, position = structure
    a = ps.emission_rates(radii, n, lam, position, tol=1e-12, **source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b = ps.emission_rates(radii, n, lam, position, tol=1e-12, route="log", **source)
    assert a.route == "normalized" and b.route == "log" and a.converged
    assert np.isnan(b.shift)
    assert abs(a.radiative / b.radiative - 1) < 2e-14
    assert np.allclose(a.radiative_helicity, b.radiative_helicity, rtol=3e-14, atol=0)
    assert np.allclose(a.absorption, b.absorption, rtol=1e-15, atol=0)
    if b.balance_error < 1e-13:
        assert abs(a.total / b.total - 1) < 3e-13
    assert a.balance_error < 2e-13


@pytest.mark.parametrize("d", [1.0, 0.1, 0.05])
def test_energy_balance_next_to_lossless_interfaces(d):
    """Every source, 1 to 0.05 nm from the interfaces of a lossless shell, inside and outside:
    total = radiative.  (The logarithmic route misses this by up to 27 % for quadrupoles.)"""
    names = list(SOURCES) if d > 0.05 else ["ED radial", "MD tangential", "chiral p || m", "EQ xz", "EQ + MQ + p + m"]
    for position in ([0, 0, 100.0 + d], [0, 0, 40.0 - d]):
        for name in names:
            source = SOURCES[name]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # quadrupoles 0.05 nm out need > l_cap orders
                out = ps.emission_rates(*SHELL, position, tol=1e-13, l_cap=40000, **source)
            # in the subwavelength core the radiative part of the low-order reflected field is ~1e-3 (dipole)
            # to ~1e-4 (quadrupole) of its reactive part and is carried to ~1e-16 of the latter
            assert abs(out.total / out.radiative - 1) < 1.5e-13, (name, position)


def test_logarithmic_route_flags_its_energy_balance():
    """A quadrupole 0.1 nm from a lossless interface: the log route's l-sums look converged, the balance does not."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log = ps.emission_rates(*SHELL, [0, 0, 100.1], [0, 0, 0], quadrupole=np.diag([1.0, -1.0, 0]), route="log")
    assert not log.converged and log.balance_error > 1e-3 and "energy balance" in log.notes[-1]
    norm = ps.emission_rates(*SHELL, [0, 0, 100.1], [0, 0, 0], quadrupole=np.diag([1.0, -1.0, 0]))
    assert norm.converged and norm.balance_error < 1e-13


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
def test_dipole_shift_of_emission_rates_equals_decay_rates(dipole):
    radii, n, lam, r = [50.0, 55.0], [1.45, AU, 1.33], 614.0, 60.0
    rates = ps.decay_rates(radii, n, lam, [r], dipole=dipole, tol=1e-12)
    assert rates.route == "normalized"
    for column, moment in ((0, [0, 0, 1.0]), (1, [1.0, 0, 0])):
        out = ps.emission_rates(radii, n, lam, [0, 0, r], moment, dipole=dipole, tol=1e-12)
        assert np.isclose(out.total, rates.total[0, column], rtol=1e-13, atol=0)
        assert np.isclose(out.shift, rates.shift[0, column], rtol=1e-13, atol=1e-13 * out.total)


def test_shift_of_a_chiral_emitter_is_the_weighted_sum():
    """For p || m near an achiral multilayer the shift, like the rates, has no p-m interference."""
    radii, n, lam, position = [50.0, 55.0], [1.45, AU, 1.33], 614.0, [0, 0, 60.0]
    p, m = np.array([0, 0.6, 0.8]), 0.3j * np.array([0, 0.6, 0.8])
    both = ps.emission_rates(radii, n, lam, position, p, magnetic_moment=m, normalization="layer", tol=1e-13)
    e = ps.emission_rates(radii, n, lam, position, p, normalization="layer", tol=1e-13)
    h = ps.emission_rates(radii, n, lam, position, m, dipole="magnetic", normalization="layer", tol=1e-13)
    w = np.linalg.norm(m) ** 2 * 1.33**2  # |m~|^2 = |m / Z|^2 in units of |p|^2, Z = mu/n = 1/1.33 in water
    for key in ("total", "shift"):
        weighted = (getattr(e, key) + w * getattr(h, key)) / (1 + w)
        assert np.isclose(getattr(both, key), weighted, rtol=1e-13, atol=1e-13 * both.total), key


def test_decay_rates_routes():
    radii, n, lam = [50.0, 55.0], [1.45, AU, 1.33], 614.0
    norm = ps.decay_rates(radii, n, lam, [49.0, 56.0, 60.0], tol=1e-12)
    log = ps.decay_rates(radii, n, lam, [49.0, 56.0, 60.0], tol=1e-12, route="log")
    assert norm.route == "normalized" and log.route == "log" and np.all(np.isnan(log.shift))
    assert np.allclose(norm.total, log.total, rtol=1e-10, atol=0)
    assert np.allclose(norm.radiative, log.radiative, rtol=1e-12, atol=0)
    assert np.array_equal(norm.nonradiative, log.nonradiative) or np.allclose(
        norm.nonradiative, log.nonradiative, rtol=1e-12
    )
    assert norm.balance_error.max() < 1e-12
    with pytest.raises(ValueError):
        ps.decay_rates(radii, n, lam, [60.0], route="nope")
    with pytest.raises(ValueError):
        ps.decay_rates(radii, n, lam, [60.0], route="normalized", sheets={1: ps.Sheet(conductivity=0.01)})


def test_decay_rates_next_to_a_lossless_sphere():
    """0.25 nm from a lossless sphere the normalized route conserves energy; the log route loses ~1e-8."""
    out = ps.decay_rates([250.0], [3.87, 1.0], 633.0, [250.25], tol=1e-13, l_cap=40000)
    assert out.converged.all() and np.all(np.abs(out.total / out.radiative - 1) < 1e-13)
    assert np.all(out.shift < 0)  # an electric dipole next to a dielectric is red-shifted


@pytest.mark.parametrize(
    "radii, n, positions",
    [
        ([40.0, 50.0], [2.0, 3.0, 1.0], ([0, 0, 20.0], [10.0, 0, 45.0], [0, 30.0, 55.0])),
        ([40.0, 50.0], [2 + 0.3j, 3.0, 1.0], ([10.0, 0, 45.0], [0, 30.0, 55.0])),
    ],
    ids=["dual lossless", "dual lossy"],
)
def test_magnetic_quadrupole_by_duality(radii, n, positions):
    """In a dual structure (eps = mu in every layer and the host) Q_m radiates and decays as Q_e = Q_m."""
    for position in positions:
        kw = dict(mu=n, tol=1e-13)
        e = ps.emission_rates(radii, n, 600.0, position, [0, 0, 0], quadrupole=Q_RANDOM, **kw)
        m = ps.emission_rates(radii, n, 600.0, position, [0, 0, 0], magnetic_quadrupole=Q_RANDOM, **kw)
        for key in ("total", "radiative", "shift"):
            assert np.isclose(getattr(m, key), getattr(e, key), rtol=1e-13, atol=0), key
        assert np.allclose(m.radiative_helicity, e.radiative_helicity, rtol=1e-13, atol=0)
        assert np.allclose(m.absorption, e.absorption, rtol=1e-13, atol=0)


def test_magnetic_quadrupole_free_and_validated():
    free = ps.emission_rates([40.0, 100.0], [1.33] * 3, 650.0, [0, 0, 70.0], [0, 0, 0], magnetic_quadrupole=Q_RANDOM)
    assert abs(free.total - 1) < 1e-14 and abs(free.radiative - 1) < 1e-14
    with pytest.raises(ValueError):
        ps.emission_rates(
            [40.0], [1.5, 1.0], 650.0, [0, 0, 70.0], [0, 0, 0], magnetic_quadrupole=np.triu(np.ones((3, 3)))
        )
