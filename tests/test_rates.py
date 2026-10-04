"""Decay rates of electric, magnetic and chiral sources in or near (chiral) multilayers.

Checked against: :func:`~pystratify.decay_rates` (achiral layers, radial and
tangential dipoles); the far field of :func:`~pystratify.dipole_far_field`
(reciprocity - an independent route) for lossless chiral multilayers, where
total = radiated, per helicity; energy conservation total = radiative +
absorbed with absorbing, magnetic and chiral layers (three independent
computations); mirror symmetry (kappa, m) -> (-kappa, -m); exact orientation
averages over finite rotation groups; the closed-form free power in a chiral
medium.
"""

import itertools

import warnings

import numpy as np
import pytest

import pystratify as ps

LAM = 614.0
AU = 0.27 + 2.93j
RADII = [50.0, 70.0, 90.0]
SOURCES = [
    ([0, 0, 1.0], None),
    ([1.0, 0, 0], None),
    ([0, 0, 1.0], [0, 0, 0.3j]),
    ([0.3, -0.5, 0.8], [0.2j, 0.1, -0.4]),
    ([1.0, 1j, 0], [0.1, -0.2j, 0.05]),
]
LOSSLESS = ([1.45, 1.8, 1.6, 1.33], [0.05, 0.0, -0.08, 0.0], [1.3, 1.0, 1.7, 1.1])
LOSSY = {
    # circular dichroism (Im kappa) with magnetic loss, so that eps'' mu'' >= kappa''^2 (passive)
    "chiral core, Au": ([1.45 + 0.02j, 1.8, AU, 1.33], [0.05 + 0.01j, 0.1, 0, 0], [1 + 0.003j, 1, 1, 1]),
    "magnetic chiral": (
        [1.5 + 0.1j, 1.6, 2.0 + 0.3j, 1.0],
        [0.1 + 0.02j, -0.2, -0.3 + 0.05j, 0],
        [1.2 + 0.05j, 1.0, 0.9 + 0.1j, 1.0],
    ),
}


def _rotations():
    """The 24 proper rotations of the cube: they average any quadratic form isotropically."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            r = np.zeros((3, 3))
            r[range(3), perm] = signs
            if np.linalg.det(r) > 0:
                out.append(r)
    return out


@pytest.mark.parametrize("r0", [40.0, 80.0, 95.0, 120.0])
@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
def test_achiral_limit_is_decay_rates(r0, dipole):
    radii, n, mu = RADII, [1.45, AU, 1.6, 1.33], [1.3, 1.0, 1.7, 1.1]
    ref = ps.decay_rates(radii, n, LAM, [r0], mu, dipole=dipole, tol=1e-10)
    for j, moment in enumerate(([0, 0, 1.0], [1.0, 0, 0])):
        out = ps.emission_rates(radii, n, LAM, [0, 0, r0], moment, mu=mu, dipole=dipole, tol=1e-10)
        assert out.converged
        assert out.total == pytest.approx(ref.total[0, j], rel=1e-12)
        assert out.radiative == pytest.approx(ref.radiative[0, j], rel=1e-12)
        assert out.nonradiative == pytest.approx(ref.nonradiative[0, j], rel=1e-11)
        assert out.balance_error < 1e-11


@pytest.mark.parametrize(
    "position, rel",
    [([0, 0, 30.0], 1e-12), ([0, 20.0, 50.0], 1e-11), ([60.0, 0, 0], 1e-12), ([0, 0, 89.0], 2e-9),
     ([30, 40, 100.0], 1e-12), ([0, 0, 150.0], 1e-12)],
)  # fmt: skip
def test_lossless_chiral_total_equals_far_field(position, rel):
    """With nothing absorbing, the local density of states equals the radiated power (from reciprocity).

    1 nm from a lossless interface the reflected LDOS terms are small real parts of large evanescent
    terms, so rounding limits the total (not the radiative part) to ~1e-9 there."""
    n, kappa, mu = LOSSLESS
    for p, m in SOURCES:
        ff = ps.dipole_far_field(RADII, n, LAM, position, p, 0.0, mu=mu, kappa=kappa, magnetic_moment=m, tol=1e-13)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = ps.emission_rates(RADII, n, LAM, position, p, magnetic_moment=m, mu=mu, kappa=kappa, tol=1e-12)
        # chiral layers take the logarithmic route, which flags the energy balance it cannot keep at 1e-12
        assert out.converged or "energy balance" in out.notes[-1]
        assert out.nonradiative == 0
        assert out.total == pytest.approx(ff.power, rel=rel)
        assert out.radiative == pytest.approx(ff.power, rel=1e-12)
        assert np.allclose(out.radiative_helicity, ff.helicity_power, rtol=0, atol=1e-12 * ff.power)
        assert out.dissymmetry == pytest.approx(ff.dissymmetry, abs=1e-11)


@pytest.mark.parametrize("config", LOSSY)
@pytest.mark.parametrize("r0", [55.0, 60.0, 69.0, 100.0, 300.0])
def test_energy_balance_with_absorbing_chiral_layers(config, r0):
    n, kappa, mu = LOSSY[config]
    for p, m in SOURCES if r0 != 69.0 else SOURCES[3:4]:  # 1 nm from gold: ~1200 orders
        out = ps.emission_rates(RADII, n, LAM, [0, 0, r0], p, magnetic_moment=m, mu=mu, kappa=kappa, tol=1e-10)
        assert out.converged
        assert out.balance_error < 5e-12
        assert out.absorption[1] == 0 and out.absorption[-1] == 0  # the emitter's layer and the host


def test_energy_balance_next_to_an_interface():
    """0.1 nm from absorbing chiral/magnetic layers: l-sums unconverged at l_cap, balance still holds."""
    n, kappa, mu = LOSSY["magnetic chiral"]
    for r0 in (50.1, 69.9, 90.1):
        out = ps.emission_rates(
            RADII, n, LAM, [0, 0, r0], [0.3, -0.5, 0.8], magnetic_moment=[0.2j, 0.1, -0.4], mu=mu, kappa=kappa,
            l_cap=800, warn=False,
        )  # fmt: skip
        assert not out.converged and out.notes
        assert out.balance_error < 1e-10
        assert out.nonradiative / out.total > 0.999


@pytest.mark.parametrize("r0", [60.0, 100.0])
def test_mirror_image(r0):
    """(kappa, p z, m z) and (-kappa, p z, -m z) are mirror images: same rates, helicities exchanged."""
    n, kappa, mu = LOSSY["magnetic chiral"]
    kappa = np.array(kappa)
    a = ps.emission_rates(RADII, n, LAM, [0, 0, r0], [0, 0, 1.0], magnetic_moment=[0, 0, 0.3j], mu=mu, kappa=kappa)
    b = ps.emission_rates(RADII, n, LAM, [0, 0, r0], [0, 0, 1.0], magnetic_moment=[0, 0, -0.3j], mu=mu, kappa=-kappa)
    assert b.total == pytest.approx(a.total, rel=1e-13)
    assert np.allclose(b.absorption, a.absorption, rtol=1e-12, atol=0)
    assert np.allclose(b.radiative_helicity[::-1], a.radiative_helicity, rtol=1e-12, atol=0)
    assert b.dissymmetry == pytest.approx(-a.dissymmetry, abs=1e-12)
    # the enantiomer beside the same particle decays differently
    c = ps.emission_rates(RADII, n, LAM, [0, 0, r0], [0, 0, 1.0], magnetic_moment=[0, 0, -0.3j], mu=mu, kappa=kappa)
    assert abs(c.total / a.total - 1) > 0.1


@pytest.mark.parametrize("position", [[0, 0, 60.0], [10.0, 20.0, 95.0]])
def test_orientation_averages_are_exact(position):
    n, kappa, mu = LOSSY["chiral core, Au"]
    p, m = np.array([0.3, -0.5, 0.8]), np.array([0.2j, 0.1, -0.4])

    def rates(**kw):
        return ps.emission_rates(RADII, n, LAM, position, mu=mu, kappa=kappa, **kw)

    iso = rates(moment=p, magnetic_moment=m, orientation="isotropic")
    fixed = [rates(moment=r @ p, magnetic_moment=r @ m) for r in _rotations()]
    assert iso.total == pytest.approx(np.mean([f.total for f in fixed]), rel=1e-13)
    assert np.allclose(iso.radiative_helicity, np.mean([f.radiative_helicity for f in fixed], axis=0), rtol=1e-13)
    assert np.allclose(iso.absorption, np.mean([f.absorption for f in fixed], axis=0), rtol=1e-12, atol=1e-15)
    axis = np.array([0.3, 0.4, 0.5]) / np.linalg.norm([0.3, 0.4, 0.5])
    turn = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    spins = [np.eye(3) + np.sin(t) * turn + (1 - np.cos(t)) * turn @ turn for t in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    about = rates(moment=p, magnetic_moment=m, orientation=axis)
    fixed = [rates(moment=r @ p, magnetic_moment=r @ m) for r in spins]
    assert about.total == pytest.approx(np.mean([f.total for f in fixed]), rel=1e-13)
    assert np.allclose(about.radiative_helicity, np.mean([f.radiative_helicity for f in fixed], axis=0), rtol=1e-13)


def test_positions_in_one_call():
    n, kappa, mu = LOSSY["chiral core, Au"]
    pos = np.array([[0, 0, r] for r in (55.0, 60.0, 66.0)] + [[3.0, 4.0, 95.0], [0, 0, 150.0]])
    kw = dict(moment=[1.0, 0.5j, 0.2], magnetic_moment=[0.1, 0, 0.2j], mu=mu, kappa=kappa, tol=1e-11)
    batch = ps.emission_rates(RADII, n, LAM, pos, **kw)
    assert batch.total.shape == (5,) and batch.absorption.shape == (5, 4) and batch.radiative_helicity.shape == (5, 2)
    assert list(batch.shell) == [1, 1, 1, 3, 3]
    for i, v in enumerate(pos):
        # the orders integrated per absorbing layer follow the closest emitter: equal to tol, not bitwise
        one = ps.emission_rates(RADII, n, LAM, v, l_max=batch.orders_used, **kw)
        assert one.total == pytest.approx(batch.total[i], rel=1e-13)
        assert one.radiative == pytest.approx(batch.radiative[i], rel=1e-13)
        assert np.abs(one.absorption - batch.absorption[i]).max() < 1e-10 * batch.total[i]


def test_free_power_in_a_chiral_layer_and_normalisations():
    """free_in_layer = sum_s (Z_d k_s^2 / Z_h k_h^2) |p + i s m / Z_d|^2 / 2 / (|p|^2 + |m|^2 / Z_h^2)."""
    n, kappa, mu = LOSSLESS
    p, m = np.array([0.3, -0.5, 0.8]), np.array([0.2j, 0.1, -0.4])
    host = ps.emission_rates(RADII, n, LAM, [0, 0, 30.0], p, magnetic_moment=m, mu=mu, kappa=kappa)
    layer = ps.emission_rates(
        RADII, n, LAM, [0, 0, 30.0], p, magnetic_moment=m, mu=mu, kappa=kappa, normalization="layer"
    )
    zd, zh = mu[0] / n[0], mu[-1] / n[-1]
    ks = np.array([n[0] + kappa[0], n[0] - kappa[0]])
    q = [p + 1j * s * m / zd for s in (1, -1)]
    free = sum(0.5 * zd * k**2 / (zh * n[-1] ** 2) * np.vdot(v, v).real for k, v in zip(ks, q))
    assert host.free_in_layer == pytest.approx(free / (np.vdot(p, p).real + np.vdot(m, m).real / zh**2), rel=1e-14)
    assert layer.free_in_layer == pytest.approx(host.free_in_layer, rel=1e-15)
    assert layer.total == pytest.approx(host.total / host.free_in_layer, rel=1e-14)
    assert np.allclose(layer.radiative_helicity, host.radiative_helicity / host.free_in_layer, rtol=1e-14)
    assert layer.quantum_yield() == pytest.approx(host.quantum_yield(), rel=1e-14) == pytest.approx(1.0)
    assert host.quantum_yield(0.5) == pytest.approx(host.radiative / (host.total + host.free_in_layer), rel=1e-14)


def test_free_chiral_emitter_in_the_host():
    """Far from a small particle: rates -> 1, g_lum -> 4 Im(p . m'*) / (|p|^2 + |m'|^2), m' = m n_h / mu_h."""
    p, m = np.array([0, 0, 1.0]), np.array([0, 0, 0.2j])
    out = ps.emission_rates([5.0], [1.5, 1.33], LAM, [0, 0, 5000.0], p, magnetic_moment=m, mu=[1, 1.1])
    mm = m * 1.33 / 1.1
    g = 4 * np.imag(np.vdot(mm, p)) / (np.vdot(p, p).real + np.vdot(mm, mm).real)
    assert out.total == pytest.approx(1.0, abs=1e-7) and out.radiative == pytest.approx(1.0, abs=1e-7)
    assert out.dissymmetry == pytest.approx(g, abs=1e-7)


def test_rates_input_validation():
    n, kappa, mu = LOSSY["chiral core, Au"]
    with pytest.raises(ValueError, match="absorbing"):
        ps.emission_rates(RADII, n, LAM, [0, 0, 80.0], [0, 0, 1.0], mu=mu, kappa=kappa)  # inside gold
    with pytest.raises(ValueError, match="absorbing"):
        ps.emission_rates(RADII, n, LAM, [0, 0, 10.0], [0, 0, 1.0], mu=mu, kappa=kappa)  # lossy chiral core
    with pytest.raises(ValueError, match="host"):
        ps.emission_rates(RADII, n[:-1] + [1.33 + 0.01j], LAM, [0, 0, 100.0], [0, 0, 1.0])
    with pytest.raises(ValueError, match="host"):
        ps.emission_rates(RADII, n, LAM, [0, 0, 100.0], [0, 0, 1.0], kappa=[0, 0, 0, 0.1])
    with pytest.raises(ValueError, match="normalization"):
        ps.emission_rates(RADII, n, LAM, [0, 0, 100.0], [0, 0, 1.0], normalization="shell")
    with pytest.raises(ValueError, match="nonzero"):
        ps.emission_rates(RADII, n, LAM, [0, 0, 100.0], [0, 0, 0])
    with pytest.raises(ValueError, match="magnetic_moment"):
        ps.emission_rates(RADII, n, LAM, [0, 0, 100.0], [0, 0, 1.0], magnetic_moment=[1, 0, 0], dipole="magnetic")
    with pytest.raises(ValueError, match="shape"):
        ps.emission_rates(RADII, n, LAM, [[0, 100.0]], [0, 0, 1.0])
    assert not ps.emission_rates(RADII, n, LAM, [0, 0, 100.0], [0, 0, 1.0], mu=mu, kappa=kappa).notes
    with pytest.warns(RuntimeWarning, match="not passive"):  # kappa'' without magnetic loss
        out = ps.emission_rates(RADII, n, LAM, [0, 0, 100.0], [0, 0, 1.0], kappa=kappa)
    assert any("not passive" in note for note in out.notes)


@pytest.mark.parametrize(
    "eps, mu, chi",
    [
        (6.0, 1.0, 0.1),
        (4.0 + 0.1j, 1.0, 0.2),
        (-4.0 + 0.3j, 1.0, 0.1),
        (-4.0 + 0.1j, -1.11 + 0.05j, 0.2),
        (-2.5 + 0.1j, -1.6, 0.2),
    ],
)
def test_chiral_molecule_near_small_chiral_sphere_guzatov_klimov(eps, mu, chi):
    """Guzatov & Klimov, New J. Phys. 14, 123009 (2012), Eq. 46: orientation-averaged radiative rate of a
    molecule (d0, m = -i xi d0), xi = +-0.1 (the two enantiomers), at r0 = 2a from a small Drude-Born-Fedorov
    sphere (their Eq. 48 polarisabilities; Pasteur form as in test_chiral).  Agreement to O((k0 a)^2)."""
    wavelength = 600.0
    k = 2 * np.pi / wavelength
    n2 = eps * mu
    s = 1 - n2 * chi**2
    n_p = np.sqrt(n2 + 0j) / s
    n_p = -n_p if n_p.imag < 0 else n_p
    den = (eps + 2) * (mu + 2) - 4 * chi**2 * eps * mu
    for ka, tol in ((0.01, 3e-4), (0.005, 8e-5)):
        a = ka / k
        r0 = 2 * a
        a_ee = a**3 * ((eps - 1) * (mu + 2) + 2 * chi**2 * eps * mu) / den
        a_hh = a**3 * ((mu - 1) * (eps + 2) + 2 * chi**2 * eps * mu) / den
        a_eh = a**3 * 3j * chi * eps * mu / den
        for xi in (0.1, -0.1):
            eq46 = 1 + xi**2 + 2 / r0**6 * (abs(a_ee - 1j * xi * a_eh) ** 2 + abs(-a_eh - 1j * xi * a_hh) ** 2)
            out = ps.emission_rates(
                [a], [n_p, 1.0], wavelength, [0, 0, r0], [0, 0, 1.0], magnetic_moment=[0, 0, -1j * xi],
                mu=[mu / s, 1.0], kappa=[n2 * chi / s, 0], orientation="isotropic", tol=1e-10,
            )  # fmt: skip
            assert out.radiative == pytest.approx(eq46 / (1 + xi**2), rel=tol)
