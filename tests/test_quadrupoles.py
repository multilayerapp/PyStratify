"""Electric-quadrupole sources and current-loop magnetic moments.

Checked against: finite differences (40 digits) of the Bohren-Huffman vector
spherical wave functions, for the quadrupole coupling (1/6) Q : grad W at the
emitter; Jackson's free quadrupole power; total = radiative for lossless
chiral multilayers and energy balance for absorbing ones; a pair of opposite
dipoles +-p at r0 +- delta/2 computed by the independent reciprocity route
(dipole_far_field), which tends to the quadrupole 3(p d + d p) - 2 (p.d) I
plus the current loop (i k0/2) p x d - in amplitude, near and inside chiral
particles; the Green's-function far field against the reciprocity far field
for dipoles; exact orientation averages.
"""

import itertools

import mpmath as mp
import warnings

import numpy as np
import pytest

import pystratify as ps
from pystratify.rates import _functionals, _icosahedral_rotations

LAM = 614.0
AU = 0.27 + 2.93j
RADII = [40.0, 55.0, 70.0]
CHIRAL = ([1.45, AU, 1.6, 1.33], [0.03, 0.0, 0.08, 0.0], [1.0, 1.0, 1.2, 1.0])


def _random_quadrupole(seed=2, scale=20.0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    q = q + q.T
    return scale * (q - np.trace(q) / 3 * np.eye(3))


def _bh_mode(l, m, parity, kind, k, point):
    """Cartesian M and N of Bohren & Huffman at a point, 40 digits."""
    x, y, z = point
    r = mp.sqrt(x * x + y * y + z * z)
    th, ph = mp.acos(z / r), mp.atan2(y, x)
    rho = k * r

    def zf(order, t):
        j = mp.besselj(order + 0.5, t)
        return mp.sqrt(mp.pi / (2 * t)) * (j if kind == "psi" else j + 1j * mp.bessely(order + 0.5, t))

    zl = zf(l, rho)
    dz = zf(l - 1, rho) - l * zl / rho  # (1/rho) d(rho z_l)/drho

    def legendre(order, t):
        return mp.legenp(order, m, t, type=2) * (-1) ** m if order >= m else 0  # no Condon-Shortley phase

    mu_ = mp.cos(th)
    p_lm = legendre(l, mu_)
    dp = (l * mu_ * p_lm - (l + m) * legendre(l - 1, mu_)) / mp.sin(th)  # d P_l^m / d theta
    cm, sm, st = mp.cos(m * ph), mp.sin(m * ph), mp.sin(th)
    even = parity == "e"
    a, b = (-sm, cm) if even else (cm, sm)  # angular factors of the P/sin and dP/dtheta terms
    m_th, m_ph = m * a * p_lm / st * zl, -b * dp * zl
    n_r = zl / rho * b * l * (l + 1) * p_lm
    n_th, n_ph = b * dp * dz, m * a * p_lm / st * dz
    r_hat = [st * mp.cos(ph), st * mp.sin(ph), mp.cos(th)]
    th_hat = [mp.cos(th) * mp.cos(ph), mp.cos(th) * mp.sin(ph), -st]
    ph_hat = [-mp.sin(ph), mp.cos(ph), 0]
    return (
        [m_th * th_hat[i] + m_ph * ph_hat[i] for i in range(3)],
        [n_r * r_hat[i] + n_th * th_hat[i] + n_ph * ph_hat[i] for i in range(3)],
    )


@pytest.mark.parametrize("l", [2, 5])
def test_quadrupole_coupling_against_finite_differences(l):
    """(1/6) sum Q_ij d_j W_i of W_s = M + s N at (0, 0, r0), all axial families, both radial kinds."""
    mp.mp.dps = 40
    q = _random_quadrupole(scale=1.0)
    k, r0, h = 0.013, 47.0, mp.mpf("1e-9")
    x = k * r0
    from pystratify.riccati import log_riccati

    log_psi, log_xi = log_riccati(np.array([x + 0j]), l + 1)
    families = [(0, "e"), (1, "e"), (1, "o"), (2, "e"), (2, "o")]
    for kind, logs in (("psi", log_psi[0]), ("xi", log_xi[0])):
        dlog = np.exp(logs[l - 1] - logs[l]) - l / x
        values = (
            _functionals(
                np.full((1, 1, 2), dlog),
                np.full((1, 2), x),
                np.full(2, k),
                np.zeros((1, 1, 2, 3)),
                q[None, None],
                np.array([l * (l + 1.0)]),
            )[0, 0, 0]
            * np.exp(logs[l])
            / x
        )  # undo the division by f/x
        for f, (m, parity) in enumerate(families):
            for c, sign in enumerate((1, -1)):
                base = [mp.mpf("1e-14"), mp.mpf("2e-14"), mp.mpf(r0)]
                total = 0
                for j in range(3):
                    plus, minus = list(base), list(base)
                    plus[j] += h
                    minus[j] -= h
                    (mp_, np_), (mm_, nm_) = (_bh_mode(l, m, parity, kind, mp.mpf(k), pt) for pt in (plus, minus))
                    for i in range(3):
                        total += complex(q[i, j]) * ((mp_[i] + sign * np_[i]) - (mm_[i] + sign * nm_[i])) / (2 * h)
                reference = complex(total / 6)
                assert abs(values[f, c] - reference) < 1e-9 * max(abs(reference), 1e-300), (kind, m, parity, sign)


@pytest.mark.parametrize("position", [[0, 0, 30.0], [10.0, -20.0, 25.0], [0, 0, 120.0]])
def test_free_quadrupole_power(position):
    """In a transparent 'particle' the multipole sum reproduces (k^2/120) sum |Q_ij|^2 (Jackson)."""
    q = _random_quadrupole()
    for moment, magnetic in (([0, 0, 0], None), ([0.3, -0.5, 0.8], [0.2j, 0.1, -0.4])):
        out = ps.emission_rates(
            [50.0], [1.33, 1.33], LAM, position, moment, magnetic_moment=magnetic, quadrupole=q, tol=1e-12
        )
        assert out.total == pytest.approx(1.0, abs=1e-13) and out.radiative == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("position", [[0, 0, 30.0], [0, 20.0, 50.0], [60.0, 0, 0], [30, 40, 100.0]])
def test_lossless_chiral_total_equals_radiated_with_quadrupole(position):
    n, kappa, mu = [1.45, 1.8, 1.6, 1.33], [0.05, 0.0, -0.08, 0.0], [1.3, 1.0, 1.7, 1.1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = ps.emission_rates(
            [50.0, 70.0, 90.0], n, LAM, position, [0.3, -0.5, 0.8], magnetic_moment=[0.2j, 0.1, -0.4], mu=mu,
            kappa=kappa, quadrupole=_random_quadrupole(), tol=1e-12,
        )  # fmt: skip
    # chiral layers take the logarithmic route, which flags the energy balance it cannot keep at 1e-12
    assert out.converged or "energy balance" in out.notes[-1]
    assert out.total == pytest.approx(out.radiative, rel=1e-10)


@pytest.mark.parametrize("r0", [55.0, 62.0, 69.0, 100.0])
def test_energy_balance_with_quadrupole(r0):
    radii = [50.0, 70.0, 90.0]
    n, kappa, mu = [1.45 + 0.02j, 1.8, AU, 1.33], [0.05 + 0.01j, 0.1, 0, 0], [1 + 0.003j, 1, 1, 1]
    for orientation in ("fixed", "isotropic", [0.2, 0.5, 1.0]) if r0 != 69.0 else ("fixed",):  # 1 nm from Au
        out = ps.emission_rates(
            radii, n, LAM, [0, 0, r0], [0.3, -0.5, 0.8], magnetic_moment=[0.2j, 0.1, -0.4], mu=mu, kappa=kappa,
            quadrupole=_random_quadrupole(), orientation=orientation, tol=1e-10,
        )  # fmt: skip
        assert out.converged and out.balance_error < 5e-12


@pytest.mark.parametrize("center", [[0, 0, 90.0], [0, 0, 62.0], [10.0, 20.0, 25.0]])
def test_dipole_pair_limit(center):
    """+-p at r0 +- delta u/2 (reciprocity route) -> quadrupole + current loop (Green's route), in amplitude."""
    n, kappa, mu = CHIRAL
    center = np.array(center)
    rng = np.random.default_rng(5)
    theta, phi = rng.uniform(0, np.pi, 7), rng.uniform(-np.pi, np.pi, 7)
    p, u = np.array([0.3, 1.0j, 0.5]), np.array([0.6, -0.3, 0.74]) / np.linalg.norm([0.6, -0.3, 0.74])
    amplitudes = []
    for delta in (0.2, 0.1):
        plus, minus = (
            ps.dipole_far_field(RADII, n, LAM, center + s * delta * u / 2, p, theta, phi, mu=mu, kappa=kappa, tol=1e-14)
            for s in (1, -1)
        )
        amplitudes.append(np.stack([plus.e_theta - minus.e_theta, plus.e_phi - minus.e_phi], -1) / delta)
    pair = (4 * amplitudes[1] - amplitudes[0]) / 3  # Richardson: the O(delta^2) octupole terms drop
    q = 3 * (np.outer(p, u) + np.outer(u, p)) - 2 * (p @ u) * np.eye(3)
    loop = 0.5j * (2 * np.pi / LAM) * np.cross(p, u)
    got = ps.dipole_far_field(
        RADII, n, LAM, center, [0, 0, 0], theta, phi, mu=mu, kappa=kappa, magnetic_moment=loop, quadrupole=q,
        magnetic_convention="current", tol=1e-13,
    )  # fmt: skip
    assert np.abs(np.stack([got.e_theta, got.e_phi], -1) - pair).max() < 1e-10 * np.abs(pair).max()


@pytest.mark.parametrize("position", [[0, 0, 20.0], [0, 40.0, 45.0], [30.0, -40.0, 60.0], [300.0, 0, 400.0]])
def test_green_and_reciprocity_far_fields_agree(position):
    """The two far-field routes, for a chiral dipole source: complex amplitudes."""
    from pystratify.rates import _emission

    n, kappa, mu = CHIRAL
    rng = np.random.default_rng(4)
    theta, phi = rng.uniform(0, np.pi, 9), rng.uniform(-np.pi, np.pi, 9)
    p, m = np.array([0.3, 1.0j, 0.5]), np.array([0.1, -0.2j, 0.05])
    ff = ps.dipole_far_field(RADII, n, LAM, position, p, theta, phi, mu=mu, kappa=kappa, magnetic_moment=m, tol=1e-13)
    _, pattern = _emission(
        RADII, n, LAM, position, p, m, "fixed", mu, kappa, "electric", "host", None, 1e-13, 1200, None, True, None,
        "dual", None, directions=(theta, phi),
    )  # fmt: skip
    reference = np.stack([ff.e_theta, ff.e_phi], -1)
    assert np.abs(pattern["fields"][:, 0] - reference).max() < 1e-12 * np.abs(reference).max()
    assert pattern["reference"] == pytest.approx(ff.reference, rel=1e-14)


@pytest.mark.parametrize("orientation", ["fixed", "isotropic", [0, 0, 1.0]])
def test_quadrupole_pattern_integrates_to_power(orientation):
    n, kappa, mu = CHIRAL
    ct, wt = np.polynomial.legendre.leggauss(40)
    phi = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    theta, phi = np.meshgrid(np.arccos(ct), phi, indexing="ij")
    weights = wt[:, None] * (2 * np.pi / 80)
    f = ps.dipole_far_field(
        RADII, n, LAM, [0, 0, 62.0], [0.3, 1j, 0.5], theta, phi, mu=mu, kappa=kappa, magnetic_moment=[0.1, 0.2j, 0],
        quadrupole=_random_quadrupole(5, 10.0), orientation=orientation,
    )  # fmt: skip
    assert np.sum(weights * f.power_density) == pytest.approx(f.power, rel=1e-12)
    for h in (0, 1):
        assert np.sum(weights * f.helicity_intensity[h]) / f.reference == pytest.approx(f.helicity_power[h], rel=1e-12)
    assert (f.e_theta is None) == (orientation != "fixed")


def test_icosahedral_group_is_an_exact_design():
    rotations = _icosahedral_rotations()
    assert len(rotations) == 60
    keys = {tuple(np.round(r, 8).ravel()) for r in rotations}
    assert all(tuple(np.round(a @ b, 8).ravel()) in keys for a, b in itertools.product(rotations, rotations))
    from scipy.spatial.transform import Rotation

    tilt = Rotation.random(random_state=3).as_matrix()
    fourth = [np.kron(np.kron(r, r), np.kron(r, r)) for r in rotations]
    tilted = [np.kron(np.kron(r @ tilt, r @ tilt), np.kron(r @ tilt, r @ tilt)) for r in rotations]
    assert np.abs(np.mean(fourth, axis=0) - np.mean(tilted, axis=0)).max() < 1e-14


def test_current_loop_moments():
    """m_A acts as the dual mu_d m_A (and i kappa_d m_A electric in a chiral layer); the reference is the same loop in the host."""
    radii, n, mu = [50.0, 70.0], [1.45, 1.6, 1.33], [1.3, 1.7, 1.1]
    p, m = np.array([0.3, 0.1, 0.8]), np.array([0.2j, 0.1, -0.4])
    for position in ([0, 0, 30.0], [0, 0, 60.0], [0, 0, 100.0]):
        mu_d = mu[ps.locate_shell(radii, position[2])]
        loop = ps.emission_rates(radii, n, LAM, position, p, magnetic_moment=m, mu=mu, magnetic_convention="current")
        dual = ps.emission_rates(radii, n, LAM, position, p, magnetic_moment=mu_d * m, mu=mu)
        z_h = 1.1 / 1.33
        ratio = (np.vdot(p, p).real + mu_d**2 * np.vdot(m, m).real / z_h**2) / (
            np.vdot(p, p).real + 1.1**2 * np.vdot(m, m).real / z_h**2
        )
        assert loop.total == pytest.approx(dual.total * ratio, rel=1e-13)
        far = ps.dipole_far_field(
            radii, n, LAM, position, p, 0.0, mu=mu, magnetic_moment=m, magnetic_convention="current"
        )
        assert far.power == pytest.approx(loop.radiative, rel=1e-12)
    with pytest.raises(ValueError, match="magnetic_convention"):
        ps.emission_rates(radii, n, LAM, [0, 0, 30.0], p, magnetic_convention="amperian")
    with pytest.raises(ValueError, match="chiral layer"):
        ps.dipole_far_field(
            radii, n, LAM, [0, 0, 30.0], m, 0.0, kappa=[0.1, 0, 0], dipole="magnetic", magnetic_convention="current"
        )


def test_quadrupole_input_validation():
    with pytest.raises(ValueError, match="symmetric"):
        ps.emission_rates(
            [50.0], [1.5, 1.0], LAM, [0, 0, 60.0], [0, 0, 1.0], quadrupole=[[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
        )
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        ps.emission_rates([50.0], [1.5, 1.0], LAM, [0, 0, 60.0], [0, 0, 1.0], quadrupole=[1, 2, 3])
    with pytest.raises(ValueError, match="dipole='electric'"):
        ps.dipole_far_field(
            [50.0], [1.5, 1.0], LAM, [0, 0, 60.0], [0, 0, 1.0], 0.0, dipole="magnetic", quadrupole=np.eye(3)
        )
    # the trace does not radiate: a pure trace is no source, and it does not change a traceless one
    with pytest.raises(ValueError, match="nonzero"):
        ps.emission_rates([50.0], [1.5, 1.0], LAM, [0, 0, 60.0], [0, 0, 0], quadrupole=np.eye(3))
    q = _random_quadrupole()
    a = ps.emission_rates([50.0], [1.5, 1.0], LAM, [0, 0, 60.0], [0, 0, 0], quadrupole=q)
    b = ps.emission_rates([50.0], [1.5, 1.0], LAM, [0, 0, 60.0], [0, 0, 0], quadrupole=q + 3 * np.eye(3))
    assert a.total == pytest.approx(b.total, rel=1e-14)
