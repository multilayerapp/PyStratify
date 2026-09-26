"""Emitters inside chiral multilayers, chiral (electric + magnetic) sources and orientation averages.

Checked against: the achiral solver as kappa -> 0; the interface conditions of
Pasteur media (tangential E, H and the normal D = eps E + i kappa H,
B = mu H - i kappa E continuous) seen through emitters on either side;
helicity conservation of dual (impedance-matched) chiral particles; the
closed-form dissymmetry of a free chiral emitter; exact orientation averages
over finite rotation groups; quadrature; and mirror symmetry.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pystratify as ps

LAM = 614.0
AU = 0.27 + 2.93j
RADII = [40.0, 55.0, 70.0]
N = [1.45 + 0.01j, AU, 1.6 + 0.02j, 1.33]
MU = [1.0, 1.0, 1.2, 1.0]
KAPPA = [0.03 + 0.002j, 0.0, 0.08 + 0.01j, 0.0]  # chiral core and outer shell
P = np.array([1.0, 0.2j, 0.1])
M = np.array([0.05j, 0.1, 0.3j])


def _grid(nodes=60):
    ct, wt = np.polynomial.legendre.leggauss(nodes)
    phi = np.linspace(0, 2 * np.pi, 2 * nodes, endpoint=False)
    theta, phi = np.meshgrid(np.arccos(ct), phi, indexing="ij")
    return theta, phi, wt[:, None] * (2 * np.pi / phi.shape[1])


def _directions(count=7, seed=4):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, np.pi, count), rng.uniform(-np.pi, np.pi, count)


@pytest.mark.parametrize(
    "position", [[5.0, 10.0, -20.0], [20.0, -35.0, 30.0], [30.0, 40.0, 30.0]], ids=["core", "Au", "mu-shell"]
)
@pytest.mark.parametrize("source", ["electric", "magnetic", "chiral"])
def test_emitter_inside_chiral_particle_reduces_to_achiral(position, source):
    theta, phi = _directions()
    kwargs = {"magnetic": {"dipole": "magnetic"}, "chiral": {"magnetic_moment": M}}.get(source, {})
    achiral = ps.dipole_far_field(RADII, N, LAM, position, P, theta, phi, MU, **kwargs)
    chiral = ps.dipole_far_field(RADII, N, LAM, position, P, theta, phi, MU, kappa=[1e-14, 0, 0, 0], **kwargs)
    scale = np.abs(achiral.e_theta).max()
    assert np.abs(chiral.e_theta - achiral.e_theta).max() < 1e-12 * scale
    assert np.abs(chiral.e_phi - achiral.e_phi).max() < 1e-12 * scale
    assert chiral.power == pytest.approx(achiral.power, rel=1e-12)


@pytest.mark.parametrize("interface", [0, 1, 2])
def test_emission_sees_the_pasteur_interface_conditions(interface):
    """Radial and tangential dipoles just inside and outside interfaces between chiral and achiral layers."""
    theta, phi = _directions()
    radius = RADII[interface]
    eps = np.array(N) ** 2 / np.array(MU)
    scale = N[-1] / MU[-1]  # E_r = F_e(z), H_r = -(n_h/mu_h) F_m(z)

    def far(r, moment, dipole="electric"):
        return ps.dipole_far_field(RADII, N, LAM, [0, 0, r], moment, theta, phi, MU, KAPPA, dipole, l_max=40)

    sides = [(radius * (1 - 1e-11), interface), (radius * (1 + 1e-11), interface + 1)]
    for dipole in ("electric", "magnetic"):  # tangential E and H
        inner, outer = (far(r, [1.0, 0.3j, 0.0], dipole) for r, _ in sides)
        assert np.abs(inner.e_theta - outer.e_theta).max() < 1e-9 * np.abs(outer.e_theta).max()
    normal_d, normal_b = [], []
    for r, s in sides:
        e_r, h_r = far(r, [0, 0, 1.0]), far(r, [0, 0, 1.0], "magnetic")
        for component in ("e_theta", "e_phi"):
            e, h = getattr(e_r, component), -scale * getattr(h_r, component)
            normal_d.append(eps[s] * e + 1j * KAPPA[s] * h)
            normal_b.append(MU[s] * h - 1j * KAPPA[s] * e)
    for values in (normal_d, normal_b):
        inner, outer = np.concatenate(values[:2]), np.concatenate(values[2:])
        assert np.abs(inner - outer).max() < 1e-9 * np.abs(outer).max()


def test_helicity_source_inside_a_dual_chiral_particle():
    """(q, m = -i Z q) couples only to E + i Z H: in an impedance-matched particle it emits helicity +1 only."""
    theta, phi, _ = _grid(30)
    n = np.array([1.8, 1.3, 2.2, 1.0]) * np.array([1 + 0.05j, 1, 1 + 0.01j, 1])  # Z = mu/n = 1 everywhere
    q = np.array([0.3, -1.1 + 0.4j, 0.7])
    for position in ([5.0, 10.0, 20.0], [30.0, -40.0, 50.0], [0.0, 70.0, 60.0], [0, 0, 130.0]):
        f = ps.dipole_far_field(
            [40.0, 70.0, 100.0], n, LAM, position, q, theta, phi, mu=n, kappa=[0.2, -0.1 + 0.01j, 0.15, 0],
            magnetic_moment=-1j * q,
        )  # fmt: skip
        plus, minus = f.helicity
        assert np.abs(minus).max() < 1e-13 * np.abs(plus).max()
        assert abs(f.helicity_power[1]) < 1e-13 * f.power and f.dissymmetry == pytest.approx(2, abs=1e-12)


@pytest.mark.parametrize("orientation", ["fixed", "isotropic", [0, 1, 1]])
def test_free_chiral_emitter_dissymmetry(orientation):
    """In a medium of index n: g = 4 n Im(p . m*) / (|p|^2 + n^2 |m|^2) for any orientation average."""
    theta, phi, weight = _grid()
    nh = 1.33
    f = ps.dipole_far_field([30.0], [nh, nh], LAM, [10.0, 5.0, 50.0], P, theta, phi, magnetic_moment=M,
                           orientation=orientation)  # fmt: skip
    expected = 4 * nh * np.imag(np.vdot(M, P)) / (np.vdot(P, P).real + nh**2 * np.vdot(M, M).real)
    assert f.dissymmetry == pytest.approx(expected, rel=1e-12)
    assert f.power == pytest.approx(1, rel=1e-13)
    plus, minus = f.helicity_intensity
    for density, power in ((plus, f.helicity_power[0]), (minus, f.helicity_power[1])):
        assert np.sum(weight * density) / f.reference == pytest.approx(power, rel=1e-12)
    if orientation == "isotropic":  # an isotropic ensemble emits the same everywhere
        assert np.allclose(f.intensity, f.intensity.mean(), rtol=1e-12)
        assert np.allclose(2 * f.circular_polarization, expected, rtol=1e-12)


@pytest.mark.parametrize("kappa", [KAPPA, None], ids=["chiral-particle", "achiral-particle"])
@pytest.mark.parametrize("position", [[10.0, -20.0, 60.0], [10.0, -20.0, 80.0]], ids=["shell", "host"])
def test_orientation_averages_are_exact(kappa, position):
    """Quadratic quantities averaged over the 24 rotations of the cube equal the isotropic average;
    over 8 rotations about an axis, the uniaxial one."""
    theta, phi = _directions(5)
    iso = ps.dipole_far_field(
        RADII, N, LAM, position, P, theta, phi, MU, kappa, magnetic_moment=M, orientation="isotropic"
    )
    runs = [
        ps.dipole_far_field(RADII, N, LAM, position, r @ P, theta, phi, MU, kappa, magnetic_moment=r @ M)
        for r in Rotation.create_group("O").as_matrix()
    ]
    assert np.allclose(np.mean([r.coherency for r in runs], axis=0), iso.coherency, rtol=1e-12, atol=0)
    assert iso.power == pytest.approx(np.mean([r.power for r in runs]), rel=1e-12)
    assert np.allclose(iso.helicity_power, np.mean([r.helicity_power for r in runs], axis=0), rtol=1e-12)
    axis = np.array([0.3, -0.2, 1.0]) / np.linalg.norm([0.3, -0.2, 1.0])
    uni = ps.dipole_far_field(RADII, N, LAM, position, P, theta, phi, MU, kappa, magnetic_moment=M, orientation=axis)
    rotations = [Rotation.from_rotvec(a * axis).as_matrix() for a in np.arange(8) * np.pi / 4]
    runs = [
        ps.dipole_far_field(RADII, N, LAM, position, r @ P, theta, phi, MU, kappa, magnetic_moment=r @ M)
        for r in rotations
    ]
    assert np.allclose(np.mean([r.coherency for r in runs], axis=0), uni.coherency, rtol=1e-12, atol=0)
    assert uni.power == pytest.approx(np.mean([r.power for r in runs]), rel=1e-12)
    with pytest.raises(ValueError):
        _ = uni.helicity  # an average has no amplitudes


def test_powers_and_polarisation_by_quadrature():
    theta, phi, weight = _grid(70)
    for position, kappa in (([10.0, 20.0, -50.0], KAPPA), ([0, -30.0, 75.0], KAPPA), ([10.0, 20.0, -50.0], None)):
        f = ps.dipole_far_field(
            RADII, N, LAM, position, P, theta, phi, MU, kappa, magnetic_moment=M, orientation="isotropic"
        )
        assert np.sum(weight * f.power_density) == pytest.approx(f.power, rel=1e-12)
        assert sum(f.helicity_power) == pytest.approx(f.power, rel=1e-13)
        plus, minus = f.helicity_intensity
        assert np.sum(weight * plus) / f.reference == pytest.approx(f.helicity_power[0], rel=1e-12)
        stokes = f.stokes()
        assert np.allclose(stokes[..., 3], -stokes[..., 0] * f.circular_polarization)
        assert np.all(stokes[..., 0] ** 2 >= np.sum(stokes[..., 1:] ** 2, -1) * (1 - 1e-12))  # partially polarised


def test_source_normalisation():
    """m = 0 is the electric dipole; with p = 0 the magnetic moment enters as (n_h/mu_h) m."""
    theta, phi = _directions()
    kw = dict(mu=MU, kappa=KAPPA)
    for position in ([5.0, 10.0, -20.0], [0, 0, 90.0]):
        electric = ps.dipole_far_field(RADII, N, LAM, position, P, theta, phi, **kw)
        both = ps.dipole_far_field(RADII, N, LAM, position, P, theta, phi, **kw, magnetic_moment=[0, 0, 0])
        assert np.allclose(both.e_theta, electric.e_theta, rtol=1e-14) and both.power == pytest.approx(electric.power)
        magnetic = ps.dipole_far_field(RADII, N, LAM, position, M, theta, phi, dipole="magnetic", **kw)
        only_m = ps.dipole_far_field(RADII, N, LAM, position, [0, 0, 0], theta, phi, **kw, magnetic_moment=M)
        assert np.allclose(only_m.e_theta, N[-1] / MU[-1] * magnetic.e_theta, rtol=1e-13)
        assert only_m.power == pytest.approx(magnetic.power, rel=1e-13)


def test_valley_exciton_near_a_sphere():
    """A circular in-plane dipole (a 2D-material valley exciton) emits no net helicity alone, but beside
    a sphere its emission is: mirror images (other valley, or other side) have the opposite g."""
    for radii, n in (([50.0], [AU, 1.33]), ([75.0], [3.5, 1.0])):
        r0 = radii[0] + 2.0
        g = {
            (s, z): ps.dipole_far_field(radii, n, LAM, [0, 0, z * r0], [1, s * 1j, 0], 0.0).dissymmetry
            for s in (1, -1)
            for z in (1, -1)
        }
        assert abs(g[1, 1]) > 1e-2
        assert g[-1, 1] == pytest.approx(-g[1, 1], rel=1e-12) and g[1, -1] == pytest.approx(-g[1, 1], rel=1e-12)
        assert g[-1, -1] == pytest.approx(g[1, 1], rel=1e-12)
    free = ps.dipole_far_field([50.0], [1.33, 1.33], LAM, [0, 0, 52.0], [1, 1j, 0], 0.0)
    assert abs(free.dissymmetry) < 1e-14


def test_emitter_input_validation():
    with pytest.raises(ValueError):
        ps.dipole_far_field([50.0], [1.5, 1.33], LAM, [0, 0, 60.0], [1, 0, 0], 0.0, orientation="random")
    with pytest.raises(ValueError):
        ps.dipole_far_field([50.0], [1.5, 1.33], LAM, [0, 0, 60.0], [1, 0, 0], 0.0, orientation=[0, 0, 0])
    with pytest.raises(ValueError):
        ps.dipole_far_field(
            [50.0], [1.5, 1.33], LAM, [0, 0, 60.0], [1, 0, 0], 0.0, dipole="magnetic", magnetic_moment=[1, 0, 0]
        )
    with pytest.raises(ValueError):
        ps.dipole_far_field([50.0], [1.5, 1.33], LAM, [0, 0, 60.0], [0, 0, 0], 0.0, magnetic_moment=[0, 0, 0])
