"""Angle-resolved far field and directivity: plane-wave scattering and dipole emission.

Plane waves: the amplitude matrix against textbook Mie theory with
independently computed angular functions, the far-zone limit of the near
field, Bohren & Huffman's backscattering reference value, and the Mueller
matrix identities.  Dipoles: closed forms in a homogeneous medium,
reciprocity against the plane-wave near field (which fixes every phase),
radiated power against the Green's-function decay rates, and quadrature.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

import pystratify as ps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import sphere_emitter_checks as emitters  # noqa: E402
import sphere_checks as spheres  # noqa: E402
from checks import assert_rows  # noqa: E402
from sphere_emitter_checks import _basis  # noqa: E402

LAM = 614.0
AU = 0.27 + 2.93j
NANOSHELL = ([50.0, 55.0], [1.45, AU, 1.33])



def _sphere_grid(nodes=80):
    ct, wt = np.polynomial.legendre.leggauss(nodes)
    phi = np.linspace(0, 2 * np.pi, 2 * nodes, endpoint=False)
    theta, phi = np.meshgrid(np.arccos(ct), phi, indexing="ij")
    return theta, phi, wt[:, None] * (2 * np.pi / phi.shape[1])


# ----------------------------------------------------------------- plane wave


def test_amplitude_matrix_against_textbook_mie():
    assert_rows(spheres.amplitude_matrix_against_textbook_mie())


def test_bohren_huffman_backscattering_from_the_pattern():
    assert_rows([r for r in spheres.bohren_huffman_reference_values() if 'pattern' in r.quantity])


@pytest.mark.parametrize("theta, phi", [(1.1, 0.7), (2.9, -2.0), (0.2, 3.0)])
def test_pattern_is_the_far_zone_limit_of_the_near_field(theta, phi):
    """Scattered near field times -ikr exp(-ikr).

    The scattered field is taken without the incident wave: at kr ~ 1e7 it is ~1e-7 of an O(1) plane
    wave, so subtracting a particle-free field would round away the digits this checks."""
    sol = ps.solve(*NANOSHELL, LAM, l_max=20)
    k = 2 * np.pi * 1.33 / LAM
    pattern = ps.scattering_pattern(sol, theta, phi, (1, 0))
    residual = []
    for r in (3e6, 3e8):
        point = r * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        f = ps.near_field(sol, *point, incident=False)
        scale = -1j * k * r * np.exp(-1j * k * r)
        residual.append(
            max(
                abs(f.e["theta"] * scale - pattern.e_theta[0]),
                abs(f.e["phi"] * scale - pattern.e_phi[0]),
            )
            / abs(pattern.e_theta[0])
        )
    assert residual[1] < 1e-6 and residual[0] / residual[1] == pytest.approx(100, rel=0.01)  # O(1/kr)


def test_polarisation_rotation_and_superposition():
    sol = ps.solve(*NANOSHELL, LAM, l_max=20)
    theta, phi = np.linspace(0.1, 3.0, 7), np.linspace(-2, 2, 7)
    x = ps.scattering_pattern(sol, theta, phi, (1, 0))
    y = ps.scattering_pattern(sol, theta, phi + np.pi / 2, (0, 1))  # y-polarised = x rotated by 90 degrees
    assert np.allclose(x.e_theta, y.e_theta, rtol=1e-12) and np.allclose(x.e_phi, y.e_phi, rtol=1e-12)
    jones = np.array([0.3, 0.8 + 0.4j])
    mixed = ps.scattering_pattern(sol, theta, phi, jones)
    y0 = ps.scattering_pattern(sol, theta, phi, (0, 1))
    norm = np.linalg.norm(jones)
    assert np.allclose(mixed.e_theta, (jones[0] * x.e_theta + jones[1] * y0.e_theta) / norm, rtol=1e-12)


def test_directivity_and_differential_cross_section_integrate():
    lam = np.array([500.0, 614.0])
    n = np.array([[1.45, AU, 1.33], [1.45, 0.3 + 2.5j, 1.33]])
    sol = ps.solve(NANOSHELL[0], n, lam, l_max=20)
    theta, phi, weight = _sphere_grid()
    pattern = ps.scattering_pattern(sol, theta, phi, (0.3, 0.8 + 0.4j))
    assert pattern.directivity.shape == (2,) + theta.shape
    assert np.allclose(np.sum(weight * pattern.directivity, axis=(1, 2)), 4 * np.pi, rtol=1e-12)
    assert np.allclose(np.sum(weight * pattern.differential_cross_section, axis=(1, 2)), ps.cross_sections(sol).sca)
    hc = ps.helicity_cross_sections(sol)
    cs = ps.cross_sections(sol)
    for i in (0, 1):
        assert np.allclose(hc.ext[:, i], cs.ext, rtol=1e-12) and np.allclose(hc.abs[:, i], cs.abs, rtol=1e-11)


def test_circular_polarisation_forward_and_backward():
    """A sphere keeps the helicity forward and reverses it in backscattering."""
    sol = ps.solve(*NANOSHELL, LAM, l_max=20)
    pattern = ps.scattering_pattern(sol, np.array([0.0, np.pi]), 0.0, (1, 1j))
    assert pattern.circular_polarization[0] == pytest.approx([1, -1], abs=1e-12)
    stokes = pattern.stokes()
    assert np.allclose(stokes[0, :, 3], -stokes[0, :, 0] * pattern.circular_polarization[0])
    plus, minus = pattern.helicity
    assert np.allclose(np.abs(plus) ** 2 + np.abs(minus) ** 2, pattern.intensity)


def test_mueller_matrix():
    theta = np.linspace(0, np.pi, 9)
    sol = ps.solve(*NANOSHELL, LAM, l_max=20)
    s1, s2, _, _ = (s[0] for s in ps.amplitude_matrix(sol, theta))
    m = ps.mueller_matrix(sol, theta)[0]
    assert np.allclose(m[:, 0, 0], (abs(s1) ** 2 + abs(s2) ** 2) / 2, rtol=1e-12)
    assert np.allclose(m[:, 0, 1], (abs(s2) ** 2 - abs(s1) ** 2) / 2, atol=1e-12 * m[:, 0, 0].max())
    assert np.allclose(m[:, 2, 2], np.real(s1 * np.conj(s2)), atol=1e-12 * m[:, 0, 0].max())
    assert np.allclose(m[:, 2, 3], np.imag(s2 * np.conj(s1)), atol=1e-12 * m[:, 0, 0].max())
    chiral = ps.mueller_matrix(ps.solve_chiral([40.0, 70.0], [1.45, 1.6 + 0.02j, 1.33], [0, 0.08, 0], LAM), theta)[0]
    # a Mueller matrix from one Jones matrix is non-depolarising: tr(M^T M) = 4 M11^2
    assert np.allclose(np.einsum("tij,tij->t", chiral, chiral), 4 * chiral[:, 0, 0] ** 2, rtol=1e-11)
    assert np.abs(chiral[:, 0, 3]).max() > 1e-6 * chiral[:, 0, 0].max()  # circular dichroism in scattering


# -------------------------------------------------------------------- dipoles


@pytest.mark.parametrize("dipole", emitters.DIPOLES)
@pytest.mark.parametrize("position", emitters.FREE_POSITIONS)
def test_dipole_in_homogeneous_medium_is_the_closed_form(position, dipole):
    assert_rows(emitters.dipole_in_homogeneous_medium(position, dipole))


@pytest.mark.parametrize("dipole", emitters.DIPOLES)
@pytest.mark.parametrize("where", emitters.RECIPROCITY_POSITIONS)
def test_dipole_emission_is_reciprocal_to_plane_wave_near_field(where, dipole):
    assert_rows(emitters.reciprocity_with_near_field(where, dipole))


@pytest.mark.parametrize("dipole", emitters.DIPOLES)
def test_radiated_power_equals_decay_rates(dipole):
    assert_rows(emitters.radiated_power_equals_decay_rates(dipole))


def test_power_directivity_and_quadrature():
    radii, n, mu = [50.0, 70.0, 90.0], [1.45, AU, 1.6, 1.33], [1.3, 1.0, 1.7, 1.1]
    theta, phi, weight = _sphere_grid(90)
    for position, moment, dipole in (
        ([10.0, 20.0, -55.0], [1.0, 1j, 0.2], "electric"),  # inside the absorbing shell
        ([30.0, -60.0, 70.0], [0.3, 0.2 - 0.5j, 1.0], "magnetic"),
        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], "electric"),  # centre: a pure dipole pattern
    ):
        f = ps.dipole_far_field(radii, n, LAM, position, moment, theta, phi, mu, dipole=dipole)
        assert np.sum(weight * f.power_density) == pytest.approx(f.power, rel=1e-12)
        assert np.sum(weight * f.directivity) == pytest.approx(4 * np.pi, rel=1e-12)
    assert f.directivity.max() == pytest.approx(1.5, rel=1e-9)  # sin^2 pattern of the centred dipole


def test_chiral_particle_emission():
    radii, n, kappa = [40.0, 55.0, 70.0], [1.45 + 0.01j, AU, 1.6 + 0.02j, 1.33], [0.03 + 0.002j, 0, 0.08 + 0.01j, 0]
    k = 2 * np.pi * 1.33 / LAM
    theta, phi = np.array([0.3, 1.2, 2.0, 2.9]), np.array([0.1, -1.0, 2.5, 1.7])
    directions, e_theta, e_phi = _basis(theta, phi)
    plane = ps.scattering_pattern(ps.solve_chiral(radii, n, kappa, LAM), theta, phi, (1, 0))
    # a distant x dipole on the -z axis illuminates the particle like an x-polarised plane wave
    residual = []
    for distance in (1e6, 1e8):
        position = np.array([0.0, 0.0, -distance])
        f = ps.dipole_far_field(radii, n, LAM, position, [1.0, 0, 0], theta, phi, kappa=kappa)
        direct = (np.array([1.0, 0, 0]) - directions[:, :1] * directions) * np.exp(-1j * k * (directions @ position))[
            :, None
        ]
        scale = -1j * k * distance * np.exp(-1j * k * distance)
        scattered = (f.e_theta - np.sum(direct * e_theta, -1)) * scale
        residual.append(np.abs(scattered - plane.e_theta[0]).max() / np.abs(plane.e_theta[0]).max())
    assert residual[1] < 1e-5 and residual[0] / residual[1] == pytest.approx(100, rel=0.05)
    # a dual chiral particle keeps the helicity of the helicity source (p, m = -i p)
    grid = _sphere_grid(40)[:2]
    nd = np.array([1.8, 1.3, 1.0]) * np.array([1 + 0.05j, 1, 1])
    moment = np.array([0.3, -1.1 + 0.4j, 0.7])
    for position in ([0.0, 0.0, 75.0], [30.0, -40.0, 50.0]):
        e = ps.dipole_far_field([40.0, 70.0], nd, LAM, position, moment, *grid, nd, [0.2, -0.1 + 0.01j, 0])
        m = ps.dipole_far_field(
            [40.0, 70.0], nd, LAM, position, -1j * moment, *grid, nd, [0.2, -0.1 + 0.01j, 0], "magnetic"
        )
        plus, minus = (a + b for a, b in zip(e.helicity, m.helicity))
        assert np.abs(minus).max() < 1e-13 * np.abs(plus).max()
    # circular dipoles of opposite handedness radiate differently next to a chiral particle
    powers = [ps.dipole_far_field(radii, n, LAM, [0, 0, 75.0], [1, s * 1j, 0], 0.0, kappa=kappa).power for s in (1, -1)]
    assert abs(powers[0] - powers[1]) > 1e-3 * powers[0]
    inside = ps.dipole_far_field(radii, n, LAM, [0, 0, 60.0], [1, 0, 0], 0.0, kappa=kappa)  # in the chiral shell
    assert inside.shell == 2 and inside.converged and inside.power > 0


def test_dipole_input_validation():
    with pytest.raises(ValueError):
        ps.dipole_far_field([50.0], [1.5, 1.33 + 0.01j], LAM, [0, 0, 60.0], [1, 0, 0], 0.0)  # lossy host
    with pytest.raises(ValueError):
        ps.dipole_far_field([50.0], [1.5, 1.33], LAM, [0, 0, 60.0], [0, 0, 0], 0.0)
    with pytest.raises(ValueError):
        ps.dipole_far_field([50.0], [1.5, 1.33], LAM, [0, 60.0], [1, 0, 0], 0.0)
    with pytest.raises(ValueError):
        ps.dipole_far_field([50.0], [1.5, 1.33], LAM, [0, 0, 60.0], [1, 0, 0], 0.0, dipole="quadrupole")
