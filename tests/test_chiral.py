"""Chiral (Pasteur) multilayers: exact references, an independent code, and symmetries.

The solver's numerics are checked against direct 4x4 transfer matrices in 80
digits; its physics against treams (Beutel et al., CPC 297, 109076 (2024)),
an independent T-matrix code, when that is installed, and against laws no
implementation detail can fake: kappa = 0 is the achiral solver, lossless
particles absorb nothing, reciprocity makes the (TM, TE) block symmetric,
kappa -> -kappa exchanges the helicities, and an impedance-matched (dual)
particle cannot flip helicity.
"""

import numpy as np
import pytest

import pystratify as ps

from .mp_reference import chiral_t_matrix

LAM = 614.0
AU = 0.27 + 2.93j
MATRYOSHKA = ([10.0, 13.0, 36.0, 48.0], [1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33])  # 3-nm Au shell
CHIRAL_SHELL = ([40.0, 55.0, 70.0], [1.45 + 0.01j, AU, 1.6 + 0.02j, 1.33], [0.03 + 0.002j, 0, 0.08 + 0.01j, 0])


def _norm_error(log_got, log_ref):
    """max |got - ref| / max |ref| for matrices given as complex logarithms."""
    shift = np.max(np.real(log_ref))
    return float(np.max(np.abs(np.exp(log_got - shift) - np.exp(log_ref - shift))))


@pytest.mark.parametrize(
    "radii, n, kappa, mu, tol",
    [
        ([40.0, 55.0], [1.45 + 0.01j, 1.6 + 0.02j, 1.33], [0, 0.05 + 0.01j, 0], [1, 1, 1], 1e-11),
        ([30.0, 33.0, 60.0], [1.5, AU, 1.55 + 0.001j, 1.33], [0.02, 0, 0.1 + 0.003j, 0], [1, 1, 1.2, 1], 1e-11),
        (MATRYOSHKA[0], MATRYOSHKA[1], [0.01, 0, 0.2 + 0.01j, 0, 0], [1] * 5, 1e-11),
        ([40.0, 60.0], [1.5, 1.6 - 0.02j, 1.33], [0.03, 0.05, 0], [1, 1, 1], 1e-11),  # gain shell
        # a chirality contrast of 1e-6 alone: T is known to ~ eps / contrast
        ([40.0, 60.0], [1.5, 1.5, 1.5], [0, 1e-6, 0], [1, 1, 1], 1e-9),
    ],
)
def test_t_matrix_against_80_digit_transfer_matrices(radii, n, kappa, mu, tol):
    sol = ps.solve_chiral(radii, n, kappa, 690.0, mu, l_max=160)
    for order in (1, 7, 40, 100, 160):
        reference = np.array(chiral_t_matrix(radii, n, kappa, mu, 690.0, order))
        assert _norm_error(sol.log_t_helicity[0, order - 1], reference) < tol, order


@pytest.mark.parametrize(
    "radii, n, mu",
    [
        ([50.0, 55.0], [1.45, AU, 1.33], [1, 1, 1]),
        ([40.0, 60.0, 65.0, 80.0], [1.45 + 0.02j, 2.1, AU, 1.5, 1.0], [1, 1.2, 1, 1, 1]),
        ([30.0, 31.0, 60.0], [2.5 + 0.01j, 0.05 + 4.0j, 1.5, 1.33], [1, 1, 1, 1]),  # 1-nm Ag-like film
        ([40.0, 60.0], [1.5, 1.6 - 0.02j, 1.33], [1, 1, 1]),  # gain shell
    ],
)
def test_achiral_limit_is_the_achiral_solver(radii, n, mu):
    chiral = ps.solve_chiral(radii, n, 0, LAM, mu, l_max=40).t_matrix
    achiral = ps.solve(radii, n, LAM, mu, l_max=40).t_matrix
    scale = np.abs(achiral).max(axis=(-2, -1), keepdims=True)
    assert np.max(np.abs(chiral - achiral) / scale) < 1e-12


def test_against_treams():
    """An independent code (helicity basis; its negative-helicity waves differ by a sign)."""
    treams = pytest.importorskip("treams")
    cases = [
        ([40.0, 55.0], [1.45 + 0.01j, 1.6 + 0.02j, 1.33], [0, 0.05 + 0.01j, 0], [1, 1, 1], 614.0, 12),
        ([30.0, 33.0, 60.0], [1.5, AU, 1.55 + 0.001j, 1.33], [0.02, 0, 0.1 + 0.003j, 0], [1, 1, 1.2, 1], 614.0, 20),
        ([200.0, 230.0], [2.0 + 0.05j, 1.5, 1.0], [0.3, -0.2 + 0.01j, 0], [1, 1, 1], 500.0, 30),
    ]
    for radii, n, kappa, mu, lam, lmax in cases:
        materials = [treams.Material(v**2 / m, m, k) for v, k, m in zip(n, kappa, mu)]
        tm = treams.TMatrix.sphere(lmax, 2 * np.pi / lam, radii, materials, poltype="helicity")
        basis, block = tm.basis, np.asarray(tm)
        ours = ps.solve_chiral(radii, n, kappa, lam, mu, l_max=lmax).t_helicity[0]
        for l in range(1, lmax + 1):
            plus, minus = (
                int(np.flatnonzero((basis.l == l) & (basis.m == 0) & (basis.pol == pol))[0]) for pol in (1, 0)
            )
            theirs = np.array([[block[plus, plus], -block[plus, minus]], [-block[minus, plus], block[minus, minus]]])
            assert np.abs(ours[l - 1] - theirs).max() < 1e-11 * np.abs(theirs).max(), (radii, l)


def test_lossless_chiral_multilayers_absorb_nothing():
    rng = np.random.default_rng(11)
    for _ in range(10):
        radii = np.sort(rng.uniform(20, 300, 4))
        n = np.r_[rng.uniform(1.2, 3.0, 4), 1.33]
        kappa = np.r_[rng.uniform(-0.3, 0.3, 4), 0]
        hc = ps.helicity_cross_sections(ps.solve_chiral(radii, n, kappa, LAM))
        assert np.abs(hc.abs).max() < 1e-12 * hc.ext.max()


def test_absorbing_chiral_shells_absorb_for_both_helicities():
    rng = np.random.default_rng(12)
    for _ in range(20):
        radii = np.sort(rng.uniform(5, 150, 3))
        n = np.r_[rng.uniform(1.2, 3.0, 3) + 1j * rng.uniform(0.01, 1, 3), 1.0]
        kappa = np.r_[rng.uniform(-0.2, 0.2, 3) + 1j * rng.uniform(-0.005, 0.005, 3), 0]
        hc = ps.helicity_cross_sections(ps.solve_chiral(radii, n, kappa, LAM))
        assert np.all(hc.abs_by_order >= -1e-15 * hc.ext.max())


def test_reciprocity_makes_the_block_symmetric():
    t = ps.solve_chiral(*CHIRAL_SHELL[:2], CHIRAL_SHELL[2], LAM).t_matrix
    assert np.abs(t[..., 0, 1] - t[..., 1, 0]).max() < 1e-12 * np.abs(t[..., 0, 1]).max()
    assert np.abs(t[..., 0, 1]).max() > 1e-4  # the coupling is really there


def test_mirror_image_exchanges_helicities():
    radii, n, kappa = CHIRAL_SHELL
    t = ps.solve_chiral(radii, n, kappa, LAM).t_helicity
    mirror = ps.solve_chiral(radii, n, -np.asarray(kappa), LAM).t_helicity
    assert np.allclose(mirror, t[..., ::-1, ::-1], rtol=1e-13, atol=1e-16)
    hc, hm = (
        ps.helicity_cross_sections(ps.solve_chiral(radii, n, kappa, LAM)),
        ps.helicity_cross_sections(ps.solve_chiral(radii, n, -np.asarray(kappa), LAM)),
    )
    assert hc.g_abs[0] == pytest.approx(-hm.g_abs[0], rel=1e-10) and abs(hc.g_abs[0]) > 1e-3


def test_dual_particle_conserves_helicity():
    """Helicity flips only through impedance contrast: none for Z = mu/n equal everywhere."""
    radii = [60.0, 90.0, 130.0]
    n = np.array([1.8, 1.3, 2.5, 1.0]) * (1 + np.array([0.05j, 0, 0.02j, 0]))
    kappa = [0.2, -0.1 + 0.01j, 0.3, 0]
    flip = []
    for contrast in (0.0, 1e-6, 1e-3):
        mu = n * np.array([1 + contrast, 1, 1, 1])
        t = ps.solve_chiral(radii, n, kappa, LAM, mu).t_helicity
        flip.append(np.abs(t[..., 0, 1]).max() / np.abs(t[..., 0, 0]).max())
    assert flip[0] == 0.0
    assert flip[2] / flip[1] == pytest.approx(1e3, rel=1e-2)  # linear in the contrast, down to 1e-6


def test_identical_chiral_shells_are_one_sphere():
    layered = ps.solve_chiral(
        np.linspace(2.0, 100.0, 50), [2.0 + 0.1j] * 50 + [1.33], [0.1] * 50 + [0], 600.0, l_max=30
    )
    single = ps.solve_chiral([100.0], [2.0 + 0.1j, 1.33], [0.1, 0], 600.0, l_max=30)
    assert np.allclose(layered.t_matrix, single.t_matrix, rtol=1e-11, atol=1e-18)


def test_chiral_core_behind_thick_metal_is_invisible():
    """rho falls by ~e^-600 across the metal (below the double range); an index-matched
    interface inside the metal must still pass it through without NaN."""
    reference = ps.solve([3210.0], [0.3 + 11j, 1.0], 500.0, l_max=80).t_matrix
    sol = ps.solve_chiral([1000.0, 3200.0, 3210.0], [1.5, 0.3 + 11j, 0.3 + 11j, 1.0], [0.1, 0, 0, 0], 500.0, l_max=80)
    assert np.all(np.isfinite(sol.log_t_helicity.real) | (sol.log_t_helicity.real == -np.inf))
    assert np.abs(sol.t_matrix - reference).max() < 1e-12 * np.abs(reference).max()


def test_far_field_of_a_chiral_sphere():
    """S3 = -S4, the optical theorem for each helicity, and the scattered power."""
    sol = ps.solve_chiral(*CHIRAL_SHELL[:2], CHIRAL_SHELL[2], LAM)
    s1, s2, s3, s4 = ps.amplitude_matrix(sol, np.linspace(0, np.pi, 9))
    assert np.abs(s3 + s4).max() < 1e-12 * np.abs(s3).max()
    hc = ps.helicity_cross_sections(sol)
    k = 2 * np.pi * 1.33 / LAM
    ct, wt = np.polynomial.legendre.leggauss(60)
    phi = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    theta, phi = np.meshgrid(np.arccos(ct), phi, indexing="ij")
    for i, jones in enumerate(((1, 1j), (1, -1j))):
        e = np.array(jones) / np.sqrt(2)
        forward = ps.scattering_pattern(sol, 0.0, 0.0, jones)
        ext = 4 * np.pi / k**2 * np.real(forward.e_theta[0] * np.conj(e[0]) + forward.e_phi[0] * np.conj(e[1]))
        assert ext == pytest.approx(hc.ext[0, i], rel=1e-12)
        pattern = ps.scattering_pattern(sol, theta, phi, jones)
        sca = np.sum(wt[:, None] * pattern.differential_cross_section[0]) * (2 * np.pi / phi.shape[1])
        assert sca == pytest.approx(hc.sca[0, i], rel=1e-12)
    linear = ps.scattering_pattern(sol, 0.3, 0.0, (1, 0))
    assert linear.sca[0] == pytest.approx(hc.sca[0].mean(), rel=1e-13)


def test_chiral_batch_equals_single():
    lam = np.array([500.0, 600.0, 700.0])
    n = np.array([[1.45, 0.3 + 2.0j + 0.1j * i, 1.5, 1.33] for i in range(3)])
    kappa = np.array([[0.0, 0.0, 0.05 + 0.01j * i, 0.0] for i in range(3)])
    batch = ps.solve_chiral([40, 50, 60], n, kappa, lam, l_max=12)
    for i in range(3):
        single = ps.solve_chiral([40, 50, 60], n[i], kappa[i], lam[i], l_max=12)
        assert np.allclose(batch.t_matrix[i], single.t_matrix[0], rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "kappa, n",
    [([0.1, 0.1], [1.5, 1.33]), ([2.0, 0], [1.5, 1.33]), ([0.1], [1.5, 1.33]), ([np.nan, 0], [1.5, 1.33])],
)
def test_chiral_input_validation(kappa, n):
    with pytest.raises(ValueError):
        ps.solve_chiral([50.0], n, kappa, LAM, l_max=5)


def test_achiral_functions_refuse_chiral_solutions():
    sol = ps.solve_chiral([50.0], [1.5, 1.33], [0.1, 0], LAM, l_max=5)
    with pytest.raises(TypeError):
        ps.cross_sections(sol)
    with pytest.raises(TypeError):
        ps.scattering_amplitudes(sol, [0.1])


@pytest.mark.parametrize(
    "eps, mu, chi",
    [(4.0, 1.0, 0.1), (-4.0 + 0.3j, 1.0, 0.05), (6.0, 1.2, 0.15), (2.25 + 0.1j, 0.9, -0.08), (-2.2 + 0.1j, 1.0, 0.1)],
)
def test_small_chiral_sphere_matches_published_polarisabilities(eps, mu, chi):
    """Quasi-static polarisabilities of Klimov, Guzatov & Ducloy, EPL 97, 47004 (2012), Eq. 48, for the
    Drude-Born-Fedorov medium D = eps (E + eta curl E), B = mu (H + eta curl H), chi = k0 eta.  In Pasteur
    form eps_P = eps/s, mu_P = mu/s, kappa = n^2 chi/s with s = 1 - n^2 chi^2 (same impedance, n +- kappa =
    n/(1 -+ n chi) = their k_L, k_R), and the dipole block is T = (2i/3) k^3 [[a_EE, -i a_EH], [-i a_EH, a_HH]]."""
    k = 2 * np.pi / 600.0
    a = 1e-3 / k
    den = (eps + 2) * (mu + 2) - 4 * chi**2 * eps * mu
    a_ee = a**3 * ((eps - 1) * (mu + 2) + 2 * chi**2 * eps * mu) / den
    a_hh = a**3 * ((mu - 1) * (eps + 2) + 2 * chi**2 * eps * mu) / den
    a_eh = a**3 * 3j * chi * eps * mu / den
    n2 = eps * mu
    s = 1 - n2 * chi**2
    n_p = np.sqrt(n2) / s
    n_p = -n_p if n_p.imag < 0 else n_p
    t = ps.solve_chiral([a], [n_p, 1.0], [n2 * chi / s, 0], 600.0, [mu / s, 1.0], l_max=3).t_matrix[0, 0]
    c = 2j * k**3 / 3
    # (k a)^2 = 1e-6 dynamic corrections, resonantly enhanced near (eps + 2)(mu + 2) = 4 chi^2 eps mu
    assert t[0, 0] == pytest.approx(c * a_ee, rel=1e-4)
    assert t[1, 1] == pytest.approx(c * a_hh, rel=1e-4)
    assert t[0, 1] == pytest.approx(-1j * c * a_eh, rel=1e-4) and t[1, 0] == pytest.approx(t[0, 1], rel=1e-12)


def test_chiral_metal_with_negative_real_part_of_n_minus_kappa():
    """Re(n - kappa) < 0 is fine inside the particle (psi and xi are only a basis there)."""
    sol = ps.solve_chiral([50.0], [0.2 + 3.0j, 1.33], [0.5 + 0.01j, 0], LAM)
    hc = ps.helicity_cross_sections(sol)
    assert np.all(np.isfinite(hc.ext)) and np.all(hc.abs > 0) and abs(hc.g_ext[0]) > 1e-4
