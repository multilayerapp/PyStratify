"""Focused illumination: every semi-analytic form against an independent Python path.

Brute force here is a plane-wave superposition on a dense pupil grid through the
existing engines (amplitude_matrix, oblique solve_cylinder, coh_tmm/position_resolved,
near_field, CylinderSolution.field), called, never re-derived. The reference helpers and
the checks cited in benchmarks/MATRIX.md live in benchmarks/focused_checks.py, which prints
every observed value (FOCUSED_CHECKS.md); these tests assert its rows.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

import pystratify as ps
from pystratify import focused as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import focused_checks as focused  # noqa: E402
from checks import assert_rows  # noqa: E402
from focused_checks import CYLINDER, SPHERE, collection_grid, film_brute, pupil_field, pupil_grid  # noqa: E402

inf = np.inf
TM, TE = ps.TM, ps.TE


# ---------------------------------------------------------------- pupil model

def test_pupil_is_normalized_and_aplanatic():
    u, w = pupil_grid(0.23, 0.5, 30, 8)
    A = pupil_field(u, (0.23, 0.5), (1.0, 0.0), 1.0, (0, 0, 0))
    assert np.isclose(np.sum(np.abs(A) ** 2, -1) @ w, 1, atol=1e-13)
    # power per direction uniform in sin^2: half the power below the median sin^2
    s = np.hypot(u[:, 0], u[:, 1])
    below = s ** 2 < (0.23 ** 2 + 0.5 ** 2) / 2
    assert np.isclose(np.sum(np.abs(A[below]) ** 2, -1) @ w[below], 0.5, atol=0.02)
    # the focal field of the x pupil is along x with |E| = focal_amplitude
    e, h = F.focal_field(np.zeros((1, 3)), 2.0, (0.23, 0.5), (1.0, 0.0))
    assert np.isclose(abs(e[0, 0]), F.focal_amplitude((0.23, 0.5))) and np.allclose(e[0, 1:], 0)
    assert np.allclose(np.sum(A * w[:, None], 0), e[0], atol=1e-12)


def test_vector_harmonics_are_orthonormal_and_match_bohren_huffman():
    L = 6
    x, w = F._gauss(40)
    phi = 2 * np.pi * np.arange(32) / 32
    blocks = {}
    for m, P, pi, tau in F.vector_harmonics(L, x):
        blocks[m] = (P, pi, tau)
    X = {}
    for m, (P, pi, tau) in blocks.items():
        for l in range(max(1, abs(m)), L + 1):
            scale = 1 / np.sqrt(2 * np.pi * l * (l + 1))
            wave = np.exp(1j * m * phi)[None]
            X[(l, m)] = (np.stack((-pi[l][:, None] * wave, -1j * tau[l][:, None] * wave)) * scale,
                         np.stack((1j * tau[l][:, None] * wave, -pi[l][:, None] * wave)) * scale)
    keys = list(X)
    weights = w[:, None] * (2 * np.pi / 32)
    for a in keys[::3]:
        for b in keys[::2]:
            for i in range(2):
                for j in range(2):
                    inner = np.sum(np.conj(X[a][i]) * X[b][j] * weights)
                    assert abs(inner - (a == b and i == j)) < 1e-12
    # m = 1: pi_l1 and tau_l1 are the Bohren-Huffman pi_l, tau_l up to the normalization (and CS sign)
    pi_bh, tau_bh = ps.angular_functions(np.arange(1, L + 1), np.arccos(x))
    P, pi, tau = blocks[1]
    for l in range(1, L + 1):
        c = -np.sqrt((2 * l + 1) / (2 * l * (l + 1)))
        assert np.allclose(pi[l], c * pi_bh[l - 1]) and np.allclose(tau[l], c * tau_bh[l - 1])


# ---------------------------------------------------------------- films

@pytest.mark.parametrize("mode", focused.FILM_MODES)
def test_film_cone_quadrature_against_pupil_grid(mode):
    assert_rows(focused.film_cone_quadrature(mode))


def test_film_incoherent_window_follows_the_coherence_column():
    n = np.array([1.0, 1.55 + 0.03j, 1.42 + 1e-5j, 1.0])
    d = np.array([inf, 1.2, 400.0, inf])
    c = ["i", "c", "i", "i"]
    beam = ps.FocusedBeam(0.5, 0.23)
    r = F.focused_films(n, d, 4.0, beam, c, reference=(n[[0, 2, 3]], d[[0, 2, 3]], ["i", "i", "i"]))
    b = film_brute(n, d, c, 4.0, beam, (1.0, 0.0))
    ref = film_brute(n[[0, 2, 3]], d[[0, 2, 3]], ["i", "i", "i"], 4.0, beam, (1.0, 0.0))
    assert np.allclose(np.ravel([r["reflectance"], r["transmittance"], r.detected]), b[:3], rtol=1e-8)
    assert np.isclose(r.apparent_absorbance, -np.log10(b[2] / ref[2]), rtol=1e-8)


def test_film_path_factor_is_the_aplanatic_one():
    assert_rows(focused.film_path_factor())


def test_empty_or_transparent_film_stack_has_zero_absorbance():
    assert_rows(focused.transparent_film_zero_absorbance())


def test_film_koehler_equals_coherent_and_beams_batch():
    assert_rows(focused.film_koehler_equals_coherent())


# ---------------------------------------------------------------- spheres

@pytest.mark.parametrize("offset", focused.SPHERE_OFFSETS)
def test_sphere_against_amplitude_matrix_superposition(offset):
    assert_rows(focused.sphere_amplitude_superposition(offset))


def test_sphere_axis_path_equals_general_path():
    assert_rows(focused.sphere_axis_path())


def test_mixed_models_in_one_call():
    """A list of beams may mix the vector and scalar models: each keeps its own per-order shares."""
    radii, n = SPHERE
    beams = [ps.FocusedBeam(0.5, 0.23), ps.FocusedBeam(0.5, 0.23, model="scalar"),
             ps.FocusedBeam(0.5, 0.23, illumination="kohler", field_stop=30.0)]
    r = F.focused_spheres(radii, n, [2.5, 4.0], beams)
    assert r.detected.shape == (3, 2)
    vector, scalar, kohler = r.multipoles["extinction"]
    assert vector.shape[1] == 2 and scalar.shape[1] == 1 and kohler.shape == vector.shape
    assert r.multipoles["orders"][1][0] == 0 and r.multipoles["orders"][0][0] == 1
    for b, beam in enumerate(beams):
        one = F.focused_spheres(radii, n, [2.5, 4.0], beam)
        assert np.allclose(r.detected[b], one.detected) and np.allclose(r.multipoles["extinction"][b], one.multipoles["extinction"])
    c = F.focused_cylinders(CYLINDER[0], CYLINDER[1], 3.0, [ps.FocusedBeam(0.5, 0.23), ps.FocusedBeam(0.5, 0.23, model="scalar")],
                            m_max=14)
    assert c.detected.shape == (2, 1) and not c.by_polarization


def test_sphere_energy_and_shares():
    radii, n = SPHERE
    beam = ps.FocusedBeam(0.5, 0.23, offset=(0.4, 0.1, -0.2))
    lossless = F.focused_spheres(radii, [1.5, 1.3, 1.0], [2.5, 4.0], beam)
    assert np.allclose(lossless["absorption"], 0, atol=1e-13)
    r = F.focused_spheres(radii, n, [2.5, 4.0], beam)
    assert np.all(r["absorption"] > 0)
    assert np.allclose(r.multipoles["extinction"].sum(axis=(1, 2)), r["extinction"])
    assert np.allclose(r.multipoles["scattering"].sum(axis=(1, 2)), r["scattering"])


def test_sphere_index_matched_is_invisible_and_na_limit_is_plane_wave():
    assert_rows(focused.sphere_invisible_and_na_limit())


def test_sphere_koehler_against_plane_wave_cone_average():
    assert_rows(focused.sphere_koehler_cone_average())


def scalar_sphere_coefficient(l, x, m):
    """U = j_l + T h_l outside a homogeneous sphere, U and dU/dr continuous (spherical Bessel)."""
    j, jd = spherical_jn(l, x), spherical_jn(l, x, True)
    h = j + 1j * spherical_yn(l, x)
    hd = jd + 1j * spherical_yn(l, x, True)
    jp, jpd = spherical_jn(l, m * x), spherical_jn(l, m * x, True)
    return (m * jpd * j - jp * jd) / (jp * hd - m * h * jpd)


def test_scalar_sphere_coefficients_and_energy():
    x, m = 4.3, 1.4 + 0.05j
    sol = ps.solve([x], [m, 1.0], 2 * np.pi)
    assert np.allclose(sol.t[TE, 0], [scalar_sphere_coefficient(l, x, m) for l in sol.orders], rtol=1e-10)
    assert np.isclose(F.monopole_coefficient([x], [m, 1.0], 2 * np.pi)[0], scalar_sphere_coefficient(0, x, m), rtol=1e-12)
    # a two-layer monopole against direct 2x2 matching with spherical Bessel functions
    radii, n = [1.0, 1.7], [1.6 + 0.1j, 1.2, 1.0]
    k = 2 * np.pi * np.array(n) / 2.5
    def j(derivative, z):
        return spherical_jn(0, z, derivative)

    def y(derivative, z):
        return spherical_yn(0, z, derivative)

    def h(derivative, z):
        return j(derivative, z) + 1j * y(derivative, z)

    M = np.array([[j(0, k[0] * 1.0), -j(0, k[1] * 1.0), -y(0, k[1] * 1.0), 0],
                  [k[0] * j(1, k[0] * 1.0), -k[1] * j(1, k[1] * 1.0), -k[1] * y(1, k[1] * 1.0), 0],
                  [0, j(0, k[1] * 1.7), y(0, k[1] * 1.7), -h(0, k[2] * 1.7)],
                  [0, k[1] * j(1, k[1] * 1.7), k[1] * y(1, k[1] * 1.7), -k[2] * h(1, k[2] * 1.7)]], complex)
    rhs = np.array([0, 0, j(0, k[2] * 1.7), k[2] * j(1, k[2] * 1.7)], complex)
    assert np.isclose(np.linalg.solve(M, rhs)[3], F.monopole_coefficient(radii, n, 2.5)[0], rtol=1e-12)
    beam = ps.FocusedBeam(0.5, 0.23, model="scalar", offset=(0.3, 0.0, 0.1))
    r = F.focused_spheres(radii, [1.6, 1.2, 1.0], [2.5, 4.0], beam)
    assert np.allclose(r["absorption"], 0, atol=1e-13)
    r = F.focused_spheres(radii, n, [2.5, 4.0], beam)
    assert np.all(r["absorption"] > 0) and r.multipoles["extinction"].shape[-2:] == (1, r.multipoles["orders"].size)


def test_scalar_sphere_against_its_own_brute_force_sum():
    radii, n, lam = [1.0, 1.7], [1.6 + 0.1j, 1.2 + 0.01j, 1.0], 2.5
    sol = ps.solve(radii, n, lam)
    tl = np.concatenate((F.monopole_coefficient(radii, n, lam), sol.t[TE, 0]))
    l = np.arange(tl.size)
    k = 2 * np.pi / lam
    for offset in ((0.0, 0.0, 0.4), (0.6, 0.2, 0.0)):
        beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, collection_obscuration=0.0, model="scalar", offset=offset)
        r = F.focused_spheres(radii, n, lam, beam)
        u_in, w_in = pupil_grid(0.2, 0.6, 24, 40)
        a = F.aplanatic_amplitude(u_in[:, 2], 0.2, 0.6) * np.exp(-1j * k * u_in @ np.array(offset))

        def scattered(u):  # G = (1/2pi) sum (2l+1) T_l Int P_l(u.u') A dOmega'
            P = np.polynomial.legendre.legval(np.clip(u @ u_in.T, -1, 1), (2 * l + 1) * tl)
            return P @ (a * w_in) / (2 * np.pi)

        u_e, w_e = pupil_grid(0.2, 0.6, 24, 40, shift=0.0)
        a_e = F.aplanatic_amplitude(u_e[:, 2], 0.2, 0.6) * np.exp(-1j * k * u_e @ np.array(offset))
        assert np.isclose(r["extinction"], -2 * np.real(np.conj(a_e) * scattered(u_e) @ w_e), rtol=1e-8)
        u_all, w_all = pupil_grid(0.0, 1.0, 40, 48, shift=0.25)
        u_all, w_all = np.concatenate((u_all, u_all * [1, 1, -1])), np.concatenate((w_all, w_all))
        assert np.isclose(r["scattering"], np.abs(scattered(u_all)) ** 2 @ w_all, rtol=1e-8)
        u_c, w_c = collection_grid((0.2, 0.6), (0.0, 0.8), 24, 40)
        s = np.hypot(u_c[:, 0], u_c[:, 1])
        a_c = np.where((s >= 0.2) & (s <= 0.6), F.aplanatic_amplitude(u_c[:, 2], 0.2, 0.6), 0) * np.exp(-1j * k * u_c @ np.array(offset))
        assert np.isclose(r.detected, np.abs(a_c + scattered(u_c)) ** 2 @ w_c, rtol=1e-8)


# ---------------------------------------------------------------- cylinders

def test_cylinder_rows_against_plane_wave_sums():
    assert_rows(focused.cylinder_rows())


def test_cylinder_whole_beam_against_fixed_quadrature():
    assert_rows(focused.cylinder_whole_beam())


def test_cylinder_energy_and_na_limit():
    assert_rows(focused.cylinder_energy_and_na_limit())


def test_cylinder_koehler_against_plane_wave_cone_average():
    assert_rows(focused.cylinder_koehler_cone_average())


def test_scalar_cylinder_coefficient_energy_and_brute_force():
    assert_rows(focused.scalar_cylinder())


# ---------------------------------------------------------------- near fields

def test_sphere_near_field_against_plane_wave_near_fields():
    assert_rows(focused.sphere_near_field())


def test_cylinder_near_field_against_plane_wave_fields():
    assert_rows(focused.cylinder_near_field())


def test_film_near_field_against_position_resolved_and_parseval():
    assert_rows(focused.film_near_field_and_parseval())


# ---------------------------------------------------------------- the problem model

def test_solve_problem_focused_and_validation():
    beam = ps.FocusedBeam(0.5, 0.23)
    film = ps.Problem("films", [inf, 1.0, inf], [1.0, 1.5 + 0.02j, 1.0], 3.0, beam, coherence=("i", "c", "i"))
    reference = ps.Problem("films", [inf, inf], [1.0, 1.0], 3.0, beam)
    out = ps.solve_problem(film, reference=reference)
    direct = F.focused_films([1.0, 1.5 + 0.02j, 1.0], [inf, 1.0, inf], 3.0, beam, reference=([1.0, 1.0], [inf, inf]))
    assert np.isclose(out["apparent_absorbance"], direct.apparent_absorbance)
    assert ps.solve_focused(film, reference=reference)["apparent_absorbance"] == out["apparent_absorbance"]
    sphere = ps.solve_problem(ps.Problem("spheres", [1.0], [1.4 + 0.01j, 1.0], 3.0, beam), ("rates", "field"),
                              points=np.zeros((1, 3)))
    assert sphere["field"].e.shape == (2, 1, 3)
    cylinder = ps.solve_problem(ps.Problem("cylinders", [1.0], [1.4 + 0.01j, 1.0], 3.0, beam))
    assert set(cylinder["by_polarization"]) == {"x", "y"}
    for bad in (dict(na=0.5, obscuration=0.5), dict(na=0.5, illumination="kohler"),
                dict(na=0.5, illumination="kohler", field_stop=10, model="scalar"),
                dict(na=0.5, mode="reflection", collection_na=0.4), dict(na=0.5, offset=(0, 0)),
                dict(na=0.5, field_stop=10.0), dict(na=0.5, collection_na=0.2, collection_obscuration=0.3)):
        with pytest.raises(ValueError):
            ps.FocusedBeam(**bad)
    with pytest.raises(ValueError):  # NA beyond the host
        ps.Problem("spheres", [1.0], [1.4, 1.0], 3.0, ps.FocusedBeam(1.2))
    with pytest.raises(ValueError):  # the condenser beyond the exit medium
        ps.Problem("films", [inf, 1.0, inf], [1.5, 1.4, 1.0], 3.0, ps.FocusedBeam(1.2, collection_na=1.1))
    with pytest.raises(ValueError):
        ps.Problem("cylinders", [1.0], [1.4, 1.0], 3.0, ps.FocusedBeam(0.5, mode="reflection"))
    with pytest.raises(ValueError):
        ps.Problem("films", [inf, 1.0, inf], [1.0, 1.5, 1.0], 3.0, ps.PointDipole(0.5, layer=1), coherence=("i", "i", "i"))
