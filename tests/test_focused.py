"""Focused illumination: every semi-analytic form against an independent Python path.

Brute force here is a plane-wave superposition on a dense pupil grid through the
existing engines (amplitude_matrix, oblique solve_cylinder, coh_tmm/position_resolved,
near_field, CylinderSolution.field), called, never re-derived.
"""

import numpy as np
import pytest
from scipy.special import hankel1, jv, jvp, h1vp, spherical_jn, spherical_yn

import pystratify as ps
from pystratify import focused as F
from pystratify.cylindrical import solve_cylinder, cross_widths
from pystratify.planar import coh_tmm, layered_response, position_resolved, find_in_structure_with_inf

inf = np.inf
TM, TE = ps.TM, ps.TE


# ---------------------------------------------------------------- brute-force helpers

def pupil_grid(s1, s2, n_theta, n_phi, shift=0.5):
    x, w = F._nodes(np.sqrt(1 - s2 * s2), np.sqrt(1 - s1 * s1), n_theta)
    phi = 2 * np.pi * (np.arange(n_phi) + shift) / n_phi
    X, P = np.meshgrid(x, phi, indexing="ij")
    st = np.sqrt(1 - X * X)
    u = np.stack((st * np.cos(P), st * np.sin(P), X), -1).reshape(-1, 3)
    return u, (w[:, None] * np.full(n_phi, 2 * np.pi / n_phi)[None]).reshape(-1)


def collection_grid(illumination, collection, n_theta, n_phi, shift=0.25):
    (s1, s2), (c1, c2) = illumination, collection
    us, ws = [], []
    for a, b in F._cuts(c1, c2, (s1, s2)):
        u, w = pupil_grid(a, b, n_theta, n_phi, shift)
        us.append(u)
        ws.append(w)
    return np.concatenate(us), np.concatenate(ws)


def pupil_field(u, illumination, jones, k, offset):
    """A(u) of the normalized aplanatic beam, lab vectors."""
    s1, s2 = illumination
    s = np.hypot(u[:, 0], u[:, 1])
    a = np.where((s >= s1 - 1e-12) & (s <= s2 + 1e-12), F.aplanatic_amplitude(u[:, 2], s1, s2), 0.0)
    return a[:, None] * F._pupil_polarization(u, jones) * np.exp(-1j * k * u @ np.asarray(offset, float))[:, None]


def sphere_scattered(sol, u_out, u_in, A_in, w_in):
    """G_sca(u_out) = -(1/2pi) Int [S2 (A.e_par_i) e_par_s + S1 (A.e_perp) e_perp] dOmega' (units 2pi/(ik))."""
    cos = np.clip(u_out @ u_in.T, -1, 1)  # (O, I)
    S1, S2, _, _ = ps.amplitude_matrix(sol, np.arccos(cos).ravel())
    S1, S2 = S1[0].reshape(cos.shape), S2[0].reshape(cos.shape)
    cross = np.cross(u_out[:, None], u_in[None])
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    perp = cross / norm
    par_in = np.cross(u_in[None], perp)
    par_out = np.cross(u_out[:, None], perp)
    a_par = np.sum(A_in[None] * par_in, -1) * w_in
    a_perp = np.sum(A_in[None] * perp, -1) * w_in
    return -(np.einsum("oi,oi,oic->oc", S2, a_par, par_out) + np.einsum("oi,oi,oic->oc", S1, a_perp, perp)) / (2 * np.pi)


def brute_sphere(radii, n, lam, beam, jones=(1.0, 0.0)):
    sol = ps.solve(radii, n, lam)
    host = np.real(n[-1])
    k = 2 * np.pi * host / lam
    ill = (beam.obscuration / host, beam.na / host)
    col = tuple(v / host for v in beam.collection)
    u_in, w_in = pupil_grid(*ill, 24, 40)
    A_in = pupil_field(u_in, ill, jones, k, beam.offset)
    u_ext, w_ext = pupil_grid(*ill, 24, 40, shift=0.0)
    G = sphere_scattered(sol, u_ext, u_in, A_in, w_in)
    A_ext = pupil_field(u_ext, ill, jones, k, beam.offset)
    ext = -2 * np.real(np.sum(np.conj(A_ext) * G, -1) @ w_ext)
    u_all, w_all = pupil_grid(0.0, 1.0, 40, 48, shift=0.25)
    u_all = np.concatenate((u_all, u_all * [1, 1, -1]))
    w_all = np.concatenate((w_all, w_all))
    sca = np.sum(np.abs(sphere_scattered(sol, u_all, u_in, A_in, w_in)) ** 2, -1) @ w_all
    u_c, w_c = collection_grid(ill, col, 24, 40)
    total = pupil_field(u_c, ill, jones, k, beam.offset) + sphere_scattered(sol, u_c, u_in, A_in, w_in)
    det = np.sum(np.abs(total) ** 2, -1) @ w_c
    ref = np.sum(np.abs(pupil_field(u_c, ill, jones, k, beam.offset)) ** 2, -1) @ w_c
    return dict(extinction=ext, scattering=sca, detected=det, reference=ref)


def row_brute(radii, n, lam, sy, beam, jones, m_max):
    """One pupil row of a cylinder, signed sy: every plane wave alpha solved and its far field summed
    on observation angles, with powers by direct quadrature (no Fourier bookkeeping)."""
    host = np.real(n[-1])
    k = 2 * np.pi * host / lam
    ill = (beam.obscuration / host, beam.na / host)
    col = tuple(v / host for v in beam.collection)
    sol = solve_cylinder(radii, n, lam, beta=k * sy, m_max=m_max)
    intervals = F._row_intervals(abs(sy), *ill)
    alpha = np.concatenate([F._nodes(a, b, 60)[0] for a, b in intervals])
    w = np.concatenate([F._nodes(a, b, 60)[1] for a, b in intervals])
    u, e_n, e_m = F._row_basis(alpha, sy)
    A = pupil_field(u, ill, jones, k, beam.offset)
    c_n, c_m = np.sum(A * e_n, -1), np.sum(A * e_m, -1)

    def scattered(phi):  # (1/pi) sum over plane waves of their rotated far fields, on (e_N, e_M)
        out = np.zeros((2, phi.size), complex)
        for cn, cm, a, wa in zip(c_n, c_m, alpha, w):
            b = sol.t @ np.array([cn, 1j * cm])
            phase = np.exp(1j * np.outer(phi - a, sol.orders))
            out[0] += wa * phase @ b[:, 0]
            out[1] += wa * (-1j) * phase @ b[:, 1]
        return out / np.pi

    def beam_at(phi):
        u, e_n, e_m = F._row_basis(phi, sy)
        A = pupil_field(u, ill, jones, k, beam.offset)
        return np.stack((np.sum(A * e_n, -1), np.sum(A * e_m, -1)))

    ext = 0.0
    for a, b in intervals:
        p, wq = F._nodes(a, b, 80)
        ext += -2 * np.real(np.sum(np.conj(beam_at(p)) * scattered(p), 0) @ wq)
    phi = 2 * np.pi * np.arange(256) / 256  # periodic: the trapezoid rule is spectral
    sca = np.sum(np.abs(scattered(phi)) ** 2, 0).sum() * 2 * np.pi / 256
    det = ref = 0.0
    edges = [e for a, b in intervals for e in (a, b)]
    for lo, hi in F._row_intervals(abs(sy), *col):
        for a, b in F._cuts(lo, hi, edges):
            p, wq = F._nodes(a, b, 80)
            det += np.sum(np.abs(beam_at(p) + scattered(p)) ** 2, 0) @ wq
            ref += np.sum(np.abs(beam_at(p)) ** 2, 0) @ wq
    return np.array([ext, sca, det, ref])


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

def film_brute(n, d, c, lam, beam, jones):
    """2-D pupil grid of x/y pupil plane waves, each split into s and p, through layered_response."""
    ambient = n[0].real
    ill = (beam.obscuration / ambient, beam.na / ambient)
    col = tuple(v / ambient for v in beam.collection)
    grids = [pupil_grid(a, b, 32, 16) for a, b in F._cuts(*ill, col)]  # split where the condenser's edges cross
    u, w = np.concatenate([g[0] for g in grids]), np.concatenate([g[1] for g in grids])
    A = pupil_field(u, ill, jones, 1.0, (0, 0, 0))
    out = np.zeros(4 + len(d) - 2)
    cache = {}
    for ui, Ai, wi in zip(u, A, w):
        s = np.hypot(ui[0], ui[1])
        theta = np.arcsin(s)
        if theta not in cache:
            cache[theta] = {p: layered_response(p, n, d, c, theta, lam) for p in "sp"}
        r = cache[theta]
        e_phi = np.array([-ui[1], ui[0], 0]) / s
        power_s = abs(Ai @ e_phi) ** 2
        power_p = np.sum(abs(Ai) ** 2) - power_s
        R = power_s * r["s"]["R"] + power_p * r["p"]["R"]
        T = power_s * r["s"]["T"] + power_p * r["p"]["T"]
        seen = R if beam.mode == "reflection" else T * (col[0] <= s <= col[1])
        layers = power_s * r["s"]["absorption"][1:-1] + power_p * r["p"]["absorption"][1:-1]
        out += wi * np.concatenate(([R, T, seen, 0.0], layers))
    return out


@pytest.mark.parametrize("mode", ["transmission", "reflection"])
def test_film_cone_quadrature_against_pupil_grid(mode):
    n = np.array([1.0, 1.55 + 0.03j, 1.42, 1.0 if mode == "transmission" else 0.3 + 18j])
    d = np.array([inf, 1.2, 3.0, inf])
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7 if mode == "transmission" else None,
                          collection_obscuration=0.35 if mode == "transmission" else None, mode=mode)
    r = F.focused_films(n, d, 4.0, beam)
    for jones in ((1.0, 0.0), (0.0, 1.0)):
        b = film_brute(n, d, None, 4.0, beam, jones)
        assert np.allclose(np.ravel([r["reflectance"], r["transmittance"], r.detected]), b[:3], rtol=1e-9, atol=1e-12)
        assert np.allclose(np.ravel(r["layer_absorptance"]), b[4:], rtol=1e-9)
    assert np.isclose(r["reflectance"] + r["transmittance"] + r["absorptance"], 1)
    assert np.isclose(np.sum(r["layer_absorptance"]), r["absorptance"], atol=1e-12)


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
    """Ruling 4: <1/cos t> over NA 0.23-0.5 at n = 1.5 is 1.0357 for the aplanatic pupil."""
    alpha, lam = 1e-5, np.array([3.0, 7.0])
    kappa = alpha * lam / (4 * np.pi)
    n = np.stack((np.full(2, 1.5), 1.5 + 1j * kappa, np.full(2, 1.5)), 1)
    beam = ps.FocusedBeam(0.5, 0.23)  # n sin t in the n = 1.5 medium, as the film's internal cone
    r = F.focused_films(n, [inf, 1.0, inf], lam, beam, reference=([1.5, 1.5], [inf, inf]))
    assert np.allclose(r.apparent_absorbance * np.log(10) / alpha, 1.0357, atol=1e-4)


def test_empty_or_transparent_film_stack_has_zero_absorbance():
    beam = ps.FocusedBeam(0.5, 0.23)
    r = F.focused_films([1.0, 1.0, 1.0], [inf, 2.0, inf], [3.0, 5.0], beam, reference=([1.0, 1.0], [inf, inf]))
    assert np.allclose(r.apparent_absorbance, 0, atol=1e-14) and np.allclose(r["transmittance"], 1)
    # a sample that is the reference's own window: identical detected powers
    n = [1.0, 1.42, 1.0]
    r = F.focused_films(n, [inf, 50.0, inf], 3.0, beam, reference=(n, [inf, 50.0, inf]))
    assert abs(r.apparent_absorbance) < 1e-14


def test_film_koehler_equals_coherent_and_beams_batch():
    n, d = [1.0, 1.55 + 0.03j, 1.0], [inf, 1.2, inf]
    beams = [ps.FocusedBeam(0.5, 0.23), ps.FocusedBeam(0.5, 0.23, illumination="kohler", field_stop=40.0),
             ps.FocusedBeam(0.3)]
    r = F.focused_films(n, d, [3.0, 4.0], beams, reference=([1.0, 1.0], [inf, inf]))
    assert r.detected.shape == (3, 2)
    assert np.allclose(r.detected[0], r.detected[1], rtol=1e-12)
    single = F.focused_films(n, d, [3.0, 4.0], beams[2], reference=([1.0, 1.0], [inf, inf]))
    assert np.allclose(single.apparent_absorbance, r.apparent_absorbance[2])


# ---------------------------------------------------------------- spheres

SPHERE = ([0.8, 1.4], [1.5 + 0.08j, 1.3 + 0.02j, 1.0])


@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.6), (0.5, -0.3, 0.2)])
def test_sphere_against_amplitude_matrix_superposition(offset):
    radii, n = SPHERE
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, collection_obscuration=0.1, polarization="x", offset=offset)
    r = F.focused_spheres(radii, n, 3.0, beam)
    b = brute_sphere(radii, n, 3.0, beam)
    for key in ("extinction", "scattering", "detected", "reference"):
        assert np.isclose(r[key], b[key], rtol=2e-8, atol=1e-11), key


def test_sphere_axis_path_equals_general_path():
    radii, n = SPHERE
    lam = np.array([2.5, 3.0, 5.0])
    sol = ps.solve(radii, n, lam)
    k = 2 * np.pi / lam
    for z in (0.0, 0.9):
        a = F._sphere_axis(sol.t, k, sol.orders, (0.23, 0.5), (0.1, 0.7), z)
        for jones in ((1.0, 0.0), (0.0, 1.0), (np.sqrt(0.5), 1j * np.sqrt(0.5))):
            g = F._sphere_general(sol.t, k, sol.orders, (0.23, 0.5), (0.1, 0.7), (0.0, 0.0, z), jones, False)
            for key in a:
                assert np.allclose(a[key], g[key], rtol=1e-12, atol=1e-14), key


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
    r = F.focused_spheres([1.0], [1.33, 1.33], 3.0, ps.FocusedBeam(0.6, 0.2))
    assert abs(r.apparent_absorbance) < 1e-14 and abs(r["extinction"]) < 1e-14
    radii, n = SPHERE
    lam = np.array([3.0, 6.0])
    sigma = ps.cross_sections(ps.solve(radii, n, lam)).ext
    ratios = []
    for na in (0.04, 0.02):
        r = F.focused_spheres(radii, n, lam, ps.FocusedBeam(na))
        focus = (2 * np.pi / lam / (2 * np.pi)) ** 2 * F.focal_amplitude((0.0, na)) ** 2  # |E(0)|^2 per beam power
        ratios.append(r["extinction"] / focus / sigma)
    assert np.all(np.abs(ratios[1] - 1) < np.abs(ratios[0] - 1)) and np.allclose(ratios[1], 1, atol=1e-3)


def test_sphere_koehler_against_plane_wave_cone_average():
    radii, n = SPHERE
    lam = 3.0
    D = 25.0
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7, collection_obscuration=0.3, illumination="kohler", field_stop=D)
    r = F.focused_spheres(radii, n, lam, beam)
    sol = ps.solve(radii, n, lam)
    cs = ps.cross_sections(sol)
    k = 2 * np.pi / lam
    G = np.pi * D * D / 4
    grids = [pupil_grid(a, b, 30, 1) for a, b in F._cuts(0.2, 0.6, (0.3,))]  # one azimuth (symmetry); split at the condenser
    u_in, w_in = np.concatenate([g[0] for g in grids]), np.concatenate([g[1] for g in grids])
    power = np.abs(pupil_field(u_in, (0.2, 0.6), (1.0, 0.0), k, (0, 0, 0))) ** 2
    dP = np.sum(power, -1) * w_in  # power per incident direction (all azimuths)
    irradiance = dP / (G * u_in[:, 2])
    u_c, w_c = collection_grid((0.2, 0.6), (0.3, 0.7), 30, 64)
    landed = []
    for ui in u_in:
        S1, S2, _, _ = ps.amplitude_matrix(sol, np.arccos(np.clip(u_c @ ui, -1, 1)))
        landed.append(((np.abs(S1[0]) ** 2 + np.abs(S2[0]) ** 2) / (2 * k * k)) @ w_c)
    s = np.hypot(u_in[:, 0], u_in[:, 1])
    seen = (s >= 0.3) & (s <= 0.7)
    detected = np.sum(dP * seen) - cs.ext[0] * np.sum(irradiance * seen) + np.sum(irradiance * np.array(landed))
    assert np.isclose(r["extinction"], cs.ext[0] * np.sum(irradiance), rtol=1e-10)
    assert np.isclose(r["scattering"], cs.sca[0] * np.sum(irradiance), rtol=1e-10)
    assert np.isclose(r.detected, detected, rtol=1e-9)
    assert np.isclose(r.reference, np.sum(dP * seen), rtol=1e-12)


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

CYLINDER = ([0.7, 1.2], np.array([1.5 + 0.08j, 1.3 + 0.02j, 1.0]))


def test_cylinder_rows_against_plane_wave_sums():
    radii, n = CYLINDER
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, collection_obscuration=0.1, offset=(0.4, 0.0, 0.3))
    rows = F._CylinderRows(radii, n, 3.0, [beam], 14)
    for sy in (0.05, 0.3, 0.55):
        value = rows(sy) / 2
        for p, jones in enumerate(((1.0, 0.0), (0.0, 1.0))):
            brute = row_brute(radii, n, 3.0, sy, beam, jones, 14)
            assert np.allclose(value[p * rows.width:p * rows.width + 4], brute, rtol=1e-9, atol=1e-12)
            # the mirror row -sy carries the same powers
            assert np.allclose(row_brute(radii, n, 3.0, -sy, beam, jones, 14), brute, rtol=1e-10)


def test_cylinder_whole_beam_against_fixed_quadrature():
    radii, n = CYLINDER
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, offset=(0.4, 0.0, 0.3))
    r = F.focused_cylinders(radii, n, 3.0, beam, m_max=14, tolerance=1e-9)
    rows = F._CylinderRows(radii, n, 3.0, [beam], 14)
    total = 0
    for a, b in F._cuts(0, 0.6, (0.2,)):
        sy, w = F._mapped_nodes(a, b, 40)
        total = total + sum(wi * rows(s) for s, wi in zip(sy, w))
    for p, name in enumerate("xy"):
        block = total[p * rows.width:p * rows.width + 4]
        bp = r.by_polarization[name]
        assert np.allclose(np.ravel([bp["extinction"], bp["scattering"], bp["detected"], bp["reference"]]), block, rtol=1e-8)
    assert np.isclose(r.reference, 1.0, rtol=1e-8)  # the condenser sees the whole beam
    assert r.diagnostics["converged"]


def test_cylinder_energy_and_na_limit():
    radii, n = CYLINDER
    beam = ps.FocusedBeam(0.5, 0.23, offset=(0.3, 0.0, 0.2))
    lossless = F.focused_cylinders(radii, n.real, [3.0, 5.0], beam)
    for p in "xy":
        assert np.allclose(lossless.by_polarization[p]["absorption"], 0, atol=1e-12)
    r = F.focused_cylinders(radii, n, [3.0, 5.0], beam)
    assert np.all(r["absorption"] > 0)
    assert np.allclose(r.multipoles["extinction"].sum(-1), r["extinction"])
    lam = 3.0
    k = 2 * np.pi / lam
    sol = solve_cylinder(radii, n, lam, beta=0)
    for name, pol in (("x", "axial-magnetic"), ("y", "axial-electric")):
        ratios = []
        for na in (0.04, 0.02):
            r = F.focused_cylinders(radii, n, lam, ps.FocusedBeam(na, polarization=name))
            # Int |E(0, y, 0)|^2 dy per beam power = (k/2pi) Int ds_y |Int A dalpha|^2
            sy, wy = F._mapped_nodes(0, na, 60)
            line = 0
            for s, w in zip(sy, wy):
                (a, b), = F._row_intervals(s, 0, na)
                al, wa = F._nodes(a, b, 40)
                u, _, _ = F._row_basis(al, s)
                A = pupil_field(u, (0, na), (1.0, 0.0) if name == "x" else (0.0, 1.0), k, (0, 0, 0))
                line += 2 * w * np.sum(np.abs(np.sum(A * wa[:, None], 0)) ** 2)
            ratios.append(r["extinction"] / (k / (2 * np.pi) * line) / cross_widths(sol, pol)["extinction"])
        assert abs(ratios[1] - 1) < abs(ratios[0] - 1) and abs(ratios[1] - 1) < 2e-3


def test_cylinder_koehler_against_plane_wave_cone_average():
    radii, n = CYLINDER
    lam, D = 3.0, 30.0
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7, collection_obscuration=0.3, illumination="kohler", field_stop=D)
    r = F.focused_cylinders(radii, n, lam, beam, m_max=14, tolerance=1e-9)
    k = 2 * np.pi / lam
    G = np.pi * D * D / 4
    totals = {p: np.zeros(3) for p in "xy"}
    for a, b in F._cuts(0, 0.6, (0.2, 0.3)):
        sys_, wys = F._mapped_nodes(a, b, 30)
        for sy, wy in zip(sys_, wys):
            sol = solve_cylinder(radii, n, lam, beta=k * sy, m_max=14)
            edges = [e for a_, b_ in F._row_intervals(sy, 0.3, 0.7) for e in (a_, b_)]
            for lo, hi in [c for a_, b_ in F._row_intervals(sy, 0.2, 0.6) for c in F._cuts(a_, b_, edges)]:
                al, wa = F._nodes(lo, hi, 30)
                u, e_n, e_m = F._row_basis(al, sy)
                for name, jones in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
                    e = F._pupil_polarization(u, jones)
                    for ui, cn, cm, wai in zip(u, np.sum(e * e_n, -1), np.sum(e * e_m, -1), wa):
                        b_ = sol.t @ np.array([cn, 1j * cm])
                        c_ext = -(4 / k) * np.real(np.sum(cn * b_[:, 0] - 1j * cm * b_[:, 1]))
                        # dC/dphi of this wave, observed at its own row, rotated to its incidence angle
                        alpha = np.arctan2(ui[0], ui[2])
                        landed = 0
                        for clo, chi in F._row_intervals(sy, 0.3, 0.7):
                            ph, wp = F._nodes(clo, chi, 120)
                            amp = np.exp(1j * np.outer(ph - alpha, sol.orders)) @ b_
                            landed += (2 / (np.pi * k)) * np.sum(np.abs(amp) ** 2, 1) @ wp
                        s = np.hypot(ui[0], ui[1])
                        seen = 0.3 <= s <= 0.7
                        weight = 2 * wy * wai * F.aplanatic_amplitude(ui[2], 0.2, 0.6) ** 2  # dP, both +-sy
                        irradiance = weight / (G * ui[2]) * D  # times the lit chord
                        totals[name] += [irradiance * c_ext, irradiance * (landed - seen * c_ext), weight * seen]
    for name in "xy":
        bp = r.by_polarization[name]
        assert np.isclose(bp["extinction"], totals[name][0], rtol=1e-8)
        assert np.isclose(bp["detected"], totals[name][2] + totals[name][1], rtol=1e-8)
        assert np.isclose(bp["reference"], totals[name][2], rtol=1e-8)


def scalar_cylinder_coefficient(m, x_out, x_in, n_out, n_in):
    """U = J_m + t H_m outside a homogeneous cylinder, U and dU/drho continuous (MATLAB cyl_exp_coef's a_m)."""
    J, dJ, H, dH = jv(m, x_out), jvp(m, x_out), hankel1(m, x_out), h1vp(m, x_out)
    Jp, dJp = jv(m, x_in), jvp(m, x_in)
    return (n_in * dJp * J - n_out * Jp * dJ) / (n_out * dH * Jp - n_in * H * dJp)


def test_scalar_cylinder_coefficient_energy_and_brute_force():
    lam, R, m_in = 3.0, 1.1, 1.45 + 0.06j
    k = 2 * np.pi / lam
    sy = 0.4
    n_v, n_p = np.sqrt(1 - sy * sy), np.sqrt(m_in ** 2 - sy * sy)
    t = solve_cylinder([R], [n_p, n_v], lam, beta=0, m_max=12).t[:, 0, 0]
    expected = [scalar_cylinder_coefficient(abs(m), k * R * n_v, k * R * n_p, n_v, n_p) for m in range(-12, 13)]
    assert np.allclose(t, expected, rtol=1e-10)
    radii, n = CYLINDER
    beam = ps.FocusedBeam(0.6, 0.2, model="scalar", offset=(0.3, 0.0, 0.2))
    lossless = F.focused_cylinders(radii, n.real, [3.0, 5.0], beam)
    assert np.allclose(lossless["absorption"], 0, atol=1e-12)
    r = F.focused_cylinders(radii, n, 3.0, beam, m_max=14, tolerance=1e-9)
    assert r["absorption"] > 0 and not r.by_polarization
    # one row by direct superposition of scalar plane waves on its observation angles
    rows = F._CylinderRows(radii, n, 3.0, [beam], 14)
    for sy in (0.1, 0.45):
        value = rows(sy)[:4] / 2
        eff = np.sqrt(n ** 2 - sy ** 2)
        ts = solve_cylinder(radii, eff, 3.0, beta=0, m_max=14).t[:, 0, 0]
        orders = np.arange(-14, 15)
        intervals = F._row_intervals(sy, 0.2, 0.6)
        al = np.concatenate([F._nodes(a, b, 60)[0] for a, b in intervals])
        wa = np.concatenate([F._nodes(a, b, 60)[1] for a, b in intervals])

        def amplitude(phi):
            u, _, _ = F._row_basis(phi, sy)
            s = np.hypot(u[:, 0], u[:, 1])
            return np.where((s >= 0.2) & (s <= 0.6), F.aplanatic_amplitude(u[:, 2], 0.2, 0.6), 0) * np.exp(
                -1j * k * u @ np.array(beam.offset))

        U = amplitude(al) * wa

        def scattered(phi):  # (1/pi) sum over the row's plane waves of their rotated scattered amplitudes
            return np.array([np.sum(U * (np.exp(1j * np.outer(p - al, orders)) @ ts)) for p in phi]) / np.pi

        ph = 2 * np.pi * np.arange(256) / 256
        assert np.isclose(value[1], np.sum(np.abs(scattered(ph)) ** 2) * 2 * np.pi / 256, rtol=1e-9)
        ext = sum(-2 * np.real(np.conj(amplitude(p)) * scattered(p) @ wq) for p, wq in (F._nodes(a, b, 80) for a, b in intervals))
        assert np.isclose(value[0], ext, rtol=1e-9)


# ---------------------------------------------------------------- near fields

def test_sphere_near_field_against_plane_wave_near_fields():
    radii, n, lam = [0.6, 1.2], [1.5 + 0.1j, 1.3 + 0.02j, 1.0], 3.0
    sol = ps.solve(radii, n, lam, regime="near")
    beam = ps.FocusedBeam(0.6, 0.2, offset=(0.4, -0.3, 0.25), polarization="y")
    points = np.array([[0.1, 0.2, 0.3], [0.9, 0.1, -0.2], [1.6, 0.4, 0.5], [2.5, -1.0, -1.5]])
    field = F.focused_field_spheres(radii, n, lam, beam, points, l_max=sol.orders[-1])
    k = 2 * np.pi / lam
    u, w = pupil_grid(0.2, 0.6, 36, 56)
    A = pupil_field(u, (0.2, 0.6), (0.0, 1.0), k, beam.offset)
    E, H = np.zeros((4, 3), complex), np.zeros((4, 3), complex)
    for ui, Ai, wi in zip(u, A, w):
        e = F._pupil_polarization(ui[None], (0.0, 1.0))[0]  # real unit polarization; A = amplitude e
        amplitude = Ai @ e
        R = np.stack((e, np.cross(ui, e), ui), axis=1)
        p = points @ R
        nf = ps.near_field(sol, p[:, 0], p[:, 1], p[:, 2])
        E += wi * amplitude * np.stack([nf.e[c] for c in "xyz"], -1) @ R.T
        H += wi * amplitude * np.stack([nf.h[c] for c in "xyz"], -1) @ R.T
    unit = F.focal_amplitude((0.2, 0.6))
    assert np.allclose(field.e[0], E / unit, rtol=1e-8, atol=1e-9)
    assert np.allclose(field.h[0], H / unit, rtol=1e-8, atol=1e-9)
    assert np.all(np.isnan(field.e_incident[0, :2])) and np.all(np.isfinite(field.e_incident[0, 2:]))


def test_cylinder_near_field_against_plane_wave_fields():
    radii, n, lam = [0.6, 1.2], np.array([1.5 + 0.1j, 1.3 + 0.02j, 1.0]), 3.0
    beam = ps.FocusedBeam(0.6, 0.2, offset=(0.3, 0.2, 0.25), polarization="x")
    points = np.array([[0.1, 0.0, 0.3], [0.9, 0.1, -0.2], [-0.5, 0.8, 1.9]])
    field = F.focused_field_cylinders(radii, n, lam, beam, points, m_max=14)
    k = 2 * np.pi / lam
    cyl = np.stack((points[:, 2], points[:, 0], points[:, 1]), -1)
    E = np.zeros((3, 3), complex)
    for a, b in F._cuts(-0.6, 0.6, (-0.2, 0.2)):
        for sy, wy in zip(*F._mapped_nodes(a, b, 40)):
            sol = solve_cylinder(radii, n, lam, beta=k * sy, m_max=14)
            for lo, hi in F._row_intervals(abs(sy), 0.2, 0.6):
                al, wa = F._nodes(lo, hi, 30)
                u, e_n, e_m = F._row_basis(al, sy)
                A = pupil_field(u, (0.2, 0.6), (1.0, 0.0), k, beam.offset)
                for alpha, cn, cm, wi in zip(al, np.sum(A * e_n, -1), np.sum(A * e_m, -1), wa):
                    c, s = np.cos(alpha), np.sin(alpha)
                    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    p = cyl @ Rz
                    En, _ = sol.field(p, "axial-electric")
                    Em, _ = sol.field(p, "axial-magnetic")
                    Ec = (cn * En + cm * Em) @ Rz.T
                    E += wi * wy * np.stack((Ec[:, 1], Ec[:, 2], Ec[:, 0]), -1)
    assert np.allclose(field.e[0], E / F.focal_amplitude((0.2, 0.6)), rtol=1e-7, atol=1e-8)


def test_film_near_field_against_position_resolved_and_parseval():
    n, d, lam = np.array([1.0, 1.6 + 0.05j, 1.42, 1.0]), np.array([inf, 0.8, 2.0, inf]), 3.0
    beam = ps.FocusedBeam(0.6, 0.2, offset=(0.3, -0.2, 0.5))
    points = np.array([[0.1, 0.0, -0.7], [0.9, 0.4, 0.3], [-0.6, 0.2, 1.5], [0.3, -0.8, 3.2]])
    field = F.focused_field_films(n, d, lam, beam, points)
    k = 2 * np.pi / lam
    u, w = pupil_grid(0.2, 0.6, 48, 48)
    located = [find_in_structure_with_inf(d, z) if z >= 0 else (0, z) for z in points[:, 2]]
    for index, jones in enumerate(((1.0, 0.0), (0.0, 1.0))):
        E = np.zeros((4, 3), complex)
        cache = {}
        for ui, wi in zip(u, w):
            s = np.hypot(ui[0], ui[1])
            theta = np.arcsin(s)
            if theta not in cache:
                data = {p: coh_tmm(p, n, d, theta, lam) for p in "sp"}
                cache[theta] = {p: np.array([[position_resolved(L, z, data[p])[c] for c in ("Ex", "Ey", "Ez")]
                                             for L, z in located]) for p in "sp"}
            cf, sf = ui[0] / s, ui[1] / s
            a = F.aplanatic_amplitude(ui[2], 0.2, 0.6) * np.exp(-1j * k * ui[2] * beam.offset[2])
            amp_p, amp_s = jones[0] * cf + jones[1] * sf, -jones[0] * sf + jones[1] * cf
            local = amp_p * cache[theta]["p"] + amp_s * cache[theta]["s"]
            lab = np.stack((local[:, 0] * cf - local[:, 1] * sf, local[:, 0] * sf + local[:, 1] * cf, local[:, 2]), -1)
            lateral = np.exp(1j * k * s * ((points[:, 0] - beam.offset[0]) * cf + (points[:, 1] - beam.offset[1]) * sf))
            E += wi * a * lateral[:, None] * lab
        assert np.allclose(field.e[index], E / F.focal_amplitude((0.2, 0.6)), rtol=1e-10, atol=1e-12)
    # Parseval: the lateral integral of |E|^2 at a depth is (2pi/k)^2 Int |A|^2 |E_plane(z)|^2 / cos t dOmega
    depth = 1.5
    radius = np.linspace(0, 60 * lam, 6001)
    phi = 2 * np.pi * np.arange(8) / 8
    Rr, Pp = np.meshgrid(radius, phi, indexing="ij")
    grid = np.stack((Rr * np.cos(Pp), Rr * np.sin(Pp), np.full(Rr.shape, depth)), -1)
    centred = ps.FocusedBeam(0.6, 0.2, polarization="x")
    intensity = F.focused_field_films(n, d, lam, centred, grid).intensity_e * F.focal_amplitude((0.2, 0.6)) ** 2
    ring = np.mean(intensity, axis=1) * 2 * np.pi * radius
    cumulative = np.concatenate(([0], np.cumsum((ring[1:] + ring[:-1]) / 2 * np.diff(radius))))
    lateral = 2 * cumulative[-1] - cumulative[3000]  # the |E|^2 ~ rho^-3 tail extrapolated
    x, wx = F._nodes(np.sqrt(1 - 0.36), np.sqrt(1 - 0.04), 64)
    expected = 0
    L, z = find_in_structure_with_inf(d, depth)
    for xi, wi in zip(x, wx):
        theta = np.arccos(xi)
        plane = sum(0.5 * np.sum(np.abs([position_resolved(L, z, coh_tmm(p, n, d, theta, lam))[c] for c in ("Ex", "Ey", "Ez")]) ** 2)
                    for p in "sp")
        expected += 2 * np.pi * wi * F.aplanatic_amplitude(xi, 0.2, 0.6) ** 2 * plane / xi
    assert np.isclose(lateral, (2 * np.pi / k) ** 2 * expected, rtol=3e-4)  # unextrapolated: 4e-3 short


# ---------------------------------------------------------------- the problem model

def test_solve_problem_focused_and_validation():
    beam = ps.FocusedBeam(0.5, 0.23)
    film = ps.Problem("films", [inf, 1.0, inf], [1.0, 1.5 + 0.02j, 1.0], 3.0, beam, coherence=("i", "c", "i"))
    reference = ps.Problem("films", [inf, inf], [1.0, 1.0], 3.0, beam)
    out = ps.solve_problem(film, reference=reference)
    direct = F.focused_films([1.0, 1.5 + 0.02j, 1.0], [inf, 1.0, inf], 3.0, beam, reference=([1.0, 1.0], [inf, inf]))
    assert np.isclose(out["apparent_absorbance"], direct.apparent_absorbance)
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
