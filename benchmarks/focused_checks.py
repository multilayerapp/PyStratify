"""Focused illumination: every semi-analytic form against an independent path (benchmarks/MATRIX.md).

    python benchmarks/focused_checks.py        # prints FOCUSED_CHECKS.md (~3 min)

The independent path is a plane-wave superposition on a dense pupil grid through the existing engines
(amplitude_matrix, oblique solve_cylinder, coh_tmm / layered_response / position_resolved, near_field,
CylinderSolution.field), called, never re-derived; and limits: an index-matched particle is invisible,
NA -> 0 is the plane wave, Kohler is a plane-wave cone average. The brute-force helpers below are the
reference implementation; ``tests/test_focused.py`` imports them and asserts every row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402
from checks import close, flag, report, row  # noqa: E402
from pystratify import focused as F  # noqa: E402
from pystratify.cylindrical import cross_widths, solve_cylinder  # noqa: E402
from pystratify.planar import coh_tmm, find_in_structure_with_inf, layered_response, position_resolved  # noqa: E402

inf = np.inf
SPHERE = ([0.8, 1.4], [1.5 + 0.08j, 1.3 + 0.02j, 1.0])
CYLINDER = ([0.7, 1.2], np.array([1.5 + 0.08j, 1.3 + 0.02j, 1.0]))


# ---------------------------------------------------------------- brute-force helpers (the reference)

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


def scalar_cylinder_coefficient(m, x_out, x_in, n_out, n_in):
    """U = J_m + t H_m outside a homogeneous cylinder, U and dU/drho continuous (MATLAB cyl_exp_coef's a_m)."""
    J, dJ, H, dH = jv(m, x_out), jvp(m, x_out), hankel1(m, x_out), h1vp(m, x_out)
    Jp, dJp = jv(m, x_in), jvp(m, x_in)
    return (n_in * dJp * J - n_out * Jp * dJ) / (n_out * dH * Jp - n_in * H * dJp)


# ---------------------------------------------------------------- films

FILM_MODES = ("transmission", "reflection")


def film_cone_quadrature(mode):
    name = f"film stack, {mode}: cone quadrature against a 2D pupil grid"
    n = np.array([1.0, 1.55 + 0.03j, 1.42, 1.0 if mode == "transmission" else 0.3 + 18j])
    d = np.array([inf, 1.2, 3.0, inf])
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7 if mode == "transmission" else None,
                          collection_obscuration=0.35 if mode == "transmission" else None, mode=mode)
    r = F.focused_films(n, d, 4.0, beam)
    rows = []
    for jones in ((1.0, 0.0), (0.0, 1.0)):
        b = film_brute(n, d, None, 4.0, beam, jones)
        rows += [close(name, f"R, T, detected, Jones {jones}", np.ravel([r["reflectance"], r["transmittance"], r.detected]), b[:3], 1e-9, 1e-12),
                 close(name, f"layer absorptance, Jones {jones}", np.ravel(r["layer_absorptance"]), b[4:], 1e-9, 1e-8)]
    return rows + [close(name, "R + T + A against 1", r["reflectance"] + r["transmittance"] + r["absorptance"], 1, 1e-5, 1e-8),
                   close(name, "sum of layer absorptances against A", np.sum(r["layer_absorptance"]), r["absorptance"], 1e-5, 1e-12)]


def film_path_factor():
    """Ruling 4: <1/cos t> over NA 0.23-0.5 at n = 1.5 is 1.0357 for the aplanatic pupil."""
    alpha, lam = 1e-5, np.array([3.0, 7.0])
    kappa = alpha * lam / (4 * np.pi)
    n = np.stack((np.full(2, 1.5), 1.5 + 1j * kappa, np.full(2, 1.5)), 1)
    beam = ps.FocusedBeam(0.5, 0.23)
    r = F.focused_films(n, [inf, 1.0, inf], lam, beam, reference=([1.5, 1.5], [inf, inf]))
    return [close("weak absorber under an aplanatic NA 0.23-0.5 cone", "apparent absorbance ln10 / (alpha d) against 1.0357",
                  r.apparent_absorbance * np.log(10) / alpha, 1.0357, 1e-5, 1e-4)]


def transparent_film_zero_absorbance():
    beam = ps.FocusedBeam(0.5, 0.23)
    r = F.focused_films([1.0, 1.0, 1.0], [inf, 2.0, inf], [3.0, 5.0], beam, reference=([1.0, 1.0], [inf, inf]))
    n = [1.0, 1.42, 1.0]
    same = F.focused_films(n, [inf, 50.0, inf], 3.0, beam, reference=(n, [inf, 50.0, inf]))
    name = "an empty stack, and a sample that is its own reference"
    return [row(name, "max |apparent absorbance| (empty)", np.abs(r.apparent_absorbance), 1e-14),
            close(name, "transmittance against 1 (empty)", r["transmittance"], 1, 1e-5, 1e-8),
            row(name, "|apparent absorbance| (own reference)", abs(same.apparent_absorbance), 1e-14, "<")]


def film_koehler_equals_coherent():
    n, d = [1.0, 1.55 + 0.03j, 1.0], [inf, 1.2, inf]
    beams = [ps.FocusedBeam(0.5, 0.23), ps.FocusedBeam(0.5, 0.23, illumination="kohler", field_stop=40.0), ps.FocusedBeam(0.3)]
    r = F.focused_films(n, d, [3.0, 4.0], beams, reference=([1.0, 1.0], [inf, inf]))
    single = F.focused_films(n, d, [3.0, 4.0], beams[2], reference=([1.0, 1.0], [inf, inf]))
    name = "film: Kohler against coherent; beams in one call"
    return [flag(name, "detected has shape (beams, wavelengths)", r.detected.shape == (3, 2)),
            close(name, "coherent detected against Kohler", r.detected[0], r.detected[1], 1e-12, 1e-8),
            close(name, "a beam alone against the same beam in a batch", single.apparent_absorbance, r.apparent_absorbance[2], 1e-5, 1e-8)]


# ---------------------------------------------------------------- spheres

SPHERE_OFFSETS = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.6), (0.5, -0.3, 0.2))


def sphere_amplitude_superposition(offset):
    radii, n = SPHERE
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, collection_obscuration=0.1, polarization="x", offset=offset)
    r = F.focused_spheres(radii, n, 3.0, beam)
    b = brute_sphere(radii, n, 3.0, beam)
    name = f"coated sphere, focus offset {offset}: against amplitude-matrix superposition"
    return [close(name, key, r[key], b[key], 2e-8, 1e-11) for key in ("extinction", "scattering", "detected", "reference")]


def sphere_axis_path():
    radii, n = SPHERE
    lam = np.array([2.5, 3.0, 5.0])
    sol = ps.solve(radii, n, lam)
    k = 2 * np.pi / lam
    rows = []
    for z in (0.0, 0.9):
        a = F._sphere_axis(sol.t, k, sol.orders, (0.23, 0.5), (0.1, 0.7), z)
        for jones in ((1.0, 0.0), (0.0, 1.0), (np.sqrt(0.5), 1j * np.sqrt(0.5))):
            g = F._sphere_general(sol.t, k, sol.orders, (0.23, 0.5), (0.1, 0.7), (0.0, 0.0, z), jones, False)
            rows += [close(f"sphere, focus on the axis at z = {z:g}, Jones {np.round(jones, 3).tolist()}", f"axis path {key} against the general path",
                           a[key], g[key], 1e-12, 1e-14) for key in a]
    return rows


def sphere_invisible_and_na_limit():
    r = F.focused_spheres([1.0], [1.33, 1.33], 3.0, ps.FocusedBeam(0.6, 0.2))
    radii, n = SPHERE
    lam = np.array([3.0, 6.0])
    sigma = ps.cross_sections(ps.solve(radii, n, lam)).ext
    ratios = []
    for na in (0.04, 0.02):
        out = F.focused_spheres(radii, n, lam, ps.FocusedBeam(na))
        focus = (2 * np.pi / lam / (2 * np.pi)) ** 2 * F.focal_amplitude((0.0, na)) ** 2  # |E(0)|^2 per beam power
        ratios.append(out["extinction"] / focus / sigma)
    name = "sphere: index-matched, and NA -> 0"
    return [row(name, "|apparent absorbance| (index-matched)", abs(r.apparent_absorbance), 1e-14, "<"),
            row(name, "|extinction| (index-matched)", abs(r["extinction"]), 1e-14, "<"),
            flag(name, "extinction / (|E(0)|^2 sigma_ext) closer to 1 at NA 0.02 than 0.04", np.all(np.abs(ratios[1] - 1) < np.abs(ratios[0] - 1))),
            close(name, "extinction / (|E(0)|^2 sigma_ext) at NA 0.02 against 1", ratios[1], 1, 1e-5, 1e-3)]


def sphere_koehler_reference(lam, D=25.0):
    """Kohler illumination of SPHERE (NA 0.6/0.2, collection 0.7/0.3) as a cone average of plane waves:
    {extinction, scattering, detected, reference}."""
    radii, n = SPHERE
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
    return dict(extinction=cs.ext[0] * np.sum(irradiance), scattering=cs.sca[0] * np.sum(irradiance), detected=detected, reference=np.sum(dP * seen))


def sphere_koehler_cone_average():
    radii, n = SPHERE
    lam, D = 3.0, 25.0
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7, collection_obscuration=0.3, illumination="kohler", field_stop=D)
    r = F.focused_spheres(radii, n, lam, beam)
    ref = sphere_koehler_reference(lam, D)
    name = "sphere, Kohler (field stop 25 um): against a plane-wave cone average"
    return [close(name, "extinction", r["extinction"], ref["extinction"], 1e-10, 1e-8),
            close(name, "scattering", r["scattering"], ref["scattering"], 1e-10, 1e-8),
            close(name, "detected", r.detected, ref["detected"], 1e-9, 1e-8),
            close(name, "reference", r.reference, ref["reference"], 1e-12, 1e-8)]


# ---------------------------------------------------------------- cylinders

def cylinder_rows():
    radii, n = CYLINDER
    beam = CYLINDER_ROW_BEAM
    rows_ = F._CylinderRows(radii, n, 3.0, [beam], 14)
    out = []
    for sy in (0.05, 0.3, 0.55):
        value = rows_(sy) / 2
        for p, jones in enumerate(((1.0, 0.0), (0.0, 1.0))):
            brute = row_brute(radii, n, 3.0, sy, beam, jones, 14)
            name = f"coated cylinder, pupil row s_y = {sy:g}, Jones {jones}"
            out += [close(name, "ext, sca, detected, reference against plane-wave sums", value[p * rows_.width:p * rows_.width + 4], brute, 1e-9, 1e-12),
                    close(name, "the mirror row -s_y", row_brute(radii, n, 3.0, -sy, beam, jones, 14), brute, 1e-10, 1e-8)]
    return out


CYLINDER_ROW_BEAM = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, collection_obscuration=0.1, offset=(0.4, 0.0, 0.3))


def cylinder_whole_beam_pair(lam=3.0):
    """(PyStratify's adaptive whole-beam result, the same rows summed by a fixed 40-node quadrature over s_y) for
    CYLINDER under NA 0.6/0.2, collection 0.8, focus at (0.4, 0, 0.3); the fixed sum is [ext, sca, det, ref] for x
    then y polarization at index p * width."""
    radii, n = CYLINDER
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, offset=(0.4, 0.0, 0.3))
    r = F.focused_cylinders(radii, n, lam, beam, m_max=14, tolerance=1e-9)
    rows_ = F._CylinderRows(radii, n, lam, [beam], 14)
    total = 0
    for a, b in F._cuts(0, 0.6, (0.2,)):
        sy, w = F._mapped_nodes(a, b, 40)
        total = total + sum(wi * rows_(s) for s, wi in zip(sy, w))
    return r, total, rows_.width


def cylinder_whole_beam():
    r, total, width = cylinder_whole_beam_pair()
    name = "coated cylinder, whole beam: adaptive against a fixed 40-node quadrature"
    out = []
    for p, pol in enumerate("xy"):
        bp = r.by_polarization[pol]
        out.append(close(name, f"{pol}: ext, sca, detected, reference", np.ravel([bp["extinction"], bp["scattering"], bp["detected"], bp["reference"]]),
                         total[p * width:p * width + 4], 1e-8, 1e-8))
    return out + [close(name, "reference against 1 (the condenser sees the whole beam)", r.reference, 1.0, 1e-8, 1e-8),
                  flag(name, "converged", r.diagnostics["converged"])]


def cylinder_na_ratio(na, pol_name, lam=3.0):
    """CYLINDER's extinction under an unobscured beam of this NA, polarized x (axial-magnetic) or y (axial-electric),
    divided by the beam's |E|^2 integrated along the axis line and the plane-wave extinction width: 1 as NA -> 0."""
    radii, n = CYLINDER
    k = 2 * np.pi / lam
    sol = solve_cylinder(radii, n, lam, beta=0)
    out_ = F.focused_cylinders(radii, n, lam, ps.FocusedBeam(na, polarization=pol_name))
    sy, wy = F._mapped_nodes(0, na, 60)
    line = 0
    for s, w in zip(sy, wy):
        (a, b), = F._row_intervals(s, 0, na)
        al, wa = F._nodes(a, b, 40)
        u, _, _ = F._row_basis(al, s)
        A = pupil_field(u, (0, na), (1.0, 0.0) if pol_name == "x" else (0.0, 1.0), k, (0, 0, 0))
        line += 2 * w * np.sum(np.abs(np.sum(A * wa[:, None], 0)) ** 2)
    pol = "axial-magnetic" if pol_name == "x" else "axial-electric"
    return float(np.ravel(out_["extinction"])[0] / (k / (2 * np.pi) * line) / cross_widths(sol, pol)["extinction"])


def cylinder_energy_and_na_limit():
    radii, n = CYLINDER
    beam = ps.FocusedBeam(0.5, 0.23, offset=(0.3, 0.0, 0.2))
    lossless = F.focused_cylinders(radii, n.real, [3.0, 5.0], beam)
    r = F.focused_cylinders(radii, n, [3.0, 5.0], beam)
    name = "coated cylinder: energy, and NA -> 0"
    out = [row(name, f"max |absorption|, lossless, {p}", np.abs(lossless.by_polarization[p]["absorption"]), 1e-12) for p in "xy"]
    out += [flag(name, "absorption > 0, lossy", np.all(r["absorption"] > 0)),
            close(name, "multipole extinction shares summed", r.multipoles["extinction"].sum(-1), r["extinction"], 1e-5, 1e-8)]
    for pol_name in "xy":
        ratios = [cylinder_na_ratio(na, pol_name) for na in (0.04, 0.02)]
        out += [flag(name, f"{pol_name}: extinction ratio closer to 1 at NA 0.02 than 0.04", abs(ratios[1] - 1) < abs(ratios[0] - 1)),
                row(name, f"{pol_name}: |extinction / (line |E|^2 C_ext) - 1| at NA 0.02", abs(ratios[1] - 1), 2e-3, "<")]
    return out


def cylinder_koehler_reference(lam, D=30.0):
    """Kohler illumination of CYLINDER (NA 0.6/0.2, collection 0.7/0.3, field stop D) as a cone average of plane
    waves: {pol: [extinction, detected - reference, reference]} for x and y."""
    radii, n = CYLINDER
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
                for pol, jones in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
                    e = F._pupil_polarization(u, jones)
                    for ui, cn, cm, wai in zip(u, np.sum(e * e_n, -1), np.sum(e * e_m, -1), wa):
                        b_ = sol.t @ np.array([cn, 1j * cm])
                        c_ext = -(4 / k) * np.real(np.sum(cn * b_[:, 0] - 1j * cm * b_[:, 1]))
                        alpha = np.arctan2(ui[0], ui[2])  # dC/dphi of this wave, observed at its own row
                        landed = 0
                        for clo, chi in F._row_intervals(sy, 0.3, 0.7):
                            ph, wp = F._nodes(clo, chi, 120)
                            amp = np.exp(1j * np.outer(ph - alpha, sol.orders)) @ b_
                            landed += (2 / (np.pi * k)) * np.sum(np.abs(amp) ** 2, 1) @ wp
                        s = np.hypot(ui[0], ui[1])
                        seen = 0.3 <= s <= 0.7
                        weight = 2 * wy * wai * F.aplanatic_amplitude(ui[2], 0.2, 0.6) ** 2  # dP, both +-sy
                        irradiance = weight / (G * ui[2]) * D  # times the lit chord
                        totals[pol] += [irradiance * c_ext, irradiance * (landed - seen * c_ext), weight * seen]
    return totals


def cylinder_koehler_cone_average():
    radii, n = CYLINDER
    lam, D = 3.0, 30.0
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7, collection_obscuration=0.3, illumination="kohler", field_stop=D)
    r = F.focused_cylinders(radii, n, lam, beam, m_max=14, tolerance=1e-9)
    totals = cylinder_koehler_reference(lam, D)
    name = "coated cylinder, Kohler (field stop 30 um): against a plane-wave cone average"
    out = []
    for pol in "xy":
        bp = r.by_polarization[pol]
        out += [close(name, f"{pol}: extinction", bp["extinction"], totals[pol][0], 1e-8, 1e-8),
                close(name, f"{pol}: detected", bp["detected"], totals[pol][2] + totals[pol][1], 1e-8, 1e-8),
                close(name, f"{pol}: reference", bp["reference"], totals[pol][2], 1e-8, 1e-8)]
    return out


def scalar_cylinder():
    lam, R, m_in = 3.0, 1.1, 1.45 + 0.06j
    k = 2 * np.pi / lam
    sy = 0.4
    n_v, n_p = np.sqrt(1 - sy * sy), np.sqrt(m_in ** 2 - sy * sy)
    t = solve_cylinder([R], [n_p, n_v], lam, beta=0, m_max=12).t[:, 0, 0]
    expected = [scalar_cylinder_coefficient(abs(m), k * R * n_v, k * R * n_p, n_v, n_p) for m in range(-12, 13)]
    name = "scalar cylinder model"
    out = [close(name, "coefficients against the scalar boundary equations", t, expected, 1e-10, 1e-8)]
    radii, n = CYLINDER
    beam = SCALAR_BEAM
    lossless = F.focused_cylinders(radii, n.real, [3.0, 5.0], beam)
    r = F.focused_cylinders(radii, n, 3.0, beam, m_max=14, tolerance=1e-9)
    out += [row(name, "max |absorption|, lossless", np.abs(lossless["absorption"]), 1e-12),
            flag(name, "absorption > 0, lossy; no polarization split", r["absorption"] > 0 and not r.by_polarization)]
    for sy in (0.1, 0.45):
        value, (sca, ext) = scalar_row_pair(sy)
        out += [close(name, f"row s_y = {sy:g}: scattering against a direct superposition", value[1], sca, 1e-9, 1e-8),
                close(name, f"row s_y = {sy:g}: extinction against a direct superposition", value[0], ext, 1e-9, 1e-8)]
    return out


SCALAR_BEAM = ps.FocusedBeam(0.6, 0.2, model="scalar", offset=(0.3, 0.0, 0.2))


def scalar_row_pair(sy, lam=3.0):
    """(PyStratify's scalar row s_y of CYLINDER under SCALAR_BEAM: [ext, sca, det, ref], (scattering, extinction)
    of the row's plane waves superposed directly with the effective-index coefficients)."""
    radii, n = CYLINDER
    beam = SCALAR_BEAM
    k = 2 * np.pi / lam
    value = F._CylinderRows(radii, n, lam, [beam], 14)(sy)[:4] / 2
    eff = np.sqrt(n ** 2 - sy ** 2)
    ts = solve_cylinder(radii, eff, lam, beta=0, m_max=14).t[:, 0, 0]
    orders = np.arange(-14, 15)
    intervals = F._row_intervals(sy, 0.2, 0.6)
    al = np.concatenate([F._nodes(a, b, 60)[0] for a, b in intervals])
    wa = np.concatenate([F._nodes(a, b, 60)[1] for a, b in intervals])

    def amplitude(phi):
        u, _, _ = F._row_basis(phi, sy)
        s = np.hypot(u[:, 0], u[:, 1])
        return np.where((s >= 0.2) & (s <= 0.6), F.aplanatic_amplitude(u[:, 2], 0.2, 0.6), 0) * np.exp(-1j * k * u @ np.array(beam.offset))

    U = amplitude(al) * wa

    def scattered(phi):  # (1/pi) sum over the row's plane waves of their rotated scattered amplitudes
        return np.array([np.sum(U * (np.exp(1j * np.outer(p - al, orders)) @ ts)) for p in phi]) / np.pi

    ph = 2 * np.pi * np.arange(256) / 256
    ext = sum(-2 * np.real(np.conj(amplitude(p)) * scattered(p) @ wq) for p, wq in (F._nodes(a, b, 80) for a, b in intervals))
    return value, (np.sum(np.abs(scattered(ph)) ** 2) * 2 * np.pi / 256, ext)


# ---------------------------------------------------------------- near fields

NEAR_SPHERE = ([0.6, 1.2], [1.5 + 0.1j, 1.3 + 0.02j, 1.0], 3.0)
NEAR_BEAM = dict(na=0.6, obscuration=0.2, offset=(0.4, -0.3, 0.25), polarization="y")


def sphere_near_field_pair(points):
    """(PyStratify's focused near field, its plane-wave superposition): (E, H) each, at points (P, 3), for NEAR_SPHERE
    under NEAR_BEAM (y polarization, 36 x 56 pupil directions)."""
    radii, n, lam = NEAR_SPHERE
    sol = ps.solve(radii, n, lam, regime="near")
    beam = ps.FocusedBeam(NEAR_BEAM["na"], NEAR_BEAM["obscuration"], offset=NEAR_BEAM["offset"], polarization=NEAR_BEAM["polarization"])
    field = F.focused_field_spheres(radii, n, lam, beam, points, l_max=sol.orders[-1])
    k = 2 * np.pi / lam
    u, w = pupil_grid(0.2, 0.6, 36, 56)
    A = pupil_field(u, (0.2, 0.6), (0.0, 1.0), k, beam.offset)
    E, H = np.zeros((len(points), 3), complex), np.zeros((len(points), 3), complex)
    for ui, Ai, wi in zip(u, A, w):
        e = F._pupil_polarization(ui[None], (0.0, 1.0))[0]  # real unit polarization; A = amplitude e
        amplitude = Ai @ e
        R = np.stack((e, np.cross(ui, e), ui), axis=1)
        p = points @ R
        nf = ps.near_field(sol, p[:, 0], p[:, 1], p[:, 2])
        E += wi * amplitude * np.stack([nf.e[c] for c in "xyz"], -1) @ R.T
        H += wi * amplitude * np.stack([nf.h[c] for c in "xyz"], -1) @ R.T
    unit = F.focal_amplitude((0.2, 0.6))
    return field, E / unit, H / unit


def sphere_near_field():
    points = np.array([[0.1, 0.2, 0.3], [0.9, 0.1, -0.2], [1.6, 0.4, 0.5], [2.5, -1.0, -1.5]])
    field, E, H = sphere_near_field_pair(points)
    name = "coated sphere, focused near field at 4 points: against plane-wave near fields"
    return [close(name, "E", field.e[0], E, 1e-8, 1e-9), close(name, "H", field.h[0], H, 1e-8, 1e-9),
            flag(name, "the incident field is withheld inside the sphere", np.all(np.isnan(field.e_incident[0, :2])) and np.all(np.isfinite(field.e_incident[0, 2:])))]


def cylinder_near_field_pair(points):
    """(PyStratify's focused near field, E as a superposition of plane-wave cylinder fields) at points (P, 3), for a
    coated cylinder (0.6/1.2 um, axis along y) under NA 0.6/0.2, x polarized, focus at (0.3, 0.2, 0.25), 3 um."""
    radii, n, lam = [0.6, 1.2], np.array([1.5 + 0.1j, 1.3 + 0.02j, 1.0]), 3.0
    beam = ps.FocusedBeam(0.6, 0.2, offset=(0.3, 0.2, 0.25), polarization="x")
    field = F.focused_field_cylinders(radii, n, lam, beam, points, m_max=14)
    k = 2 * np.pi / lam
    cyl = np.stack((points[:, 2], points[:, 0], points[:, 1]), -1)
    E = np.zeros((len(points), 3), complex)
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
    return field, E / F.focal_amplitude((0.2, 0.6))


def cylinder_near_field():
    field, E = cylinder_near_field_pair(np.array([[0.1, 0.0, 0.3], [0.9, 0.1, -0.2], [-0.5, 0.8, 1.9]]))
    return [close("coated cylinder, focused near field at 3 points: against plane-wave fields", "E", field.e[0], E, 1e-7, 1e-8)]


def film_near_field_and_parseval():
    n, d, lam = np.array([1.0, 1.6 + 0.05j, 1.42, 1.0]), np.array([inf, 0.8, 2.0, inf]), 3.0
    beam = ps.FocusedBeam(0.6, 0.2, offset=(0.3, -0.2, 0.5))
    points = np.array([[0.1, 0.0, -0.7], [0.9, 0.4, 0.3], [-0.6, 0.2, 1.5], [0.3, -0.8, 3.2]])
    field = F.focused_field_films(n, d, lam, beam, points)
    k = 2 * np.pi / lam
    u, w = pupil_grid(0.2, 0.6, 48, 48)
    located = [find_in_structure_with_inf(d, z) if z >= 0 else (0, z) for z in points[:, 2]]
    name = "film stack, focused near field at 4 depths: against position-resolved plane waves"
    out = []
    for index, jones in enumerate(((1.0, 0.0), (0.0, 1.0))):
        E = np.zeros((4, 3), complex)
        cache = {}
        for ui, wi in zip(u, w):
            s = np.hypot(ui[0], ui[1])
            theta = np.arcsin(s)
            if theta not in cache:
                data = {p: coh_tmm(p, n, d, theta, lam) for p in "sp"}
                cache[theta] = {p: np.array([[position_resolved(L, z, data[p])[c] for c in ("Ex", "Ey", "Ez")] for L, z in located]) for p in "sp"}
            cf, sf = ui[0] / s, ui[1] / s
            a = F.aplanatic_amplitude(ui[2], 0.2, 0.6) * np.exp(-1j * k * ui[2] * beam.offset[2])
            amp_p, amp_s = jones[0] * cf + jones[1] * sf, -jones[0] * sf + jones[1] * cf
            local = amp_p * cache[theta]["p"] + amp_s * cache[theta]["s"]
            lab = np.stack((local[:, 0] * cf - local[:, 1] * sf, local[:, 0] * sf + local[:, 1] * cf, local[:, 2]), -1)
            lateral = np.exp(1j * k * s * ((points[:, 0] - beam.offset[0]) * cf + (points[:, 1] - beam.offset[1]) * sf))
            E += wi * a * lateral[:, None] * lab
        out.append(close(name, f"E, Jones {jones}", field.e[index], E / F.focal_amplitude((0.2, 0.6)), 1e-10, 1e-12))
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
        plane = sum(0.5 * np.sum(np.abs([position_resolved(L, z, coh_tmm(p, n, d, theta, lam))[c] for c in ("Ex", "Ey", "Ez")]) ** 2) for p in "sp")
        expected += 2 * np.pi * wi * F.aplanatic_amplitude(xi, 0.2, 0.6) ** 2 * plane / xi
    return out + [close("film stack, Parseval at depth 1.5 um (lateral |E|^2 integral, tail extrapolated)",
                        "lateral integral against (2pi/k)^2 Int |A|^2 |E_plane|^2 / cos t", lateral, (2 * np.pi / k) ** 2 * expected, 3e-4, 1e-8)]


def all_rows():
    rows = []
    for mode in FILM_MODES:
        rows += film_cone_quadrature(mode)
    rows += film_path_factor() + transparent_film_zero_absorbance() + film_koehler_equals_coherent()
    for offset in SPHERE_OFFSETS:
        rows += sphere_amplitude_superposition(offset)
    rows += sphere_axis_path() + sphere_invisible_and_na_limit() + sphere_koehler_cone_average()
    rows += cylinder_rows() + cylinder_whole_beam() + cylinder_energy_and_na_limit() + cylinder_koehler_cone_average() + scalar_cylinder()
    return rows + sphere_near_field() + cylinder_near_field() + film_near_field_and_parseval()


if __name__ == "__main__":
    report("Focused illumination: semi-analytic forms against plane-wave superpositions and limits",
           "Generated by `python benchmarks/focused_checks.py`; `tests/test_focused.py` asserts every row (lengths and "
           "wavelengths in um). close rows are numpy's isclose / allclose: max |a - b| / (atol + rtol |b|) <= 1.",
           all_rows())
