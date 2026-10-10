"""Focused illumination of films, cylinders and spheres.

Pupil model. An aplanatic objective (sine condition) whose pupil is filled uniformly
illuminates the directions u = (sin t cos f, sin t sin f, cos t) of the annulus
NA_obs <= n sin t <= NA in the medium of index n where it is focused. The beam is the
angular spectrum E(r) = Int A(u) exp(i k u.(r - r_f)) dOmega focused at r_f, with

    A(u) = a(t) [jx (cos f e_t - sin f e_f) + jy (sin f e_t + cos f e_f)],
    a(t)^2 = cos t / (pi (s2^2 - s1^2)),  s1 = NA_obs / n,  s2 = NA / n,

for a pupil Jones vector (jx, jy): the Richards-Wolf polarization map, whose power per
direction is uniform in sin^2 t, normalized so that Int |A|^2 dOmega = 1. Outgoing
far-field amplitudes F (E ~ F exp(ikr)/r) are written in units of 2 pi/(i k): the beam's
own is F = A, so every power below, Int |F|^2 dOmega, is a fraction of the beam's.
"Unpolarized" averages the x and y pupil polarizations.

Films (laterally invariant): a stack never mixes plane-wave directions, so the detected
power is the cone average of plane-wave T (transmission, inside the condenser's annulus,
n sin t conserved) or R (reflection, the objective collects): s and p equally, uniform
in sin^2 t, by adaptive quadrature of :func:`~pystratify.planar.layered_response`.
Coherent and Koehler illumination coincide.

Spheres. In the multipole basis (X_lm, Z_lm = u x X_lm, orthonormal on the unit
sphere) A = sum alpha_lm X_lm + beta_lm Z_lm; with the T-matrix (T = -b_l on X, the
magnetic multipoles; -a_l on Z, the electric) the scattered amplitude is
2 sum (T_TE alpha X + T_TM beta Z), so

    P_ext = -4 Re sum (T_TE |alpha|^2 + T_TM |beta|^2),  P_sca = 4 sum |T_TE alpha|^2 + |T_TM beta|^2,

and the collected power is Int_C |A + F_sca|^2 dOmega over the condenser's annulus. A
beam focused on the centre (or on the axis through it) has only m = +-1 and its
coefficients follow from a 1-D pupil integral of the Bohren-Huffman pi_l + tau_l; a
lateral offset needs every m, from the pupil by FFT in the azimuth.

Cylinders (axis along y, the beam along z). Each axial direction cosine s_y is an
independent 2-D problem: one oblique solve gives t_m for every transverse angle alpha
of that pupil row (s_x = sqrt(1 - s_y^2) sin alpha), dOmega = d alpha d s_y, and the
rows add incoherently in power. With g_m = Int A_N,M(alpha) exp(-i m alpha) d alpha on
the row's axial-electric/axial-magnetic polarizations, b_m = t_m (g_N, i g_M):

    P_ext = -(2/pi) Re sum (g_N* b_N - i g_M* b_M),  P_sca = (2/pi) sum |b_N|^2 + |b_M|^2,

per unit s_y, and the outgoing amplitude at the observation angle phi (s_x = sqrt(1 -
s_y^2) sin phi) is A(phi) + (1/pi) sum exp(i m phi) (b_N e_N - i b_M e_M). The s_y
integral is adaptive with an error estimate, split where rows cross an annulus edge.

Corrected scalar model (coherent spheres and cylinders): the scalar field
U = Int a(t) exp(i k u.r) dOmega with continuous U and dU/dr. Its sphere coefficient of
order l is the TE one (l >= 1, mu = 1) and a monopole l = 0; its cylinder coefficient
on row s_y is the axial-electric one at normal incidence with indices
sqrt(n^2 - (n_host s_y)^2). Scattered light is collected over the whole condenser
annulus at its true angle, with the pupil weighting of the vector model.

Koehler illumination: every pupil direction is an independent plane wave filling the
field stop (area G = pi D^2 / 4; a cylinder through the field centre is lit over the
chord D). A direction carrying power dP has irradiance dP/(G cos t), so with the
aplanatic weighting a cross section sigma enters as Int sigma dOmega / (G pi (s2^2 -
s1^2)). The detector sees the direct light inside its annulus less each such
direction's extinction, plus the scattering that lands inside it.

Lengths share the caller's unit; the focus offset is (x, y, z).
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .problem import FocusedBeam, check_focused

__all__ = ["focused_films", "focused_spheres", "focused_cylinders", "FocusedResult", "aplanatic_amplitude",
           "focused_field_films", "focused_field_spheres", "focused_field_cylinders", "FocusedField", "focal_field",
           "focal_amplitude", "vector_harmonics", "monopole_coefficient"]


# ---------------------------------------------------------------- pupil and quadrature

@lru_cache(maxsize=128)
def _gauss(count):
    return np.polynomial.legendre.leggauss(int(count))


def _nodes(a, b, count):
    x, w = _gauss(max(4, int(count)))
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _mapped_nodes(a, b, count):
    """Gauss-Legendre in t of x = a + (b - a) sin^2 t, t in [0, pi/2]: square-root edges at a or b
    (a row of the pupil shrinking to nothing) become smooth."""
    t, w = _nodes(0.0, np.pi / 2, count)
    return a + (b - a) * np.sin(t) ** 2, w * (b - a) * np.sin(2 * t)


def _cuts(a, b, points=()):
    """Consecutive segments of [a, b] split at the given points."""
    edges = sorted({a, b, *(p for p in points if a < p < b)})
    return list(zip(edges[:-1], edges[1:]))


def _beams(beams):
    if isinstance(beams, FocusedBeam):
        return [beams], True
    beams = list(beams)
    if not beams or not all(isinstance(b, FocusedBeam) for b in beams):
        raise ValueError("beams must be a FocusedBeam or a non-empty sequence of them")
    return beams, False


def _jones(beam):
    """(weight, (jx, jy)) of the pupil polarizations averaged."""
    return {"x": [(1.0, (1.0, 0.0))], "y": [(1.0, (0.0, 1.0))]}.get(
        beam.polarization, [(0.5, (1.0, 0.0)), (0.5, (0.0, 1.0))])


def aplanatic_amplitude(cos_theta, s1, s2):
    """a(t) of the uniformly filled aplanatic pupil between sines s1 < s2, Int |A|^2 dOmega = 1."""
    return np.sqrt(np.asarray(cos_theta, float) / (np.pi * (s2 * s2 - s1 * s1)))


def _overlap(lo, hi, lo2, hi2):
    return max(0.0, min(hi, hi2) - max(lo, lo2))


def _per_wavelength(n, count, layers):
    n = np.asarray(n, complex)
    try:
        return np.ascontiguousarray(np.broadcast_to(n, (count, layers)))
    except ValueError:
        raise ValueError(f"n needs shape ({layers},) or ({count}, {layers})") from None


@dataclass(frozen=True)
class FocusedResult:
    """Powers of a focused beam as fractions of the beam's power, shape ``(B, W)``
    (``(W,)`` for a single beam): B beams, W wavelengths.

    ``detected`` reaches the detector through the collecting aperture and ``reference``
    in the reference setup (films: the reference stack; spheres and cylinders: the empty
    host); ``apparent_absorbance`` = -log10(detected / reference). Films carry
    ``reflectance``, ``transmittance``, ``absorptance`` and ``layer_absorptance`` (B, W,
    internal layers); spheres and cylinders ``extinction``, ``scattering`` and
    ``absorption``, their per-order shares in ``multipoles`` (a list per beam when the beams mix the
    vector and scalar models, whose orders differ) and, for cylinders, every
    quantity per pupil polarization in ``by_polarization`` ('x' across the axis, 'y'
    along it). ``diagnostics`` holds orders, quadrature errors and convergence.
    """

    detected: np.ndarray
    reference: np.ndarray
    values: dict
    multipoles: dict
    by_polarization: dict
    diagnostics: dict

    @property
    def apparent_absorbance(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.log10(self.detected / self.reference)

    def __getitem__(self, name):
        if name in ("detected", "reference"):
            return getattr(self, name)
        if name == "apparent_absorbance":
            return self.apparent_absorbance
        return self.values[name]


def _result(rows, single, keys, multipole_keys=(), polarized=False):
    """Stack per-beam dictionaries of (W,)-arrays into a FocusedResult. Beams whose arrays differ in
    shape (a scalar beam's orders start at l = 0 with one wave type, a vector beam's at 1 with two)
    stay a list, one array per beam."""
    def stack(name, source):
        values = [np.asarray(row[name]) for row in source]
        if single:
            return values[0]
        return np.array(values) if len({v.shape for v in values}) == 1 else values
    values = {k: stack(k, rows) for k in keys}
    multipoles = {k: stack(k, [row["multipoles"] for row in rows]) for k in multipole_keys}
    if multipole_keys:
        orders = [np.asarray(row["multipoles"]["orders"]) for row in rows]
        same = all(o.shape == orders[0].shape and np.array_equal(o, orders[0]) for o in orders)
        multipoles["orders"] = orders[0] if single or same else orders
    by_polarization = {}
    if polarized:
        for p in ("x", "y"):
            source = [row["by_polarization"][p] for row in rows]
            by_polarization[p] = {k: stack(k, source) for k in ("detected", "reference", *keys)}
    diagnostics = {k: stack(k, [row["diagnostics"] for row in rows]) for k in rows[0]["diagnostics"]}
    return FocusedResult(stack("detected", rows), stack("reference", rows), values, multipoles, by_polarization, diagnostics)


# ---------------------------------------------------------------- films

def focused_films(n, thickness, wavelength, beams, coherence=None, reference=None, *, tolerance=1e-6,
                  max_evaluations=20000):
    """A focused beam on a planar stack: cone-averaged R, T, absorptance per layer and the
    detected power of the sample and of a reference stack.

    ``n`` (L,) or (W, L) and ``thickness`` (L,) as :class:`~pystratify.Problem` (infinite
    half-spaces first and last); ``coherence`` the per-region 'c'/'i' column (None: all
    coherent); ``reference`` = (n, thickness[, coherence]) of the reference stack (the
    sample stack without its sample layers), sharing the ambient and exit media.
    ``beams``: a :class:`FocusedBeam` or a sequence of them (an NA sweep, say).
    """
    from .integration import integrate
    from .planar import layered_response

    wavelength = np.atleast_1d(np.asarray(wavelength, float))
    thickness = np.asarray(thickness, float)
    n = _per_wavelength(n, wavelength.size, thickness.size)
    beams, single = _beams(beams)
    stacks = [(n, thickness, None if coherence is None else tuple(coherence))]
    if reference is not None:
        n_ref, d_ref = reference[0], np.asarray(reference[1], float)
        n_ref = _per_wavelength(n_ref, wavelength.size, d_ref.size)
        if not (np.array_equal(n_ref[:, 0], n[:, 0]) and np.array_equal(n_ref[:, -1], n[:, -1])):
            raise ValueError("the reference stack shares the sample's ambient and exit media")
        stacks.append((n_ref, d_ref, None if len(reference) < 3 or reference[2] is None else tuple(reference[2])))
    for beam in beams:
        for row in n:
            check_focused("films", row, beam)

    rows = []
    for beam in beams:
        out = {k: np.full(wavelength.size, np.nan) for k in ("reflectance", "transmittance", "absorptance", "detected",
                                                               "reference", "error")}
        layers = np.full((wavelength.size, max(0, thickness.size - 2)), np.nan)
        evaluations, converged = np.zeros(wavelength.size, int), np.zeros(wavelength.size, bool)
        for w, lam in enumerate(wavelength):
            ambient = n[w, 0].real
            u1, u2 = (beam.obscuration / ambient) ** 2, (beam.na / ambient) ** 2
            c1, c2 = ((v / ambient) ** 2 for v in beam.collection)
            reflection = beam.mode == "reflection"
            detected, total_error, total_evaluations, ok = [], 0.0, 0, True
            for index, (n_stack, d_stack, c_stack) in enumerate(stacks):
                media = n_stack[w]
                # kinks: the condenser's edges and every lossless layer's critical angle
                kinks = [c1, c2] + [(m.real / ambient) ** 2 for m in media[1:] if m.imag == 0]

                def integrand(u, media=media, d_stack=d_stack, c_stack=c_stack):
                    theta = np.arcsin(np.sqrt(u))
                    s = layered_response("s", media, d_stack, c_stack, theta, lam)
                    p = layered_response("p", media, d_stack, c_stack, theta, lam)
                    r, t = 0.5 * (s["R"] + p["R"]), 0.5 * (s["T"] + p["T"])
                    seen = r if reflection else (t if c1 < u < c2 else 0.0)
                    return np.concatenate(([r, t, seen], 0.5 * (s["absorption"] + p["absorption"])[1:-1])) / (u2 - u1)

                breaks = [u1, u2] + [k for k in kinks if u1 < k < u2]
                integral = integrate(integrand, breaks, tolerance, max_evaluations)
                total_error += integral.error
                total_evaluations += integral.evaluations
                ok = ok and integral.converged
                value = integral.value
                detected.append(value[2])
                if index == 0:
                    out["reflectance"][w], out["transmittance"][w] = value[0], value[1]
                    out["absorptance"][w] = 1 - value[0] - value[1]
                    layers[w] = value[3:]
            out["detected"][w] = detected[0]
            out["reference"][w] = detected[1] if reference is not None else np.nan
            out["error"][w], evaluations[w], converged[w] = total_error, total_evaluations, ok
        rows.append(dict(**out, layer_absorptance=layers,
                         diagnostics=dict(error=out["error"], evaluations=evaluations, converged=converged)))
    return _result(rows, single, ("reflectance", "transmittance", "absorptance", "layer_absorptance"))


# ---------------------------------------------------------------- spheres

def _normalized_legendre_columns(degree, x):
    """Fully normalized associated Legendre functions (Condon-Shortley phase,
    Int_{-1}^{1} P^2 dx = 1), yielded as (m, P[l]) for m = 0..degree + 1, l = 0..degree."""
    x = np.asarray(x, float)
    s = np.sqrt(np.clip(1 - x * x, 0, None))
    diagonal = np.full(x.shape, np.sqrt(0.5))
    for m in range(degree + 2):
        if m:
            diagonal = -np.sqrt((2 * m + 1) / (2 * m)) * s * diagonal
        column = np.zeros((degree + 1,) + x.shape)
        if m <= degree:
            column[m] = diagonal
            if m + 1 <= degree:
                column[m + 1] = np.sqrt(2 * m + 3) * x * diagonal
            for l in range(m + 2, degree + 1):
                a = np.sqrt((4 * l * l - 1) / (l * l - m * m))
                b = np.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
                column[l] = a * (x * column[l - 1] - b * column[l - 2])
        yield m, column


def vector_harmonics(degree, x):
    """Normalized angular functions of the vector spherical harmonics, per m = -degree..degree:
    yields (m, P, pi, tau), each (degree + 1, len(x)) over l = 0..degree (rows l < |m| zero),
    with P = P_l^m(cos t) normalized, pi = m P / sin t and tau = dP/dt, so that

        Y_lm = P e^{imf} / sqrt(2 pi),
        X_lm = e^{imf} (-pi e_t - i tau e_f) / sqrt(2 pi l (l + 1)),  Z_lm = u x X_lm.
    """
    l = np.arange(degree + 1)[:, None]
    cache = {}
    for m, column in _normalized_legendre_columns(degree, x):
        cache[m] = column
        cache.pop(m - 3, None)
        if m < 1:
            continue
        mm = m - 1  # the column m - 1 is complete once m and m - 2 are known
        upper, lower = cache[mm + 1], (cache[mm - 1] if mm >= 1 else -cache[1])  # P^{-1} = -P^1
        shifted_upper = np.vstack((np.zeros_like(upper[:1]), upper[:-1]))  # P_{l-1}^{m+1}
        shifted_lower = np.vstack((np.zeros_like(lower[:1]), lower[:-1]))  # P_{l-1}^{m-1}
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(l > 0, np.sqrt((2 * l + 1) / np.maximum(2 * l - 1, 1)), 0.0)
            pi = -0.5 * factor * (np.sqrt(np.clip((l - mm) * (l - mm - 1), 0, None)) * shifted_upper
                                  + np.sqrt(np.clip((l + mm) * (l + mm - 1), 0, None)) * shifted_lower)
            tau = 0.5 * (np.sqrt(np.clip((l - mm) * (l + mm + 1), 0, None)) * upper
                         - np.sqrt(np.clip((l + mm) * (l - mm + 1), 0, None)) * lower)
        valid = l >= mm
        P, pi, tau = cache[mm] * valid, pi * valid, tau * valid
        yield mm, P, pi, tau
        if mm:
            sign = (-1) ** mm
            yield -mm, sign * P, -sign * pi, sign * tau
        if mm == degree:
            return


def monopole_coefficient(radii, n, wavelength):
    """Scalar l = 0 coefficient T_0 (U = j_0 + T_0 h_0 outside; U and dU/dr continuous) of a
    multilayered sphere, shape (W,): the 1-D problem for r U, swept outwards by its log-derivative."""
    radii = np.atleast_1d(np.asarray(radii, float))
    wavelength = np.atleast_1d(np.asarray(wavelength, float))
    n = _per_wavelength(n, wavelength.size, radii.size + 1)
    k = 2 * np.pi * n / wavelength[:, None]

    def tangent(z):
        with np.errstate(all="ignore"):
            value = np.tan(z)
        return np.where(np.abs(z.imag) > 20, 1j * np.sign(z.imag), value)

    # y = (rU)'/(rU): k cot(k r) in the core, carried through each shell
    with np.errstate(all="ignore"):
        y = k[:, 0] / tangent(k[:, 0] * radii[0])
    for j in range(1, radii.size):
        t = tangent(k[:, j] * (radii[j] - radii[j - 1]))
        y = (y - k[:, j] * t) / (1 + y * t / k[:, j])
    kh, R = k[:, -1], radii[-1]
    return (y * np.sin(kh * R) - kh * np.cos(kh * R)) / ((kh + 1j * y) * np.exp(1j * kh * R))


def _sphere_axis(t, k, orders, illumination, collection, z):
    """On-axis (offset only along z) vector beam: m = +-1, Bohren-Huffman pi_l + tau_l.

    t (2, W, L) TM/TE; k (W,); illumination/collection (s1, s2). Returns fractions per wavelength.
    """
    from .farfield import angular_functions

    s1, s2 = illumination
    x1, x2 = np.sqrt(1 - s2 * s2), np.sqrt(1 - s1 * s1)
    L = orders.size
    spread = float(np.max(np.abs(k * z)))
    norm = 2 * np.pi * (orders * (orders + 1.0)) ** 2 / (2 * orders + 1)  # Int |m_l|^2 dOmega

    def incident(x):
        inside = (x >= x1) & (x <= x2)
        return np.where(inside, aplanatic_amplitude(x, s1, s2), 0.0)[None] * np.exp(-1j * np.outer(k * z, x))

    x, w = _nodes(x1, x2, L + 40 + spread * (x2 - x1))
    pi, tau = angular_functions(orders, np.arccos(x))
    c = np.pi * ((incident(x) * w) @ (pi + tau).T) / norm  # (W, L): alpha-like = beta-like
    tm, te = t[0], t[1]
    weight = norm * np.abs(c) ** 2
    ext_by_order = -4 * np.stack((np.real(tm) * weight, np.real(te) * weight), axis=1)  # (W, 2, L)
    sca_by_order = 4 * np.stack((np.abs(tm) ** 2 * weight, np.abs(te) ** 2 * weight), axis=1)
    c1, c2 = collection
    y1, y2 = np.sqrt(1 - c2 * c2), np.sqrt(1 - c1 * c1)
    detected, reference = np.zeros(k.size), np.zeros(k.size)
    for a, b in _cuts(y1, y2, (x1, x2)):
        x, w = _nodes(a, b, L + 40 + spread * (b - a))
        pi, tau = angular_functions(orders, np.arccos(x))
        beam = incident(x)
        g_theta = beam + 2 * ((c * te) @ pi + (c * tm) @ tau)
        g_phi = beam + 2 * ((c * te) @ tau + (c * tm) @ pi)
        detected += np.pi * (np.abs(g_theta) ** 2 + np.abs(g_phi) ** 2) @ w
        reference += 2 * np.pi * np.abs(beam) ** 2 @ w
    return dict(detected=detected, reference=reference, ext_by_order=ext_by_order, sca_by_order=sca_by_order)


def _pupil_grid(s1, s2, count_theta, count_phi):
    x, w = _nodes(np.sqrt(1 - s2 * s2), np.sqrt(1 - s1 * s1), count_theta)
    phi = 2 * np.pi * np.arange(count_phi) / count_phi
    return x, w, phi


def _beam_spectrum(x, phi, k, offset, illumination, jones, scalar):
    """Azimuthal Fourier coefficients Int A exp(-i m f) df of the beam on directions cos t = x:
    shape (W, components, len(x), len(phi)) in FFT order (components: (A_t, A_f) or (U,))."""
    s1, s2 = illumination
    sin = np.sqrt(1 - x * x)
    inside = (x >= np.sqrt(1 - s2 * s2)) & (x <= np.sqrt(1 - s1 * s1))
    a = np.where(inside, aplanatic_amplitude(x, s1, s2), 0.0)[:, None]
    cp, sp = np.cos(phi)[None], np.sin(phi)[None]
    projection = sin[:, None] * (cp * offset[0] + sp * offset[1]) + x[:, None] * offset[2]
    phase = np.exp(-1j * k[:, None, None] * projection[None])  # (W, X, F)
    if scalar:
        fields = (a * phase)[:, None]
    else:
        jx, jy = jones
        fields = np.stack((a * (jx * cp + jy * sp) * phase, a * (-jx * sp + jy * cp) * phase), axis=1)
    return 2 * np.pi * np.fft.fft(fields, axis=-1) / phi.size


def _sphere_general(t, k, orders, illumination, collection, offset, jones, scalar, t0=None):
    """Any focus offset: every m, coefficients by pupil quadrature with an FFT in azimuth."""
    s1, s2 = illumination
    c1, c2 = collection
    L = orders.size
    kmax = float(np.max(k))
    lateral = np.hypot(offset[0], offset[1])
    spread = kmax * (lateral + abs(offset[2]))
    count_phi = int(2 ** np.ceil(np.log2(2 * (L + kmax * lateral * s2) + 48)))
    x, w, phi = _pupil_grid(s1, s2, L + 40 + spread, count_phi)
    x1, x2 = np.sqrt(1 - s2 * s2), np.sqrt(1 - s1 * s1)
    y1, y2 = np.sqrt(1 - c2 * c2), np.sqrt(1 - c1 * c1)
    cuts = _cuts(y1, y2, (x1, x2))
    pieces = [_nodes(a, b, L + 40 + spread * (b - a)) for a, b in cuts]
    xc = np.concatenate([p[0] for p in pieces]) if pieces else np.zeros(0)
    wc = np.concatenate([p[1] for p in pieces]) if pieces else np.zeros(0)
    spectrum = _beam_spectrum(x, phi, k, offset, illumination, jones, scalar)
    seen = _beam_spectrum(xc, phi, k, offset, illumination, jones, scalar) / (2 * np.pi)  # A's own Fourier series
    W = k.size
    degree = L
    l = np.arange(degree + 1)
    ext = np.zeros((W, 1 if scalar else 2, degree + 1))
    sca = np.zeros_like(ext)
    detected = 2 * np.pi * np.einsum("wcxf,x->w", np.abs(seen) ** 2, wc)  # unscattered part of every m
    if scalar:
        tl = np.concatenate((t0[:, None], t[1]), axis=1)  # (W, L + 1): monopole, then TE
        for m, P, _, _ in vector_harmonics(degree, np.concatenate((x, xc))):
            Pi, Pc = P[:, :x.size], P[:, x.size:]
            col = m % count_phi
            alpha = (spectrum[:, 0, :, col] * w) @ Pi.T / np.sqrt(2 * np.pi)  # (W, L + 1)
            ext[:, 0] += -4 * np.real(tl) * np.abs(alpha) ** 2
            sca[:, 0] += 4 * np.abs(tl * alpha) ** 2
            if xc.size:
                g = 2 * (tl * alpha) @ Pc / np.sqrt(2 * np.pi)
                before = np.abs(seen[:, 0, :, col]) ** 2
                detected += 2 * np.pi * ((np.abs(seen[:, 0, :, col] + g) ** 2 - before) @ wc)
    else:
        tm = np.concatenate((np.zeros((W, 1)), t[0]), axis=1)
        te = np.concatenate((np.zeros((W, 1)), t[1]), axis=1)
        with np.errstate(divide="ignore"):
            scale = np.where(l > 0, 1 / np.sqrt(2 * np.pi * l * (l + 1.0)), 0.0)
        for m, _, pi, tau in vector_harmonics(degree, np.concatenate((x, xc))):
            pi_i, tau_i, pi_c, tau_c = pi[:, :x.size], tau[:, :x.size], pi[:, x.size:], tau[:, x.size:]
            col = m % count_phi
            a_t, a_f = spectrum[:, 0, :, col] * w, spectrum[:, 1, :, col] * w
            alpha = (-(a_t @ pi_i.T) + 1j * (a_f @ tau_i.T)) * scale
            beta = (-1j * (a_t @ tau_i.T) - (a_f @ pi_i.T)) * scale
            ext[:, 0] += -4 * np.real(tm) * np.abs(beta) ** 2
            ext[:, 1] += -4 * np.real(te) * np.abs(alpha) ** 2
            sca[:, 0] += 4 * np.abs(tm * beta) ** 2
            sca[:, 1] += 4 * np.abs(te * alpha) ** 2
            if xc.size:
                ea, mb = te * alpha * scale, tm * beta * scale
                g_t = 2 * (-(ea @ pi_c) + 1j * (mb @ tau_c))
                g_f = 2 * (-1j * (ea @ tau_c) - (mb @ pi_c))
                before = np.abs(seen[:, 0, :, col]) ** 2 + np.abs(seen[:, 1, :, col]) ** 2
                after = np.abs(seen[:, 0, :, col] + g_t) ** 2 + np.abs(seen[:, 1, :, col] + g_f) ** 2
                detected += 2 * np.pi * ((after - before) @ wc)
        ext, sca = ext[..., 1:], sca[..., 1:]
    overlap = 2 * np.pi * np.einsum("wcxf,x->w", np.abs(seen) ** 2, wc)
    return dict(detected=detected, reference=overlap, ext_by_order=ext, sca_by_order=sca)


def _legendre_integrals(degree, a, b):
    """Int_a^b P_j(x) dx for j = 0..degree."""
    def table(x):
        P = np.zeros(degree + 2)
        P[0], P[1] = 1.0, x
        for j in range(1, degree + 1):
            P[j + 1] = ((2 * j + 1) * x * P[j] - j * P[j - 1]) / (j + 1)
        return P
    pa, pb = table(a), table(b)
    j = np.arange(degree + 1)
    out = np.empty(degree + 1)
    out[0] = b - a
    out[1:] = ((pb[2:] - pb[:-2]) - (pa[2:] - pa[:-2]))[: degree] / (2 * j[1:] + 1)
    return out


def _sphere_kohler(cs, phase, k, beam, illumination, collection):
    """Koehler: cross sections over the cone and the scattering landing inside the condenser.

    ``cs`` the plane-wave CrossSections and ``phase`` the Legendre coefficients of the
    unpolarized dsigma/dOmega, (W, 2L + 1), of the wavelengths at hand."""
    s1, s2 = illumination
    c1, c2 = collection
    x1, x2 = np.sqrt(1 - s2 * s2), np.sqrt(1 - s1 * s1)
    y1, y2 = np.sqrt(1 - c2 * c2), np.sqrt(1 - c1 * c1)
    degree = phase.shape[1] - 1
    # Int_I dOmega Int_C dOmega' P_j(u.u') = (2 pi)^2 Int_I P_j dx Int_C P_j dx' (addition theorem)
    inside = (2 * np.pi) ** 2 * phase @ (_legendre_integrals(degree, x1, x2) * _legendre_integrals(degree, y1, y2))
    norm = np.pi * beam.field_stop ** 2 / 4 * np.pi * (s2 * s2 - s1 * s1)
    solid = 2 * np.pi * (x2 - x1)
    overlap = _overlap(s1 * s1, s2 * s2, c1 * c1, c2 * c2) / (s2 * s2 - s1 * s1)
    shadow = 2 * np.pi * _overlap(x1, x2, y1, y2)
    detected = overlap - cs["ext"] * shadow / norm + inside / norm
    return dict(detected=detected, reference=np.full(k.size, overlap),
                ext_by_order=cs["ext_by_order"] * solid / norm, sca_by_order=cs["sca_by_order"] * solid / norm)


def _phase_legendre(sol):
    """Legendre coefficients of the unpolarized differential cross section (|S1|^2 + |S2|^2)/(2k^2)
    in cos(scattering angle), degree 2L (exact: it is a polynomial of that degree)."""
    from numpy.polynomial import legendre
    from .farfield import amplitude_matrix

    degree = 2 * sol.orders.size
    xs, ws = _gauss(degree + 2)
    S1, S2, _, _ = amplitude_matrix(sol, np.arccos(xs))
    k = sol.k[:, -1].real
    phase = (np.abs(S1) ** 2 + np.abs(S2) ** 2) / (2 * k[:, None] ** 2)
    return (phase * ws) @ legendre.legvander(xs, degree) * (2 * np.arange(degree + 1) + 1) / 2


def focused_spheres(radii, n, wavelength, beams, *, l_max=None):
    """A focused beam on a multilayered sphere in a lossless host, batched over wavelengths.

    ``radii``, ``n`` ((N + 1,) or (W, N + 1), host last) as :func:`~pystratify.solve`;
    ``beams`` a :class:`FocusedBeam` or a sequence of them. Returns a :class:`FocusedResult`
    with ``extinction``, ``scattering`` and ``absorption`` (fractions of the beam power;
    Koehler: of the power through the field stop) and ``multipoles['extinction'|
    'scattering']`` per (TM, TE) order, (B, W, 2, L) (scalar: (B, W, 1, L + 1) from l = 0).
    """
    from .solver import solve

    beams, single = _beams(beams)
    wavelength = np.atleast_1d(np.asarray(wavelength, float))
    radii = np.atleast_1d(np.asarray(radii, float))
    n = _per_wavelength(n, wavelength.size, radii.size + 1)
    for beam in beams:
        for row in n:
            check_focused("spheres", row, beam)
    sol = solve(radii, n, wavelength, l_max=l_max)
    t = sol.t  # (2, W, L): TM, TE
    hosts = n[:, -1].real
    t0 = monopole_coefficient(radii, n, wavelength) if any(b.model == "scalar" for b in beams) else None
    kohler = None
    if any(b.illumination == "kohler" for b in beams):
        from .farfield import cross_sections
        plane = cross_sections(sol)
        kohler = dict(ext=plane.ext, ext_by_order=plane.ext_by_order, sca_by_order=plane.sca_by_order), _phase_legendre(sol)
    rows = []
    for beam in beams:
        parts = []
        for host in np.unique(hosts):
            chosen = hosts == host
            k = 2 * np.pi * host / wavelength[chosen]
            illumination = (beam.obscuration / host, beam.na / host)
            collection = tuple(v / host for v in beam.collection)
            if beam.illumination == "kohler":
                cs = {key: value[chosen] for key, value in kohler[0].items()}
                part = _sphere_kohler(cs, kohler[1][chosen], k, beam, illumination, collection)
            elif beam.model == "scalar":
                part = _sphere_general(t[:, chosen], k, sol.orders, illumination, collection, beam.offset, None, True,
                                       t0[chosen])
            elif beam.offset[0] == 0 and beam.offset[1] == 0:  # polarization-independent by symmetry
                part = _sphere_axis(t[:, chosen], k, sol.orders, illumination, collection, beam.offset[2])
            else:
                part = None
                for weight, jones in _jones(beam):
                    one = _sphere_general(t[:, chosen], k, sol.orders, illumination, collection, beam.offset, jones,
                                          False)
                    part = {key: weight * value + (0 if part is None else part[key]) for key, value in one.items()}
            parts.append((chosen, part))
        merged = {}
        for chosen, part in parts:
            for key, value in part.items():
                if key not in merged:
                    merged[key] = np.zeros((wavelength.size,) + value.shape[1:])
                merged[key][chosen] = value
        ext, sca = merged["ext_by_order"].sum(axis=(1, 2)), merged["sca_by_order"].sum(axis=(1, 2))
        top = merged["sca_by_order"][..., -1].sum(axis=1)
        tail = np.divide(top, sca, out=np.zeros_like(sca), where=sca > 0)
        rows.append(dict(detected=merged["detected"], reference=merged["reference"], extinction=ext, scattering=sca,
                         absorption=ext - sca,
                         multipoles=dict(extinction=merged["ext_by_order"], scattering=merged["sca_by_order"],
                                         orders=np.arange(0 if beam.model == "scalar" else 1, sol.orders[-1] + 1)),
                         diagnostics=dict(orders=np.full(wavelength.size, sol.orders[-1]), tail_fraction=tail)))
    return _result(rows, single, ("extinction", "scattering", "absorption"), ("extinction", "scattering"))


# ---------------------------------------------------------------- cylinders

def _row_intervals(sy, s1, s2):
    """Transverse angles alpha (s_x = sqrt(1 - sy^2) sin alpha) of the annulus s1 <= |s| <= s2 on row sy."""
    if sy >= s2:
        return []
    root = np.sqrt(1 - sy * sy)
    outer = np.arcsin(min(1.0, np.sqrt(s2 * s2 - sy * sy) / root))
    if sy >= s1:
        return [(-outer, outer)]
    inner = np.arcsin(min(1.0, np.sqrt(s1 * s1 - sy * sy) / root))
    return [(-outer, -inner), (inner, outer)]


def _row_basis(alpha, sy):
    """Lab vectors of the pupil direction on row sy at transverse angle alpha, and of the
    cylinder's axial-electric (e_N) and axial-magnetic (e_M) polarizations for it (axis = y)."""
    root = np.sqrt(1 - sy * sy)
    ca, sa = np.cos(alpha), np.sin(alpha)
    zero, one = np.zeros_like(alpha), np.ones_like(alpha)
    u = np.stack((root * sa, sy * one, root * ca), axis=-1)
    e_n = np.stack((-sy * sa, root * one, -sy * ca), axis=-1)
    e_m = np.stack((ca, zero, -sa), axis=-1)
    return u, e_n, e_m


def _pupil_polarization(u, jones):
    """Lab unit vector jx (cos f e_t - sin f e_f) + jy (sin f e_t + cos f e_f) at directions u."""
    st = np.hypot(u[..., 0], u[..., 1])
    safe = np.where(st > 0, st, 1.0)
    cf, sf = np.where(st > 0, u[..., 0] / safe, 1.0), np.where(st > 0, u[..., 1] / safe, 0.0)
    ct = u[..., 2]
    e_t = np.stack((ct * cf, ct * sf, -st), axis=-1)
    e_f = np.stack((-sf, cf, np.zeros_like(cf)), axis=-1)
    jx, jy = jones
    return (jx * cf + jy * sf)[..., None] * e_t + (-jx * sf + jy * cf)[..., None] * e_f


def _row_field(alpha, sy, k, illumination, offset, jones, scalar):
    """The beam's amplitude on row sy at angles alpha: (A_N, A_M) on the cylinder's polarizations, or (U,)."""
    s1, s2 = illumination
    u, e_n, e_m = _row_basis(alpha, sy)
    s = np.hypot(u[..., 0], u[..., 1])
    inside = (s >= s1) & (s <= s2)
    a = np.where(inside, aplanatic_amplitude(u[..., 2], s1, s2), 0.0)
    phase = np.exp(-1j * k * (u[..., 0] * offset[0] + u[..., 1] * offset[1] + u[..., 2] * offset[2]))
    if scalar:
        return (a * phase)[None]
    e = _pupil_polarization(u, jones)
    return np.stack((np.sum(e * e_n, -1), np.sum(e * e_m, -1))) * (a * phase)[None]


def _autocorrelation_integral(coefficients, lo, hi):
    """Int_lo^hi |sum_m c_m e^{i m psi}|^2 d psi for coefficients over orders -M..M (last axis), any batch shape
    of (lo, hi) broadcast against the leading axes."""
    size = coefficients.shape[-1]
    lags = np.arange(-(size - 1), size)
    corr = np.stack([np.sum(coefficients[..., max(0, d):size + min(0, d)]
                            * np.conj(coefficients[..., max(0, -d):size - max(0, d)]), axis=-1) for d in lags], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(lags == 0, (hi - lo)[..., None],
                          (np.exp(1j * lags * hi[..., None]) - np.exp(1j * lags * lo[..., None])) / (1j * np.where(lags == 0, 1, lags)))
    return np.real(np.sum(corr * kernel, axis=-1))


class _CylinderRows:
    """Per-row integrands of one wavelength: everything a row s_y contributes, for every beam."""

    def __init__(self, radii, n, wavelength, beams, m_max):
        from .cylindrical import solve_cylinder
        self.solve = solve_cylinder
        self.radii, self.n, self.wavelength, self.beams = radii, n, wavelength, beams
        self.host = n[-1].real
        self.k = 2 * np.pi * self.host / wavelength
        if m_max is None:
            size = max(abs(2 * np.pi * n / wavelength * radii[-1]))
            m_max = max(8, int(np.ceil(size + 4 * size ** (1 / 3) + 8)))
        self.m_max = int(m_max)
        self.orders = np.arange(-self.m_max, self.m_max + 1)
        self.layout = []
        for beam in beams:
            pols = [(1.0, (1.0, 0.0)), (1.0, (0.0, 1.0))] if beam.model == "vector" else [(1.0, None)]
            self.layout.append(pols)
        self.width = 4 + 2 * (self.m_max + 1)  # ext, sca, detected, reference, ext_m, sca_m
        self.size = sum(len(p) for p in self.layout) * self.width

    def sines(self, beam):
        return (beam.obscuration / self.host, beam.na / self.host), tuple(v / self.host for v in beam.collection)

    def breaks(self):
        points = {0.0}
        for beam in self.beams:
            illumination, collection = self.sines(beam)
            points.update(illumination + collection)
        top = max(self.sines(b)[0][1] for b in self.beams)
        return sorted(p for p in points if 0 <= p <= top)

    def __call__(self, sy):
        beta = self.k * sy
        vector = solution = scalar = None
        out = np.zeros(self.size)
        offset = 0
        for beam, pols in zip(self.beams, self.layout):
            illumination, collection = self.sines(beam)
            if sy >= illumination[1]:
                offset += len(pols) * self.width
                continue
            if beam.model == "vector" and vector is None:
                solution = self.solve(self.radii, self.n, self.wavelength, beta=beta, m_max=self.m_max)
                vector = solution.t
            if beam.model == "scalar" and scalar is None:
                effective = np.sqrt(self.n.astype(complex) ** 2 - (self.host * sy) ** 2)
                scalar = self.solve(self.radii, effective, self.wavelength, beta=0, m_max=self.m_max).t[:, 0, 0]
            for _, jones in pols:
                block = out[offset:offset + self.width]
                if beam.illumination == "kohler":
                    self._kohler(block, sy, beam, illumination, collection, jones, vector)
                else:
                    self._coherent(block, sy, beam, illumination, collection, jones,
                                   scalar if beam.model == "scalar" else vector)
                offset += self.width
        return 2 * out  # rows +-s_y contribute equally (mirror symmetry y -> -y)

    def _angles(self, intervals, spread):
        if not intervals:
            return np.zeros(0), np.zeros(0)
        pieces = [_nodes(a, b, self.m_max * (b - a) / np.pi * 2 + 24 + spread * (b - a)) for a, b in intervals]
        return np.concatenate([p[0] for p in pieces]), np.concatenate([p[1] for p in pieces])

    def _collection_cuts(self, sy, illumination, collection):
        edges = [e for a, b in _row_intervals(sy, *illumination) for e in (a, b)]
        return [cut for a, b in _row_intervals(sy, *collection) for cut in _cuts(a, b, edges)]

    def _coherent(self, block, sy, beam, illumination, collection, jones, t):
        scalar = beam.model == "scalar"
        spread = self.k * float(np.hypot(beam.offset[0], beam.offset[2]))
        alpha, w = self._angles(_row_intervals(sy, *illumination), spread)
        field = _row_field(alpha, sy, self.k, illumination, beam.offset, jones, scalar)  # (C, A)
        g = (field * w) @ np.exp(-1j * np.outer(alpha, self.orders))  # (C, 2M + 1)
        if scalar:
            b = t[None] * g
            ext = -(2 / np.pi) * np.real(np.conj(g) * b)[0]
            sca = (2 / np.pi) * np.abs(b[0]) ** 2
            outgoing = b / np.pi
        else:
            b = np.einsum("mij,jm->im", t, np.stack((g[0], 1j * g[1])))
            ext = -(2 / np.pi) * np.real(np.conj(g[0]) * b[0] - 1j * np.conj(g[1]) * b[1])
            sca = (2 / np.pi) * (np.abs(b[0]) ** 2 + np.abs(b[1]) ** 2)
            outgoing = np.stack((b[0], -1j * b[1])) / np.pi  # on (e_N, e_M)
        detected = reference = 0.0
        for a, c in self._collection_cuts(sy, illumination, collection):
            phi, wc = _nodes(a, c, self.m_max * (c - a) / np.pi * 2 + 24 + spread * (c - a))
            beam_field = _row_field(phi, sy, self.k, illumination, beam.offset, jones, scalar)
            total = beam_field + outgoing @ np.exp(1j * np.outer(self.orders, phi))
            detected += np.sum(np.abs(total) ** 2 @ wc)
            reference += np.sum(np.abs(beam_field) ** 2 @ wc)
        self._store(block, ext, sca, detected, reference)

    def _kohler(self, block, sy, beam, illumination, collection, jones, t):
        # the condenser's edges cut the lit angles: the shadow is seen on one side only
        edges = [e for a, b in _row_intervals(sy, *collection) for e in (a, b)]
        alpha, w = self._angles([cut for a, b in _row_intervals(sy, *illumination) for cut in _cuts(a, b, edges)], 0.0)
        u, e_n, e_m = _row_basis(alpha, sy)
        e = _pupil_polarization(u, jones)
        c_n, c_m = np.sum(e * e_n, -1), np.sum(e * e_m, -1)  # real, c_n^2 + c_m^2 = 1
        unit_n, unit_m = t[:, :, 0], 1j * t[:, :, 1]  # scattered (N, M) of unit axial-electric / -magnetic waves
        b = c_n[:, None, None] * unit_n[None] + c_m[:, None, None] * unit_m[None]  # (A, m, 2)
        ext = -(4 / self.k) * np.real(c_n[:, None] * b[..., 0] - 1j * c_m[:, None] * b[..., 1])  # (A, m)
        sca = (4 / self.k) * np.sum(np.abs(b) ** 2, axis=-1)
        s = np.hypot(u[:, 0], u[:, 1])
        seen = (s >= collection[0]) & (s <= collection[1])
        landed = np.zeros(alpha.size)
        coefficients = np.stack((b[..., 0], -1j * b[..., 1]), axis=1)  # (A, 2, m) on (e_N, e_M)
        for a, c in _row_intervals(sy, *collection):
            lo, hi = (a - alpha)[:, None], (c - alpha)[:, None]
            landed += (2 / (np.pi * self.k)) * np.sum(_autocorrelation_integral(coefficients, lo, hi), axis=-1)
        scale = (beam.field_stop / (np.pi * beam.field_stop ** 2 / 4)) / (np.pi * (illumination[1] ** 2 - illumination[0] ** 2))
        shadow = np.sum(w * seen * ext.sum(axis=1))
        self._store(block, scale * (ext.T @ w), scale * (sca.T @ w), scale * (np.sum(w * landed) - shadow), 0.0)

    def _store(self, block, ext, sca, detected, reference):
        M = self.m_max
        index = np.abs(self.orders)
        block[0], block[1], block[2], block[3] = np.sum(ext), np.sum(sca), detected, reference
        block[4:4 + M + 1] = np.bincount(index, weights=ext, minlength=M + 1)
        block[4 + M + 1:] = np.bincount(index, weights=sca, minlength=M + 1)


def focused_cylinders(radii, n, wavelength, beams, *, m_max=None, tolerance=1e-6, max_evaluations=20000):
    """A focused beam on infinite concentric cylinders (axis along y) in a lossless host.

    ``radii``, ``n`` ((N + 1,) or (W, N + 1), host last); ``beams`` a :class:`FocusedBeam`
    or a sequence of them, integrated together so that every oblique solve is shared.
    Coherent and Koehler vector beams report both pupil polarizations in
    ``by_polarization`` ('x' across the axis, 'y' along it); ``multipoles`` holds the
    extinction and scattering of each order |m|, (B, W, M + 1).
    """
    from .integration import integrate

    beams, single = _beams(beams)
    wavelength = np.atleast_1d(np.asarray(wavelength, float))
    radii = np.atleast_1d(np.asarray(radii, float))
    n = _per_wavelength(n, wavelength.size, radii.size + 1)
    for beam in beams:
        for row in n:
            check_focused("cylinders", row, beam)
    W = wavelength.size
    per_wavelength = []
    for w, lam in enumerate(wavelength):
        rows = _CylinderRows(radii, n[w], lam, beams, m_max)
        integral = integrate(rows, rows.breaks(), tolerance, max_evaluations)
        per_wavelength.append((rows, integral))
    M = max(r.m_max for r, _ in per_wavelength)
    results = []
    for b, beam in enumerate(beams):
        pols = per_wavelength[0][0].layout[b]
        keys = ("extinction", "scattering", "detected", "reference")
        values = {p: {k: np.zeros(W) for k in keys} for p in range(len(pols))}
        shares = {p: {k: np.zeros((W, M + 1)) for k in ("extinction", "scattering")} for p in range(len(pols))}
        error, evaluations, converged, tail = np.zeros(W), np.zeros(W, int), np.zeros(W, bool), np.zeros(W)
        for w, (rows, integral) in enumerate(per_wavelength):
            start = sum(len(p) for p in rows.layout[:b]) * rows.width
            m = rows.m_max
            for p in range(len(pols)):
                block = integral.value[start + p * rows.width:start + (p + 1) * rows.width]
                for i, key in enumerate(keys):
                    values[p][key][w] = block[i]
                shares[p]["extinction"][w, :m + 1] = block[4:4 + m + 1]
                shares[p]["scattering"][w, :m + 1] = block[4 + m + 1:]
            if beam.illumination == "kohler":
                illumination, collection = rows.sines(beam)
                overlap = _overlap(illumination[0] ** 2, illumination[1] ** 2, collection[0] ** 2, collection[1] ** 2)
                for p in range(len(pols)):
                    values[p]["reference"][w] = overlap / (illumination[1] ** 2 - illumination[0] ** 2)
                    values[p]["detected"][w] += values[p]["reference"][w]
            error[w], evaluations[w], converged[w] = integral.error, integral.evaluations, integral.converged
            top = sum(shares[p]["scattering"][w, m] for p in range(len(pols)))
            total = sum(values[p]["scattering"][w] for p in range(len(pols)))
            tail[w] = top / total if total > 0 else 0.0
        for p in values:
            values[p]["absorption"] = values[p]["extinction"] - values[p]["scattering"]
        if beam.model == "vector":
            weights = {"x": [1.0, 0.0], "y": [0.0, 1.0], "unpolarized": [0.5, 0.5]}[beam.polarization]
        else:
            weights = [1.0]
        chosen = {k: sum(wt * values[p][k] for p, wt in enumerate(weights)) for k in values[0]}
        multipoles = {k: sum(wt * shares[p][k] for p, wt in enumerate(weights)) for k in ("extinction", "scattering")}
        multipoles["orders"] = np.arange(M + 1)
        by_polarization = {} if beam.model == "scalar" else {"x": values[0], "y": values[1]}
        results.append(dict(**chosen, multipoles=multipoles, by_polarization=by_polarization,
                            diagnostics=dict(error=error, evaluations=evaluations, converged=converged,
                                             orders=np.array([r.m_max for r, _ in per_wavelength]), tail_fraction=tail)))
    polarized = all(beam.model == "vector" for beam in beams)
    if not polarized:
        for row in results:
            row["by_polarization"] = {}
    return _result(results, single, ("extinction", "scattering", "absorption"), ("extinction", "scattering"), polarized)


# ---------------------------------------------------------------- near fields

@dataclass(frozen=True)
class FocusedField:
    """Fields of a focused beam at Cartesian points, per pupil polarization: ``e``, ``h``
    (P, points..., 3) of the total field and ``e_incident``, ``h_incident`` of the beam
    alone (NaN where it is not defined: inside a particle, or within a film stack), P = 1
    (x or y) or 2 (x, y; unpolarized averages their intensities). Normalized so that
    the beam alone has |E|^2 = 1 at its focus (the scalar model: |U|^2 = 1, in ``e[..., 0]``);
    H is in Gaussian units, |H| = n |E| for a plane wave.
    """

    e: np.ndarray
    h: np.ndarray
    e_incident: np.ndarray
    h_incident: np.ndarray
    polarizations: tuple

    @property
    def intensity_e(self):
        return np.mean(np.sum(np.abs(self.e) ** 2, axis=-1), axis=0)

    @property
    def intensity_h(self):
        return np.mean(np.sum(np.abs(self.h) ** 2, axis=-1), axis=0)


def _field_jones(beam):
    pols = {"x": [(1.0, 0.0)], "y": [(0.0, 1.0)]}.get(beam.polarization, [(1.0, 0.0), (0.0, 1.0)])
    return pols, tuple("x" if j[0] else "y" for j in pols)


def _focal_integrals(points, k, illumination, z_phase=None, scalar=False):
    """Richards-Wolf integrals of the aplanatic beam focused at the origin of ``points`` (..., 3):
    I0 = pi Int a (1 + c) J0, I1 = 2 pi Int a s J1, I2 = pi Int a (1 - c) J2 (s dt = dx), with the
    phase exp(i k z c); the scalar beam's 2 pi Int a J0. Returns them and (rho, phi)."""
    from scipy.special import jv

    s1, s2 = illumination
    x1, x2 = np.sqrt(1 - s2 * s2), np.sqrt(1 - s1 * s1)
    rho = np.hypot(points[..., 0], points[..., 1]).ravel()
    phi = np.arctan2(points[..., 1], points[..., 0]).ravel()
    z = points[..., 2].ravel()
    count = 40 + k * (float(np.max(rho, initial=0)) * (s2 - s1) + float(np.max(np.abs(z), initial=0)) * (x2 - x1))
    x, w = _nodes(x1, x2, count)
    sin = np.sqrt(1 - x * x)
    a = aplanatic_amplitude(x, s1, s2) * w
    phase = np.exp(1j * k * np.outer(z, x)) * a[None]
    argument = k * np.outer(rho, sin)
    if scalar:
        return (2 * np.pi * np.sum(phase * jv(0, argument), axis=1),), rho, phi
    i0 = np.pi * np.sum(phase * (1 + x) * jv(0, argument), axis=1)
    i1 = 2 * np.pi * np.sum(phase * sin * jv(1, argument), axis=1)
    i2 = np.pi * np.sum(phase * (1 - x) * jv(2, argument), axis=1)
    return (i0, i1, i2), rho, phi


def focal_field(points, k, illumination, jones, n=1.0, scalar=False):
    """E and H of the aplanatic beam (Int |A|^2 dOmega = 1) focused at the origin, at points (..., 3)
    in its own medium (wavenumber k, index n); the scalar beam's U in E[..., 0]."""
    points = np.asarray(points, float)
    integrals, rho, phi = _focal_integrals(points, k, illumination, scalar=scalar)
    shape = points.shape
    if scalar:
        e = np.zeros((rho.size, 3), complex)
        e[:, 0] = integrals[0]
        return e.reshape(shape), np.zeros(shape, complex)
    i0, i1, i2 = integrals
    c2, s2, c1, s1 = np.cos(2 * phi), np.sin(2 * phi), np.cos(phi), np.sin(phi)
    ex = np.stack((i0 + i2 * c2, i2 * s2, -1j * i1 * c1), axis=-1)  # x pupil
    ey = np.stack((i2 * s2, i0 - i2 * c2, -1j * i1 * s1), axis=-1)  # y pupil
    jx, jy = jones
    e = jx * ex + jy * ey
    h = n * (jx * ey - jy * ex)  # u x e^x = e^y, u x e^y = -e^x
    return e.reshape(shape), h.reshape(shape)


def focal_amplitude(illumination, scalar=False):
    """|E| (scalar: |U|) at the focus of the normalized beam: the near fields' unit."""
    (value, *_), _, _ = _focal_integrals(np.zeros((1, 3)), 1.0, illumination, scalar=scalar)
    return float(np.real(value[0]))


def _sphere_coefficients(k, orders, illumination, offset, jones, scalar):
    """alpha_lm, beta_lm (vector) or alpha_lm (scalar, l from 0) of the beam, as {m: arrays over l = 0..L}."""
    s1, s2 = illumination
    L = orders.size
    lateral = np.hypot(offset[0], offset[1])
    count_phi = int(2 ** np.ceil(np.log2(2 * (L + k * lateral * s2) + 48)))
    x, w, phi = _pupil_grid(s1, s2, L + 40 + k * (lateral + abs(offset[2])), count_phi)
    spectrum = _beam_spectrum(x, phi, np.array([k]), offset, illumination, jones, scalar)[0]
    l = np.arange(L + 1)
    with np.errstate(divide="ignore"):
        scale = np.where(l > 0, 1 / np.sqrt(2 * np.pi * l * (l + 1.0)), 0.0)
    out = {}
    for m, P, pi, tau in vector_harmonics(L, x):
        col = m % count_phi
        if scalar:
            out[m] = (spectrum[0, :, col] * w) @ P.T / np.sqrt(2 * np.pi)
        else:
            a_t, a_f = spectrum[0, :, col] * w, spectrum[1, :, col] * w
            out[m] = ((-(pi @ a_t) + 1j * (tau @ a_f)) * scale, (-1j * (tau @ a_t) - (pi @ a_f)) * scale)
    return out


def focused_field_spheres(radii, n, wavelength, beam, points, *, l_max=None):
    """Near field of a focused beam on a multilayered sphere, at Cartesian points (..., 3)
    about its centre, one wavelength: a :class:`FocusedField`."""
    from .nearfield import radial_functions
    from .solver import solve, TM, TE

    radii = np.atleast_1d(np.asarray(radii, float))
    n = np.asarray(n, complex)
    check_focused("spheres", n, beam)
    if beam.illumination != "coherent":
        raise ValueError("near fields are computed for coherent illumination")
    points = np.asarray(points, float)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    sol = solve(radii, n, wavelength, l_max=l_max)
    host, k = n[-1].real, 2 * np.pi * n[-1].real / wavelength
    illumination = (beam.obscuration / host, beam.na / host)
    scalar = beam.model == "scalar"
    unit = focal_amplitude(illumination, scalar)
    L = sol.orders.size
    l = np.arange(L + 1)
    r = np.maximum(np.linalg.norm(flat, axis=-1), 1e-9 * radii[0])
    theta = np.arctan2(np.hypot(flat[:, 0], flat[:, 1]), flat[:, 2])
    phi = np.arctan2(flat[:, 1], flat[:, 0])
    radii_unique, index = np.unique(r, return_inverse=True)
    functions, kr, shell = radial_functions(sol, 0, radii_unique, polarisations=(TM, TE), offsets=(-1, 0),
                                            with_incident=False, host_scattered_only=True)
    shell = shell[index]
    in_host = shell == radii.size
    def pad(v):  # an l = 0 column, then back to every point
        return np.concatenate((np.zeros((v.shape[0], 1), complex), v), axis=1)[index]

    radial = {}
    for p in (TM, TE):
        f_prev, f = functions[p]
        g = f / kr[:, None]
        radial[p] = (pad(f), pad(g), pad(f_prev - sol.orders * g))
    weight = 4 * np.pi * 1j ** l
    if scalar:
        t0 = monopole_coefficient(radii, n, wavelength)[0]
        # the scalar radial function of order l is the TE one (U, dU/dr continuous), plus the monopole
        monopole = _scalar_monopole_radial(radii, n, wavelength, t0, r, shell)
        f_scalar = radial[TE][0].copy()
        f_scalar[:, 0] = monopole
    root = np.sqrt(l * (l + 1.0))
    with np.errstate(divide="ignore"):
        scale = np.where(l > 0, 1 / np.sqrt(2 * np.pi * l * (l + 1.0)), 0.0)
    jones_list, names = _field_jones(beam)
    e_all, h_all, e_inc_all, h_inc_all = [], [], [], []
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    for jones in jones_list:
        coefficients = _sphere_coefficients(k, sol.orders, illumination, beam.offset, jones, scalar)
        e_sph = np.zeros((flat.shape[0], 3), complex)  # (r, theta, phi)
        h_sph = np.zeros_like(e_sph)
        for m, P, pi, tau in vector_harmonics(L, ct):
            wave = np.exp(1j * m * phi)
            if scalar:
                e_sph[:, 0] += wave * np.sum(weight * coefficients[m] * f_scalar * P.T, axis=1) / np.sqrt(2 * np.pi)
                continue
            alpha, beta = coefficients[m]
            fe, ge, de = radial[TM]
            fm, gm, dm = radial[TE]
            A, B = weight * alpha, weight * beta
            Pt, pit, taut = P.T / np.sqrt(2 * np.pi), pi.T * scale, tau.T * scale
            e_sph[:, 0] += wave * np.sum(B * root * ge * Pt, axis=1)
            e_sph[:, 1] += wave * np.sum(-A * fm * pit + B * de * taut, axis=1)
            e_sph[:, 2] += wave * np.sum(-1j * A * fm * taut + 1j * B * de * pit, axis=1)
            h_sph[:, 0] += wave * np.sum(1j * A * root * gm * Pt, axis=1)
            h_sph[:, 1] += wave * np.sum(1j * A * dm * taut + 1j * B * fe * pit, axis=1)
            h_sph[:, 2] += wave * np.sum(-A * dm * pit - B * fe * taut, axis=1)
        h_sph *= -1j * (sol.n[0] / sol.mu[0])[shell][:, None]

        def cartesian(v):
            return np.stack((st * cp * v[:, 0] + ct * cp * v[:, 1] - sp * v[:, 2],
                             st * sp * v[:, 0] + ct * sp * v[:, 1] + cp * v[:, 2],
                             ct * v[:, 0] - st * v[:, 1]), axis=-1)

        e, h = (cartesian(e_sph), cartesian(h_sph)) if not scalar else (e_sph, h_sph)
        e_inc, h_inc = focal_field(flat - np.array(beam.offset), k, illumination, jones, host, scalar)
        e = e + np.where(in_host[:, None], e_inc, 0)
        h = h + np.where(in_host[:, None], h_inc, 0)
        e_inc = np.where(in_host[:, None], e_inc, np.nan)
        h_inc = np.where(in_host[:, None], h_inc, np.nan)
        for target, value in ((e_all, e), (h_all, h), (e_inc_all, e_inc), (h_inc_all, h_inc)):
            target.append((value / unit).reshape(shape + (3,)))
    return FocusedField(np.array(e_all), np.array(h_all), np.array(e_inc_all), np.array(h_inc_all), names)


def _scalar_monopole_radial(radii, n, wavelength, t0, r, shell):
    """The l = 0 scalar radial function at radii r: j_0 + T_0 h_0 scattered part (h_0 only) in the
    host and the regular solution inside, matched in value and derivative through every interface."""
    k = 2 * np.pi * np.asarray(n, complex) / wavelength
    out = np.zeros(r.size, complex)
    R = radii[-1]
    kh = k[-1]
    host = shell == radii.size
    out[host] = t0 * np.exp(1j * kh * r[host]) / (1j * kh * r[host])  # h_0 = -i e^{ix}/x
    # inside: W = rU, with W and W' continuous; start from the host's total W at R and integrate inwards
    w_val = (np.sin(kh * R) + t0 * (-1j) * np.exp(1j * kh * R)) / kh
    w_der = np.cos(kh * R) + t0 * np.exp(1j * kh * R)
    for j in range(radii.size - 1, -1, -1):
        outer = radii[j]
        inner = radii[j - 1] if j else 0.0
        kj = k[j]
        chosen = shell == j
        d = r[chosen] - outer
        out[chosen] = (w_val * np.cos(kj * d) + w_der * np.sin(kj * d) / kj) / r[chosen]
        d = inner - outer
        w_val, w_der = w_val * np.cos(kj * d) + w_der * np.sin(kj * d) / kj, -w_val * kj * np.sin(kj * d) + w_der * np.cos(kj * d)
    return out


def _cylinder_row_fields(sol, flat, incident, host_scattered_only):
    """E, H (lab frame, axis along y) of one row's waves at lab points (P, 3), for each of the per-order
    incident vectors ``incident`` (I, orders, 2) on (axial-electric, axial-magnetic): shape (I, P, 3).
    The host keeps only outgoing waves when ``host_scattered_only``. Mirrors CylinderSolution.field,
    evaluating the radial functions once per distinct radius."""
    from .cylindrical import vectors
    from .special import cylinder_pair

    xc, yc, zc = flat[:, 2], flat[:, 0], flat[:, 1]  # cylinder frame: x_c = z, y_c = x, z_c = y
    rho, phi = np.hypot(xc, yc), np.arctan2(yc, xc)
    rho = np.maximum(rho, 1e-12 * sol.wavelength)
    count = incident.shape[0]
    E, H = np.zeros((count, flat.shape[0], 3), complex), np.zeros((count, flat.shape[0], 3), complex)
    host = len(sol.radii)
    layer = np.searchsorted(sol.radii, rho)
    unique, index = np.unique(rho, return_inverse=True)
    order = np.argsort(index, kind="stable")
    bounds = np.searchsorted(index[order], np.arange(unique.size + 1))
    orders = sol.orders
    phase_axis = (1j ** orders)[None] * np.exp(1j * np.outer(phi, orders) + 1j * sol.beta * zc[:, None])  # (P, m)
    for u, radius in enumerate(unique):
        chosen = order[bounds[u]:bounds[u + 1]]
        j = int(layer[chosen[0]])
        pairs = cylinder_pair(sol.q[j], radius, orders)
        k = 2 * np.pi * sol.n[j] / sol.wavelength
        logs = [sol.log_a[j], sol.log_b[j]]
        if j == host and host_scattered_only:
            logs = [None, sol.log_b[j]]
        phase = phase_axis[chosen]
        local_e = np.zeros((count, chosen.size, 3), complex)
        local_h = np.zeros_like(local_e)
        for b, log in enumerate(logs):
            if log is None:
                continue
            M, N = vectors(sol.q[j], k, sol.beta, radius, orders, pairs[b])
            with np.errstate(under="ignore"):
                coefficient = np.einsum("mij,cmj->cmi", np.exp(log + pairs[b][2][:, None, None]), incident)
            local_e += np.einsum("pm,cmk->cpk", phase, coefficient[..., :1] * N + coefficient[..., 1:] * M)
            local_h += -1j * sol.n[j] / sol.mu[j] * np.einsum("pm,cmk->cpk", phase, coefficient[..., :1] * M + coefficient[..., 1:] * N)
        c, s = np.cos(phi[chosen]), np.sin(phi[chosen])
        for target, v in ((E, local_e), (H, local_h)):
            ex, ey, ez = c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]
            target[:, chosen] = np.stack((ey, ez, ex), axis=-1)  # back to lab (x, y, z) = (y_c, z_c, x_c)
    return E, H


def focused_field_cylinders(radii, n, wavelength, beam, points, *, m_max=None, rows=16):
    """Near field of a focused beam on concentric cylinders (axis along y), at lab points (..., 3),
    one wavelength: the coherent sum over the pupil rows s_y (Gauss-Legendre in the sin^2-mapped
    row variable, ``rows`` per segment; both pupil polarizations share every row's solve)."""
    from .cylindrical import solve_cylinder

    radii = np.atleast_1d(np.asarray(radii, float))
    n = np.asarray(n, complex)
    check_focused("cylinders", n, beam)
    if beam.illumination != "coherent":
        raise ValueError("near fields are computed for coherent illumination")
    points = np.asarray(points, float)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    host = n[-1].real
    k = 2 * np.pi * host / wavelength
    illumination = (beam.obscuration / host, beam.na / host)
    scalar = beam.model == "scalar"
    unit = focal_amplitude(illumination, scalar)
    if m_max is None:
        size = max(abs(2 * np.pi * n / wavelength * radii[-1]))
        m_max = max(8, int(np.ceil(size + 4 * size ** (1 / 3) + 8)))
    orders = np.arange(-m_max, m_max + 1)
    s1, s2 = illumination
    extent = float(np.max(np.abs(flat[:, 1]), initial=0)) + abs(beam.offset[1])
    nodes = [_mapped_nodes(a, b, rows + k * extent * (b - a)) for a, b in _cuts(-s2, s2, (-s1, s1))]
    sy_nodes = np.concatenate([p[0] for p in nodes])
    sy_weights = np.concatenate([p[1] for p in nodes])
    in_host = np.hypot(flat[:, 0], flat[:, 2]) >= radii[-1]
    jones_list, names = _field_jones(beam)
    if scalar:
        jones_list, names = [None], ("scalar",)
    E = np.zeros((len(jones_list), flat.shape[0], 3), complex)
    H = np.zeros_like(E)
    for sy, weight in zip(sy_nodes, sy_weights):
        intervals = _row_intervals(abs(sy), s1, s2)
        if not intervals:
            continue
        spread = k * np.hypot(beam.offset[0], beam.offset[2])
        pieces = [_nodes(a, b, 2 * m_max * (b - a) / np.pi + 24 + spread * (b - a)) for a, b in intervals]
        alpha = np.concatenate([p[0] for p in pieces])
        w = np.concatenate([p[1] for p in pieces])
        basis = np.exp(-1j * np.outer(alpha, orders))
        if scalar:
            effective = np.sqrt(n ** 2 - (host * sy) ** 2)
            sol = _with_beta(solve_cylinder(radii, effective, wavelength, beta=0, m_max=m_max), k * sy)
            g = (_row_field(alpha, sy, k, illumination, beam.offset, None, True) * w) @ basis
            incident = np.stack((g[0], np.zeros_like(g[0])), axis=-1)[None]
        else:
            sol = solve_cylinder(radii, n, wavelength, beta=k * sy, m_max=m_max)
            incident = []
            for jones in jones_list:
                g = (_row_field(alpha, sy, k, illumination, beam.offset, jones, False) * w) @ basis
                incident.append(np.stack((g[0], 1j * g[1]), axis=-1))
            incident = np.array(incident)
        e_row, h_row = _cylinder_row_fields(sol, flat, incident, True)
        if scalar:  # U is the axial component of the axial-electric wave at normal incidence
            e_row = np.stack((e_row[..., 1], np.zeros_like(e_row[..., 1]), np.zeros_like(e_row[..., 1])), axis=-1)
            h_row = np.zeros_like(h_row)
        E += weight * e_row
        H += weight * h_row
    e_all, h_all, e_inc_all, h_inc_all = [], [], [], []
    for index, jones in enumerate(jones_list):
        e_inc, h_inc = focal_field(flat - np.array(beam.offset), k, illumination, jones or (1.0, 0.0), host, scalar)
        e = E[index] + np.where(in_host[:, None], e_inc, 0)
        h = H[index] + np.where(in_host[:, None], h_inc, 0)
        e_inc = np.where(in_host[:, None], e_inc, np.nan)
        h_inc = np.where(in_host[:, None], h_inc, np.nan)
        for target, value in ((e_all, e), (h_all, h), (e_inc_all, e_inc), (h_inc_all, h_inc)):
            target.append((value / unit).reshape(shape + (3,)))
    return FocusedField(np.array(e_all), np.array(h_all), np.array(e_inc_all), np.array(h_inc_all), names)


def _with_beta(sol, beta):
    """A normal-incidence scalar solution carried along y with axial wavenumber beta (phase only)."""
    from dataclasses import replace
    return replace(sol, beta=beta)


def _film_waves(data, layer, depth):
    """Forward and backward amplitudes of a coh_tmm solution in ``layer`` at local depths, as
    :func:`~pystratify.planar.position_resolved` forms them (incoming |E| = 1)."""
    kz = data["kz_list"][layer]
    if layer > 0:
        v, w = data["vw_list"][layer]
    else:
        v, w = 1, data["r"]
    forward = v * np.exp(1j * kz * depth)
    if 0 < layer < len(data["d_list"]) - 1 and "backward_bottom" in data:
        backward = data["backward_bottom"][layer] * np.exp(1j * kz * (data["d_list"][layer] - depth))
    else:
        backward = w * np.exp(-1j * kz * depth)
    return forward, backward


def focused_field_films(n, thickness, wavelength, beam, points):
    """Near field of a focused beam in a coherent planar stack, at points (..., 3) with z the
    depth below the first interface (negative in the ambient): every pupil direction's coh_tmm
    field, summed over the azimuth in closed form (Bessel J0, J1, J2), one wavelength."""
    from scipy.special import jv
    from .planar import coh_tmm, find_in_structure_with_inf

    n = np.asarray(n, complex)
    thickness = np.asarray(thickness, float)
    check_focused("films", n, beam)
    if beam.illumination != "coherent":
        raise ValueError("near fields are computed for coherent illumination")
    points = np.asarray(points, float)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    ambient = n[0].real
    k = 2 * np.pi * ambient / wavelength
    s1, s2 = beam.obscuration / ambient, beam.na / ambient
    unit = focal_amplitude((s1, s2))
    rel = flat - np.array([beam.offset[0], beam.offset[1], 0.0])
    rho, phi_r = np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])
    depths, at_depth = np.unique(flat[:, 2], return_inverse=True)
    located = [find_in_structure_with_inf(thickness, z) if z >= 0 else (0, z) for z in depths]
    layers = np.array([p[0] for p in located])
    local = np.array([p[1] for p in located], float)
    x1, x2 = np.sqrt(1 - s2 * s2), np.sqrt(1 - s1 * s1)
    count = 40 + k * (float(np.max(rho, initial=0)) * (s2 - s1) + (float(np.max(np.abs(depths), initial=0)) + abs(beam.offset[2])) * (x2 - x1))
    x, w = _nodes(x1, x2, count)
    sin = np.sqrt(1 - x * x)
    amplitude = aplanatic_amplitude(x, s1, s2) * w * np.exp(-1j * k * x * beam.offset[2])
    # per node and depth: local p fields (Px, Pz; H: Phy) and s fields (Sy; H: Shx, Shz)
    P = np.zeros((x.size, depths.size, 3), complex)
    S = np.zeros((x.size, depths.size, 3), complex)
    for i, c in enumerate(x):
        theta = np.arccos(c)
        for pol, target in (("p", P), ("s", S)):
            data = coh_tmm(pol, n, thickness, theta, wavelength)
            for layer in np.unique(layers):
                chosen = layers == layer
                forward, backward = _film_waves(data, int(layer), local[chosen])
                th, index = data["th_list"][layer], n[layer]
                if pol == "p":
                    target[i, chosen, 0] = (forward - backward) * np.cos(th)
                    target[i, chosen, 1] = (-forward - backward) * np.sin(th)
                    target[i, chosen, 2] = index * (forward + backward)
                else:
                    target[i, chosen, 0] = forward + backward
                    target[i, chosen, 1] = index * np.cos(th) * (backward - forward)
                    target[i, chosen, 2] = index * np.sin(th) * (forward + backward)
    jones_list, names = _field_jones(beam)
    e_all = np.zeros((len(jones_list), flat.shape[0], 3), complex)
    h_all = np.zeros_like(e_all)
    chunk = max(1, int(2e6 // max(1, x.size)))
    for start in range(0, flat.shape[0], chunk):
        part = slice(start, start + chunk)
        argument = k * np.outer(sin, rho[part])  # (nodes, points)
        J0, J1, J2 = jv(0, argument), jv(1, argument), jv(2, argument)
        Px, Pz, Phy = (P[:, at_depth[part], c] for c in range(3))
        Sy, Shx, Shz = (S[:, at_depth[part], c] for c in range(3))
        for index, (jx, jy) in enumerate(jones_list):
            angle = 0.0 if (jx, jy) == (1.0, 0.0) else np.pi / 2  # the y pupil is the x pupil turned by +90 degrees
            ph = phi_r[part] - angle
            c2, s2_, c1, s1_ = np.cos(2 * ph), np.sin(2 * ph), np.cos(ph), np.sin(ph)
            Ex = np.pi * ((Px + Sy) * J0 + (Sy - Px) * J2 * c2)
            Ey = -np.pi * (Px - Sy) * J2 * s2_
            Ez = 2j * np.pi * Pz * J1 * c1
            Hx = np.pi * (Phy + Shx) * J2 * s2_
            Hy = np.pi * ((Phy - Shx) * J0 - (Phy + Shx) * J2 * c2)
            Hz = -2j * np.pi * Shz * J1 * s1_
            e = np.stack([amplitude @ v for v in (Ex, Ey, Ez)], axis=-1)
            h = np.stack([amplitude @ v for v in (Hx, Hy, Hz)], axis=-1)
            if angle:
                e = np.stack((-e[:, 1], e[:, 0], e[:, 2]), axis=-1)
                h = np.stack((-h[:, 1], h[:, 0], h[:, 2]), axis=-1)
            e_all[index, part], h_all[index, part] = e, h
    e_all = (e_all / unit).reshape((len(jones_list),) + shape + (3,))
    h_all = (h_all / unit).reshape((len(jones_list),) + shape + (3,))
    nan_all = np.full(e_all.shape, np.nan + 0j)
    return FocusedField(e_all, h_all, nan_all, nan_all, names)
