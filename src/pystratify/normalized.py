"""The normalized formulation of the accompanying paper: Green's dyadic at the source, per order.

Every per-order term is an explicit geometric factor times quantities of order
unity built from three auxiliary functions of a single argument z,

    A_l(z) = psi_l'/psi_l                        (downward recurrence),
    P_l(z) = psi_l xi_l = -i z jbar_l hbar_l/(2l+1) = i/(B_l - A_l),
    jbar_l(z) = (2l+1)!! j_l(z) / z^l            (from consecutive ratios, as a logarithm),

with B_l = xi_l'/xi_l from its own upward recurrence.  P_l is formed order by
order, so no product runs through a real zero of psi_l, and the ratios of jbar
reuse the sums A_l + l/z rounded in the recurrence for A, so that P/jbar stays
exact at such a zero (see :func:`auxiliary`).  The shell structure enters through the local
reflection ratios rho = R xi/psi (swept outwards) and sigma = S psi/xi (swept
inwards), carried across a shell by the propagator

    G(r_a, r_b) = (r_a/r_b)^(2l+2) [jbar(k r_a)/jbar(k r_b)]^2 P(k r_b)/P(k r_a),

and across an interface by Moebius maps of logarithmic derivatives.  The
maps are written with the mismatch terms m11 = (l+1)(g-1)/x~ - (f r_in - r_out)
and m33 = (f X_in - X_out) - l(g-1)/x~, where r = psi_{l+1}/psi_l,
X = xi_{l-1}/xi_l and g - 1 is the material contrast (exactly zero for TE at
non-magnetic interfaces), so no difference of nearly equal logarithmic
derivatives is ever formed.

At the emitter (argument x, polarization TM or TE) the scattered Green's
function of two radial functionals f = a u + b u' and g = c u + d u' is

    P [a c S + (a d + b c) S^m + b d S^d],
    S   = (rho + sigma + 2 rho sigma)/(1 - rho sigma),
    S^m = (rho B + sigma A + rho sigma (A + B))/(1 - rho sigma),
    S^d = (rho B^2 + sigma A^2 + 2 rho sigma A B)/(1 - rho sigma),

and the radiated amplitude of f is a F + b F' with F = core (1 + rho) and
F' = core (A + rho B), where the core carries the single explicit factor
(k_h r)^(l+1)/(2l+1)!!.  For a dipole the total rate is 1 + sum Re[...] (shell
normalization), the frequency shift (1/2) sum Im[...] in units of the same free
rate, and the radiative rate sum |...|^2.  Nothing is computed that can
overflow; only the geometric factors can underflow, which is harmless.

Near a lossless interface the high-order terms are almost purely reactive.  Every
quantity above is then a dominant part plus a small one that is carried
multiplicatively (Re P = psi^2 = Im B/|B - A|^2, with Im B from a recurrence that
multiplies it), so Re and Im of the per-order forms keep full relative precision
as long as TM and TE are kept apart and sources enter through real weights.

This module is independent of :mod:`pystratify.solver` and
:mod:`pystratify.decay` (which carry the same physics as logarithms); the tests
check the two against each other and against extended-precision transfer
matrices.  Shells 1..N+1 of the paper are indices 0..N here; the emitter must
be in a lossless shell and the host must be lossless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from .convergence import orders_needed, tail_estimate, truncation_order
from .solver import TE, TM

__all__ = [
    "auxiliary",
    "log_double_factorial",
    "NormalizedTerms",
    "normalized_terms",
    "NormalizedRates",
    "normalized_decay_rates",
]

#: extra orders above max(L, |z|) at which the downward recurrence for A_l starts
_A_HEADROOM = 20
#: for Im z <= -_GAIN_SWITCH (strongly amplifying media) B_l is not recurred upwards (see auxiliary)
_GAIN_SWITCH = 1.0
#: default most orders tried by the automatic truncation (as :func:`~pystratify.decay_rates`)
L_CAP = 20000


def log_double_factorial(l):
    """ln (2l+1)!! = ln Gamma(2l+2) - l ln 2 - ln Gamma(l+1)."""
    l = np.asarray(l, dtype=float)
    return gammaln(2 * l + 2) - l * np.log(2) - gammaln(l + 1)


def auxiliary(z, L):
    """A_l, B_l, P_l and ln jbar_l for l = 0..L at the complex argument z (arrays of length L+1).

    ``A``: downward recurrence A_{l-1} = l/z - 1/s_l, s_l = A_l + l/z = psi_{l-1}/psi_l,
    from A = 0 at l = max(L, |z|) + 20 (Lentz 1976, Wiscombe 1980).

    ``B``, ``P``: B_l = 1/(l/z - B_{l-1}) - l/z upwards from B_0 = i (xi has no zeros)
    and P_l = i/(B_l - A_l) (Wronskian psi xi' - psi' xi = i), one order at a time.  A
    product recurrence for P (Yang 2003) is a 0*inf form at a real zero of psi_{l-1},
    whose small and large factors are rounded independently; here no product runs
    through a zero, the cancellation of l/z - A_{l-1} for l >> |z| never occurs, and for
    real z the small real part psi_l^2 = Im B_l/|B_l - A_l|^2 is a ratio of positive
    factors.  In strongly amplifying media (Im z <= -1) the forward recurrence of B is
    unstable, but psi has no zero within distance 1 of z: there P_l = P_{l-1} r_l
    (r_l - i/P_{l-1}), r_l = 1/s_l, P_0 = (1 - exp(2iz))/2, and B_l = A_l + i/P_l.

    ``log_jbar``: jbar_l/jbar_{l-1} = (2l+1)/(z s_l) with the very s_l rounded in the
    recurrence for A, from jbar_0 = sin z/z, or from psi_0 = s_1 psi_1 when |s_1| < 1
    (z near a zero of sin z).  At a real zero of psi_l the large ratio of order l + 1
    then divides by the same rounded s_{l+1} that made jbar_l small, as P_l ~ i s_{l+1}
    is, so jbar stays accurate and P/jbar exact.
    """
    z = complex(z)
    if z == 0:
        raise ValueError("argument 0")
    top = L + _A_HEADROOM + int(abs(z))
    A = np.empty(top + 1, complex)
    s = np.empty(top + 1, complex)  # s_l = A_l + l/z, exactly as rounded in the recurrence
    A[top] = 0.0
    for l in range(top, 0, -1):
        s[l] = A[l] + l / z
        A[l - 1] = l / z - 1 / s[l]
    A, s = A[: L + 1], s[: L + 1]
    B = np.empty(L + 1, complex)
    B[0] = 1j
    if z.imag > -_GAIN_SWITCH:
        for l in range(1, L + 1):
            B[l] = 1 / (l / z - B[l - 1]) - l / z
        with np.errstate(divide="ignore", invalid="ignore"):
            P = 1j / (B - A)
    else:
        P = np.empty(L + 1, complex)
        P[0] = (1 - np.exp(2j * z)) / 2
        for l in range(1, L + 1):
            r = 1 / s[l]  # psi_l/psi_{l-1}
            P[l] = P[l - 1] * r * (r - 1j / P[l - 1])  # l/z - B_{l-1} = r - i/P_{l-1}
            B[l] = A[l] + 1j / P[l]
    l = np.arange(1, L + 1)
    m = abs(z.imag)
    with np.errstate(divide="ignore", invalid="ignore"):
        if L >= 1 and m < 1 and abs(s[1]) < 1:  # near a zero of sin z: psi_0 = s_1 psi_1
            log_j0 = np.log(s[1] * (np.sin(z) / z - np.cos(z)) / z)
        else:
            log_j0 = m + np.log((np.exp(1j * z - m) - np.exp(-1j * z - m)) / 2j) - np.log(z)
        log_jbar = np.r_[log_j0, log_j0 + np.cumsum(np.log((2 * l + 1) / (z * s[1:])))]
    return A, B, P, log_jbar


def _auxiliary_real(x, L):
    """:func:`auxiliary` at many real positive arguments ``x`` (P,) at once: arrays (P, L+1).

    The same recurrences, operation for operation (l/z is l/x for a real z, also in
    Python's complex division), so every element equals the scalar result bit for bit;
    each argument keeps its own starting order of the downward recurrence.
    """
    x = np.asarray(x, dtype=float).ravel()
    if np.any(~(x > 0)):
        raise ValueError("arguments must be positive")
    z = x + 0j
    tops = L + _A_HEADROOM + x.astype(int)  # int(abs(z)) of the scalar version
    top = int(tops.max())
    A = np.zeros((x.size, top + 1), complex)
    s = np.zeros((x.size, top + 1), complex)
    for l in range(top, 0, -1):
        live = l <= tops
        lz = l / x
        sl = A[:, l] + lz
        with np.errstate(divide="ignore", invalid="ignore"):
            nxt = lz - 1 / sl
        s[:, l] = np.where(live, sl, 0)
        A[:, l - 1] = np.where(live, nxt, 0)
    A, s = A[:, : L + 1], s[:, : L + 1]
    B = np.empty((x.size, L + 1), complex)
    B[:, 0] = 1j
    for l in range(1, L + 1):
        lz = l / x
        B[:, l] = 1 / (lz - B[:, l - 1]) - lz
    with np.errstate(divide="ignore", invalid="ignore"):
        P = 1j / (B - A)
    l = np.arange(1, L + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        near = (np.abs(s[:, 1]) < 1) if L >= 1 else np.zeros(x.size, bool)
        log_j0 = np.where(
            near,
            np.log(s[:, 1] * (np.sin(z) / z - np.cos(z)) / z),
            0.0 + np.log((np.exp(1j * z - 0.0) - np.exp(-1j * z - 0.0)) / 2j) - np.log(z),
        )
        steps = np.log((2 * l + 1) / (z[:, None] * s[:, 1:]))
        log_jbar = np.concatenate([log_j0[:, None], log_j0[:, None] + np.cumsum(steps, axis=1)], axis=1)
    return A, B, P, log_jbar


def _args(k, radii, outer_side):
    """Arguments k_n r_n (inner side) or k_{n+1} r_n (outer side) of every interface."""
    return [(k[j + 1] if outer_side else k[j]) * radii[j] for j in range(radii.size)]


def _propagator(a, b, r_a, r_b, l):
    """[xi/psi](b)/[xi/psi](a) for arguments of one shell at radii r_a < r_b.

    ``a`` and ``b`` are :func:`auxiliary` tuples; either may hold many arguments (rows),
    with ``r_a`` or ``r_b`` then an array of matching length.
    """
    la, lb = a[3][..., l], b[3][..., l]
    log_ratio = np.log(np.asarray(r_a) / np.asarray(r_b))
    if np.ndim(log_ratio):
        log_ratio = log_ratio[:, None]
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        return np.exp((2 * l + 2) * log_ratio + 2 * (la - lb)) * b[2][..., l] / a[2][..., l]


class _Sweep:
    """The emitter-independent part of the formulation at one wavelength and truncation L.

    Holds the auxiliary functions on both sides of every interface and, for TM and TE,
    the outward ratios rho just outside each interface (``rho_t``) and the inward ratios
    sigma on its inner (``sig_in``) and outer (``sig_t``) side.  :meth:`at` evaluates the
    per-order quantities at any number of emitter radii.
    """

    def __init__(self, radii, n, mu, k, L):
        self.radii, self.n, self.mu, self.k, self.L = radii, n, mu, k, L
        N = radii.size
        l = np.arange(1, L + 1)
        self.l = l
        self.inner = [auxiliary(k[j] * radii[j], L + 1) for j in range(N)]  # x_n = k_n r_n
        self.outer = [auxiliary(k[j + 1] * radii[j], L + 1) for j in range(N)]  # x~_n = k_{n+1} r_n
        inner, outer = self.inner, self.outer
        eta, mr = n[:-1] / n[1:], mu[:-1] / mu[1:]
        n_in, n_out, mu_in, mu_out = n[:-1], n[1:], mu[:-1], mu[1:]
        # g - 1 (g = f x~/x) from the material contrast: exact for similar media and exactly 0 for the
        # TE terms of nonmagnetic interfaces, where the leading (l+1)/x parts of the mismatch cancel
        g_minus_1 = {
            TM: (mu_in * (n_out - n_in) * (n_out + n_in) + n_in**2 * (mu_in - mu_out)) / (mu_out * n_in**2),
            TE: (mu_out - mu_in) / mu_in,
        }
        # the small ratios r = psi_{l+1}/psi_l = 1/(A_{l+1} + (l+1)/z) and X = xi_{l-1}/xi_l = 1/(l/z - B_{l-1})
        small = []
        for side in (inner, outer):
            small.append(
                [
                    (1 / (a[0][l + 1] + (l + 1) / z), 1 / (l / z - a[1][l - 1]))
                    for a, z in zip(side, _args(k, radii, side is outer))
                ]
            )
        self.pol = {}
        for p in (TM, TE):
            c_v, c_d = (mr, eta) if p == TM else (eta, mr)
            f = c_v / c_d
            # interface mismatches without cancellation: m11 = f A - A~, m33 = f B - B~
            m11, m33 = [], []
            for j in range(N):
                (r_in, X_in), (r_out, X_out) = small[0][j], small[1][j]
                xt = k[j + 1] * radii[j]
                m11.append((l + 1) * g_minus_1[p][j] / xt - (f[j] * r_in - r_out))
                m33.append((f[j] * X_in - X_out) - l * g_minus_1[p][j] / xt)
            # outward sweep: rho just outside each interface (rho_t)
            rho_t = np.zeros((N, L), complex)
            rho = np.zeros(L, complex)
            with np.errstate(all="ignore"):
                for j in range(N):
                    if j:
                        rho = rho_t[j - 1] * _propagator(outer[j - 1], inner[j], radii[j - 1], radii[j], l)
                    A, B = inner[j][0][l], inner[j][1][l]
                    At, Bt = outer[j][0][l], outer[j][1][l]
                    rho_t[j] = (m11[j] + rho * (f[j] * B - At)) / ((Bt - f[j] * A) - rho * m33[j])
                # inward sweep: sigma just outside interface j (sig_t) and on its inner side (sig_in)
                sig_in, sig_t = np.zeros((N, L), complex), np.zeros((N, L), complex)
                sig = np.zeros(L, complex)
                for j in range(N - 1, -1, -1):
                    if j < N - 1:
                        sig = sig_in[j + 1] * _propagator(outer[j], inner[j + 1], radii[j], radii[j + 1], l)
                    sig_t[j] = sig
                    A, B = inner[j][0][l], inner[j][1][l]
                    At, Bt = outer[j][0][l], outer[j][1][l]
                    sig_in[j] = (-m33[j] + sig * (At - f[j] * B)) / ((f[j] * A - Bt) + sig * m11[j])
            self.pol[p] = dict(c_v=c_v, c_d=c_d, rho_t=rho_t, sig_in=sig_in, sig_t=sig_t)

    def transmission(self, p, d, stop):
        """The outgoing solution carried outwards from shell ``d`` to shell ``stop`` > ``d``, polarization ``p``:
        C_stop/C_d = (k_stop/k_d)^(l+1) exp(log) chain, with ``log`` the sum of log jbar(x~_j) - log jbar(x_j)
        over interfaces j = d..stop-1 and ``chain`` the product of P(x_j)/P(x~_j) and the better conditioned
        of the value and derivative continuity ratios; (L,) each."""
        q, l = self.pol[p], self.l
        inner, outer = self.inner, self.outer
        log, chain = np.zeros(self.L, complex), np.ones(self.L, complex)
        with np.errstate(all="ignore"):
            for j in range(d, stop):
                sv, st = q["sig_in"][j], q["sig_t"][j]
                A, B = inner[j][0][l], inner[j][1][l]
                At, Bt = outer[j][0][l], outer[j][1][l]
                value = (1 + sv) / (q["c_v"][j] * (1 + st))
                deriv = (sv * A + B) / (q["c_d"][j] * (st * At + Bt))
                cond_v = np.minimum(np.abs(1 + sv) / (1 + np.abs(sv)), np.abs(1 + st) / (1 + np.abs(st)))
                cond_d = np.minimum(
                    np.abs(sv * A + B) / (np.abs(sv * A) + np.abs(B)),
                    np.abs(st * At + Bt) / (np.abs(st * At) + np.abs(Bt)),
                )
                log = log + outer[j][3][l] - inner[j][3][l]
                chain = chain * np.where(cond_v >= cond_d, value, deriv) * inner[j][2][l] / outer[j][2][l]
        return log, chain

    def at(self, r):
        """Per-order quantities at emitter radii ``r`` (P,), each lossless: dict of (2, P, L) arrays
        indexed [TM, TE] (``P``, ``A``, ``B``, ``rho``, ``sigma``, ``S``, ``Sm``, ``Sd``, ``F``, ``Fd``),
        plus ``x`` (P,), ``shell`` (P,) and ``f_rad`` (P,)."""
        radii, n, mu, k, L, l = self.radii, self.n, self.mu, self.k, self.L, self.l
        N = radii.size
        r = np.asarray(r, dtype=float).ravel()
        shells = np.searchsorted(radii, r, side="right")
        names = ("P", "A", "B", "rho", "sigma", "S", "Sm", "Sd", "F", "Fd", "rp", "rx", "Fr", "lj")
        out = {name: np.zeros((2, r.size, L), complex) for name in names}
        x_all = np.zeros(r.size, dtype=complex if np.any(k[np.unique(shells)].imag != 0) else float)
        inner, outer = self.inner, self.outer
        for d in np.unique(shells):
            idx = np.flatnonzero(shells == d)
            if n[d].imag != 0 or mu[d].imag != 0 or n[-1].imag != 0:
                raise ValueError("the emitter's shell and the host must be lossless")
            rd = r[idx]
            if k[d].imag == 0:
                x = k[d].real * rd
                emit = _auxiliary_real(x, L + 1)
            else:  # imaginary frequency (Casimir-Polder): z = i kappa r, one argument at a time
                x = k[d] * rd
                emit = tuple(np.array(v) for v in zip(*(auxiliary(z, L + 1) for z in x)))
            x_all[idx] = x
            for p in (TM, TE):
                q = self.pol[p]
                rho_t, sig_in = q["rho_t"], q["sig_in"]
                with np.errstate(all="ignore"):
                    if d:
                        rho_e = rho_t[d - 1] * _propagator(outer[d - 1], emit, radii[d - 1], rd, l)
                    else:
                        rho_e = np.zeros((idx.size, L), complex)
                    if d < N:
                        sig_e = sig_in[d] * _propagator(emit, inner[d], rd, radii[d], l)
                    else:
                        sig_e = np.zeros((idx.size, L), complex)
                    a, b, P = emit[0][:, l], emit[1][:, l], emit[2][:, l]
                    # psi_{l+1}/psi_l = 1/s_{l+1} as rounded in the recurrence, and xi_{l+1}/xi_l: A = (l+1)/x - rp,
                    # B = (l+1)/x - rx, so functionals whose leading small-x parts cancel are formed without loss
                    rp = 1 / (emit[0][:, l + 1] + (l + 1) / x[:, None])
                    rx = (l + 1) / x[:, None] - b
                    delta = 1 - rho_e * sig_e
                    S = (rho_e + sig_e + 2 * rho_e * sig_e) / delta
                    Sm = (rho_e * b + sig_e * a + rho_e * sig_e * (a + b)) / delta
                    Sd = (rho_e * b**2 + sig_e * a**2 + 2 * rho_e * sig_e * a * b) / delta
                    # radiated amplitude: (k_h r)^(l+1)/(2l+1)!! jbar(x) (1+rho)/(1-rho sigma) prod_n C_n
                    # (none at imaginary frequency, where k_h is imaginary: nan)
                    log_amp = (l + 1) * np.log(k[-1].real * rd + 0j)[:, None] - log_double_factorial(l) + emit[3][:, l]
                    if k[-1].imag != 0:
                        log_amp = np.full_like(log_amp, np.nan)
                    log_c, chain = self.transmission(p, d, N)
                    core = np.exp(log_amp + log_c) * chain / delta
                for name, value in (
                    ("P", P), ("A", a), ("B", b), ("rho", rho_e), ("sigma", sig_e), ("S", S), ("Sm", Sm), ("Sd", Sd),
                    ("F", core * (1 + rho_e)), ("Fd", core * (a + rho_e * b)), ("rp", rp), ("rx", rx),
                    ("Fr", core * (rp + rho_e * rx)), ("lj", emit[3][:, l]),
                ):  # fmt: skip
                    out[name][p, idx] = value
        out["x"] = x_all
        out["shell"] = shells
        out["f_rad"] = (n[shells] * mu[shells] / (n[-1] * mu[-1])).real
        return out


def _dipole_series(t, dipole):
    """Complex scattered terms (P, L, 2) [perp, par] (Re: total, Im: twice the shift) and radiative
    terms (P, L, 2) of a dipole, shell normalization, from the output of :meth:`_Sweep.at`."""
    radial, other = (TM, TE) if dipole == "electric" else (TE, TM)
    l = np.arange(1, t["P"].shape[2] + 1)
    x = t["x"][:, None]
    c1, c2 = l * (l + 1) * (2 * l + 1), 2 * l + 1
    P, S, Sd, F, Fd = t["P"], t["S"], t["Sd"], t["F"], t["Fd"]
    g = np.stack(
        [1.5 / x**4 * c1 * (P[radial] * S[radial]), 0.75 / x**2 * c2 * (P[other] * S[other] + P[radial] * Sd[radial])],
        axis=2,
    )
    f_rad = t["f_rad"][:, None]
    rad = np.stack(
        [
            f_rad * 1.5 / x**4 * c1 * np.abs(F[radial]) ** 2,
            f_rad * 0.75 / x**2 * c2 * (np.abs(F[other]) ** 2 + np.abs(Fd[radial]) ** 2),
        ],
        axis=2,
    )
    return g, rad


#: complex numbers per block of emitter positions evaluated at once (bounds the memory of long sweeps)
_BLOCK = 8_000_000


def _dipole_sums(sweep, r, dipole, tol):
    """Summed complex scattered series (P, 2), radiative sums (P, 2) and convergence (P,) of a dipole
    at radii ``r`` (shell normalization), evaluated in blocks of positions."""
    step = max(1, _BLOCK // (20 * sweep.L))
    g_sum = np.zeros((r.size, 2), complex)
    rad_sum = np.zeros((r.size, 2))
    ok = np.zeros(r.size, bool)
    for start in range(0, r.size, step):
        part = slice(start, start + step)
        g, rad = _dipole_series(sweep.at(r[part]), dipole)
        g_sum[part], rad_sum[part] = g.sum(axis=1), rad.sum(axis=1)
        ok[part] = _complex_converged(g, tol)
    return g_sum, rad_sum, ok


def _starting_order(radii, n, wavelength, r, tol, l_cap):
    """The truncation estimate of :func:`~pystratify.decay_rates`: particle size and proximity."""
    q = np.max(np.minimum(radii[None, :] / r[:, None], r[:, None] / radii[None, :]))
    near = truncation_order(radii[-1], n[-1], wavelength, "near")
    return int(min(l_cap, max(32, near, orders_needed(q, tol, l_cap))))


def _complex_converged(g, tol):
    """Remainder test on the modulus of complex series (P, L, K) against tol |1 + sum|: the
    reactive part is included, so a frequency shift converges together with the rate."""
    tail = tail_estimate(np.moveaxis(np.abs(g[:, -4:]), 1, 0))
    scale = np.abs(1 + g.sum(axis=1))
    return np.all(np.isfinite(scale) & (tail <= tol * scale), axis=1)


def _prepare(radii, n, mu, wavelength, r):
    radii = np.atleast_1d(np.asarray(radii, float))
    n = np.asarray(n, complex)
    mu = np.ones(n.size, complex) if mu is None else np.asarray(mu, complex)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if n.shape != (radii.size + 1,) or mu.shape != n.shape:
        raise ValueError(f"n and mu need shape ({radii.size + 1},), host last")
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("emitter positions must be positive and finite")
    if np.ndim(wavelength) != 0:
        raise ValueError("one wavelength at a time")
    k = 2 * np.pi * n / wavelength
    return radii, n, mu, r, k


def _evaluate(radii, n, mu, wavelength, r, k, l_max, tol, l_cap):
    """Sweep and emitter side at a given or automatic truncation: (terms dict, L, converged (P,))."""
    L = int(l_max) if l_max is not None else _starting_order(radii, n, wavelength, r, tol, l_cap)
    while True:
        t = _Sweep(radii, n, mu, k, L).at(r)
        ok = np.ones(r.size, bool)
        for dipole in ("electric", "magnetic"):
            ok &= _complex_converged(_dipole_series(t, dipole)[0], tol)
        if l_max is not None or ok.all() or L >= l_cap:
            return t, L, ok
        L = min(2 * L, l_cap)


@dataclass(frozen=True)
class NormalizedTerms:
    """Per-order quantities of the normalized formulation at emitter radii ``r`` (P,), orders 1..L.

    Complex arrays of shape (2, P, L), indexed by polarization ``[TM, TE]`` (``pystratify.TM``,
    ``pystratify.TE``): ``P`` = psi xi, ``A`` = psi'/psi and ``B`` = xi'/xi at x = k_d r; the local
    reflection ratios ``rho`` and ``sigma``; the scattered Green's forms ``S``, ``Sm`` and ``Sd``
    (value-value, value-derivative, derivative-derivative; multiply by ``P``); the radiated
    amplitudes ``F`` (value) and ``Fd`` (derivative), which include the explicit factor
    (k_h r)^(l+1)/(2l+1)!! and so may underflow to 0.  ``x`` (P,) is k_d r, ``shell`` (P,) the
    emitter's shell and ``f_rad`` (P,) = n_d mu_d/(n_h mu_h) the radiative factor of a dipole.
    ``r_psi`` = psi_{l+1}/psi_l and ``r_xi`` = xi_{l+1}/xi_l give A = (l+1)/x - r_psi and
    B = (l+1)/x - r_xi, and ``Fr`` = core (r_psi + rho r_xi) the matching radiated amplitude
    (F' = (l+1) F/x - Fr): functionals whose leading small-x parts cancel, such as A - 2/x
    of a quadrupole at l = 1, are then formed without cancellation.
    ``converged`` (P,) tells whether the electric and magnetic dipole series, rate and shift, met
    ``tol``.  Shell normalization throughout.
    """

    r: np.ndarray
    shell: np.ndarray
    x: np.ndarray
    f_rad: np.ndarray
    orders: int
    converged: np.ndarray
    P: np.ndarray
    A: np.ndarray
    B: np.ndarray
    rho: np.ndarray
    sigma: np.ndarray
    S: np.ndarray
    Sm: np.ndarray
    Sd: np.ndarray
    F: np.ndarray
    Fd: np.ndarray
    r_psi: np.ndarray
    r_xi: np.ndarray
    Fr: np.ndarray

    def dipole_series(self, dipole="electric"):
        """Per-order complex scattered terms (P, L, 2) [perp, par] of a dipole: Re gives the total
        rate (1 + sum), Im twice the frequency shift; and the radiative terms (P, L, 2)."""
        if dipole not in ("electric", "magnetic"):
            raise ValueError("dipole must be 'electric' or 'magnetic'")
        t = {name: getattr(self, name) for name in ("P", "S", "Sd", "F", "Fd", "x", "f_rad")}
        return _dipole_series(t, dipole)


def normalized_terms(radii, n, wavelength, r, l_max=None, mu=None, tol=1e-13, l_cap=L_CAP) -> NormalizedTerms:
    """Per-order quantities of the normalized formulation at emitter radius or radii ``r``.

    Geometry and materials as in :func:`pystratify.decay_rates` (one wavelength, lengths in
    one unit); every emitter in a lossless shell, lossless host.  ``l_max`` is the number of
    multipoles; ``None`` starts from the estimate of :func:`~pystratify.decay_rates` and
    doubles (up to ``l_cap``) until the complex dipole series - rate and shift - meet ``tol``.
    """
    radii, n, mu, r, k = _prepare(radii, n, mu, wavelength, r)
    t, L, ok = _evaluate(radii, n, mu, wavelength, r, k, l_max, tol, l_cap)
    return NormalizedTerms(
        r=r,
        shell=t["shell"],
        x=t["x"],
        f_rad=t["f_rad"],
        orders=L,
        converged=ok,
        **{name: t[name] for name in ("P", "A", "B", "rho", "sigma", "S", "Sm", "Sd", "F", "Fd")},
        r_psi=t["rp"],
        r_xi=t["rx"],
        Fr=t["Fr"],
    )


@dataclass(frozen=True)
class NormalizedRates:
    """Shell-normalized decay rates at one emitter position; arrays ``[perp, par]``.

    ``nonradiative = total - radiative``; ``shift`` is the frequency shift (omega - omega_0)/Gamma_0
    in units of the same free rate (exp(-i omega t); the free self-energy is part of omega_0), so
    an electric dipole near a metal has ``shift < 0``.  With ``terms=True`` the per-order
    contributions (length ``l_max``) are kept in ``terms_total`` (scattered part, without the
    free-space 1), ``terms_radiative`` and ``terms_shift``, each of shape ``(l_max, 2)``.
    """

    r: float
    orders: int
    total: np.ndarray
    radiative: np.ndarray
    terms_total: np.ndarray | None = None
    terms_radiative: np.ndarray | None = None
    shift: np.ndarray | None = None
    terms_shift: np.ndarray | None = None
    converged: bool = True
    route: str = "normalized"

    @property
    def nonradiative(self) -> np.ndarray:
        return self.total - self.radiative


def normalized_decay_rates(
    radii, n, wavelength, r, l_max=None, mu=None, dipole="electric", terms=False, tol=1e-13, l_cap=L_CAP
):
    """Decay rates and frequency shift of an electric or magnetic dipole at radius ``r``.

    Parameters as in :func:`pystratify.decay_rates` (one wavelength, lengths in one unit);
    ``l_max`` is the number of multipoles summed, ``None`` for the automatic truncation of
    :func:`normalized_terms` (``tol``, ``l_cap``).  Returns :class:`NormalizedRates` in the
    shell normalization.
    """
    if dipole not in ("electric", "magnetic"):
        raise ValueError("dipole must be 'electric' or 'magnetic'")
    if np.ndim(r) != 0:
        raise ValueError("one emitter radius; use normalized_terms for many")
    radii, n, mu, rr, k = _prepare(radii, n, mu, wavelength, r)
    t, L, ok = _evaluate(radii, n, mu, wavelength, rr, k, l_max, tol, l_cap)
    g, rad = _dipole_series(t, dipole)
    t_tot, t_rad = np.real(g[0]), rad[0]
    t_shift = 0.5 * np.imag(g[0])
    return NormalizedRates(
        r=float(r),
        orders=L,
        total=1 + t_tot.sum(axis=0),
        radiative=t_rad.sum(axis=0),
        terms_total=t_tot if terms else None,
        terms_radiative=t_rad if terms else None,
        shift=t_shift.sum(axis=0),
        terms_shift=t_shift if terms else None,
        converged=bool(ok[0]),
    )
