"""Decay rates from normalized quantities only: the formulation of the accompanying paper.

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
derivatives is ever formed.  The
total rate is 1 + sum Re[P (rho + sigma + 2 rho sigma)/(1 - rho sigma)] (shell
normalization) and the radiated amplitude carries the single explicit factor
(k_h r)^(l+1)/(2l+1)!!.  Nothing is computed that can overflow; only the
geometric factors can underflow, which is harmless.

This module is independent of :mod:`pystratify.solver` and
:mod:`pystratify.decay` (which carry the same physics as logarithms) and
serves as the reference implementation of the paper's equations; the tests
check the two against each other and against extended-precision transfer
matrices.  Shells 1..N+1 of the paper are indices 0..N here; the emitter must
be in a lossless shell and the host must be lossless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from .solver import TE, TM

__all__ = ["auxiliary", "log_double_factorial", "NormalizedRates", "normalized_decay_rates"]

#: extra orders above max(L, |z|) at which the downward recurrence for A_l starts
_A_HEADROOM = 20
#: for Im z <= -_GAIN_SWITCH (strongly amplifying media) B_l is not recurred upwards (see auxiliary)
_GAIN_SWITCH = 1.0


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


@dataclass(frozen=True)
class NormalizedRates:
    """Shell-normalized decay rates at one emitter position; arrays ``[perp, par]``.

    ``nonradiative = total - radiative``.  With ``terms=True`` the per-order
    contributions (length ``l_max``) are kept in ``terms_total`` (scattered
    part, without the free-space 1) and ``terms_radiative``, each of shape
    ``(l_max, 2)``.
    """

    r: float
    orders: int
    total: np.ndarray
    radiative: np.ndarray
    terms_total: np.ndarray | None = None
    terms_radiative: np.ndarray | None = None

    @property
    def nonradiative(self) -> np.ndarray:
        return self.total - self.radiative


def _args(k, radii, outer_side):
    """Arguments k_n r_n (inner side) or k_{n+1} r_n (outer side) of every interface."""
    return [(k[j + 1] if outer_side else k[j]) * radii[j] for j in range(radii.size)]


def _propagator(a, b, r_a, r_b, l):
    """[xi/psi](b)/[xi/psi](a) for arguments of one shell at radii r_a < r_b."""
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        return np.exp((2 * l + 2) * np.log(r_a / r_b) + 2 * (a[3][l] - b[3][l])) * b[2][l] / a[2][l]


def normalized_decay_rates(radii, n, wavelength, r, l_max, mu=None, dipole="electric", terms=False):
    """Decay rates of an electric or magnetic dipole at radius ``r``, normalized formulation.

    Parameters as in :func:`pystratify.decay_rates` (one wavelength, lengths in
    one unit); ``l_max`` is the number of multipoles summed.  Returns
    :class:`NormalizedRates` in the shell normalization.
    """
    radii = np.atleast_1d(np.asarray(radii, float))
    n = np.asarray(n, complex)
    mu = np.ones(n.size, complex) if mu is None else np.asarray(mu, complex)
    if dipole not in ("electric", "magnetic"):
        raise ValueError("dipole must be 'electric' or 'magnetic'")
    N = radii.size
    L = int(l_max)
    k = 2 * np.pi * n / wavelength
    l = np.arange(1, L + 1)
    inner = [auxiliary(k[j] * radii[j], L + 1) for j in range(N)]  # x_n = k_n r_n
    outer = [auxiliary(k[j + 1] * radii[j], L + 1) for j in range(N)]  # x~_n = k_{n+1} r_n
    d = int(np.searchsorted(radii, r, side="right"))
    if n[d].imag != 0 or mu[d].imag != 0 or n[-1].imag != 0:
        raise ValueError("the emitter's shell and the host must be lossless")
    x = k[d].real * r
    emit = auxiliary(x + 0j, L + 1)
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
    res = {}
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
            rho_e = rho_t[d - 1] * _propagator(outer[d - 1], emit, radii[d - 1], r, l) if d else np.zeros(L, complex)
            sig_e = sig_in[d] * _propagator(emit, inner[d], r, radii[d], l) if d < N else np.zeros(L, complex)
            a, b, P = emit[0][l], emit[1][l], emit[2][l]
            delta = 1 - rho_e * sig_e
            S = (rho_e + sig_e + 2 * rho_e * sig_e) / delta
            Sd = (rho_e * b**2 + sig_e * a**2 + 2 * rho_e * sig_e * a * b) / delta
            # radiated amplitude: (k_h r)^(l+1)/(2l+1)!! jbar(x) (1+rho)/(1-rho sigma) prod_n C_n
            log_amp = (l + 1) * np.log(k[-1].real * r) - log_double_factorial(l) + emit[3][l]
            chain = np.ones(L, complex)
            for j in range(d, N):
                sv, st = sig_in[j], sig_t[j]
                A, B = inner[j][0][l], inner[j][1][l]
                At, Bt = outer[j][0][l], outer[j][1][l]
                value = (1 + sv) / (c_v[j] * (1 + st))
                deriv = (sv * A + B) / (c_d[j] * (st * At + Bt))
                cond_v = np.minimum(np.abs(1 + sv) / (1 + np.abs(sv)), np.abs(1 + st) / (1 + np.abs(st)))
                cond_d = np.minimum(
                    np.abs(sv * A + B) / (np.abs(sv * A) + np.abs(B)),
                    np.abs(st * At + Bt) / (np.abs(st * At) + np.abs(Bt)),
                )
                log_amp = log_amp + outer[j][3][l] - inner[j][3][l]
                chain = chain * np.where(cond_v >= cond_d, value, deriv) * inner[j][2][l] / outer[j][2][l]
            core = np.exp(log_amp) * chain / delta
        res[p] = dict(S=S, Sd=Sd, P=P, F=core * (1 + rho_e), Fd=core * (a + rho_e * b))
    radial, other = (TM, TE) if dipole == "electric" else (TE, TM)
    R, O = res[radial], res[other]
    c1, c2 = l * (l + 1) * (2 * l + 1), 2 * l + 1
    t_tot = np.stack(
        [1.5 / x**4 * c1 * np.real(R["P"] * R["S"]), 0.75 / x**2 * c2 * np.real(O["P"] * O["S"] + R["P"] * R["Sd"])], 1
    )
    f_rad = (n[d] * mu[d] / (n[-1] * mu[-1])).real
    t_rad = np.stack(
        [
            f_rad * 1.5 / x**4 * c1 * np.abs(R["F"]) ** 2,
            f_rad * 0.75 / x**2 * c2 * (np.abs(O["F"]) ** 2 + np.abs(R["Fd"]) ** 2),
        ],
        1,
    )
    return NormalizedRates(
        r=float(r),
        orders=L,
        total=1 + t_tot.sum(axis=0),
        radiative=t_rad.sum(axis=0),
        terms_total=t_tot if terms else None,
        terms_radiative=t_rad if terms else None,
    )
