"""Decay rates from normalized quantities only: the formulation of the accompanying paper.

Every per-order term is an explicit geometric factor times quantities of order
unity built from three auxiliary functions of a single argument z,

    A_l(z) = psi_l'/psi_l                        (downward recurrence),
    P_l(z) = psi_l xi_l = -i z jbar_l hbar_l/(2l+1)   (upward product recurrence, Yang 2003),
    jbar_l(z) = (2l+1)!! j_l(z) / z^l            (from consecutive ratios, as a logarithm),

with B_l = A_l + i/P_l.  The shell structure enters through the local
reflection ratios rho = R xi/psi (swept outwards) and sigma = S psi/xi (swept
inwards), carried across a shell by the propagator

    G(r_a, r_b) = (r_a/r_b)^(2l+2) [jbar(k r_a)/jbar(k r_b)]^2 P(k r_b)/P(k r_a),

and across an interface by Moebius maps of logarithmic derivatives.  The
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


def log_double_factorial(l):
    """ln (2l+1)!! = ln Gamma(2l+2) - l ln 2 - ln Gamma(l+1)."""
    l = np.asarray(l, dtype=float)
    return gammaln(2 * l + 2) - l * np.log(2) - gammaln(l + 1)


def auxiliary(z, L):
    """A_l, B_l, P_l and ln jbar_l for l = 0..L at the complex argument z (arrays of length L+1).

    ``A``: downward recurrence A_{l-1} = l/z - 1/(A_l + l/z) from A = 0 at
    l = max(L, |z|) + 20 (Lentz 1976, Wiscombe 1980).  ``P``, ``B``: upward
    product recurrence P_l = P_{l-1} (l/z - A_{l-1})(l/z - B_{l-1}),
    P_0 = (1 - exp(2iz))/2, B_0 = i, B_l = A_l + i/P_l.  ``log_jbar``:
    jbar_l/jbar_{l-1} = (2l+1)/(z A_l + l), jbar_0 = sin z / z.
    """
    z = complex(z)
    if z == 0:
        raise ValueError("argument 0")
    top = L + _A_HEADROOM + int(abs(z))
    A = np.empty(top + 1, complex)
    A[top] = 0.0
    for l in range(top, 0, -1):
        A[l - 1] = l / z - 1 / (A[l] + l / z)
    A = A[: L + 1]
    P = np.empty(L + 1, complex)
    B = np.empty(L + 1, complex)
    P[0] = (1 - np.exp(2j * z)) / 2
    B[0] = 1j
    for l in range(1, L + 1):
        P[l] = P[l - 1] * (l / z - A[l - 1]) * (l / z - B[l - 1])
        B[l] = A[l] + 1j / P[l]
    l = np.arange(1, L + 1)
    m = abs(z.imag)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_j0 = m + np.log((np.exp(1j * z - m) - np.exp(-1j * z - m)) / 2j) - np.log(z)
        log_jbar = np.r_[log_j0, log_j0 + np.cumsum(np.log((2 * l + 1) / (z * A[1:] + l)))]
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
    res = {}
    for p in (TM, TE):
        c_v, c_d = (mr, eta) if p == TM else (eta, mr)
        f = c_v / c_d
        # outward sweep: rho at the outer radius of each shell (rho_out) and just outside it (rho_t)
        rho_t = np.zeros((N, L), complex)
        rho = np.zeros(L, complex)
        with np.errstate(all="ignore"):
            for j in range(N):
                if j:
                    rho = rho_t[j - 1] * _propagator(outer[j - 1], inner[j], radii[j - 1], radii[j], l)
                A, B = inner[j][0][l], inner[j][1][l]
                At, Bt = outer[j][0][l], outer[j][1][l]
                D = f[j] * (A + rho * B) / (1 + rho)
                rho_t[j] = (D - At) / (Bt - D)
            # inward sweep: sigma just outside interface j (sig_t) and on its inner side (sig_in)
            sig_in, sig_t = np.zeros((N, L), complex), np.zeros((N, L), complex)
            sig = np.zeros(L, complex)
            for j in range(N - 1, -1, -1):
                if j < N - 1:
                    sig = sig_in[j + 1] * _propagator(outer[j], inner[j + 1], radii[j], radii[j + 1], l)
                sig_t[j] = sig
                A, B = inner[j][0][l], inner[j][1][l]
                At, Bt = outer[j][0][l], outer[j][1][l]
                D = (sig * At + Bt) / (1 + sig) / f[j]
                sig_in[j] = (D - B) / (A - D)
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
