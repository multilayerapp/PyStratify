"""Riccati-Bessel functions in logarithmic form, for any order and complex argument.

psi_n(z) = z j_n(z) and xi_n(z) = z h_n^(1)(z) over- and underflow as soon as
the order exceeds the argument (psi_n ~ z^(n+1)/(2n+1)!!, xi_n ~ (2n-1)!!/z^n)
or the argument has a large imaginary part (both ~ exp(|Im z|)).  Every
quantity PyStratify needs is a *product or ratio* of these functions that
stays O(1), so they are carried as complex logarithms and only combined
before exponentiation.  This is the prefactor-extraction idea of Majic & Le
Ru, Appl. Opt. 59, 1293 (2020), in its most general form.

Evaluation:

* Where representable, directly from the exponentially scaled AMOS routines
  (``scipy.special.jve`` / ``hankel1e``): exact near zeros of psi_n, no
  cancellation, uniform in |Im z|.
* Beyond that (order well past the turning point, where neither function has
  zeros) by the stable recurrences: the *downward* ratio recurrence for the
  minimal solution psi (Miller / Lentz; Wiscombe, Appl. Opt. 19, 1505
  (1980)) and the *upward* recurrence for the dominant solution xi.

Derivatives are never formed from logarithmic derivatives alone:
psi_n' = psi_{n-1} - n psi_n / z, so zeros of psi_n cause no division by zero.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["log_riccati", "log_derivatives"]

_TINY = 1e-280
_HUGE = 1e280


def _log_psi(z, nmax):
    """log psi_n(z), n = 0..nmax; z is 1-D."""
    n = np.arange(nmax + 1)
    zc = z[:, None]
    with np.errstate(all="ignore"):
        v = special.jve(n + 0.5, zc)
        out = np.log(np.sqrt(np.pi * zc / 2)) + np.log(v) + np.abs(zc.imag)
    bad = ~np.isfinite(v) | (np.abs(v) < _TINY)
    # underflow only beyond the turning point; zeros of psi below it are real zeros
    bad &= n[None, :] > np.abs(zc) + 1
    rows = np.flatnonzero(bad.any(1))
    if rows.size:
        zr = z[rows][:, None]
        start = np.argmax(bad[rows], axis=1)  # first underflowing order per row
        # downward ratio recurrence r_n = psi_n / psi_{n-1} = 1 / ((2n+1)/z - r_{n+1})
        top = nmax + 40 + int(np.max(np.abs(zr)))
        r = np.zeros(zr.shape, dtype=complex)
        ratios = np.empty((rows.size, nmax + 1), dtype=complex)
        with np.errstate(all="ignore"):
            for k in range(top, 0, -1):
                r = 1.0 / ((2 * k + 1) / zr - r)
                if k <= nmax:
                    ratios[:, k] = r[:, 0]
            logr = np.log(ratios)
        s = np.maximum(start, 1)[:, None]
        after = n[None, :] >= s
        cs = np.cumsum(np.where(after, logr, 0), axis=1)
        base = np.take_along_axis(out[rows], s - 1, axis=1)
        out[rows] = np.where(after, base + cs, out[rows])
    return out


def _log_xi(z, nmax):
    """log xi_n(z), n = 0..nmax; z is 1-D."""
    n = np.arange(nmax + 1)
    zc = z[:, None]
    with np.errstate(all="ignore"):
        v = special.hankel1e(n + 0.5, zc)
        out = np.log(np.sqrt(np.pi * zc / 2)) + np.log(v) + 1j * zc
    bad = ~np.isfinite(v) | (np.abs(v) > _HUGE)
    rows = np.flatnonzero(bad.any(1))
    if rows.size:
        # upward recurrence q_n = xi_n / xi_{n-1} = (2n-1)/z - 1/q_{n-1}, stable for
        # the dominant solution; run for all affected rows, used only where needed
        sub, bd, zr = out[rows], bad[rows], z[rows]
        bd[:, :2] = False
        with np.errstate(all="ignore"):
            q = np.exp(sub[:, 1] - sub[:, 0])
            for k in range(2, nmax + 1):
                q_rec = (2 * k - 1) / zr - 1 / q
                use = bd[:, k]
                sub[:, k] = np.where(use, sub[:, k - 1] + np.log(q_rec), sub[:, k])
                q = np.where(use, q_rec, np.exp(sub[:, k] - sub[:, k - 1]))
        out[rows] = sub
    return out


def log_riccati(z, nmax):
    """Complex logarithms of psi_n(z) and xi_n(z) for n = 0..nmax.

    ``z`` may have any shape S (complex, nonzero); returns two arrays of shape
    ``S + (nmax + 1,)``.  exp() of the results reproduces psi_n, xi_n wherever
    they are representable; the logs themselves are finite for any order
    (``-inf`` only at an exact zero of psi_n).
    """
    z = np.asarray(z, dtype=complex)
    shape = z.shape
    zf = z.ravel()
    if np.any(zf == 0):
        raise ValueError("log_riccati: argument 0 (psi_n(0) = 0, xi_n(0) = inf)")
    lp = _log_psi(zf, nmax).reshape(shape + (nmax + 1,))
    lx = _log_xi(zf, nmax).reshape(shape + (nmax + 1,))
    return lp, lx


def log_derivatives(lp, lx, z, l):
    """Logarithmic derivatives D1 = psi'/psi and D3 = xi'/xi at orders ``l``.

    ``lp``/``lx`` from :func:`log_riccati` (last axis = order 0..nmax), ``z``
    broadcastable to their leading shape, ``l`` an array of orders >= 1.
    """
    z = np.asarray(z, dtype=complex)[..., None]
    with np.errstate(all="ignore"):
        d1 = np.exp(lp[..., l - 1] - lp[..., l]) - l / z
        d3 = np.exp(lx[..., l - 1] - lx[..., l]) - l / z
    return d1, d3
