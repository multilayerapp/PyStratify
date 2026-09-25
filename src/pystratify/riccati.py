"""Riccati-Bessel functions in logarithmic form, for any order and complex argument.

psi_n(z) = z j_n(z) and xi_n(z) = z h_n^(1)(z) over- and underflow as soon as
the order exceeds the argument (psi_n ~ z^(n+1)/(2n+1)!!, xi_n ~ (2n-1)!!/z^n)
or the argument has a large imaginary part (both ~ exp(|Im z|)).  Every
quantity PyStratify needs is a product or ratio of these functions that stays
O(1), so they are carried as complex logarithms and combined before
exponentiation (the prefactor extraction of Majic & Le Ru, Appl. Opt. 59,
1293 (2020), in its general form).

Evaluation, per argument:

* orders up to ``|z| + ANCHOR_MARGIN`` from the exponentially scaled AMOS
  routines: exact near the zeros of psi_n, uniform in |Im z|.  In the lower
  half plane (gain media) ``hankel1e`` returns spurious zeros and ``yve``
  is mis-scaled, so h^(1) - there the growing solution, ~exp(|Im z|) - is
  taken unscaled from ``hankel1`` (finite up to |Im z| ~ 700);
* higher orders, past the turning point where neither function has zeros, by
  the stable recurrences: downward ratio recurrence for the minimal solution
  psi (Wiscombe, Appl. Opt. 19, 1505 (1980); Lentz, Appl. Opt. 15, 668
  (1976)) and upward recurrence for the dominant solution xi.  This also
  keeps the number of AMOS calls independent of the truncation order.

Derivatives are never formed from logarithmic derivatives alone:
psi_n' = psi_{n-1} - n psi_n / z, so zeros of psi_n cause no division by zero.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["log_riccati", "log_derivatives", "ANCHOR_MARGIN"]

#: orders beyond |z| + ANCHOR_MARGIN come from the recurrences
ANCHOR_MARGIN = 32
_TINY = 1e-250
_HUGE = 1e250
_DOWNWARD_START = 40


def _prefactor(z):
    return np.log(np.sqrt(np.pi * z / 2))


def _amos(z, n0):
    """log psi_n, log xi_n for n = 0..n0 from AMOS, plus masks of unusable values."""
    nu = np.arange(n0 + 1) + 0.5
    zc = z[:, None]
    lower = zc.imag < 0
    with np.errstate(all="ignore"):
        j = special.jve(nu, zc)
        h = special.hankel1e(nu, zc)
        if lower.any():
            h = np.where(lower, special.hankel1(nu, zc), h)
        pre = _prefactor(zc)
        log_psi = pre + np.log(j) + np.abs(zc.imag)
        log_xi = pre + np.log(h) + np.where(lower, 0, 1j * zc)
    bad_psi = ~np.isfinite(j) | (np.abs(j) < _TINY)
    # psi has genuine zeros only on the real axis and below the turning point; anywhere
    # else a zero is underflow: off the axis jve underflows long before the turning point
    # (at |z| = 3e4, Im z = 1e4 from order ~1.2e4), because |J| grows slower than exp|Im z|
    bad_psi &= (np.arange(n0 + 1) > np.abs(zc) + 1) | (zc.imag != 0)
    bad_xi = ~np.isfinite(h) | (np.abs(h) > _HUGE) | (h == 0)
    return log_psi, log_xi, bad_psi, bad_xi


def _psi_downward(z, first, log_psi, nmax):
    """Replace log psi_n for n >= first (per row) by the downward ratio recurrence.

    r_n = psi_n / psi_{n-1} = 1 / ((2n + 1)/z - r_{n+1}), started at r = 0 far
    above nmax; log psi_n = log psi_{first-1} + sum of log r up to n.
    """
    lowest = int(first.min())
    top = nmax + _DOWNWARD_START + int(np.ceil(np.abs(z).max()))
    inv_z = 1 / z
    r = np.zeros(z.size, dtype=complex)
    ratios = np.ones((nmax + 1, z.size), dtype=complex)
    with np.errstate(all="ignore"):
        for k in range(top, lowest - 1, -1):
            r = 1.0 / ((2 * k + 1) * inv_z - r)
            if k <= nmax:
                ratios[k] = r
        log_r = np.log(ratios.T)
    after = np.arange(nmax + 1)[None, :] >= first[:, None]
    base = np.take_along_axis(log_psi, (first - 1)[:, None], axis=1)
    return np.where(after, base + np.cumsum(np.where(after, log_r, 0), axis=1), log_psi)


def _xi_upward(z, bad, log_xi, nmax):
    """Replace log xi_n where ``bad`` by the upward recurrence
    q_n = xi_n / xi_{n-1} = (2n - 1)/z - 1/q_{n-1}, continued from the last good order."""
    bad[:, :2] = False
    start = int(np.argmax(bad.any(axis=0)))
    inv_z = 1 / z
    with np.errstate(all="ignore"):
        good_q = np.exp(np.diff(log_xi[:, start - 2 :], axis=1))  # ratios where AMOS is usable
        q = good_q[:, 0]
        ratios = np.empty((nmax + 1 - start, z.size), dtype=complex)
        for i, k in enumerate(range(start, nmax + 1)):
            q = (2 * k - 1) * inv_z - 1 / q
            use = bad[:, k]
            if not use.all():
                q = np.where(use, q, good_q[:, i + 1])
            ratios[i] = q
        log_q = np.log(ratios.T)
    for i, k in enumerate(range(start, nmax + 1)):
        use = bad[:, k]
        log_xi[use, k] = log_xi[use, k - 1] + log_q[use, i]
    return log_xi


def _log_riccati_1d(z, nmax):
    n0 = min(nmax, int(np.ceil(np.abs(z).max())) + ANCHOR_MARGIN)
    lp0, lx0, bad_p, bad_x = _amos(z, n0)
    log_psi = np.empty((z.size, nmax + 1), dtype=complex)
    log_xi = np.empty((z.size, nmax + 1), dtype=complex)
    log_psi[:, : n0 + 1], log_xi[:, : n0 + 1] = lp0, lx0

    # psi: recurrence from the first unusable AMOS order, or from n0 + 1
    first = np.where(bad_p.any(axis=1), np.argmax(bad_p, axis=1), n0 + 1)
    first = np.maximum(first, 1)
    rows = np.flatnonzero(first <= nmax)
    if rows.size:
        log_psi[rows] = _psi_downward(z[rows], first[rows], log_psi[rows], nmax)

    # xi: recurrence wherever AMOS failed and for every order above n0
    bad = np.zeros((z.size, nmax + 1), dtype=bool)
    bad[:, : n0 + 1] = bad_x
    bad[:, n0 + 1 :] = True
    rows = np.flatnonzero(bad[:, 2:].any(axis=1))
    if rows.size:
        log_xi[rows] = _xi_upward(z[rows], bad[rows], log_xi[rows], nmax)
    return log_psi, log_xi


def log_riccati(z, nmax):
    """Complex logarithms of psi_n(z) and xi_n(z) for n = 0..nmax.

    ``z`` may have any shape S (complex, nonzero, finite); returns two arrays of
    shape ``S + (nmax + 1,)``.  exp() of the results reproduces psi_n, xi_n
    wherever they are representable; the logs themselves are finite for any
    order (``-inf`` only at an exact zero of psi_n).
    """
    z = np.asarray(z, dtype=complex)
    nmax = int(nmax)
    if nmax < 1:
        raise ValueError("nmax must be >= 1")
    flat = z.ravel()
    if np.any(flat == 0):
        raise ValueError("log_riccati: argument 0 (psi_n(0) = 0, xi_n(0) = inf)")
    if not np.all(np.isfinite(flat)):
        raise ValueError("log_riccati: non-finite argument")
    if flat.size == 0:
        empty = np.empty(z.shape + (nmax + 1,), dtype=complex)
        return empty, empty.copy()
    log_psi, log_xi = _log_riccati_1d(flat, nmax)
    return log_psi.reshape(z.shape + (nmax + 1,)), log_xi.reshape(z.shape + (nmax + 1,))


def log_derivatives(log_psi, log_xi, z, orders):
    """Logarithmic derivatives psi'/psi and xi'/xi at ``orders`` (>= 1).

    ``log_psi``/``log_xi`` from :func:`log_riccati` (last axis = order 0..nmax),
    ``z`` broadcastable to their leading shape.
    """
    z = np.asarray(z, dtype=complex)[..., None]
    with np.errstate(all="ignore"):
        d_psi = np.exp(log_psi[..., orders - 1] - log_psi[..., orders]) - orders / z
        d_xi = np.exp(log_xi[..., orders - 1] - log_xi[..., orders]) - orders / z
    return d_psi, d_xi
