"""Multipole truncation order for a sphere of given size."""

from __future__ import annotations

import numpy as np

__all__ = ["truncation_order", "tail_estimate", "orders_needed"]


def truncation_order(radius, n_host, wavelength, regime="far") -> int:
    """Number of multipole orders needed for a sphere of outer radius ``radius``.

    ``regime='far'``: cross sections and far field, Wiscombe, Appl. Opt. 19,
    1505 (1980).  ``regime='near'``: fields at the surface, Allardice & Le Ru,
    Appl. Opt. 53, 7224 (2014).  ``wavelength`` may be an array (the shortest
    is used) and ``n_host`` complex (its modulus is used).
    """
    x = 2 * np.pi * float(radius) * abs(complex(n_host)) / float(np.min(wavelength))
    if not np.isfinite(x) or x <= 0:
        raise ValueError("radius, n_host and wavelength must give a positive, finite size parameter")
    if regime == "near":
        order = x + 11 * x ** (1 / 3) + 1
    elif regime == "far":
        order = x + 4 * x ** (1 / 3) + (1 if x <= 8 else 2)
    else:
        raise ValueError("regime must be 'far' or 'near'")
    return max(1, int(order))


def tail_estimate(terms):
    """Remainder of a series estimated from its last terms (axis 0 = order).

    The decay-rate series are asymptotically geometric in l, ratio
    (r_< / r_>)^2 times a polynomial, so the neglected tail is ~ t_L q/(1 - q)
    with q the worst recent ratio: far larger than t_L when q -> 1.  +inf
    where the terms are not decreasing.
    """
    a = np.abs(terms[-4:])
    with np.errstate(all="ignore"):
        q = np.max(a[1:] / a[:-1], axis=0)
        return np.where(a[-1] == 0, 0.0, np.where(q < 1, a[-1] * q / (1 - q), np.inf))


def orders_needed(q, tol, l_cap):
    """Orders for a geometric series of ratio q^2 to leave a tail below tol."""
    if q >= 1:
        return l_cap
    return 1.25 * np.log(tol * (1 - q * q)) / (2 * np.log(q)) + 16
