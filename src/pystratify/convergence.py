"""Multipole truncation order for a sphere of given size."""

from __future__ import annotations

import numpy as np

__all__ = ["truncation_order"]


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
