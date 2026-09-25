"""Multipole truncation (STRATIFY util/l_conv.m, OSAC Eqs. 35-36)."""

from __future__ import annotations

import numpy as np

__all__ = ["l_max"]


def l_max(r_out, n_host, lam, regime="far") -> int:
    """Truncation order for a sphere of outer radius ``r_out``.

    ``regime='far'``: Wiscombe, Appl. Opt. 19, 1505 (1980).
    ``regime='near'``: Allardice & Le Ru, Appl. Opt. 53, 7224 (2014).
    ``lam`` may be an array; the shortest wavelength is used.  Returns an int
    (STRATIFY returns the unrounded value and relies on ``1:l_max``, which
    floors; we do the same).
    """
    x = 2 * np.pi * r_out * abs(complex(n_host)) / np.min(lam)
    if regime == "near":
        v = x + 11 * x ** (1 / 3) + 1
    elif regime == "far":
        if x <= 8:
            v = x + 4 * x ** (1 / 3) + 1
        elif x < 4200:
            v = x + 4.05 * x ** (1 / 3) + 2
        else:
            raise ValueError("size parameter too large (x >= 4200)")
    else:
        raise ValueError("regime must be 'far' or 'near'")
    return max(1, int(np.floor(v)))
