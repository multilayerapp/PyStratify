"""Spherical and Riccati-Bessel functions of complex argument.

Everything is built on ``scipy.special.jv`` / ``hankel1`` at half-integer order
(the AMOS library, the same one MATLAB's ``besselj``/``besselh`` wrap), so the
values match STRATIFY's MATLAB implementation to round-off.

All functions broadcast ``l`` against ``z``; the usual call is ``l`` of shape
``(L, 1)`` and ``z`` of shape ``(1, M)`` (or a scalar), giving ``(L, M)``.

Conventions::

    j_l(z)      spherical Bessel of the first kind
    h_l(z)      spherical Hankel of the first kind, h_l = j_l + i y_l
    psi_l(z)  = z j_l(z)          Riccati-Bessel
    xi_l(z)   = z h_l(z)          Riccati-Hankel  (zeta_l in the papers)
    primes are derivatives with respect to the argument.

At ``z == 0``: ``j_0 = 1``, ``j_l = 0`` for ``l >= 1``, and ``h_l`` is infinite.
(The MATLAB ``sbesselj`` returns ``1`` for the *first requested order* at
``z == 0`` regardless of what that order is, so ``sbesselj(1:L, 0)`` gives
``j_1(0) = 1``.  See AUDIT.md, item M7.)
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = [
    "sph_jn",
    "sph_hn",
    "sph_jn_d",
    "sph_hn_d",
    "ric_j",
    "ric_h",
    "ric_j_d",
    "ric_h_d",
]


def _broadcast(l, z):
    l = np.asarray(l, dtype=float)
    z = np.asarray(z, dtype=complex)
    return np.broadcast_arrays(l, z)


def sph_jn(l, z):
    """Spherical Bessel function j_l(z), complex z."""
    l, z = _broadcast(l, z)
    out = np.empty(l.shape, dtype=complex)
    zero = z == 0
    nz = ~zero
    out[nz] = np.sqrt(np.pi / (2 * z[nz])) * special.jv(l[nz] + 0.5, z[nz])
    out[zero] = np.where(l[zero] == 0, 1.0, 0.0)
    return out


def sph_hn(l, z):
    """Spherical Hankel function of the first kind h_l(z), complex z.

    Returns ``inf`` at ``z == 0``.
    """
    l, z = _broadcast(l, z)
    out = np.full(l.shape, np.inf + 0j, dtype=complex)
    nz = z != 0
    out[nz] = np.sqrt(np.pi / (2 * z[nz])) * special.hankel1(l[nz] + 0.5, z[nz])
    return out


def sph_jn_d(l, z):
    """d j_l / dz.  At z == 0 this is 1/3 for l == 1 and 0 otherwise."""
    l, z = _broadcast(l, z)
    out = np.empty(l.shape, dtype=complex)
    nz = z != 0
    ln, zn = l[nz], z[nz]
    out[nz] = -sph_jn(ln + 1, zn) + (ln / zn) * sph_jn(ln, zn)
    out[~nz] = np.where(l[~nz] == 1, 1.0 / 3.0, 0.0)
    return out


def sph_hn_d(l, z):
    """d h_l / dz."""
    l, z = _broadcast(l, z)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -sph_hn(l + 1, z) + (l / z) * sph_hn(l, z)


def ric_j(l, z):
    """Riccati-Bessel psi_l(z) = z j_l(z)."""
    l, z = _broadcast(l, z)
    return z * sph_jn(l, z)


def ric_h(l, z):
    """Riccati-Hankel xi_l(z) = z h_l(z)."""
    l, z = _broadcast(l, z)
    out = np.full(l.shape, np.inf + 0j, dtype=complex)
    nz = z != 0
    out[nz] = np.sqrt(np.pi * z[nz] / 2) * special.hankel1(l[nz] + 0.5, z[nz])
    return out


def ric_j_d(l, z):
    """psi_l'(z) = j_l(z) + z j_l'(z)."""
    l, z = _broadcast(l, z)
    return sph_jn(l, z) + z * sph_jn_d(l, z)


def ric_h_d(l, z):
    """xi_l'(z) = h_l(z) + z h_l'(z)."""
    l, z = _broadcast(l, z)
    with np.errstate(invalid="ignore"):
        return sph_hn(l, z) + z * sph_hn_d(l, z)
