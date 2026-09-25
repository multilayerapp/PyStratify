"""Recursive transfer matrices for a multilayered sphere.

Port of STRATIFY ``util/t_mat.m``; equation numbers refer to
Rasskazov, Carney & Moroz, OSA Continuum 3, 2290 (2020) ("OSAC").

Geometry (OSAC Fig. 1): shells ``n = 1..N`` with outer radii ``rad[n-1]``;
the core is ``n = 1`` and the host is ``n = N + 1``.  ``ref`` and ``mu`` have
``N + 1`` entries, the last one being the host.

Storage follows the MATLAB code so the port can be read side by side with it.
With 0-based python index ``j`` (``0 <= j < N``)::

    te[:, j] = T+(j+1) T+(j) ... T+(1)      == OSAC T_E(j + 2)   (Eq. 14)
    me[:, j] = T-(j+1) T-(j+2) ... T-(N)    == OSAC M_E(j + 1)   (Eq. 14)

and likewise ``tm`` / ``mm`` for the magnetic (TE) polarisation.  Use the
:meth:`TransferMatrices.T` and :meth:`TransferMatrices.M` accessors to index
with the paper's 1-based shell numbers instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bessel import ric_h, ric_h_d, ric_j, ric_j_d

__all__ = ["TransferMatrices", "t_mat", "interface_matrices"]


def _as_l(l):
    l = np.atleast_1d(np.asarray(l, dtype=int))
    if l.ndim != 1 or np.any(l < 1):
        raise ValueError("l must be a 1-D array of multipole orders >= 1")
    return l


def interface_matrices(rad, ref, mu, lam, l):
    """Single-interface matrices T-(n), T+(n) for n = 1..N (OSAC Eqs. 10-13).

    Returns ``(tmb, tmf, teb, tef)`` each of shape ``(L, N, 2, 2)``: backward
    (T-) and forward (T+) matrices for the magnetic (m) and electric (e)
    polarisations.
    """
    rad = np.atleast_1d(np.asarray(rad, dtype=float))
    ref = np.atleast_1d(np.asarray(ref, dtype=complex))
    mu = np.atleast_1d(np.asarray(mu, dtype=complex))
    l = _as_l(l)
    if ref.size != rad.size + 1 or mu.size != rad.size + 1:
        raise ValueError("ref and mu need len(rad) + 1 entries (last = host)")
    if np.any(np.diff(rad) <= 0) or rad[0] <= 0:
        raise ValueError("radii must be positive and strictly increasing")

    ks = 2 * np.pi * ref[:-1] / lam
    kp = 2 * np.pi * ref[1:] / lam
    xs = (ks * rad)[None, :]  # x_n       = k_n     r_n
    xp = (kp * rad)[None, :]  # x~_n      = k_{n+1} r_n
    eta = (ref[:-1] / ref[1:])[None, :]
    mur = (mu[:-1] / mu[1:])[None, :]
    lc = l[:, None]

    S_s, dS_s, X_s, dX_s = ric_j(lc, xs), ric_j_d(lc, xs), ric_h(lc, xs), ric_h_d(lc, xs)
    S_p, dS_p, X_p, dX_p = ric_j(lc, xp), ric_j_d(lc, xp), ric_h(lc, xp), ric_h_d(lc, xp)

    def back(a, b):  # T-  : a multiplies the primed-internal terms
        return -1j * np.stack(
            [
                np.stack([dX_s * S_p * a - X_s * dS_p * b, dX_s * X_p * a - X_s * dX_p * b], -1),
                np.stack([-dS_s * S_p * a + S_s * dS_p * b, -dS_s * X_p * a + S_s * dX_p * b], -1),
            ],
            -2,
        )

    def fwd(a, b):  # T+
        return -1j * np.stack(
            [
                np.stack([dX_p * S_s / a - X_p * dS_s / b, dX_p * X_s / a - X_p * dX_s / b], -1),
                np.stack([-dS_p * S_s / a + S_p * dS_s / b, -dS_p * X_s / a + S_p * dX_s / b], -1),
            ],
            -2,
        )

    tmb, tmf = back(eta, mur), fwd(eta, mur)  # Eqs. (10), (12)
    teb, tef = back(mur, eta), fwd(mur, eta)  # Eqs. (11), (13)
    return tmb, tmf, teb, tef


def _forward_products(t):
    out = np.empty_like(t)
    out[:, 0] = t[:, 0]
    for j in range(1, t.shape[1]):
        out[:, j] = t[:, j] @ out[:, j - 1]
    return out


def _backward_products(t):
    out = np.empty_like(t)
    out[:, -1] = t[:, -1]
    for j in range(t.shape[1] - 2, -1, -1):
        out[:, j] = t[:, j] @ out[:, j + 1]
    return out


@dataclass
class TransferMatrices:
    """Ordered products of transfer matrices (see module docstring)."""

    l: np.ndarray
    tm: np.ndarray
    te: np.ndarray
    mm: np.ndarray
    me: np.ndarray

    @property
    def n_shells(self) -> int:
        return self.te.shape[1]

    def T(self, n: int, pol: str) -> np.ndarray:
        """Composite T_p(n), 2 <= n <= N + 1 (paper numbering). Shape (L, 2, 2)."""
        if not 2 <= n <= self.n_shells + 1:
            raise IndexError("T(n) is defined for 2 <= n <= N + 1")
        return (self.te if pol == "e" else self.tm)[:, n - 2]

    def M(self, n: int, pol: str) -> np.ndarray:
        """Composite M_p(n), 1 <= n <= N (paper numbering). Shape (L, 2, 2)."""
        if not 1 <= n <= self.n_shells:
            raise IndexError("M(n) is defined for 1 <= n <= N")
        return (self.me if pol == "e" else self.mm)[:, n - 1]

    def tmatrix(self, pol: str) -> np.ndarray:
        """Scattering T-matrix T_pl = T21(N+1)/T11(N+1) (OSAC Eq. 18)."""
        t = self.T(self.n_shells + 1, pol)
        return t[:, 1, 0] / t[:, 0, 0]

    @property
    def a(self) -> np.ndarray:
        """Bohren-Huffman a_l = -T_El."""
        return -self.tmatrix("e")

    @property
    def b(self) -> np.ndarray:
        """Bohren-Huffman b_l = -T_Ml."""
        return -self.tmatrix("m")

    def swapped(self) -> "TransferMatrices":
        """E <-> M swapped copy (used for magnetic-dipole emitters)."""
        return TransferMatrices(self.l, tm=self.te, te=self.tm, mm=self.me, me=self.mm)


def t_mat(rad, ref, mu, lam, l) -> TransferMatrices:
    """Ordered products of electric and magnetic transfer matrices.

    Parameters
    ----------
    rad : array (N,)     outer radius of each shell (same length unit as ``lam``)
    ref : array (N+1,)   complex refractive index of each shell, host last
    mu  : array (N+1,)   relative permeability of each shell, host last
    lam : float          vacuum wavelength
    l   : array (L,)     multipole orders (normally ``1..l_max``)
    """
    l = _as_l(l)
    # Very high orders overflow (h_l ~ (2l-1)!!/x^(l+1)); callers truncate at
    # the first non-finite term, exactly as STRATIFY's rmmissing did.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        tmb, tmf, teb, tef = interface_matrices(rad, ref, mu, lam, l)
        return TransferMatrices(
            l=l,
            tm=_forward_products(tmf),
            te=_forward_products(tef),
            mm=_backward_products(tmb),
            me=_backward_products(teb),
        )
