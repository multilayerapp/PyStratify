"""Casimir-Polder potential of a ground-state atom in or near a multilayered sphere.

For an isotropic electric polarizability alpha(i xi) (Buhmann & Welsch, Prog. Quantum Electron. 31,
51 (2007), Eq. (110)),

    U(r) = (hbar mu_0 / 2 pi) int_0^inf dxi xi^2 alpha(i xi) Tr G_s(r, r, i xi),

with the scattered Green's dyadic at imaginary frequency.  The trace is the dipole series of the
normalized formulation at k = i xi n(i xi)/c, Tr G_s = (i k / 6 pi)(g_perp + 2 g_par) (shell
normalization, as for the decay rates and the shift), and with alpha = 4 pi eps_0 alpha_V in volume
units

    U(r) = -(hbar / 3 pi c^3) int_0^inf dxi xi^3 n_d(i xi) alpha_V(i xi) [g_perp + 2 g_par](i xi).

At imaginary frequency every auxiliary function is real up to a fixed phase (P real, A and B
imaginary, jbar real), the Hankel-type solution decays like exp(-kappa r), and the normalized
series converge geometrically at the emitter as for real frequencies; nothing can overflow.
The integral is done by Gauss-Legendre quadrature in ln xi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .energy import gauss_legendre
from .normalized import L_CAP, _complex_converged, _dipole_series, _Sweep
from .convergence import orders_needed

__all__ = ["CasimirPolder", "casimir_polder"]

HBAR = 1.054571817e-34  # J s
C_NM = 2.99792458e17  # speed of light, nm/s


@dataclass(frozen=True)
class CasimirPolder:
    """Casimir-Polder potential ``energy`` (J) at the radii ``r``; ``xi`` and ``integrand`` (P, X) are the
    quadrature nodes (rad/s) and xi^3 n_d alpha_V [g_perp + 2 g_par], ``orders`` the most multipoles used."""

    r: np.ndarray
    energy: np.ndarray
    xi: np.ndarray
    integrand: np.ndarray
    orders: int
    converged: np.ndarray

    @property
    def frequency(self) -> np.ndarray:
        """U / h in Hz."""
        return self.energy / (2 * np.pi * HBAR)


def casimir_polder(radii, eps, r, polarizability, mu=None, xi_range=(1e9, 1e19), nodes=96, tol=1e-10, l_cap=L_CAP):
    """Casimir-Polder potential of an atom at radius (radii) ``r`` (nm), in any layer.

    ``eps(xi)`` and ``mu(xi)`` (optional) return the (N + 1,) permittivities and permeabilities of the
    layers at imaginary frequency i xi (xi in rad/s; real and >= 1 for passive media), host last.
    ``polarizability(xi)`` is the atom's isotropic polarizability alpha(i xi) / (4 pi eps_0) in nm^3.
    The integral over xi runs over ``xi_range`` with ``nodes`` Gauss-Legendre nodes in ln xi; every
    order sum meets ``tol`` (automatic truncation, at most ``l_cap`` orders).  Lengths in nm.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(~(r > 0)) or np.any(np.isin(r, radii)):
        raise ValueError("positions must be positive and off the interfaces")
    shells = np.searchsorted(radii, r, side="right")
    t, w = gauss_legendre(nodes)
    lo, hi = np.log(xi_range[0]), np.log(xi_range[1])
    ln_xi = 0.5 * (hi - lo) * t + 0.5 * (hi + lo)
    xi, w = np.exp(ln_xi), 0.5 * (hi - lo) * w * np.exp(ln_xi)  # d xi = xi d ln xi
    q = np.max(np.minimum(radii[None, :] / r[:, None], r[:, None] / radii[None, :]))
    integrand = np.zeros((r.size, xi.size))
    ok = np.ones(r.size, dtype=bool)
    used = 0
    for i, x in enumerate(xi):
        e = np.atleast_1d(np.asarray(eps(x), dtype=complex))
        m = np.ones_like(e) if mu is None else np.atleast_1d(np.asarray(mu(x), dtype=complex))
        if e.shape != (radii.size + 1,) or m.shape != e.shape:
            raise ValueError(f"eps and mu must return ({radii.size + 1},) arrays, host last")
        n = np.sqrt(e * m).real + 0j  # real at imaginary frequency for passive media
        k = 1j * (x / C_NM) * n
        L = int(min(l_cap, max(32, orders_needed(q, tol, l_cap))))
        while True:
            terms = _Sweep(radii, n, m, k, L).at(r)
            g, _ = _dipole_series(terms, "electric")
            good = _complex_converged(g, tol)
            if good.all() or L >= l_cap:
                break
            L = min(2 * L, l_cap)
        used = max(used, L)
        ok &= good
        trace = (g[:, :, 0] + 2 * g[:, :, 1]).sum(axis=1).real
        integrand[:, i] = x**3 * n[shells].real * polarizability(x) * trace
    energy = -HBAR / (3 * np.pi * C_NM**3) * integrand @ w
    return CasimirPolder(r=r, energy=energy, xi=xi, integrand=integrand, orders=used, converged=ok)
