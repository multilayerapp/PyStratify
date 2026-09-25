"""Far-field properties: cross sections and scattering amplitudes (OSAC Eqs. 19-20).

Replaces STRATIFY ``field/crs_sec.m`` and ``field/far_fld.m``; vectorised
over the wavelength batch of a :class:`~pystratify.solver.Solution`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import Solution

__all__ = ["CrossSections", "cross_sections", "scattering_amplitudes", "angular_functions"]


@dataclass
class CrossSections:
    """Cross sections (length^2) for every wavelength of the batch.

    ``sca_l``, ``ext_l``, ``abs_l`` have shape ``(W, 2, L)``: row 0 electric
    (TM, a_l), row 1 magnetic (TE, b_l).  Totals have shape ``(W,)``;
    ``q_*`` are efficiencies (normalised to pi r_N^2).
    """

    l: np.ndarray
    sca_l: np.ndarray
    ext_l: np.ndarray
    abs_l: np.ndarray
    geometric: float

    sca = property(lambda self: self.sca_l.sum((-2, -1)))
    ext = property(lambda self: self.ext_l.sum((-2, -1)))
    abs = property(lambda self: self.abs_l.sum((-2, -1)))
    q_sca = property(lambda self: self.sca / self.geometric)
    q_ext = property(lambda self: self.ext / self.geometric)
    q_abs = property(lambda self: self.abs / self.geometric)


def cross_sections(sol: Solution) -> CrossSections:
    """Scattering, extinction and absorption (OSAC Eq. 19).

    As in STRATIFY the real part of the host wavenumber is used, so an
    absorbing host is only handled approximately.
    """
    l = sol.l
    k = sol.k[:, -1][:, None, None]  # (W,1,1)
    kr = k.real
    ab = np.stack([sol.a, sol.b], axis=1)  # (W,2,L)
    sca = 2 * np.pi / kr**2 * (2 * l + 1) * np.abs(ab) ** 2
    ext = 2 * np.pi / kr * (2 * l + 1) * (ab / k).real
    return CrossSections(l=l, sca_l=sca, ext_l=ext, abs_l=ext - sca, geometric=float(np.pi * sol.rad[-1] ** 2))


def angular_functions(l, theta):
    """Bohren-Huffman pi_l = P_l^1/sin(theta), tau_l = dP_l^1/dtheta (no
    Condon-Shortley phase), regular at theta = 0 and pi (AUDIT.md M6).
    Returns arrays of shape ``(len(l), n_theta)``."""
    l = np.asarray(l)
    mu = np.cos(np.atleast_1d(np.asarray(theta, dtype=float)))
    lmax = int(l.max())
    pi = np.zeros((lmax + 1, mu.size))
    tau = np.zeros((lmax + 1, mu.size))
    pi[1] = 1.0
    tau[1] = mu
    for n in range(2, lmax + 1):
        pi[n] = ((2 * n - 1) * mu * pi[n - 1] - n * pi[n - 2]) / (n - 1)
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    return pi[l], tau[l]


def scattering_amplitudes(sol: Solution, theta):
    """Polarised scattering amplitudes S_par, S_per (OSAC Eq. 20), shape (W, n_theta).

    Paper / far_fld.m convention: S_par = -S2 and S_per = -S1 of Bohren &
    Huffman (the sign is irrelevant for intensities and Stokes parameters).
    """
    l = sol.l
    pi, tau = angular_functions(l, theta)
    coef = (2 * l + 1) / (l * (l + 1))
    a, b = sol.a * coef, sol.b * coef  # (W, L)
    par = -(a @ tau + b @ pi)
    per = -(a @ pi + b @ tau)
    return par, per
