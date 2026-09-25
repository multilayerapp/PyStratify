"""Far field: cross sections and scattering amplitudes, vectorised over wavelengths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import Solution

__all__ = ["CrossSections", "cross_sections", "scattering_amplitudes", "angular_functions"]


@dataclass(frozen=True)
class CrossSections:
    """Cross sections (length^2) for every wavelength of the batch.

    ``*_by_order`` have shape ``(W, 2, L)``: axis 1 is the polarisation
    (``TM`` = electric multipoles, ``TE`` = magnetic).  Totals have shape
    ``(W,)``; ``q_*`` are efficiencies, normalised to pi R^2 of the outer radius.
    """

    orders: np.ndarray
    sca_by_order: np.ndarray
    ext_by_order: np.ndarray
    abs_by_order: np.ndarray
    geometric: float

    @property
    def sca(self) -> np.ndarray:
        return self.sca_by_order.sum(axis=(-2, -1))

    @property
    def ext(self) -> np.ndarray:
        return self.ext_by_order.sum(axis=(-2, -1))

    @property
    def abs(self) -> np.ndarray:
        return self.abs_by_order.sum(axis=(-2, -1))

    @property
    def q_sca(self) -> np.ndarray:
        return self.sca / self.geometric

    @property
    def q_ext(self) -> np.ndarray:
        return self.ext / self.geometric

    @property
    def q_abs(self) -> np.ndarray:
        return self.abs / self.geometric


def cross_sections(sol: Solution) -> CrossSections:
    """Scattering, extinction and absorption cross sections.

    The real part of the host wavenumber is used, so an absorbing host is
    handled only approximately.
    """
    l = sol.orders
    k = sol.k[:, -1][:, None, None]
    t = np.moveaxis(sol.t, 0, 1)  # (W, 2, L)
    weight = 2 * np.pi * (2 * l + 1) / k.real
    sca = weight / k.real * np.abs(t) ** 2
    ext = -weight * (t / k).real
    return CrossSections(
        orders=l,
        sca_by_order=sca,
        ext_by_order=ext,
        abs_by_order=ext - sca,
        geometric=float(np.pi * sol.radii[-1] ** 2),
    )


def angular_functions(orders, theta):
    """Bohren-Huffman pi_l = P_l^1 / sin(theta) and tau_l = dP_l^1 / d(theta).

    Upward recurrence in cos(theta), regular at theta = 0 and pi.  Returns two
    arrays of shape ``(len(orders), len(theta))``.
    """
    orders = np.asarray(orders)
    c = np.cos(np.atleast_1d(np.asarray(theta, dtype=float)))
    top = int(orders.max())
    pi = np.zeros((top + 1, c.size))
    tau = np.zeros((top + 1, c.size))
    pi[1] = 1.0
    tau[1] = c
    for n in range(2, top + 1):
        pi[n] = ((2 * n - 1) * c * pi[n - 1] - n * pi[n - 2]) / (n - 1)
        tau[n] = n * c * pi[n] - (n + 1) * pi[n - 1]
    return pi[orders], tau[orders]


def scattering_amplitudes(sol: Solution, theta):
    """Scattering amplitudes (S_par, S_per), each of shape ``(W, len(theta))``.

    S_par = -S2 and S_per = -S1 in Bohren & Huffman's notation, the convention
    of Rasskazov, Carney & Moroz (2020); intensities and Stokes parameters are
    unaffected by the sign.
    """
    l = sol.orders
    pi, tau = angular_functions(l, theta)
    weight = (2 * l + 1) / (l * (l + 1))
    a, b = sol.a * weight, sol.b * weight
    return -(a @ tau + b @ pi), -(a @ pi + b @ tau)
