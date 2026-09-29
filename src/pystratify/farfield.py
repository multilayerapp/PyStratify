"""Far field under plane-wave illumination, vectorised over wavelengths.

Cross sections, scattering amplitudes, the full amplitude and Mueller
matrices, helicity-resolved cross sections, and angle-resolved scattering
patterns with directivity for any incident polarisation.

Conventions (Bohren & Huffman, chs. 3-4): the incident wave travels along +z
with Jones vector (E_x, E_y) and time dependence exp(-i omega t).  In the far
zone E_sca = exp(ikr) / (-ikr) X(theta, phi), with

    (X_par, X_perp) = [[S2, S3], [S4, S1]] (E_par, E_perp),
    E_par = E_x cos(phi) + E_y sin(phi),   E_perp = E_x sin(phi) - E_y cos(phi),
    X_theta = X_par,   X_phi = -X_perp,

so that dsigma/dOmega = |X|^2 / k^2.  Everything here is computed from the
T-matrix blocks ``sol.t_matrix`` (W, L, 2, 2) in the (TM, TE) basis: diagonal
for achiral spheres (S3 = S4 = 0); chiral shells couple TM and TE and give
S3 = -S4.

Helicity: +1 is the field (x + iy)/sqrt(2) exp(ikz) - left-circular in the
optics convention (Jackson, sec. 7.2) - and -1 is (x - iy)/sqrt(2) exp(ikz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import TE, TM, Solution, _partial_waves

__all__ = [
    "CrossSections",
    "cross_sections",
    "scattering_amplitudes",
    "angular_functions",
    "amplitude_matrix",
    "mueller_matrix",
    "HelicityCrossSections",
    "helicity_cross_sections",
    "ScatteringPattern",
    "scattering_pattern",
    "HELICITY",
]

#: helicity of index 0 and 1 of every helicity axis
HELICITY = (+1, -1)


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


def _achiral(sol, name):
    if not isinstance(sol, Solution):
        raise TypeError(f"{name} needs an achiral Solution; for chiral spheres use helicity_cross_sections")


def cross_sections(sol: Solution) -> CrossSections:
    """Scattering, extinction and absorption cross sections.

    The real part of the host wavenumber is used, so an absorbing host is
    handled only approximately.
    """
    _achiral(sol, "cross_sections")
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


def scattering_amplitudes(sol: Solution, theta, orders=None, polarisations=(TM, TE)):
    """Scattering amplitudes (S_par, S_per), each of shape ``(W, len(theta))``.

    S_par = -S2 and S_per = -S1 in Bohren & Huffman's notation, the convention
    of Rasskazov, Carney & Moroz (2020); intensities and Stokes parameters are
    unaffected by the sign.  :func:`amplitude_matrix` gives S1..S4 themselves.

    ``orders`` and ``polarisations`` (:data:`TM`, the electric multipoles a_l;
    :data:`TE`, the magnetic b_l) keep only those partial waves, as in
    :func:`near_field`: the far field of disjoint selections adds up amplitude
    by amplitude, not in |S|^2, which carries their interference.
    """
    _achiral(sol, "scattering_amplitudes")
    l = sol.orders
    pi, tau = angular_functions(l, theta)
    weight = (2 * l + 1) / (l * (l + 1))
    kept = _partial_waves(l, orders, polarisations)
    tm, te = (1.0, 1.0) if kept is None else (kept[TM], kept[TE])
    a, b = sol.a * weight * tm, sol.b * weight * te
    return -(a @ tau + b @ pi), -(a @ pi + b @ tau)


def _host_k(sol) -> np.ndarray:
    """Real host wavenumber per wavelength (as :func:`cross_sections`)."""
    return sol.k[:, -1].real


def amplitude_matrix(sol, theta):
    """Bohren-Huffman amplitude scattering matrix (S1, S2, S3, S4).

    ``sol``: :class:`~pystratify.Solution` or :class:`~pystratify.ChiralSolution`.
    Each element has shape ``(W, len(theta))``.
    """
    l = sol.orders
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    unique, index = np.unique(theta, return_inverse=True)
    pi, tau = angular_functions(l, unique)
    w = (2 * l + 1) / (l * (l + 1))
    t = sol.t_matrix * w[:, None, None]  # (W, L, 2, 2)
    tm_tm, tm_te, te_tm, te_te = t[..., 0, 0], t[..., 0, 1], t[..., 1, 0], t[..., 1, 1]
    s1 = -(tm_tm @ pi + te_te @ tau)
    s2 = -(tm_tm @ tau + te_te @ pi)
    s3 = -1j * (tm_te @ tau + te_tm @ pi)
    s4 = 1j * (tm_te @ pi + te_tm @ tau)
    return tuple(s[:, index] for s in (s1, s2, s3, s4))


#: Stokes vector (I, Q, U, V) from the coherency vector (E_par E_par*, E_par E_perp*, E_perp E_par*, E_perp E_perp*)
_STOKES = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])


def mueller_matrix(sol, theta):
    """Mueller matrix, shape ``(W, len(theta), 4, 4)``, in Bohren & Huffman's
    convention (their Eq. 3.16 and Stokes vector, prefactor 1/(k r)^2 omitted):
    for an achiral sphere M11 = (|S1|^2 + |S2|^2)/2, M12 = (|S2|^2 - |S1|^2)/2,
    M33 = Re(S1 S2*), M34 = Im(S2 S1*).
    """
    s1, s2, s3, s4 = amplitude_matrix(sol, theta)
    jones = np.stack([np.stack([s2, s3], -1), np.stack([s4, s1], -1)], -2)  # (W, T, 2, 2)
    coherency = np.einsum("...ik,...jl->...ijkl", jones, np.conj(jones)).reshape(jones.shape[:-2] + (4, 4))
    return np.real(_STOKES @ coherency @ np.linalg.inv(_STOKES))


@dataclass(frozen=True)
class HelicityCrossSections:
    """Cross sections for circularly polarised illumination.

    ``*_by_order`` have shape ``(W, 2, L)`` with axis 1 the helicity
    (index 0: +1, left-circular in the optics convention; index 1: -1).
    Totals ``ext``, ``sca``, ``abs`` have shape ``(W, 2)``; linear or
    unpolarised light gives their mean.  ``cd_*`` = (+1) - (-1) and
    ``g_*`` = 2 cd / ((+1) + (-1)) are the circular dichroism and the
    dissymmetry factor of extinction, scattering and absorption.
    """

    orders: np.ndarray
    sca_by_order: np.ndarray
    ext_by_order: np.ndarray
    abs_by_order: np.ndarray
    geometric: float

    @property
    def sca(self) -> np.ndarray:
        return self.sca_by_order.sum(axis=-1)

    @property
    def ext(self) -> np.ndarray:
        return self.ext_by_order.sum(axis=-1)

    @property
    def abs(self) -> np.ndarray:
        return self.abs_by_order.sum(axis=-1)

    @property
    def q_sca(self) -> np.ndarray:
        return self.sca / self.geometric

    @property
    def q_ext(self) -> np.ndarray:
        return self.ext / self.geometric

    @property
    def q_abs(self) -> np.ndarray:
        return self.abs / self.geometric

    @staticmethod
    def _cd(v):
        return v[..., 0] - v[..., 1]

    @staticmethod
    def _g(v):
        return 2 * (v[..., 0] - v[..., 1]) / (v[..., 0] + v[..., 1])

    @property
    def cd_ext(self) -> np.ndarray:
        return self._cd(self.ext)

    @property
    def cd_sca(self) -> np.ndarray:
        return self._cd(self.sca)

    @property
    def cd_abs(self) -> np.ndarray:
        return self._cd(self.abs)

    @property
    def g_ext(self) -> np.ndarray:
        return self._g(self.ext)

    @property
    def g_sca(self) -> np.ndarray:
        return self._g(self.sca)

    @property
    def g_abs(self) -> np.ndarray:
        return self._g(self.abs)


#: (TM, TE) amplitudes of the multipoles in a plane wave of helicity +1 and -1 (up to a common factor)
_HELICITY_VECTORS = np.array([[1.0, 1.0], [1.0, -1.0]])


def helicity_cross_sections(sol) -> HelicityCrossSections:
    """Extinction, scattering and absorption for both circular polarisations.

    Works for achiral (identical helicities) and chiral spheres.  A plane
    wave of helicity s excites only m = s multipoles with (TM, TE) amplitudes
    proportional to (1, s), so with T the (TM, TE) T-matrix block of order l

        C_ext = -(2 pi / k^2) sum (2l + 1) Re[v^T T v],
        C_sca =  (2 pi / k^2) sum (2l + 1) |T v|^2,     v = (1, s).
    """
    l = sol.orders
    k = _host_k(sol)[:, None, None]
    t = sol.t_matrix  # (W, L, 2, 2)
    tv = np.einsum("wlij,hj->whli", t, _HELICITY_VECTORS)  # (W, 2, L, 2)
    weight = 2 * np.pi * (2 * l + 1) / k**2
    ext = -weight * np.einsum("hi,whli->whl", _HELICITY_VECTORS, tv).real
    sca = weight * np.sum(np.abs(tv) ** 2, axis=-1)
    return HelicityCrossSections(
        orders=l,
        sca_by_order=sca,
        ext_by_order=ext,
        abs_by_order=ext - sca,
        geometric=float(np.pi * sol.radii[-1] ** 2),
    )


class FarFieldPolarization:
    """Polarisation analysis shared by far-field patterns (fields ``e_theta``, ``e_phi``)."""

    e_theta: np.ndarray
    e_phi: np.ndarray

    @property
    def intensity(self) -> np.ndarray:
        """|e_theta|^2 + |e_phi|^2."""
        return np.abs(self.e_theta) ** 2 + np.abs(self.e_phi) ** 2

    @property
    def helicity(self) -> tuple:
        """Amplitudes (e_+, e_-) on (theta_hat +- i phi_hat)/sqrt(2): helicity +1 and -1 along the outgoing direction."""
        return (self.e_theta - 1j * self.e_phi) / np.sqrt(2), (self.e_theta + 1j * self.e_phi) / np.sqrt(2)

    @property
    def circular_polarization(self) -> np.ndarray:
        """Degree of circular polarisation (|e_+|^2 - |e_-|^2) / (|e_+|^2 + |e_-|^2), +1 for pure helicity +1."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return -2 * np.imag(self.e_theta * np.conj(self.e_phi)) / self.intensity

    def stokes(self) -> np.ndarray:
        """Stokes parameters (I, Q, U, V) stacked on the last axis, in Bohren &
        Huffman's convention (par = theta_hat, perp = -phi_hat, as :func:`mueller_matrix`);
        note their V = -I x ``circular_polarization``."""
        par, perp = self.e_theta, -self.e_phi
        return np.stack(
            [
                np.abs(par) ** 2 + np.abs(perp) ** 2,
                np.abs(par) ** 2 - np.abs(perp) ** 2,
                2 * np.real(par * np.conj(perp)),
                -2 * np.imag(par * np.conj(perp)),
            ],
            axis=-1,
        )


@dataclass(frozen=True)
class ScatteringPattern(FarFieldPolarization):
    """Scattered far field for one incident polarisation.

    ``e_theta``, ``e_phi``: components of the vector amplitude X (E_sca =
    exp(ikr)/(-ikr) X for a unit incident field), shape ``(W,) + S`` with S
    the broadcast shape of ``theta`` and ``phi``.  ``sca``: scattering cross
    section for this polarisation, ``(W,)``, from the multipole coefficients.
    """

    theta: np.ndarray
    phi: np.ndarray
    e_theta: np.ndarray
    e_phi: np.ndarray
    k: np.ndarray
    sca: np.ndarray
    polarization: np.ndarray

    def _per_wavelength(self, v):
        return v.reshape(v.shape + (1,) * self.theta.ndim)

    @property
    def differential_cross_section(self) -> np.ndarray:
        """dsigma/dOmega = |X|^2 / k^2 (length^2 per steradian)."""
        return self.intensity / self._per_wavelength(self.k) ** 2

    @property
    def directivity(self) -> np.ndarray:
        """4 pi (dsigma/dOmega) / sigma_sca: 1 for isotropic scattering, integrates to 4 pi."""
        return 4 * np.pi * self.differential_cross_section / self._per_wavelength(self.sca)

    @property
    def phase_function(self) -> np.ndarray:
        """(dsigma/dOmega) / sigma_sca, normalised to 1 over the sphere."""
        return self.directivity / (4 * np.pi)


def _jones(polarization):
    p = np.asarray(polarization, dtype=complex).ravel()
    norm = np.linalg.norm(p)
    if p.shape != (2,) or not np.isfinite(norm) or norm == 0:
        raise ValueError("polarization must be a nonzero Jones vector (E_x, E_y)")
    return p / norm


def scattering_pattern(sol, theta, phi=0.0, polarization=(1.0, 0.0)) -> ScatteringPattern:
    """Angle-resolved scattered far field, differential cross section and directivity.

    ``sol``: :class:`~pystratify.Solution` or :class:`~pystratify.ChiralSolution`.
    ``theta``, ``phi``: scattering directions (broadcast together; theta from
    the propagation direction +z).  ``polarization``: incident Jones vector
    (E_x, E_y), normalised internally; e.g. (1, 1j) for helicity +1.
    """
    jones = _jones(polarization)
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    s1, s2, s3, s4 = amplitude_matrix(sol, theta.ravel())
    cp, sp = np.cos(phi.ravel()), np.sin(phi.ravel())
    e_par = jones[0] * cp + jones[1] * sp
    e_perp = jones[0] * sp - jones[1] * cp
    shape = (s1.shape[0],) + theta.shape
    x_theta = (s2 * e_par + s3 * e_perp).reshape(shape)
    x_phi = -(s4 * e_par + s1 * e_perp).reshape(shape)
    helicity = np.array([jones[0] - 1j * jones[1], jones[0] + 1j * jones[1]]) / np.sqrt(2)
    sca = helicity_cross_sections(sol).sca @ np.abs(helicity) ** 2
    return ScatteringPattern(
        theta=theta, phi=phi, e_theta=x_theta, e_phi=x_phi, k=_host_k(sol), sca=sca, polarization=jones
    )
