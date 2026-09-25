"""Electromagnetic energy density and stored energy under plane-wave illumination.

Rasskazov, Moroz & Carney, JOSA A 36, 1591 (2019); Rasskazov, Carney & Moroz,
OSA Continuum 3, 2290 (2020), Eqs. (21)-(27).

Energy in a dispersive medium is weighted by the prefactor G_e, which for a
Drude metal is Re eps + 2 (omega/gamma) Im eps (Loudon); a non-dispersive
medium has G_e = Re eps and G_m = Re mu.

Stored energy per shell uses the closed-form Lommel integrals.  Their lossy
form divides Im(x f f*) by x^2 - x*^2, both of which vanish as Im k -> 0, so
it loses ~1e-17/Im(n) relative accuracy; shells with
0 < |Im k| < WEAK_LOSS |k| are integrated by Gauss-Legendre quadrature of the
energy density instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .drude import DRUDE, DrudeModel
from .nearfield import radial_functions
from .riccati import log_riccati
from .solver import TE, TM, Solution

__all__ = [
    "electric_prefactor",
    "energy_prefactors",
    "EnergyDensity",
    "energy_density",
    "ShellEnergy",
    "shell_energy",
    "WEAK_LOSS",
]

#: below this |Im k|/|k| a lossy shell is integrated numerically (see module doc)
WEAK_LOSS = 1e-3


@lru_cache(maxsize=64)
def gauss_legendre(n):
    """Cached Gauss-Legendre nodes and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def electric_prefactor(eps, wavelength, damping_wavelength=None):
    """G_e = Re eps, or Re eps + 2 (omega/gamma) Im eps for a Drude metal.

    ``damping_wavelength`` = 2 pi c / gamma in the unit of ``wavelength``;
    ``None`` for a non-dispersive medium.
    """
    eps = np.asarray(eps, dtype=complex)
    if damping_wavelength is None:
        return eps.real
    return eps.real + 2 * (damping_wavelength / wavelength) * eps.imag


def energy_prefactors(n, mu, wavelength_nm, drude=None):
    """(G_e, G_m) for every shell and the host at one wavelength (in nm).

    ``drude``: optional sequence (length N + 1) of :class:`DrudeModel`,
    :data:`DRUDE` keys or ``None``; listed metals use their damping, every
    other medium is treated as non-dispersive.
    """
    n = np.atleast_1d(np.asarray(n, dtype=complex))
    mu = np.atleast_1d(np.asarray(mu, dtype=complex))
    eps = n**2 / mu
    drude = [None] * n.size if drude is None else list(drude)
    if len(drude) != n.size:
        raise ValueError("drude needs one entry per shell and the host")
    g_e = np.empty(n.size)
    for i, model in enumerate(drude):
        model = DRUDE[model] if isinstance(model, str) else model
        damping = model.damping_wavelength if isinstance(model, DrudeModel) else None
        g_e[i] = electric_prefactor(eps[i], wavelength_nm, damping)
    return g_e, mu.real.copy()


def _default_prefactors(sol, w):
    eps = sol.n[w] ** 2 / sol.mu[w]
    return eps.real, sol.mu[w].real


@dataclass(frozen=True)
class EnergyDensity:
    """Orientation-averaged intensities and energy densities at radii ``r``.

    ``intensity_e = <|E|^2>/|E0|^2`` and ``intensity_h = <|H|^2>/|E0|^2``,
    averaged over the sphere of radius r.  With ``normalized`` the densities
    are relative to the incident wave's electric (magnetic) energy density in
    the host.
    """

    r: np.ndarray
    intensity_e: np.ndarray
    intensity_h: np.ndarray
    density_e: np.ndarray
    density_h: np.ndarray
    normalized: bool

    @property
    def density_em(self) -> np.ndarray:
        total = self.density_e + self.density_h
        return total / 2 if self.normalized else total


def _intensities(sol: Solution, w: int, r):
    l = sol.orders
    functions, _, shell = radial_functions(sol, w, r, offsets=(-1, 0, 1), with_incident=False)
    fe_prev, fe, fe_next = functions[TM]
    fm_prev, fm, fm_next = functions[TE]

    def sq(v):
        return np.abs(v) ** 2

    i_e = ((2 * l + 1) * sq(fm) + (l + 1) * sq(fe_prev) + l * sq(fe_next)).sum(axis=1) / 2
    i_h = ((2 * l + 1) * sq(fe) + (l + 1) * sq(fm_prev) + l * sq(fm_next)).sum(axis=1) / 2
    eps = sol.n[w] ** 2 / sol.mu[w]
    return i_e, i_h * (np.abs(eps) / np.abs(sol.mu[w]))[shell], shell


def energy_density(sol: Solution, r, prefactors=None, wavelength_index: int = 0, normalize=True) -> EnergyDensity:
    """Energy density and averaged intensities at radii ``r``.

    ``prefactors = (G_e, G_m)`` per shell (default: non-dispersive).  Points
    on an interface belong to the outer shell; r = 0 is evaluated at 1e-9 R_0.
    """
    w = wavelength_index
    r = np.maximum(np.atleast_1d(np.asarray(r, dtype=float)), 1e-9 * sol.radii[0])
    g_e, g_m = prefactors if prefactors is not None else _default_prefactors(sol, w)
    i_e, i_h, shell = _intensities(sol, w, r)
    d_e = np.pi * np.asarray(g_e)[shell] * i_e
    d_h = np.pi * np.asarray(g_m)[shell] * i_h
    if normalize:
        host = np.pi * (sol.n[w, -1] ** 2 / sol.mu[w, -1]).real
        d_e, d_h = d_e / host, d_h / host
    return EnergyDensity(r=r, intensity_e=i_e, intensity_h=i_h, density_e=d_e, density_h=d_h, normalized=normalize)


@dataclass(frozen=True)
class ShellEnergy:
    """Electric and magnetic energy in each shell 0..N-1 (arrays of length N).

    Normalised (default): relative to the incident wave's energy in the same
    volume, ``total = (electric + magnetic)/2``.  Otherwise in |E0|^2 x
    length^3, ``total = electric + magnetic``.  ``method`` records how each
    shell was integrated.
    """

    electric: np.ndarray
    magnetic: np.ndarray
    total: np.ndarray
    normalized: bool
    method: tuple


def _boundary_functions(sol, w, s, radius, polarisation):
    """f_{l+o}(k_s R), o = -1..2, with plane-wave coefficients."""
    l = sol.orders
    x = sol.k[w, s] * radius
    log_psi, log_xi = log_riccati(np.array([x]), l.size + 2)
    la, lb = sol.log_a[polarisation, s, w], sol.log_b[polarisation, s, w]
    with np.errstate(under="ignore"):
        return {o: (np.exp(la + log_psi[0, l + o]) + np.exp(lb + log_xi[0, l + o])) / x for o in (-1, 0, 1, 2)}, x


def _lommel(sol, w, s, radius, lossy):
    """Closed-form radial integrals at one boundary: (electric, magnetic)."""
    l = sol.orders
    fe, x = _boundary_functions(sol, w, s, radius, TM)
    fm, _ = _boundary_functions(sol, w, s, radius, TE)

    def pair(f, o):  # Lommel combination of orders (l+o-1, l+o)
        fa, fb = f[o - 1], f[o]
        if lossy:
            return 2j * np.imag(x * np.conj(fa) * fb)
        nu = l + o - 1
        return x * (np.abs(fa) ** 2 + np.abs(fb) ** 2) - (2 * nu + 1) * np.real(np.conj(fa) * fb)

    e = np.sum((2 * l + 1) * pair(fm, 1) + (l + 1) * pair(fe, 0) + l * pair(fe, 2))
    m = np.sum((2 * l + 1) * pair(fe, 1) + (l + 1) * pair(fm, 0) + l * pair(fm, 2))
    den = (x**2 - np.conj(x) ** 2) if lossy else 2 * x
    return 0.5 * radius**3 * e / den, 0.5 * radius**3 * m / den


def _quadrature(sol, w, s, r_in, r_out, nodes=None):
    """int <|E|^2> r^2 dr and int <|H|^2> r^2 dr over shell s."""
    nodes = nodes or int(max(64, sol.orders.size + 48, 6 * abs(sol.k[w, s]) * (r_out - r_in)))
    t, wt = gauss_legendre(nodes)
    half = 0.5 * (r_out - r_in)
    r = half * t + 0.5 * (r_out + r_in)
    i_e, i_h, _ = _intensities(sol, w, r)
    return np.sum(half * wt * i_e * r**2), np.sum(half * wt * i_h * r**2)


def shell_energy(
    sol: Solution, prefactors=None, wavelength_index: int = 0, normalize=True, method="auto"
) -> ShellEnergy:
    """Electric and magnetic energy stored in each shell.

    ``method``: ``'auto'`` (closed form where well conditioned, quadrature for
    weak loss), ``'lommel'`` or ``'quadrature'`` (for cross-checks).
    """
    if method not in ("auto", "lommel", "quadrature"):
        raise ValueError("method must be 'auto', 'lommel' or 'quadrature'")
    w = wavelength_index
    g_e, g_m = prefactors if prefactors is not None else _default_prefactors(sol, w)
    radii = sol.radii
    N = radii.size
    eps = sol.n[w] ** 2 / sol.mu[w]
    eps_over_mu = np.abs(eps) / np.abs(sol.mu[w])
    electric, magnetic, used = np.zeros(N), np.zeros(N), []
    for s in range(N):
        k = sol.k[w, s]
        r_in = radii[s - 1] if s else 0.0
        weak = k.imag != 0 and abs(k.imag) < WEAK_LOSS * abs(k)
        how = method if method != "auto" else ("quadrature" if weak else "lommel")
        if how == "quadrature":
            e, m = _quadrature(sol, w, s, r_in, radii[s])
            electric[s], magnetic[s] = e * g_e[s], m * g_m[s]  # intensity_h already carries |eps|/|mu|
        else:
            lossy = k.imag != 0
            outer = _lommel(sol, w, s, radii[s], lossy)
            inner = _lommel(sol, w, s, r_in, lossy) if r_in > 0 else (0.0, 0.0)
            electric[s] = np.real(outer[0] - inner[0]) * g_e[s]
            magnetic[s] = np.real(outer[1] - inner[1]) * g_m[s] * eps_over_mu[s]
        used.append(how)
    if normalize:
        volume = np.diff(np.r_[0, radii**3])
        host = (sol.n[w, -1] ** 2 / sol.mu[w, -1]).real
        electric, magnetic = 3 * electric / volume / host, 3 * magnetic / volume / host
        return ShellEnergy(electric, magnetic, (electric + magnetic) / 2, True, tuple(used))
    electric, magnetic = np.pi * electric, np.pi * magnetic
    return ShellEnergy(electric, magnetic, electric + magnetic, False, tuple(used))
