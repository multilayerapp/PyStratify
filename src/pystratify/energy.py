"""Electromagnetic energy density and stored energy (plane-wave illumination).

Replaces STRATIFY ``energy/nrg_dns.m``, ``nrg_tot.m``, ``F_lossy.m``,
``F_lossless.m`` and ``G_prefac.m`` (OSAC Eqs. 21-27; Rasskazov, Moroz &
Carney, JOSA A 36, 1591 (2019)).

* ``G_prefac.m`` uses omega_p/gamma where OSAC Eq. (23) has omega/gamma, and
  the Drude-fit eps instead of the shell's eps (AUDIT.md M5);
  :func:`g_electric` implements Eq. (23).
* The lossy Lommel formula (Eq. 27) divides Im(x f f*) by x^2 - x*^2, both of
  which vanish as Im(k) -> 0; STRATIFY switches to the lossless form only at
  Im(n) == 0 exactly, so a weakly absorbing shell loses ~1e-17/Im(n) relative
  accuracy (2 % at Im n = 1e-14; AUDIT.md M10).  Here
  shells with 0 < |Im k| < ``WEAK_LOSS`` |k| are integrated by Gauss-Legendre
  quadrature of the energy density instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import DRUDE
from .nearfield import radial_functions
from .riccati import log_riccati
from .solver import Solution

__all__ = [
    "g_electric",
    "g_prefactors",
    "EnergyDensity",
    "energy_density",
    "ShellEnergy",
    "total_energy",
    "WEAK_LOSS",
]

#: below this |Im k|/|k| a lossy shell is integrated numerically (see module doc)
WEAK_LOSS = 1e-3


def g_electric(eps, lam, lam_gamma=None):
    """Electric energy prefactor G_e (OSAC Eqs. 22-23).

    ``lam_gamma = 2 pi c / gamma_D`` in the unit of ``lam``; ``None`` for a
    non-dispersive shell (G_e = Re eps).
    """
    eps = np.asarray(eps, dtype=complex)
    if lam_gamma is None:
        return eps.real
    return eps.real + 2 * (lam_gamma / lam) * eps.imag  # 2 (omega/gamma) Im eps


def g_prefactors(ref, mu, lam_nm, materials=None):
    """(G_e, G_m) for every shell and the host at one wavelength (in nm).

    ``materials``: optional list (len N + 1) of :data:`~pystratify.materials.DRUDE`
    keys or ``None``; listed metals use Eq. (23), everything else Eq. (22).
    """
    ref = np.atleast_1d(np.asarray(ref, dtype=complex))
    mu = np.atleast_1d(np.asarray(mu, dtype=complex))
    eps = ref**2 / mu
    if materials is None:
        materials = [None] * ref.size
    ge = np.array(
        [g_electric(e, lam_nm, DRUDE[m]["lam_gamma"] if m else None) for e, m in zip(eps, materials)], dtype=float
    )
    return ge, mu.real.copy()


def _default_g(sol, w):
    eps = sol.ref[w] ** 2 / sol.mu[w]
    return eps.real, sol.mu[w].real


# ----------------------------------------------------------------- density


@dataclass
class EnergyDensity:
    """Orientation-averaged intensities (Eq. 24) and energy densities (Eq. 21).

    ``I_e = <|E|^2>/|E0|^2`` and ``I_m = <|H|^2>/|E0|^2`` averaged over the
    sphere of radius r.  (STRATIFY's ``I.m`` omits the |eps|/|mu| factor.)
    With ``normalized`` the densities are relative to the incident wave's
    electric (resp. magnetic) energy density in the host.
    """

    r: np.ndarray
    I_e: np.ndarray
    I_m: np.ndarray
    w_e: np.ndarray
    w_m: np.ndarray
    normalized: bool

    @property
    def w_em(self):
        return (self.w_e + self.w_m) / 2 if self.normalized else self.w_e + self.w_m


def _intensities(sol: Solution, w: int, r):
    l = sol.l
    fe_m, fe_0, fe_p = radial_functions(sol, w, r, "e", orders=(-1, 0, 1), with_incident=False)[0]
    (fm_m, fm_0, fm_p), x, shell = radial_functions(sol, w, r, "m", orders=(-1, 0, 1), with_incident=False)

    def a2(v):
        return np.abs(np.nan_to_num(v)) ** 2

    I_e = np.sum((2 * l + 1) * a2(fm_0) + (l + 1) * a2(fe_m) + l * a2(fe_p), 1) / 2
    I_m = np.sum((2 * l + 1) * a2(fe_0) + (l + 1) * a2(fm_m) + l * a2(fm_p), 1) / 2
    eps = sol.ref[w] ** 2 / sol.mu[w]
    epsmu = np.abs(eps) / np.abs(sol.mu[w])
    return I_e, I_m * epsmu[shell], shell


def energy_density(sol: Solution, r, G=None, w: int = 0, normalize=True) -> EnergyDensity:
    """Energy density and averaged intensities at radii ``r``.

    ``G = (G_e, G_m)`` per shell (default: non-dispersive, Eq. 22).  Points on
    an interface belong to the outer shell; r = 0 is evaluated at 1e-9 r_1.
    """
    r = np.atleast_1d(np.asarray(r, float))
    r = np.maximum(r, 1e-9 * sol.rad[0])
    ge, gm = G if G is not None else _default_g(sol, w)
    I_e, I_m, shell = _intensities(sol, w, r)
    w_e = np.pi * np.asarray(ge)[shell] * I_e
    w_m = np.pi * np.asarray(gm)[shell] * I_m
    if normalize:
        eh = (sol.ref[w, -1] ** 2 / sol.mu[w, -1]).real
        w_e, w_m = w_e / (np.pi * eh), w_m / (np.pi * eh)
    return EnergyDensity(r=r, I_e=I_e, I_m=I_m, w_e=w_e, w_m=w_m, normalized=normalize)


# ------------------------------------------------------------- total energy


@dataclass
class ShellEnergy:
    """Electric / magnetic energy in each shell 1..N (arrays of length N).

    Normalised (default): relative to the incident wave's energy in the same
    volume, ``em = (e + m)/2``.  Otherwise |E0|^2 x length^3 in STRATIFY's
    units, ``em = e + m``.  ``method`` records how each shell was integrated.
    """

    e: np.ndarray
    m: np.ndarray
    em: np.ndarray
    normalized: bool
    method: list


def _boundary_f(sol, w, s, rr, pol):
    """f_{l+o}(k_s rr), o = -1..2, with plane-wave (A_{N+1} = 1) coefficients."""
    l = sol.l
    x = sol.k[w, s] * rr
    lp, lx = log_riccati(np.array([x]), l.size + 2)
    la, lb = sol.logA[pol][s, w], sol.logB[pol][s, w]
    with np.errstate(all="ignore"):
        return {o: np.nan_to_num((np.exp(la + lp[0, l + o]) + np.exp(lb + lx[0, l + o])) / x) for o in (-1, 0, 1, 2)}, x


def _lommel(sol, w, s, rr, lossy):
    """Eq. (27) at one boundary: (electric, magnetic) sums x r^3/(2 den)."""
    l = sol.l
    fe, x = _boundary_f(sol, w, s, rr, "e")
    fm, _ = _boundary_f(sol, w, s, rr, "m")

    def F(f, n):  # F-bar for orders (l+n-1, l+n)
        fa, fb = f[n - 1], f[n]
        if lossy:
            return 2j * np.imag(x * np.conj(fa) * fb)
        nu = l + n - 1
        return x * (np.abs(fa) ** 2 + np.abs(fb) ** 2) - (2 * nu + 1) * np.real(np.conj(fa) * fb)

    e = np.sum((2 * l + 1) * F(fm, 1) + (l + 1) * F(fe, 0) + l * F(fe, 2))
    m = np.sum((2 * l + 1) * F(fe, 1) + (l + 1) * F(fm, 0) + l * F(fm, 2))
    den = (x**2 - np.conj(x) ** 2) if lossy else 2 * x
    return 0.5 * rr**3 * e / den, 0.5 * rr**3 * m / den


def _quadrature(sol, w, s, r_in, r_out, n=None):
    """int <|E|^2> r^2 dr and int <|H|^2> r^2 dr over shell s (Gauss-Legendre)."""
    L = sol.l.size
    n = n or int(max(64, L + 48, 6 * abs(sol.k[w, s]) * (r_out - r_in)))
    t, wt = np.polynomial.legendre.leggauss(n)
    r = 0.5 * (r_out - r_in) * t + 0.5 * (r_out + r_in)
    wt = 0.5 * (r_out - r_in) * wt
    I_e, I_m, _ = _intensities(sol, w, r)
    return np.sum(wt * I_e * r**2), np.sum(wt * I_m * r**2)


def total_energy(sol: Solution, G=None, w: int = 0, normalize=True, method="auto") -> ShellEnergy:
    """Electric and magnetic energy stored in each shell (Eqs. 21, 27).

    ``method``: ``'auto'`` (Lommel where well conditioned, quadrature for weak
    loss), ``'lommel'`` or ``'quadrature'`` (for cross-checks).
    """
    ge, gm = G if G is not None else _default_g(sol, w)
    rad = sol.rad
    N = rad.size
    eps = sol.ref[w] ** 2 / sol.mu[w]
    epsmu = np.abs(eps) / np.abs(sol.mu[w])
    we, wm, used = np.zeros(N), np.zeros(N), []
    for s in range(N):
        k = sol.k[w, s]
        r_in = rad[s - 1] if s else 0.0
        weak = k.imag != 0 and abs(k.imag) < WEAK_LOSS * abs(k)
        how = method if method != "auto" else ("quadrature" if weak else "lommel")
        if how == "quadrature":
            e, m = _quadrature(sol, w, s, r_in, rad[s])
            we[s], wm[s] = e * ge[s], m * gm[s]  # I_m already includes |eps|/|mu|
        else:
            lossy = k.imag != 0
            up = _lommel(sol, w, s, rad[s], lossy)
            lo = _lommel(sol, w, s, r_in, lossy) if r_in > 0 else (0.0, 0.0)
            we[s] = np.real(up[0] - lo[0]) * ge[s]
            wm[s] = np.real(up[1] - lo[1]) * gm[s] * epsmu[s]
        used.append(how)
    if normalize:
        vol = np.diff(np.r_[0, rad**3])
        eh = (sol.ref[w, -1] ** 2 / sol.mu[w, -1]).real
        we, wm = 3 * we / vol / eh, 3 * wm / vol / eh
        return ShellEnergy(e=we, m=wm, em=(we + wm) / 2, normalized=True, method=used)
    we, wm = np.pi * we, np.pi * wm
    return ShellEnergy(e=we, m=wm, em=we + wm, normalized=False, method=used)
