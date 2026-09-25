"""Near fields for plane-wave illumination (OSAC Eqs. 6-7, 25-26).

Replaces STRATIFY ``field/near_fld.m``.  The incident wave is
``E0 = x_hat exp(i k_h z)`` (|E0| = 1) propagating along +z; magnetic fields
use STRATIFY's Gaussian normalisation, H = -i (n/mu) sum(...), so |H0| = n_h/mu_h.

Differences from near_fld.m:

* the m = +-1 vector harmonics are combined analytically into the
  Bohren-Huffman pi_l, tau_l, regular at both poles (AUDIT.md M6);
* expansion coefficients come from the overflow-free solver and are applied
  in logarithmic form, A_l psi_l(kr) = exp(log A_l + log psi_l(kr)), so
  thin metal shells and high orders are accurate (AUDIT.md M9);
* special functions are evaluated once per unique k*r, and the sum over l is
  a matrix product over chunks of points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .farfield import angular_functions
from .riccati import log_riccati
from .solver import Solution

__all__ = ["NearField", "near_field", "radial_functions"]


@dataclass
class NearField:
    """Complex field components on the requested points (shape of X).
    ``E``/``H`` dicts hold Cartesian (x, y, z) and spherical (r, th, ph) parts."""

    E: dict
    H: dict

    @property
    def intensity_E(self) -> np.ndarray:
        return sum(np.abs(self.E[c]) ** 2 for c in "xyz")

    @property
    def intensity_H(self) -> np.ndarray:
        return sum(np.abs(self.H[c]) ** 2 for c in "xyz")


def _log_incident(l):
    return np.log(1j**l * np.sqrt((2 * l + 1) * np.pi))


def radial_functions(sol: Solution, w: int, r, pol: str, orders=(0,), with_incident=True):
    """f_{l+o}(r) = A_l j_{l+o}(k r) + B_l h_{l+o}(k r) for offsets ``o``.

    Also returns x = k r and the shell index of every point.  Shape of each
    f: ``(len(r), L)``.  Points exactly on an interface belong to the outer
    shell.  r must be > 0.
    """
    r = np.asarray(r, dtype=float)
    l = sol.l
    shell = np.searchsorted(sol.rad, r, side="right")
    k = sol.k[w]
    x = k[shell] * r
    L = l.size
    omax = max(orders)
    lp, lx = log_riccati(x, L + omax)
    la = sol.logA[pol][shell, w]  # (M, L)
    lb = sol.logB[pol][shell, w]
    if with_incident:
        c = _log_incident(l)
        la, lb = la + c, lb + c
    out = []
    with np.errstate(all="ignore"):
        for o in orders:
            v = np.exp(la + lp[:, l + o]) + np.exp(lb + lx[:, l + o])
            out.append(v / x[:, None])
    return out, x, shell


def near_field(sol: Solution, X, Y, Z, w: int = 0, chunk: int = 4096) -> NearField:
    """Electric and magnetic near field at Cartesian points (X, Y, Z).

    ``w`` selects the wavelength of a batched solution.  Points at the origin
    are moved to r = 1e-9 r_1 (the field is continuous there).
    """
    X, Y, Z = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float))
    shape = X.shape
    x, y, z = X.ravel(), Y.ravel(), Z.ravel()
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-9 * sol.rad[0])
    th = np.arctan2(np.hypot(x, y), z)
    ph = np.arctan2(y, x)
    l = sol.l
    nmu = sol.ref[w] / sol.mu[w]

    # special functions on unique radii
    ur, r_idx = np.unique(r, return_inverse=True)
    fields = {}
    for p in ("e", "m"):
        (f_m1, f0), xx, shell_u = radial_functions(sol, w, ur, p, orders=(-1, 0))
        g = f0 / xx[:, None]  # f_l / x
        d = f_m1 - l * g  # (1/x) d(x f_l)/dx = f_{l-1} - l f_l / x
        fields[p] = (f0, g, d)
    shell = shell_u[r_idx]

    ucos, c_idx = np.unique(np.round(np.cos(th), 14), return_inverse=True)
    pi_u, tau_u = angular_functions(l, np.arccos(np.clip(ucos, -1, 1)))
    mepl = 1j * np.sqrt((2 * l + 1) / (4 * np.pi)) / (l * (l + 1))
    me2, op2 = 2 * mepl, 2 * mepl * l * (l + 1)  # factors of the combined harmonics

    E = {c: np.empty(r.size, complex) for c in ("r", "th", "ph")}
    H = {c: np.empty(r.size, complex) for c in ("r", "th", "ph")}
    fe, ge, de = fields["e"]
    fm, gm, dm = fields["m"]
    for s in range(0, r.size, chunk):
        sl = slice(s, s + chunk)
        ri, ci = r_idx[sl], c_idx[sl]
        p_, t_ = pi_u[:, ci].T, tau_u[:, ci].T  # (M, L)
        cph, sph, st = np.cos(ph[sl]), np.sin(ph[sl]), np.sin(th[sl])
        with np.errstate(invalid="ignore"):
            a_de, a_fm, a_ge = de[ri] * me2, fm[ri] * me2, ge[ri] * op2
            a_fe, a_dm, a_gm = fe[ri] * me2, dm[ri] * me2, gm[ri] * op2
            E["ph"][sl] = sph * np.sum(a_de * p_ + 1j * a_fm * t_, 1)
            E["th"][sl] = -cph * np.sum(a_de * t_ + 1j * a_fm * p_, 1)
            E["r"][sl] = -st * cph * np.sum(a_ge * p_, 1)
            H["ph"][sl] = cph * np.sum(a_fe * t_ - 1j * a_dm * p_, 1)
            H["th"][sl] = sph * np.sum(a_fe * p_ - 1j * a_dm * t_, 1)
            H["r"][sl] = -1j * st * sph * np.sum(a_gm * p_, 1)
    fac = -1j * nmu[shell]
    for c in H:
        H[c] = H[c] * fac

    ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)

    def cart(F):
        return {
            "x": (-sp * F["ph"] + ct * cp * F["th"] + st * cp * F["r"]).reshape(shape),
            "y": (cp * F["ph"] + ct * sp * F["th"] + st * sp * F["r"]).reshape(shape),
            "z": (-st * F["th"] + ct * F["r"]).reshape(shape),
            "r": F["r"].reshape(shape),
            "th": F["th"].reshape(shape),
            "ph": F["ph"].reshape(shape),
        }

    return NearField(E=cart(E), H=cart(H))
