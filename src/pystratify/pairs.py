"""Two-point Green's dyadic of a multilayered sphere: energy transfer and dipole-dipole coupling.

For two points r1, r2 in the same lossless shell the scattered part of the radial Green's function of
order l and polarization p, with radial functionals F at r1 and G at r2 (F, G = a u + b u'), is for
r1 <= r2

    psi_1 xi_2 [sigma_2 F(A_1) G(A_2) + rho_1 sigma_2 F(B_1) G(A_2) + rho_1 F(B_1) G(B_2)
                + R S F(A_1) G(B_2)] / (1 - R S),

with psi_1 xi_2 = P(x_2) (x_1/x_2)^(l+1) jbar(x_1)/jbar(x_2) an explicit geometric factor times
quantities of order unity (and r1 <-> r2, F <-> G otherwise); R S = rho sigma at either point.  At
r1 = r2 it reduces to the forms S, S^m, S^d of :mod:`pystratify.normalized`.  The dyadic follows
from the source functionals of an electric dipole at r1 (local frame, r1 on z) and the M and N
fields at r2 of the axial families m = 0, 1e, 1o, with the conventions of :mod:`pystratify.rates`;
the free part is added in closed form.  Normalization: curl curl G - k^2 G = delta I with k the
wavenumber of the shell, so that (6 pi / k) u.G_s(r, r).u = i g of the decay rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .convergence import orders_needed, tail_estimate, truncation_order
from .emission import _local_frame
from .normalized import L_CAP, _Sweep
from .rates import _legendre_derivatives, _tetm_functionals
from .solver import TE, TM

__all__ = ["GreenDyadic", "green_dyadic"]


@dataclass(frozen=True)
class GreenDyadic:
    """Green's dyadic G(r2, r1) (3, 3), global axes: ``total`` = ``free`` + ``scattered``, in inverse length
    units of the radii; the field of an electric dipole p at r1 is E(r2) = (omega^2 mu_0 mu_d) G p (SI).
    ``k`` is the wavenumber of the points' shell, ``orders`` the multipoles summed."""

    r1: np.ndarray
    r2: np.ndarray
    total: np.ndarray
    scattered: np.ndarray
    free: np.ndarray
    k: float
    orders: int
    converged: bool


def _free_dyadic(k, r1, r2):
    """Closed-form free dyadic of a homogeneous medium of wavenumber k."""
    d = np.asarray(r2, float) - np.asarray(r1, float)
    R = np.linalg.norm(d)
    u = d / R
    kr = k * R
    a = 1 + 1j / kr - 1 / kr**2
    b = 1 + 3j / kr - 3 / kr**2
    return (a * np.eye(3) - b * np.outer(u, u)) * np.exp(1j * kr) / (4 * np.pi * R)


def _angular(L, gamma):
    """P_l, P_l^1, tau^0, pi^1, tau^1 at cos(gamma), l = 1..L (Bohren-Huffman, no Condon-Shortley phase)."""
    ct, st = np.cos(gamma), np.sin(gamma)
    p = np.zeros(L + 2)
    p[0], p[1] = 1.0, ct
    for l in range(1, L + 1):
        p[l + 1] = ((2 * l + 1) * ct * p[l] - l * p[l - 1]) / (l + 1)
    d1, d2, _ = (v[1:, 0] for v in _legendre_derivatives(L, np.array([ct])))
    return p[1 : L + 1], st * d1, -st * d1, d1, ct * d1 - st**2 * d2


def _pair_terms(radii, n, mu, k, L, a, b, gamma, series_free=False):
    """Per-order contributions (L, 3, 3) to (6 pi/(i k)) G(r2, r1), rows r, theta, phi at r2 (local frame,
    r2 at polar angle gamma, azimuth 0), columns local x, y, z dipoles at r1; scattered part, or the free
    part as a series with ``series_free``."""
    l = np.arange(1, L + 1)
    ll = (l * (l + 1)).astype(float)
    t = _Sweep(radii, n, mu, k, L).at(np.array([a, b]))
    x1, x2 = t["x"][0], t["x"][1]
    d = int(t["shell"][0])
    kd, zd = k[d].real, (mu[d] / n[d]).real
    cv, cd = _tetm_functionals(np.array([x1]), np.array([kd]), zd, np.c_[np.eye(3), np.zeros((3, 3))][None], False, l)
    cv, cd = cv[0], cd[0]  # (3 sources, L, 3 families, 2 polarizations)
    pick = lambda name, i: np.moveaxis(t[name][:, i], 0, -1)[None, :, None, :]  # noqa: E731  -> (1, L, 1, 2)
    rho1, sig1, rho2, sig2 = pick("rho", 0), pick("sigma", 0), pick("rho", 1), pick("sigma", 1)
    rp1, rx1, rp2, rx2 = pick("rp", 0), pick("rx", 0), pick("rp", 1), pick("rx", 1)
    P1, P2, lj1, lj2 = pick("P", 0), pick("P", 1), pick("lj", 0), pick("lj", 1)
    rs = rho1 * sig1
    f_psi, f_xi = (cv - cd * rp1) / x1, (cv - cd * rx1) / x1  # source functionals per unit u, applied to psi, xi
    up = ((l + 1) / x2)[None, :, None, None]
    obs = {"val": (1.0, 1.0), "der": (up - rp2, up - rx2)}  # G applied to psi, xi at r2
    out = {}
    with np.errstate(all="ignore"):
        if a <= b:
            pref = P2 * np.exp(((l + 1) * np.log(x1 / x2))[None, :, None, None] + lj1 - lj2)
        else:
            pref = P1 * np.exp(((l + 1) * np.log(x2 / x1))[None, :, None, None] + lj2 - lj1)
        for key, (g_psi, g_xi) in obs.items():
            if series_free:
                out[key] = pref * (f_psi * g_xi if a <= b else g_psi * f_xi)
            elif a <= b:
                br = sig2 * f_psi * g_psi + rho1 * sig2 * f_xi * g_psi + rho1 * f_xi * g_xi + rs * f_psi * g_xi
                out[key] = pref * br / (1 - rs)
            else:
                br = sig1 * g_psi * f_psi + rho2 * sig1 * g_xi * f_psi + rho2 * g_xi * f_xi + rs * g_psi * f_xi
                out[key] = pref * br / (1 - rs)
    tai0, tai1 = (2 * l + 1) / ll, 2 * (2 * l + 1) / ll**2
    P_l, P1_l, tau0, pi1, tau1 = _angular(L, gamma)
    val, der = out["val"], out["der"]  # (3 sources, L, 3 families, 2)
    terms = np.zeros((L, 3, 3), dtype=complex)
    for j in range(3):
        e_r = tai0 * val[j, :, 0, TM] * ll / x2**2 * P_l + tai1 * val[j, :, 1, TM] * ll / x2**2 * P1_l
        e_t = (tai0 * der[j, :, 0, TM] * tau0 + tai1 * der[j, :, 1, TM] * tau1 + tai1 * val[j, :, 2, TE] * pi1) / x2
        e_p = (-tai0 * val[j, :, 0, TE] * tau0 - tai1 * val[j, :, 1, TE] * tau1 + tai1 * der[j, :, 2, TM] * pi1) / x2
        terms[:, 0, j], terms[:, 1, j], terms[:, 2, j] = 1.5 * e_r, 1.5 * e_t, 1.5 * e_p
    return terms


def green_dyadic(radii, n, wavelength, r1, r2, mu=None, l_max=None, tol=1e-12, l_cap=L_CAP) -> GreenDyadic:
    """Green's dyadic G(r2, r1) between two points (Cartesian, length units of ``radii``) in the same
    lossless shell, from the normalized formulation; geometry and materials as in
    :func:`~pystratify.decay_rates`.  ``l_max=None`` sums until the remainder meets ``tol`` relative to
    the largest element (doubling up to ``l_cap``)."""
    radii = np.atleast_1d(np.asarray(radii, float))
    n = np.atleast_1d(np.asarray(n, complex))
    mu = np.ones(n.size, complex) if mu is None else np.atleast_1d(np.asarray(mu, complex))
    r1, r2 = np.asarray(r1, float).ravel(), np.asarray(r2, float).ravel()
    a, b = np.linalg.norm(r1), np.linalg.norm(r2)
    if r1.shape != (3,) or r2.shape != (3,) or not (a > 0 and b > 0) or np.allclose(r1, r2):
        raise ValueError("r1 and r2 must be distinct, nonzero 3-vectors")
    d1, d2 = (int(np.searchsorted(radii, v, side="right")) for v in (a, b))
    if d1 != d2:
        raise NotImplementedError("points in different shells: not yet in the normalized formulation")
    if n[d1].imag != 0 or mu[d1].imag != 0:
        raise ValueError("the points' shell must be lossless")
    k = 2 * np.pi * n / wavelength
    frame = _local_frame(r1)
    v = frame @ r2
    phi = np.arctan2(v[1], v[0])
    turn = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1.0]])
    frame = turn @ frame  # local axes with r1 on z and r2 in the xz-plane
    gamma = float(np.arccos(np.clip((frame @ r2)[2] / b, -1, 1)))
    if l_max is not None:
        L = int(l_max)
    else:
        q = max(np.max(np.minimum(radii / p, p / radii)) for p in (a, b))
        L = int(
            min(l_cap, max(32, truncation_order(radii[-1], n[-1], wavelength, "near"), orders_needed(q, tol, l_cap)))
        )
    while True:
        terms = _pair_terms(radii, n, mu, k, L, a, b, gamma)
        scale = np.max(np.abs(terms.sum(axis=0)))
        tail = tail_estimate(np.abs(terms[-4:]).reshape(4, -1))
        ok = bool(np.all(tail <= tol * max(scale, np.finfo(float).tiny)))
        if l_max is not None or ok or L >= l_cap:
            break
        L = min(2 * L, l_cap)
    kd = k[d1].real
    ct, st = np.cos(gamma), np.sin(gamma)
    basis = np.array([[st, ct, 0.0], [0.0, 0.0, 1.0], [ct, -st, 0.0]])  # columns r, theta, phi at r2 (local)
    g_local = (1j * kd / (6 * np.pi)) * basis @ terms.sum(axis=0)
    scattered = frame.T @ g_local @ frame
    free = _free_dyadic(kd, r1, r2)
    return GreenDyadic(
        r1=r1, r2=r2, total=free + scattered, scattered=scattered, free=free, k=kd, orders=L, converged=ok
    )
