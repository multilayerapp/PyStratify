"""Multilayered spheres with hydrodynamic (nonlocal) metal shells.

Basis of shell j for multipole order l, radius r, x = k_j r (Riccati functions
u = psi_l, xi_l) and, in a hydrodynamic shell, x_L = k_L r (spherical
z = j_l, h_l^(1)); traces [E_t, H_t, J_n, M] in the conventions of
:mod:`pystratify.references` (E_t coefficient of r x X for TM, of X for TE; H_t
that of r x H):

    TM transverse   E_t = u'/x,           H_t = -i y u/x,   E_r = i s u/x^2,   s = sqrt(l(l+1))
    TM longitudinal E_t = -i s z(x_L)/r,  H_t = 0,          E_r = k_L z'(x_L), Phi = z(x_L)
    TE              E_t = u/x,            H_t = i y u'/x                       (no longitudinal wave)

with y = n/mu, J_n = (eps_T - eps_b) E_r^T - eps_b E_r^L and M = eps_T/(eps_b - eps_T) Phi.
Hydrodynamic shells only modify TM (the longitudinal field has no magnetic field and
TE has no radial E).  See :mod:`pystratify.nonlocal_sweep` for the recursion and
:mod:`pystratify.hydrodynamic` for the material model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .convergence import truncation_order
from .hydrodynamic import Hydrodynamic
from .nonlocal_sweep import ChannelSweep, Traces, channel_sweep
from .riccati import log_riccati
from .solver import TE, TM, Solution, _batch

__all__ = ["NonlocalSphere", "solve_nonlocal_sphere", "hydrodynamic_regions"]


def hydrodynamic_regions(hydrodynamic, regions, host_allowed=False):
    """Validate {region: Hydrodynamic}; returns a tuple of length ``regions`` (None where local)."""
    out = [None] * regions
    if hydrodynamic is None:
        return tuple(out)
    if not hasattr(hydrodynamic, "items"):
        raise ValueError("hydrodynamic must map region indices to Hydrodynamic objects")
    for j, model in hydrodynamic.items():
        if not (isinstance(j, (int, np.integer)) and 0 <= j < regions):
            raise ValueError(f"hydrodynamic region index {j!r} not in 0..{regions - 1}")
        if not host_allowed and j == regions - 1:
            raise ValueError("the exterior region cannot be hydrodynamic")
        if not isinstance(model, Hydrodynamic):
            raise ValueError("hydrodynamic values must be Hydrodynamic objects")
        out[j] = model
    return tuple(out)


def _riccati_parts(log_u, orders, x):
    """Normalised (value, derivative) of a Riccati or spherical function at ``orders`` and its log scale.

    log_u: (..., nmax + 1) logs of u_n(x) for n = 0..nmax; x broadcastable to (...).
    """
    x = np.asarray(x, complex)[..., None]
    with np.errstate(all="ignore"):
        d = np.exp(log_u[..., orders - 1] - log_u[..., orders]) - orders / x  # u'/u
        scale = np.maximum(1.0, np.abs(d))
    return 1 / scale, d / scale, log_u[..., orders] + np.log(scale)


class _SphereTraces:
    def __init__(self, radii, n, mu, wavelength, orders, hydro, polarization):
        self.radii, self.orders, self.polarization = radii, orders, polarization
        self.k0 = 2 * np.pi / wavelength  # (W,)
        self.n, self.mu = n, mu  # (W, N + 1)
        self.k = self.k0[:, None] * n
        self.y = n / mu
        self.hydro = hydro if polarization == TM else (None,) * n.shape[1]
        self.s = np.sqrt(orders * (orders + 1.0))
        self.eps_T, self.eps_b, self.kL = {}, {}, {}
        for j, model in enumerate(self.hydro):
            if model is None:
                continue
            eps_T = n[:, j] ** 2
            self.eps_T[j] = eps_T
            self.eps_b[j] = model.background(wavelength, eps_T)
            self.kL[j] = model.longitudinal_wavenumber(wavelength, eps_T)
        self._cache = {}

    def _logs(self, z):
        key = z.tobytes()
        if key not in self._cache:
            self._cache[key] = log_riccati(z, int(self.orders[-1]) + 1)
        return self._cache[key]

    def __call__(self, region, interface):
        """Traces of ``region`` at ``radii[interface]``.  The continuity rows are reduced by the
        quasi-static admittance of the outer region of that interface (the same for both sides):
        TM stores E_t - i(l+1)/(k0 r eps_ref) H_t, TE stores H_t - i(l+1)/(k0 r mu_ref) E_t, each
        formed from psi_(l+1)/psi_l and xi_(l-1)/xi_l and the exact contrast, never by subtraction."""
        r = self.radii[interface]
        ref = interface + 1
        l = self.orders
        x = self.k[:, region] * r  # (W,)
        lp, lx = self._logs(x)
        W, L = x.size, l.size
        hydro = self.hydro[region] is not None
        c = 2 if hydro else 1
        F, G = np.zeros((W, L, 4, c), complex), np.zeros((W, L, 4, c), complex)
        LF, LG = np.zeros((W, L, c), complex), np.zeros((W, L, c), complex)
        xx, y = x[:, None], self.y[:, region][:, None]
        k0r = (self.k0 * r)[:, None]
        mu, mu_ref = self.mu[:, region][:, None], self.mu[:, ref][:, None]
        eps = self.n[:, region][:, None] ** 2 / mu
        eps_ref = self.n[:, ref][:, None] ** 2 / mu_ref
        with np.errstate(all="ignore"):
            ratio_psi = np.exp(lp[..., l + 1] - lp[..., l])  # psi_(l+1) / psi_l
            ratio_xi = np.exp(lx[..., l - 1] - lx[..., l])  # xi_(l-1) / xi_l
        U = np.zeros((W, L, 4, 4), complex)
        U[..., 0, 0] = U[..., 1, 1] = U[..., 2, 2] = U[..., 3, 3] = 1
        for out, logs, log_u, kind in ((F, LF, lp, "psi"), (G, LG, lx, "xi")):
            v, d, lg = _riccati_parts(log_u, l, x)
            logs[..., 0] = lg
            if self.polarization == TM:
                if kind == "psi":
                    reduced = (l + 1) * (eps_ref - eps) / (eps_ref * xx) - ratio_psi
                else:
                    reduced = ratio_xi - l / xx - (l + 1) * (eps / eps_ref) / xx
                out[..., 0, 0] = v * reduced / xx  # E_t - c' H_t, c' = i(l+1)/(k0 r eps_ref)
                out[..., 1, 0] = -1j * y * v / xx
                if hydro:
                    e_r = 1j * self.s * v / xx**2
                    out[..., 2, 0] = (self.eps_T[region] - self.eps_b[region])[:, None] * e_r
            else:
                if kind == "psi":
                    reduced = (l + 1) / k0r * (1 / mu - 1 / mu_ref) - y * ratio_psi
                else:
                    reduced = y * ratio_xi - l / (k0r * mu) - (l + 1) / (k0r * mu_ref)
                out[..., 0, 0] = v / xx
                out[..., 1, 0] = 1j * v * reduced / xx  # H_t - c E_t, c = i(l+1)/(k0 r mu_ref)
        if self.polarization == TM:
            U[..., 0, 1] = 1j * (l + 1) / (k0r * eps_ref)
        else:
            U[..., 1, 0] = 1j * (l + 1) / (k0r * mu_ref)
        if hydro:
            kL = self.kL[region]
            xL = kL * r
            lpL, lxL = self._logs(xL)
            eps_T, eps_b = self.eps_T[region][:, None], self.eps_b[region][:, None]
            for out, logs, log_u in ((F, LF, lpL), (G, LG, lxL)):
                # spherical z = u / x_L: log z = log u - log x_L and z'/z = u'/u - 1/x_L
                with np.errstate(all="ignore"):
                    du = np.exp(log_u[..., l - 1] - log_u[..., l]) - l / xL[:, None]
                    dz = du - 1 / xL[:, None]
                    scale = np.maximum(1.0, np.abs(dz))
                v, d = 1 / scale, dz / scale
                logs[..., 1] = log_u[..., l] - np.log(xL)[:, None] + np.log(scale)
                out[..., 0, 1] = -1j * self.s * v / r  # no magnetic field: the reduction leaves E_t
                out[..., 2, 1] = -eps_b * kL[:, None] * d
                out[..., 3, 1] = eps_T / (eps_b - eps_T) * v
        return Traces(F, G, LF, LG, U)


@dataclass(frozen=True)
class NonlocalSphere:
    """Solution of a multilayered sphere with hydrodynamic shells.

    ``solution`` is a :class:`~pystratify.Solution` whose T-matrix is exact; its amplitude
    arrays hold the transverse fields (the longitudinal field lives in ``tm``).  The outgoing-
    solution arrays (``log_b_out``, ``log_s``) are not unique inside a hydrodynamic shell and are
    NaN there.  ``tm``, ``te``: the channel sweeps of each polarisation; ``hydrodynamic``: the
    model per region (None where local)."""

    solution: Solution
    tm: ChannelSweep
    te: ChannelSweep
    hydrodynamic: tuple
    traces: tuple

    @property
    def t(self):
        return self.solution.t


def _amplitude_logs(sweep, traces_pair, N):
    """log_a, log_b (regular solution, A = 1 in the host) of the transverse channel, (W, L) per shell."""
    W, L = sweep.R_out[-1].shape[:2]
    host = sweep.outer[N - 1]
    regular = sweep.regular_amplitudes(np.ones((W, L, 1, 1), complex))
    log_a = np.full((N + 1, W, L), -np.inf + 0j)
    log_b = np.full((N + 1, W, L), -np.inf + 0j)
    ref = host.LF[..., 0]  # A_host = 1: physical A = A^ / nu_f
    for j in range(N + 1):
        inner, outer = regular[j]
        if j == N:
            a, b, lg = inner
            lf, lgg = host.LF[..., 0], host.LG[..., 0]
        else:
            a, b, lg = outer
            lf, lgg = sweep.inner[j].LF[..., 0], sweep.inner[j].LG[..., 0]
        with np.errstate(divide="ignore"):
            log_a[j] = np.log(a[..., 0, 0]) + lg[..., 0] - lf + ref
            log_b[j] = np.log(b[..., 0, 0]) + lg[..., 0] - lgg + ref
    log_b[0] = -np.inf
    return log_a, log_b


def _outgoing_logs(sweep, N, hydro):
    """log_b_out, log_s (outgoing solution, B = 1 in the host) of the transverse channel in local
    shells; NaN in hydrodynamic shells, where the outgoing solution is not unique."""
    W, L = sweep.R_out[-1].shape[:2]
    log_bo = np.full((N + 1, W, L), np.nan + 0j)
    log_s = np.full((N + 1, W, L), np.nan + 0j)
    log_bo[N], log_s[N] = 0, -np.inf
    for j in range(N):
        if hydro[j] is not None:
            continue
        # B^_j(r_j) = 1 continued outwards: B_host / B_j, then invert
        chain = sweep.outgoing_amplitudes(j, np.ones((W, L, 1, 1), complex))
        _, b_host, lg = chain[N][0]
        host = sweep.outer[N - 1]
        mine = sweep.inner[j]
        with np.errstate(divide="ignore"):
            log_host = np.log(b_host[..., 0, 0]) + lg[..., 0] - host.LG[..., 0]  # B_host for B^_j(r_j) = 1
        log_bo[j] = -mine.LG[..., 0] - log_host
        with np.errstate(divide="ignore"):
            log_s[j] = np.log(sweep.S_in[j][..., 0, 0]) + mine.LG[..., 0] - mine.LF[..., 0]
    return log_bo, log_s


def solve_nonlocal_sphere(radii, n, wavelength, hydrodynamic, mu=None, l_max=None, regime="far") -> NonlocalSphere:
    """Solve a multilayered sphere with hydrodynamic regions for orders l = 1..l_max.

    ``radii`` (N,), ``n`` (N + 1,) or (W, N + 1) host last, ``wavelength`` scalar or (W,), as for
    :func:`~pystratify.solve`; ``hydrodynamic``: {region: :class:`~pystratify.Hydrodynamic`},
    region 0 the core, the host excluded.  ``n`` of a hydrodynamic region is its transverse
    (local) index.  Vectorised over wavelengths and orders.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.ndim != 1 or radii.size == 0 or radii[0] <= 0 or np.any(np.diff(radii) <= 0) or not np.all(np.isfinite(radii)):
        raise ValueError("radii must be positive, finite and strictly increasing")
    N = radii.size
    n, mu, wavelength = _batch(n, mu, wavelength, N + 1)
    hydro = hydrodynamic_regions(hydrodynamic, N + 1)
    for j, model in enumerate(hydro):
        if model is not None and np.any(mu[:, j] != 1):
            raise ValueError("hydrodynamic regions must be nonmagnetic (mu = 1)")
    if l_max is None:
        l_max = max(truncation_order(radii[-1], abs(v), lam, regime) for v, lam in zip(n[:, -1], wavelength))
    orders = np.arange(1, int(l_max) + 1)
    flags = [m is not None for m in hydro]
    tm_traces = _SphereTraces(radii, n, mu, wavelength, orders, hydro, TM)
    te_traces = _SphereTraces(radii, n, mu, wavelength, orders, hydro, TE)
    tm = channel_sweep(tm_traces, radii, 1, flags)
    te = channel_sweep(te_traces, radii, 1, [False] * (N + 1))
    log_t = np.empty((2, wavelength.size, orders.size), complex)
    log_a = np.empty((2, N + 1, wavelength.size, orders.size), complex)
    log_b, log_bo, log_s = np.empty_like(log_a), np.empty_like(log_a), np.empty_like(log_a)
    for p, sweep, h in ((TM, tm, hydro), (TE, te, (None,) * (N + 1))):
        host = sweep.outer[N - 1]
        with np.errstate(divide="ignore"):
            log_t[p] = np.log(sweep.R_out[N - 1][..., 0, 0]) + host.LF[..., 0] - host.LG[..., 0]
        log_a[p], log_b[p] = _amplitude_logs(sweep, None, N)
        log_bo[p], log_s[p] = _outgoing_logs(sweep, N, h)
    solution = Solution(radii=radii, n=n, mu=mu, wavelength=wavelength, orders=orders, log_t=log_t,
                        log_a=log_a, log_b=log_b, log_r=log_b - log_a, log_b_out=log_bo, log_s=log_s)
    return NonlocalSphere(solution, tm, te, hydro, (tm_traces, te_traces))
