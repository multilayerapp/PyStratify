"""Concentric infinite cylinders with hydrodynamic (nonlocal) metal shells, at any axial wavenumber.

Fields of order m and axial wavenumber beta, ~ exp(i m phi + i beta z), in region j with
q = sqrt(k^2 - beta^2) and Z_m(q rho) (J regular, H^(1) outgoing), in the vector basis of
:mod:`pystratify.cylindrical`:

    N = (i beta Z'/k, -beta m Z/(k q rho), q Z/k),   M = (i m Z/(q rho), -Z', 0),
    E = a_N N + a_M M,   H = -i (n/mu)(a_N M + a_M N),

and in a hydrodynamic region the longitudinal wave with q_L = sqrt(k_L^2 - beta^2),

    L = grad[Z_m(q_L rho) e^(i m phi + i beta z)] = (q_L Z', i m Z/rho, i beta Z),   Phi = Z.

At beta != 0 the longitudinal wave enters E_z and E_phi and couples to both N and M (including
m = 0); at beta = 0 it couples to M only (E perpendicular to the axis) and not at m = 0.  Traces
[E_z, E_phi, H_z, H_phi, J_n, M] with J_n = (eps_T - eps_b) E_rho^T - eps_b E_rho^L and
M = eps_T/(eps_b - eps_T) Phi; see :mod:`pystratify.nonlocal_sweep`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cylindrical import outgoing_root
from .nonlocal_sphere import hydrodynamic_regions
from .nonlocal_sweep import ChannelSweep, Traces, channel_sweep
from .special import cylinder_logs

__all__ = ["NonlocalCylinder", "solve_nonlocal_cylinder"]


class _CylinderTraces:
    def __init__(self, radii, n, mu, wavelength, beta, orders, hydro):
        self.radii, self.orders, self.beta = radii, orders, complex(beta)
        self.k0 = 2 * np.pi / wavelength
        self.n, self.mu, self.y = n, mu, n / mu
        self.k = self.k0 * n
        self.q = outgoing_root(self.k**2 - self.beta**2)
        self.hydro = hydro
        self.eps_T, self.eps_b, self.qL = {}, {}, {}
        for j, model in enumerate(hydro):
            if model is None:
                continue
            eps_T = complex(n[j] ** 2)
            self.eps_T[j] = eps_T
            self.eps_b[j] = complex(model.background(wavelength, eps_T))
            kL = complex(model.longitudinal_wavenumber(wavelength, eps_T))
            self.qL[j] = complex(outgoing_root(kL**2 - self.beta**2))

    def _functions(self, q, r):
        """Per order: normalised (value, derivative) of J and H at q r, their log scales, and
        J_(|m|+1)/J_|m| (the small ratio, for the reduced rows), like cylinder_pair."""
        mm = np.abs(self.orders)
        lj, lh, dj, dh = cylinder_logs(q * r, int(mm.max()) + 1)
        sign = np.where((self.orders < 0) & (mm % 2 == 1), 1j * np.pi, 0j)
        out = []
        for log_f, deriv in ((lj[mm] + sign, dj[mm]), (lh[mm] + sign, dh[mm])):
            scale = np.maximum(1, np.abs(deriv))
            out.append((1 / scale, deriv / scale, log_f + np.log(scale)))
        with np.errstate(over="ignore", under="ignore"):
            ratio = np.exp(lj[mm + 1] - lj[mm])
        return out, ratio

    def __call__(self, region, interface):
        """Traces of ``region`` at ``radii[interface]``.  Two continuity rows are stored reduced by the
        quasi-static admittances of the outer region of the interface (the same on both sides):
        H_phi - c E_z (static N wave) and E_phi - c' H_z (static M wave), with
        c = i y_r |m| k_r / (q_r^2 rho) and c' = -i |m| k_r / (y_r q_r^2 rho); the regular
        functions' reduced entries are formed from J_(|m|+1)/J_|m| and the exact contrast."""
        r = self.radii[interface]
        ref = interface + 1
        m = self.orders.astype(float)
        am = np.abs(m)
        hydro = self.hydro[region] is not None
        c = 3 if hydro else 2
        L = m.size
        F, G = np.zeros((L, 6, c), complex), np.zeros((L, 6, c), complex)
        LF, LG = np.zeros((L, c), complex), np.zeros((L, c), complex)
        q, k, y, b = self.q[region], self.k[region], self.y[region], self.beta
        qr, kr, yr = self.q[ref], self.k[ref], self.y[ref]
        n, nr, mu, mur = self.n[region], self.n[ref], self.mu[region], self.mu[ref]
        k0 = self.k0
        eps, eps_r = n * n / mu, nr * nr / mur
        d_eps = (nr - n) * (nr + n) / mu if mu == mur else eps_r - eps  # eps_r - eps
        c_n = 1j * yr * am * kr / (qr**2 * r)  # H_phi - c_n E_z
        c_m = -1j * am * kr / (yr * qr**2 * r)  # E_phi - c_m H_z
        # exact contrasts: kappa_N = y/q - y_r k_r q/(q_r^2 k), kappa_M = 1/q - k_r y q/(y_r q_r^2 k)
        num_n = k0 * (eps * eps_r * k0**2 * (mur - mu) + b**2 * d_eps)
        kappa_n = num_n / (q * qr**2 * k)
        num_m = n * nr * k0 * (k0**2 * d_eps - b**2 * (1 / mur - 1 / mu))
        kappa_m = num_m / (q * yr * qr**2 * k)
        jn = (self.eps_T[region] - self.eps_b[region]) if hydro else 0.0
        (fj, fh), ratio = self._functions(q, r)
        for out, logs, (v, d, lg), regular in ((F, LF, fj, True), (G, LG, fh, False)):
            logs[:, 0] = logs[:, 1] = lg
            # N column
            out[:, 0, 0] = q * v / k
            out[:, 1, 0] = -b * m * v / (k * q * r)
            out[:, 4, 0] = jn * 1j * b * d / k
            if regular:
                out[:, 3, 0] = 1j * v * (am / r * kappa_n - y * ratio)
            else:
                out[:, 3, 0] = 1j * y * d - c_n * q / k * v
            # M column
            if regular:
                out[:, 1, 1] = v * (ratio - am / r * kappa_m)
            else:
                out[:, 1, 1] = -d + 1j * c_m * y * q / k * v
            out[:, 2, 1] = -1j * y * q * v / k
            out[:, 3, 1] = 1j * y * b * m * v / (k * q * r)
            out[:, 4, 1] = jn * 1j * m * v / (q * r)
        if hydro:
            qL, eps_T, eps_b = self.qL[region], self.eps_T[region], self.eps_b[region]
            (fj, fh), _ = self._functions(qL, r)
            for out, logs, (v, d, lg) in ((F, LF, fj), (G, LG, fh)):
                logs[:, 2] = lg
                out[:, 0, 2] = 1j * b * v
                out[:, 1, 2] = 1j * m * v / r
                out[:, 3, 2] = -c_n * 1j * b * v
                out[:, 4, 2] = -eps_b * qL * d
                out[:, 5, 2] = eps_T / (eps_b - eps_T) * v
        U = np.zeros((L, 6, 6), complex)
        U[:, range(6), range(6)] = 1
        U[:, 1, 2] = c_m
        U[:, 3, 0] = c_n
        return Traces(F, G, LF, LG, U)


@dataclass(frozen=True)
class NonlocalCylinder:
    """Solution for one axial wavenumber.  ``t`` (orders, 2, 2): host T blocks [scattered (N, M),
    incident (N, M)] as in :class:`~pystratify.CylinderSolution`, so :func:`~pystratify.cross_widths`
    and :func:`~pystratify.cylinder_pattern` accept it; ``sweep`` holds every response."""

    radii: np.ndarray
    n: np.ndarray
    mu: np.ndarray
    wavelength: float
    beta: complex
    orders: np.ndarray
    hydrodynamic: tuple
    sweep: ChannelSweep
    traces: _CylinderTraces
    t: np.ndarray


def solve_nonlocal_cylinder(radii, n, wavelength, hydrodynamic, *, mu=None, beta=0, m_max=None) -> NonlocalCylinder:
    """Concentric cylinders with hydrodynamic regions ({region: Hydrodynamic}, region 0 the core,
    the exterior excluded) for orders -m_max..m_max at axial wavenumber ``beta``."""
    radii, n = np.asarray(radii, float), np.asarray(n, complex)
    mu = np.ones_like(n) if mu is None else np.asarray(mu, complex)
    if radii.ndim != 1 or not len(radii) or radii[0] <= 0 or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must be positive and strictly increasing")
    if n.shape != (len(radii) + 1,) or mu.shape != n.shape or not np.all(np.isfinite(n)) or np.any(n == 0):
        raise ValueError("one finite nonzero index per radial region, host last, is required")
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be positive and finite")
    hydro = hydrodynamic_regions(hydrodynamic, len(n))
    for j, model in enumerate(hydro):
        if model is not None and mu[j] != 1:
            raise ValueError("hydrodynamic regions must be nonmagnetic (mu = 1)")
    k = 2 * np.pi * n / wavelength
    q = outgoing_root(k * k - complex(beta) ** 2)
    if np.any(np.abs(q) < 1e-13 / wavelength):
        raise ValueError("exact axial incidence/light-line evaluation needs its limiting form")
    if m_max is None:
        size = max(abs(q * radii[-1]))
        m_max = max(8, int(np.ceil(size + 4 * size ** (1 / 3) + 8)))
    if int(m_max) != m_max or not 1 <= m_max <= 1500:
        raise ValueError("m_max must be an integer in 1..1500")
    orders = np.arange(-int(m_max), int(m_max) + 1)
    traces = _CylinderTraces(radii, n, mu, float(wavelength), beta, orders, hydro)
    sweep = channel_sweep(traces, radii, 2, [h is not None for h in hydro])
    host = sweep.outer[-1]
    with np.errstate(over="ignore", under="ignore"):
        t = sweep.R_out[-1] * np.exp(host.LF[:, 0] - host.LG[:, 0])[:, None, None]
    if not np.all(np.isfinite(t)):
        raise ArithmeticError("cylindrical response exceeded numerical precision")
    return NonlocalCylinder(radii, n, mu, float(wavelength), complex(beta), orders, hydro, sweep, traces, t)
