"""Planar multilayers with hydrodynamic (nonlocal) metal films: p polarisation.

A p-polarised wave of in-plane wavenumber K, ~ exp(i K x), in a region of permittivity eps has
H = y H_y with k_z = sqrt(k^2 - K^2) and, per unit H_y (dropping 1/(omega eps_0) from E),

    forward  (+z)  E_x =  k_z/(k0 eps),   E_z = -K/(k0 eps),
    backward (-z)  E_x = -k_z/(k0 eps),   E_z = -K/(k0 eps);

in a hydrodynamic region the longitudinal wave Phi = exp(+-i k_zL z), k_zL = sqrt(k_L^2 - K^2), has
E = grad Phi = (i K, 0, +-i k_zL) Phi.  s polarisation has no normal field and is untouched by the
electron gas (use :func:`pystratify.planar.coh_tmm`).

The recursion (:mod:`pystratify.nonlocal_sweep`) runs from the exit half-space (region 0: only
forward, i.e. transmitted, waves) to the incident medium, in the coordinate zeta = z_exit - z:
forward waves are its regular functions, backward waves its outgoing ones.  Amplitudes are local
(referred to each interface), so r is the reflection amplitude of H_y at the first interface; it
equals the p reflection amplitude of :func:`~pystratify.planar.coh_tmm` for E.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cylindrical import outgoing_root
from .hydrodynamic import contact_weights
from .nonlocal_sphere import hydrodynamic_regions
from .nonlocal_sweep import ChannelSweep, Traces, channel_sweep

__all__ = ["FilmResponse", "film_response", "solve_nonlocal_film"]


class _FilmTraces:
    """Traces per sweep region s (medium M - 1 - s) at interface zeta_j, batched over K."""

    def __init__(self, n, d, wavelength, K, hydro):
        self.n, self.K = np.asarray(n, complex), np.atleast_1d(np.asarray(K, complex))
        M = self.n.size
        self.M = M
        self.k0 = 2 * np.pi / wavelength
        self.medium = [M - 1 - s for s in range(M)]
        self.zeta = np.concatenate(([0.0], np.cumsum([d[M - 1 - s] for s in range(1, M - 1)])))
        self.kz = [outgoing_root((self.k0 * v) ** 2 - self.K**2) for v in self.n]
        self.eps = self.n**2
        self.hydro = [hydro[self.medium[s]] for s in range(M)]
        self.eps_b, self.kzL = {}, {}
        for s, model in enumerate(self.hydro):
            if model is None:
                continue
            j = self.medium[s]
            self.eps_b[s] = complex(model.background(wavelength, self.eps[j]))
            kL = complex(model.longitudinal_wavenumber(wavelength, self.eps[j]))
            self.kzL[s] = outgoing_root(kL**2 - self.K**2)

    def __call__(self, region, interface):
        j = self.medium[region]
        zeta = self.zeta[interface]
        hydro = self.hydro[region] is not None
        c = 2 if hydro else 1
        B = self.K.size
        F, G = np.zeros((B, 4, c), complex), np.zeros((B, 4, c), complex)
        LF, LG = np.zeros((B, c), complex), np.zeros((B, c), complex)
        kz, eps, K = self.kz[j], self.eps[j], self.K
        jn = (eps - self.eps_b[region]) if hydro else 0.0
        LF[:, 0], LG[:, 0] = -1j * kz * zeta, 1j * kz * zeta
        for out, sign in ((F, 1), (G, -1)):
            out[:, 0, 0] = sign * kz / (self.k0 * eps)
            out[:, 1, 0] = 1
            out[:, 2, 0] = jn * (-K / (self.k0 * eps))
        if hydro:
            kzL, eps_b = self.kzL[region], self.eps_b[region]
            LF[:, 1], LG[:, 1] = -1j * kzL * zeta, 1j * kzL * zeta
            for out, sign in ((F, 1), (G, -1)):
                out[:, 0, 1] = 1j * K
                out[:, 2, 1] = -eps_b * sign * 1j * kzL
                out[:, 3, 1] = eps / (eps_b - eps)
        return Traces(F, G, LF, LG)


@dataclass(frozen=True)
class FilmResponse:
    """p-polarised response of a stack for in-plane wavenumbers ``K``.

    ``r``: reflection amplitude of H_y (= coh_tmm's r_p); ``t_h``: transmitted H_y at the last
    interface per unit incident H_y at the first; ``flux``: Re(E_x conj(H_y)) (-z oriented, in the
    units of the traces: E without 1/(omega eps_0), H_y) of the incident-normalised field on the
    incident side of every interface, first interface first, and ``hydro_flux`` the hydrodynamic
    energy flux k0 Im(conj(J_n) M) through it (nonzero only between two electron gases); per layer
    absorption is the difference of consecutive totals."""

    K: np.ndarray
    r: np.ndarray
    t_h: np.ndarray
    flux: np.ndarray
    hydro_flux: np.ndarray
    sweep: ChannelSweep
    traces: _FilmTraces


def film_response(n, d, wavelength, K, hydrodynamic, contact="electrochemical") -> FilmResponse:
    """``n`` (M,) incident medium first, ``d`` (M,) thicknesses with infinite half-spaces at both
    ends, ``K`` scalar or array of in-plane wavenumbers, ``hydrodynamic`` {medium: Hydrodynamic}
    (the incident medium excluded; the exit half-space may be a hydrodynamic metal)."""
    n, d = np.asarray(n, complex), np.asarray(d, float)
    M = n.size
    if M < 2 or d.shape != n.shape or not np.isinf(d[0]) or not np.isinf(d[-1]) or np.any(d[1:-1] < 0):
        raise ValueError("films need matching media/thickness arrays with infinite half-spaces at both ends")
    hydro_list = list(hydrodynamic_regions(hydrodynamic, M, host_allowed=True))
    if hydro_list[0] is not None:
        raise ValueError("the incident medium cannot be hydrodynamic")
    traces = _FilmTraces(n, d, float(wavelength), K, hydro_list)
    flags = [traces.hydro[s] is not None for s in range(M)]
    sweep = channel_sweep(traces, traces.zeta, 1, flags, contact_weights(traces.hydro, contact))
    r = sweep.R_out[-1][:, 0, 0]
    B = traces.K.size
    regular = sweep.regular_amplitudes(np.ones((B, 1, 1), complex))
    # field on the incident side of each interface (sweep region j + 1 at zeta_j), first interface first
    flux = np.zeros((B, M - 1))
    hydro_flux = np.zeros((B, M - 1))
    for j in range(M - 1):
        s = M - 2 - j  # sweep interface
        inner, _ = regular[s + 1]
        a, b, lg = inner
        tr = sweep.outer[s]
        vec = (tr.F @ a + tr.G @ b)[..., 0] * np.exp(lg)  # (B, 4)
        flux[:, j] = np.real(vec[:, 0] * np.conj(vec[:, 1]))
        if flags[s] and flags[s + 1]:
            hydro_flux[:, j] = traces.k0 * np.imag(np.conj(vec[:, 2]) * vec[:, 3])  # +z oriented
    _, outer0 = regular[0]
    a0, _, lg0 = outer0
    t_h = a0[:, 0, 0] * np.exp(lg0[:, 0])
    return FilmResponse(traces.K, r, t_h, flux, hydro_flux, sweep, traces)


def solve_nonlocal_film(n, d, wavelength, hydrodynamic, angle=0.0, contact="electrochemical"):
    """Plane wave from the first medium (lossless) at ``angle`` (radians): p from the hydrodynamic
    recursion, s from :func:`~pystratify.planar.coh_tmm`.  Returns r, R, T, absorption per layer
    (p and s) and the unpolarised R, T, A."""
    from .planar import coh_tmm
    n, d = np.asarray(n, complex), np.asarray(d, float)
    if n[0].imag != 0 or n[0].real <= 0:
        raise ValueError("the incident medium must be lossless")
    K = 2 * np.pi * n[0].real / wavelength * np.sin(angle)
    res = film_response(n, d, wavelength, K, hydrodynamic, contact)
    k0 = 2 * np.pi / wavelength
    kz0 = outgoing_root((k0 * n[0]) ** 2 - K**2)
    incident = np.real(kz0 / (k0 * n[0] ** 2))
    R_p = float(np.abs(res.r[0]) ** 2)
    kzN = outgoing_root((k0 * n[-1]) ** 2 - K**2)
    T_p = 0.0 if (n[-1].imag != 0 or (hydrodynamic or {}).get(len(n) - 1) is not None) else float(np.real(kzN / (k0 * n[-1] ** 2)) * np.abs(res.t_h[0]) ** 2 / incident)
    fluxes = (res.flux[0] + res.hydro_flux[0]) / incident
    layers_p = fluxes[:-1] - fluxes[1:]
    exit_p = fluxes[-1] - T_p
    s = coh_tmm("s", n, d, angle, wavelength)
    return dict(r_p=complex(res.r[0]), R_p=R_p, T_p=T_p, A_p=1 - R_p - T_p, absorption_p=np.r_[layers_p, exit_p],
                r_s=complex(s["r"]), R_s=float(s["R"]), T_s=float(s["T"]),
                reflectance=(R_p + float(s["R"])) / 2, transmittance=(T_p + float(s["T"])) / 2,
                absorptance=1 - (R_p + float(s["R"])) / 2 - (T_p + float(s["T"])) / 2,
                balance_error=abs(fluxes[0] - (1 - R_p)))
