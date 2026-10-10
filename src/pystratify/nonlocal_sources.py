"""Point dipoles near hydrodynamic (nonlocal) metals: films and concentric cylinders.

The sources reuse the angular-spectrum (films) and axial-spectrum (cylinders) machinery of
:mod:`pystratify.planar_emission` and :mod:`pystratify.cylinder_emission`; only the response of
the stack on either side of the source changes.  The source region and the exteriors stay local
and lossless, so the response seen *at* the source has the local shape (a scalar per film
polarisation, a 2x2 block per cylinder order) and the Green's function there is formed exactly as
before.  The power reaching the exteriors and the power absorbed in each region come from the
Poynting plus hydrodynamic energy flux through every interface, taken through the Wronskian in
lossless regions, so that the balance total = escape + guided + absorbed tests two independent
paths.

Electric dipoles excite the electron gas through p (TM) waves; the duality trick of the planar
code (a magnetic dipole as an electric one in the dual medium, mu = n^2) swaps the labels, and the
longitudinal wave follows the physical polarisation.
"""

from __future__ import annotations

import numpy as np

from .nonlocal_film import FilmResponse, film_response
from .nonlocal_sphere import hydrodynamic_regions
from .cylinder_emission import CylinderSource
from .planar_emission import FilmSource

__all__ = ["NonlocalFilmSource", "NonlocalCylinderSource"]


class NonlocalFilmSource(FilmSource):
    """:class:`~pystratify.planar_emission.FilmSource` with hydrodynamic films ({region: Hydrodynamic};
    neither the source region nor the exteriors)."""

    def __init__(self, n, thickness, wavelength, layer, depth, dipole="electric", hydrodynamic=None):
        super().__init__(n, thickness, wavelength, layer, depth, dipole)
        M = len(self.n)
        hydro = hydrodynamic_regions(hydrodynamic, M, host_allowed=True)
        if hydro[0] is not None or hydro[-1] is not None or hydro[layer] is not None:
            raise ValueError("the source region and the exteriors of a film emitter must be local")
        self.hydro = hydro
        self.dipole = dipole
        L = int(layer)
        size_up, size_down = len(self.upper[0]), len(self.lower[0])
        self._up = {i: hydro[L if i == 0 else L - (i - 1)] for i in range(size_up)}
        self._down = {i: hydro[L if i == 0 else L + (i - 1)] for i in range(size_down)}
        self._up = {i: m for i, m in self._up.items() if m is not None}
        self._down = {i: m for i, m in self._down.items() if m is not None}

    def _hydro_of(self, half):
        return self._up if half is self.upper else self._down

    def _physical(self, pol):
        return pol if self.dipole == "electric" else ("p" if pol == "s" else "s")

    def half(self, u, half, pol):
        hydro = self._hydro_of(half)
        if self._physical(pol) == "s" or not hydro:
            return super().half(u, half, pol)
        n, d, _ = half  # n are the physical indices (the dual medium only changes mu)
        response = film_response(n, d, self.wavelength, self.ks * u, hydro)
        return complex(response.r[0]), response

    def half_power(self, half, data, pol):
        if len(data) != 2 or not isinstance(data[1], FilmResponse):
            return super().half_power(half, data, pol)
        response = data[1]
        flux = response.flux[0] + response.hydro_flux[0]
        # unit incident H_y at the source plane: the planar code's units are n_s times these
        escape = self.ns * flux[-1]
        absorbed = self.ns * (flux[0] - flux[-1])
        return escape, absorbed


# ----------------------------------------------------------------------------------------------- cylinders
_SWAP = np.array([[0, 1], [1, 0]], complex)


class _SourceView:
    """What :meth:`CylinderSource.coefficients` reads from a solution, built from the hydrodynamic
    sweep: orders, beta, q, the responses at the source radius and (axis sources) the core's
    log S.  For a magnetic dipole they are mapped into the dual picture of the planar/cylinder
    code (mu = n^2): R_dual = P R P with P swapping N and M."""

    def __init__(self, solution, layer, magnetic, axis=False):
        self.solution, self.layer, self.magnetic = solution, layer, magnetic
        self.orders, self.beta = solution.orders, solution.beta
        self.q = solution.traces.q
        if axis:  # a source on the axis reads the core's unscaled S (only m = -1, 0, 1 are kept)
            core = solution.sweep.inner[0]
            S = solution.sweep.S_in[0] * np.exp(core.LG[:, 0] - core.LF[:, 0])[:, None, None]
            self.log_s = np.log(self._dual(S))[None]

    def _dual(self, M):
        return _SWAP @ M @ _SWAP if self.magnetic else M

    def reflection_at(self, radius):
        from .special import cylinder_pair
        s, sw = self.layer, self.solution.sweep
        pairs = cylinder_pair(self.q[s], radius, self.orders)
        LF0, LG0 = pairs[0][2], pairs[1][2]
        L = self.orders.size
        R = np.zeros((L, 2, 2), complex)
        if s > 0:
            prev = sw.outer[s - 1]
            R = sw.R_out[s - 1] * np.exp(LG0 - prev.LG[:, 0] + prev.LF[:, 0] - LF0)[:, None, None]
        S = np.zeros((L, 2, 2), complex)
        if s < sw.count:
            nxt = sw.inner[s]
            S = sw.S_in[s] * np.exp(LF0 - nxt.LF[:, 0] + nxt.LG[:, 0] - LG0)[:, None, None]
        return s, pairs, self._dual(R), self._dual(S)


class NonlocalCylinderSource(CylinderSource):
    """:class:`~pystratify.cylinder_emission.CylinderSource` with hydrodynamic regions
    ({region: Hydrodynamic}; not the source region or the exterior).  Each axial wavenumber is
    one hydrodynamic sweep (TE, TM and the longitudinal wave coupled); the absorbed power of a
    region is the jump of its own Poynting plus hydrodynamic flux, taken inside it (well
    conditioned where it absorbs)."""

    def __init__(self, radii, n, wavelength, radius, dipole="electric", m_max=None, tolerance=1e-6, hydrodynamic=None):
        super().__init__(radii, n, wavelength, radius, dipole, m_max, tolerance)
        hydro = hydrodynamic_regions(hydrodynamic, len(self.n))
        if hydro[self.layer] is not None:
            raise ValueError("the source region must be local")
        self.hydro = hydro
        self.magnetic = dipole == "magnetic"

    def solution(self, b, maximum=None):
        from .nonlocal_cylinder import solve_nonlocal_cylinder
        return solve_nonlocal_cylinder(self.radii, self.n, self.wavelength, self.hydro, beta=self.ks * b,
                                       m_max=self.m_max if maximum is None else maximum)

    def guided_poles(self):
        if any(h is not None for h in self.hydro) and not np.any(self.n.imag):
            raise NotImplementedError("guided modes of lossless hydrodynamic stacks (gamma = 0) are not supported")
        return super().guided_poles()

    def _physical(self, amplitudes):
        """Source-layer amplitudes (orders, 2, 3) from the dual picture into the physical one."""
        if not self.magnetic:
            return amplitudes
        return (1j / self.n[self.layer].real) * (_SWAP @ amplitudes)

    def _chains(self, solution, outward, inward):
        """(A^, B^, log) of every region at its inner ('in') and outer ('out') radius for the source
        field, from the scaled source-layer amplitudes at the source radius (physical picture)."""
        from .special import cylinder_pair
        s, sw = self.layer, solution.sweep
        N = sw.count
        pairs = cylinder_pair(self.q_at(solution), self.radius, solution.orders) if self.radius else None
        amp = {}
        zero = np.zeros(outward.shape[:1] + (3,))
        if s < N:
            LGs = sw.inner[s].LG[:, 0]
            shift = LGs - (pairs[1][2] if pairs is not None else 0)
            b_s = outward * np.exp(1j * np.imag(shift))[:, None, None]
            lg = np.broadcast_to(np.real(shift)[:, None], zero.shape).copy()
            amp[(s, "out")] = (sw.S_in[s] @ b_s, b_s, lg)
            chain = sw.outgoing_amplitudes(s, b_s, lg)
            for j in range(s + 1, N + 1):
                amp[(j, "in")] = chain[j][0]
                if chain[j][1] is not None:
                    amp[(j, "out")] = chain[j][1]
        if s > 0 and inward is not None:
            prev = sw.outer[s - 1]
            shift = prev.LF[:, 0] - pairs[0][2]
            a_s = inward * np.exp(1j * np.imag(shift))[:, None, None]
            lg = np.broadcast_to(np.real(shift)[:, None], zero.shape).copy()
            amp[(s, "in")] = (a_s, sw.R_out[s - 1] @ a_s, lg)
            chain = sw.regular_chain(s, a_s, lg)
            for j in range(s - 1, -1, -1):
                if chain[j][0] is not None:
                    amp[(j, "in")] = chain[j][0]
                amp[(j, "out")] = chain[j][1]
        return amp

    def q_at(self, solution):
        return solution.traces.q[self.layer]

    def _host(self, solution, amp):
        """Physical outgoing amplitudes of the host (orders, 2, 3), in the dual picture for a magnetic dipole."""
        sw = solution.sweep
        N = sw.count
        if self.layer == N:
            return None
        a, b, lg = amp[(N, "in")]
        host = sw.outer[N - 1]
        coefficient = b * np.exp(lg[:, None, :] - host.LG[:, 0][:, None, None])
        if self.magnetic:
            coefficient = (self.n[-1].real / 1j) * (_SWAP @ coefficient)
        return coefficient

    def spectral(self, b, maximum=None, powers=True):
        solution = self.solution(b, maximum)
        view = _SourceView(solution, self.layer, self.magnetic, self.radius == 0)
        work, outward, inward, pairs = self.coefficients(view)
        green = 1.5 * np.sum(work, axis=0)
        escape, absorbed = np.zeros(3), np.zeros(3)
        if powers:
            amp = self._chains(solution, self._physical(outward), None if inward is None else self._physical(inward))
            if self.ks * b < self.k0 * self.n[-1].real:
                if self.layer == len(self.radii):
                    coefficient = outward * np.exp(-pairs[1][2])[:, None, None]
                else:
                    coefficient = self._host(solution, amp)
                escape = 1.5 * (self.mu[self.layer] / self.mu[-1]).real * np.sum(np.abs(coefficient) ** 2, axis=(0, 1))
            sw = solution.sweep
            for j in range(len(self.radii)):
                if self.n[j].imag == 0 and self.hydro[j] is None:
                    continue
                flux = []
                for side, radius, tr in (("in", self.radii[j - 1] if j else 0.0, sw.outer[j - 1] if j else None),
                                         ("out", self.radii[j], sw.inner[j])):
                    if radius == 0:
                        flux.append(np.zeros(3))
                        continue
                    a, bb, lg = amp[(j, side)]
                    vec = tr.physical(tr.F @ a + tr.G @ bb) * np.exp(lg)[:, None, :]  # (orders, 6, 3)
                    poynting = np.real(vec[:, 1] * np.conj(vec[:, 2]) - vec[:, 0] * np.conj(vec[:, 3]))
                    hydro = self.k0 * np.imag(np.conj(vec[:, 4]) * vec[:, 5])
                    flux.append(radius * np.sum(poynting + hydro, axis=0))
                absorbed += .75 * np.pi * self.mu[self.layer].real * self.k0 * (flux[0] - flux[1])
        tail = 0.0 if self.radius == 0 else float(np.max(np.sum(np.abs(work[np.abs(solution.orders) >= max(1, solution.orders[-1] - 3)]), axis=0)))
        return green, escape, absorbed, tail

    def pattern(self, theta, phi=0):
        theta, phi = np.broadcast_arrays(theta, phi)
        density = np.zeros(theta.shape + (3,))
        prefactor = 3 / (8 * np.pi) * (self.mu[self.layer] * self.n[-1] / (self.mu[-1] * self.n[self.layer])).real
        beta = self.n[-1].real / self.n[self.layer].real * np.cos(np.clip(theta, 1e-6, np.pi - 1e-6))
        for b in np.unique(beta):
            solution = self.solution(b)
            view = _SourceView(solution, self.layer, self.magnetic, self.radius == 0)
            _, outward, inward, pairs = self.coefficients(view)
            if self.layer == len(self.radii):
                coefficient = outward * np.exp(-pairs[1][2])[:, None, None]
            else:
                amp = self._chains(solution, self._physical(outward), None if inward is None else self._physical(inward))
                coefficient = self._host(solution, amp)
            selected = beta == b
            phase = np.exp(1j * (phi[selected, None] - np.pi / 2) * solution.orders)
            far = np.einsum("pm,mij->pij", phase, coefficient)
            density[selected] = prefactor * np.sum(abs(far) ** 2, axis=1)
        return density
