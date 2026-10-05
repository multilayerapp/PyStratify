"""Angular-spectrum electric/magnetic point sources in isotropic planar stacks.

Sources and exteriors are lossless. Guided channels are real-axis pole residues;
power in attenuated modes is counted through actual dissipative layer flux.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .cylindrical import outgoing_root
from .integration import integrate
from .response import emitted_amplitudes
from .planar import _coh_tmm_amplitudes_numpy


@dataclass(frozen=True)
class FilmRates:
    total: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    guided: np.ndarray
    absorbed: np.ndarray
    error: float
    evaluations: int
    converged: bool
    balance_error: np.ndarray
    poles: tuple

    @property
    def escape(self):
        return self.upper + self.lower


class FilmSource:
    def __init__(self, n, thickness, wavelength, layer, depth, dipole="electric"):
        self.n, self.d = np.asarray(n, complex), np.asarray(thickness, float)
        self.wavelength, self.layer, self.depth = float(wavelength), int(layer), float(depth)
        if self.n.ndim != 1 or self.d.shape != self.n.shape or len(n) < 2:
            raise ValueError("planar media/thickness arrays must have matching length")
        if not np.all(np.isfinite(self.n)) or np.any(self.n == 0) or np.any(self.n.imag < 0):
            raise ValueError("finite passive nonzero indices are required")
        if not np.isinf(self.d[0]) or not np.isinf(self.d[-1]) or np.any(self.d[1:-1] < 0):
            raise ValueError("planar thicknesses start/end with infinite half-spaces")
        if dipole not in ("electric", "magnetic"):
            raise ValueError("dipole must be electric or magnetic")
        if not 0 <= layer < len(n) or wavelength <= 0 or not np.isfinite(depth):
            raise ValueError("invalid source region, position or wavelength")
        if (0 < layer < len(n) - 1 and not 0 < depth < self.d[layer]) or (layer == 0 and depth >= 0) or (layer == len(n) - 1 and depth <= 0):
            raise ValueError("source position must lie strictly inside its region")
        if any(self.n[j].imag != 0 or self.n[j].real <= 0 for j in (0, layer, len(n) - 1)):
            raise ValueError("source medium and both exteriors must be lossless with positive index")
        self.mu = np.ones_like(self.n) if dipole == "electric" else self.n ** 2
        self.ns = self.n[layer].real
        self.k0, self.ks = 2 * np.pi / wavelength, 2 * np.pi * self.ns / wavelength
        if layer == 0:
            self.upper = (self.n[:1].repeat(2), np.array([np.inf, np.inf]), self.mu[:1].repeat(2))
            self.lower = (np.r_[self.n[0], self.n], np.r_[np.inf, -depth, self.d[1:]], np.r_[self.mu[0], self.mu])
            gap = -depth
        elif layer == len(n) - 1:
            self.upper = (np.r_[self.n[layer], self.n[::-1]], np.r_[np.inf, depth, self.d[-2::-1]], np.r_[self.mu[layer], self.mu[::-1]])
            self.lower = (self.n[-1:].repeat(2), np.array([np.inf, np.inf]), self.mu[-1:].repeat(2))
            gap = depth
        else:
            self.upper = (np.r_[self.n[layer], self.n[layer::-1]], np.r_[np.inf, depth, self.d[layer - 1::-1]], np.r_[self.mu[layer], self.mu[layer::-1]])
            self.lower = (np.r_[self.n[layer], self.n[layer:]], np.r_[np.inf, self.d[layer] - depth, self.d[layer + 1:]], np.r_[self.mu[layer], self.mu[layer:]])
            gap = min(depth, self.d[layer] - depth)
        self.maximum = max(2.0, np.max(np.abs(self.n)) / self.ns + 16 / (self.ks * gap))

    def half(self, u, half, pol):
        n, d, mu = half
        kz = outgoing_root((self.k0 * n) ** 2 - (self.ks * u) ** 2)
        cosine = kz / (self.k0 * n)
        angles = np.arccos(cosine)
        delta = np.zeros(len(n), complex)
        delta[1:-1] = kz[1:-1] * d[1:-1]
        r, t, vw = _coh_tmm_amplitudes_numpy(pol, n / mu, angles, delta)
        vw[0] = 1, r
        return r, t, vw, kz, cosine

    @staticmethod
    def flux(pol, y, cosine, forward, backward):
        if pol == "s":
            return np.real(y * cosine * np.conj(forward + backward) * (forward - backward))
        return np.real(y * np.conj(cosine) * (forward + backward) * np.conj(forward - backward))

    def half_power(self, half, data, pol):
        n, d, mu = half
        r, t, vw, kz, cosine = data
        ys = self.ns / self.mu[self.layer].real
        escape = self.flux(pol, n[-1] / mu[-1], cosine[-1], t, 0) / ys
        absorbed = 0.0
        for j in range(1, len(n) - 1):
            if n[j].imag == 0 and mu[j].imag == 0:
                continue
            f, b = vw[j]
            top = self.flux(pol, n[j] / mu[j], cosine[j], f, b)
            # Recover the bottom from the following layer, avoiding exp(+Im kz d).
            fn, bn = vw[j + 1]
            bottom = self.flux(pol, n[j + 1] / mu[j + 1], cosine[j + 1], fn, bn)
            absorbed += (top - bottom) / ys
        return escape, absorbed

    def spectral(self, u, powers=True):
        v = complex(outgoing_root(1 - u * u))
        work = np.zeros(2, complex)
        channels = np.zeros((2, 3))
        for pol in ("s", "p"):
            upper, lower = self.half(u, self.upper, pol), self.half(u, self.lower, pol)
            ru, rd = upper[0], lower[0]
            denominator = 1 - ru * rd
            projections = ((0, 0), (1, 1)) if pol == "s" else ((-u, -u), (v, -v))
            for orientation, (plus, minus) in enumerate(projections):
                if plus == minus == 0:
                    continue
                up, down = emitted_amplitudes(ru, rd, minus, plus)
                weight = 1 if orientation == 0 else .5
                work[orientation] += weight * (plus * ru * up + minus * rd * down)
                if powers:
                    eu, au = self.half_power(self.upper, upper, pol)
                    ed, ad = self.half_power(self.lower, lower, pol)
                    channels[orientation] += weight * np.array([eu * abs(up) ** 2, ed * abs(down) ** 2, au * abs(up) ** 2 + ad * abs(down) ** 2])
        green = 1.5 * u * work / v
        channels *= .75 * u / abs(v) ** 2
        return green, channels

    def guided_poles(self):
        if np.any(self.n.imag):
            return []
        low, high = max(self.n[0].real, self.n[-1].real) / self.ns, max(self.n.real) / self.ns
        if high <= low:
            return []
        count = min(10000, max(400, int(40 * self.k0 * np.sum(self.n[1:-1].real * self.d[1:-1]))))
        split = np.unique(np.r_[low, self.n.real / self.ns, high])
        split = split[(split >= low) & (split <= high)]
        poles = []
        for pol in ("s", "p"):
            def product(u):
                return self.half(u, self.upper, pol)[0] * self.half(u, self.lower, pol)[0]
            for a, b in zip(split[:-1], split[1:]):
                gap = 1e-9 * max(1, b)
                grid = np.linspace(a + gap, b - gap, max(16, int(count * (b - a) / (high - low))))
                values = np.array([product(x) for x in grid])
                for j in np.flatnonzero(values[:-1].imag * values[1:].imag < 0):
                    root = brentq(lambda u: product(u).imag, grid[j], grid[j + 1], xtol=2e-13)
                    if abs(1 - product(root)) < 1e-6 and not any(abs(root - p) < 1e-8 for p in poles):
                        poles.append(root)
        return sorted(poles)

    def rates(self, tolerance=1e-6, max_evaluations=20000):
        poles = self.guided_poles()
        guided = np.zeros(2)
        residues = []
        regular_parts = []
        pole_error = 0.0
        for root in poles:
            step = min(1e-5 * max(1, root), abs(root - 1) / 100)
            def residue(h):
                return h * (self.spectral(root + h, False)[0] - self.spectral(root - h, False)[0]) / 2
            r1, r2 = residue(step), residue(step / 2)
            r2 = (4 * r2 - r1) / 3
            pole_error += float(np.max(np.abs(r2 - r1))) * np.pi
            residues.append((root, r2))
            left = self.spectral(root - step, False)[0].real + r2.real / step
            right = self.spectral(root + step, False)[0].real - r2.real / step
            regular_parts.append((root, step, left, right))
            guided -= np.pi * r2.imag
        def integrand(u):
            for root, step, left, right in regular_parts:
                if abs(u - root) < step:
                    fraction = (u - root + step) / (2 * step)
                    return np.r_[left * (1 - fraction) + right * fraction, np.zeros(6)]
            green, channels = self.spectral(u)
            # Lossless pole principal parts are imaginary on the real axis.
            # Removing the numerical real remainder prevents false quadrature spikes.
            for root, residue in residues:
                green -= residue.real / (u - root)
            return np.r_[green.real, channels.ravel()]
        breaks = [0, 1, *[v / self.ns for v in self.n.real if v > 0], self.maximum]
        breaks += [p - 1e-7 * max(1, p) for p in poles] + [p + 1e-7 * max(1, p) for p in poles]
        integral = integrate(integrand, [b for b in breaks if 0 <= b <= self.maximum], tolerance, max_evaluations)
        total = 1 + integral.value[:2] + guided
        channels = integral.value[2:].reshape(2, 3)
        upper, lower, absorbed = channels.T
        balance = np.abs(total - upper - lower - absorbed - guided) / np.maximum(1, np.abs(total))
        error = integral.error + pole_error
        valid = integral.converged and np.max(balance) <= tolerance and np.min(np.r_[total, upper, lower, absorbed, guided]) >= -tolerance
        return FilmRates(total, upper, lower, guided, absorbed, error, integral.evaluations, bool(valid), balance, tuple(poles))

    def pattern(self, theta, phi=0):
        theta, phi = np.broadcast_arrays(theta, phi)
        out = np.zeros(theta.shape + (2,))
        for idx in np.ndindex(theta.shape):
            upper = theta[idx] <= np.pi / 2
            half = self.upper if upper else self.lower
            angle = theta[idx] if upper else np.pi - theta[idx]
            nh = half[0][-1].real
            u = nh / self.ns * np.sin(angle)
            if abs(np.cos(angle)) < 1e-7 or abs(u - 1) < 1e-12:
                angle = max(0, min(angle - 1e-5, np.pi / 2 - 1e-5))
                u = nh / self.ns * np.sin(angle)
            v = complex(outgoing_root(1 - u * u))
            for pol in ("s", "p"):
                hu, hd = self.half(u, self.upper, pol), self.half(u, self.lower, pol)
                ru, rd = hu[0], hd[0]
                projections = ((0, 0), (np.sin(phi[idx]), np.sin(phi[idx]))) if pol == "s" else ((-u, -u), (v * np.cos(phi[idx]), -v * np.cos(phi[idx])))
                for orientation, (plus, minus) in enumerate(projections):
                    up, down = emitted_amplitudes(ru, rd, minus, plus)
                    amplitude = up if upper else down
                    data = hu if upper else hd
                    escape, _ = self.half_power(half, data, pol)
                    out[idx + (orientation,)] += 3 / (8 * np.pi) * (nh / self.ns) ** 2 * np.cos(angle) * escape * abs(amplitude) ** 2 / abs(v) ** 2
        return out
