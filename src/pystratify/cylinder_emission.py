"""Three-dimensional point dipoles: cylindrical orders and continuous axial spectrum."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize_scalar, brentq

from .cylindrical import solve_cylinder, vectors
from .special import cylinder_pair
from .integration import integrate
from .response import source_coefficients


@dataclass(frozen=True)
class CylinderRates:
    total: np.ndarray
    escape: np.ndarray
    guided: np.ndarray
    absorbed: np.ndarray
    error: float
    evaluations: int
    orders: int
    converged: bool
    balance_error: np.ndarray
    poles: tuple
    grazing_error: float = 0.0


def axis_vectors(beta, k, q, orders):
    M, N = np.zeros((len(orders), 3), complex), np.zeros((len(orders), 3), complex)
    for i, m in enumerate(orders):
        if m == 0:
            N[i, 2] = q / k
        elif m == 1:
            M[i, :2], N[i, :2] = (1j / 2, -.5), (1j * beta / (2 * k), -beta / (2 * k))
        elif m == -1:
            M[i, :2], N[i, :2] = (1j / 2, .5), (-1j * beta / (2 * k), -beta / (2 * k))
    return M, N


class CylinderSource:
    def __init__(self, radii, n, wavelength, radius, dipole="electric", m_max=None, tolerance=1e-6):
        self.radii, self.n = np.asarray(radii, float), np.asarray(n, complex)
        self.wavelength, self.radius = float(wavelength), float(radius)
        self.layer = int(np.searchsorted(self.radii, radius))
        if radius < 0 or not np.isfinite(radius) or wavelength <= 0:
            raise ValueError("source radius must be nonnegative and wavelength positive")
        if dipole not in ("electric", "magnetic"):
            raise ValueError("dipole must be electric or magnetic")
        if any(self.n[j].imag != 0 or self.n[j].real <= 0 for j in (self.layer, len(n) - 1)):
            raise ValueError("source medium and exterior must be lossless with positive index")
        self.mu = np.ones_like(self.n) if dipole == "electric" else self.n ** 2
        self.k0 = 2 * np.pi / wavelength
        self.ks = self.k0 * self.n[self.layer].real
        self.tolerance = tolerance
        gap = np.min(np.abs(self.radii - radius))
        if gap == 0:
            raise ValueError("source cannot lie on an interface")
        self.maximum = max(2.0, np.max(np.abs(self.n)) / self.n[self.layer].real + 16 / (self.ks * gap))
        qratio = max(np.minimum(self.radii / max(radius, 1e-300), radius / self.radii)) if radius else 0
        size = self.k0 * max(abs(self.n)) * self.radii[-1]
        needed = max(8, int(np.ceil(size + 4 * size ** (1 / 3) + 8)), int(np.ceil(-np.log(tolerance) / max(1e-8, -2 * np.log(max(qratio, 1e-300))))) + 10)
        self.m_max = int(m_max) if m_max is not None else (1 if radius == 0 else min(1500, needed))
        self.explicit_order = m_max is not None

    def solution(self, b, maximum=None):
        # the integrand is integrable at a layer's light line but q^2 = k^2 - beta^2 rounds to zero within
        # ~1e-15 of it: evaluate 1e-12 away, on the same side (an ulp in b is not an ulp in q^2)
        for line in self.n.real[self.n.imag == 0] / self.n[self.layer].real:
            if abs(b - line) < 1e-12 * line:
                b = line + (1e-12 if b >= line else -1e-12) * line
        return solve_cylinder(self.radii, self.n, self.wavelength, mu=self.mu, beta=self.ks * b, m_max=self.m_max if maximum is None else maximum)

    def coefficients(self, solution):
        d, radius, orders = self.layer, self.radius, solution.orders
        k, beta, q = self.ks, solution.beta, solution.q[d]
        if radius == 0:
            M, N = axis_vectors(beta, k, q, orders)
            Mr, Nr = axis_vectors(-beta, k, q, -orders)
            parity = np.where(np.abs(orders) % 2, -1, 1)[:, None]
            Mr, Nr = parity * Mr, parity * Nr
            direct = np.stack((Nr, Mr), axis=-2)
            S = np.exp(solution.log_s[0])
            reflected = S @ direct
            field = np.einsum("mip,mij->mpj", np.stack((N, M), axis=-2), reflected)
            reflected_work = np.einsum("mii->mi", field)
            return reflected_work, direct, None, None
        d, pairs, R, S = solution.reflection_at(radius)
        Mj, Nj = vectors(q, k, beta, radius, orders, pairs[0])
        Mh, Nh = vectors(q, k, beta, radius, orders, pairs[1])
        Mjr, Njr = vectors(q, k, -beta, radius, -orders, pairs[0])
        Mhr, Nhr = vectors(q, k, -beta, radius, -orders, pairs[1])
        factor = np.exp(pairs[0][2] + pairs[1][2])[:, None, None]
        direct_regular = factor * np.stack((Nhr, Mhr), axis=-2)
        direct_outgoing = factor * np.stack((Njr, Mjr), axis=-2)
        A, B = source_coefficients(R, S, direct_regular, direct_outgoing, blocks=True)
        field = np.einsum("mip,mij->mpj", np.stack((Nj, Mj), axis=-2), A) + np.einsum("mip,mij->mpj", np.stack((Nh, Mh), axis=-2), B)
        reflected_work = np.einsum("mii->mi", field)
        return reflected_work, direct_outgoing + B, direct_regular + A, pairs

    @staticmethod
    def inverse_reference(log_matrix, function_log=0):
        reference = log_matrix + (function_log[:, None, None] if np.ndim(function_log) else function_log)
        scale = np.max(reference.real, axis=(-2, -1))
        inverse = np.linalg.inv(np.exp(reference - scale[:, None, None]))
        return inverse, scale

    def amplitudes(self, solution, outward, inward, pairs):
        d = self.layer
        lh = pairs[1][2] if pairs is not None else 0
        inverse_out, scale_out = self.inverse_reference(solution.log_b_out[d], lh)
        transformed_out = inverse_out @ outward
        if inward is not None:
            inverse_in, scale_in = self.inverse_reference(solution.log_a[d], pairs[0][2])
            transformed_in = inverse_in @ inward
        else:
            transformed_in, scale_in = None, None
        return transformed_out, scale_out, transformed_in, scale_in

    def field_at(self, solution, amplitudes, layer, radius):
        out, scale_out, inward, scale_in = amplitudes
        pairs = cylinder_pair(solution.q[layer], radius, solution.orders)
        if layer >= self.layer:
            B = np.exp(solution.log_b_out[layer] + pairs[1][2][:, None, None] - scale_out[:, None, None]) @ out
            A = np.exp(solution.log_s[layer] + (pairs[0][2] - pairs[1][2])[:, None, None]) @ B
        else:
            A = np.exp(solution.log_a[layer] + pairs[0][2][:, None, None] - scale_in[:, None, None]) @ inward
            B = np.exp(solution.log_b[layer] + pairs[1][2][:, None, None] - scale_in[:, None, None]) @ inward
        k = self.k0 * self.n[layer]
        Mj, Nj = vectors(solution.q[layer], k, solution.beta, radius, solution.orders, pairs[0])
        Mh, Nh = vectors(solution.q[layer], k, solution.beta, radius, solution.orders, pairs[1])
        E = np.einsum("mip,mij->mpj", np.stack((Nj, Mj), axis=-2), A) + np.einsum("mip,mij->mpj", np.stack((Nh, Mh), axis=-2), B)
        H = -1j * self.n[layer] / self.mu[layer] * (np.einsum("mip,mij->mpj", np.stack((Mj, Nj), axis=-2), A) + np.einsum("mip,mij->mpj", np.stack((Mh, Nh), axis=-2), B))
        return E, H

    def spectral(self, b, maximum=None, powers=True):
        solution = self.solution(b, maximum)
        work, outward, inward, pairs = self.coefficients(solution)
        green = 1.5 * np.sum(work, axis=0)
        escape, absorbed = np.zeros(3), np.zeros(3)
        if powers:
            amplitudes = self.amplitudes(solution, outward, inward, pairs)
            coefficient = np.exp(-amplitudes[1])[:, None, None] * amplitudes[0]
            if self.ks * b < self.k0 * self.n[-1].real:
                escape = 1.5 * (self.mu[self.layer] / self.mu[-1]).real * np.sum(np.abs(coefficient) ** 2, axis=(0, 1))
            for j in range(len(self.radii)):
                if self.n[j].imag == 0:
                    continue
                flux = []
                for r in (0 if j == 0 else self.radii[j - 1], self.radii[j]):
                    if r == 0:
                        flux.append(np.zeros(3))
                        continue
                    E, H = self.field_at(solution, amplitudes, j, r)
                    flux.append(r * np.sum(np.real(E[:, 1] * H[:, 2].conj() - E[:, 2] * H[:, 1].conj()), axis=0))
                absorbed += .75 * np.pi * self.mu[self.layer].real * self.k0 * (flux[0] - flux[1])
        tail = 0.0 if self.radius == 0 else float(np.max(np.sum(np.abs(work[np.abs(solution.orders) >= max(1, solution.orders[-1] - 3)]), axis=0)))
        return green, escape, absorbed, tail

    def guided_poles(self):
        if np.any(self.n.imag):
            return []
        low, high = self.n[-1].real / self.n[self.layer].real, max(self.n.real) / self.n[self.layer].real
        if high <= low:
            return []
        count = min(4000, max(160, int(30 * self.k0 * self.radii[-1] * max(self.n.real))))
        # modes near cutoff sit just above the host light line: sample it geometrically, or a root in
        # the first cell of the uniform grid could never be an interior minimum
        near_cutoff = low + (high - low) * np.geomspace(1e-8, 1 / count, 24)
        grid = np.unique(np.r_[near_cutoff, np.linspace(low + 1e-8, high - 1e-8, count)])
        positive = np.arange(self.m_max, 2 * self.m_max + 1)
        def determinant(b):
            sol = solve_cylinder(self.radii, self.n, self.wavelength, mu=self.mu, beta=self.ks * b, m_max=self.m_max, _response_only=True)
            C = sol.transfers[-1]
            D = C[:, :2, :2] + C[:, :2, 2:] @ sol.response.regular_in[-1]
            return np.linalg.det(D)[positive] / np.maximum(1e-300, np.sum(np.abs(D[positive]) ** 2, axis=(-2, -1)))
        values = np.abs(np.array([determinant(b) for b in grid]))
        poles = []
        for m in range(values.shape[1]):
            indices = np.flatnonzero((values[1:-1, m] < values[:-2, m]) & (values[1:-1, m] < values[2:, m])) + 1
            for j in indices:
                fit = minimize_scalar(lambda b: abs(determinant(b)[m]), bounds=(grid[j - 1], grid[j + 1]), method="bounded", options={"xatol":1e-13})
                # |det| is V-shaped at a root, where Brent's parabolic steps stall short of zero (a TM01
                # mode stopped at 1.2e-7 and was dropped): bracket the projected determinant and judge
                # the root itself, not the minimiser's last value
                h = min(1e-5, (fit.x - grid[j - 1]) / 2, (grid[j + 1] - fit.x) / 2)
                if h <= 0:
                    continue
                ends = determinant(fit.x - h)[m], determinant(fit.x + h)[m]
                phase = np.exp(-1j * np.angle(ends[1] - ends[0]))
                try:
                    root = brentq(lambda b: np.real(phase * determinant(b)[m]), fit.x - h, fit.x + h, xtol=2e-13)
                except ValueError:  # no sign change: a minimum of |det| that is not a zero
                    continue
                # a zero, judged against the determinant's own scale: toward the host light line the
                # normalised determinant of every m >= 1 falls like (b - b_light), below any fixed bar
                value = abs(determinant(root)[m])
                if value < 1e-7 and value < 1e-3 * max(abs(ends[0]), abs(ends[1])) and not any(abs(root - p) < 2e-7 for p in poles):
                    poles.append(root)
        return sorted(poles)

    def pattern(self, theta, phi=0):
        theta, phi = np.broadcast_arrays(theta, phi)
        density = np.zeros(theta.shape + (3,))
        prefactor = 3 / (8 * np.pi) * (self.mu[self.layer] * self.n[-1] / (self.mu[-1] * self.n[self.layer])).real
        beta = self.n[-1].real / self.n[self.layer].real * np.cos(np.clip(theta, 1e-6, np.pi - 1e-6))
        for b in np.unique(beta):
            solution = self.solution(b)
            _, outward, inward, pairs = self.coefficients(solution)
            transformed, scale, _, _ = self.amplitudes(solution, outward, inward, pairs)
            coefficient = np.exp(-scale)[:, None, None] * transformed
            selected = beta == b
            phase = np.exp(1j * (phi[selected, None] - np.pi / 2) * solution.orders)
            far = np.einsum("pm,mij->pij", phase, coefficient)
            density[selected] = prefactor * np.sum(abs(far) ** 2, axis=1)
        return density

    def _rates_once(self, tolerance=1e-6, max_evaluations=20000):
        poles = self.guided_poles()
        light_line = self.n[-1].real / self.n[self.layer].real
        # a mode near cutoff must not regularise across the host light line, a branch point
        steps = [min(1e-5 * max(1, root), (root - light_line) / 2) for root in poles]
        guided, regular_parts, residue_error = np.zeros(3), [], 0.0
        for root, step in zip(poles, steps):
            def residue(h):
                return h * (self.spectral(root + h, powers=False)[0] - self.spectral(root - h, powers=False)[0]) / 2
            # two Richardson extrapolates (the h^2 term cancelled): their difference is the error of
            # the one used; the raw-against-extrapolated difference overstated it ~1e3x near cutoff
            r1, r2, r4 = residue(step), residue(step / 2), residue(step / 4)
            coarse, r2 = (4 * r2 - r1) / 3, (4 * r4 - r2) / 3
            residue_error += np.pi * float(np.max(np.abs(coarse - r2)))
            guided -= np.pi * r2.imag
            left = self.spectral(root - step, powers=False)[0].real + r2.real / step
            right = self.spectral(root + step, powers=False)[0].real - r2.real / step
            regular_parts.append((root, step, left, right))
        def sample(b):
            green, escape, absorbed, tail = self.spectral(b)
            return np.r_[green.real, escape, absorbed, tail]
        caps, grazing_error, cap_evaluations = [], 0.0, 0
        if np.any(self.n.imag):
            lines = np.unique(self.n[self.n.imag == 0].real / self.n[self.layer].real)
            for line in lines:
                neighbours = [abs(line - v) for v in lines if v != line]
                width = min(1e-5 * max(1, line), line / 4, min(neighbours, default=np.inf) / 4)
                if width <= 0:
                    continue
                for side in (-1, 1):
                    edge, outer = line + side * width, line + side * 2 * width
                    if not 0 < outer < self.maximum:
                        continue
                    if cap_evaluations + 2 >= max_evaluations:
                        raise ArithmeticError("adaptive integration evaluation budget exceeded")
                    value, away = sample(edge), sample(outer)
                    cap_evaluations += 2
                    # The vector basis loses digits at grazing incidence. Continue
                    # only the narrow end cap and charge its variation to the error
                    # budget; never relax the requested convergence tolerance.
                    grazing_error += 4 * width * float(np.max(abs(value - away)))
                    caps.append((line, side, width, value))
        def integrand(b):
            for root, step, left, right in regular_parts:
                if abs(b - root) < step:
                    f = (b - root + step) / (2 * step)
                    return np.r_[left * (1 - f) + right * f, np.zeros(7)]
            for line, side, width, value in caps:
                if 0 <= side * (b - line) < width:
                    return value
            return sample(b)
        breaks = [0, *[v / self.n[self.layer].real for v in self.n.real if v > 0], self.maximum]
        breaks += [p - step for p, step in zip(poles, steps)] + [p + step for p, step in zip(poles, steps)]
        integral = integrate(integrand, [b for b in breaks if 0 <= b <= self.maximum], tolerance, max_evaluations - cap_evaluations)
        total, escape, absorbed = 1 + integral.value[:3] + guided, integral.value[3:6], integral.value[6:9]
        balance = np.abs(total - escape - guided - absorbed) / np.maximum(1, abs(total))
        tail_ok = self.radius == 0 or integral.value[-1] <= tolerance * max(1, np.max(abs(total)))
        error = integral.error + residue_error + grazing_error
        converged = integral.converged and error <= tolerance * max(1, np.max(abs(total))) and tail_ok and np.max(balance) <= tolerance and residue_error <= tolerance * max(1, np.max(abs(total)))
        return CylinderRates(total, escape, guided, absorbed, error, integral.evaluations + cap_evaluations, self.m_max, bool(converged), balance, tuple(poles), grazing_error)

    def rates(self, tolerance=1e-6, max_evaluations=20000):
        """Refine automatic angular order within one total quadrature budget."""
        spent = 0
        for attempt in range(4):
            result = self._rates_once(tolerance, max_evaluations - spent)
            spent += result.evaluations
            if result.converged or self.explicit_order or self.radius == 0 or self.m_max == 1500 or max_evaluations - spent < 200:
                return replace(result, evaluations=spent)
            self.m_max = min(1500, int(np.ceil(1.5 * self.m_max)) + 4)
        return replace(result, evaluations=spent)
