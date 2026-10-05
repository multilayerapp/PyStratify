"""Vector cylindrical basis for concentric, infinite circular cylinders."""

from dataclasses import dataclass

import numpy as np

from .response import block_sweep, log_value, log_add
from .special import cylinder_pair


def outgoing_root(value):
    q = np.sqrt(np.asarray(value, complex))
    return np.where((q.imag < 0) | ((q.imag == 0) & (q.real < 0)), -q, q)


def log_apply(matrix, logarithms):
    terms = log_value(matrix)[..., :, :, None] + logarithms[..., None, :, :]
    result = terms[..., 0, :]
    for j in range(1, terms.shape[-2]):
        result = log_add(result, terms[..., j, :])
    return result


def vectors(q, k, beta, radius, orders, pair):
    f, df, _ = pair
    m = np.asarray(orders)
    M = np.stack((1j * m * f / (q * radius), -df, np.zeros_like(f)), axis=-1)
    N = np.stack((1j * beta * df / k, -beta * m * f / (k * q * radius), q * f / k), axis=-1)
    return M, N


def traces(q, k, y, beta, radius, orders, pair):
    M, N = vectors(q, k, beta, radius, orders, pair)
    return np.stack((
        np.stack((N[..., 2], np.zeros_like(N[..., 2])), axis=-1),
        np.stack((np.zeros_like(N[..., 2]), -1j * y * N[..., 2]), axis=-1),
        np.stack((N[..., 1], M[..., 1]), axis=-1),
        np.stack((-1j * y * M[..., 1], -1j * y * N[..., 1]), axis=-1),
    ), axis=-2)


@dataclass
class CylinderSolution:
    radii: np.ndarray
    n: np.ndarray
    mu: np.ndarray
    wavelength: float
    beta: complex
    orders: np.ndarray
    q: np.ndarray
    response: object
    transfers: np.ndarray
    inner_pairs: list
    outer_pairs: list
    log_a: np.ndarray
    log_b: np.ndarray
    log_b_out: np.ndarray
    log_s: np.ndarray
    t: np.ndarray

    def reflection_at(self, radius):
        layer = int(np.searchsorted(self.radii, radius))
        pairs = cylinder_pair(self.q[layer], radius, self.orders)
        shape = (len(self.orders), 2, 2)
        regular, outgoing = np.zeros(shape, complex), np.zeros(shape, complex)
        jlog, hlog = pairs[0][2], pairs[1][2]
        if layer:
            old_j, old_h = self.outer_pairs[layer - 1][0][2], self.outer_pairs[layer - 1][1][2]
            regular = self.response.regular_out[layer - 1] * np.exp(old_j - old_h + hlog - jlog)[:, None, None]
        if layer < len(self.radii):
            old_j, old_h = self.inner_pairs[layer][0][2], self.inner_pairs[layer][1][2]
            outgoing = self.response.outgoing_in[layer] * np.exp(jlog - hlog + old_h - old_j)[:, None, None]
        return layer, pairs, regular, outgoing

    def field(self, points, polarization="axial-electric"):
        points = np.asarray(points, float)
        rho, phi, z = np.hypot(points[..., 0], points[..., 1]), np.arctan2(points[..., 1], points[..., 0]), points[..., 2]
        out_e, out_h = np.zeros(points.shape, complex), np.zeros(points.shape, complex)
        incident = np.array([1, 0], complex) if polarization == "axial-electric" else np.array([0, 1j], complex)
        for idx in np.ndindex(rho.shape):
            radius = max(rho[idx], 1e-12 * self.wavelength)
            layer = int(np.searchsorted(self.radii, radius))
            pairs = cylinder_pair(self.q[layer], radius, self.orders)
            E, H = np.zeros(3, complex), np.zeros(3, complex)
            k = 2 * np.pi * self.n[layer] / self.wavelength
            for b, logs in enumerate((self.log_a[layer], self.log_b[layer])):
                M, N = vectors(self.q[layer], k, self.beta, radius, self.orders, pairs[b])
                coefficient = np.exp(logs + pairs[b][2][:, None, None]) @ incident
                phase = (1j ** self.orders) * np.exp(1j * self.orders * phi[idx] + 1j * self.beta * z[idx])
                E += np.sum(phase[:, None] * (coefficient[:, :1] * N + coefficient[:, 1:] * M), axis=0)
                H += np.sum(-1j * self.n[layer] / self.mu[layer] * phase[:, None] * (coefficient[:, :1] * M + coefficient[:, 1:] * N), axis=0)
            c, s = np.cos(phi[idx]), np.sin(phi[idx])
            out_e[idx] = c * E[0] - s * E[1], s * E[0] + c * E[1], E[2]
            out_h[idx] = c * H[0] - s * H[1], s * H[0] + c * H[1], H[2]
        return out_e, out_h


def solve_cylinder(radii, n, wavelength, *, mu=None, beta=0, m_max=None, _response_only=False):
    radii, n = np.asarray(radii, float), np.asarray(n, complex)
    mu = np.ones_like(n) if mu is None else np.asarray(mu, complex)
    if radii.ndim != 1 or not len(radii) or radii[0] <= 0 or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must be positive and strictly increasing")
    if n.shape != (len(radii) + 1,) or mu.shape != n.shape or not np.all(np.isfinite(n)) or np.any(n == 0):
        raise ValueError("one finite nonzero index per radial region, host last, is required")
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be positive and finite")
    k = 2 * np.pi * n / wavelength
    q = outgoing_root(k * k - beta * beta)
    if np.any(np.abs(q) < 1e-13 / wavelength):
        raise ValueError("exact axial incidence/light-line evaluation needs its limiting form")
    if m_max is None:
        size = max(abs(q * radii[-1]))
        m_max = max(8, int(np.ceil(size + 4 * size ** (1 / 3) + 8)))
    if int(m_max) != m_max or not 1 <= m_max <= 1500:
        raise ValueError("m_max must be an integer in 1..1500")
    orders = np.arange(-int(m_max), int(m_max) + 1)
    inner = [cylinder_pair(q[j], r, orders) for j, r in enumerate(radii)]
    outer = [cylinder_pair(q[j + 1], r, orders) for j, r in enumerate(radii)]
    transforms = []
    for j, radius in enumerate(radii):
        ti = np.concatenate([traces(q[j], k[j], n[j] / mu[j], beta, radius, orders, v) for v in inner[j]], axis=-1)
        to = np.concatenate([traces(q[j + 1], k[j + 1], n[j + 1] / mu[j + 1], beta, radius, orders, v) for v in outer[j]], axis=-1)
        transforms.append(np.linalg.solve(to, ti))
    transfers = np.array(transforms)
    propagation = np.array([
        np.exp(outer[j][0][2] - outer[j][1][2] + inner[j + 1][1][2] - inner[j + 1][0][2])[:, None, None]
        for j in range(len(radii) - 1)
    ]).reshape(len(radii) - 1, len(orders), 1, 1)
    response = block_sweep(transfers, propagation)
    if _response_only:
        return CylinderSolution(radii, n, mu, float(wavelength), beta, orders, q, response, transfers, inner, outer, None, None, None, None, None)
    log_a = np.full((len(n), len(orders), 2, 2), -np.inf + 0j)
    log_a[-1, :, 0, 0] = log_a[-1, :, 1, 1] = 0
    for j in range(len(radii) - 1, -1, -1):
        C = transfers[j]
        inverse = np.linalg.inv(C[:, :2, :2] + C[:, :2, 2:] @ response.regular_in[j])
        log_a[j] = log_apply(inverse, log_a[j + 1]) + (outer[j][0][2] - inner[j][0][2])[:, None, None]
    log_b = np.full_like(log_a, -np.inf)
    for j in range(1, len(n)):
        reflection = response.regular_out[j - 1]
        offset = outer[j - 1][0][2] - outer[j - 1][1][2]
        log_b[j] = log_apply(reflection, log_a[j]) + offset[:, None, None]
    t = np.exp(log_b[-1])
    if not np.all(np.isfinite(t)):
        raise ArithmeticError("cylindrical response exceeded numerical precision")
    log_bo = np.full_like(log_a, -np.inf)
    log_so = np.full_like(log_a, -np.inf)
    log_bo[-1, :, 0, 0] = log_bo[-1, :, 1, 1] = 0
    for j in range(len(radii) - 1, -1, -1):
        C = transfers[j]
        inverse = np.linalg.inv(C[:, 2:, :2] @ response.outgoing_in[j] + C[:, 2:, 2:])
        log_bo[j] = log_apply(inverse, log_bo[j + 1]) + (outer[j][1][2] - inner[j][1][2])[:, None, None]
        log_so[j] = log_value(response.outgoing_in[j]) + (inner[j][1][2] - inner[j][0][2])[:, None, None]
    return CylinderSolution(radii, n, mu, float(wavelength), beta, orders, q, response, transfers, inner, outer, log_a, log_b, log_bo, log_so, t)


def cross_widths(solution, polarization="unpolarized"):
    if solution.n[-1].imag != 0 or abs(solution.beta.imag) > 0:
        raise ValueError("cross widths require a lossless exterior and real incidence")
    a = np.eye(2, dtype=complex)
    columns = [0, 1] if polarization == "unpolarized" else [0 if polarization == "axial-electric" else 1]
    power = np.sum(np.abs(solution.t[..., columns]) ** 2, axis=(0, 1))
    extinction = -np.real(np.sum(solution.t[:, [0, 1], [0, 1]], axis=0))[columns]
    factor = 4 / (2 * np.pi * solution.n[-1].real / solution.wavelength)
    sca, ext = factor * np.mean(power), factor * np.mean(extinction)
    return dict(extinction=ext, scattering=sca, absorption=ext - sca,
                efficiency_extinction=ext / (2 * solution.radii[-1]),
                efficiency_scattering=sca / (2 * solution.radii[-1]),
                efficiency_absorption=(ext - sca) / (2 * solution.radii[-1]))


def cylinder_pattern(solution, phi, polarization="unpolarized"):
    phase = np.exp(1j * np.asarray(phi)[:, None] * solution.orders[None])
    amplitudes = np.einsum("pm,mij->pij", phase, solution.t)
    power = np.sum(np.abs(amplitudes) ** 2, axis=1)
    columns = [0, 1] if polarization == "unpolarized" else [0 if polarization == "axial-electric" else 1]
    return 2 / (np.pi * (2 * np.pi * solution.n[-1].real / solution.wavelength)) * np.mean(power[:, columns], axis=-1)
