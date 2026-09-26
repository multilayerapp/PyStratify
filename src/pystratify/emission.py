"""Far field and directivity of an electric or magnetic dipole inside or near a multilayered sphere.

The emission pattern follows from reciprocity: the far-field amplitude of a
dipole p at r0, observed in direction n with polarisation e, equals
p . E(r0), where E is the total field at r0 of a plane wave of polarisation
e incident from direction n (e exp(-i k n.r) in the host).  That plane-wave
solution is the one :func:`~pystratify.solve` computes, so emitters in the
core, in any shell (absorbing or not) and in the host are all covered, with
the phases between multipoles exact.  With the emitter on the local z axis
only m = 0 and m = +-1 multipoles radiate; any position and orientation is
reduced to that case by a rotation.

Normalisation: ``e_theta``, ``e_phi`` are components of F in
E_far = (k_h^2 / eps_h) exp(i k_h R) / R  F  (Gaussian units), so that in
the homogeneous host F = (n x p) x n exp(-i k_h n.r0) for an electric and
F = -(n x m) exp(-i k_h n.r0) for a magnetic dipole (E_far scaled by k_h^2
Z_h instead).  ``power`` is the radiated power over that of the same dipole
in the homogeneous host - the host-normalised radiative rate of
:func:`~pystratify.decay_rates` - and is summed analytically from the
multipole coefficients, not integrated over the requested directions.

For an emitter in the host the direct dipole field is added in closed form
and only the scattered field is expanded, so the truncation depends on the
particle's size and not on how far away the emitter is.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .decay import _tail_estimate, locate_shell
from .farfield import FarFieldPolarization, angular_functions
from .riccati import log_riccati
from .solver import TE, TM, solve

__all__ = ["EmissionPattern", "dipole_far_field"]

_TYPES = ("Ne", "No", "Me", "Mo")  # Bohren-Huffman N_e1n, N_o1n, M_e1n, M_o1n (and N_e0n for m = 0)


@dataclass(frozen=True)
class EmissionPattern(FarFieldPolarization):
    """Far field of a dipole emitter (see the module docstring for the normalisation).

    ``e_theta``, ``e_phi`` have the broadcast shape of ``theta`` and ``phi``;
    ``power`` is P_rad / P_0 with P_0 the power of the same dipole in the
    homogeneous host.
    """

    theta: np.ndarray
    phi: np.ndarray
    e_theta: np.ndarray
    e_phi: np.ndarray
    power: float
    position: np.ndarray
    moment: np.ndarray
    dipole: str
    shell: int
    orders_used: int
    converged: bool
    notes: tuple = field(default_factory=tuple)

    @property
    def power_density(self) -> np.ndarray:
        """(dP/dOmega) / P_0: integrates over the sphere to ``power``."""
        return 3 / (8 * np.pi) * self.intensity / np.sum(np.abs(self.moment) ** 2)

    @property
    def directivity(self) -> np.ndarray:
        """4 pi (dP/dOmega) / P_rad: 1 for an isotropic emitter, 1.5 at the maximum of a free dipole."""
        return 4 * np.pi * self.power_density / self.power


def _local_frame(position):
    """Rows: local x, y, z axes (global coordinates) with z along ``position``."""
    r = np.linalg.norm(position)
    z = position / r if r > 0 else np.array([0.0, 0.0, 1.0])
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = a - (a @ z) * z
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


def _spherical_basis(theta, phi):
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    n = np.stack([st * cp, st * sp, ct], -1)
    e_theta = np.stack([ct * cp, ct * sp, -st], -1)
    e_phi = np.stack([-sp, cp, np.zeros_like(sp)], -1)
    return n, e_theta, e_phi


def _coefficients(c, log_u, log_du, x):
    """(value, derivative, radial) terms c u/x, c u'/x, c u/x^2 of one multipole type."""
    with np.errstate(under="ignore", over="ignore"):
        return c * np.exp(log_u) / x, c * np.exp(log_du) / x, c * np.exp(log_u) / x**2


def _swap_for_magnetic(coef, factor):
    """H of the field E = sum [c_Ne N_e + c_No N_o + c_Me M_e + c_Mo M_o] swaps N and M;
    ``factor`` carries -(mu_h/n_h)/(i Z_d)."""
    swap = {"Ne": "Me", "No": "Mo", "Me": "Ne", "Mo": "No"}
    return {t: tuple(factor * v for v in coef[swap[t]]) for t in _TYPES}


def _interior_coefficients(sol, d, x, magnetic):
    """Multipole terms of the plane-wave field in (achiral) shell d at x = k_d r0."""
    l = sol.orders
    log_psi, log_xi = log_riccati(np.array([x]), l.size + 1)
    log_psi, log_xi = log_psi[0], log_xi[0]
    terms = {}
    for pol in (TM, TE):
        la, lb = sol.log_a[pol, d, 0], sol.log_b[pol, d, 0]
        with np.errstate(under="ignore", over="ignore"):
            u = np.exp(la + log_psi[l]) + np.exp(lb + log_xi[l])
            du = np.exp(la + log_psi[l - 1]) + np.exp(lb + log_xi[l - 1]) - l * u / x
        terms[pol] = (u / x, du / x, u / x**2)
    zero = (np.zeros(l.size, complex),) * 3
    coef = {"Ne": tuple(-1j * v for v in terms[TM]), "Mo": terms[TE], "No": zero, "Me": zero}
    if magnetic:
        n, mu = sol.n[0], sol.mu[0]
        coef = _swap_for_magnetic(coef, 1j * (mu[-1] * n[d]) / (n[-1] * mu[d]))
    return coef


def _host_coefficients(t_matrix, orders, x, magnetic):
    """Multipole terms at x = k_h r0 of the scattered field (T-matrix, any coupling) and of the incident wave."""
    l = orders
    log_psi, log_xi = log_riccati(np.array([x]), l.size + 1)
    log_psi, log_xi = log_psi[0], log_xi[0]
    t = t_matrix[0]  # (L, 2, 2), (TM, TE) = (N, M)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_du_xi = log_xi[l] + np.log(np.exp(log_xi[l - 1] - log_xi[l]) - l / x)
        log_du_psi = log_psi[l] + np.log(np.exp(log_psi[l - 1] - log_psi[l]) - l / x)
    # x-polarised incidence scatters into sum E_n [-i t_NN N_e + t_NM N_o - i t_MN M_e + t_MM M_o]
    c = {"Ne": -1j * t[:, TM, TM], "No": t[:, TM, TE], "Me": -1j * t[:, TE, TM], "Mo": t[:, TE, TE]}
    scattered = {name: _coefficients(v, log_xi[l], log_du_xi, x) for name, v in c.items()}
    zero = (np.zeros(l.size, complex),) * 3
    incident = {
        "Ne": _coefficients(-1j, log_psi[l], log_du_psi, x),
        "Mo": _coefficients(1.0, log_psi[l], log_du_psi, x),
        "No": zero,
        "Me": zero,
    }
    if magnetic:  # host: -(mu_h/n_h)/(i Z_h) = i
        scattered, incident = _swap_for_magnetic(scattered, 1j), _swap_for_magnetic(incident, 1j)
    return scattered, incident


def _weights(l):
    g = (-1j) ** l * (2 * l + 1) / (l * (l + 1))
    return g, g * l * (l + 1)


def _power_by_order(coef, p, l):
    """Contribution of each order to the integral of |F|^2 over the sphere.

    m = 0 (p_z) radiates sin(theta) pi_l in theta_hat and phi_hat; m = +-1
    (p_x -+ i p_y) the forms (tau theta_hat +- i pi phi_hat) and
    (+-i pi theta_hat - tau phi_hat); all are orthogonal on the sphere.
    """
    g, g_radial = _weights(l)
    e0, h0 = g_radial * coef["Ne"][2], g_radial * coef["No"][2]
    a1, a2 = g * coef["Mo"][0], g * coef["Ne"][1]
    b1, b2 = g * coef["Me"][0], g * coef["No"][1]
    q_plus, q_minus = abs(p[0] - 1j * p[1]) ** 2 / 4, abs(p[0] + 1j * p[1]) ** 2 / 4
    nu0 = 4 * np.pi * l * (l + 1) / (2 * l + 1)
    nu1 = nu0 * l * (l + 1)

    def sq(v):
        return np.abs(v) ** 2

    m0 = abs(p[2]) ** 2 * nu0 * (sq(e0) + sq(h0))
    plus = sq(-a2 + 1j * b1) + sq(b2 - 1j * a1)
    minus = sq(-a2 - 1j * b1) + sq(b2 + 1j * a1)
    return m0 + nu1 * (q_plus * plus + q_minus * minus)


def _assemble(coef, l, theta, phi, p):
    """Local far field (F_theta, F_phi) of a dipole p on the local z axis."""
    pi, tau = angular_functions(l, theta)
    g, g_radial = _weights(l)
    st, cp, sp = np.sin(theta), np.cos(phi), np.sin(phi)
    t_c = p[0] * cp + p[1] * sp
    t_s = p[1] * cp - p[0] * sp
    z = {t: g * coef[t][0] for t in _TYPES}
    d = {t: g * coef[t][1] for t in _TYPES}
    f_theta = (
        p[2] * st * ((g_radial * coef["Ne"][2]) @ pi)
        + t_c * (z["Mo"] @ pi - d["Ne"] @ tau)
        + t_s * (z["Me"] @ tau + d["No"] @ pi)
    )
    f_phi = (
        p[2] * st * ((g_radial * coef["No"][2]) @ pi)
        - t_c * (z["Me"] @ pi + d["No"] @ tau)
        + t_s * (z["Mo"] @ tau - d["Ne"] @ pi)
    )
    return f_theta, f_phi


def _add(a, b):
    return {t: tuple(u + v for u, v in zip(a[t], b[t])) for t in _TYPES}


def _starting_order(radii, n, wavelength):
    k = 2 * np.pi * np.abs(n) / wavelength
    x = max(float(np.max(k[:-1] * radii)), float(k[-1] * max(radii[-1], 0)))
    x = max(x, 1e-3)
    return int(x + 4 * x ** (1 / 3) + 2) + 8


def dipole_far_field(
    radii,
    n,
    wavelength,
    position,
    moment,
    theta,
    phi=0.0,
    mu=None,
    kappa=None,
    dipole="electric",
    l_max=None,
    tol=1e-10,
    l_cap=4000,
    warn=True,
) -> EmissionPattern:
    """Far-field emission pattern, directivity and radiated power of a dipole.

    Parameters
    ----------
    radii, n, wavelength, mu : geometry and materials at one wavelength (see
        :func:`~pystratify.solve`); the host must be lossless.
    position : (3,) Cartesian position of the emitter (same unit as ``radii``),
        anywhere: core, any shell (lossy ones included) or host; a point on an
        interface belongs to the outer shell.
    moment : (3,) dipole moment, complex for elliptical dipoles (e.g. (1, 1j, 0)).
    theta, phi : observation directions (broadcast together).
    kappa : chirality parameters per shell (host 0), see
        :func:`~pystratify.solve_chiral`; with a chiral particle the emitter
        must be in the host.
    dipole : ``'electric'`` or ``'magnetic'``.
    l_max : truncation; ``None`` starts from the particle's size and doubles
        until the power in the last orders is below ``tol`` (up to ``l_cap``).
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    n = np.atleast_1d(np.asarray(n, dtype=complex))
    mu = np.ones(n.size, dtype=complex) if mu is None else np.atleast_1d(np.asarray(mu, dtype=complex))
    position = np.asarray(position, dtype=float).ravel()
    moment = np.asarray(moment, dtype=complex).ravel()
    if dipole not in ("electric", "magnetic"):
        raise ValueError("dipole must be 'electric' or 'magnetic'")
    if np.ndim(wavelength) != 0:
        raise ValueError("dipole_far_field takes a single wavelength")
    if n.shape != (radii.size + 1,) or mu.shape != n.shape:
        raise ValueError(f"n and mu need shape ({radii.size + 1},), host last")
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position must be a finite 3-vector")
    if moment.shape != (3,) or not np.all(np.isfinite(moment)) or not np.any(moment):
        raise ValueError("moment must be a finite, nonzero 3-vector")
    if n[-1].imag != 0 or mu[-1].imag != 0 or n[-1].real <= 0:
        raise ValueError("the host must be lossless for a far field to exist")
    chiral = kappa is not None and np.any(np.asarray(kappa) != 0)
    magnetic = dipole == "magnetic"

    r0 = max(float(np.linalg.norm(position)), 1e-9 * radii[0])
    d = int(locate_shell(radii, r0))
    host = d == radii.size
    if chiral and not host:
        raise NotImplementedError("emitters inside a chiral particle are not supported; place the emitter in the host")
    frame = _local_frame(position)
    p_local = frame @ moment

    def coefficients(L):
        if chiral:
            from .chiral import solve_chiral

            sol = solve_chiral(radii, n, kappa, wavelength, mu, l_max=L)
        else:
            sol = solve(radii, n, wavelength, mu, l_max=L)
        x = sol.k[0, d] * r0
        l = sol.orders
        if not host:
            coef = _interior_coefficients(sol, d, x, magnetic)
            per_order = _power_by_order(coef, p_local, l)
            return l, coef, None, per_order, per_order
        scattered, incident = _host_coefficients(sol.t_matrix, l, x.real, magnetic)
        total = _add(scattered, incident)
        per_order = _power_by_order(total, p_local, l) - _power_by_order(incident, p_local, l)
        size_s, size_i = _power_by_order(scattered, p_local, l), _power_by_order(incident, p_local, l)
        return l, scattered, incident, per_order, size_s + 2 * np.sqrt(size_s * size_i)

    p2 = float(np.sum(np.abs(moment) ** 2))
    free = 8 * np.pi / 3 * p2
    L = int(l_max) if l_max is not None else min(l_cap, _starting_order(radii, n, wavelength))
    while True:
        l, coef, incident, per_order, bound = coefficients(L)
        integral = per_order.sum() + (free if host else 0.0)
        tail = _tail_estimate(bound[-4:, None])[0] if L >= 4 else np.inf
        converged = bool(np.isfinite(integral) and tail <= tol * abs(integral))
        if l_max is not None or converged or L >= l_cap:
            break
        L = min(2 * L, l_cap)

    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    n_hat, e_theta, e_phi = _spherical_basis(theta.ravel(), phi.ravel())
    n_local = n_hat @ frame.T
    theta_l = np.arccos(np.clip(n_local[:, 2], -1.0, 1.0))
    phi_l = np.arctan2(n_local[:, 1], n_local[:, 0])
    f_theta, f_phi = _assemble(coef, l, theta_l, phi_l, p_local)
    _, et_l, ep_l = _spherical_basis(theta_l, phi_l)
    far = (f_theta[:, None] * et_l + f_phi[:, None] * ep_l) @ frame
    if host:  # direct field in closed form
        k_h = 2 * np.pi * n[-1].real / wavelength
        phase = np.exp(-1j * k_h * (n_hat @ position))[:, None]
        if magnetic:
            far = far - np.cross(n_hat, moment) * phase
        else:
            far = far + (moment - (n_hat @ moment)[:, None] * n_hat) * phase
    notes = ()
    if not converged and l_max is None:
        notes = (f"multipole sum not converged to tol={tol:g} with l_max={L}",)
        if warn:
            warnings.warn(notes[0], RuntimeWarning, stacklevel=2)
    return EmissionPattern(
        theta=theta,
        phi=phi,
        e_theta=np.sum(far * e_theta, -1).reshape(theta.shape),
        e_phi=np.sum(far * e_phi, -1).reshape(theta.shape),
        power=float(integral / free),
        position=position,
        moment=moment,
        dipole=dipole,
        shell=d,
        orders_used=int(L),
        converged=converged,
        notes=notes,
    )
