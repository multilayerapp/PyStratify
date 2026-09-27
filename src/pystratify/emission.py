"""Far field of electric, magnetic and chiral (electric + magnetic) dipole emitters in or near a multilayered sphere.

The emission pattern follows from reciprocity: the far-field amplitude of a
source (p, m) at r0, observed in direction n with polarisation e, is

    F . e = p . E(r0) - m . H(r0),

with E, H the total fields at r0 of a plane wave e exp(-i k_h n.r) incident
from n (Gaussian units).  That plane-wave solution is what
:func:`~pystratify.solve` and :func:`~pystratify.solve_chiral` compute, so
emitters in the core, in any shell - absorbing, magnetic or chiral - and in
the host are covered, with exact phases between multipoles.  Pasteur media
are reciprocal, so the relation holds inside chiral layers too.  With the
emitter on the local z axis only m = 0, +-1 multipoles radiate; any position
is reduced to that case by a rotation.

Normalisation (Gaussian units): E_far = (k_h^2 / eps_h) exp(i k_h R)/R F, so
that in the homogeneous host F = (n x p) x n exp(-i k_h n.r0) - (n_h/mu_h)
(n x m) exp(-i k_h n.r0).  A magnetic dipole alone (``dipole='magnetic'``)
keeps its own normalisation E_far = (k_h^2 / n_h) exp(i k_h R)/R F with
F = -(n x m) exp(-i k_h n.r0).  m is the dual of p (Moroz 2005, as in
:func:`~pystratify.decay_rates`); ``magnetic_convention='current'`` takes a
current-loop moment m_A, which acts as mu_d m_A plus the electric dipole
i kappa_d m_A in a layer with mu_d, kappa_d.  In SI pass m / c.  Sources with an
electric quadrupole are synthesised from the Green's function instead
(:mod:`pystratify.rates`), in the same normalisation.

``power`` is the radiated power over that of the same source in the
homogeneous host (the host-normalised radiative rate); it and the helicity-
resolved powers are summed analytically from orthogonal multipole
coefficients.  For an emitter in the host the direct field is added in
closed form, so the truncation depends on the particle, not the distance.

Orientation averages (``orientation='isotropic'`` or an axis for rotations
about it, e.g. the normal of a 2D material) rotate p and m rigidly and use
<(Rp)(Rm)^H> = (p . m*) I / 3 (isotropic); the result is a partially
polarised pattern given by its coherency matrix.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .decay import _tail_estimate, locate_shell
from .farfield import angular_functions
from .riccati import log_riccati
from .solver import TE, TM, solve

__all__ = ["EmissionPattern", "dipole_far_field", "source_covariance"]

_TYPES = ("Ne", "No", "Me", "Mo")  # Bohren-Huffman N_e1n, N_o1n, M_e1n, M_o1n (and N_e0n for m = 0)
_SIGN = np.array([1.0, -1.0])


@dataclass(frozen=True)
class EmissionPattern:
    """Far field of a dipole source (see the module docstring for the normalisation).

    ``coherency``: <F_a F_b*> for a, b in (theta, phi), shape S + (2, 2) with S
    the broadcast shape of ``theta`` and ``phi``.  ``e_theta``, ``e_phi``:
    the amplitudes themselves for a fixed (coherent) source, ``None`` for an
    orientation average.  ``power`` = P_rad / P_0 and ``helicity_power`` =
    (P_+, P_-) (summing to ``power``), with P_0 the power of the same source
    in the homogeneous host.
    """

    theta: np.ndarray
    phi: np.ndarray
    coherency: np.ndarray
    e_theta: np.ndarray | None
    e_phi: np.ndarray | None
    power: float
    helicity_power: tuple
    reference: float
    position: np.ndarray
    moment: np.ndarray
    magnetic_moment: np.ndarray | None
    dipole: str
    orientation: str
    shell: int
    orders_used: int
    converged: bool
    notes: tuple = field(default_factory=tuple)
    quadrupole: np.ndarray | None = None

    @property
    def intensity(self) -> np.ndarray:
        """<|F|^2>."""
        return np.real(self.coherency[..., 0, 0] + self.coherency[..., 1, 1])

    @property
    def power_density(self) -> np.ndarray:
        """(dP/dOmega) / P_0: integrates over the sphere to ``power``."""
        return self.intensity / self.reference

    @property
    def directivity(self) -> np.ndarray:
        """4 pi (dP/dOmega) / P_rad: 1 for an isotropic emitter, 1.5 at the maximum of a free dipole."""
        return 4 * np.pi * self.power_density / self.power

    @property
    def helicity_intensity(self) -> tuple:
        """(<|e_+|^2>, <|e_-|^2>) on (theta_hat +- i phi_hat)/sqrt(2)."""
        total = self.intensity
        chiral = 2 * np.imag(self.coherency[..., 0, 1])
        return (total - chiral) / 2, (total + chiral) / 2

    @property
    def helicity(self) -> tuple:
        """Amplitudes (e_+, e_-) of a fixed source (helicity +1 and -1 along the outgoing direction)."""
        if self.e_theta is None:
            raise ValueError("an orientation average has no amplitudes; use helicity_intensity")
        return (self.e_theta - 1j * self.e_phi) / np.sqrt(2), (self.e_theta + 1j * self.e_phi) / np.sqrt(2)

    @property
    def circular_polarization(self) -> np.ndarray:
        """Degree of circular polarisation (I_+ - I_-) / (I_+ + I_-) per direction; g_lum = 2 x this."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return -2 * np.imag(self.coherency[..., 0, 1]) / self.intensity

    @property
    def dissymmetry(self) -> float:
        """Luminescence dissymmetry of the total emission, g = 2 (P_+ - P_-) / (P_+ + P_-)."""
        plus, minus = self.helicity_power
        return 2 * (plus - minus) / (plus + minus)

    def stokes(self) -> np.ndarray:
        """Stokes parameters (I, Q, U, V) on the last axis, Bohren & Huffman's convention
        (par = theta_hat, perp = -phi_hat, as :func:`~pystratify.mueller_matrix`): V = -I x
        ``circular_polarization``."""
        j = self.coherency
        return np.stack(
            [
                self.intensity,
                np.real(j[..., 0, 0] - j[..., 1, 1]),
                -2 * np.real(j[..., 0, 1]),
                2 * np.imag(j[..., 0, 1]),
            ],
            axis=-1,
        )


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
    """Multipole terms of the plane-wave field (E, or H for ``magnetic``) in achiral shell d at x = k_d r0."""
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


def _log_derivative(log_f, l, x):
    """log of f_l' = f_{l-1} - l f_l / x for Riccati functions f given as logs (orders 0..)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = log_f[..., l - 1], np.log(-l / x[..., None]) + log_f[..., l]
        m = np.maximum(a.real, b.real)
        m = np.where(np.isfinite(m), m, 0.0)
        return m + np.log(np.exp(a - m) + np.exp(b - m))


def _chiral_interior_coefficients(csol, d, r0):
    """Multipole terms of E and of -(mu_h/n_h) H (x-polarised plane wave) in layer d of a chiral solution.

    In layer d the field is sum_c (alpha_c psi_l(k_c r) + beta_c xi_l(k_c r)) (M + c N) and
    H = (1 / (i Z_d)) sum_c c (...)(M + c N); the incident x-polarised wave has helicity
    amplitudes (-i/2, i/2) in the e pair (N_e1n, M_e1n) and (1/2, 1/2) in the o pair.
    """
    from .chiral import log_matmul

    l = csol.orders
    k = csol.k_helicity[0, d]  # (2,)
    x = k * r0
    log_psi, log_xi = log_riccati(x, l.size + 1)  # (2, L + 2)
    lp, lx = log_psi[:, l], log_xi[:, l]  # (2, L)
    lpd, lxd = _log_derivative(log_psi, l, x), _log_derivative(log_xi, l, x)
    log_alpha = csol.log_alpha[0, d]  # (L, 2 channels, 2 helicities)
    log_beta = log_matmul(csol.log_r[0, d], log_alpha)
    with np.errstate(under="ignore", over="ignore"):
        u = np.exp(log_alpha + lp.T[:, :, None]) + np.exp(log_beta + lx.T[:, :, None])  # (L, c, h)
        du = np.exp(log_alpha + lpd.T[:, :, None]) + np.exp(log_beta + lxd.T[:, :, None])
    pairs = {"e": np.array([-0.5j, 0.5j]), "o": np.array([0.5, 0.5])}
    n, mu = csol.n[0], csol.mu[0]
    z_d = mu[d] / n[d]
    h_factor = -(mu[-1] / n[-1]) / (1j * z_d)
    zero = np.zeros(l.size, complex)
    e_coef, h_coef = {}, {}
    for pair, weights in pairs.items():
        v = (u @ weights) / x  # (L, c): channel values u/x for this pair
        dv = (du @ weights) / x
        m_part, n_value, n_radial = v.sum(-1), dv @ _SIGN, (v / x) @ _SIGN
        e_coef["N" + pair] = (zero, n_value, n_radial)
        e_coef["M" + pair] = (m_part, zero, zero)
        h_coef["N" + pair] = (zero, h_factor * dv.sum(-1), h_factor * (v / x).sum(-1))
        h_coef["M" + pair] = (h_factor * (v @ _SIGN), zero, zero)
    return e_coef, h_coef


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


def _modes(coef, p, l):
    """Orthogonal far-field mode amplitudes (L, 6) of a local dipole p on the z axis:

    F = sum_l [A0t sin(theta) pi_l theta_hat + A0p sin(theta) pi_l phi_hat]
        + e^{i phi} [A+N (tau theta_hat + i pi phi_hat) + A+M (i pi theta_hat - tau phi_hat)]
        + e^{-i phi} [A-N (tau theta_hat - i pi phi_hat) + A-M (-i pi theta_hat - tau phi_hat)].
    """
    g = (-1j) ** l * (2 * l + 1) / (l * (l + 1))
    e0, h0 = g * l * (l + 1) * coef["Ne"][2], g * l * (l + 1) * coef["No"][2]
    a1, a2 = g * coef["Mo"][0], g * coef["Ne"][1]
    b1, b2 = g * coef["Me"][0], g * coef["No"][1]
    q_plus, q_minus = (p[0] - 1j * p[1]) / 2, (p[0] + 1j * p[1]) / 2
    return np.stack(
        [
            p[2] * e0,
            p[2] * h0,
            q_plus * (-a2 + 1j * b1),
            q_plus * (b2 - 1j * a1),
            q_minus * (-a2 - 1j * b1),
            q_minus * (b2 + 1j * a1),
        ],
        axis=-1,
    )


def _mode_matrix(e_coef, h_coef, frame, l):
    """(L, 6 modes, 6 sources) for unit p and m' along the global x, y, z."""
    return np.stack(
        [_modes(e_coef, frame[:, i], l) for i in range(3)] + [_modes(h_coef, frame[:, i], l) for i in range(3)],
        axis=-1,
    )


def _powers(modes, cov, l):
    """Per-order (total, helicity +1, helicity -1) power of mode matrices (L, 6, K) and source covariance (K, K)."""
    c = np.einsum("lak,kq,lbq->lab", modes, cov, np.conj(modes))
    nu0 = 4 * np.pi * l * (l + 1) / (2 * l + 1)
    nu1 = nu0 * l * (l + 1)
    diag = np.real(np.einsum("laa->la", c))
    total = nu0 * (diag[:, 0] + diag[:, 1]) + nu1 * diag[:, 2:].sum(-1)
    chiral = nu0 * np.imag(c[:, 0, 1]) - nu1 * (np.imag(c[:, 2, 3]) + np.imag(c[:, 4, 5]))
    return total, (total - 2 * chiral) / 2, (total + 2 * chiral) / 2


def _far_field(modes, theta, phi):
    """Local (F_theta, F_phi), each (directions, K), from mode matrices (L, 6, K)."""
    pi, tau = angular_functions(np.arange(1, modes.shape[0] + 1), theta)
    pi, tau = pi.T, tau.T
    st, ep, em = np.sin(theta)[:, None], np.exp(1j * phi)[:, None], np.exp(-1j * phi)[:, None]
    a = [modes[:, j, :] for j in range(6)]
    f_theta = st * (pi @ a[0]) + ep * (tau @ a[2] + 1j * pi @ a[3]) + em * (tau @ a[4] - 1j * pi @ a[5])
    f_phi = st * (pi @ a[1]) + ep * (1j * pi @ a[2] - tau @ a[3]) + em * (-1j * pi @ a[4] - tau @ a[5])
    return f_theta, f_phi


def source_covariance(moment, magnetic_moment=None, orientation="fixed"):
    """<s s^H> for s = (p, m) (6 components), for a fixed source or averaged over its orientations.

    ``orientation``: ``'fixed'``; ``'isotropic'`` (p and m rotated rigidly over all
    directions: <(Rv)(Rw)^H> = (v . w*) I / 3); or a 3-vector axis for rotations
    about it (e.g. the normal of a 2D material: in-plane dipoles of random azimuth).
    """
    p = np.asarray(moment, dtype=complex).ravel()
    m = np.zeros(3, complex) if magnetic_moment is None else np.asarray(magnetic_moment, dtype=complex).ravel()
    vectors = (p, m)
    cov = np.zeros((6, 6), dtype=complex)
    if isinstance(orientation, str) and orientation not in ("fixed", "isotropic"):
        raise ValueError("orientation must be 'fixed', 'isotropic' or a nonzero 3-vector axis")
    if isinstance(orientation, str) and orientation == "fixed":
        s = np.concatenate(vectors)
        return np.outer(s, np.conj(s))
    if isinstance(orientation, str) and orientation == "isotropic":

        def average(v, w):
            return np.vdot(w, v) * np.eye(3) / 3

    else:
        axis = np.asarray(orientation, dtype=float).ravel()
        if axis.shape != (3,) or not np.linalg.norm(axis) > 0:
            raise ValueError("orientation must be 'fixed', 'isotropic' or a nonzero 3-vector axis")
        axis = axis / np.linalg.norm(axis)
        project = np.eye(3) - np.outer(axis, axis)

        def average(v, w):
            along = (axis @ v) * np.conj(axis @ w) * np.outer(axis, axis)
            return along + 0.5 * (
                np.outer(project @ v, np.conj(project @ w)) + np.outer(np.cross(axis, v), np.conj(np.cross(axis, w)))
            )

    for i, v in enumerate(vectors):
        for j, w in enumerate(vectors):
            cov[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = average(v, w)
    return cov


def _starting_order(radii, n, wavelength):
    k = 2 * np.pi * np.abs(n) / wavelength
    x = max(float(np.max(k[:-1] * radii)), float(k[-1] * max(radii[-1], 0)))
    x = max(x, 1e-3)
    return int(x + 4 * x ** (1 / 3) + 2) + 8


def _with_quadrupole(
    radii, n, wavelength, position, moment, theta, phi, mu, kappa, l_max, tol, l_cap, warn,
    magnetic_moment, orientation, sheets, magnetic_convention, quadrupole,
):  # fmt: skip
    """dipole_far_field for sources with an electric quadrupole, via the Green's function route."""
    from .rates import _emission

    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    kappa = None if kappa is None else np.atleast_1d(np.asarray(kappa, dtype=complex))
    rates, pattern = _emission(
        radii, n, wavelength, position, moment, magnetic_moment, orientation, mu, kappa, "electric", "host",
        l_max, tol, l_cap, None, warn, sheets, magnetic_convention, quadrupole,
        directions=(theta.ravel(), phi.ravel()),
    )  # fmt: skip
    fields = pattern["fields"]  # (D, K, 2)
    coherency = np.einsum("dka,dkb->dab", fields, np.conj(fields)).reshape(theta.shape + (2, 2))
    fixed = rates.orientation == "fixed"
    return EmissionPattern(
        theta=theta,
        phi=phi,
        coherency=coherency,
        e_theta=fields[:, 0, 0].reshape(theta.shape) if fixed else None,
        e_phi=fields[:, 0, 1].reshape(theta.shape) if fixed else None,
        power=float(rates.radiative),
        helicity_power=(float(rates.radiative_helicity[0]), float(rates.radiative_helicity[1])),
        reference=pattern["reference"],
        position=position,
        moment=moment,
        magnetic_moment=magnetic_moment,
        dipole="electric",
        orientation=rates.orientation,
        shell=int(rates.shell),
        orders_used=int(rates.orders_used),
        converged=bool(rates.converged),
        notes=rates.notes,
        quadrupole=np.asarray(quadrupole, dtype=complex),
    )


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
    magnetic_moment=None,
    orientation="fixed",
    sheets=None,
    magnetic_convention="dual",
    quadrupole=None,
) -> EmissionPattern:
    """Far-field emission pattern, directivity, radiated power and helicity content of a dipole source.

    Parameters
    ----------
    radii, n, wavelength, mu : geometry and materials at one wavelength (see
        :func:`~pystratify.solve`); the host must be lossless.
    position : (3,) Cartesian position of the emitter, anywhere: core, any shell
        (lossy, magnetic or chiral) or host; a point on an interface belongs to the
        outer shell.
    moment : (3,) dipole moment, complex for elliptical dipoles, e.g. (1, 1j, 0)
        for a valley exciton of a 2D material lying in the xy plane.  Electric,
        unless ``dipole='magnetic'``.
    magnetic_moment : (3,) magnetic moment m emitted coherently with the electric
        ``moment`` p - a chiral emitter when Im(p . m*) != 0 (Gaussian units, m in
        the units of p; SI: m / c).
    orientation : ``'fixed'``, ``'isotropic'`` (random orientation, p and m rigidly
        together) or an axis vector (random rotation about it).
    theta, phi : observation directions (broadcast together).
    kappa : chirality parameters per shell (host 0), see :func:`~pystratify.solve_chiral`.
    l_max : truncation; ``None`` starts from the particle's size and doubles
        until the power in the last orders is below ``tol`` (up to ``l_cap``).
    sheets : 2D materials on interfaces, ``{j: Sheet(...)}`` (see :mod:`pystratify.sheets`).
    magnetic_convention : ``'dual'`` (default) - m is the dual moment, the magnetic
        current -i omega m of the Maxwell equations - or ``'current'`` - m is a
        current-loop (Amperian) moment m_A, which in a layer with mu_d, kappa_d acts as
        the dual moment mu_d m_A plus the electric dipole i kappa_d m_A (D = eps E +
        i kappa H).  The two differ only where mu != 1 or kappa != 0; the reference
        power is that of the same physical source in the host.
    quadrupole : (3, 3) electric quadrupole moment (Jackson's Q; see
        :func:`~pystratify.emission_rates`), emitted coherently with ``moment`` and
        ``magnetic_moment``.  The pattern is then synthesised from the outgoing host
        amplitudes of the Green's function (same normalisation; the route agrees with the
        reciprocity route above to ~1e-15 for dipoles) and the emitter must sit in a
        lossless layer.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    n = np.atleast_1d(np.asarray(n, dtype=complex))
    mu = np.ones(n.size, dtype=complex) if mu is None else np.atleast_1d(np.asarray(mu, dtype=complex))
    position = np.asarray(position, dtype=float).ravel()
    moment = np.asarray(moment, dtype=complex).ravel()
    if dipole not in ("electric", "magnetic"):
        raise ValueError("dipole must be 'electric' or 'magnetic'")
    if magnetic_moment is not None:
        if dipole != "electric":
            raise ValueError("with magnetic_moment, moment is the electric dipole: leave dipole='electric'")
        magnetic_moment = np.asarray(magnetic_moment, dtype=complex).ravel()
        if magnetic_moment.shape != (3,) or not np.all(np.isfinite(magnetic_moment)):
            raise ValueError("magnetic_moment must be a finite 3-vector")
    if np.ndim(wavelength) != 0:
        raise ValueError("dipole_far_field takes a single wavelength")
    if n.shape != (radii.size + 1,) or mu.shape != n.shape:
        raise ValueError(f"n and mu need shape ({radii.size + 1},), host last")
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position must be a finite 3-vector")
    if moment.shape != (3,) or not np.all(np.isfinite(moment)):
        raise ValueError("moment must be a finite 3-vector")
    if not (np.any(moment) or (magnetic_moment is not None and np.any(magnetic_moment)) or quadrupole is not None):
        raise ValueError("the source must have a nonzero moment")
    if n[-1].imag != 0 or mu[-1].imag != 0 or n[-1].real <= 0:
        raise ValueError("the host must be lossless for a far field to exist")
    chiral = kappa is not None and np.any(np.asarray(kappa) != 0)
    if magnetic_convention not in ("dual", "current"):
        raise ValueError("magnetic_convention must be 'dual' or 'current'")
    if quadrupole is not None:
        if dipole != "electric":
            raise ValueError("with a quadrupole pass the magnetic dipole as magnetic_moment (dipole='electric')")
        return _with_quadrupole(
            radii, n, wavelength, position, moment, theta, phi, mu, kappa, l_max, tol, l_cap, warn,
            magnetic_moment, orientation, sheets, magnetic_convention, quadrupole,
        )  # fmt: skip

    r0 = max(float(np.linalg.norm(position)), 1e-9 * radii[0])
    d = int(locate_shell(radii, r0))
    host = d == radii.size
    # a current-loop moment m_A is the dual moment mu m_A of the medium it sits in: mu_d at the
    # emitter, mu_h for the reference (the same source in the unbounded host)
    at_source, in_host = (mu[d].real, mu[-1].real) if magnetic_convention == "current" else (1.0, 1.0)
    kappa_d = 0.0
    if magnetic_convention == "current" and kappa is not None:
        kappa_d = np.atleast_1d(np.asarray(kappa, dtype=complex))[d].real  # a loop in a chiral layer: p += i kappa m_A

    # source vector s = (p, m') in the units of F: m' = (n_h/mu_h) m with an electric
    # dipole, m' = m for a magnetic dipole alone (its own normalisation)
    if dipole == "magnetic":
        if kappa_d:
            raise ValueError(
                "a current loop in a chiral layer also radiates as an electric dipole: "
                "pass it as magnetic_moment with moment=(0, 0, 0)"
            )
        p, m_scaled = np.zeros(3, complex), moment * at_source
        m_reference = moment * in_host
    else:
        m = np.zeros(3, complex) if magnetic_moment is None else magnetic_moment
        p, m_scaled = moment + 1j * kappa_d * m, (n[-1] / mu[-1]).real * m * at_source
        m_reference = (n[-1] / mu[-1]).real * m * in_host
    cov = source_covariance(p, m_scaled, orientation)
    orientation_name = orientation if isinstance(orientation, str) else "axis"
    p_reference = moment if dipole == "electric" else np.zeros(3, complex)
    reference = 8 * np.pi / 3 * float(np.real(np.trace(source_covariance(p_reference, m_reference, orientation))))
    frame = _local_frame(position)

    def modes(L):
        if chiral:
            from .chiral import solve_chiral

            sol = solve_chiral(radii, n, kappa, wavelength, mu, l_max=L, sheets=sheets)
        else:
            sol = solve(radii, n, wavelength, mu, l_max=L, sheets=sheets)
        l = sol.orders
        if host:
            x = (sol.k[0, d] * r0).real
            (e_sca, e_inc), (h_sca, h_inc) = (_host_coefficients(sol.t_matrix, l, x, flag) for flag in (False, True))
            return l, _mode_matrix(e_sca, h_sca, frame, l), _mode_matrix(e_inc, h_inc, frame, l)
        if chiral:
            e_coef, h_coef = _chiral_interior_coefficients(sol, d, r0)
        else:
            x = sol.k[0, d] * r0
            e_coef, h_coef = _interior_coefficients(sol, d, x, False), _interior_coefficients(sol, d, x, True)
        return l, _mode_matrix(e_coef, h_coef, frame, l), None

    L = int(l_max) if l_max is not None else min(l_cap, _starting_order(radii, n, wavelength))
    while True:
        l, particle, direct = modes(L)
        if direct is None:
            per_order = _powers(particle, cov, l)
            bound = per_order[0]
            powers = [float(v.sum()) for v in per_order]
        else:
            total, free = _powers(particle + direct, cov, l), _powers(direct, cov, l)
            scattered = _powers(particle, cov, l)[0]
            bound = scattered + 2 * np.sqrt(scattered * np.maximum(free[0], 0))
            chiral_free = 2 * np.imag(np.trace(cov[:3, 3:])) * 4 * np.pi / 3  # (4 pi/3)|p +- i m'|^2
            analytic = (reference, reference / 2 + chiral_free, reference / 2 - chiral_free)
            powers = [a + float((t - f).sum()) for a, t, f in zip(analytic, total, free)]
        tail = _tail_estimate(bound[-4:, None])[0] if L >= 4 else np.inf
        converged = bool(np.isfinite(powers[0]) and tail <= tol * abs(powers[0]))
        if l_max is not None or converged or L >= l_cap:
            break
        L = min(2 * L, l_cap)

    # the sources whose patterns are needed: s itself, or sqrt(lambda) v of the covariance's eigenpairs
    if orientation_name == "fixed":
        sources = np.concatenate([p, m_scaled])[:, None]
    else:
        weights, vectors = np.linalg.eigh(cov)
        keep = weights > 1e-14 * weights.max()
        sources = vectors[:, keep] * np.sqrt(weights[keep])  # cov = sources sources^H

    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    n_hat, e_theta, e_phi = _spherical_basis(theta.ravel(), phi.ravel())
    n_local = n_hat @ frame.T
    theta_l = np.arccos(np.clip(n_local[:, 2], -1.0, 1.0))
    phi_l = np.arctan2(n_local[:, 1], n_local[:, 0])
    f_theta, f_phi = _far_field(particle @ sources, theta_l, phi_l)  # (D, K): the particle's part for a host emitter
    _, et_l, ep_l = _spherical_basis(theta_l, phi_l)
    far = (f_theta[:, :, None] * (et_l @ frame)[:, None, :]) + (f_phi[:, :, None] * (ep_l @ frame)[:, None, :])
    if host:  # direct field in closed form: p - (n.p) n - n x m'
        k_h = 2 * np.pi * n[-1].real / wavelength
        phase = np.exp(-1j * k_h * (n_hat @ position))[:, None, None]
        p_k, m_k = sources[:3].T, sources[3:].T  # (K, 3)
        along = (n_hat @ sources[:3])[:, :, None] * n_hat[:, None, :]
        far = far + (p_k[None] - along - np.cross(n_hat[:, None, :], m_k[None])) * phase
    fields = np.stack([np.einsum("dkg,dg->dk", far, e_theta), np.einsum("dkg,dg->dk", far, e_phi)], axis=-1)
    coherency = np.einsum("dka,dkb->dab", fields, np.conj(fields)).reshape(theta.shape + (2, 2))
    amplitudes = fields[:, 0, :] if orientation_name == "fixed" else None
    notes = ()
    if not converged and l_max is None:
        notes = (f"multipole sum not converged to tol={tol:g} with l_max={L}",)
        if warn:
            warnings.warn(notes[0], RuntimeWarning, stacklevel=2)
    return EmissionPattern(
        theta=theta,
        phi=phi,
        coherency=coherency,
        e_theta=None if amplitudes is None else amplitudes[:, 0].reshape(theta.shape),
        e_phi=None if amplitudes is None else amplitudes[:, 1].reshape(theta.shape),
        power=powers[0] / reference,
        helicity_power=(powers[1] / reference, powers[2] / reference),
        reference=reference,
        position=position,
        moment=moment,
        magnetic_moment=magnetic_moment,
        dipole=dipole,
        orientation=orientation_name,
        shell=d,
        orders_used=int(L),
        converged=converged,
        notes=notes,
    )
