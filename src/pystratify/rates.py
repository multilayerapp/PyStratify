"""Decay rates of electric, magnetic, chiral (electric + magnetic) and electric-quadrupole
emitters in or near multilayered spheres whose layers may be chiral and whose interfaces may
carry 2D sheets.

This generalises :func:`~pystratify.decay_rates` (achiral layers, radial or
tangential electric or magnetic dipoles) to any source (p, m, Q) - complex,
coherent, fixed or orientation-averaged - to Pasteur layers
(:func:`~pystratify.solve_chiral`) and to sheets (:mod:`pystratify.sheets`): the
setting of Guzatov & Klimov, New J. Phys. 14, 123009 (2012), for an arbitrary
multilayer.

Theory (Gaussian units, exp(-i omega t); m is the dual magnetic moment, as in
:func:`~pystratify.dipole_far_field`).  In the emitter's layer d (lossless,
possibly chiral, Z_d = mu_d/n_d) a source drives each Beltrami field
Q_s = E + s i Z_d H (s = +-1, wavenumber k_s) through the functional
q_s . W + (1/6) Q : grad W of the modes W_s = M + s N at r0, q_s = p + i s m / Z_d,
and delivers the power P = (omega/4) Im sum_s [q_s* . Q_s + (1/6) Q* : grad Q_s](r0).
The direct field follows from (curl + s k_s) G_0(k_s) and the dyadic expansion of
Tai; with the emitter on the local z axis only the axial families m = 0, +-1
(and +-2 for a quadrupole) couple.  Per order the source's field in layer d is
psi waves A + D_psi below r0 and xi waves B + D_xi above it, with D the direct
parts; the regular solution below (B = R (A + D_psi)) and the outgoing one above
(A = S (B + D_xi)) give

    B = (1 - R S)^-1 R (D_psi + S D_xi),   A = S (B + D_xi),

with R, S the 2x2 helicity matrices of the layer, evaluated scaled at r0 so
every factor is O(1).  Then

* total rate: free power of the layer (closed form: sum_s k_s^2 (|q_s|^2 +
  (k_s^2/120) sum |Q_ij|^2) up to constants) plus the reflected part at r0;
* radiative rate: the xi amplitudes above r0 carried to the host by the
  interface maps of :class:`~pystratify.ChiralSolution`; host channel s is
  helicity s, so the helicity content comes for free.  For an emitter in the
  host the free part is added in closed form;
* absorption per layer a: the field carried to layer a (inwards as a regular,
  outwards as an outgoing solution) and the loss density
  (omega/8 pi)[eps'' |E|^2 + mu'' |H|^2 - 2 kappa'' Im(E* . H)] integrated by
  Gauss-Legendre quadrature, radially per order in the basis
  (psi_+, psi_-, xi_+, xi_-) with per-function logarithmic scales and over
  angles in closed form; per sheet, (c/8 pi)[Re sigma |<E_t>|^2 +
  k0 Im zeta |<D_n>|^2] from the fields on its two sides.

The three are computed independently, so ``balance_error`` = |total -
radiative - nonradiative| / total tests energy conservation.  Rates are
normalised to the power of the same source in the unbounded host (or, with
``normalization='layer'``, in the unbounded medium of its own layer, chiral
if that layer is).  Within ~1 nm of a lossless interface the reflected part
of the total is the small real part of large evanescent terms: rounding then
limits ``total`` (not ``radiative`` or ``nonradiative``) of this logarithmic
route to ~1e-9 relative at 1 nm, ~1e-5 at 0.1 nm (far worse for quadrupoles), and
the l-sums stop at that rounding floor; the route flags it through the energy balance.

For achiral layers, with or without sheets, the default route (``route='auto'``) takes the
total, the frequency shift and the radiative rates from the normalized formulation
(:mod:`pystratify.normalized`) instead: each source functional is split into its TM
(odd in s) and TE (even in s) parts, value and derivative coefficients c_v, c_d, and
the reflected power is sum_p P_p (W_vv S + 2 W_vd S^m + W_dd S^d) with the real
weights W_vv = sum |c_v|^2, W_vd = sum Re(c_v* c_d), W_dd = sum |c_d|^2 over the
sources, which keeps full precision at any distance; the helicity of the radiation
comes from the TM and TE amplitudes in the host.  Absorption per layer stays on the
logarithmic route, so the energy balance remains an independent test.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np

from .chiral import _log, _log_add, log_matmul, solve_chiral
from .convergence import truncation_order
from .convergence import tail_estimate
from .decay import _orders_needed, _tail_estimate, locate_shell
from .emission import _local_frame, source_covariance
from .energy import gauss_legendre
from .riccati import log_riccati
from .sheets import _feibelman_arrays, _sheet_arrays
from .solver import TE, TM, solve

__all__ = ["EmissionRates", "emission_rates"]

_SIGN = np.array([1.0, -1.0])
_BASIS_SIGN = np.array([1.0, -1.0, 1.0, -1.0])  # helicity of (psi_+, psi_-, xi_+, xi_-)
_CHUNK = 2_000_000  # complex numbers per (positions x sources x orders x 6) block


@dataclass(frozen=True)
class EmissionRates:
    """Decay rates of a dipole (and quadrupole) source, normalised to its free power (see ``normalization``).

    Arrays over the emitter positions (P,): ``total`` (the Purcell factor with
    ``normalization='host'``), ``radiative``, ``radiative_helicity`` (P, 2) =
    (P_+, P_-) of the far field (helicity +1 is left-circular in the optics
    convention, as :func:`~pystratify.dipole_far_field`), ``absorption`` (P,
    N + 1) = power absorbed per layer (host last, always 0), ``sheet_absorption``
    (P, N) = power absorbed by the 2D sheet on each interface, ``free_in_layer``
    = free power of the source in the unbounded medium of its own layer over
    that in the host (depends on the source's handedness in a chiral layer).
    ``shift`` (P,) is the frequency shift (omega - omega_0) in units of the same
    free power (exp(-i omega t); the free self-energy is part of omega_0), NaN on
    the logarithmic route; ``route`` is ``'normalized'`` or ``'log'``.
    """

    position: np.ndarray
    total: np.ndarray
    radiative: np.ndarray
    radiative_helicity: np.ndarray
    absorption: np.ndarray
    sheet_absorption: np.ndarray
    free_in_layer: np.ndarray
    shell: np.ndarray
    orders_used: int
    converged: np.ndarray
    normalization: str
    orientation: str
    notes: tuple = field(default_factory=tuple)
    shift: np.ndarray | None = None
    route: str = "log"

    @property
    def nonradiative(self) -> np.ndarray:
        """Total absorbed power (all layers and sheets)."""
        return self.absorption.sum(axis=-1) + self.sheet_absorption.sum(axis=-1)

    @property
    def balance_error(self) -> np.ndarray:
        """|total - (radiative + nonradiative)| / total: energy conservation, computed independently."""
        return np.abs(self.total - self.radiative - self.nonradiative) / np.abs(self.total)

    @property
    def dissymmetry(self) -> np.ndarray:
        """Dissymmetry of the radiated power, g_lum = 2 (P_+ - P_-) / (P_+ + P_-)."""
        plus, minus = self.radiative_helicity[..., 0], self.radiative_helicity[..., 1]
        return 2 * (plus - minus) / (plus + minus)

    def quantum_yield(self, intrinsic: float = 1.0) -> np.ndarray:
        """Quantum yield for an emitter of quantum yield ``intrinsic`` in the unbounded medium of its layer."""
        if not 0 < intrinsic <= 1:
            raise ValueError("intrinsic quantum yield must be in (0, 1]")
        free = self.free_in_layer if self.normalization == "host" else 1.0
        return self.radiative / (self.total + free * (1 - intrinsic) / intrinsic)


def _icosahedral_rotations():
    """The 60 proper rotations of the icosahedron, from the 120 unit quaternions of the binary
    icosahedral group: averaging over them is exact for every quantity of degree <= 5 in the rotation."""
    phi = (1 + np.sqrt(5)) / 2
    quats = [np.roll([1.0, 0, 0, 0], i) * s for i in range(4) for s in (1, -1)]
    quats += [np.array(v) / 2 for v in np.array(np.meshgrid(*[[1, -1]] * 4)).T.reshape(-1, 4)]
    even = [p for p in itertools.permutations(range(4)) if _parity(p) == 0]
    for signs in np.array(np.meshgrid(*[[1, -1]] * 3)).T.reshape(-1, 3):
        base = np.array([0.0, 1.0 * signs[0], signs[1] / phi, signs[2] * phi]) / 2
        quats += [base[list(p)] for p in even]
    out = {}
    for w, x, y, z in quats:
        r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        out[tuple(np.round(r, 12).ravel())] = r
    return list(out.values())


def _parity(perm):
    perm, swaps = list(perm), 0
    for i in range(len(perm)):
        while perm[i] != i:
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
            swaps += 1
    return swaps % 2


def _group_covariance(p, m, quadrupole, orientation, magnetic_quadrupole=None):
    """<s s^H> of s = (p, m, vec Q[, vec Q_m]) for a fixed source or averaged over rigid rotations: the
    icosahedral group ('isotropic') or 8 turns about an axis, both exact for these rank <= 4 moments."""
    if isinstance(orientation, str) and orientation == "fixed":
        rotations = [np.eye(3)]
    elif isinstance(orientation, str) and orientation == "isotropic":
        rotations = _icosahedral_rotations()
    elif isinstance(orientation, str):
        raise ValueError("orientation must be 'fixed', 'isotropic' or a nonzero 3-vector axis")
    else:
        axis = np.asarray(orientation, dtype=float).ravel()
        if axis.shape != (3,) or not np.linalg.norm(axis) > 0:
            raise ValueError("orientation must be 'fixed', 'isotropic' or a nonzero 3-vector axis")
        axis = axis / np.linalg.norm(axis)
        turn = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        angles = 2 * np.pi * np.arange(8) / 8
        rotations = [np.eye(3) + np.sin(t) * turn + (1 - np.cos(t)) * turn @ turn for t in angles]
    size = 15 if magnetic_quadrupole is None else 24
    cov = np.zeros((size, size), dtype=complex)
    for r in rotations:
        s = [r @ p, r @ m, (r @ quadrupole @ r.T).ravel()]
        if magnetic_quadrupole is not None:
            s.append((r @ magnetic_quadrupole @ r.T).ravel())
        s = np.concatenate(s)
        cov += np.outer(s, np.conj(s))
    return cov / len(rotations)


def _log_matvec(log_m, log_v):
    """log(M v) for log M (L, 2, 2) and log v (..., L, F, 2)."""
    t = log_m[:, None] + log_v[..., None, :]
    return _log_add(t[..., 0], t[..., 1])


def _chains(sol, d):
    """Maps (log, (L, 2, 2)) carrying the layer-d amplitudes of the source's field to every
    layer: psi amplitudes of the regular solution inwards, xi amplitudes of the outgoing one outwards."""
    L, N = sol.orders.size, sol.n_shells
    chain = {d: _log(np.broadcast_to(np.eye(2), (L, 2, 2)) + 0j)}
    for a in range(d - 1, -1, -1):
        chain[a] = log_matmul(sol.log_in[0, a], chain[a + 1])
    for a in range(d + 1, N + 1):
        chain[a] = log_matmul(sol.log_out[0, a - 1], chain[a - 1])
    return chain


def _layer_loss(sol, a, L, nodes, block=128):
    """Loss matrices of layer ``a`` for orders 1..L, in the basis (psi_+, psi_-, xi_+, xi_-) of W_s = M + s N.

    Returns Q (L, 4, 4) Hermitian and scales s (L, 4): a field of channel
    amplitudes v absorbs sum_orders nu_t Re(v~^H Q v~) with v~ = v e^s (in the
    units of :func:`_source_terms`).  Radial integrals by Gauss-Legendre
    quadrature, each basis function scaled by its maximum over the nodes;
    L + 48 nodes integrate the quasi-static r^(2l+2) profiles exactly.
    """
    n, mu, kappa = sol.n[0, a], sol.mu[0, a], sol.kappa[0, a]
    k = sol.k_helicity[0, a]
    r_in = sol.radii[a - 1] if a else 0.0
    r_out = sol.radii[a]
    count = nodes or int(max(64, L + 48, 6 * np.max(np.abs(k)) * (r_out - r_in)))
    t, w = gauss_legendre(count)
    half = 0.5 * (r_out - r_in)
    r = half * t + 0.5 * (r_out + r_in)
    wr = half * w * r**2
    x = r[:, None] * k[None, :]  # (M, 2)
    lp, lx = log_riccati(x, L + 1)  # (M, 2, L + 2)
    xx = np.concatenate([x, x], axis=1)[..., None]  # (M, 4, 1)
    gm = np.empty((L, 4, 4), dtype=complex)
    gn = np.empty((L, 4, 4), dtype=complex)
    shift = np.empty((L, 4))
    with np.errstate(all="ignore"):
        log_x = np.log(xx)
        for start in range(0, L, block):
            l = np.arange(start + 1, min(start + block, L) + 1)
            lu = np.concatenate([lp[..., l], lx[..., l]], axis=1)  # (M, 4, B)
            lu_prev = np.concatenate([lp[..., l - 1], lx[..., l - 1]], axis=1)
            log_m = lu - log_x  # u / x
            log_t = log_m + np.log(np.exp(lu_prev - lu) - l / xx)  # u' / x
            log_e = lu - 2 * log_x + 0.5 * np.log(l * (l + 1.0))  # sqrt(l(l+1)) u / x^2
            if a == 0:  # the core holds psi only
                log_m[:, 2:] = log_t[:, 2:] = log_e[:, 2:] = -np.inf
            s = np.max(np.maximum(np.maximum(log_m.real, log_t.real), log_e.real), axis=0)  # (4, B)
            s = np.where(np.isfinite(s), s, 0.0)
            parts = [np.moveaxis(np.exp(v - s), 2, 0) for v in (log_m, log_t, log_e)]  # (B, M, 4)
            weighted = [np.conj(v) * wr[None, :, None] for v in parts]
            gm[l - 1] = np.swapaxes(weighted[0], 1, 2) @ parts[0]
            gn[l - 1] = np.swapaxes(weighted[1], 1, 2) @ parts[1] + np.swapaxes(weighted[2], 1, 2) @ parts[2]
            shift[l - 1] = s.T
    sign = _BASIS_SIGN
    z = mu / n
    eps_loss, mu_loss, kappa_loss = (n * n / mu).imag, mu.imag, kappa.imag
    cross = gm * sign[None, None, :] + sign[None, :, None] * gn  # int E* . H = (1/(i Z)) v^H cross v
    wz = 1 / (1j * z)
    im_part = (wz * cross - np.conj(wz) * np.conj(np.swapaxes(cross, -1, -2))) / 2j
    q = (
        eps_loss * (gm + sign[:, None] * gn * sign[None, :])
        + (mu_loss / abs(z) ** 2) * (sign[:, None] * gm * sign[None, :] + gn)
        - 2 * kappa_loss * im_part
    )
    return q, shift


def _functionals(dlog, x, k, q, quad, ll):
    """Source functionals W_s(r0) . q_s + (1/6) Q : grad W_s(r0) of the axial families (local
    frame, emitter on z): m = 0, even and odd m = 1 and, with a quadrupole, even and odd m = 2.

    Divided by f/x; (P, K, L, F, 2) from f'/f (P, L, 2), x = k r0 (P, 2), k (2,),
    q = p + i s m / Z_d (P, K, 2, 3) and Q (P, K, 2, 3, 3) = Q_e + i s Q_m / Z_d per channel, or ``None``.  The gradients
    of M and N at the axis are closed forms in f'/f, x and l (checked against finite
    differences of the Bohren-Huffman functions in the tests).
    """
    h = (ll / 2)[None, None, :, None]
    qx, qy, qz = (q[..., i][:, :, None, :] for i in range(3))
    dd = dlog[:, None]
    xx = x[:, None, None, :]
    f0 = _SIGN * 2 * h / xx * qz
    fe = _SIGN * h * dd * qx - h * qy
    fo = h * qx + _SIGN * h * dd * qy
    if quad is None:
        return np.stack([f0, fe, fo], axis=3)
    if quad.ndim == 4:  # the same quadrupole in both channels
        quad = quad[:, :, None]
    c, kk, lf = _SIGN, k[None, None, None, :], ll[None, None, :, None]
    l = (np.sqrt(4 * ll + 1) - 1) / 2
    d2 = ((l - 1) * l * (l + 1) * (l + 2) / 8)[None, None, :, None]
    qxx, qyy, qzz, qxy, qxz, qyz = (
        quad[:, :, :, i, j][:, :, None, :] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    )
    tilt = 2 * lf / xx**2 - 2 * dd / xx - 1
    f0 = f0 + (kk / xx) * c * ((lf / xx - h * dd) * (qxx + qyy) + lf * (dd - 2 / xx) * qzz) / 6
    fe = fe + h * kk * (c * qxz * tilt + qyz * (2 / xx - dd)) / 6
    fo = fo + h * kk * (qxz * (dd - 2 / xx) + c * qyz * tilt) / 6
    f2e = 2 * d2 * kk / xx * (c * dd * (qxx - qyy) - 2 * qxy) / 6
    f2o = 2 * d2 * kk / xx * ((qxx - qyy) + 2 * c * dd * qxy) / 6
    return np.stack([f0, fe, fo, f2e, f2o], axis=3)


def _interface_fields(sol, s, radius, log_al, log_be):
    """Tangential E (M and N parts) and the M part of tangential H at r = ``radius`` in layer s of the
    field with channel amplitudes alpha, beta (logs, (..., L, F, 2)); each (..., L, F)."""
    l = sol.orders
    x = sol.k_helicity[0, s] * radius  # (2,)
    lp, lx = log_riccati(x, l.size + 1)
    with np.errstate(all="ignore"):
        log_x = np.log(x)[:, None]
        values = [(f[:, l] - log_x).T[:, None, :] for f in (lp, lx)]  # log(f/x), (L, 1, 2)
        slopes = [
            (f[:, l] + np.log(np.exp(f[:, l - 1] - f[:, l]) - l / x[:, None]) - log_x).T[:, None, :] for f in (lp, lx)
        ]
        u = np.exp(log_al + values[0]) + np.exp(log_be + values[1])  # u/x per channel
        du = np.exp(log_al + slopes[0]) + np.exp(log_be + slopes[1])  # u'/x
    z = sol.mu[0, s] / sol.n[0, s]
    return u.sum(-1), (du * _SIGN).sum(-1), (u * _SIGN).sum(-1) / (1j * z)


def _emitter_side(sol, d, r, src, tai, quadrupole):
    """Direct and reflected amplitudes, per order, of sources ``src`` (P, K, 6 or 15) (local
    frames) at radii ``r`` in lossless layer d, all scaled by the Riccati functions at r0."""
    L = sol.orders.size
    l = sol.orders
    ll = (l * (l + 1)).astype(float)
    n, mu = sol.n[0], sol.mu[0]
    zd = (mu[d] / n[d]).real
    kd = sol.k_helicity[0, d].real  # (2,)
    x = r[:, None] * kd[None, :]  # (P, 2)
    lp, lx = log_riccati(x.astype(complex), L + 1)
    with np.errstate(all="ignore"):
        log_x = np.log(x)[..., None]
        log_psi = np.swapaxes(lp[..., l] - log_x, 1, 2)  # (P, L, 2): log(psi/x)
        log_xi = np.swapaxes(lx[..., l] - log_x, 1, 2)
        d1 = np.swapaxes(np.exp(lp[..., l - 1] - lp[..., l]) - l / x[..., None], 1, 2)
        d3 = np.swapaxes(np.exp(lx[..., l - 1] - lx[..., l]) - l / x[..., None], 1, 2)
        gamma = np.exp(log_psi + log_xi)  # psi xi / x^2
    q = src[:, :, None, :3] + (1j * _SIGN / zd)[None, None, :, None] * src[:, :, None, 3:6]  # (P, K, 2, 3)
    quad = None
    if quadrupole:  # Q_s = Q_e + i s Q_m / Z_d, as q_s = p + i s m / Z_d (duality)
        quad = np.repeat(src[:, :, None, 6:15], 2, axis=2)
        if src.shape[-1] == 24:
            quad = quad + (1j * _SIGN / zd)[None, None, :, None] * src[:, :, None, 15:24]
        quad = quad.reshape(src.shape[:2] + (2, 3, 3))
    pi_psi, pi_xi = (_functionals(dl, x, kd, q, quad, ll) for dl in (d1, d3))
    bar_quad = None if quad is None else np.conj(quad)
    bar_psi, bar_xi = (_functionals(dl, x, kd, np.conj(q), bar_quad, ll) for dl in (d1, d3))
    g = (kd**2 * zd)[None, None, :] * tai[:, :, None]  # (L, F, 2): direct amplitude per unit functional
    gg = g[None, None] * gamma[:, None, :, None, :]
    src_psi, src_xi = gg * pi_psi, gg * pi_xi  # direct xi (above r0) and psi (below) amplitudes, scaled
    with np.errstate(under="ignore", over="ignore"):
        rho = np.exp(sol.log_r[0, d][None] + log_xi[..., :, None] - log_psi[..., None, :])  # R xi/psi at r0
        sig = np.exp(sol.log_s[0, d][None] + log_psi[..., :, None] - log_xi[..., None, :])  # S psi/xi at r0

    def mv(m, v):
        return np.einsum("plce,pklfe->pklfc", m, v)

    b_hat = mv(np.linalg.solve(np.eye(2) - rho @ sig, rho), src_xi + mv(sig, src_psi))
    a_hat = mv(sig, src_psi + b_hat)
    return dict(
        x=x, kd=kd, zd=zd, q=q, quad=quad, log_psi=log_psi, log_xi=log_xi, bar_psi=bar_psi, bar_xi=bar_xi,
        src_psi=src_psi, src_xi=src_xi, a_hat=a_hat, b_hat=b_hat,
    )  # fmt: skip


def _source_terms(sol, d, r, src, losses, chains, consts, sheets):
    """Per-order terms for sources ``src`` (P, K, 6 or 15) (local frames) at radii ``r`` in lossless
    layer d: free power (P,), reflected power (P, L), radiated power per helicity (P, L, 2) (for a
    host emitter the correction to the free part), absorption per layer (P, L, N + 1) and per sheet
    (P, L, N); normalised units."""
    L, N = sol.orders.size, sol.n_shells
    nu, tai, c_refl, c_rad, c_abs, zh, kh = consts
    e = _emitter_side(sol, d, r, src, tai, nu.shape[1] == 5)
    kd, zd, q, quad = e["kd"], e["zd"], e["q"], e["quad"]
    log_psi, log_xi, bar_psi, bar_xi = e["log_psi"], e["log_xi"], e["bar_psi"], e["bar_xi"]
    src_psi, src_xi, a_hat, b_hat = e["src_psi"], e["src_xi"], e["a_hat"], e["b_hat"]
    terms = bar_psi * a_hat + bar_xi * b_hat
    refl = c_refl * np.real(np.sum(terms, axis=(1, 3, 4)))
    # Next to a lossless interface Re(terms) << |terms| at high orders, and the O(1) scaled
    # R, S come from logarithms of size ~ l ln(l/x): below this floor a term is rounding
    # noise, not a remainder still to be summed
    amplify = 1 + 2 * np.max(np.abs(log_psi) + np.abs(log_xi), axis=-1)  # (P, L)
    floor = c_refl * 16 * np.finfo(float).eps * amplify * np.sum(np.abs(terms), axis=(1, 3, 4))
    q2 = np.sum(np.abs(q) ** 2, axis=(1, 3))  # (P, 2)
    if quad is not None:  # free quadrupole power: (k^2 / 120) sum |Q_ij|^2 in units of |p|^2 (Jackson 9.49 / 9.24)
        q2 = q2 + (kd**2 / 120)[None, :] * np.sum(np.abs(quad) ** 2, axis=(1, 3, 4))
    free = 0.5 * (zd * kd**2 / (zh * kh**2)) @ q2.T  # (P,)
    with np.errstate(all="ignore"):
        log_up = _log(b_hat + src_psi) - log_xi[:, None, :, None, :]  # xi amplitudes above r0
        log_down = _log(a_hat + src_xi) - log_psi[:, None, :, None, :]  # psi amplitudes below r0
        if d == N:
            gain = 2 * np.real(np.conj(src_psi) * b_hat) + np.abs(b_hat) ** 2
            rad = c_rad * np.einsum("pklfc,lf->plc", gain, nu) * np.exp(-2 * log_xi.real)
            rad_free = 0.5 * q2
        else:
            log_h = _log_matvec(chains[N], log_up)
            rad = c_rad * np.einsum("pklfc,lf->plc", np.exp(2 * log_h.real), nu)
            rad_free = np.zeros_like(q2)
        absorbed = np.zeros((r.size, L, N + 1))
        for a, (loss, scale) in losses.items():
            c = loss.shape[0]  # orders integrated in layer a
            if a < d:
                log_al = _log_matvec(chains[a][:c], log_down[:, :, :c])
                log_be = _log_matvec(sol.log_r[0, a, :c], log_al)
            else:
                log_be = _log_matvec(chains[a][:c], log_up[:, :, :c])
                log_al = _log_matvec(sol.log_s[0, a, :c], log_be)
            v = np.exp(np.concatenate([log_al, log_be], axis=-1) + scale[None, None, :, None, :])
            form = np.real(np.einsum("pklfi,lij,pklfj->plf", np.conj(v), loss, v))
            absorbed[:, :c, a] = c_abs * np.einsum("plf,lf->pl", form, nu[:c])
    on_sheets = _on_sheets(sol, d, log_up, log_down, chains, c_abs, nu, sheets)
    return free, (refl, floor), rad, rad_free, absorbed, on_sheets


def _on_sheets(sol, d, log_up, log_down, chains, c_abs, nu, sheets):
    """Absorption per order and sheet (P, L, N) of sources in layer d with amplitudes ``log_up`` and
    ``log_down``: (c/8 pi) [Re sigma |<E_t>|^2 + k0 Im zeta |<D_n>|^2] over the sphere, fields averaged
    over both sides; D_r = i l(l+1) H_M / (k0 R)."""
    L, N = sol.orders.size, sol.n_shells
    ll = (sol.orders * (sol.orders + 1)).astype(float)
    k0 = 2 * np.pi / sol.wavelength[0]
    on_sheets = np.zeros((log_up.shape[0], L, N))
    with np.errstate(all="ignore"):
        for j, (sigma, zeta) in sheets.items():
            sides = []
            for side in (j, j + 1):
                if j < d:  # below the emitter: the regular solution continued inwards
                    log_al = _log_matvec(chains[side], log_down)
                    log_be = _log_matvec(sol.log_r[0, side], log_al)
                else:
                    log_be = _log_matvec(chains[side], log_up)
                    log_al = _log_matvec(sol.log_s[0, side], log_be)
                sides.append(_interface_fields(sol, side, sol.radii[j], log_al, log_be))
            e_m, e_n, h_m = ((inner + outer) / 2 for inner, outer in zip(*sides))
            form = sigma.real * sol.radii[j] ** 2 * (np.abs(e_m) ** 2 + np.abs(e_n) ** 2)
            form = form + zeta.imag * (ll / k0)[None, None, :, None] * np.abs(h_m) ** 2
            on_sheets[:, :, j] = c_abs / k0 * np.einsum("pklf,lf->pl", form, nu)
    return on_sheets


def _legendre_derivatives(L, mu):
    """First three derivatives of P_l(mu) for l = 0..L, each (L + 1, D), by the recurrence
    P^(k)_(l+1) = P^(k)_(l-1) + (2l + 1) P^(k-1)_l."""
    p = np.zeros((4, L + 2, mu.size))
    p[0, 0], p[0, 1] = 1.0, mu
    for l in range(1, L + 1):
        p[0, l + 1] = ((2 * l + 1) * mu * p[0, l] - l * p[0, l - 1]) / (l + 1)
    for k in (1, 2, 3):
        p[k, 1] = p[k - 1, 0]
        for l in range(1, L + 1):
            p[k, l + 1] = p[k, l - 1] + (2 * l + 1) * p[k - 1, l]
    return p[1, : L + 1], p[2, : L + 1], p[3, : L + 1]


def _angular_parts(L, theta):
    """tau^0, pi^1, tau^1, pi^2, tau^2 for l = 1..L at polar angles theta (D,), each (D, L): the
    theta dependence of the Bohren-Huffman M and N of azimuthal order 0, 1, 2 (pi^m = P^m / sin,
    tau^m = dP^m / dtheta, no Condon-Shortley phase)."""
    st, ct = np.sin(theta), np.cos(theta)
    d1, d2, d3 = (v[1:].T for v in _legendre_derivatives(L, ct))
    st, ct = st[:, None], ct[:, None]
    return -st * d1, d1, ct * d1 - st**2 * d2, st * d2, 2 * st * ct * d2 - st**3 * d3


def _synthesise(weights, theta, phi, chunk=4096):
    """Local (F_theta, F_phi), each (D, K), of the far fields sum_(l,f,s) w (m_f + i s n_t,f) with
    w (K, L, F, 2) per family f = 0, 1e, 1o[, 2e, 2o]: angular functions per distinct theta,
    azimuthal factors applied analytically, sums over l as real matrix products."""
    K, L, F, _ = weights.shape
    m_part = weights.sum(-1)  # (K, L, F): coefficient of m_f
    n_part = (weights * (1j * _SIGN)).sum(-1)  # coefficient of n_t,f
    # the (part, family) blocks each angular function multiplies, as one real (L, 2 K B) matrix
    blocks = {
        0: [(n_part, 0), (m_part, 0)],
        1: [(m_part, 1), (m_part, 2), (n_part, 1), (n_part, 2)],
        2: [(n_part, 1), (n_part, 2), (m_part, 1), (m_part, 2)],
    }
    if F == 5:
        blocks[3] = [(m_part, 3), (m_part, 4), (n_part, 3), (n_part, 4)]
        blocks[4] = [(n_part, 3), (n_part, 4), (m_part, 3), (m_part, 4)]
    stacked = {}
    for a, parts in blocks.items():
        cols = np.concatenate([part[:, :, f].T for part, f in parts], axis=1)  # (L, K B) complex
        stacked[a] = np.concatenate([cols.real, cols.imag], axis=1)
    D = theta.size
    f_theta = np.empty((D, K), dtype=complex)
    f_phi = np.empty((D, K), dtype=complex)
    cosines, where = np.unique(np.round(np.cos(theta), 14), return_inverse=True)
    order = np.argsort(where, kind="stable")
    bounds = np.searchsorted(where[order], np.arange(0, cosines.size + chunk, chunk))
    for c, start in enumerate(range(0, cosines.size, chunk)):
        rows = order[bounds[c] : bounds[c + 1]]
        angular = _angular_parts(L, np.arccos(np.clip(cosines[start : start + chunk], -1, 1)))
        idx = where[rows] - start
        sums = {}
        for a, mat in stacked.items():
            prod = angular[a] @ mat  # (chunk, 2 K B)
            half = prod.shape[1] // 2
            full = (prod[:, :half] + 1j * prod[:, half:])[idx]
            sums[a] = [full[:, b * K : (b + 1) * K] for b in range(len(blocks[a]))]
        c1, s1 = np.cos(phi[rows])[:, None], np.sin(phi[rows])[:, None]
        (t0_n0, t0_m0), (p1_me, p1_mo, p1_ne, p1_no), (t1_ne, t1_no, t1_me, t1_mo) = sums[0], sums[1], sums[2]
        ft = t0_n0 + s1 * (t1_no - p1_me) + c1 * (t1_ne + p1_mo)
        fp = -t0_m0 - c1 * (t1_me - p1_no) - s1 * (p1_ne + t1_mo)
        if F == 5:
            c2, s2 = np.cos(2 * phi[rows])[:, None], np.sin(2 * phi[rows])[:, None]
            (p2_me, p2_mo, p2_ne, p2_no), (t2_ne, t2_no, t2_me, t2_mo) = sums[3], sums[4]
            ft = ft + s2 * (t2_no - 2 * p2_me) + c2 * (t2_ne + 2 * p2_mo)
            fp = fp - c2 * (t2_me - 2 * p2_no) - s2 * (2 * p2_ne + t2_mo)
        f_theta[rows], f_phi[rows] = ft, fp
    return f_theta, f_phi


def _pattern(sol, d, r, frame, position, local, sources, directions, current):
    """Far field F (E_far = (k_h^2/eps_h) exp(i k_h R)/R F) of the eigen-sources in the global
    spherical basis: the outgoing host amplitudes h of W_s = M + s N radiate
    F = (i eps_h k0 / 2 k_h^3) sum h (-i)^(l+1) (m + i s n_t); for an emitter in the host the
    direct part is added in closed form, (p - i k_h Q n / 6)_transverse - n x m', phase exp(-i k_h n.r0)."""
    from .emission import _spherical_basis

    N = sol.n_shells
    if local.shape[-1] > 15:
        raise ValueError("far fields of magnetic quadrupoles are not implemented")
    quad = local.shape[-1] == 15
    l = sol.orders.astype(float)
    ll = l * (l + 1)
    nu = [4 * np.pi * ll / (2 * l + 1)] + [2 * np.pi * ll**2 / (2 * l + 1)] * 2
    if quad:
        nu += [2 * np.pi * ll**2 * (l - 1) * (l + 2) / (2 * l + 1)] * 2
    nu = np.stack(nu, axis=1)
    with np.errstate(divide="ignore"):
        tai = np.where(nu > 0, 4 * np.pi / nu, 0.0)
    e = _emitter_side(sol, d, np.array([r]), local[None], tai, quad)
    with np.errstate(all="ignore"):
        if d == N:  # the scattered part only; the direct one below
            log_h = _log(e["b_hat"]) - e["log_xi"][:, None, :, None, :]
        else:
            log_up = _log(e["b_hat"] + e["src_psi"]) - e["log_xi"][:, None, :, None, :]
            log_h = _log_matvec(_chains(sol, d)[N], log_up)
        h = np.exp(log_h)[0]  # (K, L, F, 2)
    theta, phi = directions
    n_hat, e_theta, e_phi = _spherical_basis(theta, phi)
    n_local = n_hat @ frame.T
    theta_l = np.arccos(np.clip(n_local[:, 2], -1.0, 1.0))
    phi_l = np.arctan2(n_local[:, 1], n_local[:, 0])
    n_h, mu_h = sol.n[0, N].real, sol.mu[0, N].real
    k0 = 2 * np.pi / sol.wavelength[0]
    k_h = k0 * n_h
    phase = (1j * (n_h**2 / mu_h) * k0 / (2 * k_h**3)) * (-1j) ** (l + 1)  # (L,)
    f_theta, f_phi = _synthesise(h * phase[None, :, None, None], theta_l, phi_l)
    _, et_l, ep_l = _spherical_basis(theta_l, phi_l)
    far = f_theta[:, :, None] * (et_l @ frame)[:, None, :] + f_phi[:, :, None] * (ep_l @ frame)[:, None, :]
    if d == N:
        p, m = sources[:3].T, sources[3:6].T * (sol.mu[0, N].real if current else 1.0)  # (K, 3); dual m
        dual = (n_h / mu_h) * m
        p_eff = np.broadcast_to(p, (n_hat.shape[0],) + p.shape).astype(complex)
        if quad:
            q = sources[6:].T.reshape(-1, 3, 3)
            p_eff = p_eff - (1j * k_h / 6) * np.einsum("kij,dj->dki", q, n_hat)
        along = np.einsum("dki,di->dk", p_eff, n_hat)[..., None] * n_hat[:, None, :]
        direct = p_eff - along - np.cross(n_hat[:, None, :], dual[None])
        far = far + direct * np.exp(-1j * k_h * (n_hat @ position))[:, None, None]
    fields = np.stack([np.einsum("dkg,dg->dk", far, e_theta), np.einsum("dkg,dg->dk", far, e_phi)], axis=-1)
    return {"fields": fields}


def _series(terms, base, tol, floor=None):
    """base + sum over orders (axis 1), and whether the remainder is below tol x |sum|
    (terms below their rounding ``floor`` count as zero in that estimate)."""
    total = base + terms.sum(axis=1)
    last = terms[:, -4:]
    if floor is not None:
        last = np.where(np.abs(last) <= floor[:, -4:], 0.0, last)
    tail = _tail_estimate(np.moveaxis(last, 1, 0))
    ok = np.isfinite(total) & (tail <= tol * np.maximum(np.abs(total), np.finfo(float).tiny))
    return total, ok


def _rates_one(sol, r, shells, local, tol, nodes, quadrupole=False):
    L, N = sol.orders.size, sol.n_shells
    l = sol.orders.astype(float)
    n, mu, kappa = sol.n[0], sol.mu[0], sol.kappa[0]
    eps = n * n / mu
    kh = sol.k_helicity[0, N, 0].real
    zh = (mu[N] / n[N]).real
    k0 = 2 * np.pi / sol.wavelength[0]
    ll = l * (l + 1)
    # angular norms nu = int |m_(lm)|^2 dOmega of the families (m = 0, 1e, 1o[, 2e, 2o]); Tai weights 4 pi / nu
    nu = [4 * np.pi * ll / (2 * l + 1)] + [2 * np.pi * ll**2 / (2 * l + 1)] * 2
    if quadrupole:
        nu += [2 * np.pi * ll**2 * (l - 1) * (l + 2) / (2 * l + 1)] * 2
    nu = np.stack(nu, axis=1)
    with np.errstate(divide="ignore"):
        tai = np.where(nu > 0, 4 * np.pi / nu, 0.0)
    consts = (
        nu,
        tai,
        3 / (4 * zh * kh**2),
        3 / (16 * np.pi * zh**2 * kh**4),
        3 * k0 / (32 * np.pi * zh * kh**2),
        zh,
        kh,
    )
    lossy = [a for a in range(N) if eps[a].imag != 0 or mu[a].imag != 0 or kappa[a].imag != 0]
    # The loss series of layer a converges like (r_< / r_>)^(2l) at its boundary nearest to an
    # emitter - usually far sooner than the LDOS series: integrate that many orders, then check.
    counts = {}
    for a in lossy:
        bounds = np.array(([sol.radii[a - 1]] if a else []) + [sol.radii[a]])
        q = np.max(np.minimum(bounds[None, :] / r[:, None], r[:, None] / bounds[None, :]))
        counts[a] = int(min(L, max(32, _orders_needed(q, tol, L))))
    P, K = local.shape[:2]
    step = max(1, _CHUNK // (K * L * 2 * nu.shape[1]))
    chains = {d: _chains(sol, d) for d in np.unique(shells)}
    sheets = {}
    if sol.has_sheets:
        sheets = {j: (sol.sheet_sigma[0, j], sol.sheet_zeta[0, j]) for j in range(N)}
        sheets = {j: v for j, v in sheets.items() if v[0] != 0 or v[1] != 0}
    losses = {}
    while True:
        for a in lossy:
            if a not in losses or losses[a][0].shape[0] != counts[a]:
                losses[a] = _layer_loss(sol, a, counts[a], nodes)
        free = np.zeros(P)
        refl = np.zeros((P, L))
        floor = np.zeros((P, L))
        rad = np.zeros((P, L, 2))
        rad_free = np.zeros((P, 2))
        absorbed = np.zeros((P, L, N + 1))
        on_sheets = np.zeros((P, L, N))
        for d in chains:
            idx = np.flatnonzero(shells == d)
            for start in range(0, idx.size, step):
                j = idx[start : start + step]
                free[j], (refl[j], floor[j]), rad[j], rad_free[j], absorbed[j], on_sheets[j] = _source_terms(
                    sol, d, r[j], local[j], losses, chains[d], consts, sheets
                )
        absorption = np.zeros((P, N + 1))
        ok_abs = np.ones(P, dtype=bool)
        again = False
        for a in lossy:
            absorption[:, a], ok_a = _series(absorbed[:, : counts[a], a], 0.0, tol)
            if counts[a] < L and not ok_a.all():
                counts[a], again = L, True
            ok_abs &= ok_a
        if not again:
            break
    total, ok = _series(refl, free, tol, floor)
    radiative = np.empty((P, 2))
    for c in (0, 1):
        radiative[:, c], ok_c = _series(rad[..., c], rad_free[:, c], tol)
        ok &= ok_c
    sheet_absorption = np.zeros((P, N))
    for j in sheets:
        sheet_absorption[:, j], ok_j = _series(on_sheets[..., j], 0.0, tol)
        ok &= ok_j
    return total, radiative, absorption, sheet_absorption, free, ok, ok_abs


def _tetm_functionals(x, kd, zd, src, quadrupole, l):
    """The source functionals of :func:`_functionals` in an achiral layer, split into the parts
    odd (TM, the N modes) and even (TE, the M modes) in the helicity s of W_s = M + s N.

    Returns coefficients ``cv``, ``cd`` (P, K, L, F, 2), last axis [TM, TE], such that the
    functional of family F (m = 0, 1e, 1o[, 2e, 2o]) and polarization p is cv - cd r, with
    r = f_{l+1}/f_l of the radial function f (psi or xi), i.e. cv - cd (l+1)/x + cd f'/f
    (divided by f/x, as :func:`_functionals`).  The quadrupole parts of cv are written so
    that their leading small-x terms cancel analytically (e.g. A - 2/x = (l-1)/x - r for
    Q_zz).  x = k_d r0 (P,), kd (P,) and zd the wavenumber and Z = mu/n of the layer, src
    (P, K, 6, 15 or 24) the local sources (p, m[, Q[, Q_m]]); a magnetic quadrupole enters by
    duality, Q_s = Q + i s Q_m/Z, like m in q_s = p + i s m/Z.  Every coefficient is a real
    factor times one source component.
    """
    P, K = src.shape[:2]
    L = l.size
    ll = (l * (l + 1)).astype(float)
    F = 5 if quadrupole else 3
    cv = np.zeros((P, K, L, F, 2), dtype=complex)
    cd = np.zeros_like(cv)
    X = x[:, None, None]
    H = (ll / 2)[None, None, :]
    up = ((l + 1) / 1.0)[None, None, :] / X  # (l+1)/x: f'/f = (l+1)/x - r
    px, py, pz = (src[:, :, i][:, :, None] for i in range(3))
    mx, my, mz = (src[:, :, 3 + i][:, :, None] / zd for i in range(3))
    cv[..., 0, TM] = 2 * H / X * pz
    cv[..., 0, TE] = 1j * 2 * H / X * mz
    cd[..., 1, TM], cv[..., 1, TM] = H * px, -1j * H * my + H * px * up
    cd[..., 1, TE], cv[..., 1, TE] = 1j * H * mx, -H * py + 1j * H * mx * up
    cv[..., 2, TE], cd[..., 2, TE] = H * px + 1j * H * my * up, 1j * H * my
    cv[..., 2, TM], cd[..., 2, TM] = 1j * H * mx + H * py * up, H * py
    if quadrupole:
        parts = [(src[:, :, 6:15], (TM, TE))]
        if src.shape[-1] == 24:  # duality: Q_s = Q_e + i s Q_m / Z swaps the parity in s, so TM <-> TE
            parts.append((1j * src[:, :, 15:24] / zd, (TE, TM)))
        for vec, (odd, even) in parts:
            Q = vec.reshape(P, K, 3, 3)
            qxx, qyy, qzz, qxy, qxz, qyz = (
                Q[:, :, i, j][:, :, None] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
            )
            kk = kd[:, None, None]
            lf = ll[None, None, :]
            lm = (l - 1.0)[None, None, :]
            d2 = ((l - 1) * l * (l + 1) * (l + 2) / 8)[None, None, :]
            # m = 0, odd: (k/x) [(lf/x - h f'/f)(Qxx + Qyy) + lf (f'/f - 2/x) Qzz]/6
            cv[..., 0, odd] += kk * lf * lm / (6 * X**2) * (qzz - (qxx + qyy) / 2)
            cd[..., 0, odd] += kk / X * (-H * (qxx + qyy) + lf * qzz) / 6
            # m = 1: h k [s Qxz t + Qyz (2/x - f'/f)]/6 and h k [Qxz (f'/f - 2/x) + s Qyz t]/6,
            # t = 2 lf/x^2 - 2 f'/f/x - 1 = 2 (l+1)(l-1)/x^2 - 1 + 2 r/x
            tilt = 2 * (l + 1.0)[None, None, :] * lm / X**2 - 1
            cv[..., 1, odd] += H * kk * qxz * tilt / 6
            cd[..., 1, odd] += H * kk * qxz * (-2 / X) / 6
            cv[..., 1, even] += H * kk * qyz * (-lm / X) / 6
            cd[..., 1, even] += -H * kk * qyz / 6
            cv[..., 2, even] += H * kk * qxz * (lm / X) / 6
            cd[..., 2, even] += H * kk * qxz / 6
            cv[..., 2, odd] += H * kk * qyz * tilt / 6
            cd[..., 2, odd] += H * kk * qyz * (-2 / X) / 6
            # m = 2: 2 d2 k/x [s f'/f (Qxx - Qyy) - 2 Qxy]/6 and 2 d2 k/x [(Qxx - Qyy) + 2 s f'/f Qxy]/6
            c2 = 2 * d2 * kk / X / 6
            cd[..., 3, odd] += c2 * (qxx - qyy)
            cv[..., 3, odd] += c2 * (qxx - qyy) * up
            cv[..., 3, even] += c2 * (-2 * qxy)
            cv[..., 4, even] += c2 * (qxx - qyy)
            cd[..., 4, odd] += c2 * (2 * qxy)
            cv[..., 4, odd] += c2 * (2 * qxy) * up
    return cv, cd


def _normalized_terms(t, x, kd, zd, src, quadrupole, tai, zh, kh):
    """Per-order terms of sources ``src`` (P, K, 6 or 15) at positions of one achiral lossless
    layer, from the normalized quantities ``t`` of :meth:`pystratify.normalized._Sweep.at`.

    Returns the free power (P,), the complex reflected terms (P, L) (Re: power, Im: twice the
    frequency shift) and the radiated power per helicity (P, L, 2), including the direct part
    for an emitter in the host; units of :func:`_source_terms`.

    Only the real symmetric part of c c^H enters the reflected power (the Green's forms are
    symmetric), and with c = c' + i c'' it is c' c'^T + c'' c''^T: each source contributes the
    forms of two *real* functionals e = e_v - e_d r (see :func:`_tetm_functionals`), evaluated
    directly as P (rho B~^2 + sigma A~^2 + 2 rho sigma A~ B~)/(1 - rho sigma), A~ = e_v - e_d r_psi,
    B~ = e_v - e_d r_xi.
    Real coefficients keep the small parts of the normalized quantities at full precision, and a
    near cancellation between the value and the derivative coupling (a quadrupole near the
    centre) is paid once, in A~ and B~.  Helicity is formed only from the radiated amplitudes,
    (a_TE + s a_TM)/2 for W_s = M + s N.
    """
    l = np.arange(1, tai.shape[0] + 1)
    cv, cd = _tetm_functionals(x, kd, zd, src, quadrupole, l)
    pick = lambda a: np.moveaxis(a, 0, -1)[:, :, None, :]  # noqa: E731  (2, P, L) -> (P, L, 1, 2)
    rp, rx, rho, sig = pick(t["rp"]), pick(t["rx"]), pick(t["rho"]), pick(t["sigma"])
    delta = 1 - rho * sig
    forms = np.zeros(cv.shape[:1] + cv.shape[2:], dtype=complex)  # (P, L, F, 2)
    for k in range(src.shape[1]):
        for part in (np.real, np.imag):
            ev, ed = part(cv[:, k]), part(cd[:, k])
            if not (np.any(ev) or np.any(ed)):
                continue
            At, Bt = ev - ed * rp, ev - ed * rx
            forms = forms + (rho * Bt**2 + sig * At**2 + 2 * rho * sig * At * Bt) / delta
    forms = pick(t["P"]) * forms
    free_unit = zd * kd**2 / (zh * kh**2)  # free power of a unit electric dipole in the layer
    X2 = (x**2)[:, None]
    refl = 1.5 * free_unit[:, None] * np.einsum("plfc,lf->pl", forms, tai) / X2
    Fv = np.moveaxis(t["F"], 0, -1)[:, None, :, None, :]  # (P, 1, L, 1, 2)
    Fr = np.moveaxis(t["Fr"], 0, -1)[:, None, :, None, :]
    with np.errstate(under="ignore"):
        amp = cv * Fv - cd * Fr  # host amplitudes per source, order, family and polarization
        rad = np.stack(
            [np.sum(np.abs(amp[..., TE] + s * amp[..., TM]) ** 2, axis=1) / 2 for s in (1.0, -1.0)], axis=-1
        )  # (P, L, F, 2)
    rad = 1.5 * (free_unit * t["f_rad"])[:, None, None] * np.einsum("plfs,lf->pls", rad, tai) / X2[..., None]
    q2 = np.zeros(x.size)
    for s in (1.0, -1.0):
        q = src[:, :, :3] + 1j * s / zd * src[:, :, 3:6]
        q2 = q2 + np.sum(np.abs(q) ** 2, axis=(1, 2))
        if quadrupole:
            qs = src[:, :, 6:15] + (1j * s / zd * src[:, :, 15:24] if src.shape[-1] == 24 else 0)
            q2 = q2 + kd**2 / 120 * np.sum(np.abs(qs) ** 2, axis=(1, 2))
    free = 0.5 * zd * kd**2 / (zh * kh**2) * q2
    return free, refl, rad


def _lommel_absorption(sol, r, shells, local, quadrupole, tol, nodes, tai):
    """Absorption per layer (P, N + 1) and per interface response (P, N; sheets, d-parameters) of sources
    ``local`` (P, K, 6, 15 or 24) in achiral layers, in the units of :func:`_source_terms`, and whether
    every loss series met ``tol`` (P,).  An interface absorbs the jump of the radial flux
    (:meth:`pystratify.decay._ShellLosses.log_surface_loss`).

    The field of a source in an absorbing shell a is the regular (a below the emitter) or outgoing
    (a above) solution of that shell, with the amplitude of the source functional applied to the
    other solution at the source, as for the dipoles of :mod:`pystratify.decay`; the radial loss
    integrals are those of :class:`pystratify.decay._ShellLosses` (closed-form Lommel terms, O(L) in
    time and memory), and families and polarizations add (orthogonal vector harmonics).  The
    functionals are applied in the ratio form c_v f - c_d f_(l+1) of :func:`_tetm_functionals`, with
    f_(l+1) the same combination of Riccati functions of order l + 1.
    """
    from .decay import _ShellLosses

    L, N = sol.orders.size, sol.n_shells
    l = sol.orders
    n, mu, k = sol.n[0], sol.mu[0], sol.k[0]
    zh, kh = (mu[N] / n[N]).real, k[N].real
    losses = _ShellLosses(sol, nodes)
    absorption, surface = np.zeros((r.size, N + 1)), np.zeros((r.size, N))
    ok = np.ones(r.size, dtype=bool)
    items = [("layer", a) for a in losses.absorbing] + [("surface", j) for j in losses.surfaces]
    if not items:
        return absorption, surface, ok

    def log1p(z):
        with np.errstate(all="ignore"):
            return np.log(1 + np.exp(z))

    for d in np.unique(shells):
        idx = np.flatnonzero(shells == d)
        x = (k[d] * r[idx]).real
        zd, kd = (mu[d] / n[d]).real, k[d].real
        cv, cd = _tetm_functionals(x, np.full(idx.size, kd), zd, local[idx], quadrupole, l)
        lps, lxs = log_riccati(x.astype(complex), L + 1)  # orders 0..L+1
        lp, lp1, lx, lx1 = lps[:, l], lps[:, l + 1], lxs[:, l], lxs[:, l + 1]
        pre = 1.5 * zd * kd**2 / (zh * kh**2) * kd**3 * (mu[d] / n[d] ** 2).real
        amp = {}  # log (u, u_(l+1), normalisation) of the solution at the source that continues into the shell
        for p in (TM, TE):
            log_r, log_s = sol.log_r[p, d, 0], sol.log_s[p, d, 0]
            with np.errstate(all="ignore"):
                log_delta = np.log(1 - np.exp(log_r + log_s))
            # below the emitter: u_out = S psi + xi over the regular solution of shell d (A_d)
            amp[p, "below"] = (
                lx + log1p(log_s + lp - lx),
                lx1 + log1p(log_s + lp1 - lx1),
                log_delta + sol.log_a[p, d, 0],
            )
            # above it: u_in = psi + R xi over the outgoing solution (B_out,d)
            amp[p, "above"] = (
                lp + log1p(log_r + lx - lp),
                lp1 + log1p(log_r + lx1 - lp1),
                log_delta + sol.log_b_out[p, d, 0],
            )
        start = losses.orders(r[idx], tol, L)
        for kind, a in items:  # a shell, or an interface (whose inner side is shell a)
            side = "below" if a < d else "above"
            for count in dict.fromkeys((start, L)):
                terms = np.zeros((idx.size, L))
                for p in (TM, TE):
                    if kind == "layer":
                        log_loss = losses.log_loss(a, side, p, count)
                    else:
                        log_loss = losses.log_surface_loss(a, side, p, count)
                    lu, lu1, ref = amp[p, side]
                    with np.errstate(all="ignore"):
                        m = np.maximum(lu.real, lu1.real)
                        m = np.where(np.isfinite(m), m, 0.0)
                        eu, eu1 = np.exp(lu - m), np.exp(lu1 - m)
                        scale = 2 * (m - ref.real) + log_loss[None, :]  # (P, L)
                    for kk in range(local.shape[1]):
                        f = cv[:, kk, :, :, p] * eu[:, :, None] - cd[:, kk, :, :, p] * eu1[:, :, None]  # (P, L, F)
                        with np.errstate(all="ignore"):
                            w = np.real(np.exp(2 * np.log(np.abs(f)) + scale[:, :, None]))
                        terms += np.einsum("plf,lf->pl", np.where(np.isfinite(w), w, 0.0), tai) / x[:, None] ** 2
                terms = pre * terms
                total = terms.sum(axis=1)
                tail = tail_estimate(np.moveaxis(terms[:, :count][:, -4:], 1, 0))
                scale = np.maximum(np.abs(total), np.finfo(float).tiny)
                if count == L or np.all(
                    tail <= 0.1 * tol * scale
                ):  # as decay_rates: shorter sums only when clearly done
                    break
            (absorption if kind == "layer" else surface)[idx, a] = total
            ok[idx] &= tail <= tol * scale
    return absorption, surface, ok


#: complex numbers per block of positions in the normalized route (bounds the memory of long sweeps)
_NORMALIZED_BLOCK = 4_000_000


def _rates_normalized(radii, n, mu, wavelength, r, shells, local, tol, nodes, l_max, l_cap, quadrupole, sheets=None):
    """Total, shift and helicity-resolved radiative power from the normalized formulation; absorption
    per layer and per sheet from the logarithmic route (independent, so the energy balance remains a test).

    Positions whose sums do not meet ``tol`` are recomputed with twice the orders (up to ``l_cap``).
    Returns total, shift (P,), radiative (P, 2), absorption (P, N + 1), free (P,), converged (P,)
    and the largest order used; units of :func:`_source_terms`."""
    from .normalized import _Sweep

    N, P = radii.size, r.size
    k = 2 * np.pi * n / wavelength
    eps = n * n / mu
    kappa0 = np.zeros(n.size, dtype=complex)
    lossy = bool(np.any((eps[:N].imag != 0) | (mu[:N].imag != 0)))
    zh, kh = (mu[N] / n[N]).real, k[N].real
    out = dict(
        total=np.zeros(P), shift=np.zeros(P), radiative=np.zeros((P, 2)), absorption=np.zeros((P, N + 1)),
        free=np.zeros(P), sheet_absorption=np.zeros((P, N)),
    )  # fmt: skip
    ok = np.zeros(P, dtype=bool)
    used = 0
    todo = np.arange(P)
    L = int(l_max) if l_max is not None else _starting_order(radii, n, kappa0, wavelength, r, tol, l_cap)
    while todo.size:
        used = max(used, L)
        l = np.arange(1, L + 1).astype(float)
        ll = l * (l + 1)
        nu = [4 * np.pi * ll / (2 * l + 1)] + [2 * np.pi * ll**2 / (2 * l + 1)] * 2
        if quadrupole:
            nu += [2 * np.pi * ll**2 * (l - 1) * (l + 2) / (2 * l + 1)] * 2
        nu = np.stack(nu, axis=1)
        with np.errstate(divide="ignore"):
            tai = np.where(nu > 0, 4 * np.pi / nu, 0.0)
        sweep = _Sweep(radii, n, mu, k, L, sheets, 2 * np.pi / wavelength)
        K = local.shape[1]
        step = max(1, _NORMALIZED_BLOCK // (K * L * tai.shape[1] * 4))
        ok_core = np.zeros(todo.size, dtype=bool)
        for d in np.unique(shells[todo]):
            here = np.flatnonzero(shells[todo] == d)
            zd, kd = (mu[d] / n[d]).real, k[d].real
            for start in range(0, here.size, step):
                part = here[start : start + step]
                j = todo[part]
                t = sweep.at(r[j])
                free, refl, rad = _normalized_terms(
                    t, t["x"], np.full(j.size, kd), zd, local[j], quadrupole, tai, zh, kh
                )
                total = free + refl.real.sum(axis=1)
                scale = np.abs(free + refl.sum(axis=1))
                good = np.isfinite(scale) & (tail_estimate(np.moveaxis(np.abs(refl[:, -4:]), 1, 0)) <= tol * scale)
                rad_sum = rad.sum(axis=1)
                tail = tail_estimate(np.moveaxis(rad[:, -4:].sum(axis=-1), 1, 0))
                good &= tail <= tol * np.maximum(rad_sum.sum(axis=1), np.finfo(float).tiny)
                out["total"][j], out["shift"][j], out["radiative"][j], out["free"][j] = (
                    total, refl.imag.sum(axis=1) / 2, rad_sum, free,
                )  # fmt: skip
                ok_core[part] = good
        ok_abs = np.ones(todo.size, dtype=bool)
        if lossy or sheets:
            sol = solve(radii, n, wavelength, mu, L, sheets=sheets)
            absorption, surface, ok_abs = _lommel_absorption(
                sol, r[todo], shells[todo], local[todo], quadrupole, tol, nodes, tai
            )
            out["absorption"][todo], out["sheet_absorption"][todo] = absorption, surface
        ok[todo] = ok_core & ok_abs
        if l_max is not None or L >= l_cap:
            break
        todo = todo[~ok[todo]]
        L = min(2 * L, l_cap)
    return out, ok, used


def _starting_order(radii, n, kappa, wavelength, r, tol, l_cap):
    """Particle size (largest |k_s| R over the layers, the host at the outer radius) and proximity."""
    k = 2 * np.pi * np.abs(n[:, None] + _SIGN * kappa[:, None]) / wavelength
    x = max(float(np.max(k[:-1].max(axis=1) * radii)), float(k[-1, 0] * radii[-1]), 1e-3)
    size = x + 4 * x ** (1 / 3) + 2
    q = np.max(np.minimum(radii[None, :] / r[:, None], r[:, None] / radii[None, :]))
    near = truncation_order(radii[-1], n[-1], wavelength, "near")
    return int(min(l_cap, max(32, size, near, _orders_needed(q, tol, l_cap))))


def emission_rates(
    radii,
    n,
    wavelength,
    position,
    moment,
    magnetic_moment=None,
    orientation="fixed",
    mu=None,
    kappa=None,
    dipole="electric",
    normalization="host",
    l_max=None,
    tol=1e-6,
    l_cap=None,
    quadrature_nodes=None,
    warn=True,
    sheets=None,
    magnetic_convention="dual",
    quadrupole=None,
    route="auto",
    magnetic_quadrupole=None,
) -> EmissionRates:
    """Total, radiative (helicity-resolved) and nonradiative decay rates of a dipole or quadrupole source.

    Parameters
    ----------
    radii, n, wavelength, mu : geometry and materials at one wavelength (see
        :func:`~pystratify.solve`); the host must be lossless.
    kappa : Pasteur chirality parameters per layer (host 0), see
        :func:`~pystratify.solve_chiral`; complex for a lossy chiral response.
    position : (3,) or (P, 3) Cartesian emitter position(s), in any lossless
        layer (core, shell or host); a point on an interface belongs to the outer layer.
    moment : (3,) dipole moment (global axes, complex for elliptical dipoles):
        electric, unless ``dipole='magnetic'``.
    magnetic_moment : (3,) magnetic moment m emitted coherently with the electric
        ``moment`` - a chiral emitter when Im(p . m*) != 0 (dual moment, Gaussian
        units; SI: m / c; see :func:`~pystratify.dipole_far_field`).
    orientation : ``'fixed'``, ``'isotropic'`` (p and m rotated rigidly together)
        or an axis vector (random rotation about it).
    normalization : ``'host'`` - over the power of the same source in the unbounded
        host - or ``'layer'`` - in the unbounded medium of its own layer.
    l_max : truncation; ``None`` starts from the particle size and the distance to
        the nearest interface and doubles until every l-sum meets ``tol`` (up to ``l_cap``).
    l_cap : most orders tried; ``None`` for 20000 on the normalized route (whose cost per
        order is that of :func:`~pystratify.decay_rates`) and 1200 on the logarithmic one.
    tol : relative accuracy of every l-sum (remainder estimated from the last terms).
    quadrature_nodes : Gauss-Legendre nodes per absorbing layer (default: enough
        for polynomial exactness at the highest order and 6 per radian of phase).
    sheets : 2D materials on interfaces, ``{j: Sheet(...)}`` (see :mod:`pystratify.sheets`);
        their absorption is reported in ``sheet_absorption``.
    magnetic_convention : ``'dual'`` or ``'current'`` (a current-loop moment m_A: the dual
        moment mu_d m_A and, in a chiral layer, the electric dipole i kappa_d m_A), as
        :func:`~pystratify.dipole_far_field`.
    quadrupole : (3, 3) electric quadrupole moment Q_ij = int (3 x_i x_j - r^2 delta_ij) rho dV
        (Jackson; global axes, symmetric, complex allowed; the trace is dropped), emitted
        coherently with the dipoles: it couples through (1/6) Q : grad E and radiates
        (k^2/120) sum |Q_ij|^2 in units of |p|^2 when free.  Orientation averages rotate
        p, m and Q rigidly (exact averages over the icosahedral group).
    magnetic_quadrupole : (3, 3) magnetic quadrupole in the dual convention of m, emitted
        coherently with the other moments: it enters as Q_e + i s Q_m / Z in the helicity
        channel s, as m enters q_s = p + i s m / Z, so that a dual (eps = mu) structure gives it
        the rates of the electric quadrupole Q_e = Q_m (duality).  Rates only (no far field).
    route : ``'auto'`` (default) or ``'normalized'`` - total, shift and radiative rates from
        the normalized formulation, in the TE/TM basis with real source weights, whenever
        every layer is achiral (sheets included); absorption per layer and per sheet from
        the logarithmic route - or ``'log'`` - everything from the helicity-basis logarithmic
        route below, without the shift.  ``'auto'`` falls back to ``'log'`` for chiral layers.
    """
    return _emission(
        radii, n, wavelength, position, moment, magnetic_moment, orientation, mu, kappa, dipole, normalization,
        l_max, tol, l_cap, quadrature_nodes, warn, sheets, magnetic_convention, quadrupole, route=route,
        magnetic_quadrupole=magnetic_quadrupole,
    )[0]  # fmt: skip


def _emission(
    radii, n, wavelength, position, moment, magnetic_moment, orientation, mu, kappa, dipole, normalization,
    l_max, tol, l_cap, quadrature_nodes, warn, sheets, magnetic_convention, quadrupole, directions=None, route="auto",
    magnetic_quadrupole=None,
):  # fmt: skip
    """Body of :func:`emission_rates`; with ``directions`` = (theta, phi) also the far-field pattern
    of a single emitter, synthesised from the host amplitudes of the Green's function."""
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    n = np.atleast_1d(np.asarray(n, dtype=complex))
    mu = np.ones(n.size, dtype=complex) if mu is None else np.atleast_1d(np.asarray(mu, dtype=complex))
    kappa = np.zeros(n.size, dtype=complex) if kappa is None else np.atleast_1d(np.asarray(kappa, dtype=complex))
    positions = np.asarray(position, dtype=float)
    single = positions.ndim == 1
    positions = np.atleast_2d(positions)
    moment = np.asarray(moment, dtype=complex).ravel()
    if dipole not in ("electric", "magnetic"):
        raise ValueError("dipole must be 'electric' or 'magnetic'")
    if normalization not in ("host", "layer"):
        raise ValueError("normalization must be 'host' or 'layer'")
    if np.ndim(wavelength) != 0:
        raise ValueError("emission_rates takes a single wavelength")
    if n.shape != (radii.size + 1,) or mu.shape != n.shape or kappa.shape != n.shape:
        raise ValueError(f"n, mu and kappa need shape ({radii.size + 1},), host last")
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("position must be a finite 3-vector or an array of them, shape (P, 3)")
    if moment.shape != (3,) or not np.all(np.isfinite(moment)):
        raise ValueError("moment must be a finite 3-vector")
    if magnetic_moment is not None:
        if dipole != "electric":
            raise ValueError("with magnetic_moment, moment is the electric dipole: leave dipole='electric'")
        magnetic_moment = np.asarray(magnetic_moment, dtype=complex).ravel()
        if magnetic_moment.shape != (3,) or not np.all(np.isfinite(magnetic_moment)):
            raise ValueError("magnetic_moment must be a finite 3-vector")
    if n[-1].imag != 0 or mu[-1].imag != 0 or n[-1].real <= 0 or kappa[-1] != 0:
        raise ValueError("the host must be lossless and achiral")

    if dipole == "magnetic":
        p, m = np.zeros(3, complex), moment
    else:
        p, m = moment, (np.zeros(3, complex) if magnetic_moment is None else magnetic_moment)

    def _symmetric_traceless(q, name):
        if q is None:
            return None
        q = np.asarray(q, dtype=complex)
        if q.shape != (3, 3) or not np.all(np.isfinite(q)):
            raise ValueError(f"{name} must be a finite (3, 3) array")
        if np.abs(q - q.T).max() > 1e-12 * max(np.abs(q).max(), 1e-300):
            raise ValueError(f"{name} must be symmetric (an antisymmetric part is a dipole)")
        q = q - np.trace(q) / 3 * np.eye(3)
        return q if np.any(q) else None

    quadrupole = _symmetric_traceless(quadrupole, "quadrupole")
    magnetic_quadrupole = _symmetric_traceless(magnetic_quadrupole, "magnetic_quadrupole")
    if magnetic_quadrupole is not None:
        if directions is not None:
            raise ValueError("far fields of magnetic quadrupoles are not implemented")
        if quadrupole is None:
            quadrupole = np.zeros((3, 3), dtype=complex)
    if not (np.any(p) or np.any(m) or quadrupole is not None):
        raise ValueError("the source must have a nonzero moment")
    if magnetic_convention not in ("dual", "current"):
        raise ValueError("magnetic_convention must be 'dual' or 'current'")
    if quadrupole is None:
        cov = source_covariance(p, m, orientation)
    else:
        cov = _group_covariance(p, m, quadrupole, orientation, magnetic_quadrupole)
    orientation_name = orientation if isinstance(orientation, str) else "axis"
    if orientation_name == "fixed":  # the source itself: an eigenvector of s s^H has an arbitrary phase
        parts = [p, m] + ([] if quadrupole is None else [quadrupole.ravel()])
        parts += [] if magnetic_quadrupole is None else [magnetic_quadrupole.ravel()]
        sources = np.concatenate(parts)[:, None]
    else:
        weights, vectors = np.linalg.eigh(cov)
        keep = weights > 1e-14 * weights.max()
        sources = vectors[:, keep] * np.sqrt(weights[keep])  # cov = sources sources^H
    zh = (mu[-1] / n[-1]).real
    kh = 2 * np.pi * n[-1].real / wavelength
    current = magnetic_convention == "current"
    in_host = mu[-1].real if current else 1.0  # a current-loop m_A is the dual moment mu m_A
    s_ref = np.trace(cov[:3, :3]) + in_host**2 * np.trace(cov[3:6, 3:6]) / zh**2
    s_ref = s_ref + kh**2 / 120 * np.trace(cov[6:15, 6:15])
    s_ref = float(np.real(s_ref + kh**2 / 120 * np.trace(cov[15:, 15:]) / zh**2))

    r = np.maximum(np.linalg.norm(positions, axis=1), 1e-9 * radii[0])
    shells = locate_shell(radii, r)
    lossy_here = (n[shells].imag != 0) | (mu[shells].imag != 0) | (kappa[shells].imag != 0)
    if np.any(lossy_here):
        raise ValueError("emitter inside an absorbing (or gain) layer: the rates are undefined")
    frames = np.array([_local_frame(v) for v in positions])  # rows: local axes
    # a current loop m_A in a Pasteur medium (D = eps E + i kappa H) acts as the dual moment
    # mu_d m_A together with the electric dipole i kappa_d m_A
    at_source = mu[shells].real if current else np.ones(shells.size)
    chiral_part = 1j * kappa[shells].real if current else np.zeros(shells.size)
    m_local = np.einsum("pij,jk->pki", frames, sources[3:6])
    parts = [
        np.einsum("pij,jk->pki", frames, sources[:3]) + chiral_part[:, None, None] * m_local,
        at_source[:, None, None] * m_local,
    ]
    if quadrupole is not None:  # Q -> F Q F^T, i.e. (F x F) vec(Q)
        pairs = np.einsum("pai,pbj->pabij", frames, frames).reshape(-1, 9, 9)
        parts.append(np.einsum("pij,jk->pki", pairs, sources[6:15]))
        if magnetic_quadrupole is not None:
            parts.append(np.einsum("pij,jk->pki", pairs, sources[15:24]))
    local = np.concatenate(parts, axis=2)  # (P, K, 6, 15 or 24)

    notes = []
    eps = n * n / mu
    lossy = (eps.imag != 0) | (mu.imag != 0) | (kappa.imag != 0)
    passive = (eps.imag >= 0) & (mu.imag >= 0) & (eps.imag * mu.imag >= kappa.imag**2 * (1 - 1e-12))
    if np.any(lossy & ~passive):
        notes.append(
            "layer(s) " + ", ".join(map(str, np.flatnonzero(lossy & ~passive))) + " not passive: eps'' mu'' < "
            "kappa''^2 or a negative loss - absorption can be negative (for one handedness); circular dichroism "
            "(Im kappa) needs magnetic loss too"
        )
        if warn:
            warnings.warn(notes[-1], RuntimeWarning, stacklevel=3)
    sigma, zeta = _sheet_arrays(sheets, radii.size, 1)
    if np.any(sigma.real < 0) or np.any(zeta.imag < 0):
        notes.append("sheet(s) with Re(conductivity) < 0 or Im(normal) < 0: gain")

    if route not in ("auto", "normalized", "log"):
        raise ValueError("route must be 'auto', 'normalized' or 'log'")
    normalized = route != "log" and not np.any(kappa) and directions is None
    if route == "normalized" and not normalized:
        raise ValueError("chiral layers are not yet part of the normalized formulation: use route='auto' or 'log'")
    if not normalized and _feibelman_arrays(sheets, radii.size, 1) is not None:
        raise ValueError("d-parameters (Feibelman) are on the normalized route only (achiral layers, no pattern)")
    if l_cap is None:
        l_cap = 20000 if normalized else 1200
    if normalized:
        res, ok, L = _rates_normalized(
            radii, n, mu, wavelength, r, shells, local, tol, quadrature_nodes, l_max, l_cap, quadrupole is not None,
            sheets,
        )  # fmt: skip
        total, radiative, absorption, sheet_absorption, free, shift = (
            res[key] for key in ("total", "radiative", "absorption", "sheet_absorption", "free", "shift")
        )
    else:
        if l_max is not None:
            L = int(l_max)
        else:
            L = _starting_order(radii, n, kappa, wavelength, r, tol, l_cap)
        while True:
            sol = solve_chiral(radii, n, kappa, wavelength, mu, l_max=L, sheets=sheets)
            total, radiative, absorption, sheet_absorption, free, ok, ok_abs = _rates_one(
                sol, r, shells, local, tol, quadrature_nodes, quadrupole=quadrupole is not None
            )
            ok = ok & ok_abs
            if l_max is not None or ok.all() or L >= l_cap:
                break
            L = min(2 * L, l_cap)
        shift = np.full(r.size, np.nan)
    if not ok.all():
        notes.append(
            f"l-sum not converged to tol={tol:g} at {np.count_nonzero(~ok)} of {ok.size} position(s) "
            f"with l_max={L}; the emitter is very close to an interface"
        )
        if warn:
            warnings.warn(notes[-1], RuntimeWarning, stacklevel=3)
    if not normalized:
        # next to a lossless interface the logarithmic route loses the small real parts of the reflected
        # terms while its l-sums still look converged: energy conservation exposes it
        nonrad = absorption.sum(axis=-1) + sheet_absorption.sum(axis=-1)
        balance = np.abs(total - radiative.sum(axis=-1) - nonrad) / np.abs(total)
        off = ok & ~(balance <= tol)
        if off.any():
            ok = ok & ~off
            notes.append(
                f"energy balance off by up to {balance[off].max():.1e} > tol={tol:g} at {np.count_nonzero(off)} "
                "position(s): the logarithmic route loses the small real parts of the reflected terms next to "
                "lossless interfaces (route='auto' avoids this for achiral layers without sheets)"
            )
            if warn:
                warnings.warn(notes[-1], RuntimeWarning, stacklevel=3)
    pattern = None
    if directions is not None:
        pattern = _pattern(sol, shells[0], r[0], frames[0], positions[0], local[0], sources, directions, current)
        pattern["reference"] = 8 * np.pi / 3 * s_ref
    scale = s_ref * (free / s_ref if normalization == "layer" else 1.0)
    out = {
        "total": total / scale,
        "radiative": radiative.sum(axis=1) / scale,
        "radiative_helicity": radiative / np.asarray(scale)[..., None],
        "absorption": absorption / np.asarray(scale)[..., None],
        "sheet_absorption": sheet_absorption / np.asarray(scale)[..., None],
        "free_in_layer": free / s_ref,
        "shift": shift / scale,
    }
    if single:
        out = {key: value[0] for key, value in out.items()}
    rates = EmissionRates(
        position=positions[0] if single else positions,
        shell=shells[0] if single else shells,
        orders_used=L,
        converged=ok[0] if single else ok,
        normalization=normalization,
        orientation=orientation_name,
        notes=tuple(notes),
        route="normalized" if normalized else "log",
        **out,
    )
    return rates, pattern
