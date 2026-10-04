"""Decay rates of an electric or magnetic dipole near or inside a multilayered sphere.

Theory: Moroz, Ann. Phys. 315, 352 (2005); Rasskazov, Carney & Moroz, OSA
Continuum 3, 2290 (2020), Eqs. (28)-(34), with the corrections listed in
AUDIT.md (normalisation, spherical Hankel functions in the absorption
integrals, tolerance-controlled sums).

In the emitter's shell d (argument x = k_d r, real) let u_in = psi + R xi be
the regular and u_out = S psi + xi the outgoing solution of
:func:`~pystratify.solve`.  With rho = R xi/psi, sigma = S psi/xi (both O(1))
and Delta = 1 - R S, the Green's function gives

* total rate (LDOS):   Re[u_in u_out / Delta]
* radiated amplitude:  u_in / (B_out,d Delta), B_out,d the outgoing amplitude
  normalised to B = 1 in the host
* field in an absorbing shell a: the regular (a < d) or outgoing (a > d)
  solution continued to shell a, times u_out(x)/Delta or u_in(x)/Delta.

Losses are Im(eps)|E|^2 + Im(mu)|H|^2 in every lossy shell.  The total rate
is computed independently of the radiative and loss parts,
and ``balance_error = |total - rad - nonrad| / total`` checks energy
conservation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .convergence import orders_needed as _orders_needed
from .convergence import tail_estimate as _tail_estimate
from .convergence import truncation_order
from .energy import WEAK_LOSS, gauss_legendre
from .riccati import log_riccati
from .solver import TE, TM, Solution, solve

__all__ = ["DecayRates", "decay_rates", "locate_shell"]


def locate_shell(radii, r):
    """Shell index (0 = core, N = host) of radial position(s) r; interfaces belong outward."""
    return np.searchsorted(np.asarray(radii), np.asarray(r), side="right")


@dataclass(frozen=True)
class DecayRates:
    """Normalised decay rates, arrays of shape ``(len(r), 2)``.

    Column 0: radial (perpendicular) dipole; column 1: tangential (parallel).
    """

    r: np.ndarray
    radiative: np.ndarray
    nonradiative: np.ndarray
    total: np.ndarray
    shell: np.ndarray
    orders_used: np.ndarray
    converged: np.ndarray
    normalization: str
    dipole: str
    notes: tuple = field(default_factory=tuple)

    @property
    def balance_error(self) -> np.ndarray:
        """|total - (radiative + nonradiative)| / total per position and orientation."""
        return np.abs(self.total - self.radiative - self.nonradiative) / np.abs(self.total)

    @staticmethod
    def averaged(rates: np.ndarray) -> np.ndarray:
        """Orientation average (perpendicular + 2 parallel) / 3."""
        return (rates[:, 0] + 2 * rates[:, 1]) / 3

    def quantum_yield(self, intrinsic: float = 1.0) -> np.ndarray:
        """Orientation-averaged quantum yield for an emitter of intrinsic yield ``intrinsic``."""
        rad, nonrad = self.averaged(self.radiative), self.averaged(self.nonradiative)
        return rad / (rad + nonrad + (1 - intrinsic) / intrinsic)


def _normalization(n, mu, d, normalization, dipole):
    """(radiative, nonradiative, total) factors relative to the free-space rate.

    The shell factors follow from the Green's function; normalising to the
    host multiplies by Gamma_0(shell) / Gamma_0(host), which is n mu for an
    electric and n eps for a magnetic dipole.
    """
    nd, nh, md, mh = n[d], n[-1], mu[d], mu[-1]
    radiative = ((nd * md) / (nh * mh)).real
    nonradiative = (md / nd**2).real
    if normalization == "shell":
        ratio = 1.0
    elif dipole == "electric":
        ratio = ((nd * md) / (nh * mh)).real
    else:
        ratio = ((nd**3 / md) / (nh**3 / mh)).real
    return radiative * ratio, nonradiative * ratio, ratio


def _log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a)


#: closed-form absorption integrals are used while they lose fewer digits than this
_LOMMEL_MAX_CANCELLATION = 1e5
#: a boundary term is taken from its direct formula while that loses fewer digits than this
_DIRECT_MAX_CANCELLATION = 1e3
#: extra orders above the top one, so the damped downward recurrence forgets its start
_RECURRENCE_HEADROOM = 16


def _pick(direct, cancellation, recurred):
    return direct if cancellation < _DIRECT_MAX_CANCELLATION else recurred


def _lommel_boundary(k, radius, count):
    """Logs of the Lommel boundary terms at r = ``radius`` for orders nu = 0..count + 1.

    With j = j_nu(k r), h = h_nu(k r) and complex k (Im k^2 > 0):

        int |j|^2 r^2 dr  = [ r^2 Im(k* j_{nu-1}* j_nu) / Im(k^2) ]        = [ G_jj ]
        int |h|^2 r^2 dr  = [ r^2 Im(k* h_{nu-1}* h_nu) / Im(k^2) ]        = [ G_hh ]
        int j h* r^2 dr   = [ r^2 (k* j_nu h_{nu-1}* - k j_{nu-1} h_nu*) / (k^2 - k*^2) ] = [ G_jh ]

    Past the turning point k* j_{nu-1}* j_nu is real to leading order, so its
    imaginary part is a small difference (it would lose ~log10(2 nu^2/Im x^2)
    digits).  Instead, with rho = j_nu/j_{nu-1}, sigma = h_{nu-1}/h_nu and
    J = Im(k* rho), K = Im(k sigma), the three-term recurrence gives

        J_nu = |k|^2 r^2 (J_{nu+1} + 2 Im k Re rho_{nu+1}) / |2 nu + 1 - x rho_{nu+1}|^2     (downward)
        K_nu = r (Im k^2 Re E + Re k^2 r K_{nu-1}) / |E|^2,  E = 2 nu - 1 - x sigma_{nu-1}   (upward)

    both sums of positive terms at high order and damped in their direction.
    Each order takes the direct value where that is well conditioned.
    Returns (log G_jj, log(-G_hh), log G_jh), G_jj > 0 > G_hh.
    """
    x = k * radius
    top = count + 1
    log_psi, log_xi = log_riccati(np.array([x]), top + _RECURRENCE_HEADROOM + 1)
    with np.errstate(all="ignore"):
        m = abs(x.imag)  # psi_{-1} = cos x, xi_{-1} = exp(i x)
        log_psi = np.r_[m + np.log((np.exp(1j * x - m) + np.exp(-1j * x - m)) / 2), log_psi[0]]
        log_xi = np.r_[1j * x, log_xi[0]]
    lj, lh = log_psi - np.log(x), log_xi - np.log(x)  # index nu + 1
    with np.errstate(all="ignore"):
        rho = np.exp(lj[1:] - lj[:-1])  # j_nu / j_{nu-1}, nu = 0..
        sigma = np.exp(lh[:-1] - lh[1:])  # h_{nu-1} / h_nu
    nu = np.arange(rho.size)
    k2 = k * k
    direct_j, direct_h = np.imag(np.conj(k) * rho), np.imag(k * sigma)
    with np.errstate(all="ignore"):
        cond_j = np.abs(k * rho) / np.abs(direct_j)
        cond_h = np.abs(k * sigma) / np.abs(direct_h)
    j_rec = np.empty(nu.size)
    j_rec[-1] = direct_j[-1]
    for i in range(nu.size - 2, -1, -1):
        prev = _pick(direct_j[i + 1], cond_j[i + 1], j_rec[i + 1])
        d = 2 * nu[i] + 1 - x * rho[i + 1]
        j_rec[i] = abs(k) ** 2 * radius**2 * (prev + 2 * k.imag * rho[i + 1].real) / abs(d) ** 2
    h_rec = np.empty(nu.size)
    h_rec[0] = direct_h[0]  # sigma_0 = i exactly
    for i in range(1, nu.size):
        prev = _pick(direct_h[i - 1], cond_h[i - 1], h_rec[i - 1])
        e = 2 * nu[i] - 1 - x * sigma[i - 1]
        h_rec[i] = radius * (k2.imag * e.real + k2.real * radius * prev) / abs(e) ** 2
    big_j = np.where(cond_j < _DIRECT_MAX_CANCELLATION, direct_j, j_rec)[: top + 1]
    big_k = np.where(cond_h < _DIRECT_MAX_CANCELLATION, direct_h, h_rec)[: top + 1]
    n = slice(0, top + 1)
    log_r2 = 2 * np.log(radius)
    with np.errstate(all="ignore"):
        log_gjj = log_r2 + 2 * lj[:-1][n].real + np.log(big_j) - np.log(k2.imag)
        log_ghh = log_r2 + 2 * lh[1:][n].real + np.log(big_k) - np.log(k2.imag)
        cross = np.conj(k) * rho[n] * np.conj(sigma[n]) - k
        log_gjh = log_r2 + lj[:-1][n] + np.conj(lh[1:][n]) + np.log(cross) - np.log(2j * k2.imag)
    return log_gjj, log_ghh, log_gjh


def _log_lommel_integrals(sol: Solution, a: int, count: int, log_amp_psi, log_amp_xi, boundary):
    """The absorption integrals of :func:`_log_absorption_integrals` in closed form.

    |A j + B h|^2 = |A|^2 |j|^2 + |B|^2 |h|^2 + 2 Re(A B* j h*), each integrated
    with the boundary terms of :func:`_lommel_boundary` (``boundary[i]`` for
    the outer and, for a shell, inner radius), and the TM integrand is
    ((l+1)|f_{l-1}|^2 + l |f_{l+1}|^2)/(2l+1).  O(L) instead of the O(L^2) of
    quadrature.  Returns (log TM, log TE, cancellation), the last being the
    worst ratio of the summed magnitudes to the result.
    """
    l = sol.orders[:count]
    la, lb = log_amp_psi[None, :], log_amp_xi[None, :]
    nu = l[None, :] + np.array([-1, 0, 1])[:, None]  # orders l-1, l, l+1
    pieces = []
    for sign, (gjj, ghh, gjh) in zip((1.0, -1.0), boundary):
        pieces += [
            (sign, 2 * la.real + gjj[nu]),
            (-sign, 2 * lb.real + ghh[nu]),
            (2 * sign, la + np.conj(lb) + gjh[nu]),
        ]
    with np.errstate(all="ignore"):
        shift = np.max([np.real(t) for _, t in pieces], axis=(0, 1))
        shift = np.where(np.isfinite(shift), shift, 0.0)
        values = [c * np.real(np.exp(t - shift)) for c, t in pieces]
        integral = sum(values)
        scale = sum(np.abs(v) for v in values)
        cancellation = float(np.max(np.where(scale > 0, scale / np.abs(integral), 1.0)))
        tm = ((l + 1) * integral[0] + l * integral[2]) / (2 * l + 1)
        return np.log(tm) + shift, np.log(integral[1]) + shift, cancellation


#: special functions of a quadrature rule are cached up to this many (node, order) values
_RULE_CACHE_SIZE = 4_000_000


def _quadrature_rule(sol: Solution, a: int, count: int, nodes=None):
    """Weights, k r and (when small enough to keep) the special functions for shell ``a``."""
    k = sol.k[0, a]
    r_in = sol.radii[a - 1] if a else 0.0
    r_out = sol.radii[a]
    nodes = nodes or int(max(64, count + 48, 6 * abs(k) * (r_out - r_in)))
    t, wt = gauss_legendre(nodes)
    half = 0.5 * (r_out - r_in)
    x = k * (half * t + 0.5 * (r_out + r_in))
    logs = log_riccati(x, count) if nodes * count <= _RULE_CACHE_SIZE else None
    return half * wt, x, logs


def _log_absorption_integrals(sol: Solution, a: int, rule, log_amp_psi, log_amp_xi):
    """Logs of the absorption integrals of shell ``a`` for the field A psi_l + B xi_l:

        TE:  int |A j_l + B h_l|^2 r^2 dr
        TM:  ( l(l+1) int |A j_l + B h_l|^2 dr + int |A psi_l' + B xi_l'|^2 dr ) / |k|^2

    by Gauss-Legendre quadrature with a per-order logarithmic shift, so any
    order is representable.  Large rules are processed in chunks of nodes
    with a running shift, so memory stays bounded.
    """
    wt, x, logs = rule
    L = log_amp_psi.size
    l = sol.orders[:L]
    k2 = abs(sol.k[0, a]) ** 2
    step = max(1, _RULE_CACHE_SIZE // max(L, 1))
    shift = np.full(L, -np.inf)
    te = np.zeros(L)
    tm = np.zeros(L)
    for start in range(0, x.size, step):
        part = slice(start, start + step)
        log_psi, log_xi = (logs[0][part], logs[1][part]) if logs is not None else log_riccati(x[part], L)
        xs, ws = x[part], wt[part]
        tp = log_amp_psi + log_psi[:, l]
        tx = log_amp_xi + log_xi[:, l]
        tp_prev = log_amp_psi + log_psi[:, l - 1]
        tx_prev = log_amp_xi + log_xi[:, l - 1]
        with np.errstate(all="ignore"):
            local = np.maximum(np.maximum(tp.real, tx.real), np.maximum(tp_prev.real, tx_prev.real)).max(axis=0)
            new = np.fmax(shift, local)
            new = np.where(np.isfinite(new), new, 0.0)
            rescale = np.where(np.isfinite(shift), np.exp(2 * (shift - new)), 0.0)
            u = np.exp(tp - new) + np.exp(tx - new)
            du = np.exp(tp_prev - new) + np.exp(tx_prev - new) - l * u / xs[:, None]
            u2 = u.real**2 + u.imag**2
            te = te * rescale + (ws @ u2) / k2
            tm = (
                tm * rescale
                + (l * (l + 1) * (ws @ (u2 / (xs.real**2 + xs.imag**2)[:, None])) + ws @ (du.real**2 + du.imag**2)) / k2
            )
        shift = new
    with np.errstate(divide="ignore"):
        return np.log(tm) + 2 * shift, np.log(te) + 2 * shift


def _converged_sum(terms, tol):
    """Sum over orders (axis 1 of (P, L, K)) with a remainder-based convergence test.

    A position whose terms turn non-finite is summed up to that order and
    flagged unconverged.
    """
    P, L, K = terms.shape
    finite = np.all(np.isfinite(terms), axis=2)
    stop = np.where(finite.all(axis=1), L, np.argmin(finite, axis=1))
    upto = np.arange(L)[None, :] < stop[:, None]
    total = np.where(upto[..., None], terms, 0).sum(axis=1)
    last = np.clip(stop[:, None] - 4 + np.arange(4)[None, :], 0, L - 1)
    tail = _tail_estimate(np.moveaxis(np.take_along_axis(terms, last[..., None], axis=1), 1, 0))
    scale = np.maximum(np.abs(total), np.finfo(float).tiny)
    converged = (stop == L) & np.all(tail <= tol * scale, axis=1)
    total[stop == 0] = np.nan
    return total, np.where(stop >= 4, stop, 0), converged


def _emitter_side(sol, d, x, pol):
    """Logs of the regular/outgoing solutions and their derivatives at the emitter."""
    l = sol.orders
    log_psi, log_xi = log_riccati(x.astype(complex), l.size)
    lp, lx = log_psi[:, l], log_xi[:, l]
    log_r, log_s = sol.log_r[pol, d, 0], sol.log_s[pol, d, 0]
    with np.errstate(all="ignore"):
        rho = np.exp(log_r + lx - lp)
        sigma = np.exp(log_s + lp - lx)
        d_psi = np.exp(log_psi[:, l - 1] - lp) - l / x[:, None]
        d_xi = np.exp(log_xi[:, l - 1] - lx) - l / x[:, None]
        return {
            "u_in": lp + _log(1 + rho),
            "du_in": lp + _log(d_psi + rho * d_xi),
            "u_out": lx + _log(1 + sigma),
            "du_out": lx + _log(sigma * d_psi + d_xi),
            "log_delta": np.log(1 - np.exp(log_r + log_s)),
            "log_b_out": sol.log_b_out[pol, d, 0],
            "log_a": sol.log_a[pol, d, 0],
            "log_r": log_r,
            "log_s": log_s,
            "lp": lp,
            "lx": lx,
            "lp_d": lp + _log(d_psi),
            "lx_d": lx + _log(d_xi),
        }


def _ldos(side, derivative):
    """Re[u_in u_out / Delta] written so every term keeps full relative accuracy:
    psi xi + (R xi^2 + S psi^2 + 2 R S psi xi) / (1 - R S), with Re(psi xi) = psi^2
    for real x (Re(psi xi) itself is ~1e-600 at high l while Im(psi xi) ~ 1/l)."""
    p, x = (side["lp_d"], side["lx_d"]) if derivative else (side["lp"], side["lx"])
    lr, ls = side["log_r"], side["log_s"]
    with np.errstate(all="ignore"):
        bracket = np.exp(lr + 2 * x) + np.exp(ls + 2 * p) + 2 * np.exp(lr + ls + p + x)
        return np.exp(2 * p).real + (bracket * np.exp(-side["log_delta"])).real


def _rates_one(sol: Solution, r, shells, normalization, dipole, tol, quadrature_nodes):
    """Terms of the six l-sums for a single-wavelength Solution; returns the converged sums."""
    l = sol.orders
    L = l.size
    N = sol.n_shells
    n, mu = sol.n[0], sol.mu[0]
    eps = n**2 / mu
    k = sol.k[0]
    radial_pol, other_pol = (TM, TE) if dipole == "electric" else (TE, TM)
    integral_kind = {TM: 0, TE: 1}
    # Ohmic loss Im(eps)|E|^2 and magnetic loss Im(mu)|H|^2; in shell a, H = -i (n/mu) x (E of the
    # other polarisation's radial form), so the magnetic loss uses the other integral kind, times |eps/mu|
    loss_e = np.where(eps.imag > 0, eps.imag, 0.0)
    loss_m = np.where(mu.imag > 0, mu.imag * np.abs(eps / mu), 0.0)
    absorbing = [a for a in range(N) if loss_e[a] > 0 or loss_m[a] > 0]
    # The Ohmic-loss series converges like (r_< / r_>)^(2l) at the absorbing shell's
    # nearest boundary - usually far sooner than the LDOS series, which also feels
    # nearby dielectric interfaces.  Integrate only that many orders (checked below).
    bounds = np.array([b for a in absorbing for b in ((sol.radii[a - 1],) if a else ()) + (sol.radii[a],)])
    if bounds.size:
        q = np.max(np.minimum(bounds[None, :] / r[:, None], r[:, None] / bounds[None, :]))
        l_abs = int(min(L, max(32, _orders_needed(q, tol, L))))
    else:
        l_abs = L
    cache, rules, boundaries = {}, {}, {}

    def integral(a, side, pol, count, kind):
        key = (a, side, pol, count, kind)
        if key not in cache:
            if side == "below":  # regular solution in shell a, plane-wave normalisation
                la, lb = sol.log_a[pol, a, 0], sol.log_b[pol, a, 0]
            else:  # outgoing solution in shell a, B = 1 in the host
                lb = sol.log_b_out[pol, a, 0]
                la = lb + sol.log_s[pol, a, 0]
            la, lb = la[:count], lb[:count]
            result, cancellation = None, np.inf
            if quadrature_nodes is None and abs(k[a].imag) >= WEAK_LOSS * abs(k[a]):
                if (a, count) not in boundaries:
                    radii = [sol.radii[a]] + ([sol.radii[a - 1]] if a else [])
                    boundaries[a, count] = [_lommel_boundary(k[a], radius, count) for radius in radii]
                *result, cancellation = _log_lommel_integrals(sol, a, count, la, lb, boundaries[a, count])
            if cancellation > _LOMMEL_MAX_CANCELLATION:  # weak loss, or a closed form that cancels
                if (a, count) not in rules:
                    rules[a, count] = _quadrature_rule(sol, a, count, quadrature_nodes)
                result = _log_absorption_integrals(sol, a, rules[a, count], la, lb)
            for which in (0, 1):
                v = np.full(L, -np.inf, dtype=complex)
                v[:count] = result[which]
                cache[a, side, pol, count, which] = v
        return cache[key]

    def log_loss(a, side, pol, count):
        """log of the loss integral of shell a for polarisation pol, electric plus magnetic."""
        kind = integral_kind[pol]
        with np.errstate(divide="ignore"):
            electric = np.log(loss_e[a]) + integral(a, side, pol, count, kind).real if loss_e[a] > 0 else None
            magnetic = np.log(loss_m[a]) + integral(a, side, pol, count, 1 - kind).real if loss_m[a] > 0 else None
        if electric is None:
            return magnetic
        return electric if magnetic is None else np.logaddexp(electric, magnetic)

    c_radial = l * (l + 1) * (2 * l + 1)
    c_tangential = 2 * l + 1
    terms = np.zeros((r.size, L, 6))
    for d in np.unique(shells):
        idx = np.flatnonzero(shells == d)
        x = (k[d] * r[idx]).real  # the emitter's shell is lossless
        f_rad, f_nonrad, f_tot = _normalization(n, mu, d, normalization, dipole)
        e = _emitter_side(sol, d, x, radial_pol)
        m = _emitter_side(sol, d, x, other_pol)
        xx = x[:, None]

        def mag2(v):
            with np.errstate(under="ignore", over="ignore"):
                return np.exp(2 * np.real(v))

        # radiative: F = u_in / (B_out,d Delta)
        f_e = e["u_in"] - e["log_b_out"] - e["log_delta"]
        df_e = e["du_in"] - e["log_b_out"] - e["log_delta"]
        f_m = m["u_in"] - m["log_b_out"] - m["log_delta"]
        terms[idx, :, 0] = 1.5 / xx**4 * f_rad * c_radial * mag2(f_e)
        terms[idx, :, 1] = 0.75 / xx**2 * f_rad * c_tangential * (mag2(f_m) + mag2(df_e))

        # total, from the local density of states
        terms[idx, :, 4] = 1.5 / xx**4 * f_tot * c_radial * _ldos(e, False)
        terms[idx, :, 5] = 0.75 / xx**2 * f_tot * c_tangential * (_ldos(m, False) + _ldos(e, True))

        def nonradiative(count):
            out = np.zeros((idx.size, L, 2))
            kd3 = k[d].real ** 3
            for a in absorbing:
                pre = kd3 * f_nonrad
                if a < d:  # regular solution in a, amplitude u_out(x)/Delta / A_d
                    side, amp, ref = "below", "u_out", "log_a"
                else:  # outgoing solution in a, amplitude u_in(x)/Delta / B_out,d
                    side, amp, ref = "above", "u_in", "log_b_out"
                c_e = e[amp] - e["log_delta"] - e[ref]
                dc_e = e["d" + amp] - e["log_delta"] - e[ref]
                c_m = m[amp] - m["log_delta"] - m[ref]
                i_e = log_loss(a, side, radial_pol, count)
                i_m = log_loss(a, side, other_pol, count)
                with np.errstate(under="ignore"):
                    out[:, :, 0] += 1.5 / xx**4 * pre * c_radial * np.exp(i_e + 2 * c_e.real)
                    out[:, :, 1] += (
                        0.75 / xx**2 * pre * c_tangential * (np.exp(i_m + 2 * c_m.real) + np.exp(i_e + 2 * dc_e.real))
                    )
            return out

        nonrad = nonradiative(l_abs)
        if l_abs < L:
            total = np.abs(nonrad.sum(axis=1))
            tail = _tail_estimate(np.moveaxis(nonrad[:, :l_abs], 1, 0))
            if not np.all(tail <= 0.1 * tol * np.maximum(total, np.finfo(float).tiny)):
                nonrad = nonradiative(L)
        terms[idx, :, 2:4] = nonrad
    return _converged_sum(terms, tol)


def _rates_with_sheets(radii, n, wavelength, r, mu, l_max, tol, normalization, dipole, nodes, l_cap, warn, sheets):
    """Radial and tangential dipoles through the general route of :func:`~pystratify.emission_rates`."""
    from .rates import emission_rates

    positions = np.column_stack([np.zeros_like(r), np.zeros_like(r), r])
    runs = [
        emission_rates(
            radii,
            n,
            wavelength,
            positions,
            moment,
            mu=mu,
            dipole=dipole,
            l_max=l_max,
            tol=tol,
            l_cap=l_cap,
            quadrature_nodes=nodes,
            warn=False,
            sheets=sheets,
            normalization="layer" if normalization == "shell" else "host",
        )  # fmt: skip
        for moment in ([0, 0, 1.0], [1.0, 0, 0])
    ]
    ok = runs[0].converged & runs[1].converged
    notes = tuple(dict.fromkeys(note for run in runs for note in run.notes))
    if warn and not ok.all():
        warnings.warn(notes[-1] if notes else "l-sum not converged", RuntimeWarning, stacklevel=3)
    return DecayRates(
        r=r,
        radiative=np.stack([run.radiative for run in runs], axis=1),
        nonradiative=np.stack([run.nonradiative for run in runs], axis=1),
        total=np.stack([run.total for run in runs], axis=1),
        shell=runs[0].shell,
        orders_used=np.full(r.size, max(run.orders_used for run in runs)),
        converged=ok,
        normalization=normalization,
        dipole=dipole,
        notes=notes,
    )


def decay_rates(
    radii,
    n,
    wavelength,
    r,
    mu=None,
    l_max=None,
    tol=1e-6,
    normalization="host",
    dipole="electric",
    quadrature_nodes=None,
    l_cap=20000,
    warn=True,
    sheets=None,
) -> DecayRates:
    """Radiative, nonradiative and total decay rates of a dipole emitter.

    Parameters
    ----------
    radii, n, wavelength, mu : geometry and materials at one wavelength (see :func:`~pystratify.solve`).
    r : emitter distance(s) from the centre; not in an absorbing or gain
        shell, and the host must be lossless.
    l_max : truncation; ``None`` starts from a geometric estimate and doubles
        until every sum meets ``tol`` (up to ``l_cap``).
    tol : relative accuracy of every l-sum (remainder estimated from the
        geometric decay of the last terms).
    normalization : ``'host'`` or ``'shell'`` - free-space rate in the host or
        in the emitter's own shell.
    dipole : ``'electric'`` or ``'magnetic'``.
    quadrature_nodes : ``None`` integrates the Ohmic loss from closed-form
        (Lommel) boundary terms, falling back to Gauss-Legendre quadrature for
        weakly lossy shells; a number forces quadrature with that many nodes.
    l_cap : most orders tried; the cost is linear in the order, so 20000
        (an emitter 1 nm from a 1.3-um sphere at tol = 1e-9) takes ~1 s.
    sheets : 2D materials on interfaces (see :mod:`pystratify.sheets`); their
        absorption is part of the nonradiative rate.  Computed by
        :func:`~pystratify.emission_rates`.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    n = np.atleast_1d(np.asarray(n, dtype=complex))
    mu = np.ones(n.size, dtype=complex) if mu is None else np.atleast_1d(np.asarray(mu, dtype=complex))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if dipole not in ("electric", "magnetic"):
        raise ValueError("dipole must be 'electric' or 'magnetic'")
    if normalization not in ("host", "shell"):
        raise ValueError("normalization must be 'host' or 'shell'")
    if np.ndim(wavelength) != 0:
        raise ValueError("decay_rates takes a single wavelength")
    if n.shape != (radii.size + 1,) or mu.shape != n.shape:
        raise ValueError(f"n and mu need shape ({radii.size + 1},), host last")
    eps = n**2 / mu
    if eps[-1].imag != 0 or n[-1].imag != 0:
        raise ValueError("the host must be lossless for decay rates to be defined")
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("emitter positions must be positive and finite")
    shells = locate_shell(radii, r)
    if np.any(eps[shells].imag != 0) or np.any(n[shells].imag != 0):
        raise ValueError("emitter inside an absorbing or gain shell: the rates are undefined")
    if sheets:
        return _rates_with_sheets(
            radii, n, wavelength, r, mu, l_max, tol, normalization, dipole, quadrature_nodes, l_cap, warn, sheets
        )

    if l_max is not None:
        sol = solve(radii, n, wavelength, mu, int(l_max))
        values, used, ok = _rates_one(sol, r, shells, normalization, dipole, tol, quadrature_nodes)
    else:
        # the slowest series decays like (r_< / r_>)^(2l) at the interface nearest the
        # emitter: start where that reaches tol, then double while not converged
        q = np.max(np.minimum(radii[None, :] / r[:, None], r[:, None] / radii[None, :]))
        L = int(
            min(l_cap, max(32, truncation_order(radii[-1], n[-1], wavelength, "near"), _orders_needed(q, tol, l_cap)))
        )
        while True:
            sol = solve(radii, n, wavelength, mu, L)
            values, used, ok = _rates_one(sol, r, shells, normalization, dipole, tol, quadrature_nodes)
            if ok.all() or L >= l_cap:
                break
            L = min(2 * L, l_cap)
    notes = ()
    if not ok.all():
        notes = (
            f"l-sum not converged to tol={tol:g} at {np.count_nonzero(~ok)} of {ok.size} position(s) "
            f"with l_max={sol.orders.size}; the emitter is very close to an interface",
        )
        if warn:
            warnings.warn(notes[0], RuntimeWarning, stacklevel=2)
    return DecayRates(
        r=r,
        radiative=values[:, 0:2],
        nonradiative=values[:, 2:4],
        total=values[:, 4:6],
        shell=shells,
        orders_used=used,
        converged=ok,
        normalization=normalization,
        dipole=dipole,
        notes=notes,
    )
