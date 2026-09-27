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

The total rate is computed independently of the radiative and Ohmic parts,
and ``balance_error = |total - rad - nonrad| / total`` checks energy
conservation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .convergence import truncation_order
from .energy import gauss_legendre
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


def _quadrature_rule(sol: Solution, a: int, count: int, nodes=None):
    """Nodes, weights, k r and the special functions for shell ``a`` (reused by every field)."""
    k = sol.k[0, a]
    r_in = sol.radii[a - 1] if a else 0.0
    r_out = sol.radii[a]
    nodes = nodes or int(max(64, count + 48, 6 * abs(k) * (r_out - r_in)))
    t, wt = gauss_legendre(nodes)
    half = 0.5 * (r_out - r_in)
    x = k * (half * t + 0.5 * (r_out + r_in))
    log_psi, log_xi = log_riccati(x, count)
    return half * wt, x, log_psi, log_xi


def _log_absorption_integrals(sol: Solution, a: int, rule, log_amp_psi, log_amp_xi):
    """Logs of the absorption integrals of shell ``a`` for the field A psi_l + B xi_l:

        TE:  int |A j_l + B h_l|^2 r^2 dr
        TM:  ( l(l+1) int |A j_l + B h_l|^2 dr + int |A psi_l' + B xi_l'|^2 dr ) / |k|^2

    by Gauss-Legendre quadrature with a per-order logarithmic shift, so any
    order is representable.
    """
    wt, x, log_psi, log_xi = rule
    L = log_amp_psi.size
    l = sol.orders[:L]
    k2 = abs(sol.k[0, a]) ** 2
    tp = log_amp_psi + log_psi[:, l]
    tx = log_amp_xi + log_xi[:, l]
    tp_prev = log_amp_psi + log_psi[:, l - 1]
    tx_prev = log_amp_xi + log_xi[:, l - 1]
    with np.errstate(all="ignore"):
        shift = np.maximum(np.maximum(tp.real, tx.real), np.maximum(tp_prev.real, tx_prev.real)).max(axis=0)
        shift = np.where(np.isfinite(shift), shift, 0.0)
        u = np.exp(tp - shift) + np.exp(tx - shift)
        du = np.exp(tp_prev - shift) + np.exp(tx_prev - shift) - l * u / x[:, None]
        u2 = u.real**2 + u.imag**2
        te = (wt @ u2) / k2
        tm = (l * (l + 1) * (wt @ (u2 / (x.real**2 + x.imag**2)[:, None])) + wt @ (du.real**2 + du.imag**2)) / k2
        return np.log(tm) + 2 * shift, np.log(te) + 2 * shift


def _tail_estimate(terms):
    """Remainder of a series estimated from its last terms (axis 0 = order).

    The decay-rate series are asymptotically geometric in l, ratio
    (r_< / r_>)^2 times a polynomial, so the neglected tail is ~ t_L q/(1 - q)
    with q the worst recent ratio: far larger than t_L when q -> 1.  +inf
    where the terms are not decreasing.
    """
    a = np.abs(terms[-4:])
    with np.errstate(all="ignore"):
        q = np.max(a[1:] / a[:-1], axis=0)
        return np.where(a[-1] == 0, 0.0, np.where(q < 1, a[-1] * q / (1 - q), np.inf))


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


def _orders_needed(q, tol, l_cap):
    """Orders for a geometric series of ratio q^2 to leave a tail below tol."""
    if q >= 1:
        return l_cap
    return 1.25 * np.log(tol * (1 - q * q)) / (2 * np.log(q)) + 16


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


def _rates_one(sol: Solution, r, shells, normalization, dipole, tol, nodes):
    """Terms of the six l-sums for a single-wavelength Solution; returns the converged sums."""
    l = sol.orders
    L = l.size
    N = sol.n_shells
    n, mu = sol.n[0], sol.mu[0]
    eps = n**2 / mu
    k = sol.k[0]
    radial_pol, other_pol = (TM, TE) if dipole == "electric" else (TE, TM)
    integral_kind = {TM: 0, TE: 1}
    absorbing = [a for a in range(N) if eps[a].imag > 0]
    # The Ohmic-loss series converges like (r_< / r_>)^(2l) at the absorbing shell's
    # nearest boundary - usually far sooner than the LDOS series, which also feels
    # nearby dielectric interfaces.  Integrate only that many orders (checked below).
    bounds = np.array([b for a in absorbing for b in ((sol.radii[a - 1],) if a else ()) + (sol.radii[a],)])
    if bounds.size:
        q = np.max(np.minimum(bounds[None, :] / r[:, None], r[:, None] / bounds[None, :]))
        l_abs = int(min(L, max(32, _orders_needed(q, tol, L))))
    else:
        l_abs = L
    cache, rules = {}, {}

    def integral(a, side, pol, count):
        key = (a, side, pol, count)
        if key not in cache:
            if (a, count) not in rules:
                rules[a, count] = _quadrature_rule(sol, a, count, nodes)
            if side == "below":  # regular solution in shell a, plane-wave normalisation
                la, lb = sol.log_a[pol, a, 0], sol.log_b[pol, a, 0]
            else:  # outgoing solution in shell a, B = 1 in the host
                lb = sol.log_b_out[pol, a, 0]
                la = lb + sol.log_s[pol, a, 0]
            v = np.full(L, -np.inf, dtype=complex)
            v[:count] = _log_absorption_integrals(sol, a, rules[a, count], la[:count], lb[:count])[integral_kind[pol]]
            cache[key] = v
        return cache[key]

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
                pre = kd3 * f_nonrad * eps[a].imag
                if a < d:  # regular solution in a, amplitude u_out(x)/Delta / A_d
                    side, amp, ref = "below", "u_out", "log_a"
                else:  # outgoing solution in a, amplitude u_in(x)/Delta / B_out,d
                    side, amp, ref = "above", "u_in", "log_b_out"
                c_e = e[amp] - e["log_delta"] - e[ref]
                dc_e = e["d" + amp] - e["log_delta"] - e[ref]
                c_m = m[amp] - m["log_delta"] - m[ref]
                i_e = integral(a, side, radial_pol, count).real
                i_m = integral(a, side, other_pol, count).real
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
    l_cap=1200,
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
