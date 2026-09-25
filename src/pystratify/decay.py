"""Decay rates of an electric or magnetic dipole emitter near / inside a multilayered sphere.

Replaces STRATIFY ``decay/edcy.m``, ``mdcy.m`` and ``I_abs.m`` (OSAC Eqs.
28-34; Moroz, Ann. Phys. 315, 352 (2005)), with the corrections of AUDIT.md:
correct host/shell normalisation (M1, M4), spherical Hankel functions in the
absorption integrals (M2), tolerance-controlled l-sums (M3) and an
overflow-free formulation valid to l ~ 1000 (M9; cf. Majic & Le Ru, Appl.
Opt. 59, 1293 (2020)).

Formulation.  In the emitter's shell d (argument x = k_d r_d, real) let
u_in = psi + R xi be the regular and u_out = S psi + xi the outgoing
solution of :func:`pystratify.solve`.  With rho = R xi/psi, sigma = S psi/xi
(both O(1)) and Delta = 1 - R S, the Green's function gives:

* total rate (LDOS):  Re[u_in u_out / Delta] = Re[psi xi (1+rho)(1+sigma)/(1-rho sigma)]
* radiated amplitude: F = u_in / (B^o_d Delta), B^o_d the outgoing amplitude
  normalised to B_{N+1} = 1 in the host (OSAC Eq. 30)
* field in an absorbing shell a: the regular (a < d) or outgoing (a > d)
  solution continued to shell a, times u_out(x)/Delta resp. u_in(x)/Delta
  (OSAC Eqs. 31, 33, 34 in normalisation-free form).

Every result carries the total rate computed independently from the Green's
function, and ``balance_error = |total - rad - nrad| / total`` (energy
conservation).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .convergence import l_max as l_max_estimate
from .riccati import log_riccati
from .solver import Solution, solve

__all__ = ["DecayRates", "decay_rates", "locate_shell"]


def locate_shell(rad, r):
    """1-based shell index for radial position(s) r (interfaces belong outward)."""
    return np.searchsorted(np.asarray(rad), np.asarray(r), side="right") + 1


@dataclass
class DecayRates:
    """Normalised decay rates, arrays of shape ``(n_positions, 2)``.

    Column 0: radial (perpendicular) dipole, column 1: tangential (parallel).
    """

    r: np.ndarray
    radiative: np.ndarray
    nonradiative: np.ndarray
    total: np.ndarray
    shell: np.ndarray
    l_used: np.ndarray
    converged: np.ndarray
    norm: str
    dipole: str
    notes: list = field(default_factory=list)

    @property
    def balance_error(self) -> np.ndarray:
        """|total - (rad + nrad)| / total per position and orientation."""
        return np.abs(self.total - self.radiative - self.nonradiative) / np.abs(self.total)

    @staticmethod
    def averaged(a: np.ndarray) -> np.ndarray:
        """Orientation average (perp + 2 par) / 3."""
        return (a[:, 0] + 2 * a[:, 1]) / 3

    def quantum_yield(self, q0: float = 1.0) -> np.ndarray:
        """Orientation-averaged quantum yield (OSAC Eq. 38)."""
        gr, gnr = self.averaged(self.radiative), self.averaged(self.nonradiative)
        return gr / (gr + gnr + (1 - q0) / q0)


# ------------------------------------------------------------------ helpers


def _norm_factors(ref, mu, d, norm, dipole):
    """(N_rad, N_nrad, N_tot) of OSAC Eq. (28), corrected (AUDIT.md M1/M4).

    Shell-normalised factors follow from the Green's function; the host
    normalisation multiplies by Gamma_0,shell / Gamma_0,host, which is
    n mu (electric) or n eps (magnetic dipole)."""
    nd, nh, md, mh = ref[d], ref[-1], mu[d], mu[-1]
    n_rad = ((nd * md) / (nh * mh)).real
    n_nrad = (md / nd**2).real
    if norm == "shell":
        ratio = 1.0
    elif norm == "host":
        if dipole == "electric":
            ratio = ((nd * md) / (nh * mh)).real
        else:
            ratio = ((nd * nd**2 / md) / (nh * nh**2 / mh)).real
    else:
        raise ValueError("norm must be 'host' or 'shell'")
    return n_rad * ratio, n_nrad * ratio, ratio


def _log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a)


@lru_cache(maxsize=64)
def _gauss_legendre(n):
    return np.polynomial.legendre.leggauss(n)


def _log_abs_integrals(sol: Solution, a: int, la, lb, n_quad=None):
    """log of the absorption integrals of shell a (0-based) for coefficients
    exp(la), exp(lb) (arrays (L,)):

        I_TE = int |A j_l + B h_l|^2 r^2 dr
        I_TM = ( l(l+1) int |A j_l + B h_l|^2 dr + int |A psi_l' + B xi_l'|^2 dr ) / |k|^2

    (OSAC Eq. 32 up to the |k|^2 convention; Moroz Eq. 116.)  Gauss-Legendre
    quadrature with a per-order logarithmic shift, so any l works.
    """
    L = la.size  # may be fewer orders than the solution holds
    l = sol.l[:L]
    k = sol.k[0, a]
    r_in = sol.rad[a - 1] if a else 0.0
    r_out = sol.rad[a]
    n = n_quad or int(max(64, L + 48, 6 * abs(k) * (r_out - r_in)))
    t, wt = _gauss_legendre(n)
    r = 0.5 * (r_out - r_in) * t + 0.5 * (r_out + r_in)
    wt = 0.5 * (r_out - r_in) * wt
    x = k * r
    lp, lx = log_riccati(x, L)  # (n, L+1)
    tp = la[None, :] + lp[:, l]  # log(A psi_l)
    tx = lb[None, :] + lx[:, l]
    tpm = la[None, :] + lp[:, l - 1]  # log(A psi_{l-1})
    txm = lb[None, :] + lx[:, l - 1]
    with np.errstate(all="ignore"):
        shift = np.max(np.real(np.stack([tp, tx, tpm, txm])), axis=(0, 1))  # (L,)
        shift = np.where(np.isfinite(shift), shift, 0.0)
        e = lambda v: np.nan_to_num(np.exp(v - shift))  # noqa: E731
        u = e(tp) + e(tx)  # (A psi + B xi) / e^shift
        du = e(tpm) + e(txm) - l * u / x[:, None]  # psi' = psi_{l-1} - l psi / x
        ite = (wt @ np.abs(u) ** 2) / abs(k) ** 2
        itm = (l * (l + 1) * (wt @ (np.abs(u) ** 2 / np.abs(x[:, None]) ** 2)) + wt @ np.abs(du) ** 2) / abs(k) ** 2
        return np.log(ite) + 2 * shift, np.log(itm) + 2 * shift


def _tail_estimate(t):
    """Estimated remainder of a series from its last terms (axis 0 = l).

    The decay-rate series are asymptotically geometric in l (ratio (r_</r_>)^2
    times a polynomial), so the neglected tail is ~ t_L r/(1 - r) with r the
    observed ratio of the last terms - far larger than t_L when r -> 1.
    Returns +inf where the terms are not (yet) decreasing.
    """
    a = np.abs(t[-4:])
    with np.errstate(all="ignore"):
        r = np.max(a[1:] / a[:-1], axis=0)  # worst recent ratio
        rem = np.where(a[-1] == 0, 0.0, np.where(r < 1, a[-1] * r / (1 - r), np.inf))
    return rem


def _converged_sum(terms, tol):
    """Sum over l (axis 1 of (P, L, K)) with a remainder-based convergence test."""
    P, L, K = terms.shape
    val = np.zeros((P, K))
    used = np.zeros(P, dtype=int)
    ok = np.zeros(P, dtype=bool)
    for i in range(P):
        t = terms[i]
        finite = np.all(np.isfinite(t), axis=1)
        stop = L if finite.all() else int(np.argmin(finite))
        t = t[:stop]
        if stop < 4:
            val[i] = t.sum(0) if stop else np.nan
            continue
        tot = t.sum(0)
        scale = np.maximum(np.abs(tot), np.finfo(float).tiny)
        ok[i] = stop == L and bool(np.all(_tail_estimate(t) <= tol * scale))
        val[i], used[i] = tot, stop
    return val, used, ok


def _orders_needed(q, tol, l_cap):
    """Orders for a geometric series of ratio q^2 to leave a tail below tol."""
    if q >= 1:
        return l_cap
    return 1.25 * np.log(tol * (1 - q * q)) / (2 * np.log(q)) + 16


# --------------------------------------------------------------------- main


def _rates_one(sol: Solution, r_dip, shells, norm, dipole, tol, n_quad):
    """Rates for a single-wavelength Solution (shells 0-based)."""
    l = sol.l
    L = l.size
    N = sol.n_shells
    ref, mu = sol.ref[0], sol.mu[0]
    eps = ref**2 / mu
    k = sol.k[0]
    perp_pol, oth_pol = ("e", "m") if dipole == "electric" else ("m", "e")
    # which integral type goes with which role: TM-type for 'e' coefficients, TE-type for 'm'
    itype = {"e": 1, "m": 0}
    absorbing = [a for a in range(N) if eps[a].imag > 0]
    # The Ohmic-loss series converges like (r_< / r_>)^(2l) for the absorbing
    # shell's nearest boundary - usually far sooner than the LDOS series, which
    # also feels nearby dielectric interfaces.  Integrate only that many orders
    # (checked below; fall back to all orders if the tail is not small).
    bounds = [b for a in absorbing for b in ((sol.rad[a - 1],) if a else ()) + (sol.rad[a],)]
    if bounds:
        q = np.max(np.minimum(np.array(bounds)[None, :] / r_dip[:, None], r_dip[:, None] / np.array(bounds)[None, :]))
        l_abs = int(min(L, max(32, _orders_needed(q, tol, L))))
    else:
        l_abs = L
    cache = {}

    def integ(a, side, pol):
        key = (a, side, pol, l_abs)
        if key not in cache:
            if side == "below":  # regular solution in shell a (plane-wave normalisation)
                la, lb = sol.logA[pol][a, 0], sol.logB[pol][a, 0]
            else:  # outgoing solution in shell a (B_{N+1} = 1)
                lb = sol.logBo[pol][a, 0]
                la = lb + sol.logS[pol][a, 0]
            v = np.full(L, -np.inf, dtype=complex)
            v[:l_abs] = _log_abs_integrals(sol, a, la[:l_abs], lb[:l_abs], n_quad)[itype[pol]]
            cache[key] = v
        return cache[key]

    cffl = l * (l + 1) * (2 * l + 1)
    cfrd = 2 * l + 1
    P = r_dip.size
    terms = np.zeros((P, L, 6))
    for d in np.unique(shells):
        idx = np.flatnonzero(shells == d)
        x = (k[d] * r_dip[idx]).real  # emitter shell is lossless
        lp, lx = log_riccati(x.astype(complex), L)  # (p, L+1)
        n_rad, n_nrad, n_tot = _norm_factors(ref, mu, d, norm, dipole)
        U = {}
        with np.errstate(all="ignore"):
            for p in ("e", "m"):
                LP, LX = lp[:, l], lx[:, l]
                logR, logS = sol.logR[p][d, 0], sol.logS[p][d, 0]
                rho = np.exp(logR + LX - LP)  # R xi/psi at the emitter
                sig = np.exp(logS + LP - LX)  # S psi/xi
                l1pd = np.log(1 - np.exp(logR + logS))  # log Delta, Delta = 1 - R S
                d1 = np.exp(lp[:, l - 1] - LP) - l / x[:, None]
                d3 = np.exp(lx[:, l - 1] - LX) - l / x[:, None]
                U[p] = dict(
                    uin=LP + _log(1 + rho),  # log u_in, A_d = 1
                    duin=LP + _log(d1 + rho * d3),
                    uout=LX + _log(1 + sig),  # log u_out, B_d = 1
                    duout=LX + _log(sig * d1 + d3),
                    ldelta=l1pd,
                    lbo=sol.logBo[p][d, 0],
                    la=sol.logA[p][d, 0],
                    lR=logR,
                    lS=logS,
                    lp=LP,
                    lx=LX,
                    lpd=LP + _log(d1),  # log psi'
                    lxd=LX + _log(d3),  # log xi'
                )
            e, m = U[perp_pol], U[oth_pol]
            xx = x[:, None]

            def mag2(v):
                return np.exp(2 * np.real(v))

            # radiative: F = u_in / (B^o_d Delta)
            fE = e["uin"] - e["lbo"] - e["ldelta"]
            dfE = e["duin"] - e["lbo"] - e["ldelta"]
            fM = m["uin"] - m["lbo"] - m["ldelta"]
            terms[idx, :, 0] = 1.5 / xx**4 * n_rad * cffl * mag2(fE)
            terms[idx, :, 1] = 0.75 / xx**2 * n_rad * cfrd * (mag2(fM) + mag2(dfE))

            # total (LDOS), shell normalisation times the conversion factor.
            # u_in u_out / Delta = psi xi + [R xi^2 + S psi^2 + 2 R S psi xi] / (1 - R S)
            # and Re(psi xi) = psi^2 exactly for real x; written this way every
            # term keeps full relative accuracy (Re(psi xi) itself is ~1e-600
            # at high l while Im(psi xi) ~ 1/l).
            def ldos(U, deriv):
                p_, x_ = (U["lpd"], U["lxd"]) if deriv else (U["lp"], U["lx"])
                rs = U["lR"] + U["lS"]
                bracket = np.exp(U["lR"] + 2 * x_) + np.exp(U["lS"] + 2 * p_) + 2 * np.exp(rs + p_ + x_)
                return np.exp(2 * p_).real + (bracket * np.exp(-U["ldelta"])).real

            terms[idx, :, 4] = 1.5 / xx**4 * n_tot * cffl * ldos(e, False)
            terms[idx, :, 5] = 0.75 / xx**2 * n_tot * cfrd * (ldos(m, False) + ldos(e, True))

            # nonradiative
            def nonradiative():
                out = np.zeros((idx.size, L, 2))
                kd3 = k[d].real ** 3
                for a in absorbing:
                    pre = kd3 * n_nrad * eps[a].imag
                    if a < d:  # field ~ regular solution in a, amplitude u_out(x)/Delta / A_d
                        side = "below"
                        cE = e["uout"] - e["ldelta"] - e["la"]
                        dcE = e["duout"] - e["ldelta"] - e["la"]
                        cM = m["uout"] - m["ldelta"] - m["la"]
                    else:  # field ~ outgoing solution in a, amplitude u_in(x)/Delta / B^o_d
                        side = "above"
                        cE = e["uin"] - e["ldelta"] - e["lbo"]
                        dcE = e["duin"] - e["ldelta"] - e["lbo"]
                        cM = m["uin"] - m["ldelta"] - m["lbo"]
                    iE = integ(a, side, perp_pol)
                    iM = integ(a, side, oth_pol)
                    out[:, :, 0] += 1.5 / xx**4 * pre * cffl * np.exp(iE.real + 2 * cE.real)
                    out[:, :, 1] += (
                        0.75 / xx**2 * pre * cfrd * (np.exp(iM.real + 2 * cM.real) + np.exp(iE.real + 2 * dcE.real))
                    )
                return out

            nr = nonradiative()
            if l_abs < L:
                tot = np.abs(nr.sum(axis=1))  # (p, 2)
                rem = _tail_estimate(np.moveaxis(nr[:, :l_abs], 1, 0))
                if not np.all(rem <= 0.1 * tol * np.maximum(tot, np.finfo(float).tiny)):
                    l_abs = L
                    nr = nonradiative()
            terms[idx, :, 2:4] = nr
    return _converged_sum(terms, tol)


def decay_rates(
    rad,
    ref,
    mu,
    lam,
    r_dip,
    l_max=None,
    tol=1e-6,
    norm="host",
    dipole="electric",
    n_quad=None,
    l_cap=1200,
    warn=True,
) -> DecayRates:
    """Radiative, nonradiative and total decay rates of a dipole emitter.

    Parameters
    ----------
    rad, ref, mu, lam : geometry and materials at one wavelength (see :func:`solve`).
    r_dip : float or array   emitter distance(s) from the centre (not inside
        an absorbing shell, not in an absorbing host).
    l_max : int, optional    truncation; ``None`` doubles l_max from a size-
        based estimate until every sum meets ``tol`` (up to ``l_cap``).
    tol : float              relative accuracy of every l-sum (estimated
        remainder / sum, from the geometric decay of the last terms).
    norm : ``'host'`` or ``'shell'`` - free-space rate of the host or of the
        emitter's own shell.
    dipole : ``'electric'`` or ``'magnetic'``.
    """
    rad = np.atleast_1d(np.asarray(rad, dtype=float))
    ref = np.atleast_1d(np.asarray(ref, dtype=complex))
    mu = np.atleast_1d(np.asarray(mu, dtype=complex))
    r_dip = np.atleast_1d(np.asarray(r_dip, dtype=float))
    if dipole not in ("electric", "magnetic"):
        raise ValueError("dipole must be 'electric' or 'magnetic'")
    if norm not in ("host", "shell"):
        raise ValueError("norm must be 'host' or 'shell'")
    if np.ndim(lam) != 0:
        raise ValueError("decay_rates takes a single wavelength")
    eps = ref**2 / mu
    if eps[-1].imag > 0:
        raise ValueError("absorbing host: decay rates are not defined")
    if np.any(r_dip <= 0):
        raise ValueError("emitter positions must be > 0")
    shells = locate_shell(rad, r_dip) - 1
    if np.any(eps[shells].imag != 0) or np.any(ref[shells].imag != 0):
        raise ValueError("emitter inside an absorbing (or gain) shell: the rates diverge / are undefined")

    if l_max is not None:
        sol = solve(rad, ref, mu, lam, int(l_max))
        val, used, ok = _rates_one(sol, r_dip, shells, norm, dipole, tol, n_quad)
    else:
        # The slowest series decays like (r_< / r_>)^(2l) for the interface
        # nearest the emitter; start where that factor reaches tol (times a
        # polynomial margin), then double only if the sums are not yet converged.
        q = np.max(np.minimum(rad[None, :] / r_dip[:, None], r_dip[:, None] / rad[None, :]))
        L = int(min(l_cap, max(32, l_max_estimate(rad[-1], ref[-1], lam, "near"), _orders_needed(q, tol, l_cap))))
        while True:
            sol = solve(rad, ref, mu, lam, L)
            val, used, ok = _rates_one(sol, r_dip, shells, norm, dipole, tol, n_quad)
            if ok.all() or L >= l_cap:
                break
            L = min(2 * L, l_cap)
    notes = []
    if not ok.all():
        notes.append(
            f"l-sum not converged to tol={tol:g} at {np.count_nonzero(~ok)} of {ok.size} position(s) "
            f"with l_max={sol.l.size}; the emitter is very close to an interface"
        )
        if warn:
            warnings.warn(notes[-1], RuntimeWarning, stacklevel=2)
    return DecayRates(
        r=r_dip,
        radiative=val[:, 0:2],
        nonradiative=val[:, 2:4],
        total=val[:, 4:6],
        shell=shells + 1,
        l_used=used,
        converged=ok,
        norm=norm,
        dipole=dipole,
        notes=notes,
    )
