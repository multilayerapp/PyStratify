"""Overflow-free recursive transfer-matrix solution for a multilayered sphere.

STRATIFY multiplies the 2x2 transfer matrices of OSAC Eqs. (10)-(14)
directly.  Their entries are products of psi_l and xi_l at different
arguments and overflow once l exceeds the size parameter (AUDIT.md M9;
Majic & Le Ru, Appl. Opt. 59, 1293 (2020)).  Here the same two-sided
recursion is carried by scaled quantities that stay O(1), following the
ratio formulations of Yang, Appl. Opt. 42, 1710 (2003), Pena & Pal, Comput.
Phys. Commun. 180, 2348 (2009) and Ladutenko et al., Comput. Phys. Commun.
214, 225 (2017), and - for the singularities of the logarithmic
derivatives - the hybrid matching of Zhang, JQSRT (2025), arXiv:2409.10877.

In shell n (1 <= n <= N+1, host = N+1) the radial function of multipole l,
polarisation p is  A_n psi_l(k_n r) + B_n xi_l(k_n r).

*Regular* solution (B_1 = 0), swept outwards with  rho = R xi/psi,
R = B/A;  *outgoing* solution (A_{N+1} = 0), swept inwards with
sigma = S psi/xi, S = A/B.  Across interface n (inner argument x = k_n r_n,
outer x~ = k_{n+1} r_n) the field matching of OSAC Eqs. (10)-(13) reads

    A  psi(x) (1 + rho)        = c_f A' psi(x~) (1 + rho')
    A  psi(x) (D1 + rho D3)    = c_d A' psi(x~) (D1~ + rho' D3~)

with (c_f, c_d) = (eta~, mu~) for TE ("m", magnetic multipoles) and
(mu~, eta~) for TM ("e").  The ratio of the two gives rho' from rho; the
better-conditioned of the two gives the amplitude ratio.  Within a shell rho
and sigma are propagated with ratios xi(k r_a)/xi(k r_b) etc., which decay
for large l, so nothing overflows.  Amplitudes are kept as complex
logarithms.

The result reproduces STRATIFY's t_mat.m products wherever those are
representable (tests/test_solver.py) and keeps going for any l.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .riccati import log_derivatives, log_riccati

__all__ = ["Solution", "solve"]

POLS = ("e", "m")


def _log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a)


def _mix(d1, d3, r):
    """(d1 + r d3) / (1 + r), evaluated without overflow for any r."""
    with np.errstate(all="ignore"):
        big = np.abs(r) > 1
        rr = np.where(big, 1 / np.where(big, r, 1), 0)
        return np.where(big, (d1 * rr + d3) / (rr + 1), (d1 + r * d3) / (1 + r))


def _solve_ratio(y, d1, d3):
    """r such that (d1 + r d3)/(1 + r) = y, robust for |y| -> inf."""
    with np.errstate(all="ignore"):
        big = np.abs(y) > 1
        yi = np.where(big, 1 / np.where(big, y, 1), 0)
        return np.where(big, (1 - d1 * yi) / (d3 * yi - 1), (y - d1) / (d3 - y))


def _amp_ratio(c_f, c_d, v_num, v_den, d_num, d_den):
    """log of the amplitude ratio from the better-conditioned matching equation.

    v_* are the 'value' factors (1 + rho), d_* the 'derivative' factors
    (d1 + rho d3), each with its own scale for the conditioning test.
    """
    (vn, vn_s), (vd, vd_s) = v_num, v_den
    (dn, dn_s), (dd, dd_s) = d_num, d_den
    with np.errstate(all="ignore"):
        cond_v = np.minimum(np.abs(vn) / vn_s, np.abs(vd) / vd_s)
        cond_d = np.minimum(np.abs(dn) / dn_s, np.abs(dd) / dd_s)
        use_v = cond_v >= cond_d
        return np.where(use_v, _log(c_f * vn / vd), _log(c_d * dn / dd))


@dataclass
class Solution:
    """Scaled RTMM solution for a batch of wavelengths.

    Arrays have leading shape ``(W, L)`` (wavelength, multipole l = 1..L)
    after the shell index where present.  Shell indices are 0-based here:
    ``0`` = core, ``N`` = host.  Each per-polarisation field is a dict keyed
    by ``'e'`` (TM, a_l) and ``'m'`` (TE, b_l).
    """

    rad: np.ndarray  # (N,)
    ref: np.ndarray  # (W, N+1)
    mu: np.ndarray  # (W, N+1)
    lam: np.ndarray  # (W,)
    l: np.ndarray  # (L,)
    tpl: dict  # T_pl = B_{N+1}/A_{N+1}, OSAC Eq. 18, (W, L)
    logA: dict  # plane-wave regular solution, A_{N+1} = 1: (N+1, W, L)
    logB: dict
    logR: dict  # log(B/A) of the regular solution per shell
    logBo: dict  # outgoing solution with B_{N+1} = 1: log B per shell
    logS: dict  # log(A/B) of the outgoing solution per shell

    @property
    def n_shells(self) -> int:
        return self.rad.size

    @property
    def k(self) -> np.ndarray:
        return 2 * np.pi * self.ref / self.lam[:, None]

    @property
    def a(self) -> np.ndarray:
        """Bohren-Huffman a_l = -T_El, shape (W, L)."""
        return -self.tpl["e"]

    @property
    def b(self) -> np.ndarray:
        return -self.tpl["m"]


def _as_batch(ref, mu, lam, nlayer):
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    ref = np.asarray(ref, dtype=complex)
    mu = np.asarray(mu, dtype=complex)
    if ref.ndim == 1:
        ref = np.broadcast_to(ref, (lam.size, ref.size))
    if mu.ndim == 1:
        mu = np.broadcast_to(mu, (lam.size, mu.size))
    if ref.shape != (lam.size, nlayer) or mu.shape != (lam.size, nlayer):
        raise ValueError(f"ref and mu need shape ({nlayer},) or ({lam.size}, {nlayer}) (last = host)")
    return np.ascontiguousarray(ref), np.ascontiguousarray(mu), lam


def solve(rad, ref, mu, lam, l_max: int) -> Solution:
    """Solve the multilayered sphere for multipoles l = 1..l_max.

    ``rad`` (N,) outer radii; ``ref``/``mu`` (N+1,) or (W, N+1) complex
    refractive index / permeability, host last; ``lam`` scalar or (W,)
    vacuum wavelength(s), same length unit as ``rad``.  Vectorised over the
    wavelength batch and over l.
    """
    rad = np.atleast_1d(np.asarray(rad, dtype=float))
    if rad.ndim != 1 or rad.size == 0 or rad[0] <= 0 or np.any(np.diff(rad) <= 0):
        raise ValueError("radii must be positive and strictly increasing")
    N = rad.size
    ref, mu, lam = _as_batch(ref, mu, lam, N + 1)
    if np.any(ref == 0) or np.any(mu == 0):
        raise ValueError("refractive indices and permeabilities must be nonzero")
    l_max = int(l_max)
    if l_max < 1:
        raise ValueError("l_max must be >= 1")
    l = np.arange(1, l_max + 1)
    W = lam.size
    k = 2 * np.pi * ref / lam[:, None]  # (W, N+1)
    x_in = k[:, :N] * rad  # (W, N): x_n = k_n r_n
    x_out = k[:, 1:] * rad  # (W, N): x~_n = k_{n+1} r_n
    lp_in, lx_in = log_riccati(x_in, l_max)  # (W, N, l_max+1)
    lp_out, lx_out = log_riccati(x_out, l_max)
    d1_in, d3_in = log_derivatives(lp_in, lx_in, x_in, l)
    d1_out, d3_out = log_derivatives(lp_out, lx_out, x_out, l)
    LPi, LXi, LPo, LXo = (a[..., l] for a in (lp_in, lx_in, lp_out, lx_out))  # (W, N, L)
    eta = (ref[:, :N] / ref[:, 1:])[..., None]  # (W, N, 1)
    mur = (mu[:, :N] / mu[:, 1:])[..., None]

    tpl, logA, logB, logR, logBo, logS = {}, {}, {}, {}, {}, {}
    for p in POLS:
        c_f, c_d = (eta, mur) if p == "m" else (mur, eta)

        # ---- regular solution, outward
        rho_in = np.zeros((W, N, l_max), dtype=complex)
        rho_out = np.zeros((W, N, l_max), dtype=complex)
        for j in range(N):
            if j:
                with np.errstate(all="ignore"):
                    rho_in[:, j] = np.exp(
                        _log(rho_out[:, j - 1]) + LXi[:, j] - LXo[:, j - 1] + LPo[:, j - 1] - LPi[:, j]
                    )
            y = _mix(d1_in[:, j], d3_in[:, j], rho_in[:, j]) * (c_f[:, j] / c_d[:, j])
            rho_out[:, j] = _solve_ratio(y, d1_out[:, j], d3_out[:, j])
        with np.errstate(all="ignore"):
            log_tpl = _log(rho_out[:, -1]) + LPo[:, -1] - LXo[:, -1]  # kept as a log: T_pl underflows at high l
            tpl[p] = np.exp(log_tpl)

        # ---- outgoing solution, inward
        sig_in = np.zeros((W, N, l_max), dtype=complex)
        sig_out = np.zeros((W, N, l_max), dtype=complex)
        for j in range(N - 1, -1, -1):
            if j < N - 1:
                with np.errstate(all="ignore"):
                    sig_out[:, j] = np.exp(
                        _log(sig_in[:, j + 1]) + LPo[:, j] - LPi[:, j + 1] + LXi[:, j + 1] - LXo[:, j]
                    )
            # (sigma d1 + d3)/(sigma + 1) == _mix(d3, d1, 1/sigma); write it directly
            with np.errstate(all="ignore"):
                s = sig_out[:, j]
                small = np.abs(s) <= 1
                zz = np.where(
                    small, (s * d1_out[:, j] + d3_out[:, j]) / (s + 1), (d1_out[:, j] + d3_out[:, j] / s) / (1 + 1 / s)
                )
            zin = zz * (c_d[:, j] / c_f[:, j])
            # sigma_in solves (sigma d1 + d3)/(sigma + 1) = zin
            with np.errstate(all="ignore"):
                big = np.abs(zin) > 1
                zi = np.where(big, 1 / np.where(big, zin, 1), 0)
                sig_in[:, j] = np.where(
                    big, (1 - d3_in[:, j] * zi) / (d1_in[:, j] * zi - 1), (zin - d3_in[:, j]) / (d1_in[:, j] - zin)
                )

        # ---- amplitudes (logs)
        la = np.empty((N + 1, W, l_max), dtype=complex)
        la[N] = 0.0
        lbo = np.empty((N + 1, W, l_max), dtype=complex)
        lbo[N] = 0.0
        for j in range(N - 1, -1, -1):
            a1, a3 = d1_in[:, j], d3_in[:, j]
            b1, b3 = d1_out[:, j], d3_out[:, j]
            ri, ro = rho_in[:, j], rho_out[:, j]
            si, so = sig_in[:, j], sig_out[:, j]
            with np.errstate(all="ignore"):
                # regular: A_j psi(x)(..) = c A_{j+1} psi(x~)(..)
                lr = _amp_ratio(
                    c_f[:, j],
                    c_d[:, j],
                    (1 + ro, 1 + np.abs(ro)),
                    (1 + ri, 1 + np.abs(ri)),
                    (b1 + ro * b3, np.abs(b1) + np.abs(ro * b3)),
                    (a1 + ri * a3, np.abs(a1) + np.abs(ri * a3)),
                )
                la[j] = la[j + 1] + lr + LPo[:, j] - LPi[:, j]
                # outgoing: B_j xi(x)(sigma + 1) = c B_{j+1} xi(x~)(sigma' + 1)
                lo = _amp_ratio(
                    c_f[:, j],
                    c_d[:, j],
                    (1 + so, 1 + np.abs(so)),
                    (1 + si, 1 + np.abs(si)),
                    (so * b1 + b3, np.abs(so * b1) + np.abs(b3)),
                    (si * a1 + a3, np.abs(si * a1) + np.abs(a3)),
                )
                lbo[j] = lbo[j + 1] + lo + LXo[:, j] - LXi[:, j]

        lr_ = np.empty_like(la)
        ls_ = np.empty_like(la)
        with np.errstate(all="ignore"):
            lr_[0] = -np.inf
            for n in range(1, N):
                lr_[n] = _log(rho_in[:, n]) + LPi[:, n] - LXi[:, n]
            lr_[N] = log_tpl
            for n in range(N):
                ls_[n] = _log(sig_in[:, n]) + LXi[:, n] - LPi[:, n]
            ls_[N] = -np.inf
        logA[p], logR[p], logBo[p], logS[p] = la, lr_, lbo, ls_
        logB[p] = la + lr_
    return Solution(
        rad=rad, ref=ref, mu=mu, lam=lam, l=l, tpl=tpl, logA=logA, logB=logB, logR=logR, logBo=logBo, logS=logS
    )
