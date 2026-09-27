"""Overflow-free recursive transfer-matrix solution for a multilayered sphere.

Theory: Moroz, Ann. Phys. 315, 352 (2005); Rasskazov, Carney & Moroz, OSA
Continuum 3, 2290 (2020), Eqs. (10)-(18).  The 2x2 transfer matrices of that
method are products of Riccati-Bessel functions at different arguments and
overflow once the order exceeds the size parameter (Majic & Le Ru, Appl. Opt.
59, 1293 (2020)).  Here the same two-sided recursion is carried by scaled
quantities that stay O(1), after the ratio formulations of Yang, Appl. Opt.
42, 1710 (2003), Pena & Pal, Comput. Phys. Commun. 180, 2348 (2009) and
Ladutenko et al., Comput. Phys. Commun. 214, 225 (2017), with the hybrid
value/derivative matching of Zhang, JQSRT (2025), arXiv:2409.10877.

Shells are indexed 0..N (0 = core, N = host).  In shell s the radial function
of order l and polarisation p is  A_s psi_l(k_s r) + B_s xi_l(k_s r).

* regular solution (B_0 = 0), swept outwards with rho = R xi/psi, R = B/A;
* outgoing solution (A_N = 0), swept inwards with sigma = S psi/xi, S = A/B.

Across interface j (radius R_j; inner argument x = k_j R_j, outer x' =
k_{j+1} R_j) the field matching reads

    A psi(x) (1 + rho)      = c_f A' psi(x') (1 + rho')
    A psi(x) (D1 + rho D3)  = c_d A' psi(x') (D1' + rho' D3')

with (c_f, c_d) = (mu_j/mu_{j+1}, n_j/n_{j+1}) for TM (electric multipoles)
and the reverse for TE.  At high order D1 ~ (l+1)/x on both sides, so the
mismatch f D1 - D1' (f = c_f/c_d) is a small difference of large numbers.
It is therefore never formed by subtraction: with D1 = (l+1)/x - r and
D3 = X - l/x (r = psi_{l+1}/psi_l, X = xi_{l-1}/xi_l, both small past the
turning point) the large parts combine analytically into (l+1)(g-1)/x', and
g - 1 is computed directly from the material contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .convergence import truncation_order
from .riccati import log_riccati
from .sheets import _sheet_arrays, _sheet_terms

__all__ = ["Solution", "solve", "TM", "TE"]

#: polarisation axis of every coefficient array: TM = electric multipoles (a_l), TE = magnetic (b_l)
TM, TE = 0, 1


def _log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a)


def _log_add(log_x, log_y):
    """log(exp(log_x) + exp(log_y)) for complex logarithms, exact for -inf."""
    with np.errstate(all="ignore"):
        m = np.maximum(log_x.real, log_y.real)
        m = np.where(np.isfinite(m), m, 0.0)
        return m + np.log(np.exp(log_x - m) + np.exp(log_y - m))


#: |log t| below which t is a normal double, so the value arithmetic is exact
_SAFE_LOG = 600.0


def _log_transfer(num_small, num_big, den_small, den_big, log_t):
    """log[(num_small + t num_big) / (den_small + t den_big)] with t = exp(log_t).

    Evaluated with t as a value where it is a normal double, and as log-sums
    where it would be subnormal or overflow, so t keeps full relative precision
    however small or large it is - which matters when num_small -> 0
    (index-matched layers).
    """
    safe = np.abs(log_t.real) < _SAFE_LOG
    with np.errstate(all="ignore"):
        t = np.exp(np.where(safe, log_t, 0))
        out = _log((num_small + t * num_big) / (den_small + t * den_big))
    if not safe.all():
        u = ~safe
        out[u] = _log_add(_log(num_small[u]), log_t[u] + _log(num_big[u])) - _log_add(
            _log(den_small[u]), log_t[u] + _log(den_big[u])
        )
    return out


def _amplitude_log_ratio(c_value, c_deriv, value, deriv):
    """log(A_inner / A_outer) from the better-conditioned matching equation.

    ``value`` and ``deriv`` are pairs (outer factor, inner factor), each a
    tuple (value, scale) whose |value|/scale measures cancellation.
    """
    (v_out, v_out_scale), (v_in, v_in_scale) = value
    (d_out, d_out_scale), (d_in, d_in_scale) = deriv
    with np.errstate(all="ignore"):
        cond_v = np.minimum(np.abs(v_out) / v_out_scale, np.abs(v_in) / v_in_scale)
        cond_d = np.minimum(np.abs(d_out) / d_out_scale, np.abs(d_in) / d_in_scale)
        return np.where(cond_v >= cond_d, _log(c_value * v_out / v_in), _log(c_deriv * d_out / d_in))


@dataclass(frozen=True)
class Solution:
    """Scaled solution for a batch of wavelengths.

    Coefficient arrays have shape ``(2, N + 1, W, L)``: polarisation (:data:`TM`,
    :data:`TE`), shell (0 = core, N = host), wavelength, order l = 1..L.  All
    are complex logarithms; ``-inf`` stands for an exact zero.

    ``log_a``, ``log_b``: regular solution normalised to A = 1 in the host
    (the plane-wave expansion coefficients); ``log_r`` = log(B/A) of the
    regular solution; ``log_b_out``: outgoing solution normalised to B = 1 in
    the host; ``log_s`` = log(A/B) of the outgoing solution.
    """

    radii: np.ndarray  # (N,)
    n: np.ndarray  # (W, N + 1), host last
    mu: np.ndarray  # (W, N + 1)
    wavelength: np.ndarray  # (W,)
    orders: np.ndarray  # (L,)
    log_t: np.ndarray  # (2, W, L): T = B_host / A_host
    log_a: np.ndarray
    log_b: np.ndarray
    log_r: np.ndarray
    log_b_out: np.ndarray
    log_s: np.ndarray
    sheet_sigma: np.ndarray | None = None  # (W, N): 2D sheets at the interfaces (see pystratify.sheets)
    sheet_zeta: np.ndarray | None = None  # (W, N)

    @property
    def has_sheets(self) -> bool:
        return self.sheet_sigma is not None and bool(np.any(self.sheet_sigma) or np.any(self.sheet_zeta))

    @property
    def n_shells(self) -> int:
        """Number of interfaces N (the host is shell N)."""
        return self.radii.size

    @property
    def k(self) -> np.ndarray:
        """Wavenumbers, shape (W, N + 1)."""
        return 2 * np.pi * self.n / self.wavelength[:, None]

    @property
    def t(self) -> np.ndarray:
        """T-matrix elements T = B_host / A_host, shape (2, W, L)."""
        return np.exp(self.log_t)

    @property
    def a(self) -> np.ndarray:
        """Bohren-Huffman a_l = -T_TM, shape (W, L)."""
        return -np.exp(self.log_t[TM])

    @property
    def b(self) -> np.ndarray:
        """Bohren-Huffman b_l = -T_TE, shape (W, L)."""
        return -np.exp(self.log_t[TE])

    @property
    def t_matrix(self) -> np.ndarray:
        """T-matrix blocks, shape (W, L, 2, 2), in the (TM, TE) basis: [out, in].

        Diagonal for an achiral sphere; :class:`~pystratify.ChiralSolution` has
        the same property with the TM-TE coupling of chiral shells.
        """
        out = np.zeros(self.log_t.shape[1:] + (2, 2), dtype=complex)
        with np.errstate(under="ignore"):
            out[..., TM, TM] = np.exp(self.log_t[TM])
            out[..., TE, TE] = np.exp(self.log_t[TE])
        return out


def _batch(n, mu, wavelength, layers):
    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=float))
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("wavelengths must be positive and finite")
    n = np.asarray(n, dtype=complex)
    mu = np.ones(layers, dtype=complex) if mu is None else np.asarray(mu, dtype=complex)
    shape = (wavelength.size, layers)
    try:
        n = np.broadcast_to(n, shape)
        mu = np.broadcast_to(mu, shape)
    except ValueError:
        raise ValueError(f"n and mu need shape ({layers},) or ({wavelength.size}, {layers}), host last") from None
    if not (np.all(np.isfinite(n)) and np.all(np.isfinite(mu))):
        raise ValueError("refractive indices and permeabilities must be finite")
    if np.any(n == 0) or np.any(mu == 0):
        raise ValueError("refractive indices and permeabilities must be nonzero")
    return np.ascontiguousarray(n), np.ascontiguousarray(mu), wavelength


def _side(x, log_psi, log_xi, orders):
    """Per-order functions on one side of every interface: logs, r, X, D1, D3."""
    lp, lx = log_psi[..., orders], log_xi[..., orders]
    xx = x[..., None]
    with np.errstate(all="ignore"):
        r = np.exp(log_psi[..., orders + 1] - lp)  # psi_{l+1} / psi_l
        big_x = np.exp(log_xi[..., orders - 1] - lx)  # xi_{l-1} / xi_l
    d1 = (orders + 1) / xx - r
    d3 = big_x - orders / xx
    return lp, lx, r, big_x, d1, d3


def solve(radii, n, wavelength, mu=None, l_max=None, sheets=None) -> Solution:
    """Solve the multilayered sphere for multipole orders l = 1..l_max.

    Parameters
    ----------
    radii : (N,) outer radii of the core and shells, strictly increasing.
    n : (N + 1,) or (W, N + 1) complex refractive indices, host last.
    wavelength : scalar or (W,) vacuum wavelength(s), same unit as ``radii``.
    mu : like ``n``, relative permeabilities (default 1).
    l_max : truncation order; default :func:`truncation_order` (Wiscombe) for
        the shortest wavelength.
    sheets : 2D materials at interfaces, ``{j: Sheet(...)}`` or ``{j: conductivity}``
        with j the interface index (0 = surface of the core); see :mod:`pystratify.sheets`.

    Vectorised over wavelengths and orders; Python loops only over interfaces.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.ndim != 1 or radii.size == 0 or not np.all(np.isfinite(radii)):
        raise ValueError("radii must be a non-empty 1-D array of finite values")
    if radii[0] <= 0 or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must be positive and strictly increasing")
    N = radii.size
    n, mu, wavelength = _batch(n, mu, wavelength, N + 1)
    if l_max is None:
        l_max = max(truncation_order(radii[-1], abs(v), lam) for v, lam in zip(n[:, -1], wavelength))
    l_max = int(l_max)
    if l_max < 1:
        raise ValueError("l_max must be >= 1")
    orders = np.arange(1, l_max + 1)
    W = wavelength.size

    k = 2 * np.pi * n / wavelength[:, None]
    x_in = k[:, :N] * radii  # (W, N): k_j R_j
    x_out = k[:, 1:] * radii  # (W, N): k_{j+1} R_j
    lp_in, lx_in, r_in, X_in, d1_in, d3_in = _side(x_in, *log_riccati(x_in, l_max + 1), orders)
    lp_out, lx_out, r_out, X_out, d1_out, d3_out = _side(x_out, *log_riccati(x_out, l_max + 1), orders)

    n_in, n_out, mu_in, mu_out = n[:, :N], n[:, 1:], mu[:, :N], mu[:, 1:]
    eta = (n_in / n_out)[..., None]
    mu_ratio = (mu_in / mu_out)[..., None]
    sigma, zeta = _sheet_arrays(sheets, N, W)
    sheet_a, sheet_p, sheet_tau = _sheet_terms(sigma, zeta, 2 * np.pi / wavelength, radii, orders)
    z_in, z_out, sg = (mu_in / n_in)[..., None], (mu_out / n_out)[..., None], sigma[..., None]
    # g - 1 with g = f x'/x, from the contrast directly (no cancellation for similar media)
    g_minus_1 = {
        TM: ((mu_in * (n_out - n_in) * (n_out + n_in) + n_in**2 * (mu_in - mu_out)) / (mu_out * n_in**2))[..., None],
        TE: ((mu_out - mu_in) / mu_in)[..., None],
    }
    lead_psi = (orders + 1) / x_out[..., None]
    lead_xi = orders / x_out[..., None]

    shape = (2, N + 1, W, l_max)
    log_t = np.empty((2, W, l_max), dtype=complex)
    log_a, log_r, log_b_out, log_s = (np.empty(shape, dtype=complex) for _ in range(4))

    for p in (TM, TE):
        c_value, c_deriv = (mu_ratio, eta) if p == TM else (eta, mu_ratio)
        f = c_value / c_deriv
        m11 = lead_psi * g_minus_1[p] - (f * r_in - r_out)  # f D1 - D1'
        m33 = (f * X_in - X_out) - lead_xi * g_minus_1[p]  # f D3 - D3'
        f_d1_minus_d3 = f * d1_in - d3_out
        d1_minus_f_d3 = d1_out - f * d3_in
        # 2D sheets (pystratify.sheets): TE sees the jump of H_t, TM that of H_t and of E_t;
        # the matching becomes value' = (value + t_value deriv) / tau, deriv' = (deriv - t_deriv value) / tau
        if p == TE:
            s = 1j * z_out * sg
            m11, m33 = m11 - s, m33 - s
            d1_minus_f_d3, f_d1_minus_d3 = d1_minus_f_d3 + s, f_d1_minus_d3 - s
            t_value, t_deriv, tau = np.zeros_like(sg), 1j * z_in * sg, np.ones_like(sg)
        else:
            t = 1j * z_in * sg / sheet_p
            w = sheet_a / (1j * z_out * sheet_p)
            m11, m33 = m11 - w - t * d1_out * d1_in, m33 - w - t * d3_out * d3_in
            d1_minus_f_d3 = d1_minus_f_d3 + w + t * d1_out * d3_in
            f_d1_minus_d3 = f_d1_minus_d3 - w - t * d3_out * d1_in
            t_value, t_deriv, tau = t, sheet_a / (1j * z_in * sheet_p), sheet_tau

        # regular solution, outwards: rho' = [m11 + rho (f D3 - D1')] / [(D3' - f D1) - rho m33]
        log_rho_in = np.full((W, N, l_max), -np.inf, dtype=complex)
        log_rho_out = np.empty((W, N, l_max), dtype=complex)
        for j in range(N):
            if j:
                log_rho_in[:, j] = (
                    log_rho_out[:, j - 1] + lx_in[:, j] - lx_out[:, j - 1] + lp_out[:, j - 1] - lp_in[:, j]
                )
            log_rho_out[:, j] = _log_transfer(
                m11[:, j], -d1_minus_f_d3[:, j], -f_d1_minus_d3[:, j], -m33[:, j], log_rho_in[:, j]
            )

        # outgoing solution, inwards: sigma = [-m33 + sigma' (D1' - f D3)] / [(f D1 - D3') + sigma' m11]
        log_sig_out = np.full((W, N, l_max), -np.inf, dtype=complex)
        log_sig_in = np.empty((W, N, l_max), dtype=complex)
        for j in range(N - 1, -1, -1):
            if j < N - 1:
                log_sig_out[:, j] = (
                    log_sig_in[:, j + 1] + lp_out[:, j] - lp_in[:, j + 1] + lx_in[:, j + 1] - lx_out[:, j]
                )
            log_sig_in[:, j] = _log_transfer(
                -m33[:, j], d1_minus_f_d3[:, j], f_d1_minus_d3[:, j], m11[:, j], log_sig_out[:, j]
            )
        with np.errstate(under="ignore", over="ignore"):
            rho_in, rho_out = np.exp(log_rho_in), np.exp(log_rho_out)
            sig_in, sig_out = np.exp(log_sig_in), np.exp(log_sig_out)

        # amplitudes, as logarithms, from the host inwards
        la, lbo = log_a[p], log_b_out[p]
        la[N] = 0.0
        lbo[N] = 0.0
        for j in range(N - 1, -1, -1):
            a1, a3, b1, b3 = d1_in[:, j], d3_in[:, j], d1_out[:, j], d3_out[:, j]
            ri, ro, si, so = rho_in[:, j], rho_out[:, j], sig_in[:, j], sig_out[:, j]
            tv, td = t_value[:, j], t_deriv[:, j]
            cv, cd = c_value[:, j] / tau[:, j], c_deriv[:, j] / tau[:, j]
            with np.errstate(all="ignore"):
                val, der = 1 + ri, a1 + ri * a3
                val_scale, der_scale = 1 + np.abs(ri), np.abs(a1) + np.abs(ri * a3)
                ratio = _amplitude_log_ratio(
                    cv,
                    cd,
                    ((1 + ro, 1 + np.abs(ro)), (val + tv * der, val_scale + np.abs(tv) * der_scale)),
                    (
                        (b1 + ro * b3, np.abs(b1) + np.abs(ro * b3)),
                        (der - td * val, der_scale + np.abs(td) * val_scale),
                    ),
                )
                la[j] = la[j + 1] + ratio + lp_out[:, j] - lp_in[:, j]
                val, der = 1 + si, si * a1 + a3
                val_scale, der_scale = 1 + np.abs(si), np.abs(si * a1) + np.abs(a3)
                ratio = _amplitude_log_ratio(
                    cv,
                    cd,
                    ((1 + so, 1 + np.abs(so)), (val + tv * der, val_scale + np.abs(tv) * der_scale)),
                    (
                        (so * b1 + b3, np.abs(so * b1) + np.abs(b3)),
                        (der - td * val, der_scale + np.abs(td) * val_scale),
                    ),
                )
                lbo[j] = lbo[j + 1] + ratio + lx_out[:, j] - lx_in[:, j]

        log_t[p] = log_rho_out[:, -1] + lp_out[:, -1] - lx_out[:, -1]
        log_r[p, 0] = -np.inf
        log_r[p, 1:N] = np.moveaxis(log_rho_in[:, 1:] + lp_in[:, 1:] - lx_in[:, 1:], 1, 0)
        log_r[p, N] = log_t[p]
        log_s[p, :N] = np.moveaxis(log_sig_in + lx_in - lp_in, 1, 0)
        log_s[p, N] = -np.inf

    return Solution(
        radii=radii,
        n=n,
        mu=mu,
        wavelength=wavelength,
        orders=orders,
        log_t=log_t,
        log_a=log_a,
        log_b=log_a + log_r,
        log_r=log_r,
        log_b_out=log_b_out,
        log_s=log_s,
        sheet_sigma=sigma,
        sheet_zeta=zeta,
    )
