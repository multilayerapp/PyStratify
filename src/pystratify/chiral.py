"""Multilayered spheres with chiral (Pasteur) layers.

Constitutive relations (Gaussian units, time dependence exp(-i omega t)):

    D = eps E + i kappa H,     B = mu H - i kappa E,

the convention of treams (Beutel et al., Comput. Phys. Commun. 297, 109076
(2024)) and of Lindell et al., *Electromagnetic Waves in Chiral and
Bi-Isotropic Media* (1994) after exp(jwt) -> exp(-iwt).  In every layer the
field splits into the Beltrami fields Q_s = E + s i Z H (s = +-1,
Z = sqrt(mu/eps) = mu/n), which obey curl Q_s = s k_s Q_s with
k_s = k0 (n + s kappa): kappa > 0 slows helicity +1 (left-circular in the
optics convention).  The host must be achiral.

Per multipole order the fields of a layer are W_s = M + s N with wavenumber
k_s, and E_tan, H_tan continuous at an interface gives, with channel values
v_s = u_s/x_s and derivatives d_s = u_s'/x_s (u = A psi + B xi),

    v' = C_v v,   d' = C_d d,   C_v = [[1+z, 1-z], [1-z, 1+z]]/2,
    C_d = [[1+z, z-1], [z-1, 1+z]]/2,   z = Z_{j+1} / Z_j,

so helicity mixes only through the impedance contrast (a dual, impedance-
matched particle conserves helicity).  The regular solution is swept
outwards as the 2x2 matrix rho = Xi R Psi^-1 (B = R A, scaled at the
interface as in the achiral solver): with D1 = psi'/psi, D3 = xi'/xi,

    rho' = Lambda^-1 (N0 + N1 rho) (D0 + D1m rho)^-1 Lambda,
    N0 = D1' C_v - C_d D1,  N1 = D1' C_v - C_d D3,
    D0 = C_d D1 - D3' C_v,  D1m = C_d D3 - D3' C_v,  Lambda = D1' - D3',

the matrix form of the achiral Moebius step.  As there, the large parts
(l+1)/x of D1 and l/x of D3 are combined analytically, so N0 and D1m - which
vanish for identical media - are built from the material contrasts
directly, and rho is carried as element-wise logarithms so it may fall far
below the double range across thick absorbing shells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .convergence import truncation_order
from .riccati import log_riccati
from .solver import _batch, _side

__all__ = ["ChiralSolution", "solve_chiral", "HELICITY_TO_TMTE", "log_matmul"]

#: (TM, TE) = (N, M) amplitudes of W_+ = M + N and W_- = M - N, as columns
HELICITY_TO_TMTE = np.array([[1.0, -1.0], [1.0, 1.0]])
_TMTE_TO_HELICITY = np.linalg.inv(HELICITY_TO_TMTE)
_SIGN = np.array([1.0, -1.0])
_SAFE_LOG = 600.0


def _inv2(m):
    a, b, c, d = m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1]
    out = np.empty_like(m)
    det = a * d - b * c
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = d / det, -b / det, -c / det, a / det
    return out


def _log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a)


def _log_add(log_x, log_y):
    with np.errstate(all="ignore"):
        m = np.maximum(log_x.real, log_y.real)
        m = np.where(np.isfinite(m), m, 0.0)
        return m + np.log(np.exp(log_x - m) + np.exp(log_y - m))


def log_matmul(log_a, log_b):
    """log(A @ B) for stacks of 2x2 matrices given as element-wise complex logarithms."""
    return _log_add(
        log_a[..., :, 0, None] + log_b[..., None, 0, :],
        log_a[..., :, 1, None] + log_b[..., None, 1, :],
    )


@dataclass(frozen=True)
class ChiralSolution:
    """T-matrix of a multilayered sphere with chiral layers, for a batch of wavelengths.

    ``log_t_helicity`` has shape ``(W, L, 2, 2)``: log of the host T-matrix
    block in the helicity basis [out, in] (index 0 = +1, 1 = -1) for the
    waves W_s = M + s N; ``t_matrix`` is the same block in the (TM, TE)
    basis used by the far-field functions.

    Internal fields, shape ``(W, N + 1, L, 2, 2)`` [layer, ..., channel, incident
    helicity]: in layer j the regular solution excited by a unit incident wave
    W_h in the host is sum_c (alpha_ch psi_l(k_jc r) + beta_ch xi_l(k_jc r)) W_c,
    with ``log_alpha`` = log alpha (the identity in the host) and ``log_r`` =
    log R, beta = R alpha (zero in the core, T in the host).
    """

    radii: np.ndarray
    n: np.ndarray  # (W, N + 1)
    mu: np.ndarray  # (W, N + 1)
    kappa: np.ndarray  # (W, N + 1)
    wavelength: np.ndarray  # (W,)
    orders: np.ndarray  # (L,)
    log_t_helicity: np.ndarray  # (W, L, 2, 2)
    log_alpha: np.ndarray  # (W, N + 1, L, 2, 2)
    log_r: np.ndarray  # (W, N + 1, L, 2, 2)

    @property
    def n_shells(self) -> int:
        return self.radii.size

    @property
    def k(self) -> np.ndarray:
        """Mean wavenumbers k0 n, shape (W, N + 1) (the host's is exact)."""
        return 2 * np.pi * self.n / self.wavelength[:, None]

    @property
    def k_helicity(self) -> np.ndarray:
        """Wavenumbers k0 (n + s kappa) of helicity s = +1, -1: shape (W, N + 1, 2)."""
        return 2 * np.pi * (self.n[..., None] + _SIGN * self.kappa[..., None]) / self.wavelength[:, None, None]

    @property
    def t_helicity(self) -> np.ndarray:
        with np.errstate(under="ignore"):
            return np.exp(self.log_t_helicity)

    @property
    def t_matrix(self) -> np.ndarray:
        """T-matrix blocks (W, L, 2, 2) in the (TM, TE) basis [out, in]."""
        return HELICITY_TO_TMTE @ self.t_helicity @ _TMTE_TO_HELICITY


def _interface_matrices(l, x_in, x_out, diff, side_in, side_out, delta):
    """N0, N1, D0, D1m (W, L, 2, 2) of one interface.

    ``x_*``, ``diff`` = x_in - x_out (from the contrasts): (W, 2); ``side_*`` =
    (r, X) = (psi_{l+1}/psi_l, xi_{l-1}/xi_l), each (W, 2, L); ``delta`` = 1 - z.
    """
    (r_i, big_i), (r_o, big_o) = side_in, side_out
    xi, xo, diff = x_in[..., None], x_out[..., None], diff[..., None]  # (W, 2, 1)
    h = (1 - delta / 2)[:, None, None]
    half = (delta / 2)[:, None, None]
    shape = r_i.shape[:1] + (l.size, 2, 2)
    n0, n1, d0, d1 = (np.empty(shape, dtype=complex) for _ in range(4))

    def put(target, c, e, value):
        target[..., c, e] = value

    for c in (0, 1):
        e = 1 - c
        # diagonal: the large parts combine into contrasts
        put(n0, c, c, h[:, 0] * ((l + 1) * diff[:, c] / (xi[:, c] * xo[:, c]) + r_i[:, c] - r_o[:, c]))
        put(n1, c, c, h[:, 0] * ((l + 1) / xo[:, c] - r_o[:, c] - big_i[:, c] + l / xi[:, c]))
        put(d0, c, c, h[:, 0] * ((l + 1) / xi[:, c] - r_i[:, c] - big_o[:, c] + l / xo[:, c]))
        put(d1, c, c, h[:, 0] * (big_i[:, c] - big_o[:, c] + l * diff[:, c] / (xi[:, c] * xo[:, c])))
        # off-diagonal: proportional to the impedance contrast
        put(n0, c, e, half[:, 0] * ((l + 1) * (1 / xo[:, c] + 1 / xi[:, e]) - r_o[:, c] - r_i[:, e]))
        put(n1, c, e, half[:, 0] * ((l + 1) / xo[:, c] - r_o[:, c] + big_i[:, e] - l / xi[:, e]))
        put(d0, c, e, -half[:, 0] * ((l + 1) / xi[:, e] - r_i[:, e] + big_o[:, c] - l / xo[:, c]))
        put(d1, c, e, -half[:, 0] * (big_i[:, e] + big_o[:, c] - l / xi[:, e] - l / xo[:, c]))
    return n0, n1, d0, d1


def _log_moebius(n0, n1, d0, d1, log_rho):
    """log[(N0 + N1 rho)(D0 + D1m rho)^-1] element-wise, rho = exp(log_rho), for any magnitude of rho,
    and log[(D0 + D1m rho)^-1] (the inverse amplitude transfer up to Lambda).

    Where every element of rho is below e^-600, rho = e^s rho_hat and the result
    is N0 D0^-1 + e^s N1 rho_hat D0^-1: the neglected terms are e^-600 relative to
    one of these two (the N0 D0^-1 D1m rho D0^-1 cross term matters only if N0 is
    itself below e^-600).  Above e^600, rho is scaled down to e^600, which changes
    the ratio only by e^-600 relative.
    """
    with np.errstate(invalid="ignore"):
        s = np.max(log_rho.real, axis=(-2, -1))  # (W, L)
    tiny = np.isfinite(s) & (s <= -_SAFE_LOG)
    shift = np.where(np.isfinite(s) & (s > _SAFE_LOG), s - _SAFE_LOG, 0.0)[..., None, None]
    with np.errstate(all="ignore"):
        rho = np.exp(np.where(tiny[..., None, None], -np.inf, log_rho - shift))
        den_inv = _inv2(d0 + d1 @ rho)
        out = _log((n0 + n1 @ rho) @ den_inv)
        log_den_inv = _log(den_inv) - shift
    if tiny.any():
        rho_hat = np.exp(log_rho[tiny] - s[tiny][:, None, None])
        inv = _inv2(d0[tiny])
        out[tiny] = _log_add(_log(n0[tiny] @ inv), s[tiny][:, None, None] + _log(n1[tiny] @ rho_hat @ inv))
    return out, log_den_inv


def solve_chiral(radii, n, kappa, wavelength, mu=None, l_max=None) -> ChiralSolution:
    """Solve a multilayered sphere whose shells may be chiral.

    Parameters
    ----------
    radii : (N,) outer radii of the core and shells, strictly increasing.
    n : (N + 1,) or (W, N + 1) refractive indices n = sqrt(eps mu), host last.
    kappa : like ``n``, Pasteur chirality parameters (dimensionless, complex
        for a lossy chiral response); the host's must be 0, and n +- kappa must
        not be zero or negative real.
    wavelength : scalar or (W,) vacuum wavelength(s), same unit as ``radii``.
    mu : like ``n``, relative permeabilities (default 1).
    l_max : truncation order; default :func:`truncation_order` (Wiscombe) for
        the shortest wavelength, as :func:`~pystratify.solve`.

    With ``kappa = 0`` everywhere the result equals :func:`~pystratify.solve`'s
    (``t_matrix`` then is diagonal).
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.ndim != 1 or radii.size == 0 or not np.all(np.isfinite(radii)):
        raise ValueError("radii must be a non-empty 1-D array of finite values")
    if radii[0] <= 0 or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must be positive and strictly increasing")
    N = radii.size
    n, mu, wavelength = _batch(n, mu, wavelength, N + 1)
    try:
        kappa = np.ascontiguousarray(np.broadcast_to(np.asarray(kappa, dtype=complex), n.shape))
    except ValueError:
        raise ValueError(f"kappa needs shape ({N + 1},) or ({wavelength.size}, {N + 1}), host last") from None
    if not np.all(np.isfinite(kappa)):
        raise ValueError("chirality parameters must be finite")
    if np.any(kappa[:, -1] != 0):
        raise ValueError("the host must be achiral (kappa = 0 in the last entry)")
    helicity_n = n[..., None] + _SIGN * kappa[..., None]
    if np.any(helicity_n == 0) or np.any((helicity_n.imag == 0) & (helicity_n.real < 0)):
        raise ValueError(
            "n +- kappa must not be zero or negative real (a lossless backward-wave helicity is not supported)"
        )
    if l_max is None:
        l_max = max(truncation_order(radii[-1], abs(v), lam) for v, lam in zip(n[:, -1], wavelength))
    l_max = int(l_max)
    if l_max < 1:
        raise ValueError("l_max must be >= 1")
    l = np.arange(1, l_max + 1)
    W = wavelength.size

    k0 = 2 * np.pi / wavelength
    kc = k0[:, None, None] * helicity_n  # (W, N + 1, 2)
    x_in = kc[:, :N] * radii[:, None]  # (W, N, 2): k_{j,s} R_j
    x_out = kc[:, 1:] * radii[:, None]  # (W, N, 2): k_{j+1,s} R_j
    lp_in, lx_in, r_in, big_in, _, _ = _side(x_in, *log_riccati(x_in, l_max + 1), l)
    lp_out, lx_out, r_out, big_out, _, _ = _side(x_out, *log_riccati(x_out, l_max + 1), l)
    # 1 - Z_{j+1}/Z_j from the contrasts, Z = mu/n
    n_i, n_o, mu_i, mu_o = n[:, :N], n[:, 1:], mu[:, :N], mu[:, 1:]
    delta = (mu_i * (n_o - n_i) + n_i * (mu_i - mu_o)) / (mu_i * n_o)  # (W, N)
    # x - x' per helicity from the contrasts, not by subtraction of the products
    diff_n = (n_i - n_o)[..., None] + _SIGN * (kappa[:, :N] - kappa[:, 1:])[..., None]  # (W, N, 2)

    log_rho = np.full((W, l_max, 2, 2), -np.inf, dtype=complex)
    log_rho_out = np.empty((N, W, l_max, 2, 2), dtype=complex)
    log_transfer_inv = np.empty((N, W, l_max, 2, 2), dtype=complex)  # a = K^-1 a' at interface j
    for j in range(N):
        if j:  # carry rho across shell j: rho_ce *= [xi_c(o)/xi_c(i)] [psi_e(i)/psi_e(o)]
            grow_xi = np.moveaxis(lx_in[:, j] - lx_out[:, j - 1], 1, -1)  # (W, L, 2)
            fall_psi = np.moveaxis(lp_out[:, j - 1] - lp_in[:, j], 1, -1)
            log_rho = log_rho + grow_xi[..., :, None] + fall_psi[..., None, :]
        mats = _interface_matrices(
            l,
            x_in[:, j],
            x_out[:, j],
            k0[:, None] * radii[j] * diff_n[:, j],
            (r_in[:, j], big_in[:, j]),
            (r_out[:, j], big_out[:, j]),
            delta[:, j],
        )
        log_m, log_den_inv = _log_moebius(*mats, log_rho)
        # similarity with Lambda = diag(-i / (psi xi)) at the outer arguments
        lam = np.moveaxis(lp_out[:, j] + lx_out[:, j], 1, -1)  # (W, L, 2): log(psi xi)
        log_rho = log_m + lam[..., :, None] - lam[..., None, :]
        log_rho_out[j] = log_rho
        log_transfer_inv[j] = log_den_inv + (np.log(-1j) - lam)[..., None, :]  # (D0 + D1m rho)^-1 Lambda

    # amplitudes from the host inwards: scaled a = diag(psi/x) alpha, a_inner = K^-1 a_outer
    log_x_in, log_x_out = np.log(x_in), np.log(x_out)  # (W, N, 2)
    log_alpha = np.empty((W, N + 1, l_max, 2, 2), dtype=complex)
    log_r = np.full((W, N + 1, l_max, 2, 2), -np.inf, dtype=complex)
    log_alpha[:, N] = _log(np.eye(2) + 0j)
    for j in range(N - 1, -1, -1):
        outer = np.moveaxis(lp_out[:, j], 1, -1) - log_x_out[:, j, None, :]  # (W, L, 2): log(psi/x)
        inner = np.moveaxis(lp_in[:, j], 1, -1) - log_x_in[:, j, None, :]
        scaled = log_matmul(log_transfer_inv[j], log_alpha[:, j + 1] + outer[..., :, None])
        log_alpha[:, j] = scaled - inner[..., :, None]
        # R of layer j + 1 from rho at its inner boundary: R_ce = rho_ce (x_c / xi_c) (psi_e / x_e)
        to_r_rows = log_x_out[:, j, None, :] - np.moveaxis(lx_out[:, j], 1, -1)
        log_r[:, j + 1] = log_rho_out[j] + to_r_rows[..., :, None] + outer[..., None, :]
    return ChiralSolution(
        radii=radii,
        n=n,
        mu=mu,
        kappa=kappa,
        wavelength=wavelength,
        orders=l,
        log_t_helicity=log_r[:, N],
        log_alpha=log_alpha,
        log_r=log_r,
    )
