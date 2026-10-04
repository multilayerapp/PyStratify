"""Emitters spread through a shell, and the spectral density of the field at an emitter.

* :func:`shell_average` - decay rates and quantum yield of emitters distributed uniformly
  (volume weight r^2 dr) through one shell and oriented at random, from
  :func:`~pystratify.decay_rates` at Gauss-Legendre nodes graded geometrically towards
  absorbing neighbours.  Next to an absorbing layer the nonradiative rate grows as d^-3 with
  the distance d, so its average diverges as d_min^-2: emitters closer than ``d_min`` are left out.
* :func:`spectral_density` - J(omega) = Gamma_tot(omega)/(2 pi) of a dipole of fixed moment at a
  fixed position, and the frequency shift, over a range of wavelengths, in units of the free rate
  in the host at a reference wavelength (the Wigner-Weisskopf kernel of strong coupling; the shift
  is the Hilbert partner of J).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decay import decay_rates
from .energy import gauss_legendre

__all__ = ["ShellAverage", "shell_average", "SpectralDensity", "spectral_density"]


@dataclass(frozen=True)
class ShellAverage:
    """Orientation- and volume-averaged rates over the free rate in the host (``total``,
    ``radiative``, ``nonradiative``, ``shift``) and the average quantum yield of statically,
    randomly oriented emitters of intrinsic yield ``intrinsic`` (``quantum_yield``); ``r`` and
    ``weights`` are the nodes and normalized volume weights used, ``span`` the radii averaged over."""

    total: float
    radiative: float
    nonradiative: float
    shift: float
    quantum_yield: float
    intrinsic: float
    span: tuple
    r: np.ndarray
    weights: np.ndarray
    converged: bool


def shell_average(
    radii, n, wavelength, shell, mu=None, dipole="electric", d_min=0.0, nodes=24, intrinsic=1.0, tol=1e-10,
    l_cap=20000,
) -> ShellAverage:  # fmt: skip
    """Rates of emitters spread uniformly through shell ``shell`` (0 = core) and oriented at random.

    Parameters as in :func:`~pystratify.decay_rates`; the shell must be lossless and not the host.
    ``d_min`` (length units of ``radii``) keeps emitters that far from both boundaries of the shell; it
    must be positive next to an absorbing layer.  ``nodes`` Gauss-Legendre nodes per half of the shell,
    graded geometrically towards an absorbing neighbour.  Rates are normalized to the free rate in the
    host and averaged with the volume weight r^2; the quantum yield is averaged over positions and the
    two orientations (1/3 radial, 2/3 tangential) of emitters that do not rotate during their lifetime.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    n = np.atleast_1d(np.asarray(n, dtype=complex))
    mu_arr = np.ones(n.size, complex) if mu is None else np.atleast_1d(np.asarray(mu, dtype=complex))
    if not 0 <= shell < radii.size:
        raise ValueError("shell must be the core or a shell (0 .. N - 1), not the host")
    if not 0 < intrinsic <= 1:
        raise ValueError("intrinsic quantum yield must be in (0, 1]")
    eps = n**2 / mu_arr
    lossy = (eps.imag != 0) | (mu_arr.imag != 0)
    if lossy[shell]:
        raise ValueError("emitters in an absorbing shell: the rates are undefined")
    a = (radii[shell - 1] if shell else 0.0) + (d_min if shell else 0.0)
    b = radii[shell] - d_min
    if not b > a:
        raise ValueError("d_min leaves no room in the shell")
    graded_a = bool(shell and lossy[shell - 1])
    graded_b = bool(lossy[shell + 1])
    if (graded_a or graded_b) and not d_min > 0:
        raise ValueError("next to an absorbing layer the average nonradiative rate diverges: give d_min > 0")
    r_in = radii[shell - 1] if shell else 0.0
    r, w = _nodes(a, b, r_in, radii[shell], graded_a, graded_b, nodes)
    rates = decay_rates(radii, n, wavelength, r, mu, tol=tol, normalization="host", dipole=dipole, l_cap=l_cap,
                        warn=False)  # fmt: skip
    vol = w * r**2
    vol = vol / vol.sum()
    avg = lambda x: float(np.sum(vol * (x[:, 0] + 2 * x[:, 1]) / 3))  # noqa: E731
    # the free rate in the emitter's own shell, in host units, sets the intrinsic nonradiative rate
    ratio = (n[shell] * mu_arr[shell] / (n[-1] * mu_arr[-1])).real
    if dipole == "magnetic":
        ratio = ((n[shell] ** 3 / mu_arr[shell]) / (n[-1] ** 3 / mu_arr[-1])).real
    intrinsic_nr = ratio * (1 - intrinsic) / intrinsic
    eta = rates.radiative / (rates.total + intrinsic_nr)
    return ShellAverage(
        total=avg(rates.total),
        radiative=avg(rates.radiative),
        nonradiative=avg(rates.nonradiative),
        shift=avg(rates.shift) if rates.shift is not None else float("nan"),
        quantum_yield=avg(eta),
        intrinsic=intrinsic,
        span=(float(a), float(b)),
        r=r,
        weights=vol,
        converged=bool(rates.converged.all()),
    )


def _nodes(a, b, r_in, r_out, graded_a, graded_b, count):
    """Gauss-Legendre nodes on [a, b] in two halves; a half next to an absorbing boundary uses
    u = ln(distance to that boundary), so the nodes crowd towards the boundary geometrically."""
    t, w = gauss_legendre(count)
    mid = 0.5 * (a + b)
    rs, ws = [], []
    for lo, hi, graded, boundary in ((a, mid, graded_a, r_in), (mid, b, graded_b, r_out)):
        if graded:
            u0, u1 = np.log(abs(lo - boundary)), np.log(abs(hi - boundary))
            u = 0.5 * (u1 - u0) * t + 0.5 * (u1 + u0)
            dist = np.exp(u)
            rs.append(boundary + np.sign(lo - boundary) * dist)
            ws.append(np.abs(0.5 * (u1 - u0) * w * dist))
        else:
            rs.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
            ws.append(0.5 * (hi - lo) * w)
    return np.concatenate(rs), np.concatenate(ws)


@dataclass(frozen=True)
class SpectralDensity:
    """``J`` = Gamma_tot(omega)/(2 pi) and ``shift`` (omega_0 - omega shift of an emitter whose transition
    frequency is omega), both over the free rate in the host at ``reference`` (wavelength), for a dipole of
    fixed moment; arrays over ``wavelength``, orientation ``orientation``.  ``omega`` is omega/omega_ref."""

    wavelength: np.ndarray
    omega: np.ndarray
    J: np.ndarray
    shift: np.ndarray
    reference: float
    orientation: str
    converged: np.ndarray


def spectral_density(
    radii, n, wavelengths, r, mu=None, dipole="electric", orientation="average", reference=None, tol=1e-10,
    l_cap=20000,
) -> SpectralDensity:  # fmt: skip
    """Spectral density J(omega) and frequency shift of a dipole at radius ``r`` over ``wavelengths``.

    ``n`` (and ``mu``) are (N + 1,) or one row per wavelength, (W, N + 1).  For a dipole of fixed
    moment the free rate scales as omega^3 times n_h mu_h (electric) or n_h^3/mu_h (magnetic, dual
    moment), so J(omega)/Gamma_0(omega_ref) = (omega/omega_ref)^3 f_h(omega)/f_h(omega_ref) Gamma_tot/Gamma_0
    (omega) / (2 pi), with the rate from :func:`~pystratify.decay_rates` (host normalization); likewise
    the shift.  ``orientation``: ``'perp'``, ``'par'`` or ``'average'`` ((perp + 2 par)/3).
    ``reference`` defaults to the middle wavelength.
    """
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    n_rows = np.asarray(n, dtype=complex)
    n_rows = np.broadcast_to(n_rows, (wl.size,) + n_rows.shape[-1:]) if n_rows.ndim == 1 else n_rows
    if mu is None:
        mu_rows = np.ones_like(n_rows)
    else:
        mu_rows = np.asarray(mu, dtype=complex)
        mu_rows = np.broadcast_to(mu_rows, n_rows.shape) if mu_rows.ndim == 1 else mu_rows
    if n_rows.shape[0] != wl.size or mu_rows.shape != n_rows.shape:
        raise ValueError("n and mu need one row per wavelength (or a single row)")
    if orientation not in ("perp", "par", "average"):
        raise ValueError("orientation must be 'perp', 'par' or 'average'")
    ref = float(np.median(wl)) if reference is None else float(reference)
    tot, sh, ok = np.empty(wl.size), np.empty(wl.size), np.empty(wl.size, dtype=bool)
    f_h = np.empty(wl.size)
    for i, lam in enumerate(wl):
        out = decay_rates(radii, n_rows[i], lam, [r], mu_rows[i], tol=tol, normalization="host", dipole=dipole,
                          l_cap=l_cap, warn=False)  # fmt: skip
        pick = {"perp": lambda x: x[0, 0], "par": lambda x: x[0, 1], "average": lambda x: (x[0, 0] + 2 * x[0, 1]) / 3}
        tot[i], sh[i], ok[i] = pick[orientation](out.total), pick[orientation](out.shift), out.converged.all()
        nh, mh = n_rows[i, -1], mu_rows[i, -1]
        f_h[i] = (nh * mh).real if dipole == "electric" else (nh**3 / mh).real
    omega = ref / wl
    # f_h at the reference: interpolate in wavelength (exact when the host is not dispersive)
    f_ref = float(np.interp(ref, wl[np.argsort(wl)], f_h[np.argsort(wl)]))
    scale = omega**3 * f_h / f_ref
    return SpectralDensity(
        wavelength=wl, omega=omega, J=scale * tot / (2 * np.pi), shift=scale * sh, reference=ref,
        orientation=orientation, converged=ok,
    )  # fmt: skip
