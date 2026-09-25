"""Near fields for plane-wave illumination.

The incident wave is E0 = x_hat exp(i k_h z), |E0| = 1, propagating along +z;
magnetic fields are in Gaussian units, H = -i (n/mu) sum(...), so |H0| = n_h/mu_h.
The m = +-1 vector spherical harmonics are combined analytically into the
Bohren-Huffman pi_l, tau_l, which are regular on both poles.  Expansion
coefficients are applied in logarithmic form,
A_l psi_l(kr) = exp(log A_l + log psi_l(kr)), so thin metal shells and high
orders are accurate; special functions are evaluated once per distinct k r.

In the host only the scattered field is summed; the incident plane wave is
added in closed form.  Its multipole series would need l ~ k_h r terms, far
more than the sphere's truncation away from the particle (with Wiscombe's
l_max, |E|^2 two radii from a 50-nm gold sphere was off by a factor of two),
whereas the scattered series converges with the sphere's orders at any r > R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .farfield import angular_functions
from .riccati import log_riccati
from .solver import TE, TM, Solution

__all__ = ["NearField", "near_field", "radial_functions"]


@dataclass(frozen=True)
class NearField:
    """Complex field components on the requested points (shape of the inputs).

    ``e`` and ``h`` map ``'x', 'y', 'z'`` (Cartesian) and ``'r', 'theta',
    'phi'`` (spherical) to arrays.
    """

    e: dict
    h: dict

    @property
    def intensity_e(self) -> np.ndarray:
        return sum(np.abs(self.e[c]) ** 2 for c in "xyz")

    @property
    def intensity_h(self) -> np.ndarray:
        return sum(np.abs(self.h[c]) ** 2 for c in "xyz")


def _log_incident(orders):
    return np.log(1j**orders * np.sqrt((2 * orders + 1) * np.pi))


def radial_functions(
    sol: Solution,
    wavelength_index: int,
    r,
    polarisations=(TM, TE),
    offsets=(0,),
    with_incident=True,
    host_scattered_only=False,
):
    """f_{l+o}(r) = A_l j_{l+o}(k r) + B_l h_{l+o}(k r) for each polarisation and offset ``o``.

    Returns ``({polarisation: [(len(r), L) array per offset]}, x = k r, shell
    index per point)``; the special functions are evaluated once for all
    polarisations.  Points exactly on an interface belong to the outer shell;
    r must be > 0.  ``with_incident`` includes the plane-wave factor
    i^l sqrt((2l + 1) pi); ``host_scattered_only`` drops the A_l j_l (incident)
    part in the host.
    """
    r = np.asarray(r, dtype=float)
    l = sol.orders
    shell = np.searchsorted(sol.radii, r, side="right")
    x = sol.k[wavelength_index][shell] * r
    log_psi, log_xi = log_riccati(x, l.size + max(max(offsets), 1))
    incident = _log_incident(l) if with_incident else 0.0
    out = {}
    with np.errstate(under="ignore", over="ignore"):
        for p in polarisations:
            la = sol.log_a[p, shell, wavelength_index] + incident
            if host_scattered_only:
                la = np.where((shell == sol.n_shells)[:, None], -np.inf, la)
            lb = sol.log_b[p, shell, wavelength_index] + incident
            out[p] = [(np.exp(la + log_psi[:, l + o]) + np.exp(lb + log_xi[:, l + o])) / x[:, None] for o in offsets]
    return out, x, shell


def near_field(sol: Solution, x, y, z, wavelength_index: int = 0, chunk: int = 4096) -> NearField:
    """Electric and magnetic near field at Cartesian points (x, y, z).

    Points at the origin are evaluated at r = 1e-9 R_0 (the field is
    continuous there).  In the host the incident wave is exact and the
    scattered series converges with the Solution's orders at any distance;
    fields right at the surface converge more slowly, see
    ``solve(..., regime='near')``.
    """
    x, y, z = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y, z)))
    shape = x.shape
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    r = np.maximum(np.sqrt(x**2 + y**2 + z**2), 1e-9 * sol.radii[0])
    theta = np.arctan2(np.hypot(x, y), z)
    phi = np.arctan2(y, x)
    l = sol.orders
    w = wavelength_index

    radii, r_index = np.unique(r, return_inverse=True)
    functions, kr, shell = radial_functions(sol, w, radii, offsets=(-1, 0), host_scattered_only=True)
    radial = {}
    for p, (f_prev, f) in functions.items():
        g = f / kr[:, None]  # f_l / x
        radial[p] = (f, g, f_prev - l * g)  # (1/x) d(x f_l)/dx = f_{l-1} - l f_l / x
    shell = shell[r_index]

    cosines, c_index = np.unique(np.round(np.cos(theta), 14), return_inverse=True)
    pi_u, tau_u = angular_functions(l, np.arccos(np.clip(cosines, -1, 1)))
    pi_u, tau_u = pi_u.T, tau_u.T  # (n_cos, L)
    m = 2j * np.sqrt((2 * l + 1) / (4 * np.pi)) / (l * (l + 1))
    m_radial = m * l * (l + 1)

    fe, ge, de = radial[TM]
    fm, gm, dm = radial[TE]
    e = {c: np.empty(r.size, complex) for c in ("r", "theta", "phi")}
    h = {c: np.empty(r.size, complex) for c in ("r", "theta", "phi")}

    def dot(a, b):
        return np.einsum("ij,ij->i", a, b)

    for start in range(0, r.size, chunk):
        s = slice(start, start + chunk)
        ri, ci = r_index[s], c_index[s]
        p_, t_ = pi_u[ci], tau_u[ci]
        cp, sp, st = np.cos(phi[s]), np.sin(phi[s]), np.sin(theta[s])
        a_de, a_fm, a_ge = de[ri] * m, fm[ri] * m, ge[ri] * m_radial
        a_fe, a_dm, a_gm = fe[ri] * m, dm[ri] * m, gm[ri] * m_radial
        e["phi"][s] = sp * (dot(a_de, p_) + 1j * dot(a_fm, t_))
        e["theta"][s] = -cp * (dot(a_de, t_) + 1j * dot(a_fm, p_))
        e["r"][s] = -st * cp * dot(a_ge, p_)
        h["phi"][s] = cp * (dot(a_fe, t_) - 1j * dot(a_dm, p_))
        h["theta"][s] = sp * (dot(a_fe, p_) - 1j * dot(a_dm, t_))
        h["r"][s] = -1j * st * sp * dot(a_gm, p_)
    factor = -1j * (sol.n[w] / sol.mu[w])[shell]
    for c in h:
        h[c] = h[c] * factor

    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    host = shell == sol.n_shells
    if host.any():  # E_inc = x_hat exp(i k z), H_inc = (n/mu) y_hat exp(i k z)
        wave = np.exp(1j * sol.k[w, -1] * z[host])
        h0 = sol.n[w, -1] / sol.mu[w, -1] * wave
        e["r"][host] += st[host] * cp[host] * wave
        e["theta"][host] += ct[host] * cp[host] * wave
        e["phi"][host] -= sp[host] * wave
        h["r"][host] += st[host] * sp[host] * h0
        h["theta"][host] += ct[host] * sp[host] * h0
        h["phi"][host] += cp[host] * h0

    def cartesian(f):
        return {
            "x": (-sp * f["phi"] + ct * cp * f["theta"] + st * cp * f["r"]).reshape(shape),
            "y": (cp * f["phi"] + ct * sp * f["theta"] + st * sp * f["r"]).reshape(shape),
            "z": (-st * f["theta"] + ct * f["r"]).reshape(shape),
            "r": f["r"].reshape(shape),
            "theta": f["theta"].reshape(shape),
            "phi": f["phi"].reshape(shape),
        }

    return NearField(e=cartesian(e), h=cartesian(h))
