"""Interface responses: two-dimensional materials - graphene, TMD monolayers, thin films - and the
Feibelman d-parameters of metal surfaces (:class:`Feibelman`) at the interfaces of a sphere.

A sheet at interface j (radius R_j) enters as generalised sheet transition
conditions with the fields averaged over its two sides (Idemen, IEEE TAP 38,
1017 (1990); Kuester, Mohamed, Piket-May & Holloway, IEEE TAP 51, 2641
(2003)).  Gaussian units, exp(-i omega t):

    r x (H+ - H-) = sigma <E_t>,          E_t+ - E_t- = -zeta grad_t <D_n>,

with ``conductivity`` sigma = 4 pi sigma_s / c the dimensionless in-plane
surface conductivity (= sigma_s Z0 in SI; a graphene sheet at the universal
conductivity e^2/4 hbar has sigma = pi alpha) and ``normal`` zeta a length
describing the out-of-plane (normal) polarisation.  The in-plane response is
isotropic; neither term mixes TE and TM or helicities by itself.

A film of thickness d << R, lambda and permittivity eps (in-plane) and
eps_normal (out-of-plane), replacing medium eps_b, is to first order in d the
sheet ``Sheet.from_film``:

    sigma = -i k0 d (eps - eps_b),        zeta = d (1/eps_b - 1/eps_normal),

and absorbs, per area, (c/8 pi)[Re sigma |E_t|^2 + k0 Im zeta |D_n|^2].
In the quasi-static limit a sheet on a sphere of eps_1 in eps_2 supports
TM resonances at eps_1 l + eps_2 (l + 1) + i sigma l (l + 1) / (k0 R) = 0 -
the plasmons of graphene-coated spheres (Christensen et al., PRB 91, 125414
(2015)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Feibelman", "Sheet", "graphene_conductivity"]

#: fine-structure constant and hbar c in eV nm
_ALPHA = 7.2973525693e-3
_HC_EV_NM = 1239.841984
_KB_EV = 8.617333262e-5


@dataclass(frozen=True)
class Sheet:
    """A 2D material at an interface (see the module docstring for the conditions).

    ``conductivity``: dimensionless sigma = sigma_s Z0 (SI) = 4 pi sigma_s / c
    (Gaussian); ``normal``: zeta, a length in the unit of the radii.  Scalars or
    arrays over the wavelengths of the solve.
    """

    conductivity: complex | np.ndarray = 0.0
    normal: complex | np.ndarray = 0.0

    @classmethod
    def from_film(cls, eps, thickness, wavelength, eps_background, eps_normal=None) -> Sheet:
        """First-order sheet of a thin film: in-plane ``eps``, out-of-plane ``eps_normal``
        (default ``eps``: an isotropic film), displacing ``eps_background`` - for a film on a
        particle, the medium outside it.  ``thickness`` and ``wavelength`` in the unit of the radii."""
        eps = np.asarray(eps, dtype=complex)
        eps_normal = eps if eps_normal is None else np.asarray(eps_normal, dtype=complex)
        eps_background = np.asarray(eps_background, dtype=complex)
        k0 = 2 * np.pi / np.asarray(wavelength, dtype=float)
        return cls(-1j * k0 * thickness * (eps - eps_background), thickness * (1 / eps_background - 1 / eps_normal))


@dataclass(frozen=True)
class Feibelman:
    """Feibelman d-parameters of a metal surface at an interface (mesoscopic boundary conditions).

    ``d_perp`` is the centroid of the induced charge and ``d_par`` that of the normal derivative of
    the tangential current, complex lengths in the unit of the radii (scalars or arrays over the
    wavelengths), positive towards the other medium (spill-out); ``metal`` is ``'inner'`` or
    ``'outer'``, the side of the interface they belong to.  With n from the metal into the other
    medium and [f] = f(other) - f(metal), Gaussian units and exp(-i omega t) (Yang et al., Nature 576,
    248 (2019); K = i omega d_par [P_par] from the definition of d_par):

        [E_t] = -d_perp grad_t [E_n],        n x [H] = i k0 d_par [D_t].

    The jumps on the right are evaluated from the metal-side fields (D_n and E_t continuous at
    zeroth order), which makes the matching linear in the d-parameters as seen from the metal: for a
    sphere this is exactly the mesoscopic Mie theory of Goncalves et al., Nat. Commun. 11, 366 (2020),
    Eqs. (4), and with ``d_par = 0`` it is exact.  TE sees only d_par, as a sheet of conductivity
    sigma = i k0 d_par (eps_other - eps_metal).
    """

    d_perp: complex | np.ndarray = 0.0
    d_par: complex | np.ndarray = 0.0
    metal: str = "inner"

    def __post_init__(self):
        if self.metal not in ("inner", "outer"):
            raise ValueError("metal must be 'inner' or 'outer'")


def graphene_conductivity(wavelength, fermi_energy, damping, temperature=300.0):
    """Dimensionless sheet conductivity sigma_s Z0 of graphene (local random-phase approximation).

    Intraband (Drude) term at temperature T and the interband term for
    k_B T << E_F (Falkovsky, J. Phys.: Conf. Ser. 129, 012004 (2008); Hanson,
    J. Appl. Phys. 103, 064302 (2008)).  ``wavelength`` in nm; ``fermi_energy``
    E_F and ``damping`` hbar gamma in eV (hbar gamma = 6.6 meV for a 0.1 ps
    relaxation time); ``temperature`` in K.  At high frequency the result tends
    to pi alpha = e^2 Z0 / 4 hbar.
    """
    hw = _HC_EV_NM / np.asarray(wavelength, dtype=float)
    ef = abs(float(fermi_energy))
    kt = max(_KB_EV * float(temperature), 1e-9)
    log_cosh = ef / (2 * kt) + np.log1p(np.exp(-ef / kt))  # ln 2cosh(E_F / 2kT), overflow-free
    intra = 8j * _ALPHA * kt * log_cosh / (hw + 1j * damping)
    inter = (
        np.pi
        * _ALPHA
        * (
            0.5
            + np.arctan((hw - 2 * ef) / (2 * kt)) / np.pi
            - 0.5j / np.pi * np.log((hw + 2 * ef) ** 2 / ((hw - 2 * ef) ** 2 + (2 * kt) ** 2))
        )
    )
    return intra + inter


def _sheet_arrays(sheets, n_interfaces, n_wavelengths):
    """(sigma, zeta), each (W, N) complex, from a mapping {interface: Sheet or conductivity}."""
    sigma = np.zeros((n_wavelengths, n_interfaces), dtype=complex)
    zeta = np.zeros((n_wavelengths, n_interfaces), dtype=complex)
    if sheets is None:
        return sigma, zeta
    if not hasattr(sheets, "items"):
        raise ValueError("sheets must map interface indices (0 = core surface) to Sheet objects or conductivities")
    for j, sheet in sheets.items():
        if not (isinstance(j, (int, np.integer)) and 0 <= j < n_interfaces):
            raise ValueError(f"sheet interface index {j!r} not in 0..{n_interfaces - 1}")
        if isinstance(sheet, Feibelman):
            continue
        if not isinstance(sheet, Sheet):
            sheet = Sheet(conductivity=sheet)
        for target, value in ((sigma, sheet.conductivity), (zeta, sheet.normal)):
            try:
                target[:, j] = np.broadcast_to(np.asarray(value, dtype=complex), (n_wavelengths,))
            except ValueError:
                raise ValueError(
                    f"sheet parameters must be scalars or have one value per wavelength ({n_wavelengths})"
                ) from None
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(zeta))):
        raise ValueError("sheet parameters must be finite")
    return sigma, zeta


def _sheet_terms(sigma, zeta, k0, radii, orders):
    """Per-interface GSTC coefficients, (W, N, L): a = i zeta l(l+1)/(k0 R^2) (normal response in
    units of the tangential fields) and tau = (1 - a sigma/4)/(1 + a sigma/4)."""
    ll = orders * (orders + 1.0)
    a = 1j * zeta[..., None] * ll / (k0[:, None, None] * radii[None, :, None] ** 2)
    quarter = a * sigma[..., None] / 4
    return a, 1 - quarter, (1 - quarter) / (1 + quarter)


def _feibelman_arrays(sheets, n_interfaces, n_wavelengths):
    """(d_perp, d_par) (W, N) complex and the metal-outer flags (N,) of the Feibelman interfaces, or None."""
    found = {j: v for j, v in (sheets or {}).items() if isinstance(v, Feibelman)}
    if not found:
        return None
    d_perp = np.zeros((n_wavelengths, n_interfaces), dtype=complex)
    d_par = np.zeros((n_wavelengths, n_interfaces), dtype=complex)
    outer = np.zeros(n_interfaces, dtype=bool)
    for j, f in found.items():
        for target, value in ((d_perp, f.d_perp), (d_par, f.d_par)):
            try:
                target[:, j] = np.broadcast_to(np.asarray(value, dtype=complex), (n_wavelengths,))
            except ValueError:
                raise ValueError(
                    f"d-parameters must be scalars or have one value per wavelength ({n_wavelengths})"
                ) from None
        outer[j] = f.metal == "outer"
    if not (np.all(np.isfinite(d_perp)) and np.all(np.isfinite(d_par))):
        raise ValueError("d-parameters must be finite")
    return d_perp, d_par, outer


def _interface_terms(sheets, k0, radii, n, mu, orders):
    """Matching terms of the interface responses (sheets and d-parameters), per polarization:
    {TM: (tv, td, tau, tau - 1), TE: (...)}, each (W, N, L), with

        value' = tau (value + tv deriv) / c_v,     deriv' = tau (deriv - td value) / c_d

    across every interface (zero, zero, one, zero where there is none), value and deriv the
    Riccati function and its derivative on the inner (unprimed) and outer side.  ``n``, ``mu``:
    (W, N + 1); ``k0``: (W,).  tau - 1 is formed directly (the flux jump at the interface needs it)."""
    from .solver import TE, TM

    W, N = n.shape[0], radii.size
    L = orders.size
    ll = orders * (orders + 1.0)
    out = {p: [np.zeros((W, N, L), complex), np.zeros((W, N, L), complex), np.ones((W, N, L), complex),
               np.zeros((W, N, L), complex)] for p in (TM, TE)}  # fmt: skip
    n_in, n_out, mu_in, mu_out = n[:, :N], n[:, 1:], mu[:, :N], mu[:, 1:]
    z_in, z_out = (mu_in / n_in)[..., None], (mu_out / n_out)[..., None]
    e_in, e_out = (n_in**2 / mu_in)[..., None], (n_out**2 / mu_out)[..., None]
    kk = np.asarray(k0, dtype=float).reshape(-1)[:, None, None]
    R = radii[None, :, None]
    sigma, zeta = _sheet_arrays(sheets, N, W)
    if np.any(sigma) or np.any(zeta):
        sg = sigma[..., None]
        a = 1j * zeta[..., None] * ll / (kk * R**2)
        quarter = a * sg / 4
        p_ = 1 - quarter
        on = (sigma != 0) | (zeta != 0)
        tm, te = out[TM], out[TE]
        tm[0] = np.where(on[..., None], 1j * z_in * sg / p_, tm[0])
        tm[1] = np.where(on[..., None], a / (1j * z_in * p_), tm[1])
        tm[2] = np.where(on[..., None], p_ / (1 + quarter), tm[2])
        tm[3] = np.where(on[..., None], -2 * quarter / (1 + quarter), tm[3])
        te[1] = np.where(on[..., None], 1j * z_in * sg, te[1])
    dp = _feibelman_arrays(sheets, N, W)
    if dp is not None:
        d_perp, d_par, outer = dp
        dpe, dpa = d_perp[..., None], d_par[..., None]
        x_in, x_out = kk * n_in[..., None] * R, kk * n_out[..., None] * R
        on = (d_perp != 0) | (d_par != 0)
        for p in (TM, TE):
            if p == TM:  # metal-side forms: tv (value from the derivative) and td (derivative from the value)
                inner = (z_in * kk * dpa * (e_in - e_out), -dpe * ll * (e_out - e_in) / (e_out * R * x_in))
                outer_ = (z_out * kk * dpa * (e_in - e_out), -dpe * ll * (e_out - e_in) / (e_in * R * x_out))
                f = (mu_in / mu_out)[..., None] / (n_in / n_out)[..., None]  # c_v / c_d
            else:
                inner = (np.zeros_like(dpa), z_in * kk * dpa * (e_in - e_out))
                outer_ = (np.zeros_like(dpa), z_out * kk * dpa * (e_in - e_out))
                f = (n_in / n_out)[..., None] / (mu_in / mu_out)[..., None]
            # metal outside: the forms map the outer side to the inner one; inverted, tau = 1/(1 + tv' td')
            q = outer_[0] * outer_[1]
            tv = np.where(outer[None, :, None], -f * outer_[0], inner[0])
            td = np.where(outer[None, :, None], -outer_[1] / f, inner[1])
            tau = np.where(outer[None, :, None], 1 / (1 + q), 1.0)
            tau_m1 = np.where(outer[None, :, None], -q / (1 + q), 0.0)
            for i, v in enumerate((tv, td, tau, tau_m1)):
                out[p][i] = np.where(on[..., None], v, out[p][i])
    return {p: tuple(v) for p, v in out.items()}
