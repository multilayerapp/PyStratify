"""Two-dimensional materials - graphene, TMD monolayers, thin films - at the interfaces of a sphere.

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

__all__ = ["Sheet", "graphene_conductivity"]

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
