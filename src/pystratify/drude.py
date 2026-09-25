"""Drude fits of Au, Ag and Al and the electron free-path correction for thin metal shells.

Tabulated optical constants are deliberately not shipped: take n + ik from a
database (e.g. refractiveindex.info) and pass it to :func:`~pystratify.solve`.
The Drude fits are needed only for what tables cannot supply - the damping
of the free electrons, which enters the free-path correction and the energy
prefactor of a dispersive shell.  Wavelengths in this module are in
nanometres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DrudeModel", "DRUDE", "free_path_correction"]


@dataclass(frozen=True)
class DrudeModel:
    """Free-electron parameters, in wavelength form (lambda = 2 pi c / omega).

    ``plasma_wavelength`` = 2 pi c / omega_p and ``damping_wavelength`` =
    2 pi c / gamma, both in nm; ``fermi_velocity`` is v_F / c.
    """

    plasma_wavelength: float
    damping_wavelength: float
    fermi_velocity: float

    @property
    def damping_ratio(self) -> float:
        """gamma / omega_p."""
        return self.plasma_wavelength / self.damping_wavelength


#: Ordal et al., Appl. Opt. 24, 4493 (1985) ("_Ord") and Blaber et al.,
#: J. Phys. Chem. C 113, 3041 (2009) ("_Blb"), as used by STRATIFY.
DRUDE = {
    "Au_Ord": DrudeModel(137.36, 46436.02734, 0.014 / 3),
    "Ag_Ord": DrudeModel(137.56, 68880.10723, 0.0139 / 3),
    "Al_Ord": DrudeModel(84.05708, 15156.99181, 0.0206 / 3),
    "Au_Blb": DrudeModel(144.93, 67382.71359, 0.014 / 3),
    "Ag_Blb": DrudeModel(129.15, 54379.03202, 0.0139 / 3),
    "Al_Blb": DrudeModel(81.03542, 2071.92836, 0.0206 / 3),
}


def effective_path(radii_nm) -> float:
    """Mean free path limited by the boundaries: 4/3 R for a sphere, and
    4/3 (R2^3 - R1^3) / (R1^2 + R2^2) for a shell (Moroz, J. Phys. Chem. C
    112, 10641 (2008))."""
    radii = np.atleast_1d(np.asarray(radii_nm, dtype=float))
    if radii.size == 1:
        return 4 / 3 * float(radii[0])
    r1, r2 = float(radii[0]), float(radii[1])
    return 4 / 3 * (r2**3 - r1**3) / (r1**2 + r2**2)


def surface_damping_wavelength(model: DrudeModel, radii_nm) -> float:
    """2 pi c / gamma_s with gamma_s = gamma + v_F / L_eff."""
    gamma_s = model.damping_ratio + model.fermi_velocity * model.plasma_wavelength / (
        2 * np.pi * effective_path(radii_nm)
    )
    return model.plasma_wavelength / gamma_s


def free_path_correction(wavelength_nm, n_bulk, radii_nm, model):
    """Refractive index of a thin metal shell with surface-limited damping.

    eps = eps_bulk + wp^2/(w(w + i gamma)) - wp^2/(w(w + i gamma_s)): the bulk
    Drude damping is swapped for gamma_s = gamma + v_F / L_eff.  ``radii_nm``
    is ``(R,)`` for a core or ``(R_in, R_out)`` for a shell; ``model`` a
    :class:`DrudeModel` or a key of :data:`DRUDE`.
    """
    model = DRUDE[model] if isinstance(model, str) else model
    ratio = model.damping_ratio
    ratio_s = model.plasma_wavelength / surface_damping_wavelength(model, radii_nm)
    w = model.plasma_wavelength / np.asarray(wavelength_nm, dtype=float)  # omega / omega_p
    eps = np.asarray(n_bulk) ** 2 + 1 / (w * (w + 1j * ratio)) - 1 / (w * (w + 1j * ratio_s))
    return np.sqrt(eps)
