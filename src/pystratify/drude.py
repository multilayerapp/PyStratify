"""Drude fits of Au, Ag and Al and the electron free-path correction for thin metal shells.

Tabulated optical constants are deliberately not shipped: take n + ik from a
database (e.g. refractiveindex.info) and pass it to :func:`~pystratify.solve`.
The Drude fits are needed only for what tables cannot supply - the damping
of the free electrons, which enters the free-path correction and the energy
prefactor of a dispersive shell.  Wavelengths in this module are in
nanometres.

The energy form (:class:`DrudeEnergyModel`, :func:`fit_drude`, :func:`free_path_permittivity`) works
from photon energies in eV and fits the free electrons of a metal to its own tabulated permittivity
instead of taking them from the named fits above.  Lengths there are in nm, like everywhere in
this module (the multilayer.app API, which first wrote this code, used micrometres).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import constants

__all__ = [
    "DrudeModel",
    "DRUDE",
    "free_path_correction",
    "DrudeEnergyModel",
    "layer_effective_path",
    "free_electron_fermi_velocity",
    "fit_drude",
    "free_path_permittivity",
]

_HBAR_EV_S = constants.hbar / constants.e


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


# ------------------------------------------------------------------ energy-form Drude model
#
# The free electrons of a metal layer, fitted to the layer's own optical data rather than taken
# from a table of fits for named metals: whatever the source says gold is, its infrared tail is
# what fixes omega_p and gamma.  Everything here is in photon energy, E = hbar omega, in eV.
# Ported from the multilayer.app API (backend/spheres.py); tests/oracles/api_spheres_drude.py keeps
# that code frozen and tests/test_drude_energy.py compares against it.


@dataclass(frozen=True)
class DrudeEnergyModel:
    """eps_inf - omega_p^2 / (omega (omega + i gamma)), and what the free path needs.

    ``plasma_energy`` = hbar omega_p and ``damping_energy`` = hbar gamma in eV, ``fermi_velocity``
    in m/s, ``scattering_constant`` the dimensionless A of gamma_s = gamma + A v_F / L_eff.
    ``window`` (start, end in nm), ``points`` and ``rms`` (relative rms misfit |eps_D - eps| / |eps|)
    record the fit when the model came from :func:`fit_drude`.
    """

    eps_inf: float
    plasma_energy: float
    damping_energy: float
    fermi_velocity: float
    scattering_constant: float = 1.0
    window: Optional[tuple] = None
    points: int = 0
    rms: Optional[float] = None

    def surface_damping(self, path_nm: float) -> float:
        """hbar gamma_s = hbar (gamma + A v_F / L_eff) in eV, for an effective path in nm."""
        return self.damping_energy + _HBAR_EV_S * self.scattering_constant * self.fermi_velocity / (path_nm * 1e-9)

    def permittivity(self, energy, damping: Optional[float] = None):
        """eps at photon energies in eV; ``damping`` (eV) replaces hbar gamma, e.g. by gamma_s."""
        damping = self.damping_energy if damping is None else damping
        return self.eps_inf - self.plasma_energy**2 / (energy * (energy + 1j * damping))

    def summary(self, path_nm: Optional[float] = None) -> dict:
        """The parameters (and, with ``path_nm``, the effective path in nm and gamma_s in eV) as a dict."""
        body = {
            "eps_inf": self.eps_inf,
            "plasma_energy": self.plasma_energy,
            "damping_energy": self.damping_energy,
            "fermi_velocity": self.fermi_velocity,
            "fermi_velocity_free_electron": free_electron_fermi_velocity(self.plasma_energy),
            "scattering_constant": self.scattering_constant,
            "fit": None
            if self.window is None
            else {"start": self.window[0], "end": self.window[1], "points": self.points, "rms": self.rms},
        }
        if path_nm is not None:
            body["effective_path"] = path_nm
            body["surface_damping_energy"] = self.surface_damping(path_nm)
        return body


def layer_effective_path(radii, i: int) -> float:
    """L_eff = 4V/S of layer ``i`` of a radii list (core first): 4R/3 for the core and
    4/3 (R2^3 - R1^3) / (R1^2 + R2^2) for a shell (Moroz, J. Phys. Chem. C 112, 10641 (2008)),
    in the unit of ``radii`` (:func:`effective_path` is the same for one sphere or one shell)."""
    if i == 0:
        return 4 / 3 * float(radii[0])
    r1, r2 = float(radii[i - 1]), float(radii[i])
    return 4 / 3 * (r2**3 - r1**3) / (r1**2 + r2**2)


def free_electron_fermi_velocity(plasma_energy: float) -> float:
    """v_F in m/s of a free-electron gas with this plasma energy (eV): n = eps_0 m omega_p^2 / e^2 and
    v_F = hbar (3 pi^2 n)^(1/3) / m, with the bare electron mass.  1.39e6 m/s at 9 eV (gold)."""
    omega = plasma_energy / _HBAR_EV_S
    density = constants.epsilon_0 * constants.m_e * omega**2 / constants.e**2
    return constants.hbar * (3 * math.pi**2 * density) ** (1 / 3) / constants.m_e


def fit_drude(energy, eps) -> tuple:
    """(eps_inf, hbar omega_p, hbar gamma, rms) of eps_inf - omega_p^2 / (E (E + i gamma)) fitted
    to eps(E), each point weighted by 1/|eps| so the misfit is relative across the window.

    ``energy`` in eV.  For a fixed gamma the model is linear in eps_inf and omega_p^2: Re eps = eps_inf -
    omega_p^2 / (E^2 + gamma^2) and Im eps = omega_p^2 gamma / (E (E^2 + gamma^2)).  So gamma
    alone is searched -- a scan in log gamma, then golden sections -- and the rest solved,
    with eps_inf >= 1: bound electrons only add to the vacuum's polarisability, and a window
    that catches an interband band would otherwise pay for it with eps_inf < 0.  Raises ValueError
    when the fit has no free-electron response (omega_p^2 <= 0).
    """
    energy = np.asarray(energy, dtype=float)
    eps = np.asarray(eps, dtype=complex)
    weight = 1 / np.abs(eps)
    target = np.concatenate([weight * eps.real, weight * eps.imag])

    def solve(gamma: float):
        denominator = energy**2 + gamma**2
        design = np.concatenate(
            [
                np.stack([weight, -weight / denominator], axis=1),
                np.stack([np.zeros_like(weight), weight * gamma / (energy * denominator)], axis=1),
            ]
        )
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        if coefficients[0] < 1:
            # the quadratic's minimum on the boundary eps_inf = 1
            rest = target - design[:, 0]
            plasma_squared = float(design[:, 1] @ rest / (design[:, 1] @ design[:, 1]))
            coefficients = np.array([1.0, plasma_squared])
        residual = design @ coefficients - target
        return coefficients, float(residual @ residual)

    grid = np.geomspace(1e-4, 5.0, 90)
    costs = [solve(g)[1] for g in grid]
    best = int(np.argmin(costs))
    lo, hi = math.log(grid[max(best - 1, 0)]), math.log(grid[min(best + 1, grid.size - 1)])
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = solve(math.exp(a))[1], solve(math.exp(b))[1]
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = solve(math.exp(a))[1]
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = solve(math.exp(b))[1]
    gamma = math.exp((lo + hi) / 2)
    (eps_inf, plasma_squared), cost = solve(gamma)
    if not plasma_squared > 0:
        raise ValueError("no free-electron response")
    return float(eps_inf), math.sqrt(plasma_squared), gamma, math.sqrt(cost / energy.size)


def free_path_permittivity(eps, energy, model: DrudeEnergyModel, path_nm: float):
    """The layer's own permittivity with its free electrons' damping raised to gamma_s:
    eps + omega_p^2 / (omega (omega + i gamma)) - omega_p^2 / (omega (omega + i gamma_s)).
    The data keep everything else -- the interband transitions, and any misfit of the model.
    ``energy`` in eV, ``path_nm`` the effective path in nm."""
    gamma_s = model.surface_damping(path_nm)
    return eps + model.plasma_energy**2 * (
        1 / (energy * (energy + 1j * model.damping_energy)) - 1 / (energy * (energy + 1j * gamma_s))
    )
