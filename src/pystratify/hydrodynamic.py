"""Hydrodynamic (nonlocal) response of the free electrons of a metal region.

The conduction electrons of a region obey the linearised hydrodynamic equation
(SI, exp(-i omega t))

    beta^2 grad(div J) + omega (omega + i gamma) J = i omega omega_p^2 eps_0 E,

on top of a local background eps_b (interband transitions, core electrons).  A
transverse wave sees the local permittivity

    eps_T = eps_b - omega_p^2 / (omega (omega + i gamma)),

and a longitudinal wave E_L = grad(Phi), (lap + k_L^2) Phi = 0, exists with

    k_L^2 = omega (omega + i gamma) eps_T / (eps_b beta^2),   Im k_L >= 0.

The refractive index of the region, as passed to every solver, is the
transverse response: n^2 = eps_T.  eps_b is not an independent input; it is
recovered as eps_b = n^2 + omega_p^2 / (omega (omega + i gamma)), so measured
optical constants are used as they are (Raza et al., J. Phys.: Condens. Matter
27, 183204 (2015)) and every observable reduces to the local result as
beta -> 0.  For a pure Drude metal pass n = :meth:`Hydrodynamic.transverse_index`.

``model`` selects beta^2:

* ``"high-frequency"``  beta^2 = 3/5 v_F^2 (omega >> gamma, the usual choice);
* ``"thomas-fermi"``    beta^2 = 1/3 v_F^2 (the static limit);
* ``"halevi"``          beta^2 = v_F^2 (3/5 omega + 1/3 i gamma) / (omega + i gamma)
  (Halevi, Phys. Rev. B 51, 7497 (1995)), which interpolates the two.

``diffusion`` adds the generalised nonlocal optical response (GNOR) of
Mortensen et al., Nat. Commun. 5, 3809 (2014): beta^2 -> beta^2 + D (gamma - i omega).

Parameters are in PyStratify's wavelength form, unit-agnostic like
:class:`~pystratify.DrudeModel`: ``plasma_wavelength`` = 2 pi c / omega_p and
``damping_wavelength`` = 2 pi c / gamma in the length unit of the problem,
``fermi_velocity`` = v_F / c, ``diffusion`` = D / c (a length).  Use
:meth:`Hydrodynamic.from_ev` for electronvolts and SI velocities.

Boundary conditions (``pystratify.nonlocal_sweep``): E_t and H_t are continuous
everywhere.  Where a hydrodynamic region meets a region without an electron
gas the normal current vanishes on its side, n.J = 0 (the hard wall).  Where two
hydrodynamic regions meet, ``contact="electrochemical"`` (the default) makes n.J
and the perturbation of the electrochemical potential, proportional to
(beta^2 / omega_p^2) div J, continuous (Forstmann & Stenschke); ``"boardman"``
makes n.v and the pressure perturbation, proportional to beta^2 div J,
continuous.  Both conserve energy; they differ when omega_p differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Hydrodynamic", "HC_EV_NM", "CONTACTS", "contact_weights"]

#: h c in eV nm: lambda[nm] = HC_EV_NM / E[eV]
HC_EV_NM = 1239.841984
_C = 299792458.0
_MODELS = ("high-frequency", "thomas-fermi", "halevi")


#: boundary conditions between two electron gases: continuity of n.J and of the electrochemical-
#: potential perturbation (beta^2/omega_p^2) div J (Forstmann & Stenschke, Phys. Rev. Lett. 38, 1365
#: (1977); the default), or of n.v and of the pressure beta^2 div J (Boardman 1982; Dong et al.)
CONTACTS = ("electrochemical", "boardman")


def contact_weights(regions, contact="electrochemical"):
    """Per-region weights of the metal/metal rows (``pystratify.nonlocal_sweep.interface_rows``):
    None for the electrochemical condition, lambda_p^2 (proportional to 1/omega_p^2) for Boardman's."""
    if contact not in CONTACTS:
        raise ValueError(f"contact must be one of {CONTACTS}")
    if contact == "electrochemical":
        return None
    return [m.plasma_wavelength**2 if m is not None else 1.0 for m in regions]


def _branch(value):
    """Square root with Im >= 0 (decaying, or outgoing when real)."""
    root = np.sqrt(np.asarray(value, dtype=complex))
    return np.where((root.imag < 0) | ((root.imag == 0) & (root.real < 0)), -root, root)


@dataclass(frozen=True)
class Hydrodynamic:
    """Free-electron gas of a region in the hydrodynamic Drude model (see the module docstring)."""

    plasma_wavelength: float
    damping_wavelength: float
    fermi_velocity: float
    model: str = "high-frequency"
    diffusion: float = 0.0

    def __post_init__(self):
        if not (np.isfinite(self.plasma_wavelength) and self.plasma_wavelength > 0):
            raise ValueError("plasma_wavelength must be positive and finite")
        if not self.damping_wavelength > 0:
            raise ValueError("damping_wavelength must be positive (np.inf for an undamped gas)")
        if not (np.isfinite(self.fermi_velocity) and 0 < self.fermi_velocity < 1):
            raise ValueError("fermi_velocity is v_F / c, in (0, 1)")
        if self.model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}")
        if not (np.isfinite(self.diffusion) and self.diffusion >= 0):
            raise ValueError("diffusion is D / c, a nonnegative length")

    @classmethod
    def from_ev(cls, plasma_energy, damping, fermi_velocity, *, length_unit=1e-9, model="high-frequency",
                diffusion=0.0) -> Hydrodynamic:
        """From hbar omega_p and hbar gamma in eV, v_F and D (diffusion constant) in SI units;
        ``length_unit`` is the problem's length unit in metres (default nanometres)."""
        scale = 1e-9 / length_unit
        return cls(HC_EV_NM / plasma_energy * scale, (HC_EV_NM / damping * scale) if damping > 0 else np.inf,
                   fermi_velocity / _C, model, diffusion / _C / length_unit)

    @classmethod
    def from_drude(cls, drude, *, model="high-frequency", diffusion=0.0) -> Hydrodynamic:
        """From a :class:`~pystratify.DrudeModel` (or a key of :data:`~pystratify.DRUDE`), whose
        wavelengths are in nanometres."""
        from .drude import DRUDE
        drude = DRUDE[drude] if isinstance(drude, str) else drude
        return cls(drude.plasma_wavelength, drude.damping_wavelength, drude.fermi_velocity, model, diffusion)

    # --------------------------------------------------------------------------- per wavelength
    def _ratios(self, wavelength):
        wavelength = np.asarray(wavelength, dtype=float)
        return self.plasma_wavelength / wavelength, self.plasma_wavelength / self.damping_wavelength  # w/wp, g/wp

    def free_susceptibility(self, wavelength):
        """omega_p^2 / (omega (omega + i gamma)) = eps_b - eps_T."""
        w, g = self._ratios(wavelength)
        return 1 / (w * (w + 1j * g))

    def beta_squared(self, wavelength):
        """(beta / c)^2, including the frequency dependence of the model and the GNOR diffusion."""
        w, g = self._ratios(wavelength)
        v2 = self.fermi_velocity**2
        if self.model == "high-frequency":
            b2 = 0.6 * v2 + 0j * w
        elif self.model == "thomas-fermi":
            b2 = v2 / 3 + 0j * w
        else:
            b2 = v2 * (0.6 * w + 1j * g / 3) / (w + 1j * g)
        if self.diffusion:
            k0 = 2 * np.pi / np.asarray(wavelength, dtype=float)
            b2 = b2 + self.diffusion * k0 * (g / w - 1j)  # D (gamma - i omega) / c^2
        return b2

    def background(self, wavelength, eps_T):
        """eps_b = eps_T + omega_p^2 / (omega (omega + i gamma))."""
        return np.asarray(eps_T, dtype=complex) + self.free_susceptibility(wavelength)

    def transverse_index(self, wavelength, eps_b=1.0):
        """sqrt(eps_T) of a Drude metal with background ``eps_b``: the n to pass for it."""
        return _branch(np.asarray(eps_b, dtype=complex) - self.free_susceptibility(wavelength))

    def longitudinal_wavenumber(self, wavelength, eps_T):
        """k_L with Im k_L >= 0, from k_L^2 = k0^2 (1 + i gamma/omega) eps_T / (eps_b (beta/c)^2)."""
        wavelength = np.asarray(wavelength, dtype=float)
        eps_T = np.asarray(eps_T, dtype=complex)
        w, g = self._ratios(wavelength)
        k0 = 2 * np.pi / wavelength
        return _branch(k0**2 * (1 + 1j * g / w) * eps_T / (self.background(wavelength, eps_T) * self.beta_squared(wavelength)))
