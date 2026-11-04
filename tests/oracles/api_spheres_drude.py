"""FROZEN REFERENCE -- do not edit, do not import from production code.

The multilayer.app API's energy-form Drude model, copied verbatim (code only, no edits) from
repository ``multilayer``, commit ``ef9c946``, ``backend/spheres.py``:
  * lines 71-76   the exact-or-truncated constants (hbar in eV s is the TRUNCATED 6.582119569e-16),
  * lines 380-491 the Drude section header, ``Drude``, ``effective_path``,
                  ``free_electron_fermi_velocity`` and ``fit_drude``,
  * lines 598-605 ``free_path_permittivity``.
Omitted: ``_data_coverage``/``_permittivity``/``_default_fit_window``/``resolve_drude`` (lines 493-596),
which need the API's material resolver.  This file is the independent check kept after the API
calls ``pystratify.drude`` (assurance WP07-D2); tests/test_drude_energy.py compares against it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

HC_EV_UM = 6.62607015e-34 * 299_792_458 / 1.602176634e-19 * 1e6
HBAR_EV_S = 6.582119569e-16  # hbar in eV s
HBAR_J_S = 1.054571817e-34
ELECTRON_MASS = 9.1093837015e-31  # kg
ELECTRON_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m

# ------------------------------------------------------------------ Drude model
#
# The free electrons of a metal layer, fitted to the layer's own optical data rather
# than taken from a table of fits for named metals: whatever the source says gold is,
# its infrared tail is what fixes omega_p and gamma. Everything here is in photon
# energy, E = hbar omega = h c / lambda, in eV.


@dataclass(frozen=True)
class Drude:
    """eps_inf - omega_p^2 / (omega (omega + i gamma)), and what the free path needs."""

    eps_inf: float
    plasma_energy: float  # hbar omega_p, eV
    damping_energy: float  # hbar gamma, eV
    fermi_velocity: float  # m/s
    scattering_constant: float  # A
    window: Optional[tuple] = None  # (start, end) in um, when fitted
    points: int = 0
    rms: Optional[float] = None  # relative rms misfit |eps_D - eps| / |eps| over the window

    def surface_damping(self, path_um: float) -> float:
        """hbar gamma_s = hbar (gamma + A v_F / L_eff), in eV."""
        return self.damping_energy + HBAR_EV_S * self.scattering_constant * self.fermi_velocity / (path_um * 1e-6)

    def permittivity(self, energy, damping: Optional[float] = None):
        damping = self.damping_energy if damping is None else damping
        return self.eps_inf - self.plasma_energy**2 / (energy * (energy + 1j * damping))

    def summary(self, path_um: Optional[float] = None) -> dict:
        body = {
            "eps_inf": self.eps_inf,
            "plasma_energy": self.plasma_energy,
            "damping_energy": self.damping_energy,
            "fermi_velocity": self.fermi_velocity,
            "fermi_velocity_free_electron": free_electron_fermi_velocity(self.plasma_energy),
            "scattering_constant": self.scattering_constant,
            "fit": None if self.window is None else {"start": self.window[0], "end": self.window[1],
                                                     "points": self.points, "rms": self.rms},
        }
        if path_um is not None:
            body["effective_path"] = path_um
            body["surface_damping_energy"] = self.surface_damping(path_um)
        return body


def effective_path(radii, i: int) -> float:
    """L_eff = 4V/S of layer i: 4R/3 for the core and 4/3 (R2^3 - R1^3) / (R1^2 + R2^2) for a
    shell (Moroz, J. Phys. Chem. C 112, 10641 (2008)), in the unit of ``radii``."""
    if i == 0:
        return 4 / 3 * float(radii[0])
    r1, r2 = float(radii[i - 1]), float(radii[i])
    return 4 / 3 * (r2**3 - r1**3) / (r1**2 + r2**2)


def free_electron_fermi_velocity(plasma_energy: float) -> float:
    """v_F of a free-electron gas with this plasma frequency: n = eps_0 m omega_p^2 / e^2 and
    v_F = hbar (3 pi^2 n)^(1/3) / m, with the bare electron mass. 1.39e6 m/s at 9 eV (gold)."""
    omega = plasma_energy / HBAR_EV_S
    density = VACUUM_PERMITTIVITY * ELECTRON_MASS * omega**2 / ELECTRON_CHARGE**2
    return HBAR_J_S * (3 * math.pi**2 * density) ** (1 / 3) / ELECTRON_MASS


def fit_drude(energy: np.ndarray, eps: np.ndarray) -> tuple[float, float, float, float]:
    """(eps_inf, hbar omega_p, hbar gamma, rms) of eps_inf - omega_p^2 / (E (E + i gamma)) fitted
    to eps(E), each point weighted by 1/|eps| so the misfit is relative across the window.

    For a fixed gamma the model is linear in eps_inf and omega_p^2: Re eps = eps_inf -
    omega_p^2 / (E^2 + gamma^2) and Im eps = omega_p^2 gamma / (E (E^2 + gamma^2)). So gamma
    alone is searched -- a scan in log gamma, then golden sections -- and the rest solved,
    with eps_inf >= 1: bound electrons only add to the vacuum's polarisability, and a window
    that catches an interband band would otherwise pay for it with eps_inf < 0."""
    weight = 1 / np.abs(eps)
    target = np.concatenate([weight * eps.real, weight * eps.imag])

    def solve(gamma: float):
        denominator = energy**2 + gamma**2
        design = np.concatenate([
            np.stack([weight, -weight / denominator], axis=1),
            np.stack([np.zeros_like(weight), weight * gamma / (energy * denominator)], axis=1),
        ])
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



def free_path_permittivity(eps, energy, model: Drude, path_um: float):
    """The layer's own permittivity with its free electrons' damping raised to gamma_s:
    eps + omega_p^2 / (omega (omega + i gamma)) - omega_p^2 / (omega (omega + i gamma_s)).
    The data keep everything else -- the interband transitions, and any misfit of the model."""
    gamma_s = model.surface_damping(path_um)
    return eps + model.plasma_energy**2 * (
        1 / (energy * (energy + 1j * model.damping_energy)) - 1 / (energy * (energy + 1j * gamma_s))
    )
