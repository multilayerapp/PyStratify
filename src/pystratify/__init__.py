"""PyStratify: light scattering by multilayered (stratified) spheres.

Quick start::

    import numpy as np
    import pystratify as ps

    wavelength = np.linspace(500, 900, 401)              # nm
    radii = [50, 55]                                     # nm: SiO2 core, Au shell
    n_au = ...                                           # (401,) complex, e.g. from refractiveindex.info
    n = np.stack([np.full(401, 1.45), n_au, np.full(401, 1.33)], axis=1)   # host last
    sol = ps.solve(radii, n, wavelength)
    ps.cross_sections(sol).q_ext                         # (401,)
"""

from .convergence import truncation_order
from .decay import DecayRates, decay_rates, locate_shell
from .drude import DRUDE, DrudeModel, free_path_correction, surface_damping_wavelength
from .energy import (
    EnergyDensity,
    ShellEnergy,
    electric_prefactor,
    energy_density,
    energy_prefactors,
    shell_energy,
)
from .farfield import CrossSections, angular_functions, cross_sections, scattering_amplitudes
from .nearfield import NearField, near_field
from .normalized import NormalizedRates, normalized_decay_rates
from .riccati import log_riccati
from .solver import TE, TM, Solution, solve

__version__ = "0.3.0"

__all__ = [
    "solve",
    "Solution",
    "TM",
    "TE",
    "truncation_order",
    "cross_sections",
    "CrossSections",
    "scattering_amplitudes",
    "angular_functions",
    "near_field",
    "NearField",
    "energy_density",
    "EnergyDensity",
    "shell_energy",
    "ShellEnergy",
    "electric_prefactor",
    "energy_prefactors",
    "decay_rates",
    "DecayRates",
    "locate_shell",
    "DRUDE",
    "DrudeModel",
    "free_path_correction",
    "surface_damping_wavelength",
    "log_riccati",
    "normalized_decay_rates",
    "NormalizedRates",
]
