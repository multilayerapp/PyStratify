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

from .casimir import CasimirPolder, casimir_polder
from .chiral import ChiralSolution, solve_chiral
from .convergence import truncation_order
from .decay import DecayRates, decay_rates, locate_shell
from .drude import DRUDE, DrudeModel, free_path_correction, surface_damping_wavelength
from .emission import EmissionPattern, dipole_far_field, source_covariance
from .ensemble import ShellAverage, SpectralDensity, shell_average, spectral_density
from .energy import (
    EnergyDensity,
    ShellEnergy,
    electric_prefactor,
    energy_density,
    energy_prefactors,
    shell_energy,
)
from .farfield import (
    CrossSections,
    HelicityCrossSections,
    ScatteringPattern,
    amplitude_matrix,
    angular_functions,
    cross_sections,
    helicity_cross_sections,
    mueller_matrix,
    scattering_amplitudes,
    scattering_pattern,
)
from .nearfield import NearField, near_field
from .pairs import GreenDyadic, green_dyadic
from .normalized import NormalizedRates, NormalizedTerms, normalized_decay_rates, normalized_terms
from .rates import EmissionRates, emission_rates
from .sheets import Feibelman, Sheet, graphene_conductivity
from .riccati import log_riccati
from .solver import TE, TM, Solution, solve
from .problem import Problem, PlaneWave, PointDipole
from .observables import solve_problem
from .cylindrical import solve_cylinder, CylinderSolution, cross_widths, cylinder_pattern

__version__ = "0.9.0"

__all__ = [
    "Problem", "PlaneWave", "PointDipole", "solve_problem",
    "solve_cylinder", "CylinderSolution", "cross_widths", "cylinder_pattern",
    "solve",
    "Solution",
    "solve_chiral",
    "ChiralSolution",
    "TM",
    "TE",
    "truncation_order",
    "cross_sections",
    "CrossSections",
    "helicity_cross_sections",
    "HelicityCrossSections",
    "scattering_amplitudes",
    "amplitude_matrix",
    "mueller_matrix",
    "scattering_pattern",
    "ScatteringPattern",
    "dipole_far_field",
    "EmissionPattern",
    "source_covariance",
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
    "emission_rates",
    "EmissionRates",
    "locate_shell",
    "Feibelman",
    "Sheet",
    "graphene_conductivity",
    "DRUDE",
    "DrudeModel",
    "free_path_correction",
    "surface_damping_wavelength",
    "log_riccati",
    "normalized_decay_rates",
    "casimir_polder",
    "CasimirPolder",
    "green_dyadic",
    "GreenDyadic",
    "shell_average",
    "ShellAverage",
    "spectral_density",
    "SpectralDensity",
    "normalized_terms",
    "NormalizedTerms",
    "NormalizedRates",
]
