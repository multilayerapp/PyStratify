"""PyStratify: light scattering by multilayered (stratified) spheres.

A Python re-implementation of STRATIFY (Rasskazov, Carney & Moroz, OSA
Continuum 3, 2290 (2020)) on an overflow-free formulation of its recursive
transfer-matrix method, with the corrections listed in AUDIT.md.

Quick start::

    import numpy as np
    import pystratify as ps

    lam = np.linspace(500, 900, 401)                      # nm
    rad = [50, 55]                                        # nm; SiO2 core, Au shell
    ref = np.stack([np.full(lam.size, 1.45),
                    ps.refractive_index("Au_JC", lam),
                    np.ones(lam.size)], axis=1)          # (W, N+1), host last
    sol = ps.solve(rad, ref, [1, 1, 1], lam, l_max=ps.l_max(55, 1.0, lam))
    cs = ps.cross_sections(sol)                           # vectorised over lam
    print(cs.q_ext.max())
"""

from .convergence import l_max
from .decay import DecayRates, decay_rates, locate_shell
from .energy import (
    EnergyDensity,
    ShellEnergy,
    energy_density,
    g_electric,
    g_prefactors,
    total_energy,
)
from .farfield import (
    CrossSections,
    angular_functions,
    cross_sections,
    scattering_amplitudes,
)
from .materials import (
    DRUDE,
    TABULATED,
    free_path_correction,
    list_materials,
    refractive_index,
)
from .nearfield import NearField, near_field
from .riccati import log_riccati
from .solver import Solution, solve

__version__ = "0.2.0"

__all__ = [
    "solve",
    "Solution",
    "l_max",
    "cross_sections",
    "CrossSections",
    "scattering_amplitudes",
    "angular_functions",
    "near_field",
    "NearField",
    "energy_density",
    "EnergyDensity",
    "total_energy",
    "ShellEnergy",
    "g_electric",
    "g_prefactors",
    "decay_rates",
    "DecayRates",
    "locate_shell",
    "refractive_index",
    "free_path_correction",
    "list_materials",
    "log_riccati",
    "DRUDE",
    "TABULATED",
]
