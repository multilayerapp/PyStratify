"""Optical constants: tabulated data, Drude parameters, electron free-path correction.

Wavelengths in this module are in **nanometres** (argument names say so);
the solvers themselves are unit-agnostic and only need ``rad`` and ``lam`` in
the same unit.

Tabulated data are the STRATIFY ``materials/*.mat`` files converted to CSV.
``Ag_P.mat`` is *not* shipped: its n ~ 1.000, k ~ 1e-5 over 124-2066 nm are
not silver (AUDIT.md M8).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["DRUDE", "TABULATED", "refractive_index", "free_path_correction", "list_materials"]

#: Drude fits used by STRATIFY, constants copied verbatim.  ``lam_p`` and
#: ``lam_gamma`` are 2 pi c / omega_p and 2 pi c / gamma_D in nm (G_prefac.m);
#: ``gamma_p`` is gamma_D / omega_p as el_fr_pth.m writes it (ratio of the eV
#: values - equal to lam_p / lam_gamma up to the rounding of the constants);
#: ``vf_c`` is the Fermi velocity over c.
#: Ord = Ordal et al., Appl. Opt. 24, 4493 (1985);
#: Blb = Blaber et al., J. Phys. Chem. C 113, 3041 (2009).
DRUDE = {
    "Au_Ord": dict(lam_p=137.36, lam_gamma=46436.02734, gamma_p=0.0267 / 9.026, vf_c=0.014 / 3),
    "Ag_Ord": dict(lam_p=137.56, lam_gamma=68880.10723, gamma_p=0.018 / 9.013, vf_c=0.0139 / 3),
    "Al_Ord": dict(lam_p=84.05708, lam_gamma=15156.99181, gamma_p=0.0818 / 14.75, vf_c=0.0206 / 3),
    "Au_Blb": dict(lam_p=144.93, lam_gamma=67382.71359, gamma_p=0.0184 / 8.55, vf_c=0.014 / 3),
    "Ag_Blb": dict(lam_p=129.15, lam_gamma=54379.03202, gamma_p=0.0228 / 9.6, vf_c=0.0139 / 3),
    "Al_Blb": dict(lam_p=81.03542, lam_gamma=2071.92836, gamma_p=0.5984 / 15.3, vf_c=0.0206 / 3),
}

#: Tabulated n, k shipped with the package.
TABULATED = {
    "Au_JC": "Au, Johnson & Christy (1972)",
    "Ag_JC": "Ag, Johnson & Christy (1972)",
    "Au_MP": "Au, McPeak et al. (2015)",
    "Ag_MP": "Ag, McPeak et al. (2015)",
    "Al_MP": "Al, McPeak et al. (2015)",
    "Au_P": "Au, Palik",
    "Al_P": "Al, Palik",
    "Si_AS": "Si, Aspnes & Studna (1983)",
}


def list_materials():
    return dict(TABULATED)


@lru_cache(maxsize=None)
def _table(name):
    if name not in TABULATED:
        raise KeyError(f"unknown material {name!r}; available: {sorted(TABULATED)}")
    with resources.files("pystratify.data").joinpath(f"{name}.csv").open() as f:
        data = np.loadtxt(f, delimiter=",", comments="#")
    return data


def refractive_index(name, lam_nm):
    """Complex refractive index n + ik from a tabulated material (cubic spline,
    as STRATIFY's ``interp1(..., 'spline')``).  Raises outside the table."""
    data = _table(name)
    lam_nm = np.asarray(lam_nm, dtype=float)
    lo, hi = data[0, 0], data[-1, 0]
    if np.any(lam_nm < lo) or np.any(lam_nm > hi):
        raise ValueError(f"{name}: data cover {lo:g}-{hi:g} nm")
    n = CubicSpline(data[:, 0], data[:, 1])(lam_nm)
    k = CubicSpline(data[:, 0], data[:, 2])(lam_nm)
    return n + 1j * k


def free_path_correction(lam_nm, n_bulk, radii_nm, drude):
    """Surface-scattering correction for a thin metal shell (OSAC Eq. 37; el_fr_pth.m).

    ``radii_nm`` is ``(r_out,)`` for a core or ``(r_in, r_out)`` for a shell;
    ``drude`` a key of :data:`DRUDE`.  L_eff = 4/3 (r2^3 - r1^3)/(r1^2 + r2^2)
    (Moroz, J. Phys. Chem. C 112, 10641 (2008)).
    """
    p = DRUDE[drude]
    radii = np.atleast_1d(np.asarray(radii_nm, dtype=float))
    if radii.size == 1:
        L = 4 / 3 * radii[0]
    else:
        r1, r2 = radii[0], radii[1]
        L = 4 / 3 * (r2**3 - r1**3) / (r1**2 + r2**2)
    lam_p = p["lam_p"]
    gamma_p = p["gamma_p"]  # gamma_D / omega_p
    gamma_s = gamma_p + p["vf_c"] * lam_p / (2 * np.pi * L)  # (gamma_D + v_F/L) / omega_p
    w = lam_p / np.asarray(lam_nm, dtype=float)  # omega / omega_p
    eps = np.asarray(n_bulk) ** 2 + 1 / (w * (w + 1j * gamma_p)) - 1 / (w * (w + 1j * gamma_s))
    return np.sqrt(eps)
