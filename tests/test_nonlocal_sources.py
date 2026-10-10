"""Point dipoles near hydrodynamic metals: films and concentric cylinders.

Checked against: the local emitters (FilmSource, CylinderSource) when nothing is hydrodynamic and
in the local limit beta -> 0; energy balance between the total rate (Green's function at the source)
and the escaping plus absorbed power (Poynting plus hydrodynamic fluxes), independent paths; the
reduction of quenching by nonlocality next to a metal (Ford & Weber, Phys. Rep. 113, 195 (1984);
Christensen et al., ACS Nano 8, 1745 (2014)), vanishing a few nanometres away.
"""

import numpy as np
import pytest

from pystratify.cylinder_emission import CylinderSource
from pystratify.hydrodynamic import Hydrodynamic
from pystratify.nonlocal_sources import NonlocalCylinderSource, NonlocalFilmSource
from pystratify.planar_emission import FilmSource

AG = Hydrodynamic.from_ev(9.0, 0.07, 1.39e6)
W = 1239.841984 / 3.0
N_AG = complex(AG.transverse_index(W, 4.5))
TINY = Hydrodynamic(AG.plasma_wavelength, AG.damping_wavelength, AG.fermi_velocity * 1e-6)


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("depth", [19.0, 15.0])
def test_film_emitter(dipole, depth):
    n, d = [1.0, 1.46, N_AG, 1.5], [np.inf, 20.0, 30.0, np.inf]
    local = FilmSource(n, d, W, 1, depth, dipole).rates(1e-8)
    same = NonlocalFilmSource(n, d, W, 1, depth, dipole, {}).rates(1e-8)
    assert np.allclose(same.total, local.total, rtol=1e-14)
    limit = NonlocalFilmSource(n, d, W, 1, depth, dipole, {2: TINY}).rates(1e-8)
    assert np.max(np.abs(limit.total - local.total) / local.total) < 1e-5
    nonlocal_ = NonlocalFilmSource(n, d, W, 1, depth, dipole, {2: AG}).rates(1e-8)
    assert nonlocal_.converged and np.max(nonlocal_.balance_error) <= 1e-9
    if dipole == "electric" and depth == 19.0:  # 1 nm from silver: nonlocality caps the quenching
        assert np.all(nonlocal_.total < 0.5 * local.total)
    if dipole == "magnetic":  # the radial magnetic dipole radiates TE only: untouched
        assert abs(nonlocal_.total[0] - local.total[0]) <= 1e-9 * local.total[0]


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("radii, n, r, layer", [([20.0, 25.0], [1.46, N_AG, 1.33], 15.0, 1),
                                               ([20.0, 25.0], [1.46, N_AG, 1.33], 28.0, 1)],
                         ids=["in the core", "outside"])
def test_cylinder_emitter(radii, n, r, layer, dipole):
    local = CylinderSource(radii, n, W, r, dipole, m_max=60)._rates_once(1e-6, 20000)
    same = NonlocalCylinderSource(radii, n, W, r, dipole, m_max=60, hydrodynamic={})._rates_once(1e-6, 20000)
    assert np.max(np.abs(same.total - local.total) / local.total) <= 1e-8
    assert np.max(np.abs(same.escape - local.escape) / local.escape) <= 1e-8
    nonlocal_ = NonlocalCylinderSource(radii, n, W, r, dipole, m_max=60, hydrodynamic={layer: AG})._rates_once(1e-6, 20000)
    assert np.max(nonlocal_.balance_error) <= 1e-8
    if dipole == "electric":  # the electric near field of the shell is screened: less quenching
        assert np.all(nonlocal_.total < local.total)
    else:  # a magnetic dipole sees the shift of the shell's resonances, either way
        assert np.max(np.abs(nonlocal_.total / local.total - 1)) > 1e-3
