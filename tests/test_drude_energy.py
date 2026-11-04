"""Energy-form Drude model, ported from the multilayer.app API (assurance WP07-D2), against the frozen API code."""

import numpy as np
import pytest

from pystratify import drude
from tests.oracles import api_spheres_drude as api

ENERGY = np.linspace(0.3, 3.5, 200)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def test_existing_surface_unchanged():
    assert drude.effective_path([10.0]) == 4 / 3 * 10.0
    assert set(drude.DRUDE) == {"Au_Ord", "Ag_Ord", "Al_Ord", "Au_Blb", "Ag_Blb", "Al_Blb"}


@pytest.mark.parametrize("radii", [[12.0], [10.0, 25.0], [5.0, 8.0, 40.0]])
def test_layer_effective_path_matches_api(radii):
    for i in range(len(radii)):
        assert abs(drude.layer_effective_path(radii, i) - api.effective_path(radii, i)) <= 1e-14 * radii[-1]
    # nm in, nm out: one sphere / one shell agree with the older helper
    assert drude.layer_effective_path(radii[:2], len(radii[:2]) - 1) == pytest.approx(drude.effective_path(radii[:2]), rel=1e-14)


def test_model_matches_api():
    new = drude.DrudeEnergyModel(3.9, 8.9, 0.07, 1.4e6, 0.9)
    old = api.Drude(3.9, 8.9, 0.07, 1.4e6, 0.9)
    path_um = 0.0123
    assert rel(new.surface_damping(path_um * 1000), old.surface_damping(path_um)) <= 1e-9
    assert rel(new.permittivity(ENERGY), old.permittivity(ENERGY)) <= 1e-9
    gamma_s = old.surface_damping(path_um)
    assert rel(new.permittivity(ENERGY, gamma_s), old.permittivity(ENERGY, gamma_s)) <= 1e-9
    assert rel(drude.free_path_permittivity(1.0 + 2j + 0 * ENERGY, ENERGY, new, path_um * 1000),
               api.free_path_permittivity(1.0 + 2j + 0 * ENERGY, ENERGY, old, path_um)) <= 1e-9
    s_new, s_old = new.summary(12.3), old.summary(0.0123)
    assert s_new["effective_path"] == 12.3 and s_new["fit"] is None
    assert rel(s_new["surface_damping_energy"], s_old["surface_damping_energy"]) <= 1e-9
    assert rel(s_new["fermi_velocity_free_electron"], s_old["fermi_velocity_free_electron"]) <= 1e-9


@pytest.mark.parametrize("plasma", [0.5, 5.0, 9.0, 13.0])
def test_fermi_velocity(plasma):
    assert rel(drude.free_electron_fermi_velocity(plasma), api.free_electron_fermi_velocity(plasma)) <= 1e-9
    assert drude.free_electron_fermi_velocity(9.0) == pytest.approx(1.39e6, rel=0.01)


def _lorentz(e, strength, e0, width):
    return strength / (e0**2 - e**2 - 1j * width * e)


@pytest.mark.parametrize(
    "eps_inf, plasma, gamma, interband",
    [
        (1.0, 9.0, 0.07, False),
        (3.0, 8.5, 0.05, False),
        (6.0, 9.0, 0.07, True),  # gold-like: Drude plus an interband Lorentz band
        (2.0, 7.0, 0.3, True),
    ],
)
def test_fit_drude_matches_api(eps_inf, plasma, gamma, interband):
    eps = eps_inf - plasma**2 / (ENERGY * (ENERGY + 1j * gamma))
    if interband:
        eps = eps + _lorentz(ENERGY, 4.0, 2.6, 0.5)
    new, old = drude.fit_drude(ENERGY, eps), api.fit_drude(ENERGY, eps)
    assert rel(new, old) <= 1e-8
    if not interband and eps_inf >= 1:
        assert new[0] == pytest.approx(eps_inf, rel=1e-6) and new[1] == pytest.approx(plasma, rel=1e-6)


def test_fit_drude_no_free_electrons():
    with pytest.raises(ValueError):
        drude.fit_drude(ENERGY, 4.0 - 0.5j * np.ones_like(ENERGY))
