"""Sign conventions the Learn pages state (Fresnel equations, ellipsometry): phases, which R and T cannot see.

r_p = -r_s at normal incidence (Born and Wolf §1.5.2, Byrnes); r_p > 0 below Brewster for external reflection;
planar.ellips (the calculator's card) gives Delta = arg(-r_p/r_s); the bare-substrate inversion of
docs/learn/films/01-interfaces/01-fresnel-equations.md takes rho = r_p/r_s and returns n + ik.
"""

import numpy as np
import pytest

from pystratify import ellipsometry as el
from pystratify import planar

SUBSTRATES = (1.5, 3.9 + 0.02j, 1.5 + 0.3j, 0.2 + 3.0j)


def amplitudes(n2, angle_deg, n1=1.0):
    th1 = np.radians(angle_deg)
    th2 = planar.snell(n1, n2, th1)
    return planar.interface_r("s", n1, n2, th1, th2), planar.interface_r("p", n1, n2, th1, th2)


def invert(rho, angle_deg, n1=1.0):
    th = np.radians(angle_deg)
    return n1 * np.sin(th) * np.sqrt(1 + ((1 - rho) / (1 + rho)) ** 2 * np.tan(th) ** 2)


@pytest.mark.parametrize("n2", SUBSTRATES)
def test_rp_is_minus_rs_at_normal_incidence(n2):
    r_s, r_p = amplitudes(n2, 0.0)
    assert abs(r_p - (n2 - 1) / (n2 + 1)) < 1e-14
    assert abs(r_p + r_s) < 1e-14


def test_rp_is_positive_below_brewster_and_changes_sign_through_it():
    brewster = np.degrees(np.arctan(1.5))
    for angle in (10.0, 30.0, brewster - 1):
        r_s, r_p = amplitudes(1.5, angle)
        assert r_s.real < 0 and r_p.real > 0
    for angle in (brewster + 1, 70.0, 85.0):
        r_s, r_p = amplitudes(1.5, angle)
        assert r_s.real < 0 and r_p.real < 0
    assert abs(amplitudes(1.5, brewster)[1]) < 1e-14


def test_the_card_delta_is_the_phase_of_minus_rp_over_rs():
    for angle, expected in ((30.0, 0.0), (50.0, 0.0), (70.0, 180.0)):
        delta = np.degrees(planar.ellips([1.0, 1.5], [np.inf, np.inf], np.radians(angle), 500.0)["Delta"]) % 360
        assert abs((delta - expected + 180) % 360 - 180) < 1e-9
    for n2 in SUBSTRATES:
        r_s, r_p = amplitudes(n2, 65.0)
        delta = planar.ellips([1.0, n2], [np.inf, np.inf], np.radians(65.0), 500.0)["Delta"]
        assert abs(np.exp(1j * delta) - (-r_p / r_s) / abs(r_p / r_s)) < 1e-12


@pytest.mark.parametrize("n2", SUBSTRATES)
@pytest.mark.parametrize("angle", (45.0, 70.0))
def test_bare_substrate_inversion_takes_rp_over_rs(n2, angle):
    r_s, r_p = amplitudes(n2, angle)
    assert abs(invert(r_p / r_s, angle) - n2) < 1e-9
    card = planar.ellips([1.0, n2], [np.inf, np.inf], np.radians(angle), 500.0)
    rho_card = np.tan(card["psi"]) * np.exp(1j * card["Delta"])
    assert abs(invert(-rho_card, angle) - n2) < 1e-9  # the card's Delta + 180 deg
    spectra = el.simulate(el.Sample(layers=[], substrate=n2), 500.0, angle)
    rho_instrument = np.tan(np.radians(float(spectra.psi))) * np.exp(1j * np.radians(float(spectra.delta)))
    assert abs(invert(rho_instrument, angle) - np.conj(n2)) < 1e-9  # an ellipsometer's own n - ik
