"""Sign conventions the Learn pages state (Fresnel equations, ellipsometry, transfer matrix, Bragg mirrors): phases,
which R and T of a lossless stack cannot see.

r_p = -r_s at normal incidence (Born and Wolf §1.5.2, Byrnes); r_p > 0 below Brewster for external reflection;
planar.ellips (the calculator's card) gives Delta = arg(-r_p/r_s); the bare-substrate inversion of
docs/learn/films/01-interfaces/01-fresnel-equations.md takes rho = r_p/r_s and returns n + ik. The characteristic
matrix of docs/learn/films/02-transfer-matrix/01-transfer-matrix.md carries -i with n + ik (Macleod's +i belongs to
n - ik), and the R_N of docs/learn/films/03-periodic-stacks/01-bragg-mirrors.md takes the amplitude-basis C.
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


def characteristic_matrix(n, d, wavelength, sign):
    delta = 2 * np.pi * n * d / wavelength
    return np.array([[np.cos(delta), sign * 1j * np.sin(delta) / n], [sign * 1j * n * np.sin(delta), np.cos(delta)]])


def admittance_RT(layers, wavelength, sign, n0=1.0, ns=1.52):
    product = np.eye(2, dtype=complex)
    for n, d in layers:
        product = product @ characteristic_matrix(n, d, wavelength, sign)
    B, C = product @ np.array([1, ns])
    return abs((n0 - C / B) / (n0 + C / B)) ** 2, 4 * n0 * ns / abs(n0 * B + C) ** 2


@pytest.mark.parametrize("layers", ([(2.0 + 0.5j, 120.0)], [(0.13 + 3.9j, 30.0)], [(1.46, 90.0), (2.3 + 0.05j, 60.0)]))
def test_characteristic_matrix_takes_minus_i_with_n_plus_ik(layers):
    n = [1.0, *(n for n, _ in layers), 1.52]
    d = [np.inf, *(d for _, d in layers), np.inf]
    engine = planar.coh_tmm("s", n, d, 0.0, 550.0)
    R, T = admittance_RT(layers, 550.0, -1)
    assert abs(R - engine["R"]) < 1e-12 and abs(T - engine["T"]) < 1e-12
    R_macleod, _ = admittance_RT(layers, 550.0, +1)  # Macleod's +i belongs to n - ik
    assert abs(R_macleod - engine["R"]) > 1e-2


@pytest.mark.parametrize("wavelength", (550.0, 620.0, 760.0))
@pytest.mark.parametrize("periods", (3, 8))
def test_yeh_reflectance_takes_the_amplitude_basis(wavelength, periods):
    n1, n2 = 1.45, 2.3
    a, b = 600.0 / 4 / n1, 600.0 / 4 / n2
    r12, t12, t21 = (n1 - n2) / (n1 + n2), 2 * n1 / (n1 + n2), 2 * n2 / (n1 + n2)
    phase = lambda n, d: np.diag([np.exp(-2j * np.pi * n * d / wavelength), np.exp(2j * np.pi * n * d / wavelength)])
    cell = (np.array([[1, r12], [r12, 1]]) / t12 @ phase(n2, b)
            @ np.array([[1, -r12], [-r12, 1]]) / t21 @ phase(n1, a))
    K = np.arccos(complex((cell[0, 0] + cell[1, 1]) / 2))
    U = np.sin(periods * K) / np.sin(K)
    R_N = abs(cell[1, 0]) ** 2 * abs(U) ** 2 / (abs(cell[1, 0]) ** 2 * abs(U) ** 2 + 1)
    n = [n1, *[n2, n1] * (periods - 1), n2, n1]
    d = [np.inf, *[b, a] * (periods - 1), b, np.inf]
    assert abs(R_N - planar.coh_tmm("s", n, d, 0.0, wavelength)["R"]) < 1e-12
