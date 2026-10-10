"""Hydrodynamic (nonlocal) planar multilayers, p polarisation.

Checked against: the extended-precision global matrix (tests/nonlocal_reference.py) for thin and
thick films, a hydrodynamic half-space, a gap between two films and a metal/metal bilayer, at
propagating and evanescent in-plane wavenumbers; coh_tmm when nothing is hydrodynamic; the
invariance under splitting one metal into two identical layers (the metal/metal ABCs); energy:
the dissipation of every layer integrated over its volume (free-electron friction, interband
loss and the GNOR diffusion) against the jump of the Poynting plus hydrodynamic energy flux,
across an interface between two different metals; the Tonks-Dattner (Melnyk-Harrison) absorption
peaks of a thin film above the plasma frequency, omega_n^2 = omega_p^2 + beta^2 [(n pi/d)^2 + K^2].
"""

import numpy as np
import pytest

from pystratify.hydrodynamic import Hydrodynamic
from pystratify.nonlocal_film import film_response, solve_nonlocal_film
from pystratify.planar import coh_tmm
from tests.nonlocal_reference import film_rp

AG = Hydrodynamic.from_ev(9.0, 0.07, 1.39e6)
AU = Hydrodynamic.from_ev(9.03, 0.053, 1.40e6, model="halevi", diffusion=1.9e-4)


def lam(E):
    return 1239.841984 / E


def nd(model, w, eps_b):
    return complex(model.transverse_index(w, eps_b))


CASES = [
    ("Ag 2 nm", lambda w: [1.0, nd(AG, w, 4.5), 1.5], [np.inf, 2.0, np.inf], {1: AG}),
    ("Ag 20 nm", lambda w: [1.0, nd(AG, w, 4.5), 1.5], [np.inf, 20.0, np.inf], {1: AG}),
    ("Ag half-space", lambda w: [1.0, nd(AG, w, 4.5)], [np.inf, np.inf], {1: AG}),
    ("Au|Ag on SiO2", lambda w: [1.0, nd(AU, w, 9.5), nd(AG, w, 4.5), 1.46], [np.inf, 3.0, 4.0, np.inf], {1: AU, 2: AG}),
    ("Ag|1 nm|Ag", lambda w: [1.0, nd(AG, w, 4.5), 1.46, nd(AG, w, 4.5)], [np.inf, 10.0, 1.0, np.inf], {1: AG, 3: AG}),
]


@pytest.mark.parametrize("energy", [3.0, 4.6, 5.5])
@pytest.mark.parametrize("name, index, d, hydro", CASES, ids=[c[0] for c in CASES])
def test_against_global_matrix(name, index, d, hydro, energy):
    w = lam(energy)
    n = index(w)
    k0 = 2 * np.pi / w
    for K in (0.0, 0.7 * k0, 3 * k0, 0.5):
        r = film_response(n, d, w, K, hydro).r[0]
        ref = film_rp(n, d[1:-1], w, hydro, K)
        assert abs(r - ref) <= 1e-13 * abs(ref)


def test_local_stack_is_coh_tmm():
    for n, d, w, th in (([1.0, 1.5, 0.14 + 3.15j, 1.33], [np.inf, 100, 30, np.inf], 614.0, 0.6),
                        ([1.5, 2.0, 1.0], [np.inf, 200, np.inf], 500.0, 0.9)):
        a = solve_nonlocal_film(n, d, w, {}, th)
        b = coh_tmm("p", np.array(n), np.array(d), th, w)
        assert abs(a["r_p"] - b["r"]) <= 1e-14 and abs(a["R_p"] - b["R"]) <= 1e-14 and abs(a["T_p"] - b["T"]) <= 1e-14


def test_identical_metals_have_no_interface():
    model = Hydrodynamic.from_ev(9.0, 0.07, 1.39e6, model="halevi", diffusion=2e-4)
    for energy in (3.0, 5.5):
        w = lam(energy)
        n = nd(model, w, 4.5)
        K = 0.8 * 2 * np.pi / w
        one = film_response([1.0, n, 1.5], [np.inf, 7.0, np.inf], w, K, {1: model})
        two = film_response([1.0, n, n, 1.5], [np.inf, 3.0, 4.0, np.inf], w, K, {1: model, 2: model})
        assert abs(one.r[0] - two.r[0]) <= 1e-14 * abs(one.r[0])


def _dissipation(res, w, n, d, hydro, layer, points=400):
    """Volume dissipation of film ``layer`` (medium index) per unit incident H_y, flux units:
    k0 [ (gamma omega/omega_p^2) |P|^2 + Im(eps_b) |E|^2 - Im(beta^2)/omega_p^2 |eps_b k_L^2 Phi|^2 ],
    P = (eps_T - eps_b) E_T - eps_b E_L (J = -i omega eps_0 P), integrated by Gauss-Legendre."""
    tr = res.traces
    M = len(n)
    s = M - 1 - layer  # sweep region
    regular = res.sweep.regular_amplitudes(np.ones((1, 1, 1), complex))
    a, b, lg = regular[s][1]  # at zeta_s, the incident-side boundary
    a, b = a[0, :, 0] * np.exp(lg[0, 0]), b[0, :, 0] * np.exp(lg[0, 0])
    k0, K = tr.k0, tr.K[0]
    eps = n[layer] ** 2
    kz = tr.kz[layer][0]
    model = hydro[layer]
    eps_b = complex(model.background(w, eps))
    kzL = tr.kzL[s][0]
    kL2 = kzL**2 + K**2
    x, wts = np.polynomial.legendre.leggauss(points)
    t = (x + 1) / 2 * d[layer]  # depth below the incident-side boundary: zeta = zeta_s - t
    wts = wts * d[layer] / 2
    f, g = a[0] * np.exp(1j * kz * t), b[0] * np.exp(-1j * kz * t)  # e^{-i kz (zeta - zeta_s)}
    fL, gL = a[1] * np.exp(1j * kzL * t), b[1] * np.exp(-1j * kzL * t)
    ET = np.stack((kz / (k0 * eps) * (f - g), -K / (k0 * eps) * (f + g)))
    EL = np.stack((1j * K * (fL + gL), 1j * kzL * (fL - gL)))
    Phi = fL + gL
    P = (eps - eps_b) * ET - eps_b * EL
    ratio = model.plasma_wavelength / w  # omega / omega_p
    gratio = model.plasma_wavelength / model.damping_wavelength
    kp2 = (2 * np.pi / model.plasma_wavelength) ** 2
    density = k0 * (gratio * ratio * np.sum(np.abs(P) ** 2, axis=0)
                    + np.imag(eps_b) * np.sum(np.abs(ET + EL) ** 2, axis=0)
                    - np.imag(model.beta_squared(w)) / kp2 * np.abs(eps_b * kL2 * Phi) ** 2)
    return float(np.sum(wts * density))


@pytest.mark.parametrize("energy", [3.0, 5.5])
def test_energy_flux_across_two_metals(energy):
    """Each layer's volume dissipation equals the jump of (Poynting + hydrodynamic) flux; across Au|Ag
    this holds only if n.J and (beta^2/omega_p^2) div J are continuous."""
    w = lam(energy)
    n = [1.0, nd(AU, w, 9.5), nd(AG, w, 4.5), 1.46]
    d = [np.inf, 3.0, 4.0, np.inf]
    hydro = {1: AU, 2: AG}
    res = film_response(n, d, w, 0.6 * 2 * np.pi / w, hydro)
    total = res.flux[0] + res.hydro_flux[0]
    for layer in (1, 2):
        jump = total[layer - 1] - total[layer]
        volume = _dissipation(res, w, n, d, hydro, layer)
        assert abs(jump - volume) <= 1e-9 * abs(volume)
    assert abs(res.hydro_flux[0, 1]) > 1e-6 * abs(total[0])  # the hydrodynamic flux is not negligible here


def test_tonks_dattner_peaks():
    """Free-standing Na-like film (eps_b = 1, d = 1 nm) in vacuum, p light at 45 degrees: absorption
    maxima above omega_p at k_zL d = n pi, omega_n^2 = omega_p^2 + beta^2 (n pi/d)^2 (K^2 beta^2 is
    1e-4 eV^2 here), strong for odd n (Melnyk & Harrison, Phys. Rev. B 2, 835 (1970)), the even ones
    hundreds of times weaker."""
    model = Hydrodynamic.from_ev(5.89, 0.01, 1.05e6)
    d = 1.0
    hb = np.sqrt(0.6) * 1.05e6 * 6.582119569e-16 * 1e9  # hbar beta, eV nm
    energies = np.linspace(5.95, 8.0, 4101)
    A = np.array([solve_nonlocal_film([1.0, nd(model, lam(E), 1.0), 1.0], [np.inf, d, np.inf], lam(E), {1: model},
                                      np.pi / 4)["A_p"] for E in energies])
    is_peak = (A[1:-1] > A[:-2]) & (A[1:-1] > A[2:])
    peaks, heights = energies[1:-1][is_peak], A[1:-1][is_peak]
    predicted = {order: np.sqrt(5.89**2 + (hb * order * np.pi / d) ** 2) for order in (1, 2, 3)}
    found = {order: np.argmin(np.abs(peaks - e)) for order, e in predicted.items()}
    for order in (1, 2, 3):
        assert abs(peaks[found[order]] - predicted[order]) < 2e-3
    assert heights[found[2]] < heights[found[1]] / 100 and heights[found[2]] < heights[found[3]] / 100
