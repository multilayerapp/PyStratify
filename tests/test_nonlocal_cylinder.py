"""Hydrodynamic (nonlocal) concentric cylinders at any axial wavenumber.

Checked against: the extended-precision global matrix (tests/nonlocal_reference.py) with normal,
oblique and evanescent axial wavenumbers, m = 0 included, metal/metal interfaces and two
hydrodynamic layers, up to m = 60; the closed form of a single wire at normal incidence (Ruppin,
Opt. Commun. 190, 205 (2001); with the hard-wall ABC n.J = 0, which for an interband background
eps_b reads eps_b E_rho^L = (eps_T - eps_b) E_rho^T); the multipole resonances
omega_l = omega_p/sqrt(2) + l beta/(2R) of a small Drude wire (Raza et al., Phys. Rev. B 84,
121412(R) (2011)); the legacy local solver; the local limit; passivity.
"""

import mpmath as mp
import numpy as np
import pytest

import pystratify as ps
from pystratify.hydrodynamic import Hydrodynamic
from pystratify.nonlocal_cylinder import solve_nonlocal_cylinder
from tests.nonlocal_reference import cylinder_t

AG = Hydrodynamic.from_ev(9.0, 0.07, 1.39e6)
AU = Hydrodynamic.from_ev(9.03, 0.053, 1.40e6, model="halevi", diffusion=1.9e-4)


def lam(E):
    return 1239.841984 / E


def nd(model, w, eps_b):
    return complex(model.transverse_index(w, eps_b))


CASES = [
    ("Ag wire", [5.0], lambda w: [nd(AG, w, 4.5), 1.33], {0: AG}),
    ("SiO2@Ag", [10.0, 12.0], lambda w: [1.46, nd(AG, w, 4.5), 1.33], {1: AG}),
    ("Au@Ag", [6.0, 8.0], lambda w: [nd(AU, w, 9.5), nd(AG, w, 4.5), 1.33], {0: AU, 1: AG}),
    ("Ag|SiO2|Ag", [4.0, 5.0, 7.0], lambda w: [nd(AG, w, 4.5), 1.46, nd(AG, w, 4.5), 1.33], {0: AG, 2: AG}),
]


@pytest.mark.parametrize("energy", [3.0, 5.5])
@pytest.mark.parametrize("axial", [0.0, 0.6, 2.5])  # beta / k_host; 2.5 is evanescent in the host
@pytest.mark.parametrize("name, radii, index, hydro", CASES, ids=[c[0] for c in CASES])
def test_against_global_matrix(name, radii, index, hydro, axial, energy):
    w = lam(energy)
    n = index(w)
    beta = axial * 2 * np.pi * 1.33 / w
    sol = solve_nonlocal_cylinder(radii, n, w, hydro, beta=beta, m_max=12)
    for m in (0, 1, -3, 12):
        ref = cylinder_t(radii, n, w, hydro, m, beta)
        got = sol.t[list(sol.orders).index(m)]
        mask = np.abs(ref) > 1e-12 * np.max(np.abs(ref))  # the cross terms vanish at beta = 0
        # 1e-11: above the screened plasma frequency eps_T ~ 1.8 nearly matches the host (1.77), and
        # the m = 0 coefficient, proportional to the contrast, carries the usual eps/contrast
        assert np.all(np.abs(got - ref)[mask] <= 1e-11 * np.abs(ref)[mask])
        assert np.all(np.abs(got - ref)[~mask] <= 1e-13 * np.max(np.abs(ref)))


@pytest.mark.parametrize("axial", [0.0, 0.3, 3.0])
def test_high_orders(axial):
    """The quasi-static row reduction keeps m = 60 at machine precision (beta = 0 included, where the
    unreduced E-parallel-to-axis rows lose ~(m/x)^2 of the working precision)."""
    w = lam(3.0)
    n = [1.46, nd(AG, w, 4.5), 1.33]
    beta = axial * 2 * np.pi * 1.33 / w
    sol = solve_nonlocal_cylinder([10.0, 12.0], n, w, {1: AG}, beta=beta, m_max=60)
    for m in (40, 60):
        ref = cylinder_t([10.0, 12.0], n, w, {1: AG}, m, beta, dps=80)
        got = sol.t[list(sol.orders).index(m)]
        ok = np.abs(ref) > 1e-300
        assert np.all(np.abs(got - ref)[ok] <= 1e-12 * np.abs(ref)[ok])


def test_local_stack_matches_legacy():
    for radii, n, w in (([0.05, 0.1], [1.5, 2, 1], 0.6), ([20.0, 25.0], [1.45, 0.14 + 3.15j, 1.33], 614.0)):
        k = 2 * np.pi * np.real(n[-1]) / w
        for beta in (0.0, 0.5 * k, 0.95 * k):
            a = solve_nonlocal_cylinder(radii, n, w, {}, beta=beta, m_max=30)
            b = ps.solve_cylinder(radii, n, w, beta=beta, m_max=30)
            assert np.max(np.abs(a.t - b.t)) <= 1e-12 * np.max(np.abs(b.t))


def test_closed_form_wire():
    """Normal incidence, E perpendicular to the axis (the M wave): T_m = -[c_m J(x0) + sh J(x0) J'(xT)
    - sT J'(x0) J(xT)] / [c_m H(x0) + ...] with c_m = sh m^2 (eps_T/eps_b - 1) J(xT) J(xL)/(xL xT J'(xL))."""
    for energy in (2.5, 3.5, 6.0):
        w = lam(energy)
        n_m = nd(AG, w, 4.5)
        sol = solve_nonlocal_cylinder([3.0], [n_m, 1.33], w, {0: AG}, beta=0.0, m_max=6)
        with mp.workdps(40):
            eT, eh = mp.mpc(n_m) ** 2, mp.mpf(1.33) ** 2
            eb = mp.mpc(complex(AG.background(w, n_m**2)))
            kL = mp.mpc(complex(AG.longitudinal_wavenumber(w, n_m**2)))
            k0 = 2 * mp.pi / w
            x0, xT, xL = k0 * mp.sqrt(eh) * 3, k0 * mp.sqrt(eT) * 3, kL * 3
            sh, sT = mp.sqrt(eh), mp.sqrt(eT)
            J = mp.besselj
            dJ = lambda m, z: (J(m - 1, z) - J(m + 1, z)) / 2
            H = lambda m, z: mp.hankel1(m, z)
            dH = lambda m, z: (H(m - 1, z) - H(m + 1, z)) / 2
            for m in range(1, 7):
                c = sh * m**2 * (eT / eb - 1) * J(m, xT) * J(m, xL) / (xL * xT * dJ(m, xL))
                num = c * J(m, x0) + sh * J(m, x0) * dJ(m, xT) - sT * dJ(m, x0) * J(m, xT)
                den = c * H(m, x0) + sh * H(m, x0) * dJ(m, xT) - sT * dH(m, x0) * J(m, xT)
                ref = complex(-num / den)
                for mm in (m, -m):
                    got = sol.t[list(sol.orders).index(mm), 1, 1]
                    assert abs(got - ref) <= 1e-12 * abs(ref)


def test_multipole_blue_shift():
    """Small Drude wire in vacuum (eps_b = 1): the l-th resonance sits at w_p/sqrt(2) + l beta/(2R) to
    first order in beta/(w_p R) (Raza et al. 2011); R = 10 nm, l = 1, 2."""
    model = Hydrodynamic.from_ev(5.89, 1e-4, 1.05e6)
    beta_ev_nm = np.sqrt(0.6) * 1.05e6 * 6.582119569e-16 * 1e9  # hbar beta in eV nm
    R = 10.0
    for l in (1, 2):
        energies = np.linspace(4.0, 4.4, 4001)
        local, nonlocal_ = [], []
        for E in energies:
            w = lam(E)
            n_m = complex(model.transverse_index(w, 1.0))
            nonlocal_.append(abs(solve_nonlocal_cylinder([R], [n_m, 1.0], w, {0: model}, m_max=l).t[l + l, 1, 1]))
            local.append(abs(ps.solve_cylinder([R], [n_m, 1.0], w, m_max=l).t[l + l, 1, 1]))
        shift = energies[np.argmax(nonlocal_)] - energies[np.argmax(local)]
        expected = l * beta_ev_nm / (2 * R)
        assert abs(shift / expected - 1) < 0.03


def test_local_limit_and_passivity():
    w = lam(3.0)
    n = [1.46, nd(AG, w, 4.5), 1.33]
    k = 2 * np.pi * 1.33 / w
    for beta in (0.0, 0.5 * k):
        local = ps.solve_cylinder([10.0, 12.0], n, w, beta=beta, m_max=8).t
        for scale in (1e-3, 1e-5):
            model = Hydrodynamic(AG.plasma_wavelength, AG.damping_wavelength, AG.fermi_velocity * scale)
            t = solve_nonlocal_cylinder([10.0, 12.0], n, w, {1: model}, beta=beta, m_max=8).t
            assert np.max(np.abs(t - local)) <= 30 * scale * np.max(np.abs(local))
        widths = ps.cross_widths(solve_nonlocal_cylinder([10.0, 12.0], n, w, {1: AG}, beta=beta, m_max=8))
        assert widths["absorption"] > 0 and widths["scattering"] > 0
