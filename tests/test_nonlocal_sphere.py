"""Hydrodynamic (nonlocal) multilayered spheres.

Checked against: an independent extended-precision global matrix (tests/nonlocal_reference.py) for
metal/dielectric and metal/metal interfaces, several hydrodynamic shells, sub-nanometre shells and
frequencies below and above the plasma frequency; the closed form of Ruppin (Phys. Rev. Lett. 31,
1434 (1973)) with an interband background (Raza et al., J. Phys.: Condens. Matter 27, 183204
(2015), Eq. 33); the exact equivalence of the hard-wall model and an order-dependent Feibelman
d_perp at a single surface, through PyStratify's own d-parameter Mie theory; the legacy local solver
up to l = 1200 when no region is hydrodynamic; the local limit beta -> 0; passivity.
"""

import mpmath as mp
import numpy as np
import pytest

import pystratify as ps
from pystratify.hydrodynamic import Hydrodynamic
from pystratify.nonlocal_sphere import solve_nonlocal_sphere
from tests.nonlocal_reference import sphere_t

AG = Hydrodynamic.from_ev(9.0, 0.07, 1.39e6)  # Ag-like, with eps_b = 4.5 below
AU = Hydrodynamic.from_ev(9.03, 0.053, 1.40e6, model="halevi", diffusion=1.9e-4)  # GNOR, Halevi beta(omega)


def lam(E):
    return 1239.841984 / E


def n_drude(model, wavelength, eps_b):
    return complex(model.transverse_index(wavelength, eps_b))


CASES = [
    ("Ag sphere", [5.0], lambda w: [n_drude(AG, w, 4.5), 1.33], {0: AG}),
    ("SiO2@Ag", [10.0, 12.0], lambda w: [1.46, n_drude(AG, w, 4.5), 1.33], {1: AG}),
    ("SiO2@Ag 0.3 nm", [10.0, 10.3], lambda w: [1.46, n_drude(AG, w, 4.5), 1.33], {1: AG}),
    ("Au@Ag metal-metal", [6.0, 8.0], lambda w: [n_drude(AU, w, 9.5), n_drude(AG, w, 4.5), 1.0], {0: AU, 1: AG}),
    ("Ag|SiO2|Ag|Au", [4.0, 5.0, 7.0, 7.5],
     lambda w: [n_drude(AG, w, 4.5), 1.46, n_drude(AG, w, 4.5), n_drude(AU, w, 9.5), 1.2], {0: AG, 2: AG, 3: AU}),
]


@pytest.mark.parametrize("energy", [3.0, 5.5])
@pytest.mark.parametrize("name, radii, index, hydro", CASES, ids=[c[0] for c in CASES])
def test_against_global_matrix(name, radii, index, hydro, energy):
    w = lam(energy)
    n = index(w)
    t = solve_nonlocal_sphere(radii, n, w, hydro, l_max=20).solution.t[ps.TM, 0]
    for l in (1, 4, 20):
        ref = sphere_t(radii, n, w, hydro, l, "TM")
        assert abs(t[l - 1] - ref) <= 1e-12 * abs(ref)


def test_te_is_local_and_matches_legacy():
    w = lam(3.0)
    n = [1.46, n_drude(AG, w, 4.5), 1.33]
    sol = solve_nonlocal_sphere([10.0, 12.0], n, w, {1: AG}, l_max=30).solution
    ref = ps.solve([10.0, 12.0], n, w, l_max=30)
    assert np.max(np.abs(np.expm1(sol.log_t[ps.TE] - ref.log_t[ps.TE]))) <= 1e-12  # rounding of |log T| ~ 300


@pytest.mark.parametrize("radii, n, w", [([5.0], [0.14 + 3.15j, 1.33], 614.0),
                                         ([10.0, 10.2], [1.5, 1.5000001, 1.0], 500.0),
                                         ([400.0, 420.0], [1.45, 0.14 + 3.15j, 1.33], 614.0)])
def test_local_stack_matches_legacy_to_high_order(radii, n, w):
    """No hydrodynamic region: the channel recursion is the legacy solver, l = 1..1200, |T| down
    to 1e-9800 carried in logarithms (nearly matched shells included: the quasi-static row reduction)."""
    a = solve_nonlocal_sphere(radii, n, w, {}, l_max=1200).solution
    b = ps.solve(radii, n, w, l_max=1200)
    assert np.max(np.abs(np.expm1(a.log_t - b.log_t))) <= 1e-11


def test_ruppin_closed_form():
    """a_l = [eps_m j(x_m) psi'(x_d) - eps_d j(x_d)(psi'(x_m) + Delta_l)] / [... h ...],
    Delta_l = l(l+1) j_l(x_m) (eps_m/eps_b - 1) j_l(x_L) / (x_L j_l'(x_L))."""
    for energy in (2.5, 3.4, 6.0):
        w = lam(energy)
        n_m = n_drude(AG, w, 4.5)
        t = solve_nonlocal_sphere([3.0], [n_m, 1.33], w, {0: AG}, l_max=6).solution.t[ps.TM, 0]
        with mp.workdps(40):
            em, ed, eb = mp.mpc(n_m) ** 2, mp.mpf(1.33) ** 2, mp.mpc(complex(AG.background(w, n_m**2)))
            kL = mp.mpc(complex(AG.longitudinal_wavenumber(w, n_m**2)))
            k0 = 2 * mp.pi / w
            xm, xd, xL = k0 * mp.sqrt(em) * 3, k0 * mp.mpf(1.33) * 3, kL * 3

            def j(l, z):
                return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(l + 0.5, z)

            def h(l, z):
                return j(l, z) + 1j * mp.sqrt(mp.pi / (2 * z)) * mp.bessely(l + 0.5, z)

            for l in range(1, 7):
                dpsi = lambda f, z: z * f(l - 1, z) - l * f(l, z)  # (z f_l)'
                djL = j(l - 1, xL) - (l + 1) * j(l, xL) / xL
                delta = l * (l + 1) * j(l, xm) * (em / eb - 1) * j(l, xL) / (xL * djL)
                num = em * j(l, xm) * dpsi(j, xd) - ed * j(l, xd) * (dpsi(j, xm) + delta)
                den = em * j(l, xm) * dpsi(h, xd) - ed * h(l, xd) * (dpsi(j, xm) + delta)
                a_l = complex(num / den)  # Bohren-Huffman a_l = -T_TM
                assert abs(-t[l - 1] - a_l) <= 1e-12 * abs(a_l)


def test_single_surface_is_order_dependent_d_perp():
    """At one surface the hard wall is exactly Feibelman's d_perp,l, d_par = 0:
    d_perp,l = -[(eps_T/eps_b - 1)/(eps_T/eps_d - 1)] j_l(k_L R)/(k_L j_l'(k_L R))."""
    w = lam(3.4)
    n_m = n_drude(AG, w, 4.5)
    eps_T, eps_d = n_m**2, 1.33**2
    eps_b = complex(AG.background(w, eps_T))
    kL = complex(AG.longitudinal_wavenumber(w, eps_T))
    t = solve_nonlocal_sphere([4.0], [n_m, 1.33], w, {0: AG}, l_max=12).solution.t[ps.TM, 0]
    with mp.workdps(40):
        for l in (1, 2, 5, 12):
            z = mp.mpc(kL) * 4
            jl = mp.sqrt(mp.pi / (2 * z)) * mp.besselj(l + 0.5, z)
            jm = mp.sqrt(mp.pi / (2 * z)) * mp.besselj(l - 0.5, z)
            ratio = complex(jl / (jm - (l + 1) * jl / z))  # j_l / j_l'
            d_perp = -((eps_T / eps_b - 1) / (eps_T / eps_d - 1)) * ratio / kL
            sol = ps.solve([4.0], [n_m, 1.33], w, l_max=l, sheets={0: ps.Feibelman(d_perp, 0.0)})
            assert abs(sol.t[ps.TM, 0, l - 1] - t[l - 1]) <= 1e-12 * abs(t[l - 1])


def test_local_limit():
    w = lam(3.0)
    n = [1.46, n_drude(AG, w, 4.5), 1.33]
    local = ps.solve([10.0, 12.0], n, w, l_max=8).t[ps.TM, 0]
    for scale, bound in ((1e-2, 3e-2), (1e-4, 3e-4), (1e-6, 3e-6)):
        model = Hydrodynamic(AG.plasma_wavelength, AG.damping_wavelength, AG.fermi_velocity * scale)
        t = solve_nonlocal_sphere([10.0, 12.0], n, w, {1: model}, l_max=8).solution.t[ps.TM, 0]
        assert np.max(np.abs(t - local) / np.abs(local)) <= bound


def test_passive_and_blue_shifted():
    """Absorption >= 0 across the spectrum, and the dipole resonance of a small Drude sphere moves
    to higher energy (Ruppin 1973)."""
    energies = np.linspace(3.0, 4.6, 321)
    w = lam(energies)
    model = Hydrodynamic.from_ev(9.0, 0.02, 1.39e6)
    n_m = model.transverse_index(w, 4.5)
    n = np.stack([n_m, np.full_like(n_m, 1.0)], axis=1)
    cs_nl = ps.cross_sections(solve_nonlocal_sphere([3.0], n, w, {0: model}, l_max=6).solution)
    cs_loc = ps.cross_sections(ps.solve([3.0], n, w, l_max=6))
    q_abs = cs_nl.abs_by_order.sum(axis=(1, 2))
    assert np.all(q_abs > 0)
    shift = energies[np.argmax(cs_nl.ext_by_order.sum(axis=(1, 2)))] - energies[np.argmax(cs_loc.ext_by_order.sum(axis=(1, 2)))]
    assert 0.02 < shift < 0.2


# ------------------------------------------------------------------------------------------- emitters
from pystratify.nonlocal_sphere import nonlocal_sphere_rates  # noqa: E402

AU_STRATIFY = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm


@pytest.mark.parametrize("radii, n, r", [([5.0], [AU_STRATIFY, 1.0], 6.0), ([8.0, 10.0], [1.45, AU_STRATIFY, 1.33], 7.0),
                                         ([8.0, 10.0, 12.0], [1.45, AU_STRATIFY, 1.6, 1.33], 11.0),
                                         ([60.0, 70.0], [1.45, AU_STRATIFY, 1.33], 59.0)])
@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
def test_rates_local_limit(radii, n, r, dipole):
    """Nothing hydrodynamic: decay_rates (normalized route), total, radiative and nonradiative."""
    a = nonlocal_sphere_rates(radii, n, 614.0, {}, r, dipole=dipole, tol=1e-12)
    b = ps.decay_rates(radii, n, 614.0, r, dipole=dipole, normalization="shell", tol=1e-12, warn=False)
    assert np.max(np.abs(a.total - b.total[0]) / b.total[0]) <= 1e-9
    assert np.max(np.abs(a.radiative - b.radiative[0]) / b.radiative[0]) <= 1e-12
    assert np.max(np.abs(a.absorbed - b.nonradiative[0]) / np.abs(b.nonradiative[0])) <= 1e-9


EMITTERS = [
    ("Ag sphere, outside", [5.0], lambda w: [n_drude(AG, w, 4.5), 1.33], {0: AG}, 6.0),
    ("SiO2@Ag, in the core", [8.0, 10.0], lambda w: [1.46, n_drude(AG, w, 4.5), 1.33], {1: AG}, 7.0),
    ("Au@Ag, 0.5 nm outside", [6.0, 8.0], lambda w: [n_drude(AU, w, 9.5), n_drude(AG, w, 4.5), 1.0], {0: AU, 1: AG}, 8.5),
    ("Ag|SiO2|Ag, in the gap", [4.0, 6.0, 8.0], lambda w: [n_drude(AG, w, 4.5), 1.46, n_drude(AG, w, 4.5), 1.33],
     {0: AG, 2: AG}, 5.0),
]


@pytest.mark.parametrize("energy", [3.0, 5.5])
@pytest.mark.parametrize("name, radii, index, hydro, r", EMITTERS, ids=[e[0] for e in EMITTERS])
def test_rates_energy_balance(name, radii, index, hydro, r, energy):
    """The total rate (field reflected at the source) equals the power reaching the host plus the
    power absorbed in every region (Poynting plus hydrodynamic fluxes): two independent paths."""
    w = lam(energy)
    for dipole in ("electric", "magnetic"):
        rates = nonlocal_sphere_rates(radii, index(w), w, hydro, r, dipole=dipole, tol=1e-10)
        assert rates.converged and np.max(rates.balance_error) <= 1e-12
        assert np.all(rates.absorbed_by_region >= -1e-12 * rates.total)


def test_rates_against_global_matrix_t():
    """Source in the host: total radial rate 1 + (3/2) Re sum l(l+1)(2l+1) T_l xi_l(x0)^2 / x0^4 with T_l
    from the extended-precision global matrix, the same 25 orders."""
    for energy in (3.0, 5.5):
        w = lam(energy)
        n = [1.46, n_drude(AG, w, 4.5), 1.33]
        rates = nonlocal_sphere_rates([8.0, 10.0], n, w, {1: AG}, 11.0, l_max=25)
        with mp.workdps(30):
            x0 = mp.mpf(2 * np.pi * 1.33 / w * 11.0)
            total = mp.mpf(1)
            for l in range(1, 26):
                xi = mp.sqrt(mp.pi * x0 / 2) * (mp.besselj(l + 0.5, x0) + 1j * mp.bessely(l + 0.5, x0))
                total += mp.re(1.5 * l * (l + 1) * (2 * l + 1) * mp.mpc(sphere_t([8.0, 10.0], n, w, {1: AG}, l, "TM")) * xi**2 / x0**4)
        assert abs(rates.total[0] - float(total)) <= 1e-11 * float(total)
