"""Dispersion models: Kramers-Kronig consistency, limits, conventions, and pyElli (benchmarks/dispersion_codes.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

from pystratify import dispersion as dm
from pystratify import planar

ENERGY = np.linspace(0.25, 9.0, 120)
WAVELENGTH = dm.HC_EV_NM / ENERGY


def eps2_of(model):
    return lambda energy: model.epsilon(dm.HC_EV_NM / np.asarray(energy, float)).imag


@pytest.mark.parametrize("model, kinks", [
    (dm.TaucLorentz(A=120, E0=3.6, C=2.4, Eg=1.6), [1.6, 3.6]),
    (dm.TaucLorentz(A=40, E0=5.0, C=0.4, Eg=4.2), [4.2, 5.0]),
    (dm.Gaussian(A=2.0, E0=4.0, B=0.6), [4.0]),
    (dm.Lorentz(A=3.0, E0=4.5, B=0.8), [4.5]),
])
def test_closed_form_real_parts_are_the_kramers_kronig_transform_of_the_imaginary(model, kinks):
    eps = model.epsilon(WAVELENGTH)
    transform = dm.kramers_kronig_real(eps2_of(model), ENERGY, kinks)
    assert np.max(np.abs(transform - eps.real)) < 1e-12 * np.max(np.abs(eps))


def cauchy_weight_kk(eps2, energy, breaks):
    """P int_0^inf E' eps2 / (E'^2 - E^2) by QUADPACK, split at the kinks, with the Cauchy weight on the segment holding
    E: a quadrature independent of the module's subtraction."""
    g = lambda e: e * eps2(np.array([e]))[0]  # noqa: E731
    edges, total = [0.0, *breaks], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if a < energy < b:
            value, _ = integrate.quad(lambda e: g(e) / (e + energy), a, b, weight="cauchy", wvar=energy,
                                      limit=2000, epsabs=1e-14, epsrel=1e-13)
        else:
            value, _ = integrate.quad(lambda e: g(e) / (e**2 - energy**2), a, b, limit=2000, epsabs=1e-14, epsrel=1e-13)
        total += value
    tail, _ = integrate.quad(lambda e: g(e) / (e**2 - energy**2), edges[-1], np.inf, epsabs=1e-15)
    return 2 / np.pi * (total + tail)


def test_cody_lorentz_against_an_independent_principal_value_and_its_tail():
    model = dm.CodyLorentz(A=100, E0=3.6, Gamma=2.4, Eg=1.6, Ep=0.8, Et=1.8, Eu=0.05)
    energies = np.array([0.5, 1.7, 1.79, 1.81, 2.5, 3.0, 6.0])
    eps = model.epsilon(dm.HC_EV_NM / energies)
    reference = np.array([cauchy_weight_kk(model.eps2, e, (1.6, 1.8, 3.6, 20.0, 200.0)) for e in energies])
    assert np.max(np.abs(eps.real - reference)) < 1e-12 * np.max(np.abs(eps))
    # at a kink itself (Eg, Et) eps1 stays continuous
    for kink in (1.6, 1.8):
        around = model.epsilon(dm.HC_EV_NM / np.array([kink - 1e-7, kink, kink + 1e-7])).real
        assert np.max(np.abs(np.diff(around))) < 1e-5
    # the Urbach tail joins the Cody-Lorentz absorption continuously at Et and decays as exp(E/Eu) below
    below, above = model.eps2(np.array([1.8 - 1e-9])), model.eps2(np.array([1.8 + 1e-9]))
    assert below == pytest.approx(above, rel=1e-6)
    tail = model.eps2(np.array([1.5, 1.55]))
    assert np.log(tail[1] * 1.55 / (tail[0] * 1.5)) == pytest.approx(0.05 / 0.05)
    # Et = 0: no tail, nothing below the gap
    assert np.all(dm.CodyLorentz(A=100, E0=3.6, Gamma=2.4, Eg=1.6, Ep=0.8).eps2(np.array([0.5, 1.6])) == 0)


def test_tauc_lorentz_absorbs_only_above_the_gap_and_is_finite_at_it():
    model = dm.TaucLorentz(A=120, E0=3.6, C=2.4, Eg=1.6)
    eps = model.epsilon(dm.HC_EV_NM / np.array([1.0, 1.6, 1.6 + 1e-9, 2.0]))
    assert np.all(np.isfinite(eps))
    assert eps[0].imag == eps[1].imag == 0 and eps[3].imag > 0
    assert eps[1].real == pytest.approx(eps[2].real, rel=1e-7)


def test_index_glass_catalogue_and_cauchy_forms():
    silica = dm.Sellmeier(B=(0.6961663, 0.4079426, 0.8974794), C=(0.0684043**2, 0.1162414**2, 9.896161**2))
    assert silica.index(587.56).real == pytest.approx(1.458464, abs=2e-6)  # Malitson's n_d
    cauchy = dm.Cauchy(A=1.45, B=0.0036, C=1e-4, k_amplitude=0.01, k_exponent=1.5, band_edge_nm=400)
    n = cauchy.index(np.array([400.0, 633.0]))
    assert n[0] == pytest.approx(1.45 + 0.0036 / 0.16 + 1e-4 / 0.0256 + 0.01j)
    assert n[1].imag == pytest.approx(0.01 * np.exp(1.5 * (dm.HC_EV_NM / 633 - dm.HC_EV_NM / 400)))


def test_drude_forms_agree_and_are_metallic():
    tau, rho = 9e-15, 2.2e-8
    gamma = dm.HBAR_EV_S / tau
    plasma = np.sqrt(dm.HBAR_EV_S**2 / (dm.VACUUM_PERMITTIVITY * rho * tau))
    resistivity = dm.DrudeResistivity(rho=rho, tau=tau).epsilon(WAVELENGTH)
    energy = dm.DrudeEnergy(Ep=plasma, Gamma=gamma).epsilon(WAVELENGTH)
    assert np.max(np.abs(resistivity - energy) / np.abs(energy)) < 1e-13
    below = ENERGY < plasma
    assert np.all(energy.real[below] < 0) and np.all(energy.imag > 0)


def test_poles_are_lossless_and_the_ir_pole_is_e0_zero():
    assert np.all(dm.Pole(A=40, E0=9.5).epsilon(WAVELENGTH).imag == 0)
    assert np.allclose(dm.Pole(A=0.3).epsilon(WAVELENGTH), -0.3 / ENERGY**2, rtol=1e-14)


METAL, VOID, SILICON = dm.Constant(0.18, 3.43), dm.Constant(1.0), dm.Constant(3.88, 0.02)


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("a, b", [(METAL, VOID), (SILICON, VOID), (dm.Constant(0.05, 4.0), dm.Constant(1.5))])
def test_bruggeman_solves_its_equation_on_the_passive_branch(a, b, fraction):
    eps = dm.Bruggeman(a, b, fraction).epsilon(500.0)
    ea, eb = a.epsilon(500.0), b.epsilon(500.0)
    residual = fraction * (ea - eps) / (ea + 2 * eps) + (1 - fraction) * (eb - eps) / (eb + 2 * eps)
    assert abs(residual) < 1e-13
    assert eps.imag >= 0  # multilayer.app's oscillator-dialog mix returned gain here for metals (2026-10-16)
    assert eps == pytest.approx(dm.Bruggeman(b, a, 1 - fraction).epsilon(500.0), rel=1e-13)


def test_bruggeman_epsilon_is_the_model_on_permittivities():
    ea, eb = METAL.epsilon(500.0), VOID.epsilon(500.0)
    assert isinstance(dm.bruggeman_epsilon(ea, eb, 0.3), complex)
    assert dm.bruggeman_epsilon(ea, eb, 0.3) == dm.Bruggeman(METAL, VOID, 0.3).epsilon(500.0)
    grid = dm.bruggeman_epsilon(np.full(3, ea), np.full(3, eb), 0.3)
    assert grid.shape == (3,) and np.all(grid == dm.bruggeman_epsilon(ea, eb, 0.3))


def test_effective_medium_limits():
    film = dm.TaucLorentz(A=120, E0=3.6, C=2.4, Eg=1.6) + dm.EpsilonInf(1.2)
    assert np.allclose(dm.Bruggeman(film, VOID, 1.0).epsilon(WAVELENGTH), film.epsilon(WAVELENGTH), rtol=1e-13)
    assert np.allclose(dm.Bruggeman(film, VOID, 0.0).epsilon(WAVELENGTH), 1.0, rtol=1e-13)
    assert dm.MaxwellGarnett(1.5, METAL, 0.0).epsilon(500.0) == pytest.approx(2.25)
    assert dm.MaxwellGarnett(1.5, METAL, 1.0).epsilon(500.0) == pytest.approx(METAL.epsilon(500.0))
    # dilute: both reduce to eps_h + 3 f eps_h (eps_i - eps_h) / (eps_i + 2 eps_h)
    f, eh, ei = 1e-6, 2.25, METAL.epsilon(500.0)
    first_order = eh + 3 * f * eh * (ei - eh) / (ei + 2 * eh)
    slope = (first_order - eh) / f
    assert (dm.MaxwellGarnett(1.5, METAL, f).epsilon(500.0) - eh) / f == pytest.approx(slope, rel=1e-5)
    assert (dm.Bruggeman(METAL, 1.5, f).epsilon(500.0) - eh) / f == pytest.approx(slope, rel=1e-5)


def test_table_interpolates_and_never_extrapolates():
    table = dm.Table([400, 500, 600], [1.5, 1.6, 1.7], [0.0, 0.1, 0.0])
    assert table.index(450.0) == pytest.approx(1.55 + 0.05j)
    with pytest.raises(ValueError, match="does not extrapolate"):
        table.epsilon([350.0, 450.0])
    with pytest.raises(ValueError):
        dm.Table([500, 400], [1.5, 1.6], [0, 0])


def test_sums_shapes_and_validation():
    model = dm.EpsilonInf(2.0) + dm.Lorentz(1, 4, 1) + dm.Pole(5, 9.5)
    assert isinstance(model, dm.Sum) and len(model.terms) == 3
    parts = dm.Lorentz(1, 4, 1).epsilon(WAVELENGTH) + dm.Pole(5, 9.5).epsilon(WAVELENGTH) + 2.0
    assert np.allclose(model.epsilon(WAVELENGTH), parts, rtol=1e-15)
    with pytest.raises(TypeError):
        2.0 + dm.Lorentz(1, 4, 1)
    with pytest.raises(TypeError):
        dm.Lorentz(1, 4, 1) + 2.0
    assert isinstance(model.epsilon(633.0), complex)
    assert model.epsilon(np.full((2, 3), 633.0)).shape == (2, 3)
    assert np.all(dm.index_from_epsilon(np.array([-10 + 0.1j, 2.25, -1 - 1e-20j])).imag >= 0)
    with pytest.raises(ValueError):
        dm.TaucLorentz(A=1, E0=1.0, C=2.0, Eg=0.5)
    with pytest.raises(ValueError):
        dm.Gaussian(1, 4, 0)
    with pytest.raises(ValueError):
        dm.Bruggeman(METAL, VOID, 1.5)
    with pytest.raises(ValueError):
        dm.Lorentz(1, 4, 1).epsilon(-500.0)


def test_a_model_feeds_the_film_solver():
    film = dm.Cauchy(A=1.46, B=0.0035)
    n = film.index(633.0)
    r = planar.coh_tmm("s", [1.0, n, 3.88], [np.inf, 100.0, np.inf], 0.0, 633.0)["r"]
    r01, r12 = (1 - n) / (1 + n), (n - 3.88) / (n + 3.88)
    phase = np.exp(2j * 2 * np.pi * n * 100.0 / 633.0)
    assert r == pytest.approx((r01 + r12 * phase) / (1 + r01 * r12 * phase), rel=1e-13)


def test_dispersion_models_agree_with_pyelli():
    pytest.importorskip("elli")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from dispersion_codes import compare

    worst = compare()
    for name, (total, imaginary) in worst.items():
        if name.startswith("Cody-Lorentz"):  # pyElli's eps1 is a coarse 1000-point KK sum
            assert total < 5e-3 and imaginary < 1e-13, name
        else:
            assert total < 1e-13 and imaginary < 1e-13, name
