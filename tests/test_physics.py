"""Physics checks independent of both the MATLAB code and the solver internals.

Every quantity is tested against something computed another way: textbook Mie
theory, published reference values, the optical theorem, plane-wave limits,
interface continuity, brute-force quadrature, or energy conservation.
"""

import sys
from pathlib import Path
import numpy as np
import pytest


import pystratify as ps


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import sphere_checks as spheres  # noqa: E402
import sphere_emitter_checks as emitters  # noqa: E402
from checks import assert_rows  # noqa: E402

LAM = 614.0
AU = 0.27 + 2.93j


def _sol(radii, n, mu=None, wavelength=LAM, L=20):
    return ps.solve(radii, n, wavelength, mu, l_max=L)


def _drude_gold(wavelength_nm):
    """A Drude-Lorentz-free gold stand-in: enough to exercise a dispersive metal."""
    model = ps.DRUDE["Au_Ord"]
    w = model.plasma_wavelength / np.asarray(wavelength_nm)
    return np.sqrt(9.5 - 1 / (w * (w + 1j * model.damping_ratio)))




# ------------------------------------------------------------------ far field



@pytest.mark.parametrize("m, x", spheres.MIE_CASES)
def test_homogeneous_sphere_equals_mie(m, x):
    assert_rows(spheres.homogeneous_sphere_equals_mie(m, x))


def test_bohren_huffman_reference_values():
    assert_rows(spheres.bohren_huffman_reference_values())


def test_identical_shells_collapse_to_single_sphere():
    S3 = _sol([20, 45, 60], [2.0 + 0.1j] * 3 + [1.33], L=25)
    S1 = _sol([60], [2.0 + 0.1j, 1.33], L=25)
    assert np.allclose(S3.a, S1.a, rtol=1e-10) and np.allclose(S3.b, S1.b, rtol=1e-10)


def test_optical_theorem_and_backscattering():
    assert_rows(spheres.optical_theorem_and_backscattering())


def test_wavelength_batch_is_vectorised():
    lam = np.linspace(500, 900, 81)
    n = np.stack([np.full(lam.size, 1.45), _drude_gold(lam), np.ones(lam.size)], 1)
    S = ps.solve([50, 55], n, lam, l_max=12)
    cs = ps.cross_sections(S)
    single = ps.cross_sections(ps.solve([50, 55], n[40], lam[40], l_max=12))
    assert cs.q_ext.shape == (81,) and cs.q_ext[40] == pytest.approx(single.q_ext[0], rel=1e-13)


# ----------------------------------------------------------------- near field


def test_near_field_without_sphere_is_plane_wave():
    nh = 1.33
    S = _sol([30, 60], [nh] * 3, L=40)
    g = np.linspace(-120, 120, 13)
    X, Z = np.meshgrid(g, g)
    f = ps.near_field(S, X, 0 * X + 7.0, Z)
    k = 2 * np.pi * nh / LAM
    assert np.allclose(f.e["x"], np.exp(1j * k * Z), atol=1e-9)
    assert np.allclose(f.e["y"], 0, atol=1e-9) and np.allclose(f.e["z"], 0, atol=1e-9)
    assert np.allclose(f.h["y"], nh * np.exp(1j * k * Z), atol=1e-9)


def test_near_field_regular_on_both_poles_and_at_origin():
    S = _sol([40.0, 50.0], [1.45, AU, 1.33])
    z = np.array([-70.0, -70.0, -45.0, -45.0, 70.0, 70.0, 0.0, 1e-6])
    x = np.array([0.0, 1e-7, 0.0, 1e-7, 0.0, 1e-7, 0.0, 0.0])
    f = ps.near_field(S, x, 0 * x, z)
    mag = np.abs(f.e["x"][0::2])
    for c in "xyz":
        assert np.all(np.abs(f.e[c][0::2] - f.e[c][1::2]) < 1e-6 * mag)


def test_tangential_fields_continuous_across_interfaces():
    rad, ref, mu = [40.0, 50.0], [1.45, AU, 1.33], [1.0, 1.3, 1.0]
    S = _sol(rad, ref, mu)
    th, ph = 1.1, 0.4
    for i, r0 in enumerate(rad):
        rr = np.array([r0 * (1 - 1e-10), r0 * (1 + 1e-10)])
        X, Y, Z = rr * np.sin(th) * np.cos(ph), rr * np.sin(th) * np.sin(ph), rr * np.cos(th)
        f = ps.near_field(S, X, Y, Z)
        for c in ("theta", "phi"):
            assert f.e[c][0] == pytest.approx(f.e[c][1], rel=1e-7)
            assert f.h[c][0] == pytest.approx(f.h[c][1], rel=1e-7)
        eps = np.array(ref) ** 2 / np.array(mu)
        assert eps[i] * f.e["r"][0] == pytest.approx(eps[i + 1] * f.e["r"][1], rel=1e-7)


def test_fields_far_from_the_sphere_converge_with_the_sphere_truncation():
    """The incident wave is exact in the host, so the sphere's own truncation
    (5 orders here) suffices at any distance; summing the incident series with it
    put |E|^2 at 2.03 instead of 1.09 at r = 400 nm."""
    rad, ref, lam = [50.0], [0.25 + 3.0j, 1.33], 600.0
    coarse, fine = ps.solve(rad, ref, lam), ps.solve(rad, ref, lam, l_max=120)
    assert coarse.orders.size == 5
    r = np.array([100.0, 200.0, 400.0, 1000.0, 5000.0])
    x, z = r * np.sin(0.3), r * np.cos(0.3)
    a, b = ps.near_field(coarse, x, 0 * r, z), ps.near_field(fine, x, 0 * r, z)
    for c in ("x", "y", "z"):
        assert np.allclose(a.e[c], b.e[c], rtol=0, atol=1e-6) and np.allclose(a.h[c], b.h[c], rtol=0, atol=1e-6)
    ea, eb = ps.energy_density(coarse, r), ps.energy_density(fine, r)
    assert np.allclose(ea.intensity_e, eb.intensity_e, rtol=1e-6)
    assert np.allclose(ea.intensity_h, eb.intensity_h, rtol=1e-6)
    empty = _sol([30.0], [1.33, 1.33], L=2)  # no sphere, two orders, kr up to 70
    f = ps.near_field(empty, x, 0 * r, z)
    assert np.allclose(f.e["x"], np.exp(2j * np.pi * 1.33 / LAM * z), atol=1e-12)


def test_partial_waves_add_up_to_the_field():
    """The fields of disjoint partial-wave selections add up, and all of them together are
    the field itself wherever the incident series has converged (k r ~ 1 here)."""
    S = _sol([40.0, 50.0], [1.45, AU, 1.33])
    r = np.array([10.0, 39.0, 45.0, 55.0, 80.0])
    th, ph = np.array([0.3, 1.1, 2.0, 2.9, 1.6]), np.array([0.2, 0.9, 2.5, 4.0, 5.5])
    X, Y, Z = r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)
    full = ps.near_field(S, X, Y, Z)
    every = ps.near_field(S, X, Y, Z, orders=range(1, 21))
    low, high = ps.near_field(S, X, Y, Z, orders=[1, 2, 3]), ps.near_field(S, X, Y, Z, orders=range(4, 21))
    tm, te = ps.near_field(S, X, Y, Z, polarisations=ps.TM), ps.near_field(S, X, Y, Z, polarisations=[ps.TE])
    for f in ("e", "h"):
        for c in ("x", "y", "z"):
            whole = getattr(full, f)[c]
            assert np.allclose(getattr(every, f)[c], whole, rtol=0, atol=1e-10 * np.abs(whole).max())
            assert np.allclose(getattr(low, f)[c] + getattr(high, f)[c], getattr(every, f)[c], rtol=1e-12, atol=1e-14)
            assert np.allclose(getattr(tm, f)[c] + getattr(te, f)[c], getattr(every, f)[c], rtol=1e-12, atol=1e-14)
    inside = r < 50
    scattered = ps.near_field(S, X, Y, Z, incident=False, orders=range(1, 21))
    for c in ("x", "y", "z"):
        assert np.allclose(scattered.e[c], ps.near_field(S, X, Y, Z, incident=False).e[c], rtol=1e-12, atol=1e-14)
        assert np.allclose(scattered.e[c][inside], every.e[c][inside], rtol=1e-12, atol=1e-14)


def test_partial_waves_of_the_far_field_add_up_in_amplitude():
    """Chosen partial waves of the scattering amplitudes add up amplitude by amplitude (their
    |S|^2 carries the interference), and an electric dipole alone has the dipole's pattern:
    S_per flat, S_par = S_per cos(theta)."""
    S = _sol([40.0, 50.0], [1.45, AU, 1.33])
    theta = np.linspace(0, np.pi, 37)
    full = ps.scattering_amplitudes(S, theta)
    low, high = ps.scattering_amplitudes(S, theta, orders=[1, 2, 3]), ps.scattering_amplitudes(S, theta, orders=range(4, 21))
    tm, te = ps.scattering_amplitudes(S, theta, polarisations=ps.TM), ps.scattering_amplitudes(S, theta, polarisations=ps.TE)
    for i in range(2):
        scale = np.abs(full[i]).max()
        assert np.allclose(low[i] + high[i], full[i], rtol=1e-12, atol=1e-14 * scale)
        assert np.allclose(tm[i] + te[i], full[i], rtol=1e-12, atol=1e-14 * scale)
    par, per = ps.scattering_amplitudes(S, theta, orders=[1], polarisations=ps.TM)
    assert np.allclose(per, per[:, :1], rtol=1e-12)
    assert np.allclose(par, per * np.cos(theta), rtol=1e-12, atol=1e-14 * np.abs(per).max())


def test_partial_waves_of_the_energy_add_up():
    """Over a sphere the partial waves do not interfere: the averaged intensities and the stored
    energy of disjoint selections add up, and every order together is the whole -- inside the
    particle exactly, in the host once the incident series has converged (k r ~ 1 here)."""
    S = _sol([40.0, 50.0], [1.45, AU, 1.33])
    r = np.array([10.0, 39.0, 45.0, 55.0, 80.0])
    full = ps.energy_density(S, r)
    every = ps.energy_density(S, r, orders=range(1, 21))
    low, high = ps.energy_density(S, r, orders=[1, 2, 3]), ps.energy_density(S, r, orders=range(4, 21))
    tm, te = ps.energy_density(S, r, polarisations=ps.TM), ps.energy_density(S, r, polarisations=[ps.TE])
    for f in ("intensity_e", "intensity_h", "density_e", "density_h"):
        whole = getattr(every, f)
        assert np.allclose(getattr(low, f) + getattr(high, f), whole, rtol=1e-12)
        assert np.allclose(getattr(tm, f) + getattr(te, f), whole, rtol=1e-12)
        assert np.allclose(whole, getattr(full, f), rtol=1e-9)
    stored = ps.shell_energy(S)
    parts = [ps.shell_energy(S, orders=[l], polarisations=p) for l in range(1, 21) for p in (ps.TM, ps.TE)]
    for f in ("electric", "magnetic", "total"):
        assert np.allclose(sum(getattr(part, f) for part in parts), getattr(stored, f), rtol=1e-10)
    dipole = {"orders": [1], "polarisations": ps.TM}
    lommel, quadrature = ps.shell_energy(S, method="lommel", **dipole), ps.shell_energy(S, method="quadrature", **dipole)
    assert np.allclose(lommel.electric, quadrature.electric, rtol=1e-6)
    assert np.allclose(lommel.magnetic, quadrature.magnetic, rtol=1e-6)


def test_partial_wave_selection_is_validated():
    S = _sol([50.0], [AU, 1.33], L=6)
    for bad in ({"orders": [0, 1]}, {"orders": [7]}, {"orders": []}, {"polarisations": ()}, {"polarisations": [2]}):
        with pytest.raises(ValueError):
            ps.near_field(S, 60.0, 0.0, 0.0, **bad)
        with pytest.raises(ValueError):
            ps.scattering_amplitudes(S, [0.0, 1.0], **bad)
        with pytest.raises(ValueError):
            ps.energy_density(S, [60.0], **bad)
        with pytest.raises(ValueError):
            ps.shell_energy(S, **bad)


def test_near_regime_truncation_for_surface_fields():
    rad, ref, lam = [50.0], [0.25 + 3.0j, 1.33], 600.0
    near = ps.solve(rad, ref, lam, regime="near")
    assert near.orders.size == ps.truncation_order(50.0, 1.33, lam, "near")
    point = (55 * np.sin(0.3), 0.0, 55 * np.cos(0.3))
    exact = ps.near_field(ps.solve(rad, ref, lam, l_max=120), *point).intensity_e
    far = ps.near_field(ps.solve(rad, ref, lam), *point).intensity_e
    assert abs(ps.near_field(near, *point).intensity_e / exact - 1) < 1e-6 < abs(far / exact - 1)


# --------------------------------------------------------------------- energy


def _sphere_average(f, r, n=40):
    ct, wt = np.polynomial.legendre.leggauss(n)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    C, P = np.meshgrid(ct, ph, indexing="ij")
    Sn = np.sqrt(1 - C**2)
    out = f(r * Sn * np.cos(P), r * Sn * np.sin(P), r * C)
    return np.sum(wt[:, None] * out) * (2 * np.pi / P.shape[1]) / (4 * np.pi)


def test_energy_density_equals_angular_average_of_near_field():
    rad, ref = [30.0, 42.0], [1.45, AU, 1.33]
    S = _sol(rad, ref, L=16)
    for r in (10.0, 36.0, 80.0):
        ed = ps.energy_density(S, [r])
        avgE = _sphere_average(lambda X, Y, Z: ps.near_field(S, X, Y, Z).intensity_e, r)
        avgH = _sphere_average(lambda X, Y, Z: ps.near_field(S, X, Y, Z).intensity_h, r)
        assert ed.intensity_e[0] == pytest.approx(avgE, rel=1e-9)
        assert ed.intensity_h[0] == pytest.approx(avgH, rel=1e-9)


def test_energy_density_without_sphere_is_one():
    ed = ps.energy_density(_sol([30, 60], [1.5] * 3, L=40), np.linspace(1, 200, 9))
    assert np.allclose(ed.density_e, 1) and np.allclose(ed.density_h, 1) and np.allclose(ed.intensity_e, 1)


@pytest.mark.parametrize(
    "ref",
    [
        [1.45, AU, 1.6, 0.25 + 3.5j, 1.33],
        [2.0, 1.45 + 0.02j, 1.5, 1.2, 1.0],
        [1.45 + 1e-7j, AU, 1.5 + 1e-5j, 1.2 + 2e-4j, 1.0],  # weakly lossy shells
    ],
)
def test_shell_energy_lommel_equals_quadrature(ref):
    S = _sol([10.0, 13.0, 36.0, 48.0], ref, L=16)
    auto = ps.shell_energy(S)
    quad = ps.shell_energy(S, method="quadrature")
    assert np.allclose(auto.electric, quad.electric, rtol=1e-9) and np.allclose(auto.magnetic, quad.magnetic, rtol=1e-9)


def test_weak_loss_lommel_is_ill_conditioned_and_avoided():
    """STRATIFY switches Lommel formulas at Im(n) == 0 only (AUDIT.md M10); the
    lossy form then loses ~1e-17/Im(n) relative accuracy (2e-2 at Im n = 1e-14)."""
    ref = [1.5 + 1e-14j, 1.0]
    S = _sol([200.0], ref, L=30)
    good = ps.shell_energy(S, method="quadrature").electric[0]
    lossless = ps.shell_energy(_sol([200.0], [1.5, 1.0], L=30)).electric[0]
    naive = ps.shell_energy(S, method="lommel").electric[0]
    assert good == pytest.approx(lossless, rel=1e-6)
    assert ps.shell_energy(S).electric[0] == pytest.approx(good, rel=1e-12)
    assert abs(naive / good - 1) > 1e-3


def test_loudon_energy_prefactor():
    model = ps.DRUDE["Au_Ord"]
    lam = 600.0
    w, g, wp = 1 / lam, 1 / model.damping_wavelength, 1 / model.plasma_wavelength
    eps = 1 - wp**2 / (w**2 + 1j * g * w)
    exact = 1 + wp**2 / (w**2 + g**2)
    assert ps.electric_prefactor(eps, lam, model.damping_wavelength) == pytest.approx(exact, rel=1e-12)
    g_e, g_m = ps.energy_prefactors([1.5, np.sqrt(eps), 1.0], [1, 1, 1], lam, [None, "Au_Ord", None])
    assert g_e == pytest.approx([2.25, exact, 1.0], rel=1e-12) and np.all(g_m == 1)


def test_free_path_correction_limits():
    n = 0.2 + 3.0j
    assert ps.free_path_correction(600, n, [1e12], "Au_Ord") == pytest.approx(n, rel=1e-9)
    assert (ps.free_path_correction(600, n, [50, 55], "Au_Ord") ** 2).imag > (n**2).imag


# ---------------------------------------------------------------------- decay

CASES = [
    ([2.5, 1.45, 1.8, 1.33], [1, 1, 1, 1], [20, 60, 80, 120]),
    ([1.45, AU, 1.6, 1.33], [1, 1, 1, 1], [40, 80, 95, 120]),
    ([AU, 1.45, 1.6, 1.33], [1, 1, 1, 1], [51, 60, 80, 95, 120]),
    ([1.45, AU, 1.6, 1.33], [1.3, 1, 1.7, 1.1], [40, 80, 95, 120]),
    ([1.45 + 0.2j, 2.0, 1.6 + 0.05j, 1.0], [1, 1, 1, 1], [60, 95, 300]),
]


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("ref, mu, rd", CASES)
def test_decay_energy_balance(ref, mu, rd, dipole):
    """radiated power + Ohmic loss == total rate from the Green's function."""
    r = ps.decay_rates([50, 70, 90], ref, LAM, rd, mu, dipole=dipole, normalization="shell", tol=1e-9)
    assert r.converged.all()
    assert r.balance_error.max() < 1e-11


def test_decay_close_to_metal_needs_and_gets_high_orders():
    r = ps.decay_rates([50.0], [AU, 1.33], LAM, [50.5, 51.0], tol=1e-6)
    assert r.converged.all() and r.orders_used.min() > 400
    assert r.balance_error.max() < 1e-10


@pytest.mark.parametrize("radius, gap", emitters.MIE_RATE_CASES)
def test_decay_rates_against_mie_reference(radius, gap):
    assert_rows(emitters.decay_rates_against_mie(radius, gap))


@pytest.mark.parametrize("case", emitters.LAYERED_RATE_CASES)
def test_layered_decay_rates_against_transfer_matrices(case):
    assert_rows(emitters.layered_rates_against_transfer_matrices(*case))


@pytest.mark.parametrize("case", emitters.MAGNETIC_RATE_CASES)
def test_magnetic_dipole_rates_against_transfer_matrices(case):
    assert_rows(emitters.magnetic_rates_against_transfer_matrices(*case))


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("eps_s, mu_s", [(4.0, 2.0 + 0.3j), (4.0 + 0.8j, 1.2 + 0.5j)])
def test_magnetic_losses_conserve_energy(dipole, eps_s, mu_s):
    """Loss Im(mu)|H|^2 in a magnetically lossy shell (it was ignored: zero Ohmic loss with real eps)."""
    n = [1.45, np.sqrt(eps_s * mu_s), 1.33]
    r = ps.decay_rates([40.0, 50.0], n, LAM, [20.0, 50.5, 55.0], [1, mu_s, 1], dipole=dipole, tol=1e-10)
    assert r.converged.all() and np.all(r.nonradiative > 0)
    assert r.balance_error.max() < 1e-9


@pytest.mark.parametrize("ref, mu, rd", CASES)
def test_closed_form_absorption_equals_quadrature(ref, mu, rd):
    """Lommel boundary terms (the default) against Gauss-Legendre quadrature."""
    kw = dict(mu=mu, normalization="shell", tol=1e-10, l_max=120, warn=False)
    closed = ps.decay_rates([50, 70, 90], ref, LAM, rd, **kw)
    quadrature = ps.decay_rates([50, 70, 90], ref, LAM, rd, quadrature_nodes=200, **kw)
    assert np.allclose(closed.nonradiative, quadrature.nonradiative, rtol=1e-10, atol=0)


def test_decay_one_nm_from_a_micron_sphere():
    """~17000 orders: the Ohmic loss from boundary terms is O(L) in time and
    memory (quadrature needed O(L^2) and ran out of memory here)."""
    r = ps.decay_rates([1000.0], [0.093 + 4j, 1.0], 633.0, [1001.0], tol=1e-9)
    assert r.converged.all() and r.orders_used.min() > 15000
    assert r.balance_error.max() < 1e-10


def test_decay_free_space_is_one():
    r = ps.decay_rates([50, 70], [1.4, 1.4, 1.4], LAM, [10, 60, 100])
    assert np.allclose(r.radiative, 1, atol=1e-12) and np.allclose(r.nonradiative, 0)
    assert np.allclose(r.total, 1, atol=1e-12)


def test_decay_host_vs_shell_normalisation():
    ref, mu = [1.45, AU, 1.6, 1.33], [1, 1, 1.2, 1]
    for dip, ratio in (("electric", 1.6 * 1.2 / 1.33), ("magnetic", 1.6 * (1.6**2 / 1.2) / 1.33**3)):
        h = ps.decay_rates([50, 70, 90], ref, LAM, [80], mu, dipole=dip, normalization="host")
        s = ps.decay_rates([50, 70, 90], ref, LAM, [80], mu, dipole=dip, normalization="shell")
        for a, b in ((h.radiative, s.radiative), (h.nonradiative, s.nonradiative), (h.total, s.total)):
            assert np.allclose(a / b, ratio)


def test_decay_rejects_bad_input():
    with pytest.raises(ValueError):
        ps.decay_rates([50, 70], [1.45, AU, 1.33], LAM, [60])  # inside metal
    with pytest.raises(ValueError):
        ps.decay_rates([50, 70], [1.45, AU, 1.33 + 0.1j], LAM, [100])  # lossy host
    with pytest.raises(ValueError):
        ps.decay_rates([50, 70], [1.45, AU, 1.33], LAM, [0.0])
    with pytest.raises(ValueError):
        ps.decay_rates([50, 70], [1.45, 1.6 - 0.01j, 1.33], LAM, [60])  # inside a gain shell


# ------------------------------------------------------------ conservation


@pytest.mark.parametrize("size", spheres.LOSSLESS_SIZES)
def test_lossless_multilayers_absorb_nothing(size):
    assert_rows(spheres.lossless_multilayers_absorb_nothing(size))


def test_absorbing_shells_absorb_and_gain_shells_emit():
    rng = np.random.default_rng(7)
    for _ in range(20):
        radii = np.sort(rng.uniform(5, 150, 4))
        n = np.r_[rng.uniform(1.2, 3.0, 4) + 1j * rng.uniform(0, 3, 4), 1.0]
        cs = ps.cross_sections(ps.solve(radii, n, LAM))
        assert np.all(cs.abs_by_order >= -1e-15 * cs.ext.max())
    gain = ps.cross_sections(ps.solve([60.0], [2.0 - 0.05j, 1.0], LAM))
    assert gain.q_abs[0] < 0


def test_rayleigh_limit():
    assert_rows(spheres.rayleigh_limit())
