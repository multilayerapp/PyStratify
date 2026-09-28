"""Physics checks independent of both the MATLAB code and the solver internals.

Every quantity is tested against something computed another way: textbook Mie
theory, published reference values, the optical theorem, plane-wave limits,
interface continuity, brute-force quadrature, or energy conservation.
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from scipy.special import spherical_jn, spherical_yn

import pystratify as ps

from pystratify.references import layered_decay_rates, sphere_decay_rates

LAM = 614.0
AU = 0.27 + 2.93j


def _sol(radii, n, mu=None, wavelength=LAM, L=20):
    return ps.solve(radii, n, wavelength, mu, l_max=L)


def _drude_gold(wavelength_nm):
    """A Drude-Lorentz-free gold stand-in: enough to exercise a dispersive metal."""
    model = ps.DRUDE["Au_Ord"]
    w = model.plasma_wavelength / np.asarray(wavelength_nm)
    return np.sqrt(9.5 - 1 / (w * (w + 1j * model.damping_ratio)))


def _riccati(l, z, derivative=False):
    """psi, xi (or derivatives) from scipy's spherical Bessel functions."""
    j, y = spherical_jn(l, z), spherical_yn(l, z)
    if not derivative:
        return z * j, z * (j + 1j * y)
    jd, yd = spherical_jn(l, z, derivative=True), spherical_yn(l, z, derivative=True)
    return j + z * jd, (j + 1j * y) + z * (jd + 1j * yd)


# ------------------------------------------------------------------ far field


def _mie_ab(m, x, lmax):
    """Bohren & Huffman Eq. 4.53, written independently."""
    l = np.arange(1, lmax + 1)
    mx = m * x
    (psi, xi), (dpsi, dxi) = _riccati(l, x), _riccati(l, x, True)
    psim, dpsim = _riccati(l, mx)[0], _riccati(l, mx, True)[0]
    a = (m * psim * dpsi - psi * dpsim) / (m * psim * dxi - xi * dpsim)
    b = (psim * dpsi - m * psi * dpsim) / (psim * dxi - m * xi * dpsim)
    return a, b


@pytest.mark.parametrize("m, x", [(1.55, 5.213), (1.5 + 0.1j, 2.0), (AU / 1.33, 1.1)])
def test_homogeneous_sphere_equals_mie(m, x):
    nh = 1.33
    r = x * LAM / (2 * np.pi * nh)
    L = ps.truncation_order(r, nh, LAM) + 5
    S = _sol([r], [m * nh, nh], L=L)
    a, b = _mie_ab(m, x, L)
    assert np.allclose(S.a[0], a, rtol=1e-10, atol=1e-15)
    assert np.allclose(S.b[0], b, rtol=1e-10, atol=1e-15)


def test_bohren_huffman_reference_values():
    # BHMIE test case: m = 1.55, a = 0.525 um, lambda = 0.6328 um
    x = 2 * np.pi * 0.525 / 0.6328
    r = x * LAM / (2 * np.pi)
    S = _sol([r], [1.55, 1.0], L=20)
    cs = ps.cross_sections(S)
    assert cs.q_ext[0] == pytest.approx(3.10543, abs=1e-5)
    assert cs.q_abs[0] == pytest.approx(0, abs=1e-12)
    l = S.orders
    q_back = np.abs(np.sum((2 * l + 1) * (-1) ** l * (S.a[0] - S.b[0]))) ** 2 / x**2
    assert q_back == pytest.approx(2.92534, abs=1e-5)


def test_identical_shells_collapse_to_single_sphere():
    S3 = _sol([20, 45, 60], [2.0 + 0.1j] * 3 + [1.33], L=25)
    S1 = _sol([60], [2.0 + 0.1j, 1.33], L=25)
    assert np.allclose(S3.a, S1.a, rtol=1e-10) and np.allclose(S3.b, S1.b, rtol=1e-10)


def test_optical_theorem_and_backscattering():
    S = _sol([50.0, 55.0], [1.45, AU, 1.0])
    cs = ps.cross_sections(S)
    par, per = ps.scattering_amplitudes(S, np.array([0.0, np.pi / 2, np.pi]))
    k = 2 * np.pi / LAM
    assert cs.ext[0] == pytest.approx(4 * np.pi / k**2 * np.real(-par[0, 0]), rel=1e-12)
    assert np.all(np.isfinite(par)) and par[0, 0] == pytest.approx(per[0, 0])
    p2, _ = ps.scattering_amplitudes(S, np.array([np.pi - 1e-6]))
    assert p2[0, 0] == pytest.approx(par[0, 2], rel=1e-8)
    th = np.linspace(0, np.pi, 4001)
    p, q = ps.scattering_amplitudes(S, th)
    csca = np.pi / k**2 * trapezoid((np.abs(p[0]) ** 2 + np.abs(q[0]) ** 2) * np.sin(th), th)
    assert csca == pytest.approx(cs.sca[0], rel=1e-5)


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


@pytest.mark.parametrize("radius, gap", [(50.0, 1.0), (100.0, 5.0), (10.0, 5.0)])
def test_decay_rates_against_mie_reference(radius, gap):
    """Majic & Le Ru, Appl. Opt. 59, 1293 (2020): silver at 633 nm, emitter in air,
    against their Eqs. 34-37 summed in mpmath."""
    ag = 0.093 + 4j
    total_perp, total_par, rad_perp, rad_par = sphere_decay_rates(radius, ag, 633.0, radius + gap, orders=1400)
    r = ps.decay_rates([radius], [ag, 1.0], 633.0, [radius + gap], tol=1e-10)
    assert r.converged.all()
    assert np.allclose(r.total[0], [total_perp, total_par], rtol=1e-9, atol=0)
    assert np.allclose(r.radiative[0], [rad_perp, rad_par], rtol=1e-12, atol=0)
    nonrad = np.array([total_perp - rad_perp, total_par - rad_par])
    assert np.allclose(r.nonradiative[0], nonrad, rtol=1e-9, atol=0)


@pytest.mark.parametrize(
    "radii, n, wavelength, rd, orders, dps",
    [
        ([50.0, 55.0], [1.45, AU, 1.33], LAM, 49.0, 1250, 110),  # SiO2@Au nanoshell, 1 nm inside the core
        ([50.0, 55.0], [1.45, AU, 1.33], LAM, 56.0, 1400, 115),  # ... and 1 nm outside
        ([50.0, 52.0], [1.45, AU, 1.33], LAM, 49.5, 2400, 100),  # 2-nm shell, 0.5 nm inside
        ([10.0, 13.0, 36.0, 48.0], [1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33], 690.0, 14.0, 450, 240),  # matryoshka
    ],
)
def test_layered_decay_rates_against_transfer_matrices(radii, n, wavelength, rd, orders, dps):
    """Emitters inside and outside metal shells, against high-precision transfer matrices."""
    total, radiative = layered_decay_rates(radii, n, wavelength, rd, orders, dps)
    r = ps.decay_rates(radii, n, wavelength, [rd], normalization="shell", tol=1e-10)
    assert r.converged.all()
    assert np.allclose(r.total[0], total, rtol=2e-10, atol=0)
    assert np.allclose(r.radiative[0], radiative, rtol=1e-12, atol=0)
    assert np.allclose(r.nonradiative[0], np.subtract(total, radiative), rtol=2e-10, atol=0)


@pytest.mark.parametrize(
    "radii, n, mu, wavelength, rd, orders, dps",
    [
        ([50.0], [0.093 + 4j, 1.0], [1, 1], 633.0, 51.0, 1250, 60),  # 1 nm from silver
        ([50.0, 55.0], [1.45, AU, 1.33], [1, 1, 1], LAM, 49.5, 2400, 150),  # inside a nanoshell's core
        ([50.0, 60.0], [AU, 1.45, 1.33], [1, 1, 1], LAM, 55.0, 400, 100),  # in the shell over a metal core
        ([40.0, 45.0, 60.0], [1.45, AU, 2.0, 1.33], [1, 1, 2.0, 1], LAM, 50.0, 400, 120),  # mu = 2 spacer
        ([100.0], [3.7 + 0.005j, 1.0], [1, 1], 800.0, 110.0, 350, 60),  # silicon, magnetic Mie mode
    ],
)
def test_magnetic_dipole_rates_against_transfer_matrices(radii, n, mu, wavelength, rd, orders, dps):
    total, radiative = layered_decay_rates(radii, n, wavelength, rd, orders, dps, mu=mu, dipole="magnetic")
    r = ps.decay_rates(radii, n, wavelength, [rd], mu, dipole="magnetic", normalization="shell", tol=1e-10)
    assert r.converged.all()
    assert np.allclose(r.total[0], total, rtol=1e-11, atol=0)
    assert np.allclose(r.radiative[0], radiative, rtol=1e-12, atol=0)
    assert np.allclose(r.nonradiative[0], np.subtract(total, radiative), rtol=1e-10, atol=0)


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


@pytest.mark.parametrize("size", [0.5, 20.0, 300.0])
def test_lossless_multilayers_absorb_nothing(size):
    """Energy conservation of the scattering solution: Q_abs = 0 to rounding."""
    rng = np.random.default_rng(int(size * 10))
    radii = np.sort(rng.uniform(0.05, 1.0, 8)) * size * LAM / (2 * np.pi)
    n = np.r_[rng.uniform(1.2, 3.5, 8), 1.33]
    cs = ps.cross_sections(ps.solve(radii, n, LAM))
    assert abs(cs.q_abs[0]) < 1e-12 * cs.q_ext[0]


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
    m, x = 1.5 + 0.2j, 1e-4
    radius = x * LAM / (2 * np.pi)
    a1 = ps.solve([radius], [m, 1.0], LAM, l_max=3).a[0, 0]
    assert a1 == pytest.approx(-2j * x**3 / 3 * (m**2 - 1) / (m**2 + 2), rel=1e-7)
