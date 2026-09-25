"""Physics checks independent of both the MATLAB code and the solver internals.

Every quantity is tested against something computed another way: textbook Mie
theory, published reference values, the optical theorem, plane-wave limits,
interface continuity, brute-force quadrature, or energy conservation.
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

import pystratify as ps
from pystratify.legacy import ric_h, ric_h_d, ric_j, ric_j_d

LAM = 614.0
AU = 0.27 + 2.93j


def _sol(rad, ref, mu=None, lam=LAM, L=20):
    mu = [1] * len(ref) if mu is None else mu
    return ps.solve(rad, ref, mu, lam, L)


# ------------------------------------------------------------------ far field


def _mie_ab(m, x, lmax):
    """Bohren & Huffman Eq. 4.53, written independently."""
    l = np.arange(1, lmax + 1)
    mx = m * x
    psi, dpsi, xi, dxi = ric_j(l, x), ric_j_d(l, x), ric_h(l, x), ric_h_d(l, x)
    psim, dpsim = ric_j(l, mx), ric_j_d(l, mx)
    a = (m * psim * dpsi - psi * dpsim) / (m * psim * dxi - xi * dpsim)
    b = (psim * dpsi - m * psi * dpsim) / (psim * dxi - m * xi * dpsim)
    return a, b


@pytest.mark.parametrize("m, x", [(1.55, 5.213), (1.5 + 0.1j, 2.0), (AU / 1.33, 1.1)])
def test_homogeneous_sphere_equals_mie(m, x):
    nh = 1.33
    r = x * LAM / (2 * np.pi * nh)
    L = ps.l_max(r, nh, LAM, "far") + 5
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
    l = S.l
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
    ref = np.stack([np.full(lam.size, 1.45), ps.refractive_index("Au_JC", lam), np.ones(lam.size)], 1)
    S = ps.solve([50, 55], ref, [1, 1, 1], lam, 12)
    cs = ps.cross_sections(S)
    single = ps.cross_sections(ps.solve([50, 55], ref[40], [1, 1, 1], lam[40], 12))
    assert cs.q_ext.shape == (81,) and cs.q_ext[40] == pytest.approx(single.q_ext[0], rel=1e-13)


# ----------------------------------------------------------------- near field


def test_near_field_without_sphere_is_plane_wave():
    nh = 1.33
    S = _sol([30, 60], [nh] * 3, L=40)
    g = np.linspace(-120, 120, 13)
    X, Z = np.meshgrid(g, g)
    f = ps.near_field(S, X, 0 * X + 7.0, Z)
    k = 2 * np.pi * nh / LAM
    assert np.allclose(f.E["x"], np.exp(1j * k * Z), atol=1e-9)
    assert np.allclose(f.E["y"], 0, atol=1e-9) and np.allclose(f.E["z"], 0, atol=1e-9)
    assert np.allclose(f.H["y"], nh * np.exp(1j * k * Z), atol=1e-9)


def test_near_field_regular_on_both_poles_and_at_origin():
    S = _sol([40.0, 50.0], [1.45, AU, 1.33])
    z = np.array([-70.0, -70.0, -45.0, -45.0, 70.0, 70.0, 0.0, 1e-6])
    x = np.array([0.0, 1e-7, 0.0, 1e-7, 0.0, 1e-7, 0.0, 0.0])
    f = ps.near_field(S, x, 0 * x, z)
    mag = np.abs(f.E["x"][0::2])
    for c in "xyz":
        assert np.all(np.abs(f.E[c][0::2] - f.E[c][1::2]) < 1e-6 * mag)


def test_tangential_fields_continuous_across_interfaces():
    rad, ref, mu = [40.0, 50.0], [1.45, AU, 1.33], [1.0, 1.3, 1.0]
    S = _sol(rad, ref, mu)
    th, ph = 1.1, 0.4
    for i, r0 in enumerate(rad):
        rr = np.array([r0 * (1 - 1e-10), r0 * (1 + 1e-10)])
        X, Y, Z = rr * np.sin(th) * np.cos(ph), rr * np.sin(th) * np.sin(ph), rr * np.cos(th)
        f = ps.near_field(S, X, Y, Z)
        for c in ("th", "ph"):
            assert f.E[c][0] == pytest.approx(f.E[c][1], rel=1e-7)
            assert f.H[c][0] == pytest.approx(f.H[c][1], rel=1e-7)
        eps = np.array(ref) ** 2 / np.array(mu)
        assert eps[i] * f.E["r"][0] == pytest.approx(eps[i + 1] * f.E["r"][1], rel=1e-7)


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
        avgE = _sphere_average(lambda X, Y, Z: ps.near_field(S, X, Y, Z).intensity_E, r)
        avgH = _sphere_average(lambda X, Y, Z: ps.near_field(S, X, Y, Z).intensity_H, r)
        assert ed.I_e[0] == pytest.approx(avgE, rel=1e-9)
        assert ed.I_m[0] == pytest.approx(avgH, rel=1e-9)


def test_energy_density_without_sphere_is_one():
    ed = ps.energy_density(_sol([30, 60], [1.5] * 3, L=40), np.linspace(1, 200, 9))
    assert np.allclose(ed.w_e, 1) and np.allclose(ed.w_m, 1) and np.allclose(ed.I_e, 1)


@pytest.mark.parametrize(
    "ref",
    [
        [1.45, AU, 1.6, 0.25 + 3.5j, 1.33],
        [2.0, 1.45 + 0.02j, 1.5, 1.2, 1.0],
        [1.45 + 1e-7j, AU, 1.5 + 1e-5j, 1.2 + 2e-4j, 1.0],  # weakly lossy shells
    ],
)
def test_total_energy_lommel_equals_quadrature(ref):
    S = _sol([10.0, 13.0, 36.0, 48.0], ref, L=16)
    auto = ps.total_energy(S)
    quad = ps.total_energy(S, method="quadrature")
    assert np.allclose(auto.e, quad.e, rtol=1e-9) and np.allclose(auto.m, quad.m, rtol=1e-9)


def test_weak_loss_lommel_is_ill_conditioned_and_avoided():
    """STRATIFY switches Lommel formulas at Im(n) == 0 only (AUDIT.md M10); the
    lossy form then loses ~1e-17/Im(n) relative accuracy (2e-2 at Im n = 1e-14)."""
    ref = [1.5 + 1e-14j, 1.0]
    S = _sol([200.0], ref, L=30)
    good = ps.total_energy(S, method="quadrature").e[0]
    lossless = ps.total_energy(_sol([200.0], [1.5, 1.0], L=30)).e[0]
    naive = ps.total_energy(S, method="lommel").e[0]
    assert good == pytest.approx(lossless, rel=1e-6)
    assert ps.total_energy(S).e[0] == pytest.approx(good, rel=1e-12)
    assert abs(naive / good - 1) > 1e-3


def test_loudon_energy_prefactor():
    p = ps.DRUDE["Au_Ord"]
    lam = 600.0
    w, g, wp = 1 / lam, 1 / p["lam_gamma"], 1 / p["lam_p"]
    eps = 1 - wp**2 / (w**2 + 1j * g * w)
    assert ps.g_electric(eps, lam, p["lam_gamma"]) == pytest.approx(1 + wp**2 / (w**2 + g**2), rel=1e-12)


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
    r = ps.decay_rates([50, 70, 90], ref, mu, LAM, rd, dipole=dipole, norm="shell", tol=1e-9)
    assert r.converged.all()
    assert r.balance_error.max() < 1e-11


def test_decay_close_to_metal_needs_and_gets_high_orders():
    r = ps.decay_rates([50.0], [AU, 1.33], [1, 1], LAM, [50.5, 51.0], tol=1e-6)
    assert r.converged.all() and r.l_used.min() > 400
    assert r.balance_error.max() < 1e-10


def test_decay_free_space_is_one():
    r = ps.decay_rates([50, 70], [1.4, 1.4, 1.4], [1, 1, 1], LAM, [10, 60, 100])
    assert np.allclose(r.radiative, 1, atol=1e-12) and np.allclose(r.nonradiative, 0)
    assert np.allclose(r.total, 1, atol=1e-12)


def test_decay_host_vs_shell_normalisation():
    ref, mu = [1.45, AU, 1.6, 1.33], [1, 1, 1.2, 1]
    for dip, ratio in (("electric", 1.6 * 1.2 / 1.33), ("magnetic", 1.6 * (1.6**2 / 1.2) / 1.33**3)):
        h = ps.decay_rates([50, 70, 90], ref, mu, LAM, [80], dipole=dip, norm="host")
        s = ps.decay_rates([50, 70, 90], ref, mu, LAM, [80], dipole=dip, norm="shell")
        for a, b in ((h.radiative, s.radiative), (h.nonradiative, s.nonradiative), (h.total, s.total)):
            assert np.allclose(a / b, ratio)


def test_decay_rejects_bad_input():
    with pytest.raises(ValueError):
        ps.decay_rates([50, 70], [1.45, AU, 1.33], [1, 1, 1], LAM, [60])  # inside metal
    with pytest.raises(ValueError):
        ps.decay_rates([50, 70], [1.45, AU, 1.33 + 0.1j], [1, 1, 1], LAM, [100])  # lossy host
    with pytest.raises(ValueError):
        ps.decay_rates([50, 70], [1.45, AU, 1.33], [1, 1, 1], LAM, [0.0])
