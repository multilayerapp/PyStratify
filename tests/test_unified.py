"""Independent identities and channel closure for the six common families.

The checks cited in benchmarks/MATRIX.md live in benchmarks/unified_checks.py, which prints every observed
value (UNIFIED_CHECKS.md); these tests assert its rows. The rest are implementation tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import pystratify as ps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import unified_checks as unified  # noqa: E402
from checks import assert_rows  # noqa: E402


def test_planar_fresnel_and_opaque():
    assert_rows(unified.fresnel_and_opaque())


def test_cylinder_against_bessel_boundary_equations():
    assert_rows(unified.cylinder_bessel_boundary())


@pytest.mark.parametrize('geometry',unified.GEOMETRIES)
@pytest.mark.parametrize('dipole',unified.DIPOLES)
def test_homogeneous_point_source(geometry,dipole):
    assert_rows(unified.homogeneous_point_source(geometry,dipole))


@pytest.mark.parametrize('geometry',unified.GEOMETRIES)
@pytest.mark.parametrize('dipole',unified.DIPOLES)
def test_absorbing_coating_independent_power_budget(geometry,dipole):
    assert_rows(unified.absorbing_coating_power_budget(geometry,dipole))


def test_guided_point_sources_not_line_sources():
    assert_rows(unified.guided_point_sources())


@pytest.mark.parametrize('tolerance',[1e-6,1e-8])
def test_planar_guided_pole_uncertainty_meets_tolerance(tolerance):
    assert_rows(unified.planar_guided_pole_uncertainty(tolerance))


def test_pattern_normalization_and_film_grazing_limit():
    assert_rows(unified.pattern_normalization())


def test_small_order_reports_failure():
    r=ps.solve_problem(ps.Problem('cylinders',[.1],[1,1],.6,ps.PointDipole(.09,'electric',0),1e-5,2))
    assert not r['diagnostics']['converged']


@pytest.mark.parametrize('radii,n,beta',unified.TREAMS_CYLINDERS)
def test_oblique_cylinder_against_treams(radii,n,beta):
    pytest.importorskip('treams')
    assert_rows(unified.oblique_cylinder_against_treams(radii,n,beta))


@pytest.mark.parametrize('beta',[0,3])
def test_cylinder_homogeneous_fields(beta):
    assert_rows(unified.cylinder_homogeneous_fields(beta))


@pytest.mark.parametrize('dipole',['electric','magnetic'])
@pytest.mark.parametrize('radius',[.03,.08,.2])
def test_sphere_local_pattern_power(dipole,radius):
    x,w=np.polynomial.legendre.leggauss(40)
    problem=ps.Problem('spheres',[.05,.1],[1.5,2,1],.6,ps.PointDipole(radius,dipole))
    r=ps.solve_problem(problem,('rates','pattern'),theta=np.arccos(x)[:,None],phi=np.array([0,np.pi/2])[None,:])
    power=np.sum(r['pattern']*w[:,None,None],axis=(0,1))*np.pi
    np.testing.assert_allclose(power,r['escape'],atol=1e-10)


@pytest.mark.parametrize('dipole',unified.DIPOLES)
def test_metal_cylinder_grazing_caps_have_a_bounded_error(dipole):
    assert_rows(unified.metal_cylinder_grazing_caps(dipole))


def test_cylinder_grazing_uncertainty_cannot_pass_a_tighter_tolerance():
    assert_rows(unified.cylinder_grazing_uncertainty_at_a_tighter_tolerance())


@pytest.mark.parametrize('wavelength',unified.CUTOFF_WAVELENGTHS)
def test_cylinder_finds_every_guided_mode_near_cutoff(wavelength):
    assert_rows(unified.fibre_modes_near_cutoff(wavelength))


def test_cylinder_rates_are_smooth_through_a_mode_cutoff():
    assert_rows(unified.rates_smooth_through_cutoff())


def test_cylinder_density_near_the_light_line_is_the_power_balance():
    assert_rows(unified.light_line_power_balance())


def test_cylinder_pattern_reuses_modes_for_azimuthal_cuts(monkeypatch):
    from pystratify.cylinder_emission import CylinderSource
    source=CylinderSource([.1],[1.5,1],.6,.2,m_max=20)
    theta=np.linspace(.1,np.pi-.1,13)[:,None]
    phi=np.array([0,np.pi/2])[None,:]
    separate=np.stack([source.pattern(theta[:,0],p) for p in phi[0]],axis=1)
    original=source.solution
    calls=[]
    def counted(b,*args):
        calls.append(b)
        return original(b,*args)
    monkeypatch.setattr(source,'solution',counted)
    np.testing.assert_allclose(source.pattern(theta,phi),separate,atol=1e-14)
    assert len(calls)==len(theta)


@pytest.mark.parametrize('z',[.01+0j,1+2j,20+10j])
def test_scaled_hankel_direct_and_overflow_recurrence(z):
    from pystratify.special import cylinder_logs
    from scipy.special import hankel1e
    regular,outgoing,dj,dh=cylinder_logs(z,500)
    assert np.all(np.isfinite(regular)) and np.all(np.isfinite(outgoing))
    np.testing.assert_allclose(np.exp(outgoing[:30]-1j*z),hankel1e(np.arange(30),z),rtol=2e-12)
    # The high orders overflow direct evaluation; logarithmic recurrence still holds.
    for m in [100,499]:
        expected=2*m/z-np.exp(outgoing[m-1]-outgoing[m])
        np.testing.assert_allclose(np.exp(outgoing[m+1]-outgoing[m]),expected,rtol=2e-10)


def test_opaque_coherent_segment_keeps_incoherent_recursions_finite():
    """A 10 um k=6 coherent layer inside an incoherent stack underflows to T=0.

    The upstream clip of Im(delta) at 35 kept T near e^-70; without a floor the
    two-flux L-matrix divided by zero and inc_tmm/pim_tmm reported R=T=nan.
    """
    import numpy as np
    from pystratify import planar

    n = [1, 1.5, 0.9 + 6j, 1.5, 1.52]
    d = [np.inf, 1000, 10, 0.3, np.inf]
    c = ["i", "i", "c", "c", "i"]
    for solver in (planar.inc_tmm, planar.pim_tmm):
        result = solver("s", n, d, c, np.deg2rad(10), 0.5)
        assert np.isfinite(result["R"]) and np.isfinite(result["T"])
        assert abs(result["R"] - 0.8722616) < 1e-6
        assert 0 <= result["T"] < 1e-29
