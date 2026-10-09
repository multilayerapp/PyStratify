"""Independent identities and channel closure for the six common families."""
import numpy as np
import pytest
from scipy.special import jv, jvp, hankel1, h1vp
import pystratify as ps
from pystratify.cylindrical import solve_cylinder, cross_widths, cylinder_pattern
from pystratify.planar import coh_tmm, position_resolved


def test_planar_fresnel_and_opaque():
    for pol in ('s','p'):
        s=coh_tmm(pol,[1,1.5],[np.inf,np.inf],0,.6)
        assert s['R']==pytest.approx(.04,abs=1e-14)
        assert s['T']==pytest.approx(.96,abs=1e-14)
        opaque=coh_tmm(pol,[1,.2+3j,1],[np.inf,20,np.inf],0,.6)
        assert np.isfinite(opaque['R']) and opaque['T']==0
        assert all(np.isfinite(position_resolved(1,z,opaque)['poyn']) for z in (0,10,20))
        assert opaque['R']==pytest.approx(abs((1-(.2+3j))/(1+(.2+3j)))**2,abs=1e-14)


def test_cylinder_against_bessel_boundary_equations():
    radius,lam,n=.1,.6,1.5
    s=solve_cylinder([radius],[n,1],lam,m_max=12)
    x=2*np.pi*radius/lam
    for index,m in enumerate(s.orders):
        for pol,eta in [(0,n),(1,1/n)]:
            want=-(jvp(m,x)*jv(m,n*x)-eta*jv(m,x)*jvp(m,n*x))/(h1vp(m,x)*jv(m,n*x)-eta*hankel1(m,x)*jvp(m,n*x))
            assert s.t[index,pol,pol]==pytest.approx(want,abs=2e-14)
    result=cross_widths(s)
    assert result['extinction']==pytest.approx(.129405795136107,abs=2e-14)
    assert result['extinction']==pytest.approx(result['scattering'],abs=2e-14)
    phi=np.linspace(0,2*np.pi,2001)
    assert np.trapezoid(cylinder_pattern(s,phi),phi)==pytest.approx(result['scattering'],abs=1e-12)


@pytest.mark.parametrize('geometry',['films','cylinders','spheres'])
@pytest.mark.parametrize('dipole',['electric','magnetic'])
def test_homogeneous_point_source(geometry,dipole):
    dimensions=[np.inf,.2,np.inf] if geometry=='films' else [.1]
    n=[1.5]*len(dimensions) if geometry=='films' else [1.5,1.5]
    source=ps.PointDipole(.1 if geometry=='films' else 0,dipole,1 if geometry=='films' else 0,.4)
    result=ps.solve_problem(ps.Problem(geometry,dimensions,n,.6,source,1e-6,15 if geometry=='cylinders' else None))
    assert result['diagnostics']['converged']
    np.testing.assert_allclose(result['total'],1,atol=1e-8)
    np.testing.assert_allclose(result['escape'],1,atol=1e-8)
    np.testing.assert_allclose(result['eta_escape'],.4,atol=1e-8)
    np.testing.assert_allclose(result['guided'],0,atol=1e-8)
    np.testing.assert_allclose(result['absorbed'],0,atol=1e-8)


@pytest.mark.parametrize('geometry',['films','cylinders','spheres'])
@pytest.mark.parametrize('dipole',['electric','magnetic'])
def test_absorbing_coating_independent_power_budget(geometry,dipole):
    dims=[np.inf,.1,np.inf] if geometry=='films' else [.05,.1]
    n=[1,.2+3j,1] if geometry=='films' else [1.5,.2+3j,1]
    source=ps.PointDipole(-.1 if geometry=='films' else .2,dipole,0 if geometry=='films' else 2)
    result=ps.solve_problem(ps.Problem(geometry,dims,n,.6,source,1e-5,30 if geometry=='cylinders' else None))
    assert result['diagnostics']['converged']
    assert np.min(result['absorbed'])>=-1e-8
    np.testing.assert_allclose(result['total'],result['escape']+result['guided']+result['absorbed'],rtol=1e-5,atol=1e-6)


def test_guided_point_sources_not_line_sources():
    for g,d,n,pos,layer in [('films',[np.inf,.2,np.inf],[1,1.5,1],.1,1),('cylinders',[.1],[1.5,1],.03,0)]:
        r=ps.solve_problem(ps.Problem(g,d,n,.6,ps.PointDipole(pos,'electric',layer),1e-5,20 if g=='cylinders' else None))
        assert r['diagnostics']['converged']
        assert np.max(r['guided'])>.1
        assert len(r['diagnostics']['poles'])>0
        np.testing.assert_allclose(r['total'],r['escape']+r['guided'],atol=1e-6)


@pytest.mark.parametrize('tolerance',[1e-6,1e-8])
def test_planar_guided_pole_uncertainty_meets_tolerance(tolerance):
    from pystratify.planar_emission import FilmSource
    source=FilmSource([1,3.5,1],[np.inf,.8,np.inf],.6,1,.4)
    rates=source.rates(tolerance)
    assert len(rates.poles)==18
    if tolerance==1e-6:assert rates.converged
    assert not rates.converged or rates.error <= tolerance*max(1,np.max(abs(rates.total)))
    np.testing.assert_allclose(rates.total,rates.escape+rates.guided,atol=tolerance)


def test_pattern_normalization_and_film_grazing_limit():
    x,w=np.polynomial.legendre.leggauss(48)
    theta=np.arccos(x)[:,None]; phi=np.linspace(0,2*np.pi,64,endpoint=False)[None,:]
    for g,d,n,pos,layer in [('films',[np.inf,.2,np.inf],[1,1,1],.1,1),('cylinders',[.1],[1,1],.03,0),('spheres',[.1],[1,1],0,0)]:
        r=ps.solve_problem(ps.Problem(g,d,n,.6,ps.PointDipole(pos,'electric',layer),1e-6,15 if g=='cylinders' else None),('rates','pattern'),theta=theta,phi=phi)
        power=np.sum(r['pattern']*w[:,None,None],axis=(0,1))*2*np.pi/phi.size
        np.testing.assert_allclose(power,r['escape'],atol=1e-7)
    r=ps.solve_problem(ps.Problem('films',[np.inf,.2,np.inf],[1,1,1],.6,ps.PointDipole(.1,'electric',1)),('rates','pattern'),theta=np.array([np.pi/2]),phi=0)
    assert r['pattern'][0,0]==pytest.approx(3/(8*np.pi),rel=1e-3)


def test_small_order_reports_failure():
    r=ps.solve_problem(ps.Problem('cylinders',[.1],[1,1],.6,ps.PointDipole(.09,'electric',0),1e-5,2))
    assert not r['diagnostics']['converged']


@pytest.mark.parametrize('radii,n,beta',[([.1],[1.5,1],0),([.05,.1],[1.5,.2+3j,1],2),([.05,.1],[1.5,2,1],8)])
def test_oblique_cylinder_against_treams(radii,n,beta):
    treams=pytest.importorskip('treams')
    tm=treams.TMatrixC.cylinder([beta],12,2*np.pi/.6,radii,[treams.Material(v*v) for v in n])
    sol=solve_cylinder(radii,n,.6,beta=beta,m_max=12)
    own=cross_widths(sol)
    assert own['extinction']==pytest.approx(tm.xw_ext_avg,abs=1e-12)
    assert own['scattering']==pytest.approx(tm.xw_sca_avg,abs=1e-12)
    phi=np.linspace(0,2*np.pi,2001)
    assert np.trapezoid(cylinder_pattern(sol,phi),phi)==pytest.approx(own['scattering'],abs=1e-12)


@pytest.mark.parametrize('beta',[0,3])
def test_cylinder_homogeneous_fields(beta):
    n=1.5;lam=.6;k=2*np.pi*n/lam;q=np.sqrt(k*k-beta*beta)
    s=solve_cylinder([.1],[n,n],lam,beta=beta,m_max=24)
    points=np.array([[0,0,0],[.02,.03,.01],[.18,-.02,.04]])
    phase=np.exp(1j*(q*points[:,0]+beta*points[:,2]))[:,None]
    for pol,direction in [('axial-electric',[-beta/k,0,q/k]),('axial-magnetic',[0,1,0])]:
        E,H=s.field(points,pol)
        np.testing.assert_allclose(E,phase*np.array(direction),atol=1e-10)
        np.testing.assert_allclose(np.sum(abs(E)**2,axis=-1),1,atol=1e-10)
        np.testing.assert_allclose(np.sum(abs(H)**2,axis=-1),n*n,atol=1e-10)


@pytest.mark.parametrize('dipole',['electric','magnetic'])
@pytest.mark.parametrize('radius',[.03,.08,.2])
def test_sphere_local_pattern_power(dipole,radius):
    x,w=np.polynomial.legendre.leggauss(40)
    problem=ps.Problem('spheres',[.05,.1],[1.5,2,1],.6,ps.PointDipole(radius,dipole))
    r=ps.solve_problem(problem,('rates','pattern'),theta=np.arccos(x)[:,None],phi=np.array([0,np.pi/2])[None,:])
    power=np.sum(r['pattern']*w[:,None,None],axis=(0,1))*np.pi
    np.testing.assert_allclose(power,r['escape'],atol=1e-10)


@pytest.mark.parametrize('dipole',['electric','magnetic'])
def test_metal_cylinder_grazing_caps_have_a_bounded_error(dipole):
    from pystratify.cylinder_emission import CylinderSource
    source = CylinderSource([.01],[.97+1.87j,1],.5,.015,dipole)
    rates = source.rates()
    assert rates.converged
    assert rates.evaluations < 2500
    assert rates.grazing_error > 0
    assert rates.error >= rates.grazing_error
    assert rates.error <= 1e-6 * np.max(rates.total)
    np.testing.assert_allclose(rates.total,rates.escape+rates.absorbed,rtol=1e-6)
    # Moving the precision guard inward checks its charged error independently.
    reference = CylinderSource([.01],[.97+1.87j,1],.5,.015,dipole,m_max=rates.orders)
    original = reference.spectral
    reference.spectral = lambda b, *args, **kwargs: original(1+np.copysign(3e-6,b-1) if abs(b-1)<3e-6 else b,*args,**kwargs)
    from pystratify.integration import integrate
    def sample(b):
        green,escape,absorbed,tail = reference.spectral(b)
        return np.r_[green.real,escape,absorbed,tail]
    integral = integrate(sample,[0,.97,1,reference.maximum],1e-6)
    assert integral.converged
    for actual,expected in [(rates.total,1+integral.value[:3]),(rates.escape,integral.value[3:6]),(rates.absorbed,integral.value[6:9])]:
        assert np.max(abs(actual-expected)) < rates.grazing_error


def test_cylinder_grazing_uncertainty_cannot_pass_a_tighter_tolerance():
    from pystratify.cylinder_emission import CylinderSource
    rates=CylinderSource([.01],[.97+1.87j,1],.5,.015,m_max=40).rates(tolerance=1e-9)
    assert rates.error >= rates.grazing_error > 1e-9*np.max(rates.total)
    assert not rates.converged


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
