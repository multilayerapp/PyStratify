"""PyStratify versus the original STRATIFY MATLAB code (run in GNU Octave).

Two kinds of test:

* ``test_same_*``  - the port reproduces MATLAB to round-off where the MATLAB
  is correct;
* ``test_deviation_*`` - pin down each MATLAB defect listed in AUDIT.md, and
  where possible confirm its root cause by patching that one line in a copy of
  the MATLAB code and showing the discrepancy disappears.

Skipped when Octave is not installed.
"""

import numpy as np
import pytest

import pystratify as ps
from pystratify.legacy import t_mat

from . import octave
from .octave import mat

pytestmark = [pytest.mark.octave, pytest.mark.skipif(not octave.available(), reason="GNU Octave not installed")]

LAM = 614e-9  # metres, as in the STRATIFY examples
AU = 0.27 + 2.93j
RAD3 = np.array([50, 70, 90]) * 1e-9


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def _setup(rad, ref, mu, l):
    return f"rad={mat(rad)}; ref={mat(ref)}; mu={mat(mu)}; lam={LAM:.17g}; l=1:{l};\nT=t_mat(rad,ref,mu,lam,l);\n"


# ------------------------------------------------------------ same results


def test_same_tmatrix_and_cross_sections():
    rad = np.array([10, 13, 36, 48]) * 1e-9
    ref = [1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33]
    o = octave.run(
        _setup(rad, ref, np.ones(5), 12)
        + "tm=T.tm; te=T.te; mm=T.mm; me=T.me;\n[sc,ab,ex]=crs_sec(rad,lam,1.33,l,T); q=[sc.csem ab.csem ex.csem];",
        ["tm", "te", "mm", "me", "q"],
    )
    T = t_mat(rad, ref, np.ones(5), LAM, np.arange(1, 13))
    for k in ("tm", "te", "mm", "me"):
        assert rel(getattr(T, k), o[k]) < 1e-11
    cs = ps.cross_sections(ps.solve(rad, ref, np.ones(5), LAM, 12))
    assert np.allclose([cs.sca[0], cs.abs[0], cs.ext[0]], o["q"], rtol=1e-11)


def test_same_near_field_energy_and_far_field():
    rad = np.array([10, 13, 36, 48]) * 1e-9
    ref = [1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33]
    g = np.linspace(-70, 70, 21) * 1e-9
    X, Z = np.meshgrid(g, g)
    Y = np.full_like(X, 0.3e-9)  # off the z axis (see test_deviation_near_field_pole)
    rw = np.linspace(1, 80, 30) * 1e-9
    th = np.linspace(0, np.pi - 0.01, 40)
    body = _setup(rad, ref, np.ones(5), 10) + (
        f"X=reshape({mat(X)},{X.shape[0]},{X.shape[1]}); Y=reshape({mat(Y)},{X.shape[0]},{X.shape[1]}); Z=reshape({mat(Z)},{X.shape[0]},{X.shape[1]});\n"
        "[E,H,I]=near_fld(rad,ref,mu,lam,l,T,X,Y,Z); Ex=E.x; Ey=E.y; Ez=E.z; Hx=H.x; Hy=H.y; Hz=H.z;\n"
        "G.e=real(ref.^2).'; G.m=ones(5,1);\n"
        f"[wr,Id]=nrg_dns({mat(rw)},rad,ref,mu,lam,G,l,T,true); we=wr.e; wm=wr.m; Ie=Id.e;\n"
        "w=nrg_tot(rad,ref,mu,lam,G,l,T,true); te=w.e; tm=w.m; w=nrg_tot(rad,ref,mu,lam,G,l,T,false); te2=w.e;\n"
        f"S=far_fld(l,T,{mat(th)}); par=S.par; per=S.per;"
    )
    o = octave.run(body, ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "we", "wm", "Ie", "te", "tm", "te2", "par", "per"])
    S = ps.solve(rad, ref, np.ones(5), LAM, 10)
    # Octave stores X column-major; rebuild the grid the same way
    Xo, Yo, Zo = (np.reshape(a.ravel(), X.shape, order="F") for a in (X, Y, Z))
    nf = ps.near_field(S, Xo, Yo, Zo)
    for c in "xyz":
        assert rel(nf.E[c], o["E" + c]) < 1e-11
        assert rel(nf.H[c], o["H" + c]) < 1e-11
    ed = ps.energy_density(S, rw)
    assert rel(ed.w_e, o["we"]) < 1e-11 and rel(ed.w_m, o["wm"]) < 1e-11 and rel(ed.I_e, o["Ie"]) < 1e-11
    W = ps.total_energy(S)
    assert rel(W.e, o["te"]) < 1e-10 and rel(W.m, o["tm"]) < 1e-10
    assert rel(ps.total_energy(S, normalize=False).e, o["te2"]) < 1e-10
    par, per = ps.scattering_amplitudes(S, th)
    assert rel(par[0], o["par"]) < 1e-11 and rel(per[0], o["per"]) < 1e-11


def test_same_free_path_and_l_conv():
    o = octave.run(
        f"n1=el_fr_pth(600e-9,{mat(AU)},[50e-9 55e-9],'Au_Ord'); n2=el_fr_pth(700e-9,{mat(0.05 + 4.2j)},20e-9,'Ag_Blb');"
        "L1=l_conv(90e-9,1.33,500e-9,'far'); L2=l_conv(90e-9,1.33,500e-9,'near'); L3=l_conv(900e-9,1.0,500e-9,'far');",
        ["n1", "n2", "L1", "L2", "L3"],
    )
    assert ps.free_path_correction(600, AU, [50, 55], "Au_Ord") == pytest.approx(o["n1"], rel=1e-13)
    assert ps.free_path_correction(700, 0.05 + 4.2j, [20], "Ag_Blb") == pytest.approx(o["n2"], rel=1e-13)
    for L, args in (
        (o["L1"], (90e-9, 1.33, 500e-9, "far")),
        (o["L2"], (90e-9, 1.33, 500e-9, "near")),
        (o["L3"], (900e-9, 1.0, 500e-9, "far")),
    ):
        assert ps.l_max(*args) == int(np.floor(L))


def test_same_decay_where_matlab_is_right():
    """Au *core* (B = 0, so M2 is masked), weak-loss metal (so M3 is inactive):
    MATLAB rates equal ours once the radiative normalisation label is swapped."""
    ref = [0.3 + 0.9j, 1.45, 1.6, 1.33]  # Im(n) < 1 keeps edcy.m's truncation heuristic off
    rd = np.array([60, 80, 95, 120]) * 1e-9
    body = _setup(RAD3, ref, np.ones(4), 40) + f"[gr,gnr]=edcy(rad,{mat(rd)},ref,mu,lam,l,20000,T,'host');"
    o = octave.run(body, ["gr", "gnr"])
    p = octave.run(body, ["gr", "gnr"], patch={"bessel/sbesselj.m": SBESSELJ_FIXED})
    r = ps.decay_rates(RAD3, ref, np.ones(4), LAM, rd, l_max=40, norm="shell", n_quad=400, warn=False)
    assert rel(r.radiative, o["gr"]) < 1e-12  # MATLAB 'host' == true shell normalisation (M1)
    conv = _nrad_host_to_shell(ref, rd)
    assert rel(r.nonradiative, p["gnr"] * conv) < 1e-7  # with M7 fixed: trapz accuracy
    assert np.max(np.abs(o["gnr"] * conv / r.nonradiative - 1)) > 1e-4  # M7 alone costs ~3e-4


# sbesselj.m with the z == 0 branch fixed (j_0(0) = 1, j_l(0) = 0 for l >= 1)
SBESSELJ_FIXED = """function j = sbesselj( nu, z )
if z == 0
    j = double(nu == 0);
else
    j = sqrt( pi./(2*z) ).*besselj( nu+0.5, z );
end
end
"""


def _nrad_host_to_shell(ref, rd):
    d = ps.locate_shell(RAD3, rd)
    return (np.real(np.asarray(ref)[-1]) / np.real(np.asarray(ref)[d - 1]))[:, None]


# -------------------------------------------------------------- deviations


def test_deviation_M1_radiative_normalisation_swapped():
    """Lossless sphere: rad == total.  MATLAB 'host' equals the true
    shell-normalised rate and MATLAB 'shell' equals the true host-normalised
    one - OSAC Eq. (29) / Moroz Eqs. (126), (129) swap N_rad^host, N_rad^shell."""
    ref = [2.5, 1.45, 1.8, 1.33]
    rd = np.array([20, 60, 80]) * 1e-9
    o = octave.run(
        _setup(RAD3, ref, np.ones(4), 40)
        + f"[gh,~]=edcy(rad,{mat(rd)},ref,mu,lam,l,50,T,'host'); [gs,~]=edcy(rad,{mat(rd)},ref,mu,lam,l,50,T,'shell');",
        ["gh", "gs"],
    )
    sh = ps.decay_rates(RAD3, ref, np.ones(4), LAM, rd, l_max=40, norm="shell")
    ho = ps.decay_rates(RAD3, ref, np.ones(4), LAM, rd, l_max=40, norm="host")
    assert rel(sh.total, sh.radiative) < 1e-8  # lossless: rad == total (LDOS)
    assert rel(o["gh"], sh.radiative) < 1e-12  # MATLAB 'host' is really 'shell'
    assert rel(o["gs"], ho.radiative) < 1e-12  # MATLAB 'shell' is really 'host'
    assert rel(o["gs"], sh.total) > 0.05


def test_deviation_M2_cylindrical_hankel_in_I_abs():
    """Absorbing *shell* (B != 0): MATLAB rad+nrad misses the total by up to 2x;
    replacing besselh/dbesselh by the spherical sbesselh/dsbesselh fixes it."""
    ref = [1.45, 0.3 + 0.9j, 1.6, 1.33]  # Im(n) < 1: keeps M3 out of the picture
    rd = np.array([40, 95, 120]) * 1e-9
    body = _setup(RAD3, ref, np.ones(4), 50) + f"[gr,gnr]=edcy(rad,{mat(rd)},ref,mu,lam,l,20000,T,'host');"
    o = octave.run(body, ["gr", "gnr"])
    src = (octave.STRATIFY / "decay" / "I_abs.m").read_text(encoding="latin-1")
    fixed = src.replace("besselh(l, xa(i))", "sbesselh(l, xa(i))").replace("dbesselh(l, xa(i))", "dsbesselh(l, xa(i))")
    assert fixed != src
    p = octave.run(body, ["gr", "gnr"], patch={"decay/I_abs.m": fixed})
    ours = ps.decay_rates(RAD3, ref, np.ones(4), LAM, rd, l_max=50, norm="shell")
    # compare in MATLAB's (mislabelled) 'host' = true shell normalisation; nrad 'host' -> shell
    conv = _nrad_host_to_shell(ref, rd)
    assert rel(o["gnr"] * conv, ours.nonradiative) > 0.2  # original: badly off
    assert rel(p["gnr"] * conv, ours.nonradiative) < 1e-6  # patched: agrees (trapz accuracy)
    assert rel(p["gr"], ours.radiative) < 1e-12


def test_deviation_M3_truncation_heuristic():
    """Im(n) > 1 switches edcy.m to an ad-hoc partial sum: Gamma_nrad too low
    for an emitter close to a metal surface; the `tol` argument is unused."""
    ref = [AU, 1.45, 1.6, 1.33]
    rd = np.array([52, 56]) * 1e-9
    body = _setup(RAD3, ref, np.ones(4), 150) + f"[gr,gnr]=edcy(rad,{mat(rd)},ref,mu,lam,l,4000,T,'host');"
    o = octave.run(body, ["gnr"])
    ours = ps.decay_rates(RAD3, ref, np.ones(4), LAM, rd, l_max=150, norm="shell", tol=1e-8)
    matlab = o["gnr"] * _nrad_host_to_shell(ref, rd)
    assert np.all(matlab < ours.nonradiative * 0.995)
    assert ours.balance_error.max() < 1e-7


def test_deviation_M4_magnetic_dipole_host_normalisation():
    """mdcy.m uses the electric-dipole Gamma_0 ratio for 'host'; a magnetic
    dipole's free-space rate scales as n*eps, not n*mu."""
    ref = [2.5, 1.45, 1.8, 1.33]
    rd = np.array([80, 120]) * 1e-9
    o = octave.run(
        _setup(RAD3, ref, np.ones(4), 40).replace("T=t_mat", "T0=t_mat")
        + f"[gh,~]=mdcy(rad,{mat(rd)},ref,mu,lam,l,50,T0,'host');",
        ["gh"],
    )
    sh = ps.decay_rates(RAD3, ref, np.ones(4), LAM, rd, l_max=40, norm="shell", dipole="magnetic")
    ho = ps.decay_rates(RAD3, ref, np.ones(4), LAM, rd, l_max=40, norm="host", dipole="magnetic")
    assert rel(o["gh"][1], ho.radiative[1]) < 1e-10  # emitter in host: no normalisation involved
    assert rel(o["gh"][0], sh.radiative[0]) < 1e-10  # like edcy: 'host' is really 'shell'
    assert rel(ho.radiative[0], sh.radiative[0] * 1.8**3 / 1.33**3) < 1e-12


def test_deviation_M5_energy_prefactor():
    """G_prefac.m: Re(eps_D) + 2 Im(eps_D) lam_g/lam_p, i.e. omega_p/gamma,
    instead of omega/gamma (OSAC Eq. 23; Loudon)."""
    lam = 600e-9
    # Octave has no MATLAB string arrays, so run the 'Au_Ord' case of G_prefac.m verbatim
    src = (octave.STRATIFY / "energy" / "G_prefac.m").read_text(encoding="latin-1")
    case = src.split("case 'Au_Ord'")[1].split("case '")[0].split("\n        case")[0]
    case = case.replace("G.e(i)", "ge").replace("G.m(i)", "gm")
    o = octave.run(f"lam={lam:.17g};\n{case}", ["ge"])
    p = ps.DRUDE["Au_Ord"]
    w, g, wp = 1 / 600.0, 1 / p["lam_gamma"], 1 / p["lam_p"]
    eps = 1 - wp**2 / (w**2 + 1j * g * w)
    loudon = 1 + wp**2 / (w**2 + g**2)
    assert ps.g_electric(eps, 600.0, p["lam_gamma"]) == pytest.approx(loudon, rel=1e-12)
    wrong = eps.real + 2 * eps.imag * p["lam_gamma"] / p["lam_p"]
    assert o["ge"] == pytest.approx(wrong, rel=1e-10)
    assert abs(o["ge"] / loudon - 1) > 0.5


def test_deviation_M6_theta_pi():
    """far_fld.m returns NaN at theta = pi; near_fld.m uses the theta = 0 limit
    of P_l^1/sin(theta) on the -z axis too (wrong sign for even l)."""
    ref = [1.45, AU, 1.33]
    rad = RAD3[:2]
    body = _setup(rad, ref, np.ones(3), 12) + (
        "S=far_fld(l,T,[0 pi/2 pi]); par=S.par;\n"
        "[E,~,~]=near_fld(rad,ref,mu,lam,l,T,[0;1e-12],[0;0],[-80e-9;-80e-9]); Ex=E.x;"
    )
    o = octave.run(body, ["par", "Ex"])
    assert np.isnan(o["par"][2])
    f = ps.near_field(ps.solve(rad, ref, np.ones(3), LAM, 12), [0, 1e-12], [0, 0], [-80e-9, -80e-9])
    assert f.E["x"][0] == pytest.approx(f.E["x"][1], rel=1e-8)
    assert o["Ex"][1] == pytest.approx(f.E["x"][1], rel=1e-8)  # 1e-12 m off axis: MATLAB fine
    assert abs(o["Ex"][0] - o["Ex"][1]) > 0.1 * abs(o["Ex"][1])  # on axis: MATLAB jumps


def test_deviation_M7_sbesselj_at_zero():
    o = octave.run("j=sbesselj(1:3,0);", ["j"])
    assert np.allclose(o["j"], [1, 0, 0])  # j_1(0) is 0, not 1
    from pystratify.legacy import sph_jn

    assert np.allclose(sph_jn(np.arange(1, 4), 0), 0)


def test_deviation_M9_high_l_cancellation():
    """Emitter outside the sphere: I_abs.m builds the absorbing-shell
    coefficients as M11 + M12 T21/T11, which cancels catastrophically at large
    l.  Gamma_nrad then drifts as l_max grows (the tst_dcy.m example uses
    l = 1:100); the regular-solution form T(a)(1,0)/T11 used here does not."""
    ref = [1.45, 0.3 + 0.9j, 1.6, 1.33]
    fixed = (octave.STRATIFY / "decay" / "I_abs.m").read_text(encoding="latin-1")
    fixed = fixed.replace("besselh(l, xa(i))", "sbesselh(l, xa(i))").replace(
        "dbesselh(l, xa(i))", "dsbesselh(l, xa(i))"
    )
    body = "".join(
        _setup(RAD3, ref, np.ones(4), L).replace("T=", f"T{L}=")
        + f"[~,g{L}]=edcy(rad,95e-9,ref,mu,lam,1:{L},4000,T{L},'host');\n"
        for L in (50, 70)
    )
    o = octave.run(body, ["g50", "g70"], patch={"decay/I_abs.m": fixed})
    ours = [ps.decay_rates(RAD3, ref, np.ones(4), LAM, [95e-9], l_max=L).nonradiative[0] for L in (50, 70)]
    assert rel(ours[1], ours[0]) < 1e-9  # converged and stable
    assert rel(o["g70"], o["g50"]) > 1e-4  # MATLAB drifts with l_max
