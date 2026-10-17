"""Dispersion models against pyElli's, an independent ellipsometry code (benchmarks/MATRIX.md, films).

    pip install -e ".[benchmarks]"
    python benchmarks/dispersion_codes.py        # prints a Markdown report (DISPERSION_CODES.md)

``pystratify.dispersion`` and pyElli's ``elli.dispersions`` / ``elli.materials`` were written
apart, so agreement checks both the formulas and the parameter conventions. The conventions
mapped here (pyElli -> PyStratify):

* pyElli's Lorentz, Sellmeier and resistivity Drude add a vacuum 1; PyStratify's do not, so
  1 is subtracted from pyElli's eps.
* LorentzEnergy A = PyStratify's A B E0; Gaussian sigma = PyStratify's B (both full widths at
  half maximum); DrudeEnergy A = Ep^2, gamma = Gamma.
* **pyElli's two Drude models have the sign of eps1 reversed** (0.23.1, and upstream master on
  2026-10-16): ``DrudeEnergy`` returns A / (E^2 - i gamma E), whose real part A / (E^2 + gamma^2)
  is positive at every energy, so a free-electron metal never turns metallic; its eps2 is the
  textbook one. ``DrudeResistivity`` has the same form. The free-electron result is
  eps = -Ep^2 / (E^2 + i Gamma E) (eps1 < 0 below the plasma energy), which is what PyStratify
  returns; the Drude rows therefore compare against -conj(eps_pyElli), and the eps2 column
  needs no such step.
* DrudeResistivity rho_opt is in ohm cm, PyStratify's rho in ohm m.
* Cauchy: pyElli's n1, n2 multiply 1e2 / lambda[nm]^2 and 1e7 / lambda[nm]^4, so
  B[um^2] = 1e-4 n1 and C[um^4] = 1e-5 n2.
* Bruggeman: pyElli takes the GUEST fraction; ``Bruggeman(a, b, f)`` here is f of a in b.

pyElli's Cody-Lorentz takes eps1 from a 1000-point Kramers-Kronig sum on a fixed 50-10000 nm grid,
so it is a coarse reference: its row measures pyElli's quadrature, not this module's (which is
checked against the closed forms in ``tests/test_dispersion.py`` to 1e-13).

Every value is max |eps - eps_pyElli| / max |eps_pyElli| over 200-1700 nm (0.73-6.2 eV),
400 wavelengths: scaled by the curve's largest value, because the poles' eps crosses zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pystratify import dispersion as dm  # noqa: E402

WAVELENGTHS_NM = np.linspace(200.0, 1700.0, 400)


def _elli():
    import elli

    return elli


def cases():
    """name -> (PyStratify model, a function returning pyElli's eps as PyStratify defines it)."""
    elli = _elli()
    au, si = dm.Constant(0.18, 3.43), dm.Constant(3.88, 0.02)

    def material(model):
        return elli.IsotropicMaterial(elli.Table(lbda=WAVELENGTHS_NM, n=model.index(WAVELENGTHS_NM)))

    def tensor(mat):
        return np.asarray(mat.get_tensor(WAVELENGTHS_NM))[..., 0, 0]

    return {
        "Tauc-Lorentz": (dm.TaucLorentz(A=120, E0=3.6, C=2.4, Eg=1.6),
                         lambda: elli.TaucLorentz(Eg=1.6).add(A=120, E=3.6, C=2.4).get_dielectric(WAVELENGTHS_NM)),
        "Lorentz": (dm.Lorentz(A=3.0, E0=4.5, B=0.8),
                    lambda: elli.LorentzEnergy().add(A=3.0 * 0.8 * 4.5, E=4.5, gamma=0.8)
                    .get_dielectric(WAVELENGTHS_NM) - 1),
        "Gaussian": (dm.Gaussian(A=2.0, E0=4.0, B=0.6),
                     lambda: elli.Gaussian().add(A=2.0, E=4.0, sigma=0.6).get_dielectric(WAVELENGTHS_NM)),
        "UV + IR poles": (dm.Pole(A=40.0, E0=9.0) + dm.Pole(A=0.3),
                          lambda: elli.Poles(A_ir=-0.3, A_uv=40.0, E_uv=9.0).get_dielectric(WAVELENGTHS_NM)),
        "Drude (energy)": (dm.DrudeEnergy(Ep=8.9, Gamma=0.07),
                           lambda: -np.conj(elli.DrudeEnergy(A=8.9**2, gamma=0.07).get_dielectric(WAVELENGTHS_NM))),
        "Drude (resistivity)": (dm.DrudeResistivity(rho=2.2e-8, tau=9e-15),
                                lambda: -np.conj(elli.DrudeResistivity(rho_opt=2.2e-6, tau=9e-15)
                                                 .get_dielectric(WAVELENGTHS_NM))),
        "Sellmeier (fused silica)": (dm.Sellmeier(B=(0.6961663, 0.4079426, 0.8974794),
                                                  C=(0.0684043**2, 0.1162414**2, 9.896161**2)),
                                     lambda: elli.Sellmeier().add(A=0.6961663, B=0.0684043**2)
                                     .add(A=0.4079426, B=0.1162414**2).add(A=0.8974794, B=9.896161**2)
                                     .get_dielectric(WAVELENGTHS_NM)),
        "Cauchy": (dm.Cauchy(A=1.45, B=0.0036, C=1e-4),
                   lambda: elli.Cauchy(n0=1.45, n1=36.0, n2=10.0).get_dielectric(WAVELENGTHS_NM)),
        "Bruggeman, 50% void in Si": (dm.Bruggeman(1.0, si, 0.5),
                                      lambda: tensor(elli.BruggemanEMA(material(si), material(dm.Constant(1.0)), 0.5))),
        "Bruggeman, 30% Au in air": (dm.Bruggeman(au, 1.0, 0.3),
                                     lambda: tensor(elli.BruggemanEMA(material(dm.Constant(1.0)), material(au), 0.3))),
        "Maxwell Garnett, 10% Au in glass": (dm.MaxwellGarnett(1.5, au, 0.1),
                                             lambda: tensor(elli.MaxwellGarnettEMA(material(dm.Constant(1.5)),
                                                                                   material(au), 0.1))),
        "Cody-Lorentz (pyElli's KK is a 1000-point sum)": (
            dm.CodyLorentz(A=100, E0=3.6, Gamma=2.4, Eg=1.6, Ep=0.8, Et=1.8, Eu=0.05),
            lambda: elli.CodyLorentz(Eg=1.6, A=100, Et=1.8, gamma=2.4, Ep=0.8, E0=3.6, Eu=0.05)
            .get_dielectric(WAVELENGTHS_NM)),
    }


def compare():
    worst = {}
    for name, (model, theirs) in cases().items():
        ours = model.epsilon(WAVELENGTHS_NM)
        reference = np.asarray(theirs(), dtype=complex)
        scale = np.max(np.abs(reference))
        worst[name] = (float(np.max(np.abs(ours - reference)) / scale),
                       float(np.max(np.abs(ours.imag - reference.imag)) / scale))
    return worst


def cauchy_weight_kk(eps2, energy, breaks):
    """(2/pi) P int_0^inf E' eps2 / (E'^2 - E^2) dE' by QUADPACK, split at the kinks, with the
    Cauchy weight on the segment holding E: a quadrature independent of ``dispersion.kramers_kronig_real``'s subtraction
    (tests/test_dispersion.py and benchmarks/figures.py call it)."""
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


def report(worst):
    print("# PyStratify dispersion models against pyElli\n")
    print("Generated by `python benchmarks/dispersion_codes.py`. max |Δε| / max |ε_pyElli| over")
    print(f"{WAVELENGTHS_NM[0]:g}–{WAVELENGTHS_NM[-1]:g} nm ({WAVELENGTHS_NM.size} wavelengths); parameter mappings")
    print("in the script's docstring. pyElli's Drude ε′ has the wrong sign and is compared reversed (docstring).\n")
    print("| model | ε | ε″ only |")
    print("|---|---|---|")
    for name, (total, imaginary) in worst.items():
        print(f"| {name} | {total:.1e} | {imaginary:.1e} |")


if __name__ == "__main__":
    report(compare())
