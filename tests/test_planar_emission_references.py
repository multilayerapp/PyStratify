"""Film emitters against an independent Sommerfeld-integral reference and the image-dipole closed form.

The reference shares no code with ``pystratify``: the decay rates of a dipole at height d above a
stack are the classical integrals over the in-plane wavenumber s = k_rho / k_1 (Chance, Prock &
Silbey, Adv. Chem. Phys. 37, 1 (1978); in the form of Novotny & Hecht, Principles of Nano-Optics,
ch. 10), with s_z = sqrt(1 - s^2), Im s_z >= 0:

    perpendicular  1 + (3/2) Re  int  s^3 / s_z  r_p                e^{2 i k_1 d s_z} ds
    parallel       1 + (3/4) Re  int  s   / s_z  (r_s - s_z^2 r_p)  e^{2 i k_1 d s_z} ds

r_p is the ratio of the magnetic fields (r_p = +1, r_s = -1 at a perfect electric conductor), and
the stack's r_s, r_p come from Parratt's recursion, not from ``pystratify.planar``. A magnetic
dipole in nonmagnetic media exchanges r_s and r_p. s = sin(theta) on the propagating part and
s = cosh(t) on the evanescent part remove the 1/s_z singularity.

The reference is checked first against the closed form at a perfect mirror, where it must reduce to
an image dipole at distance x = 2 k_1 d (Drexhage; Chance, Prock & Silbey):

    perpendicular  1 + 3   [sin x / x^3 - cos x / x^2]
    parallel       1 - 3/2 [sin x / x + cos x / x^2 - sin x / x^3]

Stacks with real guided poles (lossless guides) are left out: the plain quadrature cannot integrate
through them, which is what ``FilmSource`` does with its pole residues.
"""

import numpy as np
import pytest
from numpy import inf
from scipy.integrate import quad

import pystratify as ps

WAVELENGTH = 633.0
SILVER = 0.056 + 4.28j  # a silver-like index at 633 nm; any lossy metal serves


def parratt(n, d, s, k0):
    """r_s, r_p of the stack below the source medium n[0], at s = k_rho / (n[0] k0)."""
    eps = np.asarray(n, complex) ** 2
    kz = k0 * np.sqrt(eps - eps[0] * s ** 2 + 0j)
    kz = np.where(kz.imag < 0, -kz, kz)
    result = []
    for p in (False, True):
        def interface(i, j):
            a, b = (eps[j] * kz[i], eps[i] * kz[j]) if p else (kz[i], kz[j])
            return (a - b) / (a + b)

        r = interface(len(n) - 2, len(n) - 1)
        for j in range(len(n) - 3, -1, -1):
            phase = np.exp(2j * kz[j + 1] * d[j + 1])
            r = (interface(j, j + 1) + r * phase) / (1 + interface(j, j + 1) * r * phase)
        result.append(r)
    return result


def reference_rates(n, d, height, dipole="electric", mirror=False):
    """(perpendicular, parallel) rates of a dipole at ``height`` above the stack n[1:], d[1:]."""
    k0 = 2 * np.pi / WAVELENGTH
    x = 2 * n[0].real * k0 * height

    def coefficients(s):
        rs, rp = (-1.0, 1.0) if mirror else parratt(n, d, s, k0)
        return (rp, rs) if dipole == "magnetic" else (rs, rp)

    def propagating(theta, which):
        s, sz = np.sin(theta), np.cos(theta)
        rs, rp = coefficients(s)
        value = s ** 3 * rp if which == 0 else s * (rs - sz ** 2 * rp)
        return (value * np.exp(1j * x * sz)).real

    def evanescent(t, which):
        s, sinh = np.cosh(t), np.sinh(t)
        rs, rp = coefficients(s)
        value = s ** 3 * rp if which == 0 else s * (rs + sinh ** 2 * rp)
        return value.imag * np.exp(-x * sinh)

    top = np.arcsinh(60 / x)  # e^{-x sinh t} < 1e-26 beyond
    # a surface-plasmon quasi-pole on the first interface, if it lies in the evanescent range
    plasmon = abs(np.sqrt(n[1] ** 2 / (n[1] ** 2 + n[0] ** 2)))
    points = [np.arccosh(plasmon)] if not mirror and plasmon > 1 and np.arccosh(plasmon) < top else None
    rates = []
    for which, scale in ((0, 1.5), (1, 0.75)):
        inner = quad(propagating, 0, np.pi / 2, args=(which,), epsabs=1e-13, epsrel=1e-12, limit=400)[0]
        outer = quad(evanescent, 0, top, args=(which,), epsabs=1e-13, epsrel=1e-12, limit=800, points=points)[0]
        rates.append(1 + scale * (inner + outer))
    return np.array(rates)


def image_dipole(x):
    perpendicular = 1 + 3 * (np.sin(x) / x ** 3 - np.cos(x) / x ** 2)
    parallel = 1 - 1.5 * (np.sin(x) / x + np.cos(x) / x ** 2 - np.sin(x) / x ** 3)
    return np.array([perpendicular, parallel])


@pytest.mark.parametrize("x", [0.3, 1.0, 2.5, 7.0, 20.0])
def test_reference_reduces_to_the_image_dipole_at_a_perfect_mirror(x):
    height = x * WAVELENGTH / (4 * np.pi)
    assert np.allclose(reference_rates([1.0, 1.0], [inf, inf], height, mirror=True), image_dipole(x),
                       rtol=1e-10, atol=1e-12)
    # a magnetic dipole sees the mirror with the opposite sign
    swapped = reference_rates([1.0, 1.0], [inf, inf], height, dipole="magnetic", mirror=True)
    assert np.allclose(swapped, 2 - image_dipole(x), rtol=1e-10, atol=1e-12)


STACKS = {
    "air above glass": ([1.0, 1.52], [inf, inf]),
    "glass above air (total internal reflection)": ([1.52, 1.0], [inf, inf]),
    "air above 60 nm silver on glass": ([1.0, SILVER, 1.52], [inf, 60.0, inf]),
    "air above an absorbing 100 nm film on glass": ([1.0, 2.0 + 0.3j, 1.52], [inf, 100.0, inf]),
    "air above silver, a lossy film and glass": ([1.0, SILVER, 1.8 + 0.05j, 1.52], [inf, 30.0, 80.0, inf]),
}


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("height", [10.0, 50.0, 300.0])
@pytest.mark.parametrize("name", list(STACKS))
def test_film_emitter_rates_against_the_sommerfeld_reference(name, height, dipole):
    n, d = STACKS[name]
    problem = ps.Problem("films", d, n, WAVELENGTH, ps.PointDipole(-height, dipole, layer=0), tolerance=1e-10)
    out = ps.solve_problem(problem)
    assert out["diagnostics"]["converged"]
    reference = reference_rates(n, d, height, dipole)
    assert np.allclose(out["total"][:2], reference, rtol=1e-10, atol=0), (out["total"][:2], reference)
