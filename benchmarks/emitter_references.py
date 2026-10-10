"""Film emitters against an independent Sommerfeld reference, the image dipole and smuthi (benchmarks/MATRIX.md).

    python benchmarks/emitter_references.py        # prints a Markdown report (EMITTER_REFERENCES.md)

**Sommerfeld reference.** Written without ``pystratify``: the decay rates of a dipole at height d above
a stack are the classical integrals over s = k_rho / k_1 (Chance, Prock & Silbey, Adv. Chem. Phys. 37,
1 (1978); in the form of Novotny & Hecht, Principles of Nano-Optics, ch. 10), s_z = sqrt(1 - s^2):

    perpendicular  1 + (3/2) Re  int  s^3 / s_z  r_p                e^{2 i k_1 d s_z} ds
    parallel       1 + (3/4) Re  int  s   / s_z  (r_s - s_z^2 r_p)  e^{2 i k_1 d s_z} ds

r_p is the ratio of the magnetic fields (+1 at a perfect electric conductor, where r_s = -1); the
stack's r_s, r_p come from Parratt's recursion. A magnetic dipole in nonmagnetic media exchanges r_s
and r_p. s = sin(theta) and s = cosh(t) remove the 1/s_z singularity.

**Image dipole.** At a perfect mirror the reference must reduce to an image dipole at x = 2 k_1 d
(Drexhage; Chance, Prock & Silbey): perpendicular 1 + 3 [sin x / x^3 - cos x / x^2], parallel
1 - 3/2 [sin x / x + cos x / x^2 - sin x / x^3].

**smuthi** (Egel et al., T-matrix code with its own Sommerfeld contour): ``DipoleSource.dissipated_power``
over the homogeneous-medium power. It integrates through lossless guides, where the plain quadrature
above cannot: their poles lie on the real axis. Optional; it builds only on Python <= 3.10 with
NumPy < 2 (``numpy.distutils``), a Fortran compiler and an OpenMP-capable C compiler, e.g. on macOS
``CC=gcc-16 LDSHARED="gcc-16 -bundle -undefined dynamic_lookup" pip install --no-build-isolation smuthi``.
Its contour resolution sets its accuracy (~1e-6 here); its field chunks are sized from a memory
budget, so a very fine resolution or a source closer than ~30 nm can exhaust it.

**Guided-mode continuity.** A lossless guide is the limit of a weakly absorbing one, so the rates of a
source outside the guide must be continuous as the core's k -> 0. Before 0.10.3 they were not: from an
exterior half-space ``FilmSource.guided_poles`` found no poles and the guided power was dropped.
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
from numpy import inf
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402

WAVELENGTH = 633.0
SILVER = 0.056 + 4.28j  # a silver-like index at 633 nm; any lossy metal serves

# name -> (indices from the source's half-space down, thicknesses in nm); no lossless guide
STACKS = {
    "air above glass": ([1.0, 1.52], [inf, inf]),
    "glass above air (total internal reflection)": ([1.52, 1.0], [inf, inf]),
    "air above 60 nm silver on glass": ([1.0, SILVER, 1.52], [inf, 60.0, inf]),
    "air above an absorbing 100 nm film on glass": ([1.0, 2.0 + 0.3j, 1.52], [inf, 100.0, inf]),
    "air above silver, a lossy film and glass": ([1.0, SILVER, 1.8 + 0.05j, 1.52], [inf, 30.0, 80.0, inf]),
}
HEIGHTS = (10.0, 50.0, 300.0)

# lossless guides: two TE/TM modes in 200 nm TiO2, a single-mode SiN film
GUIDES = {
    "air above 200 nm TiO2 on glass": ([1.0, 2.3, 1.52], [inf, 200.0, inf]),
    "air above 150 nm SiN on SiO2": ([1.0, 1.99, 1.444], [inf, 150.0, inf]),
}
GUIDE_HEIGHTS = (30.0, 100.0)


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


def reference_rates(n, d, height, dipole="electric", mirror=False, poles=()):
    """(perpendicular, parallel) rates of a dipole at ``height`` above the stack n[1:], d[1:].

    ``poles`` (effective indices) become quadrature breakpoints; they must be off the real axis.
    """
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
    breaks = [np.arccosh(p / n[0].real) for p in poles if p > n[0].real]
    plasmon = abs(np.sqrt(n[1] ** 2 / (n[1] ** 2 + n[0] ** 2)))  # a surface plasmon on the first interface
    if not mirror and plasmon > 1:
        breaks.append(np.arccosh(plasmon))
    edges = [0.0, *sorted(b for b in breaks if b < top), top]
    rates = []
    for which, scale in ((0, 1.5), (1, 0.75)):
        inner = quad(propagating, 0, np.pi / 2, args=(which,), epsabs=1e-13, epsrel=1e-12, limit=400)[0]
        outer = sum(quad(evanescent, a, b, args=(which,), epsabs=1e-13, epsrel=1e-12, limit=800)[0]
                    for a, b in zip(edges[:-1], edges[1:]))
        rates.append(1 + scale * (inner + outer))
    return np.array(rates)


def image_dipole(x):
    perpendicular = 1 + 3 * (np.sin(x) / x ** 3 - np.cos(x) / x ** 2)
    parallel = 1 - 1.5 * (np.sin(x) / x + np.cos(x) / x ** 2 - np.sin(x) / x ** 3)
    return np.array([perpendicular, parallel])


def stratify_rates(n, d, height, dipole="electric", below=False, tolerance=1e-10):
    """PyStratify's (perpendicular, parallel) totals; ``below`` puts the source in the last half-space.

    Next to a lossless guide's poles the integration reaches ~1e-8, so guides use ``tolerance=1e-8``.
    """
    layer, position = (len(n) - 1, height) if below else (0, -height)
    problem = ps.Problem("films", d, n, WAVELENGTH, ps.PointDipole(position, dipole, layer=layer), tolerance=tolerance)
    out = ps.solve_problem(problem)
    if not out["diagnostics"]["converged"]:
        raise ArithmeticError(f"PyStratify did not converge: {out['diagnostics']}")
    return out["total"][:2]


def smuthi_rates(n, d, height, resolution=2e-3):
    """smuthi's (perpendicular, parallel) rates, source in n[0] at ``height`` above the stack."""
    stdout = sys.stdout
    with contextlib.redirect_stdout(io.StringIO()):
        import smuthi.fields as fields
        import smuthi.initial_field as initial
        import smuthi.layers as layers
    sys.stdout = stdout  # importing smuthi replaces the stream it was given
    bottom_up = list(n[::-1])
    thicknesses = [0.0] + list(d[1:-1][::-1]) + [0.0]
    system = layers.LayerSystem(thicknesses=thicknesses, refractive_indices=bottom_up)
    x = 2 * 2 * np.pi / WAVELENGTH * n[0].real * height
    neff_max = min(40.0, n[0].real * np.sqrt(1 + (40 / x) ** 2) + max(abs(np.asarray(n))))
    kpar = fields.reasonable_Sommerfeld_kpar_contour(WAVELENGTH, layer_refractive_indices=bottom_up,
                                                     neff_resolution=resolution, neff_max=neff_max)
    rates = []
    for moment in ([0, 0, 1], [1, 0, 0]):
        # a dipole couples to azimuthal orders |m| <= 1, so 24 angles integrate exactly
        source = initial.DipoleSource(WAVELENGTH, moment, [0, 0, sum(thicknesses) + height], k_parallel_array=kpar,
                                      azimuthal_angles_array=np.linspace(0, 2 * np.pi, 25))
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            power = float(np.squeeze(source.dissipated_power([], system, show_progress=False)))
            free = float(np.squeeze(source.dissipated_power_homogeneous_background(system)))
        rates.append(power / free)
    return np.array(rates)


def guide_with_loss(n, k):
    return [n[0]] + [v + 1j * k for v in n[1:-1]] + [n[-1]]


def compare(smuthi=True):
    """Worst relative difference per (case, reference)."""
    worst = {}

    def record(name, reference, value):
        value = float(value) if np.isfinite(value) else np.inf
        worst[(name, reference)] = max(worst.get((name, reference), 0.0), value)

    for x in (0.3, 1.0, 2.5, 7.0, 20.0):
        height = x * WAVELENGTH / (4 * np.pi)
        record("perfect mirror, electric", "image dipole",
               np.max(np.abs(reference_rates([1.0, 1.0], [inf, inf], height, mirror=True) - image_dipole(x))))
        record("perfect mirror, magnetic", "image dipole",
               np.max(np.abs(reference_rates([1.0, 1.0], [inf, inf], height, "magnetic", mirror=True)
                             - (2 - image_dipole(x)))))
    for name, (n, d) in STACKS.items():
        for height in HEIGHTS:
            for dipole in ("electric", "magnetic"):
                ours, theirs = stratify_rates(n, d, height, dipole), reference_rates(n, d, height, dipole)
                record(name, "Sommerfeld", np.max(np.abs(ours / theirs - 1)))
    for name, (n, d) in GUIDES.items():
        for height in GUIDE_HEIGHTS:
            for dipole in ("electric", "magnetic"):
                for below in (False, True):
                    lossless = stratify_rates(n, d, height, dipole, below, 1e-8)
                    absorbing = stratify_rates(guide_with_loss(n, 1e-7), d, height, dipole, below, 1e-8)
                    record(name, "continuity, core k = 1e-7", np.max(np.abs(lossless / absorbing - 1)))
            if smuthi:
                ours = stratify_rates(n, d, height, tolerance=1e-8)
                record(name, "smuthi", np.max(np.abs(ours / smuthi_rates(n, d, height) - 1)))
    return worst


def report(worst):
    print("# PyStratify film emitters against independent references\n")
    print("Generated by `python benchmarks/emitter_references.py`. Worst relative difference of the")
    print("perpendicular and parallel total rates over electric and magnetic dipoles and heights "
          f"{', '.join(f'{h:g}' for h in HEIGHTS)} nm")
    print(f"(guides: {', '.join(f'{h:g}' for h in GUIDE_HEIGHTS)} nm, sources above and below; smuthi: electric,")
    print("above). The image-dipole rows check the Sommerfeld reference itself (absolute difference).\n")
    references = sorted({reference for _, reference in worst})
    names = list(dict.fromkeys(name for name, _ in worst))
    print("| case | " + " | ".join(references) + " |")
    print("|---|" + "---|" * len(references))
    for name in names:
        cells = [f"{worst[(name, r)]:.1e}" if (name, r) in worst else "-" for r in references]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import smuthi  # noqa: F401
        available = True
    except ImportError:
        available = False
    sys.stdout = stdout
    if not available:
        print("<!-- smuthi not installed: its column is omitted -->")
    report(compare(smuthi=available))
