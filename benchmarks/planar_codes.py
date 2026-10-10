"""Planar films against two independent codes: PyMoosh and pyElli (benchmarks/MATRIX.md, films).

    pip install -e ".[benchmarks]"
    python benchmarks/planar_codes.py            # prints a Markdown report
    python benchmarks/planar_codes.py --quick    # one wavelength, two angles

``pystratify.planar`` descends from Byrnes's ``tmm``, so agreeing with ``tmm`` proves nothing
about the formulation. These two codes do not share that ancestry:

* PyMoosh (Langevin, Moreau et al.): a scattering-matrix code, unconditionally stable,
  with its own incoherent recursion (``PyMoosh.incoherent``), field solver and guided-mode
  search (``PyMoosh.modes``);
* pyElli (Müller, Dobener et al.): ellipsometry with a 2x2 and a Berreman 4x4 solver.

Conventions found while writing this, all three codes using n + ik for absorption:

* PyMoosh's p-polarised ``t`` is the ratio of the magnetic fields; ``tmm``'s is the ratio of the
  electric fields. They differ by (n_in / mu_in) / (n_out / mu_out), which is applied here.
* pyElli defines rho = tan(Psi) exp(-i Delta) with r_p of the opposite sign, so its
  Delta = 180 deg - Delta(tmm), modulo 360 deg. Psi needs no conversion.
* PyMoosh's ``field`` expands a Gaussian beam in 2 nmod + 1 Fourier orders, nmod = floor(0.8366
  width / waist). A waist wider than the window leaves one order: a single plane wave, compared
  exactly. With a zero-thickness superstrate its phase origin is the first interface, as in tmm.
  It returns E_y (s) and H_y (p); H_y = (n_j/mu_j)/(n_0/mu_0) (v e^{i kz z} + w e^{-i kz z}) from
  tmm's amplitudes.

Guided modes of lossless stacks (the poles behind the guided channel of a film's point source,
``FilmSource.guided_poles``) are compared with ``PyMoosh.modes.guided_modes`` and with the textbook
three-layer dispersion relation. PyMoosh's point source (``PyMoosh.green``) is a 2D TE line source
on a periodic window, not a 3D dipole, so it is no reference for PyStratify's film emitters.

Every quantity is compared as an absolute difference; Delta in degrees.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from numpy import inf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pystratify import planar  # noqa: E402

# name -> (layers, thicknesses in nm, incoherent?) ; a layer is n or (n, mu)
CASES = {
    "AR pair on glass": ([1.0, 2.35, 1.46, 1.52], [inf, 64.0, 105.0, inf], False),
    "Bragg mirror, 10 pairs": ([1.0] + [2.35, 1.46] * 10 + [1.52], [inf] + [67.3, 108.4] * 10 + [inf], False),
    "Au 20 nm on glass": ([1.0, 0.27 + 2.93j, 1.52], [inf, 20.0, inf], False),
    "opaque Au 500 nm": ([1.0, 0.27 + 2.93j, 1.52], [inf, 500.0, inf], False),
    "absorber 20 um on Si": ([1.0, 3.5 + 0.4j, 3.9 + 0.02j], [inf, 20000.0, inf], False),
    "glass to air beyond critical": ([1.52, 1.0], [inf, inf], False),
    "frustrated TIR, 200 nm gap": ([1.52, 1.0, 1.52], [inf, 200.0, inf], False),
    "magnetic layer (n 1.5, mu 2)": ([1.0, (1.5, 2.0), 1.52], [inf, 150.0, inf], False),
    "TiO2 on 1 mm incoherent glass": ([1.0, 2.3 + 0.01j, 1.52 + 1e-6j, 1.0], [inf, 80.0, 1e6, inf], True),
}
ANGLES_DEG = (0.0, 30.0, 60.0, 85.0)
WAVELENGTHS_NM = (400.0, 633.0, 1550.0)


def _split(layers):
    n = [complex(x[0]) if isinstance(x, tuple) else complex(x) for x in layers]
    mu = [complex(x[1]) if isinstance(x, tuple) else 1.0 for x in layers]
    return n, mu


def stratify(case, wavelength, theta, pol):
    layers, d, incoherent = case
    n, mu = _split(layers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if incoherent:
            out = planar.inc_tmm(pol, n, d, ["i", "c", "i", "i"], theta, wavelength)
            return dict(R=out["R"], T=out["T"])
        magnetic = any(m != 1 for m in mu)
        out = planar.coh_tmm(pol, n, d, theta, wavelength, mu_list=mu) if magnetic else \
            planar.coh_tmm(pol, n, d, theta, wavelength)
        absorbed = np.asarray(planar.absorp_in_each_layer(out))[1:-1]
    return dict(R=out["R"], T=out["T"], r=out["r"], t=out["t"], A=absorbed)


def moosh(case, wavelength, theta, pol):
    import PyMoosh as pm
    from PyMoosh.core import absorption_S
    from PyMoosh.incoherent import incoherent_coefficient_S

    layers, d, incoherent = case
    n, mu = _split(layers)
    materials = [[n[i] ** 2 / mu[i], mu[i]] if mu[i] != 1 else n[i] ** 2 for i in range(len(n))]
    structure = pm.Structure(materials, list(range(len(n))), [0.0] + list(d[1:-1]) + [0.0], verbose=False)
    polarization = 0 if pol == "s" else 1
    if incoherent:
        R, T = incoherent_coefficient_S(structure, True, wavelength, theta, polarization)
        return dict(R=R, T=T)
    absorbed, r, t, R, T = absorption_S(structure, wavelength, theta, polarization)
    if pol == "p":  # H-field ratio to E-field ratio
        t = t * (n[0] / mu[0]) / (n[-1] / mu[-1])
    return dict(R=R, T=T, r=r, t=t, A=np.asarray(absorbed)[1:-1])


def elli_result(case, wavelength, theta, solver_name):
    import elli

    layers, d, _ = case
    n, _ = _split(layers)
    material = lambda value: elli.IsotropicMaterial(elli.ConstantRefractiveIndex(n=value))  # noqa: E731
    structure = elli.Structure(material(n[0]), [elli.Layer(material(v), t) for v, t in zip(n[1:-1], d[1:-1])],
                               material(n[-1]))
    solver = dict(two=elli.Solver2x2, four=elli.Solver4x4)[solver_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = structure.evaluate(np.array([wavelength]), np.degrees(theta), solver=solver)
    return dict(Rs=result.R_matrix[0, 1, 1], Rp=result.R_matrix[0, 0, 0], psi=result.psi[0], delta=result.delta[0])


def moosh_field(case, wavelength, theta, pol, step=5.0, substrate=200.0):
    """PyMoosh's field as one plane wave: depths z within each layer below the first interface."""
    import contextlib
    import io

    import PyMoosh as pm
    from PyMoosh.core import field

    layers, d, _ = case
    n, mu = _split(layers)
    materials = [[n[i] ** 2 / mu[i], mu[i]] if mu[i] != 1 else n[i] ** 2 for i in range(len(n))]
    thickness = [0.0] + list(d[1:-1]) + [substrate]
    structure = pm.Structure(materials, list(range(len(n))), thickness, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        width = 50.0
        beam = pm.Beam(wavelength, theta, 0 if pol == "s" else 1, 1e9 * width)
        values = field(structure, beam, pm.Window(width, 0.5, width, step))[:, 0]
    samples = [(layer, (m + 1) * t / np.floor(t / step))
               for layer, t in enumerate(thickness) if layer > 0 for m in range(int(np.floor(t / step)))]
    return samples, values


def stratify_field(case, wavelength, theta, pol, samples):
    layers, d, _ = case
    n, mu = _split(layers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        magnetic = any(m != 1 for m in mu)
        data = planar.coh_tmm(pol, n, d, theta, wavelength, mu_list=mu) if magnetic else \
            planar.coh_tmm(pol, n, d, theta, wavelength)
        if pol == "s":
            return np.array([planar.position_resolved(layer, z, data)["Ey"] for layer, z in samples])
        out = []
        for layer, z in samples:
            v, w = data["vw_list"][layer]
            kz = data["kz_list"][layer]
            out.append((n[layer] / mu[layer]) / (n[0] / mu[0]) * (v * np.exp(1j * kz * z) + w * np.exp(-1j * kz * z)))
    return np.array(out)


# name -> (core and cladding indices, thicknesses in nm, wavelength in nm); lossless, source in layer 1
GUIDES = {
    "symmetric slab, n 2.0 in air, 500 nm": ([1.0, 2.0, 1.0], [inf, 500.0, inf], 633.0),
    "asymmetric slab, n 2.0 on 1.45, 400 nm": ([1.0, 2.0, 1.45], [inf, 400.0, inf], 633.0),
    "two-layer guide in 1.46": ([1.46, 1.9, 1.6, 1.46], [inf, 300.0, 400.0, inf], 633.0),
    "SiN on SiO2 at 1550 nm": ([1.0, 1.99, 1.444], [inf, 800.0, inf], 1550.0),
}


def slab_modes(n_top, n_core, n_bottom, thickness, wavelength):
    """Textbook three-layer dispersion: k d = m pi + atan(r1 g1 / k) + atan(r3 g3 / k), TE (r = 1) and TM."""
    from scipy.optimize import brentq

    k0, low = 2 * np.pi / wavelength, max(n_top, n_bottom)
    modes = []
    for tm in (False, True):
        def residual(neff, m):
            kappa = k0 * np.sqrt(n_core ** 2 - neff ** 2)
            g1, g3 = k0 * np.sqrt(neff ** 2 - n_top ** 2), k0 * np.sqrt(neff ** 2 - n_bottom ** 2)
            r1, r3 = ((n_core / n_top) ** 2, (n_core / n_bottom) ** 2) if tm else (1.0, 1.0)
            return kappa * thickness - m * np.pi - np.arctan(r1 * g1 / kappa) - np.arctan(r3 * g3 / kappa)
        a, b = low * (1 + 1e-14), n_core * (1 - 1e-14)
        m = 0
        while residual(a, m) > 0:
            modes.append(brentq(residual, a, b, args=(m,), xtol=1e-15, rtol=4 * np.finfo(float).eps))
            m += 1
    return sorted(modes)


def guided_mode_sets(guide):
    import contextlib
    import io

    import PyMoosh as pm
    from PyMoosh.modes import guided_modes
    from pystratify.planar_emission import FilmSource

    n, d, wavelength = guide
    source = FilmSource(n, d, wavelength, 1, d[1] / 3)
    ours = sorted(np.array(source.guided_poles()) * source.ns)
    structure = pm.Structure([complex(x) ** 2 for x in n], list(range(len(n))), [0.0] + d[1:-1] + [0.0], verbose=False)
    low, high = max(n[0], n[-1]), max(n)
    theirs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for pol in (0, 1):
            for value in guided_modes(structure, wavelength, pol, low + 1e-3, high - 1e-3, initial_points=200):
                value = complex(value)
                # guided_modes also returns descents stalled near the light line; keep real guided indices
                if abs(value.imag) < 1e-8 and low < value.real < high:
                    theirs.append(value.real)
    analytic = slab_modes(n[0], n[1], n[2], d[1], wavelength) if len(n) == 3 else None
    return ours, sorted(theirs), analytic


def _set_difference(ours, theirs):
    if len(ours) != len(theirs):
        return np.inf
    return float(np.max(np.abs(np.array(ours) - np.array(theirs)), initial=0))


def _wrap(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def compare(cases=CASES, angles_deg=ANGLES_DEG, wavelengths=WAVELENGTHS_NM):
    """Worst absolute difference per (case, reference, quantity)."""
    worst = {}

    def record(name, reference, quantity, value):
        # max(0.0, nan) is 0.0: a NaN from either code must read as a failure, not agreement
        value = float(value) if np.isfinite(value) else np.inf
        key = (name, reference, quantity)
        worst[key] = max(worst.get(key, 0.0), value)

    for name, case in cases.items():
        layers, d, incoherent = case
        n, mu = _split(layers)
        magnetic = any(m != 1 for m in mu)
        for wavelength in wavelengths:
            for angle in angles_deg:
                theta = np.radians(angle)
                ours = {pol: stratify(case, wavelength, theta, pol) for pol in "sp"}
                for pol in "sp":
                    theirs = moosh(case, wavelength, theta, pol)
                    for quantity in ("R", "T", "r", "t"):
                        if quantity in ours[pol]:
                            record(name, "PyMoosh", quantity, abs(ours[pol][quantity] - theirs[quantity]))
                    if "A" in ours[pol]:
                        record(name, "PyMoosh", "A per layer", np.max(np.abs(ours[pol]["A"] - theirs["A"]), initial=0))
                if not incoherent:
                    for pol in "sp":
                        samples, theirs = moosh_field(case, wavelength, theta, pol)
                        ours_field = stratify_field(case, wavelength, theta, pol, samples)
                        record(name, "PyMoosh", "E_y (s), H_y (p)", np.max(np.abs(ours_field - theirs), initial=0))
                if incoherent or magnetic:
                    continue  # pyElli: no incoherent layers, no permeability
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ellips = planar.ellips(n, d, theta, wavelength)
                for solver in ("two", "four"):
                    label = "pyElli 2x2" if solver == "two" else "pyElli 4x4"
                    theirs = elli_result(case, wavelength, theta, solver)
                    record(name, label, "R",
                           max(abs(ours["s"]["R"] - theirs["Rs"]), abs(ours["p"]["R"] - theirs["Rp"])))
                    record(name, label, "Psi (deg)", abs(np.degrees(ellips["psi"]) - theirs["psi"]))
                    if np.degrees(ellips["psi"]) > 1e-6:  # Delta is undefined where r_p = 0
                        record(name, label, "Delta (deg)",
                               abs(_wrap(180.0 - np.degrees(ellips["Delta"]) - theirs["delta"])))
    for name, guide in GUIDES.items():
        ours, theirs, analytic = guided_mode_sets(guide)
        record(name, "PyMoosh", "guided n_eff", _set_difference(ours, theirs))
        if analytic is not None:
            record(name, "slab dispersion", "guided n_eff", _set_difference(ours, analytic))
    return worst


def report(worst):
    references = ("PyMoosh", "pyElli 2x2", "pyElli 4x4", "slab dispersion")
    quantities = ("R", "T", "r", "t", "A per layer", "E_y (s), H_y (p)", "Psi (deg)", "Delta (deg)",
                  "guided n_eff")
    print("# PyStratify planar films against PyMoosh and pyElli\n")
    print("Generated by `python benchmarks/planar_codes.py`. Worst absolute difference over")
    print(f"angles {', '.join(f'{a:g}' for a in ANGLES_DEG)} deg, wavelengths "
          f"{', '.join(f'{w:g}' for w in WAVELENGTHS_NM)} nm and both polarisations; - = not computed")
    print("(pyElli has no incoherent layers and no permeability). Fields: every 5 nm through each layer and")
    print("200 nm of substrate. Guided n_eff: every guided mode of a lossless guide, TE and TM; inf = the sets")
    print("differ in number.\n")
    for reference in references:
        rows = [name for name in [*CASES, *GUIDES] if any((name, reference, q) in worst for q in quantities)]
        used = [q for q in quantities if any((name, reference, q) in worst for name in rows)]
        print(f"## {reference}\n")
        print("| stack | " + " | ".join(used) + " |")
        print("|---|" + "---|" * len(used))
        for name in rows:
            cells = [f"{worst[(name, reference, q)]:.1e}" if (name, reference, q) in worst else "-" for q in used]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    report(compare(angles_deg=(0.0, 60.0) if quick else ANGLES_DEG,
                   wavelengths=(633.0,) if quick else WAVELENGTHS_NM))
