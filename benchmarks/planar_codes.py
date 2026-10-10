"""Planar films against two independent codes: PyMoosh and pyElli (benchmarks/MATRIX.md, films).

    pip install -e ".[benchmarks]"
    python benchmarks/planar_codes.py            # prints a Markdown report
    python benchmarks/planar_codes.py --quick    # one wavelength, two angles

``pystratify.planar`` descends from Byrnes's ``tmm``, so agreeing with ``tmm`` proves nothing
about the formulation. These two codes do not share that ancestry:

* PyMoosh (Langevin, Moreau et al.): a scattering-matrix code, unconditionally stable,
  with its own incoherent recursion (``PyMoosh.incoherent``);
* pyElli (Müller, Dobener et al.): ellipsometry with a 2x2 and a Berreman 4x4 solver.

Conventions found while writing this, all three codes using n + ik for absorption:

* PyMoosh's p-polarised ``t`` is the ratio of the magnetic fields; ``tmm``'s is the ratio of the
  electric fields. They differ by (n_in / mu_in) / (n_out / mu_out), which is applied here.
* pyElli defines rho = tan(Psi) exp(-i Delta) with r_p of the opposite sign, so its
  Delta = 180 deg - Delta(tmm), modulo 360 deg. Psi needs no conversion.

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
    return worst


def report(worst):
    references = ("PyMoosh", "pyElli 2x2", "pyElli 4x4")
    quantities = ("R", "T", "r", "t", "A per layer", "Psi (deg)", "Delta (deg)")
    print("# PyStratify planar films against PyMoosh and pyElli\n")
    print("Generated by `python benchmarks/planar_codes.py`. Worst absolute difference over")
    print(f"angles {', '.join(f'{a:g}' for a in ANGLES_DEG)} deg, wavelengths "
          f"{', '.join(f'{w:g}' for w in WAVELENGTHS_NM)} nm and both polarisations; - = not computed")
    print("(pyElli has no incoherent layers and no permeability).\n")
    for reference in references:
        rows = [name for name in CASES if any((name, reference, q) in worst for q in quantities)]
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
