"""Writes tests/stable_fixtures.json: the harvester's own numbers for the physics `pystratify.stable` took over from it.

    HARVESTER=/path/to/multilayerapp/harvester
    $HARVESTER/.venv/bin/python benchmarks/stable_fixtures.py --harvester $HARVESTER

Run by hand only, in an environment that can import multilayer.app's (non-public) harvester: its
``tmm_stable.py`` (the scattering-matrix kernel with uniaxial layers and the perfect-conductor exit) and
``verify.py``'s ``bloch_exit_rt`` (the Bloch-terminated exit). Nothing in PyStratify imports the harvester;
``benchmarks/stable_checks.py`` reads the committed fixture and recomputes every case with ``pystratify.stable``
(``tests/test_stable.py`` asserts 1e-12 in R and T).

Every input is written into the fixture, so the comparison needs no random generator and no harvester. Complex
numbers are ``[re, im]`` pairs; an index is one pair (constant) or one pair per point; ``th0`` (radians) and
``lam`` (nm) list one value per point, and the cases are evaluated point by point by the harvester's Bloch path and
vectorised by its ``coh_rt``. A case whose ``repeat`` is k > 1 is its layers k times over (the harvester detects
the period itself).
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests" / "stable_fixtures.json"
SEED = 20261103


def pair(z):
    z = np.asarray(z, dtype=complex)
    if z.ndim == 0:
        return [float(z.real), float(z.imag)]
    return [[float(v.real), float(v.imag)] for v in z]


def layer(n, d, nz=None):
    return {"n": pair(n), "d": float(d), "nz": None if nz is None else pair(nz)}


def decode(value, points):
    z = np.asarray(value, dtype=float)
    z = z[..., 0] + 1j * z[..., 1]
    return np.broadcast_to(z, (points,)).copy()


def harvester_case(tmm_stable, verify, case):
    """R, T of one fixture case by the harvester's own code."""
    th0 = np.asarray(case["th0"], dtype=float)
    lam = np.asarray(case["lam"], dtype=float)
    points = len(th0)
    layers = [decode(item["n"], points) for item in case["layers"]] * case["repeat"]
    d = [item["d"] for item in case["layers"]] * case["repeat"]
    nz = [None if item["nz"] is None else decode(item["nz"], points) for item in case["layers"]] * case["repeat"]
    n_in = decode(case["n_in"], points)
    exit_ = case["exit"]
    if isinstance(exit_, dict):
        R, T = np.zeros(points), np.zeros(points)
        for i in range(points):
            R[i], T[i] = verify.bloch_exit_rt(case["pol"], [x[i] for x in layers], d, exit_["bloch"], n_in[i],
                                              th0[i], lam[i])
        return R, T
    n_out = tmm_stable.PEC_INDEX if exit_ == "PEC" else decode(exit_, points)
    uniaxial = any(value is not None for value in nz)
    return tmm_stable.coh_rt(case["pol"], layers, d, n_in, n_out, th0, lam, layers_nz=nz if uniaxial else None)


def cases():
    rng = np.random.default_rng(SEED)
    out = []

    def add(name, pol, layers, n_in, exit_, th0, lam, repeat=1):
        th0, lam = np.broadcast_arrays(np.asarray(th0, float), np.asarray(lam, float))
        out.append({"name": name, "pol": pol, "layers": layers, "repeat": repeat, "n_in": pair(n_in),
                    "exit": exit_, "th0": [float(x) for x in th0], "lam": [float(x) for x in lam]})

    # Uniaxial layers (optic axis along the normal): random stacks, lossless and lossy, both signs of
    # birefringence, a hyperbolic (eps_z < 0) lossy layer, oblique s and p, over wavelengths.
    for k in range(8):
        lam = np.sort(rng.uniform(400.0, 1200.0, 6))
        count = int(rng.integers(1, 6))
        layers = []
        for j in range(count):
            n = rng.uniform(1.3, 2.6, 6) + 1j * rng.uniform(0.0, 0.2, 6) * (rng.random() < 0.5)
            nz = None if rng.random() < 0.3 else rng.uniform(1.2, 3.0) + 1j * rng.uniform(0.0, 0.1) * (k % 2)
            layers.append(layer(n, rng.uniform(10.0, 300.0), nz))
        if k == 7:
            layers.append(layer(1.9 + 0.05j, 40.0, np.sqrt(-4.0 + 0.6j)))
        n_out = rng.uniform(1.0, 3.0) + 0j
        th0 = float(rng.uniform(0.2, 1.3))
        for pol in ("s", "p"):
            add(f"uniaxial stack {k}, {pol}", pol, layers, rng.uniform(1.0, 1.8) if k else 1.0, pair(n_out), th0, lam)
    # Uniaxial against angle at one wavelength, from a prism (beyond the critical angle of the exit).
    angles = np.deg2rad(np.linspace(0.0, 85.0, 18))
    prism = [layer(1.38, 120.0, 1.52), layer(2.1 + 0.01j, 80.0, 1.8 + 0.02j), layer(1.46, 300.0)]
    for pol in ("s", "p"):
        add(f"uniaxial under a prism, angle scan, {pol}", pol, prism, 1.72, pair(1.33), angles, 633.0)
    # A periodic uniaxial stack: the harvester's repeated-block path.
    cell = [layer(2.2, 70.0, 2.0), layer(1.46 + 0.001j, 105.0, 1.6)]
    for pol in ("s", "p"):
        add(f"uniaxial period x 40, {pol}", pol, cell, 1.0, pair(1.52), 0.6, np.linspace(450.0, 800.0, 8), repeat=40)

    # Perfect-conductor exit.
    mirror = [layer(1.46, 95.0), layer(2.3, 60.0), layer(1.46, 95.0)]
    for pol in ("s", "p"):
        add(f"dielectric stack on PEC, angle scan, {pol}", pol, mirror, 1.0, "PEC", angles, 550.0)
        add(f"absorbing film on PEC, {pol}", pol, [layer(0.2 + 3.4j, 15.0), layer(1.5, 120.0)], 1.0, "PEC",
            0.7, np.linspace(400.0, 900.0, 6))
        add(f"bare PEC, {pol}", pol, [], 1.33, "PEC", angles[:6], 600.0)
        add(f"uniaxial film on PEC, {pol}", pol, [layer(1.7, 150.0, 2.2 + 0.03j)], 1.5, "PEC", angles, 700.0)

    # Bloch-terminated exit: 10.1103/physrevb.101.075127's Al2O3/TiO2 crystal from ZnSe at 1 um (App. E), with and
    # without a finite stack in front, a lossy crystal, a three-layer period, over angle and over wavelength.
    znse, al2o3, tio2 = np.sqrt(6.14), np.sqrt(3.08), np.sqrt(6.18)
    crystal = [layer(al2o3, 38.0), layer(tio2, 57.0)]
    scan = np.deg2rad(np.linspace(0.0, 89.0, 30))
    fronts = {"no front": [], "two-layer front": [layer(tio2, 57.0), layer(1.38, 110.0)]}
    for front_name, front in fronts.items():
        for pol in ("s", "p"):
            add(f"Bloch exit, Al2O3/TiO2 crystal, {front_name}, {pol}", pol, front + crystal, znse,
                {"bloch": 2}, scan, 1000.0)
    lossy = [layer(1.75 + 0.1j, 38.0), layer(2.49 + 0.1j, 57.0)]
    three = [layer(1.46, 100.0), layer(2.3, 60.0), layer(1.9 + 0.02j, 30.0)]
    for pol in ("s", "p"):
        add(f"Bloch exit, lossy crystal, {pol}", pol, [layer(1.6 + 0.05j, 80.0)] + lossy, znse, {"bloch": 2},
            scan, 1000.0)
        add(f"Bloch exit, three-layer period, {pol}", pol, three, 1.0, {"bloch": 3}, scan[:20], 700.0)
        add(f"Bloch exit, over wavelength at 30 deg, {pol}", pol, [layer(1.38, 90.0)] + crystal, znse,
            {"bloch": 2}, np.deg2rad(30.0), np.linspace(500.0, 1500.0, 21))
        # At (and 1e-12 to 1e-6 rad from) the Al2O3 layers' own critical angle, in front of and inside the crystal:
        # the harvester's (E, H) matrices do not degenerate there.
        critical = np.arcsin(al2o3 / znse) + np.array([0.0, 1e-12, -1e-12, 1e-9, -1e-6])
        add(f"Bloch exit at the Al2O3 critical angle, {pol}", pol, [layer(al2o3, 50.0), layer(1.38, 100.0)] + crystal,
            znse, {"bloch": 2}, critical, 1000.0)

    # The ruled long stack of 10.1119/1.4789883: (n1 90 nm / n2 90 nm)^65536 with n = 1.55/1.68 + i 1e-4 lambda /
    # (2 pi 90 nm), air both sides -- the harvester's block path, the corpus's longest record.
    lam = np.linspace(560.0, 600.0, 9)
    k = 1e-4 * lam / (2 * np.pi * 90.0)
    ruled = [layer(1.55 + 1j * k, 90.0), layer(1.68 + 1j * k, 90.0)]
    add("10.1119/1.4789883 ruled stack, 65536 cells, s, normal", "s", ruled, 1.0, pair(1.0), 0.0, lam, repeat=65536)
    add("10.1119/1.4789883 ruled stack, 65536 cells, p, 0.3 rad", "p", ruled, 1.0, pair(1.0), 0.3, lam, repeat=65536)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harvester", required=True, help="the harvester checkout (tmm_stable.py, verify.py)")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    root = Path(args.harvester).resolve()
    sys.path.insert(0, str(root))
    tmm_stable = importlib.import_module("tmm_stable")
    verify = importlib.import_module("verify")
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # provenance only
        commit = None
    fixture = {"generator": "benchmarks/stable_fixtures.py", "harvester_commit": commit,
               "harvester_files": ["tmm_stable.py (coh_rt)", "verify.py (bloch_exit_rt)"], "cases": []}
    for case in cases():
        R, T = harvester_case(tmm_stable, verify, case)
        fixture["cases"].append({**case, "R": [float(x) for x in R], "T": [float(x) for x in T]})
    Path(args.out).write_text(json.dumps(fixture, indent=None, separators=(",", ":")) + "\n")
    print(f"{len(fixture['cases'])} cases -> {args.out}")


if __name__ == "__main__":
    main()
