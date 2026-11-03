"""Films: the stable scattering-matrix kernel (`pystratify.stable`) against `planar.coh_tmm`, physics and the harvester.

    python benchmarks/stable_checks.py        # prints STABLE_CHECKS.md (~15 s)

`pystratify.stable` is the one coherent planar kernel of multilayer.app's replication corpus and calculator: Redheffer
star products of bounded scattering matrices, repeated blocks raised by binary doubling, uniaxial layers, a perfect
conductor and a semi-infinite periodic crystal as exits (its module docstring). What this script checks:

* **against `planar.coh_tmm`** (a different formulation: projective reflection sweeps), r, t, R, T on random stacks
  (lossless and absorbing, oblique s and p, absorbing exits), under a prism beyond the critical angle with
  evanescent layers, over a wavelength x angle grid, with an absorbing incident medium at normal incidence, a
  400-layer random stack, and an 8192-layer periodic stack at its stop band and on both sides of it;
* **repeated stacks**: nested groups and the automatic detection of a periodic stack equal the literal stack;
  131072 layers stay finite and take milliseconds;
* **uniaxial layers** (optic axis along the normal): s does not see eps_z (identical numbers), eps_z = eps_x is the
  isotropic layer (identical numbers), normal incidence does not see eps_z, and p agrees with a direct integration
  of Maxwell's equations for (E_x, H_y) through each layer (a matrix exponential, written here) -- a method that
  shares nothing with the kernel's (n, cos) pair; the same integration checks layers AT their own critical angle,
  where their two plane waves coincide (`planar.coh_tmm` returns R = 1 there; the kernel switches such a layer to
  its characteristic matrix);
* **perfect-conductor exit**: |r| = 1 and T = 0 on lossless stacks, and `coh_tmm` with an exit index i K tends to it;
* **Bloch exit**: a period of one homogeneous layer is that half-space (`coh_tmm`), a lossy crystal is a long lossy
  stack, a lossless crystal conserves energy and reflects everything in its stop band, and a long finite periodic
  stack's complex r tends to the crystal's inside a band gap (uniaxial layers too);
* **the harvester's own numbers** (`tests/stable_fixtures.json`, written by `benchmarks/stable_fixtures.py` from the
  harvester's `tmm_stable.coh_rt` and `verify.bloch_exit_rt`): uniaxial stacks, PEC exits, Bloch exits and the ruled
  65536-cell stack of 10.1119/1.4789883.

``tests/test_stable.py`` asserts every row (lengths and wavelengths in nm).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checks import flag, report, row  # noqa: E402
from pystratify import stable  # noqa: E402
from pystratify.planar import coh_tmm  # noqa: E402
from pystratify.stable import BlochExit, Layer, PEC_INDEX, Repeat, stable_tmm  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "stable_fixtures.json"
TOL = 1e-12
TOL_LONG = 1e-10
KEYS = ("r", "t", "R", "T")


def literal(stack):
    """The stack written out layer by layer: [(n, d, nz)]."""
    out = []
    for entry in stack:
        if isinstance(entry, Repeat):
            out += literal(entry.members) * entry.count
        elif isinstance(entry, tuple) and len(entry) == 3 and isinstance(entry[0], str):
            out += literal(entry[1]) * entry[2]
        else:
            entry = Layer(*entry)
            out.append((entry.n, entry.d, entry.nz))
    return out


def reference(pol, stack, n_in, n_out, th0, lam):
    """`planar.coh_tmm` point by point over the broadcast grid of (th0, lam) (isotropic layers)."""
    layers = literal(stack)
    th0, lam = np.broadcast_arrays(np.asarray(th0, dtype=complex), np.asarray(lam, dtype=float))
    out = {key: np.zeros(th0.shape, dtype=complex) for key in KEYS}
    for index in np.ndindex(th0.shape):
        pick = (lambda value: complex(np.broadcast_to(value, th0.shape)[index]) if np.ndim(value) else complex(value))
        n_list = [pick(n_in)] + [pick(n) for n, _, _ in layers] + [pick(n_out)]
        res = coh_tmm(pol, np.array(n_list), [np.inf] + [d for _, d, _ in layers] + [np.inf], th0[index].real
                      if th0[index].imag == 0 else th0[index], float(lam[index]))
        for key in KEYS:
            out[key][index] = res[key]
    return out


def worst(a, b, keys=KEYS):
    return max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in keys)


def random_layer(rng, absorbing, points=None):
    shape = () if points is None else (points,)
    n = rng.uniform(1.2, 3.5, shape) + 1j * rng.uniform(0.0, 0.8, shape) * (absorbing and rng.random() < 0.6)
    return (n, float(rng.uniform(5.0, 400.0)))


# --- isotropic stacks against coh_tmm --------------------------------------------------------------------------

def isotropic_against_coh_tmm():
    rng = np.random.default_rng(20261103)
    rows = []
    for absorbing in (False, True):
        dev = 0.0
        for case in range(80):
            stack = [random_layer(rng, absorbing, 3) for _ in range(rng.integers(0, 9))]
            lam = rng.uniform(300.0, 1500.0, 3)
            n_in = complex(rng.uniform(1.0, 1.8))
            n_out = complex(rng.uniform(1.0, 4.0), rng.uniform(0.0, 2.0) if absorbing else 0.0)
            th0 = float(rng.uniform(0.0, 1.45)) if case % 3 else 0.0
            for pol in ("s", "p"):
                dev = max(dev, worst(stable_tmm(pol, stack, n_in, n_out, th0, lam),
                                     reference(pol, stack, n_in, n_out, th0, lam)))
        label = "absorbing layers and exits" if absorbing else "lossless"
        rows.append(row(f"160 random stacks of 0-8 layers, {label}, oblique s and p", "max |a - coh_tmm|, a = r, t, R, T",
                        dev, TOL))
    # A prism beyond the critical angle: the air layer and the exit are evanescent, a group repeats it.
    stack = [(1.0 + 0j, 350.0), ("group", [(2.2 + 0j, 70.0), (1.0 + 0j, 900.0)], 5), (2.0 + 0.01j, 100.0)]
    dev = max(worst(stable_tmm(pol, stack, 1.5, 1.0, np.deg2rad(th), 633.0),
                    reference(pol, stack, 1.5, 1.0, np.deg2rad(th), 633.0))
              for pol in ("s", "p") for th in (45.0, 60.0, 80.0))
    rows.append(row("glass prism at 45-80 deg (beyond critical), evanescent layers, s and p", "max |a - coh_tmm|", dev, TOL))
    # Vectorised over a wavelength x angle grid.
    stack = [(1.9 + 0.02j, 110.0), Repeat([(2.3 + 0j, 60.0), (1.46 + 0j, 95.0)], 12)]
    lam = np.linspace(400.0, 900.0, 7)
    th = np.deg2rad(np.array([0.0, 30.0, 70.0]))[:, None]
    dev = 0.0
    for pol in ("s", "p"):
        ours = stable_tmm(pol, stack, 1.0, 1.52, th, lam)
        dev = max(dev, worst(ours, reference(pol, stack, 1.0, 1.52, th, lam)))
    rows.append(row("a 3 x 7 angle x wavelength grid in one call, s and p", "max |a - coh_tmm|", dev, TOL))
    rows.append(flag("the same grid", "result shape (3, 7)", ours["R"].shape == (3, 7)))
    # An absorbing incident medium at normal incidence (coh_tmm's power convention).
    stack = [(2.0 + 0.1j, 80.0), (1.4 + 0j, 120.0)]
    dev = max(worst(stable_tmm(pol, stack, 1.5 + 0.02j, 1.0, 0.0, lam), reference(pol, stack, 1.5 + 0.02j, 1.0, 0.0, lam))
              for pol in ("s", "p"))
    rows.append(row("absorbing incident medium n = 1.5 + 0.02i at normal incidence", "max |a - coh_tmm|", dev, TOL))
    # Unpolarised: the mean of s and p.
    res = stable_tmm("unpolarized", stack, 1.0, 1.52, 0.7, lam)
    mean = max(float(np.max(np.abs(res[k] - (res["s"][k] + res["p"][k]) / 2))) for k in ("R", "T"))
    rows.append(row("unpolarized at 0.7 rad", "max |R - (R_s + R_p)/2|, T alike", mean, 0.0))
    return rows


def long_stacks():
    rows = []
    rng = np.random.default_rng(7)
    stack = [random_layer(rng, True) for _ in range(400)]
    dev = max(worst(stable_tmm(pol, stack, 1.0, 1.52, 0.5, lam), reference(pol, stack, 1.0, 1.52, 0.5, lam))
              for pol in ("s", "p") for lam in (450.0, 800.0))
    rows.append(row("400 random absorbing layers at 0.5 rad, s and p", "max |a - coh_tmm|", dev, TOL_LONG))
    # 10.1119/1.4789883's cell, 4096 periods (8192 layers): its stop band at 581 nm and both sides of it.
    dev = 0.0
    for lam in (450.0, 581.4, 700.0):
        k = 1e-4 * lam / (2 * np.pi * 90.0)
        cell = [(1.55 + 1j * k, 90.0), (1.68 + 1j * k, 90.0)]
        for pol in ("s", "p"):
            for th in (0.0, 0.5):
                ours = stable_tmm(pol, [Repeat(cell, 4096)], 1.0, 1.0, th, lam)
                ref = reference(pol, [Repeat(cell, 4096)], 1.0, 1.0, th, lam)
                if not all(np.isfinite(ref[key]).all() for key in KEYS):
                    raise AssertionError("coh_tmm is not finite on the 8192-layer stack")
                dev = max(dev, worst(ours, ref))
    rows.append(row("8192-layer periodic stack (4096 cells of 10.1119/1.4789883), 450/581/700 nm, 0 and 0.5 rad",
                    "max |a - coh_tmm|", dev, TOL_LONG))
    # 131072 layers: 65536 cells, as a group and as a flat list the kernel finds the period of.
    lam = np.linspace(400.0, 800.0, 401)
    k = 1e-4 * lam / (2 * np.pi * 90.0)
    cell = [(1.55 + 1j * k, 90.0), (1.68 + 1j * k, 90.0)]
    start = time.perf_counter()
    grouped = stable_tmm("s", [Repeat(cell, 65536)], 1.0, 1.0, 0.0, lam)
    seconds = time.perf_counter() - start
    start = time.perf_counter()
    flat = stable_tmm("s", cell * 65536, 1.0, 1.0, 0.0, lam)
    flat_seconds = time.perf_counter() - start
    R, T = grouped["R"], grouped["T"]
    finite = all(np.isfinite(grouped[key]).all() for key in KEYS)
    rows += [
        flag("131072 layers (65536 cells) x 401 wavelengths", "r, t, R, T finite", finite),
        row("131072 layers x 401 wavelengths", "max(R + T) - 1 (passive)", float(np.max(R + T)) - 1, 1e-12),
        row("131072 layers x 401 wavelengths", "min(R), min(T) (>= 0)", min(float(R.min()), float(T.min())), 0.0, ">="),
        row("131072 layers x 401 wavelengths", "max R (the stop band)", float(R.max()), 0.99, ">"),
        row("131072 layers x 401 wavelengths, as a repeat group", "seconds", seconds, 1.0),
        row("131072 layers as a flat list (period detected)", "seconds", flat_seconds, 3.0),
        row("131072 layers as a flat list (period detected)", "max |a - grouped|", worst(flat, grouped), TOL),
    ]
    return rows


def repeats():
    rows = []
    rng = np.random.default_rng(11)
    dev_nested = dev_detect = 0.0
    for case in range(40):
        absorbing = case % 2 == 0
        inner = Repeat([random_layer(rng, absorbing) for _ in range(rng.integers(1, 4))], int(rng.integers(1, 5)))
        outer = Repeat([random_layer(rng, absorbing), inner], int(rng.integers(1, 6)))
        stack = [random_layer(rng, absorbing), outer, ("group", [random_layer(rng, absorbing)], 3)]
        th, lam = float(rng.uniform(0.0, 1.3)), float(rng.uniform(300.0, 1500.0))
        n_out = complex(rng.uniform(1.0, 3.0), 0.05 if absorbing else 0.0)
        for pol in ("s", "p"):
            dev_nested = max(dev_nested, worst(stable_tmm(pol, stack, 1.0, n_out, th, lam),
                                               reference(pol, stack, 1.0, n_out, th, lam)))
        block = [random_layer(rng, absorbing) for _ in range(rng.integers(1, 5))]
        flat = block * int(rng.integers(2, 30))
        for pol in ("s", "p"):
            dev_detect = max(dev_detect, worst(stable_tmm(pol, flat, 1.0, n_out, th, lam),
                                               stable_tmm(pol, flat, 1.0, n_out, th, lam, detect_repeats=False)))
    rows.append(row("40 stacks of groups nested two deep, written as Repeat and as ('group', ...)",
                    "max |a - coh_tmm of the literal stack|", dev_nested, TOL))
    rows.append(row("40 flat periodic stacks (2-29 periods): detected period vs layer by layer",
                    "max |a - literal|", dev_detect, TOL))
    lam = np.linspace(400.0, 800.0, 11)
    copies = [((1.55 + 0.01j) * np.ones(11), 90.0), ((1.68 + 0j) * np.ones(11), 80.0)]
    by_value = [(n.copy(), d) for _ in range(6) for n, d in copies]
    rows.append(flag("12 layers, two distinct arrays copied by value", "block_period = 2",
                     stable.block_period(by_value) == 2))
    by_value[5] = (by_value[5][0], 80.0000001)
    rows.append(flag("the same with one thickness off by 1e-7 nm", "block_period = 12 (no period)",
                     stable.block_period(by_value) == 12))
    rows.append(flag("expanded_layer_count of 1 + 2 x 65536 layers", "131073",
                     stable.expanded_layer_count([(1.5, 0.1), ("group", [(2.0, 0.1), (1.4, 0.1)], 65536)]) == 131073))
    return rows


# --- uniaxial layers ---------------------------------------------------------------------------------------------

def ode_rt(pol, layers, n_in, n_out, th0, lam):
    """r, R, T of a stack of (uniaxial) layers by integrating Maxwell's equations through each layer.

    With fields ~ exp(i k0 beta x - i w t), H in units of the vacuum admittance and eps = diag(eps_x, eps_x, eps_z),
    the tangential fields obey d/dz (E, H) = i k0 K (E, H) with K = [[0, 1], [eps_x - beta^2, 0]] for s (E_y) and
    K = [[0, 1 - beta^2/eps_z], [eps_x, 0]] for p (E_x, H_y), integrated by scipy's matrix exponential: no plane-wave
    basis inside a layer, so nothing degenerates at a layer's own critical angle. The half-spaces are isotropic with
    admittance H/E = n cos(theta) (s) or n/cos(theta) (p). r is returned in planar's conventions: the ratio of the
    reflected to the incident E_y for s, minus that ratio of E_x for p.
    """
    k0 = 2 * np.pi / lam
    beta = n_in * np.sin(th0)
    M = np.eye(2, dtype=complex)
    for n, d, nz in layers:
        eps_x, eps_z = n * n, (n if nz is None else nz) ** 2
        K = [[0, 1], [eps_x - beta ** 2, 0]] if pol == "s" else [[0, 1 - beta ** 2 / eps_z], [eps_x, 0]]
        M = expm(1j * k0 * d * np.array(K, dtype=complex)) @ M
    cos_out = np.lib.scimath.sqrt(1 - (beta / n_out) ** 2)
    if (n_out * cos_out).imag < 0 or ((n_out * cos_out).imag == 0 and (n_out * cos_out).real < 0):
        cos_out = -cos_out
    if pol == "s":
        eta_in, eta_out = n_in * np.cos(th0), n_out * cos_out
    else:
        eta_in, eta_out = n_in / np.cos(th0), n_out / cos_out
    front = np.linalg.solve(M, np.array([1.0, eta_out]))        # (E, H) at the front face for E_t = 1
    incident = (front[0] + front[1] / eta_in) / 2
    reflected = (front[0] - front[1] / eta_in) / 2
    r = (1 if pol == "s" else -1) * reflected / incident
    T = eta_out.real / (eta_in * abs(incident) ** 2)
    return r, abs(r) ** 2, T


def _ode_layers(stack):
    return [(complex(l.n), l.d, None if l.nz is None else complex(l.nz)) for l in stack]


def uniaxial_physics():
    rows = []
    rng = np.random.default_rng(5)
    lam = np.linspace(450.0, 900.0, 5)
    stacks = []
    for _ in range(20):
        stacks.append([Layer(rng.uniform(1.3, 2.6) + 1j * rng.uniform(0, 0.1), rng.uniform(20, 250),
                             rng.uniform(1.2, 3.0) + 1j * rng.uniform(0, 0.05) if rng.random() < 0.7 else None)
                       for _ in range(rng.integers(1, 6))])
    s_dev = iso_dev = normal_dev = ode_dev = 0.0
    for stack in stacks:
        isotropic = [Layer(layer.n, layer.d) for layer in stack]
        same = [Layer(layer.n, layer.d, layer.n) for layer in stack]
        th = float(rng.uniform(0.1, 1.3))
        s_dev = max(s_dev, worst(stable_tmm("s", stack, 1.0, 1.52, th, lam), stable_tmm("s", isotropic, 1.0, 1.52, th, lam)))
        for pol in ("s", "p"):
            iso_dev = max(iso_dev, worst(stable_tmm(pol, same, 1.0, 1.52, th, lam),
                                         stable_tmm(pol, isotropic, 1.0, 1.52, th, lam)))
        normal_dev = max(normal_dev, worst(stable_tmm("p", stack, 1.0, 1.52, 0.0, lam),
                                           stable_tmm("p", isotropic, 1.0, 1.52, 0.0, lam)))
        ours = stable_tmm("p", stack, 1.2, 1.52, th, lam)
        for i, wavelength in enumerate(lam):
            r, R, T = ode_rt("p", _ode_layers(stack), 1.2, 1.52, th, wavelength)
            ode_dev = max(ode_dev, abs(ours["r"][i] - r), abs(ours["R"][i] - R), abs(ours["T"][i] - T))
    # A hyperbolic (eps_z < 0, lossy) layer under a prism, past the exit's critical angle.
    hyper = [Layer(1.9 + 0.05j, 40.0, np.sqrt(-4.0 + 0.6j)), Layer(1.46, 200.0, 1.6)]
    for th in np.deg2rad([10.0, 40.0, 70.0]):
        ours = stable_tmm("p", hyper, 1.72, 1.33, th, 633.0)
        r, R, T = ode_rt("p", _ode_layers(hyper), 1.72, 1.33, th, 633.0)
        ode_dev = max(ode_dev, abs(ours["r"] - r), abs(ours["R"] - R), abs(ours["T"] - T))
    # Layers at their own critical angle (cos = 0 inside them: the plane-wave basis degenerates) and 1e-12 to 1e-3 rad
    # from it, isotropic and uniaxial, against the same integration.
    critical_dev = 0.0
    glass = [Layer(1.0, 100.0), Layer(2.0, 50.0), Layer(1.3, 300.0, 1.0)]
    for pol in ("s", "p"):
        for target in (1.0, 1.3):                               # the air layer's, then (s) the uniaxial n_x's
            th_c = np.arcsin(target / 1.5)
            for offset in (0.0, 1e-12, -1e-12, 1e-9, -1e-6, 1e-3):
                ours = stable_tmm(pol, glass, 1.5, 1.52, th_c + offset, 600.0)
                r, R, T = ode_rt(pol, _ode_layers(glass), 1.5, 1.52, th_c + offset, 600.0)
                critical_dev = max(critical_dev, abs(ours["r"] - r), abs(ours["R"] - R), abs(ours["T"] - T))
    rows += [
        row("layers at their own critical angle (cos theta = 0) and 1e-12 to 1e-3 rad from it, s and p, isotropic and "
            "uniaxial: (E, H) ODE by matrix exponential", "max |r - r_ODE|, |R - R_ODE|, |T - T_ODE|", critical_dev, TOL),
        row("20 random uniaxial stacks, s", "max |a(eps_z) - a(isotropic n_x)|", s_dev, 0.0),
        row("the same stacks with eps_z = eps_x, s and p", "max |a - isotropic|", iso_dev, 0.0),
        row("the same stacks at normal incidence, p", "max |a(eps_z) - a(isotropic)|", normal_dev, 0.0),
        row("the same stacks (p, oblique) and a hyperbolic lossy layer under a prism: (E_x, H_y) ODE by matrix exponential",
            "max |r_p - r_ODE|, |R - R_ODE|, |T - T_ODE|", ode_dev, TOL),
    ]
    return rows


# --- the perfect-conductor exit ----------------------------------------------------------------------------------

def pec_physics():
    rows = []
    angles = np.deg2rad(np.linspace(0.0, 85.0, 18))
    lossless = {
        "three dielectric layers from air": ([(1.46, 95.0), (2.3, 60.0), (1.46, 95.0)], 1.0),
        "a uniaxial film under a prism (TIR in the film)": ([Layer(1.38, 120.0, 1.52), Layer(1.7, 150.0, 2.2)], 1.72),
        "bare conductor under water": ([], 1.33),
    }
    for name, (stack, n_in) in lossless.items():
        dev, t_max = 0.0, 0.0
        for pol in ("s", "p"):
            res = stable_tmm(pol, stack, n_in, PEC_INDEX, angles, 550.0)
            dev = max(dev, float(np.max(np.abs(res["R"] - 1))))
            t_max = max(t_max, float(np.max(np.abs(res["T"]))), float(np.max(np.abs(res["t"]))))
        rows.append(row(f"PEC exit, {name}, 0-85 deg, s and p", "max |R - 1|", dev, 1e-14))
        rows.append(row(f"PEC exit, {name}, 0-85 deg, s and p", "max |T|, |t|", t_max, 0.0))
    res = stable_tmm("p", [(0.2 + 3.4j, 15.0), (1.5, 120.0)], 1.0, PEC_INDEX, 0.7, np.linspace(400, 900, 6))
    rows.append(row("PEC under an absorbing film", "max R (< 1: the film absorbs)", float(res["R"].max()), 1.0, "<"))
    dev = 0.0
    stack = [(1.46, 95.0), (2.3 + 0.01j, 60.0)]
    for pol in ("s", "p"):
        for th in np.deg2rad([0.0, 30.0, 60.0]):
            ours = stable_tmm(pol, stack, 1.0, PEC_INDEX, th, 550.0)
            ref = reference(pol, stack, 1.0, 1e7j, th, 550.0)
            dev = max(dev, worst(ours, ref, ("r", "R")))
    rows.append(row("coh_tmm with exit index 1e7 i (a nearly perfect conductor) vs the PEC exit, s and p",
                    "max |r - r_PEC|, |R - R_PEC| (O(1/K))", dev, 1e-6))
    return rows


# --- the Bloch exit ----------------------------------------------------------------------------------------------

ZNSE, AL2O3, TIO2 = np.sqrt(6.14), np.sqrt(3.08), np.sqrt(6.18)


def bloch_physics():
    rows = []
    angles = np.deg2rad([0.0, 10.0, 30.0, 44.0, 45.2, 50.0, 70.0, 85.0])
    dev = 0.0
    for n_exit, front, split in ((AL2O3, [], 1), (TIO2, [], 3), (2.0 + 0.5j, [], 1),
                                 (1.6 + 0.01j, [(TIO2, 57.0), (1.38, 110.0)], 1)):
        period = [(n_exit, 120.0 / split)] * split
        for pol in ("s", "p"):
            ours = stable_tmm(pol, front + period, ZNSE, BlochExit(split), angles, 1000.0)
            dev = max(dev, worst(ours, reference(pol, front, ZNSE, n_exit, angles, 1000.0), ("R", "T")))
    rows.append(row("a period of one homogeneous layer (or three) is that half-space: lossless (incl. TIR), absorbing, "
                    "behind a front stack", "max |R, T - coh_tmm|", dev, TOL))
    # A lossy crystal is a long lossy stack: 1200 periods return less than exp(-140) of the light.
    period = [(1.75 + 0.1j, 38.0), (2.49 + 0.1j, 57.0)]
    dev = 0.0
    for pol in ("s", "p"):
        for th in np.deg2rad([0.0, 30.0, 55.0, 70.0]):
            ours = stable_tmm(pol, period, ZNSE, BlochExit(2), th, 1000.0)
            long = stable_tmm(pol, [Repeat(period, 1200)], ZNSE, 1.0, th, 1000.0)
            dev = max(dev, worst(ours, long, ("r", "R")))
    rows.append(row("lossy crystal (k = 0.1) vs 1200 periods on air", "max |r - r_1200|, |R - R_1200|", dev, TOL))
    # 10.1103/physrevb.101.075127's lossless crystal from ZnSe: energy, and total reflection past the knee.
    crystal = [(AL2O3, 38.0), (TIO2, 57.0)]
    scan = np.deg2rad(np.linspace(0.0, 89.0, 60))
    balance = knee = 0.0
    for pol in ("s", "p"):
        res = stable_tmm(pol, crystal, ZNSE, BlochExit(2), scan, 1000.0)
        balance = max(balance, float(np.max(np.abs(res["R"] + res["T"] - 1))))
        beyond = scan > np.deg2rad(66.0)
        knee = max(knee, float(np.max(np.abs(res["R"][beyond] - 1))))
    rows.append(row("lossless Al2O3/TiO2 crystal from ZnSe, 0-89 deg, s and p", "max |R + T - 1|", balance, TOL))
    rows.append(row("the same past 66 deg (no propagating Bloch wave)", "max |R - 1|", knee, TOL))
    # A long finite periodic stack tends to the crystal inside a band gap: complex r, so the phase too.
    gap = {
        "quarter-wave TiO2/SiO2 at its gap, 30 deg, behind MgF2": ([(1.38, 100.0)], [(2.3, 60.0), (1.46, 95.0)], 1.0,
                                                                   np.deg2rad(30.0), 560.0),
        "uniaxial cell at its gap, 40 deg": ([], [Layer(2.2, 62.0, 1.9), Layer(1.46, 95.0, 1.7 + 0.001j)], 1.0,
                                            np.deg2rad(40.0), 540.0),
    }
    for name, (front, period, n_in, th, lam) in gap.items():
        far = near = 0.0
        for pol in ("s", "p"):
            crystal_r = stable_tmm(pol, front + period, n_in, BlochExit(len(period)), th, lam)["r"]
            for cells, store in ((80, "far"), (6, "near")):
                finite = stable_tmm(pol, front + [Repeat(period, cells)], n_in, 1.52, th, lam)["r"]
                gap_dev = abs(complex(finite) - complex(crystal_r))
                if store == "far":
                    far = max(far, gap_dev)
                else:
                    near = max(near, gap_dev)
        rows.append(row(f"{name}: 80 periods on glass vs the semi-infinite crystal, s and p", "max |r_80 - r_Bloch|",
                        far, TOL))
        rows.append(row(f"{name}: 6 periods (not yet converged)", "|r_6 - r_Bloch|", near, 1e-6, ">"))
    return rows


# --- the harvester's numbers -------------------------------------------------------------------------------------

def _decode(value, points):
    z = np.asarray(value, dtype=float)
    return np.broadcast_to(z[..., 0] + 1j * z[..., 1], (points,)).copy()


def fixture_case(case):
    """`pystratify.stable` on one fixture case: R, T over its points."""
    points = len(case["th0"])
    layers = [Layer(_decode(item["n"], points), item["d"], None if item["nz"] is None else _decode(item["nz"], points))
              for item in case["layers"]]
    stack = [Repeat(layers, case["repeat"])] if case["repeat"] > 1 else layers
    exit_ = case["exit"]
    n_out = (BlochExit(exit_["bloch"]) if isinstance(exit_, dict) else PEC_INDEX if exit_ == "PEC"
             else _decode(exit_, points))
    res = stable_tmm(case["pol"], stack, _decode(case["n_in"], points), n_out, np.asarray(case["th0"]),
                     np.asarray(case["lam"]))
    return res["R"], res["T"]


def harvester_fixture():
    fixture = json.loads(FIXTURE.read_text())
    families = {"uniaxial": [], "PEC": [], "Bloch": [], "10.1119": []}
    for case in fixture["cases"]:
        R, T = fixture_case(case)
        dev = max(float(np.max(np.abs(R - case["R"]))), float(np.max(np.abs(T - case["T"]))))
        exit_ = case["exit"]
        family = ("Bloch" if isinstance(exit_, dict) else "PEC" if exit_ == "PEC"
                  else "10.1119" if case["repeat"] == 65536 else "uniaxial")
        families[family].append((len(case["th0"]), dev))
    labels = {
        "uniaxial": "uniaxial stacks (random, hyperbolic, under a prism, 40 periods): tmm_stable.coh_rt",
        "PEC": "PEC exits (dielectric, absorbing, bare, uniaxial; angle scans): tmm_stable.coh_rt",
        "Bloch": ("Bloch exits (physrevb.101.075127's crystal, front stacks, lossy, 3-layer period, at a layer's critical "
                  "angle): verify.bloch_exit_rt"),
        "10.1119": "10.1119/1.4789883's ruled stack, 65536 cells, s and p: tmm_stable.coh_rt",
    }
    commit = fixture.get("harvester_commit")
    return [row(f"{labels[k]} (harvester {commit}), {len(v)} cases, {sum(p for p, _ in v)} points",
                "max |R - R_harvester|, |T - T_harvester|", max(d for _, d in v), TOL) for k, v in families.items()]


CHECKS = (isotropic_against_coh_tmm, long_stacks, repeats, uniaxial_physics, pec_physics, bloch_physics,
          harvester_fixture)


if __name__ == "__main__":
    report("Films: the stable scattering-matrix kernel",
           "Generated by `python benchmarks/stable_checks.py`; `tests/test_stable.py` asserts every row (lengths and "
           "wavelengths in nm). `pystratify.stable` against `planar.coh_tmm`, closed-form physics, a direct integration of "
           "Maxwell's equations for uniaxial layers, and the harvester's own numbers (`tests/stable_fixtures.json`).",
           [r for check in CHECKS for r in check()])
