"""Films against Byrnes's `tmm` 0.2.0, the package `planar.py` descends from (benchmarks/MATRIX.md, **lineage**).

    pip install tmm==0.2.0
    python benchmarks/byrnes_tmm.py         # prints BYRNES_TMM.md (~5 s)

`pystratify.planar` keeps `tmm`'s interface, conventions and incoherent ladder (MIT notice in
src/pystratify/PLANAR-LICENSE.txt), so this comparison is a regression check, not an independent one -- except in the
coherent kernel: `tmm.coh_tmm` multiplies unscaled 2 x 2 transfer matrices (and clips a layer's Im(delta) at 35),
while `planar.coh_tmm` composes reflection ratios through projective (Moebius) sweeps (`response.sweep`), and
`pystratify.stable` composes bounded scattering matrices. Agreement of the coherent r, t, R, T therefore checks the
owned kernels against the original arithmetic; agreement of `inc_tmm` checks that the incoherent ladder built on
them has not drifted. multilayer.app measured the same agreement over its replication corpus: <= 3.5e-12 in r and t
over 5.86 million `coh_tmm` calls (WP07, 2026-11-03).

Cases: random coherent stacks (lossless and absorbing layers and exits, normal and oblique, s and p), a prism beyond
the critical angle with evanescent layers, a 10-pair Bragg mirror, and mixed coherent/incoherent stacks (thick
lossless and absorbing incoherent layers between coherent films). ``tests/test_stable.py`` asserts every row and
skips when `tmm` is not importable.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checks import report, row  # noqa: E402
from pystratify import planar  # noqa: E402
from pystratify.stable import stable_tmm  # noqa: E402

TOL_COHERENT = 1e-11
TOL_INCOHERENT = 1e-11


def _byrnes():
    import tmm

    return tmm


def coherent_cases(seed=20261103, count=120):
    """[(name, n_list, d_list, th0, lam)] with lengths in nm."""
    rng = np.random.default_rng(seed)
    out = []
    for case in range(count):
        absorbing = case % 2 == 1
        inner = int(rng.integers(0, 9))
        n = [complex(rng.uniform(1.2, 3.5), rng.uniform(0.0, 0.8) if absorbing and rng.random() < 0.6 else 0.0)
             for _ in range(inner)]
        n_list = [complex(rng.uniform(1.0, 1.8))] + n + [complex(rng.uniform(1.0, 4.0),
                                                                rng.uniform(0.0, 1.0) if absorbing else 0.0)]
        d_list = [np.inf] + [float(rng.uniform(5.0, 400.0)) for _ in range(inner)] + [np.inf]
        th0 = float(rng.uniform(0.0, 1.45)) if case % 3 else 0.0
        out.append((f"random {'absorbing' if absorbing else 'lossless'} stack {case}", n_list, d_list, th0,
                    float(rng.uniform(300.0, 1500.0))))
    prism = [1.5, 1.0, 2.2, 1.0, 2.0 + 0.01j, 1.0]
    for angle in (45.0, 60.0, 80.0):
        out.append((f"prism at {angle:g} deg", prism, [np.inf, 350.0, 70.0, 900.0, 100.0, np.inf], np.deg2rad(angle), 633.0))
    mirror = [1.0] + [2.35, 1.46] * 10 + [1.52]
    for lam in (450.0, 550.0, 650.0):
        out.append((f"10-pair Bragg mirror at {lam:g} nm", mirror, [np.inf] + [58.5, 94.2] * 10 + [np.inf], 0.3, lam))
    return out


def incoherent_cases(seed=11, count=60):
    """[(name, n_list, d_list, c_list, th0, lam)]: coherent films around thick incoherent layers."""
    rng = np.random.default_rng(seed)
    out = []
    for case in range(count):
        inner = int(rng.integers(1, 7))
        n, d, c = [], [], []
        for _ in range(inner):
            if rng.random() < 0.35:
                n.append(complex(rng.uniform(1.3, 2.0), rng.uniform(0.0, 1e-4) if rng.random() < 0.5 else 0.0))
                d.append(float(rng.uniform(2e4, 1e6)))
                c.append("i")
            else:
                n.append(complex(rng.uniform(1.2, 3.0), rng.uniform(0.0, 0.3) if rng.random() < 0.4 else 0.0))
                d.append(float(rng.uniform(10.0, 300.0)))
                c.append("c")
        n_list = [1.0 + 0j] + n + [complex(rng.uniform(1.0, 3.5), rng.uniform(0.0, 0.5) if case % 2 else 0.0)]
        th0 = float(rng.uniform(0.0, 1.2)) if case % 3 else 0.0
        out.append((f"mixed stack {case}", n_list, [np.inf] + d + [np.inf], ["i"] + c + ["i"], th0,
                    float(rng.uniform(400.0, 1200.0))))
    return out


def compare():
    """{label: max deviation} of PyStratify against tmm 0.2.0."""
    tmm = _byrnes()
    out = {"coh_tmm r, t": 0.0, "coh_tmm R, T": 0.0, "stable_tmm r, t": 0.0, "stable_tmm R, T": 0.0,
           "inc_tmm R, T": 0.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, n_list, d_list, th0, lam in coherent_cases():
            for pol in ("s", "p"):
                ref = tmm.coh_tmm(pol, n_list, d_list, th0, lam)
                ours = planar.coh_tmm(pol, n_list, d_list, th0, lam)
                stab = stable_tmm(pol, list(zip(n_list[1:-1], d_list[1:-1])), n_list[0], n_list[-1], th0, lam)
                out["coh_tmm r, t"] = max(out["coh_tmm r, t"], abs(ours["r"] - ref["r"]), abs(ours["t"] - ref["t"]))
                out["coh_tmm R, T"] = max(out["coh_tmm R, T"], abs(ours["R"] - ref["R"]), abs(ours["T"] - ref["T"]))
                out["stable_tmm r, t"] = max(out["stable_tmm r, t"], abs(complex(stab["r"]) - ref["r"]),
                                             abs(complex(stab["t"]) - ref["t"]))
                out["stable_tmm R, T"] = max(out["stable_tmm R, T"], abs(float(stab["R"]) - ref["R"]),
                                             abs(float(stab["T"]) - ref["T"]))
        for _, n_list, d_list, c_list, th0, lam in incoherent_cases():
            for pol in ("s", "p"):
                ref = tmm.inc_tmm(pol, n_list, d_list, c_list, th0, lam)
                ours = planar.inc_tmm(pol, n_list, d_list, c_list, th0, lam)
                out["inc_tmm R, T"] = max(out["inc_tmm R, T"], abs(ours["R"] - ref["R"]), abs(ours["T"] - ref["T"]))
    return {key: float(value) for key, value in out.items()}


def rows(results):
    coherent = f"{len(coherent_cases())} coherent stacks (random, prism beyond critical, Bragg mirror) x s, p"
    mixed = f"{len(incoherent_cases())} mixed coherent/incoherent stacks x s, p"
    return [
        row(f"planar.coh_tmm vs tmm.coh_tmm, {coherent}", "max |r - r_tmm|, |t - t_tmm|", results["coh_tmm r, t"],
            TOL_COHERENT),
        row(f"planar.coh_tmm vs tmm.coh_tmm, {coherent}", "max |R - R_tmm|, |T - T_tmm|", results["coh_tmm R, T"],
            TOL_COHERENT),
        row(f"stable.stable_tmm vs tmm.coh_tmm, {coherent}", "max |r - r_tmm|, |t - t_tmm|", results["stable_tmm r, t"],
            TOL_COHERENT),
        row(f"stable.stable_tmm vs tmm.coh_tmm, {coherent}", "max |R - R_tmm|, |T - T_tmm|", results["stable_tmm R, T"],
            TOL_COHERENT),
        row(f"planar.inc_tmm vs tmm.inc_tmm, {mixed}", "max |R - R_tmm|, |T - T_tmm|", results["inc_tmm R, T"],
            TOL_INCOHERENT),
    ]


if __name__ == "__main__":
    import tmm  # noqa: F401  (fail early with the import error)

    version = getattr(tmm, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as _version

            version = _version("tmm")
        except Exception:  # noqa: BLE001 - the report names it when it can
            version = "unknown"
    report("Films: PyStratify against Byrnes's tmm (lineage)",
           f"Generated by `python benchmarks/byrnes_tmm.py` (tmm {version}); `tests/test_stable.py` asserts every row "
           "(skipped without `tmm`). **Lineage**: `planar.py` keeps `tmm`'s interface, conventions and incoherent "
           "ladder; only the coherent kernels (projective sweeps in `planar.coh_tmm`, scattering matrices in "
           "`pystratify.stable`, unscaled transfer matrices in `tmm`) are independent of it.", rows(compare()))
