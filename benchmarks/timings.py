"""The timings that README.md ("Speed") and docs/NUMERICS.md ("Typical timings on one core") quote, as a run.

    python benchmarks/timings.py > benchmarks/TIMINGS.md     # ~1 min; on an otherwise idle machine
    python benchmarks/timings.py --quick                      # 3 calls per task: a smoke test, not a report

Both tables had named their tasks in a few words with no run behind them: no date, machine or versions. Each task is
written out here, from the setup a document of this repository already gives where one exists (each says which):
the worked example of docs/GUIDE.md (a SiO2@Au 50/55 nm nanoshell in water, its chiral shell, its emitters), the
examples/ scripts, and benchmarks/convergence.py. Where a row does not say which particle, the GUIDE's nanoshell is it.
Gold is the examples' analytic model (examples/_common.py, au-elm-2007e) at the task's wavelength, or the 614-nm index
of benchmarks/convergence.py where the task is taken from there.

Method: one thread (OMP, OpenBLAS, MKL and vecLib set to 1 before NumPy loads); each task's inputs (indices, grids,
positions) are built once, outside the clock; the PyStratify calls are made once to warm up and then timed with
``time.perf_counter`` over enough calls to fill ~3 s (at least 5, at most 51); the report gives the median, the
fastest and the slowest call. A number measured under load means nothing: run it with nothing else running (the
report records the load average before and after).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"  # before NumPy is imported: its BLAS reads them once

import argparse  # noqa: E402
import datetime  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import pystratify as ps  # noqa: E402
from _common import gold  # noqa: E402  (examples/_common.py: au-elm-2007e)

FILL_SECONDS, MIN_CALLS, MAX_CALLS = 3.0, 5, 51
SILICA, WATER = 1.45, 1.33
SPECTRUM = np.linspace(400.0, 1000.0, 601)  # nm: 601 wavelengths at 1 nm, the grid of examples/multipole_spectra.py
GUIDE_WAVELENGTH = 700.0  # docs/GUIDE.md: wavelength[200] of its linspace(500, 900, 401)
NANOSHELL = [50.0, 55.0]  # docs/GUIDE.md: SiO2 core, Au shell, in water
CHIRAL_NANOSHELL = [50.0, 55.0, 60.0]  # docs/GUIDE.md: the same nanoshell under a 5-nm chiral shell (n 1.5, kappa 1e-3)
GRID = np.meshgrid(np.linspace(0, np.pi, 181), np.linspace(0, 2 * np.pi, 361), indexing="ij")  # docs/GUIDE.md
QUADRUPOLE = np.diag([-1.0, -1.0, 2.0])  # examples/quadrupole_emitter.py: radial (along z)


@dataclass
class Task:
    key: str
    what: str  # the task in one line
    source: str  # where its setup comes from
    call: object  # () -> result, the timed PyStratify calls
    check: object  # result -> str, what the call computed (so a reader can see it is the task described)


def spectrum(radii, layers, host):
    """(W, N + 1) indices over SPECTRUM; `layers` are numbers or 'Au'."""
    columns = [gold(SPECTRUM) if m == "Au" else np.full(SPECTRUM.size, complex(m)) for m in layers]
    return np.stack(columns + [np.full(SPECTRUM.size, complex(host))], axis=1)


def q_ext_peak(q, wavelengths):
    i = int(np.argmax(q))
    return f"max Q_ext {q[i]:.3f} at {wavelengths[i]:.0f} nm"


def nanoshell_spectrum():
    n = spectrum(NANOSHELL, [SILICA, "Au"], WATER)
    return Task(
        "nanoshell spectrum", "601-wavelength spectrum (400-1000 nm), Au nanoshell: solve + cross_sections",
        "docs/GUIDE.md nanoshell, SiO2@Au 50/55 nm in water; the 601-point grid of examples/multipole_spectra.py",
        lambda: ps.cross_sections(ps.solve(NANOSHELL, n, SPECTRUM)),
        lambda cs: f"{cs.orders.size} orders; {q_ext_peak(cs.q_ext, SPECTRUM)}",
    )


def matryoshka_spectrum():
    radii = [4.0, 7.0, 10.0, 16.0, 34.0, 42.0]
    n = spectrum(radii, [SILICA if i % 2 == 0 else "Au" for i in range(len(radii))], WATER)
    return Task(
        "matryoshka spectrum", "601-wavelength spectrum (400-1000 nm), 6-interface matryoshka: solve + cross_sections",
        "examples/near_field_map.py, third particle: SiO2/Au {4, 7, 10, 16, 34, 42} nm in water (Meng et al. 2017)",
        lambda: ps.cross_sections(ps.solve(radii, n, SPECTRUM)),
        lambda cs: f"{cs.orders.size} orders; {q_ext_peak(cs.q_ext, SPECTRUM)}",
    )


def large_sphere():
    m, x = 1.33 + 1e-8j, 500.0
    return Task(
        "x = 500 sphere", "sphere of size parameter 500, all orders (Wiscombe truncation): solve + cross_sections",
        "benchmarks/convergence.py large_spheres: x = 500, m = 1.33 + 1e-8i (a water droplet), wavelength 2 pi",
        lambda: ps.cross_sections(ps.solve([x], [m, 1.0], 2 * np.pi)),
        lambda cs: f"{cs.orders.size} orders; Q_ext {cs.q_ext[0]:.6f}",
    )


def near_field_map():
    radii, lam = [4.0, 7.0, 10.0, 16.0, 34.0, 42.0], 690.0
    n = [SILICA if i % 2 == 0 else gold(lam) for i in range(len(radii))] + [WATER]
    grid = np.linspace(-70.0, 70.0, 300)
    x, y = np.meshgrid(grid, grid)
    l_max = ps.truncation_order(radii[-1], WATER, lam, "near")

    def call():
        return ps.near_field(ps.solve(radii, n, lam, l_max=l_max), x, y, 0 * x)

    return Task(
        "near-field map", "300 x 300 near-field map (E and H) at 690 nm, z = 0 plane, +-70 nm: solve + near_field",
        "examples/near_field_map.py, its 6-interface matryoshka and window, on 300 x 300 points instead of 281 x 281",
        call, lambda f: f"{l_max} orders; max |E|^2 {f.intensity_e.max():.1f}",
    )


def nanoshell_decay(gap_from):
    au, lam = 0.27 + 2.93j, 614.0  # benchmarks/convergence.py
    r = NANOSHELL[-1] + np.linspace(gap_from, 25.0, 100)

    def check(rates):
        return (f"orders up to {int(np.max(rates.orders_used))}; all converged {bool(np.all(rates.converged))}; "
                f"max balance error {np.max(rates.balance_error):.1e}")

    return Task(
        f"decay rates from {gap_from:g} nm",
        f"decay_rates at 100 emitter positions, gaps {gap_from:g}-25 nm (linear) outside the Au nanoshell, 614 nm",
        "benchmarks/convergence.py: SiO2@Au 50/55 nm in water, Au = 0.27 + 2.93i at 614 nm; default tol 1e-6",
        lambda: ps.decay_rates(NANOSHELL, [SILICA, au, WATER], lam, r), check,
    )


def chiral_shell_spectrum():
    n = spectrum(CHIRAL_NANOSHELL, [SILICA, "Au", 1.5], WATER)
    kappa = np.zeros_like(n)
    kappa[:, 2] = 1e-3
    return Task(
        "chiral-shell spectrum",
        "601-wavelength spectrum (400-1000 nm), Au nanoshell with a chiral shell, both helicities: solve_chiral + "
        "helicity_cross_sections",
        "docs/GUIDE.md chiral nanoshell: SiO2/Au/chiral {50, 55, 60} nm, shell n 1.5, kappa 1e-3, water",
        lambda: ps.helicity_cross_sections(ps.solve_chiral(CHIRAL_NANOSHELL, n, kappa, SPECTRUM)),
        lambda cs: f"{cs.orders.size} orders; max |CD_ext| {np.max(np.abs(cs.cd_ext)):.3g} nm^2",
    )


def scattering_pattern():
    n = [SILICA, gold(GUIDE_WAVELENGTH), WATER]
    theta, phi = GRID
    return Task(
        "scattering pattern",
        "scattering pattern on a 181 x 361 (theta, phi) grid, helicity +1, one wavelength: solve + scattering_pattern",
        "docs/GUIDE.md: its nanoshell at 700 nm, polarization (1, 1j), the GUIDE's grid",
        lambda: ps.scattering_pattern(ps.solve(NANOSHELL, n, GUIDE_WAVELENGTH), theta, phi, polarization=(1, 1j)),
        lambda p: f"max directivity {np.max(p.directivity):.3f}",
    )


def dipole_pattern():
    n = [SILICA, gold(GUIDE_WAVELENGTH), WATER]
    theta, phi = GRID
    return Task(
        "dipole emission pattern", "dipole emission pattern on a 181 x 361 grid: dipole_far_field",
        "docs/GUIDE.md: x dipole at (0, 20, 60) nm beside its nanoshell, 700 nm",
        lambda: ps.dipole_far_field(NANOSHELL, n, GUIDE_WAVELENGTH, [0, 20, 60], [1, 0, 0], theta, phi),
        lambda f: f"{f.orders_used} orders; P_rad/P_0 {f.power:.4f}",
    )


def chiral_emitter_pattern():
    n = [SILICA, gold(GUIDE_WAVELENGTH), 1.5, WATER]
    theta, phi = GRID
    return Task(
        "chiral emitter pattern",
        "orientation-averaged chiral emitter inside a chiral shell, 181 x 361 grid: dipole_far_field",
        "docs/GUIDE.md: its chiral molecule (p = z, m = 0.01i z, isotropic) at r = 57.5 nm in the chiral nanoshell's "
        "chiral shell, 700 nm",
        lambda: ps.dipole_far_field(CHIRAL_NANOSHELL, n, GUIDE_WAVELENGTH, [0, 0, 57.5], [0, 0, 1], theta, phi,
                                    kappa=[0, 0, 1e-3, 0], magnetic_moment=[0, 0, 0.01j], orientation="isotropic"),
        lambda f: f"{f.orders_used} orders; g_lum {f.dissymmetry:+.4f}",
    )


def chiral_emission_rates():
    lam, radii = 600.0, [40.0, 50.0]
    n, kappa, mu = [gold(lam), 1.5 + 0.005j, 1.0], [0, 0.02 + 0.002j, 0], [1, 1 + 1e-3j, 1]
    gaps = np.geomspace(1, 60, 100)
    positions = np.column_stack([np.zeros_like(gaps), np.zeros_like(gaps), radii[-1] + gaps])

    def call():
        return ps.emission_rates(radii, n, lam, positions, [0, 0, 1.0], magnetic_moment=[0, 0, -0.1j], kappa=kappa,
                                 mu=mu, orientation="isotropic", tol=1e-8)

    return Task(
        "chiral emission_rates",
        "emission_rates, 100 positions, chiral source, orientation average, chiral shell on Au: emission_rates",
        "examples/chiral_decay.py, 'Au core, chiral shell' (40/50 nm, 600 nm), its right enantiomer (xi = +0.1) and "
        "tol 1e-8, at 100 gaps geomspace(1, 60) instead of 36",
        call, lambda r: f"{r.orders_used} orders; max balance error {np.max(r.balance_error):.1e}",
    )


def single_position_rates(gap, source):
    lam, radius = 600.0, 40.0
    n = [gold(lam), WATER]
    position = [0, 0, radius + gap]
    if source == "p + m":
        kwargs, label = {"magnetic_moment": [0, 0, 0.01j]}, "electric + magnetic dipole (p = z, m = 0.01i z)"
        moment = [0, 0, 1.0]
    else:
        kwargs, label = {"quadrupole": QUADRUPOLE}, "radial electric quadrupole"
        moment = [0, 0, 0]
    return Task(
        f"rates {source}, {gap:g} nm",
        f"emission_rates, one position {gap:g} nm from gold, {label}: emission_rates",
        "examples/quadrupole_emitter.py: Au sphere R = 40 nm in water, 600 nm, tol 1e-8",
        lambda: ps.emission_rates([radius], n, lam, position, moment, tol=1e-8, **kwargs),
        lambda r: f"{r.orders_used} orders; max balance error {np.max(r.balance_error):.1e}",
    )


def quadrupole_pattern(on_axis):
    lam, radius = 600.0, 40.0
    n = [gold(lam), WATER]
    theta, phi = GRID
    # off the axis: the same radial quadrupole 10 nm from the surface, on x instead of z
    position, q = ([0, 0, radius + 10], QUADRUPOLE) if on_axis else ([radius + 10, 0, 0], np.diag([2.0, -1.0, -1.0]))
    where = "on the grid's axis (z)" if on_axis else "off it (on x)"
    return Task(
        f"quadrupole pattern, {'on' if on_axis else 'off'} axis",
        f"quadrupole emission pattern, 181 x 361 grid, emitter {where}: dipole_far_field(quadrupole=)",
        "examples/quadrupole_emitter.py: radial quadrupole 10 nm from an Au sphere R = 40 nm in water, 600 nm",
        lambda: ps.dipole_far_field([radius], n, lam, position, [0, 0, 0], theta, phi, quadrupole=q),
        lambda f: f"{f.orders_used} orders; P_rad/P_0 {f.power:.4g}",
    )


def graphene_spectrum():
    wavelengths, radius = np.linspace(2000.0, 12000.0, 601), 25.0
    sigma = ps.graphene_conductivity(wavelengths, 0.5, 0.005, 300.0)
    return Task(
        "graphene-coated sphere",
        "601-wavelength spectrum (2-12 um), graphene-coated sphere: solve(sheets=) + cross_sections",
        "examples/monolayer_sheet.py: graphene (E_F 0.5 eV, 5 meV, 300 K) on silica (eps 2.1, R = 25 nm) in air, "
        "601 wavelengths instead of 1001",
        lambda: ps.cross_sections(ps.solve([radius], [np.sqrt(2.1), 1.0], wavelengths, sheets={0: sigma})),
        lambda cs: f"{cs.orders.size} orders; {q_ext_peak(cs.q_ext, wavelengths)}",
    )


def tasks():
    return [
        nanoshell_spectrum(), matryoshka_spectrum(), large_sphere(), near_field_map(), nanoshell_decay(0.5),
        nanoshell_decay(0.1), chiral_shell_spectrum(), scattering_pattern(), dipole_pattern(),
        chiral_emitter_pattern(), chiral_emission_rates(),
        single_position_rates(10.0, "p + m"), single_position_rates(10.0, "quadrupole"),
        single_position_rates(1.0, "p + m"), single_position_rates(1.0, "quadrupole"),
        quadrupole_pattern(True), quadrupole_pattern(False), graphene_spectrum(),
    ]


# The two tables' rows: (document, row as written, stated time, [[task keys], ...] - one group per " / " part)
ROWS = [
    ("README", "601-wavelength spectrum, gold nanoshell", "25 ms", [["nanoshell spectrum"]]),
    ("README", "sphere of size parameter 500, all orders", "8 ms", [["x = 500 sphere"]]),
    ("README", "decay rates at 100 positions 0.5–25 nm from a gold nanoshell", "0.15 s",
     [["decay rates from 0.5 nm"]]),
    ("README", "scattering pattern on a 181 × 361 (θ, φ) grid", "10 ms", [["scattering pattern"]]),
    ("README", "300 × 300 near-field map", "1.4 s", [["near-field map"]]),
    ("NUMERICS", "601-wavelength spectrum, Au nanoshell", "25 ms", [["nanoshell spectrum"]]),
    ("NUMERICS", "601-wavelength spectrum, 6-interface matryoshka", "70 ms", [["matryoshka spectrum"]]),
    ("NUMERICS", "x = 500 sphere, all orders", "8 ms", [["x = 500 sphere"]]),
    ("NUMERICS", "300 × 300 near-field map", "1.4 s", [["near-field map"]]),
    ("NUMERICS", "decay rates at 100 emitter positions 0.5–25 nm from an Au nanoshell", "0.15 s",
     [["decay rates from 0.5 nm"]]),
    ("NUMERICS", "the same from 0.1 nm (6700 orders)", "0.85 s", [["decay rates from 0.1 nm"]]),
    ("NUMERICS", "601-wavelength spectrum, Au nanoshell with a chiral shell, both helicities", "46 ms",
     [["chiral-shell spectrum"]]),
    ("NUMERICS", "scattering pattern on a 181 × 361 (θ, φ) grid", "10 ms", [["scattering pattern"]]),
    ("NUMERICS", "dipole emission pattern on a 181 × 361 grid", "90 ms", [["dipole emission pattern"]]),
    ("NUMERICS", "orientation-averaged chiral emitter inside a chiral shell, 181 × 361 grid", "0.12 s",
     [["chiral emitter pattern"]]),
    ("NUMERICS", "`emission_rates`, 100 positions, chiral source, orientation average, chiral + Au multilayer", "0.6 s",
     [["chiral emission_rates"]]),
    ("NUMERICS", "`emission_rates`, one position 10 nm / 1 nm from gold (p + m, quadrupole)", "4 ms / 25–85 ms",
     [["rates p + m, 10 nm", "rates quadrupole, 10 nm"], ["rates p + m, 1 nm", "rates quadrupole, 1 nm"]]),
    ("NUMERICS", "quadrupole emission pattern, 181 × 361 grid, emitter on / off the grid's axis", "0.15 s / 1.1 s",
     [["quadrupole pattern, on axis"], ["quadrupole pattern, off axis"]]),
    ("NUMERICS", "601-wavelength spectrum, graphene-coated sphere", "10 ms", [["graphene-coated sphere"]]),
]


def seconds(text):
    """'25 ms' -> 0.025, '0.15 s' -> 0.15; a range 'a–b ms' -> (a, b)."""
    value, unit = text.split()
    scale = 1e-3 if unit == "ms" else 1.0
    parts = [float(p) * scale for p in value.replace("–", "-").split("-")]
    return tuple(parts) if len(parts) > 1 else parts[0]


def fmt(t):
    """Two significant figures, in ms below 0.1 s, as the tables write them."""
    return f"{float(f'{t * 1e3:.2g}'):g} ms" if t < 0.1 else f"{float(f'{t:.2g}'):g} s"


def fmt_group(times):
    """One time, or a range 'a–b ms' (one unit) over a group of tasks."""
    if len(times) == 1:
        return fmt(times[0])
    lo, hi = fmt(min(times)), fmt(max(times))
    if lo.split()[1] != hi.split()[1]:
        return f"{lo}–{hi}"
    return f"{lo.split()[0]}–{hi}" if lo != hi else lo


def measure(task, quick):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        start = time.perf_counter()
        result = task.call()  # warm-up (and the result the check reads)
        first = time.perf_counter() - start
        calls = 3 if quick else int(np.clip(FILL_SECONDS / max(first, 1e-6), MIN_CALLS, MAX_CALLS))
        times = []
        for _ in range(calls):
            start = time.perf_counter()
            task.call()
            times.append(time.perf_counter() - start)
    return np.array(times), task.check(result)


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def machine():
    """Chip, cores, memory and OS, nothing that names the computer or its owner."""
    def sysctl(name):
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    if platform.system() == "Darwin":
        chip = sysctl("machdep.cpu.brand_string") or platform.processor()
        perf, eff = sysctl("hw.perflevel0.physicalcpu"), sysctl("hw.perflevel1.physicalcpu")
        cores = f"{perf} performance + {eff} efficiency cores" if perf and eff else f"{os.cpu_count()} cores"
        memory = int(sysctl("hw.memsize") or 0) / 2**30
        system = f"macOS {platform.mac_ver()[0]}"
    else:
        chip, cores = platform.processor() or platform.machine(), f"{os.cpu_count()} cores"
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except (ValueError, OSError, AttributeError):
            memory = 0
        system = f"{platform.system()} {platform.release()}"
    return f"{chip}, {cores}, {memory:.0f} GB RAM, {system} ({platform.machine()})"


def blas():
    try:
        return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:  # noqa: BLE001 - the report names it when it can
        return "unknown"


def load():
    try:
        return " ".join(f"{x:.2f}" for x in os.getloadavg())
    except OSError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="3 calls per task: a smoke test, not a report")
    args = parser.parse_args()
    load_before, started = load(), datetime.datetime.now()
    results = {}
    for task in tasks():
        print(f"{task.key} ...", end=" ", file=sys.stderr, flush=True)
        times, check = measure(task, args.quick)
        results[task.key] = (task, times, check)
        print(f"{fmt(float(np.median(times)))} ({times.size} calls)", file=sys.stderr, flush=True)
    minutes = (datetime.datetime.now() - started).total_seconds() / 60
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    src_state = "src/ as committed" if not git("status", "--porcelain", "--", "src") else "**src/ modified**"

    print("# Timings: the tables of README.md and docs/NUMERICS.md\n")
    print(f"Generated by `python benchmarks/timings.py{' --quick' if args.quick else ''}` on "
          f"{started.date().isoformat()}; README.md (\"Speed\") and docs/NUMERICS.md (\"Typical timings on one core\") "
          "quote its medians. Each task is defined in the script, from the setup a document of this repository "
          "already gives (named below); one thread, inputs built outside the clock, one warm-up call, then the "
          f"median of 5-51 calls timed with `time.perf_counter` (the run took {minutes:.1f} min).\n")
    if args.quick:
        print("**--quick: 3 calls per task, a smoke test; not the numbers the tables quote.**\n")
    print(f"* machine: {machine()}")
    print(f"* Python {platform.python_version()}, NumPy {np.__version__} (BLAS: {blas()}), SciPy {scipy.__version__}")
    print(f"* PyStratify {ps.__version__}, commit {commit} ({src_state})")
    print("* threads: OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = VECLIB_MAXIMUM_THREADS = 1")
    print(f"* load average (1, 5, 15 min): {load_before} before, {load()} after\n")

    print("## Tasks\n")
    print("| task | median | fastest – slowest (calls) | what was computed | setup from |")
    print("|---|---|---|---|---|")
    for task, times, check in results.values():
        spread = f"{fmt(float(times.min()))} – {fmt(float(times.max()))} ({times.size})"
        cells = (task.what, f"**{fmt(float(np.median(times)))}**", spread, check, task.source)
        print("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")

    print("\n## The tables' rows\n")
    print("Measured / stated is the median over the stated value; **more than 2x either way** is marked. The stated "
          "values are those the tables gave before this run (none had a run recorded).\n")
    print("| document | row | stated | measured | measured / stated |")
    print("|---|---|---|---|---|")
    for document, text, stated, groups in ROWS:
        measured, ratios = [], []
        for group, stated_part in zip(groups, stated.split(" / ")):
            times = [float(np.median(results[key][1])) for key in group]
            measured.append(fmt_group(times))
            want = seconds(stated_part)
            want = (want, want) if not isinstance(want, tuple) else want
            for t in times:  # a range is compared end to end: fastest with its low end, slowest with its high end
                ratios.append(t / (want[0] if t == min(times) else want[1]))
        worst = max(ratios, key=lambda r: max(r, 1 / r))
        ratio = "; ".join(f"{r:.2g}" for r in ratios)
        if max(worst, 1 / worst) > 2:
            ratio = f"**{ratio}**"
        print(f"| {document} | {text.replace('|', chr(92) + '|')} | {stated} | {' / '.join(measured)} | {ratio} |")


if __name__ == "__main__":
    main()
