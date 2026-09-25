"""Convergence and accuracy benchmarks from the multilayered-sphere literature.

    python benchmarks/convergence.py            # prints a Markdown report
    python benchmarks/convergence.py --quick    # skips the slowest cases

Every case is checked against something computed another way (see
tests/references.py): mpmath sums of textbook Mie theory, BHMIE, 60-digit
transfer matrices, energy conservation, or the same quantity at a much higher
truncation.  Sources of the cases:

* [MLR] Majic & Le Ru, Appl. Opt. 59, 1293 (2020): emitter near a sphere;
* [WW91] Wu & Wang, Radio Sci. 26, 1393 (1991): coated and graded spheres;
* [W97] Wu, Guo, Ren, Gouesbet & Grehan, Appl. Opt. 36, 5188 (1997): many
  layers, large size parameters, rainbow of a graded droplet;
* [Y23], [Y24] Yuan, Zhu & Zhu, IEEE TAP 71, 5178 (2023) and Opt. Express 32,
  3062 (2024): layered magnetic sphere with a dipole, Luneburg lens.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pystratify as ps  # noqa: E402
from tests.mp_reference import coefficients  # noqa: E402
from tests.references import bhmie, sphere_decay_rates, sphere_extinction  # noqa: E402

AG, SI = 0.093 + 4j, 3.87 + 0.016j  # [MLR] at 633 nm


def timed(f, *a, **k):
    t = time.perf_counter()
    out = f(*a, **k)
    return out, time.perf_counter() - t


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) / np.asarray(b) - 1)))


def decay_near_spheres(quick):
    print("## Decay rates near a sphere [MLR]\n")
    print("Silver (n = 0.093 + 4i) or silicon (3.87 + 0.016i) at 633 nm, emitter in air a gap d from the")
    print("surface; reference: [MLR] Eqs. 34-37 in mpmath.  Errors are the worst of both orientations.\n")
    print("| a (nm) | d (nm) | tol | orders | total | radiative | nonradiative | balance | time |")
    print("|---|---|---|---|---|---|---|---|---|")
    cases = [(10, AG, 5), (100, AG, 5), (50, AG, 1), (250, SI, 5), (1000, AG, 5), (100, AG, 0.5), (50, AG, 0.25)]
    if not quick:
        cases.append((1000, AG, 1))
    for a, m, d in cases:
        q = (a / (a + d)) ** 2
        orders = int(np.log(1e-12 * (1 - q)) / np.log(q) * 1.3 + 200)
        t_perp, t_par, r_perp, r_par = sphere_decay_rates(a, m, 633.0, a + d, orders=orders, dps=30)
        name = "Si" if m == SI else "Ag"
        for tol in (1e-6, 1e-9):
            r, dt = timed(ps.decay_rates, [a], [m, 1.0], 633.0, [a + d], tol=tol)
            print(
                f"| {a} {name} | {d} | {tol:g} | {r.orders_used[0]} | {rel(r.total[0], [t_perp, t_par]):.1e} "
                f"| {rel(r.radiative[0], [r_perp, r_par]):.1e} "
                f"| {rel(r.nonradiative[0], [t_perp - r_perp, t_par - r_par]):.1e} "
                f"| {r.balance_error.max():.1e} | {dt:.2f} s |"
            )
    print()


def layered_magnetic_dipole():
    print("## Dipole in a layered magnetic sphere [Y24, Table 1]\n")
    print("a = (1, 11), (eps, mu) = (2, 8), (1, 1), (8, 2), k0 = 1; lossless, so total = radiative")
    print("(the balance column) is the check.  Close to an interface the total, from the LDOS, carries")
    print("the rounding of the reactive near field (~eps x its size), as [MLR] note.\n")
    eps, mu = np.array([2, 1, 8.0]), np.array([8, 1, 2.0])
    r = np.array([1.001, 1.01, 1.1, 1.29, 5.0, 10.9, 10.99])
    rates, dt = timed(
        ps.decay_rates, [1.0, 11.0], np.sqrt(eps * mu), 2 * np.pi, r, mu=mu, tol=1e-9, normalization="shell"
    )
    print("| r | orders | converged | total (perp) | balance |")
    print("|---|---|---|---|---|")
    for i, ri in enumerate(r):
        print(
            f"| {ri} | {rates.orders_used[i]} | {rates.converged[i]} | {rates.total[i, 0]:.10f} "
            f"| {rates.balance_error[i].max():.1e} |"
        )
    print(f"\nAll seven positions: {dt:.2f} s.\n")


def large_spheres(quick):
    print("## Large homogeneous spheres\n")
    print("Q_ext against BHMIE (absorbing) or mpmath (weakly absorbing, where BHMIE itself loses digits).\n")
    print("| x | m | reference | dQ_ext | dQ_sca | time |")
    print("|---|---|---|---|---|---|")
    for x in (100, 1000, 5000, 20000):
        for m in (1.33 + 1e-8j, 1.5 + 0.5j, 0.1 + 4j, 10 + 10j):
            weak = m.imag < 1e-3
            if weak and (x > 1000 or quick and x > 100):
                continue
            cs, dt = timed(lambda: ps.cross_sections(ps.solve([x], [m, 1.0], 2 * np.pi)))
            if weak:
                orders = int(x + 4 * x ** (1 / 3) + 2)
                ref, name, d_sca = sphere_extinction(x, m, orders), "mpmath", "-"
            else:
                ref, ref_sca = bhmie(x, m)
                name, d_sca = "BHMIE", f"{rel(cs.q_sca, ref_sca):.1e}"
            print(f"| {x} | {m} | {name} | {rel(cs.q_ext, ref):.1e} | {d_sca} | {dt:.2f} s |")
    print()


def many_layers(quick):
    print("## Many layers [W97, WW91, Y24]\n")
    print("| case | layers | x | check | value | time |")
    print("|---|---|---|---|---|---|")
    lam = 2 * np.pi
    for x, N in ((100, 100), (1000, 300)) if quick else ((100, 100), (1000, 1000)):
        m = 1.33 + 0.001j
        cs, dt = timed(lambda: ps.cross_sections(ps.solve(np.linspace(x / N, x, N), [m] * N + [1.0], lam)))
        single = ps.cross_sections(ps.solve([x], [m, 1.0], lam))
        print(
            f"| identical layers = one sphere [W97] | {N} | {x} | dQ_ext | {rel(cs.q_ext, single.q_ext):.1e} | {dt:.2f} s |"
        )
    for xl, N in ((100, 100), (1000, 300)) if quick else ((100, 100), (1000, 1000)):
        i = np.arange(1, N + 1)
        n = 1.33 + 0.5 * (1.01 * 1.33 - 1.33) * (1 - np.cos(np.pi * (i - 1) / (N - 1)))
        radii = 0.001 * xl + (xl - 0.001 * xl) * (i - 1) / (N - 1)
        cs, dt = timed(lambda: ps.cross_sections(ps.solve(radii, list(n) + [1.0], lam)))
        print(
            f"| Kai-Massoli profile, lossless [W97] | {N} | {xl} | Q_abs/Q_ext | {abs(cs.q_abs[0]) / cs.q_ext[0]:.1e} | {dt:.2f} s |"
        )
    N, b = 50, 5.0
    radii = np.linspace(b / N, b, N)
    n = 1.2 / (1 + 0.253 * (radii - b / N / 2) ** 2 / b**2)
    sol = ps.solve(radii, list(n) + [1.0], lam, l_max=16)
    worst = 0.0
    for order in (1, 5, 10, 16):
        for pol in (ps.TM, ps.TE):
            regular, _ = coefficients(list(radii), list(n) + [1.0], [1.0] * (N + 1), lam, order, te=pol == ps.TE)
            worst = max(worst, abs(np.exp(sol.log_t[pol, 0, order - 1]) / complex(regular[-1][1]) - 1))
    print(f"| Cauchy profile [WW91] | {N} | {b} | T vs 60-digit | {worst:.1e} | |")
    lam, R, N = 0.5145, 125.0, 128
    radii = np.linspace(R / N, R, N)
    n = list(1.33 + 0.03 * (radii - R / N / 2) / R) + [1.0]
    sol, dt = timed(ps.solve, radii, n, lam)
    more = ps.solve(radii, n, lam, l_max=sol.orders.size + 200)
    theta = np.radians(np.linspace(130, 150, 41))
    change = rel(np.abs(ps.scattering_amplitudes(sol, theta)[0]), np.abs(ps.scattering_amplitudes(more, theta)[0]))
    print(
        f"| rainbow of a graded droplet, 250 um [W97] | {N} | {2 * np.pi * R / lam:.0f} | S(130-150 deg) vs l_max + 200 | {change:.1e} | {dt:.2f} s |"
    )
    N, R = 300, 20.0
    radii = np.linspace(R / N, R, N)
    n = np.sqrt(2 - ((radii - R / N / 2) / R) ** 2)
    cs, dt = timed(lambda: ps.cross_sections(ps.solve(radii, list(n) + [1.0], 2 * np.pi)))
    print(f"| Luneburg lens [Y24] | {N} | {R:.0f} | Q_abs/Q_ext | {abs(cs.q_abs[0]) / cs.q_ext[0]:.1e} | {dt:.2f} s |")
    print()


def fields_away_from_the_sphere():
    print("## Near field and energy density away from the sphere\n")
    print("50-nm gold-like sphere (n = 0.25 + 3i) in water, 600 nm, default (Wiscombe) truncation, 5 orders;")
    print("error of |E|^2 against 120 orders.\n")
    rad, ref, lam = [50.0], [0.25 + 3.0j, 1.33], 600.0
    coarse, fine = ps.solve(rad, ref, lam), ps.solve(rad, ref, lam, l_max=120)
    near = ps.solve(rad, ref, lam, regime="near")
    r = np.array([55.0, 100.0, 400.0, 1000.0, 5000.0])
    x, z = r * np.sin(0.3), r * np.cos(0.3)
    exact = ps.near_field(fine, x, 0 * r, z).intensity_e
    print("| r (nm) | near_field, default | near_field, regime='near' | energy_density, default |")
    print("|---|---|---|---|")
    e_coarse = ps.near_field(coarse, x, 0 * r, z).intensity_e
    e_near = ps.near_field(near, x, 0 * r, z).intensity_e
    d_coarse, d_fine = ps.energy_density(coarse, r).intensity_e, ps.energy_density(fine, r).intensity_e
    for i, ri in enumerate(r):
        print(
            f"| {ri:.0f} | {abs(e_coarse[i] / exact[i] - 1):.1e} | {abs(e_near[i] / exact[i] - 1):.1e} "
            f"| {abs(d_coarse[i] / d_fine[i] - 1):.1e} |"
        )
    print()


def main():
    quick = "--quick" in sys.argv
    warnings.simplefilter("ignore", RuntimeWarning)
    print("# PyStratify convergence benchmarks\n")
    print(f"Generated by `python benchmarks/convergence.py{' --quick' if quick else ''}`; one core.\n")
    decay_near_spheres(quick)
    layered_magnetic_dipole()
    large_spheres(quick)
    many_layers(quick)
    fields_away_from_the_sphere()


if __name__ == "__main__":
    main()
