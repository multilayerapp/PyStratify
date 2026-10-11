"""Films against tmmax, a JAX transfer-matrix code (benchmarks/MATRIX.md, films).

    pip install tmmax                       # 1.1.4; JAX
    python benchmarks/tmmax_films.py        # prints TMMAX_FILMS.md (~20 s)

tmmax (Danis et al.) computes R and T of coherent multilayers with ``jax.lax.scan`` in single precision: its
scan carry is complex64, so 64-bit JAX cannot run it and agreement stops at float32 (~1e-6). Materials come
from its own n,k database (wavelength in um; ``tmm`` takes lengths in metres); constant-index materials are
written into it as (3, N) arrays, the shape its loader reads (its ``add_material_to_nk_database`` saves
(N, 3), which the loader misreads, returning nan). It returns (angle, wavelength) arrays.

**tmmax 1.1.4 is wrong for two or more coherent layers.** Single layers agree with PyStratify to float32, but
a two-layer stack at normal incidence came out R = 0.27 where PyStratify, PyMoosh, pyElli and a
characteristic-matrix product written independently give 0.0147. Every ingredient matches (angles, k_z,
phases, Fresnel r and t); the defect is in ``cascaded_matmul.coh_matmul``: its matrix elements are shape-(1,)
arrays, so ``jnp.array([[a, b], [c, d]])`` is (2, 2, 1) and ``jnp.matmul(carry, ...)`` broadcasts it as a
stack of column vectors instead of multiplying 2 x 2 matrices; with the identity as the first carry a single
layer survives, a second does not. Already reported upstream (bahremsd/tmmax #45, #44) and fixed on its main
branch on 2026-02-11 ("matmul bug fix": squeeze, then dot), but not released: PyPI's latest is still 1.1.4
(2025-10-15), the version ``pip install tmmax`` gives. ``fixed_coh_matmul`` below builds the 2 x 2 matrix from the scalar
elements; with it tmmax agrees with PyStratify to float32 on every stack here. The report shows tmmax as
shipped and as fixed; the test asserts single layers as shipped and multilayers as fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checks import report, row  # noqa: E402
from pystratify.planar import coh_tmm  # noqa: E402

# name -> (indices from the incident medium out, inner thicknesses in nm)
STACKS = {
    "single layer n = 2.1 in air": ([1.0, 2.1, 1.0], [100.0]),
    "single absorbing layer on glass": ([1.0, 1.46 + 0.02j, 1.52], [230.0]),
    "two layers on glass": ([1.0, 2.1, 1.46 + 0.02j, 1.52], [110.0, 230.0]),
    "quarter-wave mirror, 5 pairs": ([1.0] + [2.35, 1.46] * 5 + [1.52], [58.5, 94.2] * 5),
    "metal-dielectric: 20 nm n = 0.2 + 3i under 80 nm n = 1.46": ([1.0, 1.46, 0.2 + 3j, 1.52], [80.0, 20.0]),
}
WAVELENGTHS = (500.0, 550.0, 800.0)  # nm
ANGLES_DEG = (0.0, 30.0, 60.0)


def _materials(n):
    """Constant-index materials in tmmax's database, one per distinct index."""
    import tmmax.data as data

    names = []
    for value in n:
        name = f"pystratify_{complex(value).real:.6g}_{complex(value).imag:.6g}"
        grid = np.linspace(0.2, 3.0, 50)
        np.save(f"{data.nk_data_dir}/numpy/{name}.npy", np.array([grid, np.full(50, np.real(value)), np.full(50, np.imag(value))]))
        names.append(name)
    return names


def fixed_coh_matmul(carry, phase_r_t):
    """tmmax's coherent scan step with the 2 x 2 matrix built from scalar elements."""
    import jax.numpy as jnp

    phase, r, t = phase_r_t[0], phase_r_t[1], phase_r_t[2]
    a, b = jnp.exp(-1j * phase), jnp.exp(1j * phase)
    result = carry @ (jnp.array([[a, r * a], [r * b, b]]) / t)
    return result, result


def tmmax_rt(n, d, polarization, fixed=False):
    """tmmax's (R, T) on the wavelength x angle grid."""
    import jax.numpy as jnp
    import tmmax.cascaded_matmul as cascaded
    from tmmax.tmm import tmm

    import jax

    original = cascaded.coh_matmul
    if fixed:
        cascaded.coh_matmul = fixed_coh_matmul
    jax.clear_caches()  # tmmax's jit would otherwise reuse a scan traced with the other step
    try:
        R, T = tmm(_materials(n), jnp.array(np.array(d) * 1e-9), jnp.array(np.array(WAVELENGTHS) * 1e-9),
                   jnp.array(np.radians(ANGLES_DEG)), polarization=polarization)
    finally:
        cascaded.coh_matmul = original
        jax.clear_caches()
    return np.asarray(R, float).T, np.asarray(T, float).T  # tmmax returns (angle, wavelength)


def stratify_rt(n, d, polarization):
    out = [[coh_tmm(polarization, n, [np.inf, *d, np.inf], np.radians(a), w) for a in ANGLES_DEG] for w in WAVELENGTHS]
    return np.array([[o["R"] for o in line] for line in out]), np.array([[o["T"] for o in line] for line in out])


def compare(stacks=tuple(STACKS)):
    """{(stack, polarization, version): max |R - R_PyStratify|, max |T - T_PyStratify|}."""
    out = {}
    for name in stacks:
        n, d = STACKS[name]
        for pol in ("s", "p"):
            R0, T0 = stratify_rt(n, d, pol)
            for version, fixed in (("as shipped", False), ("fixed scan step", True)):
                R, T = tmmax_rt(n, d, pol, fixed)
                out[(name, pol, version)] = (float(np.max(np.abs(R - R0))), float(np.max(np.abs(T - T0))))
    return out


TOLERANCE = 2e-6  # float32 accumulated over a few layers


def rows(results):
    """Asserted: single layers as shipped and every stack with the fixed step; shipped multilayers are shown."""
    out = []
    for (name, pol, version), (dR, dT) in results.items():
        single = len(STACKS[name][1]) == 1
        asserted = single or version == "fixed scan step"
        check = f"{name}, {pol}, tmmax {version} (3 wavelengths x 3 angles)"
        if asserted:
            out += [row(check, "max |R - R_PyStratify|", dR, TOLERANCE), row(check, "max |T - T_PyStratify|", dT, TOLERANCE)]
        else:  # the defect: shown with an always-true row so the report keeps the number
            out += [row(check, "max |R - R_PyStratify| (shown, not asserted)", dR, 0, ">="),
                    row(check, "max |T - T_PyStratify| (shown, not asserted)", dT, 0, ">=")]
    return out


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")
    report("Films: PyStratify against tmmax (JAX, float32)",
           "Generated by `python benchmarks/tmmax_films.py` (tmmax 1.1.4); `tests/test_tmmax_films.py` asserts the asserted rows. "
           "tmmax as shipped is wrong for two or more coherent layers (a broadcasting defect in its scan step, see the script); "
           "with the step fixed it agrees to float32 on every stack.", rows(compare()))
