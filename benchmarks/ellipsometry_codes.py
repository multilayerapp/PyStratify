"""Ellipsometry against pyElli: the forward model, the file readers, and a fit of a real measurement.

    pip install -e ".[benchmarks]" lmfit
    python benchmarks/ellipsometry_codes.py      # prints a Markdown report (ELLIPSOMETRY_CODES.md)

pyElli (Müller, Dobener et al.) is an ellipsometry code written apart from PyStratify, with 2x2 and
Berreman 4x4 solvers; it reports Psi/Delta in the same instrument convention as
``pystratify.ellipsometry``, so no conversion is applied here.

1. **Forward.** Psi and Delta of dispersive stacks built from ``pystratify.dispersion`` models,
   handed to pyElli as the same eps on the same grid (``elli.TableEpsilon``), so only the stack
   optics are compared.
2. **Readers.** ``read_woollam`` and ``read_spectraray`` against pyElli's readers on the two
   measured files in ``examples/data/ellipsometry``.
3. **Fit.** pyElli's TiO2 example (ALD TiO2 on 276 nm SiO2 on Si, SENTECH, 70.06 deg, 400-800 nm):
   a Cauchy film's thickness and three coefficients, fitted by ``pystratify.ellipsometry.fit`` and
   by lmfit on pyElli's model, both minimising the same unweighted Psi/Delta sum of squares
   (pyElli's own decorator fits rho instead, a different objective with a different minimum).
   Silicon is pyElli's Aspnes table sampled at the data wavelengths, given to both codes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pystratify import dispersion as dm  # noqa: E402
from pystratify import ellipsometry as el  # noqa: E402

DATA = ROOT / "examples" / "data" / "ellipsometry"
WAVELENGTHS_NM = np.linspace(250.0, 1450.0, 121)  # the Green 2008 silicon table
ANGLES_DEG = (45.0, 60.0, 70.0, 75.0)

SILICON = dm.Table.from_refractiveindex(ROOT / "crosscheck/classics/data/Si_Green-2008.yml")
FILM = dm.EpsilonInf(1.1) + dm.TaucLorentz(A=150, E0=3.8, C=2.0, Eg=1.7)
STACKS = {
    "Tauc-Lorentz film, 25% void roughness, on Si": [
        (dm.Bruggeman(FILM, 1.0, 0.75), 3.0), (FILM, 80.0), (SILICON, None)],
    "Cauchy film on thermal oxide on Si": [
        (dm.Cauchy(A=2.1, B=0.02, C=0.001, k_amplitude=0.002, k_exponent=1.0, band_edge_nm=400), 150.0),
        (dm.Sellmeier(B=(0.6961663, 0.4079426, 0.8974794), C=(0.0684043**2, 0.1162414**2, 9.896161**2)), 300.0),
        (SILICON, None)],
    "semi-transparent Drude metal on glass": [
        (dm.EpsilonInf(1.0) + dm.DrudeEnergy(Ep=8.9, Gamma=0.07) + dm.Lorentz(A=2.0, E0=3.0, B=1.0), 15.0),
        (dm.Constant(1.5), None)],
    "Cody-Lorentz film, Maxwell Garnett inclusions, on Si": [
        (dm.MaxwellGarnett(dm.EpsilonInf(2.0), dm.Constant(0.2, 3.0), 0.05), 40.0),
        (dm.EpsilonInf(1.2) + dm.CodyLorentz(A=80, E0=3.7, Gamma=2.3, Eg=1.6, Ep=0.9, Et=1.8, Eu=0.05), 60.0),
        (SILICON, None)],
}


def _elli():
    import elli

    return elli


def _elli_material(model):
    elli = _elli()
    return elli.IsotropicMaterial(elli.TableEpsilon(lbda=WAVELENGTHS_NM, epsilon=model.epsilon(WAVELENGTHS_NM)))


def _wrap(value):
    return np.abs((value + 180.0) % 360.0 - 180.0)


def forward():
    elli = _elli()
    worst = {}
    for name, stack in STACKS.items():
        layers, substrate = stack[:-1], stack[-1][0]
        sample = el.Sample([el.Layer(model, thickness) for model, thickness in layers], substrate=substrate)
        structure = elli.Structure(elli.AIR, [elli.Layer(_elli_material(model), thickness)
                                              for model, thickness in layers], _elli_material(substrate))
        ours = el.simulate(sample, WAVELENGTHS_NM, ANGLES_DEG)
        for solver_name, solver in (("2x2", elli.Solver2x2), ("4x4", elli.Solver4x4)):
            psi, delta = 0.0, 0.0
            for row, angle in enumerate(ANGLES_DEG):
                theirs = structure.evaluate(WAVELENGTHS_NM, angle, solver=solver)
                psi = max(psi, np.max(np.abs(ours.psi[row] - theirs.psi)))
                delta = max(delta, np.max(_wrap(ours.delta[row] - theirs.delta)))
            worst[(name, solver_name)] = (psi, delta)
    return worst


def readers():
    elli = _elli()
    rows = {}
    woollam, reference = el.read_woollam(DATA / "complete_ease_example.dat"), \
        elli.read_woollam_psi_delta(DATA / "complete_ease_example.dat")
    spectraray, reference_s = el.read_spectraray(DATA / "TiO2_400cycles.txt"), \
        elli.read_spectraray_psi_delta(DATA / "TiO2_400cycles.txt")
    for name, ours, theirs in (("Woollam CompleteEASE", woollam, reference),
                               ("SENTECH SpectraRay", spectraray, reference_s)):
        difference = [0.0, 0.0, 0.0]
        for angle in ours.angles:
            mine, table = ours.select(angles=[angle]), theirs.loc[angle]
            assert mine.wavelength.size == len(table)
            difference = [max(difference[0], np.max(np.abs(mine.wavelength - table.index.to_numpy()))),
                          max(difference[1], np.max(np.abs(mine.psi - table["Ψ"].to_numpy()))),
                          max(difference[2], np.max(_wrap(mine.delta - table["Δ"].to_numpy())))]
        rows[name] = (len(ours), *difference)
    return rows


def _tio2_fits():
    """Both fits of pyElli's TiO2 measurement: (data, PyStratify's sample builder, its fit, lmfit's result, pyElli's
    Psi/Delta for a parameter set)."""
    import lmfit

    elli = _elli()
    data = el.read_spectraray(DATA / "TiO2_400cycles.txt").select(wavelength=(400, 800))
    wavelength = data.wavelength
    si_index = elli.db.RII().get_mat("Si", "Aspnes").get_refractive_index(wavelength)[:, 0, 0]
    silicon = dm.Table(wavelength, si_index.real, si_index.imag)
    start = {"n0": 2.236, "n1": 451.0, "n2": 251.0, "d": 20.0}

    def sample(p):  # pyElli's Cauchy: n0 + 1e2 n1 / lambda[nm]^2 + 1e7 n2 / lambda[nm]^4
        film = dm.Cauchy(A=p["n0"], B=1e-4 * p["n1"], C=1e-5 * p["n2"])
        return el.Sample([el.Layer(film, p["d"]), el.Layer(dm.Cauchy(1.452, 36e-4), 276.36)], substrate=silicon)

    params = el.Parameters()
    for name, value in start.items():
        params.add(name, value, *((0.0, 40000.0) if name == "d" else ()))
    ours = el.fit(sample, params, data)

    si_elli = elli.IsotropicMaterial(elli.Table(lbda=wavelength, n=si_index))

    def elli_spectra(p):
        film = elli.Cauchy(p["n0"], p["n1"], p["n2"]).get_mat()
        oxide = elli.Cauchy(1.452, 36.0, 0).get_mat()
        return elli.Structure(elli.AIR, [elli.Layer(film, p["d"]), elli.Layer(oxide, 276.36)], si_elli) \
            .evaluate(wavelength, 70.06, solver=elli.Solver2x2)

    def residual(p):
        result = elli_spectra(p)
        return np.concatenate([result.psi - data.psi, (result.delta - data.delta + 180) % 360 - 180])

    lp = lmfit.Parameters()
    for name, value in start.items():
        lp.add(name, value, **({"min": 0, "max": 40000} if name == "d" else {}))
    theirs = lmfit.minimize(residual, lp, method="least_squares", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return data, sample, ours, theirs, elli_spectra


def tio2_fit():
    _, _, ours, theirs, _ = _tio2_fits()
    rows = {name: (ours.values[name], theirs.params[name].value, ours.stderr[name], theirs.params[name].stderr)
            for name in ("n0", "n1", "n2", "d")}
    return rows, ours.mse, float(np.sqrt(theirs.redchi))


def tio2_fit_spectra():
    """The measurement and each fit's Psi, Delta at its wavelengths (benchmarks/figures.py draws them)."""
    data, sample, ours, theirs, elli_spectra = _tio2_fits()
    mine = el.simulate(sample(ours.values), data.wavelength, 70.06)
    reference = elli_spectra({name: theirs.params[name].value for name in ("n0", "n1", "n2", "d")})
    return data, mine, reference, ours, theirs


def backside_phase_average(n, d, wavelength_nm, angle_deg, samples=64):
    """|r_s|^2, |r_p|^2 and r_s conj(r_p) of the coherent stack (indices n, thicknesses d in nm, the thick substrate
    second to last) averaged over one period of the substrate's round-trip phase: the incoherent backside sum by brute
    force, independent of the partial-wave chain in ``pystratify.ellipsometry`` (tests/test_ellipsometry.py and
    benchmarks/figures.py call it)."""
    from pystratify import planar

    n_sub = n[-2]
    cos_t = np.sqrt(1 - (n[0] * np.sin(np.radians(angle_deg)) / n_sub) ** 2)
    period = wavelength_nm / (2 * n_sub * cos_t)
    sums = np.zeros(3, complex)
    for k in range(samples):
        thick = [*d[:-2], d[-2] + k * period / samples, d[-1]]
        r_s = planar.coh_tmm("s", n, thick, np.radians(angle_deg), wavelength_nm)["r"]
        r_p = planar.coh_tmm("p", n, thick, np.radians(angle_deg), wavelength_nm)["r"]
        sums += [abs(r_s) ** 2, abs(r_p) ** 2, r_s * np.conj(r_p)]
    return sums / samples


def report():
    print("# PyStratify ellipsometry against pyElli\n")
    print("Generated by `python benchmarks/ellipsometry_codes.py`; what each part compares is in its docstring.\n")
    print(f"## Forward Ψ, Δ\n\nWorst absolute difference in degrees over {WAVELENGTHS_NM[0]:g}–{WAVELENGTHS_NM[-1]:g} nm "
          f"({WAVELENGTHS_NM.size} wavelengths) at {', '.join(f'{a:g}' for a in ANGLES_DEG)}°.\n")
    print("| stack | pyElli solver | Ψ | Δ |\n|---|---|---|---|")
    for (name, solver), (psi, delta) in forward().items():
        print(f"| {name} | {solver} | {psi:.1e} | {delta:.1e} |")
    print("\n## File readers\n\n| format | points | wavelength (nm) | Ψ | Δ |\n|---|---|---|---|---|")
    for name, (count, wavelength, psi, delta) in readers().items():
        print(f"| {name} | {count} | {wavelength:.1e} | {psi:.1e} | {delta:.1e} |")
    rows, mse, mse_lmfit = tio2_fit()
    print("\n## Fit of a measurement: pyElli's TiO₂ example\n")
    print("| parameter | PyStratify | pyElli + lmfit | relative difference | standard error (PyStratify / lmfit) |")
    print("|---|---|---|---|---|")
    for name, (value, reference, error, error_ref) in rows.items():
        print(f"| {name} | {value:.8g} | {reference:.8g} | {abs(value - reference) / abs(reference):.1e} | "
              f"{error:.4g} / {error_ref:.4g} |")
    print(f"\nMSE (unweighted, degrees): PyStratify {mse:.10g}, pyElli + lmfit √χ²ᵣ {mse_lmfit:.10g}.")


if __name__ == "__main__":
    report()
