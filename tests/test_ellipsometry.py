"""Ellipsometry: forward model against planar, backside incoherence, file readers, fits (synthetic and measured),
and pyElli (benchmarks/ellipsometry_codes.py)."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from pystratify import dispersion as dm
from pystratify import ellipsometry as el
from pystratify import planar

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "ellipsometry"
SILICON = dm.Table.from_refractiveindex(ROOT / "crosscheck/classics/data/Si_Green-2008.yml")
SILICA = dm.Sellmeier(B=(0.6961663, 0.4079426, 0.8974794), C=(0.0684043**2, 0.1162414**2, 9.896161**2))
FILM = dm.EpsilonInf(1.1) + dm.TaucLorentz(A=150, E0=3.8, C=2.0, Eg=1.7)


def coherent(pol, sample, wavelength, angle_deg):
    media = [sample.ambient, *(layer.material for layer in sample.layers), sample.substrate]
    n = [m.index(wavelength) for m in media]
    d = [np.inf, *(layer.thickness for layer in sample.layers), np.inf]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return planar.coh_tmm(pol, n, d, np.radians(angle_deg), wavelength)["r"]


@pytest.mark.parametrize("sample", [
    el.Sample([el.Layer(dm.Bruggeman(FILM, 1.0, 0.7), 4.0), el.Layer(FILM, 90.0)], substrate=SILICON),
    el.Sample([el.Layer(dm.Cauchy(1.46, 0.0035), 1200.0)], substrate=SILICON),
    el.Sample([el.Layer(dm.Constant(0.2, 3.4), 200.0)], substrate=dm.Constant(1.5)),
    el.Sample([], substrate=dm.Constant(0.05, 4.0), ambient=dm.Constant(1.33)),
])
def test_forward_is_planar_coh_tmm_in_the_instrument_convention(sample):
    wavelengths, angles = np.array([300.0, 633.0, 1300.0]), np.array([0.0, 30.0, 65.0, 75.0, 88.0])
    spectra = el.simulate(sample, wavelengths, angles)
    for i, angle in enumerate(angles):
        for j, wavelength in enumerate(wavelengths):
            r_s, r_p = coherent("s", sample, wavelength, angle), coherent("p", sample, wavelength, angle)
            assert spectra.r_s[i, j] == pytest.approx(r_s, abs=1e-14)
            assert spectra.r_p[i, j] == pytest.approx(r_p, abs=1e-14)
            rho = np.conj(r_p / r_s)  # tan(Psi) e^{i Delta}
            assert spectra.psi[i, j] == pytest.approx(np.degrees(np.arctan(abs(rho))), abs=1e-11)
            if abs(rho) > 1e-9:
                assert abs((spectra.delta[i, j] - np.degrees(np.angle(rho)) + 180) % 360 - 180) < 1e-10
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                planar_delta = np.degrees(planar.ellips(
                    [m.index(wavelength) for m in (sample.ambient, *(x.material for x in sample.layers),
                                                   sample.substrate)],
                    [np.inf, *(x.thickness for x in sample.layers), np.inf], np.radians(angle), wavelength)["Delta"])
            if abs(rho) > 1e-9:
                assert abs((spectra.delta[i, j] - (180 - planar_delta) + 180) % 360 - 180) < 1e-10
    assert np.allclose(spectra.N**2 + spectra.C**2 + spectra.S**2, 1, atol=1e-13)
    assert np.all(spectra.depolarization < 1e-12)


def test_instrument_convention_for_a_bare_dielectric():
    brewster = np.degrees(np.arctan(1.5))
    spectra = el.simulate(el.Sample([], substrate=1.5), 600.0, [10.0, brewster - 1, brewster + 1, 80.0])
    assert np.allclose(spectra.delta, [180, 180, 0, 0])
    assert el.simulate(el.Sample([], substrate=1.5), 600.0, brewster).psi == pytest.approx(0, abs=1e-6)
    # Si at 70 deg: Delta just under 180 deg, and a 2 nm oxide pulls it down by about 6 deg (the textbook picture)
    bare = el.simulate(el.Sample([], substrate=SILICON), 633.0, 70.0).delta
    oxidised = el.simulate(el.Sample([el.Layer(SILICA, 2.0)], substrate=SILICON), 633.0, 70.0).delta
    assert 178 < bare < 180 and 5 < bare - oxidised < 7


GLASS_STACK = [el.Layer(dm.Cauchy(2.0, 0.01), 120.0)]


@pytest.mark.parametrize("angle", [0.0, 45.0, 70.0])
@pytest.mark.parametrize("glass", [dm.Constant(1.52), dm.Constant(1.52, 2e-6)])
def test_backside_intensities_are_planar_inc_tmm(glass, angle):
    sample = el.Sample(GLASS_STACK, substrate=glass, substrate_thickness=1e6)
    for wavelength in (450.0, 800.0):
        ss, pp, _, _, _ = el._response(sample, np.array([wavelength]), np.array([angle]))
        n = [1.0, GLASS_STACK[0].material.index(wavelength), glass.index(wavelength), 1.0]
        d, c = [np.inf, 120.0, 1e6, np.inf], ["i", "c", "i", "i"]
        for pol, value in (("s", ss), ("p", pp)):
            assert value[0] == pytest.approx(planar.inc_tmm(pol, n, d, c, np.radians(angle), wavelength)["R"], rel=1e-12)


@pytest.mark.parametrize("angle", [20.0, 65.0])
def test_backside_cross_term_is_the_phase_average_of_the_coherent_amplitudes(angle):
    """Averaging r_s conj(r_p) of the coherent stack over one period of the substrate's round-trip phase
    is the incoherent sum: a check of the partial-wave chain, polarisation bases included."""
    glass, wavelength, thickness = dm.Constant(1.52), 633.0, 1e5
    sample = el.Sample(GLASS_STACK, substrate=glass, substrate_thickness=thickness)
    ss, pp, sp, _, _ = el._response(sample, np.array([wavelength]), np.array([angle]))
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from ellipsometry_codes import backside_phase_average

    n = [1.0, GLASS_STACK[0].material.index(wavelength), 1.52, 1.0]
    sums = backside_phase_average(n, [np.inf, 120.0, thickness, np.inf], wavelength, angle)
    assert np.allclose([ss[0], pp[0], sp[0]], sums, rtol=1e-12, atol=1e-15)
    spectra = el.simulate(sample, wavelength, angle)
    assert spectra.depolarization > 1e-5 and spectra.r_s is None


def test_an_opaque_substrate_returns_nothing_from_its_back():
    opaque = el.Sample(GLASS_STACK, substrate=dm.Constant(1.5, 0.05), substrate_thickness=1e5)
    infinite = el.Sample(GLASS_STACK, substrate=dm.Constant(1.5, 0.05))
    a, b = el.simulate(opaque, [500.0, 900.0], [60.0, 70.0]), el.simulate(infinite, [500.0, 900.0], [60.0, 70.0])
    assert np.allclose(a.psi, b.psi, atol=1e-12) and np.allclose(a.delta, b.delta, atol=1e-10)


def test_read_woollam_completeease_wvase_and_units(tmp_path):
    complete = tmp_path / "c.dat"
    complete.write_text("2nm SiO2 on Si\nVASEmethod[...]\nAngstroms\n"
                        "E\t5000.0\t70.0\t10.0\t170.0\t0.01\t0.03\n"
                        "E\t6000.0\t70.0\t11.0\t171.0\t0.02\t0.04\n"
                        "dPolE\t5000.0\t70.0\t0.5\t0.1\n"
                        "dPolE\t6000.0\t70.0\t1.5\t0.1\n"
                        "uR\t5000.0\t70.0\tinf\t1.0\n")
    data = el.read_woollam(complete)
    assert np.allclose(data.wavelength, [500, 600]) and np.allclose(data.psi_sigma, [0.01, 0.02])
    assert np.allclose(data.depolarization, [0.005, 0.015])
    N, C, S = data.ncs
    assert np.sqrt(N[0] ** 2 + C[0] ** 2 + S[0] ** 2) == pytest.approx(0.995)
    wvase = tmp_path / "w.dat"
    wvase.write_text("sample\neV\n2.0\t65.0\t20.0\t100.0\n3.0\t65.0\t21.0\t110.0\n")
    data = el.read_woollam(wvase)
    assert np.allclose(data.wavelength, dm.HC_EV_NM / np.array([2.0, 3.0])) and data.psi_sigma is None
    (tmp_path / "t.dat").write_text("x\nnm\nnm TRIG\n")
    with pytest.raises(ValueError):
        el.read_woollam(tmp_path / "t.dat")


def test_read_spectraray_multi_angle(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("; WAVELENGTH 50.2 50.2 70.2 70.2\n400.0 10.0 350.0 20.0 -5.0\n500.0 11.0 10.0 21.0 15.0\n")
    data = el.read_spectraray(path)
    assert len(data) == 4 and np.allclose(data.angles, [50.2, 70.2])
    assert np.allclose(data.select(angles=[70.2]).delta, [355.0, 15.0])


def test_the_published_measurements_read():
    wafer = el.read_woollam(DATA / "complete_ease_example.dat")
    assert len(wafer) == 3264 and np.allclose(wafer.angles, [50, 60, 70])
    assert wafer.wavelength.min() == pytest.approx(193.0) and wafer.depolarization is not None
    tio2 = el.read_spectraray(DATA / "TiO2_400cycles.txt")
    assert len(tio2) == 1227 and np.allclose(tio2.angles, [70.06])


def synthetic(sample, wavelengths, angles, psi_sigma=None, delta_sigma=None, seed=0):
    spectra = el.simulate(sample, wavelengths, angles)
    angle, wavelength = np.meshgrid(angles, wavelengths, indexing="ij")
    psi, delta = spectra.psi.ravel(), spectra.delta.ravel()
    if psi_sigma:
        rng = np.random.default_rng(seed)
        psi = psi + rng.normal(0, psi_sigma, psi.size)
        delta = (delta + rng.normal(0, delta_sigma, delta.size)) % 360
    return el.Data(wavelength.ravel(), angle.ravel(), psi, delta,
                   None if psi_sigma is None else psi_sigma, None if delta_sigma is None else delta_sigma)


def oxide_on_silicon(p):
    return el.Sample([el.Layer(dm.Cauchy(A=p["A"], B=p["B"]), p["d"])], substrate=SILICON)


def test_thickness_and_cauchy_from_noisy_data_with_honest_errors():
    truth = {"d": 612.3, "A": 1.462, "B": 0.0041}
    data = synthetic(oxide_on_silicon(truth), np.linspace(300, 1400, 221), [55.0, 65.0, 75.0], 0.02, 0.05)
    params = el.Parameters().add("d", 300, min=0, max=2000).add("A", 1.5, min=1.3, max=1.8).add("B", 0.003, min=0)
    result = el.fit(oxide_on_silicon, params, data, grid={"d": np.arange(10, 2000, 10.0)})
    assert result.success and 0.9 < result.mse < 1.1  # weighted by the true sigma, MSE ~ 1
    for name, value in truth.items():
        assert abs(result.values[name] - value) < 4 * result.stderr[name], name
    assert result.ci90["d"] == pytest.approx(el.Z90 * result.stderr["d"])
    assert result.correlation.shape == (3, 3) and np.allclose(np.diag(result.correlation), 1)
    assert "MSE" in result.report() and "± " in result.report()
    # without the grid the local fit from 300 nm lands on another fringe order
    trapped = el.fit(oxide_on_silicon, params, data)
    assert trapped.mse > 10 * result.mse


def test_a_tauc_lorentz_film_and_its_roughness_are_recovered_exactly():
    def sample(p):
        film = dm.EpsilonInf(p["einf"]) + dm.TaucLorentz(A=p["A"], E0=p["E0"], C=p["C"], Eg=p["Eg"])
        return el.Sample([el.Layer(dm.Bruggeman(film, 1.0, 0.5), p["rough"]), el.Layer(film, p["d"])],
                         substrate=SILICON)

    truth = {"einf": 1.15, "A": 140.0, "E0": 3.75, "C": 2.1, "Eg": 1.65, "d": 85.0, "rough": 3.0}
    data = synthetic(sample(truth), np.linspace(260, 1400, 115), [60.0, 70.0])
    params = el.Parameters()
    for name, value in truth.items():
        params.add(name, value * 1.04, min=0)
    result = el.fit(sample, params, data, quantity="ncs")
    assert result.mse < 1e-6
    for name, value in truth.items():
        assert result.values[name] == pytest.approx(value, rel=1e-5), name


def test_a_film_on_glass_with_backside_reflection_fits_in_ncs():
    def sample(p):
        return el.Sample([el.Layer(dm.Cauchy(2.05, 0.012), p["d"])], substrate=SILICA, substrate_thickness=1e6)

    data = synthetic(sample({"d": 143.0}), np.linspace(350, 1000, 66), [55.0, 65.0])
    spectra = el.simulate(sample({"d": 143.0}), data.wavelength[:3], 55.0)
    assert np.all(spectra.depolarization > 0)
    data.depolarization = el._spectra(sample({"d": 143.0}), data.wavelength, data.angle).depolarization
    result = el.fit(sample, el.Parameters().add("d", 100, min=0, max=500), data, quantity="ncs",
                    grid={"d": np.arange(5, 500, 5.0)})
    assert result.values["d"] == pytest.approx(143.0, rel=1e-8) and result.mse < 1e-5


def test_point_by_point_optical_constants():
    truth = dm.EpsilonInf(1.2) + dm.TaucLorentz(A=120, E0=3.6, C=2.4, Eg=1.6)
    wavelengths = np.linspace(300, 1200, 46)
    data = synthetic(el.Sample([el.Layer(truth, 40.0)], substrate=SILICON), wavelengths, [60.0, 70.0])
    result = el.fit_nk(lambda index: el.Sample([el.Layer(index, 40.0)], substrate=SILICON), data,
                       truth.index(1200.0), reverse=True)
    expected = truth.index(wavelengths)
    assert np.max(np.abs(result.n + 1j * result.k - expected)) < 1e-7
    assert np.all(result.mse < 1e-6)
    assert result.table().index(700.0) == pytest.approx(np.interp(700.0, wavelengths, result.n)
                                                        + 1j * np.interp(700.0, wavelengths, result.k))


def test_the_native_oxide_of_the_published_rc2_wafer():
    """A real Woollam RC2 scan labelled "2nm SiO2 on Si" (pyElli's example), with Green 2008 silicon."""
    data = el.read_woollam(DATA / "complete_ease_example.dat").select(wavelength=(250, 1000))
    result = el.fit(lambda p: el.Sample([el.Layer(SILICA, p["d"])], substrate=SILICON),
                    el.Parameters().add("d", 5.0, min=0, max=50), data, grid={"d": np.arange(0.0, 20.0, 1.0)})
    assert result.values["d"] == pytest.approx(2.245, abs=0.01)
    assert result.mse == pytest.approx(10.8, rel=0.02)


def test_validation():
    data = el.Data([500.0, 600.0], 70.0, [10.0, 11.0], [170.0, 171.0])
    params = el.Parameters().add("d", 10, min=0, max=100).add("A", 1.5).add("B", 0.0).add("x", 1.0)
    with pytest.raises(ValueError, match="more free parameters"):
        el.fit(lambda p: oxide_on_silicon(p), params, data)
    with pytest.raises(ValueError, match="grid names"):
        el.fit(lambda p: oxide_on_silicon(p), el.Parameters().add("d", 10).add("A", 1.5, vary=False)
               .add("B", 0.0, vary=False), data, grid={"A": [1.4]})
    with pytest.raises(ValueError, match="outside"):
        el.Parameters().add("d", -1, min=0)
    with pytest.raises(ValueError, match="quantity"):
        el.fit(oxide_on_silicon, el.Parameters().add("d", 10).add("A", 1.5, vary=False).add("B", 0, vary=False),
               data, quantity="rho")
    with pytest.raises(ValueError, match="lossless"):
        el.simulate(el.Sample([], substrate=SILICON, ambient=dm.Constant(1.0, 0.1)), 500.0, 70.0)
    with pytest.raises(ValueError):
        el.Layer(1.5, -1.0)
    with pytest.raises(ValueError):
        el.Data([500.0], 70.0, [10.0], [170.0], psi_sigma=[0.0])
    fixed = el.fit(oxide_on_silicon, el.Parameters().add("d", 10, vary=False).add("A", 1.5, vary=False)
                   .add("B", 0.0, vary=False), data)
    assert fixed.free == [] and fixed.mse > 0


def test_ellipsometry_agrees_with_pyelli():
    pytest.importorskip("elli")
    pytest.importorskip("lmfit")
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from ellipsometry_codes import forward, readers, tio2_fit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert all(psi < 1e-10 and delta < 1e-10 for psi, delta in forward().values())
        assert all(max(row[1:]) < 1e-11 for row in readers().values())
        rows, mse, mse_lmfit = tio2_fit()
    assert all(abs(ours - theirs) <= 1e-5 * abs(theirs) for ours, theirs, _, _ in rows.values())
    assert all(abs(error - error_ref) <= 1e-4 * error_ref for _, _, error, error_ref in rows.values())
    assert mse == pytest.approx(mse_lmfit, rel=1e-9)
