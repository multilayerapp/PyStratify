"""The examples' analytic gold (examples/_common.py) is Eq. 3 of Etchegoin, Le Ru & Meyer, J. Chem. Phys. 125, 164705
(2006), with the parameters of their erratum, J. Chem. Phys. 127, 189901 (2007): the model identity au-elm-2007e.
The 2006 Table I set had been fitted to a slightly different equation. The independent reference is the Johnson and
Christy data both sets were fitted to: in the visible and near ultraviolet the erratum set is the closer one."""

import importlib.util
from pathlib import Path

import numpy as np

from pystratify import dispersion as dm

ROOT = Path(__file__).resolve().parents[1]
ELM_2006 = (1.53, 145.0, 17000.0, ((0.94, -np.pi / 4, 468.0, 2300.0), (1.36, -np.pi / 4, 331.0, 940.0)))
ELM_2007E = (1.54, 143.0, 14500.0, ((1.27, -np.pi / 4, 470.0, 1900.0), (1.1, -np.pi / 4, 325.0, 1060.0)))


def eq3(lam, eps_inf, lam_p, gamma_p, critical_points):
    """Eq. 3 of the 2006 paper: Drude plus two critical points, wavelengths in nm."""
    eps = eps_inf - 1 / (lam_p**2 * (1 / lam**2 + 1j / (gamma_p * lam)))
    for amplitude, phase, lam_i, gamma_i in critical_points:
        eps = eps + amplitude / lam_i * (np.exp(1j * phase) / (1 / lam_i - 1 / lam - 1j / gamma_i)
                                         + np.exp(-1j * phase) / (1 / lam_i + 1 / lam + 1j / gamma_i))
    return eps


def examples_gold():
    spec = importlib.util.spec_from_file_location("examples_common", ROOT / "examples" / "_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gold


def test_the_examples_gold_is_the_erratum_set():
    gold = examples_gold()
    lam = np.linspace(250.0, 1000.0, 31)
    assert np.allclose(gold(lam) ** 2, eq3(lam, *ELM_2007E), rtol=1e-12)
    assert abs(gold(614.0) - (0.2035 + 3.1793j)) < 1e-4  # the 2006 set gives 0.1412 + 3.1518j


def test_the_erratum_set_is_closer_to_johnson_and_christy_in_the_visible():
    jc = dm.Table.from_refractiveindex(ROOT / "crosscheck" / "classics" / "data" / "Au_Johnson.yml")
    lam = np.asarray(jc.wavelength_nm, dtype=float)
    keep = (lam >= 300) & (lam <= 1000)
    measured = (np.asarray(jc.n) + 1j * np.asarray(jc.k))[keep]

    def rms(params):
        return np.sqrt(np.mean(np.abs(np.sqrt(eq3(lam[keep], *params)) - measured) ** 2 / np.abs(measured) ** 2))

    assert keep.sum() >= 20
    assert rms(ELM_2007E) < 0.025 < 0.03 < rms(ELM_2006)  # 0.022 against 0.036 on 24 points (WP03, 2026-10-31)
