"""Spheres against pyGDM2 (volume Green dyadic method) through benchmarks/gdm_spheres.py.

Skipped unless pyGDM2 is installed (the ``benchmarks`` extra installs it). A volume lattice carries a
few-percent error at affordable meshes, so the finest mesh is held to a band: cross sections within 4%,
decay rates within 0.5% at 40 nm from the surface and 10% at 20 nm.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pyGDM2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import gdm_spheres  # noqa: E402

BANDS = {"C_ext": 0.04, "C_sca": 0.04, "C_abs": 0.04, "radial, 40 nm": 0.005, "tangential, 40 nm": 0.005,
         "radial, 20 nm": 0.10, "tangential, 20 nm": 0.10}


@pytest.mark.parametrize("name", [name for name, case in gdm_spheres.CASES.items() if case[3]])
def test_pygdm2_agrees_with_pystratify_on_dielectric_spheres(name):
    rows = gdm_spheres.compare({name: gdm_spheres.CASES[name]})[name]
    _, finest = rows[-1]
    for quantity, band in BANDS.items():
        assert abs(gdm_spheres.difference(quantity, finest)) <= band, (quantity, finest[quantity])
