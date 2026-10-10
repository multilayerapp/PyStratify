"""Spheres against scattnlay and treams through benchmarks/mie_codes.py.

The code comparisons are skipped unless both codes are installed (the ``benchmarks`` extra installs
them); the centre against Bohren & Huffman needs only scipy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import pystratify as ps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from mie_codes import EXTRA_ORDERS, WAVELENGTH, bohren_huffman_centre, compare, compare_near_fields  # noqa: E402


def test_spheres_agree_with_scattnlay_and_treams():
    pytest.importorskip("scattnlay")
    pytest.importorskip("treams")
    results = compare(chiral={})
    for (name, reference), values in results.items():
        # converged: the formulation at machine precision; default: Wiscombe's truncation, ~1e-8
        limit = 1e-7 if "default truncation" in reference else 1e-10
        assert all(value <= limit for value in values.values()), (name, reference, values)
    assert any(f"+ {EXTRA_ORDERS} orders" in reference for _, reference in results)


def test_near_fields_agree_with_scattnlay_and_treams():
    pytest.importorskip("scattnlay")
    pytest.importorskip("treams")
    results = compare_near_fields()
    for name, values in results.items():
        # reported, not asserted: scattnlay's own centre loses digits as k r -> 0, and its internal field
        # fails toward the core of the 10-layer graded sphere (only the outer half of its layers is checked;
        # PyStratify's coefficients there are pinned by the 60-digit test in test_solver.py)
        layers = values.pop("inside by layer")
        if name == "graded index, 10 layers, x = 12":
            assert max(values["E inside"], values["H inside"]) > 1e-3  # scattnlay's failure, still there
            layers = layers[len(layers) // 2:]
            values["E inside"], values["H inside"] = max(e for _, e, _ in layers), max(h for _, _, h in layers)
        checked = {q: v for q, v in values.items() if "scattnlay" not in q}
        assert all(value <= 1e-10 for value in checked.values()), (name, checked)
        assert {"E inside", "H inside", "E host", "H host"} <= checked.keys()
    assert sum("centre E, BH" in values for values in results.values()) == 4
    assert sum("E host, treams" in values for values in results.values()) == 6


@pytest.mark.parametrize("radius, m", [(20.0, np.sqrt(-10 + 1j + 0j)), (95.5, 1.5), (75.0, 3.5 + 0.01j), (200.0, 0.2 + 3j)])
def test_field_at_the_centre_is_bohren_huffman_d1_and_c1(radius, m):
    """Only l = 1 survives at the centre: E(0) = d_1 x_hat, H(0) = m c_1 y_hat, exactly (also at -0.0)."""
    d1, mc1 = bohren_huffman_centre(m, 2 * np.pi * radius / WAVELENGTH)
    sol = ps.solve([radius], np.array([[m, 1.0]]), np.array([WAVELENGTH]), l_max=30)
    f = ps.near_field(sol, [0.0, -0.0], [0.0, 0.0], [0.0, -0.0])
    assert np.allclose(f.e["x"], d1, rtol=1e-13, atol=0) and np.allclose(f.h["y"], mc1, rtol=1e-13, atol=0)
    assert np.all(np.abs([f.e["y"], f.e["z"], f.h["x"], f.h["z"]]) == 0)
