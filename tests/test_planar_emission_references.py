"""Film emitters against the public references in benchmarks/emitter_references.py.

The Sommerfeld reference is written without ``pystratify`` and is itself checked against the
image-dipole closed form; smuthi is optional (``pytest.importorskip``).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy import inf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from emitter_references import (  # noqa: E402
    GUIDES, HEIGHTS, STACKS, WAVELENGTH, guide_with_loss, image_dipole, reference_rates, smuthi_rates,
    stratify_rates, weakly_absorbing_guide,
)


@pytest.mark.parametrize("x", [0.3, 1.0, 2.5, 7.0, 20.0])
def test_reference_reduces_to_the_image_dipole_at_a_perfect_mirror(x):
    height = x * WAVELENGTH / (4 * np.pi)
    assert np.allclose(reference_rates([1.0, 1.0], [inf, inf], height, mirror=True), image_dipole(x),
                       rtol=1e-10, atol=1e-12)
    # a magnetic dipole sees the mirror with the opposite sign
    swapped = reference_rates([1.0, 1.0], [inf, inf], height, dipole="magnetic", mirror=True)
    assert np.allclose(swapped, 2 - image_dipole(x), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("height", HEIGHTS)
@pytest.mark.parametrize("name", list(STACKS))
def test_film_emitter_rates_against_the_sommerfeld_reference(name, height, dipole):
    n, d = STACKS[name]
    assert np.allclose(stratify_rates(n, d, height, dipole), reference_rates(n, d, height, dipole), rtol=1e-10, atol=0)


@pytest.mark.parametrize("below", [False, True])
@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
@pytest.mark.parametrize("name", list(GUIDES))
def test_source_outside_a_lossless_guide_is_the_lossless_limit(name, dipole, below):
    """From an exterior half-space the guided power must be counted (it was dropped before 0.10.3)."""
    n, d = GUIDES[name]
    lossless = stratify_rates(n, d, 50.0, dipole, below, 1e-8)
    absorbing = stratify_rates(guide_with_loss(n, 1e-7), d, 50.0, dipole, below, 1e-8)
    assert np.allclose(lossless, absorbing, rtol=1e-5, atol=0), (lossless, absorbing)


@pytest.mark.parametrize("name", list(GUIDES))
def test_weakly_absorbing_guide_against_the_sommerfeld_reference(name):
    """The reference integrates a weakly absorbing guide, its quasi-poles as breakpoints."""
    ours, theirs = weakly_absorbing_guide(*GUIDES[name], 50.0)
    assert np.allclose(ours, theirs, rtol=1e-8, atol=0)


@pytest.mark.parametrize("name", list(GUIDES))
def test_lossless_guide_against_smuthi(name):
    pytest.importorskip("smuthi")
    n, d = GUIDES[name]
    assert np.allclose(stratify_rates(n, d, 100.0, tolerance=1e-8), smuthi_rates(n, d, 100.0), rtol=5e-6, atol=0)
