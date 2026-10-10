"""Spheres against MNPBEM (boundary elements) through benchmarks/mnpbem_spheres.py.

Skipped unless GNU Octave is installed; the first run downloads MNPBEM17 from the CPC program library
(``benchmarks/mnpbem_octave.py``). Agreement is at the percent level by design: the check is that the
boundary-element answer approaches PyStratify's exact Mie result as its mesh is refined.
"""

import shutil
import sys
from pathlib import Path

import pytest

if not (shutil.which("octave-cli") or shutil.which("octave")):
    pytest.skip("GNU Octave is not installed", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import mnpbem_spheres  # noqa: E402


@pytest.mark.parametrize("case", ["sphere in vacuum", "sphere in water", "nanoshell"])
def test_mnpbem_converges_to_pystratify(case):
    exact, meshes = mnpbem_spheres.compare([case])[case]
    coarsest, finest = meshes[min(meshes)], meshes[max(meshes)]
    for quantity, value in exact.items():
        first, last = coarsest[quantity] / value - 1, finest[quantity] / value - 1
        assert abs(last) < 0.05, (quantity, last)
        if abs(first) > 0.01:
            assert abs(last) < abs(first), (quantity, first, last)
