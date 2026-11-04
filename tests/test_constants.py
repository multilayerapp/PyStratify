"""PyStratify's constants are its own, not the installed SciPy's CODATA edition (assurance WP07)."""
import math

from pystratify import _constants as k
from pystratify import dispersion as dm


def test_si_exact_constants():
    assert (k.H, k.C, k.E) == (6.62607015e-34, 299_792_458.0, 1.602176634e-19)
    assert k.HBAR == k.H / (2 * math.pi)


def test_measured_constants_are_codata_2022():
    # SciPy < 1.15 ships CODATA 2018 (8.8541878128e-12, 9.1093837015e-31); results must not follow it
    assert k.EPSILON_0 == 8.8541878188e-12
    assert k.M_E == 9.1093837139e-31


def test_dispersion_uses_them():
    assert dm.VACUUM_PERMITTIVITY == k.EPSILON_0
    assert dm.HBAR_EV_S == k.HBAR / k.E
    assert abs(dm.HC_EV_NM - 1239.8419843320025) < 1e-9
