"""solve_problem with hydrodynamic regions: dispatch, validation, the local limit through the API."""

import numpy as np
import pytest

import pystratify as ps

AG = ps.Hydrodynamic.from_ev(9.0, 0.07, 1.39e6)
W = 1239.841984 / 3.0
N_AG = complex(AG.transverse_index(W, 4.5))
TINY = ps.Hydrodynamic(AG.plasma_wavelength, AG.damping_wavelength, AG.fermi_velocity * 1e-7)

CASES = [
    ("films", [np.inf, 20.0, 30.0, np.inf], [1.0, 1.46, N_AG, 1.5], ps.PlaneWave(angle=0.5), 2, "reflectance"),
    ("films", [np.inf, 20.0, 30.0, np.inf], [1.0, 1.46, N_AG, 1.5], ps.PointDipole(15.0, layer=1), 2, "total"),
    ("spheres", [10.0, 12.0], [1.46, N_AG, 1.33], ps.PlaneWave(), 1, "cross_sections"),
    ("spheres", [10.0, 12.0], [1.46, N_AG, 1.33], ps.PointDipole(14.0), 1, "total"),
    ("spheres", [10.0, 12.0], [1.46, N_AG, 1.33], ps.PointDipole(0.0), 1, "total"),
    ("cylinders", [10.0, 12.0], [1.46, N_AG, 1.33], ps.PlaneWave(angle=np.pi / 3), 1, "extinction"),
]


def _value(out, key):
    return out[key].ext_by_order.sum() if key == "cross_sections" else np.asarray(out[key])


@pytest.mark.parametrize("geometry, dims, n, source, region, key", CASES)
def test_local_limit_through_the_api(geometry, dims, n, source, region, key):
    local = _value(ps.solve_problem(ps.Problem(geometry, dims, n, W, source)), key)
    limit = _value(ps.solve_problem(ps.Problem(geometry, dims, n, W, source, hydrodynamic={region: TINY})), key)
    full = _value(ps.solve_problem(ps.Problem(geometry, dims, n, W, source, hydrodynamic={region: AG})), key)
    assert np.max(np.abs(limit - local) / np.abs(local)) < 1e-5
    assert np.max(np.abs(full - local) / np.abs(local)) > 1e-4  # and the full model does differ (2.5e-4 for the 30 nm film)


def test_validation():
    with pytest.raises(ValueError):  # the exterior of a sphere
        ps.Problem("spheres", [10.0], [N_AG, 1.0], W, hydrodynamic={1: AG})
    with pytest.raises(ValueError):  # the incident medium of a film
        ps.Problem("films", [np.inf, 10.0, np.inf], [1.0, N_AG, 1.0], W, hydrodynamic={0: AG})
    with pytest.raises(ValueError):  # a source inside the electron gas
        ps.Problem("spheres", [10.0, 12.0], [1.46, N_AG, 1.33], W, ps.PointDipole(11.0), hydrodynamic={1: AG})
    with pytest.raises(ValueError):  # not a Hydrodynamic
        ps.Problem("cylinders", [10.0], [N_AG, 1.0], W, hydrodynamic={0: "silver"})
