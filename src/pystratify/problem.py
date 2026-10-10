"""One public problem model for planar, cylindrical and spherical optics."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class PlaneWave:
    angle: float | None = None
    polarization: str = "unpolarized"
    kind: str = field(default="plane_wave", init=False)


@dataclass(frozen=True)
class PointDipole:
    position: float
    dipole_type: Literal["electric", "magnetic"] = "electric"
    layer: int | None = None
    intrinsic_quantum_yield: float = 1.0
    kind: str = field(default="dipole", init=False)

    def __post_init__(self):
        if self.dipole_type not in ("electric", "magnetic") or not np.isfinite(self.position):
            raise ValueError("a finite electric/magnetic point source is required")
        if not 0 <= self.intrinsic_quantum_yield <= 1:
            raise ValueError("intrinsic quantum yield must lie in [0,1]")


@dataclass(frozen=True)
class FocusedBeam:
    """A beam focused by an aplanatic objective with a uniformly filled annular pupil.

    ``na`` is n sin(theta) of the objective in the ambient (films) or host (cylinders,
    spheres) medium and ``obscuration`` that of its central obstruction: 0 is a refractive
    objective, > 0 a Cassegrain. In transmission a condenser of ``collection_na`` and
    ``collection_obscuration`` (both default to the objective's; films: n sin(theta) in the
    exit medium) collects; in film ``"reflection"`` (transflection) the objective collects.
    ``illumination`` is ``"coherent"`` (a focused spot) or ``"kohler"`` (every pupil
    direction an independent plane wave filling a field stop of diameter ``field_stop``).
    ``model`` ``"scalar"`` is the corrected scalar theory of the coherent cylinder and
    sphere spot. ``polarization`` is the pupil's: ``"unpolarized"``, ``"x"`` or ``"y"``
    (a cylinder's axis is y). ``offset`` is the focus (x, y, z) relative to the sphere
    centre, the cylinder axis or, for films, the first interface (z is depth into the
    stack); coherent only. See :mod:`pystratify.focused`.
    """

    na: float
    obscuration: float = 0.0
    collection_na: float | None = None
    collection_obscuration: float | None = None
    mode: Literal["transmission", "reflection"] = "transmission"
    illumination: Literal["coherent", "kohler"] = "coherent"
    field_stop: float | None = None
    model: Literal["vector", "scalar"] = "vector"
    polarization: Literal["unpolarized", "x", "y"] = "unpolarized"
    offset: tuple = (0.0, 0.0, 0.0)
    kind: str = field(default="focused", init=False)

    def __post_init__(self):
        values = [self.na, self.obscuration] + [v for v in (self.collection_na, self.collection_obscuration) if v is not None]
        if not all(np.isfinite(v) for v in values) or not 0 <= self.obscuration < self.na:
            raise ValueError("the objective needs 0 <= obscuration < NA")
        if self.mode not in ("transmission", "reflection"):
            raise ValueError("mode is transmission or reflection")
        if self.mode == "reflection" and (self.collection_na is not None or self.collection_obscuration is not None):
            raise ValueError("in reflection the objective collects; leave the collection aperture unset")
        low, high = self.collection
        if not 0 <= low < high:
            raise ValueError("the collection aperture needs 0 <= obscuration < NA")
        if self.illumination not in ("coherent", "kohler"):
            raise ValueError("illumination is coherent or kohler")
        if self.model not in ("vector", "scalar"):
            raise ValueError("model is vector or scalar")
        if self.polarization not in ("unpolarized", "x", "y"):
            raise ValueError("pupil polarization is unpolarized, x or y")
        offset = np.asarray(self.offset, float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("offset is a finite (x, y, z) focus position")
        object.__setattr__(self, "offset", tuple(float(v) for v in offset))
        if self.illumination == "kohler":
            if self.field_stop is None or not np.isfinite(self.field_stop) or self.field_stop <= 0:
                raise ValueError("Koehler illumination needs a positive field-stop diameter")
            if self.model != "vector" or np.any(offset):
                raise ValueError("Koehler illumination is vector and shift-invariant (no scalar model, no offset)")
        elif self.field_stop is not None:
            raise ValueError("a field stop applies to Koehler illumination only")

    @property
    def collection(self):
        """(obscuration, NA) of the collecting aperture."""
        if self.mode == "reflection":
            return self.obscuration, self.na
        high = self.na if self.collection_na is None else self.collection_na
        low = self.obscuration if self.collection_obscuration is None else self.collection_obscuration
        return low, high


@dataclass(frozen=True)
class Problem:
    geometry: Literal["films", "cylinders", "spheres"]
    dimensions: object
    n: object
    wavelength: float
    source: PlaneWave | PointDipole | FocusedBeam = field(default_factory=PlaneWave)
    tolerance: float = 1e-6
    order: int | None = None
    max_evaluations: int = 20000
    coherence: object = None
    hydrodynamic: object = None
    contact: str = "electrochemical"

    def __post_init__(self):
        if self.geometry not in ("films", "cylinders", "spheres"):
            raise ValueError("geometry must be films, cylinders or spheres")
        if not np.isfinite(self.wavelength) or self.wavelength <= 0 or not 1e-10 <= self.tolerance <= 1e-3:
            raise ValueError("positive wavelength and tolerance in 1e-10..1e-3 are required")
        if self.order is not None and (int(self.order) != self.order or not 1 <= self.order <= 1500):
            raise ValueError("order must be an integer in 1..1500")
        if int(self.max_evaluations) != self.max_evaluations or not 100 <= self.max_evaluations <= 20000:
            raise ValueError("max_evaluations must be an integer in 100..20000")
        n, dimensions = np.asarray(self.n, complex), np.asarray(self.dimensions, float)
        if n.ndim != 1 or np.any(~np.isfinite(n)) or np.any(n == 0) or np.any(n.imag < 0):
            raise ValueError("indices must be finite, passive and nonzero")
        if self.geometry == "films":
            if dimensions.shape != n.shape or len(n) < 2 or not np.isinf(dimensions[0]) or not np.isinf(dimensions[-1]) or np.any(~np.isfinite(dimensions[1:-1])) or np.any(dimensions[1:-1] < 0):
                raise ValueError("films require matching media/thickness arrays and infinite exterior thicknesses")
        elif dimensions.ndim != 1 or len(n) != len(dimensions) + 1 or not len(dimensions) or np.any(~np.isfinite(dimensions)) or dimensions[0] <= 0 or np.any(np.diff(dimensions) <= 0):
            raise ValueError("radial geometries require increasing positive radii and host-last media")
        if self.coherence is not None:
            coherence = tuple(self.coherence)
            if self.geometry != "films" or len(coherence) != len(n) or any(c not in ("c", "i") for c in coherence) or coherence[0] != "i" or coherence[-1] != "i":
                raise ValueError("a coherence column is per film region, 'c' or 'i', with incoherent half-spaces")
            if isinstance(self.source, PointDipole) and "i" in coherence[1:-1]:
                raise ValueError("point sources need a coherent stack")
            object.__setattr__(self, "coherence", coherence)
        if isinstance(self.source, FocusedBeam):
            check_focused(self.geometry, n, self.source)
        if self.hydrodynamic is not None:
            from .nonlocal_sphere import hydrodynamic_regions
            regions = hydrodynamic_regions(self.hydrodynamic, len(n), host_allowed=self.geometry == "films")
            if self.geometry == "films" and regions[0] is not None:
                raise ValueError("the incident medium of a film cannot be hydrodynamic")
            if isinstance(self.source, PointDipole):
                layer = self.source.layer if self.geometry == "films" else int(np.searchsorted(dimensions, self.source.position))
                if layer is not None and regions[layer] is not None or (self.geometry == "films" and regions[-1] is not None):
                    raise ValueError("a point source and the exteriors of its problem must be local")
            if isinstance(self.source, FocusedBeam):
                raise ValueError("focused beams on hydrodynamic stacks are not supported yet")
            if self.coherence is not None and "i" in self.coherence[1:-1]:
                raise ValueError("hydrodynamic films need a fully coherent stack")
            from .hydrodynamic import CONTACTS
            if self.contact not in CONTACTS:
                raise ValueError(f"contact must be one of {CONTACTS}")
            object.__setattr__(self, "hydrodynamic", {j: m for j, m in enumerate(regions) if m is not None})
        object.__setattr__(self, "n", n.copy())
        object.__setattr__(self, "dimensions", dimensions.copy())


def check_focused(geometry, n, beam):
    """A focused beam's apertures fit its media: NA < n where it is measured, lossless there."""
    n = np.asarray(n, complex)
    illumination = n[0] if geometry == "films" else n[-1]
    collection = n[-1] if geometry == "films" and beam.mode == "transmission" else illumination
    if illumination.imag != 0 or illumination.real <= 0 or collection.imag != 0 or collection.real <= 0:
        raise ValueError("a focused beam needs a lossless medium where it is focused and collected")
    if beam.na >= illumination.real or beam.collection[1] >= collection.real:
        raise ValueError("NA must be smaller than the index of the medium it is measured in")
    if geometry != "films" and beam.mode != "transmission":
        raise ValueError("cylinders and spheres are measured in transmission")
