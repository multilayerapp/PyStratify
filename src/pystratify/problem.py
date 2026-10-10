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
class Problem:
    geometry: Literal["films", "cylinders", "spheres"]
    dimensions: object
    n: object
    wavelength: float
    source: PlaneWave | PointDipole = field(default_factory=PlaneWave)
    tolerance: float = 1e-6
    order: int | None = None
    max_evaluations: int = 20000
    hydrodynamic: object = None

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
        if self.hydrodynamic is not None:
            from .nonlocal_sphere import hydrodynamic_regions
            regions = hydrodynamic_regions(self.hydrodynamic, len(n), host_allowed=self.geometry == "films")
            if self.geometry == "films" and regions[0] is not None:
                raise ValueError("the incident medium of a film cannot be hydrodynamic")
            if isinstance(self.source, PointDipole):
                layer = self.source.layer if self.geometry == "films" else int(np.searchsorted(dimensions, self.source.position))
                if layer is not None and regions[layer] is not None or (self.geometry == "films" and regions[-1] is not None):
                    raise ValueError("a point source and the exteriors of its problem must be local")
            object.__setattr__(self, "hydrodynamic", {j: m for j, m in enumerate(regions) if m is not None})
        object.__setattr__(self, "n", n.copy())
        object.__setattr__(self, "dimensions", dimensions.copy())
