"""Spectroscopic ellipsometry: the forward model, measurement files, and least-squares fits.

A measurement is fitted by writing the sample as a function of the parameters to vary::

    import numpy as np
    from pystratify import dispersion as dm, ellipsometry as el

    data = el.read_woollam("wafer.dat").select(wavelength=(300, 1000))
    silicon = dm.Table(...)                                   # tabulated substrate n, k

    def sample(p):
        oxide = dm.Cauchy(A=p["A"], B=p["B"])
        return el.Sample([el.Layer(oxide, p["d"])], substrate=silicon)

    params = el.Parameters()
    params.add("d", 100, min=0, max=2000)                    # nm
    params.add("A", 1.46, min=1.2, max=2.0)
    params.add("B", 0.004, min=0, max=0.05)                   # um^2
    result = el.fit(sample, params, data, grid={"d": np.arange(10, 2000, 20)})
    print(result.report())                                    # values, 90% limits, correlations, MSE

**Conventions.** Psi and Delta follow the instrument (Nebraska) convention that Woollam,
Sentech and pyElli report: tan(Psi) e^{i Delta} = conj(r_p / r_s) in this package's n + ik
fields, so a bare dielectric below its Brewster angle has Delta = 180 deg. That is
180 deg - Delta of :func:`pystratify.planar.ellips` (and of multilayer.app's Ellipsometry card),
which keep the transfer-matrix phase. Delta is reported in [0, 360). N = cos 2Psi,
C = sin 2Psi cos Delta, S = sin 2Psi sin Delta. Wavelengths and thicknesses in nm, angles of
incidence in degrees, the ambient lossless.

**Backside reflection.** A ``Sample`` with ``substrate_thickness`` set adds the light returned
from the substrate's back surface, summed incoherently (a substrate many coherence lengths
thick): the reflected light is then partly depolarised, and (N, C, S) shrink to a length
1 - depolarisation. Psi = arccos(N/p)/2 and Delta = atan2(S, C) with p that length; a fit
of such a sample should use ``quantity="ncs"``.

**Fits.** :func:`fit` minimises the instrument makers' mean-squared error with SciPy's
trust-region least squares inside the parameter bounds: on Psi/Delta,
MSE = sqrt(sum[((Psi_m - Psi_e)/s_Psi)^2 + ((Delta_m - Delta_e)/s_Delta)^2] / (2P - M)) with
the file's standard deviations s (1 deg when it has none), and on N, C, S
MSE = 1000 sqrt(sum[(N_m - N_e)^2 + (C_m - C_e)^2 + (S_m - S_e)^2] / (3P - M)), for P points
and M free parameters. Uncertainties come from the Jacobian at the minimum, scaled by the MSE;
the 90 % limits are 1.645 standard errors, as ellipsometry software prints them. A thickness
fit has many local minima, so pass ``grid`` to start from the best point of a coarse search.
:func:`fit_nk` inverts one layer's n and k wavelength by wavelength (point-by-point).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize

from . import dispersion as dm
from .planar import _interface_coefficients
from .response import sweep

__all__ = [
    "Layer",
    "Sample",
    "Spectra",
    "Data",
    "Parameter",
    "Parameters",
    "FitResult",
    "NKResult",
    "simulate",
    "fit",
    "fit_nk",
    "read_woollam",
    "read_spectraray",
]

EPSILON = np.finfo(float).eps
#: two-sided 90 % point of the normal distribution, the confidence ellipsometry software reports
Z90 = 1.6448536269514722


def _material(value) -> dm.Dispersion:
    if isinstance(value, dm.Dispersion):
        return value
    if np.ndim(value) == 0:
        return dm.Constant(complex(value).real, complex(value).imag)
    raise TypeError("a material is a pystratify.dispersion model or a scalar refractive index")


@dataclass(frozen=True)
class Layer:
    """A film: a dispersion model (or a scalar n + ik) and a thickness in nm."""

    material: object
    thickness: float

    def __post_init__(self):
        object.__setattr__(self, "material", _material(self.material))
        if not np.isfinite(self.thickness) or self.thickness < 0:
            raise ValueError("layer thickness must be finite and non-negative (nm)")


@dataclass(frozen=True)
class Sample:
    """Films on a substrate, ``layers`` listed from the ambient side down.

    ``substrate_thickness`` (nm) makes the substrate finite and adds its back-surface reflection
    incoherently, with ``exit`` the medium behind it; ``None`` (the default) is a semi-infinite
    substrate, or a back surface roughened so that nothing returns from it.
    """

    layers: Sequence[Layer]
    substrate: object
    ambient: object = 1.0
    substrate_thickness: Optional[float] = None
    exit: object = 1.0

    def __post_init__(self):
        layers = tuple(self.layers)
        if not all(isinstance(layer, Layer) for layer in layers):
            raise TypeError("Sample layers must be Layer objects")
        object.__setattr__(self, "layers", layers)
        for name in ("substrate", "ambient", "exit"):
            object.__setattr__(self, name, _material(getattr(self, name)))
        if self.substrate_thickness is not None and not (np.isfinite(self.substrate_thickness)
                                                         and self.substrate_thickness > 0):
            raise ValueError("substrate_thickness must be positive (nm) or None")


@dataclass
class Spectra:
    """Ellipsometric quantities in degrees (Psi, Delta) and as N, C, S; one value per point."""

    wavelength: np.ndarray
    angle: np.ndarray
    psi: np.ndarray
    delta: np.ndarray
    N: np.ndarray
    C: np.ndarray
    S: np.ndarray
    depolarization: np.ndarray
    r_s: Optional[np.ndarray] = None
    r_p: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------- forward model


def _angles(n, sin_ambient):
    """Complex propagation angles, forward branch in the last medium (as planar.list_snell)."""
    theta = np.arcsin((sin_ambient / n).astype(complex))
    last = n[-1] * np.cos(theta[-1])
    forward = np.where(np.abs(last.imag) > 100 * EPSILON, last.imag > 0, last.real > 0)
    theta[-1] = np.where(forward, theta[-1], np.pi - theta[-1])
    return theta


def _stack(pol, n, theta, delta):
    """r and t of a coherent stack, vectorised over trailing axes (planar's kernel, batched)."""
    r, t = _interface_coefficients(pol, n[:-1], n[1:], theta[:-1], theta[1:])
    response = sweep((-r, np.ones_like(r), np.ones_like(r), -r), np.exp(2j * delta[1:-1]))
    amplitude = np.ones(r.shape[1:], complex)
    with np.errstate(all="ignore"):
        for j in range(len(r)):
            if j:
                amplitude = amplitude * np.exp(1j * delta[j])
            amplitude = amplitude * t[j] / (1 + r[j] * response.outgoing_out[j])
    return response.outgoing_in[0], amplitude


def _indices(materials, wavelength):
    unique, inverse = np.unique(wavelength, return_inverse=True)
    return np.stack([np.atleast_1d(material.index(unique))[inverse] for material in materials])


def _response(sample: Sample, wavelength, angle_deg):
    """(sum |r_s|^2, sum |r_p|^2, sum r_s conj(r_p), r_s, r_p) at equal-length point arrays."""
    wavelength = np.asarray(wavelength, float)
    front = [sample.ambient, *(layer.material for layer in sample.layers), sample.substrate]
    n = _indices(front + ([sample.exit] if sample.substrate_thickness is not None else []), wavelength)
    if np.any(np.abs(n[0].imag) > 1e-12):
        raise ValueError("the ambient must be lossless")
    if np.any(n.real * n.imag < 0):
        raise ValueError("a medium has gain (n k < 0); ellipsometry here needs passive media")
    sin_ambient = n[0].real * np.sin(np.radians(angle_deg))
    count = len(front)
    thickness = np.array([0.0, *(layer.thickness for layer in sample.layers), 0.0])
    theta = _angles(n[:count], sin_ambient)
    delta = 2 * np.pi * n[:count] * np.cos(theta) * thickness[:, None] / wavelength
    amplitudes = {pol: _stack(pol, n[:count], theta, delta) for pol in "sp"}
    r_s, r_p = amplitudes["s"][0], amplitudes["p"][0]
    ss, pp, sp = np.abs(r_s) ** 2, np.abs(r_p) ** 2, r_s * np.conj(r_p)
    if sample.substrate_thickness is not None:
        back = {}
        reversed_theta = theta[::-1]
        for pol in "sp":
            t_in = amplitudes[pol][1]
            r_up, t_out = _stack(pol, n[count - 1::-1], reversed_theta, delta[::-1])
            exit_theta = _angles(n[count - 1:], sin_ambient)[-1]
            r_back, _ = _interface_coefficients(pol, n[count - 1], n[count], theta[-1], exit_theta)
            back[pol] = (t_in * r_back * t_out, r_up * r_back)
        kz = 2 * np.pi * n[count - 1] * np.cos(theta[-1]) / wavelength
        # each returning beam crosses the substrate twice more than the last: |P|^4 = (exp(-2 Im kz d))^2
        attenuation = np.exp(-2 * kz.imag * sample.substrate_thickness) ** 2
        with np.errstate(all="ignore"):
            def series(first_a, ratio_a, first_b, ratio_b):
                return first_a * np.conj(first_b) * attenuation / (1 - ratio_a * np.conj(ratio_b) * attenuation)

            ss = ss + series(*back["s"], *back["s"]).real
            pp = pp + series(*back["p"], *back["p"]).real
            sp = sp + series(*back["s"], *back["p"])
    return ss, pp, sp, r_s, r_p


def _spectra(sample, wavelength, angle) -> Spectra:
    ss, pp, sp, r_s, r_p = _response(sample, wavelength, angle)
    total = ss + pp
    # tan(Psi) e^{i Delta} = conj(r_p / r_s), so sin 2Psi e^{i Delta} = 2 r_s conj(r_p) / (|r_s|^2 + |r_p|^2)
    N, C, S = (ss - pp) / total, 2 * sp.real / total, 2 * sp.imag / total
    polarized = np.sqrt(N**2 + C**2 + S**2)
    psi = 0.5 * np.degrees(np.arccos(np.clip(N / polarized, -1.0, 1.0)))
    delta = np.degrees(np.arctan2(S, C)) % 360.0
    delta = np.where(delta >= 360.0, 0.0, delta)
    coherent = sample.substrate_thickness is None
    return Spectra(np.asarray(wavelength, float), np.asarray(angle, float), psi, delta, N, C, S,
                   np.clip(1 - polarized, 0.0, None), r_s if coherent else None, r_p if coherent else None)


def simulate(sample: Sample, wavelength_nm, angle_deg) -> Spectra:
    """Psi, Delta, N, C, S and depolarisation on the grid angle x wavelength: arrays (angles, wavelengths).

    Scalars give scalars' shapes; ``r_s`` and ``r_p`` (the coherent amplitudes) are filled when the
    substrate is semi-infinite.
    """
    wavelength = np.asarray(wavelength_nm, float)
    angle = np.asarray(angle_deg, float)
    grid_angle, grid_wavelength = np.meshgrid(np.atleast_1d(angle), np.atleast_1d(wavelength), indexing="ij")
    flat = _spectra(sample, grid_wavelength.ravel(), grid_angle.ravel())
    shape = angle.shape + wavelength.shape

    def reshape(value):
        return None if value is None else value.reshape(shape)

    return Spectra(wavelength, angle, *(reshape(getattr(flat, name)) for name in
                                        ("psi", "delta", "N", "C", "S", "depolarization", "r_s", "r_p")))


# ---------------------------------------------------------------------------- data


@dataclass
class Data:
    """Measured points: wavelength (nm), angle (deg), Psi, Delta (deg), optional standard deviations
    (deg) and depolarisation (a fraction, 0-1). All one-dimensional, one entry per point."""

    wavelength: np.ndarray
    angle: np.ndarray
    psi: np.ndarray
    delta: np.ndarray
    psi_sigma: Optional[np.ndarray] = None
    delta_sigma: Optional[np.ndarray] = None
    depolarization: Optional[np.ndarray] = None
    source: str = ""

    def __post_init__(self):
        names = ("wavelength", "angle", "psi", "delta", "psi_sigma", "delta_sigma", "depolarization")
        values = {name: getattr(self, name) for name in names}
        size = np.size(values["wavelength"])
        for name, value in values.items():
            if value is None:
                continue
            value = np.broadcast_to(np.asarray(value, float), (size,)).copy()
            if np.any(~np.isfinite(value)):
                raise ValueError(f"Data {name} has non-finite values")
            setattr(self, name, value)
        if np.any(self.wavelength <= 0):
            raise ValueError("Data wavelengths must be positive (nm)")
        for name in ("psi_sigma", "delta_sigma"):
            if getattr(self, name) is not None and np.any(getattr(self, name) <= 0):
                raise ValueError(f"Data {name} must be positive")

    def __len__(self):
        return self.wavelength.size

    @property
    def angles(self) -> np.ndarray:
        return np.unique(self.angle)

    @property
    def ncs(self):
        """(N, C, S) of the measurement, shortened by its depolarisation when the file has one."""
        psi, delta = np.radians(self.psi), np.radians(self.delta)
        scale = 1.0 if self.depolarization is None else 1.0 - self.depolarization
        return (scale * np.cos(2 * psi), scale * np.sin(2 * psi) * np.cos(delta),
                scale * np.sin(2 * psi) * np.sin(delta))

    def select(self, wavelength=None, angles=None) -> "Data":
        """A subset: ``wavelength=(min, max)`` in nm and/or ``angles`` (each matched to 1e-6 deg)."""
        keep = np.ones(len(self), bool)
        if wavelength is not None:
            keep &= (self.wavelength >= wavelength[0]) & (self.wavelength <= wavelength[1])
        if angles is not None:
            keep &= np.any(np.abs(self.angle[:, None] - np.atleast_1d(angles)[None, :]) < 1e-6, axis=1)
        return self._take(keep)

    def _take(self, keep):
        return Data(*(None if getattr(self, name) is None else getattr(self, name)[keep] for name in
                      ("wavelength", "angle", "psi", "delta", "psi_sigma", "delta_sigma", "depolarization")),
                    source=self.source)


_UNITS = {"nm": 1.0, "nanometers": 1.0, "angstroms": 0.1, "angstrom": 0.1, "a": 0.1, "å": 0.1,
          "um": 1000.0, "µm": 1000.0, "microns": 1000.0, "micrometers": 1000.0}


def _to_nm(values, unit):
    if unit in ("ev",):
        return dm.HC_EV_NM / values
    return values * _UNITS[unit]


def read_woollam(path) -> Data:
    """Psi/Delta from a J.A. Woollam text file: CompleteEASE (rows tagged ``E``, depolarisation rows
    ``dPolE`` in percent) or WVASE (untagged rows). Columns: wavelength, angle, Psi, Delta and, when
    present, their standard deviations. The spectral unit line (nm, Angstroms, eV, um) is honoured.
    Rows of other kinds (``uR``, Mueller-matrix elements) are skipped.
    """
    unit, rows, depolarization = "nm", [], {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) == 1 and fields[0].lower() in (*_UNITS, "ev"):
            unit = fields[0].lower()
            continue
        if fields[0].upper() == "TRIG" or (len(fields) > 1 and fields[1].upper() == "TRIG"):
            raise ValueError("tan(Psi)/cos(Delta) WVASE files are not supported; export Psi/Delta")
        tag = None if _is_number(fields[0]) else fields[0]
        numbers = fields if tag is None else fields[1:]
        if len(numbers) < 4 or not all(_is_number(value) for value in numbers):
            continue
        values = [float(value) for value in numbers]
        if tag is None or tag == "E":
            rows.append(values[:6])
        elif tag.lower() == "dpole":
            depolarization[(values[0], values[1])] = values[2] / 100.0
    if not rows:
        raise ValueError(f"{path}: no Psi/Delta rows found")
    width = min(len(row) for row in rows)
    table = np.array([row[:width] for row in rows])
    keys = list(zip(table[:, 0], table[:, 1]))
    depol = np.array([depolarization.get(key, np.nan) for key in keys]) if depolarization else None
    if depol is not None and np.any(np.isnan(depol)):
        depol = None
    return Data(_to_nm(table[:, 0], unit), table[:, 1], table[:, 2], table[:, 3],
                table[:, 4] if width >= 6 else None, table[:, 5] if width >= 6 else None, depol,
                source=str(path))


def read_spectraray(path) -> Data:
    """Psi/Delta from a SENTECH SpectraRay export: a header ``; WAVELENGTH a1 a1 a2 a2 ...`` naming the
    angle of each Psi, Delta column pair, then rows of wavelength (nm) and those pairs."""
    angles, rows = None, []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.replace(";", " ").split()
        if not fields:
            continue
        if fields[0].upper() == "WAVELENGTH":
            angles = [float(value) for value in fields[1::2]]
            continue
        if all(_is_number(value) for value in fields) and angles is not None:
            rows.append([float(value) for value in fields])
    if angles is None or not rows:
        raise ValueError(f"{path}: no SpectraRay Psi/Delta table found")
    pieces = []
    for row in rows:
        for column, angle in enumerate(angles):
            if len(row) >= 3 + 2 * column:
                pieces.append((row[0], angle, row[1 + 2 * column], row[2 + 2 * column]))
    table = np.array(pieces)
    order = np.lexsort((table[:, 0], table[:, 1]))
    table = table[order]
    return Data(table[:, 0], table[:, 1], table[:, 2], table[:, 3] % 360.0, source=str(path))


def _is_number(text):
    return re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|[+-]?inf|nan", text, re.IGNORECASE) is not None


# ---------------------------------------------------------------------------- fitting


@dataclass
class Parameter:
    """A fit parameter: its value, bounds, and whether it varies."""

    value: float
    min: float = -np.inf
    max: float = np.inf
    vary: bool = True

    def __post_init__(self):
        if not self.min <= self.value <= self.max:
            raise ValueError(f"start value {self.value} lies outside [{self.min}, {self.max}]")


class Parameters(dict):
    """Named :class:`Parameter` objects, in insertion order."""

    def add(self, name: str, value: float, min: float = -np.inf, max: float = np.inf, vary: bool = True):
        self[name] = Parameter(float(value), float(min), float(max), bool(vary))
        return self

    def valuesdict(self) -> dict:
        """{name: value} for every parameter, the argument a model function receives."""
        return {name: parameter.value for name, parameter in self.items()}

    @property
    def free(self) -> list:
        return [name for name, parameter in self.items() if parameter.vary]


def _wrap(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def _residuals(model, data: Data, quantity: str):
    if quantity == "psi_delta":
        psi_sigma = 1.0 if data.psi_sigma is None else data.psi_sigma
        delta_sigma = 1.0 if data.delta_sigma is None else data.delta_sigma

        def residual(values):
            spectra = _spectra(model(values), data.wavelength, data.angle)
            return np.concatenate([(spectra.psi - data.psi) / psi_sigma,
                                   _wrap(spectra.delta - data.delta) / delta_sigma])

        return residual, 2
    if quantity == "ncs":
        measured = np.concatenate(data.ncs)

        def residual(values):
            spectra = _spectra(model(values), data.wavelength, data.angle)
            return 1000.0 * (np.concatenate([spectra.N, spectra.C, spectra.S]) - measured)

        return residual, 3
    raise ValueError('quantity must be "psi_delta" or "ncs"')


@dataclass
class FitResult:
    """A converged (or not) fit. ``values`` holds every parameter, ``stderr``/``ci90`` the free ones."""

    values: dict
    free: list
    stderr: dict
    ci90: dict
    correlation: np.ndarray
    mse: float
    quantity: str
    points: int
    success: bool
    message: str
    evaluations: int
    sample: Sample
    data: Data = field(repr=False)
    start: dict = field(default_factory=dict)

    def model_spectra(self) -> Spectra:
        """The fitted model at the measured points."""
        return _spectra(self.sample, self.data.wavelength, self.data.angle)

    def report(self) -> str:
        width = max([len(name) for name in self.values] + [9])
        lines = [f"MSE = {self.mse:.4g} ({self.quantity}, {self.points} points, {len(self.free)} free)"
                 + ("" if self.success else f"  NOT CONVERGED: {self.message}")]
        for name, value in self.values.items():
            if name in self.free:
                lines.append(f"  {name:<{width}} = {value:.6g} ± {self.ci90[name]:.3g}  (90 %)")
            else:
                lines.append(f"  {name:<{width}} = {value:.6g}  (fixed)")
        strong = [(a, b, self.correlation[i, j]) for (i, a), (j, b) in
                  itertools.combinations(enumerate(self.free), 2) if abs(self.correlation[i, j]) > 0.9]
        for a, b, value in strong:
            lines.append(f"  correlated: {a} / {b} = {value:+.3f}")
        return "\n".join(lines)


def fit(model: Callable[[dict], Sample], params: Parameters, data: Data, *, quantity: str = "psi_delta",
        grid: Optional[Mapping[str, Sequence[float]]] = None, **least_squares) -> FitResult:
    """Fit ``model(values) -> Sample`` to ``data`` by varying the free ``params``.

    ``quantity`` is ``"psi_delta"`` (weighted by the file's standard deviations) or ``"ncs"``
    (depolarising samples, or data whose Delta is noisy near 0/180 deg). ``grid`` maps free
    parameter names to trial values: their Cartesian product is scored first and the best point
    starts the local fit. Extra keywords go to ``scipy.optimize.least_squares``.
    """
    if len(data) == 0:
        raise ValueError("no data points")
    free = params.free
    fixed = params.valuesdict()
    residual, per_point = _residuals(model, data, quantity)
    dof = per_point * len(data) - len(free)
    if dof <= 0:
        raise ValueError("more free parameters than measured values")

    def values_of(x):
        values = dict(fixed)
        values.update(zip(free, x))
        return values

    def vector(x):
        return residual(values_of(x))

    x0 = np.array([params[name].value for name in free], float)
    lower = np.array([params[name].min for name in free], float)
    upper = np.array([params[name].max for name in free], float)
    if grid:
        unknown = set(grid) - set(free)
        if unknown:
            raise ValueError(f"grid names must be free parameters: {sorted(unknown)}")
        names = list(grid)
        best, best_cost = x0, np.inf
        for trial in itertools.product(*(np.asarray(grid[name], float) for name in names)):
            x = x0.copy()
            for name, value in zip(names, trial):
                x[free.index(name)] = value
            if np.any(x < lower) or np.any(x > upper):
                continue
            try:
                with np.errstate(all="ignore"):
                    cost = np.sum(vector(x) ** 2)
            except ValueError:  # a model that refuses the point (e.g. a Tauc-Lorentz C above sqrt(2) E0)
                continue
            if np.isfinite(cost) and cost < best_cost:
                best, best_cost = x, cost
        x0 = best
    start = values_of(x0)
    if not free:
        r = vector(x0)
        mse = float(np.sqrt(np.sum(r**2) / dof))
        return FitResult(start, [], {}, {}, np.zeros((0, 0)), mse, quantity, len(data), True,
                         "nothing to vary", 1, model(start), data, start)
    span = np.where(np.isfinite(upper - lower), upper - lower, np.abs(x0) + 1.0)
    x0 = np.clip(x0, lower + 1e-12 * span, upper - 1e-12 * span)
    options = {"x_scale": "jac", "method": "trf", "xtol": 1e-12, "ftol": 1e-12, "gtol": 1e-12}
    options.update(least_squares)
    solution = optimize.least_squares(vector, x0, bounds=(lower, upper), **options)
    mse = float(np.sqrt(2 * solution.cost / dof))
    jacobian = np.atleast_2d(solution.jac)
    curvature = jacobian.T @ jacobian
    covariance = np.linalg.pinv(curvature) * mse**2
    stderr = np.sqrt(np.clip(np.diag(covariance), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / np.outer(stderr, stderr)
    values = values_of(solution.x)
    return FitResult(values, free, dict(zip(free, stderr)), dict(zip(free, Z90 * stderr)), correlation, mse,
                     quantity, len(data), bool(solution.success), solution.message, int(solution.nfev),
                     model(values), data, start)


@dataclass
class NKResult:
    """Point-by-point optical constants: one n, k (and MSE) per measured wavelength."""

    wavelength: np.ndarray
    n: np.ndarray
    k: np.ndarray
    mse: np.ndarray

    def table(self) -> dm.Table:
        """The result as a :class:`pystratify.dispersion.Table`."""
        return dm.Table(self.wavelength, self.n, self.k)


def fit_nk(model: Callable[[complex], Sample], data: Data, start: complex, *, quantity: str = "psi_delta",
           k_min: float = 0.0, reverse: bool = False) -> NKResult:
    """Invert one material's n + ik at each wavelength, all angles there fitted together.

    ``model(index)`` builds the sample with that material as a scalar index (everything else fixed,
    thicknesses included, usually from a previous :func:`fit`). Each wavelength starts from the last
    one's answer, beginning at ``start`` at the first (the shortest, or the longest with
    ``reverse=True``: begin where the film is transparent and the answer unique).
    """
    wavelengths = np.unique(data.wavelength)
    if reverse:
        wavelengths = wavelengths[::-1]
    current = np.array([complex(start).real, max(complex(start).imag, k_min)])
    n_out, k_out, mse_out = (np.empty(wavelengths.size) for _ in range(3))
    for index, wavelength in enumerate(wavelengths):
        subset = data._take(np.abs(data.wavelength - wavelength) <= 1e-9 * wavelength)
        residual, per_point = _residuals(lambda values: model(complex(values["n"], values["k"])), subset, quantity)
        solution = optimize.least_squares(lambda x: residual({"n": x[0], "k": x[1]}), current,
                                          bounds=([-np.inf, k_min], [np.inf, np.inf]), x_scale="jac",
                                          xtol=1e-12, ftol=1e-12, gtol=1e-12)
        current = solution.x
        n_out[index], k_out[index] = current
        dof = max(per_point * len(subset) - 2, 1)
        mse_out[index] = np.sqrt(2 * solution.cost / dof)
    order = np.argsort(wavelengths)
    return NKResult(wavelengths[order], n_out[order], k_out[order], mse_out[order])
