"""Parametric dispersion models: the dielectric functions thin-film ellipsometry fits.

Every model is a small frozen dataclass with two methods, ``epsilon(wavelength)`` and
``index(wavelength)``, taking vacuum wavelengths in **nanometres** (scalar or array) and
returning complex arrays in the package's n + ik, e^{-i omega t} convention (absorption is
Im eps >= 0, k >= 0). Oscillator parameters are in **eV**, as in ellipsometry software.
Models add: ``EpsilonInf(1.9) + TaucLorentz(...) + Lorentz(...)`` is one material, and the
result feeds any solver here through ``index``::

    from pystratify import dispersion as dm, planar

    film = dm.EpsilonInf(1.2) + dm.TaucLorentz(A=120, E0=3.6, C=2.4, Eg=1.6)
    n_film = film.index(633.0)
    planar.coh_tmm("s", [1, n_film, 3.88 + 0.02j], [np.inf, 50, np.inf], 0.0, 633.0)

The oscillator forms (pole, Lorentz, Gaussian, Tauc-Lorentz, resistivity Drude, Bruggeman)
are the ones multilayer.app's oscillator dialog offers, with the same parameter meanings.
Cauchy, Sellmeier, Cody-Lorentz, energy-form Drude, Maxwell Garnett and tabulated data are
added for the rest of the usual ellipsometry set. No fitting is done here: these are the
forward models a fit would vary.

Conventions (each also in its class docstring):

* photon energy E = hc / lambda, hc = 1239.841984 eV nm (CODATA, exact);
* every oscillator contributes eps(E) -> 0 as E -> infinity, so the high-frequency constant
  is always an explicit ``EpsilonInf`` (none is hidden inside Tauc-Lorentz or Cody-Lorentz);
* ``Cauchy`` and ``Sellmeier`` take lambda in micrometres inside their formulas, as their
  published coefficients do;
* ``Table`` never extrapolates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from scipy import constants, integrate, special

__all__ = [
    "Dispersion",
    "Sum",
    "Constant",
    "EpsilonInf",
    "Table",
    "Cauchy",
    "Sellmeier",
    "Pole",
    "Lorentz",
    "Gaussian",
    "TaucLorentz",
    "CodyLorentz",
    "DrudeEnergy",
    "DrudeResistivity",
    "Bruggeman",
    "MaxwellGarnett",
    "bruggeman_epsilon",
    "photon_energy",
    "index_from_epsilon",
    "kramers_kronig_real",
]

#: h c in eV nm, hbar in eV s and the vacuum permittivity in F/m (CODATA, via scipy.constants).
HC_EV_NM = constants.h * constants.c / constants.e * 1e9
HBAR_EV_S = constants.hbar / constants.e
VACUUM_PERMITTIVITY = constants.epsilon_0


def photon_energy(wavelength_nm):
    """Photon energy in eV for a vacuum wavelength in nm."""
    wavelength = np.asarray(wavelength_nm, dtype=float)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("wavelengths must be finite and positive")
    return HC_EV_NM / wavelength


def index_from_epsilon(epsilon):
    """n + ik = sqrt(eps) on the passive branch, k >= 0."""
    index = np.sqrt(np.asarray(epsilon, dtype=complex))
    return np.where(index.imag < 0, -index, index)


def _as_model(value) -> "Dispersion":
    if isinstance(value, Dispersion):
        return value
    if np.ndim(value) == 0 and np.isfinite(value):
        return Constant(complex(value).real, complex(value).imag)
    raise TypeError("expected a Dispersion or a scalar refractive index")


class Dispersion:
    """Base class: subclasses define ``_epsilon(energy_eV, wavelength_nm)``."""

    def epsilon(self, wavelength_nm):
        """Complex relative permittivity at vacuum wavelengths in nm (shape of the input)."""
        wavelength = np.asarray(wavelength_nm, dtype=float)
        energy = photon_energy(wavelength)
        value = np.asarray(self._epsilon(np.atleast_1d(energy), np.atleast_1d(wavelength)), dtype=complex)
        value = np.broadcast_to(value, np.atleast_1d(wavelength).shape)
        return complex(value[0]) if wavelength.ndim == 0 else value.reshape(wavelength.shape).copy()

    def index(self, wavelength_nm):
        """Complex refractive index n + ik, k >= 0."""
        result = index_from_epsilon(self.epsilon(wavelength_nm))
        return complex(result) if np.ndim(result) == 0 else result

    def _epsilon(self, energy, wavelength):  # pragma: no cover - abstract
        raise NotImplementedError

    def __add__(self, other):
        # A bare number is refused: it could mean eps_inf or an index, and the two differ by a square.
        if not isinstance(other, Dispersion):
            return NotImplemented
        left = self.terms if isinstance(self, Sum) else (self,)
        right = other.terms if isinstance(other, Sum) else (other,)
        return Sum(left + right)


@dataclass(frozen=True)
class Sum(Dispersion):
    """eps = sum of the terms' eps. Built by ``+``; a ``Constant`` term adds its eps = (n + ik)^2.

    Terms must be models: ``EpsilonInf(2.0) + Lorentz(...)``, never ``2.0 + Lorentz(...)``.
    """

    terms: tuple

    def __post_init__(self):
        if not all(isinstance(term, Dispersion) for term in self.terms):
            raise TypeError("Sum terms must be Dispersion models; wrap a constant in EpsilonInf or Constant")
        object.__setattr__(self, "terms", tuple(self.terms))

    def _epsilon(self, energy, wavelength):
        total = np.zeros(energy.shape, complex)
        for term in self.terms:
            total = total + term._epsilon(energy, wavelength)
        return total


@dataclass(frozen=True)
class Constant(Dispersion):
    """Non-dispersive n + ik; eps = (n + ik)^2."""

    n: float
    k: float = 0.0

    def _epsilon(self, energy, wavelength):
        return np.full(energy.shape, complex(self.n, self.k) ** 2)


@dataclass(frozen=True)
class EpsilonInf(Dispersion):
    """A constant permittivity offset (eps_inf, or the "e1 offset" of ellipsometry software)."""

    value: float

    def _epsilon(self, energy, wavelength):
        return np.full(energy.shape, complex(self.value))


@dataclass(frozen=True)
class Table(Dispersion):
    """Tabulated n and k, interpolated linearly in wavelength; outside the table raises ValueError."""

    wavelength_nm: tuple
    n: tuple
    k: tuple

    def __post_init__(self):
        rows = [np.asarray(column, dtype=float).reshape(-1) for column in (self.wavelength_nm, self.n, self.k)]
        if len({row.size for row in rows}) != 1 or rows[0].size < 2:
            raise ValueError("Table needs wavelength, n and k of one length, at least two rows")
        if np.any(~np.isfinite(np.concatenate(rows))) or np.any(np.diff(rows[0]) <= 0):
            raise ValueError("Table wavelengths must be finite and strictly increasing")
        if np.any(rows[2] < 0):
            raise ValueError("Table k must be non-negative")
        for name, row in zip(("wavelength_nm", "n", "k"), rows):
            object.__setattr__(self, name, tuple(row))

    @classmethod
    def from_refractiveindex(cls, path) -> "Table":
        """A refractiveindex.info database page (YAML, ``tabulated nk`` / ``tabulated n`` / ``tabulated k``,
        wavelengths in um). Formula pages are refused: write them as a ``Sellmeier`` or ``Cauchy``."""
        blocks, current = {}, None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- type:"):
                current = stripped.split(":", 1)[1].strip()
                if not current.startswith("tabulated"):
                    raise ValueError(f"{path}: '{current}' is not a tabulated page; use the matching model")
                blocks[current] = []
            elif current and stripped and stripped[0] in "0123456789.-+":
                blocks[current].append([float(value) for value in stripped.split()])
            elif current and stripped and not stripped.startswith("data:"):
                current = None
        if "tabulated nk" in blocks:
            rows = np.array(blocks["tabulated nk"])
            return cls(1000 * rows[:, 0], rows[:, 1], rows[:, 2])
        if "tabulated n" not in blocks:
            raise ValueError(f"{path}: no tabulated n or nk data")
        n_rows = np.array(blocks["tabulated n"])
        k = np.zeros(len(n_rows))
        if "tabulated k" in blocks:
            k_rows = np.array(blocks["tabulated k"])
            k = np.interp(n_rows[:, 0], k_rows[:, 0], k_rows[:, 1])
        return cls(1000 * n_rows[:, 0], n_rows[:, 1], k)

    def _epsilon(self, energy, wavelength):
        grid = np.asarray(self.wavelength_nm)
        tolerance = 1e-9 * grid[-1]
        if np.any(wavelength < grid[0] - tolerance) or np.any(wavelength > grid[-1] + tolerance):
            raise ValueError(f"Table covers {grid[0]:g}-{grid[-1]:g} nm; it does not extrapolate")
        n = np.interp(wavelength, grid, self.n)
        k = np.interp(wavelength, grid, self.k)
        return (n + 1j * k) ** 2


@dataclass(frozen=True)
class Cauchy(Dispersion):
    """Cauchy index with an Urbach absorption tail (the form in Woollam's software).

    n = A + B / lambda^2 + C / lambda^4 with lambda in micrometres (B in um^2, C in um^4);
    k = k_amplitude * exp(k_exponent * (E - E_band)), E_band the photon energy of
    ``band_edge_nm``. Not Kramers-Kronig consistent: use it only where the film is
    transparent or nearly so.
    """

    A: float
    B: float = 0.0
    C: float = 0.0
    k_amplitude: float = 0.0
    k_exponent: float = 0.0
    band_edge_nm: float = 400.0

    def __post_init__(self):
        if self.k_amplitude < 0:
            raise ValueError("Cauchy k_amplitude must be non-negative")
        photon_energy(self.band_edge_nm)

    def _epsilon(self, energy, wavelength):
        um = wavelength / 1000.0
        n = self.A + self.B / um**2 + self.C / um**4
        k = self.k_amplitude * np.exp(self.k_exponent * (energy - HC_EV_NM / self.band_edge_nm))
        return (n + 1j * k) ** 2


@dataclass(frozen=True)
class Sellmeier(Dispersion):
    """eps = A + sum_i B_i lambda^2 / (lambda^2 - C_i), lambda in um and C_i in um^2.

    With ``A = 1`` (the default) this is the glass-catalogue form n^2 - 1 = sum B_i lambda^2 /
    (lambda^2 - C_i), so Schott or refractiveindex.info coefficients go in unchanged. Lossless.
    """

    B: tuple
    C: tuple
    A: float = 1.0

    def __post_init__(self):
        B, C = np.atleast_1d(np.asarray(self.B, float)), np.atleast_1d(np.asarray(self.C, float))
        if B.shape != C.shape or B.ndim != 1:
            raise ValueError("Sellmeier needs B and C of one length")
        object.__setattr__(self, "B", tuple(B))
        object.__setattr__(self, "C", tuple(C))

    def _epsilon(self, energy, wavelength):
        um2 = (wavelength / 1000.0) ** 2
        eps = np.full(energy.shape, complex(self.A))
        for b, c in zip(self.B, self.C):
            eps = eps + b * um2 / (um2 - c)
        return eps


@dataclass(frozen=True)
class Pole(Dispersion):
    """Lossless pole eps = A / (E0^2 - E^2), A in eV^2, E0 in eV.

    E0 > 0 is the "UV pole" for absorption above the measured band; E0 = 0 is the "IR pole"
    eps = -A / E^2 for free carriers or phonons below it.
    """

    A: float
    E0: float = 0.0

    def _epsilon(self, energy, wavelength):
        return self.A / (self.E0**2 - energy**2) + 0j


@dataclass(frozen=True)
class Lorentz(Dispersion):
    """eps = A B E0 / (E0^2 - E^2 - i E B): amplitude A (dimensionless), centre E0 and
    broadening B in eV (the Lorentz form of Woollam's software and multilayer.app)."""

    A: float
    E0: float
    B: float

    def _epsilon(self, energy, wavelength):
        return self.A * self.B * self.E0 / (self.E0**2 - energy**2 - 1j * energy * self.B)


@dataclass(frozen=True)
class Gaussian(Dispersion):
    """Gaussian absorption band, Kramers-Kronig consistent (De Sousa Meneses et al., J. Non-Cryst.
    Solids 351, 124 (2005)).

    eps2 = A [exp(-((E - E0)/s)^2) - exp(-((E + E0)/s)^2)], s = B / (2 sqrt(ln 2)), so B is the
    full width at half maximum in eV; eps1 = (2A/sqrt(pi)) [D((E + E0)/s) - D((E - E0)/s)] with
    D Dawson's integral.
    """

    A: float
    E0: float
    B: float

    def __post_init__(self):
        if self.B <= 0:
            raise ValueError("Gaussian B must be positive")

    def _epsilon(self, energy, wavelength):
        s = self.B / (2 * np.sqrt(np.log(2)))
        plus, minus = (energy + self.E0) / s, (energy - self.E0) / s
        eps1 = self.A * 2 / np.sqrt(np.pi) * (special.dawsn(plus) - special.dawsn(minus))
        eps2 = self.A * (np.exp(-(minus**2)) - np.exp(-(plus**2)))
        return eps1 + 1j * eps2


def _tauc_lorentz(energy, A, E0, C, Eg):
    eps2 = A * E0 * C * (energy - Eg) ** 2 / (energy * ((energy**2 - E0**2) ** 2 + C**2 * energy**2))
    eps2 = np.where(energy <= Eg, 0.0, eps2)
    gamma = np.sqrt(E0**2 - C**2 / 2)
    alpha = np.sqrt(4 * E0**2 - C**2)
    zeta4 = (energy**2 - gamma**2) ** 2 + alpha**2 * C**2 / 4
    a_ln = (Eg**2 - E0**2) * energy**2 + Eg**2 * C**2 - E0**2 * (E0**2 + 3 * Eg**2)
    a_tan = (energy**2 - E0**2) * (E0**2 + Eg**2) + Eg**2 * C**2
    gap = np.maximum(np.abs(energy - Eg), 1e-300)
    eps1 = (
        A * C * a_ln / (2 * np.pi * zeta4 * alpha * E0)
        * np.log((E0**2 + Eg**2 + alpha * Eg) / (E0**2 + Eg**2 - alpha * Eg))
        - A * a_tan / (np.pi * zeta4 * E0)
        * (np.pi - np.arctan((2 * Eg + alpha) / C) + np.arctan((alpha - 2 * Eg) / C))
        + 4 * A * E0 * Eg * (energy**2 - gamma**2) / (np.pi * zeta4 * alpha)
        * (np.arctan((alpha + 2 * Eg) / C) + np.arctan((alpha - 2 * Eg) / C))
        - A * E0 * C * (energy**2 + Eg**2) / (np.pi * zeta4 * energy) * np.log(gap / (energy + Eg))
        + 2 * A * E0 * C * Eg / (np.pi * zeta4)
        * np.log(gap * (energy + Eg) / np.sqrt((E0**2 - Eg**2) ** 2 + Eg**2 * C**2))
    )
    return eps1 + 1j * eps2


@dataclass(frozen=True)
class TaucLorentz(Dispersion):
    """Tauc-Lorentz (Jellison & Modine, Appl. Phys. Lett. 69, 371 and 2137 (1996)).

    eps2 = A E0 C (E - Eg)^2 / (E [(E^2 - E0^2)^2 + C^2 E^2]) above Eg and 0 below; eps1 is its
    closed-form Kramers-Kronig transform, which tends to 0 at high energy, so add the
    ``EpsilonInf`` yourself (Jellison's eps1(inf)). A in eV, E0, C, Eg in eV; needs C < sqrt(2) E0.
    """

    A: float
    E0: float
    C: float
    Eg: float

    def __post_init__(self):
        if self.C <= 0 or self.E0 <= 0 or self.Eg < 0:
            raise ValueError("Tauc-Lorentz needs C > 0, E0 > 0 and Eg >= 0")
        if 4 * self.E0**2 - self.C**2 <= 0 or self.E0**2 - self.C**2 / 2 <= 0:
            raise ValueError("Tauc-Lorentz needs 4*E0^2 > C^2 and E0^2 > C^2/2")

    def _epsilon(self, energy, wavelength):
        return _tauc_lorentz(energy, self.A, self.E0, self.C, self.Eg)


@dataclass(frozen=True)
class CodyLorentz(Dispersion):
    """Cody-Lorentz with an Urbach tail (Ferlauto et al., J. Appl. Phys. 92, 2424 (2002)).

    Above Et: eps2 = G(E) L(E), G = (E - Eg)^2 / ((E - Eg)^2 + Ep^2),
    L = A E0 Gamma E / ((E^2 - E0^2)^2 + Gamma^2 E^2). Below Et: eps2 = (E1/E) exp((E - Et)/Eu),
    E1 = Et G(Et) L(Et), continuous at Et. eps1 is the numerical Kramers-Kronig transform of
    eps2 (``kramers_kronig_real``, relative error ~1e-10), tending to 0 at high energy: add an
    ``EpsilonInf``. Et = 0 removes the tail (Eu is then unused). Energies in eV, A in eV.
    """

    A: float
    E0: float
    Gamma: float
    Eg: float
    Ep: float
    Et: float = 0.0
    Eu: float = 0.05

    def __post_init__(self):
        if self.Gamma <= 0 or self.E0 <= 0 or self.Ep <= 0 or self.Eg < 0:
            raise ValueError("Cody-Lorentz needs Gamma, E0, Ep > 0 and Eg >= 0")
        if self.Et < 0 or (self.Et > 0 and self.Eu <= 0):
            raise ValueError("Cody-Lorentz needs Et >= 0 and, with a tail, Eu > 0")

    def eps2(self, energy):
        """Imaginary part at photon energies in eV."""
        energy = np.asarray(energy, dtype=float)

        def cody(e):
            return (e - self.Eg) ** 2 / ((e - self.Eg) ** 2 + self.Ep**2)

        def lorentz(e):
            return self.A * self.E0 * self.Gamma * e / ((e**2 - self.E0**2) ** 2 + self.Gamma**2 * e**2)

        above = np.where(energy > self.Eg, cody(energy) * lorentz(energy), 0.0)
        if self.Et <= 0:
            return above
        e1 = self.Et * cody(self.Et) * lorentz(self.Et) if self.Et > self.Eg else 0.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            tail = np.where(energy > 0, e1 / energy * np.exp((energy - self.Et) / self.Eu), 0.0)
        return np.where(energy > self.Et, above, tail)

    def _epsilon(self, energy, wavelength):
        kinks = [self.Eg, self.Et, self.E0, self.E0 - self.Gamma, self.E0 + self.Gamma]
        return kramers_kronig_real(self.eps2, energy, kinks) + 1j * self.eps2(energy)


@dataclass(frozen=True)
class DrudeEnergy(Dispersion):
    """Free carriers, eps = -Ep^2 / (E^2 + i Gamma E): plasma energy Ep and damping Gamma in eV."""

    Ep: float
    Gamma: float

    def _epsilon(self, energy, wavelength):
        return -(self.Ep**2) / (energy**2 + 1j * self.Gamma * energy)


@dataclass(frozen=True)
class DrudeResistivity(Dispersion):
    """Free carriers from optical resistivity and scattering time (multilayer.app's Drude form):
    eps = -1 / (eps0 rho (i omega + tau omega^2)), rho in ohm m and tau in s.

    The same as ``DrudeEnergy`` with Gamma = hbar / tau and Ep^2 = hbar^2 / (eps0 rho tau).
    """

    rho: float
    tau: float

    def __post_init__(self):
        if self.rho <= 0 or self.tau <= 0:
            raise ValueError("Drude rho and tau must be positive")

    def _epsilon(self, energy, wavelength):
        omega = energy / HBAR_EV_S
        return -1.0 / (VACUUM_PERMITTIVITY * self.rho * (1j * omega + self.tau * omega**2))


def bruggeman_epsilon(eps_a, eps_b, fraction_a):
    """Two-phase Bruggeman permittivity from the phases' permittivities (arrays broadcast).

    Of the quadratic's two roots it keeps the passive one (Im eps >= 0) nearest the volume-weighted
    mean f_a eps_a + f_b eps_b, which is the physical branch for every passive pair, metals included.
    """
    eps_a, eps_b = np.asarray(eps_a, dtype=complex), np.asarray(eps_b, dtype=complex)
    fraction_b = 1.0 - fraction_a
    b = fraction_a * (2 * eps_a - eps_b) + fraction_b * (2 * eps_b - eps_a)
    root = np.sqrt(b**2 + 8 * eps_a * eps_b)
    plus, minus = (b + root) / 4, (b - root) / 4
    linear = fraction_a * eps_a + fraction_b * eps_b
    plus_ok, minus_ok = plus.imag >= -1e-10, minus.imag >= -1e-10
    nearer_plus = np.abs(plus - linear) <= np.abs(minus - linear)
    pick_plus = np.where(plus_ok & minus_ok, nearer_plus, np.where(plus_ok | minus_ok, plus_ok, nearer_plus))
    result = np.where(pick_plus, plus, minus)
    return complex(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class Bruggeman(Dispersion):
    """Two-phase Bruggeman effective medium: ``fraction_a`` of ``a`` (by volume) in ``b``.

    Solves f_a (eps_a - eps)/(eps_a + 2 eps) + f_b (eps_b - eps)/(eps_b + 2 eps) = 0 (spherical
    inclusions) and keeps the passive root nearest the volume-weighted mean. The usual surface
    roughness layer is ``Bruggeman(film, Constant(1.0), 0.5)``. ``a`` and ``b`` are models or
    scalar indices.
    """

    a: Union[Dispersion, complex]
    b: Union[Dispersion, complex]
    fraction_a: float

    def __post_init__(self):
        object.__setattr__(self, "a", _as_model(self.a))
        object.__setattr__(self, "b", _as_model(self.b))
        if not 0 <= self.fraction_a <= 1:
            raise ValueError("Bruggeman fraction_a must be between 0 and 1")

    def _epsilon(self, energy, wavelength):
        return bruggeman_epsilon(self.a._epsilon(energy, wavelength), self.b._epsilon(energy, wavelength),
                                 self.fraction_a)


@dataclass(frozen=True)
class MaxwellGarnett(Dispersion):
    """Maxwell Garnett: spherical ``inclusion`` at volume ``fraction`` in a continuous ``host``.

    (eps - eps_h)/(eps + 2 eps_h) = f (eps_i - eps_h)/(eps_i + 2 eps_h). Unlike Bruggeman it is
    not symmetric in the two phases; use it for dilute, isolated inclusions.
    """

    host: Union[Dispersion, complex]
    inclusion: Union[Dispersion, complex]
    fraction: float

    def __post_init__(self):
        object.__setattr__(self, "host", _as_model(self.host))
        object.__setattr__(self, "inclusion", _as_model(self.inclusion))
        if not 0 <= self.fraction <= 1:
            raise ValueError("MaxwellGarnett fraction must be between 0 and 1")

    def _epsilon(self, energy, wavelength):
        host = self.host._epsilon(energy, wavelength)
        inclusion = self.inclusion._epsilon(energy, wavelength)
        x = self.fraction * (inclusion - host) / (inclusion + 2 * host)
        return host * (1 + 2 * x) / (1 - x)


def kramers_kronig_real(eps2, energy, breakpoints=(), rtol=1e-10):
    """eps1(E) - eps1(inf) = (2/pi) P int_0^inf E' eps2(E') / (E'^2 - E^2) dE'.

    ``eps2`` is a callable of photon energy (eV, array in, array out), odd-symmetric in the
    physical sense and integrable as E' eps2(E') / E'^2 at infinity. The principal value is taken
    by subtracting E eps2(E), which is legitimate because P int_0^inf dE' / (E'^2 - E^2) = 0, so the
    integrand is regular at E' = E; the integral is adaptive (``scipy.integrate.quad_vec``) over
    all energies at once, with ``breakpoints`` (kinks or narrow features of eps2) as interval
    ends.
    """
    energy = np.atleast_1d(np.asarray(energy, dtype=float))
    if np.any(energy <= 0):
        raise ValueError("energies must be positive")
    g_at = energy * eps2(energy)

    def integrand(e):
        e = np.asarray(e, dtype=float)
        denominator = e**2 - energy**2
        with np.errstate(divide="ignore", invalid="ignore"):
            value = (e * eps2(e) - g_at) / denominator
        close = np.abs(denominator) < 1e-12 * np.maximum(energy**2, 1.0)
        if np.any(close):  # removable point: the derivative of E eps2 over 2E, by a central step
            step = 1e-6 * np.maximum(energy[close], 1e-3)
            g = lambda x: x * eps2(x)  # noqa: E731
            value = np.where(close, 0.0, value)
            value[close] = (g(energy[close] + step) - g(energy[close] - step)) / (2 * step) / (2 * energy[close])
        return value

    top = 10.0 * max(np.max(energy), *(abs(b) for b in breakpoints), 1.0)
    points = sorted({float(b) for b in [*breakpoints, *energy] if 0 < b < top})
    finite, _ = integrate.quad_vec(integrand, 0.0, top, epsrel=rtol, epsabs=0.0, points=points, limit=4000)
    tail, _ = integrate.quad_vec(integrand, top, np.inf, epsrel=rtol, epsabs=0.0, limit=4000)
    return 2 / np.pi * (finite + tail)
