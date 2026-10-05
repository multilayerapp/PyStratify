"""Bounded adaptive integration with explicit error and evaluation counts."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec


@dataclass(frozen=True)
class Integral:
    value: np.ndarray
    error: float
    evaluations: int
    converged: bool


def integrate(function, breaks, tolerance=1e-6, max_evaluations=20000):
    evaluations = 0

    def bounded(x):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_evaluations:
            raise ArithmeticError("adaptive integration evaluation budget exceeded")
        value = np.asarray(function(x))
        if not np.all(np.isfinite(value)):
            raise ArithmeticError("non-finite spectral integrand")
        return value

    breaks = np.unique(np.asarray(breaks, float))
    value, error, success = None, 0.0, True
    for a, b in zip(breaks[:-1], breaks[1:]):
        def mapped(t):
            x = a + (b - a) * np.sin(t) ** 2
            x = min(np.nextafter(b, a), max(np.nextafter(a, b), x))
            return bounded(x) * (b - a) * np.sin(2 * t)
        part, err, info = quad_vec(mapped, 0, np.pi / 2, epsabs=tolerance / (4 * len(breaks)),
                                  epsrel=tolerance / 4, limit=max_evaluations // 21, full_output=True)
        value = part if value is None else value + part
        error += err
        success = success and info.success
    converged = bool(success and error <= tolerance * max(1.0, float(np.max(np.abs(value)))))
    return Integral(value, float(error), evaluations, converged)
