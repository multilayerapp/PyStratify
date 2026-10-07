"""Scaled cylindrical functions; no unscaled high-order Hankel evaluation."""

import numpy as np
from scipy.special import gammaln, hankel1e, jve

from .response import log_add


def cylinder_logs(z, maximum):
    z = complex(z)
    if z == 0:
        raise ValueError("cylindrical argument is zero at a light line")
    orders = np.arange(maximum + 2)
    values = jve(orders, z)
    with np.errstate(all="ignore"):
        regular = np.log(values.astype(complex)) + abs(z.imag)
    for m in np.flatnonzero(~np.isfinite(regular)):
        term, series = 1 + 0j, 1 + 0j
        for j in range(1, 10000):
            term *= -z * z / (4 * j * (m + j))
            series += term
            if abs(term) < 2e-16 * abs(series):
                break
        else:
            raise ArithmeticError("cylindrical regular-function series did not converge")
        regular[m] = m * np.log(z / 2) - gammaln(m + 1) + np.log(series)
    with np.errstate(all="ignore"):
        outgoing = np.log(hankel1e(orders, z)) + 1j * z
    # Scaled direct evaluation avoids a Python recurrence at ordinary orders;
    # logarithmic recurrence retains orders whose scaled Hankel overflows.
    for m in np.flatnonzero(~np.isfinite(outgoing)):
        if m < 2:
            raise ArithmeticError("cylindrical outgoing-function seeds exceeded numerical precision")
        outgoing[m] = log_add(np.log(2 * (m - 1) / z) + outgoing[m - 1], outgoing[m - 2] + 1j * np.pi)
    with np.errstate(all="ignore"):
        dj = orders[:-1] / z - np.exp(regular[1:] - regular[:-1])
        dh = orders[:-1] / z - np.exp(outgoing[1:] - outgoing[:-1])
    return regular[:-1], outgoing[:-1], dj, dh


def cylinder_pair(q, radius, orders):
    orders = np.asarray(orders, int)
    lj, lh, dj, dh = cylinder_logs(q * radius, int(np.max(np.abs(orders))))
    indices = np.abs(orders)
    sign = np.where((orders < 0) & (indices % 2 == 1), 1j * np.pi, 0j)
    result = []
    for log_f, deriv in ((lj[indices] + sign, dj[indices]), (lh[indices] + sign, dh[indices])):
        scale = np.maximum(1, np.abs(deriv))
        result.append((1 / scale, deriv / scale, log_f + np.log(scale)))
    return result
