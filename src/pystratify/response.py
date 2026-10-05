"""Geometry-independent projective composition of layered Maxwell responses."""

from dataclasses import dataclass

import numpy as np


def log_value(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a)


def log_add(x, y):
    with np.errstate(all="ignore"):
        scale = np.maximum(np.real(x), np.real(y))
        scale = np.where(np.isfinite(scale), scale, 0.0)
        return scale + np.log(np.exp(x - scale) + np.exp(y - scale))


def log_transfer(a, b, c, d, log_t):
    a, b, c, d, log_t = np.broadcast_arrays(a, b, c, d, log_t)
    safe = np.abs(log_t.real) < 600
    with np.errstate(all="ignore"):
        t = np.exp(np.where(safe, log_t, 0))
        out = log_value((a + t * b) / (c + t * d))
        if not safe.all():
            u = ~safe
            out = out.copy()
            out[u] = log_add(log_value(a[u]), log_t[u] + log_value(b[u])) - log_add(
                log_value(c[u]), log_t[u] + log_value(d[u])
            )
    return out


def amplitude_log_ratio(c_value, c_deriv, value, deriv):
    (v_out, v_out_scale), (v_in, v_in_scale) = value
    (d_out, d_out_scale), (d_in, d_in_scale) = deriv
    with np.errstate(all="ignore"):
        cond_v = np.minimum(np.abs(v_out) / v_out_scale, np.abs(v_in) / v_in_scale)
        cond_d = np.minimum(np.abs(d_out) / d_out_scale, np.abs(d_in) / d_in_scale)
        return np.where(cond_v >= cond_d, log_value(c_value * v_out / v_in), log_value(c_deriv * d_out / d_in))


@dataclass(frozen=True)
class Response:
    regular_in: np.ndarray
    regular_out: np.ndarray
    outgoing_in: np.ndarray
    outgoing_out: np.ndarray
    logarithmic: bool = False


def sweep(coefficients, propagation, *, logarithmic=False):
    """Two-sided scalar composition. Interface is axis 0, remaining axes are batches.

    Each interface maps rho -> (a + b rho)/(c + d rho); the inverse
    maps sigma -> (d - b sigma)/(a sigma - c). Propagation carries a
    reflection between adjacent interfaces. Boundary reflection is zero.
    """
    a, b, c, d = np.broadcast_arrays(*coefficients)
    count = a.shape[0]
    propagation = np.broadcast_to(propagation, (max(0, count - 1),) + a.shape[1:])
    shape = a.shape
    zero = -np.inf + 0j if logarithmic else 0j
    ri, ro, si, so = (np.full(shape, zero, complex) for _ in range(4))
    with np.errstate(all="ignore"):
        for j in range(count):
            if j:
                ri[j] = ro[j - 1] + propagation[j - 1] if logarithmic else ro[j - 1] * propagation[j - 1]
            ro[j] = log_transfer(a[j], b[j], c[j], d[j], ri[j]) if logarithmic else (a[j] + b[j] * ri[j]) / (c[j] + d[j] * ri[j])
        for j in range(count - 1, -1, -1):
            if j < count - 1:
                so[j] = si[j + 1] + propagation[j] if logarithmic else si[j + 1] * propagation[j]
            si[j] = log_transfer(d[j], -b[j], -c[j], a[j], so[j]) if logarithmic else (d[j] - b[j] * so[j]) / (a[j] * so[j] - c[j])
    return Response(ri, ro, si, so, logarithmic)


def right_solve(a, b):
    return np.swapaxes(np.linalg.solve(np.swapaxes(b, -1, -2), np.swapaxes(a, -1, -2)), -1, -2)


def block_sweep(transfers, propagation):
    """Coupled-polarization version of :func:`sweep`, with two-sided closure."""
    count, size = len(transfers), transfers.shape[-1] // 2
    shape = transfers.shape[:-2] + (size, size)
    ri, ro, si, so = (np.zeros(shape, complex) for _ in range(4))
    for j in range(count):
        if j:
            ri[j] = propagation[j - 1] * ro[j - 1]
        c11, c12 = transfers[j, ..., :size, :size], transfers[j, ..., :size, size:]
        c21, c22 = transfers[j, ..., size:, :size], transfers[j, ..., size:, size:]
        ro[j] = right_solve(c21 + c22 @ ri[j], c11 + c12 @ ri[j])
    for j in range(count - 1, -1, -1):
        if j < count - 1:
            so[j] = propagation[j] * si[j + 1]
        c11, c12 = transfers[j, ..., :size, :size], transfers[j, ..., :size, size:]
        c21, c22 = transfers[j, ..., size:, :size], transfers[j, ..., size:, size:]
        si[j] = np.linalg.solve(c11 - so[j] @ c21, so[j] @ c22 - c12)
    return Response(ri, ro, si, so)


def source_coefficients(regular, outgoing, direct_regular, direct_outgoing, *, blocks=False):
    """Scattered regular/outgoing amplitudes at a source in a two-sided cavity."""
    if blocks:
        identity = np.eye(regular.shape[-1])
        b = np.linalg.solve(identity - regular @ outgoing, regular @ (direct_regular + outgoing @ direct_outgoing))
        return outgoing @ (b + direct_outgoing), b
    b = regular * (direct_regular + outgoing * direct_outgoing) / (1 - regular * outgoing)
    return outgoing * (b + direct_outgoing), b


def emitted_amplitudes(left_response, right_response, direct_left, direct_right):
    """Total two-sided emission including all repeated reflections."""
    denominator = 1 - left_response * right_response
    return ((direct_left + right_response * direct_right) / denominator,
            (direct_right + left_response * direct_left) / denominator)
