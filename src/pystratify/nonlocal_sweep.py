"""Numerically stable layered response with longitudinal (hydrodynamic) waves.

One recursion serves films, concentric cylinders and spheres.  Each mode of a
separable geometry (an in-plane wavenumber and polarisation; an azimuthal order
and axial wavenumber; a multipole order) is a radial (normal) problem with
*channels*: the transverse waves of a region (one, or two when TE and TM
couple) and, in a hydrodynamic region, the longitudinal wave.  Channel counts
may differ from region to region.

Basis functions and their traces.  In region j every channel c has a regular
solution f_c and an outgoing solution g_c (the solution that is finite at the
centre, or that travels towards the exit side of a film, and its partner).  At
a radius r a basis function is represented by its *trace*, the vector

    [E_t (n_t components), H_t (n_t), J_n, M]

of tangential fields, the normal free-electron current (up to -i omega eps_0)

    J_n = (eps_T - eps_b) E_n^T - eps_b E_n^L,

and the electrochemical-potential perturbation (beta^2 / omega_p^2) div J, up to
the same factor,

    M = eps_T / (eps_b - eps_T) Phi,          E_L = grad Phi,

both zero outside hydrodynamic regions and M zero for transverse waves.  Traces
are *normalised* by a complex scale nu_c(r) of the function (its value times
max(1, |log-derivative|) for Bessel functions, the plane wave itself for films),
and amplitudes are carried as A^ = A nu(r): every number below is O(1) or a
ratio of a function at two radii taken in the direction in which it decays.

Interface conditions at radius r_j between region j (inside) and j + 1:
E_t and H_t continuous; a hard wall J_n = 0 on the side of a hydrodynamic
region facing a region without an electron gas; J_n and M continuous between
two hydrodynamic regions.  There are always c_j + c_(j+1) conditions.

Recursion.  The regular solution (no outgoing wave in region 0) is carried
outwards as a response matrix R, B^ = R A^, the outgoing solution (no regular
wave in the last region) inwards as S, A^ = S B^.  An interface step is one
square linear solve of size c_j + c_(j+1); a step across region j multiplies
R_ab by [nu_g,a(r_j) / nu_g,a(r_(j-1))] [nu_f,b(r_(j-1)) / nu_f,b(r_j)], a
product of two ratios in their decaying directions.  The longitudinal wave of
one surface reaches the other only through these factors, which underflow
harmlessly to zero when it cannot, and are kept exactly when it can (above the
plasma frequency, thin layers, high orders).  Eliminating it instead in a
transfer matrix solves for the amplitude of one surface from the equations of
the other and loses ~exp(Im k_L d) of the working precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Traces", "ChannelSweep", "channel_sweep", "interface_rows"]


@dataclass(frozen=True)
class Traces:
    """Normalised traces of the regular (F) and outgoing (G) functions of one region at one
    radius, shape (..., n_comp, c), and their complex log scales LF, LG, shape (..., c).

    ``U`` (..., n_comp, n_comp), when given, maps the stored rows to the physical traces:
    a basis may store an exact row combination of the continuity conditions (the same on both
    sides of an interface), such as H_t minus the quasi-static admittance times E_t, computed
    analytically so that nearly parallel high-order traces do not cancel."""

    F: np.ndarray
    G: np.ndarray
    LF: np.ndarray
    LG: np.ndarray
    U: np.ndarray | None = None

    def physical(self, vectors):
        """Physical traces from stored ones, vectors (..., n_comp, k)."""
        return vectors if self.U is None else self.U @ vectors


def interface_rows(n_t, hydro_in, hydro_out):
    """Row selectors (C_in, C_out), each (rows, 2 n_t + 2): the conditions read
    C_out . trace_out = C_in . trace_in."""
    n_comp = 2 * n_t + 2
    jn, mu = 2 * n_t, 2 * n_t + 1
    rows_in, rows_out = [], []
    for c in range(2 * n_t):  # E_t and H_t continuous
        e = np.zeros(n_comp)
        e[c] = 1
        rows_in.append(e)
        rows_out.append(e)
    if hydro_in and hydro_out:  # two electron gases: n.J and the potential perturbation continuous
        for c in (jn, mu):
            e = np.zeros(n_comp)
            e[c] = 1
            rows_in.append(e)
            rows_out.append(e)
    elif hydro_in or hydro_out:  # hard wall on the hydrodynamic side
        e = np.zeros(n_comp)
        e[jn] = 1
        rows_in.append(e if hydro_in else np.zeros(n_comp))
        rows_out.append(np.zeros(n_comp) if hydro_in else e)
    return np.array(rows_in), np.array(rows_out)


def _solve(K, rhs):
    """Batched square solve with row equilibration; non-finite results raise."""
    scale = np.max(np.abs(K), axis=-1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    with np.errstate(all="ignore"):
        out = np.linalg.solve(K / scale, rhs / scale)
    if not np.all(np.isfinite(out)):
        raise ArithmeticError("singular interface matching (an exact eigenmode of the structure)")
    return out


def _decay(log_to, log_from):
    """exp(log_to - log_from): a function's ratio between two radii (bounded in its decaying direction)."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = np.exp(log_to - log_from)
    out = np.where(np.isfinite(log_from) & ~np.isfinite(log_to) & (np.real(log_to) < 0), 0, out)
    if not np.all(np.isfinite(out)):
        raise ArithmeticError("basis-function ratio exceeded floating-point range")
    return out


@dataclass
class ChannelSweep:
    """Two-sided responses of a layered structure, per interface j (radius ``radii[j]``).

    ``inner[j]`` / ``outer[j]``: traces of region j / j + 1 at r_j.  ``R_in[j]``, ``R_out[j]``:
    regular-solution responses of region j / j + 1 at r_j; ``S_in[j]``, ``S_out[j]``: outgoing-
    solution responses; ``X[j]`` maps A^_(j+1)(r_j) to A^_j(r_j) for the regular solution and
    ``Y[j]`` maps B^_j(r_j) to B^_(j+1)(r_j) for the outgoing one.
    """

    radii: np.ndarray
    n_t: int
    hydro: tuple
    inner: list
    outer: list
    R_in: list
    R_out: list
    X: list
    S_in: list
    S_out: list
    Y: list

    @property
    def count(self):
        return len(self.radii)

    def channels(self, region):
        tr = self.inner[region] if region < self.count else self.outer[-1]
        return tr.F.shape[-1]

    def log_scales(self, region, interface):
        """(LF, LG) of region at interface radius (interface = region - 1 or region)."""
        tr = self.inner[interface] if interface == region else self.outer[interface]
        return tr.LF, tr.LG

    def regular_factor(self, region, to_interface, from_interface):
        """nu_f(r_to) / nu_f(r_from) of every channel of ``region``, shape (..., c)."""
        return _decay(self.log_scales(region, to_interface)[0], self.log_scales(region, from_interface)[0])

    def outgoing_factor(self, region, to_interface, from_interface):
        return _decay(self.log_scales(region, to_interface)[1], self.log_scales(region, from_interface)[1])

    # ------------------------------------------------------------------ amplitudes
    def regular_amplitudes(self, a_host):
        """Amplitudes of the regular solution with host regular amplitude ``a_host`` (scaled at
        the outermost radius), shape (..., c_N, k).  Returns per region (A^, B^) at its inner
        boundary (None for region 0) and its outer boundary (None for the host), each as
        (array, log) with the array's magnitude carried in the real log (..., k)."""
        N = self.count
        out = [None] * (N + 1)
        A, log = _normalise(a_host, np.zeros(a_host.shape[:-2] + a_host.shape[-1:]))
        out[N] = ((A, self.R_out[N - 1] @ A, log), None)
        for j in range(N - 1, -1, -1):
            # A here is A^ of region j + 1 at r_j; X[j] carries it into region j
            a_o, log = _normalise(self.X[j] @ A, log)
            outer = (a_o, self.R_in[j] @ a_o, log)
            inner = None
            if j:
                a_i, log = _normalise(self.regular_factor(j, j - 1, j)[..., :, None] * a_o, log)
                inner = (a_i, self.R_out[j - 1] @ a_i, log)
                A = a_i
            out[j] = (inner, outer)
        return out

    def outgoing_amplitudes(self, region, b_outer):
        """Outgoing solution continued outwards from ``region``: ``b_outer`` is B^ of ``region`` at
        its outer radius (..., c, k).  Returns per region > ``region`` (A^, B^, log) at its inner
        boundary and at its outer boundary (None for the host)."""
        N = self.count
        out = {}
        B, log = _normalise(b_outer, np.zeros(b_outer.shape[:-2] + b_outer.shape[-1:]))
        for j in range(region, N):
            b_in = self.Y[j] @ B
            b_in, log = _normalise(b_in, log)
            inner = (self.S_out[j] @ b_in, b_in, log)
            if j + 1 < N:
                b_o = self.outgoing_factor(j + 1, j + 1, j)[..., :, None] * b_in
                b_o, log = _normalise(b_o, log)
                outer = (self.S_in[j + 1] @ b_o, b_o, log)
                B = b_o
            else:
                outer = None
            out[j + 1] = (inner, outer)
        return out


def _normalise(v, log):
    """(v / max|v|, log + log max|v|) over the channel axis (-2), per column."""
    m = np.max(np.abs(v), axis=-2)
    safe = np.where(m > 0, m, 1.0)
    return v / safe[..., None, :], log + np.log(safe)


def channel_sweep(traces, radii, n_t, hydro):
    """Run both recursions.  ``traces(region, interface)`` returns the :class:`Traces` of
    ``region`` at ``radii[interface]``; ``hydro[j]`` says whether region j carries a
    longitudinal channel (regions 0..N, N = len(radii))."""
    radii = np.asarray(radii, float)
    N = len(radii)
    inner = [traces(j, j) for j in range(N)]
    outer = [traces(j + 1, j) for j in range(N)]
    rows = [interface_rows(n_t, hydro[j], hydro[j + 1]) for j in range(N)]
    R_in, R_out, X = [None] * N, [None] * N, [None] * N
    for j in range(N):
        ti, to = inner[j], outer[j]
        ci = ti.F.shape[-1]
        if j == 0:
            R = np.zeros(ti.F.shape[:-2] + (ci, ci), complex)
        else:
            gain = _decay(ti.LG, outer[j - 1].LG)  # nu_g(r_j) / nu_g(r_{j-1}), region j
            loss = _decay(outer[j - 1].LF, ti.LF)  # nu_f(r_{j-1}) / nu_f(r_j)
            R = gain[..., :, None] * R_out[j - 1] * loss[..., None, :]
        R_in[j] = R
        Ci, Co = rows[j]
        K = np.concatenate((Ci @ (ti.F + ti.G @ R), -(Co @ to.G)), axis=-1)
        Z = _solve(K, Co @ to.F)
        X[j], R_out[j] = Z[..., :ci, :], Z[..., ci:, :]
    S_in, S_out, Y = [None] * N, [None] * N, [None] * N
    for j in range(N - 1, -1, -1):
        ti, to = inner[j], outer[j]
        ci, co = ti.F.shape[-1], to.F.shape[-1]
        if j == N - 1:
            S = np.zeros(to.F.shape[:-2] + (co, co), complex)
        else:
            loss = _decay(to.LF, inner[j + 1].LF)  # nu_f(r_j) / nu_f(r_{j+1}), region j + 1
            gain = _decay(inner[j + 1].LG, to.LG)  # nu_g(r_{j+1}) / nu_g(r_j)
            S = loss[..., :, None] * S_in[j + 1] * gain[..., None, :]
        S_out[j] = S
        Ci, Co = rows[j]
        K = np.concatenate((-(Ci @ ti.F), Co @ (to.F @ S + to.G)), axis=-1)
        W = _solve(K, Ci @ ti.G)
        S_in[j], Y[j] = W[..., :ci, :], W[..., ci:, :]
    return ChannelSweep(radii, n_t, tuple(bool(h) for h in hydro), inner, outer, R_in, R_out, X, S_in, S_out, Y)
