"""The stable coherent planar kernel: scattering matrices for stacks of any length.

`planar.coh_tmm` solves one wavelength and one angle per call, in a Python pass over every layer. This module
solves the same coherent stack over a whole grid of points (wavelengths, angles, or both, broadcast together) by
composing SCATTERING matrices with the Redheffer star product. A section is four amplitudes ``(r, t, r', t')`` --
reflection and transmission from the left, then from the right -- each bounded for a passive stack, and every layer
is put on its decaying (forward) branch, so no intermediate quantity grows: the result is finite for any layer
count and any absorption. A block repeated ``k`` times is built once and raised to the ``k``-th power by binary
doubling (``ceil(log2 k)`` squarings), so the 131072 layers of ``(n1 n2)^65536`` cost 16 squarings.

It is the one kernel behind both of multilayer.app's former copies (WP07-D1, Ilia 2026-11-03): the replication
corpus's ``harvester/tmm_stable.py`` with the Bloch-terminated exit of ``harvester/verify.py``, and the calculator's
``multilayer/backend/tmm_stable.py``. It covers everything either did:

* isotropic coherent stacks, s and p and the unpolarised mean, complex ``r``, ``t`` and ``R``, ``T``;
* repeated stacks both ways: explicit, nestable ``Repeat`` groups (the API's segments), and automatic detection of
  a flat stack that is ``k`` exact repetitions of a block (the harvester's way; exact equality of n, n_z and d);
* uniaxial layers with the optic axis along the stack normal (``Layer.nz``);
* a perfect-electric-conductor exit (``PEC_INDEX``);
* a Bloch-terminated exit: a semi-infinite periodic crystal behind the stack (``BlochExit``).

Conventions (``planar.coh_tmm``'s, which are Byrnes's ``tmm``'s)
------------------------------------------------------------------
* ``exp(-i w t)``, ``n = n' + i k`` with ``k >= 0`` for loss; angles by complex Snell's law ``n_j sin(th_j) =
  n_0 sin(th_0)`` with ``numpy.lib.scimath.arcsin``; every medium takes the FORWARD branch of
  ``planar.is_forward_angle`` (the decaying wave, or positive Poynting flux). ``coh_tmm`` fixes only the two
  half-spaces because interior branches cancel in its product; here every layer is put on the forward branch,
  which leaves r and t unchanged and keeps a star product from propagating a growing exponential.
* s and p Fresnel amplitudes are ``planar._interface_coefficients``'s: ``r_s = (n_i c_i - n_f c_f)/(n_i c_i + n_f
  c_f)``, ``r_p = (n_f c_i - n_i c_f)/(n_f c_i + n_i c_f)``, ``t = 2 n_i c_i / denominator``; the reverse
  amplitudes are ``r' = -r`` and ``t' = 2 n_f c_f / denominator`` (= ``(1 - r^2)/t``, coh_tmm's interface matrix).
  So r_p = -r_s at normal incidence, and ellipsometry's ``-r_p/r_s`` is ``planar.ellips``'s.
* ``r`` is referred to the first interface and ``t`` to the last, as ``coh_tmm``'s are; ``R = |r|^2`` and ``T`` is
  ``planar.T_from_t``'s power-flux normalisation (``Re(n_f c_f)/Re(n_i c_i)`` for s, the conjugated-cosine form for
  p), so an absorbing exit medium gives ``coh_tmm``'s T. ``planar.coh_tmm`` and this kernel agree to ~1e-14
  (``benchmarks/STABLE_CHECKS.md``); both agree with Byrnes's ``tmm`` to ~1e-14 (``benchmarks/BYRNES_TMM.md``; over
  the replication corpus ``coh_tmm`` and ``tmm`` agree to <= 3.5e-12).

The physics the two former kernels added
----------------------------------------
**Uniaxial layers** (optic axis along the normal; ``eps = diag(eps_x, eps_x, eps_z)``, ``Layer.n = sqrt(eps_x)``,
``Layer.nz = sqrt(eps_z)``). s sees only eps_x: the layer is the isotropic medium ``n`` for s, exactly. For p (H
along y) Maxwell's equations give ``k_z = k0 q`` with ``q^2 = eps_x (1 - beta^2/eps_z)`` (``beta = n_0 sin th_0``)
and the TM admittance ``H_y/E_x = eps_x/q``. The interface and phase formulas read a medium only through the
pair ``(n, cos)`` -- admittance ``n/cos`` and phase ``k0 n cos d`` -- and the pair ``(n_x, cos(arcsin(beta/n_z)))``
reproduces both (``n cos = q``, ``n/cos = eps_x/q``); the amplitude normalisation it implies inside the layer
cancels between the layer's two faces. ``nz = n`` is the isotropic expression itself (identical numbers), and at
normal incidence ``cos = 1`` for any ``nz``. The half-spaces are isotropic. (The harvester's derivation,
``harvester/tmm_stable.py``; checked here against a direct integration of Maxwell's equations.)

**Perfect electric conductor exit** (``n_out = PEC_INDEX = i*inf``, eps -> -inf). The tangential E vanishes at the
last interface: ``r = -1`` (s) and ``r = +1`` (p) there, the limit of the Fresnel amplitudes as ``n_f -> i*inf``,
set exactly rather than through a large finite index; ``t = 0``, ``T = 0``.

**Bloch-terminated exit** (``n_out = BlochExit(p)``): the last ``p`` layers of the stack are one period of a
crystal that continues semi-infinitely; the layers before them are the finite stack in front of it. The period's
cell S-matrix ``C`` (inside its first layer, front face to the next period's front face) has Bloch multipliers
``lambda`` from ``t' lambda^2 - (1 + t t' - r r') lambda + t = 0`` (the eigenvalues of its transfer matrix), and
the wave with multiplier ``lambda`` has reflection ``rho = r/(1 - t' lambda)`` seen from inside the first layer at
the crystal's face. The physical wave is the harvester's choice: the one that decays, ``|lambda| < 1``, when the
two moduli differ by more than 1e-9 in their logarithms; otherwise (a lossless pass band) the one whose admittance
``Y = H/E`` has the larger real part (power flowing into the crystal). The crystal is then a terminal section of
reflection ``rho``; ``T`` is the power flux entering it, ``max(F, 0)`` with F the Poynting flux of the forward and
reflected waves at the face, so a lossless crystal has ``R + T = 1`` and ``R = 1`` in a stop band. ``t`` is the
forward amplitude in the crystal's first layer at its face (planar's amplitude normalisation there). This is
``verify.bloch_exit_rt``'s physics (eigenvectors of the period's (E, H) transfer matrix, the front stack carried
back by solving) in scattering form: the same selection, vectorised, and stable for any front stack.

**A layer at its own critical angle.** Where ``cos(theta) -> 0`` inside a layer its two plane waves coincide and
the star product through them loses ~eps/|cos| (0/0 at cos = 0; ``planar.coh_tmm`` has the same singularity). There
(|cos| < ``NEAR_CRITICAL_COS``, |Im delta| < ``NEAR_CRITICAL_IM_DELTA``) the layer is taken from its characteristic
matrix, smooth at cos = 0, referred to a zero-thickness auxiliary medium on both faces (``_near_critical_body``) --
what the harvester's Bloch exit did with its (E, H) matrices.

Inputs
------
``stable_tmm(pol, stack, n_in, n_out, th_0, lam_vac, detect_repeats=True)``:

* ``pol``: ``"s"``, ``"p"`` or ``"unpolarized"`` (the mean of s and p, the per-polarization results beside it).
* ``stack``: the finite layers front to back. Each entry is ``Layer(n, d, nz=None)``, ``Repeat(members, count)``
  (members are entries again, so groups nest), or the tuple forms ``(n, d)``, ``(n, d, nz)`` and the API's
  ``("group", members, count)``. ``n`` and ``nz`` broadcast to the points axis (a scalar, or an array over
  wavelengths); ``d`` is a scalar in the unit of ``lam_vac``.
* ``n_in``: the incident half-space (isotropic). ``n_out``: the exit half-space, ``PEC_INDEX`` (or an array that is
  ``i*inf`` everywhere), or ``BlochExit(period_layers)`` -- any object with an integer ``period_layers`` attribute
  is read as one, so the harvester's ``verify.BlochExit`` marker passes through.
* ``th_0`` (radians, in the incident medium) and ``lam_vac`` broadcast with the indices: e.g. ``th_0`` of shape
  ``(A, 1)`` against wavelengths ``(W,)`` gives ``(A, W)`` results.
* ``detect_repeats``: a flat stack (no ``Repeat``) that is ``k > 1`` exact repetitions of a block is raised to the
  ``k``-th power (``block_period``); a stack with explicit groups is taken as written. Either way the result is the
  literal stack's to rounding (1e-12).

It returns ``{"r", "t", "R", "T"}``, arrays over the broadcast points shape (for ``"unpolarized"``:
``{"R", "T", "s", "p"}``).

Refusals (``ValueError``)
-------------------------
* a lossy incident medium at oblique incidence, ``|Im(n_0 sin th_0)| >= 100 eps`` (``coh_tmm``'s and both former
  kernels' rule; the corpus's own k/n <= 1e-5 zeroing stays the harvester's ruling, ``incident_medium_for_tmm``);
* an angle that is not the incoming (forward) wave in ``n_in`` (``coh_tmm`` asserts it), and gain (``n' k < 0``) in
  either half-space, where incoming and outgoing waves are ambiguous (``planar.is_forward_angle``);
* with a Bloch exit, any loss in the incident medium beyond ``1e-9 max(|n'|, 1)`` at any angle (a smaller k is
  dropped), as ``verify.bloch_exit_rt``; a period longer than the stack, or one that is not plain layers;
* a perfect conductor as the incident medium, or an exit that is ``i*inf`` at only some points;
* a polarization other than s, p or unpolarized; a negative or non-finite thickness, a non-positive wavelength, a
  repeat count below 1, an empty group, an entry that is none of the forms above.

The kernel is coherent and non-magnetic (mu = 1): an incoherent layer cannot be expressed, so callers keep their
own refusals for it (the API refuses incoherent and magnetic layers inside repeat groups; ``verify.py`` refuses a
uniaxial, PEC or Bloch stack with incoherent layers).

How the former calls map onto it (WP07 steps b and c)
-----------------------------------------------------
``harvester/verify.py`` (the corpus verifier), with ``from pystratify import stable``:

* ``tmm_stable.coh_rt(p, layers_nk, d_list[1:-1], n_in, n_out, th0, wls, layers_nz=nz)`` returns ``(R, T)``:
  ``res = stable.stable_tmm(p, stable.from_lists(layers_nk, d_list[1:-1], nz), n_in, n_out, th0, wls)`` and
  ``res["R"], res["T"]``. ``use_blocks=False`` is ``detect_repeats=False``; ``tmm_stable.PEC_INDEX`` is
  ``stable.PEC_INDEX`` (the same value, ``complex(0, inf)``); ``tmm_stable.block_period(layers_n, d)`` is
  ``stable.block_period(stable.from_lists(layers_n, d))``.
* the Bloch exit, ``bloch_exit_rt(p, values, d_list[1:-1], n_sub.period_layers, n_super[i], th0, wl)`` per point:
  one vectorised call, ``stable.stable_tmm(p, stable.from_lists(layers_nk, d_list[1:-1]), n_super,
  stable.BlochExit(n_sub.period_layers), th0, wls)`` (or ``n_out=n_sub`` itself: its ``period_layers`` is read).
  ``compute_RTA_angles``'s per-angle loop is the same call with ``th0`` an array of angles.
* ``compute_RTA``'s unpolarised mean is ``pol="unpolarized"`` (or the mean of the s and p calls, as it does now for
  each ``polarization_modes`` split). ``_require_coherent_bloch``, ``_stable_only_inputs``'s incoherent refusal
  and ``incident_medium_for_tmm`` stay in the verifier: they are corpus rulings about inputs, not kernel physics.

``multilayer/backend/app.py`` (the calculator API):

* ``tmm_stable.stack_amplitudes(pol, segments, n_in, n_out, th0, lam)`` -> ``stable.stable_tmm(pol, segments,
  n_in, n_out, th0, lam)``: the segments ``(n, d)`` and ``("group", [(n, d), ...], repeats)`` are accepted as
  they are, the result has the same keys ``r, t, R, T``, and a refusal is a ``ValueError`` (the API's 422).
* ``tmm_stable.expanded_layer_count(segments)`` -> ``stable.expanded_layer_count(segments)``.

Where the two former kernels differed, this one follows: planar's conventions (the API's; the harvester's are
Byrnes's, the same formulas); the forward branch for EVERY layer (the API's; the harvester took the principal
branch inside the stack, equal in r and t); a group's last period closed directly onto what follows (the API's; the
harvester closed every period onto the block's first layer and added that layer's interface to the exit, equal to
rounding); the refusal of a backward or gain incident wave (planar's; the harvester flipped a backward angle
silently); uniaxial layers, PEC and Bloch exits (the harvester's; the API had none). The harvester's Bloch path
ignored ``layers_nz``; here a uniaxial layer is uniaxial in front of and inside the crystal too. Both former
scattering-matrix kernels lost ~eps/|cos| next to a layer's own critical angle (at it, R = 1.0 for s and 0.107 for
p on a stack whose R is 0.337 and 0.094); this one takes such a layer from its characteristic matrix there.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import numpy as np
from numpy.lib.scimath import arcsin as _arcsin

__all__ = [
    "Layer",
    "Repeat",
    "BlochExit",
    "PEC_INDEX",
    "is_pec",
    "stable_tmm",
    "from_lists",
    "block_period",
    "expanded_layer_count",
]

EPSILON = np.finfo(float).eps

#: The exit index of a perfect electric conductor: n = i*inf (eps -> -inf), the harvester's ``PEC_INDEX``.
PEC_INDEX = complex(0.0, np.inf)

#: A Bloch exit refuses an incident medium whose k exceeds this times max(|n'|, 1) (``verify.bloch_exit_rt``).
BLOCH_INCIDENT_LOSS = 1e-9

#: Below this gap between log|lambda| of the two Bloch waves the crystal is in a pass band (``verify``'s 1e-9).
BLOCH_PASS_BAND = 1e-9

POLARIZATIONS = ("s", "p")
UNPOLARIZED = ("unpolarized", "unpolarised", "u")


class Layer(NamedTuple):
    """One finite layer: index ``n`` (the in-plane ``n_x`` of a uniaxial layer), thickness ``d``, and ``nz``,
    the out-of-plane index ``sqrt(eps_z)`` of a uniaxial layer whose optic axis is the stack normal (None: isotropic).
    """

    n: Any
    d: float
    nz: Any = None


class Repeat(NamedTuple):
    """``members`` (layers or groups, front to back), ``count`` times over."""

    members: Sequence
    count: int


class BlochExit(NamedTuple):
    """Exit marker: the stack's last ``period_layers`` layers repeat semi-infinitely (a Bloch-terminated exit)."""

    period_layers: int


def is_pec(n) -> bool:
    """True when every point of ``n`` is the perfect-conductor index ``PEC_INDEX`` (``i*inf``)."""
    if _bloch_marker(n) is not None:
        return False
    n = np.asarray(n, dtype=complex)
    return bool(n.size) and bool(np.all(np.isinf(n.imag) & (n.imag > 0)))


def _bloch_marker(n_out):
    if isinstance(n_out, BlochExit):
        return n_out
    count = getattr(n_out, "period_layers", None)
    if count is None:
        return None
    if isinstance(count, bool) or int(count) != count:
        raise ValueError(f"a Bloch exit needs an integer period_layers, got {count!r}")
    return BlochExit(int(count))


# --- inputs -----------------------------------------------------------------------------------------------------

def _count(value) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"a repeat count must be an integer, got {value!r}") from None
    if isinstance(value, bool) or count != value or count < 1:
        raise ValueError(f"a repeat count must be an integer >= 1, got {value!r}")
    return count


def _normalize(entry):
    """``Layer`` or ``Repeat`` from any accepted form, checked."""
    if isinstance(entry, Repeat):
        members, count = entry.members, entry.count
    elif isinstance(entry, Layer):
        return _checked_layer(entry)
    elif isinstance(entry, (tuple, list)) and len(entry) == 3 and isinstance(entry[0], str):
        if entry[0] != "group":
            raise ValueError(f"a tagged stack entry must be ('group', members, count), got tag {entry[0]!r}")
        _, members, count = entry
    elif isinstance(entry, (tuple, list)) and len(entry) in (2, 3):
        return _checked_layer(Layer(*entry))
    else:
        raise ValueError(f"a stack entry must be a Layer, a Repeat, (n, d), (n, d, nz) or ('group', members, "
                         f"count); got {type(entry).__name__}")
    if isinstance(members, (str, bytes)) or not hasattr(members, "__iter__"):
        raise ValueError("a repeat group's members must be a sequence of layers or groups")
    members = tuple(_normalize(member) for member in members)
    if not members:
        raise ValueError("a repeat group needs at least one member")
    return Repeat(members, _count(count))


def _checked_layer(layer: Layer) -> Layer:
    try:
        d = float(layer.d)
    except (TypeError, ValueError):
        raise ValueError(f"a layer's thickness must be a real scalar, got {layer.d!r}") from None
    if not np.isfinite(d) or d < 0:
        raise ValueError(f"a layer's thickness must be finite and >= 0, got {layer.d!r}")
    return layer if type(layer.d) is float else Layer(layer.n, d, layer.nz)


def from_lists(layers_n, d, layers_nz=None) -> list:
    """The harvester's per-layer lists as a stack: one ``Layer(n, d, nz)`` per finite layer.

    ``layers_n`` holds one index per finite layer (a scalar or an array over the points axis), ``d`` the thicknesses,
    ``layers_nz`` (optional) one out-of-plane index per layer, None where the layer is isotropic.
    """
    if len(layers_n) != len(d):
        raise ValueError("one thickness per finite layer")
    if layers_nz is None:
        return [Layer(n, thickness) for n, thickness in zip(layers_n, d)]
    if len(layers_nz) != len(layers_n):
        raise ValueError("layers_nz needs one entry (None when isotropic) per finite layer")
    return [Layer(n, thickness, nz) for n, thickness, nz in zip(layers_n, d, layers_nz)]


def expanded_layer_count(stack) -> int:
    """How many physical layers ``stack`` holds once every repeat group is written out."""
    total = 0
    for entry in stack:
        entry = entry if isinstance(entry, (Layer, Repeat)) else _normalize(entry)
        total += expanded_layer_count(entry.members) * int(entry.count) if isinstance(entry, Repeat) else 1
    return total


def _same(a, b) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    return bool(np.array_equal(a, b))


def block_period(stack) -> int:
    """The length of the shortest block whose exact repetitions make up a flat stack (its length when none).

    Only divisors of the layer count are tried and each trial stops at its first mismatch, so a periodic stack costs
    O(N). Equality is exact, the harvester's: identity first, then ``np.array_equal`` on n and n_z, ``==`` on d.
    """
    layers = [entry if isinstance(entry, Layer) else _normalize(entry) for entry in stack]
    if any(not isinstance(layer, Layer) for layer in layers):
        raise ValueError("block_period takes a flat stack of layers (no repeat groups)")
    count = len(layers)
    for p in range(1, count // 2 + 1):
        if count % p:
            continue
        if all(layers[i].d == layers[i % p].d and _same(layers[i].n, layers[i % p].n)
               and _same(layers[i].nz, layers[i % p].nz) for i in range(p, count)):
            return p
    return count


# --- the scattering-matrix algebra ------------------------------------------------------------------------------

class _S:
    """``(r, t, r', t')`` of a section, each an array over the points axis."""

    __slots__ = ("r", "t", "rb", "tb")

    def __init__(self, r, t, rb, tb):
        self.r, self.t, self.rb, self.tb = r, t, rb, tb


def _star(a: _S, b: _S) -> _S:
    """Redheffer star product: section ``a`` followed (to its right) by ``b``.

    The multiple-reflection sum between the two is ``1/(1 - a.r' b.r)``, finite for any passive pair.
    """
    inv = 1.0 / (1.0 - a.rb * b.r)
    return _S(a.r + a.t * b.r * a.tb * inv, a.t * b.t * inv, b.rb + b.tb * a.rb * b.t * inv, b.tb * a.tb * inv)


def _power(cell: _S, k: int) -> _S:
    """``cell`` star-multiplied with itself ``k`` times, by binary doubling."""
    result, base = None, cell
    while True:
        if k & 1:
            result = base if result is None else _star(result, base)
        k >>= 1
        if not k:
            return result
        base = _star(base, base)


class _Medium(NamedTuple):
    """A medium as the interface formulas read it: index ``n`` and forward-branch ``c = cos(theta)``."""

    n: Any
    c: Any


_PEC = object()


def _forward_cos(n, angle_index, n0sin):
    """cos(theta) for the conserved ``n0sin`` in ``angle_index``, on the forward branch of the wave ``n cos``.

    ``planar.is_forward_angle``, vectorised: when ``n cos`` has a significant imaginary part the decaying wave
    (Im > 0) is forward, otherwise the one with positive real part. theta -> pi - theta flips the cosine.
    """
    c = np.cos(_arcsin(n0sin / angle_index))
    q = n * c
    forward = np.where(np.abs(q.imag) > 100 * EPSILON, q.imag > 0, q.real > 0)
    return np.where(forward, c, -c)


def _interface(pol, a: _Medium, b) -> _S:
    """From medium ``a`` into medium ``b`` (or onto a perfect conductor)."""
    if b is _PEC:
        shape = np.broadcast_shapes(np.shape(a.n), np.shape(a.c))
        r = np.full(shape, -1.0 if pol == "s" else 1.0, dtype=complex)
        zero = np.zeros(shape, dtype=complex)
        return _S(r, zero, -r, zero)
    if pol == "s":
        left, right = a.n * a.c, b.n * b.c
    else:
        left, right = b.n * a.c, a.n * b.c
    denom = left + right
    return _S((left - right) / denom, 2 * a.n * a.c / denom, (right - left) / denom, 2 * b.n * b.c / denom)


def _layer_section(pol, basis: _Medium, body, after) -> _S:
    """From just inside a layer at its front face, through it, into the medium ``after``.

    ``body`` is the layer's phase ``delta`` (its amplitudes are its own plane waves), or the layer's scattering matrix
    in the auxiliary basis ``basis`` where it is close to its own critical angle (`_near_critical_body`).
    """
    step = _interface(pol, basis, after)
    if isinstance(body, _S):
        return _star(body, step)
    e = np.exp(1j * body)
    return _S(step.r * e * e, step.t * e, step.rb, step.tb * e)


#: A layer whose cos(theta) is below this (and |Im delta| below NEAR_CRITICAL_IM_DELTA) is computed from its
#: characteristic matrix: its forward and backward plane waves become one at cos = 0, where their basis degenerates.
NEAR_CRITICAL_COS = 1e-2
NEAR_CRITICAL_IM_DELTA = 30.0
_AUXILIARY = 1.0 + 0j


def _near_critical_body(pol, n, c, delta, k0d, near):
    """The layer's body, from its front face to its back face, between two zero-thickness auxiliary media.

    Near its own critical angle (``c -> 0``) a layer's two plane waves coincide and a star product through them
    divides a vanishing transmission by a vanishing denominator (error ~ eps/|c|; at ``c = 0`` it is 0/0). The
    tangential fields do not degenerate: through the layer (E, H) -> M (E, H) with ``M = [[cos d, i sin d/eta],
    [i eta sin d, cos d]]`` (eta = n c for s, n/c for p; ``sin d/eta`` and ``eta sin d`` written with sinc, smooth at
    c = 0, as ``verify._layer_forward_matrix``). Referred to an auxiliary medium of pair (n, cos) = (1, 1) on both
    faces -- a zero-thickness layer, so the physics is unchanged -- M becomes the bounded scattering matrix
    ``r = r' = i (q - p)/(2 D)`` (s; the sign flips for p), ``t = t' = 1/D``, ``D = cos d - i (p + q)/2``, with
    ``p = sin d/eta`` and ``q = eta sin d``. Elsewhere the body is plain propagation, ``(0, e, 0, e)``.
    """
    with np.errstate(all="ignore"):
        small = np.abs(delta) < 1e-6
        sinc = np.where(small, 1 - delta * delta / 6, np.sin(delta) / np.where(small, 1, delta))
        if pol == "s":
            p, q = k0d * sinc, k0d * (n * c) ** 2 * sinc
            sign = 1
        else:
            p, q = k0d * c * c * sinc, k0d * n * n * sinc
            sign = -1
        d = np.cos(delta) - 0.5j * (p + q)
        r = sign * 0.5j * (q - p) / d
        t = 1 / d
        e = np.exp(1j * delta)
    r = np.where(near, r, 0)
    t = np.where(near, t, e)
    return _S(r, t, r, t)


class _Stack:
    """Evaluates layers once (memoised by object) and composes sections."""

    def __init__(self, pol, n0sin, k0):
        self.pol, self.n0sin, self.k0 = pol, n0sin, k0
        self.memo = {}

    def terms(self, layer: Layer):
        """``(basis, body)``: the medium the layer's face amplitudes are referred to, and its phase or body."""
        key = id(layer)
        if key not in self.memo:
            n = np.asarray(layer.n, dtype=complex)
            uniaxial_p = layer.nz is not None and self.pol == "p"
            c = _forward_cos(n, np.asarray(layer.nz, dtype=complex) if uniaxial_p else n, self.n0sin)
            delta = self.k0 * n * c * layer.d
            near = (np.abs(c) < NEAR_CRITICAL_COS) & (np.abs(delta.imag) < NEAR_CRITICAL_IM_DELTA)
            if np.any(near):
                basis = _Medium(np.where(near, _AUXILIARY, n), np.where(near, _AUXILIARY, c))
                self.memo[key] = (basis, _near_critical_body(self.pol, n, c, delta, self.k0 * layer.d, near))
            else:
                self.memo[key] = (_Medium(n, c), delta)
        return self.memo[key]

    def head(self, entry):
        while isinstance(entry, Repeat):
            entry = entry.members[0]
        return self.terms(entry)

    def section(self, entry, head, after) -> _S:
        """From just inside ``entry``'s first layer (its terms: ``head``) through it, into ``after``."""
        if isinstance(entry, Layer):
            return _layer_section(self.pol, head[0], head[1], after)
        members = entry.members
        heads = [self.head(member) for member in members]
        prefix = None
        for j in range(len(members) - 1):
            step = self.section(members[j], heads[j], heads[j + 1][0])
            prefix = step if prefix is None else _star(prefix, step)
        last = self.section(members[-1], heads[-1], after)
        out = last if prefix is None else _star(prefix, last)
        if entry.count == 1:
            return out
        closing = self.section(members[-1], heads[-1], heads[0][0])
        period = closing if prefix is None else _star(prefix, closing)
        return _star(_power(period, entry.count - 1), out)

    def compose(self, entries, incident: _Medium, exit_medium) -> _S:
        """Incident medium, the entries, into ``exit_medium``; one entry's layers in memory at a time."""
        if not entries:
            return _interface(self.pol, incident, exit_medium)
        head = self.head(entries[0])
        s = _interface(self.pol, incident, head[0])
        for i, entry in enumerate(entries):
            following = self.head(entries[i + 1]) if i + 1 < len(entries) else None
            after = following[0] if following is not None else exit_medium
            s = _star(s, self.section(entry, head, after))
            self.memo = {} if following is None else {id(_first_layer(entries[i + 1])): following}
            head = following
        return s


def _first_layer(entry) -> Layer:
    while isinstance(entry, Repeat):
        entry = entry.members[0]
    return entry


# --- the Bloch exit ---------------------------------------------------------------------------------------------

def _admittance(pol, medium: _Medium, rho):
    """H/E at a face, inside ``medium``, of a forward wave and its reflection ``rho`` (planar's amplitudes)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if pol == "s":
            return medium.n * medium.c * (1 - rho) / (1 + rho)
        return medium.n / medium.c * (1 + rho) / (1 - rho)


def _bloch_rho(pol, cell: _S, medium: _Medium):
    """Reflection of the semi-infinite crystal of period ``cell``, seen from inside its first layer at its face."""
    r, t, rb, tb = cell.r, cell.t, cell.rb, cell.tb
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        b = 1 + t * tb - r * rb
        disc = np.sqrt(b * b - 4 * t * tb)
        disc = np.where((np.conj(b) * disc).real >= 0, disc, -disc)
        q = (b + disc) / 2                                   # the larger root's numerator: no cancellation
        lam_small, lam_large = t / q, q / tb                 # lam_small * lam_large = t / t' (= 1, reciprocal)
        gap = np.abs(2 * np.log(np.abs(q)) - np.log(np.abs(t)) - np.log(np.abs(tb)))

        def rho_of(lam):
            # The eigenvector's two rows; the better-conditioned one (both vanish only for a reflectionless cell).
            first, second = 1 - tb * lam, rb * lam
            rho = np.where(np.abs(first) >= np.abs(second), r / first, (lam - t) / second)
            return np.where(np.isfinite(rho), rho, np.nan)

        rho_small, rho_large = rho_of(lam_small), rho_of(lam_large)
        real_small = np.nan_to_num(_admittance(pol, medium, rho_small).real, nan=-np.inf)
        real_large = np.nan_to_num(_admittance(pol, medium, rho_large).real, nan=-np.inf)
    decaying = ~(gap <= BLOCH_PASS_BAND)                     # also when a modulus is 0 or the gap is inf / nan
    pick_small = np.where(decaying, True, real_small >= real_large)
    rho = np.where(pick_small, rho_small, rho_large)
    # An opaque cell (t = 0) is its own reflection; a degenerate reflectionless one (lambda = +-1 twice) passes
    # the forward wave, rho = r = 0.
    return np.where(np.isfinite(rho) & (t * tb != 0), rho, r)


def _bloch_flux(pol, medium: _Medium, rho):
    """Poynting flux into the crystal at its face for a forward amplitude 1 and reflection ``rho``."""
    if pol == "s":
        return (medium.n * medium.c * (1 + np.conj(rho)) * (1 - rho)).real
    return (medium.n * np.conj(medium.c) * (1 + rho) * (1 - np.conj(rho))).real


# --- the kernel -------------------------------------------------------------------------------------------------

def stable_tmm(pol, stack, n_in, n_out, th_0, lam_vac, *, detect_repeats=True):
    """Coherent ``r``, ``t``, ``R``, ``T`` of a planar stack over a grid of points (see the module docstring).

    ``pol`` is ``"s"``, ``"p"`` or ``"unpolarized"``; ``stack`` the finite layers front to back (``Layer``,
    ``Repeat``, ``(n, d)``, ``(n, d, nz)`` or ``("group", members, count)``); ``n_in`` the incident half-space;
    ``n_out`` the exit half-space, ``PEC_INDEX`` or ``BlochExit(period_layers)``; ``th_0`` the angle in the incident
    medium (radians) and ``lam_vac`` the vacuum wavelength, in the unit of the thicknesses. Indices, ``th_0`` and
    ``lam_vac`` broadcast together. Returns ``{"r", "t", "R", "T"}`` (for ``"unpolarized"``: ``{"R", "T", "s",
    "p"}``, R and T the means). Raises ``ValueError`` on the inputs the module docstring lists.
    """
    entries = [_normalize(entry) for entry in stack]
    if pol in UNPOLARIZED:
        s = _solve("s", entries, n_in, n_out, th_0, lam_vac, detect_repeats)
        p = _solve("p", entries, n_in, n_out, th_0, lam_vac, detect_repeats)
        return {"R": (s["R"] + p["R"]) / 2, "T": (s["T"] + p["T"]) / 2, "s": s, "p": p}
    if pol not in POLARIZATIONS:
        raise ValueError(f"polarization must be 's', 'p' or 'unpolarized', got {pol!r}")
    return _solve(pol, entries, n_in, n_out, th_0, lam_vac, detect_repeats)


def _solve(pol, entries, n_in, n_out, th_0, lam_vac, detect_repeats):
    bloch = _bloch_marker(n_out)
    n_in = np.asarray(n_in, dtype=complex)
    if np.any(np.isinf(n_in)):
        raise ValueError("the incident medium cannot be a perfect conductor (or any infinite index)")
    if np.any(n_in.real * n_in.imag < 0):
        raise ValueError("gain in the incident medium (n' k < 0) makes the incoming wave ambiguous "
                         "(planar.is_forward_angle)")
    if bloch is not None:
        loss = np.abs(n_in.imag)
        if np.any(loss > BLOCH_INCIDENT_LOSS * np.maximum(np.abs(n_in.real), 1.0)):
            raise ValueError("the incident medium must be lossless for a Bloch exit: a lossy incident medium has "
                             "no unambiguous incident/reflected split (verify.bloch_exit_rt)")
        n_in = n_in.real.astype(complex)
    th0 = np.asarray(th_0, dtype=complex)
    lam = np.asarray(lam_vac, dtype=float)
    if np.any(~(lam > 0)) or np.any(~np.isfinite(lam)):
        raise ValueError("lam_vac must be positive and finite")
    n0sin = n_in * np.sin(th0)
    if np.any(np.abs(n0sin.imag) >= 100 * EPSILON):
        raise ValueError("Error in n0 or th0: the incident medium must be lossless at oblique incidence "
                         "(|Im(n0 sin th0)| < 100 eps, as planar.coh_tmm requires)")
    c_in = np.cos(th0)
    q_in = n_in * c_in
    forward = np.where(np.abs(q_in.imag) > 100 * EPSILON, q_in.imag > 0, q_in.real > 0)
    if np.any(~forward & np.isfinite(q_in)):                 # a NaN input propagates to NaN results, as before
        raise ValueError("Error in n0 or th0: th_0 is not the incoming (forward) wave in the incident medium")
    incident = _Medium(n_in, c_in)
    k0 = 2 * np.pi / lam
    stack = _Stack(pol, n0sin, k0)

    if bloch is not None:
        count = bloch.period_layers
        if not 1 <= count <= len(entries):
            raise ValueError(f"a Bloch period of {count} layers in a {len(entries)}-entry stack")
        front, period = entries[:len(entries) - count], entries[len(entries) - count:]
        if any(not isinstance(entry, Layer) for entry in period):
            raise ValueError("a Bloch period must be plain layers, not repeat groups")
        exit_shape = ()
    else:
        front, period = entries, None
        if is_pec(n_out):
            exit_medium = _PEC
            exit_shape = np.shape(n_out)
        else:
            n_out = np.asarray(n_out, dtype=complex)
            if np.any(np.isinf(n_out)):
                raise ValueError("the exit index is infinite at some points only: a perfect-conductor exit is "
                                 "PEC_INDEX at every point")
            if np.any(n_out.real * n_out.imag < 0):
                raise ValueError("gain in the exit medium (n' k < 0) makes the outgoing wave ambiguous "
                                 "(planar.is_forward_angle)")
            exit_medium = _Medium(n_out, _forward_cos(n_out, n_out, n0sin))
            exit_shape = n_out.shape

    if detect_repeats and len(front) > 1 and all(isinstance(entry, Layer) for entry in front):
        p = block_period(front)
        if len(front) // p > 1:
            front = [Repeat(tuple(front[:p]), len(front) // p)]

    if period is not None:
        first = stack.terms(period[0])
        cell = stack.section(Repeat(tuple(period), 1), first, first[0])
        rho = _bloch_rho(pol, cell, first[0])
        flux = _bloch_flux(pol, first[0], rho)
        exit_medium = first[0]
        s = stack.compose(front, incident, exit_medium)
        s = _star(s, _S(rho, np.ones_like(rho), np.zeros_like(rho), np.zeros_like(rho)))
    else:
        s = stack.compose(front, incident, exit_medium)

    r, t = s.r, s.t
    shape = np.broadcast_shapes(np.shape(r), np.shape(t), n_in.shape, th0.shape, lam.shape, exit_shape)
    R = np.abs(r) ** 2
    if period is not None:
        norm = (n_in * c_in).real if pol == "s" else (n_in * np.conj(c_in)).real
        T = np.maximum(np.abs(t) ** 2 * flux / norm, 0.0)
    elif exit_medium is _PEC:
        T = np.zeros(shape)
    elif pol == "s":
        T = np.abs(t ** 2) * ((exit_medium.n * exit_medium.c).real / (n_in * c_in).real)
    else:
        T = np.abs(t ** 2) * ((exit_medium.n * np.conj(exit_medium.c)).real / (n_in * np.conj(c_in)).real)
    return {key: np.broadcast_to(value, shape).copy() for key, value in (("r", r), ("t", t), ("R", R), ("T", T))}
