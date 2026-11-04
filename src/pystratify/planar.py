"""Planar Maxwell optics and mixed-coherence compatibility functions.

MIT-derived planar utilities retain their attribution in PLANAR-LICENSE.txt.
Coherent responses use the same projective core as cylindrical/spherical modes.
"""
from __future__ import division, print_function, absolute_import

from .response import sweep
from .stable import stable_tmm as _stable_tmm

import itertools

from numpy import cos, inf, zeros, array, exp, conj, nan, isnan, pi, sin, seterr
from numpy.lib.scimath import arcsin

import numpy as np

JAX_AVAILABLE = False

import sys
EPSILON = sys.float_info.epsilon # typical floating-point calculation error

def make_2x2_array(a, b, c, d, dtype=float):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]

    Same as "numpy.array([[a,b],[c,d]], dtype=float)", but ten times faster
    """
    my_array = np.empty((2,2), dtype=dtype)
    my_array[0,0] = a
    my_array[0,1] = b
    my_array[1,0] = c
    my_array[1,1] = d
    return my_array

def is_forward_angle(n, theta, y=None):
    """
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.

    ``y`` is the admittance index n/mu_r (see ``admittance_index``). It defaults
    to n, which is the non-magnetic case and leaves every branch below exactly
    as it was. The two are different quantities and each test below needs the
    right one:

    * the DECAY test is about the wave vector, kz proportional to n cos(theta),
      so it keeps n. A wave decaying along +z is the forward one whatever the
      permeability does.
    * the POYNTING test is about power, proportional to Re(y cos(theta)) for s
      (and Re(y cos(theta*)) for p, which agrees). In a double-negative medium
      n and mu_r are negative together, so y stays positive and the forward wave
      is the one whose phase runs BACKWARD -- negative refraction. Testing
      Re(n cos(theta)) there would pick the gain-like branch and quietly return
      the positive-index twin.
    """
    if y is None:
        y = n
    if y is n:
        assert n.real * n.imag >= 0, ("For materials with gain, it's ambiguous which "
                                      "beam is incoming vs outgoing. See "
                                      "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                      "n: " + str(n) + "   angle: " + str(theta))
    else:
        # Passivity for a magnetic medium is Im(n) >= 0 together with
        # Re(n/mu_r) >= 0; the non-magnetic product test would reject every
        # passive negative-index material, whose n has a negative real part.
        assert n.imag >= -100 * EPSILON and y.real >= -100 * EPSILON, (
            "For materials with gain, it's ambiguous which beam is incoming vs "
            "outgoing. A passive medium has Im(n) >= 0 and Re(n/mu_r) >= 0.\n"
            "n: " + str(n) + "   n/mu_r: " + str(y) + "   angle: " + str(theta))
    ncostheta = n * cos(theta)
    ycostheta = ncostheta if y is n else y * cos(theta)
    if abs(ncostheta.imag) > 100 * EPSILON:
        # Either evanescent decay or lossy medium. Either way, the one that
        # decays is the forward-moving wave
        answer = (ncostheta.imag > 0)
    else:
        # Forward is the one with positive Poynting vector
        # Poynting vector is Re[y cos(theta)] for s-polarization or
        # Re[y cos(theta*)] for p-polarization, but it turns out they're consistent
        # so I'll just assume s then check both below
        answer = (ycostheta.real > 0)
    # convert from numpy boolean to the normal Python boolean
    answer = bool(answer)
    # double-check the answer ... can't be too careful!
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
                    " index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))
    if answer is True:
        assert ncostheta.imag > -100 * EPSILON, error_string
        assert ycostheta.real > -100 * EPSILON, error_string
        assert (y * cos(theta.conjugate())).real > -100 * EPSILON, error_string
    else:
        assert ncostheta.imag < 100 * EPSILON, error_string
        assert ycostheta.real < 100 * EPSILON, error_string
        assert (y * cos(theta.conjugate())).real < 100 * EPSILON, error_string
    return answer


def admittance_index(n_list, mu_list=None):
    """
    Return the admittance index y = n/mu_r used by every interface and power
    formula in this module.

    Where a formula in the non-magnetic package writes ``n``, the magnetic
    generalization writes ``n/mu_r`` and leaves ``cos(theta)`` alone -- both
    Fresnel coefficients reduce to that single substitution, and so do T_from_t,
    power_entering_from_r and the Poynting/absorption densities. What keeps
    using the true (signed) ``n`` is Snell's law and kz, because those describe
    the wave vector rather than the power.

    ``mu_list=None`` returns the SAME ARRAY OBJECT, not a copy or a division by
    1.0, so every non-magnetic caller runs the identical code path and produces
    bit-identical output.
    """
    if mu_list is None:
        return n_list
    mu = array(mu_list)
    if mu.shape != np.shape(n_list):
        raise ValueError("mu_list must have the same shape as n_list")
    if np.all(mu == 1):
        return n_list
    return array(n_list) / mu

def is_evanescent_angle(n, theta, rel_tol=1e-9):
    """
    True when a wave at (complex) angle theta in a medium of index n is
    evanescent, i.e. carries essentially no propagating power along the surface
    normal because the medium is beyond the critical angle for the incident
    tangential wavevector.

    The along-normal power flux is proportional to Re(n cos(theta)). For a
    lossless medium beyond the critical angle this is exactly zero (n cos(theta)
    is purely imaginary); a small loss makes it tiny but nonzero, so the test is
    relative to |n cos(theta)|. Genuinely absorbing-but-propagating layers keep
    a non-negligible real part and are *not* flagged.

    This is the physical signature of total internal reflection into an
    incoherent layer. When it is present the incoherent (intensity) transfer
    matrix in inc_tmm() divides by a vanishing interface transmission and breaks
    down, so the dispatch routes such stacks to phase_average_tmm instead.
    """
    n_cos = n * cos(theta)
    magnitude = abs(n_cos)
    if magnitude == 0:
        return True
    return abs(n_cos.real) <= rel_tol * magnitude

def snell(n_1, n_2, th_1, y_2=None):
    """
    return angle theta in layer 2 with refractive index n_2, assuming
    it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
    that "angles" may be complex!!

    y_2 is medium 2's admittance index n/mu_r; it selects the forward branch and
    defaults to n_2, the non-magnetic case.
    """
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    th_2_guess = arcsin(n_1*np.sin(th_1) / n_2)
    if is_forward_angle(n_2, th_2_guess, y_2):
        return th_2_guess
    else:
        return pi - th_2_guess

def list_snell(n_list, th_0, y_list=None):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    if y_list is None:
        y_list = n_list
    angles = arcsin(n_list[0]*np.sin(th_0) / n_list)
    # The first and last entry need to be the forward angle (the intermediate
    # layers don't matter, see https://arxiv.org/abs/1603.02720 Section 5)
    if not is_forward_angle(n_list[0], angles[0], y_list[0]):
        angles[0] = pi - angles[0]
    if not is_forward_angle(n_list[-1], angles[-1], y_list[-1]):
        angles[-1] = pi - angles[-1]
    return angles

def _interface_coefficients(pol, y_i, y_f, th_i, th_f, xp=np):
    """Fresnel amplitudes from the ADMITTANCE indices y = n/mu_r.

    The formulas are the package's own, with n replaced by n/mu_r; at mu_r = 1
    the caller passes n_list itself and nothing changes. That single
    substitution is the whole magnetic generalization for both polarizations:
    matching the tangential fields gives r_s = (y_i c_i - y_f c_f)/(y_i c_i +
    y_f c_f) and r_p = (y_f c_i - y_i c_f)/(y_f c_i + y_i c_f).
    """
    cos_i = xp.cos(th_i)
    cos_f = xp.cos(th_f)
    if pol == 's':
        denom = y_i * cos_i + y_f * cos_f
        return (y_i * cos_i - y_f * cos_f) / denom, 2 * y_i * cos_i / denom
    if pol == 'p':
        denom = y_f * cos_i + y_i * cos_f
        return (y_f * cos_i - y_i * cos_f) / denom, 2 * y_i * cos_i / denom
    raise ValueError("Polarization must be 's' or 'p'")

def _warn_opaque_layers():
    global opacity_warning
    if 'opacity_warning' not in globals():
        opacity_warning = True
        print("Warning: Layers that are almost perfectly opaque "
              "are modified to be slightly transmissive, "
              "allowing 1 photon in 10^30 to pass through. It's "
              "for numerical stability. This warning will not "
              "be shown again.")

def _opaque_floor(transmission):
    """The two-flux recursions divide by a coherent segment's power transmission.

    The owned amplitude kernel evaluates an opaque segment exactly, so a thick
    metal underflows to T = 0 and the incoherent L-matrix divides by zero
    (R = T = nan, which the API reported as a critical-angle failure). The
    retained upstream clip of Im(delta) at 35 never let T below about e^-70;
    the same 1-photon-in-10^30 floor the incoherent propagation already uses
    keeps the recursion finite without changing any finite result.
    """
    if transmission < 1e-30:
        _warn_opaque_layers()
        return 1e-30
    return transmission

#: An interior layer whose |cos(theta)| is below this is at (or within ~1e-12 rad of) its own critical angle.
#: There the reflection recursion meets an interface that reflects perfectly while the layer's phase vanishes,
#: and loses the 0 * inf limit between them: an air gap between two glass prisms at its critical angle came back
#: as R_s = 1 where the characteristic matrix gives 0.3304 (assurance WP07, O-W7-10; Byrnes's `tmm` does the
#: same). Its error grows as eps / |cos(theta)|, so below this bound (2e-10) r, t, R and T come from
#: `stable.stable_tmm`, whose layer takes the limit, and every other point is computed exactly as before.
CRITICAL_COS = 1e-6


def _critical_layers(th_list):
    """Indices (into the stack, half-spaces included) of the interior layers at their own critical angle."""
    return [int(i) + 1 for i in np.flatnonzero(np.abs(cos(np.asarray(th_list)[1:-1])) < CRITICAL_COS)]


def _critical_amplitudes(pol, n_list, d_list, th_0, lam_vac, mu_list, critical):
    """r and t of a stack with a layer at its own critical angle, from `stable.stable_tmm`."""
    if mu_list is not None and np.any(np.asarray(mu_list) != 1):
        raise ValueError(f"layer {critical[0]} is at its own critical angle and the stack is magnetic; the "
                         "critical-angle path supports mu = 1 only")
    stack = [(n_list[i], d_list[i]) for i in range(1, len(n_list) - 1)]
    out = _stable_tmm(pol, stack, n_list[0], n_list[-1], th_0, lam_vac, detect_repeats=False)
    return out['r'], out['t']


def _refuse_critical(coh_tmm_data, what):
    """Per-layer outputs are undefined where a layer sits at its own critical angle."""
    critical = coh_tmm_data.get('critical_layers') or []
    if critical:
        raise ValueError(
            f"layer {critical[0]} is at its own critical angle (cos theta = 0 inside it): its forward and "
            f"backward waves coincide, so {what} is not defined there. R, T, r and t are exact; move the "
            "angle off the critical one for per-layer quantities")


def _prepare_coherent_stack(n_list, d_list, th_0, mu_list=None):
    n_list = array(n_list)
    d_list = array(d_list, dtype=float)
    y_list = admittance_index(n_list, mu_list)

    if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
        raise ValueError("Problem with n_list or d_list!")
    assert d_list[0] == d_list[-1] == inf, 'd_list must start and end with inf!'
    assert abs((n_list[0]*np.sin(th_0)).imag) < 100*EPSILON, 'Error in n0 or th0!'
    assert is_forward_angle(n_list[0], th_0, y_list[0]), 'Error in n0 or th0!'
    return n_list, d_list, list_snell(n_list, th_0, y_list), y_list

def _prepare_coherent_inputs(n_list, d_list, th_0, lam_vac, mu_list=None):
    if ((hasattr(lam_vac, 'size') and lam_vac.size > 1)
          or (hasattr(th_0, 'size') and th_0.size > 1)):
        raise ValueError('This function is not vectorized; you need to run one '
                         'calculation at a time (1 wavelength, 1 angle, etc.)')

    n_list, d_list, th_list, y_list = _prepare_coherent_stack(
        n_list, d_list, th_0, mu_list
    )
    # kz is a WAVE VECTOR, so it keeps the true signed n: in a negative-index
    # layer the phase runs backward, which is exactly what makes the Lequime
    # white Fabry-Perot achromatic.
    kz_list = 2 * np.pi * n_list * cos(th_list) / lam_vac

    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)

    return n_list, d_list, th_list, kz_list, delta, y_list

def _coh_tmm_amplitudes_numpy(pol, y_list, th_list, delta):
    r, t = _interface_coefficients(pol, y_list[:-1], y_list[1:], th_list[:-1], th_list[1:])
    response = sweep((-r, np.ones_like(r), np.ones_like(r), -r), np.exp(2j * delta[1:-1]))
    vw = np.zeros((len(y_list), 2), complex)
    amplitude = 1 + 0j
    for j in range(len(r)):
        if j:
            amplitude *= np.exp(1j * delta[j])
        amplitude *= t[j] / (1 + r[j] * response.outgoing_out[j])
        vw[j + 1] = amplitude, amplitude * response.outgoing_out[j]
    return response.outgoing_in[0], amplitude, vw

def _coh_tmm_amplitudes(pol, y_list, th_list, delta):
    return _coh_tmm_amplitudes_numpy(pol, y_list, th_list, delta)

def _coh_tmm_amplitudes_batch(pol, y_list, th_list, delta_batch, include_vw_list=True):
    r, t = _interface_coefficients(pol, y_list[:-1], y_list[1:], th_list[:-1], th_list[1:])
    count = len(delta_batch)
    rb = np.broadcast_to(r[:, None], (len(r), count))
    response = sweep((-rb, np.ones_like(rb), np.ones_like(rb), -rb), np.exp(2j * delta_batch[:, 1:-1].T))
    vw = np.zeros((count, len(y_list), 2), complex) if include_vw_list else None
    amplitude = np.ones(count, complex)
    for j in range(len(r)):
        if j:
            amplitude *= np.exp(1j * delta_batch[:, j])
        amplitude *= t[j] / (1 + r[j] * response.outgoing_out[j])
        if vw is not None:
            vw[:, j + 1, 0] = amplitude
            vw[:, j + 1, 1] = amplitude * response.outgoing_out[j]
    return response.outgoing_in[0], amplitude, vw

def _stabilize_delta_batch(delta_batch):
    return delta_batch

def _coh_tmm_absorp_in_each_layer_batch(pol, y_list, th_list, th_0, power_entering, T, vw_list):
    num_points, num_layers = vw_list.shape[:2]
    power_entering_each_layer = np.zeros((num_points, num_layers), dtype=float)
    power_entering_each_layer[:, 0] = 1.0
    power_entering_each_layer[:, 1] = power_entering
    power_entering_each_layer[:, -1] = T

    if num_layers > 3:
        v = vw_list[:, 2:-1, 0]
        w = vw_list[:, 2:-1, 1]
        y = y_list[None, 2:-1]
        th = th_list[None, 2:-1]
        if pol == 's':
            boundary_power = (
                (y * np.cos(th) * np.conj(v + w) * (v - w)).real
                / (y_list[0] * np.cos(th_0)).real
            )
        else:
            boundary_power = (
                (y * np.conj(np.cos(th)) * (v + w) * np.conj(v - w)).real
                / (y_list[0] * np.conj(np.cos(th_0))).real
            )
        power_entering_each_layer[:, 2:-1] = boundary_power

    final_answer = np.zeros_like(power_entering_each_layer)
    final_answer[:, :-1] = -np.diff(power_entering_each_layer, axis=1)
    final_answer[:, -1] = power_entering_each_layer[:, -1]
    return final_answer

def coh_tmm_wavelength_sweep(
    pol,
    n_list,
    d_list,
    th_0,
    lam_vac_list,
    include_layer_absorption=True,
    mu_list=None,
):
    lam_vac_array = array(lam_vac_list, dtype=float)
    if lam_vac_array.ndim != 1:
        raise ValueError("lam_vac_list must be one-dimensional")

    n_list, d_list, th_list, y_list = _prepare_coherent_stack(
        n_list, d_list, th_0, mu_list
    )
    delta_template = np.zeros(n_list.shape, dtype=complex)
    delta_template[1:-1] = 2 * np.pi * n_list[1:-1] * cos(th_list[1:-1]) * d_list[1:-1]
    delta_batch = _stabilize_delta_batch(delta_template[None, :] / lam_vac_array[:, None])
    r, t, vw_list = _coh_tmm_amplitudes_batch(
        pol,
        y_list,
        th_list,
        delta_batch,
        include_vw_list=include_layer_absorption,
    )
    critical = _critical_layers(th_list)
    if critical:
        r, t = _critical_amplitudes(pol, n_list, d_list, th_0, lam_vac_array, mu_list, critical)
        if vw_list is not None:
            vw_list = np.full_like(vw_list, nan)
    R = np.array(R_from_r(r), dtype=float)
    T = np.array(T_from_t(pol, t, y_list[0], y_list[-1], th_0, th_list[-1]), dtype=float)
    power_entering = np.array(power_entering_from_r(pol, r, y_list[0], th_0), dtype=float)
    result = {
        'R': R,
        'T': T,
        'power_entering': power_entering,
        'critical_layers': critical,
    }
    if include_layer_absorption:
        result['layer_absorption'] = _coh_tmm_absorp_in_each_layer_batch(
            pol,
            y_list,
            th_list,
            th_0,
            power_entering,
            T,
            vw_list,
        )
        if critical:
            # Undefined in every interior layer, as `absorp_in_each_layer` refuses: one layer's share can come
            # from R and T by energy balance, but not a stack's split between its layers.
            result['layer_absorption'][:, 1:-1] = nan
    return result

def interface_r(polarization, n_i, n_f, th_i, th_f):
    """
    reflection amplitude (from Fresnel equations)

    polarization is either "s" or "p" for polarization

    n_i, n_f are (complex) refractive index for incident and final

    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    return _interface_coefficients(polarization, n_i, n_f, th_i, th_f)[0]

def interface_t(polarization, n_i, n_f, th_i, th_f):
    """
    transmission amplitude (frem Fresnel equations)

    polarization is either "s" or "p" for polarization

    n_i, n_f are (complex) refractive index for incident and final

    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    return _interface_coefficients(polarization, n_i, n_f, th_i, th_f)[1]

def R_from_r(r):
    """
    Calculate reflected power R, starting with reflection amplitude r.
    """
    return abs(r)**2

def T_from_t(pol, t, n_i, n_f, th_i, th_f):
    """
    Calculate transmitted power T, starting with transmission amplitude t.

    n_i,n_f are refractive indices of incident and final medium.

    th_i, th_f are (complex) propegation angles through incident & final medium
    (in radians, where 0=normal). "th" stands for "theta".

    In the case that n_i,n_f,th_i,th_f are real, formulas simplify to
    T=|t|^2 * (n_f cos(th_f)) / (n_i cos(th_i)).

    See https://arxiv.org/abs/1603.02720 for discussion of formulas
    """
    if pol == 's':
        return abs(t**2) * (((n_f*cos(th_f)).real) / (n_i*cos(th_i)).real)
    elif pol == 'p':
        return abs(t**2) * (((n_f*conj(cos(th_f))).real) /
                                (n_i*conj(cos(th_i))).real)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def power_entering_from_r(pol, r, n_i, th_i):
    """
    Calculate the power entering the first interface of the stack, starting with
    reflection amplitude r. Normally this equals 1-R, but in the unusual case
    that n_i is not real, it can be a bit different than 1-R. See
    https://arxiv.org/abs/1603.02720

    n_i is refractive index of incident medium.

    th_i is (complex) propegation angle through incident medium
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if pol == 's':
        return ((n_i*cos(th_i)*(1+conj(r))*(1-r)).real
                     / (n_i*cos(th_i)).real)
    elif pol == 'p':
        return ((n_i*conj(cos(th_i))*(1+r)*(1-conj(r))).real
                      / (n_i*conj(cos(th_i))).real)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def interface_R(polarization, n_i, n_f, th_i, th_f):
    """
    Fraction of light intensity reflected at an interface.
    """
    r = interface_r(polarization, n_i, n_f, th_i, th_f)
    return R_from_r(r)

def interface_T(polarization, n_i, n_f, th_i, th_f):
    """
    Fraction of light intensity transmitted at an interface.
    """
    t = interface_t(polarization, n_i, n_f, th_i, th_f)
    return T_from_t(polarization, t, n_i, n_f, th_i, th_f)

def coh_tmm(pol, n_list, d_list, th_0, lam_vac, mu_list=None):
    """
    Main "coherent transfer matrix method" calc. Given parameters of a stack,
    calculates everything you could ever want to know about how light
    propagates in it. (If performance is an issue, you can delete some of the
    calculations without affecting the rest.)

    pol is light polarization, "s" or "p".

    n_list is the list of refractive indices, in the order that the light would
    pass through them. The 0'th element of the list should be the semi-infinite
    medium from which the light enters, the last element should be the semi-
    infinite medium to which the light exits (if any exits).

    th_0 is the angle of incidence: 0 for normal, pi/2 for glancing.
    Remember, for a dissipative incoming medium (n_list[0] is not real), th_0
    should be complex so that n0 sin(th0) is real (intensity is constant as
    a function of lateral position).

    d_list is the list of layer thicknesses (front to back). Should correspond
    one-to-one with elements of n_list. First and last elements should be "inf".

    lam_vac is vacuum wavelength of the light.
    
    This function, like everything else in the package, implicitly requires you
    to pick a unit of length.You can use any unit, but keep it consistent.
    For example, if you input the wavelength in nanometers, then you must also
    input the layer thicknesses in nanometers. And then any angular wavenumber
    outputs will be in radians per nanometer, and any absorption outputs will
    be in (fraction of incoming light power per nanometer of depth), and so on.

    Then the function outputs the following as a dictionary (see
    https://arxiv.org/abs/1603.02720 for details)

    * r--reflection amplitude
    * t--transmission amplitude
    * R--reflected wave power (as fraction of incident)
    * T--transmitted wave power (as fraction of incident)
    * power_entering--Power entering the first layer, usually (but not always)
      equal to 1-R (see https://arxiv.org/abs/1603.02720 ).
    * vw_list-- n'th element is [v_n,w_n], the forward- and backward-traveling
      amplitudes, respectively, in the n'th medium just after interface with
      (n-1)st medium.
    * kz_list--normal component of complex angular wavenumber for
      forward-traveling wave in each layer.
    * th_list--(complex) propagation angle (in radians) in each layer
    * pol, n_list, d_list, th_0, lam_vac--same as input

    """
    n_list, d_list, th_list, kz_list, delta, y_list = _prepare_coherent_inputs(
        n_list, d_list, th_0, lam_vac, mu_list
    )
    r, t, vw_list = _coh_tmm_amplitudes(pol, y_list, th_list, delta)
    critical = _critical_layers(th_list)
    if critical:
        r, t = (complex(np.squeeze(amplitude)) for amplitude in
                _critical_amplitudes(pol, n_list, d_list, th_0, lam_vac, mu_list, critical))
        vw_list = np.full_like(vw_list, nan)

    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r(r)
    T = T_from_t(pol, t, y_list[0], y_list[-1], th_0, th_list[-1])
    power_entering = power_entering_from_r(pol, r, y_list[0], th_0)

    reflection, _ = _interface_coefficients(pol, y_list[:-1], y_list[1:], th_list[:-1], th_list[1:])
    response = sweep((-reflection, np.ones_like(reflection), np.ones_like(reflection), -reflection), np.exp(2j * delta[1:-1]))
    backward_bottom = np.zeros(len(n_list), complex)
    backward_bottom[1:-1] = vw_list[1:-1, 0] * np.exp(1j * delta[1:-1]) * response.outgoing_in[1:]
    return {'r': r, 't': t, 'R': R, 'T': T, 'power_entering': power_entering,
            'backward_bottom': backward_bottom, 'vw_list': vw_list, 'kz_list': kz_list, 'th_list': th_list,
            'pol': pol, 'n_list': n_list, 'd_list': d_list, 'th_0': th_0,
            'lam_vac':lam_vac, 'y_list': y_list, 'mu_list': mu_list,
            'critical_layers': critical}

def coh_tmm_reverse(pol, n_list, d_list, th_0, lam_vac):
    """
    Reverses the order of the stack then runs coh_tmm.
    """
    th_f = snell(n_list[0], n_list[-1], th_0)
    return coh_tmm(pol, n_list[::-1], d_list[::-1], th_f, lam_vac)

#: Below this many machine epsilons (times max(|r_s|, |r_p|, 1)) a reflected amplitude is rounding, not a wave:
#: its phase is noise, so the angles that need it are undefined. multilayer.app's API rule (WP07-D2).
AMPLITUDE_FLOOR_EPS = 64


def psi_delta(r_s, r_p):
    """
    Ellipsometric angles (psi, Delta), in radians, of the reflection amplitudes r_s and r_p; vectorised.

    psi = atan2(|r_p|, |r_s|) and Delta = arg(-r_p conj(r_s)), the phase of -r_p/r_s: this package's p basis
    flips sign at normal incidence (r_p = -r_s there), so the minus sign gives Delta = 0 at normal incidence and
    for a bare dielectric below Brewster. Delta is in numpy.angle's range, (-pi, pi]; an instrument's Delta (the
    Nebraska convention of `pystratify.ellipsometry`) is pi minus this one.

    An amplitude at or below 64 eps max(|r_s|, |r_p|, 1) is rounding, not a reflected wave. psi needs one of the
    two above that floor and Delta needs both; an undefined angle is NaN. At a bare dielectric's Brewster angle
    psi is 0 and Delta is NaN (not the phase of a rounding-level r_p).

    Returns ``(psi, Delta)``, numpy scalars for scalar input and arrays of the broadcast shape otherwise.
    """
    r_s = np.asarray(r_s, dtype=complex)
    r_p = np.asarray(r_p, dtype=complex)
    s_magnitude = np.abs(r_s)
    p_magnitude = np.abs(r_p)
    floor = AMPLITUDE_FLOOR_EPS * EPSILON * np.maximum(np.maximum(s_magnitude, p_magnitude), 1.0)
    s_defined = s_magnitude > floor
    p_defined = p_magnitude > floor
    psi = np.where(s_defined | p_defined, np.arctan2(p_magnitude, s_magnitude), nan)
    delta = np.where(s_defined & p_defined, np.angle(-r_p * np.conj(r_s)), nan)
    return psi[()], delta[()]


def ellips(n_list, d_list, th_0, lam_vac, mu_list=None):
    """
    Ellipsometric parameters of a coherent stack, in radians: ``{'psi': psi, 'Delta': Delta}`` from
    `psi_delta` of the s and p reflection amplitudes (Delta = arg(-r_p/r_s), in (-pi, pi]; NaN where an
    amplitude vanishes, e.g. at a Brewster angle). ``mu_list`` is passed to `coh_tmm`.

    Warning: conventions differ. An ellipsometer's Delta (the Nebraska convention) is pi minus this one;
    `pystratify.ellipsometry` reports that one.
    """
    s_data = coh_tmm('s', n_list, d_list, th_0, lam_vac, mu_list=mu_list)
    p_data = coh_tmm('p', n_list, d_list, th_0, lam_vac, mu_list=mu_list)
    psi, Delta = psi_delta(s_data['r'], p_data['r'])
    return {'psi': psi, 'Delta': Delta}

def unpolarized_RT(n_list, d_list, th_0, lam_vac):
    """
    Calculates reflected and transmitted power for unpolarized light.
    """

    s_data = coh_tmm('s', n_list, d_list, th_0, lam_vac)
    p_data = coh_tmm('p', n_list, d_list, th_0, lam_vac)
    R = (s_data['R'] + p_data['R']) / 2.
    T = (s_data['T'] + p_data['T']) / 2.
    return {'R': R, 'T': T}

def _layer_waves(layer, distance, coh_tmm_data):
    """Forward and backward amplitudes (Ef, Eb) at depth ``distance`` of ``layer`` of a `coh_tmm` solution.

    The backward wave of an interior layer is propagated from the layer's far face (``backward_bottom``), so a
    thick absorbing layer never multiplies an underflowed amplitude by an overflowing exponential. Works on an
    array of depths.
    """
    if layer > 0:
        v, w = coh_tmm_data['vw_list'][layer]
    else:
        v = 1
        w = coh_tmm_data['r']
    kz = coh_tmm_data['kz_list'][layer]
    Ef = v * exp(1j * kz * distance)
    if 0 < layer < len(coh_tmm_data["d_list"]) - 1 and "backward_bottom" in coh_tmm_data:
        Eb = coh_tmm_data["backward_bottom"][layer] * exp(1j * kz * (coh_tmm_data["d_list"][layer] - distance))
    else:
        Eb = w * exp(-1j * kz * distance)
    return Ef, Eb


def _magnetic_field(pol, th, y, y_0, Ef, Eb):
    """(Hx, Hy, Hz) of the waves Ef, Eb in a layer of admittance index y at angle th, in units of |y_0| |E0|.

    H = y k_hat x E / Z0 per wave: s (E along y) gives Hx = -y cos(th) (Ef - Eb), Hz = y sin(th) (Ef + Eb);
    p gives Hy = y (Ef + Eb). Dividing by |y_0| makes the incident wave's |H| = 1 when its |E| = 1.
    """
    scale = y / abs(y_0)
    if pol == 's':
        return -scale * cos(th) * (Ef - Eb), 0j, scale * sin(th) * (Ef + Eb)
    return 0j, scale * (Ef + Eb), 0j


def position_resolved(layer, distance, coh_tmm_data):
    """
    Starting with output of coh_tmm(), calculate the Poynting vector,
    absorbed energy density, and E- and H-fields at a specific location. The
    location is defined by (layer, distance), defined the same way as in
    find_in_structure_with_inf(...).

    Returns a dictionary containing:

    * poyn - the component of Poynting vector normal to the interfaces
    * absor - the absorbed energy density at that point
    * Ex and Ey and Ez - the electric field amplitudes, where
      z is normal to the interfaces and the light rays are in the x,z plane.
    * Hx and Hy and Hz - the magnetic field amplitudes (complex), in units of
      the incident wave's |H0| = |y_0| |E0|, with y = n/mu_r the admittance
      index (`admittance_index`): s gives Hx = -(y/|y_0|) cos(th) (Ef - Eb) and
      Hz = (y/|y_0|) sin(th) (Ef + Eb), p gives Hy = (y/|y_0|) (Ef + Eb), from
      the same forward/backward amplitudes Ef, Eb as E. Re(E x H*)_z |y_0| /
      Re(y_0 cos th_0) is ``poyn`` (s; cos th_0* for p).

    The E-field is in units where the incoming |E|=1; see
    https://arxiv.org/pdf/1603.02720.pdf for formulas.
    """
    _refuse_critical(coh_tmm_data, "the field")
    th = coh_tmm_data['th_list'][layer]
    kz = coh_tmm_data['kz_list'][layer]
    # Power densities carry the admittance index n/mu_r, not n; they are the
    # same array when the stack is non-magnetic. kz keeps the true n.
    y_all = coh_tmm_data.get('y_list')
    if y_all is None:
        y_all = coh_tmm_data['n_list']
    y = y_all[layer]
    y_0 = y_all[0]
    th_0 = coh_tmm_data['th_0']
    pol = coh_tmm_data['pol']

    assert ((layer >= 1 and 0 <= distance <= coh_tmm_data['d_list'][layer])
                or (layer == 0 and distance <= 0))

    # Amplitude of forward-moving wave is Ef, backwards is Eb
    Ef, Eb = _layer_waves(layer, distance, coh_tmm_data)

    # Poynting vector
    if pol == 's':
        poyn = ((y*cos(th)*conj(Ef+Eb)*(Ef-Eb)).real) / (y_0*cos(th_0)).real
    elif pol == 'p':
        poyn = (((y*conj(cos(th))*(Ef+Eb)*conj(Ef-Eb)).real)
                    / (y_0*conj(cos(th_0))).real)

    # Absorbed energy density
    if pol == 's':
        absor = (y*cos(th)*kz*abs(Ef+Eb)**2).imag / (y_0*cos(th_0)).real
    elif pol == 'p':
        absor = (y*conj(cos(th))*
                 (kz*abs(Ef-Eb)**2-conj(kz)*abs(Ef+Eb)**2)
                ).imag / (y_0*conj(cos(th_0))).real

    # Electric field
    if pol == 's':
        Ex = 0
        Ey = Ef + Eb
        Ez = 0
    elif pol == 'p':
        Ex = (Ef - Eb) * cos(th)
        Ey = 0
        Ez = (-Ef - Eb) * sin(th)

    Hx, Hy, Hz = _magnetic_field(pol, th, y, y_0, Ef, Eb)

    return {'poyn': poyn, 'absor': absor, 'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz}


def _field_intensities(pol, th, y, y_0, Ef, Eb):
    """(|E|^2, |H|^2) of the waves Ef, Eb, with position_resolved's E and `_magnetic_field`'s H."""
    if pol == 's':
        e2 = abs(Ef + Eb) ** 2
    else:
        e2 = abs((Ef - Eb) * cos(th)) ** 2 + abs((-Ef - Eb) * sin(th)) ** 2
    return e2, sum(abs(component) ** 2 for component in _magnetic_field(pol, th, y, y_0, Ef, Eb))

def find_in_structure(d_list, distance):
    """
    d_list is list of thicknesses of layers, all of which are finite.

    distance is the distance from the front of the whole multilayer structure
    (i.e., from the start of layer 0.)

    Function returns [layer,z], where:

    * layer is what number layer you're at.
    * z is the distance into that layer.

    For large distance, layer = len(d_list), even though d_list[layer] doesn't
    exist in this case. For negative distance, return [-1, distance]
    """
    if sum(d_list) == inf:
        raise ValueError('This function expects finite arguments')
    if distance < 0:
        return [-1, distance]
    layer = 0
    while (layer < len(d_list)) and (distance >= d_list[layer]):
        distance -= d_list[layer]
        layer += 1
    return [layer, distance]

def find_in_structure_with_inf(d_list, distance):
    """
    d_list is list of thicknesses of layers [inf, blah, blah, ..., blah, inf]

    distance is the distance from the front of the whole multilayer structure
    (i.e., from the start of layer 1.)

    Function returns [layer,z], where:

    * layer is what number layer you're at,
    * z is the distance into that layer.

    For distance < 0, returns [0, distance]. So the first interface can be described as
    either [0,0] or [1,0].
    """
    if distance < 0:
        return [0, distance]
    [layer, distance_in_layer] = find_in_structure(d_list[1:-1], distance)
    return [layer+1, distance_in_layer]

def layer_starts(d_list):
    """
    Gives the location of the start of any given layer, relative to the front
    of the whole multilayer structure. (i.e. the start of layer 1)

    d_list is list of thicknesses of layers [inf, blah, blah, ..., blah, inf]

    """
    final_answer = zeros(len(d_list))
    final_answer[0] = -inf
    final_answer[1] = 0
    for i in range(2, len(d_list)):
        final_answer[i] = final_answer[i-1] + d_list[i-1]
    return final_answer

class absorp_analytic_fn:
    """
    Absorption in a given layer is a pretty simple analytical function:
    The sum of four exponentials.

    a(z) = A1*exp(a1*z) + A2*exp(-a1*z)
           + A3*exp(1j*a3*z) + conj(A3)*exp(-1j*a3*z)

    where a(z) is absorption at depth z, with z=0 being the start of the layer,
    and A1,A2,a1,a3 are real numbers, with a1>0, a3>0, and A3 is complex.
    The class stores these five parameters, as well as d, the layer thickness.

    This gives absorption as a fraction of intensity coming towards the first
    layer of the stack.
    """
    def fill_in(self, coh_tmm_data, layer):
        """
        fill in the absorption analytic function starting from coh_tmm_data
        (the output of coh_tmm), for absorption in the layer with index
        "layer".
        """
        _refuse_critical(coh_tmm_data, "per-layer absorption")
        pol = coh_tmm_data['pol']
        v = coh_tmm_data['vw_list'][layer][0]
        w = coh_tmm_data['vw_list'][layer][1]
        kz = coh_tmm_data['kz_list'][layer]
        y_all = coh_tmm_data.get('y_list')
        if y_all is None:
            y_all = coh_tmm_data['n_list']
        y = y_all[layer]
        y_0 = y_all[0]
        th_0 = coh_tmm_data['th_0']
        th = coh_tmm_data['th_list'][layer]
        self.d = coh_tmm_data['d_list'][layer]

        self.a1 = 2*kz.imag
        self.a3 = 2*kz.real

        if pol == 's':
            temp = (y*cos(th)*kz).imag / (y_0*cos(th_0)).real
            self.A1 = temp * abs(w)**2
            self.A2 = temp * abs(v)**2
            self.A3 = temp * v * conj(w)
        else: # pol=='p'
            temp = (2*(kz.imag)*(y*cos(conj(th))).real /
                    (y_0*conj(cos(th_0))).real)
            self.A1 = temp * abs(w)**2
            self.A2 = temp * abs(v)**2
            self.A3 = v * conj(w) * (-2*(kz.real)*(y*cos(conj(th))).imag /
                                     (y_0*conj(cos(th_0))).real)
        return self

    def copy(self):
        """
        Create copy of an absorp_analytic_fn object
        """
        a = absorp_analytic_fn()
        (a.A1, a.A2, a.A3, a.a1, a.a3, a.d) = (
           self.A1, self.A2, self.A3, self.a1, self.a3, self.d)
        return a

    def run(self, z):
        """
        Calculates absorption at a given depth z, where z=0 is the start of the
        layer.
        """
        return (self.A1*exp(self.a1 * z) + self.A2*exp(-self.a1 * z)
             + self.A3*exp(1j*self.a3*z) + conj(self.A3)*exp(-1j*self.a3*z))

    def flip(self):
        """
        Flip the function front-to-back, to describe a(d-z) instead of a(z),
        where d is layer thickness.
        """
        newA1 = self.A2*exp(-self.a1 * self.d)
        newA2 = self.A1*exp(self.a1 * self.d)
        self.A1, self.A2 = newA1, newA2
        self.A3 = conj(self.A3 * exp(1j * self.a3 * self.d))
        return self

    def scale(self, factor):
        """
        multiplies the absorption at each point by "factor".
        """
        self.A1 *= factor
        self.A2 *= factor
        self.A3 *= factor
        return self

    def add(self, b):
        """
        adds another compatible absorption analytical function
        """
        if (b.a1 != self.a1) or (b.a3 != self.a3):
            raise ValueError('Incompatible absorption analytical functions!')
        self.A1 += b.A1
        self.A2 += b.A2
        self.A3 += b.A3
        return self

def absorp_in_each_layer(coh_tmm_data):
    """
    An array listing what proportion of light is absorbed in each layer.

    Assumes the final layer eventually absorbs all transmitted light.

    Assumes the initial layer eventually absorbs all reflected light.

    Entries of array should sum to 1.

    coh_tmm_data is output of coh_tmm()
    """
    _refuse_critical(coh_tmm_data, "per-layer absorption")
    num_layers = len(coh_tmm_data['d_list'])
    power_entering_each_layer = zeros(num_layers)
    power_entering_each_layer[0] = 1
    power_entering_each_layer[1] = coh_tmm_data['power_entering']
    power_entering_each_layer[-1] = coh_tmm_data['T']
    for i in range(2, num_layers-1):
        power_entering_each_layer[i] = position_resolved(i, 0, coh_tmm_data)['poyn']
    final_answer = zeros(num_layers)
    final_answer[0:-1] = -np.diff(power_entering_each_layer)
    final_answer[-1] = power_entering_each_layer[-1]
    return final_answer

def _forward_angle_list(n_list, th_0):
    """
    Snell-law angles with the forward-propagating branch chosen in every layer.

    list_snell() only guarantees forward branches in the first and last media.
    That is sufficient for coherent TMM amplitudes, but phase integration over
    incoherent layers needs the forward branch everywhere so attenuation never
    turns into unphysical gain in evanescent layers.
    """
    th_list = np.array(list_snell(n_list, th_0), dtype=complex)
    for layer_index, angle in enumerate(th_list):
        if not is_forward_angle(n_list[layer_index], angle):
            th_list[layer_index] = pi - angle
    return th_list

def _power_beta(pol, n, th):
    if pol == 's':
        return n * np.cos(th)
    if pol == 'p':
        return n * np.conj(np.cos(th))
    raise ValueError("Polarization must be 's' or 'p'")

def _interface_transfer_matrix(pol, n_i, n_f, th_i, th_f):
    r = interface_r(pol, n_i, n_f, th_i, th_f)
    t = interface_t(pol, n_i, n_f, th_i, th_f)
    return make_2x2_array(1, r, r, 1, dtype=complex) / t

def _coherent_segment_transfer_data(pol, n_list, d_list, th_list, lam_vac, left_index, right_index):
    local_n = np.array(n_list[left_index:right_index + 1], dtype=complex)
    local_th = np.array(th_list[left_index:right_index + 1], dtype=complex)
    num_internal = right_index - left_index - 1

    interfaces = [
        _interface_transfer_matrix(
            pol,
            local_n[layer_index],
            local_n[layer_index + 1],
            local_th[layer_index],
            local_th[layer_index + 1],
        )
        for layer_index in range(local_n.size - 1)
    ]

    partial_matrices = [None] * (num_internal + 2)
    if num_internal == 0:
        partial_matrices[1] = interfaces[0]
        return {
            'full_matrix': interfaces[0],
            'partial_matrices': partial_matrices,
            'local_n': local_n,
            'local_th': local_th,
            'num_internal': 0,
        }

    local_d = np.array([inf, *d_list[left_index + 1:right_index], inf], dtype=float)
    kz = 2 * np.pi * local_n * np.cos(local_th) / lam_vac
    olderr = seterr(invalid='ignore')
    delta = kz * local_d
    seterr(**olderr)

    for layer_index in range(1, local_n.size - 1):
        if delta[layer_index].imag > 35:
            delta[layer_index] = delta[layer_index].real + 35j
            _warn_opaque_layers()

    internal_matrices = [None] * (num_internal + 1)
    for layer_index in range(1, num_internal + 1):
        phase_matrix = make_2x2_array(
            exp(-1j * delta[layer_index]),
            0,
            0,
            exp(1j * delta[layer_index]),
            dtype=complex,
        )
        internal_matrices[layer_index] = np.dot(phase_matrix, interfaces[layer_index])

    suffix_products = [None] * (num_internal + 2)
    suffix_products[num_internal + 1] = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for layer_index in range(num_internal, 0, -1):
        suffix_products[layer_index] = np.dot(
            internal_matrices[layer_index],
            suffix_products[layer_index + 1],
        )

    for layer_index in range(1, num_internal + 1):
        partial_matrices[layer_index] = np.dot(
            interfaces[layer_index - 1],
            suffix_products[layer_index],
        )
    partial_matrices[num_internal + 1] = interfaces[num_internal]

    return {
        'full_matrix': partial_matrices[1],
        'partial_matrices': partial_matrices,
        'local_n': local_n,
        'local_th': local_th,
        'num_internal': num_internal,
    }

def _phase_integrated_mean_terms(full_matrix, q2):
    a = full_matrix[0, 0]
    b = full_matrix[0, 1]
    denom = abs(a)**2 - abs(b)**2 * q2
    if denom <= 0:
        if abs(denom) < 1e-15:
            denom = 1e-15
        else:
            raise ValueError('Phase integration denominator became non-positive.')

    i0 = 1.0 / denom
    if q2 == 0:
        i1 = 0j
    else:
        i1 = -(a * np.conj(b) * q2) / (abs(a)**2 * denom)
    return i0, i1, np.conj(i1)

def _phase_integrated_boundary_power(pol, beta_prev, beta_inc, partial_matrix, full_matrix, q2):
    aj, bj = partial_matrix[0, 0], partial_matrix[0, 1]
    cj, dj = partial_matrix[1, 0], partial_matrix[1, 1]
    i0, i1, i1_conj = _phase_integrated_mean_terms(full_matrix, q2)

    if pol == 's':
        c0 = np.conj(aj + cj) * (aj - cj)
        cq = np.conj(bj + dj) * (bj - dj)
        c1 = np.conj(aj + cj) * (bj - dj)
        c1_conj = np.conj(bj + dj) * (aj - cj)
    else:
        c0 = (aj + cj) * np.conj(aj - cj)
        cq = (bj + dj) * np.conj(bj - dj)
        c1 = (bj + dj) * np.conj(aj - cj)
        c1_conj = (aj + cj) * np.conj(bj - dj)

    avg_inner = (c0 + cq * q2) * i0 + c1 * i1 + c1_conj * i1_conj
    return float(np.real(beta_prev * avg_inner) / np.real(beta_inc))

def _incoherent_roundtrip_factor(n, thickness, th, lam_vac):
    if np.isinf(thickness):
        return 1.0

    delta = 2 * np.pi * n * np.cos(th) * thickness / lam_vac
    if delta.imag > 35:
        delta = delta.real + 35j
        _warn_opaque_layers()

    if delta.imag < 0 and abs(delta.imag) < 1e-12:
        delta = delta.real + 0j

    if delta.imag < 0:
        raise ValueError('Incoherent layer attenuation became amplifying.')

    return float(np.exp(-4 * delta.imag))

def pim_tmm(pol, n_list, d_list, c_list, th_0, lam_vac):
    """
    Boundary-irradiance PIM for mixed coherent/incoherent stacks.

    Puhan et al. (Coatings 2019, 9, 536) show that the phase-integrated
    reflectance/transmittance recursion is mathematically identical to GTMM,
    while the retained layer-entering irradiances give layer absorptance by
    neighboring boundary differences.  The local inc_tmm() state already
    contains that GTMM recursion and its coherent-stack boundary fluxes, so this
    helper exposes the PIM layer-entering irradiances without reintroducing a
    second, sign-convention-sensitive implementation of Equations (20)-(23).

    When there is no finite incoherent layer, PIM is not applicable and the
    function deliberately returns the ordinary coherent TMM result.
    """
    n_list = array(n_list)
    d_list = array(d_list, dtype=float)

    if (n_list.ndim != 1) or (d_list.ndim != 1):
        raise ValueError("Problem with n_list or d_list!")
    if (d_list[0] != inf) or (d_list[-1] != inf):
        raise ValueError('d_list must start and end with inf!')
    if (c_list[0] != 'i') or (c_list[-1] != 'i'):
        raise ValueError('c_list should start and end with "i"')
    if not n_list.size == d_list.size == len(c_list):
        raise ValueError('List sizes do not match!')
    if (np.real_if_close(n_list[0] * np.sin(th_0))).imag != 0:
        raise ValueError('Error in n0 or th0!')

    has_incoherent_internal_layers = any(
        coherency == 'i' for coherency in c_list[1:-1]
    )
    if not has_incoherent_internal_layers:
        return coh_tmm(pol, n_list, d_list, th_0, lam_vac)

    ans = inc_tmm(pol, n_list, d_list, c_list, th_0, lam_vac)
    layer_absorption = np.array(inc_absorp_in_each_layer(ans), dtype=float)
    power_entering_each_layer = np.empty(n_list.size, dtype=float)
    power_entering_each_layer[0] = 1.0
    for layer_index in range(1, n_list.size):
        power_entering_each_layer[layer_index] = (
            power_entering_each_layer[layer_index - 1]
            - layer_absorption[layer_index - 1]
        )

    ans.update({
        'pim_power_entering_each_layer': power_entering_each_layer,
        'pim_layer_absorption': layer_absorption,
        'pim_method': 'gtmm_equivalent_phase_integrated_boundary_irradiance',
        'pol': pol,
        'n_list': n_list,
        'd_list': d_list,
        'c_list': c_list,
        'th_0': th_0,
        'lam_vac': lam_vac,
    })
    return ans

def _segment_intensity_coeffs(pol, n_list, th_list, segment, left_index, right_index):
    """
    Intensity reflectance/transmittance of one coherent segment, forward and
    backward, derived directly from the segment's coherent transfer matrix.

    The segment matrix M satisfies (v_left, w_left) = M (v_right, w_right) for
    forward/backward field amplitudes, so r = M10/M00 and t = 1/M00 for forward
    incidence, while the reverse solve reads off M01 and det(M). Returning the
    four intensity coefficients from the raw matrix -- rather than a second
    coh_tmm()/coh_tmm_reverse() call -- is what keeps pim_tmm_direct an
    independently derived cross-check of the GTMM path.

    Returns (R_forward, T_forward, R_backward, T_backward, beta_left).
    """
    matrix = segment['full_matrix']
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    beta_left = _power_beta(pol, n_list[left_index], th_list[left_index])
    beta_right = _power_beta(pol, n_list[right_index], th_list[right_index])
    flux_ratio = beta_right.real / beta_left.real
    determinant = a * d - b * c
    r_forward = abs(c / a) ** 2
    t_forward = _opaque_floor(abs(1.0 / a) ** 2 * flux_ratio)
    r_backward = abs(b / a) ** 2
    t_backward = _opaque_floor(abs(determinant / a) ** 2 / flux_ratio)
    return r_forward, t_forward, r_backward, t_backward, beta_left

def pim_tmm_direct(pol, n_list, d_list, c_list, th_0, lam_vac):
    """
    Explicit Phase Integration Method (Puhan et al., Coatings 2019, 9, 536).

    This is the *direct* analytic PIM: an independently derived second
    implementation of the mixed coherent/incoherent solver, kept as a cross-check
    of the production GTMM path (inc_tmm / pim_tmm). Where pim_tmm delegates to
    inc_tmm, this routine assembles the answer from the raw coherent transfer
    matrices of each segment and the closed-form phase integrals of Equations
    (20)-(23), so agreement between the two validates both.

    Each coherent segment between neighboring incoherent layers is described by
    its transfer matrix; the random phase of the following incoherent layer is
    integrated analytically (_phase_integrated_boundary_power / _mean_terms). The
    segments are combined with the same two-flux ladder as inc_tmm, using the
    segment intensity coefficients (_segment_intensity_coeffs) and the incoherent
    single-pass / round-trip attenuation from _incoherent_roundtrip_factor.

    Relative to the abandoned 826ac9f implementation this fixes the three
    structural defects the audit identified:

    * per-segment reflectance is the exact intensity coefficient
      R_eff = R_f + T_f T_b q2 / (1 - R_b q2), evaluated with the segment's own
      bounding-media beta and local incident flux -- not the globally normalized
      1 - power_entering, which is wrong for absorbing incoherent boundaries;
    * the transmitted flux carries the incoherent slab's single-pass factor
      exp(-2 Im delta) as it crosses each incoherent layer, instead of leaking
      attenuation in through the round-trip q2 alone;
    * back-side reflectance R_b comes from the segment's reverse transfer data
      (mirroring coh_tmm_reverse), so the inter-segment multiple-reflection ladder
      never substitutes a front-side reflectance where a back-side one is needed.

    Validity note: like inc_tmm, this is a below-critical-angle
    method. Beyond the critical angle for an incoherent layer the q2 geometric
    sum (denominators |a|^2 - |b|^2 q2 and 1 - R_b q2) is singular; the at-source
    guard below raises rather than returning a fabricated number, and the
    calculate_single_polarization dispatch routes true TIR/evanescent stacks to
    phase_average_tmm before either GTMM or this routine is reached.
    """
    n_list = array(n_list)
    d_list = array(d_list, dtype=float)

    if (n_list.ndim != 1) or (d_list.ndim != 1):
        raise ValueError("Problem with n_list or d_list!")
    if (d_list[0] != inf) or (d_list[-1] != inf):
        raise ValueError('d_list must start and end with inf!')
    if (c_list[0] != 'i') or (c_list[-1] != 'i'):
        raise ValueError('c_list should start and end with "i"')
    if not n_list.size == d_list.size == len(c_list):
        raise ValueError('List sizes do not match!')
    if (np.real_if_close(n_list[0] * np.sin(th_0))).imag != 0:
        raise ValueError('Error in n0 or th0!')

    has_incoherent_internal_layers = any(
        coherency == 'i' for coherency in c_list[1:-1]
    )
    if not has_incoherent_internal_layers:
        return coh_tmm(pol, n_list, d_list, th_0, lam_vac)

    th_list = _forward_angle_list(n_list, th_0)
    incoherent_indices = [index for index, coherency in enumerate(c_list) if coherency == 'i']
    num_segments = len(incoherent_indices) - 1

    segments = []
    coefficients = []
    for segment_index in range(num_segments):
        left_index = incoherent_indices[segment_index]
        right_index = incoherent_indices[segment_index + 1]
        segment = _coherent_segment_transfer_data(
            pol, n_list, d_list, th_list, lam_vac, left_index, right_index
        )
        segments.append((left_index, right_index, segment))
        coefficients.append(
            _segment_intensity_coeffs(pol, n_list, th_list, segment, left_index, right_index)
        )

    # Single-pass amplitude-power factor exp(-2 Im delta) of each internal
    # incoherent layer (the right layer of every segment except the last, whose
    # right boundary is the semi-infinite exit medium). The round-trip factor
    # exp(-4 Im delta) that drives the phase integral is its square.
    single_pass = [1.0] * num_segments
    for segment_index in range(num_segments - 1):
        right_index = segments[segment_index][1]
        roundtrip = _incoherent_roundtrip_factor(
            n_list[right_index], d_list[right_index], th_list[right_index], lam_vac
        )
        single_pass[segment_index] = float(np.sqrt(roundtrip))

    # Right-to-left: q2 (round-trip intensity into the downstream assembly) and
    # the effective reflectance R_eff each segment presents to its left boundary.
    q2_list = [0.0] * num_segments
    effective_reflectance = [0.0] * num_segments
    downstream_reflectance = 0.0
    for segment_index in range(num_segments - 1, -1, -1):
        r_f, t_f, r_b, t_b, _ = coefficients[segment_index]
        if segment_index == num_segments - 1:
            q2 = 0.0
        else:
            q2 = downstream_reflectance * single_pass[segment_index] ** 2
        denom = 1.0 - r_b * q2
        if denom <= 0:
            raise ValueError(
                "Explicit PIM round-trip series diverged (1 - R_back*q2 <= 0). "
                "This is the total-internal-reflection regime, where the phase "
                "integration denominator is singular; use phase_average_tmm."
            )
        q2_list[segment_index] = q2
        effective_reflectance[segment_index] = r_f + t_f * t_b * q2 / denom
        downstream_reflectance = effective_reflectance[segment_index]

    # Left-to-right: forward intensity incident on each segment, including the
    # downstream multiple reflections and the single-pass attenuation of every
    # incoherent layer the flux crosses.
    forward_intensity = [0.0] * num_segments
    forward_intensity[0] = 1.0
    for segment_index in range(num_segments - 1):
        r_f, t_f, r_b, t_b, _ = coefficients[segment_index]
        forward_intensity[segment_index + 1] = (
            forward_intensity[segment_index]
            * t_f
            * single_pass[segment_index]
            / (1.0 - r_b * q2_list[segment_index])
        )

    reflectance = effective_reflectance[0]
    transmittance = forward_intensity[num_segments - 1] * coefficients[num_segments - 1][1]

    # Net layer-entering irradiances at every internal interface, from the
    # phase-integrated boundary power scaled by the segment's incident intensity.
    segment_boundary_power = []
    layer_absorption = zeros(n_list.size)
    for segment_index in range(num_segments):
        left_index, right_index, segment = segments[segment_index]
        beta_left = coefficients[segment_index][4]
        q2 = q2_list[segment_index]
        entering = []
        for local_index in range(1, segment['num_internal'] + 2):
            beta_prev = _power_beta(
                pol,
                segment['local_n'][local_index - 1],
                segment['local_th'][local_index - 1],
            )
            entering.append(
                forward_intensity[segment_index]
                * _phase_integrated_boundary_power(
                    pol,
                    beta_prev,
                    beta_left,
                    segment['partial_matrices'][local_index],
                    segment['full_matrix'],
                    q2,
                )
            )
        segment_boundary_power.append(entering)
        for local_index in range(1, segment['num_internal'] + 1):
            layer_absorption[left_index + local_index] = (
                entering[local_index - 1] - entering[local_index]
            )

    # Internal incoherent layers: absorption is the net flux entering from the
    # left minus the net flux leaving into the next segment.
    for segment_index in range(num_segments - 1):
        incoherent_layer = segments[segment_index][1]
        power_in = segment_boundary_power[segment_index][-1]
        power_out = segment_boundary_power[segment_index + 1][0]
        layer_absorption[incoherent_layer] = power_in - power_out

    layer_absorption[0] = reflectance
    layer_absorption[-1] = transmittance

    # At-source guard: a physical intensity solution has R, T in
    # [0, 1] with R + T <= 1 and finite everywhere. Beyond the critical angle the
    # explicit series produces finite-but-non-physical numbers; refuse instead of
    # fabricating one, matching inc_tmm's guard.
    r_power = float(np.real(reflectance))
    t_power = float(np.real(transmittance))
    tol = 1e-6
    if (not (np.isfinite(r_power) and np.isfinite(t_power))
            or r_power < -tol or t_power < -tol
            or r_power > 1 + tol or t_power > 1 + tol
            or (r_power + t_power) > 1 + tol):
        raise ValueError(
            "Explicit phase-integration TMM produced a non-physical result "
            "(R={:.4g}, T={:.4g}). This happens beyond the critical angle (total "
            "internal reflection into an incoherent layer), where the analytic "
            "phase integrals are singular. Reduce the angle below the critical "
            "angle, or mark the low-index layer coherent.".format(r_power, t_power)
        )

    power_entering_each_layer = np.empty(n_list.size, dtype=float)
    power_entering_each_layer[0] = 1.0
    for layer_index in range(1, n_list.size):
        power_entering_each_layer[layer_index] = (
            power_entering_each_layer[layer_index - 1]
            - layer_absorption[layer_index - 1]
        )

    return {
        'R': reflectance,
        'T': transmittance,
        'th_list': th_list,
        'pim_power_entering_each_layer': power_entering_each_layer,
        'pim_layer_absorption': layer_absorption,
        'pim_method': 'explicit_phase_integration_boundary_irradiance',
        'pol': pol,
        'n_list': n_list,
        'd_list': d_list,
        'c_list': c_list,
        'th_0': th_0,
        'lam_vac': lam_vac,
    }

def _coh_tmm_with_phase_overrides(pol, n_list, d_list, th_0, lam_vac, phase_overrides):
    n_list = np.array(n_list, dtype=complex)
    d_list = np.array(d_list, dtype=float)
    th_list = list_snell(n_list, th_0)
    kz_list = 2 * np.pi * n_list * np.cos(th_list) / lam_vac

    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)

    for layer_index in range(1, n_list.size - 1):
        if delta[layer_index].imag > 35:
            delta[layer_index] = delta[layer_index].real + 35j
            _warn_opaque_layers()
        if layer_index in phase_overrides:
            delta[layer_index] = phase_overrides[layer_index] + 1j * delta[layer_index].imag

    r, t, vw_list = _coh_tmm_amplitudes(pol, n_list, th_list, delta)
    return {
        'r': r,
        't': t,
        'R': R_from_r(r),
        'T': T_from_t(pol, t, n_list[0], n_list[-1], th_0, th_list[-1]),
        'power_entering': power_entering_from_r(pol, r, n_list[0], th_0),
        'vw_list': vw_list,
        'kz_list': kz_list,
        'th_list': th_list,
        'pol': pol,
        'n_list': n_list,
        'd_list': d_list,
        'th_0': th_0,
        'lam_vac': lam_vac,
    }

def phase_average_tmm(pol, n_list, d_list, c_list, th_0, lam_vac, samples_per_layer=25):
    """
    Deterministic phase averaging over incoherent-layer real phases.

    This is a conservative fallback for mixed stacks with evanescent
    incoherent layers where GTMM becomes numerically unstable. The lossy
    attenuation (Im(delta)) is kept exact while the incoherent real phase is
    midpoint-averaged over one pi-period, matching the regression oracle used
    in the test suite.
    """
    n_list = np.array(n_list, dtype=complex)
    d_list = np.array(d_list, dtype=float)
    incoherent_layers = [
        layer_index
        for layer_index, coherency in enumerate(c_list)
        if coherency == 'i' and 0 < layer_index < len(c_list) - 1
    ]

    if not incoherent_layers:
        return coh_tmm(pol, n_list, d_list, th_0, lam_vac)

    phase_samples = [
        (sample_index + 0.5) * np.pi / samples_per_layer
        for sample_index in range(samples_per_layer)
    ]
    total_weight = float(samples_per_layer ** len(incoherent_layers))
    reflectance_sum = 0.0
    transmittance_sum = 0.0
    layer_absorption_sum = np.zeros(len(d_list), dtype=float)

    for phase_tuple in itertools.product(phase_samples, repeat=len(incoherent_layers)):
        coherent_data = _coh_tmm_with_phase_overrides(
            pol,
            n_list,
            d_list,
            th_0,
            lam_vac,
            dict(zip(incoherent_layers, phase_tuple)),
        )
        reflectance_sum += float(coherent_data['R'])
        transmittance_sum += float(coherent_data['T'])
        layer_absorption_sum += np.array(absorp_in_each_layer(coherent_data), dtype=float)

    return {
        'R': reflectance_sum / total_weight,
        'T': transmittance_sum / total_weight,
        'phase_average_layer_absorption': layer_absorption_sum / total_weight,
        'pol': pol,
        'n_list': n_list,
        'd_list': d_list,
        'c_list': c_list,
        'th_0': th_0,
        'lam_vac': lam_vac,
    }

def inc_group_layers(n_list, d_list, c_list):
    """
    Helper function for inc_tmm. Groups and sorts layer information.

    See coh_tmm for definitions of n_list, d_list.

    c_list is "coherency list". Each entry should be 'i' for incoherent or 'c'
    for 'coherent'.

    A "stack" is a group of one or more consecutive coherent layers. A "stack
    index" labels the stacks 0,1,2,.... The "within-stack index" counts the
    coherent layers within the stack 1,2,3... [index 0 is the incoherent layer
    before the stack starts]

    An "incoherent layer index" labels the incoherent layers 0,1,2,...

    An "alllayer index" labels all layers (all elements of d_list) 0,1,2,...

    Returns info about how the layers relate:

    * stack_d_list[i] = list of thicknesses of each coherent layer in the i'th
      stack, plus starting and ending with "inf"
    * stack_n_list[i] = list of refractive index of each coherent layer in the
      i'th stack, plus the two surrounding incoherent layers
    * all_from_inc[i] = j means that the layer with incoherent index i has
      alllayer index j
    * inc_from_all[i] = j means that the layer with alllayer index i has
      incoherent index j. If j = nan then the layer is coherent.
    * all_from_stack[i1][i2] = j means that the layer with stack index i1 and
      within-stack index i2 has alllayer index j
    * stack_from_all[i] = [j1 j2] means that the layer with alllayer index i is
      part of stack j1 with withinstack-index j2. If stack_from_all[i] = nan
      then the layer is incoherent
    * inc_from_stack[i] = j means that the i'th stack comes after the layer
      with incoherent index j, and before the layer with incoherent index j+1.
    * stack_from_inc[i] = j means that the layer with incoherent index i comes
      immediately after the j'th stack. If j=nan, it is not immediately
      following a stack.
    * num_stacks = number of stacks
    * num_inc_layers = number of incoherent layers
    * num_layers = number of layers total
    """

    if (n_list.ndim != 1) or (d_list.ndim != 1):
        raise ValueError("Problem with n_list or d_list!")
    if (d_list[0] != inf) or (d_list[-1] != inf):
        raise ValueError('d_list must start and end with inf!')
    if (c_list[0] != 'i') or (c_list[-1] != 'i'):
        raise ValueError('c_list should start and end with "i"')
    if not n_list.size == d_list.size == len(c_list):
        raise ValueError('List sizes do not match!')
    inc_index = 0
    stack_index = 0
    stack_d_list = []
    stack_n_list = []
    all_from_inc = []
    inc_from_all = []
    all_from_stack = []
    stack_from_all = []
    inc_from_stack = []
    stack_from_inc = []
    stack_in_progress = False
    for alllayer_index in range(n_list.size):
        if c_list[alllayer_index] == 'c': #coherent layer
            inc_from_all.append(nan)
            if not stack_in_progress: #this layer is starting new stack
                stack_in_progress = True
                ongoing_stack_d_list = [inf, d_list[alllayer_index]]
                ongoing_stack_n_list = [n_list[alllayer_index-1],
                                        n_list[alllayer_index]]
                stack_from_all.append([stack_index,1])
                all_from_stack.append([alllayer_index-1, alllayer_index])
                inc_from_stack.append(inc_index-1)
                within_stack_index = 1
            else: #another coherent layer in the same stack
                ongoing_stack_d_list.append(d_list[alllayer_index])
                ongoing_stack_n_list.append(n_list[alllayer_index])
                within_stack_index += 1
                stack_from_all.append([stack_index, within_stack_index])
                all_from_stack[-1].append(alllayer_index)
        elif c_list[alllayer_index] == 'i': #incoherent layer
            stack_from_all.append(nan)
            inc_from_all.append(inc_index)
            all_from_inc.append(alllayer_index)
            if not stack_in_progress: #previous layer was also incoherent
                stack_from_inc.append(nan)
            else: #previous layer was coherent
                stack_in_progress = False
                stack_from_inc.append(stack_index)
                ongoing_stack_d_list.append(inf)
                stack_d_list.append(ongoing_stack_d_list)
                ongoing_stack_n_list.append(n_list[alllayer_index])
                stack_n_list.append(ongoing_stack_n_list)
                all_from_stack[-1].append(alllayer_index)
                stack_index += 1
            inc_index += 1
        else:
            raise ValueError("Error: c_list entries must be 'i' or 'c'!")
    return {'stack_d_list':stack_d_list,
            'stack_n_list':stack_n_list,
            'all_from_inc':all_from_inc,
            'inc_from_all':inc_from_all,
            'all_from_stack':all_from_stack,
            'stack_from_all':stack_from_all,
            'inc_from_stack':inc_from_stack,
            'stack_from_inc':stack_from_inc,
            'num_stacks':len(all_from_stack),
            'num_inc_layers':len(all_from_inc),
            'num_layers':len(n_list)}

def inc_tmm(pol, n_list, d_list, c_list, th_0, lam_vac):
    """
    Incoherent, or partly-incoherent-partly-coherent, transfer matrix method.

    See coh_tmm for definitions of pol, n_list, d_list, th_0, lam_vac.

    c_list is "coherency list". Each entry should be 'i' for incoherent or 'c'
    for 'coherent'.

    If an incoherent layer has real refractive index (no absorption), then its
    thickness doesn't affect the calculation results.

    See https://arxiv.org/abs/1603.02720 for physics background and some
    of the definitions.

    Outputs the following as a dictionary:

    * R--reflected wave power (as fraction of incident)
    * T--transmitted wave power (as fraction of incident)
    * VW_list-- n'th element is [V_n,W_n], the forward- and backward-traveling
      intensities, respectively, at the beginning of the n'th incoherent medium.
    * coh_tmm_data_list--n'th element is coh_tmm_data[n], the output of
      the coh_tmm program for the n'th "stack" (group of one or more
      consecutive coherent layers).
    * coh_tmm_bdata_list--n'th element is coh_tmm_bdata[n], the output of the
      coh_tmm program for the n'th stack, but with the layers of the stack
      in reverse order.
    * stackFB_list--n'th element is [F,B], where F is light traveling forward
      towards the n'th stack and B is light traveling backwards towards the n'th
      stack.
    * num_layers-- total number both coherent and incoherent.
    * power_entering_list--n'th element is the normalized Poynting vector
      crossing the interface into the n'th incoherent layer from the previous
      (coherent or incoherent) layer.
    * th_list--(complex) propagation angle in each layer, as list_snell gives it
    * pol, n_list, d_list, c_list, th_0, lam_vac--same as input (n_list and
      d_list as arrays); `inc_field` reads them
    * Plus, all the outputs of inc_group_layers

    """
    # Convert lists to numpy arrays if they're not already.
    n_list = array(n_list)
    d_list = array(d_list, dtype=float)

    # Input tests
    if (np.real_if_close(n_list[0]*np.sin(th_0))).imag != 0:
        raise ValueError('Error in n0 or th0!')

    group_layers_data = inc_group_layers(n_list, d_list, c_list)
    num_inc_layers = group_layers_data['num_inc_layers']
    num_stacks = group_layers_data['num_stacks']
    stack_n_list = group_layers_data['stack_n_list']
    stack_d_list = group_layers_data['stack_d_list']
    all_from_stack = group_layers_data['all_from_stack']
    all_from_inc = group_layers_data['all_from_inc']
    all_from_stack = group_layers_data['all_from_stack']
    stack_from_inc = group_layers_data['stack_from_inc']
    inc_from_stack = group_layers_data['inc_from_stack']

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = list_snell(n_list, th_0)

    # coh_tmm_data_list[i] is the output of coh_tmm for the i'th stack
    coh_tmm_data_list = []
    # coh_tmm_bdata_list[i] is the same stack as coh_tmm_data_list[i] but
    # with order of layers reversed
    coh_tmm_bdata_list = []
    for i in range(num_stacks):
        coh_tmm_data_list.append(coh_tmm(pol, stack_n_list[i],
                                         stack_d_list[i],
                                         th_list[all_from_stack[i][0]],
                                         lam_vac))
        coh_tmm_bdata_list.append(coh_tmm_reverse(pol, stack_n_list[i],
                                                  stack_d_list[i],
                                                  th_list[all_from_stack[i][0]],
                                                  lam_vac))

    # P_list[i] is fraction not absorbed in a single pass through i'th incoherent
    # layer.
    P_list = zeros(num_inc_layers)
    for inc_index in range(1,num_inc_layers-1): #skip 0'th and last (infinite)
        i = all_from_inc[inc_index]
        P_list[inc_index] = exp(-4 * np.pi * d_list[i]
                     * (n_list[i] * cos(th_list[i])).imag / lam_vac)
        # For a very opaque layer, reset P to avoid divide-by-0 and similar
        # errors.
        if P_list[inc_index] < 1e-30:
            P_list[inc_index] = 1e-30
    # T_list[i,j] and R_list[i,j] are transmission and reflection powers,
    # respectively, coming from the i'th incoherent layer, going to the j'th
    # incoherent layer. Only need to calculate this when j=i+1 or j=i-1.
    # (2D array is overkill but helps avoid confusion.)
    # initialize these arrays
    T_list = zeros((num_inc_layers, num_inc_layers))
    R_list = zeros((num_inc_layers, num_inc_layers))
    for inc_index in range(num_inc_layers-1): #looking at interface i -> i+1
        alllayer_index = all_from_inc[inc_index]
        nextstack_index = stack_from_inc[inc_index+1]
        if isnan(nextstack_index): #next layer is incoherent
            R_list[inc_index, inc_index+1] = (
                   interface_R(pol, n_list[alllayer_index],
                               n_list[alllayer_index+1],
                               th_list[alllayer_index],
                               th_list[alllayer_index+1]))
            T_list[inc_index, inc_index+1] = (
                   interface_T(pol, n_list[alllayer_index],
                               n_list[alllayer_index+1],
                               th_list[alllayer_index],
                               th_list[alllayer_index+1]))
            R_list[inc_index+1, inc_index] = (
                   interface_R(pol, n_list[alllayer_index+1],
                               n_list[alllayer_index],
                               th_list[alllayer_index+1],
                               th_list[alllayer_index]))
            T_list[inc_index+1, inc_index] = (
                   interface_T(pol, n_list[alllayer_index+1],
                               n_list[alllayer_index],
                               th_list[alllayer_index+1],
                               th_list[alllayer_index]))
        else: #next layer is coherent
            R_list[inc_index,inc_index+1] = (
                    coh_tmm_data_list[nextstack_index]['R'])
            T_list[inc_index,inc_index+1] = _opaque_floor(
                    coh_tmm_data_list[nextstack_index]['T'])
            R_list[inc_index+1,inc_index] = (
                    coh_tmm_bdata_list[nextstack_index]['R'])
            T_list[inc_index+1,inc_index] = _opaque_floor(
                    coh_tmm_bdata_list[nextstack_index]['T'])

    # L is the transfer matrix from the i'th to (i+1)st incoherent layer, see
    # https://arxiv.org/abs/1603.02720
    L_list = [nan] # L_0 is not defined because 0'th layer has no beginning.
    Ltilde = (array([[1,-R_list[1,0]],
                     [R_list[0,1],
                      T_list[1,0]*T_list[0,1] - R_list[1,0]*R_list[0,1]]])
                / T_list[0,1])
    for i in range(1,num_inc_layers-1):
        L = np.dot(
           array([[1/P_list[i],0],[0,P_list[i]]]),
           array([[1,-R_list[i+1,i]],
                  [R_list[i,i+1],
                   T_list[i+1,i]*T_list[i,i+1] - R_list[i+1,i]*R_list[i,i+1]]])
           ) / T_list[i,i+1]
        L_list.append(L)
        Ltilde = np.dot(Ltilde,L)
    T = 1 / Ltilde[0,0]
    R = Ltilde[1,0] / Ltilde[0,0]

    # Physical-bounds backstop. The two-flux incoherent recursion is an intensity
    # method: R and T are powers and must lie in [0, 1] with R + T <= 1. Beyond
    # the critical angle the interface/stack transmission into the next
    # incoherent block vanishes, the L-matrix divides by ~0, and the recursion
    # silently returns non-physical numbers (e.g. negative R) that would later
    # be clamped into a plausible-looking response. Refuse instead of
    # fabricating a number; the dispatch routes true TIR cases to
    # phase_average_tmm before reaching here, so this only fires on cases that
    # slipped past that predicate.
    R_power = float(np.real(R))
    T_power = float(np.real(T))
    tol = 1e-6
    if (not (np.isfinite(R_power) and np.isfinite(T_power))
            or R_power < -tol or T_power < -tol
            or R_power > 1 + tol or T_power > 1 + tol
            or (R_power + T_power) > 1 + tol):
        raise ValueError(
            "Incoherent transfer-matrix method produced a non-physical result "
            "(R={:.4g}, T={:.4g}). This happens beyond the critical angle "
            "(total internal reflection into an incoherent layer), where the "
            "intensity recursion is singular. Reduce the angle below the "
            "critical angle, or mark the low-index layer coherent.".format(
                R_power, T_power)
        )

    # VW_list[n] = [V_n, W_n], the forward- and backward-moving intensities
    # at the beginning of the n'th incoherent layer. VW_list[0] is undefined
    # because 0'th layer has no beginning.
    VW_list=zeros((num_inc_layers, 2))
    VW_list[0,:] = [nan, nan]
    VW = array([[T],[0]])
    VW_list[-1,:] = np.transpose(VW)
    for i in range(num_inc_layers-2, 0, -1):
        VW = np.dot(L_list[i], VW)
        VW_list[i,:] = np.transpose(VW)

    # stackFB_list[n]=[F,B] means that F is light traveling forward towards n'th
    # stack and B is light traveling backwards towards n'th stack.
    # Reminder: inc_from_stack[i] = j means that the i'th stack comes after the
    # layer with incoherent index j.
    stackFB_list = []
    for stack_index, prev_inc_index in enumerate(inc_from_stack):
        if prev_inc_index == 0: #stack starts right after semi-infinite layer.
            F = 1
        else:
            F = VW_list[prev_inc_index][0] * P_list[prev_inc_index]
        B = VW_list[prev_inc_index+1][1]
        stackFB_list.append([F,B])

    # power_entering_list[i] is the normalized Poynting vector crossing the
    # interface into the i'th incoherent layer from the previous (coherent or
    # incoherent) layer. See https://arxiv.org/abs/1603.02720 .
    power_entering_list = [1] #"1" by convention for infinite 0th layer.
    for i in range(1,num_inc_layers):
        prev_stack_index = stack_from_inc[i]
        if isnan(prev_stack_index):
            #case where this layer directly follows another incoherent layer
            if i == 1: #special case because VW_list[0] & A_list[0] are undefined
                power_entering_list.append(T_list[0,1]
                                            - VW_list[1][1]*T_list[1,0])
            else:
                power_entering_list.append(
                    VW_list[i-1][0]*P_list[i-1]*T_list[i-1,i]
                    - VW_list[i][1]*T_list[i,i-1])
        else: #case where this layer follows a coherent stack
            power_entering_list.append(
                stackFB_list[prev_stack_index][0] *
                 coh_tmm_data_list[prev_stack_index]['T']
                - stackFB_list[prev_stack_index][1] *
                 coh_tmm_bdata_list[prev_stack_index]['power_entering'])
    ans = {'T':T, 'R':R, 'VW_list':VW_list,
            'coh_tmm_data_list':coh_tmm_data_list,
            'coh_tmm_bdata_list':coh_tmm_bdata_list,
            'stackFB_list':stackFB_list,
            'power_entering_list':power_entering_list,
            'th_list': th_list, 'pol': pol, 'n_list': n_list, 'd_list': d_list,
            'c_list': c_list, 'th_0': th_0, 'lam_vac': lam_vac}
    ans.update(group_layers_data)
    return ans

def inc_absorp_in_each_layer(inc_data):
    """
    A list saying what proportion of light is absorbed in each layer.

    Assumes all reflected light is eventually absorbed in the 0'th medium, and
    all transmitted light is eventually absorbed in the final medium.

    Returns a list [layer0absorp, layer1absorp, ...]. Entries should sum to 1.

    inc_data is output of inc_tmm()
    """
    # Reminder: inc_from_stack[i] = j means that the i'th stack comes after the
    # layer with incoherent index j.
    # Reminder: stack_from_inc[i] = j means that the layer
    # with incoherent index i comes immediately after the j'th stack (or j=nan
    # if it's not immediately following a stack).

    stack_from_inc = inc_data['stack_from_inc']
    power_entering_list = inc_data['power_entering_list']
    # stackFB_list[n]=[F,B] means that F is light traveling forward towards n'th
    # stack and B is light traveling backwards towards n'th stack.
    stackFB_list = inc_data['stackFB_list']
    absorp_list = []

    # loop through incoherent layers, excluding the final layer
    for i, power_entering in enumerate(power_entering_list[:-1]):
        if isnan(stack_from_inc[i+1]):
            # case that incoherent layer i is right before another incoherent layer
            absorp_list.append(power_entering_list[i]-power_entering_list[i+1])
        else: #incoherent layer i is immediately before a coherent stack
            j = stack_from_inc[i+1]
            coh_tmm_data = inc_data['coh_tmm_data_list'][j]
            coh_tmm_bdata = inc_data['coh_tmm_bdata_list'][j]
            # First, power in the incoherent layer...
            power_exiting = (
               stackFB_list[j][0] * coh_tmm_data['power_entering']
                  - stackFB_list[j][1] * coh_tmm_bdata['T'])
            absorp_list.append(power_entering_list[i]-power_exiting)
            # Next, power in the coherent stack...
            stack_absorp = ((stackFB_list[j][0] *
                        absorp_in_each_layer(coh_tmm_data))[1:-1]
                       + (stackFB_list[j][1] *
                        absorp_in_each_layer(coh_tmm_bdata))[-2:0:-1])
            absorp_list.extend(stack_absorp)
    # final semi-infinite layer
    absorp_list.append(inc_data['T'])
    return absorp_list

def pim_absorp_in_each_layer(tmm_data):
    """
    Layer absorptance from phase-integrated layer-entering irradiances.

    This is the adapted hook used by multilayer.app for the Puhan-Burmen-Tuma-
    Fajfar Phase Integration Method (PIM), Coatings 2019, 9, 536. In PIM, layer
    absorptance is the difference between neighboring normalized irradiances
    entering the layer and entering the following layer. The coherent TMM and
    incoherent GTMM state already contains those boundary fluxes:

    * coherent stacks use the exact Poynting vector at layer boundaries;
    * mixed coherent/incoherent stacks use phase-averaged intensity transfer
      matrices for incoherent layers and coherent boundary fluxes inside stacks.

    This deliberately does not estimate layer-specific absorption with a
    Beer-Lambert per-layer shortcut.
    """
    if 'pim_layer_absorption' in tmm_data:
        return np.array(tmm_data['pim_layer_absorption'], dtype=float)

    if 'phase_average_layer_absorption' in tmm_data:
        return np.array(tmm_data['phase_average_layer_absorption'], dtype=float)

    if 'coh_tmm_data_list' in tmm_data:
        return np.array(inc_absorp_in_each_layer(tmm_data), dtype=float)

    return np.array(absorp_in_each_layer(tmm_data), dtype=float)

def inc_find_absorp_analytic_fn(layer, inc_data):
    """
    Outputs an absorp_analytic_fn object for a coherent layer within a
    partly-incoherent stack.

    inc_data is output of inc_tmm()
    """
    j = inc_data['stack_from_all'][layer]
    if np.any(isnan(j)):
        raise ValueError('layer must be coherent for this function!')
    [stackindex, withinstackindex] = j
    forwardfunc = absorp_analytic_fn()
    forwardfunc.fill_in(inc_data['coh_tmm_data_list'][stackindex],
                        withinstackindex)
    forwardfunc.scale(inc_data['stackFB_list'][stackindex][0])
    backfunc = absorp_analytic_fn()
    backfunc.fill_in(inc_data['coh_tmm_bdata_list'][stackindex],
               -1-withinstackindex)
    backfunc.scale(inc_data['stackFB_list'][stackindex][1])
    backfunc.flip()
    return forwardfunc.add(backfunc)


def _flux_per_amplitude(pol, n, th):
    """Normal Poynting flux of one plane wave of unit amplitude (E itself for s; the amplitude whose E is
    (cos th, 0, -sin th) for p) in a medium of index n at angle th, in T_from_t's units: Re(n cos th) for s,
    Re(n cos th*) for p."""
    if pol == 's':
        return float((n * cos(th)).real)
    if pol == 'p':
        return float((n * conj(cos(th))).real)
    raise ValueError("Polarization must be 's' or 'p'")


def _incoherent_layer_field(layer, z, inc_data, flux_0):
    """|E|^2, |H|^2 in an incoherent layer: V and W are powers, each one plane wave of this medium."""
    pol, n_list, th_list = inc_data['pol'], inc_data['n_list'], inc_data['th_list']
    n, th, n_0 = n_list[layer], th_list[layer], n_list[0]
    flux = _flux_per_amplitude(pol, n, th)
    if is_evanescent_angle(n, th) or flux <= 0:
        raise ValueError(
            f"layer {layer} is incoherent and carries no propagating wave at this angle (Re(n cos theta) = "
            f"{(n * cos(th)).real:.3g}): inc_tmm's powers fix no field there")
    lam_vac, d_list = inc_data['lam_vac'], inc_data['d_list']
    alpha = 4 * np.pi * (n * cos(th)).imag / lam_vac
    last = len(n_list) - 1
    if layer == 0:
        forward = exp(-alpha * z)                       # the incident wave, power 1 at the first interface
        backward = float(np.real(inc_data['R'])) * exp(alpha * z)
    else:
        V, W = inc_data['VW_list'][inc_data['inc_from_all'][layer]]
        forward = V * exp(-alpha * z)
        if layer == last:
            backward = W * exp(alpha * z)               # W = 0: nothing returns from the exit medium
        else:
            # W is the backward power at the layer's front face; inc_tmm propagates it through the single-pass
            # factor P (floored at 1e-30 for an opaque layer), so take the back face's power W/P and attenuate
            # from there: no overflow, and the same numbers as W e^{alpha z} wherever P is not floored.
            d = d_list[layer]
            P = max(exp(-alpha * d), 1e-30)
            backward = (W / P) * exp(-alpha * (d - z))
    # Each power over the flux one unit amplitude of this medium carries, in units of the incident amplitude.
    amplitude2 = (forward + backward) * flux_0 / flux
    lateral = abs(cos(th)) ** 2 + abs(sin(th)) ** 2     # |E|^2 per |amplitude|^2 of a p wave, |H|^2 of an s wave
    admittance2 = abs(n) ** 2 / abs(n_0) ** 2
    if pol == 's':
        return amplitude2, amplitude2 * admittance2 * lateral
    return amplitude2 * lateral, amplitude2 * admittance2


def _coherent_layer_field(layer, z, inc_data, flux_0):
    """|E|^2, |H|^2 in a coherent layer: the stack's forward and reversed solutions, each weighted by its power
    converted to the amplitude^2 of its own incident medium."""
    pol, n_0 = inc_data['pol'], inc_data['n_list'][0]
    stack_index, within = (int(value) for value in inc_data['stack_from_all'][layer])
    forward_data = inc_data['coh_tmm_data_list'][stack_index]
    backward_data = inc_data['coh_tmm_bdata_list'][stack_index]
    F, B = inc_data['stackFB_list'][stack_index]
    d = inc_data['d_list'][layer]
    e2 = np.zeros(np.shape(z))
    h2 = np.zeros(np.shape(z))
    for data, power, index, depth in ((forward_data, F, within, z),
                                      (backward_data, B, len(backward_data['d_list']) - 1 - within, d - z)):
        power = float(np.real(power))
        if power == 0:
            continue
        _refuse_critical(data, "the field")
        flux = _flux_per_amplitude(pol, data['n_list'][0], data['th_0'])
        if flux <= 0:
            raise ValueError(f"the medium in front of the coherent stack holding layer {layer} carries no "
                             "propagating wave at this angle: inc_tmm's powers fix no field there")
        weight = power * flux_0 / flux
        Ef, Eb = _layer_waves(index, depth, data)
        layer_e2, layer_h2 = _field_intensities(pol, data['th_list'][index], data['y_list'][index], n_0, Ef, Eb)
        e2 = e2 + weight * layer_e2
        h2 = h2 + weight * layer_h2
    return e2, h2


def inc_field(layer, distance, inc_data):
    """
    Phase-averaged |E|^2 and |H|^2 in a stack with incoherent layers, at ``distance`` into ``layer``.

    ``inc_data`` is the output of `inc_tmm` (or of `pim_tmm` for a stack with an incoherent interior layer).
    ``layer`` indexes the full stack, half-spaces included, and (layer, distance) is defined as in
    `position_resolved`: distance <= 0 in the incident medium, 0 <= distance <= d in a finite layer, >= 0 in the
    exit medium. ``distance`` may be an array.

    Returns ``{'E2': |E|^2/|E0|^2, 'H2': |H|^2/|H0|^2, 'coherent': bool}`` (floats for a scalar distance,
    arrays otherwise) in `position_resolved`'s units: the incident wave has unit amplitude and |H0| = |n_0| |E0|.
    ``coherent`` says whether ``layer`` belongs to a coherent stack.

    `inc_tmm`'s intensities (``VW_list``, ``stackFB_list``) are powers, normal Poynting flux over the incident
    flux, not |E|^2. Each is converted to the field amplitude^2 of the medium it travels in, through the flux one
    unit amplitude carries there, Re(n cos th) for s and Re(n cos th*) for p (the factor of `T_from_t`):

    * In an incoherent layer (the two half-spaces included) the forward and backward powers V e^{-alpha z} and
      W e^{+alpha z}, alpha = 4 pi Im(n cos th)/lam_vac, are each one plane wave, of amplitude^2
      a = (power) Re(n_0 cos th_0)/Re(n cos th) (s; cos th* for p). The phase average has no cross terms, so
      |E|^2 = a_f + a_b for s and (|cos th|^2 + |sin th|^2)(a_f + a_b) for p, and
      |H|^2 = |n/n_0|^2 (|cos th|^2 + |sin th|^2)(a_f + a_b) for s and |n/n_0|^2 (a_f + a_b) for p. Lossy media
      and complex angles included; for a lossless slab at normal incidence |E|^2 = (V + W) n_0/n and
      |H|^2 = (V + W) n/n_0.
    * A coherent layer is the stack's forward solution (``coh_tmm_data_list``) weighted by the power F arriving
      from the incoherent medium a in front of it, plus its reversed solution (``coh_tmm_bdata_list``) weighted
      by the power B arriving from the medium b behind it. Each solution's field is per unit amplitude of its own
      incident medium, so the weights are F Re(n_0 cos th_0)/Re(n_a cos th_a) and B Re(n_0 cos th_0)/Re(n_b cos
      th_b) (cos th* for p). The two are mutually incoherent and add as intensities.

    An incoherent layer carries no standing wave: within a coherence length of an interface the true field still
    interferes with its reflection (the incident wave with the reflected light included), and this model drops
    that by construction. Refuses (ValueError) an evanescent incoherent layer, where powers fix no field, and a
    layer of a coherent stack at its own critical angle (as `position_resolved` does).

    Before this function multilayer.app plotted the powers themselves as |E|^2 and |H|^2 (assurance WP07,
    O-W7-12): |E|^2 Re(n cos th)/(n_0 cos th_0) too high inside an incoherent layer (x1.50 in glass) and |H|^2 too
    low, and a coherent film's backward share B taken as |E0|^2 of the slab behind it.
    """
    if 'VW_list' not in inc_data:
        raise ValueError("inc_field takes inc_tmm's output (or pim_tmm's for a stack with an incoherent interior "
                         "layer); for a coherent stack use position_resolved")
    num_layers = len(inc_data['n_list'])
    if not 0 <= layer < num_layers:
        raise ValueError(f"layer must be between 0 and {num_layers - 1}")
    z = np.asarray(distance, dtype=float)
    if layer == 0:
        inside = np.all(z <= 0)
    elif layer == num_layers - 1:
        inside = np.all(z >= 0)
    else:
        inside = np.all((z >= 0) & (z <= inc_data['d_list'][layer]))
    if not inside:
        raise ValueError(f"distance is outside layer {layer} (<= 0 in the incident medium, 0..d in a finite layer, "
                         ">= 0 in the exit medium)")
    flux_0 = _flux_per_amplitude(inc_data['pol'], inc_data['n_list'][0], inc_data['th_0'])
    coherent = not np.any(isnan(inc_data['stack_from_all'][layer]))
    if coherent:
        e2, h2 = _coherent_layer_field(layer, z, inc_data, flux_0)
    else:
        e2, h2 = _incoherent_layer_field(layer, z, inc_data, flux_0)
    e2 = np.real(np.broadcast_to(e2, z.shape)).astype(float)
    h2 = np.real(np.broadcast_to(h2, z.shape)).astype(float)
    if z.ndim == 0:
        return {'E2': float(e2), 'H2': float(h2), 'coherent': coherent}
    return {'E2': e2, 'H2': h2, 'coherent': coherent}

def layered_response(pol, n_list, d_list, c_list, th_0, lam_vac):
    """
    R, T and every region's absorptance (incident medium first, the exit medium's
    entry is T) of one plane wave, with the solver its coherence column calls for:
    coh_tmm when every internal layer is coherent ('c'; c_list None means all),
    else pim_tmm, or phase_average_tmm once an incoherent layer is evanescent or
    off its forward branch (beyond a critical angle the incoherent recursion
    divides by a vanishing interface transmission). This is the dispatch the
    multilayer.app planar calculator applies to a plane wave.
    """
    n_list = np.array(n_list, dtype=complex)
    if c_list is None or all(c == 'c' for c in c_list[1:-1]):
        data = coh_tmm(pol, n_list, d_list, th_0, lam_vac)
    else:
        th_list = list_snell(n_list, th_0)
        evanescent = any(
            c == 'i' and (not is_forward_angle(n_list[j], th_list[j]) or is_evanescent_angle(n_list[j], th_list[j]))
            for j, c in enumerate(c_list)
        )
        if evanescent:
            samples = 25 if sum(c == 'i' for c in c_list[1:-1]) <= 2 else 15
            data = phase_average_tmm(pol, n_list, d_list, c_list, th_0, lam_vac, samples_per_layer=samples)
        else:
            data = pim_tmm(pol, n_list, d_list, c_list, th_0, lam_vac)
    return {'R': float(np.real(data['R'])), 'T': float(np.real(data['T'])),
            'absorption': np.array(pim_absorp_in_each_layer(data), dtype=float), 'data': data}
