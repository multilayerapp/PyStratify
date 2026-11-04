# FROZEN REFERENCE -- do not edit, do not import from production code.
#
# The replication-corpus harvester's printed-model evaluator, copied verbatim (the whole file,
# unmodified below this header) from repository ``thinfilm-harvester``, commit ``36afd65``,
# ``nk_model_table.py`` (lines 1-498).  The Tauc-Lorentz-Urbach (lines 137-245), Forouhi-Bloomer
# (lines 248-295) and ``drude-n`` (``evaluate``, lines 297-374) code was ported to
# ``pystratify.dispersion`` (assurance WP07-D2); this copy is the independent check kept after
# the harvester calls PyStratify.  tests/test_dispersion_families.py compares against it.
"""Evaluate a paper's printed dispersion model into a wavelength_nm,n,k CSV for audit_probe.py --nk-file.

Every parameter carries its unit explicitly on the command line -- this helper never guesses
one, which is the whole reason verify.py refuses the bare numbers.  Sign convention of the
output: n + ik with k >= 0 (passive).  The permittivity is taken as eps = eps1 + i*eps2 with
eps2 >= 0; a model printed with the e^{+jwt} engineering convention (eps' - j eps'') is the
same physics -- enter it with positive damping and this helper returns the passive branch.

Models (all angular frequencies may be given in rad/s, Hz (=cycles/s), THz, eV or cm-1):
  drude      eps = eps_inf - wp^2 / (w^2 + i*gamma*w)
             --eps-inf X --wp VAL UNIT --gamma VAL UNIT
  drude-n    same, with wp from carriers: wp^2 = N e^2 / (m* eps0)
             --eps-inf X --N VAL cm-3|m-3 --mstar X (units of m_e) --gamma VAL UNIT
  lorentz-drude (Rakic form)
             eps = 1 - f0 wp^2/(w^2 + i g0 w) + sum_j fj wp^2/(wj^2 - w^2 - i gj w)
             --wp VAL UNIT --f0 X --g0 VAL UNIT --osc 'f,w,g' (repeat; w,g in the --osc-unit)
             [--eps-inf X  replaces the leading 1]
  sellmeier1 n^2 = a + b*lam^2/(lam^2 - c^2)          --a --b --c  (lam, c in um)
  sellmeier  n^2 = 1 + sum Bi lam^2/(lam^2 - Ci)       --B 'b1,b2,..' --C 'c1,c2,..' (Ci in um^2)
  cauchy     n = A + B/lam^2 + C/lam^4                 --A --B --C  (lam in um)
  expr       any numpy expression in lam_um / lam_nm / w_ev returning complex n or (use --eps) eps
             --expr 'np.sqrt(4.99 + 1/(96.6*lam_um**1.1) + 1/(4.6*lam_um**1.95))'
  tauc-lorentz-urbach (Foldyna's TLU, as printed in 10.3390/coatings10111063 Eqs. 36-42)
             eps2 = A E0 C (E-Eg)^2 / (E [(E^2-E0^2)^2 + C^2 E^2])   for E >= Ec
                  = (Au/E) exp(E/Eu)                                 for 0 < E < Ec
             Eu, Au from continuity of eps2 AND its derivative at Ec; eps1 = eps1_inf + KK
             --A-eV --E0-eV --C-eV --Eg-eV --Ec-eV --eps1-inf   (every energy in eV; A in eV)
  forouhi-bloomer (single-term, amorphous: Forouhi & Bloomer, Phys. Rev. B 34, 7018 (1986))
             k(E) = A (E-Eg)^2 / (E^2 - B E + C)   for E > Eg, 0 below
             n(E) = n_inf + (B0 E + C0) / (E^2 - B E + C),  Q = (4C - B^2)^(1/2) / 2
             B0 = (A/Q)(-B^2/2 + Eg B - Eg^2 + C),  C0 = (A/Q)((Eg^2 + C) B/2 - 2 Eg C)
             --A X --B-eV --C-eV2 --n-inf --Eg-eV   (A dimensionless; refuses 4C - B^2 <= 0)
Common:  --lo NM --hi NM [--step NM] --out file.csv
Example:
  nk_model_table.py drude --eps-inf 3.91 --wp 2.65e15 rad/s --gamma 2.05e14 rad/s --lo 400 --hi 2500 --out /tmp/ito.csv

The same evaluation is a library (`evaluate`, `evaluate_spec`): `nk_model_bindings.json`
holds the printed models a human read off the page, with every parameter's unit, and the
verifier evaluates them through `evaluate_spec` -- the code that wrote the reviewers'
probe tables, so a binding reproduces its review's probe input exactly
(tests/test_nk_model_bindings.py).
"""
import argparse
import sys

import numpy as np
from scipy.special import expi

HBAR_EV_S = 6.582119569e-16
C = 299792458.0
E = 1.602176634e-19
ME = 9.1093837015e-31
EPS0 = 8.8541878128e-12

MODELS = ("drude", "drude-n", "lorentz-drude", "sellmeier1", "sellmeier", "cauchy", "expr",
          "tauc-lorentz-urbach", "forouhi-bloomer")


class ModelError(ValueError):
    """A model, a parameter or a unit this evaluator will not guess."""


def to_rad_s(value, unit):
    u = str(unit).strip().lower().replace(" ", "")
    v = float(value)
    if u in ("rad/s", "rads", "rad/s^-1", "s-1_angular"):
        return v
    if u in ("hz", "1/s", "s-1", "cycles/s"):
        return 2 * np.pi * v
    if u == "thz":
        return 2 * np.pi * v * 1e12
    if u == "thz_angular":
        return v * 1e12
    if u == "ev":
        return v / HBAR_EV_S
    if u == "mev":
        return v * 1e-3 / HBAR_EV_S
    if u in ("cm-1", "cm^-1", "1/cm"):
        return 2 * np.pi * C * 100.0 * v
    raise ModelError(f"unknown frequency unit {unit!r}: use rad/s, Hz, THz, THz_angular, eV, meV or cm-1")


# Relative to |eps|: rounding leaves a near-real permittivity a few ulp below zero, a gain
# medium does not.
GAIN_TOLERANCE = 1e-9


def nk_from_eps(eps):
    """n + ik with k >= 0 from a PASSIVE permittivity (Im eps >= 0 under e^{-i w t}).

    The branch flip picks Im(n) >= 0, and for a gain permittivity (Im eps < 0) that is
    -n: a layer's transfer matrix is invariant under n -> -n, so an internal gain layer
    computed exactly while the record showed -n as its index, passing the only passivity
    check on the path (hand-off Q12, `10.1364/josab.36.002038`, refused as gain). So gain
    is refused here, loudly, until a ruling admits it. A paper writing e^{+i w t} prints
    a lossy eps with Im < 0: its binding states the conjugate. Measured 2026-10-09: no
    printed model on either corpus reaches this.
    """
    eps = np.asarray(eps).astype(complex)
    if np.any(eps.imag < -GAIN_TOLERANCE * np.maximum(1.0, np.abs(eps))):
        raise ModelError(
            "the permittivity has Im(eps) < 0, a gain medium under e^{-i w t}; gain is not "
            "admitted, and a lossy eps printed under e^{+i w t} is bound as its conjugate")
    # Rounding noise below the tolerance is zero loss: left signed, sqrt(2.25 - 1e-17j)
    # flipped to -1.5 + 0j, the same silent -n on a lossless film.
    eps = eps.real + 1j * np.where(eps.imag < 0, 0.0, eps.imag)
    n = np.sqrt(eps)
    # passive branch: Im(n) >= 0
    n = np.where(n.imag < 0, -n, n)
    return n


def _quantity(params, key):
    """A frequency parameter, which must arrive as [value, unit]: never a bare number."""
    pair = params.get(key)
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ModelError(f"parameter {key!r} must be [value, unit], got {pair!r}")
    return to_rad_s(*pair)


def _number(params, key, default=None):
    value = params.get(key, default)
    if value is None:
        raise ModelError(f"parameter {key!r} is required")
    return float(value)


def _floats(value):
    if isinstance(value, str):
        return [float(x) for x in value.split(",")]
    return [float(x) for x in value]


TLU_PARAMETERS = ("A_eV", "E0_eV", "C_eV", "Eg_eV", "Ec_eV", "eps1_inf")


def tlu_parameters(params):
    """The six TLU parameters as floats, every energy in eV (the unit is in the key's name).

    Refuses what the closed forms cannot evaluate: A, E0, C > 0 (eps2 >= 0); 0 < Eg < Ec
    (the Urbach tail joins above the gap); C < sqrt(2) E0 (gamma and alpha of Eq. 41 real).
    """
    missing = [k for k in TLU_PARAMETERS if params.get(k) is None]
    if missing:
        raise ModelError(f"tauc-lorentz-urbach needs {', '.join(missing)} (energies in eV)")
    vals = {}
    for k in TLU_PARAMETERS:
        v = params[k]
        if isinstance(v, (list, tuple)):
            raise ModelError(f"parameter {k!r} is a number in eV (the unit is in its name), got {v!r}")
        vals[k] = float(v)
    A, E0, C, Eg, Ec = (vals[k] for k in TLU_PARAMETERS[:5])
    if min(A, E0, C) <= 0:
        raise ModelError("tauc-lorentz-urbach needs A, E0 and C > 0")
    if not 0 < Eg < Ec:
        raise ModelError("tauc-lorentz-urbach needs 0 < Eg < Ec")
    if C ** 2 >= 2 * E0 ** 2:
        raise ModelError("tauc-lorentz-urbach closed forms (Eq. 41) need C < sqrt(2) E0")
    return vals


def tlu_urbach(params):
    """(Eu, Au) in eV: the Urbach energy and amplitude that make eps2 AND its first derivative
    continuous at Ec -- the condition the paper states for its Eq. (37).

    Matching d ln(eps2)/dE of the two branches at Ec gives
        1/Eu = 2/(Ec - Eg) - 2 Ec [C^2 + 2(Ec^2 - E0^2)] / [C^2 Ec^2 + (Ec^2 - E0^2)^2],
    which is Eq. (37) as printed EXCEPT that the page prints a factor 2 on (Ec^2 - E0^2)^2 in
    the denominator. With that factor the derivative is discontinuous; the continuity form is
    also the closer to every Eu in the paper's Table 2 (S1-S4: 78.5/68.9/79.1/79.1 meV against
    the printed 75/68/78/78; the printed factor gives 79.1/69.3/79.5/79.5).
    Au = exp(-Ec/Eu) A E0 C (Ec - Eg)^2 / [(Ec^2 - E0^2)^2 + C^2 Ec^2], as printed.
    """
    p = tlu_parameters(params)
    A, E0, C, Eg, Ec = (p[k] for k in TLU_PARAMETERS[:5])
    D = (Ec ** 2 - E0 ** 2) ** 2 + C ** 2 * Ec ** 2
    inv = 2.0 / (Ec - Eg) - 2.0 * Ec * (C ** 2 + 2.0 * (Ec ** 2 - E0 ** 2)) / D
    if inv <= 0:
        raise ModelError("tauc-lorentz-urbach: these parameters give no positive Urbach energy")
    Eu = 1.0 / inv
    return Eu, np.exp(-Ec / Eu) * A * E0 * C * (Ec - Eg) ** 2 / D


def tlu_eps2(params, E):
    """Eq. (36): Tauc-Lorentz above Ec, the Urbach tail below; E in eV, eps2 >= 0."""
    p = tlu_parameters(params)
    A, E0, C, Eg, Ec = (p[k] for k in TLU_PARAMETERS[:5])
    E = np.asarray(E, dtype=float)
    Eu, _ = tlu_urbach(params)
    # Au exp(E/Eu) written as eps2(Ec) Ec exp((E - Ec)/Eu): the same number, no overflow.
    tail_amp = A * E0 * C * (Ec - Eg) ** 2 / ((Ec ** 2 - E0 ** 2) ** 2 + C ** 2 * Ec ** 2)
    with np.errstate(over="ignore", under="ignore"):
        tl = A * E0 * C * (E - Eg) ** 2 / (E * ((E ** 2 - E0 ** 2) ** 2 + C ** 2 * E ** 2))
        tail = tail_amp / E * np.exp(np.minimum(E - Ec, 0.0) / Eu)
    return np.where(E >= Ec, tl, tail)


def tlu_eps1(params, E):
    """Eqs. (39)-(42): eps1 = eps1_inf + eps1_TL + eps1_UT, the closed-form KK of Eq. (38).

    eps1_TL is Eq. (40) with Eq. (41) exactly as printed (it agrees with a numerical principal
    value of Eq. 38 to < 1e-6). eps1_UT is the KK integral of the Urbach branch over (0, Ec):
        (Au / pi E) { e^{-E/Eu} [Ei(E/Eu) - Ei((Ec+E)/Eu)] + e^{E/Eu} [Ei((Ec-E)/Eu) - Ei(-E/Eu)] },
    with re(expint(-y)) = -Ei(y). Its first line is Eq. (42)'s first line; the page's SECOND
    line prints exp(-E/Eu) and -re(expint(E/Eu)) where the integral gives exp(+E/Eu) and
    +expint(E/Eu) -- as printed it misses Eq. (38) by ~4e-3 at 1 eV, this form by < 1e-6
    (tests/test_nk_model_tlu.py). Both log singularities at E = Ec cancel (eps2 is
    continuous there); a grid point within 1e-7 eV of Ec is read 1e-7 eV above it.
    """
    p = tlu_parameters(params)
    A, E0, C, Eg, Ec, einf = (p[k] for k in TLU_PARAMETERS)
    E = np.asarray(E, dtype=float)
    E = np.where(np.abs(E - Ec) < 1e-7, Ec + 1e-7, E)
    Eu, _ = tlu_urbach(params)
    gam2 = E0 ** 2 - C ** 2 / 2.0
    alpha = np.sqrt(4.0 * E0 ** 2 - C ** 2)
    s4 = (E ** 2 - E0 ** 2) ** 2 + C ** 2 * E ** 2
    aL = (Eg ** 2 - E0 ** 2) * E ** 2 + Eg ** 2 * C ** 2 - E0 ** 2 * (E0 ** 2 + 3 * Eg ** 2)
    aA = (E ** 2 - E0 ** 2) * (E0 ** 2 + Eg ** 2) + Eg ** 2 * C ** 2
    pi = np.pi
    tl = (-A * E0 * C * (E ** 2 + Eg ** 2) / (pi * s4 * E) * np.log(np.abs(Ec - E) / (Ec + E))
          + 2 * A * E0 * C * Eg / (pi * s4)
          * np.log(np.abs(Ec - E) * (Ec + E) / np.sqrt((E0 ** 2 - Ec ** 2) ** 2 + C ** 2 * Ec ** 2))
          + A * C * aL / (2 * pi * s4 * alpha * E0)
          * np.log((E0 ** 2 + Ec ** 2 + alpha * Ec) / (E0 ** 2 + Ec ** 2 - alpha * Ec))
          - A * aA / (pi * s4 * E0)
          * (pi - np.arctan((2 * Ec + alpha) / C) - np.arctan((2 * Ec - alpha) / C))
          + 4 * A * E0 * Eg * (E ** 2 - gam2) / (pi * s4 * alpha)
          * (pi / 2 - np.arctan(2 * (Ec ** 2 - gam2) / (alpha * C))))
    # Au e^{+-E/Eu} carried as eps2(Ec) Ec e^{(+-E - Ec)/Eu}, so no factor overflows.
    amp = A * E0 * C * (Ec - Eg) ** 2 / ((Ec ** 2 - E0 ** 2) ** 2 + C ** 2 * Ec ** 2)
    x = E / Eu
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lower = np.exp(-(E + Ec) / Eu) * (expi(x) - expi((Ec + E) / Eu))
        upper = np.exp((E - Ec) / Eu) * (expi((Ec - E) / Eu) - expi(-x))
    ut = amp / (pi * E) * (lower + upper)
    return einf + tl + ut


def tlu_eps(params, E):
    """Complex eps1 + i eps2 of the TLU model at photon energies E (eV)."""
    return tlu_eps1(params, E) + 1j * tlu_eps2(params, E)


FB_PARAMETERS = ("A", "B_eV", "C_eV2", "n_inf", "Eg_eV")


def fb_parameters(params):
    """The five single-term Forouhi-Bloomer parameters as floats (units in the key names:
    A dimensionless, B in eV, C in eV^2, n_inf dimensionless, Eg in eV).

    Refuses what the 1986 formulas cannot take: A > 0 (k >= 0 above the gap); 4C - B^2 > 0
    (Q real, and the denominator E^2 - B E + C = (E - B/2)^2 + Q^2 never vanishes, so k >= 0
    at every E); Eg >= 0.
    """
    missing = [k for k in FB_PARAMETERS if params.get(k) is None]
    if missing:
        raise ModelError(f"forouhi-bloomer needs {', '.join(missing)} (B in eV, C in eV^2, Eg in eV)")
    vals = {}
    for k in FB_PARAMETERS:
        v = params[k]
        if isinstance(v, (list, tuple)):
            raise ModelError(f"parameter {k!r} is a number (its unit is in its name), got {v!r}")
        vals[k] = float(v)
    if vals["A"] <= 0:
        raise ModelError("forouhi-bloomer needs A > 0")
    if 4 * vals["C_eV2"] - vals["B_eV"] ** 2 <= 0:
        raise ModelError(f"forouhi-bloomer needs 4C - B^2 > 0 (Q real), got "
                         f"4 x {vals['C_eV2']} - {vals['B_eV']}^2 = "
                         f"{4 * vals['C_eV2'] - vals['B_eV'] ** 2:.4g}")
    if vals["Eg_eV"] < 0:
        raise ModelError("forouhi-bloomer needs Eg >= 0")
    return vals


def fb_nk(params, E):
    """Complex n + ik of the single-term Forouhi-Bloomer model at photon energies E (eV).

    Forouhi & Bloomer, Phys. Rev. B 34, 7018 (1986), for amorphous semiconductors and
    dielectrics: k(E) = A (E - Eg)^2 / (E^2 - B E + C) above Eg and 0 below it;
    n(E) = n_inf + (B0 E + C0) / (E^2 - B E + C), the Kramers-Kronig partner they derive, with
    Q = (4C - B^2)^(1/2) / 2, B0 = (A/Q)(-B^2/2 + Eg B - Eg^2 + C),
    C0 = (A/Q)((Eg^2 + C) B/2 - 2 Eg C).
    """
    p = fb_parameters(params)
    A, B, C, ninf, Eg = (p[k] for k in FB_PARAMETERS)
    E = np.asarray(E, dtype=float)
    Q = 0.5 * np.sqrt(4 * C - B ** 2)
    B0 = A / Q * (-B ** 2 / 2 + Eg * B - Eg ** 2 + C)
    C0 = A / Q * ((Eg ** 2 + C) * B / 2 - 2 * Eg * C)
    D = E ** 2 - B * E + C
    n = ninf + (B0 * E + C0) / D
    k = np.where(E > Eg, A * (E - Eg) ** 2 / D, 0.0)
    return n + 1j * k


def evaluate(model, params, lam_nm):
    """Complex index n + ik (k >= 0) of a printed `model` with `params` at `lam_nm`.

    `params` uses the command line's names (`eps_inf`, `wp`, `gamma`, `N`, `mstar`, `f0`,
    `g0`, `osc`, `osc_unit`, `a`, `b`, `c`, `A`, `B`, `C`, `expr`, `eps`); every angular
    frequency is a [value, unit] pair and `N` is [value, 'cm-3'|'m-3']. The TLU family takes
    `A_eV`, `E0_eV`, `C_eV`, `Eg_eV`, `Ec_eV` (numbers in eV, the unit in the name) and
    `eps1_inf`. The Forouhi-Bloomer family takes `A`, `B_eV`, `C_eV2`, `n_inf`, `Eg_eV`.
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    lam_um = lam_nm / 1000.0
    w = 2 * np.pi * C / (lam_nm * 1e-9)
    w_ev = w * HBAR_EV_S

    if model == "drude":
        wp = _quantity(params, "wp")
        g = _quantity(params, "gamma")
        eps = _number(params, "eps_inf") - wp ** 2 / (w ** 2 + 1j * g * w)
        n = nk_from_eps(eps)
    elif model == "drude-n":
        pair = params.get("N")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ModelError(f"parameter 'N' must be [value, unit], got {pair!r}")
        val, unit = pair
        if str(unit).lower() in ("cm-3", "cm^-3", "/cm3"):
            N = float(val) * 1e6
        elif str(unit).lower() in ("m-3", "m^-3", "/m3"):
            N = float(val)
        else:
            raise ModelError(f"unknown carrier-density unit {unit!r}: use cm-3 or m-3")
        wp = np.sqrt(N * E ** 2 / (_number(params, "mstar") * ME * EPS0))
        g = _quantity(params, "gamma")
        eps = _number(params, "eps_inf") - wp ** 2 / (w ** 2 + 1j * g * w)
        n = nk_from_eps(eps)
    elif model == "lorentz-drude":
        wp = _quantity(params, "wp")
        g0 = _quantity(params, "g0")
        eps_inf = params.get("eps_inf")
        eps = (float(eps_inf) if eps_inf is not None else 1.0) \
            - _number(params, "f0") * wp ** 2 / (w ** 2 + 1j * g0 * w)
        unit = params.get("osc_unit", "eV")
        for o in params.get("osc") or []:
            f, wj, gj = _floats(o)
            wj = to_rad_s(wj, unit)
            gj = to_rad_s(gj, unit)
            eps = eps + f * wp ** 2 / (wj ** 2 - w ** 2 - 1j * gj * w)
        n = nk_from_eps(eps)
    elif model == "sellmeier1":
        n = np.sqrt(_number(params, "a") + _number(params, "b") * lam_um ** 2
                    / (lam_um ** 2 - _number(params, "c") ** 2) + 0j)
    elif model == "sellmeier":
        Bs = _floats(params["B"])
        Cs = _floats(params["C"])
        n2 = 1 + sum(b * lam_um ** 2 / (lam_um ** 2 - c) for b, c in zip(Bs, Cs))
        n = np.sqrt(n2 + 0j)
    elif model == "cauchy":
        n = (_number(params, "A") + float(params.get("B") or 0) / lam_um ** 2
             + float(params.get("C") or 0) / lam_um ** 4) + 0j
    elif model == "expr":
        expr = params.get("expr")
        if not expr:
            raise ModelError("parameter 'expr' is required")
        # One namespace, no builtins: a lambda inside a printed expression resolves its
        # free names as globals, so the variables live there with numpy.
        val = eval(expr, {"__builtins__": {}, "np": np, "lam_um": lam_um, "lam_nm": lam_nm,
                          "w_ev": w_ev, "w": w})
        val = np.asarray(val, dtype=complex) * np.ones(len(lam_nm))
        n = nk_from_eps(val) if params.get("eps") else val
        n = np.where(n.imag < 0, np.conj(n), n)
    elif model == "tauc-lorentz-urbach":
        n = nk_from_eps(tlu_eps(params, w_ev))
    elif model == "forouhi-bloomer":
        n = fb_nk(params, w_ev)
    else:
        raise ModelError(f"unknown model {model!r}: use one of {', '.join(MODELS)}")
    return np.asarray(n, dtype=complex)


def effective_medium(eps_inclusion, rule, fraction, eps_host=1.0):
    """A stated filling factor through an effective-medium rule, on permittivities.

    Maxwell-Garnett with the material as the inclusion in `eps_host` at volume `fraction`;
    Bruggeman for a two-phase symmetric mixture. Both are the reviewers' probe forms.
    """
    ei = np.asarray(eps_inclusion, dtype=complex)
    f = float(fraction)
    h = complex(eps_host)
    rule = str(rule).strip().lower()
    if rule in ("maxwell-garnett", "mg"):
        return h * (ei + 2 * h + 2 * f * (ei - h)) / (ei + 2 * h - f * (ei - h))
    if rule in ("bruggeman", "bg"):
        B = (3 * f - 1) * ei + (2 - 3 * f) * h
        root = np.sqrt(B ** 2 + 8 * ei * h)
        r1, r2 = (B + root) / 4, (B - root) / 4
        # The physical root is passive and, where both are, the one continuous with
        # the volume average -- which is the material itself at full fill.
        average = f * ei + (1 - f) * h
        nearer = np.where(np.abs(r1 - average) <= np.abs(r2 - average), r1, r2)
        return np.where(r1.imag < -1e-12, r2, np.where(r2.imag < -1e-12, r1, nearer))
    raise ModelError(f"unknown effective-medium rule {rule!r}: use maxwell-garnett or bruggeman")


def evaluate_spec(spec, lam_nm):
    """A side-table model (`nk_model_bindings.json`): family, parameters and stated extras.

    `effective_medium` applies a stated filling factor, and `k` is the extinction the paper
    states it computed with, held constant over the model's n. `k` is applied AFTER the mix,
    so it is the effective layer's; an extinction the paper states for the MATERIAL being
    mixed goes in `effective_medium.inclusion_k`, applied before the mix, so it is diluted
    by the rule like the rest of the material's permittivity (`10.1364/ome.9.004009`'s
    13.5 nm roughness layer: the paper states TiO2's k = 1.6e-3, not the layer's).
    """
    n = evaluate(spec.get("model"), spec.get("parameters") or {}, lam_nm)
    medium = spec.get("effective_medium")
    if medium:
        if medium.get("inclusion_k") is not None:
            n = np.asarray(n, dtype=complex).real + 1j * float(medium["inclusion_k"])
        n = nk_from_eps(effective_medium(n ** 2, medium.get("rule"), medium.get("fraction"),
                                         medium.get("host_eps", 1.0)))
    if spec.get("k") is not None:
        n = n.real + 1j * float(spec["k"])
    if not np.all(np.isfinite(n)):
        raise ModelError("model is not finite over the requested window")
    return np.asarray(n, dtype=complex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--eps-inf", type=float)
    ap.add_argument("--wp", nargs=2)
    ap.add_argument("--gamma", nargs=2)
    ap.add_argument("--N", nargs=2)
    ap.add_argument("--mstar", type=float)
    ap.add_argument("--f0", type=float)
    ap.add_argument("--g0", nargs=2)
    ap.add_argument("--osc", action="append", default=[])
    ap.add_argument("--osc-unit", default="eV")
    ap.add_argument("--a", type=float)
    ap.add_argument("--b", type=float)
    ap.add_argument("--c", type=float)
    ap.add_argument("--B")
    ap.add_argument("--C")
    ap.add_argument("--A", type=float)
    ap.add_argument("--expr")
    ap.add_argument("--eps", action="store_true", help="--expr returns permittivity, not n")
    ap.add_argument("--A-eV", dest="A_eV", type=float)
    ap.add_argument("--E0-eV", dest="E0_eV", type=float)
    ap.add_argument("--C-eV", dest="C_eV", type=float)
    ap.add_argument("--Eg-eV", dest="Eg_eV", type=float)
    ap.add_argument("--Ec-eV", dest="Ec_eV", type=float)
    ap.add_argument("--eps1-inf", dest="eps1_inf", type=float)
    ap.add_argument("--B-eV", dest="B_eV", type=float)
    ap.add_argument("--C-eV2", dest="C_eV2", type=float)
    ap.add_argument("--n-inf", dest="n_inf", type=float)
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--step", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    step = a.step or max((a.hi - a.lo) / 2000.0, 0.01)
    lam_nm = np.arange(a.lo, a.hi + step / 2, step)
    params = {"eps_inf": a.eps_inf, "wp": a.wp, "gamma": a.gamma, "N": a.N, "mstar": a.mstar,
              "f0": a.f0, "g0": a.g0, "osc": a.osc, "osc_unit": a.osc_unit, "a": a.a, "b": a.b,
              "c": a.c, "A": a.A, "B": a.B, "C": a.C, "expr": a.expr, "eps": a.eps,
              "A_eV": a.A_eV, "E0_eV": a.E0_eV, "C_eV": a.C_eV, "Eg_eV": a.Eg_eV,
              "Ec_eV": a.Ec_eV, "eps1_inf": a.eps1_inf,
              "B_eV": a.B_eV, "C_eV2": a.C_eV2, "n_inf": a.n_inf}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        n = evaluate(a.model, params, lam_nm)
    except ModelError as exc:
        sys.exit(str(exc))
    if a.model == "drude-n":
        val, unit = a.N
        N = float(val) * (1e6 if unit.lower() in ("cm-3", "cm^-3", "/cm3") else 1.0)
        wp = np.sqrt(N * E ** 2 / (a.mstar * ME * EPS0))
        print(f"wp = {wp:.4e} rad/s = {wp*HBAR_EV_S:.4f} eV", file=sys.stderr)
    if a.model == "tauc-lorentz-urbach":
        Eu, Au = tlu_urbach(params)
        print(f"Eu = {Eu*1e3:.1f} meV, Au = {Au:.4e}", file=sys.stderr)
    if a.model == "forouhi-bloomer":
        p = fb_parameters(params)
        Q = 0.5 * np.sqrt(4 * p["C_eV2"] - p["B_eV"] ** 2)
        print(f"Q = {Q:.5f} eV", file=sys.stderr)

    if not np.all(np.isfinite(n)):
        sys.exit("model is not finite over the requested window")
    with open(a.out, "w") as fh:
        fh.write(f"# audit probe table: {' '.join(sys.argv[1:])}\n")
        fh.write("wavelength_nm,n,k\n")
        for L, v in zip(lam_nm, n):
            fh.write(f"{L:.4f},{v.real:.8f},{abs(v.imag):.8f}\n")
    i = [0, len(lam_nm) // 2, len(lam_nm) - 1]
    for j in i:
        print(f"  {lam_nm[j]:.1f} nm: n={n[j].real:.4f} k={abs(n[j].imag):.4f}", file=sys.stderr)
    print(a.out)


if __name__ == "__main__":
    main()
