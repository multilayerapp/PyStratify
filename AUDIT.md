# Audit of STRATIFY (MATLAB) against its papers

**Scope.** The STRATIFY v1.1 MATLAB sources (<https://gitlab.com/iliarasskazov/stratify>) were
checked line by line against

* **[OSAC]** I. L. Rasskazov, P. S. Carney, A. Moroz, *STRATIFY: a comprehensive and versatile
  MATLAB code for a multilayered sphere*, OSA Continuum **3**, 2290 (2020); and
* **[AoP]** A. Moroz, *A recursive transfer-matrix solution for a dipole radiating inside and
  outside a stratified sphere*, Ann. Phys. **315**, 352 (2005), the theory the decay-rate code
  implements.

**Method.** Reading code against a paper only finds disagreements; it cannot say which side is
right. Every finding below was therefore decided numerically, in two independent ways:

1. **The original code was run in GNU Octave.** For each deviation, STRATIFY's actual output was
   pinned down and, where possible, the one suspect line was patched in a temporary copy to show
   the discrepancy disappears, which proves the root cause. That harness (`tests/test_octave.py`,
   `audit/impact.py` and the unmodified MATLAB sources under `reference/`) is in commit
   `8800f0d`; it was removed when the package dropped everything MATLAB-shaped, and
   `git checkout 8800f0d` restores it.
2. **Physics decides who is right**, with no reference to either implementation, and these
   checks remain in `tests/test_physics.py`: energy conservation of decay rates (the total rate
   from the Green's function must equal radiated plus absorbed power, to 1e-9, for electric and
   magnetic dipoles, μ ≠ 1, in cores, shells and the host), textbook Mie theory, the Bohren &
   Huffman reference case (Q_ext = 3.10543, Q_back = 2.92534), the optical theorem, plane-wave
   limits, interface continuity and brute-force quadrature.

**Result in one line.** The sphere solver itself is correct. That covers the transfer matrices,
cross sections, far field, near field, energy density and stored energy. **The decay-rate code
is not.** Its normalisation is swapped (M1), its absorption integrals use the wrong Bessel
function (M2), and its l-sum is truncated by a heuristic (M3). The energy prefactor for Drude
metals is also wrong (M5). In the configurations STRATIFY's own examples use, these give errors
of **8–39 % in decay rates** and **5–11× in stored energy inside metal shells**.

## Summary

| ID | File | Severity | What | Who is wrong | Evidence |
|---|---|---|---|---|---|
| **M1** | `decay/edcy.m`, `mdcy.m` | **high** | `'host'` and `'shell'` radiative normalisations are swapped | code **and** OSAC Eq. 29 **and** AoP Eqs. 126, 129–130 | Octave; `test_decay_host_vs_shell_normalisation`, `test_decay_energy_balance` |
| **M2** | `decay/I_abs.m` | **high** | absorption integrals use the **cylindrical** Hankel function `besselh(l,x)` instead of the spherical h_l | code (AoP Eq. 116 and OSAC Eq. 32 are right) | Octave (line patched); `test_decay_energy_balance` |
| **M3** | `decay/edcy.m`, `mdcy.m` | **high** near metals | when any Im(n) > 1, Γ_nrad is taken at an arbitrary partial sum; the `tol` argument is never used | code (OSAC §2.5 says the accuracy is set) | Octave; `test_decay_close_to_metal_needs_and_gets_high_orders` |
| **M4** | `decay/mdcy.m` | medium | magnetic dipole normalised with the *electric* dipole's free-space rate | code | Octave; `test_decay_energy_balance[magnetic]` |
| **M5** | `energy/G_prefac.m` | **high** for metals | Drude energy factor uses ω_p/γ instead of ω/γ, and the Drude ε instead of the shell's ε | code (OSAC Eq. 23 is right) | Octave; `test_loudon_energy_prefactor` |
| **M6** | `field/far_fld.m`, `near_fld.m` | low | θ = π: far field is NaN; near field on the −z axis takes the θ = 0 limit (wrong sign for even l) | code | Octave; `test_near_field_regular_on_both_poles_and_at_origin` |
| **M7** | `bessel/sbesselj.m` | low | `sbesselj(1:L, 0)` returns j₁(0) = 1, which enters the core absorption integral | code | Octave (line patched) |
| **M8** | `materials/Ag_P.mat` | **high** if used | the "Palik Ag" table is not silver (n ≈ 1.000, k ≈ 10⁻⁵ at 124–2066 nm) | data | inspection |
| **M9** | `util/t_mat.m` and every user of its products | medium–high | the composite matrices are products of ψ_l, ξ_l: they overflow for l ≫ x, and differences like M₁₁ + M₁₂T₂₁/T₁₁ cancel catastrophically. Interface conditions are violated by ~50 % at l = 14 inside the 3-nm Au shell of OSAC Fig. 2(b); Γ_nrad drifts with l_max | code (numerics) | Octave; `test_interface_conditions_hold_at_every_order` |
| M10 | `energy/nrg_tot.m` | low | lossy/lossless Lommel formula chosen by `imag(n) == 0`; the lossy form loses ~1e-17/Im(n) relative accuracy (2 % at Im n = 1e-14) | code (numerics) | `test_weak_loss_lommel_is_ill_conditioned_and_avoided` |
| M11 | `decay/edcy.m` | medium | convergence of the l-sums judged by the last terms only; for a series with ratio q² → 1 the neglected tail is ~1/(1−q²) times larger | code (numerics) | `test_decay_close_to_metal_needs_and_gets_high_orders` |
| T1 | OSAC Eqs. 30, 31 | typo | n_d = 1 case divides by M₂₁(1); AoP Eq. 71 and the code use M₂₂(1) | paper | energy balance |
| T2 | OSAC Eq. 31 | typo | n_a < n_d case reads M₂₂ψ + M₁₂ζ; correct (and coded) is M₁₂ψ + M₂₂ζ | paper | energy balance |

Everything not listed was verified correct: `t_mat.m`, `crs_sec.m`, `far_fld.m` (θ < π),
`near_fld.m` (off the −z axis), `nrg_dns.m`, `nrg_tot.m` (given correct G), `F_lossy.m`,
`F_lossless.m`, `el_fr_pth.m`, `l_conv.m`, the Bessel helpers apart from M7, `pln_fres.m`;
PyStratify reproduced all of these to ≤ 1e-10 in the Octave comparison. The planar `1D/` folder is
not part of the sphere solver and is not ported. For the record, reading it showed two problems:
`pln_get_abs_tmm.m` passes a wavenumber (µm⁻¹) where `pln_abcd.m` expects a wavelength, and
`pln_abcd_incoh.m`'s "incoherent" last layer only discards a phase.

## Impact on STRATIFY's own configurations

Produced by `audit/impact.py` at commit `8800f0d` (STRATIFY in Octave against PyStratify). Orientation-averaged rates (Γ⊥ + 2Γ∥)/3, host
normalisation, λ = 614 nm, Au from Johnson & Christy. The Au@SiO₂ rows are the `tst_dcy.m` /
OSAC Fig. 2(c) set-up (l = 1:100, `rin` = 200).

| configuration | STRATIFY | PyStratify | STRATIFY error |
|---|---|---|---|
| Au@SiO2 (tst_dcy), r_d=52 nm (SiO2 shell, 2 nm from Au): Γ_rad | 12.64 | 13.78 | −8.3% |
| Au@SiO2 (tst_dcy), r_d=52 nm (SiO2 shell, 2 nm from Au): Γ_nrad | 743.4 | 861.0 | −13.7% |
| Au@SiO2 (tst_dcy), r_d=60 nm (SiO2 shell, 10 nm from Au): Γ_rad | 6.152 | 6.707 | −8.3% |
| Au@SiO2 (tst_dcy), r_d=60 nm (SiO2 shell, 10 nm from Au): Γ_nrad | 8.545 | 8.589 | −0.5% |
| Au@SiO2 (tst_dcy), r_d=71 nm (water, 1 nm outside): Γ_rad | 4.328 | 4.328 | +0.0% |
| Au@SiO2 (tst_dcy), r_d=71 nm (water, 1 nm outside): Γ_nrad | 1.705 | 1.700 | +0.3% |
| SiO2@Au nanoshell {50, 55} nm, r_d=30 nm (inside the SiO2 core): Γ_rad | 2.012 | 2.194 | −8.3% |
| SiO2@Au nanoshell {50, 55} nm, r_d=30 nm (inside the SiO2 core): Γ_nrad | 16.75 | 27.49 | −39.1% |
| SiO2@Au nanoshell {50, 55} nm, r_d=60 nm (water, 5 nm outside): Γ_rad | 1.094 | 1.094 | +0.0% |
| SiO2@Au nanoshell {50, 55} nm, r_d=60 nm (water, 5 nm outside): Γ_nrad | 157.4 | 238.5 | −34.0% |
| SiO2@Au nanoshell {50, 55} nm, r_d=70 nm (water, 15 nm outside): Γ_rad | 0.9426 | 0.9426 | +0.0% |
| SiO2@Au nanoshell {50, 55} nm, r_d=70 nm (water, 15 nm outside): Γ_nrad | 18.73 | 30.16 | −37.9% |
| G_e for Au (Ordal Drude), 500 nm | 84.2 | 14.25 | +491% |
| G_e for Au (Ordal Drude), 700 nm | 239.7 | 26.96 | +789% |
| G_e for Au (Ordal Drude), 900 nm | 520.4 | 43.91 | +1085% |

The −8.3 % radiative rows are exactly n_SiO₂/n_water − 1 (M1). The nanoshell Γ_nrad rows are
M2. The 2-nm row is M3 (plus M9/M11 at l = 100). The OSAC Fig. 2(c) benchmark against MNPBEM passed because it used an
Au **core**: there the Hankel term of M2 is multiplied by B = 0, and M1 only affects emitters
inside a shell.

## Details

### M1 — radiative normalisation swapped (code, OSAC Eq. 29, AoP Eqs. 126/129)

For an emitter in shell d, the rate normalised to the free-space rate *in the same medium* is
fixed by the Green's function (LDOS) at the emitter, with no normalisation choice involved. For a
lossless sphere it must equal Γ_rad. It does, to 1e-12, for **STRATIFY's `'host'` output**.
STRATIFY's `'shell'` output equals the LDOS value multiplied by n_dμ_d/(n_hμ_h), which is the
host-normalised rate. The two labels are exchanged.

The physics is simple. A dipole's free-space rate scales as Γ₀ ∝ nμ (electric dipole, fixed
moment). So Γ/Γ₀,host = (Γ/Γ₀,shell) · n_dμ_d/(n_hμ_h), and the shell- and host-normalised
rates differ by that factor. AoP Eq. 130 states the ratio the other way round (|n|/n_h larger
for *shell*), and OSAC Eq. 29 copies it. AoP's own *nonradiative* factors (Eqs. 132–134) have
the correct ratio, so the paper is internally inconsistent. Correct factors, for any μ:

    N_rad^shell  = n_d μ_d / (n_h μ_h)             (OSAC's N_rad^host)
    N_rad^host   = [n_d μ_d / (n_h μ_h)]^2         (OSAC's N_rad^shell)
    N_nrad^shell = μ_d / n_d^2,   N_nrad^host = μ_d^2 / (n_d n_h μ_h)   (as published)

Only emitters **inside** the sphere are affected; for an emitter in the host both factors are 1.
In the Au@SiO₂ example the host-normalised Γ_rad inside the silica is 8.3 % too low. Any
quantum yield built from it (OSAC Eq. 38) is correspondingly off.

### M2 — cylindrical Hankel function in the absorption integrals (`I_abs.m`)

```matlab
Sx(:,i)   =  sbesselj(l, xa(i));   % spherical j_l      (correct)
xix(:,i)  =  besselh(l, xa(i));    % CYLINDRICAL H_l^(1) (should be sbesselh)
dxix(:,i) = dbesselh(l, xa(i));    % CYLINDRICAL        (should be dsbesselh)
```

AoP Eq. 116 and OSAC Eq. 32 use the spherical h_l⁽¹⁾. The error vanishes when the absorbing
shell is the core (B = 0), which is the case the paper benchmarked. For every absorbing *shell*
(nanoshells, matryoshkas) Γ_nrad is wrong: 34–39 % low in the nanoshell rows above, and up to
2× in `test_physics` configurations. Replacing the two calls with `sbesselh`/`dsbesselh` makes
STRATIFY agree with PyStratify to trapezoid accuracy (Octave (line patched); `test_decay_energy_balance`).

### M3 — ad-hoc truncation of the nonradiative sum (`edcy.m`, `mdcy.m`)

```matlab
if max (imag(ref),[],'all') > 1
    gnr1 = gnrcum(diff(gnrcum(:,1),1,1)./gnrcum(2:end,1)<1e-2,1);
    gnr(i,:) = [gnr1(ceil(end/10)) ...];
```

If any shell has Im(n) > 1 (every noble metal in the visible), the result is the partial sum at
index ⌈K/10⌉ among the K partial sums whose relative increment is below 1 %. That value
depends on the `l` the caller passes, and it is not converged. Near a metal surface the series
converges slowly, so Γ_nrad comes out too low (−10.6 % at 2 nm from the Au core in the
`tst_dcy.m` set-up). The documented `tol` argument is never read. PyStratify sums until the last
terms leave an estimated remainder below `tol`·|sum| (M11) and flags positions that do not converge.

### M4 — magnetic-dipole host normalisation (`mdcy.m`)

`mdcy.m` reuses `edcy.m`'s normalisation factors. The shell normalisation happens to be the same
for both dipole types, so, as with M1, `mdcy.m`'s `'host'` output is really the shell-normalised
rate. But a magnetic dipole's free-space rate scales as nε, not nμ (∝ n³ for non-magnetic media).
Its host-normalised rate is therefore (shell rate)·n_dε_d/(n_hε_h), while `mdcy.m`'s `'shell'`
output uses (shell rate)·n_dμ_d/(n_hμ_h), off by n_d²/n_h² for non-magnetic media. `mdcy.m` also
inherits M2, M3 and M9. With the correct factors, PyStratify's magnetic-dipole rates satisfy
energy balance to 1e-9.

### M5 — energy prefactor for Drude metals (`G_prefac.m`)

OSAC Eq. 23 (Loudon) is G_e = Re ε + 2(ω/γ) Im ε. For a Drude ε this gives
1 + ω_p²/(ω² + γ²). The code computes

```matlab
G.e(i) = real(eps) + 2*imag(eps)*lam_g/lam_p;   % lam_g/lam_p = omega_p/gamma, should be lam_g/lam = omega/gamma
```

That is 5–11× too large for gold across 500–900 nm (table above), so the electric energy stored
in any metal shell is overstated by that factor. G_prefac also evaluates ε from the Drude fit
rather than the shell's actual (e.g. tabulated) ε. PyStratify's `electric_prefactor` implements
Eq. 23 with the shell's ε; the Loudon identity is a unit test.

### M6 — θ = π

`far_fld.m` divides P¹_l(cos θ) by sin θ and patches only θ = 0. At θ = π (backscattering) the
result is NaN. `near_fld.m` patches both poles with the θ = 0 limit −l(l+1)/2. On the −z axis
the correct value is (−1)^l l(l+1)/2, so fields exactly on the −z axis are wrong (they jump
relative to a point 1e-12 m off the axis). PyStratify combines the m = ±1 harmonics into the
Bohren–Huffman π_l, τ_l, which are regular at both poles.

### M7 — `sbesselj(nu, 0)`

`j = zeros(1,numel(nu)); j(1,1) = 1;` sets the *first requested order* to 1. `I_abs.m` calls it
with `l = 1:L`, so j₁(0) = 1 at the first quadrature node of the core integral. That is a
~3e-4 relative error in Γ_nrad for an absorbing core in the tests. Patching it makes STRATIFY
agree with PyStratify to 1e-7.

### M8 — `materials/Ag_P.mat`

n = 0.997–1.003 and k = 1.3e-6 … 2.7e-3 over 124–2066 nm is not silver: silver is a metal with
k ≈ 2–14 from 400 to 2000 nm (compare `Ag_JC.mat` in the same folder). PyStratify ships no
tabulated optical constants at all; n, k come from the caller (e.g. refractiveindex.info).

### M9 — direct transfer-matrix products (`t_mat.m` and its users)

The composite matrices T(n), M(n) are products of Riccati–Bessel functions at different
arguments, with entries ~(x̃/x)^l. Two consequences:

* Differences of large products cancel. Two examples: the absorbing-shell coefficient
  M₁₁ + M₁₂·T₂₁/T₁₁ of OSAC Eq. 33, and the internal plane-wave coefficients of Eqs. 25–26.
  Inside the 3-nm Au shell of the OSAC Fig. 2(b) matryoshka, the interface conditions are
  already violated by 30–80 % at l = 13–14. For an emitter 5 nm outside a lossy shell, Γ_nrad
  changes by ~1e-3 between l_max = 50 and 70 even after M2 is fixed, and `tst_dcy.m` uses
  l = 1:100.
* They overflow for l ≫ x, which is why STRATIFY cannot reach the l ~ 500–1000 needed for
  emitters within a few nm of a metal (Majic & Le Ru 2020).

PyStratify's scaled formulation (README, "Numerics") satisfies the interface conditions to 1e-9
at every order tested, including a 2-nm shell with |n| = 40 at l = 400, and gives decay rates
that conserve energy to 1e-11 up to l = 1200.

### M10 — weakly lossy shells in `nrg_tot.m`

Eq. 27's lossy form divides Im(x f f*) by x² − x*², and both vanish as Im k → 0. With the switch
at `imag(n) == 0` the relative error is ~1e-17/Im(n): 1e-8 at Im n = 1e-9, 2 % at 1e-14.
PyStratify integrates shells with 0 < |Im k| < 1e-3|k| by Gauss–Legendre quadrature of the
energy density.

### M11 — convergence test for the decay-rate series

For an emitter at distance δ from an interface of radius a, the terms decay like
(a/(a+δ))^{2l}·l³. Stopping when the last term is small relative to the sum leaves a tail about
1/(1 − q²) times larger, i.e. ×35 at 1 nm from a 70-nm sphere. PyStratify's test uses the
estimated remainder t_L·r/(1 − r).

### Minor, by reading only

* `near_fld.m` computes Hankel functions only for `ru > r1`. A point exactly at r = r₁, which
  belongs to shell 2, gets h_l = 0. PyStratify computes them for every point outside the core.
* `nrg_dns.m` assigns a point exactly on an interface to the inner shell; PyStratify assigns it to
  the outer one, as `near_fld.m` does. |E|² jumps there, so values on interfaces differ.
* `G_prefac.m` returns scalars per material and fails for vector `lam`.

## Limits

* Emitters within ~0.5 nm of a metal need more than the default cap of 1200 multipoles for
  tol = 1e-8. PyStratify then reports the position as not converged (`DecayRates.converged`,
  `notes`) and does not return a silently truncated value. At that distance classical
  electrodynamics itself is questionable (nonlocality).
* The magnetic-dipole results (M4) are validated by energy balance only; OSAC does not publish
  them.
* Running STRATIFY in Octave needed small shims for `rmmissing` and `max(...,'all')` (at
  `8800f0d`, `tests/octave_compat/`); they reproduce MATLAB semantics for the arrays used.
* The papers Majic & Le Ru (2020), Zhang (2025) and Ladutenko et al. (2017) were consulted
  through their abstracts. The formulation here is derived independently and verified
  numerically, not transcribed from them.
