# Classic papers, recomputed

Fifteen classic papers on planar, cylindrical and spherical structures, each recomputed from the inputs
the paper prints. Every script writes a CSV, a PNG and `results/<paper>.json`. That JSON records each
check against a number printed in, or read off, the paper, with its tolerance and source. The checks come
from the multilayer.app classics page (`multilayer_web/classics.html`), which opens the same inputs live in
the calculator. That page's generator, `multilayer_web/tools/classics/build.py`, copies its agreement
lines from here.

Optical constants come from the refractiveindex.info database at the commit the calculator pins
(`a66ef88`, copied into `data/`, public domain). They are interpolated linearly, as the backend does. Where
a paper prints its own constants (Mie, Anger, Ford & Weber, Neyts, Katsidis, Ruan & Fan, Ruppin 2001),
those are used instead. Where it prints none, the substitute is named in the script's docstring.

Run any script from this folder, e.g. `python mie_1908.py`, with one BLAS thread on a small machine.
`results/` is generated (git-ignored, like `crosscheck/results/`).

| script | paper | figure | checks passed | notes |
|---|---|---|---|---|
| `mie_1908.py` | Mie, Ann. Phys. 330, 377 (1908) | Fig. 25 and table, p. 438 | 56 / 56 (within 10%; 51 within 5%) | His seven-point gold table. For diameters ≥ 140 nm, his truncation a₁ + a₂ + p₁ (Eq. 100) |
| `yeh_1977.py` | Yeh, Yariv & Hong, JOSA 67, 423 | Figs. 7–8 | 4 / 4 | The figures print no indices. GaAs/AlGaAs from the paper's Fig. 12; angles in GaAs |
| `lukosz_kunz_1977.py` | Lukosz & Kunz, JOSA 67, 1607 | Fig. 4 | 8 / 8 | n1 = 1, λ0 = 600 nm |
| `ruppin_1982.py` | Ruppin, J. Chem. Phys. 76, 1681 | Figs. 1–3 | 7 / 7 | Silver data not printed. Johnson & Christy puts the l = 1 plasmon at the paper's 3.05 eV |
| `ford_weber_1984.py` | Ford & Weber, Phys. Rep. 113, 195 | Fig. 7 | 5 / 5, one unresolved | Perpendicular photon-fraction dip: 0.11 here, 0.22 in Fig. 7. **Kept off the classics page** (ruled by Ilia). See below |
| `neyts_1998.py` | Neyts, JOSA A 15, 962 | Figs. 9–10 | 3 / 3 | Al is a 200 nm opaque layer, because film-emitter exteriors must be lossless |
| `katsidis_2002.py` | Katsidis & Siapkas, Appl. Opt. 41, 3978 | Fig. 2 | 6 / 6 | Smooth curves only (no roughness model). Si n = 3.42 |
| `prodan_2003.py` | Prodan et al., Science 302, 419 | Fig. 4A | 5 / 5 | No optical constants or host printed. Host n = 1.45 is the best fit; the bonding peak sits 85 nm short at every host |
| `mohlenhoff_2005.py` | Mohlenhoff et al., Biophys. J. 88, 3635 | Fig. 4 | 3 / 3 | Fig. 4 is van de Hulst's formula, reproduced. Exact Mie peaks higher (4.0 vs 3.17) |
| `anger_2006.py` | Anger, Bharadwaj & Novotny, PRL 96, 113002 | Fig. 1 | 12 / 12 | Printed gold ε. Fluorescence = plane-wave \|E_r\|² × quantum yield |
| `ruan_fan_2010.py` | Ruan & Fan, PRL 105, 013901 | Figs. 1, 4a | 5 / 5 | Lossy case exact (1.917, 0.731). Lossless peak 7.87 vs 7.94, on a fine grid |
| `ruppin_2001.py` | Ruppin, Opt. Commun. 190, 205 | Figs. 1, 4 (dashed) | 5 / 5 | Local (classical) curves only |
| `davis_2011.py` | Davis, Carney & Bhargava, Anal. Chem. 83, 525 | Fig. 5 | 6 / 6 | Fig. 5's NA is not stated. NA 0.4 / obscuration 0.1 (its Fig. 2) reproduces it |
| `kuznetsov_2012.py` | Kuznetsov et al., Sci. Rep. 2, 492 | Fig. 3(iv) | 10 / 10 | All md/ed/mq peaks within 5 nm. Green 2008 silicon |
| `fu_2013.py` | Fu et al., Nat. Commun. 4, 1527 | Fig. 1a | 7 / 7 | F/B 7.5 at 659 nm; F = B at 499 and 604 nm |

Read values carry the reading error stated in each script (typically ±3–5% of the axis). Several brief
readings were corrected against the page images at 300 dpi (Anger Fig. 1a, Lukosz & Kunz Fig. 4,
Davis Fig. 5a).

The calculator links compute the same numbers. On 2026-10-10 the production API returned
Q_ext = 7.917043619519 at 600 nm for the Kuznetsov 150 nm sphere, and total rates
1096.6776603714 / 1.0300025080 for the Ruppin dipole at 1 and 55 nm. Both are identical to these scripts.

## Unresolved: Ford & Weber 1984, Fig. 7

The checks pass for the photon fraction at 400 Å (0.31 vs 0.35), at 1 µm (0.96 vs 0.97), for the
parallel dipole's crossover (198 vs ~165 Å) and for full quenching at 20 Å. One check does not: near
1800 Å the perpendicular photon fraction dips to 0.11 here, but to 0.22 in Fig. 7 (PDF p. 39, journal
p. 233).

Fig. 6 (PDF p. 38, journal p. 232) plots the perpendicular photon rate dipping to about 0.3 there. A
perpendicular dipole above any good mirror radiates about 1.16 at that height (image-dipole estimate, and
this script's escape rate). Their surface-plasmon rate is about 2.3 times smaller than the local Drude
model gives, at a height where nonlocality cannot matter.

PyStratify's film-emitter rates are pinned to independent Sommerfeld integrals and the image-dipole
closed form (`tests/test_planar_emission_references.py`). **Ruling (Ilia, 2026-10-10):** a factor of two is
an easy slip on the paper's side, but not one to assert. Keep the paper off the classics page; the
script stays here with the discrepancy recorded.
