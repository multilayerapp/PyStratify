"""Katsidis & Siapkas, "General transfer-matrix method for optical multilayer systems with coherent,
partially coherent, and incoherent interference", Appl. Opt. 41, 3978 (2002), doi:10.1364/AO.41.003978:
Fig. 2 -- the infrared reflectance of a 2 µm SiC film on a 300 µm silicon wafer, the wafer treated as a
finite incoherent layer (light returns from its back face) or as a semi-infinite substrate.

Inputs: SiC from the paper's single-oscillator model (p. 3983): ε∞ = 6.7, Δε = 3.3, ω_TO = 796.5 cm-1,
γ = 20 cm-1; Si transparent, n = 3.42 (ε∞ = 11.7, quoted on p. 3983); air on both sides; normal incidence.
The paper's roughness curves (Z = 1000 Å) have no calculator counterpart; the smooth curves are compared.
Values read off Fig. 2 (±0.03).
"""

import numpy as np
from _common import curve, figure, lorentz_nk, mark, nk_rows, panel, png, report, save_csv

import pystratify.planar as pl

WN = np.linspace(700, 7000, 1261)
LAM = 1e4 / WN
SIC = lorentz_nk(WN, 6.7, 3.3, 796.5, 20.0)
finite = np.array([pl.inc_tmm("s", [1, n, 3.42, 1], [np.inf, 2.0, 300.0, np.inf], ["i", "c", "i", "i"], 0, l)["R"] for n, l in zip(SIC, LAM)])
semi = np.array([pl.coh_tmm("s", [1, n, 3.42], [np.inf, 2.0, np.inf], 0, l)["R"] for n, l in zip(SIC, LAM)])
save_csv("katsidis_2002_fig2", "wavenumber_cm-1,R_finite_incoherent_Si,R_semi_infinite_Si", [WN, finite, semi])
# the n,k table the calculator link uploads (wavelength µm, n, k), on a coarser grid
grid = np.linspace(1e4 / 7000, 1e4 / 700, 400)
(__import__("pathlib").Path(__file__).parent / "results" / "katsidis_2002_sic_nk.json").write_text(
    __import__("json").dumps(nk_rows(grid, lorentz_nk(1e4 / grid, 6.7, 3.3, 796.5, 20.0))))

band = (WN > 2000) & (WN < 6000)
rest = (WN > 800) & (WN < 860)
w_lo = 796.5 * np.sqrt((6.7 + 3.3) / 6.7)
ok = report("katsidis_2002", [
    ("ω_LO from the model (cm-1)", 973, w_lo, 1, "ω_TO √((ε∞ + Δε) / ε∞), from the model of p. 3983"),
    ("finite wafer: R maximum, 2000-6000 cm-1", 0.50, finite[band].max(), 0.04, "Fig. 2, read"),
    ("finite wafer: R minimum, 2000-6000 cm-1", 0.33, finite[band].min(), 0.04, "Fig. 2, read"),
    ("semi-infinite: R maximum, 2000-6000 cm-1", 0.30, semi[band].max(), 0.04, "Fig. 2, read"),
    ("semi-infinite: R minimum, 2000-6000 cm-1", 0.07, semi[band].min(), 0.04, "Fig. 2, read"),
    ("the two coincide in the reststrahlen band (max |ΔR|, 800-860 cm-1)", 0.0, np.abs(finite - semi)[rest].max(), 0.01, "p. 3982"),
])
# The card's figure: R against wavenumber for the wafer as a finite incoherent layer (the calculator setup) and as a
# semi-infinite substrate; the band's printed extremes are levels across the band, the model's ω_LO a rule.
BAND = (2000, 6000)
figure("katsidis_2002", "2 µm SiC on Si, normal incidence", [
    panel("r", "Reflectance", WN, "1/λ [cm⁻¹]", "R",
          [curve("finite", "R, 300 µm Si, incoherent", finite, group="finite"),
           curve("semi", "R, semi-infinite Si", semi, group="semi")],
          [mark("ω_LO from the model (cm-1)", label="ω_LO"),
           mark("finite wafer: R maximum, 2000-6000 cm-1", series="finite", at=BAND),
           mark("finite wafer: R minimum, 2000-6000 cm-1", series="finite", at=BAND),
           mark("semi-infinite: R maximum, 2000-6000 cm-1", series="semi", at=BAND),
           mark("semi-infinite: R minimum, 2000-6000 cm-1", series="semi", at=BAND)], points=420)],
    note="The light returning from the wafer's back face lifts the fringes; in the reststrahlen band, where SiC is "
         "opaque, the two coincide.")
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(WN, finite, label="300 µm Si, incoherent (finite)")
    ax.plot(WN, semi, label="semi-infinite Si")
    ax.set_xlabel("wavenumber (cm$^{-1}$)")
    ax.set_ylabel("reflectance")
    ax.legend(fontsize=8)
    ax.set_title("Katsidis & Siapkas 2002, Fig. 2 (no roughness), recomputed: 2 µm SiC on Si", fontsize=9)
    fig.tight_layout()
    fig.savefig(png("katsidis_2002"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
