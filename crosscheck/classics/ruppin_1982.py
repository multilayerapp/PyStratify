"""R. Ruppin, "Decay of an excited molecule near a small metal sphere", J. Chem. Phys. 76, 1681 (1982),
doi:10.1063/1.443196: Figs. 1-2 -- the lifetime of a molecule near a silver sphere of radius 10 nm,
against its distance from the surface, for a radial (Fig. 1) and a tangential (Fig. 2) dipole; and
Fig. 3 -- the total decay rate against photon energy at 2, 5 and 10 nm, with the dipole plasmon at
~3.05 eV.  The first exact calculation of fluorescence lifetimes beside a small metal particle.

Inputs: Ag sphere, R = 10 nm, in a host of ε = 2.25 (n = 1.5), "as Chance et al. did" (p. 1683); emission
wavelength 612 nm (Figs. 1-2); intrinsic quantum yield 1 (the lifetime is the total rate of dissipation).
The paper does not print its silver data; we use Johnson & Christy, whose l = 1 plasmon in n = 1.5 lands
at the paper's 3.05 eV.  Distances are measured from the surface; the paper's d^ = k d with k = 2πn/λ.
Curve values are read off Figs. 1-3 (±0.05).
"""

import numpy as np
from _common import material, peak, report, save_csv

import pystratify as ps

R, N_HOST, LAM = 0.010, 1.5, 0.612
k = 2 * np.pi * N_HOST / LAM
gap = np.linspace(0.0005, 0.270, 201)  # µm, from the surface
rates = ps.decay_rates([R], [material("Ag_Johnson", [LAM])[0], N_HOST], LAM, R + gap)
lifetime = 1 / rates.total  # [:, 0] radial, [:, 1] tangential
dhat = k * gap
save_csv("ruppin_1982_fig1_2", "gap_um,dhat,lifetime_radial,lifetime_tangential", [gap, dhat, lifetime[:, 0], lifetime[:, 1]])


def at(x):
    return np.interp(x, dhat, lifetime[:, 0]), np.interp(x, dhat, lifetime[:, 1])


# Fig. 3: total rate vs photon energy, radial dipole 2, 5, 10 nm from the surface.
energy = np.linspace(2.5, 3.8, 131)
lam3 = 1.23984 / energy
ag = material("Ag_Johnson", lam3)
fig3 = {d: np.array([ps.decay_rates([R], [n_ag, N_HOST], l, [R + d * 1e-3]).total[0, 0] for n_ag, l in zip(ag, lam3)]) for d in (2, 5, 10)}
save_csv("ruppin_1982_fig3", "energy_eV," + ",".join(f"total_radial_{d}nm" for d in fig3), [energy, *fig3.values()])
e_peak = peak(energy, fig3[10], 2.8, 3.3)[0]

tan_max_x, tan_max = peak(dhat, lifetime[:, 1], 0.05, 0.6)
ok = report("ruppin_1982", [
    ("radial: lifetime at d^ = 0.5", 0.86, at(0.5)[0], 0.06, "Fig. 1, read"),
    ("radial: lifetime at d^ = 1", 0.98, at(1.0)[0], 0.04, "Fig. 1, read"),
    ("radial: lifetime at d^ = 3 (no oscillation)", 1.00, at(3.0)[0], 0.02, "Fig. 1, read; p. 1683"),
    ("tangential: maximum lifetime", 1.12, tan_max, 0.06, "Fig. 2, read"),
    ("tangential: d^ of the maximum", 0.2, tan_max_x, 0.1, "Fig. 2, read"),
    ("Fig. 3: l = 1 plasmon, 10 nm (eV)", 3.05, e_peak, 0.08, "p. 1684: 'at about 3.05 eV'"),
    ("Fig. 3: total rate at the plasmon, 10 nm (order of magnitude, log10)", 2.0, np.log10(fig3[10].max()), 0.5, "Fig. 3, read ~1e2"),
])
try:
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(dhat, lifetime[:, 0], label="radial (Fig. 1)")
    a.plot(dhat, lifetime[:, 1], label="tangential (Fig. 2)")
    a.axhline(1, color="0.7", lw=0.6)
    a.set_xlabel("d^ = k d")
    a.set_ylabel("lifetime / lifetime without sphere")
    a.legend(fontsize=8)
    for d, y in fig3.items():
        b.semilogy(energy, y, label=f"{d} nm")
    b.set_xlabel("photon energy (eV)")
    b.set_ylabel("total rate / homogeneous rate (radial)")
    b.legend(fontsize=8)
    fig.suptitle("Ruppin 1982, Figs. 1-3, recomputed: Ag sphere R = 10 nm in n = 1.5", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
