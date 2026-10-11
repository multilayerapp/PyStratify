"""K. A. Neyts, "Simulation of light emission from thin-film microcavities", J. Opt. Soc. Am. A 15, 962
(1998), doi:10.1364/JOSAA.15.000962: Figs. 9-10 -- the emission of a randomly oriented dipole inside a ZnS
layer on an aluminium mirror, against layer thickness and dipole position: the fraction of the radiated
power that escapes into air (E_appl) and the total decay rate F.  The calculation behind thin-film
electroluminescent and OLED outcoupling design.

Inputs (Fig. 7 caption, Sec. 4): air (n = 1) / ZnS (n_e = 2.3, lossless) / Al (n = 1 + 6i), "roughly valid
near 600 nm"; isotropic electric dipole with quantum yield 1.  The calculator's film exteriors must be
lossless, so Al is a 200 nm layer (opaque: T < 1e-15) on air.  One column of the Fig. 10 map is one
calculator run: here the optimum thickness n_e d_e / λ = 0.73 (d_e = 190.4 nm at λ = 600 nm), the dipole
swept across the layer; z_ is measured from the mirror.  Printed checks (p. 970): maximum E_appl = 0.38
with the dipole at the air side and n_e d_e / λ = 0.73, where F = 0.817.
"""

import numpy as np
from _common import peak, report, save_csv

import pystratify as ps

LAM, N_E, AL = 0.600, 2.3, 1.0 + 6.0j
D_E = 0.73 * LAM / N_E
z_mirror = np.linspace(0.002, D_E - 0.002, 120)  # dipole distance from the Al mirror


def run(d_e, z_from_mirror):
    # incident air (layer 0) / ZnS (layer 1) / Al (layer 2) / air; depth in ZnS measured from the air side
    p = ps.Problem("films", [np.inf, d_e, 0.2, np.inf], [1.0, N_E, AL, 1.0], LAM,
                   ps.PointDipole(d_e - z_from_mirror, "electric", layer=1))
    r = ps.solve_problem(p)
    return r["upper"][2] / r["total"][2], r["total"][2]  # isotropic: escape to the air side, total rate


res = np.array([run(D_E, z) for z in z_mirror])
e_appl, total = res[:, 0], res[:, 1]
save_csv("neyts_1998_fig10_column", "z_from_mirror_um,E_appl_isotropic,F_isotropic", [z_mirror, e_appl, total])
# Fig. 10 row at the air edge: E_appl against thickness, dipole 2 nm below the air interface.
thick = np.linspace(0.25, 2.0, 71) * LAM / N_E
row = np.array([run(t, t - 0.002)[0] for t in thick])
save_csv("neyts_1998_fig10_row", "ne_de_over_lambda,E_appl_isotropic", [thick * N_E / LAM, row])
first, second = peak(thick * N_E / LAM, row, 0.4, 0.95)[0], peak(thick * N_E / LAM, row, 0.95, 1.45)[0]
ok = report("neyts_1998", [
    ("maximum E_appl (dipole at the air side)", 0.38, e_appl[-1], 0.02, "Fig. 11 caption, p. 970"),
    ("F at that point", 0.817, total[-1], 0.03, "text p. 970"),
    ("optimum thickness n_e d_e / λ (dipole at the air side)", 0.73, first, 0.03, "p. 970, Fig. 11 caption"),
    ("next thickness maximum, n_e d_e / λ (half a wave later)", 1.23, second, None, "the (0.19 + 0.5 l) rule of p. 969 is for Fig. 7's normal direction"),
    ("F near the mirror (z_ = 5 nm) is large", 5.0, total[np.argmin(abs(z_mirror - 0.005))], None, "Fig. 9: contours 2-5 near Al"),
])
try:
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(z_mirror / D_E, e_appl, label="E_appl (escape to air)")
    a.plot(z_mirror / D_E, total, label="F (total rate)")
    a.set_ylim(0, 2)
    a.set_xlabel("z_ / d_e (from the mirror)")
    a.legend(fontsize=8)
    a.set_title("n_e d_e / λ = 0.73", fontsize=9)
    b.plot(thick * N_E / LAM, row)
    b.set_xlabel("n_e d_e / λ")
    b.set_ylabel("E_appl, dipole at the air side")
    fig.suptitle("Neyts 1998, Figs. 9-10, recomputed: air / ZnS / Al at 600 nm", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
