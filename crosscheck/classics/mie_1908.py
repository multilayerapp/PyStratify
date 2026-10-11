"""G. Mie, "Beiträge zur Optik trüber Medien, speziell kolloidaler Metallösungen", Ann. Phys. 330,
377-445 (1908), doi:10.1002/andp.19083300302: the table and Fig. 25 on p. 438 -- the light loss
("Absorption", i.e. extinction) of colloidal gold in water against wavelength for particle diameters
0-180 nm, the calculation that explained the colours of gold sols.

Inputs, all printed by Mie: gold n, k at seven wavelengths (his table on p. 417, the "assumed" columns,
smoothed from Hagen & Rubens), water as the host (n_w = λ/λ', from his λ' column), dilute suspension of
C = 1e-6 gold by volume over 1 mm.  His loss per mille is K = 1000 C C_ext / V = 1500 Q_ext / d (d in nm),
so K is the extinction efficiency rescaled; the calculator shows Q_ext of one sphere.  For d >= 140 nm Mie
kept only the first terms of the series, so those rows are compared with
his truncation, a1 + a2 + p1 (Eq. 100: electric dipole and quadrupole, magnetic dipole).
"""

import numpy as np
from _common import nk_rows, report, save_csv

import pystratify as ps

# Mie's table, p. 417: λ (nm), n, k ("assumed" columns; m = n - i nκ in his e^{+iωt} convention).
MIE_GOLD = np.array([
    [420, 1.70, 1.70], [450, 1.73, 1.72], [500, 1.10, 2.02], [525, 0.79, 2.23],
    [550, 0.57, 2.45], [600, 0.38, 2.96], [650, 0.41, 3.54],
])
LAM_PRIMED = np.array([313, 336, 374, 393, 412, 450, 487])  # his λ' (wavelength in water), nm
N_WATER = float(np.mean(MIE_GOLD[:, 0] / LAM_PRIMED))  # 1.337; the seven values span 1.333-1.342
# Extinction K (per mille), p. 438.  Rows: diameter (nm); columns: the seven wavelengths.
TABLE_K = {
    20: [42.7, 38.8, 62.2, 76.6, 61.6, 18.5, 8.0],
    40: None,
    60: [48.8, 45.4, 67.4, 88.4, 96.1, 37.3, 16.4],
    80: [46.5, 44.8, 62.5, 81.0, 97.0, 65.0, 27.1],
    100: [41.0, 41.0, 52.2, 66.3, 82.2, 85.0, 44.5],
    120: [34.6, 34.8, 42.8, 51.2, 61.0, 77.5, 58.7],
    140: [29.4, 29.7, 35.9, 41.9, 48.6, 57.0, 56.1],
    160: [25.7, 26.1, 30.8, 36.0, 41.2, 45.1, 45.6],
    180: [24.0, 24.8, 28.0, 33.0, 37.0, 37.7, 36.5],
}

lam = MIE_GOLD[:, 0] * 1e-3
gold = MIE_GOLD[:, 1] + 1j * MIE_GOLD[:, 2]
N = np.column_stack([gold, np.full(lam.size, N_WATER)])

computed, checks = {}, []
for d, row in TABLE_K.items():
    if row is None:
        continue
    cs = ps.cross_sections(ps.solve([d * 1e-3 / 2], N, lam))
    geometric = np.pi * (d * 1e-3 / 2) ** 2
    # Mie's Eq. (100) keeps a1, a2 and p1: electric dipole and quadrupole, magnetic dipole.
    q_mie = (cs.ext_by_order[:, ps.TM, 0] + cs.ext_by_order[:, ps.TM, 1] + cs.ext_by_order[:, ps.TE, 0]) / geometric
    q = q_mie if d >= 140 else cs.q_ext
    computed[d] = 1500 * q / d
    for wl, paper, ours in zip(MIE_GOLD[:, 0], row, computed[d]):
        checks.append((f"d = {d} nm, {wl:.0f} nm: K (per mille){' [a1, a2, p1]' if d >= 140 else ''}",
                       paper, ours, round(max(1.5, 0.10 * paper), 2), "table p. 438"))

save_csv("mie_1908_table_p438", "wavelength_nm," + ",".join(f"K d={d}nm" for d in computed), [MIE_GOLD[:, 0], *computed.values()])
# The table the calculator link uploads: Mie's seven rows, verbatim.
(__import__("pathlib").Path(__file__).parent / "results" / "mie_1908_gold_nk.json").write_text(
    __import__("json").dumps(nk_rows(lam, gold)))
ok = report("mie_1908", checks)
within5 = sum(abs(o - p) <= max(1.5, 0.05 * p) for _, p, o, _, _ in checks)
print(f"{within5} of {len(checks)} printed values within 5% (all toleranced at 10%: a 1908 hand calculation)")

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for d, row in TABLE_K.items():
        if row is None:
            continue
        line, = ax.plot(MIE_GOLD[:, 0], computed[d], "-", lw=1, label=f"{d} nm")
        ax.plot(MIE_GOLD[:, 0], row, "o", ms=3, color=line.get_color())
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("light loss K (per mille per mm, C = 1e-6)")
    ax.set_title("Mie 1908, p. 438: lines PyStratify, dots Mie's table", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
