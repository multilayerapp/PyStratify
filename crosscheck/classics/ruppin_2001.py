"""R. Ruppin, "Extinction properties of thin metallic nanowires", Opt. Commun. 190, 205 (2001),
doi:10.1016/S0030-4018(01)01063-X: Figs. 1 and 4 -- the extinction of a sodium wire of radius 1-2 nm under
light polarized across the wire, on a log scale against ω/ωp.  The calculator draws the local (classical,
dashed) curve; the paper's solid curve adds the hydrodynamic (nonlocal) response.

Inputs (p. 207): Drude ε = 1 - ωp²/ω(ω + iγ), ωp = 8.65e15 s-1 (λp = 217.9 nm), γ = 0.01 ωp; radius 1 nm
(Fig. 1) or 2 nm (Fig. 4), host ε_m = 1 (Fig. 1) or 2 (Fig. 4); normal incidence, E across the wire (the
calculator's axial-magnetic polarization).  C_e is the extinction width over the diameter 2a, the
calculator's extinction efficiency.  Local resonance at ωp/√(1 + ε_m) (printed); peak height read off.
"""

import numpy as np
from _common import drude_nk, nk_rows, peak, report, save_csv

import pystratify as ps

LP = 2 * np.pi * 2.99792458e14 / 8.65e15  # µm
x = np.linspace(0.4, 1.4, 1001)
LAM = LP / x


def q_ext(a, host):
    return np.array([ps.cross_widths(ps.solve_cylinder([a], [drude_nk(l, LP, 0.01), host, ], l), "axial-magnetic")["efficiency_extinction"] for l in LAM])


fig1, fig4 = q_ext(0.001, 1.0), q_ext(0.002, np.sqrt(2.0))
save_csv("ruppin_2001_fig1_fig4_local", "omega_over_omega_p,wavelength_um,Qext_a1nm_host1,Qext_a2nm_host_eps2", [x, LAM, fig1, fig4])
(__import__("pathlib").Path(__file__).parent / "results" / "ruppin_2001_sodium_drude_nk.json").write_text(
    __import__("json").dumps(nk_rows(np.linspace(0.155, 0.545, 391), drude_nk(np.linspace(0.155, 0.545, 391), LP, 0.01))))
x1, y1 = peak(x, fig1)
x4, _ = peak(x, fig4)
ok = report("ruppin_2001", [
    ("λp from ωp (nm)", 217.9, LP * 1e3, 0.2, "conversion of p. 207"),
    ("a = 1 nm, ε_m = 1: local resonance (ω/ωp)", 0.7071, x1, 0.004, "ωp/√2, p. 207"),
    ("a = 1 nm: log10 of the local peak", 0.65, np.log10(y1), 0.08, "Fig. 1 dashed, read"),
    ("a = 2 nm, ε_m = 2: local resonance (ω/ωp)", 0.5774, x4, 0.004, "ωp/√3, p. 208"),
    ("a = 1 nm: log10 C_e at 1.4 ωp", -3.35, np.log10(fig1[-1]), 0.25, "Fig. 1 dashed, read"),
])
print("PASS" if ok else "CHECK")
