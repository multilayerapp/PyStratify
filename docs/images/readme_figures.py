"""The figures of the README, computed with PyStratify from the setups written here.

    pip install -e . matplotlib
    python docs/images/readme_figures.py          # writes docs/images/*.png (about a minute)

Optical constants are simple on purpose: constant indices for the dielectrics, and for gold the analytic
Drude + two-critical-point fit of examples/_common.py (Etchegoin, Le Ru & Meyer 2006, fitted to Johnson & Christy).
Real work takes tabulated n, k (refractiveindex.info). Silver at 633 nm is Johnson & Christy's tabulated value.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import pystratify as ps  # noqa: E402
from pystratify import planar  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples"))
from _common import gold  # noqa: E402

# Okabe–Ito, as the site's plots
TEAL, ROSE, ORANGE, BLUE, GREY = "#087f78", "#cc6677", "#e69f00", "#0072b2", "#6b7780"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.dpi": 100, "savefig.dpi": 200,
    "lines.linewidth": 1.8,
})


def inset(ax, draw, where=(0.70, 0.56, 0.28, 0.40)):
    """A small sketch of the geometry in the corner of a panel."""
    sub = ax.inset_axes(where)
    sub.set_aspect("equal")
    sub.axis("off")
    draw(sub)
    return sub


def film_sketch(sub):
    for i in range(8):
        sub.add_patch(Rectangle((0, 0.14 + 0.09 * i), 1, 0.09, color=TEAL if i % 2 else "#bfe3df", lw=0))
    sub.add_patch(Rectangle((0, 0), 1, 0.14, color="#d9dee2", lw=0))
    sub.annotate("", (0.35, 0.86), (0.2, 1.12), arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    sub.set_xlim(0, 1), sub.set_ylim(0, 1.15)


def cylinder_sketch(sub):
    sub.add_patch(Circle((0.5, 0.5), 0.42, color=TEAL, lw=0))
    sub.annotate("", (0.02, 0.5), (-0.3, 0.5), arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    sub.text(0.5, 0.5, "Si", ha="center", va="center", color="white", fontsize=8, weight="bold")
    sub.set_xlim(-0.3, 1), sub.set_ylim(0, 1)


def shell_sketch(sub):
    sub.add_patch(Circle((0.5, 0.5), 0.45, color=ORANGE, lw=0))
    sub.add_patch(Circle((0.5, 0.5), 0.36, color="#e9eef0", lw=0))
    sub.annotate("", (0.03, 0.5), (-0.3, 0.5), arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    sub.text(0.5, 0.5, "SiO$_2$", ha="center", va="center", color=GREY, fontsize=7)
    sub.set_xlim(-0.3, 1), sub.set_ylim(0, 1)


def geometries():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)

    # Films: a quarter-wave Bragg mirror, 8 pairs of TiO2 / SiO2 at 600 nm on glass, at 0° and 45°.
    wl = np.linspace(400, 900, 501)
    n_hi, n_lo, center = 2.35, 1.46, 600.0
    n_list = [1.0] + [n_hi, n_lo] * 8 + [1.52]
    d_list = [np.inf] + [center / 4 / n_hi, center / 4 / n_lo] * 8 + [np.inf]
    ax = axes[0]
    for angle, pol, style, label in ((0, "s", "-", "normal incidence"), (45, "s", "--", "45°, s"), (45, "p", ":", "45°, p")):
        R = [planar.coh_tmm(pol, n_list, d_list, np.radians(angle), w)["R"] for w in wl]
        ax.plot(wl, R, style, color=TEAL if angle == 0 else (ROSE if pol == "s" else BLUE), label=label)
    ax.set(title="Films: a Bragg mirror", xlabel="wavelength (nm)", ylabel="reflectance", ylim=(0, 1.08), xlim=(400, 900))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=1.6, columnspacing=1)
    inset(ax, film_sketch, (0.80, 0.70, 0.19, 0.28))

    # Cylinders: a silicon nanowire, radius 80 nm, n = 3.6, under normal incidence; both polarisations.
    wl = np.linspace(400, 1400, 401)
    q = {p: [] for p in ("axial-electric", "axial-magnetic")}
    for w in wl:
        sol = ps.solve_cylinder([80.0], [3.6 + 0.0j, 1.0], w)
        for p in q:
            q[p].append(ps.cross_widths(sol, p)["efficiency_scattering"])
    ax = axes[1]
    ax.plot(wl, q["axial-electric"], color=TEAL, label="E along the axis (TM)")
    ax.plot(wl, q["axial-magnetic"], "--", color=ROSE, label="H along the axis (TE)")
    ax.set(title="Cylinders: a silicon nanowire", xlabel="wavelength (nm)", ylabel="scattering efficiency", xlim=(400, 1400))
    ax.set_ylim(0, max(max(v) for v in q.values()) * 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6, columnspacing=1)
    inset(ax, cylinder_sketch, (0.72, 0.58, 0.27, 0.38))

    # Spheres: a gold nanoshell (SiO2 core 60 nm, Au shell 15 nm) in water, with its multipoles.
    wl = np.linspace(450, 1100, 651)
    n = np.stack([np.full(wl.size, 1.45), gold(wl), np.full(wl.size, 1.33)], 1)
    cs = ps.cross_sections(ps.solve([60.0, 75.0], n, wl))
    ax = axes[2]
    ax.plot(wl, cs.q_ext, color="black", lw=2.2, label="extinction")
    q_order = cs.ext_by_order / cs.geometric          # per-order cross sections, as efficiencies
    ax.plot(wl, q_order[:, ps.TM, 0], "--", color=TEAL, label="electric dipole")
    ax.plot(wl, q_order[:, ps.TM, 1], "-.", color=ROSE, label="electric quadrupole")
    ax.plot(wl, q_order[:, ps.TE, 0], ":", color=BLUE, label="magnetic dipole")
    ax.plot(wl, cs.q_abs, color=ORANGE, lw=1.4, label="absorption")
    ax.set(title="Spheres: a gold nanoshell in water", xlabel="wavelength (nm)", ylabel="efficiency", xlim=(450, 1100))
    ax.set_ylim(0, cs.q_ext.max() * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=1.6, columnspacing=1)
    inset(ax, shell_sketch, (0.72, 0.58, 0.27, 0.38))

    fig.savefig(HERE / "geometries.png")
    plt.close(fig)


def emitters():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)

    # A dipole beside a 40-nm gold sphere at 600 nm: radiative, non-radiative and total rates, gaps 0.2–100 nm.
    radius, wavelength = 40.0, 600.0
    gaps = np.geomspace(0.2, 100, 80)
    d = ps.decay_rates([radius], [gold(wavelength), 1.0], wavelength, r=radius + gaps)
    ax = axes[0]
    for attr, style, color, label in (("total", "-", "black", "total"), ("radiative", "--", TEAL, "radiative"),
                                      ("nonradiative", ":", ORANGE, "absorbed in the gold")):
        values = getattr(d, attr)
        ax.loglog(gaps, values[:, 0], style, color=color, label=label)   # [:, 0] radial, [:, 1] tangential
    ax.set(title="A radial dipole beside a gold sphere", xlabel="gap to the surface (nm)",
           ylabel="decay rate / free-space rate")
    ax.text(0.03, 0.04, "radius 40 nm, λ = 600 nm", transform=ax.transAxes, color=GREY, fontsize=8.5)
    ax.legend(loc="upper right")

    # A dipole above a silver mirror, 200 nm of silver on glass (Drexhage's experiment): the total rate oscillates
    # with the height.
    wavelength, n_ag = 0.633, 0.135 + 3.99j          # µm; Johnson & Christy
    heights = np.linspace(0.02, 1.2, 119)
    rates = {"perpendicular": [], "parallel": []}
    for h in heights:
        res = ps.solve_problem(ps.Problem("films", [np.inf, 0.2, np.inf], [1.0, n_ag, 1.52], wavelength,
                                          ps.PointDipole(-h, "electric", layer=0)))
        rates["perpendicular"].append(res["total"][0])   # orientations: perpendicular, parallel, isotropic
        rates["parallel"].append(res["total"][1])
    ax = axes[1]
    ax.plot(heights / wavelength, rates["perpendicular"], color=TEAL, label="perpendicular dipole")
    ax.plot(heights / wavelength, rates["parallel"], "--", color=ROSE, label="parallel dipole")
    ax.axhline(1, color=GREY, lw=0.8)
    ax.set(title="A dipole above a silver mirror (Drexhage)", xlabel="height / wavelength",
           ylabel="decay rate / free-space rate", xlim=(0, heights[-1] / wavelength), ylim=(0, 2.6))
    ax.legend(loc="upper right")

    fig.savefig(HERE / "emitters.png")
    plt.close(fig)


# The Verification comparisons, read from the figures benchmarks/figures.py commits to docs/figures/ (not recomputed:
# each was run in the reference code's own environment). (figure id, label, kind)
REFERENCES = [
    ("pyelli-bragg-mirror", "pyElli, 2×2 and 4×4 · films", "code"),
    ("pymoosh-bragg-mirror", "PyMoosh, scattering matrices · films", "code"),
    ("tmmax-bragg-mirror", "tmmax (float32) · films", "code"),
    ("sommerfeld-film-emitter", "Sommerfeld integrals · film emitter", "precision"),
    ("smuthi-film-guide", "smuthi · film emitter", "code"),
    ("treams-cylinder-oblique", "treams · cylinders", "code"),
    ("treams-nanoshell", "treams · spheres", "code"),
    ("scattnlay-matryoshka", "scattnlay · spheres", "code"),
    ("bhmie-large-spheres", "BHMIE to x = 5000 · spheres", "code"),
    ("smuthi-sphere-metal", "smuthi · sphere emitter", "code"),
    ("sphere-decay-mpmath", "40-digit Mie series · sphere emitter", "precision"),
    ("chiral-80-digits", "80-digit transfer matrices · chiral sphere", "precision"),
    ("ngsolve-cylinder-fibre", "NGSolve, finite elements · fibre emitter", "mesh"),
    ("meep-cylinder-fibre", "Meep, 3D FDTD · fibre emitter", "mesh"),
    ("gdm-spectrum", "pyGDM2, volume Green dyadic · spheres", "mesh"),
    ("mnpbem-spectrum", "MNPBEM, boundary elements · spheres", "mesh"),
]
KINDS = {"code": (TEAL, "another code, exact method"), "precision": (BLUE, "extended-precision reference"),
         "mesh": (ORANGE, "discretized solver: its mesh or grid error")}


def verification():
    figures = {name: json.loads((HERE.parent / "figures" / f"{name}.json").read_text()) for name, _, _ in REFERENCES}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(REFERENCES))[::-1]
    for yi, (name, label, kind) in zip(y, REFERENCES):
        left.hlines(yi, 1e-16, figures[name]["worst"], color="#d9dee2", lw=1)
        left.plot(figures[name]["worst"], yi, "o", color=KINDS[kind][0], ms=7)
    left.set_xscale("log")
    left.set(xlim=(1e-16, 1), yticks=y, yticklabels=[label for _, label, _ in REFERENCES],
             xlabel="largest difference from the reference", title="PyStratify against independent references")
    left.tick_params(axis="y", length=0, labelsize=8.5)
    left.spines["left"].set_visible(False)
    for kind, (color, text) in KINDS.items():
        left.plot([], [], "o", color=color, label=text)
    left.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3, fontsize=8, handletextpad=0.2, columnspacing=1)

    d = figures["meep-cylinder-fibre"]
    series = {s["key"]: s for s in d["series"]}
    for key, color in (("radial", TEAL), ("azimuthal", ROSE), ("axial", BLUE)):
        right.plot(d["x"], series[key]["y"], color=color, label=f"{key}, PyStratify")
        right.plot(d["x"][::6], series[key + "_ref"]["y"][::6], "o", mfc="none", color=color, ms=5, mew=1.2)
    right.plot([], [], "o", mfc="none", color=GREY, label="Meep, 3D FDTD")
    right.set(title="A dipole outside a fibre: against 3D FDTD", xlabel="wavelength (µm)",
              ylabel="decay rate / free-space rate")
    right.legend(loc="upper right")
    right.text(0.02, 0.03, "n = 2, radius 150 nm, dipole 100 nm outside", transform=right.transAxes, color=GREY,
               fontsize=8.5)
    fig.savefig(HERE / "verification.png")
    plt.close(fig)


if __name__ == "__main__":
    geometries()
    emitters()
    verification()
    print("wrote", ", ".join(p.name for p in sorted(HERE.glob("*.png"))))
