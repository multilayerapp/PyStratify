"""The calculator setup each verification figure freezes on its page and sketches.

multilayer.app's build reads each as the figure's `calculator` block: a share token for the calculator (whose own
sketch and badges the page shows), the frozen Source panel (`rows`: the calculator's two rows of fields, the second
the range the figure sweeps, read off the figure's own x values where it is `sweep`) and the frozen stack table. A
figure with no single setup (a closed form, a model's ε, a coefficient check without a particle) has none.

Lengths are in nm here; the blocks carry what the calculator stores (layer thicknesses in µm).
"""

from __future__ import annotations

import numpy as np

METAL = complex(np.sqrt(-10 + 1j + 0j))  # ε = −10 + i, the metal-like medium of the sphere references
SILICA = complex(np.sqrt(2.13))
SILVER = 0.056 + 4.28j  # emitter_references.SILVER


def med(label, n, mu=None, model=None, n_label=None, k_label=None):
    """A medium: n + ik; μ where it is not 1; `model` names a dispersive one (the frozen table shows the model, not n)."""
    n = complex(n)
    out = {"label": label, "n": round(n.real, 6), "k": round(n.imag, 9)}
    if mu is not None and mu != 1:
        out["mu"] = float(mu)
    if model:
        out["model"] = model
    if n_label:
        out["n_label"] = n_label
    if k_label:
        out["k_label"] = k_label
    return out


def lay(label, n, d_nm, coherency=None, shown=None, roughness=None, mix=None, **extra):
    """A layer (film) or shell (its thickness; a core's is its radius); `shown` replaces the thickness the table shows.
    `roughness` makes it the calculator's Bruggeman layer (the share of the medium below, the layer it sits on, which
    face), `mix` what its table row says it mixes."""
    out = {**med(label, n, **extra), "thickness": round(d_nm / 1000, 9)}
    if roughness:
        below, owner, side = roughness
        out["roughness"] = {"model": "bruggeman", "bottomFraction": below, "topFraction": round(1 - below, 9), "ownerLayerId": owner, "ownerSide": side}
        out["mix"] = mix
    if coherency:
        out["coherency"] = coherency
    if shown:
        out["thickness_label"] = shown
    return out


def bragg(count, *layers):
    return {"repeat": count, "layers": list(layers)}


def sweep(label, a=1.0, b=0.0):
    """The figure's x values as a + ... : the Source panel's second row shows a·x + b from its first to its last value."""
    return {"sweep": {"label": label, "a": a, "b": b}}


AIR, GLASS = med("Air", 1.0), med("Glass", 1.52)
PLANE, ELECTRIC, MAGNETIC = "Plane wave", "Electric point dipole", "Magnetic point dipole"


def _wl(wl):
    lo, hi, step = (wl, wl, 1) if np.isscalar(wl) else wl
    return lo, hi, step


def _lam_fields(wl):
    lo, hi, step = _wl(wl)
    return [("λ [nm]", lo)] if lo == hi else [("λ min [nm]", lo), ("λ max [nm]", hi), ("λ step [nm]", step)]


def _envelope(geometry, wl, dipole, angle=None):
    lo, hi, step = _wl(wl)
    env = {"start": lo, "end": hi, "step": step}
    if angle is not None:
        env["angle"] = angle
    if dipole:
        kind, position = dipole
        env.update({"kind": "dipole", "dipole_type": kind, "region": "origin" if geometry == "films" else "core",
                    "offset": position, "emitter_outputs": ["decay"]})
    return env


def film(incident, layers, substrate, wl, rows, angle=0.0, angle_sweep=None, dipole=None, link=True, note=None):
    """A film stack: plane wave at `angle` (or swept over `angle_sweep`) or `dipole` = (electric|magnetic, z in nm, z < 0
    above the stack)."""
    lo, hi, step = _wl(wl)
    controls = {"wavelength-start": lo, "wavelength-end": hi, "wavelength-step": step}
    if angle_sweep:
        controls.update({"angle-start": angle_sweep[0], "angle-end": angle_sweep[1], "angle-step": angle_sweep[2], "sweepMode": "angle"})
    else:
        controls["angle-deg"] = angle
    out = {"film": {"incident": incident, "layers": layers, "substrate": substrate, "controls": controls, "view": {"polarization": "average"}},
           "rows": rows, "link": link}
    if dipole:
        out["envelope"] = _envelope("films", wl, dipole)
    if note:
        out["link_note"] = note
    return out


def particle(geometry, shells, host, wl, rows, dipole=None, angle=None, link=True, note=None):
    """A sphere or a cylinder: shells from the core out (each its own thickness, the core's its radius) in `host`."""
    lo, hi, step = _wl(wl)
    out = {"envelope": _envelope(geometry, wl, dipole, angle), "rows": rows, "link": link}
    if geometry == "spheres":
        out["sphere"] = {"shells": shells, "host": host, "start": lo, "end": hi, "step": step}
    else:
        out["cylinder"] = {"shells": shells, "host": host}
    if note:
        out["link_note"] = note
    return out


def radii(*pairs):
    """Shells from (label, n, outer radius) triples: each shell's thickness is the step between outer radii."""
    out, inner = [], 0.0
    for label, n, outer, *rest in pairs:
        out.append(lay(label, n, outer - inner, **(rest[0] if rest else {})))
        inner = outer
    return out


MODEL_NOTE = "This setup's dispersive media are models the calculator holds only as constant indices here; it opens there approximately."
MU_NOTE = "The calculator has no permeability: this setup opens there only without μ."
SWEEP_NOTE = "The calculator opens this setup at one value of the swept quantity."

SETUPS = {
    # --- films, plane wave: consistency checks -------------------------------------------------------------------
    "fresnel-oblique": film(AIR, [], med("Glass", 1.5), 600, [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("θ [°]")]], angle_sweep=(0, 89.5, 0.5)),
    "airy-slab": film(AIR, [lay("Slab", 2.0, 300)], AIR, (400, 800, 2), [[("Source type", PLANE), ("θ from normal [°]", 0)], [sweep("λ [nm]")]]),
    "energy-conservation": film(med("Glass", 1.52), [lay("H", 2.3, 80), lay("L", 1.38, 120), lay("H", 2.3, 80), lay("L", 1.38, 120), lay("H", 2.3, 80)], AIR,
                                550, [[("Source type", PLANE), ("λ [nm]", 550)], [sweep("θ [°]")]], angle_sweep=(0, 89, 0.5)),
    "opaque-film": film(AIR, [lay("Metal", 0.2 + 3j, 150, shown="1–150")], AIR, 600,
                        [[("Source type", PLANE), ("θ from normal [°]", 0), ("λ [nm]", 600)], [sweep("d [nm]")]], note=SWEEP_NOTE),
    "bare-substrate-psi": film(AIR, [], med("Substrates", 1.5, n_label="1.5 · 1.5 · 0.2", k_label="0 · 0.3 · 3"), 633,
                               [[("Source type", PLANE), ("λ [nm]", 633)], [sweep("θ [°]")]], angle_sweep=(1, 89, 1),
                               note="Three substrates in one plot: the calculator opens the first, n = 1.5."),
    "bare-substrate-delta": film(AIR, [], med("Substrates", 1.5, n_label="1.5 · 1.5 · 0.2", k_label="0 · 0.3 · 3"), 633,
                                 [[("Source type", PLANE), ("λ [nm]", 633)], [sweep("θ [°]")]], angle_sweep=(1, 89, 1),
                                 note="Three substrates in one plot: the calculator opens the first, n = 1.5."),
    "smatrix-metal-stack": film(AIR, [lay("Silica", 1.46, 100), lay("Metal", 0.05 + 3j, 40)], GLASS, 600,
                                [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("θ [°]")]], angle_sweep=(0, 89, 1)),
    "smatrix-beyond-critical": film(med("Glass", 1.5), [lay("Air gap", 1.0, 400), lay("n 2", 2.0, 130)], med("Glass", 1.5), 550,
                                    [[("Source type", PLANE), ("λ [nm]", 550)], [sweep("θ [°]")]], angle_sweep=(0, 89, 1)),
    "incoherent-phase-average": film(AIR, [lay("Film 1", 2.08 + 0.03j, 90), lay("Thick layer", 1.47 + 0.001j, 12000, "i"), lay("Film 2", 1.82 + 0.02j, 110)],
                                     med("Substrate", 1.56), 640, [[("Source type", PLANE), ("λ [nm]", 640)], [sweep("θ [°]")]], angle_sweep=(0, 80, 2.5)),
    "rcwa-unpatterned": film(AIR, [lay("H", 2.35, 95), lay("L", 1.46, 110), lay("H", 2.35, 95)], med("Substrate", 1.5), (400, 800, 5),
                             [[("Source type", PLANE), ("θ from normal [°]", 0)], [sweep("λ [nm]")]]),
    "position-resolved-absorption": film(AIR, [lay("Layer 1", 2.1 + 0.08j, 120), lay("Layer 2", 1.46 + 0.01j, 250), lay("Metal", 0.2 + 3.2j, 20)], GLASS,
                                         (400, 1000, 10), [[("Source type", PLANE), ("θ from normal [°]", 30)], [sweep("λ [nm]")]], angle=30),
    # --- films, plane wave: pyElli, tmm, tmmax -------------------------------------------------------------------
    "pyelli-bragg-mirror": film(AIR, [bragg(10, lay("H", 2.35, 67.3), lay("L", 1.46, 108.4))], GLASS, (450, 850, 5),
                                [[("Source type", PLANE), ("θ from normal [°]", 30)], [sweep("λ [nm]")]], angle=30),
    "pyelli-gold-ellipsometry": film(AIR, [lay("Au", 0.27 + 2.93j, 20)], GLASS, 633, [[("Source type", PLANE), ("λ [nm]", 633)], [sweep("θ [°]")]],
                                     angle_sweep=(1, 85, 1)),
    "pyelli-dispersive-stack": film(AIR, [lay("Bruggeman", 1.9, 3, roughness=(0.75, 2, "top"), mix="25 % air · 75 % film"), lay("Film", 2.1 + 0.1j, 80, model="Tauc–Lorentz")],
                                    med("Si", 3.88 + 0.02j, model="Green 2008"), (250, 1450, 10),
                                    [[("Source type", PLANE), ("θ from normal [°]", 70)], [sweep("λ [nm]")]], angle=70, link=False, note=MODEL_NOTE),
    "inc-tmm-lineage": film(AIR, [lay("Film 1", 2.08 + 0.03j, 90), lay("Thick layer", 1.47 + 0.001j, 12000, "i"), lay("Film 2", 1.82 + 0.02j, 110)],
                            med("Substrate", 1.56), (450, 900, 5), [[("Source type", PLANE), ("θ from normal [°]", 31)], [sweep("λ [nm]")]], angle=31),
    "backside-reflection": film(AIR, [lay("Film", 2.0, 120, model="Cauchy 2.0 + 0.01/λ²"), lay("Glass", 1.52, 100000, "i")], AIR, (400, 900, 10),
                                [[("Source type", PLANE), ("θ from normal [°]", 65)], [sweep("λ [nm]")]], angle=65),
    "tmmax-single-layer": film(AIR, [lay("Layer", 2.1, 100)], AIR, (400, 900, 5), [[("Source type", PLANE), ("θ from normal [°]", 0)], [sweep("λ [nm]")]]),
    "tmmax-bragg-mirror": film(AIR, [bragg(5, lay("H", 2.35, 58.5), lay("L", 1.46, 94.2))], GLASS, (400, 900, 5),
                               [[("Source type", PLANE), ("θ from normal [°]", 0)], [sweep("λ [nm]")]]),
    # --- films, point dipole -------------------------------------------------------------------------------------
    "sommerfeld-film-emitter": film(AIR, [lay("Ag", SILVER, 60)], GLASS, 633, [[("Source type", ELECTRIC), ("λ [nm]", 633)], [sweep("z [nm]", -1)]],
                                    dipole=("electric", -50)),
    "film-dipole-homogeneous": film(med("n 1.5", 1.5), [lay("Film", 1.5, 200)], med("n 1.5", 1.5), 600,
                                    [[("Source type", "Electric and magnetic dipoles"), ("λ [nm]", 600)], [sweep("z [nm]")]], dipole=("electric", 100)),
    "film-coating-budget-electric": film(AIR, [lay("Absorber", 0.2 + 3j, 100)], AIR, 600, [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("z [nm]", -1)]],
                                         dipole=("electric", -50)),
    "film-coating-budget-magnetic": film(AIR, [lay("Absorber", 0.2 + 3j, 100)], AIR, 600, [[("Source type", MAGNETIC), ("λ [nm]", 600)], [sweep("z [nm]", -1)]],
                                         dipole=("magnetic", -50)),
    "film-guided-pole-tolerance": film(AIR, [lay("Film", 3.5, 800)], AIR, 600,
                                       [[("Source type", ELECTRIC), ("Offset from origin · z [nm]", 400), ("λ [nm]", 600)], [sweep("tolerance")]],
                                       dipole=("electric", 400)),
    "film-guide-budget": film(AIR, [lay("Guide", 1.5, 200)], AIR, 600, [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("z [nm]")]], dipole=("electric", 100)),
    "film-pattern-free-space": film(AIR, [lay("Film", 1.0, 200)], AIR, 600,
                                    [[("Source type", ELECTRIC), ("Offset from origin · z [nm]", 100), ("λ [nm]", 600)], [sweep("θ [°]")]], dipole=("electric", 100)),
    "weak-guide-sommerfeld": film(AIR, [lay("TiO₂", 2.3 + 0.001j, 200)], GLASS, 633, [[("Source type", ELECTRIC), ("λ [nm]", 633)], [sweep("z [nm]", -1)]],
                                  dipole=("electric", -50)),
    "lossless-guide-limit": film(AIR, [lay("TiO₂", 2.3, 200)], GLASS, 633, [[("Source type", ELECTRIC), ("λ [nm]", 633)], [sweep("z [nm]", -1)]],
                                 dipole=("electric", -50)),
    "slab-dispersion-modes": film(AIR, [lay("Slab", 2.0, 500, shown="100–600")], AIR, 633,
                                  [[("Source type", "Guided modes"), ("λ [nm]", 633)], [sweep("d [nm]")]], dipole=("electric", 167), note=SWEEP_NOTE),
    "pymoosh-guided-modes": film(AIR, [lay("Slab", 2.0, 500, shown="100–600")], AIR, 633,
                                 [[("Source type", "Guided modes"), ("λ [nm]", 633)], [sweep("d [nm]")]], dipole=("electric", 167), note=SWEEP_NOTE),
    "meep-axial-film": film(AIR, [lay("Film", 2.3, 200)], GLASS, (606, 1176, 5),
                            [[("Source type", ELECTRIC), ("Offset from origin · z [nm]", -100)], [sweep("λ [nm]", 1000)]], dipole=("electric", -100)),
    "smuthi-film-guide": film(AIR, [lay("TiO₂", 2.3, 200)], GLASS, 633, [[("Source type", ELECTRIC), ("λ [nm]", 633)], [sweep("z [nm]", -1)]],
                              dipole=("electric", -100)),
    # --- spheres, plane wave -------------------------------------------------------------------------------------
    "bhmie-large-spheres": particle("spheres", [lay("Sphere", 1.5 + 0.1j, 1000, shown="x λ / 2π")], AIR, 600,
                                    [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("x")]], note=SWEEP_NOTE),
    "sixty-digit-coefficients": particle("spheres", radii(("Silica", SILICA, 15), ("Metal", METAL, 20)), AIR, 600,
                                         [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("ℓ")]]),
    "centre-field": particle("spheres", [lay("Sphere", 3 + 0.01j, 200, shown="x λ / 2π")], AIR, 600,
                             [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("x")]], note=SWEEP_NOTE),
    "mie-homogeneous": particle("spheres", [lay("Sphere", 1.5, 1000, shown="x λ / 2π")], AIR, 1000,
                                [[("Source type", PLANE), ("λ [nm]", 1000)], [sweep("x")]], note=SWEEP_NOTE),
    "optical-theorem": particle("spheres", radii(("Silica", 1.45, 50), ("Au", 0.27 + 2.93j, 55)), AIR, (450, 1000, 5),
                                [[("Source type", PLANE)], [sweep("λ [nm]")]]),
    "rayleigh-limit": particle("spheres", [lay("Sphere", 1.5 + 0.2j, 50, shown="x λ / 2π")], AIR, 614,
                               [[("Source type", PLANE), ("λ [nm]", 614)], [sweep("x")]], note=SWEEP_NOTE),
    "chiral-80-digits": particle("spheres", radii(("Core", 1.45 + 0.01j, 40), ("Chiral shell", 1.6 + 0.02j, 55)), med("Water", 1.33), 690,
                                 [[("Source type", PLANE), ("λ [nm]", 690), ("κ of the shell", "0.05 + 0.01i")], [sweep("ℓ")]],
                                 link=False, note="The calculator holds no chirality: this sphere opens there without κ."),
    "thin-film-sheet": particle("spheres", radii(("Core", 3.5, 200), ("Film", complex(np.sqrt(15 + 5j)), 201)), med("Water", 1.33), (500, 900, 2.5),
                                [[("Source type", PLANE)], [sweep("λ [nm]")]]),
    "near-field-truncation": particle("spheres", [lay("Gold-like", 0.25 + 3j, 50)], med("Water", 1.33), 600,
                                      [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("r [nm]")]]),
    "mnpbem-spectrum": particle("spheres", [lay("Metal-like", METAL, 20)], med("Vacuum", 1.0), (400, 800, 2.5), [[("Source type", PLANE)], [sweep("λ [nm]")]]),
    "mnpbem-thin-nanoshell": particle("spheres", radii(("Silica", SILICA, 15), ("Metal-like", METAL, 20)), med("Vacuum", 1.0), (400, 800, 2.5),
                                      [[("Source type", PLANE)], [sweep("λ [nm]")]]),
    "gdm-spectrum": particle("spheres", [lay("Sphere", 2 + 0.1j, 40)], AIR, (400, 800, 2.5), [[("Source type", PLANE)], [sweep("λ [nm]")]]),
    "scattnlay-matryoshka": particle("spheres", radii(("Metal", METAL, 10, {"shown": "R/3"}), ("Silica", 1.45, 20, {"shown": "R/3"}), ("Metal", METAL, 30, {"shown": "R/3"})),
                                     med("Water", 1.33), 600,
                                     [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("R [nm]")]], note=SWEEP_NOTE),
    "scattnlay-pattern": particle("spheres", [lay("Water-like", 1.33, 19099)], AIR, 600,
                                  [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("θ [°]")]]),
    "scattnlay-near-field": particle("spheres", radii(("Metal", METAL, 10), ("Silica", 1.45, 20), ("Metal", METAL, 30)), med("Water", 1.33), 600,
                                     [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("x [nm]")]]),
    "treams-chiral": particle("spheres", radii(("Core", 1.45 + 0.01j, 40), ("Chiral shell", 1.6 + 0.02j, 55)), med("Water", 1.33), (400, 900, 5),
                              [[("Source type", PLANE), ("κ of the shell", "0.05 + 0.01i")], [sweep("λ [nm]")]],
                              link=False, note="The calculator holds no chirality: this sphere opens there without κ."),
    "treams-nanoshell": particle("spheres", radii(("Silica", SILICA, 10), ("Metal-like", METAL, 20)), AIR, (400, 900, 5), [[("Source type", PLANE)], [sweep("λ [nm]")]]),
    "treams-near-field": particle("spheres", radii(("Silica", SILICA, 10), ("Metal-like", METAL, 20)), AIR, 600,
                                 [[("Source type", PLANE), ("λ [nm]", 600)], [sweep("r [nm]")]]),
    # --- spheres, point dipole -----------------------------------------------------------------------------------
    "silver-decay-mie": particle("spheres", [lay("Silver", 0.093 + 4j, 50)], AIR, 633, [[("Source type", ELECTRIC), ("λ [nm]", 633)], [sweep("r [nm]", 1, 50)]],
                                 dipole=("electric", 60)),
    "nanoshell-decay-layers": particle("spheres", radii(("Silica", 1.45, 50), ("Au", 0.27 + 2.93j, 55)), med("Water", 1.33), 614,
                                       [[("Source type", ELECTRIC), ("λ [nm]", 614)], [sweep("r [nm]")]], dipole=("electric", 70)),
    "silicon-magnetic-decay": particle("spheres", [lay("Silicon", 3.7 + 0.005j, 100)], AIR, 800, [[("Source type", MAGNETIC), ("λ [nm]", 800)], [sweep("r [nm]", 1, 100)]],
                                       dipole=("magnetic", 120)),
    "sphere-decay-mpmath": particle("spheres", [lay("Metal", 0.05 + 3.5j, 25)], AIR, 600, [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("r [nm]", 1, 25)]],
                                    dipole=("electric", 35)),
    "radiated-power-rates": particle("spheres", radii(("Core", 1.45, 50, {"mu": 1.3}), ("Au", 0.27 + 2.93j, 70), ("Shell", 1.6, 90, {"mu": 1.7})),
                                     med("Water", 1.33, mu=1.1), 614, [[("Source type", "Electric and magnetic dipoles"), ("λ [nm]", 614)], [sweep("r [nm]")]],
                                     dipole=("electric", 120), link=False, note=MU_NOTE),
    "two-point-green": particle("spheres", radii(("Silica", 1.45, 50), ("Au", 0.27 + 2.93j, 55)), med("Water", 1.33), 614,
                                [[("Source type", ELECTRIC), ("Offset from origin · r [nm]", 56), ("observed at r [nm]", 60)], [sweep("ℓ")]],
                                dipole=("electric", 56)),
    "meep-axial-sphere": particle("spheres", [lay("Sphere", 2.0, 150)], AIR, (606, 1176, 5),
                                  [[("Source type", ELECTRIC), ("Offset from origin · r [nm]", 250)], [sweep("λ [nm]", 1000)]], dipole=("electric", 250)),
    "mnpbem-decay": particle("spheres", [lay("Metal-like", METAL, 20)], med("Vacuum", 1.0), 600, [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("r [nm]", 1, 20)]],
                             dipole=("electric", 30)),
    "gdm-decay": particle("spheres", [lay("Sphere", 2.0, 40)], AIR, 600, [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("r [nm]", 1, 40)]], dipole=("electric", 60)),
    "smuthi-sphere-dielectric": particle("spheres", [lay("Sphere", 2.0, 150)], AIR, 800, [[("Source type", ELECTRIC), ("λ [nm]", 800)], [sweep("r [nm]", 1, 150)]],
                                         dipole=("electric", 250)),
    "smuthi-sphere-metal": particle("spheres", [lay("Metal-like", METAL, 20)], AIR, 600, [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("r [nm]", 1, 20)]],
                                    dipole=("electric", 25)),
    "smuthi-sphere-water": particle("spheres", [lay("Sphere", 1.5, 100)], med("Water", 1.33), 633, [[("Source type", ELECTRIC), ("λ [nm]", 633)], [sweep("r [nm]", 1, 100)]],
                                    dipole=("electric", 120)),
    "smuthi-sphere-large": particle("spheres", [lay("Sphere", 3.5 + 0.01j, 500)], AIR, 800, [[("Source type", ELECTRIC), ("λ [nm]", 800)], [sweep("r [nm]", 1, 500)]],
                                    dipole=("electric", 550)),
    "treams-dipole-nanoshell": particle("spheres", radii(("Silica", SILICA, 10), ("Metal-like", METAL, 20)), AIR, 600,
                                        [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("r [nm]", 1, 20)]], dipole=("electric", 25)),
    # --- cylinders -----------------------------------------------------------------------------------------------
    "cylinder-bessel-direct": particle("cylinders", [lay("Cylinder", 1.5, 300, shown="20–600")], AIR, 600,
                                       [[("Source type", PLANE), ("θ from axis [°]", 90), ("λ [nm]", 600)], [sweep("R [nm]")]], angle=90, note=SWEEP_NOTE),
    "cylinder-index-matched": particle("cylinders", [lay("Cylinder", 1.5, 100)], med("Host", 1.5), 600,
                                       [[("Source type", PLANE), ("θ from axis [°]", 82.69), ("λ [nm]", 600)], [sweep("x [nm]")]], angle=82.69),
    "treams-cylinder-oblique": particle("cylinders", [lay("Cylinder", 1.5 + 0.01j, 100)], AIR, (400, 900, 10),
                                        [[("Source type", PLANE), ("θ from axis [°]", 60)], [sweep("λ [nm]")]], angle=60),
    "cylinder-dipole-homogeneous": particle("cylinders", [lay("Cylinder", 1.5, 100)], med("Host", 1.5), 600,
                                            [[("Source type", "Electric and magnetic dipoles"), ("λ [nm]", 600)], [sweep("r [nm]")]], dipole=("electric", 150)),
    "cylinder-coating-budget-electric": particle("cylinders", radii(("Core", 1.5, 50), ("Absorber", 0.2 + 3j, 100)), AIR, 600,
                                                 [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("r [nm]")]], dipole=("electric", 150)),
    "cylinder-coating-budget-magnetic": particle("cylinders", radii(("Core", 1.5, 50), ("Absorber", 0.2 + 3j, 100)), AIR, 600,
                                                 [[("Source type", MAGNETIC), ("λ [nm]", 600)], [sweep("r [nm]")]], dipole=("magnetic", 150)),
    "cylinder-guide-budget": particle("cylinders", [lay("Guide", 1.5, 100)], AIR, 600, [[("Source type", ELECTRIC), ("λ [nm]", 600)], [sweep("r [nm]")]],
                                      dipole=("electric", 50)),
    "cylinder-grazing-tolerance": particle("cylinders", [lay("Metal wire", 0.97 + 1.87j, 10)], AIR, (400, 700, 20),
                                           [[("Source type", ELECTRIC), ("Offset from origin · r [nm]", 15)], [sweep("λ [nm]")]], dipole=("electric", 15)),
    "cylinder-grazing-caps-electric": particle("cylinders", [lay("Metal wire", 0.97 + 1.87j, 10)], AIR, (400, 700, 20),
                                               [[("Source type", ELECTRIC), ("Offset from origin · r [nm]", 15)], [sweep("λ [nm]")]], dipole=("electric", 15)),
    "cylinder-grazing-caps-magnetic": particle("cylinders", [lay("Metal wire", 0.97 + 1.87j, 10)], AIR, (400, 700, 20),
                                               [[("Source type", MAGNETIC), ("Offset from origin · r [nm]", 15)], [sweep("λ [nm]")]], dipole=("magnetic", 15)),
    "fibre-dispersion": particle("cylinders", [lay("Fibre", 2.0, 150)], AIR, (250, 380, 5), [[("Source type", "Guided modes")], [sweep("λ [nm]", 1000)]],
                                 dipole=("electric", 250)),
    "fibre-rates-cutoff": particle("cylinders", [lay("Fibre", 2.0, 150)], AIR, (620, 740, 2.5),
                                   [[("Source type", ELECTRIC), ("Offset from origin · r [nm]", 250)], [sweep("λ [nm]", 1000)]], dipole=("electric", 250)),
    "meep-cylinder-fibre": particle("cylinders", [lay("Fibre", 2.0, 150)], AIR, (606, 1176, 5),
                                    [[("Source type", ELECTRIC), ("Offset from origin · r [nm]", 250)], [sweep("λ [nm]", 1000)]], dipole=("electric", 250)),
    "ngsolve-cylinder-fibre": particle("cylinders", [lay("Fibre", 2.0, 150)], med("Vacuum", 1.0), (620, 1200, 10),
                                       [[("Source type", ELECTRIC), ("Offset from origin · r [nm]", 250)], [sweep("λ [nm]", 1000)]], dipole=("electric", 250)),
}


# One line beside each figure's title (the two must fit one row): what is computed, against what.
SUMMARIES = {
    'pymoosh-bragg-mirror': "R and T for s and p at 30°, against PyMoosh's scattering matrices",
    'pymoosh-field': "the field through the pair at 60°, E_y (s) and H_y (p), against PyMoosh's field solver",
    'pymoosh-incoherent': "R and T with the glass's back reflection added in power, against PyMoosh's incoherent solver",
    'fresnel-oblique': "R for s and p through Brewster's angle, against Fresnel's closed form",
    'airy-slab': 'R of a free-standing slab against the Airy formula',
    'energy-conservation': 'nothing is absorbed, so 1 − R − T must vanish at every angle, beyond critical too',
    'opaque-film': 'thickened from 1 to 150 nm, R settles on the bulk Fresnel reflectance',
    'bare-substrate-psi': 'Ψ of three substrates against the Fresnel ratio −r_p / r_s',
    'bare-substrate-delta': 'Δ of three substrates against the Fresnel ratio −r_p / r_s',
    'smatrix-metal-stack': 'R and T against an independent scattering-matrix solution',
    'smatrix-beyond-critical': 'light tunnels through an air gap beyond 41.8°: against the scattering-matrix solution',
    'incoherent-phase-average': "the incoherent solver against 301 coherent solves averaged over the thick layer's phase",
    'rcwa-unpatterned': 'RCWA (meent) with nothing patterned must reproduce the transfer matrix',
    'position-resolved-absorption': 'absorption integrated through each layer against the Poynting flux it loses',
    'kramers-kronig-closed-forms': 'closed-form ε₁ against the numerical Kramers–Kronig transform of ε₂',
    'pyelli-bragg-mirror': "R_s at 30° against pyElli's 2×2 and Berreman 4×4 solvers",
    'pyelli-gold-ellipsometry': "Ψ and Δ of 20 nm of gold on glass against pyElli's 4×4 solver",
    'pyelli-tauc-lorentz': 'ε₁ and ε₂ of the same model, written separately in both codes',
    'pyelli-drude': "a free-electron metal's ε against pyElli's DrudeEnergy",
    'pyelli-dispersive-stack': "Ψ and Δ at 70° against pyElli's 2×2 and 4×4 solvers",
    'inc-tmm-lineage': "PyStratify's incoherent solver against inc_tmm, the code it replaced",
    'backside-reflection': "the glass's back reflection at 65° against inc_tmm and a phase average",
    'tmmax-single-layer': 'R against tmmax as shipped (JAX, float32)',
    'tmmax-bragg-mirror': 'tmmax as shipped is wrong beyond one layer; with its scan step fixed it agrees',
    'film-dipole-homogeneous': 'a film matched to its surroundings is no film: Γ/Γ₀ = 1 everywhere',
    'film-coating-budget-electric': 'the total rate against escape + guided + absorbed',
    'film-coating-budget-magnetic': 'the total rate against escape + guided + absorbed',
    'film-guided-pole-tolerance': 'the error reported against the accuracy asked for, 18 guided modes',
    'film-guide-budget': 'the total rate against escape + guided, through a lossless guide',
    'film-pattern-free-space': "the emission pattern of a dipole in nothing: the free dipole's",
    'image-dipole-mirror': 'the Sommerfeld reference itself against the image dipole of a perfect mirror',
    'sommerfeld-film-emitter': 'decay rates against the Sommerfeld integrals, written without PyStratify',
    'weak-guide-sommerfeld': "decay rates against the Sommerfeld integrals through the guide's quasi-poles",
    'lossless-guide-limit': 'guided power by residues against the same guide with a trace of loss',
    'slab-dispersion-modes': 'guided modes against the three-layer dispersion relation, thickness swept',
    'pymoosh-guided-modes': "guided modes against PyMoosh's mode solver, thickness swept",
    'meep-axial-film': "total decay rates against Meep's FDTD (cylindrical, 100 px/µm)",
    'smuthi-film-guide': 'total decay rates against smuthi, the dipole moving away',
    'bhmie-large-spheres': "Q_ext up to size parameter 5000 against Bohren & Huffman's BHMIE",
    'sixty-digit-coefficients': 'every Mie coefficient against transfer matrices in 60 digits',
    'centre-field': "the field at the centre against Bohren & Huffman's d₁ and c₁",
    'mie-homogeneous': "Mie's series, one shell and fifty identical shells must agree",
    'optical-theorem': 'extinction and scattering two ways: optical theorem, angular integral',
    'rayleigh-limit': "the dipole coefficient a₁ against Rayleigh's limit as x → 0",
    'lossless-absorbs-nothing': 'random lossless multilayers must absorb nothing',
    'chiral-80-digits': 'the T-matrix against transfer matrices in 80 digits',
    'graphene-sphere-plasmon': 'the plasmon resonance against its quasi-static position',
    'thin-film-sheet': 'a 1 nm film as a shell and as a 2D conducting sheet',
    'near-field-truncation': '|E|² outside with the truncation PyStratify picks, against 120 orders',
    'mnpbem-spectrum': "cross sections against MNPBEM's boundary elements (percent-level by design)",
    'mnpbem-thin-nanoshell': "three Mie codes and MNPBEM's boundary elements",
    'gdm-spectrum': "cross sections against pyGDM2's volume discretization (percent-level by design)",
    'scattnlay-matryoshka': 'radii scaled together, 1 : 2 : 3, outer 9–120 nm: Q against scattnlay',
    'scattnlay-pattern': 'the amplitudes |S₁| and |S₂| at every angle against scattnlay',
    'scattnlay-near-field': "|E|² through every layer against scattnlay's field code",
    'treams-chiral': 'the helicity T-matrix against treams',
    'treams-nanoshell': 'Q_ext and Q_sca against treams',
    'treams-near-field': 'the scattered field outside against treams',
    'silver-decay-mie': "decay rates against Mie's series in 40 digits",
    'nanoshell-decay-layers': 'decay rates inside and outside against transfer matrices in 60 digits',
    'silicon-magnetic-decay': "a magnetic dipole's decay rates against 60-digit transfer matrices",
    'sphere-decay-mpmath': 'total and radiative rates against the series in 40 digits',
    'radiated-power-rates': 'the radiative rate against the power reaching the far field',
    'two-point-green': "the Green's dyadic between two points, order by order, against 80 digits",
    'quadrupole-coupling': "the quadrupole's coupling to each order against finite differences",
    'casimir-non-retarded': 'the Casimir–Polder potential against its non-retarded limit −C₃/d³',
    'meep-axial-sphere': "total decay rates on the axis against Meep's FDTD",
    'mnpbem-decay': "decay rates against MNPBEM's boundary elements",
    'gdm-decay': "total decay rates against pyGDM2's discretized sphere",
    'smuthi-sphere-dielectric': 'decay rates against smuthi, the dipole moving away',
    'smuthi-sphere-metal': 'decay rates against smuthi, the dipole moving away',
    'smuthi-sphere-water': 'decay rates against smuthi, the dipole moving away',
    'smuthi-sphere-large': 'decay rates against smuthi, the dipole moving away',
    'treams-dipole-nanoshell': 'decay rates against treams, the dipole moving away',
    'cylinder-bessel-direct': 'Q_sca against the Bessel boundary equations solved directly, radius swept',
    'cylinder-index-matched': "a cylinder of its host's index is no cylinder: the field is the plane wave",
    'treams-cylinder-oblique': 'efficiencies Q (cross width over the diameter) at 60° against treams',
    'cylinder-dipole-homogeneous': 'a matched cylinder is no cylinder: Γ/Γ₀ = 1 everywhere',
    'cylinder-coating-budget-electric': 'the total rate against escape + guided + absorbed',
    'cylinder-coating-budget-magnetic': 'the total rate against escape + guided + absorbed',
    'cylinder-guide-budget': 'the total rate against escape + guided, through a lossless guide',
    'cylinder-grazing-tolerance': 'asked for 1e-9, the grazing error is charged and reported, not hidden',
    'cylinder-grazing-caps-electric': 'the error charged at the grazing caps against moving the guard',
    'cylinder-grazing-caps-magnetic': 'the error charged at the grazing caps against moving the guard',
    'fibre-dispersion': "TE₀₁, TM₀₁ and HE₁₁ against the fibre's dispersion relations",
    'fibre-rates-cutoff': 'rates stay smooth through the TE₀₁/TM₀₁ cutoff',
    'meep-cylinder-fibre': "total decay rates against Meep's 3D FDTD (60 px/µm)",
    'ngsolve-cylinder-fibre': "total decay rates against NGSolve's 3D finite elements",
}

# The text beside a figure that computes no single setup: what both sides evaluate (Markdown, $…$ typeset at build).
ASIDES = {
    "kramers-kronig-closed-forms": (
        "PyStratify evaluates each model's ε₁ in closed form. The reference rebuilds it from ε₂ alone,\n\n"
        "$$\\varepsilon_1(E) = \\varepsilon_\\infty + \\frac{2}{\\pi}\\,\\mathrm{P}\\!\\int_0^\\infty \\frac{E'\\,\\varepsilon_2(E')}{E'^2 - E^2}\\,dE'$$\n\n"
        "by QUADPACK with a Cauchy weight, split at the kinks, independent of PyStratify's own transform.\n\n"
        "Tauc–Lorentz: $A$ = 120 eV, $E_0$ = 3.6 eV, $C$ = 2.4 eV, $E_g$ = 1.6 eV, at every energy. Cody–Lorentz: the same with $E_p$ = 0.8 eV, "
        "$E_t$ = 1.8 eV, $E_u$ = 0.05 eV, at every fourth."),
    "pyelli-tauc-lorentz": (
        "Both codes evaluate the Tauc–Lorentz model (Jellison & Modine), each from its own source:\n\n"
        "$$\\varepsilon_2(E) = \\frac{A E_0 C\\,(E - E_g)^2}{(E^2 - E_0^2)^2 + C^2 E^2}\\,\\frac{1}{E} \\quad (E > E_g)$$\n\n"
        "zero below the gap, and ε₁ by its closed-form Kramers–Kronig transform. $A$ = 120 eV, $E_0$ = 3.6 eV, $C$ = 2.4 eV, $E_g$ = 1.6 eV. "
        "Agreement checks the formulas and the parameter conventions together."),
    "pyelli-drude": (
        "PyStratify's free-electron metal,\n\n"
        "$$\\varepsilon(E) = -\\frac{E_p^2}{E^2 + i\\Gamma E},$$\n\n"
        "is negative below the plasma energy. pyElli's DrudeEnergy returns $A/(E^2 - i\\gamma E)$, whose real part is positive at every "
        "energy, so its curve is drawn as $-\\overline{\\varepsilon}$, as the test compares it; ε₂ needs no correction. "
        "$E_p$ = 8.9 eV, $\\Gamma$ = 0.07 eV."),
    "image-dipole-mirror": (
        "Given a perfect mirror, $r_s = -1$ and $r_p = 1$, the Sommerfeld integrals must reproduce the image dipole "
        "(Drexhage; Chance, Prock & Silbey), with $x = kd = 4\\pi h/\\lambda$:\n\n"
        "$$\\frac{\\Gamma_\\perp}{\\Gamma_0} = 1 + 3\\left(\\frac{\\sin x}{x^3} - \\frac{\\cos x}{x^2}\\right)$$\n\n"
        "$$\\frac{\\Gamma_\\parallel}{\\Gamma_0} = 1 - \\frac{3}{2}\\left(\\frac{\\sin x}{x} + \\frac{\\cos x}{x^2} - \\frac{\\sin x}{x^3}\\right)$$\n\n"
        "A magnetic dipole sees the mirror with the opposite sign, $2 - \\Gamma_\\parallel/\\Gamma_0$. This checks the reference every other "
        "film-emitter row relies on."),
    "quadrupole-coupling": (
        "A quadrupole couples to the vector spherical waves $\\mathbf{W} = \\mathbf{M} + \\mathbf{N}$ through\n\n"
        "$$\\tfrac{1}{6}\\sum_{ij} Q_{ij}\\,\\partial_j W_i,$$\n\n"
        "which PyStratify evaluates in closed form; the reference takes central differences of the same waves in 40-digit arithmetic. "
        "The axial (m = 0) family is drawn, 47 nm from the centre with k = 0.013 nm⁻¹: the regular wave fades with order, the "
        "outgoing one grows. The test covers all five families."),
    "casimir-non-retarded": (
        "Close to the surface an atom sees a flat, non-retarded half-space:\n\n"
        "$$U(d) \\to -\\frac{C_3}{d^3}, \\qquad C_3 = \\frac{\\hbar}{4\\pi}\\int_0^\\infty \\alpha(i\\xi)\\,\\frac{\\varepsilon(i\\xi) - 1}{\\varepsilon(i\\xi) + 1}\\,d\\xi$$\n\n"
        "PyStratify sums the sphere's full retarded response. The two meet within a nanometre (0.3 % at 0.5 nm) and part further "
        "out by design, as retardation and the sphere's curvature weaken the potential (37 % at 50 nm)."),
    "lossless-absorbs-nothing": (
        "Absorption is what extinction leaves after scattering,\n\n"
        "$$Q_\\mathrm{abs} = Q_\\mathrm{ext} - Q_\\mathrm{sca},$$\n\n"
        "so for spheres with no loss anything above rounding is energy the recursion invents. Eight shells of random real indices "
        "1.2–3.5 in water, at every size up to x = 300: it stays below the test's 10⁻¹², growing slowly with the orders summed."),
    "graphene-sphere-plasmon": (
        "A conducting sheet σ on a sphere of radius R shifts the resonance of order ℓ to where, in the quasi-static limit,\n\n"
        "$$\\mathrm{Re}\\left[\\varepsilon_1 \\ell + \\varepsilon_2 (\\ell + 1) + \\frac{i\\sigma\\,\\ell(\\ell+1)\\,\\lambda}{2\\pi R}\\right] = 0.$$\n\n"
        "PyStratify's full solution with the sheet (graphene, $E_F$ = 0.5 eV, ε = 2.1 inside, 1.77 outside) finds the peak of |T₁|; the two "
        "meet as $R \\to 0$ and part by retardation, within $(kR)^2$ as the test asserts."),
}
