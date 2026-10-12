---
id: textured-surfaces
title: "Textured surfaces: pyramids and light trapping"
geometry: films
chapter: Graded and textured surfaces
topic: texture
sources: [plane_wave]
summary: >-
  Upright pyramids on silicon give reflected light a second chance: the ray geometry of a facet, the product of two
  reflectances and what a coating on the facets adds, the oblique path inside the wafer, the Lambertian limit of
  4n² from étendue and the Yablonovitch argument, how pyramids and random textures approach it, where the ray
  picture stops (feature size against wavelength), and why this needs ray tracing beside the transfer matrix.
examples: [pyramid-facet]
references:
- cite: P. Campbell and M. A. Green, J. Appl. Phys. 62, 243 (1987)
  doi: 10.1063/1.339189
- cite: E. Yablonovitch and G. D. Cody, IEEE Trans. Electron Devices 29, 300 (1982)
  doi: 10.1109/T-ED.1982.20700
- cite: E. Yablonovitch, J. Opt. Soc. Am. 72, 899 (1982)
  doi: 10.1364/JOSA.72.000899
- cite: M. A. Green, Prog. Photovolt. 10, 235 (2002)
  doi: 10.1002/pip.404
- cite: M. A. Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008) (silicon optical constants)
  doi: 10.1016/j.solmat.2008.06.009
---

Anisotropic etching of a (100) silicon wafer in hot alkali exposes the slowly etching (111) planes and leaves a field
of upright pyramids whose facets make $\alpha = 54.74°$ (the angle between (100) and (111)) with the wafer plane. The
pyramids are several micrometres across, large compared with the wavelength, so the texture does not change the
Fresnel coefficients of each facet; it changes where a reflected ray goes next, and what happens to a ray once it is
inside. Both effects are geometric, and both are why every crystalline silicon solar cell is textured.

## Two chances to enter

A vertical ray meets a facet at the local angle of incidence $\theta_1 = \alpha$. The specularly reflected ray leaves
rotated by $2\alpha$ from the vertical; for $\alpha > 45°$ it still points downward and strikes the opposite facet, at

$$
\theta_2 = |180° - 3\alpha| = 15.8° .
$$

Light escapes only if it is reflected both times, so the front-surface reflectance becomes the product

$$
R_\mathrm{eff} = R(\theta_1)\, R(\theta_2) ,
$$

and bare silicon's 35 % at 55° times 32 % at 16° gives 11 %, against 35 % for a polished wafer. Each factor is the
reflectance of the coated silicon surface at that angle, which is a film calculation: with a 75 nm silicon nitride
layer, a quarter wave at 600 nm for the facet's angle, the factors fall to a few percent each and their product well
below one percent. The coating must be designed for the facet's incidence angle, not for normal incidence, and for
the mix of s and p that the two reflections produce (the first reflection partly polarizes the light, so the second
sees a polarization-weighted average). Rays that hit near a pyramid's base can escape after one reflection, and the
fraction that does depends on the exact geometry; a hemispherical reflectance measurement of a textured wafer sums
all of this and is what the geometric model is checked against.

::: example pyramid-facet

## Trapping inside

Once in the silicon, refracted rays travel obliquely, by $\alpha - \theta_t$ from the vertical, about 41° for the
first refraction. Weakly absorbed near-infrared light (silicon's absorption length is 100 µm at 1000 nm and a
millimetre at 1100 nm, against a 150 µm wafer) reaches the back at an angle, reflects from the back contact or back
mirror, and meets the front facets beyond their critical angle of 16°, so it is reflected back in and bounces many
times before an escape cone lets it out. The path length is multiplied, and with it the absorption in the
long-wavelength tail that sets a silicon cell's current.

## The Lambertian limit: 4n²

How much can any texture achieve? Yablonovitch (1982) answered with a thermodynamic argument. A surface that
randomises the direction of light completely (a **Lambertian** surface) fills the whole solid angle inside the slab
with radiation of uniform brightness. The density of optical modes inside a medium of index $n$ is $n^2$ times that
in air, and only the modes within the escape cone, a fraction $1/n^2$ of the hemisphere's solid angle (by Snell's law
and $\sin^2\theta_c = 1/n^2$), can leave through the front. The mean path a ray travels before it escapes is
therefore enhanced by $2n^2$ relative to a single vertical pass if the back is a mirror, and by $4n^2$ once the
double pass between front and back is counted:

$$
\langle \ell \rangle = 4 n^2\, d \approx 50\, d
$$

for silicon ($n \approx 3.5$ in the near infrared). The argument is one of étendue, or equivalently of detailed
balance between the radiation inside and outside the slab, and the limit holds for any geometry that randomises
without absorbing; it is the yardstick against which every light-trapping scheme, from pyramids to nanophotonic
gratings, is measured, and in the weakly absorbing limit the absorption of a Lambertian slab is

$$
A \approx \frac{\alpha d\,\cdot 4n^2}{1 + \alpha d\,\cdot 4n^2}
$$

(Yablonovitch and Cody 1982). Pyramids come close to it: Campbell and Green (1987) traced rays through regular and
inverted pyramids and found enhancements within a factor of two of the limit, with inverted pyramids and random
"tiler's" patterns doing best, because a regular array returns some rays to their entry facet.

## Where the ray picture stops

The geometric treatment holds while the features are many wavelengths across and the surfaces between them are
locally flat. A texture of a few hundred nanometres, such as the random "black silicon" of
[the moth-eye section](learn/films/moth-eye-antireflection.html) or a periodic nanograting, is neither: it
diffracts, and its light trapping is a question of coupling into the guided modes of the wafer, which can exceed
$4n^2$ over a narrow band and falls short over a broad one. Between the two regimes, pyramids of a micrometre,
nothing simple applies and a rigorous solver is needed. Following rays through the whole wafer, with its coatings and
its back reflector, also needs **ray tracing** rather than a transfer matrix; the calculator's surface textures do it
with RayFlare, a separate engine that applies the transfer-matrix reflectances of the coated facets to each ray and
counts where it goes. The per-facet reflectance of the example above is the ingredient of that calculation, and the
check that it is right.
