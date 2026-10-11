# docs/ — the content multilayer.app is built from

PyStratify is the single source of the site's reference content: the **Learn** sections (theory, read in order
like a textbook), the **classic papers** reproduced, and the worked **examples** both open inline. Every number
and curve on those pages comes from PyStratify. `docs/API.md` (the 1.0 API contract) also lives here and is not
part of the site.

```
docs/
  learn/<geometry>/<NN-chapter>/<NN-section>.md   Markdown + LaTeX, YAML front matter    geometry: films | cylinders | spheres | common
  examples/<id>.yaml                               one setup, computed at build time
  classics/<id>.yaml                               one classic paper's page record
  materials.yaml                                   the refractiveindex.info pages the records cite
  verification.md                                  the Verification page's introduction; rows that open an example
  manifest.json                                    generated: everything above, computed (commit it)
```

Build, from the repository root:

```bash
pip install -e ".[docs]"
python -m pystratify.docs build                  # writes docs/manifest.json
python -m pystratify.docs build --run-classics   # re-runs crosscheck/classics/ first (minutes)
python -m pystratify.docs build --check          # exit 1 if docs/manifest.json is stale
```

The classics' checks come from `crosscheck/classics/results/<script>.json` (git-ignored; run the scripts, or
`--run-classics`). `tests/test_docs_build.py` fails when `docs/` and `docs/manifest.json` disagree.

The website (`multilayer_web/tools/content/build-content.py`) reads `docs/manifest.json` at a commit it pins
and writes `classics.html` and `learn/<geometry>/<slug>.html`. Equations are typeset there, at build time.

## Records (manifest schema 1)

Every record has `id`, `kind` (`learn`, `classic` or `benchmark`) and `geometry`; the site's geometry switch shows a geometry's
records (and `common` Learn chapters for every geometry).

**Learn section** — front matter, then Markdown with `$inline$` and `$$display$$` LaTeX:

| field | |
|---|---|
| `id`, `title`, `geometry`, `summary` | `geometry` must match the folder |
| `chapter` | the chapter's title (default: from the folder name) |
| `topic`, `sources` | e.g. `mie`; `[plane_wave, dipole, focused]` |
| `examples` | ids under `examples/`, each placed with a line `::: example <id>` |
| `references` | list of `{cite, doi}` or `{cite, url}` |
| `status` | `published` (default) or `draft`: deployed, but not indexed or linked until reviewed |

Order comes from the `NN-` prefixes; the build adds `chapter_order`, `order`, `slug`, `url`, `prev`, `next` and
the `body`.

**Example** (`examples/<id>.yaml`): `id`, `geometry`, `source`, `title`, `caption`; `plot` (`x_label`,
`y_label`, `series` of `{key, label}`); `params`, the sliders, each `{id, label, unit, target, scale, min, max,
step, value}` where `target` is a path into the calculator setup (e.g. `shells.0.thickness`, in µm, so
`scale: 0.001` for a slider in nm); `checks`, each `{check, reference, tolerance, source, measure}` with
`measure: {peak: <series>}`; and `calculator`, the setup (below). The build adds `data` (the curves at the
sliders' values, six significant digits), `code` (the example as a PyStratify script) and the checks'
`computed` and `pass`. A slider's `target` is a path into the `calculator` block (`film.layers.0.thickness`,
`film.layers.0.layers.1.n` for a layer inside a `repeat`, `film.substrate.n`, `sphere.shells.0.thickness`); its value
times `scale` must equal what the block holds, and it may not set the index of a medium taken from a database or
table. Measures: `peak: <series>` and `minimum: <series>` (the x of the extremum, optionally `within: [lo, hi]`),
`at: [<series>, x]` (the value at a grid point) and `crosses: [<series>, level]` (the first x reaching the level).

Computed so far, under a plane wave:

- spheres: `q_ext`, `q_sca`, `q_abs` and the multipole parts of `q_sca`, `md_sca`/`ed_sca` (dipoles) and
  `mq_sca`/`eq_sca` (quadrupoles), against wavelength;
- films: `R`, `T`, `A` (unpolarized) and `R_s`, `R_p`, `T_s`, `T_p`, against wavelength (`axis: wavelength`, the
  default, at `controls.angle-deg`) or angle (`axis: angle`, at `controls.wavelength-start`), with
  `pystratify.planar`, the engine the API's film `/calculate` runs.

**Classic** (`classics/<id>.yaml`): `id`, `position` (order on the page), `geometry`, `title`, `cite`, `doi`,
`figure`, `shows`, `inputs`, `agreement`, `perturb`, `script` and `results` (under `crosscheck/classics/`),
`calculator`. The build adds `checks`.

**Benchmark** — one per row of the geometry tables of `benchmarks/MATRIX.md`, which stays the one place the
matrix is written: `id` (`<geometry>-<source>-<row>`), `geometry`, `source` (`plane_wave`, `dipole`, `focused`),
`reference`, `reference_kind` (the matrix's *kind* column), `quantity`, `script`, `tolerance`, `status` (cells keep
their Markdown), and `example` where `verification.md` links the row to one. The matrix's legends, coverage grid and
"What the matrix shows" travel in the manifest's `verification` block with that page's introduction. A row added to
MATRIX.md appears on the site at the next build; `tests/test_docs_build.py` checks that every row arrives once.

**Calculator setup** — exactly one of:

- `film: {incident, layers, substrate, controls?, toggles?, view?, units?}`
- `sphere: {shells, host, start, end, step, changes?}` (shells inside out; `changes` overrides the sphere setup)
- `cylinder: {shells, host}`

and optionally `envelope: {…}`, the source and analysis settings of the calculator's v7 share envelope
(`kind`, `start`, `end`, `step`, `multipoles`, …; anything not given keeps the calculator's default). A medium
is `{label, n, k}` plus, optionally, `rii: <key in materials.yaml>` or `table: {name, rows | drude | lorentz}`;
a layer adds `thickness` (µm; a shell's or core's radius increment) and, in a film, `coherency: i`. A film's
`layers` may hold `{repeat: N, layers: [...]}`. The website turns the block into the share token that opens it
in the calculator.
