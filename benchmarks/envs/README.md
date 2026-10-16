# Benchmark environments

The verification study (`benchmarks/MATRIX.md`, the figures in `docs/figures/`) compares PyStratify with other codes
that cannot share one Python: smuthi builds only on Python 3.10 with NumPy < 2, Meep is a conda build, tmmax brings
JAX. Five environments hold them; they are made **once** and then only used:

| environment | Python | holds | used for |
|---|---|---|---|
| `bench` | 3.13 (venv) | PyStratify, treams, scattnlay, pyGDM2, PyMoosh, pyElli, lmfit, meent, pytest | the full test suite, 108 figures, the classics, most reports; it also drives GNU Octave for MNPBEM |
| `smuthi` | 3.10 (conda) | smuthi 2.2.4 (built from source), NumPy < 2 | 5 figures, `EMITTER_REFERENCES.md`, `SMUTHI_SPHERES.md`, the smuthi tests |
| `meep` | 3.12 (conda-forge) | Meep 1.35, MPI-free | 3 figures, `FDTD_MEEP.md`, `tests/test_fdtd_meep.py` |
| `ngsolve` | 3.13 (venv) | NGSolve 6.2 | 1 figure, `FEM_NGSOLVE.md`, `tests/test_fem_ngsolve.py` |
| `tmmax` | 3.13 (venv) | tmmax 1.1.4, JAX | 2 figures, `TMMAX_FILMS.md`, `tests/test_tmmax_films.py` |

They live in `~/.multilayer/bench/<name>` (`PYSTRATIFY_BENCH_HOME` moves them), outside the repository and out of
temporary folders. Each has PyStratify installed **editable** from this checkout, so a change to `src/` is seen
everywhere at once, with nothing reinstalled.

- **Pinned versions:** `<name>.txt` (`pip freeze` of a venv), `<name>.conda.txt` (`conda list --explicit`, osx-arm64)
  and `<name>.pip.txt` (the pip packages inside a conda env).
- **Create the missing ones:** `sh benchmarks/envs/setup.sh` (an existing environment is left alone). smuthi needs a
  Fortran compiler and an OpenMP C compiler (`gcc-16` from Homebrew; `SMUTHI_CC` names another); MNPBEM needs
  `brew install octave`, after which the script builds the toolbox into `~/.local/share/mnpbem` (`mnpbem_octave.py`).
- **Run the study:** `python3 benchmarks/verify_all.py` (see its docstring); it never installs anything and stops a
  step whose environment is missing, naming this file.
- **Refresh the pins** after deliberately upgrading a package: `<env>/bin/python -m pip freeze --exclude-editable`
  (and `conda list -p <env> --explicit`), then commit the files with the figures they produced.

On an 8 GB Mac run one heavy job at a time (the runner does): the Meep figures take ~25 minutes and 0.5 GB, the
NGSolve figure ~8 minutes and ~1.6 GB.
