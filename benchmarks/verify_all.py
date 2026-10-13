"""Re-run the whole verification study, every piece in its own environment, with nothing installed.

    python3 benchmarks/verify_all.py                    # classics, tests, figures, manifest (~1 h 15 min)
    python3 benchmarks/verify_all.py --quick            # the same without Meep and NGSolve (~25 min)
    python3 benchmarks/verify_all.py --reports          # also rewrite the benchmarks/*.md reports (~1 h more)
    python3 benchmarks/verify_all.py --steps tests figures --envs bench tmmax

Any Python 3 runs it (the standard library only). The environments are the five in benchmarks/envs/README.md, made
once by benchmarks/envs/setup.sh; a step whose environment is missing is reported, not installed. The steps, in order:

1. **classics** — every crosscheck/classics/*.py in `bench`: the papers recomputed, their checks, and the
   results/*.json and *.csv the paper figures read (git-ignored, so a fresh checkout needs this step first);
2. **tests** — the full pytest suite in `bench`, then the tests that need smuthi, tmmax, NGSolve and Meep in theirs;
3. **figures** — benchmarks/figures.py --env <env> for each environment: every docs/figures/<name>.json;
4. **reports** (only with --reports) — each benchmark script's printed report into its benchmarks/<NAME>.md;
5. **manifest** — python -m pystratify.docs build, then --check: docs/manifest.json carries the figures.

Heavy jobs run one at a time with single-threaded BLAS (an 8 GB Mac). Each step's output goes to
benchmarks/runs/<time>/<step>.log (git-ignored) and a summary to summary.md beside it; the exit status is 1 if any
step failed. Afterwards `git diff --stat docs/figures` shows which figures moved: commit them with the change that
moved them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("PYSTRATIFY_BENCH_HOME", Path.home() / ".multilayer" / "bench"))
ENVS = ("bench", "tmmax", "smuthi", "ngsolve", "meep")  # cheap first; Meep (the longest) last
HEAVY = ("ngsolve", "meep")
# the tests each environment adds (bench runs the whole suite; the others' tests skip there)
TESTS = {
    "bench": [],
    "smuthi": ["tests/test_smuthi_spheres.py", "tests/test_planar_emission_references.py", "-k", "smuthi"],
    "tmmax": ["tests/test_tmmax_films.py"],
    "ngsolve": ["tests/test_fem_ngsolve.py"],
    "meep": ["tests/test_fdtd_meep.py"],
}
# script, environment, report (each prints its report; the old file is kept if the script fails)
REPORTS = [
    ("planar_checks", "bench", "PLANAR_CHECKS.md"), ("unified_checks", "bench", "UNIFIED_CHECKS.md"),
    ("focused_checks", "bench", "FOCUSED_CHECKS.md"), ("sphere_checks", "bench", "SPHERE_CHECKS.md"),
    ("sphere_emitter_checks", "bench", "SPHERE_EMITTER_CHECKS.md"), ("planar_codes", "bench", "PLANAR_CODES.md"),
    ("mie_codes", "bench", "MIE_CODES.md"), ("gdm_spheres", "bench", "GDM_SPHERES.md"),
    ("mnpbem_spheres", "bench", "MNPBEM_SPHERES.md"), ("tmmax_films", "tmmax", "TMMAX_FILMS.md"),
    ("emitter_references", "smuthi", "EMITTER_REFERENCES.md"), ("smuthi_spheres", "smuthi", "SMUTHI_SPHERES.md"),
    ("fem_ngsolve", "ngsolve", "FEM_NGSOLVE.md"), ("fdtd_meep", "meep", "FDTD_MEEP.md"),
]
ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def python(env):
    return HOME / env / "bin" / "python"


class Run:
    def __init__(self, folder):
        self.folder = folder
        self.rows = []

    def step(self, name, env, argv, cwd=ROOT, out=None, ok=lambda text: True, extra_env=None):
        """Run argv in env; the log keeps stdout and stderr, `out` (a path) receives stdout alone on success."""
        if not python(env).exists():
            self.rows.append((name, env, "missing", 0.0, f"no {python(env)}: see benchmarks/envs/README.md"))
            print(f"{name} [{env}]: MISSING environment", flush=True)
            return False
        log = self.folder / f"{name.replace('/', '_').replace(' ', '_')}.log"
        print(f"{name} [{env}] ...", end=" ", flush=True)
        start = time.time()
        env_vars = {**os.environ, **ONE_THREAD, "PYTHONUNBUFFERED": "1", **(extra_env or {})}
        result = subprocess.run([str(python(env)), *argv], cwd=cwd, env=env_vars, capture_output=True, text=True)
        seconds = time.time() - start
        log.write_text(f"$ {python(env)} {' '.join(argv)}\n\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}")
        passed = result.returncode == 0 and ok(result.stdout)
        if passed and out is not None:
            out.write_text(result.stdout)
        tail = next((line for line in reversed(result.stdout.strip().splitlines())
                     if line.strip() and not line.startswith("Elapsed run time")), "")  # Meep prints its timer last
        self.rows.append((name, env, "ok" if passed else "FAILED", seconds, tail[:160]))
        print(f"{'ok' if passed else 'FAILED'} ({seconds:.0f} s) {tail[:100]}", flush=True)
        return passed

    def summary(self):
        lines = ["| step | environment | result | time | last line |", "|---|---|---|---|---|"]
        lines += [f"| {n} | {e} | {r} | {s:.0f} s | {t.replace('|', '/')} |" for n, e, r, s, t in self.rows]
        text = f"# verify_all.py, {self.folder.name}\n\n" + "\n".join(lines) + "\n"
        (self.folder / "summary.md").write_text(text)
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", nargs="+", choices=("classics", "tests", "figures", "reports", "manifest"),
                        default=["classics", "tests", "figures", "manifest"])
    parser.add_argument("--envs", nargs="+", choices=ENVS, default=list(ENVS))
    parser.add_argument("--quick", action="store_true", help="leave out Meep and NGSolve (the long, heavy jobs)")
    parser.add_argument("--reports", action="store_true", help="also rewrite the benchmarks/*.md reports")
    args = parser.parse_args()
    steps = list(args.steps) + (["reports"] if args.reports and "reports" not in args.steps else [])
    envs = [e for e in args.envs if not (args.quick and e in HEAVY)]
    folder = ROOT / "benchmarks" / "runs" / time.strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True)
    run = Run(folder)
    print(f"environments in {HOME}; logs in {folder.relative_to(ROOT)}", flush=True)
    if not (shutil.which("octave-cli") or shutil.which("octave")):
        print("note: GNU Octave not found, so the MNPBEM figures are skipped (brew install octave)", flush=True)

    if "classics" in steps and "bench" in envs:
        classics = ROOT / "crosscheck" / "classics"
        for script in sorted(classics.glob("[!_]*.py")):
            run.step(f"classics/{script.stem}", "bench", [script.name], cwd=classics,
                     ok=lambda text: text.strip().splitlines()[-1:] == ["PASS"])
    if "tests" in steps:
        for env in envs:
            run.step(f"tests ({env})", env, ["-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS[env]])
    if "figures" in steps:
        for env in envs:  # a figure whose package is missing prints "skip": that is a failure here
            run.step(f"figures ({env})", env, ["benchmarks/figures.py", "--env", env],
                     ok=lambda text: not any(line.startswith("skip ") for line in text.splitlines()))
    if "reports" in steps:
        for script, env, report in REPORTS:
            if env in envs:
                run.step(f"report {report}", env, [f"benchmarks/{script}.py"], out=ROOT / "benchmarks" / report)
    if "manifest" in steps and "bench" in envs:
        run.step("manifest", "bench", ["-m", "pystratify.docs", "build"])
        run.step("manifest --check", "bench", ["-m", "pystratify.docs", "build", "--check"])

    print("\n" + run.summary(), flush=True)
    changed = subprocess.run(["git", "status", "--short", "docs/figures", "docs/manifest.json", "benchmarks"], cwd=ROOT,
                             capture_output=True, text=True).stdout
    print("changed files:\n" + (changed or "  none\n"), flush=True)
    sys.exit(0 if all(r[2] == "ok" for r in run.rows) else 1)


if __name__ == "__main__":
    main()
