"""Run the original STRATIFY MATLAB code in GNU Octave (test helper)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
STRATIFY = ROOT / "reference" / "stratify-matlab"
COMPAT = Path(__file__).resolve().parent / "octave_compat"

OCTAVE = shutil.which("octave-cli") or shutil.which("octave")


def available() -> bool:
    return OCTAVE is not None and STRATIFY.is_dir()


def mat(v) -> str:
    """Python value -> Octave literal."""
    a = np.atleast_1d(np.asarray(v))
    if np.iscomplexobj(a):
        items = [f"complex({float(x.real):.17g},{float(x.imag):.17g})" for x in a.ravel()]
    else:
        items = [f"{float(x):.17g}" for x in a.ravel()]
    return "[" + ",".join(items) + "]"


def run(body: str, outputs: list[str], patch: dict[str, str] | None = None) -> dict:
    """Execute ``body`` with STRATIFY on the path and return ``outputs``.

    ``patch`` maps a STRATIFY-relative .m path to replacement source, applied
    in a temporary copy (used to confirm the root cause of a deviation).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = STRATIFY
        if patch:
            src = tmp / "stratify"
            shutil.copytree(STRATIFY, src)
            for rel, text in patch.items():
                (src / rel).write_text(text)
        out = tmp / "out.mat"
        script = (
            f"addpath('{COMPAT}'); addpath(genpath('{src}'));\n"
            + body
            + f"\nsave('-mat7-binary','{out}',{','.join(repr(o) for o in outputs)});\n"
        )
        (tmp / "stratify_job.m").write_text(script)
        res = subprocess.run(
            [OCTAVE, "-q", "--no-gui", str(tmp / "stratify_job.m")], capture_output=True, text=True, cwd=tmp
        )
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"octave failed:\n{res.stderr[-3000:]}")
        data = sio.loadmat(out, squeeze_me=True, struct_as_record=False)
        return {k: data[k] for k in outputs}
