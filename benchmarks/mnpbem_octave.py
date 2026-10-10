"""MNPBEM17 under GNU Octave: fetch, verify, build an Octave-ready copy, run (benchmarks/MATRIX.md).

    python benchmarks/mnpbem_octave.py        # fetches, builds, runs a 40 nm sphere as a smoke test

MNPBEM (Hohenester & Truegler, Comput. Phys. Commun. 183, 370 (2012); Hohenester, CPC 222, 209
(2018)) is a MATLAB boundary-element toolbox for plasmonic particles, GPL. Its author's download page
is gone; the archived MNPBEM17 is the supplementary program of the 2018 CPC paper, on Elsevier's
Mendeley Data (doi:10.17632/gbyj97hfnc.1). This script downloads that tarball into ``$MNPBEM_HOME``
(default ``~/.local/share/mnpbem``), checks its SHA-256 and builds from it ``MNPBEM17-octave``, a copy
that GNU Octave (11.3 tested; ``brew install octave``) runs; the tarball itself is never modified.
MATLAB is not needed. Every change is mechanical and listed here, so the physics is the archived
code's:

1. **Package classes leave their packages.** Octave does not see the method files of a class in a
   package's @-folder (``+shape/@tri``): ``shape.tri`` becomes the top-level class ``shape_tri`` and
   every reference ``shape.tri`` is rewritten (five classes: ``shape.tri``, ``shape.quad`` and three
   ``aca.compgreen*`` used only by hierarchical matrices).
2. **``builtin('subsref', obj, s)`` → ``octave_subsref(obj, s)``.** MATLAB hands every level of an
   indexing chain after the first to the overloaded ``subsref`` of the value it reaches
   (``p.eps{i}(enei)`` evaluates a dielectric function); Octave indexes it as an array.
3. **``@( eps ) ( eps( enei ) )`` → ``@( eps ) subsref( eps, substruct( '()', { enei } ) )``.**
   Octave asks an object's overloaded call syntax for one output inside an anonymous function;
   MNPBEM takes two (epsilon and the wavenumber).
4. **``cellfun`` → ``octave_cellfun``** in MNPBEM's own files: Octave refuses a complex value after a
   real one when assembling a uniform output; MATLAB promotes to complex.
5. **One line of ``bembase/find.m``** indexes a cell with an empty ``{}``; it takes the single entry.
6. **``superclasses``**, missing in Octave, is supplied from ``meta.class``.

The compiled MEX helpers (hierarchical matrices) are MATLAB binaries and unused: keep particles
small enough for full matrices (1.2 GB at 2044 faces).
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

URL = ("https://data.mendeley.com/public-files/datasets/gbyj97hfnc/files/"
       "f7a73b53-f7b8-4f89-b01e-75a062878291/file_downloaded")
SHA256 = "5a8bac7dfb3e632dff34c9aa04e25a34a92042d8bdf06f9c09bc770c92aa8155"
BUILD_VERSION = "2"  # bump when the transformations change
HOME = pathlib.Path(os.environ.get("MNPBEM_HOME", pathlib.Path.home() / ".local" / "share" / "mnpbem"))
OCTAVE = os.environ.get("OCTAVE", shutil.which("octave-cli") or shutil.which("octave") or "octave-cli")

COMPAT = {
    "superclasses.m": """function names = superclasses( name )
%  SUPERCLASSES - Octave stand-in for MATLAB's superclasses: every ancestor of class NAME.
names = {};
mc = meta.class.fromName( name );
if isempty( mc ),  return;  end
list = mc.SuperclassList;
for k = 1 : numel( list )
  parent = list{ k }.Name;
  names = [ names; { parent }; superclasses( parent ) ];
end
names = unique( names, 'stable' );
""",
    "octave_subsref.m": """function varargout = octave_subsref( obj, s )
%  OCTAVE_SUBSREF - builtin subsref whose later levels reach the overloads of the values they index.
%    MATLAB's builtin( 'subsref', obj, s ) hands every level after the first to the subsref of
%    the value it reaches (p.eps{ i }( enei ) evaluates the dielectric function); Octave indexes
%    those levels as arrays.  A method call obj.name( args ) stays one level.
nout = max( 1, nargout );
first = 1;
if strcmp( s( 1 ).type, '.' ) && numel( s ) > 1 && strcmp( s( 2 ).type, '()' ) ...
    && isobject( obj ) && ismethod( obj, s( 1 ).subs )
  first = 2;
end
if numel( s ) == first
  [ varargout{ 1 : nout } ] = builtin( 'subsref', obj, s );
  return
end
val = builtin( 'subsref', obj, s( 1 : first ) );
for k = first + 1 : numel( s ) - 1
  val = subsref( val, s( k ) );
end
[ varargout{ 1 : nout } ] = subsref( val, s( end ) );
""",
    "octave_cellfun.m": """function varargout = octave_cellfun( fun, varargin )
%  OCTAVE_CELLFUN - cellfun, uniform outputs that may mix real and complex values as in MATLAB.
%    Octave refuses a complex value after a real one when UniformOutput is true; MATLAB promotes.
legacy = ischar( fun ) && any( strcmp( fun, { 'isempty', 'islogical', 'isnumeric', 'isreal', ...
                                               'length', 'ndims', 'prodofsize', 'size', 'isclass' } ) );
%  every argument after the function is a cell array except option names (matched by prefix)
options = any( cellfun( @ischar, varargin ) );
nout = max( 1, nargout );
if legacy || options
  [ varargout{ 1 : nout } ] = cellfun( fun, varargin{ : } );
  return
end
[ out{ 1 : nout } ] = cellfun( fun, varargin{ : }, 'UniformOutput', false );
for k = 1 : nout
  if isempty( out{ k } )
    varargout{ k } = zeros( size( out{ k } ) );
  else
    varargout{ k } = reshape( [ out{ k }{ : } ], size( out{ k } ) );
  end
end
""",
}

LINE_PATCHES = [
    ("Base/@bembase/find.m",
     "fname = subsref( fieldnames( needs{ j } ), substruct( '{}', {} ) );",
     "fname = fieldnames( needs{ j } );  fname = fname{ 1 };"),
]


def fetch():
    """The archived tarball, downloaded once and checked against its SHA-256."""
    HOME.mkdir(parents=True, exist_ok=True)
    archive = HOME / "MNPBEM17.tar.gz"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=120) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{archive}: SHA-256 {digest} is not the archived MNPBEM17 ({SHA256})")
    return archive


def build():
    """Path of the Octave-ready MNPBEM17, built from the verified tarball if it is not current."""
    target = HOME / "MNPBEM17-octave"
    stamp = target / ".build"
    if stamp.exists() and stamp.read_text() == f"{SHA256} {BUILD_VERSION}":
        return target
    archive = fetch()
    if target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory() as scratch:
        with tarfile.open(archive) as tar:
            tar.extractall(scratch, filter="data")
        shutil.copytree(pathlib.Path(scratch) / "MNPBEM17", target)
    _transform(target)
    stamp.write_text(f"{SHA256} {BUILD_VERSION}")
    return target


def _rewrite(root, pattern, replacement):
    for path in root.rglob("*.m"):
        if path.parent.name == "octave_compat":
            continue
        text = path.read_text(errors="surrogateescape")
        new = re.sub(pattern, replacement, text)
        if new != text:
            path.write_text(new, errors="surrogateescape")


def _transform(root):
    moved = []
    for folder in sorted(root.rglob("@*")):
        if folder.is_dir() and folder.parent.name.startswith("+"):
            package, name = folder.parent.name[1:], folder.name[1:]
            destination = folder.parent.parent / f"@{package}_{name}"
            shutil.move(str(folder), str(destination))
            constructor = destination / f"{name}.m"
            text = constructor.read_text(errors="surrogateescape")
            text = re.sub(rf"^(\s*classdef\s+){name}\b", rf"\g<1>{package}_{name}", text, flags=re.M)
            text = re.sub(rf"(function\s+obj\s*=\s*){name}\s*\(", rf"\g<1>{package}_{name}(", text)
            (destination / f"{package}_{name}.m").write_text(text, errors="surrogateescape")
            constructor.unlink()
            moved.append(f"{package}.{name}")
    _rewrite(root, r"\b(" + "|".join(re.escape(m) for m in moved) + r")\b", lambda m: m.group(1).replace(".", "_"))
    _rewrite(root, r"builtin\(\s*'subsref'\s*,\s*", "octave_subsref( ")
    _rewrite(root, r"@\(\s*(\w+)\s*\)\s*\(\s*\1\(\s*enei\s*\)\s*\)",
             lambda m: f"@( {m.group(1)} ) subsref( {m.group(1)}, substruct( '()', {{ enei }} ) )")
    _rewrite(root, r"(?<![\w.])cellfun\(", "octave_cellfun(")
    for relative, old, new in LINE_PATCHES:
        path = root / relative
        text = path.read_text(errors="surrogateescape")
        if text.count(old) != 1:
            raise RuntimeError(f"{relative}: the line to patch is not unique or absent")
        path.write_text(text.replace(old, new), errors="surrogateescape")
    compat = root / "octave_compat"
    compat.mkdir()
    for name, text in COMPAT.items():
        (compat / name).write_text(text)


def run(script, timeout=3600):
    """Run MATLAB-language ``script`` with MNPBEM on the path; returns Octave's standard output."""
    root = build()
    preamble = f"addpath( genpath( '{root}' ) ); warning( 'off', 'all' );\n"
    with tempfile.NamedTemporaryFile("w", suffix=".m", delete=False) as handle:
        handle.write(preamble + script)
    try:
        result = subprocess.run([OCTAVE, "--no-gui", "--quiet", handle.name], capture_output=True, text=True,
                                timeout=timeout)
    finally:
        os.unlink(handle.name)
    if result.returncode != 0:
        raise RuntimeError(f"Octave failed:\n{result.stderr[-2000:]}")
    return result.stdout


if __name__ == "__main__":
    print(run("""
op = bemoptions( 'sim', 'ret', 'interp', 'curv', 'waitbar', 0 );
p = comparticle( { epsconst( 1.33 ^ 2 ), epsconst( -10 + 1i ) }, { trisphere( 144, 40 ) }, [ 2, 1 ], 1, op );
exc = planewave( [ 1, 0, 0 ], [ 0, 0, 1 ], op );
sig = bemsolver( p, op ) \\ exc( p, 600 );
printf( '40 nm sphere, eps -10+1i in water, 600 nm: C_ext %.6f nm^2, C_sca %.6f nm^2\\n', ...
        exc.ext( sig ), exc.sca( sig ) );
"""))
