"""The multilayer.app content manifest, built from this repository's ``docs/``.

PyStratify is the single source of the site's reference content: the Learn sections
(``docs/learn/``), the classic papers (``docs/classics/``) and their worked examples
(``docs/examples/``). ``python -m pystratify.docs build`` reads them, computes every example
with PyStratify, attaches each classic's checks (``crosscheck/classics/results/``) and writes
``docs/manifest.json``, which the website builds its pages from at a pinned commit.
``docs/README.md`` describes the layout and the record schema.

Needs PyYAML (``pip install "pystratify[docs]"``). Not part of the numerical API.
"""

from .build import SCHEMA, build_manifest, material, write_manifest

__all__ = ["SCHEMA", "build_manifest", "material", "write_manifest"]
