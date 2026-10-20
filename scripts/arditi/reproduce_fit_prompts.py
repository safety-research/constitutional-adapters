#!/usr/bin/env python
"""Re-create the paper's refusal-direction fit prompts byte for byte from the public upstream splits.

The instructions themselves are not distributed with this repository; ``data/arditi/fit_prompts.manifest.json``
records their identity (upstream andyrdt/refusal_direction at a pinned commit, the four split files and their
sha256, seed 42, the draw order and sizes, each set's sha256 and the document's sha256). This fetches the split
files, checks them, replays upstream's sampler and checks every set and the whole document against the manifest.

    python scripts/arditi/reproduce_fit_prompts.py fetch                   # into the third-party cache
    python scripts/arditi/reproduce_fit_prompts.py reproduce --out FILE    # and a copy at FILE
    python scripts/arditi/reproduce_fit_prompts.py verify FILE             # check a copy, by hash only

The same as ``python -m sisteer.data.arditi_fit_prompts``, whose docstring describes the cache and the offline mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sisteer.data.arditi_fit_prompts import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
