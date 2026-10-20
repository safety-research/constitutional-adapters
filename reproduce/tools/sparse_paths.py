#!/usr/bin/env python3
"""reproduce/tools/sparse_paths.py — the directories a lighter checkout needs, one per line, for `git sparse-checkout add`.

    uv run --no-project python reproduce/tools/sparse_paths.py figures   # regenerate the paper's figures from the snapshot
    uv run --no-project python reproduce/tools/sparse_paths.py demo      # the Qwen3.5-4B objects at the demo's one training seed + the code

Stdlib only, on any Python 3, inside a sparse clone that holds the root files and `reproduce/tools` — the state after the
README's first two commands — before any `uv sync`. The README runs it through `uv run --no-project`, so the clone needs
no Python of its own: uv supplies the one `.python-version` names, fetching it when the machine has none, with nothing
else installed. Cone-mode sparse checkout takes directories, and brings the files directly inside each parent directory
with them: `objects/<model_id>/<object_id>` also fetches `objects/MANIFEST.json`, `objects/WITHHELD.json`,
`objects/DEMO.json` and `objects/README.md`, and the root files are always present.

`figures` is a fixed list: the snapshot and the notebook, the `sisteer` package the notebook imports (`uv sync` installs it
from `src/`), the stage script with its library and overrides, and the uv workspace member `corpus_generator/` (its
`pyproject.toml` is what `uv sync --frozen` may ask for). `demo` reads the object directories from `objects/DEMO.json`
— the work tree's copy when it is checked out, else the committed one through `git show HEAD:objects/DEMO.json`, which
a partial clone fetches as a single blob — and adds the code, the models table's reader (`reproduce/lib`, which
`sisteer.objects.serve_checkpoint` reads the served checkpoint's pin through; the table itself comes with `reproduce/tools`)
and the two system-instruction texts. Both lists end with `LICENSES`, the full licence texts THIRD_PARTY_NOTICES cites
for the third-party files and fitted objects a lighter checkout holds: cone mode brings the root files, THIRD_PARTY_NOTICES
and LICENSE among them, but no directory the list leaves out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEMO_LIST = os.path.join("objects", "DEMO.json")

#: Everything `reproduce/10_figures.sh --source snapshot` (or the nbconvert route) reads, plus what `uv sync --frozen
#: --extra cpu --group notebook` needs beyond the root files.
FIGURES = ("paper/snapshot", "paper/notebooks", "src", "reproduce", "corpus_generator")
#: The code and texts beside the demo's object directories: `src` (sisteer), the models table's reader (`reproduce/lib`,
#: under 1 MB; `serve_checkpoint` loads it by path), the uv workspace member, the two prompted baselines, which are text
#: and always come along.
DEMO_CODE = ("src", "reproduce/lib", "corpus_generator", "system_instructions")
#: The licence texts THIRD_PARTY_NOTICES points at (the Apache 2.0 text for the Qwen3.5 objects, the MIT texts for the
#: third-party files in `src`), which every lighter checkout carries along with the notices.
LICENCE_TEXTS = ("LICENSES",)


def demo_list(root: str) -> dict:
    """objects/DEMO.json: the checked-out file, else the committed one (`git show HEAD:objects/DEMO.json`)."""
    path = os.path.join(root, DEMO_LIST)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    r = subprocess.run(["git", "-C", root, "show", f"HEAD:{DEMO_LIST.replace(os.sep, '/')}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"REFUSED: {DEMO_LIST} is neither checked out under {root} nor readable from HEAD "
                 f"(git show: {r.stderr.strip() or 'no output'})")
    return json.loads(r.stdout)


def paths(option: str, root: str = REPO) -> list[str]:
    if option == "figures":
        return list(FIGURES) + list(LICENCE_TEXTS)
    if option == "demo":
        doc = demo_list(root)
        dirs = [f"objects/{o['dir']}" for o in doc["objects"]]
        if not dirs:
            sys.exit(f"REFUSED: {DEMO_LIST} lists no objects")
        return dirs + list(DEMO_CODE) + list(LICENCE_TEXTS)
    raise ValueError(option)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uv run --no-project python reproduce/tools/sparse_paths.py",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("option", choices=("figures", "demo"), help="which lighter checkout")
    ap.add_argument("--root", default=REPO, help="the clone's root (default: this file's repository)")
    a = ap.parse_args(argv)
    for p in paths(a.option, a.root):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
