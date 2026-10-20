#!/usr/bin/env python3
"""The adapters whose emission moved between two trees -- the window's MATERIALISED adapter list
(`ADAPTERS="$(bumped_adapters.py <old> <new>)"`, never a typed list).

    bumped_adapters.py OLD NEW [--repo DIR] [--python PY] [--json]

``OLD`` and ``NEW`` are each a git revision of ``--repo`` (default: the repository this script lives in; ``src`` and ``configs`` -- the
adapters read their pins from ``configs/`` at construction -- are read with ``git archive``, so an uncommitted edit never counts) or a
DIRECTORY holding ``src/sisteer`` and ``configs`` (a checkout, a worktree, a copy). For each
tree the adapter registry is loaded IN A SUBPROCESS of ``--python`` (default: this interpreter) with that tree's ``src`` first on the path,
every adapter class is instantiated as the harvester instantiates it, and its emission version is read through the one function the
revision identity folds (``adapters.base.emission_version_of``: an int or short string, ``'1'`` when undeclared, and for the declared
readers the ``<own>+emitter<N>`` spelling of ``declared_common.emission_version`` -- so a bump of ``declared_common.EMITTER_VERSION`` moves
every declared reader at once and lists them all). The tree's ``EMITTER_VERSION`` is read beside it.

Printed on stdout: the names, sorted, space-separated on one line -- every adapter whose emission string differs between the trees, and
every adapter NEW carries that OLD does not (no accepted group of it can carry the new version yet). Dropped adapters and an adapter whose
version cannot be read (``ERROR ...``, the tree's path elided so the same refusal in both trees compares equal and is never listed as a
bump) are reported on stderr and, with ``--json``, in the document printed instead of the names:
``{"old", "new", "emitter_version": {"old", "new"}, "bumped": [{name, old, new}], "added", "dropped", "errors", "unchanged", "names"}``.
Exit 0 with the list (empty when nothing moved), 3 when a version could not be read in either tree (the list is still printed; the
operator fixes the tree -- a missing pins file, an adapter refusing at construction -- rather than reading a window's list past a gap),
2 on a tree that cannot be read at all.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

PROBE = r"""
import json, sys
from sisteer.registry import adapters as ADP
from sisteer.registry.adapters import base as AB
try:
    from sisteer.registry.adapters import declared_common as DC
    emitter = getattr(DC, "EMITTER_VERSION", None)
except Exception as exc:  # noqa: BLE001 -- an old tree without the module reads as no emitter version
    emitter = f"ERROR {type(exc).__name__}: {exc}"
out = {}
for name, cls in sorted(ADP.load_all().items()):
    try:
        out[name] = str(AB.emission_version_of(cls()))
    except Exception as exc:  # noqa: BLE001 -- reported by name, never guessed
        out[name] = f"ERROR {type(exc).__name__}: {exc}"[:200]
json.dump({"emitter_version": emitter, "adapters": out}, sys.stdout)
"""


def _tree(spec: str, repo: str, scratch: str) -> str:
    """The directory holding ``src/sisteer`` for ``spec``: the directory itself, or ``git archive`` of the revision into ``scratch``."""
    if os.path.isdir(spec):
        if not os.path.isdir(os.path.join(spec, "src", "sisteer")):
            raise SystemExit(f"{spec}: a tree directory holds src/sisteer; this one does not")
        return spec
    dest = os.path.join(scratch, spec.replace("/", "_"))
    os.makedirs(dest, exist_ok=True)
    archive = subprocess.run(["git", "-C", repo, "archive", "--format=tar", spec, "src/sisteer", "configs"], capture_output=True)
    if archive.returncode != 0:
        raise SystemExit(f"git archive {spec} failed in {repo}: {archive.stderr.decode(errors='replace').strip()}")
    tar = subprocess.run(["tar", "-x", "-C", dest], input=archive.stdout, capture_output=True)
    if tar.returncode != 0:
        raise SystemExit(f"untar of {spec} failed: {tar.stderr.decode(errors='replace').strip()}")
    return dest


def probe(tree: str, python: str) -> Tuple[Optional[object], Dict[str, str]]:
    """``(emitter_version, {adapter name: emission version string})`` of the tree, read in a subprocess with its ``src`` on the path."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(tree, "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([python, "-c", PROBE], capture_output=True, text=True, env=env, cwd=tree)
    if r.returncode != 0:
        raise SystemExit(f"loading the adapters of {tree} failed (rc {r.returncode}):\n{r.stderr.strip()[-2000:]}")
    doc = json.loads(r.stdout)
    # a refusal names the tree's path: elided, so the same refusal in both trees compares equal
    return doc.get("emitter_version"), {n: (v.replace(tree, "<tree>") if v.startswith("ERROR") else v) for n, v in doc["adapters"].items()}


def diff(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, object]:
    bumped = [{"name": n, "old": old[n], "new": new[n]} for n in sorted(set(old) & set(new)) if old[n] != new[n]]
    added = sorted(set(new) - set(old))
    dropped = sorted(set(old) - set(new))
    errors = {n: v for n, v in sorted(new.items()) if v.startswith("ERROR")} | {f"old:{n}": v for n, v in sorted(old.items()) if v.startswith("ERROR")}
    unchanged = sorted(n for n in set(old) & set(new) if old[n] == new[n])
    return {"bumped": bumped, "added": added, "dropped": dropped, "errors": errors, "unchanged": unchanged,
            "names": sorted({b["name"] for b in bumped} | set(added))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old", help="a git revision of --repo, or a directory holding src/sisteer")
    ap.add_argument("new", help="likewise")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="the repository `git archive` reads revisions from (default: this script's)")
    ap.add_argument("--python", default=sys.executable, help="the interpreter that loads each tree's adapters (default: this one)")
    ap.add_argument("--json", action="store_true", help="print the full diff document instead of the names")
    a = ap.parse_args(argv)
    scratch = tempfile.mkdtemp(prefix="bumped-adapters-")
    try:
        old_tree, new_tree = _tree(a.old, a.repo, scratch), _tree(a.new, a.repo, scratch)
        old_em, old = probe(old_tree, a.python)
        new_em, new = probe(new_tree, a.python)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    d = diff(old, new)
    doc = {"old": a.old, "new": a.new, "emitter_version": {"old": old_em, "new": new_em}, **d}
    if a.json:
        json.dump(doc, sys.stdout, indent=1, sort_keys=True)
        print()
    else:
        for n in d["dropped"]:
            print(f"dropped in {a.new}: {n} (was {old[n]})", file=sys.stderr)
        for n, v in d["errors"].items():
            print(f"{n}: {v}", file=sys.stderr)
        print(" ".join(d["names"]))
    return 3 if d["errors"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            sys.exit(2)
        raise
