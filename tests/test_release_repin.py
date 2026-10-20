"""scripts/release/repin.py carries a rewritten config's sha into every file that pins it, and leaves the scrub map alone.

configs/SCRUB_MAP.tsv records each rewritten file's earlier sha256 beside its sha today, so a repin pass that carried the
old shas forward would erase what the map records, and `repin.py --check` would call the map stale on every tree that
holds it (the regeneration check runs it after the map is written).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_repin(root: Path):
    spec = importlib.util.spec_from_file_location("repin_under_test", REPO / "scripts" / "release" / "repin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = str(root)          # the tool reads git and files relative to ROOT
    mod.BLOB_PINS = ()            # the real tree's pinned test files are not in the scratch repository
    mod.FILE_PINS = ()
    mod.CARRY_FILES = ()
    return mod


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture()
def scratch(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    _git(root.parent, "init", "-q", str(root))
    a = root / "configs" / "a.yaml"
    a.write_text("x: 1\n")
    old = _sha(a)
    (root / "configs" / "b.yaml").write_text(f"frozen_input_sha256: {old}\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    base = _git(root, "rev-parse", "HEAD").strip()
    a.write_text("x: 1  # reworded comment\n")
    new = _sha(a)
    (root / "configs" / "SCRUB_MAP.tsv").write_text(f"path\told_sha256\tnew_sha256\nconfigs/a.yaml\t{old}\t{new}\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "rewrite a.yaml; write the scrub map")
    return root, base, old, new


def test_check_names_the_stale_pin_and_not_the_scrub_map(scratch, capsys):
    root, base, old, new = scratch
    repin = _load_repin(root)
    assert repin.main(["--pre-ref", base, "--check"]) == 1
    out = capsys.readouterr().out
    assert "repin: stale: configs/b.yaml" in out
    assert "SCRUB_MAP" not in out


def test_repin_moves_the_pin_and_keeps_the_scrub_map_byte_for_byte(scratch, capsys):
    root, base, old, new = scratch
    smap = root / "configs" / "SCRUB_MAP.tsv"
    before = smap.read_bytes()
    repin = _load_repin(root)
    assert repin.main(["--pre-ref", base]) == 0
    assert (root / "configs" / "b.yaml").read_text() == f"frozen_input_sha256: {new}\n"
    assert smap.read_bytes() == before
    assert old in smap.read_text()
    capsys.readouterr()
    assert repin.main(["--pre-ref", base, "--check"]) == 0
    assert "0 file(s) stale" in capsys.readouterr().out
