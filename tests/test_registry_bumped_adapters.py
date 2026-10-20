"""`scripts/registry/landing/bumped_adapters.py`: the window's adapter list is MATERIALISED
from the emission versions that moved between two trees, never typed. The tool loads each tree's adapter registry in a subprocess with that
tree's `src` on the path and compares `adapters.base.emission_version_of` per adapter -- so an `EMISSION_VERSION` bump lists its adapter, a
`declared_common.EMITTER_VERSION` bump lists every declared reader (their spelling is `<own>+emitter<N>`), an added adapter is listed, a
dropped one and an unreadable version are reported (exit 3 on the latter: a gap is never read past). Trees are git revisions (`git archive`
of `src/sisteer` + `configs`, the pins the adapters read at construction) or directories."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "scripts", "registry", "landing", "bumped_adapters.py")
PY = sys.executable

spec = importlib.util.spec_from_file_location("bumped_adapters", TOOL)
BA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BA)


def _head_copy(dest: str) -> str:
    """HEAD's `src/sisteer` + `configs` as a directory tree (what the tool itself archives for a revision)."""
    os.makedirs(dest, exist_ok=True)
    tar = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", "HEAD", "src/sisteer", "configs"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=tar, check=True)
    return dest


def _run(*args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([PY, TOOL, *args], capture_output=True, text=True, env=env)


def test_diff_lists_bumps_and_additions_and_reports_drops_and_errors_without_listing_a_shared_refusal():
    old = {"a": "1", "b": "2+emitter3", "c": "7", "d": "ERROR AdapterRefusal: <tree>/configs/x.json missing", "gone": "1"}
    new = {"a": "1", "b": "2+emitter4", "c": "8", "d": "ERROR AdapterRefusal: <tree>/configs/x.json missing", "fresh": "1"}
    d = BA.diff(old, new)
    assert d["bumped"] == [{"name": "b", "old": "2+emitter3", "new": "2+emitter4"}, {"name": "c", "old": "7", "new": "8"}]
    assert d["added"] == ["fresh"] and d["dropped"] == ["gone"] and d["unchanged"] == ["a", "d"]
    assert d["names"] == ["b", "c", "fresh"]                                        # the same refusal on both sides is not a bump
    assert set(d["errors"]) == {"d", "old:d"}


def test_the_same_tree_twice_lists_nothing_and_exits_zero(tmp_path):
    r = _run("HEAD", REPO)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "", r.stdout
    doc = json.loads(_run("HEAD", REPO, "--json").stdout)
    assert doc["bumped"] == [] and doc["added"] == [] and doc["dropped"] == [] and doc["errors"] == {} and doc["names"] == []
    assert doc["emitter_version"]["old"] == doc["emitter_version"]["new"] and isinstance(doc["emitter_version"]["new"], int)
    assert len(doc["unchanged"]) > 40                                                # every registered adapter


def test_an_emission_version_bump_lists_that_adapter_alone_and_an_emitter_bump_lists_every_declared_reader(tmp_path):
    new = _head_copy(str(tmp_path / "new"))
    p = os.path.join(new, "src", "sisteer", "registry", "adapters", "arditi_vendor_mt.py")
    s = open(p).read()
    m = re.search(r"^    EMISSION_VERSION = (\d+)$", s, re.M)
    assert m, "the test edits a literal EMISSION_VERSION"
    open(p, "w").write(s[:m.start()] + f"    EMISSION_VERSION = {int(m.group(1)) + 1}" + s[m.end():])
    r = _run("HEAD", new)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["arditi_vendor_mt"], r.stdout
    doc = json.loads(_run("HEAD", new, "--json").stdout)
    assert doc["bumped"] == [{"name": "arditi_vendor_mt", "old": m.group(1), "new": str(int(m.group(1)) + 1)}]
    # the emitter version: every declared reader's spelling moves at once
    p = os.path.join(new, "src", "sisteer", "registry", "adapters", "declared_common.py")
    s = open(p).read()
    m2 = re.search(r"^EMITTER_VERSION = (\d+)$", s, re.M)
    open(p, "w").write(s[:m2.start()] + f"EMITTER_VERSION = {int(m2.group(1)) + 1}" + s[m2.end():])
    doc = json.loads(_run("HEAD", new, "--json").stdout)
    assert doc["emitter_version"] == {"old": int(m2.group(1)), "new": int(m2.group(1)) + 1}
    declared = {n for n, v in zip(doc["unchanged"] + [b["name"] for b in doc["bumped"]], [None] * 0)}  # noqa: F841 (readability)
    bumped = {b["name"]: (b["old"], b["new"]) for b in doc["bumped"]}
    emitter_readers = {n for n, (o, _) in bumped.items() if "+emitter" in o}
    assert len(emitter_readers) >= 10 and all(o.endswith(f"+emitter{m2.group(1)}") and n_.endswith(f"+emitter{int(m2.group(1)) + 1}") for n, (o, n_) in bumped.items() if n in emitter_readers)
    assert "arditi_vendor_mt" in bumped and set(bumped) == emitter_readers | {"arditi_vendor_mt"}
    assert set(_run("HEAD", new).stdout.split()) == set(bumped)


def test_a_directory_without_the_source_tree_or_an_unknown_revision_is_refused_by_name(tmp_path):
    r = _run("HEAD", str(tmp_path))
    assert r.returncode == 2 and "holds src/sisteer" in r.stderr
    r = _run("no-such-revision-0000", REPO)
    assert r.returncode == 2 and "git archive no-such-revision-0000 failed" in r.stderr
