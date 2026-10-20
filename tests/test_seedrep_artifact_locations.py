"""Artifact locations of the seed-replicate lane: IMMUTABLE per-record files
``RS/reseed/artifact_locations/<key>/<cell>/<cluster>.json`` written by atomic
rename, a conflicting rewrite refused, the aggregate derived on read — and
``colocate_cells.sh``, the consumer that copies a seed's parents onto the
derivation cluster with a sha gate, exercised against two temp "clusters"
(directories)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
if str(SEEDREP) not in sys.path:
    sys.path.insert(0, str(SEEDREP))

import artifact_locations as al  # noqa: E402

KEY1, KEY2 = "qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2"
CELL = "constv2-mix-full__psr_ce_all"
DERIVED_CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"
SHA_A = "a" * 64
SHA_B = "b" * 64
FP = "c" * 64


def _write(rs, key, cell, cluster, sha=SHA_A, fp=FP, copied_from=None,
           bank_dir=None, kind=None):
    return al.write_record(key=key, cell=cell, cluster=cluster,
                           bank_dir=bank_dir or f"/x/{cluster}/{key}/cells_all/{cell}",
                           object_sha256=sha, fit_fingerprint=fp,
                           copied_from=copied_from, kind=kind, rs=rs)


#: The record a write WITHOUT --kind produces — the shape every caller frozen
#: before the flag (the cluster-side checkout, the live DONE handler) relies on.
PRE_FLAG_KEYS = frozenset(("key", "cell", "cluster", "bank_dir", "object_sha256",
                           "fit_fingerprint", "copied_from", "verified_at",
                           "written_at"))


def test_two_seeds_same_cell_id_keep_both_records(tmp_path):
    p1, s1 = _write(tmp_path, KEY1, CELL, "cluster_a")
    p2, s2 = _write(tmp_path, KEY2, CELL, "cluster_a", sha=SHA_B)
    assert (s1, s2) == ("written", "written") and p1 != p2
    assert p1 == tmp_path / "reseed" / "artifact_locations" / KEY1 / CELL / "cluster_a.json"
    v = al.view(tmp_path)
    assert v["n_records"] == 2
    assert v["by_cell"][f"{KEY1}/{CELL}"]["object_sha256"] == SHA_A
    assert v["by_cell"][f"{KEY2}/{CELL}"]["object_sha256"] == SHA_B
    # no temp file left behind by the atomic write
    assert not list(p1.parent.glob(".*.tmp"))


def test_conflicting_rewrite_refuses_identical_is_noop(tmp_path):
    _write(tmp_path, KEY1, CELL, "cluster_a")
    _, state = _write(tmp_path, KEY1, CELL, "cluster_a")
    assert state == "unchanged"
    with pytest.raises(al.LocationError, match="immutable"):
        _write(tmp_path, KEY1, CELL, "cluster_a", sha=SHA_B)
    with pytest.raises(al.LocationError, match="immutable"):
        _write(tmp_path, KEY1, CELL, "cluster_a", bank_dir="/elsewhere")
    # the original stands
    rec = al.read_record(al.record_path(KEY1, CELL, "cluster_a", tmp_path))
    assert rec["object_sha256"] == SHA_A


def test_concurrent_writers_for_different_records_lose_nothing(tmp_path):
    cells = [f"cell{i}__arm" for i in range(12)]
    jobs = [(k, c, cl) for k in (KEY1, KEY2) for c in cells
            for cl in ("cluster_a", "cluster_b")]

    def go(job):
        k, c, cl = job
        return _write(tmp_path, k, c, cl, copied_from=(None if cl == "cluster_a"
                                                       else "cluster_a"))[1]
    with ThreadPoolExecutor(max_workers=16) as ex:
        states = list(ex.map(go, jobs))
    assert states.count("written") == len(jobs)
    v = al.view(tmp_path)
    assert v["n_records"] == len(jobs) == 48
    assert {(r["key"], r["cell"], r["cluster"]) for r in v["records"]} == set(jobs)
    # every (key, cell) lists both clusters, origin = cluster_a, consistent
    for agg in v["by_cell"].values():
        assert agg["clusters"] == ["cluster_a", "cluster_b"]
        assert agg["origin"] == "cluster_a" and agg["consistent"]


def test_view_flags_an_inconsistent_mirror_and_filters(tmp_path):
    _write(tmp_path, KEY1, CELL, "cluster_a")
    _write(tmp_path, KEY1, CELL, "cluster_b", sha=SHA_B, copied_from="cluster_a")
    _write(tmp_path, KEY2, CELL, "cluster_b", sha=SHA_B)
    v = al.view(tmp_path)
    assert v["by_cell"][f"{KEY1}/{CELL}"]["consistent"] is False
    assert v["by_cell"][f"{KEY2}/{CELL}"]["consistent"] is True
    assert v["by_cell"][f"{KEY2}/{CELL}"]["origin"] == "cluster_b"
    only = al.view(tmp_path, key=KEY2)
    assert only["n_records"] == 1 and list(only["by_cell"]) == [f"{KEY2}/{CELL}"]
    onlyc = al.view(tmp_path, cluster="cluster_b")
    assert onlyc["n_records"] == 2


def test_validation_refusals(tmp_path):
    with pytest.raises(al.LocationError, match="64 hex"):
        _write(tmp_path, KEY1, CELL, "cluster_a", sha="abc")
    with pytest.raises(al.LocationError, match="ANOTHER cluster"):
        _write(tmp_path, KEY1, CELL, "cluster_a", copied_from="cluster_a")
    with pytest.raises(al.LocationError, match="not a valid path"):
        _write(tmp_path, "../x", CELL, "cluster_a")
    # a record whose content names another (key, cell, cluster) than its path
    p = al.record_path(KEY1, CELL, "cluster_a", tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"key": KEY2, "cell": CELL, "cluster": "cluster_a",
                             "bank_dir": "/x", "object_sha256": SHA_A,
                             "fit_fingerprint": FP}))
    with pytest.raises(al.LocationError, match="sits at"):
        list(al.iter_records(tmp_path))


# --------------------------------------------------------------------------- #
# kind (SCHEMAS.md §2): optional on write, compared only when named             #
# --------------------------------------------------------------------------- #

def test_kind_is_written_idempotent_and_a_different_kind_refuses(tmp_path):
    p, state = _write(tmp_path, KEY1, CELL, "cluster_a", kind="fit")
    assert state == "written"
    assert al.read_record(p)["kind"] == "fit"
    # identical content INCLUDING kind: a no-op
    assert _write(tmp_path, KEY1, CELL, "cluster_a", kind="fit")[1] == "unchanged"
    before = p.read_bytes()
    # the same location/identity under a DIFFERENT kind: refused like any mismatch
    with pytest.raises(al.LocationError, match="immutable") as ei:
        _write(tmp_path, KEY1, CELL, "cluster_a", kind="derived")
    assert "'kind': ('fit', 'derived')" in str(ei.value)
    # content AND kind differ: both named in the refusal
    with pytest.raises(al.LocationError) as ei:
        _write(tmp_path, KEY1, CELL, "cluster_a", sha=SHA_B, kind="derived")
    assert "object_sha256" in str(ei.value) and "kind" in str(ei.value)
    assert p.read_bytes() == before
    with pytest.raises(al.LocationError, match="not one of"):
        _write(tmp_path, KEY2, CELL, "cluster_a", kind="banana")
    # a derived record is a kind too
    assert _write(tmp_path, KEY2, DERIVED_CELL, "cluster_b", kind="derived")[1] == "written"


def test_write_without_kind_is_the_pre_flag_record_byte_for_byte(tmp_path):
    """No --kind: no ``kind`` key, and the file is EXACTLY the pre-flag
    serialisation (the nine keys, indent=1, sorted) — a caller running the
    frozen checkout and one running this tree write the same bytes."""
    p, state = _write(tmp_path, KEY1, CELL, "cluster_a", copied_from=None)
    assert state == "written"
    rec = json.loads(p.read_text())
    assert set(rec) == PRE_FLAG_KEYS and "kind" not in rec
    expected = {"key": KEY1, "cell": CELL, "cluster": "cluster_a",
                "bank_dir": f"/x/cluster_a/{KEY1}/cells_all/{CELL}",
                "object_sha256": SHA_A, "fit_fingerprint": FP, "copied_from": None,
                "verified_at": rec["verified_at"], "written_at": rec["written_at"]}
    assert p.read_bytes() == (json.dumps(expected, indent=1, sort_keys=True) + "\n").encode()
    # and view still derives the same aggregate as before, kind None
    agg = al.view(tmp_path)["by_cell"][f"{KEY1}/{CELL}"]
    assert agg["kind"] is None and agg["consistent"]


def test_write_without_kind_leaves_an_existing_kind_alone(tmp_path):
    """The live DONE handler writes WITHOUT --kind and then patches ``kind``
    in; its --force-revalidate path re-issues the same write against the
    patched record and must still read "unchanged" with the kind intact."""
    p, _ = _write(tmp_path, KEY1, CELL, "cluster_a", kind="fit")
    before = p.read_bytes()
    assert _write(tmp_path, KEY1, CELL, "cluster_a")[1] == "unchanged"
    assert p.read_bytes() == before and al.read_record(p)["kind"] == "fit"
    # a content mismatch without --kind still refuses as before
    with pytest.raises(al.LocationError, match="immutable"):
        _write(tmp_path, KEY1, CELL, "cluster_a", sha=SHA_B)


def test_missing_kind_is_added_in_place_never_a_different_one(tmp_path):
    p, _ = _write(tmp_path, KEY1, CELL, "cluster_a")          # a pre-flag record
    old = json.loads(p.read_text())
    assert "kind" not in old
    # a content mismatch does NOT get to add a kind: refused, file untouched
    before = p.read_bytes()
    with pytest.raises(al.LocationError, match="immutable"):
        _write(tmp_path, KEY1, CELL, "cluster_a", sha=SHA_B, kind="fit")
    assert p.read_bytes() == before
    # the identical write naming a kind fills the missing label, keeping every
    # other byte (the timestamps are the ORIGINAL write's, not now)
    _, state = _write(tmp_path, KEY1, CELL, "cluster_a", kind="fit")
    assert state == "kind_added"
    new = json.loads(p.read_text())
    assert new == {**old, "kind": "fit"}
    assert p.read_bytes() == (json.dumps(new, indent=1, sort_keys=True) + "\n").encode()
    assert not list(p.parent.glob(".*.tmp"))
    # once present it is fixed: same -> no-op, different -> refused
    assert _write(tmp_path, KEY1, CELL, "cluster_a", kind="fit")[1] == "unchanged"
    with pytest.raises(al.LocationError, match="immutable"):
        _write(tmp_path, KEY1, CELL, "cluster_a", kind="derived")
    # a record carrying a kind outside the schema is malformed on read
    p.write_text(json.dumps({**old, "kind": "banana"}))
    with pytest.raises(al.LocationError, match="not one of"):
        al.read_record(p)


def test_view_warns_about_records_lacking_kind_and_never_refuses(tmp_path, capsys):
    _write(tmp_path, KEY1, CELL, "cluster_a", kind="fit")
    _write(tmp_path, KEY1, CELL, "cluster_b", copied_from="cluster_a")   # no kind
    _write(tmp_path, KEY2, CELL, "cluster_a")                             # no kind
    v = al.view(tmp_path)
    assert v["n_records"] == 3
    assert v["missing_kind"] == [f"{KEY1}/{CELL}/cluster_b.json",
                                 f"{KEY2}/{CELL}/cluster_a.json"]
    assert len(v["warnings"]) == 2 and all("lacks kind" in w for w in v["warnings"])
    # the aggregate carries the kind the records that have one agree on
    assert v["by_cell"][f"{KEY1}/{CELL}"]["kind"] == "fit"
    assert v["by_cell"][f"{KEY1}/{CELL}"]["consistent"] is True     # missing != wrong
    assert v["by_cell"][f"{KEY2}/{CELL}"]["kind"] is None
    # two records of one cell that DISAGREE on kind are inconsistent, and said so
    _write(tmp_path, KEY2, CELL, "cluster_b", copied_from="cluster_a", kind="derived")
    _write(tmp_path, KEY2, "other__cell", "cluster_a", kind="fit")
    _write(tmp_path, KEY2, "other__cell", "cluster_b", copied_from="cluster_a",
           kind="derived")
    v = al.view(tmp_path, key=KEY2)
    assert v["by_cell"][f"{KEY2}/other__cell"]["consistent"] is False
    assert any("disagree on kind" in w for w in v["warnings"])
    # CLI: exit 0, warnings on STDERR, stdout stays a clean table / JSON
    tool = str(SEEDREP / "artifact_locations.py")
    r = subprocess.run([sys.executable, tool, "--rs", str(tmp_path), "view"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert f"{KEY1}/{CELL}  clusters=cluster_a,cluster_b  origin=cluster_a  kind=fit  sha=" in r.stdout
    assert "WARNING" not in r.stdout
    assert r.stderr.count("[artifact-locations] WARNING: ") == 3
    assert f"{KEY2}/{CELL}/cluster_a.json lacks kind" in r.stderr
    r = subprocess.run([sys.executable, tool, "--rs", str(tmp_path), "view", "--json"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    doc = json.loads(r.stdout)                       # parseable despite the warnings
    assert len(doc["missing_kind"]) == 2 and len(doc["warnings"]) == 3
    assert "WARNING" in r.stderr
    # a fully-stamped tree is silent
    clean = tmp_path / "clean"
    _write(clean, KEY1, CELL, "cluster_a", kind="fit")
    r = subprocess.run([sys.executable, tool, "--rs", str(clean), "view"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == "" and "kind=fit" in r.stdout


# --------------------------------------------------------------------------- #
# derived records: fit_fingerprint null (SCHEMAS.md §2), and ONLY there         #
# --------------------------------------------------------------------------- #

def _derived_dir(cluster, key=KEY1, cell=DERIVED_CELL):
    return f"/x/{cluster}/derived/lora_subtraction/{key}/{cell}"


def test_derived_record_carries_a_null_fingerprint_and_stays_immutable(tmp_path):
    p, state = _write(tmp_path, KEY1, DERIVED_CELL, "cluster_a", fp=None,
                      kind="derived", bank_dir=_derived_dir("cluster_a"))
    assert state == "written"
    rec = json.loads(p.read_text())
    assert rec["kind"] == "derived" and rec["fit_fingerprint"] is None
    assert '"fit_fingerprint": null' in p.read_text()
    # the same write again, by None, "" or the CLI spelling "-": unchanged
    for fp in (None, "", al.FP_NULL):
        assert _write(tmp_path, KEY1, DERIVED_CELL, "cluster_a", fp=fp, kind="derived",
                      bank_dir=_derived_dir("cluster_a"))[1] == "unchanged"
    # null vs a hash is a content mismatch like any other — immutable
    with pytest.raises(al.LocationError, match="immutable") as ei:
        _write(tmp_path, KEY1, DERIVED_CELL, "cluster_a", fp=FP, kind="derived",
               bank_dir=_derived_dir("cluster_a"))
    assert "fit_fingerprint" in str(ei.value)
    assert json.loads(p.read_text())["fit_fingerprint"] is None
    # a derived record may still carry a 64-hex fingerprint if its writer has
    # one (the schema's value is null; the stager ignores it for derived) —
    # and it is validated like any fingerprint
    assert _write(tmp_path, KEY2, DERIVED_CELL, "cluster_b", fp=FP, kind="derived",
                  bank_dir=_derived_dir("cluster_b", KEY2))[1] == "written"
    with pytest.raises(al.LocationError, match="64 hex"):
        _write(tmp_path, KEY2, DERIVED_CELL, "cluster_a", fp="abc", kind="derived",
               bank_dir=_derived_dir("cluster_a", KEY2))
    # read back through the reader: no refusal, view derives the aggregate
    assert al.read_record(p)["fit_fingerprint"] is None
    agg = al.view(tmp_path, key=KEY1)["by_cell"][f"{KEY1}/{DERIVED_CELL}"]
    assert agg["kind"] == "derived" and agg["fit_fingerprint"] is None and agg["consistent"]


def test_null_fingerprint_is_refused_unless_the_record_is_derived(tmp_path):
    # WRITE: a fit record, or a kind-less one, never gets a null fingerprint
    for kind in ("fit", None):
        for fp in (None, "", al.FP_NULL):
            with pytest.raises(al.LocationError, match="derived record only"):
                _write(tmp_path, KEY1, CELL, "cluster_a", fp=fp, kind=kind)
    assert not al.record_path(KEY1, CELL, "cluster_a", tmp_path).exists()
    # READ: the same rule on what is already on disk
    def plant(cluster, **extra):
        q = al.record_path(KEY1, CELL, cluster, tmp_path)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps({"key": KEY1, "cell": CELL, "cluster": cluster,
                                 "bank_dir": "/x", "object_sha256": SHA_A,
                                 "fit_fingerprint": None, **extra}))
        return q
    with pytest.raises(al.LocationError, match=r"lacks \['fit_fingerprint'\]"):
        al.read_record(plant("cluster_a", kind="fit"))
    with pytest.raises(al.LocationError, match=r"lacks \['fit_fingerprint'\]"):
        al.read_record(plant("cluster_b"))                    # kind-less
    # the store holding them refuses on view too (malformed record, not "missing kind")
    with pytest.raises(al.LocationError, match="fit_fingerprint"):
        al.view(tmp_path)
    # and the very same body under kind=derived reads
    q = plant("cluster_a", kind="derived")
    assert al.read_record(q)["fit_fingerprint"] is None


def test_view_of_a_mixed_store_of_fits_and_derived(tmp_path):
    # seed key: four fits (one mirrored) + one derived object (mirrored too)
    _write(tmp_path, KEY1, CELL, "cluster_a", kind="fit")
    _write(tmp_path, KEY1, CELL, "cluster_b", kind="fit", copied_from="cluster_a")
    _write(tmp_path, KEY1, "constv2ctl2-full__lora_r4_ce", "cluster_a", sha=SHA_B, kind="fit")
    _write(tmp_path, KEY1, DERIVED_CELL, "cluster_a", fp=None, kind="derived", sha="d" * 64,
           bank_dir=_derived_dir("cluster_a"))
    _write(tmp_path, KEY1, DERIVED_CELL, "cluster_b", fp=None, kind="derived", sha="d" * 64,
           bank_dir=_derived_dir("cluster_b"), copied_from="cluster_a")
    v = al.view(tmp_path, key=KEY1)
    assert v["n_records"] == 5 and v["missing_kind"] == [] and v["warnings"] == []
    fit = v["by_cell"][f"{KEY1}/{CELL}"]
    der = v["by_cell"][f"{KEY1}/{DERIVED_CELL}"]
    assert fit["kind"] == "fit" and fit["fit_fingerprint"] == FP and fit["consistent"]
    assert der["kind"] == "derived" and der["fit_fingerprint"] is None and der["consistent"]
    assert der["clusters"] == ["cluster_a", "cluster_b"] and der["origin"] == "cluster_a"
    assert der["object_sha256"] == "d" * 64
    # CLI: both kinds in the table, JSON carries the null, no warnings
    tool = str(SEEDREP / "artifact_locations.py")
    r = subprocess.run([sys.executable, tool, "--rs", str(tmp_path), "view", "--key", KEY1],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == ""
    assert f"{KEY1}/{DERIVED_CELL}  clusters=cluster_a,cluster_b  origin=cluster_a  kind=derived  sha=dddddddddddd" in r.stdout
    assert f"{KEY1}/{CELL}  clusters=cluster_a,cluster_b  origin=cluster_a  kind=fit  sha=" in r.stdout
    r = subprocess.run([sys.executable, tool, "--rs", str(tmp_path), "view", "--key", KEY1, "--json"],
                       capture_output=True, text=True)
    doc = json.loads(r.stdout)
    assert doc["by_cell"][f"{KEY1}/{DERIVED_CELL}"]["fit_fingerprint"] is None
    assert [x["fit_fingerprint"] for x in doc["records"] if x["cell"] == DERIVED_CELL] == [None, None]
    # a derived mirror that carries a hash where its origin carries null is
    # INCONSISTENT whichever cluster's record sorts first (null is a value)
    for first, second in (("cluster_a", "cluster_b"), ("cluster_b", "cluster_a")):
        rs = tmp_path / f"order-{first}"
        _write(rs, KEY2, DERIVED_CELL, first, fp=None, kind="derived",
               bank_dir=_derived_dir(first, KEY2))
        _write(rs, KEY2, DERIVED_CELL, second, fp="e" * 64, kind="derived",
               bank_dir=_derived_dir(second, KEY2), copied_from=first)
        agg = al.view(rs)["by_cell"][f"{KEY2}/{DERIVED_CELL}"]
        assert agg["consistent"] is False, (first, second)


def test_cli_derived_write_without_a_fingerprint(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, str(SEEDREP / "artifact_locations.py"),
                               "--rs", str(tmp_path), *args],
                              capture_output=True, text=True)
    base = ["write", "--key", KEY1, "--cell", DERIVED_CELL, "--cluster", "cluster_a",
            "--bank-dir", _derived_dir("cluster_a"), "--object-sha256", SHA_A]
    r = run(*base, "--kind", "derived")                       # omitted fingerprint
    assert r.returncode == 0 and "written" in r.stdout, r
    r = run(*base, "--kind", "derived", "--fit-fingerprint", "-")
    assert r.returncode == 0 and "unchanged" in r.stdout
    rec = al.read_record(al.record_path(KEY1, DERIVED_CELL, "cluster_a", tmp_path))
    assert rec["fit_fingerprint"] is None and rec["kind"] == "derived"
    # the fit forms, and the pre-flag form, still need the fingerprint (exit 3, not argparse)
    fit = ["write", "--key", KEY1, "--cell", CELL, "--cluster", "cluster_a",
           "--bank-dir", "/b/x", "--object-sha256", SHA_A]
    r = run(*fit, "--kind", "fit")
    assert r.returncode == 3 and "derived record only" in r.stderr
    r = run(*fit, "--fit-fingerprint", "-")
    assert r.returncode == 3 and "derived record only" in r.stderr
    r = run(*fit)
    assert r.returncode == 3 and "derived record only" in r.stderr
    assert not al.record_path(KEY1, CELL, "cluster_a", tmp_path).exists()
    r = run(*fit, "--fit-fingerprint", FP)
    assert r.returncode == 0 and "written" in r.stdout


def test_cli_kind_flag(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, str(SEEDREP / "artifact_locations.py"),
                               "--rs", str(tmp_path), *args],
                              capture_output=True, text=True)
    w = ["write", "--key", KEY1, "--cell", CELL, "--cluster", "cluster_a",
         "--bank-dir", "/b/x", "--object-sha256", SHA_A, "--fit-fingerprint", FP]
    r = run(*w, "--kind", "banana")
    assert r.returncode == 2 and "invalid choice" in r.stderr
    r = run(*w, "--kind", "fit")
    assert r.returncode == 0 and "written" in r.stdout, r
    r = run(*w, "--kind", "fit")
    assert r.returncode == 0 and "unchanged" in r.stdout
    r = run(*w)                                       # the frozen callers' form
    assert r.returncode == 0 and "unchanged" in r.stdout
    r = run(*w, "--kind", "derived")
    assert r.returncode == 3 and "immutable" in r.stderr and "kind" in r.stderr
    rec = al.read_record(al.record_path(KEY1, CELL, "cluster_a", tmp_path))
    assert rec["kind"] == "fit"
    # a pre-flag record + the same write with --kind: kind_added, exit 0
    r = run("write", "--key", KEY2, "--cell", CELL, "--cluster", "cluster_b",
            "--bank-dir", "/c/x", "--object-sha256", SHA_B, "--fit-fingerprint", FP)
    assert r.returncode == 0 and "written" in r.stdout
    r = run("write", "--key", KEY2, "--cell", CELL, "--cluster", "cluster_b",
            "--bank-dir", "/c/x", "--object-sha256", SHA_B, "--fit-fingerprint", FP,
            "--kind", "fit")
    assert r.returncode == 0 and "kind_added" in r.stdout
    assert al.read_record(al.record_path(KEY2, CELL, "cluster_b", tmp_path))["kind"] == "fit"


def test_cli_write_view_roundtrip_and_exit_codes(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, str(SEEDREP / "artifact_locations.py"),
                               "--rs", str(tmp_path), *args],
                              capture_output=True, text=True)
    w = ["write", "--key", KEY1, "--cell", CELL, "--cluster", "cluster_a",
         "--bank-dir", "/b/x", "--object-sha256", SHA_A, "--fit-fingerprint", FP]
    r = run(*w)
    assert r.returncode == 0 and "written" in r.stdout, r
    r = run(*w)
    assert r.returncode == 0 and "unchanged" in r.stdout
    r = run(*w[:-1], "d" * 64)
    assert r.returncode == 3 and "immutable" in r.stderr
    r = run("view", "--json")
    doc = json.loads(r.stdout)
    assert doc["n_records"] == 1 and doc["records"][0]["fit_fingerprint"] == FP
    r = run("view")
    assert f"{KEY1}/{CELL}  clusters=cluster_a  origin=cluster_a" in r.stdout
    # $RS is the default root
    r = subprocess.run([sys.executable, str(SEEDREP / "artifact_locations.py"),
                        "view", "--json"], capture_output=True, text=True,
                       env={**os.environ, "RS": str(tmp_path)})
    assert json.loads(r.stdout)["n_records"] == 1


# --------------------------------------------------------------------------- #
# colocate_cells.sh against two temp clusters                                  #
# --------------------------------------------------------------------------- #

PARENTS = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
           "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _cell_on(root: Path, key: str, cell: str, seed_byte: int) -> tuple[Path, str]:
    d = root / "exp" / "objects" / "sdfbase" / "reseed" / key / "cells_all" / cell
    d.mkdir(parents=True)
    obj = bytes([seed_byte]) * 4096 + cell.encode()
    (d / "object.pt").write_bytes(obj)
    (d / "train_budget_meta.json").write_text(json.dumps({"cell_id": cell, "seed": 1}))
    (d / "COMPLETE").write_text(json.dumps({"cell_id": cell, "fit_fingerprint": FP,
                                            "object_sha256": _sha(obj)}))
    (d / "metrics.jsonl").write_text("{}\n")
    (d / "final_ckpt").mkdir()
    (d / "final_ckpt" / "ckpt.pt").write_bytes(b"ck")
    return d, _sha(obj)


def _world(tmp_path, kind="fit"):
    """Three parents published on cluster_a, one on cluster_b, the pool only
    on cluster_a; the ledger records each where it is (``kind`` on every
    record as the DONE handler leaves it; ``kind=None`` = pre-flag records)."""
    bulb, char = tmp_path / "cluster_a", tmp_path / "cluster_b"
    rs = tmp_path / "rs"
    for i, cell in enumerate(PARENTS):
        root, cluster = (char, "cluster_b") if i == 3 else (bulb, "cluster_a")
        d, sha = _cell_on(root, KEY1, cell, 10 + i)
        al.write_record(key=KEY1, cell=cell, cluster=cluster, bank_dir=str(d),
                        object_sha256=sha, fit_fingerprint=FP, kind=kind, rs=rs)
    pool = bulb / "vol" / "pool" / KEY1
    pool.mkdir(parents=True)
    for n in ("pairs_constv2-mix-full.jsonl", "pairs_constv2-mix-full.manifest.json",
              "pairs_constv2ctl2-full.jsonl", "pairs_constv2-heldout.jsonl"):
        (pool / n).write_text(n + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SEEDREP_")}
    env.update(SEEDREP_CLUSTER_ROOT_cluster_a=str(bulb),
               SEEDREP_CLUSTER_ROOT_cluster_b=str(char),
               SEEDREP_STAGING=str(tmp_path / "staging"), SEEDREP_USER="nobodyx")
    return bulb, char, rs, env


def _colocate(env, *args):
    return subprocess.run(["bash", str(SEEDREP / "colocate_cells.sh"), KEY1, *args],
                          capture_output=True, text=True, env=env, timeout=120)


def _needs_rsync():
    """colocate_cells.sh shells out to rsync for every copy. The Modal CPU-suite
    image (git + build-essential only) has none, so the tests that COPY skip
    there with the reason and run wherever rsync is on PATH (the grading host, the
    clusters); the dry-run / refuse-before-copy tests need no rsync and stay."""
    if shutil.which("rsync") is None:
        pytest.skip("rsync not on PATH — colocate_cells.sh shells out to it for every copy "
                    "(the Modal CPU-suite image has no rsync; the grading host and the clusters do)")


def test_colocate_copies_the_off_cluster_parent_verifies_and_records(tmp_path):
    _needs_rsync()
    bulb, char, rs, env = _world(tmp_path)
    r = _colocate(env, "cluster_a", "--rs", str(rs))
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count(": already on cluster_a at") == 3
    assert "constv2ctl2-full__lora_r4_ce: verified on cluster_a (5 files" in r.stdout
    dst = bulb / "exp/objects/sdfbase/reseed" / KEY1 / "cells_all" / PARENTS[3]
    src = char / "exp/objects/sdfbase/reseed" / KEY1 / "cells_all" / PARENTS[3]
    assert (dst / "object.pt").read_bytes() == (src / "object.pt").read_bytes()
    assert (dst / "final_ckpt" / "ckpt.pt").exists()
    # the mirror's OWN record: copied_from names the source; the origin record is untouched
    v = al.view(rs, key=KEY1, cell=PARENTS[3])
    agg = v["by_cell"][f"{KEY1}/{PARENTS[3]}"]
    assert agg["clusters"] == ["cluster_a", "cluster_b"] and agg["origin"] == "cluster_b"
    assert agg["consistent"] and agg["bank_dir"]["cluster_a"] == str(dst)
    rec = al.read_record(al.record_path(KEY1, PARENTS[3], "cluster_a", rs))
    assert rec["copied_from"] == "cluster_b"
    # the mirror carries the source record's kind (SCHEMAS.md §2 — the stager requires it)
    assert rec["kind"] == "fit" and "record written (kind=fit)" in r.stdout
    assert "carries no kind" not in r.stdout
    assert al.view(rs, key=KEY1)["missing_kind"] == []
    assert "pool: " in r.stdout and (bulb / "vol/pool" / KEY1).is_dir()
    # a second run finds everything present and copies nothing
    r2 = _colocate(env, "cluster_a", "--rs", str(rs))
    assert r2.returncode == 0 and r2.stdout.count(": already on cluster_a at") == 4
    assert "copied now" in r2.stdout and "4 already on cluster_a, 0 copied now" in r2.stdout


def test_colocate_to_cluster_b_copies_three_parents_and_the_pool(tmp_path):
    _needs_rsync()
    bulb, char, rs, env = _world(tmp_path)
    r = _colocate(env, "cluster_b", "--rs", str(rs))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 already on cluster_b, 3 copied now" in r.stdout
    pool = char / "vol" / "pool" / KEY1
    assert sorted(p.name for p in pool.iterdir()) == sorted(
        p.name for p in (bulb / "vol" / "pool" / KEY1).iterdir())
    assert "pool: verified (4 files, every sha equal)" in r.stdout
    for cell in PARENTS[:3]:
        rec = al.read_record(al.record_path(KEY1, cell, "cluster_b", rs))
        assert rec["copied_from"] == "cluster_a" and rec["kind"] == "fit"
    assert r.stdout.count("record written (kind=fit)") == 3


def test_colocate_mirror_of_a_pre_flag_record_is_stamped_fit_and_says_so(tmp_path):
    """A source record written before --kind existed carries none; the mirror
    is still a parent fit (cells_all/), so colocate stamps it ``fit`` and logs
    the fallback. The origin record is left as it was (view warns about it)."""
    _needs_rsync()
    bulb, char, rs, env = _world(tmp_path, kind=None)
    r = _colocate(env, "cluster_a", "--rs", str(rs))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (f"{PARENTS[3]}: source record on cluster_b carries no kind" in r.stdout
            and "the mirror records kind=fit" in r.stdout)
    mirror = al.read_record(al.record_path(KEY1, PARENTS[3], "cluster_a", rs))
    assert mirror["kind"] == "fit" and mirror["copied_from"] == "cluster_b"
    origin = al.read_record(al.record_path(KEY1, PARENTS[3], "cluster_b", rs))
    assert "kind" not in origin
    v = al.view(rs, key=KEY1, cell=PARENTS[3])
    assert v["missing_kind"] == [f"{KEY1}/{PARENTS[3]}/cluster_b.json"]
    assert v["by_cell"][f"{KEY1}/{PARENTS[3]}"]["kind"] == "fit"
    assert v["by_cell"][f"{KEY1}/{PARENTS[3]}"]["consistent"]


def test_colocate_refuses_a_corrupted_copy_and_writes_no_record(tmp_path):
    _needs_rsync()
    bulb, char, rs, env = _world(tmp_path)
    # an rsync that copies, then flips a byte of object.pt at the DESTINATION leg
    shim = tmp_path / "rsync-corrupt"
    shim.write_text("#!/bin/bash\nset -e\n"
                    "rsync \"$@\"\n"
                    "dst=\"${@: -1}\"\n"
                    "case \"$dst\" in */staging/*) exit 0;; esac\n"
                    "f=\"$dst/object.pt\"; [ -f \"$f\" ] || exit 0\n"
                    "printf 'X' | dd of=\"$f\" bs=1 seek=0 conv=notrunc 2>/dev/null\n")
    shim.chmod(0o755)
    r = _colocate({**env, "SEEDREP_RSYNC": str(shim)}, "cluster_a", "--rs", str(rs))
    assert r.returncode == 4, r.stdout + r.stderr
    assert "REFUSE constv2ctl2-full__lora_r4_ce: copy does not verify" in r.stdout
    assert not al.record_path(KEY1, PARENTS[3], "cluster_a", rs).exists()
    # the corrupted copy is left in place (never deleted), not recorded
    dst = bulb / "exp/objects/sdfbase/reseed" / KEY1 / "cells_all" / PARENTS[3]
    assert (dst / "object.pt").exists()


def test_colocate_refuses_when_the_source_disk_disagrees_with_its_record(tmp_path):
    bulb, char, rs, env = _world(tmp_path)
    src = char / "exp/objects/sdfbase/reseed" / KEY1 / "cells_all" / PARENTS[3]
    (src / "object.pt").write_bytes(b"tampered")
    r = _colocate(env, "cluster_a", "--rs", str(rs))
    assert r.returncode == 4
    assert "not copying a cell whose ledger and disk disagree" in r.stdout
    dst = bulb / "exp/objects/sdfbase/reseed" / KEY1 / "cells_all" / PARENTS[3]
    assert not dst.exists()


def test_colocate_dry_run_and_missing_record(tmp_path):
    bulb, char, rs, env = _world(tmp_path)
    r = _colocate(env, "cluster_a", "--rs", str(rs), "--dry-run")
    assert r.returncode == 0 and "would copy cluster_b:" in r.stdout
    dst = bulb / "exp/objects/sdfbase/reseed" / KEY1 / "cells_all" / PARENTS[3]
    assert not dst.exists()
    r = _colocate(env, "cluster_a", "--rs", str(rs), "--cells",
                  "constv2-mix-full__psr_ce_all,nonexistent__cell")
    assert r.returncode == 4 and "no artifact-location record at all" in r.stdout
    r = _colocate(env, "moon", "--rs", str(rs))
    assert r.returncode == 2
