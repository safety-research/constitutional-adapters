"""Admission by attestation, once per read operation.

Before the read array, `attest.sbatch` runs `stores.attest_plan` over every pinned generation of the plan: the authority facts (`current`
names the pin; MANIFEST / VERIFIED / DEST-STAT / SEALED / PUBLISHED beside it, agreeing with the manifest), a FULL DIRECTORY LISTING against
the manifest (`stores.list_generation`: every extra, missing or re-typed entry and every differing link target refuses by name), the
directory seal (every directory 0555; the count the SEALED record's) and the content -- the full `admit_generation` walk the first time,
recorded in `ATTESTED-<sha8>.json`, else a seeded 64-file stat-and-hash sample within a week. The record `<window root>/attestation.json`
is what every shard checks in O(1) (`stores.attestation_admits` through `cluster_reads.admit_pins`) and refuses by name without. The
NEGATIVE cases come first: an added subtree, a deleted subtree, a re-pointed link and a missing file under an unchanged
`current` are each refused by the listing; a sealed directory refuses a write; only then does the per-shard walk's removal stand (the
shard's admission test in test_registry_cluster_read.py). Also here: the escape refusal of `GenerationHashCache` (an input spelled under a
pinned base whose real path is not inside a generation pinned at that base -- either one when two components share the base), the
submission order (the attestation job first, the array behind it with `--dependency=afterok`), the `cluster_a` skip of both grading-host
revalidation passes, and the sbatch's cluster rules. The destination is a local directory and `test_registry_stores`' `Runner` passes every
shell step through to real subprocesses while intercepting `sbatch` (the verify runs in-process) and the watcher, so the seal, the records
and the listing are the real thing.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import stat as stat_module
import subprocess
import sys
import time

import pytest

from sisteer.registry import cli
from sisteer.registry import cluster_reads as CR
from sisteer.registry import records as R
from sisteer.registry import stores as ST
from sisteer.registry.adapters import base as AB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import permission_denial_enforced  # noqa: E402
from test_registry_stores import Runner, WATCH, _canon, _cfg, _rewrite_preserving_stat, _write, world  # noqa: E402,F401  (the fixture rides along)
from test_registry_store_publish import _colon_tree  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
ATTEST_SBATCH = os.path.join(REPO, "scripts", "registry", "landing", "attest.sbatch")
FILES = ("a/b/f1.json", "a/f2.jsonl", "top.txt", "x/deep/er/f3.bin")


def _dirs(top):
    return [dp for dp, _dns, _fns in os.walk(top)]


def _unseal(gen):
    for d in _dirs(gen):
        os.chmod(d, 0o755)


def _reseal(gen):
    for d in reversed(_dirs(gen)):
        os.chmod(d, ST.DIR_SEAL_MODE)


def _publish(w):
    """The world's component mirrored as a published generation: sealed directories, SEALED and PUBLISHED records, `current`."""
    r = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r.outcome == "published", r.as_dict()
    gd, sha8 = w["gen_dir"], r.content_sha8
    return gd, sha8, f"{gd}/{sha8}"


def _manifest(gd, sha8):
    mpath = f"{gd}/MANIFEST-{sha8}.sha256"
    return ST.read_manifest_lines(mpath), ST.read_manifest_symlinks(mpath)


def _later(stamp, days):
    from datetime import datetime, timedelta, timezone
    dt = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc) + timedelta(days=days)
    return R._fmt_utc(dt)


# --------------------------------------------------------------------------------------------
# the listing: every kind of difference, by true name
# --------------------------------------------------------------------------------------------


def test_listing_compares_true_names_with_the_manifest_and_names_every_kind_of_difference(world, tmp_path):
    """(b) + (c): one scandir pass -- no stat, no hash -- against the manifest's files and attested links. An added subtree is `extra`, a
    deleted one `missing`, a re-pointed link `link_mismatch`, a file replaced by a directory `retyped`, a FIFO `other`; an EMPTY directory the
    manifest implies by no entry is counted, not a mismatch (the mirror copies member directories whole; the count is the seal record's to
    judge); the directory modes are the seal check. Escaped names are unescaped before the comparison."""
    w = world
    gd, sha8, gen = _publish(w)
    entries, links = _manifest(gd, sha8)
    lst = ST.list_generation(gen, entries, links)
    assert lst["ok"] and lst["files"] == 4 and lst["symlinks"] == 1 and lst["dirs"] == 6 and lst["entries"] == 10 and lst["other"] == 0, lst
    assert lst["missing"] == lst["extra"] == lst["retyped"] == lst["link_mismatch"] == lst["extra_dirs"] == [] and lst["n_unsealed_dirs"] == 0
    assert lst["seal_checked"] and lst["elapsed_s"] >= 0 and lst["seal_elapsed_s"] >= 0
    # the seal: the owner cannot add an entry inside a sealed directory (R14 proved the same on VAST); the listing is the namespace guard whatever the seal does.
    # The kernel's refusal is the thing demonstrated: a process it does not refuse (root / CAP_DAC_OVERRIDE) would plant the file, so the
    # demonstration runs only where the probe reads the refusal; the listing assertions below hold regardless
    if permission_denial_enforced():
        with pytest.raises(PermissionError):
            open(f"{gen}/planted.txt", "wb").close()
        with pytest.raises(PermissionError):
            os.mkdir(f"{gen}/planted")
    # an ADDED subtree under an unchanged `current` (the seal lifted and restored, the new directory sealed too) -> extra entries, by name
    _unseal(gen)
    _write(f"{gen}/planted/p.txt", b"p")
    _reseal(gen)
    lst = ST.list_generation(gen, entries, links)
    assert not lst["ok"] and lst["extra"] == ["planted/p.txt"] and lst["n_extra"] == 1 and lst["extra_dirs"] == ["planted"] and lst["dirs"] == 7 and lst["n_unsealed_dirs"] == 0
    _unseal(gen)
    shutil.rmtree(f"{gen}/planted")
    # a DELETED subtree -> missing entries
    shutil.rmtree(f"{gen}/x")
    _reseal(gen)
    lst = ST.list_generation(gen, entries, links)
    assert not lst["ok"] and lst["missing"] == ["x/deep/er/f3.bin"] and lst["dirs"] == 3 and lst["extra"] == []
    _unseal(gen)
    shutil.copytree(os.path.join(w["base"], "x"), f"{gen}/x")
    # a RE-POINTED link -> the target differs from the attested one; a target that escapes the tree is refused the same way
    os.remove(f"{gen}/link_to_top")
    os.symlink("a/f2.jsonl", f"{gen}/link_to_top")
    _reseal(gen)
    lst = ST.list_generation(gen, entries, links)
    assert not lst["ok"] and lst["link_mismatch"] == ["link_to_top"] and lst["missing"] == [] and lst["extra"] == []
    _unseal(gen)
    os.remove(f"{gen}/link_to_top")
    os.symlink("../../../etc/passwd", f"{gen}/link_to_top")
    lst = ST.list_generation(gen, entries, links, seal_check=False)
    assert not lst["ok"] and lst["link_mismatch"] == ["link_to_top"] and lst["seal_checked"] is False
    os.remove(f"{gen}/link_to_top")
    os.symlink("top.txt", f"{gen}/link_to_top")
    # a RE-TYPED entry: the file replaced by a directory of the same name
    os.remove(f"{gen}/top.txt")
    os.mkdir(f"{gen}/top.txt")
    lst = ST.list_generation(gen, entries, links, seal_check=False)
    assert not lst["ok"] and lst["retyped"] == ["top.txt"] and "top.txt" in lst["missing"] and lst["extra"] == []
    os.rmdir(f"{gen}/top.txt")
    shutil.copy2(os.path.join(w["base"], "top.txt"), f"{gen}/top.txt")
    # an entry that is neither file, directory nor link
    os.mkfifo(f"{gen}/a/pipe")
    lst = ST.list_generation(gen, entries, links, seal_check=False)
    assert not lst["ok"] and lst["other"] == 1 and lst["other_entries"] == ["a/pipe"]
    os.remove(f"{gen}/a/pipe")
    # an EMPTY directory the manifest implies by nothing: counted, never a listing mismatch (attest_generation judges the count against the seal record)
    os.mkdir(f"{gen}/empty")
    _reseal(gen)
    lst = ST.list_generation(gen, entries, links)
    assert lst["ok"] and lst["extra_dirs"] == ["empty"] and lst["n_extra_dirs"] == 1 and lst["dirs"] == 7 and lst["n_unsealed_dirs"] == 0
    # the seal check: a directory whose mode is not 0555 is named (`.` is the generation root)
    os.chmod(f"{gen}/a", 0o755)
    lst = ST.list_generation(gen, entries, links)
    assert lst["ok"] and lst["n_unsealed_dirs"] == 1 and lst["unsealed_dirs"] == ["a"]
    os.chmod(gen, 0o755)
    lst = ST.list_generation(gen, entries, links)
    assert lst["n_unsealed_dirs"] == 2 and set(lst["unsealed_dirs"]) == {".", "a"}
    _unseal(gen)
    os.rmdir(f"{gen}/empty")
    _reseal(gen)
    assert ST.list_generation(gen, entries, links)["ok"]
    # escaped names: the generation spells `d:x/sub%y/f:1.txt` as `d%3Ax/sub%25y/f%3A1.txt` and the link targets likewise; the listing
    # compares the TRUE names and the escaped spelling of the attested targets, so a correctly written generation lists clean
    base = _colon_tree(tmp_path)
    cfgp = _cfg(tmp_path, {"colon": {"base": base, "authority": "grading_host_copy", "exclude": ["__pycache__"]}}, name="colon.yaml")
    cfg = ST.load_config(cfgp)
    rc = ST.mirror_component(w["root"], cfg.destination("cluster_a"), cfg.components["colon"], runner=Runner())
    assert rc.outcome == "published", rc.as_dict()
    cgd = cfg.destination("cluster_a").gen_dir("colon")
    centries, clinks = _manifest(cgd, rc.content_sha8)
    assert os.path.isfile(f"{cgd}/{rc.content_sha8}/d%3Ax/sub%25y/f%3A1.txt") and "link:to" in clinks
    clst = ST.list_generation(f"{cgd}/{rc.content_sha8}", centries, clinks)
    assert clst["ok"] and clst["symlinks"] == 3 and clst["link_mismatch"] == [] and clst["n_unsealed_dirs"] == 0, clst
    # the plain-spelled attested target would NOT match the on-disk (escaped) one: the listing compares escape_target(attested), as the verify does
    assert ST.list_generation(f"{cgd}/{rc.content_sha8}", centries, {**clinks, "link:to": "d:x/sub%y/OTHER.txt"})["link_mismatch"] == ["link:to"]


# --------------------------------------------------------------------------------------------
# attest_generation: the facts, the listing, the seal, the content -- full the first time, a sample within a week, full again after
# --------------------------------------------------------------------------------------------


def test_attest_generation_checks_the_facts_the_listing_the_seal_then_walks_in_full_once_and_samples_for_a_week(world, tmp_path):
    w = world
    gd, sha8, gen = _publish(w)
    rec = ST.attest_generation(gd, sha8, "w1")
    assert rec["ok"] and rec["refusal"] is None and rec["content_sha8"] == sha8 and rec["window"] == "w1", rec
    a = rec["authority"]
    assert a["current"] == sha8 and a["sealed_n_dirs"] == 6 and a["n_files"] == 4 and a["n_symlinks"] == 1 and a["published_utc"] and a["publish_mode"] == "mirror"
    assert rec["listing"]["ok"] and rec["listing"]["entries"] == 10 and rec["seal"] == {"ok": True, "n_dirs": 6, "recorded_n_dirs": 6, "unsealed_dirs": [], "n_unsealed_dirs": 0,
                                                                                    "elapsed_s": rec["seal"]["elapsed_s"]}
    c = rec["content"]
    assert c["mode"] == "full" and c["ok"] and c["hashed"] == 4 and c["history_written"] is True and c["full_walk_utc"] == rec["utc"] and c["why"] == "no full walk recorded"
    hist = json.load(open(ST.attested_path(gd, sha8)))
    assert hist["schema"] == ST.ATTESTED_SCHEMA and hist["ok"] is True and hist["full_walk_utc"] == rec["utc"] and hist["content_sha8"] == sha8 and hist["window"] == "w1"
    # the next operation within a week: a seeded sample (every file of a 4-file generation), the recorded walk named; the history untouched
    rec2 = ST.attest_generation(gd, sha8, "w2")
    assert rec2["ok"] and rec2["content"]["mode"] == "sample" and rec2["content"]["sampled"] == 4 and rec2["content"]["hashed"] == 4
    assert rec2["content"]["full_walk_utc"] == rec["utc"] and "within 7 d" in rec2["content"]["why"] and rec2["content"]["history_written"] is False
    assert json.load(open(ST.attested_path(gd, sha8)))["full_walk_utc"] == rec["utc"]
    # the seed is a function of the operation and the generation: one operation samples the same files twice, two operations differ in general
    assert rec2["content"]["seed"] == ST._sample_seed("w2", sha8) != ST._sample_seed("w3", sha8)
    # eight days later: the full walk again, the history re-stamped
    later = _later(rec["utc"], 8)
    rec3 = ST.attest_generation(gd, sha8, "w3", now=lambda: later)
    assert rec3["ok"] and rec3["content"]["mode"] == "full" and "8.0 d old" in rec3["content"]["why"] and rec3["content"]["history_written"] is True
    assert json.load(open(ST.attested_path(gd, sha8)))["full_walk_utc"] == later
    # a read-only run (history=False) walks in full and writes nothing
    os.remove(ST.attested_path(gd, sha8))
    rec4 = ST.attest_generation(gd, sha8, "w4", history=False)
    assert rec4["ok"] and rec4["content"]["mode"] == "full" and rec4["content"]["history_written"] is False and not os.path.exists(ST.attested_path(gd, sha8))
    # (a) the authority facts, each refusal by name, nothing else run (listing None)
    cur = f"{gd}/current"
    os.remove(cur)
    os.symlink("deadbeef", cur)
    r = ST.attest_generation(gd, sha8, "w5")
    assert r["refusal"] == "current_moved" and "names deadbeef, not the pinned" in r["reason"] and r["listing"] is None and r["authority"]["current"] == "deadbeef"
    os.remove(cur)
    assert ST.attest_generation(gd, sha8, "w5")["refusal"] == "no_current"
    os.symlink(sha8, cur)
    for name, refusal in ((f"SEALED-{sha8}.json", "unsealed"), (f"PUBLISHED-{sha8}.json", "unrecorded"), (f"DEST-STAT-{sha8}.json", "no_dest_stat"),
                          (f"VERIFIED-{sha8}.json", "unverified"), (f"MANIFEST-{sha8}.sha256", "no_manifest")):
        os.rename(f"{gd}/{name}", f"{gd}/{name}.aside")
        r = ST.attest_generation(gd, sha8, "w5")
        assert r["refusal"] == refusal and r["listing"] is None, (name, r)
        os.rename(f"{gd}/{name}.aside", f"{gd}/{name}")
    for name, key, refusal in ((f"VERIFIED-{sha8}.json", "content_sha256", "verified_mismatch"), (f"SEALED-{sha8}.json", "content_sha8", "sealed_mismatch"),
                               (f"PUBLISHED-{sha8}.json", "content_sha8", "published_mismatch")):
        doc = json.load(open(f"{gd}/{name}"))
        shutil.copy2(f"{gd}/{name}", f"{gd}/{name}.aside")
        ST._write_json(f"{gd}/{name}", {**doc, key: "0" * len(str(doc[key]))})
        assert ST.attest_generation(gd, sha8, "w5")["refusal"] == refusal, name
        os.replace(f"{gd}/{name}.aside", f"{gd}/{name}")
    os.rename(gen, gen + ".aside")
    assert ST.attest_generation(gd, sha8, "w5")["refusal"] == "dangling"
    os.rename(gen + ".aside", gen)
    assert ST.attest_generation(gd, sha8, "w5")["ok"]
    # (b) the listing refuses the negative cases under an unchanged `current` -- an added subtree, a deleted subtree, a re-pointed link, a missing file
    _unseal(gen)
    _write(f"{gen}/planted/p.txt", b"p")
    _reseal(gen)
    r = ST.attest_generation(gd, sha8, "w6")
    assert r["refusal"] == "listing_mismatch" and "1 extra" in r["reason"] and "planted/p.txt" in r["reason"] and r["content"] is None and r["listing"]["extra"] == ["planted/p.txt"]
    _unseal(gen)
    shutil.rmtree(f"{gen}/planted")
    shutil.rmtree(f"{gen}/x")
    _reseal(gen)
    r = ST.attest_generation(gd, sha8, "w6")
    assert r["refusal"] == "listing_mismatch" and "1 missing" in r["reason"] and r["listing"]["missing"] == ["x/deep/er/f3.bin"]
    _unseal(gen)
    shutil.copytree(os.path.join(w["base"], "x"), f"{gen}/x")
    os.remove(f"{gen}/link_to_top")
    os.symlink("a/f2.jsonl", f"{gen}/link_to_top")
    _reseal(gen)
    r = ST.attest_generation(gd, sha8, "w6")
    assert r["refusal"] == "listing_mismatch" and "1 link target(s) differ" in r["reason"] and r["listing"]["link_mismatch"] == ["link_to_top"]
    _unseal(gen)
    os.remove(f"{gen}/link_to_top")
    os.symlink("top.txt", f"{gen}/link_to_top")
    os.remove(f"{gen}/top.txt")
    _reseal(gen)
    r = ST.attest_generation(gd, sha8, "w6")
    assert r["refusal"] == "listing_mismatch" and r["listing"]["missing"] == ["top.txt"], "a missing file is a listing miss (the first attestation's full walk would name it too)"
    assert ST.admit_generation(gd, sha8).refusal == "generation_changed" and ST.admit_generation(gd, sha8).missing == ["top.txt"]
    _unseal(gen)
    shutil.copy2(os.path.join(w["base"], "top.txt"), f"{gen}/top.txt")
    _reseal(gen)
    # (c) the seal: a lifted directory mode refuses; an added EMPTY directory (sealed too) moves the count off the seal record's
    os.chmod(f"{gen}/a", 0o755)
    r = ST.attest_generation(gd, sha8, "w6")
    assert r["refusal"] == "unsealed_directories" and "1 of 6" in r["reason"] and r["seal"]["unsealed_dirs"] == ["a"] and r["content"] is None
    os.chmod(f"{gen}/a", 0o555)
    os.chmod(gen, 0o755)
    os.mkdir(f"{gen}/empty")
    _reseal(gen)
    r = ST.attest_generation(gd, sha8, "w6")
    assert r["refusal"] == "directory_count_changed" and "7 director" in r["reason"] and "recorded 6" in r["reason"] and r["listing"]["extra_dirs"] == ["empty"]
    _unseal(gen)
    os.rmdir(f"{gen}/empty")
    _reseal(gen)
    assert ST.attest_generation(gd, sha8, "w6")["ok"]
    # (d) the content: a same-size, stat-preserving edit -- the sample over every file names it; so does the full walk (ctime moved: a one-link file)
    p = f"{gen}/a/f2.jsonl"
    _rewrite_preserving_stat(p, b"row1\nROW2\n")
    r = ST.attest_generation(gd, sha8, "w7")
    assert r["refusal"] == "content_changed" and r["content"]["mode"] == "sample" and r["content"]["mismatched"] == ["a/f2.jsonl"] and r["reason"].startswith("content (sample):")
    os.remove(ST.attested_path(gd, sha8))
    r = ST.attest_generation(gd, sha8, "w7")
    assert r["refusal"] == "content_changed" and r["content"]["mode"] == "full" and r["content"]["mismatched"] == ["a/f2.jsonl"] and not os.path.exists(ST.attested_path(gd, sha8)), \
        "a refused full walk records no history"
    _rewrite_preserving_stat(p, b"row1\nrow2\n")
    assert ST.attest_generation(gd, sha8, "w7")["ok"] and os.path.exists(ST.attested_path(gd, sha8))
    # a sample that MISSES the edit admits the generation -- and the reader's byte guarantee still refuses the edited input by name
    _rewrite_preserving_stat(p, b"row1\nROW2\n")
    rels = sorted(rel for _s, _z, rel in _manifest(gd, sha8)[0])
    seed = next(s for s in range(1000) if random.Random(s).sample(rels, 1) != ["a/f2.jsonl"])
    r = ST.attest_generation(gd, sha8, "w8", sample_n=1, seed=seed)
    assert r["ok"] and r["content"]["mode"] == "sample" and r["content"]["sampled"] == 1
    cache = CR.GenerationHashCache.from_pins(os.path.dirname(gd), {"relu_paired_run_state": sha8})
    with pytest.raises(AB.InputRefusal, match="bytes differ from the verified generation"):
        cache.record(p, os.stat(p), R.file_sha256(p))
    _rewrite_preserving_stat(p, b"row1\nrow2\n")
    # a sampled file that is GONE is generation_changed with the entry named
    sg = ST.sample_generation(gd, sha8, n=64)
    assert sg.ok and sg.sampled == 4 and sg.hashed == 4
    _unseal(gen)
    os.remove(f"{gen}/top.txt")
    sg = ST.sample_generation(gd, sha8, n=64)
    assert sg.refusal == "generation_changed" and sg.missing == ["top.txt"]
    shutil.copy2(os.path.join(w["base"], "top.txt"), f"{gen}/top.txt")
    _reseal(gen)
    os.rename(f"{gd}/DEST-STAT-{sha8}.json", f"{gd}/DEST-STAT-{sha8}.json.aside")
    assert ST.sample_generation(gd, sha8).refusal == "no_dest_stat" and ST.sample_generation(gd, "00000000").refusal == "no_manifest"
    os.rename(f"{gd}/DEST-STAT-{sha8}.json.aside", f"{gd}/DEST-STAT-{sha8}.json")


# --------------------------------------------------------------------------------------------
# attest_plan + the record the shards check
# --------------------------------------------------------------------------------------------


def _two_components(tmp_path, root):
    """Two components over their OWN trees (the `world` fixture already built tmp_path/canon), both published."""
    base1, base2 = str(tmp_path / "first"), str(tmp_path / "second")
    _write(os.path.join(base1, "a/f2.jsonl"), b"row1\nrow2\n")
    _write(os.path.join(base1, "a/b/f1.json"), b'{"k": 1}\n')
    _write(os.path.join(base1, "top.txt"), b"top level\n")
    _write(os.path.join(base1, "x/deep/f3.bin"), bytes(range(100)))
    os.symlink("top.txt", os.path.join(base1, "link_to_top"))
    _write(os.path.join(base2, "s/one.json"), b'{"one": 1}\n')
    _write(os.path.join(base2, "two.txt"), b"two\n")
    cfgp = _cfg(tmp_path, {"c_one": {"base": base1, "authority": "grading_host_copy"}, "c_two": {"base": base2, "authority": "grading_host_copy"}},
                adapters={"adapter_a": {"shards": 2, "components": ["c_one", "c_two"]}, "adapter_b": {"shards": 1, "components": ["c_one"]}}, name="two.yaml")
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    pins = {}
    for slug in ("c_one", "c_two"):
        r = ST.mirror_component(root, dest, cfg.components[slug], runner=Runner())
        assert r.outcome == "published", r.as_dict()
        pins[slug] = r.content_sha8
    return cfg, dest, pins


def _plan_doc(dest, window, adapters):
    return {"schema": CR.PLAN_SCHEMA, "window": window, "gen_dir": f"{dest.stores_dir}/gen", "adapters": adapters, "tasks": [{"index": 0}]}


def _run_S(*args, cwd="/"):
    return subprocess.run(["python3", "-S", "-m", "sisteer.registry.stores", *args], capture_output=True, text=True,
                          env={"PYTHONPATH": SRC, "PATH": os.environ["PATH"]}, cwd=cwd)


def test_attest_plan_attests_the_union_of_pins_in_threads_and_the_record_is_what_a_shard_checks(world, tmp_path):
    """The plan's cluster-routed adapters pin two components (one shared); the record is keyed by component, carries the window, the plan's
    sha and per generation the verdict; `attestation_admits` (the shard's O(1) check through `cluster_reads.admit_pins`) answers ok for this
    operation's pins and refuses by name otherwise: no record, another operation's, a component the record lacks, another generation, a
    refused one. A refused generation is a RESULT: the record is written and the stdlib CLI exits 0; only an unreadable plan is a failure."""
    w = world
    cfg, dest, pins = _two_components(tmp_path, w["root"])
    gen_dir = f"{dest.stores_dir}/gen"
    plan = _plan_doc(dest, "w7", {"adapter_a": {"route": "cluster", "pins": dict(pins)}, "adapter_b": {"route": "cluster", "pins": {"c_one": pins["c_one"]}},
                                  "ghost": {"route": "grading_host", "pins": {"c_one": "ffffffff"}}})
    assert ST.plan_pins(plan) == (pins, {})
    plan_path, out = str(tmp_path / "plan.json"), str(tmp_path / "attestation.json")
    ST._write_json(plan_path, plan)
    seen = []
    rec = ST.attest_plan(plan_path, out, threads=2, job_id="j1", log=seen.append)
    assert rec["ok"] and rec["refused"] == [] and set(rec["generations"]) == {"c_one", "c_two"} and rec["n_generations"] == 2 and rec["threads"] == 2
    assert rec["window"] == "w7" and rec["plan_sha256"] == R.file_sha256(plan_path) and rec["job_id"] == "j1" and rec["schema"] == ST.ATTESTATION_SCHEMA
    assert all(g["ok"] and g["content"]["mode"] == "full" for g in rec["generations"].values()) and any("[attest] ok" in l for l in seen)
    assert ST.read_attestation(out) == rec and json.load(open(out))["generations"]["c_two"]["content_sha8"] == pins["c_two"]
    for slug in pins:
        assert os.path.isfile(ST.attested_path(f"{gen_dir}/{slug}", pins[slug]))
    ok = ST.attestation_admits(rec, "w7", pins, path=out)
    assert all(v["ok"] and v["refusal"] is None and v["content_mode"] == "full" and v["attestation"] == out for v in ok.values()) and set(ok) == set(pins)
    assert CR.admit_pins(out, "w7", pins) == ok
    # the refusals, by name
    assert CR.admit_pins(str(tmp_path / "nope.json"), "w7", pins)["c_one"]["refusal"] == "no_attestation"
    assert ST.attestation_admits(rec, "w8", pins)["c_one"]["refusal"] == "attestation_window" and "operation w7's, not w8's" in ST.attestation_admits(rec, "w8", pins)["c_one"]["reason"]
    assert ST.attestation_admits(rec, "w7", {"c_three": "abcdef01"})["c_three"]["refusal"] == "attestation_missing"
    assert ST.attestation_admits(rec, "w7", {"c_one": "00000000"})["c_one"]["refusal"] == "attestation_pin"
    # a record file that is not one
    ST._write_json(str(tmp_path / "other.json"), {"schema": "something/1", "generations": {}})
    assert ST.read_attestation(str(tmp_path / "other.json")) is None and CR.admit_pins(str(tmp_path / "other.json"), "w7", pins)["c_one"]["refusal"] == "no_attestation"
    # the second operation: a sample for both; a subtree added under c_two's unchanged `current` refuses c_two alone, by name, and the record says so
    gen2 = f"{gen_dir}/c_two/{pins['c_two']}"
    _unseal(gen2)
    _write(f"{gen2}/s/added.json", b"{}")
    _reseal(gen2)
    plan8 = dict(plan, window="w8")
    ST._write_json(plan_path, plan8)
    rec2 = ST.attest_plan(plan_path, out, threads=4)
    assert not rec2["ok"] and rec2["refused"] == ["c_two"] and rec2["generations"]["c_one"]["ok"] and rec2["generations"]["c_one"]["content"]["mode"] == "sample"
    g2 = rec2["generations"]["c_two"]
    assert g2["refusal"] == "listing_mismatch" and g2["listing"]["extra"] == ["s/added.json"] and g2["content"] is None
    adm = ST.attestation_admits(rec2, "w8", pins, path=out)
    assert adm["c_one"]["ok"] and not adm["c_two"]["ok"] and adm["c_two"]["refusal"] == "attestation_refused" and adm["c_two"]["attestation_refusal"] == "listing_mismatch"
    assert "s/added.json" in adm["c_two"]["reason"]
    _unseal(gen2)
    os.remove(f"{gen2}/s/added.json")
    _reseal(gen2)
    # a pin conflict (a plan FILE pinning one component at two generations) is that component's refusal
    plan9 = _plan_doc(dest, "w9", {"adapter_a": {"route": "cluster", "pins": dict(pins)}, "adapter_b": {"route": "cluster", "pins": {"c_one": "11111111"}}})
    ST._write_json(plan_path, plan9)
    assert ST.plan_pins(plan9)[1] == {"c_one": sorted([pins["c_one"], "11111111"])}
    rec3 = ST.attest_plan(plan_path, out)
    assert rec3["refused"] == ["c_one"] and rec3["generations"]["c_one"]["refusal"] == "pin_conflict" and rec3["generations"]["c_two"]["ok"]
    # a crash inside one generation's check is its refusal, the other is still attested
    plan10 = _plan_doc(dest, "w10", {"adapter_a": {"route": "cluster", "pins": dict(pins, c_two="zz")}})
    ST._write_json(plan_path, plan10)
    rec4 = ST.attest_plan(plan_path, out)
    assert rec4["generations"]["c_two"]["refusal"] in ("error", "no_current", "current_moved") and rec4["generations"]["c_one"]["ok"]
    # the stdlib CLI attest.sbatch runs: exit 0 with the record (a refusal included), non-zero and no record for an unreadable plan
    _unseal(gen2)
    _write(f"{gen2}/s/added.json", b"{}")
    _reseal(gen2)
    ST._write_json(plan_path, plan8)
    out2 = str(tmp_path / "att2.json")
    r = _run_S("attest", "--plan", plan_path, "--out", out2, "--threads", "2", "--job-id", "77", "--json")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["schema"] == ST.ATTESTATION_SCHEMA and doc["job_id"] == "77" and doc["ok"] is False and doc["refused"] == ["c_two"]
    assert json.load(open(out2))["window"] == "w8" and "[attest]" in r.stderr
    r = _run_S("attest", "--plan", plan_path, "--out", out2)
    assert r.returncode == 0 and "attestation w8:" in r.stdout and "c_two: listing_mismatch" in r.stdout
    r = _run_S("attest", "--plan", str(tmp_path / "missing-plan.json"), "--out", str(tmp_path / "att3.json"))
    assert r.returncode != 0 and not os.path.exists(str(tmp_path / "att3.json"))
    _unseal(gen2)
    os.remove(f"{gen2}/s/added.json")
    _reseal(gen2)
    # a plan whose cluster adapters pin nothing: an empty, ok record (nothing to attest)
    ST._write_json(plan_path, _plan_doc(dest, "w11", {"only": {"route": "grading_host", "pins": {}}}))
    rec5 = ST.attest_plan(plan_path, out)
    assert rec5["ok"] and rec5["generations"] == {} and rec5["n_generations"] == 0


# --------------------------------------------------------------------------------------------
# the escape refusal
# --------------------------------------------------------------------------------------------


def test_generation_hash_cache_refuses_an_input_that_escapes_the_pinned_generation_before_any_byte_is_read(world, tmp_path):
    """An input whose canonical spelling lies under a pinned component's base must address bytes inside that component's pinned
    generation: its real path under the generation tree, or a home-tree copy aliased onto a generation entry. A stray file in a real
    directory of the home tree, a link out of it, a path into ANOTHER component's generation: refused by name, before the hash (the
    `InputHashCache.admit` seam `check_input` calls right after resolving the path). An input under no pinned base is admitted as ever."""
    w = world
    cfg, dest, pins = _two_components(tmp_path, w["root"])
    gen_dir = f"{dest.stores_dir}/gen"
    gen1, gen2 = f"{gen_dir}/c_one/{pins['c_one']}", f"{gen_dir}/c_two/{pins['c_two']}"
    home = tmp_path / "node_home"
    base1, base2 = home / "canon", home / "second"
    home.mkdir()
    base1.mkdir()                                                   # a REAL directory laid out entry by entry (a shared base's shape)
    for name in ("a", "x"):
        os.symlink(f"{gen1}/{name}", base1 / name)
    shutil.copy2(f"{gen1}/top.txt", base1 / "top.txt")             # a copy of a regular-file member, aliased onto its generation entry
    os.symlink(gen2, base2)                                         # an exclusive base linked whole
    cache = CR.GenerationHashCache.from_pins(gen_dir, pins)
    cache.alias_files([[str(base1 / "top.txt"), f"{gen1}/top.txt"]])
    cache.bind_bases({str(base1): gen1, str(base2): gen2})
    assert cache.bases[0][0] == str(base1) or cache.bases[0][0] == str(base2)
    # admitted: through a link into the component's generation, the aliased copy, the whole-linked base, a repo-resident file under no base
    for uri in (str(base1 / "a" / "f2.jsonl"), str(base1 / "top.txt"), str(base2 / "two.txt"), str(tmp_path / "repo" / "decl.yaml")):
        cache.admit(uri, os.path.realpath(uri) if os.path.lexists(uri) else uri)
    assert cache.escaped == []
    # refused: a stray file written into the real home-tree directory; a link out to a node-local path; a spelling under base1 that resolves into c_two's generation
    _write(str(base1 / "stray.json"), b"{}")
    with pytest.raises(AB.InputRefusal, match="escapes the pinned generation") as ei:
        cache.admit(str(base1 / "stray.json"), os.path.realpath(base1 / "stray.json"))
    assert str(base1) in str(ei.value) and gen1 in str(ei.value) and cache.escaped == [str(base1 / "stray.json")]
    os.symlink("/tmp/elsewhere", base1 / "out")
    with pytest.raises(AB.InputRefusal, match="escapes the pinned generation"):
        cache.admit(str(base1 / "out" / "f.json"), "/tmp/elsewhere/f.json")
    os.symlink(gen2, base1 / "sibling")
    with pytest.raises(AB.InputRefusal, match="not under that component's pinned generation"):
        cache.admit(str(base1 / "sibling" / "two.txt"), os.path.realpath(base1 / "sibling" / "two.txt"))
    assert len(cache.escaped) == 3
    # through check_input, under the pass scope: the stray file is refused before it is hashed (nothing recorded, nothing hashed)
    with AB.using_hash_cache(cache):
        inp = AB.check_input(str(base1 / "a" / "f2.jsonl"))
        assert inp.path == f"{gen1}/a/f2.jsonl" and cache.checked_against_manifest == 1
        with pytest.raises(AB.InputRefusal, match="escapes the pinned generation"):
            AB.check_input(str(base1 / "stray.json"))
    assert cache.hashed == 1 and os.path.realpath(base1 / "stray.json") not in cache.entries
    # a cache with no bases bound (the grading host's own, or a pins file without a home tree) admits every spelling as before
    plain = CR.GenerationHashCache.from_pins(gen_dir, pins)
    plain.admit(str(base1 / "stray.json"), os.path.realpath(base1 / "stray.json"))
    AB.InputHashCache().admit("/anything", "/anything")


def test_generation_hash_cache_admits_an_input_from_any_generation_pinned_at_a_shared_base_and_keeps_the_longest_base_rule(world, tmp_path):
    """`run_shard` bound the escape refusal's bases as a `{base: generation}` dict, so of two components sharing a base
    (`$GRADE`: grade_alphasteer + grade_arditi, disjoint members) only one generation survived and a valid input of the other was
    refused as escaping. `bind_bases` keeps EVERY generation pinned at a base -- from pairs (the shard's call) or a mapping of one or several --
    and `admit` accepts a real path under any of them, still refuses a stray by name (the refusal names every generation at the base), and
    the longest base still claims its subtree: a nested component's spelling must lie in ITS generation, not in the enclosing base's."""
    w = world
    cfg, dest, pins = _two_components(tmp_path, w["root"])
    gen_dir = f"{dest.stores_dir}/gen"
    gen1, gen2 = f"{gen_dir}/c_one/{pins['c_one']}", f"{gen_dir}/c_two/{pins['c_two']}"
    home = tmp_path / "node_home"
    shared = home / "grade"                                         # ONE base, two components: a real directory laid out entry by entry
    shared.mkdir(parents=True)
    for name in ("a", "x"):
        os.symlink(f"{gen1}/{name}", shared / name)                 # c_one's directory members
    shutil.copy2(f"{gen1}/top.txt", shared / "top.txt")            # c_one's file member: a copy, aliased
    os.symlink(f"{gen2}/s", shared / "s")                           # c_two's directory member
    shutil.copy2(f"{gen2}/two.txt", shared / "two.txt")            # c_two's file member
    # the shard's call shape: PAIRS, the one base twice (the dict comprehension that kept one generation is gone)
    cache = CR.GenerationHashCache.from_pins(gen_dir, pins)
    cache.alias_files([[str(shared / "top.txt"), f"{gen1}/top.txt"], [str(shared / "two.txt"), f"{gen2}/two.txt"]])
    cache.bind_bases([(str(shared), gen1), (str(shared), gen2)])
    assert cache.bases == [(str(shared), (gen1, gen2))]
    for uri in (str(shared / "a" / "f2.jsonl"), str(shared / "top.txt"), str(shared / "s" / "one.json"), str(shared / "two.txt")):
        cache.admit(uri, os.path.realpath(uri))
    assert cache.escaped == []
    # through check_input: both components' bytes are hashed and checked against THEIR OWN manifests
    with AB.using_hash_cache(cache):
        one = AB.check_input(str(shared / "a" / "f2.jsonl"))
        two = AB.check_input(str(shared / "s" / "one.json"))
    assert one.path == f"{gen1}/a/f2.jsonl" and two.path == f"{gen2}/s/one.json"
    assert cache.checked_against_manifest == 2 and cache.refused == [] and cache.escaped == []
    # a stray written into the shared directory is still refused before any byte is read, and the refusal names BOTH generations
    _write(str(shared / "stray.json"), b"{}")
    with pytest.raises(AB.InputRefusal, match="escapes the pinned generation") as ei:
        cache.admit(str(shared / "stray.json"), os.path.realpath(shared / "stray.json"))
    assert "2 generations pinned at that base" in str(ei.value) and gen1 in str(ei.value) and gen2 in str(ei.value)
    assert cache.escaped == [str(shared / "stray.json")]
    # the mapping shapes: several generations at a base (a duplicate binds once) and one (the earlier callers' `{base: generation}`)
    several = CR.GenerationHashCache.from_pins(gen_dir, pins)
    several.bind_bases({str(shared): [gen1, gen2, gen1]})
    assert several.bases == [(str(shared), (gen1, gen2))]
    one_only = CR.GenerationHashCache.from_pins(gen_dir, pins)
    one_only.bind_bases({str(shared): gen1})
    assert one_only.bases == [(str(shared), (gen1,))]
    with pytest.raises(AB.InputRefusal, match="not under that component's pinned generation"):
        one_only.admit(str(shared / "s" / "one.json"), os.path.realpath(shared / "s" / "one.json"))     # the defect's binding: c_two's input "escapes"
    # the longest base still rules: a component nested at shared/a and pinned to gen2 claims that subtree, so c_one's bytes there escape
    # while a spelling outside the nested base is judged against every generation at the enclosing base
    nested = CR.GenerationHashCache.from_pins(gen_dir, pins)
    nested.bind_bases([(str(shared), gen1), (str(shared), gen2), (str(shared / "a"), gen2)])
    assert nested.bases == [(str(shared / "a"), (gen2,)), (str(shared), (gen1, gen2))]
    with pytest.raises(AB.InputRefusal, match="not under that component's pinned generation") as ei2:
        nested.admit(str(shared / "a" / "f2.jsonl"), os.path.realpath(shared / "a" / "f2.jsonl"))
    assert str(shared / "a") in str(ei2.value) and gen2 in str(ei2.value)
    nested.admit(str(shared / "x" / "deep" / "f3.bin"), os.path.realpath(shared / "x" / "deep" / "f3.bin"))
    nested.admit(str(shared / "s" / "one.json"), os.path.realpath(shared / "s" / "one.json"))
    assert nested.escaped == [str(shared / "a" / "f2.jsonl")]


# --------------------------------------------------------------------------------------------
# the submission order, the cluster_a skip, the sbatch
# --------------------------------------------------------------------------------------------


def test_submit_sends_the_attestation_job_first_and_the_array_behind_it_and_the_cli_records_both_ids(tmp_path, monkeypatch):
    plan = {"tasks": [{"index": 0}, {"index": 1}], "tree": "/shared/gates/trees/abc", "window_root": "/shared/registry/reads/w1", "host": None, "window": "w1"}
    calls = []

    def runner(argv, **kw):
        calls.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, stdout=f"{500 + len(calls)}\n", stderr="")
    assert CR.submit_attest(plan, runner=runner) == "501"
    assert calls[0] == "sbatch --parsable -J attest-w1 /shared/gates/trees/abc/scripts/registry/landing/attest.sbatch /shared/registry/reads/w1 /shared/gates/trees/abc"
    assert CR.submit_array(plan, "/shared/envs/v", "reads-w1", runner=runner, dependency="501") == "502"
    assert "--array=0-1 --dependency=afterok:501 --kill-on-invalid-dep=yes /shared/gates/trees/abc/scripts/registry/landing/read_array.sbatch" in calls[1]
    assert calls[1].endswith("/shared/envs/v 0") and "--export" not in calls[0] + calls[1]
    assert CR.submit_array(plan, "/shared/envs/v", "reads-w1", runner=runner) == "503" and "--dependency" not in calls[2]
    with pytest.raises(CR.ClusterReadsError, match="no cluster task"):
        CR.submit_attest(dict(plan, tasks=[]), runner=runner)
    bad = lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout="", stderr="sbatch: error")   # noqa: E731
    with pytest.raises(CR.ClusterReadsError, match="attestation sbatch failed"):
        CR.submit_attest(plan, runner=bad)
    # the CLI: the attestation submitted FIRST, the array with the dependency on it, both ids on the plan's submit stage, the array's printed
    root = str(tmp_path / "root")
    os.makedirs(os.path.join(root, "state", CR.STATE_SUBDIR))
    CR._write_json(CR.plan_path(root, "w1"), dict(plan, schema=CR.PLAN_SCHEMA, adapters={}, fallback={}, stages={}))
    order = []
    monkeypatch.setattr(CR, "submit_attest", lambda plan, **kw: (order.append("attest"), "900")[1])
    monkeypatch.setattr(CR, "submit_array", lambda plan, venv, job_name, **kw: (order.append(("array", kw.get("dependency"))), "901")[1])
    assert cli.main(["--root", root, "cluster-reads", "submit", "--window", "w1", "--venv", "/shared/envs/v"]) == 0
    assert order == ["attest", ("array", "900")]
    stage = CR._load_json(CR.plan_path(root, "w1"))["stages"]["submit"]
    assert stage["job_id"] == "901" and stage["attest_job_id"] == "900" and stage["tasks"] == 2


def test_both_grading_host_admission_passes_skip_a_cluster_a_component_by_name(tmp_path, monkeypatch):
    """A `cluster_a` component's canonical copy is the published generation; the grading host holds a mirror that may lag or lead the push. Its
    admission is the attestation on the cluster: `adapter_admission` (the plan's pass 1) and `revalidate_plan` (pass 2) record `skipped`
    for it -- no manifest copy, no stat snapshot, no walk is needed -- while a `grading_host_copy` component beside it is still revalidated."""
    monkeypatch.setattr(ST, "HOME", str(tmp_path))
    base_b = str(tmp_path / "pushed")
    base_c = _canon(tmp_path)
    _write(os.path.join(base_b, "f.json"), b"{}")
    cfgp = _cfg(tmp_path, {"pushed": {"base": base_b, "authority": "cluster_a", "producer": "grading_host:relu"},
                           "carried": {"base": base_c, "authority": "grading_host_copy", "exclude": ["__pycache__", "/.pytest_cache"]}},
                adapters={"mixed": {"shards": 1, "components": ["pushed", "carried"]}, "only_pushed": {"shards": 1, "components": ["pushed"]}}, name="mixed.yaml")
    cfg = ST.load_config(cfgp)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    res = ST.adapter_admission(cfg, "mixed", {"pushed": "abcdef01", "carried": "12345678"}, root=root)
    assert res["pushed"].ok and res["pushed"].skipped and res["pushed"].generation == "abcdef01" and "attestation" in res["pushed"].reason
    assert res["pushed"].as_dict()["skipped"] is True and res["pushed"].as_dict()["ok"] is True
    assert not res["carried"].ok and res["carried"].refusal == "no_manifest" and res["carried"].skipped is False
    plan = {"adapters": {"only_pushed": {"route": "cluster", "pins": {"pushed": "abcdef01"}, "revalidation_post": {}},
                         "grading_host_one": {"route": "grading_host", "pins": {}}}, "fallback": {}, "stages": {}}
    assert CR.revalidate_plan(root, plan, cfg) == []
    assert plan["adapters"]["only_pushed"]["revalidation_post"]["pushed"]["skipped"] is True and plan["stages"]["revalidate_post"]["stale"] == []
    assert plan["fallback"] == {}


def test_attest_sbatch_keeps_the_cluster_rules():
    text = open(ATTEST_SBATCH, encoding="utf-8").read()
    assert subprocess.run(["bash", "-n", ATTEST_SBATCH], capture_output=True, text=True).returncode == 0
    directives = [l for l in text.splitlines() if l.startswith("#SBATCH")]
    assert "#SBATCH --qos=high" in directives and "#SBATCH --partition=general" in directives and "#SBATCH --open-mode=append" in directives
    assert "#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out" in directives and "#SBATCH --cpus-per-task=4" in directives and "#SBATCH --mem=16G" in directives
    assert "#SBATCH --time=00:30:00" in directives, "plan §1.4: 1 task, 4 CPU, 16 G, --time 00:30:00"
    effective = "\n".join(l for l in text.splitlines() if l.startswith("#SBATCH") or not l.lstrip().startswith("#"))
    assert "--export" not in effective and "--mem=0" not in text and "--gres" not in effective and "--delete" not in text
    assert not any(d.startswith("#SBATCH --export") or d.startswith("#SBATCH --gres") or d.startswith("#SBATCH --array") for d in directives), "one task, no array"
    assert "scripts/hf_env.sh" in text and "trap '' SIGTERM" in text and "python3 -S -m sisteer.registry.stores attest" in text and "srun " in text
    assert "--plan \"$WINDOW_ROOT/plan.json\"" in text and "--out \"$WINDOW_ROOT/attestation.json\"" in text and 'WINDOW_ROOT=${1:?WINDOW_ROOT} TREE=${2:?TREE}' in text
    assert not re.search(r"(?i)(TOKEN|API_KEY|SECRET|PASSWORD)\s*=", text) and "SISTEER_SECRET_CMD" not in text and "://" not in text
    assert os.access(ATTEST_SBATCH, os.X_OK)
