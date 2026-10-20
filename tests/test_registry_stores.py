"""Immutable store generations for the cluster-read adapters (`registry stores`): a grading_host_copy component becomes a
manifest-addressed generation that is PUBLISHED (the `current` rename) only when the cluster verify matches its manifest; a
same-size corruption in the copy is never published; a canonical change during the copy invalidates the attempt; unchanged
files hard-link across generations; `--dry-run` writes nothing; the stat cache reads only changed files; a cluster_a_origin
mismatch is recorded; `revalidate` refuses by name; the sbatch keeps the cluster rules; the new surfaces keep the vocabulary.
The destination is a local directory and ONE runner passes the shell steps and rsync through to real subprocesses while
intercepting `sbatch` (the verify / manifest runs in-process, the way store_verify.sbatch runs it) and the watcher.
"""
from __future__ import annotations

import configparser
import importlib.util
import glob
import json
import os
import random
import shlex
import shutil
import stat as stat_module
import subprocess
import sys

import pytest

from sisteer.registry import cli
from sisteer.registry import records as R
from sisteer.registry import stores as ST

import _rsync_fallback as RF  # the copy step where this host has no rsync: the fallback on PATH, never a skip (tests/_rsync_fallback.py)
from _env import permission_denial_enforced

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
SBATCH = os.path.join(REPO, "scripts", "registry", "landing", "store_verify.sbatch")
ATTEST_SBATCH = os.path.join(REPO, "scripts", "registry", "landing", "attest.sbatch")        # C5: the per-operation attestation job
UNITS = os.path.join(REPO, "systemd", "user")
YAML_LIVE = os.path.join(REPO, "configs", "registry", "cluster_reads.yaml")
WATCH = "/fake/bin/watch-job.sh"


# --------------------------------------------------------------------------------------------
# the world: a canonical tree, a local destination, a registry root, a config, one runner
# --------------------------------------------------------------------------------------------


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode) as fh:
        fh.write(data)


def _canon(tmp_path, n_extra=0):
    base = tmp_path / "canon"
    _write(str(base / "a" / "b" / "f1.json"), b'{"k": 1}\n')
    _write(str(base / "a" / "f2.jsonl"), b"row1\nrow2\n")
    _write(str(base / "top.txt"), b"top level\n")
    _write(str(base / "x" / "deep" / "er" / "f3.bin"), bytes(range(200)))
    _write(str(base / "__pycache__" / "junk.pyc"), b"pyc")
    _write(str(base / ".pytest_cache" / "v" / "x"), b"cache")
    os.symlink("top.txt", str(base / "link_to_top"))
    for i in range(n_extra):
        _write(str(base / "many" / f"file{i:03d}.txt"), f"content {i}\n".encode())
    return str(base)


GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.invalid"}


def _git(repo, *args):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def _code_repo(tmp_path):
    """A tiny git checkout standing in for the running code (the pinned paths committed at HEAD) and a bare repo standing in for
    the cluster's repo mirror (`$CLUSTER_REPO_MIRROR`) -- OUTSIDE the destination's shared root, so a dry run still leaves that
    root untouched. Idempotent per tmp_path; returns (source checkout, mirror)."""
    src, mirror = tmp_path / "code", tmp_path / "mirror.git"
    if not (src / ".git").exists():
        for rel in ("src/sisteer/__init__.py", "src/sisteer/registry/__init__.py", "src/sisteer/registry/stores.py"):
            _write(str(src / rel), b"# the pinned verifier\n")
        _write(str(src / "scripts" / "registry" / "landing" / "store_verify.sbatch"), b"#!/bin/bash\n# the pinned verifier\n")
        _git(src, "init", "-q", "-b", "main")
        _git(src, "add", "-A")
        _git(src, "commit", "-q", "-m", "pinned")
        subprocess.run(["git", "init", "-q", "--bare", str(mirror)], check=True, env=GIT_ENV)
    return str(src), str(mirror)


def _cfg(tmp_path, components, adapters=None, name="cfg.yaml", schema=None):
    # every world built here copies with rsync through PATH (the Runner's `nice ionice -c3 rsync`, the on-cluster push, the CLI children):
    # the real binary where the host has one (nothing changes), else the pure-Python fallback (the Modal CPU suite has no rsync). Every
    # destination here has host None, so the operands are local paths (no `-e`, no `host:` prefix) and the fallback's refusal of a remote
    # operand never fires (no opt-in)
    RF.ensure_rsync_on_path()
    import yaml
    src, mirror = _code_repo(tmp_path)
    doc = {"schema": schema or ST.CONFIG_SCHEMA, "written_utc": "2026-09-28T00:00:00Z",
           "destinations": {"cluster_a": {"host": None, "shared": str(tmp_path / "shared"), "watch_job": WATCH, "verify_cpus": 2,
                                          "verify_rate_mb_s": 50, "verify_min_time_s": 120, "verify_max_time_s": 3600,
                                          "repo_rel": mirror, "source_repo": src}},
           "components": components, "adapters": adapters or {"relu_paired": {"shards": 8, "components": list(components)}}}
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(p)


class Runner:
    """Shell steps and rsync run for real; `sbatch` runs the verify / manifest in-process and returns a job id; the watcher
    reports DONE (or the terminal line `fail_watch` names). `before_verify(gen_dir)` runs between the copy and the verify."""

    def __init__(self, before_verify=None, fail_watch=None, before_rsync=None):
        self.calls = []
        self.before_verify = before_verify
        self.fail_watch = fail_watch
        self.before_rsync = before_rsync
        self.job = 700
        self.watch_timeouts = []                     # the in-process bound each watch was run under (round 2 HIGH)

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "sbatch":
            i = next(k for k, a in enumerate(argv) if a.endswith("store_verify.sbatch"))
            script = argv[i + 1:]
            self.job += 1
            if script[0] == "verify":
                gen, man, out, workers = script[1:5]
                rest = script[5:]
                # what follows the four positionals rides through to the entry point, as the sbatch forwards it: `--attestation FILE`
                # (a re-attestation's keyed manifest .json; `stores attest`)
                attestation = next((rest[k + 1] for k, a in enumerate(rest) if a == "--attestation"), None)
                if self.before_verify:
                    self.before_verify(gen)
                if not self.fail_watch:                       # a job that dies writes no report
                    rep, dest_stat = ST.verify_generation(gen, man, workers=int(workers), job_id=str(self.job), attestation=attestation)
                    ST.write_verified(out, rep, dest_stat)
            elif script[0] == "manifest-only":
                d, prefix, workers = script[1:4]
                rest = script[4:]
                members = [rest[k + 1] for k, a in enumerate(rest) if a == "--member"] or ["."]
                excl = [rest[k + 1] for k, a in enumerate(rest) if a == "--exclude"]
                grading_host_root = next((rest[k + 1] for k, a in enumerate(rest) if a == "--grading-host-root"), None)
                ST.manifest_in_place(d, prefix, members, excl, int(workers), job_id=str(self.job), grading_host_root=grading_host_root)
            return subprocess.CompletedProcess(argv, 0, stdout=f"{self.job}\n", stderr="")
        if argv[0] == WATCH:
            jid = argv[argv.index("--slurm") + 1]
            self.watch_timeouts.append(timeout_s)
            if self.fail_watch:
                return subprocess.CompletedProcess(argv, 0, stdout=f"RUNNING {jid} node=test\n{self.fail_watch}\n", stderr="")
            return subprocess.CompletedProcess(argv, 0, stdout=f"RUNNING {jid} node=test\nDONE {jid} state=COMPLETED elapsed=00:00:03\n", stderr="")
        if "rsync" in argv and self.before_rsync:
            self.before_rsync()
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, input=input, check=check)

    def count(self, word):
        return sum(1 for a in self.calls if word in a or any(word in x for x in a[:1]))


@pytest.fixture
def world(tmp_path):
    base = _canon(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    cfgp = _cfg(tmp_path, {"relu_paired_run_state": {"base": base, "authority": "grading_host_copy", "origin": "mixed",
                                                     "exclude": ["__pycache__", "/.pytest_cache"]}})
    cfg = ST.load_config(cfgp)
    return {"base": base, "root": root, "cfg": cfg, "cfgp": cfgp, "dest": cfg.destination("cluster_a"), "comp": cfg.components["relu_paired_run_state"],
            "gen_dir": cfg.destination("cluster_a").gen_dir("relu_paired_run_state")}


def _tree(top):
    out = {}
    for dp, dns, fns in os.walk(top):
        for fn in fns:
            p = os.path.join(dp, fn)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, top)] = fh.read()
    return out


# --------------------------------------------------------------------------------------------
# the generation flow
# --------------------------------------------------------------------------------------------


def test_mirror_publishes_a_verified_generation_with_the_documented_layout(world):
    w = world
    rn = Runner()
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert res.outcome == "published", res.as_dict()
    sha8 = res.content_sha8
    assert len(sha8) == 8 and res.published == sha8 and res.n_files == 4 and res.hashed == 4 and res.from_cache == 0
    gd = w["gen_dir"]
    # the tree: only the manifest's files, at their base-relative paths; caches excluded; the symlink carried as a symlink (not a manifest entry)
    assert sorted(ST._walk_plain_dir(f"{gd}/{sha8}")) == ["a/b/f1.json", "a/f2.jsonl", "top.txt", "x/deep/er/f3.bin"]
    assert os.path.islink(f"{gd}/{sha8}/link_to_top") and not os.path.exists(f"{gd}/{sha8}/__pycache__")
    # the manifest beside it, the hash pair, the VERIFIED report, and `current`
    lines = ST.read_manifest_lines(f"{gd}/MANIFEST-{sha8}.sha256")
    assert [rel for _s, _z, rel in lines] == ["a/b/f1.json", "a/f2.jsonl", "top.txt", "x/deep/er/f3.bin"]
    assert lines[2] == (R.file_sha256(os.path.join(w["base"], "top.txt")), 10, "top.txt")
    meta = json.load(open(f"{gd}/MANIFEST-{sha8}.json"))
    assert meta["schema"] == ST.MANIFEST_SCHEMA and meta["content_sha256"][:8] == sha8 and meta["n_files"] == 4 and meta["authority"] == "grading_host_copy"
    assert set(meta["manifest_rules"]) == {"files", "symlinks", "name_size", "content", "manifest_lines", "directories"}
    assert meta["n_symlinks"] == 1 and meta["symlinks"] == {"link_to_top": "top.txt"}, "a symlink is attested by its path and literal target"
    assert meta["name_size_sha256"] != meta["content_sha256"] and len(meta["name_size_sha256"]) == 64
    ver = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert ver["ok"] is True and ver["hashed"] == 4 and ver["content_sha256"] == meta["content_sha256"] and ver["missing"] == ver["corrupt"] == ver["extra"] == []
    assert os.readlink(f"{gd}/current") == sha8
    # the grading host state: manifest + stat snapshot + cache per slug, and the status file the page reads
    sdir = ST.component_state_dir(w["root"], "relu_paired_run_state")
    # the grading host's copy: the lines, the first attestation, the attestation keyed by its listings (immutable per (content, listings)), the stat snapshot
    assert sorted(os.listdir(sdir)) == [f"MANIFEST-{sha8}.{meta['listings_sha256'][:8]}.json", f"MANIFEST-{sha8}.json", f"MANIFEST-{sha8}.sha256",
                                        f"MANIFEST-{sha8}.stat.json", "stat_cache.json"]
    snap = json.load(open(f"{sdir}/MANIFEST-{sha8}.stat.json"))["files"]
    assert set(snap) == {rel for _s, _z, rel in lines} | {"link_to_top"} and all(len(v) == 4 for v in snap.values())
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["published"] == sha8 and st["generation"]["state"] == "published" and st["latest_manifest"]["content_sha8"] == sha8
    assert st["adapters"] == ["relu_paired"] and st["verify_rate_mb_s"] is not None
    # the submission carried the cluster rules: no --export, no GPU, high, singleton, a --time, positional script args
    sb = next(a for a in rn.calls if a[0] == "sbatch")
    joined = " ".join(sb)
    assert "--export" not in joined and "--gres" not in joined and "--qos=high" in joined and "--dependency=singleton" in joined
    assert "--job-name=store-verify-relu_paired_run_state" in joined and any(a.startswith("--time=") for a in sb) and "--mem=32G" in joined
    assert sb[sb.index(w["dest"].sbatch_path) + 1:] == ["verify", f"{gd}/{sha8}", f"{gd}/MANIFEST-{sha8}.sha256", f"{gd}/VERIFIED-{sha8}.json", "2"]
    # the watcher ran in the foreground with the job id
    assert any(a[0] == WATCH and a[2] == sb_id for a in rn.calls for sb_id in [ver["job_id"]])
    # the rsync: never --delete; nice/ionice; --partial; --relative; no --link-dest on the first generation
    rs = next(a for a in rn.calls if "rsync" in a)
    assert rs[:5] == ["nice", "-n", "10", "ionice", "-c3"] and "--delete" not in " ".join(rs) and "-rlt" in rs and "--relative" in rs
    assert not any(a.startswith("--link-dest") for a in rs) and "--exclude=__pycache__" in rs and "--exclude=/.pytest_cache" in rs
    assert f"--chmod={ST.STAGING_DIR_CHMOD}" in rs and ST.STAGING_DIR_CHMOD == "Du+w"      # the staging copy's directories are owner-writable (the seal comes after the verify)
    # a second run: nothing changed -> unchanged; no rsync, no sbatch, every sha from the cache
    rn2 = Runner()
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2)
    assert res2.outcome == "unchanged" and res2.from_cache == 4 and res2.hashed == 0
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn2.calls)


def test_same_size_corruption_in_the_copy_is_caught_by_the_verify_and_never_published(world):
    """A generation differing in ONE same-size file from its manifest is refused."""
    w = world

    def corrupt(gen):
        p = os.path.join(gen, "a", "f2.jsonl")
        data = bytearray(open(p, "rb").read())
        data[0] ^= 0x01
        st = os.stat(p)
        with open(p, "wb") as fh:
            fh.write(bytes(data))
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))       # same size, same mtime: only the bytes differ

    rn = Runner(before_verify=corrupt)
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert res.outcome == "verify_failed", res.as_dict()
    gd = w["gen_dir"]
    sha8 = res.content_sha8
    assert not os.path.lexists(f"{gd}/current") and not os.path.exists(f"{gd}/{sha8}")
    assert res.detail["corrupt"] == 1 and res.detail["first"]["corrupt"] == ["a/f2.jsonl"] and res.detail["missing"] == 0 and res.detail["extra"] == 0
    # the rename is the verdict of a PRESENT report of this job (read on the first attempt), never of an unread one
    assert res.detail["verified_present"] is True and res.detail["report_wait"] == {"attempts": 1, "waited_s": 0.0, "last": "present"}
    renamed = res.detail["renamed_to"]
    assert os.path.basename(renamed).startswith(f".failed-{sha8}-") and os.path.isdir(renamed)
    ver = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert ver["ok"] is False and ver["corrupt"] == ["a/f2.jsonl"]
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st.get("published") is None and st["generation"]["state"] == "verify_failed"
    # the same content, copied cleanly on the next run: a fresh tree (the failed one was renamed away), verified, published
    rn2 = Runner()
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2)
    assert res2.outcome == "published" and res2.content_sha8 == sha8 and os.readlink(f"{gd}/current") == sha8


def test_verify_job_failure_leaves_the_tree_for_the_next_run(world):
    w = world
    rn = Runner(fail_watch="FAILED 701 state=NODE_FAIL exit=1:0")
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert res.outcome == "verify_job_failed" and res.published is None
    gd = w["gen_dir"]
    assert os.path.isdir(f"{gd}/{res.content_sha8}") and not os.path.lexists(f"{gd}/current")
    assert ST.load_state(w["root"])["components"]["relu_paired_run_state"]["generation"]["state"] == "verify_job_failed"
    assert not os.path.exists(f"{gd}/VERIFIED-{res.content_sha8}.json")
    rn2 = Runner()
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2)
    assert res2.outcome == "published" and os.readlink(f"{gd}/current") == res.content_sha8
    assert any("rsync" in a for a in rn2.calls) and any(a[0] == "sbatch" for a in rn2.calls), "the tree is re-synced (idempotent) and verified"
    # a run that died between the verify and the publish: the matching report is already there -> published without a copy or a job
    os.remove(f"{gd}/current")
    rn3 = Runner()
    res3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn3)
    assert res3.outcome == "published_existing_verified" and os.readlink(f"{gd}/current") == res.content_sha8
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn3.calls)


def test_canonical_mutation_during_the_copy_invalidates_the_generation(world):
    """A change to the canonical copy between the manifest and the post-copy stat walk invalidates the attempt."""
    w = world
    victim = os.path.join(w["base"], "a", "b", "f1.json")

    def mutate():
        with open(victim, "wb") as fh:
            fh.write(b'{"k": 2}\n')

    rn = Runner(before_rsync=mutate)
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert res.outcome == "invalidated", res.as_dict()
    assert res.detail["moved"] == 1 and res.detail["moved_first"] == ["a/b/f1.json"]
    gd = w["gen_dir"]
    assert not os.path.lexists(f"{gd}/current") and not os.path.exists(f"{gd}/{res.content_sha8}")
    assert os.path.basename(res.detail["renamed_to"]).startswith(f".invalid-{res.content_sha8}-") and os.path.isdir(res.detail["renamed_to"])
    assert not any(a[0] == "sbatch" for a in rn.calls), "no verify is submitted for an invalidated generation"
    assert not os.path.exists(f"{gd}/MANIFEST-{res.content_sha8}.sha256"), "no manifest is pushed beside an invalidated generation"
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["generation"]["state"] == "invalidated" and st["last_run"]["moved_first"] == ["a/b/f1.json"] and st.get("published") is None
    assert res.ok, "an invalidation is an expected event (retried next tick), not a failure exit"
    # the next run (canonical quiet) publishes the NEW content under a new sha8
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res2.outcome == "published" and res2.content_sha8 != res.content_sha8 and os.readlink(f"{gd}/current") == res2.content_sha8


def test_unchanged_files_hard_link_across_generations(world):
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r1.outcome == "published"
    with open(os.path.join(w["base"], "top.txt"), "ab") as fh:
        fh.write(b"more\n")
    rn = Runner()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "published" and r2.content_sha8 != r1.content_sha8
    gd = w["gen_dir"]
    g1, g2 = f"{gd}/{r1.content_sha8}", f"{gd}/{r2.content_sha8}"
    for rel in ("a/b/f1.json", "a/f2.jsonl", "x/deep/er/f3.bin"):
        assert os.stat(f"{g1}/{rel}").st_ino == os.stat(f"{g2}/{rel}").st_ino, rel
    assert os.stat(f"{g1}/top.txt").st_ino != os.stat(f"{g2}/top.txt").st_ino
    assert open(f"{g1}/top.txt", "rb").read() == b"top level\n", "the published generation is never written again"
    assert os.readlink(f"{gd}/current") == r2.content_sha8 and r2.detail["link_dest"] == g1
    rs = next(a for a in rn.calls if "rsync" in a)
    assert f"--link-dest={g1}" in rs
    assert r2.hashed == 1 and r2.from_cache == 3, "only the changed file was read for the second manifest"
    # both manifests and VERIFIED reports stay; the old generation is intact
    assert os.path.exists(f"{gd}/VERIFIED-{r1.content_sha8}.json") and os.path.exists(f"{gd}/MANIFEST-{r1.content_sha8}.json")


def test_dry_run_writes_nothing(world):
    w = world
    before_state = _tree(w["root"])
    rn = Runner()
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, dry_run=True)
    assert res.outcome == "dry_run" and res.n_files == 4 and res.bytes == 10 + 9 + 10 + 200 and res.detail["writes"] == "nothing"
    assert res.detail["n_symlinks"] == 1 and res.detail["would_rsync_to"].endswith("/gen/relu_paired_run_state/<content_sha8>/")
    assert _tree(w["root"]) == before_state and not os.path.exists(str(w["dest"].shared)) and rn.calls == []


def test_stat_cache_reads_only_changed_files(world):
    w = world
    calls = []

    def hasher(p):
        calls.append(p)
        return R.file_sha256(p)

    sdir = ST.component_state_dir(w["root"], "relu_paired_run_state")
    m1 = ST.build_manifest(w["comp"], sdir, hasher=hasher)
    assert len(calls) == 4 and m1.hashed == 4 and m1.from_cache == 0
    calls.clear()
    m2 = ST.build_manifest(w["comp"], sdir, hasher=hasher)
    assert calls == [] and m2.hashed == 0 and m2.from_cache == 4 and m2.content_sha256 == m1.content_sha256
    # an edit: one file re-read; the others from the cache
    with open(os.path.join(w["base"], "a", "f2.jsonl"), "ab") as fh:
        fh.write(b"row3\n")
    calls.clear()
    m3 = ST.build_manifest(w["comp"], sdir, hasher=hasher)
    assert [os.path.basename(p) for p in calls] == ["f2.jsonl"] and m3.hashed == 1 and m3.from_cache == 3 and m3.content_sha256 != m2.content_sha256
    # a same-size, same-mtime rewrite still moves ctime -> re-read (full-resolution key, never mtime alone)
    p = os.path.join(w["base"], "top.txt")
    st = os.stat(p)
    with open(p, "wb") as fh:
        fh.write(b"TOP LEVEL\n")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    calls.clear()
    m4 = ST.build_manifest(w["comp"], sdir, hasher=hasher)
    assert [os.path.basename(p) for p in calls] == ["top.txt"] and m4.content_sha256 != m3.content_sha256
    cache = json.load(open(os.path.join(sdir, ST.STAT_CACHE)))
    assert set(cache) == {"a/b/f1.json", "a/f2.jsonl", "top.txt", "x/deep/er/f3.bin"} and set(cache["top.txt"]) == {"size", "mtime_ns", "ino", "ctime_ns", "sha256"}
    # a file changing WHILE hashed (twice) is refused by name
    flaky = os.path.join(w["base"], "x", "deep", "er", "f3.bin")

    def moving(p):
        if p == flaky:
            with open(p, "ab") as fh:
                fh.write(b"!")
        return R.file_sha256(p)

    os.utime(flaky)
    with pytest.raises(ST.ManifestRefusal, match="changed while being hashed"):
        ST.build_manifest(w["comp"], sdir, hasher=moving)


def test_two_concurrent_mirror_runs_publish_at_most_one(world, monkeypatch):
    w = world
    rn = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn)
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    with ST.MirrorLock(w["root"]) as held:
        assert held.held
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli.main(["stores", "mirror", "--dest", "cluster_a"])
        assert rc == 0 and "another run holds" in out.getvalue()
        assert rn.calls == [] and not os.path.exists(w["dest"].shared)
    # lock released: the run proceeds and publishes exactly one generation
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = cli.main(["stores", "mirror", "--dest", "cluster_a"])
    assert rc == 0
    lines = [json.loads(l) for l in out.getvalue().strip().splitlines()]
    assert lines[0]["outcome"] == "published" and lines[-1]["summary"] and lines[-1]["ok"] == 1
    assert [d for d in os.listdir(w["gen_dir"]) if not d.startswith((".", "MANIFEST", "VERIFIED", "DEST-STAT", "SEALED", "PUBLISHED", "current"))] == [lines[0]["content_sha8"]]
    assert os.path.exists(os.path.join(w["root"], "state", "last_run", "stores-mirror.json"))


# --------------------------------------------------------------------------------------------
# cluster_a_origin: `stores check` records match / mismatch
# --------------------------------------------------------------------------------------------


def test_cluster_a_origin_mismatch_records_authority_mismatch_then_match(tmp_path):
    mirror = _canon(tmp_path / "m")
    original = tmp_path / "shared" / "exp" / "orig"
    shutil.copytree(mirror, str(original), symlinks=True)
    with open(str(original / "a" / "f2.jsonl"), "wb") as fh:
        fh.write(b"row1\nROW2\n")                                 # same size, one byte differs
    root = str(tmp_path / "reg")
    cfgp = _cfg(tmp_path, {"anchors": {"base": mirror, "authority": "cluster_a_origin", "origin": "cluster_a", "original": "exp/orig",
                                       "exclude": ["__pycache__", "/.pytest_cache"]}},
                adapters={"relu_paired": {"shards": 8, "components": ["anchors"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    comp = cfg.components["anchors"]
    assert dest.resolve(comp.original) == str(original)
    # mirror skips it (no copy for an in-place original)
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "skipped"
    rn = Runner()
    rec = ST.check_component(root, dest, comp, runner=rn)
    assert rec["state"] == "mismatch", rec
    assert rec["differing"]["sha_or_size_differs"] == ["a/f2.jsonl"] and rec["differing"]["counts"] == {"only_grading_host": 0, "only_original": 0, "sha_or_size_differs": 1}
    assert rec["grading_host"]["n_files"] == rec["original_manifest"]["n_files"] == 4 and rec["grading_host"]["name_size_sha256"] == rec["original_manifest"]["name_size_sha256"]
    assert rec["grading_host"]["content_sha256"] != rec["original_manifest"]["content_sha256"]
    apath = os.path.join(ST.component_state_dir(root, "anchors"), ST.AUTHORITY_FILE)
    assert json.load(open(apath))["state"] == "mismatch" and ST.authority_state(root, "anchors")["state"] == "mismatch"
    st = ST.load_state(root)["components"]["anchors"]
    assert st["authority_state"]["state"] == "mismatch" and st["original"] == str(original)
    # the stamp is `stores check`'s own namespace, red here (read as a mirror outcome it would call a MATCH red too)
    assert st["last_run"]["outcome"] == "check:mismatch" and ST.last_run_level(st["last_run"]["outcome"]) == "red"
    sb = next(a for a in rn.calls if a[0] == "sbatch")
    tail = sb[sb.index(dest.sbatch_path) + 1:]
    assert tail[:4] == ["manifest-only", str(original), tail[2], "2"] and tail[2].startswith(dest.origin_manifest_dir("anchors") + "/MANIFEST-")
    assert tail[4:] == ["--member", ".", "--exclude", "__pycache__", "--exclude", "/.pytest_cache", "--grading-host-root", comp.root_spelling], \
        "the original is walked by the mirror's rules; the mirror's spelling rides along"
    assert "--export" not in " ".join(sb) and "--job-name=store-manifest-anchors" in sb
    # the cluster-built manifest records the grading host spelling it was handed, never the compute node's home (and no home_rel)
    theirs_meta = json.load(open(tail[2] + ".json"))
    assert theirs_meta["root"] == comp.root_spelling == rec["original_manifest"]["root"] and theirs_meta["home_rel"] is None and "spelling" in theirs_meta
    # the original's manifest landed on the destination beside the generations
    assert os.path.exists(tail[2] + ".sha256") and os.path.exists(tail[2] + ".json")
    # repaired: match
    with open(str(original / "a" / "f2.jsonl"), "wb") as fh:
        fh.write(b"row1\nrow2\n")
    rec2 = ST.check_component(root, dest, comp, runner=Runner())
    assert rec2["state"] == "match" and rec2["differing"]["counts"] == {"only_grading_host": 0, "only_original": 0, "sha_or_size_differs": 0}
    assert ST.authority_state(root, "anchors")["state"] == "match"
    st2 = ST.load_state(root)["components"]["anchors"]
    assert st2["last_run"]["outcome"] == "check:match" and ST.last_run_level(st2["last_run"]["outcome"]) == "green"
    from sisteer.registry import status as STATUS
    so = STATUS.slurm_only_section(root, state_dir=str(tmp_path / "landing"))
    row = next(r for r in so["authority"]["rows"] if r["slug"] == "anchors")
    assert row["mirror_outcome"] == "check:match" and row["mirror_level"] == "green" and row["level"] == "green" and so["authority"]["level"] == "green"
    assert not any(p.startswith("store mirror") for p in so["problems"]), so["problems"]
    # a failed cluster job is an error state, never a match
    rec3 = ST.check_component(root, dest, comp, runner=Runner(fail_watch="FAILED 9 state=TIMEOUT"))
    assert rec3["state"] == "error" and "TIMEOUT" in rec3["error"]
    assert ST.load_state(root)["components"]["anchors"]["last_run"]["outcome"] == "check:error" and ST.last_run_level("check:error") == "red"


# --------------------------------------------------------------------------------------------
# revalidate: the admission helper
# --------------------------------------------------------------------------------------------


def test_revalidate_admits_the_pinned_generation_and_refuses_changes_by_name(world):
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gen = r1.content_sha8
    kw = dict(root=w["root"], cfg=w["cfg"])
    rv = ST.revalidate("relu_paired_run_state", gen, **kw)
    assert rv.ok and rv.refusal is None and rv.n_files == 4 and rv.moved == 0 and rv.sampled == 4 and rv.hashed == 4, rv.as_dict()
    assert rv.seed == int(gen, 16) and ST.revalidate("relu_paired_run_state", gen, seed=7, **kw).seed == 7
    # unknown generation / component
    assert ST.revalidate("relu_paired_run_state", "00000000", **kw).refusal == "no_manifest"
    assert ST.revalidate("nope", gen, **kw).refusal == "no_component"
    # a same-size edit (mtime moves): hashed, refused by name
    p = os.path.join(w["base"], "a", "f2.jsonl")
    with open(p, "wb") as fh:
        fh.write(b"row1\nROW2\n")
    rv = ST.revalidate("relu_paired_run_state", gen, **kw)
    assert not rv.ok and rv.refusal == "content_changed" and rv.mismatched == ["a/f2.jsonl"] and rv.moved == 1 and "a/f2.jsonl" in rv.reason
    # restore the bytes: a moved stat over unchanged content is admitted and counted
    with open(p, "wb") as fh:
        fh.write(b"row1\nrow2\n")
    rv = ST.revalidate("relu_paired_run_state", gen, **kw)
    assert rv.ok and rv.moved == 1 and rv.moved_unchanged == 1
    # a same-size, mtime-preserving edit: ctime still moves on this filesystem, and the seeded sample covers every file at this size anyway
    st = os.stat(p)
    with open(p, "wb") as fh:
        fh.write(b"row1\nRoW2\n")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    rv = ST.revalidate("relu_paired_run_state", gen, **kw)
    assert not rv.ok and rv.refusal == "content_changed" and rv.mismatched == ["a/f2.jsonl"]
    with open(p, "wb") as fh:
        fh.write(b"row1\nrow2\n")
    # a file added / removed: the file set changed
    extra = os.path.join(w["base"], "a", "new.txt")
    _write(extra, b"new\n")
    rv = ST.revalidate("relu_paired_run_state", gen, **kw)
    assert not rv.ok and rv.refusal == "file_set_changed" and rv.extra == ["a/new.txt"]
    os.remove(extra)
    os.remove(os.path.join(w["base"], "top.txt"))
    rv = ST.revalidate("relu_paired_run_state", gen, **kw)
    assert rv.refusal == "file_set_changed" and rv.missing == ["top.txt"]
    # the stat snapshot is required (no snapshot -> named refusal, never a hash-only guess)
    _write(os.path.join(w["base"], "top.txt"), b"top level\n")
    os.remove(os.path.join(ST.component_state_dir(w["root"], "relu_paired_run_state"), f"MANIFEST-{gen}.stat.json"))
    assert ST.revalidate("relu_paired_run_state", gen, **kw).refusal == "no_stat_snapshot"


def test_revalidate_sample_is_one_percent_or_sixty_four_and_seeded(tmp_path):
    base = _canon(tmp_path, n_extra=300)
    root = str(tmp_path / "reg")
    cfg = ST.load_config(_cfg(tmp_path, {"big": {"base": base, "authority": "grading_host_copy", "exclude": ["__pycache__", "/.pytest_cache"]}}))
    comp = cfg.components["big"]
    r = ST.mirror_component(root, cfg.destination("cluster_a"), comp, runner=Runner())
    assert r.outcome == "published" and r.n_files == 304
    calls = []

    def hasher(p):
        calls.append(p)
        return R.file_sha256(p)

    rv = ST.revalidate("big", r.content_sha8, root=root, cfg=cfg, hasher=hasher)
    assert rv.ok and rv.sampled == 64 and rv.hashed == 64 and len(calls) == 64, "1 % of 304 is 4; the floor is 64"
    calls2 = []
    ST.revalidate("big", r.content_sha8, root=root, cfg=cfg, hasher=lambda p: (calls2.append(p), R.file_sha256(p))[1])
    assert calls == calls2, "the same seed samples the same files"
    calls3 = []
    ST.revalidate("big", r.content_sha8, root=root, cfg=cfg, seed=1, hasher=lambda p: (calls3.append(p), R.file_sha256(p))[1])
    assert set(calls3) != set(calls)
    rv = ST.revalidate("big", r.content_sha8, root=root, cfg=cfg, sample_fraction=0.5)
    assert rv.sampled == 152


# --------------------------------------------------------------------------------------------
# CLI surfaces, config validation, the cluster entry points, the sbatch, the units, the vocabulary
# --------------------------------------------------------------------------------------------


def test_cli_status_revalidate_check_and_dry_run(world, monkeypatch, capsys):
    w = world
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    row = data["components"]["relu_paired_run_state"]
    assert row["authority"] == "grading_host_copy" and row["published"] is None and row["latest_manifest"] is None and row["authority_state"] is None
    assert data["adapters"] == {"relu_paired": {"shards": 8, "components": ["relu_paired_run_state"]}}
    assert cli.main(["stores", "mirror", "--dry-run", "--component", "relu_paired_run_state"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out.splitlines()[0])["outcome"] == "dry_run" and json.loads(out.splitlines()[-1])["dry_run"] is True
    assert not os.path.exists(w["dest"].shared)
    assert cli.main(["stores", "mirror"]) == 0            # a single declared destination needs no --dest
    sha8 = json.loads(capsys.readouterr().out.splitlines()[0])["content_sha8"]
    assert cli.main(["stores", "status"]) == 0
    text = capsys.readouterr().out
    assert "relu_paired_run_state | grading_host_copy | " + sha8 in text and f"{sha8}:published" in text
    assert cli.main(["stores", "status", "--json", "--probe"]) == 0
    assert json.loads(capsys.readouterr().out)["components"]["relu_paired_run_state"]["probe"]["current"] == sha8
    assert cli.main(["stores", "revalidate", "--component", "relu_paired_run_state", "--generation", sha8]) == 0
    assert "ok" in capsys.readouterr().out
    assert cli.main(["stores", "revalidate", "--component", "relu_paired_run_state", "--generation", "deadbeef", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["refusal"] == "no_manifest"
    assert cli.main(["stores", "check"]) == 0
    assert "no cluster_a_origin component" in capsys.readouterr().out
    assert cli.main(["stores", "mirror", "--component", "nope"]) == 2
    assert "not declared" in capsys.readouterr().err
    # `--root` is the registry root everywhere (the global flag); the stores subcommands never reuse the word for a component
    assert cli.main(["--root", w["root"], "stores", "status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["components"]["relu_paired_run_state"]["published"] == sha8
    with pytest.raises(SystemExit):
        cli.main(["stores", "mirror", "--root", "relu_paired_run_state"])
    assert "unrecognized arguments: --root" in capsys.readouterr().err
    assert cli.main(["stores", "mirror", "--dest", "cluster_b"]) == 2
    assert "destination 'cluster_b' is not declared" in capsys.readouterr().err


def test_load_config_refuses_malformed_declarations(tmp_path):
    base = _canon(tmp_path)
    with pytest.raises(ST.StoresError, match="authority 'cluster' is not one of"):
        ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "cluster"}}, name="a.yaml"))
    with pytest.raises(ST.StoresError, match="names no `original`"):
        ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "cluster_a_origin"}}, name="b.yaml"))
    with pytest.raises(ST.StoresError, match="must not name an `original`"):
        ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "grading_host_copy", "original": "x"}}, name="c.yaml"))
    with pytest.raises(ST.StoresError, match="not declared"):
        ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"shards": 2, "components": ["zz"]}}, name="d.yaml"))
    with pytest.raises(ST.StoresError, match="at least 1"):
        ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"shards": 0, "components": ["c"]}}, name="e.yaml"))
    with pytest.raises(ST.StoresError, match="lacks `base`"):
        ST.load_config(_cfg(tmp_path, {"c": {"authority": "grading_host_copy"}}, name="f.yaml"))
    cfg = ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "grading_host_copy"}}, name="g.yaml"))
    with pytest.raises(ST.StoresError, match="is not declared"):
        cfg.destination("nope")
    with pytest.raises(ST.StoresError, match="not declared"):
        cfg.select(["zz"])


def test_load_config_v2_rules_refuse_by_name_and_a_v1_document_still_loads(tmp_path):
    """The `cluster_a` authority with its producer / mirror / carry keys, the per-adapter cluster refusal, the volatile roots and the
    unknown-key refusals; a `cluster_reads/1` document loads unchanged."""
    base = _canon(tmp_path)
    ok_adapters = {"a": {"shards": 1, "components": ["c"]}}

    def refuses(components, adapters=None, match="", name="x.yaml", schema=None):
        with pytest.raises(ST.StoresError, match=match):
            ST.load_config(_cfg(tmp_path, components, adapters=adapters or ok_adapters, name=name, schema=schema))

    refuses({"c": {"base": base, "authority": "cluster_a"}}, match="names no `producer`", name="b1.yaml")
    refuses({"c": {"base": base, "authority": "cluster_a", "producer": "laptop"}}, match="not grading_host:<session>, cluster or frozen", name="b2.yaml")
    refuses({"c": {"base": base, "authority": "cluster_a", "producer": "grading_host:Pipeline", "transitional_carry": True}}, match="no `carry_deadline`", name="b3.yaml")
    refuses({"c": {"base": base, "authority": "cluster_a", "producer": "cluster", "transitional_carry": True, "carry_deadline": "2026-11-01"}},
            match="a carry is for a grading host:<session> producer", name="b4.yaml")
    refuses({"c": {"base": base, "authority": "cluster_a", "producer": "grading_host:P", "transitional_carry": True, "carry_deadline": "next week"}},
            match="not YYYY-MM-DD", name="b5.yaml")
    refuses({"c": {"base": base, "authority": "cluster_a", "producer": "grading_host:P", "carry_deadline": "2026-11-01"}}, match="without transitional_carry", name="b6.yaml")
    refuses({"c": {"base": base, "authority": "cluster_a", "producer": "grading_host:P", "transitional_carry": "yes", "carry_deadline": "2026-11-01"}},
            match="not a boolean", name="b7.yaml")
    refuses({"c": {"base": base, "authority": "cluster_a", "producer": "frozen", "min_publish_interval_s": -5}}, match="not a non-negative integer", name="b8.yaml")
    refuses({"c": {"base": base, "authority": "cluster_a", "producer": "frozen", "original": "exp/x"}}, match="must not name an `original`", name="b9.yaml")
    # the cluster_a-only keys on another authority, and unknown keys anywhere
    refuses({"c": {"base": base, "authority": "grading_host_copy", "transitional_carry": True, "carry_deadline": "2026-11-01"}}, match="belong to authority cluster_a", name="d1.yaml")
    refuses({"c": {"base": base, "authority": "grading_host_copy", "grading_host_mirror": base}}, match="belong to authority cluster_a", name="d2.yaml")
    refuses({"c": {"base": base, "authority": "grading_host_copy", "basee": base}}, match="unknown keys \\['basee'\\]", name="d3.yaml")
    refuses({"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"shards": 1, "components": ["c"], "shard": 2}}, match="unknown keys \\['shard'\\]", name="d4.yaml")
    # the per-adapter cluster refusal
    refuses({"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"cluster": "maybe", "reason": "r"}}, match="is not 'refused'", name="r1.yaml")
    refuses({"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"cluster": "refused"}}, match="gives no `reason`", name="r2.yaml")
    refuses({"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"cluster": "refused", "reason": "r", "shards": 1}}, match="reads on no cluster", name="r3.yaml")
    refuses({"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"shards": 1, "components": ["c"], "reason": "r"}}, match="without cluster: refused", name="r4.yaml")
    refuses({"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"shards": 1}}, match="names no component", name="r5.yaml")
    # the volatile roots: outside the home only, with a reason
    refuses({"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"shards": 1, "components": ["c"], "volatile_roots": [{"root": "~/x", "reason": "r"}]}},
            match="lies under the home", name="v1.yaml")
    refuses({"c": {"base": base, "authority": "grading_host_copy"}}, adapters={"a": {"shards": 1, "components": ["c"], "volatile_roots": [{"root": "/tmp/x"}]}},
            match="volatile_roots entry", name="v2.yaml")
    # a well-formed v2 document with every new key
    cfg = ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "cluster_a", "producer": "grading_host:Pipeline", "transitional_carry": True,
                                                "carry_deadline": "2026-11-01", "min_publish_interval_s": 600, "origin": "grading_host", "note": "n"},
                                          "d": {"base": base, "authority": "cluster_a", "producer": "cluster", "grading_host_mirror": str(tmp_path / "mirror" / "d")},
                                          "e": {"base": base, "authority": "grading_host_copy", "producer": "frozen"}},
                              adapters={"a": {"shards": 2, "components": ["c", "d"], "volatile_roots": [{"root": "/tmp/peek", "reason": "a peek"}], "note": "sized"},
                                        "r": {"cluster": "refused", "reason": "its pinned worktree is absent (D13)"},
                                        "e": {"shards": 1, "components": ["e"]}}, name="ok.yaml"))
    c, d, e = cfg.components["c"], cfg.components["d"], cfg.components["e"]
    assert c.authority == "cluster_a" and c.producer == "grading_host:Pipeline" and c.transitional_carry and c.carry_deadline == "2026-11-01"
    assert c.min_publish_interval_s == 600 and c.grading_host_mirror == c.base                # the mirror defaults to the base
    assert d.grading_host_mirror == str(tmp_path / "mirror" / "d") and not d.transitional_carry and d.carry_deadline is None
    assert e.producer == "frozen" and e.grading_host_mirror is None
    a, r = cfg.adapters["a"], cfg.adapters["r"]
    assert a.shards == 2 and a.components == ["c", "d"] and a.note == "sized" and not a.refused
    assert [(v.root, v.reason) for v in a.volatile_roots] == [("/tmp/peek", "a peek")]
    assert r.refused and r.reason.startswith("its pinned worktree") and r.shards == 0 and r.components == [] and r.volatile_roots == []
    assert c.adapters == ["a"] and e.adapters == ["e"]
    # a cluster_a component is skipped by the carry (nothing is copied from the grading host) and is no `stores check` candidate
    res = ST.mirror_component(str(tmp_path / "root"), cfg.destination("cluster_a"), c, dry_run=True)
    assert res.outcome == "skipped" and "stores push" in res.detail["why"]
    # a v1 document loads unchanged
    v1 = ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "grading_host_copy"}}, name="v1.yaml", schema="cluster_reads/1"))
    assert v1.components["c"].authority == "grading_host_copy" and v1.adapters["relu_paired"].shards == 8
    with pytest.raises(ST.StoresError, match="is not one of"):
        ST.load_config(_cfg(tmp_path, {"c": {"base": base, "authority": "grading_host_copy"}}, name="v9.yaml", schema="cluster_reads/9"))


def test_live_config_declares_the_six_adapters_with_valid_components(tmp_path):
    reg = tmp_path / "reg"
    reg.mkdir()
    (reg / "config.toml").write_text('[repo]\ncheckout = "/checkouts/clone"\n\n[[clusters]]\nname = "cluster_a"\nhost = "login-a"\nshared = "/shared/{user}"\n')
    cfg = ST.load_config(YAML_LIVE, root=str(reg))
    six = {"relu_paired": 8, "multiturn_grades": 8, "bfcl_full": 6, "bigcodebench_full": 6, "em_misalign": 6, "sampled_static": 1}
    assert {a: e.shards for a, e in cfg.adapters.items() if a in six} == six          # the Phase-3 six keep their measured shards
    assert len(cfg.adapters) > len(six)                                                # C2: every registered adapter is declared (the coverage test pins the set)
    # the destination is a cluster's registry label and the file spells no login host, shared root or repo mirror: they come from the
    # registry configuration's [[clusters]] entry of that name (the clone named after the [repo] checkout's directory)
    dest = cfg.destination("cluster_a")
    assert dest.host == "login-a" and dest.shared.startswith("/shared/") and "{user}" not in dest.shared
    assert dest.qos == "high" and dest.verify_cpus == 16
    assert dest.sbatch_path == f"{dest.shared}/repos/clone/scripts/registry/landing/store_verify.sbatch"
    for slug, c in cfg.components.items():
        assert c.authority in ST.AUTHORITIES, slug
        assert c.home_rel is not None and not c.home_rel.startswith(".."), (slug, c.base)   # every root hangs off the home
        assert c.adapters, f"{slug} is read by no declared adapter"
        assert (c.original is not None) == (c.authority == "cluster_a_origin"), slug
        assert c.note
    for a in cfg.adapters.values():
        assert a.refused or a.components, a.name
    # the components the adapters' module constants name are all present (the HOME-anchored roots of the six)
    for slug in ("relu_paired_run_state", "relu_paired_manifests", "petri_native_ledger", "petri_drive", "relu_paired_sidecars", "sampled",
                 "multiturn_run_state", "object_bank_full22", "petri_manifests", "misalign_panels", "misalign_run_state", "misalign_grades",
                 "bigcodebench_full", "bfcl_full"):
        assert slug in cfg.components, slug
    assert cfg.components["relu_paired_manifests"].adapters == ["relu_paired", "em_misalign"]
    # the four members, then the three top-level records added later (the unsuffixed transfer_stats_full22.json the glob misses and the
    # two half-dose records): r2_export sources that lay under no member
    assert cfg.components["multiturn_run_state"].members == ["grades_*", "transcripts_*", "transfer_stats_full22_*.json", "transfer_stats_nemotron_complete.json",
                                                             "transfer_stats_full22.json", "attacker_effort_halfdose.json", "determinism_floor_halfdose.json"]
    # relu_paired reads its controls index (`sampled/report/analysis/final_rung.json`, relu_paired.CONTROLS_INDEX_REL) from the
    # `sampled` component, so that component is one of its dependencies
    assert "sampled" in cfg.adapters["relu_paired"].components
    assert cfg.components["sampled"].adapters[:2] == ["relu_paired", "sampled_static"] and set(cfg.components["sampled"].adapters) >= {"sampled_controls", "degeneracy_union"}
    # the ten `wandb/run-*/logs/debug-core.log` links into ~/.cache/wandb escape the base (2026-09-28); the adapter never opens wandb/
    assert "/wandb" in cfg.components["relu_paired_run_state"].exclude


def test_verify_time_is_twice_the_estimate_and_bounded():
    assert ST.verify_time_s(0, 0, 100.0) == 600                                  # the floor
    assert ST.verify_time_s(112_800 * 10**6, 229_000, 100.0) == 2 * (1128 + 458 + 60)
    assert ST.verify_time_s(10**15, 10**6, 100.0, max_s=14400) == 14400        # the ceiling
    assert ST.slurm_time(3600 + 61) == "01:01:01" and ST.slurm_time(600) == "00:10:00"


def test_cluster_entry_points_run_under_python_S(tmp_path):
    """What store_verify.sbatch runs: `python3 -S -m sisteer.registry.stores verify|manifest` -- stdlib only."""
    base = _canon(tmp_path)
    env = {"PYTHONPATH": SRC, "PATH": os.environ["PATH"]}
    prefix = str(tmp_path / "out" / "MANIFEST-x")
    r = subprocess.run(["python3", "-S", "-m", "sisteer.registry.stores", "manifest", base, "--out-prefix", prefix, "--workers", "2",
                        "--exclude", "__pycache__", "--exclude", "/.pytest_cache", "--job-id", "42"], capture_output=True, text=True, env=env, cwd="/")
    assert r.returncode == 0, r.stderr
    meta = json.load(open(prefix + ".json"))
    assert meta["n_files"] == 4 and meta["job_id"] == "42" and json.loads(r.stdout)["n_files"] == 4
    assert meta["root"] is None and meta["home_rel"] is None and meta["in_place_original"] == base, "no --grading-host-root: no spelling is invented from the node's home"
    gen = tmp_path / "gen"
    shutil.copytree(base, str(gen), symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    out = str(tmp_path / "VERIFIED-x.json")
    r = subprocess.run(["python3", "-S", "-m", "sisteer.registry.stores", "verify", str(gen), prefix + ".sha256", "--out", out, "--workers", "2"],
                       capture_output=True, text=True, env=env, cwd="/")
    assert r.returncode == 0, r.stderr
    assert json.load(open(out))["ok"] is True
    _write(str(gen / "unlisted.txt"), b"x")
    os.remove(str(gen / "top.txt"))
    r = subprocess.run(["python3", "-S", "-m", "sisteer.registry.stores", "verify", str(gen), prefix + ".sha256", "--out", out, "--workers", "2"],
                       capture_output=True, text=True, env=env, cwd="/")
    rep = json.load(open(out))
    assert r.returncode == 1 and rep["ok"] is False and rep["missing"] == ["top.txt"] and rep["extra"] == ["unlisted.txt"]


def test_store_verify_sbatch_keeps_the_cluster_rules():
    text = open(SBATCH, encoding="utf-8").read()
    assert subprocess.run(["bash", "-n", SBATCH], capture_output=True, text=True).returncode == 0
    directives = [l for l in text.splitlines() if l.startswith("#SBATCH")]
    assert "#SBATCH --qos=high" in directives and "#SBATCH --partition=general" in directives and "#SBATCH --open-mode=append" in directives
    # the log path is the submitter's: a directive cannot read the site's shared root, so the script names none and the store mirror's
    # submission (Remote.submit) names one under the destination's shared root, its directory created first
    assert not any(d.startswith(("#SBATCH --output", "#SBATCH --error", "#SBATCH -o", "#SBATCH -e")) for d in directives) and "#SBATCH --cpus-per-task=16" in directives
    sent = []

    def run(argv, timeout_s, input=None, check=False):
        sent.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout="4242\n", stderr="")

    shared = "/srv/cluster_b/tester"                                  # a shared root no fixed log path could name
    job, _tlim = ST.Remote(ST.Destination(name="cluster_b", host=None, shared=shared), run).submit(
        "store-verify-x", ["verify", "g", "m", "o", "2"], 10, 1, 100.0, script=SBATCH)
    sb = next(a for a in sent if a[0] == "sbatch")
    log = f"--output={shared}/exp/logs/%x_%j.out"
    assert job == "4242" and log in sb and sb.index(log) < sb.index(SBATCH), sb
    assert any(a[:2] == ["sh", "-c"] and a[2] == f"mkdir -p {shared}/exp/logs" for a in sent[:sent.index(sb)]), sent
    assert any(d.startswith("#SBATCH --time=") for d in directives) and any(d.startswith("#SBATCH --mem=") for d in directives)
    effective = "\n".join(l for l in text.splitlines() if l.startswith("#SBATCH") or not l.lstrip().startswith("#"))
    assert "--export" not in effective and "--mem=0" not in text and "--gres" not in effective and "--delete" not in text
    assert not any(d.startswith("#SBATCH --export") or d.startswith("#SBATCH --gres") for d in directives)
    assert "scripts/hf_env.sh" in text and "trap '' SIGTERM" in text and "python3 -S -m sisteer.registry.stores" in text and "srun " in text
    assert "--job-name" in text and "manifest-only" in text and "verify" in text
    assert "  push)" in text and "sisteer.registry.stores push" in text, "the producer's in-job publication mode (C3)"
    import re
    assert not re.search(r"(?i)(TOKEN|API_KEY|SECRET|PASSWORD)\s*=", text) and "SISTEER_SECRET_CMD" not in text and "://" not in text
    # the python side never passes --delete or --export either: every string constant the code could hand a process
    # (docstrings, which describe the rule, excluded)
    import ast
    tree = ast.parse(open(ST.__file__, encoding="utf-8").read())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    assert literals and not any("--delete" in v or "--export" in v for v in literals)


# --------------------------------------------------------------------------------------------
# the adversarial review's findings (2026-09-28): the seal and the destination guard (P1-1), symlinks attested by target
# (P1-2), forced copy of same-size same-mtime edits (P2-3), relu_paired's controls index under `sampled` (P2-4)
# --------------------------------------------------------------------------------------------


def _rewrite_preserving_stat(path, data):
    """Write ``data`` over a (possibly sealed) file keeping its size, mtime and mode: the edit only a hash can see."""
    st = os.stat(path)
    os.chmod(path, 0o644)
    with open(path, "wb") as fh:
        fh.write(data)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.chmod(path, stat_module.S_IMODE(st.st_mode))


def test_current_without_a_matching_report_is_never_unchanged(world):
    """P1-1 (the fast path): `current -> <sha8>` with a failed or absent VERIFIED report is reported, not called unchanged."""
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r1.outcome == "published"
    gd, sha8 = w["gen_dir"], r1.content_sha8
    rep = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    rep["ok"], rep["corrupt"] = False, ["a/f2.jsonl"]
    json.dump(rep, open(f"{gd}/VERIFIED-{sha8}.json", "w"))
    rn = Runner()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "current_unverified" and not r2.ok and r2.published == sha8 and r2.detail["verified_present"] is True, r2.as_dict()
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn.calls), "nothing is copied or re-verified behind a published pointer"
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["generation"]["state"] == "current_unverified" and st["published"] == sha8 and st["last_run"]["outcome"] == "current_unverified"
    os.remove(f"{gd}/VERIFIED-{sha8}.json")
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r3.outcome == "current_unverified" and r3.detail["verified_present"] is False
    assert not ST.admit_generation(gd, sha8).ok and ST.admit_generation(gd, sha8).refusal == "no_verified_report"


def test_verified_generation_is_sealed_and_a_post_verify_edit_refuses_admission(world):
    """P1-1: the verify seals every regular file (chmod a-w) and records DEST-STAT-<sha8>.json (size, mtime_ns, ino, mode);
    `admit_generation` -- what the cluster-read task runs beside `revalidate` -- refuses any later change by name, including
    an edit that preserves size, mtime AND mode (the re-hash of moved files plus the seeded sample, which covers every file
    at this size, the way `revalidate` does on the grading host)."""
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    gen = f"{gd}/{sha8}"
    files = ("a/b/f1.json", "a/f2.jsonl", "top.txt", "x/deep/er/f3.bin")
    for rel in files:
        assert not (os.stat(f"{gen}/{rel}").st_mode & 0o222), f"{rel} is not sealed"
    rep = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert rep["sealed"] == 4 and rep["n_symlinks"] == 1 and rep["dest_stat"] == f"DEST-STAT-{sha8}.json"
    snap = json.load(open(f"{gd}/DEST-STAT-{sha8}.json"))
    assert set(snap["files"]) == set(files) | {"link_to_top"} and snap["fields"] == ["size", "mtime_ns", "ino", "mode", "ctime_ns_if_one_link"]
    assert all(v[4] > 0 for rel, v in snap["files"].items() if rel != "link_to_top"), "first generation: every file has one link, so ctime is in its key"
    assert snap["files"]["link_to_top"][4] == 0
    ad = ST.admit_generation(gd, sha8)
    assert ad.ok and ad.n_files == 4 and ad.n_symlinks == 1 and ad.moved == 0 and ad.sampled == 4 and ad.hashed == 4, ad.as_dict()
    # the seal: a writer that does not un-seal cannot touch the bytes -- the kernel's refusal of the file's mode (chmod a-w), which the same
    # privilege that bypasses a directory's mode bypasses (root / CAP_DAC_OVERRIDE); the demonstration runs only where the probe reads the
    # refusal, and every admission check below (stat, ctime, hash) holds regardless
    if permission_denial_enforced():
        with pytest.raises(PermissionError):
            open(f"{gen}/a/f2.jsonl", "ab").close()
    # an edit preserving size, mtime and mode -> content_changed, by name (the CLI entry point the reader calls agrees); the
    # file has one link, so its ctime is in the key: the stat walk finds it (moved == 1), not merely the seeded sample
    p = f"{gen}/a/f2.jsonl"
    _rewrite_preserving_stat(p, b"row1\nROW2\n")
    ad = ST.admit_generation(gd, sha8)
    assert not ad.ok and ad.refusal == "content_changed" and ad.mismatched == ["a/f2.jsonl"] and ad.moved == 1, ad.as_dict()
    r = subprocess.run(["python3", "-S", "-m", "sisteer.registry.stores", "admit", gd, sha8, "--json"], capture_output=True, text=True,
                       env={"PYTHONPATH": SRC, "PATH": os.environ["PATH"]}, cwd="/")
    assert r.returncode == 1 and json.loads(r.stdout)["refusal"] == "content_changed", r.stderr
    _rewrite_preserving_stat(p, b"row1\nrow2\n")
    assert ST.admit_generation(gd, sha8).ok, "the bytes restored: admitted again"
    # a touch (mtime moved, bytes intact) is re-hashed and admitted; an mtime-moving edit is refused by name
    os.utime(p)
    ad = ST.admit_generation(gd, sha8)
    assert ad.ok and ad.moved == 1 and ad.hashed == 4, ad.as_dict()
    os.chmod(p, 0o644)
    with open(p, "wb") as fh:
        fh.write(b"row1\nrow3\n")
    ad = ST.admit_generation(gd, sha8)
    assert not ad.ok and ad.refusal == "content_changed" and ad.mismatched == ["a/f2.jsonl"] and ad.moved == 1
    _rewrite_preserving_stat(p, b"row1\nrow2\n")
    # the directory seal (C3) stops the owner from adding or removing an entry at all; since the final pass on 2c the admission names a
    # LIFTED top-directory mode itself (`unsealed_directories`), the deeper modes staying the attestation's check (C5) -- so the owner who
    # un-seals to stage a change seals again, and the admission's file-level guard below still names every change
    os.chmod(gen, 0o755)
    assert ST.admit_generation(gd, sha8).refusal == "unsealed_directories", "the lifted seal is named before the walk"
    # an unverified file present, a verified file gone -> generation_changed, listed
    _write(f"{gen}/planted.txt", b"x")
    os.chmod(gen, ST.DIR_SEAL_MODE)
    ad = ST.admit_generation(gd, sha8)
    assert ad.refusal == "generation_changed" and ad.extra == ["planted.txt"]
    os.chmod(gen, 0o755)
    os.remove(f"{gen}/planted.txt")
    os.remove(f"{gen}/top.txt")
    os.chmod(gen, ST.DIR_SEAL_MODE)
    ad = ST.admit_generation(gd, sha8)
    assert ad.refusal == "generation_changed" and ad.missing == ["top.txt"]
    os.chmod(gen, 0o755)
    # the snapshot itself gone -> no_dest_stat (never a hash-only guess); the manifest gone -> no_manifest
    os.remove(f"{gd}/DEST-STAT-{sha8}.json")
    assert ST.admit_generation(gd, sha8).refusal == "no_dest_stat"
    os.remove(f"{gd}/MANIFEST-{sha8}.sha256")
    assert ST.admit_generation(gd, sha8).refusal == "no_manifest"
    # a report that names a different manifest -> verified_mismatch
    _write(f"{gd}/MANIFEST-deadbeef.sha256", b"")
    _write(f"{gd}/VERIFIED-deadbeef.json", json.dumps({**rep, "content_sha256": "0" * 64}).encode())
    assert ST.admit_generation(gd, "deadbeef").refusal == "verified_mismatch"


def test_symlinks_are_attested_by_target_and_an_escaping_link_is_refused(world):
    """P1-2: a symlink is part of the identity (path + literal target) on both sides; a retarget is a new generation on the
    grading host and a refusal on the destination; a target that escapes the base is refused by name and never copied."""
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    assert os.readlink(f"{gd}/{sha8}/link_to_top") == "top.txt"
    # link_escapes: the rule
    assert ST.link_escapes("l", "/etc/hostname") and ST.link_escapes("a/b/l", "../../../x") and ST.link_escapes("l", "..") and ST.link_escapes("l", "../x")
    assert not ST.link_escapes("a/b/l", "../c") and not ST.link_escapes("l", "..hidden") and not ST.link_escapes("a/l", "../a/./b")
    # a link-only retarget on the canonical copy: revalidate refuses by name; the next mirror is a NEW generation with no file re-read
    link = os.path.join(w["base"], "link_to_top")
    os.remove(link)
    os.symlink("a/f2.jsonl", link)
    rv = ST.revalidate("relu_paired_run_state", sha8, root=w["root"], cfg=w["cfg"])
    assert not rv.ok and rv.refusal == "content_changed" and rv.mismatched == ["link_to_top"] and rv.moved == 1 and rv.moved_unchanged == 0, rv.as_dict()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r2.outcome == "published" and r2.content_sha8 != sha8 and r2.hashed == 0 and r2.from_cache == 4, r2.as_dict()
    assert os.readlink(f"{gd}/{r2.content_sha8}/link_to_top") == "a/f2.jsonl" and os.readlink(f"{gd}/current") == r2.content_sha8
    assert json.load(open(f"{gd}/MANIFEST-{r2.content_sha8}.json"))["symlinks"] == {"link_to_top": "a/f2.jsonl"}
    assert ST.revalidate("relu_paired_run_state", r2.content_sha8, root=w["root"], cfg=w["cfg"]).ok
    # a link-only retarget on the DESTINATION: the guard refuses by name; so does the verify
    gen2 = f"{gd}/{r2.content_sha8}"
    # the owner un-seals the directory to stage each retarget and seals it again (the admission names a lifted top mode itself since the
    # final pass on 2c; every directory's mode stays C5's attestation to check): the file-level guards below still name every change
    os.chmod(gen2, 0o755)
    os.remove(f"{gen2}/link_to_top")
    os.symlink("top.txt", f"{gen2}/link_to_top")
    os.chmod(gen2, ST.DIR_SEAL_MODE)
    ad = ST.admit_generation(gd, r2.content_sha8)
    assert not ad.ok and ad.refusal == "content_changed" and ad.mismatched == ["link_to_top"], ad.as_dict()
    rep, _snap = ST.verify_generation(gen2, f"{gd}/MANIFEST-{r2.content_sha8}.sha256", seal=False)
    assert rep["ok"] is False and rep["corrupt"] == ["link_to_top"]
    # an escaping link planted in the generation: refused (content_changed names it); an unlisted link: generation_changed
    os.chmod(gen2, 0o755)
    os.remove(f"{gen2}/link_to_top")
    os.symlink("/etc/hostname", f"{gen2}/link_to_top")
    os.chmod(gen2, ST.DIR_SEAL_MODE)
    assert ST.admit_generation(gd, r2.content_sha8).mismatched == ["link_to_top"]
    os.chmod(gen2, 0o755)
    os.remove(f"{gen2}/link_to_top")
    os.symlink("a/f2.jsonl", f"{gen2}/link_to_top")
    os.symlink("top.txt", f"{gen2}/planted_link")
    os.chmod(gen2, ST.DIR_SEAL_MODE)
    ad = ST.admit_generation(gd, r2.content_sha8)
    assert ad.refusal == "generation_changed" and ad.extra == ["planted_link"]
    os.chmod(gen2, 0o755)
    os.remove(f"{gen2}/planted_link")
    os.chmod(gen2, ST.DIR_SEAL_MODE)
    assert ST.admit_generation(gd, r2.content_sha8).ok
    # an external target on the canonical copy: the manifest refuses by name and nothing of it reaches the destination -- no copy, no
    # verify submission, nothing new under gen/<slug>/. The ONE destination step a refused manifest may take is the slug directory's
    # `mkdir -p` (idempotent; it already exists here): the publish lock lives in that directory and is taken BEFORE the snapshot the
    # manifest is, so that overlapping publishers publish in snapshot order
    gen_listing = sorted(os.listdir(gd))
    for target in ("/etc/hostname", "../outside", "a/../../outside"):
        os.remove(link)
        os.symlink(target, link)
        rn = Runner()
        r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
        assert r3.outcome == "refused" and f"link_to_top -> {target} escapes" in r3.detail["reason"], (target, r3.as_dict())
        destination_steps = [a for a in rn.calls if not (a[:2] == ["sh", "-c"] and a[2] == f"mkdir -p {shlex.quote(gd)}")]
        assert destination_steps == [] and sorted(os.listdir(gd)) == gen_listing, (target, rn.calls)
        assert not any("rsync" in a or a[0] == "sbatch" or a[:3] == ["bash", "-o", "pipefail"] for a in rn.calls)
        # the pinned generation attested an inside target; the link now differs from it -> the admission refuses by name
        rv = ST.revalidate("relu_paired_run_state", r2.content_sha8, root=w["root"], cfg=w["cfg"])
        assert rv.refusal == "content_changed" and rv.mismatched == ["link_to_top"], rv.as_dict()
    # a directory link inside the base is attested like any other (the object roots carry such links)
    os.remove(link)
    os.symlink("a/f2.jsonl", link)
    os.symlink("a/b", os.path.join(w["base"], "dirlink"))
    r4 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r4.outcome == "published" and json.load(open(f"{gd}/MANIFEST-{r4.content_sha8}.json"))["symlinks"] == {"dirlink": "a/b", "link_to_top": "a/f2.jsonl"}
    assert os.path.islink(f"{gd}/{r4.content_sha8}/dirlink") and sorted(ST._walk_plain_dir(f"{gd}/{r4.content_sha8}")) == ["a/b/f1.json", "a/f2.jsonl", "top.txt", "x/deep/er/f3.bin"]
    assert ST.admit_generation(gd, r4.content_sha8).ok


def test_same_size_mtime_preserving_edit_is_copied_not_hard_linked(world):
    """P2-3: with --link-dest, rsync's size/mtime quick check would hard-link a same-size, mtime-preserving edit to the OLD
    bytes and every retry would fail its verify the same way; the files whose manifest sha differs from the published
    generation's are forced (--files-from + --ignore-times), so the new generation carries the new bytes and the old one is
    untouched."""
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r1.outcome == "published"
    p = os.path.join(w["base"], "a", "f2.jsonl")
    st = os.stat(p)
    with open(p, "wb") as fh:
        fh.write(b"row1\nROW2\n")                                    # same size
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))                # same mtime: the quick check calls it unchanged
    rn = Runner()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "published" and r2.content_sha8 != r1.content_sha8 and r2.hashed == 1, r2.as_dict()
    gd = w["gen_dir"]
    g1, g2 = f"{gd}/{r1.content_sha8}", f"{gd}/{r2.content_sha8}"
    assert open(f"{g2}/a/f2.jsonl", "rb").read() == b"row1\nROW2\n" and open(f"{g1}/a/f2.jsonl", "rb").read() == b"row1\nrow2\n"
    assert os.stat(f"{g1}/a/f2.jsonl").st_ino != os.stat(f"{g2}/a/f2.jsonl").st_ino
    for rel in ("a/b/f1.json", "top.txt", "x/deep/er/f3.bin"):
        assert os.stat(f"{g1}/{rel}").st_ino == os.stat(f"{g2}/{rel}").st_ino, f"{rel}: unchanged files still hard-link"
    passes = [a for a in rn.calls if "rsync" in a]
    assert len(passes) == 2 and any(a.startswith("--link-dest=") for a in passes[0])
    assert "--files-from=-" in passes[1] and "--ignore-times" in passes[1] and not any(a.startswith("--link-dest") for a in passes[1]) and "--delete" not in " ".join(passes[1])
    assert r2.detail["rsync"]["forced"]["n"] == 1
    assert os.readlink(f"{gd}/current") == r2.content_sha8
    # both generations admit: the old one's inodes gained links (nlink, ctime moved) but the seal's key does not read those
    assert ST.admit_generation(gd, r1.content_sha8).ok and ST.admit_generation(gd, r2.content_sha8).ok
    # no published generation's manifest on the grading host (state wiped) -> the destination's copy is read; neither -> every file forced
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r3.outcome == "unchanged"
    with open(p, "wb") as fh:
        fh.write(b"row1\nRoW2\n")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.remove(os.path.join(ST.component_state_dir(w["root"], "relu_paired_run_state"), f"MANIFEST-{r2.content_sha8}.sha256"))
    rn = Runner()
    r4 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r4.outcome == "published" and r4.detail["rsync"]["forced"]["n"] == 1 and open(f"{gd}/{r4.content_sha8}/a/f2.jsonl", "rb").read() == b"row1\nRoW2\n"
    os.remove(os.path.join(ST.component_state_dir(w["root"], "relu_paired_run_state"), f"MANIFEST-{r4.content_sha8}.sha256"))
    os.remove(f"{gd}/MANIFEST-{r4.content_sha8}.sha256")
    with open(p, "wb") as fh:
        fh.write(b"row1\nroW2\n")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    r5 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r5.outcome == "published" and r5.detail["rsync"]["forced"]["n"] == 4, "no manifest of the published generation anywhere: every file is forced"


def test_relu_paired_depends_on_sampled_and_a_stale_controls_index_refuses_its_admission(tmp_path):
    """P2-4: relu_paired's controls index lives under the `sampled` component; the adapter is admitted to the cluster route
    only when EVERY component it declares re-validates, so a rewritten or absent index refuses it by the component's name."""
    rp = _canon(tmp_path / "rp")
    sampled = str(tmp_path / "sampled")
    index = os.path.join(sampled, "report", "analysis", "final_rung.json")
    _write(index, b'{"targets": {}}\n')
    _write(os.path.join(sampled, "stores", "s1.jsonl"), b"x\n")
    root = str(tmp_path / "reg")
    cfg = ST.load_config(_cfg(tmp_path, {"relu_paired_run_state": {"base": rp, "authority": "grading_host_copy", "exclude": ["__pycache__", "/.pytest_cache"]},
                                        "sampled": {"base": sampled, "authority": "grading_host_copy"}},
                             adapters={"relu_paired": {"shards": 8, "components": ["relu_paired_run_state", "sampled"]},
                                       "sampled_static": {"shards": 1, "components": ["sampled"]}}))
    assert cfg.components["sampled"].adapters == ["relu_paired", "sampled_static"]
    dest = cfg.destination("cluster_a")
    pins = {c.slug: ST.mirror_component(root, dest, c, runner=Runner()).content_sha8 for c in cfg.components.values()}
    adm = ST.adapter_admission(cfg, "relu_paired", pins, root=root)
    assert set(adm) == {"relu_paired_run_state", "sampled"} and all(rv.ok for rv in adm.values()), {k: v.as_dict() for k, v in adm.items()}
    # the index rewritten by a grading batch -> `sampled` refuses -> relu_paired is not admitted, naming the component and the file
    _write(index, b'{"targets": {"qwen35_4b": {"controls": {}}}}\n')
    adm = ST.adapter_admission(cfg, "relu_paired", pins, root=root)
    assert adm["relu_paired_run_state"].ok and not adm["sampled"].ok
    assert adm["sampled"].refusal == "content_changed" and adm["sampled"].mismatched == ["report/analysis/final_rung.json"]
    assert not all(rv.ok for rv in adm.values())
    # absent -> file_set_changed; the index is what is missing
    os.remove(index)
    adm = ST.adapter_admission(cfg, "relu_paired", pins, root=root)
    assert adm["sampled"].refusal == "file_set_changed" and adm["sampled"].missing == ["report/analysis/final_rung.json"]
    # a component the window never pinned refuses too; an undeclared adapter is an error by name
    assert ST.adapter_admission(cfg, "relu_paired", {"relu_paired_run_state": pins["relu_paired_run_state"]}, root=root)["sampled"].refusal == "no_manifest"
    with pytest.raises(ST.StoresError, match="not declared"):
        ST.adapter_admission(cfg, "nope", pins, root=root)


# --------------------------------------------------------------------------------------------
# a transport blip is never "no current"; atomic unique-temp writes and one writer of the state file; both hashes checked;
# `..cache` is a legitimate member; the cluster manifest's spelling
# --------------------------------------------------------------------------------------------


class _Blip(Runner):
    """A runner whose `readlink` shell step fails like an ssh outage (rc 255); everything else runs as usual."""

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[:2] == ["sh", "-c"] and "readlink" in argv[2]:
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: connect to host login-0 port 22: Connection timed out")
        return super().__call__(argv, timeout_s, input=input, check=check)


def test_a_transport_failure_reading_current_is_an_error_never_a_recopy(world, monkeypatch, capsys):
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r1.outcome == "published"
    gd, sha8 = w["gen_dir"], r1.content_sha8
    rn = _Blip()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "error" and not r2.ok and "readlink" in r2.detail["error"] and "rc 255" in r2.detail["error"], r2.as_dict()
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn.calls), "a blip never becomes a full re-copy without hard links"
    assert os.readlink(f"{gd}/current") == sha8 and {d for d in os.listdir(gd) if not d.startswith((".", "MANIFEST", "VERIFIED", "DEST-STAT", "SEALED", "PUBLISHED"))} == {sha8, "current"}
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["last_run"]["outcome"] == "error" and "readlink" in st["last_run"]["error"] and st["published"] == sha8
    # `current` present but not a symlink: refused by name, nothing copied
    os.remove(f"{gd}/current")
    _write(f"{gd}/current", b"junk")
    rn = Runner()
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r3.outcome == "error" and "not a symlink" in r3.detail["error"] and not any("rsync" in a for a in rn.calls)
    os.remove(f"{gd}/current")
    # absent: None, and the matching report already there publishes without a copy
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner()).outcome == "published_existing_verified"
    # `status --probe` on a blip is loud (exit 2), never a quiet `current: null`
    monkeypatch.setattr(ST, "_default_runner", _Blip())
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "status", "--json", "--probe"]) == 2
    assert "readlink" in capsys.readouterr().err


def test_writes_use_unique_temp_names_with_fsync_and_check_takes_the_mirror_lock(tmp_path, monkeypatch, capsys):
    p = str(tmp_path / "d" / "x.json")
    ST._write_json(p, {"a": 1})
    assert json.load(open(p)) == {"a": 1} and os.listdir(tmp_path / "d") == ["x.json"], "no fixed <path>.tmp, no leftover"
    seen = []
    real_replace, real_fsync = os.replace, os.fsync
    monkeypatch.setattr(os, "replace", lambda src, dst: (seen.append(os.path.basename(src)), real_replace(src, dst))[1])
    synced = []
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    ST._write_text(p, "1")
    ST._write_text(p, "2")
    assert len(set(seen)) == 2 and all(n.startswith(f"x.json.{os.getpid()}.") and n.endswith(".tmp") for n in seen), seen
    assert len(synced) == 2 and open(p).read() == "2"
    monkeypatch.undo()
    # `stores check` writes state/store_mirror.json, so it takes the mirror's lock: the loser exits 3 without submitting anything
    mirror = _canon(tmp_path / "m")
    original = tmp_path / "shared" / "exp" / "orig"
    shutil.copytree(mirror, str(original), symlinks=True)
    root = str(tmp_path / "reg")
    cfgp = _cfg(tmp_path, {"anchors": {"base": mirror, "authority": "cluster_a_origin", "origin": "cluster_a", "original": "exp/orig",
                                       "exclude": ["__pycache__", "/.pytest_cache"]}},
                adapters={"relu_paired": {"shards": 8, "components": ["anchors"]}})
    rn = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, cfgp)
    with ST.MirrorLock(root) as held:
        assert held.held
        assert cli.main(["stores", "check"]) == 3
        assert "another run holds" in capsys.readouterr().err and rn.calls == [] and not os.path.exists(ST.state_path(root))
    assert cli.main(["stores", "check"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["state"] == "match" and any(a[0] == "sbatch" for a in rn.calls)
    # `status` only reads: it runs under a held lock
    with ST.MirrorLock(root):
        assert cli.main(["stores", "status", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["components"]["anchors"]["authority_state"]["state"] == "match"


def test_a_verified_report_must_carry_both_manifest_hashes(world):
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    rep = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    pre = ST.build_manifest(w["comp"], ST.component_state_dir(w["root"], "relu_paired_run_state"))
    assert ST._verified_matches(rep, pre)
    assert not ST._verified_matches({**rep, "name_size_sha256": "0" * 64}, pre), "the D269 pair is checked as a pair"
    assert not ST._verified_matches({**rep, "content_sha256": "0" * 64}, pre)
    json.dump({**rep, "name_size_sha256": "0" * 64}, open(f"{gd}/VERIFIED-{sha8}.json", "w"))
    assert ST.admit_generation(gd, sha8).refusal == "verified_mismatch"
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner()).outcome == "current_unverified"


def test_a_member_named_dot_dot_cache_is_walked_and_an_escaping_member_is_refused(tmp_path):
    base = tmp_path / "base"
    _write(str(base / "..cache" / "f.txt"), b"x")
    _write(str(base / "..top"), b"y")
    _write(str(tmp_path / "outside" / "g.txt"), b"z")
    walk = ST.walk_component(ST.Component("c", str(base), "grading_host_copy", members=["..cache", "..top"]))
    assert [rel for rel, _p, _s in walk.files] == ["..cache/f.txt", "..top"], "a name that starts with two dots is not an escape"
    assert ST.build_manifest(ST.Component("c", str(base), "grading_host_copy", members=["..cache", "..top"]), None).n_files == 2
    for member in ("../outside", "..cache/../../outside", "../base/../outside"):
        with pytest.raises(ST.ManifestRefusal, match="outside the base"):
            ST.walk_component(ST.Component("c", str(base), "grading_host_copy", members=[member]))
    assert [rel for rel, _p, _s in ST.walk_component(ST.Component("c", str(base), "grading_host_copy", members=["..cache/../..top"])).files] == ["..top"]


# --------------------------------------------------------------------------------------------
# the second reviewer's HIGH findings (2026-09-28): the tick pins the running code on the cluster before any copy; the
# guard is a precheck and the reader's per-input hash against manifest_index is the byte guarantee (ctime for one-link files)
# --------------------------------------------------------------------------------------------


def test_the_tick_pins_the_running_revision_on_the_cluster_before_any_copy(world, monkeypatch, capsys):
    """(1) step 0: push the running revision to the repo mirror, check the verifier exists AT that revision, extract
    `git archive <sha>` into trees/<sha>/ and submit THAT sbatch; a revision without the verifier, or a dirty checkout,
    refuses the whole tick by name before a single rsync."""
    w = world
    dest = w["dest"]
    src, mirror = dest.source_repo, dest.mirror_dir
    sha = _git(src, "rev-parse", "HEAD")
    rn = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn)
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "mirror"]) == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["outcome"] == "published" and lines[0]["revision"] == sha
    assert lines[-1]["revision"] == sha and lines[-1]["pushed_via"] == f"git push gate/{sha[:7]}" and lines[-1]["verifier"] == dest.pinned_sbatch(sha)
    # the revision landed on the mirror under gate/<sha7>; the tree holds the archive and its REVISION; the sbatch submitted is the pinned copy
    assert _git(mirror, "rev-parse", f"gate/{sha[:7]}") == sha
    tree = dest.tree_dir(sha)
    assert open(f"{tree}/REVISION").read().strip() == sha and os.path.isfile(f"{tree}/scripts/registry/landing/store_verify.sbatch")
    assert os.path.isfile(f"{tree}/src/sisteer/registry/stores.py") and os.path.isfile(f"{tree}/src/sisteer/__init__.py")
    sb = next(a for a in rn.calls if a[0] == "sbatch")
    assert dest.pinned_sbatch(sha) in sb and dest.sbatch_path not in sb
    # the pin (push, cat-file, archive) all happened BEFORE the first rsync
    first_rsync = next(i for i, a in enumerate(rn.calls) if "rsync" in a)
    pin_steps = [i for i, a in enumerate(rn.calls) if any(("cat-file" in x or "archive" in x or "refs/heads/gate/" in x) for x in a)]
    assert pin_steps and max(pin_steps) < first_rsync, "nothing is copied until the code is pinned"
    assert ST.load_state(w["root"])["components"]["relu_paired_run_state"]["revision"] == sha
    # a second tick reuses the tree (REVISION untouched, no re-extract) and is `unchanged`
    before = os.stat(f"{tree}/REVISION").st_mtime_ns
    assert cli.main(["stores", "mirror"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[0])["outcome"] == "unchanged" and os.stat(f"{tree}/REVISION").st_mtime_ns == before
    # the direct API
    pin = ST.pin_cluster_tree(dest, ST.Remote(dest, Runner()))
    assert pin.revision == sha and pin.tree == tree and pin.sbatch_path == dest.pinned_sbatch(sha) and pin.pushed_via == "already on the mirror"
    # a revision whose tree LACKS the verifier: refused by name before any copy; nothing on the destination or in the state moves
    os.remove(os.path.join(src, "scripts", "registry", "landing", "store_verify.sbatch"))
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "verifier gone")
    sha2 = _git(src, "rev-parse", "HEAD")
    with open(os.path.join(w["base"], "top.txt"), "ab") as fh:
        fh.write(b"more\n")                                  # a copy WOULD be due
    state_before = open(ST.state_path(w["root"])).read()
    rn2 = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn2)
    assert cli.main(["stores", "mirror"]) == 2
    err = capsys.readouterr().err
    assert "verifier_missing_at_revision" in err and sha2[:7] in err and "store_verify.sbatch" in err
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn2.calls), "refused before any rsync"
    assert _git(mirror, "rev-parse", f"gate/{sha2[:7]}") == sha2, "the revision was pushed (that is how the mirror was asked)"
    assert not os.path.exists(dest.tree_dir(sha2)) and open(ST.state_path(w["root"])).read() == state_before
    assert {d for d in os.listdir(w["gen_dir"]) if not d.startswith((".", "MANIFEST", "VERIFIED", "DEST-STAT", "SEALED", "PUBLISHED"))} == {lines[0]["content_sha8"], "current"}
    # a dirty checkout: refused by name, nothing pushed
    _write(os.path.join(src, "src", "sisteer", "registry", "stores.py"), b"# edited, uncommitted\n")
    rn3 = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn3)
    assert cli.main(["stores", "mirror"]) == 2
    assert "dirty_checkout" in capsys.readouterr().err and not any("push" in a for a in rn3.calls)
    # a dry run pins nothing and reports the revision (`stores check` pins the same way -- it submits a job too)
    assert cli.main(["stores", "mirror", "--dry-run"]) == 0
    tail = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert tail["dry_run"] is True and tail["revision"] == sha2 and tail["dirty_tracked"] == 1 and tail["would_pin"] == dest.pinned_sbatch(sha2)
    assert not os.path.exists(dest.tree_dir(sha2))
    _git(src, "checkout", "--", ".")
    assert ST.running_revision(ST.Remote(dest, Runner()).run, src) == (sha2, [])


def test_a_restored_edit_outside_the_sample_is_caught_by_ctime_and_the_manifest_index_is_the_byte_guarantee(tmp_path):
    """(2) > 64 files; one file OUTSIDE the seeded sample is rewritten same-length with mode + mtime restored. The precheck
    catches it through ctime while the file has one link; once a later generation hard-links it, ctime leaves the key by design
    and the precheck admits it -- and `manifest_index` (the reader's per-input rule) refuses it in every case."""
    base = _canon(tmp_path, n_extra=300)
    root = str(tmp_path / "reg")
    cfg = ST.load_config(_cfg(tmp_path, {"big": {"base": base, "authority": "grading_host_copy", "exclude": ["__pycache__", "/.pytest_cache"]}}))
    comp, dest = cfg.components["big"], cfg.destination("cluster_a")
    r1 = ST.mirror_component(root, dest, comp, runner=Runner())
    assert r1.outcome == "published" and r1.n_files == 304
    gd, sha8 = dest.gen_dir("big"), r1.content_sha8
    pool = [rel for _s, _z, rel in ST.read_manifest_lines(f"{gd}/MANIFEST-{sha8}.sha256")]
    sample = set(random.Random(int(sha8, 16)).sample(pool, 64))
    victim = next(rel for rel in pool if rel.startswith("many/") and rel not in sample)
    p = f"{gd}/{sha8}/{victim}"
    assert ST.admit_generation(gd, sha8).ok and os.stat(p).st_nlink == 1
    original = open(p, "rb").read()
    _rewrite_preserving_stat(p, original.upper())                 # same length, mode + mtime restored: only ctime moves
    ad = ST.admit_generation(gd, sha8)
    assert not ad.ok and ad.refusal == "content_changed" and ad.mismatched == [victim] and ad.moved == 1 and ad.hashed == 65, ad.as_dict()
    idx = ST.manifest_index(gd, sha8)
    assert idx.sha8 == sha8 and len(idx.files) == 304 and idx.symlinks == {"link_to_top": "top.txt"} and idx.tree == f"{gd}/{sha8}"
    assert idx.check_file(victim).startswith(f"{victim}: bytes differ from the verified generation {sha8}")
    assert idx.check(victim, R.file_sha256(p)) is not None and idx.check(victim, idx.files[victim][0], idx.files[victim][1]) is None
    assert idx.check(victim, idx.files[victim][0], idx.files[victim][1] + 1) is not None
    assert "not in the verified generation" in idx.check("planted.txt", "0" * 64)
    assert idx.check_file("gone.txt") is not None
    _rewrite_preserving_stat(p, original)
    assert ST.admit_generation(gd, sha8).ok and idx.check_file(victim) is None
    # the index is never handed out unapproved: a tampered or absent report, or a manifest that does not hash to its name
    rep = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    json.dump({**rep, "ok": False}, open(f"{gd}/VERIFIED-{sha8}.json", "w"))
    with pytest.raises(ST.StoresError, match="verified_mismatch"):
        ST.manifest_index(gd, sha8)
    json.dump(rep, open(f"{gd}/VERIFIED-{sha8}.json", "w"))
    with pytest.raises(ST.StoresError, match="no_manifest"):
        ST.manifest_index(gd, "deadbeef")
    _write(f"{gd}/MANIFEST-deadbeef.sha256", open(f"{gd}/MANIFEST-{sha8}.sha256", "rb").read())
    with pytest.raises(ST.StoresError, match="manifest_mismatch"):
        ST.manifest_index(gd, "deadbeef")
    # a second generation hard-links the untouched files (nlink 2): their ctime is out of the key by design, so the same attack
    # on a file outside gen2's sample passes the precheck -- the reader's per-input hash against the index is what refuses it
    with open(os.path.join(base, "top.txt"), "ab") as fh:
        fh.write(b"more\n")
    r2 = ST.mirror_component(root, dest, comp, runner=Runner())
    assert r2.outcome == "published" and r2.content_sha8 != sha8
    sha8_2 = r2.content_sha8
    pool2 = [rel for _s, _z, rel in ST.read_manifest_lines(f"{gd}/MANIFEST-{sha8_2}.sha256")]
    sample2 = set(random.Random(int(sha8_2, 16)).sample(pool2, 64))
    victim2 = next(rel for rel in pool2 if rel.startswith("many/") and rel not in sample2 and rel != victim)
    p2 = f"{gd}/{sha8_2}/{victim2}"
    assert os.stat(p2).st_nlink == 2 and json.load(open(f"{gd}/DEST-STAT-{sha8_2}.json"))["files"][victim2][4] == 0
    assert ST.admit_generation(gd, sha8_2).ok
    original2 = open(p2, "rb").read()
    _rewrite_preserving_stat(p2, original2.upper())
    ad2 = ST.admit_generation(gd, sha8_2)
    assert ad2.ok and ad2.moved == 0, "the documented limit of the precheck: a hard-linked file's ctime is not in the key"
    idx2 = ST.manifest_index(gd, sha8_2)
    assert idx2.check_file(victim2).startswith(f"{victim2}: bytes differ from the verified generation {sha8_2}"), "the guarantee is the reader's"
    assert idx2.check(victim2, R.file_sha256(p2), os.stat(p2).st_size) is not None
    _rewrite_preserving_stat(p2, original2)
    assert idx2.check_file(victim2) is None


# --------------------------------------------------------------------------------------------
# the report-visibility defects (2026-10-05): a VERIFIED report the compute node wrote is not visible from the login node
# for up to ~60 s, so the read right after DONE saw `{}` and renamed 48/48 good trees `.failed-*` (never a real mismatch);
# the shortcuts then trusted the report and `current` over a tree that was gone. The report is now read with a bounded
# wait, an unseen report is `verify_report_missing` with the tree in place, and both shortcuts probe the tree.
# --------------------------------------------------------------------------------------------


class _LateReport(Runner):
    """A runner on which the VERIFIED report a verify job wrote is not visible from the destination for the first ``hidden``
    reads AFTER the job (the login node's NFS cache lagging the compute node's write); ``hidden=None`` never shows it. Reads
    before the job pass through (the report is genuinely absent then), and so does everything else."""

    def __init__(self, hidden=1, **kw):
        super().__init__(**kw)
        self.hidden = hidden
        self.job_ran = False
        self.hidden_reads = 0

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[0] == "sbatch":
            r = super().__call__(argv, timeout_s, input=input, check=check)
            self.job_ran = True
            return r
        if (self.job_ran and argv[:2] == ["sh", "-c"] and "VERIFIED-" in argv[2] and "cat " in argv[2]
                and (self.hidden is None or self.hidden_reads < self.hidden)):
            self.hidden_reads += 1
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 44, stdout="", stderr="")      # `read`: exit 44 = no such file
        return super().__call__(argv, timeout_s, input=input, check=check)


def _renamed_away(gd, sha8, stamp="20261004T234000445Z"):
    """What every tick before 2026-10-05 left behind a false `verify_failed`: the good tree renamed `.failed-*`, the report and
    `current` (when it pointed here) still naming it."""
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-{stamp}")
    assert not os.path.isdir(f"{gd}/{sha8}") and os.path.isfile(f"{gd}/VERIFIED-{sha8}.json")


def test_a_report_visible_only_on_the_second_read_is_waited_for_and_published(world):
    """(a) the watch says DONE, the first read of the report sees nothing, the second sees it: `published`, no rename, no
    `.failed-*`, one poll's sleep -- never the 2026-09-29..10-05 failure."""
    w = world
    sleeps = []
    rn = _LateReport(hidden=1)
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, sleep=sleeps.append)
    assert res.outcome == "published" and res.ok, res.as_dict()
    gd, sha8 = w["gen_dir"], res.content_sha8
    assert os.readlink(f"{gd}/current") == sha8 and os.path.isdir(f"{gd}/{sha8}")
    assert not any(d.startswith((".failed-", ".stale-")) for d in os.listdir(gd)), os.listdir(gd)
    assert rn.hidden_reads == 1 and sleeps == [w["dest"].report_poll_s] == [15.0]
    assert res.detail["report_wait"] == {"attempts": 2, "waited_s": 15.0, "last": "present"}
    assert res.detail["verified"]["host"] and res.job_id == json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["job_id"]
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["published"] == sha8 and st["generation"]["state"] == "published" and st["last_run"]["outcome"] == "published"
    assert ST.admit_generation(gd, sha8).ok


def test_a_report_never_visible_inside_the_bound_leaves_the_tree_and_the_next_tick_publishes_it(world):
    """(e) nine reads, 15 s apart, 120 s in all; still nothing: `verify_report_missing` -- the tree stays where it is, nothing is
    renamed, the state names the job and the wait; the next tick finds the report and publishes without a copy or a job."""
    w = world
    sleeps = []
    rn = _LateReport(hidden=None)
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, sleep=sleeps.append)
    assert res.outcome == "verify_report_missing" and not res.ok and res.published is None, res.as_dict()
    gd, sha8 = w["gen_dir"], res.content_sha8
    assert os.path.isdir(f"{gd}/{sha8}") and not os.path.lexists(f"{gd}/current"), "the tree is LEFT IN PLACE, unpublished"
    assert not any(d.startswith((".failed-", ".invalid-", ".stale-")) for d in os.listdir(gd)), os.listdir(gd)
    assert sleeps == [15.0] * 8 and rn.hidden_reads == 9
    assert res.detail["report_wait"] == {"attempts": 9, "waited_s": 120.0, "last": "absent"}
    assert res.detail["verified_present"] is False and res.detail["tree_left"] == f"{gd}/{sha8}" and "renamed_to" not in res.detail
    assert res.job_id and res.job_id in res.detail["why"] and "120 s" in res.detail["why"] and "left in place" in res.detail["why"]
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    g = st["generation"]
    assert g["state"] == "verify_report_missing" and g["job_id"] == res.job_id and g["attempts"] == 9 and g["waited_s"] == 120.0 and g["last_read"] == "absent"
    assert st.get("published") is None and st["last_run"]["outcome"] == "verify_report_missing" and st["last_run"]["last_read"] == "absent"
    # the job DID write the report (it is on the destination; only the login node could not see it yet): the next tick's
    # existing-verified shortcut publishes the tree in place -- no rsync, no sbatch
    assert json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["job_id"] == res.job_id
    rn2 = Runner()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2)
    assert r2.outcome == "published_existing_verified" and r2.ok and os.readlink(f"{gd}/current") == sha8, r2.as_dict()
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn2.calls)
    assert ST.load_state(w["root"])["components"]["relu_paired_run_state"]["generation"]["state"] == "published"
    assert ST.admit_generation(gd, sha8).ok
    # the bound and the poll are per destination
    import dataclasses
    dest = dataclasses.replace(w["dest"], report_wait_s=30.0, report_poll_s=10.0)
    with open(os.path.join(w["base"], "top.txt"), "ab") as fh:
        fh.write(b"more\n")
    sleeps.clear()
    r3 = ST.mirror_component(w["root"], dest, w["comp"], runner=_LateReport(hidden=None), sleep=sleeps.append)
    assert r3.outcome == "verify_report_missing" and sleeps == [10.0] * 3 and r3.detail["report_wait"] == {"attempts": 4, "waited_s": 30.0, "last": "absent"}


def test_a_dangling_current_is_repaired_by_a_fresh_copy_and_verify_never_called_unchanged(world):
    """(b) `current -> <sha8>` and a matching report, but the tree was renamed away (what the old code did to `bfcl_full`
    and `bigcodebench_full`): rsync + sbatch run, outcome `republished_dangling_current` under the same sha8 (its own
    name in the result, the record's `last_run` and the generation) -- not `unchanged`, not `published_existing_verified` --
    with no `--link-dest` at the dangling directory, and the stale report moved aside."""
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r1.outcome == "published"
    gd, sha8 = w["gen_dir"], r1.content_sha8
    _renamed_away(gd, sha8)
    assert os.readlink(f"{gd}/current") == sha8, "the pointer dangles"
    old = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert not ST.admit_generation(gd, sha8).ok and ST.admit_generation(gd, sha8).refusal == "generation_changed", "what the shards saw"
    rn = Runner()
    rn.job = 800                                            # a fresh Runner would mint the same job id as the first tick's
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "republished_dangling_current" and r2.content_sha8 == sha8 and r2.ok and r2.published == sha8, r2.as_dict()
    assert r2.detail["current_dangling"] is True and "not a directory" in r2.detail["why_copied"] and r2.detail["resumed_repair"] is False
    assert any("rsync" in a for a in rn.calls) and any(a[0] == "sbatch" for a in rn.calls), "copied afresh and re-verified"
    rs = next(a for a in rn.calls if "rsync" in a)
    assert not any(a.startswith("--link-dest") for a in rs) and r2.detail["link_dest"] is None, "never a link-dest at a directory that is not there"
    assert len([a for a in rn.calls if "rsync" in a]) == 1, "no forced pass without a link-dest"
    assert os.path.isdir(f"{gd}/{sha8}") and os.readlink(f"{gd}/current") == sha8
    new = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert new["ok"] is True and new["job_id"] == "801" != old["job_id"] and r2.job_id == "801"
    stale = r2.detail["stale_report_renamed_to"]
    assert os.path.basename(stale).startswith(f".stale-VERIFIED-{sha8}-") and stale.endswith(".json") and os.path.isfile(stale)
    assert json.load(open(stale))["job_id"] == old["job_id"], "the stale report is kept aside, never deleted"
    assert os.path.isdir(f"{gd}/.failed-{sha8}-20261004T234000445Z"), "nothing is deleted"
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["published"] == sha8 and st["generation"]["state"] == "published" and st["generation"]["repaired_dangling_current"] is True
    assert st["last_run"]["outcome"] == "republished_dangling_current" and st["current_dangling"] is None, st
    assert ST.admit_generation(gd, sha8).ok, "the fresh verify rewrote DEST-STAT for the fresh tree"
    # and now it IS unchanged
    rn3 = Runner()
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn3).outcome == "unchanged" and not any("rsync" in a for a in rn3.calls)


def test_a_report_over_an_absent_tree_with_no_current_is_copied_not_published_as_existing(world):
    """(c) a matching report, the tree gone, no `current`: never `published_existing_verified` over a directory that is not
    there; the copy path runs and publishes a verified tree."""
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    _renamed_away(gd, sha8)
    os.remove(f"{gd}/current")
    rn = Runner()
    rn.job = 800
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "published" and r2.content_sha8 == sha8, r2.as_dict()
    assert "current_dangling" not in r2.detail and r2.detail["link_dest"] is None
    assert any("rsync" in a for a in rn.calls) and any(a[0] == "sbatch" for a in rn.calls)
    assert os.path.isdir(f"{gd}/{sha8}") and os.readlink(f"{gd}/current") == sha8 and json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["job_id"] == "801"
    assert os.path.basename(r2.detail["stale_report_renamed_to"]).startswith(f".stale-VERIFIED-{sha8}-")
    assert ST.admit_generation(gd, sha8).ok
    # the previous generation's tree gone while a NEWER content is due: no link-dest at the missing directory either
    _renamed_away(gd, sha8, stamp="20261005T000000000Z")
    with open(os.path.join(w["base"], "top.txt"), "ab") as fh:
        fh.write(b"more\n")
    rn4 = Runner()
    rn4.job = 900
    r4 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn4)
    assert r4.outcome == "published" and r4.content_sha8 != sha8 and r4.detail["link_dest"] is None, r4.as_dict()
    assert not any(a.startswith("--link-dest") for a in next(a for a in rn4.calls if "rsync" in a))
    assert os.readlink(f"{gd}/current") == r4.content_sha8 and ST.admit_generation(gd, r4.content_sha8).ok


class _RsyncCut(Runner):
    """A runner whose rsync is cut short -- the transport died under the copy, the way an ssh outage mid-repair looks: with
    ``partial=True`` the transfer runs before rsync reports the failure (a partial tree is left), with ``partial=False`` nothing
    lands. Everything else passes through."""

    def __init__(self, partial=False, **kw):
        super().__init__(**kw)
        self.partial = partial

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if "rsync" in argv:
            self.calls.append(argv)
            if self.partial:
                subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, input=input)
            return subprocess.CompletedProcess(argv, 255, stdout="", stderr="rsync: connection unexpectedly closed (code 255)")
        return super().__call__(argv, timeout_s, input=input, check=check)


class _ListingBlip(Runner):
    """A runner on which the renamed-tree listing (the dangling path's one extra ssh step) fails like an ssh outage (rc 255);
    everything else runs as usual."""

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[:2] == ["sh", "-c"] and "for p in .failed-" in argv[2]:
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: connect to host login-0 port 22: Connection timed out")
        return super().__call__(argv, timeout_s, input=input, check=check)


def test_the_first_dangling_tick_drops_published_before_the_listing_so_a_failure_there_leaves_an_honest_record(world, monkeypatch, capsys):
    """A healthy publish, the tree renamed away, and the FIRST dangling tick's renamed-tree listing dies on the
    transport. The outcome is `error`, but the record already reads `published` None + `current_dangling` (sha8, when, why;
    `renamed_aside` not yet listed) and the generation is no longer `published` -- never the old `published: <sha8>` merged
    under an error, which would keep `cluster-reads plan` pinning a missing tree. The plan's pin set is empty and `stores
    status` prints the PROBLEM line with no probe; the next tick republishes and clears."""
    w = world
    slug = "relu_paired_run_state"
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    before = ST.load_state(w["root"])["components"][slug]
    assert before["published"] == sha8 and before["generation"]["state"] == "published" and before.get("current_dangling") is None
    _renamed_away(gd, sha8, stamp="20261005T085946168Z")
    rn = _ListingBlip()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "error" and not r2.ok and "list the renamed trees" in r2.detail["error"] and "rc 255" in r2.detail["error"], r2.as_dict()
    assert r2.detail["current_dangling"] is True and "renamed_aside" not in r2.detail
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn.calls) and not os.path.isdir(f"{gd}/{sha8}")
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] is None and st["last_run"]["outcome"] == "error", st
    d = st["current_dangling"]
    assert d["content_sha8"] == sha8 and d["found_utc"] and d["renamed_aside"] is None and d["resumed"] is False
    assert d["why"] == f"current -> {sha8} but {gd}/{sha8} is not a directory; beside it: not yet listed"
    assert st["generation"] == {"content_sha8": sha8, "state": "current_dangling", "found_utc": d["found_utc"]}
    assert {(s_, rec.get("published")) for s_, rec in ST.load_state(w["root"])["components"].items() if rec.get("published")} == set()
    assert cli.main(["stores", "status"]) == 0
    text = capsys.readouterr().out
    assert f"PROBLEM {slug}: current -> {sha8} but {gd}/{sha8} is not a directory; beside it: not yet listed" in text and "last run error" in text, text
    assert f"{sha8}:current_dangling" in text and f"| {sha8} |" not in text.split("PROBLEM")[0].replace(f"{sha8}:", ""), "the row's published column is empty"
    assert cli.main(["stores", "status", "--json"]) == 0
    row = json.loads(capsys.readouterr().out)["components"][slug]
    assert row["published"] is None and row["verified"] is False and row["current_dangling"] == d
    # the next tick lists, copies afresh, republishes; the record and the status are clean
    rn3 = Runner()
    rn3.job = 800
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn3)
    assert r3.outcome == "republished_dangling_current" and r3.detail["renamed_aside"] == [f".failed-{sha8}-20261005T085946168Z"], r3.as_dict()
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] == sha8 and st["current_dangling"] is None and st["generation"]["state"] == "published"
    assert cli.main(["stores", "status"]) == 0 and "PROBLEM" not in capsys.readouterr().out
    assert ST.admit_generation(gd, sha8).ok


def test_unchanged_content_over_a_healthy_destination_is_unchanged_with_one_tree_probe_and_no_rehash(world):
    """The healthy shape: `current -> <sha8>`, the tree a directory, the report matching -- `unchanged`, the record keeps
    `published` and carries no dangle; the destination is asked exactly six stat-class steps (mkdir -p, readlink current,
    cat VERIFIED, test -d <tree>, then the evidence beside `current`: cat SEALED, cat PUBLISHED -- the rule that a `current`
    without its evidence is never `unchanged`): no listing of renamed trees, no rsync, no sbatch, nothing re-hashed on the grading host."""
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r1.outcome == "published"
    gd, sha8 = w["gen_dir"], r1.content_sha8
    rn = Runner()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "unchanged" and r2.ok and r2.published == sha8 and r2.hashed == 0 and r2.from_cache == r2.n_files == r1.n_files, r2.as_dict()
    shell = [a[2] for a in rn.calls if a[:2] == ["sh", "-c"]]
    assert len(shell) == len(rn.calls) == 6, rn.calls
    assert shell[0].startswith("mkdir -p ") and "readlink" in shell[1] and f"VERIFIED-{sha8}.json" in shell[2], shell
    assert f"SEALED-{sha8}.json" in shell[4] and f"PUBLISHED-{sha8}.json" in shell[5], shell          # the evidence reads (C3)
    assert shell[3] == f"if test -d {gd}/{sha8}; then exit 0; else exit 44; fi", "the one probe: the tree, by its sha8 path"
    assert not any(".failed-" in c or ".invalid-" in c for c in shell), "the renamed-tree listing runs only when the pointer dangles"
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["published"] == sha8 and st["current_dangling"] is None and st["generation"]["state"] == "published" and st["last_run"]["outcome"] == "unchanged"


def test_a_dangling_current_names_the_renamed_trees_beside_it_in_the_log_and_the_record(world):
    """The `.failed-<sha8>-*` and `.invalid-<sha8>-*` trees beside a dangling `current` are listed by the one extra probe
    (this sha8's only) and named in the log line, the result and the record; the republish carries them in the generation
    record and clears the dangle; nothing is deleted."""
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    _renamed_away(gd, sha8)
    os.makedirs(f"{gd}/.invalid-{sha8}-20261004T220000000Z")
    os.makedirs(f"{gd}/.failed-0badf00d-20261004T210000000Z")        # another generation's rename: not this pointer's
    aside = [f".failed-{sha8}-20261004T234000445Z", f".invalid-{sha8}-20261004T220000000Z"]
    lines = []
    rn = Runner()
    rn.job = 800
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, log=lines.append)
    assert r2.outcome == "republished_dangling_current" and r2.ok and r2.detail["renamed_aside"] == aside, r2.as_dict()
    assert all(a in r2.detail["why_copied"] for a in aside)
    dangle = [l for l in lines if "is dangling" in l]
    assert len(dangle) == 1 and dangle[0].startswith(f"[relu_paired_run_state] current -> {sha8} is dangling: {gd}/{sha8} is not a directory"), lines
    assert all(a in dangle[0] for a in aside) and "0badf00d" not in dangle[0] and "copying afresh" in dangle[0], dangle[0]
    probes = [a[2] for a in rn.calls if a[:2] == ["sh", "-c"] and f".failed-{sha8}-*" in a[2]]
    assert len(probes) == 1 and f".invalid-{sha8}-*" in probes[0] and "test -e" in probes[0], "one stat-class listing, on the dangling path only"
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["published"] == sha8 and st["current_dangling"] is None and st["last_run"]["outcome"] == "republished_dangling_current"
    assert st["generation"]["state"] == "published" and st["generation"]["repaired_dangling_current"] is True and st["generation"]["renamed_aside"] == aside
    assert all(os.path.isdir(f"{gd}/{a}") for a in aside) and os.path.isdir(f"{gd}/.failed-0badf00d-20261004T210000000Z"), "nothing is deleted"
    # no rename beside the pointer at all: still named, as such
    _renamed_away(gd, sha8, stamp="20261005T010000000Z")
    for a in aside + [f".failed-{sha8}-20261005T010000000Z"]:
        os.rename(f"{gd}/{a}", f"{gd}/elsewhere-{a}")
    lines.clear()
    rn3 = Runner()
    rn3.job = 900
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn3, log=lines.append)
    assert r3.outcome == "republished_dangling_current" and r3.detail["renamed_aside"] == [] and "beside it: no renamed tree" in lines[0], (r3.as_dict(), lines)


def test_a_dangling_current_whose_repair_is_cut_short_is_a_recorded_problem_and_the_next_tick_resumes_it(world, monkeypatch, capsys):
    """The repair's rsync dies (1) before anything lands, then (2) after the whole tree landed but before rsync could say
    so. Either way the record drops `published` and carries `current_dangling` -- what `cluster-reads plan` pins from is empty
    and `stores status` prints a PROBLEM line with no probe -- and the next tick republishes: after (2) it RESUMES the copy over
    its own partial tree, never `current_unverified` (that stays for a pointer published behind the mirror's back)."""
    w = world
    slug = "relu_paired_run_state"
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])

    def pins():
        return {(s_, rec.get("published")) for s_, rec in ST.load_state(w["root"])["components"].items() if rec.get("published")}

    def status_text():
        assert cli.main(["stores", "status"]) == 0
        return capsys.readouterr().out

    def status_row():
        assert cli.main(["stores", "status", "--json"]) == 0
        return json.loads(capsys.readouterr().out)["components"][slug]

    assert pins() and "PROBLEM" not in status_text() and status_row()["verified"] is True
    # (1) the pointer dangles and the repair's copy dies before anything lands
    _renamed_away(gd, sha8, stamp="20261005T085946168Z")
    lines = []
    rn = _RsyncCut(partial=False)
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, log=lines.append)
    assert r2.outcome == "rsync_failed" and not r2.ok and r2.published is None and r2.detail["current_dangling"] is True, r2.as_dict()
    assert r2.detail["resumed_repair"] is False and not os.path.isdir(f"{gd}/{sha8}") and not any(a[0] == "sbatch" for a in rn.calls)
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] is None and st["generation"]["state"] == "copying" and st["last_run"]["outcome"] == "rsync_failed", st
    d = st["current_dangling"]
    assert d["content_sha8"] == sha8 and d["renamed_aside"] == [f".failed-{sha8}-20261005T085946168Z"] and d["resumed"] is False and d["found_utc"]
    assert d["why"] == f"current -> {sha8} but {gd}/{sha8} is not a directory; beside it: .failed-{sha8}-20261005T085946168Z"
    assert pins() == set(), "nothing to pin: the plan routes the adapter to the grading host with a reason, never at a dangling pointer"
    text = status_text()
    assert f"PROBLEM {slug}: current -> {sha8} but {gd}/{sha8} is not a directory; beside it: .failed-{sha8}-20261005T085946168Z" in text, text
    assert "last run rsync_failed" in text and "nothing is published until the next tick republishes it" in text
    row = status_row()
    assert row["published"] is None and row["verified"] is False and row["current_dangling"] == d
    # the next tick: still dangling (the tree is not there), copied afresh, republished; the record and the status are clean again
    rn2 = Runner()
    rn2.job = 800
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2)
    assert r3.outcome == "republished_dangling_current" and r3.detail["resumed_repair"] is False and os.readlink(f"{gd}/current") == sha8, r3.as_dict()
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] == sha8 and st["current_dangling"] is None and st["last_run"]["outcome"] == "republished_dangling_current"
    assert pins() == {(slug, sha8)} and "PROBLEM" not in status_text() and status_row()["verified"] is True
    assert ST.admit_generation(gd, sha8).ok
    # (2) the pointer dangles again and the repair's copy lands in full but rsync dies before reporting: the tree IS a directory
    # now, no report matches it (the stale one was moved aside), and `current` still names it
    _renamed_away(gd, sha8, stamp="20261005T100000000Z")
    rn = _RsyncCut(partial=True)
    r4 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r4.outcome == "rsync_failed" and r4.detail["current_dangling"] is True and os.path.isdir(f"{gd}/{sha8}"), r4.as_dict()
    assert not os.path.exists(f"{gd}/VERIFIED-{sha8}.json") and os.path.basename(r4.detail["stale_report_renamed_to"]).startswith(f".stale-VERIFIED-{sha8}-")
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] is None and st["current_dangling"]["content_sha8"] == sha8 and pins() == set()
    assert f"PROBLEM {slug}: current -> {sha8}" in status_text()
    # the next tick resumes the repair over the mirror's own partial copy: rsync (resumed, no link-dest) + verify + republish --
    # NOT `current_unverified`, which is for a pointer published behind the mirror's back
    lines.clear()
    rn5 = Runner()
    rn5.job = 900
    r5 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn5, log=lines.append)
    assert r5.outcome == "republished_dangling_current" and r5.ok and r5.detail["resumed_repair"] is True and r5.detail["link_dest"] is None, r5.as_dict()
    assert any("rsync" in a for a in rn5.calls) and any(a[0] == "sbatch" for a in rn5.calls) and r5.job_id == "901"
    assert lines[0].startswith(f"[{slug}] current -> {sha8} is dangling: {gd}/{sha8} is the partial copy of the repair begun ") and "resuming the copy" in lines[0], lines
    assert "no report matches it" in lines[0] and f".failed-{sha8}-20261005T100000000Z" in lines[0]
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] == sha8 and st["current_dangling"] is None and st["generation"]["repaired_dangling_current"] is True
    assert st["last_run"]["outcome"] == "republished_dangling_current" and pins() == {(slug, sha8)} and "PROBLEM" not in status_text()
    assert ST.admit_generation(gd, sha8).ok and json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["job_id"] == "901"
    # and a tick later it is plainly unchanged, over the six probes (the four tree probes plus the SEALED / PUBLISHED evidence reads) and nothing else
    rn6 = Runner()
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn6).outcome == "unchanged" and len(rn6.calls) == 6
    # P1-1 still holds: a pointer whose report is broken behind the mirror's back (no dangle recorded) is reported, never written into
    rep = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    rep["ok"] = False
    json.dump(rep, open(f"{gd}/VERIFIED-{sha8}.json", "w"))
    rn7 = Runner()
    r7 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn7)
    assert r7.outcome == "current_unverified" and not any("rsync" in a or a[0] == "sbatch" for a in rn7.calls), r7.as_dict()
    assert ST.load_state(w["root"])["components"][slug]["current_dangling"] is None


def test_a_transport_failure_after_the_probes_keeps_the_recorded_dangle(world):
    """A recorded dangle is cleared ONLY by a positive verdict -- the pointer names a directory, or a publish. A tick that
    errors between the probes and the repair (the renamed-tree listing over a dead ssh) leaves the record saying what it said:
    `published` None, `current_dangling` set; the tick after it repairs and clears."""
    w = world
    slug = "relu_paired_run_state"
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    _renamed_away(gd, sha8)
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=_RsyncCut(partial=False))
    assert r2.outcome == "rsync_failed" and ST.load_state(w["root"])["components"][slug]["current_dangling"]["content_sha8"] == sha8
    rn = _ListingBlip()
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r3.outcome == "error" and "list the renamed trees" in r3.detail["error"] and "rc 255" in r3.detail["error"], r3.as_dict()
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn.calls)
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] is None and st["current_dangling"]["content_sha8"] == sha8 and st["last_run"]["outcome"] == "error", st
    rn4 = Runner()
    rn4.job = 800
    r4 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn4)
    assert r4.outcome == "republished_dangling_current", r4.as_dict()
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] == sha8 and st["current_dangling"] is None
    # and the other positive verdict clears a dangle the record still carried: the pointer names a directory
    ST.update_state(w["root"], slug, {"current_dangling": {"content_sha8": "0badf00d", "why": "stale"}, "published": None}, w["dest"].name)
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner()).outcome == "unchanged"
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] == sha8 and st["current_dangling"] is None


class _SealBlip(Runner):
    """A runner on which the publication's directory seal (`find ... -exec chmod 0555`) fails like an ssh outage (rc 255) the first
    ``times`` times it is asked -- the transient destination error between a repair's verify and its switch; everything else runs as usual."""

    def __init__(self, times=1, **kw):
        super().__init__(**kw)
        self.times = times
        self.blips = 0

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[:2] == ["sh", "-c"] and "-exec chmod" in argv[2] and self.blips < self.times:
            self.blips += 1
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: connect to host login-0 port 22: Connection timed out")
        return super().__call__(argv, timeout_s, input=input, check=check)


def _dir_modes(top):
    return {dp: stat_module.S_IMODE(os.lstat(dp).st_mode) for dp, _d, _f in os.walk(top)}


def _set_dir_modes(top, mode):
    for d in sorted(_dir_modes(top), key=len, reverse=True):
        os.chmod(d, mode)


def _window_pins(w, adapter="relu_paired", window="20261007T1900Z"):
    """What a window's plan pins for the component: `cluster_reads.plan_window` reads `published` out of
    `state/store_mirror.json`, exactly as the shards do."""
    from sisteer.registry import cluster_reads as CR
    plan = CR.plan_window(w["root"], window, [adapter], w["cfg"], w["dest"], tree_sha="0" * 40)
    return plan["adapters"][adapter]["pins"]


def test_a_dangling_repair_whose_seal_fails_is_pending_until_its_own_seal_and_record_stand_never_unchanged(world, monkeypatch, capsys):
    """The dangling-pointer repair copies the replacement under the SAME sha8 the pointer names, so once its fresh verify report is
    beside it the destination reads like a healthy `current` -- and before this, when the repair's directory seal failed on the
    transport, the next tick read the pointer, the present tree, the matching report and the PREVIOUS tree's SEALED and PUBLISHED records
    as `unchanged` (green): writable directories certified by the old seal, a publication record describing a tree that no longer
    existed. Now the repair is PENDING until its own seal and publication stand: the previous tree's seal goes aside with its reports
    when the repair starts, the publication's in-flight state is written before the seal, and the next tick finishes the seal and the
    guarded switch (`republished_dangling_current`, the record rewritten) with no copy and no job -- `unchanged` only after. With the
    grading host's record lost in between, the destination alone still never reads green: a recorded, unsealed `current` is
    `current_unsealed` (red) to a plain tick, and `--adopt-unrecorded` seals it."""
    w = world
    slug = "relu_paired_run_state"
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r1.outcome == "published"
    gd, sha8 = w["gen_dir"], r1.content_sha8
    gen_path = f"{gd}/{sha8}"
    old_seal = json.load(open(f"{gd}/SEALED-{sha8}.json"))
    old_rec = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])

    def pins():
        return {(s_, rec.get("published")) for s_, rec in ST.load_state(w["root"])["components"].items() if rec.get("published")}

    def status_text():
        assert cli.main(["stores", "status"]) == 0
        return capsys.readouterr().out

    def row():
        return ST.status(w["root"], w["cfg"])["components"][slug]

    # (1) the pointer dangles; the repair copies afresh and verifies, then its seal dies on the transport: error, nothing published
    _renamed_away(gd, sha8, stamp="20261007T140000000Z")
    rn = _SealBlip()
    rn.job = 800
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert r2.outcome == "error" and not r2.ok and r2.published is None and "seal" in r2.detail["error"] and "rc 255" in r2.detail["error"], r2.as_dict()
    assert r2.detail["current_dangling"] is True and r2.detail["resumed_repair"] is False and rn.blips == 1
    assert os.path.isdir(gen_path) and os.readlink(f"{gd}/current") == sha8
    fresh = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert fresh["ok"] is True and fresh["job_id"] == "801", "the replacement is verified; only its seal and its switch are missing"
    assert not os.path.lexists(f"{gd}/current") or any(m != 0o555 for m in _dir_modes(gen_path).values()), "the replacement's directories are writable"
    # the previous tree's SEALED record went aside with its reports (never deleted); the pointer's PUBLISHED record stands for the guard
    aside = r2.detail["stale_reports_renamed_to"]
    stale_seal = [p for p in aside if os.path.basename(p).startswith(f".stale-SEALED-{sha8}-")]
    assert len(stale_seal) == 1 and json.load(open(stale_seal[0])) == old_seal and not os.path.exists(f"{gd}/SEALED-{sha8}.json"), aside
    assert json.load(open(f"{gd}/PUBLISHED-{sha8}.json")) == old_rec
    # the record: nothing published, the dangle pending in its publishing phase, the generation in flight (amber), the PROBLEM line, no pin
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] is None and st["last_run"]["outcome"] == "error", st
    assert st["generation"]["state"] == "publishing" and st["generation"]["repaired_dangling_current"] is True and st["generation"]["job_id"] == "801"
    d = st["current_dangling"]
    assert d["content_sha8"] == sha8 and d["phase"] == "publishing" and d["renamed_aside"] == [f".failed-{sha8}-20261007T140000000Z"] and d["resumed"] is False
    assert pins() == set() and f"PROBLEM {slug}: current -> {sha8}" in status_text()
    assert row()["published"] is None and row()["verified"] is False and row()["level"] == "amber"
    # (2) the next tick: the pointer, the tree, a MATCHING report and the previous tree's PUBLISHED record -- `unchanged` before this; now the
    # pending publication: sealed, switched through the guard, the record rewritten; no copy, no job
    lines = []
    rn3 = Runner()
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn3, log=lines.append)
    assert r3.outcome == "republished_dangling_current" and r3.ok and r3.published == sha8 and r3.content_sha8 == sha8, r3.as_dict()
    assert r3.detail["pending_publication"] is True and r3.detail["current_dangling"] is True and r3.detail["resumed_repair"] is False
    assert r3.detail["renamed_aside"] == [f".failed-{sha8}-20261007T140000000Z"] and "did not complete" in r3.detail["why_republished"]
    assert "never called unchanged" in r3.detail["why_republished"] and r3.job_id == "801" and r3.detail["verified"]["verified_utc"] == fresh["verified_utc"]
    assert not any("rsync" in a or a[0] == "sbatch" or a[0] == WATCH for a in rn3.calls), "nothing copied or re-verified: the fresh report stands"
    assert lines and "is dangling" in lines[0] and "sealing and republishing" in lines[0], lines
    assert all(m == 0o555 for m in _dir_modes(gen_path).values()), "the replacement's directories are sealed now"
    seal = json.load(open(f"{gd}/SEALED-{sha8}.json"))
    assert seal["resealed"] is False and seal["sealed_utc"] > old_seal["sealed_utc"] and seal["first_sealed_utc"] == seal["sealed_utc"], "the replacement's own seal, not a re-seal"
    assert seal["n_dirs"] == old_seal["n_dirs"] and r3.detail["sealed"] == {"n_dirs": seal["n_dirs"], "sealed_utc": seal["sealed_utc"], "resealed": False}
    rec = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    assert rec != old_rec and rec["published_utc"] > old_rec["published_utc"] and rec["previous"] == sha8 and rec["mode"] == "mirror" and rec["content_sha8"] == sha8
    assert r3.detail["published_record"]["published_utc"] == rec["published_utc"] and os.readlink(f"{gd}/current") == sha8
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] == sha8 and st["current_dangling"] is None and st["generation"]["state"] == "published", st
    assert st["generation"]["repaired_dangling_current"] is True and st["generation"]["pending_publication"] is True
    assert st["last_run"]["outcome"] == "republished_dangling_current" and pins() == {(slug, sha8)} and "PROBLEM" not in status_text()
    assert row()["published"] == sha8 and row()["verified"] is True and row()["level"] == "green"
    assert ST.admit_generation(gd, sha8).ok
    # (3) and only now plainly unchanged, over the six probes
    rn4 = Runner()
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn4).outcome == "unchanged" and len(rn4.calls) == 6
    # (4) the seal fails again and the grading host record is lost between the ticks: the destination alone never reads green -- the previous tree's
    # seal is aside, so a recorded `current` without its SEALED record is `current_unsealed` (red) to a plain tick, nothing adopted
    # implicitly; `--adopt-unrecorded` seals it (the record stands), and only then is it unchanged
    _renamed_away(gd, sha8, stamp="20261007T150000000Z")
    rn5 = _SealBlip()
    rn5.job = 900
    r5 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn5)
    assert r5.outcome == "error" and json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["job_id"] == "901" and not os.path.exists(f"{gd}/SEALED-{sha8}.json")
    assert ST.load_state(w["root"])["components"][slug]["current_dangling"]["phase"] == "publishing"
    ST.update_state(w["root"], slug, {"current_dangling": None, "published": None}, w["dest"].name)          # the record lost
    rn6 = Runner()
    r6 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn6)
    assert r6.outcome == "current_unsealed" and r6.level == "red" and not r6.ok and r6.detail["authority_problems"] == ["current_unsealed"], r6.as_dict()
    assert "--adopt-unrecorded" in r6.detail["why"] and f"stores push {slug}" in r6.detail["why"] and "never called unchanged" in r6.detail["why"]
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn6.calls) and not os.path.exists(f"{gd}/SEALED-{sha8}.json")
    assert any(m != 0o555 for m in _dir_modes(gen_path).values()), "nothing sealed implicitly"
    st = ST.load_state(w["root"])["components"][slug]
    assert st["generation"]["state"] == "current_unsealed" and st["last_run"]["outcome"] == "current_unsealed" and row()["level"] == "red"
    # the refusal never re-advertises the pointer it has just called red (the final pass on 2c): the seal gates publication, so the
    # state stays unpublished and the window's plan pins nothing -- before this `refuse` wrote `published = current` back because the
    # directory stood, and a shard read a verified-but-writable replacement
    assert st["published"] is None and pins() == set() and _window_pins(w) == {}
    rn7 = Runner()
    r7 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn7, adopt_unrecorded=True)
    assert r7.outcome == "unchanged" and r7.ok and set(r7.detail["adopted"]) == {"sealed"} and r7.detail["adopted"]["sealed"]["n_dirs"] == seal["n_dirs"], r7.as_dict()
    assert all(m == 0o555 for m in _dir_modes(gen_path).values()) and json.load(open(f"{gd}/SEALED-{sha8}.json"))["content_sha8"] == sha8
    assert json.load(open(f"{gd}/PUBLISHED-{sha8}.json")) == rec, "the record stood; the seal is what the adoption wrote"
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn7.calls)
    rn8 = Runner()
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn8).outcome == "unchanged" and len(rn8.calls) == 6
    assert ST.admit_generation(gd, sha8).ok and pins() == {(slug, sha8)} and row()["level"] == "green"


def test_the_seal_gates_publication_so_a_failed_adoption_publishes_nothing_and_an_unsealed_generation_is_never_admitted(world):
    """The final pass on 2c (HIGH): THE SEAL GATES PUBLICATION, and the admission reads it. Two halves, both open before this:
    the mirror kept ``published = current`` through a `current_unsealed` refusal (the directory stood, so `refuse` re-advertised the
    pointer it had just called red) and wrote ``published`` BEFORE it sealed on the adopt path (so an adoption whose seal died on the
    transport left a published pin although the adoption had failed); and `admit_generation` read MANIFEST, VERIFIED and DEST-STAT and
    admitted a verified-but-WRITABLE replacement -- exactly what a same-sha8 repair whose seal failed leaves, its predecessor's SEALED
    record already moved aside with its reports. A window could therefore pin that generation and a shard read it while the mirror's own
    outcome said the publication was incomplete. Now ``published`` is dropped the moment a `current` is read without its own seal and
    written only after one succeeds, and the admission refuses an unsealed generation by name."""
    w = world
    slug = "relu_paired_run_state"
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r1.outcome == "published", r1.as_dict()
    gd, sha8 = w["gen_dir"], r1.content_sha8
    gen_path = f"{gd}/{sha8}"
    # the plain, sealed case: admitted, and the window pins it
    assert ST.admit_generation(gd, sha8).ok and _window_pins(w) == {slug: sha8}
    assert all(m == ST.DIR_SEAL_MODE for m in _dir_modes(gen_path).values())
    # (a) the shape a repair whose seal failed leaves: the seal gone (it went aside with the tree it spoke for) and the directories
    # writable. The record is read FIRST, so a missing seal is named even over sealed directories
    os.rename(f"{gd}/SEALED-{sha8}.json", f"{gd}/SEALED-{sha8}.json.aside")
    _set_dir_modes(gen_path, 0o755)
    ad = ST.admit_generation(gd, sha8)
    assert not ad.ok and ad.refusal == "unsealed" and f"no SEALED-{sha8}.json" in ad.reason, ad.as_dict()
    os.rename(f"{gd}/SEALED-{sha8}.json.aside", f"{gd}/SEALED-{sha8}.json")
    ad = ST.admit_generation(gd, sha8)
    assert not ad.ok and ad.refusal == "unsealed_directories" and f"{ST.DIR_SEAL_MODE:04o}" in ad.reason, ad.as_dict()
    rec = json.load(open(f"{gd}/SEALED-{sha8}.json"))
    _write(f"{gd}/SEALED-{sha8}.json", json.dumps({**rec, "content_sha8": "deadbeef"}))
    assert ST.admit_generation(gd, sha8).refusal == "sealed_mismatch", "a seal of other content is not this generation's"
    _write(f"{gd}/SEALED-{sha8}.json", json.dumps(rec))
    _set_dir_modes(gen_path, ST.DIR_SEAL_MODE)
    assert ST.admit_generation(gd, sha8).ok, "the record and the mode restored: admitted again"
    # (b) a plain tick over that shape: red by name, and NOTHING advertised although the directory stands
    os.rename(f"{gd}/SEALED-{sha8}.json", f"{gd}/SEALED-{sha8}.json.aside")
    _set_dir_modes(gen_path, 0o755)
    rn2 = Runner()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2)
    assert r2.outcome == "current_unsealed" and r2.level == "red" and not r2.ok and r2.published is None, r2.as_dict()
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] is None and st["generation"]["state"] == "current_unsealed" and _window_pins(w) == {}
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn2.calls), "nothing copied over a refusal"
    # (c) the adoption that must seal, whose seal dies on the transport: an error with nothing published and no pin -- before this the
    # state had already called the generation published, and the window went on pinning it
    rn3 = _SealBlip()
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn3, adopt_unrecorded=True)
    assert r3.outcome == "error" and not r3.ok and r3.published is None and "seal" in r3.detail["error"] and "rc 255" in r3.detail["error"], r3.as_dict()
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] is None and st["last_run"]["outcome"] == "error" and _window_pins(w) == {}
    assert not os.path.exists(f"{gd}/SEALED-{sha8}.json") and any(m != ST.DIR_SEAL_MODE for m in _dir_modes(gen_path).values())
    assert not ST.admit_generation(gd, sha8).ok and rn3.blips == 1
    # (d) the same adoption with the transport back: the seal stands first, and only then is the generation advertised and pinned
    rn4 = Runner()
    r4 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn4, adopt_unrecorded=True)
    assert r4.outcome == "unchanged" and r4.ok and r4.published == sha8 and set(r4.detail["adopted"]) == {"sealed"}, r4.as_dict()
    assert all(m == ST.DIR_SEAL_MODE for m in _dir_modes(gen_path).values())
    assert json.load(open(f"{gd}/SEALED-{sha8}.json"))["content_sha8"] == sha8
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] == sha8 and st["generation"]["state"] == "published"
    assert ST.admit_generation(gd, sha8).ok and _window_pins(w) == {slug: sha8}


def test_await_report_accepts_only_this_jobs_report_and_bounds_the_wait():
    """The helper: absent, unreadable (not JSON, a transport failure) and STALE (another job's) reads are re-read every poll
    until the bound; the first read carrying this job's id returns at once; nothing inside the bound returns `{}` and the
    last reading."""
    fresh = {"schema": ST.VERIFIED_SCHEMA, "ok": True, "job_id": "801"}
    stale = {"schema": ST.VERIFIED_SCHEMA, "ok": True, "job_id": "701"}

    class _Remote:
        def __init__(self, readings):
            self.readings = list(readings)
            self.reads = 0

        def read_json(self, path):
            self.reads += 1
            r = self.readings.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

    sleeps, logs = [], []
    rep, wait = ST._await_report(_Remote([{}, stale, ST.StoresError("x is not JSON"), fresh]), "/d/VERIFIED-x.json", "801", 120.0, 15.0, sleeps.append, logs.append)
    assert rep == fresh and wait == {"attempts": 4, "waited_s": 45.0, "last": "present"} and sleeps == [15.0] * 3
    assert len(logs) == 3 and "absent" in logs[0] and "stale" in logs[1] and "701" in logs[1] and "unreadable" in logs[2] and "not JSON" in logs[2]
    rep, wait = ST._await_report(_Remote([stale] * 9), "/d/VERIFIED-x.json", "801", 120.0, 15.0, sleeps.append, logs.append)
    assert rep == {} and wait["attempts"] == 9 and wait["waited_s"] == 120.0 and wait["last"].startswith("stale (written by job '701', not '801')")
    rep, wait = ST._await_report(_Remote([ST.StoresError("rc 255")] * 3), "/d/VERIFIED-x.json", "801", 20.0, 10.0, sleeps.append, logs.append)
    assert rep == {} and wait["attempts"] == 3 and wait["waited_s"] == 20.0 and wait["last"].startswith("unreadable: rc 255")
    rep, wait = ST._await_report(_Remote([fresh]), "/d/VERIFIED-x.json", "801", 0.0, 15.0, sleeps.append, logs.append)
    assert rep == fresh and wait == {"attempts": 1, "waited_s": 0.0, "last": "present"}
    rep, wait = ST._await_report(_Remote([{}]), "/d/VERIFIED-x.json", "801", 0.0, 15.0, sleeps.append, logs.append)
    assert rep == {} and wait == {"attempts": 1, "waited_s": 0.0, "last": "absent"}, "a zero bound reads exactly once"
    # the job id is compared as text (sbatch --parsable prints digits; the sbatch passes $SLURM_JOB_ID)
    rep, _wait = ST._await_report(_Remote([{**fresh, "job_id": 801}]), "/d/VERIFIED-x.json", "801", 0.0, 15.0, sleeps.append, logs.append)
    assert rep


def test_status_probe_names_a_dangling_current_as_a_problem_line(world, monkeypatch, capsys):
    w = world
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "status", "--json", "--probe"]) == 0
    probe = json.loads(capsys.readouterr().out)["components"]["relu_paired_run_state"]["probe"]
    assert probe == {"current": sha8, "gen_dir": gd, "current_tree_present": True, "problem": None}
    assert cli.main(["stores", "status", "--probe"]) == 0
    assert "PROBLEM" not in capsys.readouterr().out
    _renamed_away(gd, sha8)
    assert cli.main(["stores", "status", "--json", "--probe"]) == 0
    probe = json.loads(capsys.readouterr().out)["components"]["relu_paired_run_state"]["probe"]
    assert probe["current"] == sha8 and probe["current_tree_present"] is False
    assert probe["problem"] == f"current -> {sha8} but {gd}/{sha8} is not a directory (dangling; the next tick copies afresh)"
    assert cli.main(["stores", "status", "--probe"]) == 0
    text = capsys.readouterr().out
    assert f"PROBLEM relu_paired_run_state: current -> {sha8} but {gd}/{sha8} is not a directory" in text
    # no `current` at all: no tree to check, no problem line
    os.remove(f"{gd}/current")
    assert cli.main(["stores", "status", "--json", "--probe"]) == 0
    probe = json.loads(capsys.readouterr().out)["components"]["relu_paired_run_state"]["probe"]
    assert probe == {"current": None, "gen_dir": gd, "current_tree_present": None, "problem": None}
    # without --probe nothing is asked of the destination
    assert cli.main(["stores", "status", "--json"]) == 0
    assert "probe" not in json.loads(capsys.readouterr().out)["components"]["relu_paired_run_state"]
    # a transport failure probing the tree is loud, never a quiet "absent" (the same rule as `current`)
    rn = Runner()
    real = rn.__call__

    def blip(argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[:2] == ["sh", "-c"] and "test -d" in argv[2]:
            return subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: connect to host login-0 port 22: Connection timed out")
        return real(argv, timeout_s, input=input, check=check)

    with pytest.raises(ST.StoresError, match="test -d .*rc 255"):
        ST.Remote(w["dest"], blip).isdir(f"{gd}/{sha8}")
    assert ST.Remote(w["dest"], Runner()).isdir(gd) is True and ST.Remote(w["dest"], Runner()).isdir(f"{gd}/{sha8}") is False
    assert ST.Remote(w["dest"], Runner()).isdir(f"{gd}/VERIFIED-{sha8}.json") is False, "a file is not a directory"


def _vocab():
    spec = importlib.util.spec_from_file_location("test_registry_vocabulary", os.path.join(HERE, "test_registry_vocabulary.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_new_surfaces_keep_the_vocabulary():
    v = _vocab()
    # pre-existing names the declarations must spell -- directories the adapters open, two registered adapter names, the controls index
    # file (relu_paired.CONTROLS_INDEX_REL) -- are data, exempt by exact spelling; a new name of ours is not
    data_spellings = ("petri_arm_views", "si-steering-wave-objects", "llama8b/wave/stage", "cap4096_lane", "llama8b_wave", "final_rung.json",
                      "petri-wave")       # seedrep_petri.PETRIWAVE_STATE_ROOT: the producer's directory the petriwave_state component holds
    for path in (YAML_LIVE, SBATCH, ATTEST_SBATCH, os.path.join(UNITS, "registry-store-mirror.service.in"), os.path.join(UNITS, "registry-store-mirror.timer.in"),
                 os.path.join(UNITS, "registry-store-mirror-full.service.in"), os.path.join(UNITS, "registry-store-mirror-full.timer.in"), ST.__file__):
        text = open(path, encoding="utf-8").read()
        for d in data_spellings:
            text = text.replace(d, " ")
        h = v.hits(text, allow_prose_view=True)
        assert not h, (path, h[:10])
    # American spelling on the new surfaces
    import re
    for path in (YAML_LIVE, SBATCH, ATTEST_SBATCH, ST.__file__, os.path.join(UNITS, "registry-store-mirror.service.in"), os.path.join(UNITS, "registry-store-mirror-full.service.in")):
        text = open(path, encoding="utf-8").read()
        assert not re.search(r"\b(behaviour|colour|honour|serialis|initialis|canonicalis|organis|synchronis)\w*", text), path


def test_store_mirror_unit_is_a_bounded_secret_free_oneshot_on_the_agent_socket():
    cp = configparser.ConfigParser(strict=False, allow_no_value=True, interpolation=None, comment_prefixes=("#",), inline_comment_prefixes=None)
    cp.optionxform = str
    cp.read_string(open(os.path.join(UNITS, "registry-store-mirror.service.in")).read())
    s = cp["Service"]
    assert s["Type"] == "oneshot" and s["Restart"] == "no" and s["Nice"] == "10" and s["IOSchedulingClass"] == "idle" and s["TimeoutStartSec"] == "5h"
    assert "stores mirror --dest @STORE_DEST@" in s["ExecStart"] and "eval " not in s["ExecStart"]   # the primary cluster's registry label, filled at install
    text = open(os.path.join(UNITS, "registry-store-mirror.service.in")).read()
    assert "Environment=SSH_AUTH_SOCK=%h/.ssh/agent.sock" in text and "harvester.lock" not in text.split("[Service]")[1]
    t = configparser.ConfigParser(strict=False, allow_no_value=True, interpolation=None, comment_prefixes=("#",), inline_comment_prefixes=None)
    t.optionxform = str
    t.read_string(open(os.path.join(UNITS, "registry-store-mirror.timer.in")).read())
    assert t["Timer"]["OnUnitActiveSec"] == "30min" and t["Timer"]["Unit"] == "registry-store-mirror.service"


# --------------------------------------------------------------------------------------------
# R2b precert (2026-10-06): three declared bases (hpo_rehpo, hpo_sync, stage_objects) each hold ONE symlink named after the base whose
# target is the base's own absolute path (`<name> -> $HOME//<name>`). link_escapes refuses every absolute target, so `stores push`
# refused all three at the manifest. The fix is declarative: an anchored exclude naming the link, in the component's own entry.
# --------------------------------------------------------------------------------------------


def test_a_self_referential_absolute_link_is_refused_until_its_anchored_exclude_names_it(tmp_path):
    """A base whose only symlink points at the base itself by absolute path: the manifest refuses it by name (the copy would resolve
    it to bytes nobody attested); with the anchored exclude ``/<link name>`` the walk, the manifest and the rsync all skip it, no
    symlink is attested, and every regular file is still carried."""
    base = tmp_path / "selfref"
    _write(str(base / "a" / "f1.txt"), b"one")
    _write(str(base / "f2.txt"), b"two")
    os.symlink(str(base) + "/", str(base / "selfref"))                       # `selfref -> /abs/path/selfref/`: resolves to the base itself
    assert os.path.isdir(base / "selfref" / "a"), "the link resolves (a walk that followed it would loop)"
    refused = ST.Component("c", str(base), "grading_host_copy")
    with pytest.raises(ST.ManifestRefusal, match="selfref -> .* escapes the component base"):
        ST.build_manifest(refused, None)
    comp = ST.Component("c", str(base), "grading_host_copy", exclude=["/selfref"])
    walk = ST.walk_component(comp)
    assert [rel for rel, _p, _s in walk.files] == ["a/f1.txt", "f2.txt"] and walk.links == []
    man = ST.build_manifest(comp, None)
    assert man.n_files == 2 and man.links == [] and man.meta({})["symlinks"] == {} and man.meta({})["exclude"] == ["/selfref"]
    # the copy excludes the same path, anchored (never a basename pattern: a nested entry of that name is still carried)
    argv = ST.rsync_argv(ST.Destination("d", None, str(tmp_path / "dest")), comp, str(tmp_path / "dest" / "gen"), None)
    assert "--exclude=/selfref" in argv and not any(a == "--exclude=selfref" for a in argv)
    # a name pattern would also drop a nested regular entry of that name; the anchored form keeps it
    _write(str(base / "a" / "selfref"), b"nested regular file")
    assert [rel for rel, _p, _s in ST.walk_component(comp).files] == ["a/f1.txt", "a/selfref", "f2.txt"]
    assert ST.walk_component(ST.Component("c", str(base), "grading_host_copy", exclude=["selfref"])).files[1][0] == "f2.txt"


def test_live_config_excludes_the_self_referential_links_and_the_link_only_shims_the_move_refused():
    """The five bases whose one symlink is named after the base and points at it by absolute path (hpo_rehpo, hpo_sync, stage_objects
    from the R2b precert; hpo_llama, hpo_r1 from the R2b run, whose pushes refused) declare the anchored exclude of their own name; the
    two link-only shims the run refused -- grade_strength's `<family>/_local_root/` (32 absolute links, no regular file; no adapter
    opens it) and full22_seedrep's `seedrep_battery/sampled/transcripts_sampled/` (192 absolute links onto attested stores, no regular
    file; seedrep_statics reads the stores by their own path) -- are excluded by name pattern and by anchored path respectively."""
    cfg = ST.load_config(YAML_LIVE)
    for slug in ("hpo_rehpo", "hpo_sync", "stage_objects", "hpo_llama", "hpo_r1"):
        c = cfg.components[slug]
        name = os.path.basename(os.path.expanduser(c.base).rstrip("/"))
        assert f"/{name}" in c.exclude, (slug, c.exclude)
        assert name in c.anchored_excludes() and name not in c.name_excludes(), (slug, c.exclude)
    gs = cfg.components["grade_strength"]
    assert "_local_root" in gs.name_excludes() and not gs.anchored_excludes(), gs.exclude
    fs = cfg.components["full22_seedrep"]
    assert "seedrep_battery/sampled/transcripts_sampled" in fs.anchored_excludes() and "seedrep_battery" in fs.members, fs.exclude
    # wandb_mirror: the run directories' logs/ by name AND the two root-level W&B convenience links whose targets lie inside those
    # excluded directories -- attested dangling, they refused wandb_runs by name every window -- anchored, so a nested entry of
    # either name is still carried
    wm = cfg.components["wandb_mirror"]
    assert wm.name_excludes() == ["logs"] and wm.anchored_excludes() == {"debug.log", "debug-internal.log"}, wm.exclude
    argv = ST.rsync_argv(ST.Destination("d", None, "/dest"), wm, "/dest/gen", None)
    assert "--exclude=/debug.log" in argv and "--exclude=/debug-internal.log" in argv and "--exclude=logs" in argv


def test_escaping_links_lists_every_link_the_manifest_would_refuse_without_hashing(tmp_path):
    """The census the precert runs is the push's own test (link_escapes) over the component walk: a base with a self-referential
    absolute link is listed (the shape five live bases had), so is a climbing relative link; an in-tree relative link is not; an
    anchored exclude of the link removes it from the census exactly as it unblocks the manifest; nothing is hashed."""
    base = tmp_path / "census"
    _write(str(base / "a" / "f1.txt"), b"one")
    _write(str(base / "deep" / "x" / "f2.txt"), b"two")
    os.symlink(str(base) + "/", str(base / "census"))                        # self-referential, absolute
    os.symlink("../../../outside", str(base / "deep" / "x" / "climb"))       # climbs above the base (../../ alone lands on <base>/outside)
    os.symlink("a/f1.txt", str(base / "intree"))                             # relative, in tree: attested, never refused
    comp = ST.Component("c", str(base), "grading_host_copy")
    assert ST.escaping_links(comp) == [("census", str(base) + "/"), ("deep/x/climb", "../../../outside")]
    with pytest.raises(ST.ManifestRefusal, match="escapes the component base"):
        ST.build_manifest(comp, None)
    fixed = ST.Component("c", str(base), "grading_host_copy", exclude=["/census", "/deep/x/climb"])
    assert ST.escaping_links(fixed) == []
    man = ST.build_manifest(fixed, None)
    assert man.n_files == 2 and dict(man.links) == {"intree": "a/f1.txt"}
    # the census over a base with no link at all, and over the declared members only
    assert ST.escaping_links(ST.Component("c", str(base), "grading_host_copy", members=["a"])) == []


# --------------------------------------------------------------------------------------------
# the manifest attests directory listings
# --------------------------------------------------------------------------------------------

def _listing_json(entries):
    return [[n, k] for n, k in entries]


def test_the_manifest_attests_every_directorys_listing_empty_ones_included_and_the_generation_reproduces_and_verifies_them(world):
    """The additive `directories` field: every directory the walk visits, keyed by relpath ('' the base), with the CANONICAL directory's whole
    listing as sorted (name, kind) pairs -- the excluded caches named in the base's listing, the symlink as `link`, an empty directory as []
    -- plus `n_directories` / `n_empty_directories` / `listings_sha256`; the two content hashes do not move with it. The published generation
    carries the empty directory, its VERIFIED report verified the listings (the staged expectation: the attestation less excluded entries),
    `manifest_index` exposes the attestation, and the grading host's manifest copy carries the same field."""
    from sisteer.registry import listing as L
    w = world
    os.makedirs(os.path.join(w["base"], "x", "killed_empty"))
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res.outcome == "published", res.as_dict()
    sha8, gd = res.content_sha8, w["gen_dir"]
    meta = json.load(open(f"{gd}/MANIFEST-{sha8}.json"))
    dirs = meta["directories"]
    assert set(dirs) == {"", "a", "a/b", "x", "x/deep", "x/deep/er", "x/killed_empty"}, sorted(dirs)       # the excluded caches are not walked
    assert dirs[""] == _listing_json([(".pytest_cache", "dir"), ("__pycache__", "dir"), ("a", "dir"), ("link_to_top", "link"), ("top.txt", "file"), ("x", "dir")])
    assert dirs["x/killed_empty"] == [] and dirs["x"] == [["deep", "dir"], ["killed_empty", "dir"]]
    assert meta["n_directories"] == 7 and meta["n_empty_directories"] == 1 and meta["listings_sha256"] == L.listings_sha256({k: v for k, v in dirs.items()})
    assert "directories" in meta["manifest_rules"]
    # the hash pair does not see the listings: the same files hash the same whether or not a directory is attested
    pre = ST.build_manifest(w["comp"], None)
    assert (pre.content_sha256, pre.name_size_sha256) == (meta["content_sha256"], meta["name_size_sha256"]) and pre.content_sha8 == sha8
    # the generation reproduces the empty directory and the verify report verified the listings
    assert os.path.isdir(f"{gd}/{sha8}/x/killed_empty") and os.listdir(f"{gd}/{sha8}/x/killed_empty") == []
    ver = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert ver["ok"] is True and ver["listings_attested"] is True and ver["n_directories"] == 7 and ver["listing_mismatch"] == [] and ver["listings_sha256"] == meta["listings_sha256"]
    # the index on the node exposes the attestation per directory; the fingerprint is the readers' (one definition)
    ix = ST.manifest_index(gd, sha8)
    assert ix.listings_attested and ix.attested_listing("x/killed_empty") == () and ix.attested_listing("a/b") == (("f1.json", "file"),) and ix.attested_listing("nope") is None
    assert L.listing_fingerprint(ix.attested_listing("x")) == L.listing_fingerprint(L.listing_entries(os.path.join(w["base"], "x")))
    # the staged expectation of the base: the attestation less the excluded caches, which the copy leaves out
    assert ST.staged_listing("", ix.attested_listing(""), ST.exclusion_test(w["comp"])) == (("a", "dir"), ("link_to_top", "link"), ("top.txt", "file"), ("x", "dir"))
    # the grading host's own manifest copy carries the field; admission on the node holds
    local = json.load(open(os.path.join(ST.component_state_dir(w["root"], "relu_paired_run_state"), f"MANIFEST-{sha8}.json")))
    assert local["directories"] == dirs and local["listings_sha256"] == meta["listings_sha256"]
    ad = ST.admit_generation(gd, sha8)
    assert ad.ok and ad.n_directories == 7 and ad.listing_mismatched == [], ad.as_dict()


def test_sols_scenario_tracked_bytes_change_as_an_unstaged_empty_directory_appears_the_next_generation_attests_and_carries_it(world):
    """The trigger of the sixth review: after a published generation the canonical gains an empty ignored directory AND a tracked file
    changes. The next generation's manifest attests the directory, the copy reproduces it (rsync -r; the verify job would create an
    attested empty directory a copy left out), its VERIFIED report verified the listings, and the attested listing fingerprints exactly
    as a reader's listing input of the canonical directory does -- so a shard comparing its listing with the attestation sees the
    directory the stale previous revision never had. A stage that lacks an attested entry a copy must carry is a verify FAILURE: nothing
    is published."""
    from sisteer.registry import listing as L
    from sisteer.registry.adapters import base as AB
    w = world
    res1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res1.outcome == "published"
    gd = w["gen_dir"]
    assert json.load(open(f"{gd}/MANIFEST-{res1.content_sha8}.json"))["directories"]["a/b"] == [["f1.json", "file"]]
    # the canonical moves: a tracked file changes and an empty, never-read directory lands beside it
    _write(os.path.join(w["base"], "a", "b", "f1.json"), b'{"k": 2}\n')
    os.makedirs(os.path.join(w["base"], "a", "b", "killed_alphasteer"))
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res2.outcome == "published" and res2.content_sha8 != res1.content_sha8, res2.as_dict()
    sha8 = res2.content_sha8
    meta = json.load(open(f"{gd}/MANIFEST-{sha8}.json"))
    assert meta["directories"]["a/b"] == [["f1.json", "file"], ["killed_alphasteer", "dir"]] and meta["directories"]["a/b/killed_alphasteer"] == []
    assert os.path.isdir(f"{gd}/{sha8}/a/b/killed_alphasteer")
    ver = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert ver["ok"] is True and ver["listing_mismatch"] == [] and ver["created_dirs"] == []
    # one definition: the attested listing IS the reader's listing input of the canonical directory
    li = AB.ListingInput.of_directory("em_misalign:qwen35_4b", "a/b", os.path.join(w["base"], "a", "b"))
    assert L.listing_fingerprint(ST.manifest_index(gd, sha8).attested_listing("a/b")) == li.sha256 and li.path == os.path.join(w["base"], "a", "b")
    # the verify job itself creates an attested EMPTY directory the copy left out, and a missing attested NON-empty entry is a failure
    # (the owner un-seals the directories first: C3's directory seal, 0555 after the verify, refuses the edits below otherwise)
    os.chmod(f"{gd}/{sha8}/a/b", 0o755)
    os.chmod(f"{gd}/{sha8}", 0o755)
    os.rmdir(f"{gd}/{sha8}/a/b/killed_alphasteer")
    rep, _ = ST.verify_generation(f"{gd}/{sha8}", f"{gd}/MANIFEST-{sha8}.sha256", workers=1, seal=False)
    assert rep["ok"] is True and rep["created_dirs"] == ["a/b/killed_alphasteer"] and os.path.isdir(f"{gd}/{sha8}/a/b/killed_alphasteer")
    os.chmod(f"{gd}/{sha8}/top.txt", 0o644)
    os.remove(f"{gd}/{sha8}/top.txt")
    rep2, _ = ST.verify_generation(f"{gd}/{sha8}", f"{gd}/MANIFEST-{sha8}.sha256", workers=1, seal=False)
    assert rep2["ok"] is False and rep2["missing"] == ["top.txt"] and rep2["listing_mismatch"] == [""], rep2
    _write(f"{gd}/{sha8}/top.txt", b"top level\n")


def test_an_empty_directory_removed_after_verification_refuses_admission_and_a_listing_only_canonical_change_is_recorded_and_refused_by_revalidation(world):
    """After the publish: the generation's attested empty directory is removed on the node -> `admit_generation` refuses `listing_changed`
    (the file walk sees nothing missing); restored -> admitted. The canonical gains ANOTHER empty directory with no file change: the mirror
    cannot publish a new generation under the same content name and records `listings_changed`; `revalidate` refuses the pinned generation
    `listing_changed`, so the readers that state listings take the grading host route."""
    w = world
    os.makedirs(os.path.join(w["base"], "x", "killed_empty"))
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res.outcome == "published"
    sha8, gd = res.content_sha8, w["gen_dir"]
    assert ST.revalidate("relu_paired_run_state", sha8, root=w["root"], cfg=w["cfg"]).ok
    os.chmod(f"{gd}/{sha8}/x", 0o755)          # the owner un-seals the parent (C3's directory seal) to stage the removal
    os.rmdir(f"{gd}/{sha8}/x/killed_empty")
    ad = ST.admit_generation(gd, sha8)
    assert not ad.ok and ad.refusal == "listing_changed" and ad.listing_mismatched == ["x", "x/killed_empty"], ad.as_dict()
    os.makedirs(f"{gd}/{sha8}/x/killed_empty")
    assert ST.admit_generation(gd, sha8).ok
    # a listing-only change of the canonical
    os.makedirs(os.path.join(w["base"], "a", "later_empty"))
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res2.outcome == "listings_changed" and res2.published == sha8 and res2.content_sha8 == sha8, res2.as_dict()
    assert res2.level == "red" and not res2.ok, "a listing moved from an ATTESTED one is red (the unattested case is listings_unattested, amber)"
    assert res2.detail["n_empty_directories"] == 2 and res2.detail["listings_sha256"] != res2.detail["published_listings_sha256"]
    assert ST.load_state(w["root"])["components"]["relu_paired_run_state"]["generation"]["state"] == "listings_changed"
    rv = ST.revalidate("relu_paired_run_state", sha8, root=w["root"], cfg=w["cfg"])
    assert not rv.ok and rv.refusal == "listing_changed" and rv.listing_mismatched == ["a", "a/later_empty"], rv.as_dict()
    # the directory gone again: the canonical is the generation's once more
    os.rmdir(os.path.join(w["base"], "a", "later_empty"))
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner()).outcome == "unchanged"
    assert ST.revalidate("relu_paired_run_state", sha8, root=w["root"], cfg=w["cfg"]).ok


def _pre_rule_generation(w, sha8, slug="relu_paired_run_state"):
    """Rewrite a published generation's evidence into the shape a tree WITHOUT the 2026-10-04 directory-listings rule left behind (the
    destination census of 2026-10-07: 42 of 54 generations, the R2b move branch): the VERIFIED report carries no listings key at all,
    the MANIFEST .json no `directories` (five `manifest_rules`), no keyed MANIFEST-<sha8>.<lsha8>.json on either side, and the state
    records no attestation. Returns the listings_sha256 the generation WAS verified against (what a re-attestation recomputes while
    the canonical's directories stand)."""
    gd, sdir = w["gen_dir"], ST.component_state_dir(w["root"], slug)
    ver = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    lsha = ver["listings_sha256"]
    for k in ("listings_sha256", "listings_attested", "n_directories", "listing_mismatch", "created_dirs", "attestation",
              "dirs_generation_only", "n_dirs_generation_only", "dirs_canonical_only", "n_dirs_canonical_only"):
        ver.pop(k, None)
    ST._write_json(f"{gd}/VERIFIED-{sha8}.json", ver)
    for jpath in (f"{gd}/MANIFEST-{sha8}.json", f"{sdir}/MANIFEST-{sha8}.json"):
        meta = json.load(open(jpath))
        for k in ("directories", "n_directories", "n_empty_directories", "listings_sha256"):
            meta.pop(k, None)
        meta["manifest_rules"] = {k: v for k, v in meta["manifest_rules"].items() if k != "directories"}
        assert len(meta["manifest_rules"]) == 5
        ST._write_json(jpath, meta)
        os.remove(ST.attestation_json_path(jpath[:-len(".json")] + ".sha256", lsha))
    st = ST.load_state(w["root"])
    rec = st["components"][slug]
    rec["published_attestation"] = None
    rec["generation"].pop("listings_sha256", None)
    ST._write_json(ST.state_path(w["root"]), st)
    return lsha


def test_a_generation_verified_before_the_listings_rule_is_listings_unattested_amber_never_listings_changed(world, monkeypatch, capsys):
    """The live misread of 2026-10-05..07: 35 of 54 components red as `listings_changed` with the content sha equal to the published one,
    nothing hashed and `published_listings_sha256: null` -- the compare read `None == <digest>` for a generation whose VERIFIED report
    attests no listing (a tree without the directory-listings rule verified it) and called the canonical's listings "moved". Such a
    generation is `listings_unattested`: amber, `ok`, the mirror exits 0, `published` is the generation, `published_attestation` stays
    None (the pin names no attestation, so `revalidate` and the node's admission check no listing for it), and the summary names the
    component and the `stores attest` command. A listing moved from an ATTESTED one is still `listings_changed`, red."""
    w = world
    slug = "relu_paired_run_state"
    os.makedirs(os.path.join(w["base"], "x", "killed_empty"))
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res.outcome == "published", res.as_dict()
    sha8, gd = res.content_sha8, w["gen_dir"]
    lsha = _pre_rule_generation(w, sha8)
    assert ST.revalidate(slug, sha8, root=w["root"], cfg=w["cfg"]).ok and ST.admit_generation(gd, sha8).ok
    live = open(f"{gd}/VERIFIED-{sha8}.json", "rb").read()
    rn = Runner()
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert res2.outcome == "listings_unattested" and res2.level == "amber" and res2.ok and res2.published == sha8 and res2.content_sha8 == sha8, res2.as_dict()
    assert res2.detail["listings_sha256"] == lsha and res2.detail["published_listings_sha256"] is None and res2.detail["n_directories"] == 7
    assert "verified before its directory listings were attested" in res2.detail["why"] and "stores attest" in res2.detail["why"]
    assert res2.detail["attest"] == f"registry stores attest --dest cluster_a --component {slug}"
    assert rn.count("sbatch") == 0 and open(f"{gd}/VERIFIED-{sha8}.json", "rb").read() == live, "nothing submitted, nothing rewritten"
    st = ST.load_state(w["root"])["components"][slug]
    assert st["generation"]["state"] == "listings_unattested" and st["generation"]["listings_sha256"] == lsha and st["published"] == sha8
    assert st["published_attestation"] is None and st["last_run"]["outcome"] == "listings_unattested"
    assert ST.published_attestations(w["root"]) == {slug: None}
    assert ST.last_run_level("listings_unattested") == "amber" and ST.status(w["root"], w["cfg"])["components"][slug]["level"] == "amber"
    assert "listings_unattested" in ST.PUSH_OK_OUTCOMES, "a producer's push of unchanged content is not failed by the operator's missing attest"
    # the window's checks: the pin names no attestation, so nothing refuses the component over a listing it never attested
    assert ST.revalidate(slug, sha8, root=w["root"], cfg=w["cfg"]).ok and ST.admit_generation(gd, sha8).ok
    assert ST.adapter_admission(w["cfg"], "relu_paired", {slug: sha8}, root=w["root"], attestations={slug: None})[slug].ok
    # the CLI: exit 0, the summary names the component and the command, the log says what to run
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "mirror"]) == 0
    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert json.loads(lines[0])["outcome"] == "listings_unattested" and json.loads(lines[0])["level"] == "amber"
    summary = json.loads(lines[-1])
    assert summary["listings_unattested"] == [slug] and summary["attest"] == "registry stores attest --dest cluster_a --all-unattested"
    assert summary["ok"] == 1 and summary["levels"] == {"green": 0, "amber": 1, "red": 0} and summary["listing_only_drift"] == []
    assert "unattested directory listings" in out.err and "stores attest --dest cluster_a --all-unattested" in out.err
    # a real move from an ATTESTED listing is still red: a file change publishes a fresh generation that attests the listings, then an empty
    # directory lands with no file change
    _write(os.path.join(w["base"], "top.txt"), b"top level, revised\n")
    res3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res3.outcome == "published" and res3.content_sha8 != sha8
    L3 = json.load(open(f"{gd}/VERIFIED-{res3.content_sha8}.json"))["listings_sha256"]
    assert ST.load_state(w["root"])["components"][slug]["published_attestation"] == L3
    os.makedirs(os.path.join(w["base"], "a", "later_empty"))
    res4 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res4.outcome == "listings_changed" and res4.level == "red" and not res4.ok and res4.published == res3.content_sha8, res4.as_dict()
    assert res4.detail["published_listings_sha256"] == L3 and res4.detail["listings_sha256"] != L3 and "stores attest" in res4.detail["why"]
    assert ST.load_state(w["root"])["components"][slug]["published_attestation"] == L3, "the pin keeps the attested listings; revalidate refuses the moved ones"
    assert ST.revalidate(slug, res3.content_sha8, root=w["root"], cfg=w["cfg"], listings_sha256=L3).refusal == "listing_changed"
    assert cli.main(["stores", "mirror"]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["listings_unattested"] == []


def test_a_manifest_written_before_the_directories_field_attests_no_listing_and_still_verifies_and_admits(world):
    """Older manifests stay readable: a `.json` without `directories` reads as attesting NO listing (`read_manifest_directories` None, never
    {}), the index says so, the verify report says `listings_attested` false, and admission / `_verified_matches` ignore listings for it."""
    w = world
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert res.outcome == "published"
    sha8, gd = res.content_sha8, w["gen_dir"]
    jpath = f"{gd}/MANIFEST-{sha8}.json"
    meta = json.load(open(jpath))
    for k in ("directories", "n_directories", "n_empty_directories", "listings_sha256"):
        meta.pop(k)
    ST._write_json(jpath, meta)
    assert ST.read_manifest_directories(f"{gd}/MANIFEST-{sha8}.sha256") is None
    ix = ST.manifest_index(gd, sha8)
    assert not ix.listings_attested and ix.attested_listing("a") is None
    rep, _ = ST.verify_generation(f"{gd}/{sha8}", f"{gd}/MANIFEST-{sha8}.sha256", workers=1, seal=False)
    assert rep["ok"] is True and rep["listings_attested"] is False and rep["n_directories"] is None and rep["listings_sha256"] is None
    ad = ST.admit_generation(gd, sha8)
    assert ad.ok and ad.n_directories is None
    pre = ST.build_manifest(w["comp"], None)
    assert ST._verified_matches(rep, pre, listings=False) and not ST._verified_matches(rep, pre), "a report that verified no listings does not admit a manifest that attests them"


def test_attestations_are_immutable_per_content_and_listings_and_every_pin_side_check_reads_the_attestation_the_pin_names(world):
    """The A -> B -> A reversion. Generation A is published over listings L1 (an empty directory present) and a window pins (A, L1). The
    content moves to B (a byte change) and back to A with the directory removed (L2). The rebuild of A ADDS the attestation
    MANIFEST-A.<L2>.json and rewrites nothing: the grading host's MANIFEST-A.json (the first attestation) and MANIFEST-A.<L1>.json stand
    byte for byte; the destination holds both keyed attestations; the re-published A tree is a FRESH copy the L2 attestation verified (the
    earlier tree renamed aside, never rewritten under a window reading it); the state records the attestation each publish verified. Then
    every pin-side check reads the attestation the pin names: revalidate(A, L1) refuses listing_changed (the canonical left L1),
    revalidate(A, L2) admits, an attestation never written refuses no_attestation, and with none named the FIRST attestation is read; the
    node's admit_generation and manifest_index behave the same against the fresh tree. A rebuild whose (content, listings) pair is attested
    already writes nothing. Before this change the rebuild overwrote MANIFEST-A.json with L2 and revalidate(A) accepted the canonical
    against listings the window never read."""
    w = world
    slug = "relu_paired_run_state"
    gd, sdir = w["gen_dir"], ST.component_state_dir(w["root"], slug)
    mpath = f"{sdir}/MANIFEST-"
    os.makedirs(os.path.join(w["base"], "x", "killed_empty"))
    resA = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert resA.outcome == "published", resA.as_dict()
    A = resA.content_sha8
    L1 = json.load(open(f"{gd}/VERIFIED-{A}.json"))["listings_sha256"]
    st = ST.load_state(w["root"])["components"][slug]
    assert st["published"] == A and st["published_attestation"] == L1 and st["generation"]["listings_sha256"] == L1
    assert ST.published_attestations(w["root"]) == {slug: L1}
    keyed1 = ST.attestation_json_path(f"{mpath}{A}.sha256", L1)
    assert keyed1 == f"{mpath}{A}.{L1[:8]}.json" and os.path.isfile(keyed1) and json.load(open(keyed1))["listings_sha256"] == L1
    assert st["latest_manifest"]["attestation"] == keyed1 and st["latest_manifest"]["listings_sha256"] == L1
    assert json.load(open(f"{gd}/MANIFEST-{A}.{L1[:8]}.json"))["listings_sha256"] == L1              # the destination's immutable copy
    plain_bytes, keyed1_bytes = open(f"{mpath}{A}.json", "rb").read(), open(keyed1, "rb").read()
    assert json.loads(plain_bytes)["listings_sha256"] == L1
    # B: a byte change, published
    _write(os.path.join(w["base"], "top.txt"), b"top level, revised\n")
    resB = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert resB.outcome == "published" and resB.content_sha8 != A
    # back to A's bytes with the directory removed: the same content, listings L2
    _write(os.path.join(w["base"], "top.txt"), b"top level\n")
    os.rmdir(os.path.join(w["base"], "x", "killed_empty"))
    resA2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert resA2.outcome == "published" and resA2.content_sha8 == A, resA2.as_dict()
    L2 = json.load(open(f"{gd}/VERIFIED-{A}.json"))["listings_sha256"]
    assert L2 != L1 and ST.build_manifest(w["comp"], None).listings_sha256 == L2
    # the earlier A tree stands aside, the published one is a fresh copy the L2 attestation verified
    superseded = resA2.detail["superseded"]
    assert superseded.startswith(f"{gd}/.superseded-{A}-") and os.path.isdir(f"{superseded}/x/killed_empty") and not os.path.exists(f"{gd}/{A}/x/killed_empty")
    assert os.path.realpath(f"{gd}/current") == os.path.realpath(f"{gd}/{A}")
    # nothing rewritten on the grading host: the first attestation and L1's stand byte for byte; L2's was added
    keyed2 = ST.attestation_json_path(f"{mpath}{A}.sha256", L2)
    assert open(f"{mpath}{A}.json", "rb").read() == plain_bytes and open(keyed1, "rb").read() == keyed1_bytes
    assert os.path.isfile(keyed2) and json.load(open(keyed2))["listings_sha256"] == L2
    assert {os.path.basename(p) for p in glob.glob(f"{mpath}{A}.*.json")} == {f"MANIFEST-{A}.{L1[:8]}.json", f"MANIFEST-{A}.{L2[:8]}.json", f"MANIFEST-{A}.stat.json"}
    # the destination holds both keyed attestations; its plain .json is the one the verify job ran against (L2)
    assert json.load(open(f"{gd}/MANIFEST-{A}.{L1[:8]}.json"))["listings_sha256"] == L1 and json.load(open(f"{gd}/MANIFEST-{A}.{L2[:8]}.json"))["listings_sha256"] == L2
    assert json.load(open(f"{gd}/MANIFEST-{A}.json"))["listings_sha256"] == L2
    st2 = ST.load_state(w["root"])["components"][slug]
    assert st2["published_attestation"] == L2 and ST.published_attestations(w["root"]) == {slug: L2}
    # every pin-side check reads the attestation the pin names
    rv1 = ST.revalidate(slug, A, root=w["root"], cfg=w["cfg"], listings_sha256=L1)
    assert not rv1.ok and rv1.refusal == "listing_changed" and rv1.listings_sha256 == L1 and rv1.listing_mismatched == ["x", "x/killed_empty"], rv1.as_dict()
    rv2 = ST.revalidate(slug, A, root=w["root"], cfg=w["cfg"], listings_sha256=L2)
    assert rv2.ok and rv2.listings_sha256 == L2 and rv2.n_directories == 6, rv2.as_dict()
    rv3 = ST.revalidate(slug, A, root=w["root"], cfg=w["cfg"], listings_sha256="0" * 64)
    assert rv3.refusal == "no_attestation" and "never wrote" in rv3.reason
    rv0 = ST.revalidate(slug, A, root=w["root"], cfg=w["cfg"])
    assert rv0.refusal == "listing_changed" and rv0.listings_sha256 == L1, "an older pin naming no attestation reads the FIRST, never the latest"
    assert ST.adapter_admission(w["cfg"], "relu_paired", {slug: A}, root=w["root"], attestations={slug: L1})[slug].refusal == "listing_changed"
    assert ST.adapter_admission(w["cfg"], "relu_paired", {slug: A}, root=w["root"], attestations={slug: L2})[slug].ok
    ad1 = ST.admit_generation(gd, A, listings_sha256=L1)
    assert not ad1.ok and ad1.refusal == "listing_changed" and ad1.listings_sha256 == L1 and ad1.listing_mismatched == ["x", "x/killed_empty"], ad1.as_dict()
    assert ST.admit_generation(gd, A, listings_sha256=L2).ok and ST.admit_generation(gd, A, listings_sha256="1" * 64).refusal == "no_attestation"
    assert ST.manifest_index(gd, A, listings_sha256=L1).attested_listing("x") == (("deep", "dir"), ("killed_empty", "dir"))
    assert ST.manifest_index(gd, A, listings_sha256=L2).attested_listing("x") == (("deep", "dir"),) and ST.manifest_index(gd, A, listings_sha256=L2).listings_sha256 == L2
    with pytest.raises(ST.StoresError, match="no_attestation"):
        ST.manifest_index(gd, A, listings_sha256="2" * 64)
    # a rebuild whose (content, listings) pair is attested already writes nothing
    before = {p: os.stat(p).st_mtime_ns for p in (f"{mpath}{A}.json", keyed1, keyed2)}
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner()).outcome == "unchanged"
    assert {p: os.stat(p).st_mtime_ns for p in before} == before
# --------------------------------------------------------------------------------------------
# the retry over a renamed-aside tree hard-links what it already copied
# --------------------------------------------------------------------------------------------


def test_a_tree_renamed_aside_is_the_link_dest_of_the_fresh_copy_so_the_retry_transfers_nothing_it_already_holds(world):
    """A tick's rsync dies after the whole tree landed (nothing published, no report); the next tick finds the tree standing from an
    earlier attempt at THIS content and renames it aside (`.superseded-<sha8>-<stamp>`) before copying afresh -- the full re-copy that
    cost misalign_grades 29.5 GB per tick. Now the renamed-aside tree is passed to rsync as a `--link-dest` beside the published
    generation (none here): every file whose size and mtime stand is hard-linked, not transferred, and the fresh verify is still the
    gate. The record names it (`link_dest_aside`); a tick with no renamed tree names none."""
    w = world
    slug = "relu_paired_run_state"
    gd = w["gen_dir"]
    # tick 1: the copy lands in full, then the transport dies before rsync reports -- a complete but unpublished, unverified tree
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=_RsyncCut(partial=True))
    sha8 = r1.content_sha8
    assert r1.outcome == "rsync_failed" and os.path.isdir(f"{gd}/{sha8}") and not os.path.exists(f"{gd}/VERIFIED-{sha8}.json") and not os.path.exists(f"{gd}/current")
    inodes_before = {rel: os.stat(os.path.join(gd, sha8, rel)).st_ino for rel in ST._walk_plain_dir(f"{gd}/{sha8}")}
    # tick 2: the standing tree is renamed aside and the fresh copy links against it
    lines = []
    rn = Runner()
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, log=lines.append)
    assert r2.outcome == "published" and r2.published == sha8, r2.as_dict()
    superseded = r2.detail["superseded"]
    assert superseded.startswith(f"{gd}/.superseded-{sha8}-") and os.path.isdir(superseded)
    assert r2.detail["link_dest"] is None and r2.detail["link_dest_aside"] == superseded
    rs = next(a for a in rn.calls if "rsync" in a and "--files-from=-" not in a)
    assert f"--link-dest={superseded}" in rs and not any(a.startswith("--link-dest=") and a != f"--link-dest={superseded}" for a in rs)
    assert r2.detail["rsync"]["files_transferred"] == 0 and r2.detail["rsync"]["bytes_transferred"] == 0, r2.detail["rsync"]
    # every file of the published tree IS the renamed tree's file (one inode, two names); the verify read the fresh tree all the same
    for rel in ST._walk_plain_dir(f"{gd}/{sha8}"):
        st_new, st_aside = os.stat(os.path.join(gd, sha8, rel)), os.stat(os.path.join(superseded, rel))
        assert st_new.st_ino == st_aside.st_ino == inodes_before[rel] and st_new.st_nlink == 2, rel
    assert json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["ok"] is True and os.readlink(f"{gd}/current") == sha8
    assert any(f"link-dest none; aside {superseded}" in ln for ln in lines), lines
    # a tick with nothing renamed aside names no aside link-dest (the published generation is the link-dest, as before)
    _write(os.path.join(w["base"], "top.txt"), b"top level, revised\n")
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r3.outcome == "published" and r3.detail["link_dest"] == f"{gd}/{sha8}" and r3.detail["link_dest_aside"] is None


# --------------------------------------------------------------------------------------------
# the controller owns its bound (`stores mirror --bound-s`; the hardening branch's review, P2): a verifier that outlives the bound is
# cancelled by id and confirmed terminal, and a recorded awaiting_verify job is settled before anything is copied or renamed
# --------------------------------------------------------------------------------------------


class SlurmRunner(Runner):
    """`Runner` whose watcher reports WATCH-TIMEOUT (the verify job outlives the bound; `fail_watch` names another terminal line, MISSING say)
    and whose `squeue` / `sacct` / `scancel` answer from a fake queue: a submitted job lands in the set `sbatch_to` names (`running` by default:
    RUNNING in the queue and the accounting; `queue_only`: RUNNING in the queue, no accounting row yet; `acct_only`: a RUNNING accounting row,
    gone from the queue; `completed` / `cancelled`: a terminal accounting row, gone from the queue; None: in neither reading) and is RUNNING
    until `scancel` names it, then CANCELLED in the accounting and gone from the queue. `queue_down` makes both queries fail (the transport
    refused: nothing may be confirmed); a job in NONE of the sets is in neither reading (both queries answer, rc 0, no row for it)."""

    def __init__(self, write_report=False, fail_watch="WATCH-TIMEOUT 0 after bound", **kw):
        super().__init__(fail_watch=fail_watch, **kw)
        self.running = set()
        self.cancelled = set()
        self.completed = set()                  # COMPLETED in the accounting, gone from the queue: a terminal row
        self.acct_only = set()                  # RUNNING in the accounting, gone from the queue: a non-terminal row, no queue row
        self.queue_only = set()                 # RUNNING in the queue, no accounting row: the accounting lagging a fresh job
        self.sbatch_to = "running"              # the set a job submitted through this runner lands in (None: in neither reading)
        self.queue_down = False
        self.scancel_calls = []
        self.write_report = write_report        # the verifier finishes AFTER its controller stopped watching: its report exists, the watch said timeout

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[0] == "sbatch":
            fw = self.fail_watch
            if self.write_report:
                self.fail_watch = None          # `Runner` writes the report only for a job whose watch will say DONE; here the job ran on and wrote it
            try:
                r = super().__call__(argv, timeout_s, input=input, check=check)
            finally:
                self.fail_watch = fw
            if self.sbatch_to:
                getattr(self, self.sbatch_to).add(str(self.job))
            return r
        if argv[0] == "bash" and len(argv) == 3 and "squeue" in argv[2] and "sacct" in argv[2]:
            self.calls.append(argv)
            if self.queue_down:
                return subprocess.CompletedProcess(argv, 0, stdout="---SQUEUE-RC=1---\n---SACCT-RC=1---\n", stderr="slurm down")
            user = os.environ.get("USER") or "tester"
            queued = "".join(f"{j}|RUNNING|store-verify-x\n" for j in sorted(self.running | self.queue_only))
            acct = ("".join(f"{j}|CANCELLED by {user}\n" for j in sorted(self.cancelled)) + "".join(f"{j}|RUNNING\n" for j in sorted(self.running))
                    + "".join(f"{j}|RUNNING\n" for j in sorted(self.acct_only)) + "".join(f"{j}|COMPLETED\n" for j in sorted(self.completed)))
            return subprocess.CompletedProcess(argv, 0, stdout=f"{queued}---SQUEUE-RC=0---\n{acct}---SACCT-RC=0---\n", stderr="")
        if argv[0] == "bash" and len(argv) == 3 and argv[2].startswith("scancel "):
            self.calls.append(argv)
            ids = argv[2].split()[1:]
            self.scancel_calls.append(ids)
            for j in ids:
                self.running.discard(j.strip("'"))
                self.cancelled.add(j.strip("'"))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return super().__call__(argv, timeout_s, input=input, check=check)


def test_a_bounded_carry_cancels_its_verifier_by_id_and_confirms_it_terminal_instead_of_abandoning_it(world):
    """`--bound-s`: the watch is bounded to it; when the verifier outlives the bound the CONTROLLER cancels the job it recorded (`scancel <id>`),
    confirms it terminal (an accounting row reading CANCELLED and no queue row -- reads_protocol.Slurm.wait_terminal) and only then returns
    `verify_cancelled`: the tree is left in place (never renamed), `current` is not switched, the generation record names the job and its
    terminal state. Before this an outer `timeout` killed the controller and its watcher while the submitted verifier ran on with the
    MirrorLock released, and the next carry renamed and repopulated the generation that verifier still read."""
    w = world
    rn = SlurmRunner()
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, bound_s=1.0, sleep=lambda s: None)
    assert res.outcome == "verify_cancelled", res.as_dict()
    assert res.job_id and rn.scancel_calls == [[res.job_id]] and res.detail["cancel"]["confirmed"] is True and res.detail["cancel"]["state"].startswith("CANCELLED")
    assert res.detail["watch"]["state"] == "WATCH-TIMEOUT" and rn.running == set() and rn.cancelled == {res.job_id}
    # the watch was bounded IN-PROCESS to the controller's bound (the runner's timeout); the script's own --max-hours is the next whole hour
    # past it (round 2 HIGH: a fractional hour is a Bash arithmetic error in the watcher); the tree stands where the copy put it, nothing is current
    watch_call = next(c for c in rn.calls if c[0] == WATCH)
    assert watch_call[watch_call.index("--max-hours") + 1] == "1" and rn.watch_timeouts == [1.0]
    assert res.detail["watch"]["bound_s"] == 1.0 and res.detail["watch"]["max_hours"] == 1
    gen_dir = w["gen_dir"]
    assert os.path.isdir(os.path.join(gen_dir, res.content_sha8)) and not os.path.lexists(os.path.join(gen_dir, "current"))
    assert not any(n.startswith(".superseded-") or n.startswith(".invalid-") for n in os.listdir(gen_dir))
    rec = ST.load_state(w["root"])["components"][w["comp"].slug]
    assert rec["generation"]["state"] == "verify_cancelled" and rec["generation"]["job_id"] == res.job_id and rec["generation"]["terminal"].startswith("CANCELLED")
    assert rec["generation"]["bound_s"] == 1.0 and rec["last_run"]["outcome"] == "verify_cancelled"
    # a cancel that cannot be CONFIRMED (the queue unreadable) keeps the awaiting_verify record with the job id: nothing claims the job is gone
    rn2 = SlurmRunner()
    rn2.queue_down = True
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2, bound_s=1.0, sleep=lambda s: None)
    assert res2.outcome == "verify_unconfirmed" and res2.detail["cancel"]["confirmed"] is False and "NOT confirmed terminal" in res2.detail["cancel"]["why"]
    rec2 = ST.load_state(w["root"])["components"][w["comp"].slug]["generation"]
    assert rec2["state"] == "awaiting_verify" and rec2["job_id"] == res2.job_id and rec2["cancel"]["confirmed"] is False


def test_a_recorded_awaiting_verify_job_is_settled_before_a_carry_copies_or_renames_anything(world):
    """The second half of P2: a run that finds the component's record saying `awaiting_verify` with a job id asks the queue first. While the
    job lives the run is REFUSED by name (`verify_in_flight`): no rsync, no rename, no pointer switch -- the earlier verifier still reads the
    tree a recopy would rename aside. When the queue cannot be read the run is `verify_unconfirmed` (fail closed, nothing touched). Once the
    job reads terminal the run goes on: here the report the (fake) verifier wrote matches, so the existing-verified path publishes."""
    w = world
    gen_dir = w["gen_dir"]
    # an earlier controller submitted the verify and its watch ended at the bound; the verifier itself ran on and WROTE its report (the case
    # the bound exists for: a slow verify, not a dead one); the record is then rewritten to say awaiting_verify as a controller that died
    # before any cancel would have left it, with the job id
    rn0 = SlurmRunner(write_report=True)
    first = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn0, bound_s=1.0, sleep=lambda s: None)
    assert first.outcome == "verify_cancelled"
    sha8, job = first.content_sha8, first.job_id
    ST.update_state(w["root"], w["comp"].slug, {"generation": {"content_sha8": sha8, "state": "awaiting_verify", "job_id": job}}, w["dest"].name)
    rn = SlurmRunner()
    rn.running.add(job)                         # the earlier verifier is alive in this run's queue
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, sleep=lambda s: None)
    assert res.outcome == "verify_in_flight", res.as_dict()
    assert res.detail["awaiting_verify"]["job_id"] == job and res.detail["awaiting_verify"]["state"] == "RUNNING" and "still reads the" in res.detail["awaiting_verify"]["why"]
    assert rn.count("rsync") == 0 and not any(c[0] == "sbatch" for c in rn.calls) and rn.scancel_calls == []
    assert os.path.isdir(os.path.join(gen_dir, sha8)) and not os.path.lexists(os.path.join(gen_dir, "current"))
    assert not any(n.startswith(".superseded-") for n in os.listdir(gen_dir)), "nothing renamed under a live verifier"
    rec = ST.load_state(w["root"])["components"][w["comp"].slug]["generation"]
    assert rec["state"] == "awaiting_verify" and rec["job_id"] == job and rec["settle"]["refusal"] == "verify_in_flight"
    # the queue unreadable: fail closed, nothing touched either
    rn_d = SlurmRunner()
    rn_d.queue_down = True
    res_d = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn_d, sleep=lambda s: None)
    assert res_d.outcome == "verify_unconfirmed" and rn_d.count("rsync") == 0 and not any(c[0] == "sbatch" for c in rn_d.calls)
    # the job terminal (cancelled, or finished): settled, and the run goes on -- the report the fake verifier wrote under `first` matches the
    # manifest, so the existing-verified path publishes it; no second copy, no rename
    rn2 = SlurmRunner()
    rn2.cancelled.add(job)
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2, sleep=lambda s: None)
    assert res2.outcome == "published_existing_verified", res2.as_dict()
    assert res2.detail["awaiting_verify_settled"]["job_id"] == job and res2.detail["awaiting_verify_settled"]["state"].startswith("CANCELLED")
    assert rn2.count("rsync") == 0 and os.readlink(os.path.join(gen_dir, "current")) == sha8
    final = ST.load_state(w["root"])["components"][w["comp"].slug]
    assert final["generation"]["state"] == "published" and final["published"] == sha8 and final["last_run"]["outcome"] == "published_existing_verified"


def test_a_recorded_awaiting_verify_job_in_neither_the_queue_nor_the_accounting_is_unconfirmed_and_never_settled(world):
    """Review round 1 MEDIUM: `settle_awaiting_verify` read `Slurm.states`, which folds "no row in the queue, no row in the accounting" into
    None, and took None for settled -- so after a bounded run left a verifier recorded `awaiting_verify` (its cancel unconfirmed), a later
    run whose `squeue` and `sacct` BOTH answered (rc 0) with no row for the job -- the accounting not yet caught up with a job that had just
    left the queue, or a row it purged -- went on to rename and recopy the generation that verifier may still read: the very thing the
    settle exists to prevent. Now the settle reads the queue and the accounting together and applies the one terminal-confirmation rule
    (`reads_protocol.terminal_verdicts`, the rule `wait_terminal` polls on): an explicit terminal accounting row AND no queue row settles;
    a known non-terminal state (the queue's, or an accounting row not yet terminal) is `verify_in_flight`; absence from both is
    `verify_unconfirmed` -- the record keeps `awaiting_verify` with the job id, nothing is copied, renamed or switched, and the next run
    asks again."""
    w = world
    gen_dir = w["gen_dir"]
    rn0 = SlurmRunner(write_report=True)
    first = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn0, bound_s=1.0, sleep=lambda s: None)
    assert first.outcome == "verify_cancelled"
    sha8, job = first.content_sha8, first.job_id
    ST.update_state(w["root"], w["comp"].slug, {"generation": {"content_sha8": sha8, "state": "awaiting_verify", "job_id": job}}, w["dest"].name)
    # both queries answer and neither has a row for the job: not proved terminal -- refused by name, the record stands, nothing moves
    rn = SlurmRunner()
    assert job not in rn.running and job not in rn.cancelled and job not in rn.acct_only
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, sleep=lambda s: None)
    assert res.outcome == "verify_unconfirmed", res.as_dict()
    settled = res.detail["awaiting_verify"]
    assert settled["job_id"] == job and settled["state"] is None and settled["where"] is None and settled["refusal"] == "verify_unconfirmed"
    assert "no accounting row" in settled["why"] and "nothing copied, renamed or switched" in settled["why"] and f"sacct -j {job}" in settled["why"]
    queries = [c for c in rn.calls if c[0] == "bash" and "squeue" in c[2] and "sacct" in c[2]]
    assert len(queries) == 1 and job in queries[0][2], "one reading of the queue and the accounting together, for this job"
    assert rn.count("rsync") == 0 and not any(c[0] == "sbatch" for c in rn.calls) and rn.scancel_calls == [] and rn.count(WATCH) == 0
    assert os.path.isdir(os.path.join(gen_dir, sha8)) and not os.path.lexists(os.path.join(gen_dir, "current"))
    assert not any(n.startswith((".superseded-", ".invalid-", ".failed-", ".incoming-")) for n in os.listdir(gen_dir)), "nothing renamed under an unproved verifier"
    rec = ST.load_state(w["root"])["components"][w["comp"].slug]
    assert rec["generation"]["state"] == "awaiting_verify" and rec["generation"]["job_id"] == job and rec["generation"]["settle"]["refusal"] == "verify_unconfirmed"
    assert rec["last_run"]["outcome"] == "verify_unconfirmed" and ST.OUTCOME_LEVELS["verify_unconfirmed"] == "red"
    # the accounting's row still RUNNING while the queue has dropped it (lagging the other way): a known non-terminal state -- in flight, by name
    rn_a = SlurmRunner()
    rn_a.acct_only.add(job)
    res_a = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn_a, sleep=lambda s: None)
    assert res_a.outcome == "verify_in_flight", res_a.as_dict()
    assert res_a.detail["awaiting_verify"]["state"] == "RUNNING" and res_a.detail["awaiting_verify"]["where"] == "accounting"
    assert rn_a.count("rsync") == 0 and not any(c[0] == "sbatch" for c in rn_a.calls) and not os.path.lexists(os.path.join(gen_dir, "current"))
    assert ST.load_state(w["root"])["components"][w["comp"].slug]["generation"]["state"] == "awaiting_verify"
    # asked again once the accounting has the terminal row and the queue is silent: confirmed, and the run goes on -- the report the fake
    # verifier wrote under `first` matches, so the existing-verified path publishes; no second copy, no rename
    rn2 = SlurmRunner()
    rn2.cancelled.add(job)
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2, sleep=lambda s: None)
    assert res2.outcome == "published_existing_verified", res2.as_dict()
    done = res2.detail["awaiting_verify_settled"]
    assert done["job_id"] == job and done["where"] == "accounting" and done["state"].startswith("CANCELLED") and done["refusal"] is None
    assert rn2.count("rsync") == 0 and os.readlink(os.path.join(gen_dir, "current")) == sha8
    final = ST.load_state(w["root"])["components"][w["comp"].slug]
    assert final["generation"]["state"] == "published" and final["published"] == sha8


# --------------------------------------------------------------------------------------------
# the watch's bound is enforced in-process and the watcher's own stop is a whole hour
# --------------------------------------------------------------------------------------------


class KilledWatchRunner(SlurmRunner):
    """`SlurmRunner` whose watcher outlives the bound: the runner's timeout fires -- what `subprocess.run(timeout=...)` does to a watcher that
    has not exited -- with the lines it had printed so far (bytes, as the exception carries them)."""

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[0] == WATCH:
            self.calls.append(argv)
            self.watch_timeouts.append(timeout_s)
            raise subprocess.TimeoutExpired(argv, timeout_s, output=f"RUNNING {argv[argv.index('--slurm') + 1]} node=test\n".encode())
        return super().__call__(argv, timeout_s, input=input, check=check)


def test_the_watch_is_bounded_in_this_process_and_the_watchers_own_stop_is_the_next_whole_hour_past_the_bound(world):
    """Review round 2 HIGH: `Remote.watch` rendered the bound as hours -- `--max-hours 1.0` for the window's `--bound-s 3600`, `0.5` for the
    authority carry's 1800, `4.0` for the destination's default -- and the watcher computes its deadline with Bash integer arithmetic, which
    refuses a fractional token before its first poll; the dead watcher read UNKNOWN and the controller cancelled every verifier it had
    submitted, so no newly declared component could be published by either carry. Now the bound is enforced IN THIS PROCESS (the runner's
    timeout; a killed watcher reads WATCH-TIMEOUT with rc None and the controller cancels and confirms as before) and `--max-hours` is a
    whole number, the next hour past the bound, a backstop that never races it; the watch record carries both."""
    w = world
    remote = ST.Remote(w["dest"], runner=Runner())
    for bound, hours in ((3600.0, 2), (1800.0, 1), (1.0, 1), (7200.0, 3), (None, int(w["dest"].watch_max_hours) + 1), (0, int(w["dest"].watch_max_hours) + 1)):
        argv, enforced, max_hours = remote.watch_argv("77", bound)
        assert argv[argv.index("--max-hours") + 1] == str(hours) and max_hours == hours, (bound, argv)
        assert enforced == (bound if bound else w["dest"].watch_max_hours * 3600.0) and max_hours * 3600 > enforced
        assert all(argv[argv.index(k) + 1].isdigit() for k in ("--max-hours", "--poll", "--stall-min", "--pending-min")), argv
        assert argv[:3] == [WATCH, "--slurm", "77"] and "--host" not in argv        # a local destination: the script's own default host
    # a bounded carry of a never-published component PUBLISHES: the watcher (DONE under the bound) was handed a whole hour and the bound itself
    rn = Runner()
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, bound_s=3600.0, sleep=lambda s: None)
    assert res.outcome == "published", res.as_dict()
    call = next(c for c in rn.calls if c[0] == WATCH)
    assert call[call.index("--max-hours") + 1] == "2" and rn.watch_timeouts == [3600.0]
    assert res.detail["watch"]["state"] == "DONE" and res.detail["watch"]["bound_s"] == 3600.0 and res.detail["watch"]["max_hours"] == 2
    assert os.readlink(os.path.join(w["gen_dir"], "current")) == res.content_sha8
    # the bound expiring IN THIS PROCESS: the runner's timeout kills the watcher; the controller reads WATCH-TIMEOUT (rc None, the lines so
    # far kept) and, as under the script's own timeout, cancels its job by id and confirms it terminal; the published generation stays current
    with open(os.path.join(w["base"], "later.txt"), "w") as fh:
        fh.write("a second generation\n")
    rn2 = KilledWatchRunner()
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2, bound_s=1800.0, sleep=lambda s: None)
    assert res2.outcome == "verify_cancelled", res2.as_dict()
    watched = res2.detail["watch"]
    assert watched["state"] == "WATCH-TIMEOUT" and watched["rc"] is None and "expired in this process" in watched["terminal"]
    assert watched["tail"][0].startswith("RUNNING") and watched["tail"][-1] == watched["terminal"]
    assert watched["bound_s"] == 1800.0 and watched["max_hours"] == 1 and rn2.watch_timeouts == [1800.0]
    assert rn2.scancel_calls == [[res2.job_id]] and res2.detail["cancel"]["confirmed"] is True
    assert res2.content_sha8 != res.content_sha8 and os.readlink(os.path.join(w["gen_dir"], "current")) == res.content_sha8
    rec = ST.load_state(w["root"])["components"][w["comp"].slug]
    assert rec["generation"]["state"] == "verify_cancelled" and rec["generation"]["watch"]["rc"] is None and rec["published"] == res.content_sha8


_FAKE_SSH = r"""#!/usr/bin/env bash
# a fake transport for the watcher: the LAST argument is the remote command; the accounting answers COMPLETED for the job, the queue is silent
cmd="${@: -1}"
printf '%s\n' "$cmd" >> "$FAKE_SSH_LOG"
id=$(sed -n 's/.*-j \([0-9_]*\).*/\1/p' <<< "$cmd" | head -1)
case "$cmd" in
  *"squeue "*) exit 0 ;;
  *"-o State,ExitCode,Elapsed,NodeList,JobID"*) printf 'COMPLETED|0:0|00:00:03|node-x|%s\n' "$id" ;;
  *"-o JobID,JobName,State,ExitCode,Elapsed,NodeList"*) printf '%s|store-verify-x|COMPLETED|0:0|00:00:03|node-x\n' "$id" ;;
  *"-o JobName,State"*) printf 'store-verify-x|COMPLETED\n' ;;
  *) exit 0 ;;
esac
"""


def test_the_real_watcher_accepts_the_argv_the_carry_builds_under_both_window_bounds_and_dies_on_a_fractional_hour(tmp_path):
    """Review round 2 HIGH, against the watcher ITSELF (the destination's default `watch_job`, the machine's watch-job.sh) with `ssh` and
    `sleep` shimmed on PATH: a fake transport whose accounting answers COMPLETED for the job and whose queue is silent. The argv
    `Remote.watch_argv` builds for the window's two bounds (3600, the mirror carry; 1800, the authority carry) runs to DONE, rc 0, through
    `Remote.watch` too; the same argv with the fractional hour the old code rendered dies in the script's deadline arithmetic before its
    first poll (no DONE, the transport never consulted) -- the reason the hour is whole."""
    dest = ST.Destination("real", None, str(tmp_path / "shared"))
    script = dest.watch_job
    if not (os.path.isfile(script) and os.access(script, os.X_OK)):
        pytest.skip(f"the destination's default watcher is not installed on this machine: {script}")
    shim = tmp_path / "shim"
    shim.mkdir()
    log = tmp_path / "ssh.log"
    (shim / "ssh").write_text(_FAKE_SSH)
    (shim / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")     # the script's `sleep 15` between its two accounting reads, and its poll
    for f in ("ssh", "sleep"):
        os.chmod(shim / f, 0o755)
    env = {**os.environ, "PATH": f"{shim}:{os.environ.get('PATH', '')}", "FAKE_SSH_LOG": str(log)}
    remote = ST.Remote(dest, runner=lambda argv, timeout_s, input=None: subprocess.run(list(argv), capture_output=True, text=True, env=env, timeout=timeout_s))
    for bound in (3600.0, 1800.0):
        argv, enforced, hours = remote.watch_argv("4242", bound)
        assert argv[0] == script and hours * 3600 > enforced == bound
        r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        assert r.returncode == 0 and any(l.startswith("DONE job=4242") for l in lines), (bound, r.stdout, r.stderr)
        watched = remote.watch("4242", bound)
        assert watched["state"] == "DONE" and watched["rc"] == 0 and watched["bound_s"] == bound and watched["max_hours"] == hours, watched
    text = log.read_text()
    assert "sacct -X -n -P -o State,ExitCode,Elapsed,NodeList,JobID -j 4242" in text and "squeue -h -j 4242" in text, text
    # the fractional hour the old code rendered (`str(3600 / 3600.0)`): the script dies in its deadline arithmetic, before any poll
    argv, _, _ = remote.watch_argv("4242", 3600.0)
    bad = list(argv)
    bad[bad.index("--max-hours") + 1] = str(3600.0 / 3600.0)
    log.write_text("")
    r = subprocess.run(bad, capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode != 0 and not any(l.startswith("DONE") for l in r.stdout.splitlines()) and "arithmetic" in r.stderr, (r.stdout, r.stderr)
    assert log.read_text() == "", "it never reached the transport"


# --------------------------------------------------------------------------------------------
# a MISSING watcher verdict is not terminal
# --------------------------------------------------------------------------------------------


MISSING_LINE = "MISSING job=? host=x -- no sacct record after 10 min; wrong job id or wrong cluster?"


@pytest.mark.parametrize("lands_in, outcome, state, where", [
    ("queue_only", "verify_in_flight", "RUNNING", "queue"),          # RUNNING in the queue, the accounting silent: the finding's scenario
    ("running", "verify_in_flight", "RUNNING", "queue"),
    ("acct_only", "verify_in_flight", "RUNNING", "accounting"),
    (None, "verify_unconfirmed", None, None),                        # in neither reading
    ("down", "verify_unconfirmed", None, None),                      # the transport refused
])
def test_a_missing_watcher_verdict_is_not_terminal_the_run_reads_the_queue_and_the_accounting_once_and_keeps_its_record(world, lands_in, outcome, state, where):
    """Review round 2 MEDIUM: the watcher says MISSING when the accounting has shown no row for ten minutes -- it never consults the queue --
    and the branch read that as FAILED: the record became `verify_job_failed`, so the next run settled nothing and could rename and recopy the
    generation the still-running verifier read. Now MISSING is the watcher LOSING the job, not a terminal state: the controller reads the
    queue and the accounting together once by the one terminal rule (`settle_awaiting_verify`, as for a record an earlier run left) and a
    live or unproven job KEEPS the `awaiting_verify` record with its job id -- `verify_in_flight` for a known non-terminal state,
    `verify_unconfirmed` for no proof -- nothing cancelled, renamed or switched; every later run settles that record by the same rule."""
    w = world
    rn = SlurmRunner(write_report=True, fail_watch=MISSING_LINE)
    rn.sbatch_to = None if lands_in in (None, "down") else lands_in
    rn.queue_down = lands_in == "down"
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, bound_s=1800.0, sleep=lambda s: None)
    assert res.outcome == outcome, res.as_dict()
    job, sha8, gen_dir = res.job_id, res.content_sha8, w["gen_dir"]
    assert res.detail["watch"]["state"] == "MISSING" and res.detail["watch"]["terminal"].startswith("MISSING") and "settled_by" not in res.detail["watch"]
    lost = res.detail["lost_watch"]
    assert lost["job_id"] == job and lost["state"] == state and lost["where"] == where and lost["refusal"] == outcome, lost
    assert "this run, whose watcher lost it (MISSING)" in lost["why"] and "nothing copied, renamed or switched" in lost["why"]
    assert rn.scancel_calls == [], "the controller's bound did not expire: its job is not cancelled on the watcher's word alone"
    queries = [c for c in rn.calls if c[0] == "bash" and "squeue" in c[2] and "sacct" in c[2]]
    assert len(queries) == 1 and job in queries[0][2], "one reading of the queue and the accounting together, for this job"
    assert os.path.isdir(os.path.join(gen_dir, sha8)) and not os.path.lexists(os.path.join(gen_dir, "current"))
    assert not any(n.startswith((".superseded-", ".invalid-", ".failed-", ".incoming-")) for n in os.listdir(gen_dir))
    rec = ST.load_state(w["root"])["components"][w["comp"].slug]
    g = rec["generation"]
    assert g["state"] == "awaiting_verify" and g["job_id"] == job and g["watch"]["state"] == "MISSING" and g["settle"] == lost and g["bound_s"] == 1800.0
    assert rec["last_run"]["outcome"] == outcome and rec["last_run"]["why"] == lost["why"] and rec.get("published") is None
    # the next run settles the SAME record by the same rule: while the job lives or is unproven, zero copies, renames and pointer switches
    rn2 = SlurmRunner()
    if lands_in == "down":
        rn2.queue_down = True
    elif lands_in:
        getattr(rn2, lands_in).add(job)
    res2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn2, sleep=lambda s: None)
    assert res2.outcome == outcome and res2.detail["awaiting_verify"]["job_id"] == job, res2.as_dict()
    assert rn2.count("rsync") == 0 and not any(c[0] == "sbatch" for c in rn2.calls) and rn2.scancel_calls == [] and rn2.count(WATCH) == 0
    assert not os.path.lexists(os.path.join(gen_dir, "current")) and os.path.isdir(os.path.join(gen_dir, sha8))
    assert not any(n.startswith((".superseded-", ".invalid-", ".failed-", ".incoming-")) for n in os.listdir(gen_dir))
    assert ST.load_state(w["root"])["components"][w["comp"].slug]["generation"]["state"] == "awaiting_verify"
    # once the accounting proves the job terminal and the queue is silent, the report the verifier wrote publishes the tree it verified
    rn3 = SlurmRunner()
    rn3.completed.add(job)
    res3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn3, sleep=lambda s: None)
    assert res3.outcome == "published_existing_verified", res3.as_dict()
    assert res3.detail["awaiting_verify_settled"]["state"] == "COMPLETED" and rn3.count("rsync") == 0
    assert os.readlink(os.path.join(gen_dir, "current")) == sha8


@pytest.mark.parametrize("lands_in, outcome, stood_in", [("completed", "published", "DONE"), ("cancelled", "verify_job_failed", "FAILED")])
def test_a_missing_watcher_verdict_the_accounting_proves_terminal_takes_the_accountings_word_for_the_watchers(world, lands_in, outcome, stood_in):
    """The other half of the round-2 MEDIUM rule: when the one reading PROVES the lost job terminal (an explicit terminal accounting row, no
    queue row) the accounting's state stands in for the watcher's line -- COMPLETED goes on to the report and publishes, any other terminal
    state is `verify_job_failed` -- and the watch record says so (`settled_by: accounting`). Nothing is cancelled either way."""
    w = world
    rn = SlurmRunner(write_report=True, fail_watch=MISSING_LINE)
    rn.sbatch_to = lands_in
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn, bound_s=1800.0, sleep=lambda s: None)
    assert res.outcome == outcome, res.as_dict()
    watched = res.detail["watch"]
    assert watched["state"] == stood_in and watched["settled_by"] == "accounting" and watched["terminal"].startswith("MISSING")
    assert res.detail["lost_watch"]["refusal"] is None and str(watched["accounting_state"]).startswith(lands_in.upper())
    assert rn.scancel_calls == []
    rec = ST.load_state(w["root"])["components"][w["comp"].slug]
    if outcome == "published":
        assert os.readlink(os.path.join(w["gen_dir"], "current")) == res.content_sha8 and rec["generation"]["state"] == "published"
    else:
        assert not os.path.lexists(os.path.join(w["gen_dir"], "current"))
        assert rec["generation"]["state"] == "verify_job_failed" and rec["generation"]["job_id"] == res.job_id and rec["generation"]["watch"]["settled_by"] == "accounting"


# --------------------------------------------------------------------------------------------
# the staging copy's directories are owner-writable: a first publication from a SEALED source completes an escaped empty directory
# --------------------------------------------------------------------------------------------


def _seal_tree(top):
    """Every directory of ``top`` 0555, deepest first (a sealed published generation standing as a canonical source); returns the paths."""
    dirs = [dp for dp, _dns, _fns in os.walk(top)]
    for d in sorted(dirs, key=len, reverse=True):
        os.chmod(d, ST.DIR_SEAL_MODE)
    return dirs


def test_a_first_publication_from_a_sealed_source_with_an_escaped_empty_directory_succeeds_and_the_seal_still_holds_after(world, monkeypatch):
    """The canonical is a SEALED tree (every directory 0555, as a published generation is) holding an attested EMPTY directory whose name the
    destination cannot spell: rsync excludes the escape top, tar carries files and links only, so the verify must CREATE it under the
    staging root. Before this the staging directories inherited the source's 0555 (rsync without -p: the source mode masked by the umask)
    and the verify, reading the seal, created nothing -- the first publication failed `verify_failed` for any user. The copy's directories
    are owner-writable now (`--chmod=Du+w`), the escaped directory is created at its escaped spelling, the generation publishes, and the
    publication seal applied after the verify still holds on every directory. Privilege-independent in what it asserts of the copy and
    the seal: `os.makedirs` is spied, so the creation must be ASKED for (a process the kernel would not refuse -- root -- cannot pass by
    writing into a sealed staging tree the seal-reading verify would have skipped), and every directory's mode is read; the one kernel
    refusal demonstrated at the end runs only where the probe reads the refusal (root and CAP_DAC_OVERRIDE create under 0555)."""
    w = world
    base = w["base"]
    os.makedirs(os.path.join(base, "d:x"))                          # an empty directory whose name needs the escape (':' -> '%3A')
    asked = []
    real_makedirs = os.makedirs

    def recording_makedirs(path, *args, **kw):
        asked.append(path)
        return real_makedirs(path, *args, **kw)

    sealed = _seal_tree(base)
    try:
        monkeypatch.setattr(os, "makedirs", recording_makedirs)
        rn = Runner()
        res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    finally:
        monkeypatch.undo()
        for d in sealed:
            os.chmod(d, 0o755)
    assert res.outcome == "published", res.as_dict()
    gd = w["gen_dir"]
    sha8 = res.content_sha8
    gen = f"{gd}/{sha8}"
    rs = next(a for a in rn.calls if "rsync" in a)
    assert f"--chmod={ST.STAGING_DIR_CHMOD}" in rs
    escaped = os.path.join(gen, ST.escape_name("d:x"))
    assert os.path.isdir(escaped) and ST.escape_name("d:x") == "d%3Ax"
    assert escaped in asked, asked                                       # the verify asked the kernel to create it: the staging root was writable
    ver = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert ver["ok"] is True and ver["created_dirs"] == ["d:x"] and ver["listing_mismatch"] == []
    meta = json.load(open(f"{gd}/MANIFEST-{sha8}.json"))
    assert meta["directories"]["d:x"] == [] and ["d:x", "dir"] in meta["directories"][""]
    # the publication seal holds on EVERY directory of the published generation, the created one included: nothing of the seal is weakened
    for dp, _dns, _fns in os.walk(gen):
        assert stat_module.S_IMODE(os.stat(dp).st_mode) == ST.DIR_SEAL_MODE, dp
    sealed_rec = json.load(open(f"{gd}/SEALED-{sha8}.json"))
    assert sealed_rec["n_dirs"] == sum(1 for _ in os.walk(gen)) and os.readlink(f"{gd}/current") == sha8
    # and the published generation is still immutable to its owner: an entry cannot be added inside it -- the kernel's refusal of the
    # directory's mode, which the same privilege that bypasses a directory's mode bypasses (root / CAP_DAC_OVERRIDE); the demonstration
    # runs only where the probe reads the refusal (the seal test's rule above), and every assertion before it holds regardless
    if permission_denial_enforced():
        with pytest.raises(PermissionError):
            os.makedirs(os.path.join(gen, "planted"))
