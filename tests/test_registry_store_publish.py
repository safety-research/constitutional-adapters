"""`registry stores push`, the publish guard, the directory seal, `check-authority`, `verify-authority` and the name escape
(VAST refuses ':' in a name). The destination is a local directory and the shared `Runner` of test_registry_stores passes every
shell step, rsync and tar through to real subprocesses while intercepting `sbatch` (the verify runs in-process) and the watcher
-- so the seal (`chmod 0555`), the guard (`test -f ... && mv -T`) and the `tar --transform` copy are the real commands the
destination runs.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat as stat_module
import subprocess
import sys
import threading
import time

import pytest
from dataclasses import replace

from sisteer.registry import cli
from sisteer.registry import cluster_reads as CR
from sisteer.registry import records as R
from sisteer.registry import stores as ST

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import permission_denial_enforced  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)
from test_registry_stores import Runner, WATCH, _canon, _cfg, _write, world  # noqa: E402,F401  (the fixture rides along)

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
FILES = ("a/b/f1.json", "a/f2.jsonl", "top.txt", "x/deep/er/f3.bin")


def _dirs(top):
    out = []
    for dp, dns, _fns in os.walk(top):
        out.append(dp)
    return out


def _mode(p):
    return stat_module.S_IMODE(os.lstat(p).st_mode)


def _set_mtimes(base, ns_by_rel):
    for rel, ns in ns_by_rel.items():
        os.utime(os.path.join(base, rel), ns=(ns, ns))


def _fmt(ns):
    from datetime import datetime, timezone
    return R._fmt_utc(datetime.fromtimestamp(ns / 1e9, timezone.utc))


def _rec_text(sha8, previous=None, producer=None, mode="push"):
    """A minimal publication record for a direct `Remote.publish` / `record_published` call (mirror_component builds the real one)."""
    return ST.record_text({"schema": ST.PUBLISHED_SCHEMA, "content_sha8": sha8, "previous": previous, "producer": producer, "mode": mode,
                           "published_utc": R.utc_now()})


# --------------------------------------------------------------------------------------------
# the seal and the guard
# --------------------------------------------------------------------------------------------


def test_publish_seals_every_directory_records_sealed_and_the_guard_refuses_an_incomplete_generation(world):
    w = world
    rn = Runner()
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=rn)
    assert res.outcome == "published", res.as_dict()
    gd, sha8 = w["gen_dir"], res.content_sha8
    gen = f"{gd}/{sha8}"
    dirs = _dirs(gen)
    assert len(dirs) == 6 and all(_mode(d) == ST.DIR_SEAL_MODE for d in dirs), {d: oct(_mode(d)) for d in dirs}
    assert _mode(gd) & 0o200, "the parent gen/<slug>/ stays writable: `current` switches and a failed tree is renamed there"
    sealed = json.load(open(f"{gd}/SEALED-{sha8}.json"))
    assert sealed["schema"] == ST.SEALED_SCHEMA and sealed["n_dirs"] == 6 and sealed["mode"] == "0555" and sealed["resealed"] is False
    assert sealed["generation"] == gen and sealed["sealed_utc"] == sealed["first_sealed_utc"] and sealed["host"]
    assert res.detail["sealed"] == {"n_dirs": 6, "sealed_utc": sealed["sealed_utc"], "resealed": False}
    assert ST.load_state(w["root"])["components"]["relu_paired_run_state"]["generation"]["sealed"]["n_dirs"] == 6
    # a sealed directory refuses a write for the owner (the plan's R14 on VAST; the local filesystem honors the mode the same way). The
    # kernel's refusal is the thing demonstrated: a process the kernel does not refuse (root / CAP_DAC_OVERRIDE, the CPU-suite container)
    # would plant the file instead, so the demonstration runs only where the probe reads the refusal; everything after it asserts regardless
    if permission_denial_enforced():
        with pytest.raises(PermissionError):
            open(f"{gen}/a/planted.txt", "w").close()
        with pytest.raises(PermissionError):
            os.mkdir(f"{gen}/x/planted")
    assert sorted(os.listdir(f"{gen}/a")) == ["b", "f2.jsonl"], "a sealed directory still lists"
    assert os.readlink(f"{gd}/current") == sha8
    # every publication is recorded inside the locked step, a plain `stores mirror` included (mode `mirror`): the next publisher requires it
    mrec = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    assert mrec["mode"] == "mirror" and mrec["previous"] is None and mrec["producer"] is None and res.detail["published_record"]["mode"] == "mirror"
    # the guard: `current` never switches onto a tree that lacks any of the four files, or that is not there
    remote = ST.Remote(w["dest"], Runner())
    os.rename(f"{gd}/current", f"{gd}/current.aside")
    for name in (f"SEALED-{sha8}.json", f"DEST-STAT-{sha8}.json", f"VERIFIED-{sha8}.json", f"MANIFEST-{sha8}.sha256"):
        os.rename(f"{gd}/{name}", f"{gd}/{name}.aside")
        with pytest.raises(ST.StoresError, match=f"publish_refused.*{name.split('-')[0]}"):
            remote.publish(gd, sha8, _rec_text(sha8))
        assert not os.path.lexists(f"{gd}/current"), name
        os.rename(f"{gd}/{name}.aside", f"{gd}/{name}")
    with pytest.raises(ST.StoresError, match="publish_refused.*the generation directory"):
        remote.publish(gd, "deadbeef", _rec_text("deadbeef"))
    with pytest.raises(ST.StoresError, match="publish_refused.*not a directory"):
        remote.publish(f"{gd}-nowhere", sha8, _rec_text(sha8))
    remote.publish(gd, sha8, _rec_text(sha8))
    assert os.readlink(f"{gd}/current") == sha8
    os.remove(f"{gd}/current.aside")
    # a run that died between the verify and the publish: the tree + VERIFIED stand, `current` does not -> published_existing_verified,
    # sealed again (the record says so) and through the guard
    os.remove(f"{gd}/current")
    r2 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r2.outcome == "published_existing_verified" and os.readlink(f"{gd}/current") == sha8 and r2.detail["sealed"]["resealed"] is True
    sealed2 = json.load(open(f"{gd}/SEALED-{sha8}.json"))
    assert sealed2["first_sealed_utc"] == sealed["first_sealed_utc"] and sealed2["sealed_utc"] >= sealed["sealed_utc"]
    # the seal is refused by name when the filesystem would not honor it (a runner whose chmod leaves a directory writable)
    class _Unsealing(Runner):
        def __call__(self, argv, timeout_s, input=None, check=False):
            if argv[:2] == ["sh", "-c"] and "chmod" in argv[2]:
                return subprocess.CompletedProcess(argv, 0, stdout="6 2 testhost\n", stderr="")
            return super().__call__(argv, timeout_s, input=input, check=check)
    with pytest.raises(ST.StoresError, match="seal_refused: 2 of 6"):
        ST.seal_generation(ST.Remote(w["dest"], _Unsealing()), gd, sha8)


# --------------------------------------------------------------------------------------------
# stores push: the PUBLISHED record, change_utc, adoption, refusals, --from, the interval
# --------------------------------------------------------------------------------------------


def test_push_writes_the_published_record_with_the_producers_change_utc_and_adopts_an_unrecorded_generation(world):
    w = world
    base = w["base"]
    t0 = 1_760_000_000_000_000_000
    _set_mtimes(base, {"a/b/f1.json": t0 + 1_000_000_000, "a/f2.jsonl": t0 + 2_000_000_000, "top.txt": t0 + 7_000_000_000, "x/deep/er/f3.bin": t0 + 3_000_000_000})
    rn = Runner()
    res = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn, note="first", session="producer-a")
    assert res.outcome == "published", res.as_dict()
    gd, sha8 = w["gen_dir"], res.content_sha8
    pub = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    assert pub["schema"] == ST.PUBLISHED_SCHEMA and pub["slug"] == "relu_paired_run_state" and pub["content_sha8"] == sha8 and pub["mode"] == "push"
    assert pub["change_utc"] == _fmt(t0 + 7_000_000_000), "the producer's committed change: the latest mtime among the changed files (every file, first generation)"
    assert pub["published_utc"] > pub["change_utc"] and pub["previous"] is None and pub["changed_files"] == 4 and pub["n_files"] == 4
    assert pub["note"] == "first" and pub["producer_session"] == "producer-a" and "@" in pub["pushed_by"] and pub["producer_host"] and pub["source"] == base
    assert pub["authority"] == "grading_host_copy" and pub["n_name_escapes"] == 0
    assert res.detail["published_record"]["change_utc"] == pub["change_utc"] and os.path.isfile(f"{gd}/SEALED-{sha8}.json")
    # unchanged: no copy, no record rewritten, the record reported
    rn2 = Runner()
    r2 = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn2, note="again")
    assert r2.outcome == "unchanged" and "adopted" not in r2.detail and r2.detail["published_record"]["note"] == "first"
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn2.calls)
    # one file changes with a known mtime: the next generation's record names the previous one, the one changed file, and ITS mtime
    _write(os.path.join(base, "a/f2.jsonl"), b"row1\nrow2\nrow3\n")
    _set_mtimes(base, {"a/f2.jsonl": t0 + 9_000_000_000})
    r3 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="second")
    assert r3.outcome == "published" and r3.content_sha8 != sha8
    pub3 = json.load(open(f"{gd}/PUBLISHED-{r3.content_sha8}.json"))
    assert pub3["previous"] == sha8 and pub3["changed_files"] == 1 and pub3["change_utc"] == _fmt(t0 + 9_000_000_000) and pub3["note"] == "second"
    assert os.readlink(f"{gd}/current") == r3.content_sha8 and all(_mode(d) == 0o555 for d in _dirs(f"{gd}/{sha8}")), "the old generation stays sealed"
    # ADOPTION: a generation an older flow published (no SEALED, no PUBLISHED, writable directories). A push that finds it unchanged
    # refuses by name -- a `current` without its record may be a publisher mid-flight -- and adopts it only when told
    # to (`--adopt-unrecorded`): sealed and recorded under the publish lock, nothing switched -- the move's and the carry's first contact
    # with every pre-rule generation
    cur = r3.content_sha8
    os.remove(f"{gd}/SEALED-{cur}.json")
    os.remove(f"{gd}/PUBLISHED-{cur}.json")
    for d in _dirs(f"{gd}/{cur}"):
        os.chmod(d, 0o755)
    rn4 = Runner()
    r4a = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn4, note="adopting")
    assert r4a.outcome == "current_unrecorded" and not r4a.ok and r4a.outcome not in ST.PUSH_OK_OUTCOMES and "--adopt-unrecorded" in r4a.detail["why"], r4a.as_dict()
    assert r4a.detail["unrecorded_generation"] == cur and not os.path.exists(f"{gd}/PUBLISHED-{cur}.json") and _mode(f"{gd}/{cur}") == 0o755, "nothing adopted implicitly"
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn4.calls) and os.readlink(f"{gd}/current") == cur
    r4 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="adopting", adopt_unrecorded=True)
    assert r4.outcome == "unchanged" and set(r4.detail["adopted"]) == {"sealed", "published_record"}, r4.as_dict()
    assert all(_mode(d) == 0o555 for d in _dirs(f"{gd}/{cur}"))
    pub4 = json.load(open(f"{gd}/PUBLISHED-{cur}.json"))
    assert pub4["mode"] == "adopt" and pub4["previous"] is None and pub4["changed_files"] == 4 and pub4["change_utc"] == _fmt(t0 + 9_000_000_000)
    assert json.load(open(f"{gd}/SEALED-{cur}.json"))["n_dirs"] == 6
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["generation"]["adopted"]["published_record"]["mode"] == "adopt"
    # only the seal missing -> only the seal adopted; the record stands
    os.remove(f"{gd}/SEALED-{cur}.json")
    r5 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r5.outcome == "unchanged" and set(r5.detail["adopted"]) == {"sealed"} and json.load(open(f"{gd}/PUBLISHED-{cur}.json"))["note"] == "adopting"
    # --from DIR publishes another tree under the component's name; the record says where the bytes came from
    other = str(w["root"]) + "-other"
    import shutil
    shutil.copytree(base, other, symlinks=True)
    _write(os.path.join(other, "top.txt"), b"other top\n")
    r6 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), from_dir=other, note="from other")
    assert r6.outcome == "published" and json.load(open(f"{gd}/PUBLISHED-{r6.content_sha8}.json"))["source"] == other
    assert json.load(open(f"{gd}/PUBLISHED-{r6.content_sha8}.json"))["declared_base"] == base
    assert open(f"{gd}/{r6.content_sha8}/top.txt", "rb").read() == b"other top\n"


def test_push_refuses_what_cannot_be_pushed_from_the_grading_host_and_defers_inside_the_publish_interval(tmp_path):
    base = _canon(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    cfgp = _cfg(tmp_path, {
        "orig": {"base": base, "authority": "cluster_a_origin", "original": "exp/orig"},
        "clusterprod": {"base": base, "authority": "cluster_a", "producer": "cluster"},
        "paced": {"base": base, "authority": "cluster_a", "producer": "grading_host:sess", "min_publish_interval_s": 3600, "exclude": ["__pycache__", "/.pytest_cache"]},
        "frozen": {"base": base, "authority": "cluster_a", "producer": "frozen", "exclude": ["__pycache__", "/.pytest_cache"]},
    }, adapters={"relu_paired": {"shards": 1, "components": ["orig", "clusterprod", "paced", "frozen"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    r = ST.push_component(root, dest, cfg.components["orig"], runner=Runner())
    assert r.outcome == "refused" and "cluster_a_origin" in r.detail["reason"] and not r.ok
    r = ST.push_component(root, dest, cfg.components["clusterprod"], runner=Runner())
    assert r.outcome == "refused" and "producer cluster" in r.detail["reason"] and "--stores-dir" in r.detail["reason"]
    assert not os.path.exists(dest.shared), "a refusal touches nothing"
    # a frozen cluster_a component pushes (the one-time move); a grading host producer's component pushes from its grading_host_mirror (the base)
    r = ST.push_component(root, dest, cfg.components["frozen"], runner=Runner(), note="frozen move")
    assert r.outcome == "published" and json.load(open(f"{dest.gen_dir('frozen')}/PUBLISHED-{r.content_sha8}.json"))["producer"] == "frozen"
    r1 = ST.push_component(root, dest, cfg.components["paced"], runner=Runner())
    assert r1.outcome == "published"
    _write(os.path.join(base, "top.txt"), b"changed\n")
    rn = Runner()
    r2 = ST.push_component(root, dest, cfg.components["paced"], runner=rn)
    assert r2.outcome == "deferred" and r2.published == r1.content_sha8 and r2.detail["min_publish_interval_s"] == 3600, r2.as_dict()
    assert "s ago" in r2.detail["why"] and not any("rsync" in a or a[0] == "sbatch" for a in rn.calls)
    assert os.readlink(f"{dest.gen_dir('paced')}/current") == r1.content_sha8
    assert r2.outcome in ST.PUSH_OK_OUTCOMES, "deferred is not a failure: the producer's next push publishes"
    # past the interval (the record's published_utc moved back) the same push publishes
    gd = dest.gen_dir("paced")
    rec = json.load(open(f"{gd}/PUBLISHED-{r1.content_sha8}.json"))
    rec["published_utc"] = "2020-01-01T00:00:00.000Z"
    json.dump(rec, open(f"{gd}/PUBLISHED-{r1.content_sha8}.json", "w"))
    r3 = ST.push_component(root, dest, cfg.components["paced"], runner=Runner())
    assert r3.outcome == "published" and r3.content_sha8 != r1.content_sha8
    assert ST._age_s("garbage", R.utc_now()) is None


def test_a_dangling_current_is_re_copied_and_re_pointed_never_called_unchanged(world):
    """The live shape of 2026-10-04 (bigcodebench_full): `current -> <sha8>` while the tree is `.failed-<sha8>-...` -- code without the
    guard had switched the pointer onto a tree a verify-report misread had renamed. A push (or a mirror) of the same content re-copies,
    re-verifies and re-publishes it through the guard instead of calling it unchanged or failing on the seal of a missing tree. Since the
    dangling-pointer follow-ups (merged 2026-10-06) the repair is its own green outcome, `republished_dangling_current`, and the renamed
    trees beside the pointer are named on the record."""
    w = world
    r1 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="first")
    gd, sha8 = w["gen_dir"], r1.content_sha8
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261004T020147011Z")
    assert os.readlink(f"{gd}/current") == sha8 and not os.path.isdir(f"{gd}/{sha8}")
    assert ST.check_authority(w["root"], w["cfg"], w["dest"], runner=Runner())["components"]["relu_paired_run_state"]["problems"] == ["dangling"]
    rn = Runner()
    r2 = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn, note="again")
    assert r2.outcome == "republished_dangling_current" and r2.ok and r2.content_sha8 == sha8 and r2.detail["current_dangling"] is True, r2.as_dict()
    assert any(a.endswith(f".failed-{sha8}-20261004T020147011Z") for a in r2.detail["renamed_aside"]) and "beside it" in r2.detail["why_copied"], r2.detail
    assert any("rsync" in a for a in rn.calls) and any(a[0] == "sbatch" for a in rn.calls), "re-copied and re-verified, not adopted"
    rs = next(a for a in rn.calls if "rsync" in a)
    assert not any(a.startswith("--link-dest") for a in rs), "nothing hard-links against a tree that is not there"
    assert os.path.isdir(f"{gd}/{sha8}") and os.readlink(f"{gd}/current") == sha8 and all(_mode(d) == 0o555 for d in _dirs(f"{gd}/{sha8}"))
    pub = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    assert pub["note"] == "again" and pub["published_utc"] > json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["verified_utc"]
    assert ST.check_authority(w["root"], w["cfg"], w["dest"], runner=Runner())["ok"] is True
    # a verified report with no tree beside it is never `published_existing_verified` either (the guard would refuse the switch)
    os.remove(f"{gd}/current")
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-again")
    r3 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r3.outcome == "published" and os.path.isdir(f"{gd}/{sha8}") and os.readlink(f"{gd}/current") == sha8 and "current_dangling" not in r3.detail
    # a plain mirror meeting a dangling pointer re-copies too
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-third")
    r4 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r4.outcome == "republished_dangling_current" and r4.detail["current_dangling"] is True and os.path.isdir(f"{gd}/{sha8}")


# --------------------------------------------------------------------------------------------
# check-authority: one batch, every current, the problems by name
# --------------------------------------------------------------------------------------------


def test_check_authority_reads_every_component_in_one_batch_and_names_the_problems(tmp_path):
    base = _canon(tmp_path)
    mirror_dir = str(tmp_path / "mirror_of_e")
    os.makedirs(mirror_dir)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    excl = ["__pycache__", "/.pytest_cache"]
    comps = {"ok_c": {"base": base, "authority": "grading_host_copy", "exclude": excl},
             "dangling_c": {"base": base, "authority": "grading_host_copy", "exclude": excl},
             "unverified_c": {"base": base, "authority": "grading_host_copy", "exclude": excl},
             "never_c": {"base": base, "authority": "grading_host_copy", "exclude": excl},
             "bulb_none": {"base": base, "authority": "cluster_a", "producer": "grading_host:sess", "grading_host_mirror": mirror_dir, "transitional_carry": True,
                           "carry_deadline": "2020-01-01"},
             "bulb_nomirror": {"base": base, "authority": "cluster_a", "producer": "cluster", "grading_host_mirror": str(tmp_path / "absent")},
             "orig": {"base": base, "authority": "cluster_a_origin", "original": "exp/orig"}}
    cfgp = _cfg(tmp_path, comps, adapters={"relu_paired": {"shards": 1, "components": list(comps)}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    shas = {}
    for slug in ("ok_c", "dangling_c", "unverified_c"):
        r = ST.push_component(root, dest, cfg.components[slug], runner=Runner(), note=slug)
        assert r.outcome == "published", r.as_dict()
        shas[slug] = r.content_sha8
    gd_d = dest.gen_dir("dangling_c")
    os.rename(f"{gd_d}/{shas['dangling_c']}", f"{gd_d}/.failed-{shas['dangling_c']}-x")
    gd_u = dest.gen_dir("unverified_c")
    os.remove(f"{gd_u}/VERIFIED-{shas['unverified_c']}.json")
    os.remove(f"{gd_u}/SEALED-{shas['unverified_c']}.json")
    rn = Runner()
    rec = ST.check_authority(root, cfg, dest, runner=rn)
    batches = [a for a in rn.calls if a[:2] == ["sh", "-c"]]
    assert len(batches) == 1, "ONE batch on the destination for every component"
    assert rec["schema"] == ST.AUTHORITY_STATE_SCHEMA and rec["ok"] is False and rec["n_components"] == 7
    c = rec["components"]
    assert c["ok_c"]["state"] == "ok" and c["ok_c"]["problems"] == [] and c["ok_c"]["current"] == shas["ok_c"] and c["ok_c"]["sealed"] and c["ok_c"]["verified"]
    assert c["ok_c"]["published_record"] and c["ok_c"]["change_utc"] and c["ok_c"]["publish_mode"] == "push" and c["ok_c"]["real"] == os.path.realpath(f"{dest.gen_dir('ok_c')}/{shas['ok_c']}")
    assert c["dangling_c"]["problems"] == ["dangling"] and c["dangling_c"]["exists"] is False and c["dangling_c"]["current"] == shas["dangling_c"]
    assert c["unverified_c"]["problems"] == ["unverified", "unsealed"] and c["unverified_c"]["exists"] is True and c["unverified_c"]["published_record"] is True
    assert c["never_c"]["problems"] == [] and c["never_c"]["notes"] == ["never_published"] and c["never_c"]["current"] is None and c["never_c"]["state"] == "ok"
    assert c["bulb_none"]["problems"] == ["no_current"] and c["bulb_none"]["grading_host_mirror_present"] is True and c["bulb_none"]["carry_overdue"] is True
    assert "carry_overdue" in c["bulb_none"]["notes"]
    assert c["bulb_nomirror"]["problems"] == ["no_current", "no_grading_host_mirror"] and c["bulb_nomirror"]["grading_host_mirror_present"] is False
    assert c["orig"]["state"] == "not_applicable" and c["orig"]["problems"] == []
    assert rec["problems"] == ["bulb_nomirror: no_current, no_grading_host_mirror", "bulb_none: no_current", "dangling_c: dangling", "unverified_c: unverified, unsealed"]
    saved = json.load(open(os.path.join(root, "state", ST.AUTHORITY_STATE_FILE)))
    assert saved["problems"] == rec["problems"] and saved["checked_utc"] == rec["checked_utc"]
    text = ST.format_authority(rec)
    assert "check-authority NOT OK" in text and "dangling_c | grading_host_copy | " + shas["dangling_c"] + " | no | yes | yes | yes" in text, text
    # a grading_host_copy component published by the OLD flow (no record) is `unrecorded` -- a PROBLEM: every publisher records inside the lock
    # now, so an unrecorded `current` is a publisher mid-flight, a crash, or a pre-rule generation the cutover's one-time
    # `stores mirror --adopt-unrecorded` has not adopted; so are its unsealed directories
    os.remove(f"{dest.gen_dir('ok_c')}/PUBLISHED-{shas['ok_c']}.json")
    rec2 = ST.check_authority(root, cfg, dest, runner=Runner())
    assert rec2["components"]["ok_c"]["problems"] == ["unrecorded"] and rec2["components"]["ok_c"]["notes"] == [] and rec2["ok"] is False
    assert "ok_c: unrecorded" in rec2["problems"]
    os.remove(f"{dest.gen_dir('ok_c')}/SEALED-{shas['ok_c']}.json")
    rec3 = ST.check_authority(root, cfg, dest, runner=Runner())
    assert rec3["components"]["ok_c"]["problems"] == ["unsealed", "unrecorded"]
    # a clean configuration is ok
    cfg_ok = ST.load_config(_cfg(tmp_path, {"ok2": {"base": base, "authority": "grading_host_copy", "exclude": excl}}, adapters={"relu_paired": {"shards": 1, "components": ["ok2"]}}, name="ok.yaml"))
    assert ST.push_component(root, cfg_ok.destination("cluster_a"), cfg_ok.components["ok2"], runner=Runner()).outcome == "published"
    rec4 = ST.check_authority(root, cfg_ok, cfg_ok.destination("cluster_a"), runner=Runner())
    assert rec4["ok"] is True and rec4["problems"] == [] and "check-authority OK" in ST.format_authority(rec4)


# --------------------------------------------------------------------------------------------
# verify-authority: the one-time move's evidence
# --------------------------------------------------------------------------------------------


def test_verify_authority_writes_the_evidence_adopts_an_equal_generation_and_names_a_diff(world, tmp_path):
    w = world
    out = str(w["root"]) + "-move"
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    # an OLD-flow generation: unsealed directories, no SEALED, no PUBLISHED (a mirror records its publication now; the old one did not)
    os.remove(f"{gd}/SEALED-{sha8}.json")
    os.remove(f"{gd}/PUBLISHED-{sha8}.json")
    for d in _dirs(f"{gd}/{sha8}"):
        os.chmod(d, 0o755)
    live_verified_mtime = os.stat(f"{gd}/VERIFIED-{sha8}.json").st_mtime_ns
    rn = Runner()
    rec = ST.verify_authority(w["root"], w["dest"], w["comp"], out, runner=rn)
    assert rec["state"] == "verified", rec
    assert rec["current"] == sha8 and rec["sealed_now"] is True and rec["published_now"] is True and rec["unsealed_after"] == 0
    assert rec["differing"]["counts"] == {"only_grading_host": 0, "only_original": 0, "sha_or_size_differs": 0, "symlinks": 0}
    for suffix in (".grading_host.sha256", ".cluster_a.sha256", ".grading_host.json", ".verified.json"):
        assert os.path.isfile(os.path.join(out, "relu_paired_run_state" + suffix)), suffix
    assert open(os.path.join(out, "relu_paired_run_state.grading_host.sha256")).read() == open(os.path.join(out, "relu_paired_run_state.cluster_a.sha256")).read()
    saved = json.load(open(os.path.join(out, "relu_paired_run_state.verified.json")))
    assert saved["state"] == "verified" and saved["report"]["hashed"] == 4 and saved["published_record"]["mode"] == "move" and saved["job_id"]
    # the verify job re-hashed the CURRENT generation and wrote its report under <stores>/move/<slug>/, never over the live VERIFIED file
    sb = next(a for a in rn.calls if a[0] == "sbatch")
    args = sb[sb.index(w["dest"].sbatch_path) + 1:]
    assert args[0] == "verify" and args[1] == f"{gd}/{sha8}" and args[2] == f"{gd}/MANIFEST-{sha8}.sha256" and args[3].startswith(f"{w['dest'].stores_dir}/move/relu_paired_run_state/VERIFIED-{sha8}-")
    assert os.stat(f"{gd}/VERIFIED-{sha8}.json").st_mtime_ns == live_verified_mtime and os.path.isfile(args[3])
    assert all(_mode(d) == 0o555 for d in _dirs(f"{gd}/{sha8}")) and json.load(open(f"{gd}/SEALED-{sha8}.json"))["sealed_by"].startswith("verify-authority")
    pub = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    assert pub["mode"] == "move" and pub["changed_files"] == 4 and pub["previous"] is None
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["move"]["state"] == "verified" and st["last_run"]["outcome"] == "verify-authority:verified"
    # the stamp is the move's, not a mirror outcome: green to `last_run_level` and to the status page's authority row (read through
    # OUTCOME_LEVELS instead, it showed the just-verified store red)
    assert ST.last_run_level(st["last_run"]["outcome"]) == "green"
    from sisteer.registry import status as STATUS
    so = STATUS.slurm_only_section(w["root"], state_dir=str(tmp_path / "landing"))
    row = next(r for r in so["authority"]["rows"] if r["slug"] == "relu_paired_run_state")
    assert row["mirror_outcome"] == "verify-authority:verified" and row["mirror_level"] == "green" and row["level"] == "green"
    assert not any(p.startswith("store mirror") for p in so["problems"]), so["problems"]
    # a second run: still verified (the re-hash is the proof each time), nothing adopted again
    rn2 = Runner()
    rec2 = ST.verify_authority(w["root"], w["dest"], w["comp"], out, runner=rn2)
    assert rec2["state"] == "verified" and "sealed_now" not in rec2 and "published_now" not in rec2 and any(a[0] == "sbatch" for a in rn2.calls)
    assert json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))["published_utc"] == pub["published_utc"]
    # the grading host copy moves on: unequal, named by file, no job spent, the stale verified.json retired beside a diff
    _write(os.path.join(w["base"], "top.txt"), b"moved on\n")
    rn3 = Runner()
    rec3 = ST.verify_authority(w["root"], w["dest"], w["comp"], out, runner=rn3)
    assert rec3["state"] == "unequal" and rec3["differing"]["sha_or_size_differs"] == ["top.txt"] and "push" in rec3["why"], rec3
    assert not any(a[0] == "sbatch" for a in rn3.calls)
    assert os.path.isfile(os.path.join(out, "relu_paired_run_state.diff.json")) and not os.path.exists(os.path.join(out, "relu_paired_run_state.verified.json"))
    assert rec3["stale_verified_renamed"] is True and any(n.startswith("relu_paired_run_state.verified.json.stale-") for n in os.listdir(out))
    st3 = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st3["last_run"]["outcome"] == "verify-authority:unequal" and ST.last_run_level(st3["last_run"]["outcome"]) == "red"
    so = STATUS.slurm_only_section(w["root"], state_dir=str(tmp_path / "landing"))
    row = next(r for r in so["authority"]["rows"] if r["slug"] == "relu_paired_run_state")
    assert row["mirror_level"] == "red" and row["level"] == "red" and any("relu_paired_run_state: verify-authority:unequal" in p for p in so["problems"]), so["problems"]
    # pushed, then equal again
    assert ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner()).outcome == "published"
    rec4 = ST.verify_authority(w["root"], w["dest"], w["comp"], out, runner=Runner())
    assert rec4["state"] == "verified" and rec4["current"] != sha8 and "sealed_now" not in rec4 and "published_now" not in rec4
    # nothing published -> no_current; a cluster_a_origin component -> not applicable
    cfg2 = ST.load_config(_cfg(tmp_path, {"fresh": {"base": w["base"], "authority": "grading_host_copy", "exclude": ["__pycache__", "/.pytest_cache"]},
                                          "orig": {"base": w["base"], "authority": "cluster_a_origin", "original": "exp/o"}},
                               adapters={"relu_paired": {"shards": 1, "components": ["fresh", "orig"]}}, name="two.yaml"))
    r5 = ST.verify_authority(w["root"], cfg2.destination("cluster_a"), cfg2.components["fresh"], out, runner=Runner())
    assert r5["state"] == "no_current" and "push" in r5["why"]
    assert ST.verify_authority(w["root"], cfg2.destination("cluster_a"), cfg2.components["orig"], out, runner=Runner())["state"] == "not_applicable"
    comps = ST.load_state(w["root"])["components"]
    assert comps["fresh"]["last_run"]["outcome"] == "verify-authority:no_current" and ST.last_run_level(comps["fresh"]["last_run"]["outcome"]) == "red"
    assert "orig" not in comps, "a not_applicable component returns before any state is stamped (its row is `stores check`'s)"
    assert ST.last_run_level("verify-authority:not_applicable") == "green"


def test_verify_authority_refuses_when_current_or_the_source_moves_while_the_job_runs(world):
    """The pairing (grading-host manifest, `current`) was compared only BEFORE the verify job, which can run for a long time, and the
    certificate was written without looking again -- a producer publishing a new generation meanwhile, or the grading host's mirror
    moving, left a `verified` certificate attesting a pairing that no longer stood, and the grading host's lock cannot stop an on-cluster
    producer. Now `current` is re-read after the job and, right before the certificate is filed, `current` AND the source are re-read
    against the start."""
    w = world
    out = str(w["root"]) + "-move"
    r1 = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner())
    gd, sha8 = w["gen_dir"], r1.content_sha8
    rec0 = ST.verify_authority(w["root"], w["dest"], w["comp"], out, runner=Runner())
    assert rec0["state"] == "verified" and [c["stage"] for c in rec0["rechecks"]] == ["after_verify", "before_certificate"], rec0
    assert rec0["rechecks"][1]["current"] == sha8 and rec0["rechecks"][1]["current_at_start"] == sha8
    assert rec0["rechecks"][1]["grading_host_content_sha8"] == sha8 == rec0["rechecks"][1]["grading_host_content_sha8_at_start"] and rec0["rechecks"][1]["grading_host_name_size_equal"]
    assert rec0["rechecks"][1]["grading_host_hashed"] == 0, "the re-manifest goes through the stat cache: nothing unchanged is read again"
    saved = json.load(open(os.path.join(out, "relu_paired_run_state.verified.json")))
    assert saved["rechecks"][1]["stage"] == "before_certificate", "the certificate carries the rechecks that admitted it"
    # the grading host copy moves on WHILE the job runs: the start comparison passed (a job was spent), the certificate is refused by name
    original = open(os.path.join(w["base"], "top.txt"), "rb").read()
    rn = Runner(before_verify=lambda gen: _write(os.path.join(w["base"], "top.txt"), b"moved during the job\n"))
    rec = ST.verify_authority(w["root"], w["dest"], w["comp"], out, runner=rn)
    assert rec["state"] == "source_changed", rec
    assert any(a[0] == "sbatch" for a in rn.calls), "the race is real: the start comparison passed and the job ran"
    last = rec["rechecks"][-1]
    assert last["stage"] == "before_certificate" and last["current"] == sha8 and last["grading_host_content_sha8"] != sha8 and last["grading_host_hashed"] == 1
    assert "changed while the verify job ran" in rec["why"] and "stores push" in rec["why"]
    assert not os.path.exists(os.path.join(out, "relu_paired_run_state.verified.json")) and rec["stale_verified_renamed"] is True
    assert json.load(open(os.path.join(out, "relu_paired_run_state.diff.json")))["state"] == "source_changed"
    assert ST.load_state(w["root"])["components"]["relu_paired_run_state"]["move"]["state"] == "source_changed"
    assert ST.load_state(w["root"])["components"]["relu_paired_run_state"]["last_run"]["outcome"] == "verify-authority:source_changed"
    assert ST.last_run_level("verify-authority:source_changed") == "red"
    _write(os.path.join(w["base"], "top.txt"), original)
    # a publisher switches `current` WHILE the job runs: refused before anything is adopted; `current` stays where the publisher put it
    os.remove(f"{gd}/PUBLISHED-{sha8}.json")

    def other_publisher(gen):
        os.makedirs(f"{gd}/0ther0ne", exist_ok=True)
        os.remove(f"{gd}/current")
        os.symlink("0ther0ne", f"{gd}/current")

    rn2 = Runner(before_verify=other_publisher)
    rec2 = ST.verify_authority(w["root"], w["dest"], w["comp"], out, runner=rn2)
    assert rec2["state"] == "current_moved" and any(a[0] == "sbatch" for a in rn2.calls), rec2
    assert [c["stage"] for c in rec2["rechecks"]] == ["after_verify"] and rec2["rechecks"][0]["current"] == "0ther0ne" and rec2["rechecks"][0]["current_at_start"] == sha8
    assert "moved from " + sha8 + " to 0ther0ne" in rec2["why"]
    assert os.readlink(f"{gd}/current") == "0ther0ne" and not os.path.exists(f"{gd}/PUBLISHED-{sha8}.json"), "nothing adopted for a generation no longer current"
    assert "published_now" not in rec2 and "sealed_now" not in rec2 and not os.path.exists(os.path.join(out, "relu_paired_run_state.verified.json"))
    assert json.load(open(os.path.join(out, "relu_paired_run_state.diff.json")))["state"] == "current_moved"
    # the pairing restored: verified again, the record adopted again, the certificate back
    os.remove(f"{gd}/current")
    os.symlink(sha8, f"{gd}/current")
    rec3 = ST.verify_authority(w["root"], w["dest"], w["comp"], out, runner=Runner())
    assert rec3["state"] == "verified" and rec3["published_now"] is True and os.path.isfile(os.path.join(out, "relu_paired_run_state.verified.json")), rec3


def test_publish_is_a_compare_and_switch_with_a_unique_temporary_pointer(world):
    """A publisher that read `current` before a long copy and verify must not switch it back onto its own, older snapshot after
    another publisher moved it, and two publishers must never share one `.current.tmp`. The switch re-reads `current` in its own shell
    step and refuses by name (`current_moved`) when it is not what this run read; the temporary pointer is unique per publication."""
    w = world
    rn = Runner()
    r1 = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn, note="first")
    gd, sha1 = w["gen_dir"], r1.content_sha8
    assert r1.outcome == "published", r1.as_dict()
    switch = next(a[2] for a in rn.calls if a[:2] == ["sh", "-c"] and "mv -T" in a[2] and "current" in a[2])
    assert ".current.tmp" not in switch and re.search(r"\.current\.[0-9a-f]{12}\.tmp", switch) and "readlink current" in switch, switch
    assert not any(n.startswith(".current") for n in os.listdir(gd)), "no temporary pointer left behind"
    # the step itself: `current` must still be what the caller read (None = there must be none)
    remote = ST.Remote(w["dest"], Runner())
    with pytest.raises(ST.CurrentMoved, match="current_moved.*is " + sha1 + ", not not-what-this-run-read"):
        remote.publish(gd, sha1, _rec_text(sha1), expect_current="not-what-this-run-read")
    with pytest.raises(ST.CurrentMoved, match="not absent as this run read it"):
        remote.publish(gd, sha1, _rec_text(sha1), expect_current=None)
    assert os.readlink(f"{gd}/current") == sha1 and not any(n.startswith(".current") or n.startswith(".PUBLISHED") for n in os.listdir(gd))
    remote.publish(gd, sha1, _rec_text(sha1, previous=sha1), expect_current=sha1)
    assert os.readlink(f"{gd}/current") == sha1
    # the race: this run reads current = sha1 and copies + verifies its snapshot; meanwhile another publisher (simulated inside the verify
    # step) switches `current` onto its own generation -> this run refuses by name and leaves `current` where the publisher put it
    _write(os.path.join(w["base"], "top.txt"), b"second\n")

    def other_publisher(gen):
        os.makedirs(f"{gd}/0ther0ne", exist_ok=True)
        os.remove(f"{gd}/current")
        os.symlink("0ther0ne", f"{gd}/current")

    rn2 = Runner(before_verify=other_publisher)
    r2 = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn2, note="second")
    sha2 = r2.content_sha8
    assert r2.outcome == "current_moved" and not r2.ok and r2.outcome not in ST.PUSH_OK_OUTCOMES, r2.as_dict()
    assert r2.published == "0ther0ne" and r2.detail["current_read"] == sha1 and "another publisher switched it meanwhile" in r2.detail["why"]
    assert os.readlink(f"{gd}/current") == "0ther0ne", "never switched back onto this run's snapshot"
    assert os.path.isdir(f"{gd}/{sha2}") and os.path.isfile(f"{gd}/VERIFIED-{sha2}.json") and os.path.isfile(f"{gd}/SEALED-{sha2}.json")
    assert all(_mode(d) == 0o555 for d in _dirs(f"{gd}/{sha2}")), "the verified tree stays, sealed"
    assert not os.path.exists(f"{gd}/PUBLISHED-{sha2}.json"), "no publication record for a generation that is not current"
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["generation"]["state"] == "current_moved" and st["generation"]["current_now"] == "0ther0ne" and st["published"] == "0ther0ne"
    assert st["last_run"]["outcome"] == "current_moved"
    # a later run reads the NEW `current`: while that generation has no publication record (its publisher mid-flight, or crashed between
    # its switch and its record) the run refuses by name and re-points nothing; once the record stands, it finds its own content verified
    # and sealed and publishes it through the same guard
    rn3a = Runner()
    r3a = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn3a, note="third")
    assert r3a.outcome == "current_unrecorded" and not r3a.ok and "0ther0ne" in r3a.detail["why"] and r3a.detail["unrecorded_generation"] == "0ther0ne", r3a.as_dict()
    assert os.readlink(f"{gd}/current") == "0ther0ne" and not os.path.exists(f"{gd}/PUBLISHED-{sha2}.json")
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn3a.calls), "nothing copied for a `current` nobody has recorded"
    _write(f"{gd}/PUBLISHED-0ther0ne.json", _rec_text("0ther0ne", previous=sha1).encode())
    rn3 = Runner()
    r3 = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn3, note="third")
    assert r3.outcome == "published_existing_verified" and r3.content_sha8 == sha2 and os.readlink(f"{gd}/current") == sha2, r3.as_dict()
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn3.calls), "nothing re-copied or re-verified"
    assert json.load(open(f"{gd}/PUBLISHED-{sha2}.json"))["previous"] == "0ther0ne"
    # `current` present but not a symlink is refused by name in the same step
    os.remove(f"{gd}/current")
    open(f"{gd}/current", "w").close()
    with pytest.raises(ST.StoresError, match="exists but is not a symlink"):
        remote.publish(gd, sha2, _rec_text(sha2), expect_current=sha2)
    os.remove(f"{gd}/current")


def test_overlapping_on_cluster_pushes_are_serialized_at_the_destination_so_the_later_snapshot_stays_current(tmp_path, monkeypatch):
    """Two overlapping producer jobs read the same `current`, copy and verify different snapshots and finish in reverse order --
    the older, slower one would switch `current` back after the newer one, and both wrote `.current.tmp`. The slug's publish lock on the
    destination, taken BEFORE the snapshot and held through the switch, orders them: the second job takes its snapshot only after the first
    has published, so `current` moves forward in snapshot order. A publisher that cannot wait is `lock_busy` by name and writes nothing."""
    stores = str(tmp_path / "shared" / "registry" / "stores")
    gd = f"{stores}/gen/drill"
    older, newer = str(tmp_path / "t1"), str(tmp_path / "t2")
    for src, top in ((older, b"v1\n"), (newer, b"v2\n")):
        _write(os.path.join(src, "a/f.json"), b"{}\n")
        _write(os.path.join(src, "top.txt"), top)
    a_in_verify, release_a = threading.Event(), threading.Event()
    real_verify = ST.verify_generation

    def slow_verify(gen_path, manifest_path, **kw):
        if open(os.path.join(gen_path, "top.txt"), "rb").read() == b"v1\n":
            a_in_verify.set()
            assert release_a.wait(30), "the test never released the slow publisher"
        return real_verify(gen_path, manifest_path, **kw)

    monkeypatch.setattr(ST, "verify_generation", slow_verify)
    results, logs = {}, {"A": [], "B": []}

    def run(name, src):
        results[name] = ST.push_on_cluster("drill", src, stores, note=name, session=f"job{name}", workers=2, lock_wait_s=60, log=logs[name].append)

    ta = threading.Thread(target=run, args=("A", older))
    ta.start()
    assert a_in_verify.wait(30), "A never reached its verify"
    tb = threading.Thread(target=run, args=("B", newer))
    tb.start()
    deadline = time.monotonic() + 20
    while not any("waiting for the publisher" in line for line in logs["B"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert any("waiting for the publisher" in line and "session=jobA" in line for line in logs["B"]), logs["B"]
    assert not os.path.lexists(f"{gd}/current"), "A has not published yet"
    sdir = ST.component_state_dir(os.path.join(stores, "push_state", "drill"), "drill")
    assert len([n for n in os.listdir(sdir) if n.endswith(".sha256")]) == 1, "B has not even taken its snapshot: the lock precedes the manifest build"
    release_a.set()
    ta.join(60)
    tb.join(60)
    ra, rb = results["A"], results["B"]
    assert ra.outcome == "published" and rb.outcome == "published", (ra.as_dict(), rb.as_dict())
    assert ra.content_sha8 != rb.content_sha8
    assert os.readlink(f"{gd}/current") == rb.content_sha8, "the later snapshot is current: the slower, older job did not switch it back"
    assert open(f"{gd}/{rb.content_sha8}/top.txt", "rb").read() == b"v2\n"
    pa, pb = (json.load(open(f"{gd}/PUBLISHED-{s}.json")) for s in (ra.content_sha8, rb.content_sha8))
    assert pa["previous"] is None and pb["previous"] == ra.content_sha8 and pb["published_utc"] > pa["published_utc"]
    assert ra.detail["publish_lock"]["waited_for"] is None and rb.detail["publish_lock"]["waited_s"] > 0 and "session=jobA" in rb.detail["publish_lock"]["waited_for"]
    assert open(f"{gd}/{ST.PUBLISH_LOCK_FILE}").read().startswith(rb.detail["publish_lock"]["holder"]), "the lock file names its last holder"
    assert not any(n.startswith(".current") or n.startswith(".incoming") for n in os.listdir(gd))
    # a publisher that cannot wait: lock_busy by name, nothing copied, exit 1 through the stdlib entry point too
    before = sorted(os.listdir(gd))
    with ST.PublishLock(gd, wait_s=0, holder="TESTHOST:1 job=99 session=held"):
        r = ST.push_on_cluster("drill", newer, stores, workers=2, lock_wait_s=0.2)
        assert r.outcome == "lock_busy" and not r.ok and r.outcome not in ST.PUSH_OK_OUTCOMES and "held by TESTHOST:1 job=99" in r.detail["why"], r.as_dict()
        p = _run_S("push", "drill", "--from", newer, "--stores-dir", stores, "--workers", "2", "--lock-wait-s", "0.2")
        assert p.returncode == 1 and p.stdout.startswith("drill: lock_busy") and "TESTHOST:1" in p.stdout, (p.stdout, p.stderr)
    assert sorted(os.listdir(gd)) == before, "a refused publisher wrote nothing into the generation directory"
    assert ST.load_state(os.path.join(stores, "push_state", "drill"))["components"]["drill"]["last_run"]["outcome"] == "lock_busy"
    assert ST.push_on_cluster("drill", newer, stores, workers=2, lock_wait_s=5).outcome == "unchanged"


# --------------------------------------------------------------------------------------------
# the name escape (R1: VAST refuses ':')
# --------------------------------------------------------------------------------------------


def _colon_tree(tmp_path):
    base = tmp_path / "colon"
    _write(str(base / "d:x" / "sub%y" / "f:1.txt"), b"one\n")
    _write(str(base / "d:x" / "plain_in_escaped.txt"), b"two\n")
    _write(str(base / "d:x" / "__pycache__" / "j.pyc"), b"pyc")
    _write(str(base / "100%done" / "g.txt"), b"three\n")
    _write(str(base / "plain" / "h.txt"), b"four\n")
    _write(str(base / "top.txt"), b"top\n")
    os.symlink("d:x/sub%y/f:1.txt", str(base / "link:to"))                       # an escaped name AND an escaped target
    os.symlink("../d:x/plain_in_escaped.txt", str(base / "plain" / "into_colon"))   # a plain name, an escaped (climbing) target
    os.symlink("top.txt", str(base / "plainlink"))
    return str(base)


def test_name_escape_is_reversible_recorded_in_the_manifest_and_carried_by_tar_under_escaped_names(tmp_path):
    for n in ("a:b", "a%3Ab", "a:%b", "100%done", "plain", "em-x::y", "%", ":", "%25", "a%253Ab", "%3A", "x%:y"):
        assert ST.unescape_name(ST.escape_name(n)) == n, n
    assert ST.escape_name("a:b") == "a%3Ab" and ST.escape_name("a%3Ab") == "a%253Ab" and ST.escape_name("plain") == "plain"
    assert ST.escape_rel("m/em8q/em-x::y/f.json") == "m/em8q/em-x%3A%3Ay/f.json" and ST.unescape_rel("m/em8q/em-x%3A%3Ay/f.json") == "m/em8q/em-x::y/f.json"
    assert ST.escape_tops(["a/b:c/d/e:f", "x:y", "a/b:c/g", "p/q", "a/b:c"]) == ["a/b:c", "x:y"]
    assert ST.rel_needs_escape("p/q") is False and ST.rel_needs_escape("p/100%") is True
    base = _colon_tree(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    cfgp = _cfg(tmp_path, {"colon": {"base": base, "authority": "grading_host_copy", "exclude": ["__pycache__"]}}, adapters={"relu_paired": {"shards": 1, "components": ["colon"]}})
    cfg = ST.load_config(cfgp)
    dest, comp = cfg.destination("cluster_a"), cfg.components["colon"]
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn)
    assert res.outcome == "published", res.as_dict()
    gd, sha8 = dest.gen_dir("colon"), res.content_sha8
    gen = f"{gd}/{sha8}"
    assert res.detail["name_escapes"] == {"n": 4, "tops": ["100%done", "d:x", "link:to"]}
    assert res.detail["target_escapes"] == {"n": 2, "links": ["link:to", "plain/into_colon"]}, "the links whose TARGETS the destination spells differently"
    assert res.detail["tar"] == {"n": 5, "rc": 0}, "the four escaped names plus the plain-named link with an escaped target (link:to is both)"
    # on disk: the escaped spelling; the manifest and every index: the true names
    on_disk = sorted(os.path.relpath(os.path.join(dp, f), gen) for dp, _d, fns in os.walk(gen) for f in fns)
    assert on_disk == ["100%25done/g.txt", "d%3Ax/plain_in_escaped.txt", "d%3Ax/sub%25y/f%3A1.txt", "link%3Ato", "plain/h.txt", "plain/into_colon",
                       "plainlink", "top.txt"]
    assert not os.path.exists(f"{gen}/d%3Ax/__pycache__"), "the tar pass carries exactly the walk's files: the excludes hold"
    # a symlink's TARGET is spelled on disk the way its referent is (the literal true target dangled beside the renamed file, and the
    # verify had passed it); the manifest keeps the true target, and the link OPENS
    assert os.readlink(f"{gen}/link%3Ato") == "d%3Ax/sub%25y/f%3A1.txt" and open(f"{gen}/link%3Ato", "rb").read() == b"one\n"
    assert os.readlink(f"{gen}/plain/into_colon") == "../d%3Ax/plain_in_escaped.txt" and open(f"{gen}/plain/into_colon", "rb").read() == b"two\n"
    assert os.readlink(f"{gen}/plainlink") == "top.txt", "a target the destination spells the same is untouched"
    lines = ST.read_manifest_lines(f"{gd}/MANIFEST-{sha8}.sha256")
    assert [rel for _s, _z, rel in lines] == ["100%done/g.txt", "d:x/plain_in_escaped.txt", "d:x/sub%y/f:1.txt", "plain/h.txt", "top.txt"]
    meta = json.load(open(f"{gd}/MANIFEST-{sha8}.json"))
    assert meta["name_escapes"] == {"100%done/g.txt": "100%25done/g.txt", "d:x/plain_in_escaped.txt": "d%3Ax/plain_in_escaped.txt",
                                    "d:x/sub%y/f:1.txt": "d%3Ax/sub%25y/f%3A1.txt", "link:to": "link%3Ato"}
    assert meta["n_name_escapes"] == 4 and "%3A" in meta["name_escape_rule"]
    assert meta["symlinks"] == {"link:to": "d:x/sub%y/f:1.txt", "plain/into_colon": "../d:x/plain_in_escaped.txt", "plainlink": "top.txt"}, "true targets"
    assert meta["target_escapes"] == {"link:to": "d%3Ax/sub%25y/f%3A1.txt", "plain/into_colon": "../d%3Ax/plain_in_escaped.txt"} and meta["n_target_escapes"] == 2
    assert ST.read_manifest_escapes(f"{gd}/MANIFEST-{sha8}.sha256") == meta["name_escapes"]
    ver = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert ver["ok"] is True and ver["hashed"] == 5 and ver["name_escapes"] == 4 and ver["missing"] == ver["corrupt"] == ver["extra"] == []
    assert ver["n_symlinks"] == 3 and ver["target_escapes"] == 2
    assert set(json.load(open(f"{gd}/DEST-STAT-{sha8}.json"))["files"]) == {rel for _s, _z, rel in lines} | {"link:to", "plain/into_colon", "plainlink"}, \
        "true names in the snapshot"
    assert sorted(ST._walk_plain_dir(gen)) == [rel for _s, _z, rel in lines]
    ad = ST.admit_generation(gd, sha8)
    assert ad.ok and ad.n_files == 5 and ad.n_symlinks == 3 and ad.hashed == 5, ad.as_dict()
    idx = ST.manifest_index(gd, sha8)
    assert idx.check_file("d:x/sub%y/f:1.txt") is None and idx.path_of("d:x/sub%y/f:1.txt") == f"{gen}/d%3Ax/sub%25y/f%3A1.txt"
    assert idx.check_file("100%done/g.txt") is None and idx.check("d%3Ax/sub%25y/f%3A1.txt", "0" * 64) is not None, "the index speaks true names only"
    cache = CR.GenerationHashCache([idx])
    assert cache.index_of(f"{gen}/d%3Ax/sub%25y/f%3A1.txt") == ("d:x/sub%y/f:1.txt", idx) and cache.index_of(f"{gen}/plain/h.txt") == ("plain/h.txt", idx)
    assert cache.index_of("/elsewhere/x") is None
    # a reader opening the link lands on the manifest entry of the referent: the hash cache maps the resolved (escaped) path back
    assert cache.index_of(os.path.realpath(f"{gen}/link%3Ato")) == ("d:x/sub%y/f:1.txt", idx)
    assert cache.index_of(os.path.realpath(f"{gen}/plain/into_colon")) == ("d:x/plain_in_escaped.txt", idx)
    # the guard against the pre-fix shape: a link carrying the LITERAL true target (what tar's `S` flag wrote) dangles beside its renamed
    # referent, so the verify names it corrupt and the admission refuses it -- never `ok` on a link no reader can open
    bad = str(tmp_path / "bad_gen")
    shutil.copytree(gen, bad, symlinks=True)
    for d in _dirs(bad):
        os.chmod(d, 0o755)
    os.remove(f"{bad}/link%3Ato")
    os.symlink("d:x/sub%y/f:1.txt", f"{bad}/link%3Ato")
    assert not os.path.exists(f"{bad}/link%3Ato"), "the literal true target dangles on the destination"
    rep, _snap = ST.verify_generation(bad, f"{gd}/MANIFEST-{sha8}.sha256", seal=False)
    assert rep["ok"] is False and rep["corrupt"] == ["link:to"] and rep["missing"] == rep["extra"] == [], rep
    os.chmod(gen, 0o755)                                   # the owner un-seals to tamper, and seals again (the admission names a lifted
    os.remove(f"{gen}/link%3Ato")                          # top mode itself since the final pass on 2c; every directory's mode is C5's)
    os.symlink("d:x/sub%y/f:1.txt", f"{gen}/link%3Ato")
    os.chmod(gen, 0o555)
    bad_ad = ST.admit_generation(gd, sha8)
    assert bad_ad.refusal == "content_changed" and bad_ad.mismatched == ["link:to"], bad_ad.as_dict()
    os.chmod(gen, 0o755)
    os.remove(f"{gen}/link%3Ato")
    os.symlink("d%3Ax/sub%25y/f%3A1.txt", f"{gen}/link%3Ato")
    os.chmod(gen, 0o555)
    assert ST.admit_generation(gd, sha8).ok
    # the copy: rsync excluded the three tops (the names it cannot create) and the plain-named link whose target it would spell wrong;
    # the tar pipeline carried them under transforms that rewrite member names AND symlink targets (the `s` flag)
    rs = next(a for a in rn.calls if "rsync" in a and "--files-from=-" not in a)
    assert "--exclude=/d:x" in rs and "--exclude=/100%done" in rs and "--exclude=/link:to" in rs and "--exclude=/plain/into_colon" in rs and "--exclude=__pycache__" in rs
    tar = next(a for a in rn.calls if a[:3] == ["bash", "-o", "pipefail"])
    assert "--no-recursion --null -T -" in tar[4] and "--transform=s/%/%25/gs" in tar[4] and "--transform=s/:/%3A/gs" in tar[4] and gen in tar[4]
    assert "/gS" not in tar[4], "no transform leaves symlink targets alone"
    assert "--delete" not in " ".join(tar) and "ssh" not in tar[4], "a local destination: one pipe, no ssh"
    # a second generation changing ONLY an escaped file: no forced rsync pass (the escaped set is never hard-linked), tar carries it again,
    # the unchanged files are hard links onto the previous generation and the escaped one is a fresh inode
    _write(os.path.join(base, "d:x/sub%y/f:1.txt"), b"ONE\n")
    rn2 = Runner()
    r2 = ST.mirror_component(root, dest, comp, runner=rn2)
    assert r2.outcome == "published" and r2.content_sha8 != sha8
    gen2 = f"{gd}/{r2.content_sha8}"
    assert not any("--files-from=-" in a for a in rn2.calls if "rsync" in a), "the only changed file is escaped: nothing to force"
    assert any(a.startswith("--link-dest=") for a in next(x for x in rn2.calls if "rsync" in x))
    assert os.stat(f"{gen2}/top.txt").st_ino == os.stat(f"{gen}/top.txt").st_ino
    assert os.stat(f"{gen2}/d%3Ax/sub%25y/f%3A1.txt").st_ino != os.stat(f"{gen}/d%3Ax/sub%25y/f%3A1.txt").st_ino
    assert open(f"{gen2}/d%3Ax/sub%25y/f%3A1.txt", "rb").read() == b"ONE\n" and ST.admit_generation(gd, r2.content_sha8).ok
    assert all(_mode(d) == 0o555 for d in _dirs(gen2))
    # a dry run names the escapes without writing
    dr = ST.mirror_component(root, dest, comp, runner=Runner(), dry_run=True)
    assert dr.outcome == "dry_run" and dr.detail["n_name_escapes"] == 4 and dr.detail["name_escape_tops"] == ["100%done", "d:x", "link:to"]
    assert dr.detail["n_target_escapes"] == 2 and dr.detail["target_escaped_links"] == ["link:to", "plain/into_colon"]
    # a remote destination spells the extraction over one ssh (the argv, not run)
    remote_dest = ST.Destination("cluster_a", "login-0", "/shared/u")
    argv = ST.tar_copy_argv(remote_dest, comp, "/shared/u/registry/stores/gen/colon/abc")
    assert argv[:4] == ["bash", "-o", "pipefail", "-c"] and "| ssh -o BatchMode=yes -o ConnectTimeout=15 login-0 " in argv[4] and "--transform=s/:/%3A/gs" in argv[4]
    # a tar failure is the step's outcome, by name
    class _TarFails(Runner):
        def __call__(self, argv, timeout_s, input=None, check=False):
            if argv[:3] == ["bash", "-o", "pipefail"]:
                return subprocess.CompletedProcess(argv, 2, stdout="", stderr="tar: boom")
            return super().__call__(argv, timeout_s, input=input, check=check)
    _write(os.path.join(base, "d:x/sub%y/f:1.txt"), b"TWO\n")
    r3 = ST.mirror_component(root, dest, comp, runner=_TarFails())
    assert r3.outcome == "tar_failed" and not r3.ok and r3.detail["tar"]["rc"] == 2 and "boom" in r3.detail["stderr"]
    assert os.readlink(f"{gd}/current") == r2.content_sha8


# --------------------------------------------------------------------------------------------
# --on-cluster: the producer's own job, stdlib only
# --------------------------------------------------------------------------------------------


def _run_S(*args, cwd="/"):
    return subprocess.run(["python3", "-S", "-m", "sisteer.registry.stores", *args], capture_output=True, text=True,
                          env={"PYTHONPATH": SRC, "PATH": os.environ["PATH"]}, cwd=cwd)


def test_on_cluster_push_is_stdlib_and_publishes_in_process_without_ssh_or_sbatch(tmp_path):
    src = _colon_tree(tmp_path)
    stores = str(tmp_path / "shared" / "registry" / "stores")
    r = _run_S("push", "drill", "--from", src, "--stores-dir", stores, "--note", "first", "--session", "prod", "--workers", "2", "--exclude", "__pycache__", "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["outcome"] == "published" and out["published_record"]["mode"] == "push_on_cluster" and out["published_record"]["note"] == "first"
    gd, sha8 = f"{stores}/gen/drill", out["content_sha8"]
    # the IMMUTABLE attestation of this (content, listings) pair stands beside the tree too (MANIFEST_RULES["directories"], merged 2026-10-06)
    keyed = os.path.basename(ST.attestation_json_path(f"{gd}/MANIFEST-{sha8}.sha256", json.load(open(f"{gd}/MANIFEST-{sha8}.json"))["listings_sha256"]))
    assert set(os.listdir(gd)) == {sha8, f"DEST-STAT-{sha8}.json", f"MANIFEST-{sha8}.json", keyed, f"MANIFEST-{sha8}.sha256", f"PUBLISHED-{sha8}.json",
                                   f"SEALED-{sha8}.json", f"VERIFIED-{sha8}.json", "current", ST.PUBLISH_LOCK_FILE}, \
        "no .incoming left; every record beside the tree; the slug's publish lock"
    assert os.readlink(f"{gd}/current") == sha8 and all(_mode(d) == 0o555 for d in _dirs(f"{gd}/{sha8}"))
    assert os.path.isfile(f"{gd}/{sha8}/d%3Ax/sub%25y/f%3A1.txt") and not os.path.exists(f"{gd}/{sha8}/d%3Ax/__pycache__")
    assert open(f"{gd}/{sha8}/link%3Ato", "rb").read() == b"one\n" and open(f"{gd}/{sha8}/plain/into_colon", "rb").read() == b"two\n", "the links resolve"
    pub = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    assert pub["producer"] == "cluster" and pub["authority"] == "cluster_a" and pub["producer_session"] == "prod" and pub["source"] == src and pub["n_name_escapes"] == 4
    assert pub["n_target_escapes"] == 2 and out["publish_lock"]["waited_for"] is None and "session=prod" in out["publish_lock"]["holder"]
    assert json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["ok"] is True
    assert os.path.isfile(f"{stores}/push_state/drill/state/store_mirror.json"), "the producer's stat cache and state live under the stores directory"
    assert ST.admit_generation(gd, sha8).ok
    # unchanged, then a change: the previous generation named, the incoming tree renamed into place
    r2 = _run_S("push", "drill", "--from", src, "--stores-dir", stores, "--workers", "2", "--exclude", "__pycache__")
    assert r2.returncode == 0 and r2.stdout.startswith("drill: unchanged " + sha8), (r2.stdout, r2.stderr)
    _write(os.path.join(src, "plain/h.txt"), b"FOUR\n")
    r3 = _run_S("push", "drill", "--from", src, "--stores-dir", stores, "--workers", "2", "--exclude", "__pycache__", "--json")
    assert r3.returncode == 0, r3.stderr
    out3 = json.loads(r3.stdout)
    assert out3["outcome"] == "published" and out3["published_record"]["previous"] == sha8 and out3["published_record"]["changed_files"] == 1
    assert not any(n.startswith(".incoming") or n.startswith(".stale") for n in os.listdir(gd))
    assert os.stat(f"{gd}/{out3['content_sha8']}/top.txt").st_ino == os.stat(f"{gd}/{sha8}/top.txt").st_ino, "--link-dest against the published generation"
    # a source that is not a directory refuses (exit 1), nothing published
    r4 = _run_S("push", "drill2", "--from", str(tmp_path / "nowhere"), "--stores-dir", stores)
    assert r4.returncode == 1 and "refused" in r4.stdout and not os.path.exists(f"{stores}/gen/drill2/current")
    # a stale unpublished tree at the generation's name is moved aside, never written into
    os.makedirs(f"{stores}/gen/drill3/")
    r5 = _run_S("push", "drill3", "--from", src, "--stores-dir", stores, "--workers", "2", "--exclude", "__pycache__", "--json")
    sha3 = json.loads(r5.stdout)["content_sha8"]
    os.remove(f"{stores}/gen/drill3/current")
    for d in _dirs(f"{stores}/gen/drill3/{sha3}"):
        os.chmod(d, 0o755)
    os.remove(f"{stores}/gen/drill3/VERIFIED-{sha3}.json")
    r6 = _run_S("push", "drill3", "--from", src, "--stores-dir", stores, "--workers", "2", "--exclude", "__pycache__", "--json")
    out6 = json.loads(r6.stdout)
    # an unpublished tree at the generation's name from an earlier attempt is renamed aside, never written into -- since the follow-up
    # merged 2026-10-06 as `.superseded-<sha8>-<stamp>`, the fresh copy's second link-dest
    assert r6.returncode == 0 and out6["outcome"] == "published" and out6["superseded"].startswith(f"{stores}/gen/drill3/.superseded-{sha3}-"), out6
    assert os.readlink(f"{stores}/gen/drill3/current") == sha3
    # verify_inline against a remote destination is refused before anything runs
    with pytest.raises(ST.StoresError, match="verify_inline runs where the tree is"):
        ST.mirror_component(str(tmp_path / "r"), ST.Destination("b", "login-0", "/w"), ST.Component("c", src, "cluster_a", producer="cluster"),
                            allow=("cluster_a",), verify_inline=True)


# --------------------------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------------------------


def test_cli_push_check_authority_and_verify_authority(world, tmp_path, monkeypatch, capsys):
    w = world
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "check-authority"]) == 0            # nothing published yet: a grading_host_copy component is never_published, not a problem
    assert "never_published" in capsys.readouterr().out
    assert cli.main(["stores", "push", "relu_paired_run_state", "--note", "cli push", "--session", "s1"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("relu_paired_run_state: published ")
    sha8 = out.split()[2]
    assert json.load(open(f"{w['gen_dir']}/PUBLISHED-{sha8}.json"))["note"] == "cli push"
    assert cli.main(["stores", "push", "relu_paired_run_state", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "unchanged"
    assert cli.main(["stores", "check-authority", "--json"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["ok"] is True and rec["components"]["relu_paired_run_state"]["current"] == sha8 and os.path.isfile(os.path.join(w["root"], "state", ST.AUTHORITY_STATE_FILE))
    move = os.path.join(w["root"], "move")
    assert cli.main(["stores", "verify-authority", "--component", "relu_paired_run_state", "--out", move]) == 0
    assert capsys.readouterr().out.startswith(f"relu_paired_run_state: verified current={sha8}")
    assert os.path.isfile(os.path.join(move, "relu_paired_run_state.verified.json"))
    assert cli.main(["stores", "verify-authority", "--component", "relu_paired_run_state"]) == 0      # the default --out under the root's state
    capsys.readouterr()
    assert os.path.isfile(os.path.join(w["root"], "state", ST.MOVE_SUBDIR, "relu_paired_run_state.verified.json"))
    # an unverified current fails check-authority, by name
    os.remove(f"{w['gen_dir']}/VERIFIED-{sha8}.json")
    assert cli.main(["stores", "check-authority"]) == 1
    assert "relu_paired_run_state: unverified" in capsys.readouterr().out
    # refusals and the lock
    assert cli.main(["stores", "push", "nope"]) == 2
    assert "not declared" in capsys.readouterr().err
    assert cli.main(["stores", "push", "relu_paired_run_state", "--on-cluster"]) == 2
    assert "--from DIR is required" in capsys.readouterr().err
    with ST.MirrorLock(w["root"]):
        assert cli.main(["stores", "push", "relu_paired_run_state"]) == 3
        assert "another run holds" in capsys.readouterr().err
        assert cli.main(["stores", "verify-authority", "--component", "relu_paired_run_state"]) == 3
        capsys.readouterr()
    # --on-cluster through the registry CLI: the stores directory from the (local) destination; no sbatch, no watcher
    src2 = _colon_tree(tmp_path / "oc")
    rn = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn)
    assert cli.main(["stores", "push", "drill", "--on-cluster", "--from", src2, "--workers", "2", "--lock-wait-s", "5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["outcome"] == "published" and out["published_record"]["mode"] == "push_on_cluster" and out["publish_lock"]["waited_for"] is None
    assert os.readlink(f"{w['dest'].stores_dir}/gen/drill/current") == out["content_sha8"]
    assert not any(a[0] == "sbatch" or a[0] == WATCH for a in rn.calls)


# --------------------------------------------------------------------------------------------
# one destination lock across EVERY publisher's compare-and-switch
# --------------------------------------------------------------------------------------------


class _OverSsh(Runner):
    """The grading-host-driven publisher's exact argv against a REMOTE-shaped destination (``host`` set), run here: every ``ssh OPTS host 'cmd'``
    step is split and run locally -- a shell step through ``shim`` first, the hook the interleaving tests use to pause the publish step at
    a named point -- and an rsync's ``-e ssh`` and ``host:`` target are stripped. The destination lock, the guard and the switch are then
    the real commands the grading host runs on the login node, against a local directory; mirror_component holds no PublishLock for its flow
    (the destination is not local), so the publish step is the one thing that takes the lock, as on cluster A."""

    def __init__(self, host, shim=None, **kw):
        super().__init__(**kw)
        self.host, self.shim, self.remote_steps, self.over_ssh = host, shim, [], []

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[0] == "ssh" and argv[-2] == self.host:
            inner = shlex.split(argv[-1])
            self.over_ssh.append(inner)
            if inner[:2] == ["sh", "-c"]:
                self.remote_steps.append(inner[2])
                if self.shim:
                    inner = ["sh", "-c", self.shim(inner[2])]
            return super().__call__(inner, timeout_s, input=input, check=check)
        if "rsync" in argv:
            if "-e" in argv:
                i = argv.index("-e")
                del argv[i:i + 2]
            if argv[-1].startswith(self.host + ":"):
                argv[-1] = argv[-1][len(self.host) + 1:]
        return super().__call__(argv, timeout_s, input=input, check=check)


class _Pause:
    """A shim that pauses the PUBLISH step right after ``anchor`` (a substring of the rendered step): the step touches ``paused`` and
    waits for ``go``, which the test writes when it has done what it wants done meanwhile."""

    def __init__(self, tmp, anchor):
        os.makedirs(tmp, exist_ok=True)
        self.anchor, self.paused, self.go, self.applied = anchor, os.path.join(tmp, "paused"), os.path.join(tmp, "go"), 0

    def __call__(self, cmd):
        if "mv -T" not in cmd or "current" not in cmd or self.anchor not in cmd:
            return cmd
        self.applied += 1
        return cmd.replace(self.anchor, f"{self.anchor} : > {shlex.quote(self.paused)}; while ! test -e {shlex.quote(self.go)}; do sleep 0.02; done;", 1)

    def wait_paused(self, seconds=30):
        deadline = time.monotonic() + seconds
        while not os.path.exists(self.paused) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert os.path.exists(self.paused), "the publish step never reached the pause"

    def release(self):
        open(self.go, "w").close()


AFTER_READLINK = "elif test -e current; then exit 67; fi;"     # inside the lock, between the reading and the rename
BEFORE_LOCK = "|| exit 60;"                                     # after `cd gen/<slug>`, before the flock


def _oc_source(tmp_path, name, top):
    """A tree an on-cluster job would publish into the slug (the same file names as the world's base, its own bytes)."""
    src = str(tmp_path / name)
    for rel in FILES:
        _write(os.path.join(src, rel), top if rel == "top.txt" else rel.encode() + b"\n")
    return src


@pytest.fixture
def shared(tmp_path):
    """A slug BOTH a grading-host-driven publisher and an on-cluster publisher may publish: `grading_host_copy` declared `producer: cluster` (the
    one declaration under which neither is refused -- `_push_refusal` refuses a cluster producer for `cluster_a` authority only, and the
    producer rule refuses a grading-host-driven switch over a cluster publication only when the component is NOT declared cluster-produced), so
    the interleaving tests exercise the lock alone; the producer rule has its own test below."""
    base = _canon(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    cfgp = _cfg(tmp_path, {"shared_slug": {"base": base, "authority": "grading_host_copy", "origin": "mixed", "producer": "cluster",
                                           "exclude": ["__pycache__", "/.pytest_cache"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    return {"base": base, "root": root, "cfg": cfg, "cfgp": cfgp, "dest": dest, "comp": cfg.components["shared_slug"], "slug": "shared_slug",
            "gen_dir": dest.gen_dir("shared_slug")}


def _run_in_thread(out, key, fn, *args, **kw):
    t = threading.Thread(target=lambda: out.__setitem__(key, fn(*args, **kw)))
    t.start()
    return t


def test_every_publisher_switches_current_inside_one_destination_lock(shared, tmp_path):
    """The grading-host-driven publisher read `current` and renamed in ONE shell step, but outside the `.publish.lock` the on-cluster
    publisher holds, so an on-cluster switch between its readlink and its rename was silently overwritten. Now the step takes the SAME
    lock (flock on gen/<slug>/.publish.lock, fd 9) before the readlink and holds it through the rename -- one helper renders the locked
    step, PublishLock names the same file by construction. Proved by interleaving, with the grading host's flow run against a
    remote-shaped destination and its publish step PAUSED by a shim: (a) after its readlink, inside the lock -- an on-cluster publisher
    that arrives meanwhile WAITS before its snapshot (nothing switches under the paused step), the step lands, and the on-cluster
    publication then lands on top of it, in snapshot order; (b) before the lock -- the on-cluster publisher switches `current` during the
    pause, and the step, reading `current` under the lock, refuses `current_moved` rather than overwriting."""
    w = shared
    gd, rdest = w["gen_dir"], replace(w["dest"], host="login-0")
    stores = rdest.stores_dir
    rn = _OverSsh("login-0")
    r1 = ST.push_component(w["root"], rdest, w["comp"], runner=rn, note="first")
    assert r1.outcome == "published", r1.as_dict()
    sha1 = r1.content_sha8
    assert any(i[0] == "sbatch" for i in rn.over_ssh) and len([i for i in rn.over_ssh if i[:2] == ["sh", "-c"]]) == len(rn.remote_steps) > 0, \
        "a remote-shaped destination is reached through ssh: every shell step and the sbatch"
    step = next(s for s in rn.remote_steps if "mv -T" in s and "current" in s)
    i_lock, i_read, i_mv = step.index("flock -w "), step.index("readlink current"), step.index("mv -T")
    assert f"exec {ST.PUBLISH_LOCK_FD}>>{shlex.quote(ST.PUBLISH_LOCK_FILE)}" in step and i_lock < i_read < i_mv, step
    lock = r1.detail["publish_lock"]
    assert step.startswith(ST.locked_step(gd, "", ST.PUBLISH_LOCK_WAIT_S, lock["holder"], "").rsplit("printf", 1)[0]), "the one helper renders the locked step"
    assert lock["path"] == f"{gd}/{ST.PUBLISH_LOCK_FILE}" and lock["scope"] == "step" and "mode=push" in lock["holder"], lock
    assert ST.PublishLock(gd).path == lock["path"], "the on-cluster lock and the step's are one file by construction"
    assert open(lock["path"]).read().startswith(lock["holder"]), "the step wrote its holder line the way PublishLock does"
    # (a) paused AFTER its readlink, inside the lock
    _write(os.path.join(w["base"], "top.txt"), b"second\n")
    pause = _Pause(str(tmp_path / "pa"), AFTER_READLINK)
    rn_a = _OverSsh("login-0", shim=pause)
    out = {}
    ta = _run_in_thread(out, "a", ST.push_component, w["root"], rdest, w["comp"], runner=rn_a, note="second")
    pause.wait_paused()
    assert pause.applied == 1
    with pytest.raises(ST.PublishLockBusy, match="held by .*mode=push"):
        ST.PublishLock(gd, wait_s=0.3, poll_s=0.05).acquire()
    logs = []
    tu = _run_in_thread(out, "u", ST.push_on_cluster, w["slug"], _oc_source(tmp_path, "oc1", b"cluster one\n"), stores, note="oc", session="jobU",
                        workers=2, lock_wait_s=60, log=logs.append)
    deadline = time.monotonic() + 20
    while not any("waiting for the publisher" in line for line in logs) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert any("waiting for the publisher" in line and "mode=push" in line for line in logs), logs
    time.sleep(0.3)
    assert os.readlink(f"{gd}/current") == sha1, "nothing switched under the paused step"
    assert not any(n.startswith(".incoming") for n in os.listdir(gd)), "the on-cluster publisher has not copied: it waits before its snapshot"
    pause.release()
    ta.join(60)
    tu.join(60)
    ra, ru = out["a"], out["u"]
    assert ra.outcome == "published" and ru.outcome == "published", (ra.as_dict(), ru.as_dict())
    sha2, sha_u = ra.content_sha8, ru.content_sha8
    assert os.readlink(f"{gd}/current") == sha_u, "snapshot order: the paused step landed, then the on-cluster publication on top of it"
    assert json.load(open(f"{gd}/PUBLISHED-{sha2}.json"))["previous"] == sha1 and json.load(open(f"{gd}/PUBLISHED-{sha_u}.json"))["previous"] == sha2
    assert ru.detail["publish_lock"]["waited_s"] > 0 and "mode=push" in ru.detail["publish_lock"]["waited_for"] and ru.detail["publish_lock"]["scope"] == "flow"
    assert not any(n.startswith(".current") for n in os.listdir(gd))
    # (b) paused BEFORE the lock: the on-cluster publisher switches `current` during the pause
    _write(os.path.join(w["base"], "top.txt"), b"third\n")
    pause_b = _Pause(str(tmp_path / "pb"), BEFORE_LOCK)
    rn_b = _OverSsh("login-0", shim=pause_b)
    tb = _run_in_thread(out, "b", ST.push_component, w["root"], rdest, w["comp"], runner=rn_b, note="third")
    pause_b.wait_paused()
    assert pause_b.applied == 1 and os.readlink(f"{gd}/current") == sha_u
    rv = ST.push_on_cluster(w["slug"], _oc_source(tmp_path, "oc2", b"cluster two\n"), stores, note="oc2", session="jobV", workers=2, lock_wait_s=5)
    assert rv.outcome == "published" and rv.detail["publish_lock"]["waited_for"] is None, rv.as_dict()      # the paused step holds no lock yet
    assert os.readlink(f"{gd}/current") == rv.content_sha8
    pause_b.release()
    tb.join(60)
    rb = out["b"]
    assert rb.outcome == "current_moved" and not rb.ok and rb.detail["current_read"] == sha_u and rb.published == rv.content_sha8, rb.as_dict()
    assert os.readlink(f"{gd}/current") == rv.content_sha8, "a stale reading never overwrites the newer publication"
    sha3 = rb.content_sha8
    assert os.path.isfile(f"{gd}/VERIFIED-{sha3}.json") and os.path.isfile(f"{gd}/SEALED-{sha3}.json") and not os.path.exists(f"{gd}/PUBLISHED-{sha3}.json")
    assert not any(n.startswith(".current") for n in os.listdir(gd))
    # a later grading host run reads the new `current` and publishes its standing verified tree through the same locked step
    r4 = ST.push_component(w["root"], rdest, w["comp"], runner=_OverSsh("login-0"), note="fourth")
    assert r4.outcome == "published_existing_verified" and r4.content_sha8 == sha3 and os.readlink(f"{gd}/current") == sha3, r4.as_dict()
    assert json.load(open(f"{gd}/PUBLISHED-{sha3}.json"))["previous"] == rv.content_sha8 and r4.detail["publish_lock"]["scope"] == "step"


def test_a_grading_host_publisher_waits_for_the_lock_the_on_cluster_publisher_holds_then_refuses_its_stale_reading(shared, tmp_path, monkeypatch):
    """The reversed completion: the on-cluster publisher starts FIRST and is slow in its verify, holding the slug's lock for its whole flow;
    the grading-host-driven publisher starts later, reads `current`, copies and verifies, and reaches its publish step while the lock is held.
    The step WAITS (nothing switches), the on-cluster publication lands, and the step then reads a `current` that is not what this run
    read -- `current_moved`, never a switch back onto the grading host snapshot. The step's wait is bounded like the on-cluster publisher's:
    `lock_busy` by name, naming the holder, with the verified tree left standing for the next run."""
    w = shared
    gd, rdest = w["gen_dir"], replace(w["dest"], host="login-0")
    stores = rdest.stores_dir
    r1 = ST.push_component(w["root"], rdest, w["comp"], runner=_OverSsh("login-0"), note="first")
    assert r1.outcome == "published", r1.as_dict()
    sha1 = r1.content_sha8
    in_verify, release = threading.Event(), threading.Event()
    real_verify = ST.verify_generation

    def slow_verify(gen_path, manifest_path, **kw):
        if open(os.path.join(gen_path, "top.txt"), "rb").read() == b"cluster slow\n":
            in_verify.set()
            assert release.wait(30), "the test never released the slow publisher"
        return real_verify(gen_path, manifest_path, **kw)

    monkeypatch.setattr(ST, "verify_generation", slow_verify)
    out = {}
    ta = _run_in_thread(out, "a", ST.push_on_cluster, w["slug"], _oc_source(tmp_path, "oc", b"cluster slow\n"), stores, note="A", session="jobA",
                        workers=2, lock_wait_s=60)
    assert in_verify.wait(30), "A never reached its verify"
    _write(os.path.join(w["base"], "top.txt"), b"grading_host later\n")
    rn = _OverSsh("login-0")
    td = _run_in_thread(out, "d", ST.push_component, w["root"], rdest, w["comp"], runner=rn, note="D")
    deadline = time.monotonic() + 30
    while not any("mv -T" in s and "current" in s for s in rn.remote_steps) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert any("mv -T" in s and "current" in s for s in rn.remote_steps), "the grading host publisher reached its publish step"
    time.sleep(0.5)
    assert td.is_alive() and os.readlink(f"{gd}/current") == sha1, "the step waits on the lock the on-cluster publisher holds; nothing switched"
    release.set()
    ta.join(60)
    td.join(60)
    ra, rd = out["a"], out["d"]
    assert ra.outcome == "published" and os.readlink(f"{gd}/current") == ra.content_sha8, ra.as_dict()
    assert rd.outcome == "current_moved" and rd.detail["current_read"] == sha1 and rd.published == ra.content_sha8, rd.as_dict()
    assert json.load(open(f"{gd}/PUBLISHED-{ra.content_sha8}.json"))["previous"] == sha1
    assert os.path.isfile(f"{gd}/VERIFIED-{rd.content_sha8}.json") and not os.path.exists(f"{gd}/PUBLISHED-{rd.content_sha8}.json")
    assert open(f"{gd}/{ST.PUBLISH_LOCK_FILE}").read().startswith(rd.detail["publish_lock"]["holder"]) if "publish_lock" in rd.detail else True
    # the bound: a lock held past the wait refuses the step by name, naming the holder; the verified tree stands; the next run publishes it
    with ST.PublishLock(gd, wait_s=0, holder="TESTHOST:1 job=99 session=held"):
        with pytest.raises(ST.PublishLockBusy, match="lock_busy: .* held by TESTHOST:1 job=99"):
            ST.Remote(rdest, _OverSsh("login-0")).publish(gd, rd.content_sha8, _rec_text(rd.content_sha8, previous=ra.content_sha8), expect_current=ra.content_sha8, lock_wait_s=0.3)
        rb = ST.push_component(w["root"], rdest, w["comp"], runner=_OverSsh("login-0"), note="busy", lock_wait_s=0.3)
        assert rb.outcome == "lock_busy" and not rb.ok and rb.outcome not in ST.PUSH_OK_OUTCOMES and "TESTHOST:1 job=99" in rb.detail["why"], rb.as_dict()
        assert rb.content_sha8 == rd.content_sha8 and rb.detail["current_read"] == ra.content_sha8
        assert os.readlink(f"{gd}/current") == ra.content_sha8 and not any(n.startswith(".current") for n in os.listdir(gd))
    st = ST.load_state(w["root"])["components"][w["slug"]]
    assert st["generation"]["state"] == "lock_busy" and st["last_run"]["outcome"] == "lock_busy"
    r5 = ST.push_component(w["root"], rdest, w["comp"], runner=_OverSsh("login-0"), note="after")
    assert r5.outcome == "published_existing_verified" and r5.content_sha8 == rd.content_sha8 and os.readlink(f"{gd}/current") == rd.content_sha8, r5.as_dict()
    assert json.load(open(f"{gd}/PUBLISHED-{rd.content_sha8}.json"))["previous"] == ra.content_sha8


def test_a_grading_host_publisher_never_switches_current_onto_or_away_from_a_cluster_jobs_publication(world, tmp_path, monkeypatch, capsys):
    """The fixer's note on C3: a lagging grading host mirror of a slug a cluster job publishes would re-point `current` at an older verified tree
    (`published_existing_verified` passes the compare-and-switch: this run read the job's newer `current`), or supersede the job's
    generation with a fresh copy. A grading-host-driven run (push, mirror, carry -- every mode but `push_on_cluster`) refuses `producer_mismatch`
    by name, before any copy, when `current`'s PUBLISHED record, or the record of the verified generation it would re-point at, names
    `producer: cluster` and the component is not declared cluster-produced. The on-cluster publisher is never refused by this rule."""
    w = world
    gd, stores = w["gen_dir"], w["dest"].stores_dir
    r1 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="grading_host")
    sha1 = r1.content_sha8
    assert r1.outcome == "published" and json.load(open(f"{gd}/PUBLISHED-{sha1}.json"))["producer"] is None
    oc = _oc_source(tmp_path, "oc", b"cluster\n")
    rc = ST.push_on_cluster("relu_paired_run_state", oc, stores, note="job", session="j1", workers=2, lock_wait_s=5)
    assert rc.outcome == "published" and json.load(open(f"{gd}/PUBLISHED-{rc.content_sha8}.json"))["producer"] == "cluster", rc.as_dict()
    # the grading host copy is unchanged (its content is the verified sha1): re-pointing `current` at it is refused, nothing copied
    rn = Runner()
    r2 = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn, note="lagging")
    assert r2.outcome == "producer_mismatch" and not r2.ok and r2.outcome not in ST.PUSH_OK_OUTCOMES, r2.as_dict()
    assert r2.published == rc.content_sha8 and r2.detail["foreign_generation"] == rc.content_sha8
    assert "names producer cluster" in r2.detail["why"] and "declares producer none" in r2.detail["why"] and "lagging mirror" in r2.detail["why"]
    assert os.readlink(f"{gd}/current") == rc.content_sha8 and not any("rsync" in a or a[0] == "sbatch" for a in rn.calls), "nothing copied, nothing switched"
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["generation"]["state"] == "producer_mismatch" and st["published"] == rc.content_sha8 and st["last_run"]["outcome"] == "producer_mismatch"
    # a changed grading host copy is refused before any copy too; so is a plain `stores mirror` (grading-host-driven, no push)
    _write(os.path.join(w["base"], "top.txt"), b"grading_host v2\n")
    listing = sorted(os.listdir(gd))
    rn = Runner()
    r3 = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn, note="changed")
    assert r3.outcome == "producer_mismatch" and not any("rsync" in a or a[0] == "sbatch" for a in rn.calls) and sorted(os.listdir(gd)) == listing
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner()).outcome == "producer_mismatch"
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "push", "relu_paired_run_state"]) == 1
    assert capsys.readouterr().out.startswith("relu_paired_run_state: producer_mismatch")
    # the cluster producer publishes on
    rc2 = ST.push_on_cluster("relu_paired_run_state", _oc_source(tmp_path, "oc2", b"cluster two\n"), stores, workers=2, lock_wait_s=5)
    assert rc2.outcome == "published" and os.readlink(f"{gd}/current") == rc2.content_sha8
    # the other direction: `current` carries no foreign record (one from before any producer was declared), but the verified tree the
    # grading host copy matches is the job's FIRST generation -- re-pointing at it is refused by that generation's own record
    rec = json.load(open(f"{gd}/PUBLISHED-{rc2.content_sha8}.json"))
    rec["producer"] = None
    with open(f"{gd}/PUBLISHED-{rc2.content_sha8}.json", "w") as fh:
        json.dump(rec, fh)
    shutil.rmtree(w["base"])
    shutil.copytree(oc, w["base"])
    r4 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="lagging pull")
    assert r4.content_sha8 == rc.content_sha8 and r4.outcome == "producer_mismatch" and r4.detail["foreign_generation"] == rc.content_sha8, r4.as_dict()
    assert os.readlink(f"{gd}/current") == rc2.content_sha8 and not os.path.exists(f"{gd}/PUBLISHED-{rc.content_sha8}.json.tmp")
    # declared cluster-produced, the record names this component's own producer: not refused by this rule
    r5 = ST.push_component(w["root"], w["dest"], replace(w["comp"], producer="cluster"), runner=Runner(), note="declared")
    assert r5.outcome == "published_existing_verified" and os.readlink(f"{gd}/current") == rc.content_sha8, r5.as_dict()


# --------------------------------------------------------------------------------------------
# the producer guard and the publication record live inside the lock
# --------------------------------------------------------------------------------------------


class _Cut:
    """A shim that CUTS the publish step right before ``anchor`` (``exit 99``): the step dies after what precedes the anchor landed."""

    def __init__(self, anchor):
        self.anchor, self.applied = anchor, 0

    def __call__(self, cmd):
        if "mv -T" not in cmd or "current" not in cmd or self.anchor not in cmd:
            return cmd
        self.applied += 1
        return cmd.replace(self.anchor, f"exit 99; {self.anchor}", 1)


def test_the_producer_guard_and_the_publication_record_live_inside_the_lock(world, tmp_path):
    """The producer check read PUBLISHED-<current>.json BEFORE the destination lock, and the locked step switched `current` and wrote the
    record only AFTERWARD -- so a publisher paused (or killed) between its switch and its record left a `current` with no record that a
    push from the grading host read as "not foreign", waited for the lock, passed the comparison (current still named what it read) and
    re-pointed `current` back to its older tree through published_existing_verified. Now the step, under the lock and in order: compares
    `current`, requires its record (current_unrecorded), applies the producer rule to that record and to the generation's own
    (producer_mismatch), writes THIS record (temp name + rename), then switches -- that window no longer exists, and a `current` left
    without its record by the old flow (or a crash) is refused until adopted explicitly."""
    w = world
    gd, rdest = w["gen_dir"], replace(w["dest"], host="login-0")
    stores = rdest.stores_dir
    rn = _OverSsh("login-0")
    r1 = ST.push_component(w["root"], rdest, w["comp"], runner=rn, note="first")
    assert r1.outcome == "published", r1.as_dict()
    sha1 = r1.content_sha8
    step = next(s for s in rn.remote_steps if "mv -T" in s and "current" in s)
    i_lock, i_read, i_cmp, i_unrec, i_prod, i_rec, i_mv = (step.index("flock -w "), step.index("readlink current"), step.index("exit 66"),
                                                            step.index("exit 72"), step.index("exit 73"), step.index(f"mv -f"), step.index("mv -T"))
    assert i_lock < i_read < i_cmp < i_unrec < i_prod < i_rec < i_mv, step
    assert f"PUBLISHED-{sha1}.json" in step[i_rec:i_mv] and "mv -T" not in step[:i_rec], "the record lands before the switch, inside the lock"
    assert json.load(open(f"{gd}/PUBLISHED-{sha1}.json"))["previous"] is None
    # a publisher that switched `current` and never recorded it (killed between the two in the old flow): a cluster publication, its record gone
    rc = ST.push_on_cluster("relu_paired_run_state", _oc_source(tmp_path, "oc", b"cluster\n"), stores, note="job", session="j1", workers=2, lock_wait_s=5)
    assert rc.outcome == "published" and os.readlink(f"{gd}/current") == rc.content_sha8
    os.remove(f"{gd}/PUBLISHED-{rc.content_sha8}.json")
    # the grading host push (its content sha1: verified, sealed, standing) refuses by name before any copy and re-points nothing
    rn2 = _OverSsh("login-0")
    r2 = ST.push_component(w["root"], rdest, w["comp"], runner=rn2, note="lagging")
    assert r2.outcome == "current_unrecorded" and not r2.ok and r2.outcome not in ST.PUSH_OK_OUTCOMES, r2.as_dict()
    assert "mid-flight or crashed" in r2.detail["why"] and "--adopt-unrecorded" in r2.detail["why"] and r2.detail["unrecorded_generation"] == rc.content_sha8
    assert os.readlink(f"{gd}/current") == rc.content_sha8 and not any("rsync" in a or a[0] == "sbatch" for a in rn2.calls)
    assert not any("mv -T" in s_ for s_ in rn2.remote_steps), "no publish step was even attempted"
    # the step refuses the same under the lock, the pre-check bypassed: a direct publication whose reading IS `current`
    before = json.load(open(f"{gd}/PUBLISHED-{sha1}.json"))
    with pytest.raises(ST.CurrentUnrecorded, match="current_unrecorded.*PUBLISHED-" + rc.content_sha8 + r"\.json") as ei:
        ST.Remote(rdest, _OverSsh("login-0")).publish(gd, sha1, _rec_text(sha1, previous=rc.content_sha8), expect_current=rc.content_sha8, lock_wait_s=1)
    assert ei.value.generation == rc.content_sha8 and os.readlink(f"{gd}/current") == rc.content_sha8
    assert json.load(open(f"{gd}/PUBLISHED-{sha1}.json")) == before and not any(n.startswith((".current", ".PUBLISHED")) for n in os.listdir(gd)), "no record written, nothing switched"
    # the publisher's record appears (it finished): it names a cluster producer -> producer_mismatch, in the pre-check and in the step
    _write(f"{gd}/PUBLISHED-{rc.content_sha8}.json", _rec_text(rc.content_sha8, previous=sha1, producer="cluster", mode="push_on_cluster").encode())
    r3 = ST.push_component(w["root"], rdest, w["comp"], runner=_OverSsh("login-0"), note="lagging")
    assert r3.outcome == "producer_mismatch" and r3.detail["foreign_generation"] == rc.content_sha8, r3.as_dict()
    with pytest.raises(ST.ProducerMismatch, match="producer_mismatch.*the `current` this run read.*names producer cluster") as ei:
        ST.Remote(rdest, _OverSsh("login-0")).publish(gd, sha1, _rec_text(sha1, previous=rc.content_sha8), expect_current=rc.content_sha8, lock_wait_s=1)
    assert ei.value.generation == rc.content_sha8 and os.readlink(f"{gd}/current") == rc.content_sha8 and json.load(open(f"{gd}/PUBLISHED-{sha1}.json")) == before
    # the generation's OWN record is checked too: re-pointing at a tree a cluster job published is refused in the step
    _write(f"{gd}/PUBLISHED-{rc.content_sha8}.json", _rec_text(rc.content_sha8, previous=sha1).encode())           # the current's record: no producer
    _write(f"{gd}/PUBLISHED-{sha1}.json", _rec_text(sha1, producer="cluster", mode="push_on_cluster").encode())    # sha1 "was a cluster job's"
    with pytest.raises(ST.ProducerMismatch, match="this generation, published before"):
        ST.Remote(rdest, _OverSsh("login-0")).publish(gd, sha1, _rec_text(sha1, previous=rc.content_sha8), expect_current=rc.content_sha8, lock_wait_s=1)
    assert json.load(open(f"{gd}/PUBLISHED-{sha1}.json"))["producer"] == "cluster", "the cluster job's record is not overwritten"
    _write(f"{gd}/PUBLISHED-{sha1}.json", ST.record_text(before).encode())
    # a component declared cluster-produced is not refused (no producer guard), and the step's record lands BEFORE its switch: cut the
    # step between the two -- the record stands with its `previous`, `current` has not moved, and the next run completes the publication
    comp_c = replace(w["comp"], producer="cluster")
    cut = _Cut("ln -sfn")
    rn4 = _OverSsh("login-0", shim=cut)
    r4 = ST.push_component(w["root"], rdest, comp_c, runner=rn4, note="cut")
    assert cut.applied == 1 and r4.outcome == "error" and "rc 99" in r4.detail["error"], r4.as_dict()
    assert os.readlink(f"{gd}/current") == rc.content_sha8, "cut before the switch: `current` did not move"
    cut_rec = json.load(open(f"{gd}/PUBLISHED-{sha1}.json"))
    assert cut_rec["previous"] == rc.content_sha8 and cut_rec["note"] == "cut" and cut_rec["producer"] == "cluster", "the record landed first, naming the `current` compared under the lock"
    assert not any(n.startswith((".current", ".PUBLISHED")) for n in os.listdir(gd))
    r5 = ST.push_component(w["root"], rdest, comp_c, runner=_OverSsh("login-0"), note="completed")
    assert r5.outcome == "published_existing_verified" and os.readlink(f"{gd}/current") == sha1, r5.as_dict()
    assert json.load(open(f"{gd}/PUBLISHED-{sha1}.json"))["previous"] == rc.content_sha8 and json.load(open(f"{gd}/PUBLISHED-{sha1}.json"))["note"] == "completed"
    # the on-cluster publisher's step (lock held for its flow) has the same order and the same refusal of an unrecorded `current`
    os.remove(f"{gd}/PUBLISHED-{sha1}.json")
    rc2 = ST.push_on_cluster("relu_paired_run_state", _oc_source(tmp_path, "oc2", b"cluster two\n"), stores, workers=2, lock_wait_s=5)
    assert rc2.outcome == "current_unrecorded" and rc2.detail["unrecorded_generation"] == sha1 and os.readlink(f"{gd}/current") == sha1, rc2.as_dict()
    assert not any(n.startswith(".incoming") for n in os.listdir(gd)), "nothing copied"
    rc3 = ST.push_on_cluster("relu_paired_run_state", _oc_source(tmp_path, "oc2", b"cluster two\n"), stores, workers=2, lock_wait_s=5, adopt_unrecorded=True)
    assert rc3.outcome == "published" and os.readlink(f"{gd}/current") == rc3.content_sha8, "the explicit adoption supersedes the unrecorded legacy `current`"
    assert json.load(open(f"{gd}/PUBLISHED-{rc3.content_sha8}.json"))["previous"] == sha1


def test_an_unrecorded_current_is_adopted_only_explicitly_under_the_lock_and_never_switched(world, tmp_path, monkeypatch, capsys):
    """The missing-record-after-crash case: `current` names a verified, sealed generation with no PUBLISHED record (a publisher that died
    between its switch and its record in the old flow; a generation from before records existed). Every publisher refuses by name until
    the record exists -- push, mirror, the CLI (exit 1) -- and nothing is copied or switched. `stores push --adopt-unrecorded` from the
    producer whose content it is writes the record under the publish lock WITHOUT switching (mode adopt); the next push behaves normally
    (unchanged, then a change publishes with the adopted generation as its previous). With the flag and a different content the
    publication supersedes the unrecorded `current` (the operator asserted it is legacy, not mid-flight). The adoption step takes the same
    lock (lock_busy by name when held), writes a record only for the generation that IS current, and writes nothing twice."""
    w = world
    gd = w["gen_dir"]
    r1 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="first")
    sha1 = r1.content_sha8
    os.remove(f"{gd}/PUBLISHED-{sha1}.json")                                    # the crash / legacy shape
    listing = sorted(os.listdir(gd))
    rn = Runner()
    r2 = ST.push_component(w["root"], w["dest"], w["comp"], runner=rn, note="again")
    assert r2.outcome == "current_unrecorded" and not r2.ok and r2.published == sha1 and r2.detail["unrecorded_generation"] == sha1, r2.as_dict()
    assert "wait for that publisher, or adopt" in r2.detail["why"] and "--adopt-unrecorded" in r2.detail["why"]
    assert sorted(os.listdir(gd)) == listing and not any("rsync" in a or a[0] == "sbatch" for a in rn.calls), "nothing written, copied or switched"
    st = ST.load_state(w["root"])["components"]["relu_paired_run_state"]
    assert st["generation"]["state"] == "current_unrecorded" and st["last_run"]["outcome"] == "current_unrecorded" and st["published"] == sha1
    assert ST.mirror_component(w["root"], w["dest"], w["comp"], runner=Runner()).outcome == "current_unrecorded"
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "push", "relu_paired_run_state"]) == 1
    assert capsys.readouterr().out.startswith("relu_paired_run_state: current_unrecorded")
    # the adoption: under the lock, no switch; lock_busy by name while another publisher holds the lock
    with ST.PublishLock(gd, wait_s=0, holder="TESTHOST:1 job=99 session=held"):
        with pytest.raises(ST.PublishLockBusy, match="lock_busy: .* held by TESTHOST:1 job=99.*the record of " + sha1 + " is not written"):
            ST.Remote(w["dest"], Runner()).record_published(gd, sha1, _rec_text(sha1, mode="adopt"), lock_wait_s=0.3)
        rb = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), adopt_unrecorded=True, lock_wait_s=0.3)
        assert rb.outcome == "lock_busy" and not os.path.exists(f"{gd}/PUBLISHED-{sha1}.json")
    r3 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="adopting", adopt_unrecorded=True)
    assert r3.outcome == "unchanged" and r3.ok and set(r3.detail["adopted"]) == {"published_record"}, r3.as_dict()
    rec = json.load(open(f"{gd}/PUBLISHED-{sha1}.json"))
    assert rec["mode"] == "adopt" and rec["previous"] is None and rec["note"] == "adopting" and rec["changed_files"] == 4 and os.readlink(f"{gd}/current") == sha1
    assert r3.detail["published_record"]["mode"] == "adopt" and ST.load_state(w["root"])["components"]["relu_paired_run_state"]["generation"]["adopted"]["published_record"]["mode"] == "adopt"
    assert cli.main(["stores", "push", "relu_paired_run_state", "--adopt-unrecorded"]) == 0 and capsys.readouterr().out.startswith("relu_paired_run_state: unchanged")
    # the next push behaves normally: unchanged, then a change publishes with the adopted generation as its previous
    r4 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner())
    assert r4.outcome == "unchanged" and "adopted" not in r4.detail and r4.detail["published_record"]["note"] == "adopting"
    _write(os.path.join(w["base"], "top.txt"), b"after the adoption\n")
    r5 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="next")
    assert r5.outcome == "published" and json.load(open(f"{gd}/PUBLISHED-{r5.content_sha8}.json"))["previous"] == sha1 and os.readlink(f"{gd}/current") == r5.content_sha8
    # record_published writes only for the generation that IS current, and never twice
    with pytest.raises(ST.CurrentMoved, match="adopt_refused: current_moved.*not " + sha1):
        ST.Remote(w["dest"], Runner()).record_published(gd, sha1, _rec_text(sha1, mode="adopt"))
    assert ST.Remote(w["dest"], Runner()).record_published(gd, r5.content_sha8, _rec_text(r5.content_sha8, mode="adopt"))["written"] is False
    assert json.load(open(f"{gd}/PUBLISHED-{r5.content_sha8}.json"))["note"] == "next", "an existing record is never replaced by an adoption"
    # the supersede variant: a differing content with the flag publishes over an unrecorded legacy `current`; without it, refused
    os.remove(f"{gd}/PUBLISHED-{r5.content_sha8}.json")
    _write(os.path.join(w["base"], "top.txt"), b"third\n")
    rn6 = Runner()
    assert ST.push_component(w["root"], w["dest"], w["comp"], runner=rn6, note="third").outcome == "current_unrecorded" and not any("rsync" in a for a in rn6.calls)
    r7 = ST.push_component(w["root"], w["dest"], w["comp"], runner=Runner(), note="third", adopt_unrecorded=True)
    assert r7.outcome == "published" and json.load(open(f"{gd}/PUBLISHED-{r7.content_sha8}.json"))["previous"] == r5.content_sha8 and os.readlink(f"{gd}/current") == r7.content_sha8
    # `stores mirror --adopt-unrecorded`: the one-time sweep for generations published before records existed
    os.remove(f"{gd}/PUBLISHED-{r7.content_sha8}.json")
    assert cli.main(["stores", "mirror"]) == 1 and '"outcome": "current_unrecorded"' in capsys.readouterr().out
    assert cli.main(["stores", "mirror", "--adopt-unrecorded"]) == 0
    out = capsys.readouterr().out
    assert '"outcome": "unchanged"' in out and json.load(open(f"{gd}/PUBLISHED-{r7.content_sha8}.json"))["mode"] == "adopt" and os.readlink(f"{gd}/current") == r7.content_sha8


# --------------------------------------------------------------------------------------------
# the R2b run (2026-10-06): the destination refuses more than ':' -- VAST's Windows-reserved set, measured one `touch` per character
# in the drills namespace: `| < > " ? * \` EINVAL, `% ; & # { } [ ] @ ! = + ,` accepted. fullscale_consolidated's 477 attested names
# with '|' failed rsync's mkstemp (EINVAL) under the two-character rule; the escape now covers the measured set, '%' first.
# --------------------------------------------------------------------------------------------


def _refused_tree(tmp_path):
    base = tmp_path / "refused"
    for ch in ST.DESTINATION_REFUSED_CHARS:
        _write(str(base / "degen" / f"model{ch}attack{ch}obj.jsonl"), f"row {ch}\n".encode())
    _write(str(base / "degen" / "plain.jsonl"), b"plain\n")
    _write(str(base / "100%done" / "g.txt"), b"three\n")
    os.symlink("degen/model|attack|obj.jsonl", str(base / "link|to"))          # an escaped name AND an escaped target
    os.symlink("degen/plain.jsonl", str(base / "plainlink"))
    return str(base)


def test_the_listing_attestation_verifies_an_escaped_tree_in_its_destination_spelling(tmp_path):
    """The directory attestation spells the canonical's TRUE names (MANIFEST_RULES["directories"]) while the destination spells an escaped
    entry per NAME_ESCAPE_RULE, and the first merged verify compared the two spellings directly -- a byte-correct escaped generation read
    `listing_mismatch`, was renamed `.failed-*`, and the component's cluster-only readers lost it. `listing_mismatches` finds an attested
    directory at its ESCAPED path and compares the destination's listing in TRUE names: an escaped file, an escaped directory and an
    escaped EMPTY directory (carried by neither rsync -- excluded, since rsync would try its literal name -- nor tar; created by the
    verify at its escaped spelling) all verify and the generation publishes; a directory present in the literal true spelling, or an
    unattested escaped entry, is a mismatch."""
    base = tmp_path / "esc"
    _write(str(base / "m:odel" / "in:ner" / "f.json"), b"{}\n")
    _write(str(base / "m:odel" / "plain.json"), b"{}\n")
    _write(str(base / "top:file.txt"), b"t\n")
    _write(str(base / "plain" / "p.txt"), b"p\n")
    os.makedirs(str(base / "e:mpty"))
    os.makedirs(str(base / "plain" / "also_empty"))
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    cfgp = _cfg(tmp_path, {"esc": {"base": str(base), "authority": "grading_host_copy"}}, adapters={"relu_paired": {"shards": 1, "components": ["esc"]}})
    cfg = ST.load_config(cfgp)
    dest, comp = cfg.destination("cluster_a"), cfg.components["esc"]
    pre = ST.build_manifest(comp, ST.component_state_dir(root, "esc"))
    carried, excludes = pre.tar_carried()
    assert carried == ["m:odel/in:ner/f.json", "m:odel/plain.json", "top:file.txt"] and excludes == ["e:mpty", "m:odel", "top:file.txt"], (carried, excludes)
    res = ST.mirror_component(root, dest, comp, runner=Runner())
    assert res.outcome == "published", res.as_dict()
    gd, sha8 = dest.gen_dir("esc"), res.content_sha8
    gen = f"{gd}/{sha8}"
    ver = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))
    assert ver["ok"] is True and ver["listings_attested"] is True and ver["listing_mismatch"] == [] and ver["n_directories"] == 6, ver
    assert ver["created_dirs"] == ["e:mpty"], ver["created_dirs"]                  # the empty escaped directory, named by its TRUE rel
    assert os.path.isdir(f"{gen}/e%3Ampty") and not os.path.lexists(f"{gen}/e:mpty"), "created at the destination spelling"
    assert os.path.isdir(f"{gen}/plain/also_empty"), "a plain-named empty directory rsync carried"
    assert sorted(os.listdir(gen)) == ["e%3Ampty", "m%3Aodel", "plain", "top%3Afile.txt"]
    meta = json.load(open(f"{gd}/MANIFEST-{sha8}.json"))
    assert meta["directories"][""] == [["e:mpty", "dir"], ["m:odel", "dir"], ["plain", "dir"], ["top:file.txt", "file"]], "the attestation spells true names"
    assert meta["directories"]["m:odel/in:ner"] == [["f.json", "file"]]
    assert ST.admit_generation(gd, sha8).ok
    # the unit: the escaped on-disk tree against the canonical attestation is clean; the literal true spelling of a directory is not what
    # the destination lists (found at no escaped path), and an unattested escaped entry is a mismatch of its parent
    want = ST.read_manifest_directories(f"{gd}/MANIFEST-{sha8}.sha256")
    assert ST.listing_mismatches(gen, want) == ([], [])
    bad = str(tmp_path / "bad_gen")
    shutil.copytree(gen, bad, symlinks=True)
    for d in _dirs(bad):
        os.chmod(d, 0o755)
    os.rename(f"{bad}/m%3Aodel", f"{bad}/m:odel")
    _write(f"{bad}/x%3Ay", b"")
    assert ST.listing_mismatches(bad, want) == (["", "m:odel", "m:odel/in:ner"], [])


def test_name_escape_covers_the_measured_refused_set_and_tar_carries_every_such_entry(tmp_path):
    assert ST.DESTINATION_REFUSED_CHARS == (":", "|", "<", ">", '"', "?", "*", "\\"), "the set VAST refused, measured; a change here is a new measurement"
    assert ST.NAME_ESCAPE_CHARS[0] == "%" and ST.TAR_TRANSFORMS[0] == "s/%/%25/gs", "'%' is spelled first so every other escape stays reversible"
    assert ST.NAME_ESCAPES == {"%": "%25", ":": "%3A", "|": "%7C", "<": "%3C", ">": "%3E", '"': "%22", "?": "%3F", "*": "%2A", "\\": "%5C"}
    assert "s/\\*/%2A/gs" in ST.TAR_TRANSFORMS and "s/\\\\/%5C/gs" in ST.TAR_TRANSFORMS and "s/|/%7C/gs" in ST.TAR_TRANSFORMS, ST.TAR_TRANSFORMS
    for n in ("a|b", "q?r", "s*t", "u\\v", 'w"x', "y<z>", "100%done", "a:b", "%25", "%2525", "%7C", "plain", ':|<>"?*\\%', "em-x::y"):
        assert ST.unescape_name(ST.escape_name(n)) == n, n
    assert ST.escape_name("a:b") == "a%3Ab" and ST.escape_name("a%3Ab") == "a%253Ab", "the legacy spelling of ':' and '%' is unchanged (misalign_grades' generation)"
    assert ST.escape_name("qwen|pasttense|prompted.jsonl") == "qwen%7Cpasttense%7Cprompted.jsonl" and ST.unescape_name("a%7Cb") == "a|b"
    assert ST.escape_rel("d/m|a/f*.json") == "d/m%7Ca/f%2A.json" and ST.rel_needs_escape("d/m|a") is True and ST.rel_needs_escape("d/m-a") is False
    # rsync reads '\\' as an escape only beside a wildcard: a backslash-only name is excluded as it is (doubled, the exclude missed and
    # rsync copied the true name beside tar's escaped copy); with a wildcard present both the wildcards and the backslashes are escaped
    assert ST._rsync_literal("degen/model\\attack.jsonl") == "degen/model\\attack.jsonl" and ST._rsync_literal("a|b:c") == "a|b:c"
    assert ST._rsync_literal("f*.json") == "f\\*.json" and ST._rsync_literal("a*\\b?c[d]") == "a\\*\\\\b\\?c\\[d\\]"
    base = _refused_tree(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    cfg = ST.load_config(_cfg(tmp_path, {"refused": {"base": base, "authority": "grading_host_copy"}}, adapters={"relu_paired": {"shards": 1, "components": ["refused"]}}))
    dest, comp = cfg.destination("cluster_a"), cfg.components["refused"]
    res = ST.mirror_component(root, dest, comp, runner=Runner())
    assert res.outcome == "published", res.as_dict()
    gd, sha8 = dest.gen_dir("refused"), res.content_sha8
    gen = f"{gd}/{sha8}"
    assert res.detail["name_escapes"]["n"] == 10 and res.detail["target_escapes"] == {"n": 1, "links": ["link|to"]}, res.detail
    assert res.detail["tar"] == {"n": 10, "rc": 0}, "the eight refused names, 100%done/g.txt and the link (name and target both escaped)"
    on_disk = sorted(os.path.relpath(os.path.join(dp, f), gen) for dp, _d, fns in os.walk(gen) for f in fns)
    assert on_disk == ["100%25done/g.txt", "degen/model%22attack%22obj.jsonl", "degen/model%2Aattack%2Aobj.jsonl", "degen/model%3Aattack%3Aobj.jsonl",
                       "degen/model%3Cattack%3Cobj.jsonl", "degen/model%3Eattack%3Eobj.jsonl", "degen/model%3Fattack%3Fobj.jsonl",
                       "degen/model%5Cattack%5Cobj.jsonl", "degen/model%7Cattack%7Cobj.jsonl", "degen/plain.jsonl", "link%7Cto", "plainlink"], on_disk
    assert os.readlink(f"{gen}/link%7Cto") == "degen/model%7Cattack%7Cobj.jsonl", "the link's target is spelled as its renamed referent"
    assert open(f"{gen}/degen/model%7Cattack%7Cobj.jsonl", "rb").read() == b"row |\n"
    meta = json.load(open(f"{gd}/MANIFEST-{sha8}.json"))
    assert meta["name_escapes"]["degen/model|attack|obj.jsonl"] == "degen/model%7Cattack%7Cobj.jsonl" and meta["symlinks"]["link|to"] == "degen/model|attack|obj.jsonl"
    assert ST.unescape_rel("degen/model%7Cattack%7Cobj.jsonl") == "degen/model|attack|obj.jsonl"
    # the verified generation admits and re-verifies under the escaped spelling
    assert ST.admit_generation(gd, sha8).ok
    rep, _ds = ST.verify_generation(gen, f"{gd}/MANIFEST-{sha8}.sha256", workers=2)
    assert rep["ok"] and rep["n_files"] == 10 and not rep["missing"] and not rep["extra"], rep


def test_check_authority_strict_mode_counts_certified_and_names_a_never_published_component(tmp_path, monkeypatch, capsys):
    """The R2b run's after-check read "0 problems; 49 ok" with five components never pushed: a `grading_host_copy` component with no
    `current` has no authority to check and reads ok by default (a note, `never_published`). The one-time move's after-check needs the
    strict reading (`--require-certified`): never published is the PROBLEM `not_certified_never_published`, and the summary counts
    certified = current exists + VERIFIED + SEALED + PUBLISHED, in both modes."""
    base = _canon(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    excl = ["__pycache__", "/.pytest_cache"]
    comps = {"pushed": {"base": base, "authority": "grading_host_copy", "exclude": excl},
             "unsealed": {"base": base, "authority": "grading_host_copy", "exclude": excl},
             "never": {"base": base, "authority": "grading_host_copy", "exclude": excl},
             "orig": {"base": base, "authority": "cluster_a_origin", "original": "exp/orig"}}
    cfgp = _cfg(tmp_path, comps, adapters={"relu_paired": {"shards": 1, "components": list(comps)}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    shas = {}
    for slug in ("pushed", "unsealed"):
        r = ST.push_component(root, dest, cfg.components[slug], runner=Runner(), note=slug)
        assert r.outcome == "published", r.as_dict()
        shas[slug] = r.content_sha8
    os.remove(f"{dest.gen_dir('unsealed')}/SEALED-{shas['unsealed']}.json")
    # the default reading: never published is a note, state ok; certified counts only the fully attested current
    rec = ST.check_authority(root, cfg, dest, runner=Runner())
    c = rec["components"]
    assert rec["require_certified"] is False and rec["n_checked"] == 3 and rec["n_certified"] == 1 and rec["n_components"] == 4
    assert c["pushed"]["certified"] is True and c["pushed"]["state"] == "ok"
    assert c["unsealed"]["certified"] is False and c["unsealed"]["problems"] == ["unsealed"]
    assert c["never"]["certified"] is False and c["never"]["state"] == "ok" and c["never"]["problems"] == [] and c["never"]["notes"] == ["never_published"]
    assert "certified" not in c["orig"] and c["orig"]["state"] == "not_applicable"
    assert rec["ok"] is False and rec["problems"] == ["unsealed: unsealed"]
    text = ST.format_authority(rec)
    assert "certified 1 of 3 (current exists, VERIFIED, SEALED, PUBLISHED)" in text and "strict: never-published is a problem" not in text, text
    # the strict reading: the never-published component is a problem by name; the certified count is the same number
    strict = ST.check_authority(root, cfg, dest, runner=Runner(), require_certified=True)
    assert strict["require_certified"] is True and strict["n_certified"] == 1 and strict["ok"] is False
    assert strict["components"]["never"]["problems"] == ["not_certified_never_published"] and strict["components"]["never"]["notes"] == []
    assert strict["components"]["never"]["state"] == "not_certified_never_published"
    assert strict["problems"] == ["never: not_certified_never_published", "unsealed: unsealed"]
    assert "strict: never-published is a problem" in ST.format_authority(strict) and "never: not_certified_never_published" in ST.format_authority(strict)
    saved = json.load(open(os.path.join(root, "state", ST.AUTHORITY_STATE_FILE)))
    assert saved["require_certified"] is True and saved["n_certified"] == 1, "the state file carries the reading it was written under"
    # seal the unsealed one: default reads OK with one never-published; strict still refuses by name, through the CLI's exit code
    ST.seal_generation(ST.Remote(dest, Runner()), dest.gen_dir("unsealed"), shas["unsealed"])
    monkeypatch.setenv(ST.CONFIG_ENV, cfgp)
    assert cli.main(["--root", root, "stores", "check-authority"]) == 0
    out = capsys.readouterr().out
    assert "check-authority OK" in out and "certified 2 of 3" in out, out
    assert cli.main(["--root", root, "stores", "check-authority", "--require-certified"]) == 1
    out = capsys.readouterr().out
    assert "check-authority NOT OK: never: not_certified_never_published" in out and "certified 2 of 3" in out and "strict: never-published is a problem" in out, out
    assert cli.main(["--root", root, "stores", "check-authority", "--require-certified", "--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["n_certified"] == 2 and doc["components"]["never"]["problems"] == ["not_certified_never_published"]
