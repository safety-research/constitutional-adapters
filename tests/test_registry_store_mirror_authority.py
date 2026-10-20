"""The REVERSED store mirror: `registry stores mirror` dispatches on each component's
authority -- a `cluster_a` component's copy on the grading host is VERIFIED against the published generation (a grading host producer: `verified` /
`unpushed_changes`, or published on its behalf under `transitional_carry` with mode `carry`; a frozen producer: `frozen_differs` is red
and never repaired; a cluster producer: the generation PULLED into the registry-owned mirror with every escaped name restored, hash-
verified), a `grading_host_copy` component takes the carry as before; `--full` reads every file on the grading host and re-hashes the published
generation on the destination through the pinned verifier (`full_verify_failed` names a mismatch and renames nothing); the weekly unit;
the authority rows of `stores status`; `check-authority`'s `unrecorded` as a problem; and a verify report not yet visible after its job
never renames the tree `.failed-` (the live bigcodebench_full misread of 2026-10-04). Also here: the restore of escaped names
staged per directory (a generation holding both `a:b` and `a%3Ab`), the escaped spellings cleared before the pull's rsync, the weekly
unit without a systemd conflict (the shared mirror lock with a bounded wait instead), and a `current` called green only with its
VERIFIED / SEALED / PUBLISHED evidence (red by name otherwise; adopted only explicitly). The destination is a local directory and the shared
`Runner` of test_registry_stores passes every shell step and rsync through to real subprocesses while intercepting `sbatch` (the verify
runs in-process) and the watcher.
"""
from __future__ import annotations

import configparser
import dataclasses
import json
import os
import pathlib
import re
import shutil
import stat as stat_module
import subprocess
import sys
import threading
import time

import pytest

from sisteer.registry import cli
from sisteer.registry import records as R
from sisteer.registry import stores as ST

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registry_stores import (MISSING_LINE, Runner, SlurmRunner, WATCH, _SealBlip, _canon, _cfg, _dir_modes,  # noqa: E402,F401
                                  _write, world)                                              # (the fixture rides along)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = os.path.join(REPO, "systemd", "user")
INSTALL = os.path.join(REPO, "scripts", "registry", "install_grading_host.sh")
WINDOW = os.path.join(REPO, "scripts", "registry", "landing", "window.sh")
EXCL = ["__pycache__", "/.pytest_cache"]


def _set_mtimes(base, ns_by_rel):
    for rel, ns in ns_by_rel.items():
        os.utime(os.path.join(base, rel), ns=(ns, ns))


def _fmt(ns):
    from datetime import datetime, timezone
    return R._fmt_utc(datetime.fromtimestamp(ns / 1e9, timezone.utc))


def _mode(p):
    return stat_module.S_IMODE(os.lstat(p).st_mode)


def _dirs(top):
    return [dp for dp, _d, _f in os.walk(top)]
def _unseal_edit_reseal(gen_path, rel, data):
    """A same-size edit inside a sealed generation as the owner would do it (the seal refuses it otherwise), modes restored."""
    p = os.path.join(gen_path, rel)
    d = os.path.dirname(p)
    st = os.lstat(p)
    os.chmod(d, 0o755)
    os.chmod(p, 0o644)
    with open(p, "wb") as fh:
        fh.write(data)
    os.utime(p, ns=(st.st_mtime_ns, st.st_mtime_ns))
    os.chmod(p, stat_module.S_IMODE(st.st_mode))
    os.chmod(d, 0o555)


def _world(tmp_path, **extra_components):
    """A registry root, a local destination and the five producer shapes over one canonical tree (plus ``extra_components``)."""
    base = _canon(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    comps = {"dev": {"base": base, "authority": "cluster_a", "producer": "grading_host:sess", "exclude": EXCL},
             "carry": {"base": base, "authority": "cluster_a", "producer": "grading_host:sess", "transitional_carry": True, "carry_deadline": "2020-01-01",
                       "exclude": EXCL},
             "frozen": {"base": base, "authority": "cluster_a", "producer": "frozen", "exclude": EXCL},
             "clu": {"base": str(tmp_path / "declared_base_of_clu"), "authority": "cluster_a", "producer": "cluster"},
             "dc": {"base": base, "authority": "grading_host_copy", "exclude": EXCL},
             **extra_components}
    cfgp = _cfg(tmp_path, comps, adapters={"relu_paired": {"shards": 1, "components": list(comps)}})
    cfg = ST.load_config(cfgp)
    return {"base": base, "root": root, "cfg": cfg, "cfgp": cfgp, "dest": cfg.destination("cluster_a")}


def _no_copy(rn):
    return not any("rsync" in a or a[0] == "sbatch" or a[0] == WATCH for a in rn.calls)


# --------------------------------------------------------------------------------------------
# a grading host producer: verify only; unpushed changes are amber; the carry publishes on its behalf
# --------------------------------------------------------------------------------------------


def test_grading_host_producer_is_verified_only_and_unpushed_changes_is_amber(tmp_path):
    w = _world(tmp_path)
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["dev"]
    gd = dest.gen_dir("dev")
    # nothing published: the producer's to push; nothing copied, nothing on the destination, exit 0 (amber)
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "unpushed_changes" and r.level == "amber" and r.ok and r.published is None and "stores push dev" in r.detail["why"], r.as_dict()
    assert _no_copy(rn) and not os.path.exists(gd)
    st = ST.load_state(root)["components"]["dev"]
    assert st["published"] is None and st["generation"]["state"] == "unpushed_changes" and st["producer"] == "grading_host:sess" and st["grading_host_mirror"] == w["base"]
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "unpushed_changes" and r.detail["writes"] == "nothing"
    # the producer pushes; the mirror then VERIFIES: equal -> verified, nothing copied; `published` is the generation (the window's pin)
    pushed = ST.push_component(root, dest, comp, runner=Runner(), session="sess")
    assert pushed.outcome == "published"
    sha8 = pushed.content_sha8
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verified" and r.level == "green" and r.published == sha8 and r.content_sha8 == sha8 and r.n_files == 4, r.as_dict()
    assert _no_copy(rn) and r.detail["generation_record"]["published_record"] is True and r.detail["published_record"]["mode"] == "push"
    st = ST.load_state(root)["components"]["dev"]
    assert st["published"] == sha8 and st["generation"]["state"] == "verified" and st["generation"]["verified_utc"] and st["latest_manifest"]["content_sha8"] == sha8
    sdir = ST.component_state_dir(root, "dev")
    assert os.path.isfile(os.path.join(sdir, f"MANIFEST-{sha8}.sha256")) and os.path.isfile(os.path.join(sdir, f"MANIFEST-{sha8}.stat.json"))
    # the grading host copy changes: amber, exit 0, the generation stays the pin, NOTHING copied; the pin's manifest copy stays
    _write(os.path.join(w["base"], "top.txt"), b"changed by the producer\n")
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "unpushed_changes" and r.level == "amber" and r.ok and r.published == sha8 and r.content_sha8 != sha8, r.as_dict()
    assert r.detail["differing"]["counts"] == {"only_grading_host": 0, "only_original": 0, "sha_or_size_differs": 1, "symlinks": 0}
    assert r.detail["differing"]["sha_or_size_differs"] == ["top.txt"] and r.detail["unpushed_files"] == 1
    assert _no_copy(rn) and os.readlink(f"{gd}/current") == sha8 and os.path.isfile(os.path.join(sdir, f"MANIFEST-{sha8}.sha256"))
    st = ST.load_state(root)["components"]["dev"]
    assert st["published"] == sha8 and st["generation"]["state"] == "unpushed_changes" and st["generation"]["grading_host_content_sha8"] == r.content_sha8
    # a dry run reads only: the stat cache is read, never written; it says what a tick would conclude
    os.remove(os.path.join(sdir, ST.STAT_CACHE))
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "unpushed_changes" and r.detail["writes"] == "nothing" and r.detail["differing"]["counts"]["sha_or_size_differs"] == 1
    assert not os.path.exists(os.path.join(sdir, ST.STAT_CACHE)), "a dry run writes nothing, the cache included"
    assert ST.load_state(root)["components"]["dev"]["generation"]["state"] == "unpushed_changes", "a dry run records nothing"
    # the producer pushes the change: verified again, on the new generation
    pushed2 = ST.push_component(root, dest, comp, runner=Runner(), session="sess")
    assert pushed2.outcome == "published" and pushed2.content_sha8 == r.content_sha8
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "verified"
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "verified" and r.published == pushed2.content_sha8
    # the grading host mirror gone: red, by name
    comp_absent = ST.Component("dev", str(tmp_path / "absent"), "cluster_a", producer="grading_host:sess")
    r = ST.run_mirror(root, dest, comp_absent, runner=Runner())
    assert r.outcome == "no_grading_host_mirror" and r.level == "red" and not r.ok


def test_transitional_carry_publishes_on_the_producers_behalf_records_mode_carry_and_is_reported_overdue(tmp_path):
    w = _world(tmp_path, paced={"base": _canon(tmp_path / "p"), "authority": "cluster_a", "producer": "grading_host:pacer", "transitional_carry": True,
                                "carry_deadline": "2099-01-01", "min_publish_interval_s": 3600, "exclude": EXCL})
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["carry"]
    base, gd = w["base"], dest.gen_dir("carry")
    t0 = 1_760_000_000_000_000_000
    _set_mtimes(base, {"a/b/f1.json": t0 + 1_000_000_000, "a/f2.jsonl": t0 + 2_000_000_000, "top.txt": t0 + 7_000_000_000, "x/deep/er/f3.bin": t0 + 3_000_000_000})
    # nothing published and the producer has not adopted the push tool: the registry publishes on its behalf -- the push flow, mode carry
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "published" and r.level == "green" and r.published == r.content_sha8, r.as_dict()
    assert r.detail["carry"] == {"deadline": "2020-01-01", "overdue": True, "producer": "grading_host:sess"}
    sha8 = r.content_sha8
    pub = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    assert pub["mode"] == "carry" and pub["producer"] == "grading_host:sess" and pub["producer_session"] == "sess" and pub["transitional_carry"] is True
    assert pub["carry_deadline"] == "2020-01-01" and "OVERDUE" in pub["note"] and pub["change_utc"] == _fmt(t0 + 7_000_000_000), "change_utc = the grading host mtime of the changed files"
    assert pub["authority"] == "cluster_a" and os.readlink(f"{gd}/current") == sha8 and os.path.isfile(f"{gd}/SEALED-{sha8}.json")
    assert any(a[0] == "sbatch" for a in rn.calls), "the carry is a real publication: copied, verified by the job, sealed, recorded"
    st = ST.load_state(root)["components"]["carry"]
    assert st["published"] == sha8 and st["generation"]["state"] == "published" and st["carry"]["overdue"] is True and st["carry_overdue"] is True
    # the next tick: equal -> verified (nothing copied); the carry facts still ride in the result
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verified" and r.detail["carry"]["overdue"] is True and _no_copy(rn)
    # a change with a known mtime: the carry publishes the next generation; its record names the previous one and the changed file's mtime
    _write(os.path.join(base, "a/f2.jsonl"), b"row1\nrow2\nrow3\n")
    _set_mtimes(base, {"a/f2.jsonl": t0 + 9_000_000_000})
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "published" and r.content_sha8 != sha8
    pub2 = json.load(open(f"{gd}/PUBLISHED-{r.content_sha8}.json"))
    assert pub2["mode"] == "carry" and pub2["previous"] == sha8 and pub2["changed_files"] == 1 and pub2["change_utc"] == _fmt(t0 + 9_000_000_000)
    # a dry run of a carry says so and publishes nothing
    _write(os.path.join(base, "top.txt"), b"again\n")
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "carry" and os.readlink(f"{gd}/current") == pub2["content_sha8"]
    # min_publish_interval_s defers the carry like a push: nothing copied, green, the next tick carries
    paced = w["cfg"].components["paced"]
    r1 = ST.run_mirror(root, dest, paced, runner=Runner())
    assert r1.outcome == "published" and json.load(open(f"{dest.gen_dir('paced')}/PUBLISHED-{r1.content_sha8}.json"))["mode"] == "carry"
    _write(os.path.join(paced.base, "top.txt"), b"paced change\n")
    rn = Runner()
    r2 = ST.run_mirror(root, dest, paced, runner=rn)
    assert r2.outcome == "deferred" and r2.level == "green" and r2.ok and r2.published == r1.content_sha8 and r2.detail["min_publish_interval_s"] == 3600, r2.as_dict()
    assert _no_copy(rn) and os.readlink(f"{dest.gen_dir('paced')}/current") == r1.content_sha8
    assert ST.load_state(root)["components"]["paced"]["generation"]["state"] == "deferred" and r2.detail["carry"]["overdue"] is False


def _unseal_rmtree(top):
    """Remove a sealed generation tree (read-only directories) the way an operator would: writable again, then gone."""
    for dp, _dns, _fns in os.walk(top):
        os.chmod(dp, 0o755)
    shutil.rmtree(top)


def test_the_controllers_bound_rides_into_the_transitional_carry_on_every_route_of_stores_mirror(tmp_path, monkeypatch, capsys):
    """Review round 3: the window runs `stores mirror --bound-s 3600` over every component, and a `cluster_a` component with
    `transitional_carry` whose grading host copy CHANGED is carried through the authority route -- which dropped the bound (`run_mirror` handed it
    to the `grading_host_copy` route only; `mirror_authority` had no such parameter and its carry called `mirror_component` without one), so the
    carry's verify watch ran under the destination's four-hour default and a stalled verifier held the window, timers stopped and the
    maintenance flag up, for hours. The bound now rides `run_mirror` -> `mirror_authority` -> the carry's `mirror_component` on every path
    that carries (nothing published, changed contents, a dangling `current`), through the command the window runs, and the
    cancel-and-confirm the bound exists for is reached through the authority route too."""
    w = _world(tmp_path)
    root, dest, cfgp, comp = w["root"], w["dest"], w["cfgp"], w["cfg"].components["carry"]
    base, gd = w["base"], dest.gen_dir("carry")
    four_hours = float(dest.watch_max_hours) * 3600.0
    assert four_hours == 4 * 3600.0, "the destination's default watch bound, which the authority route fell back to"

    def watch_bound(rn):
        calls = [c for c in rn.calls if c[0] == WATCH]
        assert len(calls) == 1, calls
        return calls[0][calls[0].index("--max-hours") + 1], rn.watch_timeouts

    # (i) nothing published: the first carry through the command, bounded as the window's store-mirror step bounds it (3600 s -> --max-hours 2)
    rn = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, cfgp)
    assert cli.main(["stores", "mirror", "--component", "carry", "--bound-s", "3600"]) == 0
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first["outcome"] == "published" and first["carry"]["producer"] == "grading_host:sess", first      # as_dict flattens `detail`
    assert watch_bound(rn) == ("2", [3600.0]), (watch_bound(rn), four_hours)
    assert first["watch"]["bound_s"] == 3600.0 and first["watch"]["max_hours"] == 2
    sha8 = first["content_sha8"]
    rec = ST.load_state(root)["components"]["carry"]["generation"]
    assert os.readlink(f"{gd}/current") == sha8 and rec["state"] == "published" and rec["job_id"] == first["job_id"]
    # (ii) the finding's scenario: the grading host copy CHANGED and the window's command carries it -- the watch is bounded by the window's hour
    _write(os.path.join(base, "a/f2.jsonl"), b"row1\nrow2\nrow3\n")
    rn = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn)
    assert cli.main(["stores", "mirror", "--component", "carry", "--bound-s", "3600"]) == 0
    changed = json.loads(capsys.readouterr().out.splitlines()[0])
    assert changed["outcome"] == "published" and changed["content_sha8"] != sha8 and changed["carry"]["producer"] == "grading_host:sess", changed
    assert watch_bound(rn) == ("2", [3600.0]), (watch_bound(rn), four_hours)
    assert changed["watch"]["bound_s"] == 3600.0 and changed["watch"]["max_hours"] == 2
    sha8b = changed["content_sha8"]
    pub = json.load(open(f"{gd}/PUBLISHED-{sha8b}.json"))
    assert pub["mode"] == "carry" and pub["previous"] == sha8 and os.readlink(f"{gd}/current") == sha8b
    rec = ST.load_state(root)["components"]["carry"]["generation"]
    assert rec["state"] == "published" and rec["job_id"] == changed["job_id"] and rec["content_sha8"] == sha8b
    # (iii) a dangling `current` re-copied under the authority stage's bound (1800 s -> --max-hours 1), through run_mirror as the stage calls it
    _unseal_rmtree(f"{gd}/{sha8b}")
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn, bound_s=1800)
    assert r.outcome == "republished_dangling_current" and r.content_sha8 == sha8b and r.published == sha8b, r.as_dict()
    assert watch_bound(rn) == ("1", [1800.0]) and r.detail["watch"]["bound_s"] == 1800.0 and r.detail["watch"]["max_hours"] == 1
    assert os.readlink(f"{gd}/current") == sha8b and os.path.isdir(f"{gd}/{sha8b}")
    # (iv) the bound's purpose, reached through the authority route: a verifier that outlives it is cancelled BY ID and confirmed terminal;
    # the previous generation stays current and the new tree stands unrenamed
    _write(os.path.join(base, "top.txt"), b"a third change\n")
    srn = SlurmRunner()
    r = ST.run_mirror(root, dest, comp, runner=srn, bound_s=1.0)
    assert r.outcome == "verify_cancelled" and r.detail["carry"]["producer"] == "grading_host:sess", r.as_dict()
    assert r.job_id and srn.scancel_calls == [[r.job_id]] and r.detail["cancel"]["confirmed"] is True and srn.watch_timeouts == [1.0]
    assert r.detail["watch"]["state"] == "WATCH-TIMEOUT" and r.detail["watch"]["bound_s"] == 1.0 and srn.cancelled == {r.job_id}
    assert os.readlink(f"{gd}/current") == sha8b and os.path.isdir(f"{gd}/{r.content_sha8}") and r.content_sha8 != sha8b
    assert not any(n.startswith(".superseded-") or n.startswith(".invalid-") for n in os.listdir(gd))
    rec = ST.load_state(root)["components"]["carry"]["generation"]
    assert rec["state"] == "verify_cancelled" and rec["job_id"] == r.job_id and rec["bound_s"] == 1.0
    # and with no bound the destination's default stands on this route as on the other, a whole hour past it (4 h -> --max-hours 5)
    _write(os.path.join(base, "top.txt"), b"a fourth change\n")
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "published" and watch_bound(rn) == ("5", [four_hours]), (r.as_dict(), watch_bound(rn))


# --------------------------------------------------------------------------------------------
# a recorded verifier survives every outcome of the authority route until a run proves it terminal
# --------------------------------------------------------------------------------------------


def _queries(rn, job):
    return [c for c in rn.calls if c[0] == "bash" and "squeue" in c[2] and "sacct" in c[2] and job in c[2]]


def test_a_recorded_verifier_survives_an_absent_grading_host_mirror_and_holds_the_authority_route_until_proved_terminal(tmp_path):
    """Review round 4 MEDIUM: a transitional component's bounded carry left `awaiting_verify` with its job id (the cancel unconfirmed) and
    the next `stores mirror` found the grading host directory absent: `mirror_authority` wrote `no_grading_host_mirror` OVER that record -- `finish`
    set the generation unconditionally and `update_state` merges shallowly -- so when the directory returned the carry's guard, which
    reads only the record, saw no job and renamed and recopied the generation the verifier still read. Now the authority route settles
    the recorded job FIRST, before it looks for the grading host directory: unproved -> `verify_unconfirmed` with the record standing (job id,
    bound, watch) and the absent mirror noted beside it (`also`), never instead of it; live -> `verify_in_flight`, zero copies, renames
    and pointer switches; proved terminal -> the run goes on as before. On the tree before this change the second tick records
    `no_grading_host_mirror` with no job id and the fourth renames the verifier's tree. The held record is produced by the lost-watch route
    (MISSING, the job in neither reading: one read, no wait) rather than the unconfirmed cancel, whose confirmation waits out
    `CANCEL_BOUND_S` on the wall clock under an unreadable queue; the record either leaves is the one the route must preserve."""
    w = _world(tmp_path)
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["carry"]
    base, gd = w["base"], dest.gen_dir("carry")
    first = ST.run_mirror(root, dest, comp, runner=Runner())
    assert first.outcome == "published", first.as_dict()
    sha8 = first.content_sha8
    # the grading host copy changes; the bounded carry's watcher LOSES its verifier (MISSING) and the one reading finds it in neither the queue
    # nor the accounting: the record keeps awaiting_verify with the job id, the bound and the watch -- the branch's verify_unconfirmed handling
    _write(os.path.join(base, "top.txt"), b"changed under a verifier\n")
    srn = SlurmRunner(fail_watch=MISSING_LINE)
    srn.sbatch_to = None
    held = ST.run_mirror(root, dest, comp, runner=srn, bound_s=1800.0)
    assert held.outcome == "verify_unconfirmed" and held.detail["lost_watch"]["refusal"] == "verify_unconfirmed" and srn.scancel_calls == [], held.as_dict()
    job, sha8b = held.job_id, held.content_sha8
    rec = ST.load_state(root)["components"]["carry"]["generation"]
    assert rec["state"] == "awaiting_verify" and rec["job_id"] == job and rec["bound_s"] == 1800.0 and rec["watch"]["state"] == "MISSING"
    assert os.readlink(f"{gd}/current") == sha8 and os.path.isdir(f"{gd}/{sha8b}") and sha8b != sha8
    before = sorted(os.listdir(gd))
    # the grading host directory absent on the next tick, the queue still unreadable: the record STANDS -- state, job id, bound, cancel -- with
    # this run's settle reading and the absent mirror noted beside it; the outcome is the hold, not no_grading_host_mirror
    away = base + ".away"
    os.rename(base, away)
    rn2 = SlurmRunner()
    rn2.queue_down = True
    r2 = ST.run_mirror(root, dest, comp, runner=rn2)
    assert r2.outcome == "verify_unconfirmed" and r2.level == "red" and not r2.ok and r2.published == sha8, r2.as_dict()
    assert r2.detail["awaiting_verify"]["job_id"] == job and r2.detail["awaiting_verify"]["refusal"] == "verify_unconfirmed"
    assert r2.detail["also"] == ["no_grading_host_mirror"] and "could not be read" in r2.detail["why"] and r2.detail["current"] == sha8
    assert _no_copy(rn2) and rn2.scancel_calls == [] and len(_queries(rn2, job)) == 1
    assert sorted(os.listdir(gd)) == before and os.readlink(f"{gd}/current") == sha8
    st2 = ST.load_state(root)["components"]["carry"]
    g2 = st2["generation"]
    assert g2["state"] == "awaiting_verify" and g2["job_id"] == job and g2["bound_s"] == 1800.0 and g2["watch"]["state"] == "MISSING", g2
    assert g2["time_limit_s"] == rec["time_limit_s"] and g2["content_sha8"] == sha8b
    assert g2["settle"]["refusal"] == "verify_unconfirmed" and g2["settle"] != rec["settle"] and g2["also"] == ["no_grading_host_mirror"]
    assert st2["last_run"]["outcome"] == "verify_unconfirmed" and st2["last_run"]["also"] == ["no_grading_host_mirror"] and st2["published"] == sha8
    # a dry run of the same tick reads the queue too and says what it would conclude; it writes nothing
    rn_d = SlurmRunner()
    rn_d.queue_down = True
    r_d = ST.run_mirror(root, dest, comp, runner=rn_d, dry_run=True)
    assert r_d.outcome == "dry_run" and r_d.detail["would"] == "verify_unconfirmed" and r_d.detail["writes"] == "nothing", r_d.as_dict()
    assert r_d.detail["also"] == ["no_grading_host_mirror"] and _no_copy(rn_d) and ST.load_state(root)["components"]["carry"] == st2
    # the directory back and the verifier RUNNING in the queue: in flight, by name -- zero rsync, rename, sbatch and pointer switch; the
    # record stands with the job and nothing is noted beside it any more
    os.rename(away, base)
    rn3 = SlurmRunner()
    rn3.running.add(job)
    r3 = ST.run_mirror(root, dest, comp, runner=rn3)
    assert r3.outcome == "verify_in_flight" and r3.level == "amber" and r3.published == sha8, r3.as_dict()
    assert r3.detail["awaiting_verify"]["state"] == "RUNNING" and "still reads the" in r3.detail["why"] and "also" not in r3.detail
    assert _no_copy(rn3) and rn3.scancel_calls == [] and len(_queries(rn3, job)) == 1
    assert sorted(os.listdir(gd)) == before and os.readlink(f"{gd}/current") == sha8
    st3 = ST.load_state(root)["components"]["carry"]
    g3 = st3["generation"]
    assert g3["state"] == "awaiting_verify" and g3["job_id"] == job and g3["bound_s"] == 1800.0 and g3["watch"]["state"] == "MISSING", g3
    assert g3["settle"]["refusal"] == "verify_in_flight" and "also" not in g3 and "also" not in st3["last_run"]
    assert st3["last_run"]["outcome"] == "verify_in_flight"
    # the job proved terminal (CANCELLED in the accounting, gone from the queue): the run goes on as before -- the carry publishes the
    # changed copy; the unverified tree the bounded run left is renamed aside, never rewritten in place
    rn4 = SlurmRunner(fail_watch=None)
    rn4.cancelled.add(job)
    rn4.job = 900                               # the fake's counter restarts at 700 per runner: the fresh verifier gets an id of its own
    r4 = ST.run_mirror(root, dest, comp, runner=rn4)
    assert r4.outcome == "published" and r4.content_sha8 == sha8b and r4.published == sha8b, r4.as_dict()
    assert r4.detail["awaiting_verify_settled"]["job_id"] == job and r4.detail["awaiting_verify_settled"]["state"].startswith("CANCELLED")
    assert r4.detail["superseded"].startswith(f"{gd}/.superseded-{sha8b}-") and os.readlink(f"{gd}/current") == sha8b
    g4 = ST.load_state(root)["components"]["carry"]["generation"]
    assert g4["state"] == "published" and g4["job_id"] == r4.job_id and g4["job_id"] != job and "also" not in g4 and "settle" not in g4


def test_the_authority_routes_finish_never_writes_over_a_recorded_verifier_and_keeps_every_records_shape_when_no_job_is_pending(tmp_path, monkeypatch):
    """The other half of round 4: `finish` itself refuses to write over a record naming a job this run has not proved terminal -- whichever
    path reaches it, and for a verify-only `cluster_a` component too (its `stores push` runs the same verify job and leaves the same record
    when cut short) -- while every outcome's record keeps its documented shape where no job is pending."""
    w = _world(tmp_path)
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["dev"]
    gd = dest.gen_dir("dev")
    pushed = ST.push_component(root, dest, comp, runner=Runner(), session="sess")
    assert pushed.outcome == "published"
    sha8 = pushed.content_sha8
    absent = ST.Component("dev", str(tmp_path / "absent"), "cluster_a", producer="grading_host:sess")
    # no job pending: the records keep their shape -- verified, then no_grading_host_mirror with no job id (nothing to carry forward)
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verified" and "awaiting_verify_settled" not in r.detail and "also" not in r.detail and _queries(rn, "") == []
    g = ST.load_state(root)["components"]["dev"]["generation"]
    assert g["state"] == "verified" and g["content_sha8"] == sha8 and g["current_read"] == sha8 and g["grading_host_content_sha8"] == sha8 and g["verified_utc"]
    assert "job_id" not in g and "settle" not in g and "also" not in g
    r = ST.run_mirror(root, dest, absent, runner=Runner())
    assert r.outcome == "no_grading_host_mirror" and "also" not in r.detail
    assert ST.load_state(root)["components"]["dev"]["generation"] == {"content_sha8": None, "state": "no_grading_host_mirror", "current_read": None, "grading_host_content_sha8": None}
    # a `stores push` of this component cut short left its verifier recorded (awaiting_verify with the job id, the bound and the watch, as a
    # dead controller leaves it): the authority route holds while the job lives, the grading host copy present and equal to `current` or not
    job = "4242"
    left = {"content_sha8": "deadbeef", "state": "awaiting_verify", "job_id": job, "time_limit_s": 900, "bound_s": 1800.0, "watch": {"state": "WATCH-TIMEOUT"}}
    ST.update_state(root, "dev", {"generation": left}, dest.name)
    rn = SlurmRunner()
    rn.queue_only.add(job)
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verify_in_flight" and r.published == sha8 and r.detail["awaiting_verify"]["where"] == "queue" and _no_copy(rn), r.as_dict()
    st = ST.load_state(root)["components"]["dev"]
    assert st["generation"] == {**left, "settle": r.detail["awaiting_verify"]} and st["last_run"]["outcome"] == "verify_in_flight" and st["published"] == sha8
    # the absent directory with the job in neither reading: unconfirmed, the record standing and the absence noted beside it
    rn = SlurmRunner()
    r = ST.run_mirror(root, dest, absent, runner=rn)
    assert r.outcome == "verify_unconfirmed" and r.detail["also"] == ["no_grading_host_mirror"] and _no_copy(rn), r.as_dict()
    st = ST.load_state(root)["components"]["dev"]
    assert st["generation"] == {**left, "settle": r.detail["awaiting_verify"], "also": ["no_grading_host_mirror"]} and st["last_run"]["also"] == ["no_grading_host_mirror"]
    # the belt under the settle: a `finish` reached WITHOUT the job settled (the settle stubbed to read nothing) still never writes over
    # the record -- the outcome it would have recorded rides beside it, and `last_run` names it
    nothing = {"job_id": None, "state": None, "where": None, "refusal": None, "why": None}
    monkeypatch.setattr(ST, "settle_awaiting_verify", lambda *a, **k: dict(nothing))
    for component, outcome in ((absent, "no_grading_host_mirror"), (comp, "verified")):
        rn = SlurmRunner()
        r = ST.run_mirror(root, dest, component, runner=rn)
        assert r.outcome == outcome and r.detail["also"] == [outcome] and _no_copy(rn), r.as_dict()
        st = ST.load_state(root)["components"]["dev"]
        assert st["generation"] == {**left, "also": [outcome]} and st["last_run"]["outcome"] == outcome and st["last_run"]["also"] == [outcome], st
    monkeypatch.undo()
    # the job proved terminal: the run goes on and the fresh record replaces the held one
    rn = SlurmRunner()
    rn.completed.add(job)
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verified" and r.detail["awaiting_verify_settled"]["state"] == "COMPLETED" and _no_copy(rn), r.as_dict()
    assert "also" not in r.detail and os.readlink(f"{gd}/current") == sha8
    g = ST.load_state(root)["components"]["dev"]["generation"]
    assert g["state"] == "verified" and g["content_sha8"] == sha8 and "job_id" not in g and "settle" not in g and "also" not in g


# --------------------------------------------------------------------------------------------
# review re-check MEDIUM: a held generation is advertised as published only with its evidence
# --------------------------------------------------------------------------------------------


def _pins(root):
    """The map `cluster_reads.plan` pins the window's generations from, built exactly as the plan builds it: every component whose
    record carries a non-empty `published` (`cluster_reads.plan`, the `published is None` branch)."""
    return {slug: rec.get("published") for slug, rec in (ST.load_state(root).get("components") or {}).items() if rec.get("published")}


def test_a_held_generation_is_advertised_as_published_only_with_its_verification_and_publication_evidence(tmp_path):
    """Review re-check MEDIUM: `hold()` wrote the `current` target into `published` as a matter of course -- without checking that the
    generation was VERIFIED, and over a recorded `published: null` / `current_dangling`. The trigger is the dangling-pointer repair:
    it drops `published`, recopies the SAME sha under `current` (the stale report moved aside, so nothing attests the fresh tree) and
    submits the verify; when the watch cannot prove that job terminal it leaves `awaiting_verify` and NO publication. On the next
    authority-mirror tick the unproved verifier entered `hold()`, which wrote that UNVERIFIED sha back into `published` -- and
    `cluster_reads.plan` pins any non-empty `published`, so an adapter was routed at an incomplete destination tree and the cluster
    attestation refused it at the far end, the read wasted, instead of the component declining at plan time. Now the hold advertises
    `current` only with the evidence of its publication beside it (`held_publication`: the tree a directory, a live VERIFIED report
    admitting its manifest, the SEALED record and the PUBLISHED record) and never raises `published` from None while a recorded dangling
    repair stands; the record -- job id, bound, watch -- is carried as before. The SEAL is one of those gates on this route too (the
    sibling rule from registry-hardening-2c, which gates publication on the seal everywhere else): a held `current` whose own
    `SEALED-<sha8>.json` is not beside it is a publication that never finished, and `admit_generation` refuses it at the far end."""
    w = _world(tmp_path)
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["carry"]
    base, gd = w["base"], dest.gen_dir("carry")
    first = ST.run_mirror(root, dest, comp, runner=Runner())
    assert first.outcome == "published", first.as_dict()
    sha8 = first.content_sha8
    # the plain case, today's behavior: a verified, recorded generation held on a live verifier IS what the shards read -- `published`
    # stays `current` and the plan pins it
    job = "4242"
    left = {"content_sha8": sha8, "state": "awaiting_verify", "job_id": job, "time_limit_s": 900, "bound_s": 1800.0, "watch": {"state": "WATCH-TIMEOUT"}}
    ST.update_state(root, "carry", {"generation": left}, dest.name)
    rn = SlurmRunner()
    rn.queue_only.add(job)
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verify_in_flight" and r.published == sha8 and "not_published" not in r.detail, r.as_dict()
    assert r.detail["generation_record"]["verified_matches"] is True and r.detail["generation_record"]["published_record"] is True
    st = ST.load_state(root)["components"]["carry"]
    assert st["published"] == sha8 and st["generation"] == {**left, "settle": r.detail["awaiting_verify"]} and _no_copy(rn)
    assert _pins(root)["carry"] == sha8 and "carry" in ST.published_attestations(root)
    # THE SEAL IS ONE OF THE GATES: the same hold with the generation's own SEALED record moved aside advertises nothing. The seal is
    # written inside the publication step, so its absence IS a publication left unfinished -- and `admit_generation` refuses an unsealed
    # pin at the far end, so the plan pinning it would buy the window a refusal rather than a read
    stale_seal = f"{gd}/.stale-SEALED-{sha8}-20261007T000000Z.json"
    os.rename(f"{gd}/SEALED-{sha8}.json", stale_seal)
    rn_u = SlurmRunner()
    rn_u.queue_only.add(job)
    unsealed = ST.run_mirror(root, dest, comp, runner=rn_u)
    assert unsealed.outcome == "verify_in_flight" and unsealed.published is None, unsealed.as_dict()
    assert unsealed.detail["not_published"]["problems"] == ["current_unsealed"], unsealed.detail["not_published"]
    assert f"no SEALED-{sha8}.json seals its directories" in unsealed.detail["not_published"]["why"]
    assert unsealed.detail["generation_record"]["verified_matches"] is True and unsealed.detail["generation_record"]["sealed"] is False
    st = ST.load_state(root)["components"]["carry"]
    assert st["published"] is None and "carry" not in _pins(root) and _no_copy(rn_u)
    assert st["generation"]["state"] == "awaiting_verify" and st["generation"]["job_id"] == job
    # the seal back beside it: the plain case advertises again, so it is the seal that was decisive and nothing else
    os.rename(stale_seal, f"{gd}/SEALED-{sha8}.json")
    rn_s = SlurmRunner()
    rn_s.queue_only.add(job)
    again = ST.run_mirror(root, dest, comp, runner=rn_s)
    assert again.outcome == "verify_in_flight" and again.published == sha8 and "not_published" not in again.detail, again.as_dict()
    assert ST.load_state(root)["components"]["carry"]["published"] == sha8 and _pins(root)["carry"] == sha8 and _no_copy(rn_s)
    # the published tree renamed away: the repair recopies the same sha under `current` and submits the verify, and its watcher LOSES the
    # job (in neither the queue nor the accounting) -- `published` dropped, the dangle recorded, the verifier named and unproved
    aside = f".failed-{sha8}-20261007T000000Z"
    os.rename(f"{gd}/{sha8}", f"{gd}/{aside}")
    rn = SlurmRunner(fail_watch=MISSING_LINE)
    rn.completed.add(job)                       # the planted verifier reads terminal: this run goes on to the repair
    rn.sbatch_to = None                         # its OWN verifier is in neither reading
    repair = ST.run_mirror(root, dest, comp, runner=rn, bound_s=1800.0)
    assert repair.outcome == "verify_unconfirmed" and repair.detail["current_dangling"] is True, repair.as_dict()
    job2 = repair.job_id
    assert job2 != job and repair.detail["renamed_aside"] == [aside] and os.readlink(f"{gd}/current") == sha8 and os.path.isdir(f"{gd}/{sha8}")
    assert not os.path.exists(f"{gd}/VERIFIED-{sha8}.json") and os.path.isfile(f"{gd}/PUBLISHED-{sha8}.json"), "the tree stands, nothing attests it"
    st = ST.load_state(root)["components"]["carry"]
    assert st["published"] is None and st["current_dangling"]["content_sha8"] == sha8 and st["current_dangling"]["renamed_aside"] == [aside]
    assert st["generation"]["state"] == "awaiting_verify" and st["generation"]["job_id"] == job2 and "carry" not in _pins(root)
    before = sorted(os.listdir(gd))
    # THE FINDING: the next tick finds that verifier RUNNING and holds -- and the hold leaves `published` None. The record still names the
    # job, the dangle note stands, and the plan pins nothing for the component
    rn = SlurmRunner()
    rn.running.add(job2)
    held = ST.run_mirror(root, dest, comp, runner=rn)
    assert held.outcome == "verify_in_flight" and held.published is None and held.detail["current"] == sha8, held.as_dict()
    assert held.detail["awaiting_verify"]["job_id"] == job2 and held.detail["awaiting_verify"]["state"] == "RUNNING"
    assert held.detail["not_published"]["problems"] == ["current_unverified", "current_unsealed", "current_dangling"], held.detail["not_published"]
    assert f"no VERIFIED-{sha8}.json admits its manifest" in held.detail["not_published"]["why"]
    assert f"no SEALED-{sha8}.json seals its directories" in held.detail["not_published"]["why"], "the repair moved the replaced tree's seal aside"
    assert held.detail["generation_record"]["verified_matches"] is False and held.detail["generation_record"]["published_record"] is True
    assert _no_copy(rn) and rn.scancel_calls == [] and sorted(os.listdir(gd)) == before and os.readlink(f"{gd}/current") == sha8
    st = ST.load_state(root)["components"]["carry"]
    assert st["published"] is None, "the hold never raises `published` from None onto an unverified, unresolved pointer"
    assert st["current_dangling"]["content_sha8"] == sha8 and st["last_run"]["not_published"] == held.detail["not_published"]["why"]
    assert st["generation"]["state"] == "awaiting_verify" and st["generation"]["job_id"] == job2 and st["generation"]["bound_s"] == 1800.0
    assert st["generation"]["settle"]["refusal"] == "verify_in_flight" and st["last_run"]["outcome"] == "verify_in_flight"
    assert "carry" not in _pins(root) and "carry" not in ST.published_attestations(root), "no adapter is routed at the incomplete tree"
    # the dry run of the same tick concludes the same and writes nothing
    rn_d = SlurmRunner()
    rn_d.running.add(job2)
    r_d = ST.run_mirror(root, dest, comp, runner=rn_d, dry_run=True)
    assert r_d.outcome == "dry_run" and r_d.detail["would"] == "verify_in_flight" and r_d.published is None and r_d.detail["writes"] == "nothing"
    assert ST.load_state(root)["components"]["carry"] == st and _no_copy(rn_d)
    # the job proved terminal and the grading host copy changed: the carry publishes, its verify matching -- THAT raises `published` and clears
    # the dangle, and the plan pins the fresh generation again
    _write(os.path.join(base, "top.txt"), b"changed after the repair\n")
    rn = SlurmRunner(fail_watch=None)
    rn.cancelled.add(job2)
    rn.job = 900                                # the fake's counter restarts at 700 per runner: the fresh verifier gets an id of its own
    done = ST.run_mirror(root, dest, comp, runner=rn)
    assert done.outcome == "published" and done.published == done.content_sha8 and done.content_sha8 != sha8, done.as_dict()
    st = ST.load_state(root)["components"]["carry"]
    assert st["published"] == done.content_sha8 and st["current_dangling"] is None and st["generation"]["state"] == "published"
    assert _pins(root)["carry"] == done.content_sha8 and os.readlink(f"{gd}/current") == done.content_sha8
    assert os.path.isfile(f"{gd}/VERIFIED-{done.content_sha8}.json")


def test_the_grading_host_copy_routes_settle_refusal_holds_published_at_none_through_an_unresolved_dangling_repair(tmp_path):
    """The general half of the same rule: `mirror_component`'s settle refusal is a hold too -- it copies, renames and switches nothing --
    and it raised `published` to the live `current` the same way, so a `grading_host_copy` component whose dangling repair left an unproved
    verifier had its unverified sha pinned by the next tick. Both routes read the one rule (`held_publication`), the seal among its gates
    (the repair moves the replaced tree's SEALED record aside with its reports, and the replacement earns its own only when it publishes)."""
    w = _world(tmp_path)
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["dc"]
    gd = dest.gen_dir("dc")
    first = ST.run_mirror(root, dest, comp, runner=Runner())
    assert first.outcome == "published", first.as_dict()
    sha8 = first.content_sha8
    os.rename(f"{gd}/{sha8}", f"{gd}/.invalid-{sha8}-20261007T000000Z")
    rn = SlurmRunner(fail_watch=MISSING_LINE)
    rn.sbatch_to = None
    repair = ST.run_mirror(root, dest, comp, runner=rn)
    assert repair.outcome == "verify_unconfirmed" and repair.detail["current_dangling"] is True, repair.as_dict()
    job = repair.job_id
    assert ST.load_state(root)["components"]["dc"]["published"] is None and "dc" not in _pins(root)
    # the settle refusal of the next tick: the record stands, and `published` stays None instead of naming the unverified pointer
    rn = SlurmRunner()
    rn.running.add(job)
    held = ST.run_mirror(root, dest, comp, runner=rn)
    assert held.outcome == "verify_in_flight" and held.published is None, held.as_dict()
    assert held.detail["awaiting_verify"]["job_id"] == job
    assert held.detail["not_published"]["problems"] == ["current_unverified", "current_unsealed", "current_dangling"], held.detail["not_published"]
    assert rn.count("rsync") == 0 and not any(c[0] == "sbatch" for c in rn.calls) and os.readlink(f"{gd}/current") == sha8
    st = ST.load_state(root)["components"]["dc"]
    assert st["published"] is None and st["current_dangling"]["content_sha8"] == sha8 and "dc" not in _pins(root)
    assert st["generation"]["state"] == "awaiting_verify" and st["generation"]["job_id"] == job and st["generation"]["settle"]["refusal"] == "verify_in_flight"


# --------------------------------------------------------------------------------------------
# a frozen producer: verify only, red on a difference, never repaired
# --------------------------------------------------------------------------------------------


def test_frozen_producer_differs_is_red_and_never_repaired(tmp_path):
    w = _world(tmp_path)
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["frozen"]
    gd = dest.gen_dir("frozen")
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "no_current" and r.level == "red" and not r.ok and _no_copy(rn) and not os.path.exists(gd), "the mirror never publishes a frozen store"
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "no_current" and r.level == "green"
    moved = ST.push_component(root, dest, comp, runner=Runner(), note="the move")          # the one-time move publishes it
    assert moved.outcome == "published"
    sha8 = moved.content_sha8
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "verified" and r.published == sha8
    _write(os.path.join(w["base"], "top.txt"), b"someone edited a frozen store\n")
    listing_before = sorted(os.listdir(gd))
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "frozen_differs" and r.level == "red" and not r.ok and r.published == sha8, r.as_dict()
    assert r.detail["differing"]["sha_or_size_differs"] == ["top.txt"] and "never repaired" in r.detail["why"]
    assert _no_copy(rn) and os.readlink(f"{gd}/current") == sha8 and sorted(os.listdir(gd)) == listing_before, "nothing copied, nothing switched, nothing renamed"
    assert not any(n.startswith((".incoming", ".failed", ".invalid")) for n in os.listdir(gd))
    st = ST.load_state(root)["components"]["frozen"]
    assert st["published"] == sha8 and st["generation"]["state"] == "frozen_differs" and st["last_run"]["outcome"] == "frozen_differs"
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "frozen_differs"


# --------------------------------------------------------------------------------------------
# a cluster producer: the pull into the registry-owned mirror, names restored, hash-verified
# --------------------------------------------------------------------------------------------


def _cluster_tree(tmp_path, name="clusrc"):
    src = str(tmp_path / name)
    _write(os.path.join(src, "d:x", "f%1.txt"), b"one\n")
    _write(os.path.join(src, "d:x", "deep", "g.txt"), b"deep\n")
    _write(os.path.join(src, "plain.txt"), b"two\n")
    _write(os.path.join(src, "gone.txt"), b"to be removed by the next generation\n")
    os.symlink("d:x/f%1.txt", os.path.join(src, "lnk:a"))        # an escaped name AND an escaped target
    os.symlink("plain.txt", os.path.join(src, "plainlink"))
    return src


def test_cluster_producer_is_pulled_into_the_registry_owned_mirror_hash_verified_with_names_restored(tmp_path):
    w = _world(tmp_path)
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["clu"]
    gd = dest.gen_dir("clu")
    target = ST.grading_host_mirror_path(root, comp)
    assert target == os.path.join(root, "stores", "mirror", "clu") and comp.grading_host_mirror is None, "a cluster producer's mirror is registry-owned, under the root"
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "no_current" and r.level == "red" and "--stores-dir" in r.detail["why"] and _no_copy(rn) and not os.path.exists(target)
    state_before = open(ST.state_path(root)).read()                 # the no_current run above recorded itself; the dry run adds nothing
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "no_current" and r.detail["writes"] == "nothing" and open(ST.state_path(root)).read() == state_before
    # the producer publishes inside its job (the stdlib push); the generation carries the ESCAPED names
    src = _cluster_tree(tmp_path)
    oc = ST.push_on_cluster("clu", src, dest.stores_dir, session="job1")
    assert oc.outcome == "published", oc.as_dict()
    sha8 = oc.content_sha8
    assert sorted(os.listdir(f"{gd}/{sha8}")) == ["d%3Ax", "gone.txt", "lnk%3Aa", "plain.txt", "plainlink"]
    # the pull: rsync --delete into the mirror (and only there), names and link targets restored, every pulled byte hashed
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "pulled" and r.level == "green" and r.published == sha8 and r.content_sha8 == sha8 and r.n_files == 4, r.as_dict()
    assert r.detail["pull"]["renamed"] == 3 and r.detail["pull"]["relinked"] == 1 and r.detail["pulled_now"] is True and r.hashed == 4
    assert sorted(os.listdir(target)) == ["d:x", "gone.txt", "lnk:a", "plain.txt", "plainlink"] and sorted(os.listdir(f"{target}/d:x")) == ["deep", "f%1.txt"]
    assert os.readlink(f"{target}/lnk:a") == "d:x/f%1.txt" and os.readlink(f"{target}/plainlink") == "plain.txt"
    assert open(f"{target}/d:x/f%1.txt", "rb").read() == b"one\n" and _mode(target) & 0o200 and _mode(f"{target}/plain.txt") & 0o200, "the mirror is the registry's writable copy"
    rs = next(a for a in rn.calls if "rsync" in a)
    assert "--delete" not in rs and "--chmod=Du+w,Fu+w" in rs and not any(a.startswith("--link-dest") for a in rs) and rs[-1] == f"{target}/" and rs[-2] == f"{gd}/{sha8}/"
    assert r.detail["pull"]["pruned_files"] == 0 and r.detail["pull"]["pruned_dirs"] == 0
    assert not any(a[0] == "sbatch" for a in rn.calls), "the pull verifies by hashing here; no job"
    st = ST.load_state(root)["components"]["clu"]
    assert st["published"] == sha8 and st["generation"]["state"] == "pulled" and st["grading_host_mirror"] == target and st["latest_manifest"]["content_sha8"] == sha8
    sdir = ST.component_state_dir(root, "clu")
    assert os.path.isfile(os.path.join(sdir, f"MANIFEST-{sha8}.stat.json")), "the pin's grading host manifest copy, stat snapshot included"
    # the next tick: the mirror already is the generation (a stat walk, nothing copied)
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "unchanged" and r.detail["pulled_now"] is False and _no_copy(rn) and r.from_cache == 4
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "unchanged" and r.detail["would_pull_to"] == target
    # the producer publishes a NEW generation (one file gone, one changed): the pull updates the mirror and removes the vanished file
    os.remove(os.path.join(src, "gone.txt"))
    _write(os.path.join(src, "plain.txt"), b"two, revised\n")
    oc2 = ST.push_on_cluster("clu", src, dest.stores_dir, session="job2")
    assert oc2.outcome == "published" and oc2.content_sha8 != sha8
    _write(f"{target}/stray/left_by_someone.txt", b"not the generation's\n")        # nothing else may live in the registry-owned mirror
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "pulled" and r.published == oc2.content_sha8 and not os.path.exists(f"{target}/gone.txt") and open(f"{target}/plain.txt", "rb").read() == b"two, revised\n"
    assert sorted(os.listdir(target)) == ["d:x", "lnk:a", "plain.txt", "plainlink"], "the vanished file and the stray entry pruned by name; rsync deleted nothing"
    assert r.detail["pull"]["pruned_files"] == 2 and r.detail["pull"]["pruned_dirs"] == 1 and not os.path.exists(f"{target}/stray")
    assert r.detail["pull"]["replaced"] == 2 and r.detail["pull"]["renamed"] == 3, "the previous pull's restored `d:x` and `lnk:a` replaced by the fresh escaped copies"
    assert open(f"{target}/d:x/f%1.txt", "rb").read() == b"one\n" and os.readlink(f"{target}/lnk:a") == "d:x/f%1.txt"
    # a generation whose bytes are not its manifest's: the pulled tree does not verify -> red, by name; the mirror is left for the next pull
    cur = oc2.content_sha8
    _unseal_edit_reseal(f"{gd}/{cur}", "plain.txt", b"two, REVISED\n")   # same size, bytes differ; modes and mtime restored
    os.remove(f"{target}/plain.txt")                                 # make the pull copy it
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "pull_verify_failed" and r.level == "red" and not r.ok and r.detail["differing"]["sha_or_size_differs"] == ["plain.txt"], r.as_dict()
    assert r.published == cur and os.readlink(f"{gd}/current") == cur, "the pull never touches the destination"


def test_grading_host_mirror_path_resolves_declared_cluster_default_and_base(tmp_path):
    base = _canon(tmp_path)
    cfg = ST.load_config(_cfg(tmp_path, {
        "declared": {"base": base, "authority": "cluster_a", "producer": "cluster", "grading_host_mirror": str(tmp_path / "pulled_here")},
        "clu": {"base": base, "authority": "cluster_a", "producer": "cluster"},
        "dev": {"base": base, "authority": "cluster_a", "producer": "grading_host:s"},
        "dc": {"base": base, "authority": "grading_host_copy"}},
        adapters={"relu_paired": {"shards": 1, "components": ["declared", "clu", "dev", "dc"]}}))
    assert ST.grading_host_mirror_path("/r", cfg.components["declared"]) == str(tmp_path / "pulled_here")
    assert ST.grading_host_mirror_path("/r", cfg.components["clu"]) == "/r/stores/mirror/clu" and cfg.components["clu"].grading_host_mirror is None
    assert ST.grading_host_mirror_path(None, cfg.components["clu"]).startswith(ST.DEFAULT_ROOT)
    assert ST.grading_host_mirror_path("/r", cfg.components["dev"]) == base == cfg.components["dev"].grading_host_mirror
    assert ST.grading_host_mirror_path("/r", cfg.components["dc"]) == base
    # check-authority reads the mirror's presence at the resolved path
    rec = ST.check_authority(str(tmp_path / "reg"), cfg, cfg.destination("cluster_a"), runner=Runner())
    assert rec["components"]["clu"]["grading_host_mirror"] == str(tmp_path / "reg" / "stores" / "mirror" / "clu") and rec["components"]["clu"]["grading_host_mirror_present"] is False
    assert "no_grading_host_mirror" in rec["components"]["clu"]["problems"] and rec["components"]["dev"]["grading_host_mirror_present"] is True


# --------------------------------------------------------------------------------------------
# --full: every grading host file read, the published generation re-hashed on the destination
# --------------------------------------------------------------------------------------------



def test_full_reads_every_grading_host_file_and_rehashes_the_generation_on_the_destination(world, monkeypatch):
    w = world
    root, dest, comp, gd = w["root"], w["dest"], w["comp"], w["gen_dir"]
    monkeypatch.setattr(ST, "REPORT_REREAD_PAUSE_S", 0.0)
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "published"
    sha8 = r.content_sha8
    live_report = f"{gd}/VERIFIED-{sha8}.json"
    live_before = open(live_report).read()
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "unchanged" and r.from_cache == 4 and r.hashed == 0 and "full" not in r.detail
    # --full: the cache ignored (4 hashed, 0 from the cache) and the generation re-hashed on the destination by the pinned verifier, as a job
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn, full=True)
    assert r.outcome == "unchanged" and r.level == "green" and r.hashed == 4 and r.from_cache == 0, r.as_dict()
    fr = r.detail["full"]
    assert fr["ok"] is True and fr["report_present"] and fr["counts"] == {"missing": 0, "corrupt": 0, "extra": 0} and fr["report"]["hashed"] == 4
    assert fr["report_path"].startswith(f"{dest.stores_dir}/{ST.FULL_SUBDIR}/relu_paired_run_state/VERIFIED-{sha8}-") and os.path.isfile(fr["report_path"])
    assert open(live_report).read() == live_before, "the live report is never rewritten"
    sb = next(a for a in rn.calls if a[0] == "sbatch")
    joined = " ".join(sb)
    assert "--job-name=store-full-relu_paired_run_state" in joined and "--qos=high" in joined and "--export" not in joined and "--gres" not in joined
    assert sb[sb.index(dest.sbatch_path) + 1:] == ["verify", f"{gd}/{sha8}", f"{gd}/MANIFEST-{sha8}.sha256", fr["report_path"], "2"]
    assert any(a[0] == WATCH for a in rn.calls) and fr["job_id"] == json.load(open(fr["report_path"]))["job_id"]
    st = ST.load_state(root)["components"]["relu_paired_run_state"]
    assert st["generation"]["full"]["ok"] is True and st["generation"]["state"] == "published"
    # a same-size edit in the PUBLISHED generation (the owner un-seals, edits, restores mode and mtime): the weekly full names it, renames
    # nothing, `current` untouched -- red, exit 1, the operator decides
    _unseal_edit_reseal(f"{gd}/{sha8}", "a/f2.jsonl", b"row1\nROW2\n")
    r = ST.run_mirror(root, dest, comp, runner=Runner(), full=True)
    assert r.outcome == "full_verify_failed" and r.level == "red" and not r.ok and r.published == sha8, r.as_dict()
    assert r.detail["full"]["counts"]["corrupt"] == 1 and r.detail["full"]["first"]["corrupt"] == ["a/f2.jsonl"] and "nothing renamed" in r.detail["why"]
    assert os.readlink(f"{gd}/current") == sha8 and os.path.isdir(f"{gd}/{sha8}") and not any(n.startswith(".failed") for n in os.listdir(gd))
    assert open(live_report).read() == live_before
    assert ST.load_state(root)["components"]["relu_paired_run_state"]["generation"]["state"] == "full_verify_failed"
    # a cluster_a (frozen) component's --full: verified + the re-hash; the same corruption -> full_verify_failed
    cfg2 = ST.load_config(_cfg(pathlib.Path(root).parent, {"fz": {"base": w["base"], "authority": "cluster_a", "producer": "frozen", "exclude": EXCL}},
                               adapters={"relu_paired": {"shards": 1, "components": ["fz"]}}, name="fz.yaml"))
    fz = cfg2.components["fz"]
    assert ST.push_component(root, dest, fz, runner=Runner()).outcome == "published"
    r = ST.run_mirror(root, dest, fz, runner=Runner(), full=True)
    assert r.outcome == "verified" and r.detail["full"]["ok"] is True and r.hashed == 4 and r.from_cache == 0
    r = ST.run_mirror(root, dest, fz, runner=Runner(), dry_run=True, full=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "verified + full re-hash"
    _unseal_edit_reseal(f"{dest.gen_dir('fz')}/{r.published}", "top.txt", b"TOP LEVEL\n")
    r = ST.run_mirror(root, dest, fz, runner=Runner(), full=True)
    assert r.outcome == "full_verify_failed" and r.detail["full"]["first"]["corrupt"] == ["top.txt"] and os.readlink(f"{dest.gen_dir('fz')}/current") == r.published
    # a job that does not complete is an error, never a verdict
    r = ST.run_mirror(root, dest, fz, runner=Runner(fail_watch="FAILED 9 state=TIMEOUT"), full=True)
    assert r.outcome == "error" and "TIMEOUT" in r.detail["error"]


# --------------------------------------------------------------------------------------------
# a verify report not yet visible after its job never renames the tree
# --------------------------------------------------------------------------------------------


class _HiddenReport(Runner):
    """The verify job completes but its report is written where the grading host cannot see it yet (NFS close-to-open): the shape of the live
    bigcodebench_full misread of 2026-10-04."""

    def __init__(self, hidden_suffix=".not-yet-visible"):
        super().__init__()
        self.hidden_suffix = hidden_suffix
        self.reports = []

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[0] == "sbatch":
            i = next(k for k, a in enumerate(argv) if a.endswith("store_verify.sbatch"))
            script = argv[i + 1:]
            if script[0] == "verify":
                gen, man, out, workers = script[1:5]
                self.job += 1
                self.calls.append(argv)
                rep, dest_stat = ST.verify_generation(gen, man, workers=int(workers), job_id=str(self.job))
                ST.write_verified(out + self.hidden_suffix, rep, dest_stat)
                self.reports.append(out)
                return subprocess.CompletedProcess(argv, 0, stdout=f"{self.job}\n", stderr="")
        return super().__call__(argv, timeout_s, input=input, check=check)


def test_a_verify_report_not_yet_visible_never_renames_the_tree(world, monkeypatch):
    w = world
    root, dest, comp, gd = w["root"], w["dest"], w["comp"], w["gen_dir"]
    # the mirror waits on main's bounded re-read of THIS job's report (`_await_report`, Destination.report_wait_s / report_poll_s; since the
    # merge of 2026-10-06 the branch's REPORT_REREAD_* rereads serve verify-authority's reader alone): a zero bound reads once and gives up
    dest = dataclasses.replace(dest, report_wait_s=0.0)
    rn = _HiddenReport()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verify_report_missing" and r.level == "red" and not r.ok and r.detail["report_wait"] == {"attempts": 1, "waited_s": 0.0, "last": "absent"}, r.as_dict()
    sha8 = r.content_sha8
    assert os.path.isdir(f"{gd}/{sha8}") and not any(n.startswith(".failed") for n in os.listdir(gd)), "the tree is LEFT, never renamed on a missing report"
    assert not os.path.lexists(f"{gd}/current") and "left in place" in r.detail["why"] and "next tick publishes it" in r.detail["why"]
    st = ST.load_state(root)["components"]["relu_paired_run_state"]
    assert st["generation"]["state"] == "verify_report_missing" and st["generation"]["attempts"] == 1 and st["generation"]["last_read"] == "absent" and st.get("published") is None
    # the report becomes visible: the next run publishes the verified tree through the guard, copying nothing
    out = rn.reports[0]
    os.rename(out + rn.hidden_suffix, out)
    rn2 = Runner()
    r2 = ST.run_mirror(root, dest, comp, runner=rn2)
    assert r2.outcome == "published_existing_verified" and r2.published == sha8 and os.readlink(f"{gd}/current") == sha8
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn2.calls)
    # a report that IS present and does not match still renames `.failed-` (the verify's verdict, not a visibility gap)
    _write(os.path.join(w["base"], "top.txt"), b"next generation\n")

    def corrupt_copy(gen):
        # rsync --link-dest hands a changed file's fresh copy the mode of its sealed sibling in the reference generation (0444), so the
        # owner un-seals it first, as every seal test does; the verify catches the bytes regardless of the mode
        p = os.path.join(gen, "top.txt")
        os.chmod(p, 0o644)
        _write(p, b"corrupt in copy\n")

    r3 = ST.run_mirror(root, dest, comp, runner=Runner(before_verify=corrupt_copy))
    assert r3.outcome == "verify_failed" and any(n.startswith(f".failed-{r3.content_sha8}") for n in os.listdir(gd)), r3.as_dict()


def test_the_report_wait_covers_the_measured_visibility_delay_and_both_shortcuts_probe_the_tree():
    """The VERIFIED report a cluster verify job writes becomes visible from the login node up to ~60 s later (NFS attribute cache; every
    one of the 48 `verify_failed` outcomes since 2026-09-29 was this misread): the bounded re-read waits at least twice that, 15 s apart,
    before `verify_report_missing` (which never renames the tree -- pinned below). And the two shortcuts that skip the copy read the
    destination TREE, not only the pointer or the report: `unchanged` requires `current` to name a directory that exists, and
    `published_existing_verified` requires the verified tree to be there (the source spells both probes; the behaviour is pinned by the
    dangling-current and report-without-tree tests)."""
    # the mirror's wait is Destination.report_wait_s / report_poll_s (main's `_await_report`, merged 2026-10-06): the default bound covers the
    # measured ~60 s visibility delay twice over; the branch's REPORT_REREAD_* constants still serve verify-authority's reader
    assert ST.Destination.report_wait_s == 120.0 and ST.Destination.report_poll_s == 15.0 and ST.Destination.report_wait_s >= 120
    assert ST.REPORT_REREAD_PAUSE_S == 15.0 and ST.REPORT_REREAD_TRIES * ST.REPORT_REREAD_PAUSE_S >= 120
    src = open(ST.__file__, encoding="utf-8").read()
    # both shortcuts probe the TREE through one `tree_present` reading (the spelling since the merge)
    assert "tree_present = remote.isdir(gen_path)" in src
    assert 'current_exists = bool(current) and (tree_present if current == sha8 else remote.isdir(f"{gen_dir}/{current}"))' in src
    assert "if current == sha8 and tree_present and not resuming and not pending:" in src and "if tree_present and _verified_matches(verified, pre):" in src
    # and a tree the record names as the mirror's own unfinished repair is PENDING, never the first shortcut (the closing pass on 2c)
    assert "pending = current == sha8 and tree_present and prior.get(\"content_sha8\") == sha8 and _verified_matches(verified, pre)" in src


def test_read_verified_report_rereads_a_bounded_number_of_times(monkeypatch):
    monkeypatch.setattr(ST, "REPORT_REREAD_PAUSE_S", 0.0)

    class _Remote:
        def __init__(self, appear_after):
            self.n = 0
            self.appear_after = appear_after

        def read_json(self, path):
            self.n += 1
            return {"schema": ST.VERIFIED_SCHEMA, "ok": True} if self.n > self.appear_after else {}

    rep, rereads = ST.read_verified_report(_Remote(2), "/x/VERIFIED-a.json", tries=3)
    assert rep["ok"] is True and rereads == 2
    rm = _Remote(99)
    rep, rereads = ST.read_verified_report(rm, "/x/VERIFIED-a.json", tries=3)
    assert rep == {} and rereads == 3 and rm.n == 4, "one read plus exactly `tries` re-reads, then the caller decides"
    rep, rereads = ST.read_verified_report(_Remote(0), "/x/VERIFIED-a.json", tries=3)
    assert rep["ok"] is True and rereads == 0


# --------------------------------------------------------------------------------------------
# check-authority's `unrecorded` is a problem; `stores status` shows the authority rows and the level; the CLI's --full
# --------------------------------------------------------------------------------------------


def test_status_shows_the_authority_rows_the_level_and_the_producer_and_the_cli_takes_full(world, monkeypatch, capsys):
    w = world
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", w["root"])
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    # before anything: no level, no authority row, "authority checked never"
    assert cli.main(["stores", "status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    row = data["components"]["relu_paired_run_state"]
    assert row["level"] is None and row["authority_check"] is None and data["authority_checked_utc"] is None and row["producer"] is None
    assert cli.main(["stores", "status"]) == 0
    text = capsys.readouterr().out
    assert "authority checked never" in text and text.splitlines()[-1].endswith("| - | - | unchecked")
    # a published (recorded) generation: level green; check-authority OK; the authority row carries current / verified / sealed / recorded
    assert cli.main(["stores", "mirror"]) == 0
    sha8 = json.loads(capsys.readouterr().out.splitlines()[0])["content_sha8"]
    assert cli.main(["stores", "check-authority"]) == 0
    capsys.readouterr()
    assert cli.main(["stores", "status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    row = data["components"]["relu_paired_run_state"]
    assert row["level"] == "green" and row["published"] == sha8 and data["authority_checked_utc"]
    assert row["authority_check"]["state"] == "ok" and row["authority_check"]["current"] == sha8 and row["authority_check"]["verified"] and row["authority_check"]["sealed"]
    assert row["authority_check"]["published_record"] is True and row["authority_check"]["problems"] == [] and row["authority_check"]["publish_mode"] == "mirror"
    assert cli.main(["stores", "status"]) == 0
    text = capsys.readouterr().out
    assert "relu_paired_run_state | grading_host_copy | " + sha8 in text and f"{sha8}:published" in text
    assert text.splitlines()[-1].endswith(f"| - | green | {sha8} yes/yes/yes ok")
    # the record removed (a pre-rule generation): `unrecorded` is a PROBLEM -- check-authority exits 1 and the row says so
    os.remove(f"{w['gen_dir']}/PUBLISHED-{sha8}.json")
    assert cli.main(["stores", "check-authority"]) == 1
    out = capsys.readouterr().out
    assert "relu_paired_run_state: unrecorded" in out and "check-authority NOT OK" in out
    assert cli.main(["stores", "status"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith(f"{sha8} yes/yes/no unrecorded")
    # the mirror itself refuses the unrecorded current by name (C3) and the adoption records it; then the weekly --full through the CLI
    assert cli.main(["stores", "mirror"]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[0])["outcome"] == "current_unrecorded"
    assert cli.main(["stores", "mirror", "--adopt-unrecorded"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(ST, "REPORT_REREAD_PAUSE_S", 0.0)
    assert cli.main(["stores", "mirror", "--full", "--component", "relu_paired_run_state"]) == 0
    lines = capsys.readouterr().out.splitlines()
    first, summary = json.loads(lines[0]), json.loads(lines[-1])
    assert first["outcome"] == "unchanged" and first["level"] == "green" and first["full"]["ok"] is True and first["hashed"] == 4
    assert summary["full"] is True and summary["levels"] == {"green": 1, "amber": 0, "red": 0} and summary["outcomes"] == {"unchanged": 1}
    # a cluster_a component through the CLI: the dispatch, the level in the summary, exit 0 on amber
    cfg_path = _cfg(pathlib.Path(w["root"]).parent, {"dev": {"base": w["base"], "authority": "cluster_a", "producer": "grading_host:sess", "exclude": EXCL}},
                    adapters={"relu_paired": {"shards": 1, "components": ["dev"]}}, name="dev.yaml")
    monkeypatch.setenv(ST.CONFIG_ENV, cfg_path)
    assert cli.main(["stores", "mirror", "--component", "dev"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0])["outcome"] == "unpushed_changes" and json.loads(lines[-1])["levels"]["amber"] == 1
    assert cli.main(["stores", "status"]) == 0
    assert "| grading_host:sess | amber |" in capsys.readouterr().out
    assert cli.main(["stores", "mirror", "--dry-run", "--component", "dev"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["would"] == "unpushed_changes"


def test_mirror_result_levels_cover_every_outcome_and_ok_is_green_or_amber():
    assert set(ST.OUTCOME_LEVELS.values()) == {"green", "amber", "red"}
    for outcome in ("verified", "pulled", "unchanged", "published", "deferred", "dry_run", "skipped", "invalidated"):
        assert ST.MirrorResult("s", outcome).ok and ST.MirrorResult("s", outcome).level == "green"
    assert ST.MirrorResult("s", "unpushed_changes").ok and ST.MirrorResult("s", "unpushed_changes").level == "amber"
    for outcome in ("frozen_differs", "no_current", "pull_verify_failed", "full_verify_failed", "verify_report_missing", "current_unrecorded",
                    "current_unverified", "current_unsealed", "lock_busy", "error", "an_outcome_nobody_named"):
        assert outcome in ST.OUTCOME_LEVELS or outcome == "an_outcome_nobody_named", outcome
        assert not ST.MirrorResult("s", outcome).ok and ST.MirrorResult("s", outcome).level == "red", outcome
    assert ST.MirrorResult("s", "verified").as_dict()["level"] == "green"


def test_last_run_level_classifies_the_check_and_verify_authority_stamps_by_their_command_and_pins_every_state_the_writers_spell():
    """`stores check` stamps `last_run.outcome` as `check:<state>` and the one-time `stores verify-authority` as
    `verify-authority:<state>`; neither is a mirror outcome, and reading them through OUTCOME_LEVELS alone called a MATCHING check and a
    VERIFIED move red. `last_run_level` classifies by the command that stamped the slot: the table names EVERY state the two writers can
    record (pinned here against their source, both ways, so a new state cannot be written unnamed), a successful one is green, every
    failure red, a state nobody named red, the mirror run's in-flight stamps amber, a plain mirror outcome through OUTCOME_LEVELS, and no
    outcome is None. The writers spell the prefixes the classifier reads."""
    import inspect
    import re as _re
    for fn, const in ((ST.check_component, "CHECK_OUTCOME_PREFIX"), (ST.verify_authority, "VERIFY_AUTHORITY_OUTCOME_PREFIX")):
        prefix = getattr(ST, const)
        src = inspect.getsource(fn)
        assert "{" + const + "}" in src, (fn.__name__, "the writer spells the classifier's prefix constant")
        spelled = (set(_re.findall(r'\bstate="([a-z_]+)"', src)) | set(_re.findall(r'\["state"\]\s*=\s*"([a-z_]+)"', src))
                   | set(_re.findall(r'"state":\s*"([a-z_]+)"', src)) | set(_re.findall(r'\["state"\]\s*=\s*"[a-z_]+"\s+if\s+\w+\s+else\s+"([a-z_]+)"', src))
                   | set(_re.findall(r'\breturn "([a-z_]+)"', src)))
        assert spelled == set(ST.AUXILIARY_OUTCOME_LEVELS[prefix]), (fn.__name__, sorted(spelled ^ set(ST.AUXILIARY_OUTCOME_LEVELS[prefix])))
        for state, level in ST.AUXILIARY_OUTCOME_LEVELS[prefix].items():
            assert ST.last_run_level(f"{prefix}{state}") == level, (prefix, state)
        assert ST.last_run_level(f"{prefix}an_outcome_nobody_named") == "red"
    assert ST.CHECK_OUTCOME_PREFIX == "check:" and ST.VERIFY_AUTHORITY_OUTCOME_PREFIX == "verify-authority:"
    assert ST.last_run_level("check:match") == "green" and ST.last_run_level("verify-authority:verified") == "green"
    assert ST.last_run_level("check:not_applicable") == "green" and ST.last_run_level("verify-authority:not_applicable") == "green"
    assert {ST.last_run_level(o) for o in ("check:mismatch", "check:error", "verify-authority:unequal", "verify-authority:no_current",
                                            "verify-authority:no_manifest", "verify-authority:verify_failed", "verify-authority:current_moved",
                                            "verify-authority:source_changed", "verify-authority:error")} == {"red"}
    assert ST.IN_FLIGHT_OUTCOME_LEVELS == {"verify_inline": "amber", "verify_submitted": "amber"}
    assert ST.last_run_level("verify_inline") == "amber" and ST.last_run_level("verify_submitted") == "amber"
    assert "verify_inline" not in ST.OUTCOME_LEVELS and "verify_submitted" not in ST.OUTCOME_LEVELS, "the in-flight stamps are not MirrorResult outcomes"
    for outcome, level in ST.OUTCOME_LEVELS.items():
        assert ST.last_run_level(outcome) == level == ST.MirrorResult("s", outcome).level, outcome
    assert ST.last_run_level("an_outcome_nobody_named") == "red" and ST.last_run_level(None) is None and ST.last_run_level("") is None


# --------------------------------------------------------------------------------------------
# the weekly unit, the installer's set and the window's restore set
# --------------------------------------------------------------------------------------------


def _parse(path):
    cp = configparser.ConfigParser(strict=False, allow_no_value=True, interpolation=None, comment_prefixes=("#",), inline_comment_prefixes=None)
    cp.optionxform = str
    cp.read_string(open(path).read())
    return cp


def test_store_mirror_full_unit_is_a_weekly_calendar_oneshot_that_waits_for_the_ticks_lock_and_the_window_trap_enables_it():
    svc = _parse(os.path.join(UNITS, "registry-store-mirror-full.service.in"))
    s = svc["Service"]
    assert s["Type"] == "oneshot" and s["ExecStart"].endswith("stores mirror --dest @STORE_DEST@ --full --lock-wait-s 3600") and "eval " not in s["ExecStart"]
    assert s["Restart"] == "on-failure" and s["RestartSec"] == "1h" and s["TimeoutStartSec"] == "9h" and s["Nice"] == "10" and s["IOSchedulingClass"] == "idle"
    # no Conflicts= with the 30-min tick: a conflict lets the tick's timer start STOP the weekly mid-verify, its Slurm job left
    # without its watcher; the two serialize on the shared mirror lock -- the weekly waits up to an hour for a tick (exit 3 after a vain
    # wait, retried by Restart=on-failure), a tick exits 0 at once when the weekly holds it (test_the_mirror_lock_waits_when_asked...)
    assert "Conflicts" not in svc["Unit"] and svc["Unit"]["StartLimitBurst"] == "2" and svc["Unit"]["StartLimitIntervalSec"] == "1d"
    text = open(os.path.join(UNITS, "registry-store-mirror-full.service.in")).read()
    assert "Environment=SSH_AUTH_SOCK=%h/.ssh/agent.sock" in text and "harvester.lock" not in text.split("[Service]")[1] and "--export" not in text
    tmr = _parse(os.path.join(UNITS, "registry-store-mirror-full.timer.in"))["Timer"]
    assert tmr["OnCalendar"] == "Sun *-*-* 03:00:00 UTC" and tmr["Persistent"] == "true" and tmr["Unit"] == "registry-store-mirror-full.service"
    assert "OnUnitActiveSec" not in tmr and tmr["RandomizedDelaySec"]
    # the installer enables it by default; the window's trap enables it when the install stage added it (DEFAULT_TIMERS); the window never
    # stops it (it reads the stores and the state only)
    installer = re.search(r"^TIMERS=\((.*)\)$", open(INSTALL).read(), re.M).group(1).split()
    assert "registry-store-mirror-full.timer" in installer and installer.index("registry-store-mirror-full.timer") > installer.index("registry-store-mirror.timer")
    window = open(WINDOW).read()
    assert "registry-store-mirror-full.timer" in re.search(r'^DEFAULT_TIMERS="(.*)"$', window, re.M).group(1).split()
    assert "registry-store-mirror-full.timer" not in re.search(r'^TIMERS="(.*)"$', window, re.M).group(1).split()


# --------------------------------------------------------------------------------------------
# a generation holding both spellings of one name restores in either enumeration order
# --------------------------------------------------------------------------------------------


def _chain_dir(tmp_path, name):
    """A pulled tree AS THE DESTINATION SPELLS IT: the true names `a:b`, `a%3Ab`, `a%253Ab` (a three-link chain: each is the escaped
    spelling of the one before) as files and `d:x` / `d%3Ax` as directories, each with its own bytes; `plain` beside them."""
    top = str(tmp_path / name)
    _write(os.path.join(top, "a%3Ab"), b"true a:b\n")
    _write(os.path.join(top, "a%253Ab"), b"true a%3Ab\n")
    _write(os.path.join(top, "a%25253Ab"), b"true a%253Ab\n")
    _write(os.path.join(top, "d%3Ax", "f.txt"), b"true d:x\n")
    _write(os.path.join(top, "d%253Ax", "g.txt"), b"true d%3Ax\n")
    _write(os.path.join(top, "plain"), b"plain\n")
    return top


def _read_tree(top):
    out = {}
    for dp, _dns, fns in os.walk(top):
        for n in fns:
            p = os.path.join(dp, n)
            if not os.path.islink(p):
                out[os.path.relpath(p, top)] = open(p, "rb").read()
    return out


EXPECTED_CHAIN = {"a:b": b"true a:b\n", "a%3Ab": b"true a%3Ab\n", "a%253Ab": b"true a%253Ab\n", "d:x/f.txt": b"true d:x\n",
                  "d%3Ax/g.txt": b"true d%3Ax\n", "plain": b"plain\n"}


@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_restore_escaped_names_keeps_both_spellings_of_one_name_in_either_enumeration_order(tmp_path, monkeypatch, order):
    """`a:b` and `a%3Ab` in one generation arrive as `a%3Ab` and `a%253Ab`. Ascending order lists `a%253Ab` first ('2' < '3'): a one-pass
    rename restored it to `a%3Ab` OVER the escaped `a%3Ab` beside it -- deleting `a:b`'s source -- and then renamed that to `a:b`, so
    `a:b` carried `a%3Ab`'s bytes and `a%3Ab` was gone; descending order never collided. The staged restore is right in both,
    for files and for directories, and leaves no staging name behind."""
    real_walk = os.walk

    def ordered_walk(top, topdown=True, onerror=None, followlinks=False):
        for dp, dns, fns in real_walk(top, topdown=topdown, onerror=onerror, followlinks=followlinks):
            yield dp, sorted(dns, reverse=(order == "descending")), sorted(fns, reverse=(order == "descending"))

    monkeypatch.setattr(ST.os, "walk", ordered_walk)
    top = _chain_dir(tmp_path, f"t-{order}")
    counts = ST.restore_escaped_names(top, {})
    assert counts == {"renamed": 5, "replaced": 0, "relinked": 0}
    assert _read_tree(top) == EXPECTED_CHAIN, _read_tree(top)
    assert not any(n.startswith(ST.RESTORE_STAGE_PREFIX) for n in os.listdir(top)), "no staging name left behind"
    # the previous pull's restored copies standing beside the fresh escaped ones (an escaped subtree arrives afresh every pull): replaced
    # by the fresh copy, never merged -- and only after every escaped sibling is staged away
    top2 = _chain_dir(tmp_path, f"t2-{order}")
    _write(os.path.join(top2, "a:b"), b"STALE a:b\n")
    _write(os.path.join(top2, "d:x", "stale.txt"), b"STALE\n")
    counts = ST.restore_escaped_names(top2, {})
    assert counts == {"renamed": 5, "replaced": 2, "relinked": 0}
    assert _read_tree(top2) == EXPECTED_CHAIN, _read_tree(top2)


def test_a_pull_of_a_generation_holding_both_spellings_verifies_and_a_type_change_at_an_escaped_spelling_re_pulls(tmp_path):
    """End to end through the cluster producer's push and the pull: a generation with `a:b` + `a%3Ab` (files) and `d:x` + `d%3Ax`
    (directories) pulls and hash-verifies; the next generation turns `d:x` into a FILE, whose escaped spelling `d%3Ax` is where the
    previous pull's restored true `d%3Ax` (a non-empty directory) stands -- rsync refuses to put a file there without --delete (`could
    not make way for new regular file`, rc 23, measured), so the pull clears every entry at an escaped spelling before the rsync."""
    w = _world(tmp_path)
    root, dest, comp = w["root"], w["dest"], w["cfg"].components["clu"]
    target = ST.grading_host_mirror_path(root, comp)
    gd = dest.gen_dir("clu")
    src = str(tmp_path / "chainsrc")
    _write(os.path.join(src, "a:b"), b"true a:b\n")
    _write(os.path.join(src, "a%3Ab"), b"true a%3Ab\n")
    _write(os.path.join(src, "d:x", "f.txt"), b"true d:x\n")
    _write(os.path.join(src, "d%3Ax", "g.txt"), b"true d%3Ax\n")
    oc = ST.push_on_cluster("clu", src, dest.stores_dir, session="job1")
    assert oc.outcome == "published", oc.as_dict()
    assert sorted(os.listdir(f"{gd}/{oc.content_sha8}")) == ["a%253Ab", "a%3Ab", "d%253Ax", "d%3Ax"], "both spellings, one chain"
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "pulled" and r.level == "green" and r.published == oc.content_sha8 and r.content_sha8 == oc.content_sha8, r.as_dict()
    assert r.detail["pull"]["cleared"] == 0 and r.detail["pull"]["renamed"] == 4 and r.detail["pull"]["replaced"] == 0
    assert _read_tree(target) == {"a:b": b"true a:b\n", "a%3Ab": b"true a%3Ab\n", "d:x/f.txt": b"true d:x\n", "d%3Ax/g.txt": b"true d%3Ax\n"}
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "unchanged"
    # the next generation: `d:x` becomes a file, `a:b` changes; `a%3Ab` and `d%3Ax` keep their bytes
    shutil.rmtree(os.path.join(src, "d:x"))
    _write(os.path.join(src, "d:x"), b"now a file\n")
    _write(os.path.join(src, "a:b"), b"true a:b, revised\n")
    oc2 = ST.push_on_cluster("clu", src, dest.stores_dir, session="job2")
    assert oc2.outcome == "published" and oc2.content_sha8 != oc.content_sha8, oc2.as_dict()
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "pulled" and r.published == oc2.content_sha8 and r.content_sha8 == oc2.content_sha8, r.as_dict()
    assert r.detail["pull"]["cleared"] == 2 and sorted(r.detail["pull"]["cleared_first"]) == ["a%3Ab", "d%3Ax"], "the restored chain neighbors at this generation's escaped spellings"
    assert r.detail["pull"]["renamed"] == 4 and r.detail["pull"]["replaced"] == 2, "the previous pull's `a:b` and `d:x` replaced; `a%3Ab` and `d%3Ax` had been cleared"
    assert _read_tree(target) == {"a:b": b"true a:b, revised\n", "a%3Ab": b"true a%3Ab\n", "d:x": b"now a file\n", "d%3Ax/g.txt": b"true d%3Ax\n"}
    rs = next(a for a in rn.calls if "rsync" in a)
    assert "--delete" not in rs and "--force" not in rs, "rsync still deletes nothing; the clearing is the mirror's own, by name, before it"
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "unchanged"


# --------------------------------------------------------------------------------------------
# the weekly and the tick serialize on the mirror lock, never by a systemd conflict
# --------------------------------------------------------------------------------------------


def test_the_mirror_lock_waits_when_asked_and_the_cli_exits_3_after_a_vain_wait(tmp_path, monkeypatch, capsys):
    """The overlapping start, at the lock: a tick that finds the weekly holding the lock yields at once with exit 0 (the default, no wait);
    the weekly (`--lock-wait-s`) waits for a tick that holds it and proceeds when it lets go; a wait that runs out exits 3, which the
    unit's Restart=on-failure retries, instead of skipping the week with a 0."""
    w = _world(tmp_path)
    root = w["root"]
    holder = ST.MirrorLock(root).__enter__()                       # the tick, mid-run
    assert holder.held
    with ST.MirrorLock(root) as l0:                                 # the default: no wait -- not held, at once
        assert not l0.held and l0.waited_s < 0.5 and l0.holder_line().startswith(str(os.getpid()))
    with ST.MirrorLock(root, wait_s=0.4, poll_s=0.05) as l1:        # a bounded wait that runs out
        assert not l1.held and 0.4 <= l1.waited_s < 3.0
    threading.Timer(0.3, lambda: holder.__exit__(None, None, None)).start()
    with ST.MirrorLock(root, wait_s=10.0, poll_s=0.05) as l2:       # the holder lets go during the wait: held, early
        assert l2.held and 0.2 <= l2.waited_s < 5.0 and l2.holder_line().startswith(str(os.getpid()))
    assert not ST.MirrorLock(root).__enter__().held or True         # (released on exit above; the next take below proves it)
    holder = ST.MirrorLock(root).__enter__()
    assert holder.held
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    t0 = time.monotonic()
    assert cli.main(["stores", "mirror", "--lock-wait-s", "0.3"]) == 3
    assert time.monotonic() - t0 >= 0.3
    err = capsys.readouterr().err
    assert "still holds" in err and "exit 3" in err and str(os.getpid()) in err, err
    assert cli.main(["stores", "mirror"]) == 0
    assert "exiting without acting" in capsys.readouterr().out
    holder.__exit__(None, None, None)
    assert not os.path.exists(ST.state_path(root)), "neither run acted"


# --------------------------------------------------------------------------------------------
# a `current` is green only with its evidence; adopted only explicitly
# --------------------------------------------------------------------------------------------


def test_a_current_without_its_evidence_is_red_by_name_and_adopted_only_explicitly(tmp_path):
    w = _world(tmp_path)
    root, dest, cfg = w["root"], w["dest"], w["cfg"]
    comp = cfg.components["dev"]
    gd = dest.gen_dir("dev")
    pushed = ST.push_component(root, dest, comp, runner=Runner(), session="sess")
    assert pushed.outcome == "published"
    sha8 = pushed.content_sha8
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"
    rec_path, seal_path, ver_path = f"{gd}/PUBLISHED-{sha8}.json", f"{gd}/SEALED-{sha8}.json", f"{gd}/VERIFIED-{sha8}.json"
    # the record gone (a generation that predates the rule): red by name, `current` untouched, nothing written; the dry run says so;
    # check-authority agrees -- the two tools never disagree about the same `current`
    os.remove(rec_path)
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "current_unrecorded" and r.level == "red" and not r.ok and r.published == sha8 and r.content_sha8 == sha8, r.as_dict()
    assert r.detail["authority_problems"] == ["current_unrecorded"] and r.detail["adoptable"] is True and "--adopt-unrecorded" in r.detail["why"]
    gr = r.detail["generation_record"]
    assert gr["published_record"] is False and gr["sealed"] is True and gr["verified_matches"] is True and gr["problems"] == ["current_unrecorded"]
    assert _no_copy(rn) and not os.path.exists(rec_path) and os.readlink(f"{gd}/current") == sha8
    st = ST.load_state(root)["components"]["dev"]["generation"]
    assert st["state"] == "current_unrecorded" and st["authority_problems"] == ["current_unrecorded"]
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "current_unrecorded" and r.detail["adoptable"] is True and r.detail["writes"] == "nothing"
    assert not os.path.exists(rec_path)
    assert ST.check_authority(root, cfg, dest)["components"]["dev"]["problems"] == ["no_published_record"]
    # --adopt-unrecorded: the record (mode adopt, the producer's session) written under the publish lock, `current` untouched -> verified
    r = ST.run_mirror(root, dest, comp, runner=Runner(), adopt_unrecorded=True)
    assert r.outcome == "verified" and r.level == "green" and r.ok, r.as_dict()
    adopted = r.detail["adopted"]
    assert adopted["published_record"]["mode"] == "adopt" and adopted["published_record"]["producer_session"] == "sess" and "sealed" not in adopted
    assert adopted["publish_lock"]["path"] == f"{gd}/{ST.PUBLISH_LOCK_FILE}" and adopted["publish_lock"]["scope"] == "step"
    rec = json.load(open(rec_path))
    assert rec["mode"] == "adopt" and rec["content_sha8"] == sha8 and rec["previous"] is None and rec["producer"] == "grading_host:sess" and rec["change_utc"]
    assert r.detail["published_record"]["mode"] == "adopt" and r.detail["generation_record"]["published_record"] is True and os.readlink(f"{gd}/current") == sha8
    assert ST.load_state(root)["components"]["dev"]["generation"]["adopted"]["published_record"]["mode"] == "adopt"
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "verified" and "adopted" not in r.detail
    assert ST.check_authority(root, cfg, dest)["components"]["dev"]["problems"] == []
    # the seal record gone and the generation directory writable: current_unsealed (red); the adoption re-seals and records nothing new
    os.remove(seal_path)
    os.chmod(f"{gd}/{sha8}", 0o755)
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "current_unsealed" and r.level == "red" and r.detail["authority_problems"] == ["current_unsealed"] and r.detail["adoptable"] is True, r.as_dict()
    assert _mode(f"{gd}/{sha8}") == 0o755 and not os.path.exists(seal_path), "nothing adopted implicitly"
    r = ST.run_mirror(root, dest, comp, runner=Runner(), adopt_unrecorded=True)
    assert r.outcome == "verified" and r.detail["adopted"]["sealed"]["n_dirs"] >= 1 and "published_record" not in r.detail["adopted"], r.as_dict()
    assert _mode(f"{gd}/{sha8}") == 0o555 and json.load(open(seal_path))["content_sha8"] == sha8
    # the VERIFIED report gone: current_unverified -- never adopted, flag or no flag; nothing written
    os.rename(ver_path, ver_path + ".aside")
    for adopt in (False, True):
        r = ST.run_mirror(root, dest, comp, runner=Runner(), adopt_unrecorded=adopt)
        assert r.outcome == "current_unverified" and r.level == "red" and r.detail["authority_problems"] == ["current_unverified"], r.as_dict()
        assert r.detail["adoptable"] is False and "verify-authority" in r.detail["why"] and "adopted" not in r.detail
    assert os.path.isfile(rec_path) and os.path.isfile(seal_path)
    # a report that does not admit THIS manifest (another generation's hashes) counts as none
    other = json.load(open(ver_path + ".aside"))
    other["content_sha256"] = "0" * 64
    json.dump(other, open(ver_path, "w"))
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == "current_unverified" and r.detail["generation_record"]["verified"] is True and r.detail["generation_record"]["verified_matches"] is False
    os.replace(ver_path + ".aside", ver_path)
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"
    # a frozen producer the same: a pre-rule `current` is red, never `verified`; adopted explicitly (no producer session to record)
    fz = cfg.components["frozen"]
    assert ST.push_component(root, dest, fz, runner=Runner()).outcome == "published"
    fz_sha = os.readlink(f"{dest.gen_dir('frozen')}/current")
    os.remove(f"{dest.gen_dir('frozen')}/PUBLISHED-{fz_sha}.json")
    r = ST.run_mirror(root, dest, fz, runner=Runner())
    assert r.outcome == "current_unrecorded" and r.level == "red", r.as_dict()
    r = ST.run_mirror(root, dest, fz, runner=Runner(), adopt_unrecorded=True)
    assert r.outcome == "verified" and r.detail["adopted"]["published_record"]["mode"] == "adopt" and r.detail["adopted"]["published_record"]["producer_session"] is None
    # a cluster producer's `current` without its record: refused BEFORE the pull (nothing pulled, the mirror directory not even made) and
    # never adopted by the mirror -- its job records and seals before it switches, so a bare `current` there is mid-flight or a crash
    clu = cfg.components["clu"]
    oc = ST.push_on_cluster("clu", _cluster_tree(tmp_path), dest.stores_dir, session="job1")
    assert oc.outcome == "published"
    os.remove(f"{dest.gen_dir('clu')}/PUBLISHED-{oc.content_sha8}.json")
    target = ST.grading_host_mirror_path(root, clu)
    for adopt in (False, True):
        rn = Runner()
        r = ST.run_mirror(root, dest, clu, runner=rn, adopt_unrecorded=adopt)
        assert r.outcome == "current_unrecorded" and r.level == "red" and r.published == oc.content_sha8 and r.detail["authority_problems"] == ["current_unrecorded"], r.as_dict()
        assert _no_copy(rn) and not os.path.exists(target) and "nothing pulled" in r.detail["why"] and "adopted" not in r.detail
    r = ST.run_mirror(root, dest, clu, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "current_unrecorded" and not os.path.exists(target)
    assert ST.load_state(root)["components"]["clu"]["generation"]["state"] == "current_unrecorded"


# --------------------------------------------------------------------------------------------
# the closing pass on 2c: a dangling `current` is never re-advertised, and a deferred repair is not a successful deferral
# --------------------------------------------------------------------------------------------


def _dangle_world(tmp_path):
    """A `cluster_a` component per producer shape the reversed mirror handles itself, each in an ADAPTER OF ITS OWN so a window's plan
    names exactly the component whose pointer dangles."""
    base = _canon(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    comps = {"dev": {"base": base, "authority": "cluster_a", "producer": "grading_host:sess", "exclude": EXCL},
             "frozen": {"base": base, "authority": "cluster_a", "producer": "frozen", "exclude": EXCL},
             "carried": {"base": _canon(tmp_path / "c"), "authority": "cluster_a", "producer": "grading_host:carrier", "transitional_carry": True,
                         "carry_deadline": "2099-01-01", "exclude": EXCL},
             "paced": {"base": _canon(tmp_path / "p"), "authority": "cluster_a", "producer": "grading_host:pacer", "transitional_carry": True,
                       "carry_deadline": "2099-01-01", "min_publish_interval_s": 3600, "exclude": EXCL}}
    cfgp = _cfg(tmp_path, comps, adapters={f"paired_{s}": {"shards": 1, "components": [s]} for s in comps})
    cfg = ST.load_config(cfgp)
    return {"base": base, "root": root, "cfg": cfg, "cfgp": cfgp, "dest": cfg.destination("cluster_a")}


def _plan_entry(w, slug):
    """What a window's plan pins for the component: it reads `published` out of `state/store_mirror.json`, exactly as the shards do."""
    from sisteer.registry import cluster_reads as CR
    plan = CR.plan_window(w["root"], "20261007T1800Z", [f"paired_{slug}"], w["cfg"], w["dest"], tree_sha="0" * 40)
    return plan["adapters"][f"paired_{slug}"]


@pytest.mark.parametrize("slug", ["dev", "frozen"])
def test_the_reversed_mirror_never_re_advertises_a_dangling_pointer(tmp_path, slug):
    """The closing pass on 2c (MEDIUM): a `stores attest` that finds the published tree gone drops `published` and records the dangle --
    and the very next reversed-mirror tick named `dangling` (red) while RESTORING `published = current` in the same breath, because every
    exit of `finish` wrote the pointer as what the shards read. So a window's plan went on pinning a generation whose tree was absent and
    scheduled reads of it. Now nothing is published while the pointer dangles, whichever reader looks first: the attest's record stands
    untouched, a tick that establishes the dangle itself records it (with the trees a rename left beside the pointer), the plan pins
    nothing, and the tree standing again is what ends the dangle and re-advertises the generation."""
    w = _dangle_world(tmp_path)
    root, cfg, dest = w["root"], w["cfg"], w["dest"]
    comp = cfg.components[slug]
    gd = dest.gen_dir(slug)
    pushed = ST.push_component(root, dest, comp, runner=Runner(), **({"session": "sess"} if slug == "dev" else {}))
    assert pushed.outcome == "published", pushed.as_dict()
    sha8 = pushed.content_sha8
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"

    def row():
        return ST.status(root, cfg)["components"][slug]

    # the healthy generation IS what a plan pins
    assert row()["published"] == sha8 and _plan_entry(w, slug)["pins"] == {slug: sha8}
    # (1) the attest finds the tree gone: `published` dropped, the dangle recorded
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261007T120000000Z")
    a = ST.attest_listings(root, dest, comp, runner=Runner())
    assert a.outcome == "current_dangling" and a.level == "red", a.as_dict()
    recorded = ST.load_state(root)["components"][slug]["current_dangling"]
    assert ST.load_state(root)["components"][slug]["published"] is None and recorded["content_sha8"] == sha8
    # the next reversed-mirror tick: `dangling` (red) and NOTHING published -- before this it wrote `published = current` back
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "dangling" and r.level == "red" and not r.ok and r.published is None, r.as_dict()
    assert "nothing is published" in r.detail["why"] and _no_copy(rn)
    st = ST.load_state(root)["components"][slug]
    assert st["published"] is None and st["current_dangling"] == recorded, "the attest's record stands; the pointer is not re-advertised"
    assert row()["published"] is None and row()["verified"] is False and row()["level"] == "red"
    assert ST.published_attestations(root).get(slug) is None
    assert f"PROBLEM {slug}: current -> {sha8}" in ST.format_status(ST.status(root, cfg))
    entry = _plan_entry(w, slug)
    assert entry["pins"] == {} and entry["route"] == "grading_host" and entry["reason"] == f"no published generation for component(s): {slug}"
    assert entry["task_indices"] == [] and entry["attestations"] == {}
    # a dry run of the same tick reads only and says what it would conclude; nothing published either
    rd = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert rd.outcome == "dry_run" and rd.detail["would"] == "dangling" and rd.published is None, rd.as_dict()
    # (2) the tree standing again ends the dangle: verified, the generation re-advertised, the plan pins it
    os.rename(f"{gd}/.failed-{sha8}-20261007T120000000Z", f"{gd}/{sha8}")
    rv = ST.run_mirror(root, dest, comp, runner=Runner())
    assert rv.outcome == "verified" and rv.published == sha8, rv.as_dict()
    st = ST.load_state(root)["components"][slug]
    assert st["published"] == sha8 and st["current_dangling"] is None and "PROBLEM" not in ST.format_status(ST.status(root, cfg))
    assert _plan_entry(w, slug)["pins"] == {slug: sha8}
    # (3) a dangle the MIRROR establishes first (no attest): recorded the moment it is read, with the renamed tree named, nothing published
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261007T130000000Z")
    rn = Runner()
    r2 = ST.run_mirror(root, dest, comp, runner=rn)
    assert r2.outcome == "dangling" and r2.published is None and _no_copy(rn), r2.as_dict()
    d = ST.load_state(root)["components"][slug]["current_dangling"]
    assert d["content_sha8"] == sha8 and d["renamed_aside"] == [f".failed-{sha8}-20261007T130000000Z"] and d["resumed"] is False
    assert d["found_utc"] and f"{gd}/{sha8} is not a directory" in d["why"] and ".failed-" in d["why"]
    assert ST.load_state(root)["components"][slug]["published"] is None and _plan_entry(w, slug)["pins"] == {}
    assert f"PROBLEM {slug}:" in ST.format_status(ST.status(root, cfg))
    # the next tick keeps that first finding rather than re-dating it
    r3 = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r3.outcome == "dangling" and ST.load_state(root)["components"][slug]["current_dangling"] == d


class _ListingBlip(Runner):
    """A runner on which the ONE listing step that names the trees a rename left beside a dangling pointer
    (`Remote.renamed_aside`) fails like an ssh outage (rc 255); everything else runs as usual."""

    def __call__(self, argv, timeout_s, input=None, check=False):
        argv = list(argv)
        if argv[:2] == ["sh", "-c"] and ".failed-" in argv[2] and ".invalid-" in argv[2]:
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: connect to host port 22: Connection timed out")
        return super().__call__(argv, timeout_s, input=input, check=check)


def test_a_failing_dangle_listing_still_publishes_nothing(tmp_path):
    """The final pass on 2c (MEDIUM): `finish` is the only exit of this flow that drops `published`, and the dangle's listing step --
    one shell call that names what a rename left beside the pointer -- runs BEFORE it. An ssh outage there raises, and the error path
    flushes the dangle's record and `last_run = error` without ever passing through `finish`, so the previous sha stayed pinnable over a
    tree this tick had already read as absent: a window pinned it and scheduled reads of a tree that was not there. `published` is now
    dropped the moment the pointer and the missing tree are read, before the extra call, so a transport failure there costs the listing
    and nothing else."""
    w = _dangle_world(tmp_path)
    root, cfg, dest, slug = w["root"], w["cfg"], w["dest"], "dev"
    comp = cfg.components[slug]
    gd = dest.gen_dir(slug)
    pushed = ST.push_component(root, dest, comp, runner=Runner(), session="sess")
    assert pushed.outcome == "published", pushed.as_dict()
    sha8 = pushed.content_sha8
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"
    assert _plan_entry(w, slug)["pins"] == {slug: sha8}
    # the tree renamed away; this tick establishes the dangle and the listing of what stands beside it dies on the transport
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261007T160000000Z")
    rn = _ListingBlip()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "error" and not r.ok and r.published is None and "rc 255" in r.detail["error"], r.as_dict()
    st = ST.load_state(root)["components"][slug]
    assert st["published"] is None and st["last_run"]["outcome"] == "error", st
    # the record the dying tick left is the honest minimal one: the content, the finding, the listing not yet taken
    d = st["current_dangling"]
    assert d["content_sha8"] == sha8 and d["renamed_aside"] is None and "not yet listed" in d["why"] and d["resumed"] is False
    entry = _plan_entry(w, slug)
    assert entry["pins"] == {} and entry["route"] == "grading_host" and entry["reason"] == f"no published generation for component(s): {slug}"
    assert ST.status(root, cfg)["components"][slug]["published"] is None and ST.published_attestations(root).get(slug) is None
    # the next tick with the transport back keeps that first finding (nothing newer was learned) and still publishes nothing
    r2 = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r2.outcome == "dangling" and r2.level == "red" and r2.published is None, r2.as_dict()
    assert ST.load_state(root)["components"][slug]["current_dangling"] == d and _plan_entry(w, slug)["pins"] == {}
    # the tree standing again ends the dangle and re-advertises the generation
    os.rename(f"{gd}/.failed-{sha8}-20261007T160000000Z", f"{gd}/{sha8}")
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"
    assert _plan_entry(w, slug)["pins"] == {slug: sha8}


def test_a_carry_deferred_over_a_dangling_pointer_is_named_never_a_successful_deferral(tmp_path):
    """The closing pass on 2c (MEDIUM, the carry's half): a transitional carry is the repair of a dangling `current` (C3), and
    `min_publish_interval_s` defers it like a push -- so a tick over a dangling pointer inside the interval reported `deferred`, GREEN and
    exit 0, while no tree stood and the pointer was re-advertised for a plan to pin. A deferred REPAIR is not a successful deferral: the
    outcome is `dangling_carry_deferred` (red), nothing is published, the deferral's own facts ride in the detail, and the dangle's record
    stands until the interval elapses and the carry republishes the generation."""
    w = _dangle_world(tmp_path)
    root, cfg, dest = w["root"], w["cfg"], w["dest"]
    comp = cfg.components["paced"]
    gd = dest.gen_dir("paced")
    r1 = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r1.outcome == "published" and json.load(open(f"{gd}/PUBLISHED-{r1.content_sha8}.json"))["mode"] == "carry", r1.as_dict()
    sha8 = r1.content_sha8
    assert _plan_entry(w, "paced")["pins"] == {"paced": sha8}
    # the tree renamed away and the attest's record standing: the tick inside the publish interval
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261007T120000000Z")
    assert ST.attest_listings(root, dest, comp, runner=Runner()).outcome == "current_dangling"
    recorded = ST.load_state(root)["components"]["paced"]["current_dangling"]
    rn = Runner()
    r2 = ST.run_mirror(root, dest, comp, runner=rn)
    assert r2.outcome == "dangling_carry_deferred" and r2.level == "red" and not r2.ok and r2.published is None, r2.as_dict()
    assert r2.detail["min_publish_interval_s"] == 3600 and r2.detail["published_utc"] and _no_copy(rn)
    assert "names no tree" in r2.detail["why"] and "deferred" in r2.detail["why"] and "min_publish_interval_s" in r2.detail["why"]
    st = ST.load_state(root)["components"]["paced"]
    assert st["published"] is None and st["current_dangling"] == recorded and st["generation"]["state"] == "dangling_carry_deferred"
    assert st["last_run"]["outcome"] == "dangling_carry_deferred" and ST.last_run_level(st["last_run"]["outcome"]) == "red"
    row = ST.status(root, cfg)["components"]["paced"]
    assert row["published"] is None and row["verified"] is False and row["level"] == "red"
    assert _plan_entry(w, "paced")["pins"] == {} and f"PROBLEM paced: current -> {sha8}" in ST.format_status(ST.status(root, cfg))
    # the interval elapsed: the carry runs and republishes the dangling pointer -- the record cleared, the generation pinned again
    rec_path = f"{gd}/PUBLISHED-{sha8}.json"
    rec = json.load(open(rec_path))
    rec["published_utc"] = "2020-01-01T00:00:00.000000Z"
    json.dump(rec, open(rec_path, "w"))
    rn = Runner()
    r3 = ST.run_mirror(root, dest, comp, runner=rn)
    assert r3.outcome == "republished_dangling_current" and r3.ok and r3.published == sha8, r3.as_dict()
    assert r3.detail["carry"]["producer"] == "grading_host:pacer" and any(a[0] == "sbatch" for a in rn.calls), "the repair is a real publication"
    st = ST.load_state(root)["components"]["paced"]
    assert st["published"] == sha8 and st["current_dangling"] is None and _plan_entry(w, "paced")["pins"] == {"paced": sha8}
    assert "PROBLEM" not in ST.format_status(ST.status(root, cfg))


def test_a_tick_over_a_pending_repair_never_calls_the_dangle_over_before_the_publication_stands(tmp_path):
    """The push flow's repair of a dangling pointer copies the replacement under the SAME name, so the moment that copy lands the
    destination has a DIRECTORY where the pointer points -- and the reversed mirror read exactly that, called the dangle over, restored
    `published` and CLEARED the dangle record, which is the one thing telling the push flow its own repair was never published. With the
    seal cut short that left the plan pinning a writable, half-published tree; with only the switch cut short (the seal written, the
    previous tree's PUBLISHED record still beside it) it read `verified`, green, and no later tick could ever finish the publication. Now
    a tree standing under the pointer does not by itself end the dangle: while the record names `current` and either the pointer's
    evidence or the record's `phase` says the publication never ended, the tick names `dangling_repair_pending` (red, nothing published,
    the record intact) -- or, for a transitional carry, runs the carry, which finishes the seal and the guarded switch. A tree restored
    WHOLE still ends the dangle (the rule exercised above)."""
    w = _dangle_world(tmp_path)
    root, cfg, dest = w["root"], w["cfg"], w["dest"]

    # (1) a transitional carry: the tick that follows a cut-short repair FINISHES it instead of re-advertising the replacement
    comp = cfg.components["carried"]
    gd = dest.gen_dir("carried")
    r1 = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r1.outcome == "published", r1.as_dict()
    sha8 = r1.content_sha8
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261007T120000000Z")
    rn = _SealBlip()
    r2 = ST.run_mirror(root, dest, comp, runner=rn)
    assert r2.outcome == "error" and rn.blips == 1 and "seal" in r2.detail["error"], r2.as_dict()
    st = ST.load_state(root)["components"]["carried"]
    assert st["published"] is None and st["current_dangling"]["phase"] == "publishing", st
    assert os.path.isdir(f"{gd}/{sha8}") and not os.path.exists(f"{gd}/SEALED-{sha8}.json"), "the replacement stands, unsealed"
    rn = Runner()
    r3 = ST.run_mirror(root, dest, comp, runner=rn)
    assert r3.outcome == "republished_dangling_current" and r3.published == sha8 and r3.detail["pending_publication"] is True, r3.as_dict()
    assert not any("rsync" in a or a[0] == "sbatch" for a in rn.calls), "the fresh report stands: nothing copied, nothing re-verified"
    st = ST.load_state(root)["components"]["carried"]
    assert st["published"] == sha8 and st["current_dangling"] is None and all(m == 0o555 for m in _dir_modes(f"{gd}/{sha8}").values())
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified" and _plan_entry(w, "carried")["pins"] == {"carried": sha8}

    # (2) a producer that publishes itself: the tick NAMES the pending repair and finishes nothing; `stores push` completes it
    comp = cfg.components["dev"]
    gd = dest.gen_dir("dev")
    assert ST.push_component(root, dest, comp, runner=Runner(), session="sess").outcome == "published"
    sha8 = os.readlink(f"{gd}/current")
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261007T120000000Z")
    rn = _SealBlip()
    assert ST.push_component(root, dest, comp, runner=rn, session="sess").outcome == "error" and rn.blips == 1
    recorded = ST.load_state(root)["components"]["dev"]["current_dangling"]
    assert recorded["phase"] == "publishing" and os.path.isdir(f"{gd}/{sha8}")
    rn = Runner()
    r4 = ST.run_mirror(root, dest, comp, runner=rn)
    assert r4.outcome == "dangling_repair_pending" and r4.level == "red" and not r4.ok and r4.published is None, r4.as_dict()
    assert r4.detail["authority_problems"] == ["current_unsealed"] and r4.detail["repair"]["found_utc"] == recorded["found_utc"]
    assert "a present directory is not a repaired `current`" in r4.detail["why"] and "stores push dev" in r4.detail["why"] and _no_copy(rn)
    st = ST.load_state(root)["components"]["dev"]
    assert st["published"] is None and st["current_dangling"] == recorded and st["generation"]["state"] == "dangling_repair_pending"
    assert _plan_entry(w, "dev")["pins"] == {} and f"PROBLEM dev: current -> {sha8}" in ST.format_status(ST.status(root, cfg))
    # the seal written by hand but the switch still missing: the EVIDENCE is complete now and the pointer still does not read verified --
    # the record's `phase` is what tells a same-content replacement from a tree restored whole (the previous tree's record stands beside both)
    old_rec = json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))
    ST.seal_generation(ST.Remote(dest, Runner()), gd, sha8)
    r5 = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r5.outcome == "dangling_repair_pending" and r5.published is None and r5.detail["authority_problems"] == [], r5.as_dict()
    assert "its seal and its record stand; its publication did not" in r5.detail["why"]
    assert json.load(open(f"{gd}/PUBLISHED-{sha8}.json")) == old_rec and ST.load_state(root)["components"]["dev"]["published"] is None
    # the producer's push finishes the publication: the record rewritten, the dangle over, the generation pinned again
    rp = ST.push_component(root, dest, comp, runner=Runner(), session="sess")
    assert rp.outcome == "republished_dangling_current" and rp.published == sha8, rp.as_dict()
    assert json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))["published_utc"] > old_rec["published_utc"]
    st = ST.load_state(root)["components"]["dev"]
    assert st["published"] == sha8 and st["current_dangling"] is None
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified" and _plan_entry(w, "dev")["pins"] == {"dev": sha8}
    assert "PROBLEM" not in ST.format_status(ST.status(root, cfg))
