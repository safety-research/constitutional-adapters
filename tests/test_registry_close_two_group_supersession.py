"""A steered multi-turn store is written by TWO jobs (the g01234 and g5678 shard groups), each
registering its own run per (eval, object); the closer closes the store whole, so both runs used to file `generated` and the twin was
superseded by hand (nine records, then six). The closer files the other group's `superseded` itself:

* the SURVIVOR is the g01234 group's run — the registration whose shard group holds shard 00 (shards 00-04) — WHICHEVER group closes first
  (a survivor by shard group is predictable by construction, as in s2r / s3r; the closer race "first generated" gave a mixed set):
  closing g01234 first files its runs `generated` and g5678's `superseded`; closing g5678 first files g5678's OWN runs `superseded` and no
  `generated`; a g5678 `generated` an older closer left stands corrected by the g01234 close
* the record's shape is state superseded, evidence completion_artifact = the store's completion artifact, the survivor in the note and in
  extra.superseded_by_run_id, reason duplicate_registration, the sibling job id
* the sibling's later close skips its superseded runs by name and writes nothing; a lone registration (no sibling pending file on this cluster)
  files `generated` only; `--dry-run` says what it would file and writes nothing.

Reuses the real closer fixtures of tests/test_registry_close_stores.py (the real registration over the real seeds; synthetic job ids)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml  # noqa: E402
from sisteer.registry import emit, hooks  # noqa: E402
from test_registry_close_ruled_out_shards import _ids  # noqa: E402
from test_registry_close_stores import REPO, TAG, _close, _multiturn_store, _records, _register, _runs, _seeded_registry  # noqa: E402

G1, G2 = "810071", "810072"


def _world(tmp_path, monkeypatch, *, second: bool = True):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"
    wave = tmp_path / "exp" / "reproduce" / TAG / "waves" / "mt-s2r-cluster_a"; wave.mkdir(parents=True)
    p1 = _register(tmp_path, root, job=G1, job_name=f"f22-att-{TAG}-g01234-crescendo", evals="crescendo")
    p2 = _register(tmp_path, root, job=G2, job_name=f"f22-att-{TAG}-g5678-crescendo", evals="crescendo") if second else None
    prun = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_text())["private_run"]
    real_ids, IDS = _ids(tmp_path, p1, spool, wave)
    for (_e, obj) in _runs(p1):                                                  # the WHOLE store: all nine shards, written by the two groups
        _multiturn_store(wave, prun, "crescendo", obj, nshards=9, goals=259, goal_ids=real_ids)
    return spool, wave, p1, p2, IDS


def _state(spool, rid, state):
    return [p for t, p, *_ in _records(spool, rid) if t == "run_state" and p.get("state") == state]


def test_the_first_group_to_complete_the_store_files_generated_and_the_siblings_registration_superseded(tmp_path, monkeypatch):
    spool, wave, p1, p2, IDS = _world(tmp_path, monkeypatch)
    r1, r2 = _runs(p1), _runs(p2); assert set(r1) == set(r2) and len(r1) >= 2
    r = _close(p1, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"{len(r1)} filed" in out and f"{len(r1)} superseded (a twin registration of one store; D1816)" in out, out
    for key in r1:
        a, b = r1[key]["run_id"], r2[key]["run_id"]; assert a != b
        assert len(_state(spool, a, "generated")) == 1 and not _state(spool, a, "superseded")
        st = _state(spool, b, "superseded"); assert len(st) == 1 and not _state(spool, b, "generated"), st
        s = st[0]
        assert s["evidence_kind"] == "completion_artifact" and s["evidence_uri"].endswith("/progress.jsonl") and s["event_utc"]
        assert s["extra"]["superseded_by_run_id"] == a and s["extra"]["reason"] == "duplicate_registration" and s["extra"]["decision"] == "D1816"
        assert s["extra"]["sibling_job_id"] == G1 and f"carried by {a}" in s["note"] and "g01234 group's run survives" in s["note"] and "shard 00" in s["note"]
        assert f"superseded crescendo {key[1]} {b[:40]}… by {a[:40]}…" in out
    # the sibling's own close afterwards: every run of its already superseded — skipped by name, nothing written
    before = {key: len(_records(spool, r2[key]["run_id"])) for key in r2}
    r = _close(p2, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already superseded (the other shard group's registration of one store; D1816)") == len(r2) and f"{len(r2)} skipped" in out, out
    assert {key: len(_records(spool, r2[key]["run_id"])) for key in r2} == before and "superseded (a twin" not in out
    # and the first group's second close: already generated, nothing more superseded
    r = _close(p1, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already generated") == len(r1) and "superseded (a twin" not in out, out


def test_the_g01234_group_survives_whichever_group_closes_first(tmp_path, monkeypatch):
    """g5678 completes the store first — its close files its OWN runs `superseded` (by the g01234 registrations, not yet generated) and no
    `generated`; the g01234 close afterwards files its runs `generated` and finds its twins already retired."""
    spool, wave, p1, p2, IDS = _world(tmp_path, monkeypatch)
    r1, r2 = _runs(p1), _runs(p2)
    r = _close(p2, spool, wave, *IDS); out = r.stdout + r.stderr          # g5678 completes the store first
    assert r.returncode == 0 and f"{len(r2)} superseded" in out and "0 filed" in out, out
    assert out.count("store facts crescendo") == len(r2), out               # round-3 Major on e62b57124: the inspected store reaches the retired run's row
    for key in r1:
        st = _state(spool, r2[key]["run_id"], "superseded"); assert len(st) == 1 and not _state(spool, r2[key]["run_id"], "generated"), st
        assert st[0]["extra"]["superseded_by_run_id"] == r1[key]["run_id"] and st[0]["extra"]["sibling_job_id"] == G1 and st[0]["extra"]["decision"] == "D1816"
        _assert_store_facts_patch(spool, r2[key]["run_id"], store_tail=f"/crescendo/{key[1]}", n_items=int(r2[key]["plan"]["n_items_planned"]))
        assert not _state(spool, r1[key]["run_id"], "superseded") and not _state(spool, r1[key]["run_id"], "generated")   # the survivor: untouched until ITS close
    r = _close(p1, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"{len(r1)} filed" in out and "superseded (a twin" not in out, out
    for key in r1:
        assert len(_state(spool, r1[key]["run_id"], "generated")) == 1 and not _state(spool, r1[key]["run_id"], "superseded")
        assert len(_state(spool, r2[key]["run_id"], "superseded")) == 1 and not _state(spool, r2[key]["run_id"], "generated")


def test_the_second_group_arriving_after_a_generated_sibling_files_its_own_run_superseded(tmp_path, monkeypatch):
    """The sibling's `generated` was filed before this closer knew the pattern (a close from an earlier tree, or a reclose): the second group's
    close finds the survivor in the spool and supersedes ITS OWN run instead of filing a second `generated`."""
    spool, wave, p1, p2, IDS = _world(tmp_path, monkeypatch)
    r1, r2 = _runs(p1), _runs(p2)
    for key in r1:                                                          # the survivor's records as an older closer left them: generated, no twin filing
        w = emit.RecordWriter(str(spool), "sdfbase_full22", r1[key]["run_id"], emit.producer_block("older-closer", code_sha="c" * 40))
        w.emit_state("generated", "completion_artifact", "x/progress.jsonl")
    r = _close(p2, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"{len(r2)} superseded" in out and "0 filed" in out, out
    for key in r2:
        st = _state(spool, r2[key]["run_id"], "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == r1[key]["run_id"]
        assert not _state(spool, r2[key]["run_id"], "generated")
        _assert_store_facts_patch(spool, r2[key]["run_id"], store_tail=f"/crescendo/{key[1]}", n_items=int(r2[key]["plan"]["n_items_planned"]))


def test_a_g5678_generated_left_by_an_older_closer_is_retired_when_the_g01234_group_closes(tmp_path, monkeypatch):
    """The correction case (the s1 crescendo / pair stores, closed by both groups on trees without the rule): the g5678 run already carries
    `generated`; the g01234 close files its own runs `generated` AND the g5678 runs `superseded` — a generated twin is not a survivor."""
    spool, wave, p1, p2, IDS = _world(tmp_path, monkeypatch)
    r1, r2 = _runs(p1), _runs(p2)
    for key in r2:                                                          # the g5678 runs as an older closer left them: generated, no twin filing
        w = emit.RecordWriter(str(spool), "sdfbase_full22", r2[key]["run_id"], emit.producer_block("older-closer", code_sha="c" * 40))
        w.emit_state("generated", "completion_artifact", "x/progress.jsonl")
    r = _close(p1, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"{len(r1)} filed" in out and f"{len(r2)} superseded (a twin registration of one store; D1816)" in out, out
    for key in r1:
        assert len(_state(spool, r1[key]["run_id"], "generated")) == 1 and not _state(spool, r1[key]["run_id"], "superseded")
        st = _state(spool, r2[key]["run_id"], "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == r1[key]["run_id"] and st[0]["extra"]["sibling_job_id"] == G1
        assert len(_state(spool, r2[key]["run_id"], "generated")) == 1                                   # the older record stays; the superseded state is the correction
    # the g5678 close afterwards: its runs are superseded — skipped by name, nothing more written
    before = {key: len(_records(spool, r2[key]["run_id"])) for key in r2}
    r = _close(p2, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already superseded") == len(r2), out
    assert {key: len(_records(spool, r2[key]["run_id"])) for key in r2} == before


def test_a_lone_registration_files_generated_only_and_dry_run_writes_nothing(tmp_path, monkeypatch):
    spool, wave, p1, _p2, IDS = _world(tmp_path, monkeypatch, second=False)
    r1 = _runs(p1)
    r = _close(p1, spool, wave, *IDS, "--dry-run"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "would file run_patch" in out and "superseded" not in out and all(not _records(spool, r1[k]["run_id"]) or not _state(spool, r1[k]["run_id"], "generated") for k in r1), out
    r = _close(p1, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"{len(r1)} filed" in out and "superseded" not in out, out
    # dry run over a two-group world: says what it would supersede, writes nothing
    spool2, wave2, q1, q2, IDS2 = _world(tmp_path / "two", monkeypatch)
    r = _close(q1, spool2, wave2, *IDS2, "--dry-run"); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("would file run_state superseded") == len(_runs(q2)) and "D1816" in out, out
    assert not any(_state(spool2, _runs(q2)[k]["run_id"], "superseded") for k in _runs(q2)) and not any(_state(spool2, _runs(q1)[k]["run_id"], "generated") for k in _runs(q1))


# ---------------------------------------------------------------------------------- 284b: a retry is the same run, never its own twin
G1B = "810073"                                                     # a second attempt of the g01234 group (a job that died early and was resubmitted)
KEY_G01234 = f"sdfbase_full22-att-crescendo-{TAG}-sh0_1_2_3_4of9"  # the 282 / 282b producer key: job-independent, the shard group + total in it
KEY_G5678 = f"sdfbase_full22-att-crescendo-{TAG}-sh5_6_7_8of9"


def _world_282(tmp_path, monkeypatch, *, other_group: bool):
    """Both attempts of the g01234 group registered through the REAL 282 key -> the SAME run ids under two pending files; optionally the
    g5678 group's registration too (its own key, its own run ids)."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"
    wave = tmp_path / "exp" / "reproduce" / TAG / "waves" / "mt-s2r-cluster_a"; wave.mkdir(parents=True)
    p1a = _register(tmp_path, root, job=G1, job_name=f"f22-att-{TAG}-g01234-crescendo", evals="crescendo", producer_key=KEY_G01234)
    p1b = _register(tmp_path, root, job=G1B, job_name=f"f22-att-{TAG}-g01234-crescendo", evals="crescendo", producer_key=KEY_G01234)
    p2 = _register(tmp_path, root, job=G2, job_name=f"f22-att-{TAG}-g5678-crescendo", evals="crescendo", producer_key=KEY_G5678) if other_group else None
    prun = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_text())["private_run"]
    real_ids, IDS = _ids(tmp_path, p1a, spool, wave)
    for (_e, obj) in _runs(p1a):
        _multiturn_store(wave, prun, "crescendo", obj, nshards=9, goals=259, goal_ids=real_ids)
    return spool, wave, p1a, p1b, p2, IDS


def test_284b_two_attempts_of_one_group_register_the_same_runs_and_neither_close_supersedes_its_own_run(tmp_path, monkeypatch):
    """A retry of the canonical g01234 group under a new job id registers the SAME run ids (the job-independent key); closing
    either attempt files `generated` once and supersedes NOTHING — before 284b the other attempt was found as a twin and the canonical run
    was retired by its own closer."""
    spool, wave, p1a, p1b, _p2, IDS = _world_282(tmp_path, monkeypatch, other_group=False)
    ra, rb = _runs(p1a), _runs(p1b); assert set(ra) == set(rb) and len(ra) >= 2
    for key in ra:
        assert ra[key]["run_id"] == rb[key]["run_id"]                                          # the same run under two job ids (282)
    r = _close(p1a, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"{len(ra)} filed" in out and "superseded" not in out, out
    for key in ra:
        rid = ra[key]["run_id"]
        assert len(_state(spool, rid, "generated")) == 1 and not _state(spool, rid, "superseded"), rid
    # the retry's own close: the runs are already generated — nothing more filed, nothing superseded
    before = {key: len(_records(spool, ra[key]["run_id"])) for key in ra}
    r = _close(p1b, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already generated") == len(ra) and "superseded" not in out, out
    assert {key: len(_records(spool, ra[key]["run_id"])) for key in ra} == before
    for key in ra:
        assert len(_state(spool, ra[key]["run_id"], "generated")) == 1                          # no second `generated`


def test_284b_a_retry_of_the_canonical_group_beside_the_other_group_retires_only_the_other_groups_run(tmp_path, monkeypatch):
    spool, wave, p1a, p1b, p2, IDS = _world_282(tmp_path, monkeypatch, other_group=True)
    ra, r2 = _runs(p1a), _runs(p2)
    r = _close(p1b, spool, wave, *IDS); out = r.stdout + r.stderr                                # the RETRY closes first
    assert r.returncode == 0 and f"{len(ra)} filed" in out and f"{len(ra)} superseded (a twin registration of one store; D1816)" in out, out
    for key in ra:
        a, b = ra[key]["run_id"], r2[key]["run_id"]; assert a != b
        assert len(_state(spool, a, "generated")) == 1 and not _state(spool, a, "superseded")  # the canonical run survives its own retry
        st = _state(spool, b, "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == a and st[0]["extra"]["sibling_job_id"] == G1B
    r = _close(p1a, spool, wave, *IDS); out = r.stdout + r.stderr                                # the first attempt's close afterwards
    assert r.returncode == 0 and out.count("already generated") == len(ra) and "superseded (a twin" not in out, out
    for key in ra:
        assert len(_state(spool, ra[key]["run_id"], "generated")) == 1 and len(_state(spool, r2[key]["run_id"], "superseded")) == 1


def test_284b_both_groups_already_generated_the_g01234_reclose_retires_the_twin_without_a_second_generated(tmp_path, monkeypatch):
    """The historical both-generated case — an older closer filed `generated` for BOTH groups' runs; before 284b
    the early guard skipped the g01234 run before sibling discovery, so a reclose left two current runs. Now the reclose retires the g5678
    twin and files no second `generated`."""
    spool, wave, p1, p2, IDS = _world(tmp_path, monkeypatch)
    r1, r2 = _runs(p1), _runs(p2)
    for key in r1:                                                                               # an older closer's filings: generated on both
        for rid in (r1[key]["run_id"], r2[key]["run_id"]):
            w = emit.RecordWriter(str(spool), "sdfbase_full22", rid, emit.producer_block("older_closer", code_sha="c" * 40), push_script=None)
            w.emit_state("generated", "completion_artifact", str(wave / "x" / "progress.jsonl"), note="an older closer's filing (synthetic)")
            w.push(final=True)
    r = _close(p1, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already generated") == len(r1) and f"{len(r1)} superseded (a twin registration of one store; D1816)" in out, out
    for key in r1:
        a, b = r1[key]["run_id"], r2[key]["run_id"]
        assert len(_state(spool, a, "generated")) == 1 and not _state(spool, a, "superseded")  # no second generated for the survivor
        st = _state(spool, b, "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == a and len(_state(spool, b, "generated")) == 1
    # the twin's own reclose afterwards: already superseded, nothing written
    before = {key: len(_records(spool, r2[key]["run_id"])) for key in r2}
    r = _close(p2, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already superseded") == len(r2) and {key: len(_records(spool, r2[key]["run_id"])) for key in r2} == before, out


# ------------------------------------------------------------------------------------------- the BoN store's two registrations
# The 512 store is registered by the store512 job (f22-stat-<tag>-3) AND by the climb array (f22-bonclimb-<tag>, four tasks, one pending
# file each under the job-keyed key: the smoke's ten stores carried five current registrations). The store512 registration survives; a
# climb registration covering the SAME store (same store_uri, identical store_completion_sha256 and n_items_generated — FILED values) is
# filed `superseded` by it (reason duplicate_registration, extra.superseded_by_run_id) WHICHEVER closes last; a climb whose
# bytes or item count differ is held by name, no supersede. Race-free: decided under flock on the store's close lock — beside
# the store, <private run>/.<cell>.close.lock (bon_close_lock_path) — AND re-evaluated on every close. The real
# closer over the real registration (synthetic job ids); the records fold into a real registry.
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import test_admission as TA  # noqa: E402  (the gate's fixtures: the frozen inventory's inputs, the probe documents; it puts reproduce/lib on sys.path)
import test_registry_close_stores as TRC  # noqa: E402  (the module object: _register / _bon_store / _close read its TAG at call time)
from sisteer.registry import db as DB, emit_cli, harvest as H, synth  # noqa: E402
from test_registry_close_stores import CLOSER, OBJECTS, _bon_plan_config, _bon_store  # noqa: E402

S3 = "810081"                                           # the store512 job: f22-stat-<tag>-3
CLIMBS = ("810091", "810092", "810093", "810094")       # the climb array's four tasks: f22-bonclimb-<tag>, one job id (one pending file) each
CLIMB_KEY = f"sdfbase_full22-bonclimb-{TAG}"            # the job-independent key common.sh forms: one run per cell across the array
N_MAX = 3
D2599_LINE = "superseded (a climb registration duplicating the store512's; D2599)"
S3_NAME, CLIMB_NAME = f"f22-stat-{TAG}-3", f"f22-bonclimb-{TAG}"


def _bon_world(tmp_path, monkeypatch, *, climbs: int = 4, climb_key=None):
    """The store512 registration + `climbs` climb-task registrations of the same two BoN cells (job-keyed by default: distinct run ids per
    task, the live shape; `climb_key` = the job-independent key: the same run ids under every task), and both cells' complete 512 stores."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    p3 = _register(tmp_path, root, job=S3, job_name=S3_NAME, evals="strongreject_bon")
    pcs = [_register(tmp_path, root, job=j, job_name=CLIMB_NAME, evals="strongreject_bon", producer_key=climb_key) for j in CLIMBS[:climbs]]
    goals = int(json.loads(p3.read_text())["runs"][0]["plan"]["n_items_planned"]); assert goals > 0
    cfg = _bon_plan_config(tmp_path, goals=goals, n_max=N_MAX)
    stores = {obj: _bon_store(wave, obj, goals=goals, n_max=N_MAX, responded=goals * N_MAX, refused=0, config=cfg) for obj in OBJECTS}
    return root, spool, wave, p3, pcs, stores, goals


def _rid(pending, obj):
    return _runs(pending)[("strongreject_bon", obj)]["run_id"]


def _patch(spool, rid):
    """The last run_patch carrying the store facts (what the closer filed with `generated` — or, since round 3, before a `superseded`)."""
    pa = [p["fields"] for t, p, _pr in _records(spool, rid) if t == "run_patch" and "store_uri" in (p.get("fields") or {})]
    return pa[-1] if pa else None


FACTS = ("store_uri", "store_completion_artifact_uri", "store_completion_sha256", "n_items_generated")   # the inspected store facts a close files


def _row_facts(root, rid):
    """The folded run row's store facts and generation clock: {store_uri, store_completion_sha256, n_items_generated, generated_utc}."""
    with DB.reading(DB.db_path(str(root))) as conn:
        r = conn.execute("SELECT store_uri, store_completion_sha256, n_items_generated, generated_utc FROM run WHERE run_id = ?", (rid,)).fetchone()
    return dict(zip(("store_uri", "store_completion_sha256", "n_items_generated", "generated_utc"), r)) if r else None


def _assert_store_facts_patch(spool, rid, *, store_tail: str, n_items: int):
    """Round-3 Major on e62b57124: a run its OWN close retires as a duplicate registration of one store — never `generated` — carries a
    run_patch with the inspected store facts (store_uri, the completion artifact and its sha, n_items_generated) and NO generated_utc: the
    registry row then records the store the `superseded` record claims, which admission's producer_supersession verifies by the rows."""
    pa = _patch(spool, rid); assert pa is not None, rid
    assert pa["store_uri"].endswith(store_tail) and pa["store_completion_artifact_uri"] and len(pa["store_completion_sha256"]) == 64, pa
    assert int(pa["n_items_generated"]) == n_items and "generated_utc" not in pa, pa


def _fold(root):
    rep = H.ingest_once(str(root), workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()


def _currency(root, rids):
    """{run id: (lifecycle_state, is_current, why)} after the fold — v_run_current, the grader's and the census's one predicate."""
    out = {}
    with DB.reading(DB.db_path(str(root))) as conn:
        for rid in rids:
            st = conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()
            cur = conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()
            out[rid] = (st[0] if st else None, cur[0] if cur else None, cur[1] if cur else None)
    return out


def _superseded_events(root, rid):
    with DB.reading(DB.db_path(str(root))) as conn:
        return [json.loads(x) for (x,) in conn.execute("SELECT extra_json FROM run_state_event WHERE run_id = ? AND state = 'superseded'", (rid,))]


def _assert_one_current(root, s3_rid, climb_rids):
    cur = _currency(root, [s3_rid, *climb_rids])
    assert cur[s3_rid][0] == "generated" and cur[s3_rid][1] == 1 and cur[s3_rid][2] is None, cur
    facts3 = _row_facts(root, s3_rid)
    for c in climb_rids:
        assert cur[c][0] == "superseded" and cur[c][1] == 0 and "superseded" in str(cur[c][2]), (c, cur[c])
        evs = _superseded_events(root, c); assert len(evs) == 1 and evs[0]["superseded_by_run_id"] == s3_rid and evs[0]["decision"] == "D2599", evs
        f = _row_facts(root, c)                        # round-3 Major on e62b57124: the retired row records the store the record claims (admission reads the rows)
        assert {k: f[k] for k in ("store_uri", "store_completion_sha256", "n_items_generated")} == {k: facts3[k] for k in ("store_uri", "store_completion_sha256", "n_items_generated")}, (c, f, facts3)
    assert sum(1 for v in cur.values() if v[1] == 1) == 1


def _older_closer_filing(spool, rid, store, *, n_items, note="an older closer's filing (synthetic)"):
    """What a closer before the job-independent key left: run_patch {store facts} + run_state generated, no supersede."""
    art = str(store / "bon_served_manifest.json")
    w = emit.RecordWriter(str(spool), "sdfbase_full22", rid, emit.producer_block("older_closer", code_sha="c" * 40), push_script=None)
    w.emit_patch(fields={"store_uri": str(store), "store_completion_artifact_uri": art, "store_completion_sha256": hashlib.sha256(Path(art).read_bytes()).hexdigest(),
                         "n_items_generated": n_items, "generated_utc": "2026-10-11T00:00:00.000Z"})
    w.emit_state("generated", "completion_artifact", art, note=note)
    w.push(final=True)


def test_d456_the_store512_closing_first_files_generated_and_each_climb_close_supersedes_its_own_run(tmp_path, monkeypatch):
    """The store512 closes first (its runs `generated`); every climb task's close afterwards finds the store512's filing with the SAME store
    facts and files its OWN run `superseded` by it — reason duplicate_registration, extra.superseded_by_run_id, the matched
    facts under extra.duplicate_of — and no `generated`; the inspected store facts go onto the run
    FIRST, a run_patch without generated_utc, so the folded row records the store the record claims. A second close of a climb skips by name
    and writes nothing; the store512's second close is idempotent. Folded: exactly ONE current run per store (the store512's), the climbs
    lifecycle superseded with the store512's store facts and no generation clock."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch)
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "superseded" not in out, out
    assert out.count("has not closed `generated` — it supersedes itself at its own close") == 8, out       # four climb twins per cell, none closed yet
    for obj in OBJECTS:
        pa3 = _patch(spool, _rid(p3, obj)); assert pa3["store_uri"] == str(stores[obj]) and pa3["n_items_generated"] == goals * N_MAX
    for pc, job in zip(pcs, CLIMBS):
        r = _close(pc, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
        assert r.returncode == 0 and f"0 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in out, out
        for obj in OBJECTS:
            a, b = _rid(p3, obj), _rid(pc, obj); assert a != b
            st = _state(spool, b, "superseded"); assert len(st) == 1 and not _state(spool, b, "generated"), st
            # round-3 Major on e62b57124: the climb's row gets the inspected store facts — the store512's, no generated_utc — filed under the lock BEFORE the supersede
            pb = _patch(spool, b)
            assert pb is not None and {k: pb[k] for k in FACTS} == {k: _patch(spool, a)[k] for k in FACTS} and "generated_utc" not in pb, pb
            assert out.index(f"store facts strongreject_bon {obj} {b[:40]}") < out.index(f"superseded strongreject_bon {obj} {b[:40]}"), out
            s = st[0]
            assert s["evidence_kind"] == "completion_artifact" and s["evidence_uri"] == str(stores[obj] / "bon_served_manifest.json") and s["event_utc"]
            assert s["extra"]["superseded_by_run_id"] == a and s["extra"]["reason"] == "duplicate_registration" and s["extra"]["decision"] == "D2599"
            assert s["extra"]["sibling_job_id"] == S3 and s["extra"]["superseded_role"] == "climb" and s["extra"]["survivor_role"] == "store512"
            pa3 = _patch(spool, a)
            assert s["extra"]["duplicate_of"] == {k: pa3[k] for k in ("store_uri", "store_completion_sha256", "n_items_generated")}
            assert f"carried by {a}" in s["note"] and "store512 registration survives" in s["note"] and "D2599" in s["note"]
            assert f"superseded strongreject_bon {obj} {b[:40]}… by {a[:40]}… (a climb registration duplicating the store512's; D2599)" in out
    # a climb's second close: already superseded, by name, nothing written; the store512's second close: already generated, nothing more
    before = {obj: len(_records(spool, _rid(pcs[0], obj))) for obj in OBJECTS}
    r = _close(pcs[0], spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already superseded (a climb registration duplicating the store512's; D2599)") == 2 and "2 skipped" in out, out
    assert {obj: len(_records(spool, _rid(pcs[0], obj))) for obj in OBJECTS} == before
    before3 = {obj: len(_records(spool, _rid(p3, obj))) for obj in OBJECTS}
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already generated") == 2 and "superseded (" not in out, out
    assert {obj: len(_records(spool, _rid(p3, obj))) for obj in OBJECTS} == before3
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, _rid(p3, obj), [_rid(pc, obj) for pc in pcs])
        assert all(_row_facts(root, _rid(pc, obj))["generated_utc"] is None for pc in pcs)        # never generated: the facts without the clock


# ------------------------------------------------- round-3 Major on e62b57124: the retired climb's row must record its store for the gate
DEMO = "qwen35_4b_demo"      # the front-door row the gate can FREEZE (stage.expected_units declares its wave; `qwen35_4b` runs the single-phase recipe with none and refuses)


def _closer_module():
    """The closer as a module: its file_superseded is the record the closer BEFORE round 3 filed ALONE for a store512-first climb (the negative
    control's writer), its instrument_judge the registration's judge."""
    spec = importlib.util.spec_from_file_location("registry_close_stores_under_test", str(CLOSER))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _lock_of(store) -> Path:
    """The store's close lock as the closer names it: beside the store, <private run>/.<cell>.close.lock (bon_close_lock_path; round-4 Major on
    08c90e1cf) — the one path every close and every lifecycle writer of the store's runs take, before the store exists or after."""
    return Path(_closer_module().bon_close_lock_path(str(store)))


def _demo_inventory(state: Path, code_sha: str) -> dict:
    """The gate's frozen inventory for the demo key from the SAME unit plan 06 submits (test_admission.freeze's call), at the REGISTRATION's sha
    rather than the gate tests' forty-a placeholder (the register refuses a placeholder sha). Static + bon: the store512 unit's stage
    family is static, so `families=["bon"]` alone freezes the climb only. Returns the units by their `<family>/<unit>` tail."""
    (state / "gate").mkdir(parents=True, exist_ok=True)
    units = TA.ST.expected_units(DEMO, code_sha, str(TA.REPRO / "models.yaml"), families=["static", "bon"], goals_jsonl=str(TA.goal_fixture(state)),
                                 wave_config=str(TA.wave_config(state)), petri_bundle=str(TA.petri_bundle(state)))
    TA.ST.write_atomic(str(state / "gate" / "expected_units.json"), {"schema": 2, "model": DEMO, "view": "battery", "code_sha": code_sha, "units": units})
    return {u["logical_work_id"].split("/", 2)[2]: u for u in units}


def _grade(spool, entry: dict, store: Path, n_items: int, closer) -> None:
    """The survivor graded: a COMPLETE judgment for the registration's instrument judge through the real judging hook (judgment claim ->
    complete + run_state judged + run_patch), the daemon's shape minus the item rows (no API here; the fold reads the judgment's count)."""
    why, judge = closer.instrument_judge(entry, repo_root=str(REPO)); assert why is None, why
    scope = hooks.judgment_scope("tests.synthetic", {"run_id": entry["run_id"], "campaign": "sdfbase_full22"},
                                 {"judge_id": judge["judge_id"], "rubric_sha256": judge.get("rubric_sha256")}, spool=str(spool), push_script=None,
                                 registry_emit=True, store_uri=str(store))
    scope.claim(); scope.items([]); scope.metrics([]); scope.complete("complete", n_items_judged=n_items, n_judge_refusals=0)


def _gate_state(state: Path, units: dict, s3: dict, cl: dict, srow: dict):
    """The state root 06 --gate leaves for the two bon units over THESE run ids: the binding ledger (submitted + registered; the climb's one run
    per cell bound to its four array tasks), the coverage records (a satisfying store512 probe document per cell at the row's store; the climb's
    shared-run document over the four shards with `shard_indices` and `array_tasks`, naming the SUCCESSOR the registry names at its store) and
    the family's passing verdict. Returns (store512 lw, climb lw)."""
    US, UC = units["bon/store512"], units["bon/climb"]; lw_s, lw_c = US["logical_work_id"], UC["logical_work_id"]
    ST = TA.ST; ledger = str(state / "submissions.jsonl")
    ST.write_binding(ledger, ST.submitted_record(lw_s, S3, "cluster_a", f"{lw_s}@cluster_a@1", array=US.get("array")))
    ST.write_binding(ledger, {"logical_work_id": lw_s, "state": "registered", "run_ids": [s3[o] for o in OBJECTS], "job_id": S3, "cluster": "cluster_a",
                              "runs": [{"run_id": s3[o], "eval_id": "strongreject_bon", "object": o, "array_task": None} for o in OBJECTS]})
    ST.write_binding(ledger, ST.submitted_record(lw_c, CLIMBS[0], "cluster_a", f"{lw_c}@cluster_a@1", array=UC.get("array")))
    ST.write_binding(ledger, {"logical_work_id": lw_c, "state": "registered", "run_ids": [cl[o] for o in OBJECTS], "job_id": CLIMBS[0], "cluster": "cluster_a",
                              "runs": [{"run_id": cl[o], "eval_id": "strongreject_bon", "object": o, "array_task": t} for o in OBJECTS for t in range(4)]})
    ev_s = US["evals"]["strongreject_bon"]
    # a coverage record is tied to the inventory's coverage contract as it is to the run ids — the record 06 --gate writes
    # carries the contract block it read the wave under; this hand-written one carries the inventory's sha the same way (admission.inventory_contract_sha)
    cc = {"sha256": TA.A.inventory_contract_sha(str(state)), "version": ST.COVERAGE_CONTRACT_VERSION, "source": "test"}
    cov_s = {"logical_work_id": lw_s, "run_ids": [s3[o] for o in OBJECTS], "coverage_contract": cc,
             "runs": {s3[o]: {"eval_id": "strongreject_bon", "object": o, "array_task": None, "store_uri": srow[o]["store_uri"],
                              "probe": TA.probe_ok(ev_s, store=srow[o]["store_uri"]), "probe_error": None} for o in OBJECTS}}
    ST.write_atomic(ST.coverage_path(str(state), lw_s), cov_s)
    exp = ["gA", "gB", "gC"]
    cov_c = {"logical_work_id": lw_c, "run_ids": [cl[o] for o in OBJECTS], "coverage_contract": cc, "runs": {}}
    for o in OBJECTS:
        probe = {"layout": "bon_climb_flat", "path": f"/shared/bon_climb/{DEMO}", "exists": True, "notes": [], "shard_indices": [0, 1, 2, 3],
                 "shards": {f"shard{i:02d}": {"present": True, "expected": exp, "completed": list(exp), "errored": [], "refused": [], "notes": []} for i in range(4)}}
        cov_c["runs"][cl[o]] = {"eval_id": "strongreject_bon", "object": o, "array_task": 3, "array_tasks": [0, 1, 2, 3], "store_uri": srow[o]["store_uri"],
                                "probe": probe, "probe_error": None,
                                "successor": {"run_id": s3[o], "ruling_id": None, "store_uri": srow[o]["store_uri"], "cluster": "cluster_a", "owner": lw_s}}
    ST.write_atomic(ST.coverage_path(str(state), lw_c), cov_c)
    # the family verdict as `06 --gate bon --headline … --range …` records it: `run_ids` = the family's registered bindings (both units') at ruling
    # time — admission refuses a verdict whose run ids differ from the family's bindings, and one that names none
    (state / "gate" / "bon.json").write_text(json.dumps({"family": "bon", "headline": 0.5, "plausible_range": [0, 1], "verdict": "pass",
                                                         "run_ids": [s3[o] for o in OBJECTS] + [cl[o] for o in OBJECTS]}))
    return lw_s, lw_c


def test_d456_a_climb_the_store512_closed_before_folds_with_its_store_facts_and_the_admission_gate_reads_its_survivor_through_the_rows(tmp_path, monkeypatch):
    """When the store512 closed before the climb, the climb's close filed `superseded` DIRECTLY — no run_patch —
    and its registration supplies no store_uri either, so after the fold admission's producer_supersession (which verifies the record's claim by
    the ROWS: store_uri recorded on both and equal) refused the climb as having an unrecorded store; a normal, tested close order left the BoN
    climb unit unable to pass the gate even with its survivor graded. Now the climb's close files the inspected store facts (store_uri, artifact,
    sha, n_items; no generated_utc) under the store lock BEFORE the `superseded`. The whole path, nothing synthetic but the job ids: the REAL
    registration of the demo key's BoN store (the front-door row the gate freezes) by the store512 job and a climb task under the job-independent
    key, the real stores, the real closer store512-first (a dry run first: it names the facts and writes nothing), the survivor graded through the
    real judging hook, the records folded by the real harvester into a real registry, the gate's frozen inventory from the same seeds at the
    registration's sha, and reproduce/lib/admission.py over those rows: the climb's `supersession` resolves to the store512 run through the rows
    (no ruling), and the gate refuses the climb unit for NOTHING of its own — every problem it reports is the store512 unit's, re-spelled `via
    successor`. The negative control: the same records as the closer before this fix left them (the bare `superseded`, no run_patch) folded into
    a second registry refuse by that exact text.

    Residual outside this lane's files, named rather than hidden: the inventory freezes decode_policy_ref `qwen35_4b_demo|bon|off|grandfathered_T1`
    for both bon units (stage.expected_units takes the instrument seed's first ref for the model), a policy SUPERSEDED by the vendor row since
    2026-09-17 that v_run_decode_policy_match never stamps on a run generated after that date (these rows read `unrecorded`: the BoN producer
    files no decode_sent_json; a vendor sent set would stamp vendor_preset, which the gate then refuses against the frozen ref) — so the store512
    unit itself fails the decode check over real rows, and the climb exactly as far as its survivor does. The assertions bind the climb's admission
    to that residual: nothing else is reported, and both units pass the moment it is lifted."""
    monkeypatch.setattr(TRC, "TAG", DEMO)                                   # _register / _bon_store / _close spell the demo key's tag
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    code_sha = synth.code_sha("a")                                          # the registration's sha (test_registry_close_stores._register's default)
    state = tmp_path / "state"; units = _demo_inventory(state, code_sha)
    assert {"bon/store512", "bon/climb"} <= set(units) and units["bon/climb"]["array"] == "0-3" and units["bon/store512"]["model_id"] == DEMO
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    s3_name, climb_name = f"f22-stat-{DEMO}-3", f"f22-bonclimb-{DEMO}"
    p3 = _register(tmp_path, root, job=S3, job_name=s3_name, evals="strongreject_bon")
    pc = _register(tmp_path, root, job=CLIMBS[0], job_name=climb_name, evals="strongreject_bon", producer_key=f"sdfbase_full22-bonclimb-{DEMO}")
    goals = int(json.loads(p3.read_text())["runs"][0]["plan"]["n_items_planned"]); n_items = goals * N_MAX
    cfg = _bon_plan_config(tmp_path, goals=goals, n_max=N_MAX)
    stores = {obj: _bon_store(wave, obj, goals=goals, n_max=N_MAX, responded=n_items, refused=0, config=cfg) for obj in OBJECTS}
    closer = _closer_module(); s3 = {obj: _rid(p3, obj) for obj in OBJECTS}; cl = {obj: _rid(pc, obj) for obj in OBJECTS}
    # the store512 closes first and its runs are graded (the daemon grades a generated store while the climb array still runs)
    r = _close(p3, spool, wave, job_name=s3_name); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out, out
    for e in json.loads(p3.read_text())["runs"]:
        _grade(spool, e, stores[e["route"]["object"]], n_items, closer)
    # the negative control's registry: the records as the closer BEFORE this fix left them — the registration, the store512's filing and its
    # grading, and the climb's bare `superseded` (file_superseded alone, no run_patch)
    pre = tmp_path / "reg_pre"; _seeded_registry(pre)
    shutil.copytree(spool / "sdfbase_full22", pre / "records" / "sdfbase_full22", dirs_exist_ok=True)
    for obj in OBJECTS:
        pa3 = _patch(spool, s3[obj])
        closer.file_superseded(spool=str(pre / "records"), push_script=None, campaign="sdfbase_full22", rid=cl[obj], by_rid=s3[obj], artifact=pa3["store_completion_artifact_uri"],
                               producer=emit.producer_block(closer.SCRIPT, code_sha=synth.code_sha("b")), sibling_job=S3, eval_id="strongreject_bon", obj=obj, dry_run=False,
                               decision=closer.BON_SUPERSEDED_DECISION, rule=closer.BON_SUPERSEDED_RULE, kind=closer.BON_SUPERSEDED_KIND,
                               extra={"duplicate_of": {k: pa3[k] for k in closer.BON_DUPLICATE_FIELDS}, "superseded_role": "climb", "survivor_role": "store512"})
    # the climb's close on this tree: a dry run names the facts it would file and writes nothing; the real close files them under the lock, then the supersede
    before = {obj: len(_records(spool, cl[obj])) for obj in OBJECTS}
    r = _close(pc, spool, wave, "--dry-run", job_name=climb_name); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("(the inspected store facts of a registration this close retires)") == 2 and out.count("would file run_state superseded") == 2, out
    assert {obj: len(_records(spool, cl[obj])) for obj in OBJECTS} == before
    r = _close(pc, spool, wave, job_name=climb_name); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"0 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in out, out
    for obj in OBJECTS:
        a, b = s3[obj], cl[obj]
        assert out.index(f"store facts strongreject_bon {obj} {b[:40]}") < out.index(f"superseded strongreject_bon {obj} {b[:40]}"), out   # the facts first, then the retirement
        pb = _patch(spool, b); assert pb is not None and {k: pb[k] for k in FACTS} == {k: _patch(spool, a)[k] for k in FACTS} and "generated_utc" not in pb, pb
        assert len(_state(spool, b, "superseded")) == 1 and not _state(spool, b, "generated")
    _fold(root); _fold(pre)
    # decode-recording (the registry owner, 2026-10-13): the retired climb's store-facts patch and the survivor's `generated` patch both carry the
    # request's observed sampler -- bon_served_manifest.json#decode {"temperature": 1.0}, the producer's grandfathered_T1 row as a statement --
    # and after the real fold the stamped match reads EXACT under the producer's grandfathered_T1 row: (A) landed (origin/main
    # registry-decode-history, the registry owner 2026-10-13 22:34) -- the nine grandfathered_T1 rows are live and non-default (no successor), so
    # a T=1 draw after 2026-09-17 matches it while the vendor row (0.7) stays in scope and is no candidate. Before (A) this pin read `no_match`.
    for obj in OBJECTS:
        for rid in (s3[obj], cl[obj]):
            pb_ = _patch(spool, rid)
            assert pb_["decode_sent_json"] == {"temperature": 1.0} and pb_["decode_temperature_sent"] == 1.0, pb_
            assert pb_["extra_json"]["decode_evidence"]["producer_policy_id"] == TRC.BON_POLICY_ID and pb_["field_confidence_json"]["decode_sent_json"]["confidence"] == "recorded", pb_
    with DB.reading(DB.db_path(str(root))) as conn:
        for rid in (*s3.values(), *cl.values()):
            drow = conn.execute("SELECT decode_sent_json, decode_temperature_sent, decode_policy_match, decode_policy_id, decode_source, decode_seed_rule FROM run WHERE run_id = ?", (rid,)).fetchone()
            assert (json.loads(drow[0]), drow[1], drow[2], drow[3], drow[4], drow[5]) == ({"temperature": 1.0}, 1.0, "exact", TRC.BON_POLICY_ID, "grandfathered_T1", "per_draw"), (rid, tuple(drow))
            cands = conn.execute("SELECT candidates_json, in_scope_json FROM v_run_decode_policy_match WHERE run_id = ?", (rid,)).fetchone()
            assert json.loads(cands[0]) == [TRC.BON_POLICY_ID] and {TRC.BON_POLICY_ID, f"{DEMO}|bon|off|vendor_preset@v1"} <= set(json.loads(cands[1])), (rid, tuple(cands))
    # the rows: the climb superseded with the store512's store facts and no generation clock; its supersession resolves through the rows, no ruling
    with DB.reading(DB.db_path(str(root))) as conn:
        srow = {obj: TA.A.registry_row(conn, s3[obj]) for obj in OBJECTS}
        for obj in OBJECTS:
            assert srow[obj]["lifecycle_state"] in TA.A.COMPLETE_STATES and srow[obj]["store_uri"] == str(stores[obj]), srow[obj]
            assert TA.A.registry_row(conn, cl[obj])["lifecycle_state"] == "superseded"
            sup = TA.A.supersession(conn, cl[obj])
            assert sup["successor"] == s3[obj] and sup["ruling_id"] is None and sup["duplicate_registration"] and sup["decision"] == "D2599", sup
            assert conn.execute("SELECT COUNT(*) FROM ruling").fetchone()[0] == 0
    for obj in OBJECTS:
        f, f3 = _row_facts(root, cl[obj]), _row_facts(root, s3[obj])
        assert f["store_uri"] == f3["store_uri"] == str(stores[obj]) and f["store_completion_sha256"] == f3["store_completion_sha256"] and f["n_items_generated"] == f3["n_items_generated"] == n_items, (f, f3)
        assert f["generated_utc"] is None and f3["generated_utc"]
    assert [v[1] for v in _currency(root, [*s3.values(), *cl.values()]).values()] == [1, 1, 0, 0]
    # the gate over those rows
    lw_s, lw_c = _gate_state(state, units, s3, cl, srow)
    problems, ok = TA.A.check(str(state), DB.db_path(str(root)), {}, families=["bon"])
    # (A) landed (origin/main registry-decode-history, merged in integration round 3a): the rows read `exact` under the grandfathered_T1 row, so the
    # decode residual this block once named -- "while the unit freezes decode policy 'qwen35_4b_demo|bon|off|grandfathered_T1'", the ONE problem
    # every unit carried, the climb's via its successor -- is gone, and the gate ADMITS both units with no problem: the climb's work stands in
    # through its survivor's rows (the supersession above resolves through the rows, no ruling), the store512's through its own
    assert problems == [] and lw_c in ok and lw_s in ok, (problems, ok)
    # the negative control: the refusal, byte for byte, over the pre-fix records — the row records no store, nothing stands in for the climb
    with DB.reading(DB.db_path(str(pre))) as conn:
        for obj in OBJECTS:
            assert TA.A.registry_row(conn, cl[obj])["store_uri"] is None and TA.A.registry_row(conn, cl[obj])["lifecycle_state"] == "superseded"
            assert TA.A.supersession(conn, cl[obj]) == {"refused": f"the producer's duplicate_registration record (D2599) names {s3[obj]} as the other registration of this "
                                                                   f"run's store, but store_uri is unrecorded on this run — a duplicate registration is verified by the rows, never by the record"}
    problems_pre, ok_pre = TA.A.check(str(state), DB.db_path(str(pre)), {}, families=["bon"])
    for obj in OBJECTS:
        assert any(f"{lw_c}: run {cl[obj]} (strongreject_bon) is superseded but the producer's duplicate_registration record (D2599) names {s3[obj]} as the other registration "
                   f"of this run's store, but store_uri is unrecorded on this run" in p and p.endswith("— nothing stands in for its work") for p in problems_pre), problems_pre
    assert lw_c not in ok_pre


def test_d456_the_climbs_closing_first_file_generated_and_the_store512_closing_last_retires_every_twin(tmp_path, monkeypatch):
    """The other interleaving: the four climb tasks close before the store512 — each files `generated` (the store512 has no filing to judge
    against; the closer says so) — and the store512's close afterwards files its own `generated` and retires all eight climb registrations
    whose filed facts equal its own. The climbs' `generated` stands corrected by the supersede; folded, one current run per store."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch)
    for pc in pcs:
        r = _close(pc, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
        assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "superseded" not in out, out
        assert out.count("has not closed `generated` — this climb run closes generated; the store512 retires it when it closes last (D456)") == 2, out
        for obj in OBJECTS:
            assert len(_state(spool, _rid(pc, obj), "generated")) == 1 and _patch(spool, _rid(pc, obj))["n_items_generated"] == goals * N_MAX
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"2 filed, 0 skipped, 0 refused of 2 registered run(s), 8 {D2599_LINE}" in out, out
    for obj in OBJECTS:
        a = _rid(p3, obj); assert len(_state(spool, a, "generated")) == 1 and not _state(spool, a, "superseded")
        for pc, job in zip(pcs, CLIMBS):
            b = _rid(pc, obj); st = _state(spool, b, "superseded")
            assert len(st) == 1 and len(_state(spool, b, "generated")) == 1, (b, st)                   # the older generated stays; the supersede is the correction
            assert st[0]["extra"]["superseded_by_run_id"] == a and st[0]["extra"]["sibling_job_id"] == S3 and st[0]["extra"]["decision"] == "D2599"
            assert st[0]["extra"]["duplicate_of"]["store_completion_sha256"] == _patch(spool, a)["store_completion_sha256"]
    # a climb's reclose afterwards: already superseded, nothing written
    before = {obj: len(_records(spool, _rid(pcs[2], obj))) for obj in OBJECTS}
    r = _close(pcs[2], spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already superseded") == 2 and {obj: len(_records(spool, _rid(pcs[2], obj))) for obj in OBJECTS} == before, out
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, _rid(p3, obj), [_rid(pc, obj) for pc in pcs])


def _popen_close(pending, spool, wave, log: Path, job_name: str) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src")); env.pop("SISTEER_REGISTRY_DISABLED", None)
    argv = [sys.executable, str(CLOSER), "--pending", str(pending), "--wave-root", str(wave), "--tag", TAG, "--spool", str(spool), "--code-sha", synth.code_sha("b"),
            "--job-name", job_name]
    return subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, text=True)


def _wait_for(pred, timeout_s: float, what: str) -> None:
    t0 = time.monotonic()
    while not pred():
        assert time.monotonic() - t0 < timeout_s, f"timed out waiting for {what}"
        time.sleep(0.2)


def _run_states(spool, rids):
    return {rid: sorted(p["state"] for t, p, _pr in _records(spool, rid) if t == "run_state" and p["state"] in ("generated", "superseded")) for rid in rids}


def test_d456_two_closers_racing_on_one_store_decide_under_the_store_lock_and_leave_one_current_run(tmp_path, monkeypatch):
    """The same-instant case: the store512 closer and a climb closer run CONCURRENTLY over one store. (a) With the
    store's close lock held by the test, BOTH closers block at it (each says so) with NOTHING filed for any run — neither decides outside the
    lock; released, they serialise: whichever goes first, the second sees its filing, and the end state is the store512 `generated` once and
    the climb `superseded` once (its `generated` 0 or 1 times, by order) — one current run per store. (b) Launched at the same instant with
    no lock held, the same end state."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=1)
    pc = pcs[0]; rids3 = [_rid(p3, o) for o in OBJECTS]; ridsc = [_rid(pc, o) for o in OBJECTS]
    # (a) the test holds BOTH stores' locks (the closer takes the first store's first; both block there)
    held = []
    for obj in OBJECTS:
        fh = open(_lock_of(stores[obj]), "a+"); fcntl.flock(fh.fileno(), fcntl.LOCK_EX); held.append(fh)
    log3, logc = tmp_path / "close_s3.log", tmp_path / "close_climb.log"
    procs = [_popen_close(p3, spool, wave, log3, S3_NAME), _popen_close(pc, spool, wave, logc, CLIMB_NAME)]
    try:
        _wait_for(lambda: all("waiting for the store close lock" in lg.read_text() for lg in (log3, logc)), 180, "both closers to reach the store lock")
        assert all(p.poll() is None for p in procs), [lg.read_text() for lg in (log3, logc)]
        assert _run_states(spool, rids3 + ridsc) == {rid: [] for rid in rids3 + ridsc}                 # nothing decided, nothing filed, outside the lock
        time.sleep(1.5)                                                                               # still nothing while the lock is held
        assert _run_states(spool, rids3 + ridsc) == {rid: [] for rid in rids3 + ridsc} and all(p.poll() is None for p in procs)
    finally:
        for fh in held:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN); fh.close()
    for p in procs:
        assert p.wait(timeout=600) == 0, (log3.read_text(), logc.read_text())
    out3, outc = log3.read_text(), logc.read_text()
    assert "waiting for the store close lock" in out3 and "waiting for the store close lock" in outc
    for obj in OBJECTS:
        a, b = _rid(p3, obj), _rid(pc, obj)
        assert len(_state(spool, a, "generated")) == 1 and not _state(spool, a, "superseded"), obj
        st = _state(spool, b, "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == a and st[0]["extra"]["decision"] == "D2599", obj
        assert len(_state(spool, b, "generated")) <= 1
    # exactly one of the two closers filed the climb's supersede, and it names the order it saw
    assert (out3.count(D2599_LINE) == 1) != (outc.count(D2599_LINE) == 1), (out3, outc)
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, _rid(p3, obj), [_rid(pc, obj)])
    # (b) the same instant, no lock held by anyone but the closers themselves
    root2, spool2, wave2, q3, qcs, _stores2, _g = _bon_world(tmp_path / "b", monkeypatch, climbs=1)
    qc = qcs[0]; logs = (tmp_path / "b_s3.log", tmp_path / "b_climb.log")
    procs = [_popen_close(q3, spool2, wave2, logs[0], S3_NAME), _popen_close(qc, spool2, wave2, logs[1], CLIMB_NAME)]
    for p in procs:
        assert p.wait(timeout=600) == 0, [lg.read_text() for lg in logs]
    for obj in OBJECTS:
        a, b = _rid(q3, obj), _rid(qc, obj)
        assert len(_state(spool2, a, "generated")) == 1 and not _state(spool2, a, "superseded")
        assert len(_state(spool2, b, "superseded")) == 1 and len(_state(spool2, b, "generated")) <= 1
    _fold(root2)
    for obj in OBJECTS:
        _assert_one_current(root2, _rid(q3, obj), [_rid(qc, obj)])


def test_d456_a_both_generated_store_left_by_an_older_closer_is_settled_by_the_next_close_of_either_side(tmp_path, monkeypatch):
    """The re-evaluated half (the live shape of 2026-10-11: five `generated` registrations per store, no supersede): an older closer filed
    `generated` with identical store facts for the store512 run AND the four climb runs. A dry run says what it would file and writes nothing.
    Then the next close of EITHER side settles it — a climb's reclose supersedes its own run (its `generated` stands corrected); the
    store512's reclose files no second `generated` and retires the three climbs still current; a later sweep finds nothing left to do."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch)
    for obj in OBJECTS:
        for rid in (_rid(p3, obj), *(_rid(pc, obj) for pc in pcs)):
            _older_closer_filing(spool, rid, stores[obj], n_items=goals * N_MAX)
    # the dry run over the store512's pending file: eight supersedes named, nothing written
    before = {rid: len(_records(spool, rid)) for obj in OBJECTS for rid in (_rid(p3, obj), *(_rid(pc, obj) for pc in pcs))}
    r = _close(p3, spool, wave, "--dry-run", job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("would file run_state superseded") == 8 and out.count("D2599") >= 8 and "would file run_patch" not in out, out
    assert {rid: len(_records(spool, rid)) for rid in before} == before
    # a climb task's reclose: its own runs superseded (one record each), no second generated, 0 filed
    r = _close(pcs[1], spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"0 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in out and "already generated" not in out, out
    for obj in OBJECTS:
        b = _rid(pcs[1], obj); st = _state(spool, b, "superseded")
        assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == _rid(p3, obj) and len(_state(spool, b, "generated")) == 1
    # the store512's reclose: already generated (no second record of its own), the three remaining climb twins retired, the settled one left alone
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("already generated") == 2 and f"0 filed, 2 skipped, 0 refused of 2 registered run(s), 6 {D2599_LINE}" in out, out
    for obj in OBJECTS:
        assert len(_state(spool, _rid(p3, obj), "generated")) == 1 and not _state(spool, _rid(p3, obj), "superseded")
        for pc in pcs:
            assert len(_state(spool, _rid(pc, obj), "superseded")) == 1, (pc, obj)                     # exactly one supersede each, whoever filed it
    # a later sweep of either side: nothing left to file
    for pend, name in ((p3, S3_NAME), (pcs[3], CLIMB_NAME)):
        before = {rid: len(_records(spool, rid)) for obj in OBJECTS for rid in (_rid(p3, obj), *(_rid(pc, obj) for pc in pcs))}
        r = _close(pend, spool, wave, job_name=name); out = r.stdout + r.stderr
        assert r.returncode == 0 and "superseded strongreject_bon" not in out and {rid: len(_records(spool, rid)) for rid in before} == before, out   # no new filing: the skip lines only
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, _rid(p3, obj), [_rid(pc, obj) for pc in pcs])


def test_d456_a_climb_whose_bytes_or_item_count_differ_is_held_by_name_and_never_superseded(tmp_path, monkeypatch):
    """Not a duplicate: (a) the store512 filed, then the store's completion artifact changed bytes (the same accounting re-serialised:
    completeness still holds, the sha does not) — the climb's close of that cell is REFUSED by name (held; nothing filed; the run stays
    running and is listed for the exit path) while the untouched cell's climb run is superseded; the original bytes restored, the next
    close supersedes it. (b) An older filing of a climb run with a DIFFERENT n_items_generated: the store512's close files its own
    `generated`, HOLDS that twin by name with no supersede — both stay current, the grader's refusal is the right outcome — and the summary
    counts it."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=1)
    pc = pcs[0]
    r = _close(p3, spool, wave, job_name=S3_NAME); assert r.returncode == 0, r.stdout + r.stderr
    man = stores["none"] / "bon_served_manifest.json"; original = man.read_bytes()
    man.write_text(json.dumps(json.loads(original), indent=1))                                        # the same facts, other bytes
    assert man.read_bytes() != original
    r = _close(pc, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 2, out
    a_none, b_none = _rid(p3, "none"), _rid(pc, "none")
    assert f"REFUSED strongreject_bon none {b_none[:40]}…: not a duplicate of the store512 registration {a_none[:40]}… (job {S3}): store_completion_sha256 " in out
    assert "HELD by name (D456: bytes or item count differ), no supersede filed; the run stays running; file by hand" in out
    assert not _records(spool, b_none) or not any(t == "run_state" and p["state"] in ("generated", "superseded") for t, p, _ in _records(spool, b_none))
    assert f"1 refused of 2 registered run(s), 1 {D2599_LINE}" in out and len(_state(spool, _rid(pc, "prompted__safety_refusal"), "superseded")) == 1
    side = json.loads((Path(str(pc) + ".refused_run_ids.json")).read_text()); assert side["refused_run_ids"] == [b_none]
    man.write_bytes(original)                                                                          # the bytes the store512 filed: a duplicate again
    r = _close(pc, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"1 skipped, 0 refused of 2 registered run(s), 1 {D2599_LINE}" in out, out
    assert len(_state(spool, b_none, "superseded")) == 1 and not _state(spool, b_none, "generated") and not Path(str(pc) + ".refused_run_ids.json").exists()
    # (b) the count differs: an older filing of the climb's `none` run one item short
    root2, spool2, wave2, q3, qcs, stores2, g2 = _bon_world(tmp_path / "n", monkeypatch, climbs=1)
    qc = qcs[0]
    _older_closer_filing(spool2, _rid(qc, "none"), stores2["none"], n_items=g2 * N_MAX - 1)
    r = _close(q3, spool2, wave2, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "1 held (a BoN registration whose bytes or item count differ from its twin's; D456" in out, out
    assert f"HELD strongreject_bon none {_rid(qc, 'none')[:40]}… (climb job {CLIMBS[0]}): not a duplicate of this store512 registration — n_items_generated {g2 * N_MAX} vs {g2 * N_MAX - 1} (D456" in out
    assert "no supersede filed, both registrations stay current until a human rules" in out and "superseded (" not in out
    assert not _state(spool2, _rid(qc, "none"), "superseded") and len(_state(spool2, _rid(q3, "none"), "generated")) == 1
    _fold(root2)
    cur = _currency(root2, [_rid(q3, "none"), _rid(qc, "none")])
    assert all(v[1] == 1 for v in cur.values()), cur                                                   # two current registrations: a human rules (the grader refuses the store by name)


def test_d456_the_job_independent_climb_key_registers_one_run_per_cell_across_the_array_and_the_store512_retires_it_once(tmp_path, monkeypatch):
    """The companion (common.sh f22_registry_producer_key, f22-bonclimb -> sdfbase_full22-bonclimb-<tag>): the array's four tasks register the
    SAME run per cell under four pending files. The first task to close files `generated`; the others find their run generated and the
    store512 unfiled, and write nothing (a re-attempt / the same run is never its own twin); the store512's close retires that ONE run per
    cell — one supersede record each, never four."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climb_key=CLIMB_KEY)
    for obj in OBJECTS:
        assert len({_rid(pc, obj) for pc in pcs}) == 1 and _rid(pcs[0], obj) != _rid(p3, obj)
        assert _runs(pcs[0])[("strongreject_bon", obj)]["plan"]["producer_run_key_legacy"] == CLIMB_KEY
    r = _close(pcs[0], spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out, out
    for pc in pcs[1:]:
        before = {obj: len(_records(spool, _rid(pc, obj))) for obj in OBJECTS}
        r = _close(pc, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
        assert r.returncode == 0 and out.count("already generated") == 2 and out.count("this climb run stays generated; the store512 retires it when it closes last") == 2, out
        assert "superseded" not in out and {obj: len(_records(spool, _rid(pc, obj))) for obj in OBJECTS} == before
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"2 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in out, out
    for obj in OBJECTS:
        b = _rid(pcs[0], obj); assert len(_state(spool, b, "superseded")) == 1 and len(_state(spool, b, "generated")) == 1
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, _rid(p3, obj), [_rid(pcs[0], obj)])


# ------------------------------------------------------------------ round 1 on 572921cd6: candidates are THIS store's filings; one transition per shared run
S3_OTHER = "810070"                                     # a second store512 job of the same tag (job-keyed, as every stat-3 before the job-independent key): sorts BEFORE S3 in the pending dir
_OTHER_STORE_NOTE = "filed another store"
_PEER_GENERATED = "already generated under the lock by another close of this run"
_PEER_SUPERSEDED = "already superseded under the lock by another close of this run"
_PEER_CERTIFIED = "the no-fit certificate is already filed"


def _judgment_statuses(spool, rid):
    return [p.get("status") for t, p, _pr in _records(spool, rid) if t == "judgment"]


def _patches(spool, rid):
    return [p["fields"] for t, p, _pr in _records(spool, rid) if t == "run_patch" and "store_uri" in (p.get("fields") or {})]


def test_d456_a_store512_filing_over_another_store_is_no_candidate_never_a_hold_or_a_refusal(tmp_path, monkeypatch):
    """Sibling discovery matched the measurement (eval, object, model_id, view_legacy, item_set_id) but not the
    STORE, and a climb's close took the first store512 filing it found — so a store512 registration of the same measurement whose filing
    covers ANOTHER store (the same tag closed under another wave root; a job-keyed stat-3 re-submission) read as a `store_uri` mismatch and
    the climb was REFUSED as a non-duplicate. The duplicate relation starts at `same store_uri`: a filing over another store is no
    candidate at all — noted, never held, never refused — and the bytes-or-count rule applies to this store's filings alone."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=1, climb_key=CLIMB_KEY)
    pc = pcs[0]
    p3b = _register(tmp_path, root, job=S3_OTHER, job_name=S3_NAME, evals="strongreject_bon")      # the same measurement, another run id
    for obj in OBJECTS:
        assert len({_rid(p3, obj), _rid(p3b, obj), _rid(pc, obj)}) == 3
    cfg = _bon_plan_config(tmp_path, goals=goals, n_max=N_MAX); wave_other = tmp_path / "wave_other"
    other = {obj: _bon_store(wave_other, obj, goals=goals, n_max=N_MAX, responded=goals * N_MAX, refused=0, config=cfg) for obj in OBJECTS}
    assert all(str(other[obj]) != str(stores[obj]) for obj in OBJECTS)
    r = _close(p3b, spool, wave_other, job_name=S3_NAME); out = r.stdout + r.stderr                 # the other store's store512 filing, first in pending order
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "held" not in out, out
    for obj in OBJECTS:
        assert _patch(spool, _rid(p3b, obj))["store_uri"] == str(other[obj])
    # the climb closes THIS store: p3b's filing is no candidate (another store), p3 has not filed -> `generated`, nothing refused, nothing held
    r = _close(pc, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused of 2 registered run(s)" in out and "superseded" not in out and "held" not in out, out
    assert out.count(_OTHER_STORE_NOTE) == 2 and "not a duplicate of the store512 registration" not in out and "REFUSED" not in out and "HELD" not in out, out
    assert out.count("has not closed `generated` over this store — this climb run closes generated; the store512 retires it when it closes last (D456)") == 2, out
    for obj in OBJECTS:
        b = _rid(pc, obj); assert len(_state(spool, b, "generated")) == 1 and _patch(spool, b)["store_uri"] == str(stores[obj]) and not _state(spool, b, "superseded")
    assert not Path(str(pc) + ".refused_run_ids.json").exists()
    # p3b's reclose over ITS store: the climb's filing covers another store — noted, not held, no supersede
    before = {obj: len(_records(spool, _rid(pc, obj))) for obj in OBJECTS}
    r = _close(p3b, spool, wave_other, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("its filing covers another store") == 2 and "held" not in out and "HELD" not in out and "superseded (" not in out, out
    assert {obj: len(_records(spool, _rid(pc, obj))) for obj in OBJECTS} == before
    # p3 closes this store last: its `generated`, and the climb retired by IT (never by the other store's registration)
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"2 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in out and "held" not in out, out
    for obj in OBJECTS:
        st = _state(spool, _rid(pc, obj), "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == _rid(p3, obj) and st[0]["extra"]["sibling_job_id"] == S3
        assert st[0]["extra"]["duplicate_of"]["store_uri"] == str(stores[obj])
        assert len(_state(spool, _rid(p3b, obj), "generated")) == 1 and not _state(spool, _rid(p3b, obj), "superseded")
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, _rid(p3, obj), [_rid(pc, obj)])
        cur = _currency(root, [_rid(p3b, obj)]); assert cur[_rid(p3b, obj)][0] == "generated" and cur[_rid(p3b, obj)][1] == 1, cur   # the other store's run: out of reach


def test_d456_among_several_store512_filings_of_one_store_the_climb_is_superseded_by_the_one_whose_facts_match(tmp_path, monkeypatch):
    """Round-1 Major on 572921cd6, the masking half: with several store512 filings of ONE store (job-keyed stat-3 registrations before
    The live `_u0` tag carries three), the climb's close took the FIRST in pending order and refused when it differed, though another
    matched exactly. Now the survivor is the filing of this store whose facts match; when several match, the earliest job, said so by name;
    a retired store512 is never a survivor; the refusal names the count of differing filings when every one differs."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=1, climb_key=CLIMB_KEY)
    pc = pcs[0]
    p3b = _register(tmp_path, root, job=S3_OTHER, job_name=S3_NAME, evals="strongreject_bon")
    r = _close(p3b, spool, wave, job_name=S3_NAME); assert r.returncode == 0 and "2 filed" in (r.stdout + r.stderr), r.stdout + r.stderr   # the ORIGINAL bytes
    man = stores["none"] / "bon_served_manifest.json"; original = man.read_bytes()
    man.write_text(json.dumps(json.loads(original), indent=1)); assert man.read_bytes() != original                       # the same facts, other bytes
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr                                             # the NEW bytes for `none`
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "held" not in out, out
    assert _patch(spool, _rid(p3, "none"))["store_completion_sha256"] != _patch(spool, _rid(p3b, "none"))["store_completion_sha256"]
    assert _patch(spool, _rid(p3, "prompted__safety_refusal"))["store_completion_sha256"] == _patch(spool, _rid(p3b, "prompted__safety_refusal"))["store_completion_sha256"]
    r = _close(pc, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"0 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in out and "REFUSED" not in out, out
    st = _state(spool, _rid(pc, "none"), "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == _rid(p3, "none") and st[0]["extra"]["sibling_job_id"] == S3
    assert st[0]["extra"]["duplicate_of"]["store_completion_sha256"] == _patch(spool, _rid(p3, "none"))["store_completion_sha256"]
    st = _state(spool, _rid(pc, "prompted__safety_refusal"), "superseded")
    assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == _rid(p3b, "prompted__safety_refusal") and st[0]["extra"]["sibling_job_id"] == S3_OTHER
    assert f"2 store512 registrations of this store filed identical facts (jobs {S3_OTHER}, {S3}); D456 ranks none of them against each other — superseded by the earliest, job {S3_OTHER}" in out, out
    assert all(not _state(spool, _rid(pc, obj), "generated") for obj in OBJECTS)
    # a retired store512 is never the survivor: a second world, the earliest store512 filing retired by hand, the climb goes to the other
    root2, spool2, wave2, q3, qcs, stores2, g2 = _bon_world(tmp_path / "r", monkeypatch, climbs=1, climb_key=CLIMB_KEY)
    qc = qcs[0]; q3b = _register(tmp_path / "r", root2, job=S3_OTHER, job_name=S3_NAME, evals="strongreject_bon")
    for pend in (q3b, q3):
        r = _close(pend, spool2, wave2, job_name=S3_NAME); assert r.returncode == 0, r.stdout + r.stderr
    for obj in OBJECTS:
        w = emit.RecordWriter(str(spool2), "sdfbase_full22", _rid(q3b, obj), emit.producer_block("hand", code_sha="d" * 40), push_script=None)
        w.emit_state("superseded", "ruling", "ruling://synthetic", note="retired by hand (synthetic)"); w.push(final=True)
    r = _close(qc, spool2, wave2, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"2 {D2599_LINE}" in out and out.count("is itself retired — not a survivor") == 2, out
    for obj in OBJECTS:
        st = _state(spool2, _rid(qc, obj), "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == _rid(q3, obj)
    # every filing of this store differs: refused by name, the count of the others said
    root3, spool3, wave3, s3, scs, stores3, g3 = _bon_world(tmp_path / "d", monkeypatch, climbs=1, climb_key=CLIMB_KEY)
    sc = scs[0]; s3b = _register(tmp_path / "d", root3, job=S3_OTHER, job_name=S3_NAME, evals="strongreject_bon")
    for pend in (s3b, s3):
        r = _close(pend, spool3, wave3, job_name=S3_NAME); assert r.returncode == 0, r.stdout + r.stderr
    man3 = stores3["none"] / "bon_served_manifest.json"; man3.write_text(json.dumps(json.loads(man3.read_bytes()), indent=2))     # neither filing's bytes
    r = _close(sc, spool3, wave3, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 2 and f"not a duplicate of the store512 registration {_rid(s3b, 'none')[:40]}… (job {S3_OTHER}): store_completion_sha256 " in out, out
    assert "; and 1 more store512 filing(s) of this store, none a duplicate" in out and f"1 refused of 2 registered run(s), 1 {D2599_LINE}" in out, out
    assert not _state(spool3, _rid(sc, "none"), "superseded") and not _state(spool3, _rid(sc, "none"), "generated")


def _hold_store_locks(stores):
    held = {}
    for obj in OBJECTS:
        fh = open(_lock_of(stores[obj]), "a+"); fcntl.flock(fh.fileno(), fcntl.LOCK_EX); held[obj] = fh
    return held


def _release_store_by_store(held, stores, logs, procs):
    """Both tasks are blocked at the FIRST store's lock, past their unlocked reads. Release that store alone; the task that loses it finishes
    the store under the lock and moves on; both then block at the SECOND store's lock (the test still holds it) — past that store's unlocked
    reads too — and only then is the second released: every cell is decided by two tasks that each read `not yet` before the lock."""
    first, second = OBJECTS
    fcntl.flock(held[first].fileno(), fcntl.LOCK_UN); held[first].close()
    _wait_for(lambda: all(f"waiting for the store close lock {_lock_of(stores[second])}" in lg.read_text() for lg in logs) or any(p.poll() is not None for p in procs),
              180, "both tasks to reach the second store's lock")
    assert all(p.poll() is None for p in procs), [lg.read_text() for lg in logs]
    fcntl.flock(held[second].fileno(), fcntl.LOCK_UN); held[second].close()


def test_d456_two_tasks_of_the_climb_array_closing_one_run_at_the_same_instant_file_one_transition_and_one_certificate(tmp_path, monkeypatch):
    """Round-1 Major on 572921cd6: since the job-independent key the climb array's four tasks share ONE run id, and each task's close read
    `generated` / `superseded` / the certificate BEFORE taking the store lock — two tasks reaching the lock together both held a stale `not
    yet`, and the second filed the same transition again (and, for a no-fit store, a second complete judgment outside the lock). Now every
    BoN write is taken under the lock with the run's state re-read there: (a) the store512 unfiled, two tasks released from the lock at the
    same instant file ONE `generated` + ONE run_patch per run and ONE certificate for the no-fit cell — the later task says the peer filed
    it and writes nothing; the store512's close afterwards retires the one run per cell once; (b) the store512 filed first, two tasks race
    on the supersede: ONE `superseded` per run, no `generated`."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    cfg = _bon_plan_config(tmp_path, goals=goals, n_max=N_MAX)
    nofit = _bon_store(wave, "prompted__safety_refusal", goals=goals, n_max=N_MAX, responded=0, refused=goals * N_MAX, config=cfg)   # every slot a typed no-fit
    assert nofit == stores["prompted__safety_refusal"]
    pa, pb = pcs; rids = {obj: _rid(pa, obj) for obj in OBJECTS}
    assert all(_rid(pb, obj) == rids[obj] for obj in OBJECTS)
    # (a) both tasks block at the lock the test holds, nothing filed; released, exactly one transition and one certificate per run
    held = _hold_store_locks(stores)
    logs = (tmp_path / "task_a.log", tmp_path / "task_b.log")
    procs = [_popen_close(pa, spool, wave, logs[0], CLIMB_NAME), _popen_close(pb, spool, wave, logs[1], CLIMB_NAME)]
    try:
        _wait_for(lambda: all("waiting for the store close lock" in lg.read_text() for lg in logs), 180, "both tasks to reach the store lock")
        assert all(p.poll() is None for p in procs), [lg.read_text() for lg in logs]
        assert _run_states(spool, list(rids.values())) == {rid: [] for rid in rids.values()} and all(_judgment_statuses(spool, rid) == [] for rid in rids.values())
        _release_store_by_store(held, stores, logs, procs)
    finally:
        for fh in held.values():
            if not fh.closed:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN); fh.close()
    for p in procs:
        assert p.wait(timeout=600) == 0, [lg.read_text() for lg in logs]
    outs = [lg.read_text() for lg in logs]
    for obj in OBJECTS:
        rid = rids[obj]
        assert len(_state(spool, rid, "generated")) == 1 and len(_patches(spool, rid)) == 1 and not _state(spool, rid, "superseded"), (obj, _run_states(spool, [rid]))
        assert sum(f"generated strongreject_bon {obj} {rid[:40]}…" in o for o in outs) == 1, outs
        assert sum(f"skip strongreject_bon {obj} {rid[:40]}… {_PEER_GENERATED}" in o for o in outs) == 1, outs
    assert _judgment_statuses(spool, rids["prompted__safety_refusal"]) == ["in_progress", "complete"] and _judgment_statuses(spool, rids["none"]) == []
    assert sum("no-fit-by-design (" in o for o in outs) == 1 and sum(_PEER_CERTIFIED in o for o in outs) == 1, outs
    assert sum("1 no-fit-by-design certificate(s)" in o for o in outs) == 1
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"2 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in out and "1 no-fit-by-design certificate(s)" in out, out
    for obj in OBJECTS:
        assert len(_state(spool, rids[obj], "superseded")) == 1 and len(_state(spool, rids[obj], "generated")) == 1
    _fold(root)
    _assert_one_current(root, _rid(p3, "none"), [rids["none"]])
    cur = _currency(root, [_rid(p3, "prompted__safety_refusal"), rids["prompted__safety_refusal"]])      # the no-fit cell: the store512 judged by its certificate, the climb retired
    assert cur[_rid(p3, "prompted__safety_refusal")][0] in ("judged", "ingested") and cur[_rid(p3, "prompted__safety_refusal")][1] == 1, cur
    assert cur[rids["prompted__safety_refusal"]][0] == "superseded" and cur[rids["prompted__safety_refusal"]][1] == 0, cur
    assert len(_superseded_events(root, rids["prompted__safety_refusal"])) == 1
    # (b) the store512 filed first: two tasks race on their shared run's supersede
    root2, spool2, wave2, q3, qcs, stores2, _g = _bon_world(tmp_path / "b", monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    qa, qb = qcs; rids2 = {obj: _rid(qa, obj) for obj in OBJECTS}
    r = _close(q3, spool2, wave2, job_name=S3_NAME); assert r.returncode == 0, r.stdout + r.stderr
    held = _hold_store_locks(stores2)
    logs2 = (tmp_path / "b_task_a.log", tmp_path / "b_task_b.log")
    procs = [_popen_close(qa, spool2, wave2, logs2[0], CLIMB_NAME), _popen_close(qb, spool2, wave2, logs2[1], CLIMB_NAME)]
    try:
        _wait_for(lambda: all("waiting for the store close lock" in lg.read_text() for lg in logs2), 180, "both tasks to reach the store lock")
        assert _run_states(spool2, list(rids2.values())) == {rid: [] for rid in rids2.values()}
        _release_store_by_store(held, stores2, logs2, procs)
    finally:
        for fh in held.values():
            if not fh.closed:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN); fh.close()
    for p in procs:
        assert p.wait(timeout=600) == 0, [lg.read_text() for lg in logs2]
    outs2 = [lg.read_text() for lg in logs2]
    for obj in OBJECTS:
        rid = rids2[obj]
        st = _state(spool2, rid, "superseded"); assert len(st) == 1 and st[0]["extra"]["superseded_by_run_id"] == _rid(q3, obj) and not _state(spool2, rid, "generated"), (obj, st)
        assert sum(f"superseded strongreject_bon {obj} {rid[:40]}…" in o for o in outs2) == 1 and sum(f"skip strongreject_bon {obj} {rid[:40]}… {_PEER_SUPERSEDED}" in o for o in outs2) == 1, outs2
    assert sorted(f"0 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in o for o in outs2) == [False, True], outs2   # one task filed both
    assert sorted("0 filed, 2 skipped, 0 refused of 2 registered run(s)" in o and D2599_LINE not in o for o in outs2) == [False, True], outs2      # the other wrote nothing
    _fold(root2)
    for obj in OBJECTS:
        _assert_one_current(root2, _rid(q3, obj), [rids2[obj]])


# ---------------------------------------------------------------------------------------------------------------------------------------------
# A late array-task failure reversed the shared climb run's retirement. The climb array's
# tasks register ONE run per cell (the job-independent key's companion), but every task still ends in its own EXIT trap, and common.sh's RESULT hook filed the
# TASK's `rc=N` as the RUN's `run_state failed` — which the reducer applies over `superseded` / `generated` (a lateral or terminal event has no
# rank). The closer's array-task mode (registry_close_stores.array_task_event, routed by common.sh for a task with SLURM_ARRAY_TASK_ID): a
# task's event never reverses a close of the shared run; the task's exit is recorded on the task; the last task to exit reconciles the array.
A_JOB = "810090"                                        # the climb array's job id (SLURM_ARRAY_JOB_ID); its tasks' pending files are CLIMBS[i], one job id each
_RULE = "a task's event never reverses a close of the shared run (round-2 Major on a9e75c210)"


def _all_states(spool, rids):
    """{run id: every run_state's state in record order} — the lifecycle events as the fold reads them (_run_states keeps generated / superseded alone)."""
    return {rid: [p["state"] for t, p, *_ in _records(spool, rid) if t == "run_state"] for rid in rids}


def _task_event(pending, spool, wave, event, *, task, tasks="0-1", sched=None, rc=None, job_name=CLIMB_NAME, extra=(), bare=False):
    """The closer's array-task mode as common.sh's f22_registry_array_task_event invokes it for one task's event."""
    argv = ["--array-task-event", event, "--array-job", A_JOB, "--array-task", str(task), "--array-tasks", tasks,
            *(["--scheduler-state", sched] if sched else []), *(["--rc", str(rc)] if rc is not None else []), *extra]
    return _close(pending, spool, wave, *argv, job_name=job_name, bare=bare)


def _outcomes(pending):
    d = pending.parent / f"{A_JOB}.tasks"
    return {int(p.stem): json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


def _who(task, job):
    return f"array {A_JOB} task {task} (job {job}, {CLIMB_NAME})"


def test_d456_a_late_task_failure_never_reverses_the_shared_climb_runs_retirement_and_the_records_fold_superseded(tmp_path, monkeypatch):
    """The sequence over the real registration and the real closer: the store512 closes first; task 0's close retires the shared
    climb run (its store facts, then `superseded`) and task 0 exits clean (recorded on the task; 1 of 2 exited, nothing reconciled); task 1
    then exits `rc=2` — the RESULT hook's `failed` (FAILED:rc=2). Before: `run_state failed` on the shared id, folded over the retirement
    (the negative control below pins it: emit_cli state failed over task 1's pending file, the hook's old call, folds the row to `failed`,
    not current, the retirement and its census effect gone). Now: a dry run says what it would do and writes nothing; the real event files
    NOTHING on the runs — a close stands — records task 1's exit on the task, and, the array complete, reconciles: the close's `superseded`
    stands by name. Folded by the real harvester: the climb rows superseded with the store512's facts, one current run per store, and
    admission's supersession resolves through the rows with the bytes and count verified. The same late failure after a task's `generated`
    (the store512 not yet closed) leaves the `generated` standing, and the store512's later close still retires the climb."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    pa, pb = pcs; rids = {obj: _rid(pa, obj) for obj in OBJECTS}; s3 = {obj: _rid(p3, obj) for obj in OBJECTS}
    assert all(_rid(pb, obj) == rids[obj] for obj in OBJECTS)
    r = _close(p3, spool, wave, job_name=S3_NAME); assert r.returncode == 0, r.stdout + r.stderr
    r = _close(pa, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"2 {D2599_LINE}" in out, out
    # task 0 exits clean after its close: recorded on the task, one of two exited, nothing reconciled yet
    r = _task_event(pa, spool, wave, "completed", task=0, sched="COMPLETED"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"recorded {_who(0, CLIMBS[0])}: exit completed at {pa.parent / (A_JOB + '.tasks') / '0.json'}" in out, out
    assert f"array {A_JOB}: 1 of 2 task(s) have exited; waiting on 1 — the last task to exit reconciles the shared runs" in out and "reconciling" not in out, out
    for obj in OBJECTS:
        assert f"note strongreject_bon {obj} {rids[obj][:40]}…: {_who(0, CLIMBS[0])} exited clean after its close ran; the run reads `superseded`; the exit is recorded on the task" in out, out
    before = _all_states(spool, list(rids.values()))
    # task 1 exits non-zero AFTER the retirement: the dry run names the decision and writes nothing (no record, no outcome file)
    r = _task_event(pb, spool, wave, "failed", task=1, sched="FAILED:rc=2", extra=("--dry-run",)); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("not filed strongreject_bon") == 2 and f"DRY-RUN {_who(1, CLIMBS[1])}: would record the task's exit at" in out, out
    assert "(DRY-RUN: nothing filed)" in out and sorted(_outcomes(pb)) == [0] and _all_states(spool, list(rids.values())) == before
    r = _task_event(pb, spool, wave, "failed", task=1, sched="FAILED:rc=2"); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    for obj in OBJECTS:
        rid = rids[obj]
        assert (f"not filed strongreject_bon {obj} {rid[:40]}…: `failed` for {_who(1, CLIMBS[1])} — a close of this run stands (`superseded`); {_RULE}; the task's failed is "
                f"recorded on the task ({pb.parent / (A_JOB + '.tasks')}/1.json) and the gate reads each task's shard through its coverage probe (under the store close lock)") in out, out
        assert f"stands strongreject_bon {obj} {rid[:40]}…: `superseded` by a close; 1 task(s) failed (1) — recorded on the tasks; the lifecycle is the close's (under the store close lock)" in out, out
        assert not _state(spool, rid, "failed") and _all_states(spool, [rid]) == {rid: before[rid]}, (obj, _all_states(spool, [rid]))
    assert f"array {A_JOB}: every task has exited (2 task(s): 0, 1; 1 failed) — reconciling 2 shared run(s)" in out, out
    oc = _outcomes(pb)
    assert sorted(oc) == [0, 1] and oc[1]["event"] == "failed" and oc[1]["rc"] == 2 and oc[1]["scheduler_state"] == "FAILED:rc=2" and oc[1]["job_id"] == CLIMBS[1] \
        and oc[1]["array_job_id"] == A_JOB and oc[1]["task"] == 1 and oc[0]["event"] == "completed" and oc[0]["rc"] is None and oc[0]["job_id"] == CLIMBS[0], oc
    assert {ru["run_id"] for ru in oc[1]["runs"]} == set(rids.values()) and all(ru["family"] == "bon" and ru["eval_id"] == "strongreject_bon" for ru in oc[1]["runs"]), oc
    # folded: the retirement stands — superseded rows with the store512's facts, one current run per store, admission through the rows
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, s3[obj], [rids[obj]])
    with DB.reading(DB.db_path(str(root))) as conn:
        for obj in OBJECTS:
            sup = TA.A.supersession(conn, rids[obj])
            assert sup["successor"] == s3[obj] and sup["duplicate_registration"] and "store_completion_sha256 and n_items_generated identical" in sup["source"], sup
    # the negative control: the RESULT hook's old filing — emit_cli state failed over task 1's pending file — over the same records, folded into a
    # second registry: the row reads `failed`, not current, and nothing stands in for the retirement (pinned)
    pre = tmp_path / "reg_pre"; _seeded_registry(pre)
    shutil.copytree(spool / "sdfbase_full22", pre / "records" / "sdfbase_full22", dirs_exist_ok=True)
    assert emit_cli.state("failed", pending_dir=str(pb.parent), job=CLIMBS[1], scheduler_state="FAILED:rc=2", final=True, spool=str(pre / "records")) == 0
    _fold(pre)
    cur = _currency(pre, list(rids.values()))
    with DB.reading(DB.db_path(str(pre))) as conn:
        for obj in OBJECTS:
            assert cur[rids[obj]][0] == "failed" and cur[rids[obj]][1] == 0, cur
            assert TA.A.registry_row(conn, rids[obj])["lifecycle_state"] == "failed"
    # the same late failure after a task's `generated` (the store512 has not closed): the generated stands, and the store512's close still retires it
    root2, spool2, wave2, q3, qcs, stores2, _g = _bon_world(tmp_path / "g", monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    qa, qb = qcs; rids2 = {obj: _rid(qa, obj) for obj in OBJECTS}
    r = _close(qa, spool2, wave2, job_name=CLIMB_NAME); assert r.returncode == 0 and "2 filed" in r.stdout + r.stderr, r.stdout + r.stderr
    assert _task_event(qa, spool2, wave2, "completed", task=0, sched="COMPLETED").returncode == 0
    r = _task_event(qb, spool2, wave2, "failed", task=1, sched="FAILED:rc=2"); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    for obj in OBJECTS:
        rid = rids2[obj]
        assert f"not filed strongreject_bon {obj} {rid[:40]}…: `failed` for {_who(1, CLIMBS[1])} — a close of this run stands (`generated`)" in out, out
        assert f"stands strongreject_bon {obj} {rid[:40]}…: `generated` by a close; 1 task(s) failed (1)" in out and not _state(spool2, rid, "failed"), out
    r = _close(q3, spool2, wave2, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"2 {D2599_LINE}" in out, out
    _fold(root2)
    for obj in OBJECTS:
        _assert_one_current(root2, _rid(q3, obj), [rids2[obj]])


def test_d456_the_last_array_task_to_exit_reconciles_the_shared_runs_and_a_late_start_never_files_running_over_a_close(tmp_path, monkeypatch):
    """The rule's other faces, over the real closer. (i) A task that STARTS after a peer's close (its job-start `running`, routed as `started`)
    files no `running` over the retirement — the spool is unchanged and the start is not an exit (no outcome file); a `started` before any close
    files `running` through the state emitter as before, under the store lock. (ii) A store no close can decide (its store manifest missing: no `units` accounting):
    task 1 fails FIRST — no close stands, so its `failed` is filed once, under the store lock; a requeued attempt of the same task files nothing
    twice and replaces its record; task 0's close REFUSES the store and the RESULT hook's refusal note states `running` (closer-fix-2), the
    lifecycle's last word; task 0's clean exit completes the array, and the reconciliation files `failed` once more — the first failing task's
    scheduler state — so the folded row reads `failed`, not `running`. (iii) A partial array reconciles nothing and says who is missing; a run
    no close decided and no task failed stays as its close left it. (iv) Refusals by name — a task outside the set, a malformed or empty set,
    a missing flag — file and record nothing. (v) The mode runs under the bare interpreter (`python -S`, the job's at start)."""
    # (i) a late start after the retirement
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    pa, pb = pcs; rids = {obj: _rid(pa, obj) for obj in OBJECTS}
    assert _close(p3, spool, wave, job_name=S3_NAME).returncode == 0 and _close(pa, spool, wave, job_name=CLIMB_NAME).returncode == 0
    before = _all_states(spool, list(rids.values()))
    r = _task_event(pb, spool, wave, "started", task=1, sched="RUNNING"); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    for obj in OBJECTS:
        assert (f"not filed strongreject_bon {obj} {rids[obj][:40]}…: `running` for {_who(1, CLIMBS[1])} — a close of this run stands (`superseded`); {_RULE}; "
                f"the task's start is on the job log alone (under the store close lock)") in out, out
    assert _all_states(spool, list(rids.values())) == before and _outcomes(pb) == {} and "recorded" not in out, out
    # a start before any close files `running` as before (the state emitter's own record), under the store lock
    root1, spool1, wave1, q3, qcs, stores1, _g = _bon_world(tmp_path / "s", monkeypatch, climbs=1, climb_key=CLIMB_KEY)
    qa = qcs[0]; rid1 = _rid(qa, "none"); n0 = len(_state(spool1, rid1, "running"))
    r = _task_event(qa, spool1, wave1, "started", task=0, tasks="0-0", sched="RUNNING"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"running strongreject_bon none {rid1[:40]}…: filed for {_who(0, CLIMBS[0])} — no close has decided this run (under the store close lock)" in out, out
    st = _state(spool1, rid1, "running"); assert len(st) == n0 + 1 and st[-1]["scheduler_state"] == "RUNNING", st
    # (ii) a store no close can decide: task 1 fails first, task 0's close refuses, the last exit reconciles to `failed`
    root3, spool3, wave3, s3p, qcs3, stores3, _g = _bon_world(tmp_path / "i", monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    for obj in OBJECTS:
        (stores3[obj] / "bon_served_manifest.json").unlink()                      # inspect_bon: no `units` accounting in the store manifest — refused, never generated
    pa3, pb3 = qcs3; rids3 = {obj: _rid(pa3, obj) for obj in OBJECTS}
    r = _task_event(pb3, spool3, wave3, "failed", task=1, sched="FAILED:rc=2"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"array {A_JOB}: 1 of 2 task(s) have exited; waiting on 0" in out, out
    for obj in OBJECTS:
        rid = rids3[obj]
        assert f"failed strongreject_bon {obj} {rid[:40]}…: filed for {_who(1, CLIMBS[1])} (FAILED:rc=2; under the store close lock) — no close has decided this run" in out, out
        st = _state(spool3, rid, "failed"); assert len(st) == 1 and st[0]["scheduler_state"] == "FAILED:rc=2" and st[0]["evidence_kind"] == "scheduler_state" \
            and f"{_who(1, CLIMBS[1])} exited failed rc=2 (FAILED:rc=2)" in st[0]["note"] and _RULE in st[0]["note"], st
    r = _task_event(pb3, spool3, wave3, "failed", task=1, sched="FAILED:rc=1"); out = r.stdout + r.stderr   # a requeued attempt of task 1
    assert r.returncode == 0 and out.count("nothing filed twice") == 2 and "(replacing an earlier record of this task — a requeued attempt)" in out, out
    assert all(len(_state(spool3, rids3[obj], "failed")) == 1 for obj in OBJECTS) and _outcomes(pb3)[1]["rc"] == 1 and _outcomes(pb3)[1]["scheduler_state"] == "FAILED:rc=1"
    r = _close(pa3, spool3, wave3, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 2 and "2 refused of 2 registered run(s)" in out and "completeness cannot be certified — the run stays running" in out, out
    # the RESULT hook's refusal note (f22_registry_closer_refused -> the closer's refusal-note mode since the round-3 Major on cb9ad5267): a task's
    # `failed` is the run's last word, so the note is NOT filed over it — nothing changes on the runs
    r = _refusal_note(pa3, spool3, wave3); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("`running` (the closer's refusal note) — the run reads `failed` (a task's exit)") == 2 \
        and f"state running: stated for 0 refused run(s) of job {CLIMBS[0]} (2 listed; 2 not filed — a close or a task's exit stands" in out, out
    for obj in OBJECTS:
        assert _all_states(spool3, [rids3[obj]])[rids3[obj]][-1] == "failed" and len(_state(spool3, rids3[obj], "running")) == 0
        # an OLDER hook's note — the bare state emitter by --run-id, before the fence — left `running` as the run's last word; the reconciliation
        # below is what corrects such a row (the branch this part of the test pins)
        assert emit_cli.state("running", run_id=rids3[obj], job=CLIMBS[0], note="closer REFUSED rc=2 under python3: no end-of-run marker", spool=str(spool3)) == 0
        assert _all_states(spool3, [rids3[obj]])[rids3[obj]][-1] == "running"
    r = _task_event(pa3, spool3, wave3, "completed", task=0, sched="COMPLETED"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"array {A_JOB}: every task has exited (2 task(s): 0, 1; 1 failed) — reconciling 2 shared run(s)" in out, out
    for obj in OBJECTS:
        rid = rids3[obj]
        assert (f"failed strongreject_bon {obj} {rid[:40]}…: filed by the array's reconciliation (FAILED:rc=1; task 1 rc=1; 1 failed task(s)) — no close decided this run; "
                f"it read `running` (under the store close lock)") in out, out
        st = _state(spool3, rid, "failed"); assert len(st) == 2 and st[-1]["scheduler_state"] == "FAILED:rc=1" and "task 1 failed rc=1" in st[-1]["note"], st
        assert _all_states(spool3, [rid])[rid][-1] == "failed"
    _fold(root3)
    cur = _currency(root3, list(rids3.values()))
    assert all(cur[rid][0] == "failed" and cur[rid][1] == 0 for rid in rids3.values()), cur
    # (iii) a partial array reconciles nothing and names the missing task; a run no close decided and no task failed stays as it is
    root4, spool4, wave4, s4, qcs4, stores4, _g = _bon_world(tmp_path / "p", monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    pa4, pb4 = qcs4; rids4 = {obj: _rid(pa4, obj) for obj in OBJECTS}
    r = _task_event(pa4, spool4, wave4, "completed", task=0, tasks="0-2", sched="COMPLETED"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"array {A_JOB}: 1 of 3 task(s) have exited; waiting on 1, 2" in out and "reconciling" not in out, out
    r = _task_event(pb4, spool4, wave4, "completed", task=1, tasks="0-2", sched="COMPLETED"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"array {A_JOB}: 2 of 3 task(s) have exited; waiting on 2" in out and "reconciling" not in out, out
    (pb4.parent / f"{A_JOB}.tasks" / "2.json").write_text(json.dumps({**_outcomes(pb4)[1], "task": 2, "job_id": "810099"}))   # the third task's exit, clean
    r = _task_event(pb4, spool4, wave4, "completed", task=1, tasks="0-2", sched="COMPLETED"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "every task has exited (3 task(s): 0, 1, 2; 0 failed)" in out, out
    for obj in OBJECTS:
        assert f"stays strongreject_bon {obj} {rids4[obj][:40]}…: no close decided this run and no task failed — it reads `unfiled` as its close left it" in out, out
    assert _all_states(spool4, list(rids4.values())) == {rid: [] for rid in rids4.values()}
    # (iv) refusals file and record nothing
    for argv, why in ((["--array-task-event", "failed", "--array-job", A_JOB, "--array-task", "7", "--array-tasks", "0-1"], "array task 7 is not in the array's task set '0-1'"),
                      (["--array-task-event", "failed", "--array-job", A_JOB, "--array-task", "0", "--array-tasks", ""], "names no task"),
                      (["--array-task-event", "failed", "--array-job", A_JOB, "--array-task", "0", "--array-tasks", "3-0"], "is not an ascending range"),
                      (["--array-task-event", "failed", "--array-job", "", "--array-task", "0", "--array-tasks", "0-1"], "--array-job is empty"),
                      (["--array-task-event", "failed", "--array-task", "0", "--array-tasks", "0-1"], "needs --array-job, --array-task and --array-tasks")):
        r = _close(pa4, spool4, wave4, *argv, job_name=CLIMB_NAME); out = r.stdout + r.stderr
        assert r.returncode == 2 and "REFUSED" in out and why in out and "nothing filed" in out, (argv, out)
    assert _all_states(spool4, list(rids4.values())) == {rid: [] for rid in rids4.values()} and sorted(_outcomes(pa4)) == [0, 1, 2]
    # (v) the bare interpreter (the job's python3 at start, `-S`) runs the mode
    r = _task_event(pa4, spool4, wave4, "failed", task=0, tasks="0-2", sched="FAILED:rc=2", extra=("--dry-run",), bare=True); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("would file run_state failed (FAILED:rc=2) for array") == 2 and "(DRY-RUN: nothing filed)" in out, out
    assert out.count("1 task(s) failed, no close decided this run") == 2 and sorted(_outcomes(pa4)) == [0, 1, 2] and _outcomes(pa4)[0]["event"] == "completed", out


# ---------------------------------------------------------- a REFUSED climb close could undo a peer's supersession
# The RESULT hook's refusal note (`running` for the runs the closer refused) was filed through the bare state emitter, per run id,
# AFTER the closer had released the store lock. A climb task whose close refused an incomplete store, a peer task then closing the completed
# store and the store512 retiring the shared run, and this task's hook stating `running` LAST folded the retired run `running` and current again
# (the reducer applies `running` over a lateral `superseded`). Now the note goes through the closer's refusal-note mode (--refusal-note): per
# listed run, under the store close lock, filed only when no close has decided the run and its last word is not a task's exit.
_REFUSAL_RULE = "a refused close's `running` note never reverses a close of the run, nor a task's exit (round-3 Major on cb9ad5267)"
_NOTE = "closer REFUSED rc=2 under python3: REFUSED strongreject_bon none x: undecided slots"
_NOT_FILED_OVER_CLOSE = ("`running` (the closer's refusal note) — a close of this run stands (`{ct}`); " + _REFUSAL_RULE
                         + "; the refusal is on the job log, the RESULT surface and the reclose recipe (under the store close lock)")


def _refusal_note(pending, spool, wave, *, note=_NOTE, extra=(), wave_root=True, log=None):
    """The closer's refusal-note mode as common.sh's f22_registry_closer_refused invokes it: `--refusal-note` over the pending file's refused-run
    list, the node, the wave root and tag the store lock is named from. `log` runs it detached (Popen, its lines streaming to the file) for the
    interleaving under a lock the test holds."""
    env = dict(os.environ, PYTHONPATH=str(REPO / "src")); env.pop("SISTEER_REGISTRY_DISABLED", None)
    argv = [sys.executable, str(CLOSER), "--pending", str(pending), "--spool", str(spool), "--code-sha", synth.code_sha("b"), "--refusal-note", note,
            "--node", "TESTHOST", "--job-name", CLIMB_NAME, *(["--wave-root", str(wave), "--tag", TAG] if wave_root else []), *extra]
    if log is not None:
        return subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, text=True)
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)


def _hook_refusal(tmp_path, pending, spool, wave, job, *, why="REFUSED strongreject_bon none x: undecided slots"):
    """common.sh's f22_registry_closer_refused — the RESULT hook's refusal path, the real shell function — for `job`, whose pending file and
    refused-run list are copied into $SHARED/registry/pending as the job left them; F22_PY is this interpreter (the driver venv the trap runs
    under). Returns (rc, the merged output)."""
    shared = tmp_path / f"shared_{job}"; pdir = shared / "registry" / "pending"; pdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pending, pdir / f"{job}.json")
    side = Path(str(pending) + ".refused_run_ids.json")
    (pdir / f"{job}.json.refused_run_ids.json").unlink(missing_ok=True)
    if side.exists():
        shutil.copy(side, pdir / f"{job}.json.refused_run_ids.json")
    figs = shared / "figs"; figs.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SISTEER_REGISTRY_DISABLED")}
    env.update({"SHARED": str(shared), "SLURM_JOB_ID": job, "SLURM_JOB_NAME": CLIMB_NAME, "SLURMD_NODENAME": "TESTHOST", "REPO_DIR": str(REPO),
                "SISTEER_REGISTRY_SPOOL": str(spool), "FIGS_DIR": str(figs), "WAVE_SHA": synth.code_sha("b"), "TAG": TAG, "WAVE_ROOT": str(wave), "F22_PY": sys.executable})
    cmd = ["bash", "-c", f"source {REPO}/scripts/sdfbase/full22/common.sh && f22_registry_closer_refused 2 python3 '{why}'"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=600)
    return r.returncode, r.stdout + r.stderr


def _store_config(stores):
    return Path(json.loads((stores[OBJECTS[0]] / "bon_served_run.json").read_text())["config"])


def _peer_close_records(spool, stores, s3_rids, climb_rids, n_items):
    """What the store512's close and a peer task's close file for one store, written by the holder of the store lock: the store512's `generated`
    (an older-closer filing's shape), the climb's inspected store facts and its `superseded` by the store512 (file_superseded, the closer's own
    writer)."""
    RCS = _closer_module(); producer = emit.producer_block("peer_close_under_the_lock", code_sha="c" * 40)
    for obj in OBJECTS:
        art = str(stores[obj] / "bon_served_manifest.json")
        _older_closer_filing(spool, s3_rids[obj], stores[obj], n_items=n_items, note="the store512's close (synthetic; filed under the lock the test holds)")
        w = emit.RecordWriter(str(spool), "sdfbase_full22", climb_rids[obj], producer, push_script=None)
        w.emit_patch(fields={"store_uri": str(stores[obj]), "store_completion_artifact_uri": art, "store_completion_sha256": hashlib.sha256(Path(art).read_bytes()).hexdigest(),
                             "n_items_generated": n_items})
        w.push()
        RCS.file_superseded(spool=str(spool), push_script=None, campaign="sdfbase_full22", rid=climb_rids[obj], by_rid=s3_rids[obj], artifact=art, producer=producer,
                            sibling_job=S3, eval_id="strongreject_bon", obj=obj, dry_run=False, decision=RCS.BON_SUPERSEDED_DECISION, rule=RCS.BON_SUPERSEDED_RULE,
                            kind=RCS.BON_SUPERSEDED_KIND, extra={"superseded_role": "climb", "survivor_role": "store512"})


def test_d456_a_refused_climb_close_never_files_its_running_note_over_a_peers_supersession_and_the_records_fold_superseded(tmp_path, monkeypatch):
    """Over the real registration, the real closer and the real RESULT hook. (1) The stores are incomplete
    when task 0 closes: REFUSED by name (the inspection runs outside the store lock), the shared run ids listed, nothing filed. (2) The stores
    complete; the store512 closes `generated` and task 1's close retires the shared run by it — before task 0's hook has run. (3) Task 0's
    hook — common.sh's f22_registry_closer_refused, the real function — files NOTHING: each run's close is named, the bare state emitter is
    never called, the spool is unchanged; folded by the real harvester, the retirement stands (one current run per store, admission through
    the rows). The NEGATIVE CONTROL folds the hook's OLD filing — the bare state emitter by --run-id, the note, no lock, no check — over the
    same records into a second registry: the retired duplicate reads `running` and is CURRENT again (two current runs per store), the
    retirement undone by the task that closed nothing (the reversal, pinned). (4) THE INTERLEAVING UNDER THE LOCK, deterministic: a fresh
    world; task 0's close refuses; the test holds both stores' close locks (a close in flight) and starts task 0's refusal note, which blocks
    at the lock with nothing filed; while it waits, the lock holder completes the stores and files what the store512's and task 1's closes
    file; released, the note reads the close and files nothing — the decision is provably after the peer's filing, serialised by the lock."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    pa, pb = pcs; rids = {obj: _rid(pa, obj) for obj in OBJECTS}; s3 = {obj: _rid(p3, obj) for obj in OBJECTS}; cfg = _store_config(stores)
    assert all(_rid(pb, obj) == rids[obj] for obj in OBJECTS)
    # (1) incomplete stores: task 0's close refuses by name and lists the shared runs
    for obj in OBJECTS:
        _bon_store(wave, obj, goals=goals, n_max=N_MAX, responded=goals * N_MAX - 2, refused=0, config=cfg)
    r = _close(pa, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 2 and "2 refused of 2 registered run(s)" in out and out.count("— the run stays running") == 2, out
    side = Path(str(pa) + ".refused_run_ids.json")
    assert sorted(json.loads(side.read_text())["refused_run_ids"]) == sorted(rids.values()) and json.loads(side.read_text())["job_id"] == CLIMBS[0]
    assert _all_states(spool, list(rids.values())) == {rid: [] for rid in rids.values()}
    # (2) the stores complete; the store512 closes and task 1 retires the shared run by it
    for obj in OBJECTS:
        _bon_store(wave, obj, goals=goals, n_max=N_MAX, responded=goals * N_MAX, refused=0, config=cfg)
    r = _close(p3, spool, wave, job_name=S3_NAME); assert r.returncode == 0 and "2 filed" in r.stdout + r.stderr, r.stdout + r.stderr
    r = _close(pb, spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"2 {D2599_LINE}" in out, out
    before = _all_states(spool, list(rids.values()))
    assert all(before[rid][-1] == "superseded" for rid in rids.values()), before
    # (3) task 0's RESULT hook, the real shell function over task 0's pending file and refused-run list: nothing filed, each close named
    rc, out = _hook_refusal(tmp_path, pa, spool, wave, CLIMBS[0])
    assert rc == 0, out
    for obj in OBJECTS:
        assert f"[registry] not filed strongreject_bon {obj} {rids[obj][:40]}…: " + _NOT_FILED_OVER_CLOSE.format(ct="superseded") in out, out
    assert f"[registry] state running: stated for 0 refused run(s) of job {CLIMBS[0]} (2 listed; 2 not filed — a close or a task's exit stands; the refused-run list; D2100 (b))" in out, out
    assert (f"[registry] closer refusal note: 2 listed refused run(s) of job {CLIMBS[0]} decided by the closer's refusal-note mode under {sys.executable} (the refused-run list; "
            f"D2100 (b)) — `running` filed per run under its store lock, never over a close") in out, out
    assert "emit_cli state" not in out and "state running: wrote" not in out and "--run-id" not in out, out
    assert _all_states(spool, list(rids.values())) == before
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, s3[obj], [rids[obj]])
    with DB.reading(DB.db_path(str(root))) as conn:
        for obj in OBJECTS:
            sup = TA.A.supersession(conn, rids[obj])
            assert sup["successor"] == s3[obj] and sup["duplicate_registration"] and "store_completion_sha256 and n_items_generated identical" in sup["source"], sup
    # the NEGATIVE CONTROL: the hook's old filing over the same records, folded into a second registry — the retired duplicate is current again
    pre = tmp_path / "reg_pre"; _seeded_registry(pre)
    shutil.copytree(spool / "sdfbase_full22", pre / "records" / "sdfbase_full22", dirs_exist_ok=True)
    for obj in OBJECTS:
        assert emit_cli.state("running", run_id=rids[obj], job=CLIMBS[0], node="TESTHOST", evidence_kind="job_log", note=_NOTE, spool=str(pre / "records")) == 0
    _fold(pre)
    cur = _currency(pre, [*rids.values(), *s3.values()])
    for obj in OBJECTS:
        assert cur[rids[obj]] == ("running", 1, None), cur                       # lifecycle `running`, is_current 1, no reason: the retirement undone
        assert cur[s3[obj]][1] == 1
    assert sum(1 for v in cur.values() if v[1] == 1) == 4                        # two current runs per store — the duplicate registration the census double-counts
    with DB.reading(DB.db_path(str(pre))) as conn:
        for obj in OBJECTS:
            assert TA.A.registry_row(conn, rids[obj])["lifecycle_state"] == "running"
    # (4) the interleaving under the lock: the note starts BEFORE the peer's filing and still files nothing
    root4, spool4, wave4, q3, qcs, stores4, g4 = _bon_world(tmp_path / "i", monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    qa, _qb = qcs; rids4 = {obj: _rid(qa, obj) for obj in OBJECTS}; s34 = {obj: _rid(q3, obj) for obj in OBJECTS}; cfg4 = _store_config(stores4)
    for obj in OBJECTS:
        _bon_store(wave4, obj, goals=g4, n_max=N_MAX, responded=g4 * N_MAX - 1, refused=0, config=cfg4)
    r = _close(qa, spool4, wave4, job_name=CLIMB_NAME); assert r.returncode == 2 and "2 refused" in r.stdout + r.stderr, r.stdout + r.stderr
    held = _hold_store_locks(stores4)
    log = tmp_path / "refusal_note.log"; proc = _refusal_note(qa, spool4, wave4, log=log)
    try:
        _wait_for(lambda: "waiting for the store close lock" in log.read_text(), 180, "the refusal note to reach the store lock")
        time.sleep(1.0)
        assert proc.poll() is None and _all_states(spool4, list(rids4.values())) == {rid: [] for rid in rids4.values()}, log.read_text()   # nothing filed outside the lock
        for obj in OBJECTS:                                                       # the peer's close, under the lock the test holds
            _bon_store(wave4, obj, goals=g4, n_max=N_MAX, responded=g4 * N_MAX, refused=0, config=cfg4)
        _peer_close_records(spool4, stores4, s34, rids4, g4)
        assert all(_all_states(spool4, [rid])[rid] == ["superseded"] for rid in rids4.values())
        time.sleep(0.5)
        assert proc.poll() is None and "not filed" not in log.read_text() and "running strongreject_bon" not in log.read_text(), log.read_text()   # still waiting
    finally:
        for fh in held.values():
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN); fh.close()
    assert proc.wait(timeout=600) == 0, log.read_text()
    out = log.read_text()
    for obj in OBJECTS:
        assert f"not filed strongreject_bon {obj} {rids4[obj][:40]}…: " + _NOT_FILED_OVER_CLOSE.format(ct="superseded") in out, out
    assert f"state running: stated for 0 refused run(s) of job {CLIMBS[0]} (2 listed; 2 not filed — a close or a task's exit stands" in out, out
    assert all(_all_states(spool4, [rid])[rid] == ["superseded"] for rid in rids4.values())
    _fold(root4)
    for obj in OBJECTS:
        _assert_one_current(root4, s34[obj], [rids4[obj]])


def test_d456_the_refusal_note_is_filed_under_the_store_lock_when_no_close_stands_and_refused_by_name_without_the_lock(tmp_path, monkeypatch):
    """The refusal note's other faces. (i) No close stands: the dry run says what it would file and writes nothing; the note is filed `running`
    under the store lock through the state emitter — the hook's record shape (evidence job_log, the note, the node and job on the patch), the
    reclose's recovery as before — through the mode and through the real hook. (ii) A listed run the pending file does not register is never
    stated for. (iii) A BoN run whose store lock cannot be named (no --wave-root/--tag) is refused by name, nothing filed, rc 2 — the array
    task's `started` the same. (iv) A peer task's `failed` is the run's last word: the note is not filed over a task's exit. (v) A list naming
    another job refuses; no list states nothing, loudly; a store with no directory is decided under its close lock all the same — the lock
    lives beside the store (round-4 Major on 08c90e1cf). (vi) The mode runs under the bare interpreter (`python -S`)."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    pa, pb = pcs; rids = {obj: _rid(pa, obj) for obj in OBJECTS}; cfg = _store_config(stores)
    for obj in OBJECTS:
        _bon_store(wave, obj, goals=goals, n_max=N_MAX, responded=goals * N_MAX - 2, refused=0, config=cfg)
    r = _close(pa, spool, wave, job_name=CLIMB_NAME); assert r.returncode == 2 and "2 refused" in r.stdout + r.stderr, r.stdout + r.stderr
    side = Path(str(pa) + ".refused_run_ids.json"); listed = json.loads(side.read_text())
    # (i) no close stands: dry run, then the note filed under the store lock
    r = _refusal_note(pa, spool, wave, extra=("--dry-run",)); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("would file run_state running with the closer's refusal note — no close has decided this run (under the store close lock)") == 2, out
    assert f"state running: stated for 2 refused run(s) of job {CLIMBS[0]} (2 listed; the refused-run list; D2100 (b)) (DRY-RUN: nothing filed)" in out, out
    assert _all_states(spool, list(rids.values())) == {rid: [] for rid in rids.values()}
    r = _refusal_note(pa, spool, wave); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"state running: stated for 2 refused run(s) of job {CLIMBS[0]} (2 listed; the refused-run list; D2100 (b))" in out, out
    for obj in OBJECTS:
        rid = rids[obj]
        assert f"running strongreject_bon {obj} {rid[:40]}…: the closer's refusal note filed — no close has decided this run (under the store close lock)" in out, out
        st = _state(spool, rid, "running"); assert len(st) == 1 and st[0]["note"].startswith(_NOTE) and st[0]["evidence_kind"] == "job_log", st
        pa_ = [p["fields"] for t, p, _pr in _records(spool, rid) if t == "run_patch" and "node_hostname" in (p.get("fields") or {})]
        assert pa_ and pa_[-1]["node_hostname"] == "TESTHOST" and pa_[-1]["cluster_job_id"] == CLIMBS[0], pa_
        assert _all_states(spool, [rid])[rid] == ["running"]
    # the real hook files the same (a second note after the first: the latest word is the reclose's recovery, as before)
    rc, out = _hook_refusal(tmp_path, pa, spool, wave, CLIMBS[0])
    assert rc == 0 and f"[registry] state running: stated for 2 refused run(s) of job {CLIMBS[0]} (2 listed; the refused-run list; D2100 (b))" in out, out
    assert f"closer refusal note: 2 listed refused run(s) of job {CLIMBS[0]} decided by the closer's refusal-note mode under {sys.executable}" in out and "emit_cli state" not in out, out
    assert all(len(_state(spool, rid, "running")) == 2 and _all_states(spool, [rid])[rid] == ["running", "running"] for rid in rids.values())
    # (ii) a listed run this pending file does not register: never stated for (no record, no run directory)
    foreign = "sdfbase_full22:bon:1111111111111111:2222222222222222"
    side.write_text(json.dumps({**listed, "refused_run_ids": [foreign, *listed["refused_run_ids"]]}))
    run_dirs = sorted(os.listdir(spool / "sdfbase_full22"))
    r = _refusal_note(pa, spool, wave); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"not filed {foreign[:40]}…: the refused-run list names a run {pa} does not register — stated for this job's registered runs alone (D2100 (b))" in out, out
    assert f"stated for 2 refused run(s) of job {CLIMBS[0]} (3 listed; 1 not registered by this pending file; the refused-run list; D2100 (b))" in out, out
    assert all(len(_state(spool, rid, "running")) == 3 for rid in rids.values()) and sorted(os.listdir(spool / "sdfbase_full22")) == run_dirs
    side.write_text(json.dumps(listed))
    # (iii) no --wave-root/--tag: a BoN run's lock cannot be named — refused by name, nothing filed; the array task's `started` the same
    before = _all_states(spool, list(rids.values()))
    r = _refusal_note(pa, spool, wave, wave_root=False); out = r.stdout + r.stderr
    assert r.returncode == 2 and out.count("the closer's refusal note — a BoN run's store close lock cannot be named (no --wave-root/--tag); a BoN run's lifecycle is decided under its "
                                           "store close lock alone; nothing filed") == 2 and f"stated for 0 refused run(s) of job {CLIMBS[0]} (2 listed" in out, out
    env = dict(os.environ, PYTHONPATH=str(REPO / "src")); env.pop("SISTEER_REGISTRY_DISABLED", None)
    r = subprocess.run([sys.executable, str(CLOSER), "--pending", str(pb), "--spool", str(spool), "--array-task-event", "started", "--array-job", A_JOB, "--array-task", "1",
                        "--array-tasks", "0-1", "--scheduler-state", "RUNNING", "--job-name", CLIMB_NAME], capture_output=True, text=True, env=env, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and out.count(f"`running` for {_who(1, CLIMBS[1])} — a BoN run's store close lock cannot be named (no --wave-root/--tag); a BoN run's lifecycle is decided "
                                           "under its store close lock alone; nothing filed") == 2, out
    assert _all_states(spool, list(rids.values())) == before and _outcomes(pb) == {}
    # (iv) a peer task's `failed` is the run's last word: the note is not filed over a task's exit
    r = _task_event(pb, spool, wave, "failed", task=1, sched="FAILED:rc=2"); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("— no close has decided this run") == 2 and all(_all_states(spool, [rid])[rid][-1] == "failed" for rid in rids.values()), out
    r = _refusal_note(pa, spool, wave); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count(f"`running` (the closer's refusal note) — the run reads `failed` (a task's exit); {_REFUSAL_RULE} (under the store close lock)") == 2, out
    assert f"stated for 0 refused run(s) of job {CLIMBS[0]} (2 listed; 2 not filed — a close or a task's exit stands; the refused-run list; D2100 (b))" in out, out
    assert all(_all_states(spool, [rid])[rid][-1] == "failed" and len(_state(spool, rid, "running")) == 3 for rid in rids.values())
    # (v) a list naming another job refuses; no list states nothing, loudly; a store with no directory is decided under its lock BESIDE the absent store
    side.write_text(json.dumps({**listed, "job_id": "999999"}))
    r = _refusal_note(pa, spool, wave); out = r.stdout + r.stderr
    assert r.returncode == 2 and f"REFUSED: the refused-run list {side} names job '999999', not this pending file's '{CLIMBS[0]}'; no refusal note filed" in out, out
    side.unlink()
    r = _refusal_note(pa, spool, wave); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"no refused-run list at {side} — stating nothing (D2100 (b): never the job's whole run set)" in out and "stated for" not in out, out
    rc, out = _hook_refusal(tmp_path, pa, spool, wave, CLIMBS[0])
    assert rc == 0 and "no refused-run list at" in out and "stating nothing (D2100 (b)" in out and "registry_close_stores" not in out, out
    root2, spool2, wave2, _q3, qcs2, stores2, _g = _bon_world(tmp_path / "n", monkeypatch, climbs=1, climb_key=CLIMB_KEY)
    qa2 = qcs2[0]; rid2 = _rid(qa2, OBJECTS[0])
    for obj in OBJECTS:
        shutil.rmtree(stores2[obj])
    r = _close(qa2, spool2, wave2, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 2 and out.count("no store directory — the run stays running") == 2, out
    assert not _lock_of(stores2[OBJECTS[0]]).exists()
    r = _refusal_note(qa2, spool2, wave2); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"running strongreject_bon {OBJECTS[0]} {rid2[:40]}…: the closer's refusal note filed — no close has decided this run (under the store close lock)" in out, out
    assert "without a store lock" not in out and len(_state(spool2, rid2, "running")) == 1
    assert _lock_of(stores2[OBJECTS[0]]).is_file() and not (stores2[OBJECTS[0]]).exists()        # the lock beside the store; the store still absent
    # (vi) the bare interpreter runs the mode
    r = subprocess.run([sys.executable, "-S", str(CLOSER), "--pending", str(qa2), "--spool", str(spool2), "--wave-root", str(wave2), "--tag", TAG, "--refusal-note", _NOTE,
                        "--node", "TESTHOST", "--dry-run"], capture_output=True, text=True, env=env, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("would file run_state running with the closer's refusal note") == 2 and "(DRY-RUN: nothing filed)" in out, out


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Round-4 Major on 08c90e1cf: a missing store bypassed the shared run's close lock. _run_lock decided a BoN run's `started` and the
# refusal note WITHOUT a lock when the store directory did not exist ("no close of it can be in flight") — but the directory's absence was read
# once, and the write came later: the state emitter captures hardware between the run's close_terminal_of read and its emit_state (nvidia-smi,
# then sacct, up to 10 s), so a store created, completed and closed in that interval landed its `generated` / `superseded` BEFORE an unlocked
# `running`, which the reducer applies over the lateral `superseded` — the retired duplicate current again. Now the store's close lock lives
# BESIDE the store (<private run>/.<cell>.close.lock, bon_close_lock_path), a path the lock taker creates, so every close and every lifecycle
# writer of a BoN run take ONE lock whether or not the store exists yet; _run_lock has no unlocked branch for a BoN run.


def _lifecycle(argv_tail, *, pending, spool, wave, bin_dir=None, log=None):
    """The closer's lifecycle modes (an array task's event, the refusal note) as common.sh invokes them, with `bin_dir` first on PATH and the
    GPU-count ladder scrubbed, so the state emitter's hardware capture reaches the fakes there (nvidia-smi, then sacct)."""
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    for k in ("SISTEER_REGISTRY_DISABLED", *emit.GPU_COUNT_ENV_LADDER):
        env.pop(k, None)
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    argv = [sys.executable, str(CLOSER), "--pending", str(pending), "--wave-root", str(wave), "--tag", TAG, "--spool", str(spool), "--code-sha", synth.code_sha("b"),
            "--job-name", CLIMB_NAME, *argv_tail]
    if log is not None:
        return subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, text=True)
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)


_PEER_IN_THE_PROBE = r'''
import hashlib, importlib.util, json, os, sys
from pathlib import Path
args = json.load(open(sys.argv[1]))
_out = sys.stdout; sys.stdout = sys.stderr                                 # the capture parses stdout's FIRST line: the TRES line alone goes there
sys.path.insert(0, args["tests"]); sys.path.insert(0, args["src"])
from test_registry_close_stores import _bon_store                      # the store as bon_served_run.py leaves it
from sisteer.registry import emit
spec = importlib.util.spec_from_file_location("rcs_peer", args["closer"]); RCS = importlib.util.module_from_spec(spec); spec.loader.exec_module(RCS)
calls = Path(args["calls"]); n = len(calls.read_text().splitlines()) if calls.exists() else 0
with open(calls, "a") as fh:
    fh.write(f"{os.getpid()}\n")
if n < len(args["order"]):
    obj = args["order"][n]; goals, n_max = int(args["goals"]), int(args["n_max"])
    store = _bon_store(Path(args["wave"]), obj, goals=goals, n_max=n_max, responded=goals * n_max, refused=0, config=Path(args["config"]))   # the store appears, complete
    art = str(store / "bon_served_manifest.json"); sha = hashlib.sha256(Path(art).read_bytes()).hexdigest()
    producer = emit.producer_block("peer_close_in_the_probe", code_sha="c" * 40)
    lock = RCS.StoreCloseLock(str(store), wait_s=float(args["wait_s"]))      # the lock as the closer takes it, a short wait
    try:
        with lock:
            w3 = emit.RecordWriter(args["spool"], "sdfbase_full22", args["s3"][obj], producer, push_script=None)
            w3.emit_patch(fields={"store_uri": str(store), "store_completion_artifact_uri": art, "store_completion_sha256": sha, "n_items_generated": goals,
                                  "generated_utc": "2026-10-12T00:00:00.000Z"})
            w3.emit_state("generated", "completion_artifact", art, note="the store512's close (synthetic; the peer in the probe)"); w3.push(final=True)
            wc = emit.RecordWriter(args["spool"], "sdfbase_full22", args["climb"][obj], producer, push_script=None)
            wc.emit_patch(fields={"store_uri": str(store), "store_completion_artifact_uri": art, "store_completion_sha256": sha, "n_items_generated": goals}); wc.push()
            RCS.file_superseded(spool=args["spool"], push_script=None, campaign="sdfbase_full22", rid=args["climb"][obj], by_rid=args["s3"][obj], artifact=art,
                                producer=producer, sibling_job=args["s3_job"], eval_id="strongreject_bon", obj=obj, dry_run=False, decision=RCS.BON_SUPERSEDED_DECISION,
                                rule=RCS.BON_SUPERSEDED_RULE, kind=RCS.BON_SUPERSEDED_KIND, extra={"superseded_role": "climb", "survivor_role": "store512"})
        line = f"FILED {obj}: the peer's close landed under {lock.path}"
    except RCS.StoreCloseLockBusy as exc:
        line = f"REFUSED {obj}: {exc}"
    with open(args["log"], "a") as fh:
        fh.write(line + "\n")
_out.write("billing=1,cpu=16,gres/gpu=1,gres/gpu:h200=1,mem=128G,node=1\n"); _out.flush()
'''


def _peer_in_the_probe(where: Path, *, wave, spool, order, s3_rids, climb_rids, goals, cfg, wait_s=2.0):
    """A bin directory whose `nvidia-smi` fails and whose `sacct` IS THE PEER CLOSE. emit_cli.state captures hardware between a run's
    close_terminal_of read and its `running` write (emit.hardware_from_env: the env ladder, nvidia-smi, then sacct when a fact is still
    missing), so what the peer files there lands INSIDE the writer's window — deterministically, through the real code's own call. Invocation
    k acts on the k-th run the writer settles (`order`): it CREATES that cell's complete store, takes the store's close lock as the closer does
    (StoreCloseLock, `wait_s`) and files what the store512's and a peer task's closes file (the store512's `generated`, the climb's facts and
    its `superseded`) — or, refused at the lock, files nothing; either way its log says which. It then prints an AllocTRES line (one h200) so
    the capture reads the GPU through it. Returns (the bin directory, the peer's log)."""
    where.mkdir(parents=True, exist_ok=True); bin_dir = where / "bin"; bin_dir.mkdir()
    peer = where / "peer.py"; peer.write_text(_PEER_IN_THE_PROBE)
    args = where / "args.json"; log = where / "peer.log"; calls = where / "calls"
    args.write_text(json.dumps({"tests": str(REPO / "tests"), "src": str(REPO / "src"), "closer": str(CLOSER), "wave": str(wave), "spool": str(spool), "order": list(order),
                                "s3": dict(s3_rids), "climb": dict(climb_rids), "s3_job": S3, "goals": goals, "n_max": N_MAX, "config": str(cfg), "wait_s": wait_s,
                                "log": str(log), "calls": str(calls)}))
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 1\n")
    (bin_dir / "sacct").write_text(f"#!/bin/sh\nexec {sys.executable} {peer} {args}\n")
    for f in (bin_dir / "nvidia-smi", bin_dir / "sacct"):
        f.chmod(0o755)
    return bin_dir, log


def _hold_locks(paths):
    held = []
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "a+"); fcntl.flock(fh.fileno(), fcntl.LOCK_EX); held.append(fh)
    return held


def test_d456_a_lifecycle_writer_for_an_absent_store_takes_the_store_lock_so_a_close_landing_in_its_window_is_serialised(tmp_path, monkeypatch):
    """Deterministic. (1) THE INTERLEAVING THROUGH THE WRITER'S OWN WINDOW: nothing of the stores exists
    when the climb array's task 0 starts; its `started` reads no close and captures hardware before it writes — and the PEER CLOSE runs inside
    that capture (the sacct on PATH): the store created complete, the lock taken as the closer takes it, the store512's `generated` and the
    climb's `superseded` filed. Before: the writer held no lock for an absent store, the peer's close landed between its read and its write,
    and the shared run read `superseded` then `running` — the retirement reversed. Now: the writer holds the store's lock from before its
    read, the peer is REFUSED at it and files nothing, `running` is filed through the capture; the real store512 and task-1 closes then retire
    the run under the same lock; folded, one current run per store, admission's supersession through the rows. (2) The refusal note after a
    `no store directory` close: the same window, the same outcome. (3) The lock lives BESIDE the store, never inside it: a dry run creates
    nothing; the writer creates the lock's directory, the store directory stays absent and the climb planner still reads the cell ABSENT (a
    native pass, never a store in progress). (4) A task's `failed` for an absent store — no capture in its path — blocks at the held lock and
    files once when released, naming the lock."""
    # (1) the array task's `started`, the peer close inside its hardware capture
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    pa, pb = pcs; rids = {obj: _rid(pa, obj) for obj in OBJECTS}; s3 = {obj: _rid(p3, obj) for obj in OBJECTS}; cfg = _store_config(stores)
    order = [r["route"]["object"] for r in json.loads(pa.read_text())["runs"]]; assert sorted(order) == sorted(OBJECTS)   # the writer settles the runs in this order
    shutil.rmtree(wave / "transcripts_results")                                                                             # no store, no private run directory
    bin1, plog1 = _peer_in_the_probe(tmp_path / "probe1", wave=wave, spool=spool, order=order, s3_rids=s3, climb_rids=rids, goals=goals, cfg=cfg)
    r = _lifecycle(["--array-task-event", "started", "--array-job", A_JOB, "--array-task", "0", "--array-tasks", "0-1", "--scheduler-state", "RUNNING"],
                   pending=pa, spool=spool, wave=wave, bin_dir=bin1)
    out = r.stdout + r.stderr; peer = plog1.read_text() if plog1.exists() else ""
    assert r.returncode == 0, out
    assert all(plog1.exists() and f" {obj}:" in peer for obj in OBJECTS), (out, peer)                                     # the peer ran once per run, in the window
    for obj in OBJECTS:                                                                                                     # `running` alone: the peer's close never landed between the read and the write
        assert _all_states(spool, [rids[obj]])[rids[obj]] == ["running"], (obj, _all_states(spool, [rids[obj]]), peer, out)
        assert f"REFUSED {obj}: the store close lock " in peer and f"FILED {obj}" not in peer, peer                         # refused AT THE WRITER'S LOCK
        assert f"running strongreject_bon {obj} {rids[obj][:40]}…: filed for {_who(0, CLIMBS[0])} — no close has decided this run (under the store close lock)" in out, out
    assert out.count("gpu_count 1 from sacct_alloc_tres; gpu_type from sacct_alloc_tres") == 2 and "without a store lock" not in out, out   # the capture ran through the probe
    assert (plog1.parent / "calls").read_text().count("\n") == 2
    r3 = _close(p3, spool, wave, job_name=S3_NAME); assert r3.returncode == 0 and "2 filed" in r3.stdout + r3.stderr, r3.stdout + r3.stderr
    rb = _close(pb, spool, wave, job_name=CLIMB_NAME); assert rb.returncode == 0 and f"2 {D2599_LINE}" in rb.stdout + rb.stderr, rb.stdout + rb.stderr
    assert all(_all_states(spool, [rid])[rid] == ["running", "superseded"] for rid in rids.values()), _all_states(spool, list(rids.values()))
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, s3[obj], [rids[obj]])
    # (2) the refusal note after a `no store directory` close, the peer close inside its capture; the note's order is the refused list's (sorted ids)
    root2, spool2, wave2, q3, qcs, stores2, g2 = _bon_world(tmp_path / "n", monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    qa, qb = qcs; rids2 = {obj: _rid(qa, obj) for obj in OBJECTS}; s32 = {obj: _rid(q3, obj) for obj in OBJECTS}; cfg2 = _store_config(stores2)
    shutil.rmtree(wave2 / "transcripts_results")
    r = _close(qa, spool2, wave2, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 2 and out.count("no store directory — the run stays running") == 2, out
    order2 = sorted(OBJECTS, key=lambda o: rids2[o])
    bin2, plog2 = _peer_in_the_probe(tmp_path / "probe2", wave=wave2, spool=spool2, order=order2, s3_rids=s32, climb_rids=rids2, goals=g2, cfg=cfg2)
    r = _lifecycle(["--refusal-note", _NOTE, "--node", "TESTHOST"], pending=qa, spool=spool2, wave=wave2, bin_dir=bin2)
    out = r.stdout + r.stderr; peer = plog2.read_text() if plog2.exists() else ""
    assert r.returncode == 0, out
    for obj in OBJECTS:
        assert _all_states(spool2, [rids2[obj]])[rids2[obj]] == ["running"], (obj, _all_states(spool2, [rids2[obj]]), peer, out)
        assert f"REFUSED {obj}: the store close lock " in peer and f"FILED {obj}" not in peer, peer
        assert f"running strongreject_bon {obj} {rids2[obj][:40]}…: the closer's refusal note filed — no close has decided this run (under the store close lock)" in out, out
        st = _state(spool2, rids2[obj], "running"); assert len(st) == 1 and st[0]["note"].startswith(_NOTE) and "gpu_count 1 from sacct_alloc_tres" in st[0]["note"], st
    assert f"state running: stated for 2 refused run(s) of job {CLIMBS[0]} (2 listed; the refused-run list; D2100 (b))" in out and "without a store lock" not in out, out
    r3 = _close(q3, spool2, wave2, job_name=S3_NAME); assert r3.returncode == 0 and "2 filed" in r3.stdout + r3.stderr, r3.stdout + r3.stderr
    rb = _close(qb, spool2, wave2, job_name=CLIMB_NAME); assert rb.returncode == 0 and f"2 {D2599_LINE}" in rb.stdout + rb.stderr, rb.stdout + rb.stderr
    assert all(_all_states(spool2, [rid])[rid] == ["running", "superseded"] for rid in rids2.values())
    _fold(root2)
    for obj in OBJECTS:
        _assert_one_current(root2, s32[obj], [rids2[obj]])
    # (3) the lock beside the store: a dry run creates nothing; the writer creates the lock's directory and never the store's
    from sisteer.eval.attacks import bon_climb as BC
    RCS = _closer_module()
    root3, spool3, wave3, _r3, rcs3, stores3, _g3 = _bon_world(tmp_path / "a", monkeypatch, climbs=2, climb_key=CLIMB_KEY)
    ra, rb3 = rcs3; rids3 = {obj: _rid(rb3, obj) for obj in OBJECTS}; assert all(_rid(ra, obj) == rids3[obj] for obj in OBJECTS)
    order3 = [r["route"]["object"] for r in json.loads(rb3.read_text())["runs"]]
    shutil.rmtree(wave3 / "transcripts_results")
    locks = {obj: Path(RCS.bon_close_lock_path(str(stores3[obj]))) for obj in OBJECTS}
    for obj in OBJECTS:
        assert locks[obj] == stores3[obj].parent / f".{obj}.close.lock" and RCS.bon_close_lock_path(str(stores3[obj]) + "/") == str(locks[obj]), locks   # the shape, normalised
        assert RCS.StoreCloseLock(str(stores3[obj])).path == str(locks[obj])
    started = ["--array-task-event", "started", "--array-job", A_JOB, "--array-task", "1", "--array-tasks", "0-1", "--scheduler-state", "RUNNING"]
    r = _lifecycle([*started, "--dry-run"], pending=rb3, spool=spool3, wave=wave3); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("would file run_state running") == 2 and not (wave3 / "transcripts_results").exists(), out
    r = _lifecycle(started, pending=rb3, spool=spool3, wave=wave3); out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("— no close has decided this run (under the store close lock)") == 2 and "without a store lock" not in out, out
    for obj in OBJECTS:
        assert locks[obj].is_file() and not stores3[obj].exists() and _all_states(spool3, [rids3[obj]])[rids3[obj]] == ["running"], (obj, out)
        assert BC.store_state(stores3[obj]) == (BC.STORE_ABSENT, {}) and BC.climb_mode(stores3[obj]) == BC.MODE_NATIVE                  # the planner's cell is still absent
    assert sorted(p.name for p in stores3[OBJECTS[0]].parent.iterdir()) == sorted(f".{obj}.close.lock" for obj in OBJECTS)
    # (4) a task's `failed` for an absent store blocks at the held lock (no capture in its path) and files once when released
    held = _hold_locks([locks[obj] for obj in order3])
    log = tmp_path / "failed_absent.log"
    proc = _lifecycle(["--array-task-event", "failed", "--array-job", A_JOB, "--array-task", "1", "--array-tasks", "0-1", "--scheduler-state", "FAILED:rc=2"],
                      pending=rb3, spool=spool3, wave=wave3, log=log)
    try:
        _wait_for(lambda: "waiting for the store close lock" in log.read_text() or proc.poll() is not None, 180, "the task's `failed` to reach the store lock")
        time.sleep(1.0)
        assert proc.poll() is None and f"waiting for the store close lock {locks[order3[0]]} held by " in log.read_text(), log.read_text()
        assert all(_all_states(spool3, [rid])[rid] == ["running"] for rid in rids3.values()) and _outcomes(rb3) == {}                     # nothing filed outside the lock
    finally:
        for fh in held:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN); fh.close()
    assert proc.wait(timeout=600) == 0, log.read_text()
    out = log.read_text()
    for obj in OBJECTS:
        assert f"failed strongreject_bon {obj} {rids3[obj][:40]}…: filed for {_who(1, CLIMBS[1])} (FAILED:rc=2; under the store close lock) — no close has decided this run" in out, out
        assert _all_states(spool3, [rids3[obj]])[rids3[obj]] == ["running", "failed"] and not stores3[obj].exists()
    assert set(_outcomes(rb3)) == {1} and "without a store lock" not in out


# ------------------------------------------------- the rerun pair's job names; a rung is never a twin
# The smoke re-ran the BoN prompted__si_constitution cell as a two-rung ladder on ONE store through the rerun-units pair: the store job
# `f22-stat-<tag>-<slug>-3` (cap 512, rung 0, base_cap) and the climb `f22-bonclimb-<tag>-<slug>` (cap 4096, rung 1, ladder_rung), identical
# store facts (store_uri, store_completion_sha256, n_items_generated). bon_role_of matched twins by the EXACT plain names, so the suffixed pair
# were "another tag" to each other (neither retired). The rule: recognise the suffix (the SAME slug on both sides) ONLY with a guard
# in bon_settle_duplicates — a different token_cap_tokens or pass_label is NEVER a twin (a larger cap never supersedes a smaller one) —
# because the matcher compares store facts alone and always keeps the store512 role: widened without the guard, a close or reclose of the 512
# run would supersede the climb, the store's only judgment. Never retire this pair retroactively.
SLUG = "constitution_rerun"
S3R, CLIMBR = "810181", "810191"                                  # the rerun pair's jobs (synthetic ids)
S3R_NAME, CLIMBR_NAME = f"f22-stat-{TAG}-{SLUG}-3", f"f22-bonclimb-{TAG}-{SLUG}"
RUNG_512 = ("--token-cap", "512", "--rung-index", "0", "--ladder", "512,4096", "--pass-label", "base_cap")
RUNG_4096 = ("--token-cap", "4096", "--rung-index", "1", "--ladder", "512,4096", "--pass-label", "ladder_rung")
RUNG_NOTE_512_SEES_CLIMB = "token_cap_tokens 512 vs 4096; pass_label base_cap vs ladder_rung"       # the store512's close reading its climb twin
RUNG_NOTE_CLIMB_SEES_512 = "token_cap_tokens 4096 vs 512; pass_label ladder_rung vs base_cap"       # the climb's close reading the store512 filing


def _rerun_world(tmp_path, monkeypatch, *, store_extra=(), climb_extra=()):
    """A rerun pair — the rerun-units slug on BOTH job names — over the two BoN cells' complete 512 stores; `store_extra` / `climb_extra` are
    the registrations' rung flags (none: the pre-ladder shape, the instrument's cap 4096 and no pass label on both)."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    p3 = _register(tmp_path, root, job=S3R, job_name=S3R_NAME, evals="strongreject_bon", extra=store_extra)
    pc = _register(tmp_path, root, job=CLIMBR, job_name=CLIMBR_NAME, evals="strongreject_bon", producer_key=f"{CLIMB_KEY}-{SLUG}", extra=climb_extra)
    goals = int(json.loads(p3.read_text())["runs"][0]["plan"]["n_items_planned"]); assert goals > 0
    cfg = _bon_plan_config(tmp_path, goals=goals, n_max=N_MAX)
    stores = {obj: _bon_store(wave, obj, goals=goals, n_max=N_MAX, responded=goals * N_MAX, refused=0, config=cfg) for obj in OBJECTS}
    return root, spool, wave, p3, pc, stores, goals


def test_d471_bon_role_of_reads_the_rerun_pairs_names_under_one_slug_only_and_the_plain_names_exactly_as_before():
    """The matcher's grammar: the plain names unchanged; the rerun pair's names a role ONLY when the closing job's slug is given and spelled on
    the twin; a plain name beside a rerun job, a rerun name beside a plain job, another slug or a suffix that is not a rerun slug: no role (another
    store by construction). The tag-less read (this job's own kind) sees the suffixed names as the regexes always did. The rung guard: equal
    rungs and two unrecorded halves agree; a differing cap or label, or a side with no registration record, names itself."""
    closer = _closer_module()
    T = "synthtag_battery"
    assert closer.bon_role_of(f"f22-stat-{T}-3", T) == "store512" and closer.bon_role_of(f"f22-bonclimb-{T}", T) == "climb"
    assert closer.bon_role_of(f"f22-stat-{T}-all", T) is None and closer.bon_role_of(f"f22-stat-{T}_other-3", T) is None and closer.bon_role_of(None, T) is None
    assert closer.bon_rerun_slug_of(f"f22-stat-{T}-{SLUG}-3", T) == SLUG and closer.bon_rerun_slug_of(f"f22-bonclimb-{T}-{SLUG}", T) == SLUG
    assert closer.bon_role_of(f"f22-stat-{T}-{SLUG}-3", T, rerun=SLUG) == "store512" and closer.bon_role_of(f"f22-bonclimb-{T}-{SLUG}", T, rerun=SLUG) == "climb"
    assert closer.bon_role_of(f"f22-stat-{T}-{SLUG}-3", T) is None and closer.bon_role_of(f"f22-bonclimb-{T}-{SLUG}", T) is None            # a rerun name beside a plain job
    assert closer.bon_role_of(f"f22-stat-{T}-3", T, rerun=SLUG) is None and closer.bon_role_of(f"f22-bonclimb-{T}", T, rerun=SLUG) is None   # a plain name beside a rerun job
    assert closer.bon_role_of(f"f22-bonclimb-{T}-other_rerun", T, rerun=SLUG) is None and closer.bon_role_of(f"f22-stat-{T}-other_rerun-3", T, rerun=SLUG) is None
    for n in (f"f22-stat-{T}-foo-3", f"f22-bonclimb-{T}-foo", f"f22-stat-{T}-{SLUG}", f"f22-stat-{T}--3", f"f22-bonclimb-{T}-", f"f22-stat-{T}-3",
              f"f22-bonclimb-{T}", f"f22-bonclimb-{T}-a-b_rerun", f"f22-bonclimb-other-{SLUG}"):
        assert closer.bon_rerun_slug_of(n, T) is None, n
    assert closer.bon_rerun_slug_of(f"f22-stat-{T}-{SLUG}-3", None) is None and closer.bon_rerun_slug_of(None, T) is None
    assert closer.bon_role_of(f"f22-stat-{T}-{SLUG}-3") == "store512" and closer.bon_role_of(f"f22-bonclimb-{T}-{SLUG}") == "climb"
    assert closer.bon_rung_mismatch({"token_cap_tokens": 4096, "pass_label": None}, {"token_cap_tokens": "4096", "pass_label": None}) is None
    assert closer.bon_rung_mismatch({"token_cap_tokens": None, "pass_label": None}, {"token_cap_tokens": None, "pass_label": None}) is None
    assert closer.bon_rung_mismatch({"token_cap_tokens": 512, "pass_label": "base_cap"}, {"token_cap_tokens": 4096, "pass_label": "ladder_rung"}) == RUNG_NOTE_512_SEES_CLIMB
    assert closer.bon_rung_mismatch({"token_cap_tokens": 4096, "pass_label": None}, {"token_cap_tokens": 4096, "pass_label": "ladder_rung"}) == "pass_label None vs ladder_rung"
    assert closer.bon_rung_mismatch({"token_cap_tokens": 512, "pass_label": "base_cap"}, {"token_cap_tokens": 4096, "pass_label": "base_cap"}) == "token_cap_tokens 512 vs 4096"
    a = closer.bon_rung_mismatch(None, {"token_cap_tokens": 512, "pass_label": "base_cap"}); b = closer.bon_rung_mismatch({"token_cap_tokens": 512, "pass_label": "base_cap"}, None)
    assert a and "not on record" in a and a.endswith("this run") and b and "not on record" in b and b.endswith("the twin")
    assert "token_cap_tokens or pass_label differ" in closer.BON_SUPERSEDED_RULE and "D654" in closer.BON_SUPERSEDED_RULE


def test_d471_a_rerun_suffixed_same_rung_twin_pair_is_settled_and_the_smokes_different_rung_rerun_pair_is_never_a_twin(tmp_path, monkeypatch):
    """(i) A rerun pair at ONE rung (plain registrations: the instrument's cap 4096 and no pass label on both, every pre-ladder pair's shape) is
    settled exactly as the plain names are: the store512 closes `generated` and names its climb twin; the climb's close files its own run
    superseded by it (extra.superseded_by_run_id, sibling_job_id = the rerun store job); folded, one current run per store. Before this fix,
    the suffixed names were "another tag" to each other and the climb closed `generated` beside the store512 — two current registrations.
    (ii) The smoke's pair: the 512 store at rung 0 / base_cap and its climb at rung 1 / ladder_rung, IDENTICAL store facts. The store512 closes
    first (as the smoke's afterok order did) and files `generated`; the climb's close reads the store512 filing of its store as ANOTHER RUNG of
    the ladder -- noted, no candidate -- and files its own `generated`, never its own supersede; the store512's reclose reads the climb's filing
    the same way -- noted, never HELD, never superseded; the climb's reclose writes nothing. Folded: both runs `generated`, both current here
    (the 512 run's is_current 0 in the smoke is r2v1's, not the closer's), no superseded event on either. With the suffix recognised but no
    rung guard, the climb's close filed ITSELF superseded by the 512 run and the 512 run's reclose superseded the climb -- the store's only
    judgment -- the hazard the guard prevents. (iii) A plain-named climb beside the rerun store512 is another store, never a twin."""
    # (i) one rung: settled
    root, spool, wave, p3, pc, stores, goals = _rerun_world(tmp_path / "same", monkeypatch)
    closer = _closer_module()
    for obj in OBJECTS:
        for rid in (_rid(p3, obj), _rid(pc, obj)):
            assert closer.bon_rung_facts(str(spool), "sdfbase_full22", rid) == {"token_cap_tokens": 4096, "pass_label": None}, rid
    r = _close(p3, spool, wave, job_name=S3R_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "superseded" not in out and "another rung" not in out, out
    assert out.count("has not closed `generated` — it supersedes itself at its own close") == 2 and "another tag or another rerun slug" not in out, out
    r = _close(pc, spool, wave, job_name=CLIMBR_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"0 filed, 0 skipped, 0 refused of 2 registered run(s), 2 {D2599_LINE}" in out, out
    for obj in OBJECTS:
        a, b = _rid(p3, obj), _rid(pc, obj); assert a != b
        st = _state(spool, b, "superseded"); assert len(st) == 1 and not _state(spool, b, "generated"), st
        assert st[0]["extra"]["superseded_by_run_id"] == a and st[0]["extra"]["sibling_job_id"] == S3R and st[0]["extra"]["decision"] == "D2599"
        assert st[0]["extra"]["superseded_role"] == "climb" and st[0]["extra"]["survivor_role"] == "store512"
        pb = _patch(spool, b); assert pb is not None and {k: pb[k] for k in FACTS} == {k: _patch(spool, a)[k] for k in FACTS} and "generated_utc" not in pb
    _fold(root)
    for obj in OBJECTS:
        _assert_one_current(root, _rid(p3, obj), [_rid(pc, obj)])
    # (ii) the smoke's two-rung ladder: never a twin, in either direction, on any close or reclose
    root2, spool2, wave2, q3, qc, stores2, g2 = _rerun_world(tmp_path / "ladder", monkeypatch, store_extra=RUNG_512, climb_extra=RUNG_4096)
    for obj in OBJECTS:
        assert closer.bon_rung_facts(str(spool2), "sdfbase_full22", _rid(q3, obj)) == {"token_cap_tokens": 512, "pass_label": "base_cap"}
        assert closer.bon_rung_facts(str(spool2), "sdfbase_full22", _rid(qc, obj)) == {"token_cap_tokens": 4096, "pass_label": "ladder_rung"}
    r = _close(q3, spool2, wave2, job_name=S3R_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "superseded" not in out and "HELD" not in out, out
    assert out.count("has not closed `generated` — it supersedes itself at its own close") == 2, out       # the climb twin named: not filed yet
    r = _close(qc, spool2, wave2, job_name=CLIMBR_NAME); out = r.stdout + r.stderr                            # the hazard's first face
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "superseded" not in out and "HELD" not in out and "REFUSED" not in out, out
    for obj in OBJECTS:
        a, b = _rid(q3, obj), _rid(qc, obj)
        assert f"the store512 registration {a[:40]}… (job {S3R}) of this store is another rung of its ladder — {RUNG_NOTE_CLIMB_SEES_512} — not a duplicate of this one" in out, out
        assert "D654" in out and len(_state(spool2, b, "generated")) == 1 and not _state(spool2, b, "superseded")
        assert out.index(f"note strongreject_bon {obj} {b[:40]}…: the store512 registration") < out.index(f"generated strongreject_bon {obj} {b[:40]}"), out   # the note, then its own `generated`
    n_before = {obj: (len(_records(spool2, _rid(q3, obj))), len(_records(spool2, _rid(qc, obj)))) for obj in OBJECTS}
    r = _close(q3, spool2, wave2, job_name=S3R_NAME); out = r.stdout + r.stderr                               # the hazard's second face: the 512 run's reclose
    assert r.returncode == 0 and out.count("already generated") == 2 and "superseded" not in out and "HELD" not in out and " held (" not in out, out
    for obj in OBJECTS:
        b = _rid(qc, obj)
        assert f"{b[:40]}… (climb job {CLIMBR}): another rung of this store's ladder — {RUNG_NOTE_512_SEES_CLIMB} — not a duplicate of this store512 registration" in out, out
        assert not _state(spool2, b, "superseded") and len(_state(spool2, b, "generated")) == 1
    r = _close(qc, spool2, wave2, job_name=CLIMBR_NAME); out = r.stdout + r.stderr                            # the climb's reclose: nothing more
    assert r.returncode == 0 and out.count("already generated") == 2 and "superseded" not in out and RUNG_NOTE_CLIMB_SEES_512 in out, out
    assert {obj: (len(_records(spool2, _rid(q3, obj))), len(_records(spool2, _rid(qc, obj)))) for obj in OBJECTS} == n_before     # the recloses wrote nothing
    _fold(root2)
    for obj in OBJECTS:
        a, b = _rid(q3, obj), _rid(qc, obj)
        cur = _currency(root2, [a, b])
        assert cur[a] == ("generated", 1, None) and cur[b] == ("generated", 1, None), cur                    # both stand; no closer-side retirement
        assert _superseded_events(root2, a) == [] and _superseded_events(root2, b) == []
        fa, fb = _row_facts(root2, a), _row_facts(root2, b)
        assert fa["store_uri"] == fb["store_uri"] == str(stores2[obj]) and fa["store_completion_sha256"] == fb["store_completion_sha256"]
        assert fa["n_items_generated"] == fb["n_items_generated"] == g2 * N_MAX and fa["generated_utc"] and fb["generated_utc"]
    # (iii) the plain names beside a rerun job: another store, never a twin (the exact-name behaviour unchanged)
    root3, spool3, wave3, w3, wc, stores3, g3 = _rerun_world(tmp_path / "mixed", monkeypatch)
    plain = _register(tmp_path / "mixed", root3, job=CLIMBS[0], job_name=CLIMB_NAME, evals="strongreject_bon", producer_key=CLIMB_KEY)
    r = _close(w3, spool3, wave3, job_name=S3R_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "superseded" not in out, out
    assert out.count(f"by job(s) {CLIMBS[0]} ({CLIMB_NAME}) name another tag or another rerun slug — another store under this wave root, never a twin of this one") == 2, out
    assert out.count("has not closed `generated` — it supersedes itself at its own close") == 2           # the rerun climb twin alone is a twin
    r = _close(plain, spool3, wave3, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed, 0 skipped, 0 refused" in out and "superseded" not in out, out
    assert out.count(f"2 registration(s) of this measurement by job(s) {S3R}, {CLIMBR} ({CLIMBR_NAME}, {S3R_NAME}) name another tag or another rerun slug") == 2, out
    assert "has not closed `generated`" not in out and "another rung" not in out                       # the plain climb has NO twin here
    for obj in OBJECTS:
        assert len(_state(spool3, _rid(plain, obj), "generated")) == 1 and not _state(spool3, _rid(plain, obj), "superseded")


def test_a_climb_close_files_the_climbs_flat_root_beside_the_store_facts_and_a_store512_close_does_not(tmp_path, monkeypatch):
    """A climb run's `store_uri` is the 512 STORE it climbed, so the climb's own artifacts
    (`$WAVE_ROOT/bon_climb/<TAG>`: plan/climb_manifest.jsonl, out/<cell>/shard<k>.jsonl) had NO record — the gate could only recompute them
    from the producer convention, which is forbidden. The closer, the one filer of store locations, now records the flat root as
    `extra_json.climb_root` on every climb run it closes (the store-facts patch a store512-first climb files before its `superseded`, and the
    `generated` patch of a climb that closes first), and never on a store512 run; the fold's additive merge lands it in the run row's
    extra_json, where 06 --gate reads it (stage.py probe_store)."""
    root, spool, wave, p3, pcs, stores, goals = _bon_world(tmp_path, monkeypatch, climbs=1)
    cl = {obj: _rid(pcs[0], obj) for obj in OBJECTS}; s3 = {obj: _rid(p3, obj) for obj in OBJECTS}
    # the climb closes FIRST: `generated` + the store facts + the flat root
    r = _close(pcs[0], spool, wave, job_name=CLIMB_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    for obj in OBJECTS:
        pa = _patch(spool, cl[obj]); assert pa is not None, out
        assert (pa.get("extra_json") or {}).get("climb_root") == str(wave / "bon_climb" / TAG) and pa["store_uri"] == str(stores[obj]) and "generated_utc" in pa, pa
        # the merge of gate-fixes with decode-recording (integration round 3a): both lanes file under extra_json of the SAME run_patch -- the
        # climb's flat root beside the request's observed sampler (this fixture's manifest records it) -- so the one dict carries both, neither
        # lane's assignment replacing the other's
        assert set(pa["extra_json"]) >= {"climb_root", "decode_evidence"} and pa["extra_json"]["decode_evidence"]["family"] == "bon", pa["extra_json"]
    r = _close(p3, spool, wave, job_name=S3_NAME); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    for obj in OBJECTS:
        pa3 = _patch(spool, s3[obj]); assert pa3 is not None and "climb_root" not in (pa3.get("extra_json") or {}), pa3
    _fold(root)
    with DB.reading(DB.db_path(str(root))) as conn:
        for obj in OBJECTS:
            ej = json.loads(conn.execute("SELECT extra_json FROM run WHERE run_id = ?", (cl[obj],)).fetchone()[0] or "{}")
            assert ej.get("climb_root") == str(wave / "bon_climb" / TAG), ej
            ej3 = json.loads(conn.execute("SELECT extra_json FROM run WHERE run_id = ?", (s3[obj],)).fetchone()[0] or "{}")
            assert "climb_root" not in ej3
    # the gate's reader: with the record the climb is probed at the flat root; a climb row without it reads "no store on record" by name
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage_under_test", str(REPO / "reproduce" / "lib" / "stage.py")); ST_ = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST_)
    unit = {"evals": {"strongreject_bon": {"coverage": {"layout": "bon_climb_flat", "nshards": 4}, "expected_shards": ["shard00", "shard01", "shard02", "shard03"]}}}
    path, src, why = ST_.probe_store(unit, {"run_id": cl[OBJECTS[0]], "eval_id": "strongreject_bon", "object": OBJECTS[0], "array_task": 0, "store_uri": str(stores[OBJECTS[0]]), "extra": ej})
    assert (path, src, why) == (str(wave / "bon_climb" / TAG), "registry.run.extra_json.climb_root", None)
    path, src, why = ST_.probe_store(unit, {"run_id": "x", "eval_id": "strongreject_bon", "object": OBJECTS[0], "array_task": 0, "store_uri": str(stores[OBJECTS[0]]), "extra": {}})
    assert path is None and src == "none" and "not covered, no store on record for the climb's flat root" in why and str(stores[OBJECTS[0]]) in why and "D3003 (1)" in why
