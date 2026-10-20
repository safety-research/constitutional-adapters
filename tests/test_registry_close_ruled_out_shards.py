"""The unsteered model is not re-run: a multi-turn run whose REMAINING shards are ruled out by
the cells selection (tonight: the three attacks' `none` runs, shards 05-08 — the g5678 group runs the two ReLU cells only) can never advance,
so the closer closes it `cancelled` — the lifecycle's "partial store" terminal — with the ruling as evidence (run_state {state cancelled,
evidence_kind ruling, evidence_uri the ruling record, note}) and a run_patch n_items_generated = the goals the present shards hold. Never on a
bare flag (the three flags refuse by name without the evidence uri), never for shards merely not yet run (a missing shard outside the ruled-out
set keeps the run running), never for an object the ruling does not name. Reuses the real closer fixtures of tests/test_registry_close_stores.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml  # noqa: E402

from sisteer.registry import emit, hooks  # noqa: E402
from test_registry_close_stores import REPO, TAG, _close, _multiturn_store, _records, _register, _runs, _seeded_registry  # noqa: E402

URI = "rulings/added/2026-10-02-ruling-r2v4-exclude-qwen35-35b-a3b-s2r-none-not-the-baseline.jsonl"
RULED = ("--ruled-out-shards", "5,6,7,8", "--ruled-out-objects", "none", "--ruling-evidence-uri", URI)


def _ids(tmp_path, pending, spool, wave):
    ids_file = tmp_path / "expected_goal_ids.json"
    r = _close(pending, spool, wave, "--emit-expected-goal-ids", str(ids_file))
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(ids_file.read_text())["goal_ids"], ("--expected-goal-ids", str(ids_file))


def _world(tmp_path, monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"
    wave = tmp_path / "exp" / "reproduce" / TAG / "waves" / "mt-u0-cluster_a"; wave.mkdir(parents=True)
    pending = _register(tmp_path, root, job="810071", job_name=f"f22-att-{TAG}-g01234-crescendo", evals="crescendo")
    prun = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_text())["private_run"]
    real_ids, IDS = _ids(tmp_path, pending, spool, wave)
    return root, spool, wave, pending, prun, real_ids, IDS


def test_a_run_whose_remaining_shards_are_ruled_out_closes_cancelled_with_the_ruling_and_the_present_counts(tmp_path, monkeypatch):
    root, spool, wave, pending, prun, real_ids, IDS = _world(tmp_path, monkeypatch)
    runs = _runs(pending); none_rid = runs[("crescendo", "none")]["run_id"]; steered = [o for (e, o) in runs if o != "none"][0]; steered_rid = runs[("crescendo", steered)]["run_id"]
    present_goals = len([g for g in range(len(real_ids)) if g % 9 in (0, 1, 2, 3, 4)])
    _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=[0, 1, 2, 3, 4], goal_ids=real_ids)          # shards 05-08 absent: the other group
    _multiturn_store(wave, prun, "crescendo", steered, nshards=9, goals=259, shards=[0, 1, 2, 3, 4], goal_ids=real_ids)         # the steered cell: its 05-08 WILL run
    # without the ruling flags: both stay running, as before
    r = _close(pending, spool, wave, *IDS); out = r.stdout + r.stderr
    assert r.returncode == 2 and out.count("the run stays running") == 2 and "4 of 9 expected shard(s) missing or incomplete: shard05, shard06, shard07, shard08" in out, out
    assert not [p for t, p, *_ in _records(spool, none_rid) if t == "run_state" and p.get("state") == "cancelled"]
    # with the ruling: the none run closes cancelled with the evidence and the present shards' goals; the steered run stays running
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert r.returncode == 2 and "1 cancelled (shards ruled out; D1168)" in out and out.count("the run stays running") == 1, out
    assert f"cancelled crescendo none" in out and f"n_items={present_goals}/259" in out and "shards shard05, shard06, shard07, shard08 ruled out by design" in out, out
    st = [p for t, p, *_ in _records(spool, none_rid) if t == "run_state" and p.get("state") == "cancelled"]
    assert len(st) == 1 and st[0]["evidence_kind"] == "ruling" and st[0]["evidence_uri"] == URI and st[0]["event_utc"], st
    assert "ruled out by design" in st[0]["note"] and f"closed with shards shard00, shard01, shard02, shard03, shard04 present: {present_goals} goal(s)" in st[0]["note"], st[0]["note"]
    pt = [p for t, p, *_ in _records(spool, none_rid) if t == "run_patch"]
    assert pt and pt[-1]["fields"]["n_items_generated"] == present_goals and pt[-1]["fields"]["store_uri"].endswith("/crescendo/none")
    assert not [p for t, p, *_ in _records(spool, steered_rid) if t == "run_state" and p.get("state") in ("cancelled", "generated")]
    # idempotent: a second close with the ruling files nothing new for the cancelled run (its state is terminal: the closer skips it)
    n_before = len(_records(spool, none_rid))
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert len(_records(spool, none_rid)) == n_before and "already cancelled (shards ruled out; D1168)" in out and "1 skipped" in out, out


def test_shards_merely_not_yet_run_and_an_unnamed_object_keep_the_run_running(tmp_path, monkeypatch):
    root, spool, wave, pending, prun, real_ids, IDS = _world(tmp_path, monkeypatch)
    runs = _runs(pending); none_rid = runs[("crescendo", "none")]["run_id"]
    _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=[0, 1, 3, 4], goal_ids=real_ids)              # shard02 is NOT ruled out: not yet run
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cancelled" not in out and "5 of 9 expected shard(s) missing or incomplete: shard02, shard05" in out and "the run stays running" in out, out
    assert not [p for t, p, *_ in _records(spool, none_rid) if t == "run_state" and p.get("state") == "cancelled"]
    # the ruling names another object: this run is not the ruling's
    r = _close(pending, spool, wave, *IDS, "--ruled-out-shards", "2,5,6,7,8", "--ruled-out-objects", "prompted__safety_refusal", "--ruling-evidence-uri", URI); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cancelled" not in out and "the run stays running" in out, out


def test_the_flags_refuse_by_name_without_the_ruling_evidence(tmp_path, monkeypatch):
    root, spool, wave, pending, prun, real_ids, IDS = _world(tmp_path, monkeypatch)
    r = _close(pending, spool, wave, *IDS, "--ruled-out-shards", "5,6,7,8", "--ruled-out-objects", "none"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "need --ruling-evidence-uri" in out and "never cancelled on a bare flag" in out and "REFUSED crescendo" not in out, out
    r = _close(pending, spool, wave, *IDS, "--ruling-evidence-uri", URI); out = r.stdout + r.stderr
    assert r.returncode == 2 and "needs both --ruled-out-shards" in out, out
    r = _close(pending, spool, wave, *IDS, "--ruled-out-shards", "x", "--ruled-out-objects", "none", "--ruling-evidence-uri", URI); out = r.stdout + r.stderr
    assert r.returncode == 2 and "is not a shard index or shardNN" in out, out
    assert _records(spool, _runs(pending)[("crescendo", "none")]["run_id"]) == [] or not [p for t, p, *_ in _records(spool, _runs(pending)[("crescendo", "none")]["run_id"]) if t == "run_state" and p.get("state") == "cancelled"]
