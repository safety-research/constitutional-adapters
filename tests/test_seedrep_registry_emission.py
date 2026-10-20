"""The Qwen seed-replicate native battery: the EMISSION and COMPLETION tests.

A synthetic wave of 2 objects x 2 controls x 2 shards runs through the REAL hooks (``hooks.store_recorder`` read through the
``SISTEER_REGISTRY_RUN_IDENTITY`` door, exactly as a container child does) into attempt-unique spools, then
``logical_run_complete.py`` verifies the shard union and emits completion ONCE per logical run into the grading host spool.
Assertions are on the EMITTED records, never on the plan files: one ``run`` per (object, eval); every ``item_result`` row
joins a run whose object_id / eval_id are its own; no ``'*'``-routed row lands on the wrong object; ``generated`` exactly
once per logical run, only after the union check. Known-bad: overlapping shards, a short union, a shard that declared
completion itself — each REFUSED with nothing emitted. ATTEMPT.json here carries the spelled-out shape (leg / view /
stage / shard / W / wave_uuid at the top level); the publish test drives the ``control_key`` shape.

P1-3 (fault injection): a crash after the verdict write and before the records leaves an ADMITTED-but-PENDING admission that
``registry_publish`` refuses; a crash between the two records leaves one; each re-run finishes the partial emission without a
second ``generated`` (deterministic record ids) and commits only after re-reading both from disk; a re-run after commit is a
no-op; a committed admission is never re-derived. P1-4: one accepted attempt of the combined ``controls`` container
completes all three logical runs (xstest / or_bench / mmlu) through the plan's ``extra.container`` axis; the launch plan's
own rows (``seedrep_launch_plan/1``: ``control_key`` + ``children[].registry_identity_rel``) must agree with it.
"""
from __future__ import annotations

import contextlib
import json
import re as _re_mod
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase" / "seedrep",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import logical_run_complete as LRC  # noqa: E402
import registry_plans as RP  # noqa: E402
import registry_publish as PUB  # noqa: E402
from sisteer.registry import emit, hooks, identity, synth  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402

W = synth.code_sha("b")
W7 = W[:7]
WAVE = "11111111-2222-3333-4444-555555555555"
WAVE2 = "22222222-3333-4444-5555-666666666666"
CAMPAIGN = "synthbattery"
LEG, VIEW = "ctl", "s1"
TASK_X, TASK_Y = "judged/ctlx_v1", "judged/ctly_v1"
EVAL_X, EVAL_Y = "ctlx", "ctly"
REV = "0123456789abcdef0123456789abcdef01234567"
N_ITEMS = 20
OBJECTS = {"objA": ("objA@a1b2c3d4", synth.OBJECT_SHA, "lora", "a1b2c3d4e5f60718"),
           "none": ("none", "none", "none", None)}
HALF = ([f"q{i:03d}" for i in range(1, 11)], [f"q{i:03d}" for i in range(11, 21)])


def _ident(obj: str, eval_id: str, task: str, *, leg: str = LEG, container: str = None) -> dict:
    okey, oid, kind, _ih = OBJECTS[obj]
    prk = f"seedrep_battery:{W7}:{leg}:{VIEW}"
    ident = {"campaign": CAMPAIGN, "producer_run_key_legacy": prk, "producer_object_key_legacy": okey, "model_id": synth.MODEL_ID,
             "object_id": oid, "transfer_kind": "base_to_instruct" if kind != "none" else "none", "eval_id": eval_id,
             "item_set_id": f"{eval_id}_20", "instrument_id": synth.INSTRUMENT_ID, "instrument_match": "match", "harness": "bespoke_sisteer",
             "harness_version": "synthetic", "scoring_rule": "judge_graded_weights", "prompt_template_sha256": synth.SHA256_A,
             "decode_seed_rule": "per_item", "decode_source": "model_card", "token_cap_tokens": 512, "thinking_mode": "off",
             "serving_path": "lora_native", "applied_positions_mode": "all" if kind != "none" else "none",
             "multi_turn_steering_scope": "single_turn" if kind != "none" else "not_applicable", "code_sha": W, "substrate": "modal",
             "substrate_source": "recorded", "seed_base": 1, "n_items_planned": N_ITEMS,
             "extra": {"seed_rule": "literal_training_seed", "leg": leg, "container": container or leg, "arm": obj,
                       "arm_kind": "fitted" if kind != "none" else "none", "s": 1, "view": VIEW, "served_key": obj, "store": "controls", "store_task": task}}
    ident["run_id"] = identity.run_id(CAMPAIGN, prk, okey, eval_id, synth.INSTRUMENT_ID)
    return ident


def _spec(obj: str) -> dict:
    okey, _oid, kind, ih = OBJECTS[obj]
    return {"sweep": "synthsweep", "run_id": "synthsweep.0.t0.a0", "model_id": synth.MODEL_ID, "model_revision": REV, "arm_name": obj, "kind": kind,
            "coeff": 1.0, "intervention_hash": ih, "engine": "lora_native", "trial_id": 0, "seed": 1, "attempt": 0,
            "harness_config_hash": "0f1e2d3c4b5a6978", "env_digest": "99aabbccddeeff00", "tiers": [1], "cell_index": 0}


def _row(task: str, item: str, value: float) -> dict:
    return {"task": task, "task_version": "v1", "domain": "overrefusal_judged", "item_id": item, "metric": "over_refusal", "value": value,
            "trial_id": 0, "seed": 1, "attempt": 0, "created_at": 1755000000.0 + int(item[1:]), "extra_json": None}


def _write_plans(root: Path) -> dict:
    plans = root / "plans"
    out = {}
    for obj in OBJECTS:
        content = {TASK_X: _ident(obj, EVAL_X, TASK_X), TASK_Y: _ident(obj, EVAL_Y, TASK_Y)}
        p = plans / LEG / f"{obj}__s1.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content, indent=1, sort_keys=True) + "\n")
        out[obj] = p
    return out


def _attempt_root(root: Path, shard: int, attempt_id: str, *, superseded_by=None, complete=True, leg: str = LEG, wave: str = WAVE) -> Path:
    a = root / "attempts" / leg / VIEW / str(shard) / "attempts" / attempt_id
    a.mkdir(parents=True, exist_ok=True)
    (a / PUB.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": attempt_id, "leg": leg, "view": VIEW, "shard": shard, "stage": "base", "W": W,
                                                  "wave_uuid": wave, "superseded_by": superseded_by, "call_id": f"fc-{attempt_id}", "task_id": f"ta-{attempt_id}"}))
    if complete:
        (a / PUB.COMPLETE_FILE).write_text(json.dumps({"attempt_id": attempt_id, "utc": "2026-01-01T00:00:00.000Z"}))
    return a


def _drive_shard(attempt: Path, plans: dict, items: dict, monkeypatch, *, finish: bool = False, unrouted: bool = True, tasks=(TASK_X, TASK_Y)):
    """A shard's child process per object: the recorder read through the ENV door, manifest + item rows; ``finish`` calls
    ``on_results`` (the single-container store's own completion), the default leaves completion to the grading host."""
    spool = attempt / PUB.SPOOL_REL
    for obj, plan in plans.items():
        monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(plan))
        rec = hooks.store_recorder(f"{obj}__tier1__t0__a0__a1b2c3d4__0f1e2d3c", _spec(obj), None, spool=str(spool), registry_emit=True)
        assert rec is not None
        store_dir = attempt / "exp" / obj
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "manifest.json").write_text("{}")
        rec.on_manifest(_spec(obj), str(store_dir / "manifest.json"))
        for task in tasks:
            for it in items[obj]:
                rec.on_row(_row(task, it, 1.0 if task == TASK_X else 0.5))
        if unrouted:
            rec.on_row(_row("judged/unrouted_task", "q000", 0.0))
        rec.flush()
        if finish:
            (store_dir / "results.json").write_text("{}")
            rec.on_results({}, str(store_dir / "results.json"))
        monkeypatch.delenv(hooks.RUN_IDENTITY_ENV)


def _accepted(root: Path, ids: dict, *, container: str = LEG, wave: str = WAVE, name: str = "accepted.json") -> Path:
    p = root / name
    p.write_text(json.dumps({f"{W7}/{wave}/{container}/{VIEW}/base/{shard}": aid for shard, aid in ids.items()}))
    return p


def _records(spool: Path):
    return [(p, rec, sha) for p in R.iter_record_files(str(spool)) for _i, rec, sha in R.read_record_file(p)]


def _validate_all(spool: Path):
    for p, rec, sha in _records(spool):
        v = VAL.validate_record(rec, p, sha)
        assert isinstance(v, VAL.Validated), (p, v)


def _re_esc(x: str) -> str:
    return _re_mod.escape(x)


def _dev(spool: Path, run_id: str):
    """``(generated run_state records, run_patch records)`` the grading host spool holds for one run."""
    recs = [rec for _p, rec, _s in _records(spool) if rec["payload"]["run_id"] == run_id]
    return ([r for r in recs if r["record_type"] == "run_state" and r["payload"]["state"] == "generated"],
            [r for r in recs if r["record_type"] == "run_patch"])


@pytest.fixture
def wave(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setattr(emit.socket, "gethostname", lambda: synth.TESTHOST)
    return tmp_path


def _two_shards(wave, monkeypatch):
    plans = _write_plans(wave)
    a0 = _attempt_root(wave, 0, "att-0")
    a1 = _attempt_root(wave, 1, "att-1")
    _drive_shard(a0, plans, {"objA": HALF[0], "none": HALF[0]}, monkeypatch)
    _drive_shard(a1, plans, {"objA": HALF[1], "none": HALF[1]}, monkeypatch)
    return plans, a0, a1, _accepted(wave, {0: "att-0", 1: "att-1"})


def test_two_objects_two_controls_two_shards_emit_one_run_per_object_eval_and_generated_once(wave, monkeypatch):
    plans, a0, a1, accepted = _two_shards(wave, monkeypatch)
    grading_host = wave / "grading_host_spool"
    verdicts = wave / "union"
    out = []
    for obj, plan in plans.items():
        out += LRC.complete_plan_file(plan, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=verdicts,
                                      items_source="spool")
    assert [v["verdict"] for v in out] == [LRC.VERDICT_ADMITTED] * 4 and all(v["n_union"] == N_ITEMS and v["expected_n"] == N_ITEMS for v in out)
    assert all(len(v["shards"]) == 2 and not v["already_generated"] and len(v["emitted"]) == 2 and not v.get("noop") for v in out)
    # the ADMISSION every verdict carries: the wave, the container, the complete attempt set (each with its six axes + file shas) — committed
    for v in out:
        adm, comp = v["admission"], v["completion"]
        assert comp["state"] == PUB.COMPLETION_COMMITTED and set(comp["record_ids"]) == {"run_state", "run_patch"} and set(comp) == {"state", "admission_utc", "record_ids"}
        assert (adm["run_id"], adm["campaign"], adm["w7"], adm["wave_uuid"], adm["container"], adm["view"]) == (v["run_id"], CAMPAIGN, W7, WAVE, LEG, VIEW)
        assert set(adm["attempts"]) == {"att-0", "att-1"} and len(adm["digest"]) == 64 and adm["digest"] == R.record_sha({k: x for k, x in adm.items() if k != "digest"})
        for aid, a, shard in (("att-0", a0, 0), ("att-1", a1, 1)):
            e = adm["attempts"][aid]
            assert {ax: e[ax] for ax in PUB.AXES} == {"w7": W7, "wave_uuid": WAVE, "container": LEG, "view": VIEW, "stage": "base", "shard": shard}
            rd = a / PUB.SPOOL_REL / CAMPAIGN / identity.run_dir_name(v["run_id"])
            assert e["files"] == PUB.run_dir_files(rd) and len(e["files"]) == 3 and e["n_items"] == 10
        assert PUB.admission_for(verdicts, identity.run_dir_name(v["run_id"]))[0] == adm

    # --- the EMITTED records: shards
    shard_recs = _records(a0 / PUB.SPOOL_REL) + _records(a1 / PUB.SPOOL_REL)
    for spool in (a0 / PUB.SPOOL_REL, a1 / PUB.SPOOL_REL, grading_host):
        _validate_all(spool)
    plan_idents = {i["run_id"]: i for plan in plans.values() for i in json.loads(plan.read_text()).values()}
    runs = [rec["payload"] for _p, rec, _s in shard_recs if rec["record_type"] == "run"]
    assert len(runs) == 8 and len({r["run_id"] for r in runs}) == 4                 # one run per (object, eval), seen by both shards
    for r in runs:
        want = plan_idents[r["run_id"]]
        assert (r["object_id"], r["eval_id"], r["producer_object_key_legacy"]) == (want["object_id"], want["eval_id"], want["producer_object_key_legacy"])
        assert r["producer_run_key_legacy"] == want["producer_run_key_legacy"] and r["extra"]["registered_at"] == "submit_wrapper"
        assert r["extra"]["seed_rule"] == "literal_training_seed" and r["lifecycle_state"] == "running"
    items = [rec["payload"] for _p, rec, _s in shard_recs if rec["record_type"] == "item_result"]
    assert len(items) == 2 * 2 * 2 * 10 and not any(i["extra"]["task"] == "judged/unrouted_task" for i in items)   # the unrouted row is not recorded
    for it in items:
        want = plan_idents[it["run_id"]]
        assert it["extra"]["task"] == want["extra"]["store_task"]                     # routed by the store's task name
        assert want["eval_id"] == {TASK_X: EVAL_X, TASK_Y: EVAL_Y}[it["extra"]["task"]]
    # no '*'-routed row lands on the wrong object: every run id in objA's spool dir tree is an objA run and vice versa
    for obj in OBJECTS:
        obj_runs = {i["run_id"] for i in json.loads(plans[obj].read_text()).values()}
        for shard in (a0, a1):
            for rd in (shard / PUB.SPOOL_REL / CAMPAIGN).iterdir():
                rid = next(rec["payload"]["run_id"] for _p, rec, _s in _records(rd))
                assert (rid in obj_runs) == (plan_idents[rid]["object_id"] == OBJECTS[obj][1]), rd
    assert not [rec for _p, rec, _s in shard_recs if rec["record_type"] == "run_state" and rec["payload"]["state"] == "generated"]

    # --- the EMITTED records: the grading host completion, exactly once per logical run, after the union check, DETERMINISTIC
    dev = _records(grading_host)
    gen = [rec["payload"] for _p, rec, _s in dev if rec["record_type"] == "run_state"]
    assert sorted(g["run_id"] for g in gen) == sorted(plan_idents) and all(g["state"] == "generated" for g in gen)
    patches = {rec["payload"]["run_id"]: rec["payload"]["fields"] for _p, rec, _s in dev if rec["record_type"] == "run_patch"}
    assert set(patches) == set(plan_idents) and all(f["n_items_generated"] == N_ITEMS for f in patches.values())
    assert all(sorted(f["extra_json"]["logical_run_shards"]) == ["att-0", "att-1"] for f in patches.values())
    by_run = {v["run_id"]: v for v in out}
    for _p, rec, _s in dev:
        v = by_run[rec["payload"]["run_id"]]
        assert rec["producer"]["script"] == LRC.SCRIPT and rec["producer"]["session"] == LRC.SESSION and rec["producer"]["code_sha"] == W
        assert rec["record_id"] == identity.adapter_record_id(rec["payload"], rec["record_type"])      # uuid5 over the payload: a retry mints the same id
        assert rec["record_id"] == v["completion"]["record_ids"][rec["record_type"]]
        assert rec["written_utc"] == v["completion"]["admission_utc"] == rec["payload"].get("event_utc", rec["written_utc"])
        assert rec["payload"].get("fields", {}).get("extra_json", {}).get("admission_digest", v["admission"]["digest"]) == v["admission"]["digest"]
    for rid in plan_idents:
        v = json.loads((verdicts / f"{identity.run_dir_name(rid)}.json").read_text())
        assert v["verdict"] == "ADMITTED" and v["n_union"] == 20 and [s["items_source"] for s in v["shards"]] == ["spool", "spool"]
        assert "noop" not in v                                                        # a return-value marker, never persisted
    assert all(Path(p).is_file() for v in out for p in v["emitted"])


def test_completion_is_resumable_and_idempotent_across_both_crash_windows(wave, monkeypatch):
    """Crash window 1 = after the verdict write, before any record; window 2 = between the two records."""
    plans, a0, a1, accepted = _two_shards(wave, monkeypatch)
    plan = plans["objA"]
    rid = json.loads(plan.read_text())[TASK_X]["run_id"]
    rd_name = identity.run_dir_name(rid)
    grading_host, verdicts = wave / "grading_host_spool", wave / "union"
    kw = dict(accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=verdicts, routes=[TASK_X], items_source="spool")
    real_write = LRC.write_record_once

    # ---- window 1: the verdict is durable, no record exists -> publication FORBIDDEN, no `generated` anywhere
    def crash_before_any(*a, **k):
        raise RuntimeError("crash before the first record")
    monkeypatch.setattr(LRC, "write_record_once", crash_before_any)
    with pytest.raises(RuntimeError, match="before the first record"):
        LRC.complete_plan_file(plan, **kw)
    pending = json.loads((verdicts / f"{rd_name}.json").read_text())
    assert pending["verdict"] == "ADMITTED" and pending["completion"]["state"] == PUB.COMPLETION_PENDING and pending["emitted"] == []
    assert set(pending["completion"]["record_ids"]) == {"run_state", "run_patch"} and set(pending["admission"]["attempts"]) == {"att-0", "att-1"}
    assert not (grading_host / CAMPAIGN / rd_name).exists()
    adm, why = PUB.admission_for(verdicts, rd_name)
    assert adm is None and why.startswith("ADMITTED but completion pending")
    attempts = PUB.discover_attempts([wave / "attempts"])
    rep = PUB.publish(attempts, PUB.load_accepted(accepted), inbox=wave / "inbox1", host="h", union_dir=verdicts)
    assert rep.published == [] and not (wave / "inbox1").exists()
    assert {s["reason"].split(" (")[0] for s in rep.skipped_run_dirs if s["run_dir"] == rd_name} == {"union_verdict:ADMITTED but completion pending"}
    # ---- resume: the SAME admission stamp, both records written once, then committed; publication now admitted
    monkeypatch.setattr(LRC, "write_record_once", real_write)
    (v,) = LRC.complete_plan_file(plan, **kw)
    assert v["completion"]["state"] == PUB.COMPLETION_COMMITTED and v["completion"]["admission_utc"] == pending["completion"]["admission_utc"]
    assert v["completion"]["record_ids"] == pending["completion"]["record_ids"] and not v.get("noop") and len(v["emitted"]) == 2
    gen, patches = _dev(grading_host, rid)
    assert len(gen) == 1 and len(patches) == 1
    rep = PUB.publish(attempts, PUB.load_accepted(accepted), inbox=wave / "inbox2", host="h", union_dir=verdicts)
    assert rep.published_attempt_ids() == {"att-0", "att-1"} and {p["run_dir"] for p in rep.published} == {rd_name} and len(rep.published) == 6

    # ---- window 2 (a second logical run): one record lands, the crash hits before the second
    rid2 = json.loads(plan.read_text())[TASK_Y]["run_id"]
    rd2 = identity.run_dir_name(rid2)
    kw2 = dict(kw, routes=[TASK_Y])
    calls = {"n": 0}

    def crash_after_first(run_dir, record_type, rec):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("crash between the records")
        return real_write(run_dir, record_type, rec)
    monkeypatch.setattr(LRC, "write_record_once", crash_after_first)
    with pytest.raises(RuntimeError, match="between the records"):
        LRC.complete_plan_file(plan, **kw2)
    pending2 = json.loads((verdicts / f"{rd2}.json").read_text())
    gen2, patches2 = _dev(grading_host, rid2)
    assert pending2["completion"]["state"] == PUB.COMPLETION_PENDING and len(gen2) == 1 and patches2 == []
    assert PUB.admission_for(verdicts, rd2)[0] is None
    monkeypatch.setattr(LRC, "write_record_once", real_write)
    (v2,) = LRC.complete_plan_file(plan, **kw2)          # finishes the partial emission: ONE more record, never a second `generated`
    gen2, patches2 = _dev(grading_host, rid2)
    assert v2["completion"]["state"] == PUB.COMPLETION_COMMITTED and len(gen2) == 1 and len(patches2) == 1
    assert gen2[0]["record_id"] == pending2["completion"]["record_ids"]["run_state"] and patches2[0]["record_id"] == pending2["completion"]["record_ids"]["run_patch"]
    files_after = sorted(p.name for p in (grading_host / CAMPAIGN / rd2).iterdir())
    assert len(files_after) == 2

    # ---- idempotent: a third pass is a no-op — no record, no file, the committed verdict returned as is
    (v3,) = LRC.complete_plan_file(plan, **kw2)
    assert v3["noop"] and v3["completion"] == v2["completion"] and v3["admission"]["digest"] == v2["admission"]["digest"]
    assert sorted(p.name for p in (grading_host / CAMPAIGN / rd2).iterdir()) == files_after
    assert len(_dev(grading_host, rid2)[0]) == 1

    # ---- KNOWN-BAD: a committed admission is never re-derived or overwritten
    before = (verdicts / f"{rd2}.json").read_bytes()
    only0 = _accepted(wave, {0: "att-0"}, name="accepted_only0.json")
    with pytest.raises(LRC.CompletionRefused, match="COMMITTED admission exists.*no longer supports"):       # the tree now REFUSES (10 != 20)
        LRC.complete_plan_file(plan, **dict(kw2, accepted_path=only0))
    with pytest.raises(LRC.CompletionRefused, match="never re-derived"):                                     # ADMITTED with another shard set
        LRC.complete_plan_file(plan, **dict(kw2, accepted_path=only0, expected_items=set(HALF[0])))
    assert (verdicts / f"{rd2}.json").read_bytes() == before and len(_dev(grading_host, rid2)[0]) == 1
    # a pending admission whose records exist may not be re-derived differently either
    (verdicts / f"{rd2}.json").write_text(json.dumps({**json.loads(before), "completion": {**pending2["completion"], "state": PUB.COMPLETION_PENDING}}))
    with pytest.raises(LRC.CompletionRefused, match="already put records"):
        LRC.complete_plan_file(plan, **dict(kw2, accepted_path=only0, expected_items=set(HALF[0])))
    # a dry run writes neither a verdict nor a record
    (v4,) = LRC.complete_plan_file(plans["none"], **dict(kw, dry_run=True))
    assert v4["verdict"] == "ADMITTED" and v4["completion"]["state"] == PUB.COMPLETION_PENDING and v4["emitted"] == []
    assert not (verdicts / f"{identity.run_dir_name(v4['run_id'])}.json").exists() and not (grading_host / CAMPAIGN / identity.run_dir_name(v4["run_id"])).exists()


def test_combined_controls_container_completes_all_three_logical_runs_from_one_accepted_attempt(wave, monkeypatch):
    """The controls CellStore (xstest + or_bench + mmlu) is ONE container; its accepted attempt completes three runs."""
    from sisteer.capeval.tiers.judged import TASK_ORBENCH, TASK_XSTEST
    from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU
    tasks = {TASK_XSTEST: ("xstest", "xstest"), TASK_ORBENCH: ("or_bench", "or_bench_hard"), PARITY_TASK_MMLU: ("mmlu", "mmlu")}

    def write_map(root: Path, container):
        m = {t: _ident("objA", eval_id, t, leg=leg, container=container) for t, (leg, eval_id) in tasks.items()}
        if container is None:
            for i in m.values():
                del i["extra"]["container"]
        p = root / "plans" / "_stores" / "controls" / "objA__s1.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        return p, m
    plan, m = write_map(wave, "controls")
    pf = RP.load_plan_file(plan)
    assert pf.container == "controls" and pf.leg == "mmlu+or_bench+xstest"
    a = _attempt_root(wave, 0, "att-c", leg="controls")                       # ONE container attempt: ATTEMPT.json leg = controls
    assert PUB.read_attempt(a).container == "controls"
    items = [f"q{i:03d}" for i in range(1, 21)]
    _drive_shard(a, {"objA": plan}, {"objA": items}, monkeypatch, tasks=tuple(tasks), unrouted=False)
    accepted = _accepted(wave, {0: "att-c"}, container="controls")
    grading_host, verdicts = wave / "grading_host_spool", wave / "union"
    out = LRC.complete_plan_file(plan, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=verdicts,
                                 expected_shards=1, items_source="spool")
    assert [v["verdict"] for v in out] == ["ADMITTED"] * 3 and sorted(v["leg"] for v in out) == ["mmlu", "or_bench", "xstest"]
    assert all(v["container"] == "controls" and set(v["admission"]["attempts"]) == {"att-c"} and v["n_union"] == 20 for v in out)
    assert all(v["completion"]["state"] == PUB.COMPLETION_COMMITTED and len(v["emitted"]) == 2 for v in out)
    for t, ident in m.items():
        gen, patches = _dev(grading_host, ident["run_id"])
        assert len(gen) == 1 and len(patches) == 1 and patches[0]["payload"]["fields"]["n_items_generated"] == 20
        assert patches[0]["payload"]["fields"]["extra_json"]["logical_run_shards"] == ["att-c"]
    # KNOWN-BAD: identities with NO container axis are refused at load — a container is never guessed from the leg
    wave2 = wave / "w2"
    plan2, _m2 = write_map(wave2, None)
    with pytest.raises(RP.RegistryPlanRefused, match="lacks extra.container"):
        LRC.complete_plan_file(plan2, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave2 / "dev", verdict_dir=wave2 / "union")
    assert not (wave2 / "dev").exists()
    # the LAUNCH PLAN join (seedrep_launch_plan/1): rows whose children name the plan file place it in the controls container
    def lp_row(stage: str, leg: str = "controls"):
        return {"standup_id": f"{leg}__{VIEW}__{stage}__shard00", "kind": "base" if stage == "base" else "rung",
                "control_key": f"{W7}/{WAVE}/{leg}/{VIEW}/{stage}/0",
                "children": [{"registry_identity_rel": "registry/plans/_stores/controls/objA__s1.json", "task": t} for t in tasks]}
    lp = wave / "launch_plan.json"
    lp.write_text(json.dumps({"schema": "seedrep_launch_plan/1", "standups": [lp_row("base")], "rungs": [lp_row("8192")]}))
    out3 = LRC.complete_plan_file(plan, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / "dev3", verdict_dir=wave / "union3",
                                  items_source="spool", launch_plan_path=lp)                       # expected_shards derived: 1 base row
    assert [v["verdict"] for v in out3] == ["ADMITTED"] * 3 and all(v["container"] == "controls" for v in out3)
    assert len(_records(wave / "dev3")) == 6
    # KNOWN-BAD: a launch plan that contradicts the plan file's container, or whose children never name the file, refuses
    lp_bad = wave / "launch_plan_bad.json"
    lp_bad.write_text(json.dumps({"standups": [lp_row("base", leg="ctl2")], "rungs": []}))
    with pytest.raises(LRC.CompletionRefused, match="places run .* in container 'ctl2'"):
        LRC.complete_plan_file(plan, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=wave / "union_bad",
                               items_source="spool", launch_plan_path=lp_bad)
    lp_none = wave / "launch_plan_none.json"
    row = lp_row("base")
    row["children"] = [{"registry_identity_rel": "registry/plans/pasttense/none__s1.json", "task": "static/pasttense"}]
    lp_none.write_text(json.dumps({"standups": [row]}))
    with pytest.raises(LRC.CompletionRefused, match="no row's children name plan file"):
        LRC.complete_plan_file(plan, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=wave / "union_bad",
                               items_source="spool", launch_plan_path=lp_none)
    # a row naming the file with a MALFORMED control_key refuses (an axis set is never guessed from a key that does not parse)
    lp_badkey = wave / "launch_plan_badkey.json"
    row_bk = lp_row("base")
    row_bk["control_key"] = "opaque-row-key"
    lp_badkey.write_text(json.dumps({"standups": [row_bk]}))
    with pytest.raises(LRC.CompletionRefused, match="no axis separator"):
        LRC.complete_plan_file(plan, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=wave / "union_bad",
                               items_source="spool", launch_plan_path=lp_badkey)
    # a dict naming NEITHER standups nor rungs is another schema (refused by shape, never read as an empty plan); so is a
    # standups value that is not a list of rows; an EMPTY plan is a real plan whose rows name nothing
    ok_row = json.dumps(lp_row("base"))
    for bad in ('{"rows": "nope"}', '{"standups": "nope"}', '{"standups": [1, 2]}', '"a string"', '[1, 2]', '{"standups": 1}', '{"standups": 0}',
                '{"standups": false}', '{"rungs": ""}', '{"standups": {}}', '{"standups": null}',
                f'{{"standups": [{ok_row}], "rungs": {{}}}}', f'{{"standups": [{ok_row}], "rungs": [1]}}', f'{{"standups": [{ok_row}], "rungs": 0}}'):
        (wave / "lp_shape.json").write_text(bad)
        with pytest.raises(LRC.CompletionRefused, match="seedrep_launch_plan/1"):
            LRC.complete_plan_file(plan, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=wave / "union_bad",
                                   items_source="spool", launch_plan_path=wave / "lp_shape.json")
    (wave / "lp_empty.json").write_text('{"standups": [], "rungs": []}')
    with pytest.raises(LRC.CompletionRefused, match="no row's children name plan file"):
        LRC.complete_plan_file(plan, accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=wave / "union_bad",
                               items_source="spool", launch_plan_path=wave / "lp_empty.json")
    assert not (wave / "union_bad").exists()                                   # no known-bad launch plan wrote a verdict


def test_overlapping_or_short_shards_are_refused_and_nothing_is_emitted(wave, monkeypatch):
    plans = _write_plans(wave)
    a0 = _attempt_root(wave, 0, "att-0")
    a1 = _attempt_root(wave, 1, "att-1")
    first = [f"q{i:03d}" for i in range(1, 11)]
    overlap = [f"q{i:03d}" for i in range(10, 20)]          # q010 in both shards, q020 missing
    _drive_shard(a0, plans, {"objA": first, "none": first}, monkeypatch)
    _drive_shard(a1, plans, {"objA": overlap, "none": overlap}, monkeypatch)
    accepted = _accepted(wave, {0: "att-0", 1: "att-1"})
    grading_host = wave / "grading_host_spool"
    out = LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host, verdict_dir=wave / "union")
    assert [v["verdict"] for v in out] == ["REFUSED", "REFUSED"]
    for v in out:
        assert any("not disjoint" in r and "q010" in r for r in v["reasons"]) and any("union holds 19" in r for r in v["reasons"])
        assert v["emitted"] == [] and v["admission"] is None and v["completion"] is None
        assert PUB.admission_for(wave / "union", identity.run_dir_name(v["run_id"])) == (None, "REFUSED")
    assert not grading_host.exists()
    # a short union alone (disjoint, one item missing) is refused too
    wave2 = wave / "w2"
    plans2 = _write_plans(wave2)
    b0 = _attempt_root(wave2, 0, "att-0")
    b1 = _attempt_root(wave2, 1, "att-1")
    _drive_shard(b0, plans2, {"objA": first, "none": first}, monkeypatch)
    _drive_shard(b1, plans2, {"objA": [f"q{i:03d}" for i in range(11, 20)], "none": ["q011"]}, monkeypatch)
    out2 = LRC.complete_plan_file(plans2["objA"], accepted_path=_accepted(wave2, {0: "att-0", 1: "att-1"}), attempt_roots=[wave2 / "attempts"],
                                  spool=wave2 / "grading_host_spool", verdict_dir=wave2 / "union")
    assert all(v["verdict"] == "REFUSED" and any("union holds 19" in r for r in v["reasons"]) for v in out2)
    # an explicit expected item set names the missing item
    out3 = LRC.complete_plan_file(plans2["objA"], accepted_path=_accepted(wave2, {0: "att-0", 1: "att-1"}), attempt_roots=[wave2 / "attempts"],
                                  spool=wave2 / "grading_host_spool", verdict_dir=wave2 / "union", expected_items={f"q{i:03d}" for i in range(1, 21)})
    assert all(any("missing from the union" in r and "q020" in r for r in v["reasons"]) for v in out3)
    assert not (wave2 / "grading_host_spool").exists()
    # shards of two different waves cannot be one admission
    wave3 = wave / "w3"
    plans3 = _write_plans(wave3)
    c0 = _attempt_root(wave3, 0, "att-0")
    c1 = _attempt_root(wave3, 1, "att-1", wave=WAVE2)
    _drive_shard(c0, plans3, {"objA": HALF[0], "none": HALF[0]}, monkeypatch)
    _drive_shard(c1, plans3, {"objA": HALF[1], "none": HALF[1]}, monkeypatch)
    acc3 = wave3 / "acc.json"
    acc3.write_text(json.dumps({f"{W7}/{WAVE}/{LEG}/{VIEW}/base/0": "att-0", f"{W7}/{WAVE2}/{LEG}/{VIEW}/base/1": "att-1"}))
    out4 = LRC.complete_plan_file(plans3["objA"], accepted_path=acc3, attempt_roots=[wave3 / "attempts"], spool=wave3 / "dev", verdict_dir=wave3 / "union")
    assert all(v["verdict"] == "REFUSED" and any("do not name one wave_uuid" in r for r in v["reasons"]) for v in out4) and not (wave3 / "dev").exists()


def test_a_shard_that_declared_completion_in_a_multi_shard_run_is_refused_and_a_single_store_is_not_doubled(wave, monkeypatch):
    plans = _write_plans(wave)
    a0 = _attempt_root(wave, 0, "att-0")
    a1 = _attempt_root(wave, 1, "att-1")
    _drive_shard(a0, plans, {"objA": HALF[0], "none": HALF[0]}, monkeypatch, finish=True)      # shard 0 wrongly emits `generated`
    _drive_shard(a1, plans, {"objA": HALF[1], "none": HALF[1]}, monkeypatch)
    out = LRC.complete_plan_file(plans["objA"], accepted_path=_accepted(wave, {0: "att-0", 1: "att-1"}), attempt_roots=[wave / "attempts"],
                                 spool=wave / "grading_host_spool", verdict_dir=wave / "union")
    assert all(v["verdict"] == "REFUSED" and any("emitted run_state generated themselves" in r for r in v["reasons"]) for v in out)
    assert not (wave / "grading_host_spool").exists()
    # a SINGLE-container store completed through its own hook: the union check admits it, COMMITS with no record of its own
    wave2 = wave / "single"
    plans2 = _write_plans(wave2)
    c0 = _attempt_root(wave2, 0, "att-0")
    allitems = [f"q{i:03d}" for i in range(1, 21)]
    _drive_shard(c0, plans2, {"objA": allitems, "none": allitems}, monkeypatch, finish=True)
    out2 = LRC.complete_plan_file(plans2["none"], accepted_path=_accepted(wave2, {0: "att-0"}), attempt_roots=[wave2 / "attempts"],
                                  spool=wave2 / "grading_host_spool", verdict_dir=wave2 / "union", expected_shards=1)
    assert all(v["verdict"] == "ADMITTED" and v["already_generated"] and v["emitted"] == [] and v["n_union"] == 20 for v in out2)
    assert all(v["completion"] == {"state": PUB.COMPLETION_COMMITTED, "admission_utc": v["completion"]["admission_utc"], "record_ids": {}} for v in out2)
    assert not (wave2 / "grading_host_spool").exists()
    gen = [rec for _p, rec, _s in _records(c0 / PUB.SPOOL_REL) if rec["record_type"] == "run_state" and rec["payload"]["state"] == "generated"]
    assert len(gen) == 4                       # the store's own, one per run; nothing added
    for v in out2:                             # …and publication is admitted against that committed admission
        assert PUB.admission_for(wave2 / "union", identity.run_dir_name(v["run_id"]))[1] == PUB.VERDICT_ADMITTED


def test_items_json_door_requires_the_native_rows_and_shard_selection_rules(wave, monkeypatch):
    """The sidecar ``registry/items/<run_dir>.json`` is the shard-item INTERFACE, never a substitute for the
    native ``item_result`` rows. The door admits only when the sidecar's set EQUALS the native set and the admission binds the native
    files. Known-bad: a sidecar-only shard (no native run dir); a sidecar claiming more, or fewer, than the rows; an empty native run
    dir with no run-state evidence; plus the shard-selection rules (superseded / incomplete attempts ignored, declared shard count,
    no sidecar under ``items_json``, a sidecar naming another run) — each REFUSED with nothing emitted."""
    plans = _write_plans(wave)
    a0 = _attempt_root(wave, 0, "att-0")
    a1 = _attempt_root(wave, 1, "att-1")
    sup = _attempt_root(wave, 0, "att-0-late", superseded_by="att-0")       # a superseded re-execution of shard 0: ignored
    inc = _attempt_root(wave, 1, "att-1-dead", complete=False)              # an incomplete attempt: ignored
    _drive_shard(a0, plans, {"objA": HALF[0], "none": HALF[0]}, monkeypatch)   # the NATIVE rows, both objects, both tasks
    _drive_shard(a1, plans, {"objA": HALF[1], "none": HALF[1]}, monkeypatch)
    ident_x = json.loads(plans["none"].read_text())[TASK_X]
    rid_x, rdn = ident_x["run_id"], identity.run_dir_name(ident_x["run_id"])

    def sidecar(a: Path, ids, run_id: str = rid_x):
        d = a / PUB.ITEMS_REL
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{rdn}.json").write_text(json.dumps({"run_id": run_id, "item_ids": list(ids)}))
    for a, ids in ((a0, HALF[0]), (a1, HALF[1]), (sup, HALF[0] + HALF[1]), (inc, HALF[1])):
        sidecar(a, ids)
    accepted = _accepted(wave, {0: "att-0", 1: "att-1"})
    kw = dict(accepted_path=accepted, attempt_roots=[wave / "attempts"], routes=[TASK_X], items_source="items_json")
    (v,) = LRC.complete_plan_file(plans["none"], spool=wave / "grading_host_spool", verdict_dir=wave / "union", expected_shards=2, **kw)
    assert v["verdict"] == "ADMITTED" and [s["attempt_id"] for s in v["shards"]] == ["att-0", "att-1"] and v["n_union"] == 20
    assert [s["items_source"] for s in v["shards"]] == ["items_json", "items_json"]
    for aid in ("att-0", "att-1"):                                  # the admission binds the NATIVE record files, never an empty set
        files = v["admission"]["attempts"][aid]["files"]
        assert files and all(R.RECORD_FILE_RE.match(Path(f).name) for f in files)
    gen, patches = _dev(wave / "grading_host_spool", rid_x)
    assert len(gen) == 1 and len(patches) == 1 and patches[0]["payload"]["fields"]["n_items_generated"] == 20
    # KNOWN-BAD 1: sidecars ALONE (no native run dir) certify nothing — the exact shape the old regression admitted with an empty file set
    w1 = wave / "sideonly"
    plans1 = _write_plans(w1)
    b0, b1 = _attempt_root(w1, 0, "att-0"), _attempt_root(w1, 1, "att-1")
    for a, ids in ((b0, HALF[0]), (b1, HALF[1])):
        sidecar(a, ids)
    (v1,) = LRC.complete_plan_file(plans1["none"], accepted_path=_accepted(w1, {0: "att-0", 1: "att-1"}), attempt_roots=[w1 / "attempts"],
                                   spool=w1 / "dev", verdict_dir=w1 / "union", routes=[TASK_X], items_source="items_json", expected_shards=2)
    assert v1["verdict"] == "REFUSED" and any("no native run directory" in r and "alone certifies nothing" in r for r in v1["reasons"])
    assert not (w1 / "dev").exists()
    # KNOWN-BAD 2 / 3: a sidecar claiming MORE than the rows, then FEWER — each refused naming the difference
    sidecar(a0, HALF[0] + ["q999"])
    (v2,) = LRC.complete_plan_file(plans["none"], spool=wave / "dev2", verdict_dir=wave / "union2", expected_shards=2, **kw)
    assert v2["verdict"] == "REFUSED" and any("sidecar-only 1, e.g. ['q999']" in r and "native-only 0" in r for r in v2["reasons"])
    sidecar(a0, HALF[0][:-1])
    (v3,) = LRC.complete_plan_file(plans["none"], spool=wave / "dev3", verdict_dir=wave / "union3", expected_shards=2, **kw)
    assert v3["verdict"] == "REFUSED" and any("native-only 1, e.g. ['q010']" in r for r in v3["reasons"])
    sidecar(a0, HALF[0])
    # KNOWN-BAD 4: an EMPTY native run dir with no run_state is a shard that recorded nothing, not an empty shard — even when allowed
    w4 = wave / "emptydir"
    plans4 = _write_plans(w4)
    c0 = _attempt_root(w4, 0, "att-0")
    (c0 / PUB.SPOOL_REL / CAMPAIGN / rdn).mkdir(parents=True)
    sidecar(c0, [])
    (v4,) = LRC.complete_plan_file(plans4["none"], accepted_path=_accepted(w4, {0: "att-0"}), attempt_roots=[w4 / "attempts"], spool=w4 / "dev",
                                   verdict_dir=w4 / "union", routes=[TASK_X], items_source="items_json", expected_shards=1, allow_empty_shard=True)
    assert v4["verdict"] == "REFUSED" and any("no item_result rows and no run_state" in r for r in v4["reasons"]) and not (w4 / "dev").exists()
    # KNOWN-BAD 5 / 6 / 7 (the selection rules): the declared shard count disagrees; no sidecar for a run under items_json; a sidecar
    # naming another run
    (v5,) = LRC.complete_plan_file(plans["none"], spool=wave / "dev5", verdict_dir=wave / "union5", expected_shards=3, **kw)
    assert v5["verdict"] == "REFUSED" and any("declares 3" in r for r in v5["reasons"])
    (v6,) = LRC.complete_plan_file(plans["none"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / "dev6",
                                   verdict_dir=wave / "union6", routes=[TASK_Y], items_source="items_json")
    assert v6["verdict"] == "REFUSED" and any("no registry/items" in r for r in v6["reasons"])
    sidecar(a0, HALF[0], run_id="other:run:id:0000")
    (v7,) = LRC.complete_plan_file(plans["none"], spool=wave / "dev7", verdict_dir=wave / "union7", **kw)
    assert v7["verdict"] == "REFUSED" and any("not {run_id" in r for r in v7["reasons"])
    for d in ("dev2", "dev3", "dev5", "dev6", "dev7"):
        assert not (wave / d).exists()
    # with no sidecar at all the door is the rows themselves (source auto -> spool)
    sidecar(a0, HALF[0])
    for a in (a0, a1):
        (a / PUB.ITEMS_REL / f"{rdn}.json").unlink()
    (v8,) = LRC.complete_plan_file(plans["none"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / "dev8",
                                   verdict_dir=wave / "union8", routes=[TASK_X], expected_shards=2)
    assert v8["verdict"] == "ADMITTED" and [s["items_source"] for s in v8["shards"]] == ["spool", "spool"] and v8["n_union"] == 20


def test_two_overlapping_completion_passes_emit_exactly_one_generated(wave, monkeypatch):
    """The known-bad is the race itself: two completion passes for ONE logical run, both past the pure evaluation
    and both about to read the verdict and the spool at the same instant (a barrier holds them there). Under the per-run claim (an
    flock keyed on the verdict path, held through the pending write, the record writes, the disk verification and the commit) the
    loser reads the winner's COMMITTED verdict and returns it as a no-op: exactly one `generated`, one patch, both passes ADMITTED,
    both took the claim, the loser's verdict IS the winner's."""
    import threading
    plans, _a0, _a1, accepted = _two_shards(wave, monkeypatch)
    ident_x = json.loads(plans["objA"].read_text())[TASK_X]
    grading_host, union = wave / "grading_host_spool", wave / "union"
    barrier = threading.Barrier(2, timeout=60)
    orig_eval, orig_claim, claims = LRC.evaluate_run, LRC._completion_claim, []

    def gated_eval(*a, **k):
        v = orig_eval(*a, **k)
        barrier.wait()                                   # both passes have evaluated; now both race for the claim
        return v

    @contextlib.contextmanager
    def counted_claim(verdict_dir, run_dir):
        with orig_claim(verdict_dir, run_dir):
            claims.append(threading.get_ident())
            yield
    monkeypatch.setattr(LRC, "evaluate_run", gated_eval)
    monkeypatch.setattr(LRC, "_completion_claim", counted_claim)
    results, errors = {}, {}

    def worker(tag):
        try:
            results[tag] = LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=grading_host,
                                                  verdict_dir=union, routes=[TASK_X], items_source="spool")[0]
        except BaseException as exc:  # noqa: BLE001 — re-raised below, on the test thread
            errors[tag] = exc
    threads = [threading.Thread(target=worker, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors and all(not t.is_alive() for t in threads), errors
    va, vb = results["a"], results["b"]
    assert va["verdict"] == vb["verdict"] == "ADMITTED" and sorted([bool(va.get("noop")), bool(vb.get("noop"))]) == [False, True]
    assert len(claims) == 2 and len(set(claims)) == 2              # both passes took the claim, each from its own thread
    winner, loser = (va, vb) if not va.get("noop") else (vb, va)
    assert {k: v for k, v in loser.items() if k != "noop"} == json.loads(json.dumps(winner))   # the loser returned the winner's committed verdict
    gen, patches = _dev(grading_host, ident_x["run_id"])
    assert len(gen) == 1 and len(patches) == 1 and patches[0]["payload"]["fields"]["n_items_generated"] == 20
    v = PUB.read_verdict(union, identity.run_dir_name(ident_x["run_id"]))
    on_disk = LRC.record_ids_on_disk(grading_host / CAMPAIGN / identity.run_dir_name(ident_x["run_id"]))
    assert v["completion"]["state"] == PUB.COMPLETION_COMMITTED and set(v["completion"]["record_ids"].values()) == set(on_disk) and len(on_disk) == 2
    assert (union / f"{identity.run_dir_name(ident_x['run_id'])}.lock").is_file()      # the claim's lock file, keyed on the verdict path
    # a dry run takes no claim and writes nothing (no lock file, no verdict) — checked on a fresh verdict dir
    monkeypatch.setattr(LRC, "evaluate_run", orig_eval)
    claims.clear()
    (vd,) = LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / "dev_dry",
                                   verdict_dir=wave / "union_dry", routes=[TASK_X], items_source="spool", dry_run=True)
    assert vd["verdict"] == "ADMITTED" and claims == [] and not (wave / "union_dry").exists() and not (wave / "dev_dry").exists()


def test_a_record_naming_another_run_or_no_run_refuses_the_shard_and_nothing_is_emitted(wave, monkeypatch):
    """EVERY record of every file in a shard's run dir must carry payload.run_id == the logical run. A file whose
    first row is this run and a later row ANOTHER run, or a row with no run_id, refuses the shard — the file is never hashed into an
    admission (the old scan skipped the foreign row and admitted the file). The other logical runs of the same shards still admit."""
    plans, _a0, a1, accepted = _two_shards(wave, monkeypatch)
    ident_x = json.loads(plans["objA"].read_text())[TASK_X]
    other = json.loads(plans["none"].read_text())[TASK_X]["run_id"]
    rd = a1 / PUB.SPOOL_REL / CAMPAIGN / identity.run_dir_name(ident_x["run_id"])
    prod = synth.producer(substrate="modal")
    mixed = rd / R.record_file_name(900, "run_patch")
    R.write_records(str(mixed), [R.make_record("run_patch", {"run_id": ident_x["run_id"], "fields": {"node_hostname": "a"}}, prod, "11111111-1111-4111-8111-111111111111", synth.t(1)),
                                 R.make_record("run_patch", {"run_id": other, "fields": {"node_hostname": "b"}}, prod, "22222222-2222-4222-8222-222222222222", synth.t(2))])
    kw = dict(accepted_path=accepted, attempt_roots=[wave / "attempts"], items_source="spool")
    (v,) = LRC.complete_plan_file(plans["objA"], spool=wave / "dev", verdict_dir=wave / "union", routes=[TASK_X], **kw)
    assert v["verdict"] == "REFUSED" and any(f"{mixed}: record 2 (run_patch) names run {other!r}" in r and "never admitted" in r for r in v["reasons"])
    assert not (wave / "dev").exists()
    (vy,) = LRC.complete_plan_file(plans["objA"], spool=wave / "dev_y", verdict_dir=wave / "union_y", routes=[TASK_Y], **kw)
    assert vy["verdict"] == "ADMITTED" and vy["n_union"] == 20            # the sibling logical run in the same shard dirs is untouched
    mixed.unlink()
    (rd / R.record_file_name(901, "run_patch")).write_text(json.dumps(R.make_record("run_patch", {"fields": {}}, prod, "33333333-3333-4333-8333-333333333333", synth.t(3))) + "\n")
    (v2,) = LRC.complete_plan_file(plans["objA"], spool=wave / "dev2", verdict_dir=wave / "union2", routes=[TASK_X], **kw)
    assert v2["verdict"] == "REFUSED" and any("record 1 (run_patch) names run None" in r for r in v2["reasons"]) and not (wave / "dev2").exists()
    (rd / R.record_file_name(901, "run_patch")).unlink()
    # an ARTIFACT binds through extra.run_id (the store recorder's completion / rows artifacts; the single-container test above admits the
    # store's own): bound to another run, or unbound, it refuses; so does a record type with no run binding at all (object)
    def art(run_id, aid):
        return R.make_record("artifact", {"artifact_id": "f" * 64, "kind": "rows", "uri_primary": f"/x/{aid}", "bytes": 1,
                                          "extra": ({"run_id": run_id} if run_id is not None else {})}, prod, aid, synth.t(4))
    for n, (bad, why) in enumerate(((art(other, "55555555-5555-4555-8555-555555555555"), f"(artifact) names run {other!r}"),
                                    (art(None, "66666666-6666-4666-8666-666666666666"), "(artifact) names run None"),
                                    (R.make_record("object", {"object_id": "o" * 64}, prod, "77777777-7777-4777-8777-777777777777", synth.t(5)), "(object) names run None"))):
        f = rd / R.record_file_name(902 + n, bad["record_type"])
        R.write_records(str(f), [bad])
        (vb,) = LRC.complete_plan_file(plans["objA"], spool=wave / f"dev_b{n}", verdict_dir=wave / f"union_b{n}", routes=[TASK_X], **kw)
        assert vb["verdict"] == "REFUSED" and any(why in r for r in vb["reasons"]) and not (wave / f"dev_b{n}").exists()
        f.unlink()
    R.write_records(str(rd / R.record_file_name(905, "artifact")), [art(ident_x["run_id"], "88888888-8888-4888-8888-888888888888")])
    (v3,) = LRC.complete_plan_file(plans["objA"], spool=wave / "dev3", verdict_dir=wave / "union3", routes=[TASK_X], **kw)
    assert v3["verdict"] == "ADMITTED" and v3["n_union"] == 20              # the clean dir (+ the run's own artifact) admits
    assert R.record_file_name(905, "artifact") in v3["admission"]["attempts"]["att-1"]["files"]


def test_a_legacy_attempt_missing_stage_or_shard_fails_closed_before_any_verdict_or_record(wave, monkeypatch):
    """A spelled-out ATTEMPT.json that OMITS an axis could authorise completion — `axis_mismatch` wildcarded the absent axis
    against a full accepted key, `shards_of` read `stage None` as base, `admission_block` checked only w7/wave — and `generated` + the
    patch reached the grading host's spool without base-stage provenance. Now: an accepted attempt of this run's (container, view) that
    does not prove all six axes REFUSES the pass as an exception before evaluation writes anything (no verdict file, no verdict dir, no
    spool); `admission_block` asserts the same independently; the same tree with the axes stated admits."""
    plans, a0, a1, accepted = _two_shards(wave, monkeypatch)
    meta_path = a1 / PUB.ATTEMPT_FILE
    full = json.loads(meta_path.read_text())
    for axis in ("stage", "shard"):
        meta_path.write_text(json.dumps({k: v for k, v in full.items() if k != axis}))          # the legacy shape, one axis omitted
        a = PUB.read_attempt(a1)
        assert a.axes[axis] is None and PUB.absent_or_invalid_axes(a.axes) == [axis]
        assert PUB.load_accepted(accepted).is_accepted(a) == (False, f"ATTEMPT.json states no ['{axis}']; an attempt is accepted only on an exact "
                                                                     "six-axis match with its key")
        for routes in ([TASK_X], None):
            with pytest.raises(LRC.CompletionRefused, match=f"states no \\['{axis}'\\]; an attempt that cannot prove its axes never completes a run"):
                LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / f"dev_{axis}",
                                       verdict_dir=wave / f"union_{axis}", routes=routes, items_source="spool")
        assert not (wave / f"union_{axis}").exists() and not (wave / f"dev_{axis}").exists()      # NO verdict, NO lock, NO record
        # `shards_of` alone would also exclude it (stage None is not base; the accept fails), and `admission_block` refuses it independently
        assert LRC.shards_of([PUB.read_attempt(a0), a], PUB.load_accepted(accepted), LEG, VIEW) == [PUB.read_attempt(a0)]
        with pytest.raises(LRC.CompletionRefused, match=f"states no \\['{axis}'\\]"):
            LRC.admission_block({"run_id": "r", "campaign": CAMPAIGN, "code_sha": W}, LEG, VIEW, [LRC.ShardItems(a, set(), "spool", [], {})])
    # a STATED non-base stage is the completion's business, not the admission's: shards_of excludes it (an ordinary REFUSED union — the
    # attempt proves its axes, it is just not a base shard), while admission_block admits a fully-axed rung (a publication admission
    # legitimately binds a base attempt and a rung attempt of one run, P1-2)
    meta_path.write_text(json.dumps({**full, "stage": "8192"}))
    (v,) = LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / "dev_rung",
                                  verdict_dir=wave / "union_rung", routes=[TASK_X], items_source="spool")
    assert v["verdict"] == "REFUSED" and not (wave / "dev_rung").exists()
    assert LRC.shards_of([PUB.read_attempt(a0), PUB.read_attempt(a1)], PUB.load_accepted(accepted), LEG, VIEW) == [PUB.read_attempt(a0)]
    rung_adm = LRC.admission_block({"run_id": "r", "campaign": CAMPAIGN, "code_sha": W}, LEG, VIEW, [LRC.ShardItems(PUB.read_attempt(a1), set(), "spool", [], {})])
    assert rung_adm["attempts"]["att-1"]["stage"] == "8192"
    # control: the same tree with every axis stated admits and completes
    meta_path.write_text(json.dumps(full))
    (v,) = LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / "dev_ok",
                                  verdict_dir=wave / "union_ok", routes=[TASK_X], items_source="spool")
    assert v["verdict"] == "ADMITTED" and v["completion"]["state"] == PUB.COMPLETION_COMMITTED and len(v["emitted"]) == 2


def test_a_spelled_out_attempt_with_a_coercible_axis_value_is_refused_at_read_and_never_completes(wave, monkeypatch):
    """The legacy spelled-out ATTEMPT.json's shard went through `int(v)` — `1.5` and `true` became shard 1, `-0.5` became 0 —
    BEFORE `absent_or_invalid_axes` ran, so the coerced value passed the exact comparison against an accepted key and could authorise
    completion under an axis the attempt never stated (the string axes were `str()`-coerced likewise). Now every stated field is read
    exactly: `read_attempt` REFUSES the root naming the field and value, and a completion pass over that tree refuses before any verdict,
    lock or spool record is written; the same values stated legitimately (`1`, `"1"`) admit."""
    plans, _a0, a1, accepted = _two_shards(wave, monkeypatch)
    meta_path = a1 / PUB.ATTEMPT_FILE
    full = json.loads(meta_path.read_text())
    assert full["shard"] == 1 and "control_key" not in full                     # the spelled-out legacy shape, shard a JSON integer
    bads = [("shard", 1.5), ("shard", True), ("shard", -0.5), ("shard", "01"), ("shard", " 1"), ("shard", -1), ("shard", "x"), ("shard", 1.0),
            ("stage", True), ("view", 1.5), ("leg", ["ctl"]), ("W", "not-hex"), ("wave_uuid", 42),
            # `$` matched before a final newline and a trailing space was trimmed — a true full match, no trimming
            ("shard", "1\n"), ("shard", "1 "), ("W", W + "\n"), ("W", " " + W), ("stage", "base\n"), ("view", " " + VIEW), ("leg", LEG + " ")]
    for n, (field, value) in enumerate(bads):
        meta_path.write_text(json.dumps({**full, field: value}))
        with pytest.raises(PUB.PublishRefused, match=f"{'shard is' if field == 'shard' else field} .*never coerced|W .* is not 7..40 hex") as ei:
            PUB.read_attempt(a1)
        assert repr(value) in str(ei.value) or field == "W"                        # the refusal names the value as written
        if isinstance(value, str) and value != value.strip():
            assert "whitespace" in str(ei.value) or "not a non-negative integer" in str(ei.value)
        with pytest.raises(PUB.PublishRefused):                                     # the pulled tree is refused whole, before any evaluation
            LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / f"dev_{n}",
                                   verdict_dir=wave / f"union_{n}", routes=[TASK_X], items_source="spool")
        assert not (wave / f"union_{n}").exists() and not (wave / f"dev_{n}").exists()   # NO verdict, NO lock, NO record
    # a JSON-LIST control_key's members are read in their stated types — a numeric W is not the hex string it prints
    # as; the same for a numeric wave_uuid, a boolean stage, a floating view, a list-valued container; nothing is str()-coerced
    ck = [W[:7], WAVE, LEG, VIEW, "base", 1]
    for n, (idx, value, axis) in enumerate(((0, 1234567, "W7"), (1, 11111111, "wave_uuid"), (4, True, "stage"), (3, 1.5, "view"), (2, [LEG], "container"))):
        bad_key = list(ck); bad_key[idx] = value
        meta_path.write_text(json.dumps({**full, "control_key": bad_key}))
        with pytest.raises(PUB.PublishRefused, match=f"{axis} is {_re_esc(repr(value))} \\({type(value).__name__}\\), not a string") as ei:
            PUB.read_attempt(a1)
        assert "malformed control_key is refused" in str(ei.value)
        with pytest.raises(PUB.PublishRefused):
            LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / f"dev_l{n}",
                                   verdict_dir=wave / f"union_l{n}", routes=[TASK_X], items_source="spool")
        assert not (wave / f"union_l{n}").exists() and not (wave / f"dev_l{n}").exists()
    meta_path.write_text(json.dumps({**full, "control_key": ck}))                 # the typed list key reads: shard 1, W7 a string
    assert PUB.read_attempt(a1).axes == {"w7": W[:7], "wave_uuid": WAVE, "container": LEG, "view": VIEW, "stage": "base", "shard": 1}
    # a numeric attempt_id / a boolean superseded_by are refused too — the id is compared against the accepted map's values
    for field, value in (("attempt_id", 7), ("attempt_id", " att-1"), ("superseded_by", True), ("superseded_by", "")):
        meta_path.write_text(json.dumps({**full, field: value}))
        with pytest.raises(PUB.PublishRefused, match=f"{field} is {_re_esc(repr(value))}"):
            PUB.read_attempt(a1)
    # the same values stated legitimately read as shard 1 and admit — a JSON integer and a canonical decimal string alike
    for n, value in enumerate((1, "1")):
        meta_path.write_text(json.dumps({**full, "shard": value}))
        assert PUB.read_attempt(a1).shard == 1 and PUB.absent_or_invalid_axes(PUB.read_attempt(a1).axes) == []
        (v,) = LRC.complete_plan_file(plans["objA"], accepted_path=accepted, attempt_roots=[wave / "attempts"], spool=wave / f"dev_ok{n}",
                                      verdict_dir=wave / f"union_ok{n}", routes=[TASK_X], items_source="spool")
        assert v["verdict"] == "ADMITTED" and v["completion"]["state"] == PUB.COMPLETION_COMMITTED
    meta_path.write_text(json.dumps(full))


def test_cli_returns_nonzero_on_a_refused_union(wave, monkeypatch, capsys):
    plans = _write_plans(wave)
    a0 = _attempt_root(wave, 0, "att-0")
    _drive_shard(a0, plans, {"objA": [f"q{i:03d}" for i in range(1, 6)], "none": [f"q{i:03d}" for i in range(1, 6)]}, monkeypatch)
    accepted = _accepted(wave, {0: "att-0"})
    rc = LRC.main(["--plan", str(plans["objA"]), "--accepted", str(accepted), "--attempts", str(wave / "attempts"), "--spool", str(wave / "dev"),
                   "--verdict-dir", str(wave / "union"), "--items-source", "spool"])
    assert rc == 1
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert all(l["verdict"] == "REFUSED" and l["n_union"] == 5 and l["expected_n"] == 20 and l["completion"] is None for l in lines)
    rc = LRC.main(["--plan", str(plans["objA"]), "--accepted", str(accepted), "--attempts", str(wave / "attempts"), "--spool", str(wave / "dev"),
                   "--verdict-dir", str(wave / "union"), "--items-source", "spool", "--route", "judged/nope"])
    assert rc == 2
