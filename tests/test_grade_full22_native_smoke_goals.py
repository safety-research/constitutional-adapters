"""The multi-turn grader under the SMOKE goal-shard knob: the REAL ``main()`` of
``scripts/sdfbase/seedrep/grade_full22_native.py`` under ``--worklist registry`` over a synthetic native store laid out as the smoke's
job writes it — shard00 ALONE of N=99, holding goals[0::99] of the config's 259-goal eval set (3 goals) — whose run row is stamped
``extra.run_overrides.keys.OVERRIDE_SMOKE_GOALS=0-0/99`` (the central stamp every smoke row carries) and whose producer key bakes
``sh0of99``. Before this the grader's completeness gate refused the store as "3/259 goals have a progress record" unless an operator
supplied ``--goalset-dir`` by hand (which no production command does: grade_daemon.DEFAULT_TEMPLATES).

Now the stamped pair is graded in GOALSET mode over the config's eval set dealt goals[s::N] across shards lo..hi (the planner's own dealing,
``sisteer.eval.smoke_goals``), the store's shards held to exactly lo..hi, the producer key required to spell the same shards; the aggregate
row, completion.json and grading_provenance.json carry the stamp's provenance under ``goalset``; the judgment completes over 3 goals and
the run is ``judged``. Refused by name: the same store with no stamp (the old refusal, unchanged); a stamp whose shards the producer key
does not spell; a shard outside the stamped range; ``--goalset-dir`` beside a stamped run. The world, the synthetic registry and the
stand-in judge are tests/test_grade_full22_native_main.py's (imported), so the shape pinned there is the shape pinned here."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grade_full22_native_main as T  # noqa: E402  (the world, the synthetic registry, the stub judge, the real main())

from sisteer.eval import smoke_goals as SG  # noqa: E402
from sisteer.registry import db as DB  # noqa: E402

# pytest collects fixtures from THIS module only: the imported module's are re-exported by name (module-scoped ones stay module-scoped here)
TW = T._load("gfn_smoke_twin", T.TWIN)   # this module's own copy of the grader (never the other module's fixture object)
jbb_ids = T.jbb_ids
world = T.world
stub_judge = T.stub_judge

bf = T.bf
CELL = T.CELL
VIEW = "s1"
N = 99
SPEC = "0-0/99"
STAMP = {"id": "demo_smoke", "path": "demo_smoke.env", "sha256": "e" * 64,
         "keys": {"OVERRIDE_ID": "demo_smoke", "OVERRIDE_MODEL": "qwen35_4b_demo", SG.KEY: SPEC, "OVERRIDE_HPO_TRIALS": "16"}}
KEY_OK = f"sdfbase_full22-att-crescendo-{T.tag(VIEW)}-sh0of{N}"
KEY_WHOLE = f"sdfbase_full22-att-crescendo-{T.tag(VIEW)}-sh0_1of9"


@pytest.fixture(scope="module")
def eval_ids():
    """The config's 259 eval goal ids in resolver order (the grader resolves the same list); the smoke keeps indices 0, 99, 198."""
    import jailbreak_run as jr
    import yaml
    cfg = yaml.safe_load(T.SRC_CFG.read_text())
    ids = [g.goal_id for g in jr._eval_goals(cfg, confirmatory=True)]
    assert len(ids) == 259
    return ids


def _smoke_store(root: Path, view: str, *, goal_ids: list, shards=(0,), attack: str = "crescendo") -> Path:
    """A native multi-turn store as the smoke job leaves it: shard<s> for each s in `shards`, holding goals[s::N] — the producers' five files
    per shard, the twin fixture builder's stamps (native engine, attacker record, served-target manifest)."""
    d = root / T.run_of(view) / attack / CELL
    decode = {"do_sample": True, "temperature": 1.0, "seed": None, "top_p": None, "top_k": None}
    for s in shards:
        sd = d / f"shard{s:02d}"
        gids = goal_ids[s::N]
        rows, prog, timing = [], [], []
        for gid in gids:
            rows += [{"goal_id": gid, "order": k, "response": f"synthetic {attack} reply {k} for {gid} on {bf.NATIVE_ENGINE}", "decode": decode,
                      **bf._engine_stamp(bf.NATIVE_ENGINE)} for k in range(2)]
            prog.append({"goal_id": gid, "category": bf.GOAL_CATEGORY, "n_target_forwards": 2, "outcome": "AttackOutcome.FAILURE", "turns": 2, "capped": False})
            timing += [{"t_start": 1.0 + k, "latency_s": 0.5, "virtual_model": "petri__si_constv2_mix_full__psr_ce_all", "n_messages": 1 + 2 * k,
                        "max_tokens": 4096, "prompt_tokens": 40, "completion_tokens": 30, "finish_reason": "stop",
                        "intervention_hash": bf.INTERVENTION_HASH, "decode": decode} for k in range(2)]
        bf._jl(sd / "responses.jsonl", rows)
        bf._jl(sd / "progress.jsonl", prog)
        bf._jl(sd / "target_timing.jsonl", timing)
        bf._js(sd / "attacker_engine.json", bf.attacker_engine_record(attack))
        bf._js(sd / "served_target.json", bf.served_target_manifest(bf.NATIVE_ENGINE, cell=CELL, attack=attack))
    return d


def _world(tmp_path, world, eval_ids, *, shards=(0,)):
    """The imported world's configs / roster / models table, with THIS test's smoke store under its own private root for view s1."""
    store_root = tmp_path / "private"
    _smoke_store(store_root, VIEW, goal_ids=eval_ids, shards=shards)
    return store_root


def _argv(world, store_root: Path, root: str, grades: Path, *, run_ids=(), extra=()):
    a = ["--target", T.tag(VIEW), "--config", str(world["configs"][VIEW]), "--private-root", str(store_root), "--grades-root", str(grades),
         "--attacks", "crescendo", "--cells", CELL, "--require-engine", "lora_native",
         "--roster", str(world["roster"]), "--models-table", str(world["models_table"]),
         "--worklist", "registry", "--registry-root", root, "--session", "s-A"]
    for rid in run_ids:
        a += ["--run-id", rid]
    return a + list(extra)


def _registry(tmp_path, monkeypatch, store: Path, *, stamp=STAMP, key: str = KEY_OK):
    """A synthetic registry (the imported module's dimensions) holding ONE `generated` crescendo run on `store`, registered as the smoke job
    registers it: the producer key with its shard segment, n_items_planned 259 (the item set), n_items_generated 3 (the closer's subset),
    and — unless `stamp` is None — the central run-overrides stamp under the run record's `extra` (folded into run.extra_json)."""
    root = str(tmp_path / "reg")
    T.H.ensure_layout(root)
    T.R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), T._dims())
    inst = T._instrument("crescendo")
    p = T.synth.run_payload(key, CELL, T.synth.OBJECT_SHA, 200, eval_id="crescendo", item_set_id="crescendo_set", instrument_id=inst["instrument_id"],
                            decode_source="generation_config", store_uri=str(store), model_id=T.synth.MODEL_ID, serving_path="lora_native", n_items_planned=259)
    if stamp is not None:
        p["extra"] = {"run_overrides": dict(stamp)}
    p["run_id"] = T.I.run_id(T.synth.CAMPAIGN, key, CELL, "crescendo", str(inst["instrument_id"]))
    rid = str(p["run_id"]); t = T.synth.t
    recs = [T.synth.rec("run", p, t(200)),
            T.synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(200), "evidence_kind": "submit_wrapper"}, t(200)),
            T.synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(201), "evidence_kind": "job_log"}, t(201)),
            T.synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(203), "evidence_kind": "completion_artifact", "evidence_uri": str(store)}, t(203)),
            T.synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": 3}}, t(204))]
    T.R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    rep = T.H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with T._conn(root) as conn:
        got = conn.execute("SELECT json_extract(extra_json, '$.run_overrides.keys.OVERRIDE_SMOKE_GOALS'), producer_run_key_legacy FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert tuple(got) == ((stamp or {}).get("keys", {}).get(SG.KEY), key), tuple(got)   # the stamp and the key are on the folded row, as in production
    monkeypatch.delenv(T.emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv("REGISTRY_ROOT", raising=False)
    return root, rid


def test_a_stamped_smoke_run_grades_over_its_three_goals_and_the_outputs_carry_the_stamp(tmp_path, monkeypatch, world, eval_ids, stub_judge):
    store_root = _world(tmp_path, world, eval_ids)
    store = store_root / T.run_of(VIEW) / "crescendo" / CELL
    root, rid = _registry(tmp_path, monkeypatch, store)
    grades = tmp_path / "grades"
    TW.main(_argv(world, store_root, root, grades, run_ids=[rid]))
    T.H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default)
    # the registry: ONE judgment, complete over 3 goals; the run judged
    with T._conn(root) as conn:
        js = T._judgments(conn, rid)
        assert len(js) == 1 and (js[0]["status"], js[0]["n_items_judged"]) == ("complete", 3), js
        assert T._run_state(conn, rid) == ("judged", 3, T.synth.JUDGE_ID)
        scores = T._items(conn, rid, "strongreject_score")
        assert {s[0] for s in scores} == {eval_ids[0], eval_ids[99], eval_ids[198]} and len(scores) == 3 * T.N_TURNS
        m = T._metrics(conn, rid)
        assert m[("asr", 0.5)]["n"] == 3
    # the outputs: the aggregate row's goal set is the stamp's, completion.json and the provenance name it
    out = grades / T.tag(VIEW)
    agg = json.loads((out / "two_si_crescendo_aggregate.json").read_text())
    assert agg["n_expected_goals"] == {CELL: 3}
    assert agg["goalset"]["source"] == SG.STAMP_SOURCE and agg["goalset"]["pairs"][f"crescendo__{CELL}"]["spec"] == SPEC
    assert agg["cells"][CELL]["attack_metrics"]["goalset"] == {"source": SG.STAMP_SOURCE, "spec": SPEC, "shards": ["shard00"], "n_goals": 3}
    assert agg["cells"][CELL]["attack_metrics"]["n_goals"] == 3
    comp = json.loads((out / "completion.json").read_text())
    assert comp["expected"]["gated_goals"] == "per (attack, cell) goalset" and comp["expected"]["goalset"]["source"] == SG.STAMP_SOURCE
    assert comp["cells"][CELL]["gated"]["crescendo"]["n_expected_goals"] == 3
    assert comp["cells"][CELL]["gated"]["crescendo"]["smoke_goals"]["spec"] == SPEC
    prov = json.loads((out / "grading_provenance.json").read_text())
    assert prov["goalset"]["pairs"][f"crescendo__{CELL}"]["override_id"] == "demo_smoke" and prov["goalset"]["pairs"][f"crescendo__{CELL}"]["n"] == 3


def test_the_same_store_without_the_stamp_still_refuses_as_before(tmp_path, monkeypatch, world, eval_ids, stub_judge):
    store_root = _world(tmp_path, world, eval_ids)
    store = store_root / T.run_of(VIEW) / "crescendo" / CELL
    root, rid = _registry(tmp_path, monkeypatch, store, stamp=None)
    with pytest.raises(SystemExit) as ei:
        TW.main(_argv(world, store_root, root, tmp_path / "grades", run_ids=[rid]))
    assert "3/259 goals have a progress record" in str(ei.value)
    with T._conn(root) as conn:
        assert T._judgments(conn, rid) == [] and T._run_state(conn, rid)[0] == "generated"


def test_a_producer_key_spelling_other_shards_than_the_stamp_refuses_by_name(tmp_path, monkeypatch, world, eval_ids, stub_judge):
    store_root = _world(tmp_path, world, eval_ids)
    store = store_root / T.run_of(VIEW) / "crescendo" / CELL
    root, rid = _registry(tmp_path, monkeypatch, store, key=KEY_WHOLE)
    with pytest.raises(SystemExit) as ei:
        TW.main(_argv(world, store_root, root, tmp_path / "grades", run_ids=[rid]))
    assert "did not run the document's plan" in str(ei.value) and "sh0_1of9" in str(ei.value)


def test_a_shard_outside_the_stamped_range_refuses_by_name(tmp_path, monkeypatch, world, eval_ids, stub_judge):
    store_root = _world(tmp_path, world, eval_ids, shards=(0, 1))      # shard01 is not the smoke's: another plan's layout
    store = store_root / T.run_of(VIEW) / "crescendo" / CELL
    root, rid = _registry(tmp_path, monkeypatch, store)
    with pytest.raises(SystemExit) as ei:
        TW.main(_argv(world, store_root, root, tmp_path / "grades", run_ids=[rid]))
    assert "not the stamped smoke's ['shard00']" in str(ei.value) and "outside the range ['shard01']" in str(ei.value)
    with T._conn(root) as conn:
        assert T._judgments(conn, rid) == []


def test_goalset_dir_beside_a_stamped_run_refuses_one_source(tmp_path, monkeypatch, world, eval_ids, stub_judge):
    store_root = _world(tmp_path, world, eval_ids)
    store = store_root / T.run_of(VIEW) / "crescendo" / CELL
    root, rid = _registry(tmp_path, monkeypatch, store)
    gdir = tmp_path / "goalsets"; gdir.mkdir()
    (gdir / f"crescendo__{CELL}.txt").write_text("".join(g + "\n" for g in eval_ids[0::N]))
    with pytest.raises(SystemExit) as ei:
        TW.main(_argv(world, store_root, root, tmp_path / "grades", run_ids=[rid], extra=["--goalset-dir", str(gdir)]))
    assert "two goal-set sources for one invocation" in str(ei.value)


def test_a_malformed_stamp_refuses_before_any_store_is_read(tmp_path, monkeypatch, world, eval_ids, stub_judge):
    bad = dict(STAMP, keys={**STAMP["keys"], SG.KEY: "0-0/259"})
    store_root = _world(tmp_path, world, eval_ids)
    store = store_root / T.run_of(VIEW) / "crescendo" / CELL
    root, rid = _registry(tmp_path, monkeypatch, store, stamp=bad)
    with pytest.raises(SystemExit) as ei:
        TW.main(_argv(world, store_root, root, tmp_path / "grades", run_ids=[rid]))
    assert "N=259 > 99" in str(ei.value)


def test_the_registry_dry_run_names_the_stamp_and_touches_nothing(tmp_path, monkeypatch, world, eval_ids, capsys):
    store_root = _world(tmp_path, world, eval_ids)
    store = store_root / T.run_of(VIEW) / "crescendo" / CELL
    root, rid = _registry(tmp_path, monkeypatch, store)
    before = sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file())
    rc = TW.main(_argv(world, store_root, root, tmp_path / "grades", run_ids=[rid], extra=["--dry-run"]))
    out = capsys.readouterr().out
    assert rc in (None, 0)
    assert f"is stamped {SG.KEY}={SPEC}" in out and "GOALSET mode from the run-overrides stamp: 1 (attack, cell) pairs, 3 goal-outcomes" in out
    assert "DRY RUN: gates passed for 1 pair(s)" in out
    assert sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file()) == before
    assert not (tmp_path / "grades").exists()
