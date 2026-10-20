"""degap-fix-4: the walker's three front-door store-resolution rules, each verifying a store's identity from INSIDE the
store before a row is read — a path that merely exists is not evidence. Synthetic stores laid out exactly like the three real shapes:

  static       <root>/transcripts_results/<private_run>/<family>/<cell>/shardNN/{served_target.json, progress.jsonl, responses.jsonl}
  multi-turn   the same + attacker_engine.json / target_timing.jsonl, joined to a grading host aggregate
               <REPRODUCE_STATE>/<model>/grades/<view>/two_si_<attack>_aggregate.json on (target=view, attack, cell, private_run) and DATED
  gsm8k        <object>/thinking_card/{ladder_state.json, cells/<cell>/{manifest.json, evidence.json, gen_stats.json, private/inspect_logs/*/*.eval}}

Each shape has a NEGATIVE: a same-layout store belonging to another job / cell / object / view is refused BY NAME; an aggregate that predates
the store (or disagrees with its counts) leaves the run pending with the reason. The registry rows of the front door carry cell=None, the cell
in producer_object_key_legacy and the job in cluster_job_id — the manifest carries those columns now. Synthetic ids, placeholder shas, a
whitespace tokenizer; the union classifier, the static / multi-turn store loaders and the gsm8k column tool are the real ones."""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/sdfbase/full22/degeneracy_gap_fill.py"
UNION_SRC = REPO / "src"
OBJ = "c72f7a10362a76354fdd39a004f3f722682a350e3062ef21c683f710346cd2cc"       # a steered cell's object sha (synthetic registry object_id)
TEXT = "the quick brown fox jumps over the lazy dog and keeps going through the meadow until dusk"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("degeneracy_gap_fill_identity_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    """A whitespace tokenizer with the transformers call shape count_tokens uses."""

    def __call__(self, texts, **kw):
        return {"input_ids": [t.split() for t in texts]}


@pytest.fixture
def world(m, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(UNION_SRC))
    du, sg, stamp = m.load_union(UNION_SRC)
    m._W.clear()
    m._W.update({"du": du, "sg": sg, "mtu": m.load_mt_runner(REPO), "models": {}, "host": "grading_host", "stamp": stamp, "pre": {},
                 "xstest_safe": None, "xstest_safe_prov": {}})
    monkeypatch.setattr(m, "hf_model_for", lambda model_id, models: ("stub/tokenizer", None))
    monkeypatch.setattr(m, "tokenizer_for", lambda hf, rev: _Tok())
    monkeypatch.setattr(m, "tokenizer_snapshot", lambda hf, rev: {"stub": True})
    repro = tmp_path / "reproduce-state"
    repro.mkdir()
    monkeypatch.setattr(m, "REPRODUCE_STATE", repro)
    return m


# ------------------------------------------------------------------------------------------------------------------ store writers

def _touch_past(store: Path, written: str) -> None:
    """Pin every file's mtime to the store's written_at: on a real store the mtime IS the write time; a test writes its files now."""
    ts = _dt.datetime.fromisoformat(written.replace("Z", "+00:00")).timestamp()
    for p in store.rglob("*"):
        if p.is_file():
            os.utime(p, (ts, ts))


def _harness(cell: str) -> str:
    return cell if (cell == "none" or cell.startswith("prompted__")) else "si_" + cell.replace("-", "_")


def _served_target(*, cell, declared_cell=None, job, obj, cap, trial, ih, written, serving_dir, virtual_model=None):
    dc = declared_cell or cell
    hc = _harness(dc)
    vm = virtual_model or ("petri__base" if dc == "none" else ("petri__prompted__x" if dc.startswith("prompted__") else f"petri__{hc}"))
    return {"base_cell_id": dc, "tt_cell_id": dc, "harness_cell_id": hc, "virtual_model": vm, "intervention_hash": ih, "manifest_stage": "final",
            "routing_table": f"/x/serving/{serving_dir}_{job}/static/routing_table.json",
            "object_artifact_sha256_by_cell": ({hc: obj} if obj and not str(obj).startswith("si:") and obj != "none" else {}),
            "decode": {"cap": cap, "trial": trial, "do_sample": True, "temperature": 1.0, "top_p": 0.95}, "written_at": written}


def _static_store(root, *, private_run="pad_static_q_p16384", fam="pasttense", cell="constv2-mix-full__lora_r4_ce", declared_cell=None, job="809290",
                  obj=OBJ, cap=16384, trial=0, ih="155fd0541659e51a", family=None, shards=2, written="2026-10-03T10:00:00Z"):
    store = root / "transcripts_results" / private_run / fam / cell
    for i in range(shards):
        sd = store / f"shard{i:02d}"
        sd.mkdir(parents=True)
        (sd / "served_target.json").write_text(json.dumps(_served_target(cell=cell, declared_cell=declared_cell, job=job, obj=obj, cap=cap, trial=trial, ih=ih,
                                                                          written=written, serving_dir="q_padstatic")))
        goals = [f"g{i}a", f"g{i}b"]
        (sd / "progress.jsonl").write_text("\n".join(json.dumps({"goal_id": g, "family": family or fam, "outcome": "ok", "capped": False}) for g in goals) + "\n")
        rows = [{"response": TEXT, "goal_id": g, "sample_idx": 0, "variant": "v0", "finish_reason": "stop", "completion_tokens": 17, "at_cap": False, "max_tokens": cap} for g in goals]
        (sd / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    _touch_past(store, written)
    return store


def _static_run(store, **kw):
    r = {"run_id": "sdfbase_full22:pasttense:aaaaaaaaaaaa:0000000000000001", "model_id": "q", "eval_id": "pasttense", "view": "q_u0", "cell": None,
         "producer_object_key": "constv2-mix-full__lora_r4_ce", "producer_run_key": "padstat-q-p16384@809290", "cluster_job_id": "809290", "object_id": OBJ,
         "intervention_hash": None, "store_uri": str(store), "cap": 16384, "reply_cap": None, "n_items_generated": 4, "trial_index": 0, "lane": None,
         "judgment_ids": ["j1"], "instrument_id": "i", "pass_label": None}
    r.update(kw)
    return r


def _mt_store(root, *, private_run="sdfbase_full22_q_s1", attack="crescendo", cell="constv2-mix-full__lora_r4_ce", job="872043", obj=OBJ, cap=4096,
              ih="155fd0541659e51a", shards=2, written="2026-10-03T10:00:00Z", declared_attack=None, view="q_s1"):
    store = root / "transcripts_results" / private_run / attack / cell
    for i in range(shards):
        sd = store / f"shard{i:02d}"
        sd.mkdir(parents=True)
        (sd / "served_target.json").write_text(json.dumps(_served_target(cell=cell, job=job, obj=obj, cap=cap, trial=0, ih=ih, written=written,
                                                                          serving_dir=f"attacks_{view}_{attack}")))
        (sd / "attacker_engine.json").write_text(json.dumps({"attack": declared_attack or attack, "attacker_engine": "vllm"}))
        goals = [f"g{i}a", f"g{i}b"]
        (sd / "progress.jsonl").write_text("\n".join(json.dumps({"goal_id": g, "outcome": "AttackOutcome.SUCCESS", "capped": False, "turns": 2, "n_target_forwards": 2}) for g in goals) + "\n")
        rows = [{"response": TEXT, "goal_id": g, "order": k, "serving_engine": "lora_native"} for g in goals for k in range(2)]
        (sd / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (sd / "target_timing.jsonl").write_text("\n".join(json.dumps({"finish_reason": "stop", "completion_tokens": 17, "max_tokens": cap}) for _ in rows) + "\n")
    _touch_past(store, written)
    return store


def _aggregate(repro, *, model="q", view="q_s1", attack="crescendo", private_run="sdfbase_full22_q_s1", cell="constv2-mix-full__lora_r4_ce", n_rows=4, n_shards=2,
               written_at="2026-10-03T12:00:00+00:00"):
    d = repro / model / "grades" / view
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"two_si_{attack}_aggregate.json"
    p.write_text(json.dumps({"target": view, "attack": attack, "private_run": private_run, "written_at": written_at, "n_expected_goals": n_rows,
                             "cells": {cell: {"attack_metrics": {"n_goals": n_rows, "asr": 0.5},
                                              "completion": {"gated": {"n_progress_rows": n_rows, "n_shards": n_shards, "n_expected_goals": n_rows, "n_goals_with_progress": n_rows}}}}}))
    return p


def _mt_run(store, **kw):
    r = {"run_id": "sdfbase_full22:crescendo:bbbbbbbbbbbb:0000000000000002", "model_id": "q", "eval_id": "crescendo", "view": "q_s1", "cell": None,
         "producer_object_key": "constv2-mix-full__lora_r4_ce", "producer_run_key": "f22-att-q_s1-g01234-crescendo@872043", "cluster_job_id": "872043", "object_id": OBJ,
         "intervention_hash": None, "store_uri": str(store), "cap": 4096, "reply_cap": None, "n_items_generated": 4, "trial_index": 0, "lane": None,
         "judgment_ids": ["j2"], "instrument_id": "i", "pass_label": None}
    r.update(kw)
    return r


def _card_store(root, *, view="q_d0p75_relu", cell="constv2-mix-full__psr_ce_all_minus_ctl", job="874351", ih="ea14b3079e2b36af", obj_sha="70e950f1e32dfaff",
                cap=8192, result="COMPLETE", n=3, tag=None, texts=None):
    card = root / "g8k" / "controls_frontdoor_q" / view / cell.replace("__", "--", 1) / "thinking_card"
    cell_dir = card / "cells" / f"{cell.replace('__', '--', 1)}__controls__t0__a0__{ih[:8]}__5d1e2792"
    logs = cell_dir / "private" / "inspect_logs" / "inspect__gsm8k_paired200__s0"
    logs.mkdir(parents=True)
    (card / "ladder_state.json").write_text(json.dumps({"schema": "g8k_ladder_state/1", "accepted_generation_attempt": "a1",
                                                        "attempts": [{"attempt_id": "a1", "job_id": job, "cell": cell, "tag": tag or view, "view": view.split("_", 1)[1],
                                                                      "rung": str(cap), "result": result, "kind": "generation"}]}))
    (cell_dir / "manifest.json").write_text(json.dumps({"cell_id": cell, "slurm_job_id": int(job), "intervention_hash": ih, "harness_config_hash": "5d1e27923643f239",
                                                        "on_cap_by_task": {"gsm8k_paired200": cap}, "model_id": "stub/model", "virtual_model": f"petri__{_harness(cell)}",
                                                        "producer": {"driver": "served_capability_run"}}))
    (cell_dir / "evidence.json").write_text(json.dumps({"cell_id": cell, "intervention_hash": ih, "engine_stamp": {"object_sha": obj_sha, "target_engine": "vllm"}}))
    (cell_dir / "gen_stats.json").write_text(json.dumps({"inspect/gsm8k_paired200": {"n": n, "n_requests": n, "task": "inspect/gsm8k_paired200", "acc": 0.5}}))
    (cell_dir / "rows.jsonl").write_text("\n".join(json.dumps({"item_id": f"gsm8k_{i:04d}", "task": "inspect/gsm8k_paired200", "slurm_job_id": int(job)}) for i in range(n)) + "\n")
    ev = logs / "2026-10-03T08-53-47-00-00_gsm8k_Synthetic.eval"
    ev.write_bytes(b"not a real inspect log; the reader is stubbed in this test")
    (logs / (ev.name + ".json")).write_text(json.dumps(texts or {f"gsm8k_{i:04d}": {"text": TEXT, "stop_reason": "stop", "output_tokens": 17} for i in range(n)}))
    return card, cell_dir


def _card_run(cell_dir, **kw):
    r = {"run_id": "sdfbase_full22:gsm8k:cccccccccccc:0000000000000003", "model_id": "q", "eval_id": "gsm8k", "view": "q_d0p75_relu", "cell": None,
         "producer_object_key": "constv2-mix-full__psr_ce_all_minus_ctl", "producer_run_key": "g8k-q-d0p75_relu-difference_relu@874351", "cluster_job_id": "874351",
         "object_id": "70e950f1e32dfaff8972907737a39dcd1f05f34ac8b9467125493bf60f67d0a2", "intervention_hash": "ea14b3079e2b36af", "store_uri": str(cell_dir), "cap": 8192,
         "reply_cap": None, "n_items_generated": 3, "trial_index": 0, "lane": None, "judgment_ids": ["j3"], "instrument_id": "i", "pass_label": None}
    r.update(kw)
    return r


def _install_col(m, monkeypatch):
    col = m.load_gsm8k_column_tool(REPO)
    monkeypatch.setattr(col, "read_eval_texts", lambda path: json.loads(Path(str(path) + ".json").read_text()))
    m._W["col"] = col
    return col


# ------------------------------------------------------------------------------------------------------------------ pure helpers

def test_run_identity_helpers_read_the_registry_row_and_the_serving_dir(m):
    assert m.run_cell({"cell": None, "producer_object_key": "none"}) == "none" and m.run_cell({"cell": "c", "producer_object_key": "x"}) == "c" and m.run_cell({}) == ""
    assert m.run_job({"cluster_job_id": 809290}) == "809290" and m.run_job({"producer_run_key": "padstat-q-p16384@809290"}) == "809290"
    assert m.run_job({"producer_run_key": "sdfbase_full22_q_s3r_sampled_t0"}) is None and m.run_job({}) is None
    assert m.serving_job_of({"routing_table": "/x/exp/sdfbase_full22_native/static_padding/serving/q_padstatic_809290/static/routing_table.json"}) == "809290"
    assert m.serving_job_of({"routing_table": "/x/waves/mt-s1/serving/attacks_q_s1_crescendo_872043/routing_table.json"}) == "872043"
    assert m.serving_job_of({"routing_table": "/x/serving/static_esc_t0_q_s2r_855239/static/routing_table.json"}) == "855239"
    assert m.serving_job_of({"routing_table": "/x/serving/no_job_here/routing_table.json"}) is None and m.serving_job_of({}) is None
    assert m.pick_cell_dir(Path("/nonexistent"), "", None) == (None, "no cell id")
    assert m.cluster_of_path(str(m.CLUSTER_EXP["cluster_a"] / "x")) == "cluster_a" and m.cluster_of_path("/tmp/x") is None


def test_the_manifest_queries_carry_the_identity_columns(m):
    for sql in (m.GAP_SQL, m.CURRENT_SQL):
        for col in ("r.producer_object_key_legacy", "r.cluster_job_id", "r.job_name", "r.object_id", "r.intervention_hash", "r.harness_config_hash"):
            assert col in sql, col


def test_the_real_schema_manifest_rows_carry_the_identity_keys(m, tmp_path):
    sys.path.insert(0, str(UNION_SRC))
    from sisteer.registry import db as DB, harvest as H, identity, records as R, synth  # noqa: E402
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, recs = synth.run_records("k_identity", 100, through="judged")
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-life.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    rows = m.manifest_from_registry(Path(DB.db_path(root)), model=synth.MODEL_ID, all_current=True)
    assert [r["run_id"] for r in rows] == [rid]
    for k in ("producer_object_key", "cluster_job_id", "job_name", "object_id", "intervention_hash", "harness_config_hash"):
        assert k in rows[0], k


# ------------------------------------------------------------------------------------------------------------------ static (rule 1)

def test_static_store_named_by_the_registry_is_measured_after_its_own_records_attest_it(world, tmp_path):
    m = world
    store = _static_store(tmp_path / "exp")
    cell = m.one_run(_static_run(store))
    assert cell["status"] == "measured", cell.get("note")
    ident = cell["store"]["identity"]["transcripts_results:store"]
    assert ident["attested"] is True and set(ident["attested_by"]) == {"job", "object"}
    assert ident["checks"]["cell"] == {"registry": "constv2-mix-full__lora_r4_ce", "store": "constv2-mix-full__lora_r4_ce"}
    assert ident["checks"]["job"]["registry"] == "809290" and ident["checks"]["job"]["store"] == ["809290"] and ident["checks"]["job"]["shards_of_this_job"] == ["shard00", "shard01"]
    assert ident["checks"]["object"]["store"] == [OBJ] and ident["checks"]["family"] == {"registry": "pasttense", "store": ["pasttense"]}
    assert cell["n_rows"] == 4 and cell["pooling"]["cap"] == 16384 and cell["pooling"]["cap_registry"] == 16384 and cell["pooling"]["cap_drawn"] == 16384
    assert cell["resolution"]["cell_rule"] == "exact" and cell["resolution"]["rule"] == "transcripts_results_store"
    # the pre-fix defect: with cell=None and no producer_object_key the walker resolved the FAMILY dir and found no shards; now it says why
    cell2 = m.one_run(_static_run(store, producer_object_key=None))
    assert cell2["status"] == "pending" and cell2["note"] == "no transcript store found" and any("no cell id" in t for t in cell2["resolution"]["candidates_tried"])


def test_a_static_store_of_another_job_cell_object_or_family_is_refused_by_name(world, tmp_path):
    m = world
    other_job = _static_store(tmp_path / "a", job="809291")
    c = m.one_run(_static_run(other_job))
    assert c["status"] == "refused" and "809291" in c["note"] and "809290" in c["note"] and "serving job" in c["note"]
    other_obj = _static_store(tmp_path / "b", obj="e54842ec26c17bfb0d27158d47cedbacdde369dc65f60e93c5ed6774f0a9d472")
    c = m.one_run(_static_run(other_obj))
    assert c["status"] == "refused" and "object sha" in c["note"] and OBJ in c["note"] and "e54842ec26c17bfb" in c["note"]
    other_cell = _static_store(tmp_path / "c", declared_cell="constv2ctl2-full__lora_r4_ce")           # the dir is named by the registry cell, the store says another
    c = m.one_run(_static_run(other_cell))
    assert c["status"] == "refused" and "base_cell_id 'constv2ctl2-full__lora_r4_ce'" in c["note"] and "constv2-mix-full__lora_r4_ce" in c["note"]
    other_fam = _static_store(tmp_path / "d", family="renellm")                                          # a pasttense dir whose rows say renellm
    c = m.one_run(_static_run(other_fam))
    assert c["status"] == "refused" and "family/attack ['renellm']" in c["note"] and "'pasttense'" in c["note"]
    bare = tmp_path / "e" / "transcripts_results" / "pad_static_q_p16384" / "pasttense" / "constv2-mix-full__lora_r4_ce" / "shard00"
    bare.mkdir(parents=True)
    (bare / "responses.jsonl").write_text(json.dumps({"response": TEXT, "goal_id": "g", "sample_idx": 0}) + "\n")          # a path with rows and no identity
    c = m.one_run(_static_run(bare.parent))
    assert c["status"] == "refused" and "no shard carries served_target.json" in c["note"] and "a path that exists is not evidence" in c["note"]


def test_prompted_and_none_cells_attest_through_the_job_and_the_served_virtual_model(world, tmp_path):
    m = world
    store = _static_store(tmp_path / "exp", cell="none", obj="none", ih="2ff9cdff091aef6f")
    c = m.one_run(_static_run(store, producer_object_key="none", object_id="none"))
    assert c["status"] == "measured" and c["store"]["identity"]["transcripts_results:store"]["attested_by"] == ["job"]
    assert c["store"]["identity"]["transcripts_results:store"]["checks"]["virtual_model"]["store"] == ["petri__base"]
    wrong_vm = _static_store(tmp_path / "x", cell="prompted__safety_refusal", obj="si:ac3a58f61d53cf24", ih="faa4c8cfb8c00bf8")
    for sd in wrong_vm.glob("shard*"):
        st = json.loads((sd / "served_target.json").read_text()); st["virtual_model"] = "petri__si_constv2_mix_full__lora_r4_ce"
        (sd / "served_target.json").write_text(json.dumps(st))
    c = m.one_run(_static_run(wrong_vm, producer_object_key="prompted__safety_refusal", object_id="si:ac3a58f61d53cf24"))
    assert c["status"] == "refused" and "virtual_model" in c["note"] and "does not serve a 'prompted__safety_refusal' cell" in c["note"]


def test_the_drawn_cap_rules_the_at_cap_measure_and_the_registry_cap_is_recorded_beside_it(world, tmp_path):
    """The registry's token cap is the declared read-at cap; the rows were drawn at served_target decode.cap (the s2r static_base stores:
    labelled 8192, drawn at 4096). The at-cap detector runs at the drawn cap; both are on the record."""
    m = world
    store = _static_store(tmp_path / "exp", private_run="sdfbase_full22_q_s2r_sampled_t0", cap=4096, job="833933")
    c = m.one_run(_static_run(store, view="q_s2r", cap=8192, cluster_job_id="833933", producer_run_key="f22s-q-q_s2r-static_base_t0@833933"))
    assert c["status"] == "measured", c.get("note")
    assert c["pooling"]["cap"] == 4096 and c["pooling"]["cap_drawn"] == 4096 and c["pooling"]["cap_registry"] == 8192 and "differs from the registry cap" in c["pooling"]["cap_rule"]
    assert c["store"]["identity"]["transcripts_results:store"]["cap_registry"] == 8192
    # a store whose shards disagree on the drawn cap is refused
    bad = _static_store(tmp_path / "y", cap=4096, job="833933")
    st = json.loads((bad / "shard01" / "served_target.json").read_text()); st["decode"]["cap"] = 8192
    (bad / "shard01" / "served_target.json").write_text(json.dumps(st))
    c = m.one_run(_static_run(bad, cluster_job_id="833933"))
    assert c["status"] == "refused" and "disagree on decode.cap [4096, 8192]" in c["note"]


# ------------------------------------------------------------------------------------------------------------------ multi-turn (rule 2)

def test_mt_store_joins_the_reproduce_aggregate_by_content_and_is_measured_when_the_aggregate_postdates_it(world, tmp_path):
    m = world
    store = _mt_store(tmp_path / "exp")
    agg = _aggregate(m.REPRODUCE_STATE)
    assert m.mt_grades(_mt_run(store)) == (agg.parent.parent, agg, "q_s1")
    c = m.one_run(_mt_run(store))
    assert c["status"] == "measured", c.get("note")
    assert c["aggregate"]["read_here"] is True and c["aggregate"]["target"] == "q_s1" and c["aggregate"]["completion"]["n_progress_rows"] == 4 and c["aggregate"]["n_goals_cell"] == 4
    ident = c["store"]["identity"]["store_uri:transcripts_results"]
    assert ident["attested_by"] == ["job", "object"] and ident["checks"]["family"] == {"registry": "crescendo", "store": ["crescendo"]}
    assert ident["aggregate_join"]["attested"] is True and ident["aggregate_join"]["n_progress_rows"] == 4 and ident["aggregate_join"]["n_shards"] == 2
    assert c["resolution"]["candidates_tried"][0].startswith("store_uri:transcripts_results:") and c["n_rows"] == 8 and c["pooling"]["cap"] == 4096


def test_a_store_two_job_groups_share_passes_for_either_job_and_is_measured_whole(world, tmp_path):
    m = world
    store = _mt_store(tmp_path / "exp", shards=1)
    sd = store / "shard01"; sd.mkdir()
    for f in ("served_target.json", "attacker_engine.json", "progress.jsonl", "responses.jsonl", "target_timing.jsonl"):
        (sd / f).write_text((store / "shard00" / f).read_text().replace("872043", "872050").replace("g0a", "g1a").replace("g0b", "g1b"))
    _touch_past(store, "2026-10-03T10:00:00Z")
    _aggregate(m.REPRODUCE_STATE)
    for job in ("872043", "872050"):
        c = m.one_run(_mt_run(store, cluster_job_id=job, producer_run_key=f"f22-att-q_s1-g-crescendo@{job}", declared_jobs=["872043", "872050"]))   # degap-fix-6: both groups declared
        assert c["status"] == "measured" and c["n_rows"] == 8, c.get("note")
        assert c["store"]["identity"]["store_uri:transcripts_results"]["checks"]["job"] == {"registry": job, "store": ["872043", "872050"], "shards_of_this_job": ["shard00" if job == "872043" else "shard01"]}
    c = m.one_run(_mt_run(store, cluster_job_id="872099", producer_run_key="f22-att-q_s1-g-crescendo@872099", declared_jobs=["872043", "872050", "872099"]))
    assert c["status"] == "refused" and "['872043', '872050']" in c["note"] and "872099" in c["note"]
    # degap-fix-6: the second group's job UNDECLARED on the run refuses by name with the remedy, even though the registry job is among the shards' jobs
    c = m.one_run(_mt_run(store, cluster_job_id="872043", producer_run_key="f22-att-q_s1-g-crescendo@872043"))
    assert c["status"] == "refused" and "serving job(s) ['872050'] (routing_table) are not DECLARED jobs" in c["note"] and "extra.serving_jobs" in c["note"], c.get("note")


def test_an_aggregate_that_predates_or_disagrees_with_the_store_leaves_the_run_pending_with_the_reason(world, tmp_path):
    m = world
    store = _mt_store(tmp_path / "exp", written="2026-10-03T10:00:00Z")
    agg = _aggregate(m.REPRODUCE_STATE, written_at="2026-10-03T09:00:00+00:00")                       # graded before the store finished writing
    c = m.one_run(_mt_run(store))
    assert c["status"] == "pending" and "predates" in c["note"] and "2026-10-03T09:00:00+00:00" in c["note"] and "2026-10-03T10:00:00Z" in c["note"]
    _aggregate(m.REPRODUCE_STATE, n_rows=5, written_at="2026-10-03T12:00:00+00:00")                  # a partial / different population
    c = m.one_run(_mt_run(store))
    assert c["status"] == "pending" and "n_progress_rows 5 != the store's 4" in c["note"]
    _aggregate(m.REPRODUCE_STATE, n_rows=4, n_shards=3, written_at="2026-10-03T12:00:00+00:00")
    c = m.one_run(_mt_run(store))
    assert c["status"] == "pending" and "n_shards 3 != the store's 2" in c["note"]
    _aggregate(m.REPRODUCE_STATE, cell="constv2ctl2-full__lora_r4_ce", written_at="2026-10-03T12:00:00+00:00")   # a PARTIAL aggregate: this cell is not in it yet
    c = m.one_run(_mt_run(store))
    assert c["status"] == "pending" and "names no cell 'constv2-mix-full__lora_r4_ce'" in c["note"] and "graded before this cell completed" in c["note"]
    assert c["aggregate"]["cell_in_aggregate"] is False and c["aggregate"]["aggregate_dated"] is True
    agg.unlink()
    _aggregate(m.REPRODUCE_STATE, private_run="sdfbase_full22_q_s1_other")                            # the same view + attack, ANOTHER private run: not joined
    c = m.one_run(_mt_run(store))
    assert c["status"] == "pending" and "no two_si_crescendo_aggregate.json" in c["note"] and "1 candidate(s) for private_run sdfbase_full22_q_s1" in c["note"]
    assert c["resolution"]["reproduce_lookup"]["private_run_segment"] == "sdfbase_full22_q_s1"


def test_mt_two_pass_grading_host_remote_carries_the_dated_aggregate_the_cluster_leg_joins(world, tmp_path, monkeypatch):
    m = world
    exp = tmp_path / "exp"
    monkeypatch.setitem(m.CLUSTER_EXP, "cluster_a", exp)
    store = _mt_store(exp / "reproduce/q/waves/mt-s1-cluster_a")
    _aggregate(m.REPRODUCE_STATE)
    dev = m.one_run(_mt_run(store))
    assert dev["status"] == "remote" and dev["remote_host"] == "cluster_a", dev.get("note")
    assert dev["aggregate"]["completion"] == {"n_progress_rows": 4, "n_shards": 2, "n_expected_goals": 4, "n_goals_with_progress": 4, "n_responses": None}
    assert dev["aggregate"]["cell_in_aggregate"] is True and dev["aggregate"]["aggregate_dated"] is True
    assert dev["aggregate"]["written_at"] == "2026-10-03T12:00:00+00:00" and dev["aggregate"]["target"] == "q_s1"
    # the cluster leg: no reproduce state there; the grading host leg's aggregate facts are the join
    m._W["host"] = "cluster_a"
    m._W["pre"] = {dev["run_id"]: dev}
    monkeypatch.setattr(m, "REPRODUCE_STATE", tmp_path / "empty-reproduce-state")
    cl = m.one_run(_mt_run(store))
    assert cl["status"] == "measured", cl.get("note")
    assert cl["aggregate"]["read_here"] is False and cl["aggregate"]["from_host"] == "grading_host" and cl["store"]["identity"]["store_uri:transcripts_results"]["aggregate_join"]["attested"] is True
    # the same leg with an aggregate that predates the store: pending, never a partial measure
    stale = dict(dev); stale["aggregate"] = {**dev["aggregate"], "written_at": "2026-10-03T09:00:00+00:00"}
    m._W["pre"] = {dev["run_id"]: stale}
    cl2 = m.one_run(_mt_run(store))
    assert cl2["status"] == "pending" and "predates" in cl2["note"]


# ------------------------------------------------------------------------------------------------------------------ gsm8k (rule 3)

def test_gsm8k_thinking_card_named_by_its_cell_dir_is_measured_after_manifest_evidence_gen_stats_and_ladder_state_agree(world, tmp_path, monkeypatch):
    m = world
    _install_col(m, monkeypatch)
    card, cell_dir = _card_store(tmp_path / "exp")
    c = m.one_run(_card_run(cell_dir))
    assert c["status"] == "measured", c.get("note")
    ident = c["store"]["gsm8k"]["identity"]
    assert ident["attested"] is True and ident["attested_by"] == ["job", "intervention_hash", "object"] and ident["view_tag"] == "q_d0p75_relu" and ident["job"] == "874351"
    assert ident["cap"] == 8192 and ident["n_items"] == 3 and ident["generation_result"] == "COMPLETE" and ident["task"] == "inspect/gsm8k_paired200"
    assert c["n_rows"] == 3 and c["pooling"]["draws"] == {"source": 3} and c["store"]["gsm8k"]["source"]["shape"].startswith("thinking_card")
    assert c["resolution"]["rule"].startswith("store_uri_is_cells/<cell>") and c["store"]["paths"] == [str(card)]


def test_the_dose_row_pointed_at_the_alpha_one_twin_store_is_refused_by_name(world, tmp_path, monkeypatch):
    m = world
    _install_col(m, monkeypatch)
    _, twin = _card_store(tmp_path / "exp", view="q_s3r", job="860877", ih="5362f4111b97b569", obj_sha="fbf95bc622b8e666")   # same cell name, alpha = 1
    c = m.one_run(_card_run(twin))
    assert c["status"] == "refused" and "slurm_job_id 860877" in c["note"] and "874351" in c["note"]
    _, same_job = _card_store(tmp_path / "exp2", view="q_s3r", job="874351", ih="5362f4111b97b569", obj_sha="fbf95bc622b8e666")  # job spoofed: the hash still refuses
    c = m.one_run(_card_run(same_job))
    assert c["status"] == "refused" and "intervention_hash 5362f4111b97b569" in c["note"] and "ea14b3079e2b36af" in c["note"]
    _, same_hash = _card_store(tmp_path / "exp3", view="q_s3r", job="874351", ih="ea14b3079e2b36af", obj_sha="70e950f1e32dfaff")     # only the ladder's tag says s3r
    c = m.one_run(_card_run(same_hash))
    assert c["status"] == "refused" and "tag 'q_s3r'" in c["note"] and "'q_d0p75_relu'" in c["note"]


def test_a_gsm8k_card_at_another_cap_or_with_an_incomplete_attempt_is_not_measured(world, tmp_path, monkeypatch):
    m = world
    _install_col(m, monkeypatch)
    _, cell_dir = _card_store(tmp_path / "exp")
    c = m.one_run(_card_run(cell_dir, cap=16384))
    assert c["status"] == "refused" and "drawn at cap 8192, the registry row says 16384" in c["note"]
    _, incomplete = _card_store(tmp_path / "exp2", result="INCOMPLETE")
    c = m.one_run(_card_run(incomplete))
    assert c["status"] == "pending" and "INCOMPLETE, not COMPLETE" in c["note"]
    _, short = _card_store(tmp_path / "exp3", texts={"gsm8k_0000": {"text": TEXT, "stop_reason": "stop", "output_tokens": 17}})   # gen_stats says 3 items, the log has 1
    c = m.one_run(_card_run(short))
    assert c["status"] == "refused" and "1 samples, gen_stats.json says 3" in c["note"]


def test_a_gsm8k_cell_dir_on_a_cluster_root_is_filed_remote_from_the_grading_host(world, tmp_path, monkeypatch):
    m = world
    _install_col(m, monkeypatch)
    exp = tmp_path / "exp"
    monkeypatch.setitem(m.CLUSTER_EXP, "cluster_a", exp)
    _, cell_dir = _card_store(exp)
    c = m.one_run(_card_run(cell_dir))
    assert c["status"] == "remote" and c["remote_host"] == "cluster_a" and "store_uri: on cluster_a (not this host)" in c["resolution"]["candidates_tried"]
