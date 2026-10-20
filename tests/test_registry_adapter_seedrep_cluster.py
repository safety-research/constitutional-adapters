"""The `seedrep_cluster` reader on a SYNTHETIC mirror shaped like the seed-replicate ReLU records (tests/fixtures/registry/seedrep_cluster/:
invented shas, `/workspace/TESTUSER/...` paths, three item rows replicated to the item set; no transcript or generation text).

What it pins:

* STATICS from the ladder record and the gate decisions: a pair the ladder says left at 8,192 files a base run (`superseded_by_rung`,
  `superseded_by_run_id` naming the rung run, the emitter's `supersede` ruling placeholder) and a rung run (`final_at_rung`, cap 8,192,
  `ladder_rung`, the gate's raw / cap-coherent / loop shares as run_degeneration rows); a pair that never left 4,096 files ONE base run
  (`final_at_base`) and no phantom 8,192 unit; every static run is `generated` with no judgment and no metric, bound to the SEEDED ladder
  instrument (`static_sampled_vendor_preset_4096_8192` on qwen35_4b_it), producer key = the store directory name; ONE pending disposition
  per (object key, family) names the grading-pin reason;
* MMLU: one rule-scored run per object key (`acc` recounted from the 200 rows, `judgment_id ''`, no judgment row, the canonical
  `mmlu_letter_prefill_200` instrument, `instrument_match match`); a letter absent from the top-64 is scored 0 and counted;
* the rows carry model `qwen35_4b_it`, the seed-replicate object resolved through the roster / pins (never hardcoded), `view_legacy`
  `qwen35_4b_s2`, `trial_index 0`, `substrate cluster_b`, the tree sha, `producer_session ReLU`;
* REFUSED BY NAME: an object the roster / pins do not name; a ladder entry whose exit says a rung ran without its gate file; a gate record
  disagreeing with the ladder; a decode block matching no policy row; an MMLU manifest sha16 that is no prefix of the pinned object; a stale
  `<root>/sampled/` copy beside the producer's FLAT layout (`<root>/{ladders,mmlu,grades_sampled}`, the rsync target);
* run identity keys on the store directory NAME, the object key, the family and the seeded instrument — never the root's path — so the same
  fixture under two roots mints identical ids; and the SAME ids pick the grades up: a fake grades store turns the run `judged` in the registry;
* the MULTI-TURN stores (v4): s2 / s3 file judged runs on the canonical seeded multi-turn instruments with the per-goal items, the ASR readings
  and the contrast against the same-seed `none`; the substrate is read PER RUN from the store's routing table by a prefix-anchored rule
  on the clusters' shared roots (cluster A's for the multi-turn shards, cluster B's for the statics' — both under one tag; hostnames never); the
  unfitted keys are decode-seed trials there too; s1 files like its siblings since v5 (v4 held it by name).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil

import pytest

import _b1_fixtures as F
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry import vocab as V
from sisteer.registry.adapters import relu_paired as RP
from sisteer.registry.adapters import seedrep_cluster as SC
from sisteer.registry.adapters.base import AdapterContext

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "registry", "seedrep_cluster")
TAG = "qwen35_4b_s2"
LORA = "constv2-mix-full__lora_r4_ce"
PROMPTED = "prompted__safety_refusal"
STAGED = f"base64s2__{LORA}"
OBJ = "a570fdd8290ce6e1e3bb7c0e694a33e9042c664b282a36aaf6b2079664f535b3"        # sha256(b"synthetic seedrep lora s2")
TAG_S3 = "qwen35_4b_s3"
OBJ_S3 = hashlib.sha256(b"synthetic seedrep lora s3").hexdigest()
OBJ_BY_TAG = {TAG: OBJ, TAG_S3: OBJ_S3, "qwen35_4b_s1": hashlib.sha256(b"synthetic seedrep lora s1").hexdigest()}
TREE = "48476165d5bb08e36d6b36e56ffdccfb26a0cd2a"                                # sha1(b"synthetic seedrep tree")
GRADING_TREE = "151a1338c6d3f6d0dffb53ffa80de911409e3516"
NOW = "2026-09-21T00:00:00.000Z"
BASE_STORE = f"sdfbase_full22_{TAG}_sampled_t0"
RUNG_STORE = BASE_STORE + "_cap8192"
N_MMLU = 200


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _ctx(conn=None) -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=F.producer(), conn=conn)


def _write(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1)
    return path


def _mmlu_store(root: str, key_id: str, ihash8: str, chash8: str, correct: int, *, sha16: str = None, steered: bool = True, absent: int = 1,
                model_hf: str = "Qwen/Qwen3.5-4B", prefill_top: int = 64, tag: str = TAG) -> str:
    """One MMLU store under mmlu/<tag>_t0/cells/<key>/ from the fixture slice: 200 items, `correct` of them right, `absent` with the letter
    absent from the top-64 (scored 0); the manifest's decode seed base is the tag's seed index."""
    store_key = f"{key_id.replace('__', '--', 1)}__controls__t0__a0__{ihash8}__{chash8}"
    d = os.path.join(root, "mmlu", f"{tag}_t0", "cells", store_key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(FIX, "mmlu_rows_slice.jsonl"), encoding="utf-8") as fh:
        slice_rows = [json.loads(line) for line in fh if line.strip()]
    template = slice_rows[0]
    lines = []
    for i in range(N_MMLU):
        r = dict(template)
        ex = json.loads(r["extra_json"])
        r["item_id"] = f"m{i + 1:04d}-{i + 1:08x}"
        r["created_at"] = 1789921920.0 + i
        r["intervention_hash"], r["harness_config_hash"] = ihash8 + "0" * 8, chash8 + "0" * 8
        r["arm_name"] = key_id.replace("__", "--", 1)
        r["model_id"] = model_hf
        if i < absent:
            ex.update(pick=None, letter_read="absent", n_censored=1)
            r["value"] = 0.0
        else:
            ex.update(pick=("A" if i < absent + correct else "B"), gold="A", letter_read="all_present", n_censored=0)
            r["value"] = 1.0 if i < absent + correct else 0.0
        r["extra_json"] = json.dumps(ex, separators=(",", ":"))
        lines.append(json.dumps(r, separators=(",", ":")))
    acc = correct / N_MMLU
    agg = dict(slice_rows[-1], value=acc, extra_json=json.dumps({"n": N_MMLU, "top_logprobs": 64, "response_prefill": False, "censored_items": absent}))
    lines.append(json.dumps(agg, separators=(",", ":")))
    _write(os.path.join(d, "rows.jsonl"), "\n".join(lines) + "\n")
    man = json.load(open(os.path.join(FIX, "mmlu_manifest.json"), encoding="utf-8"))
    man["cell_id"] = key_id
    man["resolved_config"]["cell_id"] = key_id
    man["resolved_config"]["mmlu"]["top_logprobs"] = prefill_top
    man["intervention_hash"] = man["row_constants"]["intervention_hash"] = ihash8 + "0" * 8
    man["harness_config_hash"] = man["row_constants"]["harness_config_hash"] = chash8 + "0" * 8
    man["model_id"] = man["row_constants"]["model_id"] = man["resolved_config"]["model_id"] = model_hf
    man["resolved_config"]["gen"]["seed_base"] = int(tag.rsplit("_s", 1)[1])
    if steered:
        man["fit_provenance"]["object_sha"] = sha16
        man["serving_identity"] = man["resolved_config"]["serving"]["serving_identity"] = SC.STEERED_IDENTITY
    else:
        man["fit_provenance"] = {k: None for k in man["fit_provenance"]}
        man["fit_provenance"].update(served_model_id=model_hf, transfer=False, template_check="baseline")
        man["serving_identity"] = man["resolved_config"]["serving"]["serving_identity"] = SC.PASSTHROUGH_IDENTITY
    _write(os.path.join(d, "manifest.json"), man)
    gs = json.load(open(os.path.join(FIX, "mmlu_gen_stats.json"), encoding="utf-8"))
    gs["mmlu"].update(acc=acc, n=N_MMLU, censored_items=absent, censored_letter_counts=({"B": absent} if absent else {}), top_logprobs=prefill_top)
    _write(os.path.join(d, "gen_stats.json"), gs)
    ev = json.load(open(os.path.join(FIX, "mmlu_evidence.json"), encoding="utf-8"))
    ev["cell_id"] = key_id
    _write(os.path.join(d, "evidence.json"), ev)
    _write(os.path.join(root, "mmlu", f"{tag}_t0", "evidence.json"), ev)
    return d


MT_TREE = synth.code_sha("5")


def _mt_store(root: str, tag: str, attack: str = "crescendo", *, routing_root: str = "/workspace-vast/TESTUSER", served_sha: str = None, seed_replicate: int = None) -> str:
    """A seed-replicate MULTI-TURN store under grades_native_mt/<tag>/ from the fixture (three object keys, four goals) plus the first shard of every
    (attack, key) under transcripts_native_mt/sdfbase_full22_<tag>/: served tree, decode (seed base per seed, seed replicate = the seed index), window,
    the routing table on the given root, the attacker."""
    seed = int(tag.rsplit("_s", 1)[1])
    agg = _retag(json.load(open(os.path.join(FIX, "mt_aggregate_crescendo.json"), encoding="utf-8")), tag)
    agg["attack"] = attack
    d = os.path.join(root, "grades_native_mt", tag)
    _write(os.path.join(d, f"two_si_{attack}_aggregate.json"), agg)
    prov = _retag(json.load(open(os.path.join(FIX, "mt_grading_provenance.json"), encoding="utf-8")), tag)
    prov["attacks"] = [attack]
    _write(os.path.join(d, "grading_provenance.json"), prov)
    _write(os.path.join(d, "completion.json"), _retag(json.load(open(os.path.join(FIX, "mt_completion.json"), encoding="utf-8")), tag))
    st0 = json.load(open(os.path.join(FIX, "mt_served_target.json"), encoding="utf-8"))
    ae = json.load(open(os.path.join(FIX, "mt_attacker_engine.json"), encoding="utf-8"))
    for key in agg["cells"]:
        st = copy.deepcopy(st0)
        st["served_tree_sha"] = MT_TREE                    # the recorded fixture file states a placeholder tree; the store states the derived one
        steered = key == LORA
        st["serving_identity"] = SC.STEERED_IDENTITY if steered else SC.PASSTHROUGH_IDENTITY
        st["object_artifact_sha256_by_cell"] = ({"si_constv2_mix_full__lora_r4_ce": (served_sha or OBJ_BY_TAG[tag])} if steered else {})
        st["decode"]["seed_base"] = st["target_decode_resolution"]["seed_base"] = 1926967131 + (seed - 2) * 1000
        st["target_decode_resolution"]["seed_replicate"] = seed if seed_replicate is None else seed_replicate
        st["routing_table"] = f"{routing_root}/exp/sdfbase_full22_native/serving/attacks_{tag}_s{seed}_{attack}_p0_r4_583782/routing_table.json"
        st["timing_log"] = f"{routing_root}/exp/sdfbase_full22_native/transcripts_results/sdfbase_full22_{tag}/{attack}/target_timing.jsonl"
        sd = os.path.join(root, "transcripts_native_mt", f"sdfbase_full22_{tag}", attack, key, "shard_p0_a00_00")
        _write(os.path.join(sd, "served_target.json"), st)
        _write(os.path.join(sd, "attacker_engine.json"), ae)
    return d


def _static_shard(root: str, store: str, fam: str, key: str, routing_root: str = "/workspace/TESTUSER") -> str:
    """A statics shard's served_target.json (the store witness of the cluster and the base job) under transcripts_sampled/<store>/<family>/<key>/shard00/."""
    st = json.load(open(os.path.join(FIX, "static_served_target.json"), encoding="utf-8"))
    st["routing_table"] = f"{routing_root}/exp/sdfbase_full22_native/sampled/serving/static_base_t0_{fam}_{TAG}_121239/static/routing_table.json"
    return _write(os.path.join(root, "transcripts_sampled", store, fam, key, "shard00", "served_target.json"), st)


def _retag(obj, tag: str):
    """The s2 fixture re-spelled for another seed view: every `qwen35_4b_s2` -> the tag, `seed_base` 2 -> the tag's seed index, and every
    `written_at` shifted (seed - 2) hours later (the live seed views were written in seed order, which is what the frame's `later` rule reads)."""
    from datetime import datetime, timedelta
    seed = int(tag.rsplit("_s", 1)[1])
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "seed_base" and v == 2:
                out[k] = seed
            elif k == "written_at" and isinstance(v, str) and seed != 2:
                out[k] = (datetime.fromisoformat(v.replace("Z", "+00:00")) + timedelta(hours=seed - 2)).isoformat()
            else:
                out[k] = _retag(v, tag)
        return out
    if isinstance(obj, list):
        return [_retag(v, tag) for v in obj]
    if isinstance(obj, str):
        return obj.replace(TAG, tag)
    return obj


def _mirror(tmp_path, *, ladder=None, gate=True, mmlu=True, roster_names_lora: bool = True, sha16: str = None, root: str = None, tags=(TAG,), roster_tags=None) -> dict:
    """A synthetic mirror root in the producer's FLAT layout (`<root>/ladders`, `<root>/mmlu`, `<root>/grades_sampled`) plus the roster / pins
    that name its objects; a second seed view (`tags`) is the s2 fixture re-spelled with its own fitted object and seed base; `roster_tags`
    (default `tags`) are the seed selections the roster / pins name — a multi-turn-only seed view needs its selection without a ladder."""
    root = root or str(tmp_path / "seedrep")
    base_ladder = ladder if ladder is not None else json.load(open(os.path.join(FIX, "static_ladder_t0.json"), encoding="utf-8"))
    views, pins = {}, {}
    for tag in (roster_tags or tags):
        sel = f"s{tag.rsplit('_s', 1)[1]}"
        staged = f"base64{sel}__{LORA}"
        views[sel] = {"config": f"configs/sdfbase_full22_{tag}.yaml", "cells": ({LORA: staged} if roster_names_lora else {}) | {PROMPTED: None}}
        pins[staged] = {"object_sha256": OBJ_BY_TAG[tag], "source": "synthetic"}
    for tag in tags:
        sel = f"s{tag.rsplit('_s', 1)[1]}"
        _write(os.path.join(root, "ladders", f"{tag}_t0", "static_ladder_t0.json"), _retag(base_ladder, tag))
        if gate:
            shutil.copytree(os.path.join(FIX, "gates"), os.path.join(root, "ladders", f"{tag}_t0", "gates"))
        obj_sha = OBJ_BY_TAG[tag]
        if mmlu:
            _mmlu_store(root, LORA, "18984b43", "b5f5ab68", 127, sha16=(sha16 or obj_sha[:16]), tag=tag)
            _mmlu_store(root, "none", "2ff9cdff", "18eaa69a", 142, steered=False, absent=0, tag=tag)
            _mmlu_store(root, PROMPTED, "faa4c8cf", "51ebc361", 143, steered=False, absent=0, tag=tag)
    roster = {"targets": {"qwen35_4b": {"views": views}}}
    import yaml
    roster_path = _write(str(tmp_path / "roster.yaml"), yaml.safe_dump(roster))
    pins_path = _write(str(tmp_path / "pins.json"), {"qwen35_4b": pins})
    os.makedirs(str(tmp_path / "no_configs"), exist_ok=True)
    return {"root": root, "roster": roster_path, "pins": pins_path, "configs": str(tmp_path / "no_configs")}


def _adapter(fx: dict, seeds, **kw) -> SC.SeedrepClusterAdapter:
    kw.setdefault("views_path", os.path.join(os.path.dirname(fx["root"]), "no_views.json"))       # never the host's live declaration
    return SC.SeedrepClusterAdapter(root=fx["root"], seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"], configs_dir=fx.get("configs"), **kw)


def _read(ad: SC.SeedrepClusterAdapter, conn=None, group: str = TAG, groups_expected=(TAG,)):
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert sorted(groups) == sorted(groups_expected), list(groups)
    recs = []
    for batch in ad.read(_ctx(conn), group, groups[group]):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    bt = {}
    for r in recs:
        bt.setdefault(r["record_type"], []).append(r["payload"])
    return bt, ad.status_extra(group)


def _run(bt, *, eval_id: str, key: str, cap: int = None):
    hits = [r for r in bt["run"] if r["eval_id"] == eval_id and r["campaign_cell_id_legacy"] == key and (cap is None or r["token_cap_tokens"] == cap)]
    assert len(hits) == 1, (eval_id, key, cap, len(hits))
    return hits[0]


# --------------------------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------------------------

def test_discover_names_the_mirror_files_and_groups_them_by_tag(tmp_path, seeds):
    fx = _mirror(tmp_path)
    ad = _adapter(fx, seeds)
    uris = {os.path.relpath(i.uri, fx["root"]) for i in ad.discover()}
    assert f"ladders/{TAG}_t0/static_ladder_t0.json" in uris
    assert f"ladders/{TAG}_t0/gates/static_cap8192/grok420__{LORA}.decision.json" in uris
    assert sum(1 for u in uris if u.startswith(f"mmlu/{TAG}_t0/cells/") and u.endswith("rows.jsonl")) == 3
    assert not any(u.startswith("sampled/") for u in uris)
    assert {ad.comparison_group(i) for i in ad.discover()} == {TAG}
    bt, facts = _read(ad)
    assert facts["layout"] == "layout=flat" == SC.LAYOUT
    ladder_inp = next(i for i in ad.discover() if i.uri.endswith("static_ladder_t0.json"))
    states = ad.source_states(_ctx(), [ladder_inp])
    assert any(r.json_path == "$.layout" and r.current_members == ["layout=flat"] for r in states)
    assert any(r.json_path == "$.cells" and f"{LORA}/grok420" in r.current_members for r in states)
    assert SC.SeedrepClusterAdapter(root=str(tmp_path / "absent"), seeds=seeds).discover() == []       # machine-local mirror: absent is a skip
    with pytest.raises(Exception):
        ad.comparison_group(type("I", (), {"uri": "/elsewhere/other.json"})())
    assert SC.store_name(TAG, 0, 4096) == BASE_STORE and SC.store_name(TAG, 0, 8192) == RUNG_STORE


def test_a_stale_sampled_copy_beside_the_flat_layout_refuses_the_group_by_name(tmp_path, seeds):
    """The first hand mirror sat one level down (`<root>/sampled/{ladders,mmlu}`) and was moved aside; a copy at that level is never read and
    never ignored: its files are declared (so the group re-reads when it appears or goes) and the group refuses naming the path."""
    fx = _mirror(tmp_path)
    shutil.copytree(os.path.join(fx["root"], "ladders"), os.path.join(fx["root"], "sampled", "ladders"))
    ad = _adapter(fx, seeds)
    uris = {os.path.relpath(i.uri, fx["root"]) for i in ad.discover()}
    assert f"sampled/ladders/{TAG}_t0/static_ladder_t0.json" in uris and f"ladders/{TAG}_t0/static_ladder_t0.json" in uris
    bt, facts = _read(ad)
    assert "run" not in bt and "disposition" not in bt and facts["n_refusals"] == 1
    assert "stale sampled/ copy is present at" in facts["refusals"][0] and os.path.join(fx["root"], "sampled") in facts["refusals"][0]
    # only the stale copy, no flat layout: discover refuses the root naming it
    shutil.rmtree(os.path.join(fx["root"], "ladders")); shutil.rmtree(os.path.join(fx["root"], "mmlu"))
    with pytest.raises(Exception) as exc:
        _adapter(fx, seeds).discover()
    assert "flat layout" in str(exc.value) and "stale sampled/ copy" in str(exc.value)


def test_run_ids_key_on_the_store_name_not_the_root_path(tmp_path, seeds):
    """The same fixture under two roots at different paths and depths mints identical run ids (store directory NAME, object key, family, seeded
    instrument), so the units already dry-run keep their identity wherever the mirror sits."""
    fx_a = _mirror(tmp_path, root=str(tmp_path / "a" / "seedrep"))
    fx_b = _mirror(tmp_path, root=str(tmp_path / "b" / "deeper" / "mirror_of_seedrep"))
    ids_a = {(r["eval_id"], r["campaign_cell_id_legacy"], r["token_cap_tokens"]): r["run_id"] for r in _read(_adapter(fx_a, seeds))[0]["run"]}
    ids_b = {(r["eval_id"], r["campaign_cell_id_legacy"], r["token_cap_tokens"]): r["run_id"] for r in _read(_adapter(fx_b, seeds))[0]["run"]}
    assert ids_a == ids_b and len(ids_a) == 9
    inst = seeds.instrument_row(SC.STATIC_INSTRUMENT_SEED, "grok420", model_id="qwen35_4b_it")
    assert ids_a[("grok420", LORA, 8192)] == I.run_id("sdfbase_full22", RUNG_STORE, LORA, "grok420", inst["instrument_id"])
    minst = seeds.instrument_row(SC.MMLU_INSTRUMENT_SEED, "mmlu", model_id="qwen35_4b_it")
    assert ids_a[("mmlu", LORA, 1)] == I.run_id("sdfbase_full22", f"{TAG}_t0", LORA, "mmlu", minst["instrument_id"])


# --------------------------------------------------------------------------------------------
# statics
# --------------------------------------------------------------------------------------------

def test_statics_file_the_rungs_the_ladder_ran_with_the_declared_exits_and_one_pending_disposition_per_pair(tmp_path, seeds):
    fx = _mirror(tmp_path)
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0, facts["refusals"]
    statics = [r for r in bt["run"] if r["eval_id"] in SC.STATIC_FAMILIES]
    # 5 pairs -> 5 base runs + 1 rung run; the four never-left pairs have NO 8,192 unit
    assert len(statics) == 6 and facts["static_runs"] == {"base": 5, "rung": 1}
    inst = seeds.instrument_row(SC.STATIC_INSTRUMENT_SEED, "grok420", model_id="qwen35_4b_it")
    base, rung = _run(bt, eval_id="grok420", key=LORA, cap=4096), _run(bt, eval_id="grok420", key=LORA, cap=8192)
    # identity: the seeded ladder instrument, the store directory as the producer key, the object key -- the graded path's inputs
    assert base["instrument_id"] == rung["instrument_id"] == inst["instrument_id"] and base["item_set_id"] == "static_grok420_259x2"
    assert base["producer_run_key_legacy"] == BASE_STORE and rung["producer_run_key_legacy"] == RUNG_STORE
    assert base["run_id"] == I.run_id("sdfbase_full22", BASE_STORE, LORA, "grok420", inst["instrument_id"]) == SC.static_run_id(TAG, 0, 4096, LORA, "grok420", inst["instrument_id"])
    assert rung["run_id"] == SC.static_run_id(TAG, 0, 8192, LORA, "grok420", inst["instrument_id"]) != base["run_id"]
    for r in (base, rung):
        assert r["model_id"] == "qwen35_4b_it" and r["object_id"] == OBJ and r["view_legacy"] == TAG and r["trial_index"] == 0 and r["lifecycle_state"] == "generated"
        assert r["substrate"] == "cluster_b" and r["code_sha"] == TREE and r["producer_session"] == "ReLU" and r["lane_source_legacy"] == SC.SOURCE
        assert r["transfer_kind"] == "base_to_instruct" and r["serving_path"] == "lora_native" and r["applied_positions_mode"] == "all"
        assert r["judge_id"] == inst["judge_id"] and r["n_items_judged"] is None and r["n_items_planned"] == 259 and r["token_cap_ladder_json"] == [4096, 8192, 16384]
        assert r["n_items_generated"] is None and r["extra"]["n_responses_planned"] == 518
        assert r["decode_sent_json"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
        assert r["decode_source"] == "model_card" and r["decode_policy_match"] == "no_policy" and r["instrument_match"] == "match" and r["seed_base"] == 2
        assert r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|static|off|vendor_preset@v1" and r["extra"]["object_staged_id"] == STAGED
        assert r["extra"]["seed_replicate_label"] == "s2" and r["strength_nominal"] == 1.0 and r["is_operating_point"] == 1
        assert r["extra"]["ungraded_reason"] == SC.PENDING_REASON and r["extra"]["ladder_exit"] == "exit_le_1pct" and r["extra"]["ladder_exit_rung"] == 8192
    assert (base["pass_label"], base["token_cap_rung_index"], rung["pass_label"], rung["token_cap_rung_index"]) == ("base_cap", 0, "ladder_rung", 1)
    assert base["extra"]["escalation_state"] == "superseded_by_rung" and base["superseded_by_run_id"] == rung["run_id"] and base["status_ruling_id"].startswith("placeholder:")
    assert rung["extra"]["escalation_state"] == "final_at_rung" and rung["superseded_by_run_id"] is None and rung["cluster_job_id"] == "121252"
    assert rung["extra"]["n_responses"] == 518 and base["extra"]["n_responses"] == 518 and rung["extra"]["escalation"]["n_regenerated"] == 42
    assert rung["extra"]["escalated_from"] == {"cap": 4096, "kind": "static", "private_run": BASE_STORE}
    assert rung["extra"]["at_cap_share_raw"] == pytest.approx(1 / 518) and rung["extra"]["at_cap_share_coherent"] == 0.0 and rung["extra"]["loop_share"] == pytest.approx(1 / 518)
    assert base["extra"]["at_cap_share_raw"] == pytest.approx(42 / 518) and base["extra"]["at_cap_share_coherent"] is None
    ruling = next(x for x in bt["ruling"] if x["ruling_id"] == base["status_ruling_id"])
    assert ruling["action"] == "supersede" and ruling["target_json"] == {"superseded_by_run_id": rung["run_id"]}
    # the never-left pairs: one base run, final at base, no supersession, the item set as the derived denominator of a 0 at-cap count
    for key, fam in ((LORA, "codeattack"), ("none", "grok420"), ("none", "codeattack"), (PROMPTED, "grok420")):
        r = _run(bt, eval_id=fam, key=key)
        assert r["token_cap_tokens"] == 4096 and r["extra"]["escalation_state"] == "final_at_base" and r["superseded_by_run_id"] is None and r["extra"]["n_responses"] in (518, 777)
        assert r["extra"]["at_cap_share_raw"] == 0.0 and r["extra"]["ladder_exit"] is None and not any(x["token_cap_tokens"] == 8192 for x in bt["run"] if x["campaign_cell_id_legacy"] == key and x["eval_id"] == fam)
    none_run, si_run = _run(bt, eval_id="grok420", key="none"), _run(bt, eval_id="grok420", key=PROMPTED)
    assert none_run["object_id"] == "none" and none_run["serving_path"] == "passthrough" and none_run["applied_positions_mode"] == "none" and none_run["strength_unit"] == "unverified"
    assert si_run["object_id"] == seeds.si_object_id("safety_refusal") and si_run["transfer_kind"] == "prompted_only"
    # the UNFITTED keys are decode-seed TRIALS of one run key: trial_index = the seed index (2 on s2), seed base the store's (2); the fitted stay at 0
    for r in (none_run, si_run, _run(bt, eval_id="codeattack", key="none")):
        assert r["trial_index"] == 2 and r["seed_base"] == 2 and r["extra"]["decode_replicate"]["trial_index"] == 2 and r["extra"]["decode_replicate"]["seed_base"] == 2
        assert r["extra"]["decode_replicate"]["twin_views_legacy"] == ["qwen35_4b_s1", "qwen35_4b_s3"] and r["extra"]["decode_replicate"]["declared_by"] is None
    assert base["trial_index"] == rung["trial_index"] == 0 and "decode_replicate" not in base["extra"]
    assert all(x["action"] == "supersede" for x in bt["ruling"]) and facts["unfitted_trials"]["trial_index"] == 2
    assert set(facts["unfitted_trials"]["runs"]) >= {none_run["run_id"], si_run["run_id"], _run(bt, eval_id="codeattack", key="none")["run_id"]}
    assert all(r["trial_index"] == 2 for r in bt["run"] if r["object_id"] == "none" or str(r["object_id"]).startswith("si:")) and all(r["trial_index"] == 0 for r in bt["run"] if r["object_id"] == OBJ)
    # no judgment, no metric, no item on any static run; a `generated` state on each
    static_ids = {r["run_id"] for r in statics}
    assert not any(j["run_id"] in static_ids for j in bt.get("judgment", [])) and not any(m["run_id"] in static_ids for m in bt["run_metric"])
    assert not any(it["run_id"] in static_ids for it in bt["item_result"])
    # ONE state event per run, its highest state: `generated` for every ungraded run but the superseded base rung, which carries `superseded` alone
    by_run = {}
    for s in bt["run_state"]:
        by_run.setdefault(s["run_id"], []).append(s["state"])
    assert all(len(by_run[rid]) == 1 for rid in static_ids) and by_run[base["run_id"]] == ["superseded"] and by_run[rung["run_id"]] == ["generated"]
    assert all(by_run[rid] == ["generated"] for rid in static_ids - {base["run_id"]})
    # degeneration: the rung's three classified shares from the gate, the base rung's raw count from the ladder, the rest not_measured
    deg = {(d["run_id"], d["measure"]): d for d in bt["run_degeneration"]}
    assert (deg[(rung["run_id"], "at_token_cap_finish_reason")]["numerator"], deg[(rung["run_id"], "at_token_cap_finish_reason")]["denominator"]) == (1, 518)
    assert (deg[(rung["run_id"], "truncation_only")]["numerator"], deg[(rung["run_id"], "rep_ratio_loop")]["numerator"]) == (0, 1)
    assert deg[(rung["run_id"], "rep_ratio_loop")]["extra"]["n_loop_by_rule"] == {"both": 0, "degeneracy": 0, "rep_ratio": 1} and deg[(rung["run_id"], "rep_ratio_loop")]["judgment_id"] == ""
    assert (deg[(base["run_id"], "at_token_cap_finish_reason")]["numerator"], deg[(base["run_id"], "at_token_cap_finish_reason")]["denominator"]) == (42, 518)
    assert deg[(base["run_id"], "truncation_only")]["measurement_state"] == "not_measured" and deg[(base["run_id"], "classifier")]["measurement_state"] == "not_measured"
    quiet = _run(bt, eval_id="codeattack", key=LORA)
    assert (deg[(quiet["run_id"], "at_token_cap_finish_reason")]["numerator"], deg[(quiet["run_id"], "at_token_cap_finish_reason")]["denominator"]) == (0, 777)
    assert deg[(quiet["run_id"], "at_token_cap_finish_reason")]["extra"]["denominator_confidence"] == "derived"
    # ONE pending disposition per (object key, family), the grading-pin reason, naming the run ids the grades will attach to
    disp = [d for d in bt["disposition"] if d["state"] == "pending"]
    assert len(disp) == 5 == facts["dispositions"] and {d["reason"] for d in disp} == {SC.PENDING_REASON}
    d_lora = next(d for d in disp if d["extra"]["campaign_cell_id_legacy"] == LORA and d["eval_id"] == "grok420")
    assert d_lora["object_id"] == OBJ and d_lora["extra"]["rungs_generated"] == [4096, 8192] and d_lora["extra"]["run_ids"] == {"4096": base["run_id"], "8192": rung["run_id"]}
    assert d_lora["extra"]["view_legacy"] == TAG and d_lora["extra"]["lane_source_legacy"] == SC.SOURCE and d_lora["model_id"] == "qwen35_4b_it"
    assert next(d for d in disp if d["extra"]["campaign_cell_id_legacy"] == "none")["object_id"] is None
    assert facts["escalation"][f"{LORA}/grok420/t0"]["state_base"] == "superseded_by_rung" and facts["decode"]["t0"]["policy_id"] == "qwen35_4b_it|static|off|vendor_preset@v1"
    assert facts["tree_sha"]["t0"]["sha"] == TREE


def test_the_base_config_of_the_checkout_must_agree_with_the_re_draw_decode(tmp_path, seeds):
    """Where the checkout carries the base store's config, its declared `sampled_decode` is checked against the re-draw records' decode:
    agreement is named as the evidence, a disagreement holds every static row of the tag by name."""
    fx = _mirror(tmp_path, mmlu=False)
    import yaml
    cfg_dir = str(tmp_path / "configs")
    os.makedirs(cfg_dir)
    sd = {"arm": "vendor_recommended", "trial": 0, "n_trials": 1, "seed_base": 2, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
          "repetition_penalty": 1.0, "source_kind": "model_card"}
    _write(os.path.join(cfg_dir, f"sdfbase_full22_{TAG}_sampled_t0.yaml"), yaml.safe_dump({"sampled_decode": sd}))
    bt, facts = _read(_adapter(dict(fx, configs=cfg_dir), seeds))
    assert facts["n_refusals"] == 0 and "declared sampled_decode" in facts["decode"]["t0"]["source"] and len(bt["run"]) == 6
    _write(os.path.join(cfg_dir, f"sdfbase_full22_{TAG}_sampled_t0.yaml"), yaml.safe_dump({"sampled_decode": dict(sd, temperature=1.0)}))
    bt2, facts2 = _read(_adapter(dict(fx, configs=cfg_dir), seeds))
    assert "run" not in bt2 and any("disagrees with the base store's declared sampled_decode" in w for w in facts2["refusals"])


# --------------------------------------------------------------------------------------------
# MMLU
# --------------------------------------------------------------------------------------------

def test_mmlu_files_one_rule_scored_run_per_object_key_with_the_letter_absent_item_counted(tmp_path, seeds):
    fx = _mirror(tmp_path)
    bt, facts = _read(_adapter(fx, seeds))
    inst = seeds.instrument_row(SC.MMLU_INSTRUMENT_SEED, "mmlu", model_id="qwen35_4b_it")
    runs = sorted((r for r in bt["run"] if r["eval_id"] == "mmlu"), key=lambda r: r["campaign_cell_id_legacy"])
    assert [r["campaign_cell_id_legacy"] for r in runs] == [LORA, "none", PROMPTED] and facts["mmlu_runs"] == 3
    lora = runs[0]
    assert lora["instrument_id"] == inst["instrument_id"] and lora["item_set_id"] == "mmlu_200" and lora["producer_run_key_legacy"] == f"{TAG}_t0"
    assert lora["run_id"] == SC.mmlu_run_id(TAG, 0, LORA, inst["instrument_id"])
    assert lora["object_id"] == OBJ and lora["judge_id"] is None and lora["n_items_judged"] == N_MMLU and lora["token_cap_tokens"] == 1 and lora["scoring_rule"] == "letter_logprob_prefill"
    # ITEMS are the unit (planned = generated = judged = 200: the letter read scores every item at generation); NO state event — the record's
    # `generated` is the state, the harvester's completion stamp the next (a re-emitted event would fold after that stamp and regress the row)
    assert lora["n_items_planned"] == lora["n_items_generated"] == lora["n_items_judged"] == N_MMLU
    assert not any(s["run_id"] in {r["run_id"] for r in runs} for s in bt.get("run_state", []))
    assert lora["decode_source"] == "greedy" and lora["instrument_match"] == "match" and lora["thinking_mode"] == "off" and lora["applied_positions_mode"] == "all"
    assert lora["code_sha"] == TREE and lora["substrate"] == "cluster_b" and lora["cluster_job_id"] == "121264" and lora["producer_session"] == "ReLU"
    assert lora["model_revision"] == "151a1338c6d3f6d0dffb53ffa80de911409e3516" and lora["view_legacy"] == TAG and lora["trial_index"] == 0
    assert lora["n_items_planned"] == N_MMLU and lora["n_items_generated"] == N_MMLU and lora["store_completion_sha256"] and lora["extra"]["n_letter_absent"] == 1
    m = next(x for x in bt["run_metric"] if x["run_id"] == lora["run_id"])
    assert m["metric_id"] == "acc" and m["judgment_id"] == "" and m["value"] == pytest.approx(127 / N_MMLU) and m["k"] == 127 and m["n"] == N_MMLU and m["ci_method"] == "wilson_95"
    assert not any(j["run_id"] == lora["run_id"] for j in bt.get("judgment", []))
    items = [it for it in bt["item_result"] if it["run_id"] == lora["run_id"]]
    assert len(items) == N_MMLU and all(it["judgment_id"] == "" and it["metric_id"] == "acc" for it in items)
    absent = next(it for it in items if it["item_id"] == "m0001-00000001")
    assert absent["value"] == 0.0 and absent["label"] is None and absent["extra"]["letter_read"] == "absent"
    deg = [d for d in bt["run_degeneration"] if d["run_id"] == lora["run_id"]]
    assert deg and all(d["measurement_state"] == "not_applicable" for d in deg)
    none_run = runs[1]
    assert none_run["object_id"] == "none" and none_run["serving_path"] == "passthrough" and none_run["applied_positions_mode"] == "none"
    assert none_run["trial_index"] == 2 and none_run["seed_base"] == 2 and runs[2]["trial_index"] == 2 and lora["trial_index"] == 0 and lora["seed_base"] is None
    assert none_run["extra"]["decode_replicate"]["seed_base"] == 2 and none_run["decode_seed_rule"] == "none"
    assert next(x for x in bt["run_metric"] if x["run_id"] == none_run["run_id"])["value"] == pytest.approx(142 / N_MMLU)
    assert runs[2]["object_id"] == seeds.si_object_id("safety_refusal") and runs[2]["transfer_kind"] == "prompted_only"
    assert facts["mmlu"][f"t0/{LORA}"]["acc"] == pytest.approx(127 / N_MMLU)
    # no disposition for MMLU: the rule-scored run is complete at generation (the harvester's #275 completion stamps it ingested)
    assert not any(d["eval_id"] == "mmlu" for d in bt["disposition"])


# --------------------------------------------------------------------------------------------
# refusals by name
# --------------------------------------------------------------------------------------------

def test_an_object_the_roster_and_pins_do_not_name_holds_its_rows_by_name_and_the_rest_file(tmp_path, seeds):
    fx = _mirror(tmp_path, roster_names_lora=False)
    bt, facts = _read(_adapter(fx, seeds))
    assert not any(r["campaign_cell_id_legacy"] == LORA for r in bt["run"]) and not any(d["extra"]["campaign_cell_id_legacy"] == LORA for d in bt["disposition"])
    assert any(w.startswith(f"{TAG}|t0|{LORA} -- rows HELD:") and "name no object" in w for w in facts["refusals"]), facts["refusals"]
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {"none", PROMPTED}


def test_a_ladder_exit_at_a_rung_without_its_gate_file_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, gate=False, mmlu=False)
    bt, facts = _read(_adapter(fx, seeds))
    assert not any(r["campaign_cell_id_legacy"] == LORA and r["eval_id"] == "grok420" for r in bt["run"])
    assert any("without its gate decision file" in w and f"grok420__{LORA}.decision.json" in w for w in facts["refusals"]), facts["refusals"]
    assert len(bt["run"]) == 4 and len(bt["disposition"]) == 4     # the other four pairs still file


def test_a_gate_record_that_disagrees_with_the_ladder_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, mmlu=False)
    p = os.path.join(fx["root"], "ladders", f"{TAG}_t0", "gates", "static_cap8192", f"grok420__{LORA}.decision.json")
    g = json.load(open(p, encoding="utf-8"))
    g["counts"]["n_loop"] = 7
    _write(p, g)
    bt, facts = _read(_adapter(fx, seeds))
    assert not any(r["campaign_cell_id_legacy"] == LORA and r["eval_id"] == "grok420" for r in bt["run"])
    assert any("disagrees with the ladder record" in w and "n_loop 7 vs the ladder's 1" in w for w in facts["refusals"]), facts["refusals"]


def test_a_decode_block_matching_no_policy_row_holds_the_tags_static_rows_and_mmlu_still_files(tmp_path, seeds):
    lad = json.load(open(os.path.join(FIX, "static_ladder_t0.json"), encoding="utf-8"))
    lad["cells"][LORA]["grok420"]["escalation"]["decode"]["temperature"] = 0.99
    fx = _mirror(tmp_path, ladder=lad)
    bt, facts = _read(_adapter(fx, seeds))
    assert not any(r["eval_id"] in SC.STATIC_FAMILIES for r in bt["run"]) and "disposition" not in bt
    assert any("matches no registered decode_policy row" in w for w in facts["refusals"]), facts["refusals"]
    assert sum(1 for r in bt["run"] if r["eval_id"] == "mmlu") == 3


def test_a_ladder_verdict_the_reader_cannot_file_and_an_unknown_family_are_refused_by_name(tmp_path, seeds):
    lad = json.load(open(os.path.join(FIX, "static_ladder_t0.json"), encoding="utf-8"))
    lad["cells"]["none"]["codeattack"]["n_at_cap"] = {"4096": 3}            # at-cap rows and no verdict: neither final nor a declared exit
    lad["cells"]["none"]["renellm_x"] = copy.deepcopy(lad["cells"]["none"]["grok420"])
    fx = _mirror(tmp_path, ladder=lad, mmlu=False)
    bt, facts = _read(_adapter(fx, seeds))
    assert not any(r["campaign_cell_id_legacy"] == "none" and r["eval_id"] == "codeattack" for r in bt["run"])
    assert any("wrote no verdict this reader files" in w for w in facts["refusals"]) and any("is no static paper eval" in w for w in facts["refusals"]), facts["refusals"]
    assert len(bt["run"]) == 5


def test_an_mmlu_manifest_sha16_that_is_no_prefix_of_the_pinned_object_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, sha16="0123456789abcdef")
    bt, facts = _read(_adapter(fx, seeds))
    assert not any(r["eval_id"] == "mmlu" and r["campaign_cell_id_legacy"] == LORA for r in bt["run"])
    assert any("is no prefix of the pinned object" in w for w in facts["refusals"]), facts["refusals"]
    assert sum(1 for r in bt["run"] if r["eval_id"] == "mmlu") == 2 and facts["static_runs"] == {"base": 5, "rung": 1}


def test_an_object_absent_from_the_connected_registry_is_refused_by_name(tmp_path, seeds):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY)")
    fx = _mirror(tmp_path, mmlu=False)
    bt, facts = _read(_adapter(fx, seeds), conn=conn)
    assert not any(r["campaign_cell_id_legacy"] == LORA for r in bt["run"]) and any("has no object row in the registry" in w for w in facts["refusals"])
    conn.execute("INSERT INTO object VALUES (?)", (OBJ,))
    bt2, facts2 = _read(_adapter(fx, seeds), conn=conn)
    assert facts2["n_refusals"] == 0 and sum(1 for r in bt2["run"] if r["campaign_cell_id_legacy"] == LORA) == 3


# --------------------------------------------------------------------------------------------
# the grades attach to the same run ids
# --------------------------------------------------------------------------------------------

def _grades_store(root: str, store: str, fam: str, keys, *, n_goals: int = 4, k: int = 2, n_at_cap: int = 1) -> None:
    per_goal = {f"goal_{i}": {"best_score": (1.0 if i < k else 0.0), "capped": False, "n_target_forwards": 3} for i in range(n_goals)}
    agg = {"attack": fam, "attack_tier": "static", "written_at": "2026-09-22T02:00:00+00:00", "serving_engine": "lora_native", "private_run": store, "n_expected_goals": n_goals,
           "cells": {key: {"attack_metrics": {"asr": {"0.25": k / n_goals, "0.5": k / n_goals, "0.75": k / n_goals}, "n_goals": n_goals, "n_jailbroken": k, "n_graded": n_goals,
                                              "per_goal": per_goal}} for key in keys}}
    d = os.path.join(root, "grades_sampled", store)
    _write(os.path.join(d, f"two_si_{fam}_aggregate.json"), agg)
    comp_path = os.path.join(d, "completion.json")
    comp = json.load(open(comp_path, encoding="utf-8")) if os.path.isfile(comp_path) else {"cells": {}}
    for key in keys:
        comp["cells"].setdefault(key, {})[fam] = {"n_at_cap": n_at_cap, "n_responses": 518, "n_goals_with_progress": n_goals}
    _write(comp_path, comp)
    _write(os.path.join(d, "grading_provenance.json"), {"worktree": {"head": GRADING_TREE, "path": "/workspace/TESTUSER/G"}, "judge": {"judge_model": "claude-haiku-4-5", "rubric": "strongreject_v1",
                                                                                                                                         "org_initial": "lowprio", "pass_id": "p" * 64},
                                                        "org_final": "lowprio", "finished_at": "2026-09-22T02:10:00+00:00"})


def _registry_root(tmp_path, seeds, objects) -> str:
    """A registry root holding the REAL seeds' dimension rows as the registry's own loader mints them (adapters/dimensions.py: the
    per-model instrument rows, the `none` and `si:<sha>` objects) plus stand-in object rows for the synthetic seed-replicate objects."""
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in objects:
        r = LC.standin_object_record(o, seeds, F.producer(), now, "qwen35_4b_base")
        if r:
            recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def test_grades_that_land_later_attach_to_the_same_run_ids_and_turn_the_runs_judged(tmp_path, seeds):
    """The graded path mints the run id from the same inputs (store directory, object key, family, the seeded ladder instrument): a fake
    grades store for the base and the 8,192 rung of the LoRA pair leaves the ids unchanged, turns the runs `judged` in the registry (a
    judgment by the instrument's judge, the recounted ASR readings, the per-goal items), and drops that pair's pending disposition
    while the other pairs keep theirs."""
    fx = _mirror(tmp_path, mmlu=False)
    ad = _adapter(fx, seeds)
    obj = ad.resolve_object(_ctx(), LORA, "s2")
    root = _registry_root(tmp_path, seeds, [obj])
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    before = {tuple(r) for r in F.q(root, "SELECT run_id, lifecycle_state, producer_run_key_legacy FROM run WHERE campaign_cell_id_legacy = ? AND eval_id = 'grok420'", (LORA,))}
    assert {s for _, s, _ in before} <= {"generated", "superseded"} and len(before) == 2 and F.q(root, "SELECT COUNT(*) FROM judgment") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM disposition WHERE state = 'pending'") == [(5,)] and F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    ids_before = {rid for rid, _, _ in before}
    # the grades land under the store names the ladder record named
    _grades_store(fx["root"], BASE_STORE, "grok420", [LORA], n_at_cap=42)
    _grades_store(fx["root"], RUNG_STORE, "grok420", [LORA], n_at_cap=1)
    ad2 = _adapter(fx, seeds)
    bt, facts = _read(ad2)
    assert facts["n_refusals"] == 0, facts["refusals"]
    graded = [r for r in bt["run"] if r["campaign_cell_id_legacy"] == LORA and r["eval_id"] == "grok420"]
    assert {r["run_id"] for r in graded} == ids_before and facts["judged_runs"] == 2
    for r in graded:
        assert r["n_items_judged"] == 4 and r["n_items_generated"] == 4 and r["n_items_planned"] == 4 and r["store_completion_sha256"]
        assert r["extra"]["generation_state"] == "judged" and r["extra"]["ungraded_reason"] is None and r["extra"]["completion_counts"]["n_responses"] == 518
    jud = [j for j in bt["judgment"] if j["run_id"] in ids_before]
    assert len(jud) == 2 and all(j["judge_id"] == graded[0]["judge_id"] and j["grading_code_sha"] == GRADING_TREE and j["api_pool"] == "lowprio" for j in jud)
    mets = [m for m in bt["run_metric"] if m["run_id"] in ids_before and m["threshold"] == 0.5]
    assert len(mets) == 2 and all(m["value"] == 0.5 and m["k"] == 2 and m["n"] == 4 and m["judgment_id"] for m in mets)
    assert sum(1 for it in bt["item_result"] if it["run_id"] in ids_before) == 8
    assert not any(d["extra"]["campaign_cell_id_legacy"] == LORA and d["eval_id"] == "grok420" for d in bt["disposition"]) and len(bt["disposition"]) == 4
    rep2, ing3, ing4 = F.run_and_ingest(root, ad2, now="2026-09-22T03:00:00.000Z")
    assert rep2.groups_refused == 0 and ing3.refused + ing4.refused == 0, (ing3.refusals, ing4.refusals)
    after = {tuple(r) for r in F.q(root, "SELECT run_id, lifecycle_state FROM run WHERE campaign_cell_id_legacy = ? AND eval_id = 'grok420'", (LORA,))}
    assert {rid for rid, _ in after} == ids_before
    rung_id = SC.static_run_id(TAG, 0, 8192, LORA, "grok420", graded[0]["instrument_id"])
    base_id = SC.static_run_id(TAG, 0, 4096, LORA, "grok420", graded[0]["instrument_id"])
    # the rung run reached `judged` and then, its judgment covering the 4 planned GOALS (the completion predicate's unit: min(goals planned,
    # the item set's 518 rows) = 4), the harvester's completion stamped it `ingested`; the base rung is the superseded reading
    assert dict(after)[rung_id] == "ingested" and dict(after)[base_id] == "superseded"
    assert F.q(root, "SELECT judged_utc IS NOT NULL, ingested_utc IS NOT NULL FROM run WHERE run_id = ?", (rung_id,)) == [(1, 1)]
    assert F.q(root, "SELECT COUNT(*) FROM judgment WHERE run_id IN (%s)" % ",".join("?" * len(ids_before)), tuple(sorted(ids_before))) == [(2,)]
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]


def test_a_grades_store_for_a_pair_the_ladder_never_ran_or_by_another_judge_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, mmlu=False)
    _grades_store(fx["root"], f"sdfbase_full22_{TAG}_sampled_t0_cap16384", "grok420", [LORA])                 # a rung the ladder never ran
    _grades_store(fx["root"], BASE_STORE, "codeattack", ["none"], n_at_cap=0)
    p = os.path.join(fx["root"], "grades_sampled", BASE_STORE, "grading_provenance.json")
    prov = json.load(open(p, encoding="utf-8"))
    prov["judge"] = {"judge_model": "claude-sonnet-5", "rubric": "xstest_cpi/v1", "org_initial": "lowprio"}
    _write(p, prov)
    bt, facts = _read(_adapter(fx, seeds))
    assert any("names a pair the ladder record of this tag / trial does not run" in w for w in facts["refusals"]), facts["refusals"]
    assert any("not the instrument's judge" in w for w in facts["refusals"]), facts["refusals"]
    assert facts["judged_runs"] == 0 and len(bt["run"]) == 6 and len(bt["disposition"]) == 5      # the run stays ungraded, its disposition stays
    # a grades store whose completion count disagrees with the ladder's at-cap count is refused by name too (the store and the ladder are not one spool)
    fx2 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "second"))
    _grades_store(fx2["root"], RUNG_STORE, "grok420", [LORA], n_at_cap=5)
    bt2, facts2 = _read(_adapter(fx2, seeds))
    assert any("counts 5 at-cap rows for this pair, the ladder record 1" in w for w in facts2["refusals"]), facts2["refusals"]
    assert facts2["judged_runs"] == 0 and len(bt2["disposition"]) == 5


# --------------------------------------------------------------------------------------------
# multi-turn stores (v4)
# --------------------------------------------------------------------------------------------

def test_the_substrate_prefix_rule_is_anchored_and_hostnames_are_never_read():
    """Cluster A's shared root is cluster A and must never read as cluster B (a substring test on cluster B's root would); a per-user path
    under cluster B's root is cluster B; anything else is no cluster root and refuses. Node names are not read: both clusters name nodes alike."""
    b = SC.substrate_of_routing_table("/workspace-vast/TESTUSER/exp/x/serving/attacks_qwen35_4b_s2_s2_pair_p0_r4_pair_583782/routing_table.json")
    c = SC.substrate_of_routing_table("/workspace/TESTUSER/exp/x/sampled/serving/static_base_t0_grok420_qwen35_4b_s2_121239/static/routing_table.json")
    # a path prefix is a DERIVATION from the store's field (v6; v5 filed it `recorded`, a label reserved for a field that literally names the substrate)
    assert b[0] == "cluster_a" and b[1] == "derived" and "routing_table" in b[2]["source"] and c[0] == "cluster_b" and c[1] == "derived"
    for other in ("/home/TESTUSER/exp/routing_table.json", "workspace-vast/x", "/vol/x/routing_table.json", "", None, "/workspaces/x/routing_table.json"):
        assert SC.substrate_of_routing_table(other) is None, other
    assert SC.substrate_of_routing_table("/workspace-vast/anything")[0] != "cluster_b"
    assert SC.JOB_IN_ANY_ROUTING_RE.search("/workspace-vast/u/e/serving/attacks_qwen35_4b_s2_s2_pair_p0_r4_pair_583782/routing_table.json").group(1) == "583782"


def test_multi_turn_stores_file_judged_runs_on_the_canonical_instruments_with_the_substrate_read_per_run(tmp_path, seeds):
    fx = _mirror(tmp_path, mmlu=False)
    _mt_store(fx["root"], TAG)
    _static_shard(fx["root"], BASE_STORE, "grok420", LORA)                  # the statics' store witness: cluster B, the base job
    ad = _adapter(fx, seeds)
    uris = {os.path.relpath(i.uri, fx["root"]) for i in ad.discover()}
    assert f"grades_native_mt/{TAG}/two_si_crescendo_aggregate.json" in uris and f"transcripts_native_mt/sdfbase_full22_{TAG}/crescendo/none/shard_p0_a00_00/served_target.json" in uris
    assert f"transcripts_sampled/{BASE_STORE}/grok420/{LORA}/shard00/served_target.json" in uris
    bt, facts = _read(ad)
    assert facts["n_refusals"] == 0, facts["refusals"]
    mt = sorted((r for r in bt["run"] if r["eval_id"] == "crescendo"), key=lambda r: r["campaign_cell_id_legacy"])
    assert [r["campaign_cell_id_legacy"] for r in mt] == [LORA, "none", PROMPTED] and facts["mt_runs"] == 3 and facts["judged_runs"] >= 3
    inst = seeds.instrument_row("multiturn_crescendo_cap4096_generation_config", "crescendo", model_id="qwen35_4b_it")
    lora, none_run, si_run = mt
    for r in mt:
        assert r["instrument_id"] == inst["instrument_id"] and r["producer_run_key_legacy"] == f"sdfbase_full22_{TAG}" and r["run_id"] == I.run_id("sdfbase_full22", f"sdfbase_full22_{TAG}", r["campaign_cell_id_legacy"], "crescendo", inst["instrument_id"])
        assert r["substrate"] == "cluster_a" and r["substrate_source"] == "derived" and "routing_table" in r["field_confidence_json"]["substrate"]["source"]
        assert r["code_sha"] == MT_TREE and r["context_window_tokens"] == 49152 and r["token_cap_tokens"] == 4096 and r["instrument_match"] == "match"
        assert r["judge_id"] == inst["judge_id"] and r["n_items_planned"] == r["n_items_generated"] == r["n_items_judged"] == 4 and r["lifecycle_state"] == "generated"
        assert r["decode_sent_json"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
        assert r["decode_seed_rule"] == "per_goal_turn" and r["seed_base"] == 1926967131 and r["decode_source"] == "model_card" and r["cluster_job_id"] == "583782"
        assert r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|multi_turn|off|vendor_preset@v1" and r["attacker_model_id"] == "qwen36_27b_abliterated_attacker"
        assert r["producer_session"] == "ReLU" and r["view_legacy"] == TAG and r["multi_turn_steering_scope"] != "single_turn" and r["extra"]["n_capped_forward"] == 0
    assert lora["object_id"] == OBJ and lora["trial_index"] == 0 and lora["serving_path"] == "lora_native" and lora["applied_positions_mode"] == "all"
    assert none_run["object_id"] == "none" and none_run["trial_index"] == 2 and none_run["extra"]["decode_replicate"]["seed_replicate"] == 2 and none_run["serving_path"] == "passthrough"
    assert si_run["object_id"] == seeds.si_object_id("safety_refusal") and si_run["trial_index"] == 2
    ids = {r["run_id"] for r in mt}
    jud = [j for j in bt["judgment"] if j["run_id"] in ids]
    assert len(jud) == 3 and all(j["judge_id"] == inst["judge_id"] and j["grading_code_sha"] == "017bf663b81d6163a6230d37a8be1904e2732de3" and j["n_items_judged"] == 4 and j["api_pool"] == "lowprio" for j in jud)
    mets = [m for m in bt["run_metric"] if m["run_id"] == lora["run_id"]]
    assert sorted(m["threshold"] for m in mets) == [0.25, 0.5, 0.75] and all(m["value"] == 0.75 and m["k"] == 3 and m["n"] == 4 and m["metric_id"] == "asr" for m in mets)
    assert sum(1 for it in bt["item_result"] if it["run_id"] == lora["run_id"] and it["metric_id"] == "strongreject_best") == 4
    states = [(s_["run_id"], s_["state"]) for s_ in bt["run_state"] if s_["run_id"] in ids]
    assert sorted(states) == sorted((rid, "judged") for rid in ids)
    # the contrast against the SAME-SEED `none` key: the steered key only, paired per goal, the three thresholds
    con = [c for c in bt["run_contrast"] if c["run_id"] in ids]
    assert len(con) == 3 and {c["anchor_run_id"] for c in con} == {none_run["run_id"]} and all(c["anchor_kind"] == "same_standup_none" and c["n_pairs"] == 4 and c["delta_method"] == "exact_mcnemar" for c in con)
    assert next(c for c in con if c["threshold"] == 0.5)["delta"] == pytest.approx(100 * (0.75 - 0.5)) and facts["mt_contrasts"] == 3
    deg = [d for d in bt["run_degeneration"] if d["run_id"] == lora["run_id"]]
    assert deg and all(d["measurement_state"] == "not_measured" and "FORWARD-cap" in d["notes"] for d in deg)
    # the SAME tag's statics read cluster B: from the store witness where a shard is mirrored (with the base job), else from the ladder roots
    base = _run(bt, eval_id="grok420", key=LORA, cap=4096)
    assert base["substrate"] == "cluster_b" and base["substrate_source"] == "derived" and base["cluster_job_id"] == "121239" and "routing_table" in base["field_confidence_json"]["substrate"]["source"]
    other = _run(bt, eval_id="codeattack", key=LORA)
    assert other["substrate"] == "cluster_b" and other["substrate_source"] == "derived" and other["cluster_job_id"] is None
    assert facts["substrate"][f"{BASE_STORE}/grok420/{LORA}"]["witness"] == "served_target.json:routing_table"
    assert facts["multi_turn"]["runs"]["crescendo/none"]["trial_index"] == 2 and facts["multi_turn"]["runs"][f"crescendo/{LORA}"]["asr_0p5"] == 0.75


def test_seed_one_multi_turn_store_files_like_its_siblings_since_the_users_ruling(tmp_path, seeds):
    """v4 held the s1 store by name (its runs share the frame's collapse key with the Modal seed-1 rows); v5 released it: s1 files
    one judged run per key exactly as s2 / s3, the unfitted keys as decode-seed trial 1, and no `held` disposition is emitted."""
    fx = _mirror(tmp_path, mmlu=False, roster_tags=(TAG, "qwen35_4b_s1"))
    _mt_store(fx["root"], "qwen35_4b_s1")
    ad = _adapter(fx, seeds)
    bt, facts = _read(ad, group="qwen35_4b_s1", groups_expected=(TAG, "qwen35_4b_s1"))
    assert "disposition" not in bt and facts["held_dispositions"] == 0 and facts["n_refusals"] == 0, facts["refusals"]
    runs = sorted((r for r in bt["run"] if r["eval_id"] == "crescendo"), key=lambda r: r["campaign_cell_id_legacy"])
    assert [r["campaign_cell_id_legacy"] for r in runs] == [LORA, "none", PROMPTED] and facts["mt_runs"] == 3 and facts["judged_runs"] == 3
    inst = seeds.instrument_row("multiturn_crescendo_cap4096_generation_config", "crescendo", model_id="qwen35_4b_it")
    for r in runs:
        assert r["instrument_id"] == inst["instrument_id"] and r["producer_run_key_legacy"] == "sdfbase_full22_qwen35_4b_s1" and r["view_legacy"] == "qwen35_4b_s1"
        assert r["substrate"] == "cluster_a" and r["substrate_source"] == "derived" and r["n_items_judged"] == 4 and r["lifecycle_state"] == "generated"
    lora, none_run, si_run = runs
    assert lora["object_id"] == OBJ_BY_TAG["qwen35_4b_s1"] and lora["trial_index"] == 0
    assert none_run["trial_index"] == 1 and none_run["extra"]["decode_replicate"]["seed_replicate"] == 1 and none_run["seed_base"] == 1926967131 - 1000 and si_run["trial_index"] == 1
    assert sum(1 for j in bt["judgment"] if j["run_id"] in {r["run_id"] for r in runs}) == 3 and facts["multi_turn"]["held"] == []
    assert not hasattr(SC, "S1_HOLD_REASON")


def test_multi_turn_refusals_by_name(tmp_path, seeds):
    """A routing table on no cluster root, a served object the pins do not name, and a seed replicate that is not the view index each refuse
    their key's run by name; the other keys of the store still file."""
    fx = _mirror(tmp_path, mmlu=False)
    _mt_store(fx["root"], TAG, routing_root="/home/TESTUSER")
    bt, facts = _read(_adapter(fx, seeds))
    assert not any(r["eval_id"] == "crescendo" for r in bt["run"]) and sum(1 for w in facts["refusals"] if "on no cluster's shared root" in w) == 3
    fx2 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "two"))
    _mt_store(fx2["root"], TAG, served_sha="f" * 64)
    bt2, facts2 = _read(_adapter(fx2, seeds))
    assert {r["campaign_cell_id_legacy"] for r in bt2["run"] if r["eval_id"] == "crescendo"} == {"none", PROMPTED} and any("the served record and the pins disagree" in w for w in facts2["refusals"])
    fx3 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "three"))
    _mt_store(fx3["root"], TAG, seed_replicate=3)
    bt3, facts3 = _read(_adapter(fx3, seeds))
    assert {r["campaign_cell_id_legacy"] for r in bt3["run"] if r["eval_id"] == "crescendo"} == {LORA} and sum(1 for w in facts3["refusals"] if "seed_replicate is 3 but the view index is 2" in w) == 2


def test_the_unfitted_multi_turn_twins_of_two_seed_views_are_trials_2_and_3_of_one_key(tmp_path, seeds):
    fx = _mirror(tmp_path, mmlu=False, tags=(TAG, TAG_S3))
    _mt_store(fx["root"], TAG)
    _mt_store(fx["root"], TAG_S3)
    ad = _adapter(fx, seeds)
    bt2, f2 = _read(ad, group=TAG, groups_expected=(TAG, TAG_S3))
    bt3, f3 = _read(_adapter(fx, seeds), group=TAG_S3, groups_expected=(TAG, TAG_S3))
    n2 = next(r for r in bt2["run"] if r["eval_id"] == "crescendo" and r["campaign_cell_id_legacy"] == "none")
    n3 = next(r for r in bt3["run"] if r["eval_id"] == "crescendo" and r["campaign_cell_id_legacy"] == "none")
    assert (n2["trial_index"], n3["trial_index"]) == (2, 3) and n2["run_id"] != n3["run_id"] and n2["instrument_id"] == n3["instrument_id"] and n2["object_id"] == n3["object_id"] == "none"
    assert n2["seed_base"] != n3["seed_base"] and n3["extra"]["decode_replicate"]["seed_replicate"] == 3
    l2 = next(r for r in bt2["run"] if r["eval_id"] == "crescendo" and r["campaign_cell_id_legacy"] == LORA)
    l3 = next(r for r in bt3["run"] if r["eval_id"] == "crescendo" and r["campaign_cell_id_legacy"] == LORA)
    assert l2["trial_index"] == l3["trial_index"] == 0 and l2["object_id"] != l3["object_id"]


def _roster_entries(seeds) -> list:
    """Synthetic roster entries admitting the grok420 runs of the three object keys (the frame reads a run into the paper only through an entry)."""
    inst = seeds.instrument_row(SC.STATIC_INSTRUMENT_SEED, "grok420", model_id="qwen35_4b_it")["instrument_id"]
    base = {"round_id": LC.ROUND_ID, "comparison_id": "fig_synth", "model_id": "qwen35_4b_it", "eval_id": "grok420", "instrument_id": inst,
            "judge_id": seeds.judge_id(SC.JUDGE_SEED), "strength_rule": "operating_point", "source_citation": "synthetic (test)", "inferred": 0}
    entries = [dict(base, roster_entry_id="re_synth_none", object_id="none"), dict(base, roster_entry_id="re_synth_safety", object_selector_json={"paper_label": "safety_si"}),
               dict(base, roster_entry_id="re_synth_lora", object_selector_json={"paper_label": "constitution_lora"})]
    return [R.make_record("roster_entry", e, F.producer(), NOW) for e in entries]


def _coexist_ruling(now: str) -> dict:
    """The roster ruling this reader's docstring asks for, in the frame's match form (the recommended match, verbatim)."""
    return {"ruling_id": "r2v3-coexist-seedrep-unfitted-decode-seed-trials", "round_id": LC.ROUND_ID, "action": "coexist", "scope_kind": "key",
            "scope_json": {"match": {"lane_sources_legacy": [SC.SOURCE], "model_ids": ["qwen35_4b_it"], "campaigns": ["sdfbase_full22"],
                                     "object_ids": ["none", "si:ac3a58f61d53cf2412c02b8bb56063b9683a9c1154ef1f0978efc1449a836e08",
                                                    "si:21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"],
                                     "eval_ids": ["pasttense", "codeattack", "renellm", "grok420", "mmlu"],
                                     "views_legacy": ["qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3"]}},
            "reason": "synthetic copy of the seed-replicate coexist ruling: the unfitted keys' seed views are decode-seed trials of one key",
            "ruled_utc": now, "ruled_utc_precision": "datetime", "ruled_by": "test", "source_ref": "test"}


def test_the_unfitted_twins_of_two_seed_views_are_two_trials_of_one_key_and_coexist_under_the_rosters_ruling(tmp_path, seeds):
    """s2 and s3 both serve `none` and the prompted keys (one object each, no fitted weights): the reader files them as trials 2 and 3 of one
    run key (distinct run ids by store name, the SAME (model, object, eval, instrument, judge) key), the fitted keys as one trial each on their
    own objects. The paper frame's `later` rule keys on the trial since migration 0021 (token cap, trial_index and pass_label joined the
    key), so without any ruling the two trials coexist as two selected runs in `v_paper_current` -- the trial is the key, not a ruling; the
    roster's `coexist` ruling with the documented match stays in place as the belt (it still protects a cross-source pair at one (key,
    trial)) and reads the same: both selected. Ingested end to end into a scratch root."""
    fx = _mirror(tmp_path, mmlu=False, tags=(TAG, TAG_S3))
    ad = _adapter(fx, seeds)
    objs = [ad.resolve_object(_ctx(), LORA, "s2"), ad.resolve_object(_ctx(), LORA, "s3")]
    assert objs[0].object_id == OBJ and objs[1].object_id == OBJ_S3
    root = _registry_root(tmp_path, seeds, objs)
    for tag in (TAG, TAG_S3):
        _grades_store(fx["root"], f"sdfbase_full22_{tag}_sampled_t0", "grok420", [LORA, "none", PROMPTED], n_at_cap=0, k=(2 if tag == TAG else 3))
        _grades_store(fx["root"], f"sdfbase_full22_{tag}_sampled_t0", "codeattack", [LORA, "none"], n_at_cap=0)
    # the s2 fixture pair left grok420 at 8,192 with n_at_cap 42 at base: the LoRA base store's completion must say 42, the twins' 0
    import json as _json
    for tag in (TAG, TAG_S3):
        p = os.path.join(fx["root"], "grades_sampled", f"sdfbase_full22_{tag}_sampled_t0", "completion.json")
        comp = _json.load(open(p, encoding="utf-8"))
        comp["cells"][LORA]["grok420"]["n_at_cap"] = 42
        _write(p, comp)
        _grades_store(fx["root"], f"sdfbase_full22_{tag}_sampled_t0_cap8192", "grok420", [LORA], n_at_cap=1)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    assert {g for g in ad._facts} == {TAG, TAG_S3} and all(f["n_refusals"] == 0 for f in ad._facts.values()), {g: f["refusals"] for g, f in ad._facts.items()}
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, trial_index, seed_base, lifecycle_state, run_id, object_id FROM run WHERE eval_id = 'grok420' AND token_cap_tokens = 4096 ORDER BY 2, 1")
    by = {(v, k): (t, sb, st, rid, oid) for v, k, t, sb, st, rid, oid in rows}
    assert by[(TAG, "none")][:2] == (2, 2) and by[(TAG_S3, "none")][:2] == (3, 3) and by[(TAG, PROMPTED)][:2] == (2, 2) and by[(TAG_S3, PROMPTED)][:2] == (3, 3)
    assert by[(TAG, LORA)][:2] == (0, 2) and by[(TAG_S3, LORA)][:2] == (0, 3) and by[(TAG, LORA)][4] != by[(TAG_S3, LORA)][4]      # distinct fitted objects
    assert by[(TAG, "none")][3] != by[(TAG_S3, "none")][3] and by[(TAG, "none")][4] == by[(TAG_S3, "none")][4] == "none"           # one object, two run ids
    assert {by[(v, k)][2] for v in (TAG, TAG_S3) for k in ("none", PROMPTED)} == {"ingested"}
    R.write_records(os.path.join(root, "records", "roster", "000001-roster.jsonl"), _roster_entries(seeds))
    rep_r = H.ingest_once(root, workers=0, min_free=0)
    assert rep_r.refused == 0, rep_r.refusals
    sel = lambda: {(v, k): s for v, k, s in F.q(root, "SELECT r.view_legacy, r.campaign_cell_id_legacy, p.selection_state FROM v_paper_current p JOIN run r ON r.run_id = p.run_id WHERE r.eval_id = 'grok420' AND r.token_cap_tokens = 4096")}
    before = sel()
    # the frame's default since migration 0021: the twins of the two seeds are trials 2 and 3 of one key, and the later-run rule keys on
    # the trial, so neither supersedes the other -- both selected with no ruling at all (before it the newer twin superseded the older here)
    assert before[(TAG, "none")] == before[(TAG_S3, "none")] == "selected" and before[(TAG, PROMPTED)] == before[(TAG_S3, PROMPTED)] == "selected"
    # the roster's coexist ruling with the documented match stays as the belt: both trials still selected
    R.write_records(os.path.join(root, "records", "rulings", "000001-coexist.jsonl"), [R.make_record("ruling", _coexist_ruling(NOW), F.producer(), NOW)])
    rep3 = H.ingest_once(root, workers=0, min_free=0)
    assert rep3.refused == 0, rep3.refusals
    after = sel()
    assert after[(TAG, "none")] == after[(TAG_S3, "none")] == "selected" and after[(TAG, PROMPTED)] == after[(TAG_S3, PROMPTED)] == "selected"
    # the fitted keys: one trial each on their own object, never twins
    assert F.q(root, "SELECT COUNT(*) FROM run WHERE campaign_cell_id_legacy = ? AND eval_id = 'grok420' AND trial_index <> 0", (LORA,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM ruling WHERE action = 'coexist' AND ruling_id LIKE 'placeholder:%'") == [(0,)]        # the reader mints no coexist ruling


def test_the_view_set_declaration_is_declared_per_group_and_cited_on_the_unfitted_runs(tmp_path, seeds):
    fx = _mirror(tmp_path, mmlu=False)
    decl = {"sets": {"seedrep": {"title": "SYNTHETIC: the prompted keys have no fitted weights and replicate over the decode base (view sk -> seed base k)",
                                 "view": "s{seed}", "seeds": [1, 2, 3], "target_key": "qwen35_4b"}}}
    vp = _write(str(tmp_path / "views.json"), decl)
    ad = _adapter(fx, seeds, views_path=vp)
    uris = [i.uri for i in ad.discover()]
    assert f"{vp}#{TAG}" in uris and ad.comparison_group(next(i for i in ad.discover() if i.uri.endswith(f"#{TAG}"))) == TAG
    bt, facts = _read(ad)
    none_run = _run(bt, eval_id="grok420", key="none")
    assert none_run["extra"]["decode_replicate"]["declared_by"].startswith(os.path.relpath(vp, os.path.expanduser("~")) if vp.startswith(os.path.expanduser("~")) else vp)
    assert facts["unfitted_trials"]["declared_seeds"] == [1, 2, 3] and facts["unfitted_trials"]["declaration"]
    # a declaration with another view grammar is not cited
    _write(vp, {"sets": {"seedrep": {"title": "x", "view": "seed{seed}", "seeds": [1]}}})
    bt2, facts2 = _read(_adapter(fx, seeds, views_path=vp))
    assert _run(bt2, eval_id="grok420", key="none")["extra"]["decode_replicate"]["declared_by"] is None and facts2["unfitted_trials"]["declaration"] is None


def test_mmlu_runs_reach_ingested_through_the_harvesters_rule_scored_completion(tmp_path, seeds):
    """A rule-scored run's completion is its headline metric row (n = the 200 items) — no judgment, no state event from the reader: the
    harvester's completion stamps `ingested` with the scoring time as `judged_utc`; a second pass over unchanged inputs (no new revision)
    leaves it there."""
    fx = _mirror(tmp_path)
    ad = _adapter(fx, seeds)
    root = _registry_root(tmp_path, seeds, [ad.resolve_object(_ctx(), LORA, "s2")])
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT campaign_cell_id_legacy, lifecycle_state, n_items_planned, n_items_generated, n_items_judged, judged_utc IS NOT NULL, ingested_utc IS NOT NULL FROM run WHERE eval_id = 'mmlu' ORDER BY 1")
    assert [r[1:] for r in rows] == [("ingested", N_MMLU, N_MMLU, N_MMLU, 1, 1)] * 3, rows
    assert F.q(root, "SELECT COUNT(*) FROM run_state_event e JOIN run r ON r.run_id = e.run_id WHERE r.eval_id = 'mmlu'") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    rep2, ing3, ing4 = F.run_and_ingest(root, _adapter(fx, seeds), now="2026-09-21T01:00:00.000Z")
    assert rep2.groups_refreshed == 0 and F.q(root, "SELECT DISTINCT lifecycle_state FROM run WHERE eval_id = 'mmlu'") == [("ingested",)]


# --------------------------------------------------------------------------------------------
# best-of-N (v6): the graded seed-replicate best-of-N stores
# --------------------------------------------------------------------------------------------

BON_TREE = "017bf663b81d6163a6230d37a8be1904e2732de3"          # the seed-replicate tree (SC.BON_INSTRUMENT_TREES: byte-identical BoN files)
BON_SCHEDULE = (1, 2, 4, 8, 16, 32, 64, 75)
#: the live queued seed-1 `none` row's id (registry.sqlite, read 2026-09-21): the premint formula must reproduce it
LIVE_S1_PREMINT_NONE = "seedrep_battery:strongreject_bon:7b6761844d0d:62e1e84275b67d65"
MODAL_ATTEMPT = "01TESTATTEMPT" + "0" * 13 + "-" + "0" * 9 + "abc"          # a synthetic Modal attempt id in the call-id grammar
THRESHOLDS = ("0.25", "0.5", "0.75")


def _bon_goals(k: int, n: int, seed: int) -> dict:
    """n invented goals: the first k are hit within the draws (best 1.0 on even indices, 0.6 on odd — a hit at 0.25 / 0.5 but not at 0.75), the rest never."""
    per = {}
    for i in range(n):
        g = f"goal_{i:03d}"
        if i < k:
            best = 1.0 if i % 2 == 0 else 0.6
            ns = (i * 7 + seed) % 75 + 1
            hit = {t: best > float(t) for t in THRESHOLDS}
            per[g] = {"best_score": best, "hit": hit, "n_star": {t: (ns if hit[t] else -1) for t in THRESHOLDS}}
        else:
            per[g] = {"best_score": 0.001, "hit": {t: False for t in THRESHOLDS}, "n_star": {t: -1 for t in THRESHOLDS}}
    return per


def _bon_curve(per: dict, t: str) -> dict:
    n = len(per)
    return {str(N): sum(1 for r in per.values() if 0 < r["n_star"][t] <= N) / n for N in BON_SCHEDULE}


def _bon_store(root: str, tag: str, hits: dict, *, n_goals: int = 100, n_rows: int = 7500, keys=("none", LORA, PROMPTED), tree: str = BON_TREE, decode: dict = None,
               n_max: int = 75, rubric: str = "strongreject_v2", cache_kind: str = "sr2", routing_root: str = "/workspace-vast/TESTUSER", source_cluster: str = "cluster_a",
               source_block: bool = True, premint: bool = False, aggregate: bool = True, served_sha: str = None, premint_ids: dict = None) -> str:
    """A graded best-of-N root under grades_bon/<tag>/ from the two fixture templates plus invented goals (`hits` = the goals hit per key): every
    store-level file, the per-key any-hit grids and served records, and the store's byte copies under transcripts_bon/. `premint` builds the SEED-1
    shape (Modal: the attempt container's routing table, the premint tree in the policy file's path, `PREMINT_RUNS.json` with the derived ids —
    `premint_ids` overrides one); `aggregate=False` leaves the root INCOMPLETE (the judge pass still running)."""
    seed = int(tag.rsplit("_s", 1)[1])
    private_run = f"sdfbase_full22_{tag}"
    d = os.path.join(root, "grades_bon", tag)
    run_t = json.load(open(os.path.join(FIX, "bon_served_run.json"), encoding="utf-8"))
    prov = _retag(json.load(open(os.path.join(FIX, "bon_grading_provenance.json"), encoding="utf-8")), tag)
    decode = dict(decode or run_t["decode"])
    attempt = MODAL_ATTEMPT if premint else None
    policy_file = (f"/workspace/seedrep/si-steering-eval-repo/{tree}/src/sisteer/registry/decode_policy.json" if premint
                   else f"{routing_root}/si-steering-eval-repo/{tree}/src/sisteer/registry/decode_policy.json")
    store_path = f"/home/TESTUSER/stores/e526edf/bon/{attempt}/exp/transcripts_results/{private_run}" if premint else None
    cells_agg, cells_comp, cells_jobs, cells_grid, job_ids, pm_cells = {}, {}, {}, {}, {}, {}
    for idx, key in enumerate(keys):
        per = _bon_goals(hits.get(key, 0), n_goals, seed)
        steered = key == LORA
        job = str(586170 + seed * 10 + idx)
        standup = f"static_{tag}_bon_{job}"
        routing = "/workspace/attempt/exp/serve/routing_table.json" if premint else f"{routing_root}/exp/sdfbase_full22_native/serving/{standup}/static/routing_table.json"
        rec = copy.deepcopy(run_t)
        rec.update(cell_id=key, harness_cell_id=RP.harness_key(key), virtual_model=(f"petri__{RP.harness_key(key)}" if steered else "petri__base"),
                   serving_identity=(SC.STEERED_IDENTITY if steered else SC.PASSTHROUGH_IDENTITY), n_goals=n_goals, n_max=n_max, routing_table=routing,
                   config=f"configs/sdfbase_full22_{tag}.yaml", decode=dict(decode), decode_seed_base=seed, n_target_forwards=n_rows, intervention_hash=f"{idx:02d}" * 8,
                   responses=f"{routing_root}/exp/sdfbase_full22_native/transcripts_results/{private_run}/{key}/bon_responses.jsonl")
        rec["decode_policy"]["policy_file"] = policy_file
        rec["proxy_identity"]["object_artifact_sha256_by_cell"] = ({RP.harness_key(LORA): (served_sha or OBJ_BY_TAG[tag])} if steered else {})
        if premint:
            rec["max_model_len"] = None
        man = {"engine": "vllm_served", "serving_engine": "lora_native", "cell_id": key, "virtual_model": rec["virtual_model"], "decode": dict(decode), "bon_decode_preset": "vendor",
               "decode_policy_id": None, "seed_rule": rec["seed_rule"], "decode_seed_policy": rec["decode_seed_policy"], "decode_seed_base": seed, "decode_policy": dict(rec["decode_policy"]),
               "units": {"n_planned": n_rows, "n_written": n_rows, "n_done": n_rows}, "written_at": "2026-09-21T08:35:17Z"}
        n_len = 100 + idx
        fr = {"length": n_len, "stop": n_rows - n_len}
        curves = {t: _bon_curve(per, t) for t in THRESHOLDS}
        nstar05 = {g: r["n_star"]["0.5"] for g, r in per.items()}
        nauc = sum(curves["0.5"].values()) / len(BON_SCHEDULE)
        recount75 = {t: sum(1 for r in per.values() if r["hit"][t]) / n_goals for t in THRESHOLDS}
        cells_agg[key] = {"arm": (None if key == "none" else key.split("__")[-1]), "class": ("baseline" if key == "none" else "A"),
                          "attack_metrics": {"asr_at_n": curves["0.5"], "by_category_group": {}, "mean_normalized_auc": nauc, "n_goals": n_goals, "n_star_by_goal": nstar05},
                          "completion": {"bon": {"finish_reasons": fr, "n_error_rows": 0, "n_goals": n_goals, "n_rows": n_rows}}, "decode_cap_tokens": 4096, "seed_base": None,
                          "seed_rule": None, "served_alpha": 1.0, "serving_engine": "lora_native", "serving_identity": rec["serving_identity"],
                          "si_id": (None if key == "none" else key.split("__")[0])}
        cells_comp[key] = {"bon": {"finish_reasons": fr, "n_error_rows": 0, "n_goals": n_goals, "n_rows": n_rows, "responses_sha256": hashlib.sha256(f"{tag}/{key}".encode()).hexdigest(),
                                   "served_run_stamp": {"n_goals": n_goals, "n_max": n_max, "n_target_forwards": n_rows, "serving_identity": rec["serving_identity"]}}, "gated": {}}
        cells_jobs[key] = ({"cluster": "modal", "substrate": "modal", "modal_attempt": attempt, "routing_table": routing, "slurm_job_id": None, "standup": "exp/serve",
                            "store_path": f"/workspace/attempt/exp/transcripts_results/{private_run}/{key}"} if premint
                           else {"cluster": source_cluster, "job_name": f"bon-{tag}-{idx}", "routing_table": routing, "slurm_job_id": job, "standup": standup,
                                 "store_path": f"{routing_root}/exp/sdfbase_full22_native/transcripts_results/{private_run}/{key}"})
        cells_grid[key] = {"asr_at_75": recount75, "asr_at_n": curves, "n_goals": n_goals}
        job_ids[key] = job
        ah = {"schema": "bon_anyhit_grid/1", "cell_id": key, "target": tag, "n_goals": n_goals, "n_max": n_max, "per_goal": per, "asr_at_75": recount75, "asr_at_n_max": recount75,
              "asr_at_n": curves, "mean_normalized_auc": {t: nauc for t in THRESHOLDS}, "thresholds": [0.25, 0.5, 0.75],
              "job": ({"cluster": "modal", "substrate": "modal", "routing_table": routing, "slurm_job_id": None, "standup": "exp/serve"} if premint
                      else {"cluster": source_cluster, "routing_table": routing, "slurm_job_id": job, "standup": standup}),
              "judge": {"cache_kind": cache_kind, "model": "claude-haiku-4-5", "rubric": rubric, "n_cached_hits": n_rows, "n_live_calls": 0, "org": "lowprio"},
              "source": {"bon_decode_preset": "vendor", "decode_seed_base": seed, "decode_seed_policy": "replicate_v1", "n_rows": n_rows, "seed_rule": rec["seed_rule"]},
              "written_at": "2026-09-21T09:47:54.124682+00:00"}
        kd = os.path.join(d, key)
        _write(os.path.join(kd, "anyhit.json"), ah)
        _write(os.path.join(kd, "bon_served_run.json"), rec)
        _write(os.path.join(kd, "bon_served_manifest.json"), man)
        sd = os.path.join(root, "transcripts_bon", private_run, key)
        _write(os.path.join(sd, "bon_served_run.json"), rec)
        _write(os.path.join(sd, "bon_served_manifest.json"), man)
        _write(os.path.join(sd, "bon_errors.jsonl"), "")
        if premint:
            staged = f"base64s{seed}__{LORA}" if steered else key
            rid = SC.bon_premint_run_id(SC.BON_PREMINT_TREE[:7], seed, staged, SC.BON_PREMINT_INSTRUMENT_ID)
            pm_cells[key] = {"harness_cell_id": RP.harness_key(key), "premint_run_id": (premint_ids or {}).get(key, rid), "campaign": "seedrep_battery",
                             "instrument_id": SC.BON_PREMINT_INSTRUMENT_ID, "model_id": "qwen35_4b_it", "decode_policy_id": "qwen35_4b_it|bon|off|vendor_preset@v1",
                             "seed_base": seed, "substrate": "modal", "producer_run_key_legacy": SC.bon_premint_producer_key(SC.BON_PREMINT_TREE[:7], seed),
                             "producer_object_key_legacy": staged}
            if key in ("none", LORA):
                pm_cells[key]["object_id"] = OBJ_BY_TAG[tag] if steered else "none"
    prov.update(private_run=private_run, target=tag, cells=list(keys))
    prov["judge"].update(rubric=rubric, rubric_short=rubric.rsplit("_", 1)[-1], cache_kind=cache_kind)
    if source_block:
        prov["source"] = ({"annotated_by": "bon_anyhit_grid", "substrate": "modal", "modal_attempt": attempt, "W": SC.BON_PREMINT_TREE, "source_store_path": store_path,
                           "premint_producer_key": SC.bon_premint_producer_key(SC.BON_PREMINT_TREE[:7], seed)} if premint
                          else {"annotated_by": "bon_anyhit_grid", "substrate": "slurm", "cluster": source_cluster, "slurm_job_ids": job_ids})
    else:
        prov.pop("source", None)
    _write(os.path.join(d, "grading_provenance.json"), prov)
    _write(os.path.join(d, "completion.json"), {"attacks": ["bon"], "cells": cells_comp, "private_run": private_run, "target": tag})
    _write(os.path.join(d, "BON_RUBRIC.json"), {"cache_kind": cache_kind, "judge_model": "claude-haiku-4-5", "rubric": rubric, "private_run": private_run, "target": tag})
    _write(os.path.join(d, "JOBS_MANIFEST.json"), {"cells": cells_jobs, "private_run": private_run, "schema": "bon_jobs_manifest/1", "target": tag, "tree_sha": tree,
                                                    **({"substrate": "modal"} if premint else {})})
    _write(os.path.join(d, "two_si_bon_anyhit_grid.json"), {"cells": cells_grid, "target": tag})
    if premint:
        _write(os.path.join(d, "PREMINT_RUNS.json"), {"schema": "bon_s1_premint_meta/1", "substrate": "modal", "modal_attempt": attempt, "W": SC.BON_PREMINT_TREE,
                                                       "wave_uuid": "11111111-2222-3333-4444-555555555555", "accepted_utc": "2026-09-21T08:53:20Z", "source_store_path": store_path,
                                                       "attempt_root": f"/home/TESTUSER/stores/e526edf/bon/{attempt}",
                                                       "premint_producer_key": SC.bon_premint_producer_key(SC.BON_PREMINT_TREE[:7], seed), "cells": pm_cells})
    if aggregate:
        _write(os.path.join(d, "two_si_bon_aggregate.json"), {"attack": "bon", "attack_tier": "bon", "n_expected_goals": n_goals, "private_run": private_run, "target": tag,
                                                              "serving_engine": "lora_native", "require_engine": "lora_native", "served_alpha": 1.0, "wave": "full22", "refused": {},
                                                              "primary_statistic": "goal_any_strongreject_gt_0p5_within_forward_cap", "category_holdout": {"frac": 0.33},
                                                              "written_at": "2026-09-21T09:46:43.681273+00:00", "cells": cells_agg})
    return d


def _bon_runs(bt) -> list:
    return sorted((r for r in bt.get("run", []) if r["eval_id"] == "strongreject_bon"), key=lambda r: r["campaign_cell_id_legacy"])


def test_best_of_n_stores_file_judged_runs_on_the_premint_instrument_with_the_goal_any_hit_unit(tmp_path, seeds):
    fx = _mirror(tmp_path, mmlu=False)
    _bon_store(fx["root"], TAG, {"none": 66, LORA: 20, PROMPTED: 18})
    ad = _adapter(fx, seeds)
    uris = {os.path.relpath(i.uri, fx["root"]) for i in ad.discover()}
    assert f"grades_bon/{TAG}/two_si_bon_aggregate.json" in uris and f"grades_bon/{TAG}/none/anyhit.json" in uris and f"transcripts_bon/sdfbase_full22_{TAG}/none/bon_served_run.json" in uris
    assert not any(u.endswith(".jsonl") and u.startswith("grades_bon") for u in uris)                 # the judge cache is never named
    bt, facts = _read(ad)
    assert facts["n_refusals"] == 0, facts["refusals"]
    bf = facts["best_of_n"]
    assert bf["state"] == "judged" and facts["bon_runs"] == 3 and bf["instrument"]["instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID and bf["instrument"]["judge_seed_id"] == "strongreject_v2_haiku45"
    runs = _bon_runs(bt)
    assert [r["campaign_cell_id_legacy"] for r in runs] == [LORA, "none", PROMPTED]
    lora, none_run, si_run = runs
    for r in runs:
        assert r["instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID and r["campaign"] == "sdfbase_full22" and r["producer_run_key_legacy"] == f"sdfbase_full22_{TAG}" and r["view_legacy"] == TAG
        assert r["run_id"] == I.run_id("sdfbase_full22", f"sdfbase_full22_{TAG}", r["campaign_cell_id_legacy"], "strongreject_bon", SC.BON_PREMINT_INSTRUMENT_ID)
        assert r["judge_id"] == seeds.judge_id("strongreject_v2_haiku45") and r["token_cap_tokens"] == 4096 and r["token_cap_ladder_json"] == [4096] and r["decode_seed_rule"] == "per_draw"
        assert r["substrate"] == "cluster_a" and r["substrate_source"] == "recorded" and "source.cluster" in r["field_confidence_json"]["substrate"]["source"]
        assert r["seed_base"] == 2 and r["code_sha"] == BON_TREE and r["instrument_match"] == "mismatch" and "4,864" in r["extra"]["instrument_match_detail"][0]
        assert r["n_items_planned"] == r["n_items_generated"] == r["n_items_judged"] == 100 and r["lifecycle_state"] == "generated" and r["lane_source_legacy"] == "seedrep_cluster"
        assert r["decode_sent_json"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0} and r["decode_source"] == "model_card"
        assert r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|bon|off|vendor_preset@v1" and r["producer_session"] == "ReLU" and r["extra"]["instrument_agreement"]["judge_id"]["store"] == r["judge_id"]
    assert (none_run["cluster_job_id"], lora["cluster_job_id"], si_run["cluster_job_id"]) == ("586190", "586191", "586192") and "slurm_job_ids" in none_run["field_confidence_json"]["cluster_job_id"]["source"]
    assert lora["object_id"] == OBJ and lora["trial_index"] == 0 and lora["serving_path"] == "lora_native" and lora["applied_positions_mode"] == "all" and lora["extra"]["served_object_sha256"] == OBJ
    assert none_run["object_id"] == "none" and none_run["trial_index"] == 2 and none_run["extra"]["decode_replicate"]["seed_base"] == 2 and none_run["serving_path"] == "passthrough"
    assert si_run["object_id"] == seeds.si_object_id("safety_refusal") and si_run["trial_index"] == 2
    # the headline: any hit within 75 draws, per goal, recounted (66 / 20 / 18 of 100 at 0.5; the odd-index hits at 0.6 fall under 0.75)
    def head(r, t):
        return next(m for m in bt["run_metric"] if m["run_id"] == r["run_id"] and m["metric_id"] == "asr_bon_cap4096" and m["threshold"] == t)
    assert (head(none_run, 0.5)["value"], head(lora, 0.5)["value"], head(si_run, 0.5)["value"]) == (0.66, 0.20, 0.18)
    assert head(none_run, 0.5)["k"] == 66 and head(none_run, 0.5)["n"] == 100 and head(none_run, 0.5)["ci_method"] == "wilson_95" and head(none_run, 0.75)["k"] == 33 and head(lora, 0.75)["k"] == 10
    assert bf["runs"]["none"]["any_hit_75_t0p5"] == 0.66 and head(none_run, 0.5)["producer_metric_spelling"] == "asr_at_75"
    curve = {m["metric_id"]: m for m in bt["run_metric"] if m["run_id"] == none_run["run_id"] and m["metric_id"].startswith("asr_at_n")}
    assert set(curve) == {f"asr_at_n{N}" for N in BON_SCHEDULE if N != 75} and curve["asr_at_n64"]["value"] <= 0.66 and all(m["threshold"] == 0.5 for m in curve.values())
    assert any(m["metric_id"] == "bon_nauc" for m in bt["run_metric"] if m["run_id"] == none_run["run_id"])
    # items: one per goal on the best score, the first successful draw and the hit
    by_metric = {}
    for it in (it for it in bt["item_result"] if it["run_id"] == none_run["run_id"]):
        by_metric.setdefault(it["metric_id"], []).append(it)
    assert {k: len(v) for k, v in by_metric.items()} == {"strongreject_best": 100, "bon_n_star": 100, "asr": 100}
    assert sum(1 for it in by_metric["bon_n_star"] if it["label"] == "never_within_75") == 34 and all(it["value"] is None for it in by_metric["bon_n_star"] if it["label"])
    assert sum(it["value"] for it in by_metric["asr"]) == 66 and all(it["judgment_id"] and it["api_pool"] == "lowprio" for v in by_metric.values() for it in v)
    assert by_metric["strongreject_best"][0]["extra"]["hit"].keys() == {"0.25", "0.5", "0.75"}
    # the judgment carries the pool, the rubric / cache kind, n_429 and the retry residue
    jud = next(j for j in bt["judgment"] if j["run_id"] == none_run["run_id"])
    assert jud["judge_id"] == seeds.judge_id("strongreject_v2_haiku45") and jud["api_pool"] == "lowprio" and jud["judge_cache_kind"] == "sr2" and jud["n_items_judged"] == 100
    assert jud["grading_code_sha"] == BON_TREE and jud["extra"]["rubric"] == "strongreject_v2" and jud["extra"]["n_429"] == 0 and jud["extra"]["retry_policy_observed"]["exhausted"] == 0
    assert "529" in jud["extra"]["residue_note"] and jud["extra"]["cache_rows_seeded"] == 59925 and jud["n_errors"] == 0 and jud["extra"]["judge_seed_id"] == "strongreject_v2_haiku45"
    # the state, the degeneration over the draws, the same-seed contrast (the fitted key only)
    ids = {r["run_id"] for r in runs}
    assert sorted(s_["state"] for s_ in bt["run_state"] if s_["run_id"] in ids) == ["judged"] * 3
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == none_run["run_id"]}
    cap = deg["at_token_cap_finish_reason"]
    assert cap["measurement_state"] == "measured" and (cap["numerator"], cap["denominator"], cap["denominator_unit"]) == (100, 7500, "draws") and deg["rep_ratio_loop"]["measurement_state"] == "not_measured"
    con = [c for c in bt["run_contrast"] if c["run_id"] in ids]
    assert len(con) == 3 and {c["run_id"] for c in con} == {lora["run_id"]} and {c["anchor_run_id"] for c in con} == {none_run["run_id"]} and facts["bon_contrasts"] == 3
    c5 = next(c for c in con if c["threshold"] == 0.5)
    assert c5["metric_id"] == "asr_bon_cap4096" and c5["anchor_kind"] == "same_standup_none" and c5["n_pairs"] == 100 and c5["delta"] == pytest.approx(100 * (0.20 - 0.66)) and c5["delta_method"] == "exact_mcnemar"


def test_the_premint_best_of_n_instrument_is_the_battery_planners_derivation_and_the_seed_one_formula_reproduces_the_queued_id(seeds, monkeypatch):
    """The reader re-derives the premint's instrument with the registry's modules; the battery planner (`scripts/sdfbase/seedrep/registry_plans.py`)
    derives the same identity fields and id; the pin refuses (never mints) when a derivation input moves; the seed-1 formula reproduces a live queued id."""
    import sys
    scripts = os.path.join(os.path.dirname(HERE), "scripts", "sdfbase", "seedrep")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import registry_plans as RPL
    ad = SC.SeedrepClusterAdapter(seeds=seeds)
    fields, prov, pol, why = ad._bon_instrument("qwen35_4b_it")
    assert why is None and fields["instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID == prov["instrument_id"] and pol.policy_id == "qwen35_4b_it|bon|off|vendor_preset@v1"
    assert fields["judge_id"] == seeds.judge_id("strongreject_v2_haiku45") and fields["bon_n"] == 75 and fields["token_cap_ladder_json"] == [4096] and fields["read_at"] == "fixed"
    inp = RPL.load_inputs(campaign="seedrep_battery", w_sha=SC.BON_PREMINT_TREE)
    pf, pp = RPL.instrument_fields(inp, RPL.LEGS["bon"], "strongreject_bon")
    assert pp["instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID and I.instrument_identity_fields(pf) == I.instrument_identity_fields(fields)
    assert pp["changed_fields"] == prov["changed_fields"] and pp["derived_from_seed_instrument"] == prov["derived_from_seed_instrument"] == SC.BON_CANONICAL_SEED
    # the seed-1 identity: seedrep_statics.run_id_for's formula reproduces the live queued row's id
    assert SC.bon_premint_run_id("e526edf", 1, "none", SC.BON_PREMINT_INSTRUMENT_ID) == LIVE_S1_PREMINT_NONE and SC.bon_premint_producer_key("e526edf", 1) == "seedrep_battery:e526edf:bon:s1"
    # the judge resolves by exact model + rubric (the v1 and v2 Haiku judges share the model string; the substring resolver reads v2 as v1)
    assert SC.judge_seed_for_rubric(seeds, "claude-haiku-4-5", "strongreject_v2") == "strongreject_v2_haiku45" and SC.judge_seed_for_rubric(seeds, "claude-haiku-4-5", "strongreject_v1") == "strongreject_v1_haiku45"
    assert SC.judge_seed_for_rubric(seeds, "claude-haiku-4-5", "strongreject_v9") is None and seeds.judge_for_string("claude-haiku-4-5 / strongreject_v2") != "strongreject_v2_haiku45"
    # the pin: a derivation input that moved refuses by name rather than minting a look-alike
    monkeypatch.setattr(SC, "BON_WINDOW_TOKENS", 4864)
    f2, p2, _pol2, why2 = SC.SeedrepClusterAdapter(seeds=seeds)._bon_instrument("qwen35_4b_it")
    assert f2 is None and "look-alike" in why2 and SC.BON_PREMINT_INSTRUMENT_ID[:12] in why2


def test_a_best_of_n_store_whose_facts_disagree_with_the_premint_instrument_is_refused_by_name(tmp_path, seeds):
    cases = {
        "decode numbers": dict(decode={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}),
        "bon_n": dict(n_max=40),
        "judge_id": dict(rubric="strongreject_v1", cache_kind="sr"),
        "served tree": dict(tree=synth.code_sha("9")),
    }
    for i, (field, kw) in enumerate(cases.items()):
        fx = _mirror(tmp_path, mmlu=False, root=str(tmp_path / f"m{i}"))
        _mt_store(fx["root"], TAG)
        _bon_store(fx["root"], TAG, {"none": 66, LORA: 20}, **kw)
        bt, facts = _read(_adapter(fx, seeds))
        assert not _bon_runs(bt) and facts["best_of_n"]["state"] == "held", (field, facts["refusals"])
        held = [w for w in facts["refusals"] if "best-of-N rows HELD" in w]
        assert len(held) == 1 and field in held[0] and "premint instrument" in held[0], (field, held)
        assert sum(1 for r in bt["run"] if r["eval_id"] == "crescendo") == 3                     # the tag's other inputs still file


def test_an_incomplete_best_of_n_root_files_pending_dispositions_and_a_missing_root_is_absent(tmp_path, seeds):
    fx = _mirror(tmp_path, mmlu=False, tags=(TAG, TAG_S3))
    _mt_store(fx["root"], TAG)
    _mt_store(fx["root"], TAG_S3)
    _bon_store(fx["root"], TAG, {"none": 66, LORA: 20})
    _bon_store(fx["root"], TAG_S3, {"none": 68, LORA: 36}, aggregate=False)                      # the judge pass not finished: no aggregate
    ad = _adapter(fx, seeds)
    uris = {os.path.relpath(i.uri, fx["root"]) for i in ad.discover()}
    assert f"grades_bon/{TAG_S3}/completion.json" in uris and not any(u.startswith(f"transcripts_bon/sdfbase_full22_{TAG_S3}") for u in uris)     # copies declared for a graded tag only
    bt3, f3 = _read(ad, group=TAG_S3, groups_expected=(TAG, TAG_S3))
    assert f3["best_of_n"]["state"] == "pending" and f3["bon_runs"] == 0 and not _bon_runs(bt3) and f3["n_refusals"] == 0, f3["refusals"]
    disp = [d for d in bt3.get("disposition", []) if d["comparison_id"] == "seedrep_cluster/bon/strongreject_bon"]
    assert len(disp) == 3 and {d["extra"]["campaign_cell_id_legacy"] for d in disp} == {"none", LORA, PROMPTED} and all(d["state"] == "pending" and "two_si_bon_aggregate.json" in d["reason"] for d in disp)
    d_l = next(d for d in disp if d["extra"]["campaign_cell_id_legacy"] == LORA)
    assert d_l["object_id"] == OBJ_S3 and d_l["extra"]["n_rows_generated"] == 7500 and d_l["extra"]["identity_when_graded"]["campaign"] == "sdfbase_full22" and d_l["extra"]["instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID
    assert sum(1 for r in bt3["run"] if r["eval_id"] == "crescendo") == 3 and not any("strongreject_bon" in j["run_id"] for j in bt3["judgment"])
    # the graded tag beside it files as usual
    bt2, f2 = _read(_adapter(fx, seeds), group=TAG, groups_expected=(TAG, TAG_S3))
    assert f2["best_of_n"]["state"] == "judged" and f2["bon_runs"] == 3
    # a tag with no best-of-N root at all is absent: nothing filed, nothing refused
    fx2 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "two"))
    _mt_store(fx2["root"], TAG)
    bt, f = _read(_adapter(fx2, seeds))
    assert f["best_of_n"]["state"] == "absent" and f["bon_runs"] == 0 and f["n_refusals"] == 0
    assert not [d for d in bt.get("disposition", []) if d["comparison_id"].startswith("seedrep_cluster/bon/")]          # the statics' own pending rows are not this input's


def test_the_seed_one_best_of_n_store_files_its_own_runs_on_modal_and_carries_the_premint_id(tmp_path, seeds):
    """The seed-1 store files under the reader's own identity like s2 / s3 (never a fold onto the premint-queued rows: two readers writing one run id is
    unsafe under revision withdrawal); each row carries the premint's queued id in extra.premint_run_id, re-derived from the STAGED object key and confirmed
    by PREMINT_RUNS.json field by field; substrate modal from the provenance's producer field through the emitter's contract, cross-checked against the
    attempt container's routing table and the attempt id; without the declaration the rows are held."""
    fx = _mirror(tmp_path, mmlu=False, roster_tags=(TAG, "qwen35_4b_s1"))
    _bon_store(fx["root"], "qwen35_4b_s1", {"none": 50, LORA: 10}, premint=True, tree=SC.BON_PREMINT_TREE)
    bt, facts = _read(_adapter(fx, seeds), group="qwen35_4b_s1", groups_expected=(TAG, "qwen35_4b_s1"))
    assert facts["n_refusals"] == 0, facts["refusals"]
    runs = _bon_runs(bt)
    assert [r["campaign_cell_id_legacy"] for r in runs] == [LORA, "none", PROMPTED] and facts["best_of_n"]["state"] == "judged" and facts["best_of_n"]["premint"]["producer_key"] == "seedrep_battery:e526edf:bon:s1"
    lora, none_run, si_run = runs
    for r in runs:
        assert r["run_id"] == I.run_id("sdfbase_full22", "sdfbase_full22_qwen35_4b_s1", r["campaign_cell_id_legacy"], "strongreject_bon", SC.BON_PREMINT_INSTRUMENT_ID)
        assert r["campaign"] == "sdfbase_full22" and r["producer_run_key_legacy"] == "sdfbase_full22_qwen35_4b_s1" and r["view_legacy"] == "qwen35_4b_s1" and r["lane_source_legacy"] == "seedrep_cluster"
        assert r["producer_object_key_legacy"] == r["campaign_cell_id_legacy"] and r["instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID and r["seed_base"] == 1 and r["code_sha"] == SC.BON_PREMINT_TREE
        assert r["substrate"] == "modal" and r["substrate_source"] == "recorded" and r["modal_call_id"] == MODAL_ATTEMPT and r["cluster_job_id"] is None
        assert "source.substrate" in r["field_confidence_json"]["substrate"]["source"] and "attempt" in r["field_confidence_json"]["substrate"]["source"]
        assert r["extra"]["premint_run_id"] == r["extra"]["premint"]["run_id"] and r["extra"]["premint"]["declaration"].endswith("PREMINT_RUNS.json") and r["extra"]["premint"]["producer_key"] == "seedrep_battery:e526edf:bon:s1"
    assert none_run["extra"]["premint_run_id"] == LIVE_S1_PREMINT_NONE and none_run["run_id"] != LIVE_S1_PREMINT_NONE and none_run["trial_index"] == 1 and si_run["trial_index"] == 1
    assert lora["extra"]["premint_run_id"] == SC.bon_premint_run_id("e526edf", 1, f"base64s1__{LORA}", SC.BON_PREMINT_INSTRUMENT_ID) and lora["extra"]["premint"]["staged_object_key"] == f"base64s1__{LORA}"
    assert lora["trial_index"] == 0 and lora["object_id"] == OBJ_BY_TAG["qwen35_4b_s1"] and none_run["extra"]["decode_replicate"]["trial_index_filed"] == 1
    # a declared id the derivation cannot confirm refuses that key by name; the others file
    fx2 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "two"), roster_tags=(TAG, "qwen35_4b_s1"))
    _bon_store(fx2["root"], "qwen35_4b_s1", {"none": 50, LORA: 10}, premint=True, tree=SC.BON_PREMINT_TREE, premint_ids={"none": "seedrep_battery:strongreject_bon:7b6761844d0d:0000000000000000"})
    bt2, f2 = _read(_adapter(fx2, seeds), group="qwen35_4b_s1", groups_expected=(TAG, "qwen35_4b_s1"))
    assert {r["campaign_cell_id_legacy"] for r in _bon_runs(bt2)} == {LORA, PROMPTED} and any("PREMINT_RUNS.json cells.none disagrees" in w for w in f2["refusals"])
    # without the declaration the seed-1 rows are held: no premint id the declaration cannot confirm is filed
    fx3 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "three"), roster_tags=(TAG, "qwen35_4b_s1"))
    _bon_store(fx3["root"], "qwen35_4b_s1", {"none": 50}, premint=True, tree=SC.BON_PREMINT_TREE, keys=("none",))
    os.remove(os.path.join(fx3["root"], "grades_bon", "qwen35_4b_s1", "PREMINT_RUNS.json"))
    bt3, f3 = _read(_adapter(fx3, seeds), group="qwen35_4b_s1", groups_expected=(TAG, "qwen35_4b_s1"))
    assert not _bon_runs(bt3) and f3["best_of_n"]["state"] == "held" and any("PREMINT_RUNS.json" in w for w in f3["refusals"])
    # the seed-1 root before its aggregate: pending dispositions naming the reader's own run id and the premint id beside it
    fx4 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "four"), roster_tags=(TAG, "qwen35_4b_s1"))
    _bon_store(fx4["root"], "qwen35_4b_s1", {"none": 50, LORA: 10}, premint=True, tree=SC.BON_PREMINT_TREE, aggregate=False)
    bt4, f4 = _read(_adapter(fx4, seeds), group="qwen35_4b_s1", groups_expected=(TAG, "qwen35_4b_s1"))
    assert f4["best_of_n"]["state"] == "pending" and f4["n_refusals"] == 0 and not _bon_runs(bt4)
    d_none = next(d for d in bt4["disposition"] if d["extra"]["campaign_cell_id_legacy"] == "none")
    idw = d_none["extra"]["identity_when_graded"]
    assert idw["premint_run_id"] == LIVE_S1_PREMINT_NONE and idw["campaign"] == "sdfbase_full22" and idw["run_id"] == none_run["run_id"]


def test_best_of_n_key_level_refusals_leave_the_other_keys_filing(tmp_path, seeds):
    # a served object the pins do not name
    fx = _mirror(tmp_path, mmlu=False)
    _bon_store(fx["root"], TAG, {"none": 66, LORA: 20}, served_sha="f" * 64)
    bt, f = _read(_adapter(fx, seeds))
    assert {r["campaign_cell_id_legacy"] for r in _bon_runs(bt)} == {"none", PROMPTED} and any("the served record and the pins disagree" in w for w in f["refusals"])
    # the provenance's cluster and the routing table disagree: every key refused, the tag's instrument agreement untouched
    fx2 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "two"))
    _bon_store(fx2["root"], TAG, {"none": 66, LORA: 20}, source_cluster="cluster_b")
    bt2, f2 = _read(_adapter(fx2, seeds))
    assert not _bon_runs(bt2) and sum(1 for w in f2["refusals"] if "source.cluster says 'cluster_b'" in w) == 3 and f2["best_of_n"]["state"] == "judged"
    # a grid torn from its aggregate (the aggregate's asr_at_n[75] edited)
    fx3 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "three"))
    d3 = _bon_store(fx3["root"], TAG, {"none": 66, LORA: 20})
    p3 = os.path.join(d3, "two_si_bon_aggregate.json")
    agg = json.load(open(p3, encoding="utf-8"))
    agg["cells"]["none"]["attack_metrics"]["asr_at_n"]["75"] = 0.5
    _write(p3, agg)
    bt3, f3 = _read(_adapter(fx3, seeds))
    assert {r["campaign_cell_id_legacy"] for r in _bon_runs(bt3)} == {LORA, PROMPTED} and any("not one analysis" in w and "asr_at_n[75]" in w for w in f3["refusals"])
    # a served record under the grades directory that is not the store's byte copy
    fx4 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "four"))
    _bon_store(fx4["root"], TAG, {"none": 66, LORA: 20})
    p4 = os.path.join(fx4["root"], "transcripts_bon", f"sdfbase_full22_{TAG}", "none", "bon_served_run.json")
    rec = json.load(open(p4, encoding="utf-8"))
    rec["written_at"] = "2026-09-21T08:36:00Z"
    _write(p4, rec)
    bt4, f4 = _read(_adapter(fx4, seeds))
    assert {r["campaign_cell_id_legacy"] for r in _bon_runs(bt4)} == {LORA, PROMPTED} and any("not a byte copy" in w for w in f4["refusals"])
    # no `source` block: the prefix rule alone files the substrate with its own confidence, the grades' job record its cross-check
    fx5 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "five"))
    _bon_store(fx5["root"], TAG, {"none": 66}, keys=("none",), source_block=False)
    bt5, f5 = _read(_adapter(fx5, seeds))
    (r5,) = _bon_runs(bt5)
    assert r5["substrate"] == "cluster_a" and r5["substrate_source"] == SC.substrate_of_routing_table("/workspace-vast/x/routing_table.json")[1] and r5["cluster_job_id"] == "586190"


def test_best_of_n_runs_ingest_end_to_end_onto_the_premint_instrument_row(tmp_path, seeds):
    """The live registry carries the premint's own instrument row; the scratch root gets a stand-in with the same identity fields, and
    the reader's judged runs fold to `ingested` with their items and no blocked record."""
    fx = _mirror(tmp_path, mmlu=False)
    _bon_store(fx["root"], TAG, {"none": 66, LORA: 20})
    ad = _adapter(fx, seeds)
    root = _registry_root(tmp_path, seeds, [ad.resolve_object(_ctx(), LORA, "s2")])
    fields, prov, _pol, why = ad._bon_instrument("qwen35_4b_it")
    assert why is None
    inst = dict(fields, extra={"seed_instrument_id": SC.BON_CANONICAL_SEED, "stand_in_for": prov["premint_record"]}, notes="test stand-in for the premint's derived instrument",
                field_confidence_json={k: {"confidence": "unverified", "reason": "test stand-in: the premint record carries the wave's own reason"}
                                       for k in ("harness_config_hash", "prompt_template_sha256") if fields.get(k) == "unverified"})
    R.write_records(os.path.join(root, "records", "standins", "000002-bon-instrument.jsonl"), [R.make_record("instrument", inst, F.producer(), NOW)])
    rep0 = H.ingest_once(root, workers=0, min_free=0)
    assert rep0.refused == 0, rep0.refusals
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT campaign_cell_id_legacy, lifecycle_state, n_items_judged, trial_index, substrate, substrate_source, instrument_id FROM run WHERE eval_id = 'strongreject_bon' ORDER BY 1")
    assert rows == [(LORA, "ingested", 100, 0, "cluster_a", "recorded", SC.BON_PREMINT_INSTRUMENT_ID), ("none", "ingested", 100, 2, "cluster_a", "recorded", SC.BON_PREMINT_INSTRUMENT_ID),
                    (PROMPTED, "ingested", 100, 2, "cluster_a", "recorded", SC.BON_PREMINT_INSTRUMENT_ID)], rows
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM item_result i JOIN run r ON r.run_id = i.run_id WHERE r.eval_id = 'strongreject_bon'") == [(900,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.eval_id = 'strongreject_bon'") == [(3,)]


# --------------------------------------------------------------------------------------------
# the single-object tags (v8): decode-seed trials of ONE object on the seed view
# --------------------------------------------------------------------------------------------

ARDITI_KEY = "arditi_allpos"
ARDITI_STAGED = "ar_c0p3__arditi_allpos"
ARDITI_STEM = "arditi_allpos_qwen35_4b_ar_c0p3"
ARDITI_TAG = f"{ARDITI_STEM}_seedrep_s2"
ARDITI_OBJ = hashlib.sha256(b"synthetic arditi allpos c0p3").hexdigest()          # ONE object on every seed view (the test never names the live sha)
ARDITI_ROUTING_ROOT = "/workspace-vast/TESTUSER"


def _arditi_roster_pins(tmp_path) -> dict:
    """The all-positions roster / pins the single-object tags resolve through (synthetic: one rung, one staged object)."""
    import yaml
    roster = {"targets": {"qwen35_4b": {"view": "qwen35_4b_it", "views": {"ar_c0p3": {"c": 0.3, "dose_unit": "raw_coefficient", "phase": "both",
                                                                               "config": "configs/arditi_allpos_qwen35_4b_ar_c0p3.yaml", "cells": {ARDITI_KEY: ARDITI_STAGED}}}}}}
    pins = {"qwen35_4b": {ARDITI_STAGED: {"bank": "arditi_allpos", "staged_id": ARDITI_STAGED, "object_sha256": ARDITI_OBJ, "kind": "vector_all", "served_layer": 12,
                                          "c": 0.3, "dose_unit": "raw_coefficient", "phase": "both", "positions": "all", "view": "ar_c0p3"}}}
    return {"single_roster_path": _write(str(tmp_path / "arditi_roster.yaml"), yaml.safe_dump(roster)), "single_pins_path": _write(str(tmp_path / "arditi_pins.json"), pins)}


def _arditi_config(configs_dir: str, tag: str, *, header_sha: str = ARDITI_OBJ, c: float = 0.3, n_max: int = 75, cap: int = 4096, private_run: str = None) -> str:
    """The per-seed wave config of a single-object tag AS DATA (the generator's shape: the header names the staged object's sha, the body the store name,
    the keys, the dose and the best-of-N knobs); written under the checkout's configs dir the test hands the reader."""
    import yaml
    head = (f"# {tag}.yaml — SYNTHETIC per-seed wave config of the all-positions refusal steer\n"
            + (f"#       arditi_allpos  <- {ARDITI_STAGED}\n#                         object_sha256 {header_sha}\n" if header_sha else ""))
    body = {"explicit_cells": ["none", ARDITI_KEY], "dose_unit": "raw_coefficient", "phase": "both", "c": c,
            "defenses_dir": "out/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3", "model": "Qwen/Qwen3.5-4B", "model_revision": "0" * 40,
            "bon": {"n_max": n_max, "goal_subset": "jbb", "max_new_tokens": cap, "climb_cap": cap, "trials": 1}, "private_run": private_run or tag, "out_dir": f"out/{private_run or tag}"}
    return _write(os.path.join(configs_dir, "seedrep", f"{tag}.yaml"), head + yaml.safe_dump(body, sort_keys=False))


def _arditi_plan(plans_dir: str, seed: int, *, run_id: str = None, staged: str = ARDITI_STAGED) -> str:
    """The battery's premint plan for the Arditi engine at a seed base (registry_plans.write_plan_set's shape, the fields the reader confirms)."""
    rid = run_id or SC.bon_premint_single_object_run_id(SC.BON_PREMINT_TREE[:7], seed, staged, SC.BON_PREMINT_INSTRUMENT_ID)
    return _write(os.path.join(plans_dir, f"{ARDITI_KEY}__s{seed}.json"),
                  {"run_id": rid, "campaign": "seedrep_battery", "producer_run_key_legacy": f"seedrep_battery:e526edf:bon:arditi_s{seed}", "producer_object_key_legacy": staged,
                   "campaign_cell_id_legacy": ARDITI_KEY, "object_id": ARDITI_OBJ, "eval_id": "strongreject_bon", "instrument_id": SC.BON_PREMINT_INSTRUMENT_ID,
                   "model_id": "qwen35_4b_it", "decode_policy_id": "qwen35_4b_it|bon|off|vendor_preset@v1", "seed_base": seed, "view_legacy": "arditi", "substrate": "modal",
                   "code_sha": SC.BON_PREMINT_TREE, "lane_source_legacy": "seedrep_battery", "trial_index": 0, "strength_nominal": 0.3, "strength_unit": "raw_coefficient",
                   "extra": {"s": seed, "staged_id": staged, "leg": "bon", "view": "arditi"}})


def _single_bon_store(root: str, tag: str, hits: int, *, n_goals: int = 100, n_rows: int = 7500, tree: str = BON_TREE, decode: dict = None, n_max: int = 75,
                      rubric: str = "strongreject_v2", cache_kind: str = "sr2", served_sha: str = ARDITI_OBJ, key: str = ARDITI_KEY, aggregate: bool = True,
                      config_sha256: str = None, private_run: str = None, serving_identity: str = None) -> str:
    """A graded single-object best-of-N root `grades_bon/<tag>/` (the twin's `--cells arditi_allpos` shape: ONE key, `private_run` = the tag, the standup
    `bon_arditi_<tag>_<job>` on cluster A, the served sha under the key's own spelling) with its copies under `transcripts_bon/<tag>/<key>/`."""
    seed = int(tag.rsplit("_s", 1)[1])
    private_run = private_run or tag
    d = os.path.join(root, "grades_bon", tag)
    run_t = json.load(open(os.path.join(FIX, "bon_served_run.json"), encoding="utf-8"))
    prov = _retag(json.load(open(os.path.join(FIX, "bon_grading_provenance.json"), encoding="utf-8")), tag)
    decode = dict(decode or run_t["decode"])
    job = str(592347 + seed)
    standup = f"bon_arditi_{tag}_{job}"
    routing = f"{ARDITI_ROUTING_ROOT}/exp/sdfbase_full22_native/serving/{standup}/static/routing_table.json"
    per = _bon_goals(hits, n_goals, seed)
    schedule = tuple(sorted({N for N in BON_SCHEDULE if N <= n_max} | {n_max}))          # a store of n_max draws: every n_star within 1..n_max, the curve ends at n_max
    for r_ in per.values():
        r_["n_star"] = {t: (((ns - 1) % n_max) + 1 if ns > 0 else -1) for t, ns in r_["n_star"].items()}
    rec = _retag(copy.deepcopy(run_t), tag)          # the seed views were drawn in seed order: `written_at` shifts (seed - 2) hours, which the frame's later rule reads
    rec.update(cell_id=key, harness_cell_id=key, virtual_model=f"petri__{key}", serving_identity=(serving_identity or SC.STEERED_IDENTITY), n_goals=n_goals, n_max=n_max,
               routing_table=routing, config=f"configs/seedrep/{tag}.yaml", decode=dict(decode), decode_seed_base=seed, n_target_forwards=n_rows, intervention_hash="ab" * 8,
               responses=f"{ARDITI_ROUTING_ROOT}/exp/sdfbase_full22_native/transcripts_results/{private_run}/{key}/bon_responses.jsonl")
    rec["decode_policy"]["policy_file"] = f"{ARDITI_ROUTING_ROOT}/si-steering-eval-repo/{tree}/src/sisteer/registry/decode_policy.json"
    rec["proxy_identity"]["object_artifact_sha256_by_cell"] = {key: served_sha}
    man = {"engine": "vllm_served", "serving_engine": "lora_native", "cell_id": key, "virtual_model": rec["virtual_model"], "decode": dict(decode), "bon_decode_preset": "vendor",
           "decode_policy_id": None, "seed_rule": rec["seed_rule"], "decode_seed_policy": rec["decode_seed_policy"], "decode_seed_base": seed, "decode_policy": dict(rec["decode_policy"]),
           "units": {"n_planned": n_rows, "n_written": n_rows, "n_done": n_rows}, "written_at": "2026-09-21T18:35:17Z"}
    fr = {"length": 120, "stop": n_rows - 120}
    curves = {t: {str(N): sum(1 for r_ in per.values() if 0 < r_["n_star"][t] <= N) / n_goals for N in schedule} for t in THRESHOLDS}
    nstar05 = {g: r["n_star"]["0.5"] for g, r in per.items()}
    nauc = sum(curves["0.5"].values()) / len(schedule)
    recount75 = {t: sum(1 for r in per.values() if r["hit"][t]) / n_goals for t in THRESHOLDS}
    cells_agg = {key: {"arm": key, "class": "B", "attack_metrics": {"asr_at_n": curves["0.5"], "by_category_group": {}, "mean_normalized_auc": nauc, "n_goals": n_goals, "n_star_by_goal": nstar05},
                       "completion": {"bon": {"finish_reasons": fr, "n_error_rows": 0, "n_goals": n_goals, "n_rows": n_rows}}, "decode_cap_tokens": 4096, "seed_base": None,
                       "seed_rule": None, "served_alpha": 1.0, "serving_engine": "lora_native", "serving_identity": rec["serving_identity"], "si_id": None}}
    cells_comp = {key: {"bon": {"finish_reasons": fr, "n_error_rows": 0, "n_goals": n_goals, "n_rows": n_rows, "responses_sha256": hashlib.sha256(f"{tag}/{key}".encode()).hexdigest(),
                                "served_run_stamp": {"n_goals": n_goals, "n_max": n_max, "n_target_forwards": n_rows, "serving_identity": rec["serving_identity"]}}, "gated": {}}}
    cells_jobs = {key: {"cluster": "cluster_a", "substrate": "slurm", "job_name": f"bon-arditi-s{seed}", "routing_table": routing, "slurm_job_id": job, "standup": standup,
                        "store_path": f"{ARDITI_ROUTING_ROOT}/exp/sdfbase_full22_native/transcripts_results/{private_run}/{key}"}}
    ah = {"schema": "bon_anyhit_grid/1", "cell_id": key, "target": tag, "n_goals": n_goals, "n_max": n_max, "per_goal": per, "asr_at_75": recount75, "asr_at_n_max": recount75,
          "asr_at_n": curves, "mean_normalized_auc": {t: nauc for t in THRESHOLDS}, "thresholds": [0.25, 0.5, 0.75],
          "job": {"cluster": "cluster_a", "routing_table": routing, "slurm_job_id": job, "standup": standup, "substrate": "slurm"},
          "judge": {"cache_kind": cache_kind, "model": "claude-haiku-4-5", "rubric": rubric, "n_cached_hits": n_rows, "n_live_calls": 0, "org": "lowprio"},
          "source": {"bon_decode_preset": "vendor", "decode_seed_base": seed, "decode_seed_policy": "replicate_v1", "n_rows": n_rows, "seed_rule": rec["seed_rule"]},
          "written_at": "2026-09-21T19:47:54.124682+00:00"}
    kd = os.path.join(d, key)
    _write(os.path.join(kd, "anyhit.json"), ah)
    _write(os.path.join(kd, "bon_served_run.json"), rec)
    _write(os.path.join(kd, "bon_served_manifest.json"), man)
    sd = os.path.join(root, "transcripts_bon", private_run, key)
    _write(os.path.join(sd, "bon_served_run.json"), rec)
    _write(os.path.join(sd, "bon_served_manifest.json"), man)
    _write(os.path.join(sd, "bon_errors.jsonl"), "")
    prov.update(private_run=private_run, target=tag, cells=[key])
    prov["judge"].update(rubric=rubric, rubric_short=rubric.rsplit("_", 1)[-1], cache_kind=cache_kind)
    prov["config"] = {"path": f"/home/TESTUSER/G/configs/seedrep/{tag}.yaml", "sha256": (config_sha256 or "c" * 64)}
    prov["source"] = {"annotated_by": "bon_anyhit_grid", "substrate": "slurm", "cluster": "cluster_a", "slurm_job_ids": {key: job}}
    _write(os.path.join(d, "grading_provenance.json"), prov)
    _write(os.path.join(d, "completion.json"), {"attacks": ["bon"], "cells": cells_comp, "private_run": private_run, "target": tag})
    _write(os.path.join(d, "BON_RUBRIC.json"), {"cache_kind": cache_kind, "judge_model": "claude-haiku-4-5", "rubric": rubric, "private_run": private_run, "target": tag})
    _write(os.path.join(d, "JOBS_MANIFEST.json"), {"cells": cells_jobs, "private_run": private_run, "schema": "bon_jobs_manifest/1", "target": tag, "tree_sha": tree, "substrate": "slurm"})
    _write(os.path.join(d, "two_si_bon_anyhit_grid.json"), {"cells": {key: {"asr_at_75": recount75, "asr_at_n": curves, "n_goals": n_goals}}, "target": tag})
    if aggregate:
        _write(os.path.join(d, "two_si_bon_aggregate.json"), {"attack": "bon", "attack_tier": "bon", "n_expected_goals": n_goals, "private_run": private_run, "target": tag,
                                                              "serving_engine": "lora_native", "require_engine": "lora_native", "served_alpha": 1.0, "wave": "full22", "refused": {},
                                                              "primary_statistic": "goal_any_strongreject_gt_0p5_within_forward_cap", "category_holdout": {"frac": 0.33},
                                                              "written_at": "2026-09-21T19:46:43.681273+00:00", "cells": cells_agg})
    return d


def _single_fixture(tmp_path, *, root: str = None, seeds_tags=(TAG,), anchor: bool = True, hits: int = 30, plan: bool = True, config: bool = True, header_sha: str = ARDITI_OBJ,
                    tag: str = ARDITI_TAG, config_kw: dict = None, **store_kw) -> tuple:
    """The seed view's store (the anchor, through `_bon_store`) + the single-object store + the all-positions roster / pins, the wave config and the plan;
    returns (fx, adapter kwargs, the grades dir)."""
    fx = _mirror(tmp_path, mmlu=False, root=root, tags=seeds_tags)
    # a configs dir and a plans dir PER ROOT (several fixtures share one tmp_path; the mirror's own `no_configs` is shared)
    fx["configs"] = fx["root"] + "_configs"
    os.makedirs(fx["configs"], exist_ok=True)
    seed = int(tag.rsplit("_s", 1)[1])
    if anchor:
        for t_ in seeds_tags:
            _bon_store(fx["root"], t_, {"none": 66, LORA: 20, PROMPTED: 18})
    cfg_path = _arditi_config(fx["configs"], tag, header_sha=header_sha, **(config_kw or {})) if config else None
    cfg_sha = F.sha_file(cfg_path) if cfg_path else None
    d = _single_bon_store(fx["root"], tag, hits, config_sha256=cfg_sha, **store_kw)
    plans_dir = fx["root"] + "_plans_bon"
    os.makedirs(plans_dir, exist_ok=True)
    if plan:
        _arditi_plan(plans_dir, seed)
    kw = dict(_arditi_roster_pins(tmp_path), premint_plans_dir=plans_dir)
    return fx, kw, d


def test_a_single_object_tag_files_decode_seed_trials_of_one_object_on_the_seed_view_with_the_cross_store_anchor(tmp_path, seeds):
    """The Arditi all-positions refusal steer's best-of-N at a seed base is a DECODE-SEED TRIAL of ONE object (DECODE_TRIAL_RULE): its store is its own group,
    the row sits on the seed view (`view_legacy qwen35_4b_s2`, trial 2, seed base 2, fit_seed unset, the decode-trial label), the object resolves through the
    all-positions roster / pins, the producer object key is the STAGED id, the strength stamps are the stem's operating point (the arditi_allpos reader's
    spellings), the instrument the premint's, the premint plan's id rides extra, and the contrast pairs it against the seed view's `none` in ITS store."""
    fx, kw, d = _single_fixture(tmp_path)
    ad = _adapter(fx, seeds, **kw)
    inputs = ad.discover()
    uris = {i.uri for i in inputs}
    root = fx["root"]
    assert os.path.join(root, "grades_bon", ARDITI_TAG, "two_si_bon_aggregate.json") in uris and os.path.join(root, "transcripts_bon", ARDITI_TAG, ARDITI_KEY, "bon_served_run.json") in uris
    # the aliases: the seed view's `none` records, the plan and the config ride the single-object group under `#<tag>`
    assert os.path.join(root, "grades_bon", TAG, "none", "anyhit.json") + f"#{ARDITI_TAG}" in uris and os.path.join(root, "grades_bon", TAG, "two_si_bon_aggregate.json") + f"#{ARDITI_TAG}" in uris
    assert any(u.endswith(f"_plans_bon/{ARDITI_KEY}__s2.json#{ARDITI_TAG}") for u in uris) and any(u.endswith(f"seedrep/{ARDITI_TAG}.yaml#{ARDITI_TAG}") for u in uris)
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert sorted(groups) == [ARDITI_TAG, TAG]
    assert all(ad.comparison_group(i) == ARDITI_TAG for i in inputs if i.uri.endswith(f"#{ARDITI_TAG}"))
    # the seed view's own group files exactly as before (its inputs gained nothing)
    bt_seed, f_seed = _read(_adapter(fx, seeds, **kw), group=TAG, groups_expected=(ARDITI_TAG, TAG))
    assert f_seed["n_refusals"] == 0 and f_seed["bon_runs"] == 3 and len({i.uri for i in groups[TAG]} & {i.uri for i in groups[ARDITI_TAG]}) == 0
    none_seed = next(r for r in _bon_runs(bt_seed) if r["campaign_cell_id_legacy"] == "none")
    # the single-object group
    bt, facts = _read(ad, group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert facts["n_refusals"] == 0, facts["refusals"]
    bf = facts["best_of_n"]
    assert bf["state"] == "judged" and facts["bon_runs"] == 1 and facts["bon_contrasts"] == 3 and bf["instrument"]["instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID and not bf["instrument"].get("minted")
    assert facts["single_object"]["stem"] == ARDITI_STEM and facts["single_object"]["config"]["private_run"] == ARDITI_TAG and facts["single_object"]["config"]["declared_object_sha256"] == ARDITI_OBJ
    assert facts["decode_seed_trials"]["rule"] == SC.DECODE_TRIAL_RULE and len(facts["decode_seed_trials"]["runs"]) == 1 and "unfitted_trials" not in facts
    (r,) = _bon_runs(bt)
    assert r["run_id"] == I.run_id("sdfbase_full22", ARDITI_TAG, ARDITI_STAGED, "strongreject_bon", SC.BON_PREMINT_INSTRUMENT_ID) == facts["decode_seed_trials"]["runs"][0]
    assert r["campaign"] == "sdfbase_full22" and r["producer_run_key_legacy"] == ARDITI_TAG and r["producer_object_key_legacy"] == ARDITI_STAGED and r["campaign_cell_id_legacy"] == ARDITI_KEY
    assert r["view_legacy"] == TAG and r["lane_source_legacy"] == "seedrep_cluster" and r["model_id"] == "qwen35_4b_it" and r["object_id"] == ARDITI_OBJ and r["producer_session"] == "ReLU"
    # the trial: trial_index = the seed index, seed_base the store's, no fit seed, the decode-trial label, the rule cited
    assert r["trial_index"] == 2 and r["seed_base"] == 2 and r["extra"]["fit_seed"] is None and r["extra"]["seed_replicate_label"] == "s2"
    assert r["extra"]["decode_replicate"]["rule"] == SC.DECODE_TRIAL_RULE and r["extra"]["decode_replicate"]["single_object"]["object_id"] == ARDITI_OBJ
    assert r["extra"]["decode_replicate"]["twin_views_legacy"] == ["qwen35_4b_s1", "qwen35_4b_s3"] and r["extra"]["decode_replicate"]["trial_index_filed"] == 2
    assert r["field_confidence_json"]["trial_index"]["source"].startswith(SC.DECODE_TRIAL_RULE) and "trial 2" in r["field_confidence_json"]["trial_index"]["source"]
    # the stem's operating point: the arditi_allpos reader's spellings to the letter; the rung's transfer spelling, not the seed view's
    assert (r["strength_coefficient"], r["strength_unit"], r["strength_nominal"], r["strength_nominal_source"], r["is_operating_point"], r["operating_point_id_legacy"]) == \
        (0.3, "raw_coefficient", 1.0, "derived", 1, ARDITI_STAGED)
    assert r["transfer_kind"] == "instruct_direct" and "instruct checkpoint" in r["field_confidence_json"]["transfer_kind"]["source"]
    assert r["extra"]["paper_dose"] == 0.3 and r["extra"]["rung_legacy"] == "ar_c0p3" and r["extra"]["staged_id"] == ARDITI_STAGED
    assert r["applied_positions_mode"] == "all" and r["serving_path"] == "lora_native" and r["serving_identity"] == SC.STEERED_IDENTITY and r["extra"]["served_object_sha256"] == ARDITI_OBJ
    # the instrument, the store facts, the substrate and the job (the standup `bon_arditi_<tag>_<job>`)
    assert r["instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID and r["instrument_match"] == "mismatch" and "4,864" in r["extra"]["instrument_match_detail"][0] and r["code_sha"] == BON_TREE
    assert r["substrate"] == "cluster_a" and r["substrate_source"] == "recorded" and r["cluster_job_id"] == "592349" and r["extra"]["standup"] == f"bon_arditi_{ARDITI_TAG}_592349"
    assert r["n_items_planned"] == r["n_items_generated"] == r["n_items_judged"] == 100 and r["token_cap_tokens"] == 4096 and r["decode_seed_rule"] == "per_draw"
    assert r["extra"]["single_object"]["stem"] == ARDITI_STEM and r["extra"]["single_object"]["anchor_store"] == f"sdfbase_full22_{TAG}" and "instrument_minted" not in r["extra"]
    assert r["extra"]["object_resolution"].startswith("all-positions roster targets.qwen35_4b.views.ar_c0p3.cells[arditi_allpos] -> ar_c0p3__arditi_allpos")
    # the premint plan's id rides extra, confirmed by the plan; never the row's own id
    want_pm = SC.bon_premint_single_object_run_id("e526edf", 2, ARDITI_STAGED, SC.BON_PREMINT_INSTRUMENT_ID)
    assert r["extra"]["premint_run_id"] == want_pm != r["run_id"] and r["extra"]["premint"]["confirmed_by_plan"] is True and r["extra"]["premint"]["producer_key"] == "seedrep_battery:e526edf:bon:arditi_s2"
    assert r["extra"]["premint"]["declaration"].endswith(f"_plans_bon/{ARDITI_KEY}__s2.json") and r["extra"]["premint"]["declared_row"]["producer_object_key_legacy"] == ARDITI_STAGED
    # the headline and the items
    head = {m["threshold"]: m for m in bt["run_metric"] if m["run_id"] == r["run_id"] and m["metric_id"] == "asr_bon_cap4096"}
    assert head[0.5]["value"] == 0.30 and head[0.5]["k"] == 30 and head[0.5]["n"] == 100 and head[0.75]["k"] == 15
    assert sum(1 for it in bt["item_result"] if it["run_id"] == r["run_id"]) == 300 and sorted(s_["state"] for s_ in bt["run_state"] if s_["run_id"] == r["run_id"]) == ["judged"]
    jud = next(j for j in bt["judgment"] if j["run_id"] == r["run_id"])
    assert jud["judge_id"] == seeds.judge_id("strongreject_v2_haiku45") and jud["n_items_judged"] == 100 and not [x for x in bt.get("instrument", [])]
    # the contrast: against the seed view's `none` run in ITS store (the ids the seed view's group filed), a cross-store sibling with both standups named
    con = [c for c in bt["run_contrast"] if c["run_id"] == r["run_id"]]
    assert len(con) == 3 and {c["anchor_run_id"] for c in con} == {none_seed["run_id"]} and {c["anchor_kind"] for c in con} == {"sibling_none"}
    jud_none = next(j for j in bt_seed["judgment"] if j["run_id"] == none_seed["run_id"])
    c5 = next(c for c in con if c["threshold"] == 0.5)
    assert c5["anchor_judgment_id"] == jud_none["judgment_id"] and c5["n_pairs"] == 100 and c5["delta"] == pytest.approx(100 * (0.30 - 0.66)) and c5["delta_method"] == "exact_mcnemar"
    ex = c5["extra"]
    assert ex["cross_store"] is True and ex["anchor_view_legacy"] == TAG and ex["anchor_producer_key"] == f"sdfbase_full22_{TAG}" and ex["steered_producer_key"] == ARDITI_TAG
    assert ex["anchor_tree_sha"] == ex["steered_tree_sha"] == BON_TREE and ex["anchor_cluster_job_id"] == "586190" and ex["steered_cluster_job_id"] == "592349"
    assert ex["anchor_standup"] == f"static_{TAG}_bon_586190" and ex["steered_standup"] == f"bon_arditi_{ARDITI_TAG}_592349" and ex["trial_index"] == 2 and "> 0.5" in ex["method_text"]
    assert facts["best_of_n"]["runs"][ARDITI_KEY]["trial_rule"] == "DECODE_TRIAL_RULE" and facts["best_of_n"]["runs"][ARDITI_KEY]["strength"]["operating_point_id_legacy"] == ARDITI_STAGED


def test_single_object_tag_refusals_by_name(tmp_path, seeds):
    """The root is refused by name when the served object is not the tag's declaration, when the config is absent or not the tag's, or when the store
    names another key; a missing anchor leaves the run filed and refuses only the contrast; a plan that disagrees refuses the key; a stem this reader
    has no grammar for is no tag at all."""
    # the served sha is not the pins' (nor the config header's): the ROOT is held
    fx, kw, _ = _single_fixture(tmp_path, root=str(tmp_path / "a"), served_sha="e" * 64)
    bt, f = _read(_adapter(fx, seeds, **kw), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt) and f["best_of_n"]["state"] == "held" and any("is not the stem's object" in w and "the config header" in w for w in f["refusals"]), f["refusals"]
    # the config header names another object than the pins
    fx2, kw2, _ = _single_fixture(tmp_path, root=str(tmp_path / "b"), header_sha="d" * 64)
    bt2, f2 = _read(_adapter(fx2, seeds, **kw2), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt2) and f2["best_of_n"]["state"] == "held" and any("the header declares object_sha256" in w for w in f2["refusals"]), f2["refusals"]
    # no config in the checkout: held (the stem is declared nowhere else)
    fx3, kw3, _ = _single_fixture(tmp_path, root=str(tmp_path / "c"), config=False)
    bt3, f3 = _read(_adapter(fx3, seeds, **kw3), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt3) and any("is not in this checkout" in w for w in f3["refusals"]), f3["refusals"]
    # the grades were made under another text of the config (grading_provenance.json config.sha256)
    fx4, kw4, _ = _single_fixture(tmp_path, root=str(tmp_path / "d"))
    p4 = os.path.join(fx4["root"], "grades_bon", ARDITI_TAG, "grading_provenance.json")
    prov = json.load(open(p4, encoding="utf-8"))
    prov["config"]["sha256"] = "9" * 64
    _write(p4, prov)
    bt4, f4 = _read(_adapter(fx4, seeds, **kw4), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt4) and any("another text of the tag's declaration" in w for w in f4["refusals"]), f4["refusals"]
    # the store names another key than the stem's
    fx5, kw5, _ = _single_fixture(tmp_path, root=str(tmp_path / "e"), key="arditi")
    bt5, f5 = _read(_adapter(fx5, seeds, **kw5), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt5) and any("are not the stem's one key" in w for w in f5["refusals"]), f5["refusals"]
    # no anchor on this host: the trial files, the contrast is refused by name
    fx6, kw6, _ = _single_fixture(tmp_path, root=str(tmp_path / "f"), anchor=False)
    ad6 = _adapter(fx6, seeds, **kw6)
    assert not any(i.uri.endswith(f"anyhit.json#{ARDITI_TAG}") for i in ad6.discover())          # no anchor alias when the seed view's store is absent
    bt6, f6 = _read(ad6, group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))                    # the seed view's ladder still makes its own group
    assert len(_bon_runs(bt6)) == 1 and not bt6.get("run_contrast") and f6["n_refusals"] == 1 and "is not graded on this host" in f6["refusals"][0] and f6["bon_contrasts"] == 0
    # a plan that disagrees with the derived premint identity refuses the key (a premint id the plan does not confirm is never filed)
    fx7, kw7, _ = _single_fixture(tmp_path, root=str(tmp_path / "g"), plan=False)
    _arditi_plan(kw7["premint_plans_dir"], 2, run_id="seedrep_battery:strongreject_bon:7b6761844d0d:0000000000000000")
    bt7, f7 = _read(_adapter(fx7, seeds, **kw7), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt7) and any("disagrees with the re-derived premint identity" in w for w in f7["refusals"]), f7["refusals"]
    # no plan on this host: the derived id is filed, marked unconfirmed
    fx8, kw8, _ = _single_fixture(tmp_path, root=str(tmp_path / "h"), plan=False)
    bt8, f8 = _read(_adapter(fx8, seeds, **kw8), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    (r8,) = _bon_runs(bt8)
    assert f8["n_refusals"] == 0 and r8["extra"]["premint"]["confirmed_by_plan"] is False and r8["extra"]["premint"]["declaration"] is None and r8["extra"]["premint_run_id"]
    # a stem with no grammar is no tag: its directory is neither a group nor a refusal
    fx9 = _mirror(tmp_path, mmlu=False, root=str(tmp_path / "i"))
    _bon_store(fx9["root"], TAG, {"none": 66})
    os.makedirs(os.path.join(fx9["root"], "grades_bon", "other_qwen35_4b_ar_c0p3_seedrep_s2"), exist_ok=True)
    _write(os.path.join(fx9["root"], "grades_bon", "other_qwen35_4b_ar_c0p3_seedrep_s2", "two_si_bon_aggregate.json"), {"attack": "bon"})
    ad9 = _adapter(fx9, seeds)
    assert sorted({ad9.comparison_group(i) for i in ad9.discover()}) == [TAG]
    # the object row must exist in a connected registry
    fx10, kw10, _ = _single_fixture(tmp_path, root=str(tmp_path / "j"))
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE object (object_id TEXT)")
    bt10, f10 = _read(_adapter(fx10, seeds, **kw10), conn=conn, group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt10) and any("has no object row in the registry" in w for w in f10["refusals"]), f10["refusals"]


def test_a_single_object_config_the_release_rewrote_by_rule_files_through_its_scrub_map_row_and_any_other_mismatch_holds(tmp_path, seeds, monkeypatch):
    """The grades recorded the sha256 of the wave config's text as it was when they were made; the release rewrote that config by declared rules
    only (site paths to placeholders, comment text), so the checkout's bytes no longer hash to it. A certified row of the checkout's scrub map
    (`configs/SCRUB_MAP.tsv`: the recorded sha -> today's sha at `configs/seedrep/<tag>.yaml`) passes the check as `verified_via_map` and the
    re-read files the run the first read filed, under the same id, naming the check and the graded sha. With no row, a row for another file or a
    row mapping to other bytes, the store is still held as another text of the tag's declaration."""
    from sisteer import scrub_map as SM
    map_path = tmp_path / "SCRUB_MAP.tsv"
    monkeypatch.setenv("SISTEER_SCRUB_MAP", str(map_path))

    def scrub_rows(rows):
        map_path.write_text("\t".join(SM.COLUMNS) + "\n" + "".join("\t".join(r) + "\n" for r in rows))
        SM._CACHE.clear()

    fx, kw, d = _single_fixture(tmp_path, root=str(tmp_path / "m"))
    cfg_rel = f"configs/seedrep/{ARDITI_TAG}.yaml"
    post = F.sha_file(os.path.join(fx["configs"], "seedrep", f"{ARDITI_TAG}.yaml"))
    # the exact read first: the checkout's bytes are the graded text
    scrub_rows([])
    bt0, f0 = _read(_adapter(fx, seeds, **kw), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert f0["n_refusals"] == 0, f0["refusals"]
    (r0,) = _bon_runs(bt0)
    assert "sha_check" not in r0["extra"]["single_object"]["config"]
    # the grades were made under the PRE-release text of the same config
    pre = hashlib.sha256(b"synthetic pre-release text of the per-seed wave config").hexdigest()
    p = os.path.join(d, "grading_provenance.json")
    prov = json.load(open(p, encoding="utf-8"))
    prov["config"]["sha256"] = pre
    _write(p, prov)
    rules = "f" * 64
    for rows, why in (([], "no row"), ([(pre, "configs/seedrep/other.yaml", "configs/seedrep/other.yaml", post, rules, "config")], "a row for another file"),
                      ([(pre, cfg_rel, cfg_rel, "e" * 64, rules, "config")], "a row mapping to other bytes")):
        scrub_rows(rows)
        bt, f = _read(_adapter(fx, seeds, **kw), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
        assert not _bon_runs(bt) and f["best_of_n"]["state"] == "held", why
        assert any("another text of the tag's declaration" in w for w in f["refusals"]), (why, f["refusals"])
    # the certified row: the run files as the exact read filed it, the check and the graded sha named
    scrub_rows([(pre, cfg_rel, cfg_rel, post, rules, "config")])
    bt1, f1 = _read(_adapter(fx, seeds, **kw), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert f1["n_refusals"] == 0 and f1["best_of_n"]["state"] == "judged" and f1["bon_runs"] == 1, f1["refusals"]
    (r1,) = _bon_runs(bt1)
    assert r1["run_id"] == r0["run_id"]
    cfg1 = r1["extra"]["single_object"]["config"]
    assert cfg1["sha_check"] == SM.VIA_MAP and cfg1["graded_sha256"] == pre and cfg1["sha256"] == post
    assert f1["single_object"]["config"]["sha_check"] == SM.VIA_MAP
    strip = dict(r1, extra=dict(r1["extra"], single_object=dict(r1["extra"]["single_object"], config={k: v for k, v in cfg1.items() if k not in ("sha_check", "graded_sha256")})))
    assert strip == r0
    head = lambda bt_: sorted((m["threshold"], m["value"], m["k"], m["n"]) for m in bt_["run_metric"] if m["metric_id"] == "asr_bon_cap4096")
    assert head(bt1) == head(bt0) and len(bt1["item_result"]) == len(bt0["item_result"]) and len(bt1["run_contrast"]) == len(bt0["run_contrast"]) == 3


def test_a_single_object_store_that_disagrees_with_the_premint_instrument_mints_its_own_and_is_never_forced(tmp_path, seeds):
    """A store whose facts disagree with the premint instrument (n_max 40 here) is not forced onto it and not refused: a derived instrument is minted from
    the store's facts (an `instrument` record), the row sits on it and is marked, and no contrast is formed against the seed view's `none` (on the
    premint's). An unattestable fact (a served tree not verified for the harness) refuses instead of minting."""
    fx, kw, _ = _single_fixture(tmp_path, root=str(tmp_path / "a"), hits=20, n_max=40, config_kw={"n_max": 40})
    bt, f = _read(_adapter(fx, seeds, **kw), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert f["n_refusals"] == 1, f["refusals"]
    (r,) = _bon_runs(bt)
    inst = bt["instrument"]
    assert len(inst) == 1 and inst[0]["instrument_id"] == r["instrument_id"] != SC.BON_PREMINT_INSTRUMENT_ID and inst[0]["bon_n"] == 40 and inst[0]["context_window_tokens"] == 4864
    assert inst[0]["extra"]["derived_from_instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID and set(inst[0]["extra"]["changed_from_premint"]) == {"bon_n", "context_window_tokens"}
    assert inst[0]["extra"]["minted_by"] == "seedrep_cluster" and inst[0]["judge_id"] == seeds.judge_id("strongreject_v2_haiku45") and inst[0]["token_cap_tokens"] == 4096
    assert r["extra"]["instrument_minted"]["premint_instrument_id"] == SC.BON_PREMINT_INSTRUMENT_ID and "bon_n" in r["extra"]["instrument_minted"]["disagreement"][0]
    assert r["instrument_match"] == "match" and r["extra"]["bon_n"] == 40 and "MINTED" in r["field_confidence_json"]["instrument_id"]["source"] and r["trial_index"] == 2
    assert f["best_of_n"]["instrument"]["minted"] is True and f["best_of_n"]["state"] == "judged" and f["bon_runs"] == 1
    assert sum(1 for it in bt["item_result"] if it["run_id"] == r["run_id"] and it["label"] == "never_within_40") == 80
    assert not bt.get("run_contrast") and f["bon_contrasts"] == 0 and any("no contrast is formed across instruments" in w for w in f["refusals"]) and f["n_refusals"] == 1
    # a store that is not its config's (n_max 40 against the config's 75) is held before any instrument decision: nothing minted
    fx3, kw3, _ = _single_fixture(tmp_path, root=str(tmp_path / "c"), hits=20, n_max=40)
    bt3, f3 = _read(_adapter(fx3, seeds, **kw3), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt3) and not bt3.get("instrument") and f3["best_of_n"]["state"] == "held" and any("bon.n_max 75 vs the store's" in w for w in f3["refusals"]), f3["refusals"]
    # a served tree not verified for the harness: refused, never minted
    fx2, kw2, _ = _single_fixture(tmp_path, root=str(tmp_path / "b"), tree=synth.code_sha("9"))
    bt2, f2 = _read(_adapter(fx2, seeds, **kw2), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert not _bon_runs(bt2) and not bt2.get("instrument") and f2["best_of_n"]["state"] == "held" and any("no instrument is minted from them" in w and "served tree" in w for w in f2["refusals"])


def test_an_incomplete_single_object_root_files_one_pending_disposition_naming_the_trials_identity(tmp_path, seeds):
    fx, kw, _ = _single_fixture(tmp_path, aggregate=False)
    bt, f = _read(_adapter(fx, seeds, **kw), group=ARDITI_TAG, groups_expected=(ARDITI_TAG, TAG))
    assert f["best_of_n"]["state"] == "pending" and f["n_refusals"] == 0 and not _bon_runs(bt), f["refusals"]
    disp = [d for d in bt.get("disposition", []) if d["comparison_id"] == "seedrep_cluster/bon/strongreject_bon"]
    assert len(disp) == 1 and disp[0]["object_id"] == ARDITI_OBJ and disp[0]["extra"]["view_legacy"] == TAG and disp[0]["extra"]["single_object"]["trial_index"] == 2
    idw = disp[0]["extra"]["identity_when_graded"]
    assert idw["producer_key"] == ARDITI_TAG and idw["object_key"] == ARDITI_STAGED and idw["run_id"] == I.run_id("sdfbase_full22", ARDITI_TAG, ARDITI_STAGED, "strongreject_bon", SC.BON_PREMINT_INSTRUMENT_ID)
    assert idw["premint_run_id"] == SC.bon_premint_single_object_run_id("e526edf", 2, ARDITI_STAGED, SC.BON_PREMINT_INSTRUMENT_ID) and idw["premint_producer_key"] == "seedrep_battery:e526edf:bon:arditi_s2"


def _bon_roster_entries(seeds) -> list:
    """Synthetic roster entries admitting the best-of-N runs of the `none` key and the all-positions object (the frame reads a run into the paper only through an entry)."""
    base = {"round_id": LC.ROUND_ID, "comparison_id": "fig_synth_bon", "model_id": "qwen35_4b_it", "eval_id": "strongreject_bon", "instrument_id": SC.BON_PREMINT_INSTRUMENT_ID,
            "judge_id": seeds.judge_id(SC.BON_JUDGE_SEED), "strength_rule": "operating_point", "source_citation": "synthetic (test)", "inferred": 0}
    entries = [dict(base, roster_entry_id="re_synth_bon_none", object_id="none"), dict(base, roster_entry_id="re_synth_bon_arditi", object_id=ARDITI_OBJ)]
    return [R.make_record("roster_entry", e, F.producer(), NOW) for e in entries]


def _bon_coexist_ruling(now: str) -> dict:
    """The coexist ruling the module docstring calls for on the single-object trials, in the frame's match form: the three seed views' trials of the
    ONE object on the premint instrument under this source."""
    return {"ruling_id": "r2v3-coexist-seedrep-arditi-allpos-bon-decode-seed-trials", "round_id": LC.ROUND_ID, "action": "coexist", "scope_kind": "key",
            "scope_json": {"match": {"lane_sources_legacy": [SC.SOURCE], "campaigns": ["sdfbase_full22"], "model_ids": ["qwen35_4b_it"], "object_ids": [ARDITI_OBJ],
                                     "eval_ids": ["strongreject_bon"], "instrument_ids": [SC.BON_PREMINT_INSTRUMENT_ID], "views_legacy": ["qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3"]}},
            "reason": "synthetic copy of the single-object coexist ruling: the three seed views' best-of-N runs of the all-positions refusal steer are decode-seed trials of one object",
            "ruled_utc": now, "ruled_utc_precision": "datetime", "ruled_by": "test", "source_ref": "test"}


def test_single_object_trials_of_two_seed_views_share_one_object_and_coexist_under_the_rosters_ruling(tmp_path, seeds):
    """s2 and s3 both draw the ONE all-positions object at their seed base: two run ids (the store names differ), ONE object id, trials 2 and 3, the same
    (model, object, eval, instrument, judge, strength stamps) — the frame's collapse key — and since migration 0021 the trial is part of that key,
    so the frame's `later` rule keeps both selected with no ruling; the roster's `coexist` ruling with the documented match terms stays as the belt and
    reads the same. Ingested end to end into a scratch root carrying the premint instrument's stand-in and the object's row."""
    tag3 = f"{ARDITI_STEM}_seedrep_s3"
    fx, kw, _ = _single_fixture(tmp_path, seeds_tags=(TAG, TAG_S3), hits=30)
    cfg3 = _arditi_config(fx["configs"], tag3)
    _single_bon_store(fx["root"], tag3, 36, config_sha256=F.sha_file(cfg3))
    _arditi_plan(kw["premint_plans_dir"], 3)
    ad = _adapter(fx, seeds, **kw)
    so = SC.parse_single_object_tag(ARDITI_TAG)
    obj = ad.resolve_single_object(_ctx(), so, ARDITI_KEY)
    assert obj.object_id == ARDITI_OBJ and obj.object_kind == "vector_all" and obj.paper_label == "arditi_allpos" and obj.staged_id == ARDITI_STAGED
    root = _registry_root(tmp_path, seeds, [ad.resolve_object(_ctx(), LORA, "s2"), ad.resolve_object(_ctx(), LORA, "s3"), obj])
    fields, prov, _pol, why = ad._bon_instrument("qwen35_4b_it")
    assert why is None
    inst = dict(fields, extra={"seed_instrument_id": SC.BON_CANONICAL_SEED, "stand_in_for": prov["premint_record"]}, notes="test stand-in for the premint's derived instrument",
                field_confidence_json={k: {"confidence": "unverified", "reason": "test stand-in: the premint record carries the wave's own reason"}
                                       for k in ("harness_config_hash", "prompt_template_sha256") if fields.get(k) == "unverified"})
    R.write_records(os.path.join(root, "records", "standins", "000002-bon-instrument.jsonl"), [R.make_record("instrument", inst, F.producer(), NOW)])
    rep0 = H.ingest_once(root, workers=0, min_free=0)
    assert rep0.refused == 0, rep0.refusals
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    assert set(ad._facts) == {TAG, TAG_S3, ARDITI_TAG, tag3} and all(f["n_refusals"] == 0 for f in ad._facts.values()), {g: f["refusals"] for g, f in ad._facts.items()}
    rows = F.q(root, "SELECT view_legacy, producer_run_key_legacy, trial_index, seed_base, lifecycle_state, run_id, object_id, strength_nominal, strength_coefficient, operating_point_id_legacy, "
                     "instrument_id, judge_id FROM run WHERE eval_id = 'strongreject_bon' AND object_id = ? ORDER BY view_legacy", (ARDITI_OBJ,))
    assert [(v, pk, t, sb, st) for v, pk, t, sb, st, *_ in rows] == [(TAG, ARDITI_TAG, 2, 2, "ingested"), (TAG_S3, tag3, 3, 3, "ingested")]
    assert rows[0][5] != rows[1][5] and rows[0][6:] == rows[1][6:] == (ARDITI_OBJ, 1.0, 0.3, ARDITI_STAGED, SC.BON_PREMINT_INSTRUMENT_ID, seeds.judge_id(SC.BON_JUDGE_SEED))
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.object_id = ? AND c.anchor_kind = 'sibling_none'", (ARDITI_OBJ,)) == [(6,)]
    R.write_records(os.path.join(root, "records", "roster", "000001-roster.jsonl"), _bon_roster_entries(seeds))
    rep_r = H.ingest_once(root, workers=0, min_free=0)
    assert rep_r.refused == 0, rep_r.refusals
    sel = lambda: {v: s_ for v, s_ in F.q(root, "SELECT r.view_legacy, p.selection_state FROM v_paper_current p JOIN run r ON r.run_id = p.run_id WHERE r.object_id = ?", (ARDITI_OBJ,))}
    before = sel()
    # the frame's later rule since migration 0021: one collapse key, two trials -- the trial is part of the key, so both read selected with no ruling
    # (before it the key carried no trial and the later twin superseded the earlier)
    assert set(before) == {TAG, TAG_S3} and before[TAG] == before[TAG_S3] == "selected"
    R.write_records(os.path.join(root, "records", "rulings", "000001-coexist.jsonl"), [R.make_record("ruling", _bon_coexist_ruling(NOW), F.producer(), NOW)])
    rep3 = H.ingest_once(root, workers=0, min_free=0)
    assert rep3.refused == 0, rep3.refusals
    after = sel()
    assert after[TAG] == after[TAG_S3] == "selected"
    # the ruling covers the object's trials alone: the seed views' `none` twins beside them are not in its scope (their coexist ruling is the roster's own)
    cover = {v: json.loads(j) for v, j in F.q(root, "SELECT r.view_legacy, p.ruling_ids_json FROM v_paper_current p JOIN run r ON r.run_id = p.run_id WHERE r.object_id = 'none' AND r.eval_id = 'strongreject_bon'")}
    assert set(cover) == {TAG, TAG_S3} and not any("arditi" in rid for ids in cover.values() for rid in ids)
    assert all("r2v3-coexist-seedrep-arditi-allpos-bon-decode-seed-trials" in json.loads(j) for (j,) in F.q(root, "SELECT p.ruling_ids_json FROM v_paper_current p JOIN run r ON r.run_id = p.run_id WHERE r.object_id = ?", (ARDITI_OBJ,)))
    assert F.q(root, "SELECT COUNT(*) FROM ruling WHERE action = 'coexist' AND ruling_id LIKE 'placeholder:%'") == [(0,)]        # the reader files no ruling


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_and_declares_its_source_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[SC.NAME]()
    assert isinstance(inst, SC.SeedrepClusterAdapter) and inst.covers_sources == (SC.SOURCE,) and inst.owner == "ReLU"
    assert AB.emission_version_of(SC.SeedrepClusterAdapter) == "10" and inst.rng_scope == "none" and inst.roots == (SC.ROOT_DEFAULT,)
    assert SC.DECODE_TRIAL_ANCHOR_KIND in V.ANCHOR_KINDS and "SAME object on every seed view" in SC.DECODE_TRIAL_RULE and "FIT REPLICATE only when" in SC.DECODE_TRIAL_RULE
    assert SC.parse_single_object_tag(ARDITI_TAG).seed_view_legacy == TAG and SC.parse_single_object_tag("qwen35_4b_s2") is None and SC.parse_single_object_tag("other_qwen35_4b_ar_c0p3_seedrep_s2") is None
    assert inst.single_roster_path.endswith(os.path.join("configs", "arditi_allpos_roster.yaml")) and inst.premint_plans_dir.endswith(os.path.join("seedrep_battery", "registry", "plans", "bon"))
    assert SC.BON_PREMINT_INSTRUMENT_ID == "7b6761844d0dbcc49159a29c6ba1d0f07f5edea1d59212342830cba5e3bc7dcc" and SC.BON_JUDGE_SEED == "strongreject_v2_haiku45"
    assert set(SC.BON_INSTRUMENT_TREES) == {SC.BON_PREMINT_TREE, BON_TREE} and "grades_bon" in SC.SUBTREES and "transcripts_bon" in SC.SUBTREES
    assert SC.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "seedrep")) and SC.STATIC_LADDER == (4096, 8192, 16384)
    assert set(SC.ESCALATION_STATES) <= set(RP.ESCALATION_STATES)
    assert SC.PENDING_REASON.startswith("generated, ungraded:") and "grader-pin decision with the user, 2026-09-20" in SC.PENDING_REASON
    assert IC.STATIC_INSTRUMENT_SEED == SC.STATIC_INSTRUMENT_SEED == "static_sampled_vendor_preset_4096_8192" and SC.MMLU_INSTRUMENT_SEED == "mmlu_letter_prefill_200"
