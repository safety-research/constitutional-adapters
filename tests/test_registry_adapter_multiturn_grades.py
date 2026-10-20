"""The grades-root adapter on a synthetic root named by a synthetic lane file: two objects under one root are two
runs with one producer key (S1), per-goal items, the grader's judgment record, the proved cap over the grader's
stamp, and the superseding ruling on a greedy-512 root — and the per-goal item pointers: a cell's transcripts
resolve to the grading host copy when it exists, else through `transcripts_mirrors/index.json` + `grading_host_file_sha256.tsv` to
the cluster copies as ONE `store_root` artifact per cell; a declared store that resolves nowhere refuses the cell's
items by name; an undeclared store keeps the items with a null pointer and the reason on the row."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

import _b1_fixtures as F
from sisteer.registry import identity as I
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import multiturn_grades as MG
from sisteer.registry.adapters import transcript_mirrors as TM
from sisteer.registry.adapters.base import AdapterRefusal

STORE = "transcripts_synth_synthview_it"                 # the former grading host dir (relative to the run-state root)
CSTORE = "transcripts_results_synth_synthview_it"        # its cluster store
PRIV = "sdfbase_full22_synthview_it"
BU = "/shared/u/exp/sdfbase_full22"
CH = "/workspace/u/exp/sdfbase_full22"


def _root(tmp_path, root_name="grades_synth_bu_synthview", cap_proved=None, attack="crescendo", store=None, transcripts_run=False):
    rs = tmp_path / "rs"
    vd = rs / root_name / F.VIEW
    os.makedirs(vd, exist_ok=True)
    goals = {f"goal_{i:02d}": {"best_score": 1.0 if i < 6 else 0.25, "capped": i == 3, "n_target_forwards": 3 + i, "queries_to_first_success": (1 if i < 6 else None)}
             for i in range(10)}

    def cell(arm, n_jb):
        return {"arm": arm, "class": "A", "decode_cap_tokens": 512, "si_id": "x", "lens_reduction_mode": "stacked", "served_alpha": 1.0,
                "completion": {"gated": {"attacker_engine": {"attacker_model": "synthetic/attacker", "attacker_engine": "vllm"}, "n_error_outcomes": 0,
                                         "n_goals_with_progress": 10}},
                "attack_metrics": {"asr": {"0.25": n_jb / 10, "0.5": n_jb / 10, "0.75": (n_jb - 1) / 10}, "n_goals": 10, "n_graded": 10, "n_expected": 10,
                                   "n_jailbroken": n_jb, "n_errored": 0, "mean_best_score": 0.6, "n_capped": 1,
                                   "by_category_group": {"held_in": {"n_goals": 6, "n_jailbroken": 4}, "held_out": {"n_goals": 4, "n_jailbroken": 2}},
                                   "per_goal": goals}}
    agg = {"attack": attack, "attack_tier": "multi_turn", "written_at": "2026-01-05T00:00:00+00:00", "target": F.VIEW, "n_expected_goals": {F.LORA: 10, F.RELU: 10},
           "cells": {F.LORA: cell("lora_r4_ce", 6), F.RELU: cell("psr_ce_all", 8)}}
    json.dump(agg, open(vd / f"two_si_{attack}_aggregate.json", "w"))
    json.dump({"judge": {"judge_model": "judge-model-x", "rubric": "strongreject_v1", "cache_kind": "sr", "pass_id": "p" * 64, "org_initial": "lowprio"},
               "org_final": "lowprio", "finished_at": "2026-01-05T01:00:00+00:00", "started_at": "2026-01-05T00:30:00+00:00",
               "worktree": {"head": "b" * 40}, "config": {"path": "/x/config.yaml", "sha256": "c" * 64}, "n_429": 0}, open(vd / "grading_provenance.json", "w"))
    json.dump({"cells": {F.LORA: {"gated": {attack: {"n_goals_with_progress": 10}}}}, "written_at": "2026-01-05T00:59:00+00:00"}, open(vd / "completion.json", "w"))
    lane = {"meta": {"namespaces": {"synth": {"dir": root_name}}}, "namespaces": {}, "models": {}}
    if cap_proved:
        root_entry = {"root": str(rs / root_name), "view": F.VIEW}
        if store and transcripts_run:
            root_entry["transcripts_run"] = str(rs / store / PRIV)           # the llama lane's spelling
        elif store:
            root_entry.update({"transcripts_root": str(rs / store), "private_run": PRIV})
        lane = {"meta": {"roots": {root_name: root_entry}, "served_cluster_by_root": {root_name: "cluster_b"}},
                "models": {F.VIEW: {"cells": {F.LORA: {attack: {"root": str(rs / root_name), "decode_cap_tokens": cap_proved, "n": 10}},
                                              F.RELU: {attack: {"root": str(rs / root_name), "decode_cap_tokens": cap_proved, "n": 10}}}}}}
    json.dump(lane, open(rs / "transfer_stats_full22_all.json", "w"))
    return str(rs)


def _entry(name, grading_host, bu=True, ch=True, origin="cluster_a", bu_root=BU):
    return {"cluster_a_path": f"{bu_root}/{name}" if bu else None, "cluster_b_path": f"{CH}/{name}" if ch else None, "origin": origin,
            "mirror": ("cluster_b" if origin == "cluster_a" else "cluster_a") if (bu and ch) else ("cluster_a" if bu and origin != "cluster_a" else None),
            "sidecar": f"{name}.mirror.json", "files": 4, "bytes": 100, "content_manifest_sha256": "e" * 64, "former_grading_host_dir": grading_host}


def _cell_files(cells, attack="crescendo", shards=("shard00", "shard01"), names=("progress.jsonl", "responses.jsonl")):
    """(relpath, sha) for every file of the given cells in the synthetic store layout."""
    out = []
    for c in cells:
        for sh in shards:
            for n in names:
                rel = f"{PRIV}/{attack}/{c}/{sh}/{n}"
                out.append((rel, hashlib.sha256(rel.encode()).hexdigest()))
    return out


def _mirror_index(rs, stores, files, meta_extra=None):
    d = os.path.join(rs, "transcripts_mirrors")
    os.makedirs(d, exist_ok=True)
    json.dump({"_meta": dict({"built_at": "2026-09-17T05:23:53Z"}, **(meta_extra or {})), "stores": stores}, open(os.path.join(d, "index.json"), "w"))
    with open(os.path.join(d, "grading_host_file_sha256.tsv"), "w") as fh:
        for dev, rel, sha in files:
            fh.write(f"{dev}\t{rel}\t{sha}\n")
    return d


def _manifest_id(files):
    lines = sorted(f"{k.split(':', 1)[0]}\t{k.split(':', 1)[1]}\t{v}" for k, v in files.items())
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _adapter(rs, tmp_path):
    seeds = F.synthetic_seeds()
    ad = MG.MultiturnGradesAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), transfer_stats_files=("transfer_stats_full22_all.json",))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA), ad.objects.resolve(F.VIEW, F.RELU)])
    return seeds, ad, root


def test_two_objects_one_root_two_runs_with_items(tmp_path):
    rs = _root(tmp_path)
    seeds, ad, root = _adapter(rs, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and {r.group for r in rep.results} == {"grades_synth_bu_synthview", "_lane_files"}, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert len(ad.refusals) == 0, ad.refusals.items
    runs = F.q(root, "SELECT run_id, producer_run_key_legacy, producer_object_key_legacy, token_cap_tokens, status_ruling_id, attacker_model_id, "
                     "lifecycle_state, substrate FROM run ORDER BY producer_object_key_legacy")
    assert len(runs) == 2 and {r[1] for r in runs} == {"grades_synth_bu_synthview"} and {r[2] for r in runs} == {F.LORA, F.RELU}
    assert all(r[3] == 512 and r[4] == LC.supersede_instrument_ruling_id("multiturn_greedy_512_full22") and r[5] == "synthattacker" for r in runs)
    assert all(r[6] == "ingested" and r[7] == "cluster_a" for r in runs)   # completion artifact + complete judgment → ingested; _bu_ names the cluster
    j = F.q(root, "SELECT api_pool, grading_code_sha, n_items_judged, judge_cache_kind FROM judgment")
    assert len(j) == 2 and all(r == ("lowprio", "b" * 40, 10, "sr") for r in j)
    # metrics: asr at three thresholds + held-in / held-out per run; k at 0.5 is n_jailbroken
    m = F.q(root, "SELECT rm.metric_id, rm.threshold, rm.k, rm.n FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.producer_object_key_legacy = ? ORDER BY 1, 2", (F.RELU,))
    assert ("asr", 0.5, 8, 10) in {tuple(r) for r in m} and ("asr_held_in", 0.5, 4, 6) in {tuple(r) for r in m}
    # per-goal items: best_score for every goal, queries_to_first_success for the successes, one flags row per goal
    it = F.q(root, "SELECT metric_id, COUNT(*) FROM item_result GROUP BY 1")
    assert dict(it) == {"best_score": 20, "queries_to_first_success": 12}
    fl = F.q(root, "SELECT SUM(at_token_cap), COUNT(*) FROM item_flags")
    assert fl == [(2, 20)]
    # the supersede ruling itself was written (a placeholder)
    assert F.q(root, "SELECT action, scope_kind FROM ruling") == [("supersede", "key")]
    # a namespace root declares no transcript store — the items keep a NULL pointer and the row says why (no refusal)
    flags = F.q(root, "SELECT generation_artifact_id, extra_json FROM item_flags")
    assert all(r[0] is None and "names no transcript store" in json.loads(r[1])["generation_pointer_reason"] for r in flags)
    for (extra,) in F.q(root, "SELECT extra_json FROM run"):
        e = json.loads(extra)
        assert e["transcript_store"] is None and e["transcript_store_uri"] is None and "names no transcript store" in e["transcript_pointer_reason"]
    assert F.q(root, "SELECT COUNT(*) FROM artifact WHERE kind = 'store_root'") == [(0,)]


def test_proved_cap_from_the_lane_file_wins_over_the_grader_stamp(tmp_path):
    rs = _root(tmp_path, root_name="grades_mt4k_ch_synthview_it", cap_proved=4096)
    seeds, ad, root = _adapter(rs, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals)
    runs = F.q(root, "SELECT token_cap_tokens, status_ruling_id, substrate, instrument_id FROM run")
    inst = seeds.instrument_row("multiturn_crescendo_cap4096_generation_config", "crescendo")["instrument_id"]
    assert all(r[0] == 4096 and r[1] is None and r[2] == "cluster_b" and r[3] == inst for r in runs)
    # the grader's config-derived stamp is kept beside it
    assert all(json.loads(r[0])["grader_stamp_cap"] == 512 for r in F.q(root, "SELECT extra_json FROM run"))


# --------------------------------------------------------------------------------------------
# Item pointers after the grading host transcript stores were mirrored and deleted
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("transcripts_run", [False, True])
def test_d47_declared_store_relocates_the_items_through_the_mirror_index(tmp_path, transcripts_run):
    rs = _root(tmp_path, root_name="grades_mt4k_ch_synthview_it", cap_proved=4096, store=STORE, transcripts_run=transcripts_run)
    files = _cell_files([F.LORA, F.RELU])
    d = _mirror_index(rs, {CSTORE: _entry(CSTORE, STORE)}, [(STORE, rel, sha) for rel, sha in files] + [(STORE, "pull_last.txt", "0" * 64)])
    seeds, ad, root = _adapter(rs, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals + ing2.refusals)
    assert len(ad.refusals) == 0, ad.refusals.items
    assert {r.group for r in rep.results} == {"grades_mt4k_ch_synthview_it", "_lane_files"}      # the index rides the lane-files group
    # the pointer material is registered as inputs
    reg = sorted(r[0] for r in F.q(root, "SELECT uri_primary FROM artifact WHERE uri_primary LIKE ?", (f"{d}%",)))
    assert reg == [os.path.join(d, "grading_host_file_sha256.tsv"), os.path.join(d, "index.json")]
    # every item keeps flowing and names its cell's artifact
    assert dict(F.q(root, "SELECT metric_id, COUNT(*) FROM item_result GROUP BY 1")) == {"best_score": 20, "queries_to_first_success": 12}
    fl = F.q(root, "SELECT r.producer_object_key_legacy, f.generation_artifact_id, f.generation_offset, f.extra_json FROM item_flags f JOIN run r ON r.run_id = f.run_id")
    assert len(fl) == 20 and all(r[1] and r[2] is None and json.loads(r[3])["generation_pointer"] == "cell_store_dir" for r in fl)
    by_cell = {}
    for cell, aid, _, _ in fl:
        by_cell.setdefault(cell, set()).add(aid)
    assert {len(v) for v in by_cell.values()} == {1} and len(set().union(*by_cell.values())) == 2
    # ONE store_root artifact per cell: both clusters, the cluster relpaths + per-file shas, the manifest-derived id
    arts = {r[0]: r for r in F.q(root, "SELECT artifact_id, uri_primary, locations_json, extra_json, verified_utc FROM artifact WHERE kind = 'store_root'")}
    assert len(arts) == 2
    for cell in (F.LORA, F.RELU):
        aid = next(iter(by_cell[cell]))
        _, uri, locs, extra, verified = arts[aid]
        assert uri == f"cluster_a:{BU}/{CSTORE}/{PRIV}/crescendo/{cell}"
        locs = json.loads(locs)
        assert [(l["cluster"], l["role"]) for l in locs] == [("cluster_a", "origin"), ("cluster_b", "mirror")]
        assert locs[1]["uri"] == f"cluster_b:{CH}/{CSTORE}/{PRIV}/crescendo/{cell}" and locs[0]["verified_utc"] == "2026-09-17T05:23:53.000Z" == verified
        e = json.loads(extra)
        want = {f"{CSTORE}:{rel}": sha for rel, sha in files if f"/{cell}/" in rel}
        assert e["files"] == want and e["n_files"] == 4 and e["stores"] == [CSTORE] and e["former_grading_host_dir"] == STORE
        assert aid == _manifest_id(want) and e["artifact_id_rule"].startswith("sha256 over the sorted")
        assert "grading_host_file_sha256.tsv" in e["sha256_source"] and "per_file_sha256" not in e and "stale" not in e
        assert e["content_manifest_sha256"] == {CSTORE: "e" * 64}
    # the run names the cell's cluster directory beside the grades root it was read from
    for cell, extra in F.q(root, "SELECT producer_object_key_legacy, extra_json FROM run"):
        e = json.loads(extra)
        assert e["transcript_store"] == STORE and e["transcript_store_uri"] == f"cluster_a:{BU}/{CSTORE}/{PRIV}/crescendo/{cell}"
        assert "transcript_pointer_reason" not in e and "items_refused" not in e
    # the view joins the pointer
    v = F.q(root, "SELECT COUNT(*) FROM item_flags f JOIN artifact a ON a.artifact_id = f.generation_artifact_id")
    assert v == [(20,)]


def test_d47_declared_store_with_no_index_entry_refuses_the_cells_items_by_name(tmp_path):
    rs = _root(tmp_path, root_name="grades_mt4k_ch_synthview_it", cap_proved=4096, store=STORE)
    _mirror_index(rs, {"transcripts_results_other": _entry("transcripts_results_other", "transcripts_other")}, [])
    seeds, ad, root = _adapter(rs, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals + ing2.refusals)
    # the runs, judgments and metrics come from the grades root, which exists; the per-goal rows do not
    assert F.q(root, "SELECT COUNT(*) FROM run") == [(2,)] and F.q(root, "SELECT COUNT(*) FROM run_metric") != [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM item_result") == [(0,)] and F.q(root, "SELECT COUNT(*) FROM item_flags") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM artifact WHERE kind = 'store_root'") == [(0,)]
    assert len(ad.refusals) == 2 and ad.stats["items_refused"] == 20
    for it in ad.refusals.items:
        assert it["key"].endswith("|items") and STORE in it["why"] and "no entry in transcripts_mirrors/index.json" in it["why"] and "no grading host copy" in it["why"]
    # durable: the refusal is a source_state row, and the run says its items were refused and why
    ss = F.q(root, "SELECT current_members_json FROM source_state WHERE json_path = '$.row_refusals'")
    assert len(ss) == 1 and all(STORE in m for m in json.loads(ss[0][0])) and len(json.loads(ss[0][0])) == 2
    for (extra,) in F.q(root, "SELECT extra_json FROM run"):
        e = json.loads(extra)
        assert e["items_refused"] is True and e["transcript_store"] == STORE and e["transcript_store_uri"] is None and STORE in e["transcript_pointer_reason"]


def test_d47_declared_store_without_any_mirror_index_refuses_too(tmp_path):
    rs = _root(tmp_path, root_name="grades_mt4k_ch_synthview_it", cap_proved=4096, store=STORE)
    seeds, ad, root = _adapter(rs, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0
    assert F.q(root, "SELECT COUNT(*) FROM item_flags") == [(0,)] and F.q(root, "SELECT COUNT(*) FROM run") == [(2,)]
    assert len(ad.refusals) == 2 and all("no transcripts_mirrors/index.json" in it["why"] for it in ad.refusals.items)


def test_d47_indexed_store_whose_reclaim_table_lacks_the_cell_refuses_only_that_cell(tmp_path):
    rs = _root(tmp_path, root_name="grades_mt4k_ch_synthview_it", cap_proved=4096, store=STORE)
    files = _cell_files([F.LORA])                                                # the RELU cell was never on the grading host
    _mirror_index(rs, {CSTORE: _entry(CSTORE, STORE)}, [(STORE, rel, sha) for rel, sha in files])
    seeds, ad, root = _adapter(rs, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0
    got = F.q(root, "SELECT r.producer_object_key_legacy, COUNT(*) FROM item_flags f JOIN run r ON r.run_id = f.run_id GROUP BY 1")
    assert got == [(F.LORA, 10)]
    assert len(ad.refusals) == 1 and F.RELU in ad.refusals.items[0]["key"] and "lists no file under" in ad.refusals.items[0]["why"]
    assert ad.stats["items_refused"] == 10 and F.q(root, "SELECT COUNT(*) FROM artifact WHERE kind = 'store_root'") == [(1,)]


def test_d47_grading_host_copy_present_points_at_it_with_hashed_files(tmp_path):
    rs = _root(tmp_path, root_name="grades_mt4k_ch_synthview_it", cap_proved=4096, store=STORE)
    want = {}
    for cell in (F.LORA, F.RELU):
        for sh in ("shard00", "shard01"):
            d = os.path.join(rs, STORE, PRIV, "crescendo", cell, sh)
            os.makedirs(d)
            for n in ("progress.jsonl", "responses.jsonl"):
                body = f'{{"goal": "{cell}/{sh}/{n}"}}\n'.encode()
                open(os.path.join(d, n), "wb").write(body)
                want[(cell, f"{STORE}:{PRIV}/crescendo/{cell}/{sh}/{n}")] = hashlib.sha256(body).hexdigest()
    seeds, ad, root = _adapter(rs, tmp_path)                                    # no transcripts_mirrors/ at all
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0 and len(ad.refusals) == 0, ad.refusals.items
    assert F.q(root, "SELECT COUNT(*) FROM item_flags WHERE generation_artifact_id IS NOT NULL") == [(20,)]
    arts = F.q(root, "SELECT uri_primary, locations_json, extra_json, verified_utc FROM artifact WHERE kind = 'store_root' ORDER BY uri_primary")
    assert len(arts) == 2
    for uri, locs, extra, verified in arts:
        cell = os.path.basename(uri)
        assert uri == os.path.join(rs, STORE, PRIV, "crescendo", cell) and verified is None
        assert json.loads(locs) == [{"uri": uri, "cluster": "grading_host", "store": STORE, "role": "grading_host copy", "verified_utc": None, "bytes": None}]
        e = json.loads(extra)
        assert e["files"] == {k: v for (c, k), v in want.items() if c == cell} and "hashed when the adapter read" in e["sha256_source"]


def test_d47_static_items_are_outside_the_step(tmp_path):
    rs = _root(tmp_path, root_name="grades_mt4k_ch_synthview_it", cap_proved=4096, attack="pasttense", store=STORE)
    seeds, ad, root = _adapter(rs, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0 and len(ad.refusals) == 0, ([r.error for r in rep.results], ad.refusals.items)
    fl = F.q(root, "SELECT generation_artifact_id, extra_json FROM item_flags")
    assert len(fl) == 20 and all(r[0] is None and json.loads(r[1])["generation_pointer_reason"].startswith("not a multi-turn item") for r in fl)


# ---- the mirror reader on its own ---------------------------------------------------------------


def test_d47_relu_rounds_split_by_shard_name(tmp_path):
    rs = str(tmp_path / "rs"); os.makedirs(rs)
    dev = "relu_native/transcripts_relu_native_bu_synthview"
    run, cell = "sdfbase_full22_synthview", F.RELU
    stores = {f"transcripts_results_relu_native{s}": _entry(f"transcripts_results_relu_native{s}", "relu_native/transcripts_relu_native_bu_*", bu_root="/shared/u/exp/sdfbase_full22_native")
              for s in ("", "_r2", "_r3")}
    rows = [(dev, f"{run}/crescendo/{cell}/shard00/progress.jsonl", "1" * 64), (dev, f"{run}/crescendo/{cell}/shard_r2_01/progress.jsonl", "2" * 64),
            (dev, f"{run}/crescendo/{cell}/shard_r3_00/progress.jsonl", "3" * 64), (dev, f"{run}/pair/{cell}/shard00/progress.jsonl", "4" * 64)]
    _mirror_index(rs, stores, rows)
    ptr, why = TM.TranscriptMirrors(rs).locate_run_dir(dev, f"{run}/crescendo/{cell}")
    assert why is None and ptr.stores == ["transcripts_results_relu_native", "transcripts_results_relu_native_r2", "transcripts_results_relu_native_r3"]
    assert ptr.files == {f"transcripts_results_relu_native:{run}/crescendo/{cell}/shard00/progress.jsonl": "1" * 64,
                         f"transcripts_results_relu_native_r2:{run}/crescendo/{cell}/shard01/progress.jsonl": "2" * 64,
                         f"transcripts_results_relu_native_r3:{run}/crescendo/{cell}/shard00/progress.jsonl": "3" * 64}
    assert ptr.uri_primary == f"cluster_a:/shared/u/exp/sdfbase_full22_native/transcripts_results_relu_native/{run}/crescendo/{cell}"
    assert [(l["store"], l["cluster"]) for l in ptr.locations] == [(s, c) for s in ptr.stores for c in ("cluster_a", "cluster_b")]
    assert ptr.artifact_id == _manifest_id(ptr.files)
    # a renamed shard whose round is not indexed is a refusal, not a guess
    del stores["transcripts_results_relu_native_r3"]
    _mirror_index(rs, stores, rows)
    ptr, why = TM.TranscriptMirrors(rs).locate_run_dir(dev, f"{run}/crescendo/{cell}")
    assert ptr is None and "round 3" in why


def test_d47_pushed_pull_without_reclaim_rows_is_uri_keyed_and_says_so(tmp_path):
    rs = str(tmp_path / "rs"); os.makedirs(rs)
    dev, name = "transcripts_halfdose", "transcripts_halfdose"
    e = _entry(name, dev, ch=False, origin="grading_host pull (pushed to cluster A)", bu_root="/shared/u/exp/grading_host_pulls/candidate-si-refit/full22")
    e["files"] = 9666
    _mirror_index(rs, {name: e}, [("transcripts_other", "x/y/z/shard00/progress.jsonl", "9" * 64)])
    ptr, why = TM.TranscriptMirrors(rs).locate_run_dir(dev, "sdfbase_full22_synthview_halfdose/pair/none")
    assert why is None and ptr.per_file_sha is False and ptr.files == {}
    assert ptr.uri_primary == "cluster_a:/shared/u/exp/grading_host_pulls/candidate-si-refit/full22/transcripts_halfdose/sdfbase_full22_synthview_halfdose/pair/none"
    assert ptr.artifact_id == hashlib.sha256(ptr.uri_primary.encode()).hexdigest()
    assert [(l["cluster"], l["verified_utc"]) for l in ptr.locations] == [("cluster_a", None)]
    pay = ptr.artifact_payload()
    assert pay["kind"] == "store_root" and pay["verified_utc"] is None and pay["extra"]["per_file_sha256"]["confidence"] == "unverified"
    assert "predates this store's push" in pay["extra"]["per_file_sha256"]["reason"]
    assert pay["extra"]["artifact_id_rule"].startswith("sha256 of uri_primary") and "sha256_source" not in pay["extra"]


def test_d47_stale_note_marks_the_cell_and_an_unparseable_note_refuses(tmp_path):
    rs = str(tmp_path / "rs"); os.makedirs(rs)
    dev, name, run = "transcripts_mt4karr3_ch_synth", "transcripts_results_mt4karr3_ch_synth", "sdfbase_full22_synth"
    rows = [(dev, f"{run}/actorattack/arditi/shard00/progress.jsonl", "a" * 64), (dev, f"{run}/pair/arditi/shard00/progress.jsonl", "b" * 64)]
    note = f"{name} actorattack/arditi: the grading host pull predated the final writes; cluster copies are the complete ones"
    _mirror_index(rs, {name: _entry(name, dev, origin="cluster_b")}, rows, {"stale_note": note})
    tm = TM.TranscriptMirrors(rs)
    stale, why = tm.locate_run_dir(dev, f"{run}/actorattack/arditi")
    assert why is None and stale.stale == note and stale.artifact_id == hashlib.sha256(stale.uri_primary.encode()).hexdigest()
    assert stale.artifact_payload()["extra"]["stale"]["confidence"] == "unverified"
    fresh, why = tm.locate_run_dir(dev, f"{run}/pair/arditi")
    assert why is None and fresh.stale is None and fresh.artifact_id == _manifest_id(fresh.files)
    assert [(l["cluster"], l["role"]) for l in fresh.locations] == [("cluster_a", "mirror"), ("cluster_b", "origin")]
    _mirror_index(rs, {name: _entry(name, dev)}, rows, {"stale_note": "some prose that names nothing this reader can attribute"})
    with pytest.raises(AdapterRefusal) as exc:
        TM.TranscriptMirrors(rs).locate_run_dir(dev, f"{run}/pair/arditi")
    assert exc.value.path.endswith("index.json") and "stale_note" in exc.value.why


def test_d47_missing_reclaim_table_beside_the_index_is_a_refusal(tmp_path):
    rs = str(tmp_path / "rs"); os.makedirs(rs)
    _mirror_index(rs, {CSTORE: _entry(CSTORE, STORE)}, [])
    os.remove(os.path.join(rs, "transcripts_mirrors", "grading_host_file_sha256.tsv"))
    with pytest.raises(AdapterRefusal) as exc:
        TM.TranscriptMirrors(rs).inputs()
    assert exc.value.path.endswith("grading_host_file_sha256.tsv")


def test_fitted_runs_carry_the_operating_point_beside_the_filed_nominal(tmp_path):
    """The graded roots' fitted runs filed the export's nominal (1.0 / 0.5) but never `is_operating_point`, so a roster entry's
    `operating_point` rule could not admit them (2026-09-20). A fitted object at the filed nominal 1.0 is at the operating point,
    at 0.5 (a `_halfdose` root) it is not; the unit is the campaign nominal (`nominal_only`)."""
    rs = _root(tmp_path / "a")
    seeds, ad, root = _adapter(rs, tmp_path / "a")
    F.run_and_ingest(root, ad)
    rows = F.q(root, "SELECT campaign_cell_id_legacy, strength_nominal, strength_nominal_source, strength_unit, is_operating_point, "
                     "json_extract(field_confidence_json, '$.is_operating_point.confidence') FROM run ORDER BY 1")
    assert rows == [(F.LORA, 1.0, "filed", "nominal_only", 1, "derived"), (F.RELU, 1.0, "filed", "nominal_only", 1, "derived")]
    # the same root under a `_halfdose` spelling: the filed nominal is 0.5 and the runs are not at the operating point
    rs_half = _root(tmp_path / "b")
    os.rename(os.path.join(rs_half, "grades_synth_bu_synthview", F.VIEW), os.path.join(rs_half, "grades_synth_bu_synthview", F.VIEW + "_halfdose"))
    seeds, ad, root = _adapter(rs_half, tmp_path / "b")
    F.run_and_ingest(root, ad)
    rows = F.q(root, "SELECT view_legacy, strength_nominal, strength_unit, is_operating_point FROM run ORDER BY campaign_cell_id_legacy")
    assert rows == [(F.VIEW + "_halfdose", 0.5, "nominal_only", 0)] * 2


# --------------------------------------------------------------------------------------------
# declared roots (EMISSION_VERSION 4): the registry's own declaration + the attacker-identity gate under a ruling
# --------------------------------------------------------------------------------------------

RULING_ID = "r2v4-accept-independent-attacker-attestation-synth"
DECLARED_ROOT = "grades_mt4kscr2_ch_synthview_it"
ATTACKER_HF = "synthetic/attacker"
REFUSED_ERROR = ("EngineIdentityError: attacker engine: /x/shard00/attacker_engine.json records attacker_quantization='unrecorded' -- "
                 "the weight precision that produced these transcripts is unverifiable; refusing to aggregate an unknown identity even "
                 "though every shard shares it")


def _identity_check(attack, harness="REFUSED", independent="UNIFORM", quantization="none"):
    """A store's `attacker_identity_check.<attack>` block the way the grader writes it (the live run-2 stores' shape)."""
    hc = {"status": harness, "error": REFUSED_ERROR} if harness == "REFUSED" else {"status": harness}
    return {attack: {"harness_check": hc,
                     "independent_check": {"status": independent, "n_shards": 16, "n_cells": 1, "distinct_recorded_tuples": 1,
                                           "attested_elsewhere": {"config.attacker.quantization": quantization, "config.attacker.prefix_caching": True,
                                                                  "source": "configs/<target>.yaml attacker block + WAVE_MANIFEST.json attacker_pool"},
                                           "fields_unrecorded_in_sidecar": ["attacker_kv_cache_dtype", "attacker_prefix_caching", "attacker_quantization"],
                                           "recorded_identity": {"attacker_engine": "vllm", "attacker_model": ATTACKER_HF, "attacker_quantization": "unrecorded"}}}}


def _declared_root(tmp_path, attack="crescendo", harness="REFUSED", independent="UNIFORM", store_quantization="none", lane_names_it=False):
    """A synthetic graded root NO lane file names (the lane file is empty), whose provenance carries the grader's attacker-identity check."""
    rs = _root(tmp_path, root_name=DECLARED_ROOT, attack=attack, cap_proved=None)
    vd = os.path.join(rs, DECLARED_ROOT, F.VIEW)
    prov = json.load(open(os.path.join(vd, "grading_provenance.json")))
    prov["attacker_identity_check"] = _identity_check(attack, harness, independent, store_quantization)
    json.dump(prov, open(os.path.join(vd, "grading_provenance.json"), "w"))
    if not lane_names_it:
        json.dump({"meta": {}, "models": {}}, open(os.path.join(rs, "transfer_stats_full22_all.json"), "w"))
    return rs


def _declaration(tmp_path, rs, attack="crescendo", ruling_id=RULING_ID, kind="independent", quantization="none", object_keys=(F.LORA, F.RELU),
                 cap=4096, store=None):
    import yaml
    d = {"root": DECLARED_ROOT, "view_legacy": F.VIEW, "attacks": [attack], "object_keys": list(object_keys), "decode_cap_tokens": cap,
         "served_cluster": "cluster_b", "private_run": PRIV, "attestation": {"kind": kind, "attacker_quantization": quantization,
                                                                               "attacker_prefix_caching": True, "harness_check": "REFUSED: attacker_quantization unrecorded",
                                                                               "ruling_id": ruling_id,
                                                                               "evidence": [{"kind": "config_attacker_block", "path": "configs/synth.yaml", "sha256": "d" * 64,
                                                                                             "key": "attacker", "value": {"quantization": quantization}}]}}
    if store:
        d["transcripts_root"] = store
    p = tmp_path / "multiturn_declared_roots.yaml"
    yaml.safe_dump({"schema": MG.DECLARED_ROOTS_SCHEMA, "roots": [d]}, open(p, "w"))
    return str(p)


def _file_ruling(root, ruling_id=RULING_ID, gz=False):
    """A `reference` ruling on the registry root's RECORD LOG (`rulings/added/`), the way `registry ruling add` writes it."""
    from sisteer.registry import records as R
    from sisteer.registry.identity import adapter_record_id
    payload = {"ruling_id": ruling_id, "round_id": "r2", "action": "reference", "scope_kind": "key",
               "scope_json": {"match": {"eval_ids": ["crescendo"], "lane_sources_legacy": ["multiturn_grades"]}},
               "target_json": {"accepts": "the independent attacker attestation over the refused harness check"}, "reason": "synthetic",
               "ruled_utc": "2026-09-24T23:00:00.000Z", "ruled_utc_precision": "datetime", "ruled_by": "the user (relayed)", "source_ref": "synthetic",
               "expires_when_json": None, "extra": {"provenance": {"kind": "relayed", "relayed_by": "peer", "message_utc": "2026-09-24T23:00:00Z"}}}
    rec = R.make_record("ruling", payload, F.producer(), adapter_record_id(payload, "ruling"), "2026-09-24T23:01:00.000Z")
    path = os.path.join(root, "rulings", "added", f"2026-09-24-ruling-{ruling_id}.jsonl" + (".gz" if gz else ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    R.write_records(path, [rec], gz=gz)
    return path


def _declared_adapter(rs, tmp_path, decl_path):
    seeds = F.synthetic_seeds()
    ad = MG.MultiturnGradesAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), transfer_stats_files=("transfer_stats_full22_all.json",),
                                   declared_roots_file=decl_path)
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA), ad.objects.resolve(F.VIEW, F.RELU)])
    return seeds, ad, root


def test_lane_file_roots_never_read_the_declaration_and_carry_no_attestation(tmp_path):
    """The default adapter reads this checkout's declaration; a lane-file root's rows are untouched by it (byte identity of the filed
    roots: no `declared_root`, no `attacker_attestation`, the lane file's spelling on `lane_file`)."""
    rs = _root(tmp_path, root_name="grades_mt4k_ch_synthview_it", cap_proved=4096)
    seeds, ad, root = _adapter(rs, tmp_path)
    assert ad.declared_roots_file == MG.DECLARED_ROOTS_FILE and os.path.isfile(MG.DECLARED_ROOTS_FILE)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0
    for (extra,) in F.q(root, "SELECT extra_json FROM run"):
        e = json.loads(extra)
        assert "declared_root" not in e and "attacker_attestation" not in e and e["lane_file"] == "transfer_stats_full22_all.json"
    # the checkout's declaration is a sha'd input of the `_lane_files` group (the live roots it names are not under this synthetic rs)
    assert F.q(root, "SELECT COUNT(*) FROM artifact WHERE uri_primary = ?", (MG.DECLARED_ROOTS_FILE,)) == [(1,)]


def test_declared_root_with_refused_harness_check_files_under_the_ruling_with_the_attestation_on_every_row(tmp_path):
    rs = _declared_root(tmp_path)
    decl = _declaration(tmp_path, rs, store=STORE)
    files = _cell_files([F.LORA, F.RELU])
    _mirror_index(rs, {CSTORE: _entry(CSTORE, STORE, origin="cluster_b")}, [(STORE, rel, sha) for rel, sha in files])
    seeds, ad, root = _declared_adapter(rs, tmp_path, decl)
    log_path = _file_ruling(root)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals + ing2.refusals)
    assert len(ad.refusals) == 0, ad.refusals.items
    assert {r.group for r in rep.results} == {DECLARED_ROOT, "_lane_files"}
    runs = F.q(root, "SELECT producer_run_key_legacy, campaign_cell_id_legacy, token_cap_tokens, substrate, attacker_model_id, instrument_id, status_ruling_id, "
                     "extra_json, field_confidence_json FROM run ORDER BY campaign_cell_id_legacy")
    inst = seeds.instrument_row("multiturn_crescendo_cap4096_generation_config", "crescendo")["instrument_id"]
    assert [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in runs] == [(DECLARED_ROOT, F.LORA, 4096, "cluster_b", "synthattacker", inst, None),
                                                                          (DECLARED_ROOT, F.RELU, 4096, "cluster_b", "synthattacker", inst, None)]
    for *_, extra, fc in runs:
        e, c = json.loads(extra), json.loads(fc)
        assert e["declared_root"] is True and e["lane_file"] == "declared:multiturn_declared_roots.yaml" and e["grader_stamp_cap"] == 512
        assert e["declaration_file"] == LC.rel_home(decl) and e["declaration_sha256"] == F.sha_file(decl)
        a = e["attacker_attestation"]
        assert a["source"] == "independent" and a["ruling_id"] == RULING_ID and a["attacker_quantization"] == "none"
        assert a["harness_check"] == "REFUSED: attacker_quantization unrecorded" and a["harness_error"] == REFUSED_ERROR
        assert a["independent_check"] == "UNIFORM" and a["n_shards"] == 16 and a["ruling_on_record_log"] == LC.rel_home(log_path)
        assert a["evidence"][0]["kind"] == "config_attacker_block" and a["evidence"][0]["sha256"] == "d" * 64
        assert c["attacker_quantization"] == {"confidence": "derived", "source": c["attacker_quantization"]["source"], "attestation": "attested_independently",
                                              "ruling_id": RULING_ID, "value": "none"}
        assert c["attacker_quantization"]["source"].startswith("attested_independently (ruling " + RULING_ID)
        assert c["attacker_prefix_caching"]["attestation"] == "attested_independently" and c["attacker_prefix_caching"]["value"] is True
        assert c["attacker_kv_cache_dtype"]["confidence"] == "unverified"        # attested by nothing: said so, never filled
    # the store's cluster copy is the items' pointer (declared transcripts_root + private_run), the metrics and items flow as for a lane root
    assert dict(F.q(root, "SELECT metric_id, COUNT(*) FROM item_result GROUP BY 1")) == {"best_score": 20, "queries_to_first_success": 12}
    assert F.q(root, "SELECT COUNT(*) FROM artifact WHERE kind = 'store_root'") == [(2,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric WHERE metric_id = 'asr' AND threshold = 0.5") == [(2,)]
    # the declaration is a registered input of the `_lane_files` group
    assert F.q(root, "SELECT COUNT(*) FROM artifact WHERE uri_primary = ?", (decl,)) == [(1,)]


def test_declared_root_without_the_ruling_on_the_record_log_is_refused_by_name(tmp_path):
    rs = _declared_root(tmp_path)
    decl = _declaration(tmp_path, rs)
    seeds, ad, root = _declared_adapter(rs, tmp_path, decl)         # no ruling filed
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0
    assert F.q(root, "SELECT COUNT(*) FROM run") == [(0,)] and F.q(root, "SELECT COUNT(*) FROM run_metric") == [(0,)]
    assert len(ad.refusals) == 2 and all(RULING_ID in it["why"] and "not on the record log" in it["why"] for it in ad.refusals.items), ad.refusals.items
    ss = F.q(root, "SELECT current_members_json FROM source_state WHERE json_path = '$.row_refusals'")
    assert len(ss) == 1 and len(json.loads(ss[0][0])) == 2
    # the ruling filed afterwards (gzipped, as the log keeps older ones) lets the read file the rows. The inputs are unchanged, so the
    # incremental pass would treat the row-refused revision as committed (a row refusal never forces a re-read); the operator's path after
    # a ruling lands on the log is a forced reread of the source, which publishes the fresh read as a new revision
    from sisteer.registry import db as DB, harvest as H, records as R
    from sisteer.registry.adapters import runner as RUN
    _file_ruling(root, gz=True)
    ad2 = MG.MultiturnGradesAdapter(rs=rs, seeds=seeds, objects=ad.objects, transfer_stats_files=("transfer_stats_full22_all.json",), declared_roots_file=decl)
    F.run_and_ingest(root, ad2)
    assert len(ad2.refusals) == 0 and F.q(root, "SELECT COUNT(*) FROM run") == [(0,)]       # same revision id: not re-published incrementally
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        RUN.run_pass(root, [ad2], conn=conn, now=R.utc_now(), producer=F.producer(), reread=True, force=True)
    finally:
        conn.close()
    H.ingest_once(root, workers=0, min_free=0)
    H.ingest_once(root, workers=0, min_free=0)
    assert len(ad2.refusals) == 0 and F.q(root, "SELECT COUNT(*) FROM run") == [(2,)]
    assert all(json.loads(e)["attacker_attestation"]["ruling_id"] == RULING_ID for (e,) in F.q(root, "SELECT extra_json FROM run"))


@pytest.mark.parametrize("case", ["kind", "quantization", "independent"])
def test_declared_root_attestation_must_match_the_store(tmp_path, case):
    """The ruling accepts ONE attestation: kind independent, the store's independent check UNIFORM, the quantization it attests."""
    rs = _declared_root(tmp_path, independent=("MIXED" if case == "independent" else "UNIFORM"))
    decl = _declaration(tmp_path, rs, kind=("harness" if case == "kind" else "independent"), quantization=("fp8" if case == "quantization" else "none"))
    seeds, ad, root = _declared_adapter(rs, tmp_path, decl)
    _file_ruling(root)
    F.run_and_ingest(root, ad)
    assert F.q(root, "SELECT COUNT(*) FROM run") == [(0,)] and len(ad.refusals) == 2
    why = ad.refusals.items[0]["why"]
    assert {"kind": "no `attestation` of kind `independent`", "quantization": "attests attacker_quantization 'fp8' but the store's independent check attests 'none'",
            "independent": "is 'MIXED', not UNIFORM"}[case] in why, why


def test_declared_root_whose_harness_check_passed_needs_no_ruling(tmp_path):
    rs = _declared_root(tmp_path, harness="UNIFORM")
    decl = _declaration(tmp_path, rs, ruling_id="r2v4-nobody-filed-this")
    seeds, ad, root = _declared_adapter(rs, tmp_path, decl)
    F.run_and_ingest(root, ad)
    assert len(ad.refusals) == 0 and F.q(root, "SELECT COUNT(*) FROM run") == [(2,)]
    for (extra, fc) in F.q(root, "SELECT extra_json, field_confidence_json FROM run"):
        assert json.loads(extra)["attacker_attestation"] == {"source": "harness", "harness_check": "UNIFORM"}
        assert "attacker_quantization" not in json.loads(fc)


def test_declared_root_also_named_by_a_lane_file_refuses_the_declaration(tmp_path):
    rs = _declared_root(tmp_path, lane_names_it=True)                  # the lane file names the root as a namespace dir
    decl = _declaration(tmp_path, rs)
    seeds, ad, root = _declared_adapter(rs, tmp_path, decl)
    with pytest.raises(AdapterRefusal) as exc:
        ad.discover()
    assert exc.value.path == decl and "already named by lane file" in exc.value.why


def test_declared_root_undeclared_object_key_or_attack_is_refused_by_name(tmp_path):
    rs = _declared_root(tmp_path)
    decl = _declaration(tmp_path, rs, object_keys=(F.LORA,))
    seeds, ad, root = _declared_adapter(rs, tmp_path, decl)
    _file_ruling(root)
    F.run_and_ingest(root, ad)
    assert F.q(root, "SELECT campaign_cell_id_legacy FROM run") == [(F.LORA,)]
    assert len(ad.refusals) == 1 and F.RELU in ad.refusals.items[0]["key"] and "is not declared" in ad.refusals.items[0]["why"]


def test_declaration_file_schema_and_fields_are_checked(tmp_path):
    import yaml
    p = tmp_path / "decl.yaml"
    yaml.safe_dump({"schema": "something/else", "roots": []}, open(p, "w"))
    with pytest.raises(AdapterRefusal) as exc:
        MG.load_declared_roots(str(p))
    assert "is not 'multiturn_declared_roots/1'" in exc.value.why
    yaml.safe_dump({"schema": MG.DECLARED_ROOTS_SCHEMA, "roots": [{"root": "grades_x", "view_legacy": F.VIEW}]}, open(p, "w"))
    with pytest.raises(AdapterRefusal) as exc:
        MG.load_declared_roots(str(p))
    assert "lacks attacks, object_keys, decode_cap_tokens" in exc.value.why


def test_ruling_on_record_log_reads_added_and_seed_files_plain_or_gzipped(tmp_path):
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "rulings", "r2_v9"))
    assert MG.ruling_on_record_log(root, RULING_ID) is None
    with open(os.path.join(root, "rulings", "r2_v9", "000003-ruling.jsonl"), "w") as fh:
        fh.write(json.dumps({"_header": True, "round": "r2"}) + "\n")
        fh.write(json.dumps({"record_type": "ruling", "payload": {"ruling_id": "r2v9-other"}}) + "\n")
    assert MG.ruling_on_record_log(root, RULING_ID) is None and MG.ruling_on_record_log(root, "r2v9-other").endswith("000003-ruling.jsonl")
    p = _file_ruling(root, gz=True)
    assert MG.ruling_on_record_log(root, RULING_ID) == p
    # a record of another type carrying the same id is not a ruling
    with open(os.path.join(root, "rulings", "r2_v9", "000004-disposition.jsonl"), "w") as fh:
        fh.write(json.dumps({"record_type": "disposition", "payload": {"ruling_id": "r2v9-disp"}}) + "\n")
    assert MG.ruling_on_record_log(root, "r2v9-disp") is None


def test_the_checked_in_declaration_names_the_two_run2_roots_with_their_ruling_and_proved_cap():
    """The live declaration this checkout ships: the two si-constitution run-2 stores, cap 4096 proved from the serving attestation
    (the grader's stamp is 512), cluster B, one ruling id for both, evidence naming the config attacker blocks and the four attacker
    serving manifests."""
    inp, roots = MG.load_declared_roots(MG.DECLARED_ROOTS_FILE)
    by = {r["root"]: r for r in roots}
    assert set(by) == {"grades_mt4kscr2_ch_gemma4e4b_it", "grades_mt4kscr2_ch_nemotron3nano4b"}
    assert by["grades_mt4kscr2_ch_gemma4e4b_it"]["attacks"] == ["pair"] and by["grades_mt4kscr2_ch_nemotron3nano4b"]["attacks"] == ["crescendo"]
    for r in roots:
        assert r["object_keys"] == ["prompted__si_constitution"] and r["decode_cap_tokens"] == 4096 and r["served_cluster"] == "cluster_b"
        assert "distinct_max_tokens=[4096]" in r["decode_cap_proof"]["completeness_reading"]
        a = r["attestation"]
        assert a["kind"] == "independent" and a["attacker_quantization"] == "none"
        assert a["ruling_id"] == "r2v4-accept-independent-attacker-attestation-mt4kscr2-run2"
        kinds = [e["kind"] for e in a["evidence"]]
        assert kinds.count("config_attacker_block") == 2 and kinds.count("attacker_serving_manifest") == 4 and "wave_manifest_attacker_pool" in kinds
        assert all(len(e["sha256"]) == 64 for e in a["evidence"] if "sha256" in e)
