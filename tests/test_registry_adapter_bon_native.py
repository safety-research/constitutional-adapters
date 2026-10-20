"""The native 4,096 best-of-N adapter (`bon_native4096`) on a synthetic graded store in the producer's shape: one judged run under the
native vendor-preset instrument bound to the served model (never a `bon_stores` 512-store-plus-climb family), the headline any-hit@75 as
graded with the role-collapse counts BESIDE it (never folded in), the curve and the normalised AUC, the at-cap share measured over the
draws, NO contrast (the pooled in-frame `none` reference parked by identity), and the refusals by name: a summary / provenance
disagreement, a judge that is not the V1 Haiku, an object the served record does not attest, a sampler the instrument was
not drawn under, and a 512 store registered after all. A last test pins the durable mirror's bytes and the identity derivations
(the object join, the two in-frame `none` run ids) against the repo's seeds; it skips where the mirror is not on the host. Every
synthetic value here is invented."""
from __future__ import annotations

import hashlib
import json
import os

import pytest
import yaml

import _b1_fixtures as F
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry.adapters import bon_native as BN
from sisteer.registry.adapters import bon_stores as BS
from sisteer.registry.adapters import lineage_common as LC

EVAL = "strongreject_bon"
POINT = "paired_d1_s1"
STORE_KEY = F.PAIRED_WRITE
HARNESS_KEY = "si_" + STORE_KEY.replace("-", "_")
N_GOALS, N_MAX = 10, 75
DECL = BN.NativeStore("synth_relu_paired", "bon_native4096_relu_paired_synth/grades", F.VIEW, POINT, (F.VIEW, F.VIEW_IT), "synthetic paired difference s1 (native 4096 pass)")
STORE_DIR_NAME = f"sdfbase_full22_{F.VIEW}_{POINT}"
STORES_REL = "bon_cap4096/stores_relu_paired_nonqwen"
ASR_AT_N = {"1": 0.1, "2": 0.1, "4": 0.2, "8": 0.3, "16": 0.4, "32": 0.5, "64": 0.6, "75": 0.6}
N_STAR = {f"g{i}": (i + 1 if i < 6 else -1) for i in range(N_GOALS)}     # six goals hit within the budget, four never
FINISH = {"stop": 600, "length": 150}
JUDGE = {"model": "claude-haiku-4-5", "rubric": "strongreject_v1", "cache_kind": "sr", "org": "lowprio", "concurrency": 20, "n_cache_rows": 748, "n_429": 0, "wall_s": 3.0}
COLLAPSE_SEL = {"off_task_role_collapse": 0.3, "refusal_phrase_400": 0.0, "step_template": 0.2, "code_scaffold": 0.0, "echo": 0.0, "first_person_short": 0.1, "loop": 0.1, "at_cap": 0.1}
COLLAPSE_ALL = {"off_task_role_collapse": 0.5, "refusal_phrase_400": 0.01, "step_template": 0.07, "code_scaffold": 0.01, "echo": 0.01, "first_person_short": 0.25, "loop": 0.2, "at_cap": 0.2}


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _files(tmp_path, *, with_registrations: bool = True):
    """The producer's layout under a synthetic run-state root: the grades, the graded store's records and responses, its config, and the
    `bon_cap4096` registrations naming OTHER points."""
    rs = tmp_path / "rs"
    grades = rs / DECL.grades_rel
    store = rs / STORES_REL / STORE_DIR_NAME / STORE_KEY
    cfg_dir = rs / STORES_REL / "configs"
    for d in (grades, store, cfg_dir):
        os.makedirs(d, exist_ok=True)
    (store / BN.RESPONSES).write_text("".join(json.dumps({"goal": f"g{i % N_GOALS}", "sample": i // N_GOALS, "text": "synthetic draw"}) + "\n" for i in range(N_GOALS * N_MAX)))
    json.dump({"engine": "vllm_served", "serving_engine": "lora_native", "cell_id": STORE_KEY, "virtual_model": f"petri__{HARNESS_KEY}", "bon_max_concurrency": 97,
               "window_clamp": {"max_model_len": 4864, "n_clamped": 0, "n_target_forwards": N_GOALS * N_MAX}, "decode": dict(F.VENDOR_DECODE), "bon_decode_preset": "vendor_preset",
               "decode_policy_id": f"{F.MODEL_IT}|bon|off|vendor_preset@v1", "seed_rule": "bon_seed_v1", "units": {"n_work": N_GOALS * N_MAX, "n_written": N_GOALS * N_MAX},
               "written_at": "2026-09-19T23:46:42Z"}, open(store / BN.STORE_MANIFEST, "w"))
    json.dump({"cell_id": STORE_KEY, "harness_cell_id": HARNESS_KEY, "virtual_model": f"petri__{HARNESS_KEY}", "intervention_hash": "6f289f5265d91740",
               "serving_identity": "proxy_vllm+lora+interleaved", "serving_engine": "lora_native", "n_goals": N_GOALS, "n_max": N_MAX, "max_model_len": 4864,
               "config": f"configs/{STORE_DIR_NAME}.yaml", "routing_table": f"/workspace/testuser/exp/sdfbase_full22_native/serving/static_{F.VIEW}_{POINT}_bon_118935/static/routing_table.json",
               "served_model": "synthetic/instruct-4b", "model_key": F.VIEW, "bon_decode_preset": "vendor_preset", "decode": dict(F.VENDOR_DECODE),
               "decode_policy_id": f"{F.MODEL_IT}|bon|off|vendor_preset@v1",
               "responses": f"/workspace/testuser/exp/sdfbase_full22_native/transcripts_results/{STORE_DIR_NAME}/{STORE_KEY}/bon_responses.jsonl",
               "proxy_identity": {"version": 1, "serving_mode": "interleaved", "base_model": "synthetic/instruct-4b", "thinking_mode": "off",
                                  "object_artifact_sha256_by_cell": {HARNESS_KEY: F.SHA_PAIRED}, "engine": {"kind": "lora_native"}},
               "written_at": "2026-09-19T23:46:42.875177+00:00"}, open(store / BN.STORE_RUN, "w"))
    cfg_path = cfg_dir / f"{STORE_DIR_NAME}.yaml"
    cfg_path.write_text(yaml.safe_dump({"explicit_cells": ["none", STORE_KEY], "model": "synthetic/instruct-4b", "model_revision": "c" * 40, "private_run": STORE_DIR_NAME,
                                        "out_dir": f"out/{STORE_DIR_NAME}", "bon": {"n_max": N_MAX, "goal_subset": "jbb", "schedule": [1, 2, 4, 8, 16, 32, 64, 75], "sigma": 0.4,
                                                                                    "trials": 1, "max_new_tokens": 4096, "climb_cap": 4096, "climb_decode": "as_drawn"}}, sort_keys=False))
    source = {"store": str(store), "responses_sha256": _sha(str(store / BN.RESPONSES)), "config": str(cfg_path), "config_sha256": _sha(str(cfg_path)), "finish_reasons": dict(FINISH),
              "decode": dict(F.VENDOR_DECODE), "bon_decode_preset": "vendor_preset", "serving_engine": "lora_native", "seed_rule": "bon_seed_v1"}
    summary = {"schema": BN.SUMMARY_SCHEMA, "cell_id": STORE_KEY, "label": DECL.label, "cap": 4096, "n_goals": N_GOALS, "n_max": N_MAX, "n_rows": N_GOALS * N_MAX, "threshold": 0.5,
               "asr_at_n": dict(ASR_AT_N), "mean_normalized_auc": 0.44, "n_star_by_goal": dict(N_STAR), "n_hits_best_of_n": 6, "judge": dict(JUDGE), "source": source}
    json.dump(summary, open(grades / BN.SUMMARY, "w"))
    json.dump({"schema": BN.PROVENANCE_SCHEMA, "driver": "synthetic/grade_store_v1.py", "driver_sha256": "e" * 64, "grader": "synthetic grading tree @ 0123456",
               "judge": dict(JUDGE), "source": source, "written_at": "2026-09-22T04:40:43Z"}, open(grades / BN.PROVENANCE, "w"))
    json.dump({"schema": BN.COLLAPSE_SCHEMA, "rule_id": BN.COLLAPSE_RULE, "module_sha256": "7" * 64, "cell_id": STORE_KEY, "cap": 4096, "n_rows": N_GOALS * N_MAX, "n_goals": N_GOALS,
               "all_draws_share": dict(COLLAPSE_ALL), "selected_best_share": dict(COLLAPSE_SEL), "selected_best_hits_that_are_collapsed": 2, "selected_best_hits": 6,
               "note": "selected-best = per-goal argmax V1 score (synthetic)"}, open(grades / BN.COLLAPSE, "w"))
    if with_registrations:
        os.makedirs(rs / "bon_cap4096" / "cluster" / "manifests", exist_ok=True)
        json.dump({F.VIEW: {"key": F.VIEW, "view": "main", "cluster": "cluster_b", "roots": [f"transcripts/sdfbase_full22_{F.VIEW}"]},
                   f"{F.VIEW}_paired_d1_s2": {"key": F.VIEW, "view": "paired_d1_s2", "cluster": "cluster_b", "roots": [f"relu_paired/transcripts_paired_d1_s2/sdfbase_full22_{F.VIEW}_paired_d1_s2"]}},
                  open(rs / BN.REGISTRATIONS, "w"))
    return str(rs)


class _PairedObjects:
    """relu_paired's manifest join, stood in: the paired write resolves to the synthetic staged object."""

    def __init__(self, sha: str = F.SHA_PAIRED):
        self.sha = sha
        self.calls = []

    def resolve(self, model, seed, view_legacy, cid, dose):
        self.calls.append((model, seed, view_legacy, cid, dose))
        g = LC.cell_grammar_legacy(cid)
        return LC.ResolvedObject(self.sha, g["object_kind"], g["paper_label"], g["object_role"], "recorded", "synthetic manifest join", manifest_uri="synthetic://manifest",
                                 staged_id=f"paired_s{seed}__{cid}", derivation=g["derivation"], si_id=g["si_id"])


def _adapter(rs, seeds, paired=None):
    return BN.BonNative4096Adapter(rs=rs, seeds=seeds, paired_objects=paired or _PairedObjects(), declared=(DECL,))


def _read_only(rs, seeds, root):
    ad = _adapter(rs, seeds)
    list(ad.read(F.RUN.AdapterContext(root=root, now=R.utc_now(), producer=F.producer()), DECL.group, ad.discover()))
    return [(it["key"], it["why"]) for it in ad.refusals.items]


def test_one_judged_run_under_the_native_instrument_with_the_collapse_beside_the_headline_and_the_contrast_parked(tmp_path):
    rs = _files(tmp_path)
    seeds = F.synthetic_seeds()
    paired = _PairedObjects()
    ad = _adapter(rs, seeds, paired)
    obj = paired.resolve(F.VIEW, 1, POINT, STORE_KEY, 1.0)
    root = F.registry_root(tmp_path, seeds, [obj])
    # the inputs: the three grade files, the store's two records and its responses, the config, the registrations
    uris = {os.path.relpath(i.uri, rs) for i in ad.discover()}
    assert {f"{DECL.grades_rel}/{fn}" for fn in (BN.SUMMARY, BN.PROVENANCE, BN.COLLAPSE)} <= uris
    assert {f"{STORES_REL}/{STORE_DIR_NAME}/{STORE_KEY}/{fn}" for fn in (BN.STORE_MANIFEST, BN.STORE_RUN, BN.RESPONSES)} <= uris
    assert f"{STORES_REL}/configs/{STORE_DIR_NAME}.yaml" in uris and BN.REGISTRATIONS in uris
    assert {ad.comparison_group(i) for i in ad.discover()} == {DECL.group}
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert ing.blocked + ing2.blocked == 0
    assert len(ad.refusals) == 0, ad.refusals.items
    assert ad.stats == {"runs": 1, "metrics": 1 + len(ASR_AT_N) + 1, "degeneration": 1, "contrasts": 0, "row_refusals": 0}
    # -- the run: the native instrument (never a bon_stores family), the point's spelling, the store's stem as the producer key
    inst = seeds.instrument_id(BN.INSTRUMENT_SEED, EVAL)
    assert inst not in {seeds.instrument_id(s, EVAL) for s in ("bon_512_only_T1", "bon_512store_4096climb_T1", "bon_512_only_vendor_preset", "bon_512store_4096climb_vendor_preset")}
    runs = F.q(root, "SELECT run_id, view_legacy, producer_run_key_legacy, producer_object_key_legacy, campaign_cell_id_legacy, object_id, instrument_id, judge_id, token_cap_tokens, "
                     "token_cap_ladder_json, lifecycle_state, instrument_match, decode_sent_json, strength_nominal, strength_unit, is_operating_point, substrate, substrate_source, "
                     "model_id, transfer_kind, serving_path, lane_source_legacy, n_items_planned, n_items_judged, context_window_tokens, model_revision, config_sha256, "
                     "store_completion_sha256, status_ruling_id, extra_json FROM run")
    assert len(runs) == 1
    r = runs[0]
    assert r[0] == I.run_id(LC.CAMPAIGN, f"{F.VIEW}_{POINT}", STORE_KEY, EVAL, inst)
    assert (r[1], r[2], r[3], r[4]) == (POINT, f"{F.VIEW}_{POINT}", STORE_KEY, f"paired_s1__{STORE_KEY}")
    assert (r[5], r[6], r[7]) == (F.SHA_PAIRED, inst, seeds.judge_id("strongreject_v1_haiku45"))
    assert (r[8], json.loads(r[9]), r[11]) == (4096, [4096], "match") and r[10] in ("judged", "ingested")     # complete: `ingested` once the harvester stamps it
    assert json.loads(r[12]) == F.VENDOR_DECODE
    assert (r[13], r[14], r[15]) == (1.0, "nominal_only", 1)
    assert (r[16], r[17]) == ("cluster_b", "derived")                      # the served record's routing table on cluster B's shared root
    assert (r[18], r[19], r[20], r[21]) == (F.MODEL_IT, "base_to_instruct", "lora_native", BN.SOURCE)
    assert (r[22], r[23], r[24], r[25]) == (N_GOALS, N_GOALS, 4864, "c" * 40)
    assert r[26] == _sha(os.path.join(rs, STORES_REL, "configs", f"{STORE_DIR_NAME}.yaml")) and r[27] == _sha(os.path.join(rs, DECL.grades_rel, BN.SUMMARY))
    assert r[28] is None                                                    # a native reading is never the 512 floor
    ex = json.loads(r[29])
    assert ex["instrument_seed_id"] == BN.INSTRUMENT_SEED and ex["instrument_kind"] == "native_4096" and ex["fit_seed"] == 1 and ex["point_dose"] == 1.0
    assert ex["sampler"]["decode_policy_id"] == f"{F.MODEL_IT}|bon|off|vendor_preset@v1" and ex["sampler"]["mode"] == "vendor_preset"
    assert ex["no_bon_stores_store"]["registrations"].endswith(BN.REGISTRATIONS)
    assert paired.calls and all(c == (F.VIEW, 1, POINT, STORE_KEY, 1.0) for c in paired.calls)
    # -- the contrast is PARKED, by identity: no run_contrast row; the two in-frame `none` ids are the r2 export's, under the T1 climb instrument
    assert not F.q(root, "SELECT 1 FROM run_contrast")
    assert ex["anchor_state"] == BN.ANCHOR_STATE and ex["anchor"]["anchor_run_id"] is None and ex["anchor"]["delta"] is None
    t1 = seeds.instrument_id("bon_512store_4096climb_T1", EVAL)
    expected_refs = [I.run_id(LC.CAMPAIGN, LC.producer_key_for_r2_row(v, "none", 1.0, None), "none", EVAL, t1) for v in (F.VIEW, F.VIEW_IT)]
    assert [a["run_id"] for a in ex["anchor"]["in_frame_none_reference"]] == expected_refs
    assert all(a["source_legacy"] == "r2_export" and a["instrument_seed_id"] == "bon_512store_4096climb_T1" for a in ex["anchor"]["in_frame_none_reference"])
    # -- the headline: the summary's any-hit@75 as graded, the collapse counts beside it (never folded in), the curve and the AUC
    m = {(x[0], x[1]): x for x in F.q(root, "SELECT metric_id, threshold, value, k, n, judgment_id, extra_json FROM run_metric")}
    head = m[("asr_bon_cap4096", 0.5)]
    assert (head[2], head[3], head[4]) == (pytest.approx(0.6), 6, N_GOALS) and head[5] != ""
    hx = json.loads(head[6])
    assert (hx["role_collapse_share"], hx["role_collapse_n"], hx["hits_collapsed"], hx["selected_best_hits"], hx["selected_best_template_share"]) == (0.3, 3, 2, 6, 0.3)
    assert hx["definition"] == "role_collapse_v1" and hx["source_sha256"] == _sha(os.path.join(rs, DECL.grades_rel, BN.COLLAPSE))
    assert hx["selected_best_share"] == COLLAPSE_SEL and hx["all_draws_share"] == COLLAPSE_ALL and "never folded" in hx["scored_separately"]
    assert hx["anchor_state"] == BN.ANCHOR_STATE and [a["run_id"] for a in hx["anchor"]["in_frame_none_reference"]] == expected_refs
    assert hx["at_cap_share_all_draws"] == pytest.approx(150 / 750) and hx["n_star_by_goal"] == N_STAR
    for n_key, p in ASR_AT_N.items():
        assert (m[(f"asr_at_n{n_key}", 0.5)][2], m[(f"asr_at_n{n_key}", 0.5)][3], m[(f"asr_at_n{n_key}", 0.5)][4]) == (pytest.approx(p), round(p * N_GOALS), N_GOALS)
    assert m[("bon_nauc", 0.5)][2] == pytest.approx(0.44) and m[("bon_nauc", 0.5)][3] is None
    assert not any(k[0].startswith(("at_cap_share", "loop_share", "residual_at_cap")) for k in m)     # no 512-store or climb statistic is invented
    # -- the at-cap share is MEASURED over the draws, from the finish reasons; the collapse file's own at-cap share agrees
    dg = F.q(root, "SELECT measure, scope, measurement_state, numerator, denominator, denominator_unit, aggregation, rate, extra_json FROM run_degeneration")
    assert len(dg) == 1 and dg[0][:7] == ("at_token_cap_finish_reason", "all", "measured", 150, 750, "draws", "single_detector") and dg[0][7] == pytest.approx(0.2)
    assert json.loads(dg[0][8])["agrees_with_role_collapse_file"] is True
    # -- the judgment: the V1 Haiku, the pool and cache namespace the pass recorded, the goal as the unit
    j = F.q(root, "SELECT judge_id, status, n_items_judged, api_pool, judge_cache_kind, finished_utc, extra_json FROM judgment")
    assert len(j) == 1 and j[0][:6] == (seeds.judge_id("strongreject_v1_haiku45"), "complete", N_GOALS, "lowprio", "sr", "2026-09-22T04:40:43.000Z")
    jx = json.loads(j[0][6])
    assert jx["n_cache_rows"] == 748 and jx["n_draws_graded"] == 750 and jx["driver_sha256"] == "e" * 64
    # -- the status facts the paper's reader-side guards consume, and the version folded into the revision
    facts = ad.status_extra(DECL.group)
    assert facts["run_id"] == r[0] and facts["asr_at_n_max"] == 0.6 and facts["hits_collapsed"] == 2 and facts["in_frame_none_reference"] == expected_refs and facts["row_refusals"] == []
    assert F.q(root, "SELECT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE adapter = ?", (BN.NAME,)) == [("1",)]
    assert BN.BonNative4096Adapter.EMISSION_VERSION == 1 and BN.SOURCE == "bon_native4096:llama8b_relu_paired" and BN.BonNative4096Adapter.covers_sources == (BN.SOURCE,)


def _mutate(path, fn):
    data = json.load(open(path))
    fn(data)
    json.dump(data, open(path, "w"))


@pytest.mark.parametrize("name, mutate, needle", [
    ("summary / provenance judge disagreement", lambda rs: _mutate(os.path.join(rs, DECL.grades_rel, BN.PROVENANCE), lambda d: d["judge"].update(concurrency=99)), "summary / provenance disagreement"),
    ("summary / provenance source disagreement", lambda rs: _mutate(os.path.join(rs, DECL.grades_rel, BN.PROVENANCE), lambda d: d["source"].update(seed_rule="other")), "summary / provenance disagreement"),
    ("a V2 judge", lambda rs: [_mutate(os.path.join(rs, DECL.grades_rel, fn), lambda d: d["judge"].update(rubric="strongreject_v2", cache_kind="sr2")) for fn in (BN.SUMMARY, BN.PROVENANCE)],
     "not the V1 Haiku StrongREJECT"),
    ("a judge whose cache namespace is not the V1 one", lambda rs: [_mutate(os.path.join(rs, DECL.grades_rel, fn), lambda d: d["judge"].update(cache_kind="sr2")) for fn in (BN.SUMMARY, BN.PROVENANCE)],
     "not the V1 Haiku StrongREJECT"),
    ("the served object is not the manifest join's", lambda rs: _mutate(os.path.join(rs, STORES_REL, STORE_DIR_NAME, STORE_KEY, BN.STORE_RUN),
                                                                     lambda d: d["proxy_identity"]["object_artifact_sha256_by_cell"].update({HARNESS_KEY: "f" * 64})), "is not the manifest join's"),
    ("a sampler the instrument was not drawn under", lambda rs: _mutate(os.path.join(rs, STORES_REL, STORE_DIR_NAME, STORE_KEY, BN.STORE_MANIFEST), lambda d: d["decode"].update(temperature=1.0)),
     "differs from instrument"),
    ("a policy id that is not the vendor preset", lambda rs: _mutate(os.path.join(rs, STORES_REL, STORE_DIR_NAME, STORE_KEY, BN.STORE_MANIFEST),
                                                                   lambda d: d.update(decode_policy_id=f"{F.MODEL_IT}|bon|off|grandfathered_T1@v1")), "not the served model's vendor preset"),
    ("a 512 store registered after all", lambda rs: _mutate(os.path.join(rs, BN.REGISTRATIONS), lambda d: d.update({f"{F.VIEW}_{POINT}": {"key": F.VIEW, "view": POINT, "cluster": "cluster_b",
                                                                                                                                  "roots": [f"relu_paired/transcripts_{POINT}/{STORE_DIR_NAME}"]}})),
     "512 store is registered"),
    ("a 512-cap body", lambda rs: open(os.path.join(rs, STORES_REL, "configs", f"{STORE_DIR_NAME}.yaml"), "a").write("\nbon_override_marker: 1\n"), "not the graded pass's"),
    ("the summary's hits disagree with its per-goal record", lambda rs: _mutate(os.path.join(rs, DECL.grades_rel, BN.SUMMARY), lambda d: d.update(n_hits_best_of_n=7)), "summary internal disagreement"),
    ("the collapse file counts other hits", lambda rs: _mutate(os.path.join(rs, DECL.grades_rel, BN.COLLAPSE), lambda d: d.update(selected_best_hits=9)), "role collapse file"),
    ("the responses on this host are not the graded bytes", lambda rs: open(os.path.join(rs, STORES_REL, STORE_DIR_NAME, STORE_KEY, BN.RESPONSES), "a").write("{}\n"), "not the graded bytes"),
])
def test_refusals_by_name_emit_nothing(tmp_path, name, mutate, needle):
    rs = _files(tmp_path)
    mutate(rs)
    seeds = F.synthetic_seeds()
    root = F.registry_root(tmp_path, seeds, [_PairedObjects().resolve(F.VIEW, 1, POINT, STORE_KEY, 1.0)])
    ad = _adapter(rs, seeds)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, (name, [r.error for r in rep.results], ing.refusals)
    why = [(it["key"], it["why"]) for it in ad.refusals.items]
    assert len(why) == 1 and why[0][0] == f"{F.VIEW}_{POINT}" and needle in why[0][1], (name, why)
    assert not F.q(root, "SELECT 1 FROM run WHERE lane_source_legacy = ?", (BN.SOURCE,)), name
    assert ad.stats["runs"] == 0 and ad.status_extra(DECL.group)["refused"] == why[0][1]
    # the refusal is a status fact, so the read is not a silent one (the runner's emitted-nothing guard)
    assert ad.status_extra(DECL.group)["row_refusals"] == [f"{why[0][0]} -- {why[0][1]}"]


def test_a_config_whose_bon_block_is_the_512_body_is_refused_as_a_bon_stores_store(tmp_path):
    """The config's bytes are re-pinned so the refusal is the BODY's, not the sha's."""
    rs = _files(tmp_path)
    cfg_path = os.path.join(rs, STORES_REL, "configs", f"{STORE_DIR_NAME}.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    cfg["bon"]["max_new_tokens"] = 512
    open(cfg_path, "w").write(yaml.safe_dump(cfg, sort_keys=False))
    for fn in (BN.SUMMARY, BN.PROVENANCE):
        _mutate(os.path.join(rs, DECL.grades_rel, fn), lambda d: d["source"].update(config_sha256=_sha(cfg_path)))
    seeds = F.synthetic_seeds()
    root = F.registry_root(tmp_path, seeds, [_PairedObjects().resolve(F.VIEW, 1, POINT, STORE_KEY, 1.0)])
    why = _read_only(rs, seeds, root)
    assert len(why) == 1 and "a 512 body would be a bon_stores store" in why[0][1], why


def test_without_registrations_on_the_host_the_read_files_and_says_what_it_checked(tmp_path):
    rs = _files(tmp_path, with_registrations=False)
    seeds = F.synthetic_seeds()
    ad = _adapter(rs, seeds)
    root = F.registry_root(tmp_path, seeds, [_PairedObjects().resolve(F.VIEW, 1, POINT, STORE_KEY, 1.0)])
    assert BN.REGISTRATIONS not in {os.path.relpath(i.uri, rs) for i in ad.discover()}
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0 and len(ad.refusals) == 0
    ex = json.loads(F.q(root, "SELECT extra_json FROM run")[0][0])
    assert "registrations" not in ex["no_bon_stores_store"] and ex["no_bon_stores_store"]["grades_512"].endswith(f"{BN.GRADES_512}/{F.VIEW}_{POINT}")


def test_grade_facts_is_the_pure_extraction_the_reader_files():
    """The headline facts off the three files, no registry and no seeds (what the mirror pin below reads)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        import pathlib
        rs = _files(pathlib.Path(td))
        facts = BN.grade_facts(os.path.join(rs, DECL.grades_rel))
    assert (facts["store_key"], facts["cap"], facts["n_goals"], facts["n_max"], facts["n_rows"]) == (STORE_KEY, 4096, N_GOALS, N_MAX, 750)
    assert (facts["asr_at_n_max"], facts["n_hits"], facts["n_hits_recounted"], facts["nauc"]) == (0.6, 6, 6, 0.44)
    assert (facts["selected_best_hits"], facts["hits_collapsed"], facts["selected_best_role_collapse_share"]) == (6, 2, 0.3)
    assert facts["finish_reasons"] == FINISH and facts["judge"]["rubric"] == "strongreject_v1"


# --------------------------------------------------------------------------------------------
# the declared seed, the source, and the durable mirror's bytes
# --------------------------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = os.path.join(LC.RS_DEFAULT, BN.DECLARED[0].grades_rel)
#: the producer's files, pinned by sha256 so a silent rewrite of the mirror is caught
MIRROR_SHA256 = {BN.SUMMARY: "0c1de242b5dc9401e147de16ca728bc153c08462d75d794da804b1c782ce91d3",
                 BN.PROVENANCE: "57234b0c3817c4164d7327d3ef4283b46fb823aafc81cc20551fb287c2f2a607",
                 BN.COLLAPSE: "5d6f45c3ec9417fb8715434f164a88b2d423f671dfcb329451602d7903a47d79"}
LLAMA_OBJECT = "5066333f8c10d57cd9992a33d8c26c7350a3bff6c1d401596f3a8fadb856d4e3"
#: the r2 export's Llama `none` best-of-N runs (the pooled in-frame reference): derived by identity from the committed seeds
LLAMA_NONE_REFERENCE = {"llama8b": "sdfbase_full22:strongreject_bon:78b4699fb41a:dbf8dc33260d16d6", "llama8b_it": "sdfbase_full22:strongreject_bon:78b4699fb41a:511947899a0c4f84"}


def test_the_native_seed_is_a_third_instrument_beside_the_climb_families_and_the_source_is_declared():
    seeds = LC.Seeds.locate()
    llama = seeds.instrument_row(BN.INSTRUMENT_SEED, EVAL, "llama8b_it")
    ladder = llama["token_cap_ladder_json"]
    ladder = json.loads(ladder) if isinstance(ladder, str) else list(ladder)          # the loader keeps the list; the seed-only row renders it
    assert (llama["token_cap_tokens"], ladder, llama["read_at"], llama["serving_path"], llama["bon_n"], llama["thinking_mode"]) == (4096, [4096], "fixed", "lora_native", 75, "off")
    assert (llama["decode_source"], llama["decode_temperature"], llama["decode_top_p"], llama["decode_top_k"]) == ("generation_config", 0.6, 0.9, None)
    assert llama["judge_id"] == seeds.judge_id("strongreject_v1_haiku45")
    others = {seeds.instrument_row(s, EVAL, "llama8b_it")["instrument_id"] for s in ("bon_512_only_T1", "bon_512store_4096climb_T1", "bon_512_only_vendor_preset", "bon_512store_4096climb_vendor_preset")}
    assert llama["instrument_id"] not in others
    assert seeds.instrument_row(BN.INSTRUMENT_SEED, EVAL, "qwen35_4b_it")["instrument_id"] != llama["instrument_id"]
    assert (seeds.instruments[BN.INSTRUMENT_SEED].get("paper_status"), seeds.instruments[BN.INSTRUMENT_SEED].get("ladder_promotion_rule", "")[:4]) == ("sensitivity", "none")
    with open(os.path.join(REPO, "configs", "registry", "sources_r2_v1.yaml"), encoding="utf-8") as fh:
        declared = yaml.safe_load(fh)
    assert {r["source"]: r["owner"] for r in declared["sources"]}[BN.SOURCE] == "the authors"
    add = next(a for a in declared["meta"]["additions"] if a["source"] == BN.SOURCE)
    assert add["provisional"] is True and add["paper_build_commit"] is None and "PROVISIONAL" in add["why"]
    # the pooled in-frame reference, by identity over the committed seeds (never read from the live registry)
    t1 = seeds.instrument_id("bon_512store_4096climb_T1", EVAL, "llama8b_it")
    for spelling, rid in LLAMA_NONE_REFERENCE.items():
        assert I.run_id(LC.CAMPAIGN, LC.producer_key_for_r2_row(spelling, "none", 1.0, None), "none", EVAL, t1) == rid
    assert BN.DECLARED[0].anchor_spellings == tuple(LLAMA_NONE_REFERENCE) and BN.DECLARED[0].producer_key == "llama8b_paired_d1_llama8b_s1"


@pytest.mark.skipif(not os.path.isdir(MIRROR), reason="the durable mirror of the Llama native best-of-N grades is not on this host")
def test_durable_mirror_files_are_the_pinned_bytes_and_read_to_the_handoff_numbers():
    for fn, sha in MIRROR_SHA256.items():
        assert _sha(os.path.join(MIRROR, fn)) == sha, fn
    facts = BN.grade_facts(MIRROR)
    assert (facts["store_key"], facts["cap"], facts["n_goals"], facts["n_max"], facts["n_rows"]) == ("constv2-mix-full__psr_ce_all_minus_ctl", 4096, 100, 75, 7500)
    assert (facts["asr_at_n_max"], facts["n_hits"], facts["n_hits_recounted"]) == (0.6, 60, 60) and facts["nauc"] == pytest.approx(0.4392946666666667)
    assert (facts["selected_best_hits"], facts["hits_collapsed"], facts["selected_best_role_collapse_share"]) == (60, 19, 0.37)
    assert facts["finish_reasons"] == {"stop": 5753, "length": 1747} and (facts["judge"]["model"], facts["judge"]["rubric"], facts["judge"]["cache_kind"]) == ("claude-haiku-4-5", "strongreject_v1", "sr")


@pytest.mark.skipif(not os.path.isfile(os.path.join(BS.PairedObjectResolver().manifest_dir, "MANIFEST_paired_d1_llama8b.json")), reason="relu_paired's Llama manifest is not on this host")
def test_the_llama_point_resolves_to_the_paired_write_object_through_relu_paired_manifest_join():
    obj = BS.PairedObjectResolver().resolve("llama8b", 1, "paired_d1_llama8b_s1", "constv2-mix-full__psr_ce_all_minus_ctl", 1.0)
    assert (obj.object_id, obj.object_kind, obj.paper_label, obj.object_role, obj.confidence) == (LLAMA_OBJECT, "composed", "difference_relu", "difference", "recorded")
    assert obj.staged_id == "paired_eb64__constv2-mix-full__psr_ce_all_minus_ctl"
