"""The BCB/BFCL adapter resolves a fitted arm's object sha from the served store's OWN
manifest when the paper row carries none, cross-checks the manifest against the row, lifts its serving fields onto the run,
and registers every manifest it reads as a watermarked input of its parquet's group.

Fixtures are SYNTHETIC (invented numbers, placeholder shas, a temp store root, host TESTHOST); nothing here reads a real store.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import bcb_bfcl as BB
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal, RecordBatch
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

NOW = "2026-09-17T10:00:00.000Z"
PRODUCER = {"script": "tests.declared.d60", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
SHA_OBJ = hashlib.sha256(b"synthetic d60 served object").hexdigest()
SHA_OTHER = hashlib.sha256(b"synthetic d60 other object").hexdigest()
REVISION = "c" * 40
TREE = synth.code_sha("d")
CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"
INTERVENTION8, HARNESS8 = "3a3a3a3a", "5b5b5b5b"
INTERVENTION16, HARNESS16 = INTERVENTION8 + "0f0f0f0f", HARNESS8 + "e1e1e1e1"
STORE_KEY = f"constv2-mix-full--lora_r4_ce_minus_ctl__controls__t0__a0__{INTERVENTION8}__{HARNESS8}"
NONE_KEY = "none__controls__t0__a0__2ff9cdff__e95bd266"
WAVE = "synthwave-0000000"


def _ctx() -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER))


def _read_all(adapter):
    """discover -> comparison groups -> read, every record validated; returns (records, report, inputs)."""
    inputs = adapter.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(adapter.comparison_group(i), []).append(i)
    recs = []
    ctx = AF.primed_ctx(adapter, now=NOW, producer=PRODUCER, groups=groups)     # the frame of the pass AFTER the fold: every own anchor current
    for g, ins in sorted(groups.items()):
        for batch in adapter.read(ctx, g, ins):
            recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    return recs, adapter.last_report, inputs


def _by_type(recs):
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out


# --------------------------------------------------------------------------------------------
# synthetic parquet + store manifest builders
# --------------------------------------------------------------------------------------------

def build_parquet(root: str, store_root: str, name: str = "paper_rows_bcb_bfcl_200_synth.parquet", *, wave: str = WAVE, sha=None,
                  evals=("bigcodebench_200", "bfcl_200"), with_sha_column: bool = True, steered_status: str = "graded",
                  steered_value: float = 0.55, unkeyed_arm_status: str = None) -> str:
    """Rows in the paper's long schema: a `none` anchor and the diff_lora arm per eval, the arm's rows carrying ``sha`` (None ->
    no object sha, the first-wave shape), each note naming its store key and ``provenance`` the wave's store directory.
    ``unkeyed_arm_status`` adds, per eval, the widen wave's shape of an arm whose object has NO campaign cell id (a roster entry
    the lane could not stage): ``cell_id`` null, unmeasured, its status the given text."""
    import pandas as pd
    base = {c: None for c in DC.LONG_COLUMNS if c not in ("saturated", "near_floor", "suggestive")}
    base.update(saturated=False, near_floor=False, suggestive=False)
    rows = []
    for eval_id in evals:
        cap = 1024 if eval_id == "bfcl_200" else 4096
        for cell, arm, fam, ot, val, k, delta, key in (("none", "none", "none", "none", 0.5, 100, None, NONE_KEY),
                                                       (CELL, "diff_lora", "difference", "lora", steered_value, 110, 5.0, STORE_KEY)):
            measured = cell == "none" or steered_status == "graded"
            r = dict(base, source="bcb_bfcl_200", model="gemma4e4b", lane_kind="base→instruct", view="gemma4e4b", dose=1.0, cell_id=cell, arm=arm, family=fam,
                     object_type=ot, component="capability", eval_id=eval_id, eval=DC.EVAL_LABEL[eval_id], metric="acc", threshold=float("nan"), bon_n=None,
                     value=(val if measured else float("nan")), k=(k if measured else float("nan")), n=200, ci_lo=val - 0.07, ci_hi=val + 0.07,
                     delta=(delta if delta is not None and measured else float("nan")), delta_lo=(delta - 5 if delta is not None and measured else float("nan")),
                     delta_hi=(delta + 5 if delta is not None and measured else float("nan")), delta_p=(0.23 if delta is not None and measured else float("nan")),
                     delta_method=("paired exact McNemar + bootstrap" if delta is not None and measured else float("nan")),
                     at_cap_share=0.0, decode_cap_tokens=cap, decode=("greedy T=0" if eval_id == "bfcl_200" else "sampled T=1.0 top_p=0.95 top_k=64 (model card), k=1, thinking OFF"),
                     judge=("bfcl ast match" if eval_id == "bfcl_200" else "bigcodebench verify (sandboxed unit tests)"),
                     status=("graded" if measured else steered_status),
                     note=f"synthetic served leg inspect/{eval_id} on {wave}; store {key}; OFF cap {cap}; rung-1 at-cap 0/200",
                     provenance=os.path.join(store_root, wave), thinking="off", held_out=False)
            if with_sha_column:
                r["object_artifact_sha256"] = sha if cell != "none" else None
            rows.append(r)
        if unkeyed_arm_status:
            r = dict(base, source="bcb_bfcl_200", model="gemma4e4b", lane_kind="base→instruct", view="gemma4e4b", dose=1.0, cell_id=None,
                     arm="alphasteer_as_fit", family="baseline", object_type="matrix", component="capability", eval_id=eval_id, eval=DC.EVAL_LABEL[eval_id],
                     metric="acc", threshold=float("nan"), bon_n=None, value=float("nan"), k=float("nan"), n=200, ci_lo=float("nan"), ci_hi=float("nan"),
                     delta=float("nan"), delta_lo=float("nan"), delta_hi=float("nan"), delta_p=float("nan"), delta_method=float("nan"),
                     at_cap_share=float("nan"), decode_cap_tokens=cap, decode=("greedy T=0" if eval_id == "bfcl_200" else "sampled T=1.0 top_p=0.95 top_k=64 (model card), k=1, thinking OFF"),
                     judge=("bfcl ast match" if eval_id == "bfcl_200" else "bigcodebench verify (sandboxed unit tests)"), status=unkeyed_arm_status,
                     note="unblock: stage the as-fit object under a cell id and a pin (synthetic)", provenance="roster wave synth entry alphasteer_as_fit",
                     thinking="off", held_out=False)
            if with_sha_column:
                r["object_artifact_sha256"] = None
            rows.append(r)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, name)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def build_manifest(store_root: str, wave: str = WAVE, cell: str = CELL, store_key: str = STORE_KEY, sha=SHA_OBJ, **over) -> str:
    """The store's own manifest at ``<store_root>/<wave>/<cell>/cells/<store_key>/manifest.json`` (the producer's field names)."""
    m = {"cell_id": cell, "model_id": "google/gemma-4-E4B-it", "model_revision": REVISION, "intervention_hash": INTERVENTION16,
         "harness_config_hash": HARNESS16, "serving_identity": "proxy_vllm+lens+interleaved", "serving_path": "lens", "engine": "vllm",
         "contract_version": 3, "controls": ["bigcodebench_200", "bfcl_200"], "steer_positions": "response",
         "producer": {"driver": "served_capability_run", "tree_sha": TREE},
         "fit_provenance": {"object_sha": SHA_OBJ[:16], "positions_at_fit": "all", "fit_kind": "document"},
         "provenance": {"arm": "lora_r4_ce_minus_ctl", "artifact_sha256": sha, "cell_id": cell, "code_version": "cell_controls_v1",
                        "intervention_hash": INTERVENTION16, "judge_model": "synthetic-judge", "model": "google/gemma-4-E4B-it", "model_revision": REVISION,
                        "serving": {"engine": "vllm", "serving_identity": "proxy_vllm+lens+interleaved", "resolved_revision": REVISION}}}
    for k, v in over.items():
        if k.startswith("provenance__"):
            m["provenance"][k[len("provenance__"):]] = v
        elif v is ...:
            m.pop(k, None)
        else:
            m[k] = v
    path = os.path.join(store_root, wave, cell, "cells", store_key, "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1)
    return path


def _adapter(tmp_path):
    return BB.BcbBfclAdapter(pattern=os.path.join(str(tmp_path / "an"), "paper_rows_*.parquet"), store_root=str(tmp_path / "store"))


def _steered_runs(bt):
    return [r for r in bt.get("run", []) if r["campaign_cell_id_legacy"] == CELL]


def _refusals_for(rep, needle: str):
    return [x for x in rep.refusals if needle in x]


# --------------------------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------------------------

def test_object_sha_resolved_from_the_store_manifest_and_serving_fields_lifted(tmp_path):
    store = str(tmp_path / "store")
    pq = build_parquet(str(tmp_path / "an"), store, sha=None)
    mf = build_manifest(store)
    ad = _adapter(tmp_path)
    recs, rep, inputs = _read_all(ad)
    bt = _by_type(recs)
    assert rep.refusals == []
    runs = _steered_runs(bt)
    assert len(runs) == 2 and len(bt["run"]) == 4 and len(bt["run_contrast"]) == 2
    for r in runs:
        assert r["object_id"] == SHA_OBJ
        assert r["serving_path"] == "lens" and r["serving_identity"] == "proxy_vllm+lens+interleaved"
        assert r["harness_config_hash"] == HARNESS16 and r["intervention_hash"] == INTERVENTION16 and r["model_revision"] == REVISION
        fc = r["field_confidence_json"]
        assert fc["serving_path"]["confidence"] == "recorded" and "manifest" in fc["serving_path"]["source"]
        assert r["extra"]["store_manifest"] == os.path.relpath(mf, store) and r["extra"]["store_manifest_resolution"] == "exact"
        assert r["extra"]["store_producer_tree_sha"] == TREE and r["extra"]["object_sha_source"].startswith("store manifest")
        assert r["applied_positions_mode"] == "all"       # lens + fit_positions=all derives the mode; it was unverified with no serving path
    # the anchor rows are untouched: no manifest consulted, serving path still unverified with its reason
    anchors = [r for r in bt["run"] if r["campaign_cell_id_legacy"] == "none"]
    assert len(anchors) == 2 and all(r["serving_path"] == "unverified" and "store_manifest" not in r["extra"] for r in anchors)
    # the run-level instrument was minted from the lifted serving path
    inst = {i["instrument_id"]: i for i in bt["instrument"]}
    assert all(inst[r["instrument_id"]]["serving_path"] == "lens" for r in runs)
    # the manifest is a discovered input of the parquet's group, hashed and role `report`
    man = [i for i in inputs if i.uri == mf]
    assert len(man) == 1 and man[0].role == "report" and man[0].sha256 == F.sha256_file(mf)
    assert ad.comparison_group(man[0]) == ad.comparison_group([i for i in inputs if i.uri == pq][0])
    assert len(inputs) == 2
    # facts and source states name it
    facts = ad.last_facts[ad.comparison_group(man[0])][os.path.basename(pq)]
    assert facts["n_object_from_manifest"] == 2 and facts["manifests"] == [os.path.relpath(mf, store)] and facts["manifest_refused"] == []
    states = ad.source_states(_ctx(), inputs)
    assert {s.source_id for s in states} == {f"bcb_bfcl:{os.path.basename(pq)}", f"bcb_bfcl:{os.path.relpath(mf, store)}"}
    # the artifact payload the runner writes for the manifest input
    art = man[0].artifact_payload()
    assert art["artifact_id"] == man[0].sha256 and art["kind"] == "report"


def test_rows_that_carry_a_sha_do_not_consult_the_store(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store, sha=SHA_OTHER)
    build_manifest(store)                    # present, and it names a DIFFERENT object: the record's own sha wins and the manifest is never read
    ad = _adapter(tmp_path)
    recs, rep, inputs = _read_all(ad)
    bt = _by_type(recs)
    assert rep.refusals == [] and len(inputs) == 1
    assert all(r["object_id"] == SHA_OTHER and r["serving_path"] == "unverified" and "store_manifest" not in r["extra"] for r in _steered_runs(bt))
    assert ad.last_discovery[os.path.basename(inputs[0].uri)] == {"resolved": [], "unresolved": []}


def test_first_wave_shape_without_the_sha_column_resolves_too(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store, with_sha_column=False)
    build_manifest(store)
    recs, rep, inputs = _read_all(_adapter(tmp_path))
    assert rep.refusals == [] and len(inputs) == 2
    assert all(r["object_id"] == SHA_OBJ for r in _steered_runs(_by_type(recs)))


def test_glob_fallback_with_one_candidate_resolves_and_says_so(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store, wave="wave-not-on-disk-0000")
    mf = build_manifest(store, wave="synthwave-elsewhere")
    ad = _adapter(tmp_path)
    recs, rep, inputs = _read_all(ad)
    assert rep.refusals == [] and any(i.uri == mf for i in inputs)
    runs = _steered_runs(_by_type(recs))
    assert len(runs) == 2 and all(r["object_id"] == SHA_OBJ and r["extra"]["store_manifest_resolution"] == "glob" for r in runs)
    path, how, tried = ad.resolve_manifest("wave-not-on-disk-0000", CELL, STORE_KEY)
    assert path == mf and how == "glob" and tried[0].startswith(os.path.join(store, "wave-not-on-disk-0000")) and "*" in tried[1]


# --------------------------------------------------------------------------------------------
# refusals — by name, with the path tried
# --------------------------------------------------------------------------------------------

def test_missing_manifest_keeps_the_refusal_and_names_the_path_tried(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store)
    ad = _adapter(tmp_path)
    recs, rep, inputs = _read_all(ad)
    bt = _by_type(recs)
    assert len(inputs) == 1                                      # nothing to watermark: no manifest exists
    assert _steered_runs(bt) == [] and len(bt["run"]) == 2      # the anchors are filed, the steered rows refused
    assert len(rep.refusals) == 2
    expected = os.path.join(store, WAVE, CELL, "cells", STORE_KEY, "manifest.json")
    for eval_id in ("bigcodebench_200", "bfcl_200"):
        hits = _refusals_for(rep, f"gemma4e4b/{CELL}/{eval_id}: ")
        assert len(hits) == 1
        assert "fitted object with no 64-hex artifact sha" in hits[0] and expected in hits[0] and "*" in hits[0]
    assert ad.last_discovery[os.path.basename(inputs[0].uri)]["unresolved"][0]["how"] == "missing"
    # no contrast was filed for a refused row, and no run_metric either
    assert len(bt["run_contrast"]) == 0 if "run_contrast" in bt else True
    assert len(bt["run_metric"]) == 2


def test_manifest_cell_id_mismatch_is_refused_by_name(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store)
    mf = build_manifest(store, cell_id="constv2-mix-full__lora_r4_ce")
    recs, rep, inputs = _read_all(_adapter(tmp_path))
    assert _steered_runs(_by_type(recs)) == [] and len(rep.refusals) == 2
    assert all(mf in x and "names cell_id 'constv2-mix-full__lora_r4_ce'" in x and f"gemma4e4b/{CELL}/" in x for x in rep.refusals)
    assert any(i.uri == mf for i in inputs)                       # discovered (its bytes ARE what the refusal is about), still refused


def test_manifest_intervention_mismatch_is_refused_by_name(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store)
    mf = build_manifest(store, intervention_hash="7c7c7c7c" + "0" * 8, provenance__intervention_hash="7c7c7c7c" + "0" * 8)
    recs, rep, _ = _read_all(_adapter(tmp_path))
    assert _steered_runs(_by_type(recs)) == [] and len(rep.refusals) == 2
    assert all(mf in x and "does not start with the store key's intervention segment" in x and INTERVENTION8 in x for x in rep.refusals)


def test_manifest_harness_hash_mismatch_and_internal_disagreement_are_refused(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store)
    build_manifest(store, harness_config_hash="ffffffff" + "0" * 8)
    recs, rep, _ = _read_all(_adapter(tmp_path))
    assert _steered_runs(_by_type(recs)) == [] and all("harness segment" in x for x in rep.refusals) and len(rep.refusals) == 2
    # the provenance block disagreeing with the manifest's own intervention_hash is an inconsistent manifest: refused too
    build_manifest(store, provenance__intervention_hash="0" * 16)
    recs, rep, _ = _read_all(_adapter(tmp_path))
    assert _steered_runs(_by_type(recs)) == [] and all("provenance.intervention_hash" in x for x in rep.refusals) and len(rep.refusals) == 2


def test_manifest_without_a_64hex_sha_is_refused(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store)
    mf = build_manifest(store, sha=SHA_OBJ[:16])                  # a 16-hex prefix is not the object identity
    recs, rep, _ = _read_all(_adapter(tmp_path))
    assert _steered_runs(_by_type(recs)) == [] and len(rep.refusals) == 2
    assert all(mf in x and "no 64-hex provenance.artifact_sha256" in x for x in rep.refusals)
    build_manifest(store, sha=None)
    recs, rep, _ = _read_all(_adapter(tmp_path))
    assert _steered_runs(_by_type(recs)) == [] and len(rep.refusals) == 2


def test_two_candidate_manifests_under_the_glob_fallback_are_refused(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store, wave="wave-not-on-disk-0000")
    a = build_manifest(store, wave="synthwave-a")
    b = build_manifest(store, wave="synthwave-b", sha=SHA_OTHER)
    ad = _adapter(tmp_path)
    recs, rep, inputs = _read_all(ad)
    assert _steered_runs(_by_type(recs)) == [] and len(rep.refusals) == 2
    for x in rep.refusals:
        assert "resolves to 2 manifests" in x and os.path.relpath(a, store) in x and os.path.relpath(b, store) in x
    assert len(inputs) == 1                                      # neither candidate is read, so neither is registered
    assert ad.last_discovery[os.path.basename(inputs[0].uri)]["unresolved"][0]["how"] == "ambiguous"


def test_a_manifest_not_among_the_discovered_inputs_is_not_read(tmp_path):
    """read() consults only watermarked bytes — a manifest that appears between discover and read is refused, not read."""
    store = str(tmp_path / "store")
    pq = build_parquet(str(tmp_path / "an"), store)
    ad = _adapter(tmp_path)
    inputs = ad.discover()
    assert len(inputs) == 1
    build_manifest(store)                                        # lands AFTER discovery
    recs = []
    for batch in ad.read(AF.primed_ctx(ad, now=NOW, producer=PRODUCER), ad.comparison_group(inputs[0]), inputs):
        recs.extend(batch.records)
    assert _steered_runs(_by_type(recs)) == [] and len(ad.last_report.refusals) == 2
    assert all("was not among the discovered inputs" in x for x in ad.last_report.refusals)
    # and a fresh discovery picks it up
    recs, rep, inputs = _read_all(ad)
    assert rep.refusals == [] and len(inputs) == 2 and all(r["object_id"] == SHA_OBJ for r in _steered_runs(_by_type(recs)))


def test_comparison_group_refuses_an_unbound_manifest_input(tmp_path):
    store = str(tmp_path / "store")
    pq = build_parquet(str(tmp_path / "an"), store)
    mf = build_manifest(store)
    ad = _adapter(tmp_path)
    from sisteer.registry.adapters.base import check_input
    with pytest.raises(AdapterRefusal, match="not discovered by this adapter instance"):
        ad.comparison_group(check_input(mf, role="report"))
    ad.discover()
    group = ad.comparison_group(check_input(pq, role="report"))
    assert group.endswith("/paper_rows_bcb_bfcl_200_synth") and ad.comparison_group(check_input(mf, role="report")) == group


def test_unsafe_store_reference_is_refused_not_joined(tmp_path):
    ad = _adapter(tmp_path)
    assert ad.resolve_manifest(WAVE, "../" + CELL, STORE_KEY)[1] == "unsafe"
    assert ad.resolve_manifest("..", CELL, STORE_KEY)[1] == "unsafe"
    assert ad.resolve_manifest(WAVE, CELL, STORE_KEY)[1] == "missing"


# --------------------------------------------------------------------------------------------
# other behaviour unchanged
# --------------------------------------------------------------------------------------------

def test_unmeasured_rows_without_a_sha_stay_dispositions(tmp_path):
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an"), store, steered_status="pending: synthetic re-run queued")
    recs, rep, inputs = _read_all(_adapter(tmp_path))
    bt = _by_type(recs)
    assert rep.refusals == [] and len(inputs) == 1
    assert len(bt["disposition"]) == 2 and all(d["state"] == "pending" and d["object_id"] is None for d in bt["disposition"])
    assert len(bt["run"]) == 2                                   # the anchors


def test_snapshot_and_live_file_share_one_manifest_and_one_group(tmp_path):
    store = str(tmp_path / "store")
    live = build_parquet(str(tmp_path / "an"), store, "paper_rows_bcb_bfcl_200_synth.parquet")
    snap = build_parquet(str(tmp_path / "an"), store, "paper_rows_bcb_bfcl_200_synth.before_D1.parquet", steered_value=0.50)
    mf = build_manifest(store)
    ad = _adapter(tmp_path)
    recs, rep, inputs = _read_all(ad)
    bt = _by_type(recs)
    assert rep.refusals == []
    assert [i.uri for i in inputs].count(mf) == 1 and len(inputs) == 3
    assert len({ad.comparison_group(i) for i in inputs}) == 1
    runs = _steered_runs(bt)
    assert len(runs) == 4 and all(r["object_id"] == SHA_OBJ for r in runs)
    sup = {s["run_id"] for s in bt["run_state"] if s["state"] == "superseded"}
    assert len(sup) == 4
    snap_runs = [r for r in runs if r["run_id"] in sup]
    live_ids = {r["run_id"] for r in runs if r["run_id"] not in sup}
    assert len(snap_runs) == 2 and all(r["superseded_by_run_id"] in live_ids for r in snap_runs)
    assert all(r["extra"]["wave_file"].count(".") == 2 for r in snap_runs)
    facts = ad.last_facts[ad.comparison_group(inputs[0])]
    assert facts[os.path.basename(live)]["n_object_from_manifest"] == 2 and facts[os.path.basename(snap)]["n_object_from_manifest"] == 2


def test_one_manifest_named_by_two_groups_refuses_discovery(tmp_path):
    """An input belongs to one comparison group: two live files in different spec dirs naming the same store manifest is refused
    loudly at discovery rather than watermarked under one of them."""
    store = str(tmp_path / "store")
    build_parquet(str(tmp_path / "an" / "spec-a" / "analysis"), store, "paper_rows_a.parquet")
    build_parquet(str(tmp_path / "an" / "spec-b" / "analysis"), store, "paper_rows_b.parquet")
    mf = build_manifest(store)
    ad = BB.BcbBfclAdapter(pattern=os.path.join(str(tmp_path / "an"), "spec-*", "analysis", "paper_rows_*.parquet"), store_root=store)
    with pytest.raises(AdapterRefusal, match="two comparison groups") as ei:
        ad.discover()
    assert mf in str(ei.value)


def test_unreadable_parquet_refuses_its_group_by_name_and_discovery_goes_on(tmp_path):
    store = str(tmp_path / "store")
    good = build_parquet(str(tmp_path / "an"), store)
    build_manifest(store)
    bad = os.path.join(str(tmp_path / "an"), "paper_rows_bcb_bfcl_200_broken.parquet")
    with open(bad, "wb") as fh:
        fh.write(b"not a parquet file")
    ad = _adapter(tmp_path)
    inputs = ad.discover()                                       # the broken file is still an input (its bytes are watermarked); the manifest of the good one is found
    assert {os.path.basename(i.uri) for i in inputs} == {os.path.basename(good), os.path.basename(bad), "manifest.json"}
    assert "could not be read" in ad.last_discovery[os.path.basename(bad)]["error"]
    with pytest.raises(AdapterRefusal, match="could not be read") as ei:
        list(ad.read(_ctx(), ad.comparison_group([i for i in inputs if i.uri == bad][0]), [i for i in inputs if i.uri == bad]))
    assert bad in str(ei.value)
    good_inputs = [i for i in inputs if i.uri != bad]
    recs = [r for b in ad.read(AF.primed_ctx(ad, now=NOW, producer=PRODUCER, groups={ad.comparison_group(good_inputs[0]): good_inputs}),
                               ad.comparison_group(good_inputs[0]), good_inputs) for r in b.records]
    assert ad.last_report.refusals == [] and all(r["object_id"] == SHA_OBJ for r in _steered_runs(_by_type(recs)))


def test_long_rows_alone_still_builds_the_paper_columns(tmp_path):
    """The parity lever calls long_rows(path) with no registered set: the manifest is resolved from disk and the paper columns
    are what the pinned reader would build (the resolved sha lands in object_sha256, as apply_producer_identity would file it)."""
    store = str(tmp_path / "store")
    pq = build_parquet(str(tmp_path / "an"), store)
    build_manifest(store)
    rows, facts = _adapter(tmp_path).long_rows(pq)
    assert facts["n_object_from_manifest"] == 2 and facts["manifest_refused"] == []
    steered = [r for r in rows if r["cell_id"] == CELL]
    assert all(r["object_sha256"] == SHA_OBJ and r["_object_id"] == SHA_OBJ and "_manifest_refusal" not in r for r in steered)
    frame = DC.rows_frame(rows)
    assert list(frame.columns) == list(DC.LONG_COLUMNS) and len(frame) == 4


# --------------------------------------------------------------------------------------------
# a null cell in the frame is not a label: None in the disposition's extra, never a NaN (the 2026-09-17 live hold)
# --------------------------------------------------------------------------------------------

def test_an_unkeyed_arm_row_files_its_disposition_with_a_null_cell_id_never_a_nan(tmp_path):
    """The widen wave's rows for an arm whose object was never staged under a campaign cell id have a NULL ``cell_id``; pandas reads
    the null string cell as a float NaN and the emitter copied it into the disposition's ``extra`` — a non-finite float the validator
    refuses by field path, holding the WHOLE revision on a formatting artefact. The proper sentinel for "no cell id" is None."""
    import pandas as pd
    store = str(tmp_path / "store")
    pq = build_parquet(str(tmp_path / "an"), store, unkeyed_arm_status="queued: no staged object under a cell id, pin or bank entry (synthetic)")
    build_manifest(store)
    raw = pd.read_parquet(pq)
    assert len(raw) == 6 and raw["cell_id"].isna().sum() == 2 and (raw[raw["cell_id"].isna()]["arm"] == "alphasteer_as_fit").all()
    recs, rep, inputs = _read_all(_adapter(tmp_path))             # every record validates: nothing non-finite reaches a payload
    bt = _by_type(recs)
    assert rep.refusals == [] and len(bt["run"]) == 4 and len(bt["run_contrast"]) == 2 and rep.dispositions == 2
    disp = bt["disposition"]
    assert len(disp) == 2 and {d["eval_id"] for d in disp} == {"bigcodebench", "bfcl"}          # the registry's eval ids (seeds)
    for d in disp:
        assert d["state"] == "held" and d["object_id"] is None and d["reason"].startswith("queued:") and d["model_id"] == "gemma4e4b_it"
        assert d["extra"]["campaign_cell_id_legacy"] is None                  # no cell id: None — not NaN, and not the string 'NaN'
        assert d["extra"]["arm_legacy"] == "alphasteer_as_fit" and d["extra"]["view_legacy"] == "gemma4e4b"
        assert d["extra"]["lane_source_legacy"] == "bcb_bfcl_200" and d["extra"]["strength_nominal"] == 1.0
        assert d["source_ref"] == "roster wave synth entry alphasteer_as_fit"
        assert R.first_nonfinite(d) is None
    # the parity frame still spells the null cell the paper's way (NaN): the fix is at the emitter, not in the reader's rows
    rows, _ = _adapter(tmp_path).long_rows(pq)
    assert sum(1 for r in rows if isinstance(r["cell_id"], float)) == 2 and len(DC.rows_frame(rows)) == 6


def test_disposition_extra_labels_are_text_or_none_never_a_frame_nan():
    """The emitter's own guard, independent of how pandas spells a null string cell: every legacy label copied into a disposition's
    ``extra`` is text or None, and a NaN provenance / note is no reference — the adapter's name stands in, not the string 'nan'."""
    nan = float("nan")
    row = DC.blank_row(source="bcb_bfcl_200", model="gemma4e4b", view="gemma4e4b", dose=1.0, cell_id=nan, arm="alphasteer_as_fit", component="capability",
                       eval_id="bigcodebench_200", status="queued: synthetic unmeasured row", note=nan, provenance=nan, value=nan)
    batch = RecordBatch("bcb_bfcl:synth", "synth")
    rep = DC.emit_rows(batch, _ctx(), [row], adapter="bcb_bfcl", campaign="sdfbase_full22", reader="load.load_coding",
                       seed_provenance="synthetic", code_sha="unverified")
    (rec,) = [r for r in batch.records if r["record_type"] == "disposition"]
    d = rec["payload"]
    assert rep.dispositions == 1 and rep.refusals == []
    assert d["extra"] == {"view_legacy": "gemma4e4b", "campaign_cell_id_legacy": None, "arm_legacy": "alphasteer_as_fit",
                          "lane_source_legacy": "bcb_bfcl_200", "strength_nominal": 1.0, "note": None}
    assert d["source_ref"] == "bcb_bfcl" and d["state"] == "held"
    assert isinstance(VAL.validate_record(rec, "<test>"), VAL.Validated)
    assert DC.text_or_none(nan) is None and DC.text_or_none(None) is None and DC.text_or_none("") == "" and DC.text_or_none("x") == "x"
    assert DC.text_or_none(3) == "3"
