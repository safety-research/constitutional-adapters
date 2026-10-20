"""The store recorder files the request's OBSERVED sampler (decode-recording lane, the registry owner's ruling of 2026-10-13): per ROUTE,
``StoreRecorder.run_payload`` reads ``decode_evidence.from_store_manifest`` for the EVAL the route names -- ``request_samplers[<spelling>]``
first, ``resolved_config.gen`` only for a sampled over-refusal control -- and the completion patch (``on_results``) reads the standup's
request recorder when the manifest filed nothing. Over a real ``CellStore``; the helpers are ``test_registry_emit_store``'s."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registry_emit_store import CHASH, IHASH, KEY, RESOLVED, _identity, _records_in, _row  # noqa: E402
from test_registry_adapter_cellstore_recorder_decode import CARD, RESOLVED as RESOLVED6, _recorder  # noqa: E402

from sisteer.capeval import store as cs  # noqa: E402
from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import decode_evidence as DE  # noqa: E402
from sisteer.registry import emit, hooks  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402

ARM = RESOLVED["arm_name"]                               # the store's object candidate (spec.arm_name): the routed keys are `<arm>|<spelling>`
SIX = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
SIX_F = {k: (int(v) if k == "top_k" else float(v)) for k, v in SIX.items()}
GEN_SAMPLED = dict(SIX, arm="vendor_recommended", batch_size=8, do_sample=True, max_new_tokens=2048, seed_base=20260909, trial=1,
                   escalated_from={"cap": 512, "root": "/x"}, floor_sample={"k": 200, "rule": "r"})
GEN_GREEDY = {"batch_size": 8, "do_sample": False, "max_new_tokens": 512}
CONSTANTS = {k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}


def _spec(controls, gen, request_samplers=None):
    spec = dict(RESOLVED, controls=list(controls), resolved_config={"driver": "served_capability_run_v2", "engine": "vllm", "gen": dict(gen)})
    if request_samplers is not None:
        spec[DE.MANIFEST_KEY_REQUEST_SAMPLERS] = request_samplers
    return spec


def _idents(evals):
    """One registered run per (object, producer spelling): the pending file's shape (`<object>|<task>` keys), the registry's eval ids."""
    reg = {"xstest": "xstest", "or_bench_hard_1k": "or_bench_hard", "mmlu": "mmlu", "gsm8k": "gsm8k"}
    return {f"{ARM}|{sp}": _identity(eval_id=reg[sp], item_set_id=f"{reg[sp]}_set") for sp in evals}


def _runs(spool):
    return {r["payload"]["eval_id"]: r["payload"] for _, r, _ in _records_in(spool) if r["record_type"] == "run"}


def _completion_patches(spool):
    return {r["payload"]["run_id"]: r["payload"]["fields"] for _, r, _ in _records_in(spool)
            if r["record_type"] == "run_patch" and "store_uri" in (r["payload"].get("fields") or {})}


def _validate_all(spool):
    for p, rec, sha in _records_in(spool):
        v = VAL.validate_record(rec, str(p), sha)
        assert isinstance(v, VAL.Validated), (p, getattr(v, "reason", v))


PRODUCER_SHA = synth.code_sha("a")       # the producing checkout's HEAD as the store recorder's producer block states it (emit.git_sha)


def _drive(spool, spec, idents, rows, store_root, monkeypatch, *, results=True, code_sha=PRODUCER_SHA):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)   # the suite runs with emission off; these tests write to their own tmp spool
    monkeypatch.delenv(hooks.EMIT_ENV, raising=False)
    monkeypatch.setattr(emit, "git_sha", lambda cwd=None: code_sha)   # host-independent: the producer block's code sha is the run's pin
    rec = hooks.store_recorder(KEY, spec, idents, spool=str(spool), registry_emit=True)
    assert rec is not None
    store = cs.CellStore(store_root, KEY, row_constants=CONSTANTS, recorder=rec)
    store.write_cell_manifest(spec)
    for task in rows:
        store.append_row(_row("q1", task=task))
    if results:
        store.write_results({})
    return rec, store


def test_a_sampled_controls_manifest_files_the_six_keys_per_routed_control_and_the_completion_patch_is_unchanged(tmp_path, monkeypatch):
    spool = tmp_path / "spool"
    spec = _spec(["xstest", "or_bench_hard_1k"], GEN_SAMPLED)
    rec, store = _drive(spool, spec, _idents(["xstest", "or_bench_hard_1k"]), ["xstest", "or_bench_hard_1k"], tmp_path / "store", monkeypatch)
    runs = _runs(spool)
    assert set(runs) == {"xstest", "or_bench_hard"}
    for ev_id in ("xstest", "or_bench_hard"):
        p = runs[ev_id]
        assert p["decode_sent_json"] == SIX_F and (p["decode_temperature_sent"], p["decode_top_p_sent"], p["decode_top_k_sent"]) == (0.7, 0.8, 20)
        assert "decode_resolved_json" not in p                                                  # never the manifest's gen; the recorder document only
        fc = p["field_confidence_json"]
        for col in ("decode_sent_json", "decode_temperature_sent", "decode_top_p_sent", "decode_top_k_sent"):
            assert fc[col]["confidence"] == "recorded" and fc[col]["source"].startswith(f"{store.cell_dir}/manifest.json#resolved_config.gen; sha256 "), fc[col]
        dev = p["extra"]["decode_evidence"]
        assert dev["family"] == "controls" and dev["reader"] == DE.READER and dev["sha256"] == DE.sha256_file(str(store.cell_dir / "manifest.json"))
        assert dev["unmodelled_keys"] == {"resolved_config.gen": ["arm", "batch_size", "do_sample", "escalated_from", "floor_sample", "max_new_tokens", "seed_base", "trial"]}
        assert dev["witnesses"]["spelling"] == ("xstest" if ev_id == "xstest" else "or_bench_hard_1k") and dev["witnesses"]["escalated_from_cap"] == 512
    # the manifest-time read filed the columns: the completion patch carries the store facts alone (byte-identical to before this lane)
    for fields in _completion_patches(spool).values():
        assert set(fields) == {"n_items_generated", "store_completion_artifact_uri", "store_uri", "store_completion_sha256"}, fields
    _validate_all(spool)
    assert rec._decode_filed == {f"{ARM}|xstest": True, f"{ARM}|or_bench_hard_1k": True}


def test_the_mixed_cell_files_mmlus_constant_from_request_samplers_and_the_pairs_six_keys_never_the_vendor_block_on_mmlu(tmp_path, monkeypatch):
    """P2 of the plan's review: one manifest serving mmlu beside the sampled pair carries the vendor block under `gen` while mmlu's request
    sends temperature 0.0 -- the evidence is selected by the route's spelling, so mmlu reads `request_samplers.mmlu`, never `gen`."""
    spool = tmp_path / "spool"
    rs = {"mmlu": {"temperature": 0.0}, "xstest": dict(SIX), "or_bench_hard_1k": dict(SIX)}
    spec = _spec(["mmlu", "xstest", "or_bench_hard_1k"], GEN_SAMPLED, request_samplers=rs)
    _drive(spool, spec, _idents(["mmlu", "xstest", "or_bench_hard_1k"]), ["mmlu", "xstest", "or_bench_hard_1k"], tmp_path / "store", monkeypatch)
    runs = _runs(spool)
    assert set(runs) == {"mmlu", "xstest", "or_bench_hard"}
    m = runs["mmlu"]
    assert m["decode_sent_json"] == {"temperature": 0.0} and m["decode_temperature_sent"] == 0.0 and "decode_top_k_sent" not in m
    assert m["field_confidence_json"]["decode_sent_json"]["source"].split("; ")[0].endswith("manifest.json#request_samplers.mmlu")
    for ev_id in ("xstest", "or_bench_hard"):
        assert runs[ev_id]["decode_sent_json"] == SIX_F and runs[ev_id]["field_confidence_json"]["decode_sent_json"]["source"].split("; ")[0].endswith(
            f"manifest.json#request_samplers.{'xstest' if ev_id == 'xstest' else 'or_bench_hard_1k'}")
    _validate_all(spool)
    # the same cell on a LEGACY manifest (no request_samplers): the controls still carry the six keys, mmlu's run carries NOTHING but the reason
    spool2 = tmp_path / "spool2"
    _drive(spool2, _spec(["mmlu", "xstest", "or_bench_hard_1k"], GEN_SAMPLED), _idents(["mmlu", "xstest", "or_bench_hard_1k"]), ["mmlu", "xstest", "or_bench_hard_1k"], tmp_path / "store2", monkeypatch)
    runs2 = _runs(spool2)
    assert runs2["xstest"]["decode_sent_json"] == SIX_F and runs2["or_bench_hard"]["decode_sent_json"] == SIX_F
    m2 = runs2["mmlu"]
    assert "decode_sent_json" not in m2 and "decode_temperature_sent" not in m2 and "decode_evidence" not in m2["extra"]
    fc = m2["field_confidence_json"]["decode_sent_json"]
    assert fc["confidence"] == "unverified" and "built by the driver" in fc["reason"] and "request_samplers" in fc["reason"]
    _validate_all(spool2)


def test_one_identity_over_several_controls_is_unverified_by_name_and_a_contradiction_never_raises_into_the_store(tmp_path, monkeypatch):
    # `*` over two controls: which request this run sent is not stated
    rec = hooks.StoreRecorder(KEY, _spec(["xstest", "or_bench_hard_1k"], GEN_SAMPLED), {"*": _identity(eval_id="xstest")}, spool=str(tmp_path / "s1"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store1")
    p = rec.run_payload("*")
    assert "decode_sent_json" not in p and p["field_confidence_json"]["decode_sent_json"] == {"confidence": "unverified", "reason": "one identity, 2 controls (xstest, or_bench_hard_1k): which request this run sent is not stated"}
    assert rec._decode_filed == {"*": False}
    # `*` over ONE control: the control's own request
    rec = hooks.StoreRecorder(KEY, _spec(["xstest"], GEN_SAMPLED), {"*": _identity(eval_id="xstest")}, spool=str(tmp_path / "s2"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store2")
    assert rec.run_payload("*")["decode_sent_json"] == SIX_F
    # a driver disagreeing with itself (request_samplers.xstest != gen): the payload is built WITHOUT decode columns, the entry names the contradiction
    rec = hooks.StoreRecorder(KEY, _spec(["xstest"], GEN_SAMPLED, request_samplers={"xstest": dict(SIX, temperature=0.9)}), {"*": _identity(eval_id="xstest")},
                              spool=str(tmp_path / "s3"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store3")
    p = rec.run_payload("*")
    assert "decode_sent_json" not in p and "decode_evidence" not in p["extra"]
    fc = p["field_confidence_json"]["decode_sent_json"]
    assert fc["confidence"] == "unverified" and fc["reason"].startswith("decode evidence refused (DecodeEvidenceContradiction)") and "stated twice and differently" in fc["reason"]
    record = R.make_record("run", p, rec.producer, "11111111-2222-3333-4444-555555555555", synth.t(5))
    assert isinstance(VAL.validate_record(record, "<memory>", R.record_sha(record)), VAL.Validated)
    # a harness store that is not the served driver's (the flat synthetic spec): unverified, never an error, every other field as before
    rec = hooks.StoreRecorder(KEY, RESOLVED, {"*": _identity()}, spool=str(tmp_path / "s4"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store4")
    p = rec.run_payload("*")
    assert "decode_sent_json" not in p and "carries no resolved_config.gen" not in p["field_confidence_json"]["decode_sent_json"]["reason"]
    assert "names no controls list" in p["field_confidence_json"]["decode_sent_json"]["reason"]
    assert p["intervention_hash"] == IHASH and p["harness_config_hash"] == CHASH


def test_a_thinking_card_cell_files_nothing_at_the_manifest_and_the_completion_patch_carries_the_recorders_sent_and_resolved(tmp_path, monkeypatch, capsys):
    """(a'): a gsm8k / thinking_card cell applies its sampler server-side (`gen` carries no sampler key): the `run` record says why nothing is
    filed; the standup's request recorder lands at `<store root>/resolved_sampler.json` before the results, so the completion patch carries
    the 5-key `--expect` reference as sent (27b) and the 6-key resolved sampler, each with its entry; folded, the row reads them. The
    reference is a DECLARATION (sent_rows 0), so its entry is `derived` against the pinned request builder at the producer's code sha
    (the owner's Change 1, Major 1) while the resolved sampler stays `recorded`."""
    root = tmp_path / "reg"; H.ensure_layout(str(root)); spool = root / "records"
    R.write_records(os.path.join(str(root), "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    store_root = tmp_path / "store"
    spec = _spec(["gsm8k"], GEN_GREEDY, request_samplers={})
    rec, store = _drive(spool, spec, {"*": _identity(eval_id=synth.EVAL_ID)}, [synth.EVAL_ID], store_root, monkeypatch, results=False)
    run = next(iter(_runs(spool).values()))
    assert "decode_sent_json" not in run and "gsm8k request's sampler is built by the driver" in run["field_confidence_json"]["decode_sent_json"]["reason"]
    assert rec._decode_filed == {"*": False}
    # the recorder document lands (the 27b fixture's shape: the store root is two levels above the cell dir)
    _recorder(str(store_root), "AGREES")
    assert os.path.isfile(DE.recorder_path_for(str(store.cell_dir)))
    store.write_results({})
    fields = next(iter(_completion_patches(spool).values()))
    assert fields["decode_sent_json"] == {"temperature": 1.0, "top_k": 64, "top_p": 0.95} and fields["decode_resolved_json"] == RESOLVED6
    assert (fields["decode_temperature_sent"], fields["decode_top_p_sent"], fields["decode_top_k_sent"]) == (1.0, 0.95, 64)
    fc = fields["field_confidence_json"]
    for col in ("decode_sent_json", "decode_temperature_sent", "decode_top_p_sent", "decode_top_k_sent"):
        assert fc[col]["confidence"] == "derived" and "--expect reference" in fc[col]["source"] and "AGREES" in fc[col]["source"], (col, fc[col])
        assert DE.RECORDER_REQUEST_BUILDER in fc[col]["source"] and f"code sha {PRODUCER_SHA}" in fc[col]["source"] and "sent_rows 0" in fc[col]["source"]
    assert fc["decode_resolved_json"]["confidence"] == "recorded" and "distinct_resolved[0]" in fc["decode_resolved_json"]["source"]
    assert fields["extra_json"]["decode_evidence"]["family"] == "recorder" and rec._decode_filed == {"*": True}
    assert fields["extra_json"]["decode_evidence"]["sent_confidence"] == "derived" and fields["extra_json"]["decode_evidence"]["witnesses"]["request_builder_code_sha"] == PRODUCER_SHA
    _validate_all(spool)
    # the fold: the row carries the recorder's columns and the patch's per-column entries replace the manifest's unverified entry
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(str(root)), readonly=True)
    try:
        r = conn.execute("SELECT decode_sent_json, decode_resolved_json, decode_temperature_sent, decode_top_k_sent, field_confidence_json, extra_json, decode_policy_match FROM run").fetchone()
        assert json.loads(r[0]) == {"temperature": 1.0, "top_k": 64, "top_p": 0.95} and json.loads(r[1]) == RESOLVED6 and (r[2], r[3]) == (1.0, 64)
        fcj = json.loads(r[4])
        assert fcj["decode_sent_json"]["confidence"] == "derived" and fcj["decode_resolved_json"]["confidence"] == "recorded"
        assert json.loads(r[5])["decode_evidence"]["family"] == "recorder"
        assert r[6] == "no_match"                     # a sent set with no policy row in scope: derived, never `unrecorded`
    finally:
        conn.close()
    # without a recorder document the completion patch is byte-identical to before this lane
    spool_b = tmp_path / "spool_b"
    _drive(spool_b, spec, {"*": _identity(eval_id=synth.EVAL_ID)}, [synth.EVAL_ID], tmp_path / "store_b", monkeypatch)
    assert set(next(iter(_completion_patches(spool_b).values()))) == {"n_items_generated", "store_completion_artifact_uri", "store_uri", "store_completion_sha256"}
    # a producer block whose code sha is `unverified` (git unavailable in the producing checkout) pins no builder: the declared reference
    # fills NO sent column (the run's own code_sha reads `unverified` beside it) -- but the server-resolved sampler the recorder OBSERVED is
    # filed apart, `recorded`; the sent entry carries the recorder's reason (the missing pin) in place of the manifest-time one; the declared
    # keys ride the evidence block as a witness; and the notice says so (before, the completion patch dropped the observation with the sent
    # set and the durable row kept the manifest-time reason -- the cellstore adapter kept both)
    root_c = tmp_path / "reg_c"; H.ensure_layout(str(root_c)); spool_c = root_c / "records"
    R.write_records(os.path.join(str(root_c), "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    store_root_c = tmp_path / "store_c"
    rec_c, store_c = _drive(spool_c, spec, {"*": _identity(eval_id=synth.EVAL_ID)}, [synth.EVAL_ID], store_root_c, monkeypatch, results=False, code_sha="unverified")
    _recorder(str(store_root_c), "AGREES"); store_c.write_results({})
    patch_c = next(iter(_completion_patches(spool_c).values()))
    assert set(patch_c) == {"n_items_generated", "store_completion_artifact_uri", "store_uri", "store_completion_sha256", "decode_resolved_json", "field_confidence_json", "extra_json"}
    assert patch_c["decode_resolved_json"] == RESOLVED6
    for col in ("decode_sent_json", "decode_temperature_sent", "decode_top_p_sent", "decode_top_k_sent"):
        assert col not in patch_c, col
    fcc = patch_c["field_confidence_json"]
    assert set(fcc) == {"decode_sent_json", "decode_resolved_json"}
    assert fcc["decode_resolved_json"]["confidence"] == "recorded" and "distinct_resolved[0]" in fcc["decode_resolved_json"]["source"] and "verdict AGREES" in fcc["decode_resolved_json"]["source"]
    assert fcc["decode_sent_json"]["confidence"] == "unverified", fcc["decode_sent_json"]
    why_c = fcc["decode_sent_json"]["reason"]
    assert "DECLARATION" in why_c and "sent_rows 0" in why_c and "names no code sha ('unverified')" in why_c and "nothing filed for the sent columns" in why_c, why_c
    xb = patch_c["extra_json"]["decode_evidence"]
    assert xb["family"] == "recorder" and xb["sent_confidence"] is None and xb["derivation"] is None and xb["evidence_path"] == DE.recorder_path_for(str(store_c.cell_dir))
    assert xb["witnesses"]["unfiled_declared_reference"] == {"sent": {"temperature": 1.0, "top_k": 64, "top_p": 0.95}, "reference_source": "--expect", "sent_rows": 0, "request_builder": DE.RECORDER_REQUEST_BUILDER}
    assert rec_c._decode_filed == {"*": False} and next(iter(_runs(spool_c).values()))["code_sha"] == "unverified"
    err = capsys.readouterr().err
    assert "fills no sent column" in err and "names no code sha" in err and "server-resolved sampler (recorded)" in err
    _validate_all(spool_c)
    # the fold: the row keeps `{}` / NULL on the sent side and reads `unrecorded`, carries the six resolved values `recorded`, and its sent
    # entry is the recorder's reason (the missing pin), never the manifest-time one
    rep_c = H.ingest_once(str(root_c), workers=0, min_free=0)
    assert rep_c.refused == 0, rep_c.as_dict()
    conn = DB.connect(DB.db_path(str(root_c)), readonly=True)
    try:
        r = conn.execute("SELECT decode_sent_json, decode_resolved_json, decode_temperature_sent, decode_top_p_sent, decode_top_k_sent, field_confidence_json, extra_json, decode_policy_match, code_sha FROM run").fetchone()
    finally:
        conn.close()
    assert (json.loads(r[0]), json.loads(r[1]), r[2], r[3], r[4]) == ({}, RESOLVED6, None, None, None)
    fcj = json.loads(r[5])
    assert fcj["decode_resolved_json"]["confidence"] == "recorded" and fcj["decode_sent_json"]["confidence"] == "unverified"
    assert "names no code sha" in fcj["decode_sent_json"]["reason"] and "built by the driver" not in fcj["decode_sent_json"]["reason"], fcj["decode_sent_json"]
    assert json.loads(r[6])["decode_evidence"]["witnesses"]["unfiled_declared_reference"]["sent"] == {"temperature": 1.0, "top_k": 64, "top_p": 0.95}
    assert (r[7], r[8]) == ("unrecorded", "unverified")


def test_the_store_stays_byte_identical_with_the_recorder_reading_decode_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(cs.time, "time", lambda: 1755000000.0)
    spec = _spec(["xstest"], GEN_SAMPLED, request_samplers={"xstest": dict(SIX)})
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    off = cs.CellStore(tmp_path / "off", KEY, row_constants=CONSTANTS, recorder=hooks.store_recorder(KEY, spec, {"*": _identity(eval_id="xstest")}, spool=str(tmp_path / "no")))
    off.write_cell_manifest(spec); off.append_row(_row("q1", task="xstest")); off.write_results({})
    monkeypatch.delenv(emit.DISABLED_ENV)
    rec = hooks.store_recorder(KEY, spec, {"*": _identity(eval_id="xstest")}, spool=str(tmp_path / "spool"), registry_emit=True)
    on = cs.CellStore(tmp_path / "on", KEY, row_constants=CONSTANTS, recorder=rec)
    on.write_cell_manifest(spec); on.append_row(_row("q1", task="xstest")); on.write_results({})
    for name in ("manifest.json", "rows.jsonl", "results.json"):
        assert (tmp_path / "off" / "cells" / KEY / name).read_bytes() == (tmp_path / "on" / "cells" / KEY / name).read_bytes(), name
    assert next(iter(_runs(tmp_path / "spool").values()))["decode_sent_json"] == SIX_F


def test_a_mixed_atom_and_legacy_control_cell_fills_only_the_atoms_completion_patch_from_the_recorder(tmp_path, monkeypatch):
    """Review round 1, Major 1: `--controls gsm8k,xstest,mmlu --greedy` on a LEGACY manifest (no request_samplers) leaves every route
    unfiled at manifest time; the store-level recorder document describes the ATOM's requests, so the completion patch may fill gsm8k's run
    alone -- xstest (a greedy request the driver builds) and mmlu (the scoring request) keep the four store facts and their manifest-time
    reasons, never the recorder's sampler. Before the guard, every unfiled route read the same document."""
    spool, store_root = tmp_path / "spool", tmp_path / "store"
    rec, store = _drive(spool, _spec(["gsm8k", "xstest", "mmlu"], GEN_GREEDY), _idents(["gsm8k", "xstest", "mmlu"]), ["gsm8k", "xstest", "mmlu"], store_root, monkeypatch, results=False)
    routes = {sp: f"{ARM}|{sp}" for sp in ("gsm8k", "xstest", "mmlu")}
    assert rec._decode_filed == {r: False for r in routes.values()}
    assert rec._decode_recorder_eligible == {routes["gsm8k"]: True, routes["xstest"]: False, routes["mmlu"]: False}
    assert rec._decode_spelling == {routes["gsm8k"]: "gsm8k", routes["xstest"]: "xstest", routes["mmlu"]: "mmlu"}
    _recorder(str(store_root), "AGREES")
    store.write_results({})
    runs = _runs(spool); rid = {ev: p["run_id"] for ev, p in runs.items()}; patches = _completion_patches(spool)
    g = patches[rid["gsm8k"]]
    assert g["decode_sent_json"] == {"temperature": 1.0, "top_k": 64, "top_p": 0.95} and g["decode_resolved_json"] == RESOLVED6
    assert g["extra_json"]["decode_evidence"]["family"] == "recorder" and g["extra_json"]["decode_evidence"]["witnesses"]["task"] == "gsm8k"
    assert g["field_confidence_json"]["decode_sent_json"]["confidence"] == "derived" and g["field_confidence_json"]["decode_resolved_json"]["confidence"] == "recorded"
    for ev_id, phrase in (("xstest", "--greedy opt-in"), ("mmlu", "built by the driver")):
        assert set(patches[rid[ev_id]]) == {"n_items_generated", "store_completion_artifact_uri", "store_uri", "store_completion_sha256"}, ev_id
        assert "decode_sent_json" not in runs[ev_id] and phrase in runs[ev_id]["field_confidence_json"]["decode_sent_json"]["reason"]
    assert rec._decode_filed == {routes["gsm8k"]: True, routes["xstest"]: False, routes["mmlu"]: False}
    _validate_all(spool)
    # the same cell on the NEW manifest: xstest's greedy constant is the MANIFEST's (Major 3) at manifest time, gsm8k's the recorder's at completion
    spool2, root2 = tmp_path / "spool2", tmp_path / "store2"
    rec2, store2 = _drive(spool2, _spec(["gsm8k", "xstest"], GEN_GREEDY, request_samplers={"xstest": {"temperature": 0.0}}), _idents(["gsm8k", "xstest"]), ["gsm8k", "xstest"], root2, monkeypatch, results=False)
    runs2 = _runs(spool2)
    assert runs2["xstest"]["decode_sent_json"] == {"temperature": 0.0} and "decode_sent_json" not in runs2["gsm8k"]
    assert rec2._decode_recorder_eligible == {routes["gsm8k"]: True, routes["xstest"]: False}
    _recorder(str(root2), "AGREES"); store2.write_results({})
    p2 = _completion_patches(spool2); rid2 = {ev: p["run_id"] for ev, p in runs2.items()}
    assert p2[rid2["gsm8k"]]["decode_sent_json"] == {"temperature": 1.0, "top_k": 64, "top_p": 0.95} and "decode_sent_json" not in p2[rid2["xstest"]]
    # `*` over several controls names no request: the recorder fills nothing, whatever it says
    spool3, root3 = tmp_path / "spool3", tmp_path / "store3"
    rec3, store3 = _drive(spool3, _spec(["gsm8k", "xstest"], GEN_GREEDY), {"*": _identity(eval_id=synth.EVAL_ID)}, ["gsm8k", "xstest"], root3, monkeypatch, results=False)
    assert rec3._decode_recorder_eligible == {"*": False} and rec3._decode_spelling == {"*": None}
    _recorder(str(root3), "AGREES"); store3.write_results({})
    assert "decode_sent_json" not in next(iter(_completion_patches(spool3).values())) and rec3._decode_filed == {"*": False}
    # a PER-TASK recorder document (--expect-tasks) covering xstest alone fills nothing for an eligible gsm8k route: not covered, by name
    spool4, root4 = tmp_path / "spool4", tmp_path / "store4"
    rec4, store4 = _drive(spool4, _spec(["gsm8k"], GEN_GREEDY, request_samplers={}), {"*": _identity(eval_id=synth.EVAL_ID)}, [synth.EVAL_ID], root4, monkeypatch, results=False)
    assert rec4._decode_recorder_eligible == {"*": True} and rec4._decode_spelling == {"*": "gsm8k"}
    tb = {"caps": [512], "n": 9, "verdict": "AGREES", "reference": {"temperature": 0.0}, "reference_source": "greedy", "distinct_resolved": [{"sampler": dict(RESOLVED6, temperature=0.0), "n": 9}]}
    doc = {"verdict": "AGREES", "sent": {"n_rows": 0, "distinct": []}, "classes": {"client": {"n": 9, "verdict": "AGREES", "tasks": {"xstest": tb}, "reference": None,
                                                                                            "reference_source": "--expect-tasks: one reference per task (classes.client.tasks)",
                                                                                            "distinct_resolved": [{"sampler": dict(RESOLVED6, temperature=0.0), "n": 9}]}}}
    os.makedirs(os.path.dirname(DE.recorder_path_for(str(store4.cell_dir))), exist_ok=True)
    with open(DE.recorder_path_for(str(store4.cell_dir)), "w") as fh:
        json.dump(doc, fh)
    store4.write_results({})
    assert "decode_sent_json" not in next(iter(_completion_patches(spool4).values())) and rec4._decode_filed == {"*": False}
    _validate_all(spool4)


def test_a_greedy_routed_controls_cell_files_the_greedy_constant_from_the_manifest_and_the_row_reads_it(tmp_path, monkeypatch):
    """Review round 1, Major 3: under `--greedy` the driver's judged-control body sends temperature 0.0, the manifest states
    GREEDY_CHAT_SAMPLER per control, and the run record carries it from the manifest (never the recorder); folded through the real
    harvester, the row reads {temperature: 0.0} / 0.0 with a DERIVED match -- never `unrecorded`."""
    spool = tmp_path / "spool"
    rs = {"xstest": {"temperature": 0.0}, "or_bench_hard_1k": {"temperature": 0.0}}
    rec, store = _drive(spool, _spec(["xstest", "or_bench_hard_1k"], GEN_GREEDY, request_samplers=rs), _idents(["xstest", "or_bench_hard_1k"]), ["xstest", "or_bench_hard_1k"], tmp_path / "store", monkeypatch)
    runs = _runs(spool)
    for ev_id, sp in (("xstest", "xstest"), ("or_bench_hard", "or_bench_hard_1k")):
        p = runs[ev_id]
        assert p["decode_sent_json"] == {"temperature": 0.0} and p["decode_temperature_sent"] == 0.0 and "decode_top_p_sent" not in p and "decode_top_k_sent" not in p
        fc = p["field_confidence_json"]
        assert fc["decode_sent_json"]["confidence"] == fc["decode_temperature_sent"]["confidence"] == "recorded"
        assert fc["decode_sent_json"]["source"].split("; ")[0] == f"{store.cell_dir}/manifest.json#request_samplers.{sp}"
        assert p["extra"]["decode_evidence"]["witnesses"]["gen_do_sample"] is False and p["extra"]["decode_evidence"]["unmodelled_keys"] == {}
    for fields in _completion_patches(spool).values():
        assert set(fields) == {"n_items_generated", "store_completion_artifact_uri", "store_uri", "store_completion_sha256"}
    assert rec._decode_filed == {f"{ARM}|xstest": True, f"{ARM}|or_bench_hard_1k": True}
    assert rec._decode_recorder_eligible == {f"{ARM}|xstest": False, f"{ARM}|or_bench_hard_1k": False}
    _validate_all(spool)
    # the fold: one greedy control under the synthetic eval, through the real harvester, to the row
    root = tmp_path / "reg"; H.ensure_layout(str(root)); spool_r = root / "records"
    R.write_records(os.path.join(str(root), "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    _drive(spool_r, _spec(["xstest"], GEN_GREEDY, request_samplers={"xstest": {"temperature": 0.0}}), {"*": _identity(eval_id=synth.EVAL_ID)}, ["xstest"], tmp_path / "store_r", monkeypatch)
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(str(root)), readonly=True)
    try:
        r = conn.execute("SELECT decode_sent_json, decode_temperature_sent, decode_top_p_sent, decode_top_k_sent, field_confidence_json, decode_policy_match FROM run").fetchone()
    finally:
        conn.close()
    assert (json.loads(r[0]), r[1], r[2], r[3]) == ({"temperature": 0.0}, 0.0, None, None)
    assert json.loads(r[4])["decode_sent_json"]["confidence"] == "recorded" and r[5] != "unrecorded"
