"""The r2 export's Arditi alpha-0.05 operating-point objects, and the held-revision naming defect (2026-09-18).

Three findings behind "r2 held on object 87ffbc4c…", each pinned here:

1. IDENTITY. The export names an Arditi served baseline by the Petri VIEW_MANIFEST's ``object_sha256``; the ladder owner
   (``arditi_strength``) mints the object row from the serving record beside the rung, else from the pins file. Both are the
   sha256 of the staged ``object.pt`` bytes, so the two readers agree on the id without either guessing: resolved run object
   == minted object == sha256(artifact).
2. DELIVERY. ``registry adapters run --reread`` of an ACCEPTED head drops the read's records by design
   (``runner.reread_committed``), so a corrected ``_objects`` could never deliver an object the head lacks. The pins file the
   read consults is now a discovered input: its sha is in the revision id, so the corrected read publishes a NEW revision.
3. THE QUEUE. A revision whose ``run`` names a missing object was held with the marker naming a DEPENDENT — ``run <id>``, a
   parent inside the revision that can never land on its own, so ``_held_still_blocked`` skipped every re-attempt — and no
   ``run`` row in ``blocked_records`` (the transaction that blocked them rolled back; only the first-sorting constituent
   was re-blocked). Known-bad first: the marker must name the ROOT (the object), every blocked constituent must be in the
   queue under its own record type, and the revision must activate when the object arrives.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import identity as I
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import arditi_strength as AR
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN

import _b1_fixtures as B1  # noqa: E402
import registry_declared_fixtures as F  # noqa: E402
from test_registry_adapter_declared import _by_type, _read_all  # noqa: E402
from test_registry_core_fixes import StoreAdapter, _ingest, _pass, _q, _root, _store, _write  # noqa: E402

ARTIFACT = b"synthetic Arditi refusal direction, alpha 0.05, stored layer 19 -> served block 18"
ARTIFACT_SHA = hashlib.sha256(ARTIFACT).hexdigest()


def _pins(path: str, sha: str, **over) -> str:
    block = {"bank": "arditi_strength", "source": "arditi_strength/gemma4e4b/strength/cells_view/ar_r0p05/arditi", "view": "ar_r0p05",
             "staged_id": "ar_r0p05__arditi", "object_sha256": sha, "kind": "static_r", "alpha": 0.05, "dose_unit": "residual_norm_fraction",
             "phase": "prefill", "served_layer": 18, "fit_layer_stored": 19, "identity_sidecar": "fit_provenance.json",
             "selection_outcome": "ruled", "producer": "scripts/arditi/make_arditi_objects.py"}
    block.update(over)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_schema": "misalign_baseline_pins/1", "gemma4e4b": {"arditi": block}}, fh)
    return path


def test_owner_and_export_agree_on_the_operating_point_object_id(tmp_path):
    """One artefact, one id: the staged ``object.pt`` bytes hash to X; the ladder's serving record and the pins name X; the Petri
    VIEW_MANIFEST the export resolves through names X. The minted object row IS X (artifact_sha256 == object_id) and the export's
    resolver files the run against X — no placeholder, no second spelling."""
    fx = F.build_arditi(str(tmp_path / "arditi"))
    staged = tmp_path / "staged" / "ar_r0p05__arditi"
    staged.mkdir(parents=True)
    (staged / "object.pt").write_bytes(ARTIFACT)
    assert hashlib.sha256((staged / "object.pt").read_bytes()).hexdigest() == ARTIFACT_SHA
    # the owner's store: the serving record beside the operating rung names the artefact's hash
    served = os.path.join(fx["root"], "gemma4e4b_ar_r0p05_t0", "serving_synthetic", "cells_served.json")
    with open(served, "w", encoding="utf-8") as fh:
        json.dump({"bank_objects": {"arditi": {"object_sha256": ARTIFACT_SHA}}}, fh)
    pins = _pins(str(tmp_path / "pins.json"), ARTIFACT_SHA)
    ad = AR.ArditiStrengthAdapter(root=fx["root"], pins_path=pins)
    recs, rep = _read_all(ad)
    assert rep.refusals == []
    objs = {o["object_id"]: o for o in _by_type(recs)["object"]}
    assert ARTIFACT_SHA in objs, sorted(k[:16] for k in objs)
    o = objs[ARTIFACT_SHA]
    assert o["artifact_sha256"] == ARTIFACT_SHA and o["object_kind"] == "static_r_arditi" and o["base_scale"] == 0.05
    assert o["fit_layers_json"] == [18] and o["extra"]["pinned"] is True and o["extra"]["staged_id"] == "ar_r0p05__arditi"
    assert not [v for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    # the export's side: the Petri arm VIEW_MANIFEST names the same staged artefact
    root = tmp_path / "petri_manifests"
    (root / "gemma4e4b" / "arditi").mkdir(parents=True)
    (root / "gemma4e4b" / "arditi" / "VIEW_MANIFEST.json").write_text(json.dumps({
        "wave": "arditi_strength", "target": "gemma4e4b", "view": "ar_r0p05",
        "entries": [{"cell_id": "arditi", "staged_id": "ar_r0p05__arditi", "source": str(staged), "object_sha256": ARTIFACT_SHA,
                     "identity": "fit_provenance"}]}))
    res = LC.ObjectResolver(B1.synthetic_seeds(), objects_root=str(tmp_path / "objects"), petri_manifests_root=str(root))
    resolved = res.resolve("gemma4e4b_it", "arditi")
    assert resolved.object_id == ARTIFACT_SHA == o["object_id"] and resolved.staged_id == "ar_r0p05__arditi"
    assert resolved.object_kind == o["object_kind"] == "static_r_arditi" and resolved.confidence == "recorded"


def test_pins_file_is_a_discovered_input_so_a_corrected_object_set_is_a_new_revision(tmp_path):
    """The read consults the pins file for a served object's identity, so it is an input: discovered, hashed, and part
    of the revision id. A --reread of an accepted head drops the read's records, so THIS is the lever that publishes objects an
    earlier revision lacked: the pins' bytes change -> the revision id changes -> a fresh revision carries the objects."""
    fx = F.build_arditi(str(tmp_path / "arditi"))
    pins = _pins(str(tmp_path / "pins.json"), F.SHA_A)
    ad = AR.ArditiStrengthAdapter(root=fx["root"], pins_path=pins)
    inputs = ad.discover()
    pin_inputs = [i for i in inputs if i.uri == pins]
    assert len(pin_inputs) == 1 and pin_inputs[0].role == "report" and pin_inputs[0].sha256 == hashlib.sha256(open(pins, "rb").read()).hexdigest()
    assert all(ad.comparison_group(i) == "battery" for i in inputs)
    sid = RUN.source_id_for(ad.name, "battery")
    rev_a = RUN.revision_id_for(sid, inputs)
    _pins(pins, F.SHA_A, served_layer=17)                    # the pin's bytes change: a different revision, not a re-read
    rev_b = RUN.revision_id_for(sid, ad.discover())
    assert rev_a != rev_b
    # the default (pins_path=None) registers the checkout's pins file, the one `_pinned_shas` reads
    default = AR.ArditiStrengthAdapter(root=fx["root"], pins_path=None)
    assert any(i.uri == AR.PINS_PATH for i in default.discover())


def _tagged(root, rev):
    return _q(root, "SELECT record_type, missing_parent_table, missing_parent_id FROM blocked_records "
                    "WHERE json_extract(extra_json, '$.revision_id') = ? ORDER BY record_type, missing_parent_table", rev)


def test_held_revision_names_the_root_object_and_queues_every_blocked_constituent(tmp_path):
    """Known-bad (r2_export f7858cf8, live, 2026-09-18): a revision's `run` names an object no row carries. Before: the marker
    said `activation pending: run <id> (…run_metric.jsonl#N)` — the run_metric sorted first, its parent is the run staged in
    the same revision — and `blocked_records` held that ONE run_metric plus the marker; the run records were in no queue and
    the recorded parent could never arrive, so `_held_still_blocked` skipped the revision on every cycle. After: the marker
    names the object and the run record; the run, its judgment, run_state and run_metric are all queued under their own
    record types; and when the object lands the revision activates on the ordinary re-evaluation."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions(with_fitted_object=False))   # the lora object is NOT a row
    _ingest(root)
    assert _q(root, "SELECT COUNT(*) FROM object WHERE object_id = ?", synth.OBJECT_SHA) == [(0,)]
    store_root = str(tmp_path / "stores")
    os.makedirs(store_root)
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    ad = StoreAdapter(store_root)
    rep, ing = _pass(root, ad, synth.t(2000))
    rev = rep.results[0].revision_id
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert ing.revisions_held >= 1            # held by the activation, and skipped once more by the same cycle's re-evaluation
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    refusal = _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev)[0][0]
    # the ROOT: the object nobody emitted, named on the run record — not `run <id>` on a dependent
    assert refusal.startswith(f"activation pending: object {synth.OBJECT_SHA} ("), refusal
    assert "-run.jsonl#" in refusal and f"run {rid}" not in refusal, refusal
    tagged = _tagged(root, rev)
    assert ("run", "object", synth.OBJECT_SHA) in tagged, tagged
    assert ("source_revision", "object", synth.OBJECT_SHA) in tagged, tagged
    # the dependents the fixed-point pass could not apply are queued too, each on the parent IT is missing
    dependents = {(t, p) for t, p, pid in tagged if pid == rid}
    assert {("run_metric", "run"), ("judgment", "run")} <= dependents, tagged
    assert _q(root, "SELECT COUNT(*) FROM run WHERE run_id = ?", rid) == [(0,)]
    # a cycle with nothing new: still held on the same root, nothing lost, nothing duplicated
    ing2 = _ingest(root)
    assert ing2.revisions_held >= 1 or ing2.ok == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records WHERE record_type = 'run' AND missing_parent_id = ?", synth.OBJECT_SHA) == [(1,)]
    # the object arrives (the owning adapter's row): the recorded parent is present, the revision re-attempts and activates
    _write(root, "objs", "000001-object.jsonl", [synth.rec("object", synth.object_lora(), synth.t(2100))])
    ing3 = _ingest(root)
    assert ing3.unblocked >= 1 or ing3.ok >= 1
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(1,)]
    assert _q(root, "SELECT object_id FROM run WHERE run_id = ?", rid) == [(synth.OBJECT_SHA,)]
    assert _q(root, "SELECT COUNT(*) FROM run_metric WHERE run_id = ?", rid) == [(1,)]
    assert _tagged(root, rev) == []
    assert _q(root, "SELECT COUNT(*) FROM blocked_records WHERE missing_parent_id IN (?, ?)", synth.OBJECT_SHA, rid) == [(0,)]


def test_a_root_run_holds_its_dependents_without_masking_them(tmp_path):
    """The other shape of the same defect: when the root IS a run (a `run_metric` whose run nobody emitted at all), the marker
    keeps naming that run, and the constituent named is the run_metric — the revision's own dependents never masquerade as
    the missing parent."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _ingest(root)
    from sisteer.registry import records as R
    ad_name, rev = "orphan_metric", "f" * 16
    rev_dir = os.path.join(root, "records", "adapters", ad_name, rev)
    rid = I.run_id(synth.CAMPAIGN, "nobody", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    metric = synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": 1, "n": 2,
                                      "computed_by": "registry", "source_uri": "synthetic://orphan"}, synth.t(3000))
    R.write_records(os.path.join(rev_dir, R.record_file_name(1, "run_metric")), [metric])
    sid = RUN.source_id_for(ad_name, "g")
    marker = synth.rec("source_revision", {"revision_id": rev, "source_id": sid, "adapter": ad_name, "observed_utc": synth.t(3001),
                                           "inputs_json": {"synthetic://orphan": "0" * 64}, "accepted": 1, "priority": 0}, synth.t(3001))
    R.write_records(os.path.join(rev_dir, R.record_file_name(2, "source_revision")), [marker])
    ing = _ingest(root)
    assert ing.revisions_held >= 1
    refusal = _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev)[0][0]
    assert refusal.startswith(f"activation pending: run {rid} (") and "-run_metric.jsonl#1" in refusal, refusal
    assert _tagged(root, rev) == [("run_metric", "run", rid), ("source_revision", "run", rid)]
