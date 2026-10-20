"""objects_bank: a synthetic bank tree → one `object` row per distinct artifact (twins merge into locations),
superseded / withdrawn status from the bank layout, cluster locations from the staging records, refusal on
a symlinked input, the incremental pass + ingest end to end, `scan()` counts and the `objects scan` CLI.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import fit_sidecars as FS
from sisteer.registry.adapters import objects_bank as OB
from sisteer.registry.adapters import runner as RUN
from registry_fit_fixtures import CORPUS_SHA, PARENT_A, PARENT_B, REV, _blob, _sha, _write, corpus_tbm, seeds, si_tbm

T = synth.t


def _bank(tmp_path):
    """A full22-layout bank with: a base fit, the SAME bytes staged twice (a twin), a dose copy, an it-twin under
    `_superseded_by_it64/` with its `it64` successor, a withdrawn copy, `_records/` naming cluster paths, a pins file."""
    root = str(tmp_path / "full22_objects")
    cells = os.path.join(root, "synthmodel", "cells")
    base = os.path.join(cells, "base__synth-corpus-full__lora_r4_ce")
    sha = _sha(_blob(base, b"base weights"))
    _write(base, "train_budget_meta.json", corpus_tbm(sha))
    _write(base, "staged_from.json", {"source": "twosi-fits:out/synth/cells_all/synth-corpus-full__lora_r4_ce", "object_sha256": sha})
    # twin: identical bytes under another staged id (a bank listing, never a second object)
    twin = os.path.join(cells, "itdoc__synth-corpus-full__lora_r4_ce")
    _blob(twin, b"base weights")
    _write(twin, "train_budget_meta.json", corpus_tbm(sha))
    # dose copy
    d05 = os.path.join(cells, "d0p5__synth-corpus-full__lora_r4_ce")
    dsha = _sha(_blob(d05, b"half weights"))
    tbm = corpus_tbm(dsha)
    tbm["dose_scaled_from"] = {"scale": 0.5, "parent_object_sha256": sha, "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "5" * 40,
                               "written_at": "2026-09-15T16:17:45Z", "parent_dir": base}
    _write(d05, "train_budget_meta.json", tbm)
    _write(d05, "object.pt.provenance.json", {"kind": "lora_all", "layers": [3, 7, 11, 15], "dose_scale": 0.5, "parent_object_sha256": sha})
    # it64 successor + the superseded it twin
    it64 = os.path.join(cells, "d0p5it64__synth-corpus-full__lora_r4_ce")
    it64_sha = _sha(_blob(it64, b"it64 weights"))
    tbm64 = corpus_tbm(it64_sha, model_id="synthetic/instruct-4b")
    tbm64["dose_scaled_from"] = {"scale": 0.5, "parent_object_sha256": PARENT_A, "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "5" * 40,
                                 "written_at": "2026-09-13T12:00:00Z", "parent_dir": "/home/synthetic/x"}
    _write(it64, "train_budget_meta.json", tbm64)
    _write(it64, "object.pt.provenance.json", {"kind": "lora_all", "layers": [3, 7, 11, 15], "dose_scale": 0.5, "parent_object_sha256": PARENT_A})
    old = os.path.join(cells, "_superseded_by_it64", "d0p5it__synth-corpus-full__lora_r4_ce")
    old_sha = _sha(_blob(old, b"old it weights"))
    tbm_old = corpus_tbm(old_sha, model_id="synthetic/instruct-4b")
    tbm_old["dose_scaled_from"] = {"scale": 0.5, "parent_object_sha256": PARENT_B, "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "5" * 40,
                                   "written_at": "2026-09-13T11:00:00Z", "parent_dir": "/home/synthetic/y"}
    _write(old, "train_budget_meta.json", tbm_old)
    _write(old, "object.pt.provenance.json", {"kind": "lora_all", "layers": [3, 7, 11, 15], "dose_scale": 0.5, "parent_object_sha256": PARENT_B})
    _write(os.path.join(cells, "_superseded_by_it64"), "README.txt", "Superseded by the it64 ids.")
    # withdrawn
    wd = os.path.join(root, "_withdrawn", "synthmodel__d1it__synth-corpus-full__lora_r4_ce.withdrawn-2026")
    wsha = _sha(_blob(wd, b"withdrawn weights"))
    tbm_w = corpus_tbm(wsha, model_id="synthetic/instruct-4b")
    tbm_w["dose_scaled_from"] = {"scale": 1.0, "parent_object_sha256": PARENT_A, "producer": "build_strength.py", "producer_commit": "5" * 40,
                                 "written_at": "2026-09-13T09:00:00Z", "parent_dir": "/home/synthetic/z"}
    _write(wd, "train_budget_meta.json", tbm_w)
    _write(wd, "object.pt.provenance.json", {"kind": "lora_all", "layers": [3, 7, 11, 15], "dose_scale": 1.0, "parent_object_sha256": PARENT_A})
    _write(wd, "WITHDRAWN.txt", "WITHDRAWN by the coordinator.")
    # staging records + pins
    _write(os.path.join(root, "_records"), "synthmodel__base__synth-corpus-full__lora_r4_ce.json",
           {"target": "synthmodel", "staged_id": "base__synth-corpus-full__lora_r4_ce", "object_sha256": sha,
            "bytes": sum(os.path.getsize(os.path.join(base, n)) for n in os.listdir(base)),      # the staged DIRECTORY's total, as stage_objects.py sums it
            "volume_path": "twosi-fits:out/synth/cells_all/synth-corpus-full__lora_r4_ce",
            "cluster_a_path": "/shared/synthetic/exp/objects/sdfbase/full22/synthmodel/cells/base__synth-corpus-full__lora_r4_ce",
            "cluster_a_verified": True, "cluster_a_remote_sha256": sha,
            "cluster_b_path": "/workspace/synthetic/exp/objects/sdfbase/full22/synthmodel/cells/base__synth-corpus-full__lora_r4_ce",
            "cluster_b_verified": False, "cluster_b_remote_sha256": None, "pulled_at": "2026-09-07T00:10:24Z"})
    pins = _write(str(tmp_path), "pins.json", {"synthmodel": {"base__synth-corpus-full__lora_r4_ce": {"object_sha256": sha, "source": "twosi-fits:out/synth/cells_all/x",
                                                                                                    "verified_cluster_a": True, "verified_cluster_b": True}}})
    return root, pins, {"base": sha, "dose": dsha, "it64": it64_sha, "old": old_sha, "withdrawn": wsha, "base_dir": base, "twin_dir": twin}


def _adapter(root, pins):
    return OB.ObjectsBankAdapter([("full22_objects", root, "full22")], seeds=seeds(), read_blobs=False, pins_path=pins)


def test_enumerate_and_group_by_artifact_sha(tmp_path):
    root, pins, shas = _bank(tmp_path)
    ad = _adapter(root, pins)
    dirs = OB.enumerate_object_dirs(ad.banks)
    kinds = {os.path.basename(d.path): d.status_hint for d in dirs}
    assert kinds["d0p5it__synth-corpus-full__lora_r4_ce"] == "superseded_twin"
    assert kinds["synthmodel__d1it__synth-corpus-full__lora_r4_ce.withdrawn-2026"] == "withdrawn"
    assert next(d for d in dirs if d.status_hint == "superseded_twin").twin_staged_id == "d0p5it64__synth-corpus-full__lora_r4_ce"
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), set()).add(os.path.dirname(i.uri))
    assert groups[shas["base"]] == {shas["base_dir"], shas["twin_dir"]}      # identical bytes → one group
    assert len(groups) == 5
    roles = {os.path.basename(i.uri): i.role for i in inputs}
    assert roles["object.pt"] == "object_weights" and roles["train_budget_meta.json"] == "report"


def test_twins_merge_into_one_object_with_every_location(tmp_path):
    root, pins, shas = _bank(tmp_path)
    ad = _adapter(root, pins)
    ad.discover()
    rep = OB.scan(ad.banks, seeds=seeds(), read_blobs=False, pins_path=pins, now=T(9000))
    assert rep.objects_total == 5 and not rep.refusals
    by_id = {p["object_id"]: p for p in rep.payloads}
    base = by_id[shas["base"]]
    uris = [l["uri"] for l in base["artifact_locations_json"]]
    assert os.path.join(shas["base_dir"], "object.pt") in uris and os.path.join(shas["twin_dir"], "object.pt") in uris
    assert "twosi-fits:out/synth/cells_all/synth-corpus-full__lora_r4_ce" in uris
    bulba = next(l for l in base["artifact_locations_json"] if l["uri"].startswith("cluster_a:"))
    # the location's bytes are the OBJECT's (12 B of weights, the size measured here of the sha-identical copy), the record's directory total rides beside
    assert bulba["verified_utc"] == "2026-09-07T00:10:24Z" and bulba["bytes"] == 12 == base["artifact_bytes"] and bulba["staged_dir_bytes"] > 12
    assert bulba["bytes_source"] == OB.STAGED_LOCATION_BYTES_SOURCE
    charm = next(l for l in base["artifact_locations_json"] if l["uri"].startswith("cluster_b:"))
    assert charm["verified_utc"] is None and charm["bytes"] is None and "bytes_source" not in charm      # the record says not verified there: no size claimed
    assert any(l.get("source", "").startswith("pins.json") for l in base["artifact_locations_json"])
    assert sorted(base["extra"]["staged_ids"]) == ["base__synth-corpus-full__lora_r4_ce", "itdoc__synth-corpus-full__lora_r4_ce"]
    # the bank states a status only where it carries the evidence: the base object's "approved" is NOT stated (the producer's folded
    # status stands; the fold keeps it on omission, the harvester blocks a brand-new object on its own row until the producer states it)
    assert "status" not in base and "status" not in base["field_confidence_json"] and "D721" in base["extra"]["status_unstated"]
    old = by_id[shas["old"]]
    assert old["status"] == "superseded" and old["superseded_by_object_id"] == shas["it64"]
    assert by_id[shas["withdrawn"]]["status"] == "withdrawn"
    dose = by_id[shas["dose"]]
    assert dose["fit_target_kind"] == "derived" and dose["parent_object_ids_json"] == [shas["base"]]
    # EMISSION_VERSION 10: a dose copy's fit_effective_batch is its PARENT's, derived from ONE source -- the parent's
    # record copied beside the copy (the copy's own train_budget_meta.json, effective_batch 8), whether or not the parent's own record is
    # staged in a bank here (the base fit is; PARENT_A, the it64 copy's parent, is not): the same entry either way
    for copy, parent in ((dose, shas["base"]), (by_id[shas["it64"]], PARENT_A)):
        assert copy["fit_effective_batch"] == 8 and copy["fit_hyperparameters_json"]["effective_batch"] == 8
        assert copy["field_confidence_json"]["fit_effective_batch"] == {"confidence": "derived", "source": FS.dose_batch_derived_source(parent)}
        assert copy["extra"]["dose_parent_fit_effective_batch"] == {"parent_object_sha256": parent, "unresolved": None,
                                                                   "copied_record": {"effective_batch": 8, "source": "train_budget_meta.json:effective_batch (the PARENT's fit record copied beside the copy)"}}
    bc = rep.banks["full22_objects"]
    assert bc.dirs == 6 and bc.objects == 5 and bc.blobs_local == 6 and bc.blobs_read == 0 and bc.with_sidecars == 6
    assert bc.by_status == {"superseded": 1, "withdrawn": 1} and bc.status_unstated == 3
    text = OB.format_scan(rep)
    assert "objects 5" in text and "superseded=1" in text and "unstated=3" in text


def _producer_records(root, pins, statuses):
    """What the PRODUCER that staged a bank object filed for it before the bank is read: the object's own facts -- the same sidecars the
    bank mirrors, so the per-column fold mixes nothing -- with the status the producer STATES (the bank omits it), and no
    `superseded_by_object_id` (that pointer is the bank's, from superseded_by.json)."""
    rep = OB.scan([("full22_objects", root, "full22")], seeds=seeds(), read_blobs=False, pins_path=pins, now=T(10))
    by_id = {p["object_id"]: p for p in rep.payloads}
    out = []
    for sha, status in statuses:
        p = dict(by_id[sha], status=status)
        p.pop("superseded_by_object_id", None)
        fc = {k: v for k, v in p["field_confidence_json"].items() if k != "superseded_by_object_id"}
        fc["status"] = {"confidence": "recorded", "source": "the producer's statement"}
        p["field_confidence_json"] = fc
        p["extra"] = {k: v for k, v in p["extra"].items() if k not in ("status_origin", "unstated_columns", "status_unstated")}   # a producer STATES its status
        out.append(p)
    return out


def _registry(tmp_path, prior_objects=()):
    """A scratch registry with the dimension rows the bank's parents need; ``prior_objects`` = object payloads a PRODUCER filed for
    bank objects before the pass (:func:`_producer_records`: the producers' `reference` already folded when the bank is read)."""
    reg = str(tmp_path / "reg")
    H.ensure_layout(reg)
    dims = synth.seed_dimensions(with_fitted_object=False)
    dims += [synth.rec("object", synth.object_lora(sha=PARENT_A), T(20)), synth.rec("object", synth.object_lora(sha=PARENT_B, paper_label="control_lora", role="control"), T(21))]
    dims += [synth.rec("object", p, T(22 + i)) for i, p in enumerate(prior_objects)]
    R.write_records(os.path.join(reg, "records", "dims", "000001-dims.jsonl"), dims)
    H.ingest_once(reg, workers=0, min_free=0)
    return reg


def _producer_statements(root, pins, reg, when, banks=None, seeds_obj=None):
    """The PRODUCER's object records for the bank objects the bank states no status for: the bank's own payload shape with the
    producer's `status`, stamped ``when`` -- what the fit pipeline files before the bank mirrors the object."""
    rep = OB.scan(banks if banks is not None else _adapter(root, pins).banks, seeds=seeds_obj if seeds_obj is not None else seeds(),
                  read_blobs=False, pins_path=pins, now=when)
    recs = [synth.rec("object", dict(p, status="approved"), when) for p in rep.payloads if "status" not in p]
    R.write_records(os.path.join(reg, "records", "producer", f"{int(when[14:16]) * 60 + int(when[17:19]):06d}-objects.jsonl"), recs)
    return [p["object_id"] for p in rep.payloads if "status" not in p]


def test_incremental_pass_then_ingest_yields_the_rows(tmp_path):
    root, pins, shas = _bank(tmp_path)
    reg = _registry(tmp_path)
    unstated = _producer_statements(root, pins, reg, T(8000))       # the producer's statements precede the bank's pass
    assert sorted(unstated) == sorted([shas["base"], shas["dose"], shas["it64"]])
    ad = _adapter(root, pins)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refreshed == 5 and rep.groups_refused == 0, [r.error for r in rep.results]
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        # the superseded it twin points at its it64 successor, so ingest order may block it one cycle
        if ing.blocked:
            conn.close()
            ing2 = H.ingest_once(reg, workers=0, min_free=0)
            conn = DB.connect(DB.db_path(reg), readonly=True)
        n = conn.execute("SELECT COUNT(*) FROM object WHERE campaign = 'sdfbase_full22'").fetchone()[0]
        assert n == 5
        assert tuple(conn.execute("SELECT status, superseded_by_object_id FROM object WHERE object_id = ?", (shas["old"],)).fetchone()) == ("superseded", shas["it64"])
        # the bank's omission kept the producer's status and its stamp; the bank's later record moved its own columns
        st = conn.execute("SELECT status, field_versions_json FROM object WHERE object_id = ?", (shas["base"],)).fetchone()
        assert st[0] == "approved" and json.loads(st[1])["status"][0] == T(8000) and json.loads(st[1])["artifact_locations_json"][0] == T(9000)
        # the twin's bytes are ONE artifact (artifact_id = sha): six weights files, five artifacts
        assert conn.execute("SELECT COUNT(*) FROM artifact WHERE kind = 'object_weights'").fetchone()[0] == 5
        n_unv = conn.execute("SELECT SUM(n_unverified_fields) FROM v_object_provenance WHERE campaign = 'sdfbase_full22'").fetchone()[0]
        assert n_unv > 0
        # a second pass over unchanged inputs writes nothing
    finally:
        conn.close()
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep2 = RUN.run_pass(reg, [ad], conn=conn, now=T(9500))
    conn.close()
    assert rep2.inputs_changed == 0 and rep2.records_written == 0


def test_a_bank_object_with_no_producer_statement_waits_on_its_own_row_until_the_producer_states_it(tmp_path):
    """End to end on the bank: with no producer record, the three objects the bank states no status for are dependency-blocked on
    their own row (never refused), the superseded twin behind its it64 successor; the withdrawn one (a stated status) folds. The producer's
    statements then release them all, and the bank's omission keeps the producer's status."""
    root, pins, shas = _bank(tmp_path)
    reg = _registry(tmp_path)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [_adapter(root, pins)], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 0
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rows = {r[0]: r[1] for r in conn.execute("SELECT object_id, status FROM object WHERE campaign = 'sdfbase_full22'")}
    blocked = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT record_uri, missing_parent_table, missing_parent_id FROM blocked_records WHERE record_type = 'object'")}
    conn.close()
    assert rows == {shas["withdrawn"]: "withdrawn"}
    assert sorted(blocked.values()) == sorted([("object", shas["base"]), ("object", shas["dose"]), ("object", shas["it64"]), ("object", shas["it64"])])
    _producer_statements(root, pins, reg, T(9500))                    # stamped AFTER the bank's records: the producer's columns win where both state
    ing2 = H.ingest_once(reg, workers=0, min_free=0)
    assert ing2.refused == 0 and ing2.unblocked >= 3
    conn = DB.connect(DB.db_path(reg), readonly=True)
    if conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0]:
        conn.close()
        H.ingest_once(reg, workers=0, min_free=0)
        conn = DB.connect(DB.db_path(reg), readonly=True)
    rows = {r[0]: r[1] for r in conn.execute("SELECT object_id, status FROM object WHERE campaign = 'sdfbase_full22'")}
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    conn.close()
    assert rows == {shas["base"]: "approved", shas["dose"]: "approved", shas["it64"]: "approved", shas["old"]: "superseded", shas["withdrawn"]: "withdrawn"}


def _stage_bank_with_out_of_scope_bundles(tmp_path):
    """A `stage`-layout bank (the 2026-08 objects) with two bundles ruled out of scope -- a synthetic control direction under
    `_controls/` and a LoFiT fit under `objects_a/` -- beside one fitted object the bank reads as today."""
    root = str(tmp_path / "stage_objects")
    ctl = os.path.join(root, "_controls", "synthetic_si__rand_recipeB_s31")
    csha = _sha(_blob(ctl, b"gaussian direction"))
    _write(ctl, "train_budget_meta.json", {"arm": "rand_recipeB_s31", "cell_id": "synthetic_si__rand_recipeB_s31", "layer": 15, "model_id": "synthetic/instruct-4b",
                                           "si_id": "synthetic_si", "positions": "all", "synthetic_control": True, "control_note": "NOT FITTED", "object_sha256": csha})
    lofit = os.path.join(root, "synthetic_si", "objects_a", "synthetic_si__lofit")
    lsha = _sha(_blob(lofit, b"lofit weights"))
    _write(lofit, "object.pt.provenance.json", {"kind": "lofit", "layers": [15], "object_sha256": lsha})
    fitted = os.path.join(root, "synthetic_si", "objects_a", "synthetic_si__psr_kl_all")
    fsha = _sha(_blob(fitted, b"relu weights"))
    _write(fitted, "train_budget_meta.json", si_tbm(fsha))
    return root, {"control": csha, "lofit": lsha, "fitted": fsha, "control_dir": ctl, "lofit_dir": lofit}


def test_out_of_scope_bundles_are_parked_by_name_not_refused(tmp_path):
    """The bank's read of a bundle whose kind is ruled out of scope is ACCEPTED with no object record -- a parked
    group, its `source_status` row `ok = 0` with the ruling under `extra.refuse` (the explanation) and `extra.parked` (the entry the
    status page lists) -- so the adapter is not refused and the window not aborted on a by-design condition; the fitted object beside it
    reads as before; `scan()` counts the parked bundles apart from refusals; an unchanged second pass writes nothing."""
    root, shas = _stage_bank_with_out_of_scope_bundles(tmp_path)
    banks = [("stage_objects", root, "stage")]
    rep0 = OB.scan(banks, seeds=seeds(), read_blobs=False, now=T(9000))
    assert rep0.objects_total == 1 and rep0.refusals == [] and len(rep0.parked) == 2
    assert {p["kind_legacy"] for p in rep0.parked} == {"rand_recipeB_s31", "lofit"} and all("D37" in p["parked"]["ruling"] for p in rep0.parked)
    bc = rep0.banks["stage_objects/_controls"], rep0.banks["stage_objects/synthetic_si/objects_a"]
    assert (bc[0].parked, bc[0].refused, bc[1].parked, bc[1].refused, bc[1].objects) == (1, 0, 1, 0, 1)
    assert "parked 1 (D37" in OB.format_scan(rep0) and "rand_recipeB_s31" in OB.format_scan(rep0)
    reg = _registry(tmp_path)
    ad = OB.ObjectsBankAdapter(banks, seeds=seeds(), read_blobs=False)
    _producer_statements(root, None, reg, T(8000), banks=banks)           # the producer states the fitted object's status first
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 3, [r.error for r in rep.results]
    by_group = {r.group: r for r in rep.results}
    assert by_group[shas["fitted"]].accepted and by_group[shas["control"]].accepted and by_group[shas["lofit"]].accepted
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id = ?", (shas["fitted"],)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id IN (?, ?)", (shas["control"], shas["lofit"])).fetchone()[0] == 0
        rows = {r[0]: r for r in conn.execute("SELECT source_id, ok, detail, extra_json FROM source_status WHERE adapter = 'objects_bank'")}
        parked = {sid: json.loads(r[3]) for sid, r in rows.items() if json.loads(r[3]).get("parked")}
        assert len(parked) == 2 and all(rows[sid][1] == 0 for sid in parked) and all(rows[sid][1] == 1 for sid in set(rows) - set(parked))
        for sid, extra in parked.items():
            ent = extra["parked"][0]
            assert ent["ruling"] == OB.PARKED_RULING and ent["status_ruled"] == "withdrawn" and ent["object_role_ruled"] == "other"
            assert ent["dir"] in (shas["control_dir"], shas["lofit_dir"]) and ent["kind_legacy"] in ("rand_recipeB_s31", "lofit")
            assert extra["refuse"] and "parked by name" in extra["refuse"][0] and "D37" in extra["refuse"][0]
            assert "parked by name" in rows[sid][2] and "EMITTED NOTHING" not in rows[sid][2]      # explained, never a silent read
            assert extra.get("emitted_nothing_unexplained") is None
        # the status page's "parked by name" section reads the same facts off the per-group view (the per-source-string aggregate
        # `v_source_status` folds the three groups into one row, each group's facts under `extra_json.by_source`, both `refuse` lines kept)
        v = conn.execute("SELECT COUNT(*) FROM v_source_status_current WHERE adapter = 'objects_bank' AND ok = 0 AND json_extract(extra_json, '$.parked') IS NOT NULL").fetchone()[0]
        assert v == 2
        agg = conn.execute("SELECT ok, json_array_length(json_extract(extra_json, '$.refuse')) FROM v_source_status WHERE adapter = 'objects_bank'").fetchall()
        assert [tuple(r) for r in agg] == [(0, 2)]
        # the parked groups' revisions are ACCEPTED (no standing refusal for a window or tick to re-read)
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'objects_bank' AND accepted = 0").fetchone()[0] == 0
    finally:
        conn.close()
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep2 = RUN.run_pass(reg, [ad], conn=conn, now=T(9500))
    conn.close()
    assert rep2.inputs_changed == 0 and rep2.records_written == 0 and rep2.groups_refused == 0


def test_a_ruled_out_name_with_an_unruled_declared_kind_is_refused_not_parked(tmp_path):
    """A `<si>__lofit` directory whose provenance declares a kind no ruling covers is a REFUSED
    group -- `scan` lists it under refusals, the pass refuses the group (no accepted revision, no `parked` status fact, the
    refusal naming the conflict) -- while the genuine lofit and the synthetic control beside it stay parked and the fitted object reads.
    Before the rule the bank accepted an empty revision for the rewritten contents and the status attributed them to the
    out-of-scope ruling."""
    root, shas = _stage_bank_with_out_of_scope_bundles(tmp_path)
    rewritten = os.path.join(root, "synthetic_si_c", "objects_c", "synthetic_si__lofit")
    rsha = _sha(_blob(rewritten, b"a new method's weights under the old name"))
    _write(rewritten, "object.pt.provenance.json", {"kind": "new_unruled_method", "layers": [15], "object_sha256": rsha})
    _write(rewritten, "train_budget_meta.json", si_tbm(rsha, arm="lofit", cell_id="synthetic_si__lofit"))
    banks = [("stage_objects", root, "stage")]
    rep0 = OB.scan(banks, seeds=seeds(), read_blobs=False, now=T(9000))
    assert rep0.objects_total == 1 and len(rep0.parked) == 2 and {p["kind_legacy"] for p in rep0.parked} == {"rand_recipeB_s31", "lofit"}
    assert len(rep0.refusals) == 1 and rewritten in str(rep0.refusals[0]) and "NOT parked" in str(rep0.refusals[0]) and "new_unruled_method" in str(rep0.refusals[0])
    bc = rep0.banks["stage_objects/synthetic_si_c/objects_c"]
    assert (bc.parked, bc.refused, bc.objects) == (0, 1, 0) and not bc.parked_dirs
    reg = _registry(tmp_path)
    ad = OB.ObjectsBankAdapter(banks, seeds=seeds(), read_blobs=False)
    _producer_statements(root, None, reg, T(8000), banks=banks)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    by_group = {r.group: r for r in rep.results}
    assert rep.groups_refused == 1 and rep.groups_refreshed == 3, [r.error for r in rep.results]
    assert not by_group[rsha].accepted and "NOT parked" in by_group[rsha].error and "new_unruled_method" in by_group[rsha].error
    assert by_group[shas["fitted"]].accepted and by_group[shas["control"]].accepted and by_group[shas["lofit"]].accepted
    assert ad.status_extra(rsha) == {} and set(ad.status_extra(shas["lofit"])) == {"refuse", "parked"}
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id = ?", (rsha,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'objects_bank' AND accepted = 1").fetchone()[0] == 3
        # no accepted (empty) revision for the rewritten group: a refusal, where the park used to mint one
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE source_id = ? AND accepted = 1", (RUN.source_id_for(ad.name, rsha),)).fetchone()[0] == 0
        rows = {r[0]: r for r in conn.execute("SELECT source_id, ok, detail, extra_json FROM source_status WHERE adapter = 'objects_bank'")}
        parked = {sid for sid, r in rows.items() if json.loads(r[3]).get("parked")}
        assert len(parked) == 2 and all("synthetic_si_c" not in json.loads(rows[sid][3])["parked"][0]["dir"] for sid in parked)
        assert all("new_unruled_method" not in (rows[sid][2] or "") for sid in parked)
        v = conn.execute("SELECT COUNT(*) FROM v_source_status_current WHERE adapter = 'objects_bank' AND json_extract(extra_json, '$.parked') IS NOT NULL").fetchone()[0]
        assert v == 2
    finally:
        conn.close()


def test_an_unruled_synthetic_control_is_a_refused_group_not_a_parked_one(tmp_path):
    """A `_controls/` bundle declaring `synthetic_control: true` under a spelling the scope table never named
    (`arm='other_method'`, a novel directory, a declared novel kind) is a REFUSED group -- `scan` lists it under
    refusals, the pass accepts no revision and writes no `parked` fact for it -- while the Gaussian control and the
    lofit beside it stay parked and the fitted object reads. Before the rule it parked under the Gaussian row's ruling
    (the lookup ended on an unconditional `diffmean_random`): an accepted empty revision, the bundle omitted silently
    under a ruling that did not name it."""
    root, shas = _stage_bank_with_out_of_scope_bundles(tmp_path)
    novel = os.path.join(root, "_controls", "synthetic_si__other_method")
    nsha = _sha(_blob(novel, b"some other control direction"))
    _write(novel, "train_budget_meta.json", {"arm": "other_method", "cell_id": "synthetic_si__other_method", "layer": 15, "model_id": "synthetic/instruct-4b",
                                             "si_id": "synthetic_si", "positions": "all", "synthetic_control": True, "control_note": "NOT A FITTED OBJECT", "object_sha256": nsha})
    _write(novel, "object.pt.provenance.json", {"kind": "other_method_vector", "layers": [15], "object_sha256": nsha})
    banks = [("stage_objects", root, "stage")]
    rep0 = OB.scan(banks, seeds=seeds(), read_blobs=False, now=T(9000))
    assert rep0.objects_total == 1 and len(rep0.parked) == 2 and {p["kind_legacy"] for p in rep0.parked} == {"rand_recipeB_s31", "lofit"}
    assert len(rep0.refusals) == 1 and novel in str(rep0.refusals[0]) and "NOT parked" in str(rep0.refusals[0]) and "'other_method'" in str(rep0.refusals[0])
    bc = rep0.banks["stage_objects/_controls"]
    assert (bc.parked, bc.refused, bc.objects) == (1, 1, 0) and bc.parked_dirs == [{"dir": shas["control_dir"], "kind_legacy": "rand_recipeB_s31"}]
    reg = _registry(tmp_path)
    ad = OB.ObjectsBankAdapter(banks, seeds=seeds(), read_blobs=False)
    _producer_statements(root, None, reg, T(8000), banks=banks)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    by_group = {r.group: r for r in rep.results}
    assert rep.groups_refused == 1 and rep.groups_refreshed == 3, [r.error for r in rep.results]
    assert not by_group[nsha].accepted and "NOT parked" in by_group[nsha].error and "'other_method'" in by_group[nsha].error
    assert by_group[shas["fitted"]].accepted and by_group[shas["control"]].accepted and by_group[shas["lofit"]].accepted
    assert ad.status_extra(nsha) == {} and set(ad.status_extra(shas["control"])) == {"refuse", "parked"}
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id = ?", (nsha,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'objects_bank' AND accepted = 1").fetchone()[0] == 3
        # no accepted (empty) revision for the novel control: a refusal, where the park used to mint one
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE source_id = ? AND accepted = 1", (RUN.source_id_for(ad.name, nsha),)).fetchone()[0] == 0
        rows = {r[0]: r for r in conn.execute("SELECT source_id, ok, detail, extra_json FROM source_status WHERE adapter = 'objects_bank'")}
        parked = {sid for sid, r in rows.items() if json.loads(r[3]).get("parked")}
        assert len(parked) == 2 and all(json.loads(rows[sid][3])["parked"][0]["dir"] != novel for sid in parked)
        assert all("other_method" not in (rows[sid][2] or "") for sid in parked)
        v = conn.execute("SELECT COUNT(*) FROM v_source_status_current WHERE adapter = 'objects_bank' AND json_extract(extra_json, '$.parked') IS NOT NULL").fetchone()[0]
        assert v == 2
    finally:
        conn.close()


def test_a_ruled_out_twin_beside_an_in_scope_twin_refuses_the_group_naming_both_instead_of_parking_it(tmp_path):
    """The same bytes staged under an in-scope fitted name and under a RICHER directory whose provenance declares a kind
    ruled out of scope. The richer, ruled-out bundle is the group's primary, and its ruling used to park the WHOLE group:
    two directories, no object row for the fitted twin, no refusal naming the conflict. Now every twin is classified first
    (`fit_sidecars.scope_verdict`) and twins that disagree make the group a REFUSAL naming each directory with its verdict
    -- `scan` lists it under refusals, the pass accepts no revision and writes no `parked` fact -- while the genuine lofit,
    the Gaussian control and the fitted object beside it read as before."""
    root, shas = _stage_bank_with_out_of_scope_bundles(tmp_path)
    fitted_twin = os.path.join(root, "synthetic_si", "objects_b", "synthetic_si__psr_kl_all")
    tsha = _sha(_blob(fitted_twin, b"shared weights"))
    _write(fitted_twin, "train_budget_meta.json", si_tbm(tsha))
    ruled_twin = os.path.join(root, "synthetic_si", "objects_b", "synthetic_si__lofit")
    assert _sha(_blob(ruled_twin, b"shared weights")) == tsha                                   # identical bytes: one artifact group
    _write(ruled_twin, "object.pt.provenance.json", {"kind": "lofit", "layers": [15], "object_sha256": tsha})
    _write(ruled_twin, "train_budget_meta.json", si_tbm(tsha, arm="lofit", cell_id="synthetic_si__lofit"))
    banks = [("stage_objects", root, "stage")]
    ad = OB.ObjectsBankAdapter(banks, seeds=seeds(), read_blobs=False)
    ad.discover()
    bundles = ad.bundles_for(sorted([fitted_twin, ruled_twin]))
    assert {b.dir: FS.scope_verdict(b, seeds()).state for b in bundles} == {fitted_twin: FS.SCOPE_IN_SCOPE, ruled_twin: FS.SCOPE_PARKED}
    # the ruled-out twin is the richer one (two sidecars to one): the primary, the condition under which the group used to park
    with pytest.raises(FS.SidecarRefusal) as ei:
        ad.payload_for_group(bundles, seeds(), T(9000))
    assert not isinstance(ei.value, FS.OutOfScopeObject) and ei.value.where == ruled_twin
    why = ei.value.why
    assert "disagree" in why and fitted_twin in why and ruled_twin in why and "'relu_steer'" in why and "parks as 'lofit'" in why
    assert "NOT parked" in why and "NOT read" in why and "D37" in why
    rep0 = OB.scan(banks, seeds=seeds(), read_blobs=False, now=T(9000))
    assert rep0.objects_total == 1 and len(rep0.parked) == 2 and {p["kind_legacy"] for p in rep0.parked} == {"rand_recipeB_s31", "lofit"}
    assert len(rep0.refusals) == 1 and rep0.refusals[0]["group"] == tsha and set(rep0.refusals[0]["dirs"].split(";")) == {fitted_twin, ruled_twin}
    assert "disagree" in rep0.refusals[0]["why"]
    bc = rep0.banks["stage_objects/synthetic_si/objects_b"]
    assert (bc.dirs, bc.parked, bc.refused, bc.objects) == (2, 0, 2, 0) and not bc.parked_dirs
    assert "parked 0 (D37" in OB.format_scan(rep0) and "refused 2" in OB.format_scan(rep0)
    reg = _registry(tmp_path)
    _producer_statements(root, None, reg, T(8000), banks=banks)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    by_group = {r.group: r for r in rep.results}
    assert rep.groups_refused == 1 and rep.groups_refreshed == 3, [r.error for r in rep.results]
    assert not by_group[tsha].accepted and "disagree" in by_group[tsha].error and fitted_twin in by_group[tsha].error and ruled_twin in by_group[tsha].error
    assert by_group[shas["fitted"]].accepted and by_group[shas["control"]].accepted and by_group[shas["lofit"]].accepted
    assert ad.status_extra(tsha) == {} and set(ad.status_extra(shas["lofit"])) == {"refuse", "parked"}
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id = ?", (tsha,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id = ?", (shas["fitted"],)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'objects_bank' AND accepted = 1").fetchone()[0] == 3
        # no accepted (empty) revision for the conflicting group: a refusal, where the park used to mint one
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE source_id = ? AND accepted = 1", (RUN.source_id_for(ad.name, tsha),)).fetchone()[0] == 0
        rows = {r[0]: r for r in conn.execute("SELECT source_id, ok, detail, extra_json FROM source_status WHERE adapter = 'objects_bank'")}
        parked = {sid for sid, r in rows.items() if json.loads(r[3]).get("parked")}
        assert len(parked) == 2 and all("objects_b" not in json.loads(rows[sid][3])["parked"][0]["dir"] for sid in parked)
        v = conn.execute("SELECT COUNT(*) FROM v_source_status_current WHERE adapter = 'objects_bank' AND json_extract(extra_json, '$.parked') IS NOT NULL").fetchone()[0]
        assert v == 2
    finally:
        conn.close()


def test_twins_every_one_ruled_out_of_scope_park_as_one_group_listing_each_directory(tmp_path):
    """The other half of the group verdict (round 2 of the review, 2026-10-07): the lofit bytes staged a second time under a directory of
    another name whose provenance declares the kind -- EVERY twin ruled out -- still park, as ONE accepted empty revision whose status facts
    list each directory under the spelling that parked it (`ParkedGroup`): two `refuse` lines, two `parked` entries, `scan` counting each
    directory in its own bank. The pass accepts the group, nothing is refused, an unchanged second pass writes nothing."""
    root, shas = _stage_bank_with_out_of_scope_bundles(tmp_path)
    twin = os.path.join(root, "synthetic_si", "objects_b", "synthetic_si__lofit_restaged")
    assert _sha(_blob(twin, b"lofit weights")) == shas["lofit"]
    _write(twin, "object.pt.provenance.json", {"kind": "lofit", "layers": [15], "object_sha256": shas["lofit"]})
    banks = [("stage_objects", root, "stage")]
    rep0 = OB.scan(banks, seeds=seeds(), read_blobs=False, now=T(9000))
    assert rep0.objects_total == 1 and rep0.refusals == [] and len(rep0.parked) == 2
    lof = next(p for p in rep0.parked if p["group"] == shas["lofit"])
    assert set(lof["dirs"].split(";")) == {shas["lofit_dir"], twin} and lof["kind_legacy"] == "lofit"
    assert sorted(t["dir"] for t in lof["twins"]) == sorted([shas["lofit_dir"], twin]) and all(t["kind_legacy"] == "lofit" and t["ruling"] == OB.PARKED_RULING for t in lof["twins"])
    assert rep0.banks["stage_objects/synthetic_si/objects_a"].parked == 1 and rep0.banks["stage_objects/synthetic_si/objects_a"].refused == 0
    bc = rep0.banks["stage_objects/synthetic_si/objects_b"]
    assert (bc.dirs, bc.parked, bc.refused, bc.objects) == (1, 1, 0, 0) and bc.parked_dirs == [{"dir": twin, "kind_legacy": "lofit"}]
    assert OB.format_scan(rep0).count("parked ") >= 3 and twin in OB.format_scan(rep0)
    reg = _registry(tmp_path)
    ad = OB.ObjectsBankAdapter(banks, seeds=seeds(), read_blobs=False)
    _producer_statements(root, None, reg, T(8000), banks=banks)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 3, [r.error for r in rep.results]
    by_group = {r.group: r for r in rep.results}
    assert by_group[shas["fitted"]].accepted and by_group[shas["control"]].accepted and by_group[shas["lofit"]].accepted
    facts = ad.status_extra(shas["lofit"])
    assert sorted(p["dir"] for p in facts["parked"]) == sorted([shas["lofit_dir"], twin]) and {p["kind_legacy"] for p in facts["parked"]} == {"lofit"}
    assert len(facts["refuse"]) == 2 and all("parked by name" in line and "D37" in line for line in facts["refuse"])
    assert {line.split(" is a ")[0].split("parked by name: ")[1] for line in facts["refuse"]} == {shas["lofit_dir"], twin}
    assert len(ad.status_extra(shas["control"])["parked"]) == 1          # a single-bundle park lists one, as before
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id IN (?, ?)", (shas["control"], shas["lofit"])).fetchone()[0] == 0
        rows = {r[0]: r for r in conn.execute("SELECT source_id, ok, detail, extra_json FROM source_status WHERE adapter = 'objects_bank'")}
        lrow = rows[RUN.source_id_for(ad.name, shas["lofit"])]
        extra = json.loads(lrow[3])
        assert lrow[1] == 0 and len(extra["parked"]) == 2 and len(extra["refuse"]) == 2 and "EMITTED NOTHING" not in lrow[2]
        assert sorted(p["dir"] for p in extra["parked"]) == sorted([shas["lofit_dir"], twin])
        agg = conn.execute("SELECT ok, json_array_length(json_extract(extra_json, '$.refuse')) FROM v_source_status WHERE adapter = 'objects_bank'").fetchall()
        assert [tuple(r) for r in agg] == [(0, 3)]                                 # the control's one line + the lofit group's two
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'objects_bank' AND accepted = 0").fetchone()[0] == 0
    finally:
        conn.close()
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep2 = RUN.run_pass(reg, [ad], conn=conn, now=T(9500))
    conn.close()
    assert rep2.inputs_changed == 0 and rep2.records_written == 0 and rep2.groups_refused == 0


def test_symlinked_weights_file_is_refused_by_d12(tmp_path):
    root, pins, shas = _bank(tmp_path)
    ad = _adapter(root, pins)
    target = os.path.join(shas["base_dir"], "object.pt")
    link_dir = os.path.join(root, "synthmodel", "cells", "linked__synth-corpus-full__lora_r4_ce")
    os.makedirs(link_dir)
    os.symlink(target, os.path.join(link_dir, "object.pt"))
    _write(link_dir, "train_budget_meta.json", corpus_tbm(shas["base"]))
    with pytest.raises(AB.InputRefusal) as ei:
        ad.discover()
    assert "symlink" in str(ei.value)


def test_refused_object_is_isolated_to_its_group(tmp_path):
    root, pins, shas = _bank(tmp_path)
    bad = os.path.join(root, "synthmodel", "cells", "candsi__Z9__lora_r4")
    bsha = _sha(_blob(bad, b"cand"))
    _write(bad, "train_budget_meta.json", si_tbm(bsha, si_id="Z9", arm="lora_r4", cell_id="Z9__lora_r4", model_id="synthetic/base-4b"))
    rep = OB.scan([("full22_objects", root, "full22")], seeds=seeds(), read_blobs=False, pins_path=pins, now=T(9000))
    assert rep.objects_total == 5 and len(rep.refusals) == 1 and "Z9" in rep.refusals[0]["why"]
    assert rep.banks["full22_objects"].refused == 1


def test_objects_scan_cli_on_the_synthetic_bank(tmp_path, capsys):
    from sisteer.registry import cli
    root, pins, shas = _bank(tmp_path)
    sd = str(tmp_path / "seeds")
    os.makedirs(sd)
    _write(sd, "si_texts.yaml", "si_texts:\n  - {si_id: synthetic_si, si_text_sha256: %s}\n" % synth.SI_SHA)
    _write(sd, "models.yaml", "models:\n  - {model_id: %s, hf_repo: synthetic/base-4b, hf_revision: %s}\n  - {model_id: %s, hf_repo: synthetic/instruct-4b, hf_revision: %s}\n"
           % (synth.FIT_MODEL_ID, "b" * 40, synth.MODEL_ID, "c" * 40))
    _write(sd, "legacy_spellings.yaml", "paper_labels:\n  - {legacy: \"synth-corpus-full__lora_r4_ce\", paper_label: constitution_lora, object_role: constitution, fit_corpus_id: synth-corpus-full, source: s}\n")
    rc = cli.main(["--root", str(tmp_path / "noreg"), "objects", "scan", "--bank-root", f"full22_objects={root}:full22", "--seeds", sd, "--pins", pins,
                   "--no-blobs", "--no-hpo", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["objects"]["objects_total"] == 5 and out["objects"]["banks"]["full22_objects"]["objects"] == 5
    assert out["wandb"]["run_ids_named"] == 0 and out["hpo"] is None
    rc = cli.main(["--root", str(tmp_path / "noreg"), "objects", "scan", "--bank-root", f"full22_objects={root}:full22", "--seeds", sd, "--pins", pins,
                   "--no-blobs", "--no-hpo"])
    text = capsys.readouterr().out
    assert rc == 0 and "full22_objects: dirs 6, objects 5" in text


def test_the_registered_adapter_reads_the_seeds_every_other_adapter_reads(tmp_path, monkeypatch):
    """The harvester instantiates `ObjectsBankAdapter()` — no seeds, no seeds directory. `load_seeds(None)` left every label / kind /
    fit-target table EMPTY, so the live registry filed `other` on all 337 fitted objects (`no legacy_spellings.yaml paper_labels entry
    for 'constv2-mix-full__lora_r4_ce'` on every row) and refused the paired writes for a kind the seeds do spell. The adapter now resolves
    the seeds directory the way relu_paired does (`$SISTEER_REGISTRY_SEEDS`, else this checkout's configs/registry), and the objects it
    mints carry the stem's label with the fit-variant prefix parked beside it."""
    root, pins, shas = _bank(tmp_path)
    sd = str(tmp_path / "seeds")
    os.makedirs(sd)
    for name in ("evals.yaml", "judges.yaml"):                      # the seed files resolve_seeds_dir requires; empty is a valid seed
        _write(sd, name, f"{name.split('.')[0]}: []\n")
    _write(sd, "si_texts.yaml", "si_texts:\n  - {si_id: synthetic_si, si_text_sha256: %s}\n" % synth.SI_SHA)
    _write(sd, "models.yaml", "models:\n  - {model_id: %s, hf_repo: synthetic/base-4b, hf_revision: %s}\n  - {model_id: %s, hf_repo: synthetic/instruct-4b, hf_revision: %s}\n"
           % (synth.FIT_MODEL_ID, "b" * 40, synth.MODEL_ID, "c" * 40))
    _write(sd, "legacy_spellings.yaml", "paper_labels:\n  - {legacy: \"synth-corpus-full__lora_r4_ce\", paper_label: constitution_lora, object_role: constitution, fit_corpus_id: synth-corpus-full, source: s}\n")
    monkeypatch.setenv("SISTEER_REGISTRY_SEEDS", sd)
    ad = OB.ObjectsBankAdapter([("full22_objects", root, "full22")], read_blobs=False, pins_path=pins)     # no seeds, as registered
    assert ad.seeds is None and ad.seeds_dir is None and OB.ObjectsBankAdapter.EMISSION_VERSION == 10     # 10: a dose copy's batch is its parent's (2026-10-13)
    # the context carries the pass's read-only connection as every runner read does (the bank itself needs none: it states a status only
    # from its own evidence and omits it otherwise)
    reg = _registry(tmp_path)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    ctx = AB.AdapterContext(root=reg, now=T(9000), producer={"script": "t", "code_sha": "unverified", "session": None, "host": "TESTHOST"}, conn=conn)
    inputs = ad.discover()
    by_group = {}
    for i in inputs:
        by_group.setdefault(ad.comparison_group(i), []).append(i)
    payloads = {}
    for g, ins in by_group.items():
        for batch in ad.read(ctx, g, ins):
            for rec in batch.records:
                payloads[rec["payload"]["campaign_cell_id_legacy"]] = rec["payload"]
    base = payloads["base__synth-corpus-full__lora_r4_ce"]
    assert base["paper_label"] == "constitution_lora" and base["object_role"] == "constitution"
    assert base["extra"]["fit_variant_prefix_legacy"] == "base" and base["extra"]["campaign_stem_legacy"] == "synth-corpus-full__lora_r4_ce"
    dose = payloads["d0p5__synth-corpus-full__lora_r4_ce"]
    assert dose["paper_label"] == "constitution_lora" and dose["extra"]["fit_variant_prefix_legacy"] == "d0p5"
    assert all(p["paper_label"] == "constitution_lora" for p in payloads.values()), {k: p["paper_label"] for k, p in payloads.items()}
    # the seeds directory is refused LOUDLY when a seed file is missing — never a silent empty table
    os.remove(os.path.join(sd, "judges.yaml"))
    with pytest.raises(AB.AdapterRefusal):
        ad._seeds(ctx)
    conn.close()


# --------------------------------------------------------------------------------------------
# a dose ladder's 1.0 ANCHOR of a LoRA subtraction reads as the subtraction, never as a dose copy of itself (2026-09-25)
# --------------------------------------------------------------------------------------------

DIFF_BYTES = b"difference weights"
SUB_STEM = "synth-corpus-full__lora_r4_ce_minus_ctl"


def _sub_seeds():
    s = seeds()
    s.paper_label_by_spelling["synth-ctl-full__lora_r4_ce"] = {"paper_label": "control_lora", "object_role": "control"}
    return s


def _subtraction_bank(tmp_path, neg_batch=8, neg_by_spelling=False, stage_neg=True, anchor_names=None, second_neg=False):
    """A full22-layout bank in the shape `scripts/sdfbase/lora_subtract.py` + `scale_lora_dose.py --anchor` leave: the two parent
    fits (constitution / control), the builder's `derived__` original, the ladder's `d1__` ANCHOR holding the SAME bytes with the
    builder's record + a `dose_scaled_from` stamp (anchor true, scale 1.0, parent sha == own sha) + `object.pt.provenance.json`
    (dose_scale 1.0), and a `d0p5__` dose child of the anchor. Knobs: the control fit's effective batch; the control parent named
    by its staged spelling alone (no object_sha256); the control fit absent from the bank; an anchor stamp naming another sha;
    a second control fit of the same spelling (an ambiguous resolution)."""
    root = str(tmp_path / "full22_objects")
    cells = os.path.join(root, "synthmodel", "cells")
    pos = os.path.join(cells, "base__synth-corpus-full__lora_r4_ce")
    psha = _sha(_blob(pos, b"pos fit weights"))
    _write(pos, "train_budget_meta.json", corpus_tbm(psha))
    neg = os.path.join(cells, "base__synth-ctl-full__lora_r4_ce")
    nsha = _sha(_blob(neg, b"neg fit weights"))
    if stage_neg:
        _write(neg, "train_budget_meta.json", corpus_tbm(nsha, si_id="synth-ctl-full", cell_id="synth-ctl-full__lora_r4_ce", effective_batch=neg_batch))
    else:
        import shutil
        shutil.rmtree(neg)
    if second_neg:
        neg2 = os.path.join(cells, "again__synth-ctl-full__lora_r4_ce")
        n2 = _sha(_blob(neg2, b"another neg fit"))
        _write(neg2, "train_budget_meta.json", corpus_tbm(n2, si_id="synth-ctl-full", cell_id="synth-ctl-full__lora_r4_ce", effective_batch=neg_batch))

    def record(sha):
        pp = {"cell_id": "synth-corpus-full__lora_r4_ce", "object_sha256": psha, "model_id": "synthetic/base-4b", "fit_fingerprint": "0" * 64,
              "sample_manifest_sha": CORPUS_SHA, "data_scale": {"official_tokens": 42, "si": "synth-corpus-full"}, "budget_meta_sha256": "3" * 64}
        pn = {"cell_id": "synth-ctl-full__lora_r4_ce", "model_id": "synthetic/base-4b", "fit_fingerprint": "9" * 64,
              "sample_manifest_sha": CORPUS_SHA, "data_scale": {"official_tokens": 42, "si": "synth-ctl-full"}, "budget_meta_sha256": "4" * 64}
        if not neg_by_spelling:
            pn["object_sha256"] = nsha
        return {"builder": "scripts/sdfbase/lora_subtract.py", "built_at": "2026-09-04T18:09:45+00:00",
                "comparison": "scale_matched (primary: both parents trained at the same official-token budget)",
                "derivation": "delta(parent_pos) - delta(parent_neg), structural (both parents' factors stored bit-exact)",
                "derived_id": SUB_STEM, "kind": "lora_subtraction", "layers": [3, 7, 11, 15], "model_id": "synthetic/base-4b", "model_key": "synthmodel",
                "model_revision": REV, "object_sha256": sha, "parent_pos": pp, "parent_neg": pn, "positions": "all", "rank": 4,
                "response_header_len": None, "scale_matched": True}

    orig = os.path.join(cells, f"derived__{SUB_STEM}")
    dsha = _sha(_blob(orig, DIFF_BYTES))
    _write(orig, "derived_meta.json", record(dsha))
    _write(orig, "staged_from.json", {"cell_id": SUB_STEM, "object_sha256": dsha, "source": "twosi-fits:out/synth/derived/x", "staged_id": f"derived__{SUB_STEM}"})

    def stamp(scale, parent, anchor):
        return {"anchor": anchor, "scale": scale, "parent_object_sha256": parent, "parent_staged_id": f"derived__{SUB_STEM}", "parent_identity": "derived",
                "parent_dir": orig, "parent_origin": orig, "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "5" * 40,
                "written_at": "2026-09-15T16:18:05Z", "object_type": "lora_subtraction", "parent_sidecar_object_sha256_matches_bytes": True,
                "scaling_method": "NONE: byte-identical copy of the parent object.pt" if anchor else "B_pos, B_neg x scale per layer",
                "tensors_scaled": [] if anchor else ["B_pos", "B_neg"], "tensors_unchanged": ["A_pos", "A_neg", "layers", "rank"],
                "norms_by_tensor_and_layer": {"B_pos": {"3": {"before": 1.0, "after": 1.0 if anchor else 0.5}}},
                "verification": {"status": "PASS", "anchor": anchor, "scale": scale, "kind": "lora_subtraction",
                                 "checks": [{"check": "tensors", "status": "PASS", "rule": "torch.equal(new, parent) for every tensor"}]},
                "note": "synthetic stamp"}

    anc = os.path.join(cells, f"d1__{SUB_STEM}")
    _blob(anc, DIFF_BYTES)                                    # the SAME bytes: a twin of the original, one object
    ameta = record(dsha)
    ameta["dose_scaled_from"] = stamp(1.0, anchor_names or dsha, True)
    _write(anc, "derived_meta.json", ameta)
    _write(anc, "object.pt.provenance.json", {"kind": "lora_subtraction", "layers": [3, 7, 11, 15], "dose_scale": 1.0, "parent_object_sha256": anchor_names or dsha})
    _write(anc, "staged_from.json", {"cell_id": SUB_STEM, "object_sha256": dsha, "staged_by": "scripts/sdfbase/scale_lora_dose.py", "staged_id": f"d1__{SUB_STEM}",
                                      "source": f"re-served 1.0 anchor: byte-identical copy of derived__{SUB_STEM}"})
    child = os.path.join(cells, f"d0p5__{SUB_STEM}")
    csha = _sha(_blob(child, b"half difference weights"))
    cmeta = record(csha)
    cmeta["dose_scaled_from"] = stamp(0.5, dsha, False)
    _write(child, "derived_meta.json", cmeta)
    _write(child, "object.pt.provenance.json", {"kind": "lora_subtraction", "layers": [3, 7, 11, 15], "dose_scale": 0.5, "parent_object_sha256": dsha})
    _write(child, "staged_from.json", {"cell_id": SUB_STEM, "object_sha256": csha, "staged_by": "scripts/sdfbase/scale_lora_dose.py", "staged_id": f"d0p5__{SUB_STEM}",
                                        "source": f"dose-scaled copy (x0.5) of derived__{SUB_STEM}"})
    return root, {"pos": psha, "neg": nsha, "diff": dsha, "child": csha, "orig_dir": orig, "anchor_dir": anc, "neg_dir": neg}


def _scan_sub(root):
    return OB.scan([("full22_objects", root, "full22")], seeds=_sub_seeds(), read_blobs=False, now=T(9000))


def test_subtraction_anchor_reads_as_the_subtraction_with_both_parents_and_their_batch(tmp_path):
    """Four live objects (the gemma / llama / qwen `d1__constv2-mix-full__lora_r4_ce_minus_ctl` and llama's `d1it__…`) recorded
    THEMSELVES as their only parent: the anchor twin wins primary by sidecar count and its `object.pt.provenance.json:dose_scale=1.0`
    (parent sha == own sha) read as a dose copy. An anchor is the subtraction re-served, so it reads as the subtraction: derivation
    lora_subtraction, parents = the two fits the record names, fit_effective_batch through both parents' fit records, the stamp kept
    as evidence in derivation_params_json."""
    root, shas = _subtraction_bank(tmp_path)
    rep = _scan_sub(root)
    assert rep.objects_total == 4 and not rep.refusals, rep.refusals
    by_id = {p["object_id"]: p for p in rep.payloads}
    a = by_id[shas["diff"]]
    assert a["campaign_cell_id_legacy"] == f"d1__{SUB_STEM}"                  # the anchor twin IS the primary (3 sidecars to the original's 2)
    assert sorted(a["extra"]["staged_ids"]) == [f"d1__{SUB_STEM}", f"derived__{SUB_STEM}"]
    assert a["object_kind"] == "lora_subtraction" and a["derivation"] == "lora_subtraction" and a["fit_target_kind"] == "derived"
    assert a["parent_object_ids_json"] == [shas["pos"], shas["neg"]] and a.get("base_scale") is None
    assert a["fit_effective_batch"] == 8
    fc = a["field_confidence_json"]
    assert fc["fit_effective_batch"]["confidence"] == "derived" and "parent_pos=8, parent_neg=8" in fc["fit_effective_batch"]["source"]
    assert fc["parent_object_ids_json"] == {"confidence": "recorded", "source": "derived_meta.json:parent_pos.object_sha256, parent_neg.object_sha256"}
    assert fc["derivation"]["confidence"] == "derived"
    # the BANK states no derivation_params_json it assembled (an abbreviation of the producer's record): the column and its confidence entry
    # are absent, extra.producer_owned_unstated says why, and the anchor stamp -- the evidence the assembled params carried -- rides extra.dose_anchor.stamp
    assert "derivation_params_json" not in a and "derivation_params_json" not in fc
    assert a["extra"]["producer_owned_unstated"]["derivation_params_json"].startswith("assembled from derived_meta.json + the dose_scaled_from anchor stamp")
    # an anchor is the subtraction (dose None), so its code sha is derived_meta.json:builder_commit -- unrecorded here -- never the stamp's producer_commit:
    # the bank default `unverified` is omitted too
    assert "fit_code_sha" not in a and sorted(a["extra"]["producer_owned_unstated"]) == ["derivation_params_json", "fit_code_sha"]
    stamp = a["extra"]["dose_anchor"]["stamp"]
    assert stamp["anchor"] is True and stamp["scale"] == 1.0 and stamp["parent_object_sha256"] == shas["diff"] and stamp["parent_staged_id"] == f"derived__{SUB_STEM}"
    assert stamp["producer"] == "scripts/sdfbase/scale_lora_dose.py" and stamp["written_at"] == "2026-09-15T16:18:05Z"
    assert "norms_by_tensor_and_layer" not in stamp and stamp["verification"]["checks"] == [{"check": "tensors", "status": "PASS"}] and stamp["verification"]["status"] == "PASS"
    pb = a["extra"]["parent_fit_effective_batch"]
    assert pb["parent_pos"] == {"object_sha256": shas["pos"], "effective_batch": 8, "batch_size": 2, "accum_steps": 4, "resolved_batch_size": 2,
                                "source": os.path.join(root, "synthmodel", "cells", "base__synth-corpus-full__lora_r4_ce", "train_budget_meta.json") + ":effective_batch"}
    assert pb["parent_neg"]["effective_batch"] == 8 and pb["parent_neg"]["source"].endswith("base__synth-ctl-full__lora_r4_ce/train_budget_meta.json:effective_batch")
    assert a["extra"]["dose_anchor"]["spelled_by"] == ["derived_meta.json:dose_scaled_from.anchor=true",
                                                       "derived_meta.json:dose_scaled_from.scale=1.0 + parent_object_sha256 == own sha",
                                                       "object.pt.provenance.json:dose_scale=1.0 + parent_object_sha256 == own sha"]
    assert "parent_record_note" not in a["extra"] and "fit_effective_batch_unresolved" not in a["extra"]
    assert a["fit_lr_selected_by"] == f"inherited_from:{shas['pos']}" and a["fit_writer"] == "scripts/sdfbase/lora_subtract.py"
    assert a["paper_label"] == "difference_lora" and a["object_role"] == "difference" and a["fit_finished_utc"] == "2026-09-04T18:09:45.000Z"
    assert a["fit_objective"] == "not_applicable" and a["fit_seed_confidence"] == "unverified" and a["fit_rank"] == 4
    # the dose child still reads as a dose copy pointing at the anchor
    c = by_id[shas["child"]]
    assert c["derivation"] == "dose_scale" and c["parent_object_ids_json"] == [shas["diff"]] and c["base_scale"] == 0.5
    assert c["object_kind"] == "lora_subtraction" and c["fit_target_kind"] == "derived"
    # the child is a dose copy: its assembled params (the provenance spelling's eight keys) are an abbreviation too, so the bank omits them;
    # the dose itself is a verbatim column (base_scale) and stays. _dose_copy reads the stamp from object.pt.provenance.json / train_budget_meta.json,
    # so a subtraction child's derived_meta.json stamp is not folded in (a pre-existing gap, out of scope here); nothing about the anchor rule reaches it
    assert "derivation_params_json" not in c and c["extra"]["producer_owned_unstated"]["derivation_params_json"].startswith("assembled from object.pt.provenance.json:dose_scale")
    # the copy's writer and finish time are the copy's OWN stamp (derived_meta.json:dose_scaled_from's producer /
    # written_at), never the parent subtraction's builder / built_at that the copied derived_meta.json carries beside it (which is what the
    # bank used to state: `scripts/sdfbase/lora_subtract.py` and 2026-09-04T18:09:45Z, the PARENT's)
    assert c["fit_writer"] == "scripts/sdfbase/scale_lora_dose.py" and c["fit_finished_utc"] == "2026-09-15T16:18:05.000Z", (c["fit_writer"], c["fit_finished_utc"])
    assert c["field_confidence_json"]["fit_writer"] == {"confidence": "recorded", "source": "derived_meta.json:dose_scaled_from:producer"}
    assert c["field_confidence_json"]["fit_finished_utc"] == {"confidence": "recorded", "source": "derived_meta.json:dose_scaled_from:written_at"}
    assert "fit_writer" not in c["extra"]["producer_owned_unstated"] and "fit_finished_utc" not in c["extra"]["producer_owned_unstated"]
    assert c["extra"]["parent_record_note"].startswith("train_budget_meta.json is the PARENT's")
    assert "dose_anchor" not in c["extra"] and "parent_fit_effective_batch" not in c["extra"]
    # the two fits are fits
    assert by_id[shas["pos"]]["derivation"] == "none" and by_id[shas["pos"]]["fit_effective_batch"] == 8 and by_id[shas["pos"]]["paper_label"] == "constitution_lora"
    assert by_id[shas["neg"]]["derivation"] == "none" and by_id[shas["neg"]]["paper_label"] == "control_lora"
    # the same record read from the ORIGINAL directory alone agrees on every derivation field
    from sisteer.registry.adapters import fit_sidecars as FS
    b = FS.read_bundle(shas["orig_dir"], read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    o = FS.build_object_payload(b, _sub_seeds(), T(9000), blob_reads_enabled=False)
    for k in ("object_id", "object_kind", "derivation", "parent_object_ids_json", "fit_target_kind", "paper_label", "object_role", "fit_lr_selected_by", "fit_writer", "fit_finished_utc"):
        assert o[k] == a[k], k
    # the builder's DEFAULT keeps the legacy statement (the other caller, relu_paired, is an open follow-up): the assembled dict, stated
    dp = o["derivation_params_json"]
    assert dp["builder"] == "scripts/sdfbase/lora_subtract.py" and dp["parents"]["parent_pos"]["object_sha256"] == shas["pos"] and "dose_scaled_from" not in dp
    assert "producer_owned_unstated" not in o["extra"]
    # and the bank's switch on the same bundle agrees with the bank
    o2 = FS.build_object_payload(b, _sub_seeds(), T(9000), blob_reads_enabled=False, omit_producer_owned=True)
    assert "derivation_params_json" not in o2 and o2["extra"]["producer_owned_unstated"]["derivation_params_json"].startswith("assembled from derived_meta.json")


def test_subtraction_anchor_rows_ingest_with_both_parents_and_the_child_pointing_at_the_anchor(tmp_path):
    root, shas = _subtraction_bank(tmp_path)
    reg = _registry(tmp_path)
    ad = OB.ObjectsBankAdapter([("full22_objects", root, "full22")], seeds=_sub_seeds(), read_blobs=False)
    _producer_statements(root, None, reg, T(8000), banks=ad.banks, seeds_obj=_sub_seeds())   # the producer's statements precede the bank's pass
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refreshed == 4 and rep.groups_refused == 0, [r.error for r in rep.results]
    for _ in range(3):                                      # parents and children arrive in one pass; a child may wait a cycle for its parent row
        ing = H.ingest_once(reg, workers=0, min_free=0)
        assert ing.refused == 0, ing.refusals
        if not ing.blocked:
            break
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        rows = {r[0]: r for r in conn.execute("SELECT object_id, derivation, parent_object_ids_json, fit_effective_batch, base_scale, paper_label FROM object WHERE campaign = 'sdfbase_full22'")}
        assert set(rows) == {shas["pos"], shas["neg"], shas["diff"], shas["child"]}
        a = rows[shas["diff"]]
        assert a[1] == "lora_subtraction" and json.loads(a[2]) == [shas["pos"], shas["neg"]] and a[3] == 8 and a[4] is None and a[5] == "difference_lora"
        c = rows[shas["child"]]
        assert c[1] == "dose_scale" and json.loads(c[2]) == [shas["diff"]] and c[4] == 0.5
    finally:
        conn.close()


def test_anchor_parents_disagreeing_on_batch_leave_the_column_null_with_both_values(tmp_path):
    root, shas = _subtraction_bank(tmp_path, neg_batch=16)
    rep = _scan_sub(root)
    assert rep.objects_total == 4 and not rep.refusals, rep.refusals
    a = next(p for p in rep.payloads if p["object_id"] == shas["diff"])
    assert a["derivation"] == "lora_subtraction" and a["parent_object_ids_json"] == [shas["pos"], shas["neg"]]
    assert "fit_effective_batch" not in a
    why = a["extra"]["fit_effective_batch_unresolved"]
    assert "DISAGREE" in why and "parent_pos=8, parent_neg=16" in why and "left NULL" in why
    assert a["field_confidence_json"]["fit_effective_batch"] == {"confidence": "unverified", "source": why, "reason": why}
    pb = a["extra"]["parent_fit_effective_batch"]
    assert pb["parent_pos"]["effective_batch"] == 8 and pb["parent_neg"]["effective_batch"] == 16


def test_anchor_parent_missing_from_the_bank_leaves_batch_null_naming_the_parent(tmp_path):
    root, shas = _subtraction_bank(tmp_path, stage_neg=False)
    rep = _scan_sub(root)
    assert rep.objects_total == 3 and not rep.refusals, rep.refusals
    a = next(p for p in rep.payloads if p["object_id"] == shas["diff"])
    assert a["parent_object_ids_json"] == [shas["pos"], shas["neg"]]          # the ids come from the record; only the batch reaches through the bank
    assert "fit_effective_batch" not in a
    why = a["extra"]["fit_effective_batch_unresolved"]
    assert f"parent_neg ({shas['neg'][:12]}…) is not staged in a bank on this machine" in why
    assert a["field_confidence_json"]["fit_effective_batch"]["confidence"] == "unverified"
    pb = a["extra"]["parent_fit_effective_batch"]
    assert pb["parent_pos"]["effective_batch"] == 8 and pb["parent_neg"] == {"object_sha256": shas["neg"], "effective_batch": None, "batch_size": None,
                                                                              "accum_steps": None, "resolved_batch_size": None, "source": None}


def test_anchor_parent_named_by_spelling_alone_resolves_through_the_banks_fit_of_that_spelling(tmp_path):
    root, shas = _subtraction_bank(tmp_path, neg_by_spelling=True)
    rep = _scan_sub(root)
    assert rep.objects_total == 4 and not rep.refusals, rep.refusals
    a = next(p for p in rep.payloads if p["object_id"] == shas["diff"])
    assert a["parent_object_ids_json"] == [shas["pos"], shas["neg"]] and a["fit_effective_batch"] == 8
    fc = a["field_confidence_json"]["parent_object_ids_json"]
    assert fc["confidence"] == "derived" and f"parent_neg=synth-ctl-full__lora_r4_ce→{shas['neg'][:12]}…" in fc["source"] and "same checkpoint" in fc["source"]
    assert "derivation_params_json" not in a         # the assembled params (where the by-spelling parent carried no sha) are the bank's abbreviation, omitted


def test_anchor_parent_named_by_spelling_alone_is_refused_by_name_when_no_or_several_fits_match(tmp_path):
    root, shas = _subtraction_bank(tmp_path, neg_by_spelling=True, stage_neg=False)
    rep = _scan_sub(root)
    assert rep.objects_total == 2 and len(rep.refusals) == 1, rep.refusals   # the anchor's group refused; the two others read
    why = rep.refusals[0]["why"]
    assert "parent_neg" in why and "cell_id='synth-ctl-full__lora_r4_ce'" in why and "0 fit(s)" in why and "never a guess from the name" in why
    root2, shas2 = _subtraction_bank(tmp_path / "two", neg_by_spelling=True, second_neg=True)
    rep2 = _scan_sub(root2)
    assert len(rep2.refusals) == 1 and "2 fit(s)" in rep2.refusals[0]["why"] and shas2["neg"][:12] in rep2.refusals[0]["why"]


def test_anchor_stamp_naming_another_objects_sha_is_refused(tmp_path):
    root, shas = _subtraction_bank(tmp_path, anchor_names=PARENT_A)
    rep = _scan_sub(root)
    assert len(rep.refusals) == 1, rep.refusals
    why = rep.refusals[0]["why"]
    assert "says anchor" in why and PARENT_A[:12] in why and shas["diff"][:12] in why


def test_emission_version_8_is_a_new_revision_over_unchanged_inputs(tmp_path):
    """The bump is what republishes every bank object under the new rules (10: a dose copy's fit_effective_batch stated as its parent's,
    derived; 9: a bundle of an out-of-scope kind parked by name instead of refused; 8: a dose copy's fit_writer / fit_finished_utc from
    its own stamp or omitted; 7: the producer-owned omission + the per-object location bytes; 6: the status omission; 5: the restatement;
    4 re-read the four anchors): the runner folds EMISSION_VERSION into the revision id, so unchanged bank files publish a new revision
    under 10 -- the re-emission that restates the 92 dose copies' batch."""
    root, shas = _subtraction_bank(tmp_path)
    ad = OB.ObjectsBankAdapter([("full22_objects", root, "full22")], seeds=_sub_seeds(), read_blobs=False)
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == shas["diff"]]
    assert inputs and OB.ObjectsBankAdapter.EMISSION_VERSION == 10    # 10: a dose copy's batch is its parent's, derived
    src = RUN.source_id_for(ad.name, shas["diff"])
    assert RUN.revision_id_for(src, inputs, 10) != RUN.revision_id_for(src, inputs, 9)
    assert RUN.revision_id_for(src, inputs, AB.emission_version_of(ad)) == RUN.revision_id_for(src, inputs, 10)


# --------------------------------------------------------------------------------------------
# status is a statement, never the bank's opinion (omitted where the bank states none)
# --------------------------------------------------------------------------------------------


def _bank_records(reg):
    """Every `object` payload the bank published under the scratch registry, by object id."""
    out = {}
    base = os.path.join(reg, "records", "adapters", "objects_bank")
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            if R.RECORD_FILE_RE.match(fn) and "-object." in fn:
                for _n, rec, _sha in R.read_record_file(os.path.join(dirpath, fn), compute_sha=False):
                    out[rec["payload"]["object_id"]] = rec["payload"]
    return out


def test_bank_states_a_status_only_from_its_own_evidence(tmp_path):
    """`fit_sidecars.bank_status_statement`: the layout (`_withdrawn/`, `_superseded_by_it64/`) and a superseded_by.json naming this
    object's fit fingerprint are statements; a complete fit with no such evidence, the expected refit of a superseded snapshot and an
    unresolved superseded_by.json are NOT -- the bank has nothing to say and names why."""
    root, pins, shas = _bank(tmp_path)
    ad = _adapter(root, pins)
    ad.discover()
    cells = os.path.join(root, "synthmodel", "cells")
    by_dir = {b.dir: b for b in ad.bundles_for([shas["base_dir"], os.path.join(cells, "_superseded_by_it64", "d0p5it__synth-corpus-full__lora_r4_ce"),
                                                 os.path.join(root, "_withdrawn", "synthmodel__d1it__synth-corpus-full__lora_r4_ce.withdrawn-2026")])}
    plain = FS.bank_status_statement(by_dir[shas["base_dir"]])
    assert plain == (None, "no supersession or withdrawal evidence beside the object", "none")
    twin = FS.bank_status_statement(next(b for b in by_dir.values() if b.status_hint == "superseded_twin"))
    assert twin[0] == "superseded" and twin[2] == "superseded_twin" and "_superseded_by_it64/" in twin[1]
    wd = FS.bank_status_statement(next(b for b in by_dir.values() if b.status_hint == "withdrawn"))
    assert wd[0] == "withdrawn" and wd[2] == "withdrawn" and "_withdrawn/" in wd[1]
    # superseded_by.json: this object's own fingerprint → superseded; the expected refit → nothing; neither fingerprint → nothing (unresolved)
    def _with_sb(name, sb):
        d = str(tmp_path / "sb" / name)
        sha = _sha(_blob(d, name.encode()))
        _write(d, "train_budget_meta.json", corpus_tbm(sha))
        _write(d, "COMPLETE", {"fit_fingerprint": "0" * 64, "object_sha256": sha, "published_at": "2026-09-04T02:48:47+00:00", "files": {}})
        _write(d, "superseded_by.json", sb)
        return FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    own = _with_sb("own", {"fit_fingerprint": "0" * 64, "superseded_by": {"expected_refit_fingerprint": "9" * 64}, "snapshot": "/vol/out/prior"})
    assert FS.bank_status_statement(own)[0] == "superseded" and FS.bank_status_statement(own)[2] == "superseded_by_json"
    refit = _with_sb("refit", {"fit_fingerprint": "9" * 64, "superseded_by": {"expected_refit_fingerprint": "0" * 64}, "snapshot": "/vol/out/prior"})
    assert FS.bank_status_statement(refit)[0] is None and FS.bank_status_statement(refit)[2] == "refit_of_superseded"
    other = _with_sb("other", {"fit_fingerprint": "8" * 64, "superseded_by": {"expected_refit_fingerprint": "7" * 64}})
    assert FS.bank_status_statement(other)[0] is None and FS.bank_status_statement(other)[2] == "superseded_by_json_unresolved"
    # and build_object_payload files exactly these: the statement where there is one, NO status where there is none (the default
    # `approved` is the producers' to state; the payload parks why the bank states none under extra.status_unstated)
    assert FS.build_object_payload(own, seeds(), T(9000), blob_reads_enabled=False)["status"] == "superseded"
    for b in (refit, other):
        p = FS.build_object_payload(b, seeds(), T(9000), blob_reads_enabled=False)
        assert "status" not in p and "status" not in p["field_confidence_json"] and p["extra"]["status_unstated"].startswith("the bank states no status for this object (D721)")
        assert "unstated_columns" not in p["extra"] and "status_origin" not in p["extra"]
    assert FS.BANK_DEFAULT_STATUS == "approved" and set(FS.STATUS_BRANCHES) >= {"none", "withdrawn", "superseded_twin", "superseded_by_json"}


def test_a_bundle_that_states_no_status_omits_it_whatever_the_registry_row_says(tmp_path):
    """The payload-level rule. A bundle that states no status files NONE -- no `status` column, no field_confidence entry, no
    `unstated_columns` marker, no `status_origin` -- whether the registry holds a `reference` row for the id, an `approved` one or none at
    all; `extra.status_unstated` names why the bank states none. A bundle that DOES state one (the superseded twin, the withdrawn copy)
    states it whatever the row says. The scan reads the same with and without a registry connection: no row is consulted."""
    root, pins, shas = _bank(tmp_path)
    reg = _registry(tmp_path, prior_objects=_producer_records(root, pins, [(shas["base"], "reference"), (shas["old"], "reference")]))
    ad = _adapter(root, pins)
    ad.discover()
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT status FROM object WHERE object_id = ?", (shas["base"],)).fetchone()[0] == "reference"
        rep = OB.scan(ad.banks, seeds=seeds(), read_blobs=False, pins_path=pins, now=T(9000), conn=conn)
    finally:
        conn.close()
    assert rep.objects_total == 5 and not rep.refusals
    by_id = {p["object_id"]: p for p in rep.payloads}
    for sha in (shas["base"], shas["dose"], shas["it64"]):                       # a reference row, no row, no row
        p = by_id[sha]
        assert "status" not in p and "status" not in p["field_confidence_json"], sha[:12]
        assert "unstated_columns" not in p["extra"] and "status_origin" not in p["extra"], sha[:12]
        assert p["extra"]["status_unstated"] == "the bank states no status for this object (D721): no supersession or withdrawal evidence beside the object", sha[:12]
    old = by_id[shas["old"]]
    assert old["status"] == "superseded" and old["superseded_by_object_id"] == shas["it64"] and "status_unstated" not in old["extra"]
    assert by_id[shas["withdrawn"]]["status"] == "withdrawn" and "status_unstated" not in by_id[shas["withdrawn"]]["extra"]
    assert rep.banks["full22_objects"].by_status == {"superseded": 1, "withdrawn": 1} and rep.banks["full22_objects"].status_unstated == 3
    # without a registry: the very same payloads (the bank consults no row)
    rep0 = OB.scan(ad.banks, seeds=seeds(), read_blobs=False, pins_path=pins, now=T(9000))
    assert {p["object_id"]: p for p in rep0.payloads} == by_id


def test_reread_of_a_reference_object_keeps_reference_through_the_fold(tmp_path):
    """End to end, in a scratch registry: the producer's `reference` is folded, the bank is read (omitting status), the fold still
    reads `reference` with the bank's record as the carrier; an object with NO prior row is dependency-blocked on its own row (never the
    default) until its producer states a status -- and so is the superseded twin, whose stated `superseded` points at the it64
    successor nobody has stated yet; once the producers state the two, everything folds: the twin's `superseded` lands over its
    `reference` row and the two new objects carry the producers' status. The live row keeps NO confidence entry for a column
    the latest record omits (`field_confidence_json` is a latest-statement column; the producer's entry stays in the history), so the
    bank's record claims no evidence for the status it did not carry."""
    root, pins, shas = _bank(tmp_path)
    reg = _registry(tmp_path, prior_objects=_producer_records(root, pins, [(shas["base"], "reference"), (shas["old"], "reference")]))
    ad = _adapter(root, pins)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refreshed == 5 and rep.groups_refused == 0, [r.error for r in rep.results]
    published = _bank_records(reg)
    assert "status" not in published[shas["base"]] and "unstated_columns" not in published[shas["base"]]["extra"] and "status_origin" not in published[shas["base"]]["extra"]
    assert "status" not in published[shas["dose"]] and published[shas["dose"]]["extra"]["status_unstated"].startswith("the bank states no status")
    assert published[shas["old"]]["status"] == "superseded" and published[shas["withdrawn"]]["status"] == "withdrawn"
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    H.ingest_once(reg, workers=0, min_free=0)             # a second attempt for the blocked queue: the no-row objects stay blocked
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        status, uri = conn.execute("SELECT status, record_uri FROM object WHERE object_id = ?", (shas["base"],)).fetchone()
        assert status == "reference" and uri.startswith("records/adapters/objects_bank/")       # omitted, so the fold keeps it; the bank's record carries the row
        assert conn.execute("SELECT status FROM object WHERE object_id = ?", (shas["withdrawn"],)).fetchone()[0] == "withdrawn"
        # the two objects nobody has stated a status for are not rows: their bank records wait on their own row, never refused;
        # the superseded twin's record waits on the it64 successor it names, so its `reference` row stands until then
        assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id IN (?, ?)", (shas["dose"], shas["it64"])).fetchone()[0] == 0
        assert tuple(conn.execute("SELECT status, superseded_by_object_id FROM object WHERE object_id = ?", (shas["old"],)).fetchone()) == ("reference", None)
        blocked = sorted((r[0], r[1]) for r in conn.execute("SELECT missing_parent_table, missing_parent_id FROM blocked_records WHERE record_type = 'object'"))
        assert blocked == sorted([("object", shas["dose"]), ("object", shas["it64"]), ("object", shas["it64"])]), blocked
        # the base row's status is the producer's value; its confidence entry is not on the live row (the bank's record, the latest, omitted it)
        assert "status" not in json.loads(conn.execute("SELECT field_confidence_json FROM object WHERE object_id = ?", (shas["base"],)).fetchone()[0])
        assert json.loads(conn.execute("SELECT extra_json FROM object WHERE object_id = ?", (shas["base"],)).fetchone()[0]).get("status_origin") is None
    finally:
        conn.close()
    # the producers state the two: the blocked bank records fold behind them (the twin's too) with the producers' status
    R.write_records(os.path.join(reg, "records", "producer", "000001-object.jsonl"),
                    [synth.rec("object", p, T(9500)) for p in _producer_records(root, pins, [(shas["dose"], "approved"), (shas["it64"], "approved")])])
    ing2 = H.ingest_once(reg, workers=0, min_free=0)
    assert ing2.refused == 0 and ing2.unblocked >= 2
    conn = DB.connect(DB.db_path(reg), readonly=True)
    if conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0]:
        conn.close()
        H.ingest_once(reg, workers=0, min_free=0)
        conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
        assert {r[0]: r[1] for r in conn.execute("SELECT object_id, status FROM object WHERE campaign = 'sdfbase_full22'")} \
            == {shas["base"]: "reference", shas["dose"]: "approved", shas["it64"]: "approved", shas["old"]: "superseded", shas["withdrawn"]: "withdrawn"}
        assert conn.execute("SELECT superseded_by_object_id FROM object WHERE object_id = ?", (shas["old"],)).fetchone()[0] == shas["it64"]
    finally:
        conn.close()
    # a second pass over unchanged inputs writes nothing: the omission is not a change
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep2 = RUN.run_pass(reg, [ad], conn=conn, now=T(9800))
    conn.close()
    assert rep2.inputs_changed == 0 and rep2.records_written == 0


def test_read_without_a_registry_connection_publishes_the_same_records(tmp_path):
    """A read with no registry connection publishes every group: the bank consults no row, so nothing is refused and the three
    groups it states no status for publish records without the column, the two it states (superseded twin, withdrawn) with it."""
    root, pins, shas = _bank(tmp_path)
    reg = _registry(tmp_path)
    ad = _adapter(root, pins)
    rep = RUN.run_pass(reg, [ad], conn=None, now=T(9000))
    assert rep.groups_refused == 0 and rep.groups_refreshed == 5, [(r.group[:12], r.error) for r in rep.results]
    published = _bank_records(reg)
    assert set(published) == {shas["base"], shas["dose"], shas["it64"], shas["old"], shas["withdrawn"]}
    assert all("status" not in published[k] for k in (shas["base"], shas["dose"], shas["it64"]))
    assert published[shas["old"]]["status"] == "superseded" and published[shas["withdrawn"]]["status"] == "withdrawn"


# --------------------------------------------------------------------------------------------
# producer-owned columns: never the bank default, never an abbreviation (2026-10-05 -- the status rule generalized)
# --------------------------------------------------------------------------------------------


def _owner_bank(tmp_path):
    """A full22-layout bank holding ONE dose copy whose stamp records no producer commit: the legacy builder filed `fit_code_sha='unverified'`
    (the bank default) and an eight-key abbreviation of the stamp as derivation_params_json for it -- the two statements that used to overwrite
    the producers' native records."""
    root = str(tmp_path / "full22_objects")
    d = os.path.join(root, "synthmodel", "cells", "d0p25__synth-corpus-full__lora_r4_ce")
    sha = _sha(_blob(d, b"quarter weights"))
    tbm = corpus_tbm(sha)
    tbm["dose_scaled_from"] = {"scale": 0.25, "parent_object_sha256": PARENT_A, "producer": "scripts/sdfbase/scale_lora_dose.py", "written_at": "2026-10-05T07:00:00Z",
                               "parent_dir": "/home/synthetic/bank/base__x", "scaling_method": "write basis B x scale per layer; A untouched", "tensors_scaled": ["B"],
                               "tensors_unchanged": ["A", "layers", "rank"], "anchor": False, "parent_staged_id": "base__synth-corpus-full__lora_r4_ce",
                               "parent_identity": "staged", "verification": {"status": "PASS"}, "note": "a post-hoc dose-scaled copy of the parent object",
                               "norms_by_tensor_and_layer": {"B": {"3": {"before": 1.0, "after": 0.25}}}}
    _write(d, "train_budget_meta.json", tbm)
    _write(d, "object.pt.provenance.json", {"kind": "lora_all", "layers": [3, 7, 11, 15], "dose_scale": 0.25, "parent_object_sha256": PARENT_A})
    return root, sha, d


#: what the PRODUCER's native record states for the same dose copy (the shape of a live producer dose record: thirteen keys of its own, a real code sha)
PRODUCER_CODE_SHA = "7" * 40
PRODUCER_PARAMS = {"base_scale": 0.25, "dose_unit": "fraction_of_parent", "parent_object_sha256": PARENT_A, "parent_staged_id": "base__synth-corpus-full__lora_r4_ce",
                   "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": PRODUCER_CODE_SHA, "ruling": None, "scale": 0.25,
                   "scaling_method": "write basis B x scale per layer; A untouched", "staged_id": "d0p25__synth-corpus-full__lora_r4_ce", "tensors_scaled": ["B"],
                   "verification": {"status": "PASS"}, "view": "d0p25"}


def test_the_bank_omits_producer_owned_defaults_and_the_fold_keeps_the_producers_statement(tmp_path):
    """End to end: the producer's native record (a real `fit_code_sha`, its own `derivation_params_json`) is folded first; the bank is read
    LATER (a newer written_utc) and its record OMITS both columns -- `unverified` is the bank default, the eight-key dict an abbreviation -- so
    the per-column latest-statement fold keeps the producer's values with the bank's record as the carrier. Before the fix the bank's newer
    record stated both and the row read `unverified` and the abbreviation."""
    root, sha, _d = _owner_bank(tmp_path)
    rep = OB.scan([("full22_objects", root, "full22")], seeds=seeds(), read_blobs=False, now=T(9000))
    assert rep.objects_total == 1 and not rep.refusals, rep.refusals
    p = rep.payloads[0]
    # the emitted payload: neither key, no confidence entry for either, the reason for each under extra.producer_owned_unstated
    assert p["object_id"] == sha and p["derivation"] == "dose_scale" and p["base_scale"] == 0.25 and p["parent_object_ids_json"] == [PARENT_A]
    assert "fit_code_sha" not in p and "derivation_params_json" not in p
    assert "fit_code_sha" not in p["field_confidence_json"] and "derivation_params_json" not in p["field_confidence_json"]
    unstated = p["extra"]["producer_owned_unstated"]
    assert sorted(unstated) == ["derivation_params_json", "fit_code_sha"]
    assert unstated["fit_code_sha"].startswith("no sidecar records the fit code sha") and "bank default" in unstated["fit_code_sha"] and "D874" in unstated["fit_code_sha"]
    assert unstated["derivation_params_json"].startswith("assembled from train_budget_meta.json:dose_scaled_from") and "abbreviation" in unstated["derivation_params_json"]
    assert "status" not in p and "D721" in p["extra"]["status_unstated"]           # the status-omission precedent beside it
    # the copy's writer / finish time are its own stamp's producer / written_at (train_budget_meta.json:dose_scaled_from), stated
    assert p["fit_writer"] == "scripts/sdfbase/scale_lora_dose.py" and p["fit_finished_utc"] == "2026-10-05T07:00:00.000Z"
    assert p["field_confidence_json"]["fit_writer"]["source"] == "train_budget_meta.json:dose_scaled_from:producer"
    assert p["field_confidence_json"]["fit_finished_utc"]["source"] == "train_budget_meta.json:dose_scaled_from:written_at"
    assert rep.banks["full22_objects"].unverified_by_field.get("fit_code_sha") is None   # the scan counts no unverified fit_code_sha: none is stated
    # the producer's native record, folded BEFORE the bank is read
    producer = {k: v for k, v in p.items() if k not in ("extra", "field_confidence_json")}
    producer.update(status="approved", fit_code_sha=PRODUCER_CODE_SHA, derivation_params_json=dict(PRODUCER_PARAMS))
    producer["field_confidence_json"] = {**{k: v for k, v in p["field_confidence_json"].items()},
                                         "status": {"confidence": "recorded", "source": "the producer's statement"},
                                         "fit_code_sha": {"confidence": "recorded", "source": "the producer's commit"},
                                         "derivation_params_json": {"confidence": "recorded", "source": "the producer's dose record"}}
    producer["extra"] = {"minted_by": "a synthetic producer"}
    reg = _registry(tmp_path, prior_objects=[producer])
    row0 = DB.connect(DB.db_path(reg), readonly=True).execute("SELECT fit_code_sha, derivation_params_json, status FROM object WHERE object_id = ?", (sha,)).fetchone()
    assert row0[0] == PRODUCER_CODE_SHA and json.loads(row0[1]) == PRODUCER_PARAMS and row0[2] == "approved"
    # the bank's pass, stamped later than the producer's record
    ad = OB.ObjectsBankAdapter([("full22_objects", root, "full22")], seeds=seeds(), read_blobs=False)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep2 = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep2.groups_refreshed == 1 and rep2.groups_refused == 0, [r.error for r in rep2.results]
    published = _bank_records(reg)[sha]
    assert "fit_code_sha" not in published and "derivation_params_json" not in published and sorted(published["extra"]["producer_owned_unstated"]) == ["derivation_params_json", "fit_code_sha"]
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    H.ingest_once(reg, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
        code, params, status, uri, fv, fc = conn.execute("SELECT fit_code_sha, derivation_params_json, status, record_uri, field_versions_json, field_confidence_json "
                                                         "FROM object WHERE object_id = ?", (sha,)).fetchone()
        # the producer's values stand; the bank's (newer) record carries the row
        assert code == PRODUCER_CODE_SHA and json.loads(params) == PRODUCER_PARAMS and status == "approved"
        assert uri.startswith("records/adapters/objects_bank/")
        fv = json.loads(fv)
        assert fv["fit_code_sha"][0] < fv["base_scale"][0] and fv["derivation_params_json"][0] < fv["base_scale"][0], "the two columns keep the producer's stamp; base_scale is the bank's"
        # the live row keeps no confidence entry for a column the latest record omits (the producer's entry stays in the history)
        fc = json.loads(fc)
        assert "fit_code_sha" not in fc and "derivation_params_json" not in fc and "base_scale" in fc
    finally:
        conn.close()
    # a second pass over unchanged inputs writes nothing: the omission is not a change
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep3 = RUN.run_pass(reg, [ad], conn=conn, now=T(9800))
    conn.close()
    assert rep3.inputs_changed == 0 and rep3.records_written == 0


def test_a_sidecars_own_params_block_is_stated_verbatim_and_a_recorded_code_sha_stays(tmp_path):
    """The two things the bank MAY state: a `fit_code_sha` a sidecar records (a dose stamp's producer_commit, a derived_meta.json
    builder_commit) and a `derivation_params_json` that is a sidecar's own `params` block, verbatim -- the all-positions dose record's
    `params` here (`_arditi_rung_sidecars`, the all-positions staging), the same field the producer-side writer copies -- never the builder's
    assembly beside it."""
    from sisteer.registry import derive as D
    from test_registry_adapter_arditi_vendor_mt import RUNG, _arditi_rung_sidecars
    root = str(tmp_path / "full22_objects")
    d = os.path.join(root, "synthmodel", "cells", f"{RUNG}__arditi_allpos")
    sha = _sha(_blob(d, b"all positions dose weights"))
    for name, payload in _arditi_rung_sidecars(sha=sha).items():
        _write(d, name, payload)
    rep = OB.scan([("full22_objects", root, "full22")], seeds=seeds(), read_blobs=False, now=T(9000))
    assert rep.objects_total == 1 and not rep.refusals, rep.refusals
    p = rep.payloads[0]
    assert p["derivation"] == "dose_scale" and p["base_scale"] == 1.5 and p["parent_object_ids_json"] == ["f" * 64]
    assert p["derivation_params_json"] == {"base_scale": 1.5}, "the record's params block, verbatim: no scaling_method / parent_identity assembly beside it"
    assert p["field_confidence_json"]["derivation_params_json"] == {"confidence": "recorded", "source": f"{D.DOSE_SCALE_DERIVATION_SIDECARS[0]}:params (verbatim; D874)"}
    assert "fit_code_sha" not in p and list(p["extra"]["producer_owned_unstated"]) == ["fit_code_sha"]      # that record names no commit
    # the legacy default of the builder still assembles (relu_paired's statement, an open follow-up)
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    o = FS.build_object_payload(b, seeds(), T(9000), blob_reads_enabled=False)
    assert o["derivation_params_json"]["base_scale"] == 1.5 and o["derivation_params_json"]["parent_identity"]["parent_view"] == "ar_c0p1" and o["fit_code_sha"] == "unverified"
    assert "producer_owned_unstated" not in o["extra"]
    # a recorded code sha IS stated by the bank: the dose stamp's producer_commit on the shared fixture's dose copy
    root2, _pins, shas = _bank(tmp_path / "shared")
    rep2 = OB.scan([("full22_objects", root2, "full22")], seeds=seeds(), read_blobs=False, now=T(9000))
    dose = next(q for q in rep2.payloads if q["object_id"] == shas["dose"])
    assert dose["fit_code_sha"] == "5" * 40 and dose["field_confidence_json"]["fit_code_sha"]["confidence"] == "recorded"
    assert "fit_code_sha" not in dose["extra"]["producer_owned_unstated"] and "derivation_params_json" in dose["extra"]["producer_owned_unstated"]


# --------------------------------------------------------------------------------------------
# a cluster location's bytes are the OBJECT's, never the staged directory's total
# --------------------------------------------------------------------------------------------


def test_two_staged_objects_report_their_own_byte_counts_at_their_cluster_locations_not_the_directory_totals(tmp_path):
    """Two objects staged under one bank's `cells/`, each with a `_records` entry whose `bytes` is the staged DIRECTORY's total (weights plus
    sidecars, as stage_objects.py sums it): each verified cluster location reports the OBJECT's own size (what the object id hashes over, measured
    here on the sha-identical copy) with the record's figure beside it as `staged_dir_bytes`; an unverified copy claims no size; the two differ.
    Before the fix both locations carried the directory totals (9 of 9 live records)."""
    root = str(tmp_path / "full22_objects")
    cells = os.path.join(root, "synthmodel", "cells")
    objs = {}
    for staged_id, si, weights in (("base__synth-corpus-full__lora_r4_ce", "synth-corpus-full", b"w" * 1000), ("base__synth-ctl-full__lora_r4_ce", "synth-ctl-full", b"v" * 2500)):
        d = os.path.join(cells, staged_id)
        sha = _sha(_blob(d, weights))
        _write(d, "train_budget_meta.json", corpus_tbm(sha, si_id=si, cell_id=staged_id.split("__", 1)[1]))
        _write(d, "staged_from.json", {"source": f"twosi-fits:out/synth/cells_all/{staged_id}", "object_sha256": sha})
        total = sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d))
        assert total > len(weights)
        _write(os.path.join(root, "_records"), f"synthmodel__{staged_id}.json",
               {"target": "synthmodel", "staged_id": staged_id, "object_sha256": sha, "bytes": total, "n_files": 3,
                "cluster_a_path": f"/shared/synthetic/exp/objects/sdfbase/full22/synthmodel/cells/{staged_id}", "cluster_a_verified": True, "cluster_a_remote_sha256": sha,
                "cluster_b_path": f"/workspace/synthetic/exp/objects/sdfbase/full22/synthmodel/cells/{staged_id}", "cluster_b_verified": True, "cluster_b_remote_sha256": "0" * 64,
                "volume_path": f"twosi-fits:out/synth/cells_all/{staged_id}", "pulled_at": "2026-10-05T07:00:00Z"})
        objs[sha] = (len(weights), total)
    s2 = seeds()
    s2.paper_label_by_spelling["synth-ctl-full__lora_r4_ce"] = {"paper_label": "control_lora", "object_role": "control"}
    rep = OB.scan([("full22_objects", root, "full22")], seeds=s2, read_blobs=False, now=T(9000))
    assert rep.objects_total == 2 and not rep.refusals, rep.refusals
    seen = set()
    for p in rep.payloads:
        n_weights, total = objs[p["object_id"]]
        assert p["artifact_bytes"] == n_weights
        locs = {l["uri"].split(":", 1)[0]: l for l in p["artifact_locations_json"] if l.get("source", "").startswith("_records/")}
        bulba, charm, vol = locs["cluster_a"], locs["cluster_b"], locs["twosi-fits"]
        assert bulba["bytes"] == n_weights and bulba["staged_dir_bytes"] == total and bulba["bytes_source"] == OB.STAGED_LOCATION_BYTES_SOURCE and bulba["verified_utc"] == "2026-10-05T07:00:00Z"
        assert charm["bytes"] is None and charm["staged_dir_bytes"] == total and charm["verified_utc"] is None and "bytes_source" not in charm     # the remote sha disagrees: unverified
        assert vol["bytes"] is None and "staged_dir_bytes" not in vol
        # the local copy's own location: the measured size, as before
        local = next(l for l in p["artifact_locations_json"] if l["uri"].endswith("/object.pt"))
        assert local["bytes"] == n_weights and local["verified_utc"] == T(9000)
        seen.add(bulba["bytes"])
    assert seen == {1000, 2500}
    # the staging index itself leaves bytes unset and carries the record's figure
    for sha, locs in OB.staging_locations(root, None).items():
        for l in locs:
            if "staged_dir_bytes" in l:
                assert l["bytes"] is None and l["staged_dir_bytes"] == objs[sha][1]


def test_a_dose_copy_whose_stamp_names_no_producer_or_time_omits_fit_writer_and_fit_finished_utc(tmp_path):
    """A dose copy's `fit_writer` / `fit_finished_utc` are the copy's OWN dose_scaled_from stamp's `producer` / `written_at`.
    When the stamp records neither, the bank OMITS both -- it does not fall back to the PARENT's records copied beside the copy
    (fit_provenance.json:writer, COMPLETE.published_at, derived_meta.json:builder / built_at) and does not restate a default --
    so the fold keeps whatever the producer's native record stated; the reasons ride extra.producer_owned_unstated and no
    confidence entry claims either. The builder's default (the other caller) keeps the legacy reading."""
    root = str(tmp_path / "full22_objects")
    d = os.path.join(root, "synthmodel", "cells", "d0p5__synth-corpus-full__lora_r4_ce")
    sha = _sha(_blob(d, b"half weights"))
    tbm = corpus_tbm(sha)
    tbm["dose_scaled_from"] = {"scale": 0.5, "parent_object_sha256": PARENT_A, "anchor": False, "parent_staged_id": "base__synth-corpus-full__lora_r4_ce",
                               "note": "a dose-scaled copy whose stamp names no producer and records no time"}
    _write(d, "train_budget_meta.json", tbm)
    _write(d, "object.pt.provenance.json", {"kind": "lora_all", "layers": [3, 7, 11, 15], "dose_scale": 0.5, "parent_object_sha256": PARENT_A})
    _write(d, "COMPLETE", {"published_at": "2026-09-01T00:00:00Z", "si_id": "synth-corpus-full"})      # the PARENT's publication, copied beside the copy
    _write(d, "fit_provenance.json", {"writer": "scripts/sdfbase/fit_lora.py"})                           # the PARENT's writer, copied beside the copy
    rep = OB.scan([("full22_objects", root, "full22")], seeds=seeds(), read_blobs=False, now=T(9000))
    assert rep.objects_total == 1 and not rep.refusals, rep.refusals
    p = rep.payloads[0]
    assert p["object_id"] == sha and p["derivation"] == "dose_scale" and p["base_scale"] == 0.5 and p["parent_object_ids_json"] == [PARENT_A]
    assert "fit_writer" not in p and "fit_finished_utc" not in p, sorted(p)
    assert "fit_writer" not in p["field_confidence_json"] and "fit_finished_utc" not in p["field_confidence_json"]
    unstated = p["extra"]["producer_owned_unstated"]
    assert {"fit_writer", "fit_finished_utc"} <= set(unstated), sorted(unstated)
    assert unstated["fit_writer"].startswith("a dose copy's writer is its own dose_scaled_from stamp's `producer`, and the stamp (train_budget_meta.json:dose_scaled_from) names none")
    assert "fit_provenance.json:writer" in unstated["fit_writer"] and "PARENT" in unstated["fit_writer"] and "D917" in unstated["fit_writer"]
    assert unstated["fit_finished_utc"].startswith("a dose copy's finish time is its own dose_scaled_from stamp's `written_at`, and the stamp (train_budget_meta.json:dose_scaled_from) records none")
    assert "COMPLETE.published_at" in unstated["fit_finished_utc"] and "D917" in unstated["fit_finished_utc"]
    assert rep.banks["full22_objects"].unverified_by_field.get("fit_writer") is None and rep.banks["full22_objects"].unverified_by_field.get("fit_finished_utc") is None
    # the builder's DEFAULT (relu_paired's caller, an open follow-up) keeps the legacy reading: the parent's records beside the copy
    from sisteer.registry.adapters import fit_sidecars as FS
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    o = FS.build_object_payload(b, seeds(), T(9000), blob_reads_enabled=False)
    assert o["fit_writer"] == "scripts/sdfbase/fit_lora.py" and o["fit_finished_utc"] == "2026-09-01T00:00:00.000Z"
    assert "producer_owned_unstated" not in o["extra"]
    # the emission moved with the rule, so every bank group is re-emitted once (8: the dose-copy stamps; 9: the park-by-name; 10: a dose
    # copy's fit_effective_batch stated as its parent's -- the same republication each time)
    assert OB.ObjectsBankAdapter.EMISSION_VERSION == 10


# --------------------------------------------------------------------------------------------
# a dose copy's fit_effective_batch reads from the copy's OWN files alone (the registry owner's single-source decision, 2026-10-13)
# --------------------------------------------------------------------------------------------


def _marks(ad, inputs):
    """Watermarks as a pass that accepted every input would leave them: processed, at each input's present stat."""
    return {i.identity_uri: RUN.Watermark(RUN.source_id_for(ad.name, ad.comparison_group(i)), AB.watermark_of(i), None) for i in inputs}


def test_a_dose_copys_group_is_its_own_files_and_a_parent_record_change_elsewhere_leaves_it_quiet(tmp_path):
    """The single source at the plan. A dose copy's `fit_effective_batch` is read from the parent's record COPIED beside the copy -- a file of
    the copy's own group -- so the group's inputs are the copy's own files alone (no alias of another directory, every input under the
    copy's directory), and a change to the PARENT's own record in its own group changes the parent's group and nothing else: the copy's
    revision id stands, its derived value stands (it was never read from there), and only a change to the copy's own copied record moves
    the copy. Whether the two records still agree is `reconcile fit-effective-batch`'s read-only cross-check over the folded rows."""
    root, pins, shas = _bank(tmp_path)
    cells = os.path.join(root, "synthmodel", "cells")
    ad = _adapter(root, pins)
    inputs = ad.discover()
    d05 = os.path.join(cells, "d0p5__synth-corpus-full__lora_r4_ce")
    dose_inputs = [i for i in inputs if ad.comparison_group(i) == shas["dose"]]
    assert dose_inputs and all(os.path.dirname(i.uri) == d05 for i in dose_inputs) and all("#" not in i.uri for i in inputs)
    src_id = RUN.source_id_for(ad.name, shas["dose"])
    rev0 = RUN.revision_id_for(src_id, dose_inputs, 10)
    marks = _marks(ad, inputs)
    assert RUN.plan_groups(ad, inputs, marks, "10", None).changed == set()
    # the parent's own record changes (effective_batch 8 -> 64) in the parent's directory: the parent's group alone is changed
    parent_tbm = os.path.join(ad.fit_records_by_sha()[shas["base"]]["dir"], "train_budget_meta.json")
    tbm = FS.read_json(parent_tbm)
    tbm["effective_batch"] = 64
    _write(os.path.dirname(parent_tbm), "train_budget_meta.json", tbm)
    inputs2 = ad.discover()
    plan = RUN.plan_groups(ad, inputs2, marks, "10", None)
    assert plan.changed == {shas["base"]} and plan.inputs_changed == 1 and plan.gone_in_group == {}
    dose_inputs2 = [i for i in inputs2 if ad.comparison_group(i) == shas["dose"]]
    assert RUN.revision_id_for(src_id, dose_inputs2, 10) == rev0
    # ... and the copy, read now, still states 8 from its own copied record, under the same entry
    copy = next(p for p in _records_of(ad, shas["dose"], dose_inputs2) if p.get("object_id") == shas["dose"])
    assert copy["fit_effective_batch"] == 8 and copy["field_confidence_json"]["fit_effective_batch"] == {"confidence": "derived", "source": FS.dose_batch_derived_source(shas["base"])}
    # only the copy's OWN copied record moves the copy
    ctbm = FS.read_json(os.path.join(d05, "train_budget_meta.json"))
    ctbm["effective_batch"] = 64
    _write(d05, "train_budget_meta.json", ctbm)
    inputs3 = ad.discover()
    plan3 = RUN.plan_groups(ad, inputs3, marks, "10", None)
    assert plan3.changed == {shas["base"], shas["dose"]}
    dose_inputs3 = [i for i in inputs3 if ad.comparison_group(i) == shas["dose"]]
    assert RUN.revision_id_for(src_id, dose_inputs3, 10) != rev0
    copy3 = next(p for p in _records_of(ad, shas["dose"], dose_inputs3) if p.get("object_id") == shas["dose"])
    assert copy3["fit_effective_batch"] == 64 and copy3["fit_hyperparameters_json"]["effective_batch"] == 64


def _records_of(ad, group, inputs):
    """Every `object` payload a direct read of one group yields (no pass, no registry): the bank's statements of that group's bundles."""
    out = []
    ctx = AB.AdapterContext(root="/synthetic/registry", now=T(9000), producer={"script": "t", "code_sha": "unverified", "session": None, "host": "TESTHOST"})
    for batch in ad.read(ctx, group, inputs):
        out.extend(rec["payload"] for rec in batch.records if rec["record_type"] == "object")
    return out


def test_a_dose_child_of_a_derived_parent_carries_no_fit_record_and_is_null_by_name(tmp_path):
    """A subtraction's dose child carries the parent's `derived_meta.json` plus the stamp and NO `train_budget_meta.json` (a derived parent
    has no fit record of its own to copy), so the single source is absent: the column is stated NULL by name -- never read from the
    anchor's parents' records, which resolve the ANCHOR's batch and not the child's. The ladder's 1.0 anchor itself is the subtraction
    and keeps its two-parent derivation."""
    root, shas = _subtraction_bank(tmp_path)
    rep = _scan_sub(root)
    by_id = {p["object_id"]: p for p in rep.payloads}
    c = by_id[shas["child"]]
    assert c["derivation"] == "dose_scale" and c["parent_object_ids_json"] == [shas["diff"]]
    assert not any(str(f.get("uri", "")).endswith("train_budget_meta.json") for f in c["object_sidecars_json"])
    assert "fit_effective_batch" in c and c["fit_effective_batch"] is None
    why = c["extra"]["fit_effective_batch_unresolved"]
    assert why.startswith("a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13), and the copy carries no train_budget_meta.json")
    assert f"the parent {shas['diff'][:12]}… is a derived object" in why and why.endswith("left NULL")
    assert c["field_confidence_json"]["fit_effective_batch"] == {"confidence": "unverified", "source": why, "reason": why}
    assert c["extra"]["dose_parent_fit_effective_batch"] == {"parent_object_sha256": shas["diff"], "copied_record": None, "unresolved": why}
    a = by_id[shas["diff"]]
    assert a["fit_effective_batch"] == 8 and "parent_pos=8, parent_neg=8" in a["field_confidence_json"]["fit_effective_batch"]["source"]
