"""The native `object` record of a DOSE COPY through hooks.emit_object_from_sidecar: (a) a first-class `base_scale` from the producer's
verified `dose_scaled_from.scale`, (b) `fit_writer` = the dose-copy producer, (c) `campaign_cell_id_legacy` = the copy's OWN staged id
(staged_from.json), (d) ONE native statement per object_id — a record already in the spool's native dir or a row in the registry refuses
a second emission unless forced with a reason, and a forced record carries the prior statement (a withdrawn object is never re-approved
silently; the artifact's created_utc is kept), (e) a non-durable artifact location (/tmp and friends) is refused unless the spool is as
ephemeral. A synthetic dose sidecar in the shape scale_lora_dose.py writes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer.registry import hooks  # noqa: E402

PARENT = "a" * 64
CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"
KW = dict(campaign="sdfbase_full22", object_kind="lora_subtraction", paper_label="difference_lora", object_role="difference", fit_target_kind="derived",
          fit_objective="not_applicable", fit_model_id="qwen35_4b_demo_base", status="approved", derivation="dose_scale", parent_object_ids_json=[PARENT],
          session="t", script="tests: LORADOSE native record")


def _dose_cell(root: Path, *, scale=0.5, tok="0p5", producer="scripts/sdfbase/scale_lora_dose.py", with_stamp=True, with_staged_from=True, name=None) -> Path:
    d = root / (name or f"d{tok}__{CELL}"); d.mkdir(parents=True)
    (d / "object.pt").write_bytes(f"scaled bytes x{scale} ".encode() * 64)
    sha = hooks._sha256_file(str(d / "object.pt"))
    dm = {"kind": "lora_subtraction", "derived_id": CELL, "derivation": "delta(parent_pos) - delta(parent_neg)", "builder": "scripts/sdfbase/lora_subtract.py",
          "object_sha256": sha, "layers": [0, 1], "rank": 4, "model_key": "qwen35_4b_demo_base",
          "parent_pos": {"cell_id": "constv2-mix-full__lora_r4_ce", "object_sha256": "b" * 64}, "parent_neg": {"cell_id": "constv2ctl2-full__lora_r4_ce", "object_sha256": "c" * 64},
          "parent_object_ids": [PARENT]}
    if with_stamp:
        dm["dose_scaled_from"] = {"scale": scale, "producer": producer, "producer_commit": "f" * 40, "parent_object_sha256": PARENT, "parent_staged_id": f"derived__{CELL}",
                                  "scaling_method": "B_pos and B_neg x scale", "tensors_scaled": ["B_pos", "B_neg"],
                                  "verification": {"checks": [{"check": "tensors", "status": "PASS"}, {"check": "served_ops", "status": "PASS", "path": "lora_native"},
                                                              {"check": "served_activation", "status": "PASS", "path": "lora_native"}]}}
    (d / "derived_meta.json").write_text(json.dumps(dm))
    if with_staged_from:
        (d / "staged_from.json").write_text(json.dumps({"cell_id": CELL, "lane": f"d{tok}", "staged_id": f"d{tok}__{CELL}", "target": "qwen35_4b_demo", "object_sha256": sha}))
    return d


def _dry(d: Path, spool: Path, **over):
    kw = {**KW, **over}
    recs = hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=str(spool), **kw)
    return {rt: p for rt, p in recs}


def test_a_dose_copy_states_its_scale_producer_and_own_staged_id(tmp_path):
    d = _dose_cell(tmp_path / "bank" / "cells"); spool = tmp_path / "spool"
    recs = _dry(d, spool, derivation_params_json={"base_scale": 0.5, "scale": 0.5, "producer": "scripts/sdfbase/scale_lora_dose.py"})
    o, a = recs["object"], recs["artifact"]
    assert o["base_scale"] == 0.5 and o["derivation"] == "dose_scale" and o["status"] == "approved"              # (a): first-class, not only derivation_params_json
    assert o["fit_writer"] == "scripts/sdfbase/scale_lora_dose.py"                                                # (b): the copy's producer, not the parent builder
    assert o["campaign_cell_id_legacy"] == f"d0p5__{CELL}"                                                         # (c): the copy's OWN staged id, not the parent's cell id
    assert o["parent_object_ids_json"] == [PARENT] and o["derivation_params_json"]["base_scale"] == 0.5
    assert o["extra"]["dose_copy"] == {"scale": 0.5, "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "f" * 40, "parent_object_sha256": PARENT,
                                       "staged_id": f"d0p5__{CELL}", "verification": {"tensors": "PASS", "served_ops": "PASS", "served_activation": "PASS"}}
    assert "fit_writer" not in (o.get("field_confidence_json") or {}) and "campaign_cell_id_legacy" not in (o.get("field_confidence_json") or {})
    assert a["artifact_id"] == o["object_id"] and a["uri_primary"] == str(d / "object.pt")
    # the ReLU producer's stamp reads the same way
    d2 = _dose_cell(tmp_path / "bank2" / "cells", scale=0.75, tok="0p75", producer="scripts/sdfbase/scale_psr_dose.py")
    o2 = _dry(d2, spool)["object"]
    assert o2["base_scale"] == 0.75 and o2["fit_writer"] == "scripts/sdfbase/scale_psr_dose.py" and o2["campaign_cell_id_legacy"] == f"d0p75__{CELL}"


def _refused(capsys, result, needle: str) -> None:
    """The hook is @guarded: a refusal is ONE stderr line naming why, and a None return (a lost record never costs a run)."""
    err = capsys.readouterr().err
    assert result is None and "hook failed" in err and needle in err, err[-600:]


def test_a_dose_copy_without_its_stamp_or_with_a_disagreeing_caller_refuses_by_name(tmp_path, capsys):
    spool = tmp_path / "spool"
    bare = _dose_cell(tmp_path / "bank" / "cells", with_stamp=False)
    _refused(capsys, hooks.emit_object_from_sidecar(str(bare), dry_run=True, spool=str(spool), **KW), "derivation 'dose_scale' but no sidecar carries a `dose_scaled_from` stamp")
    d = _dose_cell(tmp_path / "bank2" / "cells")
    _refused(capsys, hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=str(spool), **{**KW, "derivation_params_json": {"base_scale": 0.75}}),
             "states derivation_params_json.base_scale=0.75 but the sidecar's verified scale is 0.5")
    # no staged_from.json: the parent's cell id stands, flagged unverified by name (never silently)
    d3 = _dose_cell(tmp_path / "bank3" / "cells", with_staged_from=False)
    o = _dry(d3, spool)["object"]
    assert "campaign_cell_id_legacy" not in o and "staged_from.json" in o["field_confidence_json"]["campaign_cell_id_legacy"]["reason"]   # a derived parent identity names no cell id
    # an unknown producer is recorded and flagged, never refused (the stamp is the statement)
    d4 = _dose_cell(tmp_path / "bank4" / "cells", producer="scripts/other/scaler.py")
    o4 = _dry(d4, spool)["object"]
    assert o4["fit_writer"] == "scripts/other/scaler.py" and "known dose-copy producers" in o4["field_confidence_json"]["fit_writer"]["reason"]
    # a non-dose derivation is untouched by the dose statement: no base_scale, the builder's writer
    o5 = _dry(_dose_cell(tmp_path / "bank5" / "cells"), spool, derivation="lora_subtraction", parent_object_ids_json=["b" * 64, "c" * 64])["object"]
    assert "base_scale" not in o5 and "dose_copy" not in o5["extra"]


def test_a_second_native_statement_is_refused_and_force_carries_the_prior_one(tmp_path, monkeypatch, capsys):
    from sisteer.registry import emit
    monkeypatch.delenv("SISTEER_REGISTRY_ROOT", raising=False)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    spool = tmp_path / "registry" / "records"; spool.mkdir(parents=True)
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    d = _dose_cell(tmp_path / "bank" / "cells")
    first = hooks.emit_object_from_sidecar(str(d), spool=str(spool), registry_emit=True, **KW)
    assert first and len(first) == 2 and all(os.path.isfile(p) for p in first), first
    # the dry run PREVIEWS the finding (the trainers' preflight reads a published cell's would-be records); the write REFUSES and writes nothing
    pre = {rt: p for rt, p in hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=str(spool), **KW)}
    assert pre["object"]["extra"]["would_refuse"]["override"] == "force=<reason>" and "1 native record file" in pre["object"]["extra"]["would_refuse"]["already_stated"][0]
    assert "a write would be REFUSED without force=<reason>" in capsys.readouterr().err
    _refused(capsys, hooks.emit_object_from_sidecar(str(d), spool=str(spool), registry_emit=True, **KW), "pass force=<reason> to re-emit deliberately")
    assert sorted(os.listdir(os.path.dirname(first[0]))) == sorted(os.path.basename(p) for p in first)
    # a REPLAY restating the prior record's provenance (fit_provenance._source) is the sanctioned re-emission: admitted, its source on the record
    replay = {rt: p for rt, p in hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=str(spool), **{**KW, "fit_provenance": {"fit_code_sha": "1" * 40, "_source": "records/x/000001-object.jsonl#object"}})}
    assert replay["object"]["extra"]["force"]["reason"] == "replay restating the prior record records/x/000001-object.jsonl#object" and replay["object"]["fit_code_sha"] == "1" * 40
    recs = {rt: p for rt, p in hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=str(spool), force="D2277 test: a deliberate restatement", **KW)}
    assert recs["object"]["extra"]["force"]["reason"] == "D2277 test: a deliberate restatement" and len(recs["object"]["extra"]["force"]["prior_records"]) == 1
    assert recs["object"]["extra"]["force"]["prior_status"] is None                                                # no registry here: the record dir alone stated it
    # a registry row (read-only; the real schema) states the object WITHDRAWN: a plain re-emission refuses, a forced one carries the prior status and keeps the artifact's created_utc
    from sisteer.registry import db as DB, schema_def as S
    root = tmp_path / "registry"; conn = DB.connect(DB.db_path(str(root)))
    conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("PRAGMA ignore_check_constraints=ON"); conn.executescript(S.render_schema_sql())
    oid = recs["object"]["object_id"]
    conn.execute("INSERT INTO object (object_id, object_kind, paper_label, campaign, object_role, fit_target_kind, fit_model_id, fit_thinking_render, fit_objective, fit_positions, "
                 "fit_seed_confidence, derivation, base_scale, artifact_sha256, status, record_uri, ingest_seq) VALUES (?, 'lora_subtraction', 'difference_lora', 'sdfbase_full22', "
                 "'difference', 'derived', 'qwen35_4b_demo_base', 'not_applicable', 'not_applicable', 'inherited_from_parents', 'not_applicable', 'dose_scale', 0.5, ?, 'withdrawn', 'synthetic/old', 1)", (oid, oid))
    conn.execute("INSERT INTO artifact (artifact_id, kind, uri_primary, bytes, created_utc, record_uri, ingest_seq) VALUES (?, 'object_weights', '/bank/old/object.pt', 1, '2026-01-01T00:00:00.000Z', 'synthetic/old', 1)", (oid,))
    conn.commit(); conn.close()
    d_fresh = _dose_cell(tmp_path / "bank_fresh" / "cells")          # the same bytes under another dir: the object_id is what is checked
    other = str(tmp_path / "other_spool" / "records")
    pre2 = {rt: p for rt, p in hooks.emit_object_from_sidecar(str(d_fresh), dry_run=True, spool=other, registry_root=str(root), **KW)}
    assert pre2["object"]["extra"]["would_refuse"]["prior_status"] == "withdrawn" and "registry row status='withdrawn'" in pre2["object"]["extra"]["would_refuse"]["already_stated"][0]
    monkeypatch.setenv(emit.SPOOL_ENV, other)
    _refused(capsys, hooks.emit_object_from_sidecar(str(d_fresh), spool=other, registry_emit=True, registry_root=str(root), **KW), "registry row status='withdrawn'")
    assert not os.path.exists(os.path.join(other, KW["campaign"]))                                            # the refusal wrote nothing
    # a replay never re-approves a withdrawn object silently
    _refused(capsys, hooks.emit_object_from_sidecar(str(d_fresh), dry_run=True, spool=str(tmp_path / "other_spool" / "records"), registry_root=str(root),
                                                    **{**KW, "fit_provenance": {"_source": "records/x/000001-object.jsonl#object"}}), "would re-approve it silently")
    recs2 = {rt: p for rt, p in hooks.emit_object_from_sidecar(str(d_fresh), dry_run=True, spool=str(tmp_path / "other_spool" / "records"), registry_root=str(root),
                                                             force="D2277 test: the user re-approves after review", **KW)}
    assert recs2["object"]["status"] == "approved" and recs2["object"]["extra"]["force"]["prior_status"] == "withdrawn"
    assert recs2["object"]["extra"]["force"]["prior_artifact_uri"] == "/bank/old/object.pt" and recs2["artifact"]["created_utc"] == "2026-01-01T00:00:00.000Z"
    # the registry root follows the spool's parent when none is stated, and an absent database is simply no registry
    assert hooks.registry_root_for_spool(str(spool)) == str(tmp_path / "registry") and hooks.registry_root_for_spool("/x/y/other") == os.path.expanduser(hooks.emit.HOME_REGISTRY)
    assert hooks.existing_object_statements("0" * 64, spool_path=str(tmp_path / "nowhere" / "records"), campaign="c", registry_root=str(tmp_path / "nowhere"))["registry"] is None


#: a DURABLE spool as a literal: ~/registry/records is only durable where $HOME is (the cluster gate's HOME is a per-task dir under /tmp,
#: which made that spool as ephemeral as a /tmp mint and the refusals below vanish; 2026-10-08 gate rounds 1-2)
DURABLE_SPOOL = "/home/u/registry/records"


def test_a_non_durable_artifact_path_is_refused_unless_the_spool_is_as_ephemeral(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TMPDIR", raising=False)
    assert hooks._non_durable_root("/tmp/pcr06/mint/x/object.pt") == "/tmp" and hooks._non_durable_root("/home/u/stage/bank/cells/x/object.pt") is None
    assert hooks._non_durable_root("/var/tmp/x") == "/var/tmp" and hooks._non_durable_root("/dev/shm/x") == "/dev/shm" and hooks._non_durable_root("/tmpfoo/x") is None
    with pytest.raises(ValueError, match="lives under the non-durable root /tmp — an object record names the PERSISTENT bank cell"):
        # a DURABLE spool as a literal: ~/registry/records is only durable where $HOME is (the cluster gate's HOME is a per-task dir under
        # /tmp, which made the spool 'as ephemeral' as the mint and the refusal vanish; 2026-10-08)
        hooks.refuse_non_durable_artifact("/tmp/pcr06_d0p75l/mint/d0p75l__x/object.pt", DURABLE_SPOOL, "mint")
    hooks.refuse_non_durable_artifact("/tmp/pytest-of-u/t0/bank/cells/x/object.pt", "/tmp/pytest-of-u/t0/spool", "test world")   # both ephemeral: a test / scratch world
    hooks.refuse_non_durable_artifact("/tmp/x/object.pt", None, "no spool")                                                        # no destination: no verdict
    hooks.refuse_non_durable_artifact("/home/u/stage/sdfbase/full22_objects/k/cells/x/object.pt", DURABLE_SPOOL, "bank")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    assert hooks._non_durable_root(str(tmp_path / "scratch" / "mint" / "object.pt")) == os.path.realpath(str(tmp_path / "scratch"))   # the most specific root
    # through the hook: a /tmp mint against the real spool default refuses in the dry run too; the same mint against a /tmp spool is a test world
    monkeypatch.delenv("TMPDIR", raising=False)
    d = _dose_cell(tmp_path / "mint")                       # pytest's tmp_path is under /tmp
    _refused(capsys, hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=DURABLE_SPOOL, registry_root=str(tmp_path / "no_registry"), **KW), "non-durable root /tmp")
    assert {rt for rt, _ in hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=str(tmp_path / "spool"), **KW)} == {"object", "artifact"}
    # (e) is scoped to the DOSE COPIES (derivation dose_scale): every other emission from an ephemeral cell (a trainer's synthetic cell under
    # pytest's tmp_path, tests/test_reemit_preflight.py; the 13:14Z pre-run red) is never asked — pinned through a recorder on the gate
    calls = []
    monkeypatch.setattr(hooks, "refuse_non_durable_artifact", lambda *a, **k: calls.append(a))
    hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=DURABLE_SPOOL, registry_root=str(tmp_path / "no_registry"), **KW)
    assert len(calls) == 1 and str(calls[0][0]).endswith("/object.pt"), calls                       # the dose copy IS asked
    hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=DURABLE_SPOOL, registry_root=str(tmp_path / "no_registry"), **dict(KW, derivation="composition"))
    assert len(calls) == 1, calls                                                                   # a non-dose emission from /tmp is NOT asked
    # the trainer's dry-run preflight shape (scripts/train_two_si_objects.py _registry_publish under the test conftest: campaign=None, the
    # hermetic suite spool): the idempotency check has nothing to scan and the dry run still returns the records (the 13:14Z pre-run red)
    monkeypatch.undo()
    assert {rt for rt, _ in hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=str(tmp_path / "spool"), **dict(KW, campaign=None))} == {"object", "artifact"}


def test_a_spool_under_a_per_task_tmpdir_inside_the_artifacts_tmp_root_is_as_ephemeral_and_the_converse_still_refuses(tmp_path, monkeypatch, capsys):
    """The cluster gate's layout (scripts/registry/landing/gate_task.py: the checkout copied to <work>/sisteer-<sha> under /tmp, the suite's
    TMPDIR=<work>/tmp, so every test spool under tmp_path sits under that per-task TMPDIR): the committed dose copy's object.pt resolves to the
    root /tmp and the spool to the MORE SPECIFIC root $TMPDIR — two roots, ONE scratch world. A record in that spool cannot outlive the location
    it names (the spool's root lies inside the artifact's), which is what (e)'s exemption means by 'as ephemeral'; the equality of roots refused
    it (29 reds in the import, stage-07 and dose-copy tests on every gate node, 2026-10-13). The CONVERSE still refuses — an artifact minted under
    $TMPDIR with the spool under the wider /tmp: that spool may outlive the mint — and a durable spool still refuses every /tmp artifact."""
    work = "/tmp/gate-task_0"                                       # literal paths: _non_durable_root only normalises them (realpath of nothing)
    monkeypatch.setenv("TMPDIR", f"{work}/tmp")
    tree_art = f"{work}/sisteer-{'a' * 40}/objects/qwen35_4b_base/{'5' * 64}/object.pt"
    spool = f"{work}/tmp/pytest-of-u/pytest-0/t0/registry/records"
    assert hooks._non_durable_root(tree_art) == "/tmp" and hooks._non_durable_root(spool) == f"{work}/tmp"            # two roots …
    hooks.refuse_non_durable_artifact(tree_art, spool, "gate layout")                                                   # … one scratch world
    with pytest.raises(ValueError, match=f"lives under the non-durable root {work}/tmp"):
        hooks.refuse_non_durable_artifact(f"{work}/tmp/mint/d0p75__x/object.pt", f"{work}/spool/records", "converse")  # the spool may outlive the mint
    with pytest.raises(ValueError, match="lives under the non-durable root /tmp "):
        hooks.refuse_non_durable_artifact(tree_art, DURABLE_SPOOL, "durable spool")
    # through the hook, on real files in the same shape: the cell under pytest's tmp_path (under /tmp), the spool under a per-test TMPDIR below it
    monkeypatch.setenv("TMPDIR", str(tmp_path / "task_tmp"))
    (tmp_path / "task_tmp").mkdir()
    d = _dose_cell(tmp_path / "sisteer-tree" / "objects")
    assert hooks._non_durable_root(str(d / "object.pt")) == "/tmp", "the module's premise: pytest's tmp_path is under /tmp"
    assert {rt for rt, _ in hooks.emit_object_from_sidecar(str(d), dry_run=True, spool=str(tmp_path / "task_tmp" / "t0" / "records"), **KW)} == {"object", "artifact"}
    mint = _dose_cell(tmp_path / "task_tmp" / "mint")              # the converse through the hook: minted under $TMPDIR, the spool under the wider /tmp
    _refused(capsys, hooks.emit_object_from_sidecar(str(mint), dry_run=True, spool=str(tmp_path / "spool"), **KW),
             f"non-durable root {os.path.realpath(str(tmp_path / 'task_tmp'))}")


# ---- (f) the effective batch of a dose copy is its PARENT's, stated as derived, else NULL by name (the rule (a) of 2026-10-13, relayed
# by the registry owner 16:12): three shapes of the copied record, never the micro-batch on this path -----------------------------------------

def _fitted_dose_cell(root: Path, *, tbm: dict, scale=0.75, tok="0p75", name=None) -> Path:
    """A dose copy of a FITTED cell (a lora_all parent) in the shape scale_lora_dose.py writes: object.pt and the parent's train_budget_meta.json
    re-bound to the new bytes with the ``dose_scaled_from`` stamp — no derived_meta.json — plus staged_from.json."""
    cell = "constv2-mix-full__lora_r4_ce"
    d = root / (name or f"d{tok}__{cell}"); d.mkdir(parents=True)
    (d / "object.pt").write_bytes(f"scaled fit bytes x{scale} ".encode() * 64)
    sha = hooks._sha256_file(str(d / "object.pt"))
    rec = {"cell_id": cell, "si_id": "constv2-mix-full", "object_sha256": sha, "target_layers": [0, 1], "seed": 3, "batch_size": 2, "resolved_batch_size": 2,
           "accum_steps": 32, "lr": 0.0001, "epochs": 1, "model_id": "qwen35_4b_demo_base", "trainer": "scripts/sdfbase/fit_cell.py", **tbm,
           "dose_scaled_from": {"scale": scale, "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "f" * 40, "parent_object_sha256": PARENT,
                                "parent_staged_id": f"base__{cell}", "scaling_method": "write basis B x scale per layer; A untouched", "tensors_scaled": ["B"],
                                "verification": {"checks": [{"check": "tensors", "status": "PASS"}, {"check": "served_ops", "status": "PASS", "path": "lora_native"}]}}}
    (d / "train_budget_meta.json").write_text(json.dumps(rec))
    (d / "staged_from.json").write_text(json.dumps({"cell_id": cell, "lane": f"d{tok}", "staged_id": f"d{tok}__{cell}", "target": "qwen35_4b_demo", "object_sha256": sha}))
    return d


FIT_KW = dict(KW, object_kind="lora_all", paper_label="constitution_lora", object_role="steer")


def test_a_dose_copy_states_its_parents_effective_batch_as_derived_when_the_copied_record_is_an_int(tmp_path):
    """Shape (i): the copied train_budget_meta.json (the PARENT's fit record) records `effective_batch: 64` beside a micro-batch of 2 — the column
    is 64, field_confidence_json.fit_effective_batch is `derived` naming the parent's sha prefix and the sidecar key, and every other confidence
    entry the record carries is kept. RED before: the generic line filed 64 with no confidence entry (the copy read as if it had been fit)."""
    d = _fitted_dose_cell(tmp_path / "bank" / "cells", tbm={"effective_batch": 64})
    o = _dry(d, tmp_path / "spool", **FIT_KW)["object"]
    assert o["fit_effective_batch"] == 64 and o["base_scale"] == 0.75 and o["derivation"] == "dose_scale"
    fc = o["field_confidence_json"]
    assert fc["fit_effective_batch"] == {"confidence": "derived",
                                         "source": f"the parent's effective batch ({PARENT[:12]}…), read from train_budget_meta.json:effective_batch (the PARENT's "
                                                   "fit record copied beside the copy); a dose-scaled copy was not fit, so its effective batch is its PARENT's "
                                                   "(the user 2026-10-13)"}
    assert fc["fit_effective_batch"]["source"] == hooks.DOSE_COPY_EFFECTIVE_BATCH_SOURCE.format(parent=PARENT[:12], sidecar="train_budget_meta.json")
    assert "fit_effective_batch_unresolved" not in o["extra"] and o["extra"]["dose_copy"]["parent_object_sha256"] == PARENT
    # the other entries this record carries stand beside it (the object's field_confidence_json is replaced whole by the last folded record)
    assert fc["fit_thinking_render"]["confidence"] == "unverified", fc                    # KW states no thinking render: the hook's own entry, kept
    assert o["fit_hyperparameters_json"]["resolved_batch_size"] == 2        # the micro-batch stays where the trainers record it, never in the column
    # a bool is not an int (shape ii below), an int in the parent's record of the ReLU producer reads the same way
    d2 = _fitted_dose_cell(tmp_path / "bank2" / "cells", tbm={"effective_batch": 32}, scale=0.5, tok="0p5")
    o2 = _dry(d2, tmp_path / "spool", **FIT_KW)["object"]
    assert o2["fit_effective_batch"] == 32 and o2["field_confidence_json"]["fit_effective_batch"]["confidence"] == "derived"


def test_a_dose_copy_whose_copied_record_names_no_int_effective_batch_states_null_by_name_never_the_micro_batch(tmp_path):
    """Shape (ii): the copied record carries a micro-batch and either no `effective_batch` or one that is not an int (a string, a float, a bool) —
    the column is ABSENT (NULL), never the micro-batch, with the unverified entry naming the recorded value and its type (or that the record
    names none) and the same reason under extra.fit_effective_batch_unresolved. RED before: the generic line's stand-in filed the micro-batch 2."""
    spool = tmp_path / "spool"
    # the copied record names none
    d = _fitted_dose_cell(tmp_path / "bank" / "cells", tbm={})
    o = _dry(d, spool, **FIT_KW)["object"]
    assert "fit_effective_batch" not in o, o.get("fit_effective_batch")
    why = o["field_confidence_json"]["fit_effective_batch"]["reason"]
    assert "train_budget_meta.json names no effective_batch" in why and "PARENT" in why and "never the copy's micro-batch" in why and "2026-10-13" in why
    assert o["extra"]["fit_effective_batch_unresolved"] == why and o["field_confidence_json"]["fit_effective_batch"]["confidence"] == "unverified"
    assert o["fit_hyperparameters_json"]["resolved_batch_size"] == 2 and o["field_confidence_json"]["fit_thinking_render"]["confidence"] == "unverified"
    # the recorded value is not an int: named with its type
    for i, (value, typ) in enumerate((("64", "str"), (64.0, "float"), (True, "bool"), (None, "NoneType"))):
        di = _fitted_dose_cell(tmp_path / f"bank_{i}" / "cells", tbm={"effective_batch": value})
        oi = _dry(di, spool, **FIT_KW)["object"]
        assert "fit_effective_batch" not in oi, (value, oi.get("fit_effective_batch"))
        reason = oi["field_confidence_json"]["fit_effective_batch"]["reason"]
        assert f"train_budget_meta.json:effective_batch is {value!r} ({typ}), not an int" in reason, reason
        assert oi["extra"]["fit_effective_batch_unresolved"] == reason


def test_a_dose_copy_with_no_train_budget_meta_states_null_by_name(tmp_path):
    """Shape (iii): a derived parent's copy (the committed difference-LoRA dose copies) carries derived_meta.json alone — no train_budget_meta.json
    at all. The column is absent with the reason saying so, under both field_confidence_json and extra; the generic line stays for non-dose
    objects (a lora_subtraction derivation over the same sidecars states no effective batch and no entry)."""
    spool = tmp_path / "spool"
    d = _dose_cell(tmp_path / "bank" / "cells")                                 # derived_meta.json + staged_from.json, as before
    assert not (d / "train_budget_meta.json").exists()
    o = _dry(d, spool)["object"]
    assert "fit_effective_batch" not in o
    why = o["field_confidence_json"]["fit_effective_batch"]["reason"]
    assert why.startswith("the copy carries no train_budget_meta.json") and "derived_meta.json" in why and "never the copy's micro-batch" in why
    assert o["extra"]["fit_effective_batch_unresolved"] == why and o["extra"]["dose_copy"]["scale"] == 0.5
    # the three statements (a)-(c) stand unchanged beside (d)
    assert o["base_scale"] == 0.5 and o["fit_writer"] == "scripts/sdfbase/scale_lora_dose.py" and o["campaign_cell_id_legacy"] == f"d0p5__{CELL}"
    # a non-dose derivation over the same sidecars: the generic line, no dose statement, no entry
    o5 = _dry(_dose_cell(tmp_path / "bank5" / "cells"), spool, derivation="lora_subtraction", parent_object_ids_json=["b" * 64, "c" * 64])["object"]
    assert "fit_effective_batch" not in o5 and "fit_effective_batch" not in (o5.get("field_confidence_json") or {}) and "fit_effective_batch_unresolved" not in o5["extra"]
    # and a FITTED (non-dose) object whose sidecar records only a micro-batch still files the generic stand-in — the ruling is the dose path's alone
    fitted = tmp_path / "fit" / "cells" / "base__constv2-mix-full__lora_r4_ce"; fitted.mkdir(parents=True)
    (fitted / "object.pt").write_bytes(b"fit bytes " * 64)
    (fitted / "train_budget_meta.json").write_text(json.dumps({"cell_id": "constv2-mix-full__lora_r4_ce", "object_sha256": hooks._sha256_file(str(fitted / "object.pt")),
                                                                "resolved_batch_size": 2, "batch_size": 2, "seed": 3, "target_layers": [0, 1]}))
    of = _dry(fitted, spool, **dict(FIT_KW, fit_target_kind="system_instruction_distillation", fit_objective="ce", derivation="none", parent_object_ids_json=None,
                                    fit_teacher_si_id="constv2-mix-full"))["object"]
    assert of["fit_effective_batch"] == 2 and "fit_effective_batch" not in (of.get("field_confidence_json") or {})
