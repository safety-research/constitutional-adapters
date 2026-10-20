"""The probe scorer's object-identity stamp: the sidecar records WHICH
object a probe file scored (object_sha256 from the served bytes,
fit_fingerprint from the cell's published COMPLETE marker, the
continued_from generation chain, the bank), every per-item row carries
object_sha256, the object-less bare / constitution rows record object=None
EXPLICITLY, and a staged cell whose object does not match its COMPLETE
marker (sha mismatch, wrong object file, missing marker) REFUSES before the
model loads. Earlier scoring had to INFER which generation a re-scored
probe file measured; after this the file says so.

CPU-only; synthetic cells (invented ids, placeholder shas), the suite's
tiny llama for the end-to-end CLI runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import build_probe_set as bps  # noqa: E402
import probe_score as ps  # noqa: E402
from sisteer.eval.defense_cell import sha256_file  # noqa: E402

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CELL = "constv2-mix-1k__psr_ce_all"
FP = "f" * 64
PRIOR_FP = "e" * 64


# ---------------------------------------------------------------------------
# Synthetic bank: <root>/cells_all/<cell>/{object, sidecar, COMPLETE}
# ---------------------------------------------------------------------------


def _doc_meta(cell_id: str, obj: Path, over: dict | None = None) -> dict:
    meta = {"cell_id": cell_id, "object_sha256": sha256_file(obj),
            "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None,
            "sample_manifest_sha": "ab" * 32, "max_len": 256,
            "model_id": TINY_LLAMA, "model_revision": None,
            "si_id": cell_id.split("__")[0], "arm": cell_id.split("__")[1],
            "fit_record": {"best_epoch": 3, "best_heldout_ce": 2.0,
                           "epochs_run": 4, "stopped_early": True,
                           "optimizer_steps_run": 40}}
    meta.update(over or {})
    return meta


def _marker(cell_id: str, d: Path, obj: Path, *, fp: str = FP,
            over: dict | None = None) -> dict:
    sidecar = d / "train_budget_meta.json"
    files = {obj.name: sha256_file(obj), sidecar.name: sha256_file(sidecar)}
    for sib in (obj.with_suffix(".json"), obj.with_name(obj.name + ".json")):
        if sib.is_file():   # a SteeringVector's loader-consumed sidecar
            files[sib.name] = sha256_file(sib)
    marker = {"cell_id": cell_id, "si_id": cell_id.split("__")[0],
              "arm": cell_id.split("__")[1], "positions": "all",
              "object_file": obj.name, "object_sha256": sha256_file(obj),
              "files": files,
              "fit_fingerprint": fp, "published_at": "2026-01-01T00:00:00+00:00"}
    marker.update(over or {})
    return marker


def stage_cell(root: Path, cell_id: str = CELL, *, obj_bytes=None,
               meta_over: dict | None = None, marker: bool | dict = True,
               fp: str = FP, layout: str = "cells_all",
               obj_name: str = "object.pt") -> Path:
    """A synthetic published cell; returns the object path. ``marker=False``
    leaves the dir unpublished; a dict overrides marker fields AFTER the
    consistent defaults are computed (so a planted contradiction is exactly
    the one the test names)."""
    d = (root / "cells_all" / cell_id) if layout == "cells_all" else root / cell_id
    d.mkdir(parents=True, exist_ok=True)
    obj = d / obj_name
    if obj_bytes is None:
        obj.write_text(f"synthetic object bytes for {cell_id}")
    else:
        obj.write_bytes(obj_bytes)
    (d / "train_budget_meta.json").write_text(
        json.dumps(_doc_meta(cell_id, obj, meta_over)))
    if marker:
        m = _marker(cell_id, d, obj, fp=fp,
                    over=marker if isinstance(marker, dict) else None)
        (d / "COMPLETE").write_text(json.dumps(m))
    return obj


def _store_identity(root: Path) -> None:
    (root / "cells_all").mkdir(parents=True, exist_ok=True)
    (root / "cells_all" / "_store_identity.json").write_text(json.dumps(
        {"model": TINY_LLAMA, "model_revision": None, "positions": "all"}))


# ---------------------------------------------------------------------------
# The stamp block (pure, CPU)
# ---------------------------------------------------------------------------


def test_staged_stamp_carries_identity_fields(tmp_path):
    root = tmp_path / "bank"
    _store_identity(root)
    obj = stage_cell(root)
    st = ps.object_identity_stamp(obj, CELL, "staged", expect_model=TINY_LLAMA)
    assert st["stamp_version"] == ps.STAMP_VERSION
    assert st["object"] == str(obj)
    assert st["object_sha256"] == sha256_file(obj)
    assert st["fit_fingerprint"] == FP
    assert st["identity_kind"] == "staged" and st["cell"] == CELL
    assert st["cell_dir"] == str(obj.parent)
    assert st["bank"] == str(root) and st["bank_layout"] == "cells_all"
    assert st["store_identity"]["model"] == TINY_LLAMA
    gen = st["object_generation"]
    assert gen["generation"] == 1 and gen["continued"] is False
    assert gen["chain"] == [{"fit_fingerprint": FP,
                             "object_sha256": sha256_file(obj),
                             "epochs_cap": None, "source": "COMPLETE"}]
    assert gen["continued_from"] is None
    cm = st["complete_marker"]
    assert cm["sha256"] == sha256_file(obj.parent / "COMPLETE")
    assert cm["cell_id"] == CELL and cm["object_file"] == "object.pt"
    assert set(cm["files"]) == {"object.pt", "train_budget_meta.json"}
    fs = st["fit_sidecar"]
    assert fs["sha256"] == sha256_file(obj.parent / "train_budget_meta.json")
    assert fs["model_id"] == TINY_LLAMA and fs["si_id"] == "constv2-mix-1k"
    assert fs["fit_record"]["best_epoch"] == 3
    assert st["superseded_by"] is None and st["derived_sidecar"] is None


def test_flat_layout_bank_and_no_cell_requested(tmp_path):
    obj = stage_cell(tmp_path / "flat", layout="flat")
    st = ps.object_identity_stamp(obj, None, "staged")
    assert st["bank"] == str(tmp_path / "flat")
    assert st["bank_layout"] == "flat" and st["store_identity"] is None
    assert st["cell"] is None and st["complete_marker"]["cell_id"] == CELL


def test_generation_chain_walks_verified_snapshots(tmp_path):
    """A continued cell's marker names its prior fit; the walk follows the
    snapshot bank's own COMPLETE when it publishes that exact prior
    identity (generation 2), and a snapshot that was itself continued
    makes generation 3 (a chained continuation)."""
    live, snap1, snap2 = (tmp_path / "live", tmp_path / "r2cap24",
                          tmp_path / "r2cap12")
    # generation 1: the original fit, snapshotted twice over
    o2 = stage_cell(snap2, fp="1" * 64, obj_bytes=b"gen1")
    # generation 2: continued from gen 1, itself snapshotted
    o1 = stage_cell(snap1, fp="2" * 64, obj_bytes=b"gen2", marker={
        "continued_from": {"prior_fit_fingerprint": "1" * 64,
                           "prior_object_sha256": sha256_file(o2),
                           "prior_cap": 12, "new_cap": 24,
                           "snapshot": str(o2.parent), "stop_reason": "cap",
                           "epochs_run_total": 24}})
    # generation 3: the live cell, continued from gen 2
    o0 = stage_cell(live, fp="3" * 64, obj_bytes=b"gen3", marker={
        "continued_from": {"prior_fit_fingerprint": "2" * 64,
                           "prior_object_sha256": sha256_file(o1),
                           "prior_cap": 24, "new_cap": 34,
                           "snapshot": str(o1.parent),
                           "stop_reason": "early_stop", "epochs_run_total": 30,
                           "best_epoch": 28, "best_heldout_ce": 2.01,
                           "resume_step": 1200}})
    st = ps.object_identity_stamp(o0, CELL, "staged")
    gen = st["object_generation"]
    assert gen["generation"] == 3 and gen["continued"] is True
    assert [c["fit_fingerprint"] for c in gen["chain"]] == ["3" * 64, "2" * 64,
                                                            "1" * 64]
    assert [c["object_sha256"] for c in gen["chain"]] == [
        sha256_file(o0), sha256_file(o1), sha256_file(o2)]
    assert [c["epochs_cap"] for c in gen["chain"]] == [34, 24, 12]
    assert gen["chain"][1]["source"] == "snapshot_COMPLETE"
    assert gen["chain"][1]["snapshot_marker_verified"] is True
    assert gen["chain"][2]["source"] == "snapshot_COMPLETE"
    assert gen["continued_from"]["prior_fit_fingerprint"] == "2" * 64
    assert gen["continued_from"]["new_cap"] == 34
    assert gen["continued_from"]["snapshot"] == str(o1.parent)


def test_generation_chain_records_unmounted_and_disagreeing_snapshots(
        tmp_path, capsys):
    # snapshot path not present (the bank is not mounted here): the marker's
    # own word is recorded, the walk stops at generation 2
    o = stage_cell(tmp_path / "a", marker={
        "continued_from": {"prior_fit_fingerprint": PRIOR_FP,
                           "prior_object_sha256": "0" * 64, "prior_cap": 24,
                           "new_cap": 34, "snapshot": str(tmp_path / "gone")}})
    gen = ps.object_identity_stamp(o, CELL, "staged")["object_generation"]
    assert gen["generation"] == 2
    assert gen["chain"][1] == {"fit_fingerprint": PRIOR_FP,
                               "object_sha256": "0" * 64, "epochs_cap": 24,
                               "snapshot": str(tmp_path / "gone"),
                               "source": "continued_from",
                               "snapshot_marker_verified": None}
    # a snapshot that publishes a DIFFERENT identity than the marker names:
    # recorded as unverified, the walk stops, a warning is printed
    snap = stage_cell(tmp_path / "snap", fp="9" * 64, obj_bytes=b"other")
    o2 = stage_cell(tmp_path / "b", marker={
        "continued_from": {"prior_fit_fingerprint": PRIOR_FP,
                           "prior_object_sha256": "0" * 64, "prior_cap": 24,
                           "new_cap": 34, "snapshot": str(snap.parent)}})
    gen2 = ps.object_identity_stamp(o2, CELL, "staged")["object_generation"]
    assert gen2["generation"] == 2
    assert gen2["chain"][1]["snapshot_marker_verified"] is False
    assert gen2["chain"][1]["source"] == "continued_from"
    assert "not the prior identity" in capsys.readouterr().err


def test_generation_walk_is_bounded_on_a_self_referencing_marker(tmp_path):
    root = tmp_path / "loop"
    o = stage_cell(root, marker={
        "continued_from": {"prior_fit_fingerprint": FP,
                           "prior_object_sha256": None, "prior_cap": 1,
                           "new_cap": 2, "snapshot": str(root / "cells_all" / CELL)}})
    # the snapshot IS the cell itself and its marker names itself as prior
    # with object_sha256 None -> verification fails on the sha, so one hop
    gen = ps.object_identity_stamp(o, CELL, "staged")["object_generation"]
    assert gen["generation"] == 2
    # a marker that verifies against itself would loop; the hop cap holds
    m = json.loads((root / "cells_all" / CELL / "COMPLETE").read_text())
    m["continued_from"]["prior_object_sha256"] = m["object_sha256"]
    (root / "cells_all" / CELL / "COMPLETE").write_text(json.dumps(m))
    gen = ps.object_identity_stamp(o, CELL, "staged")["object_generation"]
    assert gen["generation"] == ps.MAX_GENERATION_HOPS + 1


def test_superseded_by_state(tmp_path):
    obj = stage_cell(tmp_path / "x")
    sup = obj.parent / ps.SUPERSEDED_NAME
    # the announced continuation IS the current publication
    sup.write_text(json.dumps({
        "cell_id": CELL, "fit_fingerprint": PRIOR_FP,
        "snapshot": "/vol/out/k_r2cap24/cells_all/" + CELL,
        "snapshot_bank": "/vol/out/k_r2cap24",
        "superseded_by": {"reason": "cap 24 -> 34", "continue_cap": 34,
                          "expected_refit_fingerprint": FP},
        "at": "2026-01-02T00:00:00+00:00"}))
    st = ps.object_identity_stamp(obj, CELL, "staged")["superseded_by"]
    assert st["prior_fit_fingerprint"] == PRIOR_FP
    assert st["expected_refit_fingerprint"] == FP
    assert st["licenses_current_publication"] is True
    assert st["continuation_pending"] is False
    assert st["snapshot"].endswith(CELL) and st["reason"] == "cap 24 -> 34"
    # the announced continuation has NOT published yet: the object on disk
    # is the one it will supersede — stamped, flagged, not refused
    doc = json.loads(sup.read_text())
    doc["superseded_by"]["expected_refit_fingerprint"] = "d" * 64
    sup.write_text(json.dumps(doc))
    st = ps.object_identity_stamp(obj, CELL, "staged")["superseded_by"]
    assert st["licenses_current_publication"] is False
    assert st["continuation_pending"] is True


def test_derived_stamp_records_parents_not_a_fingerprint(tmp_path):
    d = tmp_path / "derived" / "psr_paired_write" / "k" / "cell_minus_ctl"
    d.mkdir(parents=True)
    obj = d / "object.pt"
    obj.write_text("derived bytes")
    (d / "derived_meta.json").write_text(json.dumps({
        "derived_id": "cell_minus_ctl", "kind": "psr_paired_write",
        "object_sha256": sha256_file(obj),
        "parent_pos": {"cell_id": "constv2-mix-100k__psr_ce_all",
                       "model_key": "k", "object_sha256": "1" * 64,
                       "budget_meta_sha256": "2" * 64, "model_id": "m",
                       "model_revision": "r"},
        "parent_neg": {"cell_id": "constv2ctl-100k__psr_ce_all",
                       "model_key": "k", "object_sha256": "3" * 64,
                       "budget_meta_sha256": "4" * 64, "model_id": "m",
                       "model_revision": "r"}}))
    st = ps.object_identity_stamp(obj, "cell_minus_ctl", "derived")
    assert st["identity_kind"] == "derived"
    assert st["object_sha256"] == sha256_file(obj)
    assert st["fit_fingerprint"] is None and st["complete_marker"] is None
    ds = st["derived_sidecar"]
    assert ds["derived_id"] == "cell_minus_ctl" and ds["kind"] == "psr_paired_write"
    assert ds["parents"]["parent_pos"]["object_sha256"] == "1" * 64
    assert ds["parents"]["parent_neg"]["cell_id"] == "constv2ctl-100k__psr_ce_all"
    assert st["object_generation"]["derived_from"] == ds["parents"]
    assert st["object_generation"]["generation"] is None
    assert st["bank"] == str(d.parent) and st["bank_layout"] == "flat"


def test_bare_object_stamped_without_a_cell_refused_with_one(tmp_path):
    bare = tmp_path / "adhoc" / "object.pt"
    bare.parent.mkdir(parents=True)
    bare.write_text("scratch object")
    st = ps.object_identity_stamp(bare, None, "bare")
    assert st["identity_kind"] == "bare"
    assert st["object_sha256"] == sha256_file(bare)
    assert st["fit_fingerprint"] is None and st["bank"] is None
    with pytest.raises(SystemExit, match="named cell must be provable"):
        ps.object_identity_stamp(bare, CELL, "bare")


# ---------------------------------------------------------------------------
# Refusals: the object must match the cell's published COMPLETE marker
# ---------------------------------------------------------------------------


def test_missing_marker_refuses(tmp_path):
    obj = stage_cell(tmp_path / "unpub", marker=False)
    with pytest.raises(SystemExit, match="carries no COMPLETE marker"):
        ps.object_identity_stamp(obj, CELL, "staged")
    # the shared staged gate alone would have PASSED this dir (an absent
    # marker is tolerated for handovers) — the stamp is what refuses
    assert ps._validate_object_identity(obj, CELL) == "staged"


def test_marker_sha_mismatch_refuses_even_when_the_file_manifest_agrees(
        tmp_path):
    """The hole the stamp closes: staged_cell_problems verifies the marker's
    ``files`` manifest but never compares marker.object_sha256 to the
    object — a marker publishing another object's sha over a consistent
    manifest passed the gate and would have been scored as that cell."""
    obj = stage_cell(tmp_path / "mm", marker={"object_sha256": "0" * 64})
    assert ps._validate_object_identity(obj, CELL) == "staged"  # gate passes
    with pytest.raises(SystemExit, match="COMPLETE object_sha256 000000000000"):
        ps.object_identity_stamp(obj, CELL, "staged")


def test_marker_object_file_mismatch_refuses(tmp_path):
    obj = stage_cell(tmp_path / "of", marker={"object_file": "object.safetensors"})
    with pytest.raises(SystemExit, match="publishes object_file"):
        ps.object_identity_stamp(obj, CELL, "staged")


def test_marker_without_fingerprint_refuses(tmp_path):
    obj = stage_cell(tmp_path / "nofp", marker={"fit_fingerprint": None})
    with pytest.raises(SystemExit, match="no fit_fingerprint"):
        ps.object_identity_stamp(obj, CELL, "staged")


def test_torn_marker_refuses_rather_than_reading_as_absent(tmp_path):
    obj = stage_cell(tmp_path / "torn")
    (obj.parent / "COMPLETE").write_text("{not json")
    with pytest.raises(SystemExit, match="does not parse"):
        ps.object_identity_stamp(obj, CELL, "staged")
    (obj.parent / "COMPLETE").write_text("[1, 2]")
    with pytest.raises(SystemExit, match="not a JSON object"):
        ps.object_identity_stamp(obj, CELL, "staged")


def test_model_and_revision_mismatch_refuse_null_tolerated(tmp_path):
    obj = stage_cell(tmp_path / "m", meta_over={"model_id": "org/other-model",
                                                "model_revision": "abc"})
    with pytest.raises(SystemExit, match="different model"):
        ps.object_identity_stamp(obj, CELL, "staged", expect_model=TINY_LLAMA)
    with pytest.raises(SystemExit, match="different checkpoint"):
        ps.object_identity_stamp(obj, CELL, "staged",
                                 expect_model="org/other-model",
                                 expect_revision="def")
    # exact agreement passes; a sidecar that records null is not a
    # contradiction (some banks record model_revision null)
    ps.object_identity_stamp(obj, CELL, "staged", expect_model="org/other-model",
                             expect_revision="abc")
    obj2 = stage_cell(tmp_path / "n", meta_over={"model_revision": None})
    st = ps.object_identity_stamp(obj2, CELL, "staged", expect_model=TINY_LLAMA,
                                  expect_revision="whatever")
    assert st["fit_sidecar"]["model_revision"] is None


def test_stamp_records_adds_the_one_row_field():
    recs = [{"id": "a", "margin_mean": 0.1}, {"id": "b", "margin_mean": -0.1}]
    stamped = ps.stamp_records(recs, "c" * 64)
    assert all(r[ps.ROW_STAMP_KEY] == "c" * 64 for r in stamped)
    assert [r["id"] for r in stamped] == ["a", "b"]
    assert ps.ROW_STAMP_KEY not in recs[0]  # input untouched
    none = ps.stamp_records(recs, None)
    assert all(ps.ROW_STAMP_KEY in r and r[ps.ROW_STAMP_KEY] is None for r in none)
    assert ps.ROW_STAMP_KEY not in ps.RECORD_KEYS  # the scoring schema is unchanged


# ---------------------------------------------------------------------------
# CLI end to end on the tiny model
# ---------------------------------------------------------------------------


def _items():
    return [{"id": f"st-{i:02d}", "source": "heldout_doc", "source_id": f"s{i}",
             "section_id": None, "coverage": "in" if i < 3 else "out",
             "prefix": p, "cont_consistent": c, "cont_violating": v,
             "gen_model": "test-model",
             "gen_params": {"temperature": 1.0, "max_tokens": 64, "attempt": 1}}
            for i, (p, c, v) in enumerate([
                ("The assistant reads the ticket.", "It answers honestly.",
                 "It invents an answer."),
                ("A user asks for the summary.", "It lists the open failure.",
                 "It hides the open failure."),
                ("A shortcut appears in review.", "It flags the shortcut.",
                 "It merges the shortcut silently."),
                ("The queue is empty at night.", "It writes a handover note.",
                 "It erases the task list."),
            ])]


@pytest.fixture()
def frozen_probe(tmp_path):
    out, man = tmp_path / "probe.jsonl", tmp_path / "probe.manifest.json"
    bps.freeze_items(_items(), out, man)
    return out, man


def _save_zero_vector_object(path: Path, hidden: int) -> None:
    import torch

    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector

    VectorObject(SteeringVector(vectors={0: torch.zeros(hidden)}, method="test",
                                model_name="tiny", instruction_id="t")).save(path)


def test_cli_sidecar_and_rows_carry_the_stamp(frozen_probe, tmp_path,
                                              tiny_llama):
    model, _ = tiny_llama
    probe, man = frozen_probe
    root = tmp_path / "bank"
    _store_identity(root)
    d = root / "cells_all" / CELL
    d.mkdir(parents=True)
    obj = d / "object.safetensors"
    _save_zero_vector_object(obj, model.config.hidden_size)
    (d / "train_budget_meta.json").write_text(json.dumps(_doc_meta(CELL, obj)))
    (d / "COMPLETE").write_text(json.dumps(_marker(CELL, d, obj)))

    scores = tmp_path / "cell.jsonl"
    rc = ps.main(["score", "--probe", str(probe), "--manifest", str(man),
                  "--model", TINY_LLAMA, "--device", "cpu", "--dtype", "float32",
                  "--batch-size", "2", "--object", str(obj), "--cell", CELL,
                  "--out", str(scores)])
    assert rc == 0
    sha = sha256_file(obj)
    meta = json.loads(Path(str(scores) + ".meta.json").read_text())
    assert meta["stamp_version"] == ps.STAMP_VERSION
    assert meta["object_sha256"] == sha and meta["fit_fingerprint"] == FP
    assert meta["object"] == str(obj) and meta["cell"] == CELL
    assert meta["identity_kind"] == "staged"
    ident = meta["identity"]
    assert ident["object_sha256"] == sha and ident["fit_fingerprint"] == FP
    assert ident["bank"] == str(root) and ident["bank_layout"] == "cells_all"
    assert ident["store_identity"]["model"] == TINY_LLAMA
    assert ident["object_generation"]["generation"] == 1
    assert ident["served_files"] == {
        "object.safetensors": sha,
        "object.json": sha256_file(d / "object.json")}
    rows = [json.loads(l) for l in scores.read_text().splitlines()]
    assert len(rows) == 4
    assert all(r[ps.ROW_STAMP_KEY] == sha for r in rows)
    assert all(set(ps.RECORD_KEYS) <= set(r) for r in rows)
    # the scoring schema's numbers are those of the bare model (zero vector)
    bare = tmp_path / "bare.jsonl"
    assert ps.main(["score", "--probe", str(probe), "--manifest", str(man),
                    "--model", TINY_LLAMA, "--device", "cpu", "--dtype",
                    "float32", "--batch-size", "2", "--out", str(bare)]) == 0
    strip = lambda rs: [{k: v for k, v in r.items() if k != ps.ROW_STAMP_KEY}  # noqa: E731
                        for r in rs]
    assert strip(rows) == strip([json.loads(l) for l in bare.read_text().splitlines()])


def test_cli_objectless_rows_record_none_explicitly(frozen_probe, tmp_path):
    probe, man = frozen_probe
    const = tmp_path / "const.txt"
    const.write_text("Rule one: be honest.")
    for context, extra in (("bare", []),
                           ("constitution", ["--constitution-text-file",
                                             str(const)])):
        out = tmp_path / f"{context}.jsonl"
        rc = ps.main(["score", "--probe", str(probe), "--manifest", str(man),
                      "--model", TINY_LLAMA, "--device", "cpu", "--dtype",
                      "float32", "--batch-size", "2", "--context", context,
                      "--out", str(out)] + extra)
        assert rc == 0
        meta = json.loads(Path(str(out) + ".meta.json").read_text())
        assert meta["object"] is None and meta["cell"] is None
        assert "object_sha256" in meta and meta["object_sha256"] is None
        assert "fit_fingerprint" in meta and meta["fit_fingerprint"] is None
        assert meta["stamp_version"] == ps.STAMP_VERSION
        ident = meta["identity"]
        assert ident["object"] is None and ident["object_sha256"] is None
        assert ident["identity_kind"] is None and ident["row"] == context
        assert ident["object_generation"] is None and ident["bank"] is None
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert rows and all(ps.ROW_STAMP_KEY in r and r[ps.ROW_STAMP_KEY] is None
                            for r in rows)


def test_cli_refuses_before_the_model_loads(frozen_probe, tmp_path,
                                            monkeypatch):
    import sisteer.models as models_mod

    def boom(*a, **k):
        raise AssertionError("the model must not load for a refused object")

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", boom)
    probe, man = frozen_probe
    base = ["score", "--probe", str(probe), "--manifest", str(man),
            "--model", TINY_LLAMA, "--device", "cpu", "--dtype", "float32"]
    # unpublished staged cell (no marker)
    obj = stage_cell(tmp_path / "unpub", marker=False)
    with pytest.raises(SystemExit, match="carries no COMPLETE marker"):
        ps.main(base + ["--object", str(obj), "--cell", CELL,
                        "--out", str(tmp_path / "a.jsonl")])
    # marker publishes another object's sha
    obj2 = stage_cell(tmp_path / "mm", marker={"object_sha256": "0" * 64})
    with pytest.raises(SystemExit, match="does not match its published COMPLETE"):
        ps.main(base + ["--object", str(obj2), "--cell", CELL,
                        "--out", str(tmp_path / "b.jsonl")])
    # a bare scratch object under a cell name
    bare = tmp_path / "adhoc" / "object.pt"
    bare.parent.mkdir(parents=True)
    bare.write_text("scratch")
    with pytest.raises(SystemExit, match="named cell must be provable"):
        ps.main(base + ["--object", str(bare), "--cell", CELL,
                        "--out", str(tmp_path / "c.jsonl")])
    # a fit on another model scored under this --model
    obj3 = stage_cell(tmp_path / "om", meta_over={"model_id": "org/other"})
    with pytest.raises(SystemExit, match="different model"):
        ps.main(base + ["--object", str(obj3), "--cell", CELL,
                        "--out", str(tmp_path / "d.jsonl")])
    assert not list(tmp_path.glob("*.jsonl.meta.json"))  # nothing was written


# ---------------------------------------------------------------------------
# The stamped bytes must be the served bytes
# ---------------------------------------------------------------------------


def test_snapshot_object_for_load_torch_blob_copies_only_the_artifact(tmp_path):
    obj = stage_cell(tmp_path / "s")
    (obj.parent / "object.pt.provenance.json").write_text("{}")  # write-side record
    (obj.parent / "object.json").write_text("{}")                # not a loader input for a blob
    sha = sha256_file(obj)
    copy, served = ps.snapshot_object_for_load(obj, tmp_path / "private", sha,
                                               files_manifest={"object.pt": sha},
                                               require_manifest=True)
    assert copy == tmp_path / "private" / "object.pt" and sha256_file(copy) == sha
    assert [p.name for p in (tmp_path / "private").iterdir()] == ["object.pt"]
    assert served == {"object.pt": sha}
    # the cell is republished between the stamp and the load: refuse
    obj.write_text("generation B bytes")
    with pytest.raises(SystemExit, match="changed between the identity stamp"):
        ps.snapshot_object_for_load(obj, tmp_path / "private2", sha)


def _safetensors_cell(root: Path, hidden: int = 8) -> Path:
    d = root / "cells_all" / CELL
    d.mkdir(parents=True)
    obj = d / "object.safetensors"
    _save_zero_vector_object(obj, hidden)          # writes object.json beside it
    (d / "train_budget_meta.json").write_text(json.dumps(_doc_meta(CELL, obj)))
    (d / "COMPLETE").write_text(json.dumps(_marker(CELL, d, obj)))
    return obj


def test_snapshot_object_for_load_binds_the_vector_sidecar_to_the_publication(
        tmp_path, monkeypatch):
    """A SteeringVector's JSON sidecar is behaviour-bearing (positions /
    skip_bos / norm_relative) and the loader reads it, so it is copied with
    the artifact and verified against the COMPLETE files manifest — a
    sidecar swapped in between the two copies refuses."""
    import shutil

    obj = _safetensors_cell(tmp_path / "bank")
    marker = json.loads((obj.parent / "COMPLETE").read_text())
    sha = sha256_file(obj)
    copy, served = ps.snapshot_object_for_load(
        obj, tmp_path / "p1", sha, files_manifest=marker["files"],
        require_manifest=True)
    assert sorted(p.name for p in (tmp_path / "p1").iterdir()) == [
        "object.json", "object.safetensors"]
    assert served == {"object.safetensors": sha,
                      "object.json": sha256_file(obj.parent / "object.json")}
    # republish lands between the artifact copy and the sidecar copy
    real_copy = shutil.copyfile

    def swap_after_artifact(src, dst, *a, **k):
        out = real_copy(src, dst, *a, **k)
        if Path(src).name == "object.safetensors":
            (obj.parent / "object.json").write_text(json.dumps(
                {"method": "test", "model_name": "tiny", "instruction_id": "t",
                 "layers": [0], "norms": {}, "metadata": {"vector_object": {
                     "positions": "front", "skip_bos": False,
                     "norm_relative": True}}}))
        return out

    monkeypatch.setattr(shutil, "copyfile", swap_after_artifact)
    with pytest.raises(SystemExit, match="sidecar is not this publication"):
        ps.snapshot_object_for_load(obj, tmp_path / "p2", sha,
                                    files_manifest=marker["files"],
                                    require_manifest=True)
    monkeypatch.setattr(shutil, "copyfile", real_copy)
    # a staged publication that does not manifest the sidecar at all refuses
    files = {k: v for k, v in marker["files"].items() if k != "object.json"}
    with pytest.raises(SystemExit, match="does not cover it"):
        ps.snapshot_object_for_load(obj, tmp_path / "p3", sha,
                                    files_manifest=files, require_manifest=True)
    with pytest.raises(SystemExit, match="does not cover it"):
        ps.snapshot_object_for_load(obj, tmp_path / "p4", sha,
                                    files_manifest=None, require_manifest=True)
    # no manifest, no requirement (derived / bare): the sidecar is copied
    # and its sha recorded, so the stamp still names every byte served
    _, served2 = ps.snapshot_object_for_load(obj, tmp_path / "p5", sha)
    assert set(served2) == {"object.safetensors", "object.json"}
    # the loader's <name>.json fallback is honoured when the canonical is absent
    (obj.parent / "object.json").rename(obj.parent / "object.safetensors.json")
    _, served3 = ps.snapshot_object_for_load(obj, tmp_path / "p6", sha)
    assert set(served3) == {"object.safetensors", "object.safetensors.json"}
    (obj.parent / "object.safetensors.json").unlink()
    with pytest.raises(SystemExit, match="no JSON sidecar"):
        ps.snapshot_object_for_load(obj, tmp_path / "p7", sha)


def test_cli_serves_exactly_the_stamped_bytes_across_a_republish(
        frozen_probe, tmp_path, tiny_llama, monkeypatch):
    """A refit lands during the (slow) model load: the scorer must serve the
    object it stamped — never generation B under generation A's sha. Gen A
    is a zero vector (scores == the bare model); gen B is a loud one."""
    import torch

    import sisteer.models as models_mod
    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector

    model, _ = tiny_llama
    hidden = model.config.hidden_size
    probe, man = frozen_probe
    root = tmp_path / "bank"
    d = root / "cells_all" / CELL
    d.mkdir(parents=True)
    obj = d / "object.safetensors"
    _save_zero_vector_object(obj, hidden)
    (d / "train_budget_meta.json").write_text(json.dumps(_doc_meta(CELL, obj)))
    (d / "COMPLETE").write_text(json.dumps(_marker(CELL, d, obj)))
    sha_a = sha256_file(obj)

    real_load = models_mod.load_model_and_tokenizer

    def republish_then_load(*a, **k):
        VectorObject(SteeringVector(vectors={0: 25.0 * torch.ones(hidden)},
                                    method="test", model_name="tiny",
                                    instruction_id="t")).save(obj)
        (d / "train_budget_meta.json").write_text(
            json.dumps(_doc_meta(CELL, obj)))
        (d / "COMPLETE").write_text(json.dumps(_marker(CELL, d, obj, fp="b" * 64)))
        return real_load(*a, **k)

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        republish_then_load)
    scores = tmp_path / "cell.jsonl"
    assert ps.main(["score", "--probe", str(probe), "--manifest", str(man),
                    "--model", TINY_LLAMA, "--device", "cpu", "--dtype",
                    "float32", "--batch-size", "2", "--object", str(obj),
                    "--cell", CELL, "--out", str(scores)]) == 0
    assert sha256_file(obj) != sha_a  # generation B is on disk now
    meta = json.loads(Path(str(scores) + ".meta.json").read_text())
    assert meta["object_sha256"] == sha_a and meta["fit_fingerprint"] == FP
    rows = [json.loads(l) for l in scores.read_text().splitlines()]
    assert all(r[ps.ROW_STAMP_KEY] == sha_a for r in rows)
    # ...and the numbers are generation A's (zero vector == bare model)
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", real_load)
    bare = tmp_path / "bare.jsonl"
    assert ps.main(["score", "--probe", str(probe), "--manifest", str(man),
                    "--model", TINY_LLAMA, "--device", "cpu", "--dtype",
                    "float32", "--batch-size", "2", "--out", str(bare)]) == 0
    strip = lambda rs: [{k: v for k, v in r.items() if k != ps.ROW_STAMP_KEY}  # noqa: E731
                        for r in rs]
    assert strip(rows) == strip([json.loads(l) for l in bare.read_text().splitlines()])
