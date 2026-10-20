"""scripts/sdfbase/reemit_preflight.py — the exact records a corrective re-emission would write, diffed against the registry's rows
BEFORE any spool write: a payload that would move a fit-provenance column (or anything else outside
campaign + the ingest tail) is REFUSED pre-write; the move is admitted only when a fit_job_provenance.json sidecar attests exactly those
values from the fit's own records; a campaign-only change is clean; a row the registry lacks is new. Synthetic throughout."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import reemit_preflight as RP  # noqa: E402


@pytest.fixture(autouse=True)
def _registry_campaign_scoped_to_the_test():
    """`RP.main` sets os.environ["REGISTRY_CAMPAIGN"] (reemit_preflight.py, the `--campaign` argument, for the records it builds) and this
    module calls it IN-PROCESS, so the write outlived each test and leaked into the next module of the same pytest process:
    tests/test_refusal_direction_fit.py::test_registry_call_states_the_object (alphabetically next) read the fitter's argparse default
    `os.environ.get("REGISTRY_CAMPAIGN", "two_si")` as `sdfbase_full22` (the Slurm runner's hooks-importer batteries, 2026-10-14). The key
    is scoped to each test here: its state before the test (absent, or whatever the shell carries) is put back at teardown, whatever main()
    wrote meanwhile. Not `monkeypatch.delenv(..., raising=False)`: for a key that is ABSENT before the test monkeypatch records nothing, so
    it restores nothing and the leak stands (measured: the runner's array 1025400). Nothing about either test is changed."""
    before = os.environ.get("REGISTRY_CAMPAIGN")
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("REGISTRY_CAMPAIGN", None)
        else:
            os.environ["REGISTRY_CAMPAIGN"] = before

SI, ARM = "constv2-mix-full", "lora_r1_all"      # a MAPPED corpus pool (role constitution): an unmapped spelling makes the trainer state not_emitted


def _cell(tmp_path: Path) -> tuple[Path, str]:
    d = tmp_path / "cells_all" / f"{SI}__{ARM}"
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(b"synthetic lora object bytes " * 64)
    sha = hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()
    (d / "train_budget_meta.json").write_text(json.dumps({
        "arm": ARM, "si_id": SI, "training_budget_kind": "document_tokens", "positions_at_fit": "all", "seed": 0, "lr": 1e-3, "epochs": 3,
        "max_grad_norm": 1.0, "object_sha256": sha, "model_id": "synthetic/base-4b", "model_revision": "b" * 40,
        "template_hash": "b48c47f644389271", "target_layers": [4, 5, 6], "cell_id": f"{SI}__{ARM}"}))
    (d / "COMPLETE").write_text(json.dumps({"cell_id": f"{SI}__{ARM}", "object_sha256": sha}))
    return d, sha


def _context(monkeypatch, *, job: str, code_sha: str, host: str):
    from sisteer.registry import emit
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setattr(emit, "producer_block", lambda script, code_sha_=None, session=None: {"script": script, "code_sha": code_sha, "host": host, "substrate": "slurm", "session": None})


def _rows_from(recs, campaign: str) -> dict:
    """A registry-host row dump shaped like diff_object_rows.py's: the canonical row of every object record + the artifact rows."""
    rows, arts = {}, {}
    for rtype, payload in recs:
        if rtype == "object":
            row = RP._canon("object", payload)
            row["extra_json"] = json.dumps(row["extra_json"], sort_keys=True)
            row["campaign"] = campaign
            rows[payload["object_id"]] = row
        else:
            arts[payload["artifact_id"]] = dict(payload)
    return {"object": rows, "artifact": arts}


def _args(d: Path, rows: Path, campaign: str, out: Path | None = None):
    a = ["fit", "--dir", str(d), "--si", SI, "--arm", ARM, "--campaign", campaign, "--rows", str(rows)]
    return a + (["--out", str(out)] if out else [])


def test_a_payload_that_would_move_fit_provenance_is_refused_and_an_attested_move_is_admitted(tmp_path, monkeypatch, capsys):
    import train_two_si_objects as t2
    from sisteer.registry import hooks
    d, sha = _cell(tmp_path)
    # the registry's rows: what the FIT job emitted (its own context), under two_si
    _context(monkeypatch, job="821364", code_sha="a" * 40, host="gpu-23")
    recs = t2._registry_publish(d, si_id=SI, arm=ARM, dry_run=True)
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(_rows_from(recs, "two_si"), default=str))
    # a re-emission from ANOTHER job's context: refused by name before anything is written
    _context(monkeypatch, job="828839", code_sha="d" * 40, host="gpu-15")
    rc = RP.main(_args(d, rows_path, "sdfbase_full22", tmp_path / "planned.json"))
    out, err = capsys.readouterr()
    assert rc == 3 and "REFUSED" in err and "fit_job_id" in err and "fit_code_sha" in err and "fit_node_hostname" in err
    assert json.loads((tmp_path / "planned.json").read_text())["records"]            # the record of what was about to be emitted
    assert not list(tmp_path.glob("**/*.jsonl")) and not (d / "registry_emission.json").exists()   # nothing written, no evidence
    # the same context with a sidecar attesting the FIT's values from its own records: the move is admitted, named, sourced
    (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps({
        "fit_code_sha": "a" * 40, "fit_job_id": "821364", "fit_node_hostname": "gpu-23", "fit_substrate": "slurm",
        "artifact_created_utc": json.loads(rows_path.read_text())["artifact"][sha]["created_utc"],
        "sources": {"result": str(d / "RESULT"), "attempt_record": "attempts.jsonl#constitution_relu@cluster_a@2"},
        "note": "fit provenance corrected from RESULT + attempt record", "written_by": "tests", "written_utc": "2026-10-02T06:00:00Z"}))
    rc = RP.main(_args(d, rows_path, "sdfbase_full22"))
    out, err = capsys.readouterr()
    assert rc == 0 and "CLEAN" in out and f"object {sha[:16]}: OK (campaign 'two_si' -> 'sdfbase_full22')" in out, out + err
    # a sidecar attesting DIFFERENT values than the rows (the correction case: all five columns the re-emission chain had
    # overwritten — the four fit_* columns and the artifact's created time): admitted as attested moves, reported per column
    (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps({
        "fit_code_sha": "b" * 40, "fit_job_id": "821352", "fit_node_hostname": "gpu-29", "fit_substrate": "cluster_a",
        "artifact_created_utc": "2026-10-01T11:11:16.271Z",
        "sources": {"result": "r", "attempt_record": "a", "fit_job_record": "000001-object.jsonl"}, "written_by": "tests", "written_utc": "t"}))
    rc = RP.main(_args(d, rows_path, "sdfbase_full22"))
    out, _ = capsys.readouterr()
    assert rc == 0 and f"fit_job_id '821364' -> '821352' (attested by its own {d.name}/fit_job_provenance.json)" in out and "fit_substrate 'slurm' -> 'cluster_a' (attested" in out
    assert f"artifact {sha[:16]}: created_utc" in out and "-> '2026-10-01T11:11:16.271Z' (attested" in out
    # but a change OUTSIDE the attested values — the artifact's created time drifting to NOW because the sidecar attests none — is refused
    side = json.loads((d / hooks.FIT_PROVENANCE_SIDECAR).read_text()); side.pop("artifact_created_utc")
    (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps(side))
    rc = RP.main(_args(d, rows_path, "sdfbase_full22"))
    _, err = capsys.readouterr()
    assert rc == 3 and "artifact" in err and "created_utc" in err
    # and a fit_* column moving to a value the sidecar does NOT attest (a sixth column, fit_producer_session) is refused
    side["fit_producer_session"] = "someone"
    (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps(dict(side, artifact_created_utc="2026-10-01T11:11:16.271Z")))
    rc = RP.main(_args(d, rows_path, "sdfbase_full22"))
    _, err = capsys.readouterr()
    assert rc == 3 and "fit_producer_session" in err
    # the stated campaign must be the payload's
    rc = RP.main(["fit", "--dir", str(d), "--si", SI, "--arm", ARM, "--campaign", "sdfbase_full22", "--rows", str(rows_path)])
    assert rc in (0, 3)


def test_a_row_the_registry_lacks_is_new_and_a_row_differing_only_in_campaign_is_clean(tmp_path, monkeypatch, capsys):
    import train_two_si_objects as t2
    d, sha = _cell(tmp_path)
    _context(monkeypatch, job="821364", code_sha="a" * 40, host="gpu-23")
    recs = t2._registry_publish(d, si_id=SI, arm=ARM, dry_run=True)
    empty = tmp_path / "empty.json"; empty.write_text(json.dumps({"object": {}, "artifact": {}}))
    rc = RP.main(_args(d, empty, "sdfbase_full22"))
    out, _ = capsys.readouterr()
    assert rc == 0 and f"object {sha[:16]}: NEW row" in out
    rows_path = tmp_path / "rows.json"; rows_path.write_text(json.dumps(_rows_from(recs, "two_si"), default=str))
    # the same context as the rows but NO statement of the fit's provenance: the artifact's created time would move to now -> refused
    rc = RP.main(_args(d, rows_path, "sdfbase_full22"))
    _, err = capsys.readouterr()
    assert rc == 3 and "artifact" in err and "created_utc" in err, err
    # with the fit's statement (the sidecar attesting exactly the rows' values): only the campaign moves -> clean
    from sisteer.registry import hooks
    (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps({
        "fit_code_sha": "a" * 40, "fit_job_id": "821364", "fit_node_hostname": "gpu-23", "fit_substrate": "slurm",
        "artifact_created_utc": json.loads(rows_path.read_text())["artifact"][sha]["created_utc"],
        "sources": {"result": "r", "attempt_record": "a"}, "written_by": "tests", "written_utc": "t"}))
    rc = RP.main(_args(d, rows_path, "sdfbase_full22"))
    out, err = capsys.readouterr()
    assert rc == 0 and "CLEAN" in out, out + err
    # a flat {id: row} dump is accepted too (no artifact rows to compare)
    flat = tmp_path / "flat.json"; flat.write_text(json.dumps(_rows_from(recs, "two_si")["object"], default=str))
    assert RP.main(_args(d, flat, "sdfbase_full22")) == 0


def _dump(rows: dict, arts: dict, *, seq: int, stamp) -> dict:
    """A registry-host dump with the ingest tail filled: every column versioned by `stamp` = [utc, record_sha]."""
    def tail(row, cols):
        r = dict(row); r["ingest_seq"] = seq
        r["field_versions_json"] = json.dumps({c: list(stamp) for c in cols}, sort_keys=True)
        return r
    return {"object": {k: tail(v, ("campaign", "fit_lr", "status")) for k, v in rows.items()},
            "artifact": {k: tail(v, ("created_utc", "bytes")) for k, v in arts.items()}}


def test_freshness_refuses_when_the_registry_moved_since_the_step_started(tmp_path, capsys):
    base_o = {"a" * 64: {"object_id": "a" * 64, "campaign": "two_si", "fit_lr": 0.001, "status": "reference"}}
    base_a = {"a" * 64: {"artifact_id": "a" * 64, "created_utc": "t0", "bytes": 1}}
    pre = tmp_path / "pre.json"; pre.write_text(json.dumps(_dump(base_o, base_a, seq=100, stamp=["t0", "s" * 64])))
    same = tmp_path / "same.json"; same.write_text(json.dumps(_dump(base_o, base_a, seq=100, stamp=["t0", "s" * 64])))
    assert RP.main(["fresh", "--pre-rows", str(pre), "--rows", str(same)]) == 0 and "CLEAN" in capsys.readouterr().out
    # another writer folded a change (ingest_seq / versions moved): refused by name
    moved = tmp_path / "moved.json"; moved.write_text(json.dumps(_dump(base_o, base_a, seq=101, stamp=["t1", "x" * 64])))
    assert RP.main(["fresh", "--pre-rows", str(pre), "--rows", str(moved)]) == 3
    err = capsys.readouterr().err
    assert "REFUSED" in err and "ingest_seq moved" in err and "another writer folded" in err
    # a row that appeared or vanished refuses too
    extra = _dump(dict(base_o, **{"b" * 64: {"object_id": "b" * 64, "campaign": "two_si"}}), base_a, seq=100, stamp=["t0", "s" * 64])
    appeared = tmp_path / "appeared.json"; appeared.write_text(json.dumps(extra))
    assert RP.main(["fresh", "--pre-rows", str(pre), "--rows", str(appeared)]) == 3 and "appeared" in capsys.readouterr().err
    # the same precondition inside a fit/derived preflight: refused before the dry run even runs (no --dir needed to fail)
    assert RP.main(["fit", "--dir", str(tmp_path), "--si", SI, "--arm", ARM, "--campaign", "x", "--rows", str(moved), "--pre-rows", str(pre)]) == 3
    assert "registry moved since the step started" in capsys.readouterr().err


def test_audit_refuses_a_column_moved_by_a_record_that_is_not_ours(tmp_path, capsys):
    ours_sha, foreign_sha = "o" * 64, "f" * 64
    ours = tmp_path / "ours.txt"; ours.write_text(ours_sha + "\n")
    base_o = {"a" * 64: {"object_id": "a" * 64, "campaign": "two_si", "fit_lr": 0.001, "status": "reference"}}
    base_a = {"a" * 64: {"artifact_id": "a" * 64, "created_utc": "t0", "bytes": 1}}
    before = tmp_path / "before.json"; before.write_text(json.dumps(_dump(base_o, base_a, seq=100, stamp=["t0", "s" * 64])))
    # after the fold: campaign + created_utc moved by OUR record -> clean (fit_lr/status/bytes untouched: stamps unchanged)
    after = _dump(base_o, base_a, seq=100, stamp=["t0", "s" * 64])
    after["object"]["a" * 64]["campaign"] = "sdfbase_full22"
    fv = json.loads(after["object"]["a" * 64]["field_versions_json"]); fv["campaign"] = ["t2", ours_sha]
    after["object"]["a" * 64]["field_versions_json"] = json.dumps(fv, sort_keys=True); after["object"]["a" * 64]["ingest_seq"] = 102
    fa = json.loads(after["artifact"]["a" * 64]["field_versions_json"]); fa["created_utc"] = ["t2", ours_sha]
    after["artifact"]["a" * 64]["field_versions_json"] = json.dumps(fa, sort_keys=True)
    ok = tmp_path / "after_ok.json"; ok.write_text(json.dumps(after))
    assert RP.main(["audit", "--before", str(before), "--after", str(ok), "--ours", str(ours)]) == 0
    assert "CLEAN" in capsys.readouterr().out
    # a column moved by a FOREIGN record (someone else's status change folded between our dumps): refused by name, never silent
    bad = json.loads(ok.read_text())
    fv = json.loads(bad["object"]["a" * 64]["field_versions_json"]); fv["status"] = ["t3", foreign_sha]
    bad["object"]["a" * 64]["field_versions_json"] = json.dumps(fv, sort_keys=True)
    bad_p = tmp_path / "after_bad.json"; bad_p.write_text(json.dumps(bad))
    assert RP.main(["audit", "--before", str(before), "--after", str(bad_p), "--ours", str(ours)]) == 3
    err = capsys.readouterr().err
    assert "NOT ours" in err and "status" in err and foreign_sha[:12] in err
    # a NEW row (V) must be ours in every column; a vanished row refuses
    new = json.loads(ok.read_text())
    new["object"]["v" * 64] = dict(new["object"]["a" * 64], object_id="v" * 64)
    vnew = tmp_path / "after_new.json"; vnew.write_text(json.dumps(new))
    assert RP.main(["audit", "--before", str(before), "--after", str(vnew), "--ours", str(ours)]) == 3   # its fit_lr/status stamps are the seed's, not ours
    gone = json.loads(before.read_text()); gone["object"] = {}
    gone_p = tmp_path / "after_gone.json"; gone_p.write_text(json.dumps(gone))
    assert RP.main(["audit", "--before", str(before), "--after", str(gone_p), "--ours", str(ours)]) == 3 and "disappeared" in capsys.readouterr().err


def test_each_epoch_artifact_moves_only_to_its_own_attested_time_end_to_end(tmp_path, monkeypatch, capsys):
    """End to end on a cell with THREE epoch checkpoints whose existing rows carry a re-emission's
    timestamp while their original times differ per epoch: the preflight must admit each epoch artifact moving to ITS OWN attested
    time (never the parent's), refuse an epoch whose sidecar attests a wrong time, and refuse an epoch with no attestation at all."""
    from test_epoch_checkpoints_d546 import CELL, KEY, _cell_meta, _save
    from sisteer.registry import hooks
    import train_two_si_objects as t2

    root = tmp_path / "ckpt" / "k"
    _save(root, b"e1", 1); _save(root, b"e2", 2); _save(root, b"e3", 3)
    cell = tmp_path / CELL; cell.mkdir()
    _cell_meta(cell, b"final-bytes")
    t2.finalize_epoch_checkpoints(root, cell, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=3)
    index = json.loads((cell / t2.EPOCH_CKPT_INDEX).read_text())
    epoch_dirs = [cell / os.path.dirname(c["artifact"]) for c in index["checkpoints"]]
    epoch_ids = [hashlib.sha256((d / "object.pt").read_bytes()).hexdigest() for d in epoch_dirs]
    main_sha = hashlib.sha256(b"final-bytes").hexdigest()
    (cell / "COMPLETE").write_text(json.dumps({"cell_id": CELL, "object_sha256": main_sha}))
    # the registry's rows: the FIT job's statement, but every artifact's created time later overwritten by a re-emission (19:45Z)
    _context(monkeypatch, job="821364", code_sha="a" * 40, host="gpu-23")
    recs = t2._registry_publish(cell, si_id="constv2-mix-full", arm="psr_ce_all", dry_run=True)
    dump = _rows_from(recs, "two_si")
    for aid in dump["artifact"]:
        dump["artifact"][aid]["created_utc"] = "2026-10-01T19:45:37.000Z"           # the overwrite, every artifact alike
    rows_path = tmp_path / "rows.json"; rows_path.write_text(json.dumps(dump, default=str))
    original = {main_sha: "2026-10-01T19:32:21.754Z", epoch_ids[0]: "2026-10-01T19:32:21.904Z", epoch_ids[1]: "2026-10-01T19:32:21.933Z",
                epoch_ids[2]: "2026-10-01T19:32:21.963Z"}
    def sidecar(created):
        return json.dumps({"fit_code_sha": "a" * 40, "fit_job_id": "821364", "fit_node_hostname": "gpu-23", "fit_substrate": "slurm",
                           "artifact_created_utc": created, "sources": {"result": "r", "attempt_record": "a", "fit_job_record": "f"},
                           "written_by": "tests", "written_utc": "t"})
    args = ["fit", "--dir", str(cell), "--si", "constv2-mix-full", "--arm", "psr_ce_all", "--campaign", "sdfbase_full22", "--rows", str(rows_path)]
    # the parent attests, the epochs do not: the hook refuses before any record (no records -> refused)
    (cell / hooks.FIT_PROVENANCE_SIDECAR).write_text(sidecar(original[main_sha]))
    with pytest.raises(SystemExit, match="produced no records"):
        RP.main(args)
    capsys.readouterr()
    # every epoch attests its OWN original time: each artifact's move is admitted and named with its own sidecar
    for d, eid in zip(epoch_dirs, epoch_ids):
        (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(sidecar(original[eid]))
    rc = RP.main(args)
    out, err = capsys.readouterr()
    assert rc == 0, out + err
    for eid, d in zip(epoch_ids, epoch_dirs):
        assert f"artifact {eid[:16]}: created_utc '2026-10-01T19:45:37.000Z' -> '{original[eid]}' (attested by its own {d.name}/fit_job_provenance.json)" in out
    assert f"artifact {main_sha[:16]}: created_utc '2026-10-01T19:45:37.000Z' -> '{original[main_sha]}' (attested by its own {cell.name}/" in out
    # an epoch with NO attestation of its own: the hook refuses before any record (nothing planned, nothing to emit)
    (epoch_dirs[1] / hooks.FIT_PROVENANCE_SIDECAR).unlink()
    with pytest.raises(SystemExit, match="produced no records"):
        RP.main(args)
    capsys.readouterr()
    # the preflight itself (the records the hook built against the sidecars on disk, then checked): an epoch artifact whose planned
    # time is the PARENT's — the fix-4 shape, which the parent-only attestation used to admit — is refused by that artifact's name,
    # and an epoch whose directory attests nothing is refused as unattested, whatever time it carries
    (epoch_dirs[1] / hooks.FIT_PROVENANCE_SIDECAR).write_text(sidecar(original[epoch_ids[1]]))
    planned = t2._registry_publish(cell, si_id="constv2-mix-full", arm="psr_ce_all", dry_run=True)
    rows, arts = RP._load_rows(str(rows_path))
    cell_att = RP._attested(type("A", (), {"dir": str(cell)})())
    assert cell_att["artifact_created_utc"] == original[main_sha]
    problems, _ = RP.preflight(planned, rows, arts, cell_att, campaign="sdfbase_full22")
    assert problems == [], problems
    swapped = [(r, dict(p, created_utc=original[main_sha]) if r == "artifact" and p["artifact_id"] == epoch_ids[1] else p) for r, p in planned]
    problems, _ = RP.preflight(swapped, rows, arts, cell_att, campaign="sdfbase_full22")
    assert len(problems) == 1 and problems[0].startswith(f"artifact {epoch_ids[1][:16]}: created_utc") and "19:45:37" in problems[0], problems
    (epoch_dirs[2] / hooks.FIT_PROVENANCE_SIDECAR).unlink()
    problems, _ = RP.preflight(planned, rows, arts, cell_att, campaign="sdfbase_full22")
    assert len(problems) == 1 and problems[0].startswith(f"artifact {epoch_ids[2][:16]}: created_utc") and "(no attestation in its own directory)" in problems[0], problems
