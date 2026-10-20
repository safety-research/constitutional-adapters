"""Snapshot (relocate) of to-be-superseded COMPLETE cells
(scripts/sdfbase/snapshot_cells.py).

Pins: a snapshot is a COMPLETE bank layout (cells_<leg>/<cell> +
_store_identity.json) whose files verify against the source's COMPLETE
manifest on BOTH sides; supersede records land on both sides naming the
fingerprint that was snapshotted; a source that no longer matches its own
manifest (overwritten after publish) REFUSES; an existing snapshot of
another identity is never overwritten; the copy passes the SAME staged-cell
identity gate as the live cell; the CLI reports refusals as rc 2.

CPU-only, synthetic bytes only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import snapshot_cells as sc  # noqa: E402

CELL = "constv2-mix-1m__psr_ce_all"
FP = "a" * 64
IDENTITY = {"model": "fake/base", "model_revision": "r0", "positions": "all"}


def _doc_meta(cell_id: str, obj: Path) -> dict:
    return {"cell_id": cell_id, "object_sha256": sc.sha256_file(obj),
            "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None,
            "sample_manifest_sha": "ab" * 32, "model_id": IDENTITY["model"],
            "model_revision": IDENTITY["model_revision"]}


def stage_bank(root: Path, cells=(CELL,), *, fp: str = FP,
               identity: dict | None = None, leg: str = "all") -> Path:
    """A synthetic live bank: cells_<leg>/<cell>/{object.pt, metrics.jsonl,
    train_budget_meta.json, COMPLETE} + the store identity."""
    bank = Path(root)
    cells_dir = bank / f"cells_{leg}"
    cells_dir.mkdir(parents=True, exist_ok=True)
    (cells_dir / sc.STORE_IDENT).write_text(
        json.dumps(identity or IDENTITY, indent=1))
    for cell in cells:
        d = cells_dir / cell
        d.mkdir(parents=True, exist_ok=True)
        (d / "object.pt").write_bytes(f"object bytes of {cell}".encode())
        (d / "metrics.jsonl").write_text('{"step": 1}\n')
        (d / "train_budget_meta.json").write_text(
            json.dumps(_doc_meta(cell, d / "object.pt")))
        files = {n: sc.sha256_file(d / n)
                 for n in ("object.pt", "metrics.jsonl",
                           "train_budget_meta.json")}
        (d / sc.COMPLETE).write_text(json.dumps(
            {"cell_id": cell, "fit_fingerprint": fp, "files": files,
             "object_file": "object.pt",
             "object_sha256": files["object.pt"]}))
    return bank


@pytest.fixture()
def banks(tmp_path):
    src = stage_bank(tmp_path / "out" / "key")
    dst = tmp_path / "out" / "key_r1"
    return src, dst


def test_snapshot_copies_verifies_and_records_both_sides(banks):
    src, dst = banks
    rec = sc.snapshot_cell(src, dst, CELL,
                           superseded_by={"reason": "cap-24 refit",
                                          "expected_refit_fingerprint": "b" * 64})
    snap = dst / "cells_all" / CELL
    # complete bank layout: every manifested file, byte-identical, plus the
    # store identity carried over
    for name in ("object.pt", "metrics.jsonl", "train_budget_meta.json",
                 sc.COMPLETE):
        assert (snap / name).read_bytes() == (src / "cells_all" / CELL / name
                                              ).read_bytes()
    assert json.loads((dst / "cells_all" / sc.STORE_IDENT).read_text()) == \
        IDENTITY
    # snapshot-side record
    assert rec["already_present"] is False
    assert rec["fit_fingerprint"] == FP
    assert set(rec["files"]) == {"object.pt", "metrics.jsonl",
                                 "train_budget_meta.json"}
    on_disk = json.loads((snap / sc.SNAPSHOT_RECORD).read_text())
    assert on_disk["source"] == str(src / "cells_all" / CELL)
    assert on_disk["fit_fingerprint"] == FP
    # source-side record: the identity that WAS snapshotted + the successor
    sup = sc.supersede_record(src / "cells_all" / CELL)
    assert sup["fit_fingerprint"] == FP
    assert sup["snapshot"] == str(snap)
    assert sup["superseded_by"] == {"reason": "cap-24 refit",
                                    "expected_refit_fingerprint": "b" * 64}
    # no tmp leftovers
    assert not list(dst.rglob("*tmp*"))
    assert not list(src.rglob("*.tmp"))


def test_snapshot_passes_the_shared_staged_cell_gate(banks):
    """The copy is a normal bank: the same identity gate that validates the
    live cell validates the snapshot, unchanged."""
    from sisteer.eval.train_budget import staged_cell_problems

    src, dst = banks
    sc.snapshot_cell(src, dst, CELL)
    assert staged_cell_problems(dst / "cells_all" / CELL / "object.pt",
                                CELL) == []
    assert sc.verify_snapshot(dst, CELL)["fit_fingerprint"] == FP


def test_snapshot_is_idempotent_on_an_identical_existing_copy(banks):
    src, dst = banks
    sc.snapshot_cell(src, dst, CELL)
    rec = sc.snapshot_cell(src, dst, CELL, superseded_by={"reason": "again"})
    assert rec["already_present"] is True
    assert sc.supersede_record(src / "cells_all" / CELL)["superseded_by"] == \
        {"reason": "again"}


def test_source_overwritten_after_publish_refuses(banks):
    """The plan's named refusal: a source whose bytes no longer match its
    COMPLETE manifest is not the publication the marker describes."""
    src, dst = banks
    (src / "cells_all" / CELL / "object.pt").write_bytes(b"a refit landed")
    with pytest.raises(sc.SnapshotError, match="no longer matches"):
        sc.snapshot_cell(src, dst, CELL)
    assert not dst.exists()  # nothing half-copied


def test_existing_snapshot_of_another_identity_is_never_overwritten(banks):
    src, dst = banks
    sc.snapshot_cell(src, dst, CELL)
    # the live cell is refit (new bytes + new fingerprint, coherent marker)
    live = src / "cells_all" / CELL
    (live / "object.pt").write_bytes(b"refit object bytes")
    marker = json.loads((live / sc.COMPLETE).read_text())
    marker["fit_fingerprint"] = "c" * 64
    marker["files"]["object.pt"] = sc.sha256_file(live / "object.pt")
    (live / sc.COMPLETE).write_text(json.dumps(marker))
    with pytest.raises(sc.SnapshotError, match="DIFFERENT snapshot"):
        sc.snapshot_cell(src, dst, CELL)
    # the first snapshot is intact
    assert sc.verify_snapshot(dst, CELL)["fit_fingerprint"] == FP


def test_store_identity_mismatch_refuses(tmp_path):
    src = stage_bank(tmp_path / "a")
    dst = stage_bank(tmp_path / "a_r1", cells=(),
                     identity={**IDENTITY, "model": "other/base"})
    with pytest.raises(sc.SnapshotError, match="belongs to another"):
        sc.snapshot_cell(src, dst, CELL)
    # a source with NO store identity is not a bank
    (src / "cells_all" / sc.STORE_IDENT).unlink()
    with pytest.raises(sc.SnapshotError, match="not a bank"):
        sc.snapshot_cell(src, tmp_path / "fresh_r1", CELL)


def test_unpublished_or_unprovable_sources_refuse(tmp_path):
    src = stage_bank(tmp_path / "b")
    live = src / "cells_all" / CELL
    marker = json.loads((live / sc.COMPLETE).read_text())
    # wrong-identity marker
    (live / sc.COMPLETE).write_text(json.dumps(
        {**marker, "cell_id": "constv2-mix-1k__psr_ce_all"}))
    with pytest.raises(sc.SnapshotError, match="wrong-identity"):
        sc.snapshot_cell(src, tmp_path / "b_r1", CELL)
    # pre-fingerprint marker
    (live / sc.COMPLETE).write_text(json.dumps(
        {k: v for k, v in marker.items() if k != "fit_fingerprint"}))
    with pytest.raises(sc.SnapshotError, match="no fit_fingerprint"):
        sc.snapshot_cell(src, tmp_path / "b_r1", CELL)
    # no marker at all
    (live / sc.COMPLETE).unlink()
    with pytest.raises(sc.SnapshotError, match="no COMPLETE marker"):
        sc.snapshot_cell(src, tmp_path / "b_r1", CELL)
    # same dir on both sides
    with pytest.raises(sc.SnapshotError, match="same dir"):
        sc.snapshot_cell(src, src, CELL)


def test_earlier_supersede_record_is_not_carried_into_the_snapshot(banks):
    src, dst = banks
    (src / "cells_all" / CELL / sc.SUPERSEDE_RECORD).write_text(
        json.dumps({"fit_fingerprint": "0" * 64, "snapshot": "elsewhere"}))
    sc.snapshot_cell(src, dst, CELL)
    assert not (dst / "cells_all" / CELL / sc.SUPERSEDE_RECORD).exists()


def test_verify_snapshot_refusals(banks):
    src, dst = banks
    sc.snapshot_cell(src, dst, CELL)
    snap = dst / "cells_all" / CELL
    rp = snap / sc.SNAPSHOT_RECORD
    rec = json.loads(rp.read_text())
    rp.write_text(json.dumps({**rec, "fit_fingerprint": "9" * 64}))
    with pytest.raises(sc.SnapshotError, match="records fingerprint"):
        sc.verify_snapshot(dst, CELL)
    rp.unlink()
    with pytest.raises(sc.SnapshotError, match="not a snapshot cell"):
        sc.verify_snapshot(dst, CELL)
    rp.write_text(json.dumps(rec))
    (dst / "cells_all" / sc.STORE_IDENT).unlink()
    with pytest.raises(sc.SnapshotError, match="_store_identity.json"):
        sc.verify_snapshot(dst, CELL)


def test_cli_snapshot_then_verify_and_rc2_on_refusal(banks, capsys):
    src, dst = banks
    other = "constv2-mix-1m__lora_r4_ce"  # not staged: a refusal
    rc = sc.main(["snapshot", "--src-bank", str(src), "--dst-bank", str(dst),
                  "--cells", f"{CELL},{other}", "--reason", "cap-24 refit",
                  "--expected-fingerprint", f"{CELL}={'b' * 64}"])
    out, err = capsys.readouterr()
    assert rc == 2
    assert f"[snapshot] {CELL}:" in out and "REFUSED " + other in err
    sup = sc.supersede_record(src / "cells_all" / CELL)
    assert sup["superseded_by"] == {"reason": "cap-24 refit",
                                    "expected_refit_fingerprint": "b" * 64}
    assert sc.main(["verify", "--bank", str(dst), "--cells", CELL]) == 0
    assert sc.main(["verify", "--bank", str(dst), "--cells", other]) == 2
