"""The Qwen seed-replicate battery: ``escalate_static``'s re-drawn row
carries the ``request_id`` / ``completion_id`` its detailed record carries — the ids ``bind_rows`` reads off the
assembled static rung store — copied forward ONLY when present on the record.

  * records without the ids (today's served re-draw): the rung store is byte for byte the script's before this change
    (the REAL script at :data:`SCRIPT_BEFORE`, loaded by path, same stub, same base store);
  * records with the ids: every re-drawn row (``regenerated`` true) carries exactly them; every copied-forward row
    (``regenerated`` false) carries neither and is byte-identical to the reference — ``source_cap`` and its provenance
    untouched; KNOWN-BAD: the script before this change drops them from the very same records.

Offline; test_sampled_static's stub world."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import test_sampled_static as tss  # noqa: E402
from _seedrep_testlib import _load  # noqa: E402
from seedrep_history_fixtures import pinned_commit_or_skip  # noqa: E402

SCRIPT = "scripts/sdfbase/full22/sampled/escalate_static.py"
#: the script before the copy-forward (the branch tip that preceded it)
SCRIPT_BEFORE = "acfa735921983326a8ec88b4d3ee40df6a35dfc6"
AT_CAP = {("g0", 1), ("g1", 1), ("g3", 0)}          # the stub's at-cap items: the rows this rung re-draws
ID_KEYS = ("request_id", "completion_id")



def _script_before(tmp_path: Path):
    """The script at :data:`SCRIPT_BEFORE` out of the object store, under a repo-shaped directory (its ``REPO`` is
    ``parents[4]``; the sibling modules it imports are already on ``sys.path`` from the tree's own load)."""
    pinned_commit_or_skip(SCRIPT_BEFORE, "seedrep-ws1c", "escalate_static before the id copy-forward", repo=REPO, anchor=Path(__file__))
    show = subprocess.run(["git", "-C", str(REPO), "show", f"{SCRIPT_BEFORE}:{SCRIPT}"], capture_output=True)
    if show.returncode != 0:
        pytest.fail(f"git show {SCRIPT_BEFORE}:{SCRIPT} failed: {show.stderr.decode()[-300:]}")
    src = tmp_path / "before" / Path(SCRIPT)
    src.parent.mkdir(parents=True)
    src.write_bytes(show.stdout)
    return _load(f"escalate_static_{SCRIPT_BEFORE}", str(src))


def _with_ids(target, monkeypatch):
    """The target's records carry the ids the streaming ladder's materialised draws (or a client) put on them."""
    orig = target.generate_detailed

    def generate_detailed(prompts, **kw):
        recs = orig(prompts, **kw)
        for rec, seed in zip(recs, kw["seeds"]):
            rec["request_id"], rec["completion_id"] = f"rq-{seed}", f"cmpl-{seed}"
        return recs
    monkeypatch.setattr(target, "generate_detailed", generate_detailed)


def _rung(es, root: Path, monkeypatch, *, ids: bool):
    """One rung assembly of the stub world under ``root`` through ``es.main``; the shard's rows."""
    tss._base_store(root)
    target, _ = tss._wire_escalate(monkeypatch, es, tss._StubChat(decide=tss._redraw))
    if ids:
        _with_ids(target, monkeypatch)
    assert es.main(tss._esc_argv(root, tss._esc_config(root))) == 0
    shard = root / tss.ESC_RUN / "pasttense" / "none" / "shard00"
    return (shard / "responses.jsonl").read_bytes(), tss._rows(shard / "responses.jsonl")


def test_records_without_ids_assemble_the_rung_store_byte_for_byte_as_before(tmp_path, monkeypatch, capsys):
    es_now = tss._load_script("escalate_static")
    now_bytes, now_rows = _rung(es_now, tmp_path / "now", monkeypatch, ids=False)
    assert not any(set(ID_KEYS) & set(r) for r in now_rows)                              # history-independent: outside the skipped portion
    assert {(r["goal_id"], r["sample_idx"]) for r in now_rows if r["regenerated"]} == AT_CAP
    es_before = _script_before(tmp_path)                                                  # the object store: skip/fail decided here
    before_bytes, _ = _rung(es_before, tmp_path / "before_root", monkeypatch, ids=False)
    assert now_bytes == before_bytes


def test_records_with_ids_put_them_on_every_redrawn_row_and_leave_copied_rows_untouched(tmp_path, monkeypatch, capsys):
    es_now = tss._load_script("escalate_static")
    _, reference = _rung(es_now, tmp_path / "ref", monkeypatch, ids=False)
    _, rows = _rung(es_now, tmp_path / "ids", monkeypatch, ids=True)
    assert len(rows) == len(reference) and [r["goal_id"] for r in rows] == [r["goal_id"] for r in reference]
    for got, ref in zip(rows, reference):
        if got["regenerated"]:
            assert (got["goal_id"], got["sample_idx"]) in AT_CAP and got["source_cap"] == 8192
            assert got["request_id"] == f"rq-{got['sampling_seed']}" and got["completion_id"] == f"cmpl-{got['sampling_seed']}"
            assert {k: v for k, v in got.items() if k not in ID_KEYS} == ref              # the ids are the ONLY addition
        else:
            assert got == ref and not set(ID_KEYS) & set(got)                              # copied forward: untouched
            assert got["regenerated"] is False and got["source_cap"] == 4096 and got["escalated_from_cap"] == 4096
    # KNOWN-BAD: the script before this change drops the ids from the very same records
    _, old_rows = _rung(_script_before(tmp_path), tmp_path / "old", monkeypatch, ids=True)
    assert not any(set(ID_KEYS) & set(r) for r in old_rows) and [r["regenerated"] for r in old_rows] == [r["regenerated"] for r in rows]
