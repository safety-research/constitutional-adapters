"""Review round 4, P2-β — the static rung's idempotent resume requires the FINAL `served_target.json`
and validates its decode against the rung's block.

Before: `escalate_static.existing_complete` accepted a rung root as complete on `escalation.json` + `responses.jsonl` +
`progress.jsonl` + the row count, and `main` returned success without looking at `served_target.json`. A fresh publication
writes the sidecar INTO the staging dir and renames the whole root into place, so a complete-looking root without it is a
partial copy or a lost file — the corruption rule `served_target.assert_target_decode_compatible` documents — and its rows'
sampler / seed policy / trial cannot be established; downstream `ladder_decide.py` then read those rows with no identity
check of its own. Now the resume requires the sidecar, requires `manifest_stage == "final"` (the published manifest, not a
`started` marker copied from somewhere), and runs the strict compatibility check against `rung["block_to"]`, refusing
(exit 2, nothing written, no server contact, no fresh manifest stamped over the root) on a copied-without-sidecar root, a
lost sidecar, a non-final sidecar, a sidecar recording another target decode or none, or an unreadable one.

The scenarios run `escalate_static.main` over the sampled static lane's own builders (`tests/test_sampled_static.py`: the
two-shard base at 4,096, the stub chat with the re-draw oracle, the `_cap8192` rung config).
"""
from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import test_sampled_static as tss  # the sampled static lane's builders (after the tests/ path insert)


def _complete_rung(monkeypatch, tmp_path, capsys):
    """Run the rung once for real (stub chat): the complete root at <tmp>/<ESC_RUN>/pasttense/none/shard00."""
    es = tss._load_script("escalate_static")
    tss._base_store(tmp_path)
    chat = tss._StubChat(decide=tss._redraw)
    tss._wire_escalate(monkeypatch, es, chat)
    cfg_path = tss._esc_config(tmp_path)
    argv = tss._esc_argv(tmp_path, cfg_path)
    assert es.main(argv) == 0
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    shard = tmp_path / tss.ESC_RUN / "pasttense" / "none" / "shard00"
    assert sorted(p.name for p in shard.iterdir()) == ["escalation.json", "progress.jsonl", "responses.jsonl", "served_target.json"]
    return es, chat, cfg_path, argv, shard, first


def _snapshot(shard):
    return {p.name: p.read_bytes() for p in shard.iterdir()}


def _refused(es, argv, capsys):
    with pytest.raises(SystemExit) as ei:
        es.main(argv)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "[escalate-static] REFUSED" in err, err
    return err


def test_the_happy_path_resume_reads_the_final_sidecar_and_touches_nothing(monkeypatch, tmp_path, capsys):
    es, chat, _, argv, shard, first = _complete_rung(monkeypatch, tmp_path, capsys)
    doc = json.loads((shard / "served_target.json").read_text())
    assert doc["manifest_stage"] == "final" and doc["decode"]["cap"] == 8192 and doc["decode"]["trial"] == 0
    n_requests, before = len(chat.payloads), _snapshot(shard)
    assert es.main(argv) == 0
    again = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert again == first and len(chat.payloads) == n_requests and _snapshot(shard) == before


def test_a_rung_root_copied_without_its_sidecar_is_refused_never_resumed_or_stamped(monkeypatch, tmp_path, capsys):
    # a partial copy: the three data files land under a second private root, the sidecar does not (the rsync that stopped
    # early, the cp that missed a file); the base store is copied whole so only the rung root differs
    es, chat, cfg_path, _, shard, _ = _complete_rung(monkeypatch, tmp_path, capsys)
    n_requests = len(chat.payloads)
    copy_root = tmp_path / "copy"
    shutil.copytree(tmp_path / tss.BASE_RUN, copy_root / tss.BASE_RUN)
    dst = copy_root / tss.ESC_RUN / "pasttense" / "none" / "shard00"
    dst.mkdir(parents=True)
    for name in ("escalation.json", "responses.jsonl", "progress.jsonl"):
        (dst / name).write_bytes((shard / name).read_bytes())
    before = _snapshot(dst)
    err = _refused(es, tss._esc_argv(tmp_path, cfg_path, **{"--private-root": str(copy_root)}), capsys)
    assert "served_target.json" in err and "partial copy or a lost file" in err and str(dst) in err
    assert not (dst / "served_target.json").exists()                      # no fresh manifest stamped over the copy
    assert _snapshot(dst) == before and len(chat.payloads) == n_requests  # nothing written, no server contact
    assert not (dst.parent / es.STAGING_NAME).exists()


def test_a_lost_sidecar_is_refused_and_the_root_left_alone(monkeypatch, tmp_path, capsys):
    es, chat, _, argv, shard, _ = _complete_rung(monkeypatch, tmp_path, capsys)
    n_requests = len(chat.payloads)
    (shard / "served_target.json").unlink()
    before = _snapshot(shard)
    err = _refused(es, argv, capsys)
    assert "served_target.json" in err and "partial copy or a lost file" in err
    assert _snapshot(shard) == before and len(chat.payloads) == n_requests
    assert not (shard / "served_target.json").exists()


def test_a_non_final_foreign_legacy_or_unreadable_sidecar_is_refused(monkeypatch, tmp_path, capsys):
    es, chat, _, argv, shard, first = _complete_rung(monkeypatch, tmp_path, capsys)
    n_requests = len(chat.payloads)
    p = shard / "served_target.json"
    doc = json.loads(p.read_text())
    # a `started` marker beside a full set of rows: not the manifest the rows were published with
    p.write_text(json.dumps({**doc, "manifest_stage": "started"}))
    err = _refused(es, argv, capsys)
    assert "manifest_stage='started'" in err and "not 'final'" in err
    # a sidecar recording another target decode: the base cap, another trial, another temperature — each is another store
    for other in ({"cap": 4096}, {"trial": 1}, {"temperature": 1.0}, {"seed_base": 1}):
        p.write_text(json.dumps({**doc, "decode": {**doc["decode"], **other}}))
        err = _refused(es, argv, capsys)
        assert "different target decode" in err, (other, err)
    # a legacy sidecar (no decode block) over a sampled rung: never relabelled
    p.write_text(json.dumps({k: v for k, v in doc.items() if k != "decode"}))
    err = _refused(es, argv, capsys)
    assert "LEGACY" in err
    # an unreadable sidecar
    p.write_text("{not json")
    err = _refused(es, argv, capsys)
    assert "unreadable" in err
    # a progress file recording no finished goal beside the rows: not a complete root either (the identity check must
    # never be skipped for want of covered goals)
    p.write_text(json.dumps(doc))
    prog = shard / "progress.jsonl"
    prog_bytes = prog.read_bytes()
    prog.write_text("")
    err = _refused(es, argv, capsys)
    assert "no successfully finished goal" in err
    prog.write_bytes(prog_bytes)
    # throughout: nothing was written and the server was never contacted; the restored root resumes
    rows_now = (shard / "responses.jsonl").read_bytes()
    assert len(chat.payloads) == n_requests and not (shard.parent / es.STAGING_NAME).exists()
    assert es.main(argv) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == first
    assert (shard / "responses.jsonl").read_bytes() == rows_now and len(chat.payloads) == n_requests


def test_main_hands_the_rungs_block_to_the_resume_check():
    es = tss._load_script("escalate_static")
    assert "expect_decode" in inspect.signature(es.existing_complete).parameters
    src = (tss.SAMPLED_DIR / "escalate_static.py").read_text()
    assert 'existing_complete(shard_dir, n_rows=len(rows), expect_decode=rung["block_to"])' in src
