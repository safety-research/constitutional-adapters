"""scripts/verify_fit_template_provenance.py: the honest
analogue of a provenance back-fill — it verifies every staged object's
recorded fit-time template hash against the tokenizer its record pins, and
WRITES NOTHING into the bank.

Synthetic bank + fake tokenizers. Pinned: the four verdicts (a template-free
tokenizer whose record carries sha256("") AGREES — a measured absence, not a
gap; a shipped template agrees when the hash matches; a record that does not
describe the pinned tokenizer is a DISAGREE and fails the run; a partial
record is UNRECORDED), that a document-fit cell stays distinguishable from a
chat-rendered one in the report (provenance_status), that the tool is a
no-op on the bank (tree digest unchanged; two runs produce the same rows),
and that it refuses to write its report under the bank.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_fit_provenance_lineages import (  # noqa: E402
    BASE_MODEL, BASE_REV, CELL, EMPTY_HASH, IT_MODEL, IT_REV, SERVED_HASH,
    _stage_doc, _stage_pair_fit)


def _tool():
    spec = importlib.util.spec_from_file_location(
        "verify_fit_template_provenance",
        REPO / "scripts" / "verify_fit_template_provenance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeTok:
    def __init__(self, template):
        self.chat_template = template
        self.pad_token = "x"
        self.eos_token = "x"


IT_TEMPLATE = "{% for m in messages %}fake instruct template{% endfor %}"
assert hashlib.sha256(IT_TEMPLATE.encode()).hexdigest()[:16] != SERVED_HASH


def _bank(tmp_path: Path) -> Path:
    bank = tmp_path / "bank"
    cells = bank / "tgt" / "cells"
    _stage_doc(cells / f"base__{CELL}")                       # empty hash, template-free base
    # a chat-rendered fit on the -it whose record DESCRIBES the fake -it template
    _stage_pair_fit(cells / "si12__safety_refusal__psr_kl_all")
    meta = cells / "si12__safety_refusal__psr_kl_all" / "train_budget_meta.json"
    rec = json.loads(meta.read_text())
    rec["template_hash"] = hashlib.sha256(IT_TEMPLATE.encode()).hexdigest()[:16]
    meta.write_text(json.dumps(rec))
    # a record that does NOT describe its pinned tokenizer: says the -it model
    # but carries a hash no template of it produces
    _stage_pair_fit(cells / "si12__bogus__psr_kl_all")
    meta = cells / "si12__bogus__psr_kl_all" / "train_budget_meta.json"
    rec = json.loads(meta.read_text()); rec["template_hash"] = "deadbeefdeadbeef"
    meta.write_text(json.dumps(rec))
    # a partial record: the object's own save sidecar names the model only
    d = cells / "legacy__antidote"; d.mkdir(parents=True)
    torch.save({"kind": "vector"}, d / "object.pt")
    (d / "object.json").write_text(json.dumps({"model_name": IT_MODEL}))
    return bank


def _fake_tokenizers(monkeypatch, tool):
    toks = {(BASE_MODEL, BASE_REV): FakeTok(None),          # ships NO template
            (IT_MODEL, IT_REV): FakeTok(IT_TEMPLATE)}

    def load(model_id, revision, overrides):
        return toks[(model_id, revision)]
    monkeypatch.setattr(tool, "_load_tok", load)
    return toks


def test_verdicts_and_no_write(tmp_path, monkeypatch):
    tool = _tool()
    bank = _bank(tmp_path)
    _fake_tokenizers(monkeypatch, tool)
    before = tool.tree_digest(bank)

    report = tool.verify_bank(bank)
    by = {r["staged_id"]: r for r in report["rows"]}
    doc = by[f"base__{CELL}"]
    assert doc["verdict"] == "agree"
    assert doc["recorded_template_hash"] == doc["recomputed_template_hash"] == EMPTY_HASH
    assert doc["pinned_tokenizer_ships_chat_template"] is False
    assert doc["provenance_status"] == "document_fit"          # distinguishable
    pair = by["si12__safety_refusal__psr_kl_all"]
    assert pair["verdict"] == "agree" and pair["provenance_status"] == "verified"
    assert pair["pinned_tokenizer_ships_chat_template"] is True
    bogus = by["si12__bogus__psr_kl_all"]
    assert bogus["verdict"] == "disagree"
    assert bogus["recorded_template_hash"] == "deadbeefdeadbeef"
    assert bogus["recomputed_template_hash"] == pair["recomputed_template_hash"]
    assert by["legacy__antidote"]["verdict"] == "unrecorded"
    assert by["legacy__antidote"]["provenance_status"] == "partial"
    assert report["counts"] == {"agree": 2, "disagree": 1, "unrecorded": 1,
                                "unresolved": 0}
    # no-write: the bank is byte-for-byte (path, size, mtime) as before
    assert tool.tree_digest(bank) == before == report["bank_tree_digest"]
    # idempotent: a second run yields the same rows
    again = tool.verify_bank(bank)
    assert again["rows"] == report["rows"]


def test_cli_exit_codes_and_out_path_guard(tmp_path, monkeypatch, capsys):
    tool = _tool()
    bank = _bank(tmp_path)
    _fake_tokenizers(monkeypatch, tool)
    # a disagree in the bank -> exit 2, report written atomically outside it
    out = tmp_path / "reports" / "r.json"
    rc = tool.main(["--bank-root", str(bank), "--out", str(out)])
    assert rc == 2
    rep = json.loads(out.read_text())
    assert rep["counts"]["disagree"] == 1
    assert not list(out.parent.glob("*.tmp-*"))
    text = capsys.readouterr().out
    assert "disagree" in text and "si12__bogus__psr_kl_all" in text
    # a report path under the bank is refused before anything runs
    with pytest.raises(SystemExit, match="under the bank"):
        tool.main(["--bank-root", str(bank), "--out", str(bank / "tgt" / "r.json")])
    # only the agreeing cells -> exit 0
    for bad in ("si12__bogus__psr_kl_all", "legacy__antidote"):
        d = bank / "tgt" / "cells" / bad
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    assert tool.main(["--bank-root", str(bank)]) == 0


def test_unresolved_when_the_pinned_tokenizer_cannot_load(tmp_path, monkeypatch):
    tool = _tool()
    bank = _bank(tmp_path)

    def load(model_id, revision, overrides):
        raise OSError("no such snapshot in the cache")
    monkeypatch.setattr(tool, "_load_tok", load)
    report = tool.verify_bank(bank)
    verdicts = {r["staged_id"]: r["verdict"] for r in report["rows"]}
    assert verdicts[f"base__{CELL}"] == "unresolved"
    assert verdicts["legacy__antidote"] == "unrecorded"   # never needed a tokenizer
    assert report["counts"]["unresolved"] == 3
