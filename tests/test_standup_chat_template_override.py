"""mt_serving_up.serve_chat_template: the served template is the tokenizer's own unless a declared override
file replaces it; the record names both hashes so provenance shows the substitution."""
from __future__ import annotations

import hashlib
import pathlib
import sys
from types import SimpleNamespace

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
import mt_serving_up as mt  # noqa: E402


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def test_default_keeps_the_tokenizers_template():
    tok = SimpleNamespace(chat_template="{{ messages }}")
    rec = mt.serve_chat_template(tok, None)
    assert tok.chat_template == "{{ messages }}"
    assert rec == {"override": None, "tokenizer_chat_template_sha16": _sha16("{{ messages }}")}


def test_override_replaces_the_template_and_records_both_hashes(tmp_path):
    tok = SimpleNamespace(chat_template="{{ minimal }}")
    f = tmp_path / "stock.jinja"; f.write_text("{{ bos_token }}Cutting Knowledge Date: December 2023\n{{ messages }}")
    rec = mt.serve_chat_template(tok, f)
    assert tok.chat_template == f.read_text()
    assert rec["override"] == str(f) and rec["override_sha16"] == _sha16(f.read_text())
    assert rec["tokenizer_chat_template_sha16"] == _sha16("{{ minimal }}")


def test_override_must_exist_and_be_non_empty(tmp_path):
    tok = SimpleNamespace(chat_template="x")
    with pytest.raises(SystemExit, match="no such file"):
        mt.serve_chat_template(tok, tmp_path / "missing.jinja")
    e = tmp_path / "empty.jinja"; e.write_text("  \n")
    with pytest.raises(SystemExit, match="empty"):
        mt.serve_chat_template(tok, e)


def test_the_vendored_stock_template_is_what_the_sbatch_serves_for_llama8b():
    sb = (REPO / "scripts/sdfbase/full22/misalign/misalign_proxy.sbatch").read_text()
    assert 'llama8b_org) TPL_OVERRIDE="configs/capeval/chat_templates/llama31_instruct_stock.jinja"' in sb
    assert "--chat-template" in sb
