"""`configs/capeval/tasks/tool_parser_fixtures.json` — the fixture file is
self-consistent under the grammar (schema 2: the definite fixtures, the either-of class,
the edges), both lane families pin their row's hf / rev / chat-template hash / manifest,
the manifests the probe standup composes pin ``tool_parser: qwen3_xml`` at the same
template hash, and (when the pinned tokenizers are in the local HF cache) the pinned
TEMPLATE-RENDERED spelling regenerates byte for byte."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import bfcl_parser_probe as P  # noqa: E402


@pytest.fixture(scope="module")
def fixtures():
    return P.load_fixtures(REPO)


def test_fixture_file_is_self_consistent_and_names_the_lane_rows(fixtures):
    assert fixtures["schema"] == P.FIXTURES_SCHEMA == 2 and fixtures["generated_by"] == P.GENERATED_BY
    assert set(fixtures) == {"schema", "generated_by", "generated_at", "parser", "grammar", "tools", "fixtures", "either_of", "edges", "families"}
    assert P.fixture_problems(fixtures) == []
    # the two ruled-in-grammar cases are the ASSERTED either-of class, not definite fixtures and not record-only edges
    assert [f["id"] for f in fixtures["fixtures"]] == ["single_string", "typed_params", "parallel_two", "multiline_value",
                                                      "content_then_call", "no_call_plain", "irrelevance_answer", "unclosed_trailing_block"]
    assert [f["id"] for f in fixtures["either_of"]] == ["null_value", "undeclared_param"]
    assert [f["id"] for f in fixtures["edges"]] == ["bare_function_no_wrapper", "empty_block"]
    for fx in fixtures["either_of"]:
        assert set(fx) == set(P.EITHER_OF_FIELDS) and len(fx["expected_any_of"]) == 2 and "EITHER admitted" in fx["note"]
    assert "EITHER admitted" in fixtures["grammar"] and "literal null -> null" not in fixtures["grammar"]
    for fam, slug in P.PROBE_ROWS.items():
        row = C.model_row(slug)
        blk = fixtures["families"][fam]
        assert (blk["row"], blk["hf"], blk["rev"]) == (row.slug, row.hf, row.rev)
        assert blk["chat_template_hash"] == P.TEMPLATE_HASHES[fam] and blk["parser"] == "qwen3_xml"
        assert blk["manifest"] == P.PROBE_MANIFESTS[fam] == (row.petri_manifest or blk["manifest"])
        assert blk["rendered"].startswith("<tool_call>") and blk["rendered"].endswith("</tool_call>")
        assert blk["rendered"].count("<tool_call>") == len(P.CANONICAL_CALLS) == 2
    # per family: both pairings are PINNED — qwen3_5 since 2026-09-22 (the offline fixtures ruled in-grammar; live
    # PASS 80df464), nemotron_h since its live probe PASSED on the staged tree 8365992 (2026-09-23; the
    # verdict is checked in at configs/capeval/tasks/parser_probe/nemotron_h_verdict_8365992.json) — each row carries the
    # manifest the fixtures name and its family's pin in vllm_server.TOOL_PARSER_BY_FAMILY
    assert C.model_row(P.PROBE_ROWS["qwen3_5"]).petri_manifest == P.PROBE_MANIFESTS["qwen3_5"] == C.PETRI_MANIFEST_FULL22_QWEN35_4B
    assert C.model_row(P.PROBE_ROWS["nemotron_h"]).petri_manifest == P.PROBE_MANIFESTS["nemotron_h"] == C.PETRI_MANIFEST_FULL22_NEMOTRON3NANO4B
    from sisteer.capeval.serving import vllm_server as vs
    assert vs.TOOL_PARSER_BY_FAMILY.get("qwen3_5") == "qwen3_xml" == P.PARSER == vs.TOOL_PARSER_BY_FAMILY.get("nemotron_h")
    assert C.TOOL_PARSER_LIVE_PASS["nemotron_h"]["sha7"] == "8365992"


def test_the_probe_manifests_pin_qwen3_xml_at_the_fixtures_template_hash(fixtures):
    for fam, rel in P.PROBE_MANIFESTS.items():
        doc = yaml.safe_load((REPO / rel).read_text())["petri"]
        row = C.model_row(P.PROBE_ROWS[fam])
        assert doc["tool_parser"] == "qwen3_xml" and doc["chat_template_hash"] == P.TEMPLATE_HASHES[fam]
        assert (doc["target_model"], doc["target_revision"]) == (row.hf, row.rev)
    # the full22 config generator's own pins (the source of the hashes) agree
    text = (REPO / "scripts/sdfbase/make_full22_configs.py").read_text()
    for key, fam in (("qwen35_4b", "qwen3_5"), ("nemotron3nano4b", "nemotron_h")):
        block = re.search(rf'"{key}": \{{.*?\}}', text, re.DOTALL).group(0)
        assert P.TEMPLATE_HASHES[fam] in block and '"tool_parser": "qwen3_xml"' in block


def test_the_two_families_render_the_same_grammar_spelling(fixtures):
    """Both templates spell the assistant tool-call turn identically (the parser pairing is the
    SAME for both, which is what lets one parser serve both families)."""
    q, n = fixtures["families"]["qwen3_5"]["rendered"], fixtures["families"]["nemotron_h"]["rendered"]
    assert q == n
    parsed = P.parse_tool_calls_xml(q, fixtures["tools"])
    assert parsed["tool_calls"] == [{"name": c["name"], "arguments": dict(sorted(c["arguments"].items()))} for c in P.CANONICAL_CALLS]


def _cached_tokenizers():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out = {}
    for fam, slug in P.PROBE_ROWS.items():
        try:
            out[fam] = P.load_pinned_tokenizer(C.model_row(slug))
        except Exception as e:  # noqa: BLE001 — not cached here: the Modal check renders on the volume's cache
            pytest.skip(f"pinned tokenizer for {fam} not in the local HF cache ({type(e).__name__})")
    return out


def test_pinned_rendered_spelling_regenerates_through_the_cached_pinned_tokenizers(fixtures):
    toks = _cached_tokenizers()
    assert P.template_spelling_problems(fixtures, toks) == []
    for fam, tok in toks.items():
        assert P.template_hash(tok) == P.TEMPLATE_HASHES[fam]
        # the eos every raw capture ends with, stripped before the replay
        assert tok.eos_token == "<|im_end|>"
        ids = tok.encode(fixtures["families"][fam]["rendered"] + tok.eos_token, add_special_tokens=False)
        assert P.strip_trailing_eos(tok.decode(ids, skip_special_tokens=False), tok.eos_token) == fixtures["families"][fam]["rendered"]
