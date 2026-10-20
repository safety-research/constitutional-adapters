"""``scripts/sdfbase/gsm8k/degeneracy_union_column.py`` — the campaign's degeneracy UNION classifier as a non-deciding column beside a
GSM8K paired-200 cell's cap ladder, over the SYNTHETIC stores of ``tests/gsm8k_synthetic_store.py`` (the real writers, a real Inspect
``.eval`` log): the per-item pairing of the ladder's kind with the union's content detectors, a constructed disagreement landing in
the right table cell by id, the text-identity refusal (the log's text must hash to the ladder record's), the rung store read with the
ladder's rule re-derived (no ``cap_ladder.json`` there), and the write-beside contract (a new file, never an overwrite)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import gsm8k_synthetic_store as G
import pytest
from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.eval import degeneracy_union as DU

REPO = Path(__file__).resolve().parents[1]
HAS_INSPECT = importlib.util.find_spec("inspect_ai") is not None
pytestmark = pytest.mark.skipif(not HAS_INSPECT, reason="reads a real Inspect .eval log")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("degeneracy_union_column", REPO / "scripts" / "sdfbase" / "gsm8k" / "degeneracy_union_column.py")
ROWS = G.matrix_rows()
ESC = ("qwen35_4b", "main", "constv2-mix-full__lora_r4_ce")
LOOPS, TRUNCATED, ZLIB_ONLY = range(0, 5), (5, 6), 7          # item INDICES into G.IDS
AT_CAP = (*LOOPS, *TRUNCATED, ZLIB_ONLY)
LOG_NAME = "2026-01-01T00-00-00"


@pytest.fixture(scope="module")
def escalated(tmp_path_factory):
    """One escalated cell (8 at-cap draws: 5 loops, 2 short truncated draws, 1 draw of a single 3,000-character word — degenerate to the
    union's zlib detector, `truncated` to the ladder whose rep_ratio reads 0 under 8 words) with its rung store (one item still at the
    ceiling), both with their Inspect logs."""
    root = tmp_path_factory.mktemp("g8k_union")
    row = G.row_for(ROWS, *ESC)
    out = G.build_escalated_cell(root, row, at_cap=AT_CAP, loops=LOOPS, write_rung=True, rung_still_at_cap=[TRUNCATED[0]],
                                 texts={ZLIB_ONLY: ("a" * 3000, None)})
    G.write_inspect_log(out["source_cell_dir"], name=LOG_NAME)
    G.write_inspect_log(out["rung_cell_dir"], name=LOG_NAME)
    return out


def test_the_ladders_kind_pairs_with_the_unions_content_detectors_item_by_item_and_the_constructed_disagreement_lands_by_id(escalated):
    r = D.classify_cell(escalated["source_out_root"], tokenizer_cache={}, no_retokenize=True)
    assert r["schema"] == D.SCHEMA and r["deciding"] is False and r["cell"]["is_rung"] is False and r["cell"]["cap"] == G.CAP_BASE
    assert r["sources"]["ladder_facts"] == D.SOURCE_FILED and r["sources"]["cap_ladder"]["exit"] == CL.ESCALATE
    assert r["sources"]["union_module"]["definition_sha"] == DU.DEFINITION_SHA
    c = r["counts"]
    assert c["n_at_cap"] == len(AT_CAP) == escalated["record"]["n_at_cap"]
    assert c["ladder_non_terminating"] == len(LOOPS) == escalated["record"]["n_non_terminating"]
    assert c["ladder_truncated_or_boundary"] == len(TRUNCATED) + 1 == escalated["record"]["n_eligible"]
    by_id = {it["item_id"]: it for it in r["items"]}
    assert set(by_id) == {G.IDS[j] for j in AT_CAP}
    for j in LOOPS:
        it = by_id[G.IDS[j]]
        assert it["ladder_kind"] == CL.KIND_NON_TERMINATING and it["ladder_degenerate"] is True
        assert it["union"]["classifier"] and it["union"]["rep_ratio_loop"] and it["union"]["union_content"]
        assert it["union"]["reason"] and "classifier" in it["union"]["reason"]
    for j in TRUNCATED:
        it = by_id[G.IDS[j]]
        assert it["ladder_kind"] == CL.KIND_TRUNCATED and it["union"]["union_content"] is False and it["union"]["union_content_reason"] is None
    z = by_id[G.IDS[ZLIB_ONLY]]
    assert z["ladder_kind"] == CL.KIND_TRUNCATED and z["ladder_rep_ratio"] == 0.0            # one word: the ladder's rule reads no repetition
    # the union's flags on that draw are the module's own reading of the same text (zlib collapses a 3,000-character run; the
    # rep_ratio detector, like the ladder, reads no 4-gram repetition in one word; 3,000 letters are not contentless)
    expect = DU.classify_response("a" * 3000, cap=G.CAP_BASE, n_tokens=G.CAP_BASE, finish_reason="length", at_cap=True)
    assert expect["zlib_collapse"] is True and expect["rep_ratio_loop"] is False and expect["contentless"] is False
    assert {k: z["union"][k] for k in DU.DETECTORS} == {k: expect[k] for k in DU.DETECTORS}
    assert z["union"]["union_content"] is True and "zlib_collapse" in z["union"]["union_content_reason"]
    # --no-retokenize: n_tokens = the producer's completion_tokens (the cap on an at-cap draw), so the fifth detector fires on every row
    assert all(it["union"]["at_cap"] is True and it["union"]["degenerate"] is True for it in r["items"])
    assert c["union_detectors"]["at_cap"] == c["union_detectors"]["union_any"] == len(AT_CAP)
    assert c["union_detectors"]["union_content"] == len(LOOPS) + 1
    t = r["agreement"]["ladder_x_union_content"]
    assert t["counts"] == {"ladder_degenerate__union_degenerate": len(LOOPS), "ladder_degenerate__union_clean": 0,
                           "ladder_truncated__union_degenerate": 1, "ladder_truncated__union_clean": len(TRUNCATED)}
    assert t["disagreements"] == {"ladder_degenerate_union_clean": [], "ladder_truncated_union_degenerate": [G.IDS[ZLIB_ONLY]]}
    assert t["n_agree"] == len(AT_CAP) - 1 and abs(t["agreement_share"] - (len(AT_CAP) - 1) / len(AT_CAP)) < 1e-12
    # classifier alone (tail_loop / line_loop): it agrees with the ladder on every loop and every short truncated draw; on the
    # 3,000-character run it reads whatever the module reads, so the table is exact either way
    assert r["agreement"]["ladder_x_classifier"]["n_agree"] == len(AT_CAP) - (1 if expect["classifier"] else 0)
    assert r["agreement"]["ladder_x_rep_ratio_loop"]["n_agree"] == len(AT_CAP)
    # the ladder's rule re-derived over the log text agrees with the filed record on every item; the strict-periodic flag is the record's
    assert r["ladder_rule_rederived_disagreements"] == []
    assert all(it["strict_periodic"] in (True, False) for it in r["items"])
    assert c["periodic_block_filed_n_loop"] == escalated["record"]["periodic_block"]["n_loop"] == c["strict_periodic"]
    # no text is stored: ids, hashes, counts and flags only
    blob = json.dumps(r)
    assert "step again" not in blob and "aaaa" not in blob


def test_a_log_whose_text_does_not_hash_to_the_ladder_record_refuses_by_name(escalated, tmp_path):
    import shutil

    root = escalated["source_out_root"]
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    rec_path = copy / CL.RECORD_NAME
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    victim = next(f for f in rec["item_facts"] if f["at_cap"])
    victim["text_sha256"] = "0" * 64
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(D.Refuse):
        D.classify_cell(copy, tokenizer_cache={}, no_retokenize=True)


def test_the_rung_store_has_no_ladder_record_so_the_ladder_rule_and_the_periodic_block_are_rederived_and_filed_as_such(escalated):
    rung_root = escalated["rung_cell_dir"].parents[1]
    assert not (rung_root / CL.RECORD_NAME).exists()
    r = D.classify_cell(rung_root, tokenizer_cache={}, no_retokenize=True)
    assert r["cell"]["is_rung"] is True and r["cell"]["cap"] == G.CAP_RUNG
    assert r["sources"]["cap_ladder"] is None and r["sources"]["ladder_facts"] == D.SOURCE_REDERIVED
    # the one redraw still at the ceiling is the rung's only at-cap draw; its short text is `truncated` under the ladder's rule
    assert r["counts"]["n_at_cap"] == 1
    (it,) = r["items"]
    assert it["item_id"] == G.IDS[TRUNCATED[0]] and it["stop_reason"] == "max_tokens" and it["source_cap"] == G.CAP_RUNG
    assert it["ladder_kind"] == CL.KIND_TRUNCATED and it["union"]["union_content"] is False
    assert it["strict_periodic"] is False and it["periodic_block"]["loop"] is False
    assert r["counts"]["filed_record"] is None and r["counts"]["periodic_block_filed_n_loop"] is None


def test_main_writes_one_new_file_beside_each_store_and_a_summary_and_refuses_to_overwrite(escalated, tmp_path, capsys):
    root = escalated["source_out_root"]
    summary = tmp_path / "summary.md"
    out_name = "degeneracy_union_column.TEST.json"
    target = root / out_name
    try:
        assert D.main(["--cell-root", str(root), "--out-name", out_name, "--summary", str(summary), "--no-retokenize"]) == 0
        assert target.is_file() and summary.is_file()
        doc = json.loads(target.read_text(encoding="utf-8"))
        assert doc["schema"] == D.SCHEMA and doc["counts"]["n_at_cap"] == len(AT_CAP)
        text = summary.read_text(encoding="utf-8")
        assert G.IDS[ZLIB_ONLY] in text and "zlib_collapse" in text and "non-deciding" in text
        before = target.read_bytes()
        with pytest.raises(SystemExit) as e:
            D.main(["--cell-root", str(root), "--out-name", out_name, "--no-retokenize"])
        assert e.value.code == 2 and "exists" in capsys.readouterr().err and target.read_bytes() == before
        assert D.main(["--cell-root", str(root), "--out-name", out_name, "--no-retokenize", "--overwrite"]) == 0
    finally:
        if target.exists():
            target.unlink()
    # the shipped file name and the mapping the union's finish_reason estimand is read through are the module's constants
    assert D.OUT_NAME == "degeneracy_union_column.json" and D.FINISH_REASON_OF_STOP["max_tokens"] == "length"
