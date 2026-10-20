"""Offline tests for scripts/compare_served_hf_mmlu.py (the Phase-2
served-vs-HF MMLU cross-validation) and the instrument-parity pin that
keeps the served MMLU render equal to the HF instrument's. All synthetic
(invented questions, placeholder ids) per the repo privacy rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import compare_served_hf_mmlu as cmp_mod  # noqa: E402
import served_capability_run as srun  # noqa: E402

TASK = "s2parity/mmlu_letter"


def _row(item_id, acc, pick, gold, engine):
    return {"task": TASK, "metric": "acc", "item_id": item_id,
            "value": float(acc), "engine": engine,
            "extra_json": json.dumps({"pick": pick, "gold": gold})}


def _write_rows(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _stores(tmp_path, *, served_flip="m0002-cccccccc"):
    """Four items; the served side flips exactly one pick vs HF."""
    items = [("m0000-aaaaaaaa", "B", "B"), ("m0001-bbbbbbbb", "A", "C"),
             ("m0002-cccccccc", "D", "D"), ("m0003-dddddddd", "C", "C")]
    hf_rows, sv_rows = [], []
    for iid, pick, gold in items:
        hf_rows.append(_row(iid, pick == gold, pick, gold, "hf"))
        sv_pick = "A" if iid == served_flip else pick
        sv_rows.append(_row(iid, sv_pick == gold, sv_pick, gold, "vllm"))
        # the aggregate row must be excluded from pairing
    hf_rows.append({"task": TASK, "metric": "acc_aggregate",
                    "item_id": "__aggregate__", "value": 0.75,
                    "engine": "hf", "extra_json": None})
    return (_write_rows(tmp_path / "hf_rows.jsonl", hf_rows),
            _write_rows(tmp_path / "sv_rows.jsonl", sv_rows))


def test_paired_comparison(tmp_path):
    hf, sv = _stores(tmp_path)
    r = cmp_mod.compare_cell(hf, sv)
    assert r["n_items"] == 4
    assert r["acc_hf"] == pytest.approx(0.75)
    assert r["acc_served"] == pytest.approx(0.50)
    assert r["acc_delta_served_minus_hf"] == pytest.approx(-0.25)
    assert r["pick_agreement"] == pytest.approx(0.75)
    assert len(r["acc_discordant"]) == 1
    assert r["acc_discordant"][0]["item_id"] == "m0002-cccccccc"
    mc = r["mcnemar"]
    assert mc["n_items"] == 4 and mc["n_discordant"] == 1
    assert mc["a_only"] == 1 and mc["b_only"] == 0


def test_perfect_agreement_has_no_discordants(tmp_path):
    hf, sv = _stores(tmp_path, served_flip="none-such")
    r = cmp_mod.compare_cell(hf, sv)
    assert r["acc_hf"] == r["acc_served"]
    assert r["pick_agreement"] == 1.0
    assert r["acc_discordant"] == []
    assert r["mcnemar"]["n_discordant"] == 0
    assert r["mcnemar"]["p_value"] is None  # exact test: no discordants


def test_item_set_mismatch_refuses(tmp_path):
    hf, sv = _stores(tmp_path)
    rows = [json.loads(l) for l in sv.read_text().splitlines()]
    _write_rows(sv, rows[:-1])  # drop one served item
    with pytest.raises(cmp_mod.CompareError, match="item sets differ"):
        cmp_mod.compare_cell(hf, sv)


def test_wrong_engine_on_either_side_refuses(tmp_path):
    hf, sv = _stores(tmp_path)
    with pytest.raises(cmp_mod.CompareError, match="engine"):
        cmp_mod.compare_cell(sv, hf)  # swapped: served handed in as HF


# --------------------------------------------------------------------------- #
# instrument parity: the served render's pieces == parity's render             #
# --------------------------------------------------------------------------- #


class _NoTemplateTok:
    """build_chat_prompt's template-less fallback path — enough to pin
    that _mmlu_user_text + MMLU_ANSWER_SEED reconstructs the instrument's
    own prompt text exactly (the templated equality is verified against
    the real pinned tokenizers by the Modal preflight)."""

    chat_template = None


def test_user_text_plus_seed_is_the_instrument_render():
    from sisteer.capeval.tiers import parity

    row = {"question": "  Which letter names the second choice?  ",
           "choices": ["alpha", "beta", "gamma", "delta", "IGNORED-5TH"],
           "answer": 1}
    tok = _NoTemplateTok()
    want = parity._mmlu_prompt(tok, row, system=None)
    from sisteer.models import build_chat_prompt

    got = (build_chat_prompt(tok, srun._mmlu_user_text(row))
           + srun.MMLU_ANSWER_SEED)
    assert got == want
    # the 5th choice is truncated on BOTH sides (the instrument's zip)
    assert "IGNORED-5TH" not in got
