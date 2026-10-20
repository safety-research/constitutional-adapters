"""Tests for scripts/sdfbase/build_heldout_split.py.

Pins: carve determinism; advice carved at (principle_id, prompt_type_id)
 — planted A3 siblings NEVER straddle the split; stories group-key
disjointness + call_id never split; the 5% walk measured under the
AUTHORITATIVE set rule (min over tokenizers of the set's per-tokenizer
SUMS — never sum of per-record mins).

Mechanics tests run on tiny synthetic corpora (CPU, fast). The full-corpus
check reads the real v2 finals read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sdfbase"))

import build_heldout_split as hs  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic corpus helpers (invented text, placeholder ids — no real data)
# ---------------------------------------------------------------------------

def adv_rec(p: int, t: int, s: int, counts=(10, 12, 11), system="sys text"):
    pid = f"p{p:02d}"
    tid = f"{pid}-t{t:02d}"
    sid = f"{tid}-s{s:02d}"
    return {
        "system": system,
        "user": f"user text {sid}",
        "assistant": f"assistant text {sid}",
        "principle_id": pid,
        "prompt_type_id": tid,
        "scenario_id": sid,
        "lineage": {"custom_id": f"adv-{sid}"},
        "token_counts": {"llama3": counts[0], "gemma": counts[1], "qwen": counts[2]},
    }


def sto_rec(c: int, u: int, k: int, y: int, counts=(20, 22, 21)):
    cid = f"c{c:02d}"
    uid = f"{cid}-u{u:02d}"
    kid = f"{uid}-k{k:02d}"
    return {
        "text": f"story text {kid}-y{y}",
        "category_id": cid,
        "subcategory_id": uid,
        "call_id": kid,
        "lineage": {"custom_id": f"sto-{kid}-y{y}"},
        "token_counts": {"llama3": counts[0], "gemma": counts[1], "qwen": counts[2]},
    }


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


def synthetic_corpus():
    # advice: 6 (p,t) groups x 4 scenarios (the scenarios are the planted A3
    # siblings — they share one generation context and must never straddle)
    advice = [adv_rec(p, 0, s) for p in range(6) for s in range(4)]
    # stories: 4 subcategories x 3 calls x 2 records per call
    stories = [
        sto_rec(c, u, k, y)
        for c in range(2)
        for u in range(2)
        for k in range(3)
        for y in range(2)
    ]
    return advice, stories


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_carve_determinism():
    advice, _ = synthetic_corpus()
    a = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.2)
    b = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.2)
    assert a == b


def test_build_split_determinism(tmp_path):
    advice, stories = synthetic_corpus()
    ap = write_jsonl(tmp_path / "advice.jsonl", advice)
    sp = write_jsonl(tmp_path / "stories.jsonl", stories)
    s1 = hs.build_split(ap, sp, frac=0.2, eval_n=4)
    s2 = hs.build_split(ap, sp, frac=0.2, eval_n=4)
    assert s1 == s2
    assert s1["eval_subset"]["n"] == 4
    assert sum(s1["eval_subset"]["family_counts"].values()) == 4
    # eval ids come from the held-out side only
    held = set(s1["advice"]["ids"]) | set(s1["stories"]["ids"])
    assert set(s1["eval_subset"]["ids"]) <= held


# ---------------------------------------------------------------------------
# Group integrity — planted siblings never straddle
# ---------------------------------------------------------------------------

def test_planted_a3_siblings_never_straddle():
    advice, _ = synthetic_corpus()
    res = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.3)
    held_ids = set(res["ids"])
    assert held_ids, "carve held nothing out"
    by_group: dict[str, set] = {}
    for r in advice:
        by_group.setdefault(hs.advice_group_key(r), set()).add(hs.doc_id(r))
    straddlers = [
        g for g, ids in by_group.items()
        if ids & held_ids and ids - held_ids
    ]
    assert straddlers == [], f"A3 siblings straddle the split: {straddlers}"
    # carve granularity is the (principle_id, prompt_type_id) group, and the
    # held-out groups list matches the held-out ids exactly
    assert set(res["groups"]) == {
        g for g, ids in by_group.items() if ids <= held_ids
    }


def test_story_groups_disjoint_and_call_never_split():
    _, stories = synthetic_corpus()
    res = hs.carve_family(stories, hs.story_group_key, hs.SALT_HELDOUT, 0.3)
    held_ids = set(res["ids"])
    assert held_ids
    by_subcat: dict[str, set] = {}
    by_call: dict[str, set] = {}
    for r in stories:
        by_subcat.setdefault(r["subcategory_id"], set()).add(hs.doc_id(r))
        by_call.setdefault(r["call_id"], set()).add(hs.doc_id(r))
    for name, groups in (("subcategory", by_subcat), ("call", by_call)):
        split_groups = [
            g for g, ids in groups.items() if ids & held_ids and ids - held_ids
        ]
        assert split_groups == [], f"{name} split across sides: {split_groups}"


def test_call_prefix_violation_fails_loudly():
    advice, stories = synthetic_corpus()
    bad = dict(stories[0])
    bad["call_id"] = "zz99-x00-k00"  # does not embed subcategory_id
    bad["lineage"] = {"custom_id": "sto-planted-bad-call"}
    with pytest.raises(hs.SplitError, match="call_id"):
        hs.verify_structure(advice, stories + [bad])


# ---------------------------------------------------------------------------
# Authoritative set rule (never sum-of-per-record-mins)
# ---------------------------------------------------------------------------

def test_walk_uses_authoritative_set_rule():
    # Group keys first, so the A/B count patterns can be assigned to the
    # WALK order (the sha rank of the keys), not the file order.
    keys = [f"p{p:02d}|p{p:02d}-t00" for p in range(4)]
    ranked = sorted(sorted(keys), key=lambda k: (hs.sha_rank(hs.SALT_HELDOUT, k), k))
    # walk position 0 gets pattern A, position 1 pattern B, then A, B.
    # A: min lives in gemma; B: min lives in llama3 — the limiting tokenizer
    # flips between subsets, so sum-of-record-mins diverges from the set rule.
    pattern = {
        ranked[0]: (10, 2, 10),
        ranked[1]: (2, 10, 10),
        ranked[2]: (10, 2, 10),
        ranked[3]: (2, 10, 10),
    }
    advice = []
    for key in keys:
        p = int(key.split("|")[0][1:])
        advice.append(adv_rec(p, 0, 0, counts=pattern[key]))

    # Family totals: llama3 24, gemma 24, qwen 40 -> official 24.
    # frac 0.5 -> threshold 12. Walk: after ranked[0] cum={10,2,10} min 2;
    # after ranked[1] cum={12,12,20} min 12 >= 12 -> STOP at 2 groups.
    # Sum-of-record-mins of those 2 groups is 4 — an implementation using it
    # would keep walking past 2 groups.
    res = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.5)
    assert res["groups"] == ranked[:2]
    acc = res["accounting"]
    assert acc["family_official"] == 24
    assert acc["heldout_official"] == 12
    assert acc["heldout_per_tokenizer"] == {"llama3": 12, "gemma": 12, "qwen": 20}
    sum_of_mins = sum(
        min(r["token_counts"].values())
        for r in advice
        if hs.advice_group_key(r) in set(res["groups"])
    )
    assert sum_of_mins == 4 != acc["heldout_official"]


def test_walk_stops_minimally():
    # 10 uniform groups of 10 official tokens; frac 0.25 -> threshold 25;
    # the walk must stop at 3 groups (30 >= 25) and not earlier/later.
    advice = [adv_rec(p, 0, 0, counts=(10, 10, 10)) for p in range(10)]
    res = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.25)
    acc = res["accounting"]
    assert len(res["groups"]) == 3
    assert acc["heldout_official"] == 30
    # dropping the last-added group would fall below the threshold
    assert acc["heldout_official"] - 10 < 0.25 * acc["family_official"]


def test_unattainable_frac_fails_loudly():
    advice = [adv_rec(0, 0, 0)]
    with pytest.raises(hs.SplitError, match="not attainable"):
        hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 1.5)


def test_eval_subset_larger_than_heldout_fails(tmp_path):
    advice, stories = synthetic_corpus()
    ap = write_jsonl(tmp_path / "advice.jsonl", advice)
    sp = write_jsonl(tmp_path / "stories.jsonl", stories)
    with pytest.raises(hs.SplitError, match="eval subset"):
        hs.build_split(ap, sp, frac=0.2, eval_n=10_000)


# ---------------------------------------------------------------------------
# Full corpus (read-only on the real v2 finals)
# ---------------------------------------------------------------------------

REAL_FILES = hs.ADVICE_FINAL.exists() and hs.STORIES_FINAL.exists()


@pytest.mark.skipif(not REAL_FILES, reason="real corpus finals not present")
def test_real_split_build():
    split = hs.build_split(hs.ADVICE_FINAL, hs.STORIES_FINAL)
    acc_a = split["token_totals"]["advice"]
    acc_s = split["token_totals"]["stories"]
    # 372 A3-call groups in the advice family
    assert acc_a["family_groups"] == 372
    assert acc_a["family_records"] == 1960
    assert acc_s["family_records"] == 16321
    for acc in (acc_a, acc_s):
        # >= 5%, and not wildly past it (whole-group overshoot only)
        assert 0.05 <= acc["heldout_frac_realized"] < 0.07
        assert acc["heldout_records"] + acc["train_records"] == acc["family_records"]
    assert split["eval_subset"]["n"] == 128
    assert len(split["eval_subset"]["ids"]) == 128
    held = set(split["advice"]["ids"]) | set(split["stories"]["ids"])
    assert set(split["eval_subset"]["ids"]) <= held
    # the committed artifact, if present, must be exactly this build
    if hs.SPLIT_OUT.exists():
        committed = json.loads(hs.SPLIT_OUT.read_text(encoding="utf-8"))
        assert committed == split, (
            "committed data/sdfbase/heldout_split.json differs from a fresh "
            "deterministic rebuild"
        )
