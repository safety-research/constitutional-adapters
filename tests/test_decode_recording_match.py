"""The decode-policy match verdict table, through the REAL matcher: `v_run_decode_policy_match`'s SQL body and
`decode_match.tiebreak` over the real seeds ingested (the worked example's `qwen35_4b_demo` policy rows and the seeded `eval` table),
with one hypothetical run per family carrying exactly the sent set the producers' evidence yields -- what the forward fix and the
backfill will file. Expected (the registry owner, 2026-10-13): controls / statics / agentic / mmlu -> EXACT; BoN -> no_match until the
owner's (A) (the grandfathered T=1 row is superseded by the vendor row since 2026-09-17); multiturn -> no_match until (B) (the runs'
`decode_seed_rule none` vs the vendor row's `per_goal_turn`); gsm8k -> ambiguous in the SQL (eval row beside its category row), resolved
exact by the stamp pass's eval-over-category tie-break; an unmodelled key in the sent set -> no_match; an empty sent set -> unrecorded."""
from __future__ import annotations

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registry_decode_llama_thinking_na import idx, seeded_db, seeds  # noqa: E402,F401  (fixtures: the real seeds, ingested)

from sisteer.registry import decode_match as DM  # noqa: E402

MODEL = "qwen35_4b_demo"
SIX = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
T1 = {"temperature": 1.0}
CARD5 = {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.6, "top_k": 20, "top_p": 0.95}       # the gsm8k thinking_card `--expect` reference
GEN = "2026-10-11T15:00:00.000Z"                                                                       # the demo smoke's generation day
P = {"over_refusal": "qwen35_4b_demo|over_refusal|off|vendor_preset@v1", "static": "qwen35_4b_demo|static|off|vendor_preset@v1",
     "agentic": "qwen35_4b_demo|agentic|off|runner_forced_T1@v1", "mmlu": "qwen35_4b_demo|mmlu|off|greedy@v1",
     "bon_T1": "qwen35_4b_demo|bon|off|grandfathered_T1@v1", "bon_vendor": "qwen35_4b_demo|bon|off|vendor_preset@v1",
     "mt_vendor": "qwen35_4b_demo|multi_turn|off|vendor_preset@v1", "gsm8k_card": "qwen35_4b_demo|gsm8k|on|thinking_card@v1",
     "cap_card": "qwen35_4b_demo|capability|on|thinking_card@v1"}

#: (run id, eval, thinking, seed rule, decode_source, sent set, generated_utc, expected SQL match, expected policy id, expected candidates)
CASES = [
    ("xstest_vendor", "xstest", "off", "per_item", "model_card", SIX, GEN, "exact", P["over_refusal"], [P["over_refusal"]]),
    ("orbench_vendor", "or_bench_hard", "off", "per_item", "model_card", SIX, GEN, "exact", P["over_refusal"], [P["over_refusal"]]),
    ("codeattack_vendor", "codeattack", "off", "per_item", "model_card", SIX, GEN, "exact", P["static"], [P["static"]]),
    ("grok420_padding_vendor", "grok420", "off", "per_item", "model_card", SIX, GEN, "exact", P["static"], [P["static"]]),
    ("agentic_murder_T1", "agentic_murder", "off", "none", "runner_forced_T1", T1, GEN, "exact", P["agentic"], [P["agentic"]]),
    ("mmlu_greedy_stated", "mmlu", "off", "none", "greedy", {"temperature": 0.0}, GEN, "exact", P["mmlu"], [P["mmlu"]]),
    ("mmlu_unrecorded", "mmlu", "off", "none", "greedy", {}, GEN, "unrecorded", None, []),
    # (A) landed (origin/main registry-decode-history, the registry owner 2026-10-13 22:34: the canonical BoN stays T=1; the nine grandfathered_T1 rows
    # live and non-default, superseded_by_policy_id null): a T=1 draw after 2026-09-17 reads EXACT under the grandfathered row; the vendor row (0.7) stays
    # in scope and is not a candidate
    ("bon_T1_generated_after_0917", "strongreject_bon", "off", "per_draw", "grandfathered_T1", T1, GEN, "exact", P["bon_T1"], [P["bon_T1"]]),
    ("bon_T1_generated_before_0917", "strongreject_bon", "off", "per_draw", "grandfathered_T1", T1, "2026-09-15T00:00:00.000Z", "exact", P["bon_T1"], [P["bon_T1"]]),
    ("actorattack_seed_none", "actorattack", "off", "none", "model_card", SIX, GEN, "no_match", None, []),
    ("actorattack_seed_per_goal_turn", "actorattack", "off", "per_goal_turn", "model_card", SIX, GEN, "exact", P["mt_vendor"], [P["mt_vendor"]]),
    ("gsm8k_thinking_card", "gsm8k", "on", "per_item", "model_card", CARD5, GEN, "ambiguous", None, [P["cap_card"], P["gsm8k_card"]]),
    ("xstest_with_do_sample_leaked", "xstest", "off", "per_item", "model_card", dict(SIX, do_sample=True), GEN, "no_match", None, []),
    # a greedy over-refusal control now states its sent set ({temperature: 0.0}); the demo key has no greedy
    # over_refusal twin (decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09), so it reads no_match -- DERIVED from a recorded set, never unrecorded
    ("xstest_greedy_stated", "xstest", "off", "none", "greedy", {"temperature": 0.0}, GEN, "no_match", None, []),
]


def _conn(seeded_db):
    conn = sqlite3.connect(seeded_db)
    conn.row_factory = sqlite3.Row
    # a TEMP `run` shadows the real (empty) table for the SQL object's body: the seeded policy and eval tables are read as they are
    conn.execute("CREATE TEMP TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, model_id TEXT, eval_id TEXT, thinking_mode TEXT, decode_seed_rule TEXT, "
                 "decode_sent_json TEXT, decode_temperature_sent REAL, decode_top_p_sent REAL, decode_top_k_sent INTEGER, "
                 "generated_utc TEXT, started_utc TEXT, queued_utc TEXT, decode_source TEXT, extra_json TEXT, field_versions_json TEXT, decode_policy_id TEXT, decode_policy_match TEXT)")
    for rid, eval_id, thinking, seed_rule, source, sent, generated, *_ in CASES:
        conn.execute("INSERT INTO run VALUES (?, 'sdfbase_full22', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '{}', NULL, 'no_policy')",
                     (rid, MODEL, eval_id, thinking, seed_rule, json.dumps(sent), sent.get("temperature"), sent.get("top_p"), sent.get("top_k"),
                      generated, generated, generated, source))
    return conn


def test_the_demo_policy_rows_are_seeded_as_the_plan_read_them(seeded_db):
    conn = sqlite3.connect(seeded_db)
    try:
        rows = {r[0]: r for r in conn.execute("SELECT policy_id, temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, seed_rule, effective_from_utc, "
                                                "superseded_by_policy_id FROM decode_policy WHERE model_id = ?", (MODEL,)).fetchall()}
        assert len(rows) == 28, sorted(rows)
        # (A): the grandfathered_T1 row names NO successor -- decode history kept as data beside the vendor_preset default (configs/registry/README.md)
        assert rows[P["bon_T1"]][1:7] == (1.0, None, None, None, None, None) and rows[P["bon_T1"]][9] is None
        assert rows[P["bon_vendor"]][1:7] == (0.7, 0.8, 20, 0.0, 1.0, 1.5) and rows[P["bon_vendor"]][8] == "2026-09-17T00:00:00.000Z"
        assert rows[P["mt_vendor"]][7] == "per_goal_turn" and rows[P["over_refusal"]][7] == "per_item" and rows[P["mmlu"]][1] == 0.0
        cats = dict(conn.execute("SELECT eval_id, eval_category FROM eval WHERE eval_id IN ('xstest', 'or_bench_hard', 'codeattack', 'grok420', 'agentic_murder', "
                                 "'mmlu', 'strongreject_bon', 'actorattack', 'gsm8k')").fetchall())
        assert cats == {"xstest": "over_refusal", "or_bench_hard": "over_refusal", "codeattack": "static_attack", "grok420": "static_attack", "agentic_murder": "agentic",
                        "mmlu": "capability", "strongreject_bon": "best_of_n", "actorattack": "multi_turn", "gsm8k": "capability"}
    finally:
        conn.close()


def test_the_verdict_table_through_the_real_matcher_and_the_tiebreak(seeded_db):
    conn = _conn(seeded_db)
    try:
        src = f"({DM.match_select_sql()})"
        got = {r["run_id"]: dict(r) for r in conn.execute(f"SELECT run_id, policy_id, match, candidates_json, in_scope_json FROM {src}").fetchall()}
        bad = []
        for rid, eval_id, thinking, seed_rule, source, sent, generated, match, policy_id, candidates in CASES:
            g = got[rid]
            if (g["match"], g["policy_id"], json.loads(g["candidates_json"])) != (match, policy_id, candidates):
                bad.append((rid, g["match"], g["policy_id"], g["candidates_json"], DM.no_match_reasons(conn, rid)))
        assert not bad, bad
        # the BoN run generated after 2026-09-17: both rows IN SCOPE; under (A) the grandfathered T1 row is the one candidate (it is live and names no
        # successor), the vendor row is refused on 0.7 != 1.0 alone
        bon = got["bon_T1_generated_after_0917"]
        assert set(json.loads(bon["in_scope_json"])) >= {P["bon_T1"], P["bon_vendor"]} and json.loads(bon["candidates_json"]) == [P["bon_T1"]]
        reasons = DM.no_match_reasons(conn, "bon_T1_generated_after_0917")
        assert any(P["bon_vendor"] in r and "0.7" in r for r in reasons) and any(P["bon_T1"] in r and "matches" in r for r in reasons), reasons
        # the multi-turn run with seed rule `none`: the vendor row is in scope and refused on the seed rule alone ((B), the instrument correction)
        mt = DM.no_match_reasons(conn, "actorattack_seed_none")
        assert any(P["mt_vendor"] in r and "seed" in r for r in mt), mt
        # the gsm8k tie: the stamp pass's eval-over-category rule picks the gsm8k row and lists the capability row as the loser
        tb = DM.tiebreak(conn, src)
        assert set(tb) == {"gsm8k_thinking_card"} and tb["gsm8k_thinking_card"]["policy_id"] == P["gsm8k_card"] and tb["gsm8k_thinking_card"]["over"] == [P["cap_card"]]
        # the leaked unmodelled key: refused by the `run_sent_unmodelled` CTE, not by any parameter
        assert "do_sample" in json.loads(conn.execute("SELECT decode_sent_json FROM run WHERE run_id = 'xstest_with_do_sample_leaked'").fetchone()[0])
    finally:
        conn.close()
