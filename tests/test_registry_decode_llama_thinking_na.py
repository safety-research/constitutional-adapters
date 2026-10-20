"""Llama-8B `thinking_mode not_available` twins (ruling `decode-llama-thinking-not-available-2026-09-18`): a Llama-8B run
filing `not_available` and a decode_policy row keyed `off` describe the same decode decision -- they are matched, and any
true duplicate / ambiguity is reported.

The ruling is DATA, not a matcher rule: every `llama8b_it` row keyed `off` (explicit or wildcard-expanded) has a sibling keyed
`not_available`, identical in every parameter, in `configs/registry/decode_policy_seed.yaml`. Three things are pinned here:

1. the pairing is complete and exact, enumerated from the seed (never a hard-coded list): each `off` row has exactly one twin, each twin
   exactly one `off` row, every compared column equal, supersession mirrored, `same_values_as` pointing at the `off` row, the ruling bound;
2. the twins create no ambiguity under the matcher's rule (`v_run_decode_policy_match`) -- with ONE named exception, which answers
   whether the twins lead to true duplicates or ambiguity: a synthetic pair proves one candidate per run for each recorded thinking
   state, and every REAL twin is exercised through the SQL object over the seeded policy table -- a run filing `not_available` with the
   twin's parameters matches the twin alone; the same run filing `off` matches the `off` row alone. The exception is the IFEval twin
   (:data:`KNOWN_AMBIGUOUS`): Llama's `capability` CATEGORY row was already keyed `not_available` (`llama8b_it|capability|not_available|card`,
   the same 0.6 / 0.9, `per_item`), so a Llama IFEval run filing `not_available` now has two in-force candidates, the eval row and the
   category row -- the matcher weighs them equally (unlike `resolve()`, where the eval row wins), the same structure behind the 50
   pre-existing MATH-500 `ambiguous` runs on every model. No live run hits it on 2026-09-18 (no Llama IFEval run is filed in either arm);
   it is pinned by name so the artifact sees it, and the fix belongs in the matcher (eval row beats category row), not in the data here;
3. no other model carries the ruling (Llama is the one served model with no thinking channel).
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import decode_match as DM
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import dimensions as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
PROD = synth.producer("tests.registry_decode_llama_thinking_na")

MODEL = "llama8b_it"
RULING = "decode-llama-thinking-not-available-2026-09-18"
#: payload columns a twin may differ on from its `off` row (identity / the arm / provenance prose); everything else must be equal
TWIN_MAY_DIFFER = {"policy_id", "policy_ref", "thinking_mode", "notes", "source_citation", "superseded_by_policy_id", "field_confidence_json"}
#: extra_json keys a twin must carry equal to its `off` row
TWIN_EXTRA_EQUAL = ("mode", "params_sent", "effective_from_confidence", "legacy_contracts_row", "deliberate_departure")
#: a representative eval per category-scoped row (the category's eval as the seeded `eval` table spells it)
EVAL_OF_CATEGORY = {"static_attack": "codeattack", "over_refusal": "xstest", "multi_turn": "pair", "petri": "petri_pooled",
                    "agentic": "agentic_leaking", "em_organism": "em_bad_medical", "best_of_n": "strongreject_bon", "capability": "gsm8k"}
#: the ONE ambiguity the twins expose: a Llama IFEval run filing `not_available` with the vendor numbers has two
#: in-force candidates, the IFEval twin and the pre-existing `capability` category row Llama already keyed `not_available` with the same numbers
#: and seed rule. The `off` arm has no such pair (the category row is not in scope for `off`). No live run hits it (2026-09-18: no Llama IFEval
#: run in either arm). The SQL object still reads it as a tie (pinned here); the STAMP PASS resolves it for the eval row (eval row beats
#: category row, as `resolve()` already does) -- `decode_match.tiebreak`, pinned in tests/test_registry_decode_match.py.
KNOWN_AMBIGUOUS = {
    "probe:llama8b_it|ifeval|not_available|vendor_preset@v1:not_available": ["llama8b_it|capability|not_available|card@v1",
                                                                             "llama8b_it|ifeval|not_available|vendor_preset@v1"],
}


@pytest.fixture(scope="module")
def seeds():
    return D.load_seeds(SEEDS)


@pytest.fixture(scope="module")
def idx(seeds):
    return D.build_records(seeds, PROD, now="2026-09-18T12:00:00.000Z").policies


def _sibling_id(pid: str) -> str:
    return pid.replace("|not_available|", "|off|")


def _twins(idx):
    return {pid: pe for pid, pe in idx.rows.items() if pe[1].get("decode_ruling_ref") == RULING}


def _llama_off(idx):
    return {pid: pe for pid, pe in idx.rows.items() if pe[0]["model_id"] == MODEL and pe[0]["thinking_mode"] == "off"}


# --------------------------------------------------------------------------------------------
# 1. the pairing, enumerated
# --------------------------------------------------------------------------------------------


def test_the_ruling_is_seeded_with_the_user_words(seeds):
    text = seeds.decode["rulings"][RULING]
    assert "are matched" in text and "duplicates or ambiguity" in text and "user_direct" in text and "2026-09-18" in text


def test_every_llama_off_row_has_exactly_one_not_available_twin_identical_in_every_parameter(idx):
    off = _llama_off(idx)
    twins = _twins(idx)
    assert off, "no llama8b_it `off` rows -- the seed changed shape"
    # both directions, by name
    assert sorted(twins) == sorted(pid.replace("|off|", "|not_available|") for pid in off), (
        sorted(set(twins) ^ {pid.replace("|off|", "|not_available|") for pid in off}))
    # 9 explicit llama8b_it rows + the 7 wildcard rows expanded onto llama8b_it + the two `vendor` defaults of the rule of
    # 2026-09-21 (decode-vendor-default-all-evals-2026-09-21: llama8b_it|agentic|off|vendor, llama8b_it|em_organism|off|vendor)
    assert len(twins) == 18
    for tid, (tp, te) in twins.items():
        oid = _sibling_id(tid)
        op, oe = off[oid]
        assert tp["thinking_mode"] == "not_available" and op["thinking_mode"] == "off"
        assert tp["policy_ref"] == op["policy_ref"].replace("|off|", "|not_available|") and tp["model_id"] == MODEL
        # every payload column equal except the arm, the identity and the provenance prose
        cols = (set(tp) | set(op)) - TWIN_MAY_DIFFER
        diff = {c: (op.get(c), tp.get(c)) for c in cols if op.get(c) != tp.get(c)}
        assert not diff, (tid, diff)
        for k in TWIN_EXTRA_EQUAL:
            assert te.get(k) == oe.get(k), (tid, k, oe.get(k), te.get(k))
        # the loader checked the six numbers against the `off` row; the twin says so
        assert te["same_values_as"] == oid, (tid, te.get("same_values_as"))
        assert te["decode_ruling_ref"] == RULING and tp["ruling_id"] is None   # the FK stays NULL like every seed ruling
        # supersession mirrored: the successor's twin where the successor has one, else the same target; never dropped, never added
        osup, tsup = op.get("superseded_by_policy_id"), tp.get("superseded_by_policy_id")
        assert (osup is None) == (tsup is None), (tid, osup, tsup)
        if osup is not None:
            expected = osup.replace("|off|", "|not_available|") if osup.replace("|off|", "|not_available|") in idx.rows else osup
            assert tsup == expected, (tid, osup, tsup)
            assert idx.rows[tsup][0]["effective_from_utc"] == idx.rows[osup][0]["effective_from_utc"]
        assert not te.get("superseded_by_unresolved")
        # the `off` row's own ruling is named in the twin's notes (the twin inherits the decision, it does not re-rule it)
        assert (oe.get("decode_ruling_ref") or "none") in str(tp.get("notes")), tid
        assert "no thinking channel" in str(tp.get("notes"))


def test_no_other_model_carries_the_ruling_and_the_existing_not_available_rows_are_untouched(idx):
    models = {pe[0]["model_id"] for pe in _twins(idx).values()}
    assert models == {MODEL}
    # the three `card` rows Llama already keyed not_available (capability, gsm8k, math500) are not twins and have no `off` sibling
    for ref in ("llama8b_it|capability|not_available|card", "llama8b_it|gsm8k|not_available|card", "llama8b_it|math500|not_available|card"):
        p, e = idx.rows[ref + "@v1"]
        assert e.get("decode_ruling_ref") != RULING and _sibling_id(ref + "@v1") not in idx.rows
    # at most one default per (model, key, thinking) still holds with the twins in (the loader would have refused otherwise); no key is
    # without a default since the 9B agentic vendor row became its key's default (the rule of 2026-09-21)
    assert idx.no_default_keys == []


# --------------------------------------------------------------------------------------------
# 2. no ambiguity under the matcher
# --------------------------------------------------------------------------------------------

LLAMA_VENDOR = dict(temperature=0.6, top_p=0.9)
NOT_SENT = {"top_k": "0 = disabled", "min_p": "0.0", "repetition_penalty": "1.0", "presence_penalty": "0.0", "frequency_penalty": "0.0"}


def _policy(policy_id, ref, thinking, **params):
    p = {"policy_id": policy_id, "policy_ref": ref, "policy_version": 1, "model_id": synth.MODEL_ID, "thinking_mode": thinking,
         "eval_category": "static_attack", "decode_source": "generation_config", "max_new_tokens_rule": "cap 4096", "seed_rule": "per_item",
         "params_not_sent_json": dict(NOT_SENT), "source_citation": "synthetic", "effective_from_utc": synth.t(1)}
    p.update(params)
    return p


def _run(key, sent, thinking, i0):
    rid, recs = synth.run_records(key, i0, through="generated")
    p = recs[0]["payload"]
    p["decode_sent_json"] = dict(sent)
    p["decode_temperature_sent"] = sent.get("temperature")
    p["decode_top_p_sent"] = sent.get("top_p")
    p["decode_top_k_sent"] = sent.get("top_k")
    p["decode_seed_rule"] = "per_item"
    p["decode_source"] = "generation_config"
    p["thinking_mode"] = thinking
    return rid, recs


def test_a_synthetic_off_row_and_its_not_available_twin_each_admit_exactly_one_recorded_thinking_state(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    off = _policy("p_off", "synthmodel|static|off|vendor_preset", "off", **LLAMA_VENDOR)
    twin = _policy("p_na", "synthmodel|static|not_available|vendor_preset", "not_available", **LLAMA_VENDOR)
    na_run = _run("na", LLAMA_VENDOR, "not_available", 100)
    off_run = _run("off", LLAMA_VENDOR, "off", 200)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    R.write_records(os.path.join(root, "records", "na", "000001-policies.jsonl"), [synth.rec("decode_policy", p, synth.t(2)) for p in (off, twin)])
    R.write_records(os.path.join(root, "records", "na", "000002-runs.jsonl"), na_run[1] + off_run[1])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        for rid, want in ((na_run[0], "p_na"), (off_run[0], "p_off")):
            row = conn.execute("SELECT decode_policy_id, decode_policy_match FROM run WHERE run_id = ?", (rid,)).fetchone()
            view = conn.execute(f"SELECT candidates_json, in_scope_json FROM {DM.MATCH_SQL_NAME} WHERE run_id = ?", (rid,)).fetchone()
            assert (row[0], row[1]) == (want, "exact"), (rid, tuple(row))
            assert json.loads(view[0]) == [want] and json.loads(view[1]) == [want], (rid, tuple(view))   # the other arm's row is never in scope
    finally:
        conn.close()


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory):
    """A fresh registry with the real seeds ingested (the loader's own path), for the SQL object to read the twins from."""
    root = str(tmp_path_factory.mktemp("reg") / "root")
    H.ensure_layout(root)
    _, paths = D.write_seed_records(root, SEEDS, producer=PROD, now="2026-09-18T12:00:00.000Z")
    assert paths
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return DB.db_path(root)


def _probe_runs(idx, conn) -> list:
    """One probe run per twin, in BOTH recorded thinking states: the twin's pinned parameters sent verbatim, its seed rule, a generation
    time inside the row's in-force window (before the successor of a superseded row took effect)."""
    cats = {r[0]: r[1] for r in conn.execute("SELECT eval_id, eval_category FROM eval").fetchall()}
    out = []
    for tid, (tp, te) in sorted(_twins(idx).items()):
        eval_id = tp.get("eval_id") or EVAL_OF_CATEGORY[tp["eval_category"]]
        assert cats.get(eval_id) == (tp.get("eval_category") or cats[eval_id]), (tid, eval_id)
        sent = {k: tp[k] for k in DM.PARAMS if tp.get(k) is not None}
        sup = tp.get("superseded_by_policy_id")
        generated = "2026-09-10T00:00:00.000Z" if sup and idx.rows[sup][0]["effective_from_utc"] != "unverified" else "2026-09-15T00:00:00.000Z"
        for thinking, want in (("not_available", tid), ("off", _sibling_id(tid))):
            out.append((f"probe:{tid}:{thinking}", eval_id, thinking, tp["seed_rule"], sent, generated, want))
    return out


def test_every_real_twin_admits_a_not_available_run_alone_and_its_off_row_admits_an_off_run_alone(idx, seeded_db):
    conn = sqlite3.connect(seeded_db)
    conn.row_factory = sqlite3.Row
    try:
        # a TEMP `run` shadows the real (empty) table for the SQL object's body: the seeded policy table is read as is
        conn.execute("CREATE TEMP TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, model_id TEXT, eval_id TEXT, thinking_mode TEXT, decode_seed_rule TEXT, "
                     "decode_sent_json TEXT, decode_temperature_sent REAL, decode_top_p_sent REAL, decode_top_k_sent INTEGER, "
                     "generated_utc TEXT, started_utc TEXT, queued_utc TEXT, decode_source TEXT)")   # decode_source: read by the by-source rule (0021)
        probes = _probe_runs(idx, conn)
        assert len(probes) == 36   # 18 twins x the two recorded thinking states
        for rid, eval_id, thinking, seed_rule, sent, generated, _ in probes:
            conn.execute("INSERT INTO run VALUES (?, 'probe', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'generation_config')",
                         (rid, MODEL, eval_id, thinking, seed_rule, json.dumps(sent), sent.get("temperature"), sent.get("top_p"), sent.get("top_k"),
                          generated, generated, generated))
        got = {r["run_id"]: dict(r) for r in conn.execute(f"SELECT run_id, policy_id, match, candidates_json, in_scope_json FROM ({DM.match_select_sql()})").fetchall()}
        bad = []
        ambiguous = {}
        for rid, eval_id, thinking, seed_rule, sent, generated, want in probes:
            g = got[rid]
            if g["match"] == "ambiguous":
                ambiguous[rid] = json.loads(g["candidates_json"])
            elif (g["match"], g["policy_id"], json.loads(g["candidates_json"])) != ("exact", want, [want]):
                bad.append((rid, g["match"], g["policy_id"], g["candidates_json"], DM.no_match_reasons(conn, rid)))
            # the other arm's rows are never in scope: a not_available probe sees only not_available rows, and vice versa
            arms = {conn.execute("SELECT thinking_mode FROM decode_policy WHERE policy_id = ?", (pid,)).fetchone()[0] for pid in json.loads(g["in_scope_json"])}
            assert arms <= {thinking}, (rid, arms)
        assert not bad, bad
        # exactly the one known ambiguity, by name, in both directions: a new one fails, and so does its disappearance (then remove the pin)
        assert ambiguous == KNOWN_AMBIGUOUS, ambiguous
        # ...and it is the pre-existing category row's doing, not the twin's: the same probe in the `off` arm is exact on the IFEval `off` row
        assert got["probe:llama8b_it|ifeval|not_available|vendor_preset@v1:off"]["policy_id"] == "llama8b_it|ifeval|off|vendor_preset@v1"
        card = "llama8b_it|capability|not_available|card@v1"
        assert card in {pid for pid in KNOWN_AMBIGUOUS[next(iter(KNOWN_AMBIGUOUS))]} and idx.rows[card][1].get("decode_ruling_ref") != RULING
    finally:
        conn.close()
