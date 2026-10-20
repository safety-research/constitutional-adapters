"""The ruling of 2026-09-21 (`decode-vendor-default-all-evals-2026-09-21`): the vendor preset is the default decode for EVERY eval

1. NO existing run reads differently. Over the REAL seeds (the loader's own path, ingested), a T = 1.0 probe on every served model's agentic
   key -- temperature 1.0 alone, seed rule `none`, exactly what every existing agentic run sent -- still reads `exact` -> its `runner_forced_T1`
   row, whether generated before or after the ruling (the match never reads the default flag; an unsuperseded row is a candidate for ever); a
   vendor-shaped probe reads `exact` -> the `|vendor` row; nothing is ambiguous. The family-default expression then says which of the two IS the
   default: 0 for the T1 probe (a recorded, NON-default policy), 1 for the vendor probe.
2. The paper views render it. On a synthetic corpus: `decode_family_default_policy_id` is what `resolve()` returns for the run's key (an eval
   row over a category row, C18; a superseded row never), `decode_is_family_default` is 1 / 0 / NULL for a run matching the default / another
   recorded row / nothing, in `v_paper_current` and `v_paper_long` alike.
"""
from __future__ import annotations

import json
import os
import sqlite3

import test_registry_decode_llama_thinking_na as LT
from test_registry_decode_llama_thinking_na import idx, seeded_db, seeds  # noqa: F401  (fixtures: the real seeds, ingested)

from sisteer.registry import db as DB
from sisteer.registry import decode_match as DM
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth

RULING = "decode-vendor-default-all-evals-2026-09-21"
#: (model, thinking, a representative eval of the category) -> the key's T1 row (None: no T1 row was ever seeded) and its vendor default
AGENTIC_KEYS = (
    ("gemma4e4b_it", "off", "agentic_leaking"), ("glm46vflash", "off", "agentic_murder"), ("llama8b_it", "off", "agentic_leaking"),
    ("llama8b_it", "not_available", "agentic_leaking"), ("nemotron3nano4b", "off", "agentic_murder"), ("qwen35_4b_it", "off", "agentic_leaking"),
    ("qwen35_9b_it", "off", "agentic_leaking"),
)
EM_KEYS = (("llama8b_it", "off", "em_bad_medical"), ("llama8b_it", "not_available", "em_bad_medical"))
BEFORE, AFTER = "2026-09-15T00:00:00.000Z", "2026-09-22T00:00:00.000Z"   # generated before / after the ruling's effective date


def _probe_conn(seeded_db):
    """The seeded policy table read as is; a TEMP `run` shadows the real (empty) table for the SQL object's body and the family-default expression."""
    conn = sqlite3.connect(seeded_db)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TEMP TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, model_id TEXT, eval_id TEXT, thinking_mode TEXT, decode_seed_rule TEXT, "
                 "decode_sent_json TEXT, decode_temperature_sent REAL, decode_top_p_sent REAL, decode_top_k_sent INTEGER, "
                 "generated_utc TEXT, started_utc TEXT, queued_utc TEXT, decode_policy_id TEXT, decode_source TEXT)")   # decode_source: read by the by-source rule (0021)
    return conn


def _insert(conn, rid, model, eval_id, thinking, seed_rule, sent, generated):
    conn.execute("INSERT INTO run VALUES (?, 'probe', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'generation_config')",
                 (rid, model, eval_id, thinking, seed_rule, json.dumps(sent), sent.get("temperature"), sent.get("top_p"), sent.get("top_k"),
                  generated, generated, generated))


def _match_all(conn):
    got = {r["run_id"]: dict(r) for r in conn.execute(f"SELECT run_id, policy_id, match, candidates_json, in_scope_json FROM ({DM.match_select_sql()})")}
    for rid, e in DM.tiebreak(conn, f"({DM.match_select_sql()})").items():
        assert not e["policy_id"], (rid, e)          # nothing here needs the tie-break
    return got


def _family(conn):
    """(decode_family_default_policy_id, decode_is_family_default) per probe, with `run.decode_policy_id` stamped from the match first."""
    return {r["run_id"]: (r["decode_family_default_policy_id"], r["decode_is_family_default"])
            for r in conn.execute(f"SELECT r.run_id, {S.decode_family_default_sql('r')} FROM run r")}


def test_every_existing_agentic_run_shape_still_matches_its_t1_row_and_the_vendor_row_is_the_family_default(idx, seeded_db):
    conn = _probe_conn(seeded_db)
    try:
        want = {}
        for model, thinking, eval_id in AGENTIC_KEYS + EM_KEYS:
            cat = "em_organism" if eval_id.startswith("em_") else "agentic"
            t1 = f"{model}|{cat}|{thinking}|runner_forced_T1@v1"
            vendor = f"{model}|{cat}|{thinking}|vendor@v1"
            assert vendor in idx.rows, vendor
            vp, ve = idx.rows[vendor]
            assert vp["is_default"] == 1 and vp.get("superseded_by_policy_id") is None
            if t1 in idx.rows:
                tp, te = idx.rows[t1]
                assert tp["is_default"] == 0 and tp.get("superseded_by_policy_id") is None, t1
            else:
                assert model == "qwen35_9b_it", t1            # the one key that never had a T1 row (no 9B agentic run exists)
                t1 = None
            vsent = {k: vp[k] for k in DM.PARAMS if vp.get(k) is not None}
            for when in (BEFORE, AFTER):
                rid = f"t1:{vendor}:{when[:10]}"
                _insert(conn, rid, model, eval_id, thinking, "none", {"temperature": 1.0}, when)
                want[rid] = (t1, vendor, "no_match" if t1 is None else "exact", None if t1 is None else 0)
                rid = f"vendor:{vendor}:{when[:10]}"
                _insert(conn, rid, model, eval_id, thinking, vp["seed_rule"], vsent, when)
                want[rid] = (vendor, vendor, "exact", 1)
        got = _match_all(conn)
        bad = []
        for rid, (policy, family, match, is_default) in want.items():
            g = got[rid]
            if (g["match"], g["policy_id"]) != (match, policy):
                bad.append((rid, g["match"], g["policy_id"], g["candidates_json"], DM.no_match_reasons(conn, rid)))
                continue
            # the two rows of the key are both IN SCOPE (the arm's rows only); exactly one is a candidate
            scope = json.loads(g["in_scope_json"])
            assert family in scope and (policy is None or policy in scope), (rid, scope)
            assert json.loads(g["candidates_json"]) == ([] if policy is None else [policy]), (rid, g["candidates_json"])
            conn.execute("UPDATE run SET decode_policy_id = ? WHERE run_id = ?", (policy, rid))
        assert not bad, bad
        fam = _family(conn)
        for rid, (policy, family, match, is_default) in want.items():
            assert fam[rid] == (family, is_default), (rid, fam[rid], (family, is_default))
    finally:
        conn.close()


def test_the_seed_rows_of_the_ruling_are_shaped_as_the_ruling_says(idx):
    """Each new vendor row restates its model's static vendor preset (`same_values_as`, loader-checked) under the runner's `vendor` label with a
    per-response seed rule; each T1 row keeps its text and loses only its default flag; the ruling text itself is on every bound row."""
    for model, thinking, eval_id in AGENTIC_KEYS + EM_KEYS:
        cat = "em_organism" if eval_id.startswith("em_") else "agentic"
        p, e = idx.rows[f"{model}|{cat}|{thinking}|vendor@v1"]
        assert e["mode"] == "vendor" and p["is_default"] == 1 and p["effective_from_utc"] in ("2026-09-11T00:00:00.000Z", "2026-09-21T00:00:00.000Z")
        if thinking == "not_available":
            assert e["decode_ruling_ref"] == LT.RULING and e["same_values_as"] == f"{model}|{cat}|off|vendor@v1"
        else:
            assert e["decode_ruling_ref"] == RULING and RULING in str(p["source_citation"])
            if model != "qwen35_9b_it":
                assert e.get("same_values_as", "").startswith(f"{model}|static|off|vendor_preset"), (model, e.get("same_values_as"))
        if model == "qwen35_9b_it":
            continue
        assert p["seed_rule"] == "per_response" and "seed" in e["params_sent"] and "max_tokens" in e["params_sent"], (model, thinking)
    # the 9B row keeps the seed rule it was seeded with (no 9B agentic run exists to say otherwise) -- only its default flag and binding moved
    p9, e9 = idx.rows["qwen35_9b_it|agentic|off|vendor@v1"]
    assert p9["seed_rule"] == "none" and p9["is_default"] == 1 and e9["decode_ruling_ref"] == RULING


# --------------------------------------------------------------------------------------------
# 2. the paper views: decode_family_default_policy_id / decode_is_family_default (synthetic corpus)
# --------------------------------------------------------------------------------------------

VENDOR = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=1.5)
T1_NOT_SENT = {"top_p": "0.8 (server default)", "top_k": "20", "min_p": "0.0", "repetition_penalty": "1.0", "presence_penalty": "0.0"}


def _policy(policy_id, ref, *, eval_id=None, eval_category="static_attack", is_default=1, seed_rule="per_item", superseded_by=None,
            not_sent=None, effective=synth.t(1), **params):
    p = {"policy_id": policy_id, "policy_ref": ref, "policy_version": 1, "model_id": synth.MODEL_ID, "thinking_mode": "off",
         "decode_source": "generation_config", "max_new_tokens_rule": "cap 512", "seed_rule": seed_rule, "is_default": is_default,
         "params_not_sent_json": not_sent or {}, "source_citation": "synthetic", "effective_from_utc": effective}
    if eval_id:
        p["eval_id"] = eval_id
    else:
        p["eval_category"] = eval_category
    if superseded_by:
        p["superseded_by_policy_id"] = superseded_by
    p.update(params)
    return p


def _run(key, sent, seed_rule, i0):
    rid, recs = synth.run_records(key, i0, through="judged")
    p = recs[0]["payload"]
    p["decode_sent_json"] = dict(sent)
    p["decode_temperature_sent"] = sent.get("temperature")
    p["decode_top_p_sent"] = sent.get("top_p")
    p["decode_top_k_sent"] = sent.get("top_k")
    p["decode_seed_rule"] = seed_rule
    p["decode_source"] = "generation_config"
    return rid, recs


def _corpus(root, policies, runs):
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    R.write_records(os.path.join(root, "records", "fd", "000001-policies.jsonl"), [synth.rec("decode_policy", p, synth.t(2)) for p in policies])
    recs = []
    for _, rr in runs:
        recs += rr
    R.write_records(os.path.join(root, "records", "fd", "000002-runs.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals


COLS = "run_id, decode_source, decode_policy_id, decode_policy_match, decode_family_default_policy_id, decode_is_family_default"


def _views(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        pc = {r["run_id"]: dict(r) for r in conn.execute(f"SELECT {COLS} FROM v_paper_current")}
        pl = {r["run_id"]: dict(r) for r in conn.execute(f"SELECT {COLS} FROM v_paper_long")}
    finally:
        conn.close()
    return pc, pl


def test_paper_views_name_the_family_default_and_whether_the_matched_row_is_it(tmp_path):
    """An eval-scoped default beats the category default (C18); a superseded row is never the default; 1 / 0 / NULL for a run matching the
    default / another recorded row / nothing -- the same in v_paper_current and v_paper_long."""
    root = str(tmp_path / "reg")
    policies = [
        _policy("p_cat_vendor", "synthmodel|static|off|vendor", **VENDOR),
        _policy("p_cat_t1", "synthmodel|static|off|runner_forced_T1", is_default=0, seed_rule="none", not_sent=T1_NOT_SENT, temperature=1.0),
        # a superseded row that still carries the flag (a stale default): never the family default
        _policy("p_cat_old", "synthmodel|static|off|old", superseded_by="p_cat_vendor", temperature=0.9),
        _policy("p_eval_default", "synthmodel|eval|off|vendor", eval_id=synth.EVAL_ID, temperature=0.3),
    ]
    runs = [_run("t1", {"temperature": 1.0}, "none", 100), _run("eval", {"temperature": 0.3}, "per_item", 200),
            _run("vendor", VENDOR, "per_item", 300), _run("nomatch", {"temperature": 0.55}, "per_item", 400)]
    _corpus(root, policies, runs)
    pc, pl = _views(root)
    ids = {k: rid for k, (rid, _) in zip(("t1", "eval", "vendor", "nomatch"), runs)}
    assert set(pc) >= set(ids.values()) and set(pl) >= set(ids.values())
    for views in (pc, pl):
        t1, ev, ve, no = (views[ids[k]] for k in ("t1", "eval", "vendor", "nomatch"))
        # the family default of the key is the EVAL-scoped default row, for every run of the eval
        assert {r["decode_family_default_policy_id"] for r in (t1, ev, ve, no)} == {"p_eval_default"}
        # exact -> a recorded NON-default row: 0 (the T = 1.0 run; and the category default, which the eval row outranks)
        assert (t1["decode_policy_match"], t1["decode_policy_id"], t1["decode_is_family_default"]) == ("exact", "p_cat_t1", 0)
        assert (ve["decode_policy_match"], ve["decode_policy_id"], ve["decode_is_family_default"]) == ("exact", "p_cat_vendor", 0)
        # exact -> THE default: 1
        assert (ev["decode_policy_match"], ev["decode_policy_id"], ev["decode_is_family_default"]) == ("exact", "p_eval_default", 1)
        # nothing matched: NULL (nothing to compare), the family default still named
        assert (no["decode_policy_match"], no["decode_policy_id"], no["decode_is_family_default"]) == ("no_match", None, None)
        assert all(r["decode_source"] == "generation_config" for r in (t1, ev, ve, no))
    # the two views agree row for row
    assert {k: pc[k] for k in ids.values()} == {k: pl[k] for k in ids.values()}


def test_without_an_eval_row_the_category_default_is_the_family_default_and_the_t1_run_reads_non_default(tmp_path):
    """The agentic shape exactly: a category-scoped vendor default beside a non-default T1 row; the T = 1.0 run matches the T1 row (exact) and
    reads decode_is_family_default 0, the vendor run 1."""
    root = str(tmp_path / "reg")
    policies = [_policy("p_cat_vendor", "synthmodel|static|off|vendor", **VENDOR),
                _policy("p_cat_t1", "synthmodel|static|off|runner_forced_T1", is_default=0, seed_rule="none", not_sent=T1_NOT_SENT, temperature=1.0)]
    runs = [_run("t1", {"temperature": 1.0}, "none", 100), _run("vendor", VENDOR, "per_item", 200)]
    _corpus(root, policies, runs)
    pc, pl = _views(root)
    for views in (pc, pl):
        t1, ve = views[runs[0][0]], views[runs[1][0]]
        assert (t1["decode_policy_match"], t1["decode_policy_id"], t1["decode_family_default_policy_id"], t1["decode_is_family_default"]) == ("exact", "p_cat_t1", "p_cat_vendor", 0)
        assert (ve["decode_policy_match"], ve["decode_policy_id"], ve["decode_family_default_policy_id"], ve["decode_is_family_default"]) == ("exact", "p_cat_vendor", "p_cat_vendor", 1)
    # a key with NO default at all: the matched row is not the family default (0), and no default is named
    root2 = str(tmp_path / "reg2")
    _corpus(root2, [_policy("p_cat_t1", "synthmodel|static|off|runner_forced_T1", is_default=0, seed_rule="none", not_sent=T1_NOT_SENT, temperature=1.0)],
            [_run("t1", {"temperature": 1.0}, "none", 100)])
    pc2, pl2 = _views(root2)
    for views in (pc2, pl2):
        r = views[runs[0][0]]
        assert (r["decode_policy_match"], r["decode_policy_id"], r["decode_family_default_policy_id"], r["decode_is_family_default"]) == ("exact", "p_cat_t1", None, 0)
