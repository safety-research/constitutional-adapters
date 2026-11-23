"""A synthetic registry for the generator-comparison tests (test_appendix_generator.py, test_appendix_generator_sidecar.py).

`registry(committed)` builds an in-memory SQLite holding the tables `appendix_generator_sidecar.derive` reads, populated so that a
derivation gives back the committed file's static family exactly: the six objects, the 40 static runs (28 judged final rungs, 12
unjudged lower rungs), the judgments and item rows behind each payload, the static comparison filing and its inputs, the three
rulings and the four judges. The `add_*` helpers then file runs of the later families at any lifecycle, with synthetic item rows of
their eval's family on synthetic item ids (`IDS`), so a test can drive the derivation through a wave as it is filed. A synthetic
registry holds no filing of the grid's comparison (the pinned one names the live runs), so a derivation from it runs `unpinned()`.
Synthetic throughout: invented item ids, run ids and judgment ids; no recorded data beyond the committed file itself.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3

import numpy as np

import appendix_generator_sidecar as G
import registry_nb as R

SCHEMA_SQL = """
CREATE TABLE record_ingest (ingest_seq INTEGER);
CREATE TABLE analysis (analysis_id TEXT, name TEXT, report_sha256 TEXT, report_uri TEXT);
CREATE TABLE analysis_input (analysis_id TEXT, input_kind TEXT, run_id TEXT, object_id TEXT);
CREATE TABLE object (object_id TEXT, campaign TEXT, fit_corpus_id TEXT, fit_seed INTEGER, parent_object_id TEXT, ingest_seq INTEGER);
CREATE TABLE run (run_id TEXT, campaign TEXT, object_id TEXT, eval_id TEXT, model_id TEXT, instrument_id TEXT, item_set_id TEXT,
                  lifecycle_state TEXT, token_cap_tokens REAL, token_cap_rung_index INTEGER, pass_label TEXT, transfer_kind TEXT,
                  thinking_mode TEXT, serving_path TEXT, decode_policy_id TEXT, decode_source TEXT, view_legacy TEXT,
                  lane_source_legacy TEXT, trial_index INTEGER, ingest_seq INTEGER, store_uri TEXT, store_completion_artifact_uri TEXT);
CREATE TABLE instrument (instrument_id TEXT, bon_n REAL, padding_tokens REAL);
CREATE TABLE v_paper_current (run_id TEXT, held_out INTEGER);
CREATE TABLE judge (judge_id TEXT, judge_model TEXT, rubric_name TEXT, rubric_version TEXT);
CREATE TABLE judgment (judgment_id TEXT, run_id TEXT, judge_id TEXT, status TEXT, superseded_by_judgment_id TEXT, finished_utc TEXT,
                       extra_json TEXT, ingest_seq INTEGER);
CREATE TABLE item_result (run_id TEXT, item_id TEXT, metric_id TEXT, trial_index INTEGER, sample_index INTEGER, turn_index INTEGER,
                          judgment_id TEXT, value REAL, label TEXT, extra_json TEXT, written_utc TEXT, record_sha256 BLOB,
                          ingest_seq INTEGER, withdrawn_by_revision_id TEXT, withdrawn_by_record_sha256 TEXT);
CREATE VIEW v_item_result_current AS SELECT * FROM item_result WHERE withdrawn_by_revision_id IS NULL AND withdrawn_by_record_sha256 IS NULL;
CREATE TABLE item_flags (run_id TEXT, item_id TEXT, trial_index INTEGER, at_token_cap INTEGER);
CREATE TABLE run_metric (run_id TEXT, metric_id TEXT, value REAL, k REAL, n REAL);
CREATE TABLE run_contrast (run_id TEXT, anchor_run_id TEXT);
CREATE TABLE ruling (ruling_id TEXT, action TEXT, scope_kind TEXT, scope_json TEXT, expired_utc TEXT);
CREATE TABLE run_state_event (run_id TEXT, state TEXT, evidence_kind TEXT, evidence_uri TEXT, note TEXT, event_utc TEXT);
"""
CORPUS = {"opus": "constv2-mix-1m", "qwen": "constqwen38-mix-1m"}
OBJECT_OF = {(a, s): oid for oid, (a, s) in G.OBJECTS.items()}
#: synthetic item ids of the later families' item sets (the static payloads keep the committed goal ids)
IDS = {"multiturn_goals_259": [f"goal_{i:03d}" for i in range(259)], "jbb_goals_100": [f"jbb_{i:03d}" for i in range(100)],
       "xstest_safe_250": [f"xst{i:04d}" for i in range(450)], "or_bench_hard_1319": [f"orb{i:04d}" for i in range(1319)],
       "mmlu_1531": [f"m{i:04d}" for i in range(1531)], "gsm8k_platinum_paired_200": [f"g{i:03d}" for i in range(200)]}
CPI_LABEL = {2: "full_compliance", 1: "partial_refusal", 0: "full_refusal"}
_SEQ = [1000]


def _seq() -> int:
    _SEQ[0] += 1
    return _SEQ[0]


def _sha(*parts) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()


@contextlib.contextmanager
def unpinned():
    """Derive as before the grid's comparison was pinned: the pinned filing (`G.COMPARISON_V2`) is the live registry's, its inputs the
    committed file's read runs, and a synthetic registry holds no such filing (its later runs are invented), so a derivation from one
    runs with the pin lifted and the file it gives carries no cross-check of it. The cross-check itself is pinned on a synthetic filing
    (test_the_grids_comparison_is_cross_checked_once_pinned)."""
    pin = G.COMPARISON_V2
    G.COMPARISON_V2 = None
    try:
        yield
    finally:
        G.COMPARISON_V2 = pin


def run_id(arm, seed, e, salt="") -> str:
    """A synthetic run id in the registry's spelling: campaign, eval, instrument head, a hash."""
    return f"{G.CAMPAIGN}:{e}:{R.CTLSTOP_EVALS[e]['instrument'][:12]}:{_sha(arm, seed, e, salt)[:16]}"


def registry(committed: dict) -> sqlite3.Connection:
    """An in-memory registry from which `derive` gives back the committed file's static family (and every later reading pending)."""
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA_SQL)
    for (model, rubric, version), jid in G.LANE_JUDGES.items():
        c.execute("INSERT INTO judge VALUES (?,?,?,?)", (jid, model, rubric, version))
    for spec in R.CTLSTOP_EVALS.values():
        c.execute("INSERT INTO instrument VALUES (?,?,?)", (spec["instrument"], spec.get("bon_n"), 0.0))
    for oid, (arm, seed) in G.OBJECTS.items():
        c.execute("INSERT INTO object VALUES (?,?,?,?,?,?)", (oid, G.CAMPAIGN, CORPUS[arm], seed, None, _seq()))
    c.execute("INSERT INTO analysis VALUES (?,?,?,?)", (G.ANALYSIS_ID, G.ANALYSIS_NAME, G.REPORT_SHA256, None))
    for oid in G.OBJECTS:
        c.execute("INSERT INTO analysis_input VALUES (?,?,?,?)", (G.ANALYSIS_ID, "object", None, oid))
    for rid in G.RUNS:
        c.execute("INSERT INTO analysis_input VALUES (?,?,?,?)", (G.ANALYSIS_ID, "run", rid, None))
    for r in committed["rows"]:
        if r["eval_id"] not in G.STATIC:
            continue
        _insert_run(c, r["run_id"], r["object_id"], r["eval_id"], r["lifecycle_state"], r["run_token_cap_tokens"],
                    rung_index=0 if r["run_token_cap_tokens"] == 4096 else 1, pass_label=r["pass_label"],
                    transfer_kind=r["transfer_kind"], view=r["view_legacy"], decode=(r["decode_policy_id"], r["decode_source"]))
        if r["judgment_id"]:
            _insert_judgment(c, r["judgment_id"], r["run_id"], r["judge_id"])
            entry = committed["items"][r["run_id"]]
            sr_items(c, r["run_id"], r["judgment_id"], G.item_ids(committed, entry["item_set"]), entry["best"][0], entry["n_units"][0])
    unsteered = sorted(r["run_id"] for r in committed["rows"] if r["object_id"] == G.UNSTEERED)
    for rid_, (action, kind, covers) in G.RULINGS.items():
        scope = {"match": {"object_ids": sorted(G.OBJECTS)} if kind == "object" else {"run_ids": unsteered}}
        c.execute("INSERT INTO ruling VALUES (?,?,?,?,?)", (rid_, action, kind, json.dumps(scope), None))
    c.execute("INSERT INTO record_ingest VALUES (?)", (_seq(),))
    c.commit()
    return c


def _insert_run(c, rid, oid, e, state, cap, rung_index=0, pass_label="base_cap", transfer_kind=None, view=None, instrument=None,
                thinking=None, store=None, decode=(None, "model_card")):
    steered = oid != G.UNSTEERED
    seed = G.OBJECTS.get(oid, (None, 1))[1] if steered else 1
    spec = R.CTLSTOP_EVALS[e]
    store = store or f"store-{_sha(rid)[:12]}"
    c.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (rid, G.CAMPAIGN, oid, e, R.MODEL_ID, instrument or spec["instrument"], spec["item_set"], state, cap, rung_index, pass_label,
               transfer_kind or ("base_to_instruct" if steered else "none"), thinking or ("on" if e == "gsm8k" else "off"),
               "lora_native", decode[0], decode[1], view or G.VIEW.format(seed=seed), None, 0, _seq(), store, f"{store}/COMPLETE"))
    c.execute("INSERT INTO v_paper_current VALUES (?,?)", (rid, 0))


def _insert_judgment(c, jid, rid, judge_id, status="complete", extra=None, superseded=None):
    c.execute("INSERT INTO judgment VALUES (?,?,?,?,?,?,?,?)",
              (jid, rid, judge_id, status, superseded, f"2026-11-15T0{_SEQ[0] % 10}:00:00Z", json.dumps(extra or {}), _seq()))


def _item(c, rid, jid, item, metric, value=None, label=None, sample=0, extra=None):
    c.execute("INSERT INTO item_result VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (rid, item, metric, 0, sample, 0, jid, value, label, json.dumps(extra) if extra else None, "2026-11-15T00:00:00Z",
               bytes.fromhex(_sha(rid, item, metric, sample)[:32]), _seq(), None, None))


def sr_items(c, rid, jid, ids, best, n_units):
    """StrongREJECT item rows reproducing a payload: per goal, its units (the best one, the rest 0) and the judgment's per-goal row."""
    for g, b, u in zip(ids, best, n_units):
        for k in range(u):
            _item(c, rid, jid, g, "strongreject", value=(b if k == 0 else 0.0), sample=k)
        _item(c, rid, jid, g, "strongreject_best", value=b)


# ---- the later families, filed at any lifecycle ----------------------------------------------------------------------------------
def add_run(c, arm, seed, e, state="generated", cap=None, salt="", rung_index=0, pass_label="base_cap", instrument=None, oid=None):
    """File one run of (arm, seed, eval) at `state`; returns its id. `cap` defaults to the instrument's run cap (an over-refusal run
    holds its whole ladder at 8,192; GSM8K's base pass 8,192)."""
    spec = R.CTLSTOP_EVALS[e]
    if cap is None:
        cap = {"judged": 8192.0, "ladder": 8192.0, "run": 4096.0}.get(spec["drawn_cap"], spec["drawn_cap"])
    oid = oid or (OBJECT_OF[(arm, seed)] if arm != G.UNSTEERED else G.UNSTEERED)
    rid = run_id(arm, seed, e, salt)
    _insert_run(c, rid, oid, e, state, cap, rung_index=rung_index, pass_label=pass_label, instrument=instrument)
    return rid


def judge(c, rid, e, judge_id=None, status="complete", judged_cap=None, superseded=None, rng=None, labels=None):
    """File a judgment of `rid` by `judge_id` (default the eval's policy judge) with synthetic item rows of the eval's family; returns
    the judgment id. Over-refusal judgments record `judged_cap` (default the eval's read rung) and the store they judged."""
    spec = R.CTLSTOP_EVALS[e]
    pol = R.ctlstop_judging.policy(e)
    judge_id = judge_id or pol.judge_id
    rng = rng if rng is not None else np.random.default_rng(int(_sha(rid)[:8], 16))
    jid = f"jg-{_sha(rid, judge_id, status)[:12]}"
    extra = {}
    if spec["family"] == "over_refusal":
        store = c.execute("SELECT store_uri FROM run WHERE run_id = ?", (rid,)).fetchone()[0]
        extra = dict(judged_cap=judged_cap or spec.get("judged_cap", 8192.0), judged_store_origin=store)
    _insert_judgment(c, jid, rid, judge_id, status=status, extra=extra, superseded=superseded)
    ids = IDS[spec["item_set"]]
    if spec["family"] == "strongreject":
        units = 75 if e == "strongreject_bon" else 3
        best = [float(rng.choice((0.0, 0.25, 0.5, 0.75, 1.0), p=(0.5, 0.15, 0.1, 0.1, 0.15))) for _ in ids]
        sr_items(c, rid, jid, ids, best, [units] * len(ids))
    else:
        metric = "xst_label" if e == "xstest" else "orb_label"
        labs = labels if labels is not None else [int(rng.choice([0, 1, 2], p=[0.1, 0.1, 0.8])) for _ in ids]
        for i, (item, lab) in enumerate(zip(ids, labs)):
            extra_i = {"subset": ("safe" if i < 250 else "contrast")} if e == "xstest" else None
            _item(c, rid, jid, item, metric, label=CPI_LABEL[lab], extra=extra_i)
    return jid


def score(c, rid, e, rng=None, p=0.75, unscored=0.0, redraw_of=None):
    """File a rule-scored run's item rows (`acc` 1 / 0, or `unscored` for an answer the cap cut) and its filed `acc`; `redraw_of` (a base
    run id) scores exactly the base's unscored items (GSM8K's ladder rung)."""
    spec = R.CTLSTOP_EVALS[e]
    rng = rng if rng is not None else np.random.default_rng(int(_sha(rid)[:8], 16))
    ids = IDS[spec["item_set"]]
    if redraw_of is not None:
        ids = [r[0] for r in c.execute("SELECT item_id FROM item_result WHERE run_id = ? AND metric_id = 'unscored'", (redraw_of,))]
    k = n = 0
    for item in ids:
        if rng.random() < unscored:
            _item(c, rid, "", item, "unscored")
        else:
            v = int(rng.random() < p)
            _item(c, rid, "", item, "acc", value=float(v))
            k, n = k + v, n + 1
    c.execute("INSERT INTO run_metric VALUES (?,?,?,?,?)", (rid, "acc", (k / n) if n else None, k, n))


def ladder_exit(c, rid, rung, n_at_cap, n):
    """The registry's record of where an over-refusal run's cap ladder exited: the `generated` state filed from its store's completion
    artifact, its note ending in the closing facts."""
    store, art = c.execute("SELECT store_uri, store_completion_artifact_uri FROM run WHERE run_id = ?", (rid,)).fetchone()
    facts = {"final_rung": rung, "sidecars": {"n_at_cap": n_at_cap, "n_total": n}, "rederived": {"n_at_cap": n_at_cap}}
    c.execute("INSERT INTO run_state_event VALUES (?,?,?,?,?,?)",
              (rid, "generated", "completion_artifact", art, f"closed from {store}: {json.dumps(facts)}", "2026-11-15T00:00:00Z"))


def file_reading(c, arm, seed, e, rng=None):
    """One complete reading of (arm, seed, eval), filed as the wave files it: the run, and its judgment or its rule scores (GSM8K with a
    ladder rung for seed 2's cells); returns the drawn run's id."""
    spec = R.CTLSTOP_EVALS[e]
    if e == "gsm8k":
        rid = add_run(c, arm, seed, e, state="ingested")
        score(c, rid, e, rng=rng, unscored=(0.05 if seed == 2 else 0.0))
        if seed == 2:
            rung = add_run(c, arm, seed, e, state="generated", cap=16384.0, salt="rung", rung_index=1, pass_label="ladder_rung")
            score(c, rung, e, rng=rng, redraw_of=rid)
        return rid
    rid = add_run(c, arm, seed, e, state="ingested" if spec["family"] == "rule" else "generated")
    if spec["family"] == "rule":
        score(c, rid, e, rng=rng)
    else:
        judge(c, rid, e, rng=rng)
    return rid


def file_wave(c, evals=None, rng=None):
    """Every later reading (the 48 of the eight non-static evals, or those of `evals`), complete."""
    for e in (evals or [x for x in G.EVALS if x not in G.STATIC]):
        for arm in ("opus", "qwen"):
            for seed in G.SEEDS:
                file_reading(c, arm, seed, e, rng=rng)
    c.commit()
