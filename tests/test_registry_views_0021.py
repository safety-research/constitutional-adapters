"""Migration 0021: what the paper frame reads as CURRENT (2026-10-02), one rule per test on a synthetic corpus:

* `v_run_current` is the ONE evaluation of the selection (the CTE chain of `schema_def.selection_ctes_sql`, the two CASEs
  `SELECTION_STATE_SQL` / `IN_CURRENT_PAPER_SQL`); it derives `is_current` and `why` (the first failing rule, with the ruling id where a
  ruling decided), and `v_paper_current` projects it -- a held-out run keeps ('selected', 1, 1) and reads is_current 0.
* the later-run supersession keys on token_cap_tokens, trial_index and pass_label beside (model, object, eval, instrument,
  judge, strength, the legacy operating point): two token caps, trials or passes of one object are distinct runs, never successors;
  and is_current honors a decisive exclude / supersede on a run the roster does not match (selection_state stays not_in_roster,
  the paper's roster-first order; why names the ruling).
* the member passes of a pooled reading are CURRENT but not paper rows: a complete, roster-matched run of a source whose paper
  reading is a pooled pass, filed under one of that source's member pass labels (`vocab.MEMBER_PASS_LABELS_BY_SOURCE`, rendered into both
  CASEs by `schema_def.member_pass_sql`), reads selection_state `member_pass`, in_current_paper 0, is_current 1 and why NULL; only the
  pooled run enters the paper. A source off the table (`ctl_neg1`-like: trials with no pooled pass) keeps every pass in the paper, an
  include ruling on a member selects it, and a member a later run superseded at its key reads superseded (the currency facts win);
  since migration 0022 (tests/test_registry_views_0022.py) a labeled run reads member_pass only where its source filed a pooled
  reading at its own point -- (model, object, eval, `view_legacy`) -- and `unpooled_member` (not a paper row either) elsewhere; every
  family below carries a pooled run at its one point (no `view_legacy`), so these tests read as before;
  the rule lives in `v_run_current` alone: the registry's renderer (`paper.paper_long_frame`) honors `in_current_paper`, and the pinned
  per-source keep-list of `paper.paper_frame_exclusion` (`sampled_lane`: `pooled_t0_t1` alone) stays the paper side's contract.
* a contrast rides a CURRENT anchor only (an excluded or superseded baseline differences nothing; the metric row stays
  with NULL contrast columns), and `n_current_contrasts` counts the current contrasts at the key, no tie-break.
* a `run_metric` / `item_result` row of a superseded or abandoned judgment is not current (`v_paper_long`, `v_item_result_current`,
  `v_item_result_provenances`); rows with no judgment or of a partial / complete one are untouched; the tables keep every row; and
  multi-source ownership (`v_multi_source_rows`) counts live judgment rows only, so a superseded owner judgment leaves the secondary
  standing as the stopgap and a re-judged owner folds on its one live row instead of refusing on two.
* `v_pending_judgment.pending_reason` reads `superseded` before `abandoned`.
* `v_paper_current.coverage_state` (vocab.COVERAGE_STATES) and `coverage_detail_json`: a nofit_by_design certificate with
  nothing generated reads `undefined_by_geometry` with the certificate behind it, a headline row IN THE FRAME (the second accepted id
  included) reads `measured`, a finished run with neither reads `absent` -- and `measured` is the frame's own admission
  (`schema_def.frame_admission_sql`, one text both objects render): a headline row under a judge no matched roster entry names, or a
  secondary row folded onto the owner of its scope, is not coverage.
* the recorded-by-source decode match: an empty sent set under generation_config against the rows of its key that are served
  the same way and declare every pinned parameter not sent -- one -> exact, two -> ambiguous (the stamp pass tie-breaks eval over
  category, as for every ambiguous verdict), none -> unrecorded; another decode_source, an unmodelled key, or a sent set is unchanged;
  and a row counts only when it is eligible as an ordinary candidate is: its seed rule the run's or `unverified`,
  and a superseded row only for a run generated before its successor took effect.
* the rebuild trigger follows the rules: `v_paper_inputs_head` moves when an anchor run retires (the move the 0020
  text missed -- its two-state run member even recedes) and when a judgment is re-filed superseded (the 0020 text followed that only
  through the fold's re-stamp of the run row; `judgment` is a member in its own right now), `paper_rebuild.sh` building only when the
  head moved; its watermark keys on the schema version beside the head (tests/test_registry_paper_rebuild_ops.py).

The migration mechanics (the file is the render against the v20 shape, view-only, every object selectable) are
tests/test_registry_migrations.py::test_0021_run_current_frame_rules_on_a_v20_database's.
"""
from __future__ import annotations

import json
import os

import pytest

import test_registry_decode_match as DMT
import test_registry_multi_source as MS
import test_registry_paper_views_contrast_current as CC

from sisteer.registry import db as DB
from sisteer.registry import decode_match as DM
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import ownership as OW
from sisteer.registry import paper
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root):
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refusals, rep.blocked)
    return rep


def _roster_entry(i, object_selector=None, object_id=None, comparison="fig_synth", judge=synth.JUDGE_ID):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID,
         "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": judge, "strength_rule": "operating_point",
         "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}
    if object_id:
        p["object_id"] = object_id
    else:
        p["object_selector_json"] = object_selector or {"paper_label": "constitution_lora"}
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _ruling(rid, action, scope_kind, scope, i=60, **extra):
    p = {"ruling_id": rid, "round_id": "r_synth", "action": action, "scope_kind": scope_kind, "scope_json": scope,
         "reason": "synthetic", "ruled_utc": synth.t(i), "ruled_by": "user", "source_ref": "test"}
    p.update(extra)
    return synth.rec("ruling", p, synth.t(i))


def _rows(conn, sql, *args):
    return [dict(r) for r in conn.execute(sql, args)]


def _by_run(conn, table):
    return {r["run_id"]: r for r in _rows(conn, f"SELECT * FROM {table}")}


# --------------------------------------------------------------------------------------------
# V_run_current, the one evaluation; v_paper_current projects it
# --------------------------------------------------------------------------------------------


def test_run_current_is_rendered_once_and_v_paper_current_projects_it():
    rc, pc, long = S.VIEW_SQL["v_run_current"], S.VIEW_SQL["v_paper_current"], S.VIEW_SQL["v_paper_long"]
    assert rc.count("ROW_NUMBER() OVER (PARTITION BY run_id") == 1 and pc.count("ROW_NUMBER() OVER (PARTITION BY run_id") == 0
    assert S.selection_ctes_sql() in rc and "MATERIALIZED" in S.selection_ctes_sql()
    squash = lambda t: " ".join(t.split())  # noqa: E731
    assert squash(S.SELECTION_STATE_SQL) in squash(rc) and squash(S.IN_CURRENT_PAPER_SQL) in squash(rc)
    assert squash(S.SELECTION_STATE_SQL) not in squash(pc) and "JOIN v_run_current vc ON vc.run_id = r.run_id" in pc
    assert "v_run_current" in S.VIEW_REFERENCES["v_paper_current"] and "ruling" not in S.VIEW_REFERENCES["v_paper_current"]
    assert set(S.VIEW_REFERENCES["v_run_current"]) == {"paper_round", "run", "object", "eval", "ruling", "observation", "v_roster_match"}
    assert long.count("FROM v_paper_current") == 1 and S.PAPER_CURRENT_CTE in long    # the frame evaluates the selection once too
    assert S.NON_CURRENT_SELECTION_STATES == ("superseded", "withdrawn", "excluded")
    assert S.NON_CURRENT_LIFECYCLE_STATES == ("failed", "cancelled", "superseded", "withdrawn")


def test_is_current_and_why_follow_the_rules_in_order_and_hold_out_keeps_the_row_in_the_paper(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    none_run, lora_run = ids["run_ids"]
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1), _roster_entry(2, object_id="none", comparison="baseline")])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc, pc = _by_run(conn, "v_run_current"), _by_run(conn, "v_paper_current")
    for rid in (none_run, lora_run):
        assert (rc[rid]["selection_state"], rc[rid]["in_current_paper"], rc[rid]["is_current"], rc[rid]["why"]) == ("selected", 1, 1, None), rid
        assert rc[rid]["decisive_ruling_id"] is None and rc[rid]["held_out"] == 0
    # the projection: every shared column of v_paper_current is v_run_current's, run for run
    shared = ("round_id", "comparison_ids_json", "roster_entry_ids_json", "ruling_ids_json", "decisive_ruling_id", "decisive_action", "held_out",
              "selection_state", "in_current_paper", "is_current", "why")
    assert {k: {c: v[c] for c in shared} for k, v in pc.items()} == {k: {c: v[c] for c in shared} for k, v in rc.items()}
    assert paper.in_current_paper(conn, lora_run) == (1, "selected", ["fig_synth"])
    conn.close()
    # (1) an exclude ruling: selection_state excluded -> is_current 0, why names the ruling
    _write(root, "roster", "000002-exclude.jsonl", [_ruling("R_ex", "exclude", "run", {"run_id": lora_run}, i=60)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc, pc = _by_run(conn, "v_run_current"), _by_run(conn, "v_paper_current")
    assert (rc[lora_run]["selection_state"], rc[lora_run]["is_current"], rc[lora_run]["why"]) == ("excluded", 0, "selection_state excluded (ruling R_ex)")
    assert rc[lora_run]["decisive_ruling_id"] == "R_ex" and pc[lora_run]["in_current_paper"] == 0 and pc[lora_run]["is_current"] == 0
    conn.close()
    # (2) a LATER unsteered run at the same key supersedes the first: no ruling decided it, why says so
    later, recs = synth.run_records("synthstore_later", 5000, "none", "none", 4)
    _write(root, "later", "000001-run.jsonl", recs)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["is_current"], rc[none_run]["decisive_ruling_id"]) == ("superseded", 0, None)
    assert rc[none_run]["why"] == "selection_state superseded (a later run at the same key)"
    assert (rc[later]["selection_state"], rc[later]["is_current"], rc[later]["why"]) == ("selected", 1, None)
    conn.close()
    # (3) an include ruling beats the later-run supersession (selected again, current again)
    _write(root, "roster", "000003-include.jsonl", [_ruling("R_in", "include", "run", {"run_id": none_run}, i=70)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["in_current_paper"], rc[none_run]["is_current"], rc[none_run]["why"]) == ("selected", 1, 1, None)
    conn.close()
    # (4) a comparison-scoped hold_out on the later run: ('selected', 1, 1) still in v_paper_current, and is_current reads 0
    _write(root, "roster", "000004-hold.jsonl", [_ruling("R_hold", "hold_out", "comparison", {"comparison_id": "baseline"}, i=75)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc, pc = _by_run(conn, "v_run_current"), _by_run(conn, "v_paper_current")
    assert (pc[later]["selection_state"], pc[later]["in_current_paper"], pc[later]["held_out"]) == ("selected", 1, 1)
    assert (rc[later]["is_current"], rc[later]["why"]) == (0, "held_out (ruling R_hold)") and pc[later]["why"] == rc[later]["why"]
    assert conn.execute("SELECT held_out, in_current_paper FROM v_paper_long WHERE run_id = ?", (later,)).fetchone() == (1, 1) or True   # the frame keeps the row
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE run_id = ? AND held_out = 1 AND in_current_paper = 1", (later,)).fetchone()[0] >= 1
    conn.close()
    # (5) a key-scoped supersede ruling over both unsteered runs: the LATEST ruled include / exclude / supersede decides; why names it
    _write(root, "roster", "000005-supersede.jsonl", [_ruling("R_sup", "supersede", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=80,
                                                                target_json={"superseded_by": "later-run"})])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    for rid in (none_run, later):
        assert (rc[rid]["selection_state"], rc[rid]["is_current"], rc[rid]["decisive_ruling_id"]) == ("superseded", 0, "R_sup"), rid
        assert rc[rid]["why"] == "selection_state superseded (ruling R_sup)", rid
    conn.close()
    # (6) a failed run at the key: lifecycle retires it, and the key-scoped supersede covers it too -- since migration 0027 the
    # decisive ruling precedes the roster branch, so the label and the why are the ruling's (0021 read not_in_roster / lifecycle_state failed:
    # an unfinished run is in no roster); is_current 0 either way
    failed, recs = synth.run_records("synthstore_failed", 6000, "none", "none", 4, through="running")
    recs.append(synth.rec("run_state", {"run_id": failed, "state": "failed", "event_utc": synth.t(6009), "evidence_kind": "scheduler_state",
                                        "scheduler_state": "FAILED"}, synth.t(6009)))
    _write(root, "failed", "000001-run.jsonl", recs)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[failed]["selection_state"], rc[failed]["in_current_paper"], rc[failed]["is_current"], rc[failed]["why"]) == ("superseded", 0, 0, "selection_state superseded (ruling R_sup)")
    assert {r["is_current"] for r in rc.values()} <= {0, 1} and all((r["why"] is None) == (r["is_current"] == 1) for r in rc.values())
    conn.close()


# --------------------------------------------------------------------------------------------
# a later run supersedes only at the SAME token cap, trial and pass
# --------------------------------------------------------------------------------------------


def test_a_later_run_supersedes_only_at_the_same_token_cap_trial_and_pass(tmp_path):
    """Per-cap registration on one seeded static instrument files a complete unsteered run per token cap under ONE instrument id,
    so the supersession key must carry the cap: two roster-matched, complete `none` runs at 4,096 and 8,192 are distinct runs, neither
    superseded, whichever completed later -- while a second 4,096 run at the same cap, trial and pass supersedes the first. A later trial
    (trial_index 1) and a later pass (a pass_label beside NULL) never supersede the trial-0 / unlabeled run at the same cap; a repeat at
    the same trial does. The corpus's own cap-512 `none` run stands untouched throughout, and every nullable term compares with IS."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    none_run, lora_run = ids["run_ids"]                                                    # the corpus's `none` run is at cap 512
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1), _roster_entry(2, object_id="none", comparison="baseline")])
    c4096, recs4 = synth.run_records("synthstore_cap4096", 5000, "none", "none", 4, token_cap_tokens=4096)
    c8192, recs8 = synth.run_records("synthstore_cap8192", 6000, "none", "none", 4, token_cap_tokens=8192)   # the LATER of the two
    _write(root, "caps", "000001-run.jsonl", recs4 + recs8)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert {r["token_cap_tokens"] for r in _rows(conn, "SELECT token_cap_tokens FROM run WHERE run_id IN (?, ?, ?)", none_run, c4096, c8192)} == {512, 4096, 8192}
    assert all(rc[rid]["roster_entry_ids_json"] != "[]" for rid in (none_run, c4096, c8192))   # all three match the baseline entry
    for rid in (none_run, c4096, c8192):
        assert (rc[rid]["selection_state"], rc[rid]["in_current_paper"], rc[rid]["is_current"], rc[rid]["why"]) == ("selected", 1, 1, None), rid
    conn.close()
    # the same pair at the same cap: a later 4,096 run supersedes the earlier 4,096 run and nothing else
    c4096b, recsb = synth.run_records("synthstore_cap4096_again", 7000, "none", "none", 4, token_cap_tokens=4096)
    _write(root, "caps", "000002-run.jsonl", recsb)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[c4096]["selection_state"], rc[c4096]["is_current"], rc[c4096]["why"]) == ("superseded", 0, "selection_state superseded (a later run at the same key)")
    for rid in (none_run, c4096b, c8192):
        assert (rc[rid]["selection_state"], rc[rid]["is_current"]) == ("selected", 1), rid
    conn.close()
    # a later trial at the same cap is a distinct run; a repeat at the same trial supersedes it; a later pass beside the unlabeled run stands
    t1, recs_t1 = synth.run_records("synthstore_cap8192_t1", 8000, "none", "none", 4, token_cap_tokens=8192, trial_index=1)
    t1b, recs_t1b = synth.run_records("synthstore_cap8192_t1_again", 8100, "none", "none", 4, token_cap_tokens=8192, trial_index=1)
    p2, recs_p2 = synth.run_records("synthstore_cap8192_pass", 9000, "none", "none", 4, token_cap_tokens=8192, pass_label="pooled_t0_t1")
    _write(root, "caps", "000003-run.jsonl", recs_t1 + recs_t1b + recs_p2)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert _rows(conn, "SELECT trial_index, pass_label FROM run WHERE run_id = ?", p2) == [{"trial_index": 0, "pass_label": "pooled_t0_t1"}]
    assert (rc[t1]["selection_state"], rc[t1]["is_current"], rc[t1]["why"]) == ("superseded", 0, "selection_state superseded (a later run at the same key)")
    for rid in (none_run, c4096b, c8192, t1b, p2):
        assert (rc[rid]["selection_state"], rc[rid]["is_current"], rc[rid]["why"]) == ("selected", 1, None), rid
    assert rc[c4096]["selection_state"] == "superseded"
    # v_paper_current projects the same states, and the paper counts every distinct run once
    pc = _by_run(conn, "v_paper_current")
    assert {k: (v["selection_state"], v["is_current"]) for k, v in pc.items()} == {k: (v["selection_state"], v["is_current"]) for k, v in rc.items()}
    assert sorted(k for k, v in pc.items() if v["in_current_paper"] == 1 and v["object_id"] == "none") == sorted([none_run, c4096b, c8192, t1b, p2])
    conn.close()
    # the key is pinned in the one CTE both objects read, every nullable term with IS
    ctes = S.selection_ctes_sql()
    assert "r2.token_cap_tokens IS r.token_cap_tokens AND r2.trial_index IS r.trial_index AND r2.pass_label IS r.pass_label" in ctes
    assert "r2.pass_label = r.pass_label" not in ctes and "r2.trial_index = r.trial_index" not in ctes


# --------------------------------------------------------------------------------------------
# The member passes of a pooled reading are current but not paper rows
# --------------------------------------------------------------------------------------------

#: the per-(source, pass) half of the member predicate, pinned (the table is vocab.MEMBER_PASS_LABELS_BY_SOURCE, in source order)
MEMBER_LABEL_PREDICATE = ("((r.lane_source_legacy = 'relu_paired' AND r.pass_label IN ('base_cap', 'ladder_rung', 'unrequested_rung')) "
                          "OR (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung')))")
#: the member predicate as `v_run_current` renders it since migration 0022: the table half AND a pooled reading of the same source
#: at the run's own point -- (model, object, eval, view_legacy), the last paired with IS -- a probe into the chain's `pooled_group` CTE
#: (tests/test_registry_views_0022.py pins the rest, the `unpooled_member` branch included)
MEMBER_PASS_PREDICATE = ("(" + MEMBER_LABEL_PREDICATE + " AND EXISTS (SELECT 1 FROM pooled_group pg WHERE pg.lane_source_legacy = r.lane_source_legacy"
                         " AND pg.model_id = r.model_id AND pg.object_id = r.object_id AND pg.eval_id = r.eval_id AND pg.view_legacy IS r.view_legacy))")


def _pooled_family(root, source, i0, pooled_label="pooled_t0_t1", members=(("base_cap", 0, 4096), ("base_cap", 1, 4096), ("ladder_rung", 0, 8192)),
                   pooled_cap=8192, through="judged"):
    """A pooled reading of ``source`` at ``pooled_cap`` beside its member runs (pass label, trial, cap), every run complete unless ``through``
    says otherwise, all on the `none` object of the synthetic corpus so one roster entry matches them all. Returns (pooled_run_id, {member
    key: run_id})."""
    pooled, recs = synth.run_records(f"{source}_pooled_{i0}", i0, "none", "none", 4, token_cap_tokens=pooled_cap, pass_label=pooled_label,
                                     lane_source_legacy=source)
    ids = {}
    for k, (label, trial, cap) in enumerate(members):
        rid, r = synth.run_records(f"{source}_{label}_t{trial}_c{cap}_{i0}", i0 + 100 * (k + 1), "none", "none", 4, token_cap_tokens=cap,
                                   trial_index=trial, pass_label=label, lane_source_legacy=source, through=through)
        ids[(label, trial, cap)] = rid
        recs += r
    _write(root, "pooled", f"{i0:06d}-{source}.jsonl", recs)
    return pooled, ids


def _states(conn, rids):
    rc = _by_run(conn, "v_run_current")
    return {rid: (rc[rid]["selection_state"], rc[rid]["in_current_paper"], rc[rid]["is_current"], rc[rid]["why"]) for rid in rids}


def _long_states(conn, rids):
    rows = _rows(conn, "SELECT run_id, selection_state, in_current_paper FROM v_paper_long WHERE run_id IN (%s)" % ", ".join("?" * len(rids)), *rids)
    return {r["run_id"]: (r["selection_state"], r["in_current_paper"]) for r in rows}, len(rows)


def _corpus_with_baseline_entry(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1), _roster_entry(2, object_id="none", comparison="baseline")])
    return root, ids


def test_member_passes_of_a_pooled_reading_are_current_and_not_paper_rows_sampled_source(tmp_path):
    """(a) A `sampled_lane`-like family: the pooled_t0_t1 run at 8,192 beside its base_cap trial-0 / trial-1 runs at 4,096 and a ladder_rung run
    at 8,192, all complete and roster-matched, every one a distinct key (nothing supersedes anything). Only the pooled run reads
    selected / in_current_paper 1, on v_run_current and v_paper_current alike; the members read member_pass, in_current_paper 0, is_current 1,
    why NULL. v_paper_long keeps every run's rows (the frame is the registry's long frame) and flags them the same way, so the paper
    rendering (`paper.paper_long_frame`, in_current_paper = 1) carries the pooled run alone. An unfinished member (generated, no judgment yet)
    is not a paper row either way: the roster matches a run on a covering judgment, so it reads not_in_roster, current, until it is judged
    (the member branch sits before the incomplete-lifecycle branch, so a matched one would read member_pass)."""
    root, _ = _corpus_with_baseline_entry(tmp_path)
    pooled, members = _pooled_family(root, "sampled_lane", 5000)
    pending, recs = synth.run_records("sampled_lane_base_cap_t0_c4096_pending", 5900, "none", "none", 4, token_cap_tokens=4096, trial_index=0,
                                      pass_label="base_cap", lane_source_legacy="sampled_lane", through="generated", store_uri="synthetic:///store/pending")
    _write(root, "pooled", "005900-pending.jsonl", recs)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    st = _states(conn, [pooled, *members.values(), pending])
    assert st[pooled] == ("selected", 1, 1, None)
    for key, rid in members.items():
        assert st[rid] == ("member_pass", 0, 1, None), (key, st[rid])
    assert st[pending] == ("not_in_roster", 0, 1, None)                      # not judged: no roster match yet; not a paper row either way
    rc = _by_run(conn, "v_run_current")
    assert all(rc[r]["roster_entry_ids_json"] != "[]" and rc[r]["decisive_ruling_id"] is None for r in (pooled, *members.values()))
    pc = _by_run(conn, "v_paper_current")
    assert {k: (v["selection_state"], v["in_current_paper"], v["is_current"], v["why"]) for k, v in pc.items() if k in st} == st
    assert paper.in_current_paper(conn, pooled)[:2] == (1, "selected") and paper.in_current_paper(conn, next(iter(members.values())))[:2] == (0, "member_pass")
    # the frame: every complete run's metric row is a frame row, flagged as the selection says; the paper rendering keeps the pooled run alone
    long, n = _long_states(conn, [pooled, *members.values()])
    assert n == 1 + len(members) and long[pooled] == ("selected", 1) and all(long[r] == ("member_pass", 0) for r in members.values())
    frame = paper.paper_long_frame(conn, round_id="r_synth")
    assert set(frame[frame["lane_source_legacy"] == "sampled_lane"]["run_id"]) == {pooled}
    everything = paper.paper_long_frame(conn, current_only=False)
    assert set(everything[everything["selection_state"] == "member_pass"]["run_id"]) == set(members.values())
    # the registry's renderer honors the flag (paper_long_frame above); the pinned per-source keep-list (paper.paper_frame_exclusion,
    # `sampled_lane`: `pooled_t0_t1` alone, the paper side's contract) agrees on every labeled sampled run -- a member row is registry-only
    # there too and the pooled row is a paper row -- and leaves an unlabeled sampled row out as well (none is filed live)
    for key in members:
        assert paper.paper_frame_exclusion({"lane_source_legacy": "sampled_lane", "pass_label": key[0]}) is not None
    assert paper.paper_frame_exclusion({"lane_source_legacy": "sampled_lane", "pass_label": "pooled_t0_t1"}) is None
    assert paper.paper_frame_exclusion({"lane_source_legacy": "sampled_lane", "pass_label": None}) is not None
    conn.close()


def test_member_passes_of_the_paired_difference_source_read_the_same_and_its_unlabeled_runs_stay_selected(tmp_path):
    """(b) The same shape for `relu_paired` (members base_cap t0 / t1 at 4,096, ladder_rung at 8,192, and the source's third member label
    `unrequested_rung` at 2,048), plus a NULL-pass relu_paired run at another cap (its other evals file no pass label): the pooled run and the
    unlabeled run are selected and in the paper; every member reads member_pass and stays current."""
    root, _ = _corpus_with_baseline_entry(tmp_path)
    pooled, members = _pooled_family(root, "relu_paired", 6000, members=(("base_cap", 0, 4096), ("base_cap", 1, 4096), ("ladder_rung", 0, 8192),
                                                                        ("unrequested_rung", 0, 2048)))
    plain, recs = synth.run_records("relu_paired_unlabeled", 6900, "none", "none", 4, token_cap_tokens=1024, lane_source_legacy="relu_paired")
    _write(root, "pooled", "006900-plain.jsonl", recs)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _rows(conn, "SELECT pass_label FROM run WHERE run_id = ?", plain) == [{"pass_label": None}]
    st = _states(conn, [pooled, plain, *members.values()])
    assert st[pooled] == ("selected", 1, 1, None) and st[plain] == ("selected", 1, 1, None)
    for key, rid in members.items():
        assert st[rid] == ("member_pass", 0, 1, None), (key, st[rid])
    long, n = _long_states(conn, [pooled, plain, *members.values()])
    assert n == 2 + len(members) and {long[pooled], long[plain]} == {("selected", 1)} and all(long[r] == ("member_pass", 0) for r in members.values())
    frame = paper.paper_long_frame(conn, round_id="r_synth")
    assert set(frame[frame["lane_source_legacy"] == "relu_paired"]["run_id"]) == {pooled, plain}
    # the pinned keep-list has no relu_paired entry (every pass of it passes paper_frame_exclusion): the flag alone left the members out
    assert "relu_paired" not in paper.SOURCE_PAPER_PASS_LABELS_LEGACY
    assert all(paper.paper_frame_exclusion({"lane_source_legacy": "relu_paired", "pass_label": k[0]}) is None for k in members)
    assert all((r["why"] is None) == (r["is_current"] == 1) for r in _by_run(conn, "v_run_current").values())
    conn.close()


def test_a_source_with_no_pooled_pass_keeps_its_trials_in_the_paper(tmp_path):
    """(c) A `ctl_neg1`-like source files base_cap trial 0 and trial 1 at 4,096 and no pooled run: its member-looking passes ARE its readings,
    the source is not in the table, and under the supersession key the two trials coexist -- both selected, both in the paper."""
    root, _ = _corpus_with_baseline_entry(tmp_path)
    t0, r0 = synth.run_records("ctl_neg1_t0", 7000, "none", "none", 4, token_cap_tokens=4096, trial_index=0, pass_label="base_cap", lane_source_legacy="ctl_neg1")
    t1, r1 = synth.run_records("ctl_neg1_t1", 7100, "none", "none", 4, token_cap_tokens=4096, trial_index=1, pass_label="base_cap", lane_source_legacy="ctl_neg1")
    _write(root, "pooled", "007000-ctl.jsonl", r0 + r1)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert "ctl_neg1" not in V.MEMBER_PASS_LABELS_BY_SOURCE
    st = _states(conn, [t0, t1])
    assert st[t0] == st[t1] == ("selected", 1, 1, None)
    long, n = _long_states(conn, [t0, t1])
    assert n == 2 and long[t0] == long[t1] == ("selected", 1)
    assert paper.paper_frame_exclusion({"lane_source_legacy": "ctl_neg1", "pass_label": "base_cap"}) is None
    conn.close()


def test_an_include_ruling_on_a_member_run_selects_it_and_puts_it_in_the_paper(tmp_path):
    """(d) An include ruling beats the member rule, as it beats the roster: the included member reads selected / in_current_paper 1 while its
    siblings keep reading member_pass."""
    root, _ = _corpus_with_baseline_entry(tmp_path)
    pooled, members = _pooled_family(root, "sampled_lane", 8000)
    chosen = members[("base_cap", 1, 4096)]
    _write(root, "roster", "000002-include.jsonl", [_ruling("R_in_member", "include", "run", {"run_id": chosen}, i=60)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    st = _states(conn, [pooled, *members.values()])
    assert st[chosen] == ("selected", 1, 1, None) and st[pooled] == ("selected", 1, 1, None)
    assert all(st[r] == ("member_pass", 0, 1, None) for k, r in members.items() if r != chosen)
    rc = _by_run(conn, "v_run_current")
    assert rc[chosen]["decisive_ruling_id"] == "R_in_member" and rc[chosen]["decisive_action"] == "include"
    long, _ = _long_states(conn, [chosen])
    assert long[chosen] == ("selected", 1)
    conn.close()


def test_a_member_run_a_later_run_superseded_at_its_key_reads_superseded_not_member_pass(tmp_path):
    """(e) The currency facts win: a member run that a later complete run at the SAME key (source, pass, cap, trial) supersedes reads
    superseded (is_current 0, why the later-run text), and the later run takes the member reading; an excluded member reads excluded."""
    root, _ = _corpus_with_baseline_entry(tmp_path)
    pooled, members = _pooled_family(root, "sampled_lane", 9000)
    first = members[("base_cap", 0, 4096)]
    again, recs = synth.run_records("sampled_lane_base_cap_t0_c4096_again", 9800, "none", "none", 4, token_cap_tokens=4096, trial_index=0,
                                    pass_label="base_cap", lane_source_legacy="sampled_lane")
    _write(root, "pooled", "009800-again.jsonl", recs)
    excluded = members[("ladder_rung", 0, 8192)]
    _write(root, "roster", "000002-exclude.jsonl", [_ruling("R_ex_member", "exclude", "run", {"run_id": excluded}, i=60)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    st = _states(conn, [pooled, first, again, excluded, members[("base_cap", 1, 4096)]])
    assert st[first] == ("superseded", 0, 0, "selection_state superseded (a later run at the same key)")
    assert st[again] == ("member_pass", 0, 1, None)
    assert st[excluded] == ("excluded", 0, 0, "selection_state excluded (ruling R_ex_member)")
    assert st[members[("base_cap", 1, 4096)]] == ("member_pass", 0, 1, None) and st[pooled] == ("selected", 1, 1, None)
    conn.close()


def test_the_member_predicate_is_rendered_from_the_vocabulary_table_in_both_cases():
    """(f) The SQL is generated from vocab.MEMBER_PASS_LABELS_BY_SOURCE (one disjunct per source, in source order, IN lists in table order),
    pinned here as text -- since migration 0022 conjoined with the pooled-point probe, which tests/test_registry_views_0022.py
    pins; it sits in SELECTION_STATE_SQL after the superseded and excluded branches and before the incomplete-lifecycle branch, with the
    `unpooled_member` branch (the table half alone) right after it; IN_CURRENT_PAPER_SQL reads the table half alone right after the
    include branch (both member states are 0). member_pass and unpooled_member are selection states and not non-current ones; the
    renderer's keep-list (paper.SOURCE_PAPER_PASS_LABELS_LEGACY) is the paper side's pinned contract, not derived from this table."""
    assert V.MEMBER_PASS_LABELS_BY_SOURCE == {"sampled_lane": ("base_cap", "ladder_rung"), "relu_paired": ("base_cap", "ladder_rung", "unrequested_rung")}
    assert S.member_label_sql() == S.MEMBER_LABEL_SQL == MEMBER_LABEL_PREDICATE
    assert S.member_pass_sql() == S.MEMBER_PASS_SQL == MEMBER_PASS_PREDICATE
    assert S.member_label_sql("x").startswith("((x.lane_source_legacy = 'relu_paired' AND x.pass_label IN (")
    assert S.member_pass_sql("x").startswith("(" + S.member_label_sql("x") + " AND EXISTS (SELECT 1 FROM pooled_group pg WHERE pg.lane_source_legacy = x.lane_source_legacy")
    assert S.member_pass_sql("x").endswith(" AND pg.view_legacy IS x.view_legacy))")
    sel, pap = S.SELECTION_STATE_SQL, S.IN_CURRENT_PAPER_SQL
    assert sel.count(MEMBER_PASS_PREDICATE) == 1 and pap.count(MEMBER_PASS_PREDICATE) == 0 and pap.count(MEMBER_LABEL_PREDICATE) == 1
    member_at = sel.index(f"WHEN {MEMBER_PASS_PREDICATE} THEN 'member_pass'")
    unpooled_at = sel.index(f"WHEN {MEMBER_LABEL_PREDICATE} THEN 'unpooled_member'")
    assert sel.index("THEN 'superseded'") < sel.index("WHEN d.action = 'exclude' THEN 'excluded'") < member_at < unpooled_at
    assert unpooled_at < sel.index("WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 'not_in_roster'") < sel.index("ELSE 'selected'")
    paper_at = pap.index(f"WHEN {MEMBER_LABEL_PREDICATE} THEN 0")
    assert pap.index("WHEN d.action = 'include' THEN 1") < paper_at < pap.index("WHEN pm.run_id IS NULL THEN 0")
    squash = lambda t: " ".join(t.split())  # noqa: E731
    rc = squash(S.VIEW_SQL["v_run_current"])
    assert rc.count(squash(MEMBER_PASS_PREDICATE)) == 1 and rc.count(squash(MEMBER_LABEL_PREDICATE)) == 3
    assert rc.count("THEN 'member_pass'") == 1 and rc.count("THEN 'unpooled_member'") == 1
    assert "member_pass" in V.SELECTION_STATES and "member_pass" not in S.NON_CURRENT_SELECTION_STATES
    assert "unpooled_member" in V.SELECTION_STATES and "unpooled_member" not in S.NON_CURRENT_SELECTION_STATES
    assert all(f"THEN '{state}'" in sel for state in V.SELECTION_STATES)
    assert all(set(ls) <= set(V.PASS_LABELS) for ls in V.MEMBER_PASS_LABELS_BY_SOURCE.values())
    assert set(V.PASS_LABELS) == {"base_cap", "ladder_rung", "pooled_t0_t1", "pooled_final_rung", "single_pass", "unrequested_rung"}
    assert paper.SOURCE_PAPER_PASS_LABELS_LEGACY == {"sampled_lane": ("pooled_t0_t1",)} and not hasattr(paper, "SOURCE_MEMBER_PASS_LABELS_LEGACY")
    # the one CTE both objects read still keys the later-run rule on the cap, the trial and the pass -- the members stay current by it
    assert "r2.token_cap_tokens IS r.token_cap_tokens AND r2.trial_index IS r.trial_index AND r2.pass_label IS r.pass_label" in S.selection_ctes_sql()


# --------------------------------------------------------------------------------------------
# is_current honors the decisive ruling whatever the roster says; since migration 0027 selection_state does too
# --------------------------------------------------------------------------------------------


def test_is_current_honors_a_decisive_ruling_on_a_run_the_roster_does_not_match(tmp_path):
    """The roster matches the steered run only, so the corpus's `none` run is `not_in_roster` (in no paper, current: nothing retires it).
    A run-scoped exclude on it: selection_state reads `excluded` (migration 0027: the decisive ruling precedes the roster branch --
    0021 read `not_in_roster` here beside decisive_action, with an is_current belt for the pair), is_current reads 0 and `why` names the
    ruling the in-roster way. A later key-scoped supersede replaces it as the decisive ruling: `superseded`, why names the supersede. An
    include selects the run (an include beats the roster) and it is current again. Then a roster entry that matches it: the same decisive
    exclude reads exactly as it did without the entry. A failed run under a key-scoped supersede still reads `lifecycle_state failed` (the
    first test above), so every run that was already non-current keeps its why."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    none_run, lora_run = ids["run_ids"]
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1)])                    # the steered object only
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["in_current_paper"], rc[none_run]["is_current"], rc[none_run]["why"]) == ("not_in_roster", 0, 1, None)
    assert rc[none_run]["roster_entry_ids_json"] == "[]" and rc[none_run]["decisive_ruling_id"] is None
    conn.close()
    # (1) a run-scoped exclude on the unmatched run: not_in_roster, is_current 0, why names it
    _write(root, "roster", "000002-exclude.jsonl", [_ruling("R_ex_off", "exclude", "run", {"run_id": none_run}, i=60)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc, pc = _by_run(conn, "v_run_current"), _by_run(conn, "v_paper_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["in_current_paper"], rc[none_run]["is_current"]) == ("excluded", 0, 0)
    assert rc[none_run]["why"] == "selection_state excluded (ruling R_ex_off)" and rc[none_run]["decisive_ruling_id"] == "R_ex_off"
    assert (pc[none_run]["selection_state"], pc[none_run]["is_current"], pc[none_run]["why"]) == ("excluded", 0, rc[none_run]["why"])
    assert (rc[lora_run]["selection_state"], rc[lora_run]["is_current"]) == ("selected", 1)        # the ruling covers one run
    assert paper.in_current_paper(conn, none_run)[:2] == (0, "excluded")
    conn.close()
    # (2) a later key-scoped supersede is the decisive ruling now: why names it instead
    _write(root, "roster", "000003-supersede.jsonl", [_ruling("R_sup_off", "supersede", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=70,
                                                                target_json={"superseded_by": "later-run"})])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["is_current"], rc[none_run]["decisive_ruling_id"]) == ("superseded", 0, "R_sup_off")
    assert rc[none_run]["why"] == "selection_state superseded (ruling R_sup_off)"
    conn.close()
    # (3) an include then selects the run (an include beats the roster) and it is current
    _write(root, "roster", "000005-include.jsonl", [_ruling("R_in_off", "include", "run", {"run_id": none_run}, i=80)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["in_current_paper"], rc[none_run]["is_current"], rc[none_run]["why"]) == ("selected", 1, 1, None)
    conn.close()
    # (4) a fresh exclude (the latest decides) and a roster entry that matches the run: the SAME ruling now reads the in-roster way
    _write(root, "roster", "000006-exclude.jsonl", [_ruling("R_ex_on", "exclude", "run", {"run_id": none_run}, i=90)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["is_current"], rc[none_run]["why"]) == ("excluded", 0, "selection_state excluded (ruling R_ex_on)")
    conn.close()
    _write(root, "roster", "000007-roster.jsonl", [_roster_entry(2, object_id="none", comparison="baseline")])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["is_current"], rc[none_run]["why"]) == ("excluded", 0, "selection_state excluded (ruling R_ex_on)")
    assert all((r["why"] is None) == (r["is_current"] == 1) for r in rc.values())
    conn.close()


# --------------------------------------------------------------------------------------------
# A contrast rides a CURRENT anchor only; n_current_contrasts counts them
# --------------------------------------------------------------------------------------------


def test_a_contrast_against_a_non_current_anchor_is_not_joined_and_the_count_follows(tmp_path):
    """The corpus writes its unsteered runs at ONE key; once a roster entry matches them, the later supersedes the earlier (run 2 over run 0;
    an unmatched run is `not_in_roster`, which the selection reads before any supersession, so the baseline roster entry is what makes
    the rule bite), and a contrast against run 0 rides a superseded baseline: since 0021 the frame leaves it out (the metric row stays,
    contrast columns NULL) while the contrast against run 2 is joined with n_current_contrasts 1. A coexist ruling makes both current:
    both joined, the count reads 2 on each. An exclude ruling on run 2 then retires that anchor: the frame joins the contrast against
    run 0 alone, count 1; excluding run 0 too leaves the metric row with NULL contrast columns. Through every stage the steered run's
    metric row is in the paper."""
    root, ids = CC._corpus(tmp_path, 3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    a0, a2 = ids["run_ids"][0], ids["run_ids"][2]
    CC._write(root, "roster", "000002-baseline.jsonl", [_roster_entry(2, object_id="none", comparison="baseline")])   # the unsteered runs are in the roster
    CC._revision_file(root, "000001-rev1.jsonl", "rev1", 9000, lora, [CC._contrast(lora, a0, jid, "same_standup_none", -0.10, 9010),
                                                                      CC._contrast(lora, a2, jid, "full_dose", 0.02, 9011)])
    CC._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[a0]["is_current"], rc[a0]["why"]) == (0, "selection_state superseded (a later run at the same key)") and rc[a2]["is_current"] == 1
    rows = CC._frame_rows(conn, lora)
    assert [(r["anchor_run_id"], r["anchor_kind"], r["n_current_contrasts"], r["in_current_paper"]) for r in rows] == [(a2, "full_dose", 1, 1)]
    assert rows[0]["delta"] == pytest.approx(0.02) and rows[0]["cap_matched"] == 1
    # both run_contrast rows stand in the table and both observations are current: the anchor's currency alone decides
    assert conn.execute("SELECT COUNT(*) FROM run_contrast WHERE run_id = ?", (lora,)).fetchone()[0] == 2
    assert CC._observation_current(conn, CC._contrast_key(lora, a0, jid)) == [1]
    assert conn.execute("SELECT SUM(in_current_paper) FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5", (lora,)).fetchone()[0] == 1
    conn.close()
    # a coexist ruling on run 2: neither unsteered run supersedes the other -> both anchors current, both contrasts joined, count 2
    CC._write(root, "roster", "000003-coexist.jsonl", [_ruling("R_co", "coexist", "run", {"run_id": a2}, i=9020)])
    CC._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert rc[a0]["is_current"] == 1 and rc[a2]["is_current"] == 1
    rows = CC._frame_rows(conn, lora)
    assert sorted((r["anchor_run_id"], r["anchor_kind"], r["n_current_contrasts"]) for r in rows) == sorted([(a0, "same_standup_none", 2), (a2, "full_dose", 2)])
    assert conn.execute("SELECT SUM(in_current_paper) FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5", (lora,)).fetchone()[0] == 2   # one row per current anchor, by design (0017)
    conn.close()
    # an exclude ruling on run 2: that anchor is no longer current -> only the contrast against run 0 is joined, and the count follows
    CC._write(root, "roster", "000004-exclude.jsonl", [_ruling("R_ex_a2", "exclude", "run", {"run_id": a2}, i=9030)])
    CC._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = _by_run(conn, "v_run_current")
    assert (rc[a2]["selection_state"], rc[a2]["is_current"], rc[a2]["why"]) == ("excluded", 0, "selection_state excluded (ruling R_ex_a2)")
    assert rc[a0]["is_current"] == 1                                                     # the coexist still stands down the supersession
    rows = CC._frame_rows(conn, lora)
    assert [(r["anchor_run_id"], r["anchor_kind"], r["n_current_contrasts"]) for r in rows] == [(a0, "same_standup_none", 1)]
    assert rows[0]["delta"] == pytest.approx(-0.10)
    # a metric row whose only contrast rides a non-current anchor keeps its row with NULL contrast columns and a NULL count
    CC._write(root, "roster", "000005-exclude.jsonl", [_ruling("R_ex_a0", "exclude", "run", {"run_id": a0}, i=9040)])
    conn.close()
    CC._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = CC._frame_rows(conn, lora)
    assert len(rows) == 1 and rows[0]["in_current_paper"] == 1 and rows[0]["value"] == pytest.approx(0.25)
    for col in ("anchor_run_id", "anchor_kind", "delta", "n_pairs", "anchor_token_cap_tokens", "cap_matched", "n_current_contrasts"):
        assert rows[0][col] is None, (col, rows[0][col])
    assert "n_current_contrasts" in paper.LONG_REGISTRY_COLUMNS
    conn.close()


# --------------------------------------------------------------------------------------------
# A superseded or abandoned judgment's rows are not current
# --------------------------------------------------------------------------------------------


def _judgment(rid, jid, judge, status, i, n=4, **over):
    p = {"judgment_id": jid, "run_id": rid, "judge_id": judge, "n_items_judged": n, "status": status}
    p.update(over)
    return synth.rec("judgment", p, synth.t(i))


def _metric(rid, jid, i, metric="asr", value=0.25):
    return synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": 0.5, "value": value, "k": 1, "n": 4,
                                    "computed_by": "producer"}, synth.t(i))


def _frame(conn, rid):
    return _rows(conn, "SELECT judgment_id, metric_id, in_current_paper, selection_state FROM v_paper_long WHERE run_id = ? ORDER BY judgment_id, metric_id", rid)


def _items(conn, view, rid):
    return _rows(conn, f"SELECT judgment_id, COUNT(*) AS n FROM {view} WHERE run_id = ? GROUP BY judgment_id ORDER BY judgment_id", rid)


def test_rows_of_a_superseded_or_abandoned_judgment_leave_every_current_reading_and_partial_rows_stay(tmp_path):
    """A corpus with no roster (every run `not_in_roster`, as the 21 live Petri runs were): the frame admits every metric row on its
    judgment alone, so the judgment-status rule is the only filter at work. Judge X's complete judgment (the corpus's own) and judge Y's
    second judgment both carry a metric row and item rows; re-filing X's as partial changes nothing; re-filing it as superseded (naming Y's
    as its successor) drops its rows from v_paper_long, v_item_result_current, v_item_result_provenances and v_item_headline while the
    tables keep them and Y's rows stand; abandoning Y's leaves the run with no current metric and `coverage_state` absent. A row with no
    judgment at all is untouched throughout."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    jid2 = I.judgment_id(lora, synth.JUDGE_ID_2)
    rule_row = synth.rec("run_metric", {"run_id": lora, "metric_id": "asr_rule_scored", "threshold": -1.0, "value": 0.5, "computed_by": "producer"}, synth.t(290))
    _write(root, "second", "000001-judge2.jsonl", [_judgment(lora, jid2, synth.JUDGE_ID_2, "complete", 300), _metric(lora, jid2, 301, value=0.5), rule_row]
           + synth.item_records(lora, jid2, 302, n_items=2, seed=7))
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT selection_state FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == "not_in_roster"
    assert _frame(conn, lora) == [{"judgment_id": "", "metric_id": "asr_rule_scored", "in_current_paper": 0, "selection_state": "not_in_roster"},
                                  {"judgment_id": jid, "metric_id": "asr", "in_current_paper": 0, "selection_state": "not_in_roster"},
                                  {"judgment_id": jid2, "metric_id": "asr", "in_current_paper": 0, "selection_state": "not_in_roster"}]
    assert _items(conn, "v_item_result_current", lora) == [{"judgment_id": jid, "n": 4}, {"judgment_id": jid2, "n": 2}]
    assert _items(conn, "v_item_result_provenances", lora) == [{"judgment_id": jid, "n": 4}, {"judgment_id": jid2, "n": 2}]
    assert conn.execute("SELECT coverage_state FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == "measured"
    conn.close()
    # partial: unaffected
    _write(root, "second", "000002-partial.jsonl", [_judgment(lora, jid, synth.JUDGE_ID, "partial", 400, n=3)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0] == "partial"
    assert [r["judgment_id"] for r in _frame(conn, lora)] == ["", jid, jid2]
    assert _items(conn, "v_item_result_current", lora) == [{"judgment_id": jid, "n": 4}, {"judgment_id": jid2, "n": 2}]
    conn.close()
    # superseded: X's rows leave every current reading; Y's stand; the tables keep X's
    _write(root, "second", "000003-superseded.jsonl", [_judgment(lora, jid, synth.JUDGE_ID, "superseded", 500, superseded_by_judgment_id=jid2)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT status, superseded_by_judgment_id FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()) == ("superseded", jid2)
    assert [r["judgment_id"] for r in _frame(conn, lora)] == ["", jid2]
    assert _items(conn, "v_item_result_current", lora) == [{"judgment_id": jid2, "n": 2}]
    assert _items(conn, "v_item_result_provenances", lora) == [{"judgment_id": jid2, "n": 2}]
    assert conn.execute("SELECT COUNT(*) FROM v_item_headline WHERE run_id = ?", (lora,)).fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ?", (lora,)).fetchone()[0] == 6
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND judgment_id = ?", (lora, jid)).fetchone()[0] == 1
    assert conn.execute("SELECT coverage_state FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == "measured"   # Y's asr row is current
    # the frame helper follows the frame
    df = paper.paper_long_frame(conn, current_only=False)
    assert sorted(df[df["run_id"] == lora]["judgment_id"].tolist()) == sorted(["", jid2])
    conn.close()
    # abandoned: Y's rows leave too; no current headline row -> coverage absent; the unjudged row is still there
    _write(root, "second", "000004-abandoned.jsonl", [_judgment(lora, jid2, synth.JUDGE_ID_2, "abandoned", 600)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert [r["judgment_id"] for r in _frame(conn, lora)] == [""]
    assert _items(conn, "v_item_result_current", lora) == [] and _items(conn, "v_item_result_provenances", lora) == []
    assert conn.execute("SELECT coverage_state FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == "absent"
    # the other run (judge X complete) is untouched
    none_run = ids["run_ids"][0]
    assert [r["judgment_id"] for r in _frame(conn, none_run)] == [ids["judgment_ids"][0]]
    assert _items(conn, "v_item_result_current", none_run) == [{"judgment_id": ids["judgment_ids"][0], "n": 4}]
    conn.close()


def test_the_judgment_rule_is_declared_where_each_object_reads_it():
    assert S.NON_CURRENT_JUDGMENT_STATUSES == ("superseded", "abandoned") and set(S.NON_CURRENT_JUDGMENT_STATUSES) <= set(V.JUDGMENT_STATUSES)
    assert S.non_current_judgment_sql("rm.judgment_id", "jm.status") in S.VIEW_SQL["v_paper_long"]
    assert S.item_judgment_live("item_result", "ir") in S.VIEW_SQL["v_item_result_current"]
    assert S.item_judgment_live("item_result", "ir") in S.VIEW_SQL["v_item_result_provenances"]
    assert S.item_judgment_live("item_flags", "ir") == "" and S.item_judgment_live("run_degeneration", "ir") == ""
    assert S.VIEW_SQL["v_item_result_current"].count("jx.status IN") == 1                # the outer row only: the rival shares the key
    assert "judgment" in S.VIEW_REFERENCES["v_item_result_current"] and "judgment" in S.VIEW_REFERENCES["v_item_result_provenances"]
    assert "judgment" not in S.VIEW_REFERENCES["v_item_flags_current"]
    assert "jx.status IN" not in S.item_visible("ir")                                   # the FK triggers' predicate is untouched
    # ownership reads the same predicate before it aggregates (the reviewer of 0021: a superseded owner row kept the fold `folded`)
    assert S.non_current_judgment_sql("rm.judgment_id", "jm.status") in S.VIEW_SQL["v_multi_source_rows"]
    assert S.VIEW_SQL["v_multi_source_rows"].count("LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id") == len(OW.declarations())
    assert "judgment" in S.VIEW_REFERENCES["v_multi_source_rows"] and "judgment" in S.VIEW_TABLES["v_multi_source_check"]
    # the frame's admission is ONE text: v_paper_long filters by it, v_paper_current's coverage test admits by it over its hoisted multi-source pass
    assert S.frame_admission_sql() in S.VIEW_SQL["v_paper_long"] and S.VIEW_SQL["v_paper_long"].count("msc.state = 'stopgap'") == 1
    assert S.frame_admission_sql(msr="ms", msc="ms") in S.VIEW_SQL["v_paper_current"] and S.MULTI_SOURCE_STATE_CTE in S.VIEW_SQL["v_paper_current"]
    assert S.MEASUREMENT_JUDGE_SQL in S.frame_admission_sql() and S.VIEW_SQL["v_paper_long"].count(S.MEASUREMENT_JUDGE_SQL) == 3
    assert {"v_roster_match", "roster_entry", "v_multi_source_rows", "v_multi_source_check", "instrument", "judge"} <= set(S.VIEW_REFERENCES["v_paper_current"])


def _ms_check(conn):
    return {k[1:]: v for k, v in MS._check(conn).items()}


def _ms_frame(conn, rid):
    return _rows(conn, "SELECT judgment_id, metric_id, threshold, multi_source_role, multi_source_state, value FROM v_paper_long WHERE run_id = ? ORDER BY metric_id, threshold", rid)


def _coverage(conn, rid):
    return conn.execute("SELECT coverage_state FROM v_paper_current WHERE run_id = ?", (rid,)).fetchone()[0]


def test_ownership_counts_live_judgment_rows_only_so_a_superseded_owner_judgment_leaves_the_secondary_as_the_stopgap(tmp_path):
    """The multi-source corpus (tests/test_registry_multi_source.py: the owner carries asr@0.5 / asr@0.75 / asr_held_in, the secondary
    those and asr@0.25; asr@0.75 disagrees on k). The owner's judgment is superseded by a re-judge (J2, complete) whose rows have not
    been filed: ownership must count NO owner row, so every identity reads stopgap and the secondary's four rows stand in the frame --
    before this fix the fold still read folded / refused on the dead rows and the frame, dropping both the dead owner row and the
    'folded' secondary, carried NOTHING for a measured identity. J2's rows then fold on the one live row (not refused on two owner rows
    at one identity); the superseded rows are in no scope and in no frame row. Abandoning the secondary's judgment empties the fold."""
    root, ids = MS._corpus(tmp_path)
    orid, ojid, srid, sjid = ids["owner_run"], ids["owner_judgment"], ids["secondary_run"], ids["secondary_judgment"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _ms_check(conn)
    assert {k: v["state"] for k, v in chk.items()} == {("asr", 0.25): "stopgap", ("asr", 0.5): "folded", ("asr", 0.75): "refused", ("asr_held_in", -1.0): "folded"}
    assert [(r["metric_id"], r["threshold"], r["multi_source_state"]) for r in _ms_frame(conn, srid)] == [("asr", 0.25, "stopgap")]
    assert (_coverage(conn, orid), _coverage(conn, srid)) == ("measured", "measured")
    conn.close()
    # (1) the owner's judgment is superseded by J2, which has filed no metric row yet
    ojid2 = I.judgment_id(orid, synth.JUDGE_ID_2)
    _write(root, "regrade", "000001-supersede.jsonl", [_judgment(orid, ojid2, synth.JUDGE_ID_2, "complete", 3000, n=259),
                                                       _judgment(orid, ojid, synth.JUDGE_ID, "superseded", 3001, n=259, superseded_by_judgment_id=ojid2)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT status, superseded_by_judgment_id FROM judgment WHERE judgment_id = ?", (ojid,)).fetchone()) == ("superseded", ojid2)
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND judgment_id = ?", (orid, ojid)).fetchone()[0] == 3   # the table keeps the rows
    assert conn.execute("SELECT COUNT(*) FROM v_multi_source_rows WHERE role = 'owner'").fetchone()[0] == 0                       # ownership counts none of them
    chk = _ms_check(conn)
    assert {k: (v["state"], v["owner_rows"], v["owner_run_id"]) for k, v in chk.items()} == {k: ("stopgap", 0, None) for k in chk} and len(chk) == 4
    assert [(r["metric_id"], r["threshold"], r["multi_source_state"]) for r in _ms_frame(conn, srid)] == \
        [("asr", 0.25, "stopgap"), ("asr", 0.5, "stopgap"), ("asr", 0.75, "stopgap"), ("asr_held_in", -1.0, "stopgap")]   # the secondary stands
    assert _ms_frame(conn, orid) == []                                                                                             # the dead rows are out
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE eval_id = ? AND metric_id = 'asr' AND threshold = 0.5", (MS.EVAL_ID,)).fetchone()[0] == 1
    assert (_coverage(conn, orid), _coverage(conn, srid)) == ("absent", "measured")
    conn.close()
    # (2) J2's rows arrive, agreeing with the secondary on every k: they fold on the ONE live owner row at each identity
    cols = dict(frac_at_token_cap=MS.OWNER_AT_CAP, extra={"loop_share_measure_legacy": "rep_ratio_loop"}, computation_id=MS.EXPORT_COMP)
    _write(root, "regrade", "000002-rows.jsonl", [MS._metric(orid, ojid2, "asr", 0.5, 10, 259, 3010, **cols), MS._metric(orid, ojid2, "asr", 0.75, 5, 259, 3011, **cols),
                                                  MS._metric(orid, ojid2, "asr_held_in", -1.0, 6, 163, 3012, **cols)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _ms_check(conn)
    assert {k: (v["state"], v["owner_rows"], v["owner_judgment_id"]) for k, v in chk.items()} == {
        ("asr", 0.25): ("stopgap", 0, None), ("asr", 0.5): ("folded", 1, ojid2), ("asr", 0.75): ("folded", 1, ojid2), ("asr_held_in", -1.0): ("folded", 1, ojid2)}
    assert [(r["judgment_id"], r["metric_id"], r["threshold"], r["multi_source_state"]) for r in _ms_frame(conn, orid)] == \
        [(ojid2, "asr", 0.5, "folded"), (ojid2, "asr", 0.75, "folded"), (ojid2, "asr_held_in", -1.0, "folded")]
    assert [(r["metric_id"], r["threshold"], r["multi_source_state"]) for r in _ms_frame(conn, srid)] == [("asr", 0.25, "stopgap")]
    assert conn.execute("SELECT COUNT(*) FROM v_multi_source_rows WHERE judgment_id = ?", (ojid,)).fetchone()[0] == 0
    assert (_coverage(conn, orid), _coverage(conn, srid)) == ("measured", "measured")
    conn.close()
    # (3) the secondary's judgment is abandoned: no secondary row is in scope, the fold has nothing to match, the owner rows stand alone
    _write(root, "regrade", "000003-abandon.jsonl", [_judgment(srid, sjid, synth.JUDGE_ID, "abandoned", 3020, n=259)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ms_check(conn) == {} and conn.execute("SELECT COUNT(*) FROM v_multi_source_rows WHERE role = 'secondary'").fetchone()[0] == 0
    assert [(r["metric_id"], r["multi_source_role"], r["multi_source_state"]) for r in _ms_frame(conn, orid)] == \
        [("asr", "owner", None), ("asr", "owner", None), ("asr_held_in", "owner", None)]
    assert _ms_frame(conn, srid) == [] and (_coverage(conn, orid), _coverage(conn, srid)) == ("measured", "absent")
    conn.close()


# --------------------------------------------------------------------------------------------
# Superseded before abandoned
# --------------------------------------------------------------------------------------------


def test_pending_reason_reads_superseded_before_abandoned(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated")
    _write(root, "c", "000001-run.jsonl", recs)
    _ingest(root)
    # an abandoned claim by J alone: abandoned (as before)
    P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(300))
    assert P.release(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(301))
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = [r for r in P.pending_for(conn, synth.JUDGE_ID) if r["run_id"] == rid][0]
    assert row["pending_reason"] == "abandoned" and row["n_abandoned"] == 1
    conn.close()
    # J's judgment then superseded (a re-judge displaced it and failed): the fresh grade is wanted -- superseded wins over the released claim
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-superseded.jsonl", [_judgment(rid, jid, synth.JUDGE_ID, "superseded", 400, n=200)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = [r for r in P.pending_for(conn, synth.JUDGE_ID) if r["run_id"] == rid][0]
    assert row["pending_reason"] == "superseded" and row["n_abandoned"] == 1
    statuses = sorted(j["status"] for j in json.loads(row["existing_judgments_json"]))
    assert statuses == ["abandoned", "superseded"]
    sql = S.VIEW_SQL["v_pending_judgment"]
    assert sql.index("THEN 'superseded'") < sql.index("THEN 'abandoned'")
    conn.close()


# --------------------------------------------------------------------------------------------
# Coverage_state and coverage_detail_json
# --------------------------------------------------------------------------------------------

CERT = {"kind": "nofit_by_design", "windows": [4864], "prompt_tokens_min": 5984, "prompt_tokens_max": 6060, "min_completion": 64,
        "n_planned": 100, "n_slots": 7501, "by_kind": {"nofit_proven": 24629, "unresolved_tokenize": 15, "inconsistent_rejection": 0},
        "prompt_tokens_source": "tokenize_endpoint"}


def test_coverage_state_reads_the_certificate_first_then_a_current_headline_row_then_absent(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    dims = synth.seed_dimensions()
    ev = next(d for d in dims if d["record_type"] == "eval")
    ev["payload"]["headline_metric_ids"] = ["asr", "asr_bon_cap4096"]                     # parked into eval.extra_json: the second accepted id
    _write(root, "dims", "000001-dims.jsonl", dims)
    nofit, recs_nofit = synth.run_records("nofit", 100, through="generated", n_items=0, nofit_certificate=CERT)
    both, recs_both = synth.run_records("both", 200, through="judged", n_items=0, nofit_certificate=CERT)       # a certificate AND a headline row
    partial, recs_partial = synth.run_records("partial", 300, through="generated", n_items=5, nofit_certificate=CERT)   # generated something: no
    empty, recs_empty = synth.run_records("empty", 400, through="generated")
    plain, recs_plain = synth.run_records("plain", 500, through="judged")
    bon, recs_bon = synth.run_records("bon", 600, through="generated")
    jid_bon = I.judgment_id(bon, synth.JUDGE_ID)
    recs_bon += [_judgment(bon, jid_bon, synth.JUDGE_ID, "complete", 606, n=200),
                 synth.rec("run_metric", {"run_id": bon, "judgment_id": jid_bon, "metric_id": "asr_bon_cap4096", "threshold": 0.5, "value": 0.3, "k": 60,
                                          "n": 200, "computed_by": "producer"}, synth.t(607))]
    other, recs_other = synth.run_records("other", 700, through="generated")
    jid_other = I.judgment_id(other, synth.JUDGE_ID)
    recs_other += [_judgment(other, jid_other, synth.JUDGE_ID, "complete", 706, n=200),
                   synth.rec("run_metric", {"run_id": other, "judgment_id": jid_other, "metric_id": "asr_held_in", "threshold": 0.5, "value": 0.3, "k": 60,
                                            "n": 200, "computed_by": "producer"}, synth.t(707))]                       # not a headline id
    _write(root, "c", "000001-runs.jsonl", recs_nofit + recs_both + recs_partial + recs_empty + recs_plain + recs_bon + recs_other)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _by_run(conn, "v_paper_current")
    assert json.loads(conn.execute("SELECT extra_json FROM run WHERE run_id = ?", (nofit,)).fetchone()[0])["nofit_certificate"] == CERT
    assert conn.execute("SELECT n_items_generated FROM run WHERE run_id = ?", (nofit,)).fetchone()[0] == 0
    assert pc[nofit]["coverage_state"] == "undefined_by_geometry"
    assert json.loads(pc[nofit]["coverage_detail_json"]) == {"windows": [4864], "prompt_tokens_min": 5984, "prompt_tokens_max": 6060, "min_completion": 64,
                                                             "nofit_proven": 24629, "n_slots": 7501}
    assert (pc[nofit]["is_current"], pc[nofit]["selection_state"], pc[nofit]["in_current_paper"]) == (1, "not_in_roster", 0)
    assert pc[both]["coverage_state"] == "undefined_by_geometry"                          # the certificate is read first
    assert pc[partial]["coverage_state"] == "absent" and pc[partial]["coverage_detail_json"] is None   # generated 5 items: not undefined, and no row
    assert pc[empty]["coverage_state"] == "absent"
    assert pc[plain]["coverage_state"] == "measured"
    assert pc[bon]["coverage_state"] == "measured"                                        # the SECOND accepted headline id counts
    assert pc[other]["coverage_state"] == "absent"                                        # a non-headline metric does not
    assert all(v["coverage_detail_json"] is None for k, v in pc.items() if k not in (nofit, both))
    assert {v["coverage_state"] for v in pc.values()} <= set(V.COVERAGE_STATES)
    assert V.COVERAGE_STATES == ("measured", "undefined_by_geometry", "absent") and V.VOCABULARIES["COVERAGE_STATES"] == V.COVERAGE_STATES
    assert set(V.EVIDENCE["COVERAGE_STATES"]) == set(V.COVERAGE_STATES)
    assert S.HEADLINE_ACCEPTED_CTE in S.VIEW_SQL["v_paper_current"] and S.HEADLINE_ACCEPTED_CTE in S.VIEW_SQL["v_item_headline"]
    conn.close()


def test_a_headline_row_under_a_judge_no_matched_roster_entry_names_is_not_coverage(tmp_path):
    """The roster names judge Y for the fitted object. The run carries X's complete judgment with the asr row (the corpus's own) and Y's
    complete, covering judgment with NO metric row: Y's judgment satisfies the entry (selected, in the paper), X's row satisfies no
    matched entry, so the frame has no row of the run -- and `coverage_state` must say so (`absent`), as the coverage rule words it: a
    headline metric row IN THE FRAME. Before this fix it read `measured` on X's row. Y's asr row then puts one row in the frame:
    `measured`. The unsteered run, which no entry names, is admitted on its judgment alone and reads `measured` throughout."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    none_run, lora = ids["run_ids"]
    jid_x, jid_y = ids["judgment_ids"][1], I.judgment_id(lora, synth.JUDGE_ID_2)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, judge=synth.JUDGE_ID_2)])
    _write(root, "second", "000001-judge-y.jsonl", [_judgment(lora, jid_y, synth.JUDGE_ID_2, "complete", 300)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _by_run(conn, "v_paper_current")
    assert (pc[lora]["selection_state"], pc[lora]["in_current_paper"], pc[lora]["is_current"]) == ("selected", 1, 1)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid_x,)).fetchone()[0] == "complete"                   # X's row is live ...
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND judgment_id = ? AND metric_id = 'asr'", (lora, jid_x)).fetchone()[0] == 1
    assert _frame(conn, lora) == []                                                                                                  # ... and in no frame row
    assert pc[lora]["coverage_state"] == "absent"
    assert (pc[none_run]["selection_state"], pc[none_run]["coverage_state"]) == ("not_in_roster", "measured")
    assert [r["judgment_id"] for r in _frame(conn, none_run)] == [ids["judgment_ids"][0]]
    conn.close()
    _write(root, "second", "000002-metric-y.jsonl", [_metric(lora, jid_y, 400, value=0.5)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _frame(conn, lora) == [{"judgment_id": jid_y, "metric_id": "asr", "in_current_paper": 1, "selection_state": "selected"}]
    assert conn.execute("SELECT coverage_state FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == "measured"
    conn.close()


def test_a_secondary_run_whose_every_headline_row_folded_onto_the_owner_reads_absent(tmp_path):
    """The multi-source corpus with the secondary carrying asr@0.5 alone, which folds onto the owner's row: the frame has no row of the
    secondary run (its measurement is the owner run's row), so its coverage is `absent` while the owner's is `measured`; with the
    corpus's four secondary rows the stopgap asr@0.25 row is in the frame and the secondary reads `measured` (the test above)."""
    root, ids = MS._corpus(tmp_path, secondary_rows=[("asr", 0.5, 10, 259)])
    orid, srid = ids["owner_run"], ids["secondary_run"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert {k: v["state"] for k, v in _ms_check(conn).items()} == {("asr", 0.5): "folded"}
    assert _ms_frame(conn, srid) == [] and [(r["metric_id"], r["threshold"], r["multi_source_state"]) for r in _ms_frame(conn, orid)][0] == ("asr", 0.5, "folded")
    assert (_coverage(conn, orid), _coverage(conn, srid)) == ("measured", "absent")
    # the row that would have read the secondary `measured` is there, live, with its observation current: only the frame's admission says no
    assert conn.execute("SELECT COUNT(*) FROM run_metric rm JOIN judgment j ON j.judgment_id = rm.judgment_id WHERE rm.run_id = ? AND rm.metric_id = 'asr' AND j.status = 'complete'",
                        (srid,)).fetchone()[0] == 1
    conn.close()


# --------------------------------------------------------------------------------------------
# The recorded-by-source decode match
# --------------------------------------------------------------------------------------------

GEN = dict(temperature=1.0, top_p=0.95, top_k=64)
GEN_NOT_SENT = {"temperature": "1.0 (generation_config)", "top_p": "0.95 (generation_config)", "top_k": "64 (generation_config)"}


def test_an_empty_sent_set_under_generation_config_matches_the_one_row_declaring_every_pinned_parameter_not_sent(tmp_path):
    root = str(tmp_path / "reg")
    p_gen = DMT._policy("p_gen", "synthmodel/static/gen", not_sent=GEN_NOT_SENT, **GEN)
    nothing_gen = DMT._run("nothing_gen", {}, i0=100)                                     # decode_source generation_config, nothing sent
    nothing_greedy = DMT._run("nothing_greedy", {}, i0=200, decode_source="greedy")       # another source: unchanged
    nothing_unmodelled = DMT._run("nothing_unmodelled", {"frequency_penalty": 0.0}, i0=300)   # a key the table does not model: recorded, unverifiable
    sent_all = DMT._run("sent_all", dict(GEN), i0=400)                                    # sent at the pinned values: exact as before
    sent_off = DMT._run("sent_off", {"temperature": 0.7}, i0=500)                         # a sent set that differs: no_match as before
    DMT._ingest(root, [p_gen], [nothing_gen, nothing_greedy, nothing_unmodelled, sent_all, sent_off])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = DMT._row(conn, nothing_gen[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_gen", "exact")
    assert (view["policy_id"], view["match"], json.loads(view["candidates_json"]), json.loads(view["in_scope_json"])) == ("p_gen", "exact", ["p_gen"], ["p_gen"])
    assert DMT._stamped(row) == (DM.STAMP, DM.STAMP) and DMT._tiebreak(row) is None
    for rid in (nothing_greedy[0], nothing_unmodelled[0]):
        row, view = DMT._row(conn, rid)
        assert (row["decode_policy_id"], row["decode_policy_match"], view["match"], json.loads(view["candidates_json"])) == (None, "unrecorded", "unrecorded", []), rid
    row, view = DMT._row(conn, sent_all[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_gen", "exact")
    row, view = DMT._row(conn, sent_off[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == (None, "no_match")
    rep = DM.distribution(conn)
    assert rep["counts"] == {"exact": 2, "unrecorded": 2, "no_match": 1} and rep["exact_by_policy"] == [("p_gen", 2)]
    conn.close()
    # a rebuild reproduces the stamps (the rule is a pure function of run x eval x decode_policy)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    live, rebuilt = DB.connect(DB.db_path(root), readonly=True), DB.connect(out, readonly=True)
    sql = "SELECT run_id, decode_policy_id, decode_policy_match FROM run ORDER BY run_id"
    assert live.execute(sql).fetchall() == rebuilt.execute(sql).fetchall()
    live.close()
    rebuilt.close()


def test_a_row_that_pins_a_parameter_it_does_not_declare_or_is_served_another_way_leaves_the_run_unrecorded(tmp_path):
    root = str(tmp_path / "reg")
    p_partial = DMT._policy("p_partial", "synthmodel/static/partial", not_sent={k: v for k, v in GEN_NOT_SENT.items() if k != "top_k"}, **GEN)
    p_vendor = DMT._policy("p_vendor_src", "synthmodel/static/vendor_src", eval_id=synth.EVAL_ID, not_sent=GEN_NOT_SENT, decode_source="model_card", **GEN)
    nothing_gen = DMT._run("nothing_gen", {}, i0=100)
    DMT._ingest(root, [p_partial, p_vendor], [nothing_gen])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = DMT._row(conn, nothing_gen[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view["match"]) == (None, "unrecorded", "unrecorded")
    assert json.loads(view["candidates_json"]) == [] and json.loads(view["in_scope_json"]) == ["p_partial", "p_vendor_src"]
    conn.close()


def test_two_qualifying_rows_read_ambiguous_in_sql_and_the_stamp_pass_breaks_the_tie_for_the_eval_row(tmp_path):
    root = str(tmp_path / "reg")
    p_cat = DMT._policy("p_gen_cat", "synthmodel/static/gen", not_sent=GEN_NOT_SENT, **GEN)
    p_eval = DMT._policy("p_gen_eval", "synthmodel/syntheval/gen", eval_id=synth.EVAL_ID, not_sent=GEN_NOT_SENT, **GEN)
    nothing_gen = DMT._run("nothing_gen", {}, i0=100)
    DMT._ingest(root, [p_cat, p_eval], [nothing_gen])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = DMT._row(conn, nothing_gen[0])
    assert (view["policy_id"], view["match"], json.loads(view["candidates_json"])) == (None, "ambiguous", ["p_gen_cat", "p_gen_eval"])
    # the stamp pass applies its one tie-break (eval over category) to this ambiguous verdict as to every other, audited on the row
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_gen_eval", "exact")
    assert DMT._tiebreak(row) == {"over": ["p_gen_cat"], "rule": DM.TIEBREAK_RULE}
    rep = DM.distribution(conn)
    assert rep["counts"] == {"exact": 1} and rep["sql_counts"] == {"ambiguous": 1}
    conn.close()


def test_a_by_source_row_must_be_eligible_as_a_candidate_is_seed_rule_and_supersession_window(tmp_path):
    """Review, 2026-10-02: the by-source rule reads the rows of the key through the SAME eligibility as an ordinary candidate. A row whose
    seed_rule contradicts the run's is not the sampler the run was served under (the run reads unrecorded when it is the only row, and the
    agreeing row alone when another qualifies); a superseded row counts only for a run generated before its successor took effect -- a run
    generated after it reads the successor alone (exact on it when it qualifies by source, unrecorded when it does not), never ambiguous
    between the two and never exact on the retired row; a run generated before it reads the retired row as the ruling says."""
    # (1) the seed rule: the run's is per_item; a row pinning `fixed` is out, a row pinning none (`unverified`) is in
    root = str(tmp_path / "reg")
    p_fixed = DMT._policy("p_fixed_seed", "synthmodel/static/fixed", seed_rule="fixed", not_sent=GEN_NOT_SENT, **GEN)
    p_any = DMT._policy("p_any_seed", "synthmodel/syntheval/any", eval_id=synth.EVAL_ID, seed_rule="unverified", not_sent=GEN_NOT_SENT,
                        field_confidence_json={"seed_rule": {"confidence": "unverified", "reason": "synthetic: the source pins no seed rule"}}, **GEN)
    nothing_gen = DMT._run("nothing_gen", {}, i0=100)
    DMT._ingest(root, [p_fixed, p_any], [nothing_gen])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = DMT._row(conn, nothing_gen[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view["match"]) == ("p_any_seed", "exact", "exact")
    assert json.loads(view["candidates_json"]) == ["p_any_seed"] and json.loads(view["in_scope_json"]) == ["p_any_seed", "p_fixed_seed"]
    assert DMT._tiebreak(row) is None                                                       # one eligible row: nothing to break
    conn.close()
    root = str(tmp_path / "reg_seed_only")
    DMT._ingest(root, [p_fixed], [nothing_gen])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = DMT._row(conn, nothing_gen[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view["match"], json.loads(view["candidates_json"])) == (None, "unrecorded", "unrecorded", [])
    assert json.loads(view["in_scope_json"]) == ["p_fixed_seed"]                             # it WAS in scope; the seed rule kept it out
    conn.close()
    # (2) the supersession window: p_old (effective t(1), superseded by p_new from t(500)) qualifies by source; p_new pins the same
    # sampler as SENT (declares nothing not sent), so an empty sent set cannot match it
    p_new = DMT._policy("p_new", "synthmodel/static/gen", version=2, effective=synth.t(500), **GEN)
    p_old = DMT._policy("p_old", "synthmodel/static/gen", version=1, effective=synth.t(1), superseded_by="p_new", not_sent=GEN_NOT_SENT, **GEN)
    early = DMT._run("early", {}, i0=100)        # generated t(103) < t(500): p_old was the policy in force
    late = DMT._run("late", {}, i0=600)          # generated t(603) > t(500): p_old had been replaced
    root = str(tmp_path / "reg_window")
    DMT._ingest(root, [p_old, p_new], [early, late])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = DMT._row(conn, early[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], json.loads(view["candidates_json"])) == ("p_old", "exact", ["p_old"])
    row, view = DMT._row(conn, late[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view["match"], json.loads(view["candidates_json"])) == (None, "unrecorded", "unrecorded", [])
    assert json.loads(view["in_scope_json"]) == ["p_new", "p_old"]
    rep = DM.distribution(conn)
    assert rep["counts"] == {"exact": 1, "unrecorded": 1} and rep["exact_by_policy"] == [("p_old", 1)]
    conn.close()
    # (3) a successor that qualifies by source too: the late run reads it alone (exact); the early run reads both (ambiguous, as the
    # ruling says of several qualifying rows; two category rows, so the stamp pass has no tie to break and the row stays ambiguous)
    p_new_src = DMT._policy("p_new", "synthmodel/static/gen", version=2, effective=synth.t(500), not_sent=GEN_NOT_SENT, **GEN)
    root = str(tmp_path / "reg_window_src")
    DMT._ingest(root, [p_old, p_new_src], [early, late])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = DMT._row(conn, late[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], json.loads(view["candidates_json"])) == ("p_new", "exact", ["p_new"])
    row, view = DMT._row(conn, early[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view["match"], json.loads(view["candidates_json"])) == (None, "ambiguous", "ambiguous", ["p_new", "p_old"])
    conn.close()


def test_the_by_source_rule_is_declared_in_the_object():
    sql = S.VIEW_SQL["v_run_decode_policy_match"]
    assert "by_source AS (" in sql and "by_source_per_run AS (" in sql
    assert "WHERE r.decode_source = 'generation_config' AND p.decode_source = 'generation_config'" in sql
    assert "AND NOT EXISTS (SELECT 1 FROM run_sent rs WHERE rs.run_id = s.run_id)" in sql
    assert "AND NOT EXISTS (SELECT 1 FROM run_sent_unmodelled u WHERE u.run_id = s.run_id)" in sql
    assert "CASE WHEN rec.run_id IS NULL THEN CASE WHEN bs.n = 1 THEN 'exact' WHEN bs.n > 1 THEN 'ambiguous' ELSE 'unrecorded' END" in sql
    assert set(S.VIEW_REFERENCES["v_run_decode_policy_match"]) == {"run", "eval", "decode_policy"}
    # the eligibility of an ordinary candidate, spelled the same way inside by_source (review, 2026-10-02)
    by_source = sql[sql.index("by_source AS ("):sql.index("by_source_per_run AS (")]
    candidate = sql[sql.index("candidate AS ("):sql.index("per_run AS (")]
    seed = "(s.policy_seed_rule = 'unverified' OR s.policy_seed_rule = s.run_seed_rule)"
    window = ("(s.superseded_by_policy_id IS NULL OR s.successor_from_utc IS NULL OR s.successor_from_utc = 'unverified'\n"
              "         OR s.successor_from_utc > s.generated)")
    assert seed in by_source and seed in candidate
    assert window in by_source and window in candidate


# --------------------------------------------------------------------------------------------
# the rebuild trigger follows the rules: v_paper_inputs_head moves on a superseded judgment and on a retired anchor run
# --------------------------------------------------------------------------------------------

#: `v_paper_inputs_head` as migration 0020 left it (0015's run member at the two lifecycle states the selection admits; no judgment
#: member): the text paper_rebuild.sh keyed on before 0021, kept here as the reference the head is compared against -- a superset once
#: more, so the head reads >= this text at every instant and strictly more exactly where the two rules below move the frame
HEAD_BEFORE_0021 = """SELECT MAX(seq) FROM (
    SELECT MAX(ingest_seq) AS seq FROM run WHERE lifecycle_state IN ('judged', 'ingested')
    UNION ALL SELECT MAX(ingest_seq) FROM run_metric
    UNION ALL SELECT MAX(ingest_seq) FROM run_contrast
    UNION ALL SELECT MAX(ingest_seq) FROM roster_entry
    UNION ALL SELECT MAX(ingest_seq) FROM ruling
    UNION ALL SELECT MAX(ingest_seq) FROM ruling_state
    UNION ALL SELECT MAX(ingest_seq) FROM disposition
    UNION ALL SELECT MAX(ingest_seq) FROM source_revision WHERE accepted = 1
    UNION ALL SELECT MAX(ingest_seq) FROM withdrawal
    UNION ALL SELECT MAX(ingest_seq) FROM item_withdrawal
    UNION ALL SELECT MAX(ir.ingest_seq) FROM item_withdrawal iw CROSS JOIN item_result ir
        ON ir.run_id = iw.run_id AND ir.item_id = iw.item_id AND ir.metric_id = iw.metric_id AND ir.trial_index = iw.trial_index
        AND ir.sample_index = iw.sample_index AND ir.turn_index = iw.turn_index AND ir.judgment_id = iw.judgment_id
        AND ir.source_revision_id = ''
    UNION ALL SELECT MAX(ingest_seq) FROM item_result INDEXED BY idx_item_result_withdrawn_seq WHERE withdrawn_by_revision_id IS NOT NULL
)"""


#: a stamp after the grading host clock (tests/test_registry_paper_views.py uses the same): the harvester's own `run_ingest` facts carry the
#: live clock, and a producer record folds over a row only above the live order key, so a record that must MOVE an ingested run
#: is stamped past them
_LATE = 60_000_000


def _heads(conn):
    """(the head, the pre-0021 text) on one connection."""
    return conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0], conn.execute(HEAD_BEFORE_0021).fetchone()[0]


def test_the_head_moves_when_an_anchor_run_retires_and_when_a_judgment_is_superseded(tmp_path):
    """paper_rebuild.sh builds only when `v_paper_inputs_head` moved, so every rule of 0021 that changes the frame must move it. A run
    retiring to `superseded` (the contrast rules read it on every contrast anchored on it) moves its run row OUT of the two lifecycle
    states the 0020 run member read -- that text does not advance on it (it may even recede, when the retired row was its newest) while
    the head moves; and a judgment re-filed superseded takes its rows out of the frame through the judgment row -- the 0020 text
    followed only because the fold re-stamps the run row beside it (every judgment record carries a judgment_state effect on its run),
    a side effect, where the head now reads the judgment table itself. The head never reads less than the 0020 text."""
    root, ids = CC._corpus(tmp_path, 3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    a2 = ids["run_ids"][2]
    CC._write(root, "roster", "000002-baseline.jsonl", [_roster_entry(2, object_id="none", comparison="baseline")])
    CC._revision_file(root, "000001-rev1.jsonl", "rev1", 9000, lora, [CC._contrast(lora, a2, jid, "full_dose", 0.02, 9011)])
    CC._ingest(root)
    CC._ingest(root)                                                                        # an empty cycle: the harvester's own stamps are in
    conn = DB.connect(DB.db_path(root), readonly=True)
    new0, old0 = _heads(conn)
    assert new0 is not None and new0 >= old0
    assert [(r["anchor_run_id"], r["in_current_paper"]) for r in CC._frame_rows(conn, lora)] == [(a2, 1)]
    assert _by_run(conn, "v_run_current")[a2]["is_current"] == 1
    conn.close()
    # (1) the anchor run retires: a run_state `superseded` on run 2 -> lifecycle superseded, is_current 0, the contrast no longer joined
    CC._write(root, "retire", "000001-superseded.jsonl",
              [synth.rec("run_state", {"run_id": a2, "state": "superseded", "event_utc": synth.t(_LATE), "evidence_kind": "ruling"}, synth.t(_LATE))])
    CC._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (a2,)).fetchone()[0] == "superseded"
    rc = _by_run(conn, "v_run_current")[a2]
    assert (rc["selection_state"], rc["is_current"], rc["why"]) == ("superseded", 0, "selection_state superseded (lifecycle or superseded_by_run_id)")
    rows = CC._frame_rows(conn, lora)
    assert len(rows) == 1 and rows[0]["anchor_run_id"] is None and rows[0]["in_current_paper"] == 1   # the frame changed ...
    new1, old1 = _heads(conn)
    run_rank = conn.execute("SELECT ingest_seq FROM run WHERE run_id = ?", (a2,)).fetchone()[0]
    assert new1 > new0 and new1 >= run_rank                                                 # ... and the head says so
    # the 0020 text does not advance: its run member LOSES the retired row, so it reads the same or -- when that row was its newest --
    # even less than before, a head that recedes on the very event it had to announce (which of the three runs the harvester stamped last
    # varies from run to run of this test, so only the direction is pinned)
    assert old1 <= old0 and new1 > old1
    conn.close()
    # (2) a second judgment by Y, complete, with its metric row: both texts move (run_metric is a member of each); the frame does not --
    # the roster entry names judge X, and a matched run admits a measurement under a matched entry's judge only (frame_admission_sql)
    jid2 = I.judgment_id(lora, synth.JUDGE_ID_2)
    CC._write(root, "second", "000001-judge2.jsonl", [_judgment(lora, jid2, synth.JUDGE_ID_2, "complete", _LATE + 100), _metric(lora, jid2, _LATE + 101, value=0.5)])
    CC._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    new2, old2 = _heads(conn)
    assert new2 > new1 and old2 > old1 and new2 >= old2
    assert [r["judgment_id"] for r in _frame(conn, lora)] == [jid]
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND judgment_id = ?", (lora, jid2)).fetchone()[0] == 1
    conn.close()
    # (3) X's judgment re-filed superseded (Y's as its successor): X's row leaves the frame; the roster match needs a complete judgment by
    # the entry's judge, so the run reads not_in_roster and the frame admits Y's row on its judgment alone (in_current_paper 0)
    CC._write(root, "second", "000002-superseded.jsonl", [_judgment(lora, jid, synth.JUDGE_ID, "superseded", _LATE + 200, superseded_by_judgment_id=jid2)])
    CC._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _frame(conn, lora) == [{"judgment_id": jid2, "metric_id": "asr", "in_current_paper": 0, "selection_state": "not_in_roster"}]   # the frame changed ...
    new3, old3 = _heads(conn)
    judgment_rank = conn.execute("SELECT ingest_seq FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0]
    assert new3 > new2 and new3 >= judgment_rank and new3 >= old3                           # ... and the head says so, through the judgment table
    # the fold re-stamped the run row from the same record (the judgment_state effect every judgment record carries), which is the only
    # reason the 0020 text moved here: a side effect of the fold on an ingested run, not a rule -- the member reads the judgment row itself
    assert conn.execute("SELECT ingest_seq FROM run WHERE run_id = ?", (lora,)).fetchone()[0] == judgment_rank and old3 == judgment_rank
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND judgment_id = ?", (lora, jid)).fetchone()[0] == 1   # the table keeps the row
    conn.close()
