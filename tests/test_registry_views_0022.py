"""Migration 0022 (2026-10-02): a member pass is a member only where its source filed a pooled reading at
its own POINT; elsewhere it is an `unpooled_member`, not a paper row either.

Migration 0021 rendered `member_pass` from `vocab.MEMBER_PASS_LABELS_BY_SOURCE` alone, so a roster-matched run of a listed
(source, pass) read member_pass whether or not its source had filed the pooled reading the paper reads for its point -- and the sources
pool by POINT: (model, object, eval, `view_legacy`), not by (model, object, eval). Live, 2026-10-02: 406 complete `sampled_lane` runs
read member_pass at a point with no pooled run of their source (266 in a (model, object, eval) with no pooled run at all, 140 beside a
pooled run of ANOTHER point). The fix: `schema_def.member_pass_sql` conjoins the per-(source, pass) table with an EXISTS probe into the
chain's `pooled_group` CTE (`schema_def.POOLED_GROUP_CTE`: the distinct (source, model_id, object_id, eval_id, view_legacy) of every run
under a pooled pass label, `vocab.POOLED_PASS_LABELS`, in ANY lifecycle or selection state; `view_legacy` paired with IS), and a labeled
run whose point carries no pooled reading of its source reads the new `unpooled_member` (`vocab.SELECTION_STATES`; not a non-current
state; `in_current_paper` 0 as for `member_pass`, which is what the paper side's pinned renderer contract reads -- the paper's reading of
a listed source's point is its pooled run, so a labeled run never enters the rendered paper by the label alone). One rule per test on a
synthetic corpus:

* (a) a point with no pooled run: its complete, roster-matched labeled runs read `unpooled_member` / 0 / is_current 1 / why NULL on
  v_run_current, v_paper_current and v_paper_long, and leave both the registry's current frame and the rendered paper frame;
* (b) a pooled run marks the members of its own point only: the same source's members at another `view_legacy`, or on another object, of
  the same (model, eval) stay `unpooled_member`; a pooled run with NO point string marks the unlabeled members only, and one with a point
  string never marks them;
* (c) a pooled run a later pooled run superseded, or a ruling excluded -- no pooled run selected at all -- still marks its members;
* (d) `relu_paired`: another source's pooled run at the same point marks nothing; its own `pooled_final_rung` does, as `pooled_t0_t1`
  does; on the rendered path the flag alone leaves its members out (no keep-list entry);
* (e) an incomplete member at an unpooled point reads not_in_roster (the lifecycle rule);
* (g) an include ruling on an unpooled member selects it (the view and the raw frame), while the pinned keep-list still leaves a
  `sampled_lane` member pass out of the rendered paper -- the paper side's contract, unchanged by this migration;
* (f) the rendered text is pinned: the table half AND the probe with its `view_legacy` term in SELECTION_STATE_SQL, the `unpooled_member`
  branch right after it, the label half alone in IN_CURRENT_PAPER_SQL, the CTE rendered from vocab.POOLED_PASS_LABELS into the chain and
  nowhere else.

The migration mechanics (the file is the render against the v21 shape, one recreate, nothing else) are
tests/test_registry_migrations.py::test_0022_member_pass_requires_pooled_reading_on_a_v21_database's.
"""
from __future__ import annotations

import test_registry_views_0021 as T21

from sisteer.registry import db as DB
from sisteer.registry import paper
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V

MEMBERS = (("base_cap", 0, 4096), ("base_cap", 1, 4096), ("ladder_rung", 0, 8192))
RP_MEMBERS = MEMBERS + (("unrequested_rung", 0, 2048),)
LORA_KEY = "constitution_lora@deadbeef"
#: a rendered-path column set (every name a long-schema column the registry supplies); the registry columns ride beside them
COLS = ["source", "model", "metric", "value"]

SELECTED = ("selected", 1, 1, None)
MEMBER = ("member_pass", 0, 1, None)
UNPOOLED = ("unpooled_member", 0, 1, None)


def _family(root, source, i0, members=MEMBERS, pooled=None, pooled_cap=8192, through="judged", object_key="none", object_id="none",
            name="fam", view=None):
    """``members`` runs of ``source`` (pass label, trial, cap) on one object at one point (``view`` = `view_legacy`, NULL by default), every
    one complete unless ``through`` says otherwise, plus -- when ``pooled`` names a pass label -- one pooled run of that label at ``pooled_cap``
    at the same point. Returns ({member key: run_id}, pooled_run_id or None)."""
    recs, ids = [], {}
    extra = {} if view is None else {"view_legacy": view}
    for k, (label, trial, cap) in enumerate(members):
        rid, r = synth.run_records(f"{source}_{name}_{label}_t{trial}_c{cap}_{i0}", i0 + 100 * (k + 1), object_key, object_id, 4,
                                   token_cap_tokens=cap, trial_index=trial, pass_label=label, lane_source_legacy=source, through=through, **extra)
        ids[(label, trial, cap)] = rid
        recs += r
    pooled_id = None
    if pooled:
        pooled_id, r = synth.run_records(f"{source}_{name}_{pooled}_{i0}", i0, object_key, object_id, 4, token_cap_tokens=pooled_cap,
                                         pass_label=pooled, lane_source_legacy=source, **extra)
        recs += r
    T21._write(root, "pooled", f"{i0:06d}-{source}-{name}.jsonl", recs)
    return ids, pooled_id


def _open(tmp_path):
    root, _ = T21._corpus_with_baseline_entry(tmp_path)
    return root


def _conn(root):
    T21._ingest(root)
    return DB.connect(DB.db_path(root), readonly=True)


def _rendered(conn, *rids):
    """Which of ``rids`` are on the RENDERED paper path (`paper_long_frame(columns=...)`: `paper_frame_exclusion` applied, in_current_paper 1;
    the frame carries the paper spelling `source` beside the registry's run_id)."""
    frame = paper.paper_long_frame(conn, "r_synth", columns=COLS)
    assert "run_id" in frame.columns and "source" in frame.columns and "lane_source_legacy" not in frame.columns
    return set(frame["run_id"]) & set(rids)


def _current(conn, *rids):
    """Which of ``rids`` are in the registry's own current frame (`paper_long_frame()`: in_current_paper 1, every row kept)."""
    return set(paper.paper_long_frame(conn, round_id="r_synth")["run_id"]) & set(rids)


# --------------------------------------------------------------------------------------------
# (a) a point with no pooled reading: unpooled_member, not a paper row
# --------------------------------------------------------------------------------------------


def test_a_point_with_no_pooled_run_reads_unpooled_member_and_is_not_a_paper_row(tmp_path):
    """(a) base_cap trial 0 / trial 1 at 4,096 and ladder_rung trial 0 at 8,192 of `sampled_lane` at one point, all complete and
    roster-matched, no pooled run anywhere: all three read unpooled_member / in_current_paper 0 / is_current 1 / why NULL (distinct keys
    under the supersession rule: nothing supersedes anything; matched, so not off-roster), v_paper_current projects the same, v_paper_long
    flags every row (unpooled_member, 0), and neither the registry's current frame nor the rendered paper frame carries them."""
    root = _open(tmp_path)
    members, _ = _family(root, "sampled_lane", 5000, view="pt_a")
    conn = _conn(root)
    assert conn.execute("SELECT COUNT(*) FROM run WHERE pass_label IN (%s)" % ", ".join("?" * len(V.POOLED_PASS_LABELS)), V.POOLED_PASS_LABELS).fetchone()[0] == 0
    assert T21._rows(conn, "SELECT DISTINCT view_legacy FROM run WHERE lane_source_legacy = 'sampled_lane'") == [{"view_legacy": "pt_a"}]
    st = T21._states(conn, list(members.values()))
    assert st == {rid: UNPOOLED for rid in members.values()}, st
    rc = T21._by_run(conn, "v_run_current")
    assert all(rc[r]["roster_entry_ids_json"] != "[]" and rc[r]["decisive_ruling_id"] is None for r in members.values())
    pc = T21._by_run(conn, "v_paper_current")
    assert {k: (v["selection_state"], v["in_current_paper"], v["is_current"], v["why"]) for k, v in pc.items() if k in st} == st
    assert all(paper.in_current_paper(conn, r)[:2] == (0, "unpooled_member") for r in members.values())
    long, n = T21._long_states(conn, list(members.values()))
    assert n == len(members) and all(long[r] == ("unpooled_member", 0) for r in members.values())
    assert _current(conn, *members.values()) == set() and _rendered(conn, *members.values()) == set()
    everything = paper.paper_long_frame(conn, current_only=False)
    assert set(everything[everything["selection_state"] == "unpooled_member"]["run_id"]) == set(members.values())
    assert not (everything["selection_state"] == "member_pass").any()
    conn.close()


# --------------------------------------------------------------------------------------------
# (b) a pooled reading marks the members of its own point only
# --------------------------------------------------------------------------------------------


def test_a_pooled_run_marks_the_members_of_its_own_point_and_no_other(tmp_path):
    """(b, i) A pooled_t0_t1 run at point pt_a of the `none` object beside its members: they read member_pass / 0. The same source's
    members at pt_b of the same (model, object, eval) -- trials 2 and 3, so the supersession key (which has no point term) makes them
    distinct runs and not successors; no pooled run at pt_b -- stay unpooled_member, and so do its members at pt_a of the lora object
    (matched by the other roster entry): the pooled run marks its own (model, object, eval, view_legacy) alone. Only the pooled run is a
    paper row, on the registry's current frame and the rendered path alike."""
    root = _open(tmp_path)
    a_members, pooled = _family(root, "sampled_lane", 5000, pooled="pooled_t0_t1", view="pt_a")
    b_members, _ = _family(root, "sampled_lane", 6000, members=(("base_cap", 2, 4096), ("base_cap", 3, 4096), ("ladder_rung", 2, 8192)), view="pt_b", name="ptb")
    lora_members, _ = _family(root, "sampled_lane", 7000, object_key=LORA_KEY, object_id=synth.OBJECT_SHA, name="lora", view="pt_a")
    conn = _conn(root)
    every = [pooled, *a_members.values(), *b_members.values(), *lora_members.values()]
    st = T21._states(conn, every)
    assert st[pooled] == SELECTED
    assert {k: st[r] for k, r in a_members.items()} == {k: MEMBER for k in a_members}
    assert {k: st[r] for k, r in b_members.items()} == {k: UNPOOLED for k in b_members}
    assert {k: st[r] for k, r in lora_members.items()} == {k: UNPOOLED for k in lora_members}
    rc = T21._by_run(conn, "v_run_current")
    assert all(rc[r]["roster_entry_ids_json"] != "[]" for r in [*b_members.values(), *lora_members.values()])   # matched: unpooled by the rule, not off-roster
    assert _current(conn, *every) == {pooled} and _rendered(conn, *every) == {pooled}
    everything = paper.paper_long_frame(conn, current_only=False)
    assert set(everything[everything["selection_state"] == "member_pass"]["run_id"]) == set(a_members.values())
    assert set(everything[everything["selection_state"] == "unpooled_member"]["run_id"]) == set(b_members.values()) | set(lora_members.values())
    conn.close()


def test_a_pooled_run_with_no_point_string_marks_the_unlabeled_members_only(tmp_path):
    """(b, ii) `view_legacy` pairs with IS: a pooled run filing NO point string marks the members with none (NULL with NULL) and not the
    members at pt_a; in another corpus a pooled run at pt_a marks the pt_a members and not the unlabeled ones."""
    root = _open(tmp_path / "null_pooled")
    null_members, pooled = _family(root, "sampled_lane", 5000, pooled="pooled_t0_t1")
    a_members, _ = _family(root, "sampled_lane", 6000, members=(("base_cap", 2, 4096), ("ladder_rung", 2, 8192)), view="pt_a", name="pta")
    conn = _conn(root)
    assert T21._rows(conn, "SELECT view_legacy FROM run WHERE run_id = ?", pooled) == [{"view_legacy": None}]
    st = T21._states(conn, [pooled, *null_members.values(), *a_members.values()])
    assert st[pooled] == SELECTED
    assert {k: st[r] for k, r in null_members.items()} == {k: MEMBER for k in null_members}
    assert {k: st[r] for k, r in a_members.items()} == {k: UNPOOLED for k in a_members}
    conn.close()
    root = _open(tmp_path / "pta_pooled")
    null_members, _ = _family(root, "sampled_lane", 5000, members=(("base_cap", 2, 4096), ("ladder_rung", 2, 8192)))
    a_members, pooled = _family(root, "sampled_lane", 6000, pooled="pooled_t0_t1", view="pt_a", name="pta")
    conn = _conn(root)
    st = T21._states(conn, [pooled, *null_members.values(), *a_members.values()])
    assert st[pooled] == SELECTED
    assert {k: st[r] for k, r in a_members.items()} == {k: MEMBER for k in a_members}
    assert {k: st[r] for k, r in null_members.items()} == {k: UNPOOLED for k in null_members}
    conn.close()


# --------------------------------------------------------------------------------------------
# (c) a pooled run in any state marks: superseded by a later pooled run, excluded by ruling
# --------------------------------------------------------------------------------------------


def test_a_superseded_or_excluded_pooled_run_still_marks_its_members(tmp_path):
    """(c) Two pooled_t0_t1 runs of one point at the same key: the later one supersedes the first (same cap, trial, pass), and a
    run-scoped exclude ruling then retires the later one -- NO pooled run of the point is selected -- yet the members read member_pass:
    the source filed a pooled reading, so the members are never the paper reading. The two pooled runs read superseded / excluded with
    their why texts, and the point has no paper row on either frame."""
    root = _open(tmp_path)
    members, first = _family(root, "sampled_lane", 5000, pooled="pooled_t0_t1", view="pt_a")
    later, recs = synth.run_records("sampled_lane_pooled_again", 5900, "none", "none", 4, token_cap_tokens=8192, pass_label="pooled_t0_t1",
                                    lane_source_legacy="sampled_lane", view_legacy="pt_a")
    T21._write(root, "pooled", "005900-again.jsonl", recs)
    T21._write(root, "roster", "000002-exclude.jsonl", [T21._ruling("R_ex_pooled", "exclude", "run", {"run_id": later}, i=60)])
    conn = _conn(root)
    st = T21._states(conn, [first, later, *members.values()])
    assert st[first] == ("superseded", 0, 0, "selection_state superseded (a later run at the same key)")
    assert st[later] == ("excluded", 0, 0, "selection_state excluded (ruling R_ex_pooled)")
    assert {k: st[r] for k, r in members.items()} == {k: MEMBER for k in members}
    assert conn.execute("SELECT COUNT(*) FROM v_run_current WHERE selection_state = 'selected' AND run_id IN (?, ?)", (first, later)).fetchone()[0] == 0
    assert _current(conn, first, later, *members.values()) == set() and _rendered(conn, first, later, *members.values()) == set()
    conn.close()


# --------------------------------------------------------------------------------------------
# (d) relu_paired: either pooled label of ITS OWN source at the point marks; another source's pooled run does not
# --------------------------------------------------------------------------------------------


def test_relu_paired_members_are_not_marked_by_another_sources_pooled_run_at_their_point(tmp_path):
    """(d, i) `relu_paired` members (its three member labels) at pt_a beside a `sampled_lane` pooled_t0_t1 run at the same point: the
    pooled run is another source's, so the relu_paired runs read unpooled_member / 0 and the sampled run selected / 1; the rendered path
    carries the sampled pooled run alone."""
    root = _open(tmp_path)
    rp, _ = _family(root, "relu_paired", 5000, members=RP_MEMBERS, view="pt_a")
    _, sampled_pooled = _family(root, "sampled_lane", 6000, members=(), pooled="pooled_t0_t1", name="other", view="pt_a")
    conn = _conn(root)
    st = T21._states(conn, [sampled_pooled, *rp.values()])
    assert st[sampled_pooled] == SELECTED
    assert {k: st[r] for k, r in rp.items()} == {k: UNPOOLED for k in rp}
    assert _current(conn, sampled_pooled, *rp.values()) == {sampled_pooled} and _rendered(conn, sampled_pooled, *rp.values()) == {sampled_pooled}
    conn.close()


def test_relu_paired_members_are_marked_by_either_pooled_label_of_their_own_source_at_their_point(tmp_path):
    """(d, ii) A relu_paired pooled_final_rung run (the controls' pooled label, at each pair's final cap) at the members' point marks them
    as pooled_t0_t1 does: every member reads member_pass / 0, the pooled run selected / 1, v_paper_long flags them the same way, and on
    the rendered path -- `relu_paired` has no keep-list entry -- the flag alone leaves the members out and keeps the pooled run."""
    root = _open(tmp_path)
    rp, pooled = _family(root, "relu_paired", 5000, members=RP_MEMBERS, pooled="pooled_final_rung", pooled_cap=2048, view="pt_a")
    conn = _conn(root)
    assert "pooled_final_rung" in V.POOLED_PASS_LABELS and "pooled_t0_t1" in V.POOLED_PASS_LABELS
    st = T21._states(conn, [pooled, *rp.values()])
    assert st[pooled] == SELECTED and {k: st[r] for k, r in rp.items()} == {k: MEMBER for k in rp}
    long, n = T21._long_states(conn, [pooled, *rp.values()])
    assert n == 1 + len(rp) and long[pooled] == ("selected", 1) and all(long[r] == ("member_pass", 0) for r in rp.values())
    assert "relu_paired" not in paper.SOURCE_PAPER_PASS_LABELS_LEGACY
    assert _current(conn, pooled, *rp.values()) == {pooled} and _rendered(conn, pooled, *rp.values()) == {pooled}
    conn.close()


# --------------------------------------------------------------------------------------------
# (e) an incomplete member at an unpooled point: the lifecycle rule, not the member rule
# --------------------------------------------------------------------------------------------


def test_an_incomplete_member_at_an_unpooled_point_reads_not_in_roster(tmp_path):
    """(e) A generated, not yet judged base_cap run of `sampled_lane` at a point with no pooled run: the roster matches a run on a covering
    judgment, so it reads not_in_roster / 0 / current, while its complete siblings read unpooled_member / 0."""
    root = _open(tmp_path)
    members, _ = _family(root, "sampled_lane", 5000, members=(("base_cap", 1, 4096), ("ladder_rung", 0, 8192)), view="pt_a")
    pending, recs = synth.run_records("sampled_lane_base_cap_t0_pending", 5900, "none", "none", 4, token_cap_tokens=4096, trial_index=0,
                                      pass_label="base_cap", lane_source_legacy="sampled_lane", view_legacy="pt_a", through="generated",
                                      store_uri="synthetic:///store/pending")
    T21._write(root, "pooled", "005900-pending.jsonl", recs)
    conn = _conn(root)
    st = T21._states(conn, [pending, *members.values()])
    assert st[pending] == ("not_in_roster", 0, 1, None)
    assert {k: st[r] for k, r in members.items()} == {k: UNPOOLED for k in members}
    assert T21._by_run(conn, "v_run_current")[pending]["roster_entry_ids_json"] == "[]"
    conn.close()


# --------------------------------------------------------------------------------------------
# (g) an include ruling selects an unpooled member; the rendered path keeps the pinned keep-list
# --------------------------------------------------------------------------------------------


def test_an_include_ruling_selects_an_unpooled_member_and_the_rendered_path_keeps_the_pinned_contract(tmp_path):
    """(g) An include ruling precedes the member branches: the included unpooled member reads selected / in_current_paper 1 on
    v_run_current, v_paper_current and v_paper_long, its siblings keep reading unpooled_member. The rendered paper path still leaves a
    `sampled_lane` member pass out -- `paper_frame_exclusion`'s keep-list (`pooled_t0_t1` alone) is the paper side's pinned contract and
    this migration does not touch it -- so the registry's current frame and the rendered frame differ on exactly that run."""
    root = _open(tmp_path)
    members, _ = _family(root, "sampled_lane", 5000, view="pt_a")
    chosen = members[("base_cap", 1, 4096)]
    T21._write(root, "roster", "000002-include.jsonl", [T21._ruling("R_in_unpooled", "include", "run", {"run_id": chosen}, i=60)])
    conn = _conn(root)
    st = T21._states(conn, list(members.values()))
    assert st[chosen] == SELECTED and all(st[r] == UNPOOLED for k, r in members.items() if r != chosen)
    rc = T21._by_run(conn, "v_run_current")
    assert rc[chosen]["decisive_ruling_id"] == "R_in_unpooled" and rc[chosen]["decisive_action"] == "include"
    long, _ = T21._long_states(conn, [chosen])
    assert long[chosen] == ("selected", 1)
    assert _current(conn, *members.values()) == {chosen}
    assert _rendered(conn, *members.values()) == set()
    assert paper.paper_frame_exclusion({"lane_source_legacy": "sampled_lane", "pass_label": "base_cap"}) is not None
    conn.close()


# --------------------------------------------------------------------------------------------
# (f) the rendered text
# --------------------------------------------------------------------------------------------


def test_the_member_predicate_probes_the_pooled_point_and_the_unpooled_branch_follows_it():
    """(f) member_pass_sql = the per-(source, pass) table half (pinned in the 0021 test) AND an EXISTS into `pooled_group` correlated
    on source, model, object, eval and -- with IS -- view_legacy; it stands once in SELECTION_STATE_SQL where the 0021 predicate stood,
    the `unpooled_member` branch (the label half alone) right after it, and IN_CURRENT_PAPER_SQL reads the label half alone where the
    0021 predicate stood (both member states are 0). POOLED_GROUP_CTE is rendered from vocab.POOLED_PASS_LABELS (the IN list in tuple
    order), carries view_legacy, is MATERIALIZED, closes selection_ctes_sql, and appears in v_run_current alone. unpooled_member is a
    selection state and not a non-current one."""
    assert V.POOLED_PASS_LABELS == ("pooled_t0_t1", "pooled_final_rung")
    assert set(V.POOLED_PASS_LABELS) <= set(V.PASS_LABELS)
    assert not any(set(V.POOLED_PASS_LABELS) & set(ls) for ls in V.MEMBER_PASS_LABELS_BY_SOURCE.values())
    probe = ("EXISTS (SELECT 1 FROM pooled_group pg WHERE pg.lane_source_legacy = r.lane_source_legacy AND pg.model_id = r.model_id"
             " AND pg.object_id = r.object_id AND pg.eval_id = r.eval_id AND pg.view_legacy IS r.view_legacy)")
    assert S.MEMBER_PASS_SQL == S.member_pass_sql() == "(" + S.member_label_sql() + " AND " + probe + ")" == T21.MEMBER_PASS_PREDICATE
    assert S.MEMBER_LABEL_SQL == S.member_label_sql() == T21.MEMBER_LABEL_PREDICATE
    assert S.member_pass_sql("x", "pg_x") == "(" + S.member_label_sql("x") + " AND " + probe.replace("pooled_group pg", "pg_x pg").replace("= r.", "= x.").replace("IS r.", "IS x.") + ")"
    sel, pap = S.SELECTION_STATE_SQL, S.IN_CURRENT_PAPER_SQL
    assert sel.count(S.MEMBER_PASS_SQL) == 1 and sel.count(S.MEMBER_LABEL_SQL) == 2
    member_at, unpooled_at = sel.index(f"WHEN {S.MEMBER_PASS_SQL} THEN 'member_pass'"), sel.index(f"WHEN {S.MEMBER_LABEL_SQL} THEN 'unpooled_member'")
    assert sel.index("WHEN d.action = 'exclude' THEN 'excluded'") < member_at < unpooled_at < sel.index("WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 'not_in_roster'")
    assert pap.count(S.MEMBER_PASS_SQL) == 0 and pap.count(S.MEMBER_LABEL_SQL) == 1 and "pooled_group" not in pap
    assert pap.index("WHEN d.action = 'include' THEN 1") < pap.index(f"WHEN {S.MEMBER_LABEL_SQL} THEN 0") < pap.index("WHEN pm.run_id IS NULL THEN 0")
    cte = S.POOLED_GROUP_CTE
    assert cte.startswith("pooled_group AS MATERIALIZED (") and cte.rstrip().endswith(")")
    assert "SELECT DISTINCT lane_source_legacy, model_id, object_id, eval_id, view_legacy FROM run WHERE pass_label IN ('pooled_t0_t1', 'pooled_final_rung')" in cte
    ctes = S.selection_ctes_sql()
    assert ctes.count(cte) == 1 and ctes.rstrip().endswith(cte.rstrip()) and "later_runs AS MATERIALIZED (SELECT DISTINCT run_id FROM later),\n" + cte in ctes
    squash = lambda t: " ".join(t.split())  # noqa: E731
    rc = squash(S.VIEW_SQL["v_run_current"])
    assert rc.count(squash(S.MEMBER_PASS_SQL)) == 1 and rc.count(squash(S.MEMBER_LABEL_SQL)) == 3 and rc.count(squash(cte)) == 1
    assert rc.count("THEN 'member_pass'") == 1 and rc.count("THEN 'unpooled_member'") == 1
    assert all("pooled_group" not in S.VIEW_SQL[v] and "unpooled_member" not in S.VIEW_SQL[v] for v in S.VIEW_NAMES if v != "v_run_current")
    # the live-shape lifecycle: a pooled run in any state marks -- the CTE reads `run` with no lifecycle, ruling or supersession term
    assert "lifecycle_state" not in cte and "ruling" not in cte and "later" not in cte and "withdrawn" not in cte
    assert "unpooled_member" in V.SELECTION_STATES and "unpooled_member" not in S.NON_CURRENT_SELECTION_STATES
    assert all(f"THEN '{state}'" in sel for state in V.SELECTION_STATES)
    assert set(V.SELECTION_STATES) == {"selected", "excluded", "superseded", "withdrawn", "indeterminate", "not_in_roster", "member_pass", "unpooled_member",
                                       "source_local_anchor"}      # + the source-local none (migration 0027; tests/test_registry_source_local_anchor.py)
