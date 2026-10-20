"""The shared anchor-currency helper (`src/sisteer/registry/adapters/anchor_current.py`): eligibility
(e1 current, e2 complete, e3 one usable reading under the row's judge, e4 items where a family pairs over them), the own-anchor rule
(kept verbatim when current and usable, whatever the pair; otherwise `own_reason` names why and the search CONTINUES), the exact-key
tiers (one same-standup run; same source; the paper's roster-matched in-paper baseline of another source) with the roster-first /
earliest-complete tie-break, the IS-equality of the key with decode compared only when both sides carry one, the named refusals, the
byte-stable timestamp-free `selection_json`, the loader's refusal on a database without `v_run_current`, the readings the frame
COUNTS (a `run_metric` row whose observation key a revision withdrew is no reading though its run stays current -- the paper frame's own
predicate, `schema_def.WITHDRAWN_METRIC_CTE`, in the frame and through the real fold), and the context's one load per pass. The frame is an
in-memory registry with a TABLE `v_run_current` (the sidecar's shape, which the live `_conn()` pattern of
`test_registry_em_misalign_paired_allmodels.py` also uses), plus real registries so the loader's SQL is pinned against the schema and the fold.
`withdrawn_reading_world` / `withdraw_twin` are shared with `test_registry_cluster_read.py`, which exports the same world to a sidecar.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import replace
from typing import Dict, Optional

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters.base import AdapterContext

MODEL, EVAL, INST, CAP, TRIAL = "model_a", "static_x", "inst:a", 4096, 0
METRIC, THR, JUDGE = "asr", 0.5, "judge:x"
SRC, OTHER_SRC = "src_a", "src_b"
STANDUP = {"view_legacy": "tagA", "lane_source_legacy": SRC}

DDL = """
CREATE TABLE run (run_id TEXT PRIMARY KEY, model_id TEXT, object_id TEXT, eval_id TEXT, instrument_id TEXT, token_cap_tokens INTEGER,
                  trial_index INTEGER, pass_label TEXT, decode_policy_id TEXT, lane_source_legacy TEXT, view_legacy TEXT, cluster_job_id TEXT,
                  producer_run_key_legacy TEXT, lifecycle_state TEXT, generated_utc TEXT, queued_utc TEXT);
CREATE TABLE v_run_current (run_id TEXT PRIMARY KEY, is_current INTEGER, why TEXT, held_out INTEGER, in_current_paper INTEGER,
                            selection_state TEXT, roster_entry_ids_json TEXT);
CREATE TABLE judgment (judgment_id TEXT PRIMARY KEY, run_id TEXT, judge_id TEXT, status TEXT, superseded_by_judgment_id TEXT);
CREATE TABLE run_metric (run_id TEXT, judgment_id TEXT, metric_id TEXT, threshold REAL, computation_id TEXT, k INTEGER, n INTEGER,
                         value REAL, ci_lo REAL, ci_hi REAL, computed_by TEXT, source_uri TEXT);
"""


class Frame:
    """An in-memory frame in the sidecar's shape; `run()` adds an unsteered run with its currency row and, by default, one complete
    judgment under JUDGE carrying one reading at (METRIC, THR)."""

    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(DDL)

    def run(self, rid: str, *, model: str = MODEL, eval_id: str = EVAL, inst: str = INST, cap: Optional[int] = CAP, trial: Optional[int] = TRIAL,
            pass_label: Optional[str] = None, decode: Optional[str] = None, source: Optional[str] = SRC, standup: str = "tagA", job: Optional[str] = None,
            key: Optional[str] = None, lifecycle: str = "ingested", utc: Optional[str] = "2026-09-01T00:00:00Z", queued: Optional[str] = "2026-08-01T00:00:00Z",
            object_id: str = "none",
            current: int = 1, why: Optional[str] = None, held_out: int = 0, in_paper: int = 1, roster: bool = True, selection: Optional[str] = None,
            judge: Optional[str] = JUDGE, k: int = 10, n: int = 100, value: float = 10.0, computation: str = "", reading: bool = True) -> str:
        self.con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (rid, model, object_id, eval_id, inst, cap, trial, pass_label, decode, source, standup, job, key or f"key/{rid}", lifecycle,
                          utc, queued))
        sel = selection or ("selected" if in_paper else ("superseded" if not current else "not_in_roster"))
        self.con.execute("INSERT INTO v_run_current VALUES (?,?,?,?,?,?,?)",
                         (rid, current, why, held_out, in_paper, sel, (json.dumps([f"re:{rid}"]) if roster else "[]")))
        if reading:
            jid = self.judgment(rid, judge=judge) if judge is not None else AC.NO_JUDGMENT
            self.metric(rid, jid, k=k, n=n, value=value, computation=computation)
        return rid

    def judgment(self, rid: str, jid: Optional[str] = None, *, judge: Optional[str] = JUDGE, status: str = "complete", superseded_by: Optional[str] = None) -> str:
        jid = jid or f"j:{rid}:{judge}:{1 + self.con.execute('SELECT COUNT(*) FROM judgment WHERE run_id = ?', (rid,)).fetchone()[0]}"
        self.con.execute("INSERT INTO judgment VALUES (?,?,?,?,?)", (jid, rid, judge, status, superseded_by))
        return jid

    def metric(self, rid: str, jid: str, *, metric: str = METRIC, thr: float = THR, computation: str = "", k: Optional[int] = 10, n: Optional[int] = 100,
               value: Optional[float] = 10.0, ci: tuple = (5.0, 15.0), computed_by: str = "producer", source_uri: Optional[str] = None) -> None:
        self.con.execute("INSERT INTO run_metric VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (rid, jid, metric, thr, computation, k, n, value, ci[0], ci[1], computed_by, source_uri or f"file:///{rid}"))

    def steered(self, rid: str, **kw) -> str:
        return self.run(rid, object_id="obj:1", **kw)

    def observe(self, rid: str, jid: Optional[str], *, metric: str = METRIC, thr: float = THR, computation: str = "", table: str = "run_metric",
                revision: str = "rev1", current: int = 0, source: str = "src:g") -> None:
        """One `observation` row (the live table's key columns; created on first use, so a frame without one is the sidecar's shape) for the
        reading's key -- or the RUN's key when `table` is 'run'. `current=0` under every revision naming the key is what withdraws it."""
        if not self.con.execute("SELECT 1 FROM sqlite_master WHERE name = 'observation'").fetchone():
            self.con.execute("CREATE TABLE observation (source_id TEXT, revision_id TEXT, observation_key TEXT, table_name TEXT, current INTEGER)")
        k: Dict[str, object] = {"run_id": rid} if table == "run" else {"run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": thr,
                                                                       "computation_id": computation}
        self.con.execute("INSERT INTO observation VALUES (?,?,?,?,?)", (source, revision, I.canonical_json(k), table, current))

    def item_objects(self, *, conflicts: bool = True, flags: bool = True) -> None:
        """The grading host's item objects, created once: `v_item_result_current` over an `item_result` table with the columns the item digest reads
        (`AC.ITEM_COLUMNS`) and -- unless `conflicts=False` builds the half-defined shape -- a `v_item_conflict` TABLE in the real relation's
        column shape (`schema_def`'s `v_item_conflict`); a frame without them is a shard's shape. With `flags` (the default) also
        `v_item_flags_current` over an `item_flags` table with the columns the cap-evidence digest reads (`AC.FLAG_COLUMNS`); a frame with the
        item objects and no flags object is the shape a `needs_flags` comparison refuses by name."""
        if not self.con.execute("SELECT 1 FROM sqlite_master WHERE name = 'item_result'").fetchone():
            self.con.executescript("""
                CREATE TABLE item_result (run_id TEXT, judgment_id TEXT, metric_id TEXT, item_id TEXT, trial_index INTEGER, sample_index INTEGER,
                                          turn_index INTEGER, value REAL, label TEXT);
                CREATE VIEW v_item_result_current AS SELECT * FROM item_result;
            """)
        if conflicts and not self.con.execute("SELECT 1 FROM sqlite_master WHERE name = 'v_item_conflict'").fetchone():
            self.con.executescript("""
                CREATE TABLE v_item_conflict (run_id TEXT, item_id TEXT, metric_id TEXT, trial_index INTEGER, sample_index INTEGER, turn_index INTEGER,
                                              judgment_id TEXT, n_provenances INTEGER, n_distinct_values INTEGER, value_min REAL, value_max REAL,
                                              revision_ids_json TEXT);
            """)
        if flags and not self.con.execute("SELECT 1 FROM sqlite_master WHERE name = 'item_flags'").fetchone():
            self.con.executescript("""
                CREATE TABLE item_flags (run_id TEXT, judgment_id TEXT, item_id TEXT, trial_index INTEGER, sample_index INTEGER, turn_index INTEGER,
                                         at_token_cap INTEGER);
                CREATE VIEW v_item_flags_current AS SELECT * FROM item_flags;
            """)

    def items(self, rid: str, jid: str, rows, *, metric: str = METRIC, conflicts: bool = True) -> None:
        """Current item rows of `(rid, jid)` at `metric` -- the row `v_item_result_current` selects per key (`item_objects`). `rows` =
        [(item_id, value), ...] at trial 0, sample 0, turn -1, no label."""
        self.item_objects(conflicts=conflicts)
        for item_id, value in rows:
            self.con.execute("INSERT INTO item_result VALUES (?,?,?,?,?,?,?,?,?)", (rid, jid, metric, item_id, 0, 0, -1, value, None))

    def flags(self, rid: str, jid: str, rows) -> None:
        """Current `item_flags` rows of `(rid, jid)` -- the row `v_item_flags_current` selects per key. `rows` = [(item_id, at_token_cap), ...]
        at trial 0, sample 0, turn -1 (`at_token_cap` 0 / 1 / None)."""
        self.item_objects()
        for item_id, cap in rows:
            self.con.execute("INSERT INTO item_flags VALUES (?,?,?,?,?,?,?)", (rid, jid, item_id, 0, 0, -1, cap))

    def conflict(self, rid: str, jid: str, item_id: str, values=(1.0, 0.0), *, metric: str = METRIC, revisions=("rev:a", "rev:b")) -> None:
        """One `v_item_conflict` row: the key `(rid, item_id, metric, 0, 0, -1, jid)` carries `len(values)` visible provenances with
        `len(set(values))` distinct values -- a second accepted provenance that disagrees with the selected row, which `items()` leaves as it is."""
        self.item_objects()
        self.con.execute("INSERT INTO v_item_conflict VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (rid, item_id, metric, 0, 0, -1, jid, len(values), len(set(values)), min(values), max(values), json.dumps(list(revisions))))

    def load(self) -> AC.AnchorCurrency:
        return AC.AnchorCurrency.load(self.con)


def key(**over) -> AC.AnchorKey:
    base = dict(model_id=MODEL, eval_id=EVAL, instrument_id=INST, token_cap_tokens=CAP, trial_index=TRIAL, pass_label=None, decode_policy_id=None,
                lane_source_legacy=SRC, standup=dict(STANDUP), own_anchor_run_id=None, own_anchor_kind=None)
    base.update(over)
    return AC.AnchorKey(**base)


def choose(fr: Frame, k: Optional[AC.AnchorKey] = None, **kw) -> AC.AnchorChoice:
    return fr.load().choose(k or key(), METRIC, THR, kw.pop("judge_id", JUDGE), **kw)


def _closed(ch: AC.AnchorChoice) -> None:
    assert ch.state in AC.STATES and ch.own_reason in AC.OWN_REASONS
    assert ch.anchor_kind is None or ch.anchor_kind in V.ANCHOR_KINDS
    assert (ch.run_id is None) == (ch.state == "no_current_anchor") and ((ch.measurement is None) == (ch.run_id is None))
    assert ch.n_candidates == len(ch.candidates)
    assert list(ch.rejected) == sorted(ch.rejected)
    assert ch.anchor_judgment_id == (ch.measurement.judgment_id if ch.measurement is not None else "")


# ============================================================================================
# step 0: the own anchor
# ============================================================================================

def test_an_own_anchor_that_is_current_and_usable_is_kept_verbatim_whatever_the_pair():
    """The ladder readers' exit-cap pairs: the own none at cap 2,048 anchors a 4,096 row even though a same-key none exists."""
    fr = Frame()
    own = fr.run("none:own", cap=2048, k=3, n=50, value=6.0, computation="c:own")
    same_key = fr.run("none:same", standup="tagB")
    ch = choose(fr, key(own_anchor_run_id=own, own_anchor_kind="same_standup_none"))
    _closed(ch)
    assert ch.state == "own_current" and ch.own_reason == "current" and ch.run_id == own and ch.anchor_kind == "same_standup_none"
    assert ch.cross_source is False and ch.why == "" and ch.own_why == ""
    assert ch.measurement is not None and (ch.measurement.k, ch.measurement.n, ch.measurement.value) == (3.0, 50, 6.0)
    assert ch.measurement.computation_id == "c:own" and ch.measurement.judge_id == JUDGE and ch.measurement.judgment_status == "complete"
    assert ch.measurement.ci_lo == 5.0 and ch.measurement.ci_hi == 15.0 and ch.measurement.computed_by == "producer" and ch.measurement.source_uri == "file:///none:own"
    assert ch.anchor_judgment_id == ch.measurement.judgment_id == f"j:{own}:{JUDGE}:1"
    # the same-key pool is still recorded (visible beside the cross-cap pairing), never used
    assert [c.run_id for c in ch.candidates] == [same_key] and ch.n_candidates == 1 and ch.rejected == ()
    # the own label is kept whatever it is (r2's second kind included); absent -> same_standup_none
    assert choose(fr, key(own_anchor_run_id=own, own_anchor_kind="full_dose")).anchor_kind == "full_dose"
    assert choose(fr, key(own_anchor_run_id=own)).anchor_kind == "same_standup_none"


def test_an_own_anchor_at_the_exact_key_is_both_the_pick_and_a_recorded_candidate():
    fr = Frame()
    own = fr.run("none:own")
    ch = choose(fr, key(own_anchor_run_id=own, own_anchor_kind="same_standup_none"))
    assert ch.state == "own_current" and [c.run_id for c in ch.candidates] == [own] and ch.candidates[0].same_standup and ch.candidates[0].same_source


@pytest.mark.parametrize("why_text, reason", [
    ("held_out (ruling r2v4-hold-out)", "held_out"),
    ("selection_state excluded (ruling r2v4-exclude)", "excluded"),
    ("selection_state superseded (a later run at the same key)", "superseded"),
    ("selection_state superseded (ruling r1)", "superseded"),
    ("superseded_by_run_id other", "superseded"),
    ("decisive ruling exclude (ruling r3)", "excluded"),
    ("decisive ruling supersede (ruling r4)", "superseded"),
    ("selection_state withdrawn", "withdrawn"),
    ("lifecycle_state failed", "lifecycle"),
])
def test_a_retired_own_anchor_names_why_and_the_search_continues(why_text, reason):
    """Strict: a held-out own none is NOT kept -- e1 makes it ineligible like any retired run; the fallback runs."""
    fr = Frame()
    own = fr.run("none:own", current=0, why=why_text, held_out=int(reason == "held_out"), in_paper=0, roster=False)
    sib = fr.run("none:sib", standup="tagB")
    ch = choose(fr, key(own_anchor_run_id=own, own_anchor_kind="same_standup_none"))
    _closed(ch)
    assert ch.state == "sibling_same_source" and ch.own_reason == reason and ch.run_id == sib and ch.anchor_kind == "sibling_none"
    assert ch.why == why_text == ch.own_why and ch.cross_source is False
    assert ch.rejected == ((own, why_text),) and [c.run_id for c in ch.candidates] == [sib]
    # with nothing at the key: no contrast, the refusal names the own anchor's reason and the nearest miss
    fr2 = Frame()
    own2 = fr2.run("none:own", current=0, why=why_text, held_out=int(reason == "held_out"), in_paper=0, roster=False)
    ch2 = choose(fr2, key(own_anchor_run_id=own2, own_anchor_kind="same_standup_none"))
    _closed(ch2)
    assert ch2.state == "no_current_anchor" and ch2.own_reason == reason and ch2.run_id is None and ch2.anchor_kind is None
    assert f"own anchor {own2}: {why_text}" in ch2.why and own2 in dict(ch2.rejected) and ch2.n_candidates == 0


def test_an_own_id_the_registry_never_folded_is_not_in_frame_and_the_search_continues():
    """A composed-but-unfolded own id (petri_stats naming a refused none row) is a reason not to use THAT id, never a dependency block."""
    fr = Frame()
    sib = fr.run("none:sib", standup="tagB")
    ch = choose(fr, key(own_anchor_run_id="composed:never:folded", own_anchor_kind="same_standup_none"))
    _closed(ch)
    assert ch.state == "sibling_same_source" and ch.own_reason == "not_in_frame" and ch.run_id == sib and ch.anchor_kind == "sibling_none"
    assert "not a run the registry has folded" in ch.own_why and ch.why == ch.own_why
    # the paper's roster-matched in-paper baseline of ANOTHER source when the own source has none at the key
    fr2 = Frame()
    ref = fr2.run("none:ref", source=OTHER_SRC, standup="other")
    ch2 = choose(fr2, key(own_anchor_run_id="composed:never:folded"))
    _closed(ch2)
    assert ch2.state == "borrowed_reference" and ch2.own_reason == "not_in_frame" and ch2.run_id == ref and ch2.anchor_kind == "borrowed" and ch2.cross_source is True
    # nothing at the key at all
    ch3 = choose(Frame(), key(own_anchor_run_id="composed:never:folded"))
    _closed(ch3)
    assert ch3.state == "no_current_anchor" and ch3.own_reason == "not_in_frame"
    assert "composed:never:folded is not a run the registry has folded" in ch3.why and f"no unsteered run of {MODEL} / {EVAL} in the frame at any cap" in ch3.why


def test_no_own_anchor_declared_runs_the_search_too():
    """arditi_vendor_mt without a `none` entry, sampled_static without a same-tag none: `choose(own=None)`, never a silent skip."""
    fr = Frame()
    sib = fr.run("none:sib", standup="tagB")
    ch = choose(fr, key())
    _closed(ch)
    assert ch.state == "sibling_same_source" and ch.own_reason == "none_declared" and ch.run_id == sib
    assert ch.selection_json()["own_anchor_run_id"] is None and "declares no anchor" in ch.own_why


def test_an_own_id_that_is_a_current_steered_run_is_not_an_unsteered_anchor():
    fr = Frame()
    st = fr.steered("steered:1")
    sib = fr.run("none:sib", standup="tagB")
    ch = choose(fr, key(own_anchor_run_id=st, own_anchor_kind="same_standup_none"))
    _closed(ch)
    assert ch.state == "sibling_same_source" and ch.own_reason == "not_unsteered" and ch.run_id == sib and "not an unsteered run" in ch.own_why


@pytest.mark.parametrize("case", ["lifecycle", "no_judgment", "ambiguous_judgments", "no_reading", "ambiguous_readings"])
def test_an_own_anchor_that_is_current_but_unusable_names_the_eligibility_rule_it_fails(case):
    fr = Frame()
    if case == "lifecycle":
        own = fr.run("none:own", lifecycle="generated", reading=False)
        want = ("lifecycle", "lifecycle_state generated (not complete)")
    elif case == "no_judgment":
        own = fr.run("none:own", reading=False)
        want = ("no_reading", f"no complete judgment under judge {JUDGE}")
    elif case == "ambiguous_judgments":
        own = fr.run("none:own")
        fr.judgment(own, "j:second")
        want = ("ambiguous", f"2 complete judgments under judge {JUDGE} (j:none:own:judge:x:1, j:second): ambiguous")
    elif case == "no_reading":
        own = fr.run("none:own", reading=False)
        jid = fr.judgment(own)
        fr.metric(own, jid, thr=0.75)
        want = ("no_reading", "no reading at (asr, 0.5) under judgment j:none:own:judge:x:1")
    else:
        own = fr.run("none:own")
        fr.metric(own, f"j:{own}:{JUDGE}:1", computation="c:2", value=11.0)
        want = ("ambiguous", "2 readings at (asr, 0.5) under judgment j:none:own:judge:x:1 with different values")
    sib = fr.run("none:sib", standup="tagB")
    ch = choose(fr, key(own_anchor_run_id=own, own_anchor_kind="same_standup_none"))
    _closed(ch)
    assert ch.state == "sibling_same_source" and ch.own_reason == want[0] and ch.run_id == sib
    assert ch.own_why.startswith(want[1]), ch.own_why
    assert dict(ch.rejected)[own] == ch.own_why


# ============================================================================================
# eligibility of a candidate
# ============================================================================================

def test_a_queued_or_running_or_generated_run_is_not_eligible():
    fr = Frame()
    for i, st in enumerate(("queued", "running", "generated")):
        fr.run(f"none:{st}", standup=f"tag{i}", lifecycle=st, reading=False)
    ch = choose(fr, key())
    _closed(ch)
    assert ch.state == "no_current_anchor" and ch.n_candidates == 0
    assert dict(ch.rejected) == {f"none:{st}": f"lifecycle_state {st} (not complete)" for st in ("queued", "running", "generated")}
    assert ch.why.startswith("the unsteered run(s) at the key are not eligible: ")


def test_a_run_that_is_not_current_is_rejected_with_the_folds_why():
    fr = Frame()
    fr.run("none:old", standup="tagB", current=0, why="selection_state superseded (a later run at the same key)", in_paper=0, roster=False)
    fr.run("none:held", standup="tagC", current=0, why="held_out (ruling r)", held_out=1)
    ch = choose(fr, key())
    _closed(ch)
    assert ch.state == "no_current_anchor"
    assert dict(ch.rejected) == {"none:old": "selection_state superseded (a later run at the same key)", "none:held": "held_out (ruling r)"}


def test_two_complete_judgments_under_one_judge_are_ambiguous_but_a_superseded_one_does_not_count():
    fr = Frame()
    a = fr.run("none:a", standup="tagB")
    fr.judgment(a, "j:a:second")
    ch = choose(fr, key())
    assert ch.state == "no_current_anchor" and dict(ch.rejected)[a] == f"2 complete judgments under judge {JUDGE} (j:a:second, j:none:a:judge:x:1): ambiguous"
    # the second judgment superseded by the first -> one usable judgment -> eligible, the reading under the surviving one
    fr.con.execute("UPDATE judgment SET superseded_by_judgment_id = ? WHERE judgment_id = 'j:a:second'", (f"j:{a}:{JUDGE}:1",))
    ch2 = choose(fr, key())
    _closed(ch2)
    assert ch2.state == "sibling_same_source" and ch2.run_id == a and ch2.anchor_judgment_id == f"j:{a}:{JUDGE}:1"
    # a judgment under ANOTHER judge is not this row's judge: no reading under JUDGE -> rejected by name
    fr2 = Frame()
    b = fr2.run("none:b", standup="tagB", reading=False)
    fr2.metric(b, fr2.judgment(b, judge="judge:other"))
    ch3 = choose(fr2, key())
    assert ch3.state == "no_current_anchor" and dict(ch3.rejected)[b] == f"no complete judgment under judge {JUDGE}"


def test_two_readings_at_one_identity_refuse_unless_the_steered_rows_computation_matches_one():
    """346 live unsteered (run, metric, threshold) groups carry several rows; six petri_score groups carry DIFFERENT values."""
    fr = Frame()
    a = fr.run("none:a", standup="tagB", computation="", value=10.0)
    jid = f"j:{a}:{JUDGE}:1"
    fr.metric(a, jid, computation="c:registry", value=12.0, k=12, computed_by="registry")
    ch = choose(fr, key())
    assert ch.state == "no_current_anchor"
    assert dict(ch.rejected)[a] == ("2 readings at (asr, 0.5) under judgment j:none:a:judge:x:1 with different values (computation none: value 10.0 k 10.0 n 100; "
                                    "computation c:registry: value 12.0 k 12.0 n 100): ambiguous")       # rows in computation-id order
    # the steered row's computation id picks its twin
    ch2 = choose(fr, key(), computation_id="c:registry")
    _closed(ch2)
    assert ch2.state == "sibling_same_source" and ch2.measurement.computation_id == "c:registry" and ch2.measurement.value == 12.0 and ch2.measurement.computed_by == "registry"
    ch3 = choose(fr, key(), computation_id="")
    assert ch3.state == "no_current_anchor"      # the sentinel '' is no computation: nothing to match, the two rows still differ
    # a computation id no row carries falls back to the remaining rows (still ambiguous here)
    assert choose(fr, key(), computation_id="c:unknown").state == "no_current_anchor"
    # EQUAL readings twice (a producer twin) are one reading: eligible, the first by computation id
    fr2 = Frame()
    b = fr2.run("none:b", standup="tagB", computation="c:z", value=10.0)
    fr2.metric(b, f"j:{b}:{JUDGE}:1", computation="c:a", value=10.0)
    ch4 = choose(fr2, key())
    _closed(ch4)
    assert ch4.state == "sibling_same_source" and ch4.measurement.computation_id == "c:a"


def test_a_missing_reading_at_the_threshold_refuses_by_name():
    fr = Frame()
    a = fr.run("none:a", standup="tagB", reading=False)
    fr.metric(a, fr.judgment(a), thr=0.75)
    ch = choose(fr, key())
    assert ch.state == "no_current_anchor" and dict(ch.rejected)[a] == f"no reading at (asr, 0.5) under judgment j:{a}:{JUDGE}:1"
    assert fr.load().choose(key(), METRIC, 0.75, JUDGE).state == "sibling_same_source"


def test_a_rule_scored_row_reads_the_readings_filed_without_a_judgment():
    """bcbw: `judge_id` NULL on the steered row; the anchor's reading is the `run_metric` row at the '' judgment sentinel."""
    fr = Frame()
    a = fr.run("none:a", standup="tagB", judge=None)
    ch = choose(fr, key(), judge_id=None)
    _closed(ch)
    assert ch.state == "sibling_same_source" and ch.anchor_judgment_id == "" and ch.measurement.judgment_id == "" and ch.measurement.judge_id is None
    assert ch.measurement.judgment_status == "none"
    # the same run asked under a judge has no judgment under it
    assert dict(choose(fr, key()).rejected)[a] == f"no complete judgment under judge {JUDGE}"


def test_items_are_required_only_when_a_family_asks_and_are_unknown_off_the_grading_host():
    fr = Frame()
    a = fr.run("none:a", standup="tagB")
    b = fr.run("none:b", standup="tagC", utc="2026-09-02T00:00:00Z")
    assert fr.load().items_known is False
    assert choose(fr, key()).state == "sibling_same_source"                      # e4 is not asked
    ch = choose(fr, key(), needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    assert ch.state == "no_current_anchor" and all("item presence is not known to this frame" in w for w in dict(ch.rejected).values())
    assert all("defines neither v_item_result_current nor v_item_conflict" in w for w in dict(ch.rejected).values())
    assert fr.load().items_unknown_why == "this database defines neither v_item_result_current nor v_item_conflict"
    # a frame that knows items (the grading host: v_item_result_current): presence per (run, judgment) at the metric decides
    fr.items(b, f"j:{b}:{JUDGE}:1", [("item:1", 1.0)])
    fr.items(a, f"j:{a}:{JUDGE}:1", [("item:1", 1.0)], metric="other_metric")
    ac = fr.load()
    assert ac.items_known is True and ac.items_at(METRIC) == {(b, f"j:{b}:{JUDGE}:1")} and ac.items_at("other_metric") == {(a, f"j:{a}:{JUDGE}:1")}
    ch2 = ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    _closed(ch2)
    assert ch2.state == "sibling_same_source" and ch2.run_id == b and dict(ch2.rejected)[a] == f"no current item rows at {METRIC} under judgment j:{a}:{JUDGE}:1"
    assert ac.choose(key(), METRIC, THR, JUDGE).run_id == a                      # without e4 the earliest wins


# ============================================================================================
# the readings the frame counts: a withdrawn observation
# ============================================================================================

def test_a_reading_whose_observation_was_withdrawn_is_no_reading_though_its_run_stays_current():
    """A revision can withdraw an anchor's metric row and keep its unsteered run: the run is current (e1 passes), the reading is not the
    frame's (e3 refuses `no_reading`) -- for the exact-key pool and for the adapter's own id alike."""
    fr = Frame()
    a = fr.run("none:a")
    assert choose(fr).run_id == a
    fr.observe(a, f"j:{a}:{JUDGE}:1", current=0)
    ac = fr.load()
    assert ac.is_current(a) is True and ac.counts["metric_rows"] == 0 and ac.counts["withdrawn_metric_rows"] == 1
    ch = ac.choose(key(), METRIC, THR, JUDGE)
    _closed(ch)
    assert ch.state == "no_current_anchor" and ch.rejected == ((a, f"no reading at ({METRIC}, {THR:g}) under judgment j:{a}:{JUDGE}:1"),)
    own = ac.choose(key(own_anchor_run_id=a, own_anchor_kind="same_standup_none"), METRIC, THR, JUDGE)
    _closed(own)
    assert own.state == "no_current_anchor" and own.own_reason == "no_reading" and own.own_why.startswith("no reading at")
    assert own.selection_json()["reading"] is None and own.selection_json()["rejected"] == [[a, own.own_why]]


def test_a_withdrawn_computed_twin_no_longer_makes_the_producer_reading_ambiguous():
    """Two rows at one identity with different values refuse as ambiguous while both are the frame's; once a revision withdraws the computed
    twin's key the producer's row is the ONE reading and the run anchors with it."""
    fr = Frame()
    a = fr.run("none:a")                                          # the producer's row: computation '', value 10
    jid = f"j:{a}:{JUDGE}:1"
    fr.metric(a, jid, computation="c:twin", value=12.0, k=12, computed_by="registry")
    before = choose(fr)
    assert before.state == "no_current_anchor" and before.rejected[0][0] == a
    assert "2 readings at" in before.rejected[0][1] and "ambiguous" in before.rejected[0][1]
    fr.observe(a, jid, computation="c:twin", current=0)           # the twin's key is current under no revision; the producer's row was never observed
    after = choose(fr)
    _closed(after)
    assert after.state == "sibling_same_source" and after.run_id == a and after.rejected == ()
    assert after.measurement is not None and after.measurement.computation_id == "" and after.measurement.value == 10.0 and after.measurement.computed_by == "producer"
    assert fr.load().counts == {"runs": 1, "unsteered": 1, "judgments": 1, "metric_rows": 1, "withdrawn_metric_rows": 1, "non_current_judgment_rows": 0, "items_known": 0, "flags_known": 0}


def test_the_predicate_is_current_under_no_revision_and_a_never_observed_row_is_kept():
    """The paper frame's rule verbatim: a key observed by a withdrawn revision AND by a current one is current; a key no revision observed
    has no observation and is kept; an observation of the RUN's key is not a metric observation."""
    fr = Frame()
    a = fr.run("none:a")
    b = fr.run("none:b", standup="tagB")
    ja = f"j:{a}:{JUDGE}:1"
    fr.observe(a, ja, revision="rev1", current=0)
    fr.observe(a, ja, revision="rev2", current=1)
    fr.observe(a, None, table="run", revision="rev1", current=0)
    ac = fr.load()
    assert ac.counts["metric_rows"] == 2 and ac.counts["withdrawn_metric_rows"] == 0
    ch = ac.choose(key(), METRIC, THR, JUDGE)
    _closed(ch)
    assert ch.run_id == a and [c.run_id for c in ch.candidates] == [a, b] and ch.rejected == ()
    # a frame without an `observation` table (the sidecar's shape) evaluates no predicate and counts nothing withdrawn
    assert Frame().load().counts["withdrawn_metric_rows"] == 0


def test_the_loaders_predicate_is_the_paper_frames_own_text(monkeypatch):
    """`withdrawn_metric_cte` IS `schema_def.WITHDRAWN_METRIC_CTE` (qualified only in its `observation` read, for the export); the join is
    schema_def's; a constant whose text no longer carries the read refuses by name instead of qualifying nothing."""
    assert AC.withdrawn_metric_cte() == S.WITHDRAWN_METRIC_CTE and AC.WITHDRAWN_METRIC_JOIN == S._WITHDRAWN_JOIN
    live = AC.withdrawn_metric_cte("live")
    assert live.count("FROM live.observation WHERE") == 1 and "FROM observation WHERE" not in live
    assert live.replace("FROM live.observation WHERE", "FROM observation WHERE") == S.WITHDRAWN_METRIC_CTE
    assert "LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id" in AC.WITHDRAWN_METRIC_JOIN and "wm.computation_id = rm.computation_id" in AC.WITHDRAWN_METRIC_JOIN
    assert "HAVING MAX(current) = 0" in AC.withdrawn_metric_cte() and "table_name = 'run_metric'" in AC.withdrawn_metric_cte()
    monkeypatch.setattr(S, "WITHDRAWN_METRIC_CTE", S.WITHDRAWN_METRIC_CTE.replace("FROM observation WHERE", "FROM observation ob WHERE"))
    with pytest.raises(RuntimeError, match=r"reads 'FROM observation WHERE' 0 times, not once"):
        AC.withdrawn_metric_cte("live")
    assert AC.withdrawn_metric_cte() == S.WITHDRAWN_METRIC_CTE          # unqualified: the constant as it stands


# ---- through the fold: a synthetic source's revisions withdraw the anchor's readings, the run stays current ------------------

WORLD_SRC = "src:g"
WORLD_ADAPTER = "synthetic"
TWIN_COMPUTATION = I.computation_id("synthetic.twin", "1", "none")


def _rec_rev(rev: str, i: int):
    return synth.rec("source_revision", {"revision_id": rev, "source_id": WORLD_SRC, "adapter": WORLD_ADAPTER, "observed_utc": synth.t(i),
                                         "inputs_json": {"a": "%064x" % i}, "accepted": 1}, synth.t(i))


def _rec_obs(rev: str, keys, i: int, table: str = "run"):
    return synth.rec("observation", {"source_id": WORLD_SRC, "revision_id": rev, "table_name": table,
                                     "observation_keys": [I.canonical_json(k) for k in keys]}, synth.t(i))


def _rec_withdrawal(k, frm: str, to: str, i: int):
    return synth.rec("withdrawal", {"source_id": WORLD_SRC, "observation_key": I.canonical_json(k), "from_revision_id": frm, "to_revision_id": to,
                                    "table_name": "run_metric", "reason": "removed_measurement"}, synth.t(i))


def _ingest_world(root: str) -> None:
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refused, rep.blocked, rep.refusals)


def withdrawn_reading_world(root: str) -> Dict[str, object]:
    """A real registry (the fold, `v_run_current`, `observation`): one unsteered run and one steered run, a contrast at (asr, 0.5) so the
    loader reads the metric, and TWO readings of the unsteered run at that identity -- the producer's row (value 0.25, computation '') and a
    computed twin (value 0.5) -- both observed by revision rev1 of a synthetic source that also observes the run's key. Returns the ids and
    the anchor key of the unsteered run (its standup is its cluster job). `withdraw_twin` / `withdraw_producer` advance the source."""
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    none, lora = ids["run_ids"]
    j_none, j_lora = ids["judgment_ids"]
    k_prod = {"run_id": none, "judgment_id": j_none, "metric_id": "asr", "threshold": 0.5, "computation_id": ""}
    k_twin = dict(k_prod, computation_id=TWIN_COMPUTATION)
    comp = synth.rec("computation", {"computation_id": TWIN_COMPUTATION, "algorithm": "synthetic twin", "version": "1", "seed_provenance": "none",
                                     "code_sha": synth.CODE_SHA}, synth.t(8990))
    twin = synth.rec("run_metric", {"run_id": none, "judgment_id": j_none, "metric_id": "asr", "threshold": 0.5, "computation_id": TWIN_COMPUTATION,
                                    "value": 0.5, "k": 50, "n": 100, "computed_by": "registry"}, synth.t(8991))
    contrast = synth.rec("run_contrast", {"run_id": lora, "anchor_run_id": none, "judgment_id": j_lora, "metric_id": "asr", "threshold": 0.5,
                                          "anchor_kind": "same_standup_none", "delta": 0.1, "delta_method": "exact McNemar", "n_pairs": 3,
                                          "computed_by": "registry"}, synth.t(8992))
    R.write_records(os.path.join(root, "records", "a", "000001-rev1.jsonl"),
                    [_rec_rev("rev1", 9000), _rec_obs("rev1", [{"run_id": none}], 9001), _rec_obs("rev1", [k_prod, k_twin], 9002, table="run_metric"),
                     comp, twin, contrast])
    _ingest_world(root)
    with DB.reading(DB.db_path(root)) as conn:
        row = dict(conn.execute("SELECT * FROM run WHERE run_id = ?", (none,)).fetchone())
    anchor_key = AC.AnchorKey(model_id=row["model_id"], eval_id=row["eval_id"], instrument_id=row["instrument_id"], token_cap_tokens=row["token_cap_tokens"],
                              trial_index=row["trial_index"], pass_label=row["pass_label"], decode_policy_id=None, lane_source_legacy=row["lane_source_legacy"],
                              standup={"cluster_job_id": row["cluster_job_id"]})
    return {"none": none, "lora": lora, "judgment": j_none, "key": anchor_key, "k_prod": k_prod, "k_twin": k_twin}


def withdraw_twin(root: str, w: Dict[str, object]) -> None:
    """rev2: the run's key and the producer's reading stay observed; the twin's key is withdrawn."""
    R.write_records(os.path.join(root, "records", "a", "000002-rev2.jsonl"),
                    [_rec_rev("rev2", 9010), _rec_obs("rev2", [{"run_id": w["none"]}], 9011), _rec_obs("rev2", [w["k_prod"]], 9011, table="run_metric"),
                     _rec_withdrawal(w["k_twin"], "rev1", "rev2", 9012)])
    _ingest_world(root)


def rejudge_none(root: str, w: Dict[str, object], *, replacement: Optional[float] = None, i: int = 9100) -> str:
    """The unsteered run re-judged under the same judge: its judgment re-filed `superseded` naming a new complete judgment as its
    successor; `replacement` also files the successor's (asr, 0.5) reading (unobserved: a never-observed row is a reading). Returns the
    successor's id (the same id each call, so a later call files under the same successor)."""
    j_new = I.judgment_id(str(w["none"]), synth.JUDGE_ID, "s", synth.t(9050))
    recs = [synth.rec("judgment", {"judgment_id": w["judgment"], "run_id": w["none"], "judge_id": synth.JUDGE_ID, "n_items_judged": 3, "status": "superseded",
                                   "superseded_by_judgment_id": j_new}, synth.t(i)),
            # the successor states the form its id was minted in (main's identity rule: `claimed_by_session` + `claimed_utc` reproduce it)
            synth.rec("judgment", {"judgment_id": j_new, "run_id": w["none"], "judge_id": synth.JUDGE_ID, "n_items_judged": 3, "status": "complete",
                                   "claimed_by_session": "s", "claimed_utc": synth.t(9050),
                                   "started_utc": synth.t(9050), "finished_utc": synth.t(9051), "api_pool": "lowprio"}, synth.t(i + 1))]
    if replacement is not None:
        recs.append(synth.rec("run_metric", {"run_id": w["none"], "judgment_id": j_new, "metric_id": "asr", "threshold": 0.5, "value": replacement,
                                             "k": int(round(100 * replacement)), "n": 100, "computed_by": "producer"}, synth.t(i + 2)))
    R.write_records(os.path.join(root, "records", "c", f"{i:06d}-rejudge.jsonl"), recs)
    _ingest_world(root)
    return j_new


def withdraw_producer(root: str, w: Dict[str, object]) -> None:
    """rev3: the run's key stays observed; the producer's reading is withdrawn too."""
    R.write_records(os.path.join(root, "records", "a", "000003-rev3.jsonl"),
                    [_rec_rev("rev3", 9020), _rec_obs("rev3", [{"run_id": w["none"]}], 9021), _rec_withdrawal(w["k_prod"], "rev2", "rev3", 9022)])
    _ingest_world(root)


def test_through_the_fold_a_revision_that_withdraws_the_anchors_reading_leaves_the_run_current_and_the_reading_unusable(tmp_path):
    """The real registry end to end: `observation` rows from the fold of a synthetic source's revisions, `v_run_current` keeping the run
    current at every step (the run's key stays observed, so no run withdrawal), the loader's readings following the paper frame's predicate
    -- ambiguous while both rows count, the producer's row once the twin is withdrawn, nothing once both are."""
    root = str(tmp_path / "reg")
    w = withdrawn_reading_world(root)
    none, k, jid = str(w["none"]), w["key"], str(w["judgment"])

    def state():
        with DB.reading(DB.db_path(root)) as conn:
            vrc = tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (none,)).fetchone())
            ac = AC.AnchorCurrency.load(conn)
            ch = ac.choose(k, "asr", 0.5, synth.JUDGE_ID)
            own = ac.choose(replace(k, own_anchor_run_id=none, own_anchor_kind="same_standup_none"), "asr", 0.5, synth.JUDGE_ID)
            withdrawn = [str(r[0]) for r in conn.execute(f"WITH {S.WITHDRAWN_METRIC_CTE} SELECT computation_id FROM withdrawn_metric ORDER BY 1")]
            table_rows = conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ?", (none,)).fetchone()[0]
        _closed(ch)
        _closed(own)
        return vrc, ac, ch, own, withdrawn, table_rows

    # rev1: two observed readings with different values at one identity -- ambiguous, refused by name (the table holds both)
    vrc, ac, ch, own, withdrawn, table_rows = state()
    assert vrc == (1, None) and ac.is_current(none) is True and withdrawn == [] and table_rows == 2
    assert (ac.counts["metric_rows"], ac.counts["withdrawn_metric_rows"]) == (2, 0)
    assert ch.state == "no_current_anchor" and ch.rejected[0][0] == none and "2 readings at (asr, 0.5)" in ch.rejected[0][1] and "ambiguous" in ch.rejected[0][1]
    assert own.state == "no_current_anchor" and own.own_reason == "ambiguous"
    # rev2 withdraws the computed twin: the producer's row is the ONE reading, the run anchors with it (the table still holds both rows)
    withdraw_twin(root, w)
    vrc, ac, ch, own, withdrawn, table_rows = state()
    assert vrc == (1, None) and withdrawn == [TWIN_COMPUTATION] and table_rows == 2
    assert (ac.counts["metric_rows"], ac.counts["withdrawn_metric_rows"]) == (1, 1)
    assert ch.state == "sibling_same_source" and ch.anchor_kind == "same_standup_none" and ch.run_id == none and ch.rejected == ()
    assert ch.measurement is not None and ch.measurement.computation_id == "" and ch.measurement.value == 0.25 and ch.measurement.judgment_id == jid
    assert own.state == "own_current" and own.run_id == none and own.measurement == ch.measurement
    # rev3 withdraws the producer's row too: the run is still current, and it anchors nothing -- no reading at the identity
    withdraw_producer(root, w)
    vrc, ac, ch, own, withdrawn, table_rows = state()
    assert vrc == (1, None) and ac.is_current(none) is True and withdrawn == ["", TWIN_COMPUTATION] and table_rows == 2
    assert (ac.counts["metric_rows"], ac.counts["withdrawn_metric_rows"]) == (0, 2)
    assert ch.state == "no_current_anchor" and ch.rejected == ((none, f"no reading at (asr, 0.5) under judgment {jid}"),)
    assert own.state == "no_current_anchor" and own.own_reason == "no_reading" and own.why.startswith(f"own anchor {none}: no reading at (asr, 0.5)")


# ============================================================================================
# the tiers and the tie-break
# ============================================================================================

def test_exactly_one_same_standup_run_is_same_standup_none_and_two_refuse_by_name():
    fr = Frame()
    a = fr.run("none:a")                                             # tagA / src_a: the steered row's own standup
    fr.run("none:other", standup="tagB")                             # same source, another standup: outranked by the standup tier
    ch = choose(fr, key())
    _closed(ch)
    assert ch.state == "sibling_same_source" and ch.anchor_kind == "same_standup_none" and ch.run_id == a and ch.cross_source is False
    assert ch.n_candidates == 2 and ch.candidates[0].same_standup and ch.candidates[0].same_source
    fr.run("none:a2")                                                # a second current unsteered run of the same standup at the key
    ch2 = choose(fr, key())
    _closed(ch2)
    assert ch2.state == "no_current_anchor" and ch2.anchor_kind is None
    assert ch2.why == "2 current unsteered runs of this standup at one key (none:a, none:a2); a paired contrast needs exactly one"
    # a standup declared with a NULL-valued column identifies nothing: no same-standup tier, the same-source tier decides
    ch3 = fr.load().choose(key(standup={"cluster_job_id": None}), METRIC, THR, JUDGE)
    assert ch3.state == "sibling_same_source" and ch3.anchor_kind == "sibling_none"
    assert fr.load().choose(key(standup={}), METRIC, THR, JUDGE).anchor_kind == "sibling_none"


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
def test_three_same_source_candidates_pick_the_earliest_complete_by_utc_then_run_id_whatever_the_insertion_order(order):
    """strength_mtdiff's 42 rows: 11-12 current nones at one key; the pick must not depend on which registered first."""
    rows = [("none:c", "tagC", "2026-09-03T00:00:00Z"), ("none:a", "tagB", "2026-09-01T00:00:00Z"), ("none:b", "tagD", "2026-09-01T00:00:00Z")]
    fr = Frame()
    for i in order:
        rid, standup, utc = rows[i]
        fr.run(rid, standup=standup, utc=utc)
    ch = choose(fr, key())
    _closed(ch)
    assert ch.state == "sibling_same_source" and ch.anchor_kind == "sibling_none" and ch.run_id == "none:a"
    assert [c.run_id for c in ch.candidates] == ["none:a", "none:b", "none:c"] and ch.n_candidates == 3
    assert ch.selection_json()["candidates"] == ["none:a", "none:b", "none:c"]
    # the completion time is COALESCE(generated_utc, queued_utc): a run without generated_utc sorts by its queue time ...
    fr.run("none:q", standup="tagE", utc=None, queued="2026-08-15T00:00:00Z")
    assert [c.run_id for c in choose(fr, key()).candidates][0] == "none:q"
    # ... and one with neither sorts after every dated one
    fr.run("none:z", standup="tagF", utc=None, queued=None)
    assert [c.run_id for c in choose(fr, key()).candidates][-1] == "none:z"


def test_roster_matched_or_in_paper_candidates_come_before_an_earlier_off_roster_one_inside_a_tier():
    """Live, four same-source keys have an earliest candidate that is NOT roster-matched beside one that is."""
    fr = Frame()
    fr.run("none:early", standup="tagB", utc="2026-09-01T00:00:00Z", in_paper=0, roster=False, selection="not_in_roster")
    late = fr.run("none:late", standup="tagC", utc="2026-09-05T00:00:00Z")
    ch = choose(fr, key())
    _closed(ch)
    assert ch.run_id == late and [c.run_id for c in ch.candidates] == [late, "none:early"]
    assert ch.candidates[0].roster_matched == 1 and ch.candidates[1].roster_matched == 0 and ch.candidates[1].in_current_paper == 0
    # an in-paper candidate that is not roster-matched (an include ruling) ranks with the roster-matched ones
    fr2 = Frame()
    fr2.run("none:early", standup="tagB", utc="2026-09-01T00:00:00Z", in_paper=0, roster=False, selection="not_in_roster")
    inc = fr2.run("none:included", standup="tagC", utc="2026-09-05T00:00:00Z", in_paper=1, roster=False)
    assert choose(fr2, key()).run_id == inc


def test_the_papers_roster_matched_in_paper_none_of_another_source_is_borrowed_and_an_off_roster_one_is_nothing():
    fr = Frame()
    off = fr.run("none:off", source=OTHER_SRC, standup="o1", in_paper=0, roster=False, selection="not_in_roster")
    ch = choose(fr, key())
    _closed(ch)
    assert ch.state == "no_current_anchor" and ch.n_candidates == 1 and ch.candidates[0].run_id == off and not ch.candidates[0].same_source
    assert ch.why.startswith(f"the current unsteered run(s) at the key ({off}) are neither of this source nor the paper's roster-matched baseline")
    # in the paper but NOT roster-matched (an include ruling) is not the paper's own baseline either
    fr.con.execute("UPDATE v_run_current SET in_current_paper = 1, selection_state = 'selected' WHERE run_id = ?", (off,))
    assert choose(fr, key()).state == "no_current_anchor"
    fr.con.execute("UPDATE v_run_current SET roster_entry_ids_json = '[\"re:1\"]' WHERE run_id = ?", (off,))
    ch2 = choose(fr, key())
    _closed(ch2)
    assert ch2.state == "borrowed_reference" and ch2.anchor_kind == "borrowed" and ch2.cross_source is True and ch2.run_id == off
    assert ch2.candidates[0].in_current_paper == 1 and ch2.candidates[0].roster_matched == 1
    # a same-source candidate outranks the borrowed one even when it registered later
    sib = fr.run("none:sib", standup="tagB", utc="2026-12-01T00:00:00Z")
    ch3 = choose(fr, key())
    assert ch3.state == "sibling_same_source" and ch3.run_id == sib and ch3.n_candidates == 2


# ============================================================================================
# the exact key
# ============================================================================================

@pytest.mark.parametrize("axis", ["cap", "trial", "pass", "instrument", "model", "eval"])
def test_the_key_is_exact_on_every_axis_and_never_relaxed(axis):
    fr = Frame()
    kw = {"cap": dict(cap=2048), "trial": dict(trial=1), "pass": dict(pass_label="pooled_t0_t1"), "instrument": dict(inst="inst:b"),
          "model": dict(model="model_b"), "eval": dict(eval_id="static_y")}[axis]
    fr.run("none:near", standup="tagB", **kw)
    ch = choose(fr, key())
    _closed(ch)
    assert ch.state == "no_current_anchor" and ch.n_candidates == 0 and ch.rejected == ()
    if axis == "cap":
        assert ch.why == "a current unsteered run exists only at cap 2048"
    elif axis == "instrument":
        assert ch.why == "a current unsteered run at this cap exists only under instrument inst:b"
    elif axis in ("trial", "pass"):
        assert ch.why.startswith(f"no current unsteered run of {MODEL} / {EVAL} at the key")
    else:
        assert ch.why == f"no unsteered run of {MODEL} / {EVAL} in the frame at any cap"


def test_null_cap_trial_and_pass_compare_is():
    fr = Frame()
    a = fr.run("none:nullcap", standup="tagB", cap=None, pass_label=None)
    assert choose(fr, key(token_cap_tokens=None)).run_id == a
    assert choose(fr, key()).state == "no_current_anchor"
    fr2 = Frame()
    p = fr2.run("none:pooled", standup="tagB", pass_label="pooled_t0_t1")
    assert choose(fr2, key(pass_label="pooled_t0_t1")).run_id == p and choose(fr2, key()).state == "no_current_anchor"


def test_decode_policy_is_compared_only_when_both_sides_carry_one():
    """368 of 1,185 live current nones carry no decode_policy_id: a NULL on either side never excludes."""
    fr = Frame()
    none_dp = fr.run("none:nodp", standup="tagB", decode=None)
    fr.run("none:dp2", standup="tagC", decode="dp:2")
    ch = choose(fr, key(decode_policy_id="dp:1"))
    _closed(ch)
    assert ch.state == "sibling_same_source" and [c.run_id for c in ch.candidates] == [none_dp]
    assert set(c.run_id for c in choose(fr, key(decode_policy_id=None)).candidates) == {none_dp, "none:dp2"}
    assert [c.run_id for c in choose(fr, key(decode_policy_id="dp:2")).candidates] == ["none:dp2", none_dp]       # both eligible; by run id


# ============================================================================================
# the loader, the lookups, the context
# ============================================================================================

def test_a_database_without_v_run_current_refuses_by_name_with_its_schema_version():
    con = DB.connect_memory()
    con.executescript("CREATE TABLE schema_migration (version INTEGER, slug TEXT, applied_utc TEXT); INSERT INTO schema_migration VALUES (20, 'x', 'now');"
                      "CREATE TABLE run (run_id TEXT); CREATE TABLE judgment (judgment_id TEXT); CREATE TABLE run_metric (run_id TEXT);")
    with pytest.raises(AC.AnchorCurrencyUnavailable, match=r"v_run_current is not defined in this database \(schema 20\)"):
        AC.AnchorCurrency.load(con)
    # a sidecar-shaped connection names the exported database's version; a bare one says unknown
    con2 = sqlite3.connect(":memory:")
    con2.executescript("CREATE TABLE reader_state_meta (schema TEXT, schema_version INTEGER); INSERT INTO reader_state_meta VALUES ('reader_state/1', 21);")
    with pytest.raises(AC.AnchorCurrencyUnavailable, match=r"\(schema 21\)"):
        AC.AnchorCurrency.load(con2)
    with pytest.raises(AC.AnchorCurrencyUnavailable, match=r"\(schema unknown\)"):
        AC.AnchorCurrency.load(sqlite3.connect(":memory:"))
    # the currency table present but a relation table missing: named too
    con3 = sqlite3.connect(":memory:")
    con3.executescript("CREATE TABLE v_run_current (run_id TEXT); CREATE TABLE run (run_id TEXT);")
    with pytest.raises(AC.AnchorCurrencyUnavailable, match="no judgment table"):
        AC.AnchorCurrency.load(con3)


def test_the_loader_reads_a_real_registry_schema(tmp_path):
    """The loader's SQL against the real schema (the fold of `v_run_current` over an empty registry, every column it names, the
    current-items object): a fresh registry loads, knows no run, knows items, and refuses a key with the nearest-miss text."""
    conn = DB.open_fresh(str(tmp_path / "registry.sqlite"))
    try:
        ac = AC.AnchorCurrency.load(conn)
        assert ac.counts == {"runs": 0, "unsteered": 0, "judgments": 0, "metric_rows": 0, "withdrawn_metric_rows": 0, "non_current_judgment_rows": 0, "items_known": 1, "flags_known": 1} and ac.items_known is True and ac.flags_known is True
        assert ac.is_current("x") is None and ac.why("x") is None and ac.currency("x") is None and ac.unsteered_run("x") is None
        assert ac.items_at(METRIC) == set() and ac.items_conflicts(METRIC, "x", "j") == [] and ac.items_unknown_why == ""
        ch = ac.choose(key(own_anchor_run_id="composed:x"), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
        _closed(ch)
        assert ch.state == "no_current_anchor" and ch.own_reason == "not_in_frame" and f"no unsteered run of {MODEL} / {EVAL} in the frame at any cap" in ch.why
    finally:
        conn.close()


def test_is_current_why_and_the_row_lookups():
    fr = Frame()
    a = fr.run("none:a")
    old = fr.run("none:old", standup="tagB", current=0, why="selection_state excluded (ruling r)", in_paper=0, roster=False)
    ac = fr.load()
    assert ac.is_current(a) is True and ac.why(a) is None and ac.is_current(old) is False and ac.why(old) == "selection_state excluded (ruling r)"
    assert ac.is_current("never") is None and ac.why("never") is None
    assert ac.currency(a) == {"is_current": 1, "why": None, "held_out": 0, "in_current_paper": 1, "selection_state": "selected", "roster_matched": 1}
    assert ac.currency(old)["roster_matched"] == 0 and ac.currency(old)["selection_state"] == "superseded"
    row = ac.unsteered_run(a)
    assert row["model_id"] == MODEL and row["token_cap_tokens"] == CAP and row["lifecycle_state"] == "ingested" and row["complete_utc"] == "2026-09-01T00:00:00Z"
    assert ac.counts == {"runs": 2, "unsteered": 2, "judgments": 2, "metric_rows": 2, "withdrawn_metric_rows": 0, "non_current_judgment_rows": 0, "items_known": 0, "flags_known": 0}
    # EVERY reading of every unsteered run loads, with or without a `run_contrast` table: a metric is an anchor metric the moment a steered row
    # intends a contrast on it, so the set is never read off `run_contrast` (the first contrast at a new metric -- and every contrast of a fresh
    # registry -- would otherwise find no reading and park forever; commit 3 of the re-anchoring sequencing)
    fr.metric(a, f"j:{a}:{JUDGE}:1", metric="never_contrasted", thr=-1.0)
    assert fr.load().counts["metric_rows"] == 3
    fr.con.executescript("CREATE TABLE run_contrast (metric_id TEXT); INSERT INTO run_contrast VALUES ('asr');")
    assert fr.load().counts["metric_rows"] == 3
    assert fr.load().choose(key(own_anchor_run_id=a, own_anchor_kind="same_standup_none"), "never_contrasted", -1.0, JUDGE).state == "own_current"


def test_selection_json_is_timestamp_free_and_byte_stable_across_two_loads():
    fr = Frame()
    own = fr.run("none:own", current=0, why="held_out (ruling r)", held_out=1, in_paper=0, roster=False)
    fr.run("none:b", standup="tagB", utc="2026-09-02T00:00:00Z")
    fr.run("none:a", standup="tagC", utc="2026-09-01T00:00:00Z", computation="c:1")
    fr.run("none:bad", standup="tagD", lifecycle="generated", reading=False)
    k = key(own_anchor_run_id=own, own_anchor_kind="same_standup_none", decode_policy_id="dp:1")
    j1 = fr.load().choose(k, METRIC, THR, JUDGE, computation_id="c:1").selection_json()
    j2 = fr.load().choose(k, METRIC, THR, JUDGE, computation_id="c:1").selection_json()
    assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)
    assert set(j1) == {"state", "own_reason", "own_anchor_run_id", "own_anchor_why", "why", "n_candidates", "candidates", "rejected", "key", "reading", "order", "rule"}
    text = json.dumps(j1)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}", text) and "derived_at" not in text        # no timestamp value anywhere in the block
    assert not any(k_.endswith(("_utc", "_at")) for k_ in j1) and not any(k_.endswith(("_utc", "_at")) for k_ in j1["key"]) \
        and not any(k_.endswith(("_utc", "_at")) for k_ in j1["reading"])
    assert j1["state"] == "sibling_same_source" and j1["own_reason"] == "held_out" and j1["own_anchor_run_id"] == own
    assert j1["own_anchor_why"] == "held_out (ruling r)" == j1["why"] and j1["n_candidates"] == 2 and j1["candidates"] == ["none:a", "none:b"]
    assert j1["rejected"] == [["none:bad", "lifecycle_state generated (not complete)"], [own, "held_out (ruling r)"]]
    assert j1["key"] == {"token_cap_tokens": CAP, "trial_index": TRIAL, "pass_label": None, "decode_policy_id": "dp:1"}
    assert j1["reading"] == {"judgment_id": f"j:none:a:{JUDGE}:1", "computation_id": "c:1", "computed_by": "producer", "source_uri": "file:///none:a"}
    assert j1["order"] == AC.ORDER == "roster-matched first, then earliest complete (generated_utc, run_id)" and j1["rule"] == AC.RULE == "anchor_current/1"
    # a refusal carries no reading
    assert choose(Frame(), key()).selection_json()["reading"] is None


def test_the_context_loads_the_frame_once_per_pass_and_refuses_without_a_connection():
    fr = Frame()
    fr.run("none:a")
    ctx = AdapterContext(root="", now="2026-10-02T00:00:00.000Z", producer={}, conn=fr.con)
    assert ctx.anchors is None
    ac = ctx.anchor_currency()
    assert isinstance(ac, AC.AnchorCurrency) and ctx.anchor_currency() is ac and ctx.anchors is ac
    none_ctx = AdapterContext(root="", now="2026-10-02T00:00:00.000Z", producer={})
    with pytest.raises(AC.AnchorCurrencyUnavailable, match="no registry connection"):
        none_ctx.anchor_currency()
    assert none_ctx.anchors is None
    # a failed load is not cached: the next call asks the connection again
    bad = AdapterContext(root="", now="2026-10-02T00:00:00.000Z", producer={}, conn=sqlite3.connect(":memory:"))
    with pytest.raises(AC.AnchorCurrencyUnavailable):
        bad.anchor_currency()
    bad.conn = fr.con
    assert bad.anchor_currency().counts["unsteered"] == 1
    # the positional construction of the context is unchanged (the new field is last and defaulted)
    assert AdapterContext("", "now", {}, {}, None).anchors is None


def test_the_vocabularies_are_closed_and_an_out_of_vocabulary_choice_cannot_be_built():
    assert set(AC.STATES) == {"own_current", "sibling_same_source", "borrowed_reference", "no_current_anchor"}
    assert "current" in AC.OWN_REASONS and "not_in_frame" in AC.OWN_REASONS and "none_declared" in AC.OWN_REASONS and "held_out" in AC.OWN_REASONS
    assert {AC.KIND_SAME_STANDUP, AC.KIND_SIBLING, AC.KIND_BORROWED} <= set(V.ANCHOR_KINDS)
    fields = dict(own_reason="current", run_id=None, anchor_kind=None, cross_source=False, why="", own_why="", n_candidates=0, candidates=(), rejected=(),
                  anchor_judgment_id="", measurement=None, key=key())
    with pytest.raises(ValueError, match="state"):
        AC.AnchorChoice(state="provisional", **fields)
    with pytest.raises(ValueError, match="own_reason"):
        AC.AnchorChoice(state="no_current_anchor", **dict(fields, own_reason="staged_this_pass"))
    with pytest.raises(ValueError, match="carries run_id"):
        AC.AnchorChoice(state="own_current", **fields)                       # a chosen state without a run id
    assert AC.AnchorChoice(state="no_current_anchor", **fields).selection_json()["n_candidates"] == 0


def test_the_module_prose_keeps_the_registry_vocabulary():
    """The plan's banned-token rule holds for this module and this test as text, not only for identifiers."""
    import test_registry_vocabulary as TV
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (os.path.join(here, "..", "src", "sisteer", "registry", "adapters", "anchor_current.py"), os.path.abspath(__file__),
                os.path.join(here, "..", "src", "sisteer", "registry", "reader_state.py")):
        with open(rel, "r", encoding="utf-8") as fh:
            h = TV.hits(fh.read())
        assert not h, (rel, h[:5])


# ============================================================================================
# the judgment-currency rule: a displaced or released judgment's rows are no readings
# ============================================================================================

def test_a_superseded_or_abandoned_judgments_rows_are_no_readings_and_the_successors_row_is_the_one_reading():
    """The frame applies `schema_def.non_current_judgment_sql` to its metric read (the predicate text is schema_def's): a run whose complete
    judgment a re-judge displaced contributes nothing when the successor filed no row (`no_reading`, not the displaced number), the
    successor's row when it did (never the displaced judgment's, which sorts first by key), and a released (`abandoned`) judgment's row is
    no reading either; `counts["non_current_judgment_rows"]` says how many rows the rule dropped and a row of no judgment is untouched."""
    assert AC.CURRENT_JUDGMENT_SQL == S.non_current_judgment_sql("rm.judgment_id", "jm.status")
    assert AC.current_judgment_join() == "LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id"
    assert AC.current_judgment_join("live") == "LEFT JOIN live.judgment jm ON jm.judgment_id = rm.judgment_id"
    fr = Frame()
    # a: displaced, successor without a row
    fr.run("none:a", reading=False)
    old_a = fr.judgment("none:a", "j:a:1", status="superseded", superseded_by="j:a:2")
    fr.judgment("none:a", "j:a:2")
    fr.metric("none:a", old_a, value=10.0)
    # b: displaced, successor with its own row (the displaced row sorts first by judgment id)
    fr.run("none:b", reading=False)
    old_b = fr.judgment("none:b", "j:b:1", status="superseded", superseded_by="j:b:2")
    new_b = fr.judgment("none:b", "j:b:2")
    fr.metric("none:b", old_b, value=10.0)
    fr.metric("none:b", new_b, value=30.0, k=30)
    # c: released
    fr.run("none:c", reading=False)
    fr.metric("none:c", fr.judgment("none:c", "j:c:1", status="abandoned"), value=10.0)
    # d: a row of no judgment beside a displaced one -- the unjudged row is untouched
    fr.run("none:d", reading=False)
    fr.metric("none:d", fr.judgment("none:d", "j:d:1", status="superseded"), value=10.0)
    fr.metric("none:d", AC.NO_JUDGMENT, value=20.0, k=20)
    ac = fr.load()
    assert (ac.counts["metric_rows"], ac.counts["non_current_judgment_rows"], ac.counts["withdrawn_metric_rows"], ac.counts["judgments"]) == (2, 4, 0, 2)
    # a's own anchor is unusable (no reading under its successor) and the search continues: the one same-standup run with a reading (b) anchors
    a = ac.choose(key(own_anchor_run_id="none:a", own_anchor_kind="same_standup_none"), METRIC, THR, JUDGE)
    _closed(a)
    assert a.own_reason == "no_reading" and a.own_why == f"no reading at ({METRIC}, {THR:g}) under judgment j:a:2"
    assert a.state == "sibling_same_source" and a.run_id == "none:b" and a.anchor_kind == "same_standup_none" and a.measurement is not None
    assert (a.measurement.judgment_id, a.measurement.value) == ("j:b:2", 30.0) and [c.run_id for c in a.candidates] == ["none:b"]
    b = ac.choose(key(own_anchor_run_id="none:b", own_anchor_kind="same_standup_none"), METRIC, THR, JUDGE)
    _closed(b)
    assert b.state == "own_current" and b.measurement is not None and (b.measurement.judgment_id, b.measurement.value, b.measurement.k) == ("j:b:2", 30.0, 30)
    assert b.anchor_judgment_id == "j:b:2" and b.rejected == ((
        "none:a", f"no reading at ({METRIC}, {THR:g}) under judgment j:a:2"), ("none:c", f"no complete judgment under judge {JUDGE}"),
        ("none:d", f"no complete judgment under judge {JUDGE}"))
    d = ac.choose(key(own_anchor_run_id="none:d", own_anchor_kind="same_standup_none"), METRIC, THR, None)
    _closed(d)
    assert d.state == "own_current" and d.measurement is not None and (d.measurement.judgment_id, d.measurement.value) == (AC.NO_JUDGMENT, 20.0)


def test_through_the_fold_a_rejudge_displaces_the_anchors_reading_until_the_successor_files_its_own(tmp_path):
    """The real registry end to end: `withdrawn_reading_world` after `withdraw_twin` (the producer's row is the ONE reading and the run
    anchors with it); a same-judge re-judge leaves the run current (this world has no roster: `not_in_roster`, and the frame admits every
    row on its judgment alone) while `v_paper_long` counts no reading for it -- the
    frame drops both rows of the displaced judgment (the withdrawn twin among them: a row failing both rules is counted under the
    judgment rule) and the run refuses `no_reading` under the successor; the successor's own row then anchors it."""
    root = str(tmp_path / "reg")
    w = withdrawn_reading_world(root)
    withdraw_twin(root, w)
    none, k = str(w["none"]), w["key"]
    own_key = replace(k, own_anchor_run_id=none, own_anchor_kind="same_standup_none")

    def state():
        with DB.reading(DB.db_path(root)) as conn:
            vrc = tuple(conn.execute("SELECT is_current, selection_state FROM v_run_current WHERE run_id = ?", (none,)).fetchone())
            frame = [str(r[0]) for r in conn.execute("SELECT judgment_id FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr'", (none,))]
            table_rows = conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ?", (none,)).fetchone()[0]
            ac = AC.AnchorCurrency.load(conn)
            own = ac.choose(own_key, "asr", 0.5, synth.JUDGE_ID)
        _closed(own)
        return vrc, frame, table_rows, ac, own

    vrc, frame, table_rows, ac, own = state()
    assert vrc == (1, "not_in_roster") and frame == [str(w["judgment"])] and table_rows == 2
    assert (ac.counts["metric_rows"], ac.counts["withdrawn_metric_rows"], ac.counts["non_current_judgment_rows"]) == (1, 1, 0)
    assert own.state == "own_current" and own.measurement is not None and own.measurement.judgment_id == str(w["judgment"]) and own.measurement.value == 0.25
    j_new = rejudge_none(root, w)
    vrc, frame, table_rows, ac, own = state()
    assert vrc == (1, "not_in_roster") and frame == [] and table_rows == 2                       # the table keeps the displaced judgment's rows
    assert (ac.counts["metric_rows"], ac.counts["withdrawn_metric_rows"], ac.counts["non_current_judgment_rows"]) == (0, 0, 2)
    assert own.state == "no_current_anchor" and own.own_reason == "no_reading" and own.own_why == f"no reading at (asr, 0.5) under judgment {j_new}"
    assert rejudge_none(root, w, replacement=0.4, i=9200) == j_new
    vrc, frame, table_rows, ac, own = state()
    assert vrc == (1, "not_in_roster") and frame == [j_new] and table_rows == 3
    assert (ac.counts["metric_rows"], ac.counts["withdrawn_metric_rows"], ac.counts["non_current_judgment_rows"]) == (1, 0, 2)
    assert own.state == "own_current" and own.measurement is not None and (own.measurement.judgment_id, own.measurement.value, own.measurement.k) == (j_new, 0.4, 40)
    assert own.anchor_judgment_id == j_new and own.selection_json()["reading"]["judgment_id"] == j_new


# ============================================================================================
# the anchor decision as an input of the read: the log, the records, the fingerprint, the synthetic input
# ============================================================================================

def test_the_decision_input_is_the_fingerprint_as_a_watermark_and_never_a_file():
    sid = "synthetic_store:s1"
    fp = "ab" * 32
    s = AC.AnchorDecisionInput.of(sid, fp)
    assert s.uri == s.identity_uri == f"registry://anchor-decision/{sid}" == AC.decision_uri(sid)
    assert (s.sha256, s.size, s.mtime_utc, s.path, s.role) == (fp, 0, AC.DECISION_MTIME, "", AC.DECISION_ROLE)
    # the watermark IS the fingerprint (base.watermark_of reads sha256, size, mtime_utc), and only the fingerprint moves it
    from sisteer.registry.adapters import base as AB
    assert AB.watermark_of(s) == (fp, 0, AC.DECISION_MTIME) and AB.watermark_of(AC.AnchorDecisionInput.of(sid, "cd" * 32)) != AB.watermark_of(s)
    assert AC.is_decision_input(s) and AC.is_decision_uri(s.uri) and AC.decision_source_id(s.uri) == sid
    assert not AC.is_decision_input(AB.Input("/x/store.json", "/x/store.json", "a" * 64, 1, "2026-01-01T00:00:00.000Z"))
    assert not AC.is_decision_uri("/x/registry://anchor-decision/not-a-prefix") and not AC.is_decision_uri(None)
    with pytest.raises(ValueError, match="64 hex"):
        AC.AnchorDecisionInput.of(sid, "short")
    with pytest.raises(ValueError, match="not an anchor decision URI"):
        AC.decision_source_id("/x/store.json")
    # the context's identity map leaves a non-path URI alone: the grading host and a relocated shard spell the same input
    assert AB.IdentityMap.build({"/home/local": "/home/canonical"}).canonical(s.uri) == s.uri


def test_comparison_record_round_trips_the_key_and_decision_record_spells_the_outcome():
    k = key(own_anchor_run_id="none:own", own_anchor_kind="same_standup_none", decode_policy_id="dp:1", pass_label="pooled_t0_t1")
    rec = AC.comparison_record(k, METRIC, THR, JUDGE, computation_id="c:1", needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    assert rec["run_id"] == "run:s" and rec["judgment_id"] == "j:s" and rec["metric_id"] == METRIC and rec["threshold"] == THR
    assert rec["computation_id"] == "c:1" and rec["judge_id"] == JUDGE and rec["needs_items"] is True
    assert rec["key"]["standup"] == STANDUP and rec["key"]["own_anchor_run_id"] == "none:own" and rec["key"]["pass_label"] == "pooled_t0_t1"
    assert AC.key_of_comparison(rec) == k
    # a threshold-less (rule-scored) comparison records the sentinel, and json round-trips to the same record
    rec2 = AC.comparison_record(key(), METRIC, None, None)
    assert rec2["threshold"] == -1.0 and rec2["judge_id"] is None and rec2["run_id"] is None
    assert json.loads(I.canonical_json(rec)) == rec
    # the decision: the chosen run and its reading, or "none" with the park reason
    fr = Frame()
    own = fr.run("none:own", computation="c:own")
    chosen = AC.decision_record(choose(fr, key(own_anchor_run_id=own, own_anchor_kind="same_standup_none"), computation_id="c:own"))
    assert chosen == {"anchor_run_id": own, "state": "own_current", "own_reason": "current", "anchor_kind": "same_standup_none",
                      "anchor_judgment_id": f"j:{own}:{JUDGE}:1", "computation_id": "c:own",
                      "reading": {"value": 10.0, "k": 10, "n": 100, "ci_lo": 5.0, "ci_hi": 15.0}, "items_digest": None,
                      "steered_items_digest": None, "steered_item_conflicts": None, "flags_digest": None, "steered_flags_digest": None, "why": ""}
    parked = AC.decision_record(choose(fr, key(instrument_id="inst:other")))
    assert parked["anchor_run_id"] == AC.PARKED == "none" and parked["state"] == "no_current_anchor" and parked["anchor_kind"] is None
    assert parked["anchor_judgment_id"] == "" and parked["computation_id"] is None and "instrument" in parked["why"]
    assert parked["reading"] is None and parked["items_digest"] is None and parked["steered_items_digest"] is None and parked["steered_item_conflicts"] is None


def test_choose_logs_the_comparison_and_its_decision_when_a_log_is_bound_and_replay_does_not():
    fr = Frame()
    own = fr.run("none:own")
    ac = fr.load()
    assert ac.decision_log is None
    k = key(own_anchor_run_id=own, own_anchor_kind="same_standup_none")
    ch0 = ac.choose(k, METRIC, THR, JUDGE, steered_run_id="run:s")             # unbound: decided, nothing logged
    log: list = []
    ac.decision_log = log
    ch = ac.choose(k, METRIC, THR, JUDGE, steered_run_id="run:s", steered_judgment_id="j:s")
    assert ch == ch0 and len(log) == 1 and set(log[0]) == {"comparison", "decision"}
    assert log[0]["comparison"] == AC.comparison_record(k, METRIC, THR, JUDGE, steered_run_id="run:s", steered_judgment_id="j:s")
    assert log[0]["decision"] == AC.decision_record(ch)
    # replay re-decides a recorded comparison on this frame without logging; decisions_of orders the set canonically
    assert ac.replay(log[0]["comparison"]) == log[0]["decision"] and len(log) == 1
    other = AC.comparison_record(key(), METRIC, THR, JUDGE, steered_run_id="run:a")
    entries = ac.decisions_of([log[0]["comparison"], other])
    assert len(log) == 1 and entries == AC.sorted_entries(entries) and {e["comparison"]["run_id"] for e in entries} == {"run:s", "run:a"}
    assert entries[0]["decision"]["anchor_run_id"] == entries[1]["decision"]["anchor_run_id"] == own     # both anchor on the one current run
    # a parked comparison is in the set as "none" exactly like a filed one
    ac.choose(key(instrument_id="inst:other"), METRIC, THR, JUDGE, steered_run_id="run:p")
    assert len(log) == 2 and log[1]["decision"]["anchor_run_id"] == AC.PARKED


def test_the_fingerprint_is_order_free_byte_stable_and_moves_only_with_the_decision():
    k = key(own_anchor_run_id="none:own", own_anchor_kind="same_standup_none")
    cmp_a = AC.comparison_record(k, METRIC, THR, JUDGE, steered_run_id="run:a")
    cmp_b = AC.comparison_record(k, METRIC, THR, JUDGE, steered_run_id="run:b")

    def fingerprint(fr: Frame):
        ac = fr.load()
        return AC.decision_fingerprint(ac.decisions_of([cmp_a, cmp_b])), ac.decisions_of([cmp_b, cmp_a])

    base = Frame()
    base.run("none:own")
    fp0, entries0 = fingerprint(base)
    assert len(fp0) == 64 and fp0 == AC.decision_fingerprint(entries0) == AC.decision_fingerprint(list(reversed(entries0)))     # order-free
    assert fingerprint(base)[0] == fp0                                                                                           # byte-stable
    # a new same-key unsteered run that does NOT move the pick changes the selection block's candidate list, not the decision
    wider = Frame()
    wider.run("none:own")
    wider.run("none:other", standup="tagB")
    ch = choose(wider, k)
    assert ch.state == "own_current" and [c.run_id for c in ch.candidates] == ["none:other", "none:own"] and ch.n_candidates == 2
    assert fingerprint(wider)[0] == fp0
    # the own run retired: the pick moves to the other run -- a new fingerprint; held out with nothing else: parked -- another one
    retired = Frame()
    retired.run("none:own", current=0, why="selection_state excluded (ruling r1)")
    retired.run("none:other", standup="tagB")
    fp1, entries1 = fingerprint(retired)
    assert fp1 != fp0 and all(e["decision"]["anchor_run_id"] == "none:other" and e["decision"]["own_reason"] == "excluded" for e in entries1)
    held = Frame()
    held.run("none:own", current=0, why="held_out (ruling r2)", held_out=1)
    fp2, entries2 = fingerprint(held)
    assert fp2 not in (fp0, fp1) and all(e["decision"]["anchor_run_id"] == AC.PARKED and e["decision"]["own_reason"] == "held_out" for e in entries2)
    # the same decision reached again (the hold-out lifted) is the same fingerprint: the revision it built is re-activated, not re-minted
    assert fingerprint(retired)[0] == fp1
    # the marker block
    block = AC.decisions_block(entries2, "synthetic_store:s1", pre_read_fingerprint=fp1)
    assert block["rule"] == AC.DECISION_RULE and block["fingerprint"] == fp2 and block["pre_read_fingerprint"] == fp1
    assert block["input_uri"] == AC.decision_uri("synthetic_store:s1") and (block["n_comparisons"], block["n_parked"]) == (2, 2)
    assert block["comparisons"] == AC.sorted_entries(entries2)
    assert not any(key_ in json.dumps(block) for key_ in ("_utc", "observed", "derived_at"))


def test_the_context_binds_the_log_to_the_frame_and_begin_group_clears_it_in_place():
    fr = Frame()
    own = fr.run("none:own")
    ctx = AdapterContext(root="", now="2026-10-02T00:00:00.000Z", producer={}, conn=fr.con)
    assert ctx.group is None and ctx.decision_log == []
    ac = ctx.anchor_currency()
    assert ac.decision_log is ctx.decision_log
    ac.choose(key(own_anchor_run_id=own, own_anchor_kind="same_standup_none"), METRIC, THR, JUDGE, steered_run_id="run:s")
    assert len(ctx.decision_log) == 1
    ctx.begin_group("g2")
    assert ctx.group == "g2" and ctx.decision_log == [] and ac.decision_log is ctx.decision_log       # the same list, emptied
    ctx.anchor_currency().choose(key(), METRIC, THR, JUDGE, steered_run_id="run:t")
    assert len(ctx.decision_log) == 1 and ctx.decision_log[0]["comparison"]["run_id"] == "run:t"
    # a frame handed to the context from outside is bound on the next call too
    ctx2 = AdapterContext(root="", now="2026-10-02T00:00:00.000Z", producer={}, conn=fr.con, anchors=fr.load())
    assert ctx2.anchor_currency().decision_log is ctx2.decision_log
    # the positional construction is unchanged (the new fields are last and defaulted)
    assert AdapterContext("", "now", {}, {}, None).decision_log == []


def test_a_reading_corrected_under_an_unchanged_identity_is_a_changed_decision():
    """A `run_metric` row corrected in place keeps its run, judgment and computation ids and moves `value` / `k` /
    `n` (and the contrast's delta with them); a decision over the ids alone replayed to the same fingerprint and the stale delta stood. The
    decision carries the reading's statistic (`Measurement.statistic_json`), so each of value, k, n and the filed interval moves the
    fingerprint on its own; the same reading again is byte-stable; a parked comparison carries no reading."""
    k = key(own_anchor_run_id="none:own", own_anchor_kind="same_standup_none")
    cmp_ = AC.comparison_record(k, METRIC, THR, JUDGE, computation_id="c:own", steered_run_id="run:s")

    def fp(*, ci=(5.0, 15.0), current=1, **reading):
        fr = Frame()
        fr.run("none:own", computation="c:own", reading=False, current=current, why=(None if current else "held_out (ruling r)"), held_out=int(not current))
        fr.metric("none:own", fr.judgment("none:own"), computation="c:own", ci=ci, **reading)
        entries = fr.load().decisions_of([cmp_])
        return AC.decision_fingerprint(entries), entries[0]["decision"]

    fp0, d0 = fp(value=0.1, k=1, n=10)
    assert d0["anchor_run_id"] == "none:own" and d0["computation_id"] == "c:own" and d0["items_digest"] is None
    assert d0["reading"] == {"value": 0.1, "k": 1, "n": 10, "ci_lo": 5.0, "ci_hi": 15.0}
    assert fp(value=0.1, k=1, n=10)[0] == fp0                                                           # byte-stable
    fp_value, d_value = fp(value=0.9, k=1, n=10)
    assert fp_value != fp0 and d_value["reading"]["value"] == 0.9
    assert {kk: v for kk, v in d_value.items() if kk != "reading"} == {kk: v for kk, v in d0.items() if kk != "reading"}   # the identity unchanged
    fps = {fp0, fp_value, fp(value=0.1, k=9, n=10)[0], fp(value=0.1, k=1, n=20)[0], fp(value=0.1, k=1, n=10, ci=(0.0, 0.3))[0]}
    assert len(fps) == 5                                                                                 # value, k, n and the interval each move it
    # the same change under a parked comparison moves nothing: a park carries no reading
    fp_p, d_p = fp(value=0.1, k=1, n=10, current=0)
    assert d_p["anchor_run_id"] == AC.PARKED and d_p["reading"] is None and fp(value=0.9, k=1, n=10, current=0)[0] == fp_p


def test_an_item_paired_decision_digests_the_anchors_current_item_rows_and_moves_with_them():
    """e4 (a statistic paired over registry items) reads the anchor's item rows, not its aggregate: the decision of a comparison that asked
    for items carries a digest of the chosen anchor's CURRENT rows at the metric under the reading's judgment (`AnchorCurrency.items_digest`,
    in key order), so a corrected or re-filed item under the same judgment is a changed decision; a comparison that did not ask carries none,
    and a pair with no rows has none."""
    def frame(rows_b):
        fr = Frame()
        a = fr.run("none:a", standup="tagB")
        b = fr.run("none:b", standup="tagC", utc="2026-09-02T00:00:00Z")
        fr.items(b, f"j:{b}:{JUDGE}:1", rows_b)
        return fr, a, b

    fr, a, b = frame([("item:1", 1.0), ("item:2", 0.0)])
    ac = fr.load()
    ch = ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    _closed(ch)
    assert ch.run_id == b and ch.items_digest == ac.items_digest(METRIC, b, f"j:{b}:{JUDGE}:1") and len(ch.items_digest) == 64
    assert AC.decision_record(ch)["items_digest"] == ch.items_digest
    assert ac.choose(key(), METRIC, THR, JUDGE).items_digest is None                                   # e4 not asked: no digest (and a, the earliest)
    assert ac.items_digest(METRIC, a, f"j:{a}:{JUDGE}:1") is None and ac.items_at(METRIC) == {(b, f"j:{b}:{JUDGE}:1")}
    # the same rows filed in another order: the same digest (key order, not arrival)
    fr2, _, b2 = frame([("item:2", 0.0), ("item:1", 1.0)])
    assert fr2.load().items_digest(METRIC, b2, f"j:{b2}:{JUDGE}:1") == ch.items_digest
    # one item's value corrected under the same judgment: another digest, another fingerprint of the same comparison
    fr3, _, b3 = frame([("item:1", 1.0), ("item:2", 1.0)])
    ch3 = fr3.load().choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    assert ch3.run_id == b3 and ch3.items_digest != ch.items_digest and ch3.measurement == ch.measurement        # the aggregate unchanged
    cmp_ = AC.comparison_record(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    assert AC.decision_fingerprint(fr.load().decisions_of([cmp_])) != AC.decision_fingerprint(fr3.load().decisions_of([cmp_]))
    assert AC.decision_fingerprint(fr2.load().decisions_of([cmp_])) == AC.decision_fingerprint(fr.load().decisions_of([cmp_]))
    # an item added (a re-filed row) moves it too
    fr4, _, b4 = frame([("item:1", 1.0), ("item:2", 0.0), ("item:3", 0.5)])
    assert fr4.load().items_digest(METRIC, b4, f"j:{b4}:{JUDGE}:1") != ch.items_digest


# ============================================================================================
# e4 and conflicting visible provenances (review fix 2)
# ============================================================================================

def test_a_pair_with_conflicting_item_provenances_is_refused_by_name_and_the_conflict_moves_the_fingerprint():
    """The item digest hashes only the row `v_item_result_current` SELECTS per key, so a second accepted provenance of an
    existing item that DISAGREES with the selected row (`v_item_conflict`) left the digest, the decision and the fingerprint unchanged while
    the registry's paired-reading API refused the pair. Now e4 refuses a conflicted pair by name (`item_conflict`, the conflicted item count
    and ids), as the API does, and the conflict reaches the fingerprint through the decision: the comparison parks (or moves to another
    anchor) and its `own_reason` / `why` name the conflict. The digest of the selected rows itself is unchanged -- it is not what carries it."""
    fr = Frame()
    a = fr.run("none:a", standup="tagB")
    b = fr.run("none:b", standup="tagC", utc="2026-09-02T00:00:00Z")
    jb = f"j:{b}:{JUDGE}:1"
    fr.items(b, jb, [("item:1", 1.0), ("item:2", 0.0)])
    cmp_ = AC.comparison_record(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    ac0 = fr.load()
    ch0 = ac0.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    assert ch0.run_id == b and ch0.items_digest and ac0.items_conflicts(METRIC, b, jb) == []
    fp0 = AC.decision_fingerprint(ac0.decisions_of([cmp_]))
    # a second accepted provenance of item:1 disagrees; the selected row (value 1.0) and the digest over the selected rows are unchanged
    fr.conflict(b, jb, "item:1", values=(1.0, 0.0))
    ac1 = fr.load()
    assert ac1.items_digest(METRIC, b, jb) == ch0.items_digest and ac1.items_at(METRIC) == {(b, jb)}
    assert ac1.items_conflicts(METRIC, b, jb) == ["item:1"] and ac1.items_conflicts(METRIC, a, f"j:{a}:{JUDGE}:1") == []
    ch1 = ac1.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    _closed(ch1)
    assert ch1.state == "no_current_anchor" and ch1.run_id is None and ch1.items_digest is None
    why_b = dict(ch1.rejected)[b]
    assert why_b == (f"1 item(s) at ({METRIC}, {THR:g}) under judgment {jb} carry conflicting visible provenances (item:1; v_item_conflict): "
                     "a statistic paired over items refuses the pair while the sources disagree")
    assert why_b in ch1.why and f"{a} (no current item rows" in ch1.why
    fp1 = AC.decision_fingerprint(ac1.decisions_of([cmp_]))
    assert fp1 != fp0 and ac1.replay(cmp_)["anchor_run_id"] == AC.PARKED and why_b in ac1.replay(cmp_)["why"]
    # without e4 the conflict is not asked: the aggregate comparison is unchanged
    assert ac1.choose(key(), METRIC, THR, JUDGE).run_id == a and AC.decision_fingerprint(ac1.decisions_of([AC.comparison_record(key(), METRIC, THR, JUDGE)])) \
        == AC.decision_fingerprint(ac0.decisions_of([AC.comparison_record(key(), METRIC, THR, JUDGE)]))
    # the own anchor conflicted: `own_reason` names it and the search continues (here to nothing)
    own = ac1.choose(key(own_anchor_run_id=b, own_anchor_kind="same_standup_none"), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    _closed(own)
    assert own.state == "no_current_anchor" and own.own_reason == "item_conflict" and own.own_why == why_b and own.why.startswith(f"own anchor {b}: {why_b}")
    # a conflict at ANOTHER metric does not touch this pair (lazy per metric); a second conflicted item changes the count, so the decision again
    fr.conflict(b, jb, "item:9", metric="other_metric")
    assert fr.load().items_conflicts(METRIC, b, jb) == ["item:1"]
    fr.conflict(b, jb, "item:2", values=(0.0, 0.5, 0.5))
    ac2 = fr.load()
    assert ac2.items_conflicts(METRIC, b, jb) == ["item:1", "item:2"] and "2 item(s) at" in dict(ac2.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s").rejected)[b]
    assert AC.decision_fingerprint(ac2.decisions_of([cmp_])) not in (fp0, fp1)
    # a conflicted sibling leaves the other candidate the anchor: the conflict moves the pick, not only the park
    fr.items(a, f"j:{a}:{JUDGE}:1", [("item:1", 1.0), ("item:2", 0.0)])
    ch3 = fr.load().choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    _closed(ch3)
    assert ch3.run_id == a and ch3.state == "sibling_same_source" and "conflicting visible provenances" in dict(ch3.rejected)[b]


def test_an_item_conflict_refusal_names_a_bounded_list_of_the_conflicted_items():
    fr = Frame()
    b = fr.run("none:b")
    jb = f"j:{b}:{JUDGE}:1"
    fr.items(b, jb, [(f"item:{i}", 1.0) for i in range(9)])
    for i in range(7):
        fr.conflict(b, jb, f"item:{i}")
    ch = fr.load().choose(key(own_anchor_run_id=b), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    assert ch.own_reason == "item_conflict" and ch.own_why.startswith("7 item(s) at")
    assert "(item:0, item:1, item:2, item:3, item:4 (+2 more); v_item_conflict)" in ch.own_why
    assert AC.CONFLICT_NAMED == 5


def test_item_presence_is_known_only_where_both_item_objects_are_defined():
    """A database defining `v_item_result_current` without `v_item_conflict` (or the reverse) is not a shard and not the grading host: the frame knows
    no items and e4 names the half-defined shape, rather than reading the selected rows as conflict-free."""
    fr = Frame()
    b = fr.run("none:b")
    fr.items(b, f"j:{b}:{JUDGE}:1", [("item:1", 1.0)], conflicts=False)
    ac = fr.load()
    assert ac.items_known is False and ac.counts["items_known"] == 0
    assert ac.items_unknown_why == "this database defines v_item_result_current without v_item_conflict; the item rule needs both"
    ch = ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="run:s", steered_judgment_id="j:s")
    assert ch.state == "no_current_anchor" and "defines v_item_result_current without v_item_conflict" in dict(ch.rejected)[b]
    assert ac.items_at(METRIC) == set() and ac.items_digest(METRIC, b, f"j:{b}:{JUDGE}:1") is None and ac.items_conflicts(METRIC, b, f"j:{b}:{JUDGE}:1") == []
    fr2 = Frame()
    b2 = fr2.run("none:b")
    fr2.item_objects()
    fr2.con.execute("DROP VIEW v_item_result_current")
    ac2 = fr2.load()
    assert ac2.items_known is False and ac2.items_unknown_why == "this database defines v_item_conflict without v_item_result_current; the item rule needs both"
    assert ac2.items_conflicts(METRIC, b2, f"j:{b2}:{JUDGE}:1") == []


# ============================================================================================
# e4 reads the STEERED side too
# ============================================================================================

def test_an_item_paired_decision_carries_the_steered_sides_items_and_conflicts_and_moves_with_them():
    """The item reads covered the unsteered candidates only, so a second accepted provenance disagreeing with a STEERED item
    (selected row unchanged, no store file changed) replayed to the same clean anchor and the same fingerprint while `registry.paired`
    refused the pair, and a corrected steered selected item changed the paired statistic without invalidating the head. The decision now
    carries the steered side: the digest of the steered row's current item rows (`steered_items_digest`, one indexed probe per pair) and its
    conflicted key count (`steered_item_conflicts`); a steered conflict parks the comparison under `item_conflict` with the steered run named,
    whatever the pool holds, and a steered item correction is another fingerprint."""
    fr = Frame()
    b = fr.run("none:b")
    jb = f"j:{b}:{JUDGE}:1"
    fr.items(b, jb, [("item:1", 1.0), ("item:2", 0.0)])
    s = fr.steered("lora:s")
    js = f"j:{s}:{JUDGE}:1"
    fr.items(s, js, [("item:1", 0.0), ("item:2", 0.0)])
    cmp_ = AC.comparison_record(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id=s, steered_judgment_id=js)
    ac0 = fr.load()
    ch0 = ac0.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id=s, steered_judgment_id=js)
    _closed(ch0)
    assert ch0.run_id == b and ch0.steered_item_conflicts == 0 and ch0.steered_items_digest == ac0.pair_items_digest(METRIC, s, js) and len(ch0.steered_items_digest) == 64
    assert ac0.pair_items_digest(METRIC, b, jb) == ac0.items_digest(METRIC, b, jb) == ch0.items_digest          # one line format, both reads
    d0 = AC.decision_record(ch0)
    assert d0["steered_items_digest"] == ch0.steered_items_digest and d0["steered_item_conflicts"] == 0 and d0["items_digest"] == ch0.items_digest
    assert ac0.replay(cmp_) == d0                                                                                  # replay equals the logged decision
    fp0 = AC.decision_fingerprint(ac0.decisions_of([cmp_]))
    # e4 not asked: no steered state
    plain = ac0.choose(key(), METRIC, THR, JUDGE, steered_run_id=s, steered_judgment_id=js)
    assert plain.steered_items_digest is None and plain.steered_item_conflicts is None
    # the steered rows not folded yet (a first read): digest None, no conflict, the anchor chosen; the fold moves the fingerprint
    cmp_new = AC.comparison_record(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="lora:unfolded", steered_judgment_id="j:u")
    unfolded = ac0.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="lora:unfolded", steered_judgment_id="j:u")
    assert unfolded.run_id == b and unfolded.steered_items_digest is None and unfolded.steered_item_conflicts == 0
    fr.items("lora:unfolded", "j:u", [("item:1", 1.0)])
    assert AC.decision_fingerprint(fr.load().decisions_of([cmp_new])) != AC.decision_fingerprint(ac0.decisions_of([cmp_new]))
    # a steered selected item corrected (the anchor and every file untouched): another steered digest, another fingerprint, the anchor's digest equal
    fr.con.execute("UPDATE item_result SET value = 1.0 WHERE run_id = ? AND item_id = 'item:2'", (s,))
    ac1 = fr.load()
    ch1 = ac1.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id=s, steered_judgment_id=js)
    assert ch1.run_id == b and ch1.items_digest == ch0.items_digest and ch1.steered_items_digest != ch0.steered_items_digest
    fp1 = AC.decision_fingerprint(ac1.decisions_of([cmp_]))
    assert fp1 != fp0
    # a second accepted provenance disagreeing with a steered item (its selected row unchanged): the comparison PARKS, the steered run named
    fr.conflict(s, js, "item:1", values=(0.0, 1.0))
    ac2 = fr.load()
    assert ac2.pair_items_digest(METRIC, s, js) == ch1.steered_items_digest and ac2.pair_items_conflicts(METRIC, s, js) == ["item:1"]
    ch2 = ac2.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id=s, steered_judgment_id=js)
    _closed(ch2)
    why = (f"the steered run {s}: 1 item(s) at ({METRIC}, {THR:g}) under judgment {js} carry conflicting visible provenances (item:1; v_item_conflict): "
           "a statistic paired over items refuses the pair while the sources disagree")
    assert ch2.state == "no_current_anchor" and ch2.run_id is None and ch2.own_reason == "item_conflict" and ch2.own_why == why and ch2.why == why
    assert ch2.steered_item_conflicts == 1 and ch2.steered_items_digest == ch1.steered_items_digest and ch2.items_digest is None
    assert [c.run_id for c in ch2.candidates] == [b] and ch2.rejected == ()                                        # the pool stays on the record
    d2 = AC.decision_record(ch2)
    assert d2["anchor_run_id"] == AC.PARKED and d2["steered_item_conflicts"] == 1 and d2["why"] == why and ac2.replay(cmp_) == d2
    assert AC.decision_fingerprint(ac2.decisions_of([cmp_])) not in (fp0, fp1)
    # the own anchor named and usable: still a park, `own_reason` item_conflict (the pair could not be formed)
    own = ac2.choose(key(own_anchor_run_id=b, own_anchor_kind="same_standup_none"), METRIC, THR, JUDGE, needs_items=True, steered_run_id=s, steered_judgment_id=js)
    assert own.state == "no_current_anchor" and own.own_reason == "item_conflict" and own.why == why
    # the pair reads are per metric: a conflict at another metric leaves this comparison alone
    fr.conflict(s, js, "item:2", metric="other_metric")
    assert fr.load().pair_items_conflicts(METRIC, s, js) == ["item:1"] and fr.load().pair_items_conflicts("other_metric", s, js) == ["item:2"]


def test_an_item_paired_comparison_names_its_steered_run_and_judgment():
    fr = Frame()
    b = fr.run("none:b")
    fr.items(b, f"j:{b}:{JUDGE}:1", [("item:1", 1.0)])
    ac = fr.load()
    with pytest.raises(ValueError, match="names its steered run and judgment"):
        ac.choose(key(), METRIC, THR, JUDGE, needs_items=True)
    with pytest.raises(ValueError, match="names its steered run and judgment"):
        ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="lora:s")
    assert ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="lora:s", steered_judgment_id="").run_id == b   # rule-scored: judgment ''
    assert ac.choose(key(), METRIC, THR, JUDGE).run_id == b                                                              # not asked: nothing required
    with pytest.raises(ValueError):
        ac.replay({"run_id": None, "judgment_id": None, "metric_id": METRIC, "threshold": THR, "judge_id": JUDGE, "needs_items": True,
                   "key": AC.comparison_record(key(), METRIC, THR, JUDGE)["key"]})


def test_the_steered_side_is_unknown_where_the_frame_knows_no_items():
    """A shard: the candidates refuse e4 by name as before; the steered fields read None (not asked of a frame that cannot answer)."""
    fr = Frame()
    fr.run("none:b")
    ch = fr.load().choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id="lora:s", steered_judgment_id="j:s")
    assert ch.state == "no_current_anchor" and ch.steered_items_digest is None and ch.steered_item_conflicts is None
    assert fr.load().pair_items_digest(METRIC, "lora:s", "j:s") is None and fr.load().pair_items_conflicts(METRIC, "lora:s", "j:s") == []



# ============================================================================================
# the currency-only check of a STEERED anchor (plan commit 4: r2_export's `full_dose` 1x reading) and the item values a paired statistic reads
# ============================================================================================

def test_a_steered_anchors_currency_is_checked_never_re_picked_and_the_check_is_a_recorded_comparison():
    """The export's second kind anchors a half-dose row on the same object's 1x run -- a steered run, so `choose`'s exact-key search does not
    apply: the fold's currency decides (current -> file, retired or unfolded -> park), and the check is logged under its own `kind` so the §4
    decision input re-reads the group when that currency moves (the first pass emits both runs: the park must be in the set)."""
    fr = Frame()
    cur = fr.steered("lora:1x")
    gone = fr.steered("lora:old", current=0, why="selection_state superseded (a later run at the same key)", in_paper=0, roster=False)
    held = fr.steered("lora:held", current=0, why="held_out (ruling r_synth_hold)", held_out=1, in_paper=0, roster=False)
    frame = fr.load()
    ok = frame.check_currency(cur, "full_dose", METRIC, THR, computation_id="c:x", steered_run_id="lora:half", steered_judgment_id="j:half")
    assert ok.state == "own_current" and ok.own_reason == "current" and ok.is_current is True and ok.why == "" and ok.anchor_kind == "full_dose"
    sel = ok.selection_json()
    assert sel["state"] == "own_current" and sel["own_anchor_run_id"] == cur and sel["rule"] == AC.RULE and sel["n_candidates"] == 0 and sel["key"] is None
    assert sel["currency_only"] and sel["anchor_kind"] == "full_dose" and sel["why"] == ""
    retired = frame.check_currency(gone, "full_dose", METRIC, THR)
    assert retired.state == "no_current_anchor" and retired.own_reason == "superseded" and retired.is_current is False and "later run" in retired.why
    assert retired.selection_json()["why"] == retired.why
    assert frame.check_currency(held, "full_dose", METRIC, THR).own_reason == "held_out"
    unknown = frame.check_currency("lora:never", "full_dose", METRIC, THR)
    assert unknown.state == "no_current_anchor" and unknown.own_reason == "not_in_frame" and unknown.is_current is None
    with pytest.raises(ValueError):
        AC.CurrencyCheck("x", "not_a_kind", True, "own_current", "current", "")
    # logged when a log is bound, in the ONE decision shape (the marker's n_parked and the fingerprint read it), never on a bare frame
    assert frame.decision_log is None
    log: list = []
    frame.decision_log = log
    frame.check_currency(cur, "full_dose", METRIC, THR, computation_id="c:x", steered_run_id="lora:half", steered_judgment_id="j:half")
    frame.check_currency("lora:never", "full_dose", METRIC, THR, computation_id="c:x", steered_run_id="lora:half2", steered_judgment_id="j:half2")
    assert len(log) == 2
    cmp, dec = log[0]["comparison"], log[0]["decision"]
    assert AC.is_currency_comparison(cmp) and cmp["kind"] == AC.CURRENCY_KIND and cmp["anchor_run_id"] == cur and cmp["anchor_kind"] == "full_dose"
    assert (cmp["run_id"], cmp["judgment_id"], cmp["metric_id"], cmp["threshold"], cmp["computation_id"]) == ("lora:half", "j:half", METRIC, THR, "c:x")
    assert dec["anchor_run_id"] == cur and dec["state"] == "own_current" and dec["own_reason"] == "current" and dec["reading"] is None
    assert set(dec) == set(AC.decision_record(choose(fr, key(own_anchor_run_id=fr.run("none:a")))))         # one shape for both kinds
    assert log[1]["decision"]["anchor_run_id"] == AC.PARKED and log[1]["decision"]["own_reason"] == "not_in_frame"
    block = AC.decisions_block(log, "src:x")
    assert (block["n_comparisons"], block["n_parked"], block["rule"]) == (2, 1, "anchor_decision/7")


def test_replay_decides_a_recorded_currency_comparison_again_on_another_frame_and_mixes_both_kinds():
    """The pre-read evaluation replays the head's recorded set against the pass's frame: a currency check parked as
    `not_in_frame` on the first pass re-evaluates to `own_current` once the 1x run folds (the fingerprint moves, the group re-reads), and a
    set mixing unsteered comparisons and currency checks replays whole."""
    first = Frame()
    none = first.run("none:a")
    log: list = []
    frame1 = first.load()
    frame1.decision_log = log
    frame1.choose(key(own_anchor_run_id=none, own_anchor_kind="same_standup_none"), METRIC, THR, JUDGE, steered_run_id="lora:s", steered_judgment_id="j:s")
    frame1.check_currency("lora:1x", "full_dose", METRIC, THR, steered_run_id="lora:half", steered_judgment_id="j:half")
    fp1 = AC.decision_fingerprint(log)
    comparisons = [e["comparison"] for e in log]
    assert frame1.decisions_of(comparisons) == AC.sorted_entries(log)                       # the same frame replays to the same set
    second = Frame()
    second.run("none:a")
    second.steered("lora:1x")
    replayed = second.load().decisions_of(comparisons)
    assert AC.decision_fingerprint(replayed) != fp1
    by_kind = {("currency" if AC.is_currency_comparison(e["comparison"]) else "anchor"): e["decision"] for e in replayed}
    assert by_kind["currency"]["anchor_run_id"] == "lora:1x" and by_kind["currency"]["state"] == "own_current"
    assert by_kind["anchor"]["anchor_run_id"] == none and by_kind["anchor"]["state"] == "own_current"
    # replay never logs
    assert len(log) == 2


def test_item_values_read_both_sides_rows_by_item_id_refuse_a_repeated_id_and_are_unknown_off_the_grading_host():
    fr = Frame()
    fr.item_objects()
    fr.items("lora:s", "j:s", [("beh_a/s1", 4.0), ("beh_a/s2", 5.0), ("beh_b/s1", None)])
    fr.items("none:a", "j:a", [("beh_a/s1", 5.5), ("beh_b/s1", 6.0)])
    frame = fr.load()
    assert frame.item_values(METRIC, "lora:s", "j:s") == {"beh_a/s1": 4.0, "beh_a/s2": 5.0, "beh_b/s1": None}
    assert frame.item_values(METRIC, "none:a", "j:a") == {"beh_a/s1": 5.5, "beh_b/s1": 6.0}
    assert frame.item_values(METRIC, "none:never", "j:x") == {}
    fr.con.execute("INSERT INTO item_result VALUES (?,?,?,?,?,?,?,?,?)", ("none:a", "j:a", METRIC, "beh_a/s1", 1, 0, -1, 5.0, None))
    with pytest.raises(ValueError, match="filed more than once"):
        fr.load().item_values(METRIC, "none:a", "j:a")
    shard = Frame()
    shard.run("none:b")
    assert shard.load().item_values(METRIC, "none:b", "j:b") is None


# ============================================================================================
# an item-paired comparison names the metric its item rows are filed at
# ============================================================================================

def test_an_item_paired_comparison_names_the_metric_of_its_item_rows_and_a_stores_rows_can_be_checked_against_the_digest():
    """The store readers' statistics are paired over per-item LABEL rows filed at the control's label metric
    (`xst_label` / `orb_label`), never at the contrast's (`xst_refused`), so an e4 keyed on the contrast's metric finds no rows there. The
    comparison names the metric (`items_metric_id`, recorded and replayed): presence, conflicts and both digests read it, on both sides; a
    rule-5 record without the field reads its own metric as before; the digest of a store's own payloads (`item_rows_digest` over
    `item_row_values`) equals the frame's, so an adapter can check the rows it re-read against the decision before filing; a label swap at
    the label metric that keeps the anchor's reading moves the fingerprint; a conflict at the label metric is named with both metrics."""
    LABEL = "xst_label"
    fr = Frame()
    b = fr.run("none:b")
    jb = f"j:{b}:{JUDGE}:1"
    fr.item_objects()
    rows_b = [("item:1", 2.0, "full_compliance"), ("item:2", 0.0, "full_refusal"), ("item:3", None, None)]
    for iid, val, lab in rows_b:
        fr.con.execute("INSERT INTO item_result VALUES (?,?,?,?,?,?,?,?,?)", (b, jb, LABEL, iid, 0, 0, -1, val, lab))
    s = fr.steered("lora:s")
    js = f"j:{s}:{JUDGE}:1"
    for iid, val, lab in (("item:1", 0.0, "full_refusal"), ("item:2", 0.0, "full_refusal")):
        fr.con.execute("INSERT INTO item_result VALUES (?,?,?,?,?,?,?,?,?)", (s, js, LABEL, iid, 0, 0, -1, val, lab))
    fr.con.commit()
    ac = fr.load()
    # e4 at the contrast's own metric: no rows there, rejected by name
    plain = ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id=s, steered_judgment_id=js)
    assert plain.state == "no_current_anchor" and dict(plain.rejected)[b] == f"no current item rows at {METRIC} under judgment {jb}"
    # the comparison names the label metric: chosen, both digests over the label rows, the reading the contrast's
    ch = ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, steered_run_id=s, steered_judgment_id=js)
    _closed(ch)
    assert ch.run_id == b and ch.items_digest == ac.items_digest(LABEL, b, jb) and len(ch.items_digest) == 64
    assert ch.steered_items_digest == ac.pair_items_digest(LABEL, s, js) and ch.steered_item_conflicts == 0
    assert ch.measurement == ac.choose(key(), METRIC, THR, JUDGE, steered_run_id=s, steered_judgment_id=js).measurement and ch.measurement.metric_id == METRIC
    # recorded on the comparison, replayed from it; None when e4 is not asked (the field is meaningless without it) or reads the own metric
    cmp_ = AC.comparison_record(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, steered_run_id=s, steered_judgment_id=js)
    assert cmp_["items_metric_id"] == LABEL and cmp_["needs_items"] is True and cmp_["metric_id"] == METRIC
    assert AC.comparison_record(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id=s, steered_judgment_id=js)["items_metric_id"] is None
    assert AC.comparison_record(key(), METRIC, THR, JUDGE, items_metric_id=LABEL)["items_metric_id"] is None
    assert ac.replay(cmp_) == AC.decision_record(ch)
    old = {k: v for k, v in AC.comparison_record(key(), METRIC, THR, JUDGE, needs_items=True, steered_run_id=s, steered_judgment_id=js).items() if k != "items_metric_id"}
    assert ac.replay(old) == AC.decision_record(plain)                      # a rule-5 record reads its own metric
    # the frame's digest equals the digest over the payloads an adapter emits (in any order; a whole number reads as its float; None stays)
    payloads = [{"run_id": b, "judgment_id": jb, "metric_id": LABEL, "item_id": iid, "trial_index": 0, "sample_index": 0, "turn_index": -1,
                 "value": (int(val) if val is not None else None), "label": lab} for iid, val, lab in rows_b]
    assert AC.item_rows_digest([AC.item_row_values(p) for p in reversed(payloads)]) == ch.items_digest
    assert AC.item_row_values(payloads[0]) == ("item:1", 0, 0, -1, 2.0, "full_compliance") and AC.item_rows_digest([]) is None
    assert AC.item_rows_digest([AC.item_row_values(dict(payloads[0], label="full_refusal"))] + [AC.item_row_values(p) for p in payloads[1:]]) != ch.items_digest
    # two labels exchanged under the same judgment, the reading kept: another digest, another fingerprint; the steered side unchanged
    fp0 = AC.decision_fingerprint(ac.decisions_of([cmp_]))
    fr.con.execute("UPDATE item_result SET value = 0.0, label = 'full_refusal' WHERE run_id = ? AND item_id = 'item:1'", (b,))
    fr.con.execute("UPDATE item_result SET value = 2.0, label = 'full_compliance' WHERE run_id = ? AND item_id = 'item:2'", (b,))
    fr.con.commit()
    ac1 = fr.load()
    ch1 = ac1.choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, steered_run_id=s, steered_judgment_id=js)
    assert ch1.run_id == b and ch1.measurement == ch.measurement and ch1.items_digest != ch.items_digest and ch1.steered_items_digest == ch.steered_items_digest
    assert AC.decision_fingerprint(ac1.decisions_of([cmp_])) != fp0
    # a conflict at the label metric: the candidate refused by name, the text naming the contrast's metric and the item rows' metric
    fr.conflict(b, jb, "item:3", values=(0.0, 2.0), metric=LABEL)
    ch2 = fr.load().choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, steered_run_id=s, steered_judgment_id=js)
    assert ch2.state == "no_current_anchor" and f"1 item(s) at ({METRIC}, {THR:g}; item rows at {LABEL}) under judgment {jb}" in dict(ch2.rejected)[b]
    fr.conflict(s, js, "item:1", values=(0.0, 2.0), metric=LABEL)
    ch3 = fr.load().choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, steered_run_id=s, steered_judgment_id=js)
    assert ch3.own_reason == "item_conflict" and ch3.why.startswith(f"the steered run {s}: 1 item(s) at ({METRIC}, {THR:g}; item rows at {LABEL}) under judgment {js}")
    assert AC.DECISION_RULE == "anchor_decision/7"


# ============================================================================================
# an item-paired comparison may ask the per-item CAP evidence too
# ============================================================================================

def test_an_item_paired_comparison_may_ask_the_per_item_cap_evidence_and_a_cap_flag_swap_moves_the_fingerprint_with_labels_and_reading_kept():
    """The store readers' graded mean scores an item by its label AND by whether it ran to the token cap, and the cap reading is
    filed per item as `item_flags.at_token_cap`, never on a label row -- so a cap flag exchanged between two equally labeled items of the
    unsteered store keeps its aggregate graded value and its label-row digest while the paired inputs move, and a decision over the label rows
    alone replayed quiet. The comparison asks the cap evidence too (`needs_flags`, recorded and replayed): both sides' current flags under the
    pair's judgment are digested (`flags_digest` / `steered_flags_digest`); the frame's digest equals the digest over an adapter's payloads
    (`flag_rows_digest` over `flag_row_values`); the swap moves the fingerprint with the items digests and the reading kept; nothing moved
    replays equal; a frame knowing items but no flags refuses the candidate by name; a rule-6 record carries both digests as None."""
    LABEL = "xst_label"
    fr = Frame()
    b = fr.run("none:b")
    jb = f"j:{b}:{JUDGE}:1"
    fr.item_objects()
    # two equally labeled items (item:1 and item:2, full_compliance) and a refusal; item:1 ran to the cap
    for iid, val, lab in (("item:1", 2.0, "full_compliance"), ("item:2", 2.0, "full_compliance"), ("item:3", 0.0, "full_refusal")):
        fr.con.execute("INSERT INTO item_result VALUES (?,?,?,?,?,?,?,?,?)", (b, jb, LABEL, iid, 0, 0, -1, val, lab))
    flags_b = [("item:1", 1), ("item:2", 0), ("item:3", 0)]
    fr.flags(b, jb, flags_b)
    s = fr.steered("lora:s")
    js = f"j:{s}:{JUDGE}:1"
    for iid, val, lab in (("item:1", 0.0, "full_refusal"), ("item:2", 1.0, "partial_refusal"), ("item:3", 0.0, "full_refusal")):
        fr.con.execute("INSERT INTO item_result VALUES (?,?,?,?,?,?,?,?,?)", (s, js, LABEL, iid, 0, 0, -1, val, lab))
    fr.flags(s, js, [("item:1", 0), ("item:2", 0), ("item:3", None)])
    fr.con.commit()
    ac = fr.load()
    assert ac.flags_known and ac.counts["flags_known"] == 1 and ac.flags_unknown_why == ""
    plain = ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, steered_run_id=s, steered_judgment_id=js)
    ch = ac.choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, needs_flags=True, steered_run_id=s, steered_judgment_id=js)
    _closed(ch)
    # the same anchor, reading and item digests as without the ask; both sides' flags digested on top
    assert ch.run_id == b and ch.measurement == plain.measurement and ch.items_digest == plain.items_digest and ch.steered_items_digest == plain.steered_items_digest
    assert plain.flags_digest is None and plain.steered_flags_digest is None
    assert ch.flags_digest == ac.pair_flags_digest(b, jb) and len(ch.flags_digest) == 64
    assert ch.steered_flags_digest == ac.pair_flags_digest(s, js) and len(ch.steered_flags_digest) == 64 and ch.steered_flags_digest != ch.flags_digest
    assert ac.pair_flags_digest("none:never", "j:x") is None
    # recorded on the comparison, replayed from it; never without e4 (meaningless), and a rule-6 record carries both digests as None
    cmp_ = AC.comparison_record(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, needs_flags=True, steered_run_id=s, steered_judgment_id=js)
    assert cmp_["needs_flags"] is True and cmp_["needs_items"] is True and cmp_["items_metric_id"] == LABEL
    assert AC.comparison_record(key(), METRIC, THR, JUDGE, needs_flags=True)["needs_flags"] is False
    dec = AC.decision_record(ch)
    assert ac.replay(cmp_) == dec and dec["flags_digest"] == ch.flags_digest and dec["steered_flags_digest"] == ch.steered_flags_digest
    old = {k: v for k, v in cmp_.items() if k != "needs_flags"}
    assert ac.replay(old) == AC.decision_record(plain) and ac.replay(old)["flags_digest"] is None and ac.replay(old)["steered_flags_digest"] is None
    with pytest.raises(ValueError, match="needs_items"):
        ac.choose(key(), METRIC, THR, JUDGE, needs_flags=True, steered_run_id=s, steered_judgment_id=js)
    # the frame's digest equals the digest over the payloads an adapter emits (any order; a bool or whole-number flag reads as its integer)
    payloads = [{"run_id": b, "judgment_id": jb, "item_id": iid, "trial_index": 0, "sample_index": 0, "turn_index": -1, "at_token_cap": cap} for iid, cap in flags_b]
    assert AC.flag_rows_digest([AC.flag_row_values(p) for p in reversed(payloads)]) == ch.flags_digest
    assert AC.flag_row_values(dict(payloads[0], at_token_cap=True)) == ("item:1", 0, 0, -1, 1) and AC.flag_row_values(dict(payloads[0], at_token_cap=1.0))[4] == 1
    assert AC.flag_rows_digest([]) is None and AC.flag_rows_digest([AC.flag_row_values(dict(p, at_token_cap=None)) for p in payloads]) != ch.flags_digest
    assert AC.flag_rows_digest([AC.flag_row_values(p) for p in payloads]) != AC.item_rows_digest(
        [AC.item_row_values(dict(p, metric_id=LABEL, value=None, label=None)) for p in payloads])
    # nothing moved: the same frame reloaded replays to the same fingerprint (the mirror negative)
    fp0 = AC.decision_fingerprint(ac.decisions_of([cmp_]))
    assert AC.decision_fingerprint(fr.load().decisions_of([cmp_])) == fp0
    # the cap flag exchanged between the two equally labeled items: the label rows, the reading and the steered side stand; the anchor's
    # flags digest moves, and the fingerprint with it -- while a record without the ask stays blind to the swap
    fr.con.execute("UPDATE item_flags SET at_token_cap = 0 WHERE run_id = ? AND item_id = 'item:1'", (b,))
    fr.con.execute("UPDATE item_flags SET at_token_cap = 1 WHERE run_id = ? AND item_id = 'item:2'", (b,))
    fr.con.commit()
    ac1 = fr.load()
    ch1 = ac1.choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, needs_flags=True, steered_run_id=s, steered_judgment_id=js)
    assert ch1.run_id == b and ch1.measurement == ch.measurement and ch1.items_digest == ch.items_digest and ch1.steered_items_digest == ch.steered_items_digest
    assert ch1.flags_digest != ch.flags_digest and ch1.steered_flags_digest == ch.steered_flags_digest
    assert AC.decision_fingerprint(ac1.decisions_of([cmp_])) != fp0
    assert AC.decision_fingerprint(ac1.decisions_of([old])) == AC.decision_fingerprint(ac.decisions_of([old]))
    # a frame that knows items but no flags: the candidate is refused by name on the ask (and chosen as before without it); the steered side reads None
    fr.con.execute("DROP VIEW v_item_flags_current")
    fr.con.commit()
    ac2 = fr.load()
    assert ac2.flags_known is False and ac2.flags_unknown_why == "this database does not define v_item_flags_current" and ac2.items_known
    ch2 = ac2.choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, needs_flags=True, steered_run_id=s, steered_judgment_id=js)
    assert ch2.state == "no_current_anchor" and "item cap flags are not known to this frame (this database does not define v_item_flags_current)" in dict(ch2.rejected)[b]
    assert ch2.steered_flags_digest is None and ac2.pair_flags_digest(b, jb) is None
    assert ac2.choose(key(), METRIC, THR, JUDGE, needs_items=True, items_metric_id=LABEL, steered_run_id=s, steered_judgment_id=js).run_id == b
    # a shard ships no flags either, and says so
    shard = Frame()
    shard.run("none:b")
    shard.con.executescript("CREATE TABLE reader_state_meta (schema TEXT, schema_version INTEGER); INSERT INTO reader_state_meta VALUES ('reader_state/2', 21);")
    acs = shard.load()
    assert acs.flags_known is False and acs.flags_unknown_why == "a shard ships no item flags" and acs.counts["flags_known"] == 0
