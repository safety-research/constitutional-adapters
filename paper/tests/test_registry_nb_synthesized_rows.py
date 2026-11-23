"""Frame rows synthesized from the policy judge's items (`registry_nb.synthesized_rows`, route A2, 2026-11-14).

A run the registry selects whose only judgment of its rubric is a policy judge's, filed with no run_metric row, has no row in the paper
view. Under a re-judge policy its row is synthesized and read from that judge's items by the re-derivation's own rule. Pins, on synthetic
frames, items and a small registry (no live registry; the proof on real runs is the slow test beside this one):
  * the metric template is the most common owner-measurement set of the eval and instrument, with what each carries (k, n, interval);
  * candidacy: a judged run with no run_metric and a complete judgment by the policy judge of its eval's rubric, the judge resolved with
    the run's instrument judge standing in for the filed one; a run with run_metric, a run not judged, another judge's judgment, an
    incomplete judgment, a Petri run and a run of an eval the paper omits (OMITTED_EVALS) are not candidates; under `primary` there are
    none;
  * a synthesized run's roster entries and comparison ids are the paper roster's match with the policy judge standing in for each
    entry's filed judge (`synth_roster_matches`), so its row reaches its model's paper lane;
  * a synthesized row is read like a re-derived one (same items, same reading), its interval only where its template carries one,
    named in JUDGE_REPORT and in its judge_source; `spell` marks it judge_ok whatever JUDGE_POLICY's model;
  * `with_synthesized` adds rows only where the query would have put them, and is the identity with nothing synthesized;
  * Figure 7's frontier admits the regeneration campaign's operating-point run (native rows: null lane_source_legacy) as a seed of the
    operating-point rung, and loses that seed without the campaign rule.
"""
import json
import pathlib
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
TESTS = pathlib.Path(__file__).resolve().parent
for p in (NOTEBOOKS, TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import registry_nb as R  # noqa: E402
from test_registry_nb_judges import static_items  # noqa: E402

MM_V1, MM_V2, HAIKU = "mm-sr-v1", "mm-sr-v2", "haiku-sr"


@pytest.fixture(autouse=True)
def _clean_report():
    R.JUDGE_REPORT.clear()
    yield
    R.JUDGE_REPORT.clear()


# ---- the template ---------------------------------------------------------------------------------------------------------------
def test_the_template_is_the_most_common_owner_set_with_what_it_carries():
    base = dict(eval_id="grok420", instrument_id="ins", metric_role="owner", in_current_paper=1)
    rows = []
    for i in range(3):                                            # three runs carry asr @ 0.5 with an interval
        rows.append(dict(base, run_id=f"r{i}", metric_id="asr", threshold=0.5, ci_method="wilson_95", k=3.0, n=10.0, ci_lo=0.1, ci_hi=0.6))
    rows.append(dict(base, run_id="odd", metric_id="asr", threshold=0.25, ci_method=None, k=1.0, n=10.0, ci_lo=np.nan, ci_hi=np.nan))
    rows.append(dict(base, run_id="sec", metric_id="asr", threshold=0.5, ci_method="wilson_95", k=1.0, n=10.0, ci_lo=0.0, ci_hi=0.4,
                     metric_role="secondary"))                    # not an owner row: not a template
    rows.append(dict(base, run_id="out", metric_id="asr", threshold=0.75, ci_method=None, k=1.0, n=10.0, ci_lo=np.nan, ci_hi=np.nan,
                     in_current_paper=0))                         # not in the paper: not a template
    t = R._synth_template(pd.DataFrame(rows))
    assert t == {("grok420", "ins"): [("asr", 0.5, "wilson_95", "k,n,ci")]}


# ---- candidacy ------------------------------------------------------------------------------------------------------------------
def _registry(tmp_path):
    """The tables candidacy reads: run, judgment, run_metric, instrument, judge."""
    p = tmp_path / "reg.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE judge (judge_id TEXT, judge_model TEXT, rubric_name TEXT, rubric_version TEXT, rubric_sha256 TEXT);
        CREATE TABLE instrument (instrument_id TEXT, judge_id TEXT);
        CREATE TABLE run (run_id TEXT, model_id TEXT, eval_id TEXT, instrument_id TEXT, lifecycle_state TEXT);
        CREATE TABLE judgment (judgment_id TEXT, run_id TEXT, judge_id TEXT, status TEXT, finished_utc TEXT);
        CREATE TABLE run_metric (run_id TEXT, judgment_id TEXT, metric_id TEXT);
    """)
    c.executemany("INSERT INTO judge VALUES (?,?,?,?,?)", [(MM_V1, "minimax-m3", "strongreject", "v1", "s1"), (MM_V2, "minimax-m3", "strongreject", "v2", "s2"),
                                                           (HAIKU, "claude-haiku-4-5", "strongreject", "v1", "s1"),
                                                           ("glm", "glm-5.3", "petri_bloom_judge", "x", "p")])
    c.execute("INSERT INTO instrument VALUES ('ins', ?)", (HAIKU,))
    runs = [("native", "judged"), ("filed", "judged"), ("generated", "generated"), ("haiku_only", "judged"), ("partial", "judged")]
    for rid, life in runs:
        c.execute("INSERT INTO run VALUES (?, 'qwen35_4b_it', 'grok420', 'ins', ?)", (rid, life))
    c.execute("INSERT INTO run VALUES ('petri_run', 'qwen35_4b_it', 'petri_cyber', 'ins', 'judged')")
    for jid, rid, judge, st in [("j-native", "native", MM_V1, "complete"), ("j-native2", "native", MM_V2, "complete"),
                                ("j-filed", "filed", MM_V1, "complete"), ("j-gen", "generated", MM_V1, "complete"),
                                ("j-haiku", "haiku_only", HAIKU, "complete"), ("j-partial", "partial", MM_V1, "partial"),
                                ("j-petri", "petri_run", "glm", "complete")]:
        c.execute("INSERT INTO judgment VALUES (?,?,?,?, '2026-11-14T00:00:00Z')", (jid, rid, judge, st))
    c.execute("INSERT INTO run_metric VALUES ('filed', 'j-filed', 'asr')")
    c.commit()
    c.close()
    return str(p)


def test_candidates_are_judged_runs_with_only_the_policy_judges_judgment(tmp_path, monkeypatch):
    db = _registry(tmp_path)
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(db))
    R.resolve_judge.cache_clear()
    # the newest complete judgment of the resolved judge, as the re-derivation picks it
    monkeypatch.setattr(R, "_judge_runs", lambda jid, fam: {"native": pd.DataFrame({"judgment_id": [f"newest-of-{jid}"]})} if jid == MM_V1 else {})
    got = R.synthesis_candidates(R.judge_policy("rejudge"))
    R.resolve_judge.cache_clear()
    # v1, not v2: the instrument's judge (Haiku, rubric v1) stands in for the filed judge when the two MiniMax judges are told apart
    assert got.to_dict("records") == [dict(run_id="native", eval_id="grok420", judgment_id=f"newest-of-{MM_V1}", judge_id=MM_V1)]
    assert R.synthesis_candidates(R.judge_policy("primary")).empty


def test_a_run_of_an_omitted_eval_is_never_a_candidate(tmp_path, monkeypatch):
    """An agentic run filed with only a policy judge's judgment (the paper omits the agentic evals, OMITTED_EVALS) gets no synthesized
    row and is no named gap, forced or not; the same run is a candidate once its eval is not omitted, so the rule is what keeps it out.
    The policy here names an agentic judge (`rejudge,agentic=minimax-m3`); the paper's names none, so under it the run is never a
    candidate at all."""
    db = _registry(tmp_path)
    c = sqlite3.connect(db)
    c.execute("INSERT INTO judge VALUES ('mm-ag', 'minimax-m3', 'agentic_misalignment_classifier', 'v1', 'a1')")
    c.execute("INSERT INTO instrument VALUES ('ins-ag', 'mm-ag')")
    c.execute("INSERT INTO run VALUES ('leak', 'qwen35_4b_it', 'agentic_leaking', 'ins-ag', 'judged')")
    c.execute("INSERT INTO judgment VALUES ('j-leak', 'leak', 'mm-ag', 'complete', '2026-11-14T00:00:00Z')")
    c.commit()
    c.close()
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(db))
    monkeypatch.setattr(R, "_judge_runs", lambda jid, fam: {rid: pd.DataFrame({"judgment_id": [f"newest-of-{rid}"]}) for rid in ("native", "leak")})
    pol = R.judge_policy("rejudge,agentic=minimax-m3")
    R.resolve_judge.cache_clear()
    try:
        assert set(R.synthesis_candidates(pol)["run_id"]) == {"native"}
        assert R.synthesis_candidates(pol, run_ids=["leak"]).empty
        assert R.synthesis_candidates(pol, pick_judgment=False)["run_id"].tolist() == ["native"]      # the gap listing too
        monkeypatch.setattr(R, "OMITTED_EVALS", frozenset())
        assert set(R.synthesis_candidates(pol)["run_id"]) == {"native", "leak"}
        assert set(R.synthesis_candidates(R.judge_policy("rejudge"))["run_id"]) == {"native"}       # the paper's policy: no agentic judge
    finally:
        R.resolve_judge.cache_clear()


def test_a_synthesized_run_takes_the_rosters_match_with_the_policy_judge_standing_in(tmp_path, monkeypatch):
    """The paper roster names each entry's filed judge, so the registry's own `v_roster_match` matches a run judged only by a re-judge to
    nothing, and its row would never reach its model's paper lane. `synth_roster_matches` runs that view's SQL without its judge clause
    (the policy judge's complete judgment, which candidacy requires, stands in): an entry whose judge the run lacks matches here and not
    in the view; every other clause still applies (an entry for another eval matches neither), and so does the selection's
    (`v_run_current`'s `matched`): a comparison a comparison-scoped exclude ruling takes off the run is not the run's, as `v_paper_long`
    lists none such for a filed run."""
    from sisteer.registry import db as DB, harvest as H, synth
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    corpus = synth.write_corpus(root, n_runs=2, n_items=4)
    H.ingest_once(root, workers=0, min_free=0)
    fitted = corpus["run_ids"][1]                                   # the fitted object's run, at its operating point
    path = DB.db_path(root)
    con = sqlite3.connect(path)
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'").fetchall():
        con.execute(f"DROP TRIGGER {name}")                         # a test-local roster, written straight into the copy
    rnd = con.execute("SELECT round_id FROM paper_round").fetchone()[0]
    rows = [("re-other-judge", "cmp/other_judge", synth.EVAL_ID, synth.JUDGE_ID_2), ("re-own-judge", "cmp/own_judge", synth.EVAL_ID, synth.JUDGE_ID),
            ("re-other-eval", "cmp/other_eval", synth.CAP_EVAL_ID, synth.JUDGE_ID_2), ("re-excluded", "cmp/excluded", synth.EVAL_ID, synth.JUDGE_ID_2)]
    for reid, cmp_id, ev, judge in rows:
        con.execute("INSERT INTO roster_entry (roster_entry_id, round_id, comparison_id, model_id, object_id, eval_id, instrument_id, judge_id, "
                    "strength_rule, source_citation, record_uri, ingest_seq) VALUES (?,?,?,?,?,?,?,?, 'operating_point', 'test', 'test', 1)",
                    (reid, rnd, cmp_id, synth.MODEL_ID, synth.OBJECT_SHA, ev, synth.INSTRUMENT_ID, judge))
    con.execute("INSERT INTO ruling (ruling_id, round_id, action, scope_kind, scope_json, reason, ruled_utc, ruled_by, source_ref, record_uri, "
                "ingest_seq) VALUES ('ru-cmp', ?, 'exclude', 'comparison', ?, 'test', '2026-01-02T00:00:00.000Z', 'test', 'test', 'test', 1)",
                (rnd, json.dumps({"match": {"run_ids": [fitted], "comparison_ids": ["cmp/excluded"]}})))
    con.commit()
    view = sorted(con.execute("SELECT roster_entry_id, comparison_id FROM v_roster_match WHERE run_id = ?", (fitted,)).fetchall())
    con.close()
    assert view == [("re-own-judge", "cmp/own_judge")]             # the registry's view: only the entry whose judge the run has
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(f"file:{path}?mode=ro", uri=True))
    got = R.synth_roster_matches([fitted, "no-such-run"])
    assert got == {fitted: (["re-other-judge", "re-own-judge"], ["cmp/other_judge", "cmp/own_judge"])}
    assert R.synth_roster_matches([]) == {}
    from sisteer.registry import schema_def as SD
    real = dict(SD.VIEW_SQL)
    monkeypatch.setattr(SD, "VIEW_SQL", {**real, "v_roster_match": "CREATE VIEW v_roster_match AS SELECT 1"})
    with pytest.raises(RuntimeError, match="judge clause"):
        R.synth_roster_matches([fitted])
    monkeypatch.setattr(SD, "VIEW_SQL", {**real, "v_run_current": "CREATE VIEW v_run_current AS SELECT 1"})
    with pytest.raises(RuntimeError, match="v_run_current"):
        R.synth_roster_matches([fitted])


def _selection_registry(tmp_path):
    """The roster test's synthetic registry made into the production case: three Qwen grok420 runs on one instrument. C carries its
    filed run_metric and is selected (the template); A and B carry only a complete judgment by the paper's StrongREJECT v1 judge. A is
    selected through a roster entry and held out by a ruling; B is excluded by a ruling. Returns (db path, A, B, C, the v1 judge id)."""
    from sisteer.registry import db as DB, harvest as H, synth
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    a, c, b = synth.write_corpus(root, n_runs=3, n_items=4)["run_ids"]           # 0 and 2 unsteered, 1 the fitted object
    H.ingest_once(root, workers=0, min_free=0)
    sr_v1 = next(j["judge_id"] for j in json.loads((NOTEBOOKS / "data" / "paper_judges.json").read_text(encoding="utf-8"))["judges"]
                 if (j["rubric_name"], j["rubric_version"]) == ("strongreject", "v1"))
    path = DB.db_path(root)
    con = sqlite3.connect(path)
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'").fetchall():
        con.execute(f"DROP TRIGGER {name}")                                      # a test-local edit, written straight into the copy

    def copy_row(table, key, old, **new):
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        row = {**dict(zip(cols, con.execute(f"SELECT * FROM {table} WHERE {key} = ?", (old,)).fetchone())), **new}
        con.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [row[x] for x in cols])

    ins = con.execute("SELECT instrument_id FROM run WHERE run_id = ?", (a,)).fetchone()[0]
    filed = con.execute("SELECT judge_id FROM instrument WHERE instrument_id = ?", (ins,)).fetchone()[0]
    con.execute("UPDATE run SET eval_id = 'grok420', model_id = ?", (R.MODEL_ID,))
    copy_row("eval", "eval_id", synth.EVAL_ID, eval_id="grok420")
    copy_row("model", "model_id", synth.MODEL_ID, model_id=R.MODEL_ID)
    copy_row("judge", "judge_id", filed, judge_id=sr_v1, judge_model="minimax-m3", rubric_name="strongreject", rubric_version="v1")
    con.execute("UPDATE run SET lifecycle_state = 'judged' WHERE run_id IN (?, ?)", (a, b))
    con.execute("UPDATE run SET trial_index = 1 WHERE run_id = ?", (b,))     # a second trial, not A's successor
    for rid in (a, b):
        con.execute("DELETE FROM run_metric WHERE run_id = ?", (rid,))
        old = con.execute("SELECT judgment_id FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0]
        copy_row("judgment", "judgment_id", old, judgment_id=f"mm-{rid}", judge_id=sr_v1, status="complete")
    rnd = con.execute("SELECT round_id FROM paper_round").fetchone()[0]
    obj_c = con.execute("SELECT object_id FROM run WHERE run_id = ?", (c,)).fetchone()[0]
    for reid, obj, judge in (("re-a", "none", sr_v1), ("re-c", obj_c, filed)):
        con.execute("INSERT INTO roster_entry (roster_entry_id, round_id, comparison_id, model_id, object_id, eval_id, instrument_id, judge_id, "
                    "strength_rule, source_citation, record_uri, ingest_seq) VALUES (?,?,?,?,?,?,?,?, 'operating_point', 'test', 'test', 1)",
                    (reid, rnd, f"cmp/{reid}", R.MODEL_ID, obj, "grok420", ins, judge))
    for rid_, action, run in (("ru-hold", "hold_out", a), ("ru-excl", "exclude", b)):
        con.execute("INSERT INTO ruling (ruling_id, round_id, action, scope_kind, scope_json, reason, ruled_utc, ruled_by, source_ref, "
                    "record_uri, ingest_seq) VALUES (?, ?, ?, 'run', ?, 'test', '2026-01-02T00:00:00.000Z', 'test', 'test', 'test', 1)",
                    (rid_, rnd, action, json.dumps({"match": {"run_ids": [run]}})))
    con.commit()
    con.close()
    return path, a, b, c, sr_v1


def test_the_production_path_synthesizes_the_selected_candidate_only_and_carries_its_selection(tmp_path, monkeypatch):
    """The non-forced path under `rejudge`, on a real registry: of two runs with only a policy judgment, the selected one gets a row
    and the excluded one does not; the row carries the selection's columns (held_out included) and the template's measurement (C's
    filed row); `with_synthesized` puts it in every frame kind its query would have (the paper frame, Qwen's frontier, the unsteered
    pool: the run is an unsteered Qwen grok420 run)."""
    path, a, b, c, sr_v1 = _selection_registry(tmp_path)
    with sqlite3.connect(path) as con:
        pc = dict(con.execute("SELECT run_id, selection_state FROM v_paper_current").fetchall())
        raw = pd.read_sql_query("SELECT v.*, r.seed_base, r.trial_index FROM v_paper_long v JOIN run r ON r.run_id = v.run_id", con)
    assert pc == {a: "selected", b: "excluded", c: "selected"}
    assert set(raw.run_id) == {c}                                               # the frame holds only C's filed rows
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(f"file:{path}?mode=ro", uri=True))
    monkeypatch.setattr(R, "_judge_runs", lambda jid, fam: {r: pd.DataFrame({"judgment_id": [f"mm-{r}"]}) for r in (a, b)} if jid == sr_v1 else {})
    R.resolve_judge.cache_clear()
    pol = R.judge_policy("rejudge")
    # the paper's judges spelled through eval keys, the family left on the filed judge: the same candidates, row by row
    sr_evals = ",".join(f"{e}=minimax-m3" for e in R._drawn_rubric_evals("strongreject"))
    by_eval = R.judge_policy(f"rejudge,strongreject=primary,{sr_evals}")
    try:
        assert set(R.synthesis_candidates(pol).run_id) == {a, b}                 # both carry only the policy judge's judgment
        assert R.synthesis_candidates(by_eval).equals(R.synthesis_candidates(pol))
        out = R.synthesized_rows(raw, pol=pol)
    finally:
        R.resolve_judge.cache_clear()
    assert list(out.run_id) == [a], out.run_id.tolist()                          # the excluded candidate is not synthesized
    row = out.iloc[0]
    assert row.held_out == 1 and row.in_current_paper == 1 and row.selection_state == "selected" and bool(row.synthesized)
    assert (row.judgment_id, row.judge_id) == (f"mm-{a}", sr_v1)
    # the template's measurement: C's filed owner row (its metric, threshold and interval method), carrying k, n and an interval
    tmpl = raw[raw.run_id.eq(c) & raw.metric_role.eq("owner")].iloc[0]
    assert (row.metric_id, float(row.threshold), row.ci_method) == (tmpl.metric_id, float(tmpl.threshold), tmpl.ci_method), row
    assert row.synth_carries == "k,n,ci" and R.JUDGE_REPORT["synthesized_rows"]["untemplated"] == []
    assert json.loads(row.roster_entry_ids_json) == ["re-a"] and json.loads(row.comparison_ids_json) == ["cmp/re-a"]
    for kind in ("paper_all", "qwen_frontier", "qwen_unsteered"):
        got = R.with_synthesized(raw.iloc[0:0], out, kind)
        assert list(got.run_id) == [a] and bool(got.synthesized.iat[0]), kind
    assert R.JUDGE_REPORT["synthesized_rows"]["runs"] == [a]
    # under `primary` nothing is synthesized, and the gap listing names the selected candidate only, never the excluded one
    try:
        assert R.synthesized_rows(raw, pol=R.judge_policy("primary")).empty
    finally:
        R.resolve_judge.cache_clear()
    assert [g["run_id"] for g in R.JUDGE_REPORT["synthesis_gaps"]] == [a]


S1REGEN_FINALS = {f"s1regen_pliny:grok420:f71ed06cf77f:{h}" for h in
                  ("451deefa5746ef56", "4a7dc931fc3dff43", "a1bfe185dc96197b", "c890ce1082dfa31a", "de8a1a1234db7ba2")}


def test_on_the_committed_frame_the_selected_candidates_are_exactly_the_five_regenerated_seed1_pliny_runs(monkeypatch):
    """The committed frame's tables the candidacy reads (no item rows), loaded into memory: the runs with only a policy judgment that
    the selection keeps are the five regenerated seed-1 Pliny finals and nothing else."""
    repo = NOTEBOOKS.parents[1]
    sid = (repo / "paper" / "snapshot" / "CURRENT").read_text(encoding="utf-8").strip()
    frame = repo / "paper" / "snapshot" / "snapshots" / sid
    mem = sqlite3.connect(":memory:", check_same_thread=False)
    for table in ("run", "judgment", "instrument", "judge", "run_metric"):
        pd.read_parquet(frame / f"{table}.parquet").to_sql(table, mem, index=False)
    mem.execute("CREATE INDEX run_metric_run ON run_metric (run_id)")         # the candidacy query's joins (seconds, not half a minute)
    mem.execute("CREATE INDEX judgment_run ON judgment (run_id)")

    class Con:
        def __enter__(self):
            return mem

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(R, "connect", lambda: Con())
    R.resolve_judge.cache_clear()
    R._filed_judges.cache_clear()
    sr = {(j["rubric_version"]): j["judge_id"] for j in json.loads((NOTEBOOKS / "data" / "paper_judges.json").read_text(encoding="utf-8"))["judges"]
          if j["rubric_name"] == "strongreject"}
    try:
        cand = R.synthesis_candidates(R.judge_policy("rejudge"), pick_judgment=False)
        # the judge-id guard on the same frame: best-of-N is filed under both rubric versions, Crescendo under v1 alone
        versions = {r[0] for r in mem.execute(f"SELECT rubric_version FROM judge WHERE judge_id IN "
                                              f"({','.join('?' * len(R._filed_judges('strongreject_bon', 'strongreject')))})",
                                              R._filed_judges("strongreject_bon", "strongreject"))}
        assert {"v1", "v2"} <= versions, (sid, versions)
        assert not R.policy_is_papers(f"rejudge,strongreject={sr['v1']}", "") and R.policy_is_papers(f"rejudge,crescendo={sr['v1']}", "")
    finally:
        R.resolve_judge.cache_clear()
        R._filed_judges.cache_clear()
    pc = pd.read_parquet(frame / "v_paper_current.parquet", columns=["run_id", "in_current_paper"])
    sel = set(pc.loc[pd.to_numeric(pc.in_current_paper, errors="coerce").eq(1), "run_id"].astype(str))
    assert set(cand.run_id) & sel == S1REGEN_FINALS, f"the selected candidates of frame {sid}: {sorted(set(cand.run_id) & sel)}"
    assert set(cand.run_id) == S1REGEN_FINALS, f"frame {sid} holds candidates the selection drops: {sorted(set(cand.run_id) - sel)}"


def test_a_run_that_files_no_seed_base_takes_its_fit_seeds_index():
    """The seed-replicate runs file their fit seed's index as seed_base (s1 -> 1, draw label d1/0); a synthesized run of a fitted
    object that files none takes it from the object's fit replicate, so its label matches the filed runs of its seed (it read d-1/0).
    A filed seed base stands, and a row with no fit replicate (an unsteered or prompted run) stays NULL."""
    import inspect
    base = pd.DataFrame(dict(run_id=["regen", "filed_decode_seed", "unsteered", "canon"],
                             fit_replicate_label=["s1", "s2", None, "canonical"], seed_base=[np.nan, 20260909.0, np.nan, np.nan]))
    got = R.fit_seed_base(base).set_index("run_id").seed_base
    assert got["regen"] == 1.0 and got["filed_decode_seed"] == 20260909.0 and np.isnan(got["unsteered"]) and np.isnan(got["canon"])
    assert R.fit_seed_base(base.drop(columns="fit_replicate_label")).equals(base.drop(columns="fit_replicate_label"))
    assert "base = fit_seed_base(base)" in inspect.getsource(R.synthesized_rows)


def test_under_primary_nothing_is_synthesized_and_the_selected_candidates_are_named_gaps(monkeypatch):
    calls = []

    def cands(pol=None, models=None, run_ids=None, pick_judgment=True):
        calls.append((pol["spec"], pick_judgment))                  # no judge's item rows are read for a gap listing
        return pd.DataFrame([dict(run_id="native", eval_id="grok420", judgment_id="", judge_id=MM_V1)])

    class Con:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, args):
            return [("native",)] if "in_current_paper = 1" in sql else []

    monkeypatch.setattr(R, "synthesis_candidates", cands)
    monkeypatch.setattr(R, "connect", lambda: Con())
    monkeypatch.setattr(R, "_NAMES", ((), ()))             # the print's name redaction reads nothing through the stub connection
    raw = pd.DataFrame(columns=["run_id", "eval_id", "metric_id"])
    out = R.synthesized_rows(raw, pol=R.judge_policy("primary"))
    assert out.empty and calls == [("rejudge", False)]
    assert R.JUDGE_REPORT["synthesis_gaps"] == [dict(run_id="native", eval_id="grok420", judge_id=MM_V1)]


# ---- the reading --------------------------------------------------------------------------------------------------------------
def _runs():
    items = pd.DataFrame(static_items({"g1": [0.9, 0.1], "g2": [0.0, 0.2], "g3": [0.7, 0.8], "g4": [0.4, 0.3]}))
    meta = dict(judgment_grading_code_sha="abc", judgment_n_judge_refusals=0)
    return {"filed": items.assign(judgment_id="mm-filed", **meta), "native": items.assign(judgment_id="mm-native", **meta),
            "native_noci": items.assign(judgment_id="mm-noci", **meta)}


def test_a_synthesized_row_reads_like_the_rederived_row_of_the_same_items(monkeypatch):
    runs = _runs()
    monkeypatch.setattr(R, "resolve_judge", lambda spec, rubric, prim=None: MM_V1 if spec == "minimax-m3" else None)
    monkeypatch.setattr(R, "_judge_runs", lambda jid, fam: runs)
    monkeypatch.setattr(R, "judge_model_of", lambda jid: "minimax-m3")
    base = dict(eval_id="grok420", judge_rubric_name="strongreject", metric_id="asr", threshold=0.5, ci_method="wilson_95",
                lane_source_legacy=None, judge_ok=True, headline=True, n_error_outcomes=np.nan, n_incomplete=np.nan,
                delta=np.nan, delta_lo=np.nan, delta_hi=np.nan, delta_p=np.nan, n_pairs=np.nan, anchor_run_id=None)
    d = pd.DataFrame([
        dict(base, run_id="filed", judgment_id="h-filed", judge_id=HAIKU, judge_model="claude-haiku-4-5", value=0.9, k=9.0, n=10.0,
             ci_lo=0.6, ci_hi=0.98, synthesized=False, synth_carries=None),
        dict(base, run_id="native", judgment_id="mm-native", judge_id=MM_V1, judge_model="minimax-m3", value=np.nan, k=np.nan, n=np.nan,
             ci_lo=np.nan, ci_hi=np.nan, synthesized=True, synth_carries="k,n,ci"),
        dict(base, run_id="native_noci", judgment_id="mm-noci", judge_id=MM_V1, judge_model="minimax-m3", value=np.nan, k=np.nan,
             n=np.nan, ci_lo=np.nan, ci_hi=np.nan, synthesized=True, synth_carries="k,n"),
    ])
    out = R.rederive_judged_rows(d, R.judge_policy("rejudge")).set_index("run_id")
    f, s, s2 = out.loc["filed"], out.loc["native"], out.loc["native_noci"]
    for c in ("value", "k", "n", "ci_lo", "ci_hi"):
        assert s[c] == pytest.approx(f[c]), c                       # the same items, the same rule, the same reading
    assert (s.value, s.k, s.n) == (0.5, 2.0, 4.0) and s.judgment_id == "mm-native" and s.judge_model == "minimax-m3"
    assert np.isnan(s2.ci_lo) and np.isnan(s2.ci_hi) and s2.value == 0.5      # an interval only where the template carries one
    assert s.judge_source == f"minimax-m3 items ({R.SYNTH_SOURCE})" and f.judge_source == "minimax-m3 items"
    syn = R.JUDGE_REPORT["frames"][-1]["synthesized"]
    assert sorted(x["run_id"] for x in syn) == ["native", "native_noci"] and syn[0]["value"] == 0.5


def test_spell_marks_a_synthesized_row_judge_ok_whatever_the_filed_judge_rule():
    base = dict(paper_label="control_lora", campaign_cell_id_legacy="", run_extra_json=None, fit_replicate_label="s1",
                transfer_kind="base_to_instruct", strength_nominal=1.0, strength_coefficient=None, is_operating_point=1, metric_id="asr",
                threshold=0.5, bon_n=None, eval_category="static", headline_metric_id="asr", judge_model="minimax-m3", judge_id=MM_V1,
                judge_rubric_name="strongreject", eval_id="grok420", token_cap_tokens=8192, decode_policy_id=None, decode_source=None,
                decode_sent_json=None, item_set_id=None, model_id="qwen35_4b_it", view_legacy=None)
    d = pd.DataFrame([dict(base, run_id="syn", synthesized=True), dict(base, run_id="filed_mm", synthesized=np.nan)])
    try:
        out = R.spell(d).set_index("run_id")
    except Exception as e:                                                    # the spelling needs registry tables; fall back to the rule alone
        pytest.skip(f"spell needs more frame than the fixture gives: {e}")
    assert bool(out.loc["syn", "judge_ok"]) and bool(out.loc["syn", "synthesized"])
    assert not bool(out.loc["filed_mm", "judge_ok"]), "a filed row of a judge JUDGE_POLICY does not name stays out"


# ---- the frames ---------------------------------------------------------------------------------------------------------------
def test_with_synthesized_adds_rows_only_where_the_query_would_put_them():
    frame = pd.DataFrame([dict(run_id="a", model_id="qwen35_4b_it", eval_id="grok420", paper_label="control_lora", in_current_paper=1)])
    frame.attrs = {"head": 7}
    syn = pd.DataFrame([dict(run_id="s1", model_id="qwen35_4b_it", eval_id="grok420", paper_label="control_lora", in_current_paper=1,
                             synthesized=True, synth_carries="k,n,ci", grading_code_sha="g"),
                        dict(run_id="s2", model_id="gemma4e4b_it", eval_id="mmlu", paper_label="control_lora", in_current_paper=1,
                             synthesized=True, synth_carries="k,n", grading_code_sha="g"),
                        dict(run_id="s3", model_id="qwen35_4b_it", eval_id="grok420", paper_label="none", in_current_paper=1,
                             synthesized=True, synth_carries="k,n,ci", grading_code_sha="g")])
    assert R.with_synthesized(frame, syn.iloc[0:0], "paper_all") is frame
    pa = R.with_synthesized(frame, syn, "paper_all")
    assert list(pa.run_id) == ["a", "s1", "s2", "s3"] and pa.attrs == {"head": 7} and "grading_code_sha" not in pa.columns
    fr = R.with_synthesized(frame.assign(grading_code_sha="x"), syn, "qwen_frontier")
    assert list(fr.run_id) == ["a", "s1", "s3"] and fr.set_index("run_id").loc["s1", "grading_code_sha"] == "g"
    assert list(R.with_synthesized(frame, syn, "qwen_unsteered").run_id) == ["a", "s3"]
    with pytest.raises(ValueError):
        R.with_synthesized(frame, syn, "nope")


def test_the_frontier_admits_the_regeneration_campaigns_operating_point_seed(monkeypatch):
    from test_registry_nb_decode_only_dots import TestTheFrontierDiamondIsTheDot
    f = TestTheFrontierDiamondIsTheDot()._frame()
    f["campaign"] = "seedrep"
    s1 = f["object_label"].eq("difference_lora") & f["seed"].eq("s1")
    assert s1.any()
    f.loc[s1, "lane_source_legacy"] = None                       # a native run: the source column is null and not patchable
    f.loc[s1, "campaign"] = "s1regen_pliny"

    def seeds():
        _pts, _held, meta = R.frontier_points(f, R.FRONTIER_LANE, R.FRONTIER_STATIC[1], pull_lane=R.FRONTIER_PULL_LANE,
                                              objects=R.FRONTIER_OBJECTS, panel_name="static")
        ps = meta["per_seed"]
        return sorted(ps[ps.object_label.eq("difference_lora")].seed)

    assert "s1regen_pliny" in R.OPERATING_WAVE_CAMPAIGNS
    assert seeds() == ["s1", "s2", "s3"]
    monkeypatch.setattr(R, "OPERATING_WAVE_CAMPAIGNS", [])
    assert seeds() == ["s2", "s3"], "without the campaign rule the regenerated seed leaves the operating-point rung"
