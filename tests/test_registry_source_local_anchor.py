"""The `source_local_anchor` frame rule (migration 0027).

An object-`none` run filed by a source whose `none` is its OWN standup's unsteered reading -- not the paper's unsteered reading (the
campaign's `none`: r2_export / sampled_lane / the capability readers / every source whose none the roster or a ruling admits) -- reads
selection_state `source_local_anchor`: in_current_paper 0 (never a paper row) and is_current 1 (an eligible anchor for its own source's
contrasts, `anchor_current` e1). The sources are the declarative list `vocab.SOURCE_LOCAL_ANCHOR_SOURCES` (exact spellings and GLOB
patterns, rendered into both CASEs by `schema_def.source_local_anchor_sql`); every other source an adapter declares is classified
`vocab.CAMPAIGN_NONE_SOURCES` with a one-line reason, and the first test here is the belt: a new adapter's source that is in neither
list fails loudly. The branch sits AFTER the decisive ruling, the withdrawn / lifecycle tests and the later-run rule (as a guard: a none
a ruling, its producer or a later run retired is never resurrected) and BEFORE the roster branch, so it relabels only `none` runs that
would otherwise read `selected` or `not_in_roster`.
"""
from __future__ import annotations

import fnmatch
import importlib
import inspect
import re
from dataclasses import replace

import test_registry_views_0021 as T21

from sisteer.registry import adapters as A
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import paper
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import strength_ladder as SL

LOCAL = ("source_local_anchor", 0, 1, None)
#: source spellings the adapters mint from DATA at read time (the EM root grammar's `_vendor_leak` trees; the ladder's directory names)
#: rather than declare statically: a list entry may name one of these and match no declared source
DYNAMIC_SPELLINGS = re.compile(r"^(em/agentic/[a-z0-9_]+_vendor_leak|em/agentic/[a-z0-9_]+_vendor|strength_lane_[a-z0-9_]+)$")


def _matches(source: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(source, p) if "*" in p else source == p for p in patterns)


def _declared_sources():
    """Every source the adapters declare (`covers_sources`), plus the strength ladder's dynamic spellings: its two mapped directories, its
    declared sources and a directory the map does not name (`source_for_dir` mints `strength_lane_<dir>`)."""
    out = set()
    for n in A.adapter_modules():
        mod = importlib.import_module(f"sisteer.registry.adapters.{n}")
        for cls in A.adapter_classes(mod):
            cs = inspect.getattr_static(cls, "covers_sources", ())
            if isinstance(cs, property):
                continue                                                           # strength_ladder: discovered from disk, covered below
            out.update(str(s) for s in (cs or ()))
    out.update(SL.DECLARED_SOURCES)
    out.update(SL.source_for_dir(d) for d in ("lora", "lora_base2it_native", "lora_base2it_native_mmlu", "a_directory_the_map_does_not_name"))
    return sorted(out)


# --------------------------------------------------------------------------------------------
# the classification is total over the declared sources, and the two lists are disjoint
# --------------------------------------------------------------------------------------------


def test_every_declared_source_is_classified_source_local_or_campaign_and_the_lists_are_disjoint():
    declared = _declared_sources()
    assert len(declared) > 40
    unclassified = [s for s in declared if not _matches(s, V.SOURCE_LOCAL_ANCHOR_SOURCES) and not _matches(s, V.CAMPAIGN_NONE_SOURCES)]
    assert unclassified == [], (f"a source an adapter declares is in neither vocab.SOURCE_LOCAL_ANCHOR_SOURCES nor vocab.CAMPAIGN_NONE_SOURCES: "
                                f"{unclassified}; classify its object-none runs (its own standup's unsteered reading, or the paper's) with a reason")
    both = [s for s in declared if _matches(s, V.SOURCE_LOCAL_ANCHOR_SOURCES) and _matches(s, V.CAMPAIGN_NONE_SOURCES)]
    assert both == [], both
    # every exact entry names a declared source or a spelling the adapters mint from data; every pattern matches at least one declared source
    for entry in V.SOURCE_LOCAL_ANCHOR_SOURCES + V.CAMPAIGN_NONE_SOURCES:
        if "*" in entry:
            assert any(fnmatch.fnmatchcase(s, entry) for s in declared), entry
        else:
            assert entry in declared or DYNAMIC_SPELLINGS.match(entry), entry
    # the ladder's dynamic spelling is matched by prefix: a new ladder directory needs no list entry
    assert _matches(SL.source_for_dir("a_directory_the_map_does_not_name"), V.SOURCE_LOCAL_ANCHOR_SOURCES)
    # the census' eight and the coordinator's three groups are source-local; the campaign's own unsteered readings are not
    for s in ("alphasteer_rebake", "arditi_strength", "arditi_allpos", "strength_lane", "strength_lane_base2it_native", "strength_lane_base2it_native_mmlu",
              "strength_lane_lora_base2it_native_gemma_hidose", "em/agentic/qwen35_4b_allpos", "arditi_vendor_mt", "strength_mtdiff_s1",
              "em/agentic/gemma4e4b_bl", "em/agentic/llama8b_bl", "em/agentic/gemma4e4b_bl__lam3-superseded", "em/agentic/llama8b_bl__r0p1-superseded",
              "em/agentic/qwen35_4b_bl_vendor_leak"):
        assert _matches(s, V.SOURCE_LOCAL_ANCHOR_SOURCES), s
    for s in ("r2_export", "sampled_lane", "bfcl_full", "bigcodebench_full", "llama8b_lane", "relu_paired", "multiturn_grades", "em/agentic/gemma4e4b",
              "em/agentic/qwen35_4b_it", "bigcodebench_full_shared", "seedrep_battery"):
        assert _matches(s, V.CAMPAIGN_NONE_SOURCES) and not _matches(s, V.SOURCE_LOCAL_ANCHOR_SOURCES), s
    assert "source_local_anchor" in V.SELECTION_STATES and "source_local_anchor" not in S.NON_CURRENT_SELECTION_STATES
    assert "lane" not in [seg for st in V.SELECTION_STATES for seg in st.split("_")]


def test_the_rule_is_rendered_from_the_list_after_the_decisive_and_later_run_branches_and_before_the_roster_branch():
    sql = S.source_local_anchor_sql("r")
    assert sql == S.SOURCE_LOCAL_ANCHOR_SQL and sql.startswith("(r.object_id = 'none' AND (")
    for entry in V.SOURCE_LOCAL_ANCHOR_SOURCES:
        assert (f"r.lane_source_legacy GLOB '{entry}'" if "*" in entry else f"'{entry}'") in sql, entry
    # the guards: a none a later run, its producer or a ruling retired is never resurrected (the decisive branches precede this one)
    assert sql.endswith(" AND l.run_id IS NULL AND r.lifecycle_state <> 'superseded' AND r.superseded_by_run_id IS NULL)")
    sel, pap = S.SELECTION_STATE_SQL, S.IN_CURRENT_PAPER_SQL
    at = sel.index(f"WHEN {sql} THEN 'source_local_anchor'")
    assert sel.index("WHEN d.action = 'exclude' THEN 'excluded'") < at < sel.index("WHEN pm.run_id IS NULL THEN 'not_in_roster'")
    assert sel.count("THEN 'source_local_anchor'") == 1
    pat = pap.index(f"WHEN {sql} THEN 0")
    assert pap.index("WHEN d.action = 'include' THEN 1") < pat < pap.index(f"WHEN {S.MEMBER_LABEL_SQL} THEN 0")
    squash = lambda t: " ".join(t.split())  # noqa: E731
    assert squash(S.VIEW_SQL["v_run_current"]).count(squash(sql)) == 2


# --------------------------------------------------------------------------------------------
# the rule on a synthetic corpus
# --------------------------------------------------------------------------------------------


def _world(tmp_path, local_source="arditi_strength", campaign_source="r2_export"):
    """The synthetic corpus (its `none` and its steered run under `synth_source`, both roster-matched through the baseline entry) plus one
    complete `none` run of a source-local source and one of a campaign source at the same key, keyed apart by their producer keys."""
    root, ids = T21._corpus_with_baseline_entry(tmp_path)
    local, recs = synth.run_records("local_none", 700, "none", "none", 4, lane_source_legacy=local_source, view_legacy="tagA")
    T21._write(root, "local", "000700-local.jsonl", recs)
    campaign, recs = synth.run_records("campaign_none", 800, "none", "none", 4, lane_source_legacy=campaign_source)
    T21._write(root, "campaign", "000800-campaign.jsonl", recs)
    return root, ids, local, campaign


def _state(rc, rid):
    return (rc[rid]["selection_state"], rc[rid]["in_current_paper"], rc[rid]["is_current"], rc[rid]["why"])


def test_a_source_local_none_reads_source_local_anchor_and_a_campaign_none_is_unchanged(tmp_path):
    """Both nones are complete and roster-matched (the baseline entry names object none). The campaign source's none and the corpus's own
    read as before -- but the three nones share one measurement key, so the two earlier ones are superseded by the later run (the campaign none,
    filed last): the source-local none, filed between them, reads `superseded` by the later-run rule, never `source_local_anchor` -- the
    rule resurrects nothing. On its own key (a ruling that lets it coexist) it reads source_local_anchor / 0 / 1 / NULL; `v_paper_current`
    projects the same; the paper's renderer leaves it out while the registry's frame still carries its rows."""
    root, ids, local, campaign = _world(tmp_path)
    corpus_none, lora_run = ids["run_ids"]
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    assert _state(rc, campaign) == ("selected", 1, 1, None)
    assert _state(rc, local) == ("superseded", 0, 0, "selection_state superseded (a later run at the same key)")
    assert _state(rc, corpus_none) == ("superseded", 0, 0, "selection_state superseded (a later run at the same key)")
    conn.close()
    # let the three coexist: the source-local none is then a current, anchor-eligible, non-paper row
    T21._write(root, "roster", "000002-coexist.jsonl", [T21._ruling("R_co", "coexist", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=60)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc, pc = T21._by_run(conn, "v_run_current"), T21._by_run(conn, "v_paper_current")
    assert _state(rc, local) == LOCAL and rc[local]["roster_entry_ids_json"] != "[]" and rc[local]["decisive_action"] is None
    assert (pc[local]["selection_state"], pc[local]["in_current_paper"], pc[local]["is_current"]) == ("source_local_anchor", 0, 1)
    assert _state(rc, campaign) == ("selected", 1, 1, None) and _state(rc, corpus_none) == ("selected", 1, 1, None)
    assert _state(rc, lora_run) == ("selected", 1, 1, None)
    assert paper.in_current_paper(conn, local)[:2] == (0, "source_local_anchor")
    assert local not in set(paper.paper_long_frame(conn, round_id="r_synth")["run_id"])          # not a paper row
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE run_id = ?", (local,)).fetchone()[0] >= 1   # the frame keeps its rows
    conn.close()


def test_a_roster_less_source_local_none_reads_source_local_anchor_and_an_incomplete_one_too(tmp_path):
    """Without a baseline entry the corpus's `none` reads not_in_roster as before (its source is on neither list), the source-local none
    reads source_local_anchor (the branch precedes the roster branch), and so does an unfinished source-local none (the branch precedes
    the incomplete-lifecycle branch, as the member branch does; is_current 1, as not_in_roster read)."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    corpus_none, _ = ids["run_ids"]
    T21._write(root, "roster", "000001-roster.jsonl", [T21._roster_entry(1)])
    local, recs = synth.run_records("local_none", 700, "none", "none", 4, lane_source_legacy="strength_lane_base2it_native", instrument_id=synth.INSTRUMENT_ID)
    T21._write(root, "local", "000700-local.jsonl", recs)
    pending, recs = synth.run_records("local_pending", 760, "none", "none", 4, lane_source_legacy="strength_lane", through="generated",
                                      store_uri="synthetic:///store/local_pending")
    T21._write(root, "local", "000760-pending.jsonl", recs)
    T21._write(root, "roster", "000002-coexist.jsonl", [T21._ruling("R_co", "coexist", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=60)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    assert _state(rc, corpus_none) == ("not_in_roster", 0, 1, None)
    assert _state(rc, local) == LOCAL and rc[local]["roster_entry_ids_json"] == "[]"
    assert _state(rc, pending) == LOCAL
    conn.close()


def test_a_decisive_ruling_a_producer_retirement_or_a_later_run_still_wins_over_the_frame_rule(tmp_path):
    """A key-scoped exclude on the source-local none: `excluded`, is_current 0, why names the ruling (the decisive branch precedes the frame
    rule); a supersede ruling: `superseded`; an include: `selected` and in the paper (an include beats the frame rule, as it beats the
    roster); and a later run of the same source at the key retires the earlier none by the later-run rule, never resurrected."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    T21._write(root, "roster", "000001-roster.jsonl", [T21._roster_entry(1), T21._roster_entry(2, object_id="none", comparison="baseline")])
    T21._write(root, "roster", "000002-coexist.jsonl", [T21._ruling("R_co", "coexist", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=55)])
    local, recs = synth.run_records("local_none", 700, "none", "none", 4, lane_source_legacy="alphasteer_rebake", view_legacy="tagA")
    T21._write(root, "local", "000700-local.jsonl", recs)
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _state(T21._by_run(conn, "v_run_current"), local) == LOCAL
    conn.close()
    T21._write(root, "roster", "000003-exclude.jsonl", [T21._ruling("R_ex", "exclude", "run", {"run_id": local}, i=60)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _state(T21._by_run(conn, "v_run_current"), local) == ("excluded", 0, 0, "selection_state excluded (ruling R_ex)")
    conn.close()
    T21._write(root, "roster", "000004-supersede.jsonl", [T21._ruling("R_sup", "supersede", "run", {"run_id": local}, i=70, target_json={"superseded_by": "later"})])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _state(T21._by_run(conn, "v_run_current"), local) == ("superseded", 0, 0, "selection_state superseded (ruling R_sup)")
    conn.close()
    T21._write(root, "roster", "000005-include.jsonl", [T21._ruling("R_in", "include", "run", {"run_id": local}, i=80)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _state(T21._by_run(conn, "v_run_current"), local) == ("selected", 1, 1, None)
    conn.close()
    # a later same-source none at the key, both roster-matched through the baseline entry and no coexist ruling: the later-run rule retires
    # the earlier one (superseded, is_current 0) and the frame rule reads only the later -- the guard, not a resurrection
    root2 = str(tmp_path / "reg2")
    H.ensure_layout(root2)
    synth.write_corpus(root2, n_runs=1, n_items=4)
    T21._write(root2, "roster", "000001-roster.jsonl", [T21._roster_entry(1), T21._roster_entry(2, object_id="none", comparison="baseline")])
    first, recs = synth.run_records("alphasteer_none_first", 700, "none", "none", 4, lane_source_legacy="alphasteer_rebake", view_legacy="tagA")
    T21._write(root2, "local", "000700-first.jsonl", recs)
    later, recs = synth.run_records("alphasteer_none_later", 900, "none", "none", 4, lane_source_legacy="alphasteer_rebake", view_legacy="tagB")
    T21._write(root2, "local", "000900-later.jsonl", recs)
    T21._ingest(root2)
    conn = DB.connect(DB.db_path(root2), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    assert _state(rc, later) == LOCAL
    assert _state(rc, first) == ("superseded", 0, 0, "selection_state superseded (a later run at the same key)"), "never resurrected by the frame rule"
    conn.close()


def test_a_source_local_none_is_an_eligible_anchor_for_its_own_sources_contrasts(tmp_path):
    """`anchor_current` e1 reads is_current: the source-local none (in no paper) is a candidate for a steered row of its source at the exact
    key -- the same-source tier picks it (`sibling_same_source`, kind sibling_none; the own-anchor path keeps it verbatim) -- and the
    paper's roster-matched none of another source is NOT preferred over it. An excluded one is rejected with the fold's why."""
    root, ids, local, campaign = _world(tmp_path)
    T21._write(root, "roster", "000002-coexist.jsonl", [T21._ruling("R_co", "coexist", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=60)])
    T21._ingest(root)
    with DB.reading(DB.db_path(root)) as conn:
        assert _state(T21._by_run(conn, "v_run_current"), local) == LOCAL
        ac = AC.AnchorCurrency.load(conn)
        assert ac.is_current(local) is True
        key = AC.AnchorKey(model_id=synth.MODEL_ID, eval_id=synth.EVAL_ID, instrument_id=synth.INSTRUMENT_ID, token_cap_tokens=512, trial_index=0,
                           pass_label=None, decode_policy_id=None, lane_source_legacy="arditi_strength", standup={"view_legacy": "tagA"})
        ch = ac.choose(key, "asr", 0.5, synth.JUDGE_ID)
        assert ch.state == "sibling_same_source" and ch.run_id == local and ch.anchor_kind in ("same_standup_none", "sibling_none"), ch.selection_json()
        own = ac.choose(replace(key, own_anchor_run_id=local, own_anchor_kind="same_standup_none"), "asr", 0.5, synth.JUDGE_ID)
        assert own.state == "own_current" and own.run_id == local and own.anchor_kind == "same_standup_none"
    T21._write(root, "roster", "000003-exclude.jsonl", [T21._ruling("R_ex", "exclude", "run", {"run_id": local}, i=70)])
    T21._ingest(root)
    with DB.reading(DB.db_path(root)) as conn:
        ac = AC.AnchorCurrency.load(conn)
        assert ac.is_current(local) is False
        ch = ac.choose(key, "asr", 0.5, synth.JUDGE_ID)
        assert ch.run_id != local and (local, "selection_state excluded (ruling R_ex)") in ch.rejected
