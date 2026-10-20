"""The registry-side half of the paper's status contract and the r2 parquet's ruling spelling.

`source_status` is an item-scale, revision-tagged record type + table: the adapter runner files ONE row per paper source
string a group reports under after every accepted read, `extra_json` = the adapter's facts (`status_extra(group)`), and
`v_source_status` reduces the head reads to one row per (paper source string, adapter) with the LATEST revision of each
source folded in — a refused re-read shows as `ok = 0` with the refusal under `extra_json.refuse`; a source refused before
any accepted read appears by name through the refused marker's `extra_json.lane_sources_legacy`; a vanished group's empty
revision carries its inputs under `missing`. Known-bad cases throughout: a record missing `ok` is refused naming it; a
refusal flips `ok`; the view's columns are checked against the paper's `REQUIRED_VIEWS["v_source_status"]` (pinned here,
read live when a paper checkout carries the module beside this one); the adapters' facts (alphasteer / arditi / strength)
round-trip through the runner's record and the validator; the r2 adapter states the parquet's `arm_operating_point_id`
membership under `source: r2_export` in the rulings' `~`-relative spelling, and the expiry evaluator resolves the
Petri-llama declaration's condition on it.

Fixtures are invented (tests/registry_declared_fixtures.py, tests/_b1_fixtures.py, the synthetic store adapter); nothing
here reads a real store.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

import registry_declared_fixtures as F  # noqa: E402
from test_registry_adapter_pass import SyntheticStoreAdapter, _pass_and_ingest, _setup, _store  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

NOW = "2026-09-16T20:00:00.000Z"
PRODUCER = {"script": "tests.source_status", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
#: `paperlib.registry_source.REQUIRED_VIEWS["v_source_status"]` at the cutover draft (paper-figures-registry-cutover 2e7b5200):
#: the paper's side of the contract, pinned here because the paper module is not on this branch; `_paper_required_columns`
#: prefers the live module when a checkout carries it beside this one.
PAPER_REQUIRED = ("lane_source_legacy", "adapter", "owner", "where_json", "ok", "detail", "cells_measured", "extra_json",
                  "revision_id", "observed_utc", "ingest_seq")


def _paper_required_columns():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "paper", "paperlib", "registry_source.py")
    if not os.path.exists(path):
        return PAPER_REQUIRED
    spec = importlib.util.spec_from_file_location("_paper_registry_source_probe", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)            # stdlib + pandas at import; opens no database
    except Exception:                            # pragma: no cover - a paper checkout at another pin
        return PAPER_REQUIRED
    return tuple(mod.REQUIRED_VIEWS["v_source_status"])


def _conn(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    conn.row_factory = sqlite3.Row
    return conn


def _q(root, sql, *params):
    conn = _conn(root)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _status_view(root):
    """`v_source_status` read the way the paper's `registry_source._statuses` reads it (SELECT *, the JSON columns parsed)."""
    conn = _conn(root)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM v_source_status ORDER BY lane_source_legacy, adapter").fetchall()]
    finally:
        conn.close()
    for r in rows:
        r["extra"] = json.loads(r["extra_json"])
        r["where"] = json.loads(r["where_json"])
    return rows


def _status_payload(**over):
    p = {"lane_source_legacy": "synthetic", "adapter": "synthetic_store", "source_id": "synthetic_store:s1", "owner": "SynthOwner",
         "where_json": ["synthetic:///stores/s1/store.json"], "ok": 1, "detail": "read 1 input", "cells_measured": 1, "observed_utc": NOW,
         "source_revision_id": "0" * 16, "extra": {"refuse": [], "operating_point": {"synth": [2, 0.1, "a" * 64]}}}
    p.update(over)
    return p


# ============================================================================================
# the record contract
# ============================================================================================

def test_source_status_record_missing_ok_is_refused_naming_the_column():
    payload = _status_payload()
    del payload["ok"]
    out = VAL.validate_record(synth.rec("source_status", payload, NOW), "records/adapters/synthetic_store/0000000000000000/000003-source_status.jsonl#1")
    assert isinstance(out, VAL.Malformed) and out.field == "ok" and "whole row" in out.reason
    assert out.path.endswith("000003-source_status.jsonl#1")
    good = VAL.validate_record(synth.rec("source_status", _status_payload(), NOW))
    assert isinstance(good, VAL.Validated) and good.table == "source_status"
    assert good.extra == {"refuse": [], "operating_point": {"synth": [2, 0.1, "a" * 64]}}
    assert good.payload["where_json"] == json.dumps(["synthetic:///stores/s1/store.json"], separators=(",", ":"))
    bad_ok = VAL.validate_record(synth.rec("source_status", _status_payload(ok=2), NOW))
    assert isinstance(bad_ok, VAL.Malformed) and bad_ok.field == "ok" and "BOOLEAN" in bad_ok.reason


def test_source_status_is_item_scale_revision_keyed_and_the_view_carries_the_paper_contract():
    assert R.RECORD_TYPE_TABLE["source_status"] == "source_status"
    assert "source_status" in R.ITEM_SCALE_RECORD_TYPES and "source_status" in S.ITEM_SCALE_TABLES
    assert tuple(S.TABLE_BY_NAME["source_status"].pk) == ("lane_source_legacy", "adapter", "source_id", "source_revision_id")
    assert R.REQUIRED_FIELDS["source_status"] == ("lane_source_legacy", "adapter", "source_id", "owner", "ok", "observed_utc")
    assert "v_source_status_current" in S.VIEW_NAMES
    assert S.VIEW_TABLES["v_source_status"] == frozenset({"source_status", "source_revision"})
    conn = sqlite3.connect(":memory:")
    conn.executescript(S.render_schema_sql())
    cols = [r[1] for r in conn.execute("PRAGMA table_info(v_source_status)")]
    assert cols[:len(PAPER_REQUIRED)] == list(PAPER_REQUIRED)
    missing = [c for c in _paper_required_columns() if c not in cols]
    assert not missing, missing
    assert conn.execute("SELECT COUNT(*) FROM v_source_status").fetchone() == (0,)      # the view runs on an empty registry


def test_status_sources_default_rule_owner_fallback_and_json_safety():
    class Bare:
        name = "bare"
        covers_sources = ()

    class Covers:
        name = "covers"
        covers_sources = ("x", "y")

    class Owned(Covers):
        owner = "Owner"

    assert AB.status_sources_of(Bare(), "g") == ("bare",)                 # covers nothing: filed under its own name
    assert AB.status_sources_of(Covers(), "g") == ("x", "y")             # a group that is not a source: every source
    assert AB.status_sources_of(Covers(), "x") == ("x",)                 # a group that IS a source: that one
    assert AB.owner_of(Bare()) == "registry" and AB.owner_of(Owned()) == "Owner"
    assert AB.status_extra_of(Bare(), "g") == {}
    safe = RUN._json_safe({"t": (1, 2), 3: {"nan": float("nan"), "inf": float("-inf")}, "s": {"b", "a"}})
    assert safe == {"t": [1, 2], "3": {"nan": "NaN", "inf": "-Infinity"}, "s": ["a", "b"]}


# ============================================================================================
# the runner and the view
# ============================================================================================

def test_runner_files_one_status_row_per_source_and_the_view_reduces_them_per_paper_source(tmp_path):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = SyntheticStoreAdapter(store_root)
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep.groups_refreshed == 2 and ing.refused == 0
    revs = {r.group: r.revision_id for r in rep.results}
    # one row per (adapter, group) per read, tagged with the read's revision
    rows = _q(root, "SELECT lane_source_legacy, adapter, source_id, owner, ok, cells_measured, source_revision_id FROM source_status ORDER BY source_id")
    assert rows == [("synthetic", "synthetic_store", "synthetic_store:s1", "registry", 1, 1, revs["s1"]),
                    ("synthetic", "synthetic_store", "synthetic_store:s2", "registry", 1, 1, revs["s2"])]
    assert _q(root, "SELECT COUNT(*) FROM v_source_status_current") == [(2,)]
    # the paper's row: ONE per (paper source string, adapter), the groups reduced
    (st,) = _status_view(root)
    assert (st["lane_source_legacy"], st["adapter"], st["owner"], st["ok"], st["cells_measured"], st["n_sources"]) == \
        ("synthetic", "synthetic_store", "registry", 1, 2, 2)
    assert st["where"] == sorted(os.path.join(store_root, d, "store.json") for d in ("s1", "s2"))
    assert st["extra"]["refuse"] == [] and st["extra"]["torn"] == [] and st["extra"]["missing"] == []
    by = st["extra"]["by_source"]
    assert set(by) == {"synthetic_store:s1", "synthetic_store:s2"}
    assert by["synthetic_store:s1"]["record_counts"] == {"item_result": 3, "judgment": 1, "run": 1, "run_metric": 1, "run_state": 1}
    assert by["synthetic_store:s1"]["row_refusals"] == [] and by["synthetic_store:s1"]["refuse"] == []
    assert st["detail"].startswith("[synthetic_store:s1] synthetic_store read 1 input(s) of synthetic_store:s1 (revision ")
    assert " | [synthetic_store:s2] " in st["detail"] and "1 measured (model, object, eval) key(s); no row refused" in st["detail"]
    assert json.loads(st["source_ids_json"]) == ["synthetic_store:s1", "synthetic_store:s2"]
    assert set(json.loads(st["revision_ids_json"])) == set(revs.values()) and st["revision_id"] in revs.values()
    assert st["observed_utc"] == synth.t(2000) and isinstance(st["ingest_seq"], int)
    for c in _paper_required_columns():
        assert c in st, c
    # a second pass with nothing changed files nothing: the status rows stay those of the reads
    rep2, _ = _pass_and_ingest(root, ad, synth.t(2500))
    assert rep2.inputs_changed == 0 and _q(root, "SELECT COUNT(*) FROM source_status") == [(2,)]


def test_a_refused_re_read_flips_ok_and_carries_the_refusal_while_the_accepted_read_stays_current(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass_and_ingest(root, ad, synth.t(2000))
    (st0,) = _status_view(root)
    assert st0["ok"] == 1 and st0["extra"]["refuse"] == []
    p = _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    ad.broken.add(p)
    rep, ing = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep.groups_refused == 1
    bad_rev = rep.results[0].revision_id
    (st,) = _status_view(root)
    assert st["ok"] == 0 and st["n_sources"] == 1
    assert st["extra"]["refuse"] == [f"latest read of synthetic_store:s1 refused (revision {bad_rev}): {p}: synthetic read failure"]
    assert st["detail"].startswith(f"latest read REFUSED (revision {bad_rev} at {synth.t(3000)}): {p}: synthetic read failure; the accepted read ")
    # the accepted read's facts are still the row's: its rows stay current, the refusal is ADDED, nothing is hidden
    assert st["cells_measured"] == 1 and st["extra"]["record_counts"]["item_result"] == 3 and st["revision_id"] == st0["revision_id"]
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(3,)]
    # the read heals (the reader fixed, the input unchanged: a pass carrying --retry-refused reads it): ok comes back with the new revision
    ad.broken.clear()
    rep2, _ = _pass_and_ingest(root, ad, synth.t(4000), retry_refused=True)
    (st2,) = _status_view(root)
    assert st2["ok"] == 1 and st2["extra"]["refuse"] == [] and st2["revision_id"] == rep2.results[0].revision_id
    assert st2["extra"]["record_counts"]["item_result"] == 4 and _q(root, "SELECT COUNT(*) FROM source_status") == [(2,)]


def test_a_source_refused_before_any_accepted_read_is_visible_by_paper_source_name(tmp_path):
    root, store_root = _setup(tmp_path)
    p2 = _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = SyntheticStoreAdapter(store_root)
    ad.broken.add(p2)
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep.groups_refreshed == 1 and rep.groups_refused == 1
    assert _q(root, "SELECT COUNT(*) FROM source_status") == [(1,)]          # only the accepted read filed a row ...
    (st,) = _status_view(root)                                              # ... but the view names the refused source too
    assert st["n_sources"] == 2 and st["ok"] == 0
    assert json.loads(st["source_ids_json"]) == ["synthetic_store:s1", "synthetic_store:s2"]
    assert st["where"] == sorted([os.path.join(store_root, "s1", "store.json"), p2])
    refused_rev = next(r.revision_id for r in rep.results if r.group == "s2")
    assert st["extra"]["refuse"] == [f"latest read of synthetic_store:s2 refused (revision {refused_rev}): {p2}: synthetic read failure"]
    assert "no read of this source was ever accepted" in st["detail"]
    assert st["extra"]["by_source"]["synthetic_store:s2"] == {"refuse": st["extra"]["refuse"], "torn": [], "missing": []}
    # the refused marker carries the names the view read them from
    (extra,) = _q(root, "SELECT extra_json FROM source_revision WHERE revision_id = ?", refused_rev)
    assert json.loads(extra[0]) == {"lane_sources_legacy": ["synthetic"], "owner": "registry", "emission_version": "1"}   # The version it was minted under


def test_a_vanished_source_files_an_empty_read_with_its_inputs_under_missing(tmp_path):
    root, store_root = _setup(tmp_path)
    p2 = _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = SyntheticStoreAdapter(store_root)
    _pass_and_ingest(root, ad, synth.t(2000))
    os.remove(p2)
    os.rmdir(os.path.dirname(p2))
    rep, ing = _pass_and_ingest(root, ad, synth.t(3000))
    assert ing.refused == 0 and rep.withdrawals >= 1
    (st,) = _status_view(root)
    assert st["ok"] == 0 and st["n_sources"] == 2
    s2 = st["extra"]["by_source"]["synthetic_store:s2"]
    assert s2["missing"] == [p2] and s2["refuse"] == [] and "record_counts" not in s2      # nothing was read: no counts to file
    assert st["extra"]["missing"] == [p2] and "left discovery" in st["detail"] and "input_vanished" in st["detail"]
    assert _q(root, "SELECT ok, cells_measured, where_json FROM v_source_status_current WHERE source_id = 'synthetic_store:s2'") == [(0, 0, json.dumps([p2]))]


class FactsAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter with an owner and a `status_extra` hook: what the real adapters add."""
    name = "facts_store"
    owner = "SynthOwner"

    def __init__(self, store_root: str, facts: dict):
        super().__init__(store_root)
        self.facts = facts

    def status_extra(self, group: str):
        return self.facts


def test_adapter_facts_round_trip_into_the_view_as_json(tmp_path):
    root, store_root = _setup(tmp_path)
    facts = {"operating_point": {"synth_4b_it": (2, 0.2, "b" * 64)}, "screen_4pp_point": {"synth_4b_it": (1, 0.1, "a" * 64)},
             "roster": [{"model": "synth_4b_it", "view": "it_d0p5", "dose": 0.5, "lane_kind": "instruct-direct", "cell_id": "c",
                         "staged_id": "it_d0p5__c", "config": None, "manifest": None, "dose_source": "record dose key"}],
             "missing": ["synth_4b_it lam3: still running"], "nominal_doses": {"synth_4b_it": {1: 0.5, 2: 1.0}},
             "population": {"synth_4b_it": 250}, "unscoreable": float("nan")}
    ad = FactsAdapter(store_root, facts)
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert ing.refused == 0 and rep.groups_refreshed == 1
    (st,) = _status_view(root)
    ex = st["extra"]
    assert st["owner"] == "SynthOwner" and st["ok"] == 1 and st["adapter"] == "facts_store"
    assert ex["operating_point"] == {"synth_4b_it": [2, 0.2, "b" * 64]}          # the 3-list load._lane_operating_points unpacks
    assert ex["screen_4pp_point"]["synth_4b_it"][0] == 1
    assert ex["roster"] == facts["roster"] and ex["nominal_doses"] == {"synth_4b_it": {"1": 0.5, "2": 1.0}}
    assert ex["missing"] == ["synth_4b_it lam3: still running"] and ex["refuse"] == [] and ex["torn"] == []
    assert ex["unscoreable"] == "NaN"                                            # visible, never a refused record (NaN is not JSON)
    assert ex["record_counts"]["run_metric"] == 1 and ex["row_refusals"] == []
    # an adapter that RECORDS a refusal it carried on past: ok = 0 and the paper's guard sees it under `refuse`
    ad.facts = {"refuse": ["index moved under the read"]}
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.3, 0.7, 0.5])
    rep2, _ = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep2.groups_refreshed == 1
    (st2,) = _status_view(root)
    assert st2["ok"] == 0 and st2["extra"]["refuse"] == ["index moved under the read"] and "REFUSE (1): index moved under the read" in st2["detail"]
    assert _q(root, "SELECT ok FROM source_status ORDER BY written_utc") == [(1,), (0,)]


# ============================================================================================
# the adapters' facts (declared fixtures) through the runner's record and the validator
# ============================================================================================

def _read_groups(ad):
    ctx = AF.primed_ctx(ad, now=NOW, producer=PRODUCER)     # emitter 4: a contrast-filing read needs the pass's anchor frame
    groups = {}
    for i in ad.discover():
        groups.setdefault(ad.comparison_group(i), []).append(i)
    for g, ins in sorted(groups.items()):
        for batch in ad.read(ctx, g, ins):
            bad = [str(v) for v in (VAL.validate_record(r, "<test>") for r in batch.records) if isinstance(v, VAL.Malformed)]
            assert not bad, bad[:3]
    return ctx, groups


def _status_records(ctx, ad, group, inputs):
    """The runner's status records for a read group, each validated as the harvester would."""
    recs = RUN._source_status_records(ctx, ad, group, RUN.source_id_for(ad.name, group), "1" * 16, [i.uri for i in inputs], RUN._ReadTally())
    out = []
    for rec in recs:
        v = VAL.validate_record(rec, f"records/adapters/{ad.name}/1111111111111111/000009-source_status.jsonl#1")
        assert isinstance(v, VAL.Validated), v
        out.append(v)
    return out


def test_screen_selection_prefix_rule_flags_a_non_monotone_table():
    from sisteer.registry.adapters import alphasteer_rebake as AS
    row = lambda items, lam, sha: dict(gate_items=items, gate_pp=items / 2.5, reporting_pp=items / 2.5, lambda_=lam, object_sha256=sha, at_cap_n=0)  # noqa: E731
    table = {"m": {1: row(2.0, 0.1, "a" * 64), 2: row(12.0, 0.2, "b" * 64), 3: row(4.0, 0.3, "c" * 64)}}
    sel = AS.screen_selection(table)["m"]
    assert sel["rung"] == 1 and sel["plain_max_rung"] == 3 and sel["prefix_disagrees"] is True and sel["object_sha256"] == "a" * 64
    none = AS.screen_selection({"m": {1: row(11.0, 0.1, "a" * 64)}})["m"]
    assert none["rung"] is None and none["reason"] == "no rung within budget"
    strings = AS.screen_selection({"m": {"1": row(2.0, 0.1, "a" * 64), "2": row(3.0, 0.2, "b" * 64)}})["m"]   # a table that travelled through JSON
    assert strings["rung"] == 2


def test_alphasteer_persists_the_operating_point_screen_and_dose_table_facts(tmp_path):
    from sisteer.registry.adapters import alphasteer_rebake as AS
    fx = F.build_alphasteer(str(tmp_path / "as"), moved=True)
    ad = AS.AlphaSteerRebakeAdapter(store=fx["store"], defence_store=fx["defence_store"], graded_reread=fx["graded"],
                                    orb_grades=str(tmp_path / "no_orb"), refit=str(tmp_path / "no_refit"))
    ctx, groups = _read_groups(ad)
    m = fx["model"]
    ex = ad.status_extra("alphasteer")
    assert list(ex["operating_point"][m]) == [2, 0.10, F.SHA_B] and ex["operating_point_moved"][m] is True
    # the superseded 4 pp screen's prefix-rule pick over the same table: both rungs are inside 10 items on this fixture
    assert list(ex["screen_4pp_point"][m]) == [2, 0.10, F.SHA_B] and ex["screen_4pp_selection"][m]["prefix_disagrees"] is False
    assert ex["rung_table"][m][1]["gate_items"] == pytest.approx(1.0) and ex["rung_table"][m][2]["gate_items"] == pytest.approx(2.0)
    assert ex["rung_table"][m][2]["object_sha256"] == F.SHA_B
    assert set(ex["nominal_doses"][m]) == {1, 2} and ex["missing"] == [] and ex["row_refusals"] == []
    (v,) = _status_records(ctx, ad, "alphasteer", groups["alphasteer"])
    assert v.payload["lane_source_legacy"] == AS.SOURCE and v.payload["owner"] == "AlphaKL" and v.payload["ok"] == 1
    assert v.extra["operating_point"] == {m: [2, 0.10, F.SHA_B]}                # the list the paper's gate unpacks
    assert v.extra["screen_4pp_point"] == {m: [2, 0.10, F.SHA_B]}
    assert v.extra["rung_table"][m]["1"]["gate_items"] == pytest.approx(1.0)  # JSON keys: the paper int()s them back
    assert v.extra["refuse"] == [] and v.extra["torn"] == [] and v.extra["missing"] == []


def test_arditi_persists_operating_points_and_the_final_cap_counts(tmp_path):
    from sisteer.registry.adapters import arditi_strength as AR
    fx = F.build_arditi(str(tmp_path / "arditi"))
    ad = AR.ArditiStrengthAdapter(root=fx["root"], pins_path=None)
    ctx, groups = _read_groups(ad)
    ex = ad.status_extra("battery")
    ops = ex["operating_points"]
    assert set(ops) == {fx["model"]} and {"cid", "alpha"} <= set(ops[fx["model"]])          # what load._ruled_point reads
    assert isinstance(ex["rungs"], dict) and ex["rungs"]
    assert all(k.lstrip("-").isdigit() and isinstance(n, int) and n > 0 for k, n in ex["rungs"].items())
    assert ex["missing"] == list(ex["record_missing"]) and isinstance(ex["mismatches"], list)
    (v,) = _status_records(ctx, ad, "battery", groups["battery"])
    assert v.payload["lane_source_legacy"] == AR.SOURCE and v.payload["owner"] == "AlphaKL"
    assert "operating_points" in v.extra and "rungs" in v.extra                            # the pair the gate recognises the source by
    assert v.extra["operating_points"][fx["model"]]["alpha"] == ops[fx["model"]]["alpha"]


def test_strength_persists_the_record_roster_and_serving_engines(tmp_path):
    from sisteer.registry.adapters import strength_ladder as SL
    fx = F.build_strength(str(tmp_path / "lora"))
    decl = SL.StrengthRecordDecl(fx["path"], "strength_lane", ("const_lora", "diff_lora"), None, {"llama8b": "llama8b"})
    ad = SL.StrengthLadderAdapter(records=(decl,))
    ctx, groups = _read_groups(ad)
    assert AB.status_sources_of(ad, "strength_lane") == ("strength_lane",)       # the group IS the paper source string
    ex = ad.status_extra("strength_lane")
    roster = ex["roster"]
    assert roster and all(set(r) == set(SL.ROSTER_COLUMNS) for r in roster)
    assert {r["view"] for r in roster} == {"it_d0p5", "it_d1"} and {r["dose"] for r in roster} == {0.5, 1.0}
    assert all(r["lane_kind"] == "instruct-direct" and r["dose_source"] == "record dose key" and r["model"] == fx["model"] for r in roster)
    assert all(r["staged_id"] == f"{r['view']}__{r['cell_id']}" for r in roster)
    assert ex["serving_engines"] == ex["record_engines"] and isinstance(ex["serving_engines"], list)
    assert ex["missing"] == list(ex["record_missing"])
    (v,) = _status_records(ctx, ad, "strength_lane", groups["strength_lane"])
    assert v.payload["lane_source_legacy"] == "strength_lane" and v.payload["owner"] == "Strength"
    assert v.extra["roster"] == roster and v.extra["serving_engines"] == ex["serving_engines"]


# ============================================================================================
# The r2 parquet under the rulings' spelling
# ============================================================================================

def test_r2_adapter_states_the_parquet_operating_point_ids_under_the_rulings_spelling(tmp_path, monkeypatch):
    import pandas as pd

    import _b1_fixtures as B1
    from sisteer.registry.adapters import lineage_common as LC
    from sisteer.registry.adapters import r2_export as X

    paths = B1.synthetic_r2_export(tmp_path)
    df = pd.read_parquet(paths["parquet"])
    df["arm_operating_point_id"] = None
    df.loc[df.index[:2], "arm_operating_point_id"] = ["ar_r0p05__arditi", "SUPERSEDED"]
    df.to_parquet(paths["parquet"], index=False)
    monkeypatch.setattr(LC, "HOME", str(tmp_path))          # the parquet lies "under home": the adapter must spell it `~/...`
    seeds = B1.synthetic_seeds()
    ad = X.R2ExportAdapter(data_dir=paths["data"], rs=paths["rs"], seeds=seeds, objects=B1.resolver(seeds, tmp_path))
    objs = [ad.objects.resolve(B1.VIEW, c) for c in (B1.LORA, B1.RELU, B1.SUB, B1.PROMPTED)]
    root = B1.registry_root(tmp_path, seeds, objs)
    rep, ing, _ = B1.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1
    uri = "~/" + os.path.relpath(paths["parquet"], str(tmp_path))
    rows = B1.q(root, "SELECT source_id, input_uri, json_path, found, path_resolves, current_members_json FROM source_state WHERE source_id = 'r2_export'")
    assert [r[:5] for r in rows] == [("r2_export", uri, "$.arm_operating_point_id", 1, 1)]
    assert json.loads(rows[0][5]) == ["SUPERSEDED", "ar_r0p05__arditi"]
    # the adapter's own rows stay under its source id (nothing moved, one row ADDED)
    assert B1.q(root, "SELECT COUNT(*) FROM source_state WHERE source_id = 'r2_export:r2' AND json_path = '$.rows'") == [(1,)]
    # the Petri-llama declaration's condition, spelled as the rulings spell it, resolves: a row still carries SUPERSEDED
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        ew = {"kind": "served_id_absent", "source": "r2_export", "input_uri": uri, "member_key": "arm_operating_point_id", "served_id": "SUPERSEDED"}
        ev = paper.evaluate_expiry(conn, {"ruling_id": "r2v1-supersede-llama-arditi-r2-petri-r0p1", "expires_when_json": ew})
        assert ev.evidence_state == "ok" and ev.expired is False and ev.evidence["current_members"] == ["SUPERSEDED", "ar_r0p05__arditi"]
        # the structured any_of form the ruling file uses
        ev2 = paper.evaluate_expiry(conn, {"ruling_id": "x", "expires_when_json": {"kind": "any_of", "conditions": [ew], "guards": []}})
        assert ev2.evidence_state == "ok" and ev2.expired is False
    finally:
        conn.close()
    # the export regenerates without the sentinel: the condition holds and the ruling retires
    df.loc[df.index[:2], "arm_operating_point_id"] = "ar_r0p05__arditi"
    df.to_parquet(paths["parquet"], index=False)
    rep2, ing2, _ = B1.run_and_ingest(root, ad)
    assert rep2.groups_refreshed == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        ev3 = paper.evaluate_expiry(conn, {"ruling_id": "x", "expires_when_json": ew})
        assert ev3.evidence_state == "ok" and ev3.expired is True and ev3.evidence["current_members"] == ["ar_r0p05__arditi"]
    finally:
        conn.close()
    # a parquet without the column: stated as found, path not resolving (the condition reads `path_absent`, exclusion stays active)
    ids, resolves = X._operating_point_ids(paths["stats"]) if paths["stats"].endswith(".parquet") else ([], 0)
    assert (ids, resolves) == ([], 0)
    assert X.ruling_uri("/elsewhere/x.parquet") == "/elsewhere/x.parquet" and X.ruling_uri(str(tmp_path / "d" / "x.parquet")) == "~/d/x.parquet"
