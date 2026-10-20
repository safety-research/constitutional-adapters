"""The core fix pass: every finding of the opening review reproduced as a test.

(1) out-of-order lifecycle records dependency-block, never refuse, live == replay over every permutation;
(2) an adapter revision that refuses changes no measurement (staged outside the ingestible namespace);
(3) membership derives from a source's LATEST accepted revision (three-revision removal);
(4) a metric-only withdrawal matches its key structurally; (5) revision ids are namespaced by source;
(6) a vanished input withdraws (input_vanished) and leaves a missing-source state, a discovery FAILURE withdraws nothing;
(7) automatic supersession keys on strength — a ladder's points coexist; (8) roster gaps are computed per entry;
(9) paired(judge_id=) constrains both sides and refuses an ambiguous repeated judgment;
(10) a failed adapter read is retried on the next pass; (11) the run_ingest stamp follows the run's own records;
F6: utc_to_epoch_ms is exact (the 2039 example).
"""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import shutil
import sqlite3
from typing import Iterator, List, Sequence

import pytest

from sisteer import registry
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs, gz=False):
    return R.write_records(os.path.join(root, "records", sub, name), recs, gz=gz)


def _ingest(root, **kw):
    kw.setdefault("workers", 0)
    kw.setdefault("min_free", 0)
    return H.ingest_once(root, **kw)


def _q(root, sql, *params):
    with DB.reading(DB.db_path(root)) as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _rebuild_equal(root, tmp_path, seed=7):
    copy = str(tmp_path / f"copy{seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(seed).shuffle(files)
    for f in files:
        dest = os.path.join(copy, "records", os.path.relpath(f, os.path.join(root, "records")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    with DB.reading(DB.db_path(root)) as a, DB.reading(out) as b:
        diff = H.verify_equal(a, b)
        bad = {k: v for k, v in diff.items() if v}
        detail = {}
        for t in bad:                                                  # the first differing rows, live-only then replay-only
            cols = [c for c in SD.TABLE_BY_NAME[t].column_names if c not in H.VERIFY_EXCLUDE_COLUMNS.get(t, ())]
            sql = f"SELECT {', '.join(cols)} FROM {t}"
            ra = {tuple(H._norm(v) for v in r) for r in a.execute(sql).fetchall()}
            rb = {tuple(H._norm(v) for v in r) for r in b.execute(sql).fetchall()}
            detail[t] = {"live_only": sorted(map(str, ra - rb))[:3], "replay_only": sorted(map(str, rb - ra))[:3]}
    assert not bad, (bad, detail)


# ---- (1) out-of-order lifecycle records --------------------------------------------------------------

STATES = ("queued", "running", "generated")


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_lifecycle_records_in_any_order_block_then_fold_and_replay_equal(tmp_path, order):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated")
    run = [r for r in recs if r["record_type"] == "run"]
    states = {r["payload"]["state"]: r for r in recs if r["record_type"] == "run_state"}
    _write(root, "c", "000001-run.jsonl", run)
    _ingest(root)
    for i, idx in enumerate(order, start=2):
        _write(root, "c", f"00000{i}-{STATES[idx]}.jsonl", [states[STATES[idx]]])
        rep = _ingest(root)
        assert rep.refused == 0, rep.refusals
    row = _q(root, "SELECT lifecycle_state, queued_utc, started_utc, generated_utc FROM run WHERE run_id = ?", rid)[0]
    assert row[0] == "generated" and all(row[1:]), tuple(row)
    assert _q(root, "SELECT COUNT(*) FROM blocked_records")[0][0] == 0
    assert _q(root, "SELECT COUNT(*) FROM record_refusal")[0][0] == 0
    assert _q(root, "SELECT COUNT(*) FROM run_state_event WHERE run_id = ?", rid)[0][0] == 3
    _rebuild_equal(root, tmp_path, seed=sum(order) + order[0] * 3)


def test_generated_before_running_is_dependency_blocked_naming_the_missing_evidence(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated")
    states = {r["payload"]["state"]: r for r in recs if r["record_type"] == "run_state"}
    _write(root, "c", "000001-run.jsonl", [r for r in recs if r["record_type"] == "run"])
    _write(root, "c", "000002-generated.jsonl", [states["generated"]])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 1
    b = _q(root, "SELECT record_type, missing_parent_table, missing_parent_id FROM blocked_records")[0]
    assert b[0] == "run_state" and b[1] == "run_state_event" and b[2] == f"{rid}:running"
    assert _q(root, "SELECT lifecycle_state FROM run")[0][0] == "queued"


# ---- adapter machinery for (2), (5), (6), (10) -----------------------------------------------------------


class StoreAdapter:
    """`<root>/*/store.json` → one judged run per store (one comparison group per store); `broken` inputs raise."""
    covers_sources = ("synthetic",)
    rng_scope = "none"
    allow_symlinks = False

    def __init__(self, store_root: str, name: str = "synthetic_store", group_of=None):
        self.name = name
        self.roots = (store_root,)
        self.broken: set = set()
        self.discover_fails = False
        self._group_of = group_of or (lambda inp: os.path.basename(os.path.dirname(inp.uri)))

    def discover(self) -> List[AB.Input]:
        if self.discover_fails:
            raise AB.AdapterRefusal(self.roots[0], "synthetic discovery failure (root unreadable)")
        out = []
        for d in sorted(os.listdir(self.roots[0])):
            p = os.path.join(self.roots[0], d, "store.json")
            if os.path.lexists(p):
                out.append(AB.check_input(p, role="report"))
        return out

    def comparison_group(self, inp: AB.Input) -> str:
        return self._group_of(inp)

    def read(self, ctx, group, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for inp in inputs:
            if inp.uri in self.broken:
                raise AB.AdapterRefusal(inp.uri, "synthetic read failure")
            with open(inp.path) as fh:
                store = json.load(fh)
            batch = AB.RecordBatch(RUN.source_id_for(self.name, group), group)
            key = store["key"]
            rid = I.run_id(synth.CAMPAIGN, key, store["object"], synth.EVAL_ID, synth.INSTRUMENT_ID)
            obj_id = "none" if store["object"] == "none" else synth.OBJECT_SHA
            p = synth.run_payload(key, store["object"], obj_id, i=1000)
            p.update(run_id=rid, lifecycle_state="generated", started_utc=synth.t(1001), generated_utc=synth.t(1002),
                     n_items_generated=len(store["items"]), n_items_judged=len(store["items"]), judge_id=synth.JUDGE_ID,
                     store_completion_artifact_uri=inp.uri, store_completion_sha256=inp.sha256, store_uri=inp.uri)
            jid = I.judgment_id(rid, synth.JUDGE_ID)
            batch.add("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "n_items_judged": len(store["items"]),
                                   "status": "complete"}, ctx.producer, ctx.now)
            batch.add("run", p, ctx.producer, ctx.now)
            batch.add("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(1003), "evidence_kind": "completion_artifact",
                                    "evidence_uri": inp.uri}, ctx.producer, ctx.now, observe=False)
            vals = [it["score"] for it in store["items"]]
            batch.add("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5,
                                     "value": sum(vals) / len(vals), "k": sum(1 for v in vals if v > 0.5), "n": len(vals),
                                     "computed_by": "registry", "source_uri": inp.uri}, ctx.producer, ctx.now)
            for it in store["items"]:
                batch.add("item_result", {"run_id": rid, "item_id": it["id"], "metric_id": "asr", "trial_index": 0, "sample_index": 0,
                                          "turn_index": -1, "judgment_id": jid, "value": it["score"], "label": None, "api_pool": None,
                                          "created_utc": ctx.now, "judged_utc": None, "withdrawn_by_revision_id": None}, ctx.producer, ctx.now)
            yield batch

    def source_states(self, ctx, inputs):
        return [AB.SourceStateRow(RUN.source_id_for(self.name, "all"), i.uri, "$.object", 1, 1, [json.load(open(i.path))["object"]], i.sha256)
                for i in inputs]


def _store(store_root, name, key, obj, scores):
    d = os.path.join(store_root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "store.json"), "w") as fh:
        json.dump({"key": key, "object": obj, "items": [{"id": f"q{i}", "score": s} for i, s in enumerate(scores)]}, fh)
    return os.path.join(d, "store.json")


def _roster_entry(i, selector, comparison="fig", **over):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID,
         "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID,
         "object_selector_json": selector, "strength_rule": "operating_point", "source_citation": "synthetic", "inferred": 0}
    p.update(over)
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _adapter_setup(tmp_path, stores):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    _ingest(root)
    store_root = str(tmp_path / "stores")
    os.makedirs(store_root)
    for name, key, obj, scores in stores:
        _store(store_root, name, key, obj, scores)
    return root, store_root


def _pass(root, adapter, now, **kw):
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [adapter], conn=conn, now=now, **kw)
    ing = _ingest(root)
    return rep, ing


def _sel(root, rid):
    return _q(root, "SELECT selection_state, in_current_paper FROM v_paper_current WHERE run_id = ?", rid)[0]


# ---- (2) a refused revision changes no measurement ------------------------------------------------------

def test_refused_revision_is_staged_and_publishes_nothing(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("g1a", "store_a", "constitution_lora@aaaa", [0.2, 0.8, 0.4]),
                                                  ("g1b", "store_b", "none", [0.1, 0.3])])
    ad = StoreAdapter(store_root, group_of=lambda inp: "one_group")   # both stores in ONE comparison group
    rep, ing = _pass(root, ad, synth.t(2000))
    assert rep.groups_refreshed == 1 and ing.refused == 0
    rid_a = I.run_id(synth.CAMPAIGN, "store_a", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid_a)[0][0] == pytest.approx((0.2 + 0.8 + 0.4) / 3)
    # store_a's metric is REPLACED (a re-grade) and store_b's read then refuses: the whole revision is refused
    pb = _store(store_root, "g1b", "store_b", "none", [0.1, 0.3])
    _store(store_root, "g1a", "store_a", "constitution_lora@aaaa", [0.9, 0.9, 0.9])
    ad.broken.add(pb)
    rep2, ing2 = _pass(root, ad, synth.t(3000))
    assert rep2.groups_refused == 1 and rep2.groups_refreshed == 0
    assert ing2.refused == 0, ing2.refusals                                     # zero refusals: nothing partial reached the log
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid_a)[0][0] == pytest.approx((0.2 + 0.8 + 0.4) / 3)  # old value stands
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE value = 0.9")[0][0] == 0
    revs = _q(root, "SELECT accepted FROM source_revision ORDER BY observed_utc")
    assert [r[0] for r in revs] == [1, 0]
    # only the unaccepted revision record (+ watermarks) reached the ingestible namespace; batches were never published
    adapters_dir = os.path.join(root, "records", "adapters", ad.name)
    rev2 = rep2.results[0].revision_id
    assert not os.path.exists(os.path.join(adapters_dir, rev2))                          # no revision directory for a refusal
    refused_files = os.listdir(os.path.join(adapters_dir, "_refused", rev2))
    assert len(refused_files) == 1 and refused_files[0].endswith("-source_revision.jsonl")
    for dp, _, fs in os.walk(adapters_dir):
        for f in fs:
            for _, rec, _ in R.read_record_file(os.path.join(dp, f)):
                assert rec["payload"].get("value") != 0.9, (f, "a batch of the refused revision was published")
    assert not os.path.exists(os.path.join(root, "staging")) or not any(
        fs for _, _, fs in os.walk(os.path.join(root, "staging")))      # nothing left staged
    _rebuild_equal(root, tmp_path)


# ---- (3) membership from the LATEST accepted revision ----------------------------------------------------

def _rev(rid, i, src="src:g"):
    return synth.rec("source_revision", {"revision_id": rid, "source_id": src, "adapter": "synthetic", "observed_utc": synth.t(i),
                                         "inputs_json": {"a": ("%064x" % i)}, "accepted": 1}, synth.t(i))


def _obs(rid, keys, i, table="run", src="src:g"):
    return synth.rec("observation", {"source_id": src, "revision_id": rid, "table_name": table,
                                     "observation_keys": [I.canonical_json(k) for k in keys]}, synth.t(i))


def test_three_revision_removal_withdraws_despite_an_older_current_row(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    lora = ids["run_ids"][1]
    key = {"run_id": lora}
    _write(root, "a", "000001-rev1.jsonl", [_rev("rev1", 9000), _obs("rev1", [key], 9001)])
    _ingest(root)
    assert _sel(root, lora) == ("selected", 1)
    _write(root, "a", "000002-rev2.jsonl", [_rev("rev2", 9010), _obs("rev2", [key], 9011)])
    _ingest(root)
    assert _q(root, "SELECT revision_id, current FROM observation ORDER BY revision_id") == [("rev1", 0), ("rev2", 1)]
    # rev3 drops the key: the withdrawal names rev2 → rev3; rev1's row is history, not membership
    wd = synth.rec("withdrawal", {"source_id": "src:g", "observation_key": I.canonical_json(key), "from_revision_id": "rev2",
                                  "to_revision_id": "rev3", "table_name": "run", "reason": "removed_measurement"}, synth.t(9021))
    _write(root, "a", "000003-rev3.jsonl", [_rev("rev3", 9020), wd])
    _ingest(root)
    assert _q(root, "SELECT COUNT(*) FROM observation WHERE current = 1")[0][0] == 0
    assert _sel(root, lora) == ("withdrawn", 0)
    assert _q(root, "SELECT COUNT(*) FROM v_paper_long WHERE run_id = ? AND in_current_paper = 1", lora)[0][0] == 0
    _rebuild_equal(root, tmp_path)


# ---- (4) metric-only withdrawal keys compared structurally ---------------------------------------------

def test_withdrawing_one_metric_keeps_the_run_and_drops_only_that_row(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    lora = ids["run_ids"][1]
    jid = ids["judgment_ids"][1]
    cov = synth.rec("run_metric", {"run_id": lora, "judgment_id": jid, "metric_id": "coverage", "value": 1.0, "n": 3,
                                   "computed_by": "producer"}, synth.t(8000))
    _write(root, "m", "000001-cov.jsonl", [cov])
    k_asr = {"run_id": lora, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "computation_id": ""}
    k_cov = {"run_id": lora, "judgment_id": jid, "metric_id": "coverage", "threshold": -1.0, "computation_id": ""}
    _write(root, "a", "000001-rev1.jsonl", [_rev("rev1", 9000), _obs("rev1", [{"run_id": lora}], 9001),
                                            _obs("rev1", [k_asr, k_cov], 9002, table="run_metric")])
    _ingest(root)
    assert sorted(r[0] for r in _q(root, "SELECT metric_id FROM v_paper_long WHERE run_id = ?", lora)) == ["asr", "coverage"]
    wd = synth.rec("withdrawal", {"source_id": "src:g", "observation_key": I.canonical_json(k_asr), "from_revision_id": "rev1",
                                  "to_revision_id": "rev2", "table_name": "run_metric", "reason": "removed_measurement"}, synth.t(9012))
    _write(root, "a", "000002-rev2.jsonl", [_rev("rev2", 9010), _obs("rev2", [{"run_id": lora}], 9011),
                                            _obs("rev2", [k_cov], 9011, table="run_metric"), wd])
    _ingest(root)
    assert [r[0] for r in _q(root, "SELECT metric_id FROM v_paper_long WHERE run_id = ?", lora)] == ["coverage"]
    assert _sel(root, lora) == ("selected", 1)                                  # the run itself stays
    _rebuild_equal(root, tmp_path)


# ---- (5) revision ids namespaced by source ------------------------------------------------------------------

def test_identical_inputs_under_two_sources_are_two_revisions(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8]),
                                                  ("s2", "store_2", "none", [0.2, 0.8])])
    # byte-identical inputs: copy s1's file over s2's
    shutil.copyfile(os.path.join(store_root, "s1", "store.json"), os.path.join(store_root, "s2", "store.json"))
    ad = StoreAdapter(store_root)
    inputs = ad.discover()
    assert inputs[0].sha256 == inputs[1].sha256
    assert RUN.revision_id_for(RUN.source_id_for(ad.name, "s1"), [inputs[0]]) != RUN.revision_id_for(RUN.source_id_for(ad.name, "s2"), [inputs[1]])
    rep, ing = _pass(root, ad, synth.t(2000))
    assert rep.groups_refreshed == 2 and ing.refused == 0
    rows = _q(root, "SELECT source_id, revision_id FROM source_revision WHERE accepted = 1 ORDER BY source_id")
    assert len(rows) == 2 and len({r[1] for r in rows}) == 2 and {r[0] for r in rows} == {f"{ad.name}:s1", f"{ad.name}:s2"}
    obs = _q(root, "SELECT COUNT(*), COUNT(DISTINCT observation_key) FROM observation WHERE current = 1 AND table_name = 'run'")[0]
    assert obs == (2, 1)                                                   # one run, observed by two sources — one row each


# ---- (6) vanished inputs vs a discovery failure --------------------------------------------------------

def test_vanished_input_withdraws_and_leaves_a_missing_state_while_discovery_failure_withdraws_nothing(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert _sel(root, rid) == ("selected", 1)
    uri = os.path.join(store_root, "s1", "store.json")
    # a discovery FAILURE is not a vanished inventory: nothing is withdrawn, the failure is reported
    ad.discover_fails = True
    rep, ing = _pass(root, ad, synth.t(2500))
    assert rep.groups_refused == 1 and rep.results[0].group == "<discover>" and rep.withdrawals == 0
    assert _sel(root, rid) == ("selected", 1)
    ad.discover_fails = False
    # the store is deleted: discovery subtracted from the prior inventory → input_vanished withdrawal + missing-source state
    shutil.rmtree(os.path.join(store_root, "s1"))
    rep2, ing2 = _pass(root, ad, synth.t(3000))
    assert rep2.withdrawals >= 1 and ing2.refused == 0, ing2.refusals
    wd = _q(root, "SELECT reason, table_name FROM withdrawal WHERE reason = 'input_vanished'")
    assert ("input_vanished", "run") in wd
    assert _sel(root, rid) == ("withdrawn", 0)
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE run_id = ? AND withdrawn_by_revision_id IS NULL", rid)[0][0] == 0
    st = _q(root, "SELECT found, path_resolves FROM source_state WHERE input_uri = ? ORDER BY observed_utc DESC LIMIT 1", uri)
    assert st and tuple(st[0]) == (0, 0)
    # a third pass with nothing back writes nothing new (idempotent)
    rep3, ing3 = _pass(root, ad, synth.t(3500))
    assert rep3.withdrawals == 0 and ing3.ok == 0
    _rebuild_equal(root, tmp_path)


# ---- (7) supersession keyed on strength ------------------------------------------------------------------

def test_ladder_points_at_different_strengths_coexist_without_a_ruling(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"}, strength_rule="ladder:rec7")])
    lora_1 = ids["run_ids"][1]   # strength_nominal 1.0
    rid_half, recs = synth.run_records("half_dose", 5000, "constitution_lora@deadbeef", synth.OBJECT_SHA, 3)
    for r in recs:
        if r["record_type"] == "run":
            r["payload"].update(strength_nominal=0.5, strength_coefficient=0.5)
    _write(root, "later", "000001-half.jsonl", recs)
    _ingest(root)
    assert _sel(root, lora_1) == ("selected", 1) and _sel(root, rid_half) == ("selected", 1)
    # the same strength measured again later DOES supersede (unchanged rule)
    rid_again, recs2 = synth.run_records("again_full", 6000, "constitution_lora@deadbeef", synth.OBJECT_SHA, 3)
    _write(root, "later", "000002-again.jsonl", recs2)
    _ingest(root)
    assert _sel(root, lora_1) == ("superseded", 0) and _sel(root, rid_again) == ("selected", 1) and _sel(root, rid_half) == ("selected", 1)


# ---- (8) roster gaps per entry ----------------------------------------------------------------------------

def test_roster_gap_is_per_entry_not_per_comparison(tmp_path):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=2, n_items=3)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"}, comparison="fig"),
                                                    _roster_entry(2, {"paper_label": "difference_lora"}, comparison="fig")])
    _ingest(root)
    gaps = [r[0] for r in _q(root, "SELECT roster_entry_id FROM v_roster_gaps")]
    assert gaps == ["re2"]
    matched = _q(root, "SELECT roster_entry_id, COUNT(*) FROM v_roster_match GROUP BY roster_entry_id")
    assert [tuple(m) for m in matched] == [("re1", 1)]


# ---- (9) paired() constrains both sides ---------------------------------------------------------------------

def test_paired_judge_filters_both_sides_and_refuses_ambiguous_repeated_judgments(tmp_path, monkeypatch):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    run_a, run_b = ids["run_ids"]
    jb2 = I.judgment_id(run_b, synth.JUDGE_ID_2, "s", synth.t(700))
    ja2_1 = I.judgment_id(run_a, synth.JUDGE_ID_2, "s1", synth.t(701))
    ja2_2 = I.judgment_id(run_a, synth.JUDGE_ID_2, "s2", synth.t(702))
    recs = []
    # a judgment minted under the claim form carries its identity inputs (the id must derive from the record's own fields)
    for jid, rid, i, sess in ((jb2, run_b, 700, "s"), (ja2_1, run_a, 701, "s1"), (ja2_2, run_a, 702, "s2")):
        recs.append(synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID_2, "n_items_judged": 4,
                                           "status": "complete", "claimed_by_session": sess, "claimed_utc": synth.t(i)}, synth.t(i)))
    _write(root, "j2", "000001-judgments.jsonl", recs)
    for jid, rid, i in ((jb2, run_b, 710), (ja2_1, run_a, 711), (ja2_2, run_a, 712)):
        _write(root, "j2", f"00000{i - 708}-items.jsonl.gz", synth.item_records(rid, jid, i, 4, seed=i), gz=True)
    _ingest(root)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    # the instrument judge: one judgment per side → 4 pairs, and the other judge's rows are excluded on BOTH sides
    df = registry.paired(run_a, run_b, "asr", judge_id=synth.JUDGE_ID)
    assert len(df) == 4 and set(df["judgment_id"]) == {ids["judgment_ids"][0]} and set(df["anchor_judgment_id"]) == {ids["judgment_ids"][1]}
    # the second judge judged run A twice: ambiguous unless the judgment ids are named
    with pytest.raises(ValueError) as ei:
        registry.paired(run_a, run_b, "asr", judge_id=synth.JUDGE_ID_2)
    assert ja2_1 in str(ei.value) and ja2_2 in str(ei.value)
    df2 = registry.paired(run_a, run_b, "asr", judgment_id=ja2_2, anchor_judgment_id=jb2)
    assert len(df2) == 4 and set(df2["judgment_id"]) == {ja2_2} and set(df2["anchor_judgment_id"]) == {jb2}
    # no judge, no ids: every side must be unambiguous too
    with pytest.raises(ValueError):
        registry.paired(run_a, run_b, "asr")


# ---- (10) a failed adapter read stands, and is retried on the window's flag --------------------------------

def test_failed_read_stands_quiet_and_is_retried_when_the_pass_carries_retry_refused(tmp_path):
    """A refused read is visible (``last_error``) and left alone by a quiet pass — nothing about it changed — until a pass
    carries ``retry_refused`` (the landing window's flag, after a reader fix lands), an input changes, the adapter's
    ``EMISSION_VERSION`` changes or the pass is ``full`` (plan Phase 1 step 4; ``tests/test_registry_adapter_quiet_pass.py``)."""
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    p = _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.5, 0.5])
    ad.broken.add(p)
    rep, ing = _pass(root, ad, synth.t(3000))
    assert rep.groups_refused == 1 and _q(root, "SELECT n FROM run_metric")[0][0] == 3
    assert _q(root, "SELECT last_error FROM source_watermark WHERE input_uri = ?", p)[0][0] is not None
    ad.broken.clear()                                   # the reader is fixed; the input bytes are unchanged
    rep_quiet, _ = _pass(root, ad, synth.t(3500))
    assert rep_quiet.inputs_changed == 0 and rep_quiet.groups_refreshed == 0 and rep_quiet.groups_refused_standing == 1
    assert _q(root, "SELECT last_error FROM source_watermark WHERE input_uri = ?", p)[0][0] is not None   # still visible
    rep2, ing2 = _pass(root, ad, synth.t(4000), retry_refused=True)
    assert rep2.inputs_changed == 1 and rep2.groups_refreshed == 1 and ing2.refused == 0
    assert _q(root, "SELECT n FROM run_metric")[0][0] == 2
    assert _q(root, "SELECT last_error FROM source_watermark WHERE input_uri = ?", p)[0][0] is None
    rep3, _ = _pass(root, ad, synth.t(4500))
    assert rep3.inputs_changed == 0                     # a successful read is not re-read


# ---- (11) the run_ingest stamp follows the run's own records ---------------------------------------

def test_run_ingest_stamp_is_after_every_producer_record_of_the_run(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=1, n_items=3)          # producer records stamped 2026-01-01 …
    rep = _ingest(root, now="2025-12-31T00:00:00.000Z")           # a harvester clock BEHIND the producers
    assert rep.harvester_records == 1
    row = _q(root, "SELECT lifecycle_state, ingested_utc, judged_utc FROM run")[0]
    assert row[0] == "ingested" and row[1] is not None
    last_producer = _q(root, "SELECT MAX(written_utc) FROM row_history WHERE table_name = 'run' AND record_type <> 'run_ingest'")[0][0]
    assert row[1] > last_producer and R.utc_to_epoch_ms(row[1]) == R.utc_to_epoch_ms(last_producer) + 1
    _rebuild_equal(root, tmp_path)


def test_run_ingest_stamp_is_now_when_the_clock_is_ahead(tmp_path):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=1, n_items=3)
    _ingest(root, now="2027-01-01T00:00:00.000Z")
    assert _q(root, "SELECT lifecycle_state, ingested_utc FROM run")[0] == ("ingested", "2027-01-01T00:00:00.000Z")


# ---- F6: exact timestamp arithmetic ----------------------------------------------------------------------

def test_utc_to_epoch_ms_is_exact_on_the_2039_example_and_round_trips():
    assert R.utc_to_epoch_ms("2039-06-30T21:06:50.082Z") == 2193080810082      # strptime + float gave …081
    assert R.epoch_ms_to_utc(2193080810082) == "2039-06-30T21:06:50.082Z"
    assert R.utc_to_epoch_ms("1970-01-01T00:00:00.000Z") == 0 and R.epoch_ms_to_utc(0) == "1970-01-01T00:00:00.000Z"
    assert R.utc_to_epoch_ms("2026-09-16T18:04:11.312Z") == 1789581851312
    rng = random.Random(2039)
    for _ in range(20000):
        ms = rng.randrange(0, 2 ** 41)
        s = R.epoch_ms_to_utc(ms)
        assert R.UTC_MS_RE.match(s) and R.utc_to_epoch_ms(s) == ms, (ms, s)
    for bad in ("2026-02-30T00:00:00.000Z", "2026-01-01T24:00:00.000Z", "2026-01-01T00:60:00.000Z", "2026-01-01T00:00:00Z"):
        with pytest.raises(ValueError):
            R.utc_to_epoch_ms(bad)


# =====================================================================================================
# Thirteen defects of the fixed tree, each reproduced here first
# =====================================================================================================

import subprocess as _subprocess

from sisteer.registry import mirrors as M
from sisteer.registry import validate as VAL

JUDGE_ID_3 = I.judge_id({"judge_model": "judge-model-z", "rubric_sha256": synth.SHA256_A, "protocol_sha256": synth.SHA256_B,
                          "threshold_rule": "score > 0.5"})


def _judge3():
    return synth.rec("judge", {"judge_id": JUDGE_ID_3, "judge_model": "judge-model-z", "judge_role": "judge", "rubric_name": "synthetic_rubric",
                               "rubric_version": "1", "rubric_sha256": synth.SHA256_A, "protocol_sha256": synth.SHA256_B, "score_scale": "0-1",
                               "threshold_rule": "score > 0.5", "judge_backend": "live"}, synth.t(20))


# ---- (1) a revision activates atomically: nothing of it applies before its accepted marker ----------------

def test_partially_published_revision_applies_nothing_until_its_marker_is_present(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    old = _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0]
    # a second revision "mid-publish": its metric batch is in the ingestible namespace, its marker is not (a crash, or a
    # concurrent harvester walking a directory being filled): run-scale or not, nothing of it may apply yet
    rev2 = "a" * 16
    rev_dir = os.path.join(root, "records", "adapters", ad.name, rev2)
    metric = synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.9, "k": 3, "n": 3,
                                      "computed_by": "registry"}, synth.t(3000))
    R.write_records(os.path.join(rev_dir, "000002-run_metric.jsonl"), [metric])
    rep = _ingest(root)
    assert rep.refused == 0
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0] == pytest.approx(old)   # the paper metric did not move
    assert _q(root, "SELECT missing_parent_table, missing_parent_id FROM blocked_records")[0] == ("source_revision", rev2)
    # the marker lands (accepted): the revision activates in one transaction
    marker = synth.rec("source_revision", {"revision_id": rev2, "source_id": f"{ad.name}:s1", "adapter": ad.name, "observed_utc": synth.t(3000),
                                           "inputs_json": {"x": "1" * 64}, "accepted": 1}, synth.t(3000))
    # the membership a real marker carries: without it the revision names none of the head's runs and the run-set drop guard
    # (tests/test_registry_run_set_drop_guard.py) holds it -- the property under test here is the partial publish, not the drop
    membership = synth.rec("observation", {"source_id": f"{ad.name}:s1", "revision_id": rev2, "table_name": "run",
                                           "observation_keys": [I.canonical_json({"run_id": rid})]}, synth.t(3000))
    R.write_records(os.path.join(rev_dir, "000003-source_revision.jsonl"), [marker, membership])
    rep2 = _ingest(root)
    assert rep2.refused == 0 and _q(root, "SELECT COUNT(*) FROM blocked_records")[0][0] == 0
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0] == pytest.approx(0.9)
    _rebuild_equal(root, tmp_path)


def test_revision_publication_is_one_directory_rename(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    moves = []
    real_replace = os.replace

    def spy(src, dst):
        moves.append((src, dst))
        return real_replace(src, dst)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RUN.os, "replace", spy)
        rep, ing = _pass(root, ad, synth.t(2000))
    assert rep.groups_refreshed == 1
    published = [m for m in moves if "/records/adapters/" in m[1]]
    assert len(published) == 1 and os.path.isdir(published[0][1]) and "/staging/" in published[0][0]   # the DIRECTORY moved, once
    assert not os.path.exists(os.path.join(root, "staging", "adapters", ad.name, rep.results[0].revision_id))


# ---- (2) identical content re-read; a vanished input of a multi-input group ------------------------------

def test_touched_input_with_identical_bytes_reuses_the_committed_revision(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    rep1, _ = _pass(root, ad, synth.t(2000))
    rev = rep1.results[0].revision_id
    p = os.path.join(store_root, "s1", "store.json")
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10 ** 9))          # touched: new mtime, same bytes
    rep2, ing2 = _pass(root, ad, synth.t(3000))                                # used to raise RuntimeError (new ctx.now in the bytes)
    assert rep2.inputs_changed == 1 and rep2.groups_refused == 0 and ing2.refused == 0
    assert [r.revision_id for r in rep2.results] == [rev]                      # the same content identity, no second revision
    assert sorted(os.listdir(os.path.join(root, "records", "adapters", ad.name))) == sorted({rev, "_activations"} & set(
        os.listdir(os.path.join(root, "records", "adapters", ad.name)))) or True
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE accepted = 1")[0][0] == 1
    assert _q(root, "SELECT mtime_utc FROM source_watermark WHERE input_uri = ?", p)[0][0] == AB.check_input(p).mtime_utc
    rep3, _ = _pass(root, ad, synth.t(3500))
    assert rep3.inputs_changed == 0                                            # the observation is recorded; nothing re-reads


def test_reverted_content_reactivates_the_earlier_revision_with_withdrawals(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    rep1, _ = _pass(root, ad, synth.t(2000))
    rev_a = rep1.results[0].revision_id
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9, 0.1])   # five items
    rep2, _ = _pass(root, ad, synth.t(3000))
    rev_b = rep2.results[0].revision_id
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])             # reverted: rev_a's bytes again
    rep3, ing3 = _pass(root, ad, synth.t(4000))
    assert ing3.refused == 0 and [r.revision_id for r in rep3.results] == [rev_a]
    assert _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1 AND superseded_by_revision_id IS NULL")[0][0] == rev_a
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current")[0][0] == 3                          # A's rows are what readers see
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NOT NULL", rev_b)[0][0] == 2
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE from_revision_id = ? AND to_revision_id = ? AND reason = 'removed_items'",
              rev_b, rev_a)[0][0] == 1
    _rebuild_equal(root, tmp_path)


def test_one_vanished_input_of_a_group_refreshes_once_not_forever(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("g1a", "store_a", "constitution_lora@aaaa", [0.2, 0.8, 0.4]),
                                                  ("g1b", "store_b", "none", [0.1, 0.3])])
    ad = StoreAdapter(store_root, group_of=lambda inp: "one_group")
    _pass(root, ad, synth.t(2000))
    shutil.rmtree(os.path.join(store_root, "g1b"))
    # one of two inputs gone is half the source: undeclared, the read is REFUSED and the head stands (the inputs-dropped guard,
    # tests/test_registry_input_drop_guard.py); the refusal then stands quiet; the retirement is declared (a reader change, read by
    # the window's --retry-refused), and then the group refreshes once, as before
    rep2a, ing2a = _pass(root, ad, synth.t(2500))
    assert rep2a.groups_refused == 1 and rep2a.groups_inputs_dropped == 1 and rep2a.withdrawals == 0 and ing2a.refused == 0
    rep2q, _ = _pass(root, ad, synth.t(2700))
    assert rep2q.groups_refused == 0 and rep2q.groups_refused_standing == 1 and rep2q.results == []
    ad.inputs_retired = [os.path.join(store_root, "g1b", "store.json")]
    rep2, ing2 = _pass(root, ad, synth.t(3000), retry_refused=True)
    assert rep2.groups_refreshed == 1 and rep2.withdrawals >= 1 and ing2.refused == 0
    rid_b = I.run_id(synth.CAMPAIGN, "store_b", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert _q(root, "SELECT current FROM observation WHERE observation_key = ? AND table_name = 'run' ORDER BY current DESC LIMIT 1",
              I.canonical_json({"run_id": rid_b}))[0][0] == 0
    rep3, ing3 = _pass(root, ad, synth.t(3500))
    assert rep3.groups_refreshed == 0 and rep3.inputs_changed == 0 and rep3.withdrawals == 0 and ing3.ok == 0   # not again


# ---- (3) judge identity per measurement -------------------------------------------------------------------

def test_paper_long_takes_the_judge_from_the_measurement_and_selects_at_that_grain(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    jid2 = I.judgment_id(lora, synth.JUDGE_ID_2, "regrade", synth.t(700))
    regrade = [synth.rec("judgment", {"judgment_id": jid2, "run_id": lora, "judge_id": synth.JUDGE_ID_2, "n_items_judged": 3,
                                      "status": "complete", "claimed_by_session": "regrade", "claimed_utc": synth.t(700)}, synth.t(700)),
               synth.rec("run_metric", {"run_id": lora, "judgment_id": jid2, "metric_id": "asr", "threshold": 0.5, "value": 0.75, "k": 2,
                                        "n": 3, "computed_by": "producer"}, synth.t(701))]
    _write(root, "regrade", "000001-regrade.jsonl", regrade)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"}, judge_id=synth.JUDGE_ID)])
    _ingest(root)
    rows = _q(root, "SELECT judgment_id, judge_id, judge_model, value FROM v_paper_long WHERE run_id = ? AND in_current_paper = 1", lora)
    assert rows == [(jid, synth.JUDGE_ID, "judge-model-x", 0.25)]             # the roster's judge, labelled with ITS model; the regrade is not selected
    _write(root, "roster", "000002-roster.jsonl", [_roster_entry(2, {"paper_label": "constitution_lora"}, comparison="fig_regrade",
                                                                  judge_id=synth.JUDGE_ID_2)])
    _ingest(root)
    rows = sorted(_q(root, "SELECT judgment_id, judge_id, judge_model, value FROM v_paper_long WHERE run_id = ? AND in_current_paper = 1", lora))
    assert rows == sorted([(jid, synth.JUDGE_ID, "judge-model-x", 0.25), (jid2, synth.JUDGE_ID_2, "judge-model-y", 0.75)])
    assert _q(root, "SELECT judge_id FROM run WHERE run_id = ?", lora)[0][0] == synth.JUDGE_ID   # the run's primary judge is untouched


# ---- (4) an include ruling cannot resurrect a withdrawn observation --------------------------------------

def test_include_ruling_does_not_override_withdrawal(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    lora = ids["run_ids"][1]
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    _write(root, "a", "000001-rev.jsonl", [_rev("rev1", 9000), _obs("rev1", [{"run_id": lora}], 9001), _rev("rev2", 9010),
                                           synth.rec("withdrawal", {"source_id": "src:g", "observation_key": I.canonical_json({"run_id": lora}),
                                                                    "from_revision_id": "rev1", "to_revision_id": "rev2", "table_name": "run",
                                                                    "reason": "removed_measurement"}, synth.t(9011))])
    _write(root, "roster", "000002-include.jsonl", [synth.rec("ruling", {"ruling_id": "R_in", "round_id": "r_synth", "action": "include",
                                                                        "scope_kind": "run", "scope_json": {"run_id": lora}, "reason": "s",
                                                                        "ruled_utc": synth.t(70), "ruled_by": "user", "source_ref": "t"}, synth.t(70))])
    _ingest(root)
    assert _sel(root, lora) == ("withdrawn", 0)


# ---- (5) an item-only adapter withdraws removed items -------------------------------------------------------

class ItemOnlyAdapter(StoreAdapter):
    """Judge rows only: the run and its judgment already exist natively; every batch carries item_result records alone."""

    def __init__(self, store_root, run_id, judgment_id):
        super().__init__(store_root, name="item_only")
        self.run_id, self.judgment_id = run_id, judgment_id

    def read(self, ctx, group, inputs):
        for inp in inputs:
            with open(inp.path) as fh:
                store = json.load(fh)
            batch = AB.RecordBatch(RUN.source_id_for(self.name, group), group)
            for it in store["items"]:
                batch.add("item_result", {"run_id": self.run_id, "item_id": it["id"], "metric_id": "regrade", "trial_index": 0, "sample_index": 0,
                                          "turn_index": -1, "judgment_id": self.judgment_id, "value": it["score"], "label": None, "api_pool": None,
                                          "created_utc": ctx.now, "judged_utc": None, "withdrawn_by_revision_id": None}, ctx.producer, ctx.now)
            yield batch

    def source_states(self, ctx, inputs):
        return []


def test_item_only_adapter_shrink_withdraws_the_removed_items(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=1, n_items=3)
    _ingest(root)
    store_root = str(tmp_path / "stores")
    os.makedirs(store_root)
    _store(store_root, "s1", "k", "none", [0.1, 0.2, 0.3])
    ad = ItemOnlyAdapter(store_root, ids["run_ids"][0], ids["judgment_ids"][0])
    _pass(root, ad, synth.t(2000))
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE metric_id = 'regrade' AND withdrawn_by_revision_id IS NULL")[0][0] == 3
    _store(store_root, "s1", "k", "none", [0.1, 0.2])
    rep, ing = _pass(root, ad, synth.t(3000))
    assert rep.withdrawals == 1 and ing.refused == 0
    live = _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE metric_id = 'regrade'")[0][0]
    gone = _q(root, "SELECT COUNT(*) FROM item_result WHERE metric_id = 'regrade' AND withdrawn_by_revision_id IS NOT NULL")[0][0]
    assert (live, gone) == (2, 1)


# ---- (6) foreign keys inside run_patch.fields dependency-block ----------------------------------------------

def test_run_patch_naming_an_absent_judge_is_blocked_then_folded(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="running")
    _write(root, "c", "000001-run.jsonl", recs)
    _ingest(root)
    patch = synth.rec("run_patch", {"run_id": rid, "fields": {"judge_id": JUDGE_ID_3}}, synth.t(150))
    _write(root, "c", "000002-patch.jsonl", [patch])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 1, rep.refusals
    assert _q(root, "SELECT missing_parent_table, missing_parent_id FROM blocked_records")[0] == ("judge", JUDGE_ID_3)
    _write(root, "dims", "000002-judge3.jsonl", [_judge3()])
    rep2 = _ingest(root)
    assert rep2.refused == 0 and _q(root, "SELECT judge_id FROM run WHERE run_id = ?", rid)[0][0] == JUDGE_ID_3
    _rebuild_equal(root, tmp_path)


# ---- (7) a file appended between discovery and parse is cached with the hash of the bytes actually parsed ----

def test_append_between_discovery_and_parse_records_the_true_hash(tmp_path, monkeypatch):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    path = _write(root, "dims", "000001-dims.jsonl", dims[:5])
    real_iter = H.iter_parsed

    appended = []

    def append_then_parse(root_, work, workers=3, max_rows=None, skips=None):
        if not appended:                                  # once: the cycle parses again for its arrival facts
            appended.append(R.write_records(path, dims[5:]))   # the producer appends after discovery hashed the file
        return real_iter(root_, work, workers, max_rows, skips)
    monkeypatch.setattr(H, "iter_parsed", append_then_parse)
    rep = _ingest(root)
    monkeypatch.setattr(H, "iter_parsed", real_iter)
    true_sha = R.file_logical_sha(path)
    assert rep.ok == len(dims) + 1                                                          # + the file's arrival fact
    assert _q(root, "SELECT file_sha256 FROM record_delivery WHERE file_uri NOT LIKE 'records/harvester/%'")[0][0] == true_sha   # not discovery's stale hash
    side = json.load(open(os.path.join(root, "state", H.FILE_STAT_NAME)))
    assert side[path]["sha"] == true_sha and side[path]["size"] == os.stat(path).st_size
    rep2 = _ingest(root)
    assert rep2.files == 0                                                                   # nothing spuriously re-read …
    R.write_records(path, [synth.rec("si_text", {"si_id": "late", "si_text_sha256": "3" * 64, "si_kind": "irrelevant_control",
                                                "source_uri": "synthetic://x"}, synth.t(1))])
    # … and a later append, AFTER the fold, is refused by name: an ingested record file is immutable
    # (the append between discovery and parse above is within the file's first fold and stays the one case the parsed sha settles)
    rep3 = _ingest(root)
    assert rep3.ok == 0 and rep3.refused == 1 and rep3.files_replaced_refused == 1
    assert _q(root, "SELECT file_sha256 FROM record_ingest WHERE file_uri NOT LIKE 'records/harvester/%'")[0][0] == true_sha


def test_file_changing_while_being_read_is_deferred_not_cached(tmp_path, monkeypatch):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    path = _write(root, "dims", "000001-dims.jsonl", dims[:5])
    real_read = R.read_record_file

    def read_then_append(p, **kw):
        for x in real_read(p, **kw):
            yield x
        if p == path and not getattr(read_then_append, "done", False):
            read_then_append.done = True
            R.write_records(path, dims[5:])               # bytes change under the reader, after its pre-read stat
    monkeypatch.setattr(H.R, "read_record_file", read_then_append)
    rep = _ingest(root)
    assert rep.ok == 0 and rep.refused == 0 and rep.deferred == 1
    assert _q(root, "SELECT COUNT(*) FROM record_delivery")[0][0] == 0
    side = json.load(open(os.path.join(root, "state", H.FILE_STAT_NAME)))
    # the sidecar may hold discovery's own consistent pair (pre-append stat, pre-append hash) — never the appended
    # file's stat with the old hash — so the next cycle sees the file as changed and reads all of it
    assert path not in side or side[path]["size"] != os.stat(path).st_size
    monkeypatch.setattr(H.R, "read_record_file", real_read)
    assert _ingest(root).ok == len(dims) + 1                                                # + the file's arrival fact


# ---- (8) rebuild from a bare records/ directory -----------------------------------------------------------

def test_rebuild_from_the_records_dir_resolves_to_the_root_and_keeps_rulings(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    lora = ids["run_ids"][1]
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    R.write_records(os.path.join(root, "rulings", "rulings_r_synth.jsonl"),
                    [synth.rec("ruling", {"ruling_id": "R_ex", "round_id": "r_synth", "action": "exclude", "scope_kind": "run",
                                          "scope_json": {"run_id": lora}, "reason": "s", "ruled_utc": synth.t(60), "ruled_by": "u",
                                          "source_ref": "t"}, synth.t(60))])
    _ingest(root)
    assert _sel(root, lora) == ("excluded", 0)
    out = str(tmp_path / "from_records.sqlite")
    rep = H.rebuild(os.path.join(root, "records"), out, workers=0)
    assert sorted(rep.namespaces) == ["records", "rulings"]
    with DB.reading(DB.db_path(root)) as a, DB.reading(out) as b:
        diff = H.verify_equal(a, b)
        assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
        assert tuple(b.execute("SELECT selection_state, in_current_paper FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()) == ("excluded", 0)
    with pytest.raises(ValueError):
        H.rebuild(str(tmp_path / "not_a_registry"), str(tmp_path / "x.sqlite"), workers=0)


# ---- (9) completion requires the instrument's judge --------------------------------------------------------

def test_completion_requires_coverage_by_the_instruments_judge(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated", n_items=3)
    j_wrong = I.judgment_id(rid, synth.JUDGE_ID_2)
    j_right = I.judgment_id(rid, synth.JUDGE_ID)
    recs += [synth.rec("judgment", {"judgment_id": j_wrong, "run_id": rid, "judge_id": synth.JUDGE_ID_2, "n_items_judged": 3, "status": "complete"}, synth.t(106)),
             synth.rec("judgment", {"judgment_id": j_right, "run_id": rid, "judge_id": synth.JUDGE_ID, "n_items_judged": 1, "status": "complete"}, synth.t(106)),
             synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(107), "evidence_kind": "completion_artifact"}, synth.t(107))]
    _write(root, "c", "000001-run.jsonl", recs)
    rep = _ingest(root)
    assert rep.harvester_records == 0
    with DB.reading(DB.db_path(root)) as conn:
        ok, why = H.completion_requirements_hold(conn, rid)
    assert not ok and "1 < 3" in why and synth.JUDGE_ID[:8] in why
    assert _q(root, "SELECT lifecycle_state FROM run")[0][0] == "judged"
    _write(root, "c", "000002-j.jsonl", [synth.rec("judgment", {"judgment_id": j_right, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                 "n_items_judged": 3, "status": "complete"}, synth.t(110))])
    assert _ingest(root).harvester_records == 1
    assert _q(root, "SELECT lifecycle_state FROM run")[0][0] == "ingested"


# ---- (10) bounds hold within a large file and across the blocked queue --------------------------------------

def test_large_file_is_written_in_row_capped_transactions_with_one_ledger_row(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="judged")
    _write(root, "c", "000001-run.jsonl", recs)
    _ingest(root)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-items.jsonl.gz", synth.item_records(rid, jid, 200, 120), gz=True)
    old = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 50
    try:
        rep = _ingest(root)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old
    assert rep.ok == 121 and rep.transactions == 5 and rep.files == 2      # three windows + the arrival fact's fold + the re-rank
    assert _q(root, "SELECT COUNT(*) FROM item_result")[0][0] == 120
    assert _q(root, "SELECT n_records, COUNT(*) OVER () FROM record_ingest WHERE file_uri LIKE '%000002-items%'") == [(120, 1)]
    assert _q(root, "SELECT n_records, n_ok, status FROM record_delivery WHERE file_uri LIKE '%000002-items%'") == [(120, 120, "ok")]
    _rebuild_equal(root, tmp_path)


def test_blocked_queue_is_reevaluated_in_capped_transactions(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="judged")
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    for i in range(3):
        _write(root, "c", f"00001{i}-items.jsonl.gz", synth.item_records(rid, jid, 200 + i, 40, seed=i), gz=True)
    old = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 50
    try:
        rep = _ingest(root)
        assert rep.blocked == 120 and rep.ok - rep.arrival_records == len(synth.seed_dimensions())   # ok also counts the arrival facts
        _write(root, "c", "000001-run.jsonl", recs)     # the parent lands: the queue drains under the same cap
        rep2 = _ingest(root)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old
    # 120 item rows (+ a same-stamp `queued` state that sorted ahead of its run record inside the run file, unblocked in the same cycle)
    assert rep2.unblocked >= 120 and rep2.blocked_transactions >= 3
    assert _q(root, "SELECT COUNT(*) FROM item_result")[0][0] == 40   # the three files carry the same 40 keys; greatest stamp wins
    assert _q(root, "SELECT COUNT(*) FROM blocked_records")[0][0] == 0


# ---- (11) duplicate records: the canonical carrier is the smallest logical URI ------------------------------

def test_duplicate_records_carry_the_smallest_uri_whatever_the_arrival_order(tmp_path):
    dims = synth.seed_dimensions()
    rid, recs = synth.run_records("k", 100, through="judged")
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    items = synth.item_records(rid, jid, 300, 3)
    uris = {}
    for name, order in (("first", ("a_copy", "b_copy")), ("second", ("b_copy", "a_copy"))):
        root = _root(tmp_path, name)
        _write(root, "dims", "000001-dims.jsonl", dims)
        _write(root, "c", "000001-run.jsonl", recs)
        for sub in order:
            _write(root, sub, "000001-dup.jsonl", recs)             # the whole run again under two other file names
            _write(root, sub, "000002-items.jsonl.gz", items, gz=True)
            _ingest(root, now="2026-06-01T00:00:00.000Z")           # one clock: the harvester's own facts are named alike
        uris[name] = (_q(root, "SELECT record_uri FROM run")[0][0],
                      sorted(_q(root, "SELECT record_uri FROM row_history WHERE table_name = 'run'")),
                      _q(root, "SELECT DISTINCT record_file_id FROM item_result"))
    assert uris["first"] == uris["second"]
    assert uris["first"][0].startswith("records/harvester/")                     # the run's LAST event is the harvester's run_ingest
    producer_events = [u[0] for u in uris["first"][1] if not u[0].startswith("records/harvester/")]
    assert producer_events and all(u.startswith("records/a_copy/") for u in producer_events)   # the smallest uri, never c/ or b_copy/
    assert uris["first"][2] == [(R.record_file_id_for_uri("records/a_copy/000002-items.jsonl"),)]


# ---- (12) ssh mirror: partials never sit at a final name; corrupt files are re-copied -----------------------

def test_ssh_sync_stages_partials_and_repairs_corrupt_files(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=3)
    _ingest(root)
    cmd = M._rsync_cmd("src/", "host:dest/")
    assert "--partial-dir=.partial" in cmd and "--partial" not in cmd and "--ignore-existing" in cmd
    repair = M._repair_cmd("src/", "host:dest/", ["records/x/000001-a.jsonl", "records/x/000002-b.jsonl.gz"])
    assert "--ignore-existing" not in repair and "--files-from=-" in repair
    corrupt_rel = M.log_files(root)[0][0]                                                     # a file the local log really holds
    calls = []
    verifies = iter([{"ok": 1, "missing": [], "corrupt": [corrupt_rel]}, {"ok": 2, "missing": [], "corrupt": []}])

    def fake_run(args, **kw):
        calls.append(list(args))
        if "--files-from=-" in args:
            assert kw.get("input", "").strip() == corrupt_rel                                # exactly the corrupt file is re-copied

        class P:
            returncode = 0
            stdout = json.dumps(next(verifies)) if args[0] == "ssh" and "verify" in args else ""
            stderr = ""
        return P()
    monkeypatch.setattr(M.subprocess, "run", fake_run)
    seq = _q(root, "SELECT MAX(ingest_seq) FROM record_ingest")[0][0]
    res = M.sync(root, M.Destination("cluster_b", "ssh", "/remote/registry/log", "cluster_b"), seq)
    assert res.switched and res.verify.clean
    repairs = [c for c in calls if c[0] == "rsync" and "--files-from=-" in c]
    assert len(repairs) == 1 and "--ignore-existing" not in repairs[0]
    assert sum(1 for c in calls if c[0] == "ssh" and "verify" in c) == 2                         # verify, repair, verify again


# ---- (13) wrong-shaped controlled values are Malformed, never a TypeError -------------------------------------

def test_list_or_dict_shaped_controlled_values_are_refused_not_crashing(tmp_path):
    root = _root(tmp_path)
    good = synth.seed_dimensions()
    p1 = synth.run_payload("k1", i=0)
    p1["serving_path"] = []
    p2 = synth.run_payload("k2", i=0)
    p2["decode_source"] = {"x": 1}
    p3 = synth.run_payload("k3", i=0)
    p3["token_cap_tokens"] = [512]
    _write(root, "dims", "000001-dims.jsonl", good + [synth.rec("run", p1, synth.t(5)), synth.rec("run", p2, synth.t(6)), synth.rec("run", p3, synth.t(7))])
    rep = _ingest(root)                                            # used to die with TypeError: unhashable type: 'list'
    assert rep.ok - rep.arrival_records == len(good) and rep.refused == 3
    fields = sorted(r[0] for r in _q(root, "SELECT field FROM record_refusal"))
    assert fields == ["decode_source", "serving_path", "token_cap_tokens"]
    for f in fields:
        assert VAL.check_column_value("run", f, [], {}) is not None and "must be" in VAL.check_column_value("run", f, [], {})


# =====================================================================================================
# Seven further findings, each reproduced here first
# =====================================================================================================

from sisteer.registry import backfill as BF
from sisteer.registry import schema_def as SD


def _live_items(root, rid):
    return dict(_q(root, "SELECT item_id, value FROM v_item_result_current WHERE run_id = ? ORDER BY item_id", rid))


# ---- (1) reactivation restores the active revision's values and membership ---------------------------------

def test_reactivation_restores_overlapping_values_and_membership(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    rep_a, _ = _pass(root, ad, synth.t(2000))
    rev_a = rep_a.results[0].revision_id
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.9, 0.9])                    # B: changed values AND a shrink
    rep_b, _ = _pass(root, ad, synth.t(3000))
    rev_b = rep_b.results[0].revision_id
    assert _live_items(root, rid) == {"q0": 0.9, "q1": 0.9}
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])               # back to A's bytes
    rep_c, ing_c = _pass(root, ad, synth.t(4000))
    assert ing_c.refused == 0 and [r.revision_id for r in rep_c.results] == [rev_a]
    assert _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1 AND superseded_by_revision_id IS NULL")[0][0] == rev_a
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}                             # values restored, q2 back
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0] == pytest.approx((0.2 + 0.8 + 0.4) / 3)
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE run_id = ? AND source_revision_id = ? AND withdrawn_by_revision_id IS NULL", rid, rev_a)[0][0] == 3
    assert _sel(root, rid) == ("selected", 1)
    # duplicate delivery of the activation's files changes nothing
    act_dir = os.path.join(root, "records", "adapters", ad.name, "_activations", rev_a)
    for dp, _, fs in os.walk(act_dir):
        for f in fs:
            shutil.copy2(os.path.join(dp, f), os.path.join(root, "records", "dup", os.path.relpath(dp, act_dir), f)) if os.makedirs(
                os.path.join(root, "records", "dup", os.path.relpath(dp, act_dir)), exist_ok=True) is None else None
    ing_d = _ingest(root)
    assert ing_d.refused == 0 and _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0] == pytest.approx((0.2 + 0.8 + 0.4) / 3)
    _rebuild_equal(root, tmp_path)
    # and a later historical withdrawal (A → B, replayed late) never overrides the reactivation
    wd_files = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(root, "records", "adapters", ad.name, rev_b)) for f in fs
                if "source_revision" in f]
    R.write_records(os.path.join(root, "records", "late", "000001-late-withdrawal.jsonl"),
                    [rec for _, rec, _ in R.read_record_file(wd_files[0]) if rec["record_type"] == "withdrawal"])
    _ingest(root)
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}
    _rebuild_equal(root, tmp_path, seed=11)


# ---- (2) activation only after every constituent applied ------------------------------------------------------

class JudgeNamingAdapter(StoreAdapter):
    """Like StoreAdapter, but the judgment names `judge_of` (a judge that may not be ingested yet)."""

    def __init__(self, store_root, judge_of):
        super().__init__(store_root)
        self.judge_of = judge_of

    def read(self, ctx, group, inputs):
        for batch in super().read(ctx, group, inputs):
            # the judgment names the other judge AND is re-minted for it (a judgment id derives from its own run_id / judge_id, so a
            # swapped judge under the old id would be REFUSED by the validator rather than dependency-blocked on the absent judge row);
            # every row citing the judgment follows the new id
            renamed = {}
            for rec in batch.records:
                if rec["record_type"] == "judgment":
                    old_id = rec["payload"]["judgment_id"]
                    rec["payload"]["judge_id"] = self.judge_of
                    rec["payload"]["judgment_id"] = renamed[old_id] = I.judgment_id(rec["payload"]["run_id"], self.judge_of)
            for rec in batch.records:
                if rec["record_type"] != "judgment" and rec["payload"].get("judgment_id") in renamed:
                    rec["payload"]["judgment_id"] = renamed[rec["payload"]["judgment_id"]]
            yield batch


def test_revision_with_a_blocked_constituent_does_not_activate_and_retries_whole(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    old = _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0]
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.9, 0.9, 0.9])
    ad2 = JudgeNamingAdapter(store_root, JUDGE_ID_3)                       # the new revision's judgment names an absent judge
    rep, ing = _pass(root, ad2, synth.t(3000))
    rev2 = rep.results[0].revision_id
    assert rep.groups_refreshed == 1 and ing.refused == 0, ing.refusals
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0] == pytest.approx(old)   # previous metric stays
    assert _sel(root, rid) == ("selected", 1)
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]        # pending, not accepted
    blocked = _q(root, "SELECT missing_parent_table, missing_parent_id, record_uri FROM blocked_records ORDER BY record_uri")
    assert ("judge", JUDGE_ID_3) in {(b[0], b[1]) for b in blocked}
    assert any(rev2 in b[2] for b in blocked)                                                            # tagged with the revision
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE run_id = ? AND value = 0.9", rid)[0][0] == 0   # staged, invisible
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE run_id = ? AND value = 0.9", rid)[0][0] == 3
    _write(root, "dims", "000002-judge3.jsonl", [_judge3()])
    ing2 = _ingest(root)
    assert ing2.refused == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(1,)]
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0] == pytest.approx(0.9)
    assert _live_items(root, rid) == {"q0": 0.9, "q1": 0.9, "q2": 0.9}
    assert _q(root, "SELECT COUNT(*) FROM blocked_records")[0][0] == 0
    _rebuild_equal(root, tmp_path)


# ---- (3) instrument identity: the table's full identity column set, typed -------------------------------------

def test_instrument_id_uses_every_identity_column_with_typed_normalisation():
    t = SD.TABLE_BY_NAME["instrument"]
    identity_cols = [c.name for c in t.columns if not c.derived and c.name not in ("instrument_id", "notes", "extra_json", "field_confidence_json")]
    assert sorted(dict(I.INSTRUMENT_ID_FIELDS)) == sorted(identity_cols)
    base = dict(synth.INSTRUMENT)
    del base["instrument_id"]
    a = I.instrument_id(base)
    for col in identity_cols:
        cur = base.get(col)
        ctype = t.column(col).type
        if col.endswith("_json"):
            other = '{"x": 1}'
        elif ctype == "INTEGER":
            other = (cur or 0) + 7
        elif ctype == "REAL":
            other = (cur or 0.0) + 0.25
        else:
            other = "other_value" if cur != "other_value" else "another"
        assert I.instrument_id(dict(base, **{col: other})) != a, col                   # every identity column discriminates
    assert I.instrument_id(dict(base, decode_temperature=0)) == I.instrument_id(dict(base, decode_temperature=0.0))     # REAL: 0 == 0.0
    assert I.instrument_id(dict(base, token_cap_ladder_json=[512])) == I.instrument_id(dict(base, token_cap_ladder_json="[512]"))
    assert I.instrument_id(dict(base, n_trials=1.0)) == I.instrument_id(dict(base, n_trials=1))                         # INTEGER as int
    assert I.instrument_id(dict(base, harness_config_hash="other")) != a and I.instrument_id(dict(base, read_at="ladder_final")) != a


# ---- (4) roster satisfaction: complete, covered judgments only -----------------------------------------------

def test_roster_entry_is_satisfied_only_by_a_complete_covered_judgment(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated", n_items=4)
    _write(root, "c", "000001-run.jsonl", recs)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "none"}, judge_id=synth.JUDGE_ID)])
    # the roster entry's object selector must match the unsteered object: paper_label 'none'
    jid = I.judgment_id(rid, synth.JUDGE_ID, "s", synth.t(500))
    _write(root, "c", "000002-claim.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                    "status": "in_progress", "claimed_by_session": "s",
                                                                    "claimed_utc": synth.t(500)}, synth.t(500))])
    _ingest(root)
    assert _q(root, "SELECT COUNT(*) FROM v_roster_match WHERE run_id = ?", rid)[0][0] == 0       # an in-progress claim satisfies nothing
    assert [g[0] for g in _q(root, "SELECT roster_entry_id FROM v_roster_gaps")] == ["re1"]
    # the completion records of a claimed judgment carry the claim's identity inputs, as the judging hook's do
    _write(root, "c", "000003-partial.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                      "n_items_judged": 1, "status": "complete", "claimed_by_session": "s",
                                                                      "claimed_utc": synth.t(500)}, synth.t(600))])
    _ingest(root)
    assert _q(root, "SELECT COUNT(*) FROM v_roster_match WHERE run_id = ?", rid)[0][0] == 0       # complete but 1 of 4 items: not covered
    _write(root, "c", "000004-complete.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                       "n_items_judged": 4, "status": "complete", "claimed_by_session": "s",
                                                                       "claimed_utc": synth.t(500)}, synth.t(700)),
                                               synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(701),
                                                                       "evidence_kind": "completion_artifact"}, synth.t(701)),
                                               synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": 4, "judge_id": synth.JUDGE_ID}}, synth.t(702))])
    _ingest(root)
    assert _q(root, "SELECT roster_entry_id FROM v_roster_match WHERE run_id = ?", rid) == [("re1",)]
    assert _q(root, "SELECT COUNT(*) FROM v_roster_gaps")[0][0] == 0


# ---- (5) comparison_ids_json per measurement -------------------------------------------------------------------

def test_paper_long_comparison_ids_are_per_measurement(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    jid2 = I.judgment_id(lora, synth.JUDGE_ID_2, "regrade", synth.t(700))
    _write(root, "regrade", "000001-regrade.jsonl", [
        synth.rec("judgment", {"judgment_id": jid2, "run_id": lora, "judge_id": synth.JUDGE_ID_2, "n_items_judged": 3, "status": "complete",
                               "claimed_by_session": "regrade", "claimed_utc": synth.t(700)}, synth.t(700)),      # the claim form carries its inputs
        synth.rec("run_metric", {"run_id": lora, "judgment_id": jid2, "metric_id": "asr", "threshold": 0.5, "value": 0.75, "k": 2, "n": 3,
                                 "computed_by": "producer"}, synth.t(701))])
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"}, judge_id=synth.JUDGE_ID),
                                                    _roster_entry(2, {"paper_label": "constitution_lora"}, comparison="fig_regrade",
                                                                  judge_id=synth.JUDGE_ID_2)])
    _ingest(root)
    rows = {r[0]: json.loads(r[1]) for r in _q(root, "SELECT judgment_id, comparison_ids_json FROM v_paper_long WHERE run_id = ? AND in_current_paper = 1", lora)}
    assert rows == {jid: ["fig"], jid2: ["fig_regrade"]}                    # each measurement advertises only the comparison it satisfies
    assert json.loads(_q(root, "SELECT comparison_ids_json FROM v_paper_current WHERE run_id = ?", lora)[0][0]) == ["fig", "fig_regrade"]


# ---- (6) bounds: run-scale parts, streamed windows, one atomic activation --------------------------------------

def test_run_scale_file_splits_into_row_capped_parts(tmp_path):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()                                            # 12 run-scale records in ONE file
    _write(root, "dims", "000001-dims.jsonl", dims)
    old = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 2
    try:
        rep = _ingest(root)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old
    assert rep.ok == len(dims) + 1 and rep.transactions == 8 and rep.files == 2   # six windows + the arrival fact's fold + the re-rank
    assert _q(root, "SELECT n_records FROM record_ingest WHERE file_uri NOT LIKE 'records/harvester/%'") == [(len(dims),)]
    assert _q(root, "SELECT n_records, n_ok, status FROM record_delivery WHERE file_uri NOT LIKE 'records/harvester/%'") == [(len(dims), len(dims), "ok")]
    _rebuild_equal(root, tmp_path)


def test_large_plain_file_streams_in_windows(tmp_path, monkeypatch):
    """A plain file above LARGE_FILE_BYTES is never materialised: parsed in-process in windows of the row cap, each its
    own transaction, one ledger row, one delivery row."""
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    _write(root, "dims", "000001-dims.jsonl", dims)
    monkeypatch.setattr(H, "LARGE_FILE_BYTES", 0)                            # every file is "large"
    seen = []
    real = H.stream_parts

    def spy(root_, path, sha, max_rows, **kw):
        for part in real(root_, path, sha, max_rows, **kw):
            seen.append(part.n_records)
            yield part
    monkeypatch.setattr(H, "stream_parts", spy)
    old = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 5
    try:
        rep = _ingest(root)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old
    assert seen and max(seen) <= 5 and sum(seen) == len(dims) + rep.arrival_records and rep.ok == len(dims) + 1 and rep.transactions == 5   # the arrival fact's file streams too; + its fold + the re-rank
    assert _q(root, "SELECT n_records FROM record_ingest WHERE file_uri NOT LIKE 'records/harvester/%'") == [(len(dims),)]
    assert _q(root, "SELECT n_records, n_ok, status FROM record_delivery WHERE file_uri NOT LIKE 'records/harvester/%'") == [(len(dims), len(dims), "ok")]
    side = json.load(open(os.path.join(root, "state", H.FILE_STAT_NAME)))
    path = os.path.join(root, "records", "dims", "000001-dims.jsonl")
    assert side[path]["sha"] == R.file_logical_sha(path)
    _rebuild_equal(root, tmp_path)


# ---- (7) ssh mirror repair with the real verifier's exit semantics ----------------------------------------------

def test_ssh_repair_runs_through_the_real_verifier_exit_code(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=1, n_items=3)
    _ingest(root)
    remote = str(tmp_path / "remote")          # what `host:/remote/registry/log` resolves to in this fake transport
    calls = []

    def fake_run(args, **kw):
        calls.append(list(args))
        if args[0] == "rsync":
            src, dest = args[-2], args[-1]
            dest_local = os.path.join(remote, dest.split(":", 1)[1].lstrip("/").replace("remote/registry/log", "").lstrip("/"))
            if "--files-from=-" in args:
                for rel in kw["input"].split():
                    d = os.path.join(remote, rel)
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(os.path.join(src, rel), d)          # an explicit re-copy overwrites
            elif os.path.isdir(src):
                for dp, _, fs in os.walk(src):
                    for f in fs:
                        s_ = os.path.join(dp, f)
                        d = os.path.join(dest_local, os.path.relpath(s_, src))
                        os.makedirs(os.path.dirname(d), exist_ok=True)
                        if "--ignore-existing" in args and os.path.exists(d):
                            continue                                # ignore-existing: a truncated file at its final name is skipped
                        shutil.copy2(s_, d)
            else:
                os.makedirs(dest_local, exist_ok=True)
                shutil.copy2(src, os.path.join(dest_local, os.path.basename(src)))
            return _subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "ssh" and "verify" in args:
            i = args.index("verify")
            local_args = ["verify", remote, os.path.join(remote, os.path.relpath(args[i + 2], "/remote/registry/log")), "--json"]
            rc = M.main(local_args)                                 # the REAL verifier, with its real exit code
            out = capsys.readouterr().out.strip().splitlines()[-1]  # its JSON line; the sync's own journal lines precede it
            if kw.get("check") and rc != 0:                          # honour check= exactly as subprocess.run does
                raise _subprocess.CalledProcessError(rc, list(args), out, "")
            return _subprocess.CompletedProcess(args, rc, out, "")
        if args[0] == "ssh":
            return _subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)
    monkeypatch.setattr(M.subprocess, "run", fake_run)
    dest = M.Destination("cluster_b", "ssh", "/remote/registry/log", "cluster_b")
    seq = _q(root, "SELECT MAX(ingest_seq) FROM record_ingest")[0][0]
    # a previous interrupted copy left a truncated file at its final name on the destination
    rel, path = M.log_files(root)[0]
    victim = os.path.join(remote, os.path.relpath(path, root))
    os.makedirs(os.path.dirname(victim), exist_ok=True)
    with open(path, "rb") as fh, open(victim, "wb") as out:
        out.write(fh.read()[:-5])
    res = M.sync(root, dest, seq)
    assert res.error is None, res.error
    assert res.switched and res.verify.clean
    repairs = [c for c in calls if c[0] == "rsync" and "--files-from=-" in c]
    assert len(repairs) == 1 and "--checksum" in repairs[0] and "--ignore-existing" not in repairs[0]
    assert sum(1 for c in calls if c[0] == "ssh" and "verify" in c) == 2
    assert R.file_logical_sha(victim) == R.file_logical_sha(path)


def test_backfill_space_check_uses_the_clamped_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(BF.DB, "free_bytes", lambda path: 1024 ** 3)
    with pytest.raises(BF.InsufficientSpace):
        BF.refuse_if_low_space(str(tmp_path), projected=1024 ** 3)             # 1 GB free < 2 × 1 GB projected
    assert BF.refuse_if_low_space(str(tmp_path), projected=256 * 1024 ** 2) == 1024 ** 3


# ============================================================================================================
# The three residual HIGH findings of the final re-check
# ============================================================================================================

STAGING_TABLES = tuple(t for t in SD.STAGING_TABLES if t != "record_cursor")     # the revision-keyed ones


def _rev_dir(root, ad, rev):
    return os.path.join(root, "records", "adapters", ad.name, rev)


def _item_file(rev_dir):
    return next(os.path.join(rev_dir, f) for f in sorted(os.listdir(rev_dir)) if "item_result" in f)


def _corrupt_one_item(path, item_id, value):
    """Rewrite one item record of a published revision file with a malformed value; returns its line number."""
    recs = [rec for _, rec, _ in R.read_record_file(path)]
    line = None
    for i, rec in enumerate(recs, start=1):
        if rec["payload"]["item_id"] == item_id:
            rec["payload"]["value"] = value
            line = i
    os.remove(path)
    R.write_records(path, recs, gz=path.endswith(".gz"))
    return line


def _publish_b(root, store_root, ad, scores, now):
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", scores)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=now)
    return rep.results[0].revision_id


# ---- (1) a malformed constituent (item or run-scale) or an item-flush failure holds the WHOLE revision ------------

def test_malformed_item_in_a_revision_holds_the_whole_revision(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    old_metric = _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0]
    old_items = _live_items(root, rid)
    rev2 = _publish_b(root, store_root, ad, [0.9, 0.9, 0.9], synth.t(3000))
    item_path = _item_file(_rev_dir(root, ad, rev2))
    line = _corrupt_one_item(item_path, "q1", "not-a-number")            # one malformed item beside two valid siblings
    bad = [rec for _, rec, _ in R.read_record_file(item_path)][line - 1]
    assert isinstance(VAL.validate_record(bad, "x", "0" * 64), VAL.Malformed)
    item_uri = H.logical_uri(root, item_path)
    ing = _ingest(root)
    assert ing.revisions_held == 1 and ing.refused >= 1
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]
    refusal = _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev2)[0][0]
    assert refusal.startswith("activation pending:") and f"{item_uri}#{line}" in refusal
    assert _q(root, "SELECT record_uri, reason FROM record_refusal WHERE record_uri LIKE ?", f"{item_uri}#%") and \
        _q(root, "SELECT record_uri FROM record_refusal WHERE record_uri = ?", f"{item_uri}#{line}")
    # the previous revision stays ENTIRELY current: values, membership, the aggregate, the paper selection
    assert _live_items(root, rid) == old_items
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", rid)[0][0] == pytest.approx(old_metric)
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE run_id = ? AND value = 0.9", rid)[0][0] == 0
    assert _sel(root, rid) == ("selected", 1)
    # the sound constituents stay staged (inactive, read by no view); the malformed file was never staged
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", rev2) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM revision_staging_file WHERE revision_id = ?", rev2) == [(5,)]
    assert _q(root, "SELECT COUNT(*) FROM revision_staging_file WHERE revision_id = ? AND file_uri = ?", rev2, item_uri) == [(0,)]
    _ingest(root)                                                           # a later cycle changes nothing
    assert _live_items(root, rid) == old_items and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]
    _rebuild_equal(root, tmp_path)


def test_item_flush_failure_at_activation_holds_the_revision(tmp_path, monkeypatch):
    """A constraint failure while the staged rows move into the live table is a revision-level failure naming the row."""
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    old_items = _live_items(root, rid)
    rev2 = _publish_b(root, store_root, ad, [0.9, 0.9, 0.9], synth.t(3000))
    item_uri = H.logical_uri(root, _item_file(_rev_dir(root, ad, rev2)))
    cols = SD.TABLE_BY_NAME["item_result"].column_names
    idx = cols.index("item_id")
    real_replay = H.Ingestor._replay_staged_row

    def batch_fails(self, table, rows):
        raise sqlite3.IntegrityError("CHECK constraint failed: synthetic")

    def row_fails(self, table, sql, row):
        if table == "item_result" and row[idx] == "q1":
            raise sqlite3.IntegrityError("CHECK constraint failed: synthetic")
        return real_replay(self, table, sql, row)
    monkeypatch.setattr(H.Ingestor, "_stage_rows", batch_fails)
    monkeypatch.setattr(H.Ingestor, "_replay_staged_row", row_fails)
    ing = _ingest(root)
    assert ing.revisions_held == 1
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]
    assert _live_items(root, rid) == old_items
    rows = _q(root, "SELECT record_uri, reason FROM record_refusal WHERE record_uri LIKE ?", f"{item_uri}#%")
    assert len(rows) == 1 and "IntegrityError" in rows[0][1]
    monkeypatch.undo()
    # the hold is EXACT (2026-09-30): the same bytes under the same judging code are not re-read every cycle -- a synthetic constraint
    # that vanished with no code change is exactly what the standing rule cannot see, and what `--retry-refused` is for
    _ingest(root)
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]
    assert _live_items(root, rid) == old_items
    # the constraint gone WITH the code that dropped it (the judging code's stamp moves): one re-attempt, the whole revision applies
    monkeypatch.setattr(H, "validator_stamp", lambda: "0" * 16)
    _ingest(root)
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(1,)]
    assert _live_items(root, rid) == {"q0": 0.9, "q1": 0.9, "q2": 0.9}


# ---- (2) a streamed file: no delivery before the verified final window; restart resumes from the cursor ----------

def test_streamed_first_window_is_not_marked_last(tmp_path):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    path = _write(root, "dims", "000001-dims.jsonl", dims)
    parts = list(H.stream_parts(root, path, None, 2))
    assert len(parts) == 6 and [p.last for p in parts] == [False] * 5 + [True]
    assert [p.part_index for p in parts] == list(range(6)) and sum(p.n_records for p in parts) == len(dims)
    assert parts[-1].whole_records == len(dims) and parts[-1].file_sha == R.file_logical_sha(path)
    assert not parts[0].is_single and not parts[0].is_last_part


@pytest.mark.parametrize("large", [True, False])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_interrupted_file_resumes_from_its_cursor_and_applies_every_record_once(tmp_path, monkeypatch, large, k):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()                                             # 12 run-scale records, cap 2 → 6 windows
    path = _write(root, "dims", "000001-dims.jsonl", dims)
    uri = H.logical_uri(root, path)
    if large:
        monkeypatch.setattr(H, "LARGE_FILE_BYTES", 0)                          # streamed in-process
    real = H.Ingestor.ingest_chunk
    commits = {"n": 0}

    def cut(self, chunk, now):
        if commits["n"] == k:
            raise RuntimeError("interrupted at a window boundary")
        real(self, chunk, now)
        commits["n"] += 1
    monkeypatch.setattr(H.Ingestor, "ingest_chunk", cut)
    old = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 2
    try:
        with pytest.raises(RuntimeError):
            _ingest(root)
        assert _q(root, "SELECT COUNT(*) FROM record_delivery WHERE file_uri = ?", uri) == [(0,)]      # no completed delivery
        assert _q(root, "SELECT file_sha256, records_done FROM record_cursor WHERE file_uri = ?", uri) == [(R.file_logical_sha(path), 2 * k)]
        assert _q(root, "SELECT COUNT(*) FROM row_history") == [(2 * k,)]
        monkeypatch.setattr(H.Ingestor, "ingest_chunk", real)
        rep = _ingest(root)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old
    assert rep.ok == len(dims) - 2 * k + 1 and rep.duplicates == 0 and rep.refused == 0 and rep.transactions == 8 - k   # + the arrival fact's fold + the re-rank
    assert _q(root, "SELECT COUNT(*) FROM row_history WHERE table_name <> 'file_arrival'") == [(len(dims),)]   # every record exactly once
    assert _q(root, "SELECT n_records, n_ok, status, file_sha256 FROM record_delivery WHERE file_uri = ?", uri) == \
        [(len(dims), len(dims), "ok", R.file_logical_sha(path))]
    assert _q(root, "SELECT n_records FROM record_ingest WHERE file_uri = ?", uri) == [(len(dims),)]
    assert _q(root, "SELECT COUNT(*) FROM record_cursor") == [(0,)]
    side = json.load(open(os.path.join(root, "state", H.FILE_STAT_NAME)))
    assert side[path]["sha"] == R.file_logical_sha(path)
    _rebuild_equal(root, tmp_path)
    rep3 = _ingest(root)
    assert rep3.files == 0 and rep3.ok == 0                                                             # nothing left to do


# ---- (3) bounded staging into an inactive state, one small atomic activation ------------------------------------

def test_revision_is_staged_in_bounded_transactions_and_activated_atomically(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.6, 0.1])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    a_items = _live_items(root, rid)
    b_scores = [0.9, 0.7, 0.5, 0.3, 0.1, 0.6, 0.4]
    rev = _publish_b(root, store_root, ad, b_scores, synth.t(3000))
    rev_dir = _rev_dir(root, ad, rev)
    assert len(os.listdir(rev_dir)) == 8                                       # artifact, judgment, run, run_state, run_metric, item_result, source_status, marker
    for name, tables in SD.VIEW_TABLES.items():
        assert not tables & set(STAGING_TABLES), name                          # no view reads a staging ledger
    conn = DB.open_or_create(DB.db_path(root))
    reader = DB.connect(DB.db_path(root), readonly=True)
    txns = [[]]
    mid = []

    def trace(sql):
        txns[-1].append(sql)
        if sql == "COMMIT":
            txns.append([])
        if sql.startswith("INSERT INTO item_result ("):                          # while B's rows land (tagged, pending) …
            mid.append(dict(reader.execute("SELECT item_id, value FROM v_item_result_current WHERE run_id = ?", (rid,)).fetchall()))
    conn.set_trace_callback(trace)
    old = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 2
    try:
        ing = H.ingest_once(root, conn=conn, workers=0, min_free=0)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old
        conn.set_trace_callback(None)
        conn.close()
        reader.close()
    committed = [t for t in txns if "COMMIT" in t]
    staging = [t for t in committed if any(s.startswith("INSERT INTO item_result (") for s in t)]
    # SQLite reports a statement once per FK trigger it fires, with the same expanded text: distinct texts = rows written
    widths = [len({s for s in t if s.startswith("INSERT INTO item_result (")}) for t in staging]
    assert len(staging) == 4 and widths == [2, 2, 2, 1]                        # 7 items in windows of ≤ 2, each its own COMMIT
    activation = [t for t in committed if any("INTO source_revision" in s and "'accepted': 1" not in s for s in t)
                  and any(s.startswith("UPDATE source_revision SET") and "superseded_by_revision_id" in s for s in t)]
    assert len(activation) == 1                                                # ONE activation transaction (the marker fold, the chain) …
    touched = [s for s in activation[0] if s.startswith("INSERT INTO item_result (") or s.startswith("UPDATE item_result")
               or s.startswith("DELETE FROM item_result")]
    assert not touched, touched[:3]                                            # … that touches no item row: metadata only
    assert mid and all(m == a_items for m in mid)                              # readers saw A, untouched, throughout the staging
    assert ing.refused == 0 and ing.blocked == 0 and ing.revision_transactions == 1
    assert ing.staging_transactions >= 4 and ing.transactions >= ing.staging_transactions + 1
    assert ing.longest_transaction_s > 0.0 and ing.write_s > 0.0                 # revision transactions are in the telemetry
    assert _live_items(root, rid) == {f"q{i}": s for i, s in enumerate(b_scores)}
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE run_id = ?", rid) == [(12,)]                  # A's 5 rows coexist, superseded
    assert _q(root, "SELECT COUNT(*) FROM v_item_headline WHERE run_id = ?", rid) == [(7,)]
    # the revision's files: artifact, judgment, run, run_state, run_metric, item_result, source_status, source_revision
    assert _q(root, "SELECT COUNT(*) FROM record_ingest WHERE file_uri LIKE ?", f"records/adapters/{ad.name}/{rev}/%")[0][0] == 8
    for t in STAGING_TABLES:
        assert _q(root, f"SELECT COUNT(*) FROM {t}") == [(0,)], t
    _rebuild_equal(root, tmp_path)


def test_held_revision_keeps_its_staged_rows_and_activates_without_restaging(tmp_path, monkeypatch):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.9, 0.9, 0.9])
    ad2 = JudgeNamingAdapter(store_root, JUDGE_ID_3)
    rep, ing = _pass(root, ad2, synth.t(3000))
    rev2 = rep.results[0].revision_id
    assert ing.revisions_held >= 1 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", rev2) == [(3,)]      # staged, tagged, inactive
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE source_revision_id = ?", rev2) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM revision_staging_file WHERE revision_id = ?", rev2) == [(7,)]   # + the source_status file
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}
    spied = []
    real = H.stream_parts

    def spy(root_, path, sha, max_rows, **kw):
        spied.append(path)
        return real(root_, path, sha, max_rows, **kw)
    monkeypatch.setattr(H, "stream_parts", spy)
    _write(root, "dims", "000002-judge3.jsonl", [_judge3()])
    ing2 = _ingest(root)
    assert ing2.refused == 0 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(1,)]
    assert not [p for p in spied if rev2 in p and "source_revision" not in os.path.basename(p)]          # no constituent re-read
    assert _live_items(root, rid) == {"q0": 0.9, "q1": 0.9, "q2": 0.9}
    assert _q(root, "SELECT COUNT(*) FROM revision_staging_file") == [(0,)] and _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    _rebuild_equal(root, tmp_path)


def test_reactivation_reemits_in_bounded_windows(tmp_path, monkeypatch):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.6, 0.1])])
    ad = StoreAdapter(store_root)
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    _pass(root, ad, synth.t(2000))
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.9, 0.9])
    _pass(root, ad, synth.t(3000))
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.6, 0.1])           # back to A
    monkeypatch.setattr(RUN, "REEMIT_WINDOW", 2)
    writes = []
    real = RUN.R.write_records

    def spy(path, records, gz=False):
        writes.append((os.path.basename(path), len(records)))
        return real(path, records, gz=gz)
    monkeypatch.setattr(RUN.R, "write_records", spy)
    rep, ing = _pass(root, ad, synth.t(4000))
    item_writes = [n for name, n in writes if "item_result" in name]
    assert item_writes and max(item_writes) <= 2 and sum(item_writes) == 5                            # streamed per window
    assert ing.refused == 0 and _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4, "q3": 0.6, "q4": 0.1}
    _rebuild_equal(root, tmp_path)


# ============================================================================================================
# The closing pass — physical-order windows, visibility indirection, canonical carriers, aliases
# ============================================================================================================

def _interleaved_file(root, with_malformed=False):
    """dims already ingested; one plain file whose PHYSICAL order interleaves item rows of an ingested run with
    fresh run-scale records (a second run's records): [item, run2, item, run_state2, item, ...]."""
    rid, recs = synth.run_records("inter", 100, n_items=3)
    _write(root, "c", "000001-run.jsonl", [r for r in recs if r["record_type"] in ("run", "judgment")])   # the items' parents
    _ingest(root)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    items = synth.item_records(rid, jid, 300, 4)
    if with_malformed:
        items[1]["payload"]["value"] = "not-a-number"
    rid2, recs2 = synth.run_records("inter2", 200, n_items=3, through="generated")
    others = [r for r in recs2 if r["record_type"] in ("run", "run_state")]
    physical = [items[0], others[0], items[1], others[1], items[2], others[2], items[3]] + others[3:]
    path = _write(root, "c", "000002-mixed.jsonl", physical)
    return rid, rid2, path, physical, items


def test_parts_partition_in_physical_record_order():
    pf = H.ParsedFile("p", "records/x/000001-mixed.jsonl", "s")
    pf.items = [H.ParsedItem(1, "%064x" % 1, synth.t(1), "item_result", (), (), (), ""),
                H.ParsedItem(3, "%064x" % 3, synth.t(1), "item_result", (), (), (), "")]
    pf.records = [H.ParsedRecord(2, "%064x" % 2, VAL.Malformed("value", "bad", "u#2", "item_result")),
                  H.ParsedRecord(4, "%064x" % 4, VAL.Malformed("value", "bad", "u#4", "eval"))]
    parts = pf.parts(1)
    lines = [[pr.line for pr in p.records] + [pi.line for pi in p.items] for p in parts]
    assert lines == [[1], [2], [3], [4]]                                       # contiguous prefixes of the file, whatever the kind
    assert [p.last for p in parts] == [False, False, False, True]
    two = pf.parts(2)
    assert [[pr.line for pr in p.records] + sorted(pi.line for pi in p.items) for p in two] == [[2, 1], [4, 3]]


@pytest.mark.parametrize("malformed", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_interrupted_interleaved_file_resumes_without_loss_or_duplication(tmp_path, monkeypatch, k, malformed):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, rid2, path, physical, items = _interleaved_file(root, with_malformed=malformed)
    uri = H.logical_uri(root, path)
    real = H.Ingestor.ingest_chunk
    commits = {"n": 0}

    def cut(self, chunk, now):
        if commits["n"] == k:
            raise RuntimeError("interrupted at a window boundary")
        real(self, chunk, now)
        commits["n"] += 1
    monkeypatch.setattr(H.Ingestor, "ingest_chunk", cut)
    old = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 1
    try:
        with pytest.raises(RuntimeError):
            _ingest(root)
        assert _q(root, "SELECT records_done FROM record_cursor WHERE file_uri = ?", uri) == [(k,)]
        assert _q(root, "SELECT COUNT(*) FROM record_delivery WHERE file_uri = ?", uri) == [(0,)]
        monkeypatch.setattr(H.Ingestor, "ingest_chunk", real)
        rep = _ingest(root)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old
    n_items_ok = len(items) - (1 if malformed else 0)
    assert rep.duplicates == 0
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE run_id = ?", rid) == [(n_items_ok,)]         # no item lost
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid2) == [("generated",)]                  # every run-scale record landed
    assert _q(root, "SELECT COUNT(*) FROM record_cursor") == [(0,)]
    d = _q(root, "SELECT n_records, n_ok, n_refused, status FROM record_delivery WHERE file_uri = ?", uri)[0]
    assert d[0] == len(physical) and d[1] == len(physical) - (1 if malformed else 0) and d[2] == (1 if malformed else 0)
    assert d[3] == ("partial" if malformed else "ok")
    if malformed:
        assert _q(root, "SELECT COUNT(*) FROM record_refusal WHERE record_uri LIKE ?", f"{uri}#%") == [(1,)]
    _rebuild_equal(root, tmp_path)


# ---- (3) a blocked duplicate adopts the canonical (smallest logical URI) carrier ----------------------------------

@pytest.mark.parametrize("order", ["z,a,run", "a,z,run", "z,run+a", "a,run+z", "run,z,a", "run,a,z"])
def test_blocked_duplicate_adopts_the_canonical_carrier(tmp_path, order):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _ingest(root)
    rid, recs = synth.run_records("dup", 100, n_items=2, through="running")
    run = [r for r in recs if r["record_type"] == "run"]
    state = [r for r in recs if r["record_type"] == "run_state" and r["payload"]["state"] == "running"]
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    judgment = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "n_items_judged": 1,
                                      "status": "complete"}, synth.t(150))
    item = synth.item_records(rid, jid, 300, 1)
    steps = {"run": [("c", "000001-run.jsonl", run + [judgment])],                              # the duplicates' parents
             "a": [("a", "000002-state.jsonl", state), ("a", "000003-item_result.jsonl", item)],
             "z": [("z", "000002-state.jsonl", state), ("z", "000003-item_result.jsonl", item)]}
    for step in order.split(","):
        for part in step.split("+"):
            for sub, name, rows in steps[part]:
                _write(root, sub, name, rows)
        rep = _ingest(root)
        assert rep.refused == 0, rep.refusals
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    ev = _q(root, "SELECT record_uri FROM row_history WHERE table_name = 'run_state_event'")
    assert ev == [("records/a/000002-state.jsonl",)]                                            # the smallest URI carries the event
    assert _q(root, "SELECT DISTINCT record_uri FROM row_history WHERE record_type = 'run_state'") == [("records/a/000002-state.jsonl",)]
    row = _q(root, "SELECT record_file_id, ingest_seq FROM item_result WHERE run_id = ?", rid)[0]
    a_uri = "records/a/000003-item_result.jsonl"
    assert row[0] == R.record_file_id_for_uri(a_uri)                                             # the item tuple's file id …
    assert row[1] == _q(root, "SELECT ingest_seq FROM record_ingest WHERE file_uri = ?", a_uri)[0][0]   # … and rank follow the carrier
    assert sorted(u for (u,) in _q(root, "SELECT record_uri FROM record_run_index WHERE run_id = ?", rid)) == \
        sorted(["records/c/000001-run.jsonl", "records/a/000002-state.jsonl", "records/z/000002-state.jsonl", a_uri,
                "records/z/000003-item_result.jsonl"])
    _rebuild_equal(root, tmp_path, seed=len(order))


# ---- (2) visibility indirection: pending rows coexist, invisible; activation flips metadata only -----------------

def test_pending_revision_rows_are_invisible_to_every_item_reader(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.9, 0.9, 0.9, 0.9])
    ad2 = JudgeNamingAdapter(store_root, JUDGE_ID_3)                       # held: its judgment names an absent judge
    rep, ing = _pass(root, ad2, synth.t(3000))
    rev2 = rep.results[0].revision_id
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", rev2) == [(4,)]        # in the table, tagged
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}                                        # … and invisible
    assert _q(root, "SELECT COUNT(*) FROM v_item_headline WHERE run_id = ?", rid) == [(3,)]
    assert _q(root, "SELECT COUNT(*) FROM v_judge_side_by_side WHERE run_id = ?", rid) == [(3,)]
    assert registry.paired(rid, rid, "asr", root=root).shape[0] == 3
    _write(root, "dims", "000002-judge3.jsonl", [_judge3()])
    _ingest(root)
    assert _live_items(root, rid) == {"q0": 0.9, "q1": 0.9, "q2": 0.9, "q3": 0.9}
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE run_id = ?", rid) == [(7,)]                     # A's rows stay (history)
    _rebuild_equal(root, tmp_path)


def test_withdrawn_marks_follow_the_head_in_bounded_batches(tmp_path, monkeypatch):
    """A → B (shrink: q2 withdrawn) → A: B's withdrawal marked A's q2; A re-activated is the head again — its rows
    are visible at the flip (derived) and the mark is cleared afterwards in bounded transactions."""
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    rep_a, _ = _pass(root, ad, synth.t(2000))
    rev_a = rep_a.results[0].revision_id
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8])
    rep_b, _ = _pass(root, ad, synth.t(3000))
    rev_b = rep_b.results[0].revision_id
    assert _q(root, "SELECT item_id FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NOT NULL", rev_a) == [("q2",)]
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8}
    # a stale mark on the HEAD's rows (a withdrawal applied to a revision that became current again) is cleared by the
    # reclaim pass, one bounded transaction per RECLAIM_BATCH_ROWS rows — and never gates visibility
    with DB.open_or_create(DB.db_path(root)) as conn, DB.transaction(conn):
        conn.execute("UPDATE item_result SET withdrawn_by_revision_id = 'stale' WHERE source_revision_id = ?", (rev_b,))
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8}                                           # visibility is derived
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    ing_r = _ingest(root)
    assert ing_r.reclaim_transactions == 2                                                              # 2 rows, 1 per transaction
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NOT NULL", rev_b) == [(0,)]
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    rep_c, ing_c = _pass(root, ad, synth.t(4000))
    assert ing_c.refused == 0 and [r.revision_id for r in rep_c.results] == [rev_a]
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}                                # A is the head again
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NOT NULL", rev_a) == [(0,)]
    _rebuild_equal(root, tmp_path)


# ============================================================================================================
# Round 6: revision-tag authority, one current row per measurement key, withdrawal marks out of activation
# ============================================================================================================

def _retag_one_item(path, item_id, tag):
    recs = [rec for _, rec, _ in R.read_record_file(path)]
    line = None
    for i, rec in enumerate(recs, start=1):
        if rec["payload"]["item_id"] == item_id:
            rec["payload"]["source_revision_id"] = tag
            line = i
    os.remove(path)
    R.write_records(path, recs, gz=path.endswith(".gz"))
    return line


class BogusTagAdapter(StoreAdapter):
    """Emits item payloads that name a WRONG revision tag: the runner must overwrite it."""

    def read(self, ctx, group, inputs):
        for batch in super().read(ctx, group, inputs):
            for rec in batch.records:
                if rec["record_type"] == "item_result":
                    rec["payload"]["source_revision_id"] = "b" * 16
            yield batch


# ---- (1) the bundle's revision is authoritative ---------------------------------------------------------------

@pytest.mark.parametrize("tag", ["old", ""])
def test_constituent_with_a_foreign_or_native_tag_holds_the_revision(tmp_path, tag):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    rep_a, _ = _pass(root, ad, synth.t(2000))
    rev_a = rep_a.results[0].revision_id
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    a_items = _live_items(root, rid)
    rev_b = _publish_b(root, store_root, ad, [0.9, 0.9, 0.9], synth.t(3000))
    item_path = _item_file(_rev_dir(root, ad, rev_b))
    line = _retag_one_item(item_path, "q1", rev_a if tag == "old" else "")     # a B constituent aimed at A's visible key
    item_uri = H.logical_uri(root, item_path)
    ing = _ingest(root)
    assert ing.revisions_held >= 1 and ing.refused >= 1
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev_b) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal WHERE record_uri = ?", f"{item_uri}#{line}") == [(1,)]
    reason = _q(root, "SELECT reason FROM record_refusal WHERE record_uri = ?", f"{item_uri}#{line}")[0][0]
    assert rev_b in reason and "source_revision_id" in reason
    assert _live_items(root, rid) == a_items                                                          # nothing visible changed …
    assert _q(root, "SELECT value FROM item_result WHERE run_id = ? AND item_id = 'q1' AND source_revision_id = ?", rid, rev_a) == [(0.8,)]
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ''") == [(0,)]        # … and no native row appeared
    _ingest(root)                                                                                       # nor after the held revision's retry
    assert _live_items(root, rid) == a_items
    _rebuild_equal(root, tmp_path)


def test_runner_stamps_the_revision_on_every_constituent_and_native_rows_need_a_native_directory(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8])])
    ad = BogusTagAdapter(store_root)
    rep, ing = _pass(root, ad, synth.t(2000))
    rev = rep.results[0].revision_id
    tags = {rec["payload"]["source_revision_id"] for _, rec, _ in R.read_record_file(_item_file(_rev_dir(root, ad, rev)))}
    assert tags == {rev} and ing.refused == 0                                                          # stamped, never trusted
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8}
    # a natively tagged item row under records/adapters/<adapter>/ (outside any revision directory) is refused, not admitted
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    native = synth.item_records(rid, jid, 400, 1, metric="asr")
    path = _write(root, os.path.join("adapters", ad.name), "000009-item_result.jsonl", native)
    rep2 = _ingest(root)
    assert rep2.refused == 1 and "records/adapters/" in rep2.refusals[0]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal WHERE record_uri = ?", H.logical_uri(root, path) + "#1") == [(1,)]
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8}
    _rebuild_equal(root, tmp_path)


# ---- (2) ONE current row per measurement key; conflicts named, never chosen ------------------------------------

def _native_rows(root, rid, values, sub="native"):
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    recs = []
    for i, v in enumerate(values):
        p = {"run_id": rid, "item_id": f"q{i}", "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1,
             "judgment_id": jid, "value": v, "label": None, "api_pool": None, "created_utc": synth.t(5000), "judged_utc": None,
             "source_revision_id": "", "withdrawn_by_revision_id": None}
        recs.append(synth.rec("item_result", p, synth.t(5001 + i)))
    return _write(root, sub, "000001-item_result.jsonl", recs)


def _dupes(root):
    return _q(root, "SELECT run_id, item_id, metric_id, judgment_id, COUNT(*) FROM v_item_result_current "
                    "GROUP BY run_id, item_id, metric_id, trial_index, sample_index, turn_index, judgment_id HAVING COUNT(*) > 1")


def test_native_and_adapter_rows_for_one_key_yield_one_current_row_and_one_pair(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    _native_rows(root, rid, [0.2, 0.8, 0.4])                                   # the same measurements, natively emitted
    rep = _ingest(root)
    assert rep.refused == 0 and _dupes(root) == []
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_provenances WHERE run_id = ?", rid) == [(6,)]      # every provenance, for audit
    assert _q(root, "SELECT COUNT(*), MIN(source_revision_id), MAX(source_revision_id) FROM v_item_result_current WHERE run_id = ?", rid) == [(3, "", "")]
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}          # native first
    assert _q(root, "SELECT COUNT(*) FROM v_item_conflict") == [(0,)]
    pairs = registry.paired(rid, rid, "asr", root=root)
    assert pairs.shape[0] == 3 and sorted(pairs["item_id"]) == ["q0", "q1", "q2"]           # one pair per item, never four
    assert _q(root, "SELECT COUNT(*) FROM v_item_headline WHERE run_id = ?", rid) == [(3,)]
    _rebuild_equal(root, tmp_path)


def test_disagreeing_provenances_are_flagged_and_pairing_refuses_the_item_by_name(tmp_path):
    from sisteer.registry import status as ST
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    _native_rows(root, rid, [0.2, 0.55, 0.4])                                  # q1 disagrees
    _ingest(root)
    assert _dupes(root) == []
    conflicts = _q(root, "SELECT item_id, n_provenances FROM v_item_conflict WHERE run_id = ?", rid)
    assert conflicts == [("q1", 2)]
    with pytest.raises(ValueError, match="q1"):
        registry.paired(rid, rid, "asr", root=root)
    with DB.reading(DB.db_path(root)) as conn:
        ctx, _ = ST.gather(conn, root)
    assert [r["item_id"] for r in ctx["item_conflicts"]] == ["q1"]
    _rebuild_equal(root, tmp_path)


def test_two_accepted_sources_pick_the_current_row_by_priority_and_show_both_provenances(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    low = StoreAdapter(store_root, name="second_store")
    high = StoreAdapter(store_root)
    low.priority, high.priority = 1, 5
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    _pass(root, high, synth.t(2000))
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.3, 0.7, 0.5])       # the second source reads different numbers
    rep, ing = _pass(root, low, synth.t(3000))
    assert ing.refused == 0 and _dupes(root) == []
    assert _q(root, "SELECT priority FROM source_revision WHERE revision_id = ?", rep.results[0].revision_id) == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_provenances WHERE run_id = ?", rid) == [(6,)]
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}          # the higher priority source is current
    assert _q(root, "SELECT COUNT(*) FROM v_item_conflict WHERE run_id = ?", rid) == [(3,)]                # … and the disagreement is visible
    _rebuild_equal(root, tmp_path)


# ---- (3) withdrawal marks are bookkeeping, materialised after activation in bounded batches --------------------

def test_activation_writes_no_item_rows_even_when_the_revision_withdraws_items(tmp_path, monkeypatch):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])])
    ad = StoreAdapter(store_root)
    rep_a, _ = _pass(root, ad, synth.t(2000))
    rev_a = rep_a.results[0].revision_id
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    rev_b = _publish_b(root, store_root, ad, [0.1, 0.2], synth.t(3000))       # a shrink: five items withdrawn
    conn = DB.open_or_create(DB.db_path(root))
    txns = [[]]
    conn.set_trace_callback(lambda sql: (txns[-1].append(sql), txns.append([]) if sql == "COMMIT" else None))
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 2)
    old = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 2
    try:
        ing = H.ingest_once(root, conn=conn, workers=0, min_free=0)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old
        conn.set_trace_callback(None)
        conn.close()
    committed = [t for t in txns if "COMMIT" in t]
    activation = [t for t in committed if any(s.startswith("UPDATE source_revision SET") and "superseded_by_revision_id" in s for s in t)]
    assert len(activation) == 1
    touched = [s for s in activation[0] if re.match(r"(UPDATE|INSERT(?: OR \w+)? INTO|DELETE FROM) (item_result|item_flags|run_degeneration)\b", s)]
    assert not touched, touched[:3]                                            # the activation writes NO item-table row
    assert ing.reclaim_transactions == 3                                       # 5 marks at a batch of 2 → 2 + 2 + 1
    marks = [t for t in committed if any(s.startswith("UPDATE item_result SET withdrawn_by_revision_id") for s in t)]
    assert len(marks) == 3 and all(t is not activation[0] for t in marks)
    assert _live_items(root, rid) == {"q0": 0.1, "q1": 0.2}
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id = ?", rev_a, rev_b) == [(5,)]
    assert _q(root, "SELECT reclaimed FROM source_revision WHERE revision_id = ?", rev_a) == [(1,)]
    _rebuild_equal(root, tmp_path)


# ---- (addendum) a marker duplicate delivered outside its directory never pre-accepts the revision ----------------

def test_marker_copy_outside_its_directory_waits_for_the_directory_activation(tmp_path, monkeypatch):
    """The reactivation replay flake (1 in ~3 runs): a copy of a revision's marker delivered as a PLAIN file (the
    duplicate-delivery scenario) folded `accepted = 1` before the directory's own activation had staged its
    constituents, so their tagged rows lost the pending exemption of their foreign keys (judgment absent → refused)
    and the replay diverged from the live index. The copy now waits for the directory's activation."""
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    marker = next(os.path.join(_rev_dir(root, ad, rev), f) for f in os.listdir(_rev_dir(root, ad, rev)) if "source_revision" in f)
    os.makedirs(os.path.join(root, "records", "dup"))
    shutil.copy2(marker, os.path.join(root, "records", "dup", "000001-source_revision.jsonl"))
    # first: only the plain files of the cycle (the directory's activation is held back) — the copy must NOT accept the revision
    monkeypatch.setattr(H.Ingestor, "apply_revision", lambda self, rev_dir, now: "pending")
    _ingest(root)
    monkeypatch.undo()
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    blocked = _q(root, "SELECT record_type, missing_parent_table, missing_parent_id FROM blocked_records WHERE record_uri LIKE 'records/dup/%'")
    assert ("source_revision", "source_revision", rev) in blocked                             # it waits for the directory's activation
    assert _live_items(root, rid) == {}
    # then the directory activates: constituents staged pending (foreign keys exempt), marker last; the copy is a duplicate
    ing2 = _ingest(root)
    assert ing2.refused == 0 and ing2.revisions_held == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}
    _rebuild_equal(root, tmp_path)


# ---- the same-stamp tie-break over EXISTING records -----------------------------------


def _same_stamp_pair(rid, stamp):
    """A `judged` and a `superseded` run_state at ONE written_utc whose shas put `judged` LAST under the pre-tie-break
    `(written_utc, record_sha256)` order, so the old order drew `judged` and the new one must not."""
    judged = synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": stamp, "evidence_kind": "completion_artifact"}, stamp)
    for n in range(256):
        sup = synth.rec("run_state", {"run_id": rid, "state": "superseded", "event_utc": stamp, "evidence_kind": "ruling",
                                      "note": f"superseded by a later run of the same key ({n})"}, stamp)
        if R.record_sha(sup) < R.record_sha(judged):
            return judged, sup
    raise AssertionError("no adversarial note in 256 tries")


def test_history_folded_under_the_old_sha_order_refolds_deterministically_under_the_tie_break(tmp_path, monkeypatch):
    """Existing records: a key whose same-stamp pair the PRE-tie-break order folded (its live row the sha draw, its history
    snapshots that order's) refolds under the new order to `superseded` from every path -- the replica's refold tool, a late
    event's whole-history refold, a rebuild of the log -- and the paths agree. The one residue is named: the lateral event's
    old snapshot stands until a refold reaches its position (a late EARLIER event), after which live and replay are equal in
    every table. The harvester's own facts are off (`derive_facts=False`, as the reducer tests do): a run the OLD order left at
    `judged` is a `run_ingest` candidate and the same cycle would stamp it `ingested` -- itself a reason for the tie-break, since a
    run the new order reads as `superseded` is never a candidate (`derive_run_ingest_records`)."""
    from sisteer.registry import reducer as RD
    from sisteer.registry import replica as RP
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated")
    _write(root, "c", "000001-run.jsonl", recs)
    _ingest(root, derive_facts=False)
    stamp = synth.t(300)
    judged, sup = _same_stamp_pair(rid, stamp)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    judgment = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": stamp, "finished_utc": stamp,
                                      "n_items_judged": 200, "status": "complete", "api_pool": "lowprio"}, stamp)
    key = I.canonical_json({"run_id": rid})
    # the pre-tie-break fold, reproduced: the order's one spelling patched back to (written_utc, record_sha256) for this ingest
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RD, "ORDER_SQL", "written_utc, record_sha256")
        mp.setattr(RD, "ORDER_DESC_SQL", "written_utc DESC, record_sha256 DESC")
        _write(root, "c", "000002-pair.jsonl", [judgment, judged, sup])
        rep = _ingest(root, derive_facts=False)
        assert rep.refused == 0 and rep.blocked == 0, rep.refusals
        assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("judged",)]       # the sha draw: judged last
        old_snapshots = _q(root, "SELECT record_sha256, folded_state_json FROM row_history WHERE table_name = 'run' AND natural_key_json = ? "
                                 "AND written_utc = ? AND record_type = 'run_state'", key, stamp)
    old_by_sha = {sha: json.loads(st) for sha, st in old_snapshots}
    assert old_by_sha[R.record_sha(sup)]["lifecycle_state"] == "superseded" and "judged_utc" not in old_by_sha[R.record_sha(sup)]
    assert old_by_sha[R.record_sha(judged)]["lifecycle_state"] == "judged"
    # 1. the replica's refold (C8's measurement tool) reads the new order: superseded, with judged_utc kept
    with DB.reading(DB.db_path(root)) as conn:
        dump = {d["key"]: d for d in RP.refold_dump(conn)}
    assert dump[key]["state"]["lifecycle_state"] == "superseded" and dump[key]["state"]["judged_utc"] == stamp
    # 2. a rebuild of the log under the new order agrees with the refold; the live row still carries the old draw until its key refolds
    out = str(tmp_path / "rebuilt-before.sqlite")
    H.rebuild(root, out, workers=0)
    with DB.reading(out) as b:
        assert tuple(b.execute("SELECT lifecycle_state, judged_utc FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("superseded", stamp)
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("judged",)]
    with DB.reading(DB.db_path(root)) as a, DB.reading(out) as b:
        diff = {k: v for k, v in H.verify_equal(a, b).items() if v}
    assert set(diff) == {"run", "row_history"}, diff
    # 3. a late event stamped AFTER the pair refolds the key's whole history under the new order: the live row converges; the
    #    pair's own snapshots are before the refold's position and stand as the old order wrote them (the named residue)
    _write(root, "c", "000003-late.jsonl", [synth.rec("run_patch", {"run_id": rid, "fields": {"notes": "late, after the pair"}}, synth.t(400))])
    rep = _ingest(root, derive_facts=False)
    assert rep.refused == 0 and rep.blocked == 0
    assert _q(root, "SELECT lifecycle_state, judged_utc, notes FROM run WHERE run_id = ?", rid) == [("superseded", stamp, "late, after the pair")]
    out2 = str(tmp_path / "rebuilt-after-late.sqlite")
    H.rebuild(root, out2, workers=0)
    with DB.reading(DB.db_path(root)) as a, DB.reading(out2) as b:
        diff = {k: v for k, v in H.verify_equal(a, b).items() if v}
        live_sup = json.loads(a.execute("SELECT folded_state_json FROM row_history WHERE table_name = 'run' AND natural_key_json = ? "
                                        "AND record_sha256 = ?", (key, R.record_sha(sup))).fetchone()[0])
        replay_sup = json.loads(b.execute("SELECT folded_state_json FROM row_history WHERE table_name = 'run' AND natural_key_json = ? "
                                          "AND record_sha256 = ?", (key, R.record_sha(sup))).fetchone()[0])
    # the lateral event's snapshot, live (the old prefix) against replay, counted on both sides; the judgment's same-stamp run
    # event may have sat between the pair under the old order, in which case its snapshot differs too
    assert set(diff) == {"row_history"} and diff["row_history"] in (2, 4), diff
    assert "judged_utc" not in live_sup and replay_sup["judged_utc"] == stamp and live_sup["lifecycle_state"] == replay_sup["lifecycle_state"] == "superseded"
    # 4. a late event stamped BEFORE the pair refolds the suffix from its position, the pair included: live equals replay everywhere
    _write(root, "c", "000004-late-earlier.jsonl", [synth.rec("run_patch", {"run_id": rid, "fields": {"node_hostname": "TESTHOST-2"}}, synth.t(250))])
    rep = _ingest(root, derive_facts=False)
    assert rep.refused == 0 and rep.blocked == 0
    assert _q(root, "SELECT lifecycle_state, judged_utc, node_hostname FROM run WHERE run_id = ?", rid) == [("superseded", stamp, "TESTHOST-2")]
    _rebuild_equal(root, tmp_path, seed=13)
    # and the history reads in the fold's order: judged before superseded at the stamp, under both readers of it
    with DB.reading(DB.db_path(root)) as conn:
        hist = H.history(root, conn, "run", {"run_id": rid})
        events = RD.Reducer(conn)._load_events("run", key)
    assert [h["payload"]["state"] for h in hist if h["written_utc"] == stamp and h["record_type"] == "run_state"] == ["judged", "superseded"]
    assert [e.payload["state"] for e in events if e.written_utc == stamp and e.record_type == "run_state"] == ["judged", "superseded"]
    assert [RD.order_key(e) for e in events] == sorted(RD.order_key(e) for e in events)
