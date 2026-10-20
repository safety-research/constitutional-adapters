"""Declared-record adapters END TO END through the runner (adapters/runner.py) and the harvester into a temp registry DB — closing
the gap where records had only been validated in memory — on synthetic fixtures:

* a GSM8K paired-200 report becomes an accepted revision whose runs, metrics, degeneration rows and per-item rows land in the DB, the
  item-scale rows stamped with the runner's revision and current, the emitter-minted instrument / computation rows resolving the
  runs' FKs, the `strength_unit` column filled -- in TWO passes since the shared emitter consults the anchor currency
  (no provisional anchors): the first pass over the fresh registry parks the contrast by name (its own
  unsteered run is emitted by that very read, the frame does not know it), the fold makes the run current and the anchor decision input
  moves, the second pass re-reads the group under a new revision id and files the contrast `own_current`; a third pass writes nothing;
* the cap-4096 lane-file declarations adapter states `source_state` rows under EXACTLY the paths the round's rulings reference, in the
  rulings' `~`-relative spelling, so a `grades_root_absent` ruling the harvester evaluates flips from `missing`
  to `ok` + expired once the adapter has run; a ruled path the file no longer resolves reads `path_absent`, and a path the source stated
  before is re-stated absent rather than left stale;
* the paths declared in the adapter equal the ones the checked-in rulings file names (configs/registry/rulings_r2_v1.jsonl; never skipped).

Fixtures are invented (tests/registry_declared_fixtures.py); nothing here reads a real store.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import cap4096_declarations as C4
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN

import registry_declared_fixtures as F  # noqa: E402

NOW = "2026-09-17T00:00:00.000Z"
LATER = "2026-09-17T01:00:00.000Z"


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _declared_dims():
    """The dimension rows a GSM8K declared record resolves against (the seeds' ids; synthetic contents)."""
    t = synth.t
    return synth.seed_dimensions() + [
        synth.rec("model", {"model_id": "gemma4e4b_it", "family": "gemma4", "size_label": "e4b", "checkpoint_kind": "instruct",
                            "hf_repo": "synthetic/gemma-4-E4B-it", "hf_revision": "unverified", "chat_template_sha256": "unverified",
                            "generation_config_present": 0, "thinking_channel": "available", "paper_status": "in_paper",
                            # the dimension tables carry field_confidence_json: a literal `unverified` needs its reason beside it
                            "field_confidence_json": {c: {"confidence": "unverified", "reason": "synthetic fixture: never read"}
                                                      for c in ("hf_revision", "chat_template_sha256")}}, t(20)),
        synth.rec("eval", {"eval_id": "gsm8k", "eval_category": "capability", "headline_metric_id": "acc", "metric_direction": "higher_is_better",
                           "attacker_required": 0, "paper_status": "in_paper"}, t(21)),
        synth.rec("item_set", {"item_set_id": "gsm8k_platinum_paired_200", "eval_id": "gsm8k", "dataset_version": "synthetic/gsm8k-platinum@paired200",
                               "n_items": 200, "items_sha256": "unverified", "source_uri": "synthetic://items/gsm8k_paired_200.jsonl",
                               "item_id_scheme": "gsm8k_<hex8>", "subsample_rule": "full",
                               "field_confidence_json": {"items_sha256": {"confidence": "unverified", "reason": "synthetic fixture: never read"}}}, t(22)),
        synth.rec("object", synth.object_lora(F.SHA_A, "constitution_lora", "constitution"), t(23)),
    ]


def _pass(root, adapter, now):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return RUN.run_pass(root, [adapter], conn=conn, now=now)
    finally:
        conn.close()


def test_gsm8k_declared_record_ingests_through_the_runner(tmp_path):
    from sisteer.registry.adapters import gsm8k_p200 as G8
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", _declared_dims())
    ing0 = H.ingest_once(root, workers=0, min_free=0)
    assert ing0.refused == 0
    fx = F.build_gsm8k_report(str(tmp_path / "g8k"))
    ad = G8.Gsm8kP200Adapter(report=fx["path"], later_reports=(), expect_pin=G8.STORE_PIN)
    # ---- pass 1 over the fresh registry: every record of the report lands; the ONE contrast parks by name (its unsteered run is not folded yet)
    rep1 = _pass(root, ad, "2026-09-16T23:00:00.000Z")
    # the report's group and the root pseudo-group (`_root`, the root inventory's publisher: gsm8k_p200.ROOT_LISTING_RULE), in explicit mode too
    assert rep1.groups_refused == 0 and rep1.groups_refreshed == 2 and rep1.records_written > 0, [r.error for r in rep1.results]
    assert sorted(r.group for r in rep1.results) == sorted(["g8k", G8.ROOT_GROUP])
    res1 = next(r for r in rep1.results if r.group == "g8k")
    ing1 = H.ingest_once(root, workers=0, min_free=0)
    assert ing1.refused == 0 and ing1.revisions_held == 0 and ing1.blocked == 0, ing1
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT accepted FROM source_revision WHERE revision_id = ?", (res1.revision_id,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 0
    parks = conn.execute("SELECT json_extract(rm.extra_json, '$.delta_block.anchor.state'), json_extract(rm.extra_json, '$.delta_block.anchor.own_reason') FROM run_metric rm "
                         "JOIN run r ON r.run_id = rm.run_id WHERE r.eval_id = 'gsm8k' AND r.object_id <> 'none'").fetchall()
    assert [tuple(r) for r in parks] == [("no_current_anchor", "not_in_frame")], parks
    block1 = json.loads(conn.execute("SELECT extra_json FROM source_revision WHERE revision_id = ?", (res1.revision_id,)).fetchone()[0])["anchor_decisions"]
    assert block1["n_comparisons"] == block1["n_parked"] == 1 and block1["pre_read_fingerprint"] is None
    conn.close()
    # ---- pass 2: the unsteered run is current now, the decision input moved -> a NEW revision id, the contrast files own_current
    rep = _pass(root, ad, NOW)
    assert rep.groups_refused == 0 and rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and rep.records_written > 0, [r.error for r in rep.results]
    res = rep.results[0]
    assert res.group == "g8k" and res.revision_id != res1.revision_id         # the root listing did not move: the root group is quiet
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.revisions_held == 0 and ing.blocked == 0, ing
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT accepted FROM source_revision WHERE revision_id = ?", (res.revision_id,)).fetchone()[0] == 1
    runs = {r[0]: r for r in conn.execute("SELECT object_id, run_id, strength_unit, instrument_id, model_id, item_set_id FROM run WHERE eval_id = 'gsm8k'").fetchall()}
    assert set(runs) == {"none", F.SHA_A}
    assert runs[F.SHA_A][2] == "nominal_only" and runs["none"][2] == "unverified"            # strength_unit landed in the DB
    assert all(r[4] == "gemma4e4b_it" and r[5] == "gsm8k_platinum_paired_200" for r in runs.values())
    run_ids = tuple(r[1] for r in runs.values())
    q = "SELECT COUNT(*) FROM {} WHERE run_id IN (?, ?)"
    assert conn.execute(q.format("run_metric"), run_ids).fetchone()[0] == 2
    assert conn.execute(q.format("run_contrast"), run_ids).fetchone()[0] == 1
    assert [tuple(r) for r in conn.execute("SELECT json_extract(extra_json, '$.anchor_selection.state'), anchor_kind, anchor_run_id FROM run_contrast")] == [("own_current", "same_standup_none", runs["none"][1])]
    assert [tuple(r) for r in conn.execute("SELECT json_type(rm.extra_json, '$.delta_block') FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.object_id <> 'none'")] == [("null",)]
    # item-scale rows carry the runner's revision stamp and are the current rows (the first pass's rows are superseded by the head)
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (res.revision_id,)).fetchone()[0] == 2 * fx["n"]
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current WHERE source_revision_id = ?", (res.revision_id,)).fetchone()[0] == 2 * fx["n"]
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current WHERE run_id IN (?, ?)", run_ids).fetchone()[0] == 2 * fx["n"]
    assert conn.execute("SELECT COUNT(*) FROM run_degeneration WHERE source_revision_id = ?", (res.revision_id,)).fetchone()[0] == 2 * 3
    assert conn.execute("SELECT COUNT(*) FROM item_flags WHERE source_revision_id = ?", (res.revision_id,)).fetchone()[0] == 2 * fx["n"]
    # the emitter-minted dimension rows resolve the runs' FKs; the instrument id is identity.instrument_id over its own row
    inst = dict(conn.execute("SELECT * FROM instrument WHERE instrument_id = ?", (runs[F.SHA_A][3],)).fetchone())
    assert inst["harness"] == "inspect_ai" and inst["instrument_id"] == I.instrument_id(inst)
    algos = {r[0] for r in conn.execute("SELECT c.algorithm FROM run_metric rm JOIN computation c ON c.computation_id = rm.computation_id "
                                        "WHERE rm.run_id IN (?, ?)", run_ids).fetchall()}
    assert len(algos) == 1 and next(iter(algos)).startswith("paperlib.gsm8k_p200.")          # the relocated reader's computation resolves
    # membership is observed per table; the report's cells are stated
    obs = dict(conn.execute("SELECT table_name, COUNT(*) FROM observation WHERE source_id = ? AND current = 1 GROUP BY table_name",
                            (RUN.source_id_for(ad.name, res.group),)).fetchall())
    assert obs.get("run") == 2 and obs.get("run_metric") == 2 and obs.get("run_contrast") == 1
    states = conn.execute("SELECT json_path, current_members_json FROM source_state WHERE input_uri = ?", (fx["path"],)).fetchall()
    assert any(p == "$.cells" and len(json.loads(m)) == 2 for p, m in states)
    conn.close()
    # an unchanged third pass writes nothing (the decision set replays to the recorded fingerprint)
    rep2 = _pass(root, ad, LATER)
    assert rep2.inputs_changed == 0 and rep2.groups_refreshed == 0 and rep2.records_written == 0 and rep2.anchor_decisions_changed == 0


# --------------------------------------------------------------------------------------------
# the cap-4096 lane-file declarations
# --------------------------------------------------------------------------------------------

def _lane(run_state: str, cells_by_tag) -> str:
    models = {tag: {"cells": {cid: {att: {"root": root, "k": 1, "n": 2, "tier": "multi_turn"} for att, root in atts.items()}
                              for cid, atts in cells.items()}} for tag, cells in cells_by_tag.items()}
    os.makedirs(run_state, exist_ok=True)
    p = os.path.join(run_state, C4.TRANSFER_STATS_FILE_NAME)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"synthetic": True}, "models": models}, fh)
    return p


def _provenance(run_state: str, root: str, tag: str) -> str:
    p = os.path.join(run_state, root, tag, "grading_provenance.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"synthetic": True, "root": root}, fh)
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _ruling_row(conn, rid):
    return dict(conn.execute("SELECT evidence_state, expired_utc FROM ruling WHERE ruling_id = ?", (rid,)).fetchone())


def _gsm8k_root_world(tmp_path, extra_models=()):
    """A gsm8k_p200 ROOT-mode world: the registry (the declared dims plus the models named) and an empty root to build directories under."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    dims = _declared_dims()
    gemma = next(r for r in dims if r["record_type"] == "model" and r["payload"]["model_id"] == "gemma4e4b_it")
    for i, (mid, family, size, repo) in enumerate(extra_models):
        dims.append(synth.rec("model", dict(gemma["payload"], model_id=mid, family=family, size_label=size, hf_repo=repo), synth.t(24 + i)))
    dims.append(synth.rec("paper_round", {"round_id": "r2", "opened_utc": synth.t(28), "description": "synthetic round r2 (the dispositions' round)"}, synth.t(28)))
    _write(root, "dims", "000001-dims.jsonl", dims)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    g = str(tmp_path / "g8k")
    os.makedirs(g)
    return root, g


def _gsm8k_states(root, source_id):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return {(r[0], r[1]): (r[2], json.loads(r[3]), r[4]) for r in conn.execute(
            "SELECT input_uri, json_path, found, current_members_json, observed_utc FROM source_state WHERE source_id = ?", (source_id,)).fetchall()}
    finally:
        conn.close()


def _gsm8k_fold(root, ad, now):
    rep = _pass(root, ad, now)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert rep.groups_refused == 0 and ing.refused == 0 and ing.blocked == 0, ([r.error for r in rep.results], ing.refusals)
    return rep


def test_gsm8k_p200_root_inventory_has_one_publisher_the_root_group_whose_listing_is_its_input(tmp_path):
    """The root inventory's publisher (`gsm8k_p200.ROOT_LISTING_RULE`). Two canonical directories A
    (gemma) and B (llama) are two groups; the ROOT pseudo-group `g8k` reads the root's listing as its one input and states `$.dirs` on it.
    Pass 1 parks, pass 2 files (A, B re-read; the root quiet: its listing did not move). A changes and commits again: the root is quiet. An
    undeclared directory C lands: ONLY the root re-reads (a listing change is its revision change) and `$.dirs` names A, B, C, keyed on the
    listing input with the new fingerprint. A's bytes restored: A's steady revision re-activates -- its own `$.cells` row replayed, the root
    untouched -- and no per-group read ever states the root inventory."""
    from sisteer.registry.adapters import gsm8k_p200 as G8
    root, g = _gsm8k_root_world(tmp_path, extra_models=[("llama8b_it", "llama31", "8b", "synthetic/llama-8b-it")])
    fa = F.build_gsm8k_report(os.path.join(g, "A"), model="gemma4e4b", view="it")
    F.build_gsm8k_report(os.path.join(g, "B"), model="llama8b", view="it")

    def adapter():
        return G8.Gsm8kP200Adapter(root=g, canonical=[("A", G8.STORE_PIN), ("B", G8.STORE_PIN)], superseded={}, not_production={})   # the plan is computed once per instance

    ad = adapter()
    rg = ad.root_group()
    assert rg == G8.ROOT_GROUP == "_root"
    sid_root, sid_a = RUN.source_id_for(ad.name, rg), RUN.source_id_for(ad.name, "A")
    (listing,) = [i for i in ad.discover() if AB.is_listing_input(i)]
    assert ad.comparison_group(listing) == rg and listing.uri == AB.listing_uri(sid_root, G8.ROOT_LISTING_LABEL) and listing.entries == ("A", "B")
    key = (listing.uri, "$.dirs")
    rep1 = _gsm8k_fold(root, ad, "2026-10-04T00:00:00.000Z")
    assert sorted(r.group for r in rep1.results) == ["A", "B", rg]
    root_res = next(r for r in rep1.results if r.group == rg)
    assert root_res.accepted and not root_res.emitted_nothing, "the root group is not a silent read"
    with DB.connect(DB.db_path(root), readonly=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM observation WHERE source_id = ?", (sid_root,)).fetchone()[0] == 0, "the root group emits no measurement record"
        assert conn.execute("SELECT ok FROM v_source_status_current WHERE source_id = ?", (sid_root,)).fetchone()[0] == 1
    rep2 = _gsm8k_fold(root, ad, "2026-10-04T00:30:00.000Z")
    assert sorted(r.group for r in rep2.results) == ["A", "B"] and rep2.anchor_decisions_changed == 2, "the fold moved A's and B's decisions; the root is quiet"
    rev_a2 = next(r.revision_id for r in rep2.results if r.group == "A")
    assert _gsm8k_states(root, sid_root) == {key: (1, ["A", "B"], "2026-10-04T00:00:00.000Z")}
    # the pre-rule root-path key under gsm8k_p200:<root basename> is stated absent by the root group's read
    assert _gsm8k_states(root, ad.legacy_root_source_id()) == {(g, "$.dirs"): (0, [], "2026-10-04T00:00:00.000Z")}
    cells_a2 = _gsm8k_states(root, sid_a)[(fa["path"], "$.cells")]
    assert cells_a2[0] == 1 and len(cells_a2[1]) == 2
    # A changes and commits again: the root is quiet
    F.build_gsm8k_report(os.path.join(g, "A"), model="gemma4e4b", view="it", k_cell=19)
    rep3 = _gsm8k_fold(root, ad, "2026-10-04T01:00:00.000Z")
    assert [r.group for r in rep3.results] == ["A"] and rep3.results[0].revision_id != rev_a2 and not rep3.results[0].reactivated
    # an undeclared directory C lands: the root's listing moved, so the root group alone re-reads and restates the inventory
    os.makedirs(os.path.join(g, "C"))
    ad = adapter()
    rep4 = _gsm8k_fold(root, ad, "2026-10-04T02:00:00.000Z")
    assert [r.group for r in rep4.results] == [rg] and not rep4.results[0].reactivated
    (listing4,) = [i for i in ad.discover() if AB.is_listing_input(i)]
    assert listing4.uri == listing.uri and listing4.sha256 != listing.sha256 and listing4.entries == ("A", "B", "C")
    assert _gsm8k_states(root, sid_root) == {key: (1, ["A", "B", "C"], "2026-10-04T02:00:00.000Z")}
    with DB.connect(DB.db_path(root), readonly=True) as conn:
        assert conn.execute("SELECT input_sha256 FROM source_state WHERE source_id = ? AND json_path = '$.dirs'", (sid_root,)).fetchone()[0] == listing4.sha256
        assert conn.execute("SELECT COUNT(*) FROM source_state WHERE json_path = '$.dirs' AND found = 1 AND source_id <> ?", (sid_root,)).fetchone()[0] == 0, "no per-group read states the root inventory"
    # A's pass-2 bytes again: A2 re-activates; the root inventory is untouched
    F.build_gsm8k_report(os.path.join(g, "A"), model="gemma4e4b", view="it")
    rep5 = _gsm8k_fold(root, ad, "2026-10-04T03:00:00.000Z")
    (res5,) = rep5.results
    assert res5.group == "A" and res5.reactivated and res5.revision_id == rev_a2, res5
    assert _gsm8k_states(root, sid_root) == {key: (1, ["A", "B", "C"], "2026-10-04T02:00:00.000Z")}
    assert _gsm8k_states(root, sid_a)[(fa["path"], "$.cells")] == (1, cells_a2[1], "2026-10-04T03:00:00.000Z"), "A's own row replayed from its committed revision"
    recs = []
    for f in res5.files:
        if "source_revision" in os.path.basename(f):
            recs += [rec["payload"] for _, rec, _ in R.read_record_file(f) if rec["record_type"] == "source_state"]
    assert {(p["source_id"], p["json_path"]) for p in recs} == {(sid_a, "$.cells")}, recs
    assert _gsm8k_fold(root, ad, "2026-10-04T04:00:00.000Z").results == []


def test_gsm8k_p200_a_retired_optional_group_leaves_the_root_inventory_restated_without_it_while_every_other_group_is_quiet(tmp_path):
    """The canonical group A and an optional check-pass directory `chk` (declared not_production: its
    own group, dispositions only). Both fold; the root group states `$.dirs` = A, chk. The check-pass directory is deleted while A's files are
    unchanged: `chk`'s inputs vanished (retired: an empty revision, its rows cleared), the root's listing changed, so the ROOT group re-reads and
    restates `$.dirs` = A -- found = 1, naming only existing directories -- and A is quiet. Before the rule every group's read wrote the root-wide
    row, the retirement cleared only chk's own rows, and `$.dirs` kept naming the deleted directory with no read left to repair it."""
    from sisteer.registry.adapters import gsm8k_p200 as G8
    root, g = _gsm8k_root_world(tmp_path)
    F.build_gsm8k_report(os.path.join(g, "A"), model="gemma4e4b", view="it")
    F.build_gsm8k_report(os.path.join(g, "chk"), model="gemma4e4b", view="it", k_cell=16)

    def adapter():
        return G8.Gsm8kP200Adapter(root=g, canonical=[("A", G8.STORE_PIN)], superseded={}, not_production={"chk": "a check pass, not production"})

    ad = adapter()
    rg = ad.root_group()
    sid_root, sid_chk, sid_a = (RUN.source_id_for(ad.name, x) for x in (rg, "chk", "A"))
    (listing,) = [i for i in ad.discover() if AB.is_listing_input(i)]
    key = (listing.uri, "$.dirs")
    rep1 = _gsm8k_fold(root, ad, "2026-10-04T00:00:00.000Z")
    assert sorted(r.group for r in rep1.results) == sorted(["A", "chk", rg]) and all(r.accepted for r in rep1.results), [(r.group, r.error) for r in rep1.results]
    _gsm8k_fold(root, ad, "2026-10-04T00:30:00.000Z")                 # the fold moved A's decision; chk files dispositions and is quiet
    assert _gsm8k_fold(root, ad, "2026-10-04T01:00:00.000Z").results == []
    assert _gsm8k_states(root, sid_root) == {key: (1, ["A", "chk"], "2026-10-04T00:00:00.000Z")}
    chk_rows = _gsm8k_states(root, sid_chk)
    assert chk_rows and all(v[0] == 1 for v in chk_rows.values())
    head_a = _q_head(root, sid_a)
    # the check-pass directory is deleted; A's files stand
    shutil.rmtree(os.path.join(g, "chk"))
    ad = adapter()
    rep = _gsm8k_fold(root, ad, "2026-10-04T02:00:00.000Z")
    by_group = {r.group: r for r in rep.results}
    assert set(by_group) == {rg, "chk"}, [(r.group, r.error) for r in rep.results]
    assert by_group["chk"].accepted and by_group["chk"].withdrawals > 0 and by_group[rg].accepted and not by_group[rg].reactivated
    assert _gsm8k_states(root, sid_root) == {key: (1, ["A"], "2026-10-04T02:00:00.000Z")}, "the root inventory restated without the deleted directory"
    assert all(v[0] == 0 for v in _gsm8k_states(root, sid_chk).values()), "the retired group's own rows cleared"
    assert _q_head(root, sid_a) == head_a, "A is quiet"
    assert _gsm8k_fold(root, ad, "2026-10-04T03:00:00.000Z").results == []


def test_gsm8k_p200_explicit_list_mode_keeps_the_root_inventorys_publisher_and_retires_the_legacy_root_path_row(tmp_path):
    """The explicit-list constructor (`report=`, the paper's read path) used to state the root-path-keyed `$.dirs`
    from every group's read; a database read under that version holds that row. An explicit-only pass now carries the root pseudo-group too
    (no other directory is classified or read): it states `$.dirs` on the root listing input and the legacy row ABSENT (found = 0), and a
    directory deleted beside the report re-reads the root group alone -- the inventory restated without it -- while the report's group is quiet.
    The report's own records are the same bytes the version-3 explicit read emitted."""
    from sisteer.registry.adapters import gsm8k_p200 as G8
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", _declared_dims())
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    g = str(tmp_path / "g8k")
    fx = F.build_gsm8k_report(os.path.join(g, "canon"))
    os.makedirs(os.path.join(g, "aside"))                                  # a directory the explicit list never names: listed, not read
    ad = G8.Gsm8kP200Adapter(report=fx["path"], later_reports=(), expect_pin=G8.STORE_PIN)
    assert ad.explicit and ad.root == g and ad.legacy_root_source_id() == f"gsm8k_p200:{os.path.basename(g)}"
    sid_root, sid_canon = RUN.source_id_for(ad.name, G8.ROOT_GROUP), RUN.source_id_for(ad.name, "canon")
    # the legacy row a version-3 read left: the root inventory keyed on the root path, naming a directory that is gone
    legacy = {"source_id": ad.legacy_root_source_id(), "input_uri": g, "json_path": "$.dirs", "found": 1, "path_resolves": 1,
              "current_members_json": ["canon", "deleted_long_ago"], "observed_utc": "2026-09-21T00:00:00.000Z", "input_sha256": None}
    _write(root, "legacy", "000001-legacy-state.jsonl", [synth.rec("source_state", legacy, "2026-09-21T00:00:00.000Z")])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _gsm8k_states(root, ad.legacy_root_source_id()) == {(g, "$.dirs"): (1, ["canon", "deleted_long_ago"], "2026-09-21T00:00:00.000Z")}
    (listing,) = [i for i in ad.discover() if AB.is_listing_input(i)]
    assert [os.path.relpath(i.uri, g) for i in ad.discover() if not AB.is_listing_input(i)] == ["canon/report.json"], "explicit mode classifies nothing else"
    assert listing.entries == ("aside", "canon")
    rep1 = _gsm8k_fold(root, ad, "2026-10-04T00:00:00.000Z")
    assert sorted(r.group for r in rep1.results) == sorted(["canon", G8.ROOT_GROUP])
    assert _gsm8k_states(root, sid_root) == {(listing.uri, "$.dirs"): (1, ["aside", "canon"], "2026-10-04T00:00:00.000Z")}
    assert _gsm8k_states(root, ad.legacy_root_source_id()) == {(g, "$.dirs"): (0, [], "2026-10-04T00:00:00.000Z")}, "the legacy root-path row is retired"
    _gsm8k_fold(root, ad, "2026-10-04T00:30:00.000Z")                     # the fold moved canon's decision; the root is quiet
    assert _gsm8k_fold(root, ad, "2026-10-04T01:00:00.000Z").results == []
    head_canon = _q_head(root, sid_canon)
    # a directory beside the report is deleted: only the root group re-reads
    os.rmdir(os.path.join(g, "aside"))
    ad = G8.Gsm8kP200Adapter(report=fx["path"], later_reports=(), expect_pin=G8.STORE_PIN)
    rep = _gsm8k_fold(root, ad, "2026-10-04T02:00:00.000Z")
    assert [r.group for r in rep.results] == [G8.ROOT_GROUP] and not rep.results[0].reactivated
    assert _gsm8k_states(root, sid_root) == {(listing.uri, "$.dirs"): (1, ["canon"], "2026-10-04T02:00:00.000Z")}
    assert _q_head(root, sid_canon) == head_canon
    assert _gsm8k_fold(root, ad, "2026-10-04T03:00:00.000Z").results == []


def _q_head(root, source_id):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", (source_id,)).fetchone()[0]
    finally:
        conn.close()


def test_cap4096_declarations_flow_to_source_state_and_the_ruling_expires(tmp_path):
    home = str(tmp_path / "home")
    run_state = os.path.join(home, "run-state", "full22")
    _lane(run_state, {"llama8b_it": {"arditi": {"pair": "grades_new", "crescendo": "grades_new"}, "alphasteer": {"pair": "grades_x"}}})
    prov_sha = _provenance(run_state, "grades_old", "llama8b_it")
    lane_uri = "~/run-state/full22/" + C4.TRANSFER_STATS_FILE_NAME
    guard_uri = "~/run-state/full22/grades_old/llama8b_it/grading_provenance.json"
    # the round's ruling shape (rulings_r2_v1.jsonl r2v1-supersede-llama-arditi-r2-multiturn-r0p1), on synthetic roots
    ew = {"kind": "any_of",
          "conditions": [{"kind": "grades_root_absent", "input_uri": lane_uri, "json_path": "models.llama8b_it.cells.arditi", "root": "grades_old"}],
          "guards": [{"kind": "input_sha_pinned", "artifact_uri": guard_uri, "sha256_prefix": prov_sha[:16]}]}
    ruling = synth.rec("ruling", {"ruling_id": "R_mt", "round_id": "r_synth", "action": "supersede", "scope_kind": "key",
                                  "scope_json": {"model_id": "llama8b_it", "paper_label": "arditi"}, "reason": "re-graded under a new grade root",
                                  "ruled_utc": synth.t(60), "ruled_by": "user", "source_ref": "test", "expires_when_json": ew}, synth.t(60))
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _write(root, "roster", "000001-roster.jsonl", [ruling])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn, "R_mt") == {"evidence_state": "missing", "expired_utc": None}      # no evidence yet: the exclusion stays active
    conn.close()

    ad = C4.Cap4096LaneDeclarationsAdapter(run_state=run_state, home=home)
    inputs = ad.discover()
    assert [i.uri for i in inputs] == [lane_uri, guard_uri]                                     # the rulings' spelling is the adapter's uri
    rep = _pass(root, ad, NOW)
    assert rep.groups_refused == 0 and rep.groups_refreshed == 1, [r.error for r in rep.results]
    # the inputs' artifact records plus the read's ONE source_status row (a declarations-only adapter files its status
    # under its own name); no measurements
    assert rep.records_written == len(inputs) + 1
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.revisions_held == 0, ing
    conn = DB.connect(DB.db_path(root), readonly=True)
    st = {(r[0], r[1]): r for r in conn.execute("SELECT input_uri, json_path, found, path_resolves, current_members_json, input_sha256 "
                                                 "FROM source_state WHERE source_id = ?", (ad.source_id,)).fetchall()}
    assert st[(lane_uri, "models.llama8b_it.cells.arditi")][2:5] == (1, 1, '["grades_new"]')
    assert st[(lane_uri, "models.llama8b_it.cells.alphasteer")][2:5] == (1, 1, '["grades_x"]')
    assert st[(guard_uri, "$")][2:4] == (1, 1) and st[(guard_uri, "$")][5] == prov_sha
    assert conn.execute("SELECT COUNT(*) FROM artifact WHERE uri_primary = ?", (lane_uri,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM source_watermark WHERE input_uri = ? AND last_error IS NULL", (lane_uri,)).fetchone()[0] == 1
    # the harvester re-evaluated the ruling against the adapter's evidence: root set {grades_new} != {grades_old} -> ok, EXPIRED
    row = _ruling_row(conn, "R_mt")
    assert row["evidence_state"] == "ok" and row["expired_utc"] is not None
    conn.close()

    # the lane file moves: the arditi node disappears and alphasteer's roots change -> the ruled path is stated ABSENT (never left stale),
    # the previously stated alphasteer path is re-stated with its new members, and the ruling reads path_absent (exclusion active again)
    _lane(run_state, {"llama8b_it": {"alphasteer": {"pair": "grades_y"}}})
    rep2 = _pass(root, ad, LATER)
    assert rep2.inputs_changed == 1 and rep2.groups_refreshed == 1, [r.error for r in rep2.results]
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    st2 = {r[0]: r for r in conn.execute("SELECT json_path, found, path_resolves, current_members_json FROM source_state WHERE source_id = ? AND input_uri = ?",
                                          (ad.source_id, lane_uri)).fetchall()}
    assert st2["models.llama8b_it.cells.arditi"][1:4] == (1, 0, "[]")
    assert st2["models.llama8b_it.cells.alphasteer"][1:4] == (1, 1, '["grades_y"]')
    assert _ruling_row(conn, "R_mt")["evidence_state"] == "path_absent"
    conn.close()


def test_cap4096_restates_a_prior_path_the_file_dropped_and_a_missing_pinned_guard(tmp_path):
    """A path this source stated in an earlier revision (here a gemma cell) that the new file lacks is re-stated `path_resolves = 0`
    through the runner's read-only connection; a pinned provenance file that vanished is stated `found = 0`."""
    home = str(tmp_path / "home")
    run_state = os.path.join(home, "run-state", "full22")
    _lane(run_state, {"llama8b_it": {"arditi": {"pair": "grades_a"}}, "gemma4e4b_it": {"alphasteer": {"pair": "grades_g"}}})
    guard_rel = C4.RULED_GUARDS[0][0]
    _provenance(run_state, guard_rel.split(os.sep)[0], guard_rel.split(os.sep)[1])
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    H.ingest_once(root, workers=0, min_free=0)
    ad = C4.Cap4096LaneDeclarationsAdapter(run_state=run_state, home=home)
    assert _pass(root, ad, NOW).groups_refreshed == 1
    H.ingest_once(root, workers=0, min_free=0)
    lane_uri = "~/run-state/full22/" + C4.TRANSFER_STATS_FILE_NAME
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT path_resolves FROM source_state WHERE source_id = ? AND input_uri = ? AND json_path = 'models.gemma4e4b_it.cells.alphasteer'",
                        (ad.source_id, lane_uri)).fetchone()[0] == 1
    guard_uri = C4.ruling_uri(os.path.join(run_state, guard_rel), home)
    assert conn.execute("SELECT found FROM source_state WHERE source_id = ? AND input_uri = ?", (ad.source_id, guard_uri)).fetchone()[0] == 1
    conn.close()
    # the gemma cell leaves the file and the pinned provenance file is deleted
    _lane(run_state, {"llama8b_it": {"arditi": {"pair": "grades_a"}}})
    os.remove(os.path.join(run_state, guard_rel))
    rep = _pass(root, ad, LATER)
    assert rep.groups_refreshed == 1, [r.error for r in rep.results]
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT found, path_resolves, current_members_json FROM source_state WHERE source_id = ? AND input_uri = ? AND json_path = 'models.gemma4e4b_it.cells.alphasteer'",
                              (ad.source_id, lane_uri)).fetchone()) == (1, 0, "[]")
    assert tuple(conn.execute("SELECT found, path_resolves, input_sha256 FROM source_state WHERE source_id = ? AND input_uri = ?", (ad.source_id, guard_uri)).fetchone()) == (0, 0, None)
    conn.close()


def test_cap4096_refuses_a_lookalike_and_a_missing_lane_file(tmp_path):
    from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal
    home = str(tmp_path / "home")
    run_state = os.path.join(home, "run-state", "full22")
    ad = C4.Cap4096LaneDeclarationsAdapter(run_state=run_state, home=home)
    with pytest.raises(AdapterRefusal, match="input missing"):
        ad.discover()
    os.makedirs(run_state)
    with open(os.path.join(run_state, C4.TRANSFER_STATS_FILE_NAME), "w", encoding="utf-8") as fh:
        json.dump({"rows": []}, fh)
    inputs = ad.discover()
    ctx = AdapterContext(root="", now=NOW, producer={"script": "test"})
    with pytest.raises(AdapterRefusal, match="no `models` block"):
        ad.source_states(ctx, inputs)
    assert C4.ruling_uri("/elsewhere/x.json", home) == "/elsewhere/x.json" and C4.ruling_uri(os.path.join(home, "a", "b.json"), home) == "~/a/b.json"
    assert C4.root_members({"pair": {"root": "b"}, "note": "x", "crescendo": {"root": "a"}, "actorattack": {"k": 1}}) == ["a", "b"]


#: the round's rulings seed as committed at configs/registry/; read relative to this file, never through an
#: out-of-tree path, so the test runs on every checkout
RULINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "registry", "rulings_r2_v1.jsonl")


#: the shipped rulings spell the run-state root with the state-directory placeholder
STATE_PREFIX = "$STATE/run-state/full22/"


def test_ruled_paths_are_exactly_the_rulings_files_paths():
    """The adapter's declared paths equal the lane-file conditions and run-state provenance guards the round's rulings name
    (`source_state` for EXACTLY the JSON paths the rulings reference). The rulings file is the committed seed, so the test always runs:
    a missing file is a failure, never a skip."""
    assert os.path.isfile(RULINGS_FILE), f"the round's rulings seed is not committed at {RULINGS_FILE}"
    conditions, guards = set(), set()
    run_state_prefix = C4.ruling_uri(LC.RS_DEFAULT) + "/"

    def walk(spec, rid):
        if not isinstance(spec, dict):
            return
        if spec.get("kind") in ("any_of", "all_of"):
            for c in (spec.get("conditions") or []) + (spec.get("guards") or []):
                walk(c, rid)
            return
        uri = str(spec.get("input_uri") or spec.get("artifact_uri") or spec.get("path") or "")
        if spec.get("kind") == "grades_root_absent" and os.path.basename(uri) == C4.TRANSFER_STATS_FILE_NAME:
            conditions.add((str(spec.get("json_path")), rid))
        prefix = next((p for p in (run_state_prefix, STATE_PREFIX) if uri.startswith(p)), None)
        if spec.get("kind") == "input_sha_pinned" and prefix and uri.endswith("grading_provenance.json"):
            guards.add((uri[len(prefix):], rid))
    with open(RULINGS_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("_header"):
                continue
            p = d.get("payload", d)
            walk(p.get("expires_when_json"), str(p.get("ruling_id")))
    assert conditions == set(C4.RULED_PATHS)
    assert guards == set(C4.RULED_GUARDS)
    assert all(fnmatch.fnmatch(rel, C4.PROVENANCE_GLOB) for rel, _ in guards)              # every pinned file is one discovery finds
