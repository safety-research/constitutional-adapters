"""The recurring incremental adapter pass on a SYNTHETIC adapter: a changed
legacy store becomes new rows and advances `v_paper_inputs_head`; unchanged inputs write nothing; a broken
input is visible (`last_error`, unaccepted revision) and the previous revision stays current; a new store
under a declared root is discovered; a removed measurement is withdrawn (its rows leave `v_paper_long`,
history keeps them); removed items get `withdrawn_by_revision_id`; the input rules (non-regular / symlink /
sha mismatch refusals).
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Sequence
from unittest import mock

import pytest

from sisteer.registry import adapters as ADP
from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import dimensions as DIM
from sisteer.registry.adapters import runner as RUN


class SyntheticStoreAdapter:
    """Reads `<root>/*/store.json` files: {"key": ..., "object": ..., "items": [{"id":..,"score":..}]} → a judged
    run with metric + items. One comparison group per store (the group boundary is the store)."""
    name = "synthetic_store"
    covers_sources = ("synthetic",)
    rng_scope = "none"
    allow_symlinks = False

    def __init__(self, store_root: str):
        self.roots = (store_root,)
        self.broken: set = set()

    def discover(self) -> List[AB.Input]:
        out = []
        for d in sorted(os.listdir(self.roots[0])):
            p = os.path.join(self.roots[0], d, "store.json")
            if os.path.lexists(p):
                out.append(AB.check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def comparison_group(self, inp: AB.Input) -> str:
        return os.path.basename(os.path.dirname(inp.uri))

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for inp in inputs:
            if inp.uri in self.broken:
                raise AB.AdapterRefusal(inp.uri, "synthetic read failure")
            with open(inp.path) as fh:
                store = json.load(fh)
            batch = AB.RecordBatch(RUN.source_id_for(self.name, group), group)
            key = store["key"]
            rid = I.run_id(synth.CAMPAIGN, key, store["object"], synth.EVAL_ID, synth.INSTRUMENT_ID)
            obj_id = "none" if store["object"] == "none" else synth.OBJECT_SHA
            # a backfilled run carries the state its store proves (generated) and its later states as run_state events
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


def _roster(root):
    R.write_records(os.path.join(root, "records", "roster", "000001-roster.jsonl"), [
        synth.rec("roster_entry", {"roster_entry_id": "re1", "round_id": "r_synth", "comparison_id": "fig", "model_id": synth.MODEL_ID,
                                   "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID,
                                   "object_selector_json": {"paper_label": "constitution_lora"}, "strength_rule": "operating_point",
                                   "source_citation": "synthetic", "inferred": 0}, synth.t(50))])


def _setup(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    _roster(root)
    H.ingest_once(root, workers=0, min_free=0)
    store_root = str(tmp_path / "stores")
    os.makedirs(store_root)
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    return root, store_root


def _pass_and_ingest(root, adapter, now, **kw):
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [adapter], conn=conn, now=now, **kw)
    conn.close()
    ing = H.ingest_once(root, workers=0, min_free=0)
    return rep, ing


def _cli(root, ad, monkeypatch, capsys, *flags):
    """``registry adapters run`` over ONE registered adapter with the fold mocked (``H.ingest_under_lock`` answers fixed counts and
    records whether it was called): ``(rc, summary, last line, stderr, the fold mock)``. The ingest line is the LAST line, one
    JSON object, whatever the pass did."""
    monkeypatch.setattr(ADP, "load_all", lambda: {ad.name: lambda: ad})
    monkeypatch.setattr(DIM, "adapter_legacy_spellings", lambda seeds: {})
    lock = mock.MagicMock(name="ingest_under_lock", return_value=type("Ing", (), {"ok": 7, "refused": 1, "blocked": 2})())
    monkeypatch.setattr(H, "ingest_under_lock", lock)
    args = cli.build_parser().parse_args(["--root", root, "adapters", "run", "--workers", "0", *flags])
    rc = args.fn(args)
    out = capsys.readouterr()
    summary, _ = json.JSONDecoder().raw_decode(out.out)
    return rc, summary, json.loads(out.out.strip().splitlines()[-1]), out.err, lock


def _head(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0]
    finally:
        conn.close()


def test_legacy_store_update_flows_to_paper_head(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep.groups_refreshed == 1 and ing.refused == 0
    head0 = _head(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1").fetchone()[0] == 1
    assert conn.execute("SELECT value FROM run_metric").fetchone()[0] == pytest.approx((0.2 + 0.8 + 0.4) / 3)
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id <> ''").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM observation WHERE current = 1").fetchone()[0] == 3  # run, judgment, run_metric
    conn.close()
    # the judge appends items: the store's sha changes → the group is refreshed, superseding rows apply, head advances
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9, 0.1])
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep2.inputs_changed == 1 and rep2.groups_refreshed == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT n FROM run_metric").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0] == 5      # the head revision's rows
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 8               # the superseded revision's rows coexist
    assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE accepted = 1").fetchone()[0] == 2
    chain = conn.execute("SELECT revision_id, superseded_by_revision_id FROM source_revision ORDER BY observed_utc").fetchall()
    assert chain[0][1] == chain[1][0] and chain[1][1] is None
    conn.close()
    assert _head(root) > head0


def test_unchanged_inputs_write_nothing(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass_and_ingest(root, ad, synth.t(2000))
    head = _head(root)
    rep, ing = _pass_and_ingest(root, ad, synth.t(2500))
    assert rep.inputs_changed == 0 and rep.groups_refreshed == 0 and rep.records_written == 0
    assert ing.ok == 0 and not ing.changed
    assert _head(root) == head


def test_broken_input_visible_and_previous_revision_stays_current(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass_and_ingest(root, ad, synth.t(2000))
    p = _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.5, 0.5])
    ad.broken.add(p)
    rep, ing = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep.groups_refused == 1 and rep.results[0].error and "synthetic read failure" in rep.results[0].error
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT last_error FROM source_watermark WHERE input_uri = ?", (p,)).fetchone()[0] is not None
    revs = conn.execute("SELECT accepted, refusal FROM source_revision ORDER BY observed_utc").fetchall()
    assert [r[0] for r in revs] == [1, 0] and "synthetic read failure" in revs[1][1]
    assert conn.execute("SELECT n FROM run_metric").fetchone()[0] == 3   # the previous revision's rows stand
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1").fetchone()[0] == 1
    conn.close()
    from sisteer.registry import status
    ctx = status.status_json(root)
    assert any(row.get("last_error") for row in ctx["adapter_pass"])


def test_new_store_under_declared_root_discovered(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass_and_ingest(root, ad, synth.t(2000))
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.1])
    rep, ing = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep.inputs_seen == 2 and rep.inputs_changed == 1 and rep.groups_refreshed == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 2
    conn.close()


def test_removed_measurement_is_withdrawn(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass_and_ingest(root, ad, synth.t(2000))
    conn = DB.connect(DB.db_path(root), readonly=True)
    rid = conn.execute("SELECT run_id FROM run").fetchone()[0]
    conn.close()
    # the store is replaced by another object's store under the SAME group: the old run's keys vanish
    _store(store_root, "s1", "legacy_store_1b", "none", [0.3])
    rep, ing = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep.withdrawals >= 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    wd = [tuple(r) for r in conn.execute("SELECT reason, table_name FROM withdrawal").fetchall()]
    assert ("removed_measurement", "run") in wd
    assert conn.execute("SELECT current FROM observation WHERE observation_key = ? AND table_name = 'run'",
                        (I.canonical_json({"run_id": rid}),)).fetchone()[0] == 0
    sel = conn.execute("SELECT selection_state FROM v_paper_current WHERE run_id = ?", (rid,)).fetchone()[0]
    assert sel == "withdrawn"
    assert conn.execute("SELECT COUNT(*) FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == 1   # history keeps the row
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE run_id = ? AND in_current_paper = 1", (rid,)).fetchone()[0] == 0
    conn.close()


def test_removed_items_shrink_the_run(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass_and_ingest(root, ad, synth.t(2000))
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2])   # two items removed
    rep, ing = _pass_and_ingest(root, ad, synth.t(3000))
    conn = DB.connect(DB.db_path(root), readonly=True)
    wd = conn.execute("SELECT reason, item_keys_json, extra_json FROM withdrawal WHERE reason = 'removed_items'").fetchone()
    assert wd is not None and json.loads(wd[1]) == [] and json.loads(wd[2])["n_items_removed"] == 2   # the count travels, the keys are the rows
    live = conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0]
    gone = conn.execute("SELECT COUNT(*) FROM item_result WHERE withdrawn_by_revision_id IS NOT NULL").fetchone()[0]
    assert (live, gone) == (1, 2)                                                            # the withdrawal marks the two removed keys
    assert conn.execute("SELECT COUNT(*) FROM v_item_headline").fetchone()[0] == 1
    assert conn.execute("SELECT n_items_judged FROM run").fetchone()[0] == 1
    conn.close()


def test_d12_input_rules(tmp_path):
    store_root = str(tmp_path / "stores")
    os.makedirs(store_root)
    p = _store(store_root, "s1", "k", "none", [0.1])
    with pytest.raises(AB.AdapterRefusal):
        AB.check_input(p + ".missing")
    with pytest.raises(AB.InputRefusal):
        AB.check_input(os.path.dirname(p))                      # a directory is not a regular file
    link = os.path.join(store_root, "link.json")
    os.symlink(p, link)
    with pytest.raises(AB.InputRefusal):
        AB.check_input(link)                                    # symlinks refused by default
    inp = AB.check_input(link, allow_symlink=True)
    assert inp.is_symlink and inp.path == os.path.realpath(p) and inp.target_device is not None
    assert inp.artifact_payload()["extra"]["target_sha256"] == inp.sha256
    with pytest.raises(AB.InputRefusal) as ei:
        AB.check_input(p, expected_sha256="0000")
    assert "no longer matches" in str(ei.value)
    assert AB.check_input(p, expected_sha256=inp.sha256[:8]).sha256 == inp.sha256
    # a tmpfs target is refused outright (when the box has a tmpfs mount)
    tmpfs = "/dev/shm"
    if os.path.isdir(tmpfs) and AB.mount_fstype(tmpfs) == "tmpfs":
        vp = os.path.join(tmpfs, f"registry-test-{os.getpid()}.json")
        with open(vp, "w") as fh:
            fh.write("{}")
        try:
            with pytest.raises(AB.InputRefusal) as ei:
                AB.check_input(vp)
            assert "tmpfs" in str(ei.value)
        finally:
            os.remove(vp)


# ============================================================================================
# free-form payload fields are made JSON-safe by the runner; a NaN in a TYPED column still refuses by name
# ============================================================================================


class DispositionAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter plus what the declared adapters file: a ``disposition`` per unmeasured row whose ``extra``
    carries the reader's legacy labels straight off a frame. The 2026-09-17 live hold: a pandas NaN standing for the absent
    cell id of an unkeyed arm's row reached ``extra.campaign_cell_id_legacy`` and the validator held the WHOLE revision on it.
    ``nan_value`` puts a NaN in a TYPED column instead (``run_metric.value``): a measurement that does not exist."""
    name = "disposition_store"

    def __init__(self, store_root: str, disposition_extra: dict, nan_value: bool = False):
        super().__init__(store_root)
        self.disposition_extra = disposition_extra
        self.nan_value = nan_value

    def read(self, ctx, group, inputs):
        for batch in super().read(ctx, group, inputs):
            if self.nan_value:
                for rec in batch.records:
                    if rec["record_type"] == "run_metric":
                        rec["payload"]["value"] = float("nan")
            payload = {"disposition_id": f"disp-{group}", "round_id": "r_synth", "roster_entry_id": None, "comparison_id": "fig",
                       "model_id": synth.MODEL_ID, "object_id": None, "eval_id": synth.EVAL_ID, "state": "pending",
                       "reason": "pending: synthetic unmeasured row", "source_ref": inputs[0].uri, "ruled_utc": ctx.now,
                       "extra": dict(self.disposition_extra)}
            batch.add("disposition", payload, ctx.producer, ctx.now)
            yield batch


def _q1(root, sql, *args):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def test_free_form_sanitiser_scope_is_extra_extra_json_and_parked_keys_never_typed_columns():
    nan, inf = float("nan"), float("inf")
    p = RUN.sanitise_free_form("run_metric", {"run_id": "r", "value": nan, "ci_lo": inf, "k": 1,
                                             "extra": {"campaign_cell_id_legacy": nan, "stats": {"ratios": (0.1, -inf)}},
                                             "extra_json": {"steer_live_ratio": inf}, "steer_live_ratio_legacy": nan})
    assert p["value"] != p["value"] and p["ci_lo"] == inf and p["k"] == 1                      # typed columns untouched: refused by name
    assert p["extra"] == {"campaign_cell_id_legacy": "NaN", "stats": {"ratios": [0.1, "-Infinity"]}}
    assert p["extra_json"] == {"steer_live_ratio": "Infinity"}
    assert p["steer_live_ratio_legacy"] == "NaN"                                                   # an unknown key is parked in extra_json
    # a SERIALISED extra_json is parsed, sanitised and re-serialised only when it carries a non-finite literal
    assert json.loads(RUN.sanitise_free_form("run", {"run_id": "r", "extra_json": '{"a": NaN}'})["extra_json"]) == {"a": "NaN"}
    assert RUN.sanitise_free_form("run", {"run_id": "r", "extra_json": '{"a": 1}'})["extra_json"] == '{"a": 1}'
    assert RUN.sanitise_free_form("run", {"run_id": "r", "extra_json": "not json"})["extra_json"] == "not json"   # the validator's refusal
    # run_patch: fields.extra / fields.extra_json only; a typed patched field and `verify` are never touched
    rp = RUN.sanitise_free_form("run_patch", {"run_id": "r", "fields": {"walltime_s": nan, "extra": {"x": nan}}, "verify": {"serving_path": "lens"}})
    assert rp["fields"]["walltime_s"] != rp["fields"]["walltime_s"] and rp["fields"]["extra"] == {"x": "NaN"} and rp["verify"] == {"serving_path": "lens"}
    # observation / run_ingest carry no free-form field
    o = {"source_id": "s", "revision_id": "r", "table_name": "run", "observation_keys": [{"run_id": "x"}]}
    assert RUN.sanitise_free_form("observation", o) == o
    assert RUN.sanitise_free_form("run_ingest", {"run_id": "r", "ingested_utc": "t"}) == {"run_id": "r", "ingested_utc": "t"}
    # a payload with nothing free-form to touch comes back equal (typed columns and their NaN included)
    clean = {"run_id": "r", "value": 0.5, "extra": {"a": [1, 2]}}
    assert RUN.sanitise_free_form("run_metric", clean) == clean


def test_a_nan_in_a_disposition_extra_folds_as_the_string_nan_while_a_nan_metric_value_still_refuses_by_name(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = DispositionAdapter(store_root, {"campaign_cell_id_legacy": float("nan"), "view_legacy": "synth_it", "ratio": float("inf")})
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0
    assert ing.revisions_held == 0 and ing.refused == 0, ing
    # the bytes on disk are JSON: no NaN / Infinity literal anywhere in the revision
    written = [rec for f in rep.results[0].files for _, rec, _ in R.read_record_file(f)]
    assert written and all(R.first_nonfinite(rec) is None for rec in written)
    ((extra_json, state),) = _q1(root, "SELECT extra_json, state FROM disposition")
    ex = json.loads(extra_json)
    assert ex["campaign_cell_id_legacy"] == "NaN" and ex["ratio"] == "Infinity" and ex["view_legacy"] == "synth_it" and state == "pending"
    assert _q1(root, "SELECT accepted FROM source_revision WHERE source_id = 'disposition_store:s1'") == [(1,)]
    assert _q1(root, "SELECT COUNT(*) FROM record_refusal") == [(0,)]
    assert _q1(root, "SELECT n FROM run_metric") == [(3,)]
    # a NaN in a TYPED column is a measurement that does not exist: refused by name, the revision held, nothing of it visible
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad.nan_value = True
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep2.groups_refreshed == 1 and ing2.revisions_held == 1 and ing2.refused >= 1, ing2
    rows = _q1(root, "SELECT reason FROM record_refusal")            # a held constituent's refusal spells the field at the head of its reason
    assert len(rows) == 1 and rows[0][0].startswith("value:") and "finite" in rows[0][0], rows
    assert _q1(root, "SELECT accepted FROM source_revision WHERE source_id = 'disposition_store:s2'") == [(0,)]
    assert _q1(root, "SELECT COUNT(*) FROM disposition") == [(1,)] and _q1(root, "SELECT COUNT(*) FROM run") == [(1,)]


def test_a_held_nan_extra_revision_is_left_alone_by_the_plain_pass_and_read_fresh_sanitised_by_reread(tmp_path, monkeypatch):
    """The live 2026-09-17 shape: a revision published BEFORE the sanitiser carries the NaN in its bytes and the harvester holds it.
    Its watermarks never folded, so every plain pass reads the group as changed — and, the same inputs naming the same published
    revision, LEAVES IT ALONE: a held revision is neither re-read nor re-emitted by the incremental pass (re-emitting its committed
    bytes as an activation bundle held it again, 59 MB a cycle, live). ``registry adapters run --reread`` is the way forward: the
    fresh read passes through the sanitiser and IS the bundle under the same revision id, the revision is accepted and
    current, and the held attempt's refused constituent stays refused by sha — visible, never folded."""
    root, store_root = _setup(tmp_path)
    ad = DispositionAdapter(store_root, {"campaign_cell_id_legacy": float("nan")})
    monkeypatch.setattr(RUN, "sanitise_free_form", lambda record_type, payload: payload)     # the pre-fix writer
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    monkeypatch.undo()
    rev = rep.results[0].revision_id
    assert rep.groups_refreshed == 1 and ing.revisions_held == 1 and ing.refused >= 1, ing
    assert _q1(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    assert [r[0].split(":")[0] for r in _q1(root, "SELECT reason FROM record_refusal")] == ["extra.campaign_cell_id_legacy"]
    assert _q1(root, "SELECT COUNT(*) FROM disposition") == [(0,)]
    # (a) the plain pass: the input still reads as changed (its watermarks are in the held marker) but the revision is NOT re-emitted —
    # neither refreshed nor refused, no bundle, no bytes; the result names the two ways forward
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep2.inputs_changed == 1 and rep2.groups_held == 1 and rep2.groups_refreshed == 0 and rep2.groups_refused == 0
    assert rep2.records_written == 0 and rep2.groups_reread == 0
    (res2,) = rep2.results
    assert res2.held and not res2.accepted and not res2.reread and res2.revision_id == rev and res2.files == []
    assert "not re-emitted" in res2.error and "`adapters run --reread`" in res2.error and "blocked-queue retry" in res2.error
    assert not os.path.isdir(RUN.activations_dir(root, ad.name, rev))
    assert ing2.ok == 0 and ing2.blocked == 0 and not ing2.changed, ing2
    assert _q1(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    assert _q1(root, "SELECT COUNT(*) FROM disposition") == [(0,)] and _q1(root, "SELECT COUNT(*) FROM run_metric") == [(0,)]
    assert _q1(root, "SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1") == [(0,)]
    # (b) `registry adapters run --reread`: a never-accepted directory is the one case where the fresh read IS the bundle. The read's
    # records pass through the sanitiser (the NaN in `extra` folds to the string "NaN" — this fixture puts the float straight into
    # `extra`; the declared emitter's `text_or_none` would have filed None upstream of the runner), published under the SAME
    # revision id as `_activations/<rev>/<stamp>/`: same inputs, same content identity, no second revision
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep3 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(4000), reread=True)
    conn.close()
    ing3 = H.ingest_once(root, workers=0, min_free=0)
    assert rep3.inputs_changed == 1 and rep3.groups_reread == 1 and rep3.groups_refreshed == 1
    assert rep3.groups_held == 0 and rep3.groups_refused == 0
    (res3,) = rep3.results
    assert res3.reread and res3.accepted and not res3.held and res3.revision_id == rev and res3.records_written > 0
    assert f"{os.sep}_activations{os.sep}{rev}{os.sep}" in res3.files[0]
    names = sorted(os.path.basename(f).split("-", 1)[1] for f in res3.files)
    assert {"disposition.jsonl", "run_metric.jsonl", "source_status.jsonl", "source_revision.jsonl"} <= set(names), names
    written = [rec for f in res3.files for _, rec, _ in R.read_record_file(f)]
    assert written and all(R.first_nonfinite(rec) is None for rec in written)              # the bundle's bytes are JSON
    disp = [rec for rec in written if rec["record_type"] == "disposition"]
    assert len(disp) == 1 and disp[0]["payload"]["extra"]["campaign_cell_id_legacy"] == "NaN"
    # the bundle applies (ok > 0, changed); the same cycle also retries the ORIGINAL held directory whole, whose NaN bytes refuse
    # again by sha — that retry is the harvester's, not the pass's, so its held/refused counts are not pinned here
    assert ing3.ok > 0 and ing3.blocked == 0 and ing3.changed, ing3
    # accepted and current under its own id: one revision row for the source, re-observed at the re-read, no second revision
    assert _q1(root, "SELECT accepted, refusal, superseded_by_revision_id, observed_utc FROM source_revision WHERE revision_id = ?", rev) \
        == [(1, None, None, synth.t(4000))]
    assert _q1(root, "SELECT COUNT(*) FROM source_revision WHERE source_id = 'disposition_store:s1'") == [(1,)]
    ((extra_json, state),) = _q1(root, "SELECT extra_json, state FROM disposition")
    assert json.loads(extra_json)["campaign_cell_id_legacy"] == "NaN" and state == "pending"
    assert _q1(root, "SELECT n FROM run_metric") == [(3,)]
    assert _q1(root, "SELECT COUNT(*) FROM v_item_result_current") == [(3,)]
    assert _q1(root, "SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1") == [(1,)]
    ((st_rev, st_ok, st_extra),) = _q1(root, "SELECT source_revision_id, ok, extra_json FROM v_source_status_current")
    rr = json.loads(st_extra)["reread"]
    assert (st_rev, st_ok) == (rev, 1) and rr["of_revision"] == rev and rr["head_revision"] is None    # no accepted head before it
    assert rr["fresh_read"] is True and rr["reactivated"] is False
    assert json.loads(st_extra)["record_counts"]["disposition"] == 1
    # the held attempt's refused constituent stays refused by sha (visible), filed against the ORIGINAL revision directory — the
    # fresh read's bundle refused nothing
    refusals = _q1(root, "SELECT reason, record_uri FROM record_refusal")
    assert [r[0].split(":")[0] for r in refusals] == ["extra.campaign_cell_id_legacy"]
    assert f"/{rev}/" in refusals[0][1] and "_activations" not in refusals[0][1], refusals
