"""The inputs-dropped guard of the legacy-store reader framework (adapters/runner.py): a read whose input set lacks more than the
threshold share (0.10 — every one of the six 2026-09-21 reads, 14.6-25.0 %, must refuse — or the adapter's declared
`inputs_dropped_threshold`) of the previous ACCEPTED revision's inputs, none of
them declared retired (`inputs_retired`: URIs or globs), is REFUSED before any store is read — an unaccepted marker whose
`refusal` starts `inputs_dropped_unexplained:` (count, share, up to 20 missing inputs; the whole comparison under
`extra_json.inputs_dropped_unexplained`), the previous revision current throughout, the inputs' watermarks carrying the refusal so
the group standing refused until an input changes or a pass carries `--retry-refused`, `v_source_status` saying `latest read
REFUSED`, the pass counting `groups_inputs_dropped`.

The live case (2026-09-21): every `relu_paired` source was read at 22:19:36Z with a store tree absent from discovery — `s1` saw 381
of the 501 inputs its 20:43Z head recorded — and published revisions carrying none of the paper's `difference_relu` × over-refusal
runs, withdrawing them for ~40 minutes, filed `ok = 1` with nothing named. The first test is that shape.

What must NOT fire: a first read (no accepted revision to compare with), a superset, a drop below the threshold, a drop the adapter
declared retired, a drop admitted by the adapter's own threshold. Every accepted marker records the comparison. Fixtures are the
synthetic store adapter of test_registry_core_fixes (one comparison group over many stores); nothing here reads a real store.
"""
from __future__ import annotations

import json
import logging
import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

from test_registry_core_fixes import StoreAdapter, _adapter_setup, _ingest, _q, _store

#: Reads stamped after the run-set drop guard landed (``H.RUN_SET_DROP_UNDECLARED_SINCE``): the guard is versioned by that stamp
#: and the test below that expects its hold needs a world it judges.
_SINCE_BASE = "2026-09-24T00:00:00.000Z"


def _t(i):
    return synth.t(i, base=_SINCE_BASE)


GROUP = "one_group"
SOURCE = RUN.source_id_for("synthetic_store", GROUP)
LOGGER = "sisteer.registry.adapters.runner"


def _stores(n, start=0, obj="none"):
    return [(f"s{i:03d}", f"store_{i:03d}", obj, [0.25 + (i % 3) * 0.25]) for i in range(start, start + n)]


def _setup(tmp_path, n):
    root, store_root = _adapter_setup(tmp_path, _stores(n))
    ad = StoreAdapter(store_root, group_of=lambda inp: GROUP)
    return root, store_root, ad


def _pass(root, ad, now, **kw):
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=now, **kw)
    return rep, _ingest(root)


def _head(root):
    rows = _q(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 AND superseded_by_revision_id IS NULL", SOURCE)
    assert len(rows) == 1, rows
    return rows[0][0]


def _paper_rows(root):
    return _q(root, "SELECT COUNT(*) FROM v_paper_long WHERE lane_source_legacy = 'synth_source'")[0][0]


def _extra(root, rev):
    return json.loads(_q(root, "SELECT extra_json FROM source_revision WHERE revision_id = ?", rev)[0][0])


def _remove(store_root, names):
    for name in names:
        os.remove(os.path.join(store_root, name, "store.json"))


def _published_extra(root, rev):
    """The comparison the runner wrote on the PUBLISHED marker record (on disk): what a held revision's placeholder row does not carry."""
    d = os.path.join(root, "records", "adapters", "synthetic_store", rev)
    for fn in sorted(os.listdir(d)):
        if "source_revision" in fn:
            for _, rec, _ in R.read_record_file(os.path.join(d, fn)):
                if rec.get("record_type") == "source_revision":
                    return dict(rec["payload"].get("extra") or {})
    raise AssertionError(f"no marker record under {d}")


# ============================================================================================
# the live shape: 501 accepted, 381 read → refused; back → accepted; declared → accepted
# ============================================================================================

def test_the_2026_09_21_shape_is_refused_the_head_stays_and_a_declared_retirement_publishes(tmp_path, caplog):
    root, store_root, ad = _setup(tmp_path, 501)
    rep1, ing1 = _pass(root, ad, _t(2000))
    assert rep1.groups_refreshed == 1 and ing1.refused == 0 and ing1.blocked == 0
    rev1 = _head(root)
    assert _paper_rows(root) == 501
    assert "inputs_previous" not in _extra(root, rev1)                       # a first read has nothing to compare with
    # 125 of the 501 stores leave discovery (a tree absent for one read): the read is refused before any store is opened
    gone = [f"s{i:03d}" for i in range(125)]
    _remove(store_root, gone)
    reads = []
    real_read = ad.read

    def counting_read(ctx, group, inputs):
        reads.append(len(inputs))
        return real_read(ctx, group, inputs)
    ad.read = counting_read
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        rep2, ing2 = _pass(root, ad, _t(3000))
    assert reads == []                                                          # no store was read
    assert rep2.groups_refused == 1 and rep2.groups_inputs_dropped == 1 and rep2.groups_refreshed == 0 and rep2.records_written == 0
    (res,) = rep2.results
    assert not res.accepted and res.inputs_dropped and res.error.startswith(f"{SOURCE}: {RUN.INPUTS_DROPPED_KEY}: 125 of the 501 inputs of the previous accepted revision {rev1}")
    assert "threshold 10%" in res.error and "(+105 more)" in res.error and "inputs_retired" in res.error
    rev2 = res.revision_id
    assert rev2 != rev1 and not os.path.isdir(os.path.join(root, "records", "adapters", "synthetic_store", rev2))   # nothing published
    assert ing2.refused == 0
    # the marker: unaccepted, the refusal by name, the whole comparison on the record; the head and the paper untouched
    assert _q(root, "SELECT accepted, substr(refusal, 1, 60) FROM source_revision WHERE revision_id = ?", rev2) == [(0, f"{SOURCE}: {RUN.INPUTS_DROPPED_KEY}: 125 of the 501 inputs of the"[:60])]
    cmp = _extra(root, rev2)[RUN.INPUTS_DROPPED_KEY]
    assert cmp["previous_revision_id"] == rev1 and cmp["inputs_previous"] == 501 and cmp["inputs_now"] == 376
    assert cmp["inputs_dropped"] == 125 and cmp["inputs_retired"] == 0 and cmp["inputs_added"] == 0 and cmp["tripped"] is True
    assert cmp["inputs_dropped_fraction"] == pytest.approx(125 / 501, abs=1e-4) and cmp["inputs_dropped_threshold"] == 0.1
    assert cmp["inputs_dropped_paths"] == [f"s{i:03d}/store.json" for i in range(20)] and cmp["rule"] == RUN.INPUTS_DROPPED_RULE
    assert _head(root) == rev1 and _paper_rows(root) == 501
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ?", SOURCE) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND current = 1 AND table_name = 'run_metric'", SOURCE) == [(501,)]
    # the present inputs' watermarks carry the refusal: the refusal STANDS (visible) and a quiet pass leaves it alone — the tree
    # coming back (an input change) or a pass carrying --retry-refused reads the group again; here the flag, refused again while absent
    assert _q(root, "SELECT COUNT(*) FROM source_watermark WHERE source_id = ? AND last_error LIKE ?", SOURCE, f"%{RUN.INPUTS_DROPPED_KEY}%") == [(376,)]
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1 and lines[0].startswith(f"inputs dropped (unexplained): synthetic_store read of {SOURCE} refused before any store was read")
    rep2q, _ = _pass(root, ad, _t(3050))
    assert rep2q.groups_refused == 0 and rep2q.groups_refused_standing == 1 and rep2q.results == [] and _head(root) == rev1
    rep2b, _ = _pass(root, ad, _t(3100), retry_refused=True)
    assert rep2b.groups_refused == 1 and rep2b.groups_inputs_dropped == 1 and _head(root) == rev1 and _paper_rows(root) == 501
    # the status the paper reads: the source is not ok, its detail says what refused, the refusal is listed under refuse
    ok, detail, extra = _q(root, "SELECT ok, detail, extra_json FROM v_source_status WHERE adapter = 'synthetic_store'")[0]
    assert ok == 0 and detail.startswith(f"latest read REFUSED (revision {rev2} at ") and RUN.INPUTS_DROPPED_KEY in detail and f"the accepted read {rev1} stays current" in detail
    assert any(RUN.INPUTS_DROPPED_KEY in r for r in json.loads(extra)["refuse"])
    # the tree is back, with five new stores: a superset of the head's inputs publishes normally and records the comparison
    for name, key, obj, scores in _stores(125) + _stores(5, start=501):
        _store(store_root, name, key, obj, scores)
    ad.read = real_read
    rep3, ing3 = _pass(root, ad, _t(4000))
    assert rep3.groups_refreshed == 1 and rep3.groups_refused == 0 and ing3.refused == 0
    rev3 = rep3.results[0].revision_id
    assert _head(root) == rev3 and _paper_rows(root) == 506
    ex3 = _extra(root, rev3)
    assert {k: ex3[k] for k in ("previous_revision_id", "inputs_previous", "inputs_now", "inputs_dropped", "inputs_retired", "inputs_added",
                                "inputs_dropped_fraction", "inputs_dropped_threshold")} == \
        {"previous_revision_id": rev1, "inputs_previous": 501, "inputs_now": 506, "inputs_dropped": 0, "inputs_retired": 0, "inputs_added": 5,
         "inputs_dropped_fraction": 0.0, "inputs_dropped_threshold": 0.1}
    assert "inputs_dropped_paths" not in ex3 and RUN.INPUTS_DROPPED_KEY not in ex3
    ok, detail, _ = _q(root, "SELECT ok, detail, extra_json FROM v_source_status WHERE adapter = 'synthetic_store'")[0]
    assert ok == 1 and not detail.startswith("latest read REFUSED")
    # the same 125 leave again, this time DECLARED: a glob for the first hundred, the other twenty-five by URI — the read publishes,
    # withdrawing the retired inputs' keys as any read that no longer carries them does
    _remove(store_root, gone)
    ad.inputs_retired = [os.path.join(store_root, "s0??", "store.json")] + [os.path.join(store_root, f"s{i:03d}", "store.json") for i in range(100, 125)]
    rep4, ing4 = _pass(root, ad, _t(5000))
    assert rep4.groups_refreshed == 1 and rep4.groups_refused == 0 and rep4.withdrawals > 0 and ing4.refused == 0
    rev4 = rep4.results[0].revision_id
    assert _head(root) == rev4 and _paper_rows(root) == 381
    ex4 = _extra(root, rev4)
    assert ex4["inputs_previous"] == 506 and ex4["inputs_now"] == 381 and ex4["inputs_dropped"] == 0 and ex4["inputs_retired"] == 125
    assert ex4["inputs_dropped_fraction"] == 0.0 and "inputs_dropped_paths" not in ex4
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ? AND to_revision_id = ? AND reason = 'removed_measurement'", SOURCE, rev4)[0][0] >= 125
    assert _q(root, "SELECT COUNT(*) FROM v_paper_current WHERE selection_state = 'withdrawn'")[0][0] == 125


# ============================================================================================
# what must not fire, and what the adapter may declare
# ============================================================================================

def test_first_read_superset_and_below_threshold_reads_publish_and_record_the_comparison(tmp_path):
    root, store_root, ad = _setup(tmp_path, 12)
    rep1, _ = _pass(root, ad, _t(2000))
    rev1 = rep1.results[0].revision_id
    assert rep1.groups_refreshed == 1 and "inputs_previous" not in _extra(root, rev1)
    for name, key, obj, scores in _stores(3, start=12):                          # superset: nothing dropped
        _store(store_root, name, key, obj, scores)
    rep2, _ = _pass(root, ad, _t(3000))
    rev2 = rep2.results[0].revision_id
    ex2 = _extra(root, rev2)
    assert rep2.groups_refreshed == 1 and (ex2["inputs_previous"], ex2["inputs_now"], ex2["inputs_dropped"], ex2["inputs_added"]) == (12, 15, 0, 3)
    _remove(store_root, ["s007"])                                                  # 1 of 15 = 6.7 %: below the threshold, published, named
    rep3, ing3 = _pass(root, ad, _t(4000))
    rev3 = rep3.results[0].revision_id
    assert rep3.groups_refreshed == 1 and rep3.groups_inputs_dropped == 0 and rep3.withdrawals > 0
    ex3 = _published_extra(root, rev3)
    assert ex3["inputs_dropped"] == 1 and ex3["inputs_dropped_fraction"] == pytest.approx(1 / 15, abs=1e-4) and ex3["inputs_dropped_paths"] == ["s007/store.json"]
    # THIS guard does not trip below the threshold -- but the run the missing store measured left the revision undeclared, and the
    # harvester's run-set drop guard (tests/test_registry_run_set_drop_guard.py) HOLDS the published revision: the head and the
    # frame stand until the store is back or the drop is declared (inputs_retired). The record still names the input comparison.
    assert ex3["run_set"] == {"previous_revision_id": rev2, "runs_previous": 15, "runs_now": 14, "runs_dropped": 1, "runs_added": 0}
    assert ing3.revisions_held_run_drop == 1 and _head(root) == rev2 and _paper_rows(root) == 15
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev3) == [(0,)]
    assert H.RUN_SET_DROP_KEY in _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev3)[0][0]
    _remove(store_root, ["s002", "s009"])                                          # 3 of 15 now missing against the head = 20 %: refused
    rep4, _ = _pass(root, ad, _t(4500))
    assert rep4.groups_inputs_dropped == 1 and _head(root) == rev2 and _paper_rows(root) == 15


def test_a_facts_only_group_never_trips_and_its_marker_says_so(tmp_path):
    """A group that emits no measurement by design (here a reader-declared `emits_no_runs`) has no key to withdraw silently: a
    large drop publishes, and the comparison on its marker carries `facts_only`."""
    root, store_root, ad = _setup(tmp_path, 4)
    ad.emits_no_runs = lambda group: group == GROUP
    ad.read = lambda ctx, group, inputs: iter([AB.RecordBatch(RUN.source_id_for(ad.name, group), group)])
    rep1, _ = _pass(root, ad, _t(2000))
    assert rep1.groups_refreshed == 1 and rep1.groups_emitted_nothing == 0
    _remove(store_root, ["s000", "s001"])                                          # 50 %
    rep2, ing2 = _pass(root, ad, _t(3000))
    assert rep2.groups_refreshed == 1 and rep2.groups_inputs_dropped == 0 and ing2.refused == 0
    ex = _extra(root, rep2.results[0].revision_id)
    assert ex["inputs_dropped"] == 2 and ex["inputs_dropped_fraction"] == 0.5 and ex["facts_only"] is True
    assert ex["inputs_dropped_paths"] == ["s000/store.json", "s001/store.json"]


def test_the_adapters_own_threshold_admits_a_drop_the_default_refuses(tmp_path):
    root, store_root, ad = _setup(tmp_path, 10)
    _pass(root, ad, _t(2000))
    rev1 = _head(root)
    _remove(store_root, ["s000", "s001", "s002"])                                 # 30 %
    rep2, _ = _pass(root, ad, _t(3000))
    assert rep2.groups_refused == 1 and rep2.groups_inputs_dropped == 1 and _head(root) == rev1 and _paper_rows(root) == 10
    ad.inputs_dropped_threshold = lambda group: 0.5 if group == GROUP else None      # a per-group declaration (a reader change: the window's flag reads it)
    rep3, _ = _pass(root, ad, _t(4000), retry_refused=True)
    assert rep3.groups_refreshed == 1 and rep3.groups_inputs_dropped == 0
    ex3 = _extra(root, rep3.results[0].revision_id)
    assert ex3["inputs_dropped"] == 3 and ex3["inputs_dropped_threshold"] == 0.5 and _paper_rows(root) == 7
    # a threshold of 0 refuses ANY undeclared drop; a bare number covers every group
    _remove(store_root, ["s003"])
    ad.inputs_dropped_threshold = 0
    rep4, _ = _pass(root, ad, _t(5000))
    assert rep4.groups_inputs_dropped == 1 and _paper_rows(root) == 7
    ad.inputs_dropped_threshold = 1.0                                                 # a reader change again: the window's flag
    rep5, _ = _pass(root, ad, _t(6000), retry_refused=True)
    assert rep5.groups_refreshed == 1 and _paper_rows(root) == 6


def test_malformed_declarations_refuse_by_name(tmp_path):
    root, store_root, ad = _setup(tmp_path, 4)
    _pass(root, ad, _t(2000))
    _remove(store_root, ["s000"])
    for bad in (1.5, -0.1, True, "0.3"):
        ad.inputs_dropped_threshold = bad
        with pytest.raises(ValueError, match="synthetic_store: inputs_dropped_threshold must be a number in \\[0, 1\\]"):
            with DB.reading(DB.db_path(root)) as conn:
                RUN.run_pass(root, [ad], conn=conn, now=_t(3000))
    ad.inputs_dropped_threshold = None
    for bad in ("a/single/glob", [1, 2], ["ok", ""]):
        ad.inputs_retired = bad
        with pytest.raises(ValueError, match="synthetic_store: inputs_retired must be a list of input URIs or globs"):
            with DB.reading(DB.db_path(root)) as conn:
                RUN.run_pass(root, [ad], conn=conn, now=_t(3000))
    assert AB.inputs_dropped_threshold_of(object(), GROUP) == AB.INPUTS_DROPPED_THRESHOLD == 0.1 and AB.inputs_retired_of(object(), GROUP) == ()


def test_a_re_activation_that_would_revert_the_head_to_a_much_smaller_published_set_is_refused(tmp_path):
    root, store_root, ad = _setup(tmp_path, 10)
    rep1, _ = _pass(root, ad, _t(2000))
    rev_small = rep1.results[0].revision_id
    for name, key, obj, scores in _stores(10, start=10):
        _store(store_root, name, key, obj, scores)
    rep2, _ = _pass(root, ad, _t(3000))
    rev_big = rep2.results[0].revision_id
    assert _head(root) == rev_big and _paper_rows(root) == 20
    # the ten added stores leave: the inputs are the published rev_small's again — a re-activation that would withdraw half the head
    _remove(store_root, [f"s{i:03d}" for i in range(10, 20)])
    rep3, ing3 = _pass(root, ad, _t(4000))
    (res,) = rep3.results
    assert res.revision_id == rev_small and not res.accepted and res.inputs_dropped and rep3.groups_inputs_dropped == 1
    assert res.error.startswith(f"{SOURCE}: {RUN.INPUTS_DROPPED_KEY}: 10 of the 20 inputs of the previous accepted revision {rev_big}")
    assert len(res.files) == 1 and "/_refused/" in res.files[0] and res.files[0].endswith("-source_watermark.jsonl")
    assert not os.path.isdir(os.path.join(root, "records", "adapters", "synthetic_store", "_activations", rev_small))   # no bundle
    assert ing3.refused == 0 and _head(root) == rev_big and _paper_rows(root) == 20
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev_small) == [(1,)]                 # never demoted
    # declared (a reader change, read by the window's --retry-refused), the re-activation goes through as before: the head reverts,
    # the ten keys are withdrawn
    ad.inputs_retired = [os.path.join(store_root, "s01?", "store.json")]
    rep4, ing4 = _pass(root, ad, _t(5000), retry_refused=True)
    assert rep4.groups_refreshed == 1 and ing4.refused == 0 and _head(root) == rev_small and _paper_rows(root) == 10


def test_the_cli_pass_reports_the_refusal_beside_the_others(tmp_path, monkeypatch, capsys):
    root, store_root, ad = _setup(tmp_path, 10)
    _pass(root, ad, _t(2000))
    _remove(store_root, [f"s{i:03d}" for i in range(3)])
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import dimensions as DIM
    monkeypatch.setattr(ADP, "load_all", lambda: {"synthetic_store": lambda: ad})
    monkeypatch.setattr(DIM, "adapter_legacy_spellings", lambda seeds: {})
    args = cli.build_parser().parse_args(["--root", root, "adapters", "run", "--workers", "0"])
    assert args.fn(args) == 0
    out = capsys.readouterr()
    summary, _ = json.JSONDecoder().raw_decode(out.out)              # the pass summary; the ingest summary follows it
    assert summary["groups_refused"] == 1 and summary["groups_inputs_dropped"] == 1 and summary["groups_refreshed"] == 0
    assert f"refused: synthetic_store {GROUP}: {SOURCE}: {RUN.INPUTS_DROPPED_KEY}: 3 of the 10 inputs" in out.err
