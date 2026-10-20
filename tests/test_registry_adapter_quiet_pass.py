"""The quiet adapter pass costs nothing and folds nothing (S2, S3; plan Phase 1 steps 3 and 4): a group whose last read REFUSED
is left standing across five consecutive quiet passes (no read, no file, the lock never taken) and read again only on an explicit
cause — `--retry-refused`, an input change, an `EMISSION_VERSION` change, `--full` — while the refusal stays visible; a retired
input that comes back is read again as before; `registry adapters run` skips the fold when it published nothing and never folds
under `--no-ingest`, its last line parseable either way; `--retry-refused` reaches the runner.
"""
from __future__ import annotations

import json
import os
import shutil
from unittest import mock

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _cli, _setup, _store  # noqa: E402
from test_registry_adapter_pass import _pass_and_ingest as _pass, _q1 as _q  # noqa: E402


class CountingAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter, counting how often a group is READ (a refusal raised inside the read counts as one)."""

    def __init__(self, store_root: str):
        super().__init__(store_root)
        self.reads = 0

    def read(self, ctx, group, inputs):
        self.reads += 1
        yield from super().read(ctx, group, inputs)


def _last_error(root, p):
    return _q(root, "SELECT last_error FROM source_watermark WHERE input_uri = ?", p)[0][0]


def _refused_setup(tmp_path):
    """An accepted read, then a store change the (now broken) reader refuses: the refusal is folded and stands."""
    root, store_root = _setup(tmp_path)
    ad = CountingAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    p = _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.5, 0.5])
    ad.broken.add(p)
    ad.reads = 0
    rep, ing = _pass(root, ad, synth.t(3000))
    assert rep.groups_refused == 1 and rep.groups_refused_standing == 0 and ad.reads == 1 and ing.refused == 0
    (res,) = rep.results
    assert not res.accepted and "synthetic read failure" in res.error
    assert res.files and all(os.path.exists(f) for f in res.files)                        # the refusal file is on the result
    assert _last_error(root, p) is not None
    return root, store_root, ad, p, res.revision_id


def _quiet(root, ad, now, refused_rev, files_before):
    ad.reads = 0
    rep, ing = _pass(root, ad, now)
    assert ad.reads == 0, "a standing refusal was read again on a quiet pass"
    assert rep.groups_refused == 0 and rep.groups_refreshed == 0 and rep.groups_refused_standing == 1 and rep.inputs_changed == 0
    assert rep.refused_standing == {ad.name: ["s1"]} and rep.results == [] and rep.records_written == 0
    assert sorted(os.listdir(RUN.refused_dir(root, ad.name, refused_rev))) == files_before  # no file written
    assert ing.ok == 0 and not ing.changed                                                  # nothing to fold
    return rep


def test_a_standing_refusal_is_left_alone_across_five_quiet_passes_and_stays_visible(tmp_path):
    root, store_root, ad, p, refused_rev = _refused_setup(tmp_path)
    rdir = RUN.refused_dir(root, ad.name, refused_rev)
    files_before = sorted(os.listdir(rdir))
    assert len(files_before) == 1
    for i in range(5):
        _quiet(root, ad, synth.t(3100 + 10 * i), refused_rev, files_before)
    # visible throughout: the watermark's last_error, the page's adapter-pass rows, the unaccepted marker, the head untouched
    assert "synthetic read failure" in _last_error(root, p)
    assert any(row.get("last_error") for row in status.status_json(root)["adapter_pass"])
    assert _q(root, "SELECT accepted FROM source_revision ORDER BY observed_utc") == [(1,), (0,)]
    assert _q(root, "SELECT n FROM run_metric") == [(3,)]


def test_retry_refused_reads_a_standing_refusal_once(tmp_path):
    root, store_root, ad, p, refused_rev = _refused_setup(tmp_path)
    rdir = RUN.refused_dir(root, ad.name, refused_rev)
    files_before = sorted(os.listdir(rdir))
    _quiet(root, ad, synth.t(3100), refused_rev, files_before)
    ad.reads = 0
    rep, ing = _pass(root, ad, synth.t(3200), retry_refused=True)
    assert ad.reads == 1 and rep.groups_refused == 1 and rep.groups_refused_standing == 0 and rep.inputs_changed == 1
    assert len(os.listdir(rdir)) == 2                                                       # the retried refusal wrote its own file
    _quiet(root, ad, synth.t(3300), refused_rev, sorted(os.listdir(rdir)))                 # and stands again afterwards
    # the reader is fixed, the input bytes unchanged: the window's flag is what reads it, and the read heals the source
    ad.broken.clear()
    _quiet(root, ad, synth.t(3400), refused_rev, sorted(os.listdir(rdir)))
    ad.reads = 0
    rep2, ing2 = _pass(root, ad, synth.t(3500), retry_refused=True)
    assert ad.reads == 1 and rep2.groups_refreshed == 1 and rep2.groups_refused_standing == 0 and ing2.refused == 0
    assert _last_error(root, p) is None and _q(root, "SELECT n FROM run_metric") == [(2,)]
    rep3, _ = _pass(root, ad, synth.t(3600), retry_refused=True)
    assert rep3.inputs_changed == 0 and rep3.groups_refreshed == 0 and rep3.groups_refused_standing == 0   # nothing standing to retry


def test_an_input_change_reads_a_standing_refusal(tmp_path):
    root, store_root, ad, p, refused_rev = _refused_setup(tmp_path)
    _quiet(root, ad, synth.t(3100), refused_rev, sorted(os.listdir(RUN.refused_dir(root, ad.name, refused_rev))))
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.6, 0.6, 0.6])   # new bytes; the reader still broken
    ad.reads = 0
    rep, _ = _pass(root, ad, synth.t(3200))
    assert ad.reads == 1 and rep.inputs_changed == 1 and rep.groups_refused == 1 and rep.groups_refused_standing == 0
    (res,) = rep.results
    assert res.revision_id != refused_rev                                                  # a changed input is a new revision id
    _quiet(root, ad, synth.t(3300), res.revision_id, sorted(os.listdir(RUN.refused_dir(root, ad.name, res.revision_id))))


def test_an_emission_version_change_reads_a_standing_refusal_once(tmp_path):
    root, store_root, ad, p, refused_rev = _refused_setup(tmp_path)
    _quiet(root, ad, synth.t(3100), refused_rev, sorted(os.listdir(RUN.refused_dir(root, ad.name, refused_rev))))
    ad.EMISSION_VERSION = 2                                                                 # the reader bumped AFTER the refusal
    ad.reads = 0
    rep, _ = _pass(root, ad, synth.t(3200))
    assert ad.reads == 1 and rep.groups_refused == 1 and rep.groups_refused_standing == 0
    (res,) = rep.results
    assert res.revision_id != refused_rev
    assert _q(root, "SELECT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE revision_id = ?", res.revision_id) == [("2",)]
    # refused under the current version: stands quiet, whatever the accepted head's older version says
    _quiet(root, ad, synth.t(3300), res.revision_id, sorted(os.listdir(RUN.refused_dir(root, ad.name, res.revision_id))))
    _quiet(root, ad, synth.t(3400), res.revision_id, sorted(os.listdir(RUN.refused_dir(root, ad.name, res.revision_id))))


def test_full_reads_a_standing_refusal(tmp_path):
    root, store_root, ad, p, refused_rev = _refused_setup(tmp_path)
    rdir = RUN.refused_dir(root, ad.name, refused_rev)
    _quiet(root, ad, synth.t(3100), refused_rev, sorted(os.listdir(rdir)))
    ad.reads = 0
    rep, _ = _pass(root, ad, synth.t(3200), full=True)
    assert ad.reads == 1 and rep.groups_refused == 1 and rep.groups_refused_standing == 0 and len(os.listdir(rdir)) == 2
    ad.reads = 0
    rep_r, _ = _pass(root, ad, synth.t(3300), reread=True)                                # --reread reads every present group too
    assert ad.reads == 1 and rep_r.groups_refused_standing == 0


def test_plan_groups_names_the_standing_refusals_and_the_rest_of_the_deal(tmp_path):
    """The deal decided before any read: the standing refusal, a changed sibling, a vanished prior-only group."""
    root, store_root, ad, p, refused_rev = _refused_setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(store_root, "s3", "legacy_store_3", "none", [0.4])
    _pass(root, ad, synth.t(3100))                                                          # s2, s3 accepted; s1 standing
    shutil.rmtree(os.path.join(store_root, "s3"))                                           # s3 vanishes; s2 unchanged
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        inputs = ad.discover()
        marks = RUN._watermarks(conn, ad.name)
        assert marks == RUN._watermarks_by_adapter(conn)[ad.name]                           # the one-scan read, the same watermarks
        assert marks[p].refused and marks[p].attempt_revision_id == refused_rev            # the refusal stamp names its attempt
        plan = RUN.plan_groups(ad, inputs, marks, "1", conn)
        assert sorted(plan.groups) == ["s1", "s2"] and plan.changed == set() and plan.refused_standing == ["s1"] and plan.refused == {"s1"}
        assert sorted(plan.vanished) == ["s3"] and plan.gone_in_group == {} and plan.inputs_changed == 0
        plan_r = RUN.plan_groups(ad, inputs, marks, "1", conn, retry_refused=True)
        assert plan_r.changed == {"s1"} and plan_r.refused == {"s1"} and plan_r.refused_standing == [] and plan_r.inputs_changed == 1
        plan_v = RUN.plan_groups(ad, inputs, marks, "2", conn)                              # a bumped reader: every group moves
        assert plan_v.changed == {"s1", "s2"} and plan_v.refused_standing == []
        plan_f = RUN.plan_groups(ad, inputs, marks, "1", conn, full=True)
        assert plan_f.changed == {"s1", "s2"} and plan_f.inputs_changed == 2 and plan_f.refused == {"s1"}   # the stamp, whatever selected it
    finally:
        conn.close()
    assert RUN.Watermark("s", ("a", 1, "t"), "x: refused").refused and not RUN.Watermark("s", ("a", 1, "t"), "x: refused").retired
    assert RUN.Watermark("s", ("a", 1, "t"), RUN.VANISHED_PREFIX + "u").retired and not RUN.Watermark("s", ("a", 1, "t"), RUN.VANISHED_PREFIX + "u").refused
    assert not RUN.Watermark("s", ("a", 1, "t"), None).refused


def test_a_reader_traversal_adapter_with_a_standing_refusal_is_not_re_read_whole(tmp_path):
    """The sampled_static shape: one refused group used to re-read EVERY group of a shared-RNG adapter, every pass."""
    root, store_root, ad, p, refused_rev = _refused_setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad.rng_scope = "reader_traversal"
    ad.reads = 0
    rep, _ = _pass(root, ad, synth.t(3100))                                                 # s2 is new: the traversal moved, s1 re-read too
    assert rep.groups_refreshed == 1 and rep.groups_refused == 1 and rep.groups_refused_standing == 0 and ad.reads == 2
    for i in range(3):
        ad.reads = 0
        rep_q, _ = _pass(root, ad, synth.t(3200 + 10 * i))
        assert ad.reads == 0 and rep_q.groups_refused_standing == 1 and rep_q.groups_refreshed == 0 and rep_q.results == []


def test_a_retired_input_that_comes_back_is_read_again(tmp_path):
    """A retired watermark (``input vanished: …``) is NOT a standing refusal: the input's return is new information."""
    root, store_root = _setup(tmp_path)
    ad = CountingAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    p = os.path.join(store_root, "s1", "store.json")
    st = os.stat(p)
    with open(p, "rb") as fh:
        saved = fh.read()
    shutil.rmtree(os.path.join(store_root, "s1"))
    rep, _ = _pass(root, ad, synth.t(3000))
    assert rep.withdrawals >= 1 and _last_error(root, p).startswith(RUN.VANISHED_PREFIX)
    os.makedirs(os.path.dirname(p))
    with open(p, "wb") as fh:
        fh.write(saved)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))                                       # identical bytes and mtime: the watermark's stat
    ad.reads = 0
    rep2, ing2 = _pass(root, ad, synth.t(4000))
    assert rep2.inputs_changed == 1 and rep2.groups_refreshed == 1 and rep2.groups_refused_standing == 0 and ing2.refused == 0
    assert _last_error(root, p) is None and _q(root, "SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1") == [(1,)]


# ============================================================================================
# the CLI: no fold when nothing was published, never under --no-ingest; the flags reach the runner
# ============================================================================================

SKIPPED_QUIET = {"ingest_ok": 0, "refused": 0, "blocked": 0, "ingest": "skipped: nothing published"}
SKIPPED_FLAG = {"ingest_ok": 0, "refused": 0, "blocked": 0, "ingest": "skipped: --no-ingest"}


def test_nothing_published_skips_the_fold_and_the_lock_is_never_taken(tmp_path, monkeypatch, capsys):
    root, store_root = _setup(tmp_path)
    ad = CountingAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rc, summary, last, err, lock = _cli(root, ad, monkeypatch, capsys)
    assert rc == 0 and summary["groups_refreshed"] == 0 and summary["inputs_changed"] == 0
    assert last == SKIPPED_QUIET and lock.call_count == 0
    # a standing refusal is not "published": five quiet passes take no lock, and each names what stands
    p = _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.5, 0.5])
    ad.broken.add(p)
    rc, summary, last, err, lock = _cli(root, ad, monkeypatch, capsys)
    assert summary["groups_refused"] == 1 and last == {"ingest_ok": 7, "refused": 1, "blocked": 2}
    assert lock.call_count == 1                                                             # the refusal file WAS published: folded
    H.ingest_once(root, workers=0, min_free=0)
    for _ in range(5):
        rc, summary, last, err, lock = _cli(root, ad, monkeypatch, capsys)
        assert rc == 0 and summary["groups_refused"] == 0 and summary["groups_refused_standing"] == 1 and last == SKIPPED_QUIET
        assert lock.call_count == 0
        assert "refusals standing (not re-read" in err and f"{ad.name}: 1 group(s): s1" in err


def test_no_ingest_never_folds_and_leaves_the_records_for_the_harvester(tmp_path, monkeypatch, capsys):
    root, store_root = _setup(tmp_path)
    ad = CountingAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    rc, summary, last, err, lock = _cli(root, ad, monkeypatch, capsys, "--no-ingest")
    assert rc == 0 and summary["groups_refreshed"] == 1 and summary["records_written"] > 0
    assert last == SKIPPED_FLAG and lock.call_count == 0
    ing = H.ingest_once(root, workers=0, min_free=0)                                        # the harvester's own tick folds it
    assert ing.ok > 0 and _q(root, "SELECT n FROM run_metric") == [(4,)]
    rc, summary, last, err, lock = _cli(root, ad, monkeypatch, capsys, "--no-ingest", "--retry-refused", "--full")
    assert rc == 0 and last == SKIPPED_FLAG and lock.call_count == 0                       # the window's shape: never a fold here


def test_something_published_folds_once_without_no_ingest(tmp_path, monkeypatch, capsys):
    root, store_root = _setup(tmp_path)
    ad = CountingAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    rc, summary, last, err, lock = _cli(root, ad, monkeypatch, capsys)
    assert rc == 0 and summary["groups_refreshed"] == 1
    assert lock.call_count == 1 and lock.call_args == mock.call(root, workers=0)
    assert last == {"ingest_ok": 7, "refused": 1, "blocked": 2}                             # the operator's shape, unchanged


def test_retry_refused_reaches_the_runner(monkeypatch, capsys, tmp_path):
    root, store_root = _setup(tmp_path)
    seen = {}

    def fake_run_pass(root_, adapters, conn=None, **kw):
        seen.update(kw)
        return RUN.PassReport()
    monkeypatch.setattr(RUN, "run_pass", fake_run_pass)
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import dimensions as DIM
    monkeypatch.setattr(ADP, "load_all", lambda: {"x": lambda: CountingAdapter(store_root)})
    monkeypatch.setattr(DIM, "adapter_legacy_spellings", lambda seeds: {})
    monkeypatch.setattr(H, "ingest_under_lock", mock.MagicMock(side_effect=AssertionError("must not fold")))
    args = cli.build_parser().parse_args(["--root", root, "adapters", "run", "--workers", "0", "--retry-refused"])
    assert args.fn(args) == 0 and seen["retry_refused"] is True and seen["full"] is False
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == SKIPPED_QUIET
    ns = cli.build_parser().parse_args(["adapters", "run", "--incremental"])
    assert ns.retry_refused is False and ns.no_ingest is False and ns.profile is None


def test_inputs_gone_before_the_refusal_do_not_re_read_it_but_an_input_gone_after_does(tmp_path):
    """The inputs-dropped shape (a store tree absent): the refused read never stamped the absent inputs, whose watermarks stay the
    accepted read's, so the tree still being away is not new information — while an input that vanishes AFTER a refusal carries a
    refused watermark and reads the group again."""
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(store_root, "s3", "legacy_store_3", "none", [0.4])
    ad = CountingAdapter(store_root)
    ad.comparison_group = lambda inp: "one_group"                                            # three stores, ONE comparison group
    ad.inputs_dropped_threshold = 0.9                                                       # the drop guard is not what this test is about
    _pass(root, ad, synth.t(2000))
    shutil.rmtree(os.path.join(store_root, "s3"))                                           # s3 gone; the reader breaks on s1
    p1 = os.path.join(store_root, "s1", "store.json")
    ad.broken.add(p1)
    ad.reads = 0
    rep, _ = _pass(root, ad, synth.t(3000))
    assert ad.reads == 1 and rep.groups_refused == 1
    refused_rev = rep.results[0].revision_id
    p3 = os.path.join(store_root, "s3", "store.json")
    assert _last_error(root, p3) is None and _last_error(root, p1) is not None              # s3's watermark is still the accepted read's
    for i in range(3):                                                                       # the tree still away: the refusal stands
        ad.reads = 0
        rep_q, _ = _pass(root, ad, synth.t(3100 + 10 * i))
        assert ad.reads == 0 and rep_q.groups_refused_standing == 1 and rep_q.results == [], rep_q
    shutil.rmtree(os.path.join(store_root, "s2"))                                           # s2 vanishes AFTER the refusal: new information
    ad.reads = 0
    rep2, _ = _pass(root, ad, synth.t(3200))
    assert ad.reads == 1 and rep2.groups_refused == 1 and rep2.groups_refused_standing == 0 and rep2.results[0].revision_id != refused_rev
    # the re-read's stamp names the NEW inventory (s1 alone), so the refusal stands again: s2's stale refused watermark — never
    # retired, since a refusal retires nothing — is not a reason to retry; before the attempt id it re-read the group every pass
    assert _last_error(root, os.path.join(store_root, "s2", "store.json")) is not None
    for i in range(5):
        ad.reads = 0
        rep_q, _ = _pass(root, ad, synth.t(3300 + 10 * i))
        assert ad.reads == 0 and rep_q.groups_refused_standing == 1 and rep_q.results == [], rep_q


def test_an_input_restored_with_its_old_bytes_and_mtime_reads_a_standing_refusal(tmp_path):
    """The other hole the attempt id closes: a group of two inputs loses one (the inputs-dropped refusal stamps the present one),
    then the missing input comes back byte-identical with its mtime restored — its own watermark is the accepted read's and
    unchanged, so no per-input flag moves; the inventory did, and the group recovers through a genuine re-read of the head (the
    revision these inputs mint is the committed head, and a refused stamp turns clean only after the reader ran)."""
    root, store_root = _setup(tmp_path)
    p2 = _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = CountingAdapter(store_root)
    ad.comparison_group = lambda inp: "one_group"
    _pass(root, ad, synth.t(2000))
    st = os.stat(p2)
    with open(p2, "rb") as fh:
        saved = fh.read()
    shutil.rmtree(os.path.join(store_root, "s2"))
    rep, _ = _pass(root, ad, synth.t(3000))                                                  # 1 of 2 inputs gone, undeclared: refused unread
    assert rep.groups_refused == 1 and rep.groups_inputs_dropped == 1
    p1 = os.path.join(store_root, "s1", "store.json")
    assert _last_error(root, p1) is not None and _last_error(root, p2) is None              # the absent input keeps its accepted watermark
    for i in range(3):
        ad.reads = 0
        rep_q, _ = _pass(root, ad, synth.t(3100 + 10 * i))
        assert ad.reads == 0 and rep_q.results == [] and rep_q.groups_refused_standing == 1
    os.makedirs(os.path.dirname(p2))
    with open(p2, "wb") as fh:
        fh.write(saved)
    os.utime(p2, ns=(st.st_atime_ns, st.st_mtime_ns))                                       # identical bytes and mtime: its watermark's stat
    ad.reads = 0
    rep2, ing2 = _pass(root, ad, synth.t(3500))
    assert rep2.groups_refreshed == 1 and rep2.groups_reread == 1 and rep2.groups_refused_standing == 0 and ing2.refused == 0 and ad.reads == 1
    assert _last_error(root, p1) is None and _last_error(root, p2) is None
    rep3, _ = _pass(root, ad, synth.t(3600))
    assert rep3.results == [] and rep3.groups_refused_standing == 0


def test_retry_refused_on_a_committed_revision_runs_the_reader_before_the_refusal_clears(tmp_path):
    """A refusal stamped on a COMMITTED revision (a re-read of the head that refused) is retried on the flag through a genuine
    re-read: a still-broken reader keeps its refusal, a fixed one runs and heals it. The incremental path would have re-stamped
    the watermarks clean off the committed bytes without running the reader."""
    root, store_root = _setup(tmp_path)
    ad = CountingAdapter(store_root)
    _pass(root, ad, synth.t(2000))                                                           # the head, accepted with its status facts
    p = os.path.join(store_root, "s1", "store.json")
    (head,) = [r[0] for r in _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1")]
    ad.broken.add(p)
    rep, _ = _pass(root, ad, synth.t(3000), reread=True)                                     # the re-read refuses: a watermark-only stamp
    assert rep.groups_refused == 1 and _last_error(root, p) is not None
    assert _q(root, "SELECT json_extract(extra_json, '$.attempt_revision_id') FROM source_watermark WHERE input_uri = ?", p) == [(head,)]
    ad.reads = 0
    rep_q, _ = _pass(root, ad, synth.t(3100))                                                # quiet: stands, no read, nothing re-stamped
    assert ad.reads == 0 and rep_q.groups_refused_standing == 1 and rep_q.results == [] and _last_error(root, p) is not None
    ad.reads = 0
    rep_r, _ = _pass(root, ad, synth.t(3200), retry_refused=True)                            # still broken: the reader runs, refuses again
    assert ad.reads == 1 and rep_r.groups_refused == 1 and rep_r.groups_refreshed == 0 and _last_error(root, p) is not None
    ad.broken.clear()
    ad.reads = 0
    rep_f, ing_f = _pass(root, ad, synth.t(3300), retry_refused=True)                        # fixed: the reader runs, the refusal clears
    assert ad.reads == 1 and rep_f.groups_refreshed == 1 and rep_f.groups_reread == 1 and rep_f.results[0].reread and ing_f.refused == 0
    assert _last_error(root, p) is None and _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1") == [(head,)]


def _refused_committed_head(tmp_path, **adapter_attrs):
    """A head accepted with its status facts, then a ``--reread`` of it that REFUSED (the reader broken): the refusal is a
    watermark-only stamp on a COMMITTED revision, standing quiet."""
    root, store_root = _setup(tmp_path)
    ad = CountingAdapter(store_root)
    for k, v in adapter_attrs.items():
        setattr(ad, k, v)
    _pass(root, ad, synth.t(2000))
    p = os.path.join(store_root, "s1", "store.json")
    ad.broken.add(p)
    rep, _ = _pass(root, ad, synth.t(3000), reread=True)
    assert rep.groups_refused == 1 and _last_error(root, p) is not None
    ad.reads = 0
    rep_q, _ = _pass(root, ad, synth.t(3100))
    assert ad.reads == 0 and rep_q.groups_refused_standing == 1 and rep_q.results == []
    return root, store_root, ad, p


def test_full_over_a_refused_committed_head_runs_the_reader_and_a_broken_one_keeps_its_refusal(tmp_path):
    """``--full`` selects every group; a refused COMMITTED head among them must be read, not re-stamped clean off its bytes."""
    root, store_root, ad, p = _refused_committed_head(tmp_path)
    ad.reads = 0
    rep, _ = _pass(root, ad, synth.t(3200), full=True)                                       # still broken: read, refused again
    assert ad.reads == 1 and rep.groups_refused == 1 and rep.groups_refreshed == 0 and _last_error(root, p) is not None
    ad.broken.clear()
    ad.reads = 0
    rep2, ing2 = _pass(root, ad, synth.t(3300), full=True)                                   # fixed: read, healed
    assert ad.reads == 1 and rep2.groups_refreshed == 1 and rep2.groups_reread == 1 and ing2.refused == 0 and _last_error(root, p) is None


def test_a_traversal_siblings_change_re_reads_a_refused_committed_head_and_a_broken_reader_keeps_its_refusal(tmp_path):
    """A ``reader_traversal`` adapter re-reads EVERY group when one changes; the refused committed head among them is read too,
    and its refusal stands while the reader is broken — it never clears without a read."""
    root, store_root, ad, p = _refused_committed_head(tmp_path, rng_scope="reader_traversal")
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])                          # a sibling appears: the traversal moved
    ad.reads = 0
    rep, _ = _pass(root, ad, synth.t(3200))
    assert ad.reads == 2 and rep.groups_refreshed == 1 and rep.groups_refused == 1 and rep.groups_refused_standing == 0
    assert _last_error(root, p) is not None                                                # s1 read again, refused again, still visible
    ad.reads = 0
    rep_q, _ = _pass(root, ad, synth.t(3300))
    assert ad.reads == 0 and rep_q.groups_refused_standing == 1 and rep_q.results == []       # and stands again
    ad.broken.clear()
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9, 0.3])                     # the sibling changes again, the reader fixed
    ad.reads = 0
    rep2, ing2 = _pass(root, ad, synth.t(3400))
    assert ad.reads == 2 and rep2.groups_refreshed == 2 and rep2.groups_reread == 1 and ing2.refused == 0 and _last_error(root, p) is None
