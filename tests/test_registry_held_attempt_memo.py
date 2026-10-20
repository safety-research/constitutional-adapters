"""A held revision is re-attempted only when something about it moved (registry follow-ups-4 item 10).

Measured on the live registry on 2026-10-06: cycles held the write lock 335-678 s with ~6 s of logged transactions, and the cycle
metric line reported bundles_held 435-856 -- the by-design refusals (objects_bank's per-object refusals, regrade_twin's, the glm
sampled_static group, the superseded em panel), each re-staged and rolled back every cycle: a ~330-s floor that timed the graders'
``pending.claim`` out on the lock. The held path now records, on the marker's queue row, the IDENTITY of what the attempt read
(``harvest.hold_identity``: the directory's sorted listing of path / size / mtime, the marker's sha, the hold's kind and parent, the judging
code's stamp and -- a blocked kind -- whether the parent is present) and when (``attempted_utc``). The blocked-queue re-evaluation SKIPS a
held directory whose identity stands and whose last attempt is younger than ``[harvest] held_sweep_s`` (default an hour): bookkeeping advances,
the held count keeps its meaning, ``revisions_held_skipped`` counts it on the report, the cycle state and the metric line. It re-attempts when
the identity moves (a file added or rewritten, a landed fix, an arrived parent), after the sweep interval, and ALWAYS under
``--retry-held`` (``registry ingest --retry-held``; the landing window's uncapped ``dimensions load`` and its ingest step set it). The hold
reason text and the hold semantics are untouched.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import cli
from sisteer.registry import config as CFG
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import status
from sisteer.registry import synth

from test_registry_harvest_budget import _metrics, _q  # noqa: E402
from test_registry_refused_hold_exact import _all_bundle_lines, _bundle_lines, _held, _marker  # noqa: E402


@pytest.fixture
def kind_rules_off(monkeypatch):
    """The kind-specific skip rules (an exact refused hold; a blocked hold whose parent is still absent) switched off, so every re-attempt
    decision below is the attempt-identity memo's alone -- the shape of a hold neither rule recognises, which re-attempted every cycle."""
    monkeypatch.setattr(H.Ingestor, "_refused_hold_exact", lambda self, ext: False)
    monkeypatch.setattr(H.Ingestor, "_held_still_blocked", lambda self, d, now: False)


def _cycle_line(root, now):
    return next(r for r in _metrics(root, now) if r["kind"] == "cycle")


def _memo(root, rev):
    """The directory's entry in state/held_attempts.json (keyed by the root-relative revision directory)."""
    memo = H.read_held_attempts(root)
    key = os.path.join("records", "adapters", "synthetic_store", rev)
    assert key in memo, (key, sorted(memo))
    return memo[key]


def test_the_hold_records_its_attempt_identity_and_the_identity_is_pure_in_the_directory_marker_code_and_parent(tmp_path):
    root, _, _, rev = _held(tmp_path)
    ext, attempts, checked = _marker(root)
    assert "attempt_identity" not in ext and "attempted_utc" not in ext, "the memo is never on the queue row: its extra_json replays byte for byte"
    rec = _memo(root, rev)
    assert len(rec["attempt_identity"]) == 64 and rec["attempted_utc"] == synth.t(2500) and attempts == 1 and checked == synth.t(2500)
    rev_dir = os.path.join(root, "records", "adapters", "synthetic_store", rev)
    assert os.path.isdir(rev_dir)
    # the marker's queue row carries what the identity is built from beside the memo: its sha, the hold's kind (extra) and parent columns
    (marker_sha, ptable, pid) = _q(root, "SELECT record_sha256, missing_parent_table, missing_parent_id FROM blocked_records WHERE record_type = 'source_revision'")[0]
    same = H.hold_identity(root, rev_dir, marker_sha, "refused", ptable, pid, parent_present=None, marker_extra=ext)
    assert same == rec["attempt_identity"] == H.hold_identity(root, rev_dir, marker_sha, "refused", ptable, pid, marker_extra=ext)
    # another marker, another kind, a parent that arrived, a stripped hold record, a touched file: each moves it; the same inputs reproduce it
    assert H.hold_identity(root, rev_dir, "0" * 64, "refused", ptable, pid, marker_extra=ext) != same
    assert H.hold_identity(root, rev_dir, marker_sha, "blocked", "object", "abc", parent_present=False) != H.hold_identity(root, rev_dir, marker_sha, "blocked", "object", "abc", parent_present=True)
    stripped = {k: v for k, v in ext.items() if k not in ("judged_with", "refused_file", "refused_file_sha")}
    assert H.hold_identity(root, rev_dir, marker_sha, "refused", ptable, pid, marker_extra=stripped) != same
    first = sorted(os.listdir(rev_dir))[0]
    st = os.stat(os.path.join(rev_dir, first))
    os.utime(os.path.join(rev_dir, first), ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    assert H.hold_identity(root, rev_dir, marker_sha, "refused", ptable, pid, marker_extra=ext) != same
    assert CFG.DEFAULTS["harvest"]["held_sweep_s"] == 3600 and H.HELD_SWEEP_S == 3600.0 and "held_sweep_s = 3600" in CFG.render_default(root)


def test_a_held_revision_is_attempted_once_skipped_while_its_identity_stands_and_the_counters_say_so(tmp_path, kind_rules_off):
    root, _, _, rev = _held(tmp_path)
    before = len(_all_bundle_lines(root))
    for k, t in enumerate((3000, 3500, 4000), start=2):
        now = synth.t(t)
        ing = H.ingest_once(root, workers=0, min_free=0, now=now)
        assert _bundle_lines(root, now) == [], _bundle_lines(root, now)                   # not re-attempted: no bundle line, nothing staged
        assert ing.revisions_held == 1 and ing.revisions_held_skipped == 1 and ing.refused == 0
        assert ing.revision_transactions == 0 and ing.staging_transactions == 0
        line = _cycle_line(root, now)
        assert line["bundles_held"] == 1 and line["bundles_held_skipped"] == 1
        ext, attempts, checked = _marker(root)
        assert (attempts, checked) == (k, now) and _memo(root, rev)["attempted_utc"] == synth.t(2500)   # the bookkeeping advances, the memo stands
        assert H.read_cycle_state(root)["revisions_held_skipped"] == 1
        assert status.harvest_cycle_section(root)["revisions_held_skipped"] == 1
    assert len(_all_bundle_lines(root)) == before and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    assert ing.as_dict()["revisions_held_skipped"] == 1


def test_a_changed_directory_re_attempts_and_the_memo_is_recorded_afresh(tmp_path, kind_rules_off):
    root, _, _, rev = _held(tmp_path)
    rev_dir = os.path.join(root, "records", "adapters", "synthetic_store", rev)
    # a touched constituent: the listing moved (its mtime), so the identity moved -- the directory is attempted again (the main pass reads
    # the changed file; the queue would otherwise) and the attempt records the new identity at its own clock
    touched = os.path.join(rev_dir, sorted(f for f in os.listdir(rev_dir) if f.endswith("-run_metric.jsonl"))[0])
    st = os.stat(touched)
    os.utime(touched, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    now = synth.t(3000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert [r for r, *_ in _bundle_lines(root, now)] == [rev] and ing.revisions_held == 1 and ing.revisions_held_skipped == 0 and ing.refused == 1
    (marker_sha, ptable, pid) = _q(root, "SELECT record_sha256, missing_parent_table, missing_parent_id FROM blocked_records WHERE record_type = 'source_revision'")[0]
    assert _memo(root, rev)["attempted_utc"] == now and _memo(root, rev)["attempt_identity"] == H.hold_identity(root, rev_dir, marker_sha, "refused", ptable, pid, marker_extra=_marker(root)[0])
    # and stands again from there
    now2 = synth.t(3500)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert _bundle_lines(root, now2) == [] and ing2.revisions_held_skipped == 1 and ing2.refused == 0


def test_the_sweep_interval_re_attempts_a_standing_hold_once_and_is_configurable(tmp_path, kind_rules_off):
    root, _, _, rev = _held(tmp_path)
    # a sweep of ten seconds: the attempt at t(2500) is older than that at t(3000) -> re-attempted (one bundle retry line), the memo refreshed
    now = synth.t(3000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now, held_sweep_s=10.0)
    assert [(r, res, retry) for r, res, _n, retry in _bundle_lines(root, now)] == [(rev, "blocked", True)] and ing.revisions_held_skipped == 0 and ing.refused == 1
    assert _memo(root, rev)["attempted_utc"] == now
    # the default sweep (an hour) from the config: a cycle a few minutes on skips it
    now2 = synth.t(3300)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert _bundle_lines(root, now2) == [] and ing2.revisions_held_skipped == 1
    # the knob in config.toml
    with open(CFG.config_path(root), "w", encoding="utf-8") as fh:
        fh.write("[harvest]\nheld_sweep_s = 100\n")
    now3 = synth.t(3450)                                                                   # 150 s after the refresh: past a 100-s sweep
    ing3 = H.ingest_once(root, workers=0, min_free=0, now=now3)
    assert [(r, retry) for r, _res, _n, retry in _bundle_lines(root, now3)] == [(rev, True)] and ing3.revisions_held_skipped == 0


def test_retry_held_re_attempts_a_standing_hold_from_the_api_and_the_cli(tmp_path, kind_rules_off, monkeypatch):
    root, _, _, rev = _held(tmp_path)
    now = synth.t(3000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now, retry_held=True)
    assert [(r, retry) for r, _res, _n, retry in _bundle_lines(root, now)] == [(rev, True)] and ing.revisions_held_skipped == 0 and ing.refused == 1
    now2 = synth.t(3200)
    assert H.ingest_once(root, workers=0, min_free=0, now=now2).revisions_held_skipped == 1 and _bundle_lines(root, now2) == []
    monkeypatch.setattr(H, "free_bytes", lambda path: 100 * H.GB)                       # the CLI's cycle reads the floor: never the host's disk
    before = len(_all_bundle_lines(root))
    assert cli.main(["--root", root, "ingest", "--workers", "0", "--retry-held"]) == 0
    lines = _all_bundle_lines(root)
    assert len(lines) == before + 1 and lines[-1]["dir"].endswith(rev) and lines[-1].get("retry") is True
    assert cli.main(["--root", root, "ingest", "--workers", "0"]) == 0
    assert len(_all_bundle_lines(root)) == before + 1                                     # without the flag the memo stands again


def test_a_marker_recorded_before_the_memo_re_attempts_once_and_is_memoized_from_then_on(tmp_path, kind_rules_off):
    root, _, _, rev = _held(tmp_path)
    os.remove(os.path.join(root, "state", H.HELD_ATTEMPTS_NAME))                         # a hold recorded before the rule: no memo on disk
    assert H.read_held_attempts(root) == {}
    now = synth.t(3000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert [(r, retry) for r, _res, _n, retry in _bundle_lines(root, now)] == [(rev, True)] and ing.revisions_held_skipped == 0
    assert _memo(root, rev)["attempted_utc"] == now
    now2 = synth.t(3500)
    assert H.ingest_once(root, workers=0, min_free=0, now=now2).revisions_held_skipped == 1 and _bundle_lines(root, now2) == []


def test_the_existing_kind_rules_still_decide_first_and_the_refused_escape_bypasses_the_memo(tmp_path):
    """Without the switch: the exact refused hold is the first rule (its own counter, no `skipped`), and `--retry-refused` re-attempts it
    through the memo too (an escape that re-attempts must re-attempt)."""
    root, _, _, rev = _held(tmp_path)
    now = synth.t(3000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert _bundle_lines(root, now) == [] and ing.revisions_held == 1 and ing.revisions_held_skipped == 0
    now2 = synth.t(3500)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2, retry_refused=True)
    assert [(r, retry) for r, _res, _n, retry in _bundle_lines(root, now2)] == [(rev, True)] and ing2.refused == 1 and ing2.revisions_held_skipped == 0
