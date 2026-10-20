"""`RecordWriter` allocates its sequence number AT WRITE TIME.

Before: `_next_seq` listed the run directory ONCE, lazily at the writer's first write, and counted up from that cached census. A writer
constructed early and used after a long phase re-issued a number that had arrived meanwhile -- the harvester's gz rename, an earlier pass,
the cluster mirror delivering a producer record into the same run directory: a pass wrote `000016-run_degeneration.jsonl` while the log
already held `000016-run_degeneration.jsonl.gz` (2026-10-06). The exclusive create (O_EXCL) in `write_many` could not see it: a `.gz` twin
is another name. Now every write re-lists the directory (every entry's numeric head, `.gz` included), takes one above the greater of that
and the writer's last number, creates the file O_EXCL, and on EEXIST lists again and retries, bounded (`SEQ_ATTEMPTS`, then a RuntimeError
naming the path). `write_many` keeps its semantics (one file, many records, one written_utc stamp) and the disabled writer still writes
nothing. Monotone within one writer: the number issued never goes down.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import emit
from sisteer.registry import records as R
from sisteer.registry import synth

RUN = "c:e:i:0"


def _writer(tmp_path, **kw):
    return emit.RecordWriter(str(tmp_path), "c", RUN, synth.producer(), disabled=False, **kw)


def _names(w):
    return [os.path.basename(p) for p in w.written]


def _seq_of(path):
    return int(os.path.basename(path).split("-", 1)[0])


def test_two_writers_on_one_directory_interleaving_never_produce_the_same_logical_uri_and_each_stays_monotone(tmp_path):
    w1, w2 = _writer(tmp_path), _writer(tmp_path)
    paths = []
    for i in range(6):
        paths.append(w1.emit_state("running", "job_log") if i % 2 == 0 else w1.emit_patch(fields={"gpu_count": 1}))
        paths.append(w2.emit_state("generated", "completion_artifact") if i % 2 == 0 else w2.emit_patch(fields={"gpu_count": 2}))
    uris = [R.logical_uri(p) for p in paths]
    assert len(set(uris)) == len(uris) == 12
    assert sorted(_seq_of(p) for p in paths) == list(range(1, 13)), "the twelve files take twelve distinct numbers"
    for w in (w1, w2):
        seqs = [_seq_of(p) for p in w.written]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs       # monotone within a writer
        assert w._seq == seqs[-1]                                                 # the attribute is the last number issued
    assert all(sum(1 for _ in R.read_record_file(p)) == 1 for p in paths)


def test_a_gz_twin_occupies_its_number_and_a_pre_created_colliding_file_makes_the_writer_skip_to_the_next(tmp_path):
    w = _writer(tmp_path)
    p1 = w.emit_state("running", "job_log")
    assert _seq_of(p1) == 1
    # the harvester renamed a producer file to .gz at number 2 (another record type: the exclusive create would never see it)
    rec = R.make_record("run_degeneration", {"x": 1}, synth.producer(), "p" * 32, synth.t(1))
    R.write_records(os.path.join(w.dir, R.record_file_name(2, "run_degeneration", gz=True)), [rec])
    p3 = w.emit_patch(fields={"gpu_count": 1})
    assert _seq_of(p3) == 3, "number 2 is taken by the .gz twin"
    # a plain colliding file at the very name the writer would take next (same type): skipped, the next number is taken
    R.write_records(os.path.join(w.dir, R.record_file_name(4, "run_state")), [rec])
    p5 = w.emit_state("generated", "completion_artifact")
    assert _seq_of(p5) == 5 and os.path.basename(p5) == R.record_file_name(5, "run_state")
    assert sorted(os.listdir(w.dir)) == sorted([os.path.basename(p1), R.record_file_name(2, "run_degeneration", gz=True), os.path.basename(p3),
                                                R.record_file_name(4, "run_state"), os.path.basename(p5)])


def test_the_cached_census_regression_a_file_that_appears_after_the_first_write_is_not_re_issued(tmp_path):
    """The shape measured on 2026-10-06: a writer built, its first write done, then a long phase during which `000002-...jsonl.gz` arrives
    (another pass, the mirror, the harvester's rename); the next write must take 3, never 2 again."""
    w = _writer(tmp_path)
    p1 = w.emit_state("running", "job_log")
    assert _seq_of(p1) == 1 and w._seq == 1
    rec = R.make_record("run_degeneration", {"x": 1}, synth.producer(), "p" * 32, synth.t(1))
    twin = os.path.join(w.dir, R.record_file_name(2, "run_degeneration", gz=True))
    R.write_records(twin, [rec])
    p = w.write_many("run_degeneration", [{"x": 2}])
    assert _seq_of(p) == 3 and p.endswith(R.record_file_name(3, "run_degeneration")), os.listdir(w.dir)
    assert os.path.exists(twin) and R.logical_uri(p) != R.logical_uri(twin)
    # a writer BUILT before any file existed and first used after the directory filled up starts above everything in it
    late = _writer(tmp_path)
    for n in (4, 5):
        R.write_records(os.path.join(w.dir, R.record_file_name(n, "run_state")), [rec])
    assert _seq_of(late.emit_state("running", "job_log")) == 6


def test_the_number_never_goes_down_within_a_writer_even_when_the_directory_empties(tmp_path):
    w = _writer(tmp_path)
    p1, p2 = w.emit_state("running", "job_log"), w.emit_state("generated", "completion_artifact")
    os.remove(p1)
    os.remove(p2)                                        # the directory census now says 0; the writer issued 2
    assert _seq_of(w.emit_patch(fields={"gpu_count": 1})) == 3


def test_write_many_keeps_one_file_many_records_one_stamp_and_the_disabled_writer_writes_nothing(tmp_path):
    w = _writer(tmp_path)
    path = w.write_many("run_degeneration", [{"x": 1}, {"x": 2}, {"x": 3}], written_utc=synth.t(7))
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    assert len(recs) == 3 and {r["written_utc"] for r in recs} == {synth.t(7)} and [r["payload"]["x"] for r in recs] == [1, 2, 3]
    assert w.written == [path] and _seq_of(path) == 1
    off = emit.RecordWriter(str(tmp_path / "other"), "c", RUN, synth.producer(), disabled=True)
    assert off.write_many("run_degeneration", [{"x": 1}]) is None and off.written == [] and not os.path.exists(str(tmp_path / "other"))


def test_the_retry_is_bounded_and_the_error_names_the_directory_and_the_last_path(tmp_path, monkeypatch):
    w = _writer(tmp_path)
    attempts = []

    def always_taken(path, records, gz=False, exclusive=False):
        attempts.append(path)
        raise FileExistsError(path)
    monkeypatch.setattr(emit, "write_records", always_taken)
    with pytest.raises(RuntimeError, match="could not create a record file under .* in 50 attempts \\(last tried .*000050-run_state.jsonl\\)"):
        w.emit_state("running", "job_log")
    assert len(attempts) == emit.RecordWriter.SEQ_ATTEMPTS == 50
    assert [_seq_of(p) for p in attempts] == list(range(1, 51)), "each attempt re-lists and moves one number up"
    assert w.written == []
