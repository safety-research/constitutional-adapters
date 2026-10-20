"""The run-overrides stamp is defaulted CENTRALLY in `emit.RecordWriter`:
every producer writer is constructed there -- `emit_cli.register_from_config`, `emit_cli.state`, `hooks.StoreRecorder`, `hooks.JudgingScope`
through the registry worklist -- so with `$SISTEER_RUN_OVERRIDES` set every record they write parks `{id, sha256, keys}` under
`extra_json.run_overrides`, and `json_extract(extra_json, '$.run_overrides.id')` reads the same id on the `run`, its
`run_state_event`s, the generation-time and judged `item_result` rows, the `judgment` and the `run_metric` rows. Driven through the REAL
writers against the REAL seeds, folded by the harvester; unset → no stamp; an unreadable file refuses."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from sisteer.registry import db as DB
from sisteer.registry import emit, emit_cli, hooks
from sisteer.registry import harvest as H
from sisteer.registry import synth
from sisteer.registry import worklist as W
from sisteer.registry.adapters import dimensions as D

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "configs" / "registry"
OVERRIDES_TEXT = ("# the pipeline test's deviations from the paper's defaults\n"
                  "OVERRIDE_ID=synthetic_pipeline_test_v1\n"
                  "export OVERRIDE_INIT_SCHEME=random_lecun_1e-3\n"
                  "OVERRIDE_HPO_WARMUP_FRAC='0.15'\n"
                  "NOT_AN_OVERRIDE=ignored\n")


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("reg") / "root")
    H.ensure_layout(root)
    D.write_seed_records(root, str(SEEDS), producer=emit.producer_block("tests.overrides_stamp", code_sha="a" * 40, substrate="grading_host"),
                         now="2026-09-16T12:00:00.000Z")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def _checkout(tmp_path: Path) -> dict:
    root = tmp_path / "checkout"
    (root / "configs").mkdir(parents=True)
    os.symlink(SEEDS, root / "configs" / "registry")
    cfg = root / "configs" / "synth_full22_qwen35_4b.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": ["none", "prompted__si_constitution"], "model": "Qwen/Qwen3.5-4B"}))
    (root / "configs" / "sdfbase_full22_roster.yaml").write_text(yaml.safe_dump({"targets": {"qwen35_4b": {"views": {"main": {"cells": {"prompted__si_constitution": None}}}}}}))
    (root / "configs" / "sdfbase_full22_pins.json").write_text("{}")
    return {"root": root, "cfg": cfg}


def test_stamp_parses_override_keys_only_and_hashes_the_file(tmp_path, monkeypatch):
    f = tmp_path / "pipeline_test.env"
    f.write_text(OVERRIDES_TEXT)
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    st = emit.run_overrides_stamp()
    assert st["id"] == "synthetic_pipeline_test_v1" and st["sha256"] == hashlib.sha256(OVERRIDES_TEXT.encode()).hexdigest()
    assert st["keys"] == {"OVERRIDE_ID": "synthetic_pipeline_test_v1", "OVERRIDE_INIT_SCHEME": "random_lecun_1e-3", "OVERRIDE_HPO_WARMUP_FRAC": "0.15"}
    assert st["path"] == "pipeline_test.env"      # the basename only: no absolute path enters a record
    # no OVERRIDE_ID: the file's stem names it
    g = tmp_path / "paper_defaults.env"
    g.write_text("# every knob at the paper's value\n")
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(g))
    assert emit.run_overrides_stamp()["id"] == "paper_defaults" and emit.run_overrides_stamp()["keys"] == {}
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, "")
    assert emit.run_overrides_stamp() is None
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(tmp_path / "missing.env"))
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)      # an ENABLED writer refuses an unreadable overrides file (a disabled one stamps nothing)
    with pytest.raises(OSError):
        emit.RecordWriter(str(tmp_path / "spool"), "c", "r", emit.producer_block("t", code_sha="a" * 40, substrate="grading_host"))


#: `KEY=value` lines with the shapes a shell reads differently from a naive split, each with the value `bash` gives it when the file is
#: `source`d (measured; the round-trip test below re-measures): an unquoted value with an inline comment; a '#' with no whitespace before
#: it (kept); a literal '#' inside single and double quotes (kept); a quoted empty value followed by a comment (the empty string); text
#: after a closing quote (appended); a tab before the '#'; a '#' kept inside the token with a comment after it; an `export` prefix; a
#: value that IS a '#'-led token; a double-quoted value with a comment after the closing quote; an EMPTY unquoted value followed by a
#: comment, with spaces and with a tab (the shell assigns the empty string; `L` is the control: no whitespace between `=` and `#` makes
#: the `#` literal)
ENV_PROBES = (("A", "64   # c", "64"), ("B", "64#notacomment", "64#notacomment"), ("C", "'64 # kept'", "64 # kept"), ("D", '"a # b"', "a # b"),
              ("E", "''   # c", ""), ("G", "'a'b", "ab"), ("H", "x\t# c", "x"), ("J", "v#x # trailing", "v#x"), ("K", "7 # e", "7"),
              ("L", "#literal", "#literal"), ("M", '"quoted"   # comment after double quotes', "quoted"),
              ("N", "   # empty means the default", ""), ("P", "\t# a tab, then the comment", ""))
ENV_PROBE_TEXT = "# a probe file\n" + "".join((f"export {k}={v}\n" if k == "K" else f"{k}={v}\n") for k, v, _want in ENV_PROBES) + "   # an indented comment line\n"


@pytest.mark.parametrize("key, raw, want", ENV_PROBES)
def test_env_value_reads_each_probe_as_the_shell_does(key, raw, want):
    assert emit.env_value(raw) == want, (key, raw)


def test_the_env_file_reader_equals_a_sourced_shell_on_the_probe_file(tmp_path):
    """The two readers of an overrides file -- the registry's stamp and the stage's `source` (reproduce/lib/common.sh) -- are pinned equal
    on every probe: what the run used is what the registry records. Before the fix the reader kept the whole right-hand side, so A read
    `64   # c`; the correction action (`registry reconcile env-comments`) restates the stored values by this same rule."""
    f = tmp_path / "probe.env"
    f.write_text(ENV_PROBE_TEXT)
    got = emit._parse_env_file(str(f))
    assert got == {k: want for k, _raw, want in ENV_PROBES}
    keys = " ".join(k for k, _r, _w in ENV_PROBES)
    sh = subprocess.run(["bash", "-c", f"set -a; source {f}; set +a; for k in {keys}; do printf '%s=%s\\n' \"$k\" \"${{!k}}\"; done"],
                        capture_output=True, text=True, check=True)
    sourced = dict(line.split("=", 1) for line in sh.stdout.splitlines())
    assert got == sourced, {k: (got.get(k), sourced.get(k)) for k in set(got) | set(sourced) if got.get(k) != sourced.get(k)}
    # the stamp carries the clean values and only the OVERRIDE_* keys; the digest is the file's as written (comments included)
    g = tmp_path / "overrides.env"
    text = "OVERRIDE_ID=probe_v1\nOVERRIDE_EFFECTIVE_BATCH=64                  # unchanged: the paper's\nOVERRIDE_LORA_RANK=''   # empty\nA=1 # x\n"
    g.write_text(text)
    st = emit.run_overrides_stamp(str(g))
    assert st["keys"] == {"OVERRIDE_ID": "probe_v1", "OVERRIDE_EFFECTIVE_BATCH": "64", "OVERRIDE_LORA_RANK": ""} and st["sha256"] == hashlib.sha256(text.encode()).hexdigest()


def _stub_grade(row, scope):
    scope.items([{"item_id": f"q{i:03d}", "metric_id": "asr", "value": float(i % 2), "label": "harmful" if i % 2 else "safe"} for i in range(4)])
    scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": 2, "n": 4}])


def test_every_producer_writer_stamps_the_overrides_and_the_fold_keeps_them(registry, tmp_path, monkeypatch, capsys):
    root = registry
    co = _checkout(tmp_path)
    f = tmp_path / "pipeline_test.env"
    f.write_text(OVERRIDES_TEXT)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    monkeypatch.setenv("REGISTRY_ROOT", root)
    spool = os.path.join(root, "records", "producer")
    pend_dir = tmp_path / "pending"
    # 1. job-start self-registration (the REAL writer common.sh calls) -> `run` records
    rc = emit_cli.register_from_config(str(co["cfg"]), model_key="qwen35_4b", evals=["agentic_murder"], serving_path="lora_native",
                                       campaign="sdfbase_full22", job_id="7001", pending_dir=str(pend_dir), job_name="f22-misalign-qwen35_4b",
                                       substrate="grading_host", code_sha=synth.code_sha("a"), spool=spool, push=False)
    assert rc == 0 and (pend_dir / "7001.json").exists()
    pending = json.loads((pend_dir / "7001.json").read_text())
    rids = [r["run_id"] for r in pending["runs"]]
    assert len(rids) == 2
    # 2. the job's lifecycle through emit_cli.state (running -> generated), the REAL writer of the job shell
    assert emit_cli.state("running", pending_dir=str(pend_dir), job="7001", node="TESTHOST", gpu_count=1, gpu_type="synthetic", spool=spool) == 0
    assert emit_cli.state("generated", pending_dir=str(pend_dir), job="7001", spool=spool) == 0
    # 3. a store recorder (the CellStore hook) files generation-time item rows on the routed run
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(pend_dir / "7001.json"))
    rec = hooks.store_recorder("none__store", {"arm_name": "none", "kind": "none", "sweep": "sdfbase_full22"}, spool=spool)
    assert rec is not None
    rec.on_generated("agentic_murder", ["s001", "s002"], "synthetic://gen")
    rec.flush()
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        for rid in rids:
            row = conn.execute("SELECT json_extract(extra_json, '$.run_overrides.id'), json_extract(extra_json, '$.run_overrides.sha256'), lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()
            assert row[0] == "synthetic_pipeline_test_v1" and row[1] == hashlib.sha256(OVERRIDES_TEXT.encode()).hexdigest() and row[2] == "generated", row
            evs = conn.execute("SELECT state, json_extract(extra_json, '$.run_overrides.id') FROM run_state_event WHERE run_id = ?", (rid,)).fetchall()
            assert {s for s, _ in evs} >= {"running", "generated"} and all(v == "synthetic_pipeline_test_v1" for _, v in evs), evs
        gen_items = conn.execute("SELECT json_extract(extra_json, '$.run_overrides.id') FROM item_result WHERE metric_id = 'generated'").fetchall()
        assert len(gen_items) == 2 and all(v[0] == "synthetic_pipeline_test_v1" for v in gen_items)
    finally:
        conn.close()
    # 4. a claimed and completed judgment through the registry worklist (hooks.JudgingScope under the claim)
    wl = W.RegistryWorklist(root, None, "session-stamp", "sdfbase_full22", driver="tests/stub_driver.py", api_pool="lowprio", run_ids=rids[:1])
    rows = wl.pending()
    assert [r["run_id"] for r in rows] == rids[:1], rows
    done = wl.grade(rows, _stub_grade)
    assert done == rids[:1]
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        j = conn.execute("SELECT status, json_extract(extra_json, '$.run_overrides.id') FROM judgment WHERE run_id = ?", (rids[0],)).fetchall()
        assert j and all(s == "complete" and v == "synthetic_pipeline_test_v1" for s, v in j), j
        items = conn.execute("SELECT json_extract(extra_json, '$.run_overrides.id') FROM item_result WHERE run_id = ? AND metric_id = 'asr'", (rids[0],)).fetchall()
        assert len(items) == 4 and all(v[0] == "synthetic_pipeline_test_v1" for v in items)
        m = conn.execute("SELECT json_extract(extra_json, '$.run_overrides.id') FROM run_metric WHERE run_id = ? AND metric_id = 'asr'", (rids[0],)).fetchone()
        assert m[0] == "synthetic_pipeline_test_v1"
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[0],)).fetchone()[0] == "judged"
        # every row of this run's records folded with the stamp; a paper-default run carries none
        assert conn.execute("SELECT COUNT(*) FROM run WHERE json_extract(extra_json, '$.run_overrides.id') IS NULL AND run_id IN (?, ?)", tuple(rids)).fetchone()[0] == 0
    finally:
        conn.close()


def test_no_overrides_file_means_no_stamp_and_an_explicit_payload_extra_wins(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(emit.RUN_OVERRIDES_ENV, raising=False)
    spool = str(tmp_path / "spool")
    w = emit.RecordWriter(spool, "c", "r", emit.producer_block("t", code_sha="a" * 40, substrate="grading_host"))
    assert w.payload_extra == {}
    p = w.write("run_state", {"run_id": "r", "state": "queued", "event_utc": "2026-01-01T00:00:00.000Z", "evidence_kind": "submit_wrapper"})
    from sisteer.registry import records as R
    (_i, rec, _sha), = list(R.read_record_file(p, compute_sha=False))
    assert "extra" not in rec["payload"] or "run_overrides" not in rec["payload"]["extra"]
    f = tmp_path / "o.env"
    f.write_text("OVERRIDE_ID=x\n")
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    w2 = emit.RecordWriter(spool, "c", "r2", emit.producer_block("t", code_sha="a" * 40, substrate="grading_host"),
                           payload_extra={"run_overrides": {"id": "explicit"}})
    assert w2.payload_extra["run_overrides"] == {"id": "explicit"}
    p2 = w2.write("run_state", {"run_id": "r2", "state": "queued", "event_utc": "2026-01-01T00:00:00.000Z", "evidence_kind": "submit_wrapper",
                                "extra": {"note": "kept"}})
    (_i, rec2, _sha), = list(R.read_record_file(p2, compute_sha=False))
    assert rec2["payload"]["extra"] == {"run_overrides": {"id": "explicit"}, "note": "kept"}


def test_a_run_patch_carries_the_writer_stamps_under_fields_extra_json_and_the_fold_keeps_them(tmp_path, monkeypatch):
    """The validator reads a `run_patch`'s `fields` and `verify` only, so the stamps `_stamped` parks under a top-level `extra`
    were dropped on every patch. `RecordWriter.write_many` now routes a patch's central stamps into `fields.extra_json`
    (`_stamped_patch`): the record carries NO top-level `extra`, `fields.extra_json.run_overrides` holds the stamp, a key the payload's own
    `fields.extra_json` already carries wins, `fields` is created for a verify-only patch -- and the fold keeps them, live and rebuilt.
    Every other kind still carries its stamps under `extra` (unchanged)."""
    from sisteer.registry import records as R
    from sisteer.registry import synth
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    f = tmp_path / "pipeline_test.env"
    f.write_text(OVERRIDES_TEXT)
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="running")
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    spool = os.path.join(root, "records", "producer")
    w = emit.RecordWriter(spool, synth.CAMPAIGN, rid, emit.producer_block("tests.d635", code_sha="a" * 40, substrate="grading_host"))
    stamp = w.payload_extra["run_overrides"]
    assert stamp["id"] == "synthetic_pipeline_test_v1"
    # 1. an ordinary fields patch: stamps under fields.extra_json, no top-level extra
    p1 = w.emit_patch(fields={"job_name": "patched"})
    (_i, r1, _s), = list(R.read_record_file(p1, compute_sha=False))
    assert "extra" not in r1["payload"]
    assert r1["payload"]["fields"] == {"job_name": "patched", "extra_json": {"run_overrides": stamp}}
    # 2. a verify-only patch: `fields` is created to carry the stamp
    p2 = w.emit_patch(verify={"serving_path": "lora_native"})
    (_i, r2, _s), = list(R.read_record_file(p2, compute_sha=False))
    assert "extra" not in r2["payload"] and r2["payload"]["verify"] == {"serving_path": "lora_native"}
    assert r2["payload"]["fields"] == {"extra_json": {"run_overrides": stamp}}
    # 3. the payload's own fields.extra_json keys win, others are kept beside the stamp
    p3 = w.emit_patch(fields={"extra_json": {"run_overrides": {"id": "explicit"}, "note": "kept"}, "notes": "third"})
    (_i, r3, _s), = list(R.read_record_file(p3, compute_sha=False))
    assert r3["payload"]["fields"] == {"notes": "third", "extra_json": {"run_overrides": {"id": "explicit"}, "note": "kept"}}
    # a run_state from the same writer still carries the stamp under a top-level extra (the other kinds are unchanged)
    p4 = w.write("run_state", {"run_id": rid, "state": "generated", "event_utc": emit.utc_now(), "evidence_kind": "completion_artifact"})
    (_i, r4, _s), = list(R.read_record_file(p4, compute_sha=False))
    assert r4["payload"]["extra"] == {"run_overrides": stamp} and "fields" not in r4["payload"]
    # every record of the three patches validates (the stamp is an ordinary parked extra)
    from sisteer.registry import validate as VAL
    for rec in (r1, r2, r3):
        v = VAL.validate_record(rec, "fixture", R.record_sha(rec))
        assert isinstance(v, VAL.Validated), v
        assert "run_overrides" in v.extra
    # the fold: the patches (stamped now, after the synthetic lifecycle) land on the run; the LAST patch's explicit id wins per key
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    live = DB.connect(DB.db_path(root), readonly=True)
    row = live.execute("SELECT job_name, notes, lifecycle_state, extra_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert (row[0], row[1], row[2]) == ("patched", "third", "generated")
    extra = json.loads(row[3])
    assert extra["run_overrides"] == {"id": "explicit"} and extra["note"] == "kept"
    assert extra["verified_fields"] == {"serving_path": "lora_native"}
    # the first patch's stamp is in the row's history (folded, then overwritten by the explicit one): the writer-stamped id folded
    hist = live.execute("SELECT json_extract(folded_state_json, '$.extra_json') FROM row_history WHERE table_name = 'run' AND record_sha256 = ?",
                        (R.record_sha(r1),)).fetchone()
    assert hist is not None and json.loads(hist[0])["run_overrides"]["id"] == "synthetic_pipeline_test_v1"
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = H.verify_equal(live, rebuilt)
    assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    sql = "SELECT job_name, notes, extra_json FROM run WHERE run_id = ?"
    assert tuple(live.execute(sql, (rid,)).fetchone()) == tuple(rebuilt.execute(sql, (rid,)).fetchone())
    rebuilt.close()
    live.close()
    # a writer with nothing to stamp leaves the patch exactly as given
    monkeypatch.delenv(emit.RUN_OVERRIDES_ENV, raising=False)
    w0 = emit.RecordWriter(str(tmp_path / "spool0"), "c", "r0", emit.producer_block("t", code_sha="a" * 40, substrate="grading_host"))
    assert w0.payload_extra == {}
    p0 = w0.emit_patch(fields={"job_name": "plain"})
    (_i, r0, _s), = list(R.read_record_file(p0, compute_sha=False))
    assert r0["payload"] == {"run_id": "r0", "fields": {"job_name": "plain"}}


def test_job_start_registration_stamps_the_attempt_handles(registry, tmp_path, monkeypatch):
    """The twin-submit election's handles: a job submitted under SISTEER_ATTEMPT_ID stamps `extra.attempt_id` and
    `extra.logical_work_id` (SISTEER_LOGICAL_WORK_ID, else `<attempts root>/<attempt>/logical_work_id`) on every run it registers and on
    its pending file; outside an attempt nothing is stamped."""
    root = registry
    co = _checkout(tmp_path)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(emit.RUN_OVERRIDES_ENV, raising=False)
    spool = os.path.join(root, "records", "attempts")
    attempts = tmp_path / "attempts"
    (attempts / "att-7").mkdir(parents=True)
    (attempts / "att-7" / "logical_work_id").write_text("qwen35_4b:u0:agentic_murder:shard0\n")
    monkeypatch.setenv(emit_cli.ATTEMPT_ID_ENV, "att-7")
    monkeypatch.setenv(emit_cli.ATTEMPTS_ROOT_ENV, str(attempts))
    monkeypatch.delenv(emit_cli.LOGICAL_WORK_ID_ENV, raising=False)
    assert emit_cli.attempt_stamp() == {"attempt_id": "att-7", "attempts_root": str(attempts), "logical_work_id": "qwen35_4b:u0:agentic_murder:shard0"}
    pend_dir = tmp_path / "pending_att"
    rc = emit_cli.register_from_config(str(co["cfg"]), model_key="qwen35_4b", evals=["agentic_murder"], serving_path="lora_native", campaign="sdfbase_full22",
                                       job_id="7101", pending_dir=str(pend_dir), job_name="f22-misalign-qwen35_4b", substrate="grading_host", code_sha=synth.code_sha("a"),
                                       spool=spool, push=False)
    assert rc == 0
    pending = json.loads((pend_dir / "7101.json").read_text())
    assert pending["attempt_id"] == "att-7" and pending["logical_work_id"] == "qwen35_4b:u0:agentic_murder:shard0"
    for r in pending["runs"]:
        assert r["plan"]["extra"]["attempt_id"] == "att-7" and r["plan"]["extra"]["logical_work_id"] == "qwen35_4b:u0:agentic_murder:shard0"
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        for r in pending["runs"]:
            row = conn.execute("SELECT json_extract(extra_json, '$.attempt_id'), json_extract(extra_json, '$.logical_work_id') FROM run WHERE run_id = ?", (r["run_id"],)).fetchone()
            assert tuple(row) == ("att-7", "qwen35_4b:u0:agentic_murder:shard0")
    finally:
        conn.close()
    # the explicit variable wins over the file; outside an attempt nothing is stamped
    monkeypatch.setenv(emit_cli.LOGICAL_WORK_ID_ENV, "explicit-unit")
    assert emit_cli.attempt_stamp()["logical_work_id"] == "explicit-unit"
    monkeypatch.delenv(emit_cli.ATTEMPT_ID_ENV)
    assert emit_cli.attempt_stamp() == {}


def test_a_run_patch_with_a_non_object_fields_extra_json_is_refused_at_the_writer_and_nothing_is_written(tmp_path, monkeypatch):
    """`_stamped_patch` handed a payload whose `fields.extra_json` was a serialized non-object (`'[]'`) or text that is not
    JSON on UNSTAMPED, trusting the validator to refuse it -- and the validator's merge dropped the value silently instead,
    so a producer forwarding serialized metadata or an operator's malformed correction changed the run row while losing its
    `run_overrides` / `cluster_label` stamps. The writer refuses the shape by name now (`ValueError`, nothing written,
    `written` unchanged), with stamps and without, and a `fields` that is not an object the same way; a serialized OBJECT
    is the payload's own keys (parsed, the stamps merged beneath them, validated whole) and an explicit null takes the stamps."""
    from sisteer.registry import records as R
    from sisteer.registry import validate as VAL
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    f = tmp_path / "pipeline_test.env"
    f.write_text(OVERRIDES_TEXT)
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    w = emit.RecordWriter(str(tmp_path / "spool"), "c", "r", emit.producer_block("t", code_sha="a" * 40, substrate="grading_host"))
    stamp = w.payload_extra["run_overrides"]
    for value in ("[]", "[1, 2]", "{not json", '"text"', "7", [], [1], 7, True):
        with pytest.raises(ValueError, match="fields.extra_json"):
            w.emit_patch(fields={"job_name": "unstamped", "extra_json": value})
    with pytest.raises(ValueError, match="fields must be an object"):
        w.write("run_patch", {"run_id": "r", "fields": ["job_name"]})
    assert w.written == []
    assert not os.path.isdir(w.dir) or not any("run_patch" in fn for fn in os.listdir(w.dir))
    # a serialized object: parsed, the payload's own keys win over the stamps, the record validates whole
    p = w.emit_patch(fields={"job_name": "stamped", "extra_json": '{"note": "kept", "run_overrides": {"id": "explicit"}}'})
    (_i, rec, _s), = list(R.read_record_file(p, compute_sha=False))
    assert rec["payload"]["fields"] == {"job_name": "stamped", "extra_json": {"run_overrides": {"id": "explicit"}, "note": "kept"}}
    v = VAL.validate_record(rec, "fixture", R.record_sha(rec))
    assert isinstance(v, VAL.Validated) and v.extra["run_overrides"] == {"id": "explicit"} and v.extra["note"] == "kept"
    # an explicit null takes the stamps (the fold's per-column statement becomes the writer's provenance)
    p2 = w.emit_patch(fields={"job_name": "nulled", "extra_json": None})
    (_i, rec2, _s), = list(R.read_record_file(p2, compute_sha=False))
    assert rec2["payload"]["fields"] == {"job_name": "nulled", "extra_json": {"run_overrides": stamp}}
    assert len(w.written) == 2
    # a writer with nothing to stamp refuses the same shapes: the validator would, and the record could only become a refusal
    monkeypatch.delenv(emit.RUN_OVERRIDES_ENV, raising=False)
    w0 = emit.RecordWriter(str(tmp_path / "spool0"), "c", "r0", emit.producer_block("t", code_sha="a" * 40, substrate="grading_host"))
    assert w0.payload_extra == {}
    for value in ("[]", "{not json", [], 7):
        with pytest.raises(ValueError, match="fields.extra_json"):
            w0.emit_patch(fields={"job_name": "unstamped", "extra_json": value})
    assert w0.written == []
    # and leaves a well-formed one exactly as given (a serialized object stays the producer's bytes)
    p0 = w0.emit_patch(fields={"job_name": "plain", "extra_json": '{"note": "kept"}'})
    (_i, r0, _s), = list(R.read_record_file(p0, compute_sha=False))
    assert r0["payload"] == {"run_id": "r0", "fields": {"job_name": "plain", "extra_json": '{"note": "kept"}'}}
