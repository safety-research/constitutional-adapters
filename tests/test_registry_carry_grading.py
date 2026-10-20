"""A retired run's GRADING records travel to its successor WITHOUT recompute —
scripts/sdfbase/full22/registry_carry_grading.py re-files the judgment, item_result, run_metric and run_degeneration records (and the grader's
`run_state judged` + post-judgment run_patch) under the successor id: payloads EQUAL except the run_id they bind and the judgment_id, which takes the
identity form for a NON-claimed judgment of the successor (identity.judgment_id(successor, judge_id)); the new judgment carries
carried_from_run_id / carried_from_judgment_id / reason / decision in extra. Refuses by name without exactly one complete judgment, without the
closer's `superseded` lineage, without the successor's run record; idempotent; dry by default. Synthetic records at the real envelope shape
(plain and gz files), placeholder ids, hostname TESTHOST."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sisteer.registry import identity as I, synth  # noqa: E402

CARRIER = REPO / "scripts" / "sdfbase" / "full22" / "registry_carry_grading.py"
CAMPAIGN = "sdfbase_full22"
OLD = "sdfbase_full22:pasttense:111111111111:aaaaaaaaaaaaaaaa"
NEW = "sdfbase_full22:pasttense:111111111111:bbbbbbbbbbbbbbbb"
JUDGE = "2" * 64
OLD_JID = "e7ae0d59-c95e-5679-8868-000000000001"
PRODUCER = {"script": "test", "code_sha": synth.code_sha("c"), "host": "TESTHOST"}   # a non-placeholder synthetic sha (the merged validator refuses c*40)


def _env(payload: dict, rtype: str, seq: int) -> dict:
    return {"record_type": rtype, "payload": payload, "producer": PRODUCER, "record_id": f"r{seq:06d}", "schema_version": 1, "written_utc": f"2026-09-30T20:00:{seq:02d}.000Z"}


def _write(d: Path, seq: int, rtype: str, payloads: list, gz: bool = False) -> None:
    d.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(_env(p, rtype, seq)) + "\n" for p in payloads)
    p = d / f"{seq:06d}-{rtype}.jsonl{'.gz' if gz else ''}"
    (gzip.open(p, "wt", encoding="utf-8") if gz else open(p, "w", encoding="utf-8")).write(text) if not gz else None
    if gz:
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            fh.write(text)


def _old_run_dir(root: Path, *, superseded_by: str | None = NEW, judgments: int = 1) -> Path:
    d = root / CAMPAIGN / I.run_dir_name(OLD)
    _write(d, 1, "run", [{"run_id": OLD, "campaign": CAMPAIGN, "lifecycle_state": "running"}])
    _write(d, 5, "run_state", [{"run_id": OLD, "state": "generated", "event_utc": "2026-09-30T16:49:53Z", "evidence_kind": "completion_artifact", "evidence_uri": "/s/shard00/progress.jsonl"}])
    _write(d, 6, "item_result", [{"run_id": OLD, "judgment_id": OLD_JID, "item_id": f"g{i}", "metric_id": "strongreject_score", "trial_index": 0, "sample_index": 0, "turn_index": 0,
                                  "value": 0.5 * i, "label": "x", "judged_utc": "2026-09-30T20:00:00Z", "created_utc": "2026-09-30T20:00:00Z", "api_pool": "lowprio",
                                  "extra": {"run_overrides": {"id": "synthetic_v1"}}} for i in range(3)], gz=True)
    _write(d, 7, "run_degeneration", [{"run_id": OLD, "judgment_id": OLD_JID, "measure": "at_token_cap_finish_reason", "scope": "all", "numerator": 1, "denominator": 3, "rate": 0.333,
                                       "measurement_state": "measured", "definition_version": "x@abc", "extra": {"run_overrides": {"id": "synthetic_v1"}}}])
    _write(d, 8, "run_metric", [{"run_id": OLD, "judgment_id": OLD_JID, "metric_id": "asr", "threshold": t, "value": v, "n": 3, "computed_by": "producer",
                                 "computed_utc": "2026-09-30T20:07:03Z", "extra": {"run_overrides": {"id": "synthetic_v1"}}} for t, v in ((0.25, 0.33), (0.5, 0.0))])
    js = [{"run_id": OLD, "judgment_id": OLD_JID, "judge_id": JUDGE, "started_utc": "2026-09-30T17:33:57Z", "finished_utc": "2026-09-30T20:07:03Z", "n_items_judged": 3,
           "n_judge_refusals": 0, "n_errors": 0, "api_pool": "lowprio", "grading_code_sha": "b" * 40, "store_uri": "/s", "status": "complete",
           "claimed_by_session": "nobody@TESTHOST", "claimed_utc": "2026-09-30T17:33:57Z", "producer_session": "nobody@TESTHOST", "extra": {"run_overrides": {"id": "synthetic_v1"}}}]
    if judgments == 2:
        js.append(dict(js[0], judgment_id="e7ae0d59-c95e-5679-8868-000000000002"))
    _write(d, 9, "judgment", js)
    _write(d, 11, "run_state", [{"run_id": OLD, "state": "judged", "event_utc": "2026-09-30T20:07:03Z", "evidence_kind": "completion_artifact", "evidence_uri": "/s", "extra": {"run_overrides": {"id": "synthetic_v1"}}}])
    _write(d, 12, "run_patch", [{"run_id": OLD, "fields": {"judge_id": JUDGE, "judged_utc": "2026-09-30T20:07:03Z", "n_items_judged": 3}, "extra": {"run_overrides": {"id": "synthetic_v1"}}}])
    if superseded_by:
        _write(d, 13, "run_state", [{"run_id": OLD, "state": "superseded", "event_utc": "2026-10-04T02:00:00Z", "evidence_kind": "completion_artifact", "evidence_uri": "/s/shard00/progress.jsonl",
                                     "extra": {"superseded_by_run_id": superseded_by, "reason": "token_cap_identity_relabel", "decision": "D1900"}}])
    return d


def _new_run_dir(spool: Path) -> Path:
    d = spool / CAMPAIGN / I.run_dir_name(NEW)
    _write(d, 1, "run", [{"run_id": NEW, "campaign": CAMPAIGN, "lifecycle_state": "running"}])
    return d


def _records(d: Path) -> list:
    out = []
    for f in sorted(d.glob("0*.jsonl*")):
        opener = gzip.open if f.name.endswith(".gz") else open
        with opener(f, "rt", encoding="utf-8") as fh:
            for ln in fh:
                rec = json.loads(ln); out.append((f.name.split("-", 1)[1].split(".")[0], rec["payload"]))
    return out


def _run(pairs: Path, root: Path, spool: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {"PATH": __import__("os").environ["PATH"], "HOME": str(root.parent), "PYTHONPATH": str(REPO / "src")}
    return subprocess.run([sys.executable, str(CARRIER), "--pairs", str(pairs), "--records-root", str(root), "--spool", str(spool), "--campaign", CAMPAIGN, "--code-sha", "d" * 40, *extra],
                          capture_output=True, text=True, env=env, timeout=120)


def _strip(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in ("run_id", "judgment_id")}


def test_the_grading_records_are_carried_equal_except_the_ids_with_provenance_on_the_new_judgment_idempotently(tmp_path):
    root = tmp_path / "records"; spool = tmp_path / "spool"
    _old_run_dir(root); _new_run_dir(spool)
    pairs = tmp_path / "pairs.tsv"; pairs.write_text("# header\n" + f"{OLD}\t{NEW}\tpasttense\tnone\tt\t0\t808182\tkey\n" + "cancelled\tsdfbase_full22:x:y:z\n")
    dry = _run(pairs, root, spool)
    assert dry.returncode == 0 and "DRY" in dry.stderr and "would carry" in dry.stderr, dry.stderr
    assert len(list((spool / CAMPAIGN / I.run_dir_name(NEW)).iterdir())) == 1                       # dry: nothing written
    r = _run(pairs, root, spool, "--write")
    assert r.returncode == 0 and "carried" in r.stderr, r.stderr
    old = _records(root / CAMPAIGN / I.run_dir_name(OLD)); new = _records(spool / CAMPAIGN / I.run_dir_name(NEW))
    new_jid = I.judgment_id(NEW, JUDGE)
    assert new_jid != OLD_JID
    by_type_old = {}; by_type_new = {}
    for t, p in old:
        by_type_old.setdefault(t, []).append(p)
    for t, p in new:
        by_type_new.setdefault(t, []).append(p)
    # every carried type, the same count; item_result / run_metric / run_degeneration payloads equal but the two ids
    for t in ("item_result", "run_metric", "run_degeneration"):
        assert len(by_type_new[t]) == len(by_type_old[t]), t
        for a, b in zip(by_type_old[t], by_type_new[t]):
            assert _strip(a) == _strip(b) and b["run_id"] == NEW and b["judgment_id"] == new_jid, t
    # the judgment: a NEW record in the non-claimed identity form, provenance in extra, claim fields gone, everything else equal
    (jn,) = by_type_new["judgment"]; (jo,) = by_type_old["judgment"]
    assert jn["judgment_id"] == new_jid and jn["run_id"] == NEW and jn["judge_id"] == JUDGE and jn["status"] == "complete"
    assert jn["extra"]["carried_from_run_id"] == OLD and jn["extra"]["carried_from_judgment_id"] == OLD_JID
    assert jn["extra"]["reason"] == "token_cap_identity_relabel" and jn["extra"]["decision"] == "D1900" and jn["extra"]["run_overrides"] == jo["extra"]["run_overrides"]
    assert "claimed_by_session" not in jn and "claimed_utc" not in jn
    assert {k: v for k, v in _strip(jo).items() if k not in ("extra", "claimed_by_session", "claimed_utc")} == {k: v for k, v in _strip(jn).items() if k != "extra"}
    # the grader's judged state + post-judgment patch travel; the job's own lifecycle records (running / generated / superseded) do not
    states = [p["state"] for p in by_type_new.get("run_state", [])]
    assert states == ["judged"] and by_type_new["run_patch"][0]["fields"] == {"judge_id": JUDGE, "judged_utc": "2026-09-30T20:07:03Z", "n_items_judged": 3}
    assert by_type_new["run"][0]["run_id"] == NEW and len(by_type_new["run"]) == 1
    # idempotent: a second --write writes nothing
    files = sorted(p.name for p in (spool / CAMPAIGN / I.run_dir_name(NEW)).iterdir())
    r2 = _run(pairs, root, spool, "--write")
    assert r2.returncode == 0 and "idempotent" in r2.stderr, r2.stderr
    assert sorted(p.name for p in (spool / CAMPAIGN / I.run_dir_name(NEW)).iterdir()) == files


def test_refusals_by_name_no_lineage_two_judgments_no_successor_run(tmp_path):
    pairs = tmp_path / "pairs.tsv"; pairs.write_text(f"{OLD}\t{NEW}\n")
    # the closer's lineage first: a retired run not filed superseded by THIS successor refuses
    root = tmp_path / "r1"; spool = tmp_path / "s1"; _old_run_dir(root, superseded_by=None); _new_run_dir(spool)
    r = _run(pairs, root, spool, "--write"); assert r.returncode == 2 and "not filed `superseded`" in r.stderr, r.stderr
    root = tmp_path / "r1b"; spool = tmp_path / "s1b"; _old_run_dir(root, superseded_by="sdfbase_full22:other:0:0"); _new_run_dir(spool)
    r = _run(pairs, root, spool, "--write"); assert r.returncode == 2 and "not filed `superseded`" in r.stderr
    # two complete judgments: a grading question first
    root = tmp_path / "r2"; spool = tmp_path / "s2"; _old_run_dir(root, judgments=2); _new_run_dir(spool)
    r = _run(pairs, root, spool, "--write"); assert r.returncode == 2 and "2 complete judgment(s)" in r.stderr, r.stderr
    # no successor run record
    root = tmp_path / "r3"; spool = tmp_path / "s3"; _old_run_dir(root); spool.mkdir()
    r = _run(pairs, root, spool, "--write"); assert r.returncode == 2 and "no `run` record" in r.stderr, r.stderr
    # no records at all for the retired run
    root = tmp_path / "r4"; spool = tmp_path / "s4"; root.mkdir(); _new_run_dir(spool)
    r = _run(pairs, root, spool, "--write"); assert r.returncode == 2 and "no records under" in r.stderr, r.stderr
    for s in (tmp_path / "s1", tmp_path / "s1b", tmp_path / "s2", tmp_path / "s4"):
        assert len(list((s / CAMPAIGN / I.run_dir_name(NEW)).iterdir())) == 1                     # nothing written on a refusal
