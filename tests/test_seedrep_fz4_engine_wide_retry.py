"""fz4: the ENGINE-WIDE identity check attributes an attempt row through its REQUEST RECORD, so a
503-then-retry on a ledger-less leg is a fully bound completion, not a spurious `<none>` partition.

Known-bad at 59765e1db: `request_evidence_audit.partitions` keyed every attempt row by its OWN (task, vmodel). The response
hook's record of a FAILED attempt carries `vmodel=None` (no body to read `model` off), so the failed request landed in a
`<none> @ <none>` partition; the engine-wide reconciliation then reported it missing from the real model's inventory and
`audit_root` forced every affected store verdict REFUSED — although the retry produced a valid, fully bound completion
(`store_partition` already attributed through the record; the engine-wide function did not).

Green: a response row's partition = its REQUEST RECORD's (task, vmodel); a response naming no record is still refused as an
orphan; contradictory labels are still refused (inside the record's partition). The COMPLETE ledger-less `audit_root` regression:
failed attempt → successful retry AGREES with `reconciliation_contract=…/2-D106`; the same root with the retry's completion
missing from the engine log still refuses. Synthetic roots only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402

CONTRACT_LEDGERLESS = "seedrep_request_evidence_reconciliation/2-D106"
FAILED_RID = "f" * 32


def _audit(ar, tmp_path: Path, *, tag="a"):
    return rea.audit_root(ar, out_dir=tmp_path / f"audits_{tag}", legs=["pasttense"], policy_file=None,
                          thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=list(fx.PETRI_BEHAVIOURS),
                          model_key=ac.MODEL_KEY, client_status="absent", siblings=[])


def _ledgerless_root_with_retry(tmp_path: Path):
    """A static (pasttense) root the Modal way — request + response evidence + the engine log, NO usage ledger — where the first
    attempt of one draw FAILED (503, fresh request id, the response hook's row with `completion_id=None`, `vmodel=None`) and the
    recorded success is its retry. The failed attempt has its own request record (the client records every attempt it sends)."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=3)
    (root.path / ac.LEDGER_FILE).unlink()
    ep, rp = root.path / ac.EVIDENCE_FILE, root.path / ac.RESPONSE_EVIDENCE_FILE
    recs, resp = ac.read_jsonl(ep), ac.read_jsonl(rp)
    success = next(r for r in resp if r["completion_id"])
    rec0 = next(r for r in recs if r["request_id"] == success["request_id"])
    failed_rec = {**rec0, "request_id": FAILED_RID}
    failed_resp = {**success, "request_id": FAILED_RID, "completion_id": None, "status": 503, "vmodel": None, "finish_reason": None,
                   ac.RESPONSE_TEXT_SHA_FIELD: None, ac.RESPONSE_TEXT_SHA_OF_FIELD: None, "error": "HTTP 503: upstream busy"}
    ep.write_text("".join(json.dumps(r) + "\n" for r in [failed_rec] + recs))   # the failed attempt was sent FIRST
    rp.write_text("".join(json.dumps(r) + "\n" for r in [failed_resp] + resp))
    return root, ac.read_attempt_root(root.path), success


def test_known_bad_failed_attempt_partitioned_by_its_record_not_its_own_none_vmodel(tmp_path: Path):
    root, ar, success = _ledgerless_root_with_retry(tmp_path)
    ev = rea.load_evidence(ar)
    assert ev.ledger == ac.LEDGER_NOT_ON_PATH and ev.attempt_source == "response_evidence"
    parts = rea.partitions(ev.records, ev.attempt_rows)
    # the 59765e1db shape: a `<none> @ <none>` partition holding the failed row — gone; it sits with its record
    assert all("<none>" not in k for k in parts)
    (task_key,) = [k for k in parts if k != ("probe", "*")]
    assert task_key == (ac.static_task("pasttense"), success["vmodel"])
    recs, rows = parts[task_key]
    assert {r["request_id"] for r in rows} >= {FAILED_RID, success["request_id"]} and len(recs) == len(rows)
    problems = rea.engine_wide_problems(ev.records, ev.attempt_rows, thinking_expect_of_task={},
                                        default_thinking=ac.THINKING_EXPECT_DEFAULT.get("pasttense"),  # the static legs: no switch (None)
                                        attempt_source=ev.attempt_source)
    assert all(v is None for v in problems.values()), problems


def test_complete_ledgerless_audit_failed_attempt_then_retry_agrees_qualified(tmp_path: Path):
    root, ar, success = _ledgerless_root_with_retry(tmp_path)
    (v,) = _audit(ar, tmp_path)
    assert v["verdict"] == "AGREES", v["problem"]
    assert v["checks"]["reconciliation"] is None and v["engine_wide_identity"] and all(x is None for x in v["engine_wide_identity"].values())
    assert v["checks"]["row_binding"]["n_bound"] == v["n_draws"] == 6 and v["n_draws_bound"] == 6
    # the failed attempt is IN the inventory (7 attempts, 6 completions), never a completion
    assert v["n_attempt_rows_partition"] == 7 and v["n_records_task"] == 7 and v["n_ledger_rows_partition"] is None
    assert v["checks"]["row_binding"]["n_authoritative_completion_ids"] == 6
    # the qualification rides on the AGREES
    assert v["ledger"] == "not_on_path" and v["attempt_source"] == "response_evidence" and v["reconciliation_contract"] == CONTRACT_LEDGERLESS
    assert v["body_digest_check"].startswith("not_available")


def test_same_root_with_the_retrys_completion_missing_from_the_engine_log_refuses(tmp_path: Path):
    root, ar, success = _ledgerless_root_with_retry(tmp_path)
    log = root.serve / "vllm_server.log"
    log.write_text("\n".join(l for l in log.read_text().splitlines() if success["completion_id"] not in l) + "\n")
    (v,) = _audit(ar, tmp_path)
    assert v["verdict"] == "REFUSED"
    assert f"completion_id {success['completion_id']!r} matches 0 engine-log `Received request` line(s)" in v["problem"]
    assert v["checks"]["row_binding"]["n_bound"] == 5 and v["ledger"] == "not_on_path" and v["reconciliation_contract"] == CONTRACT_LEDGERLESS


def test_orphan_response_and_contradictory_record_still_refuse(tmp_path: Path):
    root, ar, success = _ledgerless_root_with_retry(tmp_path)
    ep, rp = root.path / ac.EVIDENCE_FILE, root.path / ac.RESPONSE_EVIDENCE_FILE
    # (a) a response row naming NO record (a failed attempt the client never recorded): refused as an orphan, in the <none> partition
    orphan = {**json.loads(rp.read_text().splitlines()[0]), "request_id": "0" * 32}
    rp.write_text(rp.read_text() + json.dumps(orphan) + "\n")
    (v,) = _audit(ar, tmp_path, tag="a")
    assert v["verdict"] == "REFUSED" and "engine-wide identity" in v["problem"]
    assert "attempted call(s) in the response evidence have NO request-evidence record" in v["problem"] and "'0" in v["problem"]
    # the orphan kept the failed row's task but no vmodel: it is judged where ITS labels put it — `static/pasttense @ <none>` — and refused there
    assert any("<none>" in k for k in v["engine_wide_identity"]), sorted(v["engine_wide_identity"])
    rp.write_text("".join(l + "\n" for l in rp.read_text().splitlines()[:-1]))
    # (b) the failed attempt's record and its response row disagree on the sample: refused by the label clause, inside the record's partition
    recs = ac.read_jsonl(ep)
    recs = [{**r, "sample": "g9/9"} if r["request_id"] == FAILED_RID else r for r in recs]
    ep.write_text("".join(json.dumps(r) + "\n" for r in recs))
    (v,) = _audit(ar, tmp_path, tag="b")
    assert v["verdict"] == "REFUSED" and f"request_id {FAILED_RID!r}: the response evidence labels the attempt task" in v["problem"]
    assert not any("<none>" in k for k in v["engine_wide_identity"])  # attributed through its record: no <none> partition
