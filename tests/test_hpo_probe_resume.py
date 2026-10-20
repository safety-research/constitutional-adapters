"""scripts/sdfbase/hpo_probe_resume.py — the resume of an INCOMPLETE serving-probe record is DERIVED from the record:
positionals 4-6 from probe_plan.json, the attempt counter and the cap from the carrier's probe_attempt.json (else its per-exit
RESULT head, the record form before that file existed), the width from the last attempt (or --width), an attempt number the
front door's ledger already issued skipped — and refused at the cap, on a plan / attempt-record disagreement, on a record with
no attempt record, on an unrecorded width. Synthetic records only (placeholder shas, TESTHOST)."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase", REPO / "reproduce" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import hpo_probe_resume as hpr  # noqa: E402
import stage as ST  # noqa: E402

SETS = ["wo", "qkvo", "attn_mlp", "mlp", "down"]
PLAN = {"version": 1, "model_key": "qwen35_4b_demo", "model_id": "org/toy", "served_hf_id": "org/toy", "revision": "0" * 40,
        "sets": SETS, "ranks": [4, 16, 64], "tp": 1, "n_launches": 7, "launches": []}
ATTEMPT = {"version": 1, "written_utc": "2026-09-30T00:00:00Z", "attempt": 2, "max_attempts": 4, "gpus": 4, "parallel": "auto",
           "mode": "tp1", "tp": 1, "sets": ",".join(SETS), "ranks": "4,16,64", "job_id": "11111111", "job_name": "hpo-m-lora-probe",
           "sha": "a" * 40, "host": "TESTHOST"}
LEGACY = "state=INTERRUPTED exit=143 attempt=1/4 elapsed=7585 host=TESTHOST model=qwen35_4b_demo mode=tp1 job=22222222 verdict=none"
LWID = "qwen35_4b_demo/03_hpo/lora/hpo.lora.probe"


def _derive(**kw):
    args = dict(attempt_json=json.dumps(ATTEMPT), legacy_line="", width="record", ledger=None, lwid_prefix=LWID)
    args.update(kw)
    return hpr.derive(PLAN, **args)


def test_the_resume_is_attempt_plus_one_of_the_record_with_the_plans_positionals_and_the_last_width():
    spec = _derive()
    assert spec["attempt"] == 3 and spec["max_attempts"] == 4 and spec["unit_suffix"] == "a3"
    assert spec["sets"] == "wo,qkvo,attn_mlp,mlp,down" and spec["ranks"] == "4,16,64" and spec["mode"] == "tp1"
    assert spec["width"] == "4" and spec["parallel"] == "auto"          # the record's last width; the driver derives the concurrency
    assert spec["source"] == "probe_attempt.json" and spec["last_job_id"] == "11111111" and spec["notes"] == []
    # an explicit width wins; auto means the header's
    assert _derive(width="2")["width"] == "2" and _derive(width="auto")["width"] == "auto"


def test_the_cap_refuses_and_names_it():
    with pytest.raises(hpr.Refused, match=r"attempt cap 4 is reached \(last attempt 4, job 11111111; next would be 5\)"):
        _derive(attempt_json=json.dumps({**ATTEMPT, "attempt": 4}))
    assert _derive(attempt_json=json.dumps({**ATTEMPT, "attempt": 3}))["attempt"] == 4     # the last attempt under the cap runs


def test_a_record_before_the_attempt_record_reads_the_carriers_result_head_and_needs_an_explicit_width():
    with pytest.raises(hpr.Refused, match="no recorded width.*--probe-width"):
        _derive(attempt_json="", legacy_line=LEGACY)
    spec = _derive(attempt_json="", legacy_line=LEGACY, width="4")
    assert spec["attempt"] == 2 and spec["max_attempts"] == 4 and spec["width"] == "4" and spec["source"] == hpr.SOURCE_LEGACY_RESULT
    assert spec["last_job_id"] == "22222222" and any("attempt=1/4 job=22222222" in n for n in spec["notes"])
    assert spec["mode"] == "tp1" and spec["sets"] == ",".join(SETS)          # positionals from the PLAN, the head names none
    assert hpr.parse_legacy_head("verdict=none") is None and hpr.parse_legacy_head("") is None


def test_no_attempt_record_of_either_form_refuses_by_name():
    with pytest.raises(hpr.Refused, match="no probe_attempt.json.*no `attempt=N/M` head"):
        _derive(attempt_json="", legacy_line="")
    with pytest.raises(hpr.Refused, match="not JSON"):
        _derive(attempt_json="{not json")
    with pytest.raises(hpr.Refused, match="attempt is not an integer"):
        _derive(attempt_json=json.dumps({**ATTEMPT, "attempt": "two"}))


def test_the_plan_and_the_attempt_record_must_agree_on_positionals_4_to_6():
    with pytest.raises(hpr.Refused, match="disagree.*sets wo \\(attempt record\\) != wo,qkvo,attn_mlp,mlp,down \\(plan\\)"):
        _derive(attempt_json=json.dumps({**ATTEMPT, "sets": "wo"}))
    with pytest.raises(hpr.Refused, match="mode tp2 \\(attempt record\\) != tp1 \\(plan\\)"):
        _derive(attempt_json=json.dumps({**ATTEMPT, "mode": "tp2"}))
    with pytest.raises(hpr.Refused, match="tp1 \\| tp2 only"):
        hpr.plan_positionals({**PLAN, "tp": 4})
    with pytest.raises(hpr.Refused, match="names no module sets"):
        hpr.plan_positionals({**PLAN, "sets": []})
    assert hpr.plan_positionals({**PLAN, "tp": 2}) == (",".join(SETS), "4,16,64", "tp2")


def test_an_attempt_number_the_ledger_already_issued_is_skipped_with_a_note(tmp_path):
    """The front door bound `<lwid>.a3` as `submitted` (a job that left the queue before its carrier wrote the attempt record):
    the derivation takes a4 and says why; a `registered` binding is skipped the same way; an unbound number is used."""
    ledger = tmp_path / "submissions.jsonl"; ledger.write_text("")
    ST.write_binding(str(ledger), ST.submitted_record(f"{LWID}.a3", "33333333", "cluster_a", f"{LWID}.a3@cluster_a@1", "hpo-m-lora-probe"))
    spec = _derive(ledger=str(ledger))
    assert spec["attempt"] == 4 and spec["unit_suffix"] == "a4"
    assert len(spec["notes"]) == 1 and "already binds attempt 3" in spec["notes"][0] and "taking attempt 4" in spec["notes"][0]
    # the skip walks up to the cap and then refuses there
    ST.write_binding(str(ledger), ST.submitted_record(f"{LWID}.a4", "44444444", "cluster_a", f"{LWID}.a4@cluster_a@1", "hpo-m-lora-probe"))
    with pytest.raises(hpr.Refused, match="attempt cap 4 is reached"):
        _derive(ledger=str(ledger))
    # no ledger / no prefix -> no skipping
    assert _derive(ledger=None)["attempt"] == 3 and _derive(ledger=str(ledger), lwid_prefix=None)["attempt"] == 3


def test_the_cli_prints_key_value_lines_the_shell_reads_and_exits_2_on_a_refusal(tmp_path):
    plan = tmp_path / "probe_plan.json"; plan.write_text(json.dumps(PLAN))
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/hpo_probe_resume.py"), "--plan", str(plan),
                        "--attempt-json", json.dumps(ATTEMPT), "--width", "record", "--lwid-prefix", LWID], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    kv = dict(ln.split("=", 1) for ln in r.stdout.splitlines() if ln and not ln.startswith("note="))
    assert kv["attempt"] == "3" and kv["max_attempts"] == "4" and kv["sets"] == ",".join(SETS) and kv["ranks"] == "4,16,64"
    assert kv["mode"] == "tp1" and kv["width"] == "4" and kv["parallel"] == "auto" and kv["unit_suffix"] == "a3" and kv["source"] == "probe_attempt.json"
    # stdin plan, a refusal -> exit 2, REFUSED on stderr, nothing on stdout
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/hpo_probe_resume.py"), "--plan", "-",
                        "--attempt-json", json.dumps({**ATTEMPT, "attempt": 4})], input=json.dumps(PLAN), capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout == "" and "REFUSED: the record's attempt cap 4 is reached" in r.stderr
    assert hpr.main(["--plan", "-", "--attempt-json", ""], stdin=io.StringIO("{not json")) == 2
