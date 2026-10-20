"""IFEval canonical battery — request evidence VERSION 3.

Through the PRODUCTION writer (the driver's ``_RequestEvidence.record_wire`` on
real wire bytes) → the JSONL on disk → the reader
(``contracts.request_evidence_reconciliation`` with the sampling / seed checks):
a Qwen body carrying presence_penalty 1.5 + the IFEval seed PASSES; an omitted
penalty, an altered penalty and a wrong seed each FAIL by name; a gemma body has
presence_penalty_present False; a version-2 record reads as "not recorded" and
can never pass; the request-id / digest join with the ledger is kept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tiers import inspect_served as isv
import served_capability_run as srun

IFEVAL = C.inspect_task(C.IFEVAL_TASK_NAME)
QWEN = C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY["qwen35_4b"]
GEMMA = C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY["gemma4e4b"]
TARGET = "petri__constv2"
GRADER = "petri__none"


def _wire(body: Dict[str, Any], **headers: str) -> Tuple[bytes, Dict[str, str]]:
    return json.dumps(body, separators=(",", ":")).encode("utf-8"), dict(headers)


def _qwen_body(item: str, trial: int, *, seed=None, presence_penalty=1.5, omit_penalty=False,
               vmodel=TARGET, enable_thinking=False) -> Dict[str, Any]:
    """The body the provider puts on the wire for an IFEval request on the Qwen
    vendor_preset row: temperature / top_p / presence_penalty / seed top-level
    (GenerateConfig → openai_completion_params), top_k / min_p /
    repetition_penalty merged from extra_body."""
    body = {"model": vmodel, "messages": [{"role": "user", "content": f"prompt {item}"}],
            "max_tokens": 2048, "temperature": 0.7, "top_p": 0.8,
            "seed": isv.ifeval_sample_seed(item, trial) if seed is None else seed,
            "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
            **C.chat_template_kwargs(C.THINKING_OFF if not enable_thinking else C.THINKING_ON)}
    if not omit_penalty:
        body["presence_penalty"] = presence_penalty
    return body


def _gemma_body(item: str, trial: int) -> Dict[str, Any]:
    return {"model": TARGET, "messages": [{"role": "user", "content": f"prompt {item}"}],
            "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
            "seed": isv.ifeval_sample_seed(item, trial), **C.chat_template_kwargs(C.THINKING_OFF)}


def _headers(i: int, item: str, trial: int, *, task: str = IFEVAL) -> Dict[str, str]:
    return {C.HEADER_REQUEST_ID: f"rid{i:04d}", C.HEADER_TASK: task,
            C.HEADER_SAMPLE: C.sample_header_value(item, trial)}


def _ledger_row(raw: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
    row = dict.fromkeys(C.LEDGER_ROW_FIELDS)
    row.update(task=headers.get(C.HEADER_TASK), sample=headers.get(C.HEADER_SAMPLE),
               request_id=headers[C.HEADER_REQUEST_ID], body_sha256=C.wire_body_sha256(raw),
               rejected=False, status=200)
    return row


def _write_and_read(tmp_path: Path, attempts):
    """The PRODUCTION writer on the wire bytes, then the JSONL read back."""
    ev = srun._RequestEvidence(tmp_path / C.REQUEST_EVIDENCE_FILENAME)
    rows = []
    for body, headers in attempts:
        raw, hdr = _wire(body, **headers)
        ev.record_wire(raw, headers=hdr)
        rows.append(_ledger_row(raw, hdr))
    text = (tmp_path / C.REQUEST_EVIDENCE_FILENAME).read_text()
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(records) == len(attempts) == ev.n
    return records, rows


def _reconcile(records, rows, **kw):
    return C.request_evidence_reconciliation(
        records, rows, expected_enable_thinking=False, expected_sampling=QWEN,
        expected_seed=isv.seed_for_task, target_vmodel=TARGET, **kw)


# --------------------------------------------------------------------------- #


def test_the_version_3_record_carries_the_sampling_terms_the_penalty_pair_and_the_trial():
    assert C.REQUEST_EVIDENCE_VERSION == 3 and C.REQUEST_EVIDENCE_ACCEPTED_VERSIONS == (2, 3)
    # the v1 / v2 names come FIRST, in their old order (a reader keyed on them keeps working)
    assert C.REQUEST_EVIDENCE_FIELDS[:14] == C.REQUEST_EVIDENCE_FIELDS_V2 == (
        "request_evidence_version", "ts", "kind", "task", "sample", "vmodel",
        "petri_thinking_on_present", "petri_thinking_on_value",
        "enable_thinking_present", "enable_thinking_value", "body_sha256",
        "request_id", "body_bytes", "max_tokens")
    assert C.REQUEST_EVIDENCE_FIELDS[14:] == C.REQUEST_EVIDENCE_SAMPLING_FIELDS == (
        "seed", "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
        "presence_penalty_present", "presence_penalty_value", "sample_idx", "item_id")
    raw, hdr = _wire(_qwen_body("1000", 1), **_headers(1, "1000", 1))
    rec = C.request_evidence_record(raw, headers=hdr, ts="T")
    assert tuple(rec) == C.REQUEST_EVIDENCE_FIELDS
    assert rec["request_evidence_version"] == 3 and rec["kind"] == "task" and rec["vmodel"] == TARGET
    assert rec["seed"] == isv.ifeval_sample_seed("1000", 1) == 3178680996
    assert (rec["temperature"], rec["top_p"], rec["top_k"], rec["min_p"], rec["repetition_penalty"]) == (
        0.7, 0.8, 20, 0.0, 1.0)
    assert rec["presence_penalty_present"] is True and rec["presence_penalty_value"] == 1.5
    assert (rec["item_id"], rec["sample_idx"]) == ("1000", 1)
    assert rec["enable_thinking_present"] is True and rec["enable_thinking_value"] is False
    assert rec["max_tokens"] == 2048 and rec["body_sha256"] == C.wire_body_sha256(raw)
    # a gemma body: no penalty, no min_p / repetition_penalty, its own numbers
    raw, hdr = _wire(_gemma_body("1005", 0), **_headers(2, "1005", 0))
    rec = C.request_evidence_record(raw, headers=hdr, ts="T")
    assert rec["presence_penalty_present"] is False and rec["presence_penalty_value"] is None
    assert (rec["temperature"], rec["top_p"], rec["top_k"], rec["min_p"], rec["repetition_penalty"]) == (
        1.0, 0.95, 64, None, None)
    assert rec["seed"] == isv.ifeval_sample_seed("1005", 0) and (rec["item_id"], rec["sample_idx"]) == ("1005", 0)
    # a probe record: no trial identity (its sample header is <vmodel>/<n>)
    raw, hdr = _wire({"model": TARGET, "messages": [], "max_tokens": 8, "temperature": 0.0},
                     **{C.HEADER_REQUEST_ID: "p1", C.HEADER_TASK: "probe/render", C.HEADER_SAMPLE: f"{TARGET}/0"})
    rec = C.request_evidence_record(raw, headers=hdr, ts="T")
    assert rec["kind"] == "probe" and rec["item_id"] is None and rec["sample_idx"] is None
    assert rec["seed"] is None and rec["temperature"] == 0.0 and rec["presence_penalty_present"] is False
    # a body that names none of the terms reads null on every one (never 0 / False-as-value)
    raw, hdr = _wire({"model": TARGET, "messages": []}, **_headers(3, "1000", 0))
    rec = C.request_evidence_record(raw, headers=hdr, ts="T")
    assert all(rec[k] is None for k in ("seed", "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                                         "presence_penalty_value", "max_tokens"))
    assert rec["presence_penalty_present"] is False


def test_writer_to_jsonl_to_reader_a_qwen_run_passes_with_the_penalty_and_the_seeds_on_the_wire(tmp_path):
    attempts = [(_qwen_body(item, trial), _headers(i, item, trial))
                for i, (item, trial) in enumerate([("1000", 0), ("1000", 1), ("1001", 0), ("1001", 1)])]
    # the grader's calls (another vmodel, greedy) ride the same file and are NOT held to the target's row
    attempts.append(({"model": GRADER, "messages": [{"role": "user", "content": "grade"}], "max_tokens": 256,
                      "temperature": 0.0, **C.chat_template_kwargs(C.THINKING_OFF)},
                     _headers(9, "1000", 0)))
    records, rows = _write_and_read(tmp_path, attempts)
    assert all(r["request_evidence_version"] == 3 for r in records)
    filed = {(IFEVAL, "1000"), (IFEVAL, "1001")}
    assert _reconcile(records, rows, filed=filed) is None
    assert _reconcile(records, rows) is None
    # the request-id / digest join is KEPT: a torn digest or a stray record still refuses
    assert "disagree on what was sent" in _reconcile(records, [{**rows[0], "body_sha256": "0" * 64}] + rows[1:])
    assert "the ledger never received" in _reconcile(records + [{**records[0], "request_id": "ridXXXX"}], rows)
    assert "NO request-evidence record" in _reconcile(records[1:], rows)
    # every record on disk carries the exact v3 field set (the writer stamps no attempt id here)
    assert all(set(r) == set(C.REQUEST_EVIDENCE_FIELDS) for r in records)


def test_an_omitted_penalty_fails_by_name(tmp_path):
    attempts = [(_qwen_body("1000", 0), _headers(0, "1000", 0)),
                (_qwen_body("1000", 1, omit_penalty=True), _headers(1, "1000", 1))]
    records, rows = _write_and_read(tmp_path, attempts)
    why = _reconcile(records, rows)
    assert why is not None
    assert "presence_penalty was NOT on the wire" in why and "rid0001" in why
    assert "not server-defaultable and must be sent per request" in why
    # the record itself says so, before any reader interprets it
    assert records[1]["presence_penalty_present"] is False and records[1]["presence_penalty_value"] is None


def test_an_altered_penalty_fails_by_name(tmp_path):
    attempts = [(_qwen_body("1000", 0, presence_penalty=1.0), _headers(0, "1000", 0))]
    records, rows = _write_and_read(tmp_path, attempts)
    why = _reconcile(records, rows)
    assert why is not None and "presence_penalty is 1.0, not the frozen row's 1.5" in why and "rid0000" in why
    # every other term is held the same way, and a term the row leaves unset must be ABSENT
    for field, value, text in (("temperature", 0.6, "temperature is 0.6, not the frozen row's 0.7"),
                               ("top_k", 40, "top_k is 40, not the frozen row's 20"),
                               ("min_p", 0.1, "min_p is 0.1, not the frozen row's 0.0")):
        body = _qwen_body("1000", 0)
        body[field] = value
        recs, rws = _write_and_read(tmp_path / field, [(body, _headers(0, "1000", 0))])
        assert text in _reconcile(recs, rws)
    body = _gemma_body("1000", 0)
    recs, rws = _write_and_read(tmp_path / "gemma-on-qwen", [(body, _headers(0, "1000", 0))])
    assert "temperature is 1.0, not the frozen row's 0.7" in _reconcile(recs, rws)   # the first term that differs
    # the gemma row against a gemma body passes; a gemma body carrying a penalty refuses
    recs, rws = _write_and_read(tmp_path / "gemma", [(body, _headers(0, "1000", 0))])
    assert C.request_evidence_reconciliation(recs, rws, expected_enable_thinking=False, expected_sampling=GEMMA,
                                             expected_seed=isv.seed_for_task, target_vmodel=TARGET) is None
    with_pp = {**body, "presence_penalty": 1.5}
    recs, rws = _write_and_read(tmp_path / "gemma-pp", [(with_pp, _headers(0, "1000", 0))])
    why = C.request_evidence_reconciliation(recs, rws, expected_enable_thinking=False, expected_sampling=GEMMA,
                                            expected_seed=isv.seed_for_task, target_vmodel=TARGET)
    assert "the frozen row names no presence penalty" in why


def test_a_wrong_seed_fails_by_name(tmp_path):
    right = isv.ifeval_sample_seed("1000", 1)
    attempts = [(_qwen_body("1000", 1, seed=right + 1), _headers(0, "1000", 1))]
    records, rows = _write_and_read(tmp_path, attempts)
    why = _reconcile(records, rows)
    assert why is not None and f"seed {right + 1} is not the rule's {right}" in why and "'1000', trial 1" in why
    # the Inspect rule's value for the same (item, trial) is NOT the IFEval rule's — and refuses
    attempts = [(_qwen_body("1000", 1, seed=isv.sample_seed("1000", 1)), _headers(0, "1000", 1))]
    records, rows = _write_and_read(tmp_path / "inspect-rule", attempts)
    assert "is not the rule's" in _reconcile(records, rows)
    # no seed at all
    body = _qwen_body("1000", 1)
    del body["seed"]
    records, rows = _write_and_read(tmp_path / "no-seed", [(body, _headers(0, "1000", 1))])
    assert "carried NO seed" in _reconcile(records, rows)
    # without an expected_seed callable the seed is not held (the sampling terms still are)
    assert C.request_evidence_reconciliation(records, rows, expected_enable_thinking=False, expected_sampling=QWEN,
                                             target_vmodel=TARGET) is None


def test_a_version_2_record_reads_as_not_recorded_and_cannot_pass_the_gate(tmp_path):
    attempts = [(_qwen_body("1000", 0), _headers(0, "1000", 0))]
    records, rows = _write_and_read(tmp_path, attempts)
    v2 = {k: v for k, v in records[0].items() if k in C.REQUEST_EVIDENCE_FIELDS_V2}
    v2["request_evidence_version"] = 2
    assert set(v2) == set(C.REQUEST_EVIDENCE_FIELDS_V2)
    # the inventory / declaration gate still ADMITS a v2 record (every pre-v3 store)
    assert C.request_evidence_reconciliation([v2], rows, expected_enable_thinking=False) is None
    # ... but the sampling / seed gate reads it as NOT RECORDED — never a pass
    why = _reconcile([v2], rows)
    assert why is not None and "request_evidence_version 2 carries NO sampling terms" in why
    assert "NOT RECORDED, which never counts as a pass" in why
    assert C.request_evidence_sampling_problem(v2, QWEN, expected_seed=isv.seed_for_task) is not None
    # a version-1 record is refused outright, as before
    assert "another request_evidence_version" in C.request_evidence_reconciliation(
        [{**v2, "request_evidence_version": 1}], rows, expected_enable_thinking=False)
    # and a run with NO target task record to check refuses rather than passing vacuously
    grader_only = [({"model": GRADER, "messages": [], "max_tokens": 8, "temperature": 0.0,
                     **C.chat_template_kwargs(C.THINKING_OFF)}, _headers(5, "1000", 0))]
    records, rows = _write_and_read(tmp_path / "grader-only", grader_only)
    assert "no TASK record of the target" in _reconcile(records, rows)


def test_the_off_render_is_the_expected_one_for_an_ifeval_run(tmp_path):
    """Gate I3: every IFEval task request renders thinking OFF; the legacy
    thinking-ON reading is the default and is unchanged."""
    attempts = [(_qwen_body("1000", 0), _headers(0, "1000", 0))]
    records, rows = _write_and_read(tmp_path, attempts)
    assert C.request_evidence_declaration_absent(records, expected_enable_thinking=False) is None
    assert "did not render thinking-ON" in C.request_evidence_declaration_absent(records)
    on = [(_qwen_body("1000", 0, enable_thinking=True), _headers(0, "1000", 0))]
    records_on, rows_on = _write_and_read(tmp_path / "on", on)
    assert "did not render thinking-OFF" in _reconcile(records_on, rows_on)
    assert C.request_evidence_declaration_absent(records_on) is None


def test_the_driver_records_v3_from_the_provider_hook_and_gates_an_ifeval_run(tmp_path):
    """The production writer stamps an attempt id AFTER the v3 fields (byte-for-byte
    the v3 record + the id), so the attempt-scoped readers see one more key and
    nothing moved; the reconciliation's sampling check is what the driver's
    IFEval post-run gate calls."""
    ev = srun._RequestEvidence(tmp_path / C.REQUEST_EVIDENCE_FILENAME, attempt_id="a1")
    raw, hdr = _wire(_qwen_body("1000", 0), **_headers(0, "1000", 0))
    rec = ev.record_wire(raw, headers=hdr)
    assert tuple(rec) == (*C.REQUEST_EVIDENCE_FIELDS, C.REQUEST_EVIDENCE_ATTEMPT_FIELD)
    on_disk = ev.records()
    assert len(on_disk) == 1 and set(on_disk[0]) == set(C.REQUEST_EVIDENCE_FIELDS) | {C.REQUEST_EVIDENCE_ATTEMPT_FIELD}
    assert on_disk[0]["presence_penalty_value"] == 1.5 and on_disk[0]["seed"] == isv.ifeval_sample_seed("1000", 0)
    # the driver's post-run IFEval gate exists and calls the reconciliation with the sampling / seed checks
    src = Path(srun.__file__).read_text()
    i = src.index("ifeval_task in thinking_by_task and sampling.row is not None")
    block = src[i:i + 2000]
    assert "expected_sampling=sampling.row" in block and "expected_seed=_isv_seed.seed_for_task" in block
    assert "target_vmodel=route.virtual_model" in block and "request_evidence_sampling_gate" in block
    # the switch the gate requires follows the task's ARM — its value under a switch arm, its ABSENCE (None)
    # under the no-switch arm a family without a thinking channel runs (contracts.sends_thinking_kwargs)
    assert "expected_switch = tc.enable_thinking(ifeval_arm) if tc.sends_thinking_kwargs(ifeval_arm) else None" in block
    assert "expected_enable_thinking=expected_switch," in block
