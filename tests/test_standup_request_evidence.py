"""The native standup TAGS its own requests: every request LoraStandup.chat
sends carries `X-Tinycap-Task: probe/standup_controls` and its request-evidence row is committed WITH THE OUTCOME — accepted only after a
2xx (the request reached the engine and is in the server log), `rejected: true` with the status / error code when the proxy refused or the
transport failed (the recorder skips those) — into `<out>/positive_controls.request_evidence.jsonl`, a sidecar that is per ATTEMPT (a
prior attempt's is quarantined at bring-up beside its endpoint.json). record_resolved_sampler.py discovers the sidecar beside the server
log under --probe-expect kinds and filters the standup's rows to the current attempt, so the standup's positive controls classify as
`standup_controls` instead of UNCLASSIFIED_REQUESTS, and a refused control or a stale sidecar can never lend a tag to an untagged request.
CPU only; synthetic ids."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import native_standup as ns

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
import record_resolved_sampler as rrs  # noqa: E402

P1 = {"model": "none__lora0", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 48, "temperature": 0, "return_token_ids": True}


def _rows(out: Path):
    return [json.loads(l) for l in (out / ns.STANDUP_REQUEST_EVIDENCE_FILE).read_text().split("\n") if l.strip()]


def test_a_row_is_committed_with_its_outcome_accepted_after_a_2xx_rejected_on_a_refusal_or_a_transport_failure(tmp_path):
    out = tmp_path / "standup"
    sent = []

    def ok_post(url, payload, *, timeout_s, headers):
        sent.append((url, dict(payload), dict(headers)))
        assert not (out / ns.STANDUP_REQUEST_EVIDENCE_FILE).exists() or all(r["request_id"] != headers["X-Tinycap-Request-Id"] for r in _rows(out))   # nothing written before the send
        return {"id": "chatcmpl-x", "choices": []}
    resp = ns.tagged_standup_chat(ok_post, "http://proxy/v1/chat/completions", P1, out=out, attempt_id="a1", timeout_s=5.0)
    assert resp["id"] == "chatcmpl-x" and sent[0][2]["X-Tinycap-Task"] == "probe/standup_controls" == f"probe/{ns.STANDUP_CONTROLS_KIND}"
    assert sent[0][2]["X-Tinycap-Sample"] == "none__lora0/0" and len(sent[0][2]["X-Tinycap-Request-Id"]) == 32
    (row,) = _rows(out)
    assert row["rejected"] is False and row["status"] == 200 and row["task"] == "probe/standup_controls" and row["max_tokens"] == 48
    assert row["request_id"] == sent[0][2]["X-Tinycap-Request-Id"] and row["attempt_id"] == "a1" and row["kind"] == "probe" and row["committed_ts"]
    assert row["request_evidence_version"] == 2 and row["body_bytes"] > 0 and row["vmodel"] == "none__lora0"
    assert row["engine_request_id"] == "chatcmpl-x"                                                   # the ENGINE's id: the identity the recorder joins on

    def refused(url, payload, *, timeout_s, headers):
        raise ns.ProxyHTTPRefusal("POST -> HTTP 400 (thinking_regime_mismatch)", status=400, code="thinking_regime_mismatch", body_text="", error=None)
    with pytest.raises(ns.ProxyHTTPRefusal):
        ns.tagged_standup_chat(refused, "http://proxy/v1/chat/completions", {**P1, "model": "base"}, out=out, attempt_id="a1", timeout_s=5.0)

    def dead(url, payload, *, timeout_s, headers):
        raise ns.LoraStandupError("POST failed: ConnectionRefusedError")
    with pytest.raises(ns.LoraStandupError):
        ns.tagged_standup_chat(dead, "http://proxy/v1/chat/completions", P1, out=out, attempt_id="a1", timeout_s=5.0)
    rows = _rows(out)
    assert [(r["rejected"], r["status"], r["error_code"]) for r in rows] == [(False, 200, None), (True, 400, "thinking_regime_mismatch"), (True, None, "LoraStandupError")]
    assert rows[1]["sample"] == "base/0" and rows[2]["sample"] == "none__lora0/1"                    # numbering per vmodel, rejected or not
    # the recorder reads ONLY the accepted row as a tag — an IDENTITY row (kind, cap, engine id), never a count claim; the rejected
    # ones are skipped and counted
    tags = rrs.read_probe_tags([out / ns.STANDUP_REQUEST_EVIDENCE_FILE])
    assert tags["rows"] == [] and [r[:3] for r in tags["id_rows"]] == [("standup_controls", "48", "chatcmpl-x")] and tags["n_probe_rows"] == 3 and tags["n_rejected_skipped"] == 2
    assert tags["id_rows"][0][3] == {"attempt_id": "a1", "request_id": sent[0][2]["X-Tinycap-Request-Id"], "engine_id_field": "present"}
    assert tags["by_kind"] == {"standup_controls": [{"max_tokens": "48", "n": 1}]}
    # … and filtered to the current attempt when the recorder knows it
    tags2 = rrs.read_probe_tags([out / ns.STANDUP_REQUEST_EVIDENCE_FILE], standup_attempt="a2")
    assert tags2["id_rows"] == [] and tags2["n_other_attempt_skipped"] == 1 and tags2["standup_attempt"] == "a2"
    assert [r[:3] for r in rrs.read_probe_tags([out / ns.STANDUP_REQUEST_EVIDENCE_FILE], standup_attempt="a1")["id_rows"]] == [("standup_controls", "48", "chatcmpl-x")]
    # a response without an `id`: accepted, but the row says so (engine_request_id None) and the recorder refuses it by name
    ns.tagged_standup_chat(lambda *a, **k: {"choices": []}, "u", P1, out=out, attempt_id="a1", timeout_s=1.0)
    assert _rows(out)[-1]["rejected"] is False and _rows(out)[-1]["engine_request_id"] is None
    assert rrs.read_probe_tags([out / ns.STANDUP_REQUEST_EVIDENCE_FILE])["id_rows"][-1][:3] == ("standup_controls", "48", None)
    # a LEGACY accepted row (a sidecar written by a standup older than the identity join: the field ABSENT) is an identity row too —
    # never a count claim — and reads as engine_id_field "absent"
    legacy = {k: v for k, v in _rows(out)[0].items() if k != "engine_request_id"}
    with open(out / ns.STANDUP_REQUEST_EVIDENCE_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(legacy) + "\n")
    tags3 = rrs.read_probe_tags([out / ns.STANDUP_REQUEST_EVIDENCE_FILE])
    assert tags3["rows"] == [] and tags3["id_rows"][-1][:3] == ("standup_controls", "48", None) and tags3["id_rows"][-1][3]["engine_id_field"] == "absent"


def test_a_prior_attempts_sidecar_is_quarantined_at_bring_up_and_the_names_agree_across_the_modules(tmp_path):
    out = tmp_path / "standup"
    ns.tagged_standup_chat(lambda *a, **k: {"id": "x"}, "u", P1, out=out, attempt_id="old", timeout_s=1.0)
    (out / ns.ENDPOINT_FILE).write_text(json.dumps({ns.ENDPOINT_ATTEMPT_KEY: "old"}))
    stale = ns.quarantine_prior_endpoint(out, attempt_id="new12345678")
    moved = ns.quarantine_prior_standup_evidence(out, attempt_id="new12345678", stale_dir=stale)
    assert moved == stale and (stale / ns.STANDUP_REQUEST_EVIDENCE_FILE).is_file() and not (out / ns.STANDUP_REQUEST_EVIDENCE_FILE).exists()
    assert ns.quarantine_prior_standup_evidence(out, attempt_id="new12345678") is None            # nothing to move: a no-op
    ns.tagged_standup_chat(lambda *a, **k: {"id": "y"}, "u", P1, out=out, attempt_id="new12345678", timeout_s=1.0)
    (row,) = _rows(out)
    assert row["attempt_id"] == "new12345678" and row["sample"] == "none__lora0/0"                # the numbering restarts with the attempt
    # a sidecar with NO endpoint (an attempt that died before READY) is quarantined into its own stale dir
    d2 = ns.quarantine_prior_standup_evidence(out, attempt_id="third1234567")
    assert d2 is not None and d2.name.startswith(ns.STALE_DIR_PREFIX) and (d2 / ns.STANDUP_REQUEST_EVIDENCE_FILE).is_file()
    assert ns.STANDUP_REQUEST_EVIDENCE_FILE == rrs.STANDUP_EVIDENCE_FILENAME and ns.ENDPOINT_FILE == rrs.STANDUP_ENDPOINT_FILENAME
    assert ns.ENDPOINT_ATTEMPT_KEY == rrs.STANDUP_ENDPOINT_ATTEMPT_KEY
    # the boundary: chat() IS tagged_standup_chat; the bring-up quarantines the sidecar with the endpoint; bites + mixed batches ride it
    src = (REPO / "src/sisteer/capeval/serving/native_standup.py").read_text()
    body = src[src.index("    def chat(self, payload"):src.index("    def greedy_ids(self")]
    assert "return tagged_standup_chat(_post_json," in body
    assert "quarantine_prior_standup_evidence(out, attempt_id=attempt_id, stale_dir=stale)" in src
    assert "stt.probe_fn(su.chat," in src and "pool.submit(su.greedy_ids_with_request_id" in src
