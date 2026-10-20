"""The Qwen seed-replicate battery, an additive for the determinism
check: `response_evidence.jsonl` rows are version 2 and carry `text_sha256` + `text_sha256_of` — the sha256 of the
UTF-8 bytes of the EXACT response text the client extracted (`choices[0].message.content`, `"content"` — the reading
the replay hashes through `ac.text_sha256(completion_text(resp))`), else of the canonical JSON of the full body
(`"body"`), both null without a body. A strict reader refuses a row without the field (a version-1 row) by name.
Also: the v3 request record carries the wire body's sha256 and LENGTH (`body_bytes`), not its bytes, so the record now
ALSO carries them as the additive `body_b64` field (`ac.REQUEST_BODY_B64_FIELD`; the determinism check's
`bodies_from_records` holds it to `body_sha256` and `body_bytes` before re-sending) — no separate `request_bodies.jsonl`.
Default-off byte identity: neither file exists without the lane switch (`from_env()` is None)."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from seedrep_fixes_common import GSM8K_TASK, REPO, S, VMODEL, body, hook_off, lane_on, read_jsonl, wire, ws3g_modules  # noqa: F401
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval import request_evidence_client as rev

TEXT = "The answer is 42.\n\nFinal: 42 — with a non-ASCII dash and an emoji ✓"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ok(content, *, cid="cmpl-1", choices=None):
    return {"id": cid, "object": "chat.completion", "model": VMODEL,
            "choices": choices if choices is not None else [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}


def _load_determinism_check(here: Path, ac, rea):
    """`determinism_check.py` by path, bound to THE SAME `audit_common` object `ws3g_modules()` returned.

    `ws3g_modules()` puts the PRIOR `sys.modules` entries back after its load, so a bare `exec_module` here
    would let their `import audit_common` load a SECOND copy off `sys.path` (their module inserts its own directory) —
    a second `AuditRefusal` class the tool raises correctly, with the right text, that `pytest.raises(ac.AuditRefusal)`
    cannot catch: a module-identity mismatch, not a missing refusal (the same class fixed for the earlier importers).
    Register both of their modules for the length of the load, then put the prior objects
    back, so the tool and this test bind ONE `audit_common` — asserted, since that identity is the whole point."""
    prior = {name: sys.modules.get(name) for name in ("audit_common", "request_evidence_audit")}
    before = list(sys.path)
    try:
        sys.modules["audit_common"], sys.modules["request_evidence_audit"] = ac, rea
        spec = importlib.util.spec_from_file_location("determinism_check", here)
        dc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dc)  # type: ignore[union-attr]
    finally:
        for name, m in prior.items():
            if m is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = m
        sys.path[:] = before
    assert dc.ac is ac and dc.refuse is ac.refuse, "determinism_check bound a second audit_common"
    return dc


def test_record_response_hashes_the_exact_content_or_the_whole_body_and_nothing_without_one(tmp_path, monkeypatch):
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"leg": "gsm8k", "s": S})
    hdr = rev.label_headers(GSM8K_TASK, "q1", 0)
    # "content": the exact assistant text, UTF-8, byte for byte — the reading the determinism replay compares against
    row = client.record_response("r1", status=200, resp=_ok(TEXT), headers=hdr)
    assert tuple(row)[:-1] == rev.RESPONSE_EVIDENCE_FIELDS and tuple(row)[-1] == rev.EXTRA_FIELD
    assert row["response_evidence_version"] == 2 == rev.RESPONSE_EVIDENCE_VERSION
    assert row["text_sha256"] == _sha(TEXT) and row["text_sha256_of"] == "content"
    assert row["text_sha256"] != _sha(TEXT.encode("utf-8").decode("latin-1"))   # the bytes hashed are UTF-8, not a re-encoding
    # a thinking-ON reply: reasoning_content is NOT in the content hash (the stores' text and the comparison share the limit)
    thinking = _ok(TEXT, cid="cmpl-2")
    thinking["choices"][0]["message"]["reasoning_content"] = "let me think"
    assert client.record_response("r2", status=200, resp=thinking, headers=hdr)["text_sha256"] == _sha(TEXT)
    # several choices: the FIRST choice's content, exactly as completion_text reads a reply
    many = client.record_response("r-many", status=200, headers=hdr,
                                  resp=_ok("a", cid="cmpl-4", choices=[{"message": {"content": "a"}}, {"message": {"content": "b"}}]))
    assert (many["text_sha256"], many["text_sha256_of"]) == (_sha("a"), "content")
    # "body": no single text field — no choices; a non-string (multimodal) content — the canonical JSON of the whole body
    for cid, resp in (("cmpl-3", {"id": "cmpl-3", "model": VMODEL, "object": "list"}),
                      ("cmpl-5", _ok([{"type": "text", "text": "x"}], cid="cmpl-5"))):
        r = client.record_response(f"r-{cid}", status=200, resp=resp, headers=hdr)
        canonical = json.dumps(resp, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        assert (r["text_sha256"], r["text_sha256_of"]) == (hashlib.sha256(canonical).hexdigest(), "body")
    # no body (a 500, a dead transport): both null, and the completion id null with them
    dead = client.record_response("r9", status=500, error="boom", headers=hdr)
    assert (dead["text_sha256"], dead["text_sha256_of"], dead["completion_id"]) == (None, None, None)
    # the file: every row passes the strict reader, and the `response_text_shas` reading (isinstance str) picks up exactly the hashed ones
    rows = rev.read_response_evidence(client.response_path)
    assert [r["request_id"] for r in rows] == ["r1", "r2", "r-many", "r-cmpl-3", "r-cmpl-5", "r9"]
    assert {r["request_id"]: r["text_sha256"] for r in rows if isinstance(r.get("text_sha256"), str)}["r1"] == _sha(TEXT)
    assert rev.response_text_sha256(_ok(TEXT)) == (_sha(TEXT), "content")
    with pytest.raises(rev.RequestEvidenceError, match="parsed body mapping"):
        rev.response_text_sha256("not a body")  # type: ignore[arg-type]


def test_ws3g_text_sha256_reads_this_field_as_the_same_hash():
    ac, _rea = ws3g_modules()
    assert ac.text_sha256(TEXT) == rev.response_text_sha256(_ok(TEXT))[0]
    here = next((d for d in (REPO / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py",   # this tree first (W)
                             REPO.parent / "seedrep-ws3g" / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py")
                 if d.is_file()), REPO / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py")
    if not here.is_file():
        pytest.skip("WS3-G's determinism_check.py is not beside this tree")
    dc = _load_determinism_check(here, ac, _rea)
    assert dc.completion_text(_ok(TEXT)) == TEXT   # their extraction IS choices[0].message.content


@pytest.mark.parametrize("label, mutate, match", [
    ("a version-1 row (no version, no sha)", lambda r: {k: v for k, v in r.items() if k not in ("response_evidence_version", "text_sha256", "text_sha256_of")},
     r"missing \['response_evidence_version', 'text_sha256', 'text_sha256_of'\]"),
    ("missing text_sha256 only", lambda r: {k: v for k, v in r.items() if k != "text_sha256"}, r"missing \['text_sha256'\]"),
    ("version 1", lambda r: {**r, "response_evidence_version": 1}, "response_evidence_version 1"),
    ("version True", lambda r: {**r, "response_evidence_version": True}, "response_evidence_version True"),
    ("sha without its discriminator", lambda r: {**r, "text_sha256_of": None}, "both present or both null"),
    ("discriminator without a sha", lambda r: {**r, "text_sha256": None}, "both present or both null"),
    ("an unknown discriminator", lambda r: {**r, "text_sha256_of": "message"}, "not content / body / null"),
    ("a sha that is not 64-hex", lambda r: {**r, "text_sha256": "abc"}, "not a 64-hex sha256"),
    ("an uppercase sha", lambda r: {**r, "text_sha256": r["text_sha256"].upper()}, "not a 64-hex sha256"),
    ("a completion id with both nulled", lambda r: {**r, "text_sha256": None, "text_sha256_of": None}, "recorded without a text_sha256"),
    ("an unknown key", lambda r: {**r, "text": "leaked"}, r"unknown \['text'\]"),
    ("no request id", lambda r: {**r, "request_id": ""}, "request_id ''"),
])
def test_the_strict_reader_refuses_a_row_without_the_field_or_off_shape_by_name(tmp_path, monkeypatch, label, mutate, match):
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"s": S})
    good = client.record_response("r1", status=200, resp=_ok(TEXT), headers=rev.label_headers(GSM8K_TASK, "q1", 0))
    assert rev.check_response_row(good) == good
    bad = mutate(json.loads(json.dumps(good)))
    with pytest.raises(rev.RequestEvidenceError, match=match):
        rev.check_response_row(bad)
    # ... and through the file reader, naming the line
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(rev.RequestEvidenceError, match=f"{p}:2"):
        rev.read_response_evidence(p)


def test_the_request_record_carries_the_bodys_sha_its_length_and_now_its_exact_bytes(tmp_path, monkeypatch):
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"leg": "gsm8k", "s": S, "attempt_id": "ta-1"})
    raw, headers = wire(GSM8K_TASK, "q1", 0, payload=body(seed=7))
    rec = client.record_request(raw, headers=headers)
    # the v3 record proper: the sha of the exact wire bytes and their LENGTH — NOT the bytes
    assert rec["body_sha256"] == C.wire_body_sha256(raw) and rec["body_bytes"] == len(raw)
    assert rev.REQUEST_BODY_B64_FIELD not in C.request_evidence_record(raw, headers=headers)
    # ... and now the bytes themselves, additive on the record, decoding to EXACTLY the wire bytes; no side file
    assert rev.REQUEST_BODY_B64_FIELD == "body_b64" and base64.b64decode(rec["body_b64"], validate=True) == raw
    [on_disk] = read_jsonl(client.request_path)
    assert on_disk == rec and not (client.out_dir / "request_bodies.jsonl").exists()
    # The determinism check's real reader over the record (its bodies_from_records), when it is beside this tree:
    # the bytes come back, and a tampered body_b64 (one byte changed) refuses against the record's own sha
    here = next((d for d in (REPO / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py",   # this tree first (W)
                             REPO.parent / "seedrep-ws3g" / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py")
                 if d.is_file()), REPO / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py")
    if not here.is_file():
        pytest.skip("WS3-G's determinism_check.py is not beside this tree")
    ac, _rea = ws3g_modules()
    dc = _load_determinism_check(here, ac, _rea)   # bound to THIS `ac` (their module imports it by name)
    theirs = getattr(dc, "bodies_from_records", None)
    if theirs is None or getattr(ac, "REQUEST_BODY_B64_FIELD", None) != rev.REQUEST_BODY_B64_FIELD:
        pytest.skip("WS3-G's determinism check does not read body_b64 off the record on this tree")
    assert theirs([rec], where=client.request_path) == {rec["request_id"]: raw}
    tampered = dict(rec)
    tampered["body_b64"] = base64.b64encode(raw[:-1] + (b" " if raw[-1:] != b" " else b"\t")).decode("ascii")
    with pytest.raises(ac.AuditRefusal, match="not the wire bytes"):
        theirs([tampered], where=client.request_path)
    # a record without the field is simply absent from their map (a pre-fix record) — the replay of its draw refuses there by name
    assert theirs([{k: v for k, v in rec.items() if k != "body_b64"}], where=client.request_path) == {}


def test_real_httpx_hooks_write_the_content_hash_of_what_the_engine_returned(tmp_path, monkeypatch):
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"leg": "gsm8k", "s": S})
    served = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        served["n"] += 1
        if served["n"] == 2:
            return httpx.Response(500, json={"error": {"message": "hiccup"}})
        return httpx.Response(200, json=_ok(f"{TEXT} #{served['n']}", cid=f"cmpl-{served['n']}"))

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://loopback") as hc:
            client.install_httpx(hc)
            for i in range(3):
                await hc.post("/v1/chat/completions", content=json.dumps(body(seed=i)).encode(),
                              headers={**rev.label_headers(GSM8K_TASK, "q1", i), "content-type": "application/json"})
    asyncio.run(go())
    rows = rev.read_response_evidence(client.response_path)
    assert [(r["status"], r["text_sha256_of"]) for r in rows] == [(200, "content"), (500, None), (200, "content")]
    assert rows[0]["text_sha256"] == _sha(f"{TEXT} #1") and rows[2]["text_sha256"] == _sha(f"{TEXT} #3") and rows[1]["text_sha256"] is None
    # every attempt's record carries its exact bytes, the failed one included (it was sent)
    records = read_jsonl(client.request_path)
    assert [r["request_id"] for r in records] == [r["request_id"] for r in rows]
    assert [json.loads(base64.b64decode(r["body_b64"]))["seed"] for r in records] == [0, 1, 2]
    assert all(C.wire_body_sha256(base64.b64decode(r["body_b64"])) == r["body_sha256"]
               and len(base64.b64decode(r["body_b64"])) == r["body_bytes"] for r in records)


def test_default_off_writes_neither_file(tmp_path):
    assert rev.from_env() is None
    obj = object()
    assert rev.maybe_attach_proxy_client(obj) is obj and rev.maybe_wrap_post_json(obj) is obj
    assert rev.maybe_request_headers_fn(rev.LEG_BON) is None and rev.maybe_misalign_request_evidence() is None
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.parametrize("label, mutate, match", [
    ("no body_b64 (a pre-fix record, or production's writer)", lambda r: {k: v for k, v in r.items() if k != "body_b64"}, "carries no body_b64"),
    ("body_b64 not a str", lambda r: {**r, "body_b64": 7}, "not a base64 str"),
    ("body_b64 not base64", lambda r: {**r, "body_b64": "!!not-base64!!"}, "is not base64"),
    ("a tampered body (one byte)", lambda r: {**r, "body_b64": base64.b64encode(base64.b64decode(r["body_b64"])[:-1] + b" ").decode()}, "not the wire bytes"),
    ("another request's body", lambda r: {**r, "body_b64": base64.b64encode(b'{"model": "other"}').decode()}, "not the wire bytes"),
    ("a stale digest", lambda r: {**r, "body_sha256": "0" * 64}, "not the wire bytes"),
    ("a stale length", lambda r: {**r, "body_bytes": r["body_bytes"] + 1}, "body_bytes says"),
    ("a boolean length", lambda r: {**r, "body_bytes": True}, "body_bytes says"),
    ("no request id", lambda r: {**r, "request_id": ""}, "request_id ''"),
])
def test_the_strict_request_reader_refuses_a_record_whose_retained_body_is_not_its_digests(tmp_path, monkeypatch, label, mutate, match):
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"leg": "gsm8k", "s": S})
    raw, headers = wire(GSM8K_TASK, "q1", 0, payload=body(seed=3))
    good = client.record_request(raw, headers=headers)
    assert rev.check_request_record(good) == good and rev.read_request_evidence(client.request_path) == [good]
    bad = mutate(json.loads(json.dumps(good)))
    with pytest.raises(rev.RequestEvidenceError, match=match):
        rev.check_request_record(bad)
    p = tmp_path / "records.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(rev.RequestEvidenceError, match=f"{p}:2"):
        rev.read_request_evidence(p)
    # The determinism check's real reader agrees on the digest cases
    # (its bodies_from_records refuses a body that is not its record's)
    here = next((d for d in (REPO / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py",   # this tree first (W)
                             REPO.parent / "seedrep-ws3g" / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py")
                 if d.is_file()), REPO / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py")
    if here.is_file() and "not the wire bytes" in match:
        ac, _rea = ws3g_modules()
        dc = _load_determinism_check(here, ac, _rea)
        theirs = getattr(dc, "bodies_from_records", None)
        if theirs is not None:
            with pytest.raises(ac.AuditRefusal, match="not the wire bytes"):
                theirs([bad], where=p)


def test_the_writer_holds_the_retained_bytes_to_the_records_digest_and_length_before_writing(tmp_path, monkeypatch):
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"leg": "gsm8k", "s": S})
    raw, headers = wire(GSM8K_TASK, "q1", 0)
    # the good path: what the writer retains IS what its digest and length describe
    rec = client.record_request(raw, headers=headers)
    assert base64.b64decode(rec["body_b64"]) == raw and rec["body_sha256"] == C.wire_body_sha256(raw) and rec["body_bytes"] == len(raw)
    # known-bad: a record builder that misdescribes the body (a stale digest) — the writer refuses and writes NOTHING
    real = rev.C.request_evidence_record

    def stale(raw_, *, headers, ts=None):
        r = dict(real(raw_, headers=headers)) if ts is None else dict(real(raw_, headers=headers, ts=ts))
        r["body_sha256"] = "f" * 64
        return r

    monkeypatch.setattr(rev.C, "request_evidence_record", stale)
    raw2, headers2 = wire(GSM8K_TASK, "q2", 0)
    with pytest.raises(rev.RequestEvidenceError, match="misdescribes its own body.*not the wire bytes"):
        client.record_request(raw2, headers=headers2)
    # known-bad: a builder emitting body_bytes as an INTEGRAL FLOAT — `247.0 == 247`, so a numeric `!=`
    # would have written a record the reader then refuses; the writer applies the reader's original-type rule
    def floaty(raw_, *, headers, ts=None):
        r = dict(real(raw_, headers=headers)) if ts is None else dict(real(raw_, headers=headers, ts=ts))
        r["body_bytes"] = float(len(raw_))
        return r

    monkeypatch.setattr(rev.C, "request_evidence_record", floaty)
    raw3, headers3 = wire(GSM8K_TASK, "q3", 0)
    assert float(len(raw3)) == len(raw3)   # the equality the ad-hoc check would have accepted
    with pytest.raises(rev.RequestEvidenceError, match=rf"misdescribes its own body.*body_bytes says {float(len(raw3))!r}"):
        client.record_request(raw3, headers=headers3)
    # and a boolean length (`True == 1`) the same way
    def booly(raw_, *, headers, ts=None):
        r = dict(real(raw_, headers=headers)) if ts is None else dict(real(raw_, headers=headers, ts=ts))
        r["body_bytes"] = True
        return r

    monkeypatch.setattr(rev.C, "request_evidence_record", booly)
    with pytest.raises(rev.RequestEvidenceError, match="body_bytes says True"):
        client.record_request(*wire(GSM8K_TASK, "q4", 0))
    # nothing of the three refused attempts reached the file or the counter
    assert [r["request_id"] for r in read_jsonl(client.request_path)] == [rec["request_id"]]
    assert client.n_requests == 1
