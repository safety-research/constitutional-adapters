"""build_probe_set stops at the FIRST credential rejection (the WIF driver sweep).

Before: ``_call_api`` classified a 429 and retried EVERYTHING else up to ``API_ERR_RETRIES`` -- a 401 / 403, or under
``SISTEER_SECRETS_MODE=wif`` the federation identity refused at the token exchange, cost four doomed requests per item at
concurrency 300; ``verify_items`` then filed each as the item's ``api_error``, ``cmd_verify`` wrote the verified items and a
``verify_report.json`` naming them dropped, and returned 0 whenever some item had passed before the credential died -- a
rejected credential as a verification result. ``generate_probe_items``' plain ``asyncio.gather`` raised at the first task's
exhaustion and left the siblings running to theirs. Now the credential is classified FIRST, the typed
``sisteer.vault.JudgeCredentialRejected`` latches on the pass's limiter (every task raises before its wire call), the outstanding
tasks are cancelled, ``verify_items`` never files it as an item's reason, and ``generate`` / ``verify`` write NO items and NO report
and exit with the typed message as the terminal line. Offline: every client is a stand-in."""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import anthropic
import httpx
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import build_probe_set as bps  # noqa: E402
from sisteer import vault  # noqa: E402
from test_probe_set import MockClient, _right_verdict, build_sources, gen_responder, make_items  # noqa: E402


def _credential_error(status: int = 401):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if status == 401:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "synthetic"}}
        return anthropic.AuthenticationError(f"Error code: 401 - {body}", response=httpx.Response(401, request=req), body=body)
    body = {"type": "error", "error": {"type": "permission_error", "message": "synthetic"}}
    return anthropic.PermissionDeniedError(f"Error code: 403 - {body}", response=httpx.Response(403, request=req), body=body)


def _federation_rejected(status=401):
    if status is None:
        return anthropic.WorkloadIdentityError("Failed to reach token endpoint: synthetic")
    return anthropic.WorkloadIdentityError(f"Token exchange failed (HTTP {status}): synthetic", status_code=status, body="synthetic")


def _server_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.InternalServerError("Error code: 500 - synthetic", response=httpx.Response(500, request=req), body=None)


class _Block:
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class RejectingClient:
    """An async stand-in: ``fail(n)`` decides the n-th call's fate (None = ``responder(kw)``'s text); ``delay`` keeps every
    slot in flight long enough for the stop to race the queue; stamped like a vault-built client (the placeholder is not a key)."""

    def __init__(self, fail, responder=None, *, delay: float = 0.0, org: str = "lowprio"):
        self.fail, self.responder, self.delay = fail, responder or (lambda kw: "ok"), delay
        self.calls = 0
        self.sisteer_pool_stamp = vault.pool_stamp(org, "ORG-KEY-PLACEHOLDER")
        outer = self

        class _Msgs:
            async def create(self, **kw):
                outer.calls += 1
                n = outer.calls
                if outer.delay:
                    await asyncio.sleep(outer.delay)
                exc = outer.fail(n)
                if exc is not None:
                    raise exc
                return _Resp(outer.responder(kw))

        self.messages = _Msgs()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    real = asyncio.sleep

    async def fast(s):
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", fast)


def test_call_api_classifies_the_credential_first_latches_it_on_the_limiter_and_never_retries_it(monkeypatch):
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    for exc, status, error_type in ((_credential_error(401), 401, "AuthenticationError"),
                                    (_credential_error(403), 403, "PermissionDeniedError"),
                                    (_federation_rejected(401), 401, "WorkloadIdentityError")):
        client = RejectingClient(lambda n, exc=exc: exc)
        limiter = bps.AdaptiveConcurrency(8)
        with pytest.raises(vault.JudgeCredentialRejected) as ei:
            asyncio.run(bps._call_api(client, limiter, model="m", max_tokens=5, messages=[]))
        e = ei.value
        assert (e.pool, e.status, e.error_type) == ("lowprio", status, error_type)
        assert client.calls == 1, (client.calls, error_type)                 # ONE request: never API_ERR_RETRIES more
        assert limiter.credential.held is e and "wif" in str(e) and "synthetic" not in str(e)
        with pytest.raises(vault.JudgeCredentialRejected) as again:
            asyncio.run(bps._call_api(client, limiter, model="m", max_tokens=5, messages=[]))
        assert again.value is e and client.calls == 1                        # latched: no wire call
    # NOT this: the exchange's 5xx / an unreachable token endpoint / a 500 take the generic bounded retry, then the answer stands
    for transient in (_federation_rejected(503), _federation_rejected(None), _server_error()):
        client = RejectingClient(lambda n, exc=transient: exc if n == 1 else None)
        resp = asyncio.run(bps._call_api(client, bps.AdaptiveConcurrency(8), model="m", max_tokens=5, messages=[]))
        assert resp.content[0].text == "ok" and client.calls == 2


def test_verify_items_stops_at_the_first_rejection_cancels_the_outstanding_tasks_and_never_files_an_api_error_for_it(tmp_path):
    src = build_sources(tmp_path)
    items = make_items(src, n_heldout=4, n_in=4, n_out=3)
    assert len(items) >= 8

    def responder(kw):
        it = next(i for i in items if i["prefix"] in kw["messages"][0]["content"])
        return json.dumps(_right_verdict(it))

    client = RejectingClient(lambda n: None if n <= 2 else _credential_error(401), responder, delay=0.02)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        asyncio.run(bps.verify_items(client, items, lambda it: "ref", concurrency=3))
    assert ei.value.status == 401 and 3 <= client.calls <= 2 + 3, client.calls   # never one per item, never API_ERR_RETRIES per item
    # the generation pass the same way
    specs, sections = bps.plan_items, None
    specs, sections = __import__("test_probe_set").plan_specs(src, 4, 4, 3)
    gen = RejectingClient(lambda n: None if n <= 1 else _federation_rejected(403), gen_responder, delay=0.02)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        asyncio.run(bps.generate_probe_items(gen, specs, {s["id"]: s for s in sections}, src["constitution_text"], concurrency=4))
    assert ei.value.error_type == "WorkloadIdentityError" and 2 <= gen.calls <= 1 + 4, gen.calls


def test_cmd_verify_and_cmd_generate_write_no_items_and_no_report_and_end_with_the_typed_line(tmp_path, monkeypatch, capsys):
    src = build_sources(tmp_path)
    items = make_items(src, n_heldout=2, n_in=1, n_out=0)
    raw = tmp_path / "raw.jsonl"
    bps._write_jsonl(raw, items)

    def responder(kw):
        it = next(i for i in items if i["prefix"] in kw["messages"][0]["content"])
        return json.dumps(_right_verdict(it))

    # one item verified, then the credential dies: before this the run wrote the one item, a report naming the others
    # `api_error`, and returned 0
    client = RejectingClient(lambda n: None if n == 1 else _credential_error(401), responder, delay=0.02)
    monkeypatch.setattr(bps, "make_client", lambda org: client)
    out = tmp_path / "verified.jsonl"
    with pytest.raises(SystemExit) as ei:
        bps.main(["verify", "--raw", str(raw), "--sections", str(src["sections"]), "--constitution-md", str(src["md"]),
                  "--constitution-text-file", str(src["md"]), "--out", str(out), "--concurrency", "2"])
    msg = str(ei.value)
    assert msg.startswith("[probe-set] the API rejected the lowprio pool's credential: AuthenticationError (HTTP 401, authentication_error)")
    assert not out.exists() and not (tmp_path / "verified.jsonl.verify_report.json").exists()
    assert client.calls <= 1 + 2
    o = capsys.readouterr().out
    assert "[probe-set] CREDENTIAL REJECTED" in o and "nothing is written" in o
    # the generate pass: no raw items, no gen report
    gen = RejectingClient(lambda n: _credential_error(403), gen_responder, delay=0.02)
    monkeypatch.setattr(bps, "make_client", lambda org: gen)
    gout = tmp_path / "gen.jsonl"
    with pytest.raises(SystemExit) as ei:
        bps.main(["generate", "--heldout-split", str(src["split"]), "--advice", str(src["advice"]), "--stories", str(src["stories"]),
                  "--sections", str(src["sections"]), "--constitution-md", str(src["md"]), "--constitution-text-file", str(src["md"]),
                  "--n-heldout", "4", "--n-in", "4", "--n-out", "3", "--out", str(gout), "--concurrency", "3"])
    assert "PermissionDeniedError (HTTP 403" in str(ei.value) and not gout.exists() and not (tmp_path / "gen.jsonl.gen_report.json").exists()
    assert gen.calls <= 3
    # the happy path is unchanged: the report still splits its reasons and returns 0
    ok = MockClient(responder)
    monkeypatch.setattr(bps, "make_client", lambda org: ok)
    rc = bps.main(["verify", "--raw", str(raw), "--sections", str(src["sections"]), "--constitution-md", str(src["md"]),
                   "--constitution-text-file", str(src["md"]), "--out", str(out)])
    assert rc == 0 and out.exists() and json.loads((tmp_path / "verified.jsonl.verify_report.json").read_text())["n_passed"] == len(items)
