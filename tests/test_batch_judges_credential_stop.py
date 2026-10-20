"""The Batch-API judges stop at the FIRST credential rejection with the typed terminal line (the WIF driver sweep).

``scripts/judge_batch.py``: ``_create_chunk`` caught the 413 and the 429 and let a 401 / 403 -- or under
``SISTEER_SECRETS_MODE=wif`` the federation identity refused at the token exchange -- propagate RAW out of ``submit``, with the
pre-create journal entry left behind, so the next run warned of a possible orphan batch that was never created (a rejection is a
clean HTTP answer: nothing was made). Now the rejection is the typed ``sisteer.vault.JudgeCredentialRejected`` naming the Batch
org, the journal entry is dropped, ``submit`` stops at that chunk, and ``main`` ends with the typed message (``vault.credential_exit``)
-- a rejection out of a poll or a results read included. ``scripts/sdfbase/full22/judge/batch_judge_fill.py``: its three
synchronous commands end the same way. Offline: the clients are stand-ins."""
from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import anthropic
import httpx
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import judge_batch as jb  # noqa: E402
from sisteer import vault  # noqa: E402
from test_judge_batch import FakeBatches, FakeClient, HAIKU  # noqa: E402


def _credential_error(status: int = 401):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches")
    if status == 401:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "synthetic"}}
        return anthropic.AuthenticationError(f"Error code: 401 - {body}", response=httpx.Response(401, request=req), body=body)
    body = {"type": "error", "error": {"type": "permission_error", "message": "synthetic"}}
    return anthropic.PermissionDeniedError(f"Error code: 403 - {body}", response=httpx.Response(403, request=req), body=body)


def _federation_rejected(status=401):
    return anthropic.WorkloadIdentityError(f"Token exchange failed (HTTP {status}): synthetic", status_code=status,
                                           body={"error": "invalid_grant"})


def _server_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches")
    return anthropic.InternalServerError("Error code: 500 - synthetic", response=httpx.Response(500, request=req), body=None)


class RejectingBatches(FakeBatches):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc
        self.create_calls = 0

    def create(self, requests):
        self.create_calls += 1
        raise self.exc


def _judge(tmp_path, exc, **kw):
    client = FakeClient()
    client.messages = SimpleNamespace(batches=RejectingBatches(exc))
    judge = jb.BatchJudge(tmp_path / "judge_cache.jsonl", model=HAIKU, client=client, max_requests_per_batch=5, **kw)
    return judge, client.messages.batches


@pytest.mark.parametrize("exc, status, error_type", [(_credential_error(401), 401, "AuthenticationError"),
                                                      (_credential_error(403), 403, "PermissionDeniedError"),
                                                      (_federation_rejected(401), 401, "WorkloadIdentityError")])
def test_judge_batch_create_stops_at_the_first_rejection_typed_drops_the_false_journal_entry_and_submits_nothing_else(
        tmp_path, monkeypatch, exc, status, error_type):
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    judge, batches = _judge(tmp_path, exc)
    for i in range(12):
        judge.collect_questions(f"synthetic prompt {i}")
    assert len(judge.pending) == 12
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        judge.submit()
    e = ei.value
    assert (e.pool, e.status, e.error_type) == (jb.BATCH_ORG, status, error_type) and "wif" in str(e) and "synthetic" not in str(e)
    assert batches.create_calls == 1                                          # the first chunk's create, and no later chunk
    assert len(judge.pending) == 12                                           # nothing left `pending` as if submitted
    state = judge._batch_state()
    assert state.get("creating", []) == [] and state["batches"] == []         # the pre-create journal entry is gone: nothing was created
    judge._surface_possible_orphans()                                         # and so the next run warns of no orphan
    assert "possible_orphans" not in judge._batch_state()


def test_judge_batch_a_500_at_create_is_still_raised_as_the_sdk_raised_it_with_its_journal_entry_kept(tmp_path):
    judge, batches = _judge(tmp_path, _server_error())
    judge.collect_questions("synthetic prompt")
    with pytest.raises(anthropic.InternalServerError):
        judge.submit()
    assert batches.create_calls == 1 and len(judge._batch_state()["creating"]) == 1   # unchanged: a 5xx is not a clean "never created"


def test_judge_batch_main_ends_with_the_typed_line_for_a_rejection_out_of_any_call(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(sys, "argv", ["judge_batch.py", "--suite", "ifeval", "--out-dir", str(out), "--org", "batch"])
    # the typed error out of a create
    monkeypatch.setattr(jb, "_main", lambda args, ap: (_ for _ in ()).throw(vault.JudgeCredentialRejected("batch", 401, "AuthenticationError")))
    with pytest.raises(SystemExit) as ei:
        jb.main()
    assert str(ei.value).startswith("[judge-batch] the API rejected the batch pool's credential: AuthenticationError (HTTP 401")
    # the SDK's own, out of a poll / results call: typed with --org's pool
    monkeypatch.setattr(jb, "_main", lambda args, ap: (_ for _ in ()).throw(_federation_rejected(403)))
    with pytest.raises(SystemExit) as ei:
        jb.main()
    assert "[judge-batch] the API rejected the batch pool's credential: WorkloadIdentityError (HTTP 403)" in str(ei.value)
    assert not (out / "judge_batch.json").exists()
    # anything else passes as before
    monkeypatch.setattr(jb, "_main", lambda args, ap: (_ for _ in ()).throw(RuntimeError("other")))
    with pytest.raises(RuntimeError):
        jb.main()


def test_batch_judge_fill_commands_end_with_the_typed_line(tmp_path, monkeypatch):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sdfbase" / "full22" / "judge"))
    import batch_judge_fill as bf

    state = tmp_path / "state.json"
    state.write_text(json.dumps({"batches": {}}))
    monkeypatch.setattr(sys, "argv", ["batch_judge_fill.py", "poll", "--state", str(state), "--once"])
    monkeypatch.setattr(bf, "cmd_poll", lambda args: (_ for _ in ()).throw(_credential_error(401)))
    with pytest.raises(SystemExit) as ei:
        bf.main()
    assert str(ei.value).startswith("[batch-fill] the API rejected the batch pool's credential: AuthenticationError (HTTP 401, authentication_error)")
    monkeypatch.setattr(bf, "cmd_poll", lambda args: (_ for _ in ()).throw(_federation_rejected(401)))
    with pytest.raises(SystemExit) as ei:
        bf.main()
    assert "WorkloadIdentityError (HTTP 401)" in str(ei.value) and "synthetic" not in str(ei.value)
    monkeypatch.setattr(bf, "cmd_poll", lambda args: 3)
    assert bf.main() == 3
