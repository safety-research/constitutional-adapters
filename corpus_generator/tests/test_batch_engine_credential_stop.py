"""The batch engine and the scripts stop at the FIRST credential rejection (the WIF driver sweep).

``WaveEngine._create`` caught the 429 and let a 401 / 403 -- or under ``SISTEER_SECRETS_MODE=wif`` the federation identity
refused at the token exchange, the SDK's ``WorkloadIdentityError`` -- propagate RAW out of ``run_wave``, and a rejection at a
poll, a results read or a list did the same; a later wave on the same engine issued its own doomed create. Now every
control-plane call runs under ``_guard``: the rejection is the typed ``WaveCredentialRejected`` (a ``WaveHalt``: never
auto-resubmitted; the one builder ``credential_halt`` the live engine shares), latched on the engine so a later ``run_wave``
raises before any request, said once. ``register_edit_precheck.create_or_stop`` stops the sequential pre-check at the first
rejection instead of recording ``api_error`` per record. Offline: the client is the fake."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import anthropic
import httpx2
import pytest

from conftest import make_engine
from corpus_generator.batch import Ledger, WaveCredentialRejected, WaveHalt, WaveRequest, credential_halt
from fakes import FakeBatchClient, federation_rejected, live_unauthorized, overloaded, text_result


def reqs(n: int) -> list[WaveRequest]:
    return [WaveRequest(custom_id=f"r{i:03d}", params={"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": f"prompt {i}"}]})
            for i in range(n)]


def echo(cid, params, attempt):
    return text_result(cid, f"answer to {cid}")


def _raising(client: FakeBatchClient, method: str, exc: BaseException, *, after: int = 0) -> dict:
    """Make the client's ``method`` (create / retrieve / results / list) raise ``exc`` from its (after+1)-th call on."""
    real = getattr(client, f"_{method}")
    counts = {"n": 0}

    def wrapped(*a, **kw):
        counts["n"] += 1
        if counts["n"] > after:
            raise exc
        return real(*a, **kw)

    setattr(client.messages.batches, method, wrapped)
    return counts


def test_credential_halt_is_the_one_builder_and_classifies_the_federated_shape(monkeypatch):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    h = credential_halt(federation_rejected(401))
    assert isinstance(h, WaveCredentialRejected) and isinstance(h, WaveHalt)
    assert (h.status, h.error_type, h.api_error_type, h.mode) == (401, "WorkloadIdentityError", None, "wif")
    assert "the federation ids file / the VM identity" in str(h) and "invalid_grant" not in str(h)
    h2 = credential_halt(live_unauthorized(403))
    assert (h2.status, h2.error_type, h2.api_error_type) == (403, "PermissionDeniedError", "permission_error")
    for not_this in (federation_rejected(503), federation_rejected(None), overloaded(), ValueError("x")):
        assert credential_halt(not_this) is None, not_this
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "a-fourth-mode")
    assert credential_halt(live_unauthorized(401)).mode == "unknown"           # a refusal message never itself refuses


@pytest.mark.parametrize("exc", [live_unauthorized(401), live_unauthorized(403), federation_rejected(401)])
def test_a_rejection_at_the_batch_create_is_the_typed_halt_latched_on_the_engine_with_nothing_stamped(tmp_path, monkeypatch, exc):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    client = FakeBatchClient(echo)
    counts = _raising(client, "create", exc)
    events: list[str] = []
    eng = make_engine(client, on_event=events.append)
    with pytest.raises(WaveCredentialRejected) as ei:
        eng.run_wave(tmp_path / "w", reqs(6), wave_name="w")
    e = ei.value
    assert e.status == getattr(exc, "status_code", None) and e.mode == "wif" and "prompt 0" not in str(e)
    assert counts["n"] == 1                                                   # one create: the first rejection is the last request
    assert not (tmp_path / "w" / "STAMP.json").exists() and not (tmp_path / "w" / "raw_results.jsonl").exists()
    states = Ledger(tmp_path / "w" / "ledger.jsonl").chunk_states()
    assert all(st.state == "journaled" and st.batch_id is None for st in states.values())   # journaled, never `submitted`
    assert sum("CREDENTIAL REJECTED" in m for m in events) == 1
    # latched: a second wave on the same engine issues NOTHING and raises the same object
    with pytest.raises(WaveCredentialRejected) as again:
        eng.run_wave(tmp_path / "w2", reqs(3), wave_name="w2")
    assert again.value is e and counts["n"] == 1 and not (tmp_path / "w2" / "ledger.jsonl").exists()


def test_a_rejection_at_a_poll_a_results_read_or_a_list_is_the_same_typed_halt(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "op")
    # the poll: the create succeeded, the retrieve is rejected
    client = FakeBatchClient(echo)
    counts = _raising(client, "retrieve", live_unauthorized(401))
    with pytest.raises(WaveCredentialRejected) as ei:
        make_engine(client).run_wave(tmp_path / "p", reqs(4), wave_name="p")
    assert ei.value.status == 401 and "the pool's vault item" in str(ei.value) and counts["n"] == 1
    states = Ledger(tmp_path / "p" / "ledger.jsonl").chunk_states()
    assert all(st.state == "submitted" for st in states.values())            # submitted stands: a re-run re-polls it
    assert not (tmp_path / "p" / "STAMP.json").exists()
    # the results read
    client = FakeBatchClient(echo)
    counts = _raising(client, "results", federation_rejected(403))
    with pytest.raises(WaveCredentialRejected) as ei:
        make_engine(client).run_wave(tmp_path / "h", reqs(4), wave_name="h")
    assert ei.value.error_type == "WorkloadIdentityError" and counts["n"] == 1
    assert not list((tmp_path / "h").glob("results-*.jsonl")) and not (tmp_path / "h" / "STAMP.json").exists()
    # the list (the reconciliation of a journaled-terminal chunk after a transport failure at create)
    client = FakeBatchClient(echo)
    client.create_script = ["transport_no_create"]
    counts = _raising(client, "list", live_unauthorized(401))
    with pytest.raises(WaveCredentialRejected):
        make_engine(client).run_wave(tmp_path / "l", reqs(4), wave_name="l")
    assert counts["n"] == 1
    # NOT this: a 429 at create is still retried by the create loop, and a 529 at a poll is still raised as the SDK raised it
    client = FakeBatchClient(echo)
    client.create_script = ["429"]
    out = make_engine(client).run_wave(tmp_path / "ok", reqs(2), wave_name="ok")
    assert len(out) == 2 and (tmp_path / "ok" / "STAMP.json").exists()
    client = FakeBatchClient(echo)
    _raising(client, "retrieve", overloaded())
    with pytest.raises(anthropic.APIStatusError):
        make_engine(client).run_wave(tmp_path / "s", reqs(2), wave_name="s")


def _list_answering_a_lazy_page(client: FakeBatchClient, exc: BaseException) -> dict:
    """Make the client's ``list`` ANSWER iterate like the SDK's SyncPage: the first page's entries, then -- when the next page is
    fetched -- ``exc``. The call itself succeeds; only the iteration meets the rejection."""
    real = client.messages.batches.list
    counts = {"calls": 0, "next_page_fetches": 0}

    def listing(*a, **kw):
        counts["calls"] += 1
        first_page = list(real(*a, **kw))

        def page():
            yield from first_page
            counts["next_page_fetches"] += 1
            raise exc
        return page()

    client.messages.batches.list = listing
    return counts


@pytest.mark.parametrize("exc", [live_unauthorized(401), federation_rejected(403)])
def test_a_rejection_while_the_batch_list_fetches_its_next_page_is_the_same_typed_halt_latched_on_the_engine(tmp_path, monkeypatch, exc):
    """``_guard`` wrapped the ``batches.list(...)`` CALL, but the SDK's page fetches its later pages lazily
    inside ``__iter__`` -- a 401 / 403 on the second page escaped ``_list_window`` raw, nothing latched, and another attempt on the
    same engine sent requests again. Now the iteration runs under the same stop (``_guard_iter``), and so does a results stream."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    client = FakeBatchClient(echo)
    client.register_foreign_batch(4)                   # the first page has an entry: the rejection is met on the way to the second
    client.create_script = ["transport_no_create"]     # the create's transport failure sends the chunk to reconciliation, which lists
    counts = _list_answering_a_lazy_page(client, exc)
    events: list[str] = []
    eng = make_engine(client, on_event=events.append)
    with pytest.raises(WaveCredentialRejected) as ei:
        eng.run_wave(tmp_path / "w", reqs(4), wave_name="w")
    assert ei.value.status == exc.status_code and ei.value.__cause__ is exc and ei.value.mode == "wif"
    assert counts == {"calls": 1, "next_page_fetches": 1} and eng._credential_rejected is ei.value
    assert sum("CREDENTIAL REJECTED" in m for m in events) == 1
    assert not (tmp_path / "w" / "STAMP.json").exists()
    with pytest.raises(WaveCredentialRejected) as again:                       # latched: a later wave issues NOTHING
        eng.run_wave(tmp_path / "w2", reqs(2), wave_name="w2")
    assert again.value is ei.value and counts["calls"] == 1
    # the results STREAM is read under the same stop: a rejection met while consuming it is the typed halt, no results file
    client = FakeBatchClient(echo)
    real_results = client.messages.batches.results

    def results(bid):
        it = real_results(bid)
        yield next(it)
        raise exc
    client.messages.batches.results = results
    eng = make_engine(client)
    with pytest.raises(WaveCredentialRejected):
        eng.run_wave(tmp_path / "h", reqs(3), wave_name="h")
    # no results file under ANY name: not the final `results-<chunk>.jsonl` and not the `.tmp` the stream was being written into
    # (the `.tmp` of a rejected stream once survived; `results-*.jsonl` did not match it)
    assert eng._credential_rejected is not None and not list((tmp_path / "h").glob("results-*")), sorted(p.name for p in (tmp_path / "h").iterdir())


def _results_failing_after_the_first_entry(client: FakeBatchClient, exc: BaseException) -> None:
    real_results = client.messages.batches.results

    def results(bid):
        it = real_results(bid)
        yield next(it)
        raise exc
    client.messages.batches.results = results


@pytest.mark.parametrize("exc, raised", [(live_unauthorized(401), WaveCredentialRejected), (federation_rejected(403), WaveCredentialRejected),
                                         (overloaded(), anthropic.APIStatusError)])
def test_a_results_stream_that_fails_after_its_first_entry_leaves_no_tmp_file_behind_whatever_the_failure(tmp_path, monkeypatch, exc, raised):
    """``_harvest`` streams INTO ``results-<chunk>.jsonl.tmp`` and publishes by rename after the whole set validated; a failure met while
    consuming the stream -- the typed credential halt, or a 5xx raised as the SDK raised it -- must remove the tmp, or a truncated
    partial sits beside the wave under a name a later run overwrites but a reader of the directory cannot tell from a result."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    client = FakeBatchClient(echo)
    _results_failing_after_the_first_entry(client, exc)
    eng = make_engine(client)
    with pytest.raises(raised):
        eng.run_wave(tmp_path / "h", reqs(3), wave_name="h")
    left = sorted(p.name for p in (tmp_path / "h").iterdir())
    assert not [n for n in left if n.startswith("results-")], left                           # neither results-*.jsonl nor results-*.jsonl.tmp
    assert not (tmp_path / "h" / "STAMP.json").exists() and not (tmp_path / "h" / "raw_results.jsonl").exists()
    assert (eng._credential_rejected is not None) == (raised is WaveCredentialRejected)        # a 529 is not a credential stop
    # the happy path still publishes by rename: the final file, no tmp
    client = FakeBatchClient(echo)
    out = make_engine(client).run_wave(tmp_path / "ok", reqs(3), wave_name="ok")
    assert len(out) == 3 and len(list((tmp_path / "ok").glob("results-*.jsonl"))) == 1
    assert not list((tmp_path / "ok").glob("*.tmp"))


# --- scripts/archive_cancelled_batches.py: an operator's sequential archive of a cancelled run's in-flight batches

def _archive_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "archive_cancelled_batches.py"
    spec = importlib.util.spec_from_file_location("archive_cancelled_batches_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _ArchiveClient:
    """A batch client for the archive script: ``retrieve`` answers an ENDED batch with the SDK's request_counts shape; ``results``
    streams ``entries`` -- unless ``results_exc`` is set (raised AT the results call) or ``stream_exc`` (raised after the first entry)."""

    def __init__(self, entries, *, results_exc=None, stream_exc=None, retrieve_exc=None, total=None):
        self.entries, self.results_exc, self.stream_exc, self.retrieve_exc = list(entries), results_exc, stream_exc, retrieve_exc
        self.total = len(self.entries) if total is None else total
        self.calls: list[str] = []
        outer = self

        class _Counts:
            def model_dump(self_inner):
                return {"processing": 0, "succeeded": outer.total, "errored": 0, "canceled": 0, "expired": 0}

        class _Batches:
            def retrieve(self_inner, bid):
                outer.calls.append(f"retrieve {bid}")
                if outer.retrieve_exc is not None:
                    raise outer.retrieve_exc
                return type("B", (), {"id": bid, "processing_status": "ended", "request_counts": _Counts(), "created_at": "c", "ended_at": "e"})()

            def results(self_inner, bid):
                outer.calls.append(f"results {bid}")
                if outer.results_exc is not None:
                    raise outer.results_exc
                return outer._stream()

        self.messages = type("M", (), {"batches": _Batches()})()

    def _stream(self):
        for i, e in enumerate(self.entries):
            yield e
            if i == 0 and self.stream_exc is not None:
                raise self.stream_exc


def _cancelled_run(tmp_path: Path, mod, bid: str = "msgbatch_synthetic_0001") -> Path:
    """A run dir whose one ROOT wave has a chunk SUBMITTED as ``bid`` (the ledger shape the script reads)."""
    run_dir = tmp_path / "data" / "runs" / "run-synthetic"
    wave = run_dir / "advice" / "a1"; wave.mkdir(parents=True)
    Ledger(wave / "ledger.jsonl").append({"chunk": "c000", "state": "journaled", "attempt": 1, "nonce": "n"})
    Ledger(wave / "ledger.jsonl").append({"chunk": "c000", "state": "submitted", "attempt": 1, "nonce": "n", "batch_id": bid})
    mod.repo_root = lambda: tmp_path
    return run_dir


@pytest.mark.parametrize("where, exc", [("results", live_unauthorized(401)), ("results", federation_rejected(403)),
                                        ("stream", live_unauthorized(403)), ("stream", federation_rejected(401)), ("retrieve", live_unauthorized(401))])
def test_the_archive_script_stops_typed_at_a_credential_rejection_and_leaves_no_archive_under_any_name(tmp_path, monkeypatch, capsys, where, exc):
    """The script opened the FINAL ``archived-results-<batch>.jsonl`` BEFORE the ``results`` request, so a
    rejection there -- or while consuming the stream -- left an empty / truncated archive under the final name and no CANCELLED.json;
    and it had no credential stop at all (a raw traceback). Now the stream goes to a ``.tmp`` published by rename after the whole set
    validated, every failure removes the tmp, and a rejection ends the script with the typed halt as its terminal line."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    mod = _archive_script()
    run_dir = _cancelled_run(tmp_path, mod)
    entries = [text_result("r000", "a"), text_result("r001", "b"), text_result("r002", "c")]
    client = _ArchiveClient(entries, **{f"{where}_exc": exc})
    mod.build_client = lambda kind: client

    def _no_poll(seconds):
        raise AssertionError("the script must not poll past a rejection")
    monkeypatch.setattr(mod.time, "sleep", _no_poll)
    with pytest.raises(SystemExit) as ei:
        mod.main("run-synthetic", "synthetic reason")
    msg = str(ei.value)
    assert "the API rejected the credential" in msg and f"(HTTP {exc.status_code}" in msg and "SISTEER_SECRETS_MODE=wif" in msg
    assert "answer to" not in msg and "prompt" not in msg
    wave = run_dir / "advice" / "a1"
    left = sorted(p.name for p in wave.iterdir())
    assert left == ["ledger.jsonl"], left                                                     # no archived-results-*.jsonl, no .tmp
    assert not (run_dir / "CANCELLED.json").exists()
    out = capsys.readouterr().out
    assert out.count("CREDENTIAL REJECTED at ") == 1 and client.calls == (["retrieve msgbatch_synthetic_0001"] if where == "retrieve"
                                                                            else ["retrieve msgbatch_synthetic_0001", "results msgbatch_synthetic_0001"])
    # the ledger is untouched (state=submitted stands: the operator re-runs the archive after fixing the credential)
    assert Ledger(wave / "ledger.jsonl").chunk_states()["c000"].state == "submitted"


def test_the_archive_script_publishes_a_complete_archive_by_rename_and_refuses_a_short_stream(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "op")
    mod = _archive_script()
    run_dir = _cancelled_run(tmp_path, mod)
    entries = [text_result("r000", "a"), text_result("r001", "b")]
    # the stream ends early (2 of 3 the counts promised): nothing published, the refusal names the batch and the counts
    client = _ArchiveClient(entries, total=3)
    mod.build_client = lambda kind: client
    with pytest.raises(SystemExit) as ei:
        mod.main("run-synthetic", "synthetic reason")
    assert "carried 2 entries but request_counts sum to 3" in str(ei.value)
    wave = run_dir / "advice" / "a1"
    assert sorted(p.name for p in wave.iterdir()) == ["ledger.jsonl"] and not (run_dir / "CANCELLED.json").exists()
    # the happy path: the final file appears by rename with every entry, the tmp is gone, CANCELLED.json names it
    client = _ArchiveClient(entries)
    mod.build_client = lambda kind: client
    mod.main("run-synthetic", "synthetic reason")
    out = wave / "archived-results-msgbatch_synthetic_0001.jsonl"
    assert out.exists() and not list(wave.glob("*.tmp"))
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["custom_id"] for r in rows] == ["r000", "r001"] and all(r["batch_id"] == "msgbatch_synthetic_0001" for r in rows)
    marker = json.loads((run_dir / "CANCELLED.json").read_text())
    assert marker["state"] == "CANCELLED" and marker["archived_batches"]["advice/a1/c000"]["n_results"] == 2
    # a 5xx at the results call is raised as the SDK raised it (not a credential stop), and still leaves nothing under the final name
    (run_dir / "CANCELLED.json").unlink(); out.unlink()
    client = _ArchiveClient(entries, results_exc=overloaded())
    mod.build_client = lambda kind: client
    with pytest.raises(anthropic.APIStatusError):
        mod.main("run-synthetic", "synthetic reason")
    assert sorted(p.name for p in wave.iterdir()) == ["ledger.jsonl"]
    assert "CREDENTIAL REJECTED" not in capsys.readouterr().out


def _real_client_whose_second_list_page_answers(status: int):
    """A REAL ``anthropic.Anthropic`` over a mock transport: ``GET /v1/messages/batches?limit=100`` answers one full page with
    ``has_more`` (the SDK's SyncPage shape), and the SDK's own fetch of the NEXT page (``after_id=<last_id>``) answers ``status``.
    Placeholder key; no network."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    def batch(i: int) -> dict:
        return {"id": f"msgbatch_{i:04d}", "type": "message_batch", "processing_status": "ended", "created_at": now,
                "expires_at": now, "ended_at": now, "cancel_initiated_at": None, "archived_at": None, "results_url": None,
                "request_counts": {"processing": 0, "succeeded": 4, "errored": 0, "canceled": 0, "expired": 0}}
    hits: list[str] = []

    def handler(request):
        hits.append(str(request.url))
        if "after_id" in str(request.url):
            api_type = {401: "authentication_error", 403: "permission_error"}.get(status, "api_error")
            return httpx2.Response(status, json={"type": "error", "error": {"type": api_type, "message": "synthetic"}}, request=request)
        return httpx2.Response(200, json={"data": [batch(i) for i in range(100)], "has_more": True,
                                          "first_id": "msgbatch_0000", "last_id": "msgbatch_0099"}, request=request)
    client = anthropic.Anthropic(api_key="placeholder-not-a-key", base_url="https://api.anthropic.com", max_retries=0,
                                 http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    return client, hits


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_second_page_through_the_sdks_own_pagination_is_the_typed_halt(tmp_path, monkeypatch, status):
    """A reproduction through the REAL SDK: the list answers one page, the SyncPage fetches the next INSIDE the for-loop
    (``get_next_page``: a new request), and the API rejects THAT request -- before, the raw ``AuthenticationError`` escaped
    ``_list_window`` (measured: 2 requests, nothing latched, no event); a 500 on the second page is still raised as the SDK raised it."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "op")
    client, hits = _real_client_whose_second_list_page_answers(status)
    events: list[str] = []
    eng = make_engine(client, on_event=events.append)
    with pytest.raises(WaveCredentialRejected) as ei:
        eng._list_window(None)
    cause = ei.value.__cause__
    assert ei.value.status == status and isinstance(cause, anthropic.APIStatusError) and cause.status_code == status
    assert len(hits) == 2 and "after_id=msgbatch_0099" in hits[1] and eng._credential_rejected is ei.value
    assert sum("CREDENTIAL REJECTED" in m for m in events) == 1 and "the pool's vault item" in str(ei.value)
    with pytest.raises(WaveCredentialRejected):
        eng.run_wave(tmp_path / "w", reqs(2), wave_name="w")                   # latched: no request
    assert len(hits) == 2
    # NOT this: a 500 on the second page is raised as the SDK raised it, nothing latched
    client, hits = _real_client_whose_second_list_page_answers(500)
    eng = make_engine(client)
    with pytest.raises(anthropic.InternalServerError):
        eng._list_window(None)
    assert eng._credential_rejected is None and len(hits) == 2


def test_the_register_edit_precheck_stops_at_the_first_rejection_instead_of_recording_api_error_per_record(monkeypatch, capsys):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    path = Path(__file__).resolve().parents[1] / "scripts" / "register_edit_precheck.py"
    spec = importlib.util.spec_from_file_location("register_edit_precheck_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Client:
        def __init__(self, exc):
            self.exc, self.calls = exc, 0
            outer = self

            class _M:
                def create(self, **params):
                    outer.calls += 1
                    raise outer.exc
            self.messages = _M()

    client = Client(federation_rejected(401))
    with pytest.raises(SystemExit) as ei:
        mod.create_or_stop(client, {"model": "m"}, custom_id="synthetic-id", n_so_far=2)
    assert "WorkloadIdentityError (HTTP 401)" in str(ei.value) and "SISTEER_SECRETS_MODE=wif" in str(ei.value) and client.calls == 1
    out = capsys.readouterr().out
    assert "CREDENTIAL REJECTED at synthetic-id" in out and "stopping after 2 result(s)" in out
    # a non-credential failure is still the record's api_error (None), as before
    client = Client(overloaded())
    assert mod.create_or_stop(client, {"model": "m"}, custom_id="x", n_so_far=0) is None
    assert "API error" in capsys.readouterr().out
    # the script's own main still wires the helper (a source pin: the inline except that recorded every record is gone)
    src = path.read_text()
    assert "create_or_stop(client, params" in src and 'results.append((req.custom_id, "api_error"))' in src
    assert json.dumps  # the module imports json for its output; keeps the import honest


# ------------------------------------------------------------------- round 9: sibling engines, one client


def test_a_rejection_latches_on_the_shared_client_so_a_sibling_engine_issues_nothing(tmp_path, monkeypatch):
    """`corpusgen run --corpus all`, the extension sub-DAGs and `onpolicy score` build one WaveEngine per thread over ONE client
    (pipeline.make_engine). The latch was per ENGINE, so after the first engine's rejection every sibling issued its own doomed
    create (one request per corpus / extension / config). The first rejection now latches on the CLIENT as well: a sibling
    engine -- one that has not started, or one mid-wave at its next control-plane call -- refuses before any request."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    client = FakeBatchClient(echo)
    counts = _raising(client, "create", live_unauthorized(401))
    events: list[str] = []
    a, b = make_engine(client, on_event=events.append), make_engine(client, on_event=events.append)
    with pytest.raises(WaveCredentialRejected) as first:
        a.run_wave(tmp_path / "a", reqs(3), wave_name="a")
    assert counts["n"] == 1
    with pytest.raises(WaveCredentialRejected) as again:                            # the sibling: before any request
        b.run_wave(tmp_path / "b", reqs(3), wave_name="b")
    assert counts["n"] == 1 and not (tmp_path / "b").exists()
    assert again.value is first.value and again.value.status == 401
    # a third engine mid-wave stops at its next control-plane call, a retrieve included
    c = make_engine(client, on_event=events.append)
    retrieves = _raising(client, "retrieve", overloaded())
    with pytest.raises(WaveCredentialRejected):
        c._guard(lambda: client.messages.batches.retrieve("msgbatch_synthetic"))
    assert retrieves["n"] == 0
    assert sum("CREDENTIAL REJECTED" in m for m in events) == 1                     # said once, by the engine that met it
    # another client is untouched: the latch is the client's, not the process's
    other = FakeBatchClient(echo)
    other_counts = _raising(other, "create", live_unauthorized(403))
    with pytest.raises(WaveCredentialRejected) as third:
        make_engine(other).run_wave(tmp_path / "o", reqs(2), wave_name="o")
    assert other_counts["n"] == 1 and third.value.status == 403 and third.value is not first.value


def test_the_corpus_credential_exit_ends_a_scripts_main_with_the_typed_line(monkeypatch):
    """The member's twin of sisteer.vault.credential_exit (miniwave's main, the CLI group): the typed halt, or the SDK's own
    rejection out of a single call, becomes SystemExit with the typed text; anything else passes."""
    from corpus_generator.batch import credential_exit

    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    with pytest.raises(SystemExit) as ei:
        with credential_exit("[miniwave] "):
            raise live_unauthorized(401)
    assert str(ei.value).startswith("[miniwave] ") and "401" in str(ei.value) and "SISTEER_SECRETS_MODE=wif" in str(ei.value)
    with pytest.raises(SystemExit) as ei2:
        with credential_exit():
            raise credential_halt(federation_rejected(403))
    assert "WorkloadIdentityError (HTTP 403)" in str(ei2.value)
    with pytest.raises(anthropic.APIStatusError):
        with credential_exit():
            raise overloaded()
    with credential_exit():
        pass


def test_the_cli_group_ends_every_command_typed_on_a_credential_rejection(monkeypatch):
    from click.testing import CliRunner

    from corpus_generator import cli

    monkeypatch.setenv("SISTEER_SECRETS_MODE", "op")
    assert isinstance(cli.main, cli.CredentialExitGroup)                              # the real group is the typed one
    g = cli.CredentialExitGroup()

    @g.command("synthetic-rejected")
    def _rejected():
        raise live_unauthorized(401)

    @g.command("synthetic-other")
    def _other():
        raise overloaded()

    r = CliRunner().invoke(g, ["synthetic-rejected"])
    assert r.exit_code == 1 and "CREDENTIAL REJECTED" not in r.output and "401" in r.output and "the pool's vault item" in r.output, r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)                  # the typed line, not a traceback
    r2 = CliRunner().invoke(g, ["synthetic-other"])
    assert r2.exit_code == 1 and isinstance(r2.exception, anthropic.APIStatusError)   # anything else passes as before
