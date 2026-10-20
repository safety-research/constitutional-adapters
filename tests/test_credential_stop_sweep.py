"""The WIF driver sweep (2026-10-13): every Claude-calling path stops at the FIRST credential rejection, through ONE shared stop.

Several drivers each met a 401 / 403 -- or, under
``SISTEER_SECRETS_MODE=wif``, the federation identity refused at the token exchange (the SDK's ``WorkloadIdentityError``) -- and
went on issuing requests: judge_sonnet's queue, the misalign judge's, the corpus generator's live engine, then
judge_overrefusal's ``pool.map`` and build_probe_set's ``_call_api``. Finding-by-finding did not converge, so the rule
now lives ONCE in ``sisteer.vault`` -- ``CredentialLatch`` (classify every failure for the credential first, hold the first
rejection, raise it before every later wire call), ``cancel_queue`` (cancel the queued futures, wait only for the running ones),
``map_until_credential_rejected`` (a bounded executor with the stop built in) and ``credential_exit`` (the typed terminal line for
a driver's main) -- and this module proves each driver the sweep found wanting is built on it:

* ``scripts/judge_overrefusal.py`` -- the executor drained every queued item through its own doomed request after the first 401;
* ``scripts/grokmt_analyze.py`` -- a 401 became the item label ``ERROR:AuthenticationError``, WRITTEN into the cache a re-run read
  back, every queued item issued, and the read folded the error labels into the refusal fractions;
* ``sisteer.eval.attacks.llm_rewriter`` -- ``batch`` propagated the first rejection while the executor's exit drained the queue;
* ``sisteer.capeval.judging.measure_token_bounds`` -- the first wire traffic of a judged run swallowed the rejection as
  "count-tokens unavailable" and fell back to the byte bound while ``ex.map`` drained every count; ``_liveness_canary`` read it as
  "not serving";
* ``sisteer.eval.anthropic_wif`` -- inspect retries RETRYABLE statuses only and runs under ``fail_on_error=False``, so a rejected
  federation identity was one sample's hard error and the next scenario issued its own doomed request;
* ``scripts/capeval_judged_run.py`` -- the typed rejection out of the judge phase ended as a traceback.

Every client here is a stand-in or a REAL SDK client over an offline mock token endpoint; nothing reaches the network, and no key
exists anywhere (placeholder ids; ``SISTEER_SECRETS_MODE=wif`` where the federated shape is under test)."""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import threading
import time
import types

import anthropic
import httpx
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from sisteer import vault  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------------------- the shapes


def credential_error(status: int = 401):
    """The API's credential rejection on a request (401 authentication_error / 403 permission_error): body shape only."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if status == 401:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "synthetic"}}
        return anthropic.AuthenticationError(f"Error code: 401 - {body}", response=httpx.Response(401, request=req), body=body)
    body = {"type": "error", "error": {"type": "permission_error", "message": "synthetic"}}
    return anthropic.PermissionDeniedError(f"Error code: 403 - {body}", response=httpx.Response(403, request=req), body=body)


def federation_rejected(status=401):
    """The SDK's federated rejection: the token exchange refused the identity -- an AnthropicError, NOT an APIStatusError, with a
    redacted body that is an OAuth error string as often as the API's error object. ``status`` None = an unreachable endpoint."""
    if status is None:
        return anthropic.WorkloadIdentityError("Failed to reach token endpoint: synthetic")
    return anthropic.WorkloadIdentityError(f"Token exchange failed (HTTP {status}): synthetic", status_code=status,
                                           body={"error": "invalid_grant", "error_description": "synthetic"})


def overloaded():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.InternalServerError("Error code: 529 - synthetic", response=httpx.Response(529, request=req),
                                         body={"type": "error", "error": {"type": "overloaded_error", "message": "synthetic"}})


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Msg:
    stop_reason = "end_turn"

    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, owner):
        self._o = owner

    def create(self, **kw):
        return self._o._call(kw)

    def parse(self, **kw):
        return self._o._call(kw)

    def count_tokens(self, **kw):
        return self._o._call(kw)


class FakeClient:
    """Duck-types a vault-built client: ``.messages.{create,parse,count_tokens}``, a call counter, a pool stamp (the
    placeholder is not a key), ``fail(n)`` deciding the n-th call's fate (None = an answer of ``text``), ``delay`` keeping
    every worker in flight long enough for the stop to race the queue."""

    def __init__(self, fail, *, delay: float = 0.0, text: str = "C", org: str = "lowprio", stamp: bool = True):
        self.fail = fail
        self.delay = delay
        self.text = text
        self.calls = 0
        self.answered = 0           # calls that RETURNED an answer (after the delay): a fake rejection can wait on it (see the grokmt test)
        self.lock = threading.Lock()
        if stamp:
            setattr(self, vault.POOL_STAMP_ATTR, {"pool": org, "auth": "wif", "ids_sha12": "0" * 12})

    def _call(self, kw):
        with self.lock:
            self.calls += 1
            n = self.calls
        if self.delay:
            time.sleep(self.delay)
        exc = self.fail(n)
        if exc is not None:
            raise exc
        with self.lock:
            self.answered += 1
        return _Msg(self.text)

    def rejecting_after_answers(self, n_answers: int, status: int = 401, *, timeout_s: float = 5.0):
        """A ``fail`` for this client: the first ``n_answers`` calls answer; every later call is the credential rejection, raised
        only AFTER those answers have RETURNED -- so ``as_completed`` yields the answers before the rejection whatever the node's
        thread scheduling (round 7: with every call sleeping the same 30 ms, the third starter's rejection could wake before the
        second answer on a loaded compute node, and a pin on the answers flushed before the stop read one short)."""
        def fail(n):
            if n <= n_answers:
                return None
            deadline = time.time() + timeout_s
            while self.answered < n_answers and time.time() < deadline:
                time.sleep(0.001)
            return credential_error(status)
        self.fail = fail
        return self

    @property
    def messages(self):
        return _Messages(self)


# -------------------------------------------------------------------------------------------------------- the vault's stop


def test_the_classifier_and_the_pool_label():
    assert vault.is_credential_rejection(credential_error(401)) and vault.is_credential_rejection(credential_error(403))
    assert vault.is_credential_rejection(federation_rejected(401)) and vault.is_credential_rejection(federation_rejected(403))
    for not_this in (federation_rejected(None), federation_rejected(503), overloaded(),
                     anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
                     RuntimeError("401 in a message is not a status")):
        assert not vault.is_credential_rejection(not_this), not_this
    # round 12 on fd06740ae: an ALREADY-TYPED rejection classifies as what it records -- the federated clients' latched retry layer
    # raises the held typed error from its per-attempt check into a driver's `except Exception` ladder, which classifies the
    # credential first and must read it as the rejection it is (tests/test_wif_federated_destination_and_sdk_retries.py)
    typed = vault.JudgeCredentialRejected("lowprio", 401, "AuthenticationError", api_error_type="authentication_error")
    assert vault.is_credential_rejection(typed) and vault.credential_rejection(typed) == vault.CredentialRejection(401, "AuthenticationError", "authentication_error")
    assert vault.pool_of_client(FakeClient(lambda n: None, org="batch")) == "batch"
    assert vault.pool_of_client(FakeClient(lambda n: None, stamp=False)) == "unknown" and vault.pool_of_client(None) == "unknown"


def test_the_latch_holds_the_first_rejection_once_and_raises_it_before_every_later_call(monkeypatch):
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    latch = vault.CredentialLatch(lambda: "lowprio")
    assert latch.held is None and latch.rejection(overloaded()) is None and latch.held is None
    latch.check()                                                             # nothing held: a no-op
    first = latch.rejection(federation_rejected(401))
    assert isinstance(first, vault.JudgeCredentialRejected) and latch.held is first
    assert (first.pool, first.status, first.error_type, first.api_error_type) == ("lowprio", 401, "WorkloadIdentityError", None)
    assert "WorkloadIdentityError (HTTP 401)" in str(first) and "wif" in str(first) and "the federation ids file / the VM identity" in str(first)
    assert latch.rejection(credential_error(403)) is first                    # check-and-set: the FIRST filed is the run's
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        latch.check()
    assert ei.value is first
    # the guard: checked before, classified after; a non-credential failure passes through as is
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        if x == "boom":
            raise ValueError("boom")
        return x * 2

    fresh = vault.CredentialLatch("batch")
    g = fresh.guard(fn)
    assert g(2) == 4
    with pytest.raises(ValueError):
        g("boom")
    assert fresh.held is None
    rejecting = fresh.guard(lambda x: (_ for _ in ()).throw(credential_error(403)))
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        rejecting(1)
    assert ei.value.pool == "batch" and ei.value.status == 403 and fresh.held is ei.value
    with pytest.raises(vault.JudgeCredentialRejected) as again:
        g(5)                                                                  # held: the guard raises BEFORE fn runs
    assert again.value is ei.value and calls["n"] == 2
    # a pool callable that fails resolves to `unknown`, never a refusal of the message
    broken = vault.CredentialLatch(lambda: (_ for _ in ()).throw(RuntimeError("no stamp")))
    assert broken.rejection(credential_error(401)).pool == "unknown"
    # a driver's own class (callers that catch their error family) is minted instead
    from sisteer.capeval import judging
    typed = vault.CredentialLatch("lowprio", cls=judging.JudgeCredentialRejected)
    assert isinstance(typed.rejection(credential_error(401)), judging.JudgingError)


def test_map_until_credential_rejected_stops_at_the_first_rejection_cancels_the_queue_and_flushes_as_it_goes(capsys):
    client = FakeClient(lambda n: None if n <= 3 else credential_error(401), delay=0.03)
    latch = vault.CredentialLatch(lambda: vault.pool_of_client(client))
    seen = []
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        vault.map_until_credential_rejected(lambda i: client.messages.create(i=i).content[0].text, list(range(40)), workers=3,
                                            latch=latch, label="[t]", on_result=lambda item, r: seen.append(item))
    assert ei.value is latch.held and ei.value.pool == "lowprio"
    assert 4 <= client.calls <= 3 + 3, client.calls                         # the three answers, the rejection, at most the in-flight siblings
    assert len(seen) == 3                                                     # the results that arrived were flushed as they came
    out = capsys.readouterr().out
    assert out.count("[t]: CREDENTIAL REJECTED after ") == 1 and "queued request(s) cancelled" in out and "(D6)" in out
    assert "synthetic" not in out                                             # counts and the typed text only
    # the order and the completion path: every item in order, on_result per item
    ok = FakeClient(lambda n: None, text="x")
    got = vault.map_until_credential_rejected(lambda i: f"{i}:{ok.messages.create().content[0].text}", [3, 1, 2], workers=2,
                                              latch=vault.CredentialLatch("p"))
    assert got == ["3:x", "1:x", "2:x"] and ok.calls == 3
    # any OTHER exception out of an item cancels the queue too (never a drain at exit) and propagates as is
    slow = FakeClient(lambda n: ValueError("item") if n == 1 else None, delay=0.03)
    with pytest.raises(ValueError):
        vault.map_until_credential_rejected(lambda i: slow.messages.create(), list(range(40)), workers=2, latch=vault.CredentialLatch("p"))
    time.sleep(0.2)
    assert slow.calls <= 4, slow.calls


def test_cancel_queue_never_waits_on_a_cancelled_future():
    import concurrent.futures as cf

    gate = threading.Event()
    ex = cf.ThreadPoolExecutor(max_workers=1)
    futs = [ex.submit(gate.wait, 5)] + [ex.submit(lambda: 1) for _ in range(20)]
    t0 = time.monotonic()
    gate.set()
    counts = vault.cancel_queue(ex, futs)
    assert time.monotonic() - t0 < 4
    assert counts["n_requested"] == 21 and counts["n_cancelled"] + counts["n_settled"] == 21 and counts["n_cancelled"] >= 1
    assert counts["n_failed"] == 0


def test_credential_exit_turns_a_rejection_into_the_typed_terminal_line_and_passes_everything_else(monkeypatch):
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    with pytest.raises(SystemExit) as ei:
        with vault.credential_exit("lowprio", prefix="[x] "):
            raise vault.JudgeCredentialRejected("lowprio", 401, "WorkloadIdentityError")
    assert str(ei.value).startswith("[x] the API rejected the lowprio pool's credential: WorkloadIdentityError (HTTP 401")
    with pytest.raises(SystemExit) as ei:
        with vault.credential_exit(lambda: "batch", prefix="[y] "):
            raise credential_error(403)                                       # the SDK's own, out of a single synchronous call
    assert "[y] the API rejected the batch pool's credential: PermissionDeniedError (HTTP 403, permission_error)" in str(ei.value)
    with pytest.raises(SystemExit) as ei:
        with vault.credential_exit("batch"):
            raise federation_rejected(401)
    assert "WorkloadIdentityError (HTTP 401)" in str(ei.value) and "synthetic" not in str(ei.value)
    for passes in (overloaded(), federation_rejected(None), ValueError("x"), KeyboardInterrupt()):
        with pytest.raises(type(passes)):
            with vault.credential_exit("lowprio"):
                raise passes
    with vault.credential_exit("lowprio"):
        pass                                                                  # a clean run exits nothing


# ------------------------------------------------------------------------------------------------------ judge_overrefusal


def _overrefusal(monkeypatch, client):
    import sisteer.eval.api_keys as api_keys
    import judge_overrefusal as jo

    monkeypatch.setattr(api_keys, "make_client", lambda org, **kw: client)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return jo


def _records(n: int):
    return [{"prompt": f"synthetic prompt {i}", "response": f"synthetic response {i}", "dataset": "d", "subset": "s"} for i in range(n)]


def test_judge_overrefusal_stops_at_the_first_rejection_cancels_the_queue_and_caches_nothing_for_it(tmp_path, monkeypatch, capsys):
    """Previously, `one` caught the 429 / 5xx classes only, a 401 propagated out of `pool.map`, and the
    executor's `with` exit WAITED FOR THE WHOLE QUEUE -- every queued item issued its own rejected request. Now the run stops
    within one request per in-flight worker, the queue is cancelled, nothing is cached and no summary is written."""
    client = FakeClient(lambda n: credential_error(401), delay=0.03)
    jo = _overrefusal(monkeypatch, client)
    facts: dict = {}
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        jo.judge_all(_records(60), tmp_path / "judge_cache.jsonl", concurrency=4, run_facts=facts)
    e = ei.value
    assert (e.pool, e.status, e.error_type, e.api_error_type) == ("lowprio", 401, "AuthenticationError", "authentication_error")
    assert 1 <= client.calls <= 4, client.calls                               # never 60
    assert facts["pool"] == "lowprio"                                         # the routing record was filled before the first call
    assert (tmp_path / "judge_cache.jsonl").read_text() == ""                 # a rejection is the credential's outcome, never a label
    out = capsys.readouterr().out
    assert out.count("[judge]: CREDENTIAL REJECTED after 0/60 result(s)") == 1 and "synthetic" not in out
    # the federated shape through the same path, and a verdict obtained before the stop stays cached
    client = FakeClient(lambda n: None if n == 1 else federation_rejected(403), delay=0.03, text="I")
    jo = _overrefusal(monkeypatch, client)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        jo.judge_all(_records(60), tmp_path / "cache2.jsonl", concurrency=3)
    assert (ei.value.status, ei.value.error_type) == (403, "WorkloadIdentityError") and client.calls <= 1 + 3
    rows = [json.loads(l) for l in (tmp_path / "cache2.jsonl").read_text().splitlines() if l.strip()]
    assert [r["label"] for r in rows] == ["I"]
    # NOT this: the exchange's 5xx and an unreachable token endpoint are retried, then the verdict stands
    for transient in (federation_rejected(503), federation_rejected(None), overloaded()):
        client = FakeClient(lambda n, exc=transient: exc if n == 1 else None, text="P")
        jo = _overrefusal(monkeypatch, client)
        labels = jo.judge_all(_records(2), tmp_path / f"cache_{id(transient)}.jsonl", concurrency=2)
        assert set(labels.values()) == {"P"} and client.calls == 3


def test_judge_overrefusal_main_ends_with_the_typed_line_and_writes_no_summary_or_run_record(tmp_path, monkeypatch, capsys):
    client = FakeClient(lambda n: credential_error(401), delay=0.02)
    jo = _overrefusal(monkeypatch, client)
    cell = tmp_path / "overrefusal" / "si0"
    cell.mkdir(parents=True)
    (cell / "baseline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _records(30)))
    monkeypatch.setattr(sys, "argv", ["judge_overrefusal.py", "--dir", str(cell), "--concurrency", "3"])
    with pytest.raises(SystemExit) as ei:
        jo.main()
    msg = str(ei.value)
    assert msg.startswith("[judge] the API rejected the lowprio pool's credential: AuthenticationError (HTTP 401, authentication_error)")
    assert "synthetic" not in msg and client.calls <= 3
    assert not (cell / "overrefusal_summary.json").exists() and not (cell / "overrefusal_judge_run.json").exists()
    assert (cell / "judge_cache.jsonl").read_text() == ""
    # the happy path through main still writes both (and the module-level vault import the closing line needs is in place)
    ok = FakeClient(lambda n: None, text="C")
    jo = _overrefusal(monkeypatch, ok)
    jo.main()
    assert json.loads((cell / "overrefusal_summary.json").read_text()) and json.loads((cell / "overrefusal_judge_run.json").read_text())["pool"] == "lowprio"
    assert "pool lowprio (auth wif ids sha12" in capsys.readouterr().out


# -------------------------------------------------------------------------------------------------------- grokmt_analyze


def _grokmt(monkeypatch, client):
    import grokmt_analyze as ga

    monkeypatch.setattr(vault, "anthropic_client", lambda org="lowprio", **kw: client)
    return ga


def test_grokmt_classifier_stops_at_the_first_rejection_and_never_caches_an_error_label_for_it(tmp_path, monkeypatch, capsys):
    """Before: `one` caught EVERY exception as the label `ERROR:<type>`, wrote it to the cache (a re-run read it back as a
    verdict), `ex.map` issued every queued item, and the read counted the error labels as non-compliance -- a plausible
    number from a rejected credential. Now: the typed stop, the queue cancelled, nothing cached for the rejection."""
    # the two answers return before any rejection is raised (rejecting_after_answers): the flushed-before-the-stop count is exact
    client = FakeClient(None, delay=0.03, text="REFUSE").rejecting_after_answers(2)
    ga = _grokmt(monkeypatch, client)
    items = [(f"k{i}", f"synthetic reply {i}") for i in range(50)]
    facts: dict = {}
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        ga._judge_items(items, model="m", concurrency=4, cache_path=tmp_path / "labels.jsonl", run_facts=facts)
    assert ei.value.pool == "lowprio" and ei.value.status == 401 and facts["org"] == "lowprio"
    assert 3 <= client.calls <= 2 + 4, client.calls                           # the two verdicts, the rejection, the in-flight siblings; never 50
    rows = [json.loads(l) for l in (tmp_path / "labels.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 2 and {r["label"] for r in rows} == {"REFUSE"}        # the verdicts obtained before the stop; no ERROR row
    assert not any(r["label"].startswith("ERROR") for r in rows)
    out = capsys.readouterr().out
    assert out.count("[judge]: CREDENTIAL REJECTED after ") == 1 and "/50 result(s)" in out and "synthetic reply" not in out
    # the federated shape is the same stop; a non-credential failure is still the item's ERROR label, as before
    client = FakeClient(lambda n: federation_rejected(401))
    ga = _grokmt(monkeypatch, client)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        ga._judge_items(items[:5], model="m", concurrency=2, cache_path=tmp_path / "l2.jsonl")
    assert ei.value.error_type == "WorkloadIdentityError" and client.calls <= 2
    client = FakeClient(lambda n: ValueError("malformed") if n == 1 else None, text="COMPLY")
    ga = _grokmt(monkeypatch, client)
    labels = ga._judge_items(items[:3], model="m", concurrency=1, cache_path=tmp_path / "l3.jsonl")
    assert sorted(labels.values()) == ["COMPLY", "COMPLY", "ERROR:ValueError"]


def test_grokmt_main_ends_with_the_typed_line_and_writes_no_analysis(tmp_path, monkeypatch):
    import grokmt_analyze as ga

    rejected = vault.JudgeCredentialRejected("lowprio", 403, "PermissionDeniedError", api_error_type="permission_error")
    monkeypatch.setattr(ga, "refusal_read", lambda *a, **k: (_ for _ in ()).throw(rejected))
    out = tmp_path / "analysis.json"
    goals = tmp_path / "goals.txt"
    goals.write_text("g1\n")
    monkeypatch.setattr(sys, "argv", ["grokmt_analyze.py", "--run-dir", str(tmp_path), "--goal-ids-file", str(goals), "--out", str(out), "--judge"])
    with pytest.raises(SystemExit) as ei:
        ga.main()
    assert str(ei.value) == f"[judge] {rejected}" and not out.exists()


# ----------------------------------------------------------------------------------------------------------- llm_rewriter


def test_the_cached_concurrent_rewriter_stops_its_batch_at_the_first_rejection_and_freezes_no_corpus(tmp_path, monkeypatch, capsys):
    from sisteer.eval.attacks.llm_rewriter import _CachedConcurrentRewriter, build_llm_corpus

    class Rejecting(_CachedConcurrentRewriter):
        def __init__(self, fail, **kw):
            super().__init__(**kw)
            self.client = FakeClient(fail, delay=0.03, text="rewritten")
            self._client = self.client                 # the pool label reads the client's stamp, as AnthropicRewriter's does

        def _call_one(self, prompt, system):
            return self.client.messages.create(prompt=prompt, system=system).content[0].text

    rw = Rejecting(lambda n: None if n <= 2 else credential_error(401), model="m", cache_path=tmp_path / "rw.jsonl", concurrency=4)
    items = [(f"synthetic instruction {i}", "sys") for i in range(40)]
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        rw.batch(items)
    e = ei.value
    assert e.pool == "lowprio" and e.status == 401 and rw.credential_rejected is e
    assert 3 <= rw.client.calls <= 2 + 4, rw.client.calls                     # never 40
    rows = [json.loads(l) for l in (tmp_path / "rw.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 2                                                     # the two rewrites obtained before the stop, nothing for the rejection
    out = capsys.readouterr().out
    assert out.count("[rewriter m]: CREDENTIAL REJECTED after ") == 1 and "/40 result(s)" in out and "synthetic instruction" not in out
    # latched: a later one()/batch() on the same rewriter issues nothing and raises the same object
    n = rw.client.calls
    with pytest.raises(vault.JudgeCredentialRejected) as again:
        rw.one("another synthetic instruction", system="sys")
    assert again.value is e and rw.client.calls == n
    # the build writes no corpus and no manifest over a stopped batch
    goals = [types.SimpleNamespace(goal_id=f"g{i}", text=f"goal {i}") for i in range(6)]
    rw2 = Rejecting(lambda n: credential_error(403), model="m", cache_path=tmp_path / "rw2.jsonl", concurrency=2)
    out_path = tmp_path / "corpus.jsonl"
    with pytest.raises(vault.JudgeCredentialRejected):
        build_llm_corpus(goals, rewriter=rw2, plan_fn=lambda g: [("v", "sys", f"instr {g.goal_id}")], out_path=out_path,
                         fingerprint=set(), provenance={}, family_name="synthetic")
    assert not out_path.exists() and not pathlib.Path(str(out_path) + ".manifest.json").exists()
    # a non-credential failure still raises loudly and uncached, as before (and now cancels the queue behind it)
    rw3 = Rejecting(lambda n: ValueError("bad") if n == 1 else None, model="m", cache_path=tmp_path / "rw3.jsonl", concurrency=2)
    with pytest.raises(ValueError):
        rw3.batch(items[:10])
    assert rw3.credential_rejected is None


# ---------------------------------------------------------------------------------------- capeval.judging: bounds + canary


def test_the_count_tokens_bounds_stop_at_the_first_rejection_latch_the_client_and_never_fall_back_on_it(tmp_path, monkeypatch, capsys):
    from sisteer.capeval import judging as j
    from test_capeval_judging import ORBENCH_RUBRIC_V1, _single_req

    pairs = [(_single_req(f"synthetic prompt {i}", f"synthetic response {i}"), ORBENCH_RUBRIC_V1) for i in range(40)]
    client = FakeClient(lambda n: credential_error(401), delay=0.03)
    with pytest.raises(j.JudgeCredentialRejected) as ei:
        j.measure_token_bounds(pairs, client=client, max_tokens=512, count_workers=4)
    e = ei.value
    assert isinstance(e, j.JudgingError) and isinstance(e, vault.JudgeCredentialRejected) and e.pool == "lowprio" and e.status == 401
    assert client.calls <= 4, client.calls                                    # never one count per rendered request
    assert getattr(client, j.CREDENTIAL_LATCH_ATTR) is e                     # the judge over this client refuses before its first call
    out = capsys.readouterr().out
    assert "count-tokens: CREDENTIAL REJECTED" in out and "falling back to the byte bound" not in out
    # a judge built over that client stops before any wire call
    judge = j.ClaudeJudge(cache=j.JudgeCache(tmp_path / "c.sqlite"), clients={"lowprio": client}, org="lowprio")
    with pytest.raises(j.JudgeCredentialRejected) as again:
        judge._send("lowprio", {"model": "m", "max_tokens": 1, "messages": []})
    assert again.value is e and client.calls <= 4
    # the federated shape; and every OTHER failure still falls back to the byte bound, labeled
    with pytest.raises(j.JudgeCredentialRejected):
        j.measure_token_bounds(pairs[:3], client=FakeClient(lambda n: federation_rejected(403)), max_tokens=512, count_workers=2)
    bounds = j.measure_token_bounds(pairs[:3], client=FakeClient(lambda n: overloaded()), max_tokens=512, count_workers=2)
    assert "CONSERVATIVE" in bounds["input_bound_method"] and "falling back to the byte bound" in capsys.readouterr().out


def test_the_liveness_canary_raises_the_typed_rejection_instead_of_reading_it_as_not_serving(tmp_path):
    from sisteer.capeval import judging as j

    client = FakeClient(lambda n: credential_error(403))
    judge = j.ClaudeJudge(cache=j.JudgeCache(tmp_path / "c.sqlite"), clients={"lowprio": client}, org="lowprio")
    with pytest.raises(j.JudgeCredentialRejected) as ei:
        judge._liveness_canary()
    e = ei.value
    assert e.status == 403 and e.pool == "lowprio" and judge._credential_rejected is e and getattr(client, j.CREDENTIAL_LATCH_ATTR) is e
    assert client.calls == 1 and judge.n_canary_calls == 1
    with pytest.raises(j.JudgeCredentialRejected) as again:
        judge._liveness_canary()                                              # latched: no wire call
    assert again.value is e and client.calls == 1
    # NOT this: a 529 on the canary is still "not serving"
    judge2 = j.ClaudeJudge(cache=j.JudgeCache(tmp_path / "c2.sqlite"), clients={"lowprio": FakeClient(lambda n: overloaded())}, org="lowprio")
    assert judge2._liveness_canary() is False and judge2._credential_rejected is None


# ------------------------------------------------------------------------------------------------------ capeval_judged_run


def test_capeval_judged_run_ends_with_the_typed_line_when_the_judge_phase_is_rejected(monkeypatch):
    import capeval_judged_run as cjr
    from sisteer.capeval import judging as j

    rejected = j.JudgeCredentialRejected("lowprio", 401, "WorkloadIdentityError")
    monkeypatch.setattr(cjr, "cmd_judge", lambda args: (_ for _ in ()).throw(rejected))
    task = sorted(cjr.TASKS)[0]
    with pytest.raises(SystemExit) as ei:
        cjr.main(["judge", "--task", task, "--sweep", "synthetic"])
    assert str(ei.value) == f"[capeval-judged] {rejected}"
    monkeypatch.setattr(cjr, "cmd_judge", lambda args: 0)
    assert cjr.main(["judge", "--task", task, "--sweep", "synthetic"]) == 0


# --------------------------------------------------------------------------------------- the federated inspect provider


IDS = {"ANTHROPIC_FEDERATION_RULE_ID_GCP": "fr_gcp_55555555", "ANTHROPIC_ORGANIZATION_ID": "org_11111111",
       "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_33333333", "ANTHROPIC_WORKSPACE_ID": "wrkspc_22222222"}


def _exchange_rejecting_credentials(status: int):
    """REAL SDK credentials whose token exchange answers ``status`` -- offline (a mock transport for /v1/oauth/token);
    ``hits`` counts the exchanges. The SDK retries a 401 exchange once (a 403 never), so one request costs <= 2."""
    try:
        import httpx2 as _httpx            # anthropic >= 1.x is built on httpx2; 0.x on httpx -- the credentials take the SDK's own
    except ImportError:
        _httpx = httpx
    hits = {"n": 0}

    def token_endpoint(request):
        assert request.url.path.endswith("/v1/oauth/token"), request.url
        hits["n"] += 1
        return _httpx.Response(status, json={"error": "invalid_grant", "error_description": "synthetic"}, request=request)

    creds = anthropic.WorkloadIdentityCredentials(
        identity_token_provider=lambda: "header.payload.signature", federation_rule_id=IDS["ANTHROPIC_FEDERATION_RULE_ID_GCP"],
        organization_id=IDS["ANTHROPIC_ORGANIZATION_ID"], service_account_id=IDS["ANTHROPIC_SERVICE_ACCOUNT_ID"],
        workspace_id=IDS["ANTHROPIC_WORKSPACE_ID"], http_client=_httpx.Client(transport=_httpx.MockTransport(token_endpoint)))
    return creds, hits


def test_the_federated_inspect_provider_clients_share_one_latch_the_first_rejection_as_the_sdk_raised_it_and_no_request_after(
        tmp_path, monkeypatch):
    """inspect retries RETRYABLE statuses only and runs its eval under fail_on_error=False: a rejected federation identity was one
    sample's hard error and the next scenario's auditor call issued its own doomed request -- one per launched scenario. Now every
    federated client the factory builds wraps its ``messages`` resource: the first rejection any of them meets is re-raised AS the
    SDK raised it (inspect's classification of that call is unchanged) and latched; every later create / stream / count_tokens of
    every federated client raises the typed error BEFORE its wire call (no exchange, no request)."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.model._providers import anthropic as prov
    from sisteer.eval import anthropic_wif as w

    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    creds, hits = _exchange_rejecting_credentials(401)
    monkeypatch.setattr(vault, "wif_credentials", lambda org, **kw: creds)
    real_cls = prov.AsyncAnthropic
    assert getattr(real_cls, "__sisteer_wif_real__", None) is None, "a stale install leaked from another test"
    try:
        w.install_inspect_wif("lowprio")
        assert w.credential_rejected() is None
        c1 = prov.AsyncAnthropic(api_key="sentinel", base_url="http://127.0.0.1:1/gateway", max_retries=0)
        c2 = prov.AsyncAnthropic(api_key="sentinel", max_retries=0)
        assert isinstance(c1, real_cls) and c1.credentials is creds and isinstance(c1.messages, w._LatchedMessages)
        assert c1.messages.batches is not None                               # other attributes forward to the SDK's resource
        req = dict(model="claude-sonnet-5", max_tokens=1, messages=[{"role": "user", "content": "ping"}])
        # the FIRST rejection: the SDK's own exception (the exchange refused the identity), now latched for the process
        with pytest.raises(anthropic.WorkloadIdentityError) as ei:
            asyncio.run(c1.messages.create(**req))
        assert ei.value.status_code == 401 and not isinstance(ei.value, anthropic.APIStatusError) and 1 <= hits["n"] <= 2
        held = w.credential_rejected()
        assert isinstance(held, vault.JudgeCredentialRejected) and held.pool == "lowprio" and held.error_type == "WorkloadIdentityError"
        n = hits["n"]
        # every later request of EVERY federated client: the typed error, no exchange, no request
        with pytest.raises(vault.JudgeCredentialRejected) as again:
            asyncio.run(c2.messages.create(**req))
        assert again.value is held

        async def _stream():
            async with c2.messages.stream(**req):
                pass

        with pytest.raises(vault.JudgeCredentialRejected):
            asyncio.run(_stream())
        with pytest.raises(vault.JudgeCredentialRejected):
            asyncio.run(c1.messages.count_tokens(model="claude-sonnet-5", messages=[{"role": "user", "content": "ping"}]))
        assert hits["n"] == n
        # a client built AFTER the latch is held is stopped the same way
        c3 = prov.AsyncAnthropic(api_key="sentinel", max_retries=0)
        with pytest.raises(vault.JudgeCredentialRejected):
            asyncio.run(c3.messages.create(**req))
        assert hits["n"] == n
    finally:
        assert w.uninstall_inspect_wif() is True
    assert w.credential_rejected() is None and prov.AsyncAnthropic is real_cls


def test_the_latched_messages_resource_classifies_a_request_level_rejection_and_passes_the_rest():
    from sisteer.eval import anthropic_wif as w

    class Inner:
        def __init__(self, fail):
            self.fail, self.calls = fail, 0

        async def create(self, **kw):
            self.calls += 1
            exc = self.fail(self.calls)
            if exc is not None:
                raise exc
            return "answer"

        async def count_tokens(self, **kw):
            return await self.create(**kw)

        def stream(self, **kw):
            raise AssertionError("not used here")

        batches = "the batches resource"

    latch = vault.CredentialLatch("lowprio")
    inner = Inner(lambda n: overloaded() if n == 1 else credential_error(403) if n == 2 else None)
    m = w._LatchedMessages(inner, latch)
    assert m.batches == "the batches resource"
    with pytest.raises(anthropic.InternalServerError):
        asyncio.run(m.create())                                               # not a rejection: as the SDK raised it, nothing latched
    assert latch.held is None
    with pytest.raises(anthropic.PermissionDeniedError):
        asyncio.run(m.create())                                               # the first rejection: as the SDK raised it, now latched
    assert latch.held is not None and latch.held.status == 403
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        asyncio.run(m.count_tokens())                                         # a later call: the typed error, no call into the resource
    assert ei.value is latch.held and inner.calls == 2
    other = w._LatchedMessages(Inner(lambda n: credential_error(401)), latch)
    with pytest.raises(vault.JudgeCredentialRejected):
        asyncio.run(other.create())                                           # the process-wide latch covers every client
    assert other._inner.calls == 0


def test_the_first_rejection_aborts_every_running_inspect_task_once_through_inspects_own_cancel_handle(tmp_path, monkeypatch):
    """The latch alone left inspect walking every remaining scenario as an instant typed error under
    fail_on_error=False, every later eval group launching, and the runner FILING the scenarios judged before the rejection. Now the
    process's FIRST rejection fires inspect's own task-cancel handle ("abort": in-flight samples interrupted, queued ones never
    started, completed ones kept, the task never retried by the eval_set) for every task running in the process -- once; a later
    rejection fires nothing more; a task already finished or already cancelling is skipped, named; the abort record says what it
    did; the uninstall clears it."""
    pytest.importorskip("inspect_ai")
    from inspect_ai._control import eval_state as es
    from inspect_ai._display.core.display import TaskCancel
    from inspect_ai.model._providers import anthropic as prov
    from sisteer.eval import anthropic_wif as w

    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    creds, hits = _exchange_rejecting_credentials(401)
    monkeypatch.setattr(vault, "wif_credentials", lambda org, **kw: creds)
    fired = []

    def handle(task_id):
        tc = TaskCancel(can_retry=False, cancel_task=lambda _t: None)

        def cancel(kind, tc=tc, task_id=task_id):
            fired.append((task_id, kind)); tc.cancel_type = kind             # what the runner's own closure does (run.py cancel_task)
        tc.cancel_task = cancel
        return tc

    es.clear_all_eval_states()
    try:
        es.register_eval("eval-a", 4, task="bloom_audit", task_id="task-a", task_cancel=handle("task-a"))
        es.register_eval("eval-b", 4, task="bloom_audit", task_id="task-b", task_cancel=handle("task-b"))
        done = es.register_eval("eval-c", 4, task="bloom_audit", task_id="task-c", task_cancel=handle("task-c"))
        done.completed_at = 1.0                                               # a task already finished: nothing to abort
        w.install_inspect_wif("lowprio")
        assert w.inspect_abort_result() is None
        req = dict(model="claude-sonnet-5", max_tokens=1, messages=[{"role": "user", "content": "ping"}])
        c1 = prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, max_retries=0)
        with pytest.raises(anthropic.WorkloadIdentityError):
            asyncio.run(c1.messages.create(**req))                           # the FIRST rejection: as the SDK raised it -- and the abort
        assert sorted(fired) == [("task-a", "abort"), ("task-b", "abort")]
        res = w.inspect_abort_result()
        assert sorted(res["aborted"]) == ["task-a", "task-b"] and res["skipped"] == ["task-c: not running"] and res["errors"] == []
        assert "rejected the lowprio pool's credential" in res["reason"]
        c2 = prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, max_retries=0)
        with pytest.raises(vault.JudgeCredentialRejected):
            asyncio.run(c2.messages.create(**req))                           # a later rejection: the typed stop, no second abort
        assert len(fired) == 2 and w.inspect_abort_result() is res
        # inspect's own gate holds too: a task whose cancel is already pending is skipped, never re-fired
        again = w.abort_inspect_tasks("again")
        assert again["aborted"] == [] and len(fired) == 2 and any("already requested" in s for s in again["skipped"])
    finally:
        es.clear_all_eval_states()
        assert w.uninstall_inspect_wif() is True
    assert w.inspect_abort_result() is None and w.credential_rejected() is None
