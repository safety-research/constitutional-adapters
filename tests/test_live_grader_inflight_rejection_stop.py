"""In-flight live graders stop at the run's first credential rejection in EVERY retry layer.

Previously the one-shot graders checked the run's latch ONCE, when a worker entered its item: ``LiveStrongREJECTGrader``'s
``_one_inner`` read the grader's field, ``grade_bon_live._grade_tasks_concurrently``'s ``_one`` read its dict. Below that check
three layers each slept between attempts and re-sent without looking -- ``sisteer.throttle.ThrottledClient``'s 429 loop (8 attempts,
up to 120 s apart), ``_forced_tool_grade``'s bounded loop (429 / 529 / 5xx / transport, up to 60 s apart), and the SDK's own
retries (``max_retries=2`` on the live grader's client) -- so a worker B asleep in any of them when worker A's request met the
first 401 / 403 woke and issued another request on the rejected credential: cancelling the queued futures cannot reach a sleeper,
and at normal grader concurrency that is as many doomed requests as there are workers in a backoff, per layer. The rule is the
FIRST rejection, so:

- the latch RIDES THE CLIENT (``vault.CLIENT_LATCH_ATTR`` / ``client_credential_latch``): the live grader's own latch is handed to its
  ``ThrottledClient`` (which attaches it to the inner client), ``_grade_tasks_concurrently`` attaches one to its bare client, and
  every layer that is handed the client -- the throttled wrapper's loop, ``_forced_tool_grade``'s loop, the BoN twin's
  ``BoundedJudgeClient`` over the same inner -- checks it BEFORE EVERY attempt, the first and each wake; the first rejection any
  layer meets is held on it (``CredentialLatch.hold`` / ``rejection``) and an already-typed rejection passes through untouched;
- the SDK's own retries are OFF on this path under wif: an SDK-internal retry sleeps and re-sends below any latch (round 12 on
  fd06740ae moved the rule into the vault -- every federated client is built with the SDK's retries off and the count asked for is
  the vault's latched layer's budget, so the grader asks for the base's count in every mode and op / env keep the SDK's own; a
  prebuilt client carrying SDK retries is refused by name under wif only);
- ``judge_overrefusal.one``'s 8-attempt loop checks its latch before every attempt too (the same shape; the sweep).

Each two-worker test below holds B in its backoff (the sleep is a wait on A's rejection being issued and filed) and asserts that B
wakes into the typed stop: exactly TWO requests -- B's 429 / 529 and A's 401 -- never a third. On the committed tree before this
round every one of them saw the third request."""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")

from sisteer import throttle, vault  # noqa: E402
from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader  # noqa: E402
import grade_bon_live  # noqa: E402

JUDGE = "claude-haiku-4-5"
STAMP = {"pool": "lowprio", "key_sha256_12": "0" * 12}
WAKE_BOUND_S = 3.0


def _api_error(cls, status: int):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(f"Error code: {status} - synthetic", response=httpx.Response(status, request=req),
               body={"error": {"type": "authentication_error" if status == 401 else "synthetic", "message": "synthetic"}})


def rate_limited():
    return _api_error(anthropic.RateLimitError, 429)


def overloaded():
    return _api_error(anthropic.OverloadedError, 529)


def rejected(status: int = 401):
    return _api_error(anthropic.AuthenticationError if status == 401 else anthropic.PermissionDeniedError, status)


class _Script:
    """A vault-stamped stand-in client whose ``messages.create`` follows a script BY CALL ORDER under a lock: call k raises
    ``script[k-1]``; a call past the script raises the LAST entry again (the committed tree's extra requests land here and
    are counted). ``issued`` is set when the first credential rejection has been raised, for the sleeper below."""

    def __init__(self, script):
        self.script, self.calls, self._lock = list(script), 0, threading.Lock()
        self.issued = threading.Event()
        self.messages = self
        self.sisteer_pool_stamp = dict(STAMP)

    def create(self, **kwargs):
        with self._lock:
            self.calls += 1
            exc = self.script[min(self.calls, len(self.script)) - 1]
        if vault.is_credential_rejection(exc):
            self.issued.set()
        raise exc


def _sleeper(client: _Script, held, calls: list, *, grace_s: float = 0.0):
    """The backoff a worker sleeps in, made deterministic: wait until the rejection has been ISSUED to a sibling and then until it
    is HELD (``held()``; plus ``grace_s`` when the latch is a local the test cannot read), bounded, then return -- the worker wakes
    exactly when a sleeper of the old code would have re-sent. Built BEFORE ``time.sleep`` is patched: it keeps the real one."""
    real_sleep = time.sleep

    def sleep(s):
        calls.append(s)
        client.issued.wait(WAKE_BOUND_S)
        t0 = time.monotonic()
        while not held() and time.monotonic() - t0 < WAKE_BOUND_S:
            real_sleep(0.005)
        if grace_s:
            real_sleep(grace_s)
    return sleep


def _items(n: int):
    return [("d", "rev", f"synthetic goal {i}", f"synthetic reply {i}") for i in range(n)]


# ------------------------------------------------------------------------------------------------ LiveStrongREJECTGrader


@pytest.mark.parametrize("first", [rate_limited, overloaded], ids=["429-in-the-throttled-wrapper", "529-in-_forced_tool_grade"])
def test_a_live_grader_worker_asleep_in_a_backoff_wakes_into_the_stop_never_into_another_request(tmp_path, monkeypatch, capsys, first):
    """Two workers, two items: B's first call is a 429 (the throttled wrapper's loop sleeps) or a 529 (``_forced_tool_grade``'s
    loop sleeps); A's is the 401. B wakes once A's rejection is held and must issue NOTHING: two requests in all, the typed error
    out of ``grade``, held on the grader, the pool named from the stamp."""
    client = _Script([first(), rejected(401)])
    g = LiveStrongREJECTGrader(model=JUDGE, concurrency=2, cache_path=tmp_path / "cache.jsonl", client=client)
    sleeps: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(client, lambda: g.credential_rejected is not None, sleeps))
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        g.grade(_items(2))
    e = ei.value
    assert client.calls == 2, f"{client.calls} requests: the sleeper re-sent after the run's first rejection"
    assert sleeps and g.credential_rejected is e and (e.pool, e.status) == ("lowprio", 401)
    assert g.credential_latch.held is e and vault.client_credential_latch(client) is g.credential_latch
    assert "CREDENTIAL REJECTED after 0/2 graded" in capsys.readouterr().out
    assert len(g._cache.cache) == 0
    with pytest.raises(vault.JudgeCredentialRejected):
        g.grade_xstest(_items(1))
    assert client.calls == 2                                                      # latched: no call on a later pass either


def test_the_live_grader_hands_its_own_latch_to_the_throttled_wrapper_and_the_wrapper_files_the_first_rejection_on_it(tmp_path, monkeypatch):
    client = _Script([rejected(403)])
    g = LiveStrongREJECTGrader(model=JUDGE, concurrency=1, cache_path=tmp_path / "cache.jsonl", client=client)
    assert isinstance(g._client, throttle.ThrottledClient) and g._client.credential_latch is g.credential_latch
    assert vault.client_credential_latch(g._client) is g.credential_latch and vault.client_credential_latch(client) is g.credential_latch
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        g.grade(_items(1))
    assert client.calls == 1 and g.credential_rejected is ei.value and (ei.value.status, ei.value.error_type) == (403, "PermissionDeniedError")
    # the canary's hold is the same check-and-set: the first held wins
    other = vault.JudgeCredentialRejected("lowprio", 401, "AuthenticationError")
    assert g.hold_credential_rejection(other) is ei.value and g.credential_rejected is ei.value


# ------------------------------------------------------------------------------------------------ the SDK's own retries


def test_the_live_grader_asks_the_vault_for_the_bases_count_in_every_mode_and_the_vault_owns_where_it_lives(tmp_path, monkeypatch):
    """Round 12 on fd06740ae: the grader asks for the BASE's count in every mode -- the default 8 clamped
    to 2, a smaller request as is -- and never refuses one; WHERE the count lives is the vault's: under op / env the SDK's own
    retries (byte-identical to the base), under wif the SDK's retries off and the count the latched layer's budget
    (tests/test_wif_federated_destination_and_sdk_retries.py measures the vault). Round 11 had refused every non-zero count and
    built with 0 in op / env too."""
    built: list = []
    stand_in = _Script([])
    monkeypatch.setattr(grade_bon_live, "make_client", lambda org, max_retries: built.append((org, max_retries)) or stand_in)
    for mode in ("op", "env", "wif"):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        built.clear()
        g = LiveStrongREJECTGrader(model=JUDGE, concurrency=4, cache_path=tmp_path / f"{mode}.jsonl")
        assert built == [("lowprio", 2)] and g._sdk_max_retries == 2 and g.run_stats()["sdk_max_retries"] == 2, mode
        for n, clamp in ((1, 1), (2, 2), (8, 2), (0, 0)):
            built.clear()
            g = LiveStrongREJECTGrader(model=JUDGE, concurrency=4, cache_path=tmp_path / f"{mode}{n}.jsonl", max_retries=n)
            assert built == [("lowprio", clamp)] and g._sdk_max_retries == clamp, (mode, n)


def test_a_prebuilt_client_carrying_sdk_retries_is_refused_by_name_under_wif_only_and_taken_as_handed_elsewhere(tmp_path, monkeypatch):
    """Under wif every client the vault builds runs the SDK's retries OFF, so a prebuilt one carrying them was built outside the
    vault and its SDK-internal retries would re-send below the run's latch: refused by name, never rebuilt. Under op / env a
    prebuilt client (an argument this lane added; no keyed-mode history) is taken as handed, retries or not (round 12)."""
    class _Retrying(_Script):
        max_retries = 5

    class _Off(_Script):
        max_retries = 0

    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    with pytest.raises(ValueError, match="under SISTEER_SECRETS_MODE=wif the prebuilt client carries SDK max_retries=5"):
        LiveStrongREJECTGrader(model=JUDGE, concurrency=2, cache_path=tmp_path / "c.jsonl", client=_Retrying([]))
    off = _Off([])
    g = LiveStrongREJECTGrader(model=JUDGE, concurrency=2, cache_path=tmp_path / "c2.jsonl", client=off)
    assert g._client._inner is off
    # a stand-in without the attribute at all (every other test's double) is taken as before
    LiveStrongREJECTGrader(model=JUDGE, concurrency=2, cache_path=tmp_path / "c3.jsonl", client=_Script([]))
    # a real SDK client with retries off is accepted under wif; with retries on it is refused -- it cannot have come from the vault
    real = anthropic.Anthropic(api_key="synthetic-placeholder-not-a-key", base_url=vault.ANTHROPIC_BASE_URL, max_retries=0)
    setattr(real, vault.POOL_STAMP_ATTR, dict(STAMP))
    assert LiveStrongREJECTGrader(model=JUDGE, concurrency=2, cache_path=tmp_path / "c4.jsonl", client=real).pool_stamp == STAMP
    retrying = anthropic.Anthropic(api_key="synthetic-placeholder-not-a-key", base_url=vault.ANTHROPIC_BASE_URL, max_retries=5)
    setattr(retrying, vault.POOL_STAMP_ATTR, dict(STAMP))
    with pytest.raises(ValueError, match="max_retries=5"):
        LiveStrongREJECTGrader(model=JUDGE, concurrency=2, cache_path=tmp_path / "c5.jsonl", client=retrying)
    # op / env: taken as handed, nothing rebuilt
    for mode in ("op", "env"):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        g = LiveStrongREJECTGrader(model=JUDGE, concurrency=2, cache_path=tmp_path / f"{mode}.jsonl", client=retrying)
        assert g._client._inner is retrying and g.pool_stamp == STAMP
        assert LiveStrongREJECTGrader(model=JUDGE, concurrency=2, cache_path=tmp_path / f"{mode}2.jsonl", client=_Retrying([]))._client._inner.max_retries == 5


# ------------------------------------------------------------------------------------------------ the throttled wrapper alone


def test_the_throttled_wrapper_checks_the_latch_before_every_attempt_and_files_the_first_rejection():
    gate = throttle.AdaptiveGate(8)
    inner = _Script([rejected(401)])
    w = throttle.ThrottledClient(inner, gate)
    latch = w.credential_latch
    assert isinstance(latch, vault.CredentialLatch) and vault.client_credential_latch(inner) is latch and vault.client_credential_latch(w) is latch
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        w.messages.create(model=JUDGE)
    assert inner.calls == 1 and latch.held is ei.value and ei.value.pool == "lowprio"       # typed here, from the stamp
    with pytest.raises(vault.JudgeCredentialRejected) as ei2:
        w.messages.create(model=JUDGE)
    assert inner.calls == 1 and ei2.value is ei.value                                        # held: no call
    assert gate.n_rate_limited == 0
    # a latch handed in is the one used, and attached to the inner for the other layers
    mine = vault.CredentialLatch("synthetic-pool")
    inner2 = _Script([rejected(401)])
    w2 = throttle.ThrottledClient(inner2, gate, latch=mine)
    assert w2.credential_latch is mine and vault.client_credential_latch(inner2) is mine
    mine.hold(vault.JudgeCredentialRejected("synthetic-pool", 403, "PermissionDeniedError"))
    with pytest.raises(vault.JudgeCredentialRejected):
        w2.messages.create(model=JUDGE)
    assert inner2.calls == 0                                                                 # held before the first attempt: no request


def test_a_wrapper_worker_asleep_in_the_429_backoff_wakes_into_a_siblings_rejection(monkeypatch):
    """The wrapper alone, two threads: B sleeps on its 429 while A's 401 is filed on the shared latch; B's wake is the typed stop."""
    gate = throttle.AdaptiveGate(8)
    inner = _Script([rate_limited(), rejected(401)])
    w = throttle.ThrottledClient(inner, gate)
    sleeps: list = []
    monkeypatch.setattr(throttle.time, "sleep", _sleeper(inner, lambda: w.credential_latch.held is not None, sleeps))
    out: dict = {}

    def worker(name):
        try:
            w.messages.create(model=JUDGE)
        except BaseException as e:  # noqa: BLE001 -- recorded
            out[name] = e

    b = threading.Thread(target=worker, args=("B",)); b.start()
    time.sleep(0.05)
    a = threading.Thread(target=worker, args=("A",)); a.start()
    a.join(10); b.join(10)
    assert inner.calls == 2, inner.calls
    assert isinstance(out["A"], vault.JudgeCredentialRejected) and out["B"] is out["A"]       # B woke into A's held rejection
    assert sleeps and gate.n_rate_limited == 1


def test_an_inner_that_refuses_attributes_still_gets_a_wrapper_held_latch():
    class _Slotted:
        __slots__ = ("messages", "sisteer_pool_stamp")

        def __init__(self):
            self.messages = self
            self.sisteer_pool_stamp = dict(STAMP)

        def create(self, **kw):
            raise rejected(401)

    inner = _Slotted()
    w = throttle.ThrottledClient(inner, throttle.AdaptiveGate(2))
    assert vault.client_credential_latch(inner) is None and vault.client_credential_latch(w) is w.credential_latch
    with pytest.raises(vault.JudgeCredentialRejected):
        w.messages.create(model=JUDGE)
    assert w.credential_latch.held is not None


# ------------------------------------------------------------------------------------------------ _forced_tool_grade


def test_forced_tool_grade_checks_the_clients_latch_before_the_first_attempt_passes_a_typed_stop_through_and_files_a_raw_one(monkeypatch):
    from judge_batch import StrongREJECTRating
    monkeypatch.setattr(time, "sleep", lambda s: None)
    sink: list = []
    # (a) a client whose latch is already held: no request, the held object, the sink untouched
    held_client = _Script([rejected(401)])
    latch = vault.client_credential_latch(held_client, create=True)
    first = vault.JudgeCredentialRejected("lowprio", 401, "AuthenticationError")
    latch.hold(first)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        grade_bon_live._forced_tool_grade(held_client, JUDGE, "p", StrongREJECTRating, on_failure=sink.append)
    assert ei.value is first and held_client.calls == 0 and sink == []
    # (b) a client whose create raises an ALREADY-TYPED rejection (a wrapper layer's): the same object, never an item's None
    typed = vault.JudgeCredentialRejected("lowprio", 403, "PermissionDeniedError")
    passthrough = _Script([typed])
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        grade_bon_live._forced_tool_grade(passthrough, JUDGE, "p", StrongREJECTRating, on_failure=sink.append)
    assert ei.value is typed and passthrough.calls == 1 and sink == []
    # (c) a client carrying an UNHELD latch whose create raises the SDK's 401: typed, and FILED on the client's latch
    raw = _Script([rejected(401)])
    latch = vault.client_credential_latch(raw, create=True)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        grade_bon_live._forced_tool_grade(raw, JUDGE, "p", StrongREJECTRating, on_failure=sink.append)
    assert latch.held is ei.value and ei.value.pool == "lowprio" and raw.calls == 1 and sink == []
    # (d) a bare client with no latch: typed after one call as before (the one-off path; nothing to file on)
    bare = _Script([rejected(401)])
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        grade_bon_live._forced_tool_grade(bare, JUDGE, "p", StrongREJECTRating)
    assert bare.calls == 1 and vault.client_credential_latch(bare) is None and ei.value.status == 401


# ------------------------------------------------------------------------------------------------ grade_bon_live's own concurrent grader


@pytest.mark.parametrize("first", [rate_limited, overloaded], ids=["429", "529"])
def test_grade_tasks_concurrently_attaches_the_latch_to_its_bare_client_and_a_sleeper_wakes_into_it(tmp_path, monkeypatch, capsys, first):
    client = _Script([first(), rejected(401)])
    cache = grade_bon_live.ScoreCache(tmp_path / "cache.jsonl", judge_model=JUDGE)
    sleeps: list = []

    def held():
        latch = vault.client_credential_latch(client)
        return latch is not None and latch.held is not None

    monkeypatch.setattr(time, "sleep", _sleeper(client, held, sleeps))
    tasks = [("sr", "g1", "synthetic goal", "synthetic reply 1"), ("sr", "g2", "synthetic goal", "synthetic reply 2")]
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        grade_bon_live._grade_tasks_concurrently(tasks, cache, client, JUDGE, 2)
    assert client.calls == 2, f"{client.calls} requests: the sleeper re-sent after the run's first rejection"
    assert sleeps and vault.client_credential_latch(client).held is ei.value and ei.value.status == 401
    assert "[grade] CREDENTIAL REJECTED after 0/2 graded" in capsys.readouterr().out
    assert len(cache.cache) == 0


# ------------------------------------------------------------------------------------------------ the BoN twin's bounded client


def _twin():
    """The seed-replicate grader (the BoN judge's BoundedJudgeClient), loaded by path as tests/test_seedrep_grader_twin.py loads it;
    its import-time environment pin is undone after (it sets the API host under op)."""
    name = "seedrep_grade_full22_native"
    if name in sys.modules:
        return sys.modules[name]
    before = os.environ.get("ANTHROPIC_BASE_URL")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "sdfbase" / "seedrep" / "grade_full22_native.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        if before is None:
            os.environ.pop("ANTHROPIC_BASE_URL", None)
        else:
            os.environ["ANTHROPIC_BASE_URL"] = before
    return mod


def test_the_bon_twins_bounded_client_checks_the_graders_latch_before_its_one_retry():
    """The BoN judge replaces the live grader's throttled wrapper with ``BoundedJudgeClient`` over the SAME raw client, which carries
    the grader's latch: the retry after its backoff checks it, so a sibling's rejection filed during the sleep stops the retry."""
    twin = _twin()
    gate = throttle.AdaptiveGate(8)
    raw = _Script([rate_limited(), rejected(401)])
    wrapper = throttle.ThrottledClient(raw, gate)                               # what the live grader does: the latch lands on the raw client
    latch = vault.client_credential_latch(raw)
    assert latch is wrapper.credential_latch
    sibling = vault.JudgeCredentialRejected("lowprio", 401, "AuthenticationError")
    bounded = twin.BoundedJudgeClient(raw, gate, sleep=lambda s: latch.hold(sibling))   # the backoff, during which a sibling's rejection is filed
    assert vault.client_credential_latch(bounded) is latch                       # the forwarding wrapper exposes the raw client's latch
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        grade_bon_live.grade_strongreject_one(bounded, JUDGE, "goal text", "response text", rubric="v2")
    assert ei.value is sibling and raw.calls == 1, raw.calls                     # the retry never fired
    assert bounded.counts["attempts"] == 1 and bounded.counts["rate_limited"] == 1 and gate.n_rate_limited == 1
    # without a held latch the twin's two-attempt contract is untouched (tests/test_seedrep_grader_twin.py pins it in full)
    raw2 = _Script([rate_limited(), rate_limited()])
    b2 = twin.BoundedJudgeClient(raw2, throttle.AdaptiveGate(8), sleep=lambda s: None)
    assert grade_bon_live.grade_strongreject_one(b2, JUDGE, "goal text", "response text", rubric="v2") is None and raw2.calls == 2


# ------------------------------------------------------------------------------------------------ judge_overrefusal (the sweep)


def test_judge_overrefusal_worker_asleep_in_its_backoff_wakes_into_the_latch(tmp_path, monkeypatch, capsys):
    import sisteer.eval.api_keys as api_keys
    import judge_overrefusal as jo
    client = _Script([rate_limited(), rejected(401)])
    monkeypatch.setattr(api_keys, "make_client", lambda org, **kw: client)
    sleeps: list = []
    # the latch is judge_all's own local: the sleeper waits for the rejection to be issued, then a grace for the sibling to file it
    monkeypatch.setattr(time, "sleep", _sleeper(client, client.issued.is_set, sleeps, grace_s=0.3))
    records = [{"prompt": f"synthetic prompt {i}", "response": f"synthetic response {i}", "dataset": "d", "subset": "s"} for i in range(2)]
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        jo.judge_all(records, tmp_path / "judge_cache.jsonl", concurrency=2)
    assert client.calls == 2, f"{client.calls} requests: the sleeper re-sent after the run's first rejection"
    assert ei.value.status == 401 and (tmp_path / "judge_cache.jsonl").read_text() == ""
    assert "[judge]: CREDENTIAL REJECTED after 0/2 result(s)" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ the vault's pieces


def test_the_vaults_latch_rides_a_client_and_hold_is_a_check_and_set():
    latch = vault.CredentialLatch("p")
    a = vault.JudgeCredentialRejected("p", 401, "AuthenticationError")
    b = vault.JudgeCredentialRejected("p", 403, "PermissionDeniedError")
    assert latch.held is None and latch.hold(a) is a and latch.hold(b) is a and latch.held is a
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        latch.check()
    assert ei.value is a
    c = _Script([])
    assert vault.client_credential_latch(c) is None
    made = vault.client_credential_latch(c, create=True)
    assert isinstance(made, vault.CredentialLatch) and vault.client_credential_latch(c) is made and vault.client_credential_latch(c, create=True) is made
    assert getattr(c, vault.CLIENT_LATCH_ATTR) is made and vault.CLIENT_LATCH_ATTR == "sisteer_credential_latch"
    assert vault.attach_credential_latch(c, vault.CredentialLatch("q")) is False and vault.client_credential_latch(c) is made   # the first attached stays
    made.rejection(rejected(401))
    assert made.held is not None and made.held.pool == "lowprio"                 # the pool resolved from the client's stamp when first filed

    class _Slotted:
        __slots__ = ()

    assert vault.attach_credential_latch(_Slotted(), latch) is False
    alone = vault.client_credential_latch(_Slotted(), create=True)
    assert isinstance(alone, vault.CredentialLatch)                              # returned to the caller, not attached
    setattr(c, vault.CLIENT_LATCH_ATTR, "not a latch")
    assert vault.client_credential_latch(c) is None                              # a foreign value under the name is not a latch
