"""The AlphaSteer grader's judge-shedding canary classifies a credential rejection as the run's STOP.

``grade_alphasteer_f22`` re-issues a grade pass whose items came back UNSCORED (a 529-shaped shed on the judge's org makes
every in-flight item 'terminal' for the per-item helper): before re-issuing it probes the judge with ONE canary request on
the same org, waits a shed out, and otherwise counts an attempt and re-issues. Previously the canary read a 401 /
403 -- or the federation identity refused at the token exchange, a ``WorkloadIdentityError`` -- as ``other:<class>``, and the
loop then slept and called the grader AGAIN on the rejected credential: one more wave of doomed requests, as many as the
grader's concurrency. The canary now classifies the credential FIRST through the vault's one classifier, holds the typed
rejection on the grader (whose next pass then issues nothing) and raises it -- the ``except RuntimeError`` re-issue loop
never catches it (the typed error is not a RuntimeError and never says UNSCORED) -- and the driver's main ends with the
typed line (``vault.credential_exit``). The shedding semantics are unchanged: a 529 / 5xx is still 'overloaded', any other
failure still 'other:<class>'."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import anthropic
import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "alphasteer_f22"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import grade_alphasteer_f22 as gaf  # noqa: E402
import grade_bon_live  # noqa: E402
from sisteer import vault  # noqa: E402
from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader  # noqa: E402

JUDGE = "claude-haiku-4-5"


# ------------------------------------------------------------------------------------------------------------- the shapes


def credential_error(status: int = 401):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if status == 401:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "synthetic"}}
        return anthropic.AuthenticationError(f"Error code: 401 - {body}", response=httpx.Response(401, request=req), body=body)
    body = {"type": "error", "error": {"type": "permission_error", "message": "synthetic"}}
    return anthropic.PermissionDeniedError(f"Error code: 403 - {body}", response=httpx.Response(403, request=req), body=body)


def federation_rejected(status: int = 401):
    return anthropic.WorkloadIdentityError(f"Token exchange failed (HTTP {status}): synthetic", status_code=status,
                                           body={"error": "invalid_grant", "error_description": "synthetic"})


def status_error(cls, status: int):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(f"Error code: {status} - synthetic", response=httpx.Response(status, request=req), body=None)


class _Client:
    """A vault-stamped stand-in: ``messages.create`` raises ``exc_factory()`` (or answers when it is None) and counts."""

    def __init__(self, exc_factory=None, pool: str = "lowprio"):
        self._exc, self.calls, self._lock = exc_factory, 0, threading.Lock()
        self.messages = self
        self.sisteer_pool_stamp = {"pool": pool, "auth": "wif", "ids_sha12": "0" * 12}

    def create(self, **kwargs):
        with self._lock:
            self.calls += 1
        if self._exc is not None:
            raise self._exc()
        return object()


class _Inner:
    """The grader's grade pass as the loop sees it: raises UNSCORED ``n_unscored`` times, then answers."""

    def __init__(self, n_unscored: int = 1):
        self.calls, self.n_unscored = 0, n_unscored

    def __call__(self, items):
        self.calls += 1
        if self.calls <= self.n_unscored:
            raise RuntimeError(f"{len(items)} judge item(s) UNSCORED after the bounded per-item retries -- synthetic")
        return {it: 0.0 for it in items}


def _grader(tmp_path, client) -> LiveStrongREJECTGrader:
    return LiveStrongREJECTGrader(model=JUDGE, concurrency=4, cache_path=tmp_path / "cache.jsonl", client=client, allow_unscored=0)


def _loop(inner, probe, *, sleep, shed_events=None, retry_events=None, named=None):
    return gaf.make_grade_fn(inner, probe=probe, shed_wait_cap=10800, shed_events=shed_events if shed_events is not None else [],
                             unscored_retry_events=retry_events if retry_events is not None else [],
                             name_unscored=(named.append if named is not None else (lambda items: None)),
                             judge_model=JUDGE, org="lowprio", sleep=sleep)


ITEMS = [("d", "rev", f"synthetic goal {i}", f"synthetic reply {i}") for i in range(5)]


# -------------------------------------------------------------------------------------------- the stop at the canary


@pytest.mark.parametrize("exc", [lambda: credential_error(401), lambda: credential_error(403), lambda: federation_rejected(401)],
                         ids=["401", "403", "federation-401"])
def test_a_rejection_at_the_canary_is_the_run_stop_held_on_the_grader_and_nothing_is_re_issued(exc, tmp_path, monkeypatch):
    canary = _Client(exc)
    monkeypatch.setattr(grade_bon_live, "make_client", lambda *a, **k: canary)      # the canary's one client, on the same org
    gclient = _Client(exc)                                                             # the grader's: must NOT be called again
    g = _grader(tmp_path, gclient)
    inner, slept, retry_events, shed_events, named = _Inner(n_unscored=99), [], [], [], []
    grade_fn = _loop(inner, lambda: gaf.judge_shedding("lowprio", JUDGE, grader=g), sleep=slept.append,
                     shed_events=shed_events, retry_events=retry_events, named=named)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        grade_fn(ITEMS)
    e = ei.value
    assert inner.calls == 1, inner.calls                                               # the UNSCORED pass that triggered the probe; NO re-issue
    assert canary.calls == 1 and gclient.calls == 0
    assert e.pool == "lowprio" and e.status == int(getattr(exc(), "status_code", 401))
    assert not isinstance(e, RuntimeError) and "UNSCORED" not in str(e) and "synthetic goal" not in str(e)
    assert slept == [] and retry_events == [] and shed_events == [] and named == []    # no attempt counted, no wait, no unscored file
    assert g.credential_rejected is e                                                  # held on the grader
    with pytest.raises(vault.JudgeCredentialRejected) as again:                        # the grader's next pass: the same object, before any call
        g.grade(ITEMS)
    assert again.value is e and gclient.calls == 0


def test_the_canary_alone_raises_typed_and_holds_the_rejection_even_without_a_grader(monkeypatch):
    canary = _Client(lambda: credential_error(403))
    monkeypatch.setattr(grade_bon_live, "make_client", lambda *a, **k: canary)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        gaf.judge_shedding("lowprio", JUDGE)
    assert ei.value.pool == "lowprio" and ei.value.status == 403 and canary.calls == 1


def test_the_canary_still_reads_a_shed_as_overloaded_and_any_other_failure_as_other(monkeypatch):
    def probe(exc):
        c = _Client(exc)
        monkeypatch.setattr(grade_bon_live, "make_client", lambda *a, **k: c)
        return gaf.judge_shedding("lowprio", JUDGE)

    assert probe(None) == "ok"
    assert probe(lambda: status_error(anthropic.InternalServerError, 529)) == "overloaded"
    assert probe(lambda: status_error(anthropic.InternalServerError, 503)) == "overloaded"
    assert probe(lambda: status_error(anthropic.NotFoundError, 404)) == "other:NotFoundError:404"
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    assert probe(lambda: anthropic.APIConnectionError(request=req)) == "other:APIConnectionError"
    assert probe(lambda: anthropic.WorkloadIdentityError("Failed to reach token endpoint: synthetic")) == "other:WorkloadIdentityError"   # no status: not a rejection


def test_the_bounded_loop_waits_out_a_shed_and_re_issues_only_the_unscored_pass(monkeypatch):
    """The pre-existing semantics, pinned: 'overloaded' is waited out without consuming an attempt; 'ok' / 'other' count
    an attempt and re-issue; the pass answers on the third call."""
    states = iter(["overloaded", "other:NotFoundError:404"])
    inner, slept, retry_events, shed_events = _Inner(n_unscored=2), [], [], []
    grade_fn = _loop(inner, lambda: next(states), sleep=slept.append, shed_events=shed_events, retry_events=retry_events)
    assert grade_fn(ITEMS) == {it: 0.0 for it in ITEMS}
    assert inner.calls == 3
    assert [e["kind"] for e in shed_events] == ["wait"] and shed_events[0]["pause_s"] == 60 and shed_events[0]["n_unscored"] == 5
    assert [e["attempt"] for e in retry_events] == [1] and retry_events[0]["judge_probe"] == "other:NotFoundError:404"
    assert slept == [60, 10]                                                            # the shed wait, then the attempt's back-off


def test_the_loop_names_the_unscored_items_and_re_raises_after_the_bounded_attempts():
    inner, named, slept = _Inner(n_unscored=99), [], []
    grade_fn = _loop(inner, lambda: "ok", sleep=slept.append, named=named)
    with pytest.raises(RuntimeError, match="UNSCORED"):
        grade_fn(ITEMS)
    assert inner.calls == 6 and named == [ITEMS] and slept == [10, 20, 30, 40, 50]


# ------------------------------------------------------------------------------------------- the grader's hold, the main


def test_the_graders_hold_is_a_check_and_set_and_the_held_rejection_stops_the_next_pass(tmp_path):
    client = _Client(lambda: credential_error(401))
    g = _grader(tmp_path, client)
    assert g.credential_rejected is None
    first = vault.JudgeCredentialRejected("lowprio", 401, "AuthenticationError", api_error_type="authentication_error")
    second = vault.JudgeCredentialRejected("lowprio", 403, "PermissionDeniedError")
    assert g.hold_credential_rejection(first) is first
    assert g.hold_credential_rejection(second) is first and g.credential_rejected is first      # the first held wins
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        g.grade(ITEMS)
    assert ei.value is first and client.calls == 0                                      # before any wire call


def test_the_driver_classifies_through_the_vault_and_its_main_ends_typed():
    src = (REPO / "scripts" / "alphasteer_f22" / "grade_alphasteer_f22.py").read_text()
    assert "def judge_shedding(" in src and "def make_grade_fn(" in src
    assert "vault.credential_rejection(" in src and "hold_credential_rejection(" in src
    canary = src[src.index("def judge_shedding("):src.index("def make_grade_fn(")]
    assert canary.index("vault.credential_rejection(") < canary.index("APIStatusError"), "the credential is classified FIRST"
    tail = src[src.index('if __name__ == "__main__":'):]
    assert "vault.credential_exit(" in tail and "main()" in tail
