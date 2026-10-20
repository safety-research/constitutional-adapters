"""The served clients' once-only ``max_tokens`` clamp against the serving
window (``sisteer.capeval.serving.served_window``), its two users —
``eval/attacks/served_target.py`` (the gated attacks' served target) and
``eval/attacks/bon_served.py`` (BoN + the static families) — and what the
runners do with the TYPED error. CPU-only; stub transports stand in for the
proxy; the PyRIT path runs the real pinned executors over a stub defense.

Why (the per-eval geometry ruling, 2026-09-03): at the multiturn_attacks p99
window vLLM 400s ~1.4% of attack calls (prompt + max_tokens > window) while
only 0.91% are truly over the window. The policy: parse the 400, retry ONCE
with max_tokens clamped to the room the SERVER-STATED window leaves (the
launcher's threaded window is provenance + a cross-check), else raise the
typed ``PromptOverWindow`` — never an empty string standing in for a
response (the first cut did that; graders scored the infrastructure failure
as a refusal and a resume never regenerated it). The runners record such a
goal as ``outcome: "error:PromptOverWindow"`` — the same field and record
shape as an attacker-side ``error:ActorDiscoveryError`` — so the analyzers'
errored-goal rules apply unchanged and ``--resume`` retries it; BoN leaves
the slot uncovered and files a typed ledger row. The rejection bodies below
are the pinned vLLM 0.25.1 shapes the proxy tests already fix
(tests/test_petri_proxy.py, tests/test_petri_store_shape.py).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import served_window as sw
from sisteer.capeval.serving.proxy import ProxyError, ProxyRefusal, upstream_refusal
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.attacks import served_target as st
from sisteer.eval.attacks.static_common import load_covered_goals

# the pinned 0.25.1 chat-validation text (test_petri_proxy.py:966 shape)
BODY_16K = ("This model's maximum context length is 16384 tokens. However, you "
            "requested 1024 output tokens and your prompt contains 19813 input "
            "tokens, for a total of 20837 tokens. Please reduce the length of "
            "the input prompt or the number of requested output tokens.")
# the same text as petri's tool error reconstructs it (test_petri_store_shape.py)
TOOL_ERROR_16K = (
    "Target generate failed: ModelGenerateError: Error code: 400 - "
    "{'error': {'message': 'upstream vLLM returned 400: " + BODY_16K + "', 'type': "
    "'invalid_request_error', 'code': 'upstream_error'}}. Roll back "
    "(or restart) to continue, or end the conversation.")
# the v1 prompt-length form (the prompt ALONE exceeds the window)
PROMPT_FORM = ("The decoder prompt (length 16920) is longer than the maximum model "
               "length of 16384. Make sure that max_model_len is no smaller than "
               "the number of text tokens.")

#: the terminal per-goal record shape the gated runners write (progress.jsonl)
ACTOR_RECORD_KEYS = {"goal_id", "category", "n_target_forwards", "outcome",
                     "actors_used", "capped"}
PYRIT_RECORD_KEYS = {"goal_id", "category", "n_target_forwards", "outcome",
                     "turns", "capped"}
#: the exact outcome string of a goal whose served turn could not fit
ERRORED_OUTCOME = "error:PromptOverWindow"
#: an ERRORED goal's record additionally carries the typed error's own record (the sdfbase-llama8b
#: wave, 2026-09-10: the partial trajectory is in responses.jsonl, this names what outgrew the window)
ERRORED_EXTRA_KEYS = {"prompt_over_window"}


def _body(window, prompt, requested):
    return (f"This model's maximum context length is {window} tokens. However, you "
            f"requested {requested} output tokens and your prompt contains {prompt} "
            f"input tokens, for a total of {prompt + requested} tokens. Please reduce "
            "the length of the input prompt or the number of requested output tokens.")


def _client_error(body):
    """What ProxyClientEngine raises: the proxy's JSON error body inside the
    transport's own text (status in the text, not an attribute)."""
    payload = {"error": {"message": f"upstream vLLM returned 400: {body}",
                         "type": "invalid_request_error", "code": "upstream_error"}}
    return ProxyError("proxy http://127.0.0.1:1/v1/chat/completions returned 400: "
                      + json.dumps(payload))


def _aiohttp_error(body):
    """What the drivers' aiohttp posters raise (unified_sweep_run / the
    families drivers): ``RuntimeError(f"proxy {status}: {body[:400]}")``."""
    payload = {"error": {"message": f"upstream vLLM returned 400: {body}",
                         "type": "invalid_request_error", "code": "upstream_error"}}
    return RuntimeError(f"proxy 400: {json.dumps(payload)[:400]}")


def _rows(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# the parser                                                                   #
# --------------------------------------------------------------------------- #


def test_parses_every_transport_shape_of_the_pinned_text():
    for err in (upstream_refusal(400, BODY_16K),           # in-proc RoutingProxy
                _client_error(BODY_16K),                   # ProxyClientEngine
                _aiohttp_error(BODY_16K),                  # aiohttp posters
                TOOL_ERROR_16K,                            # petri's tool error
                RuntimeError(TOOL_ERROR_16K)):
        ov = sw.parse_context_overflow(err)
        assert ov is not None, err
        assert ov.server_window == 16384
        assert ov.prompt_tokens == 19813
        assert ov.requested_max_tokens == 1024
        assert ov.total_tokens == 20837
        assert ov.excerpt.startswith("maximum context length is 16384 tokens")


def test_parses_the_prompt_only_form():
    ov = sw.parse_context_overflow(upstream_refusal(400, PROMPT_FORM))
    assert ov.server_window == 16384 and ov.prompt_tokens == 16920
    assert ov.requested_max_tokens is None and ov.total_tokens is None


def test_parser_is_precise():
    # a 400 needs a context-length marker; a marker needs a 400
    assert sw.parse_context_overflow(upstream_refusal(400, "malformed request")) is None
    assert sw.parse_context_overflow(RuntimeError("proxy 400: bad shape")) is None
    assert sw.parse_context_overflow(RuntimeError("proxy 429: rate limited; "
                                                  + BODY_16K)) is None
    assert sw.parse_context_overflow(upstream_refusal(500, BODY_16K)) is None   # -> 502
    assert sw.parse_context_overflow(ProxyRefusal(BODY_16K, http_status=409,
                                                  code="cache_salt_collision")) is None
    # a target REPLY echoing the words is content, not a rejection
    assert sw.parse_context_overflow(RuntimeError(
        "I cannot discuss what my maximum context length is")) is None
    # an explicit status wins over the text
    assert sw.parse_context_overflow(BODY_16K, http_status=400) is not None
    assert sw.parse_context_overflow(BODY_16K, http_status=503) is None
    assert sw.parse_context_overflow(BODY_16K) is None   # bare body, no status evidence


# --------------------------------------------------------------------------- #
# the policy                                                                   #
# --------------------------------------------------------------------------- #


def test_clamp_once_when_room_is_at_least_the_minimum():
    ov = sw.parse_context_overflow(_client_error(_body(8704, 8000, 2048)))
    plan = sw.clamp_plan(ov, requested_max_tokens=2048, window=8704)
    assert plan.action == "retry" and plan.max_tokens == 704
    assert plan.effective_window == 8704 and not plan.window_mismatch
    assert plan.mismatch_record() is None
    # exactly the minimum still clamps; one below does not
    ov = sw.parse_context_overflow(_client_error(_body(8704, 8704 - 64, 2048)))
    assert sw.clamp_plan(ov, requested_max_tokens=2048, window=8704).max_tokens == 64
    ov = sw.parse_context_overflow(_client_error(_body(8704, 8704 - 63, 2048)))
    plan = sw.clamp_plan(ov, requested_max_tokens=2048, window=8704)
    assert plan.action == "fail" and "< 64" in plan.reason
    assert sw.MIN_CLAMPED_COMPLETION == 64


def test_an_unfittable_prompt_is_a_typed_error_not_a_turn():
    ov = sw.parse_context_overflow(upstream_refusal(400, PROMPT_FORM))
    plan = sw.clamp_plan(ov, requested_max_tokens=1024, window=16384)
    assert plan.action == "fail"
    err = sw.PromptOverWindow.from_plan(plan, requested_max_tokens=1024)
    assert isinstance(err, RuntimeError)
    assert err.prompt_tokens == 16920 and err.requested_max_tokens == 1024
    assert err.window == 16384 and err.server_window == 16384
    assert err.window_mismatch is None
    rec = err.record()
    assert rec["error"] == sw.PROMPT_OVER_WINDOW == "prompt_over_window"
    assert rec["prompt_tokens"] == 16920 and rec["completion_tokens"] == 0
    assert "prompt_over_window" in str(err) and "16920" in str(err)


def test_server_window_governs_the_clamp_threaded_is_provenance():
    """FINDING 2: the SERVER-STATED window is authoritative; the threaded
    window never shrinks the clamp. The default case that bit: a standup
    launched at 10,240 with a client assuming 8,704 — prompt 9,940 + 512 is
    clampable to 300 and must be SERVED, not filed as prompt_over_window."""
    ov = sw.parse_context_overflow(_client_error(_body(10240, 9940, 512)))
    plan = sw.clamp_plan(ov, requested_max_tokens=512, window=8704)
    assert plan.action == "retry" and plan.max_tokens == 300           # 10240 - 9940
    assert plan.effective_window == 10240                              # the server's
    assert plan.window_mismatch
    assert plan.mismatch_record() == {"threaded": 8704, "server": 10240}
    # threaded LARGER than the server's: still the server's value
    ov = sw.parse_context_overflow(_client_error(_body(8704, 8000, 2048)))
    plan = sw.clamp_plan(ov, requested_max_tokens=2048, window=16384)
    assert plan.max_tokens == 704 and plan.effective_window == 8704
    assert plan.mismatch_record() == {"threaded": 16384, "server": 8704}
    # no threaded window: the server's
    plan = sw.clamp_plan(ov, requested_max_tokens=2048)
    assert plan.action == "retry" and plan.max_tokens == 704 and plan.window is None
    assert plan.mismatch_record() is None
    # the rejection carries no window number: the threaded value stands in
    ov2 = sw.ContextOverflow(None, 8000, 2048, None, "x")
    plan = sw.clamp_plan(ov2, requested_max_tokens=2048, window=8704)
    assert plan.action == "retry" and plan.max_tokens == 704 and plan.effective_window == 8704
    # neither: fail, never guess
    assert sw.clamp_plan(ov2, requested_max_tokens=2048).action == "fail"
    # unparseable prompt count: fail
    ov3 = sw.ContextOverflow(8704, None, 2048, None, "x")
    assert "not parseable" in sw.clamp_plan(ov3, requested_max_tokens=2048, window=8704).reason


def test_plan_for_exception_passes_non_overflows_through():
    assert sw.plan_for_exception(ProxyRefusal("salt", http_status=409),
                                 requested_max_tokens=10) is None
    assert sw.plan_for_exception(RuntimeError("boom"), requested_max_tokens=10) is None
    assert sw.plan_for_exception(_client_error(_body(8704, 8000, 2048)),
                                 requested_max_tokens=2048, window=8704).action == "retry"


# --------------------------------------------------------------------------- #
# served_target: the gated attacks' target                                     #
# --------------------------------------------------------------------------- #


class _WindowedChat:
    """A stub proxy transport with a real window: a request whose
    ``prompt_tokens + max_tokens`` exceeds ``window`` is rejected the way
    ProxyClientEngine surfaces it; otherwise a normal completion."""

    def __init__(self, window, prompt_tokens, *, reject_always=False):
        self.window = window
        self.prompt_tokens = prompt_tokens
        self.reject_always = reject_always
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(dict(payload))
        if self.reject_always or self.prompt_tokens + payload["max_tokens"] > self.window:
            raise _client_error(_body(self.window, self.prompt_tokens, payload["max_tokens"]))
        return {"choices": [{"message": {"content": "served reply"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 3},
                "petri_intervention_hash": "cc" * 8}


def test_target_clamps_once_and_records_it(tmp_path):
    chat = _WindowedChat(window=8704, prompt_tokens=8000)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", max_model_len=8704)
    out = _shot(t, [{"role": "user", "content": "long"}], 2048)
    assert out == "served reply"
    assert [p["max_tokens"] for p in chat.payloads] == [2048, 704]   # ONE retry
    assert t.n_target_forwards == 1 and t.n_clamped == 1 and t.n_prompt_over_window == 0
    assert t.n_window_mismatch == 0 and t.window_mismatches == []
    (row,) = _rows(tmp_path / "t.jsonl")
    assert row["max_tokens"] == 704 and row["max_tokens_clamped_from"] == 2048
    assert row["finish_reason"] == "stop" and row["prompt_tokens"] == 8000
    assert "window_mismatch" not in row


def test_target_raises_the_typed_error_never_an_empty_turn(tmp_path):
    """FINDING 1: an unclampable call is a TYPED ERROR, not a "" response."""
    chat = _WindowedChat(window=8704, prompt_tokens=8690)          # 14 tokens of room
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", max_model_len=8704)
    with pytest.raises(sw.PromptOverWindow) as ei:
        t.chat([{"role": "user", "content": "too long"}], max_new_tokens=2048)
    err = ei.value
    assert err.prompt_tokens == 8690 and err.requested_max_tokens == 2048
    assert err.window == 8704 and err.server_window == 8704
    assert len(chat.payloads) == 1                      # no pointless retry
    assert t.n_prompt_over_window == 1 and t.n_clamped == 0 and t.n_target_forwards == 1
    # a diagnostic timing row, tagged as the error — never a response row
    (row,) = _rows(tmp_path / "t.jsonl")
    assert row["error"] == "prompt_over_window" and row["finish_reason"] is None
    assert row["prompt_tokens"] == 8690 and row["completion_tokens"] == 0
    assert row["max_tokens"] == 2048 and row["window"] == 8704 and row["server_window"] == 8704
    assert "< 2048 minimum completion" in row["window_reason"]
    # the target stays usable: the next call works normally
    t._chat = _WindowedChat(window=8704, prompt_tokens=100)
    assert t.chat([{"role": "user", "content": "short"}], max_new_tokens=2048) == "served reply"
    assert _rows(tmp_path / "t.jsonl")[1]["finish_reason"] == "stop"


def test_target_second_rejection_is_terminal(tmp_path):
    chat = _WindowedChat(window=8704, prompt_tokens=8000, reject_always=True)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", max_model_len=8704)
    with pytest.raises(sw.PromptOverWindow, match="rejected again"):
        _shot(t, [{"role": "user", "content": "x"}], 2048)
    assert [p["max_tokens"] for p in chat.payloads] == [2048, 704]   # exactly two
    (row,) = _rows(tmp_path / "t.jsonl")
    assert row["error"] == "prompt_over_window" and "rejected again" in row["window_reason"]
    assert t.n_prompt_over_window == 1 and t.n_clamped == 0


def test_target_non_overflow_errors_still_raise():
    def salt_refusal(payload):
        raise ProxyRefusal("cache_salt collision", http_status=409,
                           code="cache_salt_collision")
    t = st.ServedChatTarget(salt_refusal, virtual_model="petri__x", cell_id="x",
                            max_model_len=8704)
    with pytest.raises(ProxyRefusal):
        t.chat([{"role": "user", "content": "x"}], max_new_tokens=10)

    def transport_down(payload):
        raise ProxyError("cannot reach the petri proxy")
    t = st.ServedChatTarget(transport_down, virtual_model="petri__x", cell_id="x")
    with pytest.raises(ProxyError):
        t.chat([{"role": "user", "content": "x"}], max_new_tokens=10)


def test_target_without_a_threaded_window_clamps_against_the_servers(tmp_path):
    chat = _WindowedChat(window=8704, prompt_tokens=8000)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl")           # max_model_len=None
    assert _shot(t, [{"role": "user", "content": "x"}], 2048) == "served reply"
    assert chat.payloads[1]["max_tokens"] == 704
    assert t.max_model_len is None and t.n_window_mismatch == 0


def test_target_mismatch_serves_at_the_server_window_and_records_it(tmp_path, capsys):
    """FINDING 2 at the client: standup at 10,240, client told 8,704 — the
    clampable call is SERVED (300 tokens), the mismatch is logged loudly and
    recorded on the row and in the manifest; nothing is manufactured."""
    chat = _WindowedChat(window=10240, prompt_tokens=9940)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", max_model_len=8704)
    out = _shot(t, [{"role": "user", "content": "x"}], 512)
    assert out == "served reply"
    assert [p["max_tokens"] for p in chat.payloads] == [512, 300]
    assert t.n_clamped == 1 and t.n_prompt_over_window == 0
    assert t.n_window_mismatch == 1
    assert t.window_mismatches == [{"threaded": 8704, "server": 10240}]
    (row,) = _rows(tmp_path / "t.jsonl")
    assert row["window_mismatch"] == {"threaded": 8704, "server": 10240}
    assert row["max_tokens_clamped_from"] == 512
    assert "WINDOW MISMATCH" in capsys.readouterr().out
    # a second identical disagreement counts but does not re-print
    _shot(t, [{"role": "user", "content": "x"}], 512)
    assert t.n_window_mismatch == 2 and len(t.window_mismatches) == 1
    assert "WINDOW MISMATCH" not in capsys.readouterr().out
    # the generate (XSTest) path shares the machinery
    outs = t.generate(["g"], max_new_tokens=512)
    assert outs == ["served reply"] and t.n_clamped == 3
    with pytest.raises(st.ServedTargetError):
        st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", max_model_len=0)


def test_manifest_is_staged_and_carries_the_real_counters(tmp_path):
    chat = _WindowedChat(window=10240, prompt_tokens=9940)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            max_model_len=8704, route_info={"max_model_len": 8704})
    st.write_target_manifest(tmp_path, t, stage="started")
    doc = json.loads((tmp_path / "served_target.json").read_text())
    assert doc["manifest_stage"] == "started"
    assert doc["window_clamp"]["n_clamped"] == 0
    _shot(t, [{"role": "user", "content": "x"}], 512)
    t._chat = _WindowedChat(window=10240, prompt_tokens=10230)
    with pytest.raises(sw.PromptOverWindow):
        t.chat([{"role": "user", "content": "x"}], max_new_tokens=512)
    st.write_target_manifest(tmp_path, t, stage="final")
    doc = json.loads((tmp_path / "served_target.json").read_text())
    assert doc["manifest_stage"] == "final" and doc["max_model_len"] == 8704
    assert doc["window_clamp"].pop("multiturn_overflow_policy") == "strict"   # chat() never clamps
    assert doc["window_clamp"] == {
        "max_model_len": 8704, "min_clamped_completion": 64,
        "n_clamped": 1, "n_prompt_over_window": 1, "n_window_mismatch": 2,
        "window_mismatches": [{"threaded": 8704, "server": 10240}],
        "n_tokenize_resolved": 0, "n_floor_only_unresolvable": 0,
        "n_target_forwards": 2}


def test_build_served_target_threads_the_window(monkeypatch, tmp_path):
    from sisteer.capeval.serving import proxy as px
    from test_served_target import _routes
    table = tmp_path / "rt.json"
    table.write_text(json.dumps({
        "base_model": "m", "serving_mode": "interleaved", "prompted_prefixes": {},
        "routes": [r.__dict__ for r in _routes()]}))

    class _Client:
        def __init__(self, url, *, identity):
            self.base_url = url; self.served_models = (); self.proxy_identity = None
        def load(self, *, require_virtual_models=()): pass
        def chat_completions(self, payload): return {}
    monkeypatch.setattr(st, "ProxyClientEngine", _Client)
    monkeypatch.setattr(px, "load_routing_table", lambda p: {
        "routes": _routes(), "base_model": "m"})
    t = st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id="none", controls=False, max_model_len=8704)
    assert t.max_model_len == 8704 and t.route_info["max_model_len"] == 8704
    t = st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id="none", controls=False)
    assert t.max_model_len is None and t.route_info["max_model_len"] is None
    assert t._tokenize is None            # a client without /tokenize: unresolvable, never a guess


# --------------------------------------------------------------------------- #
# the runners: an unfittable turn is an ERRORED goal, resume retries it        #
# --------------------------------------------------------------------------- #


def _shot(target, messages, max_new_tokens):
    """The SINGLE-SHOT surface (generate): the once-only window clamp is a deliberately supported
    recovery there and stays exactly as pinned below. chat() — the multi-turn surface — is STRICT
    since 2026-09-10: an overflow is the typed PromptOverWindow, never a clamped retry."""
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return target.generate([messages[0]["content"]], max_new_tokens=max_new_tokens)[0]


class _OverWindowDefense:
    """A served-target stand-in whose every turn cannot fit: ``chat`` raises
    the typed error (the LLMHFSteered surface the runners drive)."""

    def __init__(self, *, window=8704, prompt_tokens=8690):
        self.n_target_forwards = 0
        self.mode = "x"
        self.calls = 0
        self._err = sw.PromptOverWindow(
            prompt_tokens=prompt_tokens, requested_max_tokens=2048, window=window,
            server_window=window, effective_window=window,
            reason=f"prompt {prompt_tokens} leaves {window - prompt_tokens} of "
                   f"window {window} (< 64 minimum completion)")

    def chat(self, messages, *, max_new_tokens=512, do_sample=False,
             temperature=1.0, sampling_seed=None):
        self.calls += 1
        self.n_target_forwards += 1
        raise self._err


def _goals(n=1):
    from sisteer.eval.jailbreak_data import Goal
    return [Goal(f"g{i}", f"benign objective {i}", "illegal", "harmbench", "rev")
            for i in range(n)]


def test_actorattack_records_the_goal_as_errored_like_a_discovery_error(tmp_path):
    from sisteer.eval.attacks.multiturn import run_actorattack
    import jailbreak_run
    d = _OverWindowDefense()
    goals = _goals(1)
    out = run_actorattack(
        d, goals, attacker_fn=lambda chat: '{"actors": [{"name": "a1", "queries": ["q0", "q1"]}]}',
        scorer_fn=lambda p: ("no" if "Refused" in p else "1"),
        out_dir=tmp_path / "aa", actor_num=1, discovery_retries=0, max_retries=3)
    (rec,) = out["per_goal"]
    # THE RECORD: the same field + terminal shape as an attacker-side error
    assert set(rec) == ACTOR_RECORD_KEYS
    assert rec["outcome"] == ERRORED_OUTCOME
    assert rec["goal_id"] == "g0" and rec["actors_used"] == 0 and rec["capped"] is False
    # deterministic: the repo's range(max_retries) loop did NOT re-issue it
    assert d.calls == 1 and rec["n_target_forwards"] == 1
    # no "" response was persisted anywhere
    assert not (tmp_path / "aa" / "responses.jsonl").exists()
    # progress.jsonl carries exactly that record, and the goal is NOT done:
    # a resume retries it (static_common.load_covered_goals), the graders see
    # an error: outcome (jailbreak_run._load_progress), never a scored refusal
    (prow,) = _rows(tmp_path / "aa" / "progress.jsonl")
    assert prow == rec
    assert load_covered_goals(tmp_path / "aa" / "progress.jsonl") == set()
    assert jailbreak_run._load_progress(tmp_path / "aa" / "progress.jsonl")["g0"]["outcome"] \
        == ERRORED_OUTCOME
    # ... and a resume re-attempts exactly that goal
    out2 = run_actorattack(
        d, goals, attacker_fn=lambda chat: '{"actors": [{"name": "a1", "queries": ["q0"]}]}',
        scorer_fn=lambda p: "1", out_dir=tmp_path / "aa", actor_num=1,
        discovery_retries=0, max_retries=1, resume=True)
    assert out2["n_goals_run"] == 1 and d.calls == 2
    # the attacker-side error it is modelled on has the SAME shape
    out3 = run_actorattack(
        d, goals, attacker_fn=lambda chat: "not json", scorer_fn=lambda p: "1",
        out_dir=tmp_path / "disc", actor_num=1, discovery_retries=0)
    assert set(out3["per_goal"][0]) == ACTOR_RECORD_KEYS
    assert out3["per_goal"][0]["outcome"] == "error:ActorDiscoveryError"


def test_pyrit_runner_records_errored_goal_even_when_the_attack_swallows(tmp_path):
    """PyRIT's normalizer re-raises target errors as a bare Exception and the
    TAP/PAIR node swallows them (pinned 1.0.1): the typed error is latched on
    the target and read back per goal, so the record is error:PromptOverWindow
    on both the swallowed and the wrapped path."""
    from pyrit.models import Message
    from sisteer.eval.attacks.pyrit_adapters import (AttackerChatTarget,
                                                     run_pyrit_attack_async)
    goals = _goals(2)
    d = _OverWindowDefense()

    class _Swallowing:
        """A TAP-node-like executor: the target's exception is swallowed and
        the attack reports an ordinary failure."""
        def __init__(self, target): self.target = target
        async def execute_async(self, *, objective):
            try:
                await self.target._send_prompt_to_target_async(
                    normalized_conversation=[Message.from_prompt(prompt=objective, role="user")])
            except Exception:  # noqa: BLE001 — the swallow under test
                pass
            return SimpleNamespace(outcome="failure", executed_turns=1)

    class _Wrapping(_Swallowing):
        """A Crescendo-like executor: the normalizer's bare-Exception wrap."""
        async def execute_async(self, *, objective):
            try:
                await self.target._send_prompt_to_target_async(
                    normalized_conversation=[Message.from_prompt(prompt=objective, role="user")])
            except Exception as ex:  # noqa: BLE001
                raise Exception("Error sending prompt with conversation ID: x") from ex
            return SimpleNamespace(outcome="failure", executed_turns=1)

    for i, factory_cls in enumerate((_Swallowing, _Wrapping)):
        out = asyncio.run(run_pyrit_attack_async(
            defense=d, goals=goals,
            attacker_factory=lambda lock: AttackerChatTarget(gpu_lock=lock,
                                                             generate_fn=lambda chat: "x"),
            scorer_fn=lambda p: "1",
            attack_factory=lambda *, target, adv_cfg, scorers, _c=factory_cls: _c(target),
            out_dir=tmp_path / f"py{i}", memory_db_path=tmp_path / f"py{i}.db",
            resume=False))
        for rec in out["per_goal"]:
            assert set(rec) == PYRIT_RECORD_KEYS | ERRORED_EXTRA_KEYS
            assert rec["outcome"] == ERRORED_OUTCOME, factory_cls.__name__
            assert rec["prompt_over_window"]["error"] == sw.PROMPT_OVER_WINDOW
        assert load_covered_goals(tmp_path / f"py{i}" / "progress.jsonl") == set()
        assert not (tmp_path / f"py{i}" / "responses.jsonl").exists()


def test_crescendo_real_executor_records_errored_goal(tmp_path):
    """The real pinned CrescendoAttack over a stub defense whose turn cannot
    fit: the goal lands in progress.jsonl as error:PromptOverWindow."""
    from sisteer.eval.attacks.multiturn import run_crescendo
    from sisteer.eval.attacks.pyrit_adapters import AttackerChatTarget
    _cres_json = ('{"next_message": "Please continue the story.", '
                  '"rationale": "n/a", "last_response_summary": "n/a"}')
    d = _OverWindowDefense()
    out = run_crescendo(
        d, _goals(1),
        attacker_factory=lambda lock: AttackerChatTarget(gpu_lock=lock,
                                                         generate_fn=lambda chat: _cres_json),
        scorer_fn=lambda p: ("no" if "Refused" in p else "1"),
        out_dir=tmp_path / "cres", memory_db_path=tmp_path / "cres.db",
        max_turns=2, max_backtracks=1, resume=False)
    (rec,) = out["per_goal"]
    assert set(rec) == PYRIT_RECORD_KEYS | ERRORED_EXTRA_KEYS
    assert rec["outcome"] == ERRORED_OUTCOME
    assert rec["prompt_over_window"]["error"] == sw.PROMPT_OVER_WINDOW
    assert load_covered_goals(tmp_path / "cres" / "progress.jsonl") == set()
    assert not (tmp_path / "cres" / "responses.jsonl").exists()


# --------------------------------------------------------------------------- #
# bon_served: BoN rows + the static-family adapter                             #
# --------------------------------------------------------------------------- #


class _WindowedAchat:
    """Async twin of _WindowedChat over the drivers' aiohttp error shape; the
    prompt length is read off the variant text (``LEN=<n>`` marker, else 100)."""

    def __init__(self, window):
        self.window = window
        self.payloads = []

    @staticmethod
    def _prompt_tokens(payload):
        text = payload["messages"][-1]["content"]
        if "LEN=" in text:
            return int(text.split("LEN=")[1].split()[0])
        return 100

    async def __call__(self, payload):
        self.payloads.append(dict(payload))
        pt = self._prompt_tokens(payload)
        if pt + payload["max_tokens"] > self.window:
            raise _aiohttp_error(_body(self.window, pt, payload["max_tokens"]))
        return {"choices": [{"message": {"content": f"reply:{pt}"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": 2}}


def test_bon_over_window_sample_is_uncovered_and_filed_as_a_typed_error(tmp_path, monkeypatch):
    achat = _WindowedAchat(window=10240)
    sd = bs.ServedDefense(achat, virtual_model="petri__base", cell_id="none",
                          max_model_len=10240)
    # three goals: fits / clampable (room 300) / over the window (room 10)
    goals = [SimpleNamespace(goal_id="g_fit", text="LEN=100 fits", category="illegal"),
             SimpleNamespace(goal_id="g_clamp", text="LEN=9940 clamp", category="illegal"),
             SimpleNamespace(goal_id="g_over", text="LEN=10230 over", category="illegal")]
    monkeypatch.setattr(bs, "augment", lambda text, sigma, rng: text)   # keep the marker
    monkeypatch.setattr(bs, "assert_cell_salt_isolation", lambda routes: None)
    out_dir = tmp_path / "out"
    jsonl = bs.bon_generate_served(sd, goals, out_dir=out_dir, routes=[], n_max=1,
                                   sigma=0.0, trial=0, max_new_tokens=512,
                                   max_concurrency=4, model="gemma4e4b")
    rows = {r["goal_id"]: r for r in _rows(jsonl)}
    # FINDING 1: the unfittable sample is NOT a covered row — no "" phantom
    assert set(rows) == {"g_fit", "g_clamp"}
    assert rows["g_fit"]["finish_reason"] == "stop" and "max_tokens_clamped_from" not in rows["g_fit"]
    assert rows["g_clamp"]["response"] == "reply:9940"
    assert rows["g_clamp"]["max_tokens_clamped_from"] == 512
    # ... it is a TYPED error row in the separate ledger
    (erow,) = _rows(out_dir / bs.BON_ERRORS_FILE)
    assert erow["goal_id"] == "g_over" and erow["sample_idx"] == 0 and erow["trial"] == 0
    assert erow["error"] == "prompt_over_window"
    assert erow["prompt_tokens"] == 10230 and erow["window"] == 10240
    assert erow["server_window"] == 10240 and "< 64" in erow["window_reason"]
    assert "response" not in erow
    # ... its slot stays uncovered (a resume retries it) and the completeness
    # gate reports the shortfall instead of grading a phantom sample
    from sisteer.eval.attacks.bon import _load_covered, bon_curves, load_bon_responses
    assert _load_covered(jsonl) == {("g_fit", 0, 0), ("g_clamp", 0, 0)}
    by_goal = load_bon_responses(jsonl)
    assert "g_over" not in by_goal
    with pytest.raises(RuntimeError, match="incomplete for goal_id=g_over"):
        bon_curves(goals, by_goal, lambda g, r: 0.0, n_max=1, threshold=0.5)
    # the manifest (the BoN twin of served_target.json) was written AFTER
    # generation with the real counters
    man = json.loads((out_dir / bs.BON_SERVED_MANIFEST).read_text())
    # the units state the PLANNED workload too — 3 planned, 2 covered, the over-window slot UNCOVERED
    # a prompt_over_window slot is a DECIDED, PROVEN no-fit (bon_climb.classify_over_window), so the store is complete
    # with nothing undecided — the completeness predicate the writer and the climb readers share
    assert man["units"] == {"n_planned": 3, "n_work": 3, "n_written": 2, "n_prompt_over_window": 1, "n_done": 2, "n_uncovered": 1,
                             "n_nofit": 1, "n_nofit_proven": 1, "n_over_window_undecided": 0, "n_undecided": 0, "decided": True}
    assert man["window_clamp"]["n_clamped"] == 1
    assert man["window_clamp"]["n_prompt_over_window"] == 1
    assert man["window_clamp"]["n_target_forwards"] == 3
    assert man["errors_file"] == bs.BON_ERRORS_FILE
    assert sd.n_clamped == 1 and sd.n_prompt_over_window == 1
    # request accounting: clamp = once at 512 then once at 300; over = once
    sent = [(achat._prompt_tokens(p), p["max_tokens"]) for p in achat.payloads]
    assert sent.count((9940, 512)) == 1 and sent.count((9940, 300)) == 1
    assert [s for s in sent if s[0] == 10230] == [(10230, 512)]
    # a resume regenerates ONLY the uncovered slot (and fails it again at the
    # same window — bounded, visible; a wider window would serve it)
    achat.payloads.clear()
    bs.bon_generate_served(sd, goals, out_dir=out_dir, routes=[], n_max=1,
                           sigma=0.0, trial=0, max_new_tokens=512, max_concurrency=4, model="gemma4e4b")
    assert [achat._prompt_tokens(p) for p in achat.payloads] == [10230]
    assert len(_rows(out_dir / bs.BON_ERRORS_FILE)) == 2


def test_bon_window_mismatch_serves_at_the_server_window(tmp_path, monkeypatch, capsys):
    achat = _WindowedAchat(window=10240)
    sd = bs.ServedDefense(achat, virtual_model="petri__base", cell_id="none",
                          max_model_len=8704)                          # stale provenance
    turn = asyncio.run(sd.generate_one_detailed("LEN=9940 x", sampling_seed=0,
                                                max_new_tokens=512, temperature=1.0))
    assert turn.text == "reply:9940" and turn.max_tokens == 300
    assert turn.max_tokens_clamped_from == 512
    assert turn.window_mismatch == {"threaded": 8704, "server": 10240}
    assert sd.window_mismatches == [{"threaded": 8704, "server": 10240}]
    assert "WINDOW MISMATCH" in capsys.readouterr().out
    assert sd.clamp_report()["n_window_mismatch"] == 1


def test_bon_generate_one_text_surface_and_typed_error():
    achat = _WindowedAchat(window=10240)
    sd = bs.ServedDefense(achat, virtual_model="petri__base", cell_id="none")
    text = asyncio.run(sd.generate_one("LEN=100 x", sampling_seed=0, max_new_tokens=64, temperature=1.0))
    assert text == "reply:100"
    turn = asyncio.run(sd.generate_one_detailed("LEN=100 x", sampling_seed=0,
                                                max_new_tokens=64, decode={"temperature": 1.0}))
    assert isinstance(turn, bs.ServedTurn) and turn.max_tokens == 64
    assert turn.finish_reason == "stop" and turn.max_tokens_clamped_from is None
    with pytest.raises(sw.PromptOverWindow):
        asyncio.run(sd.generate_one("LEN=10230 x", sampling_seed=0, max_new_tokens=512, temperature=1.0))
    assert sd.n_prompt_over_window == 1
    # no sampler at all: refused before any request (best-of-N has no model-independent default since 2026-09-17)
    with pytest.raises(bs.BonServedError, match="no sampler"):
        asyncio.run(sd.generate_one("LEN=100 x", sampling_seed=0, max_new_tokens=64))
    with pytest.raises(bs.BonServedError):
        bs.ServedDefense(achat, virtual_model="petri__base", cell_id="none", max_model_len=0)


def test_bon_non_overflow_errors_still_raise():
    async def boom(payload):
        raise RuntimeError("proxy 502: engine core died")
    sd = bs.ServedDefense(boom, virtual_model="petri__base", cell_id="none",
                          max_model_len=8704)
    with pytest.raises(RuntimeError, match="502"):
        asyncio.run(sd.generate_one("x", sampling_seed=0, max_new_tokens=64, temperature=1.0))


def test_static_families_rescue_clampable_and_file_over_window_as_errors():
    loop = asyncio.new_event_loop()
    try:
        achat = _WindowedAchat(window=10240)
        sd = bs.ServedDefense(achat, virtual_model="petri__base", cell_id="none",
                              max_model_len=10240)
        # record-mode: the over-window prompt is a counted placeholder, once
        d = bs.ServedStaticDefense(sd, loop=loop, max_concurrency=4,
                                   raise_on_error=False, max_retries=2)
        out = d.generate(["LEN=100 a", "LEN=9940 b", "LEN=10230 c"], max_new_tokens=512)
        assert out[0] == "reply:100" and out[1] == "reply:9940"
        assert out[2].startswith("__SERVED_ERROR__") and "prompt_over_window" in out[2]
        assert d.n_errors == 1 and d.n_prompt_over_window == 1
        # deterministic: the over-window prompt was NOT retried max_retries times
        assert sum(1 for p in achat.payloads if "LEN=10230" in p["messages"][-1]["content"]) == 1
        # loud-mode (the sweep default): a family with an unfittable prompt
        # refuses rather than writing a store that looks complete
        d2 = bs.ServedStaticDefense(sd, loop=loop, max_concurrency=4)
        with pytest.raises(bs.BonServedError, match="prompt_over_window"):
            d2.generate(["LEN=10230 c"], max_new_tokens=512)
        # and a clampable-only family is fully rescued in loud mode
        assert d2.generate(["LEN=9940 b"], max_new_tokens=512) == ["reply:9940"]
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# F3: the PRODUCTION floor-only rejection ("at least N input tokens")          #
# --------------------------------------------------------------------------- #

#: THE production rejection — the ONE template, byte-for-byte, behind all 284
#: real context-length 400s in the live Petri audit logs (vLLM 0.25.1): the
#: prompt length is a LOWER BOUND ("at least"), the trailing sentence is
#: "... the messages or completion."
FLOOR_ONLY_16K = ("This model's maximum context length is 16384 tokens. However, you "
                  "requested 1024 output tokens and your prompt contains at least 15361 "
                  "input tokens, for a total of at least 16385 tokens. Please reduce the "
                  "length of the messages or completion.")
#: a wording variant (vLLM's text drifts across versions)
FLOOR_ONLY_16K_REQUEST_HAS = (
    "This model's maximum context length is 16384 tokens. However, you requested 1024 "
    "output tokens and your request has at least 15361 input tokens, for a total of at "
    "least 16385 tokens. Please reduce the length of the messages or completion.")


def _floor_body(window, requested, *, wording="prompt contains"):
    floor = window - requested + 1
    return (f"This model's maximum context length is {window} tokens. However, you "
            f"requested {requested} output tokens and your {wording} at least "
            f"{floor} input tokens, for a total of at least {window + 1} tokens. Please "
            "reduce the length of the messages or completion.")


def test_production_fixture_is_byte_exact():
    assert _floor_body(16384, 1024) == FLOOR_ONLY_16K
    assert _floor_body(16384, 1024, wording="request has") == FLOOR_ONLY_16K_REQUEST_HAS
    assert FLOOR_ONLY_16K.startswith("This model's") and FLOOR_ONLY_16K.endswith(
        "Please reduce the length of the messages or completion.")


@pytest.mark.parametrize("text", [FLOOR_ONLY_16K, FLOOR_ONLY_16K_REQUEST_HAS])
def test_floor_only_form_parses_as_a_floor_never_a_length(text):
    for err in (upstream_refusal(400, text), _client_error(text), _aiohttp_error(text),
                "Error code: 400 - {'error': {'message': 'upstream vLLM returned 400: "
                + text + "'}}"):
        ov = sw.parse_context_overflow(err)
        assert ov is not None
        assert ov.server_window == 16384 and ov.requested_max_tokens == 1024
        assert ov.prompt_tokens is None                    # NOT 15361
        assert ov.prompt_floor == 15361 and ov.floor_only
    # the plan asks for the exact length instead of failing or guessing
    ov = sw.parse_context_overflow(upstream_refusal(400, text))
    plan = sw.clamp_plan(ov, requested_max_tokens=1024, window=16384)
    assert plan.action == sw.RESOLVE == "resolve" and plan.needs_exact_prompt_tokens
    assert plan.prompt_tokens is None and plan.prompt_tokens_source is None
    # resolved: room = window - exact prompt; one clamped retry when >= 64
    p2 = plan.with_exact_prompt_tokens(15400)
    assert p2.action == "retry" and p2.max_tokens == 984
    assert p2.prompt_tokens == 15400 and p2.prompt_tokens_source == "tokenize_endpoint"
    p3 = plan.with_exact_prompt_tokens(16384 - 63)
    assert p3.action == "fail" and "< 64" in p3.reason and p3.prompt_tokens_source == "tokenize_endpoint"
    # unresolvable: never a guess from the floor
    p4 = plan.unresolvable("HTTP 404")
    assert p4.action == "fail" and p4.reason == "floor_only_unresolvable" == sw.FLOOR_ONLY_UNRESOLVABLE
    assert p4.prompt_tokens is None
    err = sw.PromptOverWindow.from_plan(p4, requested_max_tokens=1024)
    assert err.reason == "floor_only_unresolvable" and err.prompt_floor == 15361
    assert err.record()["prompt_tokens"] is None and err.record()["prompt_floor"] == 15361
    assert err.record()["prompt_tokens_source"] is None and "404" in err.record()["window_reason_detail"]
    # the message-stated form records its source too
    ov2 = sw.parse_context_overflow(_client_error(_body(8704, 8000, 2048)))
    assert sw.clamp_plan(ov2, requested_max_tokens=2048).prompt_tokens_source == "error_message"


def test_tokenize_payload_and_count_helpers():
    req = {"model": "petri__x", "messages": [{"role": "user", "content": "hi"}],
           "max_tokens": 2048, "temperature": 0.0, "seed": 3}
    assert sw.tokenize_payload(req) == {"model": "petri__x", "messages": req["messages"],
                                        "add_generation_prompt": True}
    assert sw.exact_prompt_tokens({"count": 15400, "max_model_len": 16384, "tokens": []}) == 15400
    for bad in ({}, {"count": "15400"}, {"count": -1}, {"count": True}, None, "x"):
        with pytest.raises(ValueError):
            sw.exact_prompt_tokens(bad)


class _FloorOnlyChat(_WindowedChat):
    """A server that rejects in the PRODUCTION floor-only form (or the
    "request has" wording variant)."""

    wording = "prompt contains"

    def __call__(self, payload):
        self.payloads.append(dict(payload))
        if self.reject_always or self.prompt_tokens + payload["max_tokens"] > self.window:
            raise _client_error(_floor_body(self.window, payload["max_tokens"],
                                            wording=self.wording))
        return {"choices": [{"message": {"content": "served reply"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 3}}


class _FakeTokenize:
    """The proxy's /tokenize pass-through: returns the server's exact count
    for the identical messages (or raises, modelling a 404 / dead route)."""

    def __init__(self, count, *, fail=None):
        self.count = count
        self.fail = fail
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(dict(payload))
        if self.fail is not None:
            raise self.fail
        return {"count": self.count, "max_model_len": 16384, "tokens": []}


@pytest.mark.parametrize("wording", ["prompt contains", "request has"])
def test_target_resolves_floor_only_via_tokenize_and_clamps_once(tmp_path, wording):
    chat = _FloorOnlyChat(window=16384, prompt_tokens=15400)
    chat.wording = wording
    tok = _FakeTokenize(15400)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", max_model_len=16384,
                            tokenize_fn=tok)
    msgs = [{"role": "user", "content": "a very long transcript"}]
    assert _shot(t, msgs, 1024) == "served reply"
    assert [p["max_tokens"] for p in chat.payloads] == [1024, 984]     # 16384 - 15400
    # tokenize saw the IDENTICAL messages and model, nothing else
    assert tok.payloads == [{"model": "petri__x", "messages": msgs,
                             "add_generation_prompt": True}]
    assert t.n_clamped == 1 and t.n_tokenize_resolved == 1 and t.n_floor_only_unresolvable == 0
    (row,) = _rows(tmp_path / "t.jsonl")
    assert row["max_tokens_clamped_from"] == 1024 and row["max_tokens"] == 984
    assert row["prompt_tokens_source"] == "tokenize_endpoint"
    # a floor-only rejection whose exact length leaves < 64: the typed error,
    # with the exact length and its source on the record
    chat2 = _FloorOnlyChat(window=16384, prompt_tokens=16380)
    t2 = st.ServedChatTarget(chat2, virtual_model="petri__x", cell_id="x",
                             timing_log=tmp_path / "t2.jsonl", max_model_len=16384,
                             tokenize_fn=_FakeTokenize(16380))
    with pytest.raises(sw.PromptOverWindow) as ei:
        t2.chat(msgs, max_new_tokens=1024)
    assert ei.value.prompt_tokens == 16380 and ei.value.prompt_tokens_source == "tokenize_endpoint"
    assert ei.value.prompt_floor == 15361 and len(chat2.payloads) == 1
    (row,) = _rows(tmp_path / "t2.jsonl")
    assert row["error"] == "prompt_over_window" and row["prompt_tokens_source"] == "tokenize_endpoint"


def test_target_floor_only_without_tokenize_is_unresolvable_never_a_guess(tmp_path):
    chat = _FloorOnlyChat(window=16384, prompt_tokens=15400)      # clampable in truth
    # no transport at all
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", max_model_len=16384)
    with pytest.raises(sw.PromptOverWindow, match="floor_only_unresolvable"):
        t.chat([{"role": "user", "content": "x"}], max_new_tokens=1024)
    assert len(chat.payloads) == 1 and t.n_floor_only_unresolvable == 1 and t.n_clamped == 0
    (row,) = _rows(tmp_path / "t.jsonl")
    assert row["window_reason"] == "floor_only_unresolvable"
    assert row["prompt_tokens"] is None and row["prompt_floor"] == 15361
    # a dead route (404 from the proxy) — same verdict, detail recorded
    chat = _FloorOnlyChat(window=16384, prompt_tokens=15400)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t2.jsonl", max_model_len=16384,
                            tokenize_fn=_FakeTokenize(0, fail=ProxyError(
                                "proxy http://127.0.0.1:1/v1/tokenize returned 404: not found")))
    with pytest.raises(sw.PromptOverWindow) as ei:
        t.chat([{"role": "user", "content": "x"}], max_new_tokens=1024)
    assert ei.value.reason == "floor_only_unresolvable" and "404" in ei.value.detail
    assert len(chat.payloads) == 1
    # a malformed tokenize response is not a count either
    t = st.ServedChatTarget(_FloorOnlyChat(window=16384, prompt_tokens=15400),
                            virtual_model="petri__x", cell_id="x", max_model_len=16384,
                            tokenize_fn=lambda p: {"tokens": []})
    with pytest.raises(sw.PromptOverWindow, match="floor_only_unresolvable"):
        t.chat([{"role": "user", "content": "x"}], max_new_tokens=1024)


def test_manifest_counts_tokenize_resolutions(tmp_path):
    chat = _FloorOnlyChat(window=16384, prompt_tokens=15400)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            max_model_len=16384, tokenize_fn=_FakeTokenize(15400))
    _shot(t, [{"role": "user", "content": "x"}], 1024)
    st.write_target_manifest(tmp_path, t, stage="final")
    wc = json.loads((tmp_path / "served_target.json").read_text())["window_clamp"]
    assert wc["n_tokenize_resolved"] == 1 and wc["n_floor_only_unresolvable"] == 0


def test_build_served_target_wires_the_clients_tokenize(monkeypatch, tmp_path):
    from sisteer.capeval.serving import proxy as px
    from test_served_target import _routes
    table = tmp_path / "rt.json"
    table.write_text("{}")

    class _Client:
        def __init__(self, url, *, identity):
            self.base_url = url; self.served_models = (); self.proxy_identity = None
        def load(self, *, require_virtual_models=()): pass
        def chat_completions(self, payload): return {}
        def tokenize(self, payload): return {"count": 7}
    monkeypatch.setattr(st, "ProxyClientEngine", _Client)
    monkeypatch.setattr(px, "load_routing_table", lambda p: {"routes": _routes(), "base_model": "m"})
    t = st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id="none", controls=False)
    assert t._tokenize is not None and t._tokenize({}) == {"count": 7}


class _FloorOnlyAchat(_WindowedAchat):
    """Async twin: the production floor-only rejection over the aiohttp shape."""

    async def __call__(self, payload):
        self.payloads.append(dict(payload))
        pt = self._prompt_tokens(payload)
        if pt + payload["max_tokens"] > self.window:
            raise _aiohttp_error(_floor_body(self.window, payload["max_tokens"]))
        return {"choices": [{"message": {"content": f"reply:{pt}"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": 2}}


def test_bon_resolves_floor_only_via_tokenize(tmp_path, monkeypatch):
    achat = _FloorOnlyAchat(window=10240)
    tok_payloads = []

    async def atokenize(payload):
        tok_payloads.append(dict(payload))
        return {"count": _WindowedAchat._prompt_tokens(payload), "max_model_len": 10240}

    sd = bs.ServedDefense(achat, virtual_model="petri__base", cell_id="none",
                          max_model_len=10240, atokenize=atokenize)
    turn = asyncio.run(sd.generate_one_detailed("LEN=9940 x", sampling_seed=0, max_new_tokens=512, temperature=1.0))
    assert turn.text == "reply:9940" and turn.max_tokens == 300
    assert turn.max_tokens_clamped_from == 512 and turn.prompt_tokens_source == "tokenize_endpoint"
    assert tok_payloads and tok_payloads[0]["messages"] == achat.payloads[0]["messages"]
    assert sd.n_tokenize_resolved == 1 and sd.clamp_report()["n_tokenize_resolved"] == 1
    # the BoN row records the source
    monkeypatch.setattr(bs, "augment", lambda text, sigma, rng: text)
    monkeypatch.setattr(bs, "assert_cell_salt_isolation", lambda routes: None)
    goals = [SimpleNamespace(goal_id="g", text="LEN=9940 clamp", category="illegal")]
    jsonl = bs.bon_generate_served(sd, goals, out_dir=tmp_path / "out", routes=[], n_max=1,
                                   sigma=0.0, trial=0, max_new_tokens=512, max_concurrency=2, model="gemma4e4b")
    (row,) = _rows(jsonl)
    assert row["max_tokens_clamped_from"] == 512 and row["prompt_tokens_source"] == "tokenize_endpoint"
    # without a tokenize transport: typed error, floor_only_unresolvable, ledger row
    sd2 = bs.ServedDefense(_FloorOnlyAchat(window=10240), virtual_model="petri__base",
                           cell_id="none", max_model_len=10240)
    with pytest.raises(sw.PromptOverWindow, match="floor_only_unresolvable"):
        asyncio.run(sd2.generate_one("LEN=9940 x", sampling_seed=0, max_new_tokens=512, temperature=1.0))
    assert sd2.n_floor_only_unresolvable == 1 and sd2.n_prompt_over_window == 1
    jsonl2 = bs.bon_generate_served(sd2, goals, out_dir=tmp_path / "out2", routes=[], n_max=1,
                                    sigma=0.0, trial=0, max_new_tokens=512, max_concurrency=2, model="gemma4e4b")
    assert not jsonl2.exists() or _rows(jsonl2) == []
    (erow,) = _rows(tmp_path / "out2" / bs.BON_ERRORS_FILE)
    assert erow["error"] == "prompt_over_window" and erow["window_reason"] == "floor_only_unresolvable"
    assert erow["prompt_tokens"] is None and erow["prompt_floor"] == 10240 - 512 + 1


def test_bon_over_client_and_over_proxy_pick_up_tokenize():
    class _Client:
        def chat_completions(self, payload): return {"choices": [{"message": {"content": "r"}}]}
        def tokenize(self, payload): return {"count": 5}
    sd = bs.ServedDefense.over_client(_Client(), virtual_model="petri__x", cell_id="x")
    assert sd._atokenize is not None
    assert asyncio.run(sd._atokenize({"model": "petri__x", "messages": []})) == {"count": 5}

    class _Proxy:
        async def handle_chat_completions(self, payload, authorization=None): return {}
        async def handle_tokenize(self, payload, authorization=None): return {"count": 9}
    sd = bs.ServedDefense.over_proxy(_Proxy(), virtual_model="petri__x", cell_id="x")
    assert asyncio.run(sd._atokenize({})) == {"count": 9}
