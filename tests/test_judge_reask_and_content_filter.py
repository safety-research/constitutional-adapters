"""(c1) ONE bounded re-ask for a malformed forced-tool verdict,
(c2) ``content_filter`` — the fourth abstention channel (2026-09-02), and
(c3) ``malformed_verdict`` — the fifth: a valid label with tool-call
scaffolding welded on, reproduced on the re-ask (2026-09-09; section at the
end of this file).

Observed live on the Qwen3.5 benign-controls sonnet re-grade:

* (c1) In AlpacaEval, 1-2 of ~1,600 requests per cell came back HTTP 200
  with a ``pairwise_verdict`` tool call MISSING its required ``verdict``
  field. Re-probed twice each, every one returned a well-formed verdict —
  sampling flakiness in the forced tool call, not a property of the item.
  The judge path had NO retry for a malformed response, so one flaky call
  failed a whole cell. Now: exactly ONE re-ask (same request, same cache
  key), then unverdicted — a genuinely broken rubric costs 2x and fails
  LOUDLY rather than looping; ``n_reasked`` keeps the count visible.
* (c2) One item returned, on both alpaca orders, 4/4 probes in ~2 s:
  ``400 invalid_request_error`` whose message names the content filtering
  policy. The API declining to EMIT the judge's reply is the judge
  declining by another channel — deterministic by construction, so no
  retry and no siblings discriminator. Matched on the FULL signature
  (status + error type + message), never status alone: our own malformed
  request is also a 400 and must stay a hard error. Precedent: the
  attack-judge lineage files the same error as a per-item
  ``{"__blocked__": True}`` with no retry
  (test_judge_routing.py::test_judge_sonnet_content_filter_400_is_a_blocked_item);
  the two treatments are one policy.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sisteer.capeval.judging as j                       # noqa: E402
from sisteer.capeval.judging import (                     # noqa: E402
    ClaudeJudge, JudgeCache, JudgeRequest, JudgingError, protocol_sha)

from test_capeval_judging import (                        # noqa: E402
    ALPACA_PAIR_RUBRIC_V1, ORBENCH_RUBRIC_V1, _Client, _Msg, _StatusError,
    _ToolUse, _cpi_msg, _single_req)


def _never(_params):
    raise AssertionError("the High-Prio org must never be called")


def _is_canary(params):
    return "tools" not in params


def _judge(tmp_path, low_fn, **kw):
    return ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"),
                       clients={"lowprio": _Client(low_fn),
                                "highprio": _Client(_never)},
                       concurrency=kw.pop("concurrency", 4), **kw)


def _malformed():
    """A 200 with the right tool but NO label field — the live shape."""
    return _Msg([_ToolUse("compliance_rating",
                          {"reasoning": "synthetic reasoning"})])


class _Filtered400(_StatusError):
    """The SDK's BadRequestError shape for the output-filter refusal."""

    def __init__(self, request_id="req_synthetic_cf",
                 message="Output blocked by content filtering policy",
                 err_type="invalid_request_error"):
        super().__init__(400)
        self.request_id = request_id
        self.type = err_type
        self.body = {"type": "error",
                     "error": {"type": err_type, "message": message},
                     "request_id": request_id}


# --- (c1) the bounded re-ask ------------------------------------------------------


def test_malformed_then_wellformed_is_verdicted_with_one_reask(tmp_path):
    seen = {"n": 0}

    def low(params):
        seen["n"] += 1
        return _malformed() if seen["n"] == 1 else _cpi_msg("partial_refusal")

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "a hedged answer")
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert out[req.key()] == {"label": "partial_refusal"}
    assert judge.n_reasked == 1
    assert seen["n"] == 2  # exactly one re-ask
    assert judge.n_api_calls == 1  # one RESOLVED request
    assert judge.cache.get(req.key()) == {"label": "partial_refusal"}
    assert judge.n_abstained == 0


def test_reask_sends_the_identical_request(tmp_path):
    calls = []

    def low(params):
        calls.append(params)
        return _malformed() if len(calls) == 1 else _cpi_msg("full_compliance")

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "an answer")
    judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert len(calls) == 2 and calls[0] == calls[1]  # same request, same key


def test_malformed_twice_is_unverdicted_and_bounded_at_one_reask(tmp_path):
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        return _malformed()

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "an answer")
    with pytest.raises(JudgingError, match="unverdicted") as ei:
        judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert "malformed verdict twice" in str(ei.value)
    assert seen["n"] == 2  # never a third try
    assert judge.n_reasked == 1
    assert judge.cache.get(req.key()) is None


def test_broken_rubric_costs_2x_then_fails_loud(tmp_path):
    """Every response malformed: each request is re-asked exactly once, the
    pass fails loudly, and the re-ask count equals the batch size — a
    number a reviewer sees, not a retry loop that hides the breakage."""
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        return _malformed()

    judge = _judge(tmp_path, low)
    reqs = [_single_req(f"synthetic prompt {i}", "answer") for i in range(5)]
    with pytest.raises(JudgingError, match="5 of 5 judge request"):
        judge.judge_all([(r, ORBENCH_RUBRIC_V1) for r in reqs])
    assert seen["n"] == 10 and judge.n_reasked == 5
    assert len(judge.cache) == 0


def test_wrong_tool_and_missing_block_are_malformed_and_reasked(tmp_path):
    seen = {}

    def low(params):
        body = params["messages"][0]["content"]
        seen[body] = seen.get(body, 0) + 1
        if seen[body] == 2:
            return _cpi_msg("full_compliance")
        if "wrongtool" in body:
            return _Msg([_ToolUse("some_other_tool", {"label": "x"})])
        return _Msg([])  # no tool_use block at all

    judge = _judge(tmp_path, low)
    a = _single_req("synthetic wrongtool prompt", "answer")
    b = _single_req("synthetic noblock prompt", "answer")
    out = judge.judge_all([(a, ORBENCH_RUBRIC_V1), (b, ORBENCH_RUBRIC_V1)])
    assert out[a.key()] == out[b.key()] == {"label": "full_compliance"}
    assert judge.n_reasked == 2


def test_reask_then_refusal_is_a_refusal_abstention(tmp_path):
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        if seen["n"] == 1:
            return _malformed()
        m = _Msg([])
        m.stop_reason = "refusal"
        return m

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "answer")
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    assert out[req.key()] == {"abstained": True, "reason": "refusal"}
    assert judge.n_reasked == 1 and judge.n_abstained == 1


def test_reask_does_not_apply_to_a_max_tokens_truncation(tmp_path):
    """A malformed verdict WITH stop_reason=max_tokens is the deterministic
    truncation class (its own abstention, no re-ask), not flakiness."""
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        m = _malformed()
        m.stop_reason = "max_tokens"
        return m

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic case puzzle prompt", "aLtErNaTiNg")
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    assert out[req.key()] == {"abstained": True, "reason": "max_tokens"}
    assert seen["n"] == 1 and judge.n_reasked == 0


# --- (c2) content_filter ------------------------------------------------------------


def test_content_filter_400_abstains_with_request_id_and_no_retry(tmp_path):
    seen = {"n": 0, "canary": 0}

    def low(params):
        if _is_canary(params):
            seen["canary"] += 1
            return _Msg([])
        seen["n"] += 1
        raise _Filtered400("req_synthetic_cf_1")

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "an answer")
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    # the sentinel names WHICH channel filed it (2026-09-02, the backend
    # seam): the anthropic content-filter signature, beside the request id
    assert out[req.key()] == {"abstained": True, "reason": "content_filter",
                              "signature": "anthropic:400:content_filtering_policy",
                              "request_id": "req_synthetic_cf_1"}
    assert seen["n"] == 1  # deterministic: never retried
    assert seen["canary"] == 0 and judge.n_canary_calls == 0
    assert judge.n_abstained == 1 and judge.n_transport_refusals == 0
    assert judge.cache.get(req.key()) == out[req.key()]
    assert judge.n_calls_by_org["highprio"] == 0


def test_content_filter_400_strict_mode_fails_loud(tmp_path):
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        raise _Filtered400()

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "an answer")
    with pytest.raises(JudgingError, match="unverdicted") as ei:
        judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert "JudgeContentFiltered" in str(ei.value)
    assert seen["n"] == 1
    assert judge.cache.get(req.key()) is None


def test_unrelated_400s_stay_hard_errors(tmp_path):
    """Status alone never qualifies: a 400 with a different message, or the
    right message under a different error type, is our bug — hard fail,
    no abstention, no retry."""
    for err in (_Filtered400(message="max_tokens: 1024 > model limit"),
                _Filtered400(err_type="authentication_error")):
        seen = {"n": 0}

        def low(_params, err=err):
            seen["n"] += 1
            raise err

        judge = _judge(tmp_path / str(id(err)), low)
        req = _single_req("synthetic prompt", "an answer")
        with pytest.raises(JudgingError, match="unverdicted") as ei:
            judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                            allow_refusal_abstention=True)
        assert "JudgeContentFiltered" not in str(ei.value)
        assert seen["n"] == 1 and judge.n_abstained == 0
        assert judge.cache.get(req.key()) is None


def test_content_filter_signature_requires_all_three_parts():
    assert j._is_content_filter_400(_Filtered400())
    assert not j._is_content_filter_400(_Filtered400(message="bad json"))
    assert not j._is_content_filter_400(
        _Filtered400(err_type="permission_error"))
    e = _Filtered400()
    e.status_code = 529
    assert not j._is_content_filter_400(e)
    # message may fall back to str(e) when the body carries no message
    bare = _StatusError(400)
    bare.type = "invalid_request_error"
    assert not j._is_content_filter_400(bare)
    bare2 = _StatusError("Output blocked by content filtering policy")
    bare2.status_code = 400
    bare2.type = "invalid_request_error"
    assert j._is_content_filter_400(bare2)


def test_pairwise_content_filter_records_both_orders(tmp_path):
    """Both alpaca orders of one item filtered: each order's sentinel keeps
    ITS request id, so the item-level abstention in judged.py can name
    both; the judge never retries either."""
    ids = iter(["req_synthetic_ab", "req_synthetic_ba"])
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        raise _Filtered400(next(ids))

    judge = _judge(tmp_path, low, concurrency=1)
    psha = protocol_sha(ALPACA_PAIR_RUBRIC_V1)
    ab = JudgeRequest(prompt="synthetic instruction", candidate_a="arm text",
                      candidate_b="base text", roles=("A", "B"),
                      rubric_sha=ALPACA_PAIR_RUBRIC_V1.sha,
                      judge_model=j.JUDGE_MODEL, protocol_sha=psha)
    ba = JudgeRequest(prompt="synthetic instruction", candidate_a="base text",
                      candidate_b="arm text", roles=("A", "B"),
                      rubric_sha=ALPACA_PAIR_RUBRIC_V1.sha,
                      judge_model=j.JUDGE_MODEL, protocol_sha=psha)
    out = judge.judge_all([(ab, ALPACA_PAIR_RUBRIC_V1),
                           (ba, ALPACA_PAIR_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    assert out[ab.key()]["reason"] == out[ba.key()]["reason"] == "content_filter"
    assert {out[ab.key()]["request_id"], out[ba.key()]["request_id"]} == {
        "req_synthetic_ab", "req_synthetic_ba"}
    assert seen["n"] == 2 and judge.n_abstained == 2


def test_content_filter_is_a_judge_refusal_subclass():
    assert issubclass(j.JudgeContentFiltered, j.JudgeRefusal)
    assert j.JudgeContentFiltered("m", request_id=None).reason == "content_filter"


def test_abstention_reasons_are_the_five_named_classes():
    assert j.ABSTENTION_REASONS == ("refusal", "max_tokens",
                                    "transport_refusal", "content_filter",
                                    "malformed_verdict")
    post = j.coordination_post(n_unique_requests=10)
    assert "content_filter" in post and "re-ask" in post


# --- (c3) the welded label: a valid enum member with tool-call scaffolding ---------
#
# Observed live on the alphasteer-full22 qwen35_4b selection (2026-09-09): the
# forced-tool ``label`` came back as ``full_refusal</parameterter>\n</invoke>\n``
# — a VALID label with stray tool-call scaffolding welded on — on both the ask
# and the re-ask (1 of 1,401 on two legs of six). Exact enum membership fails,
# the item is unverdicted, and the leg refuses a partial grade: every step
# correct, and no route at all for the case. It is the max_tokens argument
# again: under the frozen protocol the judge deterministically fails to emit a
# well-formed verdict for this body, so it ABSTAINS under its own reason
# (``malformed_verdict``) — never coerced to the label it probably meant.

WELDED = "full_refusal</parameterter>\n</invoke>\n"
#: the sentinel keys the pinned post-filing gate driver accepts on an
#: abstention (served_capability_judge.py @ c88e3daa, ``_ABSTAIN_KEYS``); a
#: key outside this set makes every filed rung UNCERTIFIED and stops the run
GATE_ABSTAIN_KEYS = {"abstained", "reason", "signature", "request_id",
                     "attempts", "siblings_succeeded", "evidence"}


def test_welded_label_twice_abstains_as_malformed_verdict_under_the_opt_in(tmp_path):
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        return _cpi_msg(WELDED)

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "an answer")
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    v = out[req.key()]
    assert v["abstained"] is True and v["reason"] == "malformed_verdict"
    assert v["attempts"] == 2  # the ask AND the re-ask, both welded
    assert seen["n"] == 2 and judge.n_reasked == 1  # the re-ask still happens first
    assert judge.n_abstained == 1 and judge.n_malformed_abstained == 1
    assert judge.n_api_calls == 1  # one RESOLVED request
    assert judge.cache.get(req.key()) == v  # cached like every abstention (stable re-reads)
    # the label is NOT taken: nothing on the sentinel is a label field
    assert "label" not in v
    assert set(v) <= GATE_ABSTAIN_KEYS


def test_welded_label_sentinel_records_field_head_and_shape_but_no_bytes(tmp_path):
    judge = _judge(tmp_path, lambda _p: _cpi_msg(WELDED))
    req = _single_req("synthetic prompt", "an answer")
    v = judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                        allow_refusal_abstention=True)[req.key()]
    sig = v["signature"]
    # which channel, which field, which enum member the head matched, and the
    # SHAPE of the trailer (alnum runs -> w, whitespace -> _) — recurrence is
    # diagnosable from the label-only cache without persisting judge bytes
    assert sig.startswith("parse:200:welded_label:label=full_refusal:")
    assert sig.endswith("</w>_</w>_")
    assert "parameterter" not in sig and "invoke" not in sig
    assert len(sig) <= 64 and not any(c.isspace() for c in sig)  # JB4 put rules


def test_welded_label_strict_mode_still_fails_loud(tmp_path):
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        return _cpi_msg(WELDED)

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "an answer")
    with pytest.raises(JudgingError, match="unverdicted") as ei:
        judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert "malformed verdict twice" in str(ei.value)
    assert seen["n"] == 2 and judge.n_abstained == 0
    assert judge.cache.get(req.key()) is None


def test_welded_label_then_wellformed_is_a_verdict_not_an_abstention(tmp_path):
    """A single weld that clears on the re-ask is the flakiness class the
    re-ask already covers: verdicted, counted in n_reasked, never abstained."""
    seen = {"n": 0}

    def low(_params):
        seen["n"] += 1
        return _cpi_msg(WELDED) if seen["n"] == 1 else _cpi_msg("full_refusal")

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "an answer")
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    assert out[req.key()] == {"label": "full_refusal"}
    assert judge.n_reasked == 1 and judge.n_abstained == 0
    assert judge.n_malformed_abstained == 0


def test_mixed_malformations_stay_hard_failures(tmp_path):
    """The abstention needs the weld shape REPRODUCED. A missing field on
    one ask and a weld on the other is two malformations, not one
    deterministic item property — it stays the loud hard failure."""
    for order in (("missing", "weld"), ("weld", "missing")):
        seen = {"n": 0}

        def low(_params, order=order):
            seen["n"] += 1
            kind = order[seen["n"] - 1]
            return _cpi_msg(WELDED) if kind == "weld" else _malformed()

        judge = _judge(tmp_path / "_".join(order), low)
        req = _single_req("synthetic prompt", "an answer")
        with pytest.raises(JudgingError, match="malformed verdict twice"):
            judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                            allow_refusal_abstention=True)
        assert seen["n"] == 2 and judge.n_abstained == 0
        assert judge.cache.get(req.key()) is None


def test_weld_recognizer_is_exact_never_a_prefix_or_fuzzy_match():
    """Only a value whose text BEFORE the first ``<`` is byte-for-byte one
    enum member is the weld shape. Everything else off-enum is the plain
    malformed class (re-ask, then hard fail) — no stripping, no case
    folding, no prefix match, no non-string coercion."""
    def kind(val):
        try:
            j._sanitize_verdict({"reasoning": "r", "label": val}, ORBENCH_RUBRIC_V1)
        except j.JudgeMalformedLabel:
            return "weld"
        except JudgingError:
            return "malformed"
        return "ok"

    assert kind("full_refusal") == "ok"
    assert kind(WELDED) == "weld"
    assert kind("partial_refusal<") == "weld"
    assert kind("full_refusal <x") == "malformed"     # trailing space before the <
    assert kind("full_refusal\n</invoke>") == "malformed"
    assert kind("full_refusa<x") == "malformed"       # a prefix of a member
    assert kind("Full_refusal<x") == "malformed"      # case
    assert kind("full_refusal.") == "malformed"       # off-enum without a <
    assert kind("not_a_label") == "malformed"
    assert kind(["full_refusal<"]) == "malformed"     # not a string
    assert kind(None) == "malformed"
    # the plain class is NOT the subclass, so strict membership still governs
    assert not issubclass(j.JudgeMalformedLabel, j.JudgeRefusal)


def test_welded_label_sentinel_carries_the_request_id_when_the_message_has_one(tmp_path):
    def low(_params):
        m = _cpi_msg(WELDED)
        m._request_id = "req_synthetic_weld"  # the SDK's top-level response attribute
        return m

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "an answer")
    v = judge.judge_all([(req, ORBENCH_RUBRIC_V1)],
                        allow_refusal_abstention=True)[req.key()]
    assert v["request_id"] == "req_synthetic_weld"


def test_malformed_verdict_is_a_named_abstention_class_the_tier_aggregates():
    from sisteer.capeval.tiers import judged
    assert "malformed_verdict" in j.ABSTENTION_REASONS
    assert judged._abstention_reason(
        "xstest", "xst0000-synthetic",
        {"abstained": True, "reason": "malformed_verdict"}) == "malformed_verdict"
    post = j.coordination_post(n_unique_requests=10)
    assert "malformed_verdict" in post and "re-ask" in post
