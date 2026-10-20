"""The continuation-logprob scorer (``serving/prefill_logprob.py``): the
HTTP normalization boundary on a fixture built from the pinned
vLLM 0.25.1 response schema, every refusal, the span rule in both thinking
renders, the exact ``ll_sum``, and HF equivalence on ``tiny_llama`` — the
teacher-forced continuation logprob sum equals the scorer's sum over a
synthetic HTTP response built from the same logits.

Everything synthetic: invented token ids (the gemma-4 header ids are a
public template fact), invented logprobs, no real session data.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from sisteer.capeval.serving import prefill_logprob as pl
from sisteer.capeval.serving import proxy as px

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "capeval" / "served" / "prompt_logprobs_http.json"


@pytest.fixture()
def fx() -> dict:
    return json.loads(FIX.read_text())


def _resp(fx) -> dict:
    return copy.deepcopy(fx["response"])


def _score(fx, resp=None, **kw):
    return pl.score_candidate(
        resp_json=resp if resp is not None else _resp(fx),
        tokenize_ids=fx["tokenize_ids"], header_ids=fx["header_ids"],
        n_chars=len(fx["candidate"].strip()),
        expected_candidate_ids=fx["expected_candidate_ids"], **kw)


# --------------------------------------------------------------------------- #
# the fixture IS the pinned wire shape                                         #
# --------------------------------------------------------------------------- #


def test_fixture_is_the_pinned_0_25_1_wire_shape(fx):
    r = fx["response"]
    plp = r["prompt_logprobs"]
    assert fx["vllm_pin"] == "0.25.1"
    assert plp[0] is None and len(plp) == len(fx["tokenize_ids"])
    for entry in plp[1:]:
        assert isinstance(entry, dict) and entry
        for k, v in entry.items():
            assert isinstance(k, str) and k.isdigit(), "wire keys are token-id STRINGS"
            assert set(v) == {"logprob", "rank", "decoded_token"}
    assert r["prompt_token_ids"] == fx["tokenize_ids"]
    assert r["choices"][0]["logprobs"] is None          # logprobs: false on the request
    assert r["usage"]["prompt_tokens"] == len(fx["tokenize_ids"])
    # the header occurs exactly once and the candidate is the tail
    ids = fx["tokenize_ids"]
    h = fx["header_ids"]
    assert sum(ids[i:i + 3] == h for i in range(len(ids) - 2)) == 1
    assert ids[-2:] == fx["expected_candidate_ids"]


def test_normalizer_accepts_the_fixture_and_converts_keys(fx):
    plp = pl.normalize_prompt_logprobs(_resp(fx), expected_len=len(fx["tokenize_ids"]))
    assert plp[0] is None
    assert plp[9] == {25510: -1.25, 25517: -3.25}
    assert all(isinstance(k, int) for e in plp[1:] for k in e)
    assert all(isinstance(v, float) for e in plp[1:] for v in e.values())
    # without expected_len the length is not checked
    assert len(pl.normalize_prompt_logprobs(_resp(fx))) == 11


def test_score_candidate_on_the_fixture_is_the_hand_sum(fx):
    cs = _score(fx)
    assert cs.ll_sum == pytest.approx(fx["expected_ll_sum"]) == pytest.approx(-1.25 - 1.5)
    assert cs.n_cont_tokens == 2 and cs.span_start == 9
    assert cs.token_ids == (25510, 26314)
    assert cs.prompt_tokens == 11 and cs.n_chars == len("carbon dioxide")


def test_prompt_token_ids_echo_is_optional_but_must_agree(fx):
    r = _resp(fx)
    del r["prompt_token_ids"]
    assert pl.prompt_token_ids_of(r) is None
    assert _score(fx, r).ll_sum == pytest.approx(-2.75)   # no echo: no cross-check
    r = _resp(fx)
    r["prompt_token_ids"][3] = 9999
    with pytest.raises(pl.PrefillLogprobError, match="prompt_token_ids"):
        _score(fx, r)
    r = _resp(fx)
    r["prompt_token_ids"] = "2,1000"
    with pytest.raises(pl.PrefillLogprobError, match="not a list of ints"):
        _score(fx, r)


# --------------------------------------------------------------------------- #
# refusals — every anomaly is a hard per-item error                            #
# --------------------------------------------------------------------------- #


def _mut(fx, fn):
    r = _resp(fx)
    fn(r)
    return r


def _set_key(r, pos, old, new):
    r["prompt_logprobs"][pos][new] = r["prompt_logprobs"][pos].pop(old)


REFUSALS = {
    "non_integer_key": (lambda r: _set_key(r, 9, "25510", "abc"), "base-10"),
    "int_key_not_string": (lambda r: _set_key(r, 9, "25510", 25510), "base-10"),
    "leading_zero_key": (lambda r: _set_key(r, 9, "25510", "025510"), "base-10"),
    "negative_key": (lambda r: _set_key(r, 9, "25510", "-1"), "base-10"),
    "missing_logprob": (lambda r: r["prompt_logprobs"][9]["25510"].pop("logprob"), "'logprob'"),
    "nan_logprob": (lambda r: r["prompt_logprobs"][9]["25510"].__setitem__("logprob", float("nan")), "finite"),
    "inf_logprob": (lambda r: r["prompt_logprobs"][10]["26314"].__setitem__("logprob", float("inf")), "finite"),
    "bool_logprob": (lambda r: r["prompt_logprobs"][9]["25510"].__setitem__("logprob", True), "finite"),
    "string_logprob": (lambda r: r["prompt_logprobs"][9]["25510"].__setitem__("logprob", "-1.25"), "finite"),
    "value_not_object": (lambda r: r["prompt_logprobs"][9].__setitem__("25510", -1.25), "'logprob'"),
    "wrong_length_short": (lambda r: r["prompt_logprobs"].pop(), "positions"),
    "wrong_length_long": (lambda r: r["prompt_logprobs"].append(None), "positions"),
    "not_a_list": (lambda r: r.__setitem__("prompt_logprobs", {"0": None}), "must be a list"),
    "missing_field": (lambda r: r.pop("prompt_logprobs"), "no 'prompt_logprobs'"),
    "entry_is_a_list": (lambda r: r["prompt_logprobs"].__setitem__(9, [["25510", -1.25]]), "null or an object"),
    "actual_token_absent_in_span": (lambda r: _set_key(r, 9, "25510", "25511"), "absent"),
    "null_inside_span": (lambda r: r["prompt_logprobs"].__setitem__(10, None), "null inside"),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_every_malformation_is_refused(fx, name):
    mutate, needle = REFUSALS[name]
    with pytest.raises(pl.PrefillLogprobError, match=needle):
        _score(fx, _mut(fx, mutate))


def test_normalizer_refuses_non_object_response():
    with pytest.raises(pl.PrefillLogprobError):
        pl.normalize_prompt_logprobs([None])  # type: ignore[arg-type]
    with pytest.raises(pl.PrefillLogprobError):
        pl.prompt_token_ids_of("x")  # type: ignore[arg-type]


def test_score_candidate_refuses_bad_n_chars_and_empty_ids(fx):
    for bad in (0, -1, True, 1.5):
        with pytest.raises(pl.PrefillLogprobError, match="n_chars"):
            pl.score_candidate(resp_json=_resp(fx), tokenize_ids=fx["tokenize_ids"],
                               header_ids=fx["header_ids"], n_chars=bad)
    with pytest.raises(pl.PrefillLogprobError, match="empty"):
        pl.score_candidate(resp_json=_resp(fx), tokenize_ids=[], header_ids=fx["header_ids"], n_chars=3)


# --------------------------------------------------------------------------- #
# the span rule                                                                #
# --------------------------------------------------------------------------- #

H = [105, 4368, 107]


def test_span_rule_locates_the_single_header_and_the_candidate_tail(fx):
    ids = fx["tokenize_ids"]
    span = pl.continuation_span(ids, H, expected_candidate_ids=[25510, 26314])
    assert (span.start, span.stop, span.header_at, span.scaffold) == (9, 11, 6, 0)
    assert list(span.positions) == [9, 10] and len(span) == 2
    # without expected ids: everything after the header
    assert pl.continuation_span(ids, H).positions == range(9, 11)


def test_span_rule_refusals():
    with pytest.raises(pl.PrefillLogprobError, match="not found"):
        pl.continuation_span([1, 2, 3, 4], H)
    with pytest.raises(pl.PrefillLogprobError, match="occurs 2 times"):
        pl.continuation_span([1, *H, 5, *H, 6], H)
    with pytest.raises(pl.PrefillLogprobError, match="empty continuation span"):
        pl.continuation_span([1, 2, *H], H)
    with pytest.raises(pl.PrefillLogprobError, match="does not end with"):
        pl.continuation_span([1, *H, 7, 8], H, expected_candidate_ids=[7, 9])
    with pytest.raises(pl.PrefillLogprobError, match="does not end with"):
        pl.continuation_span([1, *H, 7], H, expected_candidate_ids=[6, 7])   # longer than the tail
    with pytest.raises(pl.PrefillLogprobError, match="header_ids"):
        pl.continuation_span([1, 2], [], )
    with pytest.raises(pl.PrefillLogprobError, match="non-empty list of ints"):
        pl.continuation_span([1, *H, 7], H, expected_candidate_ids=[])


def test_span_rule_in_both_thinking_renders_and_the_diagnostic():
    cand = [25510, 26314]
    off = [2, 50, 51, 106, 108, *H, *cand]                 # thinking OFF: native render
    on_flag = [2, 9, 108, 50, 51, 106, 108, *H, *cand]     # ON flag-only: <|think|> (9) in the system turn
    on_diag = [2, 9, 108, 50, 51, 106, 108, *H, 200, 201, 202, 203, *cand]  # ON + empty thought block
    s_off = pl.continuation_span(off, H, expected_candidate_ids=cand)
    s_on = pl.continuation_span(on_flag, H, expected_candidate_ids=cand)
    assert (len(s_off), s_off.scaffold) == (2, 0)
    assert (len(s_on), s_on.scaffold) == (2, 0)
    assert s_on.start == s_off.start + 2   # shifted by the injected system-turn tokens only
    # the diagnostic render's thought block is SCAFFOLD: refused unless declared
    with pytest.raises(pl.PrefillLogprobError, match="between the assistant header"):
        pl.continuation_span(on_diag, H, expected_candidate_ids=cand)
    s_diag = pl.continuation_span(on_diag, H, expected_candidate_ids=cand, scaffold_ok=True)
    assert (s_diag.start, s_diag.stop, s_diag.scaffold) == (len(on_diag) - 2, len(on_diag), 4)
    # without expected ids the scaffold cannot be told apart — it is scored (caller's risk)
    assert len(pl.continuation_span(on_diag, H)) == 6


def test_continuation_logprob_is_the_exact_sum_and_refuses_gaps():
    ids = [2, *H, 7, 8]
    plp = [None, {105: -0.1}, {4368: -0.1}, {107: -0.1}, {7: -1.0, 9: -2.0}, {8: -0.5}]
    span = pl.continuation_span(ids, H)
    assert pl.continuation_logprob(plp, ids, span) == pytest.approx(-1.5)
    with pytest.raises(pl.PrefillLogprobError, match="positions"):
        pl.continuation_logprob(plp[:-1], ids, span)
    with pytest.raises(pl.PrefillLogprobError, match="absent"):
        pl.continuation_logprob([*plp[:4], {9: -2.0}, plp[5]], ids, span)
    with pytest.raises(pl.PrefillLogprobError, match="null inside"):
        pl.continuation_logprob([*plp[:5], None], ids, span)
    with pytest.raises(pl.PrefillLogprobError, match="empty"):
        pl.continuation_logprob(plp, ids, pl.ContinuationSpan(4, 4, 1))


# --------------------------------------------------------------------------- #
# request builders                                                             #
# --------------------------------------------------------------------------- #


def test_build_prefill_request_shape_declaration_and_thinking():
    lens = pl.build_prefill_request(virtual_model="petri__x", user_text="Q?", candidate=" carbon dioxide",
                                    system=None, response_prefill=True, enable_thinking=False)
    assert set(lens) <= set(pl.PREFILL_REQUEST_KEYS)
    assert lens["messages"] == [{"role": "user", "content": "Q?"},
                                {"role": "assistant", "content": " carbon dioxide"}]
    assert lens["continue_final_message"] is True and lens["add_generation_prompt"] is False
    assert lens["max_tokens"] == 1 and lens["temperature"] == 0.0 and lens["seed"] == 0
    assert lens["prompt_logprobs"] == 1 and lens["logprobs"] is False
    assert lens["chat_template_kwargs"] == {"enable_thinking": False}
    assert lens[px.RESPONSE_PREFILL_KEY] is True and lens["return_token_ids"] is True
    for forbidden in (*px.FORBIDDEN_CLIENT_KEYS, "n", "stream"):
        assert forbidden not in lens
    maskless = pl.build_prefill_request(virtual_model="petri__prompted", user_text="Q?", candidate="A",
                                        system="Be safe.", response_prefill=False, enable_thinking=True,
                                        seed=7, return_token_ids=False)
    assert px.RESPONSE_PREFILL_KEY not in maskless and "return_token_ids" not in maskless
    assert maskless["messages"][0] == {"role": "system", "content": "Be safe."}
    assert maskless["chat_template_kwargs"] == {"enable_thinking": True} and maskless["seed"] == 7
    diag = pl.build_prefill_request(virtual_model="petri__x", user_text="Q?", candidate="A", system=None,
                                    response_prefill=True, enable_thinking=True, thought_block_diag=True)
    assert diag["messages"][-1] == {"role": "assistant", "content": "A", "reasoning": " "}
    # fit-support transfer S3.9 (contract §2.5.4): the thinking-ON declaration
    # rides a LENS-route (response_prefill) body that sends enable_thinking:
    # true — the JSON boolean true, in PREFILL_REQUEST_KEYS — and no other:
    # the lens body under OFF and the maskless body under ON carry NO such key
    from sisteer.capeval.tinycap import contracts as C
    assert C.OOS_THINKING_DECLARATION_KEY in pl.PREFILL_REQUEST_KEYS
    assert diag[C.OOS_THINKING_DECLARATION_KEY] is True
    assert C.OOS_THINKING_DECLARATION_KEY not in lens             # lens + OFF
    assert C.OOS_THINKING_DECLARATION_KEY not in maskless         # base + ON
    lens_on = pl.build_prefill_request(virtual_model="petri__x", user_text="Q?", candidate="A", system=None,
                                       response_prefill=True, enable_thinking=True)
    assert lens_on[C.OOS_THINKING_DECLARATION_KEY] is True and lens_on[px.RESPONSE_PREFILL_KEY] is True
    assert set(lens_on) <= set(pl.PREFILL_REQUEST_KEYS)
    # the tokenize twin never carries it (tokenize applies the kwargs SHAPE validation only)
    tok_on = pl.build_tokenize_request(virtual_model="petri__x", messages=lens_on["messages"],
                                       enable_thinking=True)
    assert C.OOS_THINKING_DECLARATION_KEY not in tok_on and set(tok_on) == set(pl.TOKENIZE_REQUEST_KEYS)
    with pytest.raises(pl.PrefillLogprobError, match="EMPTY"):
        pl.build_prefill_request(virtual_model="petri__x", user_text="Q?", candidate="   ", system=None,
                                 response_prefill=True, enable_thinking=False)
    with pytest.raises(pl.PrefillLogprobError):
        pl.build_prefill_request(virtual_model="", user_text="Q?", candidate="A", system=None,
                                 response_prefill=True, enable_thinking=False)


def test_build_tokenize_request_mirrors_the_prefill_messages():
    body = pl.build_prefill_request(virtual_model="petri__x", user_text="Q?", candidate=" A", system=None,
                                    response_prefill=True, enable_thinking=True)
    tok = pl.build_tokenize_request(virtual_model="petri__x", messages=body["messages"], enable_thinking=True)
    assert set(tok) == set(pl.TOKENIZE_REQUEST_KEYS)
    assert tok["messages"] == body["messages"]
    assert tok["continue_final_message"] is True and tok["add_generation_prompt"] is False
    assert tok["chat_template_kwargs"] == {"enable_thinking": True}
    with pytest.raises(pl.PrefillLogprobError, match="assistant"):
        pl.build_tokenize_request(virtual_model="petri__x", messages=[{"role": "user", "content": "Q"}],
                                  enable_thinking=False)


def test_rendered_candidate_is_the_trimmed_content():
    assert pl.rendered_candidate(" carbon dioxide") == "carbon dioxide"
    assert pl.rendered_candidate("A") == "A"
    with pytest.raises(pl.PrefillLogprobError):
        pl.rendered_candidate(3)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# HF equivalence on tiny_llama                                                 #
# --------------------------------------------------------------------------- #


def test_hf_teacher_forced_sum_equals_the_scorer_over_a_synthetic_http_response(tiny_llama):
    import torch
    import torch.nn.functional as F

    model, tok = tiny_llama
    V = int(model.config.vocab_size)
    assert V > 4400, "the synthetic ids must be real vocabulary entries"
    header = [105, 4368, 107]
    prefix = [1, 500, 501, 502, 503, 44, 45]
    cand = [700, 701, 702, 703]
    ids = prefix + header + cand
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]            # [n, V]
    lp = F.log_softmax(logits.float(), dim=-1)                   # lp[p-1, ids[p]] = logprob of token p
    # the synthetic HTTP response: string keys, the actual token + one distractor per position
    plp = [None]
    for p in range(1, len(ids)):
        actual = ids[p]
        alt = (actual + 1) % V
        plp.append({str(actual): {"logprob": float(lp[p - 1, actual]), "rank": 1, "decoded_token": None},
                    str(alt): {"logprob": float(lp[p - 1, alt]), "rank": 2, "decoded_token": None}})
    resp = {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                         "logprobs": None, "finish_reason": "length"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1},
            "prompt_logprobs": plp, "prompt_token_ids": list(ids)}
    cs = pl.score_candidate(resp_json=resp, tokenize_ids=ids, header_ids=header, n_chars=12,
                            expected_candidate_ids=cand)
    start = len(prefix) + len(header)
    teacher_forced = -F.cross_entropy(logits[start - 1:len(ids) - 1].float(),
                                      torch.tensor(ids[start:]), reduction="sum").item()
    assert cs.ll_sum == pytest.approx(teacher_forced, abs=1e-4)
    assert cs.token_ids == tuple(cand) and cs.n_cont_tokens == 4 and cs.span_start == start
    assert cs.prompt_tokens == len(ids)
    assert math.isfinite(cs.ll_sum) and cs.ll_sum < 0
