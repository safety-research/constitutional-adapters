"""EvalPlus on the served route.

BASE lane (no evalplus, no modal, no network): the chat-item construction
against the vendored evalplus 0.3.1 prompt pieces; the ProxyGenerateEngine's
request bodies (declared prefill on lens routes ONLY, SI as system message
on prompted routes, thinking kwargs, seed, stream=false, no forbidden keys,
never n=); the fan-out's collect-don't-retry discipline; the response
ATTESTATION check (every persisted 200 carries the proxy's
``model`` / ``petri_intervention_hash`` / ``petri.cell_id`` /
``petri.serving_identity`` stamps equal to the client route — the fakes
emit VALID stamps from a routing table keyed by the request's model, and
the refusal cases file a typed per-item error, never a sidecar row); the
once-only context-length clamp (a context-length 400 is
retried ONCE at the server-stated room with every other field byte-identical,
a floor-only rejection resolves its exact length through ``/v1/tokenize`` on
the IDENTICAL render, any other 400 stays a generic error with no retry, and
an over-window prompt is the typed ``PromptOverWindow`` with no sidecar row);
the runner's
resume sidecar + atomic samples.jsonl + smoke padding; the ``:open``
extraction; ``validate_eval_results`` refusals; served row naming + the
frozen HF default; store conversion (rows, done marker, counts-only
summary); the sandbox orchestrator against a FAKE ``modal`` (create kwargs
per contract, transfer in/out, validate-before-write, terminate always);
and the generalized pairing (item_id / paired_id joins, engine refusal,
unmatched policy, McNemar only on binary outcomes).

HEAVY lane (``importorskip("evalplus")``, the lane venv): the chat-item
pieces equal what evalplus's own ``make_raw_chat_prompt`` hands the
tokenizer, and ``sanitize`` on a fenced ``:open`` generation. Everything is
SYNTHETIC — invented problems, canned completions, placeholder hashes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import types
import typing
from pathlib import Path

import pytest

from sisteer.capeval.serving.proxy import RESPONSE_PREFILL_KEY, RouteEntry
from sisteer.capeval.serving import prefill_logprob as pl
from sisteer.capeval.serving import proxy_engine as pe
from sisteer.capeval.serving import served_window as sw
from sisteer.capeval.tiers import evalplus_runner as er
from sisteer.capeval.tiers import pairing as pr
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
import evalplus_synthetic as EP  # noqa: E402  (tests/ helper: the synthetic EvalPlus sample pair)

REPO = Path(__file__).parents[1]

LENS = "proxy_vllm+lens+interleaved"
PASS = "proxy_vllm+passthrough+interleaved"
SI_TEXT = "You are a careful assistant. Refuse unsafe requests politely."

LENS_ROUTE = RouteEntry(virtual_model="petri__safety_refusal__psr_kl_all",
                        cell_id="safety_refusal__psr_kl_all", cell_class="class_a",
                        serving_identity=LENS, intervention_hash="cc" * 8)
PROMPTED_ROUTE = RouteEntry(virtual_model="petri__prompted__safety_refusal",
                            cell_id="prompted__safety_refusal", cell_class="prompted",
                            serving_identity=PASS, intervention_hash="dd" * 8,
                            si_prefix_hash="ee" * 8)

PROBLEMS = {
    "HumanEval/0": {"prompt": "\n\ndef has_close_elements(numbers, threshold):\n"
                              '    """Check whether any two numbers are closer than threshold."""\n',
                    "entry_point": "has_close_elements"},
    "HumanEval/1": {"prompt": "def separate_paren_groups(paren_string):\n"
                              '    """Split balanced paren groups."""\n',
                    "entry_point": "separate_paren_groups"},
}
TWO_IDS = frozenset(PROBLEMS)
GEN_CFG = {"do_sample": False, "max_new_tokens": 768, "until": ["```"]}

#: the fakes' routing table — what a real proxy resolves the request's
#: ``model`` against, and therefore what it stamps the response with
ROUTES = {r.virtual_model: r for r in (LENS_ROUTE, PROMPTED_ROUTE)}

CHAT_URL = "http://127.0.0.1:1/v1/chat/completions"
TOKENIZE_URL = "http://127.0.0.1:1/v1/tokenize"


def proxy_stamps(route):
    """Exactly what ``RoutingProxy.handle_chat`` puts on a response served on
    ``route`` (proxy.py: top-level ``model`` + ``petri_intervention_hash``,
    plus the ``petri`` context block) — the serving identity as its raw
    string."""
    return {"model": route.virtual_model,
            "petri_intervention_hash": route.intervention_hash,
            "petri": {"virtual_model": route.virtual_model, "cell_id": route.cell_id,
                      "serving_identity": route.serving_identity,
                      "intervention_hash": route.intervention_hash}}


def _chat_json(text, *, finish="stop", reasoning=None, usage=True, stamps=None):
    msg = {"content": text}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    resp = {"choices": [{"message": msg, "finish_reason": finish}]}
    if usage:
        resp["usage"] = {"prompt_tokens": 120, "completion_tokens": 30}
    if stamps:
        resp.update(stamps)
    return json.dumps(resp)


def _proxy_400(message):
    """The proxy's JSON error body for an upstream vLLM 400 (``upstream_refusal``)."""
    return json.dumps({"error": {"message": f"upstream vLLM returned 400: {message}",
                                 "type": "invalid_request_error", "code": "upstream_error"}})


def _overflow_body(window, prompt, requested):
    """The pinned vLLM 0.25.1 chat-validation text (exact prompt length)."""
    return (f"This model's maximum context length is {window} tokens. However, you "
            f"requested {requested} output tokens and your prompt contains {prompt} "
            f"input tokens, for a total of {prompt + requested} tokens. Please reduce "
            "the length of the input prompt or the number of requested output tokens.")


def _floor_only_body(window, requested):
    """vLLM's PRODUCTION rejection: a LOWER BOUND on the prompt length only."""
    floor = window - requested + 1
    return (f"This model's maximum context length is {window} tokens. However, you "
            f"requested {requested} output tokens and your prompt contains at least "
            f"{floor} input tokens, for a total of at least {window + 1} tokens. Please "
            "reduce the length of the messages or completion.")


class FakeTransport:
    """Records every outbound (url, headers, body); canned completion per
    item, STAMPED the way the proxy stamps it (from the routing table, by
    the request's ``model``); ``fail`` names item_ids that get a generic
    (non-context-length) 400; ``stamps`` overrides the stamps callable
    (``route -> dict``) for the refusal cases."""

    def __init__(self, *, fail=(), text="\n    return True\n", reasoning=None, stamps=None):
        self.calls = []
        self.fail = set(fail)
        self.text = text
        self.reasoning = reasoning
        self.stamps = stamps or proxy_stamps

    async def __call__(self, url, headers, body):
        self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
        assert url == CHAT_URL, url
        if headers[C.HEADER_SAMPLE] in self.fail:      # "<item_id>/<sample_idx>"
            return 400, json.dumps({"error": {"message": "canned refusal", "code": "x"}})
        return 200, _chat_json(self.text, reasoning=self.reasoning,
                               stamps=self.stamps(ROUTES[body["model"]]))


class WindowedTransport:
    """A stub proxy with a real serving window: a chat request whose
    ``prompt_tokens + max_tokens`` exceeds ``window`` is 400-rejected with
    the pinned vLLM text (exact-length form, or the production FLOOR-ONLY
    form when ``floor_only``); otherwise a stamped completion. ``/v1/tokenize``
    answers ``{"count": prompt_tokens}`` (or ``tokenize_status`` / a
    shapeless body when ``tokenize_count`` is None). ``calls`` records both
    endpoints in order."""

    def __init__(self, *, window, prompt_tokens, floor_only=False, reject_always=False,
                 tokenize_status=200, tokenize_count="exact", text="\n    return True\n"):
        self.window = window
        self.prompt_tokens = prompt_tokens
        self.floor_only = floor_only
        self.reject_always = reject_always
        self.tokenize_status = tokenize_status
        self.tokenize_count = prompt_tokens if tokenize_count == "exact" else tokenize_count
        self.text = text
        self.calls = []

    @property
    def chat_calls(self):
        return [c for c in self.calls if c[0] == CHAT_URL]

    @property
    def tokenize_calls(self):
        return [c for c in self.calls if c[0] == TOKENIZE_URL]

    async def __call__(self, url, headers, body):
        self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
        if url == TOKENIZE_URL:
            if self.tokenize_status != 200:
                return self.tokenize_status, json.dumps({"error": {"message": "no tokenize here",
                                                                   "code": "tokenize_unsupported"}})
            if self.tokenize_count is None:
                return 200, json.dumps({"tokens": [1, 2, 3]})      # no integer count
            return 200, json.dumps({"count": self.tokenize_count, "max_model_len": self.window,
                                    "tokens": [], "token_strs": [], "model": body["model"]})
        assert url == CHAT_URL, url
        if self.reject_always or self.prompt_tokens + body["max_tokens"] > self.window:
            text = (_floor_only_body(self.window, body["max_tokens"]) if self.floor_only
                    else _overflow_body(self.window, self.prompt_tokens, body["max_tokens"]))
            return 400, _proxy_400(text)
        return 200, _chat_json(self.text, stamps=proxy_stamps(ROUTES[body["model"]]))


def _engine(route=LENS_ROUTE, *, system=None, thinking="off", transport=None, task=None,
            api_key=None, concurrency=4, max_model_len=None, reasoning_parser=None, sampling=None):
    return pe.ProxyGenerateEngine(
        "http://127.0.0.1:1", route=route, system=system, thinking=thinking,
        concurrency=concurrency, task=task or C.evalplus_task("humaneval"),
        api_key=api_key, transport=transport, max_model_len=max_model_len,
        reasoning_parser=reasoning_parser, sampling=sampling)


def _sans_max_tokens(body):
    return json.dumps({k: v for k, v in body.items() if k != "max_tokens"}, sort_keys=True)


# --------------------------------------------------------------------------- #
# chat items == evalplus's prompt pieces (vendored f-strings)                  #
# --------------------------------------------------------------------------- #


def test_chat_items_are_evalplus_prompt_pieces():
    items = er.render_chat_items(PROBLEMS)
    assert [i.item_id for i in items] == list(PROBLEMS)
    it = items[0]
    assert it.user == (f"{er.INSTRUCTION_PREFIX}\n```\n{PROBLEMS['HumanEval/0']['prompt'].strip()}\n```\n")
    # the prefill is evalplus's response f-string cut at the splitter
    assert it.assistant_prefill == er.EVALPLUS_ASSISTANT_PREFILL
    assert er.evalplus_response_content().split(er.EVALPLUS_MAGIC_SPLITTER)[0] == it.assistant_prefill
    assert it.assistant_prefill.endswith("```python\n")
    assert it.sample_idx == 0
    # the open variant has no prefill; limit is a head cap
    open_items = er.render_chat_items(PROBLEMS, open_variant=True, limit=1)
    assert len(open_items) == 1 and open_items[0].assistant_prefill is None
    assert open_items[0].user == it.user


def test_dataset_pins_are_vendored_and_exact():
    assert er.expected_task_ids("humaneval") == er.EXPECTED_TASK_IDS
    mbpp = er.expected_task_ids("mbpp")
    assert len(mbpp) == C.EXPECTED_MBPP_PLUS_N == 378
    assert all(t.startswith("Mbpp/") for t in mbpp)
    assert "Mbpp/2" in mbpp and "Mbpp/809" in mbpp and "Mbpp/0" not in mbpp  # not a range
    pin = er.dataset_pin("mbpp")
    assert pin.override_env == "MBPP_OVERRIDE_PATH" and pin.version == "v0.2.0"
    assert er.dataset_pin("humaneval").override_env == "HUMANEVAL_OVERRIDE_PATH"
    assert len(pin.gz_sha256) == 64 and len(pin.gz_md5) == 32
    with pytest.raises(er.EvalPlusSeamError, match="unknown evalplus dataset"):
        er.dataset_pin("apps")
    # the MBPP+ set is refused by the HumanEval validator and vice versa
    with pytest.raises(er.EvalPlusSeamError, match="378"):
        er.validate_task_ids(er.EXPECTED_TASK_IDS, where="t", expected=mbpp)


# --------------------------------------------------------------------------- #
# request bodies: the proxy contract, client-side                              #
# --------------------------------------------------------------------------- #


def test_lens_route_declares_the_continued_prefill():
    eng = _engine(LENS_ROUTE, thinking="off")
    item = er.render_chat_items(PROBLEMS)[0]
    body = eng.build_request(item, max_tokens=768, stop=("```",))
    assert body["model"] == LENS_ROUTE.virtual_model
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][-1]["content"] == er.EVALPLUS_ASSISTANT_PREFILL
    assert body["continue_final_message"] is True
    assert body["add_generation_prompt"] is False
    assert body[RESPONSE_PREFILL_KEY] is True          # E6: declared on the lens route
    assert body["temperature"] == 0.0 and body["max_tokens"] == 768
    assert body["stop"] == ["```"] and body["stream"] is False
    assert body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    assert "n" not in body and not (set(body) & set(C.FORBIDDEN_CLIENT_KEYS))
    assert body["seed"] == pe.stable_seed(eng.task, item.item_id, "0")
    # deterministic + distinct per item
    other = eng.build_request(er.render_chat_items(PROBLEMS)[1], max_tokens=768)
    assert other["seed"] != body["seed"]
    assert eng.build_request(item, max_tokens=768)["seed"] == body["seed"]


def test_prompted_route_sends_si_as_system_and_never_declares():
    eng = _engine(PROMPTED_ROUTE, system=SI_TEXT, thinking="on")
    item = er.render_chat_items(PROBLEMS)[0]
    body = eng.build_request(item, max_tokens=768, stop=("```",))
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant"]
    assert body["messages"][0]["content"] == SI_TEXT
    assert body["continue_final_message"] is True and body["add_generation_prompt"] is False
    assert RESPONSE_PREFILL_KEY not in body      # the proxy refuses it on a non-lens route
    assert body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: True}
    assert eng.headers(item)[C.HEADER_TASK] == C.evalplus_task("humaneval")
    assert eng.headers(item)[C.HEADER_SAMPLE] == "HumanEval/0/0"
    assert "Authorization" not in eng.headers(item)
    assert _engine(PROMPTED_ROUTE, api_key="tok").headers(item)["Authorization"] == "Bearer tok"


def test_open_variant_is_a_generation_prompt_render():
    for route in (LENS_ROUTE, PROMPTED_ROUTE):
        eng = _engine(route, system=SI_TEXT if route is PROMPTED_ROUTE else None, thinking="on",
                      task=C.evalplus_task("humaneval", open_variant=True))
        item = er.render_chat_items(PROBLEMS, open_variant=True)[0]
        body = eng.build_request(item, max_tokens=4096)
        assert body["messages"][-1]["role"] == "user"
        assert "continue_final_message" not in body and RESPONSE_PREFILL_KEY not in body
        assert "add_generation_prompt" not in body and "stop" not in body
        assert body["max_tokens"] == 4096


def test_client_body_contract_refusals():
    ok = {"model": "m", "messages": [{"role": "user", "content": "u"}], "stream": False}
    pe.assert_client_body(ok, lens=True)
    for bad, match in (
        ({**ok, "vllm_xargs": {}}, "FORBIDDEN"),
        ({**ok, "cache_salt": "x"}, "FORBIDDEN"),
        ({**ok, "expect_hash": "x"}, "FORBIDDEN"),
        ({**ok, "stream": True}, "stream"),
        ({**ok, "n": 4}, "n="),
        ({**ok, "messages": []}, "no messages"),
        ({**ok, RESPONSE_PREFILL_KEY: True}, "does not continue"),
        ({**ok, RESPONSE_PREFILL_KEY: "yes"}, "bool"),
    ):
        with pytest.raises(pe.ProxyEngineError, match=match):
            pe.assert_client_body(bad, lens=True)
    cont = {**ok, "messages": ok["messages"] + [{"role": "assistant", "content": "p"}],
            "continue_final_message": True, "add_generation_prompt": False}
    with pytest.raises(pe.ProxyEngineError, match="undeclared"):
        pe.assert_client_body(cont, lens=True)                 # lens + continued, no declaration
    pe.assert_client_body(cont, lens=False)                    # fine on a passthrough route
    with pytest.raises(pe.ProxyEngineError, match="non-lens"):
        pe.assert_client_body({**cont, RESPONSE_PREFILL_KEY: True}, lens=False)
    with pytest.raises(pe.ProxyEngineError, match="add_generation_prompt"):
        pe.assert_client_body({**cont, "add_generation_prompt": True, RESPONSE_PREFILL_KEY: True}, lens=True)
    with pytest.raises(pe.ProxyEngineError, match="assistant"):
        pe.assert_client_body({**ok, "continue_final_message": True, "add_generation_prompt": False}, lens=False)
    with pytest.raises(pe.ProxyEngineError, match="empty assistant prefill"):
        _engine().build_request(pe.ChatItem("x", "u", assistant_prefill=""), max_tokens=8)


def test_the_thinking_on_declaration_rides_every_lens_route_on_body_and_nothing_else():
    """Fit-support transfer S3.9 / T12 (contract §2.5.3–§2.5.6, §2.5.12): the
    generate engine puts ``petri_thinking_on: true`` beside
    ``chat_template_kwargs: {"enable_thinking": true}`` on a LENS route —
    required on EVERY such request regardless of support (400
    ``thinking_on_undeclared`` otherwise once the lens half lands) — and on
    nothing else: the base/prompted route under ON (400 ``thinking_on_not_lens``)
    and any route under OFF carry NO such key (asserted by absence). The client
    door (``assert_client_body``) mirrors the same predicate: it refuses the
    key on a non-lens body, the key with any value but the boolean true, the
    key without the ON kwargs, and a lens ON body WITHOUT the key."""
    key = C.OOS_THINKING_DECLARATION_KEY
    item = er.render_chat_items(PROBLEMS)[0]
    lens_on = _engine(LENS_ROUTE, thinking="on").build_request(item, max_tokens=8)
    assert lens_on[key] is True and lens_on[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: True}
    assert lens_on[RESPONSE_PREFILL_KEY] is True                  # the prefill declaration rides beside it
    for route, thinking in ((LENS_ROUTE, "off"), (PROMPTED_ROUTE, "on"), (PROMPTED_ROUTE, "off")):
        body = _engine(route, system=SI_TEXT if route is PROMPTED_ROUTE else None,
                       thinking=thinking).build_request(item, max_tokens=8)
        assert key not in body, (route.cell_id, thinking)
        assert body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: thinking == "on"}
    # the OFF lens body is byte-identical to before (the predicate adds nothing under OFF)
    off = _engine(LENS_ROUTE, thinking="off").build_request(item, max_tokens=8)
    assert set(off) == set(lens_on) - {key}
    # the client door
    ok_on = {"model": "m", "messages": [{"role": "user", "content": "u"}], "stream": False,
             **C.chat_template_kwargs("on")}
    with pytest.raises(pe.ProxyEngineError, match="thinking_on_undeclared"):
        pe.assert_client_body(ok_on, lens=True)                    # lens + ON, no declaration
    pe.assert_client_body({**ok_on, key: True}, lens=True)         # declared: fine
    pe.assert_client_body(ok_on, lens=False)                       # the base route needs none
    with pytest.raises(pe.ProxyEngineError, match="non-lens"):
        pe.assert_client_body({**ok_on, key: True}, lens=False)    # 400 thinking_on_not_lens
    for bad in (False, None, 1, "true"):                           # never false / null / another type
        with pytest.raises(pe.ProxyEngineError, match="JSON boolean"):
            pe.assert_client_body({**ok_on, key: bad}, lens=True)
    with pytest.raises(pe.ProxyEngineError, match="thinking_on_mismatch"):
        pe.assert_client_body({**ok_on, **C.chat_template_kwargs("off"), key: True}, lens=True)


def test_gen_cfg_resolution_refuses_sampling_and_unknown_knobs():
    assert pe.ProxyGenerateEngine.resolve_gen_cfg({**GEN_CFG, "batch_size": 8}) == (768, ("```",))
    assert pe.ProxyGenerateEngine.resolve_gen_cfg({"do_sample": False, "max_new_tokens": 5}) == (5, ())
    with pytest.raises(pe.ProxyEngineError, match="greedy"):
        pe.ProxyGenerateEngine.resolve_gen_cfg({**GEN_CFG, "do_sample": True})
    with pytest.raises(pe.ProxyEngineError, match="top_p"):
        pe.ProxyGenerateEngine.resolve_gen_cfg({**GEN_CFG, "top_p": 0.9})
    with pytest.raises(pe.ProxyEngineError, match="max_new_tokens"):
        pe.ProxyGenerateEngine.resolve_gen_cfg({"do_sample": False})
    with pytest.raises(ValueError):
        _engine(thinking="maybe")


def test_fanout_collects_errors_and_generate_is_all_or_nothing():
    tr = FakeTransport(fail={"HumanEval/1/0"}, reasoning="thinking…")
    eng = _engine(LENS_ROUTE, thinking="on", transport=tr, api_key="tok")
    items = er.render_chat_items(PROBLEMS)
    results, errors = eng.generate_items(items, GEN_CFG)
    assert set(results) == {"HumanEval/0/0"} and [e[0] for e in errors] == ["HumanEval/1"]
    assert "400" in errors[0][1]
    # a generic 400 is the untyped http kind — no retry, no record
    assert errors[0].kind == pe.ERR_HTTP and errors[0].record is None
    assert isinstance(errors[0], pe.ItemError) and errors[0].item_id == "HumanEval/1"
    r = results["HumanEval/0/0"]
    assert r.text == "\n    return True\n" and r.finish_reason == "stop"
    assert (r.prompt_tokens, r.completion_tokens) == (120, 30) and r.reasoning == "thinking…"
    assert r.seed == pe.stable_seed(eng.task, "HumanEval/0", "0")
    assert r.intervention_hash == LENS_ROUTE.intervention_hash      # the VERIFIED stamp
    assert r.max_tokens_clamped_from is None and r.prompt_tokens_source is None
    assert eng.stats.as_dict() == {"n_attempts": 2, "n_rejected": 1, "n_clamped": 0,
                                   "n_prompt_over_window": 0, "n_tokenize_resolved": 0,
                                   "n_floor_only_unresolvable": 0, "n_stamp_mismatch": 0,
                                   "n_content_empty_with_reasoning": 0, "n_empty_response": 0,
                                   "window_mismatches": []}
    assert len(tr.calls) == 2
    url, headers, body = tr.calls[0]
    assert url == "http://127.0.0.1:1/v1/chat/completions"
    assert headers["Authorization"] == "Bearer tok" and headers[C.HEADER_TASK] == eng.task
    assert body[RESPONSE_PREFILL_KEY] is True and body["stream"] is False
    # the Engine seam refuses to return a partial list, and refuses an arm / a string
    with pytest.raises(pe.ProxyEngineError, match="FAILED"):
        eng.generate(items, None, GEN_CFG)
    with pytest.raises(pe.ProxyEngineError, match="arm=None"):
        eng.generate(items, object(), GEN_CFG)
    with pytest.raises(pe.ProxyEngineError, match="ChatItem"):
        eng.generate(["a rendered string"], None, GEN_CFG)
    ok = _engine(LENS_ROUTE, transport=FakeTransport())
    outs = ok.generate(items, None, GEN_CFG)
    assert [o.text for o in outs] == ["\n    return True\n"] * 2 and all(o.ended_with_eos for o in outs)


def test_parse_chat_response_refuses_malformed():
    with pytest.raises(pe.ProxyEngineError, match="exactly one choice"):
        pe.parse_chat_response({"choices": []})
    with pytest.raises(pe.ProxyEngineError, match="message"):
        pe.parse_chat_response({"choices": [{"finish_reason": "stop"}]})
    got = pe.parse_chat_response({"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
    assert got["text"] == "" and got["finish_reason"] == "length" and got["prompt_tokens"] is None
    assert got["reasoning"] is None and got["content_empty_with_reasoning"] is False
    with pytest.raises(pe.ProxyEngineError, match="reasoning channel"):
        pe.parse_chat_response({"choices": [{"message": {"content": "x", "reasoning_content": 3}}]})


def test_parse_chat_response_reads_both_channels():
    """Under a composed parser vLLM splits the message: BOTH channels are
    read from every response, ``text`` is content only, and the flag is
    exactly 'reasoning present, content empty/None'."""
    both = pe.parse_chat_response({"choices": [{"message": {"content": "ans", "reasoning_content": "thought"},
                                                "finish_reason": "stop"}]})
    assert (both["text"], both["reasoning"], both["content_empty_with_reasoning"]) == ("ans", "thought", False)
    for content in ("", None):
        empty = pe.parse_chat_response({"choices": [{"message": {"content": content, "reasoning_content": "thought"},
                                                     "finish_reason": "length"}]})
        assert (empty["text"], empty["reasoning"], empty["content_empty_with_reasoning"]) == ("", "thought", True)
    neither = pe.parse_chat_response({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
    assert (neither["text"], neither["reasoning"], neither["content_empty_with_reasoning"]) == ("", None, False)
    # an EMPTY reasoning string is no reasoning (never flags an ordinary empty answer)
    blank = pe.parse_chat_response({"choices": [{"message": {"content": "", "reasoning_content": ""}}]})
    assert blank["reasoning"] == "" and blank["content_empty_with_reasoning"] is False
    # the newer field name is the same channel
    newer = pe.parse_chat_response({"choices": [{"message": {"content": "", "reasoning": "r"}}]})
    assert newer["reasoning"] == "r" and newer["content_empty_with_reasoning"] is True


# --------------------------------------------------------------------------- #
# The response stamps must attest the client route                            #
# --------------------------------------------------------------------------- #


def test_stamps_are_compared_as_the_proxy_encodes_them():
    """String equality on the four stamps — the proxy stamps the routing-table
    strings verbatim (``serving_identity`` as ``<backend>+<path>+<mode>``,
    never parsed), so like is compared with like."""
    assert pe.STAMP_FIELDS == ("model", "petri_intervention_hash", "petri.cell_id",
                               "petri.serving_identity")
    exp = pe.expected_stamps(LENS_ROUTE)
    assert exp == {"model": LENS_ROUTE.virtual_model,
                   "petri_intervention_hash": LENS_ROUTE.intervention_hash,
                   "petri.cell_id": LENS_ROUTE.cell_id,
                   "petri.serving_identity": LENS}
    assert all(isinstance(v, str) for v in exp.values())
    good = json.loads(_chat_json("x", stamps=proxy_stamps(LENS_ROUTE)))
    seen = pe.assert_response_stamps(good, LENS_ROUTE, item_id="i")
    assert seen == exp and pe.response_stamps(good) == exp
    # the nested block is context, not the contract: only the four are read
    assert "petri.virtual_model" not in seen and "petri.intervention_hash" not in seen
    # absent stamps read as None (a non-object petri block counts as absent)
    assert pe.response_stamps({"choices": []}) == dict.fromkeys(pe.STAMP_FIELDS)
    assert pe.response_stamps({"petri": "not-an-object"})["petri.cell_id"] is None


@pytest.mark.parametrize("case", ["stale_hash", "wrong_cell", "wrong_identity", "wrong_model",
                                  "missing_all", "missing_petri_block"])
def test_stamp_mismatch_is_typed_with_seen_vs_expected(case):
    stamps = proxy_stamps(LENS_ROUTE)
    if case == "stale_hash":            # an older fitted object under the same virtual name
        stamps["petri_intervention_hash"] = "ab" * 8
        stamps["petri"]["intervention_hash"] = "ab" * 8
        bad, seen_val = ["petri_intervention_hash"], "ab" * 8
    elif case == "wrong_cell":
        stamps["petri"]["cell_id"] = "safety_refusal__vector_all"
        bad, seen_val = ["petri.cell_id"], "safety_refusal__vector_all"
    elif case == "wrong_identity":      # passthrough identity returned on a lens route
        stamps["petri"]["serving_identity"] = PASS
        bad, seen_val = ["petri.serving_identity"], PASS
    elif case == "wrong_model":
        stamps["model"] = PROMPTED_ROUTE.virtual_model
        bad, seen_val = ["model"], PROMPTED_ROUTE.virtual_model
    elif case == "missing_all":
        stamps, bad, seen_val = {}, list(pe.STAMP_FIELDS), None
    else:
        del stamps["petri"]
        bad, seen_val = ["petri.cell_id", "petri.serving_identity"], None
    resp = json.loads(_chat_json("x", stamps=stamps))
    with pytest.raises(pe.StampMismatch) as ei:
        pe.assert_response_stamps(resp, LENS_ROUTE, item_id="HumanEval/0")
    err = ei.value
    assert isinstance(err, pe.ProxyEngineError) and err.item_id == "HumanEval/0"
    assert err.expected == pe.expected_stamps(LENS_ROUTE)
    if seen_val is None:
        assert err.missing == bad and err.mismatched == []
        assert all(err.seen[k] is None for k in bad)
    else:
        assert err.mismatched == bad and err.missing == []
        assert err.seen[bad[0]] == seen_val
        assert repr(seen_val) in str(err) and repr(err.expected[bad[0]]) in str(err)
    rec = err.record()
    assert rec["error"] == pe.ERR_STAMP_MISMATCH == "stamp_mismatch"
    assert rec["expected"] == err.expected and rec["seen"] == err.seen
    assert set(rec) == {"error", "expected", "seen", "missing", "mismatched"}


@pytest.mark.parametrize("case", ["stale_hash", "wrong_cell", "wrong_identity", "wrong_model",
                                  "missing_all"])
def test_unattested_200_is_a_per_item_error_never_a_sidecar_row(tmp_path, offline_runner, case):
    """The fan-out and the runner: a 200 whose stamps do not name the route is
    refused per item — no GenResult, no generations.jsonl row, the typed
    record (seen vs expected) in gen_meta's ``errors`` — while the correctly
    stamped item on the same fan-out is persisted."""
    def stamps(route):
        st = proxy_stamps(route)
        if case == "stale_hash":
            st["petri_intervention_hash"] = "ab" * 8
        elif case == "wrong_cell":
            st["petri"]["cell_id"] = "another_cell"
        elif case == "wrong_identity":
            st["petri"]["serving_identity"] = PASS
        elif case == "wrong_model":
            st["model"] = "petri__something_else"
        else:
            st = {}
        return st

    class Selective(FakeTransport):
        """only HumanEval/1 comes back mis-stamped"""

        async def __call__(self, url, headers, body):
            if headers[C.HEADER_SAMPLE] == "HumanEval/1/0":
                self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
                return 200, _chat_json(self.text, stamps=stamps(ROUTES[body["model"]]))
            return await super().__call__(url, headers, body)

    tr = Selective()
    eng = _engine(LENS_ROUTE, transport=tr)
    results, errors = eng.generate_items(er.render_chat_items(PROBLEMS), GEN_CFG)
    assert set(results) == {"HumanEval/0/0"}
    [err] = errors
    assert err.item_id == "HumanEval/1" and err.kind == pe.ERR_STAMP_MISMATCH
    assert err.message.startswith("StampMismatch: HumanEval/1:")
    assert err.record["expected"] == pe.expected_stamps(LENS_ROUTE)
    if case == "missing_all":
        assert err.record["missing"] == list(pe.STAMP_FIELDS)
    else:
        assert len(err.record["mismatched"]) == 1 and err.record["missing"] == []
        k = err.record["mismatched"][0]
        assert err.record["seen"][k] != err.record["expected"][k]
    assert eng.stats.n_stamp_mismatch == 1 and eng.stats.n_rejected == 0   # a 200, refused HERE
    assert len(tr.calls) == 2                                              # no retry of any kind

    cell = tmp_path / "cells" / "k"
    with pytest.raises(er.EvalPlusSeamError, match="stamp_mismatch") as ei:
        er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=Selective()),
                                   dataset="humaneval", cell_dir=cell,
                                   problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    assert "re-run to fill the gaps" in str(ei.value)
    tdir = er.served_task_dir(cell, "humaneval")
    side = er.read_generations(tdir / er.GENERATIONS_FILENAME)
    assert set(side) == {"HumanEval/0/0"}                                    # never the refused one
    assert side["HumanEval/0/0"]["intervention_hash"] == LENS_ROUTE.intervention_hash
    assert not (tdir / er.SAMPLES_FILENAME).exists()
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "incomplete" and meta["n_stamp_mismatch"] == 1
    assert meta["n_requests"] == 2 and meta["n_rejected"] == 0 and meta["n_errors"] == 1
    [e] = meta["errors"]
    assert e["item_id"] == "HumanEval/1" and e["kind"] == "stamp_mismatch"
    assert e["record"]["expected"] == pe.expected_stamps(LENS_ROUTE)
    assert meta["route"]["intervention_hash"] == LENS_ROUTE.intervention_hash
    # the right stamps on the next invocation fill the gap and the counter persists
    er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=FakeTransport()),
                               dataset="humaneval", cell_dir=cell,
                               problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "complete" and meta["n_stamp_mismatch"] == 1
    assert meta["n_requests"] == 3 and "errors" not in meta and meta["n_errors"] == 0


def test_prompted_route_stamps_verify_against_its_own_route():
    """The passthrough identity is the RIGHT stamp on the prompted route (and
    the lens identity would be the wrong one there)."""
    tr = FakeTransport()
    eng = _engine(PROMPTED_ROUTE, system=SI_TEXT, thinking="on", transport=tr)
    results, errors = eng.generate_items(er.render_chat_items(PROBLEMS), GEN_CFG)
    assert errors == [] and len(results) == 2
    assert {r.intervention_hash for r in results.values()} == {PROMPTED_ROUTE.intervention_hash}
    swapped = FakeTransport(stamps=lambda route: proxy_stamps(LENS_ROUTE))
    _, errors = _engine(PROMPTED_ROUTE, system=SI_TEXT, transport=swapped).generate_items(
        er.render_chat_items(PROBLEMS), GEN_CFG)
    assert [e.kind for e in errors] == [pe.ERR_STAMP_MISMATCH] * 2
    assert sorted(errors[0].record["mismatched"]) == ["model", "petri.cell_id",
                                                      "petri.serving_identity",
                                                      "petri_intervention_hash"]


# --------------------------------------------------------------------------- #
# A context-length 400 is clamped ONCE, never a permanent gap                 #
# --------------------------------------------------------------------------- #


def test_context_length_400_is_retried_once_at_the_clamp_with_an_otherwise_identical_body(
        tmp_path, offline_runner):
    # window 8704, prompt 8000: 768 requested -> 704 of room -> ONE retry at 704
    tr = WindowedTransport(window=8704, prompt_tokens=8000)
    eng = _engine(LENS_ROUTE, transport=tr, max_model_len=8704)
    items = er.render_chat_items(PROBLEMS)
    results, errors = eng.generate_items(items[:1], GEN_CFG)
    assert errors == [] and set(results) == {"HumanEval/0/0"}
    assert tr.tokenize_calls == []                            # exact length was in the rejection
    first, retry = [c[2] for c in tr.chat_calls]
    assert (first["max_tokens"], retry["max_tokens"]) == (768, 704)
    assert _sans_max_tokens(first) == _sans_max_tokens(retry)  # byte-identical otherwise
    assert retry["seed"] == first["seed"] and retry[RESPONSE_PREFILL_KEY] is True
    assert retry["messages"] == first["messages"] and retry["stop"] == ["```"]
    h_first, h_retry = (c[1] for c in tr.chat_calls)
    assert h_first == h_retry                                  # same headers
    r = results["HumanEval/0/0"]
    assert r.max_tokens_clamped_from == 768 and r.prompt_tokens_source == sw.SOURCE_ERROR_MESSAGE
    assert r.intervention_hash == LENS_ROUTE.intervention_hash
    st = eng.stats
    assert (st.n_attempts, st.n_rejected, st.n_clamped, st.n_prompt_over_window) == (2, 1, 1, 0)
    assert st.window_mismatches == []                         # threaded == server
    # through the runner: the clamped row is persisted with its provenance,
    # the counters land in gen_meta, and the task COMPLETES
    cell = tmp_path / "cells" / "k"
    tr2 = WindowedTransport(window=8704, prompt_tokens=8000)
    summary = er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=tr2, max_model_len=8704),
                                         dataset="humaneval", cell_dir=cell,
                                         problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    # two items, each: the 768 attempt then its single 704 retry (fan-out order varies)
    assert sorted(c[2]["max_tokens"] for c in tr2.chat_calls) == [704, 704, 768, 768]
    assert tr2.tokenize_calls == []
    tdir = er.served_task_dir(cell, "humaneval")
    side = er.read_generations(tdir / er.GENERATIONS_FILENAME)
    assert all(row["max_tokens_clamped_from"] == 768 for row in side.values())
    assert all(row["prompt_tokens_source"] == "error_message" for row in side.values())
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "complete" and meta["n_requests"] == 4 and meta["n_rejected"] == 2
    assert meta["window_clamp"] == {"max_model_len": 8704, "min_clamped_completion": 64,
                                    "n_clamped": 2, "n_prompt_over_window": 0,
                                    "n_tokenize_resolved": 0, "n_floor_only_unresolvable": 0,
                                    "window_mismatches": []}
    assert summary["n_requests"] == 4 and summary["n_rejected"] == 2 and summary["n"] == 2
    # an UNCLAMPED row carries none of the clamp keys
    tr3 = FakeTransport()
    er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=tr3), dataset="mbpp",
                               cell_dir=cell, problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    plain = er.read_generations(er.served_task_dir(cell, "mbpp") / er.GENERATIONS_FILENAME)
    assert "max_tokens_clamped_from" not in plain["HumanEval/0/0"]
    assert plain["HumanEval/0/0"]["intervention_hash"] == LENS_ROUTE.intervention_hash


def test_floor_only_400_resolves_the_exact_length_via_tokenize_on_the_identical_render():
    """The production rejection states only a lower bound; the engine POSTs the
    IDENTICAL render to /v1/tokenize — for the prefilled instrument that is
    Lane A's build_tokenize_request (continue_final_message, add_generation_prompt
    false, the assistant seed, chat_template_kwargs), never a generic
    add_generation_prompt tokenize."""
    tr = WindowedTransport(window=8704, prompt_tokens=8100, floor_only=True)
    eng = _engine(LENS_ROUTE, thinking="on", transport=tr)
    item = er.render_chat_items(PROBLEMS)[0]
    results, errors = eng.generate_items([item], GEN_CFG)
    assert errors == [] and results["HumanEval/0/0"].max_tokens_clamped_from == 768
    assert results["HumanEval/0/0"].prompt_tokens_source == sw.SOURCE_TOKENIZE
    assert [c[0] for c in tr.calls] == [CHAT_URL, TOKENIZE_URL, CHAT_URL]
    first, tok, retry = [c[2] for c in tr.calls]
    assert retry["max_tokens"] == 8704 - 8100 == 604
    assert _sans_max_tokens(first) == _sans_max_tokens(retry)
    # the tokenize body IS the prefill render's tokenize request
    assert tok == pl.build_tokenize_request(virtual_model=LENS_ROUTE.virtual_model,
                                            messages=first["messages"], enable_thinking=True)
    assert set(tok) == set(pl.TOKENIZE_REQUEST_KEYS)
    assert tok["continue_final_message"] is True and tok["add_generation_prompt"] is False
    assert tok["messages"][-1] == {"role": "assistant", "content": er.EVALPLUS_ASSISTANT_PREFILL}
    assert tok["messages"] == first["messages"]
    assert tok[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: True} == first[C.CHAT_TEMPLATE_KWARGS_KEY]
    assert tok["model"] == LENS_ROUTE.virtual_model          # the proxy resolves it like chat
    assert RESPONSE_PREFILL_KEY not in tok and "seed" not in tok and "max_tokens" not in tok
    assert not (set(tok) & set(C.FORBIDDEN_CLIENT_KEYS))
    # the tokenize call carries the same auth/ledger headers as the chat call
    assert tr.calls[1][1] == tr.calls[0][1]
    assert eng.stats.n_tokenize_resolved == 1 and eng.stats.n_clamped == 1
    # the OPEN variant (no prefill) tokenizes its generation-prompt render, with
    # the same chat_template_kwargs (the thinking flag changes the render)
    tr = WindowedTransport(window=8704, prompt_tokens=8100, floor_only=True)
    eng = _engine(LENS_ROUTE, thinking="on", transport=tr,
                  task=C.evalplus_task("humaneval", open_variant=True))
    open_item = er.render_chat_items(PROBLEMS, open_variant=True)[0]
    results, errors = eng.generate_items([open_item], {"do_sample": False, "max_new_tokens": 4096})
    assert errors == [] and results["HumanEval/0/0"].max_tokens_clamped_from == 4096
    first, tok, retry = [c[2] for c in tr.calls]
    assert retry["max_tokens"] == 604 and _sans_max_tokens(first) == _sans_max_tokens(retry)
    assert tok == {"model": LENS_ROUTE.virtual_model, "messages": first["messages"],
                   "add_generation_prompt": True,
                   C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: True}}
    assert tok["messages"][-1]["role"] == "user" and "continue_final_message" not in tok


def test_tokenize_body_for_dispatches_on_the_render():
    eng = _engine(PROMPTED_ROUTE, system=SI_TEXT, thinking="off")
    pre = eng.build_request(er.render_chat_items(PROBLEMS)[0], max_tokens=8)
    tok = pe.tokenize_body_for(pre)
    assert tok["continue_final_message"] is True and tok["add_generation_prompt"] is False
    assert [m["role"] for m in tok["messages"]] == ["system", "user", "assistant"]
    assert tok["messages"][0]["content"] == SI_TEXT                # the SI rides the count
    assert tok[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    opn = eng.build_request(er.render_chat_items(PROBLEMS, open_variant=True)[0], max_tokens=8)
    tok = pe.tokenize_body_for(opn)
    assert tok["add_generation_prompt"] is True and "continue_final_message" not in tok
    assert tok[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    with pytest.raises(pe.ProxyEngineError, match="chat_template_kwargs"):
        pe.tokenize_body_for({k: v for k, v in opn.items() if k != C.CHAT_TEMPLATE_KWARGS_KEY})


def test_a_400_that_is_not_context_length_is_a_generic_error_with_no_retry():
    tr = FakeTransport(fail={"HumanEval/0/0"})
    eng = _engine(LENS_ROUTE, transport=tr)
    results, errors = eng.generate_items(er.render_chat_items(PROBLEMS)[:1], GEN_CFG)
    assert results == {} and [e.kind for e in errors] == [pe.ERR_HTTP]
    assert len(tr.calls) == 1 and "canned refusal" in errors[0].message
    assert eng.stats.n_attempts == 1 and eng.stats.n_rejected == 1 and eng.stats.n_clamped == 0

    class Salt409(WindowedTransport):
        """the context-length WORDS under another status are not an overflow"""

        async def __call__(self, url, headers, body):
            self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
            return 409, _proxy_400(_overflow_body(8704, 8000, body["max_tokens"]))

    tr = Salt409(window=8704, prompt_tokens=8000)
    eng = _engine(LENS_ROUTE, transport=tr)
    results, errors = eng.generate_items(er.render_chat_items(PROBLEMS)[:1], GEN_CFG)
    assert [e.kind for e in errors] == [pe.ERR_HTTP] and len(tr.calls) == 1
    assert "409" in errors[0].message and eng.stats.n_prompt_over_window == 0


def test_over_window_prompt_is_the_typed_error_and_never_a_sidecar_row(tmp_path, offline_runner):
    # 14 tokens of room (< MIN_CLAMPED_COMPLETION): no retry, the typed error
    tr = WindowedTransport(window=8704, prompt_tokens=8690)
    eng = _engine(LENS_ROUTE, transport=tr, max_model_len=8704)
    results, errors = eng.generate_items(er.render_chat_items(PROBLEMS)[:1], GEN_CFG)
    assert results == {} and len(tr.calls) == 1                  # no pointless retry
    [err] = errors
    assert err.kind == pe.ERR_PROMPT_OVER_WINDOW == "prompt_over_window"
    assert err.message.startswith("PromptOverWindow: prompt_over_window: prompt 8690")
    assert err.record["error"] == sw.PROMPT_OVER_WINDOW
    assert err.record["prompt_tokens"] == 8690 and err.record["requested_max_tokens"] == 768
    assert err.record["server_window"] == 8704 and err.record["completion_tokens"] == 0
    assert "< 64" in err.record["window_reason"]
    assert eng.stats.n_prompt_over_window == 1 and eng.stats.n_clamped == 0
    # the runner: nothing in the sidecar for it, gen_meta names it, the raise says why
    cell = tmp_path / "cells" / "k"

    class OnlyOneOver(WindowedTransport):
        """HumanEval/1 fits comfortably; HumanEval/0 does not"""

        async def __call__(self, url, headers, body):
            if headers[C.HEADER_SAMPLE] == "HumanEval/1/0":
                self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
                return 200, _chat_json(self.text, stamps=proxy_stamps(ROUTES[body["model"]]))
            return await super().__call__(url, headers, body)

    with pytest.raises(er.EvalPlusSeamError, match="prompt_over_window") as ei:
        er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=OnlyOneOver(window=8704, prompt_tokens=8690)),
                                   dataset="humaneval", cell_dir=cell,
                                   problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    assert "cannot complete at this window" in str(ei.value) and "HumanEval/0" in str(ei.value)
    tdir = er.served_task_dir(cell, "humaneval")
    assert set(er.read_generations(tdir / er.GENERATIONS_FILENAME)) == {"HumanEval/1/0"}
    assert not (tdir / er.SAMPLES_FILENAME).exists()
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "incomplete" and meta["n_requests"] == 2 and meta["n_rejected"] == 1
    assert meta["window_clamp"]["n_prompt_over_window"] == 1 and meta["window_clamp"]["n_clamped"] == 0
    [e] = meta["errors"]
    assert e["item_id"] == "HumanEval/0" and e["kind"] == "prompt_over_window"
    assert e["record"]["prompt_tokens"] == 8690 and e["record"]["error"] == "prompt_over_window"
    # the prompt-only form (the prompt ALONE exceeds the window) is the same typed error
    class PromptOnly(WindowedTransport):
        async def __call__(self, url, headers, body):
            self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
            return 400, _proxy_400("The decoder prompt (length 16920) is longer than the maximum "
                                   "model length of 16384. Make sure that max_model_len is no "
                                   "smaller than the number of text tokens.")

    _, errors = _engine(LENS_ROUTE, transport=PromptOnly(window=16384, prompt_tokens=16920)).generate_items(
        er.render_chat_items(PROBLEMS)[:1], GEN_CFG)
    assert [e.kind for e in errors] == ["prompt_over_window"] and errors[0].record["prompt_tokens"] == 16920


def test_second_rejection_after_the_clamp_is_terminal():
    tr = WindowedTransport(window=8704, prompt_tokens=8000, reject_always=True)
    eng = _engine(LENS_ROUTE, transport=tr)
    results, errors = eng.generate_items(er.render_chat_items(PROBLEMS)[:1], GEN_CFG)
    assert results == {} and [c[2]["max_tokens"] for c in tr.chat_calls] == [768, 704]   # exactly two
    [err] = errors
    assert err.kind == "prompt_over_window" and "rejected again after the single clamped retry" in err.message
    assert "768 -> 704" in err.record["window_reason"]
    assert (eng.stats.n_attempts, eng.stats.n_rejected, eng.stats.n_clamped,
            eng.stats.n_prompt_over_window) == (2, 2, 0, 1)


def test_floor_only_rejection_without_tokenize_is_unresolvable_never_a_guess():
    for kw in ({"tokenize_status": 501}, {"tokenize_status": 404}, {"tokenize_count": None}):
        tr = WindowedTransport(window=8704, prompt_tokens=8100, floor_only=True, **kw)
        eng = _engine(LENS_ROUTE, transport=tr)
        results, errors = eng.generate_items(er.render_chat_items(PROBLEMS)[:1], GEN_CFG)
        assert results == {} and [c[0] for c in tr.calls] == [CHAT_URL, TOKENIZE_URL], kw
        [err] = errors
        assert err.kind == "prompt_over_window"
        assert err.record["window_reason"] == sw.FLOOR_ONLY_UNRESOLVABLE
        assert err.record["prompt_tokens"] is None and err.record["prompt_floor"] == 8704 - 768 + 1
        assert eng.stats.n_floor_only_unresolvable == 1 and eng.stats.n_tokenize_resolved == 0
    # a resolved floor-only length that leaves too little room is the typed error too
    tr = WindowedTransport(window=8704, prompt_tokens=8690, floor_only=True)
    _, errors = _engine(LENS_ROUTE, transport=tr).generate_items(er.render_chat_items(PROBLEMS)[:1], GEN_CFG)
    assert [c[0] for c in tr.calls] == [CHAT_URL, TOKENIZE_URL]
    assert errors[0].record["prompt_tokens"] == 8690 and errors[0].record["prompt_tokens_source"] == "tokenize_endpoint"


def test_server_window_governs_the_clamp_and_a_threaded_disagreement_is_recorded(capsys):
    tr = WindowedTransport(window=8704, prompt_tokens=8000)
    eng = _engine(LENS_ROUTE, transport=tr, max_model_len=16384)   # the caller's stale value
    results, errors = eng.generate_items(er.render_chat_items(PROBLEMS), GEN_CFG)
    assert errors == [] and len(results) == 2
    assert sorted(c[2]["max_tokens"] for c in tr.chat_calls) == [704, 704, 768, 768]   # the SERVER's room
    assert eng.stats.window_mismatches == [{"threaded": 16384, "server": 8704}]     # recorded once
    assert capsys.readouterr().out.count("WINDOW MISMATCH") == 1
    # no threaded window at all: the server's stands alone, nothing to disagree with
    tr = WindowedTransport(window=8704, prompt_tokens=8000)
    eng = _engine(LENS_ROUTE, transport=tr)
    assert eng.max_model_len is None
    assert eng.generate_items(er.render_chat_items(PROBLEMS)[:1], GEN_CFG)[1] == []
    assert eng.stats.window_mismatches == [] and eng.stats.n_clamped == 1


# --------------------------------------------------------------------------- #
# the runner: sidecar resume, atomic samples, padding, :open extraction        #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def offline_runner(monkeypatch):
    """Base-lane seams: no evalplus import, identity sanitize."""
    monkeypatch.setattr(er, "_require_pinned_evalplus", lambda: None)
    monkeypatch.setattr(er, "sanitize_solution", lambda text, ep: f"# {ep}\n{text.strip()}\n")
    return None


def test_generate_samples_served_resumes_from_sidecar_and_writes_atomically(tmp_path, offline_runner):
    cell = tmp_path / "cells" / "k"
    tr = FakeTransport(fail={"HumanEval/1/0"})
    eng = _engine(LENS_ROUTE, transport=tr)
    with pytest.raises(er.EvalPlusSeamError, match="1 of 2 items FAILED"):
        er.generate_samples_served(engine=eng, dataset="humaneval", cell_dir=cell,
                                   problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    tdir = er.served_task_dir(cell, "humaneval")
    assert not (tdir / er.SAMPLES_FILENAME).exists()          # never a partial samples file
    side = er.read_generations(tdir / er.GENERATIONS_FILENAME)
    assert set(side) == {"HumanEval/0/0"}
    assert side["HumanEval/0/0"]["intervention_hash"] == LENS_ROUTE.intervention_hash
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "incomplete" and meta["n_requests"] == 2 and meta["n_rejected"] == 1
    assert meta["n_errors"] == 1 and [e["kind"] for e in meta["errors"]] == ["http_error"]
    assert meta["errors"][0]["item_id"] == "HumanEval/1" and meta["errors"][0]["record"] is None
    # second invocation: only the gap is requested
    tr2 = FakeTransport()
    eng2 = _engine(LENS_ROUTE, transport=tr2)
    summary = er.generate_samples_served(engine=eng2, dataset="humaneval", cell_dir=cell,
                                         problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    assert [c[1][C.HEADER_SAMPLE] for c in tr2.calls] == ["HumanEval/1/0"]
    samples = er.read_samples_jsonl(tdir / er.SAMPLES_FILENAME, expected_task_ids=TWO_IDS)
    assert [s["task_id"] for s in samples] == list(PROBLEMS)
    assert samples[0]["solution"] == "# has_close_elements\nreturn True\n"
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "complete" and meta["n_requests"] == 3 and meta["n_rejected"] == 1
    assert meta["padded_item_ids"] == [] and meta["item_ids"] == list(PROBLEMS)
    assert meta["thinking"] == {"mode": "off", "enable_thinking": False, "render": None,
                                "reasoning_parser": None}
    assert meta["route"]["declared_response_prefill"] is True
    assert meta["task"] == "evalplus/humaneval_plus" and meta["finish_reasons"] == {"stop": 2}
    assert summary["n"] == 2 and summary["n_padded"] == 0 and summary["n_requests"] == 3
    # a third call is a no-op (nothing to request)
    tr3 = FakeTransport()
    er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=tr3), dataset="humaneval",
                               cell_dir=cell, problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    assert tr3.calls == []


def test_smoke_limit_pads_the_pinned_set_and_names_the_padding(tmp_path, offline_runner):
    cell = tmp_path / "cells" / "k"
    eng = _engine(PROMPTED_ROUTE, system=SI_TEXT, thinking="on", transport=FakeTransport())
    summary = er.generate_samples_served(engine=eng, dataset="humaneval", cell_dir=cell,
                                         problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG, limit=1)
    tdir = er.served_task_dir(cell, "humaneval")
    samples = er.read_samples_jsonl(tdir / er.SAMPLES_FILENAME, expected_task_ids=TWO_IDS)
    assert [(s["task_id"], s["solution"] == "") for s in samples] == [("HumanEval/0", False), ("HumanEval/1", True)]
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["padded_item_ids"] == ["HumanEval/1"] and meta["limit"] == 1
    assert meta["thinking"]["render"] == C.THINKING_RENDER_FLAG_ONLY   # prefilled under ON
    assert meta["route"]["declared_response_prefill"] is False        # passthrough route
    assert summary["n_padded"] == 1


def test_open_variant_extraction_and_meta(tmp_path, offline_runner):
    assert er.extract_first_python_block("text\n```python\nx = 1\n```\nmore") == ("x = 1", "fenced_python")
    assert er.extract_first_python_block("```\nraw\n```") == ("raw", "fenced")
    assert er.extract_first_python_block("```js\nj\n```\n```py\np\n```") == ("p", "fenced_python")
    assert er.extract_first_python_block("no fence at all") == ("no fence at all", "unfenced")
    assert er.extract_first_python_block("") == ("", "unfenced")
    cell = tmp_path / "cells" / "k"
    tr = FakeTransport(text="Sure.\n```python\ndef f():\n    return 1\n```\nDone.", reasoning="hmm")
    eng = _engine(LENS_ROUTE, thinking="on", transport=tr, task=C.evalplus_task("humaneval", open_variant=True))
    er.generate_samples_served(engine=eng, dataset="humaneval", cell_dir=cell, open_variant=True,
                               problems=PROBLEMS, expected=TWO_IDS)
    tdir = er.served_task_dir(cell, "humaneval", open_variant=True)
    assert tdir.name == "humaneval_open"
    body = tr.calls[0][2]
    assert body["max_tokens"] == 4096 and "stop" not in body and RESPONSE_PREFILL_KEY not in body
    samples = er.read_samples_jsonl(tdir / er.SAMPLES_FILENAME)
    assert samples[0]["solution"] == "# has_close_elements\ndef f():\n    return 1\n"
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["task"] == "evalplus/humaneval_plus:open"
    assert meta["thinking"]["render"] == C.THINKING_RENDER_NATIVE
    assert meta["open_extraction"] == {"fenced_python": 2}
    assert meta["route"]["declared_response_prefill"] is False
    side = er.read_generations(tdir / er.GENERATIONS_FILENAME)
    assert side["HumanEval/0/0"]["reasoning_chars"] == 3
    assert side["HumanEval/0/0"]["reasoning_content"] == "hmm"
    assert side["HumanEval/0/0"]["content_empty_with_reasoning"] is False
    assert meta["n_content_empty_with_reasoning"] == 0 and meta["content_empty_with_reasoning_item_ids"] == []


# --------------------------------------------------------------------------- #
# both channels recorded; content empty + reasoning = a FLAGGED row; neither  #
# channel under a composed parser = the typed empty_response (Evals 21:33)    #
# --------------------------------------------------------------------------- #


def test_both_channels_are_recorded_and_an_empty_content_with_reasoning_is_a_flagged_row(
        tmp_path, offline_runner):
    """A response whose content is empty while reasoning_content is
    non-empty is the item producing NO answer within max_tokens — that IS
    the measurement, so it is a row (scored as-is: an empty solution) that
    is FLAGGED in the sidecar, counted and named in gen_meta, never a
    silent empty answer. Content null and content "" are the same case."""
    cell = tmp_path / "cells" / "k"
    tr = FakeTransport(text="", reasoning="I ran out of budget while thinking")
    eng = _engine(LENS_ROUTE, thinking="on", transport=tr, reasoning_parser="gemma4",
                  task=C.evalplus_task("humaneval", open_variant=True))
    results, errors = eng.generate_items(er.render_chat_items(PROBLEMS, open_variant=True), GEN_CFG)
    assert errors == [] and all(r.content_empty_with_reasoning for r in results.values())
    assert all(r.text == "" and r.reasoning == "I ran out of budget while thinking" for r in results.values())
    assert eng.stats.n_content_empty_with_reasoning == 2 and eng.stats.n_empty_response == 0
    # through the runner: the row is persisted with both channels + the flag
    summary = er.generate_samples_served(engine=eng, dataset="humaneval", cell_dir=cell, open_variant=True,
                                         problems=PROBLEMS, expected=TWO_IDS)
    tdir = er.served_task_dir(cell, "humaneval", open_variant=True)
    side = er.read_generations(tdir / er.GENERATIONS_FILENAME)
    for rec in side.values():
        assert rec["text"] == "" and rec["reasoning_content"] == "I ran out of budget while thinking"
        assert rec["content_empty_with_reasoning"] is True and rec["reasoning_chars"] == 34
    samples = er.read_samples_jsonl(tdir / er.SAMPLES_FILENAME)      # scored as-is: the set is complete
    assert [s["task_id"] for s in samples] == list(PROBLEMS)
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "complete"
    assert meta["n_content_empty_with_reasoning"] == 2 and meta["n_empty_response"] == 0
    assert meta["content_empty_with_reasoning_item_ids"] == list(PROBLEMS)
    assert meta["thinking"]["reasoning_parser"] == "gemma4"
    assert summary["n_content_empty_with_reasoning"] == 2
    # null content is the same flagged case
    tr_null = FakeTransport(text=None, reasoning="thought")
    res, errs = _engine(LENS_ROUTE, thinking="on", transport=tr_null).generate_items(
        er.render_chat_items(PROBLEMS), GEN_CFG)
    assert errs == [] and all(r.content_empty_with_reasoning and r.text == "" for r in res.values())


def test_content_without_reasoning_records_a_null_reasoning_channel(tmp_path, offline_runner):
    """Thinking-OFF (no parser, no thought): ``reasoning_content`` is
    recorded as None — absent, not empty — and nothing is flagged."""
    cell = tmp_path / "cells" / "k"
    eng = _engine(LENS_ROUTE, transport=FakeTransport())
    er.generate_samples_served(engine=eng, dataset="humaneval", cell_dir=cell,
                               problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    tdir = er.served_task_dir(cell, "humaneval")
    for rec in er.read_generations(tdir / er.GENERATIONS_FILENAME).values():
        assert rec["reasoning_content"] is None and rec["reasoning_chars"] is None
        assert rec["content_empty_with_reasoning"] is False and rec["text"] == "\n    return True\n"
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["n_content_empty_with_reasoning"] == 0 and meta["thinking"]["reasoning_parser"] is None


@pytest.mark.parametrize("content", ["", None])
def test_neither_channel_under_a_composed_parser_is_the_typed_empty_response_never_a_row(
        tmp_path, offline_runner, content):
    """With a parser composed, a 200 carrying neither content nor
    reasoning cannot be told apart from text the parser lost: the typed
    ``empty_response`` error, filed per item with the response facts,
    never a sidecar row. WITHOUT a composed parser the identical response
    is an ordinary empty (wrong) answer and becomes a row."""
    cell = tmp_path / "cells" / "k"
    tr = FakeTransport(text=content, reasoning=None)
    eng = _engine(LENS_ROUTE, thinking="on", transport=tr, reasoning_parser="gemma4")
    items = er.render_chat_items(PROBLEMS)
    results, errors = eng.generate_items(items, GEN_CFG)
    assert results == {} and [e.kind for e in errors] == [pe.ERR_EMPTY_RESPONSE] * 2
    assert eng.stats.n_empty_response == 2 and eng.stats.n_content_empty_with_reasoning == 0
    rec = errors[0].record
    assert rec == {"error": "empty_response", "reasoning_parser": "gemma4", "finish_reason": "stop",
                   "completion_tokens": 30, "content_null": content is None}
    assert "neither content nor reasoning_content" in errors[0].message and "'gemma4'" in errors[0].message
    with pytest.raises(er.EvalPlusSeamError, match="2 of 2 items FAILED"):
        er.generate_samples_served(engine=eng, dataset="humaneval", cell_dir=cell,
                                   problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    tdir = er.served_task_dir(cell, "humaneval")
    assert not (tdir / er.GENERATIONS_FILENAME).exists() or er.read_generations(tdir / er.GENERATIONS_FILENAME) == {}
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "incomplete" and meta["n_empty_response"] == 2
    assert {e["kind"] for e in meta["errors"]} == {"empty_response"}
    assert meta["errors"][0]["record"]["reasoning_parser"] == "gemma4"
    # no parser composed: the same response is a plain (empty) row
    plain = _engine(LENS_ROUTE, transport=FakeTransport(text=content, reasoning=None))
    res, errs = plain.generate_items(items, GEN_CFG)
    assert errs == [] and all(r.text == "" and not r.content_empty_with_reasoning for r in res.values())
    assert plain.stats.n_empty_response == 0


def test_reasoning_parser_knob_is_validated_and_threaded_by_run_served(tmp_path, offline_runner, monkeypatch):
    with pytest.raises(pe.ProxyEngineError, match="reasoning_parser"):
        _engine(LENS_ROUTE, reasoning_parser="")
    with pytest.raises(pe.ProxyEngineError, match="reasoning_parser"):
        _engine(LENS_ROUTE, reasoning_parser=7)
    pins = {d: er.DatasetPin(dataset=d, name=d, version="v", url="u", gz_filename=f"{d}.gz",
                             gz_bytes=1, gz_sha256="0" * 64, gz_md5="0" * 32,
                             override_env="X", task_ids=TWO_IDS) for d in C.EVALPLUS_DATASETS}
    monkeypatch.setattr(er, "dataset_pin", lambda d: pins[d])
    monkeypatch.setattr(er, "load_problems", lambda d="humaneval": dict(PROBLEMS))
    cell = tmp_path / "on" / "cells" / "k"
    er.run_served(base_url="http://127.0.0.1:1", route=LENS_ROUTE, system=None, thinking="on",
                  concurrency=2, cell_dir=cell, transport=FakeTransport(reasoning="t"),
                  reasoning_parser="gemma4")
    for dataset, open_variant in (("humaneval", False), ("humaneval", True), ("mbpp", True)):
        meta = json.loads((er.served_task_dir(cell, dataset, open_variant) / er.GEN_META_FILENAME).read_text())
        assert meta["thinking"]["reasoning_parser"] == "gemma4"


def test_run_served_task_matrix_per_thinking_mode(tmp_path, offline_runner, monkeypatch):
    pins = {d: er.DatasetPin(dataset=d, name=d, version="v", url="u", gz_filename=f"{d}.gz",
                             gz_bytes=1, gz_sha256="0" * 64, gz_md5="0" * 32,
                             override_env="X", task_ids=TWO_IDS) for d in C.EVALPLUS_DATASETS}
    monkeypatch.setattr(er, "dataset_pin", lambda d: pins[d])
    monkeypatch.setattr(er, "load_problems", lambda d="humaneval": dict(PROBLEMS))
    on = er.run_served(base_url="http://127.0.0.1:1", route=LENS_ROUTE, system=None, thinking="on",
                       concurrency=2, cell_dir=tmp_path / "on" / "cells" / "k", transport=FakeTransport())
    assert list(on) == ["evalplus/humaneval_plus", "evalplus/humaneval_plus:open",
                        "evalplus/mbpp_plus", "evalplus/mbpp_plus:open"]
    off = er.run_served(base_url="http://127.0.0.1:1", route=LENS_ROUTE, system=None, thinking="off",
                        concurrency=2, cell_dir=tmp_path / "off" / "cells" / "k", transport=FakeTransport())
    assert list(off) == ["evalplus/humaneval_plus", "evalplus/mbpp_plus"]
    assert set(off) <= set(C.TINYCAP_TASKS) and set(on) <= set(C.TINYCAP_TASKS)


# --------------------------------------------------------------------------- #
# execution results: validation + rows                                         #
# --------------------------------------------------------------------------- #


def _results(ids=("HumanEval/0", "HumanEval/1"), plus=("pass", "fail"), base=("pass", "fail"),
             hash_="1" * 32):
    return {"date": "2026-09-03 00:00", "hash": hash_,
            "eval": {t: [{"task_id": t, "solution": "s", "base_status": b, "plus_status": p,
                          "base_fail_tests": [], "plus_fail_tests": []}]
                     for t, b, p in zip(ids, base, plus)}}


def test_validate_eval_results_accepts_the_shape_and_refuses_drift():
    fixture = EP.eval_results()
    er.validate_eval_results(fixture, expected_task_ids=TWO_IDS)
    er.validate_eval_results(_results(), expected_task_ids=TWO_IDS, expected_hash="1" * 32)
    cases = [
        ([1, 2], "not a JSON object"),
        ({"eval": {}}, "top-level keys"),
        ({**_results(), "extra": 1}, "top-level keys"),
        (_results(hash_="2" * 32), "dataset hash"),
        ({**_results(), "eval": []}, "not an object"),
        (_results(ids=("HumanEval/0",), plus=("pass",), base=("pass",)), "task-ID set"),
        ({**_results(), "eval": {**_results()["eval"], "HumanEval/2": _results()["eval"]["HumanEval/0"]}}, "task-ID set"),
        ({**_results(), "eval": {k: v * 2 for k, v in _results()["eval"].items()}}, "exactly 1 sample"),
        ({**_results(), "eval": {k: [{kk: vv for kk, vv in v[0].items() if kk != "plus_status"}]
                                 for k, v in _results()["eval"].items()}}, "missing"),
        ({**_results(), "eval": {k: [{**v[0], "foo": 1}] for k, v in _results()["eval"].items()}}, "unexpected"),
        ({**_results(), "eval": {k: [{**v[0], "task_id": "HumanEval/9"}] for k, v in _results()["eval"].items()}}, "echoes"),
        ({**_results(), "eval": {k: [{**v[0], "plus_status": None}] for k, v in _results()["eval"].items()}}, "plus_status"),
        ({**_results(), "eval": {k: [{**v[0], "base_status": "ok"}] for k, v in _results()["eval"].items()}}, "base_status"),
        ({**_results(), "eval": {k: [{**v[0], "base_fail_tests": "x"}] for k, v in _results()["eval"].items()}}, "not a list"),
    ]
    for bad, match in cases:
        with pytest.raises(er.EvalPlusSeamError, match=match):
            er.validate_eval_results(bad, expected_task_ids=TWO_IDS, expected_hash="1" * 32)


def test_eval_results_to_rows_served_naming_keeps_the_hf_default_frozen():
    fixture = EP.eval_results()
    default = er.eval_results_to_rows(fixture)
    assert {r["task"] for r in default} == {"humaneval_plus"}
    # the HF default's task / domain names stay frozen; its DEFINITION moved at PW10-8 and since PW13 the
    # row's task_version says so (the bare pin is the retained plus-only rows' literal, never written anew)
    assert {r["domain"] for r in default} == {"code"}
    assert {r["task_version"] for r in default} == {"evalplus-0.3.1+base_and_plus"}
    served = er.eval_results_to_rows(fixture, TWO_IDS, task=C.evalplus_task("mbpp", open_variant=True),
                                     domain=C.DOMAIN_EVALPLUS_SERVED, task_version=C.EVALPLUS_TASK_VERSION)
    assert {r["task"] for r in served} == {"evalplus/mbpp_plus:open"}
    assert {r["domain"] for r in served} == {"evalplus_served"}
    assert {r["task_version"] for r in served} == {"evalplus-0.3.1+base_and_plus"} == {
        C.evalplus_row_task_version(er.PLUS_METRIC_DEF)}   # P31: same framework pin; the definition rides in it (PW13)
    assert {r["metric"] for r in served} == {"pass@1_base", "pass@1_plus", "acc"}
    assert C.METRIC_PASS1_BASE in {r["metric"] for r in served}


#: A producer tree sha nobody declared, so a synthetic store built with it can
#: date itself for NO contract (E55 #1: ``STORE_ERA_UNKNOWN``) and every check
#: refuses it exactly as it refuses a current store: every field a check
#: requires must be there. ``contracts.KNOWN_PRODUCER_TREES`` holds the six
#: trees this campaign ran, each with the SET of contracts it predates.
UNDECLARED_TREE_SHA = "a1" * 20
#: A tree that predates every contract this repo declares.
PRE_EVERYTHING_TREE_SHA = "618f9ddeedc26d90fe93ed78ea96688a171d9d7e"
#: The Qwen/Llama wave's tree: it DESCENDS from the sampling contract's landing
#: (so it must carry its sampling blocks) and PREDATES the samples-digest one.
POST_SAMPLING_TREE_SHA = "c7e1796ac9a36732822e8f56d02b7c7af43f9dcf"


def _stated_compared() -> dict:
    """The three provenance blocks the served driver writes on a store whose
    fit was chat-rendered and compared: the top-level class, the immutable
    fit witness, and the fit_provenance block that mirrors it."""
    from sisteer.eval import sweep as _sweep
    witness = {**{f: None for f in _sweep.FIT_WITNESS_FIELDS},
               "model_id": "org/model", "revision": "rev", "template_hash": "cc" * 8,
               "status": "verified", "source": "train_budget_meta",
               "fit_kind": _sweep.FIT_KIND_SI_PAIR, "positions_at_fit": "response",
               "object_sha": "aa" * 8}
    return {_sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY: "compared",
            _sweep.MANIFEST_KEY_FIT_WITNESS: dict(witness),
            _sweep.MANIFEST_KEY_FIT_PROVENANCE: {
                **witness, "served_model_id": "org/model", "served_revision": "rev",
                "served_template_hash": "cc" * 8, "transfer": False, "template_check": "compared",
                "template_check_reason": _sweep.TEMPLATE_CHECK_REASON["compared"]}}


def _cell_dir(tmp_path, *, engine="vllm", meta_padded=(), flagged=None, driver_limits=None,
              samples_per_item=1, results=None, flagged_keys=None, sampling=None, seeds=None,
              producer=UNDECLARED_TREE_SHA, binding="bound", manifest_sampling=None):
    """A served EvalPlus task dir the way a real sandbox leaves it. E54 #1:
    that INCLUDES the ``samples.jsonl`` the results were produced from and an
    ``execution.json`` naming its sha256 — a store that holds results and
    cannot name their samples is the damaged state, not the ordinary one, so
    the ordinary fixture binds. ``binding``: ``"bound"`` (the digest matches
    the samples), ``"unbound"`` (a pre-digest execution), ``"none"`` (no
    execution at all) or ``"nosamples"`` (results with no samples beside them).
    ``producer`` stamps the store's tree sha, which is what dates it."""
    import hashlib as _hashlib

    cell = tmp_path / "sweep" / "cells" / "tok__controls__t0__a0__aaaaaaaa__bbbbbbbb"
    cell.mkdir(parents=True)
    rc = {"sweep": "two_si_controls_served", "run_id": "served-controls-tok", "model_id": "org/model",
          "model_revision": "rev", "arm_name": "tok", "kind": "cell", "coeff": 1.0,
          "intervention_hash": "aa" * 16, "engine": engine, "trial_id": 0, "seed": 0, "attempt": 0,
          "harness_config_hash": "bb" * 16, "env_digest": "cc" * 16, "slurm_job_id": None}
    # the identity fields CellStore.write_cell_manifest always records: the
    # merge reads harness_config_hash to say which identity produced a number
    manifest = {"row_constants": rc, "virtual_model": LENS_ROUTE.virtual_model,
                "intervention_hash": rc["intervention_hash"],
                "harness_config_hash": rc["harness_config_hash"],
                # P1-1: a store not on a declared HISTORICAL tree STATES its
                # served==trained template-check class (the reader takes a
                # stated class without dating the store) with its witness and
                # fit block, as the driver writes on every store; a store on a
                # declared historical tree states none and reads "unrecorded"
                # through the era, so "legacy store" keeps meaning what it means
                **({} if producer in C.KNOWN_PRODUCER_TREES else _stated_compared())}
    if producer is not None:
        manifest[C.MANIFEST_KEY_PRODUCER] = {"tree_sha": producer,
                                             "driver": C.PRODUCER_DRIVER_SERVED}
    if manifest_sampling is not None:      # the RUN's sampling block (E54 #3)
        manifest[C.MANIFEST_KEY_SAMPLING] = dict(manifest_sampling)
    (cell / "manifest.json").write_text(json.dumps(manifest))
    tdir = er.served_task_dir(cell, "humaneval")
    tdir.mkdir(parents=True)
    generated = [t for t in PROBLEMS if t not in meta_padded]
    meta = {"task": "evalplus/humaneval_plus", "status": "complete", "item_ids": generated,
            "padded_item_ids": list(meta_padded), "n_requests": 3, "n_rejected": 1, "wall_clock_s": 12.5,
            "thinking": {"mode": "off", "enable_thinking": False, "render": None},
            "route": {"virtual_model": LENS_ROUTE.virtual_model}}
    if driver_limits is not None:   # a gen_meta written by a limited driver run (R2)
        meta["driver_limits"] = dict(driver_limits)
    if flagged is not None:   # a gen_meta written since the channels are recorded
        meta["content_empty_with_reasoning_item_ids"] = list(flagged)
        meta["n_content_empty_with_reasoning"] = len(flagged)
    if flagged_keys is not None:   # ... and since U8, per generation
        meta["content_empty_with_reasoning_keys"] = list(flagged_keys)
    if samples_per_item != 1:      # a gen_meta written under k samples per problem (U8)
        meta["samples_per_item"] = int(samples_per_item)
    if sampling is not None:       # ... and under the sampling contract (U8 / PW-sampling)
        meta["sampling"] = dict(sampling)
    if seeds is not None:          # the resume sidecar: {generation key: seed}
        (tdir / er.GENERATIONS_FILENAME).write_text("".join(
            json.dumps({"item_id": key.rsplit("/", 1)[0], "sample_idx": int(key.rsplit("/", 1)[1]),
                        "seed": seed, "text": "s", "finish_reason": "stop"}, sort_keys=True) + "\n"
            for key, seed in seeds.items()))
    (tdir / er.GEN_META_FILENAME).write_text(json.dumps(meta))
    (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(_results() if results is None else results))
    # E54 #1: the samples the results were produced from, and the execution
    # that names them — the pair every real store holds
    sample_bytes = b"".join(json.dumps({"task_id": t, "solution": "pass"}).encode() + b"\n"
                            for t in PROBLEMS for _ in range(samples_per_item))
    if binding != "nosamples":
        (tdir / er.SAMPLES_FILENAME).write_bytes(sample_bytes)
    execution = {"wall_clock_s": 7.5}
    if binding == "bound":
        execution[er.SAMPLES_DIGEST_FIELD] = _hashlib.sha256(sample_bytes).hexdigest()
    if binding != "none":
        (tdir / er.EXECUTION_FILENAME).write_text(json.dumps(execution))
    return cell, tdir


def test_convert_eval_results_to_store_rows_marker_summary_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, tdir = _cell_dir(tmp_path)
    summary = er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert {r["task"] for r in rows} == {"evalplus/humaneval_plus"}
    assert {r["engine"] for r in rows} == {"vllm"} and {r["domain"] for r in rows} == {"evalplus_served"}
    assert {r["tier"] for r in rows} == {1} and {r["task_version"] for r in rows} == {"evalplus-0.3.1+base_and_plus"}
    by = {(r["item_id"], r["metric"]): r["value"] for r in rows}
    assert by[("HumanEval/0", "acc")] == 1.0 and by[("HumanEval/1", "acc")] == 0.0
    assert summary["pass_at_1_plus"] == 0.5 == summary["acc"] and summary["n"] == 2
    assert summary["wall_clock_s"] == 20.0 and summary["route"] == LENS_ROUTE.virtual_model
    assert set(summary) <= set(C.SUMMARY_PAYLOAD_FIELDS)
    from sisteer.eval.cell_controls import _assert_counts_only
    _assert_counts_only({"evalplus/humaneval_plus": summary})
    marker = json.loads(er.done_marker_path(cell, "evalplus/humaneval_plus").read_text())
    assert er.done_marker_path(cell, "evalplus/humaneval_plus").name == "evalplus__humaneval_plus.json"
    # E55 #2: the marker repeats the summary's samples binding as an OPTIONAL
    # key — the required set stays exact, so markers written before it (every
    # one on the volume) stay valid
    assert set(marker) == set(C.DONE_MARKER_FIELDS) | {er.SAMPLES_DIGEST_FIELD}
    assert marker[er.SAMPLES_DIGEST_FIELD] == summary[er.SAMPLES_DIGEST_FIELD]
    assert marker["n_items"] == 2 and marker["n_rows"] == 6
    assert json.loads((tdir / er.SUMMARY_FILENAME).read_text()) == summary
    # the shared per-task summary the controls_summary merge reads (R3), and no partial marker
    assert completion.read_task_summary(cell, "evalplus/humaneval_plus") == summary
    assert not completion.partial_marker_path(cell, "evalplus/humaneval_plus").exists()
    assert completion.complete_task_summaries(cell) == {"evalplus/humaneval_plus": summary}
    # idempotent: a second conversion appends nothing
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    assert len((cell / "rows.jsonl").read_text().splitlines()) == 6
    # a gen_meta that predates the channel record stamps None (unknown), never a silent False
    assert {json.loads(r["extra_json"])["content_empty_with_reasoning"] for r in rows} == {None}
    assert json.loads(by_row(rows, "HumanEval/0", "acc")["extra_json"])["source_metric"] == "pass@1_plus"


def by_row(rows, item_id, metric):
    [r] = [r for r in rows if r["item_id"] == item_id and r["metric"] == metric]
    return r


def test_convert_stamps_the_flagged_row_fact_into_every_per_item_row(tmp_path, monkeypatch):
    """The sidecar flag reaches the STORE: every per-item row's extra_json
    says whether that item produced no answer with reasoning present, so
    a downstream reader of rows.jsonl sees it without the sidecar."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, _ = _cell_dir(tmp_path, flagged=["HumanEval/1"])
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    flags = {(r["item_id"], r["metric"]): json.loads(r["extra_json"])["content_empty_with_reasoning"]
             for r in rows}
    assert flags == {("HumanEval/0", "pass@1_base"): False, ("HumanEval/0", "pass@1_plus"): False,
                     ("HumanEval/0", "acc"): False, ("HumanEval/1", "pass@1_base"): True,
                     ("HumanEval/1", "pass@1_plus"): True, ("HumanEval/1", "acc"): True}
    assert json.loads(by_row(rows, "HumanEval/1", "acc")["extra_json"])["source_metric"] == "pass@1_plus"


def test_convert_refuses_a_store_holding_rows_under_another_definition(tmp_path, monkeypatch):
    """PW13 (E21 M2): the resume deduped existing rows by ``(task, item_id,
    metric)`` alone — a store where an OLD converter appended ``plus_only``
    rows and died before the summary and the done marker had its rows
    SKIPPED and a fresh ``base_and_plus`` summary written unconditionally:
    ``rows.jsonl`` said one thing, ``summary.json`` another, both labelled
    current. The existing rows' definition (their identity-bearing
    ``task_version``, and ``extra_json.metric_def`` where a row names one) is
    now validated BEFORE any is skipped: legacy, mixed and contradictory rows
    REFUSE, whatever markers the store holds, naming the counts and the
    remedy (off-volume re-derivation into a new identity); rows under this
    converter's own definition still resume and dedupe."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, tdir = _cell_dir(tmp_path)
    current = C.evalplus_row_task_version(er.PLUS_METRIC_DEF)
    assert current == "evalplus-0.3.1+base_and_plus"
    assert C.evalplus_row_metric_def(C.EVALPLUS_TASK_VERSION) == C.EVALPLUS_PLUS_DEF_PLUS_ONLY   # the retained rows' literal
    rc = json.loads((cell / "manifest.json").read_text())["row_constants"]
    rows_path = cell / "rows.jsonl"

    def row(item, metric, value, *, version, extra=None):
        return {**rc, "tier": 1, "task": EVALPLUS_TASK, "task_version": version, "domain": C.DOMAIN_EVALPLUS_SERVED,
                "item_id": item, "position_order": None, "metric": metric, "value": value, "extra_json": extra,
                "created_at": 1.0, "wall_clock_s": None}

    def seed(rows):
        rows_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        return rows_path.read_bytes()

    def untouched(before):
        assert rows_path.read_bytes() == before
        assert not (tdir / er.SUMMARY_FILENAME).exists()
        assert not er.done_marker_path(cell, EVALPLUS_TASK).exists()
        assert completion.read_task_summary(cell, EVALPLUS_TASK) is None

    # the OLD converter (plus_only: the bare pin, no metric_def) appended HumanEval/0 and died before summary + marker
    legacy = [row("HumanEval/0", "pass@1_base", 1.0, version=C.EVALPLUS_TASK_VERSION),
              row("HumanEval/0", "pass@1_plus", 1.0, version=C.EVALPLUS_TASK_VERSION),
              row("HumanEval/0", "acc", 1.0, version=C.EVALPLUS_TASK_VERSION,
                  extra=json.dumps({"source_metric": "pass@1_plus"}))]
    before = seed(legacy)
    with pytest.raises(er.EvalPlusSeamError, match="another pass@1_plus definition") as ei:
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    msg = str(ei.value)
    assert "3 of 3 existing" in msg and f"({C.EVALPLUS_PLUS_DEF_PLUS_ONLY})" in msg and current in msg
    assert "tinycap_rederive_evalplus.py" in msg and "never by re-converting" in msg
    untouched(before)
    # MIXED — current-definition rows beside the legacy ones — refuses too, naming the split
    mixed = legacy + [row("HumanEval/1", m, 0.0, version=current, extra=e) for m, e in (
        ("pass@1_base", None), ("pass@1_plus", json.dumps({"metric_def": "base_and_plus"})),
        ("acc", json.dumps({"metric_def": "base_and_plus", "source_metric": "pass@1_plus"})))]
    before = seed(mixed)
    with pytest.raises(er.EvalPlusSeamError, match="3 of 6 existing"):
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    untouched(before)
    # a PW12-era row — base_and_plus NAMED in extra_json under the bare pin — is a label mismatch as well
    before = seed([row("HumanEval/0", "pass@1_plus", 1.0, version=C.EVALPLUS_TASK_VERSION,
                       extra=json.dumps({"metric_def": "base_and_plus"}))])
    with pytest.raises(er.EvalPlusSeamError, match="1 of 1 existing") as ei:
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    assert "metric_def='base_and_plus'" in str(ei.value)
    untouched(before)
    # ...and a current-version row whose extra_json names ANOTHER definition (contradictory) refuses
    before = seed([row("HumanEval/0", "acc", 1.0, version=current,
                       extra=json.dumps({"metric_def": "plus_only", "source_metric": "pass@1_plus"}))])
    with pytest.raises(er.EvalPlusSeamError, match="1 of 1 existing"):
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    untouched(before)
    # a DONE marker does not license it either: a completed legacy store is never re-converted in place
    before = seed(legacy)
    old_summary = {"acc": 1.0, "pass_at_1_base": 1.0, "pass_at_1_plus": 1.0, "n": 2, "n_requests": 3, "n_rejected": 1,
                   "samples_per_item": 1, "framework": C.EVALPLUS_TASK_VERSION, "engine": "vllm",
                   "route": LENS_ROUTE.virtual_model, "thinking": "off", "wall_clock_s": 20.0}
    completion.finish_task(cell, task=EVALPLUS_TASK, n_items=2, n_expected_full=2, limits={}, n_rows=6, n_requests=3,
                           n_rejected=1, wall_clock_s=20.0, summary=old_summary)
    with pytest.raises(er.EvalPlusSeamError, match="another pass@1_plus definition"):
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    assert rows_path.read_bytes() == before and completion.read_task_summary(cell, EVALPLUS_TASK) == old_summary
    assert er.done_marker_path(cell, EVALPLUS_TASK).exists() and not (tdir / er.SUMMARY_FILENAME).exists()
    # the remedy the refusal names is a pure read of the same store, into a new identity
    derived = er.rederive_evalplus(cell, dataset="humaneval")
    assert {r["task_version"] for r in derived["rows"]} == {current}
    assert derived["retained_metric_def"] == C.EVALPLUS_PLUS_DEF_PLUS_ONLY and rows_path.read_bytes() == before
    # the LEGITIMATE resume — rows under THIS converter's definition and no marker (a crash after the rows) —
    # still converts, dedupes and finishes: the idempotence the guard must not break
    shutil.rmtree(cell / C.DONE_DIR)
    shutil.rmtree(cell / C.SUMMARY_DIR)
    rows_path.unlink()
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    n_rows = len(rows_path.read_text().splitlines())
    shutil.rmtree(cell / C.DONE_DIR)
    shutil.rmtree(cell / C.SUMMARY_DIR)
    (tdir / er.SUMMARY_FILENAME).unlink()
    summary = er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == n_rows == 6 and {r["task_version"] for r in rows} == {current}
    assert summary[C.EVALPLUS_METRIC_DEF_FIELD] == er.PLUS_METRIC_DEF
    assert er.done_marker_path(cell, EVALPLUS_TASK).exists()


def test_convert_refuses_hf_store_and_never_marks_a_padded_run(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, _ = _cell_dir(tmp_path, engine="hf")
    with pytest.raises(er.EvalPlusSeamError, match="vllm rows only"):
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    cell, _ = _cell_dir(tmp_path / "p", meta_padded=("HumanEval/1",))
    summary = er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    assert {r["item_id"] for r in rows} == {"HumanEval/0"} and summary["n"] == 1
    assert not er.done_marker_path(cell, "evalplus/humaneval_plus").exists()
    # R2: a padded run is PARTIAL — the marker names the shortfall (no driver limit recorded here)
    partial = json.loads(completion.partial_marker_path(cell, "evalplus/humaneval_plus").read_text())
    assert set(partial) == set(C.PARTIAL_MARKER_FIELDS) | {er.SAMPLES_DIGEST_FIELD}
    assert (partial["n_items"], partial["n_expected_full"], partial["limits"]) == (1, 2, {})
    # a limited DRIVER run over the full set is partial too, naming the limits
    cell, _ = _cell_dir(tmp_path / "q", driver_limits={"inspect_limit": 5})
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    assert not er.done_marker_path(cell, "evalplus/humaneval_plus").exists()
    partial = json.loads(completion.partial_marker_path(cell, "evalplus/humaneval_plus").read_text())
    assert partial["limits"] == {"inspect_limit": 5} and partial["n_items"] == 2 == partial["n_expected_full"]


def test_run_served_records_the_driver_limits_in_gen_meta(tmp_path, offline_runner, monkeypatch):
    """R2: the driver's active limits ride into ``gen_meta.json`` (plus the
    runner's own ``limit`` as one of them) so conversion — which happens
    later, in the orchestrator — ends the task PARTIAL rather than done."""
    pins = {d: er.DatasetPin(dataset=d, name=d, version="v", url="u", gz_filename=f"{d}.gz",
                             gz_bytes=1, gz_sha256="0" * 64, gz_md5="0" * 32,
                             override_env="X", task_ids=TWO_IDS) for d in C.EVALPLUS_DATASETS}
    monkeypatch.setattr(er, "dataset_pin", lambda d: pins[d])
    monkeypatch.setattr(er, "load_problems", lambda d="humaneval": dict(PROBLEMS))
    cell = tmp_path / "lim" / "cells" / "k"
    er.run_served(base_url="http://127.0.0.1:1", route=LENS_ROUTE, system=None, thinking="off",
                  concurrency=2, cell_dir=cell, datasets=("humaneval",), transport=FakeTransport(),
                  limit=1, limits={"inspect_limit": 5})
    meta = json.loads((er.served_task_dir(cell, "humaneval") / er.GEN_META_FILENAME).read_text())
    assert meta["driver_limits"] == {"inspect_limit": 5, "limit": 1} and meta["limit"] == 1
    assert meta["seed_rule"] == er.SEED_RULE
    cell = tmp_path / "full" / "cells" / "k"
    er.run_served(base_url="http://127.0.0.1:1", route=LENS_ROUTE, system=None, thinking="off",
                  concurrency=2, cell_dir=cell, datasets=("humaneval",), transport=FakeTransport())
    meta = json.loads((er.served_task_dir(cell, "humaneval") / er.GEN_META_FILENAME).read_text())
    assert meta["driver_limits"] == {} and meta["limit"] is None


# --------------------------------------------------------------------------- #
# the sandbox orchestrator against a FAKE modal (P18 contract)                 #
# --------------------------------------------------------------------------- #


class _FakeFile(io.BytesIO):
    def __init__(self, fs, path, mode):
        super().__init__(fs.get(path, b"") if "r" in mode else b"")
        self._fs, self._path, self._mode = fs, path, mode

    def __exit__(self, *a):
        if "w" in self._mode:
            self._fs[self._path] = self.getvalue()
        return super().__exit__(*a)


class _FakeProc:
    def __init__(self, rc, out=b"ok", err=b""):
        self._rc = rc
        self.stdout = io.BytesIO(out)
        self.stderr = io.BytesIO(err)

    def wait(self):
        return self._rc


class FakeSandbox:
    """Records create kwargs / exec argv; ``script`` decides what the exec
    'produces' at the results path (a callable of the sandbox fs)."""

    created = []
    script = None
    rc = 0

    def __init__(self, kw):
        self.kw = kw
        self.fs = {}
        self.dirs = []
        self.execs = []
        self.terminated = False
        self.object_id = "sb-fake"

    @classmethod
    def create(cls, *args, **kw):
        sb = cls(dict(kw, _args=args))
        cls.created.append(sb)
        return sb

    def mkdir(self, path, parents=False):
        self.dirs.append((path, parents))

    def open(self, path, mode="r"):
        return _FakeFile(self.fs, path, mode)

    def exec(self, *argv, timeout=None):
        self.execs.append((argv, timeout))
        if FakeSandbox.script is not None:
            FakeSandbox.script(self.fs)
        return _FakeProc(FakeSandbox.rc, err=b"boom" if FakeSandbox.rc else b"")

    def terminate(self, *, wait=False):
        self.terminated = True


def _fake_modal():
    m = types.ModuleType("modal")

    class _Img:
        def __init__(self, tag=None):
            self.tag = tag

        @classmethod
        def debian_slim(cls, python_version=None):
            return cls("debian_slim")

        @classmethod
        def from_registry(cls, tag, **kw):
            return cls(tag)

        def __getattr__(self, name):  # pip_install / env / add_local_* chain
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *a, **k: self

    class _Vol:
        @classmethod
        def from_name(cls, *a, **k):
            return cls()

        def reload(self):
            pass

        def commit(self):
            pass

    class _App:
        def __init__(self, name):
            self.name = name
            self.functions = {}

        def function(self, **kw):
            def deco(fn):
                self.functions[fn.__name__] = kw
                return fn
            return deco

        def local_entrypoint(self):
            return lambda fn: fn

    m.App, m.Image, m.Volume, m.Sandbox = _App, _Img, _Vol, FakeSandbox
    m.Secret = types.SimpleNamespace(from_name=lambda name, **k: ("secret", name))
    m.__version__ = "fake"
    return m


@pytest.fixture()
def sandbox_mod(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", _fake_modal())
    spec = importlib.util.spec_from_file_location(
        "tinycap_sandbox_under_test", REPO / "scripts" / "modal" / "tinycap_sandbox.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    FakeSandbox.created.clear()
    FakeSandbox.script = None
    FakeSandbox.rc = 0
    return mod


@pytest.fixture()
def launcher_mod(monkeypatch):
    """The launcher (``tinycap_served.py``) against the same fake modal: the
    controls_summary merge step lives there (N1 — the orchestrator image
    carries no torch, the launcher's client venv does)."""
    monkeypatch.setitem(sys.modules, "modal", _fake_modal())
    spec = importlib.util.spec_from_file_location(
        "tinycap_served_under_sandbox_test", REPO / "scripts" / "modal" / "tinycap_served.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    """A fake staged archive whose sha256/md5 ARE the pin (the pin is
    patched to the fake bytes; the id set is the two synthetic problems)."""
    archive = b"fake-humaneval-plus-archive\n"
    real = er.dataset_pin("humaneval")
    fake_pin = er.DatasetPin(dataset="humaneval", name=real.name, version=real.version, url=real.url,
                             gz_filename=real.gz_filename, gz_bytes=len(archive),
                             gz_sha256=hashlib.sha256(archive).hexdigest(),
                             gz_md5=hashlib.md5(archive).hexdigest(),
                             override_env=real.override_env, task_ids=TWO_IDS)
    monkeypatch.setattr(er, "dataset_pin", lambda d: fake_pin if d == "humaneval" else real)
    cache = tmp_path / "cache" / "evalplus"
    cache.mkdir(parents=True)
    (cache / real.gz_filename).write_bytes(archive)
    return cache, fake_pin, archive


def _job(tmp_path, *, stage="execute", samples_per_item=1):
    cell, tdir = _cell_dir(tmp_path, samples_per_item=samples_per_item)
    (tdir / er.EVAL_RESULTS_FILENAME).unlink()
    (tdir / er.EXECUTION_FILENAME).unlink()
    er.write_samples_jsonl(tdir / er.SAMPLES_FILENAME,
                           [{"task_id": t, "solution": "def f(): pass"}
                            for t in PROBLEMS for _ in range(samples_per_item)])
    return {"cell_dir": str(cell), "task_dir": str(tdir), "dataset": "humaneval",
            "open_variant": False, "task": "evalplus/humaneval_plus", "stage": stage}


def _execution(sandbox_mod, tdir, *, samples_sha=None, archive_md5=None, **extra):
    """The ``execution.json`` a sandbox leaves beside its results (E50 #4): it
    records the sha256 of the exact ``samples.jsonl`` it evaluated, so a resume
    can tell those results from results of a later regeneration under the same
    task ids. ``samples_sha`` defaults to the file on disk NOW."""
    tdir = Path(tdir)
    doc = {sandbox_mod.SAMPLES_DIGEST_FIELD:
           samples_sha if samples_sha is not None
           else sandbox_mod.samples_sha256(tdir / er.SAMPLES_FILENAME), **extra}
    if archive_md5 is not None:
        doc["archive"] = {"md5": archive_md5}
    (tdir / er.EXECUTION_FILENAME).write_text(json.dumps(doc))
    return doc


def _tinycap_store(tmp_path, monkeypatch):
    """A tinycap cell store the way the served driver leaves it after
    generation: a REAL registry cell (its synthetic object on disk, hashed
    into the provenance), the manifest's ``controls_provenance`` stamp +
    ``cell_id`` + ``row_constants``, and a complete unpadded EvalPlus
    generation awaiting execution. Returns ``(cfg, cell, cell_dir, tdir)``;
    ``cell_dir.parent.parent`` is the out_root the consumer reads."""
    from sisteer.eval.cell_controls import controls_provenance
    from sisteer.eval.defense_cell import by_id, registry_from_cfg

    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    ddir = tmp_path / "tt_cells"
    obj = ddir / C.CELL_PSRKL_SAFETY / "object.pt"
    obj.parent.mkdir(parents=True)
    obj.write_bytes(b"synthetic-object-bytes-for-tests")
    cfg = {"model": "org/model", "model_revision": "rev", "si_ids": ["safety_refusal"],
           "class_a_arms": ["psr_kl_all"], "class_b_arms": [], "defenses_dir": str(ddir),
           "attacks": {"target_max_new_tokens": 64}, "completion_tokens": 64}
    cell = by_id(registry_from_cfg(cfg))[C.CELL_PSRKL_SAFETY]
    job = _job(tmp_path)
    cell_dir, tdir = Path(job["cell_dir"]), Path(job["task_dir"])
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    manifest["cell_id"] = C.CELL_PSRKL_SAFETY
    # PW8 (E16 A2): the geometry pass is HASHED identity; the top-level
    # geometry block copies it (a pass-1 launcher run)
    manifest["resolved_config"] = {C.MANIFEST_KEY_TINYCAP: {C.GEOMETRY_PASS_FIELD: 1}}
    manifest[C.MANIFEST_KEY_GEOMETRY] = {C.GEOMETRY_PASS_FIELD: 1}
    manifest.update(_stated_compared())                     # P1-1: the stated class + witness + block
    manifest[C.MANIFEST_KEY_PROVENANCE] = controls_provenance(
        cell, cfg, intervention_hash="aa" * 16,
        serving_identity={"engine": "vllm", "virtual_model": LENS_ROUTE.virtual_model,
                          "chat_template_hash": "cc" * 8},
        judge_model="claude-sonnet-5")
    (cell_dir / "manifest.json").write_text(json.dumps(manifest))
    return cfg, cell, cell_dir, tdir, job


def test_sandbox_create_kwargs_follow_the_contract(sandbox_mod):
    kw = sandbox_mod.sandbox_create_kwargs()
    for k, v in C.SANDBOX_CREATE_KWARGS_REQUIRED.items():
        assert kw[k] == v
    assert kw["block_network"] is True and kw["cpu"] == 8 and kw["memory"] == 16384 and kw["timeout"] == 3600
    assert not (set(kw) & set(C.SANDBOX_CREATE_KWARGS_FORBIDDEN))
    assert kw["image"].tag == C.SANDBOX_IMAGE_DIGEST and "@sha256:" in kw["image"].tag
    assert kw["workdir"] == C.SANDBOX_WORKDIR
    with pytest.raises(RuntimeError, match="forbidden"):
        sandbox_mod.assert_sandbox_kwargs({**kw, "volumes": {"/vol": object()}})
    with pytest.raises(RuntimeError, match="forbidden"):
        sandbox_mod.assert_sandbox_kwargs({**kw, "secrets": [object()]})
    with pytest.raises(RuntimeError, match="block_network"):
        sandbox_mod.assert_sandbox_kwargs({**kw, "block_network": False})
    with pytest.raises(RuntimeError, match="timeout"):
        sandbox_mod.assert_sandbox_kwargs({k: v for k, v in kw.items() if k != "timeout"})
    argv = sandbox_mod.sandbox_exec_argv("mbpp")
    assert argv[:2] == ["bash", "-lc"] and "ulimit -u 512" in argv[2]
    assert "python -m evalplus.evaluate --dataset mbpp --samples /work/samples.jsonl --parallel 8" in argv[2]
    assert sandbox_mod.sandbox_env("mbpp") == {"MBPP_OVERRIDE_PATH": "/work/MbppPlus-v0.2.0.jsonl.gz"}
    # the orchestrator function itself mounts the volume; the sandbox never does
    fn_kw = sandbox_mod.app.functions["tinycap_sandbox_execute"]
    assert fn_kw["name"] == C.CONTAINER_SANDBOX_EXECUTE and "gpu" not in fn_kw
    plan = sandbox_mod.describe_plan([])
    assert plan["sandbox"]["image"] == C.SANDBOX_IMAGE_DIGEST
    assert plan["sandbox"]["transfer_out"] == C.SANDBOX_RESULTS_PATH


def test_execute_one_happy_path_transfers_validates_then_writes(sandbox_mod, staged, tmp_path, monkeypatch):
    cache, pin, archive = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    job = _job(tmp_path)
    tdir = Path(job["task_dir"])
    good = _results(hash_=pin.gz_md5)
    FakeSandbox.script = lambda fs: fs.__setitem__(C.SANDBOX_RESULTS_PATH, json.dumps(good).encode())
    commits = []
    rec = sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                  cache_dir=str(cache), commit=lambda: commits.append(1))
    [sb] = FakeSandbox.created
    kw = sb.kw
    assert kw["app"] == "APP" and kw["image"] == "IMG"
    assert kw["block_network"] is True and kw["cpu"] == 8 and kw["memory"] == 16384 and kw["timeout"] == 3600
    assert not (set(kw) & set(C.SANDBOX_CREATE_KWARGS_FORBIDDEN))
    assert kw["env"] == {"HUMANEVAL_OVERRIDE_PATH": f"/work/{pin.gz_filename}"}
    assert kw["_args"] == ()                                  # no entrypoint: idle until exec
    # transfer IN: exactly the samples + the archive, nothing else
    assert set(sb.fs) - {C.SANDBOX_RESULTS_PATH} == {C.SANDBOX_SAMPLES_PATH, f"/work/{pin.gz_filename}"}
    assert sb.fs[f"/work/{pin.gz_filename}"] == archive
    assert sb.fs[C.SANDBOX_SAMPLES_PATH] == (tdir / er.SAMPLES_FILENAME).read_bytes()
    assert sb.dirs == [(C.SANDBOX_WORKDIR, True)]
    [(argv, timeout)] = sb.execs
    assert argv == tuple(sandbox_mod.sandbox_exec_argv("humaneval")) and timeout == C.SANDBOX_TIMEOUT_S
    assert sb.terminated
    # OUT: validated results + provenance on the volume, rows converted, commits made
    assert json.loads((tdir / er.EVAL_RESULTS_FILENAME).read_text()) == good
    ex = json.loads((tdir / er.EXECUTION_FILENAME).read_text())
    assert ex["image_digest"] == C.SANDBOX_IMAGE_DIGEST and ex["returncode"] == 0
    assert ex["archive"]["md5"] == pin.gz_md5 and ex["create_kwargs"]["block_network"] is True
    assert "volumes" not in ex["create_kwargs"] and "secrets" not in ex["create_kwargs"]
    rows = [json.loads(line) for line in (Path(job["cell_dir"]) / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and rec["summary"]["pass_at_1_plus"] == 0.5
    assert len(commits) == 2
    # the conversion COMPLETED the task (done marker); the merge is the launcher's, and the record says so
    assert rec["stage"] == sandbox_mod.STAGE_EXECUTE == "execute" and rec["converted"] is True
    assert rec["complete"] is True and rec["merge"].startswith("pending")
    assert completion.is_done(job["cell_dir"], job["task"])
    assert not (tmp_path / "sweep" / "tok" / "controls_summary.json").exists()   # never written HERE
    # nothing pending afterwards — the done marker is authoritative
    assert sandbox_mod.pending_executions(tmp_path / "sweep") == []
    assert sandbox_mod.converted_cell_dirs([{"job": job, "ok": True, "record": rec}]) == [job["cell_dir"]]


def test_convert_stage_resumes_validated_results_without_a_sandbox(sandbox_mod, staged, tmp_path, monkeypatch):
    """R7: a crash AFTER the results write and BEFORE conversion leaves a
    validated ``eval_results.json`` with no marker. The task resumes at
    CONVERSION — no second sandbox — and ends with its marker; a results
    file that does not validate against the STAGED archive refuses with the
    remedy and converts nothing. Both executions here record the samples
    digest their results were produced from (E50 #4), which is what the resume
    binds against; the unbound and contradicted cases are
    ``test_a_resume_refuses_results_it_cannot_bind_to_the_samples_beside_them``."""
    cache, pin, _ = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    job = _job(tmp_path)
    tdir = Path(job["task_dir"])
    (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(_results(hash_=pin.gz_md5)))
    _execution(sandbox_mod, tdir, archive_md5=pin.gz_md5, wall_clock_s=3.0)
    [pending] = sandbox_mod.pending_executions(tmp_path / "sweep")
    assert pending == {**job, "stage": sandbox_mod.STAGE_CONVERT}
    rec = sandbox_mod.execute_one(pending, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                  cache_dir=str(cache))
    assert FakeSandbox.created == []                                  # no sandbox was created
    assert rec["stage"] == "convert" and rec["converted"] and rec["complete"]
    assert rec["resumed_from"] == str(tdir / er.EVAL_RESULTS_FILENAME)
    assert completion.is_done(job["cell_dir"], job["task"])
    assert sandbox_mod.pending_executions(tmp_path / "sweep") == []
    rows = [json.loads(line) for line in (Path(job["cell_dir"]) / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    # results evaluated against ANOTHER archive (the execution says nothing about
    # which): the job looks convertible on disk — its samples ARE the ones those
    # results were produced from — but conversion refuses against the staged md5
    job2 = _job(tmp_path / "other")
    tdir2 = Path(job2["task_dir"])
    (tdir2 / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(_results(hash_="9" * 32)))
    _execution(sandbox_mod, tdir2)
    [pending2] = sandbox_mod.pending_executions(tmp_path / "other" / "sweep")
    assert pending2["stage"] == sandbox_mod.STAGE_CONVERT
    with pytest.raises(RuntimeError, match="remove it"):
        sandbox_mod.execute_one(pending2, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                cache_dir=str(cache))
    assert FakeSandbox.created == [] and not completion.is_done(job2["cell_dir"], job2["task"])
    assert not (Path(job2["cell_dir"]) / "rows.jsonl").exists()
    with pytest.raises(RuntimeError, match="unknown job stage"):
        sandbox_mod.execute_one({**job, "stage": "later"}, sandbox_cls=FakeSandbox, app_obj="APP",
                                image="IMG", cache_dir=str(cache))


def test_an_execution_records_the_samples_it_evaluated(sandbox_mod, staged, tmp_path, monkeypatch):
    """E50 #4: the sandbox writes the sha256 of the EXACT ``samples.jsonl`` it
    transferred in, so the results it leaves behind can be told apart from
    results of a later regeneration under the same task ids. It goes in
    ``execution.json``: ``eval_results.json`` is evalplus's own output and
    ``validate_eval_results`` fixes its top-level keys, so it has no provenance
    block to extend and gaining one would fail the seam validator."""
    cache, pin, _ = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    job = _job(tmp_path)
    tdir = Path(job["task_dir"])
    FakeSandbox.script = lambda fs: fs.__setitem__(
        C.SANDBOX_RESULTS_PATH, json.dumps(_results(hash_=pin.gz_md5)).encode())
    rec = sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                  cache_dir=str(cache))
    want = hashlib.sha256((tdir / er.SAMPLES_FILENAME).read_bytes()).hexdigest()
    ex = json.loads((tdir / er.EXECUTION_FILENAME).read_text())
    assert ex[sandbox_mod.SAMPLES_DIGEST_FIELD] == want == rec[sandbox_mod.SAMPLES_DIGEST_FIELD]
    assert sandbox_mod.samples_sha256(tdir / er.SAMPLES_FILENAME) == want
    # the digest is of the bytes the SANDBOX got, not of some other rendering
    assert hashlib.sha256(FakeSandbox.created[0].fs[C.SANDBOX_SAMPLES_PATH]).hexdigest() == want
    # ...and eval_results.json is untouched evalplus output: no provenance added
    assert set(json.loads((tdir / er.EVAL_RESULTS_FILENAME).read_text())) == {"date", "hash", "eval"}
    assert sandbox_mod.samples_binding(tdir)[0] == sandbox_mod.BINDING_BOUND


def test_a_resume_refuses_results_it_cannot_bind_to_the_samples_beside_them(
        sandbox_mod, staged, tmp_path, monkeypatch):
    """E50 #4, the finding: a crash-resume validated task ids, count, shape and
    the dataset archive md5 — every one of which survives regenerating the same
    problems with DIFFERENT solutions — so OLD results could be converted
    against NEW samples. Three states, one rule:

    * the samples are the ones the execution recorded -> convert (above);
    * the samples CHANGED -> refuse, naming both digests and the remedy;
    * nothing recorded which samples produced them (a pre-digest
      ``execution.json``, or none at all) -> refuse too. A pre-digest execution
      cannot vouch for its samples, and a refusal costs one re-execution while
      accepting it costs a wrong number.

    The refusal is per task and at the SCAN as well as at the conversion, so
    the dry run states it, one bad task never stops the others, and the task
    stays pending until the remedy is applied."""
    cache, pin, _ = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)

    def _staged_results(where):
        job = _job(where)
        tdir = Path(job["task_dir"])
        (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(_results(hash_=pin.gz_md5)))
        return job, tdir

    # (a) the samples were REGENERATED after the results: both digests named
    job, tdir = _staged_results(tmp_path / "changed")
    old = _execution(sandbox_mod, tdir, archive_md5=pin.gz_md5)[sandbox_mod.SAMPLES_DIGEST_FIELD]
    er.write_samples_jsonl(tdir / er.SAMPLES_FILENAME,
                           [{"task_id": t, "solution": "def f(): return 1  # regenerated"}
                            for t in PROBLEMS])
    new = sandbox_mod.samples_sha256(tdir / er.SAMPLES_FILENAME)
    assert new != old
    verdict, msg = sandbox_mod.samples_binding(tdir)
    assert verdict == sandbox_mod.BINDING_MISMATCH and old in msg and new in msg
    assert "re-run the sandbox stage" in msg.lower()
    [p] = sandbox_mod.pending_executions(tmp_path / "changed" / "sweep")
    assert p["stage"] == sandbox_mod.STAGE_REFUSE and p["refusal"] == msg
    with pytest.raises(RuntimeError, match="regenerated after those results"):
        sandbox_mod.execute_one(p, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                cache_dir=str(cache))
    # nothing was converted, no sandbox was created, and the task stays pending
    assert FakeSandbox.created == [] and not completion.is_done(job["cell_dir"], job["task"])
    assert not (Path(job["cell_dir"]) / "rows.jsonl").exists()
    assert sandbox_mod.pending_executions(tmp_path / "changed" / "sweep")[0]["stage"] == "refuse"
    # the refusal survives a caller that builds the job itself (the conversion
    # door checks too, not only the scan)
    with pytest.raises(RuntimeError, match="regenerated after those results"):
        sandbox_mod.execute_one({**job, "stage": sandbox_mod.STAGE_CONVERT}, sandbox_cls=FakeSandbox,
                                app_obj="APP", image="IMG", cache_dir=str(cache))

    # (b) a LEGACY execution.json, written before the digest existed
    job_b, tdir_b = _staged_results(tmp_path / "legacy")
    (tdir_b / er.EXECUTION_FILENAME).write_text(json.dumps({"archive": {"md5": pin.gz_md5},
                                                            "wall_clock_s": 3.0}))
    verdict, msg = sandbox_mod.samples_binding(tdir_b)
    assert verdict == sandbox_mod.BINDING_UNBOUND
    assert sandbox_mod.SAMPLES_DIGEST_FIELD in msg and "re-run the sandbox stage" in msg.lower()
    [pb] = sandbox_mod.pending_executions(tmp_path / "legacy" / "sweep")
    assert pb["stage"] == sandbox_mod.STAGE_REFUSE
    with pytest.raises(RuntimeError, match="cannot vouch for its samples"):
        sandbox_mod.execute_one(pb, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                cache_dir=str(cache))
    assert not completion.is_done(job_b["cell_dir"], job_b["task"])

    # (c) no execution.json at all: the same reading, and a malformed digest too
    job_c, tdir_c = _staged_results(tmp_path / "none")
    assert sandbox_mod.samples_binding(tdir_c)[0] == sandbox_mod.BINDING_UNBOUND
    assert sandbox_mod.pending_executions(tmp_path / "none" / "sweep")[0]["stage"] == "refuse"
    (tdir_c / er.EXECUTION_FILENAME).write_text(json.dumps(
        {sandbox_mod.SAMPLES_DIGEST_FIELD: "not-a-sha"}))
    assert sandbox_mod.samples_binding(tdir_c)[0] == sandbox_mod.BINDING_UNBOUND
    (tdir_c / er.EXECUTION_FILENAME).write_text("{ torn")
    assert sandbox_mod.samples_binding(tdir_c)[0] == sandbox_mod.BINDING_UNBOUND
    assert FakeSandbox.created == []

    # a refusing task does not stop the others: the plan carries both
    plan = sandbox_mod.describe_plan(sandbox_mod.pending_executions(tmp_path / "none" / "sweep")
                                     + [{**job_c, "stage": sandbox_mod.STAGE_EXECUTE}])
    assert [j["stage"] for j in plan["jobs"]] == ["refuse", "execute"] and plan["n_pending"] == 2
    # and a refusal stage with no reason recorded still raises rather than running
    with pytest.raises(RuntimeError, match="no reason recorded"):
        sandbox_mod.execute_one({**job_c, "stage": sandbox_mod.STAGE_REFUSE},
                                sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                cache_dir=str(cache))


def test_orchestration_path_converts_then_the_client_venv_merge_lands_evalplus(
        sandbox_mod, launcher_mod, staged, tmp_path, monkeypatch):
    """N1 end to end, on the REAL path: fake sandbox results -> validate ->
    ``convert_eval_results_to_store`` (done marker + summary; NO
    controls_summary here — the orchestrator image cannot import the shared
    writer) -> the launcher's merge step (``merge_cell_summaries``: a
    subprocess running ``completion.merge_controls_summary`` where
    ``sisteer.eval.cell_controls`` imports — this venv stands in for the
    client venv) -> the EXISTING consumer ``load_controls_summaries`` sees
    the EvalPlus payload under a provenance re-validated against the object
    on disk. Then the chained command the orchestrator entrypoint runs."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cache, pin, _ = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cfg, cell, cell_dir, _tdir, job = _tinycap_store(tmp_path, monkeypatch)
    out_root = tmp_path / "sweep"
    assert sandbox_mod.pending_executions(out_root) == [job]
    FakeSandbox.script = lambda fs: fs.__setitem__(
        C.SANDBOX_RESULTS_PATH, json.dumps(_results(hash_=pin.gz_md5)).encode())
    rec = sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                  cache_dir=str(cache))
    assert rec["converted"] and rec["complete"] and rec["merge"].startswith("pending")
    summary_path = out_root / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    assert completion.is_done(cell_dir, job["task"]) and not summary_path.exists()
    # the merge step, exactly as the launcher's Modal function runs it
    records = [{"job": job, "ok": True, "record": rec}]
    to_merge = sandbox_mod.converted_cell_dirs(records)
    assert to_merge == [str(cell_dir)]
    res = launcher_mod.merge_cell_summaries(to_merge, tree=REPO, python=sys.executable)
    notes = res[str(cell_dir)].pop("notes")                               # PW14: what the writer said, per store
    assert res == {str(cell_dir): {"ok": True, "merged": True, "path": str(summary_path)}} and isinstance(notes, list)
    doc = json.loads(summary_path.read_text())
    assert set(doc["controls"]) == {"evalplus/humaneval_plus"}
    assert doc["provenance"] == json.loads((cell_dir / "manifest.json").read_text())["provenance"]
    loaded = load_controls_summaries(out_root, [cell], cfg)
    got = loaded[C.CELL_PSRKL_SAFETY]["evalplus/humaneval_plus"]
    assert got["pass_at_1_plus"] == 0.5 and got["pass_at_1_base"] == 0.5 and got["n"] == 2
    assert got["framework"] == C.EVALPLUS_TASK_VERSION and got["source"] == "cell_controls"
    # idempotent: a second merge rewrites the file byte-identical
    before = summary_path.read_bytes()
    launcher_mod.merge_cell_summaries(to_merge, tree=REPO, python=sys.executable)
    assert summary_path.read_bytes() == before
    # a partial (limited) conversion never merges — and is never in the chained list
    assert sandbox_mod.converted_cell_dirs([{"job": job, "ok": True, "record": {**rec, "complete": False}}]) == []
    assert sandbox_mod.converted_cell_dirs([{"job": job, "ok": False, "error": "x"}]) == []
    # the chained launcher command the orchestrator entrypoint runs (--merge-sha)
    cmd = sandbox_mod.merge_command("e39f8ca", to_merge)
    assert cmd[:4] == [sys.executable, "-m", "modal", "run"]
    assert cmd[4] == str(REPO / "scripts" / "modal" / "tinycap_served.py")
    assert cmd[5:] == ["--sha", "e39f8ca", "--merge", "--merge-cell-dirs", str(cell_dir),
                       "--app-name", sandbox_mod.MERGE_APP_NAME]          # the launcher requires its app (E40 M7)
    with pytest.raises(ValueError, match="no cell dirs"):
        sandbox_mod.merge_command("e39f8ca", [])
    # the entrypoint refuses to run executions with no merge decision
    with pytest.raises(SystemExit, match="--merge-sha"):
        sandbox_mod.main()
    # the merge step lives in the launcher (client venv); the orchestrator image stays stdlib + pyyaml
    src = (REPO / "scripts" / "modal" / "tinycap_sandbox.py").read_text()
    assert "sisteer.eval.cell_controls" not in src.split("orchestrator_image = (")[1].split(")")[0]
    assert 'pip_install("pyyaml==6.0.3")' in src and "torch" not in src.split("orchestrator_image = (")[1].split(")")[0]


@pytest.mark.parametrize("failure", ["malformed", "wrong_hash", "nonzero_rc", "missing"])
def test_execute_one_refuses_and_writes_nothing(sandbox_mod, staged, tmp_path, monkeypatch, failure):
    cache, pin, archive = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    job = _job(tmp_path)
    tdir = Path(job["task_dir"])
    if failure == "malformed":
        bad = _results(hash_=pin.gz_md5)
        bad["eval"]["HumanEval/0"][0]["plus_status"] = None
        FakeSandbox.script = lambda fs: fs.__setitem__(C.SANDBOX_RESULTS_PATH, json.dumps(bad).encode())
        match = "plus_status"
    elif failure == "wrong_hash":
        FakeSandbox.script = lambda fs: fs.__setitem__(
            C.SANDBOX_RESULTS_PATH, json.dumps(_results(hash_="9" * 32)).encode())
        match = "dataset hash"
    elif failure == "nonzero_rc":
        FakeSandbox.rc = 3
        match = "exited 3"
    else:
        FakeSandbox.script = None          # evaluate produced no file
        match = "no readable results"
    with pytest.raises((RuntimeError, er.EvalPlusSeamError, ValueError), match=match):
        sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                cache_dir=str(cache), commit=None)
    [sb] = FakeSandbox.created
    assert sb.terminated
    assert not (tdir / er.EVAL_RESULTS_FILENAME).exists()
    assert not (tdir / er.EXECUTION_FILENAME).exists()
    assert not (Path(job["cell_dir"]) / "rows.jsonl").exists()
    assert sandbox_mod.pending_executions(tmp_path / "sweep") == [job]  # stays pending


def test_execute_refuses_a_drifted_or_missing_archive_before_any_sandbox(sandbox_mod, staged, tmp_path, monkeypatch):
    cache, pin, archive = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    job = _job(tmp_path)
    (cache / pin.gz_filename).write_bytes(archive + b"drift")
    with pytest.raises(RuntimeError, match="drifted"):
        sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG", cache_dir=str(cache))
    (cache / pin.gz_filename).unlink()
    with pytest.raises(RuntimeError, match="stage-datasets"):
        sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG", cache_dir=str(cache))
    assert FakeSandbox.created == []
    # a partial samples file never reaches a sandbox either
    (cache / pin.gz_filename).write_bytes(archive)
    er.write_samples_jsonl(Path(job["task_dir"]) / er.SAMPLES_FILENAME, [{"task_id": "HumanEval/0", "solution": "x"}])
    with pytest.raises(er.EvalPlusSeamError, match="task-ID set"):
        sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG", cache_dir=str(cache))
    assert FakeSandbox.created == []


def test_pending_executions_discovery(sandbox_mod, tmp_path, monkeypatch):
    """R7: the done marker is authoritative; validated results without one
    resume at conversion; ``{}`` / malformed / foreign-hash results are NOT
    completion (they re-execute, the stale file named)."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    job = _job(tmp_path)
    root = tmp_path / "sweep"
    assert sandbox_mod.pending_executions(root) == [job] and job["stage"] == "execute"
    assert sandbox_mod.pending_executions(root, datasets=("mbpp",)) == []
    tdir, cell = Path(job["task_dir"]), Path(job["cell_dir"])
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    (tdir / er.GEN_META_FILENAME).write_text(json.dumps({**meta, "status": "incomplete"}))
    assert sandbox_mod.pending_executions(root) == []
    (tdir / er.GEN_META_FILENAME).write_text(json.dumps(meta))
    # `{}` or malformed results: still pending, at the EXECUTE stage, the stale file named
    (tdir / er.EVAL_RESULTS_FILENAME).write_text("{}")
    [p] = sandbox_mod.pending_executions(root)
    assert p["stage"] == sandbox_mod.STAGE_EXECUTE and "top-level keys" in p["stale_results"]
    (tdir / er.EVAL_RESULTS_FILENAME).write_text("not json")
    [p] = sandbox_mod.pending_executions(root)
    assert p["stage"] == "execute" and "unreadable" in p["stale_results"]
    assert sandbox_mod.results_state(tdir, dataset="humaneval")[0] == "invalid"
    # validated results BOUND to the samples beside them, no marker: pending at
    # CONVERSION, never a new sandbox
    (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(_results()))
    _execution(sandbox_mod, tdir)
    [p] = sandbox_mod.pending_executions(root)
    assert p == {**job, "stage": sandbox_mod.STAGE_CONVERT} and "stale_results" not in p
    assert sandbox_mod.results_state(tdir, dataset="humaneval") == ("valid", None)
    # ...unless execution.json says they were evaluated against another archive
    _execution(sandbox_mod, tdir, archive_md5="9" * 32)
    [p] = sandbox_mod.pending_executions(root)
    assert p["stage"] == "execute" and "dataset hash" in p["stale_results"]
    _execution(sandbox_mod, tdir)
    # the done marker is authoritative: nothing pending, whatever else is on disk
    completion.finish_task(cell, task=job["task"], n_items=2, n_expected_full=2, limits={}, n_rows=6,
                           n_requests=2, n_rejected=0, wall_clock_s=1.0, summary={"acc": 0.5, "n": 2})
    (tdir / er.EVAL_RESULTS_FILENAME).write_text("{}")
    assert sandbox_mod.pending_executions(root) == []
    completion.done_marker_path(cell, job["task"]).unlink()
    [p] = sandbox_mod.pending_executions(root)
    assert p["stage"] == "execute"
    # a partial marker (a limited store's terminal state) is converted too: not pending
    completion.finish_task(cell, task=job["task"], n_items=2, n_expected_full=2, limits={"limit": 2},
                           n_rows=6, n_requests=2, n_rejected=0, wall_clock_s=1.0, summary={"acc": 0.5, "n": 2})
    assert sandbox_mod.pending_executions(root) == []
    assert sandbox_mod.task_dir_parts(Path("x/cells/k/evalplus/mbpp_open")) == {
        "dataset": "mbpp", "open_variant": True, "task": "evalplus/mbpp_plus:open"}
    with pytest.raises(RuntimeError):
        sandbox_mod.task_dir_parts(Path("x/cells/k/other/mbpp"))
    # the dry-run document carries the stage per job
    plan = sandbox_mod.describe_plan([{**job, "stage": "convert"}])
    assert plan["jobs"][0]["stage"] == "convert" and plan["n_pending"] == 1


# --------------------------------------------------------------------------- #
# the merge: crash-safe (E12 item 2) and bound to the producing tree (item 3)  #
# --------------------------------------------------------------------------- #


FULL_A, FULL_B = "a" * 40, "b" * 40


def _stamp_producer(cell_dir: Path, tree_sha) -> None:
    m = json.loads((cell_dir / "manifest.json").read_text())
    m[C.MANIFEST_KEY_PRODUCER] = {"tree_sha": tree_sha, "driver": C.PRODUCER_DRIVER_SERVED}
    (cell_dir / "manifest.json").write_text(json.dumps(m))


def test_merge_pending_is_derived_from_disk_and_survives_a_crash_after_conversion(
        sandbox_mod, launcher_mod, staged, tmp_path, monkeypatch):
    """E12 item 2: a conversion that committed before the merge ran (the
    process died, or a run that predates the chained merge) is FOUND on the
    next invocation — the DONE marker plus the payload's absence from
    ``controls_summary.json`` IS the durable pending state — and a clean
    retry merges it. ``pending_executions`` alone would never have: there
    the marker is authoritative and the task is simply not pending."""
    from sisteer.eval import cell_controls

    assert sandbox_mod.CONTROLS_SUMMARY_FILENAME == cell_controls.SUMMARY_FILENAME
    cache, pin, _ = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    _cfg, _cell, cell_dir, _tdir, job = _tinycap_store(tmp_path, monkeypatch)
    _stamp_producer(cell_dir, FULL_A)
    out_root = tmp_path / "sweep"
    assert sandbox_mod.unmerged_completed_stores(out_root) == []       # nothing done yet
    FakeSandbox.script = lambda fs: fs.__setitem__(
        C.SANDBOX_RESULTS_PATH, json.dumps(_results(hash_=pin.gz_md5)).encode())
    rec = sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG",
                                  cache_dir=str(cache))
    assert rec["complete"] and "unmerged_completed_stores" in rec["merge"]
    # ---- the orchestrator process dies HERE, before any merge ----------------
    assert sandbox_mod.pending_executions(out_root) == []              # the old path: nothing to do, forever
    assert sandbox_mod.converted_cell_dirs([]) == []                   # a fresh process converted nothing
    [p] = sandbox_mod.unmerged_completed_stores(out_root)
    assert p["cell_dir"] == str(cell_dir) and p["cell_id"] == C.CELL_PSRKL_SAFETY
    assert p["tasks"] == [job["task"]] and p["producer_tree_sha"] == FULL_A and "unmergeable" not in p
    assert p["summary"] == str(out_root / C.CELL_PSRKL_SAFETY / "controls_summary.json")
    assert sandbox_mod.unmerged_completed_stores(out_root, datasets=("mbpp",)) == []
    plan = sandbox_mod.plan_merges([p])
    assert list(plan["commands"]) == [FULL_A]
    assert plan["commands"][FULL_A][5:] == ["--sha", FULL_A, "--merge", "--merge-cell-dirs", str(cell_dir),
                                            "--app-name", sandbox_mod.MERGE_APP_NAME]
    assert plan["refused"] == {} and plan["unmergeable"] == [] and plan["groups"] == {FULL_A: [str(cell_dir)]}
    # the clean retry: the merge lands the payload and the store leaves the pending list
    res = launcher_mod.merge_cell_summaries([str(cell_dir)], tree=REPO, python=sys.executable)
    assert res[str(cell_dir)]["merged"] is True
    assert sandbox_mod.unmerged_completed_stores(out_root) == []
    # a summary that lost the payload, or a torn one, brings the store back (idempotent)
    summary = out_root / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    doc = json.loads(summary.read_text())
    doc["controls"] = {}
    summary.write_text(json.dumps(doc))
    assert [q["tasks"] for q in sandbox_mod.unmerged_completed_stores(out_root)] == [[job["task"]]]
    summary.write_text("not json")
    assert len(sandbox_mod.unmerged_completed_stores(out_root)) == 1
    # a store whose manifest names no cell_id is reported UNMERGEABLE, never dropped
    m = json.loads((cell_dir / "manifest.json").read_text())
    del m["cell_id"]
    (cell_dir / "manifest.json").write_text(json.dumps(m))
    [q] = sandbox_mod.unmerged_completed_stores(out_root)
    assert "no cell_id" in q["unmergeable"] and q["cell_id"] is None
    assert sandbox_mod.plan_merges([q])["unmergeable"] == [q] and sandbox_mod.plan_merges([q])["commands"] == {}
    # the Modal function returns the disk-derived list, before and after executions
    src = (REPO / "scripts" / "modal" / "tinycap_sandbox.py").read_text()
    body = src[src.index("def tinycap_sandbox_execute("):src.index("@app.local_entrypoint()")]
    assert body.count("unmerged_completed_stores(gen_root, datasets=datasets)") == 2
    assert '"merge_pending": pending' in body and '"merge_pending": plan["merge_pending"]' in body


def test_merge_groups_by_producer_and_refuses_legacy_or_foreign_trees(sandbox_mod):
    """E12 item 3 (orchestrator half): the pending stores are grouped by
    ``manifest.producer.tree_sha`` and each group gets ITS tree's merge
    command; a legacy store (no producer block) is refused with the remedy
    — no override; under ``--merge-sha`` every group produced by another
    tree is refused too."""
    def rec(k, sha, **extra):
        return {"cell_dir": f"/g/cells/{k}", "cell_id": "c", "tasks": ["evalplus/humaneval_plus"],
                "producer_tree_sha": sha, "summary": "/g/c/controls_summary.json", **extra}
    pending = [rec("k1", FULL_A), rec("k2", FULL_B), rec("k3", FULL_A), rec("k0", None),
               rec("kx", None, cell_id=None, summary=None,
                   unmergeable="manifest names no cell_id (the merge cannot locate its summary)")]
    assert sandbox_mod.merge_groups(pending) == {
        FULL_A: ["/g/cells/k1", "/g/cells/k3"], FULL_B: ["/g/cells/k2"], None: ["/g/cells/k0"]}
    plan = sandbox_mod.plan_merges(pending, python="py", launcher=Path("/L/tinycap_served.py"))
    assert set(plan["commands"]) == {FULL_A, FULL_B}
    # E40 M7: the launcher requires its app name, so the chained command states it
    # (and run_merges puts the same name into the subprocess environment)
    assert plan["commands"][FULL_A] == ["py", "-m", "modal", "run", "/L/tinycap_served.py", "--sha", FULL_A,
                                        "--merge", "--merge-cell-dirs", "/g/cells/k1,/g/cells/k3",
                                        "--app-name", sandbox_mod.MERGE_APP_NAME]
    assert sandbox_mod.MERGE_APP_NAME == C.app_name_for("merge") == "tinycap-merge"
    env = sandbox_mod.merge_env(plan["commands"][FULL_A], {"KEEP": "1"})
    assert env == {"KEEP": "1", C.MODAL_APP_NAME_ENV: "tinycap-merge"}
    assert sandbox_mod.merge_command_line(plan["commands"][FULL_A]).startswith(
        f"{C.MODAL_APP_NAME_ENV}=tinycap-merge py -m modal run ")
    [label] = plan["refused"]
    assert "legacy" in label and plan["refused"][label]["dirs"] == ["/g/cells/k0"]
    assert C.TREE_SHA_FILENAME in plan["refused"][label]["why"] and "no override" in plan["refused"][label]["why"]
    assert plan["unmergeable"] == [pending[4]] and plan["groups"] == {
        FULL_A: ["/g/cells/k1", "/g/cells/k3"], FULL_B: ["/g/cells/k2"], "None": ["/g/cells/k0"]}
    # --merge-sha: every group whose producer is not that tree is refused (the operator's expectation, audited)
    plan = sandbox_mod.plan_merges(pending, merge_sha=FULL_A[:7])
    assert list(plan["commands"]) == [FULL_A] and FULL_B in plan["refused"]
    assert "--merge-sha" in plan["refused"][FULL_B]["why"]
    with pytest.raises(ValueError, match="legacy"):
        sandbox_mod.merge_command("", ["/x"])
    with pytest.raises(ValueError, match="no cell dirs"):
        sandbox_mod.merge_command(FULL_A, [])
    assert sandbox_mod.LEGACY_PRODUCER is None


# --------------------------------------------------------------------------- #
# per-task merge provenance + the pass-selection rule (E14 A / PW7)            #
# --------------------------------------------------------------------------- #


EVALPLUS_TASK = "evalplus/humaneval_plus"
MBPP_TASK = "evalplus/mbpp_plus"
#: ``_pass_store(top_pass=SAME)``: the top-level geometry.pass equals the hashed one
SAME = object()
#: ``_pass_store(top_pass=ABSENT)``: no top-level geometry block at all
ABSENT = object()


def _evalplus_summary(task: str, *, acc: float, n: int) -> dict:
    """A finished EvalPlus task's counts-only summary as this tree writes one.
    E55 #2: it NAMES the sample bytes the number was scored on — the durable
    half of the binding, which is what lets the score stay bound when the
    ``evalplus/<dataset>/`` directory is gone (these fixtures hold none).
    Keyed on the NUMBER, not the store, so two stores that exactly witness one
    payload agree on the bytes too — which is what an equal witness means."""
    summary = {"acc": acc, "n": n, "framework": C.EVALPLUS_TASK_VERSION}
    if task in C.EVALPLUS_TASKS:
        summary[er.SAMPLES_DIGEST_FIELD] = hashlib.sha256(
            f"{task}:{acc}:{n}".encode()).hexdigest()
    return summary


def _pass_store(sweep: Path, base_manifest: dict, *, key: str, chash: str, pass_no,
                acc: float, tree_sha: str = FULL_A, task: str = EVALPLUS_TASK,
                n: int = 2, hashed: bool = True, top_pass=SAME) -> Path:
    """A SECOND (third, …) tinycap store of the SAME cell under one sweep —
    what a re-run at another geometry pass or another served identity leaves:
    a different key and ``harness_config_hash``, the same ``cell_id`` and
    ``provenance``, so all of them merge into the cell's ONE
    ``controls_summary.json``. The pass is manufactured THROUGH THE IDENTITY
    (``resolved_config.tinycap.pass``, PW8 / E16 A2) with the top-level
    ``geometry.pass`` copying it; ``hashed=False`` leaves the identity
    without a pass (a store from before PW8) and ``top_pass`` sets the
    top-level field apart (a relabelled store) or drops the block."""
    d = sweep / "cells" / key
    d.mkdir(parents=True, exist_ok=True)
    resolved = dict(base_manifest.get("resolved_config") or {})
    tinycap = dict(resolved.get(C.MANIFEST_KEY_TINYCAP) or {})
    if hashed:
        tinycap[C.GEOMETRY_PASS_FIELD] = pass_no
    else:
        tinycap.pop(C.GEOMETRY_PASS_FIELD, None)
    resolved[C.MANIFEST_KEY_TINYCAP] = tinycap
    manifest = {**base_manifest, "harness_config_hash": chash, "resolved_config": resolved,
                C.MANIFEST_KEY_GEOMETRY: {
                    C.GEOMETRY_PASS_FIELD: pass_no if top_pass is SAME else top_pass},
                C.MANIFEST_KEY_PRODUCER: {"tree_sha": tree_sha,
                                          "driver": C.PRODUCER_DRIVER_SERVED}}
    if top_pass is ABSENT:
        del manifest[C.MANIFEST_KEY_GEOMETRY]
    (d / "manifest.json").write_text(json.dumps(manifest))
    summary = _evalplus_summary(task, acc=acc, n=n)
    completion.finish_task(d, task=task, n_items=n, n_expected_full=n, limits={}, n_rows=n,
                           n_requests=n, n_rejected=0, wall_clock_s=1.0, summary=summary)
    return d


def _core(store: Path, pass_no) -> dict:
    m = json.loads((store / "manifest.json").read_text())
    return {"cell_key": store.name, "harness_config_hash": m["harness_config_hash"],
            "pass": pass_no, "producer_tree_sha": m[C.MANIFEST_KEY_PRODUCER]["tree_sha"]}


def _digest(store: Path, task: str = EVALPLUS_TASK) -> str:
    """The canonical digest of the store's CURRENT ``summary/<task>.json`` (PW14)."""
    return C.merge_payload_digest(task, completion.read_task_summary(store, task))


def _hist(store: Path, pass_no, task: str = EVALPLUS_TASK) -> dict:
    """A ``superseded`` / ``equal_witnesses`` record as the writer files it since
    PW14 (E22 M4): the store's four identity fields plus the digest of the
    payload the decision was made on — the store's own summary for ``task``."""
    return {**_core(store, pass_no), C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD: _digest(store, task)}


def _standing(store: Path, pass_no, *history, task: str = EVALPLUS_TASK, payload=None, **extra) -> dict:
    """A STANDING ``sources`` record as the writer files it since PW16 (E24):
    the store's four identity fields (plus ``extra`` annotations), its
    ``superseded`` history, and the digest of the payload the entry stands
    with — the store's own CURRENT summary for ``task`` unless ``payload``
    names the canonical payload explicitly."""
    digest = C.merge_payload_digest(task, payload) if payload is not None else _digest(store, task)
    # E55 #2: the writer RETAINS the digest of the sample bytes the entry was
    # scored on, read off the entry's own payload when the store holds no
    # generation on this filesystem
    canonical = payload if payload is not None else completion.read_task_summary(store, task)
    bound = (canonical or {}).get(er.SAMPLES_DIGEST_FIELD)
    if bound is not None:
        extra = {**extra, C.MERGE_SOURCE_SAMPLES_DIGEST_FIELD: bound}
    return {**_core(store, pass_no), C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD: _template_class(store), **extra,
            C.MERGE_SOURCE_SUPERSEDED_FIELD: list(history),
            C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD: digest}


def _template_class(store: Path) -> str:
    """The store's template-check class as the ONE reader derives it (P1-1):
    a stated class, or ``unrecorded`` for a store on a declared historical
    tree — exactly what ``completion.merge_source`` annotates the record with."""
    from sisteer.eval import sweep as _sweep
    return _sweep.template_check_of_manifest(json.loads((store / "manifest.json").read_text()))


def test_merge_records_a_source_per_task_and_the_higher_pass_wins(tmp_path, monkeypatch):
    """E14 (A): several stores of ONE cell share the cell's single
    ``controls_summary.json`` (the pass keys them apart by
    ``harness_config_hash``, not by file), so before PW7 whichever merged
    LAST won and the summary could not say which store its numbers came
    from. Now every entry the tinycap merge writes carries its source, and
    the PASS decides: a pass-2 store replaces a pass-1 entry and records it
    as superseded; a pass-1 store leaves a pass-2 entry standing (so the
    outcome is the same whichever order the two pending stores merge in);
    and a re-merge of either is byte-identical."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    p1 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11111111",
                     chash="11111111" + "0" * 8, pass_no=1, acc=0.25)
    p2 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__22222222",
                     chash="22222222" + "0" * 8, pass_no=2, acc=0.75)
    # pass 1 alone: filled, and the summary says exactly which store filled it
    completion.merge_controls_summary(p1)
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.25
    assert doc[C.SUMMARY_KEY_SOURCES] == {EVALPLUS_TASK: _standing(p1, 1)}
    # pass 2 replaces it and keeps pass 1 on the page as superseded
    completion.merge_controls_summary(p2)
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.75
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p2, 2, _hist(p1, 1))
    # pass 1 offered again does NOT take the entry back, and neither re-merge
    # grows the record: the file is byte-identical whichever runs again
    before = summary.read_bytes()
    completion.merge_controls_summary(p1)
    completion.merge_controls_summary(p2)
    completion.merge_controls_summary(p1)
    assert summary.read_bytes() == before
    # so the order the two PENDING stores merge in cannot change the outcome
    other = tmp_path / "sweep2"
    q1 = _pass_store(other, base, key="tok__controls__t0__a0__aaaaaaaa__11111111",
                     chash="11111111" + "0" * 8, pass_no=1, acc=0.25)
    q2 = _pass_store(other, base, key="tok__controls__t0__a0__aaaaaaaa__22222222",
                     chash="22222222" + "0" * 8, pass_no=2, acc=0.75)
    completion.merge_controls_summary(q2)
    completion.merge_controls_summary(q1)
    doc2 = json.loads((other / C.CELL_PSRKL_SAFETY / "controls_summary.json").read_text())
    assert doc2["controls"][EVALPLUS_TASK]["acc"] == 0.75
    assert doc2[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(q2, 2, _hist(q1, 1))
    # and the loader accepts the sourced summary (its provenance is unchanged)
    got = load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY]
    assert got[EVALPLUS_TASK]["acc"] == 0.75


def test_the_merge_stamps_the_stores_reduction_mode_onto_the_payload(tmp_path, monkeypatch):
    """E49, through the REAL merge (``completion.merge_controls_summary`` ->
    ``cell_controls.write_controls_summary``): the payload records the LENS
    REDUCTION MODE of the store's own manifest ``route`` block, so an entry
    from a STACKED store can never reach a pooling reader wearing the absent
    (per_row) reading. ``main``'s controls-summary writer stamps none — that
    is the hole this closes — and this pins that the out_root the merge writes
    under is the one the store is resolved against.

    A store that states no mode (every store in this suite, and every store
    written before the landing) is unchanged: nothing is stamped, and the
    absent per_row reading stands."""
    from sisteer.eval import cell_controls as cc
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"

    # a PRE-LANDING store — no route block at all — stamps nothing
    plain = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11110000",
                        chash="11110000" + "0" * 8, pass_no=1, acc=0.25)
    assert cc.source_store_reduction_mode(sweep, _core(plain, 1))[0] is None
    completion.merge_controls_summary(plain)
    assert C.REDUCTION_MODE_FIELD not in json.loads(summary.read_text())["controls"][EVALPLUS_TASK]
    assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][EVALPLUS_TASK]["acc"] == 0.25

    # a STACKED store, in its own sweep: the merge resolves it under the
    # out_root it writes to and records the mode on the payload
    other = tmp_path / "sweep_stacked"
    st = _pass_store(other, base, key="tok__controls__t0__a0__aaaaaaaa__22220000",
                     chash="22220000" + "0" * 8, pass_no=1, acc=0.75)
    m = json.loads((st / "manifest.json").read_text())
    m[C.MANIFEST_KEY_ROUTE] = {C.REDUCTION_MODE_FIELD: C.REDUCTION_MODE_STACKED}
    (st / "manifest.json").write_text(json.dumps(m))
    assert cc.source_store_reduction_mode(other, _core(st, 1))[0] == C.REDUCTION_MODE_STACKED
    completion.merge_controls_summary(st)
    doc = json.loads((other / C.CELL_PSRKL_SAFETY / "controls_summary.json").read_text())
    assert doc["controls"][EVALPLUS_TASK][C.REDUCTION_MODE_FIELD] == C.REDUCTION_MODE_STACKED
    # the recorded digest is the stamped payload's, so the re-merge is the
    # byte-identical no-op the source discipline promises
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(
        st, 1, payload=doc["controls"][EVALPLUS_TASK])
    before = (other / C.CELL_PSRKL_SAFETY / "controls_summary.json").read_bytes()
    completion.merge_controls_summary(st)
    assert (other / C.CELL_PSRKL_SAFETY / "controls_summary.json").read_bytes() == before
    got = load_controls_summaries(other, [cell], cfg)[C.CELL_PSRKL_SAFETY]
    assert got[EVALPLUS_TASK][C.REDUCTION_MODE_FIELD] == C.REDUCTION_MODE_STACKED
    # ... and had that store been merged by a writer that stamps nothing
    # (main's), the loader would refuse rather than pool it as per_row. The
    # file is made internally consistent (the standing digest re-filed over
    # the unstamped payload, exactly as such a writer would have filed it), so
    # what refuses is THIS guard and not the hand-edit one beside it
    doc["controls"][EVALPLUS_TASK].pop(C.REDUCTION_MODE_FIELD)
    doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD] = (
        C.merge_payload_digest(EVALPLUS_TASK, doc["controls"][EVALPLUS_TASK]))
    (other / C.CELL_PSRKL_SAFETY / "controls_summary.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True))
    with pytest.raises(ValueError) as ei:
        load_controls_summaries(other, [cell], cfg)
    why = str(ei.value)
    assert "E49" in why and "'stacked'" in why and st.name in why
    assert f"names no {C.REDUCTION_MODE_FIELD}" in why


def test_two_stores_at_one_pass_refuse_until_the_operator_names_one(tmp_path, monkeypatch):
    """E14 (A): two COMPLETE stores claiming one task at the SAME geometry
    pass — a re-run under another served identity, say — is ambiguous: the
    file cannot say which number the campaign means, so the merge refuses
    naming both sources instead of letting directory order decide.
    ``replace_source`` (the launcher's / driver's ``--replace-source``) is
    the operator saying which one, and the displaced source is recorded."""
    cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    a = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__1111aaaa",
                    chash="1111aaaa" + "0" * 8, pass_no=1, acc=0.25)
    b = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__1111bbbb",
                    chash="1111bbbb" + "0" * 8, pass_no=1, acc=0.75)
    completion.merge_controls_summary(a)
    with pytest.raises(ValueError, match="two sources claim one entry"):
        completion.merge_controls_summary(b)
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.25          # untouched by the refusal
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK]["cell_key"] == a.name
    completion.merge_controls_summary(b, replace_source={EVALPLUS_TASK: b.name})
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.75
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(b, 1, _hist(a, 1))
    # a store from ANOTHER tree at the same pass is the same ambiguity
    c = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__1111cccc",
                    chash="1111cccc" + "0" * 8, pass_no=1, acc=0.5, tree_sha=FULL_B)
    with pytest.raises(ValueError, match="two sources claim one entry"):
        completion.merge_controls_summary(c)
    # a store whose manifest cannot state the identity it ran under is not a source
    (c / "manifest.json").write_text(json.dumps(
        {k: v for k, v in json.loads((c / "manifest.json").read_text()).items()
         if k != "harness_config_hash"}))
    with pytest.raises(completion.CompletionError, match="no harness_config_hash"):
        completion.merge_controls_summary(c)


def test_a_pre_pw7_summary_is_tolerated_and_stamped_when_a_sourced_merge_lands(
        tmp_path, monkeypatch, capsys):
    """E14 (A), the migration: the summaries a tree BEFORE this discipline
    wrote carry tinycap entries with no source. They still load (the loader
    says so — which store produced them is not recorded, and they are pass 1
    by construction: no pass-2 store can exist there), and when a sourced
    merge later writes into the same file those entries are stamped
    ``UNRECORDED_MERGE_SOURCE`` — the absence RECORDED, never a guess — so a
    tinycap entry MISSING from a sources block can mean torn and nothing
    else."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    # BYTES a pre-PW7 merge left (the writer of the day took no sources): a
    # tinycap payload, no sources block
    old_task = "inspect/math"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(
        {"cell_id": cell.id, "controls": {old_task: {"acc": 0.4, "n": 100}},
         "provenance": base[C.MANIFEST_KEY_PROVENANCE]}, indent=1, sort_keys=True))
    assert C.SUMMARY_KEY_SOURCES not in json.loads(summary.read_text())
    assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][old_task]["acc"] == 0.4
    assert "no 'sources' block" in capsys.readouterr().out
    # a sourced merge lands beside it: the old entry's absence becomes a record
    p1 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__33333333",
                     chash="33333333" + "0" * 8, pass_no=1, acc=0.25)
    completion.merge_controls_summary(p1)
    doc = json.loads(summary.read_text())
    assert doc[C.SUMMARY_KEY_SOURCES][old_task] == {
        **dict(C.UNRECORDED_MERGE_SOURCE), C.MERGE_SOURCE_SUPERSEDED_FIELD: [],
        C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD: C.merge_payload_digest(old_task, {"acc": 0.4, "n": 100})}   # PW16
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p1, 1)
    assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][old_task]["acc"] == 0.4
    # an unrecorded source is never comparable, so nothing may quietly take it over
    p2 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__44444444",
                     chash="44444444" + "0" * 8, pass_no=2, acc=0.9, task=old_task, n=100)
    with pytest.raises(ValueError, match="two sources claim one entry"):
        completion.merge_controls_summary(p2)
    # ...and an unrecorded source has no cell key to name, so only the offered store can win
    completion.merge_controls_summary(p2, replace_source={old_task: p2.name})
    doc = json.loads(summary.read_text())
    assert doc["controls"][old_task]["acc"] == 0.9
    assert doc[C.SUMMARY_KEY_SOURCES][old_task][C.MERGE_SOURCE_SUPERSEDED_FIELD] == [
        {**dict(C.UNRECORDED_MERGE_SOURCE),
         C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD: C.merge_payload_digest(old_task, {"acc": 0.4, "n": 100})}]


# --------------------------------------------------------------------------- #
# E16 (PW8): the from-disk scan accounts by EXACT source; the pass is hashed;   #
# --replace-source names the entry                                             #
# --------------------------------------------------------------------------- #


def test_from_disk_scan_accounts_a_store_by_its_exact_source_never_by_task_name(
        sandbox_mod, tmp_path, monkeypatch):
    """E16 (A1): ``unmerged_completed_stores`` used to mark a store merged when
    its TASK NAME was in ``controls`` — so with pass 1 merged, a completed
    pass-2 store of the same cell never entered the merge plan and the pass
    rule was never applied to it. A store is accounted for only when the
    entry's recorded source IS this store's exact source (cell key, hash,
    pass, producer sha) or lists it as superseded; an UNRECORDED source, or a
    summary written before the discipline, satisfies no concrete store."""
    _cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    p1 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11111111",
                     chash="11111111" + "0" * 8, pass_no=1, acc=0.25)
    p2 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__22222222",
                     chash="22222222" + "0" * 8, pass_no=2, acc=0.75)

    def pending():
        return {Path(r["cell_dir"]).name: r for r in sandbox_mod.unmerged_completed_stores(sweep)}

    got = pending()
    assert set(got) == {p1.name, p2.name}                              # nothing merged yet
    assert got[p1.name]["why"] == {EVALPLUS_TASK: "no controls_summary.json"}
    assert got[p2.name]["source"] == _core(p2, 2) and got[p2.name]["tasks"] == [EVALPLUS_TASK]
    # pass 1 merged: the task NAME is in controls, and the pass-2 store is STILL pending
    completion.merge_controls_summary(p1)
    assert EVALPLUS_TASK in json.loads(summary.read_text())["controls"]
    got = pending()
    assert set(got) == {p2.name}
    why = got[p2.name]["why"][EVALPLUS_TASK]
    assert "stands from" in why and p1.name in why
    # the identical source already merged is accounted for (idempotent re-merge changes nothing)
    completion.merge_controls_summary(p1)
    assert set(pending()) == {p2.name}
    # the plan carries it to its producing tree's merge
    plan = sandbox_mod.plan_merges(list(got.values()))
    cmd = plan["commands"][FULL_A]
    # the cell dirs are located by their flag, not by position: since E40 M7 the composed
    # command ends with the launcher's REQUIRED --app-name (the launcher parses flags by name,
    # so the order carries no operational meaning); the refusal / unmergeable pins are exact
    assert cmd[cmd.index("--merge-cell-dirs") + 1] == str(p2)
    assert cmd[-2:] == ["--app-name", sandbox_mod.MERGE_APP_NAME]
    assert plan["refused"] == {} and plan["unmergeable"] == []
    # pass 2 merged: BOTH accounted for — p2 standing, p1 in its superseded list
    completion.merge_controls_summary(p2)
    assert pending() == {}
    doc = json.loads(summary.read_text())
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][C.MERGE_SOURCE_SUPERSEDED_FIELD] == [_hist(p1, 1)]
    # an entry stamped UNRECORDED satisfies no concrete store
    doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] = {**dict(C.UNRECORDED_MERGE_SOURCE),
                                                 C.MERGE_SOURCE_SUPERSEDED_FIELD: []}
    summary.write_text(json.dumps(doc))
    got = pending()
    assert set(got) == {p1.name, p2.name}
    assert all("unrecorded" in r["why"][EVALPLUS_TASK] for r in got.values())
    # a summary from before the discipline (the task in controls, no sources block): pending, named
    del doc[C.SUMMARY_KEY_SOURCES]
    summary.write_text(json.dumps(doc))
    got = pending()
    assert set(got) == {p1.name, p2.name}
    assert all("names no source" in r["why"][EVALPLUS_TASK] for r in got.values())
    # the standing source's OTHER fields matter too: the same key at another hash is not this store
    doc[C.SUMMARY_KEY_SOURCES] = {EVALPLUS_TASK: {**_core(p2, 2), "harness_config_hash": "ff" * 8,
                                                  C.MERGE_SOURCE_SUPERSEDED_FIELD: []}}
    summary.write_text(json.dumps(doc))
    assert set(pending()) == {p1.name, p2.name}
    # a RELABELLED store — top-level pass disagreeing with the hashed one — is UNMERGEABLE, named
    bad = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__33333333",
                      chash="33333333" + "0" * 8, pass_no=1, acc=0.5, top_pass=2)
    [r] = [r for r in sandbox_mod.unmerged_completed_stores(sweep) if r["cell_dir"] == str(bad)]
    assert "disagrees with the hashed identity" in r["unmergeable"]
    assert sandbox_mod.plan_merges([r])["unmergeable"] == [r]
    # a store from before the pass was identity is compared by what ITS tree's
    # merge recorded — the top-level pass — never by a guess
    old = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__44444444",
                      chash="44444444" + "0" * 8, pass_no=2, acc=0.5, hashed=False)
    src, why = sandbox_mod.store_merge_source(old, json.loads((old / "manifest.json").read_text()))
    assert why is None and src == _core(old, 2)
    hand = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__55555555",
                       chash="55555555" + "0" * 8, pass_no=None, acc=0.5, hashed=False, top_pass=ABSENT)
    src, why = sandbox_mod.store_merge_source(hand, json.loads((hand / "manifest.json").read_text()))
    assert why is None and src["pass"] is None
    # no hash at all: the source cannot be stated
    m = json.loads((hand / "manifest.json").read_text())
    del m["harness_config_hash"]
    src, why = sandbox_mod.store_merge_source(hand, m)
    assert src is None and why == "manifest names no harness_config_hash (the merge cannot record its source)"


def test_merge_source_is_the_hashed_pass_and_refuses_a_relabelled_or_pre_pass_store(
        tmp_path, monkeypatch):
    """E16 (A2): the source's pass is the HASHED identity's — the top-level
    ``geometry.pass`` is rewritten by every resume, so a source built from it
    could stamp rows with a pass they never ran under. A top-level field
    disagreeing with the hashed one refuses; an identity without a pass
    (a store from before PW8) refuses — merged through its own tree, never
    this one; a hand run records null in both."""
    _cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    p2 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__22222222",
                     chash="22222222" + "0" * 8, pass_no=2, acc=0.75)
    m = json.loads((p2 / "manifest.json").read_text())
    assert completion.hashed_pass(m) == 2 == completion.geometry_pass(m)
    assert completion.merge_source(p2, m) == {**_core(p2, 2),
                                              C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD: _template_class(p2)}  # P1-1
    # relabelled: the top-level says 2, the hash says 1 — refused, nothing written
    bad = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__33333333",
                      chash="33333333" + "0" * 8, pass_no=1, acc=0.5, top_pass=2)
    with pytest.raises(completion.CompletionError, match="disagrees with the hashed identity"):
        completion.merge_controls_summary(bad)
    assert not (sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json").exists()
    # an identity without a pass: refused (its own tree merges it)
    pre = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__44444444",
                      chash="44444444" + "0" * 8, pass_no=1, acc=0.5, hashed=False)
    assert completion.hashed_pass(json.loads((pre / "manifest.json").read_text())) is completion.PASS_UNRECORDED
    with pytest.raises(completion.CompletionError, match=f"carries no {C.GEOMETRY_PASS_FIELD!r}"):
        completion.merge_controls_summary(pre)
    # a malformed hashed pass is the same refusal
    hb = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__66666666",
                     chash="66666666" + "0" * 8, pass_no=True, acc=0.5)
    with pytest.raises(completion.CompletionError, match=f"carries no {C.GEOMETRY_PASS_FIELD!r}"):
        completion.merge_controls_summary(hb)
    # a hand run: null in the identity and the block, a source with pass null
    hand = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__55555555",
                       chash="55555555" + "0" * 8, pass_no=None, acc=0.5)
    assert completion.merge_source(hand, json.loads((hand / "manifest.json").read_text()))["pass"] is None
    # no top-level block at all: nothing to cross-check, the hashed pass stands
    solo = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__77777777",
                       chash="77777777" + "0" * 8, pass_no=2, acc=0.5, top_pass=ABSENT)
    assert completion.merge_source(solo, json.loads((solo / "manifest.json").read_text()))["pass"] == 2


def test_replace_source_names_the_entry_and_the_winner_through_the_launcher_merge(
        launcher_mod, tmp_path, monkeypatch):
    """E16 (A3): two stores of one cell claim TWO entries at one pass. Naming
    one entry leaves the other refusing (nothing written); naming both —
    one to each store — takes one and keeps the other; and the launcher's
    client-venv merge carries exactly that value into
    ``completion.merge_controls_summary`` (a real subprocess), the recorded
    argv saying which entry went to which store."""
    _cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    ka, kb = "tok__controls__t0__a0__aaaaaaaa__1111aaaa", "tok__controls__t0__a0__aaaaaaaa__1111bbbb"
    a = _pass_store(sweep, base, key=ka, chash="1111aaaa" + "0" * 8, pass_no=1, acc=0.25)
    _pass_store(sweep, base, key=ka, chash="1111aaaa" + "0" * 8, pass_no=1, acc=0.2, task=MBPP_TASK)
    b = _pass_store(sweep, base, key=kb, chash="1111bbbb" + "0" * 8, pass_no=1, acc=0.75)
    _pass_store(sweep, base, key=kb, chash="1111bbbb" + "0" * 8, pass_no=1, acc=0.7, task=MBPP_TASK)
    completion.merge_controls_summary(a)
    before = summary.read_bytes()
    with pytest.raises(ValueError, match="two sources claim one entry") as ei:
        completion.merge_controls_summary(b)
    assert f"{C.REPLACE_SOURCE_FLAG} {EVALPLUS_TASK}=<cell_key>" in str(ei.value)
    with pytest.raises(ValueError, match=f"{MBPP_TASK!r} is already filled"):
        completion.merge_controls_summary(b, replace_source={EVALPLUS_TASK: kb})
    assert summary.read_bytes() == before
    # the launcher's merge step: the SAME refusal is recorded, not raised, and the file untouched
    res = launcher_mod.merge_cell_summaries([str(b)], tree=REPO, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={kb}")
    assert res[str(b)]["ok"] is False and MBPP_TASK in res[str(b)]["err"]
    assert summary.read_bytes() == before
    # both named through the launcher: humaneval to B, mbpp kept by A
    res = launcher_mod.merge_cell_summaries(
        [str(b)], tree=REPO, python=sys.executable,
        replace_source=f"{MBPP_TASK}={ka},{EVALPLUS_TASK}={kb}")
    notes = res[str(b)].pop("notes")                                      # PW14: the writer's stdout reaches the REC
    assert res[str(b)] == {"ok": True, "merged": True, "path": str(summary)}
    assert any("now from" in n for n in notes) and any("KEPT from" in n for n in notes)
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.75 and doc["controls"][MBPP_TASK]["acc"] == 0.2
    W = launcher_mod.writer_tree_sha_of(REPO / "src")                   # PW12: the launcher's write names its tree
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(b, 1, _hist(a, 1), **{C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD: W})
    assert doc[C.SUMMARY_KEY_SOURCES][MBPP_TASK] == _standing(a, 1, _hist(b, 1, MBPP_TASK), task=MBPP_TASK)
    # a malformed value never reaches the merge
    with pytest.raises(ValueError, match="not an entry the tinycap merge writes"):
        launcher_mod.merge_cell_summaries([str(b)], tree=REPO, python=sys.executable,
                                          replace_source=f"mmlu={kb}")
    assert launcher_mod.merge_argv(["/x"], "/scratch", writer="/w", writer_tree_sha=W,
                                   replace_source=f"{MBPP_TASK}={ka},{EVALPLUS_TASK}={kb}")[4:] == [
        C.REPLACE_SOURCE_FLAG, f"{EVALPLUS_TASK}={kb},{MBPP_TASK}={ka}", "/w", W, "/scratch", "/x"]   # canonical order
    # settled: the sweep re-merges both stores without the flag, byte-identical
    settled = summary.read_bytes()
    res = launcher_mod.merge_cell_summaries([str(a), str(b)], tree=REPO, python=sys.executable)
    assert all(v["ok"] for v in res.values()) and summary.read_bytes() == settled


# --------------------------------------------------------------------------- #
# E17 (PW10): the launcher's merge is TWO-TREE — the producing tree EMITS, the  #
# current tree WRITES; a pre-source-discipline writer never touches the file    #
# --------------------------------------------------------------------------- #


LEGACY_FIXTURES = REPO / "tests" / "fixtures" / "tinycap_legacy_tree"
#: the producer sha the synthetic 618f9dd-style tree is stamped with (synthetic 40-hex)
LEGACY_SHA = "618f9dd" + "e" * 33
#: a second producing tree = THIS checkout, staged (synthetic sha)
CURRENT_SHA = "c0ffee0" + "0" * 33
#: ``_legacy_tree(completion=...)``: the REAL 618f9dd completion module (the fixture)
LEGACY_COMPLETION = LEGACY_FIXTURES / "completion_618f9dd.py"
#: a completion module from before the summary discipline (no complete_task_summaries)
STUB_COMPLETION = object()
#: no tinycap/completion.py at all (a tree from before that module)
NO_COMPLETION = object()


def _legacy_tree(repos: Path, sha: str = LEGACY_SHA, *, completion=LEGACY_COMPLETION) -> Path:
    """A staged tree ``<repos>/sisteer-<sha[:7]>`` the way ``--upload-tree
    --stage`` leaves one, whose ``tinycap/completion.py`` and
    ``eval/cell_controls.py`` are the REAL 618f9dd files (vendored under
    ``tests/fixtures/tinycap_legacy_tree/``) around a copy of today's
    torch-free ``sisteer.capeval`` package plus the registry's stdlib
    decode-policy reader and its package-data export (``contracts`` reads
    every sampling number from it since the results-registry redirect — so
    the fixture's ``contracts`` import resolves inside the tree
    alone, as a real ``git archive`` tree does). Its ``merge_controls_summary``
    is the pre-source-discipline writer — overlay, no ``sources`` — that the
    legacy planner ran against the canonical file."""
    tree = repos / f"sisteer-{sha[:7]}"
    src = tree / "src" / "sisteer"
    shutil.copytree(REPO / "src" / "sisteer" / "capeval", src / "capeval",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (src / "__init__.py").write_text("")
    (src / "registry").mkdir()
    for name in ("__init__.py", "decode_policy.py", "decode_policy.json"):
        shutil.copy(REPO / "src" / "sisteer" / "registry" / name, src / "registry" / name)
    comp = src / "capeval" / "tinycap" / "completion.py"
    if completion is NO_COMPLETION:
        comp.unlink()
    elif completion is STUB_COMPLETION:
        comp.write_text('"""A tinycap.completion from before the summary discipline (W1)."""\n'
                        "def is_done(cell_dir, task):\n    return False\n")
    else:
        shutil.copy(completion, comp)
    (src / "eval").mkdir()
    (src / "eval" / "__init__.py").write_text("")
    shutil.copy(LEGACY_FIXTURES / "cell_controls_618f9dd.py", src / "eval" / "cell_controls.py")
    (tree / "scripts").mkdir()
    (tree / C.TREE_SHA_FILENAME).write_text(sha + "\n")
    return tree


def _current_tree(repos: Path, sha: str = CURRENT_SHA) -> Path:
    """THIS checkout staged under ``sha``: ``src/sisteer`` a symlink to the
    real package, stamped — a second producing tree beside the legacy one."""
    tree = repos / f"sisteer-{sha[:7]}"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "sisteer").symlink_to(REPO / "src" / "sisteer", target_is_directory=True)
    (tree / "scripts").mkdir()
    (tree / C.TREE_SHA_FILENAME).write_text(sha + "\n")
    return tree


def _merge_through_own_tree(tree: Path, store: Path) -> subprocess.CompletedProcess:
    """What the legacy planner chained: the producing tree's OWN
    ``merge_controls_summary`` against the canonical file."""
    return subprocess.run(
        [sys.executable, "-c",
         ("import sys; from sisteer.capeval.tinycap import completion; "
          "completion.merge_controls_summary(sys.argv[1])"), str(store)],
        env={**os.environ, "PYTHONPATH": f"{tree}/src"}, capture_output=True, text=True,
        check=False)


def test_a_pre_discipline_writer_never_touches_the_canonical_summary(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """The migration regression: a sourced pass-2 winner stands in
    the canonical summary; a 618f9dd-style store of the same cell (no hashed
    pass, top-level pass 1, produced by the legacy tree) is pending. The
    legacy planner ran that tree's OWN merge — REPRODUCED here with the real
    618f9dd writer: the newer value is replaced and every source erased.
    The two-tree merge instead has the legacy tree EMIT and this tree WRITE:
    the full ``sources`` block survives, the legacy store's claim REFUSES
    until the operator names the standing store (a pass read off the mutable
    top-level block never orders), whereupon it is recorded superseded (so
    the from-disk scan accounts for it), and a legacy store recording NO
    pass is refused as ambiguous."""
    _cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    tree = _legacy_tree(tmp_path / "repos")
    p2 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__22222222",
                     chash="22222222" + "0" * 8, pass_no=2, acc=0.9)
    completion.merge_controls_summary(p2)
    sourced = summary.read_bytes()
    doc = json.loads(sourced)
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.9
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p2, 2)
    legacy = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd0",
                         chash="618f9dd0" + "0" * 8, pass_no=1, acc=0.1, hashed=False, tree_sha=LEGACY_SHA)
    assert [r["cell_dir"] for r in sandbox_mod.unmerged_completed_stores(sweep)] == [str(legacy)]
    # THE REPRODUCTION (what --merge once chained): 618f9dd's own writer
    r = _merge_through_own_tree(tree, legacy)
    assert r.returncode == 0, r.stderr
    broken = json.loads(summary.read_text())
    assert broken["controls"][EVALPLUS_TASK]["acc"] == 0.1              # the newer pass's value replaced
    assert C.SUMMARY_KEY_SOURCES not in broken                           # every source erased
    summary.write_bytes(sourced)
    # THE FIX: the legacy tree emits, this tree writes — and (PW12 / E19 A1-1) the legacy store's
    # top-level pass carries NO pass-order authority: a DIFFERING payload refuses, named, until the
    # operator says which store the campaign means (even "lower" — the integer is a claim)
    res = launcher_mod.merge_cell_summaries([str(legacy)], tree=tree, python=sys.executable)
    assert res[str(legacy)]["ok"] is False and "two sources claim one entry" in res[str(legacy)]["err"]
    assert "never orders passes" in res[str(legacy)]["err"] and summary.read_bytes() == sourced   # nothing written
    assert [r["cell_dir"] for r in sandbox_mod.unmerged_completed_stores(sweep)] == [str(legacy)]
    # ...naming the STANDING pass-2 store keeps it and records the legacy store superseded
    res = launcher_mod.merge_cell_summaries([str(legacy)], tree=tree, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={p2.name}")
    notes = res[str(legacy)].pop("notes")
    assert res == {str(legacy): {"ok": True, "merged": True, "path": str(summary)}}
    assert any("KEPT from" in n and C.REPLACE_SOURCE_FLAG in n for n in notes)
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.9                 # the newer payload survives
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p2, 2, _hist(legacy, 1))   # ...and the full block
    assert sandbox_mod.unmerged_completed_stores(sweep) == []           # the legacy store is accounted for
    settled = summary.read_bytes()
    res = launcher_mod.merge_cell_summaries([str(legacy), str(p2)], tree=tree, python=sys.executable)
    assert all(v["ok"] for v in res.values()) and summary.read_bytes() == settled   # idempotent sweep (settled)
    # PW14 (E22 M4): the settled decision holds for the PAYLOAD it was made on. The legacy store's summary
    # replaced (a mutable atomic write): the scan lists it pending again, the change named; the sweep REFUSES
    # rather than silently standing it down; naming the winner again re-records the decision with the new digest
    kept_digest = _digest(legacy)
    completion.write_task_summary(legacy, EVALPLUS_TASK, {**completion.read_task_summary(legacy, EVALPLUS_TASK), "acc": 0.15})
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    assert p["cell_dir"] == str(legacy) and "CHANGED" in p["why"][EVALPLUS_TASK] and kept_digest in p["why"][EVALPLUS_TASK]
    res = launcher_mod.merge_cell_summaries([str(legacy)], tree=tree, python=sys.executable)
    assert res[str(legacy)]["ok"] is False and "CHANGED since that decision" in res[str(legacy)]["err"]
    assert "two sources claim one entry" in res[str(legacy)]["err"] and summary.read_bytes() == settled
    res = launcher_mod.merge_cell_summaries([str(legacy)], tree=tree, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={p2.name}")
    assert res[str(legacy)]["ok"] is True
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.9
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p2, 2, _hist(legacy, 1))
    assert _digest(legacy) != kept_digest and sandbox_mod.unmerged_completed_stores(sweep) == []
    settled = summary.read_bytes()
    # a record WITHOUT a digest (a file from before PW14) accounts for nothing: pending, named; the sweep refuses
    # with the remedy; naming the winner re-records it with the digest and the store leaves the list
    d = json.loads(settled)
    d[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][C.MERGE_SOURCE_SUPERSEDED_FIELD][0].pop(C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD)
    summary.write_text(json.dumps(d, indent=1, sort_keys=True))
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    assert p["cell_dir"] == str(legacy) and "WITHOUT a payload digest" in p["why"][EVALPLUS_TASK]
    res = launcher_mod.merge_cell_summaries([str(legacy)], tree=tree, python=sys.executable)
    assert res[str(legacy)]["ok"] is False and "WITHOUT a payload digest" in res[str(legacy)]["err"]
    assert C.REPLACE_SOURCE_FLAG in res[str(legacy)]["err"]
    res = launcher_mod.merge_cell_summaries([str(legacy)], tree=tree, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={p2.name}")
    assert res[str(legacy)]["ok"] is True and summary.read_bytes() == settled
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    # a legacy store whose block records NO pass, against a sourced entry: ambiguous, refused, untouched
    hand = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd1",
                       chash="618f9dd1" + "0" * 8, pass_no=None, acc=0.3, hashed=False, top_pass=ABSENT,
                       tree_sha=LEGACY_SHA)
    res = launcher_mod.merge_cell_summaries([str(hand)], tree=tree, python=sys.executable)
    assert res[str(hand)]["ok"] is False and "two sources claim one entry" in res[str(hand)]["err"]
    assert summary.read_bytes() == settled
    # the emission never reaches for a writer: the producing tree's half has no write in it
    assert "merge_controls_summary" not in launcher_mod.EMIT_CODE
    assert "write_controls_summary" not in launcher_mod.EMIT_CODE
    assert "complete_task_summaries" in launcher_mod.EMIT_CODE and "slot_payloads" in launcher_mod.EMIT_CODE


def test_the_two_tree_merge_lands_a_legacy_store_under_the_discipline(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """E17 (A1), the realistic case — the two prompted 618f9dd stores: that
    tree's DRIVER wrote the summary's Inspect entries with no
    ``sources`` block, and its EvalPlus tasks finish later in the sandbox
    orchestrator. The two-tree merge lands the EvalPlus payload with a source
    whose pass is the store's top-level one, marked ``legacy_top_level``;
    the entries the same store already filled are ADOPTED (identical payload,
    nothing superseded) rather than refused entry by entry; the from-disk
    scan then accounts for the store; the consumer loads the file. A
    DIFFERENT payload over an unrecorded source is still refused until the
    operator names the winner; a producing tree that cannot emit is refused,
    named, and writes nothing."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    tree = _legacy_tree(tmp_path / "repos")
    W = launcher_mod.writer_tree_sha_of(REPO / "src")
    tiny = "inspect/math"
    store = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd0",
                        chash="618f9dd0" + "0" * 8, pass_no=1, acc=0.4, hashed=False,
                        tree_sha=LEGACY_SHA, task=tiny, n=100)
    # the legacy driver's own in-process merge: the entry in the file, no sources
    assert _merge_through_own_tree(tree, store).returncode == 0
    pre = json.loads(summary.read_text())
    assert set(pre["controls"]) == {tiny} and C.SUMMARY_KEY_SOURCES not in pre
    # later: the orchestrator's conversion completes EvalPlus in the SAME store
    completion.finish_task(store, task=EVALPLUS_TASK, n_items=2, n_expected_full=2, limits={}, n_rows=2,
                           n_requests=2, n_rejected=0, wall_clock_s=1.0,
                           summary=_evalplus_summary(EVALPLUS_TASK, acc=0.5, n=2))
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    assert p["tasks"] == [EVALPLUS_TASK] and p["source"] == _core(store, 1) and p["producer_tree_sha"] == LEGACY_SHA
    plan = sandbox_mod.plan_merges([p])
    assert plan["commands"][LEGACY_SHA][5:] == ["--sha", LEGACY_SHA, "--merge", "--merge-cell-dirs", str(store),
                                                "--app-name", sandbox_mod.MERGE_APP_NAME]
    res = launcher_mod.merge_cell_summaries([str(store)], tree=tree, python=sys.executable)
    notes = res[str(store)].pop("notes")
    assert res == {str(store): {"ok": True, "merged": True, "path": str(summary)}}
    assert any("ADOPTED" in n for n in notes)                             # auditable from the REC, not only the file
    doc = json.loads(summary.read_text())
    legacy = {C.MERGE_SOURCE_PASS_SOURCE_FIELD: C.PASS_SOURCE_LEGACY_TOP_LEVEL, C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD: W}
    legacy_src = _standing(store, 1, task=EVALPLUS_TASK, **legacy)         # each entry: + the digest it stands with (PW16)
    legacy_tiny = _standing(store, 1, task=tiny, **legacy)
    assert doc["controls"][tiny] == pre["controls"][tiny]                # the adopted number is unchanged
    assert doc[C.SUMMARY_KEY_SOURCES][tiny] == legacy_tiny                # ...and now names its store
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.5
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == legacy_src
    assert doc["provenance"] == pre["provenance"]
    assert sandbox_mod.unmerged_completed_stores(sweep) == []            # accounted for, by exact source
    before = summary.read_bytes()
    assert launcher_mod.merge_cell_summaries([str(store)], tree=tree, python=sys.executable)[str(store)]["ok"]
    assert summary.read_bytes() == before                               # idempotent
    loaded = load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY]
    assert loaded[EVALPLUS_TASK]["acc"] == 0.5 and loaded[tiny]["acc"] == 0.4
    # ...and, alongside, the per-entry source the file records (PW12); the retained EvalPlus
    # payload names no metric_def, so the loader stamps the OLD plus-only definition
    from sisteer.eval.cell_controls import load_controls_sources

    assert load_controls_sources(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY] == {tiny: legacy_tiny, EVALPLUS_TASK: legacy_src}
    assert loaded[EVALPLUS_TASK][C.EVALPLUS_METRIC_DEF_FIELD] == C.EVALPLUS_PLUS_DEF_PLUS_ONLY
    assert C.EVALPLUS_METRIC_DEF_FIELD not in loaded[tiny]
    # a pre-discipline file whose payload DIFFERS from the store's: the ambiguity it always was
    doc = dict(pre)
    doc["controls"] = {tiny: {**pre["controls"][tiny], "acc": 0.45}}
    summary.write_text(json.dumps(doc, indent=1, sort_keys=True))
    before = summary.read_bytes()
    res = launcher_mod.merge_cell_summaries([str(store)], tree=tree, python=sys.executable)
    assert res[str(store)]["ok"] is False and "two sources claim one entry" in res[str(store)]["err"]
    assert summary.read_bytes() == before
    # ...resolved by naming the store for THAT entry (the unrecorded side has no key to name)
    res = launcher_mod.merge_cell_summaries([str(store)], tree=tree, python=sys.executable,
                                            replace_source=f"{tiny}={store.name}")
    assert res[str(store)]["ok"] is True
    doc = json.loads(summary.read_text())
    assert doc["controls"][tiny]["acc"] == 0.4
    assert doc[C.SUMMARY_KEY_SOURCES][tiny] == {**legacy_tiny, C.MERGE_SOURCE_SUPERSEDED_FIELD: [
        {**dict(C.UNRECORDED_MERGE_SOURCE), C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD: C.merge_payload_digest(
            tiny, {**pre["controls"][tiny], "acc": 0.45})}]}
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == legacy_src
    # a producing tree that cannot emit is refused, named — nothing written either way
    before = summary.read_bytes()
    stub = _legacy_tree(tmp_path / "repos_stub", sha="f6cf462" + "f" * 33, completion=STUB_COMPLETION)
    res = launcher_mod.merge_cell_summaries([str(store)], tree=stub, python=sys.executable)
    assert res[str(store)]["ok"] is False and res[str(store)]["merged"] is False
    assert "could not emit" in res[str(store)]["err"] and "complete_task_summaries" in res[str(store)]["err"]
    none = _legacy_tree(tmp_path / "repos_none", sha="0000000" + "0" * 33, completion=NO_COMPLETION)
    res = launcher_mod.merge_cell_summaries([str(store)], tree=none, python=sys.executable)
    assert res[str(store)]["ok"] is False and "cannot emit" in res[str(store)]["err"]
    assert "tinycap.completion" in res[str(store)]["err"]
    assert summary.read_bytes() == before
    # the current tree's own recipe is the one the emission inlines
    own = completion.normalized_merge_payload(store)
    assert own == {"cell_dir": str(store), "payloads": {**completion.complete_task_summaries(store)}}
    assert set(own["payloads"]) == {tiny, EVALPLUS_TASK} and tuple(own) == C.MERGE_PAYLOAD_FIELDS
    # ...and the writer half refuses a payload that is not the contract's, or another store's
    with pytest.raises(completion.CompletionError, match="not the contract's"):
        completion.merge_emitted_payload(store, {"payloads": {}})
    with pytest.raises(completion.CompletionError, match="another directory"):
        completion.merge_emitted_payload(store, {"cell_dir": str(cell_dir), "payloads": {}})
    with pytest.raises(completion.CompletionError, match="counts-only payload mappings"):
        completion.merge_emitted_payload(store, {"cell_dir": str(store), "payloads": {tiny: 3}})
    assert completion.merge_emitted_payload(store, {"cell_dir": str(store), "payloads": {}}) is None
    assert summary.read_bytes() == before                               # the refusals above wrote nothing
    # the driver's own in-process merge stays strict about the hashed pass
    with pytest.raises(completion.CompletionError, match="carries no 'pass'"):
        completion.merge_controls_summary(store)


def test_a_legacy_top_level_pass_never_orders_passes(launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """PW12 (E19 A1-1) — the migration case the finding names: a pre-PW8
    store generated at pass 1 and resumed by the old driver at pass 2 carries
    ``geometry.pass: 2`` over pass-1 rows (the old ``write_cell_manifest``
    rewrote the whole manifest after checking only fixed resume keys), and
    ``merge_source(legacy_pass=True)`` records that integer. Under the
    pre-PW12 rule it would have REPLACED a correctly identified hashed pass-1
    entry. Now a ``legacy_top_level`` pass never orders: the claim is
    ambiguous (refused, named — both stores and the reason) until
    ``--replace-source`` names the winner; the from-disk scan still matches
    the store on the pass its own tree recorded."""
    _cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    tree = _legacy_tree(tmp_path / "repos")
    W = launcher_mod.writer_tree_sha_of(REPO / "src")
    p1 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11111111",
                     chash="11111111" + "0" * 8, pass_no=1, acc=0.3)
    completion.merge_controls_summary(p1)
    standing = summary.read_bytes()
    # pass-1 rows relabelled `geometry.pass: 2` by the old driver's resume rewrite; no hashed pass
    relabelled = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd2",
                             chash="618f9dd2" + "0" * 8, pass_no=2, acc=0.35, hashed=False,
                             tree_sha=LEGACY_SHA)
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    assert p["source"]["pass"] == 2                                     # the scan reads what its tree recorded...
    res = launcher_mod.merge_cell_summaries([str(relabelled)], tree=tree, python=sys.executable)
    assert res[str(relabelled)]["ok"] is False                          # ...the writer gives it no authority
    err = res[str(relabelled)]["err"]
    assert "two sources claim one entry" in err and "never orders passes" in err
    assert relabelled.name in err and p1.name in err and C.PASS_SOURCE_LEGACY_TOP_LEVEL in err
    assert summary.read_bytes() == standing                             # NOT replaced
    # the rule itself, both directions; the hashed reading of the same integers WOULD order
    from sisteer.eval.cell_controls import _pass_decision

    hashed1 = _core(p1, 1)
    legacy2 = {**_core(relabelled, 2), C.MERGE_SOURCE_PASS_SOURCE_FIELD: C.PASS_SOURCE_LEGACY_TOP_LEVEL}
    assert _pass_decision(hashed1, legacy2) == "ambiguous" == _pass_decision(legacy2, hashed1)
    assert _pass_decision(hashed1, _core(relabelled, 2)) == "replace"
    assert _pass_decision({**hashed1, "pass": 2}, {**legacy2, "pass": 1}) == "ambiguous"   # a legacy 1 vs a hashed 2
    assert _pass_decision(legacy2, dict(legacy2)) == "same"
    # the operator names the STANDING store: kept, the relabelled store recorded superseded, accounted
    res = launcher_mod.merge_cell_summaries([str(relabelled)], tree=tree, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={p1.name}")
    assert res[str(relabelled)]["ok"] is True
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.3
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p1, 1, _hist(relabelled, 2))
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    # ...or the RELABELLED store: taken, recorded, marked as the weaker reading, written by this tree
    summary.write_bytes(standing)
    res = launcher_mod.merge_cell_summaries([str(relabelled)], tree=tree, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={relabelled.name}")
    assert res[str(relabelled)]["ok"] is True
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.35
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(
        relabelled, 2, _hist(p1, 1),
        **{C.MERGE_SOURCE_PASS_SOURCE_FIELD: C.PASS_SOURCE_LEGACY_TOP_LEVEL,
           C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD: W})
    # the launcher's served_geometry.json beside the store is no better witness: it is rewritten
    # into the same out_dir by every run of the configuration (a resume, a pass 2)
    assert "(out_dir / SERVED_GEOMETRY_FILENAME).write_text(" in (REPO / "scripts/modal/tinycap_served.py").read_text()


def _reclass_inapplicable(store: Path) -> None:
    """Turn a stored 'compared' cell into a DOCUMENT-fit ('inapplicable') one
    the way the driver would have written it: stated class, hashed copy (with
    the identity hash recomputed), witness and block agreeing."""
    from sisteer.capeval import store as cs
    from sisteer.eval import sweep as _sweep
    mp = store / "manifest.json"
    m = json.loads(mp.read_text())
    m[_sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY] = "inapplicable"
    m["resolved_config"][_sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY] = "inapplicable"
    m["harness_config_hash"] = cs.harness_config_hash(m["resolved_config"])
    w = dict(m[_sweep.MANIFEST_KEY_FIT_WITNESS])
    w.update({"status": _sweep.DOCUMENT_FIT_STATUS, "template_hash": "e3b0c44298fc1c14",
              "fit_kind": _sweep.FIT_KIND_DOCUMENT, "training_budget_kind": "document_tokens",
              "chat_template_bypassed": True, "positions_at_fit": "all"})
    m[_sweep.MANIFEST_KEY_FIT_WITNESS] = w
    m[_sweep.MANIFEST_KEY_FIT_PROVENANCE] = {
        **m[_sweep.MANIFEST_KEY_FIT_PROVENANCE], **w, "template_check": "inapplicable",
        "template_check_reason": _sweep.TEMPLATE_CHECK_REASON["inapplicable"]}
    assert _sweep.store_provenance_problems(m) == []
    mp.write_text(json.dumps(m))


def test_a_mixed_class_equal_payload_is_contested_in_both_orders_end_to_end(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """A compared store and an inapplicable store of one cell offer
    identical payloads over an UNRECORDED standing entry. The plan left both
    unplanned, an empty plan does not prohibit adoption, and the first store
    written adopted — input order picked a provenance winner and the summary
    was modified in both orders. Now the entry is CONTESTED on every
    candidate's plan and the writer refuses the whole store write until the
    winner is named: through the launcher's two-tree subprocess merge, in
    BOTH orders, the canonical summary is byte-identical after the refusal;
    naming the compared store makes it adopt in either order, the other
    yields."""
    _cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    tiny = "inspect/math"
    a = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__c0000001", chash="c0000001" + "0" * 8,
                    pass_no=1, acc=0.4, task=tiny, n=100)
    b = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__c0000002", chash="c0000002" + "0" * 8,
                    pass_no=1, acc=0.4, task=tiny, n=100)
    _reclass_inapplicable(b)
    assert completion.read_task_summary(a, tiny) == completion.read_task_summary(b, tiny)
    pre = {"cell_id": C.CELL_PSRKL_SAFETY, "controls": {tiny: completion.read_task_summary(a, tiny)},
           "provenance": base[C.MANIFEST_KEY_PROVENANCE]}
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(pre, indent=1, sort_keys=True))
    before = summary.read_bytes()
    for order in ((a, b), (b, a)):
        res = launcher_mod.merge_cell_summaries([str(d) for d in order], tree=REPO, python=sys.executable)
        assert all(v["ok"] is False and "CONTESTED" in v["err"] for v in res.values()), res
        assert summary.read_bytes() == before                      # byte-identical, in this order too
    # the operator names the winner: the named store adopts, the other yields, either order
    for order in ((a, b), (b, a)):
        summary.write_text(before.decode())
        res = launcher_mod.merge_cell_summaries([str(d) for d in order], tree=REPO, python=sys.executable,
                                                replace_source=f"{tiny}={a.name}")
        assert all(v["ok"] for v in res.values()), res
        doc = json.loads(summary.read_text())
        assert doc[C.SUMMARY_KEY_SOURCES][tiny]["cell_key"] == a.name
        assert doc[C.SUMMARY_KEY_SOURCES][tiny][C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD] == "compared"
        assert C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD not in doc[C.SUMMARY_KEY_SOURCES][tiny]
        assert doc["controls"][tiny] == pre["controls"][tiny]     # no number moved
    # naming a store that is not a candidate refuses, nothing written
    summary.write_text(before.decode())
    res = launcher_mod.merge_cell_summaries([str(a), str(b)], tree=REPO, python=sys.executable,
                                            replace_source=f"{tiny}=tok__controls__t0__a0__aaaaaaaa__ffffffff")
    assert all(v["ok"] is False for v in res.values()) and summary.read_bytes() == before


def test_equal_witnesses_are_attributed_by_the_recorded_rule_not_directory_order(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """PW12 (E19, PW10-3 constraint ii): SEVERAL stores of one merge plan may
    exactly witness an UNRECORDED standing payload (two 618f9dd-style stores
    of one cell whose drivers both wrote the same number). Before
    PW12 the first one written adopted and the attribution stuck even when a
    later equal candidate failed its overall merge. Now
    ``completion.equal_witness_plan`` runs over EVERY store of the plan
    BEFORE the first write and applies ONE recorded deterministic rule
    (``contracts.EQUAL_WITNESS_RULE``: a hashed-pass source before a legacy
    one, then the lexically smallest cell key); the others are recorded
    beside the adopter as ``equal_witnesses`` — the same number, never a
    supersession — so the file ends the same in every order, a witness
    re-offers nothing, and naming a witness for the entry refuses."""
    from sisteer.eval.cell_controls import load_controls_sources, load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    tree = _legacy_tree(tmp_path / "repos")
    W = launcher_mod.writer_tree_sha_of(REPO / "src")
    tiny = "inspect/math"
    a = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd0", chash="618f9dd0" + "0" * 8,
                    pass_no=1, acc=0.4, hashed=False, tree_sha=LEGACY_SHA, task=tiny, n=100)
    _pass_store(sweep, base, key=a.name, chash="618f9dd0" + "0" * 8, pass_no=1, acc=0.5, hashed=False,
                tree_sha=LEGACY_SHA)                                                    # + A's EvalPlus task
    b = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd1", chash="618f9dd1" + "0" * 8,
                    pass_no=1, acc=0.4, hashed=False, tree_sha=LEGACY_SHA, task=tiny, n=100)
    _pass_store(sweep, base, key=b.name, chash="618f9dd1" + "0" * 8, pass_no=1, acc=0.6, hashed=False,
                tree_sha=LEGACY_SHA, task=MBPP_TASK)                                   # + B's (other) EvalPlus task
    assert completion.read_task_summary(a, tiny) == completion.read_task_summary(b, tiny)
    pre = {"cell_id": C.CELL_PSRKL_SAFETY, "controls": {tiny: completion.read_task_summary(a, tiny)},
           "provenance": base[C.MANIFEST_KEY_PROVENANCE]}
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(pre, indent=1, sort_keys=True))
    legacy = {C.MERGE_SOURCE_PASS_SOURCE_FIELD: C.PASS_SOURCE_LEGACY_TOP_LEVEL, C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD: W}
    # the plan, before any write, over the stores in the WRONG order: both witness; the rule picks A
    plan = completion.equal_witness_plan([(str(d), completion.normalized_merge_payload(d)) for d in (b, a)])
    assert plan == {str(b): {"witnesses": {}, "yield": [tiny]},
                    str(a): {"witnesses": {tiny: [{**_core(b, 1),
                                                   C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD: _template_class(b)}]},
                             "yield": []}}
    # the launcher's merge, B first: the file still ends with A adopting and B recorded beside it
    res = launcher_mod.merge_cell_summaries([str(b), str(a)], tree=tree, python=sys.executable)
    assert all(v["ok"] and v["merged"] for v in res.values())
    doc = json.loads(summary.read_text())
    assert doc["controls"][tiny] == pre["controls"][tiny]
    assert doc[C.SUMMARY_KEY_SOURCES][tiny] == _standing(
        a, 1, task=tiny, **legacy, **{C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD: [_hist(b, 1, tiny)],
                                      C.MERGE_SOURCE_EQUAL_WITNESS_RULE_FIELD: C.EQUAL_WITNESS_RULE})
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(a, 1, **legacy)
    assert doc[C.SUMMARY_KEY_SOURCES][MBPP_TASK] == _standing(b, 1, task=MBPP_TASK, **legacy)
    settled = summary.read_bytes()
    # idempotent alone or together, in either order: a recorded witness re-offers nothing
    for dirs in ([str(a), str(b)], [str(b)], [str(a)], [str(b), str(a)]):
        res = launcher_mod.merge_cell_summaries(dirs, tree=tree, python=sys.executable)
        assert all(v["ok"] for v in res.values()) and summary.read_bytes() == settled, dirs
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    # naming the witness refuses (the attribution follows the recorded rule); naming the adopter agrees
    res = launcher_mod.merge_cell_summaries([str(b)], tree=tree, python=sys.executable,
                                            replace_source=f"{tiny}={b.name}")
    assert res[str(b)]["ok"] is False and "EQUAL WITNESS" in res[str(b)]["err"] and summary.read_bytes() == settled
    res = launcher_mod.merge_cell_summaries([str(b)], tree=tree, python=sys.executable,
                                            replace_source=f"{tiny}={a.name}")
    assert res[str(b)]["ok"] is True and summary.read_bytes() == settled
    # the consumer: the payload, and alongside it the source with its witnesses
    loaded = load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY]
    sources = load_controls_sources(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY]
    assert loaded[tiny]["acc"] == 0.4 and set(sources) == {tiny, EVALPLUS_TASK, MBPP_TASK}
    assert sources[tiny] == doc[C.SUMMARY_KEY_SOURCES][tiny]
    # a HASHED-pass witness outranks the legacy ones, whatever its key sorts as
    summary.write_text(json.dumps(pre, indent=1, sort_keys=True))
    c = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__ffffffff", chash="ffffffff" + "0" * 8,
                    pass_no=1, acc=0.4, task=tiny, n=100)
    plan = completion.equal_witness_plan([(str(d), completion.normalized_merge_payload(d)) for d in (a, b, c)])
    assert plan[str(c)] == {"witnesses": {tiny: [{**_core(a, 1), C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD: _template_class(a)},
                                                 {**_core(b, 1), C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD: _template_class(b)}]},
                            "yield": []}
    assert plan[str(a)]["yield"] == [tiny] and plan[str(b)]["yield"] == [tiny]
    res = launcher_mod.merge_cell_summaries([str(a), str(b), str(c)], tree=tree, python=sys.executable)
    assert all(v["ok"] for v in res.values())
    doc = json.loads(summary.read_text())
    assert doc[C.SUMMARY_KEY_SOURCES][tiny] == _standing(
        c, 1, task=tiny, **{C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD: W,
                            C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD: [_hist(a, 1, tiny), _hist(b, 1, tiny)],
                            C.MERGE_SOURCE_EQUAL_WITNESS_RULE_FIELD: C.EQUAL_WITNESS_RULE})
    # a single witness, or a recorded standing source, or a differing payload: nothing is planned
    summary.write_text(json.dumps(pre, indent=1, sort_keys=True))
    assert completion.equal_witness_plan([(str(a), completion.normalized_merge_payload(a))]) == {
        str(a): {"witnesses": {}, "yield": []}}
    d_ = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__dddddddd", chash="dddddddd" + "0" * 8,
                     pass_no=1, acc=0.45, task=tiny, n=100)
    plan = completion.equal_witness_plan([(str(x), completion.normalized_merge_payload(x)) for x in (a, d_)])
    assert plan[str(a)] == plan[str(d_)] == {"witnesses": {}, "yield": []}
    launcher_mod.merge_cell_summaries([str(a)], tree=tree, python=sys.executable)         # A adopts, sourced
    plan = completion.equal_witness_plan([(str(x), completion.normalized_merge_payload(x)) for x in (a, b)])
    assert plan[str(a)] == plan[str(b)] == {"witnesses": {}, "yield": []}                # recorded: the pass rule's turn


def test_an_equal_witness_is_accounted_for_by_the_scan(launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """PW13 (E21 M3): ``unaccounted_reason`` accepted the standing source and
    the ``superseded`` list but never ``equal_witnesses`` — so the store
    that exactly witnessed an adopted payload and yielded it under the
    recorded rule (PW12) read as PENDING on every scan, and ``--merge-mode
    merge`` kept rescheduling its producer tree. Membership in
    ``equal_witnesses`` — the same four-field comparison as ``superseded`` —
    now counts as accounted. The equal entry here is an EvalPlus task: the
    scan checks only those (the PW12 test's non-EvalPlus entry was
    never scanned, which is how the gap hid)."""
    _cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    tree = _legacy_tree(tmp_path / "repos")
    a = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd0", chash="618f9dd0" + "0" * 8,
                    pass_no=1, acc=0.4, hashed=False, tree_sha=LEGACY_SHA)
    b = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd1", chash="618f9dd1" + "0" * 8,
                    pass_no=1, acc=0.4, hashed=False, tree_sha=LEGACY_SHA)
    assert completion.read_task_summary(a, EVALPLUS_TASK) == completion.read_task_summary(b, EVALPLUS_TASK)
    pre = {"cell_id": C.CELL_PSRKL_SAFETY, "controls": {EVALPLUS_TASK: completion.read_task_summary(a, EVALPLUS_TASK)},
           "provenance": base[C.MANIFEST_KEY_PROVENANCE]}
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(pre, indent=1, sort_keys=True))
    # before the merge both are pending: an unrecorded entry satisfies no concrete store
    assert sorted(r["cell_dir"] for r in sandbox_mod.unmerged_completed_stores(sweep)) == sorted([str(a), str(b)])
    res = launcher_mod.merge_cell_summaries([str(b), str(a)], tree=tree, python=sys.executable)
    assert all(v["ok"] for v in res.values()), res
    doc = json.loads(summary.read_text())
    rec = doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK]
    assert rec["cell_key"] == a.name and rec[C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD] == [_hist(b, 1)]
    assert doc["controls"][EVALPLUS_TASK][C.EVALPLUS_METRIC_DEF_FIELD] == C.EVALPLUS_PLUS_DEF_PLUS_ONLY   # stamped by the writer (PW13)
    # the finding: B exactly witnessed the payload and yielded it to A under the recorded rule — ACCOUNTED, so the
    # scan lists nothing and the planner reschedules no producer tree
    pending = sandbox_mod.unmerged_completed_stores(sweep)
    assert pending == [] and sandbox_mod.plan_merges(pending)["commands"] == {}
    src_b, bad = sandbox_mod.store_merge_source(b, json.loads((b / "manifest.json").read_text()))
    db = _digest(b)                                                                            # the store's CURRENT payload (PW14)
    assert bad is None and sandbox_mod.unaccounted_reason(src_b, rec, payload_digest=db) is None
    # membership is by the four identity fields, exactly as `superseded`: a witness under another hash is not this store
    other = {**_hist(b, 1), "harness_config_hash": "ffffffff" + "0" * 8}
    reason = sandbox_mod.unaccounted_reason(src_b, {**rec, C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD: [other]}, payload_digest=db)
    assert reason is not None and "not from this store" in reason
    src_a, _ = sandbox_mod.store_merge_source(a, json.loads((a / "manifest.json").read_text()))
    da = _digest(a)                                                                            # the standing store: at the
    assert sandbox_mod.unaccounted_reason(src_a, rec, payload_digest=da, standing_digest=da) is None   # canonical payload (PW16)
    assert "not supplied" in sandbox_mod.unaccounted_reason(src_a, rec)                        # ...both digests required
    assert sandbox_mod.unaccounted_reason(src_b, {**rec, C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD: [other],
                                                  C.MERGE_SOURCE_SUPERSEDED_FIELD: [_hist(b, 1)]},
                                          payload_digest=db) is None                           # superseded, as before
    # PW14 (E22 M4): the record accounts for the store only AT THE PAYLOAD the decision was made on
    changed = C.merge_payload_digest(EVALPLUS_TASK, {**completion.read_task_summary(b, EVALPLUS_TASK), "acc": 0.1})
    reason = sandbox_mod.unaccounted_reason(src_b, rec, payload_digest=changed)
    assert reason is not None and "CHANGED" in reason and db in reason and changed in reason
    reason = sandbox_mod.unaccounted_reason(src_b, {**rec, C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD: [_core(b, 1)]}, payload_digest=db)
    assert reason is not None and "WITHOUT a payload digest" in reason                        # a pre-PW14 record
    reason = sandbox_mod.unaccounted_reason(src_b, {**rec, C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD: [],
                                                    C.MERGE_SOURCE_SUPERSEDED_FIELD: [_core(b, 1)]}, payload_digest=db)
    assert reason is not None and "WITHOUT a payload digest" in reason and C.REPLACE_SOURCE_FLAG in reason
    reason = sandbox_mod.unaccounted_reason(src_b, rec)                                       # the scan must supply it
    assert reason is not None and "not supplied" in reason
    # idempotent: a re-merge of the witness alone changes nothing and the scan still lists nothing
    settled = summary.read_bytes()
    res = launcher_mod.merge_cell_summaries([str(b)], tree=tree, python=sys.executable)
    assert res[str(b)]["ok"] and summary.read_bytes() == settled and sandbox_mod.unmerged_completed_stores(sweep) == []
    # PW14 (E22 M4), on disk: a witness record WITHOUT a digest (a file from before PW14) accounts for nothing —
    # pending, named — and the witness's next EQUAL re-offer records the digest, after which the scan is quiet
    d = json.loads(settled)
    d[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD][0].pop(C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD)
    summary.write_text(json.dumps(d, indent=1, sort_keys=True))
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    assert p["cell_dir"] == str(b) and "WITHOUT a payload digest" in p["why"][EVALPLUS_TASK]
    res = launcher_mod.merge_cell_summaries([str(b)], tree=tree, python=sys.executable)
    assert res[str(b)]["ok"] and summary.read_bytes() == settled and sandbox_mod.unmerged_completed_stores(sweep) == []
    # ...and a witness whose summary CHANGED (0.4 -> 0.1) is pending again, the change named, and the merge refuses
    # it (the store changed under a recorded attribution, PW12) — the changed claim resurfaces instead of vanishing
    completion.write_task_summary(b, EVALPLUS_TASK, {**completion.read_task_summary(b, EVALPLUS_TASK), "acc": 0.1})
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    assert p["cell_dir"] == str(b) and "CHANGED" in p["why"][EVALPLUS_TASK] and db in p["why"][EVALPLUS_TASK]
    assert sandbox_mod.plan_merges([p])["commands"] != {}                     # ...and is scheduled again
    res = launcher_mod.merge_cell_summaries([str(b)], tree=tree, python=sys.executable)
    assert res[str(b)]["ok"] is False and "DIFFERENT payload" in res[str(b)]["err"] and summary.read_bytes() == settled


def test_the_standing_source_accounts_for_a_store_only_while_it_holds_the_canonical_payload(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """PW16 (E24), on disk: a store IS the recorded standing source of its
    entry (all four fields), and its ``summary/<task>.json`` is then
    atomically rewritten with different numbers. Before this the from-disk
    scan returned accounted on the four-field match BEFORE consulting any
    digest (the store never read pending) and the writer's ``same`` branch
    overlaid the canonical value silently — a manual re-merge changed a
    scientific result with no trace. Now the scan lists the store PENDING
    with both digests and the remedy; the sweep REFUSES (the launcher's
    two-tree merge and the driver's in-process merge alike), nothing written;
    naming the standing store ITSELF refreshes the entry — the new canonical
    value, the standing digest moved, a ``refreshed`` record with both
    digests and the writer's tree, the REC's ``notes`` saying ``REFRESHED``;
    an unchanged re-offer stays the byte-identical no-op; and a hand-edited
    canonical value under the standing record reads pending, refuses to load,
    and is re-filed from the store by the same named refresh."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    W = launcher_mod.writer_tree_sha_of(REPO / "src")
    D, REF, WAS, WT = (C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD, C.MERGE_SOURCE_REFRESHED_FIELD,
                       C.MERGE_REFRESH_DIGEST_WAS_FIELD, C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD)
    p1 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11111111",
                     chash="11111111" + "0" * 8, pass_no=1, acc=0.25)
    # merged through the launcher's two-tree path (this checkout as the producing tree): the standing record
    # carries the digest of the payload it stands with, which IS the store's summary — accounted
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and res[str(p1)]["merged"]
    doc = json.loads(summary.read_text())
    d1 = _digest(p1)
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p1, 1, **{WT: W})
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][D] == d1 == C.merge_payload_digest(
        EVALPLUS_TASK, doc["controls"][EVALPLUS_TASK])
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    settled = summary.read_bytes()
    # the unchanged re-offer: the settled no-op — bytes unchanged, nothing refreshed
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and not any("REFRESHED" in n for n in res[str(p1)]["notes"])
    assert summary.read_bytes() == settled
    # THE FINDING: the standing store's OWN summary rewritten under the identical four-field source
    completion.write_task_summary(p1, EVALPLUS_TASK, {**completion.read_task_summary(p1, EVALPLUS_TASK), "acc": 0.3})
    d2 = _digest(p1)
    assert d2 != d1
    # (a) the scan: PENDING, both digests, the standing source's own summary named, the remedy — and rescheduled
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    why = p["why"][EVALPLUS_TASK]
    assert p["cell_dir"] == str(p1) and "OWN summary changed" in why and d1 in why and d2 in why
    assert f"{C.REPLACE_SOURCE_FLAG} <entry>{C.REPLACE_SOURCE_ASSIGN}{p1.name}" in why
    assert sandbox_mod.plan_merges([p])["commands"] != {}
    src1, bad = sandbox_mod.store_merge_source(p1, json.loads((p1 / "manifest.json").read_text()))
    rec = json.loads(summary.read_text())[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK]
    assert bad is None
    assert sandbox_mod.unaccounted_reason(src1, rec, payload_digest=d1, standing_digest=d1) is None
    assert "OWN summary changed" in sandbox_mod.unaccounted_reason(src1, rec, payload_digest=d2, standing_digest=d1)
    assert "not supplied" in sandbox_mod.unaccounted_reason(src1, rec, payload_digest=d2)      # the scan supplies BOTH
    assert "not supplied" in sandbox_mod.unaccounted_reason(src1, rec, standing_digest=d1)
    assert "hand-edited" in sandbox_mod.unaccounted_reason(src1, rec, payload_digest=d2, standing_digest=d2)  # record d1
    # (b) the writer: the sweep REFUSES — both digests and the remedy named, nothing written — through the launcher's
    # two-tree merge (recorded, not raised) and the driver's in-process merge alike; naming another store refuses too
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    err = res[str(p1)]["err"]
    assert res[str(p1)]["ok"] is False and "never refreshes a standing entry silently" in err
    assert d1 in err and d2 in err and f"{C.REPLACE_SOURCE_FLAG} {EVALPLUS_TASK}{C.REPLACE_SOURCE_ASSIGN}{p1.name}" in err
    assert summary.read_bytes() == settled
    with pytest.raises(ValueError, match="never refreshes a standing entry silently"):
        completion.merge_controls_summary(p1)
    assert summary.read_bytes() == settled
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}=somebody_else")
    assert res[str(p1)]["ok"] is False and "IS the standing source" in res[str(p1)]["err"]
    assert summary.read_bytes() == settled
    # the operator names the STANDING store itself: REFRESHED — the store's current numbers are canonical, the
    # standing digest moved, a `refreshed` record with both digests and THIS writer's tree, and the REC says so
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={p1.name}")
    notes = res[str(p1)].pop("notes")
    assert res[str(p1)] == {"ok": True, "merged": True, "path": str(summary)}
    assert any("REFRESHED" in n and d1 in n and d2 in n and W in n for n in notes)
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.3
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p1, 1, **{WT: W, REF: [{WAS: d1, D: d2, WT: W}]})
    assert sandbox_mod.unmerged_completed_stores(sweep) == []                       # accounted again
    settled = summary.read_bytes()
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)   # idempotent after the refresh
    assert res[str(p1)]["ok"] and summary.read_bytes() == settled
    assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][EVALPLUS_TASK]["acc"] == 0.3
    # a hand-edited CANONICAL value under the standing record: pending (named), the loader refuses, and the same
    # named refresh re-files the entry from the store
    d = json.loads(settled)
    d["controls"][EVALPLUS_TASK]["acc"] = 0.99
    summary.write_text(json.dumps(d, indent=1, sort_keys=True))
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    assert "hand-edited" in p["why"][EVALPLUS_TASK] and d2 in p["why"][EVALPLUS_TASK]
    with pytest.raises(ValueError, match="hand-edited"):
        load_controls_summaries(sweep, [cell], cfg)
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] is False and "never refreshes a standing entry silently" in res[str(p1)]["err"]
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={p1.name}")
    assert res[str(p1)]["ok"] is True
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.3
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][REF] == [
        {WAS: d1, D: d2, WT: W},
        {WAS: C.merge_payload_digest(EVALPLUS_TASK, d["controls"][EVALPLUS_TASK]), D: d2, WT: W}]
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][EVALPLUS_TASK]["acc"] == 0.3


def test_a_stale_standing_digest_reads_pending_and_is_repaired_by_the_named_merge(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """PW19 (E27), on disk: a store IS the recorded standing source of its
    entry and a PW16 file records d1 for it; its ``summary/<task>.json`` is
    then rewritten AND the canonical payload brought to the new value WITHOUT
    the record — what a writer from before PW16 leaves (it overlays a
    same-source payload and preserves the record fields it does not know), and
    what a hand edit copying the store's summary leaves. Canonical and store
    both digest d2 under record d1. The scan reads the store PENDING as
    hand-edited with both digests and the remedy and reschedules it; the loader
    refuses; the unnamed sweep refuses (the launcher's two-tree merge and the
    driver's in-process merge alike), nothing written — and, THE FIX, the
    remedy both name now works: the named launcher merge brings the record to
    the canonical payload with a ``refreshed`` record ``{d1, d2, this writer's
    tree}`` and the REC's ``notes`` saying ``REFRESHED``, after which the store
    reads accounted, the loader loads, and a re-merge (named or not) is the
    byte-identical no-op. Before PW19 the writer's ``same`` branch exited
    settled on canonical == offered and the stamp refused the named repair. A
    record WITHOUT a digest (pre-PW16) reads accounted while the store holds the
    canonical payload and is backfilled by the next merge."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    W = launcher_mod.writer_tree_sha_of(REPO / "src")
    D, REF, WAS, WT = (C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD, C.MERGE_SOURCE_REFRESHED_FIELD,
                       C.MERGE_REFRESH_DIGEST_WAS_FIELD, C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD)
    p1 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11111111",
                     chash="11111111" + "0" * 8, pass_no=1, acc=0.25)
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and res[str(p1)]["merged"] and sandbox_mod.unmerged_completed_stores(sweep) == []
    d1 = _digest(p1)
    assert json.loads(summary.read_text())[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][D] == d1
    # the store's summary rewritten (0.25 -> 0.3) AND the canonical payload landed to it without the record
    completion.write_task_summary(p1, EVALPLUS_TASK, {**completion.read_task_summary(p1, EVALPLUS_TASK), "acc": 0.3})
    d2 = _digest(p1)
    assert d2 != d1
    d = json.loads(summary.read_text())
    d["controls"][EVALPLUS_TASK] = C.with_evalplus_metric_def(EVALPLUS_TASK, completion.read_task_summary(p1, EVALPLUS_TASK))
    summary.write_text(json.dumps(d, indent=1, sort_keys=True))
    stale = summary.read_bytes()
    assert C.merge_payload_digest(EVALPLUS_TASK, d["controls"][EVALPLUS_TASK]) == d2
    assert d[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][D] == d1
    # (a) the scan: PENDING as hand-edited, both digests, the remedy — and rescheduled
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    why = p["why"][EVALPLUS_TASK]
    assert p["cell_dir"] == str(p1) and "hand-edited" in why and d1 in why and d2 in why
    assert f"{C.REPLACE_SOURCE_FLAG} <entry>{C.REPLACE_SOURCE_ASSIGN}{p1.name}" in why
    assert sandbox_mod.plan_merges([p])["commands"] != {}
    src1, bad = sandbox_mod.store_merge_source(p1, json.loads((p1 / "manifest.json").read_text()))
    rec = d[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK]
    assert bad is None and "hand-edited" in sandbox_mod.unaccounted_reason(src1, rec, payload_digest=d2, standing_digest=d2)
    with pytest.raises(ValueError, match="hand-edited"):
        load_controls_summaries(sweep, [cell], cfg)
    # (b) the unnamed sweep REFUSES, the stale record and a working remedy named, nothing written — through the
    # launcher's two-tree merge (recorded, not raised) and the driver's in-process merge alike
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    err = res[str(p1)]["err"]
    assert res[str(p1)]["ok"] is False and "record is STALE" in err and d1 in err and d2 in err
    assert f"{C.REPLACE_SOURCE_FLAG} {EVALPLUS_TASK}{C.REPLACE_SOURCE_ASSIGN}{p1.name}" in err
    assert summary.read_bytes() == stale
    with pytest.raises(ValueError, match="record is STALE"):
        completion.merge_controls_summary(p1)
    assert summary.read_bytes() == stale
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}=somebody_else")
    assert res[str(p1)]["ok"] is False and "IS the standing source" in res[str(p1)]["err"]
    assert summary.read_bytes() == stale
    # (c) THE FIX: the named launcher merge repairs the record — the canonical value unchanged, the digest moved, a
    # `refreshed` record with both digests and THIS writer's tree, the REC's notes saying REFRESHED
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable,
                                            replace_source=f"{EVALPLUS_TASK}={p1.name}")
    notes = res[str(p1)].pop("notes")
    assert res[str(p1)] == {"ok": True, "merged": True, "path": str(summary)}
    assert any("REFRESHED" in n and d1 in n and d2 in n and W in n and "does not move" in n for n in notes)
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK] == d["controls"][EVALPLUS_TASK]                  # the value did not move
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p1, 1, **{WT: W, REF: [{WAS: d1, D: d2, WT: W}]})
    # (d) accounted afterwards, loadable, and idempotent named or not
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][EVALPLUS_TASK]["acc"] == 0.3
    repaired = summary.read_bytes()
    for flag in ("", f"{EVALPLUS_TASK}={p1.name}"):
        res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable, replace_source=flag)
        assert res[str(p1)]["ok"] and not any("REFRESHED" in n for n in res[str(p1)]["notes"])
        assert summary.read_bytes() == repaired
    # a record WITHOUT a digest (a file from before PW16): accounted while the store holds the canonical payload,
    # and the next merge backfills it, said — no refresh record
    d = json.loads(repaired)
    for k in (D, REF):
        d[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK].pop(k)
    summary.write_text(json.dumps(d, indent=1, sort_keys=True))
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and any("now carry the digest" in n for n in res[str(p1)]["notes"])
    doc = json.loads(summary.read_text())
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(p1, 1, **{WT: W}) and REF not in doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK]
    assert sandbox_mod.unmerged_completed_stores(sweep) == []


def test_an_explicit_null_standing_digest_reads_pending_and_is_never_backfilled(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """PW19b (Evals round 10), on disk: a standing record whose ``payload_digest``
    is an EXPLICIT ``null`` (or any non-digest value) read as ACCOUNTED to the
    scan (``.get()`` → None → the four-field match at the canonical payload) and
    as the legacy absence to the writer (a merge did not raise and announced a
    backfill), while the loader refused the file — so the scan suppressed the
    very repair a loadable summary needs. Now the scan reads such a store
    PENDING (named value, the remedy) and reschedules it; the loader refuses;
    the launcher's two-tree merge and the driver's in-process merge refuse,
    unnamed and named, nothing written and no backfill in the REC's ``notes``;
    the file's last good write restores it; and the ABSENT key (pre-PW16) still
    reads accounted and is backfilled by the next merge, said (PW19)."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    D = C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD
    p1 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11111111",
                     chash="11111111" + "0" * 8, pass_no=1, acc=0.25)
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and res[str(p1)]["merged"] and sandbox_mod.unmerged_completed_stores(sweep) == []
    good = summary.read_bytes()
    d1 = _digest(p1)
    src1, bad = sandbox_mod.store_merge_source(p1, json.loads((p1 / "manifest.json").read_text()))
    assert bad is None
    for value in (None, "xyz"):
        d = json.loads(good)
        d[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][D] = value
        summary.write_text(json.dumps(d, indent=1, sort_keys=True))
        broken = summary.read_bytes()
        rec = d[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK]
        # (a) the scan: PENDING — never accounted, even at the canonical payload — the value and the remedy named,
        # and rescheduled
        [p] = sandbox_mod.unmerged_completed_stores(sweep)
        why = p["why"][EVALPLUS_TASK]
        assert p["cell_dir"] == str(p1) and repr(value) in why and "explicit null or any other value" in why
        assert "last good write" in why and "PW19b" in why
        assert sandbox_mod.plan_merges([p])["commands"] != {}
        reason = sandbox_mod.unaccounted_reason(src1, rec, payload_digest=d1, standing_digest=d1)
        assert reason is not None and repr(value) in reason
        # (b) the loader refuses
        with pytest.raises(ValueError, match="16 hex"):
            load_controls_summaries(sweep, [cell], cfg)
        # (c) the merge refuses — the launcher's two-tree path unnamed and named (recorded, not raised; no backfill
        # in the notes) and the driver's in-process merge — nothing written
        for flag in ("", f"{EVALPLUS_TASK}={p1.name}"):
            res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable, replace_source=flag)
            assert res[str(p1)]["ok"] is False and "explicit null or any other value" in res[str(p1)]["err"]
            assert repr(value) in res[str(p1)]["err"]
            assert not any("now carry the digest" in n or "REFRESHED" in n for n in res[str(p1)]["notes"])
            assert summary.read_bytes() == broken
        with pytest.raises(ValueError, match="explicit null or any other value"):
            completion.merge_controls_summary(p1)
        assert summary.read_bytes() == broken
    # the remedy: the file's last good write — accounted and loadable again
    summary.write_bytes(good)
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][EVALPLUS_TASK]["acc"] == 0.25
    # the contrast that defines "absent": the KEY missing (pre-PW16) reads accounted and the next merge backfills it
    d = json.loads(good)
    d[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK].pop(D)
    summary.write_text(json.dumps(d, indent=1, sort_keys=True))
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and any("now carry the digest" in n for n in res[str(p1)]["notes"])
    assert json.loads(summary.read_text()) == json.loads(good) and sandbox_mod.unmerged_completed_stores(sweep) == []


def test_an_explicit_null_history_digest_reads_pending_and_is_never_re_recorded(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """PW19c (Evals round 11), on disk — the history twin of the PW19b test
    above: a ``superseded`` entry or an ``equal_witnesses`` record whose
    ``payload_digest`` is an EXPLICIT ``null`` read to the scan as the
    pre-PW14 absence (``WITHOUT a payload digest``) and any other non-digest
    value as a changed claim, and the writer silently RE-RECORDED either as a
    legacy record (``was unrecorded`` under the pass rule; ``now carries its
    payload digest`` for a witness) while the loader refused the file. Now
    the scan reads such a store PENDING as malformed (the field, the value
    and the remedy named) and reschedules it; the loader refuses; the
    launcher's two-tree merge (unnamed and named) and the driver's in-process
    merge refuse, nothing written and no re-recording in the REC's ``notes``;
    the file's last good write restores it; and the ABSENT key (pre-PW14)
    still reads pending as ``WITHOUT a payload digest`` and is re-recorded by
    the next merge, said, the file equal to the good one."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    D, SUP, EQW, SRC = (C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD, C.MERGE_SOURCE_SUPERSEDED_FIELD,
                        C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD, C.SUMMARY_KEY_SOURCES)

    def malformed_refuses_everywhere(sweep, summary, store, field, good, *, tree, named, re_recording, acc):
        """For ``None`` and ``"xyz"`` under ``field``'s record of ``store``: pending as malformed, refused by the
        loader and by every merge path, nothing written; then the good bytes restore it."""
        src, bad = sandbox_mod.store_merge_source(store, json.loads((store / "manifest.json").read_text()))
        assert bad is None
        for value in (None, "xyz"):
            d = json.loads(good)
            d[SRC][EVALPLUS_TASK][field][0][D] = value
            summary.write_text(json.dumps(d, indent=1, sort_keys=True))
            broken = summary.read_bytes()
            # (a) the scan: PENDING as malformed — the field, the value and the remedy named, never the pre-PW14
            # absence or a changed claim — and rescheduled
            [p] = sandbox_mod.unmerged_completed_stores(sweep)
            why = p["why"][EVALPLUS_TASK]
            assert p["cell_dir"] == str(store) and repr(value) in why and "explicit null or any other value" in why
            assert "last good write" in why and "PW19c" in why
            assert "WITHOUT a payload digest" not in why and "CHANGED" not in why
            assert sandbox_mod.plan_merges([p])["commands"] != {}
            reason = sandbox_mod.unaccounted_reason(src, d[SRC][EVALPLUS_TASK], payload_digest=_digest(store))
            assert reason is not None and repr(value) in reason and "PW19c" in reason
            # (b) the loader refuses
            with pytest.raises(ValueError, match="16 hex"):
                load_controls_summaries(sweep, [cell], cfg)
            # (c) the merge refuses — the launcher's two-tree path unnamed and named (recorded, not raised; no
            # re-recording in the notes) and the driver's in-process merge — nothing written
            for flag in ("", named):
                res = launcher_mod.merge_cell_summaries([str(store)], tree=tree, python=sys.executable,
                                                        replace_source=flag)
                err = res[str(store)]["err"]
                assert res[str(store)]["ok"] is False and "explicit null or any other value" in err, res
                assert repr(value) in err and repr(field) in err and "PW19c" in err
                assert not any(re_recording in n for n in res[str(store)]["notes"])
                assert summary.read_bytes() == broken
            with pytest.raises(ValueError, match="explicit null or any other value"):
                completion.merge_controls_summary(store)
            assert summary.read_bytes() == broken
        # the remedy: the file's last good write — accounted and loadable again
        summary.write_bytes(good)
        assert sandbox_mod.unmerged_completed_stores(sweep) == []
        assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][EVALPLUS_TASK]["acc"] == acc

    # (1) `superseded`: a pass-1 store displaced by a pass-2 store of the same cell (this tree emits and writes)
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    p1 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11111111", chash="11111111" + "0" * 8,
                     pass_no=1, acc=0.25)
    p2 = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__22222222", chash="22222222" + "0" * 8,
                     pass_no=2, acc=0.5)
    res = launcher_mod.merge_cell_summaries([str(p1), str(p2)], tree=REPO, python=sys.executable)
    assert all(v["ok"] for v in res.values()), res
    rec = json.loads(summary.read_text())[SRC][EVALPLUS_TASK]
    assert rec["cell_key"] == p2.name and rec[SUP] == [_hist(p1, 1)]
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    good = summary.read_bytes()
    malformed_refuses_everywhere(sweep, summary, p1, SUP, good, tree=REPO, named=f"{EVALPLUS_TASK}={p2.name}",
                                 re_recording="payload digest is now", acc=0.5)
    # the contrast that defines "absent": the KEY missing (pre-PW14) reads pending as WITHOUT a digest, and the next
    # merge — the pass rule keeps the entry — re-records it with the digest, said, the file equal to the good one
    d = json.loads(good)
    d[SRC][EVALPLUS_TASK][SUP][0].pop(D)
    summary.write_text(json.dumps(d, indent=1, sort_keys=True))
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    assert p["cell_dir"] == str(p1) and "WITHOUT a payload digest" in p["why"][EVALPLUS_TASK]
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and any("payload digest is now" in n and "was unrecorded" in n for n in res[str(p1)]["notes"])
    assert json.loads(summary.read_text()) == json.loads(good) and sandbox_mod.unmerged_completed_stores(sweep) == []
    # (2) `equal_witnesses`: two hashed pass-1 stores of this tree exactly witnessing an unrecorded standing payload,
    # the adopter picked by the recorded rule (PW12; hashed rather than the PW13 test's legacy stores so that the
    # driver's in-process merge — which refuses an unhashed store by design — is a live path here too)
    sweep2 = tmp_path / "sweep2"
    summary2 = sweep2 / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    a = _pass_store(sweep2, base, key="tok__controls__t0__a0__aaaaaaaa__aaaaaaaa", chash="aaaaaaaa" + "0" * 8,
                    pass_no=1, acc=0.4)
    b = _pass_store(sweep2, base, key="tok__controls__t0__a0__aaaaaaaa__bbbbbbbb", chash="bbbbbbbb" + "0" * 8,
                    pass_no=1, acc=0.4)
    pre = {"cell_id": C.CELL_PSRKL_SAFETY, "controls": {EVALPLUS_TASK: completion.read_task_summary(a, EVALPLUS_TASK)},
           "provenance": base[C.MANIFEST_KEY_PROVENANCE]}
    summary2.parent.mkdir(parents=True)
    summary2.write_text(json.dumps(pre, indent=1, sort_keys=True))
    res = launcher_mod.merge_cell_summaries([str(b), str(a)], tree=REPO, python=sys.executable)
    assert all(v["ok"] for v in res.values()), res
    rec = json.loads(summary2.read_text())[SRC][EVALPLUS_TASK]
    assert rec["cell_key"] == a.name and rec[EQW] == [_hist(b, 1)]
    assert sandbox_mod.unmerged_completed_stores(sweep2) == []
    good2 = summary2.read_bytes()
    malformed_refuses_everywhere(sweep2, summary2, b, EQW, good2, tree=REPO, named=f"{EVALPLUS_TASK}={a.name}",
                                 re_recording="now carries its payload digest", acc=0.4)
    # ...the absent key: pending as WITHOUT a digest, and the witness's next equal re-offer records it, said
    d = json.loads(good2)
    d[SRC][EVALPLUS_TASK][EQW][0].pop(D)
    summary2.write_text(json.dumps(d, indent=1, sort_keys=True))
    [p] = sandbox_mod.unmerged_completed_stores(sweep2)
    assert p["cell_dir"] == str(b) and "WITHOUT a payload digest" in p["why"][EVALPLUS_TASK]
    res = launcher_mod.merge_cell_summaries([str(b)], tree=REPO, python=sys.executable)
    assert res[str(b)]["ok"] and any("now carries its payload digest" in n for n in res[str(b)]["notes"])
    assert summary2.read_bytes() == good2 and sandbox_mod.unmerged_completed_stores(sweep2) == []


def test_rederive_evalplus_recomputes_the_retained_results_off_volume(tmp_path, monkeypatch, capsys):
    """PW12 (E19, NEW High): the retained 618f9dd stores were converted under
    the plus-only ``pass@1_plus``; their ``eval_results.json`` is retained,
    so ``rederive_evalplus`` recomputes rows + summary under the CURRENT
    definition (base AND plus) through the conversion's own functions,
    touching nothing in the store, and ``write_rederivation`` files the
    result as a sidecar OFF the store (refusing a path inside it or its
    out_root) — rows, the summary naming ``metric_def``, and a provenance
    record with the flipped items and the delta. The script drives it."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, tdir = _cell_dir(tmp_path)
    mixed = _results(plus=("pass", "pass"), base=("pass", "fail"))            # HumanEval/1: base-fail / plus-pass
    (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(mixed))
    old = {"acc": 1.0, "pass_at_1_base": 0.5, "pass_at_1_plus": 1.0, "n": 2, "n_requests": 3, "n_rejected": 1,
           "samples_per_item": 1, "framework": C.EVALPLUS_TASK_VERSION, "engine": "vllm",
           "route": LENS_ROUTE.virtual_model, "thinking": "off", "wall_clock_s": 20.0}   # the plus-only summary
    completion.finish_task(cell, task=EVALPLUS_TASK, n_items=2, n_expected_full=2, limits={}, n_rows=6,
                           n_requests=3, n_rejected=1, wall_clock_s=20.0, summary=old)
    before = {p: p.read_bytes() for p in cell.rglob("*") if p.is_file()}
    derived = er.rederive_evalplus(cell, dataset="humaneval")
    assert derived["task"] == EVALPLUS_TASK and derived["cell_key"] == cell.name
    assert derived["summary"]["pass_at_1_plus"] == 0.5 == derived["summary"]["acc"]
    assert derived["summary"]["pass_at_1_base"] == 0.5 and derived["summary"]["wall_clock_s"] == 20.0
    assert derived["summary"][C.EVALPLUS_METRIC_DEF_FIELD] == C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS
    assert derived["retained_summary"] == old and derived["retained_metric_def"] == C.EVALPLUS_PLUS_DEF_PLUS_ONLY
    assert derived["flipped_items"] == ["HumanEval/1"] and derived["delta_pass_at_1_plus"] == -0.5
    assert derived["eval_results_sha256"] == hashlib.sha256((tdir / er.EVAL_RESULTS_FILENAME).read_bytes()).hexdigest()
    assert derived["harness_config_hash"] == "bb" * 16
    assert derived["producer_tree_sha"] == UNDECLARED_TREE_SHA
    # E54 #1: the re-derivation is BOUND to the sample bytes it re-derived from
    # — it recorded eval_results_sha256 before and neither verified nor
    # retained the samples' digest, so nothing tied the sidecar's numbers to a
    # generation
    assert derived["samples_binding"] == C.SAMPLES_BINDING_BOUND
    assert derived[er.SAMPLES_DIGEST_FIELD] == hashlib.sha256(
        (tdir / er.SAMPLES_FILENAME).read_bytes()).hexdigest()
    by = {(r["item_id"], r["metric"]): r for r in derived["rows"]}
    assert len(by) == 6 and by[("HumanEval/1", "pass@1_plus")]["value"] == 0.0 == by[("HumanEval/1", "acc")]["value"]
    assert json.loads(by[("HumanEval/1", "pass@1_plus")]["extra_json"]) == {
        C.EVALPLUS_METRIC_DEF_FIELD: C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS, "content_empty_with_reasoning": None}
    # the sidecar: rows + summary + provenance, never inside the store or its out_root, no done marker
    out = tmp_path / "rederived"
    cell_out = er.write_rederivation(out, derived)
    assert cell_out == out / cell.name
    assert json.loads((cell_out / C.SUMMARY_DIR / C.marker_filename(EVALPLUS_TASK)).read_text()) == derived["summary"]
    assert len((cell_out / "rows.jsonl").read_text().splitlines()) == 6 and not (cell_out / C.DONE_DIR).exists()
    rec = json.loads((cell_out / er.REDERIVATION_FILENAME).read_text())
    assert rec["tasks"][EVALPLUS_TASK]["flipped_items"] == ["HumanEval/1"] and "rows" not in rec["tasks"][EVALPLUS_TASK]
    assert rec["tasks"][EVALPLUS_TASK][er.SAMPLES_DIGEST_FIELD] == derived[er.SAMPLES_DIGEST_FIELD]
    assert rec["metric_def"] == C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS and rec["source_cell_dir"] == str(cell)
    er.write_rederivation(out, derived)
    assert len((cell_out / "rows.jsonl").read_text().splitlines()) == 6                 # idempotent on rows
    for bad in (cell, cell / "x", cell.parent.parent, cell.parent.parent / "rederived"):
        with pytest.raises(er.EvalPlusSeamError, match="OFF the volume"):
            er.write_rederivation(bad, derived)
    assert {p: p.read_bytes() for p in cell.rglob("*") if p.is_file()} == before      # the store is untouched
    # the script, end to end
    spec = importlib.util.spec_from_file_location("rederive_script", REPO / "scripts" / "tinycap_rederive_evalplus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(cell), "--out", str(tmp_path / "r2")]) == 0
    text = capsys.readouterr().out
    assert "flipped=['HumanEval/1']" in text and C.EVALPLUS_PLUS_DEF_PLUS_ONLY in text and "delta=-0.5" in text
    assert (tmp_path / "r2" / cell.name / er.REDERIVATION_FILENAME).is_file()
    with pytest.raises(SystemExit, match="no DONE marker"):
        mod.main([str(cell), "--out", str(tmp_path / "r3"), "--tasks", MBPP_TASK])
    with pytest.raises(SystemExit, match="OFF the volume"):
        mod.main([str(cell), "--out", str(cell.parent.parent / "side")])
    # the consumer's rule: the retained (plus-only) and the re-derived numbers are never compared directly
    from sisteer.eval.cell_controls import metric_def_mismatch

    assert "not directly comparable" in metric_def_mismatch(
        {**old, C.EVALPLUS_METRIC_DEF_FIELD: C.EVALPLUS_PLUS_DEF_PLUS_ONLY}, derived["summary"])
    assert metric_def_mismatch(derived["summary"], dict(derived["summary"])) is None


def test_a_rederivation_binds_its_numbers_or_refuses_naming_the_store(tmp_path, monkeypatch):
    """E54 #1 at the off-volume re-derivation. It recorded ``eval_results_sha256``
    and never looked at the samples, so a sidecar's numbers were tied to the
    results file and to nothing that produced it — and the four shapes of "no
    binding" (no samples, no execution, a torn execution, no/invalid
    ``samples_sha256``) all sailed through as "an old store".

    They are now judged against what the store POSITIVELY says about its own
    age. A current store refuses, naming the store and the remedy; one of the
    campaign's declared pre-landing trees is re-derived and the sidecar SAYS it
    is unbound; and a contradicted pair (samples regenerated after the results)
    refuses whatever the era — that is a contradiction, not an absence."""
    from sisteer.capeval.tinycap import contracts as tc

    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    old = {"acc": 1.0, "pass_at_1_base": 0.5, "pass_at_1_plus": 1.0, "n": 2, "n_requests": 3,
           "n_rejected": 1, "samples_per_item": 1, "framework": C.EVALPLUS_TASK_VERSION,
           "engine": "vllm", "route": LENS_ROUTE.virtual_model, "thinking": "off",
           "wall_clock_s": 20.0}

    def store(root, **kw):
        cell, tdir = _cell_dir(root, **kw)
        completion.finish_task(cell, task=EVALPLUS_TASK, n_items=2, n_expected_full=2, limits={},
                               n_rows=6, n_requests=3, n_rejected=1, wall_clock_s=20.0, summary=old)
        return cell, tdir

    # a CURRENT store refuses every shape, naming the store's own dating
    for i, binding in enumerate(("none", "unbound", "nosamples")):
        cell, _tdir = store(tmp_path / f"cur{i}", binding=binding)
        with pytest.raises(er.EvalPlusSeamError) as ei:
            er.rederive_evalplus(cell, dataset="humaneval")
        assert "E54 #1" in str(ei.value) and UNDECLARED_TREE_SHA in str(ei.value)
    cell, tdir = store(tmp_path / "torn", binding="unbound")
    (tdir / er.EXECUTION_FILENAME).write_text("{ torn")
    with pytest.raises(er.EvalPlusSeamError, match="did not parse"):
        er.rederive_evalplus(cell, dataset="humaneval")

    # a store that cannot name its producing tree at all is UNKNOWN, not old
    cell, _tdir = store(tmp_path / "nameless", binding="unbound", producer=None)
    with pytest.raises(er.EvalPlusSeamError, match="no full 40-hex"):
        er.rederive_evalplus(cell, dataset="humaneval")

    # a tree that predates the samples-digest contract IS re-derived, and says
    # so. The two wave trees are the population this actually classifies:
    # both descend from the sampling contract and predate this one (E55 #1).
    for wave in (PRE_EVERYTHING_TREE_SHA, POST_SAMPLING_TREE_SHA):
        assert tc.store_era({tc.MANIFEST_KEY_PRODUCER: {"tree_sha": wave}},
                            contract=tc.CONTRACT_SAMPLES_DIGEST)[0] == tc.STORE_ERA_HISTORICAL
    hist = POST_SAMPLING_TREE_SHA
    cell, tdir = store(tmp_path / "hist", binding="unbound", producer=hist)
    derived = er.rederive_evalplus(cell, dataset="humaneval")
    assert derived["samples_binding"] == tc.SAMPLES_BINDING_HISTORICAL_UNBOUND
    assert derived[er.SAMPLES_DIGEST_FIELD] is None and derived["producer_tree_sha"] == hist

    # a CONTRADICTED pair refuses on any era: the samples were regenerated
    # after the results, which is a contradiction rather than an absence
    (tdir / er.SAMPLES_FILENAME).write_bytes(b'{"task_id": "HumanEval/0", "solution": "v2"}\n')
    (tdir / er.EXECUTION_FILENAME).write_text(json.dumps(
        {"wall_clock_s": 7.5, er.SAMPLES_DIGEST_FIELD: "0" * 64}))
    with pytest.raises(er.EvalPlusSeamError, match="regenerated after those results"):
        er.rederive_evalplus(cell, dataset="humaneval")

    # the digest field is ONE name in three modules (the sandbox's orchestrator
    # image can import none of the others), pinned equal here
    from sisteer.eval import cell_controls as cc

    assert er.SAMPLES_DIGEST_FIELD == cc.SAMPLES_DIGEST_FIELD == "samples_sha256"
    src = (REPO / "scripts" / "modal" / "tinycap_sandbox.py").read_text()
    assert f'SAMPLES_DIGEST_FIELD = "{er.SAMPLES_DIGEST_FIELD}"' in src


def test_the_conversion_door_refuses_results_with_no_samples_beside_them(sandbox_mod, tmp_path,
                                                                         monkeypatch):
    """E54 #1, the fourth shape at the CONVERSION door. ``samples_binding``
    hashed ``samples.jsonl`` before asking anything, so a task whose samples had
    been deleted between the scan's glob and the read raised an OSError out of
    the scan instead of refusing that one task with its remedy. It is now the
    same per-task refusal as the other three — the conversion door stays
    STRICTER than the merge on purpose (it refuses an unbound execution for
    every producer, old or not: there refusing costs one re-execution and no
    number exists yet)."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    _cell, tdir = _cell_dir(tmp_path)
    assert sandbox_mod.samples_binding(tdir)[0] == sandbox_mod.BINDING_BOUND
    (tdir / er.SAMPLES_FILENAME).unlink()
    verdict, msg = sandbox_mod.samples_binding(tdir)
    assert verdict == sandbox_mod.BINDING_UNBOUND
    assert f"no {er.SAMPLES_FILENAME}" in msg and "executes afresh" in msg


def test_entrypoint_chains_one_merge_per_producing_tree(sandbox_mod, launcher_mod, monkeypatch,
                                                        capsys, tmp_path):
    """The local entrypoint on the REAL two-tree path: refuses to execute
    without a merge decision; ``--merge`` runs ONE launcher command per
    producer sha from the DISK-derived pending list with ``--sha`` = that
    producer, and each command — the launcher's ``merge_summaries`` — has the
    PRODUCING tree (a real 618f9dd-style tree; this checkout staged under
    another sha) emit while THIS tree writes; the legacy group's top-level
    pass never orders (PW12), so the current group's claim on the same entry
    refuses until the operator names the winner through the launcher;
    ``--no-merge`` prints and runs nothing; ``--merge-sha`` refuses foreign
    trees; a producer with no staged tree fails its merge and the run exits
    with the re-run command; a legacy (producer-less) store is refused."""
    _cfg, _cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    repos = tmp_path / "repos"
    _legacy_tree(repos)
    _current_tree(repos)
    monkeypatch.setattr(launcher_mod.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(launcher_mod, "CLIENT_PY", sys.executable)
    legacy = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__618f9dd0",
                         chash="618f9dd0" + "0" * 8, pass_no=1, acc=0.25, hashed=False, tree_sha=LEGACY_SHA)
    current = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__c0ffee00",
                          chash="c0ffee00" + "0" * 8, pass_no=2, acc=0.75, tree_sha=CURRENT_SHA)
    monkeypatch.setattr(sandbox_mod, "tinycap_sandbox_execute", types.SimpleNamespace(
        remote=lambda **kw: {"dry_run": kw.get("dry_run", False), "plan": {}, "records": [],
                             "merge_pending": sandbox_mod.unmerged_completed_stores(sweep)}))
    ran: list = []
    recs: dict = {}

    def runner(cmd):
        # stands in for `python -m modal run tinycap_served.py --sha X --merge --merge-cell-dirs a,b`:
        # the launcher's Modal function, in this process, its subprocesses REAL
        ran.append(list(cmd))
        sha = cmd[cmd.index("--sha") + 1]
        dirs = cmd[cmd.index("--merge-cell-dirs") + 1].split(",")
        rec = launcher_mod.merge_summaries(sha, dirs, str(sweep))
        recs[sha] = rec
        return types.SimpleNamespace(returncode=0 if rec["status"] == "OK" else 1)

    real_run_merges = sandbox_mod.run_merges
    monkeypatch.setattr(sandbox_mod, "run_merges",
                        lambda plan, *, chain: real_run_merges(plan, chain=chain, runner=runner))
    with pytest.raises(SystemExit, match="--merge-sha"):
        sandbox_mod.main()
    out = tmp_path / "res.json"
    # --merge-mode no-merge prints the commands and runs nothing; a dry run merges nothing either
    sandbox_mod.main(merge_mode=sandbox_mod.MERGE_MODE_PRINT)
    assert ran == [] and "MERGE PENDING" in capsys.readouterr().out and not summary.exists()
    sandbox_mod.main(dry_run=True, out=str(out))
    assert ran == [] and json.loads(out.read_text())["merge"]["status"] == "dry run: nothing merged"
    # --merge-mode merge: one REAL merge per producing tree, this tree writing both. The legacy
    # group (sorted first) lands its pass-1 entry; the current group's hashed pass 2 then meets a
    # standing LEGACY pass, which never orders (PW12 / E19 A1-1) — that merge FAILS, named, and
    # the run exits with the re-run command; the chained commands never carry --replace-source
    W = launcher_mod.writer_tree_sha_of(REPO / "src")
    with pytest.raises(SystemExit, match="re-run"):
        sandbox_mod.main(merge_mode=sandbox_mod.MERGE_MODE_CHAIN, out=str(out))
    assert [c[6] for c in ran] == [LEGACY_SHA, CURRENT_SHA]                 # one merge per producing tree
    assert all(c[7:9] == ["--merge", "--merge-cell-dirs"] and c[4].endswith("tinycap_served.py") for c in ran)
    assert all(C.REPLACE_SOURCE_FLAG not in c for c in ran)
    assert recs[LEGACY_SHA]["status"] == "OK" and recs[CURRENT_SHA]["status"] == "FAILED"
    assert "never orders passes" in recs[CURRENT_SHA]["results"][str(current)]["err"]
    assert recs[LEGACY_SHA]["tree"].endswith("sisteer-618f9dd") and recs[LEGACY_SHA]["cell_dirs"] == [str(legacy)]
    assert recs[CURRENT_SHA]["cell_dirs"] == [str(current)]
    assert recs[LEGACY_SHA]["writer_src"] == recs[CURRENT_SHA]["writer_src"] == str(launcher_mod.current_tree_src()) == str(REPO / "src")
    assert recs[LEGACY_SHA]["writer_tree_sha"] == recs[CURRENT_SHA]["writer_tree_sha"] == W   # PW12: WHICH tree wrote
    for sha in (LEGACY_SHA, CURRENT_SHA):                                   # ...attested at both ends
        att = recs[sha]["attest"]
        assert att["emitter"]["ok"] and att["emitter"]["completion_file"].startswith(recs[sha]["tree"])
        assert att["writer"]["ok"] and att["writer"]["completion_file"] == str(
            REPO / "src" / "sisteer" / "capeval" / "tinycap" / "completion.py")
        assert att["writer"]["writer_tree_sha"] == W
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.25                    # the legacy entry stands, sourced + marked
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(
        legacy, 1, **{C.MERGE_SOURCE_PASS_SOURCE_FIELD: C.PASS_SOURCE_LEGACY_TOP_LEVEL,
                      C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD: W})
    assert [r["cell_dir"] for r in sandbox_mod.unmerged_completed_stores(sweep)] == [str(current)]
    # the operator's decision, through the launcher: the hashed pass-2 store takes the entry
    rec = launcher_mod.merge_summaries(CURRENT_SHA, [str(current)], str(sweep), f"{EVALPLUS_TASK}={current.name}")
    assert rec["status"] == "OK"
    doc = json.loads(summary.read_text())
    assert doc["controls"][EVALPLUS_TASK]["acc"] == 0.75
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK] == _standing(
        current, 2, _hist(legacy, 1), **{C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD: W})
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    # nothing pending: nothing to merge, no exit
    ran.clear()
    sandbox_mod.main(merge_mode="merge", out=str(out))
    assert ran == [] and json.loads(out.read_text())["merge"]["status"] == "nothing to merge"
    # --merge-sha: a group produced by another tree is refused; the run exits non-zero
    foreign = _pass_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__ffffffff",
                          chash="ffffffff" + "0" * 8, pass_no=2, acc=0.5, tree_sha=FULL_B, task=MBPP_TASK)
    ran.clear()
    with pytest.raises(SystemExit, match="could not be merged"):
        sandbox_mod.main(merge_sha=LEGACY_SHA[:7])
    assert ran == [] and "MERGE REFUSED" in capsys.readouterr().out
    # a producer whose tree is not staged: its merge FAILS and the run exits with the re-run command
    with pytest.raises(SystemExit, match="re-run"):
        sandbox_mod.main(merge_mode="merge")
    assert [c[6] for c in ran] == [FULL_B] and recs[FULL_B]["status"] == "NO_TREE"
    assert MBPP_TASK not in json.loads(summary.read_text())["controls"]
    # a legacy store (no producer block) fails the run, named
    m = json.loads((foreign / "manifest.json").read_text())
    del m[C.MANIFEST_KEY_PRODUCER]
    (foreign / "manifest.json").write_text(json.dumps(m))
    ran.clear()
    with pytest.raises(SystemExit, match="could not be merged"):
        sandbox_mod.main(merge_mode="merge")
    assert ran == [] and "legacy" in capsys.readouterr().out


def _modal_click_command(fn):
    """The click command ``modal run`` builds for a local entrypoint — modal
    1.5.4's ``modal/cli/run.py::_add_click_options``: one ``--<name>`` option
    per parameter (``_`` -> ``-``), typed by the annotation and defaulted by
    the signature, and a ``bool`` registered as ``--<name>/--no-<name>``.
    Replicated here because the test venv carries no modal; the rule is what
    made two bools named ``merge`` / ``no_merge`` collide on ``--no-merge``."""
    import click

    hints = typing.get_type_hints(fn)
    for name, param in inspect.signature(fn).parameters.items():
        tp = hints.get(name, str)
        flag = "--" + name.replace("_", "-")
        if tp is bool:
            flag += "/--no-" + name.replace("_", "-")
        click.option(flag, type=tp, default=param.default)(fn)
    return click.command()(fn)


def test_the_merge_decision_is_one_option_the_modal_cli_can_express(sandbox_mod, monkeypatch):
    """PW10 (operator C, 2026-09-04): ``modal run`` registers every bool
    parameter as ``--<name>/--no-<name>``, so ``merge: bool`` owned
    ``--no-merge`` and shadowed ``no_merge: bool``'s own flag — ``modal run …
    --no-merge`` set ``merge=False`` and exited at the gate before any remote
    call (the tests called ``main(no_merge=True)`` directly and never saw
    it). The decision is now ONE string option, ``--merge-mode
    {merge,no-merge}`` beside ``--merge-sha``; structurally no parameter's
    own flag is another bool's negation; and the generated command is
    invoked here with each flag."""
    from click.testing import CliRunner

    params = inspect.signature(sandbox_mod.main).parameters
    hints = typing.get_type_hints(sandbox_mod.main)
    own = {"--" + n.replace("_", "-") for n in params}
    negations = {"--no-" + n.replace("_", "-") for n in params if hints[n] is bool}
    assert not (own & negations), own & negations                       # nothing shadowed
    assert "merge" not in params and "no_merge" not in params
    assert hints["merge_mode"] is str and hints["merge_sha"] is str
    assert sandbox_mod.MERGE_MODES == ("merge", "no-merge") and sandbox_mod.MERGE_MODE_FLAG == "--merge-mode"
    pending = [{"cell_dir": "/g/cells/k1", "cell_id": "c", "tasks": ["evalplus/mbpp_plus"],
                "producer_tree_sha": FULL_A, "summary": "/g/c/controls_summary.json"}]
    monkeypatch.setattr(sandbox_mod, "tinycap_sandbox_execute", types.SimpleNamespace(
        remote=lambda **kw: {"dry_run": kw.get("dry_run", False), "plan": {}, "records": [],
                             "merge_pending": [dict(p) for p in pending]}))
    chained: list = []
    monkeypatch.setattr(sandbox_mod, "run_merges",
                        lambda plan, *, chain: (chained.append(chain), {FULL_A: "OK" if chain else "PENDING"})[1])
    cmd = _modal_click_command(sandbox_mod.main)
    runner = CliRunner()
    # no decision: refused at the gate, naming all three ways to decide
    r = runner.invoke(cmd, [])
    assert r.exit_code == 1 and "decide the controls_summary merge up front" in r.output
    assert "--merge-mode merge" in r.output and "--merge-mode no-merge" in r.output and "--merge-sha" in r.output
    assert chained == []
    # the old flag is no longer a thing the CLI accepts — not silently `merge=False`
    r = runner.invoke(cmd, ["--no-merge"])
    assert r.exit_code == 2 and "No such option" in r.output and chained == []
    # each way, through the generated command
    assert runner.invoke(cmd, ["--merge-mode", "merge"]).exit_code == 0 and chained == [True]
    assert runner.invoke(cmd, ["--merge-mode", "no-merge"]).exit_code == 0 and chained == [True, False]
    assert runner.invoke(cmd, ["--merge-sha", FULL_A[:7]]).exit_code == 0 and chained == [True, False, True]
    assert runner.invoke(cmd, ["--dry-run"]).exit_code == 0 and chained == [True, False, True]
    r = runner.invoke(cmd, ["--merge-mode", "bogus"])
    assert r.exit_code == 1 and "not one of ['merge', 'no-merge']" in r.output
    r = runner.invoke(cmd, ["--merge-mode", "no-merge", "--merge-sha", FULL_A[:7]])
    assert r.exit_code == 1 and "pick one" in r.output
    assert chained == [True, False, True]
    # the decision function alone
    assert sandbox_mod.merge_decision("merge", "", dry_run=False) is True
    assert sandbox_mod.merge_decision("no-merge", "", dry_run=False) is False
    assert sandbox_mod.merge_decision("", "abc", dry_run=False) is True
    assert sandbox_mod.merge_decision("", "", dry_run=True) is False
    with pytest.raises(SystemExit, match="decide"):
        sandbox_mod.merge_decision("", "", dry_run=False)


def test_pass_at_1_plus_is_evalplus_headline_base_and_plus(tmp_path, monkeypatch):
    """PW10 (the served-path owner, 2026-09-04): evalplus's printed "plus"
    pass@1 counts a sample only when it passes the BASE tests AND the plus
    tests (``evaluate.py``: ``base_status == plus_status == PASS``); the
    store scored ``plus_status`` alone, so a base-fail / plus-pass item
    (Mbpp/631 and Mbpp/635 on every MBPP+ run) inflated our plus number by
    0.5 pp over evalplus's. ``pass@1_plus`` and the gating ``acc`` now follow
    evalplus's definition; ``pass@1_base`` is unchanged; the served
    conversion's summary follows."""
    results = _results(ids=("HumanEval/0", "HumanEval/1"), base=("fail", "pass"), plus=("pass", "pass"))
    rows = er.eval_results_to_rows(results, frozenset(results["eval"]), task=EVALPLUS_TASK)
    by = {(r["item_id"], r["metric"]): r["value"] for r in rows}
    assert by[("HumanEval/0", "pass@1_base")] == 0.0
    assert by[("HumanEval/0", "pass@1_plus")] == 0.0 == by[("HumanEval/0", "acc")]   # base-fail: not a plus pass
    assert by[("HumanEval/1", "pass@1_base")] == 1.0
    assert by[("HumanEval/1", "pass@1_plus")] == 1.0 == by[("HumanEval/1", "acc")]
    # base-pass / plus-fail stays a plus fail; base-pass / plus-pass a pass
    results = _results(base=("pass", "pass"), plus=("fail", "pass"))
    by = {(r["item_id"], r["metric"]): r["value"] for r in er.eval_results_to_rows(results)}
    assert by[("HumanEval/0", "pass@1_plus")] == 0.0 and by[("HumanEval/1", "pass@1_plus")] == 1.0
    assert by[("HumanEval/0", "pass@1_base")] == 1.0
    # the served conversion's summary follows the rows
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, tdir = _cell_dir(tmp_path)
    (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(
        _results(base=("fail", "pass"), plus=("pass", "pass"))))
    summary = er.convert_eval_results_to_store(cell, dataset="humaneval")
    assert summary["pass_at_1_base"] == 0.5 and summary["pass_at_1_plus"] == 0.5 == summary["acc"]
    stored = {(json.loads(l)["item_id"], json.loads(l)["metric"]): json.loads(l)["value"]
              for l in (cell / "rows.jsonl").read_text().splitlines()}
    assert stored[("HumanEval/0", "pass@1_plus")] == 0.0 and stored[("HumanEval/0", "pass@1_base")] == 0.0
    assert stored[("HumanEval/1", "pass@1_plus")] == 1.0


# --------------------------------------------------------------------------- #
# generalized pairing                                                          #
# --------------------------------------------------------------------------- #


def _row(task, iid, value, engine, extra=None, metric="acc"):
    return {"task": task, "metric": metric, "item_id": iid, "value": value, "engine": engine,
            "extra_json": json.dumps(extra) if extra is not None else None}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _tiny_stores(tmp_path, *, flip="2"):
    items = [("0", "B", "B"), ("1", "A", "C"), ("2", "D", "D"), ("3", "C", "C")]
    hf, sv = [], []
    for iid, pick, gold in items:
        hf.append(_row("s2parity/mmlu_letter", iid, float(pick == gold), "hf", {"pick": pick, "gold": gold}))
        sp = "A" if iid == flip else pick
        sv.append(_row("inspect/mmlu_letter_valid", iid, float(sp == gold), "vllm", {"pick": sp, "gold": gold}))
    hf.append(_row("s2parity/mmlu_letter", "__aggregate__", 0.75, "hf", metric="acc_aggregate"))
    hf.append(_row("s2parity/gsm8k_0shot", "0", 1.0, "hf"))   # another task: ignored
    return _write(tmp_path / "hf.jsonl", hf), _write(tmp_path / "sv.jsonl", sv)


def test_pairing_item_id_join_delta_discordant_mcnemar(tmp_path):
    hf, sv = _tiny_stores(tmp_path)
    a = pr.Side("hf", hf, "s2parity/mmlu_letter", "hf")
    b = pr.Side("served", sv, "inspect/mmlu_letter_valid", "vllm")
    r = pr.compare_sides(a, b)
    assert r["n_items"] == 4 and r["mean_a"] == 0.75 and r["mean_b"] == 0.5
    assert r["delta_b_minus_a"] == pytest.approx(-0.25) and r["binary"]
    assert [d["id"] for d in r["discordant"]] == ["2"]
    assert r["discordant"][0]["a"]["pick"] == "D" and r["discordant"][0]["b"]["pick"] == "A"
    assert r["mcnemar"]["a_only"] == 1 and r["mcnemar"]["b_only"] == 0
    assert r["pick_agreement"] == 0.75 and r["n_pick_disagreements"] == 1
    assert r["only_a"] == [] and r["only_b"] == []
    assert r["sides"]["a"]["n_rows"] == 4 and "unpaired_rows" not in r["sides"]["a"]


def test_pairing_refuses_wrong_engine_duplicates_and_mismatch(tmp_path):
    hf, sv = _tiny_stores(tmp_path)
    with pytest.raises(pr.PairingError, match="engine"):
        pr.load_items(pr.Side("x", sv, "inspect/mmlu_letter_valid", "hf"))
    dup = _write(tmp_path / "dup.jsonl", [_row("t", "0", 1.0, "hf")] * 2)
    with pytest.raises(pr.PairingError, match="duplicate"):
        pr.load_items(pr.Side("x", dup, "t", "hf"))
    with pytest.raises(pr.PairingError, match="no per-item rows"):
        pr.load_items(pr.Side("x", hf, "nope", "hf"))
    with pytest.raises(pr.PairingError, match="one of"):
        pr.Side("x", hf, "t", "hf", key="doc_id")
    a, _ = pr.load_items(pr.Side("hf", hf, "s2parity/mmlu_letter", "hf"))
    b, _ = pr.load_items(pr.Side("sv", sv, "inspect/mmlu_letter_valid", "vllm"))
    b.pop("3")
    b["9"] = pr.Item(1.0, "9")
    with pytest.raises(pr.PairingError, match="item sets differ"):
        pr.compare(a, b)
    r = pr.compare(a, b, allow_unmatched=True)
    assert r["n_items"] == 3 and r["only_a"] == ["3"] and r["only_b"] == ["9"]
    with pytest.raises(pr.PairingError, match="share no items"):
        pr.compare(a, {"z": pr.Item(1.0, "z")}, allow_unmatched=True)


def test_pairing_paired_id_join_and_non_binary_means(tmp_path):
    ours = _write(tmp_path / "ours.jsonl", [
        _row("s2parity/mmlu_letter", "m0000-aaaaaaaa", 1.0, "vllm", {"pick": "B", "gold": "B"}),
        _row("s2parity/mmlu_letter", "m0001-bbbbbbbb", 0.0, "vllm", {"pick": "A", "gold": "C"}),
    ])
    inspect_rows = _write(tmp_path / "inspect.jsonl", [
        _row("inspect/mmlu_letter_valid", "mmlu_11111111", 1.0, "vllm",
             {"paired_id": "m0000-aaaaaaaa", "answer": "B", "sample_idx": 0}),
        _row("inspect/mmlu_letter_valid", "mmlu_22222222", 1.0, "vllm",
             {"paired_id": "m0001-bbbbbbbb", "answer": "C", "sample_idx": 0}),
        _row("inspect/mmlu_0_shot", "mmlu_33333333", 1.0, "vllm", {"paired_id": None}),
        _row("inspect/mmlu_letter_valid", "mmlu_44444444", 0.0, "vllm", {"paired_id": None}),
    ])
    a = pr.Side("ours", ours, "s2parity/mmlu_letter", "vllm")
    b = pr.Side("inspect", inspect_rows, "inspect/mmlu_letter_valid", "vllm", key=pr.KEY_PAIRED_ID)
    r = pr.compare_sides(a, b)
    assert r["n_items"] == 2 and r["delta_b_minus_a"] == pytest.approx(0.5)
    assert r["sides"]["b"]["unpaired_rows"] == 1 and r["sides"]["b"]["n_rows"] == 3
    assert r["discordant"][0]["b"]["item_id"] == "mmlu_22222222" and r["discordant"][0]["b"]["answer"] == "C"
    assert r["pick_agreement"] is None
    # avg@k means: paired means, no McNemar
    a2 = {"x": pr.Item(0.75, "x"), "y": pr.Item(1.0, "y")}
    b2 = {"x": pr.Item(0.5, "x"), "y": pr.Item(1.0, "y")}
    r2 = pr.compare(a2, b2)
    assert not r2["binary"] and r2["mcnemar"] is None and "avg@k" in r2["mcnemar_unavailable_reason"]
    assert r2["delta_b_minus_a"] == pytest.approx(-0.125) and [d["id"] for d in r2["discordant"]] == ["x"]


def test_compare_task_cli_writes_report(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("compare_task", REPO / "scripts" / "compare_served_hf_task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hf, sv = _tiny_stores(tmp_path)
    out = tmp_path / "rep.json"
    mod.main(["--task-a", "s2parity/mmlu_letter", "--task-b", "inspect/mmlu_letter_valid",
              "--cell", "psrkl_off", str(hf), str(sv), "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rep["psrkl_off"]["n_items"] == 4 and rep["psrkl_off"]["delta_b_minus_a"] == -0.25
    text = capsys.readouterr().out
    assert "delta_b_minus_a=-0.2500" in text and "discordant 2:" in text
    with pytest.raises(pr.PairingError, match="engine"):   # swapped sides refuse
        mod.main(["--task-a", "s2parity/mmlu_letter", "--task-b", "inspect/mmlu_letter_valid",
                  "--cell", "x", str(sv), str(hf)])


# --------------------------------------------------------------------------- #
# HEAVY lane: the real evalplus 0.3.1                                          #
# --------------------------------------------------------------------------- #


class _RecordingTok:
    """A chat-templated tokenizer that records the messages evalplus hands
    it and renders ONLY the splitter, so make_raw_chat_prompt's split
    leaves nothing and the recorded messages are the whole content."""

    chat_template = "recording"

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize=False, **kw):
        self.messages = [dict(m) for m in messages]
        return er.EVALPLUS_MAGIC_SPLITTER


def test_heavy_chat_items_equal_make_raw_chat_prompt_messages():
    pytest.importorskip("evalplus", reason="HEAVY-LANE: the tiny-capability-lmeval lane venv")
    from evalplus.provider import utility as u

    assert u._MAGIC_SPLITTER_ == er.EVALPLUS_MAGIC_SPLITTER
    tok = _RecordingTok()
    item = er.render_chat_items(PROBLEMS)[0]
    rendered = u.make_raw_chat_prompt(PROBLEMS["HumanEval/0"]["prompt"].strip() + "\n",
                                      er.INSTRUCTION_PREFIX, er.RESPONSE_PREFIX, tok)
    assert rendered == ""
    assert tok.messages == [{"role": "user", "content": item.user},
                            {"role": "assistant", "content": er.evalplus_response_content()}]
    assert tok.messages[1]["content"].split(er.EVALPLUS_MAGIC_SPLITTER)[0] == item.assistant_prefill


def test_heavy_sanitize_open_extracts_and_sanitizes():
    pytest.importorskip("evalplus", reason="HEAVY-LANE: the tiny-capability-lmeval lane venv")
    gen = ("Here is the solution.\n```python\nfrom typing import List\n\n"
           "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
           "    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n"
           "```\nHope this helps!")
    sol, mode = er.sanitize_open(gen, "has_close_elements")
    assert mode == "fenced_python"
    assert sol.startswith("from typing import List") and "def has_close_elements" in sol
    assert "Hope this helps" not in sol


def test_heavy_mbpp_pin_matches_the_installed_loader():
    pytest.importorskip("evalplus", reason="HEAVY-LANE: the tiny-capability-lmeval lane venv")
    import os
    from evalplus.data.mbpp import MBPP_PLUS_VERSION
    from evalplus.data.utils import CACHE_DIR

    pin = er.dataset_pin("mbpp")
    assert MBPP_PLUS_VERSION == pin.version
    cache_file = Path(CACHE_DIR) / f"MbppPlus-{MBPP_PLUS_VERSION}.jsonl"
    if not cache_file.exists() and not os.environ.get("MBPP_OVERRIDE_PATH"):
        pytest.skip(f"MBPP+ problem cache not seeded at {cache_file} (no network in tests)")
    problems = er.load_problems("mbpp")
    assert frozenset(problems) == pin.task_ids
    assert all("entry_point" in p for p in problems.values())


def test_the_malformed_digest_rule_is_identical_in_the_writer_and_the_scanner(sandbox_mod):
    """The digest rule now exists TWICE — the writer's
    ``cell_controls._valid_payload_digest`` and the scanner's torch-free
    ``_PAYLOAD_DIGEST_RE`` restatement, which cannot import it (that module pulls
    torch through ``sisteer.methods``; the orchestrator image is stdlib + pyyaml).
    A duplicated invariant is the shape that drifts, and this campaign has already
    been bitten by a check carrying its own copy of the rule it validated. So drive
    ONE table through BOTH and require identical verdicts: the fix is right today
    and, without this, unprotected tomorrow."""
    from sisteer.capeval.tinycap import contracts as tc
    from sisteer.eval.cell_controls import _valid_payload_digest

    n = tc.PAYLOAD_DIGEST_HEX_LEN
    good = "0" * n
    cases = [
        (good, True),                       # the canonical digest the writer files
        ("a1b2c3d4e5f60718"[:n].ljust(n, "f"), True),
        (None, False),                      # the explicit null the loader refuses
        ("", False),
        ("0" * (n - 1), False),             # one short
        ("0" * (n + 1), False),             # one long
        ("0" * (n - 1) + "g", False),       # non-hex
        (("0" * (n - 1) + "A"), False),     # uppercase — the writer files lowercase
        ("0" * n + " ", False),             # trailing whitespace
        (" " + "0" * n, False),             # leading whitespace
        ("0" * n + "\n", False),            # a newline fullmatch must not swallow
        (123, False),                       # non-str
        (True, False),
        ([good], False),
    ]
    scanner_ok = sandbox_mod._valid_payload_digest   # the function every scanner call site uses
    disagreements = [(v, _valid_payload_digest(v), scanner_ok(v))
                     for v, _ in cases if _valid_payload_digest(v) != scanner_ok(v)]
    assert not disagreements, (
        "the writer and the scanner disagree about what a payload digest is: "
        + ", ".join(f"{v!r}: writer={w} scanner={s}" for v, w, s in disagreements))
    for value, expected in cases:
        assert _valid_payload_digest(value) is expected, f"writer on {value!r}"
        assert scanner_ok(value) is expected, f"scanner on {value!r}"
    # and no scanner call site reaches the regex directly (a site drifting to `.match` would pass the
    # regex-object parity while reading a 17-char / newline-terminated value as a digest): the regex
    # is used exactly once, inside the function both sites call
    src = (REPO / "scripts" / "modal" / "tinycap_sandbox.py").read_text()
    assert src.count("_PAYLOAD_DIGEST_RE.") == 1 and src.count("_valid_payload_digest(") >= 3


# --------------------------------------------------------------------------- #
# PW-sampling (U8 item 1): k seeded samples per problem through the served    #
# EvalPlus leg, the unbiased pass@1 beside the per-sample rows                 #
# --------------------------------------------------------------------------- #

def _is_sample_row(metric: str) -> bool:
    """A per-SAMPLE `acc@<i>` row. Not `C.SAMPLE_METRIC_SEP in metric`:
    the separator is "@", which every `pass@1_*` metric also contains."""
    return metric.startswith(er.GATING_ROW_METRIC + C.SAMPLE_METRIC_SEP)


GEMMA_ROW = C.SAMPLING_BY_MODEL_KEY["gemma4e4b"]      # 1.0 / 0.95 / 64
LLAMA_ROW = C.SAMPLING_BY_MODEL_KEY["llama8b"]        # 0.6 / 0.9 / no top_k
FOUR_KEYS = ["HumanEval/0/0", "HumanEval/0/1", "HumanEval/1/0", "HumanEval/1/1"]


def _results_k(k=2, *, base=(("pass", "pass"), ("fail", "fail")), plus=(("pass", "fail"), ("fail", "fail")),
               ids=("HumanEval/0", "HumanEval/1"), hash_="1" * 32):
    """An evalplus results file with k samples per task: ``base[i][j]`` /
    ``plus[i][j]`` are task i's sample j statuses."""
    return {"date": "2026-09-06 00:00", "hash": hash_,
            "eval": {t: [{"task_id": t, "solution": f"s{j}", "base_status": base[i][j],
                          "plus_status": plus[i][j], "base_fail_tests": [], "plus_fail_tests": []}
                         for j in range(k)]
                     for i, t in enumerate(ids)}}


def test_the_engine_body_carries_the_row_and_k_items_have_distinct_seeds():
    """The engine's SamplingRow goes onto EVERY body — temperature always,
    top_p / top_k only when the row sets them (None = no key); k ChatItems of
    one problem carry distinct sample_idx and hence distinct seeds
    (``stable_seed(task, item, idx)``); results and errors are filed per
    (item, sample); a duplicate (item, sample) in one fan-out refuses."""
    eng = _engine(LENS_ROUTE, transport=FakeTransport(), sampling=GEMMA_ROW)
    items = er.render_chat_items(PROBLEMS, samples_per_item=2)
    assert [(i.item_id, i.sample_idx) for i in items] == [
        ("HumanEval/0", 0), ("HumanEval/0", 1), ("HumanEval/1", 0), ("HumanEval/1", 1)]
    bodies = [eng.build_request(i, max_tokens=10) for i in items]
    assert all((b["temperature"], b["top_p"], b["top_k"]) == (1.0, 0.95, 64) for b in bodies)
    seeds = [b["seed"] for b in bodies]
    assert len(set(seeds)) == 4
    assert seeds[0] == pe.stable_seed(eng.task, "HumanEval/0", "0")
    assert seeds[1] == pe.stable_seed(eng.task, "HumanEval/0", "1")
    # no row = greedy, the pre-U8 body; a row without top_k sends none
    b = _engine(LENS_ROUTE, transport=FakeTransport()).build_request(items[0], max_tokens=10)
    assert b["temperature"] == 0.0 and "top_p" not in b and "top_k" not in b
    b = _engine(LENS_ROUTE, transport=FakeTransport(), sampling=LLAMA_ROW).build_request(items[0], max_tokens=10)
    assert (b["temperature"], b["top_p"]) == (0.6, 0.9) and "top_k" not in b
    with pytest.raises(pe.ProxyEngineError, match="SamplingRow"):
        _engine(LENS_ROUTE, sampling={"temperature": 1.0})
    # the fan-out files one result per generation, keyed <item>/<idx>
    results, errors = eng.generate_items(items, GEN_CFG)
    assert errors == [] and sorted(results) == FOUR_KEYS
    assert results["HumanEval/0/1"].sample_idx == 1 and results["HumanEval/0/1"].seed == seeds[1]
    assert pe.generation_key("HumanEval/0", 1) == "HumanEval/0/1" == C.sample_header_value("HumanEval/0", 1)
    with pytest.raises(pe.ProxyEngineError, match="duplicate \\(item_id, sample_idx\\)"):
        eng.generate_items(items + items[:1], GEN_CFG)
    # an error names the GENERATION; the headers name it the same way
    tr = FakeTransport(fail={"HumanEval/1/1"})
    res, errs = _engine(LENS_ROUTE, transport=tr, sampling=GEMMA_ROW).generate_items(items, GEN_CFG)
    assert sorted(res) == FOUR_KEYS[:3]
    assert [(e.item_id, e.sample_idx, e.kind) for e in errs] == [("HumanEval/1", 1, pe.ERR_HTTP)]
    assert sorted(c[1][C.HEADER_SAMPLE] for c in tr.calls) == FOUR_KEYS
    # the Engine seam returns one GenOut per (item, sample), in item order
    outs = _engine(LENS_ROUTE, transport=FakeTransport(), sampling=GEMMA_ROW).generate(items, None, GEN_CFG)
    assert len(outs) == 4 and all(o.ended_with_eos for o in outs)


def test_k_samples_per_problem_generate_resume_at_the_same_k_and_write_k_lines_per_task(
        tmp_path, offline_runner):
    """The served generation at k = 2: 4 SEPARATE requests (never n), one
    sidecar record per generation, a failed sample reruns ALONE on resume, a
    resume at another k refuses (k is identity), samples.jsonl carries k
    lines per task_id in problem-major order (evalplus's own multi-sample
    shape), gen_meta records k, the row and the generation count, and a
    smoke-padded id gets k empty solutions."""
    cell = tmp_path / "cells" / "k"
    tr = FakeTransport(fail={"HumanEval/1/1"})
    eng = _engine(LENS_ROUTE, transport=tr, sampling=GEMMA_ROW)
    with pytest.raises(er.EvalPlusSeamError, match="1 of 4 items FAILED"):
        er.generate_samples_served(engine=eng, dataset="humaneval", cell_dir=cell, problems=PROBLEMS,
                                   expected=TWO_IDS, gen_cfg=GEN_CFG, samples_per_item=2)
    tdir = er.served_task_dir(cell, "humaneval")
    side = er.read_generations(tdir / er.GENERATIONS_FILENAME)
    assert sorted(side) == FOUR_KEYS[:3]
    assert side["HumanEval/0/1"]["sample_idx"] == 1
    assert side["HumanEval/0/1"]["seed"] != side["HumanEval/0/0"]["seed"]
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "incomplete" and meta["samples_per_item"] == 2
    assert meta["n_generations"] == 4 and meta["n_generated"] == 3 and meta["n_items"] == 2
    assert meta["sampling"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "source": GEMMA_ROW.source}
    [e] = meta["errors"]
    assert (e["item_id"], e["sample_idx"], e["kind"]) == ("HumanEval/1", 1, "http_error")
    assert all(b["temperature"] == 1.0 and b["top_k"] == 64 for _, _, b in tr.calls)
    # a resume at another k is refused — k is identity, never silently re-shaped
    with pytest.raises(er.EvalPlusSeamError, match="records samples_per_item=2, this run asks 1"):
        er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=FakeTransport(), sampling=GEMMA_ROW),
                                   dataset="humaneval", cell_dir=cell, problems=PROBLEMS, expected=TWO_IDS,
                                   gen_cfg=GEN_CFG, samples_per_item=1)
    # the gap only: one request, the failed generation
    tr2 = FakeTransport()
    summary = er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=tr2, sampling=GEMMA_ROW),
                                         dataset="humaneval", cell_dir=cell, problems=PROBLEMS,
                                         expected=TWO_IDS, gen_cfg=GEN_CFG, samples_per_item=2)
    assert [c[1][C.HEADER_SAMPLE] for c in tr2.calls] == ["HumanEval/1/1"]
    samples = er.read_samples_jsonl(tdir / er.SAMPLES_FILENAME, expected_task_ids=TWO_IDS, samples_per_item=2)
    assert [s_["task_id"] for s_ in samples] == ["HumanEval/0", "HumanEval/0", "HumanEval/1", "HumanEval/1"]
    with pytest.raises(er.EvalPlusSeamError, match="duplicate task_ids"):
        er.read_samples_jsonl(tdir / er.SAMPLES_FILENAME, expected_task_ids=TWO_IDS)          # k = 1 view
    with pytest.raises(er.EvalPlusSeamError, match="exactly 3 samples"):
        er.read_samples_jsonl(tdir / er.SAMPLES_FILENAME, expected_task_ids=TWO_IDS, samples_per_item=3)
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    assert meta["status"] == "complete" and meta["samples_per_item"] == 2 and meta["n_generated"] == 4
    assert meta["item_ids"] == list(PROBLEMS) and meta["n_requests"] == 5 and meta["seed_rule"] == C.SEED_RULE_EVALPLUS
    assert summary["samples_per_item"] == 2 and summary["n_generations"] == 4 and summary["n"] == 2
    assert summary["n_requests"] == 5
    # a third call requests nothing
    tr3 = FakeTransport()
    er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=tr3, sampling=GEMMA_ROW), dataset="humaneval",
                               cell_dir=cell, problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG,
                               samples_per_item=2)
    assert tr3.calls == []
    # smoke padding: k empty solutions per padded id, the file uniform
    cell2 = tmp_path / "cells" / "smoke"
    er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=FakeTransport(), sampling=GEMMA_ROW),
                               dataset="humaneval", cell_dir=cell2, problems=PROBLEMS, expected=TWO_IDS,
                               gen_cfg=GEN_CFG, limit=1, samples_per_item=2)
    padded = er.read_samples_jsonl(er.served_task_dir(cell2, "humaneval") / er.SAMPLES_FILENAME,
                                   expected_task_ids=TWO_IDS, samples_per_item=2)
    assert [(s_["task_id"], s_["solution"] == "") for s_ in padded] == [
        ("HumanEval/0", False), ("HumanEval/0", False), ("HumanEval/1", True), ("HumanEval/1", True)]
    with pytest.raises(er.EvalPlusSeamError, match="samples_per_item must be >= 1"):
        er.render_chat_items(PROBLEMS, samples_per_item=0)


def test_pass_at_k_unbiased_and_the_per_item_estimator_rows():
    """The Chen et al. unbiased pass@k, at k = 1 exactly c / n; with k
    samples per task the converter files a per-sample `acc@<i>` row per
    generation and per-item `pass@1_base` / `pass@1_plus` / `acc` rows that
    carry the estimator — named in their extra_json with the per-sample
    values and n_correct. A k = 1 call keeps the pre-U8 rows byte for byte
    unless the caller asks for per-sample rows (R16-A / E45 M-EvalPlus), in
    which case the lone draw gets its own `acc@0` row too; the DRAW's
    provenance is joined by the store conversion, which has the sidecars."""
    assert er.pass_at_k_unbiased(4, 1, 1) == 0.25 and er.pass_at_k_unbiased(4, 4, 1) == 1.0
    assert er.pass_at_k_unbiased(4, 0, 1) == 0.0 and er.pass_at_k_unbiased(1, 1, 1) == 1.0
    assert er.pass_at_k_unbiased(5, 2, 3) == pytest.approx(0.9) and er.pass_at_k_unbiased(3, 3, 2) == 1.0
    assert er.pass_at_k_unbiased(10, 3, 1) == pytest.approx(0.3)
    for n, c, k in ((0, 0, 1), (4, 5, 1), (4, 1, 5), (4, -1, 1), (4, 1, 0)):
        with pytest.raises(ValueError, match="pass@k needs"):
            er.pass_at_k_unbiased(n, c, k)
    assert "c/n" in er.PASS_AT_1_ESTIMATOR and "Chen" in er.PASS_AT_1_ESTIMATOR
    results = _results_k(2, base=(("pass", "pass"), ("pass", "fail")), plus=(("pass", "fail"), ("pass", "pass")))
    rows = er.eval_results_to_rows(results, TWO_IDS, task="evalplus/humaneval_plus",
                                   domain=C.DOMAIN_EVALPLUS_SERVED, task_version=C.EVALPLUS_TASK_VERSION,
                                   samples_per_item=2)
    by = {(r["item_id"], r["metric"]): r for r in rows}
    assert sorted(m for (i, m) in by if i == "HumanEval/0") == ["acc", "acc@0", "acc@1", "pass@1_base", "pass@1_plus"]
    assert len(rows) == 10
    # per-sample rows: the plus definition per sample, its base value beside it
    assert by[("HumanEval/0", "acc@0")]["value"] == 1.0 and by[("HumanEval/0", "acc@1")]["value"] == 0.0
    e = json.loads(by[("HumanEval/0", "acc@1")]["extra_json"])
    assert e == {"metric_def": "base_and_plus", "sample_idx": 1, "source_metric": "pass@1_plus", "value_base": 1.0}
    # HumanEval/1 sample 1 is base-fail / plus-pass -> NOT a plus pass (evalplus's headline definition)
    assert by[("HumanEval/1", "acc@1")]["value"] == 0.0 and json.loads(by[("HumanEval/1", "acc@1")]["extra_json"])["value_base"] == 0.0
    # per-item: the unbiased pass@1 = c / n, named
    assert by[("HumanEval/0", "pass@1_plus")]["value"] == 0.5 == by[("HumanEval/0", "acc")]["value"]
    assert by[("HumanEval/0", "pass@1_base")]["value"] == 1.0
    assert by[("HumanEval/1", "pass@1_plus")]["value"] == 0.5 and by[("HumanEval/1", "pass@1_base")]["value"] == 0.5
    e = json.loads(by[("HumanEval/0", "pass@1_plus")]["extra_json"])
    assert e == {"metric_def": "base_and_plus", "samples_per_item": 2, "estimator": er.PASS_AT_1_ESTIMATOR,
                 "values": [1.0, 0.0], "n_correct": 1}
    e = json.loads(by[("HumanEval/0", "pass@1_base")]["extra_json"])
    assert e == {"samples_per_item": 2, "estimator": er.PASS_AT_1_ESTIMATOR, "values": [1.0, 1.0], "n_correct": 2}
    e = json.loads(by[("HumanEval/1", "acc")]["extra_json"])
    assert e["source_metric"] == "pass@1_plus" and e["values"] == [1.0, 0.0] and e["n_correct"] == 1
    assert {r["task_version"] for r in rows} == {C.evalplus_row_task_version(er.PLUS_METRIC_DEF)}
    # the sample count is exact, never averaged over whatever came back
    with pytest.raises(ValueError, match="exactly 2 sample"):
        er.eval_results_to_rows(_results(), TWO_IDS, samples_per_item=2)
    with pytest.raises(ValueError, match="exactly 1 sample"):
        er.eval_results_to_rows(results, TWO_IDS)
    # k = 1 rows are the pre-U8 rows exactly (no estimator keys, no per-sample row)
    one = er.eval_results_to_rows(_results(), TWO_IDS)
    assert len(one) == 6 and all(r["extra_json"] is None for r in one if r["metric"] == "pass@1_base")
    assert not [r for r in one if _is_sample_row(r["metric"])]
    plus_row = next(r for r in one if r["metric"] == "pass@1_plus")
    assert json.loads(plus_row["extra_json"]) == {"metric_def": "base_and_plus"}
    # ... but asked for per-sample rows, the lone draw files acc@0 too (R16-A): the
    # per-item rows do not move, so the k = 1 estimator stays the value it always was
    one_s = er.eval_results_to_rows(_results(), TWO_IDS, per_sample_rows=True)
    assert len(one_s) == 8
    by1 = {(r["item_id"], r["metric"]): r for r in one_s}
    assert sorted(m for (i, m) in by1 if i == "HumanEval/0") == ["acc", "acc@0", "pass@1_base", "pass@1_plus"]
    assert by1[("HumanEval/0", "acc@0")]["value"] == 1.0 == by1[("HumanEval/0", "acc")]["value"]
    assert by1[("HumanEval/1", "acc@0")]["value"] == 0.0
    assert json.loads(by1[("HumanEval/0", "acc@0")]["extra_json"]) == {
        "metric_def": "base_and_plus", "sample_idx": 0, "source_metric": "pass@1_plus", "value_base": 1.0}
    for m in ("acc", "pass@1_base", "pass@1_plus"):
        assert by1[("HumanEval/0", m)] == {(r["item_id"], r["metric"]): r for r in one}[("HumanEval/0", m)]
    # validate_eval_results follows the same k
    er.validate_eval_results(results, expected_task_ids=TWO_IDS, samples_per_item=2)
    with pytest.raises(er.EvalPlusSeamError, match="exactly 1 sample"):
        er.validate_eval_results(results, expected_task_ids=TWO_IDS)
    with pytest.raises(er.EvalPlusSeamError, match="exactly 2 sample"):
        er.validate_eval_results(_results(), expected_task_ids=TWO_IDS, samples_per_item=2)


def test_convert_with_k_samples_files_per_sample_rows_the_estimator_and_the_flags(tmp_path, monkeypatch):
    """The conversion reads the task's k off gen_meta: 5 rows per item (2
    per-sample `acc@<i>` + the 3 per-item estimator rows), the summary's
    `samples_per_item` and the mean of the per-item pass@1, the done marker's
    n_rows; per-sample rows read their OWN generation's flag AND their own
    draw's provenance (R16-A: the sidecar seed + gen_meta's sampling row),
    per-item rows are flagged when any sample was; idempotent."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    results = _results_k(2, base=(("pass", "pass"), ("fail", "pass")), plus=(("pass", "fail"), ("fail", "pass")))
    row = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "source": GEMMA_ROW.source}
    seeds = {"HumanEval/0/0": 11, "HumanEval/0/1": 22,
             "HumanEval/1/0": 33, "HumanEval/1/1": 44}
    cell, tdir = _cell_dir(tmp_path, samples_per_item=2, results=results,
                           flagged=["HumanEval/1"], flagged_keys=["HumanEval/1/0"],
                           sampling=row, seeds=seeds)
    summary = er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    by = {(r["item_id"], r["metric"]): r for r in rows}
    assert by[("HumanEval/0", "acc")]["value"] == 0.5 and by[("HumanEval/1", "acc")]["value"] == 0.5
    assert by[("HumanEval/1", "pass@1_base")]["value"] == 0.5
    assert summary["samples_per_item"] == 2 and summary["pass_at_1_plus"] == 0.5 == summary["acc"]
    assert summary["pass_at_1_base"] == 0.75 and summary["n"] == 2
    assert set(summary) <= set(C.SUMMARY_PAYLOAD_FIELDS)
    from sisteer.eval.cell_controls import _assert_counts_only
    _assert_counts_only({"evalplus/humaneval_plus": summary})
    marker = json.loads(er.done_marker_path(cell, "evalplus/humaneval_plus").read_text())
    assert marker["n_rows"] == 10 and marker["n_items"] == 2
    flags = {(r["item_id"], r["metric"]): json.loads(r["extra_json"])["content_empty_with_reasoning"] for r in rows}
    assert flags[("HumanEval/1", "acc@0")] is True and flags[("HumanEval/1", "acc@1")] is False
    assert flags[("HumanEval/1", "acc")] is True and flags[("HumanEval/1", "pass@1_plus")] is True
    assert flags[("HumanEval/0", "acc@0")] is False and flags[("HumanEval/0", "acc")] is False
    # R16-A (E45 M-EvalPlus): the five provenance fields on every per-sample row —
    # sample_idx off the row, seed off the generations sidecar (per GENERATION),
    # temperature / top_p / top_k off gen_meta's sampling row (per task)
    prov = {(i, m): json.loads(r["extra_json"]) for (i, m), r in by.items() if _is_sample_row(m)}
    assert set(prov) == {("HumanEval/0", "acc@0"), ("HumanEval/0", "acc@1"),
                         ("HumanEval/1", "acc@0"), ("HumanEval/1", "acc@1")}
    for (item, metric), e in prov.items():
        assert e["sample_idx"] == int(metric.split(C.SAMPLE_METRIC_SEP)[1])
        assert (e["temperature"], e["top_p"], e["top_k"]) == (1.0, 0.95, 64)
        assert e["seed"] == seeds[f"{item}/{e['sample_idx']}"]
    assert prov[("HumanEval/0", "acc@1")]["seed"] == 22
    assert prov[("HumanEval/1", "acc@1")]["seed"] == 44
    # the per-ITEM rows are not per-sample and carry no draw of their own
    assert not any(k in json.loads(by[("HumanEval/0", "acc")]["extra_json"])
                   for k in ("seed", "temperature", "top_p", "top_k"))
    er.convert_eval_results_to_store(cell, dataset="humaneval")                     # idempotent
    assert len((cell / "rows.jsonl").read_text().splitlines()) == 10
    # a results file with one sample per task under a k = 2 meta refuses
    (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(_results()))
    with pytest.raises(er.EvalPlusSeamError, match="exactly 2 sample"):
        er.rederive_evalplus(cell, dataset="humaneval")
    # the off-volume re-derivation follows the same k
    (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(results))
    derived = er.rederive_evalplus(cell, dataset="humaneval")
    assert derived["summary"] == summary and len(derived["rows"]) == 10


def test_a_k1_store_under_the_sampling_contract_files_acc0_with_its_draws_provenance(tmp_path, monkeypatch):
    """R16-A (E45 M-EvalPlus): k = 1 is still A DRAW. A store generated under
    the sampling contract (its gen_meta carries the `sampling` block) files an
    `acc@0` row per problem beside the per-item pass@1 / acc rows, and every
    per-sample row carries the five provenance fields — sample_idx, seed,
    temperature, top_p, top_k — joined from the sidecars, so a row can say how
    it was drawn instead of leaving that in files nobody reads with it. The
    per-item mean `acc` and both pass@1 fields do not move. A PRE-sampling
    store (no `sampling` block: the retained 618f9dd conversions and every HF
    store) keeps exactly the three rows per item it was written under, here
    and in the off-volume re-derivation."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    row = {"temperature": 0.6, "top_p": 0.9, "top_k": None, "source": LLAMA_ROW.source}
    cell, tdir = _cell_dir(tmp_path / "current", sampling=row,
                           seeds={"HumanEval/0/0": 4242, "HumanEval/1/0": 777})
    summary = er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    by = {(r["item_id"], r["metric"]): r for r in rows}
    assert len(rows) == 8
    assert sorted(m for (i, m) in by if i == "HumanEval/0") == ["acc", "acc@0", "pass@1_base", "pass@1_plus"]
    # _results(): HumanEval/0 passes base and plus, HumanEval/1 fails both
    assert by[("HumanEval/0", "acc@0")]["value"] == 1.0 and by[("HumanEval/1", "acc@0")]["value"] == 0.0
    assert json.loads(by[("HumanEval/0", "acc@0")]["extra_json"]) == {
        "content_empty_with_reasoning": None, "metric_def": "base_and_plus", "sample_idx": 0,
        "seed": 4242, "source_metric": "pass@1_plus", "temperature": 0.6, "top_k": None,
        "top_p": 0.9, "value_base": 1.0}
    assert json.loads(by[("HumanEval/1", "acc@0")]["extra_json"])["seed"] == 777
    # the per-item rows are the numbers they always were, unchanged shape
    assert by[("HumanEval/0", "acc")]["value"] == 1.0 and by[("HumanEval/1", "acc")]["value"] == 0.0
    assert by[("HumanEval/0", "pass@1_base")]["value"] == 1.0
    assert json.loads(by[("HumanEval/0", "acc")]["extra_json"]) == {
        "content_empty_with_reasoning": None, "metric_def": "base_and_plus",
        "source_metric": "pass@1_plus"}
    assert summary["acc"] == 0.5 == summary["pass_at_1_plus"] and summary["samples_per_item"] == 1
    assert summary["pass_at_1_base"] == 0.5 and summary["n"] == 2
    assert json.loads(er.done_marker_path(cell, "evalplus/humaneval_plus").read_text())["n_rows"] == 8
    er.convert_eval_results_to_store(cell, dataset="humaneval")                     # idempotent
    assert len((cell / "rows.jsonl").read_text().splitlines()) == 8
    assert len(er.rederive_evalplus(cell, dataset="humaneval")["rows"]) == 8
    # E50 #8 INVERTS what a missing sidecar means under the contract: this used
    # to file the row with `seed: null` ("it cannot vouch for the seed, and says
    # so"), which publishes an unreproducible draw as a reproducible one. A
    # store DECLARING the sampling contract has a generator that writes one
    # record per (item, sample), so a missing one contradicts the declaration —
    # it now refuses, naming the rows
    (tdir / er.GENERATIONS_FILENAME).unlink()
    with pytest.raises(er.EvalPlusSeamError, match="no recorded seed") as ei:
        er.rederive_evalplus(cell, dataset="humaneval")
    assert "HumanEval/0/0" in str(ei.value)
    # the PRE-sampling store: no sampling block, so no per-sample row anywhere
    legacy, _ = _cell_dir(tmp_path / "legacy")
    legacy_summary = er.convert_eval_results_to_store(legacy, dataset="humaneval")
    legacy_rows = [json.loads(line) for line in (legacy / "rows.jsonl").read_text().splitlines()]
    assert len(legacy_rows) == 6 and not [r for r in legacy_rows if _is_sample_row(r["metric"])]
    assert legacy_summary == summary
    assert len(er.rederive_evalplus(legacy, dataset="humaneval")["rows"]) == 6


def test_run_served_threads_the_row_and_each_tasks_own_k(tmp_path, offline_runner, monkeypatch):
    pins = {d: er.DatasetPin(dataset=d, name=d, version="v", url="u", gz_filename=f"{d}.gz",
                             gz_bytes=1, gz_sha256="0" * 64, gz_md5="0" * 32,
                             override_env="X", task_ids=TWO_IDS) for d in C.EVALPLUS_DATASETS}
    monkeypatch.setattr(er, "dataset_pin", lambda d: pins[d])
    monkeypatch.setattr(er, "load_problems", lambda d="humaneval": dict(PROBLEMS))
    tr = FakeTransport()
    out = er.run_served(base_url="http://127.0.0.1:1", route=LENS_ROUTE, system=None, thinking="off",
                        concurrency=2, cell_dir=tmp_path / "cells" / "k", transport=tr, sampling=LLAMA_ROW,
                        samples_by_task={"evalplus/humaneval_plus": 2, "evalplus/mbpp_plus": 1})
    assert out["evalplus/humaneval_plus"]["samples_per_item"] == 2 and out["evalplus/mbpp_plus"]["samples_per_item"] == 1
    assert out["evalplus/humaneval_plus"]["n_generations"] == 4 and out["evalplus/mbpp_plus"]["n_generations"] == 2
    assert all((b["temperature"], b["top_p"]) == (0.6, 0.9) and "top_k" not in b for _, _, b in tr.calls)
    assert len(tr.calls) == 6
    he = json.loads((er.served_task_dir(tmp_path / "cells" / "k", "humaneval") / er.GEN_META_FILENAME).read_text())
    assert he["samples_per_item"] == 2 and he["sampling"]["temperature"] == 0.6
    # a task the map does not name refuses — never a default k
    with pytest.raises(er.EvalPlusSeamError, match="no samples-per-item entry"):
        er.run_served(base_url="http://127.0.0.1:1", route=LENS_ROUTE, system=None, thinking="off",
                      concurrency=2, cell_dir=tmp_path / "cells" / "k2", transport=FakeTransport(),
                      sampling=LLAMA_ROW, samples_by_task={"evalplus/humaneval_plus": 2})
    # no row, no map: the pre-U8 instrument (greedy, one sample)
    tr0 = FakeTransport()
    er.run_served(base_url="http://127.0.0.1:1", route=LENS_ROUTE, system=None, thinking="off",
                  concurrency=2, cell_dir=tmp_path / "cells" / "k0", transport=tr0, datasets=("humaneval",))
    assert len(tr0.calls) == 2 and all(b["temperature"] == 0.0 and "top_p" not in b for _, _, b in tr0.calls)


def test_sandbox_executes_k_samples_per_task_and_converts_them(sandbox_mod, staged, tmp_path, monkeypatch):
    """The orchestrator reads the task's k off gen_meta: samples.jsonl is
    validated at k lines per id, evalplus's k-item results validate and
    convert (5 rows per item), and the record names k."""
    cache, pin, archive = staged
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    job = _job(tmp_path, samples_per_item=2)
    tdir = Path(job["task_dir"])
    assert sandbox_mod.samples_per_item_of(tdir) == 2
    good = _results_k(2, hash_=pin.gz_md5)
    FakeSandbox.script = lambda fs: fs.__setitem__(C.SANDBOX_RESULTS_PATH, json.dumps(good).encode())
    rec = sandbox_mod.execute_one(job, sandbox_cls=FakeSandbox, app_obj="APP", image="IMG", cache_dir=str(cache))
    assert rec["samples_per_item"] == 2 and rec["n_samples"] == 4 and rec["converted"] is True
    rows = [json.loads(line) for line in (Path(job["cell_dir"]) / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 10 and rec["summary"]["samples_per_item"] == 2
    assert sandbox_mod.results_state(tdir, dataset="humaneval") == ("valid", None)
    # a one-sample results file against a k = 2 task is INVALID, never converted
    (tdir / er.EVAL_RESULTS_FILENAME).write_text(json.dumps(_results(hash_=pin.gz_md5)))
    state, why = sandbox_mod.results_state(tdir, dataset="humaneval")
    assert state == "invalid" and "exactly 2 sample" in why


# --------------------------------------------------------------------------- #
# E50 #8 — under the sampling contract, a per-sample row without its seed      #
#          REFUSES; permissive only for a PRE-SAMPLING store                   #
# --------------------------------------------------------------------------- #


def test_a_current_store_refuses_an_acc_at_i_row_whose_generation_has_no_seed(
        tmp_path, monkeypatch):
    """E50 #8: a `sampling` block in gen_meta is the store DECLARING it was
    generated under the contract, whose producer writes one generations.jsonl
    record per (item, sample) carrying the request's seed. So a missing record
    — or a record with no seed — is the store contradicting itself, not an
    unrecorded fact, and filing `acc@<i>` with `seed: null` would publish an
    unreproducible draw as a reproducible one. Both shapes refuse, naming the
    rows and the remedy."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    row = {"temperature": 0.6, "top_p": 0.9, "top_k": None, "source": LLAMA_ROW.source}
    # (a) the sidecar is missing one of the two generations
    cell, _ = _cell_dir(tmp_path / "partial", sampling=row,
                        seeds={"HumanEval/0/0": 4242})
    with pytest.raises(er.EvalPlusSeamError, match="no recorded seed") as ei:
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    msg = str(ei.value)
    assert "HumanEval/1/0" in msg and er.GENERATIONS_FILENAME in msg
    assert "evalplus/humaneval_plus" in msg
    assert not (cell / "rows.jsonl").exists()          # nothing was filed
    # (b) the record is there but carries no seed at all
    cell2, tdir2 = _cell_dir(tmp_path / "nullseed", sampling=row,
                             seeds={"HumanEval/0/0": 1, "HumanEval/1/0": 2})
    (tdir2 / er.GENERATIONS_FILENAME).write_text("".join(
        json.dumps({"item_id": t, "sample_idx": 0, "text": "s",
                    "finish_reason": "stop"}, sort_keys=True) + "\n"
        for t in TWO_IDS))
    with pytest.raises(er.EvalPlusSeamError, match="no recorded seed"):
        er.convert_eval_results_to_store(cell2, dataset="humaneval")


def test_the_sampling_contract_is_read_off_the_store_not_only_the_task(tmp_path, monkeypatch):
    """E54 #3: whether a task was generated under the sampling contract was
    decided from its OWN ``gen_meta.json`` alone, though both conversion paths
    already read the store's ``manifest.json`` — which carries the run's
    ``sampling`` block, written by the SAME driver run from the SAME row.

    So a cell whose parent declares the contract and whose task-local block is
    missing read as PRE-SAMPLING and, at k = 1, silently dropped the ``acc@0``
    row — taking E50 #8's seedless refusal with it, since that fires only on
    per-sample rows. It now refuses unless the store positively dates itself
    before the contract, and a task-local row that DISAGREES with the store's
    refuses on any era: two present statements contradicting each other is not
    an absence."""
    from sisteer.capeval.tinycap import contracts as tc

    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    row = {"temperature": 0.7, "top_p": 0.95, "top_k": 64, "source": "the model card"}
    parent = {"mode": tc.SAMPLING_CARD, "model_key": "m", "temperature": 0.7, "top_p": 0.95,
              "top_k": 64, "samples_by_task": {}, "seed_rule": er.SEED_RULE,
              "source": "the model card"}

    # the reviewer's cell: parent declares the contract, the task-local block is
    # missing, there is no generations.jsonl, k = 1 -> refuses instead of
    # dropping the acc@0 row
    cell, _tdir = _cell_dir(tmp_path / "torn", manifest_sampling=parent)
    with pytest.raises(er.EvalPlusSeamError) as ei:
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    why = str(ei.value)
    assert "E54 #3" in why and UNDECLARED_TREE_SHA in why and er.GEN_META_FILENAME in why
    assert not (cell / "rows.jsonl").exists()
    # the off-volume re-derivation reads the same way
    with pytest.raises(er.EvalPlusSeamError, match="E54 #3"):
        er.rederive_evalplus(cell, dataset="humaneval")

    # ... and would have dropped exactly the row the contract exists for
    cell, _tdir = _cell_dir(tmp_path / "whole", manifest_sampling=parent, sampling=row,
                            seeds={"HumanEval/0/0": 11, "HumanEval/1/0": 12})
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    per_sample = [json.loads(r["extra_json"]) for r in rows if _is_sample_row(r["metric"])]
    assert len(per_sample) == 2 and {e["seed"] for e in per_sample} == {11, 12}
    assert all(e["temperature"] == 0.7 for e in per_sample)

    # E55 #1 — the era is asked PER CONTRACT, and the two producers are named
    # explicitly rather than picked off a flat list. A tree that PREDATES the
    # sampling contract's landing keeps today's reading (no per-sample row at
    # k = 1, nothing moves); a tree that DESCENDS from it must carry the block,
    # so a deleted task-local block on such a store is damage and REFUSES. The
    # old test asserted the second store's behaviour was the first's, because
    # `min(HISTORICAL_PRODUCER_TREE_SHAS)` happened to be 087869ac — a
    # POST-sampling tree — and the flat allowlist called it historical anyway.
    hist = PRE_EVERYTHING_TREE_SHA
    assert tc.store_era({tc.MANIFEST_KEY_PRODUCER: {"tree_sha": hist}},
                        contract=tc.CONTRACT_SAMPLING)[0] == tc.STORE_ERA_HISTORICAL
    cell, _tdir = _cell_dir(tmp_path / "old", manifest_sampling=parent, producer=hist)
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    assert not [r for r in rows if _is_sample_row(r["metric"])]

    # the post-landing wave tree: its stores ARE under the sampling contract
    post = POST_SAMPLING_TREE_SHA
    assert tc.store_era({tc.MANIFEST_KEY_PRODUCER: {"tree_sha": post}},
                        contract=tc.CONTRACT_SAMPLING)[0] == tc.STORE_ERA_CURRENT
    cell, _tdir = _cell_dir(tmp_path / "postwave", manifest_sampling=parent, producer=post)
    with pytest.raises(er.EvalPlusSeamError) as ei:
        er.convert_eval_results_to_store(cell, dataset="humaneval")
    why = str(ei.value)
    assert "E54 #3" in why and post in why and "sampling" in why
    assert not (cell / "rows.jsonl").exists()
    # and WITH its task-local block it converts, filing the row the contract
    # exists for — which is what every store of the 57-cell wave looks like
    cell, _tdir = _cell_dir(tmp_path / "postwave_ok", manifest_sampling=parent, producer=post,
                            sampling=row, seeds={"HumanEval/0/0": 5, "HumanEval/1/0": 6})
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    assert len([r for r in rows if _is_sample_row(r["metric"])]) == 2

    # a DISAGREEMENT between the two present rows refuses whatever the era,
    # naming both sides and the field that moved
    for n, tree in enumerate((UNDECLARED_TREE_SHA, hist)):
        cell, _tdir = _cell_dir(tmp_path / f"clash{n}", producer=tree, manifest_sampling=parent,
                                sampling={**row, "temperature": 0.2})
        with pytest.raises(er.EvalPlusSeamError) as ei:
            er.convert_eval_results_to_store(cell, dataset="humaneval")
        assert "temperature" in str(ei.value) and "0.2" in str(ei.value) and "0.7" in str(ei.value)

    # a parent that says NOTHING leaves the task-local block governing, as
    # before — nothing is waived on an absence, the local block is the positive
    # statement either way
    cell, _tdir = _cell_dir(tmp_path / "silent", sampling=row,
                            seeds={"HumanEval/0/0": 3, "HumanEval/1/0": 4})
    meta = json.loads((er.served_task_dir(cell, "humaneval") / er.GEN_META_FILENAME).read_text())
    manifest = json.loads((cell / "manifest.json").read_text())
    assert er._records_sampling(meta, manifest, where=str(cell)) is True


def test_a_pre_sampling_store_still_files_a_null_seed(tmp_path, monkeypatch):
    """The permissive half, and the reason the refusal keys on the BLOCK rather
    than on the seed being null: a store written before the sampling contract
    has no `sampling` block, its sidecars predate the rule, and its rows must
    not move. At k > 1 such a store still emits per-sample rows, and those file
    `seed: null` exactly as they always did."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    results = _results_k(2, base=(("pass", "pass"), ("fail", "pass")),
                         plus=(("pass", "fail"), ("fail", "pass")))
    cell, _ = _cell_dir(tmp_path / "legacy", samples_per_item=2, results=results)
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    per_sample = [json.loads(r["extra_json"]) for r in rows if _is_sample_row(r["metric"])]
    assert per_sample and all(e["seed"] is None for e in per_sample)
    # ...and none of them can vouch for the draw either, which is the same fact
    assert all(e["temperature"] is None for e in per_sample)
    assert not er._records_sampling(
        json.loads((er.served_task_dir(cell, "humaneval") / er.GEN_META_FILENAME).read_text()),
        json.loads((cell / "manifest.json").read_text()), where=str(cell))


def test_the_generation_names_the_bytes_it_wrote_and_the_conversion_carries_them(
        tmp_path, offline_runner, monkeypatch):
    """E55 #2(a). The binding used to live only in ``execution.json``, inside
    the task directory — so it died with that directory. It now STARTS at the
    generation, which stamps the sha256 of the exact ``samples.jsonl`` it wrote
    into its own ``gen_meta.json``, and the CONVERSION carries that digest into
    the task's ``summary/<task>.json`` payload and its ``done/`` marker, both
    of which live outside ``evalplus/<dataset>/``.

    The sandbox's ``execution.json`` must then AGREE with it: two records of
    one generation naming different bytes means the sandbox scored a
    generation this task's own record does not describe."""
    cell = tmp_path / "sweep" / "cells" / "arm__c__t0__a0__aaaaaaaa__bbbbbbbb"
    (cell).mkdir(parents=True)
    er.generate_samples_served(engine=_engine(LENS_ROUTE, transport=FakeTransport()),
                               dataset="humaneval", cell_dir=cell,
                               problems=PROBLEMS, expected=TWO_IDS, gen_cfg=GEN_CFG)
    tdir = er.served_task_dir(cell, "humaneval")
    meta = json.loads((tdir / er.GEN_META_FILENAME).read_text())
    written = hashlib.sha256((tdir / er.SAMPLES_FILENAME).read_bytes()).hexdigest()
    assert meta[er.SAMPLES_DIGEST_FIELD] == written

    # the conversion carries it into the payload and the marker
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    conv, tdir2 = _cell_dir(tmp_path / "conv")
    m2 = json.loads((tdir2 / er.GEN_META_FILENAME).read_text())
    digest = hashlib.sha256((tdir2 / er.SAMPLES_FILENAME).read_bytes()).hexdigest()
    m2[er.SAMPLES_DIGEST_FIELD] = digest
    (tdir2 / er.GEN_META_FILENAME).write_text(json.dumps(m2))
    summary = er.convert_eval_results_to_store(conv, dataset="humaneval")
    assert summary[er.SAMPLES_DIGEST_FIELD] == digest
    assert set(summary) <= set(C.SUMMARY_PAYLOAD_FIELDS)
    from sisteer.eval.cell_controls import _assert_counts_only
    _assert_counts_only({EVALPLUS_TASK: summary})          # a digest is a hash, not prose
    marker = json.loads(er.done_marker_path(conv, EVALPLUS_TASK).read_text())
    assert marker[er.SAMPLES_DIGEST_FIELD] == digest
    assert set(marker) == set(C.DONE_MARKER_FIELDS) | {er.SAMPLES_DIGEST_FIELD}
    assert json.loads((conv / C.SUMMARY_DIR / "evalplus__humaneval_plus.json").read_text()) == summary

    # the sandbox must have scored the bytes the generation named
    clash, tdir3 = _cell_dir(tmp_path / "clash")
    m3 = json.loads((tdir3 / er.GEN_META_FILENAME).read_text())
    m3[er.SAMPLES_DIGEST_FIELD] = "0" * 64
    (tdir3 / er.GEN_META_FILENAME).write_text(json.dumps(m3))
    with pytest.raises(er.EvalPlusSeamError) as ei:
        er.convert_eval_results_to_store(clash, dataset="humaneval")
    assert "E55 #2" in str(ei.value) and "0" * 64 in str(ei.value)
    assert not (clash / "rows.jsonl").exists()

    # a store that predates the digest converts as it always did, with no field
    old, _t = _cell_dir(tmp_path / "old", producer=PRE_EVERYTHING_TREE_SHA, binding="unbound")
    summary = er.convert_eval_results_to_store(old, dataset="humaneval")
    assert er.SAMPLES_DIGEST_FIELD not in summary
    assert set(json.loads(er.done_marker_path(old, EVALPLUS_TASK).read_text())) == set(
        C.DONE_MARKER_FIELDS)


def test_a_current_sandbox_that_recorded_no_digest_is_damage_on_any_generation(tmp_path):
    """E55 #2(c). An EvalPlus store has TWO producers and each binding fact is
    judged by ITS OWN producer's contract. The GENERATION's age excuses only
    the generation's own silence; if the EXECUTION record positively says it
    was written under the samples-digest contract and recorded no digest, that
    is damage whatever the generation's age — an old generation re-executed by
    a CURRENT sandbox must carry the digest.

    The converse must stay true or every store on the volume refuses: an
    execution record that names NO writer (every one written before this) can
    never grant an exemption and never revoke one."""
    from sisteer.capeval.tinycap import contracts as tc

    old = PRE_EVERYTHING_TREE_SHA
    # a pre-digest generation whose execution record says nothing: unbound,
    # stated, exactly as before — this is the whole volume today
    cell, tdir = _cell_dir(tmp_path / "silent", producer=old, binding="unbound")
    manifest = json.loads((cell / "manifest.json").read_text())
    assert er.samples_binding_of_task(tdir, manifest, task=EVALPLUS_TASK) == (
        None, tc.SAMPLES_BINDING_HISTORICAL_UNBOUND)

    # the SAME store, re-executed by a sandbox that dates itself under the
    # contract and still recorded no digest: damage, refused, named
    (tdir / er.EXECUTION_FILENAME).write_text(json.dumps(
        {"wall_clock_s": 7.5, tc.MANIFEST_KEY_CONTRACT_VERSION: tc.CONTRACT_VERSION,
         tc.MANIFEST_KEY_PRODUCER: {"driver": tc.PRODUCER_DRIVER_SANDBOX}}))
    with pytest.raises(er.EvalPlusSeamError) as ei:
        er.samples_binding_of_task(tdir, manifest, task=EVALPLUS_TASK)
    assert "E55 #2c" in str(ei.value) and "current sandbox" in str(ei.value)

    # a sandbox that dates itself BEFORE the contract is excused like the store
    (tdir / er.EXECUTION_FILENAME).write_text(json.dumps(
        {"wall_clock_s": 7.5, tc.MANIFEST_KEY_CONTRACT_VERSION:
             tc.CONTRACTS[tc.CONTRACT_SAMPLES_DIGEST].stamped_from - 1}))   # pre-digest, by intent
    assert er.samples_binding_of_task(tdir, manifest, task=EVALPLUS_TASK) == (
        None, tc.SAMPLES_BINDING_HISTORICAL_UNBOUND)

    # and the sandbox NEVER grants an exemption the generation did not: a
    # pre-digest sandbox on a store that cannot date itself still refuses
    cell2, tdir2 = _cell_dir(tmp_path / "undated", producer=UNDECLARED_TREE_SHA, binding="unbound")
    (tdir2 / er.EXECUTION_FILENAME).write_text(json.dumps(
        {"wall_clock_s": 7.5, tc.MANIFEST_KEY_CONTRACT_VERSION:
             tc.CONTRACTS[tc.CONTRACT_SAMPLES_DIGEST].stamped_from - 1}))   # pre-digest, by intent
    with pytest.raises(er.EvalPlusSeamError, match="E54 #1"):
        er.samples_binding_of_task(tdir2, json.loads((cell2 / "manifest.json").read_text()),
                                   task=EVALPLUS_TASK)


def test_the_sandbox_stamps_its_own_contract_version_into_the_execution_record():
    """E55 #2c: the sandbox's orchestrator image mounts the ``sisteer``
    PACKAGE, not a tree, so it has no ``TREE_SHA`` to record — the
    ``contract_version`` constant compiled into the mounted code is the
    execution producer's positive marker, and ``store_era`` reads the same two
    keys off an execution record as off a cell manifest. Pinned by source,
    because the sandbox module cannot be imported here (it imports modal)."""
    src = (REPO / "scripts" / "modal" / "tinycap_sandbox.py").read_text()
    assert src.count("C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION") == 2   # both stages
    assert src.count('C.MANIFEST_KEY_PRODUCER: {"driver": C.PRODUCER_DRIVER_SANDBOX}') == 2
    assert C.PRODUCER_DRIVER_SANDBOX == "tinycap_sandbox"


# --------------------------------------------------------------------------- #
# E60 (R22) — the additive RE-BIND of the samples digest                       #
# --------------------------------------------------------------------------- #


def _snapshot(root: Path) -> dict:
    """Every file under ``root``, by relative path, as BYTES — the instrument
    a "nothing else moved" claim needs (a key-set comparison would miss a
    reordered or reformatted file)."""
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _converted_unbound_store(tmp_path, **kw):
    """A store in the state the 36 EvalPlus tasks on the volume are actually
    in: CONVERTED (rows, both summaries, a done marker), its ``execution.json``
    naming the samples digest (E50 #4 landed), and its three DURABLE files
    carrying none (E55 #2 had not). Built by converting and then stripping the
    field, because that is precisely what the earlier converter's output was.
    Returns ``(cell_dir, task_dir, digest, relpaths)``."""
    cell, tdir = _cell_dir(tmp_path, **kw)
    er.convert_eval_results_to_store(cell, dataset="humaneval")
    rels = {
        "task_summary": str((tdir / er.SUMMARY_FILENAME).relative_to(cell)),
        "store_summary": str(Path(C.SUMMARY_DIR) / C.marker_filename(EVALPLUS_TASK)),
        "done_marker": str(Path(C.DONE_DIR) / C.marker_filename(EVALPLUS_TASK)),
    }
    digest = None
    for rel in rels.values():
        doc = json.loads((cell / rel).read_text())
        digest = doc.pop(er.SAMPLES_DIGEST_FIELD)
        completion.atomic_write_json(cell / rel, doc)
    return cell, tdir, digest, rels


def test_rebind_writes_the_samples_digest_and_preserves_every_other_byte(
        tmp_path, monkeypatch, sandbox_mod, launcher_mod):
    """E60. The design gap: CONVERSION IS GATED ON THE ABSENCE OF THE DONE
    MARKER AND THE BINDING IS WRITTEN BY CONVERSION, so ``samples_sha256``
    became unreachable for every already-converted task the moment it landed
    (O60: no flag exists, and the two ways in are inventing one or deleting the
    markers). This mode is the designed way back in for that ONE field.

    Executed end to end, not pinned by source: the dry run states the verdict
    and the exact diff and writes NOTHING; the real run adds the digest to the
    three durable files and ``rebound_at`` to the marker, leaves ``finished_at``
    and every other byte in the store alone, keeps the task DONE and the
    launcher's marker audit happy, and is idempotent (a second pass is a
    no-op)."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, _tdir, digest, rels = _converted_unbound_store(tmp_path)
    before = _snapshot(cell)
    assert er.SAMPLES_DIGEST_FIELD not in json.loads(before[rels["done_marker"]])
    # the launcher's marker audit reads THIS store the same way before and
    # after (the synthetic fixture's 2 items are not the frozen 164, so it has
    # a standing complaint; what matters is that the re-bound key adds none)
    audit_before = launcher_mod.invalid_done_marker_reason(cell, EVALPLUS_TASK)

    # (a) the DRY RUN: one verdict, the expected diff, and not one byte written
    dry = sandbox_mod.run_rebind(tmp_path, dry_run=True)
    assert dry["counts"] == {sandbox_mod.REBIND_REBIND: 1, sandbox_mod.REBIND_NOOP: 0,
                             sandbox_mod.REBIND_REFUSED: 0, sandbox_mod.REBIND_SKIPPED: 0}
    [plan] = dry["tasks"]
    assert plan["verdict"] == sandbox_mod.REBIND_REBIND
    assert plan[er.SAMPLES_DIGEST_FIELD] == digest
    assert _snapshot(cell) == before
    added = [ln for ln in plan["diff"].splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln for ln in plan["diff"].splitlines()
               if ln.startswith("-") and not ln.startswith("---")]
    assert removed == []                        # a re-bind never takes anything away
    assert len(added) == 4                      # the digest ×3, plus rebound_at on the marker
    assert sum(1 for ln in added if er.SAMPLES_DIGEST_FIELD in ln and digest in ln) == 3
    assert sum(1 for ln in added if C.DONE_MARKER_REBOUND_AT_FIELD in ln) == 1

    # the real pass
    real = sandbox_mod.run_rebind(tmp_path, dry_run=False)
    assert real["counts"][sandbox_mod.REBIND_REBIND] == 1
    assert sorted(real["tasks"][0]["written"]) == sorted(
        str(cell / rels[k]) for k in ("task_summary", "store_summary", "done_marker"))
    after = _snapshot(cell)
    assert set(after) == set(before)            # no file created, none removed
    assert sorted(k for k in before if before[k] != after[k]) == sorted(rels.values())

    for kind in ("task_summary", "store_summary"):
        rel = rels[kind]
        assert json.loads(after[rel]) == {**json.loads(before[rel]),
                                          er.SAMPLES_DIGEST_FIELD: digest}
    marker_before, marker_after = (json.loads(before[rels["done_marker"]]),
                                   json.loads(after[rels["done_marker"]]))
    assert set(marker_after) == set(marker_before) | {er.SAMPLES_DIGEST_FIELD,
                                                      C.DONE_MARKER_REBOUND_AT_FIELD}
    assert marker_after[er.SAMPLES_DIGEST_FIELD] == digest
    assert isinstance(marker_after[C.DONE_MARKER_REBOUND_AT_FIELD], float)
    # `finished_at` is PRESERVED byte for byte: a re-bind did not re-finish the
    # task, and the conversion's own marker write (time.time()) is discarded
    assert {k: v for k, v in marker_after.items()
            if k not in (er.SAMPLES_DIGEST_FIELD, C.DONE_MARKER_REBOUND_AT_FIELD)} == marker_before

    # the marker gained a key and lost none, so the task is still DONE and the
    # launcher's store audit still trusts it
    assert completion.is_done(cell, EVALPLUS_TASK) is True
    assert launcher_mod.invalid_done_marker_reason(cell, EVALPLUS_TASK) == audit_before
    assert "fields" not in (audit_before or "")
    assert completion.complete_task_summaries(cell)[EVALPLUS_TASK] == json.loads(
        after[rels["store_summary"]])
    # a re-bound summary is a NEW payload digest, which is exactly what PW14's
    # `unaccounted_reason` compares — so an already-merged store reads as
    # merge-pending again and the re-merge that adds the payload witness is
    # found from disk rather than remembered
    assert C.merge_payload_digest(EVALPLUS_TASK, json.loads(before[rels["store_summary"]])) \
        != C.merge_payload_digest(EVALPLUS_TASK, json.loads(after[rels["store_summary"]]))

    # (b) a marker that already carries the digest is a NO-OP, not a re-write
    again = sandbox_mod.run_rebind(tmp_path, dry_run=False)
    assert again["counts"] == {sandbox_mod.REBIND_REBIND: 0, sandbox_mod.REBIND_NOOP: 1,
                               sandbox_mod.REBIND_REFUSED: 0, sandbox_mod.REBIND_SKIPPED: 0}
    assert again["tasks"][0][er.SAMPLES_DIGEST_FIELD] == digest
    assert _snapshot(cell) == after


def test_rebind_refuses_a_task_whose_two_digest_witnesses_do_not_agree(
        tmp_path, monkeypatch, sandbox_mod):
    """(c) A re-bind never INVENTS a digest. ``execution.json`` and the
    ``samples.jsonl`` bytes are both consulted; a disagreement names both
    values and refuses, and so does either one being absent. Nothing is
    written for that task and the run exits non-zero."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, tdir, digest, _rels = _converted_unbound_store(tmp_path)
    execution = json.loads((tdir / er.EXECUTION_FILENAME).read_text())
    execution[er.SAMPLES_DIGEST_FIELD] = "0" * 64
    (tdir / er.EXECUTION_FILENAME).write_text(json.dumps(execution))
    before = _snapshot(cell)

    doc = sandbox_mod.run_rebind(tmp_path, dry_run=False)
    assert doc["counts"][sandbox_mod.REBIND_REFUSED] == 1
    why = doc["tasks"][0]["why"]
    assert "0" * 64 in why and digest in why    # BOTH values, named
    assert _snapshot(cell) == before

    # ... and the same when the execution record is gone entirely
    (tdir / er.EXECUTION_FILENAME).unlink()
    before = _snapshot(cell)
    doc = sandbox_mod.run_rebind(tmp_path, dry_run=False)
    assert doc["counts"][sandbox_mod.REBIND_REFUSED] == 1
    assert er.EXECUTION_FILENAME in doc["tasks"][0]["why"]
    assert _snapshot(cell) == before


def test_rebind_refuses_before_writing_when_anything_but_the_digest_would_move(
        tmp_path, monkeypatch, sandbox_mod):
    """(d) The byte-identity assertion is part of the MODE, not only of a test
    (E60). The would-be files are the REAL converter's output against a scratch
    COPY of the store, so a store whose recorded number no longer matches what
    the converter derives is caught: the mode refuses it, naming the field,
    before a single write. Same for a store whose ``rows.jsonl`` the conversion
    would append to — a re-bind that added a row would be a re-conversion."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)

    # a score in the store's own durable summary, altered
    cell, _tdir, _digest, rels = _converted_unbound_store(tmp_path / "score")
    ss = cell / rels["store_summary"]
    doc = json.loads(ss.read_text())
    doc["pass_at_1_plus"] = 0.99
    completion.atomic_write_json(ss, doc)
    before = _snapshot(cell)
    out = sandbox_mod.run_rebind(tmp_path / "score", dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1
    assert "pass_at_1_plus" in out["tasks"][0]["why"]
    assert _snapshot(cell) == before

    # a row the conversion would have to write back
    cell2, _tdir2, _d2, _r2 = _converted_unbound_store(tmp_path / "rows")
    rows_path = cell2 / "rows.jsonl"
    kept = rows_path.read_text().splitlines()[:-1]
    rows_path.write_text("".join(line + "\n" for line in kept))
    before2 = _snapshot(cell2)
    out = sandbox_mod.run_rebind(tmp_path / "rows", dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1
    assert "rows.jsonl" in out["tasks"][0]["why"]
    assert _snapshot(cell2) == before2


def test_rebind_skips_a_task_with_no_done_marker_and_refuses_one_with_no_generation(
        tmp_path, monkeypatch, sandbox_mod):
    """(e) A task with no done marker is not this mode's business — it is the
    sandbox and convert stages', which the marker gate hands it to — so it is
    SKIPPED and reported, never re-bound.

    The converse shape is the one E55 #2 is about and it must be loud: a
    converted task whose ``evalplus/<dataset>/`` directory is gone has a marker
    and no witness at all. Enumerating task DIRECTORIES alone would never see
    it, so the mode enumerates the union of task dirs and done markers and
    REFUSES this one."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)

    unconverted = tmp_path / "unconverted"
    _cell_dir(unconverted)
    doc = sandbox_mod.run_rebind(unconverted, dry_run=True)
    assert doc["counts"] == {sandbox_mod.REBIND_REBIND: 0, sandbox_mod.REBIND_NOOP: 0,
                             sandbox_mod.REBIND_REFUSED: 0, sandbox_mod.REBIND_SKIPPED: 1}
    assert "no done marker" in doc["tasks"][0]["why"]

    dropped = tmp_path / "dropped"
    cell, tdir, _digest, _rels = _converted_unbound_store(dropped)
    shutil.rmtree(tdir)
    before = _snapshot(cell)
    doc = sandbox_mod.run_rebind(dropped, dry_run=False)
    assert doc["counts"][sandbox_mod.REBIND_REFUSED] == 1
    assert doc["tasks"][0]["task"] == EVALPLUS_TASK
    assert _snapshot(cell) == before


def test_rebind_compares_its_three_targets_in_bytes_so_a_type_that_moved_refuses(
        tmp_path, monkeypatch, sandbox_mod):
    """E63 correction 1. Gate (4) compared the three re-bind TARGETS as parsed
    JSON dicts against ``{**current, digest}`` — and Python dicts hold
    ``1 == 1.0`` and ``True == 1``, so a number whose TYPE moved passed a check
    whose entire job is "changes no number". (Everything OUTSIDE the three
    targets was already byte-compared, by sha256, as the ``stray`` set.) The
    targets are now compared the same way: both sides rendered through the
    store's one JSON writer with exactly the permitted added keys removed, then
    compared as TEXT.

    The store here is otherwise perfectly re-bindable — the control at the end
    re-binds it — so the refusal is caused by the int that became a float and
    by nothing else."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    cell, _tdir, _digest, rels = _converted_unbound_store(tmp_path)
    ss = cell / rels["store_summary"]
    doc = json.loads(ss.read_text())
    intkey, intval = next((k, v) for k, v in sorted(doc.items())
                          if isinstance(v, int) and not isinstance(v, bool))
    assert float(intval) == intval and json.loads(json.dumps(float(intval))) == intval
    completion.atomic_write_json(ss, {**doc, intkey: float(intval)})
    # the OLD comparison could not see this: the two documents are EQUAL as
    # parsed dicts, and differ only in the type of one number
    assert json.loads(ss.read_text()) == doc
    before = _snapshot(cell)

    out = sandbox_mod.run_rebind(tmp_path, dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1
    why = out["tasks"][0]["why"]
    assert "BYTE FOR BYTE" in why and intkey in why
    assert _snapshot(cell) == before                      # refused before any write

    # a target that is not in the store's canonical form has no byte
    # comparison to make either — something other than the conversion wrote it
    ss.write_text(json.dumps(doc), encoding="utf-8")      # same doc, compact
    before = _snapshot(cell)
    out = sandbox_mod.run_rebind(tmp_path, dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1
    assert "canonical" in out["tasks"][0]["why"]
    assert _snapshot(cell) == before

    # the CONTROL: put the int back and the very same store re-binds
    completion.atomic_write_json(ss, doc)
    out = sandbox_mod.run_rebind(tmp_path, dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_REBIND] == 1


def test_rebind_refuses_a_torn_marker_instead_of_reading_it_as_a_no_op(
        tmp_path, monkeypatch, sandbox_mod):
    """E63 correction 2. Gate 1 declared ``no-op`` on the MARKER key alone, so
    a store whose marker carried the digest while a durable summary did not —
    or named other bytes — was passed over as "already bound, nothing to do",
    leaving a number bound that the durable files themselves disagree on.

    These writers cannot produce that state (``finish_task`` LIFTS the digest
    out of the summary it finishes; a re-bind writes the summaries FIRST and
    the marker LAST, so a crash leaves the marker BEHIND, which is the ordinary
    re-bind path), so it means a hand edit or a partial restore and the honest
    verdict is a refusal naming the file. What is NOT torn: a settled store
    that has since lost its task directory — surviving that loss is why E60
    moved the digest outside it (E55 #2)."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)

    def settled(where):
        cell, tdir, digest, rels = _converted_unbound_store(where)
        assert sandbox_mod.run_rebind(where, dry_run=False)["counts"][
            sandbox_mod.REBIND_REBIND] == 1
        return cell, tdir, digest, rels

    # (a) all three carry the same digest and every witness agrees → no-op
    settled_cell, settled_tdir, settled_digest, settled_rels = settled(tmp_path / "settled")
    before = _snapshot(settled_cell)
    out = sandbox_mod.run_rebind(tmp_path / "settled", dry_run=False)
    assert out["counts"] == {sandbox_mod.REBIND_REBIND: 0, sandbox_mod.REBIND_NOOP: 1,
                             sandbox_mod.REBIND_REFUSED: 0, sandbox_mod.REBIND_SKIPPED: 0}
    assert out["tasks"][0][er.SAMPLES_DIGEST_FIELD] == settled_digest
    assert _snapshot(settled_cell) == before

    # (b) the marker is AHEAD of its witnesses — both summaries, then each
    for label, kinds in (("marker_only", ("task_summary", "store_summary")),
                         ("no_store_summary", ("store_summary",)),
                         ("no_task_summary", ("task_summary",))):
        cell, _tdir, _d, rels = settled(tmp_path / label)
        for kind in kinds:
            path = cell / rels[kind]
            stripped = json.loads(path.read_text())
            stripped.pop(er.SAMPLES_DIGEST_FIELD)
            completion.atomic_write_json(path, stripped)
        before = _snapshot(cell)
        out = sandbox_mod.run_rebind(tmp_path / label, dry_run=False)
        assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1, label
        why = out["tasks"][0]["why"]
        assert "TORN" in why and all(rels[k] in why for k in kinds), label
        assert "cannot produce" in why                    # says whose fault it is not
        assert _snapshot(cell) == before, label

    # (c) a summary that names DIFFERENT bytes from the marker
    cell, _tdir, digest, rels = settled(tmp_path / "disagree")
    path = cell / rels["task_summary"]
    completion.atomic_write_json(path, {**json.loads(path.read_text()),
                                        er.SAMPLES_DIGEST_FIELD: "0" * 64})
    before = _snapshot(cell)
    out = sandbox_mod.run_rebind(tmp_path / "disagree", dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1
    why = out["tasks"][0]["why"]
    assert "0" * 64 in why and digest in why and rels["task_summary"] in why
    assert _snapshot(cell) == before

    # (d) the three agree, but a witness still beside them does not
    cell, tdir, digest, _rels = settled(tmp_path / "witness")
    completion.atomic_write_json(
        tdir / er.EXECUTION_FILENAME,
        {**json.loads((tdir / er.EXECUTION_FILENAME).read_text()),
         er.SAMPLES_DIGEST_FIELD: "1" * 64})
    before = _snapshot(cell)
    out = sandbox_mod.run_rebind(tmp_path / "witness", dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1
    assert "1" * 64 in out["tasks"][0]["why"] and er.EXECUTION_FILENAME in out["tasks"][0]["why"]
    assert _snapshot(cell) == before

    # (e) and what is NOT torn: (a)'s settled store with its whole task
    # directory removed. The two copies that matter live outside it — that is
    # the point of the re-bind — so this stays a no-op, and its reason names
    # the witnesses that are gone rather than pretending they agreed.
    shutil.rmtree(settled_tdir)
    out = sandbox_mod.run_rebind(tmp_path / "settled", dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_NOOP] == 1
    why = out["tasks"][0]["why"]
    assert er.SAMPLES_FILENAME in why and er.EXECUTION_FILENAME in why
    assert settled_rels["task_summary"] in why

    # (f) nor is the CONVERSE of torn. A crash mid-re-bind leaves the summaries
    # AHEAD of the marker, because `apply_rebind` writes them first and the
    # marker last — so gate 1 sees a marker with no digest, sends the store
    # down the ordinary re-bind path, and it re-plans to the same write. That
    # is the property `run_rebind`'s docstring claims, and the byte comparison
    # gate (4) now makes must keep holding: the two summaries already carry the
    # digest they are about to be given.
    cell, _tdir, digest, rels = settled(tmp_path / "crashed")
    mpath = cell / rels["done_marker"]
    completion.atomic_write_json(mpath, {
        k: v for k, v in json.loads(mpath.read_text()).items()
        if k not in (er.SAMPLES_DIGEST_FIELD, C.DONE_MARKER_REBOUND_AT_FIELD)})
    out = sandbox_mod.run_rebind(tmp_path / "crashed", dry_run=False)
    assert out["counts"][sandbox_mod.REBIND_REBIND] == 1
    assert json.loads(mpath.read_text())[er.SAMPLES_DIGEST_FIELD] == digest


def test_a_settled_store_reads_a_present_but_unreadable_witness_as_torn_not_as_absent(
        tmp_path, monkeypatch, sandbox_mod):
    """E68. The settled path asked ``recorded_samples_digest`` for
    ``execution.json``'s digest and treated its ``None`` as an ABSENCE — but
    that same ``None`` comes back for a file that is PRESENT and unparsable,
    and for one recording a ``samples_sha256`` that is not a sha256. Both were
    filed as ``lost``: verdict ``no-op``, exit 0, and a reason saying the task
    directory was gone with the file sitting in it. Absence is the ONE state
    this mode passes over (surviving the loss of the task directory is why the
    binding was moved outside it, E55 #2); a witness that is there and cannot
    be read is a statement this mode cannot check, so it is TORN. The fourth
    present-and-``None`` shape — a record that parses and names no bytes at all
    — is not decided here at all: E69 moved it onto the converter's era rule,
    and the test below this one drives its three eras.

    Driven on a real settled store, one malformed shape per store, with the
    store asserted byte-unchanged afterwards and the refusal carried through
    the entrypoint that turns it into a non-zero exit."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)

    def settled(where):
        cell, tdir, digest, rels = _converted_unbound_store(where)
        assert sandbox_mod.run_rebind(where, dry_run=False)["counts"][
            sandbox_mod.REBIND_REBIND] == 1
        return cell, tdir, digest, rels

    # (a) the witness is GONE — unchanged, and the case the split protects:
    # the durable pair outside the task directory still binds the number
    cell, tdir, digest, _rels = settled(tmp_path / "gone")
    (tdir / er.EXECUTION_FILENAME).unlink()
    out = sandbox_mod.run_rebind(tmp_path / "gone", dry_run=False)
    assert out["counts"] == {sandbox_mod.REBIND_REBIND: 0, sandbox_mod.REBIND_NOOP: 1,
                             sandbox_mod.REBIND_REFUSED: 0, sandbox_mod.REBIND_SKIPPED: 0}
    why = out["tasks"][0]["why"]
    assert er.EXECUTION_FILENAME in why
    assert "expected once the task directory is gone" in why      # true here, and only here

    # (b) present and unreadable — not JSON, and not a JSON object — and (c)
    # present with a `samples_sha256` that is not a sha256: the shapes that
    # used to read as (a)
    for label, written in (("torn", "{not json"),
                           ("not-an-object", json.dumps([1, 2])),
                           ("nonsense", json.dumps({er.SAMPLES_DIGEST_FIELD: "nope"}))):
        cell, tdir, digest, _rels = settled(tmp_path / label)
        (tdir / er.EXECUTION_FILENAME).write_text(written, encoding="utf-8")
        before = _snapshot(cell)
        out = sandbox_mod.run_rebind(tmp_path / label, dry_run=False)
        assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1, label
        assert out["counts"][sandbox_mod.REBIND_NOOP] == 0, label
        why = out["tasks"][0]["why"]
        assert "TORN" in why and er.EXECUTION_FILENAME in why, label
        # it must NOT read as the no-op whose reason says the directory is gone
        assert "is on disk and" in why, label
        assert "expected once the task directory is gone" not in why, label
        assert _snapshot(cell) == before, label           # the settled path writes nothing
        # and the verdict reaches the operator as a non-zero exit
        monkeypatch.setattr(sandbox_mod, "tinycap_sandbox_rebind",
                            types.SimpleNamespace(remote=lambda _doc=out, **kw: _doc))
        with pytest.raises(SystemExit) as ei:
            sandbox_mod.main(rebind_samples_digest=True)
        assert "1 task(s) REFUSED" in str(ei.value), label

    # (c') the fourth shape — a record that parses and names NO BYTES — has
    # its own test below: E68 filed it as silence on the strength of the digest
    # being absent, and E69 keys it on the positive era marker instead.

    # (d) the same two shapes on the OTHER witness inside the task directory,
    # `evalplus/<ds>/summary.json`, which `read_summary` already tore on — the
    # pin the sandbox's execution.json half was missing
    for label, written in (("summary-torn", "{not json"),
                           ("summary-nonsense", None)):
        cell, tdir, digest, rels = settled(tmp_path / label)
        path = tdir / er.SUMMARY_FILENAME
        if written is None:
            completion.atomic_write_json(path, {**json.loads(path.read_text()),
                                                er.SAMPLES_DIGEST_FIELD: "nope"})
        else:
            path.write_text(written, encoding="utf-8")
        before = _snapshot(cell)
        out = sandbox_mod.run_rebind(tmp_path / label, dry_run=False)
        assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1, label
        assert "TORN" in out["tasks"][0]["why"], label
        assert rels["task_summary"] in out["tasks"][0]["why"], label
        assert _snapshot(cell) == before, label

    # (e) the CONTROL: a witness that is present, readable and agreeing is the
    # settled no-op it always was
    cell, tdir, digest, _rels = settled(tmp_path / "agrees")
    out = sandbox_mod.run_rebind(tmp_path / "agrees", dry_run=False)
    assert out["counts"] == {sandbox_mod.REBIND_REBIND: 0, sandbox_mod.REBIND_NOOP: 1,
                             sandbox_mod.REBIND_REFUSED: 0, sandbox_mod.REBIND_SKIPPED: 0}
    assert out["tasks"][0][er.SAMPLES_DIGEST_FIELD] == digest
    assert "every witness beside them agrees" in out["tasks"][0]["why"]


def test_a_silent_execution_record_is_judged_by_the_converters_own_era_rule(
        tmp_path, monkeypatch, sandbox_mod):
    """E69. A settled store's ``execution.json`` that is PRESENT, parses, and
    names no bytes at all is not one shape but THREE, and E68 called all three
    age — an absence-as-evidence reading, when a POSITIVE marker was already
    there: a hand edit and a partial restore are byte-identical to a genuinely
    pre-digest record. The sandbox stamps ``samples_sha256``,
    ``contract_version`` and ``producer`` into ONE dict in a single atomic
    write, so a record carrying the marker WITHOUT the digest was never written
    by this repo.

    The verdict is the CONVERTER's own rule, so the two cannot rule opposite
    ways on the same bytes: ``store_era(record, contract=samples-digest)``
    first — ``current`` is TORN — then the CELL MANIFEST's era for the same
    contract, where only ``historical`` is age. E68's fixture pinned the HOLE:
    it stripped the digest off a record carrying no ``contract_version``, so
    the record read ``unknown``, the settled path no-op'ed, and
    ``samples_binding_of_task`` RAISED on those very bytes.

    Each fixture's two eras are asserted rather than assumed, and both readers
    are driven over the same store: the settled path refuses exactly where the
    converter raises, and no-ops exactly where it binds the task
    historical-unbound. Nothing is written in any of the three."""
    monkeypatch.setattr(er, "expected_task_ids", lambda d: TWO_IDS)
    # what `execute_one` stamps beside the digest — the record's own marker
    sandbox_marker = {C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION,
                      C.MANIFEST_KEY_PRODUCER: {"driver": C.PRODUCER_DRIVER_SANDBOX}}

    for label, producer, stamp, record_era, manifest_era, refuses in (
            # the EXECUTION producer positively says it wrote under the
            # contract, so its silence is damage whatever the generation's age
            ("silent-current", UNDECLARED_TREE_SHA, sandbox_marker,
             C.STORE_ERA_CURRENT, C.STORE_ERA_UNKNOWN, True),
            # neither producer says anything — E68's own fixture, and the hole
            ("silent-unknown", UNDECLARED_TREE_SHA, {},
             C.STORE_ERA_UNKNOWN, C.STORE_ERA_UNKNOWN, True),
            # ...and the ONE shape that is age: a generation whose declared
            # tree does not CONTAIN the digest, plus a marker-less record
            ("silent-historical", POST_SAMPLING_TREE_SHA, {},
             C.STORE_ERA_UNKNOWN, C.STORE_ERA_HISTORICAL, False)):
        cell, tdir, digest, _rels = _converted_unbound_store(tmp_path / label,
                                                             producer=producer)
        assert sandbox_mod.run_rebind(tmp_path / label, dry_run=False)["counts"][
            sandbox_mod.REBIND_REBIND] == 1, label
        record = {**{k: v for k, v
                     in json.loads((tdir / er.EXECUTION_FILENAME).read_text()).items()
                     if k != er.SAMPLES_DIGEST_FIELD}, **stamp}
        completion.atomic_write_json(tdir / er.EXECUTION_FILENAME, record)
        manifest = json.loads((cell / "manifest.json").read_text())

        # the shape each fixture is built to have, stated
        assert C.store_era(record, contract=C.CONTRACT_SAMPLES_DIGEST)[0] == record_era, label
        assert C.store_era(manifest,
                           contract=C.CONTRACT_SAMPLES_DIGEST)[0] == manifest_era, label

        before = _snapshot(cell)
        out = sandbox_mod.run_rebind(tmp_path / label, dry_run=False)
        why = out["tasks"][0]["why"]
        assert _snapshot(cell) == before, label       # the settled path writes nothing, either way
        # the no-op's "gone" clause is never said of a record sitting in the
        # directory, whichever way the era falls
        assert "expected once the task directory is gone" not in why, label

        if refuses:
            assert out["counts"][sandbox_mod.REBIND_REFUSED] == 1, label
            assert out["counts"][sandbox_mod.REBIND_NOOP] == 0, label
            assert "TORN" in why and er.EXECUTION_FILENAME in why, label
            assert f"names no {er.SAMPLES_DIGEST_FIELD} at all" in why, label
            assert "written before the digest existed" not in why, label
            # ...and the converter RAISES on the same bytes — the contradiction
            with pytest.raises(er.EvalPlusSeamError) as ei:
                er.samples_binding_of_task(tdir, manifest, task=EVALPLUS_TASK)
            assert ("damage rather than age" if record_era == C.STORE_ERA_CURRENT
                    else "lacks the samples digest") in str(ei.value), label
            # and the verdict reaches the operator as a non-zero exit
            monkeypatch.setattr(sandbox_mod, "tinycap_sandbox_rebind",
                                types.SimpleNamespace(remote=lambda _doc=out, **kw: _doc))
            with pytest.raises(SystemExit) as ei:
                sandbox_mod.main(rebind_samples_digest=True)
            assert "1 task(s) REFUSED" in str(ei.value), label
        else:
            assert out["counts"] == {sandbox_mod.REBIND_REBIND: 0, sandbox_mod.REBIND_NOOP: 1,
                                     sandbox_mod.REBIND_REFUSED: 0,
                                     sandbox_mod.REBIND_SKIPPED: 0}, label
            assert out["tasks"][0][er.SAMPLES_DIGEST_FIELD] == digest, label
            assert f"{er.EXECUTION_FILENAME} is there and names no bytes" in why, label
            assert "written before the digest existed" in why, label
            # ...and the converter agrees on the same bytes
            assert er.samples_binding_of_task(tdir, manifest, task=EVALPLUS_TASK) == (
                None, C.SAMPLES_BINDING_HISTORICAL_UNBOUND), label


def test_the_rebind_mode_is_a_cpu_only_container_that_runs_alone(sandbox_mod, monkeypatch):
    """The mode is registered as its own CPU-only Modal function (no sandbox,
    no GPU), the entrypoint exposes it as a flag, it refuses to share a run
    with the execute/merge path — a re-bind converts nothing, so a merge
    decision has nothing to decide — and a refusal makes the run exit
    non-zero."""
    fn_kw = sandbox_mod.app.functions["tinycap_sandbox_rebind"]
    assert fn_kw["name"] == sandbox_mod.REBIND_CONTAINER_NAME and "gpu" not in fn_kw
    params = inspect.signature(sandbox_mod.main).parameters
    assert params["rebind_samples_digest"].default is False

    with pytest.raises(SystemExit) as ei:
        sandbox_mod.main(rebind_samples_digest=True, merge_mode=sandbox_mod.MERGE_MODE_CHAIN)
    assert sandbox_mod.REBIND_FLAG in str(ei.value)

    counts = {v: 0 for v in sandbox_mod.REBIND_VERDICTS}
    monkeypatch.setattr(sandbox_mod, "tinycap_sandbox_rebind", types.SimpleNamespace(
        remote=lambda **kw: {"counts": {**counts, sandbox_mod.REBIND_REFUSED: 2,
                                        sandbox_mod.REBIND_REBIND: 5}}))
    with pytest.raises(SystemExit) as ei:
        sandbox_mod.main(rebind_samples_digest=True)
    assert "2 task(s) REFUSED" in str(ei.value)
    monkeypatch.setattr(sandbox_mod, "tinycap_sandbox_rebind", types.SimpleNamespace(
        remote=lambda **kw: {"counts": {**counts, sandbox_mod.REBIND_NOOP: 3}}))
    assert sandbox_mod.main(rebind_samples_digest=True) is None


def test_finish_task_never_lifts_the_rebound_stamp_out_of_a_summary(tmp_path):
    """``rebound_at`` rides in ``DONE_MARKER_OPTIONAL_FIELDS`` so both marker
    admission checks accept it, but it is NOT in
    ``DONE_MARKER_SUMMARY_LIFTED_FIELDS``: a run that FINISHES a task has not
    re-bound it, and a marker claiming otherwise would assert something false.
    So a summary carrying the key leaves the fresh marker without it, while
    ``write_done_marker`` still admits one that carries it."""
    assert C.DONE_MARKER_REBOUND_AT_FIELD in C.DONE_MARKER_OPTIONAL_FIELDS
    assert C.DONE_MARKER_REBOUND_AT_FIELD not in C.DONE_MARKER_SUMMARY_LIFTED_FIELDS
    cell = tmp_path / "cell"
    task = "inspect/math"
    summary = {"acc": 1.0, "n": 1, C.DONE_MARKER_REBOUND_AT_FIELD: 1.0,
               er.SAMPLES_DIGEST_FIELD: "ab" * 32}
    completion.finish_task(cell, task=task, n_items=1, n_expected_full=1, limits=None,
                           n_rows=1, n_requests=1, n_rejected=0, wall_clock_s=1.0,
                           summary=summary)
    marker = json.loads(completion.done_marker_path(cell, task).read_text())
    assert C.DONE_MARKER_REBOUND_AT_FIELD not in marker
    assert marker[er.SAMPLES_DIGEST_FIELD] == "ab" * 32
    # the admission check accepts one that DOES carry it (the re-bind's write)
    completion.write_done_marker(cell, task, {**marker, C.DONE_MARKER_REBOUND_AT_FIELD: 2.0})
    assert json.loads(completion.done_marker_path(cell, task).read_text())[
        C.DONE_MARKER_REBOUND_AT_FIELD] == 2.0
    with pytest.raises(completion.CompletionError):
        completion.write_done_marker(cell, task, {**marker, "invented_key": 1})


def _pre_rebind_store(sweep: Path, base_manifest: dict, *, key: str, chash: str,
                      acc: float = 0.25, n: int = 2) -> tuple:
    """A store in the state the 36 real ones were merged in, ON DISK: the task
    generated and executed (``samples.jsonl`` + an ``execution.json`` naming
    those bytes), but its DURABLE ``summary/<task>.json`` and done marker
    naming no digest — the conversion that wrote them predates E60, and
    conversion is gated on the absence of the marker, so the field is
    unreachable for every already-converted task. Returns ``(store, digest)``.
    """
    store = _pass_store(sweep, base_manifest, key=key, chash=chash, pass_no=1, acc=acc, n=n)
    tdir = er.served_task_dir(store, "humaneval")
    tdir.mkdir(parents=True, exist_ok=True)
    samples = f'{{"task_id": "HumanEval/0", "acc": {acc}}}\n'.encode()
    (tdir / er.SAMPLES_FILENAME).write_bytes(samples)
    digest = hashlib.sha256(samples).hexdigest()
    (tdir / er.EXECUTION_FILENAME).write_text(json.dumps(
        {"wall_clock_s": 1.0, er.SAMPLES_DIGEST_FIELD: digest}))
    body = {k: v for k, v in completion.read_task_summary(store, EVALPLUS_TASK).items()
            if k != er.SAMPLES_DIGEST_FIELD}
    completion.write_task_summary(store, EVALPLUS_TASK, body)
    completion.atomic_write_json(tdir / er.SUMMARY_FILENAME, body)
    marker = {k: v for k, v in json.loads(
        completion.done_marker_path(store, EVALPLUS_TASK).read_text()).items()
        if k != er.SAMPLES_DIGEST_FIELD}
    completion.write_done_marker(store, EVALPLUS_TASK, marker)
    return store, digest


def _apply_rebind_on_disk(store: Path, digest: str, *, stamp: float = 2000.0) -> dict:
    """Exactly what ``tinycap_sandbox --rebind-samples-digest`` writes: the
    digest into the two summaries and the marker, ``finished_at`` preserved,
    ``rebound_at`` stamped. Returns the re-bound durable summary."""
    tdir = er.served_task_dir(store, "humaneval")
    body = {**completion.read_task_summary(store, EVALPLUS_TASK),
            er.SAMPLES_DIGEST_FIELD: digest}
    completion.write_task_summary(store, EVALPLUS_TASK, body)
    completion.atomic_write_json(tdir / er.SUMMARY_FILENAME, body)
    marker = json.loads(completion.done_marker_path(store, EVALPLUS_TASK).read_text())
    completion.write_done_marker(store, EVALPLUS_TASK, {
        **marker, er.SAMPLES_DIGEST_FIELD: digest, C.DONE_MARKER_REBOUND_AT_FIELD: stamp})
    return body


def test_a_re_bound_store_is_filed_by_a_plain_re_merge_and_leaves_the_pending_list(
        launcher_mod, sandbox_mod, tmp_path, monkeypatch):
    """On disk, through BOTH writers. A re-bind was expected to make the
    store merge-pending again, with the re-merge adding the payload witness;
    the real pass then found that re-merge REFUSED all 36 stores at the
    standing-source gate, whose only remedy was 36 separate
    ``--replace-source`` invocations recording a ``refreshed`` REPLACEMENT that
    did not happen.

    So the writer recognises the re-bind itself. Here: a store merged BEFORE
    the re-bind, re-bound exactly as the mode writes it, then re-merged with NO
    flag through the launcher's two-tree path — filed, ``controls[entry]``
    gaining only ``samples_sha256``, the record's ``payload_digest`` moved, a
    ``rebound`` history record and no ``refreshed`` one, the REC's ``notes``
    saying ``RE-BOUND``, the scan reading the store accounted again, the loader
    loading it, and a second re-merge byte-identical. The driver's in-process
    merge takes the same path on a second store. And the two near misses that
    must still refuse on disk: a marker that names other bytes, and a summary
    whose numbers moved beside the digest."""
    from sisteer.eval.cell_controls import load_controls_summaries

    cfg, cell, cell_dir, _tdir, _job = _tinycap_store(tmp_path, monkeypatch)
    base = json.loads((cell_dir / "manifest.json").read_text())
    sweep = tmp_path / "sweep"
    summary = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    W = launcher_mod.writer_tree_sha_of(REPO / "src")
    D, REF, WT = (C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD, C.MERGE_SOURCE_REFRESHED_FIELD,
                  C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD)
    p1, digest = _pre_rebind_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__11111111",
                                   chash="11111111" + "0" * 8)
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and res[str(p1)]["merged"]
    doc = json.loads(summary.read_text())
    d1 = _digest(p1)
    assert er.SAMPLES_DIGEST_FIELD not in doc["controls"][EVALPLUS_TASK]     # the entry names no bytes
    assert doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK][
        C.MERGE_SOURCE_SAMPLES_DIGEST_FIELD] == digest                      # the record does (E54 #1)
    assert sandbox_mod.unmerged_completed_stores(sweep) == []
    standing = dict(doc["controls"][EVALPLUS_TASK])

    # the re-bind: the digest into both summaries and the marker, nothing else
    body = _apply_rebind_on_disk(p1, digest)
    d2 = _digest(p1)
    # the writer normalizes the definition on both sides (PW13), so compare there
    assert d2 != d1
    assert C.with_evalplus_metric_def(EVALPLUS_TASK, body) == {
        **standing, er.SAMPLES_DIGEST_FIELD: digest}
    # BY DESIGN the store reads merge-pending again — and the reason now
    # names the plain re-merge that files it
    [p] = sandbox_mod.unmerged_completed_stores(sweep)
    why = p["why"][EVALPLUS_TASK]
    assert p["cell_dir"] == str(p1) and "OWN summary changed" in why and d1 in why and d2 in why
    assert "RE-BIND" in why and C.MERGE_SOURCE_SAMPLES_DIGEST_FIELD in why

    # THE FIX: a PLAIN re-merge, no --replace-source anywhere
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    notes = res[str(p1)].pop("notes")
    assert res[str(p1)] == {"ok": True, "merged": True, "path": str(summary)}
    assert any("RE-BOUND" in n and d1 in n and d2 in n and digest in n for n in notes)
    assert not any("REFRESHED" in n for n in notes)
    doc = json.loads(summary.read_text())
    rec = doc[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK]
    assert doc["controls"][EVALPLUS_TASK] == {**standing, er.SAMPLES_DIGEST_FIELD: digest}
    assert REF not in rec and rec[D] == d2
    assert rec[C.MERGE_SOURCE_REBOUND_FIELD] == [{
        C.MERGE_REBIND_AT_FIELD: 2000.0, C.MERGE_REFRESH_DIGEST_WAS_FIELD: d1, D: d2,
        C.MERGE_SOURCE_SAMPLES_DIGEST_FIELD: digest,
        C.MERGE_REBIND_WITNESSES_FIELD: list(C.MERGE_REBIND_WITNESSES), WT: W}]
    assert sandbox_mod.unmerged_completed_stores(sweep) == []               # accounted again
    assert load_controls_summaries(sweep, [cell], cfg)[C.CELL_PSRKL_SAFETY][
        EVALPLUS_TASK][er.SAMPLES_DIGEST_FIELD] == digest
    settled = summary.read_bytes()
    res = launcher_mod.merge_cell_summaries([str(p1)], tree=REPO, python=sys.executable)
    assert res[str(p1)]["ok"] and summary.read_bytes() == settled           # idempotent
    assert not any("RE-BOUND" in n for n in res[str(p1)]["notes"])

    # the DRIVER's in-process merge takes the same path, on its own store
    p2, digest2 = _pre_rebind_store(sweep, base, key="tok__controls__t0__a0__aaaaaaaa__22222222",
                                    chash="22222222" + "0" * 8, acc=0.25)
    other = sweep / C.CELL_PSRKL_SAFETY / "controls_summary.json"
    assert other == summary
    # p2 exactly witnesses the standing payload, so it is recorded as an equal
    # witness rather than taking the entry — give it its own cell to merge into
    shutil.rmtree(p2)
    p2, digest2 = _pre_rebind_store(sweep, {**base, "cell_id": "other_cell"},
                                    key="tok__controls__t0__a0__aaaaaaaa__33333333",
                                    chash="33333333" + "0" * 8, acc=0.75)
    other = sweep / "other_cell" / "controls_summary.json"
    completion.merge_controls_summary(p2)
    was = C.merge_payload_digest(EVALPLUS_TASK, json.loads(other.read_text())[
        "controls"][EVALPLUS_TASK])
    _apply_rebind_on_disk(p2, digest2, stamp=3000.0)
    completion.merge_controls_summary(p2)
    rec2 = json.loads(other.read_text())[C.SUMMARY_KEY_SOURCES][EVALPLUS_TASK]
    assert REF not in rec2
    assert rec2[C.MERGE_SOURCE_REBOUND_FIELD] == [{
        C.MERGE_REBIND_AT_FIELD: 3000.0, C.MERGE_REFRESH_DIGEST_WAS_FIELD: was,
        D: _digest(p2), C.MERGE_SOURCE_SAMPLES_DIGEST_FIELD: digest2,
        C.MERGE_REBIND_WITNESSES_FIELD: list(C.MERGE_REBIND_WITNESSES),
        WT: None}]                                     # the driver's writer IS the producer

    # NEAR MISS 1: the marker names other bytes — refused, nothing written
    p3, digest3 = _pre_rebind_store(sweep, {**base, "cell_id": "third_cell"},
                                    key="tok__controls__t0__a0__aaaaaaaa__44444444",
                                    chash="44444444" + "0" * 8, acc=0.5)
    third = sweep / "third_cell" / "controls_summary.json"
    completion.merge_controls_summary(p3)
    before = third.read_bytes()
    _apply_rebind_on_disk(p3, digest3)
    marker = json.loads(completion.done_marker_path(p3, EVALPLUS_TASK).read_text())
    completion.write_done_marker(p3, EVALPLUS_TASK,
                                 {**marker, er.SAMPLES_DIGEST_FIELD: "ab" * 32})
    res = launcher_mod.merge_cell_summaries([str(p3)], tree=REPO, python=sys.executable)
    assert res[str(p3)]["ok"] is False
    assert C.REBIND_WITNESS_DONE_MARKER in res[str(p3)]["err"]
    assert "never refreshes a standing entry silently" in res[str(p3)]["err"]
    assert third.read_bytes() == before

    # NEAR MISS 2: a number moved BESIDE the digest — refused, nothing written
    completion.write_done_marker(p3, EVALPLUS_TASK,
                                 {**marker, er.SAMPLES_DIGEST_FIELD: digest3})
    body = completion.read_task_summary(p3, EVALPLUS_TASK)
    completion.write_task_summary(p3, EVALPLUS_TASK, {**body, "acc": 0.51})
    res = launcher_mod.merge_cell_summaries([str(p3)], tree=REPO, python=sys.executable)
    assert res[str(p3)]["ok"] is False and "not byte-equal" in res[str(p3)]["err"]
    assert third.read_bytes() == before
