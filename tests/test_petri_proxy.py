"""Petri Bloom Phase 2: the routing proxy over a STUB OpenAI backend —
CPU-only, no vLLM, no model loads, no network.

Covers: OpenAI chat + tool-call in/out translation (multi-turn + an
agent tool round-trip); ``model=petri__<id>`` selects the right spec
(lens selector IDs / LoRA adapter name / passthrough — H3); unknown
virtual names refuse loudly; the prompted vmodels verify the SI prefix
per request (H8, both directions, mixed-content system messages
included); the per-request audit log (``virtual_model`` +
``petri_intervention_hash``, response field + structured log record);
selector smuggling / streaming / non-proxy-identity refusals; the exact-replay
mechanism discipline (serialize, or ATTESTED batch invariance — never an
unattested off-switch); the routing-table export/load round trip (the
recorded ``exact_replay_strategy`` included); the authenticated,
proxy-served-only ``/v1/models`` listing; and the target-client +
ProxyClientEngine wiring on the ONE canonical ``/v1`` base URL.
"""

import asyncio
import dataclasses
import json
import logging
import re

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import petri_target as pt
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving.vllm_server import BATCH_INVARIANT_ENV

REV = "0123456789abcdef0123456789abcdef01234567"
IMPL_COMMIT = "fedcba9876" * 4
SAFETY_PREFIX = b"synthetic safety instruction: be safe (test bytes)"
CONSTITUTION_PREFIX = (b"synthetic constitution instruction: many "
                       b"principles (test bytes)")
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

CANONICAL_22 = (
    ["none", "prompted__safety", "prompted__constitution"]
    # the prompt-SI matrix is over the FIXED confirmatory family
    # (PROMPTABLE_CLASS_A_METHODS), never the serving vocabulary
    # CLASS_A_METHODS, which also admits object-only methods that have no
    # cell under a prompt SI (review of 039a355c, 2026-09-10)
    + [f"si_{si}__{m}" for m in dc.PROMPTABLE_CLASS_A_METHODS
       for si in ("safety", "constitution")]
    + list(dc.CLASS_B_IDS)
)

PROMPT_FILES = {"auditor": b"# synthetic auditor prompt bytes\n",
                "judge": b"# synthetic judge prompt bytes\n"}


def _behavior_files(spec):
    """The FULL behavior-dir inventory the harness hash requires
    (canonical dirs + every in-play SI-variant dir)."""
    return {k: f"# behavior {k}\nsynthetic eval bytes\n".encode()
            for k in cm.petri_expected_behavior_keys(spec)}
BLOOM_OPTIONS = dict(max_turns=40, enable_prefill=False,
                     enable_rollback=False, compaction=False,
                     realism_filter=False, epochs=1, seed=0)


def _spec(**over) -> cm.PetriSpec:
    kwargs = dict(
        audit="petri-twosi-test",
        behaviors=list(cm.PETRI_BEHAVIORS),
        cells=list(CANONICAL_22),
        scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5",
        judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model=BASE_MODEL,
        target_revision=REV,
        serving_backend="proxy_vllm",
        serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enable-lora", "--enforce-eager"],
        chat_template_hash="12" * 8,
        tool_parser="llama3_json",
        serving_impl_commit=IMPL_COMMIT,
        prompted_prefix_hashes={
            "safety": cm.sha256_hex16(SAFETY_PREFIX),
            "constitution": cm.sha256_hex16(CONSTITUTION_PREFIX),
        },
        bloom_options=cm.BloomOptions(**BLOOM_OPTIONS),
    )
    kwargs.update(over)
    return cm.PetriSpec(**kwargs)


def _source(tmp_path, *, lora_methods=("lora_r1",)):
    entries = {
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "prompted__safety": dc.RegistryEntry(
            cell_id="prompted__safety", kind="prompted", si_id="safety",
            si_prefix=SAFETY_PREFIX),
        "prompted__constitution": dc.RegistryEntry(
            cell_id="prompted__constitution", kind="prompted",
            si_id="constitution", si_prefix=CONSTITUTION_PREFIX),
    }
    for cell_id in CANONICAL_22[3:]:
        if cell_id in dc.CLASS_B_IDS:
            kind, si, method = f"obj_{cell_id}", None, cell_id
        else:
            head, _, method = cell_id.partition("__")
            si = head[len("si_"):]
            kind = f"obj_{method}"
        artifact = tmp_path / f"{cell_id}.pt"
        artifact.write_bytes(b"synthetic-object-bytes:" + cell_id.encode())
        entries[cell_id] = dc.RegistryEntry(
            cell_id=cell_id, kind=kind, si_id=si, object_path=str(artifact),
            serving_path=("lora" if method in lora_methods else "lens"),
            resolved_spec={"kind": kind, "si_id": si, "layer": 15,
                           "scale": 1.0})
    return dc.MappingCellSource(entries)


class StubBackend:
    """A scripted upstream: records every payload; answers with a canned
    (or queued) OpenAI response; tracks call overlap for the
    serialization test."""

    def __init__(self, responses=None, delay_s: float = 0.0,
                 request_failures=None):
        self.payloads = []
        self.responses = list(responses or [])
        self.delay_s = delay_s
        self.in_flight = 0
        self.max_in_flight = 0
        #: response-id -> the worker's contained-failure record; popped
        #: by take_request_failure exactly like the live worker ledger.
        self.request_failures = dict(request_failures or {})
        self.failure_probes = []
        self.expected_workers = None          # set by RoutingProxy from the attestation's n_workers

    async def take_request_failure(self, request_id):
        self.failure_probes.append(request_id)
        return self.request_failures.pop(request_id, None)

    async def chat_completions(self, payload):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self.payloads.append(json.loads(json.dumps(payload)))
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.responses:
                return self.responses.pop(0)
            return {
                "id": "cmpl-stub", "object": "chat.completion",
                "created": 0, "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "stub reply"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }
        finally:
            self.in_flight -= 1


def _cells(spec, source):
    return dc.enumerate_cells(spec, source,
                              behavior_files=_behavior_files(spec),
                              prompt_files=PROMPT_FILES)


class _TailRender:
    """A HERMETIC agent-loop render probe for this module: every request
    renders to an ordinary tail-header render (what Llama-3.1's template
    does for tool turns), so the proxy's default-on Petri-facing entry
    never declares here and never loads a tokenizer (no test may touch the
    hub; without this the first default-on run of the tool round-trip pin
    downloaded one, 2026-09-02). The agent-loop path's own tests bring
    their own probes (test_petri_agent_loop_serving,
    test_gemma4_petri_multiturn_boundary, test_petri_proxy_launcher)."""

    header_ids = (901, 902, 903, 904)

    def render(self, messages, tools):
        return (1, 2, 3, *self.header_ids)


def _proxy(tmp_path, *, backend=None, exact_replay_strategy="serialize",
           launch_attestation=None, api_token=None, spec=None,
           source=None, agent_loop_render=None, **source_kw):
    spec = spec or _spec()
    source = source or _source(tmp_path, **source_kw)
    cells = _cells(spec, source)
    backend = backend or StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode=spec.serving_mode,
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=exact_replay_strategy,
        launch_attestation=launch_attestation, api_token=api_token,
        # HERMETIC agent-loop render probe (see _TailRender): the proxy's
        # default-on Petri-facing entry must never load a tokenizer here
        agent_loop_render=agent_loop_render or _TailRender())
    return proxy, backend, cells


def _ask(proxy, request):
    return asyncio.run(proxy.handle_chat_completions(request))


def _chat(model, content="hello", system=None):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return {"model": model, "messages": messages, "max_tokens": 32}


# --------------------------------------------------------------------------
# virtual-name -> spec selection (H3 selector / LoRA name / passthrough)
# --------------------------------------------------------------------------


def test_lens_vmodel_selects_selector_ids_only(tmp_path):
    proxy, backend, cells = _proxy(tmp_path)
    cell = {c.cell_id: c for c in cells.canonical}["si_constitution__cast"]
    out = _ask(proxy, _chat("petri__si_constitution__cast"))
    up = backend.payloads[-1]
    # base weights + the immutable selector — never tensors/callables (H3)
    assert up["model"] == BASE_MODEL
    sel = up[px.UPSTREAM_EXTRA_ARGS_KEY]
    assert sel == {"petri_object": "si_constitution__cast",
                   "expect_hash": cell.intervention_hash}
    json.dumps(sel)  # JSON scalars only, by construction
    # the client sees the VIRTUAL name + the bookkeeping stamp
    assert out["model"] == "petri__si_constitution__cast"
    assert out["petri"]["serving_identity"] == cell.serving_identity
    assert out["petri"]["intervention_hash"] == cell.intervention_hash


def test_required_audit_field_is_emitted_end_to_end(tmp_path, caplog):
    """The REQUIRED audit field name `petri_intervention_hash` appears
    verbatim on the response, in the request log, and as a structured
    logging-record attribute together with `virtual_model` — an audit
    collector consuming any of those surfaces finds the contract field,
    never only a nested/renamed variant."""
    proxy, _, cells = _proxy(tmp_path)
    cell = {c.cell_id: c for c in cells.canonical}["si_safety__loreft"]
    with caplog.at_level(logging.INFO, logger="petri.proxy"):
        out = _ask(proxy, _chat("petri__si_safety__loreft"))
    # 1. the response carries the exact top-level field
    assert out["petri_intervention_hash"] == cell.intervention_hash
    # 2. the persisted request log carries the exact field names
    entry = proxy.request_log[-1]
    assert entry["virtual_model"] == "petri__si_safety__loreft"
    assert entry["petri_intervention_hash"] == cell.intervention_hash
    # 3. the structured log record carries both as attributes
    records = [r for r in caplog.records
               if getattr(r, "petri_intervention_hash", None) is not None]
    assert records, "no structured petri-proxy log record was emitted"
    assert records[-1].virtual_model == "petri__si_safety__loreft"
    assert records[-1].petri_intervention_hash == cell.intervention_hash


def test_lora_vmodel_routes_to_adapter_name(tmp_path):
    proxy, backend, cells = _proxy(tmp_path)
    _ask(proxy, _chat("petri__si_safety__lora_r1"))
    up = backend.payloads[-1]
    # native multi-LoRA selects by model name == the cell id
    assert up["model"] == "si_safety__lora_r1"
    assert px.UPSTREAM_EXTRA_ARGS_KEY not in up


def test_base_passthrough_and_noop_legs(tmp_path):
    proxy, backend, cells = _proxy(tmp_path)
    _ask(proxy, _chat("petri__base"))
    assert backend.payloads[-1]["model"] == BASE_MODEL
    assert px.UPSTREAM_EXTRA_ARGS_KEY not in backend.payloads[-1]
    # the scale-0 lens leg: armed hook, selector scale 0.0
    _ask(proxy, _chat("petri__none__lens0"))
    sel = backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    assert sel["petri_object"] == "none__lens0"
    assert sel["scale"] == 0.0
    # the zero-adapter LoRA leg routes to its adapter name
    _ask(proxy, _chat("petri__none__lora0"))
    assert backend.payloads[-1]["model"] == "none__lora0"


def test_unknown_virtual_model_refuses_loudly(tmp_path):
    proxy, _, _ = _proxy(tmp_path)
    with pytest.raises(px.ProxyRefusal, match="unknown virtual model"):
        _ask(proxy, _chat("petri__not_a_cell"))
    try:
        _ask(proxy, _chat("gpt-4o"))
    except px.ProxyRefusal as e:
        assert e.http_status == 404
        assert e.code == "unknown_virtual_model"
    else:
        pytest.fail("unknown model did not refuse")


def _foreign_hf_route():
    """A route claiming an hf_shim serving identity. enumerate_cells can
    no longer produce one (audit cells are proxy_vllm-served by
    construction) — this can only enter the table as foreign/corrupt
    data, and the proxy's refusal is DEFENSIVE."""
    return px.RouteEntry(
        virtual_model="petri__foreign_hf_row", cell_id="si_safety__cast",
        cell_class="class_a",
        serving_identity="hf_shim+apply+exact_replay",
        intervention_hash="ab" * 8)


def test_enumerated_routes_are_all_proxy_served(tmp_path):
    """No HF-shim serving fallback exists: every route projected from a
    real enumeration carries a proxy_vllm identity."""
    _, _, cells = _proxy(tmp_path)
    for route in px.routes_from_cells(cells):
        ident = cm.parse_serving_identity(route.serving_identity)
        assert ident.backend == "proxy_vllm", route.virtual_model


def test_non_proxy_identity_route_is_refused_defensively(tmp_path):
    proxy, backend, cells = _proxy(tmp_path)
    routes = px.routes_from_cells(cells) + (_foreign_hf_route(),)
    proxy = px.RoutingProxy(
        routes, backend, base_model=BASE_MODEL,
        serving_mode="exact_replay",
        prompted_prefixes=px.prompted_prefix_specs(
            cells, _source(tmp_path)),
        exact_replay_strategy="serialize")
    with pytest.raises(px.ProxyRefusal, match="reference oracle"):
        _ask(proxy, _chat("petri__foreign_hf_row"))


def test_models_listing_never_advertises_non_proxy_routes(tmp_path):
    """/v1/models exposes only what THIS proxy serves: a foreign
    non-proxy-identity route stays in the internal table (so its request
    gets the precise 421 above) but is never listed — a
    listed-then-refused model would poison discovery and the engine
    health check."""
    proxy, backend, cells = _proxy(tmp_path)
    routes = px.routes_from_cells(cells) + (_foreign_hf_route(),)
    proxy = px.RoutingProxy(
        routes, backend, base_model=BASE_MODEL,
        serving_mode="exact_replay",
        prompted_prefixes=px.prompted_prefix_specs(
            cells, _source(tmp_path)),
        exact_replay_strategy="serialize")
    listed = {m["id"] for m in proxy.models_payload()["data"]}
    assert "petri__foreign_hf_row" not in listed
    # but the proxy-served legs all advertise
    assert {"petri__base", "petri__none__lens0",
            "petri__prompted__safety"} <= listed
    # the internal table still knows the foreign route for the 421 refusal
    assert "petri__foreign_hf_row" in proxy.virtual_models()
    # and the listing carries this proxy's identity metadata
    meta = proxy.models_payload()["petri_proxy"]
    assert meta["serving_mode"] == "exact_replay"
    assert meta["exact_replay_strategy"] == "serialize"
    assert meta["base_model"] == BASE_MODEL


def test_models_endpoint_authenticates_like_chat(tmp_path):
    """When chat requires a token, /v1/models requires the SAME token —
    otherwise a wrong key would pass the health check and fail only at
    the first generation."""
    proxy, _, _ = _proxy(tmp_path, api_token="sekrit")
    with pytest.raises(px.ProxyRefusal) as err:
        proxy.handle_models(authorization=None)
    assert err.value.http_status == 401
    with pytest.raises(px.ProxyRefusal):
        proxy.handle_models(authorization="Bearer wrong")
    payload = proxy.handle_models(authorization="Bearer sekrit")
    assert payload["data"]  # the authenticated listing serves normally


# --------------------------------------------------------------------------
# translation: multi-turn + the agent tool round trip
# --------------------------------------------------------------------------


def test_multi_turn_messages_pass_through_verbatim(tmp_path):
    proxy, backend, _ = _proxy(tmp_path)
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and now?"},
    ]
    _ask(proxy, {"model": "petri__base", "messages": messages,
                 "max_tokens": 8, "temperature": 0.7})
    up = backend.payloads[-1]
    assert up["messages"] == messages
    assert up["temperature"] == 0.7  # sampling params untouched


def test_agent_tool_call_round_trip(tmp_path):
    """The agent-modality shape: tools pass through on the way in,
    tool_calls come back untouched, and the follow-up turn carrying the
    simulated tool result (role 'tool') passes through."""
    tools = [{"type": "function", "function": {
        "name": "run_shell", "description": "Run a command",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}}]
    tool_call = {"id": "call_1", "type": "function",
                 "function": {"name": "run_shell",
                              "arguments": "{\"cmd\": \"ls\"}"}}
    backend = StubBackend(responses=[
        {"id": "c1", "object": "chat.completion", "created": 0,
         "model": BASE_MODEL,
         "choices": [{"index": 0, "finish_reason": "tool_calls",
                      "message": {"role": "assistant", "content": None,
                                  "tool_calls": [tool_call]}}]},
        {"id": "c2", "object": "chat.completion", "created": 0,
         "model": BASE_MODEL,
         "choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant",
                                  "content": "done: two files"}}]},
    ])
    proxy, backend, _ = _proxy(tmp_path, backend=backend)
    first = {"model": "petri__si_safety__psr_kl_all",
             "messages": [{"role": "user", "content": "list the files"}],
             "tools": tools, "tool_choice": "auto", "max_tokens": 64}
    out1 = _ask(proxy, first)
    assert backend.payloads[0]["tools"] == tools
    assert backend.payloads[0]["tool_choice"] == "auto"
    got_call = out1["choices"][0]["message"]["tool_calls"][0]
    assert got_call == tool_call  # tool_calls pass back untouched
    # the resumed turn: assistant tool_calls + the simulated tool result
    second = {"model": "petri__si_safety__psr_kl_all",
              "messages": first["messages"] + [
                  {"role": "assistant", "content": None,
                   "tool_calls": [tool_call]},
                  {"role": "tool", "tool_call_id": "call_1",
                   "content": "a.txt\nb.txt"}],
              "tools": tools, "max_tokens": 64}
    out2 = _ask(proxy, second)
    up = backend.payloads[1]
    assert up["messages"][-1]["role"] == "tool"
    assert up["messages"][-1]["tool_call_id"] == "call_1"
    assert out2["choices"][0]["message"]["content"] == "done: two files"
    # both turns hit the SAME cell and were logged with its hash
    assert (proxy.request_log[0] == proxy.request_log[1]
            and proxy.request_log[0]["virtual_model"]
            == "petri__si_safety__psr_kl_all")


# --------------------------------------------------------------------------
# prompted vmodels: SI-prefix verification (H8), both directions
# --------------------------------------------------------------------------


def test_prompted_vmodel_accepts_exact_prefix(tmp_path):
    proxy, backend, cells = _proxy(tmp_path)
    system = SAFETY_PREFIX.decode() + "\nYou are being evaluated."
    out = _ask(proxy, _chat("petri__prompted__safety", system=system))
    up = backend.payloads[-1]
    assert up["model"] == BASE_MODEL  # passthrough weights
    # the prefix stays client-injected — the proxy VERIFIES, never injects
    assert up["messages"][0]["content"] == system
    cell = {c.cell_id: c for c in cells.canonical}["prompted__safety"]
    assert out["petri"]["intervention_hash"] == cell.intervention_hash


def test_prompted_vmodel_accepts_developer_role(tmp_path):
    """inspect_ai's openai provider labels the target system message role
    "developer" (OpenAI's modern system-role label), not "system" — the
    role that broke the gemma4e4b audit's prompted cells (2026-08-30, every
    request 400'd "got role 'developer'"). A developer message whose
    content starts with the exact SI prefix must verify like a system one:
    developer IS a system-role message, and the SI-prefix hash check is
    unchanged."""
    proxy, backend, cells = _proxy(tmp_path)
    system = SAFETY_PREFIX.decode() + "\n\nYou are being evaluated."
    request = {"model": "petri__prompted__safety",
               "messages": [{"role": "developer", "content": system},
                            {"role": "user", "content": "hello"}],
               "max_tokens": 32}
    out = _ask(proxy, request)
    assert backend.payloads[-1]["messages"][0]["content"] == system
    cell = {c.cell_id: c for c in cells.canonical}["prompted__safety"]
    assert out["petri"]["intervention_hash"] == cell.intervention_hash
    # a developer message that is NOT the SI prefix still refuses (the role
    # acceptance does not relax the content check)
    with pytest.raises(px.ProxyRefusal, match="masquerade|prefix"):
        _ask(proxy, {"model": "petri__prompted__safety",
                     "messages": [
                         {"role": "developer",
                          "content": "You are helpful." + " " * 60},
                         {"role": "user", "content": "hello"}],
                     "max_tokens": 32})


def test_prompted_vmodel_refuses_missing_prefix(tmp_path):
    proxy, _, _ = _proxy(tmp_path)
    # no system message at all
    with pytest.raises(px.ProxyRefusal, match="carrying the SI prefix"):
        _ask(proxy, _chat("petri__prompted__safety"))
    # a system message that is not the SI prefix
    with pytest.raises(px.ProxyRefusal, match="masquerade|prefix"):
        _ask(proxy, _chat("petri__prompted__safety",
                          system="You are a helpful assistant." + " " * 60))


def test_prompted_vmodel_refuses_wrong_prefix(tmp_path):
    proxy, _, _ = _proxy(tmp_path)
    # the OTHER SI's prefix on the safety vmodel — same length class, wrong
    # bytes: must refuse (a wrong prefix never masquerades, H8)
    wrong = CONSTITUTION_PREFIX.decode() + " suffix"
    with pytest.raises(px.ProxyRefusal):
        _ask(proxy, _chat("petri__prompted__safety", system=wrong))


def test_prompted_prefix_verifies_content_part_lists(tmp_path):
    proxy, backend, _ = _proxy(tmp_path)
    request = {"model": "petri__prompted__constitution",
               "messages": [
                   {"role": "system", "content": [
                       {"type": "text",
                        "text": CONSTITUTION_PREFIX.decode()},
                       {"type": "text", "text": "\nBe careful."}]},
                   {"role": "user", "content": "hello"}],
               "max_tokens": 8}
    _ask(proxy, request)
    assert backend.payloads[-1]["messages"] == request["messages"]


def test_non_prompted_leg_refuses_a_registered_si_prefix(tmp_path):
    """The reverse masquerade: base/object legs carrying a registered SI
    prefix would corrupt that leg's intervention identity — refused."""
    proxy, _, _ = _proxy(tmp_path)
    for vmodel in ("petri__base", "petri__si_safety__psr_kl_all"):
        with pytest.raises(px.ProxyRefusal, match="not a prompted"):
            _ask(proxy, _chat(vmodel,
                              system=SAFETY_PREFIX.decode() + " extra"))


def test_mixed_content_system_message_cannot_bypass_the_reverse_check(
        tmp_path):
    """Regression: a non-prompted request whose FIRST system content part
    is the registered SI prefix followed by a NON-TEXT part (an image)
    used to skip the reverse check entirely (the unverifiable-shape
    ProxyRefusal was caught and treated as permission) — the prompted
    instruction was present but the output was attributed to a
    non-prompted leg. The leading textual bytes must be inspected and
    refused BEFORE any non-text part is considered."""
    proxy, _, _ = _proxy(tmp_path)
    request = {"model": "petri__base",
               "messages": [
                   {"role": "system", "content": [
                       {"type": "text", "text": SAFETY_PREFIX.decode()},
                       {"type": "image_url",
                        "image_url": {"url": "http://example/x.png"}}]},
                   {"role": "user", "content": "hello"}],
               "max_tokens": 8}
    with pytest.raises(px.ProxyRefusal, match="not a prompted"):
        _ask(proxy, request)
    # a genuinely non-matching mixed system message still routes fine
    ok = {"model": "petri__base",
          "messages": [
              {"role": "system", "content": [
                  {"type": "text", "text": "You are a helpful assistant."},
                  {"type": "image_url",
                   "image_url": {"url": "http://example/x.png"}}]},
              {"role": "user", "content": "hello"}],
          "max_tokens": 8}
    assert _ask(proxy, ok)["model"] == "petri__base"


def test_prefix_in_a_later_system_message_also_refuses(tmp_path):
    """EVERY system message the upstream would accept is inspected — a
    registered prefix smuggled into a second system message corrupts the
    leg's identity exactly like one in the first."""
    proxy, _, _ = _proxy(tmp_path)
    request = {"model": "petri__base",
               "messages": [
                   {"role": "system", "content": "You are terse."},
                   {"role": "system",
                    "content": CONSTITUTION_PREFIX.decode() + " tail"},
                   {"role": "user", "content": "hello"}],
               "max_tokens": 8}
    with pytest.raises(px.ProxyRefusal, match="not a prompted"):
        _ask(proxy, request)


def test_proxy_refuses_to_start_without_prefix_specs(tmp_path):
    spec = _spec()
    source = _source(tmp_path)
    cells = _cells(spec, source)
    with pytest.raises(px.ProxyError, match="cannot verify"):
        px.RoutingProxy(px.routes_from_cells(cells), StubBackend(),
                        base_model=spec.target_model,
                        serving_mode=spec.serving_mode,
                        exact_replay_strategy="serialize",
                        prompted_prefixes=())  # H8: unverifiable = no start


# --------------------------------------------------------------------------
# the (virtual -> intervention-hash) log; refusal hygiene
# --------------------------------------------------------------------------


def test_request_log_maps_vmodel_to_intervention_hash(tmp_path):
    proxy, _, cells = _proxy(tmp_path)
    by_id = {c.cell_id: c for c in cells.all_served()}
    _ask(proxy, _chat("petri__base"))
    _ask(proxy, _chat("petri__si_safety__loreft"))
    _ask(proxy, _chat("petri__none__lens0"))
    got = [(e["virtual_model"], e["petri_intervention_hash"])
           for e in proxy.request_log]
    assert got == [
        ("petri__base", by_id["none"].intervention_hash),
        ("petri__si_safety__loreft",
         by_id["si_safety__loreft"].intervention_hash),
        # the no-op leg logs ITS OWN vmodel but the base's intervention
        # hash (same intervention, different serving)
        ("petri__none__lens0", by_id["none"].intervention_hash),
    ]
    assert (proxy.request_log[2]["petri_intervention_hash"]
            == proxy.request_log[0]["petri_intervention_hash"])


def test_client_smuggled_selectors_refused(tmp_path):
    proxy, _, _ = _proxy(tmp_path)
    for key in px.FORBIDDEN_CLIENT_KEYS:
        req = _chat("petri__base")
        req[key] = {"petri_object": "si_safety__cast", "expect_hash": "00"}
        with pytest.raises(px.ProxyRefusal, match="reserved selector"):
            _ask(proxy, req)


def test_steered_route_refuses_n_above_one(tmp_path):
    """E42 / E46: vLLM forks the extra sequences INSIDE the engine, where the
    lens worker never sees them, so a forked sequence on a steered route may
    run UNSTEERED and come back beside its steered sibling. The refusal has to
    sit at the request boundary — nothing downstream can tell the two apart."""
    proxy, backend, _ = _proxy(tmp_path)
    for key in px.FORK_REQUEST_KEYS:
        req = _chat("petri__si_constitution__cast")
        req[key] = 2
        with pytest.raises(px.SteeredForkRefusal, match="may run UNSTEERED") as ei:
            _ask(proxy, req)
        # typed: a 4xx the client can act on, recognised off the wire by CODE
        # (the class does not survive HTTP) — and still a ProxyRefusal, so the
        # aiohttp surface renders it like every other refusal
        assert isinstance(ei.value, px.ProxyRefusal)
        assert ei.value.http_status == 400
        assert ei.value.code == px.STEERED_FORK_REFUSAL_CODE == "steered_fork_unsupported"
        assert key in str(ei.value) and "k separate requests" in str(ei.value)
        assert not backend.payloads          # refused BEFORE anything went upstream
    # a value that cannot be PROVEN to be one sequence refuses too: a guard
    # that cannot read the count must not assume it is 1
    for bad in (0, -1, "2", 1.0, True, [1]):
        req = _chat("petri__si_constitution__cast")
        req["n"] = bad
        with pytest.raises(px.SteeredForkRefusal):
            _ask(proxy, req)
    assert not backend.payloads


def test_steered_route_accepts_n_absent_or_one_and_other_routes_are_unaffected(tmp_path):
    proxy, backend, _ = _proxy(tmp_path)
    # n absent (every request the repo's own engine sends), explicit 1, and an
    # explicit null all pass on a steered route and still carry the selector
    for req in (_chat("petri__si_constitution__cast"),
                {**_chat("petri__si_constitution__cast"), "n": 1},
                {**_chat("petri__si_constitution__cast"), "n": None}):
        _ask(proxy, req)
        assert px.UPSTREAM_EXTRA_ARGS_KEY in backend.payloads[-1]
    assert len(backend.payloads) == 3
    # passthrough / prompted / LoRA are unaffected: their intervention is not
    # applied by a per-request hook that a fork could miss
    for vmodel, system in (("petri__base", None),
                           ("petri__prompted__safety",
                            SAFETY_PREFIX.decode() + "\nYou are being evaluated."),
                           ("petri__si_safety__lora_r1", None)):
        req = _chat(vmodel, system=system)
        req["n"] = 4
        _ask(proxy, req)
        assert backend.payloads[-1]["n"] == 4
        assert px.UPSTREAM_EXTRA_ARGS_KEY not in backend.payloads[-1]


def test_streaming_refused_loudly(tmp_path):
    proxy, _, _ = _proxy(tmp_path)
    req = _chat("petri__base")
    req["stream"] = True
    with pytest.raises(px.ProxyRefusal, match="stream"):
        _ask(proxy, req)


def test_proxy_refuses_mode_mismatched_routes(tmp_path):
    """One audit run, one mode (H5): a route enumerated in another
    determinism mode cannot be served by this proxy."""
    spec = _spec()
    source = _source(tmp_path)
    cells = _cells(spec, source)
    with pytest.raises(px.ProxyError, match="mode"):
        px.RoutingProxy(px.routes_from_cells(cells), StubBackend(),
                        base_model=spec.target_model,
                        serving_mode="interleaved",
                        prompted_prefixes=px.prompted_prefix_specs(
                            cells, source))


# --------------------------------------------------------------------------
# the exact-replay mechanism: serialize, or ATTESTED batch
# invariance — never an unattested off-switch
# --------------------------------------------------------------------------


def _run_concurrent(proxy, n=4):
    async def go():
        await asyncio.gather(*(
            proxy.handle_chat_completions(_chat("petri__base"))
            for _ in range(n)))
    asyncio.run(go())


def test_exact_replay_serialize_strategy_serializes(tmp_path):
    backend = StubBackend(delay_s=0.01)
    proxy, backend, _ = _proxy(tmp_path, backend=backend)  # "serialize"
    _run_concurrent(proxy)
    assert backend.max_in_flight == 1  # single-request serialization


def test_exact_replay_needs_a_stated_strategy(tmp_path):
    with pytest.raises(px.ProxyError, match="exact_replay_strategy"):
        _proxy(tmp_path, exact_replay_strategy=None)  # never a default


def test_unattested_batch_invariant_refuses_to_start(tmp_path):
    """Disabling serialization is legal ONLY with proof the server-side
    mechanism is active: batch_invariant without a launch attestation
    carrying VLLM_BATCH_INVARIANT=1 must refuse at startup — an
    unattested off-switch would stamp nondeterministic outputs
    +exact_replay."""
    with pytest.raises(px.ProxyError, match="attestation"):
        _proxy(tmp_path, exact_replay_strategy="batch_invariant")
    with pytest.raises(px.ProxyError, match="attestation"):
        _proxy(tmp_path, exact_replay_strategy="batch_invariant",
               launch_attestation={BATCH_INVARIANT_ENV: "0"})


def test_attested_batch_invariant_may_run_unserialized(tmp_path):
    backend = StubBackend(delay_s=0.01)
    proxy, backend, _ = _proxy(
        tmp_path, backend=backend,
        exact_replay_strategy="batch_invariant",
        launch_attestation={BATCH_INVARIANT_ENV: "1"})
    _run_concurrent(proxy)
    assert backend.max_in_flight > 1  # the server-side mechanism carries it
    assert proxy.exact_replay_strategy == "batch_invariant"


def test_strategy_on_interleaved_refuses(tmp_path):
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    with pytest.raises(px.ProxyError, match="mislabel"):
        px.RoutingProxy(px.routes_from_cells(cells), StubBackend(),
                        base_model=spec.target_model,
                        serving_mode="interleaved",
                        exact_replay_strategy="serialize",
                        prompted_prefixes=px.prompted_prefix_specs(
                            cells, source))


# --------------------------------------------------------------------------
# routing-table export/load (the on-cluster proxy's input)
# --------------------------------------------------------------------------


def test_routing_table_round_trip(tmp_path):
    spec = _spec()
    source = _source(tmp_path)
    cells = _cells(spec, source)
    path = tmp_path / "routing_table.json"
    px.export_routing_table(cells, source, path,
                            base_model=spec.target_model,
                            serving_mode=spec.serving_mode,
                            exact_replay_strategy="serialize")
    table = px.load_routing_table(path)
    assert table["base_model"] == spec.target_model
    assert table["serving_mode"] == "exact_replay"
    assert table["exact_replay_strategy"] == "serialize"  # recorded
    assert set(table["routes"]) == set(px.routes_from_cells(cells))
    assert set(table["prompted_prefixes"]) == set(
        px.prompted_prefix_specs(cells, source))
    # the table carries lengths + hashes, never the SI text (containment)
    raw = path.read_text()
    assert SAFETY_PREFIX.decode() not in raw
    assert CONSTITUTION_PREFIX.decode() not in raw
    # and a proxy built from the loaded table behaves identically
    proxy = px.RoutingProxy(table["routes"], StubBackend(),
                            base_model=table["base_model"],
                            serving_mode=table["serving_mode"],
                            exact_replay_strategy=table[
                                "exact_replay_strategy"],
                            prompted_prefixes=table["prompted_prefixes"])
    with pytest.raises(px.ProxyRefusal, match="unknown virtual model"):
        _ask(proxy, _chat("petri__nope"))


def test_routing_table_requires_a_recorded_replay_strategy(tmp_path):
    spec = _spec()
    source = _source(tmp_path)
    cells = _cells(spec, source)
    path = tmp_path / "routing_table.json"
    # export refuses an exact_replay table without a stated mechanism
    with pytest.raises(px.ProxyError, match="exact_replay_strategy"):
        px.export_routing_table(cells, source, path,
                                base_model=spec.target_model,
                                serving_mode=spec.serving_mode)
    # a STALE table (pre-strategy export) refuses at load, loudly
    px.export_routing_table(cells, source, path,
                            base_model=spec.target_model,
                            serving_mode=spec.serving_mode,
                            exact_replay_strategy="batch_invariant")
    doc = json.loads(path.read_text())
    del doc["exact_replay_strategy"]
    path.write_text(json.dumps(doc))
    with pytest.raises(px.ProxyError, match="stale"):
        px.load_routing_table(path)


# --------------------------------------------------------------------------
# the target-role client wiring (petri_target)
# --------------------------------------------------------------------------


def test_target_client_config_and_env():
    cfg = pt.target_client_config("petri__si_safety__cast",
                                  "http://localhost:8814/v1")
    assert cfg.inspect_model == "openai/petri__si_safety__cast"
    assert cfg.env() == {"OPENAI_BASE_URL": "http://localhost:8814/v1",
                         "OPENAI_API_KEY": pt.DEFAULT_API_KEY}
    # 127.0.0.1 is loopback too
    pt.target_client_config("petri__base", "http://127.0.0.1:9000/v1")
    # a duplicate trailing /v1 (the runner's legacy append on top of the
    # engine's canonical /v1 base) collapses to the ONE canonical form
    dup = pt.target_client_config("petri__base",
                                  "http://localhost:8814/v1/v1")
    assert dup.base_url == "http://localhost:8814/v1"


def test_target_client_refuses_bad_wiring():
    with pytest.raises(pt.TargetClientError, match="prefix"):
        pt.target_client_config("gpt-4o", "http://localhost:8814/v1")
    with pytest.raises(pt.TargetClientError, match="loopback"):
        pt.target_client_config("petri__base", "http://10.1.2.3:8814/v1")
    # ... unless explicitly cluster-internal (the on-cluster gate/smoke)
    pt.target_client_config("petri__base", "http://10.1.2.3:8814/v1",
                            allow_cluster_internal=True)
    with pytest.raises(pt.TargetClientError, match="/v1"):
        pt.target_client_config("petri__base", "http://localhost:8814")
    with pytest.raises(pt.TargetClientError, match="URL"):
        pt.target_client_config("petri__base", "not-a-url")


def test_make_async_client_is_import_guarded():
    cfg = pt.target_client_config("petri__base", "http://localhost:1/v1")
    try:
        import openai  # noqa: F401
        pytest.skip("openai installed here; the guard path is for the CPU "
                    "test venv")
    except ImportError:
        pass
    with pytest.raises(pt.TargetClientError, match="petri"):
        pt.make_async_client(cfg)


# --------------------------------------------------------------------------
# ProxyClientEngine (the runtime.make_engine handle)
# --------------------------------------------------------------------------


def test_proxy_client_engine_wiring(monkeypatch):
    eng = px.ProxyClientEngine("http://localhost:8814",
                               identity="proxy_vllm+lens+exact_replay")
    assert eng.name == "proxy_vllm+lens+exact_replay"
    # ONE canonical base representation: root URLs normalize to .../v1
    assert eng.base_url == "http://localhost:8814/v1"
    assert px.ProxyClientEngine(
        "http://localhost:8814/v1",
        identity="proxy_vllm+lens+exact_replay").base_url == eng.base_url
    calls = []

    def fake_http(self, url, payload):
        calls.append((url, payload))
        if url.endswith("/models"):
            return {"data": [{"id": "petri__base", "owned_by": "petri"},
                             {"id": "petri__si_safety__cast",
                              "owned_by": "petri"}],
                    "petri_proxy": {"version": 1,
                                    "serving_mode": "exact_replay"}}
        return {"choices": [], "model": payload["model"]}

    monkeypatch.setattr(px.ProxyClientEngine, "_http_json", fake_http)
    eng.load(require_virtual_models=["petri__si_safety__cast"])
    assert eng.served_models == ("petri__base", "petri__si_safety__cast")
    assert eng.proxy_identity == {"version": 1,
                                  "serving_mode": "exact_replay"}
    assert calls[0][0] == "http://localhost:8814/v1/models"
    out = eng.chat_completions(_chat("petri__base"))
    assert out["model"] == "petri__base"
    assert calls[-1][0] == "http://localhost:8814/v1/chat/completions"


def test_engine_base_url_feeds_target_client_config_directly(monkeypatch):
    """The BLOCKER regression: the engine and the target client must
    speak ONE base-URL convention — eng.base_url hands to
    target_client_config unchanged (and survives the runner's legacy
    '+ /v1' append), with the engine's own calls hitting /v1/models and
    /v1/chat/completions exactly once each."""
    eng = px.ProxyClientEngine("http://localhost:8814",
                               identity="proxy_vllm+lens+exact_replay")
    cfg = pt.target_client_config("petri__base", eng.base_url)
    assert cfg.base_url == "http://localhost:8814/v1"
    legacy = pt.target_client_config("petri__base", eng.base_url + "/v1")
    assert legacy.base_url == cfg.base_url  # never /v1/v1


def test_engine_health_check_refuses_a_foreign_endpoint(monkeypatch):
    """A stale or unrelated OpenAI endpoint ({'data':[{'id':'gpt-4o'}]})
    must fail the load-time health check — not start the audit and fail
    at the first petri__ request."""
    eng = px.ProxyClientEngine("http://localhost:8814",
                               identity="proxy_vllm+lens+exact_replay")
    monkeypatch.setattr(
        px.ProxyClientEngine, "_http_json",
        lambda self, url, payload: {"data": [{"id": "gpt-4o"}]})
    with pytest.raises(px.ProxyError, match="petri"):
        eng.load()
    # wrong ownership refuses too
    monkeypatch.setattr(
        px.ProxyClientEngine, "_http_json",
        lambda self, url, payload: {"data": [
            {"id": "petri__base", "owned_by": "someone-else"}]})
    with pytest.raises(px.ProxyError, match="owned"):
        eng.load()


def test_engine_health_check_requires_the_pending_cells_vmodel(monkeypatch):
    """A proxy whose routing table predates the pending cell must fail at
    load, before the expensive audit starts."""
    eng = px.ProxyClientEngine("http://localhost:8814",
                               identity="proxy_vllm+lens+exact_replay")
    monkeypatch.setattr(
        px.ProxyClientEngine, "_http_json",
        lambda self, url, payload: {"data": [{"id": "petri__base"}]})
    with pytest.raises(px.ProxyError, match="required virtual model"):
        eng.load(require_virtual_models=["petri__si_safety__loreft"])


def test_engine_wrong_token_fails_at_load(tmp_path, monkeypatch):
    """With the proxy authenticating, a wrong client key fails the
    health check itself (models is authenticated like chat) — never
    deferred to the first generation."""
    proxy, _, _ = _proxy(tmp_path, api_token="sekrit")
    eng = px.ProxyClientEngine("http://localhost:8814",
                               identity="proxy_vllm+lens+exact_replay",
                               api_key="wrong-key")

    def fake_http(self, url, payload):
        # emulate the HTTP surface: handle_models sees this client's
        # token; a refusal comes back as the HTTP-error ProxyError the
        # real transport raises on a 401 body
        try:
            return proxy.handle_models(
                authorization=f"Bearer {self.api_key}")
        except px.ProxyRefusal as e:
            raise px.ProxyError(
                f"proxy {url} returned {e.http_status}: {e}") from e

    monkeypatch.setattr(px.ProxyClientEngine, "_http_json", fake_http)
    with pytest.raises(px.ProxyError, match="401"):
        eng.load()
    eng.api_key = "sekrit"
    eng.load()
    assert "petri__base" in eng.served_models


def test_proxy_client_engine_refuses_bad_identity_and_missing_url(
        monkeypatch):
    from types import SimpleNamespace

    with pytest.raises(px.ProxyError, match="proxy_vllm"):
        px.ProxyClientEngine("http://localhost:1",
                             identity="hf_shim+apply+exact_replay")
    monkeypatch.delenv(px.PROXY_BASE_URL_ENV, raising=False)
    with pytest.raises(px.ProxyError, match=px.PROXY_BASE_URL_ENV):
        px.ProxyClientEngine.from_manifest(
            SimpleNamespace(engine="proxy_vllm+lora+exact_replay"))
    monkeypatch.setenv(px.PROXY_BASE_URL_ENV, "http://localhost:8814")
    eng = px.ProxyClientEngine.from_manifest(
        SimpleNamespace(engine="proxy_vllm+lora+exact_replay"))
    assert eng.base_url == "http://localhost:8814/v1"  # canonical form


def test_proxy_client_engine_dead_tunnel_is_loud():
    eng = px.ProxyClientEngine("http://127.0.0.1:1",  # nothing listens
                               identity="proxy_vllm+lens+exact_replay",
                               timeout_s=0.5)
    with pytest.raises(px.ProxyError, match="tunnel"):
        eng.load()


# --- upstream status class survives the hop (context-overflow triage) ----------
#. A vLLM 400 is a DETERMINISTIC client error —
# the pinned 0.25.1 answers an over-length transcript with HTTP 400
# "This model's maximum context length is ..." — and the old blanket 502
# re-presented it as a transient server error, which the pinned
# inspect-ai 0.3.251 retries FOREVER by default: one over-length audit
# turn hung its whole eval task group. 4xx passes through; 5xx (and
# anything else) still presents as 502.


def test_upstream_4xx_passes_through_its_status():
    err = px.upstream_refusal(
        400, "This model's maximum context length is 16384 tokens. "
             "However, you requested 1024 output tokens and your prompt "
             "contains 19813 input tokens, for a total of 20837 tokens.")
    assert err.http_status == 400
    assert err.code == "upstream_error"
    # the vLLM body text survives for the runner's overflow classifier
    assert "maximum context length" in str(err)
    assert px.upstream_refusal(404, "no such adapter").http_status == 404
    assert px.upstream_refusal(422, "bad shape").http_status == 422


def test_upstream_5xx_and_weird_statuses_present_as_502():
    assert px.upstream_refusal(500, "engine core died").http_status == 502
    assert px.upstream_refusal(503, "loading").http_status == 502
    assert px.upstream_refusal(302, "redirect?!").http_status == 502
    for status in (400, 500):
        assert px.upstream_refusal(status, "x").code == "upstream_error"


def test_upstream_refusal_never_reshapes_into_an_openai_overflow_code():
    # shaping the body as OpenAI's context_length_exceeded would make
    # inspect-ai swallow the overflow into stop_reason="model_length"
    # and the audit would CONTINUE, silently truncated — the code must
    # stay a non-OpenAI marker for every status
    err = px.upstream_refusal(
        400, "This model's maximum context length is 16384 tokens.")
    assert err.code != "context_length_exceeded"


# --------------------------------------------------------------------------
# per-request lens failures: the proxy refuses a poisoned response
# (the loud half of the 2026-08-30 EngineCore containment)
# --------------------------------------------------------------------------


_FAILURE_RECORD = {
    "req_id": "chatcmpl-poisoned", "cell_id": "si_safety__psr_kl_all",
    "layer": 0, "site": "block", "error_type": "AdapterError",
    "error": ("hook op 'relu_gated_vector' at layer 0: the assistant "
              "header occurs NOWHERE in this prompt"),
}


def _response(rid):
    return {
        "id": rid, "object": "chat.completion", "created": 0,
        "model": BASE_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": "poisoned reply"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                  "total_tokens": 2},
    }


def test_a_poisoned_lens_response_is_refused_not_forwarded(tmp_path):
    backend = StubBackend(
        responses=[_response("chatcmpl-poisoned")],
        request_failures={"chatcmpl-poisoned": dict(_FAILURE_RECORD)})
    proxy, _, _ = _proxy(tmp_path, backend=backend)
    with pytest.raises(px.ProxyRefusal) as exc:
        _ask(proxy, _chat("petri__si_constitution__cast"))
    assert exc.value.http_status == 500
    assert exc.value.code == "lens_request_failed"
    # the refusal carries the worker's own diagnosis, verbatim enough
    # to act on — never a bare "internal error"
    assert "occurs NOWHERE" in str(exc.value)
    assert "si_safety__psr_kl_all" in str(exc.value)
    assert backend.failure_probes == ["chatcmpl-poisoned"]


def test_a_clean_lens_response_is_post_checked_and_forwarded(tmp_path):
    backend = StubBackend(responses=[_response("chatcmpl-clean")])
    proxy, _, _ = _proxy(tmp_path, backend=backend)
    out = _ask(proxy, _chat("petri__si_constitution__cast"))
    assert out["choices"][0]["message"]["content"] == "poisoned reply"
    assert backend.failure_probes == ["chatcmpl-clean"]


def test_non_lens_routes_are_not_post_checked(tmp_path):
    backend = StubBackend()
    proxy, _, _ = _proxy(tmp_path, backend=backend)
    _ask(proxy, _chat("petri__base"))                    # passthrough
    _ask(proxy, _chat("petri__si_safety__lora_r1"))      # native LoRA
    assert backend.failure_probes == []


def test_a_lens_response_without_an_id_is_refused(tmp_path):
    """No id = the post-check cannot run = the steering cannot be
    verified; forwarding anyway would be the silent path."""
    resp = _response("chatcmpl-x")
    del resp["id"]
    backend = StubBackend(responses=[resp])
    proxy, _, _ = _proxy(tmp_path, backend=backend)
    with pytest.raises(px.ProxyRefusal, match="no 'id'"):
        _ask(proxy, _chat("petri__si_constitution__cast"))


def test_a_backend_without_the_post_check_refuses_lens_routes(tmp_path):
    class BareBackend(StubBackend):
        take_request_failure = None

    backend = BareBackend()
    proxy, _, _ = _proxy(tmp_path, backend=backend)
    with pytest.raises(px.ProxyRefusal, match="post-check"):
        _ask(proxy, _chat("petri__si_constitution__cast"))
    # non-lens routes are unaffected by the missing surface
    _ask(proxy, _chat("petri__base"))


# ---------------------------------------------------------------------------
# THE THINKING-ON GATE:
# the whole refusal matrix against three thinking_header fixtures (moves-true /
# moves-false / switch-false), base-route acceptance and the recorded
# render_regime, chat_template_kwargs forwarded verbatim and petri_thinking_on
# never, the tokenize SHAPE validation, the /v1/models echo, no attestation ->
# thinking_on_unattested, and the declared none__lens0 twin (Q7). Every code
# is pinned as the literal the harness recognises off the wire.
# ---------------------------------------------------------------------------

from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402

LENS = "si_safety__psr_kl_all"
LENS_VM = "petri__si_safety__psr_kl_all"
THINKING_VM = f"petri__{LENS}@thinking"
TWIN = dc.NOOP_LENS_CELL_ID
TWIN_VM = f"petri__{TWIN}"
OFF7 = [901, 902, 903, 904, 905, 906, 907]
ON5 = [901, 902, 903, 904, 908]
MOVES_TRUE = {"switch": True, "moves": True, "off_ids": OFF7, "on_ids": ON5}
MOVES_FALSE = {"switch": True, "moves": False, "off_ids": OFF7[:3], "on_ids": OFF7[:3]}
SWITCH_FALSE = {"switch": False, "moves": None, "off_ids": OFF7[:4], "on_ids": None}
DECLARED = {f"{LENS}@thinking": {"serve_out_of_support": ["thinking"]},
            TWIN: {"serve_out_of_support": ["thinking"]}}


def _thinking_source(tmp_path):
    """The canonical source plus the @thinking variant of the psr cell (the
    SAME artifact, the base's spec with the overlay)."""
    src = _source(tmp_path)
    entries = dict(src.entries)
    base = entries[LENS]
    entries[f"{LENS}@thinking"] = dc.RegistryEntry(
        cell_id=f"{LENS}@thinking", kind=base.kind, si_id=base.si_id,
        object_path=base.object_path, serving_path="lens",
        resolved_spec=sd.apply_variant(base.resolved_spec, "thinking"))
    return dc.MappingCellSource(entries)


def _thinking_proxy(tmp_path, *, header=MOVES_TRUE, declarations=DECLARED,
                    declarations_error=None, with_variant=True,
                    agent_loop_render=None, **kw):
    spec = _spec(serving_mode="interleaved",
                 cells=list(CANONICAL_22) + ([f"{LENS}@thinking"] if with_variant else []))
    source = _thinking_source(tmp_path) if with_variant else _source(tmp_path)
    cells = _cells(spec, source)
    backend = StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=None, agent_loop_render=agent_loop_render or _TailRender(),
        thinking_header=header, lens_serve_declarations=declarations,
        lens_serve_declarations_error=declarations_error, **kw)
    return proxy, backend, cells


def _think_chat(model, *, enable=True, declare=True, kwargs=...):
    body = {"model": model, "max_tokens": 8,
            "messages": [{"role": "user", "content": "What is 2+2? Think first."}]}
    if kwargs is not ...:
        body[px.CHAT_TEMPLATE_KWARGS_KEY] = kwargs
    elif enable is not None:
        body[px.CHAT_TEMPLATE_KWARGS_KEY] = {px.ENABLE_THINKING_KEY: enable}
    if declare:
        body[px.THINKING_ON_KEY] = True
    return body


class _LoopRender(_TailRender):
    """The hermetic probe with ONE more behaviour: a request whose message
    structure is a turn of the agent loop renders as a NON-tail render (the
    header followed by the open model turn), so a DECLARED ``petri_agent_loop``
    verifies here — what the §2.5.6 row-8 narrowing needs SERVED, not only
    refused. Anything else renders at the tail as before."""

    def render(self, messages, tools):
        if px.agent_loop_shape({"messages": list(messages)}) is None:
            return super().render(messages, tools)
        return (1, 2, 3, *self.header_ids, 5, 6)


def _think_prefill(model, *, enable=True, declare=True):
    """A well-formed SEEDED render under the ON kwargs — the shape gemma's
    prefill legs send (``prefill_logprob.build_prefill_request``) — with
    ``petri_response_prefill``
    declared beside the thinking-ON declaration."""
    body = _think_chat(model, enable=enable, declare=declare)
    body["messages"] = [{"role": "user", "content": "Which letter? Reply A-D."},
                        {"role": "assistant", "content": "Answer:"}]
    body["continue_final_message"] = True
    body["add_generation_prompt"] = False
    body[px.RESPONSE_PREFILL_KEY] = True
    return body


def _think_loop(model, *, enable=True, declare=True):
    """A well-formed turn of the agent loop (a tool result awaiting the model)
    under the ON kwargs, with ``petri_agent_loop`` declared beside the
    thinking-ON declaration."""
    body = _think_chat(model, enable=enable, declare=declare)
    body["messages"] = [
        {"role": "user", "content": "Please look up the weather."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_0001", "type": "function",
                         "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_0001", "content": "Sunny."}]
    body[px.AGENT_LOOP_KEY] = True
    return body


def _refuses(proxy, body, code):
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, body)
    assert ei.value.http_status == 400 and ei.value.code == code, (ei.value.code, str(ei.value))
    return ei.value


def test_the_refusal_code_literals_are_the_contracts():
    """§2.5.6 / §2.5.1 byte for byte — the harness pins these off the wire."""
    assert px.THINKING_ON_KEY == "petri_thinking_on"
    assert px.CHAT_TEMPLATE_KWARGS_KEY == "chat_template_kwargs"
    assert px.ENABLE_THINKING_KEY == "enable_thinking"
    assert px.SERVE_DECLARATIONS_KEY == "lens_serve_declarations"
    assert px.SERVE_DECLARATIONS_ERROR_KEY == "lens_serve_declarations_error"
    assert px.THINKING_HEADER_KEY == "thinking_header"
    assert px.LENS_SALT_POLICY_KEY == "lens_salt_policy"
    assert px.OBJECT_SHA256_BY_CELL_KEY == "object_artifact_sha256_by_cell"
    # + the thinking-ON STANDUP's one addition (RoutingProxy(thinking_mode="on"), the
    # sampled-rerun lineage merged 2026-09-11): a request flipping the pinned regime OFF
    assert px.THINKING_REFUSAL_CODES == (
        "thinking_kwargs_invalid", "thinking_kwargs_inert", "thinking_on_undeclared",
        "thinking_on_invalid", "thinking_on_mismatch", "thinking_on_not_lens",
        "thinking_on_cell_not_declared", "thinking_on_unattested",
        "thinking_regime_mismatch")
    assert px.THINKING_MODES == ("off", "on") and px.THINKING_MODE_KEY == "thinking_mode"
    assert px.ALL_TURNS_REFUSAL_CODES == (
        "all_turns_span_mismatch", "all_turns_render_refused",
        "all_turns_expected_unavailable", "all_turns_render_failed")
    assert (px.RENDER_REGIME_THINKING_ON, px.RENDER_REGIME_THINKING_OFF,
            px.RENDER_REGIME_UNSPECIFIED) == ("thinking_on", "thinking_off", "unspecified")
    assert px.LEDGER_KEY_RENDER_REGIME == "render_regime"
    assert px.POLICY_DECLARED_SHARES_NOTHING == "declared-request-unique"
    assert px.POLICY_FORCED_REQUEST_UNIQUE == "forced-request-unique"
    assert px.LENS_SALT_POLICIES == ("attested", "request_unique_all")
    # the two regime literals are UNCHANGED: a declared request wears the
    # ordinary request-unique (§2.5.11 Q9)
    assert px.SALT_REGIMES == ("cell-stable", "request-unique")


# --- the matrix, moves-TRUE (Qwen3.5) -------------------------------------------


def test_declared_on_request_on_the_declared_cell_is_served_and_injected(tmp_path):
    proxy, backend, _ = _thinking_proxy(tmp_path)
    out = _ask(proxy, _think_chat(THINKING_VM))
    up = backend.payloads[-1]
    sel = up[px.UPSTREAM_EXTRA_ARGS_KEY]
    assert sel["thinking_on"] == 1 and not isinstance(sel["thinking_on"], bool)
    assert sel["petri_object"] == f"{LENS}@thinking"
    # the declaration never travels upstream; the render kwargs do, verbatim
    assert px.THINKING_ON_KEY not in up
    assert up[px.CHAT_TEMPLATE_KWARGS_KEY] == {"enable_thinking": True}
    # the stamp (§2.5.11 Q9 / Q11): regime request-unique, policy the DECLARED
    # label, render regime thinking_on
    assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == "request-unique"
    assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == "declared-request-unique"
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    row = dict(proxy.request_log[-1])
    assert row[px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    assert row[px.LEDGER_KEY_RESUMED_POLICY] == "declared-request-unique"


def test_declared_twin_is_served_like_any_declared_cell(tmp_path):
    """Q7: the none__lens0 twin inherits its donor's declaration and is looked
    up by route.cell_id — its ON probe legs are declared requests."""
    proxy, backend, _ = _thinking_proxy(tmp_path)
    out = _ask(proxy, _think_chat(TWIN_VM))
    sel = backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    assert sel["petri_object"] == TWIN and sel["scale"] == 0.0 and sel["thinking_on"] == 1
    assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == "declared-request-unique"


def test_undeclared_twin_refuses_cell_not_declared(tmp_path):
    """An OFF-only standup (none + base): the twin's donor is the base, so the
    map has no entry for it and an ON probe refuses (§2.5.4 last row)."""
    proxy, _, _ = _thinking_proxy(tmp_path, declarations={}, with_variant=False)
    _refuses(proxy, _think_chat(TWIN_VM), "thinking_on_cell_not_declared")
    _refuses(proxy, _think_chat(LENS_VM), "thinking_on_cell_not_declared")


def test_matrix_moves_true(tmp_path):
    proxy, backend, _ = _thinking_proxy(tmp_path)
    # kwargs true, no declaration -> undeclared (declared cell AND base cell).
    # This proxy holds NO lens_served_positions map (as every proxy did before
    # the §2.5.4 narrowing of 2026-09-09), so the refusal is the fail-closed
    # reading — the same code an older routing table produces; the accepted
    # twin of this row (an `all` / `last` cell the table names) lives in the
    # served-positions section at the end of this module
    _refuses(proxy, _think_chat(THINKING_VM, declare=False), "thinking_on_undeclared")
    _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    # `petri_thinking_on: false` is "not declared"
    body = _think_chat(THINKING_VM, declare=False); body[px.THINKING_ON_KEY] = False
    _refuses(proxy, body, "thinking_on_undeclared")
    # declaration on the undeclared base cell -> cell_not_declared
    _refuses(proxy, _think_chat(LENS_VM), "thinking_on_cell_not_declared")
    # declaration not a bool
    body = _think_chat(THINKING_VM); body[px.THINKING_ON_KEY] = "true"
    _refuses(proxy, body, "thinking_on_invalid")
    # declaration without kwargs true -> mismatch (absent, false)
    _refuses(proxy, _think_chat(THINKING_VM, enable=None), "thinking_on_mismatch")
    _refuses(proxy, _think_chat(THINKING_VM, enable=False), "thinking_on_mismatch")
    # declaration beside a non-tail declaration -> mismatch (wins over the
    # other gates: it runs first). §2.5.6 row 8 is qualified MOVES-TRUE
    # (2026-09-07): on THIS template the switch moves the header, so a bare
    # key and a well-formed seeded / agent-loop body refuse alike, on every
    # lens route, the refusal naming its ground
    for other in (px.RESPONSE_PREFILL_KEY, px.AGENT_LOOP_KEY):
        body = _think_chat(THINKING_VM); body[other] = True
        e = _refuses(proxy, body, "thinking_on_mismatch")
        assert "MOVES the assistant header" in str(e)
    for body in (_think_prefill(THINKING_VM), _think_loop(THINKING_VM),
                 _think_prefill(LENS_VM), _think_loop(TWIN_VM)):
        e = _refuses(proxy, body, "thinking_on_mismatch")
        assert "MOVES the assistant header" in str(e)
    # declaration on a non-lens route -> not_lens
    _refuses(proxy, _think_chat("petri__base"), "thinking_on_not_lens")
    prompted = _think_chat("petri__prompted__safety")
    prompted["messages"] = [{"role": "system", "content": SAFETY_PREFIX.decode()}] + prompted["messages"]
    _refuses(proxy, prompted, "thinking_on_not_lens")
    # kwargs shape, any route
    for bad in ("x", {"enable_thinking": "yes"}, {"enable_thinking": True, "foo": 1},
                {"add_special_tokens": False}):
        _refuses(proxy, _think_chat("petri__base", declare=False, kwargs=bad), "thinking_kwargs_invalid")
        _refuses(proxy, _think_chat(THINKING_VM, kwargs=bad), "thinking_kwargs_invalid")
    assert backend.payloads == []  # every refusal happened BEFORE the engine
    # enable_thinking: false / absent -> accepted everywhere, regime recorded
    for vm in ("petri__base", LENS_VM, THINKING_VM, TWIN_VM):
        out = _ask(proxy, _think_chat(vm, enable=False, declare=False))
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_off"
        out = _ask(proxy, _think_chat(vm, enable=None, declare=False))
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "unspecified"
        assert px.CHAT_TEMPLATE_KWARGS_KEY not in backend.payloads[-1]
    # an OFF request on a declared cell carries NO thinking_on fact
    assert "thinking_on" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]


def test_base_route_accepts_the_on_render_undeclared_and_records_the_regime(tmp_path):
    """Q3 (§2.5.5): the unsteered thinking-ON `none` cell is in support by
    construction; no declaration, the kwargs forwarded, the regime stamped."""
    proxy, backend, _ = _thinking_proxy(tmp_path)
    out = _ask(proxy, _think_chat("petri__base", declare=False))
    assert backend.payloads[-1][px.CHAT_TEMPLATE_KWARGS_KEY] == {"enable_thinking": True}
    assert px.THINKING_ON_KEY not in backend.payloads[-1]
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == "cell-stable"
    assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == "not-consulted"
    assert dict(proxy.request_log[-1])[px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    # a prompted route too
    out = _ask(proxy, {"model": "petri__prompted__safety", "max_tokens": 4,
                       "messages": [{"role": "system", "content": SAFETY_PREFIX.decode()},
                                    {"role": "user", "content": "hi"}],
                       px.CHAT_TEMPLATE_KWARGS_KEY: {"enable_thinking": True}})
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"


# --- moves-FALSE (gemma-4) and switch-FALSE (llama) -------------------------------


def test_matrix_moves_false_validates_but_never_injects(tmp_path):
    """gemma's in-support ON battery (P13): the declaration is REQUIRED on
    every lens route, validated, recorded — and the fact is NOT injected."""
    proxy, backend, _ = _thinking_proxy(tmp_path, header=MOVES_FALSE, declarations={},
                                        with_variant=False, agent_loop_render=_LoopRender())
    out = _ask(proxy, _think_chat(LENS_VM))                  # declared: served
    assert "thinking_on" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    assert backend.payloads[-1][px.CHAT_TEMPLATE_KWARGS_KEY] == {"enable_thinking": True}
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == "conditional" or \
        out["petri"][px.LEDGER_KEY_RESUMED_POLICY] in px.POLICY_NON_VALUES
    # no lens_served_positions map on this proxy: the undeclared refusal is the
    # fail-closed reading (§2.5.4 narrowing, 2026-09-09 — see the module's
    # served-positions section for the cells the table lets through)
    _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    _refuses(proxy, _think_chat(TWIN_VM, declare=False), "thinking_on_undeclared")
    out = _ask(proxy, _think_chat(TWIN_VM))                  # the twin, declared: served
    assert "thinking_on" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    # §2.5.6 row 8 NARROWED to moves-TRUE (2026-09-07): here the switch is
    # certified NOT to move the header, so the declaration injects nothing and
    # a declared ON render beside petri_response_prefill / petri_agent_loop is
    # today's OFF shape plus what the ON mode adds outside the header — SERVED:
    # the declaration validated and not injected, the non-tail fact injected as
    # ever, the kwargs forwarded verbatim, the regime stamped (gemma's prefill
    # legs: prefill_logprob.build_prefill_request)
    for body, fact in ((_think_prefill(LENS_VM), "response_prefill"),
                       (_think_loop(LENS_VM), "agent_loop"),
                       (_think_prefill(TWIN_VM), "response_prefill")):
        n = len(backend.payloads)
        out = _ask(proxy, body)
        assert len(backend.payloads) == n + 1
        up = backend.payloads[-1]
        sel = up[px.UPSTREAM_EXTRA_ARGS_KEY]
        assert sel[fact] == 1 and "thinking_on" not in sel
        assert up[px.CHAT_TEMPLATE_KWARGS_KEY] == {"enable_thinking": True}
        assert px.THINKING_ON_KEY not in up
        assert px.RESPONSE_PREFILL_KEY not in up and px.AGENT_LOOP_KEY not in up
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
        assert dict(proxy.request_log[-1])[px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    # ... and every OTHER row still binds such a body: the declaration must be
    # a boolean, sit beside enable_thinking: true, on a lens route, and be
    # present at all; the non-tail gates then judge the shape as they always did
    served = len(backend.payloads)
    _refuses(proxy, _think_prefill(LENS_VM, enable=False), "thinking_on_mismatch")
    _refuses(proxy, _think_prefill(LENS_VM, enable=None), "thinking_on_mismatch")
    _refuses(proxy, _think_loop(LENS_VM, enable=False), "thinking_on_mismatch")
    _refuses(proxy, _think_prefill("petri__base"), "thinking_on_not_lens")
    _refuses(proxy, _think_loop("petri__base"), "thinking_on_not_lens")
    _refuses(proxy, _think_prefill(LENS_VM, declare=False), "thinking_on_undeclared")
    _refuses(proxy, _think_loop(LENS_VM, declare=False), "thinking_on_undeclared")
    body = _think_prefill(LENS_VM); body[px.THINKING_ON_KEY] = "true"
    _refuses(proxy, body, "thinking_on_invalid")
    body = _think_prefill(LENS_VM); del body["continue_final_message"]
    _refuses(proxy, body, "response_prefill_mismatch")
    body = _think_prefill(LENS_VM); body[px.AGENT_LOOP_KEY] = True
    _refuses(proxy, body, "agent_loop_mismatch")
    assert len(backend.payloads) == served


def test_matrix_switch_false_is_inert_everywhere(tmp_path):
    proxy, backend, _ = _thinking_proxy(tmp_path, header=SWITCH_FALSE, declarations={},
                                        with_variant=False)
    for vm in ("petri__base", LENS_VM, TWIN_VM):
        _refuses(proxy, _think_chat(vm, declare=False), "thinking_kwargs_inert")
    _refuses(proxy, _think_chat(LENS_VM), "thinking_kwargs_inert")
    # a switch-less template is NOT the moves-false exemption of §2.5.6 row 8
    # (its moves is None): the ON kwargs are inert first, and without them the
    # declaration beside a non-tail key refuses on row 8 as it always did
    _refuses(proxy, _think_prefill(LENS_VM), "thinking_kwargs_inert")
    e = _refuses(proxy, _think_prefill(LENS_VM, enable=False), "thinking_on_mismatch")
    assert "NOT certified" in str(e)
    # false / absent are fine
    _ask(proxy, _think_chat(LENS_VM, enable=False, declare=False))
    assert len(backend.payloads) == 1


# --- fail closed: no attestation, no header fact ---------------------------------


def test_no_attested_declarations_refuses_unattested(tmp_path):
    proxy, _, _ = _thinking_proxy(tmp_path, declarations=None,
                                  declarations_error="synthetic: workers disagree")
    e = _refuses(proxy, _think_chat(THINKING_VM), "thinking_on_unattested")
    assert "workers disagree" in str(e)
    # ... and an attested EMPTY map is a different world: cell_not_declared
    proxy2, _, _ = _thinking_proxy(tmp_path, declarations={})
    _refuses(proxy2, _think_chat(THINKING_VM), "thinking_on_cell_not_declared")


def test_an_old_routing_table_without_the_header_fact_refuses(tmp_path):
    proxy, _, _ = _thinking_proxy(tmp_path, header=None)
    _refuses(proxy, _think_chat(THINKING_VM), "thinking_on_unattested")
    # ... while a base-route ON request is still accepted (no mask there)
    _ask(proxy, _think_chat("petri__base", declare=False))
    # ... and no fact is not the moves-false exemption of §2.5.6 row 8: a
    # declared seeded / agent-loop ON body refuses thinking_on_mismatch, as before
    for body in (_think_prefill(THINKING_VM), _think_loop(THINKING_VM)):
        e = _refuses(proxy, body, "thinking_on_mismatch")
        assert "NOT certified" in str(e)


# --- construction-time validation ----------------------------------------------


def test_malformed_declarations_or_header_refuse_to_start(tmp_path):
    with pytest.raises(px.ProxyError, match="not exactly one closed"):
        _thinking_proxy(tmp_path, declarations={LENS: {"serve_out_of_support": ["thinking"],
                                                       "positions": "response"}})
    with pytest.raises(px.ProxyError):
        _thinking_proxy(tmp_path, declarations={LENS: {"serve_out_of_support": "thinking"}})
    with pytest.raises(px.ProxyError):
        _thinking_proxy(tmp_path, declarations={LENS: {"serve_experiment": "nope"}})
    with pytest.raises(px.ProxyError, match="contradiction"):
        _thinking_proxy(tmp_path, declarations={}, declarations_error="x")
    with pytest.raises(px.ProxyError, match="moves"):
        _thinking_proxy(tmp_path, header={**MOVES_TRUE, "moves": False})
    with pytest.raises(px.ProxyError, match="unknown key"):
        _thinking_proxy(tmp_path, header={**MOVES_TRUE, "extra": 1})
    with pytest.raises(px.ProxyError, match="lens_salt_policy"):
        _thinking_proxy(tmp_path, lens_salt_policy="sometimes")


# --- /v1/models echo + tokenize shape ----------------------------------------------


def test_models_payload_echoes_the_declarations_header_and_policy(tmp_path):
    proxy, _, _ = _thinking_proxy(tmp_path)
    block = proxy.models_payload()["petri_proxy"]
    assert block[px.SERVE_DECLARATIONS_KEY] == DECLARED
    assert block[px.SERVE_DECLARATIONS_ERROR_KEY] is None
    assert block[px.THINKING_HEADER_KEY] == MOVES_TRUE
    assert block[px.LENS_SALT_POLICY_KEY] == "attested"
    assert px.OBJECT_SHA256_BY_CELL_KEY not in block
    # Q10: an attested EMPTY map is PRESENT and empty, the error PRESENT and null
    empty, _, _ = _thinking_proxy(tmp_path, declarations={}, with_variant=False)
    b2 = empty.models_payload()["petri_proxy"]
    assert b2[px.SERVE_DECLARATIONS_KEY] == {} and b2[px.SERVE_DECLARATIONS_ERROR_KEY] is None
    # attestation failure: the map ABSENT, the error a non-null string
    failed, _, _ = _thinking_proxy(tmp_path, declarations=None, declarations_error="why not",
                                   with_variant=False)
    b3 = failed.models_payload()["petri_proxy"]
    assert px.SERVE_DECLARATIONS_KEY not in b3 and b3[px.SERVE_DECLARATIONS_ERROR_KEY] == "why not"
    # a proxy nobody told carries neither key and no header fact
    bare, _, _ = _proxy(tmp_path, exact_replay_strategy=None, spec=_spec(serving_mode="interleaved"))
    b4 = bare.models_payload()["petri_proxy"]
    assert px.SERVE_DECLARATIONS_KEY not in b4 and px.SERVE_DECLARATIONS_ERROR_KEY not in b4
    assert px.THINKING_HEADER_KEY not in b4 and b4[px.LENS_SALT_POLICY_KEY] == "attested"
    # the digest map is a typed launch fact
    proxy5, _, _ = _thinking_proxy(
        tmp_path, object_artifact_sha256_by_cell={f"{LENS}@thinking": "ab" * 32})
    assert proxy5.models_payload()["petri_proxy"][px.OBJECT_SHA256_BY_CELL_KEY] == {
        f"{LENS}@thinking": "ab" * 32}
    with pytest.raises(px.ProxyError, match="sha256 mapping"):
        _thinking_proxy(tmp_path, object_artifact_sha256_by_cell={"": "x"})


class _TokBackend(StubBackend):
    async def tokenize(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        return {"count": 3, "tokens": [1, 2, 3], "max_model_len": 64}


def test_tokenize_applies_the_kwargs_shape_validation_only(tmp_path):
    """Q12: tokenize with kwargs and no declaration -> 200; a malformed shape
    -> thinking_kwargs_invalid; inert -> thinking_kwargs_inert; a
    petri_thinking_on key is outside the §2.5.6 matrix and refuses nothing
    new (it is not forwarded either)."""
    spec = _spec(serving_mode="interleaved", cells=list(CANONICAL_22) + [f"{LENS}@thinking"])
    source = _thinking_source(tmp_path)
    cells = _cells(spec, source)
    backend = _TokBackend()
    proxy = px.RoutingProxy(px.routes_from_cells(cells), backend,
                            base_model=spec.target_model, serving_mode="interleaved",
                            prompted_prefixes=px.prompted_prefix_specs(cells, source),
                            exact_replay_strategy=None, agent_loop_render=_TailRender(),
                            thinking_header=MOVES_TRUE, lens_serve_declarations=DECLARED)
    msgs = [{"role": "user", "content": "hi"}]
    out = asyncio.run(proxy.handle_tokenize({"model": THINKING_VM, "messages": msgs,
                                             px.CHAT_TEMPLATE_KWARGS_KEY: {"enable_thinking": True}}))
    assert out["count"] == 3 and backend.payloads[-1][px.CHAT_TEMPLATE_KWARGS_KEY] == {"enable_thinking": True}
    for bad in ("x", {"enable_thinking": 1}, {"foo": True}):
        with pytest.raises(px.ProxyRefusal) as ei:
            asyncio.run(proxy.handle_tokenize({"model": THINKING_VM, "messages": msgs,
                                               px.CHAT_TEMPLATE_KWARGS_KEY: bad}))
        assert ei.value.code == "thinking_kwargs_invalid"
    out = asyncio.run(proxy.handle_tokenize({"model": THINKING_VM, "messages": msgs,
                                             px.CHAT_TEMPLATE_KWARGS_KEY: {"enable_thinking": True},
                                             px.THINKING_ON_KEY: True}))
    assert out["count"] == 3 and px.THINKING_ON_KEY not in backend.payloads[-1]
    inert = px.RoutingProxy(px.routes_from_cells(cells), _TokBackend(),
                            base_model=spec.target_model, serving_mode="interleaved",
                            prompted_prefixes=px.prompted_prefix_specs(cells, source),
                            exact_replay_strategy=None, agent_loop_render=_TailRender(),
                            thinking_header=SWITCH_FALSE, lens_serve_declarations={})
    with pytest.raises(px.ProxyRefusal) as ei:
        asyncio.run(inert.handle_tokenize({"model": "petri__base", "messages": msgs,
                                           px.CHAT_TEMPLATE_KWARGS_KEY: {"enable_thinking": True}}))
    assert ei.value.code == "thinking_kwargs_inert"


# --- the routing table carries the fact and the route additions -------------------


def test_routing_table_round_trips_the_thinking_header_and_route_fields(tmp_path):
    spec = _spec(serving_mode="interleaved", cells=list(CANONICAL_22) + [f"{LENS}@thinking"])
    source = _thinking_source(tmp_path)
    cells = _cells(spec, source)
    path = tmp_path / "rt.json"
    px.export_routing_table(cells, source, path, base_model=spec.target_model,
                            serving_mode="interleaved", thinking_header=MOVES_TRUE)
    table = px.load_routing_table(path)
    assert table[px.THINKING_HEADER_KEY] == MOVES_TRUE
    # the per-ROUTE schema is UNCHANGED (a client loader builds these rows
    # from the JSON): the variant is spelled in cell_id / virtual_model only
    by = {r.cell_id: r for r in table["routes"]}
    assert {f.name for f in dataclasses.fields(px.RouteEntry)} == {
        "virtual_model", "cell_id", "cell_class", "serving_identity", "intervention_hash",
        "control_of", "si_prefix_hash",
        # native-psr-serving §1.8: the residual slot's native position mode — default None,
        # written into a ROW only when set (proxy.route_row), so a lens table's rows are unchanged
        "petri_position_mode"}
    assert all("petri_position_mode" not in row
               for row in json.loads(path.read_text())["routes"])
    assert by[f"{LENS}@thinking"].virtual_model == THINKING_VM
    assert set(table["routes"]) == set(px.routes_from_cells(cells))
    # the fact is validated on the way in and out
    with pytest.raises(px.ProxyError, match="moves"):
        px.export_routing_table(cells, source, tmp_path / "bad.json", base_model=spec.target_model,
                                serving_mode="interleaved", thinking_header={**MOVES_TRUE, "moves": False})
    # an OLD table (no fact) loads with None
    px.export_routing_table(cells, source, tmp_path / "old.json", base_model=spec.target_model,
                            serving_mode="interleaved")
    assert px.load_routing_table(tmp_path / "old.json")[px.THINKING_HEADER_KEY] is None
    # and the proxy built from a table wires the fact
    proxy = px.RoutingProxy(table["routes"], StubBackend(), base_model=spec.target_model,
                            serving_mode="interleaved", prompted_prefixes=table["prompted_prefixes"],
                            exact_replay_strategy=None, agent_loop_render=_TailRender(),
                            thinking_header=table[px.THINKING_HEADER_KEY],
                            lens_serve_declarations=DECLARED)
    assert proxy.models_payload()["petri_proxy"][px.THINKING_HEADER_KEY] == MOVES_TRUE


# --- X1: the ON certification may be UNAVAILABLE — OFF serves, ON refuses ---

#: the partial fact a standup records when the served template has a thinking
#: switch but its ON generation header cannot be derived (Nemotron-3: the ON
#: prompt ends in an OPEN ``<think>`` where the completed turn carries
#: ``<think></think>``): switch and off_ids kept, on_ids and moves null
UNAVAILABLE = {"switch": True, "moves": None, "off_ids": OFF7, "on_ids": None}
UNAVAILABLE_WHY = "ChatMarkerError: synthetic — the ON generation prompt is not a prefix"


def test_x1_unavailable_on_certification_serves_off_and_refuses_on_unattested(tmp_path):
    """A legacy thinking-OFF standup must not depend on the ON header being
    certifiable. With the partial fact + its reason the proxy STARTS, serves
    every OFF / unspecified render, accepts the base route's ON render (no
    mask there, §2.5.5), and refuses every declared ON request on a lens
    route ``thinking_on_unattested`` naming the reason — never
    ``cell_not_declared`` (that would say the CELL is the problem) and never
    the gemma-style 'validated but not injected' acceptance (moves is UNKNOWN
    here, not false)."""
    assert px.THINKING_HEADER_ERROR_KEY == "thinking_header_error"
    proxy, backend, _ = _thinking_proxy(tmp_path, header=UNAVAILABLE, declarations={},
                                        with_variant=False, thinking_header_error=UNAVAILABLE_WHY)
    # OFF / unspecified renders are served on every route
    for vm in ("petri__base", LENS_VM, TWIN_VM):
        out = _ask(proxy, _think_chat(vm, enable=False, declare=False))
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_off"
        out = _ask(proxy, _think_chat(vm, enable=None, declare=False))
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "unspecified"
    # the base route's ON render is accepted (the switch exists, so it is not inert)
    out = _ask(proxy, _think_chat("petri__base", declare=False))
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    served = len(backend.payloads)
    # a lens route's ON render: declared -> unattested (the reason in the message);
    # undeclared -> undeclared (the client's own omission comes first, as with an
    # old table); neither reaches the engine
    e = _refuses(proxy, _think_chat(LENS_VM), "thinking_on_unattested")
    assert UNAVAILABLE_WHY in str(e)
    _refuses(proxy, _think_chat(TWIN_VM), "thinking_on_unattested")
    _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    # the partial fact (moves None) is NOT the moves-false exemption of §2.5.6
    # row 8: an uncertified ON header cannot be certified to stay put, so a
    # declared seeded / agent-loop ON body refuses thinking_on_mismatch — the
    # code it always did — naming the missing certification
    for body in (_think_prefill(LENS_VM), _think_loop(LENS_VM), _think_prefill(TWIN_VM)):
        e = _refuses(proxy, body, "thinking_on_mismatch")
        assert "NOT certified" in str(e)
    assert len(backend.payloads) == served
    # the echo: the partial fact with the reason BESIDE it
    block = proxy.models_payload()["petri_proxy"]
    assert block[px.THINKING_HEADER_KEY] == UNAVAILABLE
    assert block[px.THINKING_HEADER_ERROR_KEY] == UNAVAILABLE_WHY
    # a fully derived fact carries the error key PRESENT and null (the
    # lens_serve_declarations_error pattern); a proxy nobody told carries neither
    full, _, _ = _thinking_proxy(tmp_path)
    fb = full.models_payload()["petri_proxy"]
    assert fb[px.THINKING_HEADER_KEY] == MOVES_TRUE and fb[px.THINKING_HEADER_ERROR_KEY] is None
    bare, _, _ = _proxy(tmp_path, exact_replay_strategy=None, spec=_spec(serving_mode="interleaved"))
    bb = bare.models_payload()["petri_proxy"]
    assert px.THINKING_HEADER_KEY not in bb and px.THINKING_HEADER_ERROR_KEY not in bb


# --- S-P1: the partial fact carries its reason — every entry, both directions ---

#: the frozen fact's EXACT key set (§2.5.12): a nullable field is PRESENT and
#: null, never absent — absence is a refusal, not a null
FACT_KEYS = ("switch", "moves", "off_ids", "on_ids")
#: ``_table_doc`` sentinel: delete the top-level key
DROP = object()


def _export(tmp_path, name, **kw):
    """``export_routing_table`` over the canonical cells (no variant)."""
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    path = tmp_path / name
    px.export_routing_table(cells, source, path, base_model=spec.target_model,
                            serving_mode="interleaved", **kw)
    return path, spec


def _table_doc(tmp_path, name, **over):
    """A complete-fact table as ``export_routing_table`` writes it, then
    ``over`` laid over its TOP-LEVEL JSON keys verbatim (``DROP`` deletes one)
    — the stale or version-skewed exporter the loader must refuse rather than
    normalise."""
    path, _ = _export(tmp_path, name, thinking_header=MOVES_TRUE)
    doc = json.loads(path.read_text())
    for key, value in over.items():
        if value is DROP:
            doc.pop(key, None)
        else:
            doc[key] = value
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path


def _from_table(table, spec, **kw):
    """The proxy the way ``main`` builds it: from the loaded table alone."""
    return px.RoutingProxy(table["routes"], StubBackend(), base_model=spec.target_model,
                           serving_mode="interleaved", prompted_prefixes=table["prompted_prefixes"],
                           exact_replay_strategy=None, agent_loop_render=_TailRender(),
                           thinking_header=table[px.THINKING_HEADER_KEY],
                           thinking_header_error=table[px.THINKING_HEADER_ERROR_KEY], **kw)


@pytest.mark.parametrize("fact", [MOVES_TRUE, MOVES_FALSE, SWITCH_FALSE, UNAVAILABLE],
                         ids=["moves_true", "moves_false", "switch_false", "unavailable"])
@pytest.mark.parametrize("missing", FACT_KEYS)
def test_sp1_a_fact_missing_a_key_is_refused_never_read_as_null(tmp_path, fact, missing):
    """``raw.get()`` read an ABSENT key as null, so a mapping lacking
    ``moves``/``on_ids`` normalised into the frozen partial shape and was
    served as a certification failure that never happened. The exact key set
    is required before any value is read — at the constructor, the export and
    a table on disk alike, whatever the fact would otherwise have been."""
    bad = {k: v for k, v in fact.items() if k != missing}
    reason = UNAVAILABLE_WHY if fact is UNAVAILABLE else None
    why = re.escape(f"missing key(s) ['{missing}']")
    with pytest.raises(px.ProxyError, match=why):
        _thinking_proxy(tmp_path, header=bad, declarations={}, with_variant=False,
                        thinking_header_error=reason)
    with pytest.raises(px.ProxyError, match=why):
        _export(tmp_path, "bad.json", thinking_header=bad, thinking_header_error=reason)
    with pytest.raises(px.ProxyError, match=why):
        px.load_routing_table(_table_doc(tmp_path, "skewed.json", thinking_header=bad,
                                         thinking_header_error=reason))


def test_sp1_the_reviewers_trigger_and_an_unknown_key_refuse_by_name(tmp_path):
    """The finding's exact trigger — a version-skewed exporter supplying
    ``switch`` + ``off_ids`` only — used to normalise into the partial shape
    with no reason and be served; it now names both absent keys. An unknown
    key still refuses by name, and a non-mapping by type."""
    with pytest.raises(px.ProxyError, match=re.escape("missing key(s) ['moves', 'on_ids']")):
        _thinking_proxy(tmp_path, header={"switch": True, "off_ids": OFF7},
                        declarations={}, with_variant=False)
    with pytest.raises(px.ProxyError, match=re.escape("unknown key(s) ['extra']")):
        _thinking_proxy(tmp_path, header={**MOVES_TRUE, "extra": 1})
    with pytest.raises(px.ProxyError, match="must be a mapping"):
        _thinking_proxy(tmp_path, header=[["switch", True]])


def test_sp1_a_partial_fact_needs_its_reason_and_a_reason_needs_a_partial_fact(tmp_path):
    """Both directions of the §2.5.12 pairing, at every entry. Before the fix
    only "reason => partial" held, so a partial fact with
    ``thinking_header_error: null`` started, served and was advertised (a test
    blessed it), and ``main`` rebuilt exactly that from a table carrying the
    fact without its reason."""
    no_reason = f"no {px.THINKING_HEADER_ERROR_KEY}"
    # partial => a non-empty reason: the constructor, the export, a table on
    # disk with the key ABSENT (a stale export) and with the key null
    with pytest.raises(px.ProxyError, match=no_reason):
        _thinking_proxy(tmp_path, header=UNAVAILABLE, declarations={}, with_variant=False)
    with pytest.raises(px.ProxyError, match=no_reason):
        _export(tmp_path, "noreason.json", thinking_header=UNAVAILABLE)
    for over in ({px.THINKING_HEADER_ERROR_KEY: DROP}, {px.THINKING_HEADER_ERROR_KEY: None}):
        with pytest.raises(px.ProxyError, match=no_reason):
            px.load_routing_table(_table_doc(tmp_path, "stale.json",
                                             thinking_header=UNAVAILABLE, **over))
    # an empty or non-string reason is no reason
    for empty in ("", 5, ["x"]):
        with pytest.raises(px.ProxyError, match="non-empty"):
            _thinking_proxy(tmp_path, header=UNAVAILABLE, declarations={}, with_variant=False,
                            thinking_header_error=empty)
        with pytest.raises(px.ProxyError, match="non-empty"):
            _export(tmp_path, "empty.json", thinking_header=UNAVAILABLE,
                    thinking_header_error=empty)
        with pytest.raises(px.ProxyError, match="non-empty"):
            px.load_routing_table(_table_doc(tmp_path, "empty2.json", thinking_header=UNAVAILABLE,
                                             thinking_header_error=empty))
    # complete / no-switch => None: a reason beside either is a contradiction
    for fact in (MOVES_TRUE, MOVES_FALSE, SWITCH_FALSE):
        with pytest.raises(px.ProxyError, match="contradiction"):
            _thinking_proxy(tmp_path, header=fact, thinking_header_error="x")
        with pytest.raises(px.ProxyError, match="contradiction"):
            _export(tmp_path, "contra.json", thinking_header=fact, thinking_header_error="x")
        with pytest.raises(px.ProxyError, match="contradiction"):
            px.load_routing_table(_table_doc(tmp_path, "contra2.json", thinking_header=fact,
                                             thinking_header_error="x"))
    # the shape stays closed: no on_ids means moves is UNKNOWN, never a bool
    for guessed in (False, True):
        with pytest.raises(px.ProxyError, match="moves"):
            _thinking_proxy(tmp_path, header={**UNAVAILABLE, "moves": guessed},
                            thinking_header_error=UNAVAILABLE_WHY)


def test_sp1_the_reason_rides_the_routing_table_into_the_standalone_proxy(tmp_path, monkeypatch):
    """The routing table carries ``thinking_header_error`` TOP-LEVEL beside
    ``thinking_header`` (additive; the ``lens_serve_declarations_error``
    pattern — present and null beside a complete fact, absent when no fact was
    recorded), and ``main`` builds the proxy from the table ALONE, so it never
    reconstructs a reason-less partial."""
    path, spec = _export(tmp_path, "partial.json", thinking_header=UNAVAILABLE,
                         thinking_header_error=UNAVAILABLE_WHY)
    doc = json.loads(path.read_text())
    assert doc[px.THINKING_HEADER_KEY] == UNAVAILABLE
    assert doc[px.THINKING_HEADER_ERROR_KEY] == UNAVAILABLE_WHY
    table = px.load_routing_table(path)
    assert table[px.THINKING_HEADER_KEY] == UNAVAILABLE
    assert table[px.THINKING_HEADER_ERROR_KEY] == UNAVAILABLE_WHY
    # built the way `main` builds it (+ an attested empty map so the header
    # gate is what refuses): starts, echoes both, and a declared ON request on
    # a lens route refuses unattested NAMING the reason
    proxy = _from_table(table, spec, lens_serve_declarations={})
    block = proxy.models_payload()["petri_proxy"]
    assert block[px.THINKING_HEADER_KEY] == UNAVAILABLE
    assert block[px.THINKING_HEADER_ERROR_KEY] == UNAVAILABLE_WHY
    assert UNAVAILABLE_WHY in str(_refuses(proxy, _think_chat(LENS_VM), "thinking_on_unattested"))
    # ... and the REAL `main` from the table path, only the app/server seams
    # stubbed: the reason is on the wire (no declarations map is handed over
    # there, so a declared ON refuses on THAT omission first — still closed)
    built = {}
    monkeypatch.setattr(px, "make_app", lambda proxy: built.setdefault("proxy", proxy))
    import aiohttp.web as web
    monkeypatch.setattr(web, "run_app", lambda app, *, host, port: None)
    px.main(["--routing-table", str(path), "--upstream", "http://127.0.0.1:1", "--port", "8814"])
    block = built["proxy"].models_payload()["petri_proxy"]
    assert block[px.THINKING_HEADER_KEY] == UNAVAILABLE
    assert block[px.THINKING_HEADER_ERROR_KEY] == UNAVAILABLE_WHY
    _refuses(built["proxy"], _think_chat(LENS_VM), "thinking_on_unattested")
    # a complete fact: the key PRESENT and null on disk, None loaded, None echoed
    path, spec = _export(tmp_path, "complete.json", thinking_header=MOVES_TRUE)
    doc = json.loads(path.read_text())
    assert px.THINKING_HEADER_ERROR_KEY in doc and doc[px.THINKING_HEADER_ERROR_KEY] is None
    table = px.load_routing_table(path)
    assert table[px.THINKING_HEADER_KEY] == MOVES_TRUE and table[px.THINKING_HEADER_ERROR_KEY] is None
    block = _from_table(table, spec).models_payload()["petri_proxy"]
    assert block[px.THINKING_HEADER_KEY] == MOVES_TRUE and block[px.THINKING_HEADER_ERROR_KEY] is None
    # an OLD table (no fact): neither key on disk, both None loaded, neither echoed
    path, spec = _export(tmp_path, "old.json")
    doc = json.loads(path.read_text())
    assert px.THINKING_HEADER_KEY not in doc and px.THINKING_HEADER_ERROR_KEY not in doc
    table = px.load_routing_table(path)
    assert table[px.THINKING_HEADER_KEY] is None and table[px.THINKING_HEADER_ERROR_KEY] is None
    block = _from_table(table, spec).models_payload()["petri_proxy"]
    assert px.THINKING_HEADER_KEY not in block and px.THINKING_HEADER_ERROR_KEY not in block


# --- X2: presence is MEMBERSHIP, never truthiness -------------------------


def _x2_body(vm, **fields):
    """A request with NO render kwargs and NO declaration, then ``fields``
    laid over it verbatim — so a present JSON null is sent as a present null."""
    body = _think_chat(vm, enable=None, declare=False)
    if vm == "petri__prompted__safety":
        body["messages"] = [{"role": "system", "content": SAFETY_PREFIX.decode()},
                            {"role": "user", "content": "hi"}]
    body.update(fields)
    return body


#: (vmodel, extra fields) -> the contract's code (§2.5.5 / §2.5.6), for every
#: row the review found accepted-then-stripped: a present ``null`` is a stated
#: (wrong) value, never an absent key; the base route forbids the FIELD
#: ``petri_thinking_on`` whatever its value
X2_ROWS = (
    # "chat_template_kwargs": null -> thinking_kwargs_invalid, any route
    *[(vm, {px.CHAT_TEMPLATE_KWARGS_KEY: None}, "thinking_kwargs_invalid")
      for vm in ("petri__base", "petri__prompted__safety", LENS_VM, THINKING_VM, TWIN_VM)],
    # "petri_thinking_on": null -> thinking_on_invalid, any route, whatever the render
    *[(vm, {px.THINKING_ON_KEY: None, **kw}, "thinking_on_invalid")
      for vm in ("petri__base", "petri__prompted__safety", LENS_VM, THINKING_VM, TWIN_VM)
      for kw in ({}, {px.CHAT_TEMPLATE_KWARGS_KEY: {px.ENABLE_THINKING_KEY: False}},
                 {px.CHAT_TEMPLATE_KWARGS_KEY: {px.ENABLE_THINKING_KEY: True}})],
    # "petri_thinking_on": false on a NON-lens route -> thinking_on_not_lens
    *[(vm, {px.THINKING_ON_KEY: False, **kw}, "thinking_on_not_lens")
      for vm in ("petri__base", "petri__prompted__safety")
      for kw in ({}, {px.CHAT_TEMPLATE_KWARGS_KEY: {px.ENABLE_THINKING_KEY: False}},
                 {px.CHAT_TEMPLATE_KWARGS_KEY: {px.ENABLE_THINKING_KEY: True}})],
)


def test_x2_present_null_and_false_declarations_refuse_by_the_contract(tmp_path):
    """The validators read ``request.get(key) is None`` as ABSENT and ran
    the non-lens prohibition only for a truthy declaration, so the base
    route accepted ``chat_template_kwargs: null``,
    ``petri_thinking_on: null`` and ``petri_thinking_on: false`` — then
    stripped the field, so the 200 hid the client's contract violation.
    Every row of X2_ROWS now refuses with the contract's exact code, before
    any engine call; tokenize shares the null-kwargs refusal."""
    proxy, backend, _ = _thinking_proxy(tmp_path)
    for vm, fields, code in X2_ROWS:
        _refuses(proxy, _x2_body(vm, **fields), code)
    assert backend.payloads == []
    # the lens route's `false` keeps the reading the matrix test pins: it is
    # NOT a declaration (an ON render beside it is undeclared) — the contract
    # has no row refusing the field's presence on a lens route
    _refuses(proxy, _x2_body(THINKING_VM, **{px.THINKING_ON_KEY: False,
                                            px.CHAT_TEMPLATE_KWARGS_KEY: {px.ENABLE_THINKING_KEY: True}}),
             "thinking_on_undeclared")
    # tokenize: a present null is not an absent key here either
    spec = _spec(serving_mode="interleaved", cells=list(CANONICAL_22) + [f"{LENS}@thinking"])
    source = _thinking_source(tmp_path)
    cells = _cells(spec, source)
    tok_backend = _TokBackend()
    tproxy = px.RoutingProxy(px.routes_from_cells(cells), tok_backend,
                             base_model=spec.target_model, serving_mode="interleaved",
                             prompted_prefixes=px.prompted_prefix_specs(cells, source),
                             exact_replay_strategy=None, agent_loop_render=_TailRender(),
                             thinking_header=MOVES_TRUE, lens_serve_declarations=DECLARED)
    msgs = [{"role": "user", "content": "hi"}]
    for vm in ("petri__base", THINKING_VM):
        with pytest.raises(px.ProxyRefusal) as ei:
            asyncio.run(tproxy.handle_tokenize({"model": vm, "messages": msgs,
                                                px.CHAT_TEMPLATE_KWARGS_KEY: None}))
        assert ei.value.http_status == 400 and ei.value.code == "thinking_kwargs_invalid"
    assert tok_backend.payloads == []
    # ... while an ABSENT key is still absent (unspecified regime, nothing forwarded)
    out = _ask(proxy, _x2_body("petri__base"))
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "unspecified"
    assert px.CHAT_TEMPLATE_KWARGS_KEY not in backend.payloads[-1]
    assert px.THINKING_ON_KEY not in backend.payloads[-1]


def test_x2_over_http_the_real_handler_returns_400_with_the_code_and_no_backend_call(tmp_path):
    """The same rows through the aiohttp app (``make_app``): a JSON body with
    a literal ``null`` on the wire, HTTP 400, ``error.code`` the contract's
    literal, and the backend never called."""
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer
    proxy, backend, _ = _thinking_proxy(tmp_path)
    tok_backend = _TokBackend()
    proxy._backend = tok_backend   # the same proxy answers /tokenize with a recording backend

    async def go():
        async with TestClient(TestServer(px.make_app(proxy))) as client:
            for vm, fields, code in X2_ROWS:
                raw = json.dumps(_x2_body(vm, **fields))
                if any(v is None for v in fields.values()):
                    assert '": null' in raw, raw  # the null IS on the wire
                resp = await client.post("/v1/chat/completions", data=raw,
                                         headers={"Content-Type": "application/json"})
                body = await resp.json()
                assert resp.status == 400 and body["error"]["code"] == code, (vm, fields, body)
            for vm in ("petri__base", THINKING_VM):
                resp = await client.post("/tokenize", data=json.dumps(
                    {"model": vm, "messages": [{"role": "user", "content": "hi"}],
                     px.CHAT_TEMPLATE_KWARGS_KEY: None}),
                    headers={"Content-Type": "application/json"})
                body = await resp.json()
                assert resp.status == 400 and body["error"]["code"] == "thinking_kwargs_invalid", body
    asyncio.run(go())
    assert backend.payloads == [] and tok_backend.payloads == []


# --- §2.5.4 NARROWED (2026-09-09): undeclared thinking-ON on a cell with no ---
# --- header-located op --------------------------------------------------------
#
# The declaration protects a HEADER-LOCATED mask: the worker locates a
# `response` / `response_all_turns` op by matching the cell's OFF assistant
# header at the prompt tail, and the ON render moves that header
# (REASON_NOT_AT_TAIL). On a cell served at `all` / `last` the mask is every row
# / the last row BY CONSTRUCTION — no header, no `thinking_on` fact is read — so
# the routing table's `lens_served_positions` stamp (the served mode of every
# lens leg whose resolved spec states one, stacked cells unwrapped to the object
# leg) lets the proxy ACCEPT an undeclared ON render there, and ONLY there. A
# header-located cell, a cell the map does not name, and a table with no map at
# all (exported before the landing) keep refusing `thinking_on_undeclared`. The
# DECLARED path is untouched: a declaration on such a cell still refuses as
# vacuous, exactly as the registry refuses it (attach_thinking_header rule 3).

STACKED = "stacked__safety__psr_kl_all"
STACKED_VM = f"petri__{STACKED}"
SERVED_KEY = px.LENS_SERVED_POSITIONS_KEY


def _positioned_source(tmp_path, positions, *, stacked_positions=None):
    """The canonical source with the psr lens cell's resolved spec STATING its
    served positions (``None`` = unstated, the fixture default: the served mode
    then defaults to the artifact's and the table records nothing for it), plus
    — when ``stacked_positions`` is given — a STACKED lens cell over the same
    artifact whose OBJECT-leg spec states that mode."""
    src = _source(tmp_path)
    entries = dict(src.entries)
    base = entries[LENS]
    spec = dict(base.resolved_spec)
    if positions is not None:
        spec[sd.POSITIONS_KEY] = positions
    entries[LENS] = dc.RegistryEntry(
        cell_id=LENS, kind=base.kind, si_id=base.si_id, object_path=base.object_path,
        serving_path="lens", resolved_spec=spec)
    if stacked_positions is not None:
        # a stacked cell's SOURCE entry carries the OBJECT leg's spec; the
        # `stacked` wrapper is derived at enumeration (stacked_resolved_spec)
        entries[STACKED] = dc.RegistryEntry(
            cell_id=STACKED, kind=base.kind, si_id="safety", object_path=base.object_path,
            serving_path="lens", si_prefix=SAFETY_PREFIX,
            resolved_spec={**dict(base.resolved_spec), sd.POSITIONS_KEY: stacked_positions})
    return dc.MappingCellSource(entries)


def _positions_proxy(tmp_path, served, *, header=MOVES_TRUE, declarations=None,
                     source=None, cells_extra=(), agent_loop_render=None, **kw):
    """A proxy over the canonical cells (no variant) handed ``served`` as its
    ``lens_served_positions`` map (``None`` = no map: an older table)."""
    spec = _spec(serving_mode="interleaved", cells=list(CANONICAL_22) + list(cells_extra))
    source = source or _source(tmp_path)
    cells = _cells(spec, source)
    backend = StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=None, agent_loop_render=agent_loop_render or _TailRender(),
        thinking_header=header, lens_serve_declarations=declarations,
        lens_served_positions=served, **kw)
    return proxy, backend


def _served_on(proxy, backend, vm, body=None):
    """Issue an UNDECLARED thinking-ON request on ``vm`` and assert it was
    SERVED the way any accepted ON render is: kwargs forwarded verbatim, no
    declaration key and no ``thinking_on`` selector fact upstream, the regime
    stamped on the response and the audit row."""
    n = len(backend.payloads)
    out = _ask(proxy, body or _think_chat(vm, declare=False))
    assert len(backend.payloads) == n + 1
    up = backend.payloads[-1]
    assert up[px.CHAT_TEMPLATE_KWARGS_KEY] == {"enable_thinking": True}
    assert px.THINKING_ON_KEY not in up
    assert "thinking_on" not in up[px.UPSTREAM_EXTRA_ARGS_KEY]
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    assert dict(proxy.request_log[-1])[px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    return out


def test_the_served_positions_interface_names_are_the_published_ones():
    """The routing-table key and the constructor kwarg share ONE name (the
    ``thinking_header`` convention), and the standup passes
    ``lens_served_positions=table.get("lens_served_positions")`` — a peer
    standup builds its passthrough against exactly these."""
    import inspect
    assert px.LENS_SERVED_POSITIONS_KEY == "lens_served_positions"
    assert "lens_served_positions" in inspect.signature(px.RoutingProxy.__init__).parameters
    assert px.POSITION_MODES == ("all", "last", "response", "response_all_turns")
    assert px.HEADER_LOCATED_MODES == ("response", "response_all_turns")


def test_undeclared_on_is_accepted_on_an_all_positions_cell(tmp_path):
    for header, err in ((MOVES_TRUE, None), (MOVES_FALSE, None),
                        (UNAVAILABLE, UNAVAILABLE_WHY), (None, None)):
        proxy, backend = _positions_proxy(tmp_path, {LENS: ["all"]}, header=header,
                                          declarations={}, thinking_header_error=err)
        _served_on(proxy, backend, LENS_VM)
    # a switch-less template is inert FIRST, whatever the mask (row 1 of the matrix)
    proxy, backend = _positions_proxy(tmp_path, {LENS: ["all"]}, header=SWITCH_FALSE)
    _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_kwargs_inert")
    assert backend.payloads == []


def test_undeclared_on_is_accepted_on_a_last_positions_cell(tmp_path):
    proxy, backend = _positions_proxy(tmp_path, {LENS: ["last"]}, declarations={})
    _served_on(proxy, backend, LENS_VM)
    # ... and a request that ALSO declares the OFF-shape facts is judged by
    # their own gates, not by this one: the seeded body under the ON kwargs on
    # a moves-TRUE template is row 8's mismatch only when DECLARED; undeclared
    # it reaches the prefill gate, which serves it (the mask needs no header)
    proxy, backend = _positions_proxy(tmp_path, {LENS: ["last"]}, declarations={},
                                      agent_loop_render=_LoopRender())
    out = _served_on(proxy, backend, LENS_VM, _think_prefill(LENS_VM, declare=False))
    assert backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]["response_prefill"] == 1
    out = _served_on(proxy, backend, LENS_VM, _think_loop(LENS_VM, declare=False))
    assert backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]["agent_loop"] == 1


def test_undeclared_on_still_refuses_on_a_header_located_cell(tmp_path):
    for mode in px.HEADER_LOCATED_MODES:
        proxy, backend = _positions_proxy(tmp_path, {LENS: [mode]}, declarations={})
        e = _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
        assert "HEADER-LOCATED" in str(e) and mode in str(e)
        assert backend.payloads == []
    assert px.THINKING_ON_UNDECLARED == "thinking_on_undeclared"  # the code is the original


def test_undeclared_on_still_refuses_when_the_map_is_absent_fail_closed(tmp_path):
    # (a) no map handed over at all — what every proxy built before this landing holds
    proxy, backend = _positions_proxy(tmp_path, None, declarations={})
    e = _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    assert "records no map" in str(e)
    # (b) a map that does not NAME the cell: unknown reads header-located
    proxy, backend = _positions_proxy(tmp_path, {}, declarations={})
    e = _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    assert "does not name it" in str(e)
    proxy, backend = _positions_proxy(tmp_path, {"some_other_cell": ["all"]}, declarations={})
    _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    assert backend.payloads == []


def test_an_old_routing_table_without_the_map_still_refuses(tmp_path):
    """A table exported BEFORE the landing has no ``lens_served_positions`` key:
    it loads as None, the proxy built from it (the way ``main`` builds one)
    refuses every undeclared ON render on a lens route exactly as before, and
    the identity block carries no key. The peer's runs cross this boundary
    while their branch merge is in flight."""
    old = px.load_routing_table(_table_doc(tmp_path, "old.json", **{SERVED_KEY: DROP}))
    assert old[SERVED_KEY] is None
    spec = _spec(serving_mode="interleaved")
    proxy = _from_table(old, spec, lens_served_positions=old[SERVED_KEY], lens_serve_declarations={})
    e = _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    assert "exported before the served-positions landing" in str(e)
    assert SERVED_KEY not in proxy.models_payload()["petri_proxy"]
    # ... and a table THIS exporter writes carries the key on every export —
    # empty here, since the fixture's psr spec states no served mode, which
    # is the stated "no lens leg states a mode", present and empty
    path, _ = _export(tmp_path, "new.json")
    doc = json.loads(path.read_text())
    assert SERVED_KEY in doc and doc[SERVED_KEY] == {}
    new = px.load_routing_table(path)
    assert new[SERVED_KEY] == {}
    proxy = _from_table(new, spec, lens_served_positions=new[SERVED_KEY], lens_serve_declarations={})
    e = _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    assert "does not name it" in str(e)
    assert proxy.models_payload()["petri_proxy"][SERVED_KEY] == {}


DECLARED_CASES = (
    # (header, header_error, attested declarations) -> the code a DECLARED ON
    # request on the undeclared psr cell refuses with, TODAY and after
    (MOVES_TRUE, None, {}, "thinking_on_cell_not_declared"),
    (MOVES_TRUE, None, None, "thinking_on_unattested"),
    (UNAVAILABLE, UNAVAILABLE_WHY, {}, "thinking_on_unattested"),
    (None, None, {}, "thinking_on_unattested"),
)


@pytest.mark.parametrize("header,err,decls,code", DECLARED_CASES,
                         ids=["moves_true", "no_attestation", "unavailable", "no_fact"])
def test_declared_on_on_an_all_positions_cell_still_refuses_as_vacuous(tmp_path, header, err,
                                                                          decls, code):
    """The DECLARED path is byte-identical in behaviour with and without the
    map: a declaration on a cell no op of which can express it is vacuous at
    the registry (attach_thinking_header rule 3) and keeps refusing here."""
    for served in ({LENS: ["all"]}, {LENS: ["last"]}, None):
        proxy, backend = _positions_proxy(tmp_path, served, header=header, declarations=decls,
                                          thinking_header_error=err)
        _refuses(proxy, _think_chat(LENS_VM), code)
        assert backend.payloads == []
    # on a moves-FALSE template a declaration is validated and served, not
    # injected (test_matrix_moves_false) — the map changes nothing there either
    for served in ({LENS: ["all"]}, None):
        proxy, backend = _positions_proxy(tmp_path, served, header=MOVES_FALSE, declarations={})
        out = _ask(proxy, _think_chat(LENS_VM))
        assert "thinking_on" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"


def test_a_mixed_op_set_is_header_located(tmp_path):
    """Header-locatedness is an OP property: ANY header-located mode in the
    cell's served set makes the cell header-located (fail closed on a mix)."""
    proxy, backend = _positions_proxy(tmp_path, {LENS: ["all", "response"]}, declarations={})
    _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    proxy, backend = _positions_proxy(tmp_path, {LENS: ["last", "response_all_turns"]},
                                      declarations={})
    _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    assert backend.payloads == []
    # two non-header-located modes are not a mix that matters
    proxy, backend = _positions_proxy(tmp_path, {LENS: ["all", "last"]}, declarations={})
    _served_on(proxy, backend, LENS_VM)


def test_off_requests_and_non_lens_routes_are_unaffected_by_the_map(tmp_path):
    proxy, backend = _positions_proxy(tmp_path, {LENS: ["all"]}, declarations={})
    # OFF / absent on the all-cell: accepted, regime recorded, nothing injected
    out = _ask(proxy, _think_chat(LENS_VM, enable=False, declare=False))
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_off"
    assert "thinking_on" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    out = _ask(proxy, _think_chat(LENS_VM, enable=None, declare=False))
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "unspecified"
    assert px.CHAT_TEMPLATE_KWARGS_KEY not in backend.payloads[-1]
    # the passthrough base and the prompted route: ON undeclared accepted as ever
    out = _ask(proxy, _think_chat("petri__base", declare=False))
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    assert backend.payloads[-1][px.CHAT_TEMPLATE_KWARGS_KEY] == {"enable_thinking": True}
    prompted = _think_chat("petri__prompted__safety", declare=False)
    prompted["messages"] = [{"role": "system", "content": SAFETY_PREFIX.decode()}] + prompted["messages"]
    out = _ask(proxy, prompted)
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    # ... and the FIELD is still forbidden there, whatever its value
    served = len(backend.payloads)
    _refuses(proxy, _think_chat("petri__base"), "thinking_on_not_lens")
    body = _think_chat("petri__base", declare=False); body[px.THINKING_ON_KEY] = False
    _refuses(proxy, body, "thinking_on_not_lens")
    # the none__lens0 twin has no spec of its own and is ABSENT from the map
    # (its served ops are its donor's, a choice lens_steer.noop_twin_donor
    # makes over registry entries the table exporter does not build), so an
    # undeclared ON render on it keeps refusing — fail closed, as today
    e = _refuses(proxy, _think_chat(TWIN_VM, declare=False), "thinking_on_undeclared")
    assert "does not name it" in str(e)
    assert len(backend.payloads) == served


def test_a_stacked_cell_is_classified_by_its_object_leg(tmp_path):
    """Stacked cells carry their served mode one level down
    (``resolved_spec["object"]["positions"]``); the ONE unwrapper
    (``defense_cells.served_positions_of``) reads it there and never off the
    wrapper — the stamp, the registry's marker gates and the worker's spec
    binding therefore agree about what the cell serves."""
    obj_all = {"kind": "obj_psr_kl_all", "si_id": "safety", "layer": 15, sd.POSITIONS_KEY: "all"}
    obj_resp = {**obj_all, sd.POSITIONS_KEY: "response"}
    wrap = dc.stacked_resolved_spec
    assert dc.served_positions_of(wrap("safety", SAFETY_PREFIX, obj_all)) == "all"
    assert dc.served_positions_of(wrap("safety", SAFETY_PREFIX, obj_resp)) == "response"
    # a stray top-level key on the wrapper is IGNORED: the object leg decides
    assert dc.served_positions_of({**wrap("safety", SAFETY_PREFIX, obj_resp),
                                   sd.POSITIONS_KEY: "all"}) == "response"
    assert dc.served_positions_of({"kind": "stacked"}) is None
    assert dc.served_positions_of({"kind": "stacked", "object": "junk"}) is None
    assert dc.served_positions_of(None) is None
    # the pre-wrap object-leg spec (what a SOURCE entry carries) reads the same
    assert dc.served_positions_of(obj_all) == "all" and dc.served_positions_of({}) is None
    # through the exporter: the stacked leg's stamp equals what its WRAPPED
    # registry spec unwraps to, and the proxy classifies it from the stamp
    for inner, accepted in (("all", True), ("response", False)):
        source = _positioned_source(tmp_path, "response", stacked_positions=inner)
        spec = _spec(serving_mode="interleaved", cells=list(CANONICAL_22) + [STACKED])
        cells = _cells(spec, source)
        assert not cells.absent, cells.absent
        path = tmp_path / f"stacked_{inner}.json"
        px.export_routing_table(cells, source, path, base_model=spec.target_model,
                                serving_mode="interleaved", thinking_header=MOVES_TRUE)
        table = px.load_routing_table(path)
        assert table[SERVED_KEY] == {LENS: ["response"], STACKED: [inner]}
        assert table[SERVED_KEY][STACKED][0] == dc.served_positions_of(
            wrap("safety", SAFETY_PREFIX, source.get(STACKED).resolved_spec))
        proxy = _from_table(table, spec, lens_served_positions=table[SERVED_KEY],
                            lens_serve_declarations={})
        backend = proxy._backend
        body = _think_chat(STACKED_VM, declare=False)
        body["messages"] = [{"role": "system", "content": SAFETY_PREFIX.decode()}] + body["messages"]
        if accepted:
            out = _ask(proxy, body)
            assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
            assert backend.payloads[-1][px.CHAT_TEMPLATE_KWARGS_KEY] == {"enable_thinking": True}
        else:
            e = _refuses(proxy, body, "thinking_on_undeclared")
            assert "HEADER-LOCATED" in str(e)
        # the response-served psr cell beside it refuses either way
        _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")


def test_the_stamp_rides_the_routing_table_into_the_proxy_and_the_identity_block(tmp_path):
    """export -> load -> RoutingProxy (as ``main`` / the standups build it) ->
    /v1/models: the served mode of every lens leg whose spec states one."""
    for mode, accepted in (("all", True), ("last", True), ("response", False),
                           ("response_all_turns", False)):
        source = _positioned_source(tmp_path, mode)
        spec = _spec(serving_mode="interleaved")
        cells = _cells(spec, source)
        path = tmp_path / f"rt_{mode}.json"
        px.export_routing_table(cells, source, path, base_model=spec.target_model,
                                serving_mode="interleaved", thinking_header=MOVES_TRUE)
        doc = json.loads(path.read_text())
        assert doc[SERVED_KEY] == {LENS: [mode]}          # lens legs only, stated modes only
        table = px.load_routing_table(path)
        assert table[SERVED_KEY] == {LENS: [mode]}
        proxy = _from_table(table, spec, lens_served_positions=table[SERVED_KEY],
                            lens_serve_declarations={})
        assert proxy.models_payload()["petri_proxy"][SERVED_KEY] == {LENS: [mode]}
        assert proxy.lens_served_positions == {LENS: [mode]}
        if accepted:
            _served_on(proxy, proxy._backend, LENS_VM)
        else:
            _refuses(proxy, _think_chat(LENS_VM, declare=False), "thinking_on_undeclared")
    # the helper alone, over the cell set: non-lens legs are never listed
    source = _positioned_source(tmp_path, "all")
    cells = _cells(_spec(serving_mode="interleaved"), source)
    stamp = px.lens_served_positions(cells, source)
    assert stamp == {LENS: ["all"]}
    lens_ids = {c.cell_id for c in cells.all_served()
                if cm.parse_serving_identity(c.serving_identity).path == "lens"}
    assert set(stamp) <= lens_ids and TWIN not in stamp   # the twin states no mode of its own


def test_the_map_is_validated_on_every_way_in(tmp_path):
    """Junk never reads as "not header-located": the exporter refuses a STATED
    mode outside the served vocabulary, and the loader / constructor refuse a
    malformed map, so the gate only ever decides from a fact it can classify."""
    bad_source = _positioned_source(tmp_path, "front")   # a hooks.py mode, not lens-served
    cells = _cells(_spec(serving_mode="interleaved"), bad_source)
    with pytest.raises(px.ProxyError, match="outside the served vocabulary"):
        px.export_routing_table(cells, bad_source, tmp_path / "bad.json",
                                base_model=BASE_MODEL, serving_mode="interleaved")
    for junk in ("x", {"": ["all"]}, {LENS: "all"}, {LENS: []}, {LENS: ["prompt"]},
                 {LENS: [1]}, {LENS: None}, [LENS]):
        with pytest.raises(px.ProxyError, match=SERVED_KEY):
            _positions_proxy(tmp_path, junk)
        with pytest.raises(px.ProxyError, match=SERVED_KEY):
            px.load_routing_table(_table_doc(tmp_path, "junk.json", **{SERVED_KEY: junk}))
    # a list value is admitted (an op property may carry several modes);
    # the validated copy is a fresh list per cell
    proxy, _ = _positions_proxy(tmp_path, {LENS: ("all",)})
    assert proxy.lens_served_positions == {LENS: ["all"]}
    assert px.load_routing_table(_table_doc(tmp_path, "ok.json",
                                            **{SERVED_KEY: {LENS: ["all", "last"]}}))[SERVED_KEY] \
        == {LENS: ["all", "last"]}


# --------------------------------------------------------------------------
# RunIdentity.engine
# --------------------------------------------------------------------------


def test_run_identity_engine_is_optional_and_excluded_from_run_id_when_none(tmp_path):
    """An identity without ``engine`` has the SAME ``run_id`` and
    ``geometry_id`` as a hand-computed PRE-FIELD dict (so no existing sidecar
    or row store is re-keyed); with ``engine="lora_native"`` the ``run_id``
    changes and the ``geometry_id`` does not."""
    from sisteer.capeval.serving import run_identity as ri

    pre_field = dict(model="org/model", revision="a" * 40, tree_sha="b" * 40, model_key="gemma4e4b",
                     max_model_len=4864, max_num_seqs=121, client_concurrency=121, budget=588_544,
                     budget_source="measured", serving_versions={"vllm": "0.25.1"},
                     sampling={"max_new_tokens": 128}, input_fingerprints={"goal_ids_sha16": "c" * 16})
    ident = ri.RunIdentity(**pre_field)
    assert ident.engine is None
    assert ident.run_id == ri._sha16(pre_field)          # the pre-field hash, exactly
    geometry = {k: pre_field[k] for k in ("model", "revision", "max_model_len", "max_num_seqs",
                                          "client_concurrency", "budget", "serving_versions",
                                          "sampling")}
    assert ident.geometry_id == ri._sha16(geometry)
    assert "engine" not in ident.as_dict() and "engine" not in ident.identity_fields()
    # a PRE-FIELD sidecar (no engine key at all) verifies against it cleanly
    assert ri.identity_diff(dict(pre_field, run_id=ident.run_id, geometry_id=ident.geometry_id),
                            ident.as_dict()) == []
    native = ri.RunIdentity(**pre_field, engine="lora_native")
    assert native.run_id != ident.run_id
    assert native.geometry_id == ident.geometry_id       # geometry_fields() is unchanged
    assert native.as_dict()["engine"] == "lora_native"
    assert ri._OPTIONAL_IDENTITY_FIELDS == ("engine",)
    # the sidecar refuses a resume across engines, naming the field
    ri.write_or_verify(tmp_path, ident)
    with pytest.raises(ri.RunIdentityMismatch, match="engine: recorded None != current 'lora_native'"):
        ri.write_or_verify(tmp_path, native)
    ri.write_or_verify(tmp_path, ident)                  # and the same identity resumes


def test_hop_timeouts_are_derived_and_ordered_client_after_upstream():
    """Timeouts compose as a MIN across hops; the 600 s transport default censored every
    cap-bound GLM draw at the 8192 rung (2026-09-10). Both hops default to the ONE derived
    constant, and the client hop is strictly longer so the proxy's 504 lands before the
    client's socket gives up (a socket timeout leaves vLLM generating for nobody)."""
    worst_case_s = 16384 / 9.0 * 2          # 16384-token cap at ~9 tok/s, plus one queued generation
    assert px.HOP_TIMEOUT_S >= worst_case_s
    assert px.HOP_TIMEOUT_S == 7200.0        # == served_capability_run.AIOHTTP_TOTAL_TIMEOUT_S (the controls driver)
    assert px.CLIENT_TIMEOUT_S > px.HOP_TIMEOUT_S
    assert px.HTTPBackend("http://127.0.0.1:1/v1").timeout_s == px.HOP_TIMEOUT_S
    eng = px.ProxyClientEngine("http://127.0.0.1:1", identity="proxy_vllm+lens+exact_replay")
    assert eng.timeout_s == px.CLIENT_TIMEOUT_S
    assert px.HTTPBackend("http://127.0.0.1:1/v1", timeout_s=30).timeout_s == 30.0   # still explicit-able


# --------------------------------------------------------------------------
# organism overlay: <vmodel>@<organism> = the same leg, served on top of a
# vLLM LoRA module loaded at launch (em-misalign, EM model organisms)
# --------------------------------------------------------------------------


def _overlay_proxy(tmp_path, organisms=("em-bad-medical-advice", "em-extreme-sports")):
    spec = _spec()
    source = _source(tmp_path)
    cells = _cells(spec, source)
    backend = StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode=spec.serving_mode,
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy="serialize", agent_loop_render=_TailRender(),
        organisms=organisms)
    return proxy, backend, cells


def test_organism_overlay_routes_the_lens_leg_onto_the_adapter(tmp_path):
    proxy, backend, cells = _overlay_proxy(tmp_path)
    cell = {c.cell_id: c for c in cells.canonical}["si_constitution__cast"]
    out = _ask(proxy, _chat("petri__si_constitution__cast@em-bad-medical-advice"))
    up = backend.payloads[-1]
    assert up["model"] == "em-bad-medical-advice"           # the adapter is the upstream model
    assert up[px.UPSTREAM_EXTRA_ARGS_KEY] == {"petri_object": "si_constitution__cast",
                                              "expect_hash": cell.intervention_hash}
    assert up["cache_salt"].startswith(f"petri|si_constitution__cast|{cell.intervention_hash}|org=em-bad-medical-advice|")
    assert out["model"] == "petri__si_constitution__cast@em-bad-medical-advice"
    assert out["petri"]["organism"] == "em-bad-medical-advice"
    assert out["petri"]["intervention_hash"] == cell.intervention_hash
    assert proxy.request_log[-1]["organism"] == "em-bad-medical-advice"
    # the bare leg is unchanged: base upstream, no organism stamp
    out0 = _ask(proxy, _chat("petri__si_constitution__cast"))
    assert backend.payloads[-1]["model"] == BASE_MODEL and "organism" not in out0["petri"]


def test_organism_overlay_on_passthrough_and_listing(tmp_path):
    proxy, backend, cells = _overlay_proxy(tmp_path)
    _ask(proxy, _chat("petri__base@em-extreme-sports"))
    up = backend.payloads[-1]
    assert up["model"] == "em-extreme-sports" and px.UPSTREAM_EXTRA_ARGS_KEY not in up
    assert up["cache_salt"] == "petri|none|" + {c.cell_id: c for c in cells.canonical}["none"].intervention_hash + "|org=em-extreme-sports"
    served = proxy.served_virtual_models()
    assert "petri__base@em-extreme-sports" in served and "petri__base@em-bad-medical-advice" in served
    assert "petri__si_constitution__cast@em-extreme-sports" in served
    assert all(v in served for v in proxy.overlay_virtual_models())
    ids = {m["id"] for m in proxy.models_payload()["data"]}
    assert "petri__si_constitution__cast@em-bad-medical-advice" in ids
    # no organisms declared -> no overlays anywhere
    plain, _, _ = _proxy(tmp_path)
    assert plain.overlay_virtual_models() == () and "@" not in "".join(plain.served_virtual_models())


def test_organism_overlay_refusals(tmp_path):
    proxy, backend, cells = _overlay_proxy(tmp_path)
    with pytest.raises(px.ProxyRefusal) as e:
        _ask(proxy, _chat("petri__si_constitution__cast@em-risky-financial-advice"))  # undeclared organism
    assert e.value.http_status == 404 and e.value.code == "unknown_virtual_model"
    with pytest.raises(px.ProxyRefusal) as e:
        _ask(proxy, _chat("petri__si_safety__lora_r1@em-bad-medical-advice"))          # a lora-path leg
    assert e.value.code == "organism_on_lora_leg"
    assert "petri__si_safety__lora_r1@em-bad-medical-advice" not in proxy.served_virtual_models()
    with pytest.raises(px.ProxyError, match="carries '@'"):
        _overlay_proxy(tmp_path, organisms=("bad@name",))


def test_routing_table_round_trips_organisms(tmp_path):
    spec = _spec()
    source = _source(tmp_path)
    cells = _cells(spec, source)
    path = tmp_path / "table.json"
    kw = dict(base_model=spec.target_model, serving_mode=spec.serving_mode,
              exact_replay_strategy="serialize" if spec.serving_mode == "exact_replay" else None)
    px.export_routing_table(cells, source, path, organisms=["em-extreme-sports", "em-bad-medical-advice"], **kw)
    table = px.load_routing_table(path)
    assert table[px.ORGANISMS_KEY] == ("em-bad-medical-advice", "em-extreme-sports")
    proxy = px.RoutingProxy(table["routes"], StubBackend(), base_model=table["base_model"],
                            serving_mode=table["serving_mode"], prompted_prefixes=table["prompted_prefixes"],
                            exact_replay_strategy=table["exact_replay_strategy"],
                            agent_loop_render=_TailRender(), organisms=table[px.ORGANISMS_KEY])
    assert "petri__base@em-extreme-sports" in proxy.served_virtual_models()
    # a table exported without organisms carries no key at all (byte-stable for every existing table)
    path0 = tmp_path / "table0.json"
    px.export_routing_table(cells, source, path0, **kw)
    assert px.ORGANISMS_KEY not in json.loads(path0.read_text())
    assert px.load_routing_table(path0)[px.ORGANISMS_KEY] == ()
