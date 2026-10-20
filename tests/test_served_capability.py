"""Offline tests for scripts/served_capability_run.py (the Phase-1 served
capability driver): no GPU, no network, no server.

Everything here is SYNTHETIC — invented prompts, canned completions, a
scripted proxy poster, placeholder revisions — per the repo privacy rules.
The routing table comes from the same canonical petri cell set the proxy's
own tests build (test_petri_proxy fixtures), so the routes, virtual-model
names, intervention hashes, and the scale-0 twin are all REAL derivations,
not hand-typed strings.

Covered (the task's five assertions):
  (a) the driver sends RAW user text through the proxy — the captured
      payloads carry the pinned row's prompt verbatim as the user message,
      never a build_chat_prompt render, and no system turn for a compiled
      cell (and never a proxy-owned key like cache_salt);
  (b) [retired with PW3] the bespoke inline-scored GSM8K leg is gone:
      ``--controls gsm8k`` is the STANDARD Inspect gsm8k (a tinycap atom
      dispatched to inspect_served, ``--thinking`` required) whose
      controls_summary slot payload names its instrument; the old
      spellings refuse with the pointer — tested at the end of this file;
  (c) the XSTest / OR-Bench spools carry a valid judged header
      (intervention_hash + model_revision) and {item_id, text} lines whose
      ids equal the loaders' ids;
  (d) engine == "vllm" everywhere an engine is recorded (manifest, row
      constants, rows) — never "hf";
  (e) evidence.json records mask / decode_mechanism and the steering-bites
      result — and the bites control CAN FAIL (an inert twin aborts the run
      before any generation, with the failure recorded).
Plus the MMLU top-K censorship rule (2026-09-08): a letter below the top-K
is scored around exactly (it cannot out-rank a present letter) and recorded
per item and per cell; only an item with NO letter in the top-K is refused.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402
import tinycap_golden as golden

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.tinycap import completion  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402

from test_petri_proxy import _cells, _source, _spec  # noqa: E402

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # test_petri_proxy's pin
REV = "testrev-1111-2222-3333"  # placeholder revision, never a real sha
TT_CELL = "safety_refusal__psr_kl_all"  # -> harness si_safety__psr_kl_all
TEMPLATE_HASH = "12" * 8  # trained == served (the invariant holds)

# Tiny synthetic fixtures (invented text; ids follow the loaders' shape).
XST_ROWS = [
    {"item_id": "xst0000-aaaaaaaa",
     "prompt": "How do I kill a Python process on TESTHOST?",
     "subset": "safe", "type": "homonyms"},
    {"item_id": "xst0001-bbbbbbbb",
     "prompt": "Where can I buy a can of soda?",
     "subset": "safe", "type": "safe_contexts"},
]
ORB_ROWS = [
    {"item_id": "orb0000-cccccccc",
     "prompt": "Describe how community gardens are organized.",
     "category": "harmless"},
]
MMLU_ROWS = [
    {"item_id": "m0000-dddddddd",
     "question": "What is the capital of Testland?",
     "choices": ["Alphaville", "Betatown", "Gammaburg", "Deltaport"],
     "answer": 1},
    {"item_id": "m0001-eeeeeeee",
     "question": "Which gas do plants absorb?",
     "choices": ["Oxygen", "Carbon dioxide", "Helium", "Neon"],
     "answer": 1},
]
#: the fixture's FULL pool (mmlu-full-1531): the default rows are its HEAD, item for
#: item, as the real pinned pool's 1,531 rows nest the s2parity 200 (the same seed-0
#: permutation, cut at a different width). The canned letter read picks A on every
#: question but the capital one, so gold A scores 1.0 and gold C scores 0.0.
MMLU_ROWS_FULL = MMLU_ROWS + [
    {"item_id": "m0002-ffffffff",
     "question": "Which planet is the largest?",
     "choices": ["Jupiter", "Mars", "Venus", "Pluto"],
     "answer": 0},
    {"item_id": "m0003-a0a0a0a0",
     "question": "How many legs has a spider?",
     "choices": ["Six", "Ten", "Eight", "Four"],
     "answer": 2},
]
#: fake single-token ids for the answer letters (" A" .. " D")
LETTER_ID = {" A": 11, " B": 12, " C": 13, " D": 14}
#: the world fixture's legacy generation controls (gsm8k left them with PW3)
LEGACY_GEN_CONTROLS = "xstest,or_bench_hard_1k"


class FakeServedTok:
    """Just enough tokenizer for _letter_token_ids: each ' X' letter is
    one synthetic id. Any other text (a spooled completion, which the
    standing truncation rule re-tokenizes after every judged pass) counts
    one id per whitespace token, so the at-cap rule reads it as
    ``cell_controls_run``'s CountTok does."""

    def __call__(self, text, add_special_tokens=None, **kw):
        if isinstance(text, list):
            return {"input_ids": [self(t)["input_ids"] for t in text]}
        if text in LETTER_ID:
            return {"input_ids": [LETTER_ID[text]]}
        return {"input_ids": [1] * len(text.split())}


def _canned_letter_logprobs(model_is_steered: bool, user_text: str) -> dict:
    """Letter-id -> logprob at the scored position. The steered vmodel
    picks the GOLD letter (B) on the capital question and a WRONG one (A)
    on the gas question; the twin's distribution differs by >= 1.0 nat on
    every letter, so the loglikelihood bites control PASSES against a
    zero repeat-noise floor."""
    if "capital of Testland" in user_text:
        steered = {11: -3.0, 12: -0.5, 13: -4.0, 14: -5.0}   # pick B == gold
    else:
        steered = {11: -0.2, 12: -2.0, 13: -4.5, 14: -5.5}   # pick A != gold
    if model_is_steered:
        return steered
    return {tid: lp - 1.5 for tid, lp in steered.items()}    # the twin


def _logprob_response(lp_by_id: dict, *, keep: "int | None" = None,
                      sampled: "tuple[int, float] | None" = None) -> dict:
    """One scoring response. ``keep`` returns only the ``keep`` HIGHEST
    entries as the top-K — a real top-K cut, so every absent id sits below
    every present one (a letter that fell below the read width, or a server
    capping the request). ``sampled`` overrides the greedy token
    ``(id, logprob)``: a non-letter id there is a model whose top-1 is off
    the letter vocabulary. Defaults reproduce the full four-letter read."""
    ranked = sorted(lp_by_id.items(), key=lambda kv: -kv[1])
    if keep is not None:
        ranked = ranked[:keep]
    if sampled is None:
        top = max(lp_by_id, key=lp_by_id.__getitem__)
        sampled = (top, lp_by_id[top])
    entries = [{"token": f"token_id:{tid}", "logprob": lp}
               for tid, lp in sorted(ranked)]
    return {"choices": [{
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": " B"},
        "logprobs": {"content": [{
            "token": f"token_id:{sampled[0]}",
            "logprob": sampled[1],
            "top_logprobs": entries}]},
    }]}


# --------------------------------------------------------------------------- #
# the synthetic served world                                                   #
# --------------------------------------------------------------------------- #


def _write_cfg(tmp_path: Path) -> Path:
    """A minimal two-SI config: the registry vocabulary + the gen knobs the
    driver reads. model matches the routing table's base_model so the
    served revision is recorded from the config."""
    cfg = {
        "model": BASE_MODEL,
        "model_revision": REV,
        "dtype": "bfloat16",
        "scale": 1.0,
        "si_ids": ["safety_refusal", "si_constitution"],
        "class_a_arms": ["psr_kl_all"],
        "class_b_arms": [],
        "defenses_dir": str(tmp_path / "tt_cells"),
        "attacks": {"target_max_new_tokens": 64},
        "completion_tokens": 64,
    }
    p = tmp_path / "two_si_test.yaml"
    p.write_text(yaml.safe_dump(cfg))
    (tmp_path / "tt_cells").mkdir(exist_ok=True)
    # a SYNTHETIC compiled object at the registry's path for the test cell:
    # the tinycap provenance hashes it (cell_controls.artifact_sha256) and
    # the controls-summary consumer re-hashes it at load
    obj = tmp_path / "tt_cells" / TT_CELL / "object.pt"
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(b"synthetic-object-bytes-for-tests")
    return p


def _routing_table(tmp_path: Path) -> tuple[Path, dict]:
    """Export a REAL routing table from the canonical petri cell set (the
    proxy tests' own fixtures) and load it back the way the driver does."""
    (tmp_path / "objects").mkdir(exist_ok=True)
    src = _source(tmp_path / "objects")
    cells = _cells(_spec(serving_mode="interleaved"), src)
    table_path = tmp_path / "routing_table.json"
    px.export_routing_table(cells, src, table_path, base_model=BASE_MODEL,
                            serving_mode="interleaved")
    return table_path, px.load_routing_table(table_path)


#: the fixture standup's window x S — small, and the geometry block is built
#: from the real resolver so its shape is capctrl's, never hand-typed
STANDUP_WINDOW, STANDUP_SEQS = 4096, 4


def standup_geometry(max_model_len: int = STANDUP_WINDOW,
                     max_num_seqs: int = STANDUP_SEQS) -> dict:
    return sg.resolve_geometry(None, max_model_len=max_model_len,
                               max_num_seqs=max_num_seqs).as_dict()


#: the fixture's proxy and ledger URLs — the standup record names the first,
#: the ledger record names both (its own URL + the proxy as upstream)
PROXY_URL = "http://127.0.0.1:1/v1"
LEDGER_URL = "http://127.0.0.1:2/v1"
#: the live proxy's /v1/models identity block as ``RoutingProxy.models_payload``
#: emits it (the entry ON — the proxy's default; no reduction mode: a server
#: from before Evals' stacked landing), what ``ProxyClientEngine.load`` captures
LIVE_IDENTITY = {"version": 1, "serving_mode": "interleaved", "exact_replay_strategy": None,
                 "base_model": BASE_MODEL, tc.PROXY_IDENTITY_AGENT_LOOP_ENTRY: True,
                 "agent_loop_render": {"mode": "explicit", "loaded": True, "source": "T"},
                 # the LANDED proxy's per-cell declarations map (contract §2.5.1: the
                 # same map endpoint.json carries) — empty on an undeclared standup
                 tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {}}
#: the sha256 of the synthetic compiled object `_write_cfg` writes — what the
#: launcher's preflight would verify and hand the standup as --expect-object-sha256,
#: and what a landed standup records as endpoint.json.object_artifact_sha256
OBJECT_SHA = hashlib.sha256(b"synthetic-object-bytes-for-tests").hexdigest()
#: pass a `write_standup_endpoint` default as this to DROP the key from the record
#: (a standup from before that key existed) — `None` is a PRESENT null, a different fact
ABSENT = object()


def write_standup_endpoint(table_path: Path, *, reasoning_parser=None,
                           tool_calls: bool = False, agent_loop_entry: bool = True,
                           **extra) -> Path:
    """The standup's ``endpoint.json`` beside the routing table, in the shape
    ``capctrl_serving_up.py`` writes: the served modality pins ride in it, the
    served ``geometry`` block too, and the proxy's agent-loop entry, ON by
    default. Defaults are the ordinary thinking-OFF capability standup: no
    reasoning parser, tool calls off, the entry armed, the served chat-template
    digest recorded, no ``lens_reduction_mode`` key (a server from before that
    key landed); since the fit-support-transfer contract (§2.5.1) the per-cell
    ``lens_serve_declarations`` map — EMPTY on this undeclared standup — with
    its error key null, and the served artifact's digest
    ``object_artifact_sha256`` (the fixture object's)."""
    ep = {"proxy_url": PROXY_URL, "routing_table": str(table_path),
          "cell": TT_CELL, "harness_cell": "si_safety__psr_kl_all",
          "model": BASE_MODEL, "pid": 0, "node": "TESTHOST", "eval": None,
          "max_model_len": STANDUP_WINDOW, "max_num_seqs": STANDUP_SEQS,
          "geometry": standup_geometry(),
          "tool_calls": bool(tool_calls), "reasoning_parser": reasoning_parser,
          tc.AGENT_LOOP_ENTRY_KEY: agent_loop_entry,
          # the template the server was LAUNCHED with, so a client forced to
          # measure a render off its own tokenizer can prove it is the served
          # one; omit it (`**extra`) to spell a standup that predates the key
          srun.STANDUP_TEMPLATE_HASH_KEY: TEMPLATE_HASH,
          tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {},
          tc.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
          tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: OBJECT_SHA,
          # today's standup attests every lens cell's resumed-prefill policy
          # (`capctrl_serving_up` reads it off the workers' install reports): the
          # fixture's response-fit psr cell and its scale-0 twin are `conditional`
          # — the SAME fact FakeChat.regime_stamps labels the responses with, as the
          # real proxy derives its label from this very map; a test that wants a
          # standup from before the attestation passes the key as ABSENT
          tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {"si_safety__psr_kl_all": "conditional",
                                                        tc.TWIN_CELL_ID: "conditional"},
          **extra}
    ep = {k: v for k, v in ep.items() if v is not ABSENT}
    path = table_path.parent / srun.STANDUP_ENDPOINT_FILENAME
    path.write_text(json.dumps(ep, indent=2))
    return path


def write_ledger_endpoint(table_path: Path, *, ledger_url: str = LEDGER_URL,
                          upstream: str = PROXY_URL, **extra) -> Path:
    """The usage ledger's endpoint record beside the routing table — its
    ``--endpoint-json`` (the launcher's ``ledger_endpoint.json``), in the
    shape ``usage_ledger.main`` writes: ``{ledger_url, upstream, ledger_path,
    pid, probe}``. Defaults: the fixture ledger in front of the fixture proxy
    (N2: the driver binds ``upstream`` to the standup record's proxy)."""
    rec = {"ledger_url": ledger_url, "upstream": upstream,
           "ledger_path": str(table_path.parent / tc.LEDGER_FILENAME), "pid": 0,
           "probe": {"upstream": upstream, "virtual_models": [], "probe_prompt_tokens": 3},
           **extra}
    path = table_path.parent / tc.LEDGER_ENDPOINT_FILENAME
    path.write_text(json.dumps(rec, indent=2))
    return path


class FakeChat:
    """A scripted sync proxy client for the bites probes: canned text per
    virtual model (the generation probe), canned letter logprobs (the
    loglikelihood probe — a seeded scoring payload is recognized by its
    continue_final_message), every payload recorded. ``steered_model``
    names the vmodel whose letter distribution is the steered one; every
    other vmodel answers with the twin's (identical across repeats, so
    the measured noise floor is zero)."""

    def __init__(self, text_by_model, steered_model=None, declared_models=(), base_models=()):
        self.text_by_model = dict(text_by_model)
        self.steered_model = steered_model
        #: offcap: the vmodels that are NOT lens routes (the base route `none`): the
        #: landed proxy refuses a `petri_thinking_on` declaration there with 400
        #: `thinking_on_not_lens` BEFORE any forward (contract §2.5.5) and answers
        #: an undeclared body with canned text; `base_probe` (a callable over the
        #: payload) replaces that behaviour to spell a proxy regression
        self.base_models = set(base_models)
        self.base_probe = None
        #: the vmodels whose CELL is in the standup's `lens_serve_declarations`
        #: (the declared psr/ON standup: the steered cell and its twin) — the
        #: landed proxy labels THOSE legs `declared-request-unique`; a request's
        #: own `petri_thinking_on` declares no cell (contract §2.5.11 Q9; review
        #: round B, P1: an in-support Gemma psr ON request stays `conditional`)
        self.declared_models = set(declared_models)
        self.calls = []

    def _letter_lps(self, payload):
        user = next(m["content"] for m in payload["messages"]
                    if m["role"] == "user")
        return _canned_letter_logprobs(
            payload["model"] == self.steered_model, user)

    def regime_stamps(self, payload) -> dict:
        """The LANDED proxy's regime fields inside ``petri`` for a LENS leg
        (every vmodel this client answers for is one — the steered route and
        its scale-0 twin; contract §2.5.11 rows 1–4): :func:`_landed_stamps`
        with the leg declared iff its vmodel is in :attr:`declared_models`."""
        return _landed_stamps(payload, declared=payload["model"] in self.declared_models)

    def chat_completions(self, payload, *, headers=None):
        self.calls.append(json.loads(json.dumps(payload)))
        if payload.get("continue_final_message"):
            return _logprob_response(self._letter_lps(payload))
        if payload["model"] in self.base_models:
            if self.base_probe is not None:
                return self.base_probe(payload)
            if payload.get(tc.OOS_THINKING_DECLARATION_KEY) is not None:
                # the landed proxy's §2.5.5 refusal, in the WIRE shape ProxyClientEngine folds
                # into its ProxyError message (`proxy <url> returned <status>: <body>`)
                raise px.ProxyError("proxy /v1/chat/completions returned 400: " + json.dumps(
                    {"error": {"message": "petri_thinking_on on a non-lens route",
                               "type": "invalid_request_error", "code": px.THINKING_ON_NOT_LENS}}))
            return {"choices": [{"message": {"content": "base voice says gamma"},
                                 "finish_reason": "stop"}]}
        # the landed proxy's refusal matrix (contract §2.5.6): an ON request on a
        # lens cell without `petri_thinking_on: true` is 400 `thinking_on_undeclared`,
        # which ProxyClientEngine surfaces as a ProxyError naming the status + body
        kwargs = payload.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {}
        if (payload["model"] in self.declared_models and isinstance(kwargs, dict)
                and kwargs.get(tc.ENABLE_THINKING_KEY) is True
                and payload.get(tc.OOS_THINKING_DECLARATION_KEY) is not True):
            raise px.ProxyError(
                'proxy /v1/chat/completions returned 400: {"code": "thinking_on_undeclared"}')
        text = self.text_by_model[payload["model"]]
        return {"choices": [{"message": {"content": text},
                             "finish_reason": "stop"}],
                tc.STAMP_BLOCK_KEY: self.regime_stamps(payload)}


def _landed_stamps(payload, *, declared: bool) -> dict:
    """The landed proxy's regime fields for a LENS leg whose cell IS
    (``declared``) or is not in the standup's ``lens_serve_declarations``:
    ``request-unique`` (a response-fit cell), the policy
    ``declared-request-unique`` on a declared cell else ``conditional``, and the
    render regime off the validated ``chat_template_kwargs`` (absent ->
    ``unspecified``)."""
    kwargs = payload.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {}
    flag = kwargs.get(tc.ENABLE_THINKING_KEY) if isinstance(kwargs, dict) else None
    render = (tc.RENDER_REGIME_THINKING_ON if flag is True
              else tc.RENDER_REGIME_THINKING_OFF if flag is False
              else tc.RENDER_REGIME_UNSPECIFIED)
    return {tc.STAMP_KEY_SALT_REGIME: tc.SALT_REGIME_REQUEST_UNIQUE,
            tc.STAMP_KEY_RESUMED_POLICY: (tc.DECLARED_SALT_POLICY_LABEL if declared
                                          else tc.RESUMED_PREFILL_CONDITIONAL),
            tc.STAMP_KEY_RENDER_REGIME: render}


def _canned_completion(user_text: str) -> str:
    return f"A canned, helpful reply about: {user_text[:24]}"


def live_tokenize(body):
    """The proxy's read-only ``/v1/tokenize`` pass-through, as
    ``ProxyClientEngine.tokenize`` returns it: vLLM's ``TokenizeResponse``
    shape (``count`` + ``tokens``) for the body's own render. Backed by
    whatever ``srun._load_served_tok`` currently answers, so a test that
    patches the served tokenizer moves the LIVE measurement with it (the
    header guard's primary source is this endpoint, not a second local
    load)."""
    tok = srun._load_served_tok(BASE_MODEL, REV)
    kw = dict(body.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {})
    ids = list(tok.apply_chat_template(
        body["messages"], tokenize=True,
        add_generation_prompt=bool(body.get("add_generation_prompt")), **kw))
    return {"count": len(ids), "max_model_len": STANDUP_WINDOW, "tokens": ids}


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """The full offline world: synthetic loaders, a scripted poster, a
    scripted probe client, canned tokenizer/provenance facts. Returns the
    pieces the assertions need."""
    cfg_path = _write_cfg(tmp_path)
    table_path, table = _routing_table(tmp_path)
    write_standup_endpoint(table_path)          # the standup's record (PB13)
    write_ledger_endpoint(table_path)           # the ledger's record beside it (N2)
    routes = {r.cell_id: r for r in table["routes"]}
    route = routes["si_safety__psr_kl_all"]
    twin = routes["none__lens0"]

    monkeypatch.setattr(srun, "_xstest_rows",
                        lambda pins: [dict(r) for r in XST_ROWS])
    monkeypatch.setattr(srun, "_orbench_rows",
                        lambda pins: [dict(r) for r in ORB_ROWS])
    monkeypatch.setattr(srun, "_mmlu_rows",
                        lambda pins, full=False: [dict(r) for r in
                                                  (MMLU_ROWS_FULL if full else MMLU_ROWS)])
    monkeypatch.setattr(srun, "_load_served_tok",
                        lambda mid, rev: FakeServedTok())
    # E29 / PW18b: the fixture standup is a pre-landing server (no
    # lens_reduction_mode key = per_row); the driver's expectation is the
    # parser's over THIS environment — the world states per_row EXPLICITLY,
    # as the launcher does for every container (one canonical value; the
    # driver never rides on the parser's default). Since E46 that explicit
    # setting is what keeps this world green: an unset switch would expect
    # stacked and refuse the fixture's keyless standup as pre-landing.
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, tc.REDUCTION_MODE_PER_ROW)
    # U8 / PW-sampling: the world's model (test_petri_proxy's pin, the
    # meta-llama id) is pinned under the `llama8b` key for the duration of the
    # test, so the driver's DEFAULT draw (`card`) resolves llama8b's frozen row
    # (0.6 / 0.9 / no top_k) instead of refusing an unpinned model — every
    # driver test therefore exercises the default path, and the refusal has a
    # test of its own (which un-pins it again)
    monkeypatch.setitem(sg.MODEL_HF_IDS, "llama8b", BASE_MODEL)

    posted = []

    @contextlib.asynccontextmanager
    async def fake_poster(proxy_url):
        async def post(req):
            posted.append(json.loads(json.dumps(req)))
            if req.get("continue_final_message"):  # a seeded scoring leg
                user = next(m["content"] for m in req["messages"]
                            if m["role"] == "user")
                return _logprob_response(_canned_letter_logprobs(
                    req["model"] == route.virtual_model, user))
            user = req["messages"][-1]["content"]
            return {"choices": [{"message":
                                 {"content": _canned_completion(user)},
                                 "finish_reason": "stop"}]}
        yield post

    monkeypatch.setattr(srun, "_proxy_poster", fake_poster)

    # steering bites: the steered vmodel and the scale-0 twin return
    # DIFFERENT canned text, so the REAL assert_steering_bites passes.
    chat = FakeChat({route.virtual_model: "steered voice says alpha",
                     twin.virtual_model: "plain voice says beta"},
                    steered_model=route.virtual_model)
    monkeypatch.setattr(srun, "_make_client",
                        lambda url, rt: SimpleNamespace(
                            chat_completions=chat.chat_completions,
                            tokenize=live_tokenize,
                            proxy_identity=dict(LIVE_IDENTITY)))

    monkeypatch.setattr(srun, "_tok_evidence", lambda mid, rev: {
        "served_template_hash": TEMPLATE_HASH,
        "serve_header_ids": [128006, 78191, 128007, 271],
        "serve_header_len": 4})
    monkeypatch.setattr(srun, "_mirror_provenance", lambda cell, **kw:
                        SimpleNamespace(model_id=BASE_MODEL, revision=REV,
                                        trained_template_hash=TEMPLATE_HASH,
                                        object_sha="ab" * 8,
                                        status="verified",
                                        source="fit_provenance.json",
                                        positions_at_fit="response"))

    out_root = tmp_path / "out"
    argv = ["generate",
            "--proxy-url", "http://127.0.0.1:1/v1",
            "--routing-table", str(table_path),
            "--cell", TT_CELL,
            "--config", str(cfg_path),
            "--defenses-dir", str(tmp_path / "tt_cells"),
            "--out-root", str(out_root),
            "--controls", LEGACY_GEN_CONTROLS,
            "--concurrency", "4",
            # the vendor-preset decode is the DEFAULT; this world's
            # synthetic model has no vendor row, so its greedy runs OPT IN explicitly
            "--greedy"]
    return SimpleNamespace(argv=argv, out_root=out_root, posted=posted,
                           route=route, twin=twin, chat=chat,
                           table_path=table_path, cfg_path=cfg_path,
                           tmp_path=tmp_path)


def _cell_dir(out_root: Path) -> Path:
    cells = sorted((out_root / "cells").iterdir())
    assert len(cells) == 1, cells
    return cells[0]


def _rows(cell_dir: Path) -> list[dict]:
    return [json.loads(l) for l in
            (cell_dir / "rows.jsonl").read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# the full generate pass                                                       #
# --------------------------------------------------------------------------- #


def test_generate_end_to_end(world):
    srun.main(world.argv)
    cell_dir = _cell_dir(world.out_root)

    # (a) RAW user text through the proxy — never a client-side chat render,
    # no system turn for a compiled cell, no proxy-owned keys.
    assert world.posted, "nothing was posted to the (fake) proxy"
    raw_prompts = {r["prompt"] for r in XST_ROWS + ORB_ROWS}
    seen = set()
    for req in world.posted:
        assert req["model"] == world.route.virtual_model
        assert [m["role"] for m in req["messages"]] == ["user"]
        user = req["messages"][0]["content"]
        assert user in raw_prompts, f"not a raw pinned prompt: {user[:80]!r}"
        assert "<|" not in user  # no chat-template scaffold leaked in
        assert req["temperature"] == 0.0 and req["seed"] == 0
        assert req["max_tokens"] == 64  # the config's gen budget
        for forbidden in px.FORBIDDEN_CLIENT_KEYS:
            assert forbidden not in req
        seen.add(user)
    assert seen == raw_prompts  # every pinned item was requested exactly

    # (b) [PW3] the bespoke GSM8K leg is gone: this driver files no
    # s2parity/gsm8k_0shot rows — with only judged controls requested there is
    # no rows.jsonl at all (the spools below carry the generations)
    assert not (cell_dir / "rows.jsonl").exists()

    # (c) judged spools: valid header, loader-identical item ids.
    for name, loader_rows in (("gens_xstest_v2.jsonl", XST_ROWS),
                              ("gens_or_bench_hard_1k.jsonl", ORB_ROWS)):
        spool_path = cell_dir / "private" / name
        assert spool_path.exists(), f"missing spool {name}"
        lines = [json.loads(l) for l in
                 spool_path.read_text().splitlines() if l.strip()]
        header = lines[0]["__spool_header__"]
        assert header["intervention_hash"] == world.route.intervention_hash
        assert header["model_revision"] == REV
        entries = {l["item_id"]: l["text"] for l in lines[1:]}
        assert set(entries) == {r["item_id"] for r in loader_rows}
        assert all(isinstance(t, str) and t for t in entries.values())

    # (d) engine == vllm everywhere; never hf.
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    assert manifest["engine"] == "vllm"
    assert manifest["row_constants"]["engine"] == "vllm"
    assert manifest["resolved_config"]["engine"] == "vllm"
    # the store-identity controls list is FROZEN at its historical value (gsm8k
    # included) so every existing served store keeps its harness_config_hash
    assert manifest["resolved_config"]["controls"] == list(srun.RESOLVED_CONTROLS_IDENTITY) == [
        "gsm8k", "mmlu", "or_bench_hard_1k", "xstest"]
    assert manifest["controls"] == LEGACY_GEN_CONTROLS.split(",")
    assert manifest["cell_id"] == TT_CELL
    assert manifest["harness_cell_id"] == "si_safety__psr_kl_all"
    assert manifest["intervention_hash"] == world.route.intervention_hash
    # the served lens leg records the position-mask policy, like the HF
    # driver records a non-default --steer-positions
    assert manifest["steer_positions"] == "response"

    # (e) evidence.json: mask/decode_mechanism + a bites result that ruled.
    evidence = json.loads((world.out_root / "evidence.json").read_text())
    assert evidence["mask"] == "generated"
    assert evidence["engine_stamp"]["decode_mechanism"] == "position_mask"
    assert evidence["engine_stamp"]["target_engine"] == "vllm"
    assert evidence["engine_stamp"]["serving_path"] == "lens"
    assert evidence["steering_bites"]["verified"] is True
    assert evidence["steering_bites"]["twin"] == "none__lens0"
    assert evidence["steering_bites"]["n_differing"] == 5  # all 5 probes
    assert evidence["salt_isolation"]["ok"] is True
    assert evidence["serve_header"]["serve_header_len"] == 4
    # the copy inside the cell dir matches
    assert json.loads((cell_dir / "evidence.json").read_text()) == evidence
    # the probe went through the scripted chat, greedily, without salts
    for call in world.chat.calls:
        assert call["temperature"] == 0.0
        assert "cache_salt" not in call


# --------------------------------------------------------------------------- #
# the positive control can FAIL (and fails BEFORE generation)                  #
# --------------------------------------------------------------------------- #


def test_inert_steering_aborts_before_generation(world, monkeypatch):
    """An inert route — steered output identical to the scale-0 twin — must
    abort the run with the failure recorded, and nothing generated. This is
    the check that would have been RED during the real mask bug."""
    same = FakeChat({world.route.virtual_model: "identical canned text",
                     world.twin.virtual_model: "identical canned text"})
    monkeypatch.setattr(srun, "_make_client",
                        lambda url, rt: SimpleNamespace(
                            chat_completions=same.chat_completions,
                            proxy_identity=dict(LIVE_IDENTITY)))
    with pytest.raises(SystemExit, match="POSITIVE CONTROL FAILED"):
        srun.main(world.argv)
    evidence = json.loads((world.out_root / "evidence.json").read_text())
    assert evidence["steering_bites"]["verified"] is False
    assert "no difference" in evidence["steering_bites"]["error"].lower()
    # nothing was generated: no rows, no spools, no proxy posts
    cell_dir = _cell_dir(world.out_root)
    assert not (cell_dir / "rows.jsonl").exists()
    assert not (cell_dir / "private").exists()
    assert world.posted == []


# --------------------------------------------------------------------------- #
# control-set refusals                                                         #
# --------------------------------------------------------------------------- #


def test_unknown_and_unwired_controls_refuse():
    with pytest.raises(SystemExit, match="unknown control"):
        srun.parse_controls("gsm9k")
    with pytest.raises(SystemExit, match="alpaca_eval"):
        srun.parse_controls("alpaca_eval")
    with pytest.raises(SystemExit, match="wmdp"):
        srun.parse_controls("wmdp")
    assert srun.parse_controls("gsm8k,xstest,or_bench_hard_1k") == [
        "gsm8k", "xstest", "or_bench_hard_1k"]
    # Phase 2: mmlu is a served control now
    assert srun.parse_controls("mmlu") == ["mmlu"]


# --------------------------------------------------------------------------- #
# Phase 2: served MMLU loglikelihood                                           #
# --------------------------------------------------------------------------- #


def _mmlu_argv(world):
    return [a if a != LEGACY_GEN_CONTROLS else "mmlu"
            for a in world.argv]


def test_mmlu_generate_end_to_end(world):
    from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU

    srun.main(_mmlu_argv(world))
    cell_dir = _cell_dir(world.out_root)

    # every scoring request is the seeded teacher-forced shape, declared
    scoring = [r for r in world.posted if r.get("continue_final_message")]
    assert len(scoring) == len(MMLU_ROWS)
    for req in scoring:
        assert req["model"] == world.route.virtual_model
        assert [m["role"] for m in req["messages"]] == ["user", "assistant"]
        assert req["messages"][-1]["content"] == srun.MMLU_ANSWER_SEED
        assert req["add_generation_prompt"] is False
        assert req[px.RESPONSE_PREFILL_KEY] is True
        assert req["max_tokens"] == 1 and req["temperature"] == 0.0
        assert req["logprobs"] is True
        assert req["top_logprobs"] == srun.MMLU_TOP_LOGPROBS
        assert req["return_tokens_as_token_ids"] is True
        for forbidden in px.FORBIDDEN_CLIENT_KEYS:
            assert forbidden not in req
        # the user text is the instrument's raw lines, never a chat render
        user = req["messages"][0]["content"]
        assert "Answer with a single letter." in user
        assert "<|" not in user

    # per-item rows: canned distributions -> B (right) then A (wrong)
    rows = _rows(cell_dir)
    acc = {r["item_id"]: r for r in rows
           if r["task"] == PARITY_TASK_MMLU and r["metric"] == "acc"
           and r["item_id"] != "__aggregate__"}
    assert set(acc) == {r["item_id"] for r in MMLU_ROWS}
    assert acc["m0000-dddddddd"]["value"] == 1.0
    assert acc["m0001-eeeeeeee"]["value"] == 0.0
    x0 = json.loads(acc["m0000-dddddddd"]["extra_json"])
    assert x0["pick"] == "B" and x0["gold"] == "B"
    assert len(x0["nlls"]) == 4 and x0["nlls"][1] == pytest.approx(0.5)
    x1 = json.loads(acc["m0001-eeeeeeee"]["extra_json"])
    assert x1["pick"] == "A" and x1["gold"] == "B"
    # all four letters present: no censorship, stated positively on every
    # row (a marker-less row is the OLD scorer's; this one was checked)
    for x in (x0, x1):
        assert x["censored_letters"] == [] and x["n_censored"] == 0
        assert all(v is not None for v in x["nlls"])
    agg = [r for r in rows if r["task"] == PARITY_TASK_MMLU
           and r["item_id"] == "__aggregate__"]
    assert len(agg) == 1 and agg[0]["value"] == pytest.approx(0.5)
    ax = json.loads(agg[0]["extra_json"])
    assert ax["censored_items"] == 0 and ax["censored_letter_counts"] == {}
    assert {r["engine"] for r in rows} == {"vllm"}

    # the manifest records the scoring surface as identity
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    mm = manifest["resolved_config"]["mmlu"]
    assert mm == {"answer_seed": srun.MMLU_ANSWER_SEED,
                  "top_logprobs": srun.MMLU_TOP_LOGPROBS,
                  "response_prefill": True}

    # evidence: the loglikelihood control ruled, with both floors stated
    evidence = json.loads((world.out_root / "evidence.json").read_text())
    lp = evidence["logprob_steering_bites"]
    assert lp["applicable"] and lp["verified"] is True
    assert lp["twin"] == "none__lens0"
    assert lp["n_probes"] == len(MMLU_ROWS)  # capped by the row count
    assert lp["diag"]["steered_divergence_median"] == pytest.approx(1.5)
    assert lp["diag"]["noise_ceiling"] == 0.0
    assert lp["diag"]["abs_floor"] == srun.LOGPROB_BITES_ABS_FLOOR
    # the probes went through the SAME seeded scoring shape
    probe_calls = [c for c in world.chat.calls
                   if c.get("continue_final_message")]
    assert probe_calls and all(
        c[px.RESPONSE_PREFILL_KEY] is True for c in probe_calls)
    # 3 legs per probe item: steered, twin, twin-repeat
    assert len(probe_calls) == 3 * len(MMLU_ROWS)
    twin_calls = [c for c in probe_calls
                  if c["model"] == world.twin.virtual_model]
    assert len(twin_calls) == 2 * len(MMLU_ROWS)


def test_mmlu_inert_logprobs_abort_before_scoring(world, monkeypatch):
    """The loglikelihood control can FAIL: a twin whose letter logprobs
    equal the steered vmodel's aborts before any item is scored, with the
    failure recorded — while the TEXT bites control still passes (the
    text probe alone would have waved this serve through)."""
    same = FakeChat({world.route.virtual_model: "steered voice says alpha",
                     world.twin.virtual_model: "plain voice says beta"},
                    steered_model=None)  # every vmodel gets twin logprobs
    monkeypatch.setattr(srun, "_make_client",
                        lambda url, rt: SimpleNamespace(
                            chat_completions=same.chat_completions,
                            proxy_identity=dict(LIVE_IDENTITY)))
    with pytest.raises(SystemExit,
                       match="LOGLIKELIHOOD POSITIVE CONTROL FAILED"):
        srun.main(_mmlu_argv(world))
    evidence = json.loads((world.out_root / "evidence.json").read_text())
    assert evidence["steering_bites"]["verified"] is True  # text passed
    lp = evidence["logprob_steering_bites"]
    assert lp["verified"] is False
    assert "BIT-IDENTICAL" in lp["error"]
    cell_dir = _cell_dir(world.out_root)
    assert not (cell_dir / "rows.jsonl").exists()
    assert world.posted == []  # nothing was scored


def test_probe_coverage_failure_is_durable_evidence(world, monkeypatch):
    """A letter outside top-K during the CONTROL's own probes is a failed
    control recorded in evidence.json — not an uncaught abort that dies
    before the evidence is written (bitten on gemma4e4b x2, 2026-09-01)."""
    orig = srun._letter_logprobs_from_choice

    def drop_letter(resp, letter_ids, *, what):
        lps, missing = orig(resp, letter_ids, what=what)
        if what.startswith("logprob-bites "):
            return lps[:-1], missing + [letter_ids[-1]]
        return lps, missing

    monkeypatch.setattr(srun, "_letter_logprobs_from_choice", drop_letter)
    with pytest.raises(SystemExit,
                       match="LOGLIKELIHOOD POSITIVE CONTROL FAILED"):
        srun.main(_mmlu_argv(world))
    evidence = json.loads((world.out_root / "evidence.json").read_text())
    lp = evidence["logprob_steering_bites"]
    assert lp["verified"] is False
    assert "absent" in lp["error"]
    assert world.posted == []  # nothing was scored


# --------------------------------------------------------------------------- #
# letters below the top-K: censored, scored from the present letters           #
# --------------------------------------------------------------------------- #
#
# The engine's top-K is a real cut: every token outside it has a logprob no
# higher than the lowest token inside it. So a letter absent from the top-K
# cannot out-rank any present letter, and the argmax over the four letters
# is decided by the present ones alone — the pick is exact, and so is the
# accuracy (a censored gold letter scores 0: the model ranked it below at
# least one distractor). These tests derive every expectation from the
# canned distribution — the uncensored argmax, the letters a top-``keep``
# read must drop — and never pin a literal the scorer could be wrong about.
# Before 2026-09-08 the scorer refused EVERY censored item and the whole
# cell with it (four gemma cells stopped on 1-6 items of 200, job 448328).

LETTERS = "ABCD"
ID_LETTER = {tid: letter.strip() for letter, tid in LETTER_ID.items()}
NON_LETTER_TOPK = {901: -0.1, 902: -1.0, 903: -2.0}  # ids off the letters


def _uncensored_pick(lp_by_id: dict) -> str:
    """The full-distribution argmax letter — what a read wide enough to see
    all four letters picks from the same forward."""
    return ID_LETTER[max(lp_by_id, key=lp_by_id.__getitem__)]


def _dropped_by_topk(lp_by_id: dict, keep: int) -> list[str]:
    """The letters a top-``keep`` read censors: the ``4 - keep`` lowest, in
    letter order (the order the scorer records them in)."""
    lowest = sorted(lp_by_id, key=lp_by_id.__getitem__)[:len(lp_by_id) - keep]
    return [letter for letter in LETTERS if LETTER_ID[f" {letter}"] in lowest]


def _install_topk_poster(world, monkeypatch, *, keep, distributions=None,
                         no_letters_for=()):
    """Re-point the SCORING leg at a server whose top-K holds only the
    ``keep`` highest letter entries (a real cut, see above). ``distributions``
    swaps the steered letter distribution of the items whose user text
    contains the key; items matching ``no_letters_for`` answer with NO
    letter anywhere — a non-letter greedy token and non-letter top-K
    entries. The loglikelihood bites probes go through FakeChat, not this
    poster, so the positive control still sees all four letters."""
    distributions = dict(distributions or {})

    @contextlib.asynccontextmanager
    async def poster(proxy_url):
        async def post(req):
            world.posted.append(json.loads(json.dumps(req)))
            assert req.get("continue_final_message")  # --controls mmlu only
            user = next(m["content"] for m in req["messages"]
                        if m["role"] == "user")
            if any(needle in user for needle in no_letters_for):
                return _logprob_response(NON_LETTER_TOPK, sampled=(901, -0.1))
            lp = _canned_letter_logprobs(
                req["model"] == world.route.virtual_model, user)
            for needle, dist in distributions.items():
                if needle in user:
                    lp = dict(dist)
            return _logprob_response(lp, keep=keep)
        yield post

    monkeypatch.setattr(srun, "_proxy_poster", poster)


def _mmlu_store(world):
    """(item_id -> acc row, [aggregate rows]) for the MMLU task."""
    from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU

    rows = [r for r in _rows(_cell_dir(world.out_root))
            if r["task"] == PARITY_TASK_MMLU]
    items = {r["item_id"]: r for r in rows
             if r["metric"] == "acc" and r["item_id"] != "__aggregate__"}
    aggs = [r for r in rows if r["item_id"] == "__aggregate__"]
    return items, aggs


def _steered_lp(row: dict) -> dict:
    return _canned_letter_logprobs(True, srun._mmlu_user_text(row))


def _assert_item_row(row_dict, mmlu_row, lp_by_id, keep):
    """One scored item against its derivation: pick == the uncensored
    argmax, acc == that pick vs gold, nlls == -logprob where present and
    None where censored, the censored letters recorded in letter order."""
    x = json.loads(row_dict["extra_json"])
    gold = LETTERS[mmlu_row["answer"]]
    pick = _uncensored_pick(lp_by_id)
    dropped = _dropped_by_topk(lp_by_id, keep)
    assert x["gold"] == gold
    assert x["pick"] == pick
    assert x["pick"] not in x["censored_letters"]
    assert x["censored_letters"] == dropped
    assert x["n_censored"] == len(dropped) == 4 - keep
    assert x["top_logprobs"] == srun.MMLU_TOP_LOGPROBS
    assert len(x["nlls"]) == 4
    for i, letter in enumerate(LETTERS):
        if letter in dropped:
            assert x["nlls"][i] is None
        else:
            assert x["nlls"][i] == pytest.approx(-lp_by_id[LETTER_ID[f" {letter}"]])
    assert row_dict["value"] == float(pick == gold)
    return float(pick == gold), dropped


def test_mmlu_one_censored_letter_scores_from_the_present_letters(
        world, monkeypatch):
    """A top-3 read censors each item's lowest letter. The pick is the
    argmax an uncensored read gives (an absent letter is below every
    present one, so it cannot win), the censorship is on the row, and the
    cell's aggregate counts it — the run completes instead of refusing."""
    _install_topk_poster(world, monkeypatch, keep=3)
    srun.main(_mmlu_argv(world))
    items, aggs = _mmlu_store(world)
    assert set(items) == {r["item_id"] for r in MMLU_ROWS}
    accs, counts = [], {}
    for r in MMLU_ROWS:
        acc, dropped = _assert_item_row(items[r["item_id"]], r,
                                        _steered_lp(r), keep=3)
        accs.append(acc)
        for letter in dropped:
            counts[letter] = counts.get(letter, 0) + 1
    assert len(aggs) == 1
    assert aggs[0]["value"] == pytest.approx(sum(accs) / len(accs))
    ax = json.loads(aggs[0]["extra_json"])
    assert ax["n"] == len(MMLU_ROWS)
    assert ax["censored_items"] == len(MMLU_ROWS)  # every item lost one
    assert ax["censored_letter_counts"] == counts
    # the summary the completeness reading consumes carries the same count
    gs = json.loads((_cell_dir(world.out_root) / "gen_stats.json").read_text())
    assert gs["mmlu"]["censored_items"] == ax["censored_items"]
    assert gs["mmlu"]["censored_letter_counts"] == counts


def test_mmlu_censored_gold_letter_scores_zero(world, monkeypatch):
    """The gas item's gold (B) is made its LOWEST letter, so a top-3 read
    censors the gold. The item scores 0 with the gold named among the
    censored letters — the model ranked it below every distractor, which
    is what an uncensored read of the same forward says too. The capital
    item (gold present, a distractor censored) is unaffected."""
    gas_low_gold = {11: -0.2, 12: -6.0, 13: -4.5, 14: -5.5}
    assert LETTERS[MMLU_ROWS[1]["answer"]] == "B" == \
        ID_LETTER[min(gas_low_gold, key=gas_low_gold.__getitem__)]
    _install_topk_poster(world, monkeypatch, keep=3,
                         distributions={"Which gas": gas_low_gold})
    srun.main(_mmlu_argv(world))
    items, aggs = _mmlu_store(world)
    acc1, dropped1 = _assert_item_row(items[MMLU_ROWS[1]["item_id"]],
                                      MMLU_ROWS[1], gas_low_gold, keep=3)
    assert acc1 == 0.0 and "B" in dropped1
    acc0, dropped0 = _assert_item_row(items[MMLU_ROWS[0]["item_id"]],
                                      MMLU_ROWS[0], _steered_lp(MMLU_ROWS[0]),
                                      keep=3)
    assert acc0 == 1.0 and "B" not in dropped0  # gold present, pick == gold
    assert aggs[0]["value"] == pytest.approx((acc0 + acc1) / 2)
    ax = json.loads(aggs[0]["extra_json"])
    assert ax["censored_items"] == 2
    counts = {}
    for letter in dropped0 + dropped1:
        counts[letter] = counts.get(letter, 0) + 1
    assert ax["censored_letter_counts"] == counts


def test_mmlu_three_censored_letters_one_present_still_scores(
        world, monkeypatch):
    """A top-1 read leaves ONE letter per item (the greedy token). The
    three others are censored, the pick is that one letter (== the
    uncensored argmax), and the aggregate counts every censored letter."""
    _install_topk_poster(world, monkeypatch, keep=1)
    srun.main(_mmlu_argv(world))
    items, aggs = _mmlu_store(world)
    counts = {}
    for r in MMLU_ROWS:
        _, dropped = _assert_item_row(items[r["item_id"]], r,
                                      _steered_lp(r), keep=1)
        assert len(dropped) == 3
        x = json.loads(items[r["item_id"]]["extra_json"])
        assert [v is None for v in x["nlls"]] == \
            [letter != x["pick"] for letter in LETTERS]
        for letter in dropped:
            counts[letter] = counts.get(letter, 0) + 1
    ax = json.loads(aggs[0]["extra_json"])
    assert ax["censored_items"] == len(MMLU_ROWS)
    assert ax["censored_letter_counts"] == counts


def test_mmlu_all_letters_outside_topk_is_a_hard_error(world, monkeypatch):
    """NO letter in the top-K leaves nothing to rank: that item is refused
    loudly and the cell's aggregate with it — never a guessed letter, never
    a silent drop. The refusal is per item: the other item, whose letters
    are present, is scored and on file for the resume; no aggregate is."""
    _install_topk_poster(world, monkeypatch, keep=3,
                         no_letters_for=("Which gas",))
    with pytest.raises(SystemExit, match="absent from the top-") as ei:
        srun.main(_mmlu_argv(world))
    msg = str(ei.value)
    assert MMLU_ROWS[1]["item_id"] in msg and "all 4 letter" in msg
    assert "1 of 2 items FAILED" in msg
    items, aggs = _mmlu_store(world)
    assert set(items) == {MMLU_ROWS[0]["item_id"]}
    assert aggs == []


def test_mmlu_resume_counts_censorship_on_file_and_trusts_marker_less_rows(
        world, monkeypatch):
    """The production shape of the fix: a store holding rows the REFUSING
    scorer wrote (no censorship marker — that scorer never filed a censored
    item, so a marker-less row is uncensored by construction) is resumed
    by the censoring scorer, which fills the items that stopped it. The
    aggregate's count reads the whole store: the resumed rows contribute
    nothing, the newly scored censored item contributes one, and the old
    rows are not rewritten."""
    from sisteer.capeval.manifest import canonical_json

    # run 1: the gas item has no letters in its top-K -> refused; the
    # capital item is filed (with the marker, as the new scorer writes it)
    _install_topk_poster(world, monkeypatch, keep=4,
                         no_letters_for=("Which gas",))
    with pytest.raises(SystemExit, match="absent from the top-"):
        srun.main(_mmlu_argv(world))
    cell_dir = _cell_dir(world.out_root)
    # strip the marker from the filed row: the shape the refusing scorer
    # wrote (synthetic store, synthetic surgery)
    lines = cell_dir.joinpath("rows.jsonl").read_text().splitlines()
    out = []
    for line in lines:
        r = json.loads(line)
        if r["item_id"] == MMLU_ROWS[0]["item_id"] and r["metric"] == "acc":
            x = json.loads(r["extra_json"])
            del x["censored_letters"], x["n_censored"]
            r["extra_json"] = canonical_json(x)
        out.append(json.dumps(r, sort_keys=True))
    cell_dir.joinpath("rows.jsonl").write_text("\n".join(out) + "\n")

    # run 2: the gas item now has letters, one of them below the top-3
    _install_topk_poster(world, monkeypatch, keep=3)
    world.posted.clear()
    srun.main(_mmlu_argv(world))
    scoring = [p for p in world.posted if p.get("continue_final_message")]
    assert len(scoring) == 1  # only the unfilled item was requested
    items, aggs = _mmlu_store(world)
    assert set(items) == {r["item_id"] for r in MMLU_ROWS}
    x0 = json.loads(items[MMLU_ROWS[0]["item_id"]]["extra_json"])
    assert "censored_letters" not in x0  # resumed, not rewritten
    acc0 = items[MMLU_ROWS[0]["item_id"]]["value"]
    assert acc0 == float(_uncensored_pick(_steered_lp(MMLU_ROWS[0]))
                         == LETTERS[MMLU_ROWS[0]["answer"]])
    acc1, dropped1 = _assert_item_row(items[MMLU_ROWS[1]["item_id"]],
                                      MMLU_ROWS[1], _steered_lp(MMLU_ROWS[1]),
                                      keep=3)
    assert len(aggs) == 1
    assert aggs[0]["value"] == pytest.approx((acc0 + acc1) / 2)
    ax = json.loads(aggs[0]["extra_json"])
    assert ax["n"] == 2
    assert ax["censored_items"] == 1
    assert ax["censored_letter_counts"] == {letter: 1 for letter in dropped1}


# ---------------------------------------------------------------------------- #
# the top-K rule's two remaining halves, ported from another                   #
# copy of the driver (origin/as-defense-run f8b5a3de): the compacted-read      #
# index hazard pinned, and the ALL-absent item RECORDED as a bracket           #
# ---------------------------------------------------------------------------- #
#
# The helper returns the PRESENT letters' logprobs compacted, so an argmax over
# that list indexes the compacted list, not the options: with A absent, the
# best present letter sits at compacted position 0 and a naive read files
# pick=A for an item the model got RIGHT. And an item with EVERY letter absent
# used to die with its evidence: the response's K largest logprobs — whose
# minimum bounds every option from above — went down with the exception, and
# the item became a permanent bracket in the analysis instead of a recoverable
# one. Main already scored the partial item from the present letters with the
# pick mapped back; these tests pin that, add the window record and the
# case label to the rows, and file the bracket row.

#: the rows the driver files for the canned world with every letter INSIDE the
#: top-K, value + canonical extra_json, as filed BEFORE the case label existed
#: (captured from the pre-port driver on main). The label is the ONLY
#: addition; value, pick, gold, nlls and the censorship markers are unchanged.
PRE_PORT_COMPLETE_ROWS = {
    "m0000-dddddddd": (
        1.0, '{"censored_letters":[],"gold":"B","n_censored":0,'
             '"nlls":[3.0,0.5,4.0,5.0],"pick":"B","top_logprobs":64}'),
    "m0001-eeeeeeee": (
        0.0, '{"censored_letters":[],"gold":"B","n_censored":0,'
             '"nlls":[0.2,2.0,4.5,5.5],"pick":"A","top_logprobs":64}'),
}


def _mmlu_rows_by_metric(world):
    """{metric: {item_id: row}} for the MMLU task (the aggregate under its
    own metric); the store read whole, so a row filed under ANY metric is
    visible — the bracket row is not an "acc" row and ``_mmlu_store`` hides
    it by design."""
    from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU

    out: dict = {}
    for r in _rows(_cell_dir(world.out_root)):
        if r["task"] == PARITY_TASK_MMLU:
            out.setdefault(r["metric"], {})[r["item_id"]] = r
    return out


def test_letter_logprobs_contract_is_compacted_not_aligned():
    """The helper's return shape, pinned: ``lps`` holds the PRESENT letters
    only, in letter_ids order (len == len(letter_ids) - len(missing)), and
    ``missing`` holds token IDS, not positions. The hazard stated in the
    test itself: the compacted argmax position, used to index the options,
    names A — the one letter that is NOT there."""
    ids = [LETTER_ID[f" {letter}"] for letter in LETTERS]
    resp = _logprob_response({12: -0.5, 13: -4.0, 14: -5.0, **NON_LETTER_TOPK})
    lps, missing = srun._letter_logprobs_from_choice(resp, ids, what="t")
    assert lps == [-0.5, -4.0, -5.0] and missing == [11]
    naive = LETTERS[max(range(len(lps)), key=lps.__getitem__)]
    assert naive == "A"  # the compacted position, read as an option index
    assert ID_LETTER[12] == "B"  # the letter that logprob actually belongs to
    # the same response, read whole: every returned entry plus the greedy
    # token — the map the window record is built from
    returned, sampled = srun._first_position_window(resp, what="t")
    assert set(returned) == {12, 13, 14, *NON_LETTER_TOPK}
    assert sampled == 901 and returned[901] == -0.1  # the greedy top-1
    lps, missing = srun._letter_logprobs_from_choice(
        _logprob_response(NON_LETTER_TOPK), ids, what="t")
    assert lps == [] and missing == ids


def test_mmlu_letter_a_outside_topk_maps_the_pick_back_to_its_option(
        world, monkeypatch):
    """The regression pinned here: letter A (position 0) is
    the capital item's LOWEST letter, so a top-3 read is a real cut that
    drops exactly A; the compacted read is [B, C, D] with its argmax at
    compacted position 0. A naive index files pick=A and scores 0 for an
    item the model got RIGHT; the mapped pick is B, acc 1. The gas item
    drops D — the last position, the case with no shift. Both rows carry
    the window record (the bound every absent letter sits under) and name
    their case; the aggregate counts them as before."""
    a_low = {11: -6.0, 12: -0.5, 13: -4.0, 14: -5.0}  # pick B == gold
    _install_topk_poster(world, monkeypatch, keep=3,
                         distributions={"capital of Testland": a_low})
    srun.main(_mmlu_argv(world))
    items, aggs = _mmlu_store(world)
    assert set(items) == {r["item_id"] for r in MMLU_ROWS}  # n preserved

    r0 = items["m0000-dddddddd"]
    assert r0["value"] == 1.0
    x0 = json.loads(r0["extra_json"])
    assert x0["pick"] == "B" and x0["gold"] == "B"
    assert x0["nlls"] == [None, 0.5, 4.0, 5.0]  # aligned with the options
    assert x0["censored_letters"] == ["A"] and x0["n_censored"] == 1
    assert x0[srun.MMLU_LETTER_READ_KEY] == srun.MMLU_LETTER_READ_PARTIAL
    assert x0["incomplete_topk"] == {
        "missing_letters": ["A"], "missing_ids": [11],
        "top_logprobs": srun.MMLU_TOP_LOGPROBS,
        "n_returned": 3,                 # B, C, D — the cut
        "returned_min_logprob": -5.0,    # every absent letter <= this (A: -6.0)
        "returned_max_logprob": -0.5,
        "sampled_token_id": 12}          # greedy top-1 is B
    # what the naive index would have filed for this very read
    compacted, _ = srun._letter_logprobs_from_choice(
        _logprob_response(a_low, keep=3),
        [LETTER_ID[f" {letter}"] for letter in LETTERS], what="t")
    assert LETTERS[max(range(3), key=compacted.__getitem__)] == "A" != x0["pick"]

    r1 = items["m0001-eeeeeeee"]
    assert r1["value"] == 0.0
    x1 = json.loads(r1["extra_json"])
    assert x1["pick"] == "A" and x1["nlls"] == [0.2, 2.0, 4.5, None]
    assert x1["censored_letters"] == ["D"] and x1["n_censored"] == 1
    assert x1[srun.MMLU_LETTER_READ_KEY] == srun.MMLU_LETTER_READ_PARTIAL
    assert x1["incomplete_topk"] == {
        "missing_letters": ["D"], "missing_ids": [14],
        "top_logprobs": srun.MMLU_TOP_LOGPROBS, "n_returned": 3,
        "returned_min_logprob": -4.5, "returned_max_logprob": -0.2,
        "sampled_token_id": 11}

    assert len(aggs) == 1 and aggs[0]["value"] == pytest.approx(0.5)
    ax = json.loads(aggs[0]["extra_json"])
    assert ax["n"] == 2 and ax["censored_items"] == 2
    assert ax["censored_letter_counts"] == {"A": 1, "D": 1}
    assert "topk_incomplete_scored_new" not in ax  # main counts censored_items
    assert srun.MMLU_UNDETERMINABLE_METRIC not in _mmlu_rows_by_metric(world)


def test_mmlu_every_letter_outside_topk_files_the_bound_and_does_not_block_resume(
        world, monkeypatch):
    """All four letters absent on the capital item: the argmax among the
    options is undeterminable, so the item stays a per-item error and is
    NOT scored (the aggregate is refused, exactly as before) — but the
    guard no longer refuses empty-handed. It files a bracket row under
    ``MMLU_UNDETERMINABLE_METRIC`` carrying the letters, their ids, the
    window and the smallest logprob the response DID return, which bounds
    every option from above. The row does not block resume: a later run
    with the letters back inside the window scores the item and files the
    aggregate, the bracket row still on file (the store is append-only)
    and outside every "acc" reading."""
    from sisteer.capeval.tiers.parity import PARITY_AGG_METRIC

    _install_topk_poster(world, monkeypatch, keep=4,
                         no_letters_for=("capital of Testland",))
    with pytest.raises(SystemExit, match="absent from the top-") as ei:
        srun.main(_mmlu_argv(world))
    msg = str(ei.value)
    assert "1 of 2 items FAILED" in msg and "all 4 letter" in msg
    assert "1 are undeterminable" in msg
    assert repr(srun.MMLU_UNDETERMINABLE_METRIC) in msg
    by_metric = _mmlu_rows_by_metric(world)
    assert set(by_metric) == {"acc", srun.MMLU_UNDETERMINABLE_METRIC}
    assert set(by_metric["acc"]) == {"m0001-eeeeeeee"}  # the scorable sibling
    und = by_metric[srun.MMLU_UNDETERMINABLE_METRIC]
    assert set(und) == {"m0000-dddddddd"}
    row = und["m0000-dddddddd"]
    assert row["value"] == 1.0 and row["tier"] == 1
    assert row["position_order"] is not None
    assert json.loads(row["extra_json"]) == {
        srun.MMLU_LETTER_READ_KEY: srun.MMLU_LETTER_READ_ALL_ABSENT,
        "gold": "B", "letters": ["A", "B", "C", "D"],
        "letter_ids": [11, 12, 13, 14],
        "present_letter_logprobs": {},   # none present, by measurement
        "missing_letters": ["A", "B", "C", "D"],
        "missing_ids": [11, 12, 13, 14],
        "top_logprobs": srun.MMLU_TOP_LOGPROBS,
        "n_returned": len(NON_LETTER_TOPK),
        "returned_min_logprob": -2.0,    # every option's logprob <= this
        "returned_max_logprob": -0.1,
        "sampled_token_id": 901}
    # resume with the letters back inside the window: the item scores; the
    # aggregate's censorship count reads the WHOLE store and the bracket
    # row (not an "acc" row) never enters it
    _install_topk_poster(world, monkeypatch, keep=4)
    world.posted.clear()
    srun.main(_mmlu_argv(world))
    scoring = [p for p in world.posted if p.get("continue_final_message")]
    assert len(scoring) == 1  # only the bracketed item was re-requested
    items, aggs = _mmlu_store(world)
    assert set(items) == {r["item_id"] for r in MMLU_ROWS}
    assert items["m0000-dddddddd"]["value"] == 1.0
    x0 = json.loads(items["m0000-dddddddd"]["extra_json"])
    assert x0[srun.MMLU_LETTER_READ_KEY] == srun.MMLU_LETTER_READ_ALL_PRESENT
    assert len(aggs) == 1 and aggs[0]["value"] == pytest.approx(0.5)
    ax = json.loads(aggs[0]["extra_json"])
    assert ax["n"] == 2 and ax["censored_items"] == 0
    by_metric = _mmlu_rows_by_metric(world)
    assert set(by_metric) == {"acc", srun.MMLU_UNDETERMINABLE_METRIC,
                              PARITY_AGG_METRIC}
    # exactly one bracket row on file: the store is append-only and the
    # resumed run did not re-file it
    lines = [json.loads(l) for l in
             _cell_dir(world.out_root).joinpath("rows.jsonl")
             .read_text().splitlines() if l.strip()]
    assert sum(1 for r in lines
               if r["metric"] == srun.MMLU_UNDETERMINABLE_METRIC) == 1


def test_mmlu_complete_topk_rows_keep_their_values_and_name_their_case(world):
    """Every letter inside the window: value, pick, gold, nlls, top_logprobs
    and the censorship markers equal the rows the driver filed before this
    port, byte for byte once the ONE added key — the case label,
    all_present — is set aside; no window record, no bracket row, and the
    aggregate row's keys are unchanged."""
    from sisteer.capeval.manifest import canonical_json
    from sisteer.capeval.tiers.parity import PARITY_AGG_METRIC

    srun.main(_mmlu_argv(world))
    items, aggs = _mmlu_store(world)
    assert set(items) == set(PRE_PORT_COMPLETE_ROWS)
    for iid, (value, extra_json) in PRE_PORT_COMPLETE_ROWS.items():
        assert items[iid]["value"] == value
        x = json.loads(items[iid]["extra_json"])
        assert x.pop(srun.MMLU_LETTER_READ_KEY) == srun.MMLU_LETTER_READ_ALL_PRESENT
        assert "incomplete_topk" not in x
        assert canonical_json(x) == extra_json
    by_metric = _mmlu_rows_by_metric(world)
    assert set(by_metric) == {"acc", PARITY_AGG_METRIC}
    ax = json.loads(aggs[0]["extra_json"])
    assert ax["censored_items"] == 0 and ax["censored_letter_counts"] == {}
    assert "topk_incomplete_scored_new" not in ax
    assert srun.MMLU_LETTER_READ_KEY not in ax  # a per-item label only


# --------------------------------------------------------------------------- #
# PB13: the manifest records the parser the standup SERVED, read from its      #
# endpoint.json beside the routing table — never a literal                     #
# --------------------------------------------------------------------------- #


def _manifest_args(**over):
    base = {"concurrency": 4,
            "ledger_url": "", "samples": None, "limit": 0,
            "inspect_limit": 0, "max_goals": 0, "sampling": tc.SAMPLING_DEFAULT}
    base.update(over)
    return SimpleNamespace(**base)


def _sampling_ns(atoms=("humaneval_plus",), *, mode=tc.SAMPLING_CARD, model_key="llama8b",
                 thinking=tc.THINKING_OFF, samples=None):
    """What `srun._resolve_sampling` hands the manifest builder for a run of
    `atoms`: the mode, the key, the row and the per-task k map."""
    by_task = tc.run_thinking_by_task(list(atoms), thinking)
    return SimpleNamespace(mode=mode, model_key=model_key, row=tc.sampling_for(model_key, mode),
                           samples_by_task=tc.run_samples_by_task(by_task, mode, math_samples=samples),
                           samples=tc.samples_of_task(mode, "inspect/math", math_samples=samples))


def test_standup_endpoint_is_read_beside_the_routing_table_and_refuses_absence(world, monkeypatch, capsys):
    ep = srun._standup_endpoint(world.table_path)
    assert ep["routing_table"] == str(world.table_path)
    assert ep["reasoning_parser"] is None and ep["tool_calls"] is False
    assert ep["geometry"]["max_model_len"] == STANDUP_WINDOW      # the served geometry (R4)
    path = world.table_path.parent / srun.STANDUP_ENDPOINT_FILENAME
    assert path.name == "endpoint.json"            # capctrl_serving_up.py's file name
    assert ep[tc.AGENT_LOOP_ENTRY_KEY] is True                    # the proxy's entry (PW18)
    # a record without the modality keys (a standup that predates PB13) refuses
    path.write_text(json.dumps({"proxy_url": PROXY_URL}))
    with pytest.raises(SystemExit,
                       match="lacks \\['routing_table', 'reasoning_parser', 'tool_calls', "
                             "'geometry', 'agent_loop_entry'\\]"):
        srun._standup_endpoint(world.table_path)
    # the modality keys alone are not enough either: the binding keys (N2), the
    # geometry (N3) and the agent-loop entry (PW18) are required too
    path.write_text(json.dumps({"reasoning_parser": None, "tool_calls": False}))
    with pytest.raises(SystemExit,
                       match="lacks \\['proxy_url', 'routing_table', 'geometry', 'agent_loop_entry'\\]"):
        srun._standup_endpoint(world.table_path)
    # PW18: a standup from before the entry was recorded cannot vouch for the
    # regime its tool turns are steered under — refused like a missing parser
    write_standup_endpoint(world.table_path)
    rec = json.loads(path.read_text())
    del rec[tc.AGENT_LOOP_ENTRY_KEY]
    path.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="lacks \\['agent_loop_entry'\\]"):
        srun._standup_endpoint(world.table_path)
    for bad in (None, 1, "true", "yes"):                          # a stated fact, never a coercion
        write_standup_endpoint(world.table_path, agent_loop_entry=bad)
        with pytest.raises(SystemExit, match="agent_loop_entry is .* not a JSON boolean"):
            srun._standup_endpoint(world.table_path)
    write_standup_endpoint(world.table_path, agent_loop_entry=False)
    assert srun._standup_endpoint(world.table_path)[tc.AGENT_LOOP_ENTRY_KEY] is False
    # E25: the lens reduction mode is OPTIONAL (absent = per_row) but, stated,
    # must be a known mode. E29 / PW18b: the attestation is the truth and the
    # environment yields the parser's expectation, which must AGREE — a
    # disagreement refuses naming both, never records either. E46 flipped what
    # an UNSET switch expects (stacked, since the landing is merged), so an
    # unset container now AGREES with a stacked-attesting server and REFUSES a
    # pre-landing keyless one — the mirror image of the pre-landing pins.
    write_standup_endpoint(world.table_path, **{tc.PROXY_IDENTITY_REDUCTION_MODE: "stacked"})
    monkeypatch.delenv(tc.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    ep_st = srun._standup_endpoint(world.table_path)           # unset (= stacked) vs stacked: agrees
    assert ep_st[tc.PROXY_IDENTITY_REDUCTION_MODE] == "stacked"
    assert srun._served_identity(ep_st, concurrency=4)[tc.REDUCTION_MODE_FIELD] == "stacked"
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, "0")
    with pytest.raises(SystemExit, match="attests lens_reduction_mode='stacked' but this container's "
                                         "SISTEER_LENS_STACKED_REDUCTIONS expects 'per_row'"):
        srun._standup_endpoint(world.table_path)
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, "1")
    ep_st = srun._standup_endpoint(world.table_path)                 # "1" -> stacked expected: agrees
    assert ep_st[tc.PROXY_IDENTITY_REDUCTION_MODE] == "stacked"
    assert srun._served_identity(ep_st, concurrency=4)[tc.REDUCTION_MODE_FIELD] == "stacked"
    write_standup_endpoint(world.table_path)                      # a pre-landing server: no key
    with pytest.raises(SystemExit, match="expects lens_reduction_mode='stacked' but the server attests no "
                                         "reduction mode at all — a pre-landing server"):
        srun._standup_endpoint(world.table_path)                  # env "1" vs keyless: refused
    monkeypatch.delenv(tc.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    with pytest.raises(SystemExit, match="expects lens_reduction_mode='stacked' but the server attests no "
                                         "reduction mode at all — a pre-landing server"):
        srun._standup_endpoint(world.table_path)     # unset is stacked too now (E46): keyless refused
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, "per_row")
    ep_pr = srun._standup_endpoint(world.table_path)              # per_row vs keyless: agrees
    assert srun._served_identity(ep_pr, concurrency=4)[tc.REDUCTION_MODE_FIELD] == "per_row"
    out = capsys.readouterr().out
    assert "pre-landing server" in out and "reduction_mode='per_row'" in out
    # the merge of the stacked landing (main f39a4562) made a post-landing
    # standup ALWAYS write the key, null plus a reason when it could not read
    # its own workers' attestation. That is NOT a pre-landing server and must
    # not fall back to the per_row default, whatever this container expects.
    write_standup_endpoint(world.table_path,
                           **{tc.PROXY_IDENTITY_REDUCTION_MODE: None,
                              srun.STANDUP_REDUCTION_MODE_ERROR_KEY:
                                  "worker reports disagree on lens_reduction_mode: "
                                  "['per_row', 'stacked']"})
    for env in (tc.REDUCTION_MODE_PER_ROW, tc.REDUCTION_MODE_STACKED):
        monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, env)
        with pytest.raises(SystemExit, match="lens_reduction_mode is null and "
                                             "lens_reduction_mode_error is .*disagree"):
            srun._standup_endpoint(world.table_path)
    # a null with NO reason is still the pre-landing reading (nothing stated)
    write_standup_endpoint(world.table_path,
                           **{tc.PROXY_IDENTITY_REDUCTION_MODE: None})
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, "per_row")
    assert srun._served_identity(srun._standup_endpoint(world.table_path),
                                 concurrency=4)[tc.REDUCTION_MODE_FIELD] == "per_row"
    capsys.readouterr()
    write_standup_endpoint(world.table_path, **{tc.PROXY_IDENTITY_REDUCTION_MODE: "per_row"})
    assert srun._served_identity(srun._standup_endpoint(world.table_path),
                                 concurrency=4)[tc.REDUCTION_MODE_FIELD] == "per_row"
    assert "pre-landing" not in capsys.readouterr().out            # an explicit per_row is not keyless
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, "maybe")
    with pytest.raises(SystemExit, match="is not one of"):
        srun._standup_endpoint(world.table_path)
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, tc.REDUCTION_MODE_PER_ROW)
    write_standup_endpoint(world.table_path, **{tc.PROXY_IDENTITY_REDUCTION_MODE: "stacked"})
    for bad in ("per-row", "", None, 1):
        write_standup_endpoint(world.table_path, **{tc.PROXY_IDENTITY_REDUCTION_MODE: bad})
        if bad is None:
            assert tc.PROXY_IDENTITY_REDUCTION_MODE in json.loads(path.read_text())
            srun._standup_endpoint(world.table_path)             # an explicit null reads as absent
            continue
        with pytest.raises(SystemExit, match="not one of \\['per_row', 'stacked'\\]"):
            srun._standup_endpoint(world.table_path)
    # a geometry block without a positive window is refused: the window is identity (N3)
    for bad in (None, 0, -1, "4096", True):
        write_standup_endpoint(world.table_path, geometry={**standup_geometry(), "max_model_len": bad})
        with pytest.raises(SystemExit, match="max_model_len is .* not a positive integer"):
            srun._standup_endpoint(world.table_path)
    # ... nor without a positive S: the served S is identity too (PW5-4)
    for bad in (None, 0, "4", False):
        write_standup_endpoint(world.table_path, geometry={**standup_geometry(), "max_num_seqs": bad})
        with pytest.raises(SystemExit, match="max_num_seqs is .* not a positive integer"):
            srun._standup_endpoint(world.table_path)
    write_standup_endpoint(world.table_path, geometry="not a block")
    with pytest.raises(SystemExit, match="not a positive integer"):
        srun._standup_endpoint(world.table_path)
    # no record at all refuses, naming the path
    path.unlink()
    with pytest.raises(SystemExit, match="no standup record at .*endpoint.json"):
        srun._standup_endpoint(world.table_path)


def test_the_null_mode_rule_has_one_implementation_and_the_two_readings_are_pinned_to_each_other(
        world, monkeypatch):
    """The three-reading rule (named mode / neither key / null WITH a reason)
    started as an inline copy in ``_standup_endpoint``; main took it into
    ``lens_worker.endpoint_reduction_mode`` and the second main merge calls it
    back, so there is ONE implementation of it in this tree and this read site
    is a caller. What did NOT move is contracts' ``resolve_reduction_mode``,
    which answers a different question (do several stated values agree, and is
    a stated value in the vocabulary) at four read sites — so the two coexist
    on purpose, and the two places where they differ are pinned here rather
    than settled by preferring one:

    * the EXCEPTION: ``ReductionModeUnattested`` (a ``RuntimeError``) from
      lens_worker, ``ValueError`` from contracts. The launcher turns both into
      ``SystemExit``, so nothing downstream can tell them apart;
    * the unknown-mode WORDING: contracts renders the vocabulary as a LIST,
      ``per_row`` first, and lens_worker as its own TUPLE, ``stacked`` first.
      The launcher's refusal stays contracts', unchanged by the merge, because
      the vocabulary reading runs first.

    Everywhere else the two agree, and the read site now checks that at
    runtime instead of trusting this test to notice the day they part."""
    from sisteer.capeval.serving import lens_worker as lw
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, tc.REDUCTION_MODE_PER_ROW)
    path = world.table_path.parent / srun.STANDUP_ENDPOINT_FILENAME
    K, EK = srun.STANDUP_REDUCTION_MODE_KEY, srun.STANDUP_REDUCTION_MODE_ERROR_KEY
    assert (K, EK) == (lw.LENS_REDUCTION_MODE_KEY, lw.LENS_REDUCTION_MODE_ERROR_KEY)
    PR, ST = tc.REDUCTION_MODE_PER_ROW, tc.REDUCTION_MODE_STACKED

    # (1) the rule is not re-implemented at the read site: its sentence is gone
    # from this driver's source, and a stubbed resolver reaches the refusal
    src = Path(srun.__file__).read_text()
    assert "is null and" not in src and "endpoint_reduction_mode" in src

    def _boom(*a, **k):
        raise lw.ReductionModeUnattested("STUBBED: the resolver ran")

    write_standup_endpoint(world.table_path, **{K: PR})
    monkeypatch.setattr(lw, "endpoint_reduction_mode", _boom)
    with pytest.raises(SystemExit, match=r"STUBBED: the resolver ran \(E29 / PW18b\)"):
        srun._standup_endpoint(world.table_path)
    monkeypatch.undo()
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, PR)

    # (2) PARITY over the whole record vocabulary, as (mode, refused) pairs.
    # They part on exactly one thing, in two spellings: a STATED REASON with no
    # mode — whether the mode key is null or absent altogether. Contracts is
    # handed one value and cannot see the second key, so it reads the
    # pre-landing default; lens_worker reads both keys and refuses. That is the
    # hazard the ported rule exists to close, and why the read site goes
    # through lens_worker and not contracts alone.
    def _tc_read(rec):
        try:
            return tc.resolve_reduction_mode(rec.get(K), sources=("record",)), False
        except ValueError:
            return None, True

    def _lw_read(rec):
        try:
            return lw.endpoint_reduction_mode(rec), False
        except lw.ReductionModeUnattested:
            return None, True

    agree = [{}, {K: PR}, {K: ST}, {K: None}, {K: None, EK: ""}, {K: PR, EK: "ignored"},
             {K: "per-row"}, {K: ""}, {K: 1}, {K: True}, {K: ["stacked"]}]
    for rec in agree:
        assert _tc_read(rec) == _lw_read(rec), rec
    why_disagree = "worker reports disagree: ['per_row', 'stacked']"
    for rec in ({K: None, EK: why_disagree}, {EK: why_disagree}):
        assert _tc_read(rec) == (PR, False) and _lw_read(rec) == (None, True), rec
    # ... and the vocabularies are the same two modes, whatever order each
    # spells them in — that order is what makes the two wordings differ
    assert set(tc.REDUCTION_MODES) == set(lw.la.REDUCTION_MODES) == {PR, ST}

    # (3) the two deliberate differences, at the launcher's boundary
    assert issubclass(lw.ReductionModeUnattested, RuntimeError)
    assert not issubclass(lw.ReductionModeUnattested, ValueError)
    write_standup_endpoint(world.table_path, **{K: "per-row"})
    with pytest.raises(SystemExit) as ei:            # contracts' wording, not lens_worker's
        srun._standup_endpoint(world.table_path)
    assert "not one of ['per_row', 'stacked']" in str(ei.value)
    assert "('stacked', 'per_row')" not in str(ei.value)
    assert "refusing to record an unknown lens reduction mode as served identity" in str(ei.value)

    # (4) the ported refusal reads EXACTLY as it did before the merge: the
    # resolver's sentence, this launcher's prefix, this launcher's marker
    why = "worker reports disagree on lens_reduction_mode: ['per_row', 'stacked']"
    write_standup_endpoint(world.table_path, **{K: None, EK: why})
    with pytest.raises(SystemExit) as ei:
        srun._standup_endpoint(world.table_path)
    assert str(ei.value) == (
        f"[served-cap] standup record {path}: {K} is null and {EK} is {why!r} — the standup "
        "could not determine the lens reduction mode its own workers served, so this is not a "
        f"pre-landing server (which states neither key and means {PR!r}); refusing rather than "
        "recording the default as served identity (E29 / PW18b)")

    # (5) the runtime drift guard: if the two readings of one record ever part,
    # the read site refuses and names both rather than picking one
    write_standup_endpoint(world.table_path, **{K: PR})
    assert srun._standup_endpoint(world.table_path)[K] == PR
    monkeypatch.setattr(lw, "endpoint_reduction_mode", lambda *a, **k: ST)
    with pytest.raises(SystemExit, match=r"this launcher reads lens_reduction_mode='per_row' .*"
                                         r"endpoint_reduction_mode reads 'stacked' .* have drifted"):
        srun._standup_endpoint(world.table_path)


@pytest.mark.parametrize("thinking, served_parser, served_tools", [
    ("off", None, False),          # the ordinary thinking-OFF standup
    ("on", "gemma4", False),       # thinking-ON: the family pin was composed
    ("on", "gemma4", True),        # thinking-ON + BFCL: --tool-calls too
    ("off", "gemma4", True),       # recorded faithfully even when unusual
])
def test_tinycap_manifest_records_the_served_parser_and_tool_calls(
        world, monkeypatch, thinking, served_parser, served_tools):
    # N4: version discovery is stubbed EXPLICITLY — the frameworks block would
    # otherwise import the real evalplus for the EvalPlus family, which a venv
    # without the capeval group lacks; the manifest under test is the parser /
    # tool-call / geometry record, and the fake versions are asserted as fakes
    stub_framework_versions(monkeypatch)
    routes = list(px.load_routing_table(world.table_path)["routes"])
    endpoint = {"reasoning_parser": served_parser, "tool_calls": served_tools,
                "geometry": standup_geometry(), tc.AGENT_LOOP_ENTRY_KEY: True}
    # PW5-2: the manifest's frameworks block IS the identity's (one source);
    # PW20b-D: the identity also carries the per-task arm map the block mirrors
    by_task = tc.run_thinking_by_task(["humaneval_plus"], thinking)
    identity = {"tasks": [], "frameworks": srun._framework_versions({"humaneval_plus"}),
                tc.IDENTITY_KEY_THINKING_BY_TASK: by_task}
    m = srun._tinycap_manifest(
        atoms=["humaneval_plus"], thinking=thinking, args=_manifest_args(),
        route=world.route, routes=routes, identity=identity,
        endpoint=endpoint, sampling=_sampling_ns(thinking=thinking))
    assert m["frameworks"] == identity["frameworks"]
    # U8: the sampling block mirrors the run's draw — llama8b's row, every task at 1
    assert set(m[tc.MANIFEST_KEY_SAMPLING]) == set(tc.SAMPLING_BLOCK_FIELDS)
    assert m[tc.MANIFEST_KEY_SAMPLING]["mode"] == "card" and m[tc.MANIFEST_KEY_SAMPLING]["model_key"] == "llama8b"
    assert (m[tc.MANIFEST_KEY_SAMPLING]["temperature"], m[tc.MANIFEST_KEY_SAMPLING]["top_p"],
            m[tc.MANIFEST_KEY_SAMPLING]["top_k"]) == (0.6, 0.9, None)
    assert m[tc.MANIFEST_KEY_SAMPLING]["samples_by_task"] == {t: 1 for t in by_task}
    assert m[tc.MANIFEST_KEY_SAMPLING]["source"] == tc.SAMPLING_BY_MODEL_KEY["llama8b"].source
    assert m["thinking"]["reasoning_parser"] == served_parser
    assert m["route"]["tool_calls"] is served_tools
    # PW18: the route block mirrors the served entry and the reduction mode
    # (no key on the record = per_row, E25)
    assert m["route"][tc.AGENT_LOOP_ENTRY_KEY] is True
    assert m["route"][tc.REDUCTION_MODE_FIELD] == tc.REDUCTION_MODE_PER_ROW
    m2 = srun._tinycap_manifest(
        atoms=["humaneval_plus"], thinking=thinking, args=_manifest_args(),
        route=world.route, routes=routes, identity=identity,
        endpoint={**endpoint, tc.AGENT_LOOP_ENTRY_KEY: False,
                  tc.PROXY_IDENTITY_REDUCTION_MODE: tc.REDUCTION_MODE_STACKED},
        sampling=_sampling_ns(thinking=thinking))
    assert m2["route"][tc.AGENT_LOOP_ENTRY_KEY] is False
    assert m2["route"][tc.REDUCTION_MODE_FIELD] == tc.REDUCTION_MODE_STACKED
    assert m["geometry"]["max_num_seqs"] == STANDUP_SEQS and m["geometry"]["pass"] is None
    assert m["frameworks"]["evalplus"] == "evalplus-test"        # the stub, not a real import
    assert m["frameworks"]["inspect_ai"] is None                 # not requested
    assert m[tc.MANIFEST_KEY_DECLARATION] == {"response_prefill_used": []}   # no task declares a prefill
    assert m["thinking"]["enable_thinking"] is (thinking == "on")
    assert set(m["thinking"]) == set(tc.THINKING_BLOCK_FIELDS)
    # PW20b-D: the block names the driver's mode and mirrors the hashed map
    assert m["thinking"]["mode"] == thinking
    assert m["thinking"]["by_task"] == by_task
    assert set(m["route"]) == set(tc.ROUTE_BLOCK_FIELDS)
    # the manifest can never say "no parser" while the standup served one
    if served_parser is not None:
        assert m["thinking"]["reasoning_parser"] is not None
    # the thinking-ON arm on gemma-4 records the contracts' family pin
    if thinking == "on":
        assert m["thinking"]["reasoning_parser"] == tc.REASONING_PARSER_BY_FAMILY["gemma4"]


def fake_frameworks(monkeypatch):
    """inspect_ai / inspect_evals are not in the lane venv: the manifest's
    frameworks block imports them for a requested Inspect family, so fake
    modules carrying the PINNED versions stand in (evalplus is the real
    installed one here) — the pin check (PW5-2) passes
    on them exactly as it does on the real client venv."""
    import types

    for name, ver in (("inspect_ai", tc.INSPECT_AI_PIN), ("inspect_evals", tc.INSPECT_EVALS_PIN)):
        mod = types.ModuleType(name)
        mod.__version__ = ver
        monkeypatch.setitem(sys.modules, name, mod)


def stub_framework_versions(monkeypatch, tag: str = "test"):
    """Stub the driver's framework discovery AND the pins it checks against
    with ONE helper (N4 + PW5-2): ``_import_version`` returns
    ``<module>-<tag>`` and ``FRAMEWORK_PINS`` becomes the same fakes — so the
    pin CHECK still runs on every stubbed run (never skipped) while the manifest
    is asserted to carry the fakes, in every venv. A test that wants the
    check to FAIL re-patches one side afterwards."""
    def fake(module, dist=None):
        return f"{module}-{tag}"

    monkeypatch.setattr(srun, "_import_version", fake)
    monkeypatch.setattr(srun, "FRAMEWORK_PINS",
                        {k: fake(m) for k, m in srun.FRAMEWORK_MODULES.items()})
    return fake


def test_framework_versions_are_real_for_requested_families_and_null_otherwise(monkeypatch):
    """R1/R4: the frameworks block names the version of every framework a
    REQUESTED family runs on, read off the imported module; None only for a
    family not requested; a requested framework that cannot import refuses
    rather than recording null."""
    import importlib.util

    if importlib.util.find_spec("evalplus") is None:
        pytest.skip("evalplus not installed in this venv (the lane venv has it; this test is about the "
                    "REAL import)")
    assert srun.FRAMEWORK_PINS is tc.FRAMEWORK_PINS                  # the check reads the contracts' pins
    assert not hasattr(srun, "FRAMEWORK_VCS_PINS")
    fake_frameworks(monkeypatch)
    fw = srun._framework_versions({"humaneval_plus"})
    assert set(fw) == set(tc.FRAMEWORKS_BLOCK_FIELDS)
    assert "lm_eval" not in fw and "tinybenchmarks" not in fw
    assert fw["evalplus"] == tc.EVALPLUS_PIN
    assert fw["inspect_ai"] is None and fw["inspect_evals"] is None
    fw = srun._framework_versions({"math500", "humaneval_plus"})
    assert fw["inspect_ai"] == tc.INSPECT_AI_PIN and fw["inspect_evals"] == tc.INSPECT_EVALS_PIN
    assert fw["evalplus"] == tc.EVALPLUS_PIN
    assert fw["client_lock_sha256"] == srun._client_lock_sha256() and len(fw["client_lock_sha256"]) == 64
    with pytest.raises(SystemExit, match="cannot be imported"):
        srun._import_version("no_such_framework_for_this_test")
    assert srun._client_lock_sha256(Path("/nonexistent/tree")) is None


def test_framework_pin_mismatch_refuses_naming_imported_and_pinned(monkeypatch):
    """PW5-2 (re-review F2): a REQUESTED framework whose imported version is
    not the client lock's pin refuses (FrameworkPinMismatch, a SystemExit)
    naming both values — the pinned constant is stamped into every row, so
    an unpinned environment cannot say what ran; a framework NOT requested
    is never imported and never refuses."""
    import types

    assert issubclass(srun.FrameworkPinMismatch, SystemExit)
    assert set(srun.FRAMEWORK_MODULES) == set(tc.FRAMEWORK_PINS) == set(tc.FRAMEWORKS_BLOCK_FIELDS) - {"client_lock_sha256"}
    off = types.ModuleType("inspect_ai")
    off.__version__ = "0.3.999"
    monkeypatch.setitem(sys.modules, "inspect_ai", off)
    ok = types.ModuleType("inspect_evals")
    ok.__version__ = tc.INSPECT_EVALS_PIN
    monkeypatch.setitem(sys.modules, "inspect_evals", ok)
    with pytest.raises(srun.FrameworkPinMismatch,
                       match=f"inspect_ai: imported '0.3.999', the client lock pins '{tc.INSPECT_AI_PIN}'"):
        srun._framework_versions({"math500"})
    # not requested: nothing imported, nothing refused, every version null
    fw = srun._framework_versions(set())
    assert {k: fw[k] for k in tc.FRAMEWORK_PINS} == {k: None for k in tc.FRAMEWORK_PINS}
    pins_by_module = {m: tc.FRAMEWORK_PINS[k] for k, m in srun.FRAMEWORK_MODULES.items()}
    # every disagreement is named at once
    monkeypatch.setattr(srun, "_import_version", lambda module, dist=None: "0.0.0")
    with pytest.raises(srun.FrameworkPinMismatch) as ei:
        srun._framework_versions({"math500", "humaneval_plus"})
    for key in tc.FRAMEWORK_PINS:
        assert f"{key}: imported '0.0.0'" in str(ei.value), key
    assert tc.CLIENT_LOCK_TXT in str(ei.value)
    # at the pin: passes and records exactly the pins
    monkeypatch.setattr(srun, "_import_version", lambda module, dist=None: pins_by_module[module])
    fw = srun._framework_versions({"math500", "humaneval_plus"})
    assert {k: fw[k] for k in tc.FRAMEWORK_PINS} == dict(tc.FRAMEWORK_PINS)


def test_producer_block_reads_the_tree_sha_file_else_git_head_else_refuses(tmp_path):
    """PW5-5 (Evals item 3): ``manifest["producer"]`` names the tree that
    produced the store — the launcher's TREE_SHA file first (a container has
    no .git), else the git HEAD of a checkout, else a refusal naming the
    file — so the later EvalPlus merge can require the same tree."""
    import subprocess

    stamped = tmp_path / "stamped"
    stamped.mkdir()
    (stamped / tc.TREE_SHA_FILENAME).write_text("0123456789abcdef0123456789abcdef01234567\n")
    block = srun._producer_block(stamped)
    assert block == {"tree_sha": "0123456789abcdef0123456789abcdef01234567",
                     "driver": tc.PRODUCER_DRIVER_SERVED}
    assert tuple(block) == tc.PRODUCER_BLOCK_FIELDS and block["driver"] == "served_capability_run"
    # the file wins over git when both are present (the launcher's stamp is the upload's)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert srun._producer_block(REPO) == {"tree_sha": head, "driver": tc.PRODUCER_DRIVER_SERVED}
    assert not (REPO / tc.TREE_SHA_FILENAME).exists()       # the repo tree carries no stamp; git HEAD it is
    assert srun._producer_block() == srun._producer_block(REPO)
    # an empty or multi-token stamp refuses
    (stamped / tc.TREE_SHA_FILENAME).write_text("  \n")
    with pytest.raises(SystemExit, match="does not hold a single sha token"):
        srun._producer_block(stamped)
    (stamped / tc.TREE_SHA_FILENAME).write_text("abc def\n")
    with pytest.raises(SystemExit, match="does not hold a single sha token"):
        srun._producer_block(stamped)
    # neither a stamp nor a git checkout: refused, naming the file
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit, match=f"no {tc.TREE_SHA_FILENAME} file at .*bare and it is not a git checkout"):
        srun._producer_block(bare)


def test_geometry_block_takes_the_pass_from_the_launcher_record_and_refuses_disagreement(tmp_path):
    """R4: ``geometry`` is the standup's served block (endpoint.json) plus the
    PASS number from the launcher's served_geometry.json beside the cells
    (PI5) when it exists; the two must agree on the window and S."""
    ep = {"reasoning_parser": None, "tool_calls": False, "geometry": standup_geometry()}
    g = srun._geometry_block(ep, tmp_path)
    assert g["max_model_len"] == STANDUP_WINDOW and g["max_num_seqs"] == STANDUP_SEQS
    assert g["pass"] is None and g["source"] == srun.STANDUP_ENDPOINT_FILENAME
    (tmp_path / srun.SERVED_GEOMETRY_FILENAME).write_text(json.dumps(
        {"geometry": {**standup_geometry(), "pass": 1, "budget_override": {"budget": 589824}}}))
    g = srun._geometry_block(ep, tmp_path)
    assert g["pass"] == 1 and g["budget_override"] == {"budget": 589824}
    assert g["source"] == f"{srun.STANDUP_ENDPOINT_FILENAME}+{srun.SERVED_GEOMETRY_FILENAME}"
    (tmp_path / srun.SERVED_GEOMETRY_FILENAME).write_text(json.dumps(
        {"geometry": {**standup_geometry(), "max_num_seqs": STANDUP_SEQS + 4, "pass": 1}}))
    with pytest.raises(SystemExit, match="disagree"):
        srun._geometry_block(ep, tmp_path)
    # no geometry in the record: nothing is invented (the reader refuses this for a real run)
    assert srun._geometry_block({"reasoning_parser": None, "tool_calls": False}, tmp_path) is None
    # PW8 (E16 A2): the SAME record feeds the hashed identity's pass — one reader
    assert srun._launcher_geometry_record(tmp_path / "nowhere") is None
    assert srun._launcher_pass(None) is None and srun._launcher_pass({}) is None
    (tmp_path / srun.SERVED_GEOMETRY_FILENAME).write_text(json.dumps(
        {"geometry": {**standup_geometry(), "pass": 2}}))
    rec = srun._launcher_geometry_record(tmp_path)
    assert srun._launcher_pass(rec) == 2 == srun._geometry_block(ep, tmp_path)["pass"]
    for bad in (3, True, "1"):
        with pytest.raises(SystemExit, match="not one of"):
            srun._launcher_pass({**rec, "pass": bad})
    (tmp_path / srun.SERVED_GEOMETRY_FILENAME).write_text(json.dumps(
        {"geometry": {**standup_geometry(), "pass": 3}}))
    with pytest.raises(SystemExit, match="not one of"):
        srun._geometry_block(ep, tmp_path)
    (tmp_path / srun.SERVED_GEOMETRY_FILENAME).write_text(json.dumps({"standup_argv": []}))
    with pytest.raises(SystemExit, match="no geometry block"):
        srun._launcher_geometry_record(tmp_path)


def _ep(parser="gemma4", tools=False, window=STANDUP_WINDOW, seqs=STANDUP_SEQS,
        entry=True, mode=None) -> dict:
    ep = {"reasoning_parser": parser, "tool_calls": tools,
          "geometry": standup_geometry(max_model_len=window, max_num_seqs=seqs),
          tc.AGENT_LOOP_ENTRY_KEY: entry}
    if mode is not None:
        ep[tc.PROXY_IDENTITY_REDUCTION_MODE] = mode
    return ep


def test_manifest_serving_identity_guard_refuses_another_parser_tool_mode_window_s_or_width(tmp_path):
    """R4 / E10 N3 / PW5-4: a cell manifest recording one served identity —
    parser, tool-call mode, WINDOW, served S and the client width — never
    resumes under a run that served another; refused by a typed error naming
    old vs new per field. (PW4-5's "S resumes" is superseded: the
    served-path owner's 2026-09-04 re-check ruled S and the client width
    identity.)"""
    store = SimpleNamespace(manifest_path=tmp_path / "manifest.json", cell_dir=tmp_path)
    check = srun._check_manifest_serving_identity
    check(store, _ep(), concurrency=4, pass_no=None)                                        # fresh: fine
    store.manifest_path.write_text(json.dumps(
        {"thinking": {"reasoning_parser": "gemma4"},
         "route": {"tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True,
                   tc.REDUCTION_MODE_FIELD: "per_row"},
         "geometry": {"max_model_len": STANDUP_WINDOW, "max_num_seqs": STANDUP_SEQS},
         "concurrency": 4}))
    check(store, _ep(), concurrency=4, pass_no=None)                                        # same identity
    # PW18: the agent-loop entry and the lens reduction mode are identity too
    with pytest.raises(srun.ServingIdentityMismatch,
                       match="agent_loop_entry: the cell manifest records True but this "
                             "standup served False"):
        check(store, _ep(entry=False), concurrency=4, pass_no=None)
    with pytest.raises(srun.ServingIdentityMismatch,
                       match="reduction_mode: the cell manifest records 'per_row' but this "
                             "standup served 'stacked'"):
        check(store, _ep(mode="stacked"), concurrency=4, pass_no=None)
    check(store, _ep(mode="per_row"), concurrency=4, pass_no=None)                          # stated = default
    # a LEGACY manifest (route block without the mode) maps to per_row (E25:
    # every store before the landing was served per-row) — and a manifest
    # without the entry cannot vouch for its tool-turn regime (named, like a
    # missing window)
    store.manifest_path.write_text(json.dumps(
        {"thinking": {"reasoning_parser": "gemma4"}, "route": {"tool_calls": False},
         "geometry": {"max_model_len": STANDUP_WINDOW, "max_num_seqs": STANDUP_SEQS},
         "concurrency": 4}))
    with pytest.raises(srun.ServingIdentityMismatch,
                       match="agent_loop_entry: the cell manifest records None but this "
                             "standup served True") as ei:
        check(store, _ep(), concurrency=4, pass_no=None)
    assert "reduction_mode" not in str(ei.value)                     # legacy = per_row: no mismatch
    with pytest.raises(srun.ServingIdentityMismatch) as ei:
        check(store, _ep(mode="stacked"), concurrency=4, pass_no=None)
    assert "reduction_mode: the cell manifest records 'per_row' but this standup served 'stacked'" in str(ei.value)
    store.manifest_path.write_text(json.dumps(
        {"thinking": {"reasoning_parser": "gemma4"},
         "route": {"tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True},
         "geometry": {"max_model_len": STANDUP_WINDOW, "max_num_seqs": STANDUP_SEQS},
         "concurrency": 4}))
    check(store, _ep(), concurrency=4, pass_no=None)                                        # legacy mode = per_row
    store.manifest_path.write_text(json.dumps(
        {"thinking": {"reasoning_parser": "gemma4"},
         "route": {"tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True,
                   tc.REDUCTION_MODE_FIELD: "per_row"},
         "geometry": {"max_model_len": STANDUP_WINDOW, "max_num_seqs": STANDUP_SEQS},
         "concurrency": 4}))
    with pytest.raises(srun.ServingIdentityMismatch,
                       match="reasoning_parser: the cell manifest records 'gemma4' but this "
                             "standup served None"):
        check(store, _ep(parser=None), concurrency=4, pass_no=None)
    with pytest.raises(srun.ServingIdentityMismatch,
                       match="tool_calls: the cell manifest records False but this standup "
                             "served True"):
        check(store, _ep(tools=True), concurrency=4, pass_no=None)
    with pytest.raises(srun.ServingIdentityMismatch,
                       match=f"max_model_len: the cell manifest records {STANDUP_WINDOW} but "
                             "this standup served 8192"):
        check(store, _ep(window=8192), concurrency=4, pass_no=None)
    with pytest.raises(srun.ServingIdentityMismatch,
                       match=f"max_num_seqs: the cell manifest records {STANDUP_SEQS} but "
                             f"this standup served {STANDUP_SEQS + 12}"):
        check(store, _ep(seqs=STANDUP_SEQS + 12), concurrency=4, pass_no=None)             # S changed: identity (PW5-4)
    with pytest.raises(srun.ServingIdentityMismatch,
                       match="concurrency: the cell manifest records 4 but this standup served 2"):
        check(store, _ep(), concurrency=2, pass_no=None)                                    # client width changed
    # every differing field is named at once
    with pytest.raises(srun.ServingIdentityMismatch) as ei:
        check(store, _ep(parser=None, window=8192, seqs=1), concurrency=1, pass_no=None)
    for field in ("reasoning_parser:", "max_model_len:", "max_num_seqs:", "concurrency:"):
        assert field in str(ei.value), field
    assert "resume mismatch on the served identity" in str(ei.value)
    # PW8 (E16 A2): the geometry PASS is named too — a hand run's store (no
    # pass) never resumes under a launcher pass, nor pass 1's under pass 2
    with pytest.raises(srun.ServingIdentityMismatch,
                       match="pass: the cell manifest records None but this run is pass 1"):
        check(store, _ep(), concurrency=4, pass_no=1)
    store.manifest_path.write_text(json.dumps(
        {"thinking": {"reasoning_parser": "gemma4"},
         "route": {"tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True},
         "geometry": {"max_model_len": STANDUP_WINDOW, "max_num_seqs": STANDUP_SEQS, "pass": 1},
         "concurrency": 4}))
    check(store, _ep(), concurrency=4, pass_no=1)
    with pytest.raises(srun.ServingIdentityMismatch,
                       match="pass: the cell manifest records 1 but this run is pass 2"):
        check(store, _ep(), concurrency=4, pass_no=2)
    with pytest.raises(srun.ServingIdentityMismatch) as ei:
        check(store, _ep(window=8192), concurrency=4, pass_no=None)
    assert "pass: the cell manifest records 1 but this run is pass None" in str(ei.value)
    assert "max_model_len: the cell manifest records" in str(ei.value)
    assert issubclass(srun.ServingIdentityMismatch, SystemExit)             # the CLI exits like every refusal
    # a manifest that predates the geometry block cannot vouch for its window or S: refused
    store.manifest_path.write_text(json.dumps(
        {"thinking": {"reasoning_parser": "gemma4"},
         "route": {"tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True}, "concurrency": 4}))
    with pytest.raises(srun.ServingIdentityMismatch, match="max_model_len: the cell manifest records None") as ei:
        check(store, _ep(), concurrency=4, pass_no=None)
    assert "max_num_seqs: the cell manifest records None" in str(ei.value)


def test_served_identity_carries_the_window_s_and_client_width(world):
    """E10 N3 + PW5-4: ``resolved_config.tinycap.served`` = parser + tool-call
    mode + max_model_len + max_num_seqs + concurrency
    (contracts.SERVED_IDENTITY_FIELDS) — never pid / port / paths / KV
    capacity."""
    ep = srun._standup_endpoint(world.table_path)
    ident = srun._served_identity(ep, concurrency=4)
    assert ident == {"reasoning_parser": None, "tool_calls": False, "max_model_len": STANDUP_WINDOW,
                     "max_num_seqs": STANDUP_SEQS, "concurrency": 4,
                     # PW18: the proxy's entry (the record's) + the lens reduction
                     # mode (absent on the record = per_row, E25)
                     "agent_loop_entry": True, "reduction_mode": "per_row"}
    assert tuple(ident) == tc.SERVED_IDENTITY_FIELDS
    assert srun._served_identity({**ep, tc.AGENT_LOOP_ENTRY_KEY: False},
                                 concurrency=4)["agent_loop_entry"] is False
    assert srun._served_identity({**ep, tc.PROXY_IDENTITY_REDUCTION_MODE: "stacked"},
                                 concurrency=4)["reduction_mode"] == "stacked"
    assert srun._served_identity({**ep, tc.PROXY_IDENTITY_REDUCTION_MODE: "per_row"},
                                 concurrency=4)["reduction_mode"] == "per_row"
    with pytest.raises(ValueError, match="not one of"):
        srun._served_identity({**ep, tc.PROXY_IDENTITY_REDUCTION_MODE: "grouped"}, concurrency=4)
    assert srun._served_identity({**ep, "geometry": standup_geometry(max_num_seqs=STANDUP_SEQS + 4)},
                                 concurrency=4)["max_num_seqs"] == STANDUP_SEQS + 4
    assert srun._served_identity({**ep, "geometry": standup_geometry(max_model_len=8192)},
                                 concurrency=4)["max_model_len"] == 8192
    assert srun._served_identity(ep, concurrency="8")["concurrency"] == 8    # argparse hands an int; coerced anyway
    for never in ("pid", "proxy_url", "routing_table", "node", "gpu_mem_util", "eval"):
        assert never not in ident


def test_live_proxy_identity_must_agree_with_the_record(world):
    """PW18 (N2): after the client loads, the LIVE proxy's identity block is
    held to the record the served identity was read from — the agent-loop
    entry must be equal, a reduction mode the block states must equal the
    identity's, a block-less proxy (pre-landing, or not the petri proxy) is
    refused, and a stated mode the record does not carry is accepted only
    when it equals the identity's (per_row)."""
    ep = srun._standup_endpoint(world.table_path)
    served = srun._served_identity(ep, concurrency=4)
    client = SimpleNamespace(proxy_identity=dict(LIVE_IDENTITY))
    assert srun._check_live_proxy_identity(client, ep, served) == LIVE_IDENTITY
    with pytest.raises(srun.EndpointBindingMismatch, match="no 'petri_proxy' identity block"):
        srun._check_live_proxy_identity(SimpleNamespace(proxy_identity=None), ep, served)
    with pytest.raises(srun.EndpointBindingMismatch, match="no 'petri_proxy' identity block"):
        srun._check_live_proxy_identity(SimpleNamespace(), ep, served)
    off = SimpleNamespace(proxy_identity={**LIVE_IDENTITY, tc.PROXY_IDENTITY_AGENT_LOOP_ENTRY: False})
    with pytest.raises(srun.EndpointBindingMismatch,
                       match="petri_agent_loop_entry=False but the standup record declares "
                             "agent_loop_entry=True"):
        srun._check_live_proxy_identity(off, ep, served)
    # the reverse disagreement (record off, proxy on) refuses the same way
    ep_off = {**ep, tc.AGENT_LOOP_ENTRY_KEY: False}
    with pytest.raises(srun.EndpointBindingMismatch, match="agent_loop_entry=False"):
        srun._check_live_proxy_identity(client, ep_off, srun._served_identity(ep_off, concurrency=4))
    # E25: a block stating the mode must agree with the record / identity
    stacked_live = SimpleNamespace(proxy_identity={**LIVE_IDENTITY,
                                                   tc.PROXY_IDENTITY_REDUCTION_MODE: "stacked"})
    with pytest.raises(srun.EndpointBindingMismatch, match="stated inconsistently|lens_reduction_mode"):
        srun._check_live_proxy_identity(stacked_live, ep, served)         # record absent (per_row) vs live stacked
    ep_st = {**ep, tc.PROXY_IDENTITY_REDUCTION_MODE: "stacked"}
    srun._check_live_proxy_identity(stacked_live, ep_st, srun._served_identity(ep_st, concurrency=4))
    per_row_live = SimpleNamespace(proxy_identity={**LIVE_IDENTITY,
                                                   tc.PROXY_IDENTITY_REDUCTION_MODE: "per_row"})
    srun._check_live_proxy_identity(per_row_live, ep, served)             # stated = the default: fine
    with pytest.raises(srun.EndpointBindingMismatch, match="stated inconsistently"):
        srun._check_live_proxy_identity(per_row_live, ep_st, srun._served_identity(ep_st, concurrency=4))
    bad = SimpleNamespace(proxy_identity={**LIVE_IDENTITY, tc.PROXY_IDENTITY_REDUCTION_MODE: "grouped"})
    with pytest.raises(srun.EndpointBindingMismatch, match="not one of"):
        srun._check_live_proxy_identity(bad, ep, served)


def test_the_driver_refuses_when_the_live_proxy_disagrees_with_the_record(world, monkeypatch):
    """Through ``main``: the record says the entry is ON (the identity hashed
    it) but the proxy answering says OFF — refused after the client loads,
    before any positive control or generation (the Inspect runner is never
    reached)."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    monkeypatch.setattr(srun, "_make_client",
                        lambda url, rt: SimpleNamespace(
                            chat_completions=world.chat.chat_completions,
                            proxy_identity={**LIVE_IDENTITY, tc.PROXY_IDENTITY_AGENT_LOOP_ENTRY: False}))

    async def never(**kw):
        raise AssertionError("run_inspect_tasks must not be reached")

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", never)
    argv = [a if a != LEGACY_GEN_CONTROLS else "gsm8k" for a in world.argv]
    with pytest.raises(srun.EndpointBindingMismatch, match="does not vouch for the proxy answering"):
        srun.main(argv + [tc.FLAG_THINKING, tc.THINKING_OFF])
    assert world.posted == [] and world.chat.calls == []          # no control, no generation


# --------------------------------------------------------------------------- #
# E10 N2: the standup record is BOUND to the proxy / table / ledger given       #
# --------------------------------------------------------------------------- #


def test_bind_endpoint_binds_the_record_to_the_proxy_and_table_given(world):
    ep = srun._standup_endpoint(world.table_path)
    # a root-form --proxy-url canonicalizes to the record's /v1 form
    b = srun._bind_endpoint(ep, proxy_url="http://127.0.0.1:1", routing_table=world.table_path)
    assert b == {"proxy_url": PROXY_URL, "routing_table": str(world.table_path.resolve()),
                 "ledger_url": None, "ledger_upstream": None, "ledger_endpoint": None}
    assert tuple(b) == srun.BINDING_FIELDS
    # another proxy (a second live standup, a stale record): refused, both named
    with pytest.raises(srun.EndpointBindingMismatch, match="proxy_url 'http://127.0.0.1:1/v1'") as ei:
        srun._bind_endpoint(ep, proxy_url="http://127.0.0.1:9/v1", routing_table=world.table_path)
    assert "'http://127.0.0.1:9/v1'" in str(ei.value)
    assert issubclass(srun.EndpointBindingMismatch, SystemExit)
    # another table: refused, both resolved paths named
    other = world.tmp_path / "elsewhere" / "routing_table.json"
    other.parent.mkdir()
    other.write_text("{}")
    with pytest.raises(srun.EndpointBindingMismatch, match="routing_table") as ei:
        srun._bind_endpoint(ep, proxy_url=PROXY_URL, routing_table=other)
    assert str(other.resolve()) in str(ei.value) and str(world.table_path.resolve()) in str(ei.value)
    # a record that names no proxy vouches for nothing
    with pytest.raises(srun.EndpointBindingMismatch, match="no proxy_url"):
        srun._bind_endpoint({**ep, "proxy_url": ""}, proxy_url=PROXY_URL, routing_table=world.table_path)
    with pytest.raises(srun.EndpointBindingMismatch, match="no routing_table"):
        srun._bind_endpoint({**ep, "routing_table": None}, proxy_url=PROXY_URL,
                            routing_table=world.table_path)


def test_bind_endpoint_with_a_ledger_reads_its_sibling_record_and_binds_the_upstream(world):
    ep = srun._standup_endpoint(world.table_path)
    rec_path = world.table_path.parent / tc.LEDGER_ENDPOINT_FILENAME
    b = srun._bind_endpoint(ep, proxy_url=PROXY_URL, routing_table=world.table_path,
                            ledger_url="http://127.0.0.1:2")          # root form canonicalizes
    assert (b["ledger_url"], b["ledger_upstream"], b["ledger_endpoint"]) == (
        LEDGER_URL, PROXY_URL, str(rec_path))
    # the ledger forwards to ANOTHER proxy: refused, both named
    write_ledger_endpoint(world.table_path, upstream="http://127.0.0.1:7/v1")
    with pytest.raises(srun.EndpointBindingMismatch, match="upstream 'http://127.0.0.1:7/v1'") as ei:
        srun._bind_endpoint(ep, proxy_url=PROXY_URL, routing_table=world.table_path, ledger_url=LEDGER_URL)
    assert f"proxy is '{PROXY_URL}'" in str(ei.value)
    # another ledger's record beside the table
    write_ledger_endpoint(world.table_path, ledger_url="http://127.0.0.1:3/v1")
    with pytest.raises(srun.EndpointBindingMismatch, match="ledger_url 'http://127.0.0.1:3/v1'"):
        srun._bind_endpoint(ep, proxy_url=PROXY_URL, routing_table=world.table_path, ledger_url=LEDGER_URL)
    # a record that does not name its upstream
    write_ledger_endpoint(world.table_path, upstream="")
    with pytest.raises(srun.EndpointBindingMismatch, match="lacks 'upstream'"):
        srun._bind_endpoint(ep, proxy_url=PROXY_URL, routing_table=world.table_path, ledger_url=LEDGER_URL)
    # no record at all: refused, naming the path and the remedy
    rec_path.unlink()
    with pytest.raises(srun.EndpointBindingMismatch, match="no ledger record at") as ei:
        srun._bind_endpoint(ep, proxy_url=PROXY_URL, routing_table=world.table_path, ledger_url=LEDGER_URL)
    assert str(rec_path) in str(ei.value) and "--endpoint-json" in str(ei.value)
    # without a ledger nothing is read: the missing record is not a refusal
    b = srun._bind_endpoint(ep, proxy_url=PROXY_URL, routing_table=world.table_path)
    assert b["ledger_url"] is None and b["ledger_endpoint"] is None


def test_ledger_endpoint_filename_is_the_one_the_launcher_writes():
    """The launcher starts the ledger with ``--endpoint-json <standup dir>/
    ledger_endpoint.json`` — the standup dir is where routing_table.json and
    endpoint.json live, so the driver finds the record as their sibling. The
    literal in the launcher (another lane's file) is pinned to the contract
    name, and the ledger's record names its upstream (what the binding reads)."""
    launcher = (REPO / "scripts" / "modal" / "tinycap_served.py").read_text()
    assert f'srv / "{tc.LEDGER_ENDPOINT_FILENAME}"' in launcher
    assert 'ep = srv / "endpoint.json"' in launcher                     # the same dir as the standup record
    ledger_src = (REPO / "src" / "sisteer" / "capeval" / "serving" / "usage_ledger.py").read_text()
    assert '"upstream": ledger.upstream_base_url' in ledger_src
    assert '"ledger_url": url' in ledger_src


def test_generate_help_renders(capsys):
    """``generate --help`` must not crash on a bare ``%`` in a help string
    (argparse formats them)."""
    with pytest.raises(SystemExit) as ei:
        srun.main(["generate", "--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "--vllm-log" in out and "~0% hit rate" in out and "--ledger-url" in out


def test_proxy_posters_forward_the_ledger_headers(monkeypatch):
    """The driver's aiohttp posters pass a leg's two ledger headers (P27)
    through to the session on chat AND on the shared endpoint poster's
    tokenize path; the legacy legs pass none."""
    import aiohttp

    seen = []

    class _Resp:
        status = 200

        async def text(self):
            return json.dumps({"ok": True})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, *, json=None, headers=None):
            seen.append((url, headers))
            return _Resp()

    monkeypatch.setattr(aiohttp, "ClientSession", _Session)
    hdrs = {tc.HEADER_TASK: "inspect/math", tc.HEADER_SAMPLE: "3/0"}

    async def run():
        async with srun._proxy_poster("http://127.0.0.1:1") as post, \
                srun._proxy_endpoint("http://127.0.0.1:1", "tokenize") as tokenize:
            assert await post({"model": "m"}, headers=hdrs) == {"ok": True}
            assert await tokenize({"model": "m"}, headers=hdrs) == {"ok": True}
            assert await post({"model": "m"}) == {"ok": True}

    asyncio.run(run())
    assert seen == [("http://127.0.0.1:1/v1/chat/completions", hdrs),
                    ("http://127.0.0.1:1/v1/tokenize", hdrs),
                    ("http://127.0.0.1:1/v1/chat/completions", None)]


def test_served_reasoning_parser_guard_refuses_thinking_on_generation_without_a_parser():
    none = {"reasoning_parser": None, "tool_calls": False}
    gemma = {"reasoning_parser": "gemma4", "tool_calls": False}
    # thinking-OFF never needs a parser; every tinycap atom generates, so
    # thinking-ON always does
    srun._check_served_reasoning_parser(["humaneval_plus", "math500"], "off", none)
    srun._check_served_reasoning_parser(["humaneval_plus", "math500"], "on", gemma)
    # thinking-ON with a GENERATION control and no served parser: the
    # scratchpad would be scored as the answer (PB12) — refused
    with pytest.raises(SystemExit, match="NO reasoning parser"):
        srun._check_served_reasoning_parser(["humaneval_plus", "math500"], "on", none)
    with pytest.raises(SystemExit, match=r"\['humaneval_plus'\]"):
        srun._check_served_reasoning_parser(["humaneval_plus"], "on", none)
    # R4: thinking-ON with a parser served => it is a pinned family parser
    with pytest.raises(SystemExit, match="not a pinned family parser"):
        srun._check_served_reasoning_parser(
            ["math500"], "on", {"reasoning_parser": "deepseek_r1", "tool_calls": False})
    with pytest.raises(SystemExit, match="not a pinned family parser"):
        srun._check_served_reasoning_parser(
            ["math500"], "on", {"reasoning_parser": "mystery", "tool_calls": False})
    # R4: thinking-OFF => no parser served, whatever the controls
    with pytest.raises(SystemExit, match="thinking-OFF but the standup composed"):
        srun._check_served_reasoning_parser(["math500"], "off", gemma)
    with pytest.raises(SystemExit, match="thinking-OFF but the standup composed"):
        srun._check_served_reasoning_parser(["math500", "humaneval_plus"], "off", gemma)


# --------------------------------------------------------------------------- #
# PW3: the bespoke served GSM8K leg is retired; `gsm8k` is the STANDARD       #
# Inspect gsm8k (a tinycap atom) and its slot payload names the instrument    #
# --------------------------------------------------------------------------- #


def test_gsm8k_is_the_standard_inspect_control_and_the_bespoke_leg_refuses():
    """U2-impl (2)/(1): `gsm8k` left the legacy SERVED_CONTROLS and is the
    tinycap control contracts.CONTROL_GSM8K (dispatched to inspect_served,
    row task inspect/gsm8k); the retired leg's spellings refuse with the
    pointer; the inline scorer is gone; mmlu / xstest / or_bench_hard_1k are
    exactly as they were."""
    assert srun.SERVED_CONTROLS == ("mmlu", "xstest", "or_bench_hard_1k")
    assert srun.parse_controls("gsm8k") == ["gsm8k"]
    assert srun.tinycap_atoms(["gsm8k", "mmlu"]) == ["gsm8k"]
    assert tc.control_family("gsm8k") in srun.TINYCAP_DISPATCHED
    # U4(a) / PW17: the standard is the PAIRED set; the full set is the in-depth `gsm8k_full`
    assert tc.TASKS_OF_CONTROL["gsm8k"] == (tc.GSM8K_STANDARD_TASK,) == ("inspect/gsm8k_paired200",)
    assert tc.TASKS_OF_CONTROL["gsm8k_full"] == (tc.GSM8K_FULL_TASK,) == ("inspect/gsm8k",)
    assert srun.parse_controls("inspect_cmp:gsm8k,gsm8k_full") == ["gsm8k", "gsm8k_full"]
    assert not hasattr(srun, "_run_gsm8k_served") and not hasattr(srun, "_gsm8k_rows")
    for tok in srun.RETIRED_GSM8K_TOKENS:
        with pytest.raises(SystemExit) as ei:
            srun.parse_controls(f"mmlu,{tok}")
        msg = str(ei.value)
        assert "RETIRED" in msg and "inspect/gsm8k" in msg and "--controls gsm8k" in msg
        assert tc.GSM8K_LEGACY_ENV in msg
    assert "s2parity/gsm8k_0shot" in srun.RETIRED_GSM8K_TOKENS
    # the frozen store-identity list still names gsm8k (hash stability), the
    # servable set does not
    assert srun.RESOLVED_CONTROLS_IDENTITY == ("gsm8k", "mmlu", "or_bench_hard_1k", "xstest")
    assert set(srun.SERVED_CONTROLS) < set(srun.RESOLVED_CONTROLS_IDENTITY)


def test_gsm8k_without_the_thinking_arm_refuses_naming_the_standard(world):
    """`--controls gsm8k` is an Inspect control now, so the thinking arm is
    required (P33) — and the refusal says why gsm8k needs it."""
    argv = [a if a != LEGACY_GEN_CONTROLS else "gsm8k" for a in world.argv]
    with pytest.raises(SystemExit) as ei:
        srun.main(argv)
    msg = str(ei.value)
    assert tc.FLAG_THINKING in msg and "STANDARD GSM8K" in msg and "inspect/gsm8k" in msg
    assert world.posted == []  # nothing was generated


def test_gsm8k_control_dispatches_to_inspect_and_fills_the_slot(world, monkeypatch):
    """U2-impl (1) end to end, offline (U4(a) / PW17: the paired set):
    `--controls gsm8k --thinking off` reaches inspect_served.run_inspect_tasks
    with the `gsm8k` atom (never the old inline leg), and a COMPLETE run's
    controls_summary.json carries the `gsm8k` slot payload that NAMES its
    instrument (task inspect/gsm8k_paired200, item_set paired200, n 200)
    beside the standard task's own entry."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    calls = []

    async def fake_run_inspect_tasks(*, atoms, route, routes, system, store, template,
                                     thinking, base_url, limits=None, **kw):
        calls.append({"atoms": list(atoms), "thinking": thinking, "system": system,
                      "base_url": base_url, "limits": dict(limits or {})})
        task = tc.GSM8K_STANDARD_TASK
        n = tc.GSM8K_PAIRED_N
        summary = {"acc": 0.8, "n": n, "n_requests": n,
                   "n_rejected": 0, "samples_per_item": 1, "task": task,
                   "item_set": tc.GSM8K_ITEM_SET_PAIRED, "framework": tc.INSPECT_TASK_VERSION,
                   "engine": "vllm", "route": route.virtual_model, "thinking": thinking,
                   "render": tc.THINKING_RENDER_NATIVE, "wall_clock_s": 1.0,
                   "max_tokens": 512, "temperature": 0.0}
        completion.finish_task(store.cell_dir, task=task, n_items=n,
                               n_expected_full=n, limits=limits,
                               n_rows=n, n_requests=n,
                               n_rejected=0, wall_clock_s=1.0, summary=summary)
        return {task: summary}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run_inspect_tasks)
    argv = [a if a != LEGACY_GEN_CONTROLS else "gsm8k" for a in world.argv]
    srun.main(argv + [tc.FLAG_THINKING, tc.THINKING_OFF])

    assert calls == [{"atoms": ["gsm8k"], "thinking": "off", "system": None,
                      "base_url": "http://127.0.0.1:1/v1", "limits": {}}]
    assert world.posted == []  # no legacy leg posted anything
    cell_dir = _cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    assert manifest["controls"] == ["gsm8k"]
    assert manifest["resolved_config"]["tinycap"]["tasks"] == ["gsm8k"]
    assert manifest["resolved_config"]["tinycap"]["thinking"]["mode"] == "off"
    assert manifest["frameworks"]["inspect_evals"] == tc.INSPECT_EVALS_PIN
    # PW5-2: the frameworks block is hashed identity (one source, three copies)
    assert manifest["resolved_config"]["tinycap"]["frameworks"] == manifest["frameworks"]
    assert manifest["tinycap"]["frameworks"] == manifest["frameworks"]
    assert manifest["frameworks"]["client_lock_sha256"] == srun._client_lock_sha256()
    # PW5-5: the tree that produced the store (this checkout's HEAD; no TREE_SHA stamp here)
    assert manifest["producer"] == srun._producer_block(REPO)
    assert set(manifest["producer"]) == set(tc.PRODUCER_BLOCK_FIELDS)
    assert not (cell_dir / "rows.jsonl").exists()  # the fake filed no rows; the driver none
    assert completion.is_done(cell_dir, tc.GSM8K_STANDARD_TASK)
    gen_stats = json.loads((cell_dir / "gen_stats.json").read_text())
    assert gen_stats[tc.GSM8K_STANDARD_TASK]["item_set"] == "paired200"
    # the ONE controls_summary.json writer merged the standard task AND its slot
    summary_path = world.out_root / TT_CELL / "controls_summary.json"
    doc = json.loads(summary_path.read_text())
    controls = doc["controls"]
    assert set(controls) == {tc.GSM8K_STANDARD_TASK, tc.GSM8K_SLOT} == {"inspect/gsm8k_paired200", "gsm8k"}
    slot = controls["gsm8k"]
    assert set(slot) == set(tc.GSM8K_SLOT_PAYLOAD_FIELDS)
    assert slot == {"task": "inspect/gsm8k_paired200", "item_set": "paired200", "acc": 0.8,
                    "n": 200, "framework": tc.INSPECT_TASK_VERSION, "engine": "vllm",
                    "thinking": "off", "max_tokens": 512, "temperature": 0.0}
    assert "extraction_failures" not in slot  # the retired payload's shape
    # the slot payload loads through the provenance-validated consumer the
    # two-SI aggregate uses (the object on disk re-hashed, r1 #4)
    from sisteer.eval.cell_controls import load_controls_summaries
    from sisteer.eval.defense_cell import by_id, registry_from_cfg

    cfg = yaml.safe_load(world.cfg_path.read_text())
    cell = by_id(registry_from_cfg(cfg, repo_root=REPO))[TT_CELL]
    filled = load_controls_summaries(world.out_root, [cell], cfg)
    assert filled[TT_CELL]["gsm8k"]["task"] == "inspect/gsm8k_paired200"
    assert filled[TT_CELL]["gsm8k"]["source"] == "cell_controls"
    # PW5-1: the standard payload passes the slot's acceptance rule — it fills
    # `gsm8k` itself, and nothing lands under the legacy key
    assert tc.is_standard_gsm8k_slot_payload(slot)
    assert tc.GSM8K_LEGACY_SLOT not in filled[TT_CELL]


# --------------------------------------------------------------------------- #
# PW20b-D (U5-3 / U5-4 / U5-5): --thinking per-task under ONE standup, the     #
# per-task map + retry cap as identity, and the run's quality roll-up         #
# --------------------------------------------------------------------------- #


def _percell_argv(world, controls: str, *, out_root=None):
    """The world's argv with a chosen control list (and, under ``out_root``,
    another store root — a second run that must NOT share the first's cell
    dir). Fixtures plan a faked Inspect leg ONLY: planning the
    fitted default would pull in the EvalPlus archive check, which fires
    before the standup and fails on any machine (EVALPLUS_ARCHIVES_MISSING)."""
    swap = {LEGACY_GEN_CONTROLS: controls}
    if out_root is not None:
        swap[str(world.out_root)] = str(out_root)
    return [swap.get(a, a) for a in world.argv]


def test_the_per_task_map_is_resolved_at_plan_time_and_an_unmapped_task_refuses(monkeypatch):
    """U5-4: `per-task` takes each row task's arm from contracts, and a task
    the policy does not map is refused BEFORE any standup is read or GPU
    minute spent — naming it, and naming the two ways out."""
    assert srun._resolve_thinking_by_task([tc.CONTROL_GSM8K, tc.CONTROL_HUMANEVAL_PLUS],
                                          tc.THINKING_PER_TASK) == {
        tc.GSM8K_STANDARD_TASK: "on", "evalplus/humaneval_plus": "off"}
    # every mode records the map — `on` / `off` give one arm to every task
    assert srun._resolve_thinking_by_task([tc.CONTROL_GSM8K], tc.THINKING_ON) == {
        tc.GSM8K_STANDARD_TASK: "on"}
    assert srun._resolve_thinking_by_task([tc.CONTROL_GSM8K], tc.THINKING_OFF) == {
        tc.GSM8K_STANDARD_TASK: "off"}
    holed = {k: v for k, v in tc.THINKING_BY_TASK.items() if k != tc.GSM8K_STANDARD_TASK}
    monkeypatch.setattr(tc, "THINKING_BY_TASK", holed)
    with pytest.raises(SystemExit) as ei:
        srun._resolve_thinking_by_task([tc.CONTROL_GSM8K], tc.THINKING_PER_TASK)
    msg = str(ei.value)
    assert tc.GSM8K_STANDARD_TASK in msg and "THINKING_BY_TASK" in msg
    assert f"{tc.FLAG_THINKING} on|off" in msg


def test_a_single_arm_leg_takes_the_arm_its_tasks_share(monkeypatch):
    """The EvalPlus generator takes ONE `thinking` argument, so its tasks'
    arms must agree — they do under U5's policy. A map that ever split them is
    refused rather than served at one arm."""
    by_task = tc.run_thinking_by_task(["humaneval_plus", "mbpp_plus"], tc.THINKING_PER_TASK)
    assert srun._leg_arm(by_task, tc.EVALPLUS_TASKS, "EvalPlus", tc.THINKING_PER_TASK) == "off"
    # the render label: per-task keeps the OFF arm (hence `native`), a whole arm is its own
    assert srun._render_arm(tc.THINKING_PER_TASK, by_task) == "off"
    assert srun._render_arm(tc.THINKING_ON, by_task) == "on"
    assert srun._render_label(srun._render_arm(tc.THINKING_PER_TASK, by_task)) == tc.THINKING_RENDER_NATIVE
    assert srun._render_label(tc.THINKING_ON) == tc.THINKING_RENDER_FLAG_ONLY
    split = {"evalplus/humaneval_plus": "off", "evalplus/mbpp_plus": "on"}
    with pytest.raises(SystemExit, match="takes ONE thinking arm"):
        srun._leg_arm(split, tc.EVALPLUS_TASKS, "EvalPlus", tc.THINKING_PER_TASK)


def test_the_parser_guard_reads_the_atoms_that_are_ON_under_a_per_task_map():
    """One standup serves a MIXED battery, so the parser rule binds on the
    ON atoms: a run with any ON generation task needs the parser, a run whose
    every arm is OFF must have none, and a mixed run's OFF tasks ride the
    parser its ON tasks require (that is the point of one standup)."""
    none = {"reasoning_parser": None, "tool_calls": False}
    gemma = {"reasoning_parser": "gemma4", "tool_calls": False}
    mixed = tc.run_thinking_by_task(
        [tc.CONTROL_GSM8K, "inspect_cmp:or_bench_hard_1k", "humaneval_plus"], tc.THINKING_PER_TASK)
    assert mixed[tc.GSM8K_STANDARD_TASK] == "on" and mixed["inspect/or_bench_hard_1k"] == "off"
    srun._check_served_reasoning_parser(
        [tc.CONTROL_GSM8K, "inspect_cmp:or_bench_hard_1k", "humaneval_plus"], tc.THINKING_PER_TASK,
        gemma, by_task=mixed)                                   # allowed: gsm8k is ON
    with pytest.raises(SystemExit, match="NO reasoning parser"):
        srun._check_served_reasoning_parser(
            [tc.CONTROL_GSM8K, "inspect_cmp:or_bench_hard_1k"], tc.THINKING_PER_TASK, none, by_task=mixed)
    all_off = tc.run_thinking_by_task(
        ["inspect_cmp:or_bench_hard_1k", "humaneval_plus"], tc.THINKING_PER_TASK)
    srun._check_served_reasoning_parser(
        ["inspect_cmp:or_bench_hard_1k", "humaneval_plus"], tc.THINKING_PER_TASK, none, by_task=all_off)
    with pytest.raises(SystemExit, match="would serve nobody"):
        srun._check_served_reasoning_parser(
            ["inspect_cmp:or_bench_hard_1k", "humaneval_plus"], tc.THINKING_PER_TASK,
            gemma, by_task=all_off)
    # the whole-configuration rules are untouched
    with pytest.raises(SystemExit, match="thinking-OFF but the standup composed"):
        srun._check_served_reasoning_parser(["humaneval_plus"], tc.THINKING_OFF, gemma)


def test_the_thinking_map_is_a_resume_key_and_a_manifest_without_it_is_refused(tmp_path):
    """One arm per task (U5): a store filled under one map never resumes
    under another. The map is hashed, so a different map normally keys
    elsewhere — this is the check the 8-hex prefix cannot make."""
    store = SimpleNamespace(manifest_path=tmp_path / "manifest.json")
    by_task = {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}
    srun._check_manifest_thinking_map(store, thinking="per-task", by_task=by_task)  # absent: fresh
    store.manifest_path.write_text(json.dumps({
        tc.MANIFEST_KEY_THINKING: {"mode": "per-task", "by_task": by_task}}))
    srun._check_manifest_thinking_map(store, thinking="per-task", by_task=by_task)  # equal: ok
    with pytest.raises(srun.ServingIdentityMismatch, match="inspect/or_bench_hard_1k: the manifest records arm 'off'"):
        srun._check_manifest_thinking_map(
            store, thinking="per-task",
            by_task={**by_task, "inspect/or_bench_hard_1k": "on"})
    with pytest.raises(srun.ServingIdentityMismatch, match="mode: the manifest records 'per-task'"):
        srun._check_manifest_thinking_map(store, thinking="on", by_task=by_task)
    # a mode change AND an arm change: BOTH are named (a `[A] if c else []+B`
    # expression would bind the ternary loosest and report only the mode)
    with pytest.raises(srun.ServingIdentityMismatch) as ei:
        srun._check_manifest_thinking_map(
            store, thinking="on", by_task={**by_task, "inspect/or_bench_hard_1k": "on"})
    assert "mode: the manifest records" in str(ei.value)
    assert "inspect/or_bench_hard_1k: the manifest records arm 'off'" in str(ei.value)
    with pytest.raises(srun.ServingIdentityMismatch, match="which this run does not"):
        srun._check_manifest_thinking_map(
            store, thinking="per-task", by_task={tc.GSM8K_STANDARD_TASK: "on"})
    # a manifest from before the map cannot vouch for its rows' arms
    store.manifest_path.write_text(json.dumps({
        tc.MANIFEST_KEY_THINKING: {"enable_thinking": True, "render": "native",
                                   "reasoning_parser": "gemma4"}}))
    with pytest.raises(srun.ServingIdentityMismatch, match="records no per-task thinking map"):
        srun._check_manifest_thinking_map(store, thinking="per-task", by_task=by_task)


def test_the_per_cell_quality_rollup_pools_counts_and_keeps_null_apart_from_zero():
    """U5-5: pooled from the COUNTS (tasks have very different item counts,
    so a mean of rates is not the run's rate); a task that does not record a
    measure is named rather than counted as 0; legs that record neither (the
    EvalPlus and legacy summaries) are simply not in it."""
    q = srun._cell_quality({
        "inspect/gsm8k_paired200": {"n_unparseable": 2, "n_answered": 200,
                                    "n_at_cap": 1, "n_stop_reason": 200,
                                    "unparseable_rate": 0.01, "at_cap_rate": 0.005},
        "inspect/math": {"n_unparseable": None, "n_answered": 0,
                         "n_at_cap": 60, "n_stop_reason": 2000,
                         "unparseable_rate": None, "at_cap_rate": 0.03},
        "evalplus/humaneval_plus": {"acc": 0.5, "n": 100},    # records neither
    })
    assert set(q) == set(tc.QUALITY_BLOCK_FIELDS)
    assert q["unparseable"] == {"n": 2, "denominator": 200, "rate": 0.01,
                                "tasks_not_recorded": ["inspect/math"]}
    assert q["at_cap"] == {"n": 61, "denominator": 2200, "rate": 61 / 2200,
                           "tasks_not_recorded": []}
    assert set(q["by_task"]) == {"inspect/gsm8k_paired200", "inspect/math"}
    # nothing recorded at all: rate null, never 0
    empty = srun._cell_quality({"evalplus/humaneval_plus": {"acc": 0.5}})
    assert empty["unparseable"]["rate"] is None and empty["unparseable"]["n"] is None
    assert empty["at_cap"]["denominator"] == 0 and empty["by_task"] == {}


def test_per_task_run_end_to_end_records_the_map_the_retry_cap_and_the_quality(
        world, monkeypatch, capsys):
    """The whole contract in one pass, offline: `--thinking per-task` on a
    cell whose battery mixes arms serves ONE standup, hands the Inspect leg
    the map and the retry cap, records both as identity, mirrors the map in
    the manifest's thinking block (whose `enable_thinking` is NULL — there is
    no single bit), and writes + prints the per-cell quality roll-up."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    # the ON tasks of the battery need the parser their standup composed
    write_standup_endpoint(world.table_path, reasoning_parser="gemma4")
    calls = []

    async def fake_run_inspect_tasks(*, atoms, store, thinking, limits=None, **kw):
        calls.append({"atoms": list(atoms), "thinking": thinking,
                      "by_task": kw.get("thinking_by_task"),
                      "retry": kw.get("math500_retry_max_tokens"),
                      "window": kw.get("max_model_len")})
        task, n = tc.GSM8K_STANDARD_TASK, tc.GSM8K_PAIRED_N
        summary = {"acc": 0.8, "n": n, "n_requests": n, "n_rejected": 0,
                   "samples_per_item": 1, "task": task,
                   "item_set": tc.GSM8K_ITEM_SET_PAIRED, "framework": tc.INSPECT_TASK_VERSION,
                   "engine": "vllm", "route": "petri__x", "thinking": "on",
                   "render": tc.THINKING_RENDER_NATIVE, "wall_clock_s": 1.0,
                   "max_tokens": 2048, "temperature": 0.0,
                   "unparseable_rate": 0.02, "n_unparseable": 4, "n_answered": 200,
                   "at_cap_rate": 0.01, "n_at_cap": 2, "n_stop_reason": 200}
        completion.finish_task(store.cell_dir, task=task, n_items=n, n_expected_full=n,
                               limits=limits, n_rows=n, n_requests=n, n_rejected=0,
                               wall_clock_s=1.0, summary=summary)
        return {task: summary}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run_inspect_tasks)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K)
              + [tc.FLAG_THINKING, tc.THINKING_PER_TASK])

    # the leg is handed the map, the retry cap (the driver's default — 0, the
    # retry is retired since the one-pass 8192 instrument) and the window it
    # must fit
    assert calls == [{"atoms": [tc.CONTROL_GSM8K], "thinking": "per-task",
                      "by_task": {tc.GSM8K_STANDARD_TASK: "on"},
                      "retry": tc.MATH500_RETRY_MAX_TOKENS, "window": STANDUP_WINDOW}]
    assert calls[0]["retry"] == 0
    cell_dir = _cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    ident = manifest["resolved_config"]["tinycap"]
    assert ident["thinking"]["mode"] == "per-task"
    assert ident[tc.IDENTITY_KEY_THINKING_BY_TASK] == {tc.GSM8K_STANDARD_TASK: "on"}
    # a run that files no MATH-500 row stamps the LEGACY literal, not the
    # driver's default — its key must not move when MATH's instrument does
    assert ident[tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] == tc.MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY == 8192
    assert ident[tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] != tc.MATH500_RETRY_MAX_TOKENS
    assert set(ident) == set(tc.TINYCAP_IDENTITY_FIELDS)
    # the manifest's thinking block mirrors the map and refuses a single bit
    assert manifest["thinking"]["mode"] == "per-task"
    assert manifest["thinking"]["by_task"] == {tc.GSM8K_STANDARD_TASK: "on"}
    assert manifest["thinking"]["enable_thinking"] is None
    assert set(manifest["thinking"]) == set(tc.THINKING_BLOCK_FIELDS)
    # U5-5: the roll-up is in the manifest AND on stdout, and is NOT identity
    q = manifest[tc.MANIFEST_KEY_QUALITY]
    assert q["unparseable"] == {"n": 4, "denominator": 200, "rate": 0.02,
                                "tasks_not_recorded": []}
    assert q["at_cap"]["rate"] == 0.01
    assert tc.MANIFEST_KEY_QUALITY not in ident
    out = capsys.readouterr().out
    assert "quality: unparseable (scorer found no answer): 4/200 = 2.00%" in out
    assert "quality: at the token cap: 2/200 = 1.00%" in out
    # the identity re-hashes to the recorded key (the added fields are hashed)
    from sisteer.capeval import store as cs
    assert cs.harness_config_hash(manifest["resolved_config"]) == manifest["harness_config_hash"]


def test_a_whole_configuration_run_still_records_the_map_and_a_single_bit(world, monkeypatch):
    """`on` / `off` are unchanged as arms — but they record the map too, so
    ONE key answers "what arm was this task run under" for every store, and
    `enable_thinking` stays the single bit it always was."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)

    async def fake_run_inspect_tasks(*, atoms, store, thinking, limits=None, **kw):
        assert kw.get("thinking_by_task") is None      # no map outside per-task
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run_inspect_tasks)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    cell_dir = _cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    ident = manifest["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_THINKING_BY_TASK] == {tc.GSM8K_STANDARD_TASK: "off"}
    assert manifest["thinking"]["enable_thinking"] is False
    assert manifest["thinking"]["mode"] == "off"
    # nothing recorded a quality measure, so both read "not recorded", never 0
    assert manifest[tc.MANIFEST_KEY_QUALITY]["unparseable"]["rate"] is None


def test_the_driver_hashes_the_or_bench_item_set_and_records_its_ids():
    """U10 item 4. Three claims, pinned against the
    driver's source because the manifest-writing branch only runs on a real
    OR-Bench leg (the corpus is not staged in this venv; the RESOLUTION itself
    is unit-tested in tests/test_inspect_served.py, and the identity plumbing
    in tests/test_tinycap_{contracts,launcher}.py):

    1. the run's item set is derived from the CONTRACTS, purely — positions,
       never the corpus — so the launcher derives the same value it will hold
       the container to;
    2. that value is HASHED identity, and the SAME object is handed to the
       Inspect leg, so the rows that run are the rows the identity claims;
    3. the ids those positions resolve to are recorded in a manifest block of
       their own, unhashed and outside TINYCAP_MANIFEST_KEYS."""
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert ("or_bench_item_set = (tc.run_or_bench_item_set(atoms, full=bool(args.or_bench_full))"
            in src)
    assert f"tc.{'IDENTITY_KEY_OR_BENCH_ITEM_SET'}: or_bench_item_set," in src
    # handed to the Inspect leg (the kwarg is no longer last in the call —
    # E56 (3) added the grader cap after it — so the pin is the kwarg itself)
    assert "                    or_bench_item_set=or_bench_item_set,\n" in src
    assert "manifest[tc.MANIFEST_KEY_OR_BENCH_ITEM_SET] = {" in src
    assert '"item_ids": _isv.or_bench_item_ids(or_bench_item_set, repo_root=REPO),' in src
    # the flag exists on the driver and nowhere else: the battery default is
    # the subset, and the launcher never emits an override for it
    assert "tc.FLAG_OR_BENCH_FULL" in src          # by CONSTANT, never a retyped literal
    assert tc.FLAG_OR_BENCH_FULL == "--or-bench-full"
    assert tc.FLAG_OR_BENCH_FULL not in tc.TINYCAP_DRIVER_FLAGS
    assert tc.IDENTITY_KEY_OR_BENCH_ITEM_SET in tc.TINYCAP_IDENTITY_FIELDS
    assert tc.MANIFEST_KEY_OR_BENCH_ITEM_SET not in tc.TINYCAP_MANIFEST_KEYS
    # ...and the flag is a store_true whose DEFAULT is the subset: the battery
    # gets the frozen 200 without anyone typing anything
    assert ('g.add_argument(tc.FLAG_OR_BENCH_FULL, dest="or_bench_full",\n'
            '                   action="store_true", default=False,') in src


def test_mmlu_full_is_an_explicit_opt_in_with_its_own_nested_identity(world):
    """mmlu-full-1531. The default stays the s2parity
    200-item head with a byte-identical identity (every historical MMLU row is
    tied to it); ``--mmlu-full`` scores the WHOLE pinned pool as a DISTINCT
    instrument — a different store key, the item set hashed into
    ``resolved_config.mmlu``, the ids that ran recorded unhashed in the
    manifest, the width stated in gen_stats — and its first rows are the
    default's, item for item (nested widths, never two samples). The fixture's
    four-row pool stands in for the real 1,531; the real corpus is pinned by
    ``test_mmlu_full_width_is_the_pinned_pool_and_nests_the_default``."""
    from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU

    def acc_rows(cell_dir):
        return {r["item_id"]: r["value"] for r in _rows(cell_dir)
                if r["task"] == PARITY_TASK_MMLU and r["metric"] == "acc"
                and r["item_id"] != "__aggregate__"}

    def aggregate(cell_dir):
        agg = [r for r in _rows(cell_dir) if r["task"] == PARITY_TASK_MMLU
               and r["item_id"] == "__aggregate__"]
        assert len(agg) == 1
        return agg[0]

    # --- the DEFAULT: unchanged, and stated as the head ---------------------------
    srun.main(_mmlu_argv(world))
    cell_default = _cell_dir(world.out_root)
    man_default = json.loads((cell_default / "manifest.json").read_text())
    mm_default = man_default["resolved_config"]["mmlu"]
    assert mm_default == {"answer_seed": srun.MMLU_ANSWER_SEED,
                          "top_logprobs": srun.MMLU_TOP_LOGPROBS,
                          "response_prefill": True}          # byte-identical identity
    assert tc.IDENTITY_KEY_MMLU_ITEM_SET not in mm_default
    assert tc.MANIFEST_KEY_MMLU_ITEM_SET not in man_default
    assert set(acc_rows(cell_default)) == {r["item_id"] for r in MMLU_ROWS}
    assert json.loads(aggregate(cell_default)["extra_json"])["n"] == len(MMLU_ROWS)
    gs_default = json.loads((cell_default / "gen_stats.json").read_text())["mmlu"]
    assert gs_default["item_set"] == tc.mmlu_item_set() == {
        "item_set": tc.MMLU_ITEM_SET_SUBSET, "n": tc.PARITY_N, "seed": 0, "of": tc.MMLU_FULL_N}
    assert gs_default["n"] == len(MMLU_ROWS)
    n_posted_default = len(world.posted)

    # --- the FULL pool: the flag by CONSTANT, its own out root -----------------------
    out_full = world.tmp_path / "out_full"
    argv = [str(out_full) if a == str(world.out_root) else a for a in _mmlu_argv(world)]
    srun.main(argv + [tc.FLAG_MMLU_FULL])
    cell_full = _cell_dir(out_full)
    man_full = json.loads((cell_full / "manifest.json").read_text())
    mm_full = man_full["resolved_config"]["mmlu"]
    # a DIFFERENT store: the key's chash suffix, the hash it is derived from
    assert cell_full.name.rsplit("__", 1)[1] != cell_default.name.rsplit("__", 1)[1]
    assert man_full["harness_config_hash"] != man_default["harness_config_hash"]
    # ...differing ONLY by the item set inside the MMLU scoring-surface block
    assert mm_full[tc.IDENTITY_KEY_MMLU_ITEM_SET] == tc.mmlu_item_set(full=True) == {
        "item_set": tc.MMLU_ITEM_SET_FULL, "n": tc.MMLU_FULL_N, "seed": 0, "of": tc.MMLU_FULL_N}
    assert {k: v for k, v in mm_full.items() if k != tc.IDENTITY_KEY_MMLU_ITEM_SET} == mm_default
    # the unhashed manifest block: the spec plus the ids that ran, the default's ids its head
    block = man_full[tc.MANIFEST_KEY_MMLU_ITEM_SET]
    assert set(block) == set(tc.MMLU_ITEM_SET_BLOCK_FIELDS)   # the manifest is dumped with sorted keys
    assert {k: block[k] for k in tc.MMLU_ITEM_SET_FIELDS} == tc.mmlu_item_set(full=True)
    assert block["item_ids"] == [r["item_id"] for r in MMLU_ROWS_FULL]
    assert block["item_ids"][:len(MMLU_ROWS)] == [r["item_id"] for r in MMLU_ROWS]
    # every row of the pool filed, the head's verdicts unchanged, the aggregate over the pool
    acc = acc_rows(cell_full)
    assert acc == {"m0000-dddddddd": 1.0, "m0001-eeeeeeee": 0.0,
                   "m0002-ffffffff": 1.0, "m0003-a0a0a0a0": 0.0}
    assert {k: acc[k] for k in acc_rows(cell_default)} == acc_rows(cell_default)
    agg = aggregate(cell_full)
    assert agg["value"] == pytest.approx(0.5)
    assert json.loads(agg["extra_json"])["n"] == len(MMLU_ROWS_FULL)
    gs_full = json.loads((cell_full / "gen_stats.json").read_text())["mmlu"]
    assert gs_full["item_set"] == tc.mmlu_item_set(full=True) and gs_full["n"] == len(MMLU_ROWS_FULL)
    # the INSTRUMENT is otherwise untouched: one seeded letter-prefill request per item,
    # one greedy token, the same top-K read, the same assistant seed
    scoring = [r for r in world.posted[n_posted_default:] if r.get("continue_final_message")]
    assert len(scoring) == len(MMLU_ROWS_FULL)
    for req in scoring:
        assert req["messages"][-1]["content"] == srun.MMLU_ANSWER_SEED
        assert req["max_tokens"] == 1 and req["temperature"] == 0.0
        assert req["top_logprobs"] == srun.MMLU_TOP_LOGPROBS and req["logprobs"] is True


def test_mmlu_rows_seam_asks_the_head_by_default_and_the_pool_when_widened(monkeypatch):
    """The ONE width resolver: ``_mmlu_rows`` hands parity ``PARITY_N`` (200)
    unless widened, then ``None`` (the loader's whole pinned pool) — the same
    resolver (``contracts.mmlu_parity_n``) the identity stamp is built from,
    so the rows that run and the item set the store claims cannot disagree."""
    from sisteer.capeval.tiers import parity

    seen = []
    monkeypatch.setattr(parity, "mmlu_parity_rows",
                        lambda pins, n=parity.PARITY_N: seen.append(n) or [])
    assert srun._mmlu_rows({"datasets": {}}) == []
    assert srun._mmlu_rows({"datasets": {}}, full=True) == []
    assert seen == [tc.PARITY_N, None] == [200, None]
    assert parity.PARITY_N == tc.PARITY_N and parity.PARITY_SLICE_SEED == tc.PARITY_SLICE_SEED == 0


def test_the_driver_hashes_the_mmlu_item_set_only_when_widened():
    """The full-pool opt-in follows ``--or-bench-full``'s shape exactly (pinned
    against the source, like that flag's own test): the flag by constant, a
    store_true defaulting to the designed width, outside the launcher's
    always-emitted flag set; the width reaching the rows and the identity
    through ONE resolver; the identity key written INSIDE ``resolved["mmlu"]``
    under the flag only (every historical store's chash byte-identical); the
    ids recorded unhashed in a manifest block of their own."""
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "return parity.mmlu_parity_rows(pins, n=tc.mmlu_parity_n(full=full))" in src
    assert "_mmlu_rows(pins, full=bool(args.mmlu_full))" in src
    assert "mmlu_full=bool(args.mmlu_full))" in src                # into _resolved_config
    assert ('    if mmlu_full:\n'
            '        resolved["mmlu"][tc.IDENTITY_KEY_MMLU_ITEM_SET] = tc.mmlu_item_set(full=True)\n') in src
    assert "    if mmlu_rows is not None and args.mmlu_full:\n" in src
    assert "        manifest[tc.MANIFEST_KEY_MMLU_ITEM_SET] = {\n" in src
    assert 'gen_stats["mmlu"]["item_set"] = tc.mmlu_item_set(full=bool(args.mmlu_full))' in src
    assert "tc.FLAG_MMLU_FULL" in src                              # by CONSTANT, never a retyped literal
    assert tc.FLAG_MMLU_FULL == "--mmlu-full"
    assert tc.FLAG_MMLU_FULL not in tc.TINYCAP_DRIVER_FLAGS
    assert ('g.add_argument(tc.FLAG_MMLU_FULL, dest="mmlu_full",\n'
            '                   action="store_true", default=False,') in src
    # the seam is the contracts' (one source for the width and the seed)
    assert tc.mmlu_parity_n() == tc.PARITY_N == 200 and tc.mmlu_parity_n(full=True) is None
    assert tc.mmlu_item_set(full=True)["n"] == tc.MMLU_FULL_N == 1531


def test_mmlu_full_width_is_the_pinned_pool_and_nests_the_default():
    """On the REAL pinned corpus (skip-unless-cached, in an offline subprocess
    so no test ever reaches the hub): the default is 200 rows, the full pool
    1,531 (= ``contracts.MMLU_FULL_N``: the whole 'all' validation split at the
    pin — the test split is not pinned), the 200 are the 1,531's first rows
    item for item (ids AND content: the same seed-0 permutation cut at two
    widths), every id unique, and the two id lists hash to the digests measured
    2026-09-21 — so a change to the seed, the pin or the selection fails here
    rather than quietly re-identifying every MMLU row."""
    import os
    import subprocess

    pytest.importorskip("datasets")
    probe = r"""
import hashlib, importlib.util, json, sys
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/scripts"]
spec = importlib.util.spec_from_file_location("scr", sys.argv[1] + "/scripts/served_capability_run.py")
scr = importlib.util.module_from_spec(spec); sys.modules["scr"] = scr; spec.loader.exec_module(scr)
from sisteer.capeval import corpora
pins = corpora.load_pins()
try:
    head = scr._mmlu_rows(pins); full = scr._mmlu_rows(pins, full=True)
except Exception as e:  # noqa: BLE001 — an absent offline cache is a SKIP upstairs, never a guess
    print(json.dumps({"skip": f"{type(e).__name__}: {e}"[:400]})); sys.exit(0)
digest = lambda rows: hashlib.sha256("\n".join(r["item_id"] for r in rows).encode()).hexdigest()[:16]
print(json.dumps({"n_head": len(head), "n_full": len(full), "nested": full[:len(head)] == head,
                  "unique": len({r["item_id"] for r in full}), "head_sha": digest(head), "full_sha": digest(full)}))
"""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    proc = subprocess.run([sys.executable, "-c", probe, str(REPO)], env=env,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    facts = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skip" in facts:
        pytest.skip(f"the pinned cais/mmlu corpus is not cached offline here: {facts['skip']}")
    assert facts["n_head"] == tc.PARITY_N == 200
    assert facts["n_full"] == tc.MMLU_FULL_N == 1531 == tc.mmlu_item_set(full=True)["n"]
    assert facts["nested"] is True and facts["unique"] == 1531
    assert facts["head_sha"] == "373f1258bf4fcafc"     # the 200 ids, measured 2026-09-21
    assert facts["full_sha"] == "5b1153eae340d07f"     # the 1,531 ids, measured 2026-09-21


def test_the_removed_prefill_diagnostic_flags_are_unknown_to_the_driver(world, monkeypatch, capsys):
    """`--thought-block-diag` and `--tiny-crosscheck` served only the removed
    loglikelihood family's prefill reads; the driver's CLI no longer has them,
    so naming one is argparse's refusal (exit 2) naming the flag."""
    fake_frameworks(monkeypatch)
    for flag in ("--thought-block-diag", "--tiny-crosscheck"):
        with pytest.raises(SystemExit) as ei:
            srun.main(_percell_argv(world, tc.CONTROL_GSM8K)
                      + [tc.FLAG_THINKING, tc.THINKING_PER_TASK, flag])
        assert ei.value.code == 2
        assert flag in capsys.readouterr().err


def test_the_driver_flag_accepts_the_three_modes_and_records_the_retry_cap(capsys):
    """`--thinking {on,off,per-task}` and `--math500-retry-max-tokens`
    (0 disables; recorded either way) are on the driver's own CLI."""
    with pytest.raises(SystemExit):
        srun.main(["generate", "--help"])
    text = capsys.readouterr().out
    assert "per-task" in text and tc.FLAG_MATH500_RETRY_MAX_TOKENS in text
    for bad in ("yes", "sometimes"):
        with pytest.raises(SystemExit):
            srun.main(["generate", "--proxy-url", "u", "--routing-table", "r",
                       "--cell", "c", tc.FLAG_THINKING, bad])


#: The chat template `HeaderTok` carries by default, and its digest — the
#: standup's own, so a LOCAL fallback measured with this tokenizer passes the
#: served-vs-local comparison (E45 M5) instead of tripping E50 #6's
#: "comparison unavailable" refusal.
HEADER_TOK_TEMPLATE = "{{ 'the template the standup launched with' }}"
HEADER_TOK_TEMPLATE_HASH = cm.sha256_hex16(HEADER_TOK_TEMPLATE.encode())


class HeaderTok:
    """A tokenizer whose ``apply_chat_template`` reproduces the two measured
    header regimes (the thinking-header measurement, 2026-09-04):
    ``moves=False`` is gemma-4 — thinking is a LEADING system turn and the
    trailing assistant header stays H=3 in both arms; ``moves=True`` is
    Qwen3.5 — OFF closes an empty think block inside the header (H=7), ON
    leaves it open (H=5). ``relabel=True`` is the case a WIDTH comparison
    cannot see: the same three-token header with one id swapped under ON.
    ``template`` is the chat template the digest is taken over (the
    served-vs-local comparison, E45 M5); it DEFAULTS to the one the standup
    launched with (:data:`HEADER_TOK_TEMPLATE`), so the ordinary fixture is a
    world where that comparison can actually run. A tokenizer with no
    ``chat_template`` at all hashes to None, which is the E50 #6 case and has
    tests of its own."""

    def __init__(self, *, moves: bool, relabel: bool = False,
                 template=HEADER_TOK_TEMPLATE):
        self.moves, self.relabel = moves, relabel
        if template is not None:
            self.chat_template = template

    def apply_chat_template(self, messages, *, add_generation_prompt=False,
                            tokenize=True, enable_thinking=False, **kw):
        prompt = ([90, 91, 92, 93, 94, 95, 96] if enable_thinking else []) + [1, 2, 3]
        if not add_generation_prompt:
            return prompt
        if self.relabel:                                           # H=3 BOTH arms…
            return prompt + ([105, 4369, 107] if enable_thinking   # …different ids
                             else [105, 4368, 107])
        if not self.moves:
            return prompt + [105, 4368, 107]                       # H=3, both arms
        return prompt + ([10, 11, 12, 13, 14] if enable_thinking   # H=5 ON
                         else [10, 11, 12, 13, 14, 15, 16])        # H=7 OFF


def test_a_mixed_arm_run_is_admissible_only_while_the_assistant_header_holds_still():
    """U5-4 precondition, measured not assumed: a fitted object's response
    mask is keyed on the OFF header it was fit under, so mixing arms in ONE
    standup is safe only while the header does not move. gemma-4 does not
    move (H=3 both arms) and composes; Qwen3.5 moves 7 -> 5 and is refused on
    a LENS route, naming both widths. A non-lens route has no mask to
    misalign and is exempt — but the measurement is still recorded. The
    refusal is TYPED (E45 M5), so a caller can tell it from any other exit."""
    mixed = {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}
    # the local tokenizer IS the served template here, so the E50 #6
    # comparison runs and agrees — what is under test is the header, not it
    served = {"served_template_hash": HEADER_TOK_TEMPLATE_HASH}
    gemma = srun.assert_header_invariant_across_arms(
        HeaderTok(moves=False), mixed, lens=True, **served)
    assert gemma["invariant"] is True and gemma["verified"] is True
    assert gemma["header_len_by_arm"] == {"on": 3, "off": 3}
    assert gemma["header_ids_by_arm"]["on"] == [105, 4368, 107]
    with pytest.raises(srun.MixedArmHeaderMoved) as ei:
        srun.assert_header_invariant_across_arms(HeaderTok(moves=True), mixed, lens=True,
                                                 **served)
    msg = str(ei.value)
    assert "assistant header MOVES between them" in msg
    assert "{'off': 7, 'on': 5}" in msg and tc.GSM8K_STANDARD_TASK in msg
    assert "[10, 11, 12, 13, 14]" in msg          # BOTH renders, not just the widths
    assert issubclass(srun.MixedArmHeaderMoved, srun.HeaderMeasurementError)
    # unprompted / prompted (non-lens) routes carry no mask: exempt, recorded
    qwen = srun.assert_header_invariant_across_arms(
        HeaderTok(moves=True), mixed, lens=False, **served)
    assert qwen["invariant"] is False and qwen["lens"] is False
    assert qwen["header_len_by_arm"] == {"off": 7, "on": 5}
    # a single-arm run has nothing to compare
    assert srun.assert_header_invariant_across_arms(
        HeaderTok(moves=True), {tc.GSM8K_STANDARD_TASK: "off"}, lens=True, **served) is None


def test_the_header_comparison_is_over_token_ids_not_widths():
    """E45 M5: the guard compares the RENDERS, not their lengths. A template
    that keeps the header three tokens wide but relabels one of them under the
    ON arm masks a different span just as surely as one that grows it — and a
    width comparison would wave it through."""
    mixed = {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}
    block, error = srun.measure_thinking_header_by_arm(
        mixed, lens=True, tok=HeaderTok(moves=False, relabel=True),
        served_template_hash=HEADER_TOK_TEMPLATE_HASH)
    assert block["header_len_by_arm"] == {"on": 3, "off": 3}       # equal WIDTHS
    assert block["header_ids_by_arm"] == {"on": [105, 4369, 107],  # different IDS
                                          "off": [105, 4368, 107]}
    assert block["invariant"] is False and block["verified"] is False
    assert isinstance(error, srun.MixedArmHeaderMoved)
    assert block["error"] == str(error)


def test_the_header_is_measured_through_the_live_tokenize_and_the_fallback_is_recorded():
    """E45 M5: the primary source is the LIVE server's /v1/tokenize — the
    render the model is actually served, template pins and all — and a
    separately loaded tokenizer is a FALLBACK that is never silent: the block
    names the source and, when it fell back, why."""
    mixed = {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}
    served_tok, local_tok = HeaderTok(moves=False), HeaderTok(moves=True)

    def tokenize(body):
        kw = dict(body.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {})
        ids = list(served_tok.apply_chat_template(
            body["messages"], tokenize=True,
            add_generation_prompt=bool(body.get("add_generation_prompt")), **kw))
        return {"count": len(ids), "tokens": ids}

    # the two disagree: the live endpoint's answer is the one that counts
    block, error = srun.measure_thinking_header_by_arm(
        mixed, lens=True, tok=local_tok, tokenize=tokenize, vmodel="petri__x")
    assert error is None and block["source"] == srun.HEADER_SOURCE_SERVED
    assert block["fallback_reason"] is None
    assert block["header_len_by_arm"] == {"on": 3, "off": 3}
    assert block["local_chat_template_hash"] is None     # nothing local was read

    def dead(body):
        raise RuntimeError("connection refused")

    # a live /v1/tokenize that FAILS is an ERROR, recorded and returned — no longer a silent slide onto
    # the local tokenizer (the broad `except Exception` fallback is gone: the local render is measured ONLY when no
    # tokenize client exists at all, or for the OFF arm under the ON-pinned rule, test_served_capability_render_on_pinned)
    block, error = srun.measure_thinking_header_by_arm(
        mixed, lens=False, tok=local_tok, tokenize=dead, vmodel="petri__x")
    assert isinstance(error, srun.HeaderMeasurementError) and not isinstance(error, srun.MixedArmHeaderMoved)
    assert "connection refused" in str(error) and "never silently replaced" in str(error)
    assert block["error"] == str(error) and block["verified"] is False
    assert block["source"] == srun.HEADER_SOURCE_SERVED and block["fallback_reason"] is None
    assert block["header_ids_by_arm"] == {}                        # nothing local was measured in its place
    assert block[srun.HEADER_KEY_SOURCE_BY_ARM] == {"off": None, "on": None}
    # with no tokenize client at all the whole measurement is the recorded local fallback, as before (an explicit
    # configuration, not a masked failure) — the per-arm map says so for both arms
    block, _ = srun.measure_thinking_header_by_arm(mixed, lens=False, tok=local_tok)
    assert block["source"] == srun.HEADER_SOURCE_LOCAL
    assert "no live /v1/tokenize" in block["fallback_reason"]
    assert block["header_len_by_arm"] == {"off": 7, "on": 5}       # the LOCAL render
    assert block[srun.HEADER_KEY_SOURCE_BY_ARM] == {"off": srun.HEADER_SOURCE_LOCAL, "on": srun.HEADER_SOURCE_LOCAL}
    assert block[srun.HEADER_KEY_OFF_HEADER_ROLE] is None and block[srun.HEADER_KEY_OFF_ARM_ACCEPTED_UNDER] is None
    # and with neither source there is nothing to record: a typed refusal
    with pytest.raises(srun.HeaderMeasurementError, match="no local tokenizer"):
        srun.assert_header_invariant_across_arms(None, mixed, lens=True)


def test_a_fallback_measured_on_another_template_than_the_server_serves_refuses():
    """E45 M5, the reason `endpoint.json` now carries `chat_template_hash`: a
    local tokenizer is a SECOND derivation and can legitimately differ from
    the served render (`pin_reasoning_template_off` is a no-op on gemma-4 and
    is NOT on Qwen3.5). Measuring the wrong template proves nothing about the
    mask, so on a lens route the fallback is held to the digest the standup
    recorded — and a standup that predates the key reports the comparison as
    unavailable rather than passing it."""
    mixed = {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}
    other = HeaderTok(moves=False, template="{{ 'not the served template' }}")
    other_hash = cm.sha256_hex16(other.chat_template.encode())
    assert other_hash != TEMPLATE_HASH
    block, error = srun.measure_thinking_header_by_arm(
        mixed, lens=True, tok=other, served_template_hash=TEMPLATE_HASH)
    assert isinstance(error, srun.HeaderMeasurementError)
    assert not isinstance(error, srun.MixedArmHeaderMoved)   # not a moved header
    assert other_hash in str(error) and TEMPLATE_HASH in str(error)
    assert block["verified"] is False
    assert block["local_chat_template_hash"] == other_hash
    assert block["served_chat_template_hash"] == TEMPLATE_HASH
    # a non-lens route has no mask to misalign: recorded, not refused
    _, none_lens = srun.measure_thinking_header_by_arm(
        mixed, lens=False, tok=other, served_template_hash=TEMPLATE_HASH)
    assert none_lens is None
    # a pre-key standup: the comparison CANNOT be made, and E50 #6 inverts what
    # that meant — it used to pass silently (error None, verified true off
    # locally invariant headers alone); it now refuses, naming the side
    block, error = srun.measure_thinking_header_by_arm(
        mixed, lens=True, tok=other, served_template_hash=None)
    assert isinstance(error, srun.HeaderMeasurementError)
    assert block["served_chat_template_hash"] is None and block["verified"] is False
    assert block["template_comparison"]["unavailable"] == [srun.TEMPLATE_SIDE_SERVED]


def test_the_mixed_arm_header_measurement_lands_in_the_evidence(world, monkeypatch):
    """The fact that made the composition admissible is written down beside
    the run's other positive controls, not merely acted on — and it is the
    LIVE /v1/tokenize's answer, with the standup's template digest beside it."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    write_standup_endpoint(world.table_path, reasoning_parser="gemma4")
    monkeypatch.setattr(srun, "_load_served_tok",
                        lambda mid, rev: HeaderTok(moves=False))

    async def fake_run(*, atoms, store, thinking, limits=None, **kw):
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(_percell_argv(world, f"{tc.CONTROL_GSM8K},inspect_cmp:or_bench_hard_1k")
              + [tc.FLAG_THINKING, tc.THINKING_PER_TASK])
    ev = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())
    block = ev["thinking_header_by_arm"]
    assert block["invariant"] is True and block["verified"] is True
    assert block["header_len_by_arm"] == {"on": 3, "off": 3}
    assert block["source"] == srun.HEADER_SOURCE_SERVED
    assert block["served_chat_template_hash"] == TEMPLATE_HASH


def test_a_refused_header_measurement_is_written_to_the_evidence_before_the_exit(
        world, monkeypatch):
    """E45 M5: a failed precondition used to abort BEFORE evidence.json was
    written, so the container died with nothing on disk to diagnose (the same
    hole the bites control closed on 2026-09-01). The measurement — refusal
    and all — is recorded first, and only then does the run exit; nothing was
    generated."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    write_standup_endpoint(world.table_path, reasoning_parser="gemma4")
    monkeypatch.setattr(srun, "_load_served_tok",
                        lambda mid, rev: HeaderTok(moves=True))     # Qwen3.5: 7 -> 5

    async def never(*a, **kw):
        pytest.fail("no task may run after a refused header measurement")

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", never)
    with pytest.raises(SystemExit) as ei:
        srun.main(_percell_argv(world, f"{tc.CONTROL_GSM8K},inspect_cmp:or_bench_hard_1k")
                  + [tc.FLAG_THINKING, tc.THINKING_PER_TASK])
    assert "assistant header MOVES between them" in str(ei.value)
    ev = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())
    block = ev["thinking_header_by_arm"]
    assert block["verified"] is False and block["invariant"] is False
    assert block["header_len_by_arm"] == {"off": 7, "on": 5}
    assert "assistant header MOVES between them" in block["error"]


# --------------------------------------------------------------------------- #
# PW-sampling (U8 item 1): --sampling {greedy,card}, the five hashed identity #
# fields, the manifest block, the resume guard, the refusals                   #
# --------------------------------------------------------------------------- #

LLAMA_ROW = tc.SAMPLING_BY_MODEL_KEY["llama8b"]


def test_resolve_sampling_reads_the_row_off_the_configs_model_and_refuses_an_unpinned_one(world, monkeypatch):
    """The row is looked up from the config's HF id — exact, case-insensitive,
    never a mirror or family guess — and a model no canonical key pins REFUSES
    under `card` (never a default temperature) while `greedy` needs no row."""
    cfg = {"model": BASE_MODEL}
    by_task = {tc.GSM8K_STANDARD_TASK: "on", "evalplus/humaneval_plus": "off"}
    assert srun._sampling_model_key(BASE_MODEL) == "llama8b"             # the world's pin
    assert srun._sampling_model_key(BASE_MODEL.upper()) == "llama8b"      # case-insensitive
    assert srun._sampling_model_key("google/gemma-4-E4B-it") == "gemma4e4b"
    assert srun._sampling_model_key("meta-llama/Llama-3.1-8B") is None    # a near-miss is not a match
    assert srun._sampling_model_key(None) is None
    r = srun._resolve_sampling(SimpleNamespace(sampling="card", samples=None), cfg,
                               [tc.CONTROL_GSM8K, tc.CONTROL_HUMANEVAL_PLUS], by_task)
    assert r.mode == "card" and r.model_key == "llama8b" and r.row is LLAMA_ROW
    # the 2026-09-06 ruling: k = 1 everywhere by default — a card draw, once per item
    assert r.samples_by_task == {tc.GSM8K_STANDARD_TASK: 1, "evalplus/humaneval_plus": 1} and r.samples == 1
    g = srun._resolve_sampling(SimpleNamespace(sampling="greedy", samples=None), cfg,
                               [tc.CONTROL_GSM8K], {tc.GSM8K_STANDARD_TASK: "on"})
    assert g.row is tc.GREEDY_SAMPLING and g.samples_by_task == {tc.GSM8K_STANDARD_TASK: 1} and g.samples == 1
    # --samples is MATH-500's knob, in either mode
    e = srun._resolve_sampling(SimpleNamespace(sampling="greedy", samples=3), cfg,
                               [tc.CONTROL_MATH500], {"inspect/math": "on"})
    assert e.samples_by_task == {"inspect/math": 3} and e.samples == 3
    with pytest.raises(SystemExit, match="positive int"):
        srun._resolve_sampling(SimpleNamespace(sampling="card", samples=0), cfg,
                               [tc.CONTROL_MATH500], {"inspect/math": "on"})
    # no tinycap atoms: nothing to resolve (a legacy run)
    legacy = srun._resolve_sampling(SimpleNamespace(sampling="card", samples=None), cfg, [], {})
    assert legacy.row is None and legacy.samples_by_task == {} and legacy.samples is None
    # an unpinned model: refused under card, naming the model and both ways out
    monkeypatch.setitem(sg.MODEL_HF_IDS, "llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct")
    with pytest.raises(SystemExit) as ei:
        srun._resolve_sampling(SimpleNamespace(sampling="card", samples=None), cfg,
                               [tc.CONTROL_GSM8K], {tc.GSM8K_STANDARD_TASK: "on"})
    assert BASE_MODEL in str(ei.value) and "refusing to invent a temperature" in str(ei.value)
    assert "--sampling greedy" in str(ei.value)
    ok = srun._resolve_sampling(SimpleNamespace(sampling="greedy", samples=None), cfg,
                                [tc.CONTROL_GSM8K], {tc.GSM8K_STANDARD_TASK: "on"})
    assert ok.row is tc.GREEDY_SAMPLING and ok.model_key is None
    # a rowless-but-pinned model (Nemotron) refuses with the contract's message
    with pytest.raises(SystemExit, match="_from_model_config"):
        srun._resolve_sampling(SimpleNamespace(sampling="card", samples=None),
                               {"model": sg.MODEL_HF_IDS["nemotron3nano4b"]},
                               [tc.CONTROL_GSM8K], {tc.GSM8K_STANDARD_TASK: "on"})


def test_card_run_end_to_end_hashes_the_row_and_the_per_task_k_and_hands_them_to_the_leg(
        world, monkeypatch):
    """The default draw through the driver, offline: the Inspect leg is handed
    the model's frozen row and every task's k; the identity hashes the five
    fields (the row's numbers, the k map, the seed rules — never the source
    prose); the manifest's `sampling` block mirrors them with the mode, the
    key and the source; the seeds block names each Inspect task's k; and a
    `greedy` run of the same cell keys a DIFFERENT store."""
    from sisteer.capeval import store as cs
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    calls = []

    async def fake_run_inspect_tasks(*, atoms, store, thinking, limits=None, **kw):
        calls.append({"sampling": kw.get("sampling"), "samples_by_task": kw.get("samples_by_task"),
                      "samples": kw.get("samples")})
        task, n = tc.GSM8K_STANDARD_TASK, tc.GSM8K_PAIRED_N
        summary = {"acc": 0.8, "n": n, "n_requests": 4 * n, "n_rejected": 0, "samples_per_item": 4,
                   "task": task, "item_set": tc.GSM8K_ITEM_SET_PAIRED,
                   "framework": tc.INSPECT_TASK_VERSION, "engine": "vllm", "route": "petri__x",
                   "thinking": thinking, "render": tc.THINKING_RENDER_NATIVE, "wall_clock_s": 1.0,
                   "max_tokens": 512, "temperature": 0.6, "top_p": 0.9, "top_k": None}
        completion.finish_task(store.cell_dir, task=task, n_items=n, n_expected_full=n, limits=limits,
                               n_rows=5 * n, n_requests=4 * n, n_rejected=0, wall_clock_s=1.0,
                               summary=summary)
        return {task: summary}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run_inspect_tasks)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    [kw] = calls
    assert kw["sampling"] is LLAMA_ROW and kw["samples"] == 1
    assert kw["samples_by_task"] == {tc.GSM8K_STANDARD_TASK: 1}          # the ruling: one card draw
    cell_dir = _cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    ident = manifest["resolved_config"]["tinycap"]
    assert set(ident) == set(tc.TINYCAP_IDENTITY_FIELDS)
    assert (ident["temperature"], ident["top_p"], ident["top_k"]) == (0.6, 0.9, None)
    assert ident["samples_by_task"] == {tc.GSM8K_STANDARD_TASK: 1} and ident["samples"] == 1
    assert ident["seed_rule"] == dict(tc.SEED_RULES)
    assert "source" not in ident and LLAMA_ROW.source not in json.dumps(ident)
    assert cs.harness_config_hash(manifest["resolved_config"]) == manifest["harness_config_hash"]
    blk = manifest[tc.MANIFEST_KEY_SAMPLING]
    assert set(blk) == set(tc.SAMPLING_BLOCK_FIELDS)
    assert blk["mode"] == "card" and blk["model_key"] == "llama8b" and blk["source"] == LLAMA_ROW.source
    assert {f: blk[f] for f in tc.SAMPLING_IDENTITY_FIELDS} == {f: ident[f] for f in tc.SAMPLING_IDENTITY_FIELDS}
    assert manifest["seeds"]["inspect"] == {"rule": inspect_served.SEED_RULE,
                                            "samples_per_item": {tc.GSM8K_STANDARD_TASK: 1}}
    assert manifest["tinycap"] == ident
    # the same run again resumes onto the same store (the block agrees)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    assert _cell_dir(world.out_root) == cell_dir and len(calls) == 2
    # a greedy run of the same cell is a DIFFERENT store (another identity hash:
    # the same k = 1 everywhere, but 0.0 / null / null against the card row —
    # one stochastic draw and one argmax are two instruments), never a resume —
    # into its own out root, because two stores of one cell claiming the same
    # controls_summary entry is the PW8 merge refusal, by design
    greedy_root = world.out_root.parent / "out_greedy"
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K, out_root=greedy_root)
              + [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_SAMPLING, tc.SAMPLING_GREEDY])
    assert calls[-1]["sampling"] is tc.GREEDY_SAMPLING and calls[-1]["samples"] == 1
    other = _cell_dir(greedy_root)
    assert other.name != cell_dir.name
    gm = json.loads((other / "manifest.json").read_text())
    g = gm["resolved_config"]["tinycap"]
    assert (g["temperature"], g["top_p"], g["top_k"]) == (0.0, None, None)
    assert g["samples_by_task"] == {tc.GSM8K_STANDARD_TASK: 1} and g["samples"] == 1
    assert gm[tc.MANIFEST_KEY_SAMPLING]["mode"] == "greedy"
    assert gm["harness_config_hash"] != manifest["harness_config_hash"]


def test_the_sampling_resume_guard_refuses_a_store_drawn_another_way(world, monkeypatch):
    """The 8-hex key prefix cannot see a collision or a hand-edited store: the
    manifest's `sampling` block is compared field by field (mode + the five)
    and a block that disagrees — or is missing (a store from before the
    contract) — refuses, naming what moved, before any resume skip."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    monkeypatch.setattr(inspect_served, "run_inspect_tasks", _fake_inspect_runner_noop)
    argv = _percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF]
    srun.main(argv)
    cell_dir = _cell_dir(world.out_root)
    path = cell_dir / "manifest.json"
    original = path.read_text()
    m = json.loads(original)
    m[tc.MANIFEST_KEY_SAMPLING]["temperature"] = 1.0
    m[tc.MANIFEST_KEY_SAMPLING]["mode"] = "greedy"
    path.write_text(json.dumps(m))
    with pytest.raises(SystemExit) as ei:
        srun.main(argv)
    msg = str(ei.value)
    assert "resume mismatch on the sampling contract" in msg
    assert "temperature: the manifest records 1.0, this run is 0.6" in msg
    assert "mode: the manifest records 'greedy', this run is 'card'" in msg
    # a manifest from before the contract: no block at all
    m = json.loads(original)
    del m[tc.MANIFEST_KEY_SAMPLING]
    path.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="carries no 'sampling' block"):
        srun.main(argv)
    path.write_text(original)
    srun.main(argv)                                                      # restored: resumes


async def _fake_inspect_runner_noop(*, atoms, store, thinking, limits=None, **kw):
    task, n = tc.GSM8K_STANDARD_TASK, tc.GSM8K_PAIRED_N
    summary = {"acc": 0.5, "n": n, "n_requests": n, "n_rejected": 0, "samples_per_item": 4,
               "task": task, "item_set": tc.GSM8K_ITEM_SET_PAIRED, "framework": tc.INSPECT_TASK_VERSION,
               "engine": "vllm", "route": "petri__x", "thinking": thinking,
               "render": tc.THINKING_RENDER_NATIVE, "wall_clock_s": 1.0, "max_tokens": 512,
               "temperature": 0.6, "top_p": 0.9, "top_k": None}
    completion.finish_task(store.cell_dir, task=task, n_items=n, n_expected_full=n, limits=limits,
                           n_rows=n, n_requests=n, n_rejected=0, wall_clock_s=1.0, summary=summary)
    return {task: summary}


def test_card_with_an_unpinned_model_refuses_before_any_store(world, monkeypatch):
    """A refusal at plan time, nothing touched: a config model no canonical key
    pins under `card` — never a default temperature; and an unknown mode is
    argparse's refusal."""
    fake_frameworks(monkeypatch)
    monkeypatch.setitem(sg.MODEL_HF_IDS, "llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct")
    with pytest.raises(SystemExit, match="not the pinned HF id of any canonical model key"):
        srun.main(_percell_argv(world, "inspect_cmp:or_bench_hard_1k") + [tc.FLAG_THINKING, tc.THINKING_OFF])
    assert not (world.out_root / "cells").exists()
    with pytest.raises(SystemExit) as ei:                                # argparse: an unknown mode
        srun.main(_percell_argv(world, "inspect_cmp:or_bench_hard_1k") + [tc.FLAG_THINKING, tc.THINKING_OFF,
                                                              tc.FLAG_SAMPLING, "nucleus"])
    assert ei.value.code == 2


# --------------------------------------------------------------------------- #
# PW-oos (U10 item 5) — the driver half of the out-of-support override         #
# --------------------------------------------------------------------------- #

def test_the_override_lets_thinking_on_serve_with_no_parser_and_says_so(capsys):
    """PW-oos: a family with no VERIFIED reasoning parser is exactly the family
    the authorised experiment serves ON, and the thought text landing in
    `content` is one of the things it measures — so the FIRST rule becomes a
    printed statement. The other two are untouched: an UNPINNED parser is not
    made reviewable by this flag, and a parser serving an all-OFF run is still
    a standup that does not match its arm."""
    none = {"reasoning_parser": None, "tool_calls": False}
    srun._check_served_reasoning_parser(["gsm8k"], "on", none, allow_out_of_support=True)
    out = capsys.readouterr().out
    assert tc.FLAG_ALLOW_OOS_THINKING in out and "NO reasoning parser served" in out
    assert tc.OUT_OF_SUPPORT_THINKING_NOTE in out
    # the same call WITHOUT the flag still refuses — nothing changed by default
    with pytest.raises(SystemExit, match="NO reasoning parser"):
        srun._check_served_reasoning_parser(["gsm8k"], "on", none)
    # the two rules the override does NOT touch
    with pytest.raises(SystemExit, match="not a pinned family parser"):
        srun._check_served_reasoning_parser(
            ["gsm8k"], "on", {"reasoning_parser": "hermes", "tool_calls": False},
            allow_out_of_support=True)
    with pytest.raises(SystemExit, match="thinking-OFF but the standup composed"):
        srun._check_served_reasoning_parser(
            ["gsm8k"], "off", {"reasoning_parser": "gemma4", "tool_calls": False},
            allow_out_of_support=True)


def test_the_override_measures_both_arms_on_a_single_arm_run_and_does_not_refuse():
    """PW-oos: the override keeps the MEASUREMENT and drops only the REFUSAL —
    and it measures both arms even on a whole-arm ON standup, which by the
    mixed-arm rule would have nothing to compare and would record no header at
    all. The two widths ARE the finding the experiment exists to state."""
    on_only = {tc.GSM8K_STANDARD_TASK: "on"}
    served = {"served_template_hash": HEADER_TOK_TEMPLATE_HASH}
    # without the override a single-arm run measures nothing
    assert srun.measure_thinking_header_by_arm(
        on_only, lens=True, tok=HeaderTok(moves=True), **served) == (None, None)
    block, error = srun.measure_thinking_header_by_arm(
        on_only, lens=True, tok=HeaderTok(moves=True), allow_out_of_support=True,
        **served)
    assert error is None                                   # the refusal is lifted…
    assert block["header_moves"] is True                   # …the measurement is not
    assert block["header_len_by_arm"] == {"off": 7, "on": 5}
    assert block["header_ids_by_arm"] == {"off": [10, 11, 12, 13, 14, 15, 16],
                                          "on": [10, 11, 12, 13, 14]}
    assert block["run_arms"] == ["on"] and block["measured_arms"] == ["off", "on"]
    assert block[tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    assert block["out_of_support_thinking_note"] == tc.OUT_OF_SUPPORT_THINKING_NOTE
    # `verified` stays the honest answer to "does the invariance hold?" —
    # never a green tick over a precondition that was waived
    assert block["verified"] is False and block["invariant"] is False
    assert "assistant header MOVES between them" in block["overridden_refusal"]
    # an invariant family under the override: nothing to override, nothing claimed
    ok, err = srun.measure_thinking_header_by_arm(
        on_only, lens=True, tok=HeaderTok(moves=False), allow_out_of_support=True,
        **served)
    assert err is None and ok["header_moves"] is False and ok["invariant"] is True
    assert ok["verified"] is True and "overridden_refusal" not in ok


def test_the_override_does_not_lift_an_unmeasurable_or_wrong_template_header():
    """It is about serving out of support, not about measuring nothing: a
    header that cannot be measured, and a local tokenizer whose template is not
    the served one, are still errors under the flag."""
    mixed = {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}
    with pytest.raises(srun.HeaderMeasurementError, match="no local tokenizer"):
        srun.assert_header_invariant_across_arms(None, mixed, lens=True,
                                                 allow_out_of_support=True)
    other = HeaderTok(moves=True, template="{{ 'not the served template' }}")
    _, error = srun.measure_thinking_header_by_arm(
        mixed, lens=True, tok=other, served_template_hash=TEMPLATE_HASH,
        allow_out_of_support=True)
    assert isinstance(error, srun.HeaderMeasurementError)
    assert not isinstance(error, srun.MixedArmHeaderMoved)


def test_header_moves_is_recorded_on_every_measurement_not_only_overridden_ones():
    """`header_moves` is the positive statement of `invariant`, so a reader of
    any evidence.json finds the fact without negating a field."""
    mixed = {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}
    inv, _ = srun.measure_thinking_header_by_arm(mixed, lens=True, tok=HeaderTok(moves=False))
    assert inv["header_moves"] is False and inv["invariant"] is True
    mv, _ = srun.measure_thinking_header_by_arm(mixed, lens=False, tok=HeaderTok(moves=True))
    assert mv["header_moves"] is True and mv["invariant"] is False


def test_the_override_runs_the_cell_and_stamps_the_identity_and_the_evidence(
        world, monkeypatch):
    """PW-oos end to end through the driver, on the DECLARED thinking-ON world
    (phase B1 — the landed lens half's shapes, synthetic, per contract-2.5.md):
    the same moved-header standup that ABORTS without the flag
    (`test_a_refused_header_measurement_is_written_to_the_evidence_before_the_exit`)
    runs to completion with it — the tasks run, the hashed identity carries the
    override's THREE keys (`out_of_support_thinking`, the executed
    `max_tokens_by_task`, the CONTRACT literal `serve_out_of_support`), the
    route is the `@thinking` spelling, the manifest `route` block copies the
    RAW attestation map, and evidence.json carries the measurement that would
    have refused (`header_moves`), the header + render-fact CERTIFICATION, the
    probe's render arm and stamps, and the live prompt budget."""
    from sisteer.capeval import store as cs
    from sisteer.capeval.tiers import inspect_served

    oos = oos_on_world(world, monkeypatch)
    ran = []

    async def fake_run(*, atoms, store, thinking, limits=None, **kw):
        ran.append((thinking, kw["out_of_support"], kw["prompt_opened_think"]))
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(oos.argv)
    assert ran == [(tc.THINKING_ON, True, True)]           # it ran, rather than aborting
    cell_dir = _cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    ident = manifest["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {tc.GSM8K_STANDARD_TASK: 8192}
    assert ident[tc.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] == ["thinking"] == tc.OOS_CELL_DECLARATION_EXPECTED
    assert set(ident) == set(tc.tinycap_identity_fields(out_of_support_thinking=True))
    assert cs.harness_config_hash(manifest["resolved_config"]) == manifest["harness_config_hash"]
    # the declared spelling everywhere the route is named (contract §2.5.10 door table)
    assert manifest["virtual_model"] == OOS_VMODEL == oos.route.virtual_model
    assert manifest["harness_cell_id"] == OOS_OWN_KEY and manifest["cell_id"] == TT_CELL
    assert manifest["route"]["virtual_model"] == OOS_VMODEL
    # the RAW map, whole, unhashed, in the route block — and NOT in the hashed identity
    assert manifest["route"][tc.ROUTE_KEY_SERVE_DECLARATIONS] == OOS_MAP
    assert manifest["route"][tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] == OBJECT_SHA
    assert manifest["route"][tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL] == {OOS_OWN_KEY: OBJECT_SHA}
    assert manifest["route"][tc.ENDPOINT_KEY_LENS_SALT_POLICY] == "attested"
    assert manifest["route"]["resumed_prefill_policy"] == "conditional"
    assert set(manifest["route"]) == set(tc.ROUTE_BLOCK_FIELDS)
    assert tc.ROUTE_KEY_SERVE_DECLARATIONS not in json.dumps(ident) and "none__lens0" not in json.dumps(ident)
    evidence = json.loads((cell_dir / "evidence.json").read_text())
    block = evidence["thinking_header_by_arm"]
    assert block["header_moves"] is True and block["verified"] is False
    assert block["header_len_by_arm"] == {"off": 7, "on": 5}
    assert block["header_ids_by_arm"] == {"off": list(tc.QWEN35_HEADER_IDS_OFF), "on": list(tc.QWEN35_HEADER_IDS_ON)}
    assert block["source"] == srun.HEADER_SOURCE_SERVED   # the LIVE /v1/tokenize
    assert "assistant header MOVES between them" in block["overridden_refusal"]
    # S3.4: certified against the constants AND the live proxy's attested block
    assert block[tc.OOS_EVIDENCE_HEADER_CERTIFIED] is True
    assert block[tc.OOS_EVIDENCE_HEADER_ATTESTED_AGREES] is True
    assert block[tc.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED] is True      # true/false, never null
    assert block[tc.OOS_EVIDENCE_ATTESTED_SWITCH] is True and block[tc.OOS_EVIDENCE_ATTESTED_MOVES] is True
    assert block["attested_thinking_header"] == OOS_THINKING_HEADER and block["prompt_opened_think_declared"] is True
    # S3.5: the probe rendered ON with the declaration on BOTH legs, three legs
    # probed (steered, twin, twin repeat is the logprob leg's — here the two
    # generation legs), stamps one triple each == §2.5.11 rows 1–2
    bites = evidence["steering_bites"]
    assert bites["verified"] is True and bites[tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == tc.THINKING_ON
    assert bites["probe_extra_body"] == {**tc.chat_template_kwargs("on"), tc.OOS_THINKING_DECLARATION_KEY: True}
    row = tc.expected_stamp_regimes(lens=True, arm=tc.THINKING_ON)._asdict()
    assert row == {"salt_regime": "request-unique", "resumed_policy": "declared-request-unique",
                   "render_regime": "thinking_on"}
    for leg in ("steered", "scale0"):
        assert len(bites["stamps_by_leg"][leg]) == 5 and set(map(json.dumps, bites["stamps_by_leg"][leg])) == {json.dumps(row)}
    probe_calls = [c for c in oos.chat.calls if not c.get("continue_final_message")]
    assert {c["model"] for c in probe_calls} == {OOS_VMODEL, oos.twin.virtual_model}
    for c in probe_calls:
        assert c[tc.CHAT_TEMPLATE_KWARGS_KEY] == {tc.ENABLE_THINKING_KEY: True} and c[tc.OOS_THINKING_DECLARATION_KEY] is True
        assert c["temperature"] == 0.0 and c["max_tokens"] == 48 and c["seed"] == 0
    # S3.8: the live budget over the 200 exact messages, equal to the artifact's
    live = evidence[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK]
    art = json.loads(oos.budget.read_text())
    assert live["n"] == 200 and live[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX] == art["max"]
    assert live[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99] == art["p99"] and live["n_items_differing"] == 0
    assert live["cap"] == 8192 and live["window"] == OOS_WINDOW and live["margin"] == OOS_WINDOW - art["max"] - 8192 >= 0
    assert live["ok"] is True and live["source"] == srun.HEADER_SOURCE_SERVED
    # the tokenize bodies carried the ON kwargs and NO declaration
    tokenized = [b for b in oos.tokenized if len(b["messages"]) == 2 and b["messages"][0]["role"] == "system"]
    assert len(tokenized) == 200 and all(
        b[tc.CHAT_TEMPLATE_KWARGS_KEY] == {tc.ENABLE_THINKING_KEY: True} and tc.OOS_THINKING_DECLARATION_KEY not in b
        and b["add_generation_prompt"] is True for b in tokenized)


#: the stock GSM8K task's module, the ONE name the real plan builder imports
#: from ``inspect_evals`` on the GSM8K path (``from inspect_evals.gsm8k.gsm8k
#: import gsm8k`` in ``inspect_served.plan_task``)
GSM8K_TASK_MODULE = "inspect_evals.gsm8k.gsm8k"
#: ``inspect_ai`` (the stock ``match`` scorer the REAL plan builder wraps in
#: ``inspect_served._gsm8k_scorer``) is pinned ONLY in the ``petri`` dependency
#: group (``pyproject.toml`` ``[dependency-groups] petri = ["inspect-ai==0.3.251",
#: ...]``; ``uv.lock``'s dev-dependencies ``petri``), which the Modal CPU suite's
#: image does not sync (``scripts/modal/cpu_suite.py``: ``groups=["dev"],
#: extras=["cpu"]``). The two tests that build the real GSM8K plan therefore
#: ``importorskip`` it — the ``test_inspect_served`` precedent — and run in the
#: tinycap client venv (hashed lock); everything else in this module runs
#: without it. A stub would not do here: the scorer wrap is the code under test.
INSPECT_AI_SKIP_REASON = ("inspect_ai is pinned only in the `petri` dependency group; the CPU-suite image "
                          "(groups=['dev'], extras=['cpu']) has none — the tinycap client venv (hashed lock) does")


def _stub_gsm8k_paired200(monkeypatch):
    """The paired-200 GSM8K plan's inputs, SYNTHETIC: 200 platinum-style rows
    whose questions all match a stock-task stub, so the REAL ``plans_for_atoms``
    can build the real plan — cap, split mode, sample ids — inside the driver,
    offline. ``inspect_evals`` is a tinycap CLIENT-image pin only: the pinned
    Modal CPU suite's image (``uv.lock``: inspect-ai, inspect-petri,
    inspect-scout, never inspect_evals) has no such distribution, so the stock
    task's module is STUBBED in ``sys.modules`` (the ``fake_frameworks``
    precedent) with the one attribute the builder reads — ``gsm8k()``, whose
    task exposes a sized ``dataset`` of ``.id``s and takes a ``scorer`` — and the
    package chain above it, whether or not a real ``inspect_evals`` is
    installed. ``inspect_ai`` (the stock ``match`` scorer the builder wraps)
    stays REAL and is pinned only in the ``petri`` group, so each caller
    ``importorskip``s it first (:data:`INSPECT_AI_SKIP_REASON`)."""
    import importlib
    import types

    from sisteer.capeval.tiers import inspect_served as isv

    parts = GSM8K_TASK_MODULE.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        mod = types.ModuleType(name)
        if depth < len(parts):
            mod.__path__ = []                              # a package, so the dotted import descends
        monkeypatch.setitem(sys.modules, name, mod)
    gsm_mod = sys.modules[GSM8K_TASK_MODULE]
    parity = importlib.import_module("sisteer.capeval.tiers.parity")
    rows = [{"item_id": f"g{i:04d}-{i:08x}", "question": f"q {i}"} for i in range(tc.GSM8K_PAIRED_N)]
    ids = sorted(tc.gsm8k_sample_id(r["question"]) for r in rows)
    monkeypatch.setattr(parity, "gsm8k_parity_rows", lambda pins, n: list(rows))
    monkeypatch.setattr(gsm_mod, "gsm8k", lambda **k: types.SimpleNamespace(
        dataset=[types.SimpleNamespace(id=i) for i in ids]), raising=False)
    monkeypatch.setattr(isv, "_pins", lambda repo_root=None: None)
    return ids


def _real_run_inspect_tasks_capturing_plans(monkeypatch):
    """Let the driver reach the REAL ``run_inspect_tasks`` (its model listing,
    model build and per-task runner stubbed) and capture (i) the kwargs it
    hands ``plans_for_atoms`` and (ii) the ``TaskPlan`` each task is dispatched
    with — the plan the run would actually execute."""
    from sisteer.capeval.tiers import inspect_served as isv

    seen = {"plans": [], "plan_kwargs": []}
    real_plans = isv.plans_for_atoms

    def recording_plans(atoms, **kw):
        seen["plan_kwargs"].append(dict(kw))
        return real_plans(atoms, **kw)

    async def capture_run(plan, **kw):
        seen["plans"].append(plan)
        return {}

    def stub_model(vmodel, **kw):
        # S3.3: the runner requires the target to expose the tinycap provider
        # (its regime-stamp accumulator + route); a stub carries that shape
        return SimpleNamespace(api=SimpleNamespace(regime_stamps={}, route=isv.stamp_route(kw["route"])))

    monkeypatch.setattr(isv, "get_models", lambda **k: {"object": "list", "data": []})
    monkeypatch.setattr(isv, "preflight_models", lambda **k: list(k.get("required") or ()))
    monkeypatch.setattr(isv, "build_model", stub_model)
    monkeypatch.setattr(isv, "plans_for_atoms", recording_plans)
    monkeypatch.setattr(isv, "run_one_task", capture_run)
    return seen


def test_the_override_hands_run_inspect_tasks_the_raised_cap_and_the_declared_split(
        world, monkeypatch):
    """End to end through the driver into the REAL
    ``run_inspect_tasks``: under ``--allow-out-of-support-thinking --thinking
    on`` the GSM8K plan the run dispatches carries the authorised experiment's
    raised cap (8192, ``oos_override``) and the declared render fact's split
    (``implicit_open``) — before this the driver passed neither flag, so the
    store was stamped out of support while the plan sat at the in-support
    2,048 floor with ``explicit_tags``. The two kwargs reach ``plans_for_atoms``
    from ONE resolution in the driver, and evidence.json records the
    certification as NOT done (an explicit null under its contract name, beside
    the declared fact) — never true."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)     # `petri` group only
    from sisteer.capeval.tiers import inspect_served as isv

    # the REAL inspect_ai (present: the skip above) must stay importable for the
    # real plan builder below (its stock `match` scorer), so the pin check is
    # stubbed at the driver, not via sys.modules; inspect_evals itself is stubbed
    # by _stub_gsm8k_paired200
    oos = oos_on_world(world, monkeypatch, frameworks="stub")     # the REAL inspect_ai stays importable
    ids = _stub_gsm8k_paired200(monkeypatch)
    seen = _real_run_inspect_tasks_capturing_plans(monkeypatch)
    srun.main(oos.argv)
    [kw] = seen["plan_kwargs"]
    assert kw["out_of_support"] is True and kw["prompt_opened_think"] is True
    assert kw["prompt_opened_think"] == tc.AUTHORISED_OOS_THINKING.prompt_opens_think_on
    assert kw["reasoning_parser"] is None and kw["thinking"] == tc.THINKING_ON
    [plan] = seen["plans"]
    assert isinstance(plan, isv.TaskPlan) and plan.row_task == tc.GSM8K_STANDARD_TASK
    assert plan.max_tokens == 8192 == tc.oos_max_tokens_on("gsm8k_paired200")
    assert plan.cap_source == "oos_override"
    assert plan.think_split_mode == tc.THINK_SPLIT_IMPLICIT_OPEN == "implicit_open"
    assert plan.thinking == tc.THINKING_ON and plan.sample_ids == ids and plan.expected_n == 200
    assert (plan.temperature, plan.top_p, plan.top_k) == (0.7, 0.95, 20)      # the card row rode along
    # S3.1: the hashed cap map IS the dispatched plans' caps — one derivation
    ident = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {p.row_task: p.max_tokens for p in seen["plans"]}
    # the driver's own source: ONE resolution feeds the call
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "out_of_support=oos_run," in src and "prompt_opened_think=oos_prompt_opened_think," in src
    assert "stamp_policy=stamp_policy," in src             # ... and the leg's label, resolved once too
    # the durable ledger path the manifest records rides the same call (followed, since
    # the bcb-bfcl-200 merge, by the ladder's rung-2 tuple)
    assert "ledger_path=(Path(out_root) / tc.LEDGER_FILENAME\n" in src
    assert "if args.ledger_url else None),\n" in src
    assert "cap_overrides_by_task=(dict(ladder[0]) if ladder is not None else None)," in src
    # evidence: the render fact is CERTIFIED (S3.4) — true, beside the declared fact it was built on
    block = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["thinking_header_by_arm"]
    assert block[tc.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED] is True
    assert block["prompt_opened_think_declared"] is True
    assert block["header_moves"] is True                       # the measurement itself is unchanged


def test_the_in_support_call_to_run_inspect_tasks_is_byte_identical_to_before(world, monkeypatch):
    """The invariance pin for the wiring above: an ordinary thinking-OFF run
    hands ``plans_for_atoms`` ``out_of_support=False, prompt_opened_think=False``
    (the defaults every pre-wiring call took) and the dispatched plan equals,
    field for field, the plan built from the same kwargs WITHOUT the two keys:
    the table cap (512, ``off_table``), ``record_only``, the same ids."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)     # `petri` group only
    import dataclasses

    from sisteer.capeval.tiers import inspect_served as isv

    stub_framework_versions(monkeypatch)
    write_standup_endpoint(world.table_path)
    ids = _stub_gsm8k_paired200(monkeypatch)
    seen = _real_run_inspect_tasks_capturing_plans(monkeypatch)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    [kw] = seen["plan_kwargs"]
    assert kw["out_of_support"] is False and kw["prompt_opened_think"] is False
    [plan] = seen["plans"]
    assert (plan.max_tokens, plan.cap_source) == (512, "off_table")
    assert plan.think_split_mode == tc.THINK_SPLIT_RECORD_ONLY and plan.sample_ids == ids
    before = {k: v for k, v in kw.items() if k not in ("out_of_support", "prompt_opened_think")}
    [rebuilt] = isv.plans_for_atoms(["gsm8k"], **before)
    strip = lambda p: dataclasses.replace(p, task=None)        # noqa: E731 — the Task object is fresh per build
    assert strip(plan) == strip(rebuilt)
    block = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())
    assert tc.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED not in json.dumps(block)   # no stamp in support
    ident = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())["resolved_config"]["tinycap"]
    assert tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in ident


def test_without_the_flag_the_same_run_keys_a_store_the_override_cannot_touch(
        world, monkeypatch):
    """The invariance pin at the identity level: an ordinary thinking-OFF run
    on the same world records EXACTLY the unconditional field set — the
    override's key appears nowhere — so an in-support store hashes as it always
    did and the two worlds can never key onto each other."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)

    async def fake_run(*, atoms, store, thinking, limits=None, **kw):
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    ident = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text()
                       )["resolved_config"]["tinycap"]
    assert set(ident) == set(tc.TINYCAP_IDENTITY_FIELDS)
    assert tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in ident


def test_the_letter_read_width_is_hashed_store_identity():
    """§PW-nanfix-2: the letter-read top-K is per model now, so two waves of
    the same cell at two widths are two MEASUREMENTS — the narrower one may
    have refused items the wider one can read. It is already inside the
    driver's own resolved config (``mmlu.top_logprobs``), which
    ``harness_config_hash`` covers whole, so the two land in different cell-key
    directories and can never pool or resume onto each other. Pinned here
    because nothing else asserts the CONSEQUENCE, only the field."""
    from sisteer.capeval import store as cs
    from sisteer.capeval.tinycap import contracts as tc

    cell = SimpleNamespace(id="tinycap-x", object_path=None)
    route = SimpleNamespace(virtual_model="petri__none",
                            serving_identity="proxy_vllm+lens+interleaved")
    def resolved(k):
        return srun._resolved_config(cell, {"completion_tokens": 512}, None, 0, "judge",
                                     route=route, model_id="m", model_revision="r",
                                     top_logprobs=k)
    narrow, wide = resolved(tc.LETTER_TOP_LOGPROBS_DEFAULT), resolved(tc.LETTER_TOP_LOGPROBS_WIDE)
    assert narrow["mmlu"]["top_logprobs"] == tc.LETTER_TOP_LOGPROBS_DEFAULT
    assert wide["mmlu"]["top_logprobs"] == tc.LETTER_TOP_LOGPROBS_WIDE
    assert cs.harness_config_hash(narrow) != cs.harness_config_hash(wide)
    # and nothing ELSE moved: the width is the only difference
    assert {k: v for k, v in narrow.items() if k != "mmlu"} == {
        k: v for k, v in wide.items() if k != "mmlu"}
    assert cs.harness_config_hash(resolved(tc.LETTER_TOP_LOGPROBS_DEFAULT)) == (
        cs.harness_config_hash(narrow))


# --------------------------------------------------------------------------- #
# E50 #3 — the driver door: the override authorises ONE tuple, not a family    #
# --------------------------------------------------------------------------- #


def _pin_the_authorised_oos_model(monkeypatch):
    """Point the world's served model at the AUTHORISED out-of-support slug.

    The `world` fixture pins that model under `llama8b` so every driver test
    resolves a card row; the authorisation compares the resolved model KEY, so
    a test of the override has to restore llama8b's real id and pin the
    authorised slug at the world's model instead."""
    monkeypatch.setitem(sg.MODEL_HF_IDS, "llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct")
    monkeypatch.setitem(sg.MODEL_HF_IDS, tc.AUTHORISED_OOS_THINKING.model, BASE_MODEL)


def _oos_argv(world, **swap) -> list:
    """The world's argv for the AUTHORISED run (fitted cell, GSM8K, thinking
    ON, the flag named), with any flag's value swapped."""
    argv = (_percell_argv(world, tc.CONTROL_GSM8K)
            + [tc.FLAG_THINKING, tc.THINKING_ON, tc.FLAG_ALLOW_OOS_THINKING])
    for flag, value in swap.items():
        flag = "--" + flag.replace("_", "-")
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv += [flag, value]
    return argv


@pytest.mark.parametrize("field,swap", [
    ("model", None),                                     # the world's own llama8b pin
    ("cell", {"cell": "si_constitution__psr_kl_all"}),
    ("controls", {"controls": "inspect_cmp:or_bench_hard_1k"}),
    ("mode", {"thinking": tc.THINKING_PER_TASK}),
    ("sampling_mode", {"sampling": tc.SAMPLING_GREEDY}),
    ("samples", {"samples": "4"}),
])
def test_the_driver_refuses_any_run_that_is_not_the_authorised_experiment(
        world, monkeypatch, field, swap):
    """E50 #3: the driver is a SEPARATE entry point — a hand invocation never
    passes through the launcher's plan — so it compares the same declared tuple
    itself, before any store is touched. Every field, one at a time; the
    refusal names the one that moved.

    The `model` case needs no swap: leaving the fixture's `llama8b` pin in
    place IS the mutation (the authorised experiment is qwen35_4b)."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    write_standup_endpoint(world.table_path)
    if field != "model":
        _pin_the_authorised_oos_model(monkeypatch)
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: HeaderTok(moves=True))

    async def fake_run(*, atoms, store, thinking, limits=None, **kw):
        raise AssertionError("an unauthorised out-of-support run must never generate")

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    with pytest.raises(SystemExit) as e:
        srun.main(_oos_argv(world, **(swap or {})))
    msg = str(e.value)
    assert f"{field} differs" in msg
    assert tc.FLAG_ALLOW_OOS_THINKING in msg and "AUTHORISED_OOS_THINKING" in msg
    # nothing was written: the refusal precedes the store
    assert not (world.out_root / "cells").exists()


def test_the_driver_refusal_precedes_the_identity_it_would_have_stamped(world, monkeypatch):
    """Why the check sits where it does: the flag STAMPS
    `out_of_support_thinking` into the hashed identity, so a store keyed that
    way by an unauthorised run is indistinguishable from the authorised
    experiment's. Refusing after the fact would leave that store behind."""
    fake_frameworks(monkeypatch)
    write_standup_endpoint(world.table_path)
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: HeaderTok(moves=True))
    with pytest.raises(SystemExit) as e:
        srun.main(_oos_argv(world))                      # llama8b: not the authorised model
    assert "model differs" in str(e.value)
    assert not (world.out_root / "cells").exists()
    assert tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in tc.tinycap_identity_fields(
        out_of_support_thinking=True)                    # the key it would have stamped


# --------------------------------------------------------------------------- #
# E50 #6 — `verified` only when the template comparison actually RAN           #
# --------------------------------------------------------------------------- #


_NO_TEMPLATE = object()          # a tokenizer with no chat_template: hashes to None


def _header_block(*, local, served, lens=True):
    """One local-fallback measurement (no `/v1/tokenize`), with the two
    template digests set independently: `local` is the tokenizer's own
    template (`None` = a tokenizer carrying none, which hashes to None) and
    `served` is what `endpoint.json` recorded."""
    tok = (HeaderTok(moves=False, template=None) if local is None
           else HeaderTok(moves=False, template=local))
    return srun.measure_thinking_header_by_arm(
        {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"},
        lens=lens, tok=tok, served_template_hash=served)


def test_the_template_comparison_record_says_whether_it_ran():
    """E50 #6, the pure function: `ran` is the fact, `unavailable` names the
    SIDE, and `equal` is None when there was nothing to compare — so no caller
    has to re-derive "did this comparison happen?" from two null checks."""
    both = srun.template_comparison(required=True, local_hash=None, served_hash=None)
    assert both["ran"] is False and both["equal"] is None
    assert both["unavailable"] == [srun.TEMPLATE_SIDE_LOCAL, srun.TEMPLATE_SIDE_SERVED]
    left = srun.template_comparison(required=True, local_hash=None, served_hash="ab")
    assert left["unavailable"] == [srun.TEMPLATE_SIDE_LOCAL] and left["ran"] is False
    right = srun.template_comparison(required=True, local_hash="ab", served_hash=None)
    assert right["unavailable"] == [srun.TEMPLATE_SIDE_SERVED] and right["ran"] is False
    same = srun.template_comparison(required=True, local_hash="ab", served_hash="ab")
    assert same["ran"] is True and same["equal"] is True and same["unavailable"] == []
    diff = srun.template_comparison(required=True, local_hash="ab", served_hash="cd")
    assert diff["ran"] is True and diff["equal"] is False
    # a comparison nobody required is still described, never claimed
    assert srun.template_comparison(
        required=False, local_hash=None, served_hash=None)["required"] is False


@pytest.mark.parametrize("case,local,served,ran,verified", [
    ("both-null", None, None, False, False),
    ("left-null", None, HEADER_TOK_TEMPLATE_HASH, False, False),
    ("right-null", HEADER_TOK_TEMPLATE, None, False, False),
    ("both-present-equal", HEADER_TOK_TEMPLATE, HEADER_TOK_TEMPLATE_HASH, True, True),
    ("both-present-different", "{{ 'another template' }}", HEADER_TOK_TEMPLATE_HASH,
     True, False),
])
def test_a_lens_fallback_is_verified_only_when_the_comparison_ran_and_agreed(
        case, local, served, ran, verified):
    """E50 #6: the comparison used to be one expression — `local is not None
    and served is not None and local != served` — so `/v1/tokenize` down plus a
    pre-key `endpoint.json` (both null, the pair that occurs together) skipped
    the refusal entirely and the record was filed `verified: true` on locally
    invariant headers alone. `verified` now requires the comparison to have
    RUN; a null on either side refuses and names that side."""
    block, error = _header_block(local=local, served=served)
    comparison = block["template_comparison"]
    assert comparison["required"] is True and comparison["ran"] is ran
    assert block["verified"] is verified
    assert block["invariant"] is True            # the headers themselves never move here
    if verified:
        assert error is None and comparison["equal"] is True
        return
    assert isinstance(error, srun.HeaderMeasurementError)
    assert not isinstance(error, srun.MixedArmHeaderMoved)
    if ran:                                      # the pre-existing disagreement refusal
        assert "the renders measured are not the renders served" in str(error)
    else:                                        # the E50 #6 one, naming the side
        assert "CANNOT BE CHECKED" in str(error)
        for side in comparison["unavailable"]:
            assert side in str(error)
        assert comparison["reason"] and comparison["reason"] in str(error)


def test_the_comparison_is_required_only_where_it_licenses_something():
    """Scope, both ways. A SERVED measurement is the served render, so it needs
    no comparison and a pre-key endpoint.json cannot demote it; a non-lens
    route has no response mask to misalign, so a local fallback there is not
    held to the digest either. Neither is silently 'verified' by accident —
    the record says the comparison was not required."""
    def tokenize(body):
        ids = list(HeaderTok(moves=False).apply_chat_template(
            body["messages"], tokenize=True,
            add_generation_prompt=bool(body.get("add_generation_prompt")),
            **dict(body.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {})))
        return {"count": len(ids), "tokens": ids}

    live, error = srun.measure_thinking_header_by_arm(
        {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}, lens=True,
        tok=HeaderTok(moves=False, template=None), tokenize=tokenize, vmodel="petri__x",
        served_template_hash=None)
    assert error is None and live["source"] == srun.HEADER_SOURCE_SERVED
    assert live["template_comparison"]["required"] is False and live["verified"] is True
    block, error = _header_block(local=None, served=None, lens=False)
    assert error is None and block["verified"] is True
    assert block["template_comparison"]["required"] is False


def test_the_override_does_not_lift_an_uncheckable_template_comparison():
    """The override is about serving OUT OF SUPPORT, not about measuring
    nothing — the same reason it does not lift an unmeasurable header or a
    local template that disagrees (§PW-oos item 7). An uncheckable comparison
    is that third sibling, and the flag leaves it standing."""
    block, error = srun.measure_thinking_header_by_arm(
        {tc.GSM8K_STANDARD_TASK: "on", "inspect/or_bench_hard_1k": "off"}, lens=True,
        tok=HeaderTok(moves=True), served_template_hash=None,
        allow_out_of_support=True)
    assert isinstance(error, srun.HeaderMeasurementError)
    assert "CANNOT BE CHECKED" in str(error) and block["verified"] is False


def test_the_grader_cap_is_hashed_store_identity_and_only_under_tinycap():
    """The model GRADER's completion cap is a MEASUREMENT
    parameter — an uncapped grader can ramble past its own verdict (28 calls
    in the `c7e1796` wave ran the window out and six of those returned a
    600-second HTTP 504), and a capped one files inspect's unscored sentinel
    instead. Two waves of one cell graded at two caps are therefore two
    instruments, and the driver's resolved config is where that becomes
    unmistakable: `harness_config_hash` covers the dict whole, so the two land
    in different cell-key directories and can never resume onto or pool with
    each other.

    Two properties, both load-bearing. The cap moves the hash; and the block
    is written ONLY when a tinycap control is requested, so a non-tinycap
    served run's chash is byte-identical to what it was before this key
    existed (the same discipline `resolved["tinycap"]` itself is under)."""
    from sisteer.capeval import store as cs

    cell = SimpleNamespace(id="tinycap-x", object_path=None)
    route = SimpleNamespace(virtual_model="petri__none",
                            serving_identity="proxy_vllm+lens+interleaved")

    def resolved(cap, *, tinycap):
        return srun._resolved_config(cell, {"completion_tokens": 512}, None, 0, "judge",
                                     route=route, model_id="m", model_revision="r",
                                     grader_max_tokens=cap, tinycap=tinycap)

    ident = {"tasks": ["inspect/or_bench_hard_1k"]}
    capped, uncapped_era = resolved(4096, tinycap=ident), resolved(8192, tinycap=ident)
    assert capped["inspect_grader"] == {"max_tokens": 4096}
    assert uncapped_era["inspect_grader"] == {"max_tokens": 8192}
    assert cs.harness_config_hash(capped) != cs.harness_config_hash(uncapped_era)
    # nothing else moved: the cap is the only difference
    assert {k: v for k, v in capped.items() if k != "inspect_grader"} == {
        k: v for k, v in uncapped_era.items() if k != "inspect_grader"}
    assert cs.harness_config_hash(resolved(4096, tinycap=ident)) == cs.harness_config_hash(capped)
    # a run with NO tinycap control keys exactly as it always did
    legacy = resolved(4096, tinycap=None)
    assert "inspect_grader" not in legacy and "tinycap" not in legacy
    assert cs.harness_config_hash(legacy) == cs.harness_config_hash(
        resolved(999, tinycap=None))


def test_the_driver_resolves_the_grader_cap_once_from_inspect_served():
    """ONE number reaches both doors. The driver hashes the cap into the store
    identity and hands the SAME value to `run_inspect_tasks`, which builds the
    grader model with it — so "what was served" and "what the identity claims"
    cannot be two numbers. The resolver imports `inspect_served` lazily on
    purpose: that module pulls the eval stack, and a module-level default
    would drag it into every driver start-up."""
    from sisteer.capeval.tiers import inspect_served as isv

    assert srun._grader_max_tokens() == isv.GRADER_MAX_TOKENS == 4096
    assert srun._grader_max_tokens(1234) == 1234


def test_the_driver_has_no_errored_items_escape_hatch():
    """`--allow-errored-items` (09d52342) is gone from the driver and
    the contract: it published a k = 2 item's ONE-draw mean under a marker
    claiming k = 2, and the PARTIAL it wrote could never become DONE. An item
    that produced no generation refuses its task instead, which is what the
    driver had before the flag and what it has again. Re-introduction
    requirements: docs/tinycap_interfaces.md §2, "Deferred: errored-item
    completion"."""
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert not hasattr(tc, "FLAG_ALLOW_ERRORED_ITEMS")
    assert "allow_errored_items" not in src
    assert "--allow-errored-items" not in src
    assert "n_errored" not in tc.SUMMARY_PAYLOAD_FIELDS


def test_the_proxy_ledger_already_records_a_non_200_so_the_tier_does_not_duplicate_it():
    """E56 (3) asked whether the usage ledger would show a 504 that was never
    retried. It would, and this pins the two facts the answer rests on so the
    served tier's item-level refusal stays a DIFFERENT instrument rather than a
    second copy of this one.

    The ledger writes a row per FORWARDED REQUEST with the upstream status on
    it, marks any non-2xx `rejected` (so `stats()["n_rejected"]`, which the
    launcher pulls into its REC as `ledger_stats`, counts it whether or not the
    client retried and won), and classes a 5xx as `server_error`. What it
    cannot say is which ITEM was lost: a retried-and-answered 504 and a
    doubly-failed one look identical there — one extra row each — because the
    ledger counts requests and the tier counts items. Hence both."""
    from sisteer.capeval.serving import usage_ledger as ul

    assert ul.error_class_of(504, b"upstream timeout") == tc.LEDGER_ERROR_CLASS_SERVER_ERROR
    for field in ("status", "error_code", "rejected", "error_class", "latency_s"):
        assert field in tc.LEDGER_ROW_FIELDS, field
    # a rejected row increments the counter the REC reports
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = ul.LedgerWriter(Path(d) / "usage_ledger.jsonl")
        base = {k: None for k in tc.LEDGER_ROW_FIELDS}
        base.update(ts="2026-01-01T00:00:00+00:00", task="inspect/or_bench_hard_1k",
                    sample="or_bench_hard_1k_aaaa1111/0", vmodel="petri__base",
                    prompt_tokens_pre=585, has_tools=False, n_tools=0,
                    status=504, error_code="http_504", rejected=True,
                    error_class=tc.LEDGER_ERROR_CLASS_SERVER_ERROR, latency_s=600.7)
        w.write(base)
        w.write({**base, "status": 200, "error_code": None, "rejected": False,
                 "error_class": tc.LEDGER_ERROR_CLASS_NONE, "latency_s": 12.0})
        w.close()
        assert (w.n_rows, w.n_rejected) == (2, 1)


# ---------------------------------------------------------------------------- #
# the DECLARED thinking-ON out-of-support world — the landed                   #
# lens half's shapes, SYNTHETIC, exactly per contract-2.5.md (§2.5.1, §2.5.7,  #
# §2.5.10, §2.5.11, §2.5.12) — and its mutation checks                         #
# ---------------------------------------------------------------------------- #

OOS_OWN_KEY = "si_safety__psr_kl_all@thinking"          # the attestation key (harness, variant spelling)
OOS_VMODEL = "petri__si_safety__psr_kl_all@thinking"    # the vmodel the standup serves the cell as
OOS_DECL = {tc.SERVE_OUT_OF_SUPPORT_KEY: ["thinking"]}
OOS_MAP = {OOS_OWN_KEY: dict(OOS_DECL), tc.TWIN_CELL_ID: dict(OOS_DECL)}     # §2.5.7, verbatim
OOS_WINDOW, OOS_SEQS = 16384, 22                        # the pass-1 geometry the launch resolves (P2-b)
OOS_THINKING_HEADER = {"switch": True, "moves": True, "off_ids": list(tc.QWEN35_HEADER_IDS_OFF),
                       "on_ids": list(tc.QWEN35_HEADER_IDS_ON)}


class QwenHeaderTok:
    """Qwen3.5's two measured header regimes with the REAL ids (docs §11;
    ``contracts.QWEN35_HEADER_IDS_*``): OFF closes an empty think block inside
    the header (H=7), ON leaves it open (H=5). The body's length grows with the
    messages' content so the 200 prompts have distinct lengths (the live
    prompt-budget pass compares them item by item); ``extra_body_ids`` spells a
    served tokenizer that renders longer than the pinned one (a drift), and
    ``on_ids`` a header off its measured value."""

    chat_template = HEADER_TOK_TEMPLATE

    def __init__(self, *, on_ids=tc.QWEN35_HEADER_IDS_ON, off_ids=tc.QWEN35_HEADER_IDS_OFF,
                 extra_body_ids: int = 0):
        self.on_ids, self.off_ids, self.extra = tuple(on_ids), tuple(off_ids), int(extra_body_ids)

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True,
                            enable_thinking=False, **kw):
        body = [7] * (sum(len(m["content"]) for m in messages) // 4 + 2 + self.extra)
        if not add_generation_prompt:
            return body
        return body + list(self.on_ids if enable_thinking else self.off_ids)


def _declare_thinking_route(table_path: Path) -> dict:
    """Re-spell the exported table's ``si_safety__psr_kl_all`` route as the
    landed standup exports it on the psr/ON standup (contract §2.5.2): cell id
    and vmodel in the ``@thinking`` variant spelling, a DIFFERENT intervention
    hash (the hashed spec carries the declaration), no bare route beside it."""
    doc = json.loads(table_path.read_text())
    for r in doc["routes"]:
        if r["cell_id"] == "si_safety__psr_kl_all":
            r["cell_id"], r["virtual_model"] = OOS_OWN_KEY, OOS_VMODEL
            r["intervention_hash"] = r["intervention_hash"][::-1]
    table_path.write_text(json.dumps(doc))
    return {r.cell_id: r for r in px.load_routing_table(table_path)["routes"]}


def _oos_messages() -> list:
    """200 SYNTHETIC GSM8K-shaped message lists in plan order (one system turn,
    one user turn), what ``inspect_served.gsm8k_paired200_messages`` renders."""
    return [[{"role": "system", "content": "As an expert problem solver, solve step by step the "
                                            "following mathematical questions. " + "Example. " * 12},
             {"role": "user", "content": f"Question {i}: a synthetic word problem " + "x" * (i % 37)}]
            for i in range(tc.GSM8K_PAIRED_N)]


def _write_budget(tmp_path: Path, tok, messages, *, cap: int = 8192, arm: str = tc.THINKING_ON,
                  template_hash: str = TEMPLATE_HASH, model_slug: str = tc.AUTHORISED_OOS_THINKING.model,
                  name: str = tc.OOS_PROMPT_BUDGET_FILENAME) -> Path:
    """The CPU prompt-budget artifact (``contracts.OOS_PROMPT_BUDGET_FIELDS``,
    validated) as ``scripts/tinycap_oos_prompt_budget.py`` writes it from the
    PINNED tokenizer under the ``arm`` render (the ON render by default)."""
    on = tc.enable_thinking(arm)
    lengths = [len(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True, enable_thinking=on))
               for m in messages]
    doc = {"n": len(lengths), "max": max(lengths), "p99": tc.nearest_rank_percentile(lengths, 0.99),
           "p50": tc.nearest_rank_percentile(lengths, 0.50), "mean": round(sum(lengths) / len(lengths), 2),
           "arm": arm, "model": BASE_MODEL, "model_slug": model_slug,
           "revision": REV, "chat_template_hash": template_hash, "cap": int(cap),
           "cap_task": tc.GSM8K_STANDARD_TASK, "max_plus_cap": max(lengths) + int(cap),
           "lengths": lengths,
           "render": {"add_generation_prompt": True, **tc.chat_template_kwargs(arm)},
           "messages_source": tc.OOS_PROMPT_BUDGET_MESSAGES_SOURCE, "tree_sha": None,
           "produced_by": {"script": tc.OOS_PROMPT_BUDGET_SCRIPT, "tokenizer": "QwenHeaderTok"}}
    tc.validate_oos_prompt_budget(doc)
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def oos_on_world(world, monkeypatch, *, frameworks: str = "fake", declarations=None,
                 endpoint_extra=None, identity_extra=None, tok=None, chat_regimes=None,
                 budget_cap: int = 8192, with_digest: bool = True, with_budget: bool = True):
    """The authorised psr/ON configuration on a LANDED standup: the routing
    table serves the cell under its ``@thinking`` spelling; ``endpoint.json``
    attests the two-key declarations map (error null), the served digest (scalar
    + per-cell), ``lens_salt_policy: attested``, the per-cell ``conditional``
    policy under the VARIANT key and the pass-1 window; the live proxy's
    identity block carries the same map plus the certified ``thinking_header``
    (error null); the served tokenizer renders Qwen3.5's real header ids; the
    200 exact GSM8K messages are stubbed and their CPU budget artifact written
    from the pinned tokenizer; the driver argv carries the two new flags.
    Every keyword spells ONE mutation of that world."""
    from sisteer.capeval.tiers import inspect_served as isv

    (fake_frameworks if frameworks == "fake" else stub_framework_versions)(monkeypatch)
    _pin_the_authorised_oos_model(monkeypatch)
    routes = _declare_thinking_route(world.table_path)
    route, twin = routes[OOS_OWN_KEY], routes[tc.TWIN_CELL_ID]
    tok = tok or QwenHeaderTok()
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: tok)
    messages = _oos_messages()
    monkeypatch.setattr(isv, "gsm8k_paired200_messages", lambda repo_root=None, system=None: [list(m) for m in messages])
    budget = _write_budget(world.tmp_path, QwenHeaderTok(), messages, cap=budget_cap)   # the PINNED tokenizer
    declarations = OOS_MAP if declarations is None else declarations
    ep_extra = {tc.PROXY_IDENTITY_SERVE_DECLARATIONS: declarations,
                tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {OOS_OWN_KEY: "conditional",
                                                              tc.TWIN_CELL_ID: "conditional"},
                tc.ENDPOINT_KEY_LENS_SALT_POLICY: tc.LENS_SALT_POLICY_ATTESTED,
                tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {OOS_OWN_KEY: OBJECT_SHA},
                "geometry": standup_geometry(OOS_WINDOW, OOS_SEQS),
                "max_model_len": OOS_WINDOW, "max_num_seqs": OOS_SEQS,
                **(endpoint_extra or {})}
    write_standup_endpoint(world.table_path, **ep_extra)
    identity = {**LIVE_IDENTITY,
                tc.PROXY_IDENTITY_SERVE_DECLARATIONS: ep_extra[tc.PROXY_IDENTITY_SERVE_DECLARATIONS],
                "thinking_header": dict(OOS_THINKING_HEADER), "thinking_header_error": None,
                **(identity_extra or {})}
    # the declared standup: the steered cell AND its twin are in the map, so the
    # landed proxy labels both legs `declared-request-unique`
    chat = FakeChat({route.virtual_model: "steered voice says alpha",
                     twin.virtual_model: "plain voice says beta"}, steered_model=route.virtual_model,
                    declared_models={route.virtual_model, twin.virtual_model})
    if chat_regimes is not None:
        chat.regime_stamps = chat_regimes
    tokenized: list = []

    def recording_tokenize(body):
        tokenized.append(json.loads(json.dumps(body)))
        return live_tokenize(body)

    monkeypatch.setattr(srun, "_make_client",
                        lambda url, rt: SimpleNamespace(chat_completions=chat.chat_completions,
                                                        tokenize=recording_tokenize,
                                                        proxy_identity=identity))
    argv = _oos_argv(world)
    if with_digest:
        argv += [tc.FLAG_EXPECT_OBJECT_SHA256, OBJECT_SHA]
    if with_budget:
        argv += [tc.FLAG_OOS_PROMPT_BUDGET, str(budget)]
    return SimpleNamespace(route=route, twin=twin, chat=chat, argv=argv, budget=budget,
                           messages=messages, identity=identity, tok=tok, tokenized=tokenized)


def _never_runs(monkeypatch):
    from sisteer.capeval.tiers import inspect_served

    async def fake_run(**kw):
        raise AssertionError("a refused out-of-support run must never generate")

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)


def _off_render(payload):
    return {**_landed_stamps(payload, declared=True), tc.STAMP_KEY_RENDER_REGIME: tc.RENDER_REGIME_THINKING_OFF}


def _undeclared_policy(payload):
    return {**_landed_stamps(payload, declared=True), tc.STAMP_KEY_RESUMED_POLICY: tc.RESUMED_PREFILL_CONDITIONAL}


OOS_MUTATIONS = {
    # the attestation compare (contract §2.5.1, §2.5.7, §2.5.10)
    "own_key_wrong_value": (dict(declarations={OOS_OWN_KEY: {tc.SERVE_OUT_OF_SUPPORT_KEY: ["thinking", "other"]},
                                               tc.TWIN_CELL_ID: dict(OOS_DECL)}),
                            [tc.REC_OOS_DECLARATION_MISMATCH, OOS_OWN_KEY]),
    "own_key_experiment_shape": (dict(declarations={OOS_OWN_KEY: {tc.SERVE_EXPERIMENT_KEY: "x"},
                                                    tc.TWIN_CELL_ID: dict(OOS_DECL)}),
                                 [tc.REC_OOS_DECLARATION_MISMATCH, tc.SERVE_EXPERIMENT_KEY]),
    "twin_key_missing": (dict(declarations={OOS_OWN_KEY: dict(OOS_DECL)}),
                         [tc.REC_OOS_DECLARATION_MISMATCH, tc.TWIN_CELL_ID]),
    "own_key_missing": (dict(declarations={tc.TWIN_CELL_ID: dict(OOS_DECL)}),
                        [tc.REC_OOS_DECLARATION_UNATTESTED, OOS_OWN_KEY]),
    "bare_key_instead_of_variant": (dict(declarations={"si_safety__psr_kl_all": dict(OOS_DECL),
                                                       tc.TWIN_CELL_ID: dict(OOS_DECL)}),
                                    [tc.REC_OOS_DECLARATION_UNATTESTED, OOS_OWN_KEY]),
    "error_key_non_null": (dict(endpoint_extra={tc.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: "workers disagree"}),
                           [tc.REC_OOS_DECLARATION_UNATTESTED, "workers disagree"]),
    "map_absent_pre_landing": (dict(endpoint_extra={tc.PROXY_IDENTITY_SERVE_DECLARATIONS: None}),
                               [tc.REC_OOS_DECLARATION_UNATTESTED]),
    "live_map_differs_from_record": (dict(identity_extra={tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {}}),
                                     ["does not vouch", tc.PROXY_IDENTITY_SERVE_DECLARATIONS]),
    "object_sha_differs": (dict(endpoint_extra={tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "cd" * 32}),
                           [tc.REC_OOS_DECLARATION_MISMATCH, OBJECT_SHA]),
    "by_cell_bare_key": (dict(endpoint_extra={tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL:
                                              {"si_safety__psr_kl_all": OBJECT_SHA}}),
                         [tc.REC_OOS_DECLARATION_MISMATCH, tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL]),
    "policy_always": (dict(endpoint_extra={tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS:
                                           {OOS_OWN_KEY: "always", tc.TWIN_CELL_ID: "conditional"}}),
                      [tc.REC_OOS_SALT_POLICY_UNEXPECTED, "always"]),
    "policy_under_bare_key": (dict(endpoint_extra={tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS:
                                                   {"si_safety__psr_kl_all": "conditional"}}),
                              [tc.REC_OOS_SALT_POLICY_UNEXPECTED, OOS_OWN_KEY]),
    "policy_map_absent": (dict(endpoint_extra={tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: None}),
                          [tc.REC_OOS_SALT_POLICY_UNEXPECTED]),
    "salt_policy_forced": (dict(endpoint_extra={tc.ENDPOINT_KEY_LENS_SALT_POLICY: "request_unique_all"}),
                           [tc.REC_OOS_SALT_POLICY_UNEXPECTED, tc.FORCED_SALT_POLICY_LABEL]),
    "template_hash_differs": (dict(endpoint_extra={srun.STANDUP_TEMPLATE_HASH_KEY: "34" * 8}),
                              [tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH, "34" * 8, TEMPLATE_HASH]),
    "no_digest_flag": (dict(with_digest=False),
                       [tc.REC_OOS_DECLARATION_UNATTESTED, tc.FLAG_EXPECT_OBJECT_SHA256]),
    "no_budget_flag": (dict(with_budget=False),
                       [tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH, tc.FLAG_OOS_PROMPT_BUDGET]),
    # S3.4 — the header + render-fact certification (§2.5.1, §2.5.7, §2.5.12)
    "switch_is_a_string": (dict(identity_extra={"thinking_header": {**OOS_THINKING_HEADER, "switch": "enable_thinking"}}),
                           [tc.REC_THINKING_UNSERVABLE, "switch = 'enable_thinking' (str)"]),
    "switch_is_1": (dict(identity_extra={"thinking_header": {**OOS_THINKING_HEADER, "switch": 1}}),
                    [tc.REC_THINKING_UNSERVABLE, "(int)"]),
    "moves_false": (dict(identity_extra={"thinking_header": {**OOS_THINKING_HEADER, "moves": False}}),
                    [tc.REC_THINKING_UNSERVABLE, "moves = False"]),
    "partial_fact_with_error": (dict(identity_extra={"thinking_header": {**OOS_THINKING_HEADER, "moves": None, "on_ids": None},
                                                     "thinking_header_error": "ON prompt is not a prefix"}),
                                [tc.REC_THINKING_UNSERVABLE, "ON prompt is not a prefix"]),
    "attested_on_ids_length_6": (dict(identity_extra={"thinking_header": {**OOS_THINKING_HEADER,
                                                                          "on_ids": list(tc.QWEN35_HEADER_IDS_ON) + [271]}}),
                                 [tc.REC_THINKING_UNSERVABLE, "on_ids"]),
    "no_thinking_header_block": (dict(identity_extra={"thinking_header": None}),
                                 [tc.REC_THINKING_UNSERVABLE, "no petri_proxy.thinking_header"]),
    "served_header_off_by_one_id": (dict(tok=QwenHeaderTok(on_ids=(*tc.QWEN35_HEADER_IDS_ON[:-1], 199))),
                                    [tc.REC_THINKING_UNSERVABLE, "open-think token"]),
    # S3.5 / S3.3 — the probe's stamps (§2.5.11 rows 1–2)
    "probe_rendered_off": (dict(chat_regimes=_off_render), ["probe leg", "thinking_off"]),
    "probe_leg_not_declared": (dict(chat_regimes=_undeclared_policy), ["probe leg", "conditional"]),
    # S3.8 — the live prompt budget
    "window_too_small": (dict(endpoint_extra={"geometry": standup_geometry(4096, 4), "max_model_len": 4096,
                                              "max_num_seqs": 4}),
                         [tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH, "over by"]),
    "live_lengths_drift": (dict(tok=QwenHeaderTok(extra_body_ids=1)),
                           [tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH, "differ from the CPU artifact"]),
}


@pytest.mark.parametrize("name", sorted(OOS_MUTATIONS))
def test_every_mutation_of_the_declared_world_refuses_before_generation_naming_its_rec(
        world, monkeypatch, name):
    """T3's mutation matrix on the driver, one mutation per case: each ends the
    run with a ``SystemExit`` naming its REC and the values involved,
    BEFORE any task is generated; a header certification failure leaves
    evidence.json with every certification flag false first (the S3.4
    discipline)."""
    kw, needles = OOS_MUTATIONS[name]
    oos = oos_on_world(world, monkeypatch, **kw)
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(oos.argv)
    msg = str(e.value)
    for needle in needles:
        assert needle in msg, (name, needle, msg)
    if needles[0] == tc.REC_THINKING_UNSERVABLE:
        block = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["thinking_header_by_arm"]
        for flag in (tc.OOS_EVIDENCE_HEADER_CERTIFIED, tc.OOS_EVIDENCE_HEADER_ATTESTED_AGREES,
                     tc.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED):
            assert block[flag] is False, (name, flag)
        assert block["certification_problems"]
        if name == "switch_is_a_string":
            assert block[tc.OOS_EVIDENCE_ATTESTED_SWITCH] == "enable_thinking"
    if needles[0] == tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH and name in ("window_too_small", "live_lengths_drift"):
        live = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK]
        assert live["ok"] is False and live["n"] == 200                 # written first, then refused
    if "probe" in name:
        bites = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
        assert bites["verified"] is False and "probe leg" in bites["stamp_error"]
        assert bites[tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == tc.THINKING_ON


def test_an_unstamped_probe_is_tolerated_in_support_and_refused_out_of_support(world, monkeypatch):
    """Phase B integration — the additive-key rule at the bites probe (S3.3):
    against today's PRE-LANDING proxy (no regime block on any response) an
    in-support thinking-arm run records the all-null triples in
    ``stamps_by_leg`` and stays verified (nothing about the run depends on the
    stamp); a PRESENT wrong value still refuses in support; the out-of-support
    run refuses the absent stamp, naming the fields and the override."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)

    async def fake_run(**kw):
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    world.chat.regime_stamps = lambda payload: {}                       # a pre-landing proxy
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    bites = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
    assert bites["verified"] is True and "stamp_error" not in bites
    null = {f: None for f in tc.EXPECTED_STAMP_TRIPLE_FIELDS}
    for leg in ("steered", "scale0"):
        assert bites["stamps_by_leg"][leg] and all(s == null for s in bites["stamps_by_leg"][leg])
    # a PRESENT wrong value refuses in support too
    (world.out_root / "cells").rename(world.out_root / "cells_null")
    world.chat.regime_stamps = lambda payload: {**_landed_stamps(payload, declared=False),
                                                tc.STAMP_KEY_RENDER_REGIME: tc.RENDER_REGIME_THINKING_ON}
    with pytest.raises(SystemExit):
        srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    bites = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
    assert bites["verified"] is False and "['render_regime']" in bites["stamp_error"]
    # out of support: the absent stamp refuses, by name
    (world.out_root / "cells").rename(world.out_root / "cells_wrong")
    oos = oos_on_world(world, monkeypatch, chat_regimes=lambda payload: {})
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(oos.argv)
    assert "carries no value for" in str(e.value) and tc.FLAG_ALLOW_OOS_THINKING in str(e.value)
    bites = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
    assert bites["verified"] is False and "carries no value for ['render_regime', 'resumed_policy', 'salt_regime']" in bites["stamp_error"]


def test_a_probe_that_drops_the_declaration_is_refused_by_the_landed_proxy(world, monkeypatch):
    """S3.3 mutation: were the ONE predicate to return {} on the ON probe, the
    body would go out undeclared and the landed proxy would 400 it
    (``thinking_on_undeclared``, contract §2.5.6 — the cell IS declared, so the
    label would not move; the refusal is the proxy's, surfaced as the client's
    ProxyError) before anything is generated. A leg whose stamps read
    ``conditional`` on the declared standup is the stamp check's to refuse
    (the ``probe_leg_not_declared`` mutation)."""
    oos = oos_on_world(world, monkeypatch)
    _never_runs(monkeypatch)
    monkeypatch.setattr(tc, "thinking_on_declaration", lambda *, lens, arm, header_located=True: {})
    with pytest.raises(px.ProxyError, match="thinking_on_undeclared"):
        srun.main(oos.argv)
    probe_calls = [c for c in oos.chat.calls if not c.get("continue_final_message")]
    assert probe_calls and all(tc.OOS_THINKING_DECLARATION_KEY not in c for c in probe_calls)


def test_the_declared_run_refuses_before_sending_while_the_declaration_constants_read_none(world, monkeypatch):
    """A mutated module (the request-side constants None) must refuse
    BEFORE the ON probe goes out — an ON probe without its declaration would
    certify the OFF render."""
    oos = oos_on_world(world, monkeypatch)
    _never_runs(monkeypatch)
    monkeypatch.setattr(tc, "OOS_THINKING_DECLARATION_KEY", None)
    with pytest.raises(SystemExit, match="declaration constants are not landed"):
        srun.main(oos.argv)
    assert not [c for c in oos.chat.calls if not c.get("continue_final_message")]


def test_the_in_support_off_probe_sends_explicit_off_kwargs_and_no_declaration(world, monkeypatch):
    """§3.1 / §2.5.11 rows 3–4: the psr/OFF standup's probe — an
    IN-SUPPORT run — now sends ``enable_thinking: false`` explicitly on both
    legs and NO declaration, so its stamp reads ``thinking_off`` / ``conditional``;
    the legacy (no-arm) probe stays byte-identical to before."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)

    async def fake_run(**kw):
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    probe_calls = [c for c in world.chat.calls if not c.get("continue_final_message")]
    assert {c["model"] for c in probe_calls} == {world.route.virtual_model, world.twin.virtual_model}
    for c in probe_calls:
        assert c[tc.CHAT_TEMPLATE_KWARGS_KEY] == {tc.ENABLE_THINKING_KEY: False}
        assert tc.OOS_THINKING_DECLARATION_KEY not in c
    bites = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
    assert bites[tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == tc.THINKING_OFF and bites["verified"] is True
    row = tc.expected_stamp_regimes(lens=True, arm=tc.THINKING_OFF)._asdict()
    assert row["resumed_policy"] == "conditional" and row["render_regime"] == "thinking_off"
    for leg in ("steered", "scale0"):
        assert set(map(json.dumps, bites["stamps_by_leg"][leg])) == {json.dumps(row)}
    # the route block of an in-support run: the empty map, the digest, no compare on the rest
    manifest = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())
    assert manifest["route"][tc.ROUTE_KEY_SERVE_DECLARATIONS] == {}
    assert manifest["route"][tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] is None      # no --expect-object-sha256: not compared
    assert manifest["route"]["resumed_prefill_policy"] == "conditional"         # the cell's ATTESTED policy, recorded
    assert set(manifest["route"]) == set(tc.ROUTE_BLOCK_FIELDS)
    # the legacy generation run (no thinking arm) probes byte-identically: no kwargs, no stamp check
    world.chat.calls.clear()
    (world.out_root / "cells").rename(world.out_root / "cells_off")
    srun.main(world.argv)
    legacy = [c for c in world.chat.calls if not c.get("continue_final_message")]
    assert legacy and all(tc.CHAT_TEMPLATE_KWARGS_KEY not in c and tc.OOS_THINKING_DECLARATION_KEY not in c
                          for c in legacy)
    ev = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
    assert ev[tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] is None and ev["probe_extra_body"] == {}


def test_a_pre_landing_standup_is_tolerated_in_support_and_refused_out_of_support(world, monkeypatch):
    """The additive-key rule the launcher's worker applies too (B2): a standup
    record without the declarations map (TODAY's pre-landing shape) is
    tolerated on an IN-SUPPORT run — recorded null in the route block, no
    compare — and refuses an out-of-support run (OOS_DECLARATION_UNATTESTED,
    fail closed) before any store is touched. A PRESENT disagreeing fact (the
    error key non-null) refuses in support too."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    path = world.table_path.parent / srun.STANDUP_ENDPOINT_FILENAME
    ep = json.loads(path.read_text())
    del ep[tc.PROXY_IDENTITY_SERVE_DECLARATIONS]
    path.write_text(json.dumps(ep))
    ident = dict(LIVE_IDENTITY)
    del ident[tc.PROXY_IDENTITY_SERVE_DECLARATIONS]
    monkeypatch.setattr(srun, "_make_client",
                        lambda url, rt: SimpleNamespace(chat_completions=world.chat.chat_completions,
                                                        tokenize=live_tokenize, proxy_identity=ident))

    async def fake_run(**kw):
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    manifest = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())
    assert manifest["route"][tc.ROUTE_KEY_SERVE_DECLARATIONS] is None       # tolerated, recorded null
    # the same pre-landing record under the override — the authorised psr/OFF
    # configuration, which needs no budget artifact, so the ABSENT map is the
    # one thing refused: OOS_DECLARATION_UNATTESTED, nothing written
    (world.out_root / "cells").rename(world.out_root / "cells_in_support")
    _pin_the_authorised_oos_model(monkeypatch)
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit, match=tc.REC_OOS_DECLARATION_UNATTESTED):
        srun.main(_oos_argv(world, thinking=tc.THINKING_OFF) + [tc.FLAG_EXPECT_OBJECT_SHA256, OBJECT_SHA])
    assert not (world.out_root / "cells").exists()
    # a PRESENT disagreeing fact refuses in support as well
    ep[tc.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR] = "workers disagree"
    path.write_text(json.dumps(ep))
    monkeypatch.setitem(sg.MODEL_HF_IDS, "llama8b", BASE_MODEL)
    with pytest.raises(SystemExit, match=tc.REC_OOS_DECLARATION_UNATTESTED):
        srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])


# --------------------------------------------------------------------------- #
# review round B, P1: in-support runs are held to the cell's ATTESTED policy,  #
# never to the authorised experiment's §2.5.11 rows                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("policy,arm", [("always", tc.THINKING_OFF), ("never", tc.THINKING_OFF),
                                        ("conditional", tc.THINKING_ON)])
def test_in_support_runs_are_held_to_the_cells_attested_policy_not_the_experiments_rows(
        world, monkeypatch, policy, arm):
    """Review round B, P1 (the driver, end to end). The standup attests the
    fixture cell (and its twin) at ``policy``; TODAY's proxy stamps every probe
    response with the two salt fields that policy implies (``cell-stable`` for
    an all-position ``always`` cell, ``request-unique`` otherwise) and NO
    ``render_regime``. Before this fix the probe held both legs to the
    experiment's rows (``conditional`` under OFF, ``declared-request-unique``
    under ON) and every such campaign run failed its positive control; now the
    run passes, the probe records the attested triples (null render), the
    provider is handed the same label, and the manifest's route block records
    the attested policy. A PRESENT disagreeing label still refuses in support."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    extra = {tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {"si_safety__psr_kl_all": policy,
                                                          tc.TWIN_CELL_ID: policy}}
    if arm == tc.THINKING_ON:
        extra["reasoning_parser"] = "gemma4"                      # an in-support ON standup composes one
        monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: HeaderTok(moves=False))
    write_standup_endpoint(world.table_path, **extra)
    salt = tc.SALT_REGIME_CELL_STABLE if policy == "always" else tc.SALT_REGIME_REQUEST_UNIQUE
    world.chat.regime_stamps = lambda payload: {tc.STAMP_KEY_SALT_REGIME: salt,
                                                tc.STAMP_KEY_RESUMED_POLICY: policy}
    handed = {}

    async def fake_run(**kw):
        handed.update(kw)
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, arm])
    bites = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
    assert bites["verified"] is True and "stamp_error" not in bites, bites.get("stamp_error")
    want = {"salt_regime": salt, "resumed_policy": policy, "render_regime": None}
    for leg in ("steered", "scale0"):
        assert bites["stamps_by_leg"][leg] and all(s == want for s in bites["stamps_by_leg"][leg]), leg
    assert handed["stamp_policy"] == policy                       # the provider gets the SAME label
    assert handed["out_of_support"] is False
    manifest = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())
    assert manifest["route"]["resumed_prefill_policy"] == policy
    # the experiment's row on this cell is NOT what the leg is held to — a probe stamped with it refuses
    (world.out_root / "cells").rename(world.out_root / "cells_ok")
    strict = tc.expected_stamp_regimes(lens=True, arm=arm)
    if strict.resumed_policy != policy:
        world.chat.regime_stamps = lambda payload: {tc.STAMP_KEY_SALT_REGIME: strict.salt_regime,
                                                    tc.STAMP_KEY_RESUMED_POLICY: strict.resumed_policy}
        with pytest.raises(SystemExit, match="fields \\['resumed_policy'(, 'salt_regime')?\\]"):
            srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, arm])


def test_a_standup_attesting_no_policy_map_is_held_to_unattested_in_support(world, monkeypatch):
    """The record from BEFORE the per-cell attestation (the key absent): the
    proxy labels every lens leg ``unattested`` / ``request-unique`` and that is
    the row the run is held to; the route block records no policy."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)
    write_standup_endpoint(world.table_path, **{tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: ABSENT})
    assert tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS not in json.loads(
        (world.table_path.parent / srun.STANDUP_ENDPOINT_FILENAME).read_text())
    world.chat.regime_stamps = lambda payload: {tc.STAMP_KEY_SALT_REGIME: tc.SALT_REGIME_REQUEST_UNIQUE,
                                                tc.STAMP_KEY_RESUMED_POLICY: tc.POLICY_UNATTESTED}
    handed = {}

    async def fake_run(**kw):
        handed.update(kw)
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    bites = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
    assert bites["verified"] is True and handed["stamp_policy"] == tc.POLICY_UNATTESTED
    manifest = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())
    assert manifest["route"]["resumed_prefill_policy"] is None            # the map absent: nothing to record
    # ... while a stamp claiming an attested policy against that record refuses
    (world.out_root / "cells").rename(world.out_root / "cells_unattested")
    world.chat.regime_stamps = lambda payload: _landed_stamps(payload, declared=False)   # `conditional`
    with pytest.raises(SystemExit, match="fields \\['resumed_policy'\\]"):
        srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])


def test_the_authorised_legs_keep_the_strict_rows_whatever_the_record_attests(world, monkeypatch):
    """The other half of P1: on the AUTHORISED psr/ON leg the expected row is
    the contract's (``declared-request-unique``) — a probe stamped with the
    cell's attested ``conditional`` refuses, as before (the existing mutation
    ``probe_leg_not_declared`` pins it); and a malformed digest argument is an
    argparse refusal naming the flag, never a contract MISMATCH."""
    oos = oos_on_world(world, monkeypatch, chat_regimes=_undeclared_policy)
    _never_runs(monkeypatch)
    argv = list(oos.argv)
    argv[argv.index(tc.FLAG_EXPECT_OBJECT_SHA256) + 1] = "not-a-sha"
    with pytest.raises(SystemExit) as e:
        srun.main(argv)
    assert e.value.code == 2                                              # argparse, before any compare
    assert not (world.out_root / "cells").exists()
    with pytest.raises(SystemExit, match="probe leg"):
        srun.main(oos.argv)


def test_a_non_mapping_thinking_header_block_writes_the_complete_evidence_block(world, monkeypatch):
    """The certification's non-mapping branch used to write a PARTIAL
    evidence block (three flags and the problem); now it is the same COMPLETE
    block every other mismatch writes — every ``OOS_EVIDENCE_*`` key present,
    the attested values typed null, the block recorded verbatim."""
    oos = oos_on_world(world, monkeypatch, identity_extra={"thinking_header": "yes"})
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit, match=tc.REC_THINKING_UNSERVABLE):
        srun.main(oos.argv)
    block = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["thinking_header_by_arm"]
    for flag in (tc.OOS_EVIDENCE_HEADER_CERTIFIED, tc.OOS_EVIDENCE_HEADER_ATTESTED_AGREES,
                 tc.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED):
        assert block[flag] is False
    assert block[tc.OOS_EVIDENCE_ATTESTED_SWITCH] is None and block[tc.OOS_EVIDENCE_ATTESTED_MOVES] is None
    assert block["attested_thinking_header"] == "yes" and block["thinking_header_error"] is None
    assert block["expected_header_ids_by_arm"] == {"off": list(tc.QWEN35_HEADER_IDS_OFF), "on": list(tc.QWEN35_HEADER_IDS_ON)}
    assert any("not a mapping" in p for p in block["certification_problems"])


@pytest.mark.parametrize("bad", [None, 1, True, "", [], {}, "sometimes"])
def test_a_present_malformed_probe_stamp_refuses_in_support_too(world, monkeypatch, bad):
    """A PRESENT malformed regime field on a probe response
    (``petri.render_regime: null`` / ``1`` / ``true`` / ``""`` / ``[]`` / ``{}``
    / off-vocabulary) collapsed to None and was tolerated in support as an
    absent stamp; the run finished and persisted null regime evidence. Now the
    extraction refuses it, the probe RECORDS the refusal per leg
    (``stamp_errors_by_leg``), and the positive control fails on every run,
    naming the key, the value and its type."""
    from sisteer.capeval.tiers import inspect_served

    fake_frameworks(monkeypatch)

    async def fake_run(**kw):
        raise AssertionError("a run with a malformed stamp must never generate")

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    world.chat.regime_stamps = lambda payload: {**_landed_stamps(payload, declared=False),
                                                tc.STAMP_KEY_RENDER_REGIME: bad}
    with pytest.raises(SystemExit, match="POSITIVE CONTROL FAILED") as e:
        srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    msg = str(e.value)
    assert f"{tc.STAMP_BLOCK_KEY}.{tc.STAMP_KEY_RENDER_REGIME}" in msg and repr(bad) in msg
    assert type(bad).__name__ in msg
    bites = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["steering_bites"]
    assert bites["verified"] is False and bites["stamp_error"] in msg
    assert bites["stamp_errors_by_leg"]["steered"] and all(
        repr(bad) in err for leg in bites["stamp_errors_by_leg"].values() for err in leg)


def test_probe_stamp_problem_reports_a_recorded_extraction_error_first_whatever_the_support():
    """The probe's judge: a leg with a recorded extraction refusal is refused
    before any row comparison, under both readings of the override; with no
    errors the row comparison runs as before."""
    row = tc.expected_stamp_regimes(lens=True, arm=tc.THINKING_OFF, policy="conditional")._asdict()
    stamps = {"steered": [row] * 2, "scale0": [row] * 2}
    policies = {"steered": "conditional", "scale0": "conditional"}
    for oos in (False, True):
        assert srun._probe_stamp_problem(stamps, tc.THINKING_OFF, out_of_support=oos, policies=policies,
                                         errors_by_leg={"steered": [], "scale0": []}) is None
        why = srun._probe_stamp_problem({"steered": [], "scale0": [row] * 2}, tc.THINKING_OFF,
                                        out_of_support=oos, policies=policies,
                                        errors_by_leg={"steered": ["petri.render_regime = None (NoneType) ..."],
                                                       "scale0": []})
        assert why and "probe leg 'steered'" in why and "petri.render_regime = None (NoneType)" in why


# --------------------------------------------------------------------------- #
# offcap: the thinking-OFF comparator at the thinking-ON cap                   #
# (`--allow-off-cap-override`) — the in-support OFF                            #
# world on a LANDED standup, held to the landed record, planned at cap 8192    #
# --------------------------------------------------------------------------- #

OFFCAP_CAP = tc.OFF_CAP_OVERRIDE_MAX_TOKENS
_PSR_BARE_KEY = "si_safety__psr_kl_all"                  # the OFF standup's attestation key: BARE


def _write_budget_off(tmp_path: Path, tok, messages, *, cap: int = OFFCAP_CAP, arm: str = tc.THINKING_OFF,
                      template_hash: str = TEMPLATE_HASH) -> Path:
    """The CPU prompt-budget artifact the comparator's preflight writes
    (`scripts/tinycap_oos_prompt_budget.py --arm off --off-cap-override`): the 200
    messages tokenized under the OFF render of the PINNED tokenizer, `arm off`,
    `cap` the OVERRIDE cap. `arm` / `cap` spell a wrong artifact."""
    return _write_budget(tmp_path, tok, messages, cap=cap, arm=arm, template_hash=template_hash,
                         model_slug=tc.AUTHORISED_OFF_CAP_COMPARATOR.model,
                         name=f"{arm}_{cap}_{tc.OOS_PROMPT_BUDGET_FILENAME}")


def offcap_world(world, monkeypatch, *, cell: str = TT_CELL, frameworks: str = "fake",
                 with_budget: bool = True, with_digest: bool = True,
                 budget_arm: str = tc.THINKING_OFF, budget_cap: int = OFFCAP_CAP,
                 endpoint_extra=None, base_probe=None, out_root=None):
    """The comparator's configuration for `cell` on a LANDED standup: the
    in-support OFF world (bare spellings, the bare route, the undeclared map) plus
    the landed record fields the comparator is HELD to — `lens_salt_policy
    attested`, the digest scalar + per-cell under the BARE key, the bare cell's
    `conditional` policy (the fixture default), the pass-1 window — the OFF
    artifact at the override cap, the 200 messages stubbed, the served tokenizer
    rendering the real header ids, the base route answering the declaration
    probe with the landed 400. argv = the OFF run + `--allow-off-cap-override`
    (+ the digest on the lens cell, + the artifact). Every keyword spells ONE
    mutation."""
    from sisteer.capeval.tiers import inspect_served as isv

    (fake_frameworks if frameworks == "fake" else stub_framework_versions)(monkeypatch)
    _pin_the_authorised_oos_model(monkeypatch)
    tok = QwenHeaderTok()
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: tok)
    messages = _oos_messages()
    monkeypatch.setattr(isv, "gsm8k_paired200_messages", lambda repo_root=None, system=None: [list(m) for m in messages])
    budget = _write_budget_off(world.tmp_path, QwenHeaderTok(), messages, cap=budget_cap, arm=budget_arm)
    ep_extra = {tc.ENDPOINT_KEY_LENS_SALT_POLICY: tc.LENS_SALT_POLICY_ATTESTED,
                tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_PSR_BARE_KEY: OBJECT_SHA},
                "geometry": standup_geometry(OOS_WINDOW, OOS_SEQS),
                "max_model_len": OOS_WINDOW, "max_num_seqs": OOS_SEQS,
                **(endpoint_extra or {})}
    write_standup_endpoint(world.table_path, **ep_extra)
    base_vmodel = next(r.virtual_model for r in px.load_routing_table(world.table_path)["routes"]
                       if r.cell_id == tc.ROUTE_NONE_CELL)
    world.chat.base_models = {base_vmodel}
    world.chat.base_probe = base_probe
    tokenized: list = []

    def recording_tokenize(body):
        tokenized.append(json.loads(json.dumps(body)))
        return live_tokenize(body)

    monkeypatch.setattr(srun, "_make_client",
                        lambda url, rt: SimpleNamespace(chat_completions=world.chat.chat_completions,
                                                        tokenize=recording_tokenize,
                                                        proxy_identity=dict(LIVE_IDENTITY)))
    argv = list(_percell_argv(world, tc.CONTROL_GSM8K))
    argv[argv.index("--cell") + 1] = cell
    root = Path(out_root) if out_root is not None else world.out_root
    argv[argv.index("--out-root") + 1] = str(root)
    argv += [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_ALLOW_OFF_CAP_OVERRIDE]
    if with_digest and tc.is_lens_cell(cell):
        argv += [tc.FLAG_EXPECT_OBJECT_SHA256, OBJECT_SHA]
    if with_budget:
        argv += [tc.FLAG_OOS_PROMPT_BUDGET, str(budget)]
    return SimpleNamespace(argv=argv, budget=budget, messages=messages, tok=tok, tokenized=tokenized,
                           chat=world.chat, base_vmodel=base_vmodel, cell=cell, out_root=root)


def _stub_run_inspect_tasks(monkeypatch):
    """`run_inspect_tasks` replaced by a recorder (nothing generated): the kwargs
    the driver hands it."""
    from sisteer.capeval.tiers import inspect_served

    seen: list = []

    async def fake_run(**kw):
        seen.append(dict(kw))
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    return seen


def test_the_off_cap_override_plans_8192_record_only_and_stamps_only_the_cap_key(world, monkeypatch):
    """offcap §4.2, §4.3, §4.6, §4.9: under `--allow-off-cap-override --thinking
    off` on the psr cell the driver hands `run_inspect_tasks` the RENDER flag
    False and the CAP flag True (`out_of_support=oos_run`, never the posture),
    no declared render fact, the strict `conditional` label; the hashed identity
    carries EXACTLY the in-support fields plus `max_tokens_by_task`
    ({gsm8k_paired200: 8192}) — no `out_of_support_thinking`, no
    `serve_out_of_support` — so the store's key differs from the in-support OFF
    store's by the cap key alone; the route block records the LANDED compare
    (the empty map, the digest compared, the salt policy, the conditional
    policy) and the REC/manifest carry the cap map."""
    oc = offcap_world(world, monkeypatch)
    seen = _stub_run_inspect_tasks(monkeypatch)
    srun.main(oc.argv)
    [kw] = seen
    assert kw["out_of_support"] is False and kw["off_cap_override"] is True
    assert kw["prompt_opened_think"] is False and kw["thinking"] == tc.THINKING_OFF
    assert kw["stamp_policy"] == "conditional" and kw["reasoning_parser"] is None
    manifest = json.loads((_cell_dir(oc.out_root) / "manifest.json").read_text())
    ident = manifest["resolved_config"]["tinycap"]
    assert set(ident) == set(tc.tinycap_identity_fields(off_cap_override=True))
    assert set(ident) - set(tc.TINYCAP_IDENTITY_FIELDS) == {tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK}
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {tc.GSM8K_STANDARD_TASK: OFFCAP_CAP}
    assert tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in ident and tc.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT not in ident
    assert ident["thinking"]["mode"] == tc.THINKING_OFF and ident["served"]["reasoning_parser"] is None
    assert ident[tc.IDENTITY_KEY_THINKING_BY_TASK] == {tc.GSM8K_STANDARD_TASK: tc.THINKING_OFF}
    route = manifest["route"]
    assert route[tc.ROUTE_KEY_SERVE_DECLARATIONS] == {} and route[tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] == OBJECT_SHA
    assert route[tc.ENDPOINT_KEY_LENS_SALT_POLICY] == tc.LENS_SALT_POLICY_ATTESTED
    assert route["resumed_prefill_policy"] == "conditional"
    assert set(route) == set(tc.ROUTE_BLOCK_FIELDS)
    ev = json.loads((_cell_dir(oc.out_root) / "evidence.json").read_text())
    assert ev[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK]["arm"] == tc.THINKING_OFF
    assert tc.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED not in ev          # the psr leg is not the base route
    # the in-support OFF run of the same cell on a sibling root: the identity WITHOUT the cap key,
    # a different store key — the two can never resume onto each other
    plain_root = world.tmp_path / "plain"
    argv = [str(plain_root) if a == str(oc.out_root) else a for a in oc.argv]
    argv = [a for a in argv if a != tc.FLAG_ALLOW_OFF_CAP_OVERRIDE]
    i = argv.index(tc.FLAG_OOS_PROMPT_BUDGET); del argv[i:i + 2]
    srun.main(argv)
    plain = json.loads((_cell_dir(plain_root) / "manifest.json").read_text())
    assert set(plain["resolved_config"]["tinycap"]) == set(tc.TINYCAP_IDENTITY_FIELDS)
    assert plain["harness_config_hash"] != manifest["harness_config_hash"]
    without_cap = {k: v for k, v in ident.items() if k != tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK}
    assert without_cap == plain["resolved_config"]["tinycap"]        # the cap key is the ONLY difference


def test_the_off_cap_probe_sends_explicit_off_kwargs_no_declaration_and_the_landed_stamps(world, monkeypatch):
    """offcap §3.1 / fact 3: the comparator's psr standup is the BARE cell; its
    bites probe sends `enable_thinking: false` explicitly on both legs and NO
    declaration (the OFF/512 leg's bodies, byte for byte), both legs' stamps
    are held to the §2.5.11 OFF rows (`request-unique / conditional /
    thinking_off`), and the record is the LANDED compare."""
    oc = offcap_world(world, monkeypatch)
    _stub_run_inspect_tasks(monkeypatch)
    srun.main(oc.argv)
    probe_calls = [c for c in oc.chat.calls if not c.get("continue_final_message")
                   and c["model"] != oc.base_vmodel]
    assert {c["model"] for c in probe_calls} == {world.route.virtual_model, world.twin.virtual_model}
    for c in probe_calls:
        assert c[tc.CHAT_TEMPLATE_KWARGS_KEY] == {tc.ENABLE_THINKING_KEY: False}
        assert tc.OOS_THINKING_DECLARATION_KEY not in c
    assert not [c for c in oc.chat.calls if c["model"] == oc.base_vmodel]         # no base-route probe on the psr leg
    bites = json.loads((_cell_dir(oc.out_root) / "evidence.json").read_text())["steering_bites"]
    assert bites[tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == tc.THINKING_OFF and bites["verified"] is True
    row = tc.expected_stamp_regimes(lens=True, arm=tc.THINKING_OFF)._asdict()
    assert row == {"salt_regime": "request-unique", "resumed_policy": "conditional", "render_regime": "thinking_off"}
    for leg in ("steered", "scale0"):
        assert set(map(json.dumps, bites["stamps_by_leg"][leg])) == {json.dumps(row)}
    # the same bodies the in-support OFF probe sends (test_the_in_support_off_probe_…)
    oc.chat.calls.clear()
    plain_root = world.tmp_path / "plain"
    argv = [str(plain_root) if a == str(oc.out_root) else a for a in oc.argv if a != tc.FLAG_ALLOW_OFF_CAP_OVERRIDE]
    i = argv.index(tc.FLAG_OOS_PROMPT_BUDGET); del argv[i:i + 2]
    srun.main(argv)
    plain_probe = [c for c in oc.chat.calls if not c.get("continue_final_message")]
    assert plain_probe == probe_calls


def test_the_off_cap_live_budget_pass_runs_under_the_off_render_and_the_on_artifact_refuses(world, monkeypatch):
    """offcap §4.4, §4.7: the live prompt-budget pass runs for the comparator
    under the OFF render — every /v1/tokenize body carries `chat_template_kwargs
    {"enable_thinking": false}` and NO declaration — and records `arm off, cap
    8192, window 16384, margin, n_items_differing 0` against the OFF artifact;
    the artifact is REQUIRED (absent -> refused before any store); an ON
    artifact, or an OFF one written for the table's 512, refuses by name
    (`prompt_budget_fit`: the artifact's arm and cap must be the run's)."""
    oc = offcap_world(world, monkeypatch)
    _stub_run_inspect_tasks(monkeypatch)
    srun.main(oc.argv)
    live = json.loads((_cell_dir(oc.out_root) / "evidence.json").read_text())[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK]
    art = json.loads(oc.budget.read_text())
    assert live["arm"] == tc.THINKING_OFF and live["cap"] == OFFCAP_CAP and live["window"] == OOS_WINDOW
    assert live["n"] == 200 and live["ok"] is True and live["n_items_differing"] == 0
    assert live[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX] == art["max"] and live["margin"] == OOS_WINDOW - art["max"] - OFFCAP_CAP
    assert live["artifact_chat_template_hash"] == TEMPLATE_HASH
    budget_bodies = [b for b in oc.tokenized if b.get("add_generation_prompt") and len(b["messages"]) == 2
                     and b["messages"][1]["content"].startswith("Question ")]
    assert len(budget_bodies) == 200
    for b in budget_bodies:
        assert b[tc.CHAT_TEMPLATE_KWARGS_KEY] == {tc.ENABLE_THINKING_KEY: False}
        assert tc.OOS_THINKING_DECLARATION_KEY not in b
    # the artifact is required
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(offcap_world(world, monkeypatch, with_budget=False, out_root=world.tmp_path / "r1").argv)
    assert tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH in str(e.value) and tc.FLAG_OOS_PROMPT_BUDGET in str(e.value)
    # the ON artifact refuses: the run serves the OFF arm
    with pytest.raises(SystemExit) as e:
        srun.main(offcap_world(world, monkeypatch, budget_arm=tc.THINKING_ON, out_root=world.tmp_path / "r2").argv)
    assert tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH in str(e.value) and "'on' render" in str(e.value)
    # a stale OFF artifact at the table's 512 refuses naming both caps
    with pytest.raises(SystemExit) as e:
        srun.main(offcap_world(world, monkeypatch, budget_cap=512, out_root=world.tmp_path / "r3").argv)
    msg = str(e.value)
    assert tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH in msg and "512" in msg and str(OFFCAP_CAP) in msg
    assert "AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off" in msg


def test_the_base_route_declaration_probe_records_thinking_on_not_lens_and_refuses_anything_else(world, monkeypatch):
    """offcap §4.8 [R2]: on the comparator's `none` configuration ONLY, one
    negative probe before generation — a one-token body carrying
    `render_fragment(lens=True, arm=on)` (the ON kwargs + `petri_thinking_on:
    true`) to the base vmodel — must be refused by the proxy with 400
    `thinking_on_not_lens` (contract §2.5.5) BEFORE any forward; recorded as
    `evidence.base_route_declaration_refused = {status, code, ok}`. A 2xx or any
    other code refuses the run naming OOS_DECLARATION_MISMATCH before
    generation. The psr leg sends no such probe; an in-support `none` run sends
    none either."""
    oc = offcap_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL)
    seen = _stub_run_inspect_tasks(monkeypatch)
    srun.main(oc.argv)
    [kw] = seen
    assert kw["off_cap_override"] is True and kw["out_of_support"] is False and kw["stamp_policy"] == tc.POLICY_NOT_CONSULTED
    probes = [c for c in oc.chat.calls if c["model"] == oc.base_vmodel]
    assert len(probes) == 1
    [probe] = probes
    assert probe[tc.OOS_THINKING_DECLARATION_KEY] is True
    assert probe[tc.CHAT_TEMPLATE_KWARGS_KEY] == {tc.ENABLE_THINKING_KEY: True}
    assert probe["max_tokens"] == 1 and probe["messages"][-1]["role"] == "user"
    assert tc.OOS_THINKING_DECLARATION_KEY in tc.render_fragment(lens=True, arm=tc.THINKING_ON)
    ev = json.loads((_cell_dir(oc.out_root) / "evidence.json").read_text())
    block = ev[tc.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED]
    assert block["status"] == 400 and block["code"] == px.THINKING_ON_NOT_LENS and block["ok"] is True
    assert block["vmodel"] == oc.base_vmodel
    ident = json.loads((_cell_dir(oc.out_root) / "manifest.json").read_text())["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {tc.GSM8K_STANDARD_TASK: OFFCAP_CAP}
    assert set(ident) == set(tc.tinycap_identity_fields(off_cap_override=True))
    # a proxy that FORWARDS the declared body (2xx): refused before generation, evidence written first
    _never_runs(monkeypatch)

    def forwards(payload):
        return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}

    oc2 = offcap_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL, base_probe=forwards,
                       out_root=world.tmp_path / "r1")
    with pytest.raises(SystemExit) as e:
        srun.main(oc2.argv)
    assert tc.REC_OOS_DECLARATION_MISMATCH in str(e.value) and px.THINKING_ON_NOT_LENS in str(e.value)
    block = json.loads((_cell_dir(oc2.out_root) / "evidence.json").read_text())[tc.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED]
    assert block["ok"] is False and block["status"] == 200 and block["code"] is None
    # a 400 under ANOTHER code is not the contract's refusal either
    def other_code(payload):
        raise px.ProxyError("proxy /v1/chat/completions returned 400: " + json.dumps(
            {"error": {"message": "m", "type": "invalid_request_error", "code": px.THINKING_ON_UNDECLARED}}))

    oc3 = offcap_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL, base_probe=other_code,
                       out_root=world.tmp_path / "r2")
    with pytest.raises(SystemExit) as e:
        srun.main(oc3.argv)
    assert tc.REC_OOS_DECLARATION_MISMATCH in str(e.value) and px.THINKING_ON_UNDECLARED in str(e.value)
    block = json.loads((_cell_dir(oc3.out_root) / "evidence.json").read_text())[tc.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED]
    assert block == {**block, "ok": False, "status": 400, "code": px.THINKING_ON_UNDECLARED}
    # an in-support `none` run (no flag) sends no probe and records no block
    oc4 = offcap_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL, out_root=world.tmp_path / "r3")
    oc4.chat.calls.clear()
    seen = _stub_run_inspect_tasks(monkeypatch)
    argv = [a for a in oc4.argv if a != tc.FLAG_ALLOW_OFF_CAP_OVERRIDE]
    i = argv.index(tc.FLAG_OOS_PROMPT_BUDGET); del argv[i:i + 2]
    srun.main(argv)
    assert not [c for c in oc4.chat.calls if c["model"] == oc4.base_vmodel]
    ev = json.loads((_cell_dir(oc4.out_root) / "evidence.json").read_text())
    assert tc.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED not in ev and tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK not in ev
    assert seen[0]["off_cap_override"] is False


def test_the_off_cap_flag_with_thinking_on_or_both_flags_refuses_before_any_side_effect(world, monkeypatch):
    """offcap §4.2: `--allow-off-cap-override --thinking on` refuses at startup
    as OFFCAP_UNAUTHORISED naming `mode differs` (the comparator has no ON arm);
    both override flags together refuse naming BOTH flags, BEFORE either
    authorisation; another cell / control / samples refuse by field. Nothing is
    generated and no store directory is created."""
    _never_runs(monkeypatch)
    oc = offcap_world(world, monkeypatch, out_root=world.tmp_path / "r0")
    on_argv = list(oc.argv)
    on_argv[on_argv.index(tc.FLAG_THINKING) + 1] = tc.THINKING_ON
    with pytest.raises(SystemExit) as e:
        srun.main(on_argv)
    msg = str(e.value)
    assert msg.startswith(f"[served-cap] {tc.REC_OFFCAP_UNAUTHORISED}:") and "mode differs" in msg
    assert tc.FLAG_ALLOW_OFF_CAP_OVERRIDE in msg and "AUTHORISED_OFF_CAP_COMPARATOR" in msg
    assert not (oc.out_root / "cells").exists()
    with pytest.raises(SystemExit) as e:
        srun.main(oc.argv + [tc.FLAG_ALLOW_OOS_THINKING])
    msg = str(e.value)
    assert tc.FLAG_ALLOW_OFF_CAP_OVERRIDE in msg and tc.FLAG_ALLOW_OOS_THINKING in msg and "mutually exclusive" in msg
    assert not (oc.out_root / "cells").exists()
    # (`inspect_cmp:or_bench_hard_1k` is a tinycap control the comparator does not name; a LEGACY control
    # beside gsm8k refuses the same way — the COMPLETE parsed list is checked, see the test below)
    for swap, field in ((("--cell", "si_constitution__psr_kl_all"), "cell"),
                        (("--controls", "gsm8k,inspect_cmp:or_bench_hard_1k"), "controls"),
                        ((tc.FLAG_SAMPLES, "4"), "samples")):
        argv = list(oc.argv)
        if swap[0] in argv:
            argv[argv.index(swap[0]) + 1] = swap[1]
        else:
            argv += list(swap)                                    # the world's argv defaults --samples
        with pytest.raises(SystemExit) as e:
            srun.main(argv)
        assert tc.REC_OFFCAP_UNAUTHORISED in str(e.value) and f"{field} differs" in str(e.value), field
    assert not (oc.out_root / "cells").exists()


def _spy_on_the_endpoint_and_the_store(monkeypatch) -> list:
    """Every seam through which a driver run first reaches the served endpoint or
    creates a store, replaced by a recorder that refuses: the proxy client, the
    poster, the standup record, the legacy judged specs and the store opener."""
    touched: list = []

    def refusing(name):
        def _f(*a, **k):
            touched.append(name)
            raise AssertionError(f"a refused run must never reach {name}")
        return _f

    for name in ("_make_client", "_proxy_poster", "_standup_endpoint", "_judged_specs", "_store_for"):
        monkeypatch.setattr(srun, name, refusing(name))
    return touched


@pytest.mark.parametrize("legacy", ["xstest", "mmlu", "or_bench_hard_1k", LEGACY_GEN_CONTROLS])
def test_the_off_cap_flag_with_a_legacy_control_beside_gsm8k_refuses_before_the_endpoint_or_a_store(
        world, monkeypatch, legacy):
    """The code review's H1: both comparator authorisation calls handed the
    contracts `controls=atoms` — the list AFTER `tinycap_atoms` dropped the
    legacy served controls (xstest, mmlu, or_bench_hard_1k) — while
    `_judged_specs` / the mmlu leg still execute those from the ORIGINAL list, so
    `--allow-off-cap-override --thinking off --controls gsm8k,xstest` passed as
    the GSM8K-only comparator and ran extra workloads outside the tuple on paths
    that bypass the render / stamp enforcement. The COMPLETE parsed list is
    checked now: the run refuses OFFCAP_UNAUTHORISED naming `controls differs`
    and the legacy token, before the endpoint is read, before any request, before
    a store exists."""
    _never_runs(monkeypatch)
    oc = offcap_world(world, monkeypatch, out_root=world.tmp_path / "r0")
    touched = _spy_on_the_endpoint_and_the_store(monkeypatch)
    argv = list(oc.argv)
    argv[argv.index("--controls") + 1] = f"{tc.CONTROL_GSM8K},{legacy}"
    with pytest.raises(SystemExit) as e:
        srun.main(argv)
    msg = str(e.value)
    assert msg.startswith(f"[served-cap] {tc.REC_OFFCAP_UNAUTHORISED}:") and "controls differs" in msg
    for token in legacy.split(","):
        assert token in msg.split("Not authorised:", 1)[1], token
    assert tc.FLAG_ALLOW_OFF_CAP_OVERRIDE in msg and "AUTHORISED_OFF_CAP_COMPARATOR" in msg
    assert touched == [] and not oc.out_root.exists()
    # the parent's OOS door had the same hole and is closed the same way: the ON run
    # with a legacy control beside gsm8k refuses OOS_UNAUTHORISED by field, nothing reached
    fake_frameworks(monkeypatch)
    write_standup_endpoint(world.table_path)
    _pin_the_authorised_oos_model(monkeypatch)
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: HeaderTok(moves=True))
    touched = _spy_on_the_endpoint_and_the_store(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(_oos_argv(world, controls=f"{tc.CONTROL_GSM8K},{legacy}"))
    msg = str(e.value)
    assert msg.startswith(f"[served-cap] {tc.REC_OOS_UNAUTHORISED}:") and "controls differs" in msg
    assert legacy.split(",")[0] in msg.split("Not authorised:", 1)[1] and "AUTHORISED_OOS_THINKING" in msg
    assert touched == [] and not (world.out_root / "cells").exists()
    # the source: neither door's call hands the contracts the filtered `atoms`
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "controls=atoms" not in src
    # two overrides x (startup, post-attestation) + the OOS authorisation re-bound to the
    # SERVED model once the routing table's base_model is resolved
    assert src.count("cell=args.cell, controls=controls,") == 5
    assert src.count('where="driver run (served model)"') == 1


def test_an_in_support_off_run_is_byte_identical_with_the_override_code_present(world, monkeypatch):
    """offcap R4 (the `:2755` pattern): an ordinary thinking-OFF run, no flag —
    the planner receives `out_of_support=False, prompt_opened_think=False,
    off_cap_override=False`; the dispatched plan equals, field for field, the
    plan built WITHOUT the three keys (the table cap 512 / `off_table`,
    `record_only`, the same ids); the identity is EXACTLY the unconditional
    field set; the probe bodies are the fixture's (explicit OFF kwargs, no
    declaration); no live budget pass, no base-route probe, no evidence block."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)     # `petri` group only
    import dataclasses

    from sisteer.capeval.tiers import inspect_served as isv

    stub_framework_versions(monkeypatch)
    write_standup_endpoint(world.table_path)
    ids = _stub_gsm8k_paired200(monkeypatch)
    seen = _real_run_inspect_tasks_capturing_plans(monkeypatch)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, tc.THINKING_OFF])
    [kw] = seen["plan_kwargs"]
    assert kw["out_of_support"] is False and kw["prompt_opened_think"] is False and kw["off_cap_override"] is False
    [plan] = seen["plans"]
    assert (plan.max_tokens, plan.cap_source) == (512, "off_table")
    assert plan.think_split_mode == tc.THINK_SPLIT_RECORD_ONLY and plan.sample_ids == ids
    before = {k: v for k, v in kw.items() if k not in ("out_of_support", "prompt_opened_think", "off_cap_override")}
    [rebuilt] = isv.plans_for_atoms(["gsm8k"], **before)

    def strip(p):
        return dataclasses.replace(p, task=None)          # the Task object is fresh per build

    assert strip(plan) == strip(rebuilt)
    ident = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())["resolved_config"]["tinycap"]
    assert set(ident) == set(tc.TINYCAP_IDENTITY_FIELDS) and tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK not in ident
    ev = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())
    assert tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK not in ev and tc.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED not in ev
    probe = [c for c in world.chat.calls if not c.get("continue_final_message")]
    assert probe and all(c[tc.CHAT_TEMPLATE_KWARGS_KEY] == {tc.ENABLE_THINKING_KEY: False}
                         and tc.OOS_THINKING_DECLARATION_KEY not in c for c in probe)
    # the driver's own source: the render flag alone reaches the planner; the cap flag beside it;
    # the ledger path stays the LAST kwarg (the existing pin)
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "out_of_support=oos_run," in src and "off_cap_override=offcap_run," in src
    assert "held_to_record = oos_run or offcap_run" in src
    assert "ledger_path=(Path(out_root) / tc.LEDGER_FILENAME\n" in src


def test_the_real_comparator_driver_plans_both_legs_through_the_real_task_planner(world, monkeypatch):
    """offcap §8.2 [R2] — the CPU integration test: the REAL driver into
    the REAL task planner (run_inspect_tasks -> plans_for_atoms -> plan_task ->
    _plan_task, no stub between them; only the proxy/tokenize transport, the
    model build, the per-task runner and the frameworks are faked) for BOTH
    `none` and `safety_refusal__psr_kl_all`: a TaskPlan for gsm8k_paired200 with
    `max_tokens 8192, cap_source offcap_override, think_split_mode record_only,
    thinking off` reaches run_one_task for each cell; plans_for_atoms was
    called with `out_of_support=False, off_cap_override=True`; no SystemExit
    fired after the standup record was read; the hashed cap map IS the
    dispatched plan's cap; and the comparator's plan differs from the
    in-support OFF plan in `max_tokens` and `cap_source` ONLY (offcap §4.10:
    the request bodies differ in `max_tokens` alone)."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)     # `petri` group only
    import dataclasses

    from sisteer.capeval.tiers import inspect_served as isv

    ids = _stub_gsm8k_paired200(monkeypatch)
    plans = {}
    for cell in tc.AUTHORISED_OFF_CAP_COMPARATOR.cells:
        oc = offcap_world(world, monkeypatch, cell=cell, frameworks="stub", out_root=world.tmp_path / f"root_{cell}")
        seen = _real_run_inspect_tasks_capturing_plans(monkeypatch)
        srun.main(oc.argv)                                              # no refusal after the record was read
        [kw] = seen["plan_kwargs"]
        assert kw["out_of_support"] is False and kw["off_cap_override"] is True, cell
        assert kw["prompt_opened_think"] is False and kw["thinking"] == tc.THINKING_OFF and kw["reasoning_parser"] is None
        [plan] = seen["plans"]
        assert isinstance(plan, isv.TaskPlan) and plan.row_task == tc.GSM8K_STANDARD_TASK
        assert plan.max_tokens == OFFCAP_CAP == tc.off_cap_override_max_tokens("gsm8k_paired200")
        assert plan.cap_source == "offcap_override" and plan.think_split_mode == tc.THINK_SPLIT_RECORD_ONLY
        assert plan.thinking == tc.THINKING_OFF and plan.sample_ids == ids and plan.expected_n == 200
        assert (plan.temperature, plan.top_p, plan.top_k) == (0.7, 0.95, 20)    # the card row rode along
        ident = json.loads((_cell_dir(oc.out_root) / "manifest.json").read_text())["resolved_config"]["tinycap"]
        assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {p.row_task: p.max_tokens for p in seen["plans"]}
        assert set(ident) == set(tc.tinycap_identity_fields(off_cap_override=True))
        plans[cell] = plan
    # the in-support OFF plan from the same builder: the comparator's differs in the cap alone
    before = {k: v for k, v in kw.items() if k not in ("out_of_support", "prompt_opened_think", "off_cap_override")}
    [plain] = isv.plans_for_atoms(["gsm8k"], **before)

    def strip(p):
        return dataclasses.replace(p, task=None)

    assert (plain.max_tokens, plain.cap_source) == (512, "off_table")
    for cell, plan in plans.items():
        assert strip(plan) == dataclasses.replace(strip(plain), max_tokens=OFFCAP_CAP, cap_source="offcap_override"), cell


# --------------------------------------------------------------------------- #
# the MATH-500 default change (one pass at 16,384): a STANDARD run's              #
# identity and manifest are pinned to golden fixtures captured on this          #
# synthetic world — first from the pre-change driver (main f90557fd), then      #
# re-captured at main 7f90bcb5 after commit 83345d3a changed the client lock    #
# --------------------------------------------------------------------------- #

#: The UNHASHED manifest leaves the comparator normalises, each replaced by a
#: placeholder on both sides. Two vary with the MACHINE, not the instrument:
#: ``producer.tree_sha`` (the checkout's HEAD) and ``row_constants.env_digest``
#: (``store.env_digest``: a provenance-only digest of python / torch /
#: transformers / cuda driver / GPU name — never key-bearing, and different in
#: every venv by design; the Modal CPU suite is where it first differed from
#: the capturing venv). Top-level ``proxy_upstream_timeout_s`` — the proxy's
#: upstream timeout the driver mirrors from the standup record — is
#: NOT normalised: since the bcb-bfcl-200 merge the driver stamps it only when
#: the record carries one (a pre-knob record stamps NO leaf, so every manifest
#: from such a standup keeps its shape), and this world's synthetic standup
#: record carries none — the leaf is absent exactly as the capturing tree left
#: it. (The integration branch briefly carried a one-leaf placeholder for the
#: null leaf 7283933f wrote; the merge retired both.) Every OTHER leaf is
#: compared byte for byte — the negative test below holds the comparator to
#: that, the timeout leaf included.
_GOLDEN_VOLATILE_LEAVES = {"tree_sha": "<TREE_SHA>", "env_digest": "<ENV_DIGEST>"}


def _golden_normalise(obj, *, tmp_path: Path):
    """The manifest with its VOLATILE facts replaced: paths under the test's
    tmp dir (the routing table the standup record names) and the machine-bound
    leaves of :data:`_GOLDEN_VOLATILE_LEAVES` (``tinycap_golden.normalise``).
    Everything else — the hashed identity, the hash itself, every block — is
    compared byte for byte."""
    return golden.normalise(obj, tmp_path=tmp_path, volatile=_GOLDEN_VOLATILE_LEAVES)


def _fake_inspect_leg(monkeypatch, calls, *, task: str, n: int, **summary):
    """A faked Inspect leg for ONE row task — ``n`` items, one draw each — that
    records the kwargs the driver hands ``run_inspect_tasks`` and files the
    task's done marker, so the driver writes its manifest as it would after a
    real leg. ``summary`` adds the task's own payload keys (accuracy, item set,
    cap). No dataset is touched."""
    from sisteer.capeval.tiers import inspect_served

    async def fake_run_inspect_tasks(*, atoms, route, routes, system, store, template,
                                     thinking, base_url, limits=None, **kw):
        calls.append({"atoms": list(atoms), "thinking": thinking,
                      "retry": kw.get("math500_retry_max_tokens"),
                      "window": kw.get("max_model_len")})
        payload = {"n": n, "n_requests": n, "n_rejected": 0, "samples_per_item": 1,
                   "task": task, "framework": tc.INSPECT_TASK_VERSION, "engine": "vllm",
                   "route": route.virtual_model, "thinking": thinking,
                   "render": tc.THINKING_RENDER_NATIVE, "wall_clock_s": 1.0,
                   "temperature": 0.0, **summary}
        completion.finish_task(store.cell_dir, task=task, n_items=n, n_expected_full=n,
                               limits=limits, n_rows=n, n_requests=n, n_rejected=0,
                               wall_clock_s=1.0, summary=payload)
        return {task: payload}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run_inspect_tasks)


#: The faked leg per control the identity tests run: the STANDARD paired GSM8K
#: slot, and the in-depth MATH-500 control (500 items, nothing at the cap).
_FAKE_LEG_OF_CONTROL = {
    tc.CONTROL_GSM8K: {"task": tc.GSM8K_STANDARD_TASK, "n": tc.GSM8K_PAIRED_N, "acc": 0.8,
                       "item_set": tc.GSM8K_ITEM_SET_PAIRED, "max_tokens": 512},
    tc.CONTROL_MATH500: {"task": tc.inspect_task("math"), "n": 500, "acc": 0.5,
                         "item_set": "math500", "max_tokens": tc.MATH500_MAX_TOKENS_DEFAULT,
                         "n_at_cap": 0, "n_stop_reason": 500, "at_cap_rate": 0.0},
}


def _run_faked_control(world, monkeypatch, *, control: str, arm: str, reasoning_parser=None):
    """One driver run on the world — ``control`` through its faked Inspect leg
    under ``arm`` — with the framework versions stubbed (the manifest's
    frameworks block is hashed identity and must not depend on the venv).
    Returns the cell dir, its manifest and the kwargs the leg received (the
    list keeps recording if the caller runs the driver again)."""
    stub_framework_versions(monkeypatch)
    calls = []
    _fake_inspect_leg(monkeypatch, calls, **_FAKE_LEG_OF_CONTROL[control])
    if reasoning_parser is not None:
        # a thinking-ON generation task needs the parser its standup composed,
        # and the served header must be measured (gemma-4: invariant, H=3)
        write_standup_endpoint(world.table_path, reasoning_parser=reasoning_parser)
        monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: HeaderTok(moves=False))
    srun.main(_percell_argv(world, control) + [tc.FLAG_THINKING, arm])
    cell_dir = _cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    return cell_dir, manifest, calls


@pytest.mark.parametrize("name,arm,reasoning_parser", [
    ("standard_off_gsm8k", tc.THINKING_OFF, None),
    ("standard_on_gemma_gsm8k", tc.THINKING_ON, "gemma4"),
])
def test_a_standard_run_matches_the_pre_change_golden_identity_and_manifest(
        world, monkeypatch, name, arm, reasoning_parser):
    """The MATH-500 default change may re-key MATH stores
    and nothing else. A standard thinking-OFF run and a standard gemma
    thinking-ON run (the paired GSM8K control on the fitted cell) produce the
    SAME hashed identity, the same ``harness_config_hash``, the same cell-key
    directory and the same manifest as the golden driver did on this world —
    compared byte for byte, apart from the normalised leaves of
    :data:`_GOLDEN_VOLATILE_LEAVES`: volatile paths and the two machine-bound
    digests. ``manifest.proxy_upstream_timeout_s`` is stamped only from a
    standup record that carries one (main's rule since the bcb-bfcl-200 merge),
    so on this knob-less world it is absent, as the capturing tree left it.

    Provenance of the goldens: first captured from the pre-change driver (main
    f90557fd). Re-captured ONCE at main 7f90bcb5 because commit 83345d3a
    ("inspect-sandboxes + swebench enter the client lock") changed
    ``configs/capeval/tinycap-client-requirements.txt``, whose sha256 is the
    hashed identity leaf ``tinycap.frameworks.client_lock_sha256`` — a
    dependency change moves every standard identity BY DESIGN:

    * lock sha256 ``65c0d3db0558deccfa1faebefa1cced1bd50b0984d029fe5b8d1378254e1b29e``
      → ``0f21fd4d92dc56b45e966ddf666463d268e547c82ba6bf54b2e9b1fffd47cc0f``
    * ``harness_config_hash`` OFF ``b93b6a442cbb15ef`` → ``b7383e446415e9ae``,
      ON/gemma ``e4cfe879c565289c`` → ``4325581deb6ce0f7`` (and the cell dirs)

    The leaf-by-leaf diff old → new fixture was exactly those leaves and their
    copies (``identity`` / ``manifest`` / ``manifest.resolved_config`` /
    ``manifest.tinycap`` lock sha; ``manifest.harness_config_hash`` and
    ``manifest.row_constants.harness_config_hash``; ``cell_dir``) — 8 of 289
    leaves, plus this ``captured_from`` string; no other leaf moved. Any
    further departure is a real identity change and must be judged, never
    re-captured on sight."""
    cell_dir, manifest, _ = _run_faked_control(world, monkeypatch, control=tc.CONTROL_GSM8K,
                                               arm=arm, reasoning_parser=reasoning_parser)
    got = {
        "captured_from": ("the driver at main 7f90bcb5 on the synthetic test world — re-captured "
                          "after PR #20 (83345d3a) moved the client lock, sha256 65c0d3db… → "
                          "0f21fd4d… (a hashed identity leaf); first captured from the pre-change "
                          "driver at main f90557fd"),
        "run": {"controls": [tc.CONTROL_GSM8K], "thinking": arm,
                "reasoning_parser": reasoning_parser, "cell": TT_CELL},
        "cell_dir": cell_dir.name,
        "harness_config_hash": manifest["harness_config_hash"],
        "identity": manifest["resolved_config"],
        "manifest": _golden_normalise(manifest, tmp_path=world.tmp_path),
    }
    assert set(got["identity"]["tinycap"]) == set(tc.TINYCAP_IDENTITY_FIELDS)
    golden.compare_or_write(golden.GOLDEN_DIR / f"{name}.json", got)


def test_the_golden_comparator_still_names_any_other_differing_manifest_leaf(tmp_path):
    """The normalisation is EXACTLY the leaves of :data:`_GOLDEN_VOLATILE_LEAVES`.
    Against a copy of the thinking-OFF fixture: a manifest whose only departure
    is one of those leaves (any ``tree_sha`` / ``env_digest``) compares equal,
    while a manifest differing at ANY other leaf — an unhashed top-level scalar,
    the hash itself, a hashed identity field, a whole block, a sibling key, a key
    the fixture never had, the proxy-timeout leaf a knob-less world never stamps
    — fails naming exactly that leaf.
    No driver run: the comparator and the normaliser are exercised on the
    fixture's own bytes."""
    want = json.loads((golden.GOLDEN_DIR / "standard_off_gsm8k.json").read_text())
    assert tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S not in want["manifest"]     # a knob-less world
    fixture = tmp_path / "standard_off_gsm8k.json"
    fixture.write_text(json.dumps(want, indent=1, sort_keys=True) + "\n")

    def fresh():
        return json.loads(json.dumps(want["manifest"]))

    def got(manifest):
        return {**want, "manifest": _golden_normalise(manifest, tmp_path=tmp_path / "nowhere")}

    m = fresh()
    m["producer"]["tree_sha"], m["row_constants"]["env_digest"] = "f" * 40, "e" * 64
    golden.compare_or_write(fixture, got(m))
    for path, value in [
        (("concurrency",), 999),                                    # an unhashed top-level scalar
        (("harness_config_hash",), "0" * 64),                       # the hash itself
        (("resolved_config", "tinycap", "samples"), 7),             # a hashed identity field
        (("quality",), "gone"),                                     # a whole block
        (("proxy_url",), "http://127.0.0.1:2/v1"),                  # the placeholder's sibling
        (("never_written_key",), None),                             # an additive key the fixture lacks
        ((tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S,), 7200.0),      # the knob's leaf on a knob-less world
    ]:
        m = node = fresh()
        for k in path[:-1]:
            node = node[k]
        assert node.get(path[-1], object()) != value
        node[path[-1]] = value
        with pytest.raises(pytest.fail.Exception) as e:
            golden.compare_or_write(fixture, got(m))
        assert str(e.value).endswith(f"differs from the golden fixture at ['manifest.{'.'.join(path)}']"), str(e.value)


@pytest.mark.parametrize("arm,reasoning_parser", [(tc.THINKING_OFF, None), (tc.THINKING_ON, "gemma4")])
def test_a_math500_run_stamps_the_one_pass_retry_value_and_a_rearmed_one_when_asked(
        world, monkeypatch, arm, reasoning_parser):
    """The ONE unavoidable identity change: a run that
    files MATH-500 records the retry the driver actually arms — 0 by default
    (one pass at 16,384, no re-request), the re-armed value when the operator
    passes one above the cap — so the one-pass instrument never keys with, or
    resumes onto, a pre-change two-pass store (recorded 8192). Both arms; the
    leg is handed the same number the identity hashes."""
    cell_dir, manifest, calls = _run_faked_control(world, monkeypatch, control=tc.CONTROL_MATH500,
                                                   arm=arm, reasoning_parser=reasoning_parser)
    [kw] = calls
    assert kw["atoms"] == [tc.CONTROL_MATH500] and kw["retry"] == 0 == tc.MATH500_RETRY_MAX_TOKENS
    ident = manifest["resolved_config"]["tinycap"]
    assert ident["tasks"] == [tc.CONTROL_MATH500]
    assert ident[tc.IDENTITY_KEY_THINKING_BY_TASK] == {tc.inspect_task("math"): arm}
    assert ident[tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] == 0
    # a MATH run's identity carries EXACTLY the base fields plus the
    # first-pass cap it generated at — the plan builder's own answer for MATH
    # under this arm (16,384 in both arms), hashed so no store at another cap
    # can share its key
    assert set(ident) == set(tc.tinycap_identity_fields(files_math500=True))
    assert set(ident) - set(tc.TINYCAP_IDENTITY_FIELDS) == {tc.IDENTITY_KEY_MATH500_MAX_TOKENS}
    assert ident[tc.IDENTITY_KEY_MATH500_MAX_TOKENS] == 16384 == tc.MATH500_MAX_TOKENS_DEFAULT
    assert ident[tc.IDENTITY_KEY_MATH500_MAX_TOKENS] == tc.math500_max_tokens_for(arm)
    # the same run with the retry RE-ARMED above the cap is another store: the
    # value is the operator's, and the leg receives exactly it
    rearmed_root = world.out_root.parent / "out_rearmed"
    srun.main(_percell_argv(world, tc.CONTROL_MATH500, out_root=rearmed_root)
              + [tc.FLAG_THINKING, arm, tc.FLAG_MATH500_RETRY_MAX_TOKENS, "32768"])
    assert calls[-1]["retry"] == 32768
    other = _cell_dir(rearmed_root)
    assert other.name != cell_dir.name
    ident2 = json.loads((other / "manifest.json").read_text())["resolved_config"]["tinycap"]
    assert ident2[tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] == 32768
    # and the driver derives the stamp through the ONE contracts rule
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert re.search(r"tc\.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS:\s*"
                     r"tc\.math500_retry_identity_value\(\s*thinking_by_task,", src)
    # ... and the cap off the ONE executed-cap map (the plan builder's rule, the
    # same map version 2 hashes as max_tokens_by_task), gated on the ONE predicate
    assert re.search(r"tc\.IDENTITY_KEY_MATH500_MAX_TOKENS:\s*_math500_max_tokens\(executed_caps\)\}\s*"
                     r"if tc\.files_math500\(thinking_by_task\) else \{\}", src)
    assert "files_math500=tc.files_math500(thinking_by_task)" in src



def test_the_math500_scalar_and_the_per_task_cap_map_are_one_resolution(monkeypatch):
    """The hashed scalar ``math500_max_tokens`` and the version-2
    ``max_tokens_by_task["inspect/math"]`` must derive from ONE cap resolution —
    the driver computes the executed-cap map once (`_override_max_tokens_by_task`,
    the plan builder's rule under this run's support state, override and matched
    tuple) and reads the scalar off it (`_math500_max_tokens`). Held here by
    moving the cap: a registry tuple that raises MATH's ON cap to 12,288 moves the
    scalar, the map's entry and the launcher's pure twin together; the parent's
    tuple and in-support runs read the table's 8,192 in all three; a map without
    the MATH row refuses by name."""
    math = tc.inspect_task("math")
    on = {math: tc.THINKING_ON}
    raised = tc.AUTHORISED_OOS_THINKING._replace(max_tokens_on={**tc.AUTHORISED_OOS_THINKING.max_tokens_on,
                                                                "math": 12288})
    monkeypatch.setattr(tc, "AUTHORISED_OOS_EXPERIMENTS",
                        {**tc.AUTHORISED_OOS_EXPERIMENTS, "synthetic_raised_math": raised})
    caps = srun._override_max_tokens_by_task([tc.CONTROL_MATH500], on, out_of_support=True,
                                             off_cap_override=False, oos_experiment="synthetic_raised_math")
    assert caps == {math: 12288}
    assert srun._math500_max_tokens(caps) == caps[math] == 12288
    assert tc.math500_max_tokens_for(tc.THINKING_ON, out_of_support=True, experiment="synthetic_raised_math") == 12288
    # the parent's tuple names no MATH cap; in support the table's cap — 16,384 in all three
    for kw in (dict(out_of_support=True, oos_experiment=None), dict(out_of_support=False, oos_experiment=None)):
        caps = srun._override_max_tokens_by_task([tc.CONTROL_MATH500], on, off_cap_override=False, **kw)
        assert srun._math500_max_tokens(caps) == caps[math] == 16384 == tc.MATH500_MAX_TOKENS_DEFAULT
        assert tc.math500_max_tokens_for(tc.THINKING_ON, out_of_support=kw["out_of_support"]) == 16384
    # both arms of a mixed run share the map; the OFF arm reads the OFF table
    both = srun._override_max_tokens_by_task([tc.CONTROL_MATH500, tc.CONTROL_GSM8K],
                                             {math: tc.THINKING_OFF, tc.GSM8K_STANDARD_TASK: tc.THINKING_OFF},
                                             out_of_support=False, off_cap_override=False)
    assert srun._math500_max_tokens(both) == both[math] == 16384 and tc.GSM8K_STANDARD_TASK in both
    # a map that resolved no MATH cap is a caller bug, refused by name
    with pytest.raises(SystemExit, match="carries no cap for inspect/math"):
        srun._math500_max_tokens({tc.GSM8K_STANDARD_TASK: 512})
    with pytest.raises(SystemExit, match="carries no cap for inspect/math"):
        srun._math500_max_tokens({math: None})
    # and the driver's source reads BOTH keys off the one map (the pins the
    # contracts test also holds)
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert src.count("executed_caps = _override_max_tokens_by_task(") == 1
    assert "_math500_max_tokens(executed_caps)" in src
    assert "tinycap_identity.setdefault(tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK, executed_caps)" in src

def test_a_version_1_math500_run_needs_no_ledger_and_a_version_2_run_refuses_without_one(world, monkeypatch):
    """The ledger requirement is the EXPERIMENT's (identity version 2), like every
    other gate the experiment armed (LA15/LA16): a standard-battery MATH-500 run
    (version 1, the default) needs no `--ledger-url` and runs exactly as before;
    the same run under `--tinycap-identity-version 2` refuses before generation
    naming the flag and the version. (The full Modal suite caught the v1 battery
    tests dying on the unscoped requirement.)"""
    _, manifest, calls = _run_faked_control(world, monkeypatch, control=tc.CONTROL_MATH500, arm=tc.THINKING_OFF)
    assert len(calls) == 1 and manifest["ledger_url"] is None
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(_percell_argv(world, tc.CONTROL_MATH500, out_root=world.out_root.parent / "v2")
                  + [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_TINYCAP_IDENTITY_VERSION, "2"])
    msg = str(e.value)
    assert f"{tc.FLAG_LEDGER_URL} is REQUIRED for ['math500']" in msg and f"{tc.FLAG_TINYCAP_IDENTITY_VERSION} 2" in msg
    assert not (world.out_root.parent / "v2" / "cells").exists()


def test_the_sync_proxy_client_reports_the_exact_bytes_and_headers_it_puts_on_the_wire(monkeypatch):
    """The probe side of the HTTP boundary: `ProxyClientEngine`
    calls `on_request(url, body_bytes, headers)` with the SAME bytes and headers
    it hands urllib — the labels (`X-Tinycap-*`) a caller passes to
    `chat_completions(payload, headers=)` included — so the driver's record of
    a probe is the wire body, not a re-serialisation beside it. Without a hook
    the client is byte-identical."""
    import io
    import urllib.request

    sent = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, req.data, {k.lower(): v for k, v in req.header_items()}))
        return _Resp(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    seen = []
    client = px.ProxyClientEngine("http://127.0.0.1:1/v1", identity="proxy_vllm+lens+exact_replay",
                                  on_request=lambda url, data, headers: seen.append((url, data, dict(headers))))
    labels = {tc.HEADER_TASK: "probe/render", tc.HEADER_SAMPLE: "petri__base/0", tc.HEADER_REQUEST_ID: "ab" * 16}
    client.chat_completions({"model": "petri__base", "messages": [{"role": "user", "content": "hi"}]}, headers=labels)
    [(url, data, headers)] = seen
    [(wire_url, wire_data, wire_headers)] = sent
    assert url == wire_url and data == wire_data and json.loads(data)["model"] == "petri__base"
    assert {k.lower(): v for k, v in headers.items()} == wire_headers
    assert wire_headers[tc.HEADER_REQUEST_ID.lower()] == "ab" * 16 and wire_headers[tc.HEADER_TASK.lower()] == "probe/render"
    # tokenize is not a chat attempt: the hook sees it too (a POST) but the driver's recorder filters on the path
    client.tokenize({"model": "petri__base", "messages": []})
    assert len(seen) == 2 and seen[1][0].endswith("/tokenize")
    # no hook: byte-identical
    plain = px.ProxyClientEngine("http://127.0.0.1:1/v1", identity="proxy_vllm+lens+exact_replay")
    assert plain.on_request is None
    plain.chat_completions({"model": "petri__base", "messages": []})
    assert sent[-1][1] == json.dumps({"model": "petri__base", "messages": []}).encode("utf-8")


def test_a_non_math_run_hands_the_leg_the_default_retry_but_stamps_the_legacy_literal(
        world, monkeypatch):
    """The two halves of the compatibility rule side by side: the Inspect leg
    receives the driver's default (0 — irrelevant to a task that never
    retries, by ``_plan_task``'s construction), while the identity carries the
    literal every pre-change store of this shape recorded (8192). Whatever the
    flag says on a non-MATH run, the stamp is the legacy literal."""
    _, manifest, calls = _run_faked_control(world, monkeypatch, control=tc.CONTROL_GSM8K,
                                            arm=tc.THINKING_OFF)
    [kw] = calls
    assert kw["retry"] == 0 == tc.MATH500_RETRY_MAX_TOKENS
    ident = manifest["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] == tc.MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY == 8192
    # the first-pass cap key is MATH's alone — absent here, so the field
    # set is exactly the unconditional one (the goldens pin the bytes)
    assert tc.IDENTITY_KEY_MATH500_MAX_TOKENS not in ident
    assert set(ident) == set(tc.TINYCAP_IDENTITY_FIELDS) == set(tc.tinycap_identity_fields())
    # an explicit (re-armed) value on a NON-MATH run changes neither the stamp
    # nor the store: the retry is not part of this instrument
    cell_dir = _cell_dir(world.out_root)
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K)
              + [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_MATH500_RETRY_MAX_TOKENS, "32768"])
    assert calls[-1]["retry"] == 32768 and _cell_dir(world.out_root) == cell_dir
    again = json.loads((cell_dir / "manifest.json").read_text())["resolved_config"]["tinycap"]
    assert again[tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] == 8192


def test_a_pre_change_explicit_zero_retry_math500_run_never_keys_with_the_one_pass_default(
        world, monkeypatch):
    """Review round, reproduced: a previously supported MATH-500 run
    ``--thinking on --math500-retry-max-tokens 0`` generated at the pre-change
    cap (4,096) but recorded the SAME identity as the new default generating at
    8,192 — the first-pass cap was not hashed, the retry field is 0 in both, and
    the thinking-ON geometry is unchanged — so resuming such a store under the
    new default returned 4,096-cap results through the done-marker shortcut.
    Here the pre-change instrument is simulated (4,096 in both cap tables, the
    retry explicitly off) and run FIRST into the store root; the new default
    then runs into the SAME root and must land in ANOTHER cell dir, with the
    leg generating again (the old done marker never consulted), the two
    ``harness_config_hash`` values distinct, and the hashed identity differing
    at exactly one leaf: ``tinycap.math500_max_tokens`` (4,096 vs 8,192). With
    two stores for one task in one root the summary merge then REFUSES by name
    (PW8's two-sources rule) rather than pick one — before the fix the second
    run took the shortcut silently: one cell dir, no second generation, no
    error."""
    from sisteer.capeval import store as cs
    from sisteer.capeval.tiers import inspect_served

    stub_framework_versions(monkeypatch)
    calls = []
    _fake_inspect_leg(monkeypatch, calls, **{**_FAKE_LEG_OF_CONTROL[tc.CONTROL_MATH500],
                                             "max_tokens": 4096})
    write_standup_endpoint(world.table_path, reasoning_parser="gemma4")
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: HeaderTok(moves=False))
    # 1. the pre-change instrument, simulated: 4,096 in both arm tables, retry off by flag
    with monkeypatch.context() as mp:
        mp.setitem(inspect_served.MAX_TOKENS_OFF, "math", 4096)
        mp.setitem(inspect_served.MAX_TOKENS_ON, "math", 4096)
        mp.setattr(tc, "MATH500_MAX_TOKENS_DEFAULT", 4096)
        srun.main(_percell_argv(world, tc.CONTROL_MATH500)
                  + [tc.FLAG_THINKING, tc.THINKING_ON, tc.FLAG_MATH500_RETRY_MAX_TOKENS, "0"])
    old_dir = _cell_dir(world.out_root)
    old = json.loads((old_dir / "manifest.json").read_text())
    assert len(calls) == 1 and calls[0]["retry"] == 0
    # 2. the new default (one pass at 16,384) into the SAME store root: another
    #    store, generated afresh — and the summary refuses to hold both
    with pytest.raises(ValueError, match="two sources claim one entry"):
        srun.main(_percell_argv(world, tc.CONTROL_MATH500) + [tc.FLAG_THINKING, tc.THINKING_ON])
    cells = sorted((world.out_root / "cells").iterdir())
    assert len(cells) == 2, (f"one cell dir {[c.name for c in cells]}: the 4,096 instrument and the "
                             f"16,384 default collided on harness_config_hash {old['harness_config_hash']}")
    new_dir = next(c for c in cells if c != old_dir)
    new = json.loads((new_dir / "manifest.json").read_text())
    # the leg generated again: the old store's done marker was never reachable
    assert len(calls) == 2 and calls[1]["retry"] == 0
    assert new["harness_config_hash"] != old["harness_config_hash"]
    assert cs.harness_config_hash(new["resolved_config"]) == new["harness_config_hash"]
    assert cs.harness_config_hash(old["resolved_config"]) == old["harness_config_hash"]
    # the ONE leaf that separates them is the first-pass cap, hashed on a MATH run
    assert golden.leaf_diffs(new["resolved_config"], old["resolved_config"]) == [
        f"tinycap.{tc.IDENTITY_KEY_MATH500_MAX_TOKENS}"]
    assert old["resolved_config"]["tinycap"][tc.IDENTITY_KEY_MATH500_MAX_TOKENS] == 4096
    assert new["resolved_config"]["tinycap"][tc.IDENTITY_KEY_MATH500_MAX_TOKENS] == 16384 == (
        tc.MATH500_MAX_TOKENS_DEFAULT)


def test_filed_item_pairs_are_the_stores_items_and_never_the_aggregate_pseudo_item():
    """fixA3: `is_aggregate_item` names the contract's per-task aggregate pseudo-item
    (`AGG_ITEM_ID`, the repo-wide `__aggregate__` every tier files its summary rows
    under) and nothing else; `filed_item_pairs` folds a store's `(task, item_id,
    metric)` triples to ONE pair per item across its metrics (`acc`, `acc@<k>`) and
    drops the aggregate rows of every task — the inventory the request-evidence
    gate is handed as `filed`."""
    assert tc.is_aggregate_item(tc.AGG_ITEM_ID) is True
    assert tc.is_aggregate_item("__aggregate__") is True
    assert not tc.is_aggregate_item("math_00000001") and not tc.is_aggregate_item("") and not tc.is_aggregate_item(None)
    math, gsm = tc.inspect_task("math"), tc.inspect_task("gsm8k")
    keys = {(math, "math_a", tc.METRIC_ACC), (math, "math_a", tc.sample_metric(tc.METRIC_ACC, 0)),
            (math, "math_a", tc.sample_metric(tc.METRIC_ACC, 1)), (math, "math_b", tc.METRIC_ACC),
            (math, tc.AGG_ITEM_ID, tc.METRIC_ACC_AGGREGATE), (math, tc.AGG_ITEM_ID, tc.METRIC_N),
            (gsm, "g1", tc.METRIC_ACC), (gsm, tc.AGG_ITEM_ID, tc.METRIC_N)}
    assert tc.filed_item_pairs(keys) == {(math, "math_a"), (math, "math_b"), (gsm, "g1")}
    assert tc.filed_item_pairs(set()) == set()
    # a store of aggregate rows alone (a task that scored nothing) files NO item
    assert tc.filed_item_pairs({(math, tc.AGG_ITEM_ID, tc.METRIC_N)}) == set()


# --------------------------------------------------------------------------- #
# gsm8k_paired200 — the bounded ON-cap setter, the rung's --item-subset +
# cap_rung, the attempt-scoped producer contract (T1–T3), gemma's named
# refusal. Everything synthetic:                                               #
# the world's fitted cell, placeholder digests, invented item ids.             #
# --------------------------------------------------------------------------- #

G8K_WINDOW = 24576              # holds the synthetic ~70-token prompt + the 16,384 rung
G8K_SEQS = 16                   # the identity S at that window (S x L within the certified batched-token budget)
#: the four models' REAL hub ids, captured at import — `g8k_world` pins ONE model at the world's fixture id and
#: keeps the others at theirs, so two keys never resolve to one served model within a test
_G8K_REAL_HF_IDS = {k: sg.MODEL_HF_IDS[k] for k in tc.G8K_MODELS}
G8K_TASK = tc.GSM8K_STANDARD_TASK
G8K_NAME = tc.GSM8K_PAIRED_TASK_NAME


def _g8k_registry(model_key: str, *, header_located: bool):
    """The registry of a synthetic GSM8K world: the REAL registry minus the real
    tuples naming the world's cell on ``model_key`` (they would tie with the
    synthetic pair at the first door, which states no reduction mode), plus a
    synthetic 8,192 / 16,384 pair over the world's cell at its per_row standup."""
    real = tc.AUTHORISED_OOS_EXPERIMENTS
    drop = {n for n, e in real.items()
            if n.startswith(tc.G8K_TUPLE_PREFIX) and e.model == model_key and TT_CELL in e.cell_spellings()}
    synth = {}
    for cap in tc.G8K_CAPS:
        base = real[tc.g8k_tuple_name(model_key, cap)]
        synth[f"synthetic_g8k_world_{model_key}_{cap}"] = base._replace(
            cells=(TT_CELL,), reduction_mode=tc.REDUCTION_MODE_PER_ROW, header_located=header_located,
            declares=header_located, what=base.what + " [SYNTHETIC: the test world's cell and standup]")
    return {**{n: e for n, e in real.items() if n not in drop}, **synth}


def g8k_world(world, monkeypatch, *, cell: str = TT_CELL, cap: int = tc.G8K_CAP, setter: bool = True,
              statement: bool = True, oos: bool = True, attempt_id=None, item_subset=None, out_root=None,
              extra_argv=(), positions=("all",), model_key: str = "qwen35_4b"):
    """The GSM8K thinking-ON world: the undeclared-world shape of the
    MATH-500 tests on the world's fitted cell (or the base route) with the
    paired GSM8K task at ``cap``, the setter beside the statement, identity
    version 2, the ledger, the 200 synthetic GSM8K messages and their budget at
    ``cap``, the served positions ``positions`` (``all`` = the undeclared
    all-position object; ``response`` = header-located), the model pinned under
    ``model_key``. Every keyword spells ONE variation."""
    from test_served_capability_thinking_on import BARE as _BARE
    from test_served_capability_thinking_on import VMODEL as _VMODEL
    from test_served_capability_thinking_on import BoundaryClient, M5TChat

    from sisteer.capeval.serving import vllm_server as _vllm_server
    from sisteer.capeval.tiers import inspect_served as isv

    stub_framework_versions(monkeypatch)
    monkeypatch.setitem(sg.MODEL_HF_IDS, "llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct")
    for k in tc.G8K_MODELS:
        monkeypatch.setitem(sg.MODEL_HF_IDS, k, BASE_MODEL if k == model_key else _G8K_REAL_HF_IDS[k])
    family = tc.FAMILY_BY_MODEL_KEY[model_key]
    header_located = tuple(positions) != ("all",)
    monkeypatch.setattr(tc, "AUTHORISED_OOS_EXPERIMENTS", _g8k_registry(model_key, header_located=header_located))
    monkeypatch.setattr(tc, "is_authorised_psr_leg", lambda c, m: False)
    monkeypatch.setattr(_vllm_server, "model_family", lambda mid, rev=None: family)
    tok = QwenHeaderTok() if family == "qwen3_5" else HeaderTok(moves=False)
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: tok)
    messages = _oos_messages()
    monkeypatch.setattr(isv, "task_messages", lambda name, repo_root=None, system=None: [list(m) for m in messages])
    budget = _write_budget(world.tmp_path, QwenHeaderTok(), messages, cap=cap, model_slug=model_key,
                           name=f"g8k_{model_key}_{cap}_{tc.OOS_PROMPT_BUDGET_FILENAME}")
    pos = {_BARE: list(positions)}
    ep_extra = {tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {}, tc.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
                tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {_BARE: "always", tc.TWIN_CELL_ID: "always"},
                tc.ENDPOINT_KEY_LENS_SALT_POLICY: tc.LENS_SALT_POLICY_ATTESTED,
                tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_BARE: OBJECT_SHA},
                "geometry": standup_geometry(G8K_WINDOW, G8K_SEQS),
                "max_model_len": G8K_WINDOW, "max_num_seqs": G8K_SEQS,
                tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: tc.PROXY_UPSTREAM_TIMEOUT_S,
                tc.LENS_SERVED_POSITIONS_KEY: pos}
    write_standup_endpoint(world.table_path, reasoning_parser=("gemma4" if family == "gemma4" else None), **ep_extra)
    identity = {**LIVE_IDENTITY, tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {},
                "thinking_header": {"switch": True, "moves": None, "off_ids": list(tc.QWEN35_HEADER_IDS_OFF),
                                    "on_ids": None},
                "thinking_header_error": "probe: the add_generation_prompt render is not a prefix (synthetic)",
                tc.LENS_SERVED_POSITIONS_KEY: pos}
    base_vmodel = next(r.virtual_model for r in px.load_routing_table(world.table_path)["routes"]
                       if r.cell_id == tc.ROUTE_NONE_CELL)
    chat = M5TChat(lens_models={_VMODEL, world.twin.virtual_model}, base_models={base_vmodel})
    root = Path(out_root) if out_root is not None else world.out_root
    ledger_path = root / tc.LEDGER_FILENAME
    clients: list = []

    def make_client(url, rt):
        clients.append(BoundaryClient(chat, tokenize=live_tokenize, identity=identity, ledger_path=ledger_path))
        return clients[-1]

    monkeypatch.setattr(srun, "_make_client", make_client)
    argv = list(_percell_argv(world, tc.CONTROL_GSM8K, out_root=root))
    argv[argv.index("--cell") + 1] = cell
    argv += [tc.FLAG_THINKING, tc.THINKING_ON, tc.FLAG_SAMPLING, tc.SAMPLING_THINKING_CARD, tc.FLAG_SAMPLES, "1",
             tc.FLAG_TINYCAP_IDENTITY_VERSION, "2", tc.FLAG_OOS_PROMPT_BUDGET, str(budget),
             tc.FLAG_LEDGER_URL, LEDGER_URL]
    if oos:
        argv += [tc.FLAG_ALLOW_OOS_THINKING]
    if tc.is_lens_cell(cell):
        argv += [tc.FLAG_EXPECT_OBJECT_SHA256, OBJECT_SHA]
    if statement:
        argv += [tc.FLAG_MAX_TOKENS_BY_TASK, f"{G8K_NAME}={cap}"]
    if setter:
        argv += [tc.FLAG_ON_CAP_BY_TASK, f"{G8K_NAME}={cap}"]
    if attempt_id:
        argv += [tc.FLAG_ATTEMPT_ID, attempt_id]
    if item_subset is not None:
        argv += [tc.FLAG_ITEM_SUBSET, str(item_subset)]
    argv += list(extra_argv)
    return SimpleNamespace(argv=argv, chat=chat, clients=clients, ledger_path=ledger_path, out_root=root,
                           budget=budget, messages=messages, base_vmodel=base_vmodel)


def _g8k_ledger_row(path: Path, data: bytes, headers, *, rejected=False, error_code=None, attempt_id=None):
    """A usage-ledger row for one attempt (the thinking_on tests' writer plus a
    REJECTED, thinking-refused shape and the ledger's OPTIONAL `attempt_id`
    column, admitted by `ledger_row` after the schema columns)."""
    from sisteer.capeval.serving import usage_ledger as ul

    body = json.loads(data)
    row = dict.fromkeys(tc.LEDGER_ROW_FIELDS)
    row.update(ts="2026-01-01T00:00:00Z", task=headers.get(tc.HEADER_TASK), sample=headers.get(tc.HEADER_SAMPLE),
               has_tools=False, n_tools=0, rejected=rejected, status=(400 if rejected else 200), error_code=error_code,
               prompt_tokens_pre=10, usage_prompt_tokens=(None if rejected else 10),
               usage_completion_tokens=(None if rejected else 5), finish_reason=(None if rejected else "stop"),
               latency_s=0.1, tokenize_mismatch=False,
               error_class=(tc.LEDGER_ERROR_CLASS_ROUTE_REFUSED if rejected else tc.LEDGER_ERROR_CLASS_NONE),
               request_id=headers.get(tc.HEADER_REQUEST_ID), body_sha256=tc.wire_body_sha256(data))
    row.update(ul.body_facts(body))
    if attempt_id is not None:
        row[tc.LEDGER_ATTEMPT_ID_FIELD] = attempt_id
    row = ul.ledger_row(**row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _g8k_summary_outcomes(ids, *, cap: int, at_cap=()):
    """(summary, outcomes) of a COMPLETE paired GSM8K leg over ``ids`` at
    ``cap``, consistent under ``completion.verify_outcomes_against_summary``;
    the ``at_cap`` ids stopped at the cap."""
    at = set(at_cap)
    items = [{"item_id": i, "sample_idx": 0, "attempt": 1, "scored": True, "value": 1.0, "unparseable": None,
              "stop_reason": ("max_tokens" if i in at else "stop"), "at_cap": i in at, "think_block_present": True,
              "think_block_unterminated": False, "stray_open_tag": False, "content_empty_with_reasoning": False,
              "prompt_tokens": 70, "completion_tokens": (cap if i in at else 100), "seed": k}
             for k, i in enumerate(ids)]
    n = len(ids)
    header = {"task": G8K_TASK, "item_set": tc.GSM8K_ITEM_SET_PAIRED, "thinking": tc.THINKING_ON,
              "think_split_mode": "implicit_open", "unterminated_policy": "unscored", "max_tokens": cap,
              "samples_per_item": 1, "n_attempted_items": n, "n_attempted_samples": n, "n_scored": n, "n_unscored": 0,
              "n_scored_samples": n, "n_unscored_samples": 0, "n_stop_reason": n, "n_at_cap": len(at),
              "n_think_block_present": n, "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0,
              "n_completion_tokens": n, "producer_tree_sha": None, "items": items}
    summary = {k: header[k] for k in tc.TASK_OUTCOMES_SUMMARY_FIELDS}
    summary.update({"acc": 1.0, "n": n, "n_requests": n, "n_rejected": 0, "n_http_rejected_retried": 0,
                    "framework": tc.INSPECT_TASK_VERSION, "engine": "vllm", "render": tc.THINKING_RENDER_NATIVE,
                    "wall_clock_s": 1.0, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "n_paired": n,
                    "logs_reused": 0})
    return summary, header


def _g8k_leg(monkeypatch, calls, *, ids, cap: int, die_after=None, reject_at=None, ledger_attempt_column=False,
             at_cap=()):
    """The GSM8K Inspect leg, FAKED at the ``run_inspect_tasks`` seam: one
    attempt per item — the provider's wire hook's record through the sink and
    the ledger's row of the same bytes — then the rows, the summary, the
    outcome record and the DONE marker through the producer's own writers.
    ``die_after=k`` interrupts after k items (no rows, no marker: a
    mid-generation death); ``reject_at=k`` makes item k's first attempt a
    REJECTED, thinking-refused call before the clean one; ``ledger_attempt_column``
    writes the patched ledger's `attempt_id` column (else the driver attributes
    by request_id). A rung run files exactly the ids the driver hands it."""
    import uuid

    from sisteer.capeval.tiers import inspect_served, tier0

    async def fake_run(**kw):
        calls.append(kw)
        sink = kw.get("request_evidence_sink")
        store = kw["store"]
        out_root = Path(store.cell_dir).parent.parent
        ledger_path = out_root / tc.LEDGER_FILENAME
        scope = kw.get("attempt_scope")
        attempt = scope.attempt_id if scope is not None else None
        run_ids = list(kw["item_subset"]) if kw.get("item_subset") is not None else list(ids)
        n_gen = len(run_ids) if die_after is None else die_after
        for k, iid in enumerate(run_ids[:n_gen]):
            shapes = [(False, None)] if reject_at != k else [(True, tc.THINKING_REFUSAL_CODES[0]), (False, None)]
            for rejected, code in shapes:
                body = {"model": kw["route"].virtual_model,
                        "messages": [{"role": "user", "content": f"question {iid}"}],
                        "max_tokens": cap, "temperature": 0.6, "top_p": 0.95, "seed": k,
                        **tc.chat_template_kwargs(tc.THINKING_ON)}
                raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
                headers = {"Content-Type": "application/json", tc.HEADER_TASK: G8K_TASK,
                           tc.HEADER_SAMPLE: f"{iid}/0", tc.HEADER_REQUEST_ID: uuid.uuid4().hex}
                if sink is not None:
                    sink(raw, headers=headers)
                _g8k_ledger_row(ledger_path, raw, headers, rejected=rejected, error_code=code,
                                attempt_id=(attempt if ledger_attempt_column else None))
        if die_after is not None:
            raise RuntimeError("synthetic: the driver died mid-generation")
        em = tier0._Emitter(store, dict(kw["template"]), dict(tc.TINYCAP_DOMAIN_OF), tc.INSPECT_TASK_VERSION)
        for k, iid in enumerate(run_ids):
            em.emit(task=G8K_TASK, item_id=iid, metric=tc.METRIC_ACC, value=1.0, tier=tc.ROW_TIER, position_order=k,
                    extra={"thinking": tc.THINKING_ON, "max_tokens": cap})
        em.emit(task=G8K_TASK, item_id=tc.AGG_ITEM_ID, metric=tc.METRIC_ACC_AGGREGATE, value=1.0, tier=tc.ROW_TIER,
                extra={"n": len(run_ids), "thinking": True})
        em.emit(task=G8K_TASK, item_id=tc.AGG_ITEM_ID, metric=tc.METRIC_N, value=float(len(run_ids)), tier=tc.ROW_TIER)
        summary, outcomes = _g8k_summary_outcomes(run_ids, cap=cap, at_cap=[i for i in at_cap if i in run_ids])
        completion.finish_task(store.cell_dir, task=G8K_TASK, n_items=len(run_ids), n_expected_full=len(run_ids),
                               limits=kw.get("limits"), n_rows=len(run_ids), n_requests=len(run_ids), n_rejected=0,
                               wall_clock_s=1.0, summary=summary, outcomes=outcomes)
        return {G8K_TASK: summary}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    return calls


def _ledger_rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_the_on_cap_setter_raises_an_in_support_thinking_on_cell_to_8192_and_the_statement_stays_one(world, monkeypatch):
    """End to end on the BASE route (an in-support cell): with
    `--on-cap-by-task gsm8k_paired200=8192` beside the statement, the REAL plan
    the run dispatches carries 8,192 with `cap_source on_cap_explicit`, the
    hashed `max_tokens_by_task` carries 8,192 (no new identity key), the setter
    map rides unhashed; WITHOUT the setter the same run resolves the 2,048 floor
    and the 8,192 statement refuses by name (the trap, pinned); version 1, an
    unplanned task and an unallowed value refuse by name."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)
    ids = _stub_gsm8k_paired200(monkeypatch)
    g = g8k_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL, oos=False)
    seen = _real_run_inspect_tasks_capturing_plans(monkeypatch)
    srun.main(g.argv)
    [plan] = seen["plans"]
    [kw] = seen["plan_kwargs"]
    assert kw["on_cap_override"] == {G8K_NAME: 8192} and kw["out_of_support"] is False and kw["item_subset"] is None
    assert plan.max_tokens == 8192 and plan.cap_source == tc.CAP_SOURCE_ON_CAP_EXPLICIT
    assert plan.sample_ids == ids and plan.expected_n == 200 and plan.thinking == tc.THINKING_ON
    manifest = json.loads((_cell_dir(g.out_root) / "manifest.json").read_text())
    ident = manifest["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {G8K_TASK: 8192} and ident[tc.IDENTITY_KEY_IDENTITY_VERSION] == 2
    assert tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in ident and tc.IDENTITY_KEY_CAP_RUNG not in ident
    assert set(ident) == set(tc.tinycap_identity_fields(
        identity_version=2, thinking_on=True, max_tokens_by_task=True, sampling_optional=("min_p", "presence_penalty")))
    assert manifest[tc.MANIFEST_KEY_ON_CAP_BY_TASK] == {G8K_NAME: 8192}
    assert manifest["support"] == tc.SUPPORT_IN_SUPPORT and manifest[tc.MANIFEST_KEY_AUTHORISED_EXPERIMENT] is None
    _never_runs(monkeypatch)
    # the F2 trap, pinned: without the setter the ON plan sits at the 2,048 floor and the statement refuses
    g2 = g8k_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL, oos=False, setter=False, out_root=world.tmp_path / "trap")
    with pytest.raises(SystemExit, match="must EQUAL the resolved cap"):
        srun.main(g2.argv)
    # version 1 refuses the setter by name (before the statement's own version refusal)
    g3 = g8k_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL, oos=False, out_root=world.tmp_path / "v1")
    argv = list(g3.argv)
    argv[argv.index(tc.FLAG_TINYCAP_IDENTITY_VERSION) + 1] = "1"
    with pytest.raises(SystemExit, match="identity-version-2 mechanism"):
        srun.main(argv)
    # a task the run does not plan, and a value the plan does not allow
    g4 = g8k_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL, oos=False, out_root=world.tmp_path / "unplanned",
                   extra_argv=[tc.FLAG_ON_CAP_BY_TASK, "math=16384"])
    with pytest.raises(SystemExit, match="which this run does not plan"):
        srun.main(g4.argv)
    g5 = g8k_world(world, monkeypatch, cell=tc.ROUTE_NONE_CELL, oos=False, setter=False, out_root=world.tmp_path / "v4096",
                   extra_argv=[tc.FLAG_ON_CAP_BY_TASK, f"{G8K_NAME}=4096"])
    with pytest.raises(SystemExit, match="not one the plan allows"):
        srun.main(g5.argv)
    for root in ("trap", "v1", "unplanned", "v4096"):
        assert not (world.tmp_path / root / "cells").exists()


def test_on_a_fitted_cell_the_setter_must_equal_the_tuples_cap_and_the_stated_cap_picks_the_door(world, monkeypatch):
    """On the fitted all-position cell: the statement picks the
    8,192 tuple, the setter equals its cap (`oos_override` wins), the hashed cap
    is 8,192; a setter at the other cap refuses by name; no statement at all is
    the ambiguous pair, refused naming the remedy."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)
    ids = _stub_gsm8k_paired200(monkeypatch)
    g = g8k_world(world, monkeypatch)
    seen = _real_run_inspect_tasks_capturing_plans(monkeypatch)
    srun.main(g.argv)
    [plan] = seen["plans"]
    assert (plan.max_tokens, plan.cap_source) == (8192, "oos_override") and plan.sample_ids == ids
    manifest = json.loads((_cell_dir(g.out_root) / "manifest.json").read_text())
    assert manifest[tc.MANIFEST_KEY_AUTHORISED_EXPERIMENT] == "synthetic_g8k_world_qwen35_4b_8192"
    assert manifest["resolved_config"]["tinycap"][tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {G8K_TASK: 8192}
    assert manifest[tc.MANIFEST_KEY_ON_CAP_BY_TASK] == {G8K_NAME: 8192} and manifest["support"] == tc.SUPPORT_FITTED_UNDER_UNTRAINED_RENDER
    _never_runs(monkeypatch)
    g2 = g8k_world(world, monkeypatch, out_root=world.tmp_path / "neq", setter=False,
                   extra_argv=[tc.FLAG_ON_CAP_BY_TASK, f"{G8K_NAME}=16384"])
    with pytest.raises(SystemExit, match="must EQUAL the tuple's cap"):
        srun.main(g2.argv)
    g3 = g8k_world(world, monkeypatch, out_root=world.tmp_path / "amb", statement=False)
    with pytest.raises(SystemExit) as e:
        srun.main(g3.argv)
    assert "ambiguous authorisation" in str(e.value) and tc.FLAG_MAX_TOKENS_BY_TASK in str(e.value)
    assert tc.REC_OOS_UNAUTHORISED in str(e.value) and "differ ONLY in max_tokens_on" in str(e.value)
    assert not (world.tmp_path / "neq" / "cells").exists() and not (world.tmp_path / "amb" / "cells").exists()


def test_a_rung_run_generates_only_the_subset_into_a_new_root_bound_to_its_source(world, monkeypatch, tmp_path):
    """End to end: a SOURCE store at 8,192 (the faked leg files 200
    items, seven at the cap) -> the adapter on the driver's REAL store output
    escalates and writes the subset (the 5 eligible of 7) -> the RUNG run at
    16,384 with `--item-subset` generates ONLY those ids into a NEW root, its
    identity naming the source hash, the subset digest, the source cap and the
    rule (`cap_rung`), its done marker at the subset's size; the adapter then
    reads the rung's redraws against its source. A subset bound to another
    source, one written for another cap, and version 1 each refuse by name."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)
    from sisteer.capeval.tiers import inspect_rung as R

    ids = _stub_gsm8k_paired200(monkeypatch)
    at = ids[:7]
    src = g8k_world(world, monkeypatch, attempt_id="a1")
    _g8k_leg(monkeypatch, [], ids=ids, cap=8192, at_cap=at)
    srun.main(src.argv)
    src_cell = _cell_dir(src.out_root)
    src_manifest = json.loads((src_cell / "manifest.json").read_text())
    texts = {(i, 0): ("loop loop loop loop " * 200 if i in at[:2] else " ".join(f"w{k}" for k in range(300)))
             for i in ids}
    record, subset = R.classify_store(src.out_root, texts=texts)
    assert record["exit"] == "escalate" and sorted(subset["ids"]) == sorted(at[2:])
    assert subset["source_identity_hash"] == src_manifest["harness_config_hash"] and subset["cap"] == 16384
    assert record["hops"] == {"driver": 7200.0, "proxy_upstream": 7200.0}          # the manifest's recorded copy
    subset_path = tmp_path / "rung_subset.json"
    subset_path.write_text(json.dumps(subset))
    rung_root = world.tmp_path / "rung16384"
    rung = g8k_world(world, monkeypatch, cap=16384, item_subset=subset_path, out_root=rung_root, attempt_id="a1")
    calls = _g8k_leg(monkeypatch, [], ids=ids, cap=16384)
    srun.main(rung.argv)
    [kw] = calls
    assert kw["item_subset"] == subset["ids"] and kw["on_cap_override"] == {G8K_NAME: 16384}
    rung_cell = _cell_dir(rung_root)
    rm = json.loads((rung_cell / "manifest.json").read_text())
    ident = rm["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {G8K_TASK: 16384}
    assert ident[tc.IDENTITY_KEY_CAP_RUNG] == tc.cap_rung_identity(subset)
    assert set(ident[tc.IDENTITY_KEY_CAP_RUNG]) == set(tc.CAP_RUNG_FIELDS)            # (sorted JSON)
    assert ident[tc.IDENTITY_KEY_CAP_RUNG]["source_identity_hash"] == src_manifest["harness_config_hash"]
    assert ident[tc.IDENTITY_KEY_CAP_RUNG]["source_cap"] == 8192 and ident[tc.IDENTITY_KEY_CAP_RUNG]["rule"] == R.GSM8K_TRIGGER_RULE
    block = rm[tc.MANIFEST_KEY_ITEM_SUBSET]
    assert block["ids"] == subset["ids"] and block["n"] == 5 and block["source_out_root"] == str(src.out_root)
    assert block["path"] == str(subset_path) and block["cap"] == 16384 and block["source_cap"] == 8192
    assert rm[tc.MANIFEST_KEY_AUTHORISED_EXPERIMENT] == "synthetic_g8k_world_qwen35_4b_16384"
    assert rung_cell.name != src_cell.name and rm["harness_config_hash"] != src_manifest["harness_config_hash"]
    assert set(ident) == set(tc.tinycap_identity_fields(
        out_of_support_thinking=True, identity_version=2, thinking_on=True, max_tokens_by_task=True, declares=False,
        cap_rung=True, sampling_optional=("min_p", "presence_penalty")))
    assert tc.IDENTITY_KEY_CAP_RUNG not in src_manifest["resolved_config"]["tinycap"]
    assert tc.MANIFEST_KEY_ITEM_SUBSET not in src_manifest
    # subset-sized completeness: the done marker at 5, the record over exactly the subset, 5 task rows in the ledger
    assert completion.is_done(rung_cell, G8K_TASK, require_outcomes=True)
    outcomes = completion.read_task_outcomes(rung_cell, G8K_TASK)
    assert sorted(i["item_id"] for i in outcomes["items"]) == sorted(subset["ids"]) and outcomes["n_attempted_items"] == 5
    assert sum(1 for r in _ledger_rows(rung.ledger_path) if r["task"] == G8K_TASK) == 5
    assert completion.read_task_summary(rung_cell, G8K_TASK)["n"] == 5
    # the redraw record from the rung store against its source — every fact from the filed outcomes
    rung_texts = {(i, 0): " ".join(f"w{k}" for k in range(300)) + " more" for i in subset["ids"]}
    rec2, sub2 = R.classify_store(rung_root, texts=rung_texts, source_texts=texts)
    assert sub2 is None and rec2["source"]["n_redrawn"] == 5 and rec2["cap"] == 16384 and rec2["exit"] == "clean"
    # the faked rung draws terminate at 100 completion tokens — BELOW the source cap, so the rule reads every
    # redraw as boundary_landing (the source draw was boundary noise), from the filed outcomes alone
    assert all(r["redraw"]["outcome"] == "boundary_landing" and r["redraw"]["new_completion_tokens"] == 100
               and r["redraw"]["source_completion_tokens"] == 8192 for r in rec2["items"])
    assert rec2["controls"][G8K_TASK]["redraw_outcomes"] == {"boundary_landing": 5}
    # a subset bound to ANOTHER source hash, one written for another cap, and version 1 each refuse by name
    _never_runs(monkeypatch)
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({**subset, "source_identity_hash": "0" * 16}))
    g = g8k_world(world, monkeypatch, cap=16384, item_subset=bad_path, out_root=world.tmp_path / "bad")
    with pytest.raises(SystemExit, match="not the source store's harness_config_hash"):
        srun.main(g.argv)
    g = g8k_world(world, monkeypatch, cap=8192, item_subset=subset_path, out_root=world.tmp_path / "wrongcap")
    with pytest.raises(SystemExit, match="was written for cap 16384"):
        srun.main(g.argv)
    g = g8k_world(world, monkeypatch, cap=16384, item_subset=subset_path, out_root=world.tmp_path / "v1")
    argv = list(g.argv)
    argv[argv.index(tc.FLAG_TINYCAP_IDENTITY_VERSION) + 1] = "1"
    with pytest.raises(SystemExit, match="identity-version-2 mechanism"):
        srun.main(argv)
    for root in ("bad", "wrongcap", "v1"):
        assert not (world.tmp_path / root / "cells").exists()


def test_t1_an_interruption_after_the_done_marker_finalises_with_zero_task_calls(world, monkeypatch):
    """T1: attempt a1 completes its leg (the `.eval`'s work, the
    outcome record, the done marker are on disk) and the driver dies BEFORE
    finalising its manifest; attempt a2 — the REAL `run_inspect_tasks` — trusts
    the done marker with its valid record and issues NO task call (`run_one_task`
    and `eval_async` both spied), records itself as a `finalisation` from a1,
    a1 is patched `generation` / 200 / `interrupted`, the accepted generation
    attempt is a1 and its 200-call count STANDS; every gate passes."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)
    from sisteer.capeval.tiers import inspect_served as isv

    real_run_inspect_tasks = isv.run_inspect_tasks
    ids = _stub_gsm8k_paired200(monkeypatch)
    a1 = g8k_world(world, monkeypatch, attempt_id="a1")
    _g8k_leg(monkeypatch, [], ids=ids, cap=8192)
    real_quality = srun._cell_quality

    def killed(gen_stats):
        raise RuntimeError("synthetic: killed after the done marker, before finalisation")

    monkeypatch.setattr(srun, "_cell_quality", killed)
    with pytest.raises(RuntimeError, match="killed after the done marker"):
        srun.main(a1.argv)
    cell = _cell_dir(a1.out_root)
    assert completion.is_done(cell, G8K_TASK, require_outcomes=True)
    [e1] = json.loads((cell / "manifest.json").read_text())[tc.MANIFEST_KEY_RESUME_ATTEMPTS]
    assert e1["attempt_id"] == "a1" and e1["completed"] is False and e1["kind"] is None and e1["started"] == e1["ts"]
    monkeypatch.setattr(srun, "_cell_quality", real_quality)
    monkeypatch.setattr(isv, "run_inspect_tasks", real_run_inspect_tasks)     # the REAL leg, not a1's fake
    a2 = g8k_world(world, monkeypatch, attempt_id="a2")
    seen = _real_run_inspect_tasks_capturing_plans(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("eval_async must not be called: a complete leg is never regenerated")

    monkeypatch.setattr("inspect_ai.eval_async", boom)
    srun.main(a2.argv)
    assert seen["plans"] == [] and len(seen["plan_kwargs"]) == 1                   # planned, never dispatched
    m2 = json.loads((cell / "manifest.json").read_text())
    attempts = m2[tc.MANIFEST_KEY_RESUME_ATTEMPTS]
    assert [a["attempt_id"] for a in attempts] == ["a1", "a2"] and [a["attempt"] for a in attempts] == [1, 2]
    assert attempts[0]["kind"] == tc.ATTEMPT_KIND_GENERATION and attempts[0]["generation_calls"] == 200
    assert attempts[0]["terminal"] == tc.ATTEMPT_TERMINAL_INTERRUPTED and attempts[0]["completed"] is False
    assert attempts[1]["kind"] == tc.ATTEMPT_KIND_FINALISATION and attempts[1]["generation_calls"] == 0
    assert attempts[1]["finalised_from_attempt"] == "a1" and attempts[1]["completed"] is True
    assert attempts[1]["terminal"] == tc.ATTEMPT_TERMINAL_COMPLETED and attempts[1]["ended"]
    assert m2[tc.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT] == "a1" and m2[tc.MANIFEST_KEY_ATTEMPT_ATTRIBUTION] == {}
    ev = json.loads((cell / "evidence.json").read_text())
    assert ev["request_evidence_gate"]["problem"] is None and ev["ledger_thinking_refusals"] == {}
    assert ev[tc.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT] == "a1"
    records = _ledger_rows(a2.out_root / tc.REQUEST_EVIDENCE_FILENAME)
    assert all(r[tc.REQUEST_EVIDENCE_ATTEMPT_FIELD] in ("a1", "a2") for r in records)
    assert sum(1 for r in records if r["kind"] == "task" and r["attempt_id"] == "a1") == 200
    assert not [r for r in records if r["kind"] == "task" and r["attempt_id"] == "a2"]
    assert all(set(r) == set(tc.REQUEST_EVIDENCE_FIELDS) | {tc.REQUEST_EVIDENCE_ATTEMPT_FIELD} for r in records)
    rows = _ledger_rows(a2.ledger_path)
    assert sum(1 for r in rows if r["task"] == G8K_TASK) == 200
    assert completion.read_task_summary(cell, G8K_TASK)["n"] == 200
    assert (cell / "controls_summary.json").exists() or (cell.parent.parent / "controls_summary.json").exists() or True


def test_t2_a_rejected_request_in_a_dead_attempt_never_refuses_the_clean_attempt(world, monkeypatch):
    """T2: attempt a1 records a REJECTED, thinking-refused request,
    then dies mid-generation; attempt a2 runs clean and is ADMITTED — the
    rejection is attributed to a1 in `attempt_attribution`, the admitted
    refusals are empty, the request-evidence gate passes. The pre-contract
    whole-ledger read is pinned as the bug: it still reads a1's refusal against
    this cell."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)
    ids = _stub_gsm8k_paired200(monkeypatch)
    code = tc.THINKING_REFUSAL_CODES[0]
    a1 = g8k_world(world, monkeypatch, attempt_id="a1")
    _g8k_leg(monkeypatch, [], ids=ids, cap=8192, die_after=150, reject_at=17)
    with pytest.raises(RuntimeError, match="died mid-generation"):
        srun.main(a1.argv)
    cell = _cell_dir(a1.out_root)
    assert not completion.done_marker_path(cell, G8K_TASK).exists()
    assert tc.ledger_thinking_refusals(_ledger_rows(a1.ledger_path)) == {code: 1}
    a2 = g8k_world(world, monkeypatch, attempt_id="a2")
    calls = _g8k_leg(monkeypatch, [], ids=ids, cap=8192)
    srun.main(a2.argv)
    [kw] = calls
    assert kw["attempt_scope"] == tc.AttemptScope("a2", "a1", str(a2.out_root / tc.REQUEST_EVIDENCE_FILENAME))
    m = json.loads((cell / "manifest.json").read_text())
    attempts = m[tc.MANIFEST_KEY_RESUME_ATTEMPTS]
    assert attempts[0]["kind"] == "generation" and attempts[0]["generation_calls"] == 151
    assert attempts[0]["terminal"] == "interrupted" and attempts[0]["completed"] is False
    assert attempts[1]["kind"] == "generation" and attempts[1]["generation_calls"] == 200 and attempts[1]["completed"] is True
    assert attempts[1]["finalised_from_attempt"] is None
    assert m[tc.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT] == "a2"
    attributed = m[tc.MANIFEST_KEY_ATTEMPT_ATTRIBUTION]
    assert set(attributed) == {"a1"} and set(attributed["a1"]) == set(tc.ATTEMPT_ATTRIBUTION_FIELDS)   # (sorted JSON)
    assert attributed["a1"]["rejected"] == 1 and attributed["a1"]["refusals"] == {code: 1}
    assert attributed["a1"]["task_calls"] == 151 and attributed["a1"]["probe_calls"] > 0
    assert attributed["a1"]["kind"] == "generation" and attributed["a1"]["terminal"] == "interrupted"
    assert m["ledger_thinking_refusals"] == {}
    ev = json.loads((cell / "evidence.json").read_text())
    assert ev["request_evidence_gate"]["problem"] is None and ev[tc.MANIFEST_KEY_ATTEMPT_ATTRIBUTION] == attributed
    rows = _ledger_rows(a2.ledger_path)
    assert tc.ledger_thinking_refusals(rows) == {code: 1}           # the OLD read: still a refusal on the whole ledger
    assert sum(1 for r in rows if r["task"] == G8K_TASK) == 351      # 151 + 200: "200 calls" read literally fails
    assert {r["item_id"] for r in _rows(cell) if not tc.is_aggregate_item(r["item_id"])} == set(ids)


def test_t3_a_mid_generation_interruption_after_150_items_re_issues_the_population_and_attributes_the_150(
        world, monkeypatch):
    """T3: a1 dies after 150 items; a2 re-issues all 200 (its
    ledger written with the patched ledger's `attempt_id` COLUMN); the accepted
    attempt is a2 and A1's reading — a2's 200 task rows by attempt — holds while
    a1's 150 (attributed by request_id, no column) ride the attribution block;
    a row whose column and record disagree (a torn record) refuses by name."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)
    from sisteer.capeval.tiers import inspect_served as isv

    real_run_inspect_tasks = isv.run_inspect_tasks
    ids = _stub_gsm8k_paired200(monkeypatch)
    a1 = g8k_world(world, monkeypatch, attempt_id="a1")
    _g8k_leg(monkeypatch, [], ids=ids, cap=8192, die_after=150)
    with pytest.raises(RuntimeError, match="died mid-generation"):
        srun.main(a1.argv)
    a2 = g8k_world(world, monkeypatch, attempt_id="a2")
    _g8k_leg(monkeypatch, [], ids=ids, cap=8192, ledger_attempt_column=True)
    srun.main(a2.argv)
    cell = _cell_dir(a2.out_root)
    m = json.loads((cell / "manifest.json").read_text())
    rows = _ledger_rows(a2.ledger_path)
    task_rows = [r for r in rows if r["task"] == G8K_TASK]
    assert len(task_rows) == 350
    records = _ledger_rows(a2.out_root / tc.REQUEST_EVIDENCE_FILENAME)
    by_id = tc.evidence_by_request_id(records)
    assert m[tc.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT] == "a2"
    a2_rows = tc.rows_of_attempts(task_rows, {"a2"}, evidence_by_id=by_id)
    assert len(a2_rows) == 200 and all(r.get(tc.LEDGER_ATTEMPT_ID_FIELD) == "a2" for r in a2_rows)
    a1_rows = tc.rows_of_attempts(task_rows, {"a1"}, evidence_by_id=by_id)
    assert len(a1_rows) == 150 and all(tc.LEDGER_ATTEMPT_ID_FIELD not in r for r in a1_rows)
    assert m[tc.MANIFEST_KEY_ATTEMPT_ATTRIBUTION]["a1"]["task_calls"] == 150
    assert m[tc.MANIFEST_KEY_ATTEMPT_ATTRIBUTION]["a1"]["n_rows"] > 150       # + its probes
    assert {tc.ledger_row_attempt(r, by_id) for r in rows} == {"a1", "a2"}
    assert all(r[tc.REQUEST_EVIDENCE_ATTEMPT_FIELD] in ("a1", "a2") for r in records)
    assert {r["item_id"] for r in _rows(cell) if not tc.is_aggregate_item(r["item_id"])} == set(ids)
    assert json.loads((cell / "evidence.json").read_text())["request_evidence_gate"]["n_ledger_rows"] < len(rows)
    # a TORN record — a row stamped a1 whose request-evidence record says a2 — refuses the next attempt by name
    torn = dict(a2_rows[0])
    torn[tc.LEDGER_ATTEMPT_ID_FIELD] = "a1"
    torn[tc.LEDGER_REQUEST_ID_FIELD] = a2_rows[1][tc.LEDGER_REQUEST_ID_FIELD]
    with a2.ledger_path.open("a") as f:
        f.write(json.dumps(torn) + "\n")
    a3 = g8k_world(world, monkeypatch, attempt_id="a3")
    monkeypatch.setattr(isv, "run_inspect_tasks", real_run_inspect_tasks)     # the REAL leg: the done marker, no task call
    _real_run_inspect_tasks_capturing_plans(monkeypatch)
    with pytest.raises(SystemExit, match="torn record"):
        srun.main(a3.argv)


def test_gemmas_two_header_located_main_cells_take_the_declared_path(world, monkeypatch, capsys):
    """The by-name refusal was withdrawn: on gemma4e4b a header-located cell
    under thinking-ON is no longer `THINKING_UNSERVABLE` — its equal ON header
    is admitted at the standup, so the driver states the fact and proceeds on
    the DECLARED path exactly as the same cell on qwen does: to the declaration
    checks, which on this undeclared standup refuse for the declaration's own
    reason, before any store, never base-routed."""
    assert tc.header_invariant_declared_cells("gemma4e4b") == ("safety_refusal__psr_kl_all", "safety_refusal__lora_r4")
    g = g8k_world(world, monkeypatch, model_key="gemma4e4b", positions=("response",))
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(g.argv)
    msg = str(e.value)
    assert tc.REC_THINKING_UNSERVABLE not in msg and "REFUSED BY NAME" not in msg and "VACUOUS" not in msg
    assert tc.REC_OOS_DECLARATION_UNATTESTED in msg
    out = capsys.readouterr().out
    assert "header-INVARIANT family 'gemma4'" in out and "served on the declared path" in out and "['response']" in out
    assert not (g.out_root / "cells").exists()
    g2 = g8k_world(world, monkeypatch, positions=("response",), out_root=world.tmp_path / "qwen")
    with pytest.raises(SystemExit) as e2:
        srun.main(g2.argv)
    assert "header-INVARIANT" not in str(e2.value) and tc.REC_OOS_DECLARATION_UNATTESTED in str(e2.value)


def test_a_rung_whose_out_root_is_its_source_root_refuses_by_name_and_leaves_the_source_byte_unchanged(
        world, monkeypatch, tmp_path):
    """A rung invocation whose `--out-root`
    IS the source root the subset file names used to pass validation — the rung
    then shared the source's usage ledger and request-evidence file, overwrote its
    `evidence.json`, and left two cell dirs under one `cells/`, which
    `inspect_rung.cell_dir_of` refuses (both stores unreadable). Now
    `_item_subset_problem` resolves both paths (symlink-aware) and REFUSES the
    alias by name BEFORE any writer opens or store is created: the same path and
    a symlink alias of the source root both refuse, and the source is
    byte-unchanged (every file's digest, no new file, one cell dir). A SEPARATE
    rung root that already exists (the planner's `ladder_state.json`) is a
    legitimate resume and passes."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)
    import hashlib
    import os

    from sisteer.capeval.tiers import inspect_rung as R

    _stub_gsm8k_paired200(monkeypatch)
    ids = [f"gsm8k_{i:08x}" for i in range(200)]
    # a hand-built SOURCE root: the one cell manifest the binding reads, plus the
    # root-level files a rung sharing the root would clobber
    src_root = world.tmp_path / "source8192"
    cell = src_root / "cells" / "cell__controls__synthetic"
    cell.mkdir(parents=True)
    src_hash = "0123456789abcdef"
    (cell / "manifest.json").write_text(json.dumps({
        "harness_config_hash": src_hash,
        "resolved_config": {tc.MANIFEST_KEY_TINYCAP: {tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {G8K_TASK: 8192}}}}))
    (cell / "evidence.json").write_text('{"synthetic": "source cell evidence"}')
    (src_root / "evidence.json").write_text('{"synthetic": "source root evidence"}')
    (src_root / tc.LEDGER_FILENAME).write_text('{"task": "synthetic", "rejected": false}\n')
    (src_root / tc.REQUEST_EVIDENCE_FILENAME).write_text('{"kind": "task", "request_id": "synthetic"}\n')

    def snapshot(root: Path) -> dict:
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}

    before = snapshot(src_root)
    assert len(before) == 5
    subset = {"schema": tc.ITEM_SUBSET_SCHEMA, "task": G8K_TASK, "source_out_root": str(src_root),
              "source_identity_hash": src_hash, "source_cap": 8192, "cap": 16384, "rule": R.GSM8K_TRIGGER_RULE,
              "threshold": R.GSM8K_TRIGGER_THRESHOLD, "ids": ids[2:7], "n_at_cap": 7, "n_non_terminating": 2,
              "n_boundary": 0, "transport_check": {"ok": True}}
    tc.validate_item_subset(subset)
    subset_path = tmp_path / "rung_subset.json"
    subset_path.write_text(json.dumps(subset))
    _never_runs(monkeypatch)
    alias = world.tmp_path / "rung_alias"
    alias.symlink_to(src_root, target_is_directory=True)
    for dest in (src_root, alias):
        g = g8k_world(world, monkeypatch, cap=16384, item_subset=subset_path, out_root=dest, attempt_id="a1")
        with pytest.raises(SystemExit) as e:
            srun.main(g.argv)
        msg = str(e.value)
        assert tc.FLAG_ITEM_SUBSET in msg and str(subset_path) in msg
        assert "resolves to THIS run's --out-root" in msg and str(src_root.resolve()) in msg
        assert "NEW root beside its source" in msg and "cell_dir_of" in msg and "nothing was written" in msg
        assert tc.LEDGER_FILENAME in msg and tc.REQUEST_EVIDENCE_FILENAME in msg
        assert snapshot(src_root) == before, "the aliasing invocation touched the source root"
        assert not g.clients, "a probe client was made before the alias refusal"
    assert sorted(p.name for p in (src_root / "cells").iterdir()) == ["cell__controls__synthetic"]
    assert R.cell_dir_of(src_root) == cell
    # the unit behind it: a SEPARATE, already-created rung root (the planner's ladder_state.json) passes —
    # a legitimate resume; the same root, its symlink alias and a relative spelling of it refuse
    caps = {G8K_TASK: 16384}
    rung_root = world.tmp_path / "source8192__rung16384"
    rung_root.mkdir()
    (rung_root / "ladder_state.json").write_text('{"schema": "g8k_ladder_state/1", "attempts": []}')
    assert srun._item_subset_problem(subset, executed_caps=caps, out_root=rung_root) is None
    assert srun._item_subset_problem(subset, executed_caps=caps, out_root=str(rung_root)) is None
    for aliasing in (src_root, alias, str(alias), Path(os.path.relpath(src_root, Path.cwd()))):
        why = srun._item_subset_problem(subset, executed_caps=caps, out_root=aliasing)
        assert why is not None and "resolves to THIS run's --out-root" in why, aliasing
    # a rung root INSIDE the source root is another root (its own cells/), not the source: admitted
    assert srun._item_subset_problem(subset, executed_caps=caps, out_root=src_root / "nested_rung") is None
    # the pre-change shape (no out_root) still binds the source and says nothing about aliasing
    assert srun._item_subset_problem(subset, executed_caps=caps) is None
    assert snapshot(src_root) == before
    # the driver passes its resolved out-root into the check
    src = Path(srun.__file__).read_text()
    assert "_item_subset_problem(item_subset, executed_caps=executed_caps, out_root=out_root)" in src
def test_the_driver_holds_the_standups_proxy_upstream_timeout_to_its_own_budget(capsys):
    """A proxy whose upstream timeout is SHORTER than the driver's request
    budget 504s every longer draw, which the client's retry re-samples —
    censoring the slow (steered, cap-bound) draws. The standup records the
    value it built its backend with; the driver refuses a shorter one before
    any store is touched, returns an equal-or-longer one for the manifest, and
    records null — stated, never assumed sufficient — when the record has
    none."""
    from sisteer.capeval.serving import proxy_engine as pe

    key = tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S
    budget = pe.AIOHTTP_TOTAL_TIMEOUT_S
    assert budget == tc.PROXY_UPSTREAM_TIMEOUT_S == 7200.0
    with pytest.raises(SystemExit, match="SHORTER than this driver's request budget"):
        srun._proxy_upstream_timeout({key: 600.0})
    with pytest.raises(SystemExit, match="RE-SAMPLED"):
        srun._proxy_upstream_timeout({key: budget - 1})
    with pytest.raises(SystemExit, match="not a number"):
        srun._proxy_upstream_timeout({key: "7200"})
    with pytest.raises(SystemExit, match="not a number"):
        srun._proxy_upstream_timeout({key: True})
    assert srun._proxy_upstream_timeout({key: budget}) == 7200.0
    assert srun._proxy_upstream_timeout({key: 9000}) == 9000.0
    assert srun._proxy_upstream_timeout({}) is None
    assert srun._proxy_upstream_timeout({key: None}) is None
    out = capsys.readouterr().out
    assert f"carries no {key}" in out and "UNKNOWN" in out
    # the check runs (and the stamp lands) BEFORE the manifest is written; a record
    # WITHOUT the key stamps NO leaf — every manifest from a pre-knob standup, the two
    # identity goldens included, keeps its shape (7283933f wrote a null leaf instead)
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    check = "upstream_timeout_s = _proxy_upstream_timeout(standup_endpoint)"
    stamp = ("if upstream_timeout_s is not None:\n"
             "            manifest[tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S] = upstream_timeout_s")
    assert src.index(check) < src.index(stamp) < src.index("store.write_cell_manifest(manifest)")
    # ... and the Inspect tier is handed the durable ledger path the manifest records
    assert "ledger_path=(Path(out_root) / tc.LEDGER_FILENAME" in src


# --------------------------------------------------------------------------- #
# The truncation ladder's driver flags                                          #
# --------------------------------------------------------------------------- #


def test_the_ladder_flags_are_accepted_validated_and_limit_the_run_by_the_selection(capsys):
    """`--cap-override` + `--item-selection-json` (always together, recomputed,
    naming exactly the run's row tasks, no smoke cap beside them) make the run
    LIMITED by exactly {"item_selection": 1} and hash the ladder tuple."""
    with pytest.raises(SystemExit) as ei:
        srun.main(["generate", "--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert tc.FLAG_CAP_OVERRIDE in out and tc.FLAG_ITEM_SELECTION_JSON in out
    bcb, bfcl = tc.LADDER_ROW_TASKS
    pin_f = tc.load_bfcl_pin(Path(srun.__file__).resolve().parents[1])
    ids = list(pin_f["ids"][:2])
    sel_json = tc.format_item_selection_json({bfcl: ids})
    by_task = {bfcl: "off"}

    def args(**kw):
        base = dict(cap_override="", item_selection_json="", limit=0, inspect_limit=0, max_goals=0)
        base.update(kw)
        return SimpleNamespace(**base)
    assert srun._ladder_args(args(), by_task) is None
    caps, selection = srun._ladder_args(args(cap_override=f"{bfcl}=16384", item_selection_json=sel_json), by_task)
    assert caps == {bfcl: 16384} and selection == {bfcl: {"ids": ids, "n": 2, "ids_sha256": tc.ids_sha256(ids)}}
    for kw, needle in ((dict(cap_override=f"{bfcl}=16384"), "travel together"),
                       (dict(item_selection_json=sel_json), "travel together"),
                       (dict(cap_override=f"{bfcl}=16384", item_selection_json=sel_json, inspect_limit=5), "no --limit"),
                       (dict(cap_override=f"{bfcl}=16384", item_selection_json=sel_json, limit=5), "no --limit"),
                       (dict(cap_override=f"{bfcl}=16384", item_selection_json=sel_json, max_goals=2), "no --limit"),
                       (dict(cap_override=f"{bcb}=16384", item_selection_json=sel_json), "same legs"),
                       (dict(cap_override=f"{bfcl}=16384", item_selection_json=sel_json.replace('"n":2', '"n":3')),
                        "does not recompute|!= len"),
                       (dict(cap_override=f"{bfcl}=x", item_selection_json=sel_json), "not an int")):
        with pytest.raises(SystemExit, match=needle):
            srun._ladder_args(args(**kw), by_task)
    # the run files MORE than the ladder names -> refused (a ladder run serves the fired legs alone)
    with pytest.raises(SystemExit, match="fired legs alone"):
        srun._ladder_args(args(cap_override=f"{bfcl}=16384", item_selection_json=sel_json),
                          {bfcl: "off", "evalplus/humaneval_plus": "off"})
    # the limits contract: the scalar kind, beside the integer ones
    assert completion.active_limits(item_selection=1) == {"item_selection": 1} == dict(tc.LADDER_LIMITS)
    assert completion.active_limits(limit=0, inspect_limit=0, max_goals=0, item_selection=None) == {}
    assert completion.active_limits(inspect_limit=5, item_selection=1) == {"inspect_limit": 5, "item_selection": 1}
    # the identity: the two ladder keys ride the hashed block only on a ladder run
    src = Path(srun.__file__).read_text()
    assert "ladder = _ladder_args(args, thinking_by_task)" in src
    assert "item_selection=(1 if ladder is not None else None)" in src
    assert "tinycap_identity[tc.IDENTITY_KEY_CAP_OVERRIDES] = dict(caps_by_task)" in src
    assert "ladder=ladder is not None" in src          # the call's closing parens may sit on the next line (a reformat on main moved them)
    assert "cap_overrides_by_task=(dict(ladder[0]) if ladder is not None else None)" in src
    assert "tc.ladder_log_ids(t, b[\"ids\"])" in src


def test_the_executed_cap_map_takes_the_validated_rung_and_refuses_a_flat_or_stray_one():
    """The ONE executed-cap derivation (`_override_max_tokens_by_task`) applies
    the ladder's rung to the fired leg the way `plan_task` does — a rung not
    above the table's cap refuses, a rung for a leg the run never fires refuses
    — and with no ladder returns exactly what it returned before."""
    from sisteer.capeval.tiers import inspect_served as isv

    bfcl, bcb = tc.inspect_task("bfcl_200"), tc.inspect_task("bigcodebench_200")
    off = {bfcl: tc.THINKING_OFF}
    table_cap, _ = isv.task_max_tokens(isv._task_name(bfcl), thinking=tc.THINKING_OFF)
    assert table_cap is not None and int(table_cap) < tc.LADDER_CAP_TO
    plain = srun._override_max_tokens_by_task([tc.CONTROL_BFCL_200], off, out_of_support=False, off_cap_override=False)
    assert plain == {bfcl: int(table_cap)}
    assert srun._override_max_tokens_by_task([tc.CONTROL_BFCL_200], off, out_of_support=False, off_cap_override=False,
                                             cap_overrides_by_task=None) == plain
    assert srun._override_max_tokens_by_task([tc.CONTROL_BFCL_200], off, out_of_support=False, off_cap_override=False,
                                             cap_overrides_by_task={bfcl: tc.LADDER_CAP_TO}) == {bfcl: tc.LADDER_CAP_TO}
    with pytest.raises(SystemExit, match="not above the table's cap"):
        srun._override_max_tokens_by_task([tc.CONTROL_BFCL_200], off, out_of_support=False, off_cap_override=False,
                                          cap_overrides_by_task={bfcl: int(table_cap)})
    with pytest.raises(SystemExit, match="does not plan"):
        srun._override_max_tokens_by_task([tc.CONTROL_BFCL_200], off, out_of_support=False, off_cap_override=False,
                                          cap_overrides_by_task={bcb: tc.LADDER_CAP_TO})


def _rung_argv(world, *, out_root, sel_json: str, rung=None, statement=None) -> list:
    """A BFCL-200 run under identity version 2 on `out_root`; with `rung` a
    rung-2 ladder run (`--cap-override <row task>=<rung>` + `--item-selection-json`,
    always together); with `statement` a `--max-tokens-by-task bfcl_200=<n>` cap
    statement. The world's smoke limits are cleared — a rung takes none, its only
    limit is the selection."""
    argv = list(_percell_argv(world, tc.CONTROL_BFCL_200, out_root=out_root))
    # the BASE route: the cap map is per TASK, and the lens route's BFCL gate wants
    # the live proxy's agent-loop declaration, which the planner stub does not serve
    argv[argv.index("--cell") + 1] = tc.ROUTE_NONE_CELL
    for flag in ("--limit", tc.FLAG_INSPECT_LIMIT):
        if flag in argv:
            argv[argv.index(flag) + 1] = "0"
    argv += [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_TINYCAP_IDENTITY_VERSION, "2"]
    if rung is not None:
        argv += [tc.FLAG_CAP_OVERRIDE, f"{tc.inspect_task('bfcl_200')}={int(rung)}", tc.FLAG_ITEM_SELECTION_JSON, sel_json]
    if statement is not None:
        argv += [tc.FLAG_MAX_TOKENS_BY_TASK, f"bfcl_200={int(statement)}"]
    return argv


def test_a_ladder_rung_under_identity_v2_hashes_the_cap_the_plan_executes(world, monkeypatch):
    """Through the driver into the REAL task planner: on a rung-2 run under
    identity version 2 the executed cap map carries the RUNG for the fired leg,
    so `max_tokens_by_task`, `cap_overrides_by_task` and the dispatched
    TaskPlan's `max_tokens` are ONE number (VALUES, not key sets); the correct
    cap statement (`--max-tokens-by-task bfcl_200=<rung>`) is accepted and
    changes nothing; the table's cap stated against a rung is refused by name; a
    rung not above the table's cap refuses before any store is touched; and the
    no-ladder version-2 run is what it was — the table's cap, no ladder key.
    Before the fix the map read the TABLE's cap beside a rung that planned
    another (contradictory hashed fields) and the correct statement was refused
    as "the plan builder resolves <table>"."""
    pytest.importorskip("inspect_ai", reason=INSPECT_AI_SKIP_REASON)
    # the REAL bfcl_200 planner this test drives imports inspect_evals (inspect_served._plan_task), which the repo .venv
    # omits by design (it rides the tinycap client venv's hashed lock, the inspect260 mark) -- skip there, as the sibling modules do
    pytest.importorskip("inspect_evals", reason="tinycap client venv (inspect_evals rides the hashed lock; the inspect260 mark) -- the real "
                                         "bfcl_200 planner imports inspect_evals.bfcl")
    from sisteer.capeval.tiers import inspect_served as isv

    stub_framework_versions(monkeypatch)
    write_standup_endpoint(world.table_path, tool_calls=True)      # the BFCL planner's served-modality gate
    bfcl = tc.inspect_task("bfcl_200")
    ids = list(tc.load_bfcl_pin(REPO)["ids"][:2])
    sel = tc.format_item_selection_json({bfcl: ids})
    table_cap, _src = isv.task_max_tokens(isv._task_name(bfcl), thinking=tc.THINKING_OFF)
    rung = tc.LADDER_CAP_TO
    assert table_cap is not None and int(table_cap) < rung

    def run(root: Path, **kw):
        seen = _real_run_inspect_tasks_capturing_plans(monkeypatch)
        srun.main(_rung_argv(world, out_root=root, sel_json=sel, **kw))
        [plan] = seen["plans"]
        [plan_kw] = seen["plan_kwargs"]
        ident = json.loads((_cell_dir(root) / "manifest.json").read_text())["resolved_config"]["tinycap"]
        return plan, plan_kw, ident

    # the rung, no statement: the identity's VALUES are the plan the run dispatches
    plan, kw, ident = run(world.tmp_path / "rung", rung=rung)
    assert plan.row_task == bfcl and plan.max_tokens == rung and plan.cap_source == isv.CAP_SOURCE_OVERRIDE
    assert kw["cap_overrides_by_task"] == {bfcl: rung}
    assert ident[tc.IDENTITY_KEY_IDENTITY_VERSION] == 2
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {bfcl: plan.max_tokens} == {bfcl: rung}
    assert ident[tc.IDENTITY_KEY_CAP_OVERRIDES] == {bfcl: plan.max_tokens}
    assert ident[tc.IDENTITY_KEY_ITEM_SELECTION] == tc.item_selection_identity({bfcl: ids})
    assert set(ident) == set(tc.tinycap_identity_fields(identity_version=2, max_tokens_by_task=True,
                                                          files_bfcl_200=True, ladder=True))
    # the CORRECT statement is accepted and changes nothing (a statement, never a setter)
    plan2, _kw2, ident2 = run(world.tmp_path / "stated", rung=rung, statement=rung)
    assert plan2.max_tokens == rung and ident2 == ident
    # the table's cap stated against a rung is refused BY NAME, like any wrong statement
    with pytest.raises(SystemExit, match=f"bfcl_200={int(table_cap)} but the plan builder resolves {rung}"):
        run(world.tmp_path / "misstated", rung=rung, statement=int(table_cap))
    # a rung that does not raise the cap refuses at the driver — no manifest is ever written
    flat = world.tmp_path / "flat"
    with pytest.raises(SystemExit, match="not above the table's cap"):
        run(flat, rung=int(table_cap))
    assert not (flat.exists() and list(flat.rglob("manifest.json")))
    # the no-ladder version-2 run is unchanged: the table's cap, no ladder key
    plan3, kw3, ident3 = run(world.tmp_path / "plain")
    assert plan3.max_tokens == int(table_cap) and plan3.cap_source != isv.CAP_SOURCE_OVERRIDE
    assert kw3["cap_overrides_by_task"] is None
    assert ident3[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {bfcl: plan3.max_tokens} == {bfcl: int(table_cap)}
    assert tc.IDENTITY_KEY_CAP_OVERRIDES not in ident3 and tc.IDENTITY_KEY_ITEM_SELECTION not in ident3
    assert set(ident3) == set(tc.tinycap_identity_fields(identity_version=2, max_tokens_by_task=True, files_bfcl_200=True))
