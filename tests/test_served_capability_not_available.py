"""The DRIVER under the NO-SWITCH arm
``--thinking not_available`` on a fitted, undeclared, all-position Llama route, end to end through
``served_capability_run.main`` on the synthetic ``world`` of ``test_served_capability.py`` (the world's
fixture model is a Llama-named placeholder pinned under the ``llama8b`` key; its served tokenizer is
re-stubbed to the family's 4-id header): the same fake proxy / ``/v1/tokenize`` harness the thinking-ON
driver tests use, so the store it writes is the store the analysis fixtures read.

Pinned here — THE NO-SWITCH CERTIFICATION CLASS, emitted by the REAL producers:
  (A'') the header measured ONCE via the served ``/v1/tokenize`` with NO kwargs; ONE render probe under
        the single render whose stamp reads ``unspecified``; ``render_certified.cert_class no_switch``;
  (B')  ``lens_position_attested`` verbatim (positions ``all``, the digest == the pin, policy ``always``,
        the class's ``(cell-stable, always, unspecified)`` triple);
  (C'') the EXPLICIT ``lens_on_bite: {applicable: false, ...}`` record — never an absent block; the
        ordinary steering-bites control under ``probe_render_arm not_available``;
  the exact probe inventory on the wire (one render probe; bites on the route and the twin; NO off/on
  render probes, NO lens-ON bite); every body free of ``chat_template_kwargs`` and the declaration; the
  ledger's ``thinking`` NULL on every row; the request evidence ``enable_thinking_present`` False;
  the identity ``thinking.mode`` / ``thinking_by_task`` == ``not_available``, ``max_tokens_by_task`` at
  8,192 (the setter), NO ``out_of_support_thinking``, NO ``thinking_header_ids``; the manifest's
  ``thinking.enable_thinking`` null; the route block's ``bank_view``; every emitted row carrying
  ``thinking: not_available`` and the row provenance (``--bank-view``).
And the refusals BY NAME: ``not_available`` on a switched family; a CANDIDATE no-switch fact in production
(``--diagnostic`` admits it); a header-located cell under the no-switch arm.

Everything here is SYNTHETIC (a ``TESTHOST``-free standup, placeholder digests, invented items).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_served_capability import (  # noqa: F401  (`world` is a fixture)
    BASE_MODEL,
    LEDGER_URL,
    LIVE_IDENTITY,
    OBJECT_SHA,
    REV,
    TEMPLATE_HASH,
    TT_CELL,
    HEADER_TOK_TEMPLATE,
    REPO,
    _cell_dir,
    _percell_argv,
    live_tokenize,
    srun,
    standup_geometry,
    stub_framework_versions,
    world,
    write_standup_endpoint,
)
from test_served_capability_thinking_on import BARE, VMODEL, BoundaryClient, write_ledger_row

from sisteer.capeval.serving import geometry as _geom
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.tinycap import contracts as tc

NA = tc.THINKING_NOT_AVAILABLE
TASK = tc.GSM8K_STANDARD_TASK
TASK_NAME = tc.GSM8K_PAIRED_TASK_NAME
CAP = 8192
LLAMA_HEADER = list(tc.FAMILY_RENDER_FACTS["llama"].off_ids)
#: the GSM8K window the wave serves at (contracts / geometry tables, never a literal)
G8K_WINDOW = _geom.geometry_for("capability:gen_long_thinking").max_model_len
#: the S the world's standup record states at that window under the budget the geometry table resolves for the
#: fixture (the certified default; the llama8b measured row is the launcher's concern, not this harness's)
G8K_S = _geom.identity_seqs(G8K_WINDOW)


class LlamaHeaderTok:
    """A served tokenizer whose chat template carries NO thinking switch: the appended assistant header is
    the family's 4-id ``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` whatever is asked, and any
    ``enable_thinking`` kwarg is a contradiction the stub refuses (the driver must send none)."""

    chat_template = HEADER_TOK_TEMPLATE

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True, **kw):
        assert tc.ENABLE_THINKING_KEY not in kw, f"a no-switch render was passed {kw!r}"
        prompt = [1, 2, 3] + [4] * (len(messages) - 1)
        return prompt + (LLAMA_HEADER if add_generation_prompt else [])


EOT = 128009   # <|eot_id|>, the Llama-3 assistant turn's terminator


def _content_ids(text: str) -> list:
    """Deterministic, word-distinct content ids for a fake render (no two of chat_markers' sentinel words collide)."""
    return [200000 + sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 50000 for w in text.split()]


class UnconditionalHeaderTok(LlamaHeaderTok):
    """The NousResearch Llama-3.1 mirror's SHAPE (measured live 2026-09-12, smoke job 490518; chat_markers' mask-mirror
    ground truth): the template never consults `add_generation_prompt` and appends the 4-id assistant header after EVERY
    render, so agp True == agp False (the generation-prompt DIFF is empty where the header is always present) and a
    completed assistant turn renders header + content + <|eot_id|> and then the unconditional header again."""

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True, **kw):
        assert tc.ENABLE_THINKING_KEY not in kw, f"a no-switch render was passed {kw!r}"
        out = [1, 2, 3]
        for m in messages[1:]:
            out += (LLAMA_HEADER + _content_ids(m["content"]) + [EOT]) if m["role"] == "assistant" else [4]
        return out + LLAMA_HEADER


class FlagIgnoringNoHeaderTok(LlamaHeaderTok):
    """A template that ignores the flag WITHOUT appending the header: the generation prompt ends BEFORE where a completed
    assistant turn's content begins (a turn-opener token the prompt lacks) — chat_markers' refusal shape, refused here too."""

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True, **kw):
        out = [1, 2, 3]
        for m in messages[1:]:
            out += ([77] + _content_ids(m["content"]) + [EOT]) if m["role"] == "assistant" else [4]
        return out


class NoSwitchChat:
    """The scripted proxy for the no-switch world: every reply is stamped (lens legs ``(cell-stable, always,
    <render>)``, the base route ``(cell-stable, not-consulted, <render>)``) with the render regime derived
    from the body exactly as the proxy derives it — ``unspecified`` when no switch rides; the steered route's
    reply differs from its scale-0 twin's (the bites control); no reply carries a think tag or a reasoning
    channel. ``enable_thinking: true`` on this switch-less template is refused ``thinking_kwargs_inert``,
    as the real proxy refuses it; every body is recorded."""

    def __init__(self, *, lens_models, base_models):
        self.lens_models, self.base_models = set(lens_models), set(base_models)
        self.calls: list = []

    def chat_completions(self, payload):
        self.calls.append(json.loads(json.dumps(payload)))
        kwargs = payload.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {}
        flag = kwargs.get(tc.ENABLE_THINKING_KEY)
        if flag is True:
            raise px.ProxyError("proxy /v1/chat/completions returned 400: " + json.dumps({"error": {"code": px.THINKING_KWARGS_INERT}}))
        if tc.OOS_THINKING_DECLARATION_KEY in payload:
            raise px.ProxyError("proxy /v1/chat/completions returned 400: " + json.dumps({"error": {"code": px.THINKING_ON_NOT_LENS}}))
        regime = (tc.RENDER_REGIME_THINKING_OFF if flag is False else tc.RENDER_REGIME_UNSPECIFIED)
        lens = payload["model"] in self.lens_models
        if lens:
            text = "steered voice says alpha" if payload["model"] == VMODEL else "plain voice says beta"
            policy = "always"
        else:
            text = "base voice says gamma"
            policy = tc.POLICY_NOT_CONSULTED
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_SALT_REGIME: "cell-stable", tc.STAMP_KEY_RESUMED_POLICY: policy,
                                      tc.STAMP_KEY_RENDER_REGIME: regime}}


def _g8k_messages(n: int = tc.GSM8K_PAIRED_N):
    return [[{"role": "user", "content": f"Question {i}: a farm has {i + 3} hens laying {2 * i + 1} eggs each; how many eggs?"}]
            for i in range(n)]


def _budget(tmp_path, tok, messages, *, cap=CAP, arm=NA):
    lengths = [len(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)) for m in messages]
    doc = {"n": len(lengths), "max": max(lengths), "p99": tc.nearest_rank_percentile(lengths, 0.99),
           "p50": tc.nearest_rank_percentile(lengths, 0.50), "mean": round(sum(lengths) / len(lengths), 2),
           "arm": arm, "model": BASE_MODEL, "model_slug": "llama8b", "revision": REV,
           "chat_template_hash": TEMPLATE_HASH, "cap": int(cap), "cap_task": TASK,
           "max_plus_cap": max(lengths) + int(cap), "lengths": lengths,
           "render": {"add_generation_prompt": True, tc.CHAT_TEMPLATE_KWARGS_KEY: {}},
           "messages_source": tc.OOS_PROMPT_BUDGET_MESSAGES_SOURCE, "tree_sha": None,
           "produced_by": {"script": tc.OOS_PROMPT_BUDGET_SCRIPT, "tokenizer": "LlamaHeaderTok"}}
    tc.validate_oos_prompt_budget(doc)
    path = tmp_path / tc.OOS_PROMPT_BUDGET_FILENAME
    path.write_text(json.dumps(doc))
    return path


def certified_llama():
    """The tree's facts with the no-switch family CERTIFIED (only `evidence_status` moves) — the tree's OWN state since the
    GSM8K diagnostic smoke (2026-09-12); kept as a function so the world names the state it runs under."""
    facts = dict(tc.FAMILY_RENDER_FACTS)
    facts["llama"] = facts["llama"]._replace(evidence_status=tc.RENDER_EVIDENCE_CERTIFIED)
    return facts


def candidate_llama():
    """The pre-smoke state, SYNTHETIC now (only `evidence_status` moves): what production refuses by name."""
    facts = dict(tc.FAMILY_RENDER_FACTS)
    facts["llama"] = facts["llama"]._replace(evidence_status=tc.RENDER_EVIDENCE_CANDIDATE)
    return facts


def no_switch_world(world, monkeypatch, *, certified: bool = True, bank_view="main", extra_argv=(), chat=None, tok=None):  # noqa: F811
    """The fitted, undeclared, all-position Llama route: the standup record as the exporter writes it for such a
    cell (the served-positions map under the BARE key, an empty declarations map, the per-cell policy `always`,
    the digest under the bare key, the GSM8K window), the proxy identity echoing the map with NO thinking-header
    fact (a switch-less template attests none), the served tokenizer rendering the family's 4-id header, the
    200 GSM8K messages and their CPU budget (arm `not_available`) stubbed, and the argv the GSM8K launcher
    composes for a Llama fitted row (g8k_driver_argv): the no-switch arm, the `card` row, the cap as statement
    AND setter, identity v2, the budget, the ledger, the pin, `--bank-view`; NO out-of-support override."""
    from sisteer.capeval.tiers import inspect_served as isv
    from sisteer.capeval.serving import vllm_server as _vllm_server

    stub_framework_versions(monkeypatch)
    monkeypatch.setattr(tc, "FAMILY_RENDER_FACTS", certified_llama() if certified else candidate_llama())
    monkeypatch.setattr(_vllm_server, "model_family", lambda mid, rev=None: "llama")
    tok = tok or LlamaHeaderTok()
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: tok)
    messages = _g8k_messages()
    monkeypatch.setattr(isv, "task_messages", lambda name, repo_root=None, system=None: [list(m) for m in messages])
    budget = _budget(world.tmp_path, tok, messages)
    positions = {BARE: ["all"]}
    write_standup_endpoint(
        world.table_path,
        **{tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {}, tc.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
           tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {BARE: "always", tc.TWIN_CELL_ID: "always"},
           tc.ENDPOINT_KEY_LENS_SALT_POLICY: tc.LENS_SALT_POLICY_ATTESTED,
           tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {BARE: OBJECT_SHA},
           "geometry": standup_geometry(G8K_WINDOW, G8K_S), "max_model_len": G8K_WINDOW, "max_num_seqs": G8K_S,
           tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: tc.PROXY_UPSTREAM_TIMEOUT_S,
           tc.LENS_SERVED_POSITIONS_KEY: positions})
    identity = {**LIVE_IDENTITY, tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {}, tc.LENS_SERVED_POSITIONS_KEY: positions}
    chat = chat or NoSwitchChat(lens_models={VMODEL, world.twin.virtual_model}, base_models={"petri__base"})
    tokenized: list = []

    def recording_tokenize(body):
        tokenized.append(json.loads(json.dumps(body)))
        return live_tokenize(body)

    ledger_path = world.out_root / tc.LEDGER_FILENAME
    clients: list = []

    def make_client(url, rt):
        clients.append(BoundaryClient(chat, tokenize=recording_tokenize, identity=identity, ledger_path=ledger_path))
        return clients[-1]
    monkeypatch.setattr(srun, "_make_client", make_client)
    argv = (_percell_argv(world, tc.CONTROL_GSM8K)
            + [tc.FLAG_THINKING, NA, tc.FLAG_SAMPLING, tc.SAMPLING_CARD, tc.FLAG_SAMPLES, "1",
               tc.FLAG_MAX_TOKENS_BY_TASK, f"{TASK_NAME}={CAP}", tc.FLAG_ON_CAP_BY_TASK, f"{TASK_NAME}={CAP}",
               tc.FLAG_TINYCAP_IDENTITY_VERSION, "2", tc.FLAG_OOS_PROMPT_BUDGET, str(budget),
               tc.FLAG_LEDGER_URL, LEDGER_URL, tc.FLAG_EXPECT_OBJECT_SHA256, OBJECT_SHA])
    if bank_view is not None:
        argv += [tc.FLAG_BANK_VIEW, bank_view]
    argv += list(extra_argv)
    return SimpleNamespace(argv=argv, chat=chat, tokenized=tokenized, budget=budget, identity=identity, messages=messages,
                           clients=clients, ledger_path=ledger_path)


def _stub_gsm8k_leg(monkeypatch, calls, *, n_items: int = 3):
    """The Inspect leg, faked at the SEAM the driver hands it: for `n_items` GSM8K draws it does what the real
    provider's wire hook + the ledger do — one attempt = one set of bytes carrying NO chat_template_kwargs, one
    minted request id, one evidence record (through the sink, from the bytes) and one ledger row — and files the
    rows the producer files, through the producer's OWN emitter into the run's real CellStore with the arm and the
    row provenance the driver handed it (`thinking`, `row_identity`), plus the two per-task aggregate rows."""
    from sisteer.capeval.tiers import inspect_served, tier0

    async def fake_run(**kw):
        calls.append(kw)
        sink = kw.get("request_evidence_sink")
        out_root = Path(kw["store"].cell_dir).parent.parent
        ledger_path = out_root / tc.LEDGER_FILENAME
        em = tier0._Emitter(kw["store"], dict(kw["template"]), dict(tc.TINYCAP_DOMAIN_OF), tc.INSPECT_TASK_VERSION)
        prov = dict(kw.get("row_identity") or {})
        for i in range(n_items):
            em.emit(task=TASK, item_id=f"gsm8k_{i:08x}", metric=tc.METRIC_ACC, value=1.0, tier=tc.ROW_TIER, position_order=i,
                    extra={"thinking": kw["thinking"], "max_tokens": CAP, **prov})
        em.emit(task=TASK, item_id=tc.AGG_ITEM_ID, metric=tc.METRIC_ACC_AGGREGATE, value=1.0, tier=tc.ROW_TIER,
                extra={"n": n_items, "thinking": kw["thinking"]})
        em.emit(task=TASK, item_id=tc.AGG_ITEM_ID, metric=tc.METRIC_N, value=float(n_items), tier=tc.ROW_TIER)
        if sink is not None:
            for i in range(n_items):
                body = {"model": kw["route"].virtual_model, "messages": [{"role": "user", "content": f"question {i}"}],
                        "max_tokens": CAP, "temperature": 0.6, "top_p": 0.9, "seed": i}
                raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
                headers = {"Content-Type": "application/json", tc.HEADER_TASK: TASK,
                           tc.HEADER_SAMPLE: f"gsm8k_{i:08x}/0", tc.HEADER_REQUEST_ID: uuid.uuid4().hex}
                sink(raw, headers=headers)
                write_ledger_row(ledger_path, raw, headers)
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    return calls


def _read(p: Path):
    return json.loads(Path(p).read_text())


def _jsonl(p: Path):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def test_the_no_switch_class_runs_end_to_end_with_the_experiments_record(world, monkeypatch):  # noqa: F811
    w = no_switch_world(world, monkeypatch)
    calls = _stub_gsm8k_leg(monkeypatch, [])
    srun.main(w.argv)
    cd = _cell_dir(world.out_root)
    manifest, evidence = _read(cd / "manifest.json"), _read(cd / "evidence.json")
    # --- the hashed identity: the arm on both keys; the setter's cap; NO OOS key, NO header ids (the "present iff ON" rule)
    ident = manifest[tc.MANIFEST_KEY_TINYCAP]
    assert ident["thinking"]["mode"] == NA and ident[tc.IDENTITY_KEY_THINKING_BY_TASK] == {TASK: NA}
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {TASK: CAP} and ident[tc.IDENTITY_KEY_IDENTITY_VERSION] == 2
    for absent in (tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, tc.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT, tc.IDENTITY_KEY_THINKING_HEADER_IDS):
        assert absent not in ident, absent
    assert ident["temperature"] == 0.6 and ident["top_p"] == 0.9                      # the `card` row (P6)
    # --- the manifest's unhashed record: enable_thinking NULL (not sent), in support, undeclared, the bank view
    assert manifest[tc.MANIFEST_KEY_THINKING]["enable_thinking"] is None and manifest[tc.MANIFEST_KEY_THINKING]["mode"] == NA
    assert manifest[tc.MANIFEST_KEY_THINKING]["by_task"] == {TASK: NA} and manifest[tc.MANIFEST_KEY_THINKING]["reasoning_parser"] is None
    assert manifest["support"] == tc.SUPPORT_IN_SUPPORT and manifest["header_located"] is False
    assert manifest[tc.LENS_SERVED_POSITIONS_KEY] == ["all"] and manifest[tc.MANIFEST_KEY_AUTHORISED_EXPERIMENT] is None
    assert manifest[tc.MANIFEST_KEY_ROUTE]["bank_view"] == "main"
    assert manifest[tc.MANIFEST_KEY_SAMPLING]["mode"] == tc.SAMPLING_CARD and manifest[tc.MANIFEST_KEY_SAMPLING]["model_key"] == "llama8b"
    assert manifest[tc.MANIFEST_KEY_RENDER_FACT_EVIDENCE]["thinking_switch"] is False
    assert manifest[tc.MANIFEST_KEY_TRANSFER_PROVENANCE]["served_render"] == tc.RENDER_REGIME_UNSPECIFIED
    assert manifest[tc.MANIFEST_KEY_TRANSFER_PROVENANCE]["served_header_ids"] == {NA: LLAMA_HEADER}
    # --- (A''): the header measured ONCE via the served /v1/tokenize with NO kwargs; the one render probe `unspecified`
    hdr = evidence["thinking_header_by_arm"]
    assert hdr["cert_class"] == tc.CERT_CLASS_NO_SWITCH and hdr["measured_arms"] == [NA] and hdr["source"] == "served_tokenize"
    assert hdr["header_ids_by_arm"] == {NA: LLAMA_HEADER} and hdr["invariant"] is True and hdr["header_moves"] is False and hdr["verified"]
    assert hdr["construction"] == srun.HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF and hdr["agp_invariant"] is False
    assert hdr["render_len_by_agp"] == {"true": 3 + len(LLAMA_HEADER), "false": 3}
    assert all(tc.CHAT_TEMPLATE_KWARGS_KEY not in body for body in w.tokenized) and len(w.tokenized) >= 2
    rc = evidence["render_certified"]
    assert rc["render_certified"] is True and rc["cert_class"] == tc.CERT_CLASS_NO_SWITCH and rc["measured_arms"] == [NA]
    assert rc["measured_header_ids_by_arm"] == {NA: LLAMA_HEADER} and rc["expected_header_ids_by_arm"] == {NA: LLAMA_HEADER}
    assert rc[tc.OOS_EVIDENCE_HEADER_CERTIFIED] and rc["render_regime_certified"] and rc["think_shape_certified"]
    assert rc["split_mode"] == tc.THINK_SPLIT_NONE and rc["certification_problems"] == []
    assert set(evidence["render_probe"]) == {NA, "on_attempts"} and evidence["render_probe"][NA]["render_regime"] == tc.RENDER_REGIME_UNSPECIFIED
    # --- (B') verbatim: positions all, the digest == the pin, policy always, the class's stamp triple
    lp = evidence["lens_position_attested"]
    assert lp["served_positions"] == ["all"] and lp["header_located"] is False and lp["own_key"] == BARE
    assert lp["object_digest"]["scalar"] == OBJECT_SHA and lp["object_digest"]["by_cell"] == {BARE: OBJECT_SHA}
    assert lp["attested_policy"] == "always" and lp["expected_stamps"] == list(tc.G8K_STAMPS_FITTED_NOT_AVAILABLE)
    assert lp["expected_stamps"] == ["cell-stable", "always", tc.RENDER_REGIME_UNSPECIFIED]
    # --- (C''): the EXPLICIT inapplicability record; the ordinary bites control under the single render
    assert evidence[tc.LENS_ON_BITE_KEY] == tc.no_switch_lens_on_bite_record()
    bites = evidence["steering_bites"]
    assert bites["applicable"] and bites["verified"] and bites["twin"] == tc.TWIN_CELL_ID and bites["n_differing"] >= 1
    assert bites[tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == NA and bites["probe_extra_body"] == {}
    for leg, stamps in bites["stamps_by_leg"].items():
        assert stamps and all(s[tc.STAMP_KEY_RENDER_REGIME] == tc.RENDER_REGIME_UNSPECIFIED for s in stamps), leg
    # --- the wire: NO body carried a switch or a declaration; the exact class inventory of probes
    assert w.chat.calls and all(tc.CHAT_TEMPLATE_KWARGS_KEY not in c and tc.OOS_THINKING_DECLARATION_KEY not in c for c in w.chat.calls)
    recs = _jsonl(world.out_root / tc.REQUEST_EVIDENCE_FILENAME)
    assert recs and all(r["enable_thinking_present"] is False and r["enable_thinking_value"] is None for r in recs)
    assert all(r["petri_thinking_on_present"] is False for r in recs)
    probes = [r for r in recs if r["kind"] == tc.REQUEST_EVIDENCE_KIND_PROBE]
    inventory = {}
    for r in probes:
        kind = str(r["task"])[len(tc.PROBE_TASK_PREFIX):]
        inventory[(r["vmodel"], NA, kind)] = inventory.get((r["vmodel"], NA, kind), 0) + 1
    assert inventory == tc.no_switch_probe_inventory(VMODEL, fitted=True, n_probes=bites["n_probes"])
    assert not any(k[2] == tc.PROBE_KIND_LENS_ON_BITE for k in inventory)
    # --- the ledger: `thinking` NULL on every row (P17: no switch sent), the cap on every task row
    rows = _jsonl(w.ledger_path)
    assert rows and all(r["thinking"] is None for r in rows)
    assert all(r["max_tokens_requested"] == CAP for r in rows if r["task"] == TASK)
    assert evidence["request_evidence_gate"]["problem"] is None
    # --- the leg was handed the arm, the setter, the family, NO override; every emitted row carries the arm + provenance
    kw = calls[0]
    assert kw["thinking"] == NA and kw["family"] == "llama" and kw["out_of_support"] is False and kw["off_cap_override"] is False
    assert kw["on_cap_override"] == {TASK_NAME: CAP} and kw["oos_experiment"] is None and kw["reasoning_parser"] is None
    # the row identity (main's `row_identity_block` since the IFEval battery landed: our four provenance facts + the sampling mode)
    prov = kw["row_identity"]
    assert {k: prov[k] for k in tc.ROW_PROVENANCE_FIELDS} == {"object_artifact_sha256": OBJECT_SHA, "route_kind": tc.ROUTE_KIND_LENS, "view": "main",
                                                            "producer_tree_sha": manifest[tc.MANIFEST_KEY_PRODUCER]["tree_sha"]}
    assert prov[tc.ROW_SAMPLING_MODE_FIELD] == kw["sampling"].mode if hasattr(kw.get("sampling"), "mode") else True
    assert prov["object_artifact_sha256"] == manifest[tc.MANIFEST_KEY_PROVENANCE]["artifact_sha256"]
    store_rows = _jsonl(cd / "rows.jsonl")
    item_rows = [r for r in store_rows if r["item_id"] != tc.AGG_ITEM_ID]
    assert len(item_rows) == 3
    for r in item_rows:
        extra = json.loads(r["extra_json"])
        assert extra[tc.ROW_KEY_THINKING] == NA
        assert {f: extra[f] for f in tc.ROW_PROVENANCE_FIELDS} == {f: prov[f] for f in tc.ROW_PROVENANCE_FIELDS}
    # the live prompt-budget pass ran under the no-switch render
    assert evidence[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK]["task"] == TASK


def test_without_bank_view_the_rows_are_byte_identical_to_before(world, monkeypatch):  # noqa: F811
    w = no_switch_world(world, monkeypatch, bank_view=None)
    calls = _stub_gsm8k_leg(monkeypatch, [])
    srun.main(w.argv)
    manifest = _read(_cell_dir(world.out_root) / "manifest.json")
    assert "bank_view" not in manifest[tc.MANIFEST_KEY_ROUTE]
    assert calls[0]["row_identity"] is None
    for r in _jsonl(_cell_dir(world.out_root) / "rows.jsonl"):
        assert not set(tc.ROW_PROVENANCE_FIELDS) & set(json.loads(r["extra_json"] or "{}"))


def test_not_available_on_a_switched_family_refuses_by_name_before_any_request(world, monkeypatch):  # noqa: F811
    w = no_switch_world(world, monkeypatch)
    # the world's fixture model re-pinned under a SWITCHED family's key: the family gate fires first, by name
    monkeypatch.setitem(_geom.MODEL_HF_IDS, "gemma4e4b", BASE_MODEL)
    monkeypatch.delitem(_geom.MODEL_HF_IDS, "llama8b")
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    assert "has a thinking switch" in str(e.value) and "gemma4" in str(e.value)
    assert not w.chat.calls and not w.tokenized


def test_a_candidate_no_switch_fact_refuses_production_by_name_and_diagnostic_admits_it(world, monkeypatch):  # noqa: F811
    w = no_switch_world(world, monkeypatch, certified=False)
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    assert tc.REC_THINKING_UNSERVABLE in str(e.value) and "render fact not certified" in str(e.value)
    assert not w.chat.calls
    # the diagnostic smoke itself (--diagnostic, identity v2) admits the candidate fact and runs
    w2 = no_switch_world(world, monkeypatch, certified=False, extra_argv=[tc.FLAG_DIAGNOSTIC])
    calls = _stub_gsm8k_leg(monkeypatch, [])
    srun.main(w2.argv)
    manifest = _read(_cell_dir(world.out_root) / "manifest.json")
    assert manifest[tc.MANIFEST_KEY_TINYCAP][tc.IDENTITY_KEY_DIAGNOSTIC] is True and calls[0]["thinking"] == NA


def test_a_header_located_cell_under_the_no_switch_arm_refuses_by_name(world, monkeypatch):  # noqa: F811
    """The class certifies all-position objects only: a `response`-served cell has no declared render on a
    family with no switch — refused by name after the evidence is written, like every predicate."""
    w = no_switch_world(world, monkeypatch)
    write_standup_endpoint(
        world.table_path,
        **{tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {}, tc.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
           tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {BARE: "always", tc.TWIN_CELL_ID: "always"},
           tc.ENDPOINT_KEY_LENS_SALT_POLICY: tc.LENS_SALT_POLICY_ATTESTED,
           tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {BARE: OBJECT_SHA},
           "geometry": standup_geometry(G8K_WINDOW, G8K_S), "max_model_len": G8K_WINDOW, "max_num_seqs": G8K_S,
           tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: tc.PROXY_UPSTREAM_TIMEOUT_S,
           tc.LENS_SERVED_POSITIONS_KEY: {BARE: ["response"]}})
    _stub_gsm8k_leg(monkeypatch, [])
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    assert "HEADER-LOCATED" in str(e.value) and "all-position objects only" in str(e.value)


# --------------------------------------------------------------------------- the header of a template that never consults the flag

def test_the_no_switch_class_reads_an_unconditional_header_by_content_substitution_and_certifies(world, monkeypatch):  # noqa: F811
    """Smoke 490518 (2026-09-12): the served NousResearch Llama-3.1 template appends the assistant header UNCONDITIONALLY, so the
    generation-prompt diff — the construction the class used — is EMPTY where the header is always present (chat_markers
    calls that diff the FORBIDDEN comparison), and the wave would have refused every row. The class now reads such a template
    by chat_markers' content-substitution rule, served: the generation prompt ends EXACTLY where a completed turn's content
    begins and ENDS with the closing scaffold's trailing header; the fact is certified by tuple as before."""
    from sisteer import chat_markers as cm
    w = no_switch_world(world, monkeypatch, tok=UnconditionalHeaderTok())
    calls = _stub_gsm8k_leg(monkeypatch, [])
    srun.main(w.argv)
    cd = _cell_dir(world.out_root)
    manifest, evidence = _read(cd / "manifest.json"), _read(cd / "evidence.json")
    hdr = evidence["thinking_header_by_arm"]
    assert hdr["header_ids_by_arm"] == {NA: LLAMA_HEADER} and hdr["verified"] and hdr["source"] == srun.HEADER_SOURCE_SERVED
    assert hdr["construction"] == srun.HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING and hdr["agp_invariant"] is True
    assert hdr["render_len_by_agp"] == {"true": 3 + len(LLAMA_HEADER), "false": 3 + len(LLAMA_HEADER)}
    assert hdr["content_start"] == 3 + len(LLAMA_HEADER) and hdr["closing_scaffold"] == [EOT] + LLAMA_HEADER
    assert hdr["cert_class"] == tc.CERT_CLASS_NO_SWITCH and hdr["measured_arms"] == [NA] and hdr["invariant"] and not hdr["header_moves"]
    rc = evidence["render_certified"]
    assert rc["render_certified"] is True and rc["measured_header_ids_by_arm"] == {NA: LLAMA_HEADER} and rc[tc.OOS_EVIDENCE_HEADER_CERTIFIED]
    assert manifest[tc.MANIFEST_KEY_TRANSFER_PROVENANCE]["served_header_ids"] == {NA: LLAMA_HEADER}
    # the served measurement: the two flag renders, then the two sentinel renders — NO chat_template_kwargs on any of them
    probes = [b for b in w.tokenized if b["messages"][0] == dict(srun.HEADER_PROBE_MESSAGES[0])]
    assert len(probes) >= 4 and all(tc.CHAT_TEMPLATE_KWARGS_KEY not in b for b in w.tokenized)
    assert [b["add_generation_prompt"] for b in probes[:2]] == [True, False]
    sentinels = [b["messages"][1]["content"] for b in probes[2:4]]
    assert sentinels == [cm._SENTINEL_A, cm._SENTINEL_B] and all(b["add_generation_prompt"] is False for b in probes[2:4])
    assert calls and calls[0]["thinking"] == NA


def test_a_template_that_ignores_the_flag_without_an_unconditional_header_refuses_by_name(world, monkeypatch):  # noqa: F811
    w = no_switch_world(world, monkeypatch, tok=FlagIgnoringNoHeaderTok())
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    msg = str(e.value)
    assert tc.REC_THINKING_UNSERVABLE in msg and "does not end where a completed assistant turn's content begins" in msg
    assert not w.chat.calls


def test_the_smoke_probes_header_construction_is_the_drivers_on_every_template_shape(monkeypatch):
    """The GSM8K no-switch smoke probe (scripts/sdfbase/gsm8k/smoke_not_available_probe.py) MIRRORS the driver's served
    construction — one rule, two entry points, pinned equal on the same fakes: a gated template (the diff), the
    unconditional-header template (content substitution), a flag-ignoring template without the header (both refuse, the
    same reason)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g8k_smoke_probe_equiv", REPO / "scripts/sdfbase/gsm8k/smoke_not_available_probe.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    assert (P.HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF, P.HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING) == (
        srun.HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF, srun.HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING)
    # the probe conversation differs in wording only (the header does not depend on the user content)
    assert [m["role"] for m in P.HEADER_PROBE_MESSAGES] == [m["role"] for m in srun.HEADER_PROBE_MESSAGES] == ["user"]
    for tok, construction, agp_inv in ((LlamaHeaderTok(), srun.HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF, False),
                                       (UnconditionalHeaderTok(), srun.HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING, True)):
        monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev, _t=tok: _t)
        d_ids, d_facts = srun._served_no_switch_header(live_tokenize, "petri__base")
        p_ids, p_facts = P.served_no_switch_header(live_tokenize, "petri__base")
        assert list(d_ids) == list(p_ids) == LLAMA_HEADER, construction
        assert d_facts["construction"] == p_facts["construction"] == construction and d_facts["agp_invariant"] is p_facts["agp_invariant"] is agp_inv
        assert {k: v for k, v in d_facts.items() if k != "closing_scaffold"} == {k: v for k, v in p_facts.items() if k != "closing_scaffold"}
        assert list(d_facts.get("closing_scaffold") or []) == list(p_facts.get("closing_scaffold") or [])
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: FlagIgnoringNoHeaderTok())
    with pytest.raises(srun.HeaderMeasurementError, match="does not end where a completed assistant turn's content begins"):
        srun._served_no_switch_header(live_tokenize, "petri__base")
    with pytest.raises(RuntimeError, match="does not end where a completed assistant turn's content begins"):
        P.served_no_switch_header(live_tokenize, "petri__base")
