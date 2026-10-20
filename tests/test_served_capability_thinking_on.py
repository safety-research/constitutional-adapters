"""math500-thinking-on-full22, lane A (a3 / a4) — the DRIVER under thinking ON on
an UNDECLARED all-position object cell, end to end through
``served_capability_run.main`` on the synthetic `world` of
``test_served_capability.py`` with the standup record re-spelled the way Evals'
refinement exports it: ``lens_served_positions[<bare key>] == ["all"]``, an EMPTY
declarations map, the per-cell policy ``always``, the digest under the BARE key.

Pinned here: the plain route is selected (no ``@thinking``); NO request carries
``petri_thinking_on`` (probes, render probes, the lens-ON bite; the task calls
through the provider's sink); the bites probe renders thinking-OFF;
predicate (A) certifies the live render against the family fact, (B′) the
served-positions attestation, (C) the lens-ON bite; the version-2 identity
(`identity_version`, `max_tokens_by_task` keyed by the full row task, the
measured `thinking_header_ids`, `out_of_support_thinking` true and NO
`serve_out_of_support` key), the transfer-provenance block, `oos_experiment`,
`resume_attempts`, `request_evidence.jsonl`; every hole refused BY NAME; the
Nemotron state (a PARTIAL lens fact on the proxy) admits the undeclared object
cell; ``--ledger-url`` is required for MATH-500; a single-arm ON run on a
moves-TRUE lens cell never raises ``MixedArmHeaderMoved`` (G3).

Everything here is SYNTHETIC (a `TESTHOST` standup, placeholder digests).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_served_capability import (  # noqa: F401  (`world` is a fixture)
    BASE_MODEL,
    OBJECT_SHA,
    OOS_SEQS,
    OOS_WINDOW,
    REV,
    TEMPLATE_HASH,
    TT_CELL,
    HeaderTok,
    QwenHeaderTok,
    REPO,
    _cell_dir,
    _never_runs,
    _percell_argv,
    _pin_the_authorised_oos_model,
    live_tokenize,
    srun,
    standup_geometry,
    stub_framework_versions,
    world,
    write_standup_endpoint,
)

from sisteer.capeval.serving import proxy as px
from sisteer.capeval.tinycap import contracts as tc
from sisteer.capeval.serving import geometry as _geom

BARE = "si_safety__psr_kl_all"                  # the world's lens cell, BARE harness key
VMODEL = "petri__" + BARE
#: a SYNTHETIC registry tuple naming the world's cell (the experiment's real tuples name the
#: `constv2` object cell; the world serves the parent's fixture cell) — the machinery under
#: test is the driver's, the eight real tuples are pinned in test_tinycap_render_facts.py
SYNTH_NAME = "synthetic_m5t_world_qwen35_4b"

#: the m5t world runs MATH-500 at the harness default cap under MATH's OWN served window:
#: both derive from the contracts/geometry tables, never a literal, so a default change moves the world with it
M5T_CAP = tc.MATH500_MAX_TOKENS_DEFAULT
M5T_WINDOW = _geom.geometry_for("capability:gen_long_math").max_model_len


def _synthetic_tuple():
    exp = tc.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_qwen35_4b"]
    return exp._replace(cells=(TT_CELL,), reduction_mode=tc.REDUCTION_MODE_PER_ROW,
                        what=exp.what + " [SYNTHETIC: the test world's cell and per_row standup]")


class M5TChat:
    """A scripted proxy for the undeclared all-position world: every reply is
    stamped (lens legs `(cell-stable, always, <render>)`, the base route
    `(cell-stable, not-consulted, <render>)`); under thinking-ON the lens reply
    carries the family's think shape (`…</think>` then the answer, Qwen's
    implicit-open render) and DIFFERS from the base route's; a declared body
    on ANY lens route refuses `thinking_on_cell_not_declared` (rule 3: the
    all-position cell cannot be declared) and on the base route
    `thinking_on_not_lens`; an undeclared ON request on the lens route is
    ACCEPTED (Evals' refinement, the served-positions fact names the cell)."""

    def __init__(self, *, lens_models, base_models, lens_on_text="deep thoughts</think>\nANSWER: 7",
                 base_on_text="other thoughts</think>\nANSWER: 8", stamp_render=None, bite_lens_text=None):
        self.lens_models, self.base_models = set(lens_models), set(base_models)
        self.lens_on_text, self.base_on_text = lens_on_text, base_on_text
        #: the lens reply to the lens-ON BITE's prompt (predicate (C)) when it should differ from
        #: the render probe's (predicate (A) runs first, on the same route)
        self.bite_lens_text = bite_lens_text
        self.stamp_render = stamp_render
        self.calls: list = []

    def _regime(self, payload):
        if self.stamp_render is not None:
            return self.stamp_render
        kwargs = payload.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {}
        flag = kwargs.get(tc.ENABLE_THINKING_KEY)
        return (tc.RENDER_REGIME_THINKING_ON if flag is True else tc.RENDER_REGIME_THINKING_OFF if flag is False
                else tc.RENDER_REGIME_UNSPECIFIED)

    def chat_completions(self, payload):
        self.calls.append(json.loads(json.dumps(payload)))
        lens = payload["model"] in self.lens_models
        if tc.OOS_THINKING_DECLARATION_KEY in payload:
            code = px.THINKING_ON_CELL_NOT_DECLARED if lens else px.THINKING_ON_NOT_LENS
            raise px.ProxyError("proxy /v1/chat/completions returned 400: " + json.dumps({"error": {"code": code}}))
        on = (payload.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {}).get(tc.ENABLE_THINKING_KEY) is True
        if lens:
            text = self.lens_on_text if on else ("steered voice says alpha" if payload["model"] == VMODEL
                                                 else "plain voice says beta")
            if on and self.bite_lens_text is not None and "sky" in payload["messages"][-1]["content"]:
                text = self.bite_lens_text
            policy = "always"
        else:
            text = self.base_on_text if on else "base voice says gamma"
            policy = tc.POLICY_NOT_CONSULTED
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_SALT_REGIME: "cell-stable",
                                      tc.STAMP_KEY_RESUMED_POLICY: policy,
                                      tc.STAMP_KEY_RENDER_REGIME: self._regime(payload)}}


def _math_budget(tmp_path, tok, messages, *, cap=M5T_CAP):
    lengths = [len(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True, enable_thinking=True))
               for m in messages]
    doc = {"n": len(lengths), "max": max(lengths), "p99": tc.nearest_rank_percentile(lengths, 0.99),
           "p50": tc.nearest_rank_percentile(lengths, 0.50), "mean": round(sum(lengths) / len(lengths), 2),
           "arm": tc.THINKING_ON, "model": BASE_MODEL, "model_slug": "qwen35_4b", "revision": REV,
           "chat_template_hash": TEMPLATE_HASH, "cap": int(cap), "cap_task": tc.inspect_task("math"),
           "max_plus_cap": max(lengths) + int(cap), "lengths": lengths,
           "render": {"add_generation_prompt": True, **tc.chat_template_kwargs(tc.THINKING_ON)},
           "messages_source": "sisteer.capeval.tiers.inspect_served.math500_messages", "tree_sha": None,
           "produced_by": {"script": tc.OOS_PROMPT_BUDGET_SCRIPT, "tokenizer": "QwenHeaderTok"}}
    tc.validate_oos_prompt_budget(doc)
    path = tmp_path / tc.OOS_PROMPT_BUDGET_FILENAME
    path.write_text(json.dumps(doc))
    return path


def _math_messages(n: int = 500):
    return [[{"role": "user", "content": f"Problem {i}: compute " + "x" * (i % 41) + "\nANSWER: $ANSWER"}]
            for i in range(n)]


def undeclared_world(world, monkeypatch, *, served_positions=("all",), endpoint_extra=None, identity_extra=None,  # noqa: F811
                     chat=None, with_digest=True, with_ledger=True, extra_argv=(), drop_keys=()):
    """The UNDECLARED all-position object-cell world: the
    standup record as Evals' exporter + the cluster standup write it — the
    served-positions map under the BARE key, an empty declarations map, the
    per-cell policy `always`, the digest under the bare key, a 10,752 x S
    window; the proxy identity echoes the map and a PARTIAL lens fact (the
    Nemotron state: `on_ids: null` + a reason — NEVER consulted for an
    undeclared cell); the served tokenizer renders Qwen3.5's real ids; MATH's
    500 exact messages and their CPU budget are stubbed; the argv is the
    experiment's driver invocation for a fitted job."""
    from sisteer.capeval.tiers import inspect_served as isv

    stub_framework_versions(monkeypatch)
    _pin_the_authorised_oos_model(monkeypatch)
    monkeypatch.setattr(tc, "AUTHORISED_OOS_EXPERIMENTS", {**tc.AUTHORISED_OOS_EXPERIMENTS, SYNTH_NAME: _synthetic_tuple()})
    # the world reuses the PARENT's psr fixture cell on the parent's authorised
    # model, which would trip the parent's STRICT `conditional` stamp rows; the
    # experiment's cells are not the parent's cell, so the strict-row predicate is
    # patched out here (decisions-laneA LA18) — every other compare is the real one
    monkeypatch.setattr(tc, "is_authorised_psr_leg", lambda cell, model_key: False)
    # the world's fixture model id is a Llama-named placeholder pinned under the
    # qwen35_4b key; the driver's family cross-check (served config vs model key)
    # is told the served family the tokenizer stub renders — Qwen3.5's ids
    from sisteer.capeval.serving import vllm_server as _vllm_server
    monkeypatch.setattr(_vllm_server, "model_family", lambda mid, rev=None: "qwen3_5")
    tok = QwenHeaderTok()
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: tok)
    messages = _math_messages()
    monkeypatch.setattr(isv, "task_messages", lambda name, repo_root=None, system=None: [list(m) for m in messages])
    budget = _math_budget(world.tmp_path, QwenHeaderTok(), messages)
    positions = {BARE: list(served_positions)} if served_positions is not None else None
    ep_extra = {tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {}, tc.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
                tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {BARE: "always", tc.TWIN_CELL_ID: "always"},
                tc.ENDPOINT_KEY_LENS_SALT_POLICY: tc.LENS_SALT_POLICY_ATTESTED,
                tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {BARE: OBJECT_SHA},
                "geometry": standup_geometry(M5T_WINDOW, OOS_SEQS),
                "max_model_len": M5T_WINDOW, "max_num_seqs": OOS_SEQS,
                tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: tc.PROXY_UPSTREAM_TIMEOUT_S,
                **({tc.LENS_SERVED_POSITIONS_KEY: positions} if positions is not None else {}),
                **(endpoint_extra or {})}
    for k in drop_keys:
        ep_extra.pop(k, None)
    write_standup_endpoint(world.table_path, **ep_extra)
    from test_served_capability import LIVE_IDENTITY
    identity = {**LIVE_IDENTITY, tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {},
                # the NEMOTRON state on the proxy: a partial fact + a reason — recorded, never asserted here
                "thinking_header": {"switch": True, "moves": None, "off_ids": list(tc.QWEN35_HEADER_IDS_OFF),
                                    "on_ids": None},
                "thinking_header_error": "probe: the add_generation_prompt render is not a prefix (synthetic)",
                **({tc.LENS_SERVED_POSITIONS_KEY: positions} if positions is not None else {}),
                **(identity_extra or {})}
    chat = chat or M5TChat(lens_models={VMODEL, world.twin.virtual_model}, base_models={"petri__base"})
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
    argv = (_percell_argv(world, tc.CONTROL_MATH500)
            + [tc.FLAG_THINKING, tc.THINKING_ON, tc.FLAG_SAMPLING, tc.SAMPLING_THINKING_CARD, tc.FLAG_SAMPLES, "1",
               tc.FLAG_MATH500_RETRY_MAX_TOKENS, "0", tc.FLAG_MAX_TOKENS_BY_TASK, f"math={M5T_CAP}",
               tc.FLAG_TINYCAP_IDENTITY_VERSION, "2", tc.FLAG_OOS_PROMPT_BUDGET, str(budget),
               tc.FLAG_ALLOW_OOS_THINKING])
    if with_ledger:
        from test_served_capability import LEDGER_URL
        argv += [tc.FLAG_LEDGER_URL, LEDGER_URL]
    if with_digest:
        argv += [tc.FLAG_EXPECT_OBJECT_SHA256, OBJECT_SHA]
    argv += list(extra_argv)
    return SimpleNamespace(argv=argv, chat=chat, tokenized=tokenized, budget=budget, identity=identity,
                           messages=messages, clients=clients)


class BoundaryClient:
    """The world's stand-in for `ProxyClientEngine` AND the usage ledger in front
    of the proxy: `chat_completions(payload, headers=)` serialises the payload
    exactly as the real client does, hands the bytes + headers to the boundary
    hook the driver attaches (`on_request`), writes the ledger row the real
    ledger would (request_id from the header, body_sha256 of the same bytes,
    task / sample from the labels), then answers from the scripted proxy. So the
    driver's evidence records and the ledger rows come from ONE set of bytes, as
    on the wire."""

    def __init__(self, chat, *, tokenize, identity, ledger_path):
        self._chat, self.tokenize, self.proxy_identity = chat, tokenize, identity
        self.ledger_path = ledger_path
        self.on_request = None
        self.sent: list = []          # (bytes, headers) of every chat POST

    def chat_completions(self, payload, *, headers=None):
        data = json.dumps(dict(payload)).encode("utf-8")
        hdrs = {"Authorization": "Bearer petri-proxy", "Content-Type": "application/json", **(headers or {})}
        if self.on_request is not None:
            self.on_request("http://ledger/v1/chat/completions", data, dict(hdrs))
        self.sent.append((data, dict(hdrs)))
        write_ledger_row(self.ledger_path, data, hdrs)
        return self._chat.chat_completions(payload)


def write_ledger_row(path, data: bytes, headers, *, status=200):
    """One usage-ledger row for one attempt, as `UsageLedger.handle_chat` writes it."""
    from sisteer.capeval.serving import usage_ledger as ul
    body = json.loads(data)
    row = dict.fromkeys(tc.LEDGER_ROW_FIELDS)
    row.update(ts="2026-01-01T00:00:00Z", task=headers.get(tc.HEADER_TASK), sample=headers.get(tc.HEADER_SAMPLE),
               has_tools=False, n_tools=0, rejected=False, status=status, error_code=None,
               prompt_tokens_pre=10, usage_prompt_tokens=10, usage_completion_tokens=5, finish_reason="stop",
               latency_s=0.1, tokenize_mismatch=False, error_class=tc.LEDGER_ERROR_CLASS_NONE,
               request_id=headers.get(tc.HEADER_REQUEST_ID), body_sha256=tc.wire_body_sha256(data))
    row.update(ul.body_facts(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(ul.ledger_row(**row)) + "\n")


def _stub_math_leg(monkeypatch, calls, *, tamper=None, n_items: int = 3):
    """The Inspect leg, faked: for each of `n_items` MATH samples it does what the
    real provider's wire hook + the ledger do — one attempt = one set of bytes,
    one request id, one evidence record (through the sink, from the bytes) and
    one ledger row — and files the rows the producer files, through the
    producer's OWN emitter into the run's real `CellStore`: one `acc` row per
    item and the two per-task aggregate rows (`acc_aggregate`, `n`) under
    `AGG_ITEM_ID`, which correspond to no request (fixA3: a stub that wrote no
    rows left the post-run gate's inventory empty, so the gate never saw the
    aggregate rows the real leg always files); `tamper(evidence_path,
    ledger_path)` then edits the files before the driver's post-run gate reads
    them (the negative tests)."""
    from sisteer.capeval.tiers import inspect_served, tier0

    async def fake_run(**kw):
        calls.append(kw)
        sink = kw.get("request_evidence_sink")
        out_root = Path(kw["store"].cell_dir).parent.parent
        ledger_path = out_root / tc.LEDGER_FILENAME
        task = tc.inspect_task("math")
        em = tier0._Emitter(kw["store"], dict(kw["template"]), dict(tc.TINYCAP_DOMAIN_OF), tc.INSPECT_TASK_VERSION)
        for i in range(n_items):
            em.emit(task=task, item_id=f"math_{i:08x}", metric=tc.METRIC_ACC, value=1.0, tier=tc.ROW_TIER,
                    position_order=i, extra={"thinking": tc.THINKING_ON, "max_tokens": M5T_CAP})
        em.emit(task=task, item_id=tc.AGG_ITEM_ID, metric=tc.METRIC_ACC_AGGREGATE, value=1.0, tier=tc.ROW_TIER,
                extra={"n": n_items, "thinking": tc.enable_thinking(tc.THINKING_ON)})
        em.emit(task=task, item_id=tc.AGG_ITEM_ID, metric=tc.METRIC_N, value=float(n_items), tier=tc.ROW_TIER)
        if sink is not None:
            for i in range(n_items):
                body = {"model": kw["route"].virtual_model, "messages": [{"role": "user", "content": f"problem {i}"}],
                        "max_tokens": M5T_CAP, "temperature": 0.6, "top_p": 0.95, "seed": i,
                        "extra_body": None, **tc.chat_template_kwargs(tc.THINKING_ON)}
                body.pop("extra_body")
                raw = json.dumps(body, separators=(",", ":")).encode("utf-8")     # the SDK's compact JSON
                headers = {"Content-Type": "application/json", tc.HEADER_TASK: tc.inspect_task("math"),
                           tc.HEADER_SAMPLE: f"math_{i:08x}/0", tc.HEADER_REQUEST_ID: uuid.uuid4().hex}
                sink(raw, headers=headers)
                write_ledger_row(ledger_path, raw, headers)
            if tamper is not None:
                tamper(out_root / tc.REQUEST_EVIDENCE_FILENAME, ledger_path)
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    return calls


def _drop_lines(path: Path, keep) -> None:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    kept = [l for l in lines if keep(json.loads(l))]
    assert len(kept) < len(lines), "the tamper removed nothing"
    path.write_text("".join(l + "\n" for l in kept))


def test_the_request_evidence_is_wire_bound_and_reconciles_with_the_ledger_by_attempt(world, monkeypatch):  # noqa: F811
    """FC4 (1)+(2): every attempt — the three task calls, the
    two render probes, the bites probes on both legs, the lens-ON bite on both
    routes — has exactly ONE version-2 evidence record written from the bytes at
    the client's boundary and ONE ledger row with the same request_id and the
    same body digest; the probes are labelled probe/<kind> with <vmodel>/<n>
    samples; the gate passes and records the inventory sizes; the attempt entry
    is completed once the run passes (FC4 (3))."""
    w = undeclared_world(world, monkeypatch)
    _stub_math_leg(monkeypatch, [])
    srun.main(w.argv)
    cell_dir = _cell_dir(world.out_root)
    evidence = json.loads((cell_dir / "evidence.json").read_text())
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    records = [json.loads(l) for l in (world.out_root / tc.REQUEST_EVIDENCE_FILENAME).read_text().splitlines()]
    rows = [json.loads(l) for l in (world.out_root / tc.LEDGER_FILENAME).read_text().splitlines()]
    assert records and all(r["request_evidence_version"] == tc.REQUEST_EVIDENCE_VERSION == 3 for r in records)
    assert all(tuple(sorted(r)) == tuple(sorted(tc.REQUEST_EVIDENCE_FIELDS)) for r in records)
    # identity: one record per row, same id, same wire digest, same labels
    assert sorted(r["request_id"] for r in records) == sorted(x["request_id"] for x in rows)
    by_id = {x["request_id"]: x for x in rows}
    for r in records:
        row = by_id[r["request_id"]]
        assert r["body_sha256"] == row["body_sha256"] and (r["task"], r["sample"]) == (row["task"], row["sample"])
    # the driver's client boundary produced the probe records from the exact bytes it sent
    client_sent = {tc.wire_body_sha256(d) for c in w.clients for d, _h in c.sent}
    probes = [r for r in records if r["kind"] == "probe"]
    assert {r["body_sha256"] for r in probes} == client_sent           # every probe record IS a body the client sent
    tasks = [r for r in records if r["kind"] == "task"]
    assert len(tasks) == 3 and all(r["task"] == tc.inspect_task("math") and r["max_tokens"] == M5T_CAP
                                  for r in tasks)
    kinds = {r["task"] for r in probes}
    assert kinds == {tc.probe_task_label(k) for k in (tc.PROBE_KIND_RENDER, tc.PROBE_KIND_BITES, tc.PROBE_KIND_LENS_ON_BITE)}
    render = sorted((r["sample"], r["enable_thinking_value"]) for r in probes if r["task"] == "probe/render")
    assert render == [(f"{VMODEL}/0", False), (f"{VMODEL}/1", True)]          # one per arm, on the cell's route
    bite = {r["sample"].rsplit("/", 1)[0] for r in probes if r["task"] == "probe/lens_on_bite"}
    assert bite == {VMODEL, "petri__base"}
    assert all(r["petri_thinking_on_present"] is False for r in records)
    assert all(r["body_bytes"] > 0 and len(r["body_sha256"]) == 64 for r in records)
    gate = evidence["request_evidence_gate"]
    assert gate["problem"] is None and gate["n_records"] == len(records) == gate["n_ledger_rows"] == len(rows)
    assert manifest[tc.MANIFEST_KEY_RESUME_ATTEMPTS][-1]["completed"] is True
    # fixA3: the store holds the producer's shape — the three item rows AND the two
    # per-task aggregate rows — and the gate's coverage inventory is the ITEMS: the
    # aggregate pseudo-item corresponds to no request and is not counted as filed
    filed_rows = [json.loads(l) for l in (cell_dir / "rows.jsonl").read_text().splitlines() if l.strip()]
    assert {(r["item_id"], r["metric"]) for r in filed_rows if tc.is_aggregate_item(r["item_id"])} == {
        (tc.AGG_ITEM_ID, tc.METRIC_ACC_AGGREGATE), (tc.AGG_ITEM_ID, tc.METRIC_N)}
    assert sorted(r["item_id"] for r in filed_rows if not tc.is_aggregate_item(r["item_id"])) == [
        f"math_{i:08x}" for i in range(3)]
    assert gate["n_filed_items"] == 3


@pytest.mark.parametrize("name,tamper,expect", [
    ("one_task_record_missing",
     lambda ev, led: _drop_lines(ev, lambda r: not (r["kind"] == "task" and r["sample"].endswith("00000001/0"))),
     "have NO request-evidence record"),
    ("all_task_records_missing",
     lambda ev, led: _drop_lines(ev, lambda r: r["kind"] != "task"),
     "have NO request-evidence record"),
    ("one_ledger_row_missing",
     lambda ev, led: _drop_lines(led, lambda x: not (x["task"] == "inspect/math" and x["sample"].endswith("00000002/0"))),
     "the ledger never received"),
    ("all_probe_records_missing",
     lambda ev, led: _drop_lines(ev, lambda r: r["kind"] != "probe"),
     "have NO request-evidence record"),
    ("digest_mismatch",
     lambda ev, led: ev.write_text("".join(
         json.dumps({**json.loads(l), "body_sha256": "00" * 32}) + "\n" if json.loads(l)["kind"] == "task" else l + "\n"
         for l in ev.read_text().splitlines() if l.strip())),
     "disagree on what was sent"),
    ("duplicate_record_without_attempt",
     lambda ev, led: ev.write_text(ev.read_text() + [l for l in ev.read_text().splitlines() if '"kind": "task"' in l][0] + "\n"),
     "appears on two request-evidence records"),
    ("ledger_without_identity",
     lambda ev, led: led.write_text("".join(json.dumps({**json.loads(l), "request_id": None, "body_sha256": None}) + "\n"
                                            for l in led.read_text().splitlines() if l.strip())),
     "cannot be reconciled"),
])
def test_missing_duplicate_or_mismatched_request_evidence_refuses_the_cell_by_name(world, monkeypatch, name, tamper, expect):  # noqa: F811
    """The gate reconciles the evidence inventory with the
    ledger's attempt inventory BY IDENTITY, so deleting one task record, every
    task record, one ledger row, every probe record, corrupting a digest,
    duplicating a record or a ledger that recorded no identity each refuse the
    cell by name — never read around."""
    w = undeclared_world(world, monkeypatch)
    _stub_math_leg(monkeypatch, [], tamper=tamper)
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    msg = str(e.value)
    assert tc.REC_OOS_DECLARATION_MISMATCH in msg and "request evidence" in msg and expect in msg, (name, msg[:400])
    gate = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())["request_evidence_gate"]
    assert gate["problem"] and expect in gate["problem"]


def test_the_coverage_inventory_is_the_stores_items_never_its_aggregate_rows(tmp_path, monkeypatch):
    """fixA3 (harness P1): the Inspect runner ALWAYS files two synthetic rows per
    task under `AGG_ITEM_ID` (`acc_aggregate`, `n`) beside the item rows, and they
    correspond to no HTTP request. Through the REAL runner, emitter and `CellStore`:
    a completed leg whose every real item has its evidence record and ledger row
    PASSES reconciliation under the driver's inventory rule (`filed_item_pairs`)
    with the aggregate rows present; the pre-fix inventory — every
    `completed_keys()` pair — names `__aggregate__` as a filed item with no
    attempt, the refusal that left a finished 500-item leg `completed: false`
    with nothing a resume could clear; a genuine item stripped of its evidence
    still refuses by name, at the identity step (its ledger attempt has no
    record) and at the coverage step (no attempt at all)."""
    from test_inspect_served import _fake_eval, _Log, _Sample, _Score

    from sisteer.capeval import store as cs
    from sisteer.capeval.tiers import inspect_served as isv

    task = tc.inspect_task("math")
    ids = [f"math_{i:08x}" for i in range(3)]
    monkeypatch.setattr("inspect_ai.eval_async",
                        _fake_eval({0: _Log([_Sample(i, {"s": _Score("C")}) for i in ids])}))
    template = {"sweep": "two_si_controls_served", "run_id": "served-controls-t", "model_id": "m", "model_revision": REV,
                "arm_name": "t", "kind": "cell", "coeff": 1.0, "intervention_hash": OBJECT_SHA, "engine": "vllm",
                "trial_id": 0, "seed": 0, "attempt": 0, "harness_config_hash": "h" * 8, "env_digest": "e" * 8,
                "slurm_job_id": None}
    store = cs.CellStore(tmp_path / "sweep", "cell", row_constants=template)
    plan = isv.TaskPlan(row_task=task, name="math", task=object(), expected_n=len(ids), samples_per_item=1,
                        max_tokens=M5T_CAP, thinking=tc.THINKING_ON)
    asyncio.run(isv.run_one_task(plan, model=None, store=store, template=template, thinking=tc.THINKING_ON,
                                 concurrency=4))
    keys = store.completed_keys()
    # the producer's shape: one `acc` row per item and the two aggregate rows
    assert {k for k in keys if tc.is_aggregate_item(k[1])} == {(task, tc.AGG_ITEM_ID, tc.METRIC_ACC_AGGREGATE),
                                                              (task, tc.AGG_ITEM_ID, tc.METRIC_N)}
    assert {k for k in keys if not tc.is_aggregate_item(k[1])} == {(task, i, tc.METRIC_ACC) for i in ids}
    # one attempt per REAL item: the record from the bytes, the ledger row of the same bytes
    ledger_path = tmp_path / tc.LEDGER_FILENAME
    recs = []
    for i, item in enumerate(ids):
        body = {"model": VMODEL, "messages": [{"role": "user", "content": f"problem {i}"}],
                "max_tokens": M5T_CAP, "seed": i, **tc.chat_template_kwargs(tc.THINKING_ON)}
        raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", tc.HEADER_TASK: task, tc.HEADER_SAMPLE: f"{item}/0",
                   tc.HEADER_REQUEST_ID: uuid.uuid4().hex}
        recs.append(tc.request_evidence_record(raw, headers=headers))
        write_ledger_row(ledger_path, raw, headers)
    rows = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    # the driver's inventory rule: the store's ITEMS, the aggregate pseudo-item dropped
    filed = tc.filed_item_pairs(keys)
    assert filed == {(task, i) for i in ids}
    assert tc.request_evidence_reconciliation(recs, rows, filed=filed) is None
    # the pre-fix inventory (every completed key) is the DEFECT: the aggregate pseudo-item
    # has no attempt, so an otherwise complete leg refused by that name
    why = tc.request_evidence_reconciliation(recs, rows, filed={(t, i) for t, i, _m in keys})
    assert why is not None and "1 filed item(s) have NO recorded attempt" in why and repr((task, tc.AGG_ITEM_ID)) in why
    # a genuine item without its record: refused by name either way
    gone = ids[1]
    why = tc.request_evidence_reconciliation([r for r in recs if r["sample"] != f"{gone}/0"], rows, filed=filed)
    assert why is not None and "NO request-evidence record" in why and f"{gone}/0" in why
    why = tc.request_evidence_reconciliation([r for r in recs if r["sample"] != f"{gone}/0"],
                                             [x for x in rows if x["sample"] != f"{gone}/0"], filed=filed)
    assert why is not None and "1 filed item(s) have NO recorded attempt" in why and repr((task, gone)) in why


def test_a_resume_records_its_attempt_before_the_probes_and_completes_it_only_when_the_run_passes(world, monkeypatch):  # noqa: F811
    """FC4 (3): a resume writes its `resume_attempts` entry (completed: false)
    before any probe, so a preflight that refuses afterwards leaves the attempt
    on record; a run that passes flips its entry to completed: true. Both
    files are append-only, so the second run's attempts reconcile too."""
    w = undeclared_world(world, monkeypatch)
    _stub_math_leg(monkeypatch, [])
    srun.main(w.argv)
    cell_dir = _cell_dir(world.out_root)
    first = json.loads((cell_dir / "manifest.json").read_text())[tc.MANIFEST_KEY_RESUME_ATTEMPTS]
    assert [a["attempt"] for a in first] == [1] and first[0]["completed"] is True
    # a resume whose probes refuse: the attempt is on disk, incomplete
    w2 = undeclared_world(world, monkeypatch, chat=M5TChat(lens_models={VMODEL, world.twin.virtual_model},
                                                          base_models={"petri__base"}, stamp_render="thinking_off"))
    with pytest.raises(SystemExit):
        srun.main(w2.argv)
    second = json.loads((cell_dir / "manifest.json").read_text())[tc.MANIFEST_KEY_RESUME_ATTEMPTS]
    assert [a["attempt"] for a in second] == [1, 2] and second[1]["completed"] is False and second[0]["completed"] is True
    # a resume that passes completes attempt 3; every attempt of the store reconciles
    w3 = undeclared_world(world, monkeypatch)
    srun.main(w3.argv)
    third = json.loads((cell_dir / "manifest.json").read_text())[tc.MANIFEST_KEY_RESUME_ATTEMPTS]
    assert [a["attempt"] for a in third] == [1, 2, 3] and [a["completed"] for a in third] == [True, False, True]
    gate = json.loads((cell_dir / "evidence.json").read_text())["request_evidence_gate"]
    assert gate["problem"] is None


def test_the_undeclared_all_position_object_cell_runs_under_thinking_on_with_the_experiments_record(
        world, monkeypatch):  # noqa: F811
    w = undeclared_world(world, monkeypatch)
    calls = _stub_math_leg(monkeypatch, [])
    srun.main(w.argv)
    cell_dir = _cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    evidence = json.loads((cell_dir / "evidence.json").read_text())
    ident = manifest["resolved_config"]["tinycap"]
    # the identity: version 2, the executed cap keyed by the FULL row task, the measured header, OOS
    # true, NO declaration key (absent, not null), the retry recorded as 0 (a MATH run), the row's terms
    assert ident[tc.IDENTITY_KEY_IDENTITY_VERSION] == 2
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {tc.inspect_task("math"): M5T_CAP}
    assert ident[tc.IDENTITY_KEY_THINKING_HEADER_IDS] == {"off": list(tc.QWEN35_HEADER_IDS_OFF),
                                                          "on": list(tc.QWEN35_HEADER_IDS_ON)}
    assert ident[tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    assert tc.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT not in ident
    assert tc.IDENTITY_KEY_DIAGNOSTIC not in ident
    assert ident[tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] == 0
    assert (ident["temperature"], ident["top_p"], ident["top_k"], ident["min_p"], ident["presence_penalty"]) == (
        0.6, 0.95, 20, 0.0, 0.0)
    assert set(ident) == set(tc.tinycap_identity_fields(
        out_of_support_thinking=True, files_math500=True, identity_version=2, thinking_on=True, declares=False,
        sampling_optional=("min_p", "presence_penalty")))
    # the merge caveat: the first-pass scalar rides beside the v2
    # map and the two derive from ONE cap resolution — equal to each other and to
    # the launcher's pure twin under the matched tuple
    assert ident[tc.IDENTITY_KEY_MATH500_MAX_TOKENS] == ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK][tc.inspect_task("math")]
    assert ident[tc.IDENTITY_KEY_MATH500_MAX_TOKENS] == M5T_CAP == tc.math500_max_tokens_for(
        tc.THINKING_ON, out_of_support=True, experiment=SYNTH_NAME)
    # the manifest: the plain route (no @thinking), the experiment's unhashed record (Lane C's reader contract)
    assert manifest["virtual_model"] == VMODEL and "@thinking" not in manifest["harness_cell_id"]
    assert manifest[tc.MANIFEST_KEY_AUTHORISED_EXPERIMENT] == SYNTH_NAME and "oos_experiment" in manifest
    assert manifest["support"] == tc.SUPPORT_FITTED_UNDER_UNTRAINED_RENDER and manifest["header_located"] is False
    assert manifest["route"][tc.LENS_SERVED_POSITIONS_KEY] == ["all"]
    tp = manifest[tc.MANIFEST_KEY_TRANSFER_PROVENANCE]
    assert tp["header_check_class"] == "document_fit_inapplicable" and tp["header_located"] is False
    assert tp["served_positions"] == ["all"] and tp["served_render"] == "thinking_on"
    assert tp["fit_kind"] == "verified" and tp["fit_response_header_len"] is None      # the world's fixture provenance
    assert tp["header_moves_with_render"] is True and tp["family"] == "qwen3_5"
    assert manifest[tc.MANIFEST_KEY_RENDER_FACT_EVIDENCE]["evidence_status"] == "certified"
    assert manifest[tc.MANIFEST_KEY_RESUME_ATTEMPTS][0]["attempt"] == 1 and manifest["binding"]["ledger_url"]
    assert manifest["sampling"]["mode"] == tc.SAMPLING_THINKING_CARD and manifest["sampling"]["min_p"] == 0.0
    # the evidence: (A), (B′), (C), the OFF bites probe, the live budget on MATH's 500 prompts
    assert evidence["render_certified"]["render_certified"] is True
    assert evidence["render_certified"]["render_source"] == "served_tokenize"
    assert evidence["render_certified"]["think_shape_certified"] is True
    assert evidence["lens_position_attested"]["header_located"] is False
    assert evidence["lens_position_attested"]["attested_policy"] == "always"
    assert evidence[tc.LENS_ON_BITE_KEY]["verified"] is True and evidence[tc.LENS_ON_BITE_KEY]["reasoning_present"]
    bites = evidence["steering_bites"]
    assert bites["verified"] is True and bites[tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == tc.THINKING_OFF
    assert bites["probe_extra_body"] == tc.chat_template_kwargs(tc.THINKING_OFF)
    live = evidence[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK]
    assert live["n"] == 500 and live["task"] == tc.inspect_task("math") and live["ok"] and live["cap"] == M5T_CAP
    assert live["artifact_window_check"]["window"] == M5T_WINDOW
    assert evidence["thinking_header_by_arm"]["header_moves"] is True
    assert "lens_boundary_attestation" in evidence["thinking_header_by_arm"]          # (B) not applied
    assert evidence["ledger_thinking_refusals"] == {} and evidence["request_evidence_gate"]["problem"] is None
    # NO request carried the declaration — probes, render probes, the bite, the task calls
    assert w.chat.calls and all(tc.OOS_THINKING_DECLARATION_KEY not in c for c in w.chat.calls)
    on_calls = [c for c in w.chat.calls if (c.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {}).get(tc.ENABLE_THINKING_KEY) is True]
    assert on_calls                                                                  # the render probes and the bite
    records = [json.loads(l) for l in (world.out_root / tc.REQUEST_EVIDENCE_FILENAME).read_text().splitlines()]
    assert records and all(r["petri_thinking_on_present"] is False for r in records)
    assert {r["kind"] for r in records} == {"probe", "task"}
    assert all(set(r) == set(tc.REQUEST_EVIDENCE_FIELDS) for r in records)      # sorted on disk; the field SET
    assert [r for r in records if r["kind"] == "task"][0]["task"] == tc.inspect_task("math")
    # the leg received the experiment's kwargs
    [kw] = calls
    assert kw["family"] == "qwen3_5" and kw["header_located"] is False and kw["oos_experiment"] == SYNTH_NAME
    assert kw["require_certified_fact"] is True and kw["sampling_ref"] == manifest["sampling_ref"] if "sampling_ref" in manifest else True
    assert kw["out_of_support"] is True and kw["prompt_opened_think"] is True


@pytest.mark.parametrize("name,mutate,rec", [
    # a header-located / mixed / ABSENT served-positions fact makes the configuration a DECLARED one
    # (fail closed): the standup record then lacks the `@thinking` declaration and is refused by name
    ("served_positions_header_located", dict(served_positions=("response",)), tc.REC_OOS_DECLARATION_UNATTESTED),
    ("served_positions_mixed", dict(served_positions=("all", "response")), tc.REC_OOS_DECLARATION_UNATTESTED),
    ("served_positions_absent", dict(served_positions=None), tc.REC_OOS_DECLARATION_UNATTESTED),
    ("digest_differs", dict(endpoint_extra={tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "cd" * 32}), tc.REC_OOS_DECLARATION_MISMATCH),
    ("no_digest_flag", dict(with_digest=False), tc.REC_OOS_DECLARATION_UNATTESTED),
    ("declared_in_map", dict(endpoint_extra={tc.PROXY_IDENTITY_SERVE_DECLARATIONS: {BARE: {"serve_out_of_support": ["thinking"]}}}),
     tc.REC_OOS_DECLARATION_MISMATCH),
])
def test_every_hole_in_the_undeclared_record_refuses_by_name_before_generation(world, monkeypatch, name, mutate, rec):  # noqa: F811
    w = undeclared_world(world, monkeypatch, **mutate)
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    msg = str(e.value)
    assert rec in msg, (name, msg[:300])
    if name.startswith("served_positions"):
        assert "attests no serve_out_of_support" in msg and "@thinking" in msg   # the declaration the cell now needs
    if name == "declared_in_map":
        assert "wiring error" in msg


@pytest.mark.parametrize("drop", ["override", "override_and_digest"])
def test_a_fitted_cell_under_thinking_on_without_the_override_refuses_by_name_before_any_request(world, monkeypatch, drop):  # noqa: F811
    """With the override (and the digest) simply LEFT OUT of a fitted all-position
    cell's argv the driver ran it as `support: in_support`, `oos_experiment: null` — no
    registry tuple, no digest, no (B')/(C), no request-evidence gate — because the
    certified render fact admits thinking-ON and every check hung on the flag. Now the
    fact that this ROUTE (a lens cell) is served under this RENDER (thinking ON) is
    what triggers the requirement (`fitted_thinking_on_out_of_support`), and a run
    without the acknowledgement refuses BY NAME before the parser admission, the
    budget pass, any request or any store."""
    w = undeclared_world(world, monkeypatch, with_digest=(drop == "override"))
    argv = [a for a in w.argv if a != tc.FLAG_ALLOW_OOS_THINKING]
    assert tc.FLAG_ALLOW_OOS_THINKING not in argv
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(argv)
    msg = str(e.value)
    assert tc.REC_OOS_UNAUTHORISED in msg and tc.FLAG_ALLOW_OOS_THINKING in msg
    assert "FITTED lens route" in msg and "not the flag" in msg and tc.FLAG_EXPECT_OBJECT_SHA256 in msg
    assert "not header-located" in msg and "D12 P1-5" in msg          # the reason, from the served positions
    # nothing happened: no request of any kind reached the proxy, no store was touched
    assert w.chat.calls == [] and w.tokenized == []
    assert not (world.out_root / "cells").exists()
    assert not (world.out_root / tc.REQUEST_EVIDENCE_FILENAME).exists()
    # the SAME argv with the flag back is the admitted run (the positive control, with the digest)
    if drop == "override":
        calls = _stub_math_leg(monkeypatch, [])
        srun.main(w.argv)
        manifest = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())
        assert manifest["support"] == tc.SUPPORT_FITTED_UNDER_UNTRAINED_RENDER
        assert manifest[tc.MANIFEST_KEY_AUTHORISED_EXPERIMENT] == SYNTH_NAME and calls


def test_the_fitted_on_requirement_is_the_route_and_render_never_the_flag(world, monkeypatch):  # noqa: F811
    """The predicate behind the gate, at the driver door: the driver's source
    derives `object_on_run` from `fitted_thinking_on_out_of_support` over the
    SERVED positions and the family, and hangs (B')/(C) and the request-evidence
    gate on it; `oos_run` (the flag) stamps the identity key and nothing more.
    A header-located object on a header-invariant family (the campaign's gemma
    psr legs under thinking ON, the two golden sets) is IN support and needs no
    flag; every other fitted ON serve is out of support."""
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "fitted_on_why = tc.fitted_thinking_on_out_of_support(family, header_located=header_located)" in src
    assert "if object_on_run and not oos_run:" in src
    assert "if object_on_run and not header_located and render_error is None:" in src           # (B') / (C)
    # the evidence gate: the thinking-ON class's route + render fact, OR (llama-paper-rows P18) the no-switch class's
    # fitted undeclared route — never the flag
    assert "request_evidence is not None and ((object_on_run and not header_located)" in src
    assert "or (no_switch_run and lens_cell and not header_located))" in src
    assert "lens_cell and oos_on_run and not header_located" not in src                          # the flag-keyed form is gone
    # the truth table the driver reads
    assert tc.fitted_thinking_on_out_of_support("qwen3_5", header_located=False)
    assert tc.fitted_thinking_on_out_of_support("gemma4", header_located=False)                  # gemma too
    assert tc.fitted_thinking_on_out_of_support("qwen3_5", header_located=True)                  # the parent's declared path
    assert tc.fitted_thinking_on_out_of_support("glm4v", header_located=True)
    assert tc.fitted_thinking_on_out_of_support("gemma4", header_located=True) is None           # invariant header: in support
    assert tc.fitted_thinking_on_out_of_support(None, header_located=True) is None
    assert tc.fitted_thinking_on_out_of_support(None, header_located=False)


@pytest.mark.parametrize("served_key", ["qwen35_9b", "gemma4e4b"])
def test_the_served_model_must_pin_the_sampling_rows_key_exactly_in_every_table_backed_mode(world, monkeypatch, served_key):  # noqa: F811
    """The exact config-vs-served model-key check ran only under
    `--sampling card`; the three new table-backed modes fell through to the
    family cross-check, which cannot tell Qwen3.5-4B from Qwen3.5-9B — a wrong
    routing table ran another model under the requested tuple. Here the served
    model (the routing table's base_model) is another model of the SAME family
    (the world's family cross-check answers `qwen3_5` for any id, so the family
    check alone passes) and, as the cross-family control, gemma: both refuse by
    name before any request, naming the mode, both keys and the served id."""
    from sisteer.capeval.serving import geometry as sg
    w = undeclared_world(world, monkeypatch)
    table = json.loads(world.table_path.read_text())
    assert table["base_model"] == BASE_MODEL
    table["base_model"] = sg.MODEL_HF_IDS[served_key]
    world.table_path.write_text(json.dumps(table))
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    msg = str(e.value)
    assert tc.SAMPLING_THINKING_CARD in msg and "'qwen35_4b'" in msg and f"'{served_key}'" in msg
    assert sg.MODEL_HF_IDS[served_key] in msg and "D47 harness (3)" in msg
    assert w.chat.calls == [] and not (world.out_root / "cells").exists()


def test_every_table_backed_sampling_mode_binds_the_served_model_and_the_authorisation(world, monkeypatch):  # noqa: F811
    """The rule at the contract and in the driver's source: the table-backed
    modes are every mode with a per-model table (all but greedy), the driver's
    exact check reads that tuple (not the `card` literal), and the matched
    authorisation is re-checked against the SERVED model's key once the served
    identity is resolved — a served id that pins no key refuses too."""
    from sisteer.capeval.serving import geometry as sg
    assert tc.SAMPLING_TABLE_MODES == (tc.SAMPLING_CARD, tc.SAMPLING_THINKING_CARD, tc.SAMPLING_GENERAL,
                                       tc.SAMPLING_BATTERY, tc.SAMPLING_VENDOR_PRESET)   # ifeval-canonical-battery: the sixth
    assert tc.SAMPLING_GREEDY not in tc.SAMPLING_TABLE_MODES and set(tc.SAMPLING_TABLE_MODES) < set(tc.SAMPLING_MODES)
    assert "SAMPLING_TABLE_MODES" in tc.__all__
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "sampling.mode in tc.SAMPLING_TABLE_MODES and served_model_key != sampling.model_key" in src
    assert "sampling.mode == tc.SAMPLING_CARD" not in src
    assert 'where="driver run (served model)"' in src and "model=served_model_key" in src
    # a served model pinned under NO key: the exact check refuses (never the family fallback)
    w = undeclared_world(world, monkeypatch)
    table = json.loads(world.table_path.read_text())
    table["base_model"] = "example-org/an-unpinned-model"
    world.table_path.write_text(json.dumps(table))
    assert "example-org/an-unpinned-model" not in sg.MODEL_HF_IDS.values()
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    assert "pins key None" in str(e.value) and "D47 harness (3)" in str(e.value)
    assert w.chat.calls == []


def test_a_render_that_is_not_thinking_on_or_a_lens_reply_identical_to_base_refuses_by_name(world, monkeypatch):  # noqa: F811
    # predicate (A): the proxy stamps thinking_off on the ON probe
    chat = M5TChat(lens_models={VMODEL, world.twin.virtual_model}, base_models={"petri__base"},
                   stamp_render=tc.RENDER_REGIME_THINKING_OFF)
    w = undeclared_world(world, monkeypatch, chat=chat)
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit, match="predicate \\(A\\)"):
        srun.main(w.argv)
    ev = json.loads((world.out_root / "evidence.json").read_text())
    assert ev["render_certified"]["render_certified"] is False                    # recorded first
    # predicate (C): the lens reply under ON equals the base reply — the object did not bite
    chat = M5TChat(lens_models={VMODEL, world.twin.virtual_model}, base_models={"petri__base"},
                   lens_on_text="same</think>\nANSWER: 1", base_on_text="same</think>\nANSWER: 1")
    w = undeclared_world(world, monkeypatch, chat=chat)
    with pytest.raises(SystemExit, match="did not bite"):
        srun.main(w.argv)
    ev = json.loads((world.out_root / "evidence.json").read_text())
    assert ev[tc.LENS_ON_BITE_KEY]["verified"] is False
    # predicate (C): no reasoning block on the lens route (the ON render did not reach it)
    chat = M5TChat(lens_models={VMODEL, world.twin.virtual_model}, base_models={"petri__base"},
                   bite_lens_text="ANSWER: 7 with no think block")
    w = undeclared_world(world, monkeypatch, chat=chat)
    with pytest.raises(SystemExit, match="NO reasoning block"):
        srun.main(w.argv)
    # and a tag-free reply on the RENDER probe is predicate (A)'s refusal, which runs first
    chat = M5TChat(lens_models={VMODEL, world.twin.virtual_model}, base_models={"petri__base"},
                   lens_on_text="ANSWER: 7 with no think block")
    w = undeclared_world(world, monkeypatch, chat=chat)
    with pytest.raises(SystemExit, match="predicate \\(A\\).*think shape"):
        srun.main(w.argv)


def test_the_nemotron_state_admits_the_undeclared_object_cell_on_live_render_evidence_alone(world, monkeypatch):  # noqa: F811
    """The proxy's `thinking_header` is the PARTIAL fact
    (`on_ids: null`, `moves: null`) with a non-null error — exactly Nemotron's
    base-only standup — and the undeclared object cell is ADMITTED by (A) + (B′);
    a spy asserts `certify_thinking_header` (predicate (B), the attestation
    compare) is never called and no attestation is synthesised from the fact
    table; the partial fact is recorded verbatim."""
    w = undeclared_world(world, monkeypatch, identity_extra={
        "thinking_header": {"switch": True, "moves": None, "off_ids": list(tc.QWEN35_HEADER_IDS_OFF), "on_ids": None},
        "thinking_header_error": "probe: boundary retokenization at token 16 (synthetic Nemotron state)"})
    _stub_math_leg(monkeypatch, [])
    called = []
    real = tc.certify_thinking_header
    monkeypatch.setattr(tc, "certify_thinking_header", lambda **kw: called.append(kw) or real(**kw))
    srun.main(w.argv)
    assert called == []                                                             # (B) never ran
    ev = json.loads((_cell_dir(world.out_root) / "evidence.json").read_text())
    hdr = ev["thinking_header_by_arm"]
    assert hdr["attested_thinking_header"]["on_ids"] is None and hdr["thinking_header_error"]
    assert hdr[tc.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED] is None              # not certified BY (B); (A) did
    assert ev["render_certified"]["render_certified"] is True
    assert json.dumps(ev).count(json.dumps(list(tc.QWEN35_HEADER_IDS_ON))) >= 1     # the LIVE measurement, recorded
    # nothing wrote the fact table's ids INTO the attestation
    assert hdr["attested_thinking_header"] == w.identity["thinking_header"]


def test_ledger_url_is_required_for_math500_and_a_cap_statement_must_equal_the_resolved_cap(world, monkeypatch):  # noqa: F811
    w = undeclared_world(world, monkeypatch, with_ledger=False)
    _never_runs(monkeypatch)
    with pytest.raises(SystemExit, match="--ledger-url is REQUIRED"):
        srun.main(w.argv)
    w = undeclared_world(world, monkeypatch)
    argv = list(w.argv)
    argv[argv.index(tc.FLAG_MAX_TOKENS_BY_TASK) + 1] = "math=4096"
    with pytest.raises(SystemExit, match="must EQUAL the resolved cap"):
        srun.main(argv)
    # the wrong sampling row for the tuple refuses at the authorisation door, naming the field
    argv = list(w.argv)
    argv[argv.index(tc.FLAG_SAMPLING) + 1] = tc.SAMPLING_CARD
    with pytest.raises(SystemExit) as e:
        srun.main(argv)
    assert tc.REC_OOS_UNAUTHORISED in str(e.value) and " differs" in str(e.value)   # the nearest tuple's field, by name
    # a diagnostic marker needs identity version 2 (present here) and hashes `diagnostic: true`
    calls = _stub_math_leg(monkeypatch, [])
    w = undeclared_world(world, monkeypatch, extra_argv=[tc.FLAG_DIAGNOSTIC])
    srun.main(w.argv)
    ident = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_DIAGNOSTIC] is True and calls[0]["require_certified_fact"] is False


def test_a_single_arm_on_run_on_a_moves_true_lens_cell_measures_both_arms_and_never_raises_mixed_arm(world, monkeypatch):  # noqa: F811
    """The mixed-arm refusal keys on a run that MIXES arms
    on a HEADER-LOCATED mask; a single-arm ON run that measures both arms (the
    version-2 identity) records `header_moves` and refuses nothing; a genuinely
    mixed run on a header-located lens cell still refuses; a mixed run on an
    all-position cell does not."""
    on_only = {tc.inspect_task("math"): tc.THINKING_ON}
    served = {"served_template_hash": "12" * 8}
    from test_served_capability import HEADER_TOK_TEMPLATE_HASH
    served = {"served_template_hash": HEADER_TOK_TEMPLATE_HASH}
    block, error = srun.measure_thinking_header_by_arm(on_only, lens=True, tok=HeaderTok(moves=True),
                                                       measure_both_arms=True, header_located=False, **served)
    assert error is None and block["header_moves"] is True and block["verified"] is True
    assert block["run_arms"] == ["on"] and block["measured_arms"] == ["off", "on"]
    block, error = srun.measure_thinking_header_by_arm(on_only, lens=True, tok=HeaderTok(moves=True),
                                                       measure_both_arms=True, header_located=True, **served)
    assert error is None and not isinstance(error, srun.MixedArmHeaderMoved)      # single-arm: never
    mixed = {tc.inspect_task("math"): tc.THINKING_ON, "inspect/hellaswag": tc.THINKING_OFF}
    _, error = srun.measure_thinking_header_by_arm(mixed, lens=True, tok=HeaderTok(moves=True), **served)
    assert isinstance(error, srun.MixedArmHeaderMoved)
    _, error = srun.measure_thinking_header_by_arm(mixed, lens=True, tok=HeaderTok(moves=True),
                                                   header_located=False, **served)
    assert error is None                                                           # re-keyed on header_located


def test_the_parser_gate_admits_a_certified_fact_and_refuses_a_candidate_one_by_name(capsys, monkeypatch):
    none = {"reasoning_parser": None, "tool_calls": False}
    for fam in ("qwen3_5", "glm4v", "nemotron_h"):    # certified facts (glm4v / nemotron_h by G-PARSER, 2026-09-10)
        srun._check_served_reasoning_parser(["math500"], "on", none, render_fact=tc.FAMILY_RENDER_FACTS[fam])
        assert "IS the parser" in capsys.readouterr().out, fam
    # a candidate fact (a SYNTHETIC demotion of Nemotron's, in the table the gate reads by family): refused by name
    # in production, admitted under the diagnostic smoke's `require_certified=False`
    demoted = tc.FAMILY_RENDER_FACTS["nemotron_h"]._replace(evidence_status=tc.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(tc, "FAMILY_RENDER_FACTS", {**tc.FAMILY_RENDER_FACTS, "nemotron_h": demoted})
    with pytest.raises(SystemExit, match="render fact not certified"):
        srun._check_served_reasoning_parser(["math500"], "on", none, render_fact=demoted)
    srun._check_served_reasoning_parser(["math500"], "on", none, render_fact=demoted,
                                        require_certified=False)                    # the diagnostic smoke
    assert "IS the parser" in capsys.readouterr().out
    monkeypatch.undo()
    with pytest.raises(SystemExit, match="NO reasoning parser"):
        srun._check_served_reasoning_parser(["math500"], "on", none)                 # no fact: unchanged
    # the bites probe body on an undeclared cell: the OFF kwargs and NO declaration
    assert srun._probe_extra_body(tc.THINKING_OFF, header_located=False) == tc.chat_template_kwargs(tc.THINKING_OFF)
    assert srun._probe_extra_body(tc.THINKING_ON, header_located=False) == tc.chat_template_kwargs(tc.THINKING_ON)
    assert srun._probe_extra_body(tc.THINKING_ON) == {**tc.chat_template_kwargs(tc.THINKING_ON), "petri_thinking_on": True}


# --------------------------------------------------------------------------- #
# gsm8k-paired200 record gap (served-positions-null-diagnosis.md §1 / §4.3): a DECLARED standup keys           #
# `lens_served_positions` by `<harness>@thinking`; the driver read the bare key only, got None, and wrote the null #
# into the manifest three times while serving correctly. The lookup is now two-key.                                #
# --------------------------------------------------------------------------- #


def test_the_standup_served_positions_are_read_under_the_spelling_the_standup_serves_the_cell_as(world):  # noqa: F811
    """The two-key lookup on a DECLARED standup-endpoint fixture keyed `<harness>@thinking` ONLY: the declared key hits
    (the old bare-key read is None), the bare key hits on an undeclared standup, neither present keeps the fail-closed
    None, BOTH present refuses by name, a declared-key hit that is not header-located refuses by name; the attestation's
    resolved fact rides the route block and a legacy attestation adds no key; the driver's read site is the new one."""
    declared_key = BARE + tc.OOS_STANDUP_CELL_SUFFIX
    write_standup_endpoint(world.table_path, **{tc.LENS_SERVED_POSITIONS_KEY: {declared_key: ["response"]}})
    ep = srun._standup_endpoint(world.table_path)
    assert tc.served_positions_of_cell(ep, BARE) is None                                  # the 8f5f1495 read: None
    assert srun._standup_served_positions(ep, cell=TT_CELL) == (["response"], declared_key)
    assert srun._standup_served_positions(ep, cell=TT_CELL + tc.OOS_STANDUP_CELL_SUFFIX) == (["response"], declared_key)
    assert tc.header_located(["response"]) and tc.header_located(None)                    # the reading was right, the record was not
    # the undeclared standup: the bare key
    write_standup_endpoint(world.table_path, **{tc.LENS_SERVED_POSITIONS_KEY: {BARE: ["all"]}})
    assert srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL) == (["all"], BARE)
    # neither spelling named / no map: today's None (header-located, fail closed)
    write_standup_endpoint(world.table_path, **{tc.LENS_SERVED_POSITIONS_KEY: {"si_other__psr_kl_all": ["all"]}})
    assert srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL) == (None, None)
    write_standup_endpoint(world.table_path)
    assert srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL) == (None, None)
    # both spellings: not one standup's shape — refused by name
    write_standup_endpoint(world.table_path, **{tc.LENS_SERVED_POSITIONS_KEY: {declared_key: ["response"], BARE: ["response"]}})
    with pytest.raises(SystemExit, match=tc.REC_OOS_DECLARATION_MISMATCH):
        srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL)
    # a declared key serving a mask that is NOT header-located: the standup declared a cell that needs none
    write_standup_endpoint(world.table_path, **{tc.LENS_SERVED_POSITIONS_KEY: {declared_key: ["all"]}})
    with pytest.raises(SystemExit, match=tc.REC_OOS_DECLARATION_MISMATCH):
        srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL)
    # the attestation's resolved fact (`check_serve_attestation` files it under the cell's OWN key — the declared key
    # on a declared run) rides the route block; a legacy attestation without the key adds none
    raw = {declared_key: {tc.SERVE_OUT_OF_SUPPORT_KEY: ["thinking"]}, tc.TWIN_CELL_ID: {tc.SERVE_OUT_OF_SUPPORT_KEY: ["thinking"]}}
    attest = {"declarations": {"raw": raw, "own_key": declared_key}, "object_digest": {"scalar": OBJECT_SHA, "by_cell": {declared_key: OBJECT_SHA}},
              tc.ENDPOINT_KEY_LENS_SALT_POLICY: tc.LENS_SALT_POLICY_ATTESTED, tc.ROUTE_KEY_RESUMED_PREFILL_POLICY: "conditional",
              tc.LENS_SERVED_POSITIONS_KEY: ["response"]}
    block = srun._attestation_route_block(attest)
    assert block[tc.LENS_SERVED_POSITIONS_KEY] == ["response"] and block[tc.ROUTE_KEY_SERVE_DECLARATIONS] == raw
    assert block[tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL] == {declared_key: OBJECT_SHA}
    assert tc.LENS_SERVED_POSITIONS_KEY not in srun._attestation_route_block(None)
    assert tc.LENS_SERVED_POSITIONS_KEY not in srun._attestation_route_block({k: v for k, v in attest.items() if k != tc.LENS_SERVED_POSITIONS_KEY})
    # a record that STATES no value adds no key (the identity-v1 route block's golden shape; version 2 writes its null itself)
    assert tc.LENS_SERVED_POSITIONS_KEY not in srun._attestation_route_block({**attest, tc.LENS_SERVED_POSITIONS_KEY: None})
    # the read site: the two-key lookup, and the bare-key-only read is gone
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "served_positions, served_positions_key = _standup_served_positions(standup_endpoint, cell=args.cell)" in src
    assert "served_positions = tc.served_positions_of_cell(standup_endpoint, bare_key)" not in src
    assert "header_located = tc.header_located(served_positions)" in src


def test_the_standup_served_positions_are_read_off_the_engine_neutral_map_first_and_the_lens_map_as_the_fallback(world):  # noqa: F811
    """native-positions-resolve: the driver read the LENS map only, which a native standup writes EMPTY — so every
    natively served fitted cell recorded `served_positions: null` and a transfer-provenance block derived from the null,
    fail closed (the GSM8K reader's `RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY`). The ENGINE-NEUTRAL `served_positions` map
    (both standups stamp it beside the lens map; a native table's carries every lora route's attested mode) is read FIRST
    under the same two-spelling discipline: the bare key on an undeclared standup, the declared key (header-located) on a
    declared one; an OLD record with no neutral map falls back to the lens map exactly as before; neither map naming the
    cell keeps the fail-closed (None, None) — never a default from the engine's name; both spellings in the neutral map,
    the two maps disagreeing, and a declared-key hit that is not header-located refuse by name."""
    declared_key = BARE + tc.OOS_STANDUP_CELL_SUFFIX
    assert tc.SERVED_POSITIONS_KEY == "served_positions" and tc.SERVED_POSITIONS_KEY != tc.LENS_SERVED_POSITIONS_KEY
    # a NATIVE record: the neutral map names the bare key, the lens map is empty (what a native standup writes)
    write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {BARE: ["all"]}, tc.LENS_SERVED_POSITIONS_KEY: {}})
    ep = srun._standup_endpoint(world.table_path)
    assert tc.recorded_served_positions_of_cell(ep, BARE) == ["all"] and tc.served_positions_of_cell(ep, BARE) is None
    assert srun._standup_served_positions(ep, cell=TT_CELL) == (["all"], BARE)
    assert tc.header_located(["all"]) is False                                   # the block derives NOT header-located from it
    # the neutral map states a restricted NATIVE mode the lens gate cannot classify (`prefill_all` / `prompt_last` are in
    # NATIVE_POSITION_MODES, not in POSITION_MODES): `header_located` would read it header-located, fail closed, and the
    # manifest would stamp a WRONG POSITIVE — refused by name instead, nothing stamped (never rewritten to `all` either)
    for mode in ("prefill_all", "prompt_last"):
        assert tc.unclassified_native_modes([mode]) == [mode] and tc.header_located([mode]) is True
        write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {BARE: [mode]}, tc.LENS_SERVED_POSITIONS_KEY: {}})
        with pytest.raises(SystemExit, match=f"{tc.NATIVE_POSITION_MODE_UNCLASSIFIED}: '{mode}'"):
            srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL)
    assert tc.unclassified_native_modes(["all"]) == tc.unclassified_native_modes(["response"]) == []
    # the `none` card serves no fitted object: the driver's lookup is gated on the lens grammar and is never asked for it
    assert tc.is_lens_cell(tc.ROUTE_NONE_CELL) is False and tc.is_lens_cell(TT_CELL) is True
    # both maps present and agreeing (a fixed LENS standup): the same answer
    write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {BARE: ["all"]}, tc.LENS_SERVED_POSITIONS_KEY: {BARE: ["all"]}})
    assert srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL) == (["all"], BARE)
    # an OLD record (no neutral map): the lens map is the fallback, byte for byte the prior reading
    write_standup_endpoint(world.table_path, **{tc.LENS_SERVED_POSITIONS_KEY: {BARE: ["all"]}})
    ep_old = srun._standup_endpoint(world.table_path)
    assert tc.SERVED_POSITIONS_KEY not in ep_old and tc.recorded_served_positions_of_cell(ep_old, BARE) is None
    assert srun._standup_served_positions(ep_old, cell=TT_CELL) == (["all"], BARE)
    # a declared standup's neutral map: the declared key, header-located
    write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {declared_key: ["response"]}, tc.LENS_SERVED_POSITIONS_KEY: {}})
    assert srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL) == (["response"], declared_key)
    # neither map names the cell: the fail-closed None stands (the reader refuses it by name — never `all` by default)
    write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {"si_other__psr_kl_all": ["all"]}, tc.LENS_SERVED_POSITIONS_KEY: {}})
    assert srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL) == (None, None)
    write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {}, tc.LENS_SERVED_POSITIONS_KEY: {}})
    assert srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL) == (None, None)
    # known-bad, each by name: both spellings in the neutral map; the two maps disagreeing; a declared key not header-located
    write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {BARE: ["all"], declared_key: ["response"]}})
    with pytest.raises(SystemExit, match=f"{tc.REC_OOS_DECLARATION_MISMATCH}.*{tc.SERVED_POSITIONS_KEY}.*BOTH"):
        srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL)
    write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {BARE: ["all"]}, tc.LENS_SERVED_POSITIONS_KEY: {BARE: ["response"]}})
    with pytest.raises(SystemExit, match=f"{tc.REC_OOS_DECLARATION_MISMATCH}.*disagree"):
        srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL)
    write_standup_endpoint(world.table_path, **{tc.SERVED_POSITIONS_KEY: {declared_key: ["all"]}})
    with pytest.raises(SystemExit, match=f"{tc.REC_OOS_DECLARATION_MISMATCH}.*not header-located"):
        srun._standup_served_positions(srun._standup_endpoint(world.table_path), cell=TT_CELL)
    # the read site reads the neutral map through the contract's reader, first
    src = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "(tc.SERVED_POSITIONS_KEY, tc.recorded_served_positions_of_cell)" in src
    assert src.index("tc.recorded_served_positions_of_cell") < src.index("(tc.LENS_SERVED_POSITIONS_KEY, tc.served_positions_of_cell)")
