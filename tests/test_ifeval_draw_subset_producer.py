"""ifeval-canonical-battery, fix lane X: the DRAW-subset contract reaches the PRODUCER.

Before this the ladder's ``rung_subset/2`` (draw pairs) could not run: the driver read ``--item-subset`` with the
v1-only validator (which rejects ``draws`` / ``n_draws``), reduced the file to ``ids``, and the runner evaluated the
same ids for BOTH trials; the identity helpers hashed ids, not draws. Now:

* ``contracts.validate_item_subset`` / ``read_item_subset`` / ``cap_rung_identity`` / ``item_subset_block`` dispatch on
  ``schema`` — the v1 item contract byte-identical, the v2 draw contract equal to lane 2's ``ifeval_contracts`` twins
  (held equal here on the SAME documents); a v2 ``cap_rung`` hashes the sorted DRAWS, so trial 0 and trial 1 of one
  item are two identities;
* the plan narrows EACH TRIAL to its own items (``TaskPlan.sample_ids_by_sample``; a trial with no draw is never
  evaluated), ``expected_n = n_run = n_items``, ``expected_draws = n_draws``;
* the runner issues one ``eval_async`` per trial that has draws, with exactly those ids, files per-sample rows + the
  outcome record + the summary and NO per-item mean row (a mean over one of an item's trials is not avg@k), and
  ``finish_task`` receives the REAL draw counts (the marker carries ``n_draws`` / ``n_expected_draws``);
* a resumed rung is skipped with ZERO requests through the ONE strict validator (``rung_probe.draw_subset_task_reason``)
  and a DONE store whose content is not the subset's draws refuses by name;
* the regression the review asked for: a PLANNER-PRODUCED trial-1-only subset (``[["1000", 1]]``, written by
  ``inspect_rung.classify_store`` over a synthetic pass-1 store whose only at-cap draw is item 1000's second trial)
  through the REAL driver (``served_capability_run.main``) and the REAL runner / provider against a LOOPBACK HTTP
  proxy + ledger: exactly ONE chat request, carrying item 1000 / trial 1, the rung cap and the pass-1 seed of
  (1000, 1) — ``contracts.ifeval_sample_seed`` is cap-independent — then ZERO requests on resume.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest

from sisteer.capeval.serving import proxy_engine as pe
from sisteer.capeval.serving import usage_ledger as ul
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC
from sisteer.capeval.tinycap import rung_probe as RP

import ifeval_synthetic_store as G
import served_capability_run as srun
from test_inspect_served import _Emitter, _Log, _Store, _fake_eval
from test_inspect_served_ifeval import _IfSample, _VALUE, _offline_for_this_module, needs_dataset, real_plan  # noqa: F401
from test_served_capability import (  # noqa: F401 — the driver's offline world
    LIVE_IDENTITY, STANDUP_SEQS, _cell_dir, _landed_stamps, standup_geometry, world, write_ledger_endpoint,
    write_standup_endpoint)

REPO = Path(__file__).resolve().parents[1]
IFEVAL = C.IFEVAL_TASK
CAP1, CAP2 = 2048, 4096
SEED_1000_T1 = C.ifeval_sample_seed("1000", 1)


def _v2(draws, *, cap=CAP2, source_cap=CAP1, n_at_cap=None, n_loop=0, n_boundary=0):
    pairs = sorted({(str(i), int(t)) for i, t in draws})
    ids = sorted({i for i, _ in pairs})
    n_at = len(pairs) + n_loop if n_at_cap is None else n_at_cap
    return {"schema": C.ITEM_SUBSET_SCHEMA_V2, "task": IFEVAL, "source_out_root": "/vol/wave/controls/x/none/vendor_preset",
            "source_identity_hash": "0123456789abcdef", "source_cap": source_cap, "cap": cap, "rule": IC.IFEVAL_TRIGGER_RULE,
            "threshold": IC.IFEVAL_TRIGGER_THRESHOLD, "ids": ids, "draws": [[i, t] for i, t in pairs], "n_items": len(ids),
            "n_draws": len(pairs), "n_at_cap": n_at, "n_truncated": len(pairs), "n_loop": n_loop, "n_boundary": n_boundary,
            "transport_check": {"ok": True}}


def _v1(ids=("1000", "1001"), *, cap=CAP2, source_cap=CAP1):
    return {"schema": C.ITEM_SUBSET_SCHEMA, "task": IFEVAL, "source_out_root": "/vol/wave/controls/x/none/vendor_preset",
            "source_identity_hash": "0123456789abcdef", "source_cap": source_cap, "cap": cap, "rule": "r", "threshold": 0.01,
            "ids": list(ids), "n_at_cap": len(ids), "n_non_terminating": 0, "n_boundary": 0, "transport_check": {"ok": True}}


# --------------------------------------------------------------------------- #
# 1. the contracts: schema dispatch, lane 2's twins held equal, v1 untouched     #
# --------------------------------------------------------------------------- #
def test_the_contracts_dispatch_on_schema_and_equal_lane_2s_v2_twins_on_the_same_documents(tmp_path):
    docs = {"sample_1_only": _v2([("1000", 1)]), "both_trials": _v2([("1000", 0), ("1000", 1)]),
            "mixed": _v2([("1000", 1), ("1001", 0), ("1001", 1), ("1007", 0)], n_loop=2, n_boundary=1)}
    for name, doc in docs.items():
        assert C.item_subset_is_draw_level(doc)
        assert C.validate_item_subset(doc, where=name) == IC.validate_item_subset_v2(doc, where=name) == doc
        assert C.cap_rung_identity(doc) == IC.cap_rung_identity_v2(doc)
        assert tuple(C.cap_rung_identity(doc)) == C.CAP_RUNG_FIELDS
        assert C.item_subset_block(doc, path="/vol/s.json") == IC.item_subset_block_v2(doc, path="/vol/s.json")
        assert tuple(C.item_subset_block(doc, path="/p")) == C.ITEM_SUBSET_BLOCK_FIELDS_V2 == IC.ITEM_SUBSET_BLOCK_FIELDS_V2
        assert C.item_subset_draws(doc) == IC.draws_of_subset(doc) == sorted((str(i), int(t)) for i, t in doc["draws"])
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(doc))
        assert C.read_item_subset(p) == doc                                    # the reader dispatches too
    # the identity: the SAME items, different draws -> different keys; v1 (ids) never equals v2 (pairs)
    t0, t1 = _v2([("1000", 0)]), _v2([("1000", 1)])
    assert C.cap_rung_identity(t0)["subset_sha16"] != C.cap_rung_identity(t1)["subset_sha16"]
    assert C.cap_rung_identity(t1)["subset_sha16"] == C.item_subset_sha16_v2([["1000", 1]]) == IC.item_subset_sha16_v2([("1000", 1)])
    assert C.item_subset_sha16(["1000"]) != C.item_subset_sha16_v2([["1000", 0]])
    assert C.item_subset_sha16_v2([["b", 1], ["a", 0]]) == C.item_subset_sha16_v2([("a", 0), ("b", 1)])   # order-free
    # the names ifeval_contracts registers resolve to the contracts' and agree (its merge seam)
    rep = IC.merge_agreement_report()
    for name in ("ITEM_SUBSET_SCHEMA_V2", "ITEM_SUBSET_FIELDS_V2", "ITEM_SUBSET_BLOCK_FIELDS_V2", "DONE_MARKER_DRAW_FIELDS",
                 "item_subset_sha16_v2"):
        assert rep[name] == {"in_contracts": True, "agree": True}, (name, rep[name])
    assert C.ITEM_SUBSET_SCHEMAS == (C.ITEM_SUBSET_SCHEMA, C.ITEM_SUBSET_SCHEMA_V2)
    # a v2 inconsistency refuses through the contracts with lane 2's message
    torn = {**t1, "ids": ["1000", "1001"], "n_items": 2}
    with pytest.raises(C.ItemSubsetError, match="ids must be exactly the sorted distinct items of draws"):
        C.validate_item_subset(torn)
    with pytest.raises(C.ItemSubsetError, match="ids must be exactly the sorted distinct items of draws"):
        IC.validate_item_subset_v2(torn)
    with pytest.raises(C.ItemSubsetError, match="not a rung of the IFEval ladder"):
        C.validate_item_subset(_v2([("1000", 1)], cap=3000))


def test_the_v1_item_contract_is_byte_identical_and_names_no_draws():
    doc = _v1()
    assert not C.item_subset_is_draw_level(doc)
    assert C.validate_item_subset(doc) == doc
    assert C.cap_rung_identity(doc)["subset_sha16"] == C.item_subset_sha16(doc["ids"])
    block = C.item_subset_block(doc, path="/p")
    assert tuple(block) == C.ITEM_SUBSET_BLOCK_FIELDS and "draws" not in block
    # the pre-existing v1 refusals, verbatim: the exact-keys message and the eligible-set arithmetic
    with pytest.raises(C.ItemSubsetError, match=r"keys .* are not the contract's .* \(missing \['ids'\], unknown \[\]\)"):
        C.validate_item_subset({k: v for k, v in doc.items() if k != "ids"})
    with pytest.raises(C.ItemSubsetError, match="the subset IS the eligible set"):
        C.validate_item_subset({**doc, "n_at_cap": 5})
    with pytest.raises(C.ItemSubsetError, match="is the item-level 'rung_subset/1' contract"):
        C.item_subset_draws(doc)
    with pytest.raises(C.ItemSubsetError, match="schema 'rung_subset/9' is not 'rung_subset/1'"):
        C.validate_item_subset({**doc, "schema": "rung_subset/9"})
    with pytest.raises(C.ItemSubsetError, match="expected a JSON object"):
        C.validate_item_subset(["not", "a", "mapping"])


# --------------------------------------------------------------------------- #
# 2. the plan: each trial narrowed to its own draws, through the STOCK task     #
# --------------------------------------------------------------------------- #
def _plan(item_subset, draws, **kw):
    return isv.plan_task(C.IFEVAL_TASK_NAME, grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO,
                         samples_per_item=2, item_subset=item_subset, item_subset_draws=draws,
                         off_cap_setter={C.IFEVAL_TASK_NAME: CAP2}, **kw)


@needs_dataset
def test_the_plan_narrows_each_trial_to_its_draws_and_refuses_a_torn_pair_of_lists(real_plan):
    plan = _plan(["1000"], [["1000", 1]])
    assert plan.sample_ids == ["1000"] and plan.expected_n == plan.n_run == 1 and plan.limit is None
    assert plan.draw_subset == [("1000", 1)] and plan.expected_draws == 1 and plan.max_tokens == CAP2
    assert plan.sample_ids_by_sample == {0: [], 1: ["1000"]}
    assert plan.ids_for(0) == [] and plan.ids_for(1) == ["1000"]
    assert "RUNG (draws): 1 of the source store's (item, trial) draws over 1 items" in plan.note
    mixed = _plan(["1000", "1001"], [["1001", 0], ["1000", 1], ["1001", 1]])
    assert mixed.sample_ids_by_sample == {0: ["1001"], 1: ["1000", "1001"]} and mixed.expected_draws == 3
    assert (mixed.expected_n, mixed.n_run) == (2, 2)
    # an item-level (v1) rung plans exactly as before: every trial draws every listed item
    item_level = _plan(["1000", "1001"], None)
    assert item_level.draw_subset is None and item_level.sample_ids_by_sample is None and item_level.expected_draws is None
    assert item_level.ids_for(0) == item_level.ids_for(1) == ["1000", "1001"]
    # a full pass-1 plan is untouched (541 ids, every trial the same)
    assert real_plan.draw_subset is None and real_plan.ids_for(1) == real_plan.sample_ids and len(real_plan.sample_ids) == 541
    # torn pairs of lists refuse by name
    with pytest.raises(isv.InspectServedError, match="draw subset's items \\(1\\) are not the item subset's ids \\(2\\)"):
        _plan(["1000", "1001"], [["1000", 1]])
    with pytest.raises(isv.InspectServedError, match=r"draw sample indices \[2\] are outside \[0, 2\)"):
        _plan(["1000"], [["1000", 2]])
    with pytest.raises(isv.InspectServedError, match="repeats a draw"):
        _plan(["1000"], [["1000", 1], ["1000", 1]])


def test_a_draw_subset_is_refused_without_its_item_subset_and_on_the_item_level_gsm8k_rung():
    with pytest.raises(isv.InspectServedError, match="item_subset_draws given without item_subset"):
        isv.plan_task(C.IFEVAL_TASK_NAME, grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO,
                      item_subset_draws=[["1000", 1]])
    with pytest.raises(isv.InspectServedError, match="is the k = 2 IFEval rung's contract alone"):
        isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=C.THINKING_ON, repo_root=REPO,
                      item_subset=["gsm8k_x"], item_subset_draws=[["gsm8k_x", 0]])


# --------------------------------------------------------------------------- #
# 3. the runner: one call per trial with draws, no mean row, draw counts filed  #
# --------------------------------------------------------------------------- #
def _rung_plan(draws, **kw):
    pairs = sorted(draws)
    ids = sorted({i for i, _ in pairs})
    by = {t: sorted(i for i, tt in pairs if tt == t) for t in range(2)}
    base = dict(row_task=IFEVAL, name=C.IFEVAL_TASK_NAME, task=object(), expected_n=len(ids), n_run=len(ids),
                samples_per_item=2, primary_scorer="instruction_following", max_tokens=CAP2, thinking=C.THINKING_OFF,
                item_set="ifeval541", sample_ids=ids, draw_subset=pairs, sample_ids_by_sample=by, expected_draws=len(pairs),
                cap_source=C.CAP_SOURCE_OFF_CAP_EXPLICIT, note="synthetic draw-subset rung plan")
    base.update(kw)
    return isv.TaskPlan(**base)


def test_the_runner_draws_only_the_subsets_trials_files_no_mean_row_and_a_draw_marker_then_resumes_with_zero_calls(
        tmp_path, monkeypatch):
    seen: List[Any] = []
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({1: _Log([_IfSample("1000", _VALUE)])}, seen))
    em, store = _Emitter(), _Store(tmp_path / "cell")
    plan = _rung_plan([("1000", 1)])
    summary = asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, concurrency=4, emitter=em,
                                           sampling_ref="abc123def456"))
    # exactly ONE eval call: trial 1, item 1000 alone, at the rung cap; trial 0 was never evaluated
    assert [(s[1], s[2], s[3]) for s in seen] == [(1, CAP2, ["1000"])]
    assert em.by(C.sample_metric(C.METRIC_ACC, 0)) == [] and [r["item_id"] for r in em.by(C.sample_metric(C.METRIC_ACC, 1))] == ["1000"]
    assert em.by(C.METRIC_ACC) == []                                           # no per-item mean row on a draw rung
    assert em.by(C.METRIC_N)[0]["value"] == 1.0 and em.by(C.METRIC_ACC_AGGREGATE)[0]["extra"]["n"] == 1
    assert summary["n"] == 1 and summary["n_scored_samples"] + summary["n_unscored_samples"] == 1 and summary["n_paired"] == 0
    # the marker: DONE, with BOTH counts; the outcome record: the one draw, its pass-1 seed
    marker = json.loads(completion.done_marker_path(store.cell_dir, IFEVAL).read_text())
    assert marker["n_items"] == 1 and marker["n_draws"] == 1 and marker["n_expected_draws"] == 1
    outcomes = completion.read_task_outcomes(store.cell_dir, IFEVAL)
    assert [(i["item_id"], i["sample_idx"], i["seed"]) for i in outcomes["items"]] == [("1000", 1, SEED_1000_T1)]
    assert outcomes["n_attempted_items"] == 1 and outcomes["n_attempted_samples"] == 1
    assert RP.draw_subset_task_reason(store.cell_dir, IFEVAL, draws=[["1000", 1]], cap=CAP2) is None
    # resume: the strict validator admits the store and the runner issues ZERO calls

    async def never(*a, **k):
        raise AssertionError("a complete draw-subset rung must not be re-drawn")

    monkeypatch.setattr("inspect_ai.eval_async", never)
    again = asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, concurrency=4, emitter=em,
                                         sampling_ref="abc123def456"))
    assert again["skipped"].startswith("draw-subset rung complete") and again["n"] == 1
    # a DONE marker over OTHER draws (trial 0 of the same item) is refused by name, never generated over
    other = _rung_plan([("1000", 0)])
    with pytest.raises(isv.InspectServedError, match="NOT a valid draw-subset rung .* repair = delete this rung store"):
        asyncio.run(isv.run_one_task(other, model=None, store=store, template={}, concurrency=4, emitter=em,
                                     sampling_ref="abc123def456"))


def test_a_pass_1_ifeval_store_declares_items_x_trials_and_a_trial_missing_a_draw_refuses(tmp_path, monkeypatch):
    """Pass 1 completeness is a draw fact too: the marker carries n_draws / n_expected_draws = 541 x 2 at
    production size — here 2 items x 2 trials; every OTHER task's marker stays byte-identical (test_draw_subset_contract).
    A draw-subset trial that came back with the wrong items refuses."""
    seen: List[Any] = []
    logs = {0: _Log([_IfSample("1000", _VALUE), _IfSample("1001", _VALUE)]),
            1: _Log([_IfSample("1000", _VALUE), _IfSample("1001", _VALUE)])}
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval(logs, seen))
    em, store = _Emitter(), _Store(tmp_path / "pass1")
    plan = isv.TaskPlan(row_task=IFEVAL, name=C.IFEVAL_TASK_NAME, task=object(), expected_n=2, samples_per_item=2,
                        primary_scorer="instruction_following", max_tokens=CAP1, thinking=C.THINKING_OFF,
                        item_set="ifeval541", sample_ids=["1000", "1001"], note="pass 1")
    asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, concurrency=4, emitter=em,
                                 sampling_ref="abc123def456"))
    marker = json.loads(completion.done_marker_path(store.cell_dir, IFEVAL).read_text())
    assert (marker["n_items"], marker["n_draws"], marker["n_expected_draws"]) == (2, 4, 4)
    assert len(em.by(C.METRIC_ACC)) == 2                                       # pass 1 keeps its per-item mean rows
    # a rung trial answering with an item the subset did not name at that trial: the pass log's id-multiset
    # guard fires first (the log is not the selection); the per-trial draw check behind it is the second door
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({1: _Log([_IfSample("1001", _VALUE)])}))
    with pytest.raises(isv.InspectServedError,
                       match="does not carry this task's pass selection|a rung's population is its draws, whole"):
        asyncio.run(isv.run_one_task(_rung_plan([("1000", 1)]), model=None, store=_Store(tmp_path / "torn"), template={},
                                     concurrency=4, emitter=_Emitter(), sampling_ref="abc123def456"))


# --------------------------------------------------------------------------- #
# 4. the regression: a planner-produced ["1000", 1] subset through the REAL     #
#    driver + runner against a loopback proxy + ledger                          #
# --------------------------------------------------------------------------- #
class _Loopback:
    """A loopback OpenAI-compatible endpoint standing in for the RoutingProxy AND the usage ledger in front of it:
    ``GET /v1/models`` lists every routing-table vmodel with the proxy's identity block; ``POST /v1/chat/completions``
    answers a short compliant completion stamped the way the proxy stamps a served response (the route's four
    attestation stamps + the landed regime triple read off the wire body) and — on the LEDGER port — appends the
    usage ledger's row for the exact bytes received (the same request id / body digest the driver's request-evidence
    record carries), so the driver's post-run reconciliation reads one inventory."""

    TEXT = "Section one is highlighted here and section two follows without any commas at all so the answer is complete."

    def __init__(self, routes, ledger_path: Path):
        self.routes = {r.virtual_model: r for r in routes}
        self.ledger_path = ledger_path
        self.requests: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        #: crash injection: once set to a trial index, every chat request of that trial beyond the first
        #: `fail_after` served ones is answered 500 (the upstream died mid-trial); the ledger port records the
        #: rejected row like the real ledger would. None = never fail.
        self.fail_trial: int | None = None
        self.fail_after: int = 0
        self.served_by_trial: Dict[int, int] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            ledger = False

            def log_message(self, *a):  # quiet
                pass

            def _send(self, status, doc):
                raw = json.dumps(doc).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    return self._send(200, {"object": "list", "data": [{"id": v, "object": "model"} for v in outer.routes],
                                            C.PROXY_IDENTITY_KEY: dict(LIVE_IDENTITY)})
                self._send(404, {"error": {"message": f"no {self.path}"}})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": f"no {self.path}"}})
                body = json.loads(raw.decode("utf-8"))
                route = outer.routes[body["model"]]
                headers = {k: v for k, v in self.headers.items()}
                sample = next((v for k, v in headers.items() if k.lower() == C.HEADER_SAMPLE.lower()), None)
                trial = int(str(sample).rpartition("/")[2]) if sample and str(sample).rpartition("/")[2].isdigit() else None
                with outer.lock:
                    if trial is not None and self.ledger:
                        outer.served_by_trial[trial] = outer.served_by_trial.get(trial, 0) + 1
                    dies = (outer.fail_trial is not None and trial == outer.fail_trial
                            and outer.served_by_trial.get(trial, 0) > outer.fail_after)
                if dies:
                    # a 400 rather than a 500: the OpenAI SDK does not retry a 4xx (a 5xx costs it three attempts with
                    # exponential back-off per sample), while inspect's own `retry_on_error` still fails the sample and,
                    # under `fail_on_error`, the eval — the same death, seconds instead of minutes
                    with outer.lock:
                        outer.requests.append({"body": body, "headers": headers, "raw": raw, "ledger": self.ledger, "failed": True})
                        if self.ledger:
                            outer._ledger_row(body, raw, headers, None, status=400)
                    return self._send(400, {"error": {"message": "upstream died mid-trial (S1 crash injection)"}})
                stamps = pe.expected_stamps(route)
                doc = {"id": "cmpl-loopback", "object": "chat.completion", "created": int(time.time()),
                       "model": stamps["model"], "petri_intervention_hash": stamps["petri_intervention_hash"],
                       "petri": {"cell_id": stamps["petri.cell_id"], "serving_identity": stamps["petri.serving_identity"],
                                 **_landed_stamps(body, declared=False)},
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": outer.TEXT},
                                    "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}}
                with outer.lock:
                    outer.requests.append({"body": body, "headers": headers, "raw": raw, "ledger": self.ledger})
                    if self.ledger:
                        outer._ledger_row(body, raw, headers, doc)
                self._send(200, doc)

        class LedgerHandler(Handler):
            ledger = True

        self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.ledger = ThreadingHTTPServer(("127.0.0.1", 0), LedgerHandler)
        self.threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (self.proxy, self.ledger)]
        for t in self.threads:
            t.start()

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.proxy.server_address[1]}/v1"

    @property
    def ledger_url(self) -> str:
        return f"http://127.0.0.1:{self.ledger.server_address[1]}/v1"

    def _ledger_row(self, body, raw, headers, doc, *, status: int = 200) -> None:
        def h(name):
            return next((v for k, v in headers.items() if k.lower() == name.lower()), None)
        row = dict.fromkeys(C.LEDGER_ROW_FIELDS)
        ok = status == 200
        row.update(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), task=h(C.HEADER_TASK), sample=h(C.HEADER_SAMPLE),
                   has_tools=False, n_tools=0, rejected=not ok, status=status, error_code=None, prompt_tokens_pre=50,
                   usage_prompt_tokens=(50 if ok else None), usage_completion_tokens=(30 if ok else None),
                   finish_reason=("stop" if ok else None), latency_s=0.01,
                   tokenize_mismatch=False, error_class=(C.LEDGER_ERROR_CLASS_NONE if ok else "bad_request_other"),
                   request_id=h(C.HEADER_REQUEST_ID), body_sha256=C.wire_body_sha256(raw))
        row.update(ul.body_facts(body))
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ul.ledger_row(**row)) + "\n")

    def close(self) -> None:
        for s in (self.proxy, self.ledger):
            s.shutdown()
            s.server_close()

    def chat_requests(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["ledger"] and not r.get("failed")]

    def failed_requests(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["ledger"] and r.get("failed")]


def _planner_produced_subset(gen_root: Path, ids: List[str], monkeypatch):
    """``inspect_rung.classify_store`` over a synthetic pass-1 store whose items are the STOCK task's first 40 ids and
    whose ONLY at-cap draw is (1000, trial 1): 1 of 80 draws = 1.25 % > the 1 % trigger -> ESCALATE with the draw
    subset [["1000", 1]] — the planner's own writer, not a hand-typed document."""
    n = 40
    real = ids[:n]
    assert "1000" in real
    monkeypatch.setattr(G, "N", n)
    monkeypatch.setattr(G, "IDS", real)
    monkeypatch.setattr(G, "PROMPTS", [f"Synthetic instruction prompt {i}." for i in range(n)])
    monkeypatch.setattr(G, "N_INST", [1 + (i % 3) for i in range(n)])
    row = G.row_for(G.matrix_rows(), "llama8b", "main", "none")
    src_cell = G.build_store(gen_root, row, at_cap={(real.index("1000"), 1)}, cap=CAP1)
    src_root = src_cell.parents[1]
    record, subset = G.classify(src_root)
    assert record["exit"] == "escalate" and subset is not None, record["exit"]
    assert subset["schema"] == C.ITEM_SUBSET_SCHEMA_V2 and subset["draws"] == [["1000", 1]] and subset["n_draws"] == 1
    path = G.write_rung_subset(gen_root, row, subset, rung=CAP2)
    return subset, path, src_root


@needs_dataset
def test_a_planner_produced_trial_1_only_subset_runs_through_the_real_driver_and_runner_with_one_request_then_zero(
        world, real_plan, monkeypatch, tmp_path):
    from test_served_capability import stub_framework_versions
    stub_framework_versions(monkeypatch)          # the pin check at the driver; the REAL inspect_ai stays importable
    monkeypatch.setenv(C.INSPECT_API_KEY_ENV, "loopback-bearer")   # the provider's bearer; the loopback ignores it
    subset, subset_path, src_root = _planner_produced_subset(tmp_path / "gen", list(real_plan.sample_ids), monkeypatch)
    rung_root = tmp_path / "rung4096"
    table = json.loads(world.table_path.read_text())
    from sisteer.capeval.serving import proxy as px
    routes = px.load_routing_table(world.table_path)["routes"]
    lb = _Loopback(routes, rung_root / C.LEDGER_FILENAME)
    try:
        # the standup + ledger records name the loopback; the window comfortably holds the rung cap
        write_standup_endpoint(world.table_path, proxy_url=lb.proxy_url, max_model_len=8192,
                               geometry=standup_geometry(8192, STANDUP_SEQS))
        write_ledger_endpoint(world.table_path, ledger_url=lb.ledger_url, upstream=lb.proxy_url)
        argv = list(world.argv)
        argv[argv.index("--proxy-url") + 1] = lb.proxy_url
        argv[argv.index("--controls") + 1] = C.CONTROL_IFEVAL
        argv[argv.index("--out-root") + 1] = str(rung_root)
        argv.remove("--greedy")
        argv += [C.FLAG_LEDGER_URL, lb.ledger_url, C.FLAG_THINKING, C.THINKING_OFF,
                 C.FLAG_SAMPLING, C.SAMPLING_VENDOR_PRESET, C.FLAG_SAMPLES, "2",
                 C.FLAG_TINYCAP_IDENTITY_VERSION, "2", C.FLAG_MAX_TOKENS_BY_TASK, f"{C.IFEVAL_TASK_NAME}={CAP2}",
                 C.FLAG_OFF_CAP_BY_TASK, f"{C.IFEVAL_TASK_NAME}={CAP2}", C.FLAG_ITEM_SUBSET, str(subset_path)]
        srun.main(argv)
        reqs = lb.chat_requests()
        assert len(reqs) == 1, [r["headers"] for r in reqs]
        (req,) = reqs
        body, headers = req["body"], {k.lower(): v for k, v in req["headers"].items()}
        assert headers[C.HEADER_TASK.lower()] == IFEVAL and headers[C.HEADER_SAMPLE.lower()] == C.sample_header_value("1000", 1)
        assert body["seed"] == SEED_1000_T1 and body["max_tokens"] == CAP2 and body["model"] == world.route.virtual_model
        assert (body["temperature"], body["top_p"]) == (0.6, 0.9)              # llama8b's vendor row, on the wire
        # the world's model is llama8b, a family with NO thinking switch: no `chat_template_kwargs` on the
        # wire, the arm recorded as `not_available` everywhere the arm is written, the gate REQUIRED the absence
        assert C.CHAT_TEMPLATE_KWARGS_KEY not in body and C.OOS_THINKING_DECLARATION_KEY not in body
        cell_dir = _cell_dir(rung_root)
        manifest0 = json.loads((cell_dir / "manifest.json").read_text())
        assert manifest0["resolved_config"][C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_THINKING_BY_TASK] == {IFEVAL: C.THINKING_NOT_AVAILABLE}
        assert manifest0[C.MANIFEST_KEY_THINKING]["by_task"] == {IFEVAL: C.THINKING_NOT_AVAILABLE}
        # the hashed MODE label resolves through the same derivation as the arm
        assert manifest0[C.MANIFEST_KEY_THINKING]["mode"] == C.THINKING_NOT_AVAILABLE and manifest0[C.MANIFEST_KEY_THINKING]["enable_thinking"] is None
        assert manifest0["resolved_config"][C.MANIFEST_KEY_TINYCAP]["thinking"]["mode"] == C.THINKING_NOT_AVAILABLE
        gate = json.loads((rung_root / "evidence.json").read_text())["request_evidence_sampling_gate"]
        assert gate["problem"] is None and gate["thinking_arm"] == C.THINKING_NOT_AVAILABLE and gate["expected_enable_thinking"] is None
        records = [json.loads(l) for l in (rung_root / C.REQUEST_EVIDENCE_FILENAME).read_text().splitlines() if l.strip()]
        assert records and all(r["enable_thinking_present"] is False for r in records if r["kind"] == C.REQUEST_EVIDENCE_KIND_TASK)
        task_summary = completion.read_task_summary(cell_dir, IFEVAL)
        assert task_summary["thinking"] == C.THINKING_NOT_AVAILABLE and task_summary["render_regime"] == C.RENDER_REGIME_UNSPECIFIED
        assert completion.read_task_outcomes(cell_dir, IFEVAL)["thinking"] == C.THINKING_NOT_AVAILABLE
        marker = json.loads(completion.done_marker_path(cell_dir, IFEVAL).read_text())
        assert (marker["n_items"], marker["n_draws"], marker["n_expected_draws"]) == (1, 1, 1)
        outcomes = completion.read_task_outcomes(cell_dir, IFEVAL)
        assert [(i["item_id"], i["sample_idx"], i["seed"]) for i in outcomes["items"]] == [("1000", 1, SEED_1000_T1)]
        manifest = json.loads((cell_dir / "manifest.json").read_text())
        block = manifest[C.MANIFEST_KEY_ITEM_SUBSET]
        assert block["schema"] == C.ITEM_SUBSET_SCHEMA_V2 and block["draws"] == [["1000", 1]] and block["n_draws"] == 1
        assert block == C.item_subset_block(subset, path=str(subset_path)) == IC.item_subset_block_v2(subset, path=str(subset_path))
        ident = manifest["resolved_config"][C.MANIFEST_KEY_TINYCAP]
        assert ident[C.IDENTITY_KEY_CAP_RUNG] == IC.cap_rung_identity_v2(subset) == C.cap_rung_identity(subset)
        assert ident[C.IDENTITY_KEY_CAP_RUNG]["subset_sha16"] == C.item_subset_sha16_v2([["1000", 1]])
        assert ident[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {IFEVAL: CAP2}
        rows = [json.loads(l) for l in (cell_dir / "rows.jsonl").read_text().splitlines() if l.strip()]
        assert [(r["item_id"], r["metric"]) for r in rows if r["item_id"] != C.AGG_ITEM_ID] == [("1000", C.sample_metric(C.METRIC_ACC, 1))]
        (acc_row,) = [r for r in rows if r["item_id"] != C.AGG_ITEM_ID]
        assert json.loads(acc_row["extra_json"])["thinking"] == C.THINKING_NOT_AVAILABLE          # the per-row arm (P42 vocabulary)
        assert RP.draw_subset_task_reason(cell_dir, IFEVAL, draws=subset["draws"], cap=CAP2,
                                          source_identity_hash=subset["source_identity_hash"]) is None
        assert not (rung_root / "controls_summary.json").exists()             # a rung fills no slot
        # resume: the same invocation issues ZERO requests and leaves the store as it was
        before = (cell_dir / "manifest.json").read_text()
        srun.main(argv)
        assert len(lb.chat_requests()) == 1
        after = json.loads((cell_dir / "manifest.json").read_text())
        assert after[C.MANIFEST_KEY_ITEM_SUBSET] == block and after["harness_config_hash"] == json.loads(before)["harness_config_hash"]
        # an explicit `--thinking not_available` resumes the SAME store — the frozen matrix line
        # (`--thinking off`) and that spelling hash ONE identity, so neither re-generates the other's store
        argv_pra = list(argv)
        argv_pra[argv_pra.index(C.FLAG_THINKING) + 1] = C.THINKING_NOT_AVAILABLE
        srun.main(argv_pra)
        assert len(lb.chat_requests()) == 1
        after_pra = json.loads((cell_dir / "manifest.json").read_text())
        assert after_pra["harness_config_hash"] == json.loads(before)["harness_config_hash"]
        assert after_pra["resolved_config"][C.MANIFEST_KEY_TINYCAP] == after["resolved_config"][C.MANIFEST_KEY_TINYCAP]
    finally:
        lb.close()
