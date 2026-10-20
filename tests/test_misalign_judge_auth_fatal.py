"""The misalign judge and a REJECTED CREDENTIAL (401 authentication_error / 403 permission_error).

The demo smoke's 20 agentic runs: the daemon dispatched misalign_judge on the revoked lowprio key; it claimed every run, issued one doomed
request per item (10,000 calls, each recorded as an item error), printed `N judged`, then filed a 0-verdict `partial` judgment per run. Now
(a) ONE minimal request on the pool runs BEFORE any claim -- a rejected credential exits RC_CREDENTIAL_REJECTED with nothing claimed,
nothing filed, nothing judged; (b) a credential rejected MID-RUN latches on the PoolClient (no wire call after the first), stops the loop,
releases the claims and files no judgment -- never a 0-verdict partial, never 10,000 error rows; (c) a 0-verdict filing (every row
judge-blocked) files its judgment with the counts and NO run_metric rows (n=0 / NULL rates are not allowed); a filing with verdicts carries
both metrics as before. Driven end to end through `main` on a synthetic registry with a stubbed pool client and a stubbed per-item judge:
no API call, no key (the placeholders are not credentials), synthetic ids, hostname-free, no transcript text."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "src", REPO / "scripts" / "sdfbase" / "full22" / "misalign"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_judge():
    spec = importlib.util.spec_from_file_location("misalign_judge_auth_fatal_under_test",
                                                  REPO / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------------------------
# the synthetic registry (test_misalign_judge_filing_counts' shape): one agentic run over one cell
# --------------------------------------------------------------------------------------------

EVALS = {"agentic_leaking": ("agentic", "greedy")}


def _instrument(eval_id: str) -> dict:
    cat, dec = EVALS[eval_id]
    return synth.instrument_fields(eval_id=eval_id, item_set_id=f"{eval_id}_set", decode_source=dec)


def _dims() -> list:
    dims = synth.seed_dimensions()
    i = 50
    for eval_id, (cat, dec) in EVALS.items():
        dims.append(synth.rec("eval", {"eval_id": eval_id, "eval_category": cat, "headline_metric_id": "asr",
                                       "metric_direction": "lower_is_better", "attacker_required": 0, "paper_status": "in_paper"}, synth.t(i)))
        dims.append(synth.rec("item_set", {"item_set_id": f"{eval_id}_set", "eval_id": eval_id, "dataset_version": f"synthetic/{eval_id}@v1",
                                           "n_items": 200, "items_sha256": synth.SHA256_B, "source_uri": f"synthetic://items/{eval_id}.jsonl",
                                           "item_id_scheme": "q<index>", "subsample_rule": "full"}, synth.t(i + 1)))
        dims.append(synth.rec("instrument", _instrument(eval_id), synth.t(i + 2)))
        i += 3
    return dims


def _run_recs(key: str, i0: int, eval_id: str, store: str, n_items: int) -> tuple:
    inst = _instrument(eval_id)
    p = synth.run_payload(key, "none", "none", i0, eval_id=eval_id, item_set_id=f"{eval_id}_set", instrument_id=inst["instrument_id"],
                          decode_source=EVALS[eval_id][1], store_uri=store)
    p["run_id"] = I.run_id(synth.CAMPAIGN, key, "none", eval_id, str(inst["instrument_id"]))
    rid = str(p["run_id"])
    t = synth.t
    recs = [synth.rec("run", p, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log"}, t(i0 + 1)),
            synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                    "evidence_uri": store}, t(i0 + 3)),
            synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": n_items}}, t(i0 + 4))]
    return rid, recs


def _row(item, behavior, sample_idx=0, **more):
    return {"item_id": item, "behavior": behavior, "sample_idx": sample_idx, "response": "synthetic reply", **more}


def _world(base: Path, monkeypatch, *, n: int = 10) -> tuple:
    """A results root with one agentic cell of ``n`` leaking rows, the served-identity snapshot the preflight reads, and a registry holding
    the run (generated, pending for the judge). Returns ``(results, cell, root, run_id)``."""
    results = base / "results" / "synthtarget_s1"
    cell = results / "agentic" / "cellA"
    cell.mkdir(parents=True)
    (results / "endpoint.snapshot.json").write_text(json.dumps({"model": "org/model", "model_revision": "a" * 40}))
    rows = [_row(f"L{i}", "leaking", family="agentic", cell_id="cellA") for i in range(n)]
    (cell / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    root = str(base / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims())
    rid, recs = _run_recs("k_leak", 200, "agentic_leaking", str(cell), n)
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    return results, cell, root, rid


def _fold(root):
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    assert rep.refused == 0, rep.as_dict()


def _env(monkeypatch):
    # the Judge constructs its own (never-called) client against the env placeholders; they are not credentials
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "placeholder-not-a-key")
    monkeypatch.delenv("SISTEER_SECRETS_MODE", raising=False)


def _argv(results, root, *more):
    return ["--results-root", str(results), "--families", "agentic", "--worklist", "registry", "--registry-root", root, "--skip-rows",
            "--concurrency", "2", *more]


def _auth_error(status: int = 401):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if status == 401:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        return anthropic.AuthenticationError(f"Error code: 401 - {body}", response=httpx.Response(401, request=req), body=body)
    body = {"type": "error", "error": {"type": "permission_error", "message": "synthetic"}}
    return anthropic.PermissionDeniedError(f"Error code: 403 - {body}", response=httpx.Response(403, request=req), body=body)


class _FakeInner:
    """The vault's client stand-in behind PoolClient: ``messages.create`` (the preflight) and ``messages.parse`` (the Judge's one wire call)
    each yield their scripted outcomes in order (an Exception raises; the last outcome repeats), counted under a lock (the judge loop is
    concurrent). Carries the keyless federated stamp the vault puts on a wif client."""

    def __init__(self, create_outcomes, parse_outcomes):
        self.calls = {"create": 0, "parse": 0}
        self.lock = threading.Lock()
        self.messages = self
        self.sisteer_pool_stamp = {"pool": "lowprio", "auth": "wif", "ids_sha12": "0123456789ab"}
        self._create, self._parse = list(create_outcomes), list(parse_outcomes)

    def _next(self, seq, name):
        with self.lock:
            self.calls[name] += 1
            o = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(o, Exception):
            raise o
        return o

    def create(self, **kw):
        return self._next(self._create, "create")

    def parse(self, **kw):
        return self._next(self._parse, "parse")


def _judged_row(row, tags, client, *, blocked: bool = False) -> dict:
    out = {**row, "judge_tag": tags["leaking"], "judge_model": "synthetic-judge", "judge_pool": client.pool, "judged_at": "t", "structural_gate": 0}
    if blocked:
        out.update(judge_blocked=True, judge_blocked_reason="refusal:bio", classifier_harmful=None, harmful=None)
    else:
        out.update(judge_blocked=False, classifier_harmful=0, harmful=0)
    return out


def _judgments(root, rid) -> list:
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        js = [tuple(r) for r in conn.execute("SELECT status, n_items_judged, n_judge_refusals, n_errors FROM judgment WHERE run_id = ?", (rid,))]
        n_metric = conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ?", (rid,)).fetchone()[0]
        n_items = conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ?", (rid,)).fetchone()[0]
        pend = [r for r in P.pending_for(conn, synth.JUDGE_ID) if r["run_id"] == rid]
        life = conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0]
    finally:
        conn.close()
    return [js, n_metric, n_items, len(pend), life]


# --------------------------------------------------------------------------------------------
# (a) the preflight: a rejected credential claims nothing
# --------------------------------------------------------------------------------------------


def test_a_credential_rejected_at_the_preflight_claims_nothing_files_nothing_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    MJ = _load_judge()
    _env(monkeypatch)
    results, cell, root, rid = _world(tmp_path, monkeypatch)
    inner = _FakeInner([_auth_error(401)], [RuntimeError("the judge must never be called after a rejected preflight")])
    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: inner)
    monkeypatch.setattr(MJ, "judge_agentic", lambda *a, **k: pytest.fail("an item was judged after the preflight was rejected"))
    assert MJ.RC_CREDENTIAL_REJECTED == 3 and MJ.RC_CREDENTIAL_REJECTED != 1
    rc = MJ.main(_argv(results, root, "--session", "s-A"))
    assert rc == MJ.RC_CREDENTIAL_REJECTED
    assert inner.calls == {"create": 1, "parse": 0}                      # ONE minimal request, then nothing
    js, n_metric, n_items, n_pending, life = _judgments(root, rid)
    assert js == [] and n_metric == 0 and n_items == 0 and n_pending == 1 and life == "generated"   # nothing claimed: no judgment row at all
    assert not (results / "judge_provenance.json").exists() and not (cell / "judged.jsonl").exists()
    out = capsys.readouterr().out
    assert "CREDENTIAL REJECTED -- nothing claimed, nothing filed, nothing judged" in out
    assert "AuthenticationError" in out and "401" in out and "authentication_error" in out and "exit 3" in out and "10 item(s) NOT requested" in out
    assert "auth wif ids sha12 0123456789ab" in out                       # the pool stamp, keyless
    assert "placeholder-not-a-key" not in out and "x-api-key" not in out and "synthetic reply" not in out
    # a 403 is the same gate
    inner403 = _FakeInner([_auth_error(403)], [RuntimeError("never")])
    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: inner403)
    assert MJ.main(_argv(results, root, "--session", "s-A2")) == MJ.RC_CREDENTIAL_REJECTED
    assert inner403.calls == {"create": 1, "parse": 0} and _judgments(root, rid)[0] == []
    assert "PermissionDeniedError" in capsys.readouterr().out
    # any other preflight answer is not this gate's business: a 429 proceeds (the Judge's own loop throttles) -- the run then judges
    inner429 = _FakeInner([anthropic.RateLimitError("Error code: 429", response=httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")), body=None)],
                          [SimpleNamespace()])
    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: inner429)
    monkeypatch.setattr(MJ, "judge_agentic", lambda judge, client, row, scenarios, tags, schema, classify: _judged_row(row, tags, client))
    assert MJ.main(_argv(results, root, "--session", "s-A3")) == 0
    assert inner429.calls["create"] == 1
    _fold(root)
    assert _judgments(root, rid)[0] == [("complete", 10, 0, 0)]
    prov = json.loads((results / "judge_provenance.json").read_text())
    assert prov["preflight"] == {"ok": False, "error": "RateLimitError", "status": 429} and prov["credential_rejected"] is None
    assert "preflight: pool lowprio answered (RateLimitError); credential accepted" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# (b) mid-run: the first 401 stops the loop; the claims go back; no judgment is filed
# --------------------------------------------------------------------------------------------


def test_a_credential_rejected_mid_run_stops_the_loop_releases_the_claims_and_files_no_judgment(tmp_path, monkeypatch, capsys):
    MJ = _load_judge()
    _env(monkeypatch)
    results, cell, root, rid = _world(tmp_path, monkeypatch, n=10)
    ok = SimpleNamespace()
    inner = _FakeInner([ok], [ok, ok, _auth_error(401)])        # the preflight passes; the Judge's third call is rejected, and every later one
    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: inner)

    def fake_judge_agentic(judge, client, row, scenarios, tags, schema, classify):
        client.messages.parse(model="synthetic-judge", max_tokens=1, messages=[])     # the Judge's ONE wire call, through the PoolClient
        return _judged_row(row, tags, client)

    monkeypatch.setattr(MJ, "judge_agentic", fake_judge_agentic)
    rc = MJ.main(_argv(results, root, "--session", "s-B"))
    assert rc == MJ.RC_CREDENTIAL_REJECTED
    assert inner.calls["create"] == 1 and 3 <= inner.calls["parse"] <= 4, inner.calls     # 2 verdicts + the rejection (+ one sibling in flight), never 10
    judged = [json.loads(ln) for ln in (cell / "judged.jsonl").read_text().splitlines() if ln.strip()]
    assert len(judged) == 2 and all(r["judge_blocked"] is False for r in judged)       # the verdicts before the stop stay (a re-run resumes them)
    js, n_metric, n_items, n_pending, life = _judgments(root, rid)
    assert js == [("abandoned", None, None, None)] or js == [("abandoned", 0, 0, 0)], js   # the claim released: NEVER complete / partial
    assert n_metric == 0 and n_items == 0 and n_pending == 1 and life == "generated"
    prov = json.loads((results / "judge_provenance.json").read_text())
    cr = prov["credential_rejected"]
    assert cr["pool"] == "lowprio" and cr["status"] == 401 and cr["error_type"] == "AuthenticationError" and cr["api_error_type"] == "authentication_error"
    assert cr["filed"] is False and cr["claims_released"] == 1 and cr["n_not_requested"] >= 6 and "401" in cr["message"]
    assert prov["counts"] == {"judged": 2, "blocked": 0, "errors": 0} and prov["errors"] == []    # not one error row per item
    assert prov["preflight"]["ok"] is True
    ev = [e for e in prov["pool_events"] if e["event"] == "credential_rejected"]
    assert len(ev) == 1 and {k: v for k, v in ev[0].items() if k != "t"} == {           # ONE event, however many workers saw the latch
        "event": "credential_rejected", "pool": "lowprio", "status": 401, "error_type": "AuthenticationError",
        "api_error_type": "authentication_error", "n_429": 0, "n_529": 0, "n_ok": 3}      # n_ok: the preflight + the two verdicts
    assert prov["api_pool"] == {"pool": "lowprio", "auth": "wif", "ids_sha12": "0123456789ab"}
    out = capsys.readouterr().out
    assert "[pool] CREDENTIAL REJECTED on lowprio" in out and "CREDENTIAL REJECTED after 2 of 10 item(s)" in out
    assert "1 claim(s) released; NO judgment filed" in out and "CREDENTIAL REJECTED -- run stopped, nothing filed" in out and "exit 3" in out
    assert "-- verdicts" not in out and "no run_metric rows filed" not in out                          # no filing line at all: scope.complete never ran
    assert "placeholder-not-a-key" not in out and "synthetic reply" not in out
    # the credential fixed: the next dispatch resumes the two verdicts and completes
    inner_ok = _FakeInner([ok], [ok])
    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: inner_ok)
    assert MJ.main(_argv(results, root, "--session", "s-B2")) == 0
    assert inner_ok.calls["parse"] == 8
    _fold(root)
    js2, n_metric2, _, n_pending2, life2 = _judgments(root, rid)
    assert ("complete", 10, 0, 0) in js2 and {s for s, *_ in js2} <= {"abandoned", "complete"}, js2    # never a partial
    assert n_metric2 == 2 and n_pending2 == 0 and life2 == "judged"


# --------------------------------------------------------------------------------------------
# (c) 3a: a 0-verdict filing files its judgment and NO run_metric rows; a filing with verdicts carries both metrics
# --------------------------------------------------------------------------------------------


def test_a_zero_verdict_filing_carries_no_run_metric_rows_and_a_verdict_filing_carries_both(tmp_path, monkeypatch, capsys):
    MJ = _load_judge()
    _env(monkeypatch)
    results, cell, root, rid = _world(tmp_path / "blocked", monkeypatch, n=6)
    inner = _FakeInner([SimpleNamespace()], [SimpleNamespace()])
    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: inner)
    monkeypatch.setattr(MJ, "judge_agentic", lambda judge, client, row, scenarios, tags, schema, classify: _judged_row(row, tags, client, blocked=True))
    assert MJ.main(_argv(results, root, "--session", "s-C")) == 0
    _fold(root)
    js, n_metric, n_items, n_pending, life = _judgments(root, rid)
    assert js == [("complete", 0, 6, 0)]                 # the judgment IS filed (a 0-verdict `judged` is intended: every row a counted abstention)
    assert n_metric == 0                                 # no misaligned_rate / misaligned_rate_graded row over nothing
    assert n_items == 12 and n_pending == 0 and life == "judged"
    out = capsys.readouterr().out
    assert "0 verdicts -- no run_metric rows filed (D634(a)" in out and "complete -- verdicts 0, judge-blocked 6, errored 0 of 6 planned" in out
    # the control: one verdict among the blocked rows files both metrics, as before
    results2, cell2, root2, rid2 = _world(tmp_path / "mixed", monkeypatch, n=6)
    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: _FakeInner([SimpleNamespace()], [SimpleNamespace()]))
    monkeypatch.setattr(MJ, "judge_agentic",
                        lambda judge, client, row, scenarios, tags, schema, classify: _judged_row(row, tags, client, blocked=row["item_id"] != "L0"))
    assert MJ.main(_argv(results2, root2, "--session", "s-C2")) == 0
    _fold(root2)
    conn = DB.connect(DB.db_path(root2), readonly=True)
    m = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT metric_id, n, n_abstentions FROM run_metric WHERE run_id = ?", (rid2,))}
    js2 = [tuple(r) for r in conn.execute("SELECT status, n_items_judged, n_judge_refusals FROM judgment WHERE run_id = ?", (rid2,))]
    conn.close()
    assert js2 == [("complete", 1, 5)] and m == {"misaligned_rate": (6, 5), "misaligned_rate_graded": (1, 5)}   # Both conventions
    assert "no run_metric rows filed" not in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# (b') more queued items than workers -- the run RETURNS; nothing waits on a future cancelled off the queue
# --------------------------------------------------------------------------------------------


def test_a_mid_run_rejection_with_more_queued_items_than_workers_returns_and_never_waits_on_a_cancelled_future(tmp_path, monkeypatch, capsys):
    """Round 1 on 7778694d0: `concurrent.futures.as_completed` is woken for a future only when it finishes or is cancelled AT RUN TIME
    (CANCELLED_AND_NOTIFIED); a future cancelled off the executor's queue by `shutdown(cancel_futures=True)` never runs, so the iterator
    waited on it forever -- the claims never released, the provenance never written, the daemon never handed the family's hold. Here the two
    workers are both slow to REPORT their rejection (the time between the wire call and the raise), so the main thread's cancel is what
    empties the queue: 24 items, concurrency 2, two verdicts, two rejections, twenty items cancelled off the queue; the run returns, rc 3,
    the claims released, the twenty counted as never requested. (The test runs main in a thread with a bound: before the fix it hangs.)"""
    MJ = _load_judge()
    _env(monkeypatch)
    results, cell, root, rid = _world(tmp_path, monkeypatch, n=24)
    ok = SimpleNamespace()
    inner = _FakeInner([ok], [ok, ok, _auth_error(401)])        # the preflight passes; the third wire call and every later one are rejected
    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: inner)

    def slow_to_report_judge_agentic(judge, client, row, scenarios, tags, schema, classify):
        try:
            client.messages.parse(model="synthetic-judge", max_tokens=1, messages=[])
        except MJ.vault.JudgeCredentialRejected:
            time.sleep(0.4)                                      # the worker's time between the wire call and its raise: the queue is still full
            raise
        return _judged_row(row, tags, client)

    monkeypatch.setattr(MJ, "judge_agentic", slow_to_report_judge_agentic)
    outcome = {}
    t = threading.Thread(target=lambda: outcome.update(rc=MJ.main(_argv(results, root, "--session", "s-B3"))), daemon=True)
    t.start(); t.join(timeout=60)
    assert not t.is_alive(), "main did not return: the judge loop waited on futures cancelled off the executor's queue (round-1 finding)"
    assert outcome["rc"] == MJ.RC_CREDENTIAL_REJECTED
    assert inner.calls["create"] == 1 and 3 <= inner.calls["parse"] <= 4, inner.calls       # two verdicts + the rejection (+ the sibling in flight), never 24
    judged = [json.loads(ln) for ln in (cell / "judged.jsonl").read_text().splitlines() if ln.strip()]
    assert len(judged) == 2
    js, n_metric, n_items, n_pending, life = _judgments(root, rid)
    assert js in ([("abandoned", None, None, None)], [("abandoned", 0, 0, 0)]) and n_metric == 0 and n_items == 0 and n_pending == 1, js
    prov = json.loads((results / "judge_provenance.json").read_text())
    cr = prov["credential_rejected"]
    assert cr["status"] == 401 and cr["filed"] is False and cr["claims_released"] == 1 and cr["n_not_requested"] == 22, cr    # 24 - 2 verdicts
    assert prov["counts"] == {"judged": 2, "blocked": 0, "errors": 0} and prov["errors"] == []
    out = capsys.readouterr().out
    assert "CREDENTIAL REJECTED after 2 of 24 item(s)" in out and "1 claim(s) released; NO judgment filed" in out and "exit 3" in out
