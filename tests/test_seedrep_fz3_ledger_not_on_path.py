"""fz3 — the usage ledger is REQUIRED where the plan declares one and NOT ON THE PATH elsewhere; a ledger-less row is
reconciled over request evidence + response evidence + the engine log under a REVISED, QUALIFIED contract, recorded by name.

Known-bad at c8ae67b4c: a static (pasttense) root with request + response evidence and an engine log but NO `usage_ledger.jsonl`
— the only shape the Modal path produces on a non-GSM8K row (`RoutingProxy` writes no ledger; only `usage_ledger.py` per GSM8K
child) — refused `no 'usage_ledger.jsonl'` in `load_evidence`, so no non-GSM8K row could ever AGREE.

Green, in four requirements: (1) topology comes from the BOUND PLAN (the row's children's `ledger.ledger_path_rel`), the
plan builders' leg contract when none is bound, NEVER from file absence — a declared ledger that is missing refuses per declared
child path; (2) a ledger-less successful draw still needs the COMPLETE binding — record identity + labels → a 2xx response with a
REAL completion id → exactly one engine-log line — with the text-hash / seed / sampler / arm checks retained; (3) retries are
attempts, not completions — a 503 then a success binds ONCE, two genuine successes stay ambiguous, a contradictory or duplicate
completion record refuses (the fz2 ownership case); (4) the qualification rides on every verdict and the summary: `ledger: not_on_path`, the
evidence basis, `body_digest_check: not_available`, and the contract marker `seedrep_request_evidence_reconciliation/2-D106`.
A GSM8K root is unchanged (three sources required); a static root that unexpectedly HAS a ledger reads it (`present_undeclared`).
Synthetic roots only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

CONTRACT_LEDGERLESS = "seedrep_request_evidence_reconciliation/2-D106"
CONTRACT_LEDGER = "seedrep_request_evidence_reconciliation/1"


def _static_root(tmp_path: Path, *, ledger: bool, arms=("none",), n_items=3):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=arms)
    fx.populate_root(root, s=1, arms=arms, n_items=n_items)
    if not ledger:
        (root.path / ac.LEDGER_FILE).unlink()  # the Modal path's shape: no ledger process on a static row
    return root, ac.read_attempt_root(root.path)


def _audit(ar, tmp_path: Path, legs, *, plan=None, tag="a"):
    return rea.audit_root(ar, out_dir=tmp_path / f"audits_{tag}", legs=list(legs), policy_file=None,
                          thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=list(fx.PETRI_BEHAVIOURS),
                          model_key=ac.MODEL_KEY, client_status="absent", siblings=[], plan=plan)


def _plan_with(root: fx.Root, ar, children):
    """A bound plan whose row for this attempt carries `children` (the ledger topology the audit reads)."""
    row = {"standup_id": ar.production_standup_id, "leg": ar.leg, "view": ar.view, "stage": ar.stage, "shard": ar.shard, "children": children}
    return ac.LaunchPlan(path=root.path / "launch_plan.json", W=fx.W_FIXTURE, wave_uuid=fx.WAVE_UUID, rows={ar.production_standup_id: row})


# --------------------------------------------------------------------------- #
# known-bad → green                                                                                                  #
# --------------------------------------------------------------------------- #

def test_known_bad_static_root_without_ledger_refused_and_now_agrees_qualified(tmp_path: Path):
    root, ar = _static_root(tmp_path, ledger=False)
    # the c8ae67b4c refusal: the single-ledger rule on a root that never had one
    with pytest.raises(ac.AuditRefusal, match=r"no 'usage_ledger.jsonl'"):
        ac.RootReads().ledger_rows(ar)
    # the topology says not_on_path (leg contract, no plan bound); the attempt inventory is the response hook's
    assert ac.ledger_status(ar) == (ac.LEDGER_NOT_ON_PATH, "leg rule (no plan bound): no pasttense child declares a ledger path (only ['gsm8k'] do)")
    ev = rea.load_evidence(ar)
    assert ev.ledger == ac.LEDGER_NOT_ON_PATH and ev.attempt_source == "response_evidence" and ev.ledger_rows == []
    assert len(ev.attempt_rows) == 6 and set(ev.attempts_by_id) == set(ev.records_by_id) and ev.contract == CONTRACT_LEDGERLESS
    (v,) = _audit(ar, tmp_path, ["pasttense"])
    assert v["verdict"] == "AGREES" and v["problem"] is None and v["checks"]["reconciliation"] is None
    assert v["checks"]["row_binding"]["n_bound"] == v["n_draws"] == 6
    # (4) the qualification, by name, never a silent AGREES
    assert v["ledger"] == "not_on_path" and v["attempt_source"] == "response_evidence" and v["reconciliation_contract"] == CONTRACT_LEDGERLESS
    assert v["n_ledger_rows_partition"] is None and v["n_attempt_rows_partition"] == 6
    assert v["evidence_basis"] == ["request_evidence.jsonl", "response_evidence.jsonl", "engine log (Received request)"]
    assert v["body_digest_check"].startswith("not_available") and "D106" in v["independence_note"]
    assert v["ledger_source"].startswith("leg rule (no plan bound)")
    # the summary CSV carries it too (the optional column is blank by name, not 0)
    rows = [{c.rstrip("?"): v.get(c.rstrip("?")) for c in rea.SUMMARY_COLUMNS}]
    out = ac.write_csv(tmp_path / "s.csv", rows, rea.SUMMARY_COLUMNS)
    text = out.read_text()
    assert "not_on_path" in text and CONTRACT_LEDGERLESS in text and ",response_evidence," in text


def test_torn_completion_engine_log_lacks_the_line_refuses(tmp_path: Path):
    root, ar = _static_root(tmp_path, ledger=False)
    resp = ac.read_jsonl(root.path / ac.RESPONSE_EVIDENCE_FILE)
    victim = resp[0]["completion_id"]
    log = root.serve / "vllm_server.log"
    log.write_text("\n".join(l for l in log.read_text().splitlines() if victim not in l) + "\n")
    (v,) = _audit(ar, tmp_path, ["pasttense"])
    assert v["verdict"] == "REFUSED" and v["checks"]["row_binding"]["n_bound"] == 5
    assert f"completion_id {victim!r} matches 0 engine-log `Received request` line(s)" in v["problem"]
    assert v["ledger"] == "not_on_path"  # the qualification rides on a REFUSED verdict too


def test_ledger_less_binding_is_the_complete_one(tmp_path: Path):
    """(2): a response row with NO real completion id, a rejected response, and a record/label mismatch all fail the draw."""
    root, ar = _static_root(tmp_path, ledger=False)
    rp = root.path / ac.RESPONSE_EVIDENCE_FILE
    rows = ac.read_jsonl(rp)
    # (a) the attempt's response row says 500 / no completion: the filed row cannot come from it
    bad = dict(rows[0], status=500, completion_id=None, vmodel=None, finish_reason=None, text_sha256=None, text_sha256_of=None, error="HTTP 500")
    rp.write_text("".join(json.dumps(r) + "\n" for r in [bad] + rows[1:]))
    (v,) = _audit(ar, tmp_path, ["pasttense"], tag="a")
    assert v["verdict"] == "REFUSED" and "REJECTED response-evidence attempt (status 500" in v["problem"]
    assert v["checks"]["reconciliation"] is None  # the failed attempt (vmodel None) is still IN the partition: identity holds
    # (b) the response row names a completion the store row does not: torn binding
    torn = dict(rows[0], completion_id="chatcmpl-somethingelse")
    rp.write_text("".join(json.dumps(r) + "\n" for r in [torn] + rows[1:]))
    (v,) = _audit(ar, tmp_path, ["pasttense"], tag="b")
    assert v["verdict"] == "REFUSED" and "torn binding" in v["problem"]
    # (c) a request record missing for an attempt: the identity clause names it against the RESPONSE evidence
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for p in root.pipeline_files(ac.EVIDENCE_FILE):
        fx.rewrite_jsonl(p, lambda r: None if r["request_id"] == rows[0]["request_id"] else r)
    (v,) = _audit(ar, tmp_path, ["pasttense"], tag="c")
    assert v["verdict"] == "REFUSED" and "attempted call(s) in the response evidence have NO request-evidence record" in v["problem"]


# --------------------------------------------------------------------------- #
# (3) retries are attempts, not completions (fz2 HOLD A)                                                              #
# --------------------------------------------------------------------------- #

def _gsm8k_root(tmp_path: Path, *, idless: bool = False):
    root = fx.make_root(tmp_path / "stores", leg="gsm8k", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=2)
    if idless:
        # the Inspect family's REAL rows carry no request_id / completion_id (the writer's design): they bind through the response
        # hook's completed attempts for their (task, sample, vmodel) — the branch the retry / ambiguity rules live on
        from sisteer.capeval.tinycap import completion as _completion
        op = fx._tinycap_cell(root, "none") / C.TASK_OUTCOMES_DIR / _completion.marker_filename(ac.GSM8K_TASK)
        out = json.loads(op.read_text())
        out["items"] = [{k: v for k, v in it.items() if k not in ("request_id", "completion_id")} for it in out["items"]]
        op.write_text(json.dumps(out))
    return root, ac.read_attempt_root(root.path)


def _add_response_row(root: fx.Root, base: dict, **changes):
    (p,) = root.pipeline_files(ac.RESPONSE_EVIDENCE_FILE)
    p.write_text(p.read_text() + json.dumps({**base, **changes}) + "\n")


def test_failed_attempt_then_success_binds_once_two_successes_stay_ambiguous(tmp_path: Path):
    root, ar = _gsm8k_root(tmp_path, idless=True)
    (v,) = _audit(ar, tmp_path, ["gsm8k"], tag="0")
    assert v["verdict"] == "AGREES" and v["n_draws_bound"] == 2 and v["ledger"] == "declared"
    (rp,) = root.pipeline_files(ac.RESPONSE_EVIDENCE_FILE)
    rows = ac.read_jsonl(rp)
    # a 503 on a FRESH request id for item0, then the recorded success: the failed attempt is an attempt, never a completion
    failed = {**rows[0], "request_id": "f" * 32, "completion_id": None, "status": 503, "vmodel": None, "finish_reason": None,
              "text_sha256": None, "text_sha256_of": None, "error": "HTTP 503"}
    _add_response_row(root, failed)
    idx = ac.response_evidence_index(root.path)
    assert all(len(cands) == 1 for cands in idx.values()) and len(ac.completed_response_rows(root.path)) == 2
    assert len(ac.response_evidence_rows(root.path)) == 3  # the attempt inventory keeps it
    assert "None" not in {cid for cands in idx.values() for _, cid in cands}
    # ...but the request record of the retry is REQUIRED (identity clause): a record for it, marked probe-free, keeps the partition honest
    (ep,) = root.pipeline_files(ac.EVIDENCE_FILE)
    rec0 = next(r for r in ac.read_jsonl(ep) if r["request_id"] == rows[0]["request_id"])
    ep.write_text(ep.read_text() + json.dumps({**rec0, "request_id": "f" * 32}) + "\n")
    # ...and the ledger (declared on gsm8k) must carry the failed attempt too
    (lp,) = root.pipeline_files(ac.LEDGER_FILE)
    led0 = next(r for r in ac.read_jsonl(lp) if r["request_id"] == rows[0]["request_id"])
    lp.write_text(lp.read_text() + json.dumps({**led0, "request_id": "f" * 32, "status": 503, "rejected": True, "error_code": "http_503",
                                               "error_class": "server_error", "completion_id": None, "finish_reason": None}) + "\n")
    (v,) = _audit(ar, tmp_path, ["gsm8k"], tag="1")
    assert v["verdict"] == "AGREES" and v["n_draws_bound"] == 2, v["problem"]
    # two GENUINE successes for one (task, sample, vmodel): ambiguous, never resolved by guessing
    _add_response_row(root, rows[0], request_id="e" * 32, completion_id="chatcmpl-second-success")
    ep.write_text(ep.read_text() + json.dumps({**rec0, "request_id": "e" * 32}) + "\n")
    lp.write_text(lp.read_text() + json.dumps({**led0, "request_id": "e" * 32, "completion_id": "chatcmpl-second-success"}) + "\n")
    # the store reader refuses the root by name (never resolved by guessing) — the pre-fz2 behaviour pinned here
    with pytest.raises(ac.AuditRefusal, match=r"2 completed attempts recorded for \('inspect/gsm8k_paired200', 'item0/0', 'petri__none'\).*"
                                              r"the binding is ambiguous"):
        _audit(ar, tmp_path, ["gsm8k"], tag="2")


def test_contradictory_or_duplicate_completion_records_refuse(tmp_path: Path):
    root, ar = _gsm8k_root(tmp_path)
    (rp,) = root.pipeline_files(ac.RESPONSE_EVIDENCE_FILE)
    rows = ac.read_jsonl(rp)
    _add_response_row(root, rows[0], completion_id="chatcmpl-other")
    with pytest.raises(ac.AuditRefusal, match="two records of one attempt name two completions"):
        ac.completed_response_rows(root.path)
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _add_response_row(root, rows[0])
    with pytest.raises(ac.AuditRefusal, match="a duplicate completion record"):
        rea.load_evidence(ar)


# --------------------------------------------------------------------------- #
# (1) topology from the bound plan, never from file absence                                                          #
# --------------------------------------------------------------------------- #

def test_gsm8k_root_unchanged_three_sources_required(tmp_path: Path):
    root, ar = _gsm8k_root(tmp_path)
    assert ac.ledger_status(ar)[0] == ac.LEDGER_DECLARED
    (v,) = _audit(ar, tmp_path, ["gsm8k"])
    assert v["verdict"] == "AGREES" and v["ledger"] == "declared" and v["attempt_source"] == "ledger"
    assert v["reconciliation_contract"] == CONTRACT_LEDGER and v["n_ledger_rows_partition"] == 2
    assert v["evidence_basis"] == ["request_evidence.jsonl", "usage_ledger.jsonl", "response_evidence.jsonl", "engine log (Received request)"]
    assert v["body_digest_check"] == "usage ledger body_sha256 == client record body_sha256" and v["independence_note"] is None
    # the declared ledger missing REFUSES — it never becomes not_on_path
    for p in root.pipeline_files(ac.LEDGER_FILE):
        p.unlink()
    with pytest.raises(ac.AuditRefusal, match="no child 'usage_ledger.jsonl' under exp/"):
        rea.load_evidence(ar)


def test_bound_plan_declares_the_topology_and_a_missing_declared_ledger_refuses(tmp_path: Path):
    root, ar = _static_root(tmp_path, ledger=False)
    # a plan whose children declare no ledger: not_on_path, sourced to the plan row
    plan = _plan_with(root, ar, children=[{"eval_leg": "pasttense", "arm": "none", "s": 1}])
    status, source = ac.ledger_status(ar, plan)
    assert status == ac.LEDGER_NOT_ON_PATH and source == f"plan launch_plan.json row {ar.production_standup_id}: 0 of 1 children declare ledger.ledger_path_rel"
    (v,) = _audit(ar, tmp_path, ["pasttense"], plan=plan)
    assert v["verdict"] == "AGREES" and v["ledger"] == "not_on_path" and v["ledger_source"].startswith("plan launch_plan.json")
    # the SAME root under a plan that DECLARES a ledger for its child: the absent file refuses per declared path — never not_on_path
    plan2 = _plan_with(root, ar, children=[{"eval_leg": "pasttense", "arm": "none", "s": 1, "ledger": {"ledger_path_rel": "exp/x/usage_ledger.jsonl"}}])
    with pytest.raises(ac.AuditRefusal, match=r"1 declared child ledger\(s\) missing \(\['exp/x/usage_ledger.jsonl'\]\).*a declared ledger that is absent refuses"):
        rea.load_evidence(ar, plan=plan2)
    # a plan without this row, or a row without children: refused, never guessed
    with pytest.raises(ac.AuditRefusal, match="no row"):
        ac.ledger_status(ar, ac.LaunchPlan(path=root.path / "p.json", W=fx.W_FIXTURE, wave_uuid=fx.WAVE_UUID, rows={}))
    with pytest.raises(ac.AuditRefusal, match="carries no children list"):
        ac.ledger_status(ar, _plan_with(root, ar, children=None))


def test_static_root_that_has_a_ledger_reads_it_as_present_undeclared(tmp_path: Path):
    root, ar = _static_root(tmp_path, ledger=True)
    status, source = ac.ledger_status(ar)
    assert status == ac.LEDGER_PRESENT_UNDECLARED and "1 usage_ledger.jsonl present" in source
    (v,) = _audit(ar, tmp_path, ["pasttense"])
    assert v["verdict"] == "AGREES" and v["ledger"] == "present_undeclared" and v["attempt_source"] == "ledger"
    assert v["reconciliation_contract"] == CONTRACT_LEDGER and v["n_ledger_rows_partition"] == 6
    # ...and reconciled as before: a ledger row without its record still refuses through the ledger clause
    (lp,) = root.pipeline_files(ac.LEDGER_FILE)
    rows = ac.read_jsonl(lp)
    lp.write_text("".join(json.dumps(r) + "\n" for r in rows + [{**rows[0], "request_id": "deadbeef" * 4}]))
    (v,) = _audit(ar, tmp_path, ["pasttense"], tag="b")
    assert v["verdict"] == "REFUSED" and "attempted call(s) in the ledger have NO request-evidence record" in v["problem"]  # the original text


def test_reconciliation_engine_contract_pins():
    """The hunk lives INSIDE `contracts.request_evidence_reconciliation`: an unnamed inventory refuses; the response path skips
    exactly the body digest and nothing else."""
    assert C.request_evidence_reconciliation([], [], attempt_source="ledger") is not None  # (no records: the declaration gate speaks first)
    why = C.request_evidence_reconciliation([{"request_id": "r1", "kind": "task", "task": "static/pasttense", "sample": "g0/0",
                                              "request_evidence_version": C.REQUEST_EVIDENCE_VERSION, "vmodel": "m",
                                              "body_sha256": "x", "enable_thinking": False}], [], expected_enable_thinking=False,
                                             attempt_source="bogus")
    assert why is not None and "attempt_source 'bogus' is not 'ledger' or 'response_evidence'" in why
