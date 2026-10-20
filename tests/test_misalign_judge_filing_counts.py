"""The misalign judge's registry filing arithmetic.

The s1 control-LoRA leaking run filed `n_items_judged = 509` for 502 verdicts + 7 judge-blocked rows (`mine` held the blocked rows), so
the coverage view read 509 + 7 = 516 >= 510 and the run's one errored item was invisible to the contract; its judgment read `n_errors 0`
although the grader printed `errors 1` (the scope's count was never set). Now `run_filing_counts` is the ONE arithmetic of a registered
run's judgment record: n_items_judged = the rows WITH A VERDICT, n_judge_refusals = the judge-blocked rows, n_errors = the grader's own
errored items of THAT run (matched to its behaviour through the responses), and the status is `complete` iff verdicts + refusals reach
the run's planned rows and nothing errored, else `partial`. The judged rows of a blocked verdict carry the judge's stop as
`judge_blocked_reason`. Driven end to end through `main` on a synthetic store with a stubbed judge (no API call): the first
dispatch errors one item -> partial (7 / 2 / 1); the re-dispatch judges only that item -> complete (8 / 2 / 0) and the partial reads
`superseded`. Synthetic ids, hostname-free, no transcript text."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
    spec = importlib.util.spec_from_file_location("misalign_judge_filing_counts_under_test", REPO / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------------------------
# a synthetic registry: the agentic eval the judge grades (the drivers' tests' shape)
# --------------------------------------------------------------------------------------------

EVALS = {"agentic_leaking": ("agentic", "greedy"), "agentic_murder": ("agentic", "greedy"), "em_risky_financial": ("misalignment", "generation_config")}


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


def _registry(tmp_path, monkeypatch, runs) -> tuple:
    """``runs`` = [(key, eval_id, store_path, n_items_generated)] -> (root, [run_id])."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims())
    rids = []
    for k, (key, eval_id, store, n) in enumerate(runs):
        rid, recs = _run_recs(key, 200 + 10 * k, eval_id, store, n)
        R.write_records(os.path.join(root, "records", "c", f"{k + 1:06d}-run.jsonl"), recs)
        rids.append(rid)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    return root, rids


def _fold(root):
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    assert rep.refused == 0, rep.as_dict()


def _row(item, behavior, sample_idx=0, **more):
    return {"item_id": item, "behavior": behavior, "sample_idx": sample_idx, "response": "synthetic reply", **more}


# --------------------------------------------------------------------------------------------
# the pure arithmetic
# --------------------------------------------------------------------------------------------


def test_run_filing_counts_split_verdicts_refusals_and_errors_per_run_and_decide_the_status():
    MJ = _load_judge()
    leak = {"run_id": "r-leak", "eval_id": "agentic_leaking"}
    murder = {"run_id": "r-murder", "eval_id": "agentic_murder"}
    responses = [_row(f"L{i}", "leaking") for i in range(6)] + [_row(f"M{i}", "murder") for i in range(4)]
    judged = ([_row(f"L{i}", "leaking", judge_blocked=False, harmful=0) for i in range(4)]
              + [_row("L4", "leaking", judge_blocked=True, harmful=None, judge_blocked_reason="refusal:bio")]
              + [_row(f"M{i}", "murder", judge_blocked=False, harmful=1) for i in range(3)])
    errors = [{"family": "agentic", "cell_id": "cellA", "item_id": "L5", "sample_idx": 0, "error": "RuntimeError: synthetic"},
              {"family": "agentic", "cell_id": "cellA", "item_id": "M3", "sample_idx": 0, "error": "RuntimeError: synthetic"},
              {"family": "agentic", "cell_id": "cellOTHER", "item_id": "L0", "sample_idx": 0, "error": "RuntimeError: another cell's"}]
    fc = MJ.run_filing_counts(leak, judged, responses, errors, "agentic", "cellA", len(responses))
    assert (fc["n_planned"], fc["n_verdicts"], fc["n_blocked"], fc["n_errors"], fc["status"]) == (6, 4, 1, 1, "partial")
    assert [r["item_id"] for r in fc["mine"]] == ["L0", "L1", "L2", "L3", "L4"]          # the run's rows of ITS behaviour, blocked included
    assert fc["n_verdicts"] + fc["n_blocked"] == len(fc["mine"]) and fc["n_verdicts"] + fc["n_blocked"] + fc["n_errors"] == fc["n_planned"]
    fc_m = MJ.run_filing_counts(murder, judged, responses, errors, "agentic", "cellA", len(responses))
    assert (fc_m["n_planned"], fc_m["n_verdicts"], fc_m["n_blocked"], fc_m["n_errors"], fc_m["status"]) == (4, 3, 0, 1, "partial")
    # every item judged or blocked, nothing errored: complete; a refusal is a counted outcome, never a missing item
    judged_all = judged + [_row("L5", "leaking", judge_blocked=True, harmful=None), _row("M3", "murder", judge_blocked=False, harmful=0)]
    fc2 = MJ.run_filing_counts(leak, judged_all, responses, [errors[2]], "agentic", "cellA", len(responses))
    assert (fc2["n_verdicts"], fc2["n_blocked"], fc2["n_errors"], fc2["status"]) == (4, 2, 0, "complete")
    assert MJ.run_filing_counts(murder, judged_all, responses, [], "agentic", "cellA", len(responses))["status"] == "complete"
    # a whole-cell run (em8q): every row is its own; the cell's errors are its errors
    em = {"run_id": "r-em", "eval_id": "em_risky_financial"}
    em_resp = [_row(f"E{i}", None, sample_idx=i % 3) for i in range(9)]
    em_judged = [_row(f"E{i}", None, sample_idx=i % 3, judge_blocked=(i in (2, 7)), em=None if i in (2, 7) else 0) for i in range(8)]
    em_err = [{"family": "em8q", "cell_id": "cellB", "item_id": "E8", "sample_idx": 2, "error": "x"}]
    fc_em = MJ.run_filing_counts(em, em_judged, em_resp, em_err, "em8q", "cellB", len(em_resp))
    assert (fc_em["n_planned"], fc_em["n_verdicts"], fc_em["n_blocked"], fc_em["n_errors"], fc_em["status"]) == (9, 6, 2, 1, "partial")
    assert MJ.run_filing_counts(em, em_judged + [_row("E8", None, sample_idx=2, judge_blocked=False, em=1)], em_resp, [], "em8q", "cellB", 9)["status"] == "complete"


def test_the_judge_functions_carry_the_blocked_reason_onto_the_judged_row():
    MJ = _load_judge()
    import judge_rubrics as jr
    from sisteer.capeval.misalign import classify
    from sisteer.eval.agentic_misalignment.loader import load_scenarios_v2
    scenarios = {r["scenario_id"]: r for r in load_scenarios_v2()}
    sid = next(iter(scenarios))
    judge = SimpleNamespace(cache={}, _key=lambda *parts: "k", structured=lambda tag, key_parts, prompt, schema: {"__blocked__": True, "blocked_reason": "refusal:bio"})
    client = SimpleNamespace(pool="lowprio")
    out = MJ.judge_agentic(judge, client, _row(sid, scenarios[sid]["behavior"]), scenarios, {"leaking": jr.AM_LEAK_TAG, "murder": jr.AM_MURDER_TAG},
                           jr.HarmfulVerdict, classify)
    assert out["judge_blocked"] is True and out["judge_blocked_reason"] == "refusal:bio" and out["harmful"] is None
    em = MJ.judge_em8q(judge, client, _row("quick_buck", None), {"quick_buck": "synthetic question"}, jr.EM8Q_TAG, jr.EMAlignment, "{question} {answer}")
    assert em["judge_blocked"] is True and em["judge_blocked_reason"] == "refusal:bio" and em["alignment"] is None
    # a verdict carries no reason key
    judge_ok = SimpleNamespace(cache={}, _key=lambda *parts: "k", structured=lambda *a, **k: {"harmful": False, "reasoning": "fine"})
    out_ok = MJ.judge_agentic(judge_ok, client, _row(sid, scenarios[sid]["behavior"]), scenarios, {"leaking": jr.AM_LEAK_TAG, "murder": jr.AM_MURDER_TAG},
                              jr.HarmfulVerdict, classify)
    assert out_ok["judge_blocked"] is False and "judge_blocked_reason" not in out_ok


# --------------------------------------------------------------------------------------------
# end to end through main: dispatch -> partial (an errored item) -> re-dispatch -> complete, the partial superseded
# --------------------------------------------------------------------------------------------


def test_main_files_verdicts_refusals_and_errors_apart_then_the_rejudge_completes_and_supersedes_the_partial(tmp_path, monkeypatch, capsys):
    MJ = _load_judge()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")             # the Judge / PoolClient construct against the env placeholder;
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "placeholder-not-a-key")     # the stubbed judge below makes no call
    results = tmp_path / "results" / "synthtarget_s1"
    cell = results / "agentic" / "cellA"
    cell.mkdir(parents=True)
    (results / "endpoint.snapshot.json").write_text(json.dumps({"model": "org/model", "model_revision": "a" * 40}))   # the served identity the preflight reads
    rows = [_row(f"L{i}", "leaking", family="agentic", cell_id="cellA") for i in range(10)]
    (cell / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    root, (rid,) = _registry(tmp_path, monkeypatch, [("k_leak", "agentic_leaking", str(cell), 10)])
    state = {"fail_once": True, "calls": []}

    def fake_judge_agentic(judge, client, row, scenarios, tags, schema, classify):
        state["calls"].append(row["item_id"])
        if row["item_id"] == "L5" and state["fail_once"]:
            state["fail_once"] = False
            raise RuntimeError("synthetic judge failure")
        out = {**row, "judge_tag": tags["leaking"], "judge_model": "synthetic-judge", "judge_pool": client.pool, "judged_at": "t", "structural_gate": 0}
        if row["item_id"] in ("L8", "L9"):
            out.update(judge_blocked=True, judge_blocked_reason="refusal:bio", classifier_harmful=None, harmful=None)
        else:
            out.update(judge_blocked=False, classifier_harmful=0, harmful=0)
        return out

    monkeypatch.setattr(MJ, "judge_agentic", fake_judge_agentic)
    # main now sends ONE minimal preflight request on the pool before any claim (misalign_judge.PoolClient.preflight); the pool client
    # is a stand-in here, so the preflight is answered locally and nothing reaches the network (the stubbed judge above makes no call either)
    preflight = {"n": 0}

    def _answer(**kw):
        preflight["n"] += 1
        return SimpleNamespace()

    monkeypatch.setattr(MJ, "pool_client", lambda pool, *, base_url: SimpleNamespace(
        messages=SimpleNamespace(create=_answer, parse=lambda **kw: pytest.fail("the stubbed judge must make no wire call")),
        sisteer_pool_stamp={"pool": pool, "key_sha256_12": "0" * 12}))
    argv = ["--results-root", str(results), "--families", "agentic", "--worklist", "registry", "--registry-root", root, "--skip-rows", "--concurrency", "2"]
    # dispatch 1: one item errors -> partial, the counts apart
    assert MJ.main(argv + ["--session", "s-A"]) == 1
    assert preflight["n"] == 1                                                                                 # one preflight, before the claim
    _fold(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    js = [tuple(r) for r in conn.execute("SELECT status, n_items_judged, n_judge_refusals, n_errors, claimed_by_session FROM judgment WHERE run_id = ?", (rid,))]
    assert js == [("partial", 7, 2, 1, "s-A")]
    partial_id = conn.execute("SELECT judgment_id FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0]
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND judgment_id = ?", (rid, partial_id)).fetchone()[0] == 18   # 9 judged rows x 2 metrics
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND label = 'judge_blocked'", (rid,)).fetchone()[0] == 4            # 2 blocked rows x 2 metrics
    m = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT metric_id, n, n_abstentions FROM run_metric WHERE run_id = ? AND judgment_id = ?", (rid, partial_id))}
    assert m == {"misaligned_rate": (9, 2), "misaligned_rate_graded": (7, 2)}                                 # Both conventions, unchanged
    pend = [r for r in P.pending_for(conn, synth.JUDGE_ID) if r["run_id"] == rid]
    assert len(pend) == 1 and pend[0]["pending_reason"] == "partial"                                          # 7 + 2 + 1 = 10 >= 10 but partial: the run pends
    conn.close()
    judged = [json.loads(l) for l in (cell / "judged.jsonl").read_text().splitlines() if l.strip()]
    assert len(judged) == 9 and {r["item_id"] for r in judged if r.get("judge_blocked")} == {"L8", "L9"}
    assert all(r.get("judge_blocked_reason") == "refusal:bio" for r in judged if r.get("judge_blocked"))
    prov = json.loads((results / "judge_provenance.json").read_text())
    assert prov["counts"] == {"judged": 9, "blocked": 2, "errors": 1} and [e["item_id"] for e in prov["errors"]] == ["L5"]
    assert prov["truncation"]["ladder"] == [1500, 4000, 8000] and prov["truncation"]["blocked_by_stop"] == 0
    out = capsys.readouterr().out
    assert "partial -- verdicts 7, judge-blocked 2, errored 1 of 10 planned" in out
    # dispatch 2 (another session): only the errored item is re-requested; complete, and the partial is superseded
    assert MJ.main(argv + ["--session", "s-A2"]) == 0
    assert state["calls"].count("L5") == 2 and len(state["calls"]) == 11
    _fold(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows_j = {r[0]: tuple(r[1:]) for r in conn.execute("SELECT judgment_id, status, n_items_judged, n_judge_refusals, n_errors, superseded_by_judgment_id "
                                                       "FROM judgment WHERE run_id = ?", (rid,))}
    complete_id = next(j for j, v in rows_j.items() if v[0] == "complete")
    assert rows_j[complete_id] == ("complete", 8, 2, 0, None)
    assert rows_j[partial_id] == ("superseded", 7, 2, 1, complete_id)
    assert [r for r in P.pending_for(conn, synth.JUDGE_ID) if r["run_id"] == rid] == []                         # covered: 8 + 2 >= 10
    assert {r[0] for r in conn.execute("SELECT DISTINCT judgment_id FROM v_item_result_current WHERE run_id = ?", (rid,))} == {complete_id}
    assert tuple(conn.execute("SELECT lifecycle_state, n_items_judged FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("judged", 8)
    conn.close()
    assert len([json.loads(l) for l in (cell / "judged.jsonl").read_text().splitlines() if l.strip()]) == 10
    assert "complete -- verdicts 8, judge-blocked 2, errored 0 of 10 planned" in capsys.readouterr().out
