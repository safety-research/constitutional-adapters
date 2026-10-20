"""The canonical IFEval battery — the ladder planner.

Pinned on synthetic stores: the planner writes ``cap_ladder.json`` (the adapter's record, verbatim) at ``ifeval_paths().cap_ladder``,
the ``rung_subset/2`` body at ``ifeval_paths(next).subset`` and ``ladder_state.json`` in the rung's root for a promoted cell; a
below-threshold cell files a terminal record and no subset; it is idempotent (a second run re-prints the same lines and writes
nothing new); a planned rung with a differing subset refuses by name; a running rung is never re-planned; one launcher
invocation per promoted cell with the rung flags, the subset path and the app name; a Nemotron rung names its certification;
the RE-ESTIMATE line (predicted GPU-h from the reconciled outcomes) precedes every launch line; at rung 2 the decision is over
the MERGED reading with denominator 1,082.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tinycap import ifeval_contracts as IC

import ifeval_synthetic_store as G

REPO = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LP = _load("tinycap_ifeval_ladder_plan_under_test", "scripts/tinycap_ifeval_ladder_plan.py")
ROWS = G.matrix_rows()
ESC = G.row_for(ROWS, "gemma4e4b", "main", "constv2-mix-full__lora_r4_ce")
NEMO = G.row_for(ROWS, "nemotron3nano4b", "main", "constv2-mix-full__lora_r4_ce")
BELOW = G.row_for(ROWS, "gemma4e4b", "main", "none")


def _classify(root, **kw):
    return G.classify(root, **kw)


def test_the_planner_writes_the_record_the_v2_subset_and_the_state_and_is_idempotent(tmp_path, capsys):
    G.build_store(tmp_path, ESC, at_cap={(i, t) for i in range(12) for t in (0, 1)}, loops={(0, 0)}, ladder=False)
    G.build_store(tmp_path, BELOW, at_cap={(i, 0) for i in range(5)}, ladder=False)
    G.build_store(tmp_path, NEMO, at_cap={(i, t) for i in range(9) for t in (0, 1)}, ladder=False)
    rows = [ESC, BELOW, NEMO]
    planned = LP.plan_all(rows, wave_root=tmp_path, sha=G.FULL_SHA, from_cap=2048, classify=_classify, served_seqs_of=lambda k: 64)
    by = {p["cell_key"]: p for p in planned}
    esc = by["gemma4e4b/main/constv2-mix-full__lora_r4_ce"]
    assert esc["exit"] == CL.ESCALATE and esc["next_cap"] == 4096 and esc["n_draws"] == 23
    rec_path = tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 2048).cap_ladder
    sub_path = tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 4096).subset
    state_path = tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 4096).ladder_state
    assert rec_path.is_file() and sub_path.is_file() and state_path.is_file() and Path(esc["subset_path"]) == sub_path
    record = json.loads(rec_path.read_text())
    again, _ = G.classify(rec_path.parent)
    adapter = {k: v for k, v in record.items() if k != IC.LADDER_DECISION_KEY}
    assert adapter == json.loads(json.dumps(again, default=str))                # the adapter's record, verbatim, plus THE decision (C1)
    decision = IC.ladder_decision(record)
    assert decision["reading"] == IC.READING_PASS1 and decision["chain"] == [2048] and decision["n_rung"] == 0     # at pass 1 the reading IS the store
    assert (decision["exit"], decision["next_cap"], decision["n_draws"], decision["n_at_cap"]) == (again["exit"], again["next_cap"], 1082, 24)
    assert decision["eligible_draws"] == again["eligible_draws"] and decision["decided_by"]["script"] == LP.SCRIPT
    assert esc["decision"] == decision and esc["store_exit"] == CL.ESCALATE
    subset = json.loads(sub_path.read_text())
    assert IC.validate_item_subset_v2(subset) == subset and subset["n_draws"] == 23 and subset["cap"] == 4096
    state = json.loads(state_path.read_text())
    assert state["rung"]["state"] == "planned" and state["rung"]["subset_sha16"] == IC.item_subset_sha16_v2(subset["draws"])
    assert state["rung"]["trigger_provenance"] == IC.IFEVAL_TRIGGER_PROVENANCE and state["rung"]["prediction"]["cap"] == 4096
    pred = state["rung"]["prediction"]
    assert pred["served_seqs"] == 233 and pred["rung_concurrency"] == 23 and pred["rate_basis"]["n_rows"] == 1082      # the source leg's S, not the CLI's 64
    assert pred["tok_s"] > 50 * pred["rate_basis"]["per_request_tok_s"]                                              # C5: the aggregate, not sum / sum
    line = esc["line"]
    assert f"--model {ESC['slug']}" in line and f"--cells {ESC['cell']}" in line and "--modes off --controls ifeval" in line
    assert f"--item-subset <vol>/{IC.ifeval_paths('gemma4e4b', 'main', ESC['cell'], 4096).subset}" in line
    assert "--off-cap-by-task ifeval=4096" in line and f"--out-dir <vol>/{IC.ifeval_paths('gemma4e4b', 'main', ESC['cell'], 4096).out_root}" in line
    assert f"--app-name ifeval-canonical-{G.SHA7}-rung4096" in line and esc["certification_needed"] is None
    # the below-threshold cell: a terminal record, no subset, no line
    below = by["gemma4e4b/main/none"]
    assert below["exit"] == CL.EXIT_BELOW_THRESHOLD and below["line"] is None and below["subset_path"] is None
    assert (tmp_path / IC.ifeval_paths("gemma4e4b", "main", "none", 2048).cap_ladder).is_file()
    assert not (tmp_path / IC.ifeval_paths("gemma4e4b", "main", "none", 4096).subset).exists()
    # the Nemotron rung names its certification for that cap's geometry entry
    nemo = by["nemotron3nano4b/main/constv2-mix-full__lora_r4_ce"]
    assert nemo["line"] and "capability:ifeval_4096" in nemo["certification_needed"] and NEMO["slug"] in nemo["line"]
    # the re-estimate line precedes every launch line
    lines = LP.summary_lines(planned, s1_allowance=3.9) + LP.launch_lines(planned)
    idx_re = next(i for i, l in enumerate(lines) if l.startswith("RE-ESTIMATE:"))
    assert all(i > idx_re for i, l in enumerate(lines) if l.startswith(LP.LAUNCHER))
    assert "ADMIT" in lines[idx_re] and "1 % raw at-cap of draws" in lines[idx_re]
    # idempotent: a second run writes nothing new and re-prints the same lines
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    planned2 = LP.plan_all(rows, wave_root=tmp_path, sha=G.FULL_SHA, from_cap=2048, classify=_classify, served_seqs_of=lambda k: 64)
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert before == after and LP.launch_lines(planned2) == LP.launch_lines(planned)
    # a running rung is never re-planned; a differing subset refuses by name
    state["rung"]["state"] = "running"
    state_path.write_text(json.dumps(state))
    p3 = LP.plan_cell(ESC, wave_root=tmp_path, sha=G.FULL_SHA, from_cap=2048, classify=_classify, served_seqs=64)
    assert "not re-planned" in p3["note"] and p3["line"] is None
    state["rung"]["state"] = "planned"
    state["rung"]["subset_sha16"] = "0" * 16
    state_path.write_text(json.dumps(state))
    with pytest.raises(SystemExit, match="already planned over subset"):
        LP.plan_cell(ESC, wave_root=tmp_path, sha=G.FULL_SHA, from_cap=2048, classify=_classify, served_seqs=64)


def test_rung_2_is_decided_on_the_merged_reading_with_denominator_1082(tmp_path):
    G.build_escalated_cell(tmp_path, ESC, at_cap={(i, t) for i in range(12) for t in (0, 1)}, loops={(0, 0), (0, 1)},
                           rung_still_at_cap={(i, t) for i in range(1, 8) for t in (0, 1)}, rung_boundary={(9, 0)})
    planned = LP.plan_all([ESC], wave_root=tmp_path, sha=G.FULL_SHA, from_cap=4096, classify=_classify, served_seqs_of=lambda k: 64)
    p = planned[0]
    d = p["decision"]
    assert d["reading"] == IC.READING_MERGED and d["chain"] == [2048, 4096] and d["n_draws"] == 1082 and d["n_at_cap"] == 16 and d["n_rung"] == 22
    assert d["exit"] == CL.ESCALATE and p["exit"] == CL.ESCALATE and p["next_cap"] == 8192 and p["n_draws"] == 14 and d["n_eligible"] == 14
    assert "--off-cap-by-task ifeval=8192" in p["line"] and (tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 8192).subset).is_file()
    # the persisted rung record carries THE decision; the store's own classification (14 of 22) stays beside it as a diagnostic
    rec = json.loads((tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 4096).cap_ladder).read_text())
    assert IC.ladder_decision(rec) == d and rec["n_draws"] == 22 and rec["n_at_cap"] == 14 and rec["exit"] == CL.ESCALATE
    # C1: a rung whose merged residual is under 1 % files NO further rung even though its own population is mostly at cap
    # (3 still-at-cap of 24 redraws = 12.5 % of the store; 3 of 1,082 = 0.28 % of the reading) — THE decision is the merged one
    G.build_escalated_cell(tmp_path / "w2", ESC, at_cap={(i, t) for i in range(12) for t in (0, 1)}, rung_still_at_cap={(1, 0), (2, 0), (3, 0)})
    planned2 = LP.plan_all([ESC], wave_root=tmp_path / "w2", sha=G.FULL_SHA, from_cap=4096, classify=_classify, served_seqs_of=lambda k: 64)
    p2 = planned2[0]
    assert p2["decision"]["n_at_cap"] == 3 and p2["decision"]["n_draws"] == 1082 and p2["decision"]["exit"] == CL.EXIT_BELOW_THRESHOLD
    assert p2["exit"] == CL.EXIT_BELOW_THRESHOLD and p2["store_exit"] == CL.ESCALATE and p2["next_cap"] is None
    assert p2["line"] is None and "no rung" in p2["note"] and "store's own exit 'escalate' (3 of 24)" in p2["note"]
    rec2 = json.loads((tmp_path / "w2" / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 4096).cap_ladder).read_text())
    assert IC.ladder_decision(rec2)["exit"] == CL.EXIT_BELOW_THRESHOLD and rec2["exit"] == CL.ESCALATE and rec2["next_cap"] == 8192
    assert not (tmp_path / "w2" / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 8192).subset).exists()
    assert not (tmp_path / "w2" / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 8192).ladder_state).exists()
    lines = LP.summary_lines(planned2, s1_allowance=3.9)
    assert "exit=below_threshold store_exit=escalate next=None" in lines[0] and LP.launch_lines(planned2) == ["# no rung to launch"]
    # a planner run at the CEILING files the terminal record (the residual reported) and plans nothing above it
    esc3 = G.build_escalated_cell(tmp_path / "w3", ESC, at_cap={(i, t) for i in range(12) for t in (0, 1)}, loops={(0, 0), (0, 1)},
                                  rung_still_at_cap={(i, t) for i in range(1, 8) for t in (0, 1)}, rung8192=True, rung8192_still_at_cap={(1, 0)})
    assert esc3["rung8192_cell_dir"] is not None
    planned3 = LP.plan_all([ESC], wave_root=tmp_path / "w3", sha=G.FULL_SHA, from_cap=8192, classify=_classify, served_seqs_of=lambda k: 64)
    p3 = planned3[0]
    assert p3["decision"]["chain"] == [2048, 4096, 8192] and p3["decision"]["n_draws"] == 1082 and p3["decision"]["n_at_cap"] == 1 + 2
    assert p3["exit"] == CL.EXIT_BELOW_THRESHOLD and p3["store_exit"] == CL.EXIT_CEILING and "no rung above" in p3["note"] and p3["line"] is None
    rec3 = json.loads((tmp_path / "w3" / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 8192).cap_ladder).read_text())
    assert IC.ladder_decision(rec3)["reading"] == IC.READING_MERGED and rec3["exit"] == CL.EXIT_CEILING


def test_a_relocated_wave_re_plans_from_a_rung_through_the_wave_relative_source(tmp_path):
    """Seam 4 (M11; lane Y's C2 residual): the rung store's subset records the SOURCE root ABSOLUTELY (where the planner
    ran). Pulled to another root, `plan --from-cap 4096` used to resolve that vanished path at the adapter; the planner now
    hands `classify` the source resolved through `ifeval_paths` under the wave root, and the adapter binds by identity."""
    import shutil
    built = tmp_path / "pull_a"
    G.build_escalated_cell(built, ESC, at_cap={(i, t) for i in range(12) for t in (0, 1)}, loops={(0, 0), (0, 1)},
                           rung_still_at_cap={(i, t) for i in range(1, 8) for t in (0, 1)}, rung_boundary={(9, 0)})
    before = LP.plan_all([ESC], wave_root=built, sha=G.FULL_SHA, from_cap=4096, classify=_classify, served_seqs_of=lambda k: 64)[0]
    assert before["exit"] == CL.ESCALATE and before["next_cap"] == 8192
    moved = tmp_path / "elsewhere" / "pull_b"
    moved.parent.mkdir()
    shutil.move(str(built), str(moved))
    subset = json.loads((moved / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 4096).subset).read_text())
    assert subset["source_out_root"].startswith(str(built)) and not Path(subset["source_out_root"]).exists()
    # the planner's production closure shape: the adapter with the override, no fixture texts (the .eval logs are absent
    # here, so the texts come from the fixture — but the SOURCE root is the override's)
    after = LP.plan_all([ESC], wave_root=moved, sha=G.FULL_SHA, from_cap=4096, classify=_classify, served_seqs_of=lambda k: 64)[0]
    assert after["exit"] == before["exit"] and after["next_cap"] == before["next_cap"] and after["n_draws"] == before["n_draws"]
    assert after["decision"]["n_at_cap"] == before["decision"]["n_at_cap"] == 16 and after["decision"]["chain"] == [2048, 4096]
    assert (moved / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 8192).subset).is_file()
    # the adapter WITHOUT the override (the pre-M11 path) resolves the recorded absolute source root, which is gone
    rung_root = moved / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 4096).out_root
    with pytest.raises(Exception, match=r"cells|manifest|No such file"):
        IR.classify_store(rung_root, task=IC.IFEVAL_TASK, texts=G.store_texts(IR.cell_dir_of(rung_root)), source_texts={})
    # and the override naming the WRONG store (another cell's pass 1) refuses on the identity binding, never re-pairs draws
    G.build_store(moved, BELOW, at_cap={(i, 0) for i in range(5)}, ladder=False)
    with pytest.raises(IR.InspectRungError, match="bound to source identity"):
        G.classify(rung_root, source_out_root=moved / IC.ifeval_paths("gemma4e4b", "main", BELOW["cell"], 2048).out_root)


def test_a_small_rung_store_predicts_from_the_source_legs_rate_labelled_and_a_pass_1_store_still_refuses(tmp_path):
    """The 2026-09-13 rung wave: a 4-draw rung store (the llama-main mix cell at 4,096) forwarded fewer than MIN_RATE_ROWS
    task rows, `observed_decode_rate` refused `need >= 10`, and the SystemExit aborted the WHOLE `--from-cap 4096` run —
    every later row left unplanned although THE decision (which reads no rate) had been filed. The prediction now rests
    on the SOURCE leg's rate when the rung store measures none, labelled `rate_source.fallback` with the reason; the
    decision, the subset and the state are untouched; a pass-1 store (no source) and any other rate problem refuse."""
    at_cap = {(i, t) for i in range(12) for t in (0, 1)}                       # 24 at cap in pass 1 ...
    loops = {(i, t) for i in range(10) for t in (0, 1)}                        # ... 20 of them loops -> 4 eligible draws
    G.build_escalated_cell(tmp_path, ESC, at_cap=at_cap, loops=loops, rung_still_at_cap={(10, 0), (11, 1)})
    rung_root = tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 4096).out_root
    src_root = tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 2048).out_root
    with pytest.raises(ValueError, match="need >= 10"):
        LP.store_rate(rung_root, job_id=ESC["job_id"])                          # 4 forwarded rows: no rate of its own
    with pytest.raises(SystemExit, match="need >= 10"):
        LP.rate_for_prediction(rung_root, source_root=None, job_id=ESC["job_id"])   # a pass-1 store has no source: refuses
    rate = LP.rate_for_prediction(rung_root, source_root=src_root, job_id=ESC["job_id"])
    assert rate["rate_source"]["fallback"] is True and rate["rate_source"]["out_root"] == str(src_root)
    assert "need >= 10" in rate["rate_source"]["reason"] and rate["n_rows"] >= LP.MIN_RATE_ROWS and rate["tok_s"] > 0
    own = LP.rate_for_prediction(src_root, source_root=None, job_id=ESC["job_id"])
    assert own["rate_source"] == {"out_root": str(src_root), "fallback": False} and own["tok_s"] == rate["tok_s"]
    planned = LP.plan_all([ESC], wave_root=tmp_path, sha=G.FULL_SHA, from_cap=4096, classify=_classify, served_seqs_of=lambda k: 64)
    p = planned[0]
    assert p["decision"]["reading"] == IC.READING_MERGED and p["decision"]["n_draws"] == 1082 and p["decision"]["n_at_cap"] == 22
    assert p["exit"] == CL.ESCALATE and p["next_cap"] == 8192 and p["n_draws"] == 2 and p["line"]
    assert p["prediction"]["rate_basis"]["rate_source"]["fallback"] is True
    assert p["prediction"]["rate_basis"]["rate_source"]["out_root"] == str(src_root)
    assert p["prediction"]["tok_s"] == rate["tok_s"]
    assert (tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 8192).subset).is_file()
    state = json.loads((tmp_path / IC.ifeval_paths("gemma4e4b", "main", ESC["cell"], 8192).ladder_state).read_text())
    assert state["rung"]["prediction"]["rate_basis"]["rate_source"]["fallback"] is True
    # the summary line still prints the rate and the RE-ESTIMATE admits on the labelled basis
    lines = LP.summary_lines(planned, s1_allowance=3.9)
    assert "exit=escalate next=8192 n_draws=2 tok/s=" in lines[0] and "ADMIT" in lines[-1]


def test_predict_and_store_status_refusals(tmp_path):
    pred = LP.predict_gpu_h(38, 4096, 800.0, served_seqs=64)
    assert pred["gpu_h_central"] < pred["gpu_h_all_at_cap"] and pred["formula"].startswith("GPU-h") and "AGGREGATE" in pred["formula"]
    assert pred["rung_concurrency"] == 38 and pred["rate_basis"] is None
    with pytest.raises(ValueError):
        LP.predict_gpu_h(0, 4096, 800.0, served_seqs=64)
    assert LP.store_status(tmp_path / "nothing") == "pending"
    cd = G.build_store(tmp_path, BELOW, ladder=False)
    root = cd.parents[1]
    assert LP.store_status(root) == "complete"
    (root / "RESULT").unlink()
    with pytest.raises(SystemExit, match="disagree"):
        LP.store_status(root)


def test_the_observed_rate_is_the_aggregate_over_the_generation_wall_not_per_request(tmp_path):
    """C5: the source leg's rate is the accepted attempt's completions over the wall its requests spanned. The fixture
    schedules 1,082 requests over the served S = 233 slots at 40 tok/s per sequence, so the aggregate is ~S x 40 while
    sum(tokens) / sum(latency) — the old estimate's reading — is exactly 40: the old reading under-read the rate by ~S and
    inflated every GPU-hour prediction by the same factor."""
    cd = G.build_store(tmp_path, BELOW, ladder=False)
    root = cd.parents[1]
    rows = [json.loads(l) for l in (root / IC.LEDGER_FILENAME).read_text().splitlines()]
    rate = LP.observed_decode_rate(rows, accepted=G.ATTEMPT_ID, evidence_by_id={})
    # the independent expectation, off the same rows
    starts = [IC._utc_stamp(r["ts"]) for r in rows]
    ends = [s + __import__("datetime").timedelta(seconds=r["latency_s"]) for s, r in zip(starts, rows)]
    expected = sum(r["usage_completion_tokens"] for r in rows) / (max(ends) - min(starts)).total_seconds()
    assert rate["tok_s"] == pytest.approx(expected) and rate["n_rows"] == 1082 and rate["rule"] == IC.DECODE_RATE_RULE
    assert rate["per_request_tok_s"] == pytest.approx(G.DECODE_TOK_S, rel=1e-6)
    assert 0.5 * 233 * G.DECODE_TOK_S < rate["tok_s"] <= 233 * G.DECODE_TOK_S            # ~S x the per-sequence rate, less the tail
    assert rate["tok_s"] > 50 * rate["per_request_tok_s"]
    # the reviewer's arithmetic: 64 requests x 100 tokens, all within ONE second -> 6,400 tok/s, not 100
    overlap = [{"ts": "2026-01-01T00:00:00Z", "latency_s": 1.0, "usage_completion_tokens": 100, "task": IC.IFEVAL_TASK, "sample": f"i{i}/0",
                "rejected": False, "attempt_id": "a1"} for i in range(64)]
    agg = IC.aggregate_decode_tok_s(overlap)
    assert agg["tok_s"] == 6400.0 and agg["per_request_tok_s"] == 100.0 and agg["elapsed_s"] == 1.0
    assert LP.observed_decode_rate(overlap, accepted="a1", evidence_by_id={})["tok_s"] == 6400.0
    # attempt-scoped: another attempt's rows never enter the rate; an unattributable row refuses by name
    other = [{**r, "attempt_id": "a2", "usage_completion_tokens": 10_000} for r in overlap]
    assert LP.observed_decode_rate(overlap + other, accepted="a1", evidence_by_id={})["tok_s"] == 6400.0
    with pytest.raises(__import__("sisteer.capeval.tinycap.contracts", fromlist=["x"]).AttemptScopeError):
        LP.observed_decode_rate(overlap + [{**overlap[0], "attempt_id": None, "request_id": None}], accepted="a1", evidence_by_id={})
    # refusals: too few rows, a naive stamp, a zero wall
    with pytest.raises(ValueError, match="need >= 10"):
        IC.aggregate_decode_tok_s(overlap[:3])
    with pytest.raises(ValueError, match="placeable"):
        IC.aggregate_decode_tok_s([{**r, "ts": "2026-01-01T00:00:00"} for r in overlap])
    with pytest.raises(ValueError, match="positive latency_s"):
        IC.aggregate_decode_tok_s([{**r, "latency_s": 0} for r in overlap])
    # the prediction records the rate's basis and the source's served S (off its endpoint.json), and the rung's concurrency
    pred = LP.predict_gpu_h(23, 4096, rate["tok_s"], served_seqs=LP.source_served_seqs(root, default=64), rate=rate)
    assert pred["served_seqs"] == 233 and pred["rung_concurrency"] == 23 and pred["rate_basis"]["n_rows"] == 1082
    assert "tok_s" not in pred["rate_basis"] and pred["tok_s"] == rate["tok_s"]
    assert LP.source_served_seqs(tmp_path / "nowhere", default=64) == 64
