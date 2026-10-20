"""The S5 multi-turn window measurement has an IN-CONTAINER producer, bound to the attempt.

The launch contract (`S5_MEASUREMENT_REQUIRED`, `multiturn_window_gate`, `certificate_problems`) requires for the two S5 probe cells a
measurement with `context_max`, `clamped_calls == 0`, `window_source == measured_max_complete_store`; before this commit NO producer emitted
it -- `seedrep_s5_record.py window --record` ingested operator JSON. Now:

* `scripts/modal/seedrep_s5_measure.py` reads the probe child's COMPLETE multi-turn store (`read_multiturn_store` completeness) + the
  served target's per-call `target_timing.jsonl` (prompt_tokens + completion_tokens from the engine's usage; `max_tokens_clamped_from` /
  `error: prompt_over_window` -- the clamp is DECIDED in the served client, `served_target._one_send`, which writes that row) + the driver's
  final `served_target.json` counters, and emits `seedrep_s5_window_measurement/1` with `produced_by {writer, tree_sha, attempt_id, ...}`;
* `seedrep_served.producer_evidence` runs it per probe child and binds `audits/s5_window/<leg>/<cell>__s<s>.json` (`_bound_file`);
* `seedrep_s5_record.py window --rows` DERIVES the records off the accepted attempts (`lp.derive_smoke_pass("S5")`) and `--record` is refused;
* the FINAL plan's `multiturn_window_gate` / `window_justification` read them.

Synthetic data throughout: invented goal ids, placeholder uuids, no real transcript text.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import seedrep_launch_plan as lp  # noqa: E402

from test_seedrep_review_fixes import TestDoseStandups as _Dose, _plan, _stub_modal, WORKTREES, W  # noqa: E402  (aliased: not collected here)

_ABSENT = object()
SMOKE_WAVE = "11111111-2222-4333-8444-555555555555"
ATTEMPT = "ta-s5-probe-0001"
CELL = lp.LORA_DIFF_CELL
PROBE = {c["staged_id"]: c for c in lp.S5_PROBE_CELLS}
D1P5 = f"d1p5_eb64s1__{CELL}"
DM0P5 = f"dm0p5_eb64s1__{CELL}"
POLICY_REF = lp.MULTITURN_DECODE_POLICY_REF


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def M():
    return _load("seedrep_s5_measure_under_test", "scripts/modal/seedrep_s5_measure.py")


@pytest.fixture(scope="module")
def S5R():
    return _load("seedrep_s5_record_w16s5", "scripts/modal/seedrep_s5_record.py")


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_w16s5", REPO / "scripts/modal/seedrep_served.py")
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


@pytest.fixture(scope="module")
def inputs():
    return lp.load_inputs(REPO, config_roots=(WORKTREES / "seedrep-ws1dp",))


@pytest.fixture(scope="module")
def probe_plan(inputs):
    ro, pins = _Dose._peer(inputs)
    plan = _plan(inputs, peer_roster=ro, peer_pins=pins)
    rows = [r for r in plan["s5_probe_rows"] if r["view"] in ("d1p5_loradiff64s1", "dm0p5_loradiff64s1")]
    assert len(rows) == 4 and {r["leg"] for r in rows} == set(lp.S5_PROBE_LEGS)
    return plan


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_store(root: Path, *, leg: str, cell: str, s: int, goals, n_turns: int = 2, cap: int = 4096, window: int = 16384, prompt: int = 3000,
                clamped_calls: int = 0, over_window_goals=(), errored_goals=(), drop_reply_rows: int = 0, manifest_stage: str = "final",
                timing: bool = True, manifest_counters=None, view: str = "d1p5_loradiff64s1") -> Path:
    """A synthetic probe child's store: responses/progress (the multi-turn runner's shape), target_timing.jsonl (the served client's per-call
    rows), served_target.json (final). `clamped_calls` marks that many served rows `max_tokens_clamped_from`; `over_window_goals` end
    `error:PromptOverWindow` with a prompt_over_window timing row; `errored_goals` end with another error; `drop_reply_rows` removes stored
    replies (incomplete by the completeness rule)."""
    shard = root / "exp" / "transcripts_results" / f"qwen35_4b_{view}_sampled_t0" / leg / cell / "shard00"
    sampler = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}                       # what the REAL runners persist under progress target_decode
    td = {"do_sample": True, **sampler, "cap": cap, "arm": "vendor_recommended", "seed_base": s, "trial": 0,
          "seed_rule": "multiturn_seed", "multiturn_target_decode": "vendor", "multiturn_target_seed": "per_turn"}   # the driver's FINAL manifest decode block
    rows, progress, trows = [], [], []
    n_clamped = 0
    for gi, g in enumerate(goals):
        if g in over_window_goals:
            trows.append({"t_start": 1.0, "latency_s": 0.1, "virtual_model": f"vm-{cell}", "n_messages": 3, "max_tokens": cap, "finish_reason": None,
                          "error": "prompt_over_window", "prompt_tokens": window - 10, "completion_tokens": 0, "window": window, "server_window": window})
            progress.append({"goal_id": g, "category": "cat", "n_target_forwards": 1, "outcome": "error:PromptOverWindow", "actors_used": 0, "capped": False,
                             "served_window": window, "target_decode": dict(sampler)})      # chat() counted the attempted forward
            continue
        for turn in range(n_turns):
            ct = 400 + 7 * turn
            pt = prompt + 250 * turn + 13 * gi
            rows.append({"goal_id": g, "order": turn, "response": f"synthetic reply {g}/{turn}", "serving_engine": "lora_native",
                         "decode": {"temperature": 0.7, "seed": 1000 + gi * 10 + turn}, "finish_reason": "stop"})
            tr = {"t_start": 1.0 + turn, "latency_s": 0.2, "virtual_model": f"vm-{cell}", "n_messages": 2 * turn + 1, "max_tokens": cap, "prompt_tokens": pt,
                  "completion_tokens": ct, "finish_reason": "stop", "intervention_hash": None, "decode": {"temperature": 0.7}}
            if n_clamped < clamped_calls:
                tr["max_tokens_clamped_from"] = cap
                tr["max_tokens"] = window - pt
                tr["prompt_tokens_source"] = "error_message"
                n_clamped += 1
            trows.append(tr)
        outcome = "error:ActorDiscoveryError" if g in errored_goals else "failure"
        progress.append({"goal_id": g, "category": "cat", "n_target_forwards": n_turns, "outcome": outcome, "actors_used": 0, "capped": False,
                         "served_window": window, "target_decode": dict(sampler)})
    if drop_reply_rows:
        rows = rows[:-drop_reply_rows]
    _jsonl(shard / "responses.jsonl", rows)
    _jsonl(shard / "progress.jsonl", progress)
    if timing:
        _jsonl(shard / "target_timing.jsonl", trows)
    served = [r for r in trows if r.get("error") is None]
    counters = {"n_clamped": n_clamped, "n_prompt_over_window": len([r for r in trows if r.get("error") == "prompt_over_window"]),
                "n_target_forwards": len(trows), "n_window_mismatch": 0, "window_mismatches": []}     # ATTEMPTED forwards (the real client's semantics)
    if manifest_counters:
        counters.update(manifest_counters)
    (shard / "served_target.json").write_text(json.dumps({"target_engine": "served", "virtual_model": f"vm-{cell}", "cell_id": cell, "manifest_stage": manifest_stage,
                                                          "serving_engine": "lora_native", "window_clamp": {"max_model_len": window, **counters},
                                                          "decode": dict(td), "target_decode_resolution": {"ref": POLICY_REF, "model_key": lp.MODEL_KEY}}))
    return shard


def write_attempt(root: Path, *, standup_id: str, W: str = W, attempt_id: str = ATTEMPT, wave: str = SMOKE_WAVE) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ATTEMPT.json").write_text(json.dumps({"schema": "seedrep_attempt/1", "attempt_id": attempt_id, "call_id": "fc-0001", "task_id": "ta-x", "W": W,
                                                   "wave_uuid": wave, "execution_wave_uuid": wave, "launch_plan_row": standup_id, "control_key": None}))
    return root


def _argv(root: Path, *, leg: str, cell: str, arm: str, kind: str, staged: str, out: Path, sid: str, expected: int, sha=None):
    a = ["--root", str(root), "--leg", leg, "--cell", cell, "--s", "1", "--arm", arm, "--arm-kind", kind, "--staged-id", staged, "--decode-policy-ref", POLICY_REF,
         "--model-key", lp.MODEL_KEY, "--n-goals-expected", str(expected), "--standup-id", sid, "--out", str(out)]
    if sha:
        a += ["--object-sha256", sha]
    return a


# --------------------------------------------------------------------------- the producer over a synthetic root
def test_producer_measures_the_max_served_context_and_zero_clamped_calls_on_a_complete_store(M, tmp_path):
    root = write_attempt(tmp_path / "attempt", standup_id="crescendo__d1p5_loradiff64s1__base__shard00")
    goals = [f"g{k}" for k in range(5)]
    shard = write_store(root, leg="crescendo", cell=CELL, s=1, goals=goals, n_turns=3)
    out = root / lp.s5_window_rel("crescendo", CELL, 1, stage="base", shard=0)
    rc = M.main(_argv(root, leg="crescendo", cell=CELL, arm=D1P5, kind="dosed", staged=D1P5, out=out, sid="crescendo__d1p5_loradiff64s1__base__shard00",
                      expected=5, sha=PROBE[D1P5]["object_sha256"]))
    assert rc == 0 and out.is_file()
    rec = json.loads(out.read_text())
    assert lp.window_measurement_problems(rec) == [], lp.window_measurement_problems(rec)
    # the MAX over every served call of prompt + completion: goal 4, turn 2 -> 3000 + 500 + 52 + 414
    assert rec["context_max"] == 3000 + 250 * 2 + 13 * 4 + (400 + 14) and rec["context_max_call"]["prompt_tokens"] == 3000 + 500 + 52
    assert rec["clamped_calls"] == 0 and rec["n_calls"] == 15 and rec["n_prompt_over_window"] == 0 and rec["store_complete"] is True and rec["complete"] is True
    assert rec["window_source"] == "measured_max_complete_store" and rec["schema"] == lp.S5_WINDOW_MEASUREMENT_SCHEMA
    assert rec["arm"] == D1P5 and rec["arm_kind"] == "dosed" and rec["objects_measured"] == [D1P5] and rec["object_sha256"] == PROBE[D1P5]["object_sha256"]
    assert rec["n_goals_completed"] == 5 == rec["n_goals_expected"] and rec["cap"] == 4096 and rec["served_window"] == 16384 and rec["decode"] == "vendor"
    assert rec["attempt_id"] == ATTEMPT and rec["wave_uuid"] == SMOKE_WAVE and rec["store_path"] == str(shard.parent.relative_to(root))
    assert rec["produced_by"] == {"writer": "scripts/modal/seedrep_s5_measure.py", "tree_sha": W, "attempt_id": ATTEMPT,
                                  "standup_id": "crescendo__d1p5_loradiff64s1__base__shard00", "wave_uuid": SMOKE_WAVE}
    assert all(k in rec for k in lp.S5_WINDOW_MEASUREMENT_EXTRA_KEYS) and len(rec["store_sha256"]) == 64
    # the record round-trips through the gate with a parent + the other cell beside it
    parent = {**rec, "arm": "d1p5_loradiff64s1__none", "arm_kind": "parent", "objects_measured": ["none"], "staged_id": "none", "object_sha256": None, "cell": "none"}
    other = {**rec, "arm": DM0P5, "objects_measured": [DM0P5], "staged_id": DM0P5, "context_max": 3000}
    gate = lp.multiturn_window_gate("crescendo", 16384, [rec, parent, other], wave_uuid=SMOKE_WAVE)
    assert gate["ok"] and gate["measured_max_context"] == rec["context_max"], gate["problems"]


def test_known_bad_one_clamped_call_is_measured_as_clamped_calls_1_and_every_consumer_refuses(M, tmp_path):
    root = write_attempt(tmp_path / "attempt", standup_id="sid")
    write_store(root, leg="actorattack", cell="none", s=1, goals=["g0", "g1", "g2"], clamped_calls=1)
    out = tmp_path / "m.json"
    assert M.main(_argv(root, leg="actorattack", cell="none", arm="d1p5_loradiff64s1__none", kind="parent", staged="none", out=out, sid="sid", expected=3)) == 0
    rec = json.loads(out.read_text())
    assert rec["clamped_calls"] == 1 and rec["n_clamped_retries"] == 1 and rec["store_complete"] is True      # an honest measurement...
    assert lp.window_measurement_problems(rec) == []                                                          # ...schema-valid...
    good = [{**rec, "clamped_calls": 0, "arm": sid, "arm_kind": "dosed", "objects_measured": [sid], "staged_id": sid} for sid in PROBE]
    g = lp.multiturn_window_gate("actorattack", 16384, [rec] + good, wave_uuid=SMOKE_WAVE)
    assert not g["ok"] and any("1 clamped call" in p for p in g["problems"])                                  # ...that the FINAL gate refuses
    # a prompt that could not fit is a clamped call too (the multi-turn policy files it as error:PromptOverWindow; the store is a floor)
    root2 = write_attempt(tmp_path / "attempt2", standup_id="sid")
    write_store(root2, leg="actorattack", cell="none", s=1, goals=["g0", "g1", "g2"], over_window_goals=("g2",))
    out2 = tmp_path / "m2.json"
    assert M.main(_argv(root2, leg="actorattack", cell="none", arm="x__none", kind="parent", staged="none", out=out2, sid="sid", expected=3)) == 0
    rec2 = json.loads(out2.read_text())
    assert rec2["clamped_calls"] == 1 and rec2["n_prompt_over_window"] == 1 and rec2["store_complete"] is False and rec2["n_goals_completed"] == 2


def test_known_bad_an_incomplete_store_is_a_floor_the_gate_refuses(M, tmp_path):
    # fewer goals than the plan expects
    root = write_attempt(tmp_path / "a", standup_id="sid")
    write_store(root, leg="crescendo", cell="none", s=1, goals=["g0", "g1"])
    out = tmp_path / "m.json"
    assert M.main(_argv(root, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=5)) == 0
    rec = json.loads(out.read_text())
    assert rec["store_complete"] is False and rec["complete"] is True and rec["n_goals_completed"] == 2 and rec["n_goals_expected"] == 5
    assert lp.window_measurement_problems(rec) == []
    g = lp.multiturn_window_gate("crescendo", 16384, [rec], wave_uuid=SMOKE_WAVE)
    assert any("not COMPLETE" in p for p in g["problems"])
    # a stored-rows / progress disagreement -> complete False
    root2 = write_attempt(tmp_path / "b", standup_id="sid")
    write_store(root2, leg="crescendo", cell="none", s=1, goals=["g0", "g1"], drop_reply_rows=1)
    out2 = tmp_path / "m2.json"
    assert M.main(_argv(root2, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out2, sid="sid", expected=2)) == 2   # served rows != stored replies
    assert not out2.exists()
    # an errored goal -> complete False, store_complete False
    root3 = write_attempt(tmp_path / "c", standup_id="sid")
    write_store(root3, leg="crescendo", cell="none", s=1, goals=["g0", "g1"], errored_goals=("g1",))
    out3 = tmp_path / "m3.json"
    assert M.main(_argv(root3, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out3, sid="sid", expected=2)) == 0
    rec3 = json.loads(out3.read_text())
    assert rec3["complete"] is False and rec3["store_complete"] is False and "error" in rec3["completeness_note"]


def test_producer_refuses_without_the_per_call_record_or_a_final_manifest_or_agreeing_counters(M, tmp_path, capsys):
    root = write_attempt(tmp_path / "a", standup_id="sid")
    write_store(root, leg="crescendo", cell="none", s=1, goals=["g0"], timing=False)
    out = tmp_path / "m.json"
    assert M.main(_argv(root, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=1)) == 2
    assert "target_timing.jsonl" in capsys.readouterr().err and not out.exists()
    root2 = write_attempt(tmp_path / "b", standup_id="sid")
    write_store(root2, leg="crescendo", cell="none", s=1, goals=["g0"], manifest_stage="started")
    assert M.main(_argv(root2, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=1)) == 2
    assert "not 'final'" in capsys.readouterr().err
    root3 = write_attempt(tmp_path / "c", standup_id="sid")
    write_store(root3, leg="crescendo", cell="none", s=1, goals=["g0"], manifest_counters={"n_clamped": 1})
    assert M.main(_argv(root3, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=1)) == 2
    assert "window_clamp counters disagree" in capsys.readouterr().err
    # the wrong row / the wrong seed base / another decode row ref: refused by name
    root4 = write_attempt(tmp_path / "d", standup_id="other")
    write_store(root4, leg="crescendo", cell="none", s=1, goals=["g0"])
    assert M.main(_argv(root4, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=1)) == 2
    assert "launch_plan_row" in capsys.readouterr().err
    root5 = write_attempt(tmp_path / "e", standup_id="sid")
    write_store(root5, leg="crescendo", cell="none", s=2, goals=["g0"])
    assert M.main(_argv(root5, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=1)) == 2
    assert "seed_base" in capsys.readouterr().err
    assert M.main(_argv(root, leg="crescendo", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=1)[:-2]
                  + ["--out", str(out), "--decode-policy-ref", "other|ref"]) == 2
    assert not out.exists()


# --------------------------------------------------------------------------- writer -> reader on the REAL functions
def _stub_transport(*, window: int, over_window_on: int = -1):
    """An OpenAI-shaped chat transport for the REAL `ServedChatTarget`: usage on every reply; call #`over_window_on` is vLLM's context-length 400
    as the proxy client re-presents it (`returned 400: ...`), which the real client turns into the typed PromptOverWindow (strict multi-turn)."""
    calls = {"n": 0}

    def chat_fn(payload):
        calls["n"] += 1
        if calls["n"] == over_window_on:
            raise RuntimeError(f"proxy returned 400: This model's maximum context length is {window} tokens. However, your prompt contains {window - 100} "
                               f"input tokens and you requested {payload['max_tokens']} output tokens, for a total of {window - 100 + payload['max_tokens']} tokens. "
                               "Please reduce the length of the messages or completion.")
        n_msgs = len(payload["messages"])
        return {"id": f"chatcmpl-{calls['n']:04d}", "model": payload["model"], "choices": [{"message": {"role": "assistant", "content": f"synthetic reply {calls['n']}"},
                                                                                           "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 900 + 150 * n_msgs, "completion_tokens": 40 + calls["n"], "total_tokens": 940 + 150 * n_msgs + calls["n"]}}
    return chat_fn, calls


def _real_actorattack_store(root: Path, *, leg: str, cell: str, goals, over_window_on: int = -1, window: int = 16384, cap: int = 4096):
    """Drive the REAL ActorAttack writer (`multiturn.run_actorattack`) with the REAL served client (`ServedChatTarget`, its timing log, its
    attempted-forward counter) over a stub transport, then the REAL manifest writer (`write_target_manifest(stage="final")`). Nothing is
    hand-enriched: progress / responses / target_timing.jsonl / served_target.json are the production writers' bytes."""
    from sisteer.eval.attacks import multiturn as mt
    from sisteer.eval.attacks.served_target import ServedChatTarget, write_target_manifest
    shard = root / "exp" / "transcripts_results" / "qwen35_4b_d1p5_loradiff64s1_sampled_t0" / leg / cell / "shard00"
    shard.mkdir(parents=True)
    chat_fn, calls = _stub_transport(window=window, over_window_on=over_window_on)
    target = ServedChatTarget(chat_fn, virtual_model=f"vm-{cell}", cell_id=cell, timing_log=shard / "target_timing.jsonl", max_model_len=window,
                              serving_engine="lora_native")
    sampler = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    actors = json.dumps({"actors": [{"name": f"actor{i}", "queries": [f"q{i}a", f"q{i}b"]} for i in range(mt.ACTORATTACK_ACTOR_NUM)]})
    gs = [types.SimpleNamespace(goal_id=g, text=f"synthetic objective {g}", category="cat") for g in goals]
    mt.run_actorattack(target, gs, attacker_fn=lambda msgs: actors, scorer_fn=lambda prompt: "1", out_dir=shard, actor_num=mt.ACTORATTACK_ACTOR_NUM,
                       max_new_tokens=cap, target_decode=sampler, seed_base=1, trial=0, served_window=window)
    block = {"do_sample": True, **sampler, "cap": cap, "arm": "vendor_recommended", "seed_base": 1, "trial": 0, "seed_rule": "sha256(...)[:4]",
             "source": POLICY_REF, "source_kind": "registry_row", "multiturn_target_decode": "vendor", "multiturn_target_seed": "per_turn"}
    write_target_manifest(shard, target, stage="final", decode=block, extra={"target_decode_resolution": {"ref": POLICY_REF, "model_key": lp.MODEL_KEY}})
    return shard, target, calls


def test_real_writers_to_reader_the_actorattack_runner_and_the_served_client_produce_what_the_producer_reads(M, tmp_path):
    root = write_attempt(tmp_path / "attempt", standup_id="sid")
    shard, target, calls = _real_actorattack_store(root, leg="actorattack", cell="none", goals=["g0", "g1"])
    prog = [json.loads(l) for l in (shard / "progress.jsonl").read_text().splitlines()]
    assert prog and all("seed_base" not in p["target_decode"] and "cap" not in p["target_decode"] for p in prog), "the REAL writer persists sampler keys only"
    assert all(not str(p["outcome"]).startswith("error") for p in prog), [p["outcome"] for p in prog]
    out = tmp_path / "m.json"
    rc = M.main(_argv(root, leg="actorattack", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=2))
    assert rc == 0, (shard / "target_timing.jsonl").read_text()[:400]
    rec = json.loads(out.read_text())
    timing = [json.loads(l) for l in (shard / "target_timing.jsonl").read_text().splitlines()]
    assert rec["n_calls"] == len(timing) == target.n_target_forwards == calls["n"] and rec["clamped_calls"] == 0 and rec["store_complete"] is True
    assert rec["context_max"] == max(t["prompt_tokens"] + t["completion_tokens"] for t in timing) and rec["cap"] == 4096 and rec["served_window"] == 16384
    assert lp.window_measurement_problems(rec) == []


def test_real_client_over_window_call_is_an_attempted_forward_the_producer_reconciles_not_refuses(M, tmp_path):
    """`ServedChatTarget.chat` counts the forward BEFORE sending, so one PromptOverWindow call makes n_target_forwards = served + 1; the
    producer must reconcile (served + failed == attempted) and write the honest `clamped_calls 1, store_complete False` record, never refuse."""
    root = write_attempt(tmp_path / "attempt", standup_id="sid")
    shard, target, calls = _real_actorattack_store(root, leg="actorattack", cell="none", goals=["g0", "g1"], over_window_on=3)
    prog = [json.loads(l) for l in (shard / "progress.jsonl").read_text().splitlines()]
    assert any(p["outcome"] == "error:PromptOverWindow" for p in prog) and target.n_prompt_over_window == 1
    man = json.loads((shard / "served_target.json").read_text())
    timing = [json.loads(l) for l in (shard / "target_timing.jsonl").read_text().splitlines()]
    assert man["window_clamp"]["n_target_forwards"] == len(timing) == len([t for t in timing if t.get("error") is None]) + 1
    out = tmp_path / "m.json"
    assert M.main(_argv(root, leg="actorattack", cell="none", arm="v__none", kind="parent", staged="none", out=out, sid="sid", expected=2)) == 0
    rec = json.loads(out.read_text())
    assert rec["clamped_calls"] == 1 and rec["n_prompt_over_window"] == 1 and rec["store_complete"] is False and rec["complete"] is False
    assert "error:PromptOverWindow" in rec["completeness_note"]
    assert not lp.multiturn_window_gate("actorattack", 16384, [rec], wave_uuid=SMOKE_WAVE)["ok"]


def test_ws3g_read_multiturn_store_reads_the_real_actorattack_writers_output(tmp_path):
    """`audit_common.read_multiturn_store` int()'d `target_decode.seed_base`, which the REAL runners never persist, so every real multi-turn
    store refused. It now binds seed_base / trial / cap from the driver's FINAL manifest `decode` block and holds the progress rows' sampler
    keys to it; every identity / completeness check is kept."""
    import audit_common as ac
    root = write_attempt(tmp_path / "attempt", standup_id="sid")
    shard, target, calls = _real_actorattack_store(root, leg="actorattack", cell="none", goals=["g0", "g1"])
    sr = ac.read_multiturn_store([shard], attack="actorattack", arm="none")
    n_rows = len((shard / "responses.jsonl").read_text().splitlines())
    assert sr.complete is True and sr.seed_base == 1 and sr.trial == 0 and sr.n_rows == n_rows == target.n_target_forwards
    assert sr.reply_cap == 4096 and sr.cap is None, "the manifest's cap is METADATA; `cap` would switch filed_here to the static rung semantics"
    assert sr.leg == "actorattack" and sr.arm == "none" and len(sr.draws) == n_rows and all(d.sample_idx >= 0 for d in sr.draws)
    assert sr.filed_here == sr.filed == {(d.task, g) for d in sr.draws[:1] for g in ("g0", "g1")}
    assert any("served_target.json decode block" in n for n in sr.notes)
    # the seed rule evaluates from the bound seed base (the same rule as before)
    d0 = sr.draws[0]
    assert sr.expected_seed(d0.task, d0.item_id, d0.sample_idx) == int(ac._multiturn_seed(1, 0, "actorattack", d0.item_id, d0.sample_idx))
    # one over-window call through the REAL client -> the goal errored -> INCOMPLETE by name, never a refusal
    root2 = write_attempt(tmp_path / "attempt2", standup_id="sid")
    shard2, _, _ = _real_actorattack_store(root2, leg="actorattack", cell="none", goals=["g0", "g1"], over_window_on=3)
    sr2 = ac.read_multiturn_store([shard2], attack="actorattack", arm="none")
    assert sr2.complete is False and "error:PromptOverWindow" in sr2.completeness_note
    # a progress sampler key disagreeing with the manifest's block -> refused by name
    man = json.loads((shard / "served_target.json").read_text())
    man["decode"]["temperature"] = 0.2
    (shard / "served_target.json").write_text(json.dumps(man))
    with pytest.raises(ac.AuditRefusal, match="disagrees with the manifest"):
        ac.read_multiturn_store([shard], attack="actorattack", arm="none")
    # a non-final manifest -> incomplete; no manifest decode block AND no progress seed_base -> refused (the store states its seed base nowhere)
    man["decode"]["temperature"] = 0.7
    man["manifest_stage"] = "started"
    (shard / "served_target.json").write_text(json.dumps(man))
    assert ac.read_multiturn_store([shard], attack="actorattack", arm="none").complete is False
    del man["decode"]
    (shard / "served_target.json").write_text(json.dumps(man))
    with pytest.raises(ac.AuditRefusal, match="does not say which target seed base"):
        ac.read_multiturn_store([shard], attack="actorattack", arm="none")
    man["decode"] = {**dict(json.loads((shard2 / "served_target.json").read_text())["decode"])}
    man["manifest_stage"] = "final"
    (shard / "served_target.json").write_text(json.dumps(man))
    # a goal with a NON-error terminal record declaring ZERO forwards (no reply rows) stays FILED HERE (the manifest's cap is
    # metadata, never the static rung semantics of filed_here), so the audit's reconciliation sees a filed item with no recorded attempt
    with (shard / "progress.jsonl").open("a") as fh:
        fh.write(json.dumps({"goal_id": "g_zero", "category": "cat", "n_target_forwards": 0, "outcome": "failure", "actors_used": 0, "capped": False,
                             "served_window": 16384, "target_decode": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}}) + "\n")
    sr0 = ac.read_multiturn_store([shard], attack="actorattack", arm="none")
    assert (sr0.tasks[0], "g_zero") in sr0.filed_here and sr0.filed_here == sr0.filed and len(sr0.filed_here) == 3 and sr0.complete is True


def test_ws3g_audit_refuses_a_filed_goal_with_no_recorded_attempt_on_a_manifest_backed_multiturn_store(tmp_path):
    """AUDIT: the synthetic wave's actorattack store (its manifest now carries the real `decode` block) audits AGREES;
    one more NON-error terminal goal declaring zero forwards (no reply rows, no request) -> the audit refuses with the unattempted-item message,
    exactly as before the manifest binding (with `cap` set, filed_here was EMPTY and this passed)."""
    import audit_common as ac
    import seedrep_audit_fixtures as fx
    import request_evidence_audit as rea
    root = fx.make_root(tmp_path / "stores", leg="actorattack", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=2)
    ar = ac.read_attempt_root(root.path)
    ok = rea.audit_root(ar, out_dir=tmp_path / "audits_ok", legs=["actorattack"], policy_file=None, thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=list(fx.PETRI_BEHAVIOURS), model_key=ac.MODEL_KEY, client_status="absent", siblings=[ar])
    assert [v["verdict"] for v in ok] == ["AGREES"], [v.get("problem") for v in ok]
    shard = next(p for p in ac.walk_files(root.path, "progress.jsonl") if "actorattack" in p.parts)
    with shard.open("a") as fh:
        fh.write(json.dumps({"goal_id": "g_zero", "category": "cat", "n_target_forwards": 0, "outcome": "failure", "actors_used": 0, "capped": False,
                             "served_window": root.L, "target_decode": {}}) + "\n")
    sr = ac.read_multiturn_store([shard.parent], attack="actorattack", arm="none")
    assert sr.reply_cap == 4096 and sr.cap is None and (sr.tasks[0], "g_zero") in sr.filed_here
    bad = rea.audit_root(ar, out_dir=tmp_path / "audits_bad", legs=["actorattack"], policy_file=None, thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=list(fx.PETRI_BEHAVIOURS), model_key=ac.MODEL_KEY, client_status="absent", siblings=[ar])
    assert [v["verdict"] for v in bad] != ["AGREES"] and any("NO recorded attempt" in str(v.get("problem")) for v in bad), [v.get("problem") for v in bad]


# --------------------------------------------------------------------------- the app binds it under producer_evidence (ONE insertion)
def test_producer_evidence_runs_the_measurement_per_probe_child_and_binds_it_like_the_audit(srv, probe_plan, tmp_path):
    row = next(r for r in probe_plan["s5_probe_rows"] if r["leg"] == "crescendo" and r["view"] == "d1p5_loradiff64s1")
    assert [k["cell"] for k in row["children"]] == ["none", CELL] and row["children"][1]["arm"] == D1P5
    root = write_attempt(tmp_path / "attempt", standup_id=row["standup_id"], W=probe_plan["W"])
    (root / "children").mkdir()
    for k in row["children"]:
        write_store(root, leg="crescendo", cell=k["cell"], s=1, goals=[f"g{i}" for i in range(259)], n_turns=1)
    calls = []

    def run(argv, **kw):
        calls.append(list(argv))
        if argv[1].endswith("seedrep_s5_measure.py"):
            return subprocess.run(argv, **{**kw, "env": {**kw.get("env", {}), "PYTHONPATH": str(REPO / "src")}})
        return types.SimpleNamespace(returncode=0)

    out = srv.producer_evidence(root, tree=REPO, W=probe_plan["W"], row=row, serve_dir=root / "exp" / "serve", driver_python=sys.executable,
                                env={"PATH": os.environ.get("PATH", "/usr/bin")}, run=run)
    rels = [rel for _k, rel in lp.s5_window_rels(row)]
    assert rels == [f"audits/s5_window/crescendo/none__s1.json", f"audits/s5_window/crescendo/{CELL}__s1.json"]
    for k, rel in lp.s5_window_rels(row):
        b = out[rel]
        assert b["writer"] == "scripts/modal/seedrep_s5_measure.py" == lp.PRODUCER_WRITERS["s5_window_measurement"] and b["tree_sha"] == probe_plan["W"] and b["rc"] == 0
        assert "absent" not in b and b["sha256"] == lp.sha256_file(root / rel)
        rec = json.loads((root / rel).read_text())
        want = lp.s5_probe_arm_of(row, k)
        assert (rec["leg"], rec["cell"], rec["arm"], rec["arm_kind"]) == ("crescendo", k["cell"], want["arm"], want["arm_kind"]) and rec["clamped_calls"] == 0
        assert rec["store_complete"] is True and rec["n_goals_expected"] == 259 and rec["produced_by"]["standup_id"] == row["standup_id"]
    mea = [c for c in calls if c[1].endswith("seedrep_s5_measure.py")]
    assert len(mea) == 2 and all(c[0] == sys.executable and c[c.index("--root") + 1] == str(root) for c in mea)
    assert mea[1][mea[1].index("--object-sha256") + 1] == PROBE[D1P5]["object_sha256"] and mea[1][mea[1].index("--decode-policy-ref") + 1] == POLICY_REF
    assert (root / "children" / "s5_measure.log").is_file()
    # a non-probe row runs no measurement (the insertion is keyed on the row kind)
    base = probe_plan["standups"][0]
    calls.clear()
    out2 = srv.producer_evidence(root, tree=REPO, W=probe_plan["W"], row=base, serve_dir=root / "exp" / "serve", driver_python=sys.executable,
                                 env={"PATH": "/usr/bin"}, run=run)
    assert not any(c[1].endswith("seedrep_s5_measure.py") for c in calls) and not any(k.startswith("audits/s5_window") for k in out2)


_MEASURE = None


def measure_module():
    global _MEASURE
    if _MEASURE is None:
        _MEASURE = _load("seedrep_s5_measure_world", "scripts/modal/seedrep_s5_measure.py")
    return _MEASURE


def s5_probe_world(plan, rs: Path, tmp_path: Path, *, wave: str = SMOKE_WAVE, clamp_on=None, incomplete_on=None, mint_wave: bool = True):
    """The four probe rows' accepted attempts on a fake volume: each root's ATTEMPT.json, root facts, the REAL producer's record per child
    (generated here by seedrep_s5_measure over synthetic stores), bound under modal_evidence.json[producer_evidence] -> (rows, accepted {key: aid},
    texts {volume path: text}). The certificate tests build their S5 block THROUGH this (W16s5 (2): certify re-derives the windows)."""
    M = measure_module()
    if mint_wave:
        wp = lp.smoke_wave_path(rs, "S5")
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps({"smoke_id": "S5", "wave_uuid": wave}))
    ident = lp.smoke_identity(plan, "S5", wave)
    rows = [r for r in plan["s5_probe_rows"] if r["view"] in ("d1p5_loradiff64s1", "dm0p5_loradiff64s1")]
    accepted, texts = {}, {}
    for i, row in enumerate(rows):
        aid = f"ta-s5-{i}"
        akey = lp.accepted_key(ident["W7"], wave, row["leg"], row["engine"], row["stage"], row["shard"])
        accepted[akey] = aid
        vroot = lp.attempt_root_for(ident["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row.get("kind"))
        local = write_attempt(tmp_path / "roots" / aid, standup_id=row["standup_id"], W=plan["W"], attempt_id=aid, wave=wave)
        producer = {}
        for k, rel in lp.s5_window_rels(row):
            write_store(local, leg=row["leg"], cell=k["cell"], s=1, goals=[f"g{j}" for j in range(259 if (row["standup_id"], k["cell"]) != incomplete_on else 200)],
                        n_turns=1, view=row["view"], clamped_calls=(1 if (row["standup_id"], k["cell"]) == clamp_on else 0))
            argv = lp.s5_measure_argv(driver_python=sys.executable, tree=REPO, root=local, row=row, child=k, out=local / rel)
            assert M.main(argv[2:]) == 0, (row["standup_id"], k["cell"])
            text = (local / rel).read_text()
            texts[f"{vroot}/{rel}"] = text
            producer[rel] = {"sha256": lp._sha_text(text), "writer": lp.PRODUCER_WRITERS["s5_window_measurement"], "tree_sha": plan["W"], "rc": 0}
        texts[f"{vroot}/COMPLETE.json"] = json.dumps({"attempt_id": aid, "control_key": row["control_key"]})
        texts[f"{vroot}/RESULT"] = f"2026-09-18T00:00:00Z status=DONE rc=0 attempt={aid} standup={row['standup_id']} children=2 failed=0 S=10 elapsed_s=900\n"
        texts[f"{vroot}/modal_evidence.json"] = json.dumps({"attempt_id": aid, "readiness": {"problems": []}, "object_verification": {"ok": True},
                                                            "children": [{"rc": 0}, {"rc": 0}], "attested_max_num_seqs": 10, lp.PRODUCER_EVIDENCE_KEY: producer})
    return rows, accepted, texts


# --------------------------------------------------------------------------- the writer derives; the operator path is refused; the gate reads
class TestWindowDerivation:
    def _world(self, M, plan, rs: Path, tmp_path: Path, *, clamp_on=None, incomplete_on=None):
        rows, accepted, texts = s5_probe_world(plan, rs, tmp_path, clamp_on=clamp_on, incomplete_on=incomplete_on)
        ident = lp.smoke_identity(plan, "S5", SMOKE_WAVE)
        read_text = lp.volume_only(lambda path: texts.get(str(path)))
        getter = lp.RecordingGetter(ident["dicts"]["accepted"], accepted.get)
        return rows, getter, read_text, texts

    def test_window_derives_from_the_producers_bound_records_and_the_final_gate_reads_them(self, M, S5R, probe_plan, inputs, tmp_path):
        rs = tmp_path / "RS"
        rows, getter, read_text, texts = self._world(M, probe_plan, rs, tmp_path)
        plan_p = tmp_path / "plan.json"
        plan_p.write_text(json.dumps(probe_plan, default=str))
        ids = ",".join(r["standup_id"] for r in rows)
        rc = S5R.main(["window", "--rs", str(rs), "--plan", str(plan_p), "--rows", ids], read_text=read_text, accepted_get_for=lambda sid: getter)
        assert rc == 0
        doc = json.loads((rs / "smokes.json").read_text())
        mt = doc["S5"]["multiturn"]
        assert len(mt) == 8 and {(m["leg"], m["arm"]) for m in mt} == {(leg, a) for leg in lp.S5_PROBE_LEGS for a in (D1P5, DM0P5, "d1p5_loradiff64s1__none", "dm0p5_loradiff64s1__none")}
        assert all(m["clamped_calls"] == 0 and m["store_complete"] is True and m["produced_by"]["writer"] == "scripts/modal/seedrep_s5_measure.py" for m in mt)
        assert all(lp.window_measurement_problems(m) == [] for m in mt) and doc["S5"]["execution_wave_uuid"] == SMOKE_WAVE
        der = doc["S5"]["window_derivation"]
        assert der["schema"] == lp.SMOKE_PASS_SCHEMA and der["smoke_id"] == "S5" and sorted(der["accepted"].values()) == [f"ta-s5-{i}" for i in range(4)]
        srcs = der["checks"]["s5_window_measurement"]["sources"]
        assert len(srcs) == 8 and all("measurement" not in s and s["producer"]["writer"] == lp.PRODUCER_WRITERS["s5_window_measurement"] and s["clamped_calls"] == 0 for s in srcs)
        # the FINAL plan's gate over exactly these records
        for leg in lp.S5_PROBE_LEGS:
            g = lp.multiturn_window_gate(leg, 49152, mt, wave_uuid=SMOKE_WAVE)
            assert g["ok"] and g["n_measurements"] == 4, g["problems"]
        ro, pins = _Dose._peer(inputs)
        final = _plan(inputs, materialisation="final", s4=lp.S4_DEFAULTS, peer_roster=ro, peer_pins=pins)
        just = lp.window_justification(final, mt, execution_wave_uuid=SMOKE_WAVE)
        assert just["problems"] == [] and all(just["legs"][leg]["gate"]["ok"] for leg in lp.S5_PROBE_LEGS)
        assert final["multiturn_window_provenance"]["crescendo"]["s5_measurements"][0]["window_source"] == "measured_max_complete_store"
        # a second derivation of the same rows refuses the duplicate (leg, arm) without --replace; with it, the block is rewritten
        assert S5R.main(["window", "--rs", str(rs), "--plan", str(plan_p), "--rows", ids], read_text=read_text, accepted_get_for=lambda sid: getter) == 2
        assert S5R.main(["window", "--rs", str(rs), "--plan", str(plan_p), "--rows", ids, "--replace"], read_text=read_text, accepted_get_for=lambda sid: getter) == 0

    def test_operator_record_is_refused_by_name_and_rows_are_required(self, S5R, probe_plan, tmp_path, capsys):
        rs = tmp_path / "RS"
        wp = lp.smoke_wave_path(rs, "S5")
        wp.parent.mkdir(parents=True)
        wp.write_text(json.dumps({"smoke_id": "S5", "wave_uuid": SMOKE_WAVE}))
        plan_p = tmp_path / "plan.json"
        plan_p.write_text(json.dumps(probe_plan, default=str))
        rec_p = tmp_path / "measured.json"
        rec_p.write_text(json.dumps({"schema": lp.S5_WINDOW_MEASUREMENT_SCHEMA, "window_source": lp.S5_WINDOW_SOURCE, "leg": "crescendo", "arm": "none", "arm_kind": "parent",
                                     "model_key": lp.MODEL_KEY, "decode_policy_ref": POLICY_REF, "objects_measured": ["none"], "store_path": "x", "n_goals_completed": 259,
                                     "n_goals_expected": 259, "store_complete": True, "context_max": 9000, "clamped_calls": 0, "cap": 4096, "measured_utc": "2026-09-18T00:00:00Z",
                                     "wave_uuid": SMOKE_WAVE, "attempt_id": "ta-s5-0", "decode": "vendor"}))
        assert S5R.main(["window", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p)], read_text=lambda p: None, accepted_get_for=lambda sid: (lambda k: None)) == 2
        err = capsys.readouterr().err
        assert "operator-authored" in err and "--rows" in err
        assert not (rs / "smokes.json").exists()
        assert S5R.main(["window", "--rs", str(rs), "--plan", str(plan_p)], read_text=lambda p: None, accepted_get_for=lambda sid: (lambda k: None)) == 2
        assert "--rows" in capsys.readouterr().err
        src = (REPO / "scripts/modal/seedrep_s5_record.py").read_text()
        assert "OPERATOR_RECORD_REFUSAL" in src and 'for kind in ("reference", "bars")' in src

    def test_derivation_refuses_a_clamped_call_an_incomplete_store_and_a_missing_probe_cell(self, M, probe_plan, tmp_path):
        rs = tmp_path / "RS"
        rows, getter, read_text, texts = self._world(M, probe_plan, rs, tmp_path, clamp_on=(next(r["standup_id"] for r in rows_of(probe_plan, "actorattack", "d1p5_loradiff64s1")), CELL))
        ids = [r["standup_id"] for r in rows]
        with pytest.raises(ValueError, match="clamped call"):
            lp.derive_s5_window_measurements(rs=rs, plan=probe_plan, standup_ids=ids, accepted_get=getter, read_text=read_text)
        # the crescendo rows alone derive (their stores are clean)
        cres = [r["standup_id"] for r in rows if r["leg"] == "crescendo"]
        rec, ms = lp.derive_s5_window_measurements(rs=rs, plan=probe_plan, standup_ids=cres, accepted_get=getter, read_text=read_text)
        assert len(ms) == 4 and {m["arm"] for m in ms} >= set(PROBE)
        # one probe cell missing (one view's row only) -> refused by name
        with pytest.raises(ValueError, match="both probe cells"):
            lp.derive_s5_window_measurements(rs=rs, plan=probe_plan, standup_ids=cres[:1], accepted_get=getter, read_text=read_text)
        # an incomplete store (200 of 259 goals) -> refused as a floor
        rs2 = tmp_path / "RS2"
        rows2, getter2, read_text2, _ = self._world(M, probe_plan, rs2, tmp_path / "w2", incomplete_on=(cres[0], "none"))
        with pytest.raises(ValueError, match="not COMPLETE"):
            lp.derive_s5_window_measurements(rs=rs2, plan=probe_plan, standup_ids=cres, accepted_get=getter2, read_text=read_text2)
        # a producer record whose bytes were rewritten after binding -> refused (the sha the producer bound)
        rs3 = tmp_path / "RS3"
        rows3, getter3, read_text3, texts3 = self._world(M, probe_plan, rs3, tmp_path / "w3")
        key = next(k for k in texts3 if k.endswith(f"/audits/s5_window/crescendo/{CELL}__s1.json"))
        doc = json.loads(texts3[key])
        doc["context_max"] = 1
        texts3[key] = json.dumps(doc)
        with pytest.raises(ValueError, match="not the bytes the producer bound"):
            lp.derive_s5_window_measurements(rs=rs3, plan=probe_plan, standup_ids=cres, accepted_get=getter3, read_text=read_text3)
        # a non-probe row cannot be an S5 window source
        with pytest.raises(ValueError, match="not an S5 probe row"):
            lp.derive_s5_window_measurements(rs=rs3, plan=probe_plan, standup_ids=[probe_plan["standups"][0]["standup_id"]], accepted_get=getter3, read_text=read_text3)


class TestCertifyRederivesTheWindows:
    def test_certify_rederives_s5_windows_and_refuses_a_hand_edit_or_a_missing_derivation(self, S5R, probe_plan, tmp_path):
        rs = tmp_path / "RS"
        rows, accepted, texts = s5_probe_world(probe_plan, rs, tmp_path)
        ident = lp.smoke_identity(probe_plan, "S5", SMOKE_WAVE)
        read_text = lp.volume_only(lambda path: texts.get(str(path)))
        getter = lp.RecordingGetter(ident["dicts"]["accepted"], accepted.get)
        plan_p = tmp_path / "plan.json"
        plan_p.write_text(json.dumps(probe_plan, default=str))
        ids = ",".join(r["standup_id"] for r in rows)
        assert S5R.main(["window", "--rs", str(rs), "--plan", str(plan_p), "--rows", ids], read_text=read_text, accepted_get_for=lambda sid: getter) == 0
        doc = json.loads((rs / "smokes.json").read_text())
        # the block as written re-derives to itself (derived_utc aside)
        fresh = S5R.rederive_smokes(rs=rs, plan=probe_plan, draft=None, smokes_doc=doc, accepted_get_for=lambda sid: getter, read_text=read_text)
        assert {(m["leg"], m["arm"]): m for m in fresh["S5"]["multiturn"]} == {(m["leg"], m["arm"]): m for m in doc["S5"]["multiturn"]}
        assert fresh["S5"]["window_derivation"]["accepted"] == doc["S5"]["window_derivation"]["accepted"]
        # a hand edit after `window --rows` (a lower context_max on one record) -> refused by name
        edited = json.loads(json.dumps(doc))
        edited["S5"]["multiturn"][0]["context_max"] = 1
        with pytest.raises(ValueError, match="not what the probe attempts derive today"):
            S5R.rederive_smokes(rs=rs, plan=probe_plan, draft=None, smokes_doc=edited, accepted_get_for=lambda sid: getter, read_text=read_text)
        # a record added by hand (a fifth arm) -> refused
        added = json.loads(json.dumps(doc))
        added["S5"]["multiturn"].append({**doc["S5"]["multiturn"][0], "arm": "hand__none"})
        with pytest.raises(ValueError, match="not what the probe attempts derive today"):
            S5R.rederive_smokes(rs=rs, plan=probe_plan, draft=None, smokes_doc=added, accepted_get_for=lambda sid: getter, read_text=read_text)
        # windows with no derivation at all (the pre-W16s5 `--record` shape) -> refused by name
        bare = json.loads(json.dumps(doc))
        del bare["S5"]["window_derivation"]
        with pytest.raises(ValueError, match="no S5.window_derivation"):
            S5R.rederive_smokes(rs=rs, plan=probe_plan, draft=None, smokes_doc=bare, accepted_get_for=lambda sid: getter, read_text=read_text)
        # the producer's file changed on the volume since -> the derivation itself refuses (the bound sha)
        key = next(k for k in texts if k.endswith(f"/audits/s5_window/crescendo/{CELL}__s1.json"))
        texts[key] = json.dumps({**json.loads(texts[key]), "context_max": 2})
        with pytest.raises(ValueError, match="not the bytes the producer bound"):
            S5R.rederive_smokes(rs=rs, plan=probe_plan, draft=None, smokes_doc=doc, accepted_get_for=lambda sid: getter, read_text=read_text)

    def test_per_leg_recording_in_either_order_and_a_partial_replace_both_certify(self, S5R, probe_plan, tmp_path):
        """The recorder used to CONCATENATE source lists, so crescendo-then-actorattack recording (or a partial --replace) left
        `window_derivation.checks` in an order / with duplicates a fresh derivation never produces, and certify refused valid evidence. It now
        re-derives the complete union (sorted, deduplicated) on every call."""
        rs = tmp_path / "RS"
        rows, accepted, texts = s5_probe_world(probe_plan, rs, tmp_path)
        ident = lp.smoke_identity(probe_plan, "S5", SMOKE_WAVE)
        read_text = lp.volume_only(lambda path: texts.get(str(path)))
        getter = lp.RecordingGetter(ident["dicts"]["accepted"], accepted.get)
        plan_p = tmp_path / "plan.json"
        plan_p.write_text(json.dumps(probe_plan, default=str))
        cres = ",".join(r["standup_id"] for r in rows if r["leg"] == "crescendo")
        acto = ",".join(r["standup_id"] for r in rows if r["leg"] == "actorattack")
        run = lambda *a: S5R.main(["window", "--rs", str(rs), "--plan", str(plan_p), *a], read_text=read_text, accepted_get_for=lambda sid: getter)  # noqa: E731
        # (a) crescendo first, actorattack second -- the recorded derivation equals the fresh one (sorted ids, no duplicates) and certifies
        assert run("--rows", cres) == 0 and run("--rows", acto) == 0
        doc = json.loads((rs / "smokes.json").read_text())
        der = doc["S5"]["window_derivation"]
        assert der["standup_ids"] == sorted(r["standup_id"] for r in rows) and len(der["checks"]["s5_window_measurement"]["sources"]) == 8 and len(doc["S5"]["multiturn"]) == 8
        fresh = S5R.rederive_smokes(rs=rs, plan=probe_plan, draft=None, smokes_doc=doc, accepted_get_for=lambda sid: getter, read_text=read_text)
        assert fresh["S5"]["window_derivation"]["checks"] == der["checks"]
        # (b) a requested row already derived refuses without --replace; with it the whole set is re-derived once more -- no duplicate source, certifies
        assert run("--rows", cres) == 2
        assert run("--rows", cres, "--replace") == 0
        doc2 = json.loads((rs / "smokes.json").read_text())
        srcs = doc2["S5"]["window_derivation"]["checks"]["s5_window_measurement"]["sources"]
        assert len(srcs) == 8 and len({s_["path"] for s_ in srcs}) == 8 and len(doc2["S5"]["multiturn"]) == 8
        S5R.rederive_smokes(rs=rs, plan=probe_plan, draft=None, smokes_doc=doc2, accepted_get_for=lambda sid: getter, read_text=read_text)
        assert "merged" not in (REPO / "scripts/modal/seedrep_s5_record.py").read_text().split("def record_s5_windows")[1].split("def ")[0]


def rows_of(plan, leg, view):
    return [r for r in plan["s5_probe_rows"] if r["leg"] == leg and r["view"] == view]


def test_the_contract_vocabulary_is_pinned():
    assert lp.SMOKE_CHECKS["S5"] == ("s5_window_measurement",) and "s5_window_measurement" in lp.SMOKE_CHECK_NAMES
    assert lp.SMOKE_PASS_IDS == ("S1", "S1b", "S2", "S3", "S4") and lp.SMOKE_DERIVABLE_IDS == lp.SMOKE_PASS_IDS + ("S5",)
    assert lp.s5_window_rel("actorattack", "none", 1, stage="base", shard=0) == "audits/s5_window/actorattack/none__s1.json"
    assert lp.s5_window_rel("crescendo", "c", 2, stage="8192", shard=3) == "audits/s5_window/crescendo/c__s2__cap8192__shard03.json"
    assert int(lp.leg_table(lp.S4_DEFAULTS)["crescendo"]["items_per_arm_run"]) == 259 == int(lp.leg_table(lp.S4_DEFAULTS)["actorattack"]["items_per_arm_run"])
