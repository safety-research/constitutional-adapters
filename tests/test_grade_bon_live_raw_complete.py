"""F2 (second adversarial review, 2026-09-03): the live BoN grader must
refuse an uncovered slot. ``_curves_for_arm`` used to hand ``bon_curves``
``n_max=len(scored)`` and relaxed completeness for judge abstentions, so a
63/64 goal (a slot the served loop left UNCOVERED after a PromptOverWindow)
published a 63-sample curve with ``n_max=63``; ``bon_errors.jsonl`` was never
read. Now RAW completeness comes first — exactly ``cfg["bon"]["n_max"]`` raw
slots per expected goal and no typed error row — and a shortfall refuses
naming the goals; only then may a judge abstention drop a SCORE, the curve
staying ``n_max`` long with the CONFIGURED ``n_max`` stamped on it. CPU-only;
the grader module is loaded from scripts/ with a fake score cache."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _grade_mod():
    spec = importlib.util.spec_from_file_location(
        "grade_bon_live_raw", Path(__file__).resolve().parents[1] / "scripts" / "grade_bon_live.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Cache:
    """score cache stub: ``rubrics[(goal_id, response)] -> rubric | None``."""

    def __init__(self, mod, rubrics):
        self._mod = mod
        self._by_key = {mod._sha_key("sr", gid, r): rub for (gid, r), rub in rubrics.items()}

    def get(self, key):
        return self._by_key.get(key)


def _goals(n=2):
    return [SimpleNamespace(goal_id=f"g{i}", category="illegal", text=f"obj{i}") for i in range(n)]


def _responses(goals, n_max, short_goal=None, short_by=1):
    out = {}
    for g in goals:
        k = n_max - (short_by if g.goal_id == short_goal else 0)
        out[g.goal_id] = [f"{g.goal_id}-r{i}" for i in range(k)]
    return out


REFUSED = {"refused": 1, "convincing": 1, "specific": 1}
COMPLIED = {"refused": 0, "convincing": 5, "specific": 5}


def test_reproduced_63_of_64_goal_refuses():
    mod = _grade_mod()
    goals = _goals(2)
    resp = _responses(goals, 64, short_goal="g1")               # g1 has 63 slots
    with pytest.raises(SystemExit) as ei:
        mod.assert_bon_raw_complete(resp, [g.goal_id for g in goals], 64, arm="none")
    msg = str(ei.value)
    assert "INCOMPLETE" in msg and "g1=63/64" in msg and "g0" not in msg.split("slots")[1].split(";")[0]
    assert "esume" in msg
    # the curve builder refuses too (defensive), before any curve is built
    cache = _Cache(mod, {(gid, r): REFUSED for gid, rs in resp.items() for r in rs})
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        mod._curves_for_arm(goals, resp, cache, 64, 0.5, arm="none")


def _write_store(path, goals, n_max, *, skip=None, dup=None):
    """A bon_responses.jsonl with sample_idx 0..n_max-1 per goal; ``skip`` =
    (goal_id, sample_idx) to leave out, ``dup`` = (goal_id, sample_idx) to write
    twice (so the ROW count can be n_max while a slot is missing)."""
    rows = []
    for g in goals:
        for i in range(n_max):
            if skip == (g.goal_id, i):
                continue
            rows.append({"goal_id": g.goal_id, "sample_idx": i, "trial": 0,
                         "response": f"{g.goal_id}-r{i}"})
            if dup == (g.goal_id, i):
                rows.append({"goal_id": g.goal_id, "sample_idx": i, "trial": 0,
                             "response": f"{g.goal_id}-r{i}-dup"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from sisteer.eval.attacks.bon import load_bon_responses
    return load_bon_responses(path, trial=0)


def _error_row(gid, sidx, trial=0):
    return json.dumps({"goal_id": gid, "sample_idx": sidx, "trial": trial,
                       "error": "prompt_over_window"}) + "\n"


def test_historical_error_row_superseded_by_a_later_response_is_covered(tmp_path):
    """The re-check regression: bon_errors.jsonl is append-only, so a resume
    at a wider window that filled the slot left the old row behind and the
    first cut refused forever. A slot with a later successful response row is
    COVERED; the historical row is provenance (n_errors_superseded)."""
    mod = _grade_mod()
    goals = _goals(2)
    store = tmp_path / "bon_responses.jsonl"
    resp = _write_store(store, goals, 4)                       # full: every slot filled
    errors = tmp_path / mod.BON_ERRORS_FILE
    errors.write_text(_error_row("g0", 2))                      # the stale row
    rep = mod.assert_bon_raw_complete(resp, ["g0", "g1"], 4, arm="none",
                                      errors_path=errors, responses_path=store)
    assert rep == {"n_goals": 2, "n_raw_slots_expected": 8, "n_raw_slots_present": 8,
                   "n_errors_superseded": 1, "n_errors_active": 0}
    # a ledger row for a goal OUTSIDE the expected set, or another trial, is ignored
    errors.write_text(_error_row("zzz", 0) + _error_row("g0", 0, trial=1))
    rep = mod.assert_bon_raw_complete(resp, ["g0", "g1"], 4, arm="none",
                                      errors_path=errors, responses_path=store)
    assert rep["n_errors_superseded"] == 0 and rep["n_errors_active"] == 0
    assert mod.BON_ERRORS_FILE == "bon_errors.jsonl"
    # and the curve builder (main's path) passes on the same store
    cache = _Cache(mod, {(gid, r): REFUSED for gid, rs in resp.items() for r in rs})
    errors.write_text(_error_row("g0", 2))
    curves, n_unscored, n_dropped = mod._curves_for_arm(
        goals, resp, cache, 4, 0.5, arm="none", errors_path=errors, responses_path=store)
    assert len(curves) == 2 and n_unscored == 0 and all(c.n_max == 4 for c in curves)


def test_63_of_64_with_its_error_row_still_refuses(tmp_path):
    mod = _grade_mod()
    goals = _goals(2)
    store = tmp_path / "bon_responses.jsonl"
    resp = _write_store(store, goals, 64, skip=("g1", 40))    # the slot really is uncovered
    errors = tmp_path / mod.BON_ERRORS_FILE
    errors.write_text(_error_row("g1", 40))
    with pytest.raises(SystemExit) as ei:
        mod.assert_bon_raw_complete(resp, ["g0", "g1"], 64, arm="none",
                                    errors_path=errors, responses_path=store)
    msg = str(ei.value)
    assert "INCOMPLETE" in msg and "g1=63/64" in msg and "missing [40]" in msg
    assert "1 ACTIVE typed error slot" in msg and "prompt_over_window" in msg
    assert "0 historical row(s) superseded" in msg


def test_full_row_count_but_error_slot_absent_refuses(tmp_path):
    """The impossible pairing, checked anyway: 64 ROWS for the goal, but the
    errored slot's own row is absent (a duplicate row elsewhere makes up the
    count) — the error is active and the slot set is short."""
    mod = _grade_mod()
    goals = _goals(2)
    store = tmp_path / "bon_responses.jsonl"
    loaded = _write_store(store, goals, 64, skip=("g0", 7), dup=("g0", 8))
    # the ordered loader collapses the duplicate, so this pairing cannot come
    # out of load_bon_responses (63 here) — the check must hold at the KEY
    # level regardless, so hand it a 64-long list against the 63-slot file
    assert len(loaded["g0"]) == 63
    resp = {"g0": [f"g0-r{i}" for i in range(64)], "g1": loaded["g1"]}
    errors = tmp_path / mod.BON_ERRORS_FILE
    errors.write_text(_error_row("g0", 7))
    with pytest.raises(SystemExit) as ei:
        mod.assert_bon_raw_complete(resp, ["g0", "g1"], 64, arm="none",
                                    errors_path=errors, responses_path=store)
    msg = str(ei.value)
    assert "missing [7]" in msg and "1 ACTIVE typed error slot" in msg
    # without the ledger the missing slot alone still refuses
    with pytest.raises(SystemExit, match="missing \\[7\\]"):
        mod.assert_bon_raw_complete(resp, ["g0", "g1"], 64, arm="none",
                                    responses_path=store)


def test_64_of_64_with_one_judge_abstention_passes_with_n_max_64():
    mod = _grade_mod()
    goals = _goals(2)
    resp = _responses(goals, 64)
    rubrics = {(gid, r): REFUSED for gid, rs in resp.items() for r in rs}
    rubrics[("g1", "g1-r10")] = None                 # the judge abstained on one slot
    rubrics[("g1", "g1-r20")] = COMPLIED             # a later success
    cache = _Cache(mod, rubrics)
    curves, n_unscored, n_dropped = mod._curves_for_arm(goals, resp, cache, 64, 0.5, arm="none")
    assert n_unscored == 1 and n_dropped == 0
    by = {c.goal_id: c for c in curves}
    assert set(by) == {"g0", "g1"}
    for c in curves:
        assert c.n_max == 64 and len(c.best_so_far) == 64      # NEVER len(scored)
    g1 = by["g1"]
    assert g1.n_star == 21                                     # 1-based, on the scored slot
    assert g1.best_so_far[10] == g1.best_so_far[9]             # the abstention carries forward
    assert g1.best_so_far[20] == 1.0 and g1.best_so_far[-1] == 1.0
    assert by["g0"].n_star is None
    # fully-scored goals went through bon_curves itself with the configured n_max
    from sisteer.eval.attacks.bon import bon_curves
    ref = bon_curves([goals[0]], {"g0": resp["g0"]},
                     lambda g, r: 0.0, n_max=64, threshold=0.5)[0]
    assert by["g0"].best_so_far == ref.best_so_far and by["g0"].n_max == ref.n_max == 64


def test_all_abstained_goal_is_dropped_and_counted():
    mod = _grade_mod()
    goals = _goals(2)
    resp = _responses(goals, 3)
    rubrics = {(gid, r): REFUSED for gid, rs in resp.items() for r in rs}
    for r in resp["g1"]:
        rubrics[("g1", r)] = None
    curves, n_unscored, n_dropped = mod._curves_for_arm(goals, resp, _Cache(mod, rubrics), 3, 0.5)
    assert [c.goal_id for c in curves] == ["g0"] and n_unscored == 3 and n_dropped == 1


def test_main_checks_raw_completeness_before_grading():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "grade_bon_live.py").read_text()
    load_at = src.index("responses = load_bon_responses(jsonl, trial=0)")
    check_at = src.index("raw_by_arm[arm] = assert_bon_raw_complete(")
    grade_at = src.index("_grade_tasks_concurrently(all_tasks, cache, client, model, args.concurrency)")
    assert load_at < check_at < grade_at
    assert "n_max=len(scored)" not in src
    assert "responses_path=jsonl" in src and '"raw_completeness": raw_by_arm.get(arm)' in src
