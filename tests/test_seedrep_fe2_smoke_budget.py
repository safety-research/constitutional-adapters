"""qwen-seedrep-native-battery: smoke rows get a smoke-only ITEM budget; production rows are refused any budget flag.

Previously: `smoke_row` bounds children and the container timeout only, so every BoN smoke child still ran the
production budget `goals=100 x n_max=75` at concurrency 5 -- 7,500 draws per arm, 3-7 h per child (the earlier bon row) --
and the misalign smoke 490 items per cell (~2.3 h); the gate's PASS needs EVERY row, so both sat on the critical path to
`certify` / `--go`. Now `smoke_row` (the plan's ONE place a smoke differs from production) gives a smoke's BoN children
`--limit-goals 10` (the driver's existing SMOKE ONLY flag; 750 draws per arm) and its misalign children `--limit-items 10`
(a NEW runner flag: the first N items per behaviour in corpus order, the planned set recorded in the store's own meta.json),
records the budget in `row["smoke"]["budget"]`, keeps the reference rows at the FULL shape by name, and `validate_plan`
refuses a budget flag on a production row and a smoke BoN / misalign child without it. A bounded store must still read
COMPLETE on its own planned set -- verified here through the REAL readers (`audit_common.read_bon_store` /
`read_misalign_store`, read-only) over stores the REAL generators wrote.

Known-bad at dfbd956af: `smoke_row` added no budget flag (every S3 bon / misalign child carried the production argv verbatim);
`misalign.runner` had no `--limit-items` (its parser exits 2 on the flag); `validate_plan` accepted a production child carrying
`--limit-goals`.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_fs_bon_labels as fsb  # noqa: E402  (the fake proxy + the child's real wiring; its `tre` / `tbs` / `bs` / `bon`)
from sisteer.capeval.misalign import runner as R  # noqa: E402
from sisteer.eval import request_evidence_client as rev  # noqa: E402
from test_misalign_runner import CELLS, ITEMS, QWEN_IDENTITY, _rows  # noqa: E402
from test_seedrep_fs_misalign_labels import _UniqueIdHttp  # noqa: E402  (the stub transport that accepts headers)

W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"
SMOKE_WAVE = "11111111-2222-4333-8444-555555555555"
S3_BON, S3_MISALIGN = "bon__s1__base__shard00", "misalign__s1__base__shard00"
OTHER_S3_ROWS = ("controls__s1__base__shard00", "petri__s1__base__shard00", "gsm8k__s1__base__shard00", "pasttense__s1__base__shard00")
FULL22_DIR = REPO / "scripts" / "sdfbase" / "full22"
#: the argv placeholders the app resolves at spawn (`lp.SpawnContext`), made concrete so the REAL parsers can type-check them
SUBS = {lp.DRIVER_PY_PLACEHOLDER: sys.executable, lp.ATTEMPT_ROOT_PLACEHOLDER: "/tmp/attempt", lp.PROXY_URL_PLACEHOLDER: "http://127.0.0.1:1",
        lp.ROUTING_TABLE_PLACEHOLDER: "/tmp/routing_table.json", lp.CONCURRENCY_PLACEHOLDER: "5"}
BON_PARSE = ("import json, sys; sys.path.insert(0, sys.argv[1]); import bon_served_run as b; a = b.build_parser().parse_args(sys.argv[2:]); "
             "print(json.dumps({'limit_goals': a.limit_goals, 'defense': a.defense, 'max_concurrency': a.max_concurrency, 'sisteer_loaded': "
             "any(m.startswith('sisteer') for m in sys.modules)}))")
MIS_PARSE = ("import json, sys; from sisteer.capeval.misalign.runner import _build_parser; a = _build_parser().parse_args(sys.argv[1:]); "
             "print(json.dumps({'limit_items': a.limit_items, 'only_cells': a.only_cells, 'behaviors': a.behaviors, 'n': a.n}))")


@pytest.fixture(autouse=True)
def _hook_off(monkeypatch):
    monkeypatch.delenv(rev.ENV_DIR, raising=False)
    monkeypatch.delenv(rev.ENV_EXTRA, raising=False)
    monkeypatch.delenv(rev.ENV_SEED_BASE, raising=False)
    rev.reset_env_cache()
    yield
    rev.reset_env_cache()


@pytest.fixture(scope="module")
def draft():
    inputs = lp.load_inputs(REPO)
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                         arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z",
                         config_private_runs=inputs.get("config_private_runs"))


def _rows_of(plan):
    return list(plan["standups"]) + list(plan["rungs"]) + list(plan.get("s5_probe_rows") or [])


def _row(plan, standup_id):
    hits = [r for r in _rows_of(plan) if r["standup_id"] == standup_id]
    assert len(hits) == 1, standup_id
    return hits[0]


def _smoke(plan, standup_id, smoke_id, max_children=5):
    return lp.smoke_row(_row(plan, standup_id), lp.smoke_identity(plan, smoke_id, SMOKE_WAVE), max_children=max_children, max_timeout_s=4 * 3600)


def _concrete(argv):
    out = []
    for a in argv:
        for k, v in SUBS.items():
            a = a.replace(k, v)
        out.append(a)
    return out


def _fresh(code, args):
    """The REAL parser in a FRESH interpreter (`python -c`), the tree's src / scripts on its path -- nothing this process imported leaks in."""
    r = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=300, cwd=str(REPO),
                       env={**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/scripts"})
    assert r.returncode == 0, (r.stdout + r.stderr)[-1200:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _has(argv, flag):
    return lp._argv_flag_count(argv, flag)


# --------------------------------------------------------------------------- the budget table, keyed on the smoke id like the adapter roles

def test_budget_is_keyed_on_the_smoke_id_s1_to_s4_item_budget_s5_full_shape_unknown_refused():
    for sid in lp.SMOKE_PASS_IDS:                                     # S1, S1b, S2, S3, S4
        b = lp.smoke_budget(sid)
        assert b["kind"] == "item_budget" and (b["bon_limit_goals"], b["misalign_limit_items"]) == (10, 10) == (lp.SMOKE_BON_GOALS, lp.SMOKE_MISALIGN_ITEMS)
        assert "D107" in b["rule"] and "--limit-goals 10" in b["rule"] and "--limit-items 10" in b["rule"]
    assert lp.smoke_budget("S1b") == lp.smoke_budget("S1B") == lp.smoke_budget("s1b")          # the adapter-role dispatch, case-insensitive
    s5 = lp.smoke_budget(lp.S5_WINDOW_SMOKE_ID)
    assert s5["kind"] == "full_shape" and s5["bon_limit_goals"] is None and s5["misalign_limit_items"] is None
    assert s5["rule"].startswith("full_shape (S5 reference") and "D103" in s5["rule"]
    assert lp.SMOKE_BUDGET_FULL_SHAPE_IDS == ("S5",) and set(lp.SMOKE_BUDGET_FLAGS) == {"--limit-goals", "--limit-items"}
    with pytest.raises(ValueError, match="declares no §8 adapter role"):
        lp.smoke_budget("S9")
    # 10 goals x the plan's n_max 75 = 750 draws per arm (the ruling's ~20-40 min at ~43 draws/min), against 7,500 at the production 100
    assert lp.SMOKE_BON_GOALS * 75 == 750 and 100 * 75 == 7500


# --------------------------------------------------------------------------- (a) the S3 bon smoke: --limit-goals 10, the real driver parser

def test_a_s3_bon_smoke_children_carry_limit_goals_10_and_parse_on_the_real_driver_parser_in_a_fresh_interpreter(draft):
    prod = _row(draft, S3_BON)
    assert prod["children"] and all(not _has(k["argv"], "--limit-goals") for k in prod["children"])      # production: never
    assert lp.smoke_budget_problems(prod) == []
    srow = _smoke(draft, S3_BON, "S3")
    assert srow["smoke"]["budget"] == lp.smoke_budget("S3") and set(srow["smoke"]["budget"]) == {"kind", "bon_limit_goals", "misalign_limit_items", "rule"}
    assert srow["smoke"]["smoke_id"] == "S3" and srow["smoke"]["adapter_role"]["role"] == "S3"          # the W15d keys are untouched
    assert len(srow["children"]) == 5
    for k in srow["children"]:
        assert k["eval_leg"] == "bon" and lp._flag_values(k["argv"], "--limit-goals") == ["10"] and _has(k["argv"], "--limit-goals") == 1
        assert k["argv"][:-2] == next(p for p in prod["children"] if (p["arm"], p["s"]) == (k["arm"], k["s"]))["argv"]     # appended, nothing else moved
        assert not _has(k["argv"], "--limit-items")
        got = _fresh(BON_PARSE, [str(FULL22_DIR), *_concrete(k["argv"])[2:]])                            # the REAL bon_served_run.build_parser
        assert got == {"limit_goals": 10, "defense": k["cell"], "max_concurrency": 5, "sisteer_loaded": False}, got
    assert lp.smoke_budget_problems(srow) == []
    assert all(not _has(k["argv"], "--limit-goals") for k in prod["children"])                          # the production row is untouched (deep copy)


def test_a_the_s5_bon_reference_row_keeps_the_full_shape_no_flag_budget_full_shape(draft):
    srow = _smoke(draft, S3_BON, "S5", max_children=64)              # `--smoke-id S5 --smoke-max-children 64`: the FULL-shape reference
    assert srow["smoke"]["budget"]["kind"] == "full_shape" and srow["smoke"]["budget"] == lp.smoke_budget("S5")
    assert len(srow["children"]) == len(_row(draft, S3_BON)["children"]) == 8
    for k in srow["children"]:
        assert not _has(k["argv"], "--limit-goals") and not _has(k["argv"], "--limit-items")
        assert k["argv"] == next(p for p in _row(draft, S3_BON)["children"] if (p["arm"], p["s"]) == (k["arm"], k["s"]))["argv"]
    assert lp.smoke_budget_problems(srow) == []
    mis5 = _smoke(draft, S3_MISALIGN, "S5", max_children=64)
    assert mis5["smoke"]["budget"]["kind"] == "full_shape" and all(not _has(k["argv"], "--limit-items") for k in mis5["children"])
    assert lp.smoke_budget_problems(mis5) == []


# --------------------------------------------------------------------------- (b) the S3 misalign smoke: --limit-items 10, the real runner parser

def test_b_s3_misalign_smoke_children_carry_limit_items_10_and_parse_on_the_real_runner_parser_in_a_fresh_interpreter(draft):
    prod = _row(draft, S3_MISALIGN)
    assert prod["children"] and all(not _has(k["argv"], "--limit-items") for k in prod["children"])
    srow = _smoke(draft, S3_MISALIGN, "S3")
    assert srow["smoke"]["budget"] == lp.smoke_budget("S3")
    for k in srow["children"]:
        assert k["eval_leg"].startswith("misalign_") and lp._flag_values(k["argv"], "--limit-items") == ["10"] and _has(k["argv"], "--limit-items") == 1
        assert k["argv"][1:3] == ["-m", "sisteer.capeval.misalign.runner"] and not _has(k["argv"], "--limit-goals")
        got = _fresh(MIS_PARSE, _concrete(k["argv"])[3:])                                                # the REAL misalign.runner._build_parser
        assert got == {"limit_items": 10, "only_cells": k["cell_harness"], "behaviors": [k["behavior"]], "n": 10}, got
    assert lp.smoke_budget_problems(srow) == []
    assert all(not _has(k["argv"], "--limit-items") for k in prod["children"])


@pytest.mark.parametrize("standup_id", OTHER_S3_ROWS)
def test_other_legs_smoke_rows_carry_no_budget_flag_and_still_record_the_budget(draft, standup_id):
    """Static, controls, GSM8K, Petri (and the multi-turn legs) are bounded by children / goal shards already: no flag, the budget recorded."""
    prod = _row(draft, standup_id)
    srow = _smoke(draft, standup_id, "S3" if standup_id != "pasttense__s1__base__shard00" else "S1")
    assert srow["smoke"]["budget"]["kind"] == "item_budget"
    assert srow["children"] and len(srow["children"]) == min(5, len(prod["children"]))
    for k, p in zip(srow["children"], prod["children"]):              # the first max_children children, in order (static: one per shard)
        assert k.get("argv") is None or (not _has(k["argv"], "--limit-goals") and not _has(k["argv"], "--limit-items"))
        assert k["argv"] == p["argv"] and (k["arm"], k["s"]) == (p["arm"], p["s"])         # untouched
    assert lp.smoke_budget_problems(srow) == []


def test_every_production_row_of_the_draft_is_clean_and_every_s3_smoke_row_derives_clean(draft):
    assert all(lp.smoke_budget_problems(r) == [] for r in _rows_of(draft))
    assert not [p for p in lp.validate_plan(draft) if "budget" in p or "--limit-" in p]
    for sid in ("S3", "S5"):
        for standup_id in (S3_BON, S3_MISALIGN, *OTHER_S3_ROWS):
            assert lp.smoke_budget_problems(_smoke(draft, standup_id, sid, max_children=64)) == [], (sid, standup_id)


# --------------------------------------------------------------------------- (c) validate_plan: production refused any flag; a smoke child refused without it

def _inject(plan, standup_id, flag, value, child=0):
    plan = copy.deepcopy(plan)
    _row(plan, standup_id)["children"][child]["argv"] += [flag, str(value)]
    return plan


def test_c_validate_plan_refuses_a_production_row_carrying_either_budget_flag(draft):
    baseline = [p for p in lp.validate_plan(draft) if "budget" in p or "--limit-" in p]
    assert baseline == []
    for standup_id, flag, value in ((S3_BON, "--limit-goals", 10), (S3_MISALIGN, "--limit-items", 10), (S3_MISALIGN, "--limit-goals", 3),
                                    ("controls__s1__base__shard00", "--limit-items", 10), ("bon__arditi__base__shard00", "--limit-goals", 1)):
        bad = _inject(draft, standup_id, flag, value)
        probs = [p for p in lp.validate_plan(bad) if flag in p]
        assert len(probs) == 1, (standup_id, flag, probs)
        assert probs[0].startswith(f"{standup_id}: a production row's child") and "smoke-only budget flag" in probs[0] and "D107" in probs[0] and repr(str(value)) in probs[0]
        assert lp.smoke_budget_problems(_row(bad, standup_id)) == probs
    # the `--flag=value` spelling is the same refusal
    bad = copy.deepcopy(draft)
    _row(bad, S3_BON)["children"][1]["argv"].append("--limit-goals=10")
    assert any("--limit-goals" in p and "production row" in p for p in lp.validate_plan(bad))
    # ... and `smoke_row` refuses to budget such a row rather than stacking a second flag on it
    with pytest.raises(ValueError, match="already carries the smoke-only budget flag"):
        lp.smoke_row(_row(_inject(draft, S3_BON, "--limit-goals", 10), S3_BON), lp.smoke_identity(draft, "S3", SMOKE_WAVE), max_children=5, max_timeout_s=3600)


def test_c_a_smoke_bon_or_misalign_child_without_its_budget_or_with_a_wrong_one_is_refused_by_name(draft):
    for standup_id, flag, want in ((S3_BON, "--limit-goals", 10), (S3_MISALIGN, "--limit-items", 10)):
        srow = _smoke(draft, standup_id, "S3")
        stripped = copy.deepcopy(srow)
        stripped["children"][2]["argv"] = stripped["children"][2]["argv"][:-2]           # the pre-fix shape on ONE child: the production budget
        [p] = lp.smoke_budget_problems(stripped)
        k = stripped["children"][2]
        assert p.startswith(f"{stripped['standup_id']}: smoke S3 child {k['arm']} s{k['s']} must carry {flag} {want} exactly once") and "found 0 occurrence" in p
        wrong = copy.deepcopy(srow)
        wrong["children"][0]["argv"][-1] = str(want * 10)                                  # the flag with another value
        [p] = lp.smoke_budget_problems(wrong)
        assert f"must carry {flag} {want} exactly once" in p and f"value {str(want * 10)!r}" in p
        twice = copy.deepcopy(srow)
        twice["children"][0]["argv"] += [flag, str(want)]
        [p] = lp.smoke_budget_problems(twice)
        assert "found 2 occurrence" in p
        # the recorded budget must be the one the smoke id derives
        tampered = copy.deepcopy(srow)
        tampered["smoke"]["budget"] = {**tampered["smoke"]["budget"], "bon_limit_goals": 100}
        assert any("records budget" in p and "its smoke id derives" in p for p in lp.smoke_budget_problems(tampered))
        # through validate_plan: a plan that carries the stripped smoke row names the budget problem among its problems
        plan = copy.deepcopy(draft)
        plan["standups"].append(stripped)
        assert any(f"must carry {flag} {want} exactly once" in p for p in lp.validate_plan(plan))
    # an S5 reference row that DID get a budget flag is refused: it keeps the FULL shape
    s5 = _smoke(draft, S3_BON, "S5", max_children=64)
    s5["children"][0]["argv"] += ["--limit-goals", "10"]
    [p] = lp.smoke_budget_problems(s5)
    assert "carries --limit-goals '10': an S5 reference row keeps the FULL production shape (D103 / D104)" in p
    # a budget flag on a child of a leg the budget does not cover is refused too
    ctl = _smoke(draft, "controls__s1__base__shard00", "S3")
    ctl["children"][0]["argv"] += ["--limit-items", "10"]
    [p] = lp.smoke_budget_problems(ctl)
    assert "the controls leg is bounded by children / goal shards, never by --limit-items" in p
    # a smoke id with no role derives no budget
    stray = _smoke(draft, S3_BON, "S3")
    stray["smoke"]["smoke_id"] = "S9"
    assert lp.smoke_budget_problems(stray) == [f"{stray['standup_id']}: " + "smoke id 'S9' declares no §8 adapter role (one of ['S1', 'S1B', 'S2', 'S3', 'S4', 'S5']): a smoke row's "
                                               "expected_reused follows its role, never the production expectation (W15d, launch F-L(d))"]


# --------------------------------------------------------------------------- (d) the runner: --limit-items N, deterministic, meta declares the planned set

def _corpus():
    items, _rev = R.load_items("agentic", ["leaking", "murder"])
    return items


def _planned(items, n):
    """The rule stated independently of the helper: the first n of each behaviour, merged in corpus order."""
    first = {b: [i["item_id"] for i in items if i["behavior"] == b][:n] for b in ("leaking", "murder")}
    return [i["item_id"] for i in items if i["item_id"] in set(first["leaking"]) | set(first["murder"])]


def test_d_limit_items_helper_is_the_head_per_behaviour_in_corpus_order_not_sorted_and_refuses_nonpositive():
    items = [{"item_id": "z_murder", "behavior": "murder"}, {"item_id": "b_leak", "behavior": "leaking"}, {"item_id": "a_leak", "behavior": "leaking"},
             {"item_id": "y_murder", "behavior": "murder"}, {"item_id": "c_leak", "behavior": "leaking"}]
    kept, before, after = R.limit_items_per_behavior(items, 2)
    assert [i["item_id"] for i in kept] == ["z_murder", "b_leak", "a_leak", "y_murder"]           # corpus order, first 2 per behaviour, NOT sorted
    assert before == {"murder": 2, "leaking": 3} and after == {"murder": 2, "leaking": 2}
    assert R.limit_items_per_behavior(items, 2) == R.limit_items_per_behavior(list(items), 2)      # deterministic
    kept1, _b, after1 = R.limit_items_per_behavior(items, 1)
    assert [i["item_id"] for i in kept1] == ["z_murder", "b_leak"] and after1 == {"murder": 1, "leaking": 1}
    kept9, _b, _a = R.limit_items_per_behavior(items, 9)
    assert kept9 == [dict(i) for i in items]                                                          # a cap above the corpus keeps everything
    em8q, _b, _a = R.limit_items_per_behavior(R.em8q_items(), 3)
    assert len(em8q) == 3 and all(i["behavior"] is None for i in em8q) and _b == {"-": 8}           # em8q: one group
    for bad in (0, -1, True, 2.0):
        with pytest.raises(R.MisalignRunnerError, match="limit_items must be a positive int"):
            R.limit_items_per_behavior(items, bad)
    assert R.LIMIT_ITEMS_RULE.startswith("the first N items per behaviour in corpus order")
    # the real corpus: the planned set of `--limit-items 2` is the first two of each behaviour, in the artifact's order
    corpus = _corpus()
    kept, before, after = R.limit_items_per_behavior(corpus, 2)
    assert [i["item_id"] for i in kept] == _planned(corpus, 2) and len(kept) == 4 and after == {"leaking": 2, "murder": 2}
    assert before["leaking"] > 2 and before["murder"] > 2


def _cli(tmp_path, monkeypatch, out, *extra):
    http = _UniqueIdHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    monkeypatch.setattr(R, "HttpxJson", lambda api_key, timeout_s: http)
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(CELLS))
    # the plan's misalign child argv (`_misalign_children`), the placeholders concrete: one cell, one behaviour... here BOTH behaviours
    # so the per-behaviour head is visible in one store
    argv = ["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells), "--family", "agentic", "--out", str(out), "--concurrency", "4",
            "--n", "1", "--behaviors", "leaking", "murder", "--max-tokens", "64", "--sampler", "vendor", "--seed-policy", "per_response",
            "--seed-base", "1", "--only-cells", "none", "--timeout-s", "900", "--api-key", "petri-proxy", *extra]
    assert R.main(argv) == 0
    return http


def test_d_runner_limit_items_2_runs_exactly_two_items_per_behaviour_deterministically_and_meta_declares_the_planned_set(tmp_path, monkeypatch, capsys):
    """The CLI on the REAL corpus over the stub transport: `--limit-items 2` draws exactly the first two leaking + the first two
    murder items (corpus order), meta.json declares exactly that planned set (`item_ids`) and records the limit in `run`, a second
    run plans the identical set, and the REAL `read_misalign_store` (read-only) judges the limited store COMPLETE on it."""
    corpus = _corpus()
    planned = _planned(corpus, 2)
    http = _cli(tmp_path, monkeypatch, tmp_path / "out", "--limit-items", "2")
    out = capsys.readouterr().out
    assert f"[misalign.runner] SMOKE: --limit-items 2 -> 4 of {len(corpus)} items (leaking: 2/" in out and "never a wave result" in out
    assert len(http.posts) == 4                                                                     # 4 items x n=1, the ONE cell selected
    rows = _rows(tmp_path / "out" / "none" / "responses.jsonl")
    assert [r["item_id"] for r in sorted(rows, key=lambda r: planned.index(r["item_id"]))] == planned
    assert {r["behavior"] for r in rows} == {"leaking", "murder"} and all(r["sample_idx"] == 0 for r in rows)
    meta = json.loads((tmp_path / "out" / "none" / "meta.json").read_text())
    assert meta["item_ids"] == planned and meta["items"] == 4 and meta["n_samples"] == 1 and meta["behaviors"] == ["leaking", "murder"]
    assert meta["run"]["limit_items"] == 2 and meta["run"]["limit_items_rule"] == R.LIMIT_ITEMS_RULE
    assert meta["run"]["items_after_limit"] == {"leaking": 2, "murder": 2}
    assert meta["run"]["items_before_limit"] == {"leaking": sum(1 for i in corpus if i["behavior"] == "leaking"),
                                                 "murder": sum(1 for i in corpus if i["behavior"] == "murder")}
    assert meta["run"]["items_before_limit"]["leaking"] > 2 and meta["run"]["only_cells"] == ["none"]
    assert meta["present"] == 4 and meta["finished"] is not None
    # deterministic: a second run into a fresh root plans the identical set and draws the identical pairs
    _cli(tmp_path, monkeypatch, tmp_path / "out2", "--limit-items", "2")
    meta2 = json.loads((tmp_path / "out2" / "none" / "meta.json").read_text())
    assert meta2["item_ids"] == meta["item_ids"] == planned
    assert {(r["item_id"], r["sample_idx"]) for r in _rows(tmp_path / "out2" / "none" / "responses.jsonl")} == {(r["item_id"], r["sample_idx"]) for r in rows}
    # THE READER (audit_common.read_misalign_store, read-only): completeness is judged on meta's item_ids x n_samples -- the PLANNED set
    for behavior in ("leaking", "murder"):
        read = ac.read_misalign_store(tmp_path / "out" / "none", behavior=behavior, arm="none")
        assert read.complete is True, read.completeness_note
        assert read.completeness_note == "every declared pair stored once (item_ids (4) × n_samples (1))" and read.n_rows == 2
        assert read.filed == {(ac.misalign_task(behavior), i) for i in planned if i in {r["item_id"] for r in rows if r["behavior"] == behavior}}
    # a resume onto the limited store under the SAME limit sends nothing more
    http3 = _cli(tmp_path, monkeypatch, tmp_path / "out", "--limit-items", "2")
    assert http3.posts == []


def test_d_without_the_flag_the_runner_is_as_before_and_the_flag_is_refused_on_a_rerun_and_below_zero(tmp_path, monkeypatch):
    http = _UniqueIdHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    # run_family direct: no limit -> every item, `run` carries no limit keys (the exact-`run` pins of test_misalign_runner hold)
    R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 1, tmp_path / "off", concurrency=2, max_tokens=64, http=http, items=ITEMS,
                 suite_revision="agentic.test@000000000000", run_extra={"k": "v"})
    meta = json.loads((tmp_path / "off" / "none" / "meta.json").read_text())
    assert meta["run"] == {"k": "v"} and meta["item_ids"] == [i["item_id"] for i in ITEMS]
    # limit_items=1 on the two synthetic items (one per behaviour) keeps both; the run block records it
    R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 1, tmp_path / "lim", concurrency=2, max_tokens=64, http=http, items=ITEMS,
                 suite_revision="agentic.test@000000000000", limit_items=1)
    meta = json.loads((tmp_path / "lim" / "none" / "meta.json").read_text())
    assert meta["run"] == {"limit_items": 1, "limit_items_rule": R.LIMIT_ITEMS_RULE, "items_before_limit": {"leaking": 1, "murder": 1},
                           "items_after_limit": {"leaking": 1, "murder": 1}}
    with pytest.raises(R.MisalignRunnerError, match="limit_items rides a base run only"):
        R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 1, tmp_path / "rerun", http=http, items=ITEMS, suite_revision="r",
                     limit_items=1, only={"none": {(ITEMS[0]["item_id"], 0)}})
    with pytest.raises(R.MisalignRunnerError, match="limit_items must be a positive int"):
        R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 1, tmp_path / "zero", http=http, items=ITEMS, suite_revision="r", limit_items=0)
    # the CLI: refused by name beside --rerun-truncated-from and below zero, before any transport is touched
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(CELLS))
    base = ["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells), "--family", "agentic", "--n", "1", "--out", str(tmp_path / "x")]
    with pytest.raises(R.MisalignRunnerError, match="never rides --rerun-truncated-from"):
        R.main([*base, "--limit-items", "2", "--rerun-truncated-from", str(tmp_path / "src")])
    with pytest.raises(R.MisalignRunnerError, match="--limit-items must be >= 0"):
        R.main([*base, "--limit-items", "-1"])
    assert R._build_parser().parse_args([*base]).limit_items == 0                               # 0 = the full set (the default)


# --------------------------------------------------------------------------- (e) BoN: n_planned over the CAPPED goals; the reader says complete; bon_curves refuses it as a wave result

def test_e_a_limited_bon_run_plans_capped_goals_x_n_max_reads_complete_and_is_refused_as_a_wave_result(tmp_path):
    """The REAL `bon_generate_served` over the child's real client / defense / probe on the loopback fake proxy, with the driver's exact
    head-cap (`bon_served_run.main`: `goals = goals[: args.limit_goals]`) -- fs2's `units.n_planned` is computed from the goal list
    the generator RECEIVES (`_bon_generate_served_async`: `len(goals) * n_max`), so a capped run plans capped x n_max, covers it, and
    `read_bon_store` (read-only) says COMPLETE; while the aggregate's `bon_curves` over the config's FULL goal set refuses the
    capped root by the first uncovered goal (the driver docstring's "a capped run is never a wave result")."""
    bs, bon, tre, tbs = fsb.bs, fsb.bon, fsb.tre, fsb.tbs
    proxy = tre._FakeProxy()
    try:
        client, defense, probe = fsb._child_wiring(proxy)
        goals, limit = fsb._goals(5), 2
        limited = goals[:limit]                                                                # bon_served_run.main's cap, verbatim
        out_dir = tmp_path / "bon" / fsb.ARM
        jsonl = bs.bon_generate_served(defense, limited, out_dir=out_dir, routes=tbs._routes(tmp_path), n_max=fsb.N_MAX, max_new_tokens=fsb.CAP,
                                       max_concurrency=2, steering_arms=("r4",), steering_probe=probe, decode=fsb.PRESET,
                                       decode_seed_policy="replicate_v1", decode_seed_base=fsb.S, request_headers_fn=None)
        rows = [json.loads(l) for l in Path(jsonl).read_text().splitlines() if l.strip()]
        n_draws = limit * fsb.N_MAX
        assert len(rows) == n_draws and {r["goal_id"] for r in rows} == {"g0", "g1"}
        u = json.loads((out_dir / bs.BON_SERVED_MANIFEST).read_text())["units"]
        assert (u["n_planned"], u["n_done"], u["n_uncovered"], u["n_written"]) == (n_draws, n_draws, 0, n_draws)     # planned = CAPPED goals x n_max
        assert u["n_planned"] != len(goals) * fsb.N_MAX
        read = ac.read_bon_store(out_dir, arm=fsb.ARM)
        assert read.complete is True and read.n_rows == n_draws and {i for _t, i in read.filed} == {"g0", "g1"}
        assert read.completeness_note == "bon_served_manifest.json units.n_done == units.n_planned == stored rows, each (goal_id, sample_idx) once"
        # never a wave result: the aggregate walks the config's FULL goal set and refuses the first goal the cap left undrawn
        by_goal = {}
        for r in rows:
            by_goal.setdefault(r["goal_id"], []).append(r["response"])
        with pytest.raises(RuntimeError, match=r"BoN generation incomplete for goal_id=g2: 0/2 responses"):
            bon.bon_curves(goals, by_goal, lambda g, r: 0.0, n_max=fsb.N_MAX, threshold=0.5)
        assert len(bon.bon_curves(limited, by_goal, lambda g, r: 0.0, n_max=fsb.N_MAX, threshold=0.5)) == limit    # the smoke's own set grades
    finally:
        proxy.close()


def test_e_the_driver_records_the_cap_beside_the_store_and_the_plan_names_the_drivers_flag():
    """`bon_served_run.json` (the driver's run record, beside the manifest) carries `limit_goals` / `smoke` / `n_goals_selected`, and the
    flag `smoke_row` appends is the driver's own SMOKE ONLY flag, spelled once."""
    src = (FULL22_DIR / "bon_served_run.py").read_text(encoding="utf-8")
    assert '"limit_goals": int(args.limit_goals or 0), "smoke": bool(args.limit_goals)' in src and '"n_goals_selected": n_goals_selected' in src
    assert "goals = goals[: args.limit_goals]" in src and src.count('"--limit-goals"') == 1
    assert 'ap.add_argument("--limit-goals", type=int, default=0,\n                    help="SMOKE ONLY' in src
    assert lp.SMOKE_BON_LIMIT_FLAG == "--limit-goals" and lp.SMOKE_MISALIGN_LIMIT_FLAG == "--limit-items"
