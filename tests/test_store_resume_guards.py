"""Two follow-ups to the judge backend seam (2026-09-02), both raised by a
peer's smoke of a full fitted cell through grok-4.6:

1. THE CALLING CONVENTION. The judged tier calls
   ``judge.judge_all(pairs, allow_refusal_abstention=...)`` — a KEYWORD
   argument — and the retired in-process shim's ``_judge_all(self, pairs)``
   TypeErrored on it. The backend seam lives in ``_create``/``_send`` and
   never wrapped ``judge_all``, so nothing changed here; this pins it: every
   keyword any tier call site passes is a keyword-only parameter of
   :meth:`ClaudeJudge.judge_all`, and an openrouter judge answers the call
   exactly as the tier makes it.

2. RE-JUDGING A STORE. ``cell_controls_run.py judge`` into a store that
   already holds label rows for a task RESUMES it: :func:`_completed_items`
   reads one value per (item, metric) and only items WITHOUT a label row
   are judged; :class:`_Emitter` seeds its seen-set from the store's rows,
   so a filed (task, item, metric) — per-item or ``__aggregate__`` — is
   never appended twice (tests/test_capeval_judged.py::
   test_or_bench_judge_resume_files_once_and_skips_judged pins the
   files-once half). A store's rows therefore come from ONE judge (the
   judge model and backend key the store; a mismatch refuses), and a
   sequential re-run cannot duplicate or bias anything — the peer's
   602 -> 3476 rows was one first-pass emission (602 + 450 xstest labels +
   1211 x 2 or_bench rows before the abstention ceiling refused), not
   duplication. The ONE hole was two writers racing on the same store (a
   duplicate driver — it has happened in this project): a (task, item,
   metric) filed twice with DIFFERENT values was resolved silently by file
   order, in both the tier's ``_completed_items`` and the driver's
   ``_parity_payloads``. Both now REFUSE; a same-value repeat (a crash
   between an item's companion rows, re-filed identically) still resumes.
   No ``--rejudge`` flag: superseding rows in an append-only store is a
   separate design, and the resume path is the deliberate one.
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("sisteer.capeval.tiers.tier0",
                    reason="the store resume guards need sisteer.capeval")

import cell_controls_run as run  # noqa: E402
from sisteer.capeval.judging import (  # noqa: E402
    ORBENCH_RUBRIC_V1, ClaudeJudge, JudgeCache, JudgeRequest, protocol_sha)
from sisteer.capeval.tiers.tier0 import Tier0Error, _completed_items  # noqa: E402

from test_capeval_judging import _Client, _Msg, _ToolUse  # noqa: E402

GROK = "x-ai/grok-4.6"
TASK = "judged/or_bench_hard_1k"


# --- 1. the calling convention ------------------------------------------------------


def _tier_call_kwargs(path: Path) -> set:
    """Every keyword any ``*.judge_all(...)`` call in ``path`` passes."""
    tree = ast.parse(path.read_text())
    found: set = set()
    n_calls = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "judge_all"):
            n_calls += 1
            found |= {k.arg for k in node.keywords if k.arg is not None}
    return found if n_calls else set()


def test_judge_all_accepts_every_keyword_the_tier_passes():
    sig = inspect.signature(ClaudeJudge.judge_all)
    kw_only = {n for n, p in sig.parameters.items()
               if p.kind is inspect.Parameter.KEYWORD_ONLY}
    assert {"allow_refusal_abstention", "reattempt_abstentions"} <= kw_only
    tier = REPO / "src/sisteer/capeval/tiers/judged.py"
    used = _tier_call_kwargs(tier)
    assert "allow_refusal_abstention" in used   # the tier really passes it
    assert used <= kw_only, f"the tier passes {used - kw_only} which judge_all lacks"
    assert _tier_call_kwargs(REPO / "scripts/cell_controls_run.py") <= kw_only


def test_openrouter_judge_answers_the_tiers_exact_call(tmp_path):
    def grok(params):
        return _Msg([_ToolUse("compliance_rating",
                              {"reasoning": "synthetic", "label": "full_compliance"})])

    judge = ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"), model=GROK,
                        backend="openrouter", concurrency=2,
                        clients={"openrouter": _Client(grok)})
    rubric = ORBENCH_RUBRIC_V1
    reqs = {"orb0000-aaaaaaaa": JudgeRequest(
        prompt="synthetic prompt", candidate_a="synthetic answer",
        candidate_b=None, roles=None, rubric_sha=rubric.sha,
        judge_model=GROK, protocol_sha=protocol_sha(rubric))}
    # verbatim shape of tiers/judged.py::_judge_cpi's call
    verdicts = judge.judge_all([(req, rubric) for req in reqs.values()],
                               allow_refusal_abstention=True)
    assert verdicts[reqs["orb0000-aaaaaaaa"].key()] == {"label": "full_compliance"}
    # and the re-attempt keyword the docstring names
    again = judge.judge_all([(req, rubric) for req in reqs.values()],
                            allow_refusal_abstention=True,
                            reattempt_abstentions=True)
    assert again == verdicts and judge.n_api_calls == 1


# --- 2. a raced store refuses; a same-value repeat resumes ---------------------------


class _Writer:
    def __init__(self, rows):
        self._rows = rows

    def _parsed_rows(self):
        return list(self._rows)


def _row(item, metric, value, task=TASK):
    return {"task": task, "item_id": item, "metric": metric, "value": value}


def test_completed_items_is_one_value_per_item_and_refuses_a_conflict():
    rows = [_row("orb0000-aaaaaaaa", "orb_label", 2.0),
            _row("orb0000-aaaaaaaa", "orb_refused", 0.0),
            _row("orb0001-bbbbbbbb", "orb_label", 0.0)]      # incomplete: no refused row
    done = _completed_items(_Writer(rows), TASK, {"orb_label", "orb_refused"})
    assert done == {"orb0000-aaaaaaaa": {"orb_label": 2.0, "orb_refused": 0.0}}
    # a same-value repeat (companion row re-filed after a crash) resumes
    same = rows + [_row("orb0000-aaaaaaaa", "orb_label", 2.0)]
    assert _completed_items(_Writer(same), TASK, {"orb_label", "orb_refused"}) == done
    # a DIFFERENT value for the same key is a raced store — never file order
    raced = rows + [_row("orb0000-aaaaaaaa", "orb_label", 0.0)]
    with pytest.raises(Tier0Error, match="filed twice with DIFFERENT values"):
        _completed_items(_Writer(raced), TASK, {"orb_label", "orb_refused"})
    # rows of another task or metric never interfere
    other = rows + [_row("orb0000-aaaaaaaa", "orb_label", 0.0, task="judged/xstest_v2")]
    assert _completed_items(_Writer(other), TASK, {"orb_label", "orb_refused"}) == done


def test_parity_payloads_refuse_a_conflicting_acc_row(tmp_path):
    task = "s2parity/mmlu_letter"
    rows = [{"task": task, "item_id": f"i{k}", "metric": "acc", "value": 1.0}
            for k in range(3)]
    (tmp_path / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    assert run._parity_payloads(tmp_path, ["mmlu"]) == {
        "mmlu": {"acc": 1.0, "n": 3}}
    # same-value repeat: counted once
    with (tmp_path / "rows.jsonl").open("a") as f:
        f.write(json.dumps(rows[0]) + "\n")
    assert run._parity_payloads(tmp_path, ["mmlu"])["mmlu"]["n"] == 3
    # a conflicting repeat refuses
    with (tmp_path / "rows.jsonl").open("a") as f:
        f.write(json.dumps({**rows[0], "value": 0.0}) + "\n")
    with pytest.raises(SystemExit, match="two DIFFERENT acc rows"):
        run._parity_payloads(tmp_path, ["mmlu"])
