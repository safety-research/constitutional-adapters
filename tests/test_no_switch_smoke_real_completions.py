"""The NO-SWITCH render-fact certification on REAL completions —
G-PARSER's twin (``tests/test_think_split_real_completions.py``) for a family whose chat template carries NO thinking
switch: the diagnostic smoke's committed completions under ``contracts.NO_SWITCH_SMOKE_COMPLETIONS_DIR/<family>/``
(``smoke_completions.jsonl`` + ``smoke_report.json`` + ``usage_ledger.jsonl``, written by
``scripts/sdfbase/gsm8k/smoke_not_available_probe.py`` on the cluster), re-read here through the contracts' OWN seam
(``no_switch_render_certified``) and the real ``none`` split.

What certifies a switch-less fact: a fact may read ``certified`` ONLY when its smoke is committed and passes every check
here; a family still ``candidate`` whose smoke has not run is skipped BY NAME (never silently green). The checks: the
served header measured through the live ``/v1/tokenize`` with NO ``chat_template_kwargs`` EQUALS the fact by tuple
(the fact's two id fields equal each other — ONE render); every completion is on the base route under the ``card``
row and the seed rule; every stamp reads ``unspecified`` (the proxy's literal for an absent switch); zero think-tag
literals and zero reasoning channel on every completion (the ``none`` split returns the content byte for byte, no
block, no stray); every wire body was FREE of ``chat_template_kwargs`` (the probe's own record AND the ledger's
``body_sha256`` per request id, so what the ledger forwarded is what was built); the ledger's ``thinking`` is NULL on
every row (P17: absent, never ``false``); the items are the served task's first N in plan order.

The committed completions are model outputs on public GSM8K items; nothing here is personal data.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from sisteer.capeval.tiers import think_split as ts
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO / C.NO_SWITCH_SMOKE_COMPLETIONS_DIR
#: the families this test certifies: every switch-less family the contracts name (today: llama)
NO_SWITCH_FAMILIES = tuple(C.NO_THINKING_CHANNEL_FAMILIES)
#: the smoke's shape (g8k_cells.SMOKE_N / SMOKE_CAP, pinned equal in tests/test_sdfbase_gsm8k_sbatch.py)
SMOKE_N, SMOKE_CAP = 24, 1536
_HEX16 = re.compile(r"sha256_hex16 ([0-9a-f]{16})")


def _fact(family: str) -> C.RenderFact:
    return C.FAMILY_RENDER_FACTS[family]


def _certified(family: str) -> bool:
    return _fact(family).evidence_status == C.RENDER_EVIDENCE_CERTIFIED


def _smoke_dir(family: str) -> Path:
    return SMOKE_ROOT / family


def _rows_or_skip(family: str) -> list[dict]:
    """The committed completions, or a skip BY NAME for a candidate family whose smoke has not run.
    A CERTIFIED family with nothing committed is a failure, never a skip."""
    path = _smoke_dir(family) / "smoke_completions.jsonl"
    if not path.is_file():
        if _certified(family):
            pytest.fail(f"{family}: the render fact reads certified but no smoke completions are committed at "
                        f"{path.relative_to(REPO)} — a certification without its evidence")
        pytest.skip(f"{family}: render fact still candidate and no committed smoke completions at "
                    f"{path.relative_to(REPO)} — its diagnostic smoke has not run")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    assert rows, path
    return rows


def _report(family: str) -> dict:
    return json.loads((_smoke_dir(family) / "smoke_report.json").read_text(encoding="utf-8"))


def _ledger(family: str) -> list[dict]:
    return [json.loads(line) for line in (_smoke_dir(family) / "usage_ledger.jsonl").read_text(encoding="utf-8").split("\n")
            if line.strip()]


def _seed(item_id: str, sample_idx: int = 0) -> int:
    """``contracts.SEED_RULE_INSPECT`` re-derived HERE, independently of ``inspect_served.sample_seed`` (the
    probe's implementation) — so the seeds on the committed rows are verified from the rule's text, not by the
    code that wrote them."""
    return int.from_bytes(hashlib.sha256(f"{item_id}\x00{sample_idx}".encode("utf-8")).digest()[:4], "big") % (2 ** 31 - 1)


# --------------------------------------------------------------------------- #
# a certified fact has its evidence committed; the evidence names the smoke     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", NO_SWITCH_FAMILIES)
def test_a_certified_no_switch_fact_has_its_smoke_committed_and_cites_it(family):
    """Runs for EVERY switch-less family, certified or not — the guard that a fact cannot read ``certified`` ahead
    of its evidence. A candidate family asserts only that its evidence says so and that production refuses it."""
    fact = _fact(family)
    assert fact.thinking_switch is False and fact.split_mode == C.THINK_SPLIT_NONE and fact.server_parser is None
    assert fact.off_ids == fact.on_ids and not fact.header_moves and fact.think_open == fact.think_close == ""
    path = _smoke_dir(family) / "smoke_completions.jsonl"
    if not _certified(family):
        assert "CANDIDATE" in fact.evidence and C.NO_SWITCH_SMOKE_COMPLETIONS_DIR in fact.evidence
        assert C.render_fact_refusal(family) is not None
        return
    assert path.is_file(), f"{family}: certified without committed completions at {path.relative_to(REPO)}"
    assert (_smoke_dir(family) / "smoke_report.json").is_file() and (_smoke_dir(family) / "usage_ledger.jsonl").is_file()
    # the evidence names the committed jsonl, a smoke job id, the date and the wave sha it ran from
    assert str(path.relative_to(REPO)) in fact.evidence and "CERTIFIED 2026-" in fact.evidence
    assert "smoke job" in fact.evidence and "wave sha" in fact.evidence and "CANDIDATE" not in fact.evidence
    assert C.render_fact_certified(fact) and C.render_fact_refusal(family) is None
    assert C.check_render_fact_certified(family) == fact


@pytest.mark.parametrize("family", NO_SWITCH_FAMILIES)
def test_the_committed_smoke_is_the_diagnostic_no_switch_run_of_this_family(family):
    rows = _rows_or_skip(family)
    fact, report = _fact(family), _report(family)
    assert report["family"] == family and report["diagnostic"] is True and report["verdict"] == "PASS"
    assert report["thinking"] == C.THINKING_NOT_AVAILABLE and report["split_mode"] == fact.split_mode == C.THINK_SPLIT_NONE
    assert report["cert_class"] == C.CERT_CLASS_NO_SWITCH and report["tinycap_identity_version"] == 2
    assert report["seed_rule"] == C.SEED_RULE_INSPECT
    assert all(c["status"] == "PASS" for c in report["checks"]) and {c["check"] for c in report["checks"]} >= {
        "upstream_timeout", "header_no_switch", "items_planned", "requests_served", "wire_bodies_free_of_switch",
        "render_regime_stamps", "base_route_cell", "emitted_shape_none_split", "per_completion_render_certified", "ledger_rows"}
    # predicate (A″) rule 1: the served header measured live with NO kwargs EQUALS the fact's ONE render by tuple
    hdr = report["header"]
    assert hdr["error"] is None and tuple(hdr["got"]) == fact.off_ids == fact.on_ids and tuple(hdr["want"]) == fact.off_ids
    assert hdr["fact_ids_invariant"] is True and hdr["certification"]["render_certified"] is True
    assert hdr["certification"]["measured_header_ids_by_arm"] == {C.THINKING_NOT_AVAILABLE: list(fact.off_ids)}
    # the smoke's shape: N items at the diagnostic cap, one draw each, on the base route
    assert (report["n"], report["cap"]) == (SMOKE_N, SMOKE_CAP) and len(rows) == SMOKE_N
    assert sum(r["finish_reason"] == "length" for r in rows) == report["n_at_cap"]
    assert len({r["item_id"] for r in rows}) == len(rows) and [r["item_id"] for r in rows] == sorted(r["item_id"] for r in rows)
    model_keys = {r["model_key"] for r in rows}
    assert len(model_keys) == 1 and C.family_of_model_key(next(iter(model_keys))) == family
    key = next(iter(model_keys))
    # the served template is the fact's (its evidence names the 16-hex digest of the pinned template)
    assert report["chat_template_hash"] == _HEX16.search(fact.evidence).group(1)
    # the sampling row the smoke drew under is the model's `card` row (P6), exactly its SET parameters
    row = C.sampling_for(key, C.SAMPLING_CARD)
    assert report["sampling_mode"] == C.SAMPLING_CARD
    wanted = {"temperature": row.temperature, "top_p": row.top_p, "top_k": row.top_k}
    assert {k: report["sampling_params"].get(k) for k in wanted} == wanted
    assert set(report["sampling_params"]) == {k for k, v in wanted.items() if v is not None}
    expect = report["expected_stamps"]
    assert expect["render_regime"] == C.RENDER_REGIME_BY_ARM[C.THINKING_NOT_AVAILABLE] == C.RENDER_REGIME_UNSPECIFIED
    for r in rows:
        assert r["diagnostic"] is True and r["family"] == family and r["error"] is None and r["arm"] == C.THINKING_NOT_AVAILABLE
        assert r["split_mode"] == C.THINK_SPLIT_NONE and r["cert_class"] == C.CERT_CLASS_NO_SWITCH and r["cap"] == report["cap"]
        assert r["sampling_params"] == report["sampling_params"] and r["sampling_mode"] == report["sampling_mode"]
        assert r["seed"] == _seed(r["item_id"], 0)
        assert r["stamps"]["render_regime"] == C.RENDER_REGIME_UNSPECIFIED and r["stamps"]["cell_id"] == "none"
        assert r["stamps"]["cache_salt_regime"] == expect["cache_salt_regime"]
        assert r["stamps"]["resumed_prefill_policy"] == expect["resumed_prefill_policy"]
        assert r["stamps"]["virtual_model"] == report["vmodel"]
        assert r["finish_reason"] in ("stop", "length") and r["body_has_chat_template_kwargs"] is False
        assert re.fullmatch(r"[0-9a-f]{64}", r["body_sha256"]) and r["request_id"]
        assert (r["usage"] or {}).get("completion_tokens") is not None


# --------------------------------------------------------------------------- #
# the emitted shape: tag-free, channel-free; the `none` split returns the      #
# content verbatim; every completion certified through the contracts' seam    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", NO_SWITCH_FAMILIES)
def test_every_completion_is_tag_free_channel_free_and_the_none_split_returns_it_verbatim(family):
    rows = _rows_or_skip(family)
    fact, report = _fact(family), _report(family)
    header = list(report["header"]["got"])
    for r in rows:
        content = r["content"]
        assert isinstance(content, str) and content, r["item_id"]
        # NO channel: the content carries no think-tag literal, the reasoning channel is never populated
        assert ts.THINK_OPEN not in content and ts.THINK_CLOSE not in content, r["item_id"]
        assert r["reasoning_content"] in (None, ""), r["item_id"]
        # the `none` split (P19): the content IS the answer byte for byte — no block, never unterminated, no stray
        split = ts.split_think(content, mode=fact.split_mode)
        assert split.answer_text == content and not split.present and not split.unterminated, r["item_id"]
        assert not split.stray_open_tag and not split.stray_close_tag and split.close_tag_index is None
        # the smoke's own shape record agrees with the split it was written from
        shape = r["shape"]
        assert shape["mode"] == C.THINK_SPLIT_NONE and shape["arm"] == C.THINKING_NOT_AVAILABLE and shape["ok"] is True
        assert (shape["present"], shape["unterminated"], shape["stray_open_tag"], shape["stray_close_tag"]) == (False, False, False, False)
        assert shape["answer_is_content"] is True and shape["answer_chars"] == len(content)
        # the per-completion certification the probe filed, and the SAME seam re-run here on the committed row
        cert = r["certification"]
        assert cert["error"] is None and cert["render_certified"] is True and cert["cert_class"] == C.CERT_CLASS_NO_SWITCH
        assert cert["measured_header_ids_by_arm"] == {C.THINKING_NOT_AVAILABLE: header} and cert["render_source"] == "served_tokenize"
        fresh = C.no_switch_render_certified(family=family, header_ids=header, measured_live=True,
                                             probe_regime=r["stamps"]["render_regime"],
                                             probe_completion={"content": content, "reasoning": r["reasoning_content"],
                                                               "finish_reason": r["finish_reason"]})
        assert fresh["render_certified"] and fresh["think_shape_certified"] and fresh["render_regime_certified"]
        assert fresh[C.OOS_EVIDENCE_HEADER_CERTIFIED] is True and fresh["certification_problems"] == []
    # the OFF-arm derivation a switch-less family's rows read under is the same `none` split — never the other modes
    assert C.think_split_mode(arm=C.THINKING_NOT_AVAILABLE, reasoning_parser=None, prompt_opened_think=False) == C.THINK_SPLIT_NONE


# --------------------------------------------------------------------------- #
# the wire and the ledger: every body free of the switch; thinking NULL        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", NO_SWITCH_FAMILIES)
def test_the_ledger_records_the_absent_switch_on_every_row_and_the_bytes_the_probe_sent(family):
    rows = _rows_or_skip(family)
    report, ledger = _report(family), _ledger(family)
    assert report["ledger"]["ok"] is True and report["ledger"]["thinking_values"] == ["None"]
    assert len(ledger) == len(rows) == report["ledger"]["n_rows"]
    by_rid = {row["request_id"]: row for row in ledger}
    assert len(by_rid) == len(ledger)
    for r in rows:
        lrow = by_rid[r["request_id"]]
        # P17: the switch was ABSENT — `thinking` is NULL, never `false` (a false would record a switch that was never there)
        assert lrow["thinking"] is None, r["item_id"]
        # what the ledger FORWARDED is what the probe built — the exact bytes, free of chat_template_kwargs
        assert lrow["body_sha256"] == r["body_sha256"], r["item_id"]
        assert lrow["error_code"] is None and lrow["error_class"] == "none" and lrow["status"] == 200 and lrow["rejected"] is False
        assert lrow["max_tokens_requested"] == report["cap"] and lrow["vmodel"] == report["vmodel"]
        assert lrow["task"] == report["task_label"] and lrow["sample"] == f"{r['item_id']}/0"
        assert lrow["finish_reason"] == r["finish_reason"] and lrow[C.LEDGER_ATTEMPT_ID_FIELD] == report["attempt_id"]
        assert lrow["usage_completion_tokens"] == r["usage"]["completion_tokens"]
    assert set(C.LEDGER_ROW_FIELDS) <= set(ledger[0])


# --------------------------------------------------------------------------- #
# the items: the served task's first N in plan order (its pinned selection)   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", NO_SWITCH_FAMILIES)
def test_the_items_are_the_served_tasks_first_n_paired_items_in_plan_order(family):
    rows = _rows_or_skip(family)
    pytest.importorskip("inspect_ai")
    from sisteer.capeval.tiers import inspect_served as isv

    selection = isv.gsm8k_paired200_selection_ids(REPO)        # == plan.sample_ids whenever the plan is valid; pinned corpus only
    assert len(selection) == C.GSM8K_PAIRED_N and selection == sorted(selection)
    assert [r["item_id"] for r in rows] == selection[:len(rows)]
    # and the probe's seeds are the driver's ONE implementation of the rule (cross-checked with the re-derivation above)
    assert all(isv.sample_seed(r["item_id"], 0) == r["seed"] == _seed(r["item_id"]) for r in rows)
