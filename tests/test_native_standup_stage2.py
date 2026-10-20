"""native-psr-serving step 7 (stage 2) — the native standup's stage-2 readiness
facts:

- the POST-CONTROLS LEDGER READ: after the positive controls the worker's
  per-request failure ledger must be empty and unlatched
  (``native_standup.residual_ledger_after_controls`` / the public
  ``unclaimed_failures_problems(standup)`` the certification driver and the gate
  call after THEIR probes); an unclaimed count > 0 or a latched fault is a
  ``StandupRefused`` — "the proxy post-check is not running" — and no
  ``endpoint.json`` is published;
- the LOGPROB LIVENESS statistic of ``steering_bites`` for a restricted-mode /
  constant-kind cell: passes on a live payload whose greedy TEXT equals base
  (the text divergence reported beside, not gating), fails on a ZERO payload
  (bit-identical steered / twin logprobs — the known-bad);
- ``residual_slot_facts`` admits a restricted mode only once
  ``defense_cells.NATIVE_STAGE2_ARMED`` is True;
- ``run_positive_controls(su, output_diagnostics=None)`` INHERITS
  the standup's resolved policy; an explicit ``True`` against an OFF policy is
  refused at the controls, not only at bring-up; an explicit ``False`` tightens; a
  LoRA-only standup with the argument omitted keeps the diagnostics ON (the
  known-bad direction, pinned) — the record says which value was used and
  whether it was inherited.

The stage-2 standups arm the two stage-2 flags by monkeypatch
(``defense_cells.NATIVE_STAGE2_ARMED``, ``native_residual.STAGE2_POSITION_CLASSES``
— lane R flips the constants; every test here runs unchanged once they are
True) and stage a REAL response-fit PSR blob beside a real all-positions PSR and
a real LoRA cell over the scripted upstream. CPU-only; skips without the pinned
gemma-4 config + tokenizer in the local HF cache, like every fake standup test.
"""

from __future__ import annotations

import json
import types

import pytest

from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import fidelity
from sisteer.capeval.serving import native_residual as nr
from sisteer.capeval.serving import native_standup as ns
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.eval.defense_cell import arm_of, sha256_file

from test_lora_serving_up import (BASE_IDS, GEMMA_CELL, GEMMA_HCELL, GENUINE_IDS, OTHER_IDS,  # noqa: E402,F401
                                  PSR_CELL, ZERO, _blob_bytes, bring_up_fake_standup, cached_or_skip,
                                  fake_bank, lsu, psr_blob, stage_real_psr, synthetic_view)
from test_no_output_diagnostics import PSR_HCELL, _psr_view, _raising_diagnostics  # noqa: E402

RESTRICTED_PSR_CELL = PSR_CELL                       # constv2ctl2-full__psr_ce_all, staged RESPONSE-fit
RESTRICTED_PSR_HCELL = PSR_HCELL
ALL_PSR_CELL = "constv2-mix-full__psr_ce_all"        # staged all-positions
ALL_PSR_HCELL = ra.harness_cell_id(ALL_PSR_CELL)
ALL_PSR_IDS = [41, 42, 43]
#: a prompt carrying this marker is one the scripted worker ledgers as UNLOCATED
UNLOCATED_MARKER = "UNLOCATED-HEADER"


def served_header_len(model, revision) -> int:
    """The SERVED generation-prompt header's width for ``model`` — the exporter's
    own derivation (``lora_export._served_generation_header_ids``), which a
    response-fit blob's ``response_header_len`` must equal or the export refuses
    (gemma-4: 3 tokens)."""
    from sisteer.capeval.serving import lora_export as lx
    return len(lx._served_generation_header_ids(model, revision))


def stage_restricted_psr(bank, cell_id, model, revision, *, seed=9, header=None) -> str:
    """Stage ``cell_id`` as a REAL response-fit psr object (``positions="response"``,
    ``response_header_len=header`` — the served header's width when None) with the
    budget sidecar; returns its sha256."""
    from sisteer.methods import object_artifact_name

    header = int(header) if header is not None else served_header_len(model, revision)
    d = bank / cell_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / object_artifact_name(arm_of(cell_id))
    p.write_bytes(_blob_bytes(psr_blob(model, revision, seed=seed, positions="response", header=header)))
    sha = sha256_file(p)
    (d / "train_budget_meta.json").write_text(json.dumps(
        {"cell_id": cell_id, "completion_tokens": 512, "object_sha256": sha}))
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    return sha


def arm_stage2(monkeypatch):
    """The two stage-2 flags lane R flips on main, armed for one test."""
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", True)
    monkeypatch.setattr(nr, "STAGE2_POSITION_CLASSES", True)


def bring_up_stage2_standup(lsu, tmp_path, monkeypatch, *, failures_by_model=None, ledger_fault=False,
                            run_controls=True, restricted_chat=None, top_logprobs_by_model=None,
                            extra=None, fake_over=None):
    """The REAL ``bring_up_lora`` over a synthetic gemma view with a RESPONSE-fit
    PSR (restricted mode), an all-positions PSR and a real LoRA cell, against the
    scripted upstream whose ledger is scripted (``failures_by_model`` /
    ``ledger_fault``); every body the upstream saw lands in ``cap["chat_log"]``."""
    arm_stage2(monkeypatch)
    campaign = synthetic_view(tmp_path, "gemma4e4b", ["none", GEMMA_CELL, RESTRICTED_PSR_CELL, ALL_PSR_CELL])
    cfg = campaign[2]
    cached_or_skip(cfg)
    view, shas = fake_bank(tmp_path, cfg, real_cells=(GEMMA_CELL,))
    shas[ALL_PSR_CELL] = stage_real_psr(view, ALL_PSR_CELL, cfg["model"], cfg["model_revision"], seed=11)
    shas[RESTRICTED_PSR_CELL] = stage_restricted_psr(view, RESTRICTED_PSR_CELL, cfg["model"],
                                                     cfg["model_revision"])
    chat = {cfg["model"]: {"ids": list(BASE_IDS), "text": "the base reply"},
            ZERO: {"ids": list(BASE_IDS), "text": "the base reply"},
            GEMMA_HCELL: {"ids": list(GENUINE_IDS), "text": "a steered reply"},
            ALL_PSR_HCELL: {"ids": list(ALL_PSR_IDS), "text": "an all-positions psr reply"},
            RESTRICTED_PSR_HCELL: (dict(restricted_chat) if restricted_chat is not None
                                   else {"ids": list(OTHER_IDS), "text": "a response-fit psr reply"})}
    chat_log = tmp_path / "upstream_chat_log.jsonl"
    su, cap = bring_up_fake_standup(
        lsu, tmp_path, monkeypatch, cells=None, campaign=campaign, bank=(view, shas),
        expect=[f"{c}={shas[c]}" for c in (GEMMA_CELL, RESTRICTED_PSR_CELL, ALL_PSR_CELL)],
        run_controls=run_controls, extra=extra,
        fake_kwargs={"chat": chat, "genuine_name": GEMMA_HCELL,
                     "failures_by_model": dict(failures_by_model or {}), "ledger_fault": bool(ledger_fault),
                     "chat_log_path": chat_log, "top_logprobs_by_model": dict(top_logprobs_by_model or {}),
                     **(fake_over or {})})
    cap["chat_log"] = chat_log
    return su, cap


def test_the_boundary_refusal_control_passes_on_a_stage2_standup_and_fails_a_broken_join(lsu, tmp_path, monkeypatch):
    """O49: the sixth positive control drives ONE header-less request (a user
    turn under add_generation_prompt: false, the proxy-only marker) on every response-mode route through the
    live proxy; the upstream (the worker) ledgers it under its ENGINE id and the proxy's claim by the completion
    id must REFUSE it `response_boundary_unlocated` (500) and leave the ledger clean. KNOWN-BAD: an upstream that
    serves the header-less render without ledgering it (the broken join / worker) makes the control FAIL and the
    standup refuse, named — the very state a clean ledger could not distinguish before."""
    from sisteer.capeval.serving import proxy as px
    su, cap = bring_up_stage2_standup(lsu, tmp_path, monkeypatch)
    try:
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        assert pc["passed"] is True and pc["order"][-1] == "boundary_refusal"
        br = pc["controls"]["boundary_refusal"]
        assert br["passed"] is True and br["restricted_cells"] == [RESTRICTED_PSR_HCELL]
        cell = br["cells"][RESTRICTED_PSR_HCELL]
        assert cell["code"] == px.RESPONSE_BOUNDARY_UNLOCATED and cell["http_status"] == 500
        assert cell["worker_error_type"] == "ResponseBoundaryUnlocated"
        assert cell["unclaimed_after"] == 0 and cell["ledger_fault_after"] is False and cell["n_workers"] == 1
        assert "ResponseBoundaryUnlocated" in cell["refusal"]
        # the upstream saw the header-less body WITHOUT the proxy-only marker, on the restricted route only
        seen = [json.loads(l) for l in cap["chat_log"].read_text().splitlines()]
        ctl = [b for b in seen if b["payload"].get("add_generation_prompt") is False]
        assert len(ctl) == 1 and ctl[0]["model"] == RESTRICTED_PSR_HCELL
        assert px.BOUNDARY_REFUSAL_CONTROL_KEY not in ctl[0]["payload"] and "continue_final_message" not in ctl[0]["payload"]
        assert ctl[0]["payload"]["messages"] == [{"role": "user", "content": ns.BOUNDARY_REFUSAL_CONTROL_PROMPT}]
        assert ctl[0]["payload"]["max_tokens"] == 4 and ctl[0]["payload"]["temperature"] == 0
        # the post-controls ledger read is clean too
        assert su.residual_ledger_after_controls["unclaimed"] == 0 and su.residual_ledger_after_controls["ledger_fault"] is False
    finally:
        su.tear_down()
    # KNOWN-BAD: the join is broken (the render is served, nothing ledgered) -> the control FAILS the standup, named
    with pytest.raises(lsu.PositiveControlsFailed) as ei:
        bring_up_stage2_standup(lsu, tmp_path / "broken", monkeypatch, fake_over={"boundary_control_broken": True})
    msg = str(ei.value)
    assert "boundary_refusal" in msg and "RETURNED a completion" in msg and "fail-closed rule is NOT working" in msg
    pc = json.loads((tmp_path / "broken" / "out" / "positive_controls.json").read_text())
    assert pc["passed"] is False and pc["failed_control"] == "boundary_refusal"
    assert pc["controls"]["boundary_refusal"]["passed"] is False and pc["controls"]["boundary_refusal"]["restricted_cells"] == [RESTRICTED_PSR_HCELL]
    assert not (tmp_path / "broken" / "out" / "endpoint.json").exists()
    # KNOWN-BAD: an UNRELATED worker failure ledgered on the probe is refused under the
    # same code and its claim empties the ledger — the control must fail it BY THE WORKER'S error_type
    with pytest.raises(lsu.PositiveControlsFailed) as ei2:
        bring_up_stage2_standup(lsu, tmp_path / "wrong", monkeypatch, fake_over={"boundary_control_error_type": "PromptLenDisagreement"})
    assert "boundary_refusal" in str(ei2.value) and "PromptLenDisagreement" in str(ei2.value) and "not 'ResponseBoundaryUnlocated'" in str(ei2.value)
    assert not (tmp_path / "wrong" / "out" / "endpoint.json").exists()
    # KNOWN-BAD: a bookkeeping fault recorded during the controls' traffic is never clean after the controls (a fault
    # present at readiness is refused THERE by the attestation — the scripted upstream cannot fault only later, so
    # the post-controls rule is pinned on the judge the bring-up reads)
    clean = {"ledger_fault": False, "ledger_size": 0, "n_faults": 0}
    assert ns.ledger_problems([clean]) == []
    probs = ns.ledger_problems([{**clean, "n_faults": 2}])
    assert len(probs) == 1 and "n_faults = 2 after the controls" in probs[0]
    probs = ns.ledger_problems([{**clean, "n_faults": "2"}])
    assert len(probs) == 1 and "n_faults = '2'" in probs[0]


# --------------------------------------------------------------------------- #
# pure: the ledger judge, the statistic selector, the stage gate               #
# --------------------------------------------------------------------------- #


def test_ledger_problems_names_every_unclean_worker():
    clean = {"ledger_size": 0, "ledger_fault": False, "ledger_cap": 1024}
    assert ns.ledger_problems([clean]) == [] and ns.ledger_problems([clean, clean]) == []
    assert ns.ledger_problems([]) == ["no worker report to read the ledger from"]
    probs = ns.ledger_problems([clean, {**clean, "ledger_size": 3}])
    assert len(probs) == 1 and "worker 1" in probs[0] and "3 unclaimed" in probs[0]
    probs = ns.ledger_problems([{**clean, "ledger_fault": True, "ledger_size": 1024}])
    assert len(probs) == 2 and "ledger_fault = True" in probs[0]
    assert ns.ledger_problems([{}]) and ns.ledger_problems(["junk"])         # absent keys / non-report


def test_logprob_liveness_cells_selects_restricted_modes_and_constant_kinds():
    su = types.SimpleNamespace(
        lora_cells=("a__psr_kl_all", "b__psr_ce_all", "c__lora_r4", "d__arditi", "e__alphasteer", "f__vector"),
        adapters={"a__psr_kl_all": {"petri_source_kind": "psr"}, "b__psr_ce_all": {"petri_source_kind": "psr"},
                  "c__lora_r4": {"petri_source_kind": "lora_all"}, "d__arditi": {"petri_source_kind": "static_r"},
                  "e__alphasteer": {"petri_source_kind": "alphasteer"}, "f__vector": {"petri_source_kind": "vector"}},
        residual_slot={"armed": True, "cells": {
            "a__psr_kl_all": {"kind": "psr", "position_mode": "response"},
            "b__psr_ce_all": {"kind": "psr", "position_mode": "all"},
            "d__arditi": {"kind": "static_r", "position_mode": "prompt_last"},
            "f__vector": {"kind": "vector", "position_mode": "prefill_all"}}})
    out = ns.logprob_liveness_cells(su)
    assert set(out) == {"a__psr_kl_all", "d__arditi", "e__alphasteer", "f__vector"}
    # the RESTRICTED-MODE branch: a response-fit PSR writes nothing before the header. `psr` is
    # deliberately NOT a logprob-liveness KIND (the stage-1 PSR certificates were admitted under the
    # greedy gate), so this cell is selected by its MODE alone — which is what keeps that branch live
    assert "restricted native position mode 'response'" in out["a__psr_kl_all"]
    # the CONSTANT-KIND branch
    assert "constant-kind object (static_r)" in out["d__arditi"]
    assert "constant-kind object (alphasteer)" in out["e__alphasteer"]
    # `vector` JOINED THE KIND TUPLE 2026-09-16 (arditi-qwen-all-positions), so this cell is now
    # selected by its KIND rather than by its restricted mode. The SET is unchanged — it was already
    # selected — but the REASON moved, and the reason is what the record carries. That is the whole
    # point of the widening: a constant add is blind to greedy text at ANY position mode, so it must
    # not depend on being restricted to be measured (see the unarmed case below, which is where the
    # two branches actually come apart).
    assert "constant-kind object (vector)" in out["f__vector"]
    # AN UNARMED STANDUP: the restricted-mode branch has nothing to read, so only the constant kinds
    # remain — and `f__vector` is now among them, where before the widening it would have dropped
    # out and a correct constant add would have been graded on greedy text
    su.residual_slot = {"armed": False, "cells": {}}
    assert set(ns.logprob_liveness_cells(su)) == {"d__arditi", "e__alphasteer", "f__vector"}
    # the psr cells fall out with the slot, confirming they were mode-selected and not kind-selected
    assert "a__psr_kl_all" not in ns.logprob_liveness_cells(su)


def test_residual_slot_facts_admits_a_restricted_mode_only_once_stage2_is_armed(monkeypatch):
    fp = "ab" * 32
    att = {sa.RESIDUAL_SLOT_KEY: {"armed": True},
           "adapters": {"x__psr_kl_all": {"residual_fingerprint": fp, "petri_source_kind": "psr",
                                          "position_mode": "response"}}}
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", False)
    with pytest.raises(ns.StandupRefused, match="stage 1 serves mode 'all' only"):
        ns.residual_slot_facts(att, {"x__psr_kl_all": "psr"})
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", True)
    facts = ns.residual_slot_facts(att, {"x__psr_kl_all": "psr"})
    assert facts == {"armed": True, "cells": {"x__psr_kl_all": {"kind": "psr", "residual_fingerprint": fp,
                                                                "position_mode": "response"}}}
    bad = {**att, "adapters": {"x__psr_kl_all": {**att["adapters"]["x__psr_kl_all"], "position_mode": "last"}}}
    with pytest.raises(ns.StandupRefused, match="not a native position mode"):
        ns.residual_slot_facts(bad, {"x__psr_kl_all": "psr"})


# --------------------------------------------------------------------------- #
# the post-controls ledger read, end to end                                    #
# --------------------------------------------------------------------------- #


def test_unclaimed_failures_after_the_controls_refuse_the_standup(lsu, tmp_path, monkeypatch):
    """A worker that ledgers failures on an ALL-positions route (nothing claims
    them — the shape of a proxy post-check not running): the controls pass, the
    post-controls read finds unclaimed entries, the standup REFUSES and publishes
    no endpoint; ``cells_served.json`` carries the evidence."""
    campaign, bank, chat = _psr_view(tmp_path)
    view, shas = bank
    with pytest.raises(lsu.StandupRefused) as ei:
        bring_up_fake_standup(
            lsu, tmp_path, monkeypatch, cells=None, campaign=campaign, bank=bank,
            expect=[f"{GEMMA_CELL}={shas[GEMMA_CELL]}", f"{PSR_CELL}={shas[PSR_CELL]}"],
            fake_kwargs={"chat": chat, "genuine_name": GEMMA_HCELL,
                         "failures_by_model": {PSR_HCELL: {"marker": None}}})
    msg = str(ei.value)
    assert "not clean after the positive controls" in msg and "unclaimed request failure(s)" in msg
    out = tmp_path / "out"
    assert not (out / "endpoint.json").exists()
    pc = json.loads((out / "positive_controls.json").read_text())
    assert pc["passed"] is True                                    # the controls themselves passed
    cs = json.loads((out / "cells_served.json").read_text())
    led = cs[ns.RESIDUAL_LEDGER_AFTER_CONTROLS_KEY]
    assert led["checked"] is True and led["unclaimed"] > 0 and led["ledger_fault"] is False
    assert led["problems"] and "unclaimed" in led["problems"][0]


def test_a_latched_ledger_after_the_controls_refuses_an_all_positions_standup(lsu, tmp_path, monkeypatch):
    """The fault latched after readiness with no restricted route to surface it
    through a refusal: the post-controls read catches ``ledger_fault``."""
    campaign, bank, chat = _psr_view(tmp_path)
    view, shas = bank
    with pytest.raises(lsu.StandupRefused, match="ledger_fault = True"):
        bring_up_fake_standup(
            lsu, tmp_path, monkeypatch, cells=None, campaign=campaign, bank=bank,
            expect=[f"{GEMMA_CELL}={shas[GEMMA_CELL]}", f"{PSR_CELL}={shas[PSR_CELL]}"],
            fake_kwargs={"chat": chat, "genuine_name": GEMMA_HCELL, "ledger_fault": True})
    assert not (tmp_path / "out" / "endpoint.json").exists()


def test_a_lora_only_standup_records_the_ledger_read_as_not_checked(lsu, tmp_path, monkeypatch):
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch)
    try:
        assert su.residual_slot == {"armed": False, "cells": {}}            # unchanged (pinned elsewhere)
        assert su.residual_ledger_after_controls["checked"] is False
        assert su.residual_ledger_after_controls["problems"] == []
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        assert ep[ns.RESIDUAL_LEDGER_AFTER_CONTROLS_KEY]["checked"] is False
        # the public helper reads the live ledger: empty on an unarmed worker
        assert ns.unclaimed_failures_problems(su) == []
    finally:
        su.tear_down()


# --------------------------------------------------------------------------- #
# the logprob liveness statistic                                               #
# --------------------------------------------------------------------------- #


def test_logprob_liveness_passes_on_a_live_payload_whose_text_equals_base(lsu, tmp_path, monkeypatch):
    """The statistic's reason to exist: the restricted cell's greedy TEXT equals
    base (the text statistic fails, reported beside) while its top logprobs
    differ above the measured noise floor — the control passes."""
    live_top = {"11": -0.1, "1": -1.5, "2": -1.6, "3": -1.7, "4": -1.8}
    su, cap = bring_up_stage2_standup(
        lsu, tmp_path, monkeypatch,
        restricted_chat={"ids": list(BASE_IDS), "text": "the base reply"},
        top_logprobs_by_model={RESTRICTED_PSR_HCELL: live_top})
    try:
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        cell = pc["controls"]["steering_bites"]["cells"][RESTRICTED_PSR_HCELL]
        assert cell["statistic"] == ns.BITES_STATISTIC_LOGPROB and "restricted native position mode" in cell["why"]
        block = cell["logprob_liveness"]
        assert block["verified"] is True and "error" not in block
        assert block["n_conclusive"] == block["n_probes"] >= ns.LOGPROB_LIVENESS_MIN_CONCLUSIVE_PROBES
        assert block["noise_floor"]["measured"] == 0.0 and block["noise_floor"]["binding"] == "abs_floor"
        assert block["divergence"]["clears_bar"] is True
        assert block["divergence"]["steered_vs_scale0_median"] > ns.LOGPROB_BITES_ABS_FLOOR
        # the driver's reader accepts the block as a conclusive LIVE verdict
        from sisteer.capeval.tinycap import contracts as tc
        assert tc.logprob_liveness_conclusive_problems(block) == []
        # the text statistic, reported beside: equal to base, NOT gating
        assert cell["text_divergence"]["passed"] is False and cell["text_divergence"]["gating"] is False
        # the LoRA cell and the all-positions PSR keep the text statistic
        assert pc["controls"]["steering_bites"]["cells"][GEMMA_HCELL]["statistic"] == ns.BITES_STATISTIC_TEXT
        assert pc["controls"]["steering_bites"]["cells"][ALL_PSR_HCELL]["statistic"] == ns.BITES_STATISTIC_TEXT
        # every liveness request stayed inside the restricted-route allowlist (the upstream saw them)
        from sisteer.capeval.serving import proxy as px
        allowed = set(px.RESTRICTED_ROUTE_ALLOWED_KEYS) | {"cache_salt"}
        seen = [json.loads(l) for l in cap["chat_log"].read_text().splitlines()]
        restricted = [r for r in seen if r["model"] == RESTRICTED_PSR_HCELL]
        assert restricted and all(set(r["payload"]) <= allowed for r in restricted)
    finally:
        su.tear_down()


def test_logprob_liveness_fails_on_a_zero_payload(lsu, tmp_path, monkeypatch):
    """The known-bad: a zero residual payload serves base's logprobs bit for bit
    on every probe — the control FAILS (never a text-only or vacuous pass)."""
    with pytest.raises(lsu.PositiveControlsFailed, match="steering_bites") as ei:
        bring_up_stage2_standup(lsu, tmp_path, monkeypatch,
                                restricted_chat={"ids": list(BASE_IDS), "text": "the base reply"})
    assert "BIT-IDENTICAL" in str(ei.value)
    rec = ei.value.record
    cell = rec["controls"]["steering_bites"]["cells"][RESTRICTED_PSR_HCELL]      # the block stays on the record
    assert cell["logprob_liveness"]["verified"] is False and "BIT-IDENTICAL" in cell["logprob_liveness"]["error"]
    assert not (tmp_path / "out" / "endpoint.json").exists()


# --------------------------------------------------------------------------- #
# run_positive_controls inherits the resolved policy                             #
# --------------------------------------------------------------------------- #


def _psr_standup_without_controls(lsu, tmp_path, monkeypatch):
    campaign, bank, chat = _psr_view(tmp_path)
    view, shas = bank
    return bring_up_fake_standup(
        lsu, tmp_path, monkeypatch, cells=None, campaign=campaign, bank=bank, run_controls=False,
        expect=[f"{GEMMA_CELL}={shas[GEMMA_CELL]}", f"{PSR_CELL}={shas[PSR_CELL]}"],
        fake_kwargs={"chat": chat, "genuine_name": GEMMA_HCELL})


def test_p2a_omitted_argument_inherits_the_off_policy_on_a_residual_standup(lsu, tmp_path, monkeypatch):
    calls = _raising_diagnostics(monkeypatch)
    su, cap = _psr_standup_without_controls(lsu, tmp_path, monkeypatch)
    try:
        assert su.output_diagnostics is False and su.positive_controls is None
        rec = ns.run_positive_controls(su)                       # the one-argument fallback path
        assert calls == []                                       # the stub was never reached
        assert rec["passed"] is True
        assert rec["output_diagnostics"] is False and rec["output_diagnostics_inherited"] is True
        assert rec["output_diagnostics_policy"] is False
        assert rec["controls"]["selection_honored"]["output_diagnostics"] is False
        assert "numerics" not in rec["controls"]["selection_honored"]
        assert json.loads(su.positive_controls_path.read_text())["output_diagnostics_inherited"] is True
    finally:
        su.tear_down()


def test_p2a_explicit_true_on_an_off_policy_standup_is_refused_at_the_controls(lsu, tmp_path, monkeypatch):
    calls = _raising_diagnostics(monkeypatch)
    su, cap = _psr_standup_without_controls(lsu, tmp_path, monkeypatch)
    try:
        with pytest.raises(ns.StandupRefused, match="resolved policy is OFF") as ei:
            ns.run_positive_controls(su, output_diagnostics=True)
        assert PSR_HCELL in str(ei.value)
        assert calls == [] and not su.positive_controls_path.exists()   # refused before any control ran
        # the pure resolver states the rule
        assert ns.resolve_output_diagnostics(su, None) == (False, True)
        assert ns.resolve_output_diagnostics(su, False) == (False, False)
    finally:
        su.tear_down()


def test_p2a_explicit_false_on_a_lora_only_standup_tightens(lsu, tmp_path, monkeypatch):
    calls = _raising_diagnostics(monkeypatch)
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch, run_controls=False)
    try:
        assert su.output_diagnostics is True
        rec = ns.run_positive_controls(su, output_diagnostics=False)
        assert calls == [] and rec["passed"] is True
        assert rec["output_diagnostics"] is False and rec["output_diagnostics_inherited"] is False
        assert rec["output_diagnostics_policy"] is True
        assert "numerics" not in rec["controls"]["selection_honored"]
        assert ns.resolve_output_diagnostics(su, True) == (True, False)     # ON against an ON policy: fine
    finally:
        su.tear_down()


def test_p2a_omitted_argument_on_a_lora_only_standup_keeps_the_diagnostics_on(lsu, tmp_path, monkeypatch):
    """The known-bad DIRECTION, pinned: a LoRA-only standup's policy is ON, so the
    omitted argument inherits True and the numerics ARE computed (the raising stub
    is reached and fails selection_honored)."""
    calls = _raising_diagnostics(monkeypatch)
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch, run_controls=False)
    try:
        with pytest.raises(ns.PositiveControlsFailed, match="selection_honored") as ei:
            ns.run_positive_controls(su)
        assert calls and ei.value.record["output_diagnostics"] is True
        assert ei.value.record["output_diagnostics_inherited"] is True
    finally:
        su.tear_down()


def test_the_fidelity_stub_seam_is_the_one_the_controls_call():
    assert callable(fidelity.multi_lora_output_diagnostics)
