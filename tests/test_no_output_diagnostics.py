"""native-psr-serving step 3 — a RESIDUAL cell served by the REAL native standup
core, and the OUTPUT-DIAGNOSTICS POLICY decided centrally.

Through the real ``lora_serving_up.bring_up_lora`` (the front every wave, smoke
and timing harness uses; ``controls_fn`` defaulted) over a synthetic gemma4e4b
view whose ``constv2ctl2-full__psr_ce_all`` cell is a REAL all-positions PSR
blob beside a real ``lora_all`` cell, against the scripted upstream whose
readiness answers are derived from the launcher's own expectations:

- the psr cell is SERVED (a lora route; ``cells_skipped == []``), its bank entry
  a rank-1 zero carrier + payload; the launch env carries the residual-slot flag;
  the attestation is schema 2 with ``residual_slot.armed True`` and the psr
  adapter's ``petri_source_kind`` / ``residual_fingerprint`` / ``position_mode``;
  ``/v1/models``, ``routing_table.json``, ``cells_served.json`` and
  ``endpoint.json`` carry the kind; ``residual_slot.cells`` names the psr cell;
- ``output_diagnostics == False`` in every record and a
  ``fidelity.multi_lora_output_diagnostics`` stub that RAISES is NOT called;
  ``selection_honored`` still reaches ``routing_attested`` with NO ``numerics``
  block; the same view with only the LoRA cell gives policy ``True`` and the
  stub IS called (today's behaviour, pinned);
- the residual-slot readiness known-bads (``serving_attestation.
  residual_slot_problems`` through the real ``launch_server``): a worker that
  reports unarmed while a payload is registered, a slot with another
  fingerprint, a block left unhooked, an unbound runner, another op name, a
  faulted ledger, a slot for a plain LoRA — each refuses readiness before any
  traffic; and the rev-1 reader's shape (a schema-1 record) still reads.

CPU-only; needs the pinned gemma-4 config + tokenizer in the local HF cache
(skips otherwise, like every fake standup test)."""

from __future__ import annotations

import json

import pytest

from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import fidelity
from sisteer.capeval.serving import native_standup as ns
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import vllm_server as vs

from test_lora_serving_up import (BASE_IDS, GEMMA_CELL, GEMMA_HCELL, GENUINE_IDS, OTHER_IDS, PSR_CELL,
                                  ZERO, _campaign, bring_up_fake_standup, cached_or_skip, fake_bank,
                                  lsu, stage_real_psr, synthetic_view)  # noqa: F401

PSR_HCELL = ra.harness_cell_id(PSR_CELL)


def _psr_view(tmp_path, key="gemma4e4b"):
    """The synthetic view with the psr cell staged as a REAL all-positions PSR
    object beside the real lora cell; returns ``(campaign, (view, shas), chat)``."""
    campaign = synthetic_view(tmp_path, key, ["none", GEMMA_CELL, PSR_CELL])
    cfg = campaign[2]
    cached_or_skip(cfg)
    view, shas = fake_bank(tmp_path, cfg, real_cells=(GEMMA_CELL,))
    shas[PSR_CELL] = stage_real_psr(view, PSR_CELL, cfg["model"], cfg["model_revision"])
    chat = {cfg["model"]: {"ids": list(BASE_IDS), "text": "the base reply"},
            ZERO: {"ids": list(BASE_IDS), "text": "the base reply"},
            GEMMA_HCELL: {"ids": list(GENUINE_IDS), "text": "a steered reply"},
            PSR_HCELL: {"ids": list(OTHER_IDS), "text": "a psr-steered reply"}}
    return campaign, (view, shas), chat


def _bring_up_psr(lsu, tmp_path, monkeypatch, **fake_over):
    campaign, bank, chat = _psr_view(tmp_path)
    view, shas = bank
    return bring_up_fake_standup(
        lsu, tmp_path, monkeypatch, cells=None, campaign=campaign, bank=bank,
        expect=[f"{GEMMA_CELL}={shas[GEMMA_CELL]}", f"{PSR_CELL}={shas[PSR_CELL]}"],
        fake_kwargs={"chat": chat, "genuine_name": GEMMA_HCELL, **fake_over})


def _raising_diagnostics(monkeypatch):
    calls = []

    def raiser(**kw):
        calls.append(kw)
        raise AssertionError("multi_lora_output_diagnostics must not run on a residual-bearing standup")

    monkeypatch.setattr(fidelity, "multi_lora_output_diagnostics", raiser)
    return calls


def test_a_real_psr_cell_is_served_natively_with_the_diagnostics_off(lsu, tmp_path, monkeypatch):
    calls = _raising_diagnostics(monkeypatch)
    su, cap = _bring_up_psr(lsu, tmp_path, monkeypatch)
    try:
        assert calls == []                                              # the stub was never called
        assert su.output_diagnostics is False and su.residual_slot["armed"] is True
        assert set(su.residual_slot["cells"]) == {PSR_HCELL}
        psr_fact = su.residual_slot["cells"][PSR_HCELL]
        assert psr_fact["kind"] == "psr" and psr_fact["position_mode"] == "all"
        assert len(psr_fact["residual_fingerprint"]) == 64
        # served, not skipped: a lora route beside the LoRA cell
        assert su.cells_skipped == () and set(su.lora_cells) == {GEMMA_HCELL, PSR_HCELL}
        assert su.adapters[PSR_HCELL]["petri_source_kind"] == "psr"
        assert su.adapters[PSR_HCELL]["r"] == 1 and su.adapters[GEMMA_HCELL]["r"] == 4
        # the launch env armed the slot; a LoRA-only launch's env has no such key
        assert cap["launch"].env[vs.RESIDUAL_SLOT_ENV] == "1" and cap["launch"].residual_slot_armed is True
        # the attestation: schema 2, armed, the psr adapter's residual identity
        att = su.attestation
        assert att["schema_version"] == 2 and att[sa.RESIDUAL_SLOT_KEY]["armed"] is True
        assert att[sa.RESIDUAL_SLOT_KEY]["rpc_method"] == sa.RESIDUAL_SLOT_RPC_METHOD
        blk = att["adapters"][PSR_HCELL]
        assert blk["petri_source_kind"] == "psr" and blk["position_mode"] == "all"
        assert blk["residual_fingerprint"] == psr_fact["residual_fingerprint"]
        assert att["adapters"][GEMMA_HCELL]["residual_fingerprint"] is None
        assert att["adapters"][GEMMA_HCELL]["petri_source_kind"] == "lora_all"
        assert att["adapters"][ZERO]["petri_source_kind"] is None
        # the worker's loaded-adapter report carried the pair
        wa = {a["id"]: a for a in att["scheduler_geometry"]["worker_reports"][0]["lora_adapters"]}
        assert wa[blk["lora_int_id"]]["residual_fingerprint"] == blk["residual_fingerprint"]
        # the engine identity the proxy holds and echoes
        ident = su.engine_identity
        # the identity carries the attested worker count: the proxy binds every failure
        # claim's per-worker result list to it
        assert ident["residual_slot"] == {"armed": True, "n_workers": 1}
        assert su.attestation[sa.RESIDUAL_SLOT_KEY]["n_workers"] == 1 and su.proxy._backend.expected_workers == 1
        assert ident["adapters"][PSR_HCELL]["petri_source_kind"] == "psr"
        body = su.proxy.models_payload()
        eng = body["petri_proxy"]["engine"]
        assert eng["residual_slot"] == {"armed": True, "n_workers": 1}
        assert eng["adapters"][PSR_HCELL]["petri_source_kind"] == "psr"
        # O49: an ALL-positions psr standup has no response boundary to fail — the sixth control is not applicable,
        # recorded with the EMPTY restricted-cell set the certification requires for that verdict
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        br = pc["controls"]["boundary_refusal"]
        assert br["verdict"] == ns.NOT_APPLICABLE and br["passed"] is None and br["restricted_cells"] == []
        assert eng["adapters"][PSR_HCELL]["residual_fingerprint"] == blk["residual_fingerprint"]
        assert eng["adapters"][GEMMA_HCELL]["residual_fingerprint"] is None
        # the routing table's adapter block carries the kind
        table = json.loads(su.table_path.read_text())
        assert table["adapters"][PSR_HCELL]["petri_source_kind"] == "psr"
        # the records
        cs = json.loads((cap["out"] / "cells_served.json").read_text())
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        for rec in (cs, ep):
            assert rec["output_diagnostics"] is False
            assert rec["residual_slot"] == su.residual_slot
            assert rec["cells_skipped"] == []
            assert rec["adapters"][PSR_HCELL]["petri_source_kind"] == "psr"
        assert set(ep["lora_cells"]) == {GEMMA_HCELL, PSR_HCELL}
        assert f"petri__{PSR_HCELL}" in ep["served_models"]
        assert pc["passed"] is True and pc["output_diagnostics"] is False
        sel = pc["controls"]["selection_honored"]
        assert sel["verdict"] == "routing_attested" and sel["output_diagnostics"] is False
        assert "numerics" not in sel
        assert sel["n_requests"] == sel["n_observed"] > 0 and sel["n_misrouted"] == 0
        bites = pc["controls"]["steering_bites"]
        assert bites["passed"] is True
    finally:
        su.tear_down()


def test_a_lora_only_view_keeps_the_diagnostics_on(lsu, tmp_path, monkeypatch):
    """Today's behaviour, pinned: a LoRA-only standup computes the numerics
    block, so the raising stub IS reached and fails the control."""
    calls = _raising_diagnostics(monkeypatch)
    with pytest.raises(lsu.PositiveControlsFailed, match="selection_honored") as ei:
        bring_up_fake_standup(lsu, tmp_path, monkeypatch)
    assert calls and "must not run on a residual-bearing standup" in str(ei.value)
    assert ei.value.record["output_diagnostics"] is True


def test_a_lora_only_view_records_the_numerics_and_policy_true(lsu, tmp_path, monkeypatch):
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch)
    try:
        assert su.output_diagnostics is True and su.residual_slot == {"armed": False, "cells": {}}
        assert cap["launch"].residual_slot_armed is False and vs.RESIDUAL_SLOT_ENV not in cap["launch"].env
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        sel = pc["controls"]["selection_honored"]
        assert sel["output_diagnostics"] is True and isinstance(sel["numerics"], dict)
        assert su.attestation[sa.RESIDUAL_SLOT_KEY]["armed"] is False
        assert su.engine_identity["residual_slot"] == {"armed": False, "n_workers": 1}
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        assert ep["output_diagnostics"] is True and ep["residual_slot"] == {"armed": False, "cells": {}}
    finally:
        su.tear_down()


def test_output_diagnostics_policy():
    assert ns.output_diagnostics_policy([]) is True
    assert ns.output_diagnostics_policy(["lora_all", "lora_subtraction"]) is True
    for kind in ("psr", "vector", "vector_all", "diffmean", "static_r"):
        assert ns.output_diagnostics_policy(["lora_all", kind]) is False


@pytest.mark.parametrize("label,over,needle", [
    ("unarmed worker", {"armed": False, "slots": {}, "op_name": None, "runner_bound": False,
                        "n_layers_hooked": 0, "n_layers_model": None, "patched_classes": {},
                        "op_registered": False, "bank": None},
     "residual_slot.armed = False, expected True"),
    ("another op name", {"op_name": "sisteer.residual_slot"}, "op_name = 'sisteer.residual_slot'"),
    ("runner unbound", {"runner_bound": False}, "not bound to the model runner"),
    ("a block unhooked", {"n_layers_hooked": 41}, "decoder block(s) hooked"),
    ("a class unwrapped", {"patched_classes": {"vllm.model_executor.models.gemma4.Gemma4DecoderLayer":
                                               {"forward": "wrapped", "forward_wrapped": False}}},
     "reports no wrapped forward"),
    ("ledger faulted", {"ledger_fault": True}, "ledger_fault = True"),
    ("op unregistered", {"op_registered": False}, "not registered"),
])
def test_residual_slot_readiness_known_bads_refuse_before_traffic(lsu, tmp_path, monkeypatch, label,
                                                                   over, needle):
    with pytest.raises(vs.VllmServerError) as ei:
        _bring_up_psr(lsu, tmp_path, monkeypatch, residual_over=over)
    msg = str(ei.value)
    assert "residual slot does not match" in msg and needle in msg, (label, msg)
    assert not (tmp_path / "out" / "endpoint.json").exists()


def test_a_slot_with_another_fingerprint_or_for_a_plain_lora_refuses(lsu, tmp_path, monkeypatch):
    campaign, bank, chat = _psr_view(tmp_path)
    view, shas = bank
    expect = [f"{GEMMA_CELL}={shas[GEMMA_CELL]}", f"{PSR_CELL}={shas[PSR_CELL]}"]

    def _run(**over):
        return bring_up_fake_standup(lsu, tmp_path, monkeypatch, cells=None, campaign=campaign, bank=bank,
                                     expect=expect,
                                     fake_kwargs={"chat": chat, "genuine_name": GEMMA_HCELL,
                                                  "residual_over": over})

    # the worker's slot table replaced by a stray entry: the psr adapter's slot is missing AND an
    # unaccounted slot is loaded — both named
    with pytest.raises(vs.VllmServerError) as ei:
        _run(slots={"__probe__": {"fingerprint": "00" * 32}})
    msg = str(ei.value)
    assert "has no loaded residual slot" in msg and "no registered adapter's payload accounts" in msg
    assert not (tmp_path / "out" / "endpoint.json").exists()


def test_residual_slot_problems_refuse_a_fact_table_overflow_or_a_recorded_fault_at_readiness():
    """Stage 2 (the report facts): a worker whose per-request fact table OVERFLOWED (a new request
    refused at the cap because finished ones were not reclaimed — `overflowed` counts them, `overflow`
    latches), or that already records a serving fault, must not pass readiness — named per worker; a
    clean table with zero faults adds nothing (the stage-1 report without these keys still passes).
    `reclaimed` is the table's RECLAMATION counter (facts released for finished requests) — the readiness
    probe's own requests drive it (measured 107 at S = 107), so a table that reclaimed is HEALTHY, not
    refused; the known-bad that killed four jobs is pinned below."""
    expected = {"armed": True, "n_layers_model": 2, "adapters": {}}
    base = {"armed": True, "op_name": sa.RESIDUAL_SLOT_OP_NAME, "op_registered": True, "runner_bound": True,
            "ledger_fault": False, "ledger_size": 0, "n_layers_model": 2, "n_layers_hooked": 2,
            "patched_classes": {"C": {"forward": "wrapped", "forward_wrapped": True}}, "slots": {}}
    assert sa.residual_slot_problems([base], expected) == []
    clean = {**base, "fact_table": {"size": 0, "cap": 16384, "overflow": False, "reclaimed": 0}, "n_faults": 0, "faults": []}
    assert sa.residual_slot_problems([clean], expected) == []
    over = {**clean, "fact_table": {"size": 16384, "cap": 16384, "overflow": True, "reclaimed": 3}}
    probs = sa.residual_slot_problems([over], expected)
    assert probs and all("fact_table overflowed" in p for p in probs), probs        # the overflow, and ONLY the overflow
    # the readiness probe at S = 107: every probe request finished and its facts were RECLAIMED; the
    # table is empty, nothing overflowed, no fault — healthy, and readiness must pass it
    reclaimed = {**clean, "fact_table": {"size": 0, "cap": 16384, "overflow": False, "admitted": 107,
                                         "overflowed": 0, "reclaimed": 107}}
    assert sa.residual_slot_problems([reclaimed], expected) == []
    faulted = {**clean, "n_faults": 2, "faults": [{"error_type": "PromptLenDisagreement"}]}
    probs = sa.residual_slot_problems([faulted], expected)
    assert any("2 serving fault(s)" in p and "PromptLenDisagreement" in p for p in probs), probs


def test_residual_slot_problems_directly_on_a_synthetic_expectation():
    from test_petri_serving_gate import _native_residual_report

    fp = "ab" * 32
    expected = {"armed": True, "op_name": sa.RESIDUAL_SLOT_OP_NAME, "n_layers_model": 3, "hidden_size": 16,
                "adapters": {"none__lora0": {"lora_int_id": 1, "residual_fingerprint": None, "kind": None,
                                             "op": None, "position_mode": None, "layers": None,
                                             "boundaries": None},
                             "si_x__psr_ce_all": {"lora_int_id": 2, "residual_fingerprint": fp, "kind": "psr",
                                                  "op": "relu_gated_vector", "position_mode": "all",
                                                  "layers": [0, 1, 2], "boundaries": [0, 1, 2]}}}
    good = _native_residual_report(expected)
    assert sa.residual_slot_problems([good], expected) == []
    assert sa.residual_slot_problems([good, good], expected) == []          # every worker
    assert sa.residual_slot_problems([], expected) and sa.residual_slot_problems(None, expected)
    # a slot carrying another payload's fingerprint (a swapped or stale entry)
    bad = _native_residual_report(expected)
    bad["slots"]["2"] = {**bad["slots"]["2"], "fingerprint": "cd" * 32}
    assert any("fingerprint = 'cd" in p for p in sa.residual_slot_problems([bad], expected))
    # a slot loaded for the plain LoRA (the zero adapter)
    bad = _native_residual_report(expected)
    bad["slots"]["1"] = dict(bad["slots"]["2"])
    probs = sa.residual_slot_problems([bad], expected)
    assert any("carries no residual payload but the worker loaded slot" in p for p in probs)
    # a wrong mode / layers / boundaries / kind
    for field, value in (("mode", "response"), ("layers", [0, 1]), ("boundaries", [0, 1, 1]), ("kind", "vector_all")):
        bad = _native_residual_report(expected)
        bad["slots"]["2"] = {**bad["slots"]["2"], field: value}
        assert any(f"residual slot {field} =" in p for p in sa.residual_slot_problems([bad], expected)), field
    # the ledger at readiness
    bad = _native_residual_report(expected, ledger_size=3)
    assert any("ledger_size = 3" in p for p in sa.residual_slot_problems([bad], expected))
    # an UNARMED expectation: a worker that loaded a slot, or armed itself, refuses
    unarmed = {**expected, "armed": False, "n_layers_model": None, "hidden_size": None,
               "adapters": {n: {**i, "residual_fingerprint": None, "kind": None, "op": None,
                                "position_mode": None, "layers": None, "boundaries": None}
                            for n, i in expected["adapters"].items()}}
    ok = _native_residual_report(unarmed)
    assert sa.residual_slot_problems([ok], unarmed) == []
    assert any("UNARMED worker" in p for p in sa.residual_slot_problems(
        [{**ok, "slots": {"2": {"fingerprint": fp}}}], unarmed))
    assert any("armed = True, expected False" in p for p in sa.residual_slot_problems([good], unarmed))
